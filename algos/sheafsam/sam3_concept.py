"""SheafSAM-3: concept and tracker routing for SAM3 models.

Patches SAM3's concept-conditioned detection, exemplar-image matching,
presence-token gating, tracker-memory attention, object multiplexing,
and mask decoder with sheaf-coboundary routing.

Usage (once SAM3 model is loaded):
    from algos.sheafsam.sam3_concept import apply_sam3_sheaf_patch
    apply_sam3_sheaf_patch(sam3_model, concept_sparsity=0.5)
"""

from __future__ import annotations

import types
from typing import Optional

import torch
import torch.nn as nn

from .sheaf_score import SheafScore


# ═══════════════════════════════════════════════════════════════════════════
# Concept-conditioned detector routing
# ═══════════════════════════════════════════════════════════════════════════

class SheafConceptDetector(nn.Module):
    """Concept-image detector with sheaf-ranked region routing.

    For each concept, ranks image token groups by concept similarity +
    spatial coboundary energy. Keeps top-k groups for detection attention.
    """

    def __init__(
        self,
        scorer: SheafScore,
        concept_sparsity: float = 0.5,
        H: int = 64,
        W: int = 64,
    ):
        super().__init__()
        self.scorer = scorer
        self.concept_sparsity = concept_sparsity
        self.H = H
        self.W = W

    def rank_image_for_concept(
        self,
        image_k: torch.Tensor,
        concept_emb: torch.Tensor,
        exemplar_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Rank image tokens by concept relevance + spatial structure.

        Returns:
            (B, N) combined score per image token.
        """
        spatial = self.scorer.spatial_coboundary(image_k, self.H, self.W, graph="4")
        concept_rel = self.scorer.concept_score(image_k, concept_emb)

        score = spatial + concept_rel.amax(dim=-1) if concept_rel.dim() >= 2 else spatial + concept_rel

        if exemplar_emb is not None:
            exemplar_rel = self.scorer.concept_score(image_k, exemplar_emb)
            score = score + exemplar_rel

        return score

    def select_top_tokens(
        self,
        image_k: torch.Tensor,
        concept_emb: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return top-k image tokens for concept detection."""
        scores = self.rank_image_for_concept(image_k, concept_emb, **kwargs)
        keep_n = max(1, int(self.concept_sparsity * scores.shape[1]))
        top_idx = scores.topk(keep_n, dim=1, largest=True).indices
        return top_idx, scores


# ═══════════════════════════════════════════════════════════════════════════
# Presence-token gating
# ═══════════════════════════════════════════════════════════════════════════

class SheafPresenceGate(nn.Module):
    """Presence-token sheaf gate for concept absence/ambiguity.

    Uses the global concept-image coboundary residual to decide:
    - skip: concept is likely absent
    - sparse: concept is possibly present but ambiguous
    - dense: concept is clearly present
    """

    def __init__(
        self,
        scorer: SheafScore,
        absent_threshold: float = 0.3,
        ambiguous_threshold: float = 0.7,
    ):
        super().__init__()
        self.scorer = scorer
        self.absent_threshold = absent_threshold
        self.ambiguous_threshold = ambiguous_threshold

    def forward(
        self,
        image_k: torch.Tensor,
        concept_emb: torch.Tensor,
    ) -> dict:
        """Decide concept detection density from presence residual.

        Returns:
            dict with 'mode': 'skip' | 'sparse' | 'dense' and 'confidence'.
        """
        concept_score = self.scorer.concept_score(image_k, concept_emb)
        max_sim = concept_score.max().item()

        if max_sim < self.absent_threshold:
            mode, confidence = "skip", max_sim
        elif max_sim < self.ambiguous_threshold:
            mode, confidence = "sparse", max_sim
        else:
            mode, confidence = "dense", max_sim

        return {"mode": mode, "confidence": confidence, "max_sim": max_sim}


# ═══════════════════════════════════════════════════════════════════════════
# Tracker-memory routing
# ═══════════════════════════════════════════════════════════════════════════

class SheafTrackerMemory(nn.Module):
    """SAM3 tracker with sheaf-ranked object-memory assignment.

    For each object track, ranks memory tokens by object affinity and
    novelty. Keeps top-k memory tokens per object to reduce tracker cost.
    """

    def __init__(
        self,
        scorer: SheafScore,
        tracker_sparsity: float = 0.4,
    ):
        super().__init__()
        self.scorer = scorer
        self.tracker_sparsity = tracker_sparsity

    def rank_memory_for_object(
        self,
        mem_k: torch.Tensor,
        obj_ptr: torch.Tensor,
        curr_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score memory tokens for a specific object track.

        Returns:
            (N_mem,) per-memory-token relevance score.
        """
        obj_affinity = self.scorer.object_score(mem_k, obj_ptr)
        if curr_k is not None:
            novel = self.scorer.temporal_coboundary(
                curr_k, mem_k, H=1, W=1,
            ) if mem_k.shape[1] == curr_k.shape[1] else torch.zeros_like(obj_affinity)
        else:
            novel = 0.0

        return obj_affinity + novel

    def select_memory_for_object(
        self,
        mem_k: torch.Tensor,
        obj_ptr: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return top-k memory tokens for a specific object."""
        scores = self.rank_memory_for_object(mem_k, obj_ptr, **kwargs)
        keep_n = max(1, int(self.tracker_sparsity * scores.numel()))
        top_idx = scores.topk(keep_n, largest=True).indices
        return top_idx, scores


# ═══════════════════════════════════════════════════════════════════════════
# Object multiplexing
# ═══════════════════════════════════════════════════════════════════════════

class SheafObjectMultiplexer(nn.Module):
    """Sparse object-memory assignment for multi-object tracking.

    Each memory token is assigned to the most relevant object, avoiding
    redundant computation across objects that share the same memory region.
    """

    def __init__(self, scorer: SheafScore):
        super().__init__()
        self.scorer = scorer

    def assign_memory_to_objects(
        self,
        mem_k: torch.Tensor,
        obj_ptrs: torch.Tensor,
    ) -> torch.Tensor:
        """Assign each memory token to the most relevant object.

        Args:
            mem_k: (N_mem, C) memory key features.
            obj_ptrs: (N_obj, C) object pointer features.

        Returns:
            (N_mem,) assignment: index of best object per memory token.
        """
        scores = torch.stack([
            self.scorer.object_score(mem_k.unsqueeze(0), o.unsqueeze(0)).squeeze(0)
            for o in obj_ptrs
        ], dim=0)

        return scores.argmin(dim=0)


# ═══════════════════════════════════════════════════════════════════════════
# SAM3 decoder routing
# ═══════════════════════════════════════════════════════════════════════════

class SheafSAM3DecoderRouter(nn.Module):
    """SAM3 decoder with concept/object/memory-conditioned K/V routing."""

    def __init__(
        self,
        scorer: SheafScore,
        decoder_sparsity: float = 0.5,
        H: int = 64,
        W: int = 64,
    ):
        super().__init__()
        self.scorer = scorer
        self.decoder_sparsity = decoder_sparsity
        self.H = H
        self.W = W

    def rank_decoder_tokens(
        self,
        image_k: torch.Tensor,
        concept_emb: Optional[torch.Tensor] = None,
        obj_ptr: Optional[torch.Tensor] = None,
        mem_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Combined decoder token score."""
        score = self.scorer.spatial_coboundary(image_k, self.H, self.W, graph="4")

        if concept_emb is not None:
            score = score + self.scorer.concept_score(image_k, concept_emb)

        if obj_ptr is not None:
            score = score + self.scorer.object_score(image_k, obj_ptr)

        if mem_k is not None and mem_k.shape[1] == image_k.shape[1]:
            score = score + self.scorer.temporal_coboundary(
                image_k, mem_k, H=self.H, W=self.W,
            )

        return score


# ═══════════════════════════════════════════════════════════════════════════
# Full SAM3 patch
# ═══════════════════════════════════════════════════════════════════════════

def apply_sam3_sheaf_patch(
    sam3_model: nn.Module,
    concept_sparsity: float = 0.5,
    tracker_sparsity: float = 0.4,
    decoder_sparsity: float = 0.5,
    project_dim: int = 16,
    H: int = 72,
    W: int = 72,
    verbose: bool = True,
) -> nn.Module:
    """Apply SheafSAM-3 patches to a SAM3 model.

    Patches concept-image detection, presence gating, tracker-memory,
    and decoder cross-attention with sheaf-coboundary routing.

    Args:
        sam3_model: loaded SAM3 model.
        concept_sparsity: fraction of image tokens for concept detection.
        tracker_sparsity: fraction of memory tokens per object track.
        decoder_sparsity: fraction of decoder K/V tokens.
        project_dim: sheaf projection dimension.
        H, W: SAM3 token grid size (72×72 for SAM3).

    Returns:
        patched sam3_model.
    """
    scorer = SheafScore(
        project_dim=project_dim,
        normalize=True,
        edge_reduce="mean",
    )

    # Attach helper modules to the model for external use
    sam3_model._sheaf_concept_detector = SheafConceptDetector(
        scorer, concept_sparsity=concept_sparsity, H=H, W=W,
    )
    sam3_model._sheaf_presence_gate = SheafPresenceGate(scorer)
    sam3_model._sheaf_tracker_memory = SheafTrackerMemory(
        scorer, tracker_sparsity=tracker_sparsity,
    )
    sam3_model._sheaf_object_mux = SheafObjectMultiplexer(scorer)
    sam3_model._sheaf_decoder_router = SheafSAM3DecoderRouter(
        scorer, decoder_sparsity=decoder_sparsity, H=H, W=W,
    )
    sam3_model._sheaf_scorer = scorer

    if verbose:
        print(
            f"[SheafSAM-3] sam3 patched "
            f"concept_sparsity={concept_sparsity} "
            f"tracker_sparsity={tracker_sparsity} "
            f"decoder_sparsity={decoder_sparsity}"
        )

    return sam3_model


def detect_with_sheaf(
    model: nn.Module,
    image_k: torch.Tensor,
    concept_emb: torch.Tensor,
    exemplar_emb: Optional[torch.Tensor] = None,
) -> tuple:
    """Concept-conditioned detection with sheaf routing.

    Uses the presence gate for density decision, then concept detector
    for region selection.

    Returns:
        (density_mode, top_indices, scores_dict)
    """
    gate = model._sheaf_presence_gate(image_k, concept_emb)
    detector = model._sheaf_concept_detector

    if gate["mode"] == "skip":
        return "skip", None, gate

    top_idx, scores = detector.select_top_tokens(
        image_k, concept_emb, exemplar_emb=exemplar_emb,
    )
    return gate["mode"], top_idx, {"gate": gate, "scores": scores}


def track_with_sheaf(
    model: nn.Module,
    mem_k: torch.Tensor,
    obj_ptrs: torch.Tensor,
    curr_k: Optional[torch.Tensor] = None,
) -> tuple:
    """Multi-object tracking with sheaf memory routing.

    Assigns memory tokens to objects and selects top-k per object.
    """
    tracker = model._sheaf_tracker_memory
    mux = model._sheaf_object_mux

    assignments = mux.assign_memory_to_objects(mem_k, obj_ptrs)

    per_object = {}
    for o_idx, obj_ptr in enumerate(obj_ptrs):
        obj_mask = assignments == o_idx
        if obj_mask.sum() == 0:
            per_object[o_idx] = (None, None)
            continue
        mem_for_obj = mem_k[obj_mask]
        top_idx, scores = tracker.select_memory_for_object(
            mem_for_obj.unsqueeze(0), obj_ptr.unsqueeze(0),
            curr_k=curr_k[:, obj_mask, :] if curr_k is not None else None,
        )
        per_object[o_idx] = (top_idx, scores)

    return assignments, per_object


__all__ = [
    "SheafConceptDetector",
    "SheafPresenceGate",
    "SheafTrackerMemory",
    "SheafObjectMultiplexer",
    "SheafSAM3DecoderRouter",
    "apply_sam3_sheaf_patch",
    "detect_with_sheaf",
    "track_with_sheaf",
]
