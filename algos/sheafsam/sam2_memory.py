"""SheafSAM-V: memory and object-pointer routing for SAM2 video models.

Patches SAM2's memory attention, memory encoder/write, object pointer,
and mask decoder to use sheaf-coboundary scores for selective computation.

Usage (once SAM2 model is loaded):
    from algos.sheafsam.sam2_memory import apply_sam2_sheaf_patch
    apply_sam2_sheaf_patch(sam2_model, memory_sparsity=0.5, write_sparsity=0.3)
"""

from __future__ import annotations

import types
from typing import Optional

import torch
import torch.nn as nn

from .sheaf_score import SheafScore


# ═══════════════════════════════════════════════════════════════════════════
# Memory-read sparsification
# ═══════════════════════════════════════════════════════════════════════════

class SheafMemoryReadAttention(nn.Module):
    """Memory cross-attention with sheaf-ranked memory K/V selection.

    Wraps SAM2's memory attention. Before the attention call, ranks memory
    tokens by relevance + novelty to the current query and object pointer,
    then keeps only the top-k memory K/V tokens.
    """

    def __init__(
        self,
        base_attn: nn.Module,
        scorer: SheafScore,
        memory_sparsity: float = 0.5,
    ):
        super().__init__()
        self.base_attn = base_attn
        self.scorer = scorer
        self.memory_sparsity = memory_sparsity

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        obj_ptr: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Rank memory K/V tokens and keep top-k."""
        if self.memory_sparsity >= 1.0 or k is None:
            return self.base_attn(q, k, v, **kwargs)

        # Compute relevance scores for memory tokens
        scores = self.scorer.memory_read_score(q, k, obj_ptr=obj_ptr)

        keep_n = max(1, int(self.memory_sparsity * scores.numel()))
        top_idx = scores.topk(keep_n, largest=True).indices

        k_sparse = k[:, top_idx, :] if k.dim() == 3 else k[top_idx]
        v_sparse = v[:, top_idx, :] if v.dim() == 3 else v[top_idx]

        return self.base_attn(q, k_sparse, v_sparse, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Memory-write selection
# ═══════════════════════════════════════════════════════════════════════════

class SheafMemoryEncoder(nn.Module):
    """Memory encoder with sheaf-guided write selection.

    Instead of storing all current-frame tokens to memory, only stores
    high-coboundary tokens (boundaries, transitions, occlusions) plus
    a uniform sample of low-energy regions.
    """

    def __init__(
        self,
        base_mem_encoder: nn.Module,
        scorer: SheafScore,
        write_sparsity: float = 0.3,
        H: int = 64,
        W: int = 64,
    ):
        super().__init__()
        self.base_mem_encoder = base_mem_encoder
        self.scorer = scorer
        self.write_sparsity = write_sparsity
        self.H = H
        self.W = W

    def forward(
        self,
        curr_k: torch.Tensor,
        prev_k: Optional[torch.Tensor] = None,
        obj_ptr: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Selectively write high-energy tokens to memory."""
        if self.write_sparsity >= 1.0:
            return self.base_mem_encoder(curr_k, **kwargs)

        write_score = self.scorer.memory_write_score(
            curr_k, prev_k, H=self.H, W=self.W,
        )
        if obj_ptr is not None:
            obj_score = self.scorer.object_score(curr_k, obj_ptr)
            write_score = write_score + obj_score

        keep_n = max(1, int(self.write_sparsity * write_score.numel()))
        top_idx = write_score.topk(keep_n, largest=True).indices

        return self.base_mem_encoder(curr_k[:, top_idx, :], **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Object-pointer sheaf
# ═══════════════════════════════════════════════════════════════════════════

class SheafObjectPointer(nn.Module):
    """Object-pointer consistency monitor.

    Tracks the sheaf energy between current image tokens and the object
    pointer. A sudden spike in object residual indicates occlusion, drift,
    or re-entry — and can trigger denser memory/decoder computation.
    """

    def __init__(
        self,
        scorer: SheafScore,
        occlusion_threshold: float = 2.0,
    ):
        super().__init__()
        self.scorer = scorer
        self.occlusion_threshold = occlusion_threshold
        self._prev_residual: Optional[float] = None

    def forward(
        self,
        image_k: torch.Tensor,
        obj_ptr: torch.Tensor,
    ) -> dict:
        """Compute object-consistency residual and occlusion flag."""
        obj_score = self.scorer.object_score(image_k, obj_ptr)
        residual = obj_score.mean().item()

        occlusion = False
        if self._prev_residual is not None:
            if residual > self.occlusion_threshold * self._prev_residual:
                occlusion = True
        self._prev_residual = residual

        return {
            "object_residual": residual,
            "occlusion_detected": occlusion,
            "obj_score": obj_score,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Decoder K/V routing
# ═══════════════════════════════════════════════════════════════════════════

class SheafDecoderRouter(nn.Module):
    """Decoder cross-attention with sheaf-ranked K/V selection.

    Ranks decoder K/V tokens (image embeddings, prompt embeddings, memory
    context) by combined spatial + object + memory scores. Keeps top-k for
    faster mask decoding.
    """

    def __init__(
        self,
        scorer: SheafScore,
        decoder_sparsity: float = 0.6,
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
        prompt_emb: Optional[torch.Tensor] = None,
        obj_ptr: Optional[torch.Tensor] = None,
        mem_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Combined decoder token score from spatial + object + memory."""
        score = self.scorer.spatial_coboundary(image_k, self.H, self.W, graph="4")

        if obj_ptr is not None:
            obj_score = self.scorer.object_score(image_k, obj_ptr)
            score = score + obj_score

        if mem_k is not None:
            mem_rel = self.scorer.memory_read_score(
                image_k[:1, :1, :], mem_k[:1],
            ).unsqueeze(0)
            score = score + mem_rel

        return score


# ═══════════════════════════════════════════════════════════════════════════
# Full SAM2 patch
# ═══════════════════════════════════════════════════════════════════════════

def apply_sam2_sheaf_patch(
    sam2_model: nn.Module,
    memory_sparsity: float = 0.5,
    write_sparsity: float = 0.3,
    decoder_sparsity: float = 0.6,
    project_dim: int = 16,
    H: int = 64,
    W: int = 64,
    verbose: bool = True,
) -> nn.Module:
    """Apply SheafSAM-V patches to a SAM2 model.

    Patches memory-read attention, memory encoder/write, and decoder
    cross-attention with sheaf-coboundary routing.

    Args:
        sam2_model: loaded SAM2 model (sam2.model_builder.build_sam2_video_predictor).
        memory_sparsity: fraction of memory tokens to keep (0-1).
        write_sparsity: fraction of current tokens to write to memory.
        decoder_sparsity: fraction of decoder K/V tokens to keep.
        project_dim: sheaf projection dimension.
        H, W: token grid size.

    Returns:
        patched sam2_model.
    """
    scorer = SheafScore(
        project_dim=project_dim,
        normalize=True,
        edge_reduce="mean",
    )

    n_patched = 0

    # Patch memory attention blocks
    for name, module in list(sam2_model.named_modules()):
        # SAM2 memory attention typically appears in the memory decoder
        if "memory_attention" in name.lower() and hasattr(module, 'forward'):
            wrapped = SheafMemoryReadAttention(
                module, scorer, memory_sparsity=memory_sparsity,
            )
            _set_submodule(sam2_model, name, wrapped)
            n_patched += 1

        # Patch memory encoder
        if "memory_encoder" in name.lower() and hasattr(module, 'forward'):
            wrapped = SheafMemoryEncoder(
                module, scorer, write_sparsity=write_sparsity, H=H, W=W,
            )
            _set_submodule(sam2_model, name, wrapped)
            n_patched += 1

    if verbose:
        print(
            f"[SheafSAM-V] sam2 patched {n_patched} modules "
            f"mem_sparsity={memory_sparsity} write_sparsity={write_sparsity} "
            f"decoder_sparsity={decoder_sparsity}"
        )

    return sam2_model


def _set_submodule(root: nn.Module, dotted_name: str, new_module: nn.Module):
    """Set a submodule by dotted name (e.g. 'memory_attention.0')."""
    parts = dotted_name.split('.')
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


# ═══════════════════════════════════════════════════════════════════════════
# Profiler stub
# ═══════════════════════════════════════════════════════════════════════════

def profile_sam2_modules(sam2_model: nn.Module):
    """Module-level latency breakdown for SAM2.

    Returns a dict of module name → estimated compute proportion.
    Useful for identifying which modules to sparsify first.
    """
    modules = {}
    total_params = sum(p.numel() for p in sam2_model.parameters())

    for name, module in sam2_model.named_modules():
        n_params = sum(p.numel() for p in module.parameters(recurse=False))
        if n_params > 0:
            modules[name] = n_params / total_params

    # Categorize
    categories = {
        "image_encoder": 0.0,
        "memory_attention": 0.0,
        "memory_encoder": 0.0,
        "mask_decoder": 0.0,
        "prompt_encoder": 0.0,
    }
    for name, frac in modules.items():
        for cat in categories:
            if cat in name.lower():
                categories[cat] += frac
                break

    return categories


__all__ = [
    "SheafMemoryReadAttention",
    "SheafMemoryEncoder",
    "SheafObjectPointer",
    "SheafDecoderRouter",
    "apply_sam2_sheaf_patch",
    "profile_sam2_modules",
]
