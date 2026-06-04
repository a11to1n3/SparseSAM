"""SheafScore: reusable feature-coboundary scoring backend for SheafSAM.

Provides a unified interface for computing sheaf-coboundary token/group
energies across the SAM family (SAM-HQ, SAM2, SAM3).

Usage:
    scorer = SheafScore(project_dim=16, normalize=True)
    token_energy = scorer.spatial_cobms(keys, H, W, graph="ms4")
    group_score  = scorer.group_score(token_energy, base_perm, group_size=4)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


# ── fixed projection (training-free) ───────────────────────────────────────

def chunk_project(x: torch.Tensor, project_dim: int = 16) -> torch.Tensor:
    """Training-free channel-group projection."""
    x = x.float()
    if project_dim is None or project_dim <= 0:
        return x
    c = x.shape[-1]
    d = min(int(project_dim), c)
    if c <= d:
        return x
    trim = (c // d) * d
    if trim <= 0:
        return x
    return x[..., :trim].reshape(*x.shape[:-1], d, trim // d).mean(dim=-1)


# ── graph coboundary helpers ───────────────────────────────────────────────

def _coboundary_offsets(kind: str):
    if kind == "4":
        return [(1, 0, 1.0), (0, 1, 1.0)]
    if kind == "8":
        return [(1, 0, 1.0), (0, 1, 1.0),
                (1, 1, 1.0 / math.sqrt(2.0)), (1, -1, 1.0 / math.sqrt(2.0))]
    if kind == "ms4":
        return [(1, 0, 1.0), (0, 1, 1.0), (2, 0, 0.5), (0, 2, 0.5)]
    if kind == "ms8":
        return [(1, 0, 1.0), (0, 1, 1.0),
                (1, 1, 1.0 / math.sqrt(2.0)), (1, -1, 1.0 / math.sqrt(2.0)),
                (2, 0, 0.5), (0, 2, 0.5),
                (2, 2, 0.5 / math.sqrt(2.0)), (2, -2, 0.5 / math.sqrt(2.0))]
    raise ValueError(f"unknown graph kind={kind!r}")


# ── z-score ────────────────────────────────────────────────────────────────

def zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Batch-wise z-score over last dimension (population std)."""
    return (x - x.mean(dim=-1, keepdim=True)) / (
        x.std(dim=-1, keepdim=True, correction=0) + eps
    )


# ═══════════════════════════════════════════════════════════════════════════
# SheafScore — unified scoring backend
# ═══════════════════════════════════════════════════════════════════════════

class SheafScore:
    """Unified feature-coboundary scoring for SAM-family models."""

    def __init__(
        self,
        project_dim: int = 16,
        normalize: bool = True,
        edge_reduce: str = "mean",
        degree_normalize: bool = True,
    ):
        self.project_dim = project_dim
        self.normalize = normalize
        self.edge_reduce = edge_reduce
        self.degree_normalize = degree_normalize

    # ── spatial scores (SAM-HQ / image encoder) ─────────────────────────

    def spatial_coboundary(
        self,
        keys: torch.Tensor,
        H: int,
        W: int,
        graph: str = "ms4",
        score_kind: str = "edge_energy",
    ) -> torch.Tensor:
        """True graph-coboundary energy on image token grid.

        Args:
            keys: (B, N, C) attention key tensor.
            H, W: image token grid dimensions.
            graph: "4", "8", "ms4", "ms8".
            score_kind: "edge_energy" or "laplacian_residual".

        Returns:
            (B, H*W) per-token energy.
        """
        grid = keys.view(keys.shape[0], H, W, -1).float()
        z = chunk_project(grid, project_dim=self.project_dim)
        if self.normalize:
            z = F.normalize(z, dim=-1, eps=1e-6)

        B, Hp, Wp, C = z.shape
        offsets = _coboundary_offsets(graph)

        if score_kind == "edge_energy":
            score = torch.zeros((B, Hp, Wp), device=z.device, dtype=torch.float32)
            deg = torch.zeros((Hp, Wp), device=z.device, dtype=torch.float32)
            for dh, dw, wt in offsets:
                hs = slice(max(0, -dh), Hp - max(0, dh))
                hd = slice(max(0, dh), Hp - max(0, -dh))
                ws = slice(max(0, -dw), Wp - max(0, dw))
                wd = slice(max(0, dw), Wp - max(0, -dw))
                a, b = z[:, hs, ws, :], z[:, hd, wd, :]
                e = wt * ((a - b).pow(2).mean(dim=-1) if self.edge_reduce == "mean"
                         else (a - b).pow(2).sum(dim=-1))
                score[:, hs, ws] += e; score[:, hd, wd] += e
                deg[hs, ws] += wt; deg[hd, wd] += wt
            if self.degree_normalize:
                score = score / deg.clamp_min(1e-6).unsqueeze(0)
            return score.reshape(B, Hp * Wp)

        if score_kind == "laplacian_residual":
            lap = torch.zeros_like(z)
            for dh, dw, wt in offsets:
                hs = slice(max(0, -dh), Hp - max(0, dh))
                hd = slice(max(0, dh), Hp - max(0, -dh))
                ws = slice(max(0, -dw), Wp - max(0, dw))
                wd = slice(max(0, dw), Wp - max(0, -dw))
                diff = wt * (z[:, hs, ws, :] - z[:, hd, wd, :])
                lap[:, hs, ws, :] += diff; lap[:, hd, wd, :] -= diff
            score = lap.pow(2).mean(dim=-1) if self.edge_reduce == "mean" else lap.pow(2).sum(dim=-1)
            return score.reshape(B, Hp * Wp)

        raise ValueError(f"unknown score_kind={score_kind!r}")

    def spatial_sobel(
        self,
        keys: torch.Tensor,
        H: int,
        W: int,
        padding_mode: str = "replicate",
    ) -> torch.Tensor:
        """Sobel-weighted sheaf differential on image token grid.

        Args:
            keys: (B, N, C) attention key tensor.
            H, W: token grid size.
            padding_mode: "replicate" or "zero".

        Returns:
            (B, H*W) per-token energy.
        """
        grid = keys.view(keys.shape[0], H, W, -1).float()
        # Sobel operates on full K features by default (no projection)
        z = grid if self.project_dim <= 0 else chunk_project(grid, project_dim=self.project_dim)
        if self.normalize:
            z = F.normalize(z, dim=-1, eps=1e-6)

        B_, Hp, Wp, C = z.shape
        feat = z.permute(0, 3, 1, 2).contiguous()

        sx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]],
                          device=feat.device, dtype=feat.dtype) / 8.0
        sy = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]],
                          device=feat.device, dtype=feat.dtype) / 8.0
        wx = sx.view(1,1,3,3).expand(C,1,3,3).contiguous()
        wy = sy.view(1,1,3,3).expand(C,1,3,3).contiguous()

        if padding_mode is not None:
            fp = F.pad(feat, (1,1,1,1), mode=padding_mode)
            gx = F.conv2d(fp, wx, groups=C)
            gy = F.conv2d(fp, wy, groups=C)
        else:
            gx = F.conv2d(feat, wx, padding=1, groups=C)
            gy = F.conv2d(feat, wy, padding=1, groups=C)

        e = gx.pow(2) + gy.pow(2)
        score = e.mean(dim=1) if self.edge_reduce == "mean" else e.sum(dim=1)
        return score.reshape(B_, Hp * Wp)

    def spatial_sheaf(
        self,
        keys: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Original 4-neighbor projected sheaf coboundary."""
        return self.spatial_coboundary(keys, H, W, graph="4", score_kind="edge_energy")

    # ── group scoring ──────────────────────────────────────────────────

    @staticmethod
    def group_score(
        token_energy: torch.Tensor,
        base_perm: torch.Tensor,
        group_size: int,
        reduce: str = "mean",
    ) -> torch.Tensor:
        """Convert token energy to group energy via space-filling perm."""
        B, N = token_energy.shape
        assert N % group_size == 0, (N, group_size)
        n_groups = N // group_size
        grouped = token_energy[:, base_perm].view(B, n_groups, group_size)
        if reduce == "max":
            return grouped.max(dim=-1).values
        if reduce == "mean":
            return grouped.mean(dim=-1)
        raise ValueError(f"unknown reduce={reduce!r}")

    # ── SAM2 / SAM3 placeholder stubs ──────────────────────────────────

    def temporal_coboundary(
        self,
        curr_keys: torch.Tensor,
        prev_keys: torch.Tensor,
        H: int, W: int,
    ) -> torch.Tensor:
        """Temporal inconsistency: ||P k_i^t - P k_i^{t-1}||^2."""
        z_curr = chunk_project(curr_keys, project_dim=self.project_dim)
        z_prev = chunk_project(prev_keys, project_dim=self.project_dim)
        if self.normalize:
            z_curr = F.normalize(z_curr, dim=-1, eps=1e-6)
            z_prev = F.normalize(z_prev, dim=-1, eps=1e-6)
        return (z_curr - z_prev).pow(2).mean(dim=-1)

    def memory_read_score(
        self,
        curr_q: torch.Tensor,
        mem_k: torch.Tensor,
        obj_ptr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Relevance + novelty for memory bank K/V ranking.

        Args:
            curr_q: (1, N_q, C) current query.
            mem_k:  (1, N_mem, C) memory keys.
            obj_ptr: (1, C) optional object pointer.

        Returns:
            (N_mem,) relevance score per memory token.
        """
        z_q = chunk_project(curr_q, project_dim=self.project_dim)
        z_m = chunk_project(mem_k, project_dim=self.project_dim)
        if self.normalize:
            z_q = F.normalize(z_q, dim=-1, eps=1e-6)
            z_m = F.normalize(z_m, dim=-1, eps=1e-6)
        rel = (z_q.squeeze(0) @ z_m.squeeze(0).mT).max(dim=1).values
        if obj_ptr is not None:
            z_o = chunk_project(obj_ptr, project_dim=self.project_dim)
            if self.normalize:
                z_o = F.normalize(z_o, dim=-1, eps=1e-6)
            obj_rel = (z_o.squeeze(0) @ z_m.squeeze(0).mT)
            rel = rel + obj_rel
        return rel

    def memory_write_score(
        self,
        curr_k: torch.Tensor,
        prev_k: Optional[torch.Tensor] = None,
        H: int = 0, W: int = 0,
    ) -> torch.Tensor:
        """Which current tokens to store: spatial energy + novelty."""
        score = self.spatial_coboundary(curr_k, H, W, graph="4")
        if prev_k is not None:
            score = score + self.temporal_coboundary(curr_k, prev_k, H, W)
        return score

    def concept_score(
        self,
        image_k: torch.Tensor,
        concept_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Concept-image similarity for detection routing."""
        z_i = chunk_project(image_k, project_dim=self.project_dim)
        z_c = chunk_project(concept_emb, project_dim=self.project_dim)
        if self.normalize:
            z_i = F.normalize(z_i, dim=-1, eps=1e-6)
            z_c = F.normalize(z_c, dim=-1, eps=1e-6)
        return (z_i @ z_c.T).squeeze(-1)

    def object_score(
        self,
        image_k: torch.Tensor,
        obj_ptr: torch.Tensor,
    ) -> torch.Tensor:
        """Object-consistency score: ||P k_i - P o||^2."""
        z_i = chunk_project(image_k, project_dim=self.project_dim)
        z_o = chunk_project(obj_ptr, project_dim=self.project_dim)
        if self.normalize:
            z_i = F.normalize(z_i, dim=-1, eps=1e-6)
            z_o = F.normalize(z_o, dim=-1, eps=1e-6)
        return (z_i - z_o.unsqueeze(1)).pow(2).mean(dim=-1)


__all__ = [
    "SheafScore",
    "chunk_project",
    "zscore",
]
