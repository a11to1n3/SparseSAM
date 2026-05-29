# algos/sheafsam/sheaf_utils.py

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def zscore_like(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Batch-wise z-score over the last dimension. Uses population std (correction=0)."""
    return (x - x.mean(dim=-1, keepdim=True)) / (
        x.std(dim=-1, keepdim=True, correction=0) + eps
    )


def chunk_project(x: torch.Tensor, project_dim: int = 16) -> torch.Tensor:
    """
    Training-free projection from C channels to <= project_dim channels.

    This avoids adding learned weights. For C >> project_dim, channels are
    grouped and averaged. For small C, returns x.float().
    """
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


@torch.no_grad()
def identity_sheaf_token_energy(
    x: torch.Tensor,
    H: int,
    W: int,
    project_dim: int = 16,
    normalize: bool = True,
    edge_reduce: str = "mean",
) -> torch.Tensor:
    """
    Identity-sheaf coboundary energy on a 2D token grid.

    Args:
        x:
            Either (B, N, C) with N=H*W or (B, H, W, C).
        H, W:
            Spatial token grid.
        project_dim:
            Cheap deterministic projection dimension.
        normalize:
            L2-normalize projected token features before neighbor differences.
        edge_reduce:
            "mean" or "sum" over projected feature channels.

    Returns:
        token_energy: (B, H*W), higher means less locally glueable.
    """
    if x.dim() == 4:
        B, Hx, Wx, C = x.shape
        assert Hx == H and Wx == W, (x.shape, H, W)
        grid = x
    elif x.dim() == 3:
        B, N, C = x.shape
        assert N == H * W, (N, H, W)
        grid = x.view(B, H, W, C)
    else:
        raise ValueError(f"expected x dim 3 or 4, got {x.shape}")

    z = chunk_project(grid, project_dim=project_dim)

    if normalize:
        z = F.normalize(z, dim=-1, eps=1e-6)

    score = torch.zeros(
        (z.shape[0], H, W),
        device=z.device,
        dtype=torch.float32,
    )

    def reduce_edge(e: torch.Tensor) -> torch.Tensor:
        if edge_reduce == "sum":
            return e.sum(dim=-1)
        if edge_reduce == "mean":
            return e.mean(dim=-1)
        raise ValueError(f"unknown edge_reduce={edge_reduce!r}")

    # Horizontal edges.
    if W > 1:
        diff_w = (z[:, :, 1:, :] - z[:, :, :-1, :]).pow(2)
        e_w = reduce_edge(diff_w)
        score[:, :, 1:] += e_w
        score[:, :, :-1] += e_w

    # Vertical edges.
    if H > 1:
        diff_h = (z[:, 1:, :, :] - z[:, :-1, :, :]).pow(2)
        e_h = reduce_edge(diff_h)
        score[:, 1:, :] += e_h
        score[:, :-1, :] += e_h

    return score.reshape(z.shape[0], H * W)


@torch.no_grad()
def group_energy_from_token_energy(
    token_energy: torch.Tensor,
    base_perm: torch.Tensor,
    group_size: int,
    reduce: str = "mean",
) -> torch.Tensor:
    """
    Convert token energy (B, N) into group energy (B, N/group_size)
    following a Z/Hilbert base permutation.
    """
    B, N = token_energy.shape
    assert N % group_size == 0, (N, group_size)

    n_groups = N // group_size
    grouped = token_energy[:, base_perm].view(B, n_groups, group_size)

    if reduce == "max":
        return grouped.max(dim=-1).values
    if reduce == "mean":
        return grouped.mean(dim=-1)

    raise ValueError(f"unknown reduce={reduce!r}")


@torch.no_grad()
def sobel_sheaf_token_energy(
    x: torch.Tensor,
    H: int,
    W: int,
    project_dim: int = 0,
    normalize: bool = True,
    reduce: str = "mean",
    padding_mode: str = "replicate",
) -> torch.Tensor:
    """Sobel-weighted sheaf differential energy on a 2D token grid.
    Treats Sobel-K as a sheaf coboundary: D_x k = Sobel_x(P k), D_y = Sobel_y(P k).
    Uses depthwise convolution for efficiency."""
    if x.dim() == 3:
        B, N, C = x.shape
        assert N == H * W, (N, H, W)
        grid = x.view(B, H, W, C).float()
    elif x.dim() == 4:
        B_, H_, W_, C = x.shape
        assert H_ == H and W_ == W, (x.shape, H, W)
        grid = x.float()
    else:
        raise ValueError(f"expected x dim 3 or 4, got {x.shape}")

    if project_dim is not None and project_dim > 0:
        grid = chunk_project(grid, project_dim=project_dim)

    if normalize:
        grid = torch.nn.functional.normalize(grid, dim=-1, eps=1e-6)

    B, H, W, C = grid.shape
    feat = grid.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=feat.device, dtype=feat.dtype,
    ) / 8.0
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=feat.device, dtype=feat.dtype,
    ) / 8.0

    wx = sobel_x.view(1, 1, 3, 3).expand(C, 1, 3, 3).contiguous()
    wy = sobel_y.view(1, 1, 3, 3).expand(C, 1, 3, 3).contiguous()

    if padding_mode is not None:
        feat_pad = torch.nn.functional.pad(feat, (1, 1, 1, 1), mode=padding_mode)
        gx = torch.nn.functional.conv2d(feat_pad, wx, groups=C)
        gy = torch.nn.functional.conv2d(feat_pad, wy, groups=C)
    else:
        gx = torch.nn.functional.conv2d(feat, wx, padding=1, groups=C)
        gy = torch.nn.functional.conv2d(feat, wy, padding=1, groups=C)

    e = gx.pow(2) + gy.pow(2)
    score = e.mean(dim=1) if reduce == "mean" else e.sum(dim=1)
    return score.reshape(B, H * W)


@torch.no_grad()
def group_score_from_token_energy(
    token_energy: torch.Tensor,
    base_perm: torch.Tensor,
    group_size: int,
    reduce: str = "mean",
) -> torch.Tensor:
    """Convert token energy (B, N) to group energy (B, n_groups)
    using a space-filling base permutation."""
    B, N = token_energy.shape
    assert N % group_size == 0, (N, group_size)
    n_groups = N // group_size
    grouped = token_energy[:, base_perm].view(B, n_groups, group_size)
    if reduce == "max":
        return grouped.max(dim=-1).values
    if reduce == "mean":
        return grouped.mean(dim=-1)
    raise ValueError(f"unknown group reduce={reduce!r}")


@torch.no_grad()
def sobel_k_token_energy(
    x: torch.Tensor,
    H: int,
    W: int,
    normalize: bool = True,
    edge_reduce: str = "mean",
) -> torch.Tensor:
    """Backwards-compatible alias for sobel_sheaf_token_energy."""
    return sobel_sheaf_token_energy(
        x, H=H, W=W, project_dim=0,
        normalize=normalize, reduce=edge_reduce, padding_mode="replicate",
    )


# ── True graph-coboundary helpers ──────────────────────────────────────────

def _as_grid(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    if x.dim() == 4:
        B, Hx, Wx, C = x.shape
        assert Hx == H and Wx == W, (x.shape, H, W)
        return x.float()
    if x.dim() == 3:
        B, N, C = x.shape
        assert N == H * W, (N, H, W)
        return x.view(B, H, W, C).float()
    raise ValueError(f"expected x dim 3 or 4, got {x.shape}")


def _coboundary_offsets(kind: str):
    """True graph-coboundary edge sets. Pairwise edges only, no convolution stencils."""
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
    raise ValueError(f"unknown coboundary graph kind={kind!r}")


@torch.no_grad()
def graph_coboundary_token_energy(
    x: torch.Tensor,
    H: int,
    W: int,
    project_dim: int = 16,
    graph: str = "4",
    normalize: bool = True,
    edge_reduce: str = "mean",
    degree_normalize: bool = True,
    score_kind: str = "edge_energy",
) -> torch.Tensor:
    """True pairwise sheaf-coboundary token energy.

    Strict sheaf version: edge e=(i,j): delta_F x_e = sqrt(w_e) * (P x_i - P x_j).
    No Sobel, no 3x3 convolution. Pure graph incidence coboundary only.

    score_kind="edge_energy"  → token energy = degree-normalized incident edge energy.
    score_kind="laplacian_residual" → ||L_F z||^2 locally.
    """
    grid = _as_grid(x, H, W)
    z = chunk_project(grid, project_dim=project_dim)
    if normalize:
        z = torch.nn.functional.normalize(z, dim=-1, eps=1e-6)

    B, H, W, C = z.shape
    offsets = _coboundary_offsets(graph)

    if score_kind == "edge_energy":
        score = torch.zeros((B, H, W), device=z.device, dtype=torch.float32)
        degree = torch.zeros((H, W), device=z.device, dtype=torch.float32)
        for dh, dw, wt in offsets:
            hs = slice(max(0, -dh), H - max(0, dh))
            hd = slice(max(0, dh), H - max(0, -dh))
            ws = slice(max(0, -dw), W - max(0, dw))
            wd = slice(max(0, dw), W - max(0, -dw))
            a, b = z[:, hs, ws, :], z[:, hd, wd, :]
            e = wt * ((a - b).pow(2).mean(dim=-1) if edge_reduce == "mean" else (a - b).pow(2).sum(dim=-1))
            score[:, hs, ws] += e
            score[:, hd, wd] += e
            degree[hs, ws] += wt
            degree[hd, wd] += wt
        if degree_normalize:
            score = score / degree.clamp_min(1e-6).unsqueeze(0)
        return score.reshape(B, H * W)

    if score_kind == "laplacian_residual":
        lap = torch.zeros_like(z)
        for dh, dw, wt in offsets:
            hs = slice(max(0, -dh), H - max(0, dh))
            hd = slice(max(0, dh), H - max(0, -dh))
            ws = slice(max(0, -dw), W - max(0, dw))
            wd = slice(max(0, dw), W - max(0, -dw))
            diff = wt * (z[:, hs, ws, :] - z[:, hd, wd, :])
            lap[:, hs, ws, :] += diff
            lap[:, hd, wd, :] -= diff
        score = lap.pow(2).mean(dim=-1) if edge_reduce == "mean" else lap.pow(2).sum(dim=-1)
        return score.reshape(B, H * W)

    raise ValueError(f"unknown score_kind={score_kind!r}")
