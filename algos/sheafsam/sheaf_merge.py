# algos/sheafsam/sheaf_merge.py

from __future__ import annotations

import math
from typing import Callable, Tuple

import torch


def _do_nothing(x: torch.Tensor, mode: str | None = None) -> torch.Tensor:
    return x


@torch.no_grad()
def build_sheaf_cell_merge(
    token_energy: torch.Tensor,
    H: int,
    W: int,
    ratio: float,
    cell_h: int = 2,
    cell_w: int = 2,
) -> Tuple[Callable, Callable, dict]:
    """
    StructSAM-like merge/unmerge controlled by sheaf energy.

    Low-energy cells are mergeable. Inside each selected cell:
      - dst = token with minimum sheaf energy
      - src = remaining tokens in that cell
      - src tokens are averaged into dst during merge
      - src positions receive dst value during unmerge
    """
    B, N = token_energy.shape
    assert N == H * W, (N, H, W)

    if ratio >= 1.0:
        return _do_nothing, _do_nothing, {"actual_ratio": 1.0, "merged_cells": 0}

    if H % cell_h != 0 or W % cell_w != 0:
        return _do_nothing, _do_nothing, {
            "actual_ratio": 1.0,
            "merged_cells": 0,
            "reason": "grid_not_divisible",
        }

    area = cell_h * cell_w
    if area <= 1:
        return _do_nothing, _do_nothing, {"actual_ratio": 1.0, "merged_cells": 0}

    target_drop = int(round(N * (1.0 - ratio)))
    if target_drop <= 0:
        return _do_nothing, _do_nothing, {"actual_ratio": 1.0, "merged_cells": 0}

    n_cell_h = H // cell_h
    n_cell_w = W // cell_w
    n_cells = n_cell_h * n_cell_w

    # Each merged cell drops area-1 tokens.
    n_merge_cells = min(n_cells, math.ceil(target_drop / (area - 1)))
    actual_drop = n_merge_cells * (area - 1)
    keep_n = N - actual_drop

    device = token_energy.device

    idx_grid = torch.arange(N, device=device).view(H, W)
    cell_tokens = (
        idx_grid.view(n_cell_h, cell_h, n_cell_w, cell_w)
        .permute(0, 2, 1, 3)
        .reshape(n_cells, area)
    )  # (n_cells, area)

    all_cell_tokens = cell_tokens.unsqueeze(0).expand(B, -1, -1)
    cell_energy = token_energy.gather(
        1,
        all_cell_tokens.reshape(B, -1),
    ).view(B, n_cells, area)

    # Flatness: protect cells with any high-energy token.
    # Choose lowest max-energy cells to merge.
    cell_flatness = cell_energy.max(dim=-1).values
    chosen_cell_ids = cell_flatness.topk(
        n_merge_cells,
        dim=1,
        largest=False,
    ).indices  # (B, n_merge_cells)

    chosen_tokens = cell_tokens[chosen_cell_ids]  # (B, n_merge_cells, area)
    chosen_energy = token_energy.gather(
        1,
        chosen_tokens.reshape(B, -1),
    ).view(B, n_merge_cells, area)

    dst_local = chosen_energy.argmin(dim=-1)  # (B, n_merge_cells)
    dst_abs = chosen_tokens.gather(
        -1,
        dst_local.unsqueeze(-1),
    ).squeeze(-1)  # (B, n_merge_cells)

    src_mask = torch.ones(
        (B, n_merge_cells, area),
        device=device,
        dtype=torch.bool,
    )
    src_mask.scatter_(-1, dst_local.unsqueeze(-1), False)

    src_abs = chosen_tokens[src_mask].view(B, n_merge_cells, area - 1)
    src_flat = src_abs.reshape(B, actual_drop)

    dst_for_src = (
        dst_abs.unsqueeze(-1)
        .expand(-1, -1, area - 1)
        .reshape(B, actual_drop)
    )

    keep_mask = torch.ones((B, N), device=device, dtype=torch.bool)
    keep_mask.scatter_(1, src_flat, False)

    rank = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
    sort_key = (~keep_mask).long() * N + rank
    keep_idx = sort_key.argsort(dim=1)[:, :keep_n]  # kept tokens in raster order

    inv_pos = torch.full((B, N), -1, device=device, dtype=torch.long)
    pos = torch.arange(keep_n, device=device).unsqueeze(0).expand(B, keep_n)
    inv_pos.scatter_(1, keep_idx, pos)

    dst_pos_for_src = inv_pos.gather(1, dst_for_src)

    def merge(x: torch.Tensor, mode: str | None = "mean") -> torch.Tensor:
        Bx, Nx, C = x.shape
        assert Bx == B and Nx == N, (x.shape, B, N)

        x_keep = x.gather(
            1,
            keep_idx.unsqueeze(-1).expand(B, keep_n, C),
        )

        if mode is not None and actual_drop > 0:
            src_val = x.gather(
                1,
                src_flat.unsqueeze(-1).expand(B, actual_drop, C),
            )
            x_keep = x_keep.scatter_reduce(
                dim=1,
                index=dst_pos_for_src.unsqueeze(-1).expand(B, actual_drop, C),
                src=src_val,
                reduce=mode,
                include_self=True,
            )

        return x_keep

    def unmerge(x_keep: torch.Tensor) -> torch.Tensor:
        Bx, Kx, C = x_keep.shape
        assert Bx == B and Kx == keep_n, (x_keep.shape, B, keep_n)

        out = torch.zeros((B, N, C), device=x_keep.device, dtype=x_keep.dtype)
        out.scatter_(
            1,
            keep_idx.unsqueeze(-1).expand(B, keep_n, C),
            x_keep,
        )

        # Duplicate dst values back to source positions.
        dst_values = out.gather(
            1,
            dst_for_src.unsqueeze(-1).expand(B, actual_drop, C),
        )
        out.scatter_(
            1,
            src_flat.unsqueeze(-1).expand(B, actual_drop, C),
            dst_values,
        )

        return out

    stats = {
        "actual_ratio": keep_n / float(N),
        "target_ratio": float(ratio),
        "merged_cells": int(n_merge_cells),
        "actual_drop": int(actual_drop),
        "cell_h": int(cell_h),
        "cell_w": int(cell_w),
    }

    return merge, unmerge, stats
