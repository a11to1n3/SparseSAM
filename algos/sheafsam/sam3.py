"""SheafSAM monkey patch for SAM3 ViT encoder.

Extends sparsesam/sam3.py. The only change: MLP keep-set is selected
by sheaf token energy instead of uniform Z-order stride sampling.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Optional

import torch
import torch.nn as nn

from .sheaf_utils import identity_sheaf_token_energy

_here = os.path.dirname(__file__)
_sam3_root = os.path.normpath(os.path.join(_here, "..", "3rd_party", "sam3"))
if _sam3_root not in sys.path:
    sys.path.insert(0, _sam3_root)

from sam3.model.vitdet import (
    Attention,
    Block,
    window_partition,
    window_unpartition,
)


class SheafSAM3Block(Block):
    """Block forward with sheaf-energy-guided MLP keep-set selection."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        info = self._tome_info
        ratio = float(info["ratio"])
        mlp_merge = bool(info.get("mlp_merge", True))

        # ── attention (unchanged from stock Block.forward) ────────────────
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H_w, W_w = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.ls1(self.attn(x))
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H_w, W_w))
        x = shortcut + self.dropout(self.drop_path(x))

        # ── MLP (sheaf-energy keep-set) ───────────────────────────────────
        do_partial = mlp_merge and ratio < 1.0 and x.ndim == 4
        if do_partial:
            B, H, W, C = x.shape
            N = H * W
            x_seq = x.reshape(B, N, C)
            keep_n = max(1, round(ratio * N))

            # Cache key: same (H, W, ratio) → same Z-order indices.
            # Recompute sheaf energy only when cache misses (like SAM-HQ).
            cache_key = (H, W, keep_n)
            perm_cache = info.setdefault("perm_cache", {})

            if cache_key not in perm_cache:
                sheaf_project_dim = int(info.get("sheaf_project_dim", 16))
                token_energy = identity_sheaf_token_energy(
                    x_seq, H=H, W=W,
                    project_dim=sheaf_project_dim,
                    normalize=True,
                    edge_reduce="mean",
                )
                keep_idx = token_energy.topk(keep_n, dim=1, largest=True).indices
                perm_cache[cache_key] = keep_idx

            keep_idx = perm_cache[cache_key]
            idx_e = keep_idx.unsqueeze(-1).expand(B, -1, C)

            x_kept = x_seq.gather(1, idx_e)
            x_kept = x_kept + self.dropout(
                self.drop_path(self.ls2(self.mlp(self.norm2(x_kept))))
            )
            x_seq = x_seq.scatter(1, idx_e, x_kept)
            return x_seq.reshape(B, H, W, C)

        return x + self.dropout(self.drop_path(self.ls2(self.mlp(self.norm2(x)))))


def apply_patch(
    encoder: nn.Module,
    ratio: float = 0.9,
    mlp_merge: bool = True,
    sheaf_project_dim: int = 16,
    **_,
) -> nn.Module:
    """Patch every `Block` in a SAM3 ViT encoder with `SheafSAM3Block`."""
    assert 0.0 < ratio <= 1.0, f"ratio must be in (0, 1], got {ratio}"

    info = {
        "ratio": float(ratio),
        "mlp_merge": bool(mlp_merge),
        "sheaf_project_dim": int(sheaf_project_dim),
    }
    encoder.tome_info = info

    n_patched = 0
    for module in encoder.modules():
        if isinstance(module, Block) and not isinstance(module, SheafSAM3Block):
            module.__class__ = SheafSAM3Block
            module._tome_info = info
            n_patched += 1

    if n_patched == 0:
        raise RuntimeError(
            "apply_patch(sam3 sheaf): no sam3.model.vitdet.Block found."
        )

    n_global = n_windowed = 0
    for m in encoder.modules():
        if isinstance(m, SheafSAM3Block):
            if m.window_size == 0:
                n_global += 1
            else:
                n_windowed += 1
    print(
        f"[Sheaf-SAM3] patched  ratio={ratio}  mlp_merge={mlp_merge}  "
        f"blocks={n_patched} (global={n_global} windowed={n_windowed})"
    )
    return encoder


def remove_patch(encoder: nn.Module) -> int:
    n = 0
    for module in encoder.modules():
        if type(module) is SheafSAM3Block:
            module.__class__ = Block
            module.__dict__.pop("_tome_info", None)
            module.__dict__.pop("_forward_impl", None)
            n += 1
    encoder.__dict__.pop("tome_info", None)
    return n


__all__ = ["apply_patch", "remove_patch", "SheafSAM3Block"]
