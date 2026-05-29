"""SheafSAM PE partial: sheaf-ranked sparse attention for PE-Core models.

Extends sparsesam/pe_partial.py. The only change is the permutation:
SparseSAM uses stride-sort; SheafSAM uses sheaf-energy-ranked tile-stride
matching (from sheaf_merge.py). All other mechanics (cute FA2 kernel,
merge/MLP/unmerge) are unchanged from the SparseSAM PE partial patch.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..pe_base import (
    SelfAttention, ResidualAttentionBlock,
    _ensure_cute_deps, _get_kernel, _module_cached_cos_sin,
    _find_vision_transformer, _vit_uses_cls_token,
    _make_A_mask, flash_rope_sparse_attn,
    _ensure_block_mask,
)
from ..sheafsam.sam import tile_stride_matching


@torch.no_grad()
def _build_sheaf_partial_cache(
    self_attn: nn.Module, S: int, dtype: torch.dtype,
    k: torch.Tensor,
    sparse_ratio: float, group_size: int, has_cls: bool,
    score_mode: str, sheaf_blend: float,
    sheaf_project_dim: int, sheaf_group_reduce: str,
    device,
) -> dict:
    """Pre-build sheaf-ranked perm cache."""
    kernel, _m_blk, n_blk = _get_kernel(dtype, self_attn.head_dim)
    if kernel is None:
        return {}

    cos_full, sin_full = _module_cached_cos_sin(self_attn, dtype)
    cos = cos_full[:S]
    sin = sin_full[:S]

    # Sheaf-ranked permutation via tile_stride_matching
    # PE uses a square grid: S = 1 (CLS) + H*W, so win = sqrt(S-1) if has_cls
    cls_off = 1 if has_cls else 0
    N = S - cls_off
    win = int(math.sqrt(N))
    if win * win != N:
        # Non-square grid — fall back to stride-sort (original SparseSAM)
        from ..sparsesam.pe_partial import _build_partial_cache
        return _build_partial_cache(
            self_attn, S, dtype, sparse_ratio, group_size, has_cls, device,
        )

    # Strip CLS token for tile_stride_matching (it expects pure grid tokens)
    k_grid = k[:1, cls_off:, :].contiguous()  # (1, H*W, D)
    p, ip, _ = tile_stride_matching(
        k_grid,
        win, win,
        ratio=sparse_ratio,
        group_size=group_size,
        n_block=n_blk,
        perm_mode="z_interleave_sort",
        score_mode=score_mode,
        sheaf_blend=sheaf_blend,
        sheaf_project_dim=sheaf_project_dim,
        sheaf_group_reduce=sheaf_group_reduce,
    )
    # Prepend CLS index (0) to perm/inv_perm since tile_stride_matching works on grid only
    perm_grid = p[0]         # (H*W,)
    inv_perm_grid = ip[0]    # (H*W,)
    if has_cls:
        perm = torch.cat([torch.zeros(1, dtype=perm_grid.dtype, device=device),
                          perm_grid + cls_off], dim=0)  # (S,)
        inv_perm = torch.cat([torch.zeros(1, dtype=inv_perm_grid.dtype, device=device),
                              inv_perm_grid + cls_off], dim=0)
    else:
        perm = perm_grid
        inv_perm = inv_perm_grid

    cos_perm = cos.index_select(0, perm).contiguous()
    sin_perm = sin.index_select(0, perm).contiguous()

    cls_off = 1 if has_cls else 0
    N = S - cls_off
    if N > 0 and group_size > 0 and N % group_size == 0:
        n_groups = N // group_size
        K = max(1, round(sparse_ratio * N))
        if K >= n_groups:
            n_keep = max(0, (K - n_groups) // (group_size - 1))
            n_keep = min(n_keep, n_groups)
        else:
            n_keep = 0
        n_merge = n_groups - n_keep
        cls_part_size = cls_off + n_keep * group_size
    else:
        n_merge = 0
        cls_part_size = S

    return {
        "cos": cos_perm, "sin": sin_perm,
        "perm": perm, "inv_perm": inv_perm,
        "block_mask": None,
        "cls_part_size": cls_part_size,
        "n_merge": n_merge,
        "gs": group_size,
    }


def _kernel_dtype(self_attn) -> torch.dtype:
    w_dtype = self_attn.in_proj_weight.dtype
    if w_dtype in (torch.float16, torch.bfloat16):
        return w_dtype
    return torch.float16


class SheafPEPartialAttention(SelfAttention):
    """Sheaf-ranked block-sparse cute kernel attention for PE."""

    def forward(self, x, attn_mask=None):
        info = self._tome_info
        sr = info.get("sparse_ratio", info.get("ratio", 1.0))

        kdtype = _kernel_dtype(self)
        kernel, _m_blk, _n_blk = _get_kernel(kdtype, self.head_dim)
        if kernel is None:
            return super().forward(x, attn_mask=attn_mask)

        cache = info.get("_perm_cache")
        cache_key = (x.shape[1], kdtype, sr,
                     info.get("score_mode", "sheaf"),
                     info.get("sheaf_blend", 1.0),
                     info.get("sheaf_project_dim", 16),
                     info.get("sheaf_group_reduce", "mean"))
        if cache is None or cache.get("_key") != cache_key:
            # We need k for sheaf scoring — compute a temporary qkv
            B, S, D = x.shape
            qkv_w = self.in_proj_weight
            qkv_b = self.in_proj_bias if self.in_proj_bias is not None else None
            # Only compute k (second third of qkv)
            head_dim = self.head_dim
            num_heads = self.num_heads
            k_tmp = torch.nn.functional.linear(
                x[:1], qkv_w[head_dim*num_heads:2*head_dim*num_heads],
                qkv_b[head_dim*num_heads:2*head_dim*num_heads] if qkv_b is not None else None,
            )

            cache = _build_sheaf_partial_cache(
                self, S, kdtype, k_tmp, sr,
                info.get("group_size", 4),
                info.get("use_cls_token", False),
                info.get("score_mode", "sheaf"),
                info.get("sheaf_blend", 1.0),
                info.get("sheaf_project_dim", 16),
                info.get("sheaf_group_reduce", "mean"),
                x.device,
            )
            cache["_key"] = cache_key
            info["_perm_cache"] = cache

        _ensure_block_mask(cache, self, x, sr, dtype=kdtype)

        permuted = bool(info.get("x_is_permuted"))
        out = flash_rope_sparse_attn(
            self, x,
            cos=cache.get("cos"), sin=cache.get("sin"),
            block_mask=cache.get("block_mask"),
            perm=cache.get("perm"), inv_perm=cache.get("inv_perm"),
            assume_permuted=permuted,
        )
        if out is not None:
            return out

        if permuted and cache.get("inv_perm") is not None:
            x = x.index_select(1, cache["inv_perm"])
        out = super().forward(x, attn_mask=attn_mask)
        if permuted and cache.get("perm") is not None:
            out = out.index_select(1, cache["perm"])
        return out


class SheafPEPartialBlock(ResidualAttentionBlock):
    """Block forward: sheaf-ranked permute, cute sparse-attn, merge/MLP/unmerge."""

    def forward(self, x, attn_mask=None):
        info: dict = self._tome_info
        ratio = info.get("ratio", 1.0)
        sr = info.get("sparse_ratio", ratio)

        if not info.get("x_is_permuted"):
            kdtype = _kernel_dtype(self.attn)
            kernel, _, _ = _get_kernel(kdtype, self.attn.head_dim)
            if kernel is not None:
                perm = info.get("_perm_cache", {}).get("perm")
                if perm is not None:
                    x = x.index_select(1, perm)
                    info["x_is_permuted"] = True

        x = x + self.drop_path1(
            self.ls_1(self._call_attn(self.ln_1(x), attn_mask=attn_mask))
        )

        cache = info.get("_perm_cache")
        n_merge = (cache.get("n_merge", 0) if cache else 0)
        mlp_merge = info.get("mlp_merge", True)
        if (mlp_merge and info.get("x_is_permuted")
                and n_merge > 0 and ratio < 1.0):
            B, S, C = x.shape
            cls_part_size = cache["cls_part_size"]
            gs = cache["gs"]

            keep_part = x[:, :cls_part_size, :]
            merge_section = x[:, cls_part_size:, :]
            merge_view = merge_section.reshape(B, gs, n_merge, C)
            merge_repr = merge_view[:, 0, :, :]

            reduced_x = torch.cat([keep_part, merge_repr], dim=1)
            reduced_x = reduced_x + self.drop_path2(
                self.ls_2(self.mlp(self.ln_2(reduced_x)))
            )

            keep_out = reduced_x[:, :cls_part_size, :]
            merge_repr_out = reduced_x[:, cls_part_size:, :]
            merge_section_out = (merge_repr_out
                                  .unsqueeze(1)
                                  .expand(B, gs, n_merge, C)
                                  .reshape(B, gs * n_merge, C))

            x = torch.cat([keep_out, merge_section_out], dim=1)
        else:
            x = x + self.drop_path2(self.ls_2(self.mlp(self.ln_2(x))))

        return x


def _reset_state_hook(info):
    def _hook(_module, _inputs):
        info["x_is_permuted"] = False
    return _hook


def _transformer_unpermute_post_hook(info):
    def _hook(_module, _inputs, output):
        if not info.get("x_is_permuted"):
            return output
        cache = info.get("_perm_cache")
        inv_perm = cache.get("inv_perm") if cache else None
        if inv_perm is not None:
            output = output.index_select(1, inv_perm)
        info["x_is_permuted"] = False
        return output
    return _hook


def apply_pe_sheaf_partial_patch(
    model: nn.Module,
    ratio: float = 0.7,
    group_size: int = 4,
    sparse_ratio: Optional[float] = None,
    start_block: int = 0,
    mlp_merge: bool = False,
    score_mode: str = "sheaf",
    sheaf_blend: float = 1.0,
    sheaf_project_dim: int = 16,
    sheaf_group_reduce: str = "mean",
    verbose: bool = True,
    **_
) -> int:
    """Sheaf-ranked sparse attention for PE-Core models."""
    transformer = _find_vision_transformer(model)
    if transformer is None:
        raise RuntimeError("Could not locate the PE vision Transformer in `model`.")

    _ensure_cute_deps()

    n_blocks = len(transformer.resblocks)
    sb = max(0, min(int(start_block), n_blocks))

    use_cls_token = _vit_uses_cls_token(model)

    info = {
        "ratio": float(ratio),
        "sparse_ratio": float(sparse_ratio if sparse_ratio is not None else ratio),
        "group_size": int(group_size),
        "use_cls_token": use_cls_token,
        "mlp_merge": bool(mlp_merge),
        "score_mode": score_mode,
        "sheaf_blend": float(sheaf_blend),
        "sheaf_project_dim": int(sheaf_project_dim),
        "sheaf_group_reduce": sheaf_group_reduce,
    }
    model._tome_info = info

    if not hasattr(transformer, "_pe_partial_pre_hook"):
        transformer._pe_partial_pre_hook = transformer.register_forward_pre_hook(
            _reset_state_hook(info)
        )
    if not hasattr(transformer, "_pe_partial_post_hook"):
        transformer._pe_partial_post_hook = transformer.register_forward_hook(
            _transformer_unpermute_post_hook(info)
        )

    n_attn = 0
    for idx in range(sb, n_blocks):
        blk = transformer.resblocks[idx]

        attn = getattr(blk, "attn", None)
        if isinstance(attn, SelfAttention) and attn.rope is not None:
            if not isinstance(attn, SheafPEPartialAttention):
                attn.__class__ = SheafPEPartialAttention
            attn._tome_info = info
            n_attn += 1

        if not isinstance(blk, SheafPEPartialBlock):
            blk.__class__ = SheafPEPartialBlock
        blk._tome_info = info

    if verbose:
        print(f"[pe-sheaf-partial] L={n_blocks}  start_block={sb}  "
              f"patched_blocks={n_blocks - sb}  patched_attn={n_attn}  "
              f"ratio={info['ratio']}  sparse_ratio={info['sparse_ratio']}  "
              f"score_mode={score_mode}  sheaf_blend={sheaf_blend}  "
              f"mlp_merge={info['mlp_merge']}")
    return n_blocks - sb


def remove_pe_sheaf_partial_patch(model: nn.Module) -> int:
    from ..registry import remove_all_pe
    return remove_all_pe(model)


__all__ = [
    "apply_pe_sheaf_partial_patch",
    "remove_pe_sheaf_partial_patch",
]
