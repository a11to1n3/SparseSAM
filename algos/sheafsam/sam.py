
import sys
import os
import types
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack
import torch
import torch.nn.functional as F

_here = os.path.dirname(__file__)
_sam_root = os.path.normpath(os.path.join(_here, '..', '3rd_party', 'sam-hq'))
if _sam_root not in sys.path:
    sys.path.insert(0, _sam_root)

from ..kernels.flash_attn import FlashAttentionForwardAmpere

from segment_anything.modeling.image_encoder import (
    ImageEncoderViT,
    Block,
    Attention,
    window_partition,
    window_unpartition,
)
from ..sparsesam.hilbert_utils import get_hilbert_order
from ..sparsesam.z_utils import get_z_order
from .sheaf_utils import (
    identity_sheaf_token_energy,
    zscore_like,
    sobel_sheaf_token_energy,
    graph_coboundary_token_energy,
    group_score_from_token_energy,
)

TRUE_COB_SCORE_MODES = {
    "coboundary4", "coboundary4_full",
    "coboundary8", "coboundary8_full",
    "coboundary_ms", "coboundary_ms_full",
    "laplacian4", "laplacian8",
}
from .sheaf_merge import build_sheaf_cell_merge


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:

    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()].half()


_FA2_M_BLOCK_LOCAL = 64
_FA2_N_BLOCK_LOCAL = 64
_FA2_M_BLOCK_GLOBAL = 64
_FA2_N_BLOCK_GLOBAL = 64
_FA2_THREADS_LOCAL = 128
_FA2_THREADS_GLOBAL = 128

_FA2_DTYPE_FP16 = cutlass.dtype("Float16")


def _perm_cache_key(
    win: int,
    ratio: float,
    n_block: int,
    info: dict,
) -> tuple:
    """Canonical permutation cache key shared by attention and MLP routing.
    Avoids key-format mismatches that cause silent cache misses."""
    return (
        int(win),
        float(ratio),
        int(n_block),
        info.get("perm_mode", "z_interleave_sort"),
        info.get("score_mode", "sheaf"),
        float(info.get("sheaf_blend", 1.0)),
        int(info.get("sheaf_project_dim", 16)),
        info.get("sheaf_group_reduce", "mean"),
        int(info.get("group_size", 4)),
    )


def compute_rel_bias(
    q_bshd: torch.Tensor,
    Rh: torch.Tensor,
    Rw: torch.Tensor,
    win: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, N, D = q_bshd.shape
    r_q   = q_bshd.reshape(B , win, win, D)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh).reshape(B, N, win)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw).reshape(B, N, win)
    to_fa2 = lambda t: t.unsqueeze(1).permute(0, 2, 1, 3).contiguous()
    return to_fa2(rel_h), to_fa2(rel_w)


def _wrap_qkvo(t: torch.Tensor, dtype) -> "cute.Tensor":
    return (from_dlpack(t, assumed_align=16)
            .mark_layout_dynamic(leading_dim=3)
            .mark_compact_shape_dynamic(
                mode=3,
                stride_order=t.dim_order(),
                divisibility=128 // dtype.width))


def _wrap_bias(t: torch.Tensor) -> "cute.Tensor":
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=3)


def _wrap_perm(t: torch.Tensor) -> "cute.Tensor":
    ct = from_dlpack(t, assumed_align=4)
    if t.dim() == 2:
        ct = ct.mark_layout_dynamic(leading_dim=1)
    return ct


_CU_STREAM = None
def _get_cu_stream():
    """Wrap the default CUDA stream once and reuse (SAM-HQ inference runs on the default stream)."""
    global _CU_STREAM
    if _CU_STREAM is None:
        _CU_STREAM = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return _CU_STREAM


def tile_stride_matching(
    x: torch.Tensor,
    H: int,
    W: int,
    ratio: float = 0.0,
    group_size: int = 4,
    n_block: int = 64,
    perm_mode: str = "z_interleave_sort",
    score_mode: str = "sheaf",          # new
    sheaf_blend: float = 1.0,           # new
    sheaf_project_dim: int = 16,        # new
    sheaf_group_reduce: str = "mean",   # new
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Order tokens for sparsesam attention; return (perm, inv_perm, inv_perm_group).
    `perm_mode` is `<space>_<layout>_<ranking>` — space ∈ {z, hilbert},
    layout ∈ {interleave, naive}, ranking ∈ {sort, nosort}.
    When score_mode in ("sheaf", "hybrid") the group ranking uses identity-sheaf energy.
    """
    N = H * W
    assert N % group_size == 0, f"N={N} must be divisible by group_size={group_size}"

    parts = perm_mode.split("_")
    assert len(parts) == 3, f"perm_mode must be <space>_<layout>_<ranking>, got {perm_mode!r}"
    space, layout, ranking = parts
    assert space   in ("z", "hilbert"),         f"unknown space {space!r}"
    assert layout  in ("interleave", "naive"),  f"unknown layout {layout!r}"
    assert ranking in ("sort", "nosort"),       f"unknown ranking {ranking!r}"

    n_groups = N // group_size
    gs       = group_size

    with torch.no_grad():
        B, N, C = x.shape
        device  = x.device
        if space == "hilbert":
            base_perm = get_hilbert_order(H, W, device=device)
        else:
            base_perm = get_z_order(H, W, device=device)
        x_b    = x[:, base_perm, :].float()
        x_grp  = x_b.view(B, n_groups, gs, C)

        if ranking == "sort":
            sparse_score = None
            sheaf_score = None

            # Compute SparseSAM score only when needed
            if score_mode in ("sparse", "hybrid"):
                grp_std = x_grp.std(dim=2, correction=0).mean(dim=-1)
                grp_mean = x_grp.mean(dim=2)
                gm_norm = F.normalize(grp_mean, dim=-1)
                sim = gm_norm @ gm_norm.transpose(1, 2)
                avg_sim = (sim.sum(dim=-1) - 1.0) / max(n_groups - 1, 1)
                dissim = -avg_sim
                sparse_score = zscore_like(grp_std) + zscore_like(dissim)

            # Compute sheaf energy only when needed
            if score_mode in ("sheaf", "sheaf_full", "hybrid", "shuffled_sheaf"):
                pd = 0 if score_mode == "sheaf_full" else sheaf_project_dim
                token_energy = identity_sheaf_token_energy(
                    x, H=H, W=W, project_dim=pd,
                    normalize=True, edge_reduce="mean",
                )
                if score_mode == "shuffled_sheaf":
                    rand = torch.randperm(token_energy.shape[1], device=device)
                    token_energy = token_energy[:, rand]
                sheaf_score = group_score_from_token_energy(
                    token_energy=token_energy, base_perm=base_perm,
                    group_size=gs, reduce=sheaf_group_reduce,
                )
                sheaf_score = zscore_like(sheaf_score)

            # Sobel-K / sobel_sheaf
            if score_mode in ("sobel_sheaf", "sobel_k"):
                token_energy = sobel_sheaf_token_energy(
                    x, H=H, W=W, project_dim=0,
                    normalize=True, reduce="mean", padding_mode="replicate",
                )
                sobel_score = group_score_from_token_energy(
                    token_energy=token_energy, base_perm=base_perm,
                    group_size=gs, reduce=sheaf_group_reduce,
                )
                sobel_score = zscore_like(sobel_score)

            # True graph coboundary (strict incidence-based, no convolution)
            if score_mode in TRUE_COB_SCORE_MODES:
                if score_mode.startswith("coboundary8") or score_mode == "laplacian8":
                    graph = "8"
                elif score_mode.startswith("coboundary_ms"):
                    graph = "ms4"
                else:
                    graph = "4"
                pd = 0 if score_mode.endswith("_full") else sheaf_project_dim
                sk = "laplacian_residual" if score_mode.startswith("laplacian") else "edge_energy"
                token_energy = graph_coboundary_token_energy(
                    x, H=H, W=W, project_dim=pd, graph=graph,
                    normalize=True, edge_reduce="mean",
                    degree_normalize=True, score_kind=sk,
                )
                cob_score = group_score_from_token_energy(
                    token_energy=token_energy, base_perm=base_perm,
                    group_size=gs, reduce=sheaf_group_reduce,
                )
                cob_score = zscore_like(cob_score)

            if score_mode == "sparse":
                grp_score = sparse_score
            elif score_mode in ("sheaf", "sheaf_full", "shuffled_sheaf"):
                grp_score = sheaf_score
            elif score_mode in ("sobel_sheaf", "sobel_k"):
                grp_score = sobel_score
            elif score_mode in TRUE_COB_SCORE_MODES:
                grp_score = cob_score
            elif score_mode == "hybrid":
                grp_score = sparse_score + float(sheaf_blend) * torch.relu(sheaf_score)
            else:
                raise ValueError(f"unknown score_mode={score_mode!r}")

            grp_rank = grp_score.argsort(dim=-1, descending=True)
        else:  # nosort: keep groups in space-filling order
            grp_rank = (torch.arange(n_groups, device=device)
                        .unsqueeze(0).expand(B, -1).contiguous())

        group_raster = base_perm.view(n_groups, gs)
        all_raster = group_raster[grp_rank.reshape(-1)].reshape(B, n_groups, gs)
        perm_1d_group = all_raster.reshape(B, N)
        if layout == "interleave":
            perm_1d = all_raster.permute(0, 2, 1).reshape(B, N)
        else:  # naive
            perm_1d = perm_1d_group.clone()
    inv_perm_1d = torch.argsort(perm_1d, dim=1)
    inv_perm_1d_group = torch.argsort(perm_1d_group, dim=1)
    return perm_1d, inv_perm_1d, inv_perm_1d_group



class SheafSAMAttention(Attention):

    def forward(self, x: torch.Tensor, ratio: float, use_fa2: bool = True,  # noqa: ARG002
                m_block: int = _FA2_M_BLOCK_LOCAL,
                n_block: int = _FA2_N_BLOCK_LOCAL,
                threads: int = _FA2_THREADS_LOCAL,
                is_global: bool = False,
                return_perm: bool = False) -> torch.Tensor:
        B, H, W, _ = x.shape
        Sq  = H * W
        D   = _ // self.num_heads
        win = H

        BH = B * self.num_heads
        qkv = self.qkv(x.view(B, Sq, -1))
        qkv = qkv.view(B, Sq, 3, self.num_heads, D).permute(2, 0, 3, 1, 4).reshape(3, BH, Sq, D)
        q, k, v = qkv.unbind(0)
        o = torch.empty_like(q)

        if not hasattr(self, '_Rh') or self._Rh is None:
            self._Rh = get_rel_pos(win, win, self.rel_pos_h)
            self._Rw = get_rel_pos(win, win, self.rel_pos_w)
        Rh, Rw = self._Rh, self._Rw
        rel_h, rel_w = compute_rel_bias(q, Rh, Rw, win)

        info = self._tome_info

        perm_mode = info.get("perm_mode", "z_interleave_sort")
        score_mode = info.get("score_mode", "sheaf")

        perm_cache = info.setdefault("perm_cache", {})

        cache_key = _perm_cache_key(win=win, ratio=ratio, n_block=n_block, info=info)

        if cache_key not in perm_cache:
            sheaf_blend = float(info.get("sheaf_blend", 1.0))
            sheaf_project_dim = int(info.get("sheaf_project_dim", 16))
            sheaf_group_reduce = info.get("sheaf_group_reduce", "mean")
            group_size = int(info.get("group_size", 4))

            p, ip, ipg = tile_stride_matching(
                k, win, win,
                ratio=ratio,
                group_size=group_size,
                n_block=n_block,
                perm_mode=perm_mode,
                score_mode=score_mode,
                sheaf_blend=sheaf_blend,
                sheaf_project_dim=sheaf_project_dim,
                sheaf_group_reduce=sheaf_group_reduce,
            )
            p_i32 = p.to(torch.int32)
            perm_cache[cache_key] = (
                p, ip, ipg, p_i32,
                _wrap_perm(p_i32), _wrap_perm(p_i32),
            )
        perm, inv_perm, _, _, perm_q_c, perm_k_c = perm_cache[cache_key]

        perm_e = perm.unsqueeze(-1).expand(-1, -1, D)
        q_p = q.gather(1, perm_e)
        k_p = k.gather(1, perm_e)
        v_p = v.gather(1, perm_e)

        cu_stream = _get_cu_stream()
        q_c = _wrap_qkvo(q_p.unsqueeze(2), _FA2_DTYPE_FP16)
        k_c = _wrap_qkvo(k_p.unsqueeze(2), _FA2_DTYPE_FP16)
        v_c = _wrap_qkvo(v_p.unsqueeze(2), _FA2_DTYPE_FP16)
        o_c = _wrap_qkvo(o.unsqueeze(2), _FA2_DTYPE_FP16)
        rh_c = _wrap_bias(rel_h)
        rw_c = _wrap_bias(rel_w)

        # A-shape sparsity (paper §4.1): global gets band+keep-bar (with_diagonal=True);
        # win14 gets keep-bar only (+1 col mirrors the historical mask formula).
        num_n_blocks = (Sq + n_block - 1) // n_block
        n_keep_cols = int(ratio * num_n_blocks)
        n_init_blocks = n_keep_cols if is_global else n_keep_cols + 1
        compiled = FlashAttentionForwardAmpere.get_compiled_a_shape(
            D, m_block, n_block, threads, win, is_global,
            q_c, k_c, v_c, o_c, rh_c, rw_c, perm_q_c, perm_k_c,
            n_init_blocks, self.scale, cu_stream,
        )
        compiled(q_c, k_c, v_c, o_c, rh_c, rw_c, perm_q_c, perm_k_c,
                 n_init_blocks, self.scale, cu_stream)
        inv_perm_e = inv_perm.unsqueeze(-1).expand(-1, -1, D)
        o_out = o.gather(1, inv_perm_e).reshape(B, self.num_heads, Sq, D).permute(0, 2, 1, 3).reshape(B, Sq, -1)
        out = self.proj(o_out).reshape(B, H, W, -1)
        if return_perm:
            return out, perm, inv_perm
        return out


class SheafSAMBlock(Block):


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H_sp, W_sp, C = x.shape
        info  = self._tome_info
        ratio = info["ratio"].pop(0)

        shortcut = x
        x_n = self.norm1(x)
        if self.window_size > 0:
            ws = self.window_size
            H_w, W_w = x_n.shape[1], x_n.shape[2]
            x_n_win, pad_hw = window_partition(x_n, ws)
            x_attn, perm, inv_perm = self.attn(
                x_n_win, ratio, use_fa2=True, return_perm=True,
            )
            x_attn = window_unpartition(x_attn, ws, pad_hw, (H_w, W_w))
        else:
            x_attn, perm, inv_perm = self.attn(
                x_n, ratio,
                m_block=_FA2_M_BLOCK_GLOBAL,
                n_block=_FA2_N_BLOCK_GLOBAL,
                threads=_FA2_THREADS_GLOBAL,
                is_global=True,
                return_perm=True,
            )

        x = shortcut + x_attn
        x_seq = x.reshape(B, H_sp * W_sp, C)

        mlp_merge = info.get("mlp_merge", True)
        mlp_mode = info.get("mlp_mode", "sheaf_route")

        if mlp_merge and ratio < 1.0 and self.window_size > 0:
            keep_n = max(1, round(ratio * x_seq.shape[1]))

            if mlp_mode == "full":
                # Intentional full MLP — quality mode. No token dropping.
                x_seq = x_seq + self.mlp(self.norm2(x_seq))

            elif mlp_mode == "sheaf_route":
                # Gated sheaf MLP routing: only route if global permutation
                # cache exists (after first global block). Falls back to full MLP.
                sheaf_project_dim = int(info.get("sheaf_project_dim", 16))
                global_key = _perm_cache_key(
                    win=H_sp, ratio=ratio, n_block=_FA2_N_BLOCK_GLOBAL, info=info,
                )
                global_cached = info.get("perm_cache", {}).get(global_key)

                if global_cached is None:
                    x_seq = x_seq + self.mlp(self.norm2(x_seq))
                else:
                    token_energy = identity_sheaf_token_energy(
                        x_seq, H=H_sp, W=W_sp,
                        project_dim=sheaf_project_dim,
                        normalize=True, edge_reduce="mean",
                    )
                    top_idx = token_energy.topk(keep_n, dim=1, largest=True).indices
                    idx_e = top_idx.unsqueeze(-1).expand(-1, -1, C)
                    x_kept = x_seq.gather(1, idx_e)
                    x_kept = x_kept + self.mlp(self.norm2(x_kept))
                    x_seq = x_seq.scatter(1, idx_e, x_kept)

            elif mlp_mode == "mixed_route":
                # Quota-based split: SparseSAM-ranked tokens + sheaf-energy tokens.
                sheaf_project_dim = int(info.get("sheaf_project_dim", 16))
                sheaf_mlp_quota = float(info.get("sheaf_mlp_quota", 0.10))
                global_key = _perm_cache_key(
                    win=H_sp, ratio=ratio, n_block=_FA2_N_BLOCK_GLOBAL, info=info,
                )
                global_cached = info.get("perm_cache", {}).get(global_key)

                if global_cached is None:
                    x_seq = x_seq + self.mlp(self.norm2(x_seq))
                else:
                    n_sheaf = max(1, round(keep_n * sheaf_mlp_quota))
                    n_sparse = keep_n - n_sheaf
                    inv_perm_1d_group = global_cached[2]
                    nh = self.attn.num_heads
                    avg_rank = inv_perm_1d_group.view(B, nh, -1).float().mean(dim=1)
                    sparse_idx = avg_rank.topk(n_sparse, dim=1, largest=False).indices
                    token_energy = identity_sheaf_token_energy(
                        x_seq, H=H_sp, W=W_sp,
                        project_dim=sheaf_project_dim,
                        normalize=True, edge_reduce="mean",
                    )
                    sheaf_idx = token_energy.topk(n_sheaf, dim=1, largest=True).indices
                    merged = []
                    for b in range(B):
                        both = torch.cat([sparse_idx[b], sheaf_idx[b]], dim=0)
                        seen = torch.unique(both, sorted=False)
                        if seen.numel() < keep_n:
                            all_idx = torch.arange(x_seq.shape[1], device=seen.device)
                            mask = torch.ones(x_seq.shape[1], dtype=torch.bool, device=seen.device)
                            mask[seen] = False
                            filler = all_idx[mask][:keep_n - seen.numel()]
                            seen = torch.cat([seen, filler])
                        merged.append(seen[:keep_n])
                    top_idx = torch.stack(merged, dim=0)
                    idx_e = top_idx.unsqueeze(-1).expand(-1, -1, C)
                    x_kept = x_seq.gather(1, idx_e)
                    x_kept = x_kept + self.mlp(self.norm2(x_kept))
                    x_seq = x_seq.scatter(1, idx_e, x_kept)

            elif mlp_mode == "sparse_route":
                # SparseSAM-style partial MLP: ranked by global inverse permutation.
                global_key = _perm_cache_key(
                    win=H_sp, ratio=ratio, n_block=_FA2_N_BLOCK_GLOBAL, info=info,
                )
                global_cached = info.get("perm_cache", {}).get(global_key)

                if global_cached is not None:
                    inv_perm_1d_group = global_cached[2]
                    nh = self.attn.num_heads
                    avg_rank = inv_perm_1d_group.view(B, nh, -1).float().mean(dim=1)
                    top_idx = avg_rank.topk(keep_n, dim=1, largest=False).indices
                    idx_e = top_idx.unsqueeze(-1).expand(-1, -1, C)
                    x_kept = x_seq.gather(1, idx_e)
                    x_kept = x_kept + self.mlp(self.norm2(x_kept))
                    x_seq = x_seq.scatter(1, idx_e, x_kept)
                else:
                    x_seq = x_seq + self.mlp(self.norm2(x_seq))

            elif mlp_mode == "sheaf_merge":
                token_energy = identity_sheaf_token_energy(
                    x_seq,
                    H=H_sp,
                    W=W_sp,
                    project_dim=int(info.get("sheaf_project_dim", 16)),
                    normalize=True,
                    edge_reduce="mean",
                )

                merge, unmerge, _stats = build_sheaf_cell_merge(
                    token_energy=token_energy,
                    H=H_sp,
                    W=W_sp,
                    ratio=ratio,
                    cell_h=int(info.get("sheaf_cell_h", 2)),
                    cell_w=int(info.get("sheaf_cell_w", 2)),
                )

                x_merged = merge(x_seq, mode="mean")
                x_merged = x_merged + self.mlp(self.norm2(x_merged))
                x_seq = unmerge(x_merged)

            else:
                raise ValueError(f"unknown mlp_mode={mlp_mode!r}")

        else:
            x_seq = x_seq + self.mlp(self.norm2(x_seq))

        return x_seq.reshape(B, H_sp, W_sp, C)


def _warmup_fa2_kernels(encoder: ImageEncoderViT) -> None:
    """Pre-compile the FA2 a_shape kernel for every distinct config in the encoder,
    and prime each block's attn._Rh / attn._Rw cache."""
    device = next(encoder.parameters()).device
    seen: set = set()

    for blk in encoder.blocks:
        attn      = blk.attn
        is_global = (blk.window_size == 0)
        win = (attn.rel_pos_h.shape[0] + 1) // 2
        D   = attn.rel_pos_h.shape[1]

        if not hasattr(attn, '_Rh') or attn._Rh is None:
            attn._Rh = get_rel_pos(win, win, attn.rel_pos_h)
            attn._Rw = get_rel_pos(win, win, attn.rel_pos_w)

        m_block, n_block, threads = (
            (_FA2_M_BLOCK_GLOBAL, _FA2_N_BLOCK_GLOBAL, _FA2_THREADS_GLOBAL)
            if is_global else
            (_FA2_M_BLOCK_LOCAL, _FA2_N_BLOCK_LOCAL, _FA2_THREADS_LOCAL)
        )
        compile_key = (win, D, m_block, n_block, threads, is_global)
        if compile_key in seen or not FlashAttentionForwardAmpere.cached_can_implement(
            _FA2_DTYPE_FP16, D, m_block, n_block, threads, win,
        ):
            seen.add(compile_key)
            continue
        seen.add(compile_key)

        # Zero dummy tensors — cute.compile only needs them to bake in layouts.
        Sq, H = win * win, attn.num_heads
        zero = lambda *shape: torch.zeros(*shape, dtype=torch.float16, device=device)
        q, k, v, o = zero(1, Sq, H, D), zero(1, Sq, H, D), zero(1, Sq, H, D), zero(1, Sq, H, D)
        rel_h, rel_w = zero(1, Sq, H, win), zero(1, Sq, H, win)
        identity = torch.arange(Sq, device=device, dtype=torch.int32).unsqueeze(0)
        cu_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        label = "global" if is_global else f"win{win}"
        print(
            f"[ToMe-SAM] compiling FA2 kernel  {label}  "
            f"win={win}  D={D}  m={m_block}  n={n_block}  T={threads}   ...",
            end=" ", flush=True,
        )
        n_init_blocks_warm = int(0.5 * ((Sq + n_block - 1) // n_block))
        FlashAttentionForwardAmpere.get_compiled_a_shape(
            D, m_block, n_block, threads, win, is_global,
            _wrap_qkvo(q, _FA2_DTYPE_FP16), _wrap_qkvo(k, _FA2_DTYPE_FP16),
            _wrap_qkvo(v, _FA2_DTYPE_FP16), _wrap_qkvo(o, _FA2_DTYPE_FP16),
            _wrap_bias(rel_h), _wrap_bias(rel_w),
            _wrap_perm(identity), _wrap_perm(identity),
            n_init_blocks_warm, attn.scale, cu_stream,
        )
        print("done")


def apply_patch(
    encoder: ImageEncoderViT,
    algo: str = "sheafsam",
    ratio: float = 0.9,
    margin: float = 0.5,
    mlp_merge: bool = True,
    perm_mode: str = "z_interleave_sort",
    score_mode: str = "sheaf",
    sheaf_blend: float = 1.0,
    sheaf_project_dim: int = 16,
    sheaf_group_reduce: str = "mean",
    group_size: int = 4,
    mlp_mode: str = "sparse_route",
    sheaf_cell_h: int = 2,
    sheaf_cell_w: int = 2,
    sheaf_mlp_quota: float = 0.10,
    **_,
) -> ImageEncoderViT:
    assert 0 < ratio <= 1.0, "ratio must be in (0, 1]"

    tome_info = {
        "ratio": ratio,
        "mlp_merge": bool(mlp_merge),
        "perm_mode": perm_mode,
        "score_mode": score_mode,
        "sheaf_blend": float(sheaf_blend),
        "sheaf_project_dim": int(sheaf_project_dim),
        "sheaf_group_reduce": sheaf_group_reduce,
        "group_size": int(group_size),
        "mlp_mode": mlp_mode,
        "sheaf_cell_h": int(sheaf_cell_h),
        "sheaf_cell_w": int(sheaf_cell_w),
        "sheaf_mlp_quota": float(sheaf_mlp_quota),
    }

    encoder.tome_info = tome_info
    _orig_forward = encoder.__class__.forward

    def _patched_forward(self, x: torch.Tensor):
        n = len(self.blocks)
        r = self.tome_info["ratio"]
        self.tome_info["ratio"] = [r] * n
        self.tome_info["perm_cache"] = {}
        result = _orig_forward(self, x)
        self.tome_info["ratio"] = r
        return result

    encoder.forward = types.MethodType(_patched_forward, encoder)

    for module in encoder.modules():
        if isinstance(module, Block) and not isinstance(module, SheafSAMBlock):
            module.__class__ = SheafSAMBlock
            module._tome_info = tome_info
        elif isinstance(module, Attention) and not isinstance(module, SheafSAMAttention):
            module.__class__ = SheafSAMAttention
            module._tome_info = tome_info

    n_blocks = len(encoder.blocks)
    n_global = sum(1 for blk in encoder.blocks if blk.window_size == 0)

    print(
        f"[SheafSAM] patched ratio={ratio}"
        f" mlp_merge={tome_info['mlp_merge']}"
        f" perm_mode={perm_mode}"
        f" score_mode={score_mode}"
        f" sheaf_blend={sheaf_blend}"
        f" sheaf_project_dim={sheaf_project_dim}"
        f" mlp_mode={mlp_mode}"
        f" mlp_quota={sheaf_mlp_quota:.2f}"
        f" blocks={n_blocks} (global={n_global} local={n_blocks-n_global})"
    )

    _warmup_fa2_kernels(encoder)
    return encoder
