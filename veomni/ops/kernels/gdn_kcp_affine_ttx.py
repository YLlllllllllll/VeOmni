# Copyright (c) 2025 VeOmni Authors.
"""P1.5b-kernel: Ascend TTX fused local-affine pre-scan (NEW op — not GDR).

Scope lock: only the LTR ``{he, M}`` recurrence. Does **not** touch AG /
zigzag / prefix-merge / ``S_init``.

INV-7: kernel may mix bf16/fp32 internally; returned ``he``/``M`` are **fp32**.

Ascend UB constraint: full ``[K,K]``+``[K,V]`` with K=V=128 overflows on-chip
buffer (~192KB). State stays in HBM; the fused T-loop tiles 64×64 updates
(rank-1 expanded, same order as eager)::

    he ← exp(g) * (he - beta * k ⊗ (kᵀ he)) + (beta * k) ⊗ v
    M  ← exp(g) * (M  - beta * k ⊗ (kᵀ M))

Do **not** reuse ``mojo_chunk_gated_delta_rule``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    _TRITON_OK = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    _TRITON_OK = False


def _npu_ready() -> bool:
    return bool(getattr(torch, "npu", None)) and torch.npu.is_available()


if _TRITON_OK:

    @triton.jit(do_not_specialize=["T"])
    def _local_affine_summary_fwd_kernel(
        k_ptr,
        v_ptr,
        g_ptr,
        beta_ptr,
        he_ptr,
        M_ptr,
        T,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
    ):
        """One program = one head. HBM-resident he/M; tiled UB updates."""
        i_h = tl.program_id(0)

        # he ← 0, M ← I (written once before the T loop).
        o_k0 = tl.arange(0, BK)
        o_v0 = tl.arange(0, BV)
        NK = (K + BK - 1) // BK
        NV = (V + BV - 1) // BV
        for i_k in range(0, NK):
            for i_v in range(0, NV):
                offs_k = i_k * BK + o_k0
                offs_v = i_v * BV + o_v0
                mask = (offs_k[:, None] < K) & (offs_v[None, :] < V)
                tl.store(
                    he_ptr + (i_h * K * V) + offs_k[:, None] * V + offs_v[None, :],
                    tl.zeros([BK, BV], dtype=tl.float32),
                    mask=mask,
                )
        for i_r in range(0, NK):
            for i_c in range(0, NK):
                offs_r = i_r * BK + o_k0
                offs_c = i_c * BK + o_k0
                mask = (offs_r[:, None] < K) & (offs_c[None, :] < K)
                eye = tl.where(
                    (offs_r[:, None] == offs_c[None, :]) & mask,
                    1.0,
                    0.0,
                ).to(tl.float32)
                tl.store(
                    M_ptr + (i_h * K * K) + offs_r[:, None] * K + offs_c[None, :],
                    eye,
                    mask=mask,
                )

        p_k = k_ptr + i_h * K
        p_v = v_ptr + i_h * V
        p_g = g_ptr + i_h
        p_beta = beta_ptr + i_h
        stride_t_k = H * K
        stride_t_v = H * V
        stride_t_h = H

        for _t in range(0, T):
            # Load full k/v vectors into UB (K=V=128 → ~1KB) — fine on Ascend.
            o_k = tl.arange(0, BK)  # reused; load k in BK tiles into accum below
            # Materialize k[K], v[V] via small tiles into flat buffers held as tiles.
            # For kᵀhe we stream he tiles; keep k as per-tile loads.

            b_g = tl.load(p_g).to(tl.float32)
            b_beta = tl.load(p_beta).to(tl.float32)
            b_eg = tl.exp(b_g)

            # ---- kᵀ he → [V], accumulated in BV tiles ----
            for i_v in range(0, NV):
                offs_v = i_v * BV + tl.arange(0, BV)
                mask_v = offs_v < V
                b_khe = tl.zeros([BV], dtype=tl.float32)
                for i_k in range(0, NK):
                    offs_k = i_k * BK + tl.arange(0, BK)
                    mask_k = offs_k < K
                    b_k = tl.load(p_k + offs_k, mask=mask_k, other=0).to(tl.float32)
                    b_he = tl.load(
                        he_ptr + (i_h * K * V) + offs_k[:, None] * V + offs_v[None, :],
                        mask=mask_k[:, None] & mask_v[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    b_khe += tl.sum(b_k[:, None] * b_he, 0)
                # ---- update he tiles for this V tile ----
                b_v = tl.load(p_v + offs_v, mask=mask_v, other=0).to(tl.float32)
                for i_k in range(0, NK):
                    offs_k = i_k * BK + tl.arange(0, BK)
                    mask_k = offs_k < K
                    b_k = tl.load(p_k + offs_k, mask=mask_k, other=0).to(tl.float32)
                    b_he = tl.load(
                        he_ptr + (i_h * K * V) + offs_k[:, None] * V + offs_v[None, :],
                        mask=mask_k[:, None] & mask_v[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    b_he = b_eg * (b_he - b_beta * b_k[:, None] * b_khe[None, :]) + (
                        b_beta * b_k[:, None] * b_v[None, :]
                    )
                    tl.store(
                        he_ptr + (i_h * K * V) + offs_k[:, None] * V + offs_v[None, :],
                        b_he,
                        mask=mask_k[:, None] & mask_v[None, :],
                    )

            # ---- kᵀ M → [K], then update M tiles ----
            for i_c in range(0, NK):
                offs_c = i_c * BK + tl.arange(0, BK)
                mask_c = offs_c < K
                b_kM = tl.zeros([BK], dtype=tl.float32)
                for i_r in range(0, NK):
                    offs_r = i_r * BK + tl.arange(0, BK)
                    mask_r = offs_r < K
                    b_k = tl.load(p_k + offs_r, mask=mask_r, other=0).to(tl.float32)
                    b_M = tl.load(
                        M_ptr + (i_h * K * K) + offs_r[:, None] * K + offs_c[None, :],
                        mask=mask_r[:, None] & mask_c[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    b_kM += tl.sum(b_k[:, None] * b_M, 0)
                for i_r in range(0, NK):
                    offs_r = i_r * BK + tl.arange(0, BK)
                    mask_r = offs_r < K
                    b_k = tl.load(p_k + offs_r, mask=mask_r, other=0).to(tl.float32)
                    b_M = tl.load(
                        M_ptr + (i_h * K * K) + offs_r[:, None] * K + offs_c[None, :],
                        mask=mask_r[:, None] & mask_c[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    b_M = b_eg * (b_M - b_beta * b_k[:, None] * b_kM[None, :])
                    tl.store(
                        M_ptr + (i_h * K * K) + offs_r[:, None] * K + offs_c[None, :],
                        b_M,
                        mask=mask_r[:, None] & mask_c[None, :],
                    )

            p_k += stride_t_k
            p_v += stride_t_v
            p_g += stride_t_h
            p_beta += stride_t_h


def ttx_local_affine_he_m(
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Fused LTR affine on one contiguous segment → fp32 ``he``, ``M``."""
    if not _TRITON_OK:
        raise RuntimeError("triton unavailable for TTX local-affine kernel")
    if not _npu_ready():
        raise RuntimeError("Ascend NPU unavailable for TTX local-affine kernel")
    if key.ndim != 3 or value.ndim != 3:
        raise ValueError(f"expected [T,H,D], got key={tuple(key.shape)} value={tuple(value.shape)}")

    t_len, num_heads, k_dim = int(key.shape[0]), int(key.shape[1]), int(key.shape[2])
    v_dim = int(value.shape[-1])
    if t_len == 0:
        he = torch.zeros(num_heads, k_dim, v_dim, device=key.device, dtype=torch.float32)
        M = torch.eye(k_dim, device=key.device, dtype=torch.float32).expand(num_heads, -1, -1).contiguous()
        return he, M

    k_c = key.contiguous()
    v_c = value.contiguous()
    g_c = g.contiguous()
    b_c = beta.contiguous()
    if k_c.device.type != "npu":
        raise RuntimeError(f"TTX affine requires npu tensors, got device={k_c.device}")

    he = torch.empty(num_heads, k_dim, v_dim, device=k_c.device, dtype=torch.float32)
    M = torch.empty(num_heads, k_dim, k_dim, device=k_c.device, dtype=torch.float32)
    # Ascend-friendly tile: 64×64 (matches GDR chunk_delta_h blockdim64 class).
    bk = 64 if k_dim > 32 else max(16, k_dim)
    bv = 64 if v_dim > 32 else max(16, v_dim)
    # Round tile up to pow2 ≥ dim chunk for Triton masks.
    def _pow2_ge(n: int) -> int:
        p = 16
        while p < n:
            p <<= 1
        return p

    bk = _pow2_ge(min(bk, k_dim) if k_dim <= 64 else 64)
    bv = _pow2_ge(min(bv, v_dim) if v_dim <= 64 else 64)

    _local_affine_summary_fwd_kernel[(num_heads,)](
        k_c,
        v_c,
        g_c,
        b_c,
        he,
        M,
        t_len,
        H=num_heads,
        K=k_dim,
        V=v_dim,
        BK=bk,
        BV=bv,
    )
    if not torch.isfinite(he).all() or not torch.isfinite(M).all():
        raise RuntimeError("TTX local-affine produced non-finite he/M")
    return he, M


def ttx_local_affine_summary(
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    cu_seqlens: Optional[Tensor] = None,
    use_qk_l2norm: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    """Public TTX entry: same signature as ``mojo_local_affine_summary`` → fp32 ``hm``."""
    from veomni.distributed.context_parallel.gdn_kcp import (
        _prepare_affine_inputs,
        pack_affine_hm,
    )

    k, v, gg, bb, ranges = _prepare_affine_inputs(
        key, value, g, beta, cu_seqlens=cu_seqlens, use_qk_l2norm=use_qk_l2norm, eps=eps
    )
    if not _npu_ready():
        raise RuntimeError("Ascend NPU unavailable for TTX local-affine kernel")
    device = torch.device("npu")
    compute_dtype = torch.bfloat16
    k_c = k.to(device=device, dtype=compute_dtype)
    v_c = v.to(device=device, dtype=compute_dtype)
    gg_c = gg.to(device=device, dtype=compute_dtype)
    bb_c = bb.to(device=device, dtype=compute_dtype)

    out = []
    for _b, start, end in ranges:
        he, M = ttx_local_affine_he_m(
            k_c[_b, start:end],
            v_c[_b, start:end],
            gg_c[_b, start:end],
            bb_c[_b, start:end],
        )
        he_fp32 = he.float()
        M_fp32 = M.float()
        if he_fp32.dtype != torch.float32 or M_fp32.dtype != torch.float32:
            raise RuntimeError("INV-7: TTX affine failed to export he/M as float32")
        out.append(pack_affine_hm(he_fp32, M_fp32))
    hm = torch.stack(out, dim=0)
    if hm.dtype != torch.float32:
        raise RuntimeError(f"INV-7: TTX hm must be float32, got {hm.dtype}")
    # Return on caller's device when inputs were CPU (host inject paths).
    if key.device.type != "npu":
        hm = hm.to(device=key.device)
    return hm


__all__ = ["ttx_local_affine_he_m", "ttx_local_affine_summary"]
