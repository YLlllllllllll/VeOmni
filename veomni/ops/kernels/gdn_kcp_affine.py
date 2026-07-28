# Copyright (c) 2025 VeOmni Authors.
"""P1.5b-kernel hook: KCP local affine — prefer Ascend TTX fused body.

Acceptance vs eager (≠ P1.5a bit-close):
  ACCEPTABLE_LOSSY — slope of **per-token** absdiff (max/T, mean/T) ≈ 0
  across growing prefixes (non-accumulating rate), finite / no NaN.
  Absolute envelope may grow with T (bf16); magnitude may exceed fused_torch.

Scope: **only** this pre-scan. Do not touch AG / zigzag / prefix-merge.

INV-7 boundary:
  kernel-internal bf16/fp32 mix is allowed;
  ``{he,M}`` / packed ``hm`` returned to the hook **must be float32**
  before all-gather (serial v6 bf16×fp32 byte-misalign NaN class).

Body selection (``VEOMNI_GDN_KCP_AFFINE_IMPL=mojo``):
  1. Ascend TTX fused loop (``gdn_kcp_affine_ttx``) — **new op**, not GDR
  2. On TTX miss/error → **fail-closed to ``fused_torch``** (fp32, bit-close)
     — never silent eager fallback

Optional: ``VEOMNI_GDN_KCP_AFFINE_TORCH_BF16=1`` forces the host bf16
contract-reference body (P1.5b numerical checkpoint ``07eb18a0``).
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor


def torch_bf16_contract_local_affine_summary(
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    cu_seqlens: Optional[Tensor] = None,
    use_qk_l2norm: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    """P1.5b host bf16 contract reference (checkpoint ``07eb18a0``). Not TTX."""
    from veomni.distributed.context_parallel.gdn_kcp import (
        _eye_m,
        _prepare_affine_inputs,
        pack_affine_hm,
    )

    k, v, gg, bb, ranges = _prepare_affine_inputs(
        key, value, g, beta, cu_seqlens=cu_seqlens, use_qk_l2norm=use_qk_l2norm, eps=eps
    )
    compute_dtype = torch.bfloat16
    k_c = k.to(compute_dtype)
    v_c = v.to(compute_dtype)
    gg_c = gg.to(compute_dtype)
    bb_c = bb.to(compute_dtype)

    num_heads = int(k.shape[2])
    k_dim = int(k.shape[3])
    v_dim = int(v.shape[-1])
    eye_fp32 = _eye_m(num_heads, k_dim, device=k.device)
    eye = eye_fp32.to(compute_dtype)
    out = []
    for _b, start, end in ranges:
        kt_all = k_c[_b, start:end].contiguous()
        vt_all = v_c[_b, start:end].contiguous()
        eg_all = gg_c[_b, start:end].exp().contiguous()
        bt_all = bb_c[_b, start:end].contiguous()
        he = torch.zeros(num_heads, k_dim, v_dim, device=k.device, dtype=compute_dtype)
        M = eye.clone()
        for t in range(int(kt_all.shape[0])):
            eg = eg_all[t]
            kt = kt_all[t]
            vt = vt_all[t]
            bt = bt_all[t]
            outer = kt.unsqueeze(-1) * kt.unsqueeze(-2)
            M_step = eg[:, None, None] * (eye - bt[:, None, None] * outer)
            he_step = (bt[:, None] * kt).unsqueeze(-1) * vt.unsqueeze(-2)
            he = torch.bmm(M_step, he) + he_step
            M = torch.bmm(M_step, M)
        he_fp32 = he.float()
        M_fp32 = M.float()
        if he_fp32.dtype != torch.float32 or M_fp32.dtype != torch.float32:
            raise RuntimeError("INV-7: torch-bf16 contract failed to cast he/M to float32")
        if not torch.isfinite(he_fp32).all() or not torch.isfinite(M_fp32).all():
            raise RuntimeError("torch-bf16 contract: non-finite he/M after cast")
        out.append(pack_affine_hm(he_fp32, M_fp32))
    hm = torch.stack(out, dim=0)
    if hm.dtype != torch.float32:
        raise RuntimeError(f"INV-7: torch-bf16 hm must be float32, got {hm.dtype}")
    return hm


def mojo_local_affine_summary(
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    cu_seqlens: Optional[Tensor] = None,
    use_qk_l2norm: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    """TTX-first local pre-scan → ``hm[N,H,K,V+K]`` **float32**.

    Returns fp32 ``hm`` always (INV-7). Caller must still run
    ``ensure_affine_hm_fp32`` before AG as a fail-closed belt.
    """
    force_bf16 = os.environ.get("VEOMNI_GDN_KCP_AFFINE_TORCH_BF16", "").strip() in (
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    )
    if force_bf16:
        return torch_bf16_contract_local_affine_summary(
            key, value, g, beta, cu_seqlens=cu_seqlens, use_qk_l2norm=use_qk_l2norm, eps=eps
        )

    try:
        from veomni.ops.kernels.gdn_kcp_affine_ttx import ttx_local_affine_summary

        return ttx_local_affine_summary(
            key, value, g, beta, cu_seqlens=cu_seqlens, use_qk_l2norm=use_qk_l2norm, eps=eps
        )
    except Exception as exc:
        # Fail-closed: fused_torch (fp32, bit-close) — never silent eager.
        warnings.warn(
            f"TTX local-affine unavailable ({type(exc).__name__}: {exc}); "
            "falling back to fused_torch (not eager).",
            RuntimeWarning,
            stacklevel=2,
        )
        from veomni.distributed.context_parallel.gdn_kcp import local_affine_summary_fused_torch

        return local_affine_summary_fused_torch(
            key, value, g, beta, cu_seqlens=cu_seqlens, use_qk_l2norm=use_qk_l2norm, eps=eps
        )


def resolve_mojo_affine_body() -> str:
    """Which compute body ``mojo_local_affine_summary`` would pick right now."""
    if os.environ.get("VEOMNI_GDN_KCP_AFFINE_TORCH_BF16", "").strip() in (
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    ):
        return "torch_bf16_contract"
    try:
        from veomni.ops.kernels.gdn_kcp_affine_ttx import _TRITON_OK, _npu_ready

        if _TRITON_OK and _npu_ready():
            return "ttx"
    except Exception:
        pass
    return "fused_torch_fallback"


def mojo_local_affine_summary_with_meta(
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    cu_seqlens: Optional[Tensor] = None,
    use_qk_l2norm: bool = True,
    eps: float = 1e-6,
) -> Tuple[Tensor, str]:
    """Like ``mojo_local_affine_summary`` but also returns the body tag used."""
    body = resolve_mojo_affine_body()
    if body == "torch_bf16_contract":
        hm = torch_bf16_contract_local_affine_summary(
            key, value, g, beta, cu_seqlens=cu_seqlens, use_qk_l2norm=use_qk_l2norm, eps=eps
        )
        return hm, body
    if body == "ttx":
        from veomni.ops.kernels.gdn_kcp_affine_ttx import ttx_local_affine_summary

        try:
            hm = ttx_local_affine_summary(
                key, value, g, beta, cu_seqlens=cu_seqlens, use_qk_l2norm=use_qk_l2norm, eps=eps
            )
            return hm, "ttx"
        except Exception as exc:
            warnings.warn(
                f"TTX local-affine failed at launch ({type(exc).__name__}: {exc}); "
                "falling back to fused_torch (not eager).",
                RuntimeWarning,
                stacklevel=2,
            )
            body = "fused_torch_fallback"
    from veomni.distributed.context_parallel.gdn_kcp import local_affine_summary_fused_torch

    hm = local_affine_summary_fused_torch(
        key, value, g, beta, cu_seqlens=cu_seqlens, use_qk_l2norm=use_qk_l2norm, eps=eps
    )
    return hm, body


__all__ = [
    "mojo_local_affine_summary",
    "mojo_local_affine_summary_with_meta",
    "resolve_mojo_affine_body",
    "torch_bf16_contract_local_affine_summary",
]
