# Copyright (c) 2025 VeOmni Authors.
"""P1.5b: KCP local affine — bf16 compute → **fp32** ``hm`` (INV-7).

Acceptance vs eager (≠ P1.5a bit-close):
  ACCEPTABLE_LOSSY — slope of **per-token** absdiff (max/T, mean/T) ≈ 0
  across growing prefixes (non-accumulating rate), finite / no NaN.
  Absolute envelope may grow with T (bf16); magnitude may exceed fused_torch.

Scope: **only** this pre-scan. Do not touch AG / zigzag / prefix-merge.

INV-7 boundary (must hold even if Ascend Mojo lands):
  kernel-internal bf16 arithmetic is allowed;
  ``{he,M}`` / packed ``hm`` returned to the hook **must be float32**
  before all-gather (serial v6 bf16×fp32 byte-misalign NaN class).

Current body: torch bf16 LTR recurrence as the **Mojo numerical contract
reference** (same dtype boundary Ascend TTX must honor). Swap the compute
body for a fused Ascend Mojo/TTX kernel without changing the fp32 return
contract. Existing ``mojo_chunk_gated_delta_rule`` must **not** be reused
(rejects float32).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


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
    """Bf16-internal local pre-scan → ``hm[N,H,K,V+K]`` **float32**.

    Returns fp32 ``hm`` always (INV-7). Caller must still run
    ``ensure_affine_hm_fp32`` before AG as a fail-closed belt.
    """
    # Late import avoids circular init with gdn_kcp dispatcher.
    from veomni.distributed.context_parallel.gdn_kcp import (
        _eye_m,
        _prepare_affine_inputs,
        pack_affine_hm,
    )

    k, v, gg, bb, ranges = _prepare_affine_inputs(
        key, value, g, beta, cu_seqlens=cu_seqlens, use_qk_l2norm=use_qk_l2norm, eps=eps
    )
    # Mojo-contract compute dtype: bf16 (matches planned Ascend fused loop).
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
        # INV-7 hard boundary: cast before pack / AG.
        he_fp32 = he.float()
        M_fp32 = M.float()
        if he_fp32.dtype != torch.float32 or M_fp32.dtype != torch.float32:
            raise RuntimeError("INV-7: mojo affine failed to cast he/M to float32")
        if not torch.isfinite(he_fp32).all() or not torch.isfinite(M_fp32).all():
            raise RuntimeError("mojo affine: non-finite he/M after bf16→fp32 cast")
        out.append(pack_affine_hm(he_fp32, M_fp32))
    hm = torch.stack(out, dim=0)
    if hm.dtype != torch.float32:
        raise RuntimeError(f"INV-7: mojo affine hm must be float32, got {hm.dtype}")
    return hm


__all__ = ["mojo_local_affine_summary"]
