# Copyright (c) 2025 VeOmni Authors.
"""P1.5b hook: Ascend Mojo local affine summary for GDN KCP.

Contract (locked in #987 / GDN CP Implementation Contract):
- Golden = eager ``local_affine_summary_recurrent`` (bit-close / absdiff≈0).
- Scope = this op only (no AG / zigzag / prefix-merge).
- INV-7: output ``hm`` must be float32.

Until the TTX/Mojo kernel lands, importing ``mojo_local_affine_summary`` fails
closed — callers should use ``VEOMNI_GDN_KCP_AFFINE_IMPL=fused_torch`` (P1.5a).
"""

from __future__ import annotations

from typing import Optional

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
    """Mojo fused local pre-scan → ``hm[N,H,K,V+K]`` float32.

    Placeholder: real Ascend Mojo backend not yet wired. Do **not** silently
    fall back to eager (would hide missing kernel).
    """
    raise NotImplementedError(
        "gdn_kcp_affine Mojo kernel not implemented yet (P1.5b). "
        "Set VEOMNI_GDN_KCP_AFFINE_IMPL=fused_torch for the bit-close torch rewrite, "
        "or eager for the golden reference. "
        f"shapes key={tuple(key.shape)} value={tuple(value.shape)}"
    )


__all__ = ["mojo_local_affine_summary"]
