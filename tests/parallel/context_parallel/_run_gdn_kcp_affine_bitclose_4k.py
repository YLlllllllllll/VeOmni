#!/usr/bin/env python3
"""P1.5a: fused_torch vs eager local affine — 4K bit-close gate (host).

Acceptance (≠ CP-C lossy tol):
  max|fused - eager| ≈ 0 (fp32; allow tiny bmm vs einsum ulp noise).

Usage:
  python3 tests/parallel/context_parallel/_run_gdn_kcp_affine_bitclose_4k.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
CP = ROOT / "veomni" / "distributed" / "context_parallel"


def _load(name: str, path: Path, package: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = package
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


for name, path in [
    ("veomni", ROOT / "veomni"),
    ("veomni.distributed", ROOT / "veomni" / "distributed"),
    ("veomni.distributed.context_parallel", CP),
]:
    m = types.ModuleType(name)
    m.__path__ = [str(path)]
    sys.modules[name] = m

_load(
    "veomni.distributed.context_parallel.sharding",
    CP / "sharding.py",
    "veomni.distributed.context_parallel",
)
_load(
    "veomni.distributed.context_parallel.gdn_scan_cp",
    CP / "gdn_scan_cp.py",
    "veomni.distributed.context_parallel",
)
kcp = _load(
    "veomni.distributed.context_parallel.gdn_kcp",
    CP / "gdn_kcp.py",
    "veomni.distributed.context_parallel",
)


def _run_case(*, t: int, h: int, kd: int, vd: int, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    key = torch.randn(1, t, h, kd)
    value = torch.randn(1, t, h, vd)
    g = torch.randn(1, t, h) * 0.05
    beta = torch.sigmoid(torch.randn(1, t, h))

    t0 = time.perf_counter()
    hm_e = kcp.local_affine_summary_recurrent(key, value, g, beta, use_qk_l2norm=True)
    t_eager = time.perf_counter() - t0

    t0 = time.perf_counter()
    hm_f = kcp.local_affine_summary_fused_torch(key, value, g, beta, use_qk_l2norm=True)
    t_fused = time.perf_counter() - t0

    assert hm_e.dtype == torch.float32 and hm_f.dtype == torch.float32
    absdiff = (hm_f - hm_e).abs()
    return {
        "T": t,
        "H": h,
        "K": kd,
        "V": vd,
        "seed": seed,
        "max_absdiff": float(absdiff.max().item()),
        "mean_absdiff": float(absdiff.mean().item()),
        "eager_s": round(t_eager, 4),
        "fused_torch_s": round(t_fused, 4),
        "speedup": round(t_eager / t_fused, 3) if t_fused > 0 else None,
        "dtype": "float32",
    }


def main() -> None:
    # Small sanity + Qwen-ish head dims at 4K (host may be slow; keep H modest).
    cases = [
        _run_case(t=64, h=2, kd=8, vd=4, seed=0),
        _run_case(t=4096, h=4, kd=128, vd=128, seed=42),
    ]
    # Bit-close gate: allow tiny bmm/einsum ulp; fail if structural drift.
    atol = 1e-6
    ok = all(c["max_absdiff"] <= atol for c in cases)
    out = {
        "schema": "ai4se.kcp_affine_bitclose/v1",
        "candidate": "fused_torch",
        "golden": "eager",
        "atol": atol,
        "cases": cases,
        "green": ok,
        "note": (
            "P1.5a fused_torch is same LTR fp32 recurrence (bmm). "
            "Ascend Mojo fused loop is P1.5b; GDR mojo cannot be reused (rejects fp32)."
        ),
    }
    print("BITCLOSE_JSON " + json.dumps(out, sort_keys=True))
    if not ok:
        print("BITCLOSE_RED")
        raise SystemExit(2)
    print("BITCLOSE_OK")


if __name__ == "__main__":
    main()
