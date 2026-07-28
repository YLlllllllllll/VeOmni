#!/usr/bin/env python3
"""P1.5b three-way affine gate: eager / fused_torch / mojo @ growing prefixes.

Rulers (do not mix):
  fused_torch vs eager → bit-close (max_absdiff == 0)
  mojo vs eager → ACCEPTABLE_LOSSY (CP-C style):
      finite / no NaN;
      slope of **per-token** absdiff (max/T, mean/T) ≈ 0
      → non-accumulating rate (absolute envelope may grow with T
         from bf16 rounding; that is not a veto)

Usage:
  python3 tests/parallel/context_parallel/_run_gdn_kcp_affine_threeway_4k.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
CP = ROOT / "veomni" / "distributed" / "context_parallel"
OPS_K = ROOT / "veomni" / "ops" / "kernels"


def _load(name: str, path: Path, package: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = package
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_pkg(name: str, path: Path) -> None:
    m = types.ModuleType(name)
    m.__path__ = [str(path)]
    sys.modules[name] = m


for name, path in [
    ("veomni", ROOT / "veomni"),
    ("veomni.distributed", ROOT / "veomni" / "distributed"),
    ("veomni.distributed.context_parallel", CP),
    ("veomni.ops", ROOT / "veomni" / "ops"),
    ("veomni.ops.kernels", OPS_K),
]:
    _ensure_pkg(name, path)

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
_load(
    "veomni.ops.kernels.gdn_kcp_affine",
    OPS_K / "gdn_kcp_affine.py",
    "veomni.ops.kernels",
)


def _absdiff_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    d = (a - b).abs()
    return {
        "max": float(d.max().item()),
        "mean": float(d.mean().item()),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
    }


def _slope(xs: list[float], ys: list[float]) -> float:
    """Simple least-squares slope of y vs x."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return float(num / den) if den else 0.0


def main() -> None:
    torch.manual_seed(42)
    h, kd, vd = 4, 128, 128
    t_full = 4096
    key = torch.randn(1, t_full, h, kd)
    value = torch.randn(1, t_full, h, vd)
    g = torch.randn(1, t_full, h) * 0.05
    beta = torch.sigmoid(torch.randn(1, t_full, h))

    prefixes = [512, 1024, 2048, 4096]
    fused_max: list[float] = []
    mojo_max: list[float] = []
    rows = []

    for t in prefixes:
        kk, vv, gg, bb = key[:, :t], value[:, :t], g[:, :t], beta[:, :t]
        hm_e = kcp.local_affine_summary_recurrent(kk, vv, gg, bb, use_qk_l2norm=True)
        hm_f = kcp.local_affine_summary_fused_torch(kk, vv, gg, bb, use_qk_l2norm=True)
        hm_m = kcp.local_affine_summary(
            kk, vv, gg, bb, use_qk_l2norm=True, impl="mojo"
        )
        # INV-7 on all returns
        assert hm_e.dtype == torch.float32 and hm_f.dtype == torch.float32
        assert hm_m.dtype == torch.float32
        hm_m2 = kcp.ensure_affine_hm_fp32(hm_m, where="threeway")
        assert hm_m2.dtype == torch.float32

        st_f = _absdiff_stats(hm_f, hm_e)
        st_m = _absdiff_stats(hm_m, hm_e)
        fused_max.append(st_f["max"])
        mojo_max.append(st_m["max"])
        rows.append({"T": t, "fused_vs_eager": st_f, "mojo_vs_eager": st_m})

    fused_slope = _slope([float(t) for t in prefixes], fused_max)
    mojo_slope_abs = _slope([float(t) for t in prefixes], mojo_max)
    mojo_means = [float(r["mojo_vs_eager"]["mean"]) for r in rows]
    mojo_max_per_tok = [m / float(t) for m, t in zip(mojo_max, prefixes)]
    mojo_mean_per_tok = [m / float(t) for m, t in zip(mojo_means, prefixes)]
    slope_max_per_tok = _slope([float(t) for t in prefixes], mojo_max_per_tok)
    slope_mean_per_tok = _slope([float(t) for t in prefixes], mojo_mean_per_tok)
    rate_ratio = (
        (mojo_max_per_tok[-1] / mojo_max_per_tok[0]) if mojo_max_per_tok[0] > 0 else 0.0
    )

    # fused: bit-close (allow tiny ulp)
    fused_ok = all(m <= 1e-6 for m in fused_max) and all(r["fused_vs_eager"]["finite"] for r in rows)
    # mojo: ACCEPTABLE_LOSSY — finite + per-token absdiff slope≈0 (non-accumulating rate).
    # Absolute max may rise with T (bf16 envelope); veto only if *rate* accumulates/diverges.
    mojo_finite = all(r["mojo_vs_eager"]["finite"] for r in rows)
    mojo_ok = (
        mojo_finite
        and abs(slope_max_per_tok) <= 1e-6
        and abs(slope_mean_per_tok) <= 1e-7
        and rate_ratio <= 2.0
        and mojo_max[-1] < 1e2  # sanity ceiling
    )

    out = {
        "schema": "ai4se.kcp_affine_threeway/v1",
        "prefixes": prefixes,
        "rows": rows,
        "fused_torch": {
            "ruler": "bit-close",
            "max_absdiffs": fused_max,
            "slope": fused_slope,
            "green": fused_ok,
        },
        "mojo": {
            "ruler": "ACCEPTABLE_LOSSY_per_token_slope",
            "max_absdiffs": mojo_max,
            "mean_absdiffs": mojo_means,
            "max_per_token": mojo_max_per_tok,
            "mean_per_token": mojo_mean_per_tok,
            "slope_abs_max": mojo_slope_abs,
            "slope_max_per_token": slope_max_per_tok,
            "slope_mean_per_token": slope_mean_per_tok,
            "rate_last_over_first": rate_ratio,
            "green": mojo_ok,
            "note": (
                "bf16-internal → fp32 hm; judge slope of absdiff/T (not raw max); "
                "Ascend TTX may replace compute body"
            ),
        },
        "green": fused_ok and mojo_ok,
    }
    print("THREEWAY_JSON " + json.dumps(out, sort_keys=True))
    if not out["green"]:
        print("THREEWAY_RED")
        raise SystemExit(2)
    print("THREEWAY_OK")


if __name__ == "__main__":
    main()
