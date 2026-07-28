#!/usr/bin/env python3
"""P1.5b-kernel single-op TTX affine microbench (no distributed / no B).

Measures wall time of local-affine bodies vs sequence length on one NPU:
  T ∈ {4096, 8192, 16384, 32768, 65536}, H=4, K=V=128
  bodies: ttx / fused_torch / eager(optional short T only)

Goal: confirm HBM 64×64 TTX is O(T) with sane ms/token slope before
distributed throughput B (CP2 vs CP8 + serial 0.70× anchor).

Usage (on Ascend NPU host / via 90e5 inject):
  python3 _bench_gdn_kcp_affine_ttx_op.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import torch

_here = Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else Path(".")
if not (ROOT / "veomni" / "distributed" / "context_parallel" / "gdn_kcp.py").is_file():
    for cand in [
        Path(os.environ.get("VEOMNI_ROOT", "")),
        Path("/opt/tiger/modelchef/submodules/Open-VeOmni"),
        Path("/home/tiger/Open-VeOmni-github"),
        Path("/home/tiger/Open-VeOmni-ring-pr969-b4-runtime"),
    ]:
        if cand.is_dir() and (cand / "veomni" / "distributed" / "context_parallel" / "gdn_kcp.py").is_file():
            ROOT = cand
            break
CP = ROOT / "veomni" / "distributed" / "context_parallel"
OPS_K = ROOT / "veomni" / "ops" / "kernels"
if not (CP / "gdn_kcp.py").is_file():
    raise SystemExit(f"cannot locate gdn_kcp.py under ROOT={ROOT}")


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

_load("veomni.distributed.context_parallel.sharding", CP / "sharding.py", "veomni.distributed.context_parallel")
_load("veomni.distributed.context_parallel.gdn_scan_cp", CP / "gdn_scan_cp.py", "veomni.distributed.context_parallel")
kcp = _load("veomni.distributed.context_parallel.gdn_kcp", CP / "gdn_kcp.py", "veomni.distributed.context_parallel")
affine = _load("veomni.ops.kernels.gdn_kcp_affine", OPS_K / "gdn_kcp_affine.py", "veomni.ops.kernels")
ttx_mod = _load("veomni.ops.kernels.gdn_kcp_affine_ttx", OPS_K / "gdn_kcp_affine_ttx.py", "veomni.ops.kernels")


def _sync(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _bench_once(fn, warmup: int, reps: int, device: torch.device) -> dict:
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    _sync(device)
    elapsed = time.perf_counter() - t0
    return {"reps": reps, "total_s": elapsed, "mean_s": elapsed / reps}


def main() -> None:
    if not (getattr(torch, "npu", None) and torch.npu.is_available()):
        raise SystemExit("NPU required for TTX microbench")
    device = torch.device("npu")
    torch.manual_seed(0)
    h, kd, vd = 4, 128, 128
    lengths = [4096, 8192, 16384, 32768, 65536]
    rows = []

    for t in lengths:
        key = torch.randn(1, t, h, kd, device=device, dtype=torch.bfloat16)
        value = torch.randn(1, t, h, vd, device=device, dtype=torch.bfloat16)
        g = (torch.randn(1, t, h, device=device, dtype=torch.bfloat16) * 0.05)
        beta = torch.sigmoid(torch.randn(1, t, h, device=device, dtype=torch.float32)).to(torch.bfloat16)

        # Warm compile / first-touch separately
        hm_ttx, body = affine.mojo_local_affine_summary_with_meta(key, value, g, beta, use_qk_l2norm=True)
        assert body == "ttx", f"expected ttx body, got {body}"
        assert hm_ttx.dtype == torch.float32
        _sync(device)

        # Scale reps: keep wall ~few seconds per point
        reps = 8 if t <= 8192 else (4 if t <= 32768 else 2)
        warmup = 2

        def run_ttx():
            return affine.mojo_local_affine_summary_with_meta(key, value, g, beta, use_qk_l2norm=True)

        def run_fused():
            return kcp.local_affine_summary_fused_torch(key.float(), value.float(), g.float(), beta.float(), use_qk_l2norm=True)

        st_ttx = _bench_once(run_ttx, warmup, reps, device)
        st_fused = _bench_once(run_fused, warmup, reps, device)
        st_eager = None
        if t <= 8192:  # eager too slow for 32K+
            def run_eager():
                return kcp.local_affine_summary_recurrent(key.float(), value.float(), g.float(), beta.float(), use_qk_l2norm=True)

            st_eager = _bench_once(run_eager, 1, max(1, reps // 2), device)

        row = {
            "T": t,
            "H": h,
            "K": kd,
            "V": vd,
            "ttx_body": body,
            "ttx": {
                **st_ttx,
                "ms": st_ttx["mean_s"] * 1e3,
                "us_per_token": st_ttx["mean_s"] * 1e6 / t,
            },
            "fused_torch": {
                **st_fused,
                "ms": st_fused["mean_s"] * 1e3,
                "us_per_token": st_fused["mean_s"] * 1e6 / t,
            },
            "speedup_vs_fused": st_fused["mean_s"] / st_ttx["mean_s"] if st_ttx["mean_s"] > 0 else None,
        }
        if st_eager is not None:
            row["eager"] = {
                **st_eager,
                "ms": st_eager["mean_s"] * 1e3,
                "us_per_token": st_eager["mean_s"] * 1e6 / t,
            }
            row["speedup_vs_eager"] = st_eager["mean_s"] / st_ttx["mean_s"] if st_ttx["mean_s"] > 0 else None
        rows.append(row)
        print(
            f"BENCH_ROW T={t} ttx_ms={row['ttx']['ms']:.3f} "
            f"fused_ms={row['fused_torch']['ms']:.3f} "
            f"ttx_us/tok={row['ttx']['us_per_token']:.3f} "
            f"fused_us/tok={row['fused_torch']['us_per_token']:.3f} "
            f"x_fused={row['speedup_vs_fused']:.2f}"
            + (f" x_eager={row['speedup_vs_eager']:.2f}" if "speedup_vs_eager" in row else "")
        )

    ts = [float(r["T"]) for r in rows]
    ttx_ms = [float(r["ttx"]["ms"]) for r in rows]
    # least-squares slope ms vs T
    n = len(ts)
    mx, my = sum(ts) / n, sum(ttx_ms) / n
    num = sum((x - mx) * (y - my) for x, y in zip(ts, ttx_ms))
    den = sum((x - mx) ** 2 for x in ts)
    slope = num / den if den else 0.0
    # us/token stability
    us = [float(r["ttx"]["us_per_token"]) for r in rows]
    us_ratio = (max(us) / min(us)) if min(us) > 0 else None

    out = {
        "schema": "ai4se.kcp_affine_ttx_op_bench/v1",
        "device": str(device),
        "geometry": {"H": h, "K": kd, "V": vd},
        "rows": rows,
        "ttx_ms_vs_T_slope": slope,
        "ttx_us_per_token": us,
        "ttx_us_per_token_max_over_min": us_ratio,
        "note": (
            "O(T) expected: us/token roughly flat; "
            "HBM 64x64 tile should not explode us/token with T. "
            "Not a distributed B result."
        ),
    }
    print("BENCH_JSON " + json.dumps(out, sort_keys=True))
    # Soft sanity: us/token should not grow >3× across the sweep (tile thrash signal)
    if us_ratio is not None and us_ratio > 3.0:
        print("BENCH_WARN us_per_token_spread")
        raise SystemExit(2)
    print("BENCH_OK")


if __name__ == "__main__":
    main()
