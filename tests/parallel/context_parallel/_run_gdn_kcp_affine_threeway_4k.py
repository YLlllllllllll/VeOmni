#!/usr/bin/env python3
"""P1.5b-kernel three-way affine gate: eager / fused_torch / mojo-TTX.

Incremental order (contract):
  1. fused_torch vs eager → must stay max=0 (boundary probe)
  2. mojo-TTX vs eager → per-token slope≈0 (ACCEPTABLE_LOSSY)
  3. mojo body must be ``ttx`` — fused_torch_fallback is NOT a P1.5b-kernel green

Usage:
  python3 tests/parallel/context_parallel/_run_gdn_kcp_affine_threeway_4k.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) > 3 else Path(".")
# When injected to /tmp on NPU pods, locate Open-VeOmni via PYTHONPATH / common install roots.
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
affine = _load(
    "veomni.ops.kernels.gdn_kcp_affine",
    OPS_K / "gdn_kcp_affine.py",
    "veomni.ops.kernels",
)
# Register TTX module path for import inside affine.
_load(
    "veomni.ops.kernels.gdn_kcp_affine_ttx",
    OPS_K / "gdn_kcp_affine_ttx.py",
    "veomni.ops.kernels",
)


def _absdiff_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    d = (a - b).abs()
    return {
        "max": float(d.max().item()),
        "mean": float(d.mean().item()),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
    }


def _slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return float(num / den) if den else 0.0


def _pick_device() -> torch.device:
    if getattr(torch, "npu", None) is not None and torch.npu.is_available():
        return torch.device("npu")
    return torch.device("cpu")


def main() -> None:
    # P1.5b-kernel gate must not force torch-bf16 contract.
    os.environ.pop("VEOMNI_GDN_KCP_AFFINE_TORCH_BF16", None)

    torch.manual_seed(42)
    device = _pick_device()
    h, kd, vd = 4, 128, 128
    t_full = 4096
    key = torch.randn(1, t_full, h, kd, device=device)
    value = torch.randn(1, t_full, h, vd, device=device)
    g = torch.randn(1, t_full, h, device=device) * 0.05
    beta = torch.sigmoid(torch.randn(1, t_full, h, device=device))

    prefixes = [512, 1024, 2048, 4096]
    fused_max: list[float] = []
    mojo_max: list[float] = []
    mojo_means: list[float] = []
    rows = []
    bodies: list[str] = []

    for t in prefixes:
        kk, vv, gg, bb = key[:, :t], value[:, :t], g[:, :t], beta[:, :t]
        hm_e = kcp.local_affine_summary_recurrent(kk, vv, gg, bb, use_qk_l2norm=True)
        hm_f = kcp.local_affine_summary_fused_torch(kk, vv, gg, bb, use_qk_l2norm=True)
        hm_m, body = affine.mojo_local_affine_summary_with_meta(
            kk, vv, gg, bb, use_qk_l2norm=True
        )
        bodies.append(body)

        assert hm_e.dtype == torch.float32 and hm_f.dtype == torch.float32
        assert hm_m.dtype == torch.float32
        hm_m2 = kcp.ensure_affine_hm_fp32(hm_m, where="threeway")
        assert hm_m2.dtype == torch.float32

        st_f = _absdiff_stats(hm_f, hm_e)
        st_m = _absdiff_stats(hm_m, hm_e)
        fused_max.append(st_f["max"])
        mojo_max.append(st_m["max"])
        mojo_means.append(st_m["mean"])
        rows.append({"T": t, "fused_vs_eager": st_f, "mojo_vs_eager": st_m, "mojo_body": body})

    fused_slope = _slope([float(t) for t in prefixes], fused_max)
    mojo_max_per_tok = [m / float(t) for m, t in zip(mojo_max, prefixes)]
    mojo_mean_per_tok = [m / float(t) for m, t in zip(mojo_means, prefixes)]
    slope_max_per_tok = _slope([float(t) for t in prefixes], mojo_max_per_tok)
    slope_mean_per_tok = _slope([float(t) for t in prefixes], mojo_mean_per_tok)
    rate_ratio = (
        (mojo_max_per_tok[-1] / mojo_max_per_tok[0]) if mojo_max_per_tok[0] > 0 else 0.0
    )

    # Step 1: fused boundary probe
    fused_ok = all(m <= 1e-6 for m in fused_max) and all(r["fused_vs_eager"]["finite"] for r in rows)

    # Step 2: TTX body required for kernel milestone
    body_set = sorted(set(bodies))
    ttx_body_ok = body_set == ["ttx"]

    # Step 3: mojo ACCEPTABLE_LOSSY per-token slope
    mojo_finite = all(r["mojo_vs_eager"]["finite"] for r in rows)
    # If body fell back to fused_torch, absdiff is ~0 — that is NOT TTX evidence.
    mojo_slope_ok = (
        mojo_finite
        and abs(slope_max_per_tok) <= 1e-6
        and abs(slope_mean_per_tok) <= 1e-7
        and rate_ratio <= 2.0
        and mojo_max[-1] < 1e2
    )
    # When TTX is live, absolute envelope may be >0; when wrongly fused fallback, max≈0.
    # Kernel green requires TTX body + slope gate (slope≈0 holds for both; body tag separates).
    mojo_ok = ttx_body_ok and mojo_slope_ok

    out = {
        "schema": "ai4se.kcp_affine_threeway/v2_ttx",
        "device": str(device),
        "prefixes": prefixes,
        "rows": rows,
        "fused_torch": {
            "ruler": "bit-close_boundary_probe",
            "max_absdiffs": fused_max,
            "slope": fused_slope,
            "green": fused_ok,
            "note": "must stay max=0; drop ⇒ boundary touched, rollback (do not relax)",
        },
        "mojo": {
            "ruler": "ACCEPTABLE_LOSSY_per_token_slope",
            "bodies": bodies,
            "body_set": body_set,
            "ttx_required": True,
            "ttx_body_ok": ttx_body_ok,
            "max_absdiffs": mojo_max,
            "mean_absdiffs": mojo_means,
            "max_per_token": mojo_max_per_tok,
            "mean_per_token": mojo_mean_per_tok,
            "slope_max_per_token": slope_max_per_tok,
            "slope_mean_per_token": slope_mean_per_tok,
            "rate_last_over_first": rate_ratio,
            "slope_ok": mojo_slope_ok,
            "green": mojo_ok,
            "note": "veto: slope flip/diverge or NaN/Inf; ~1.6 abs envelope OK; fallback≠green",
        },
        "green": fused_ok and mojo_ok,
    }
    print("THREEWAY_JSON " + json.dumps(out, sort_keys=True))
    if not fused_ok:
        print("THREEWAY_RED fused_torch_boundary")
        raise SystemExit(3)
    if not ttx_body_ok:
        print("THREEWAY_INCOMPLETE mojo_body=" + ",".join(body_set))
        raise SystemExit(4)
    if not mojo_ok:
        print("THREEWAY_RED mojo_ttx_slope")
        raise SystemExit(2)
    print("THREEWAY_OK")


if __name__ == "__main__":
    main()
