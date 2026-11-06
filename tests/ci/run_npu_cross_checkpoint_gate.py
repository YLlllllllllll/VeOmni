"""Run both cross-PR checkpoint guards on real NPU; reject skipped tests."""

import argparse
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha", required=True, help="Exact 40-hex candidate commit")
    parser.add_argument("--junit-xml", type=Path, required=True, help="Persistent path for the raw pytest report")
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.expected_sha) is None:
        raise ValueError("--expected-sha must be a full lowercase commit SHA")
    os.environ["VEOMNI_REQUIRE_NPU_TESTS"] = "1"
    import pytest
    import torch
    import torch_npu  # noqa: F401

    import veomni

    root = Path(__file__).resolve().parents[2]
    if not Path(veomni.__file__).resolve().is_relative_to(root):
        raise RuntimeError("VeOmni import does not match the candidate source")
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != args.expected_sha:
        raise RuntimeError(f"Expected source {args.expected_sha}, got {actual}")
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"], text=True
    )
    if status:
        raise RuntimeError("The source checkout must have no tracked or untracked changes")
    if not torch.npu.is_available() or torch.npu.device_count() < 2:
        raise RuntimeError("The gate requires two native NPUs")
    args.junit_xml.parent.mkdir(parents=True, exist_ok=True)
    result = pytest.main(
        [str(root / "tests/ci/test_npu_cross_checkpoint_guard.py"), "-q", f"--junitxml={args.junit_xml}"]
    )
    cases = ET.parse(args.junit_xml).getroot().findall(".//testcase") if args.junit_xml.exists() else []
    names = {case.get("name") for case in cases}
    expected = {f"test_native_asymmetric_marker_policy_rejects_before_io[{op}]" for op in ("save", "load")}
    if (
        result != 0
        or len(cases) != 2
        or names != expected
        or any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped"))
    ):
        raise RuntimeError("Expected both native checkpoint guards to pass without skips")
    print("CROSS_CHECKPOINT_NPU_GATE_PASSED cases=2 skipped=0", flush=True)


if __name__ == "__main__":
    main()
