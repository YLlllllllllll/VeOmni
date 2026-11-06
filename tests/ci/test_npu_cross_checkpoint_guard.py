"""Native HCCL coverage for cross-PR checkpoint policy coordination."""

import os

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec


def _native_worker(rank, checkpoint, operation, endpoint):
    import torch_npu  # noqa: F401

    from tests.checkpoints.test_multi_optimizer_proven_resume import _asymmetric_marker_policy_worker

    _asymmetric_marker_policy_worker(rank, checkpoint, operation, "hccl", endpoint)


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="accelerator"))
@pytest.mark.parametrize("operation", ["save", "load"])
def test_native_asymmetric_marker_policy_rejects_before_io(tmp_path, operation):
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        if os.environ.get("VEOMNI_REQUIRE_NPU_TESTS") == "1":
            pytest.fail("VEOMNI_REQUIRE_NPU_TESTS=1 requires torch_npu")
        pytest.skip("requires torch_npu")
    from tests.tools.launch_utils import spawn_with_rendezvous
    from veomni.utils.device import IS_NPU_AVAILABLE

    if not IS_NPU_AVAILABLE or torch.npu.device_count() < 2:
        if os.environ.get("VEOMNI_REQUIRE_NPU_TESTS") == "1":
            pytest.fail("VEOMNI_REQUIRE_NPU_TESTS=1 requires two native NPU devices")
        pytest.skip("requires two native Ascend devices")
    spawn_with_rendezvous(
        _native_worker,
        args=(str(tmp_path / "checkpoint"), operation),
        nprocs=2,
    )
