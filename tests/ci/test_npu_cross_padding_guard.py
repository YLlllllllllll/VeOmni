"""Check cross-PR malformed padding preflight on CPU/Gloo and native HCCL."""

import os
from datetime import timedelta

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec


def _padding_guard_worker(rank, backend, endpoint):
    import torch.distributed as dist

    if backend == "hccl":
        import torch_npu  # noqa: F401

        torch.npu.set_device(rank)
        device = f"npu:{rank}"
    else:
        device = "cpu"

    from veomni.utils import loss_utils
    from veomni.utils.device import get_device_type

    if backend == "hccl":
        assert get_device_type() == "npu"
        assert torch.npu.current_device() == rank

    dist.init_process_group(backend, store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        assert dist.get_backend() == backend
        for mode in ("sft_padding", "sft_key", "objective"):
            batch = {"labels": torch.tensor([1, 2])}
            if rank == 1 and mode in ("sft_padding", "objective"):
                batch["padding_flag"] = torch.tensor([True, False])
            if rank == 1 and mode == "sft_key":
                batch[3] = torch.tensor([1])
            if mode == "objective":
                with pytest.raises(ValueError, match="does not support padded micro-batches"):
                    loss_utils.reject_objective_padding([batch], "objective")
            else:
                for check_fetch in (False, True):
                    with pytest.raises(RuntimeError, match="malformed batch metadata"):
                        loss_utils.reject_distributed_padding([batch], check_fetch=check_fetch)
            completed = torch.ones(1, dtype=torch.int32, device=device)
            dist.all_reduce(completed)
            assert completed.item() == 2
        for check_fetch in (False, True):
            assert (
                loss_utils.reject_distributed_padding([{"labels": torch.tensor([1, 2])}], check_fetch=check_fetch)
                is False
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_cpu_padding_guard_fails_collectively():
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_padding_guard_worker, args=("gloo",), nprocs=2)


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="accelerator"))
def test_native_hccl_padding_guard_fails_collectively():
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        if os.environ.get("VEOMNI_REQUIRE_NPU_TESTS") == "1":
            pytest.fail("VEOMNI_REQUIRE_NPU_TESTS=1 requires torch_npu")
        pytest.skip("requires torch_npu")
    if not torch.npu.is_available() or torch.npu.device_count() < 2:
        if os.environ.get("VEOMNI_REQUIRE_NPU_TESTS") == "1":
            pytest.fail("VEOMNI_REQUIRE_NPU_TESTS=1 requires two native NPU devices")
        pytest.skip("requires two native Ascend devices")
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_padding_guard_worker, args=("hccl",), nprocs=2)
