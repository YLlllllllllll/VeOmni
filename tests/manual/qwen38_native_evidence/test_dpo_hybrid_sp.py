"""Validation-only synthetic DPO pair mean and channel-loss parity.

This does not validate real loader provenance, Qwen3.8 QSA/GDN, FSDP/EP,
mixed precision, optimizer offload, or checkpoint cold recovery.
"""
import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_device_count
from tests.tools.launch_utils import spawn_with_rendezvous
from tests.trainer.test_dpo_pair_mean import _channel_loss_worker
from veomni.utils.device import IS_NPU_AVAILABLE


def _checked_worker(rank, world_size, options, native_npu, endpoint):
    import torch.distributed as dist

    import veomni.distributed.parallel_state as parallel

    original = parallel._init_parallel_state

    def checked_init(*args, **kwargs):
        result = original(*args, **kwargs)
        state = parallel.get_parallel_state()
        assert state.dp_size == 2 and state.ulysses_size == 2 and state.cp_size == 2 and state.sp_size == 4
        assert dist.get_process_group_ranks(state.sp_group) == list(range(rank // 4 * 4, rank // 4 * 4 + 4))
        return result

    parallel._init_parallel_state = checked_init
    _channel_loss_worker(rank, world_size, options, native_npu, endpoint)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=8))
def test_dpo_channel_loss_dp2_u2_cp2_cpu():
    options = dict(dp_size=2, ulysses_size=2, cp_size=2, allow_hybrid_cp=True)
    spawn_with_rendezvous(_checked_worker, args=(8, options, False), nprocs=8)


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator", nproc=8))
@pytest.mark.skipif(not IS_NPU_AVAILABLE or runner_device_count() < 8, reason="requires eight Ascend NPUs and HCCL")
def test_dpo_channel_loss_dp2_u2_cp2_hccl(monkeypatch):
    monkeypatch.setenv("HCCL_DETERMINISTIC", "true")
    options = dict(dp_size=2, ulysses_size=2, cp_size=2, allow_hybrid_cp=True)
    spawn_with_rendezvous(_checked_worker, args=(8, options, True), nprocs=8)
