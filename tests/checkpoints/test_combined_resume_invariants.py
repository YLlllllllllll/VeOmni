"""Integration-only guards for the combined checkpoint and data PR heads."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.optim.optimizer import MultiOptimizer


def _optimizer_kind_mismatch_worker(rank, root):
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/rdzv", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        torch.manual_seed(123)
        source = torch.nn.Linear(2, 2)
        leaf = torch.optim.AdamW(source.parameters())
        optimizer = MultiOptimizer(source, {"leaf": leaf}, ["leaf"])
        parallel = SimpleNamespace(dp_mode="ddp")
        path = f"{root}/checkpoint"
        DistributedCheckpointer.save(
            path, {"model": source, "optimizer": optimizer, "_veomni_scheduler_object": None}, parallel_state=parallel
        )

        target = torch.nn.Linear(2, 2)
        with torch.no_grad():
            target.weight.fill_(17)
        before = target.weight.detach().clone()
        target_leaf = torch.optim.AdamW(target.parameters())
        target_optimizer = MultiOptimizer(target, {"leaf": target_leaf}, ["leaf"]) if rank == 0 else target_leaf
        with pytest.raises(RuntimeError, match="Checkpoint load format differs across ranks"):
            DistributedCheckpointer.load(
                path,
                {"model": target, "optimizer": target_optimizer, "_veomni_scheduler_object": None},
                parallel_state=parallel,
            )
        torch.testing.assert_close(target.weight, before, rtol=0, atol=0)
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_optimizer_kind_mismatch_fails_coherently_before_load(tmp_path):
    mp.spawn(_optimizer_kind_mismatch_worker, args=(str(tmp_path),), nprocs=2, join=True)


def _ordinary_save_failure_worker(rank, root, stage, use_optimizer):
    from unittest.mock import patch

    dist.init_process_group(
        "gloo", init_method=f"file://{root}/rdzv", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        model = torch.nn.Linear(2, 2)
        state = {"model": model}
        if use_optimizer:
            state["optimizer"] = torch.optim.AdamW(model.parameters())
        name = "_save_extra_state" if stage == "sidecar" else "_create_storage_writer"
        original = getattr(DistributedCheckpointer, name)

        def fail_on_first_rank(*args, **kwargs):
            if rank == 0:
                raise OSError(f"injected {stage} failure")
            return original(*args, **kwargs)

        error_type = OSError if rank == 0 else RuntimeError
        message = f"injected {stage} failure" if rank == 0 else "failed on another rank"
        with patch.object(DistributedCheckpointer, name, side_effect=fail_on_first_rank):
            with pytest.raises(error_type, match=message):
                DistributedCheckpointer.save(
                    f"{root}/checkpoint", state, parallel_state=SimpleNamespace(dp_mode="ddp")
                )
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
@pytest.mark.parametrize("stage", ["sidecar", "writer"])
@pytest.mark.parametrize("use_optimizer", [False, True])
def test_ordinary_save_preparation_failure_is_rank_coordinated(tmp_path, stage, use_optimizer):
    mp.spawn(_ordinary_save_failure_worker, args=(str(tmp_path), stage, use_optimizer), nprocs=2, join=True)
