# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Reject individually canonical but byte-distinct RoPE tables before DCP load."""

import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.manual.qwen38_native_evidence import test_dynamic_rope_reset as baseline
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.checkpoint.dynamic_rope import dynamic_rope_checkpoint_contract
from veomni.checkpoint.optimizer_resume_layout import capture_multi_layout
from veomni.optim.optimizer import MultiOptimizer
from veomni.utils.device import get_device_type


def _worker(rank, backend, root, endpoint):
    if backend == "hccl":
        torch.npu.set_device(rank)
        device = torch.device("npu", rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend, store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        torch.manual_seed(13)
        source = baseline._TinyModel().to(device)
        target = baseline._TinyModel().to(device)
        if backend == "hccl":
            target.rope = Qwen3RotaryEmbedding(target.rope.config, device=device)
        else:
            # Simulate the observed CPU/device rounding difference on Gloo.
            shifted = torch.nextafter(target.rope.inv_freq, torch.full_like(target.rope.inv_freq, float("inf")))
            target.rope.inv_freq.copy_(shifted)
            target.rope.original_inv_freq.copy_(shifted)
            original = ROPE_INIT_FUNCTIONS["dynamic"]

            def skewed(config, device, seq_len=None, **kwargs):
                value, scale = original(config, device, seq_len=seq_len, **kwargs)
                if isinstance(device, torch.device):
                    value = torch.nextafter(value, torch.full_like(value, float("inf")))
                return value, scale

            ROPE_INIT_FUNCTIONS["dynamic"] = skewed
        source_rope = dynamic_rope_checkpoint_contract(source)
        target_rope = dynamic_rope_checkpoint_contract(target)
        assert len(source_rope) == len(target_rope) == 1
        assert source_rope[0][:-2] == target_rope[0][:-2] and source_rope[0][-2:] != target_rope[0][-2:]

        kwargs = {"device_ids": [rank]} if backend == "hccl" else {}
        source = DistributedDataParallel(source, broadcast_buffers=False, **kwargs)
        target = DistributedDataParallel(target, broadcast_buffers=False, **kwargs)
        source_leaf = torch.optim.AdamW(source.parameters(), lr=0.01)
        target_leaf = torch.optim.AdamW(target.parameters(), lr=0.02)
        source_optimizer = MultiOptimizer(source, {"model": source_leaf}, ["model"])
        target_optimizer = MultiOptimizer(target, {"model": target_leaf}, ["model"])
        baseline._step(source, source_optimizer, device, 8)
        source_layout = json.loads(capture_multi_layout(source, source_optimizer, None, None))
        target_layout = json.loads(capture_multi_layout(target, target_optimizer, None, None))
        assert source_layout.pop("dynamic_rope_cache_v2") != target_layout.pop("dynamic_rope_cache_v2")
        assert source_layout == target_layout
        checkpoint = str(Path(root) / "checkpoint")
        parallel = SimpleNamespace(dp_mode="ddp")
        DistributedCheckpointer.save(
            checkpoint,
            {"model": source, "optimizer": source_optimizer, "_veomni_scheduler_object": None},
            parallel_state=parallel,
        )
        before = baseline._cpu_copy(target.state_dict())
        before_optimizer = baseline._cpu_copy(target_optimizer.state_dict())
        before_rope = baseline._cpu_copy((target.module.rope.inv_freq, target.module.rope.original_inv_freq))
        with pytest.raises(RuntimeError, match="layout|identity"):
            DistributedCheckpointer.load(
                checkpoint,
                {"model": target, "optimizer": target_optimizer, "_veomni_scheduler_object": None},
                parallel_state=parallel,
            )
        baseline._equal(baseline._cpu_copy(target.state_dict()), before)
        baseline._equal(baseline._cpu_copy(target_optimizer.state_dict()), before_optimizer)
        baseline._equal(
            baseline._cpu_copy((target.module.rope.inv_freq, target.module.rope.original_inv_freq)), before_rope
        )
        assert not target_leaf.state
        value = torch.ones(1, device=device)
        dist.all_reduce(value)
        assert value.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_rope_origin_mismatch_gloo(tmp_path):
    spawn_with_rendezvous(_worker, args=("gloo", str(tmp_path)), nprocs=2)


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator", nproc=2))
def test_rope_origin_mismatch_hccl(tmp_path):
    available = get_device_type() == "npu" and torch.npu.is_available() and torch.npu.device_count() >= 2
    if not available:
        if os.environ.get("VEOMNI_REQUIRE_NPU_TESTS") == "1":
            pytest.fail("This acceptance gate requires two usable Ascend NPUs")
        pytest.skip("requires two Ascend NPUs")
    spawn_with_rendezvous(_worker, args=("hccl", str(tmp_path)), nprocs=2)
