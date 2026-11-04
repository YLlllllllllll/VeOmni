# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Small distributed native-Qwen3 dynamic-RoPE checkpoint acceptance gate."""

import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.optim.optimizer import MultiOptimizer
from veomni.utils.device import get_device_type


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(8, 8)
        config = Qwen3Config(
            hidden_size=32,
            head_dim=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=16,
            rope_parameters={"rope_type": "dynamic", "factor": 2.0, "rope_theta": 10000},
        )
        self.rope = Qwen3RotaryEmbedding(config)

    def forward(self, inputs, positions):
        projected = self.projection(inputs)
        cosine, sine = self.rope(projected, positions)
        return projected, cosine, sine


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_copy(item) for item in value)
    return value


def _equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert isinstance(expected, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _equal(left, right)
    else:
        assert actual == expected


def _step(model, optimizer, device, length):
    inputs = torch.arange(length * 8, dtype=torch.float32, device=device).reshape(1, length, 8) / (length * 8)
    inputs = inputs + dist.get_rank() / 8
    positions = torch.arange(length, device=device).reshape(1, -1)
    projected, cosine, sine = model(inputs, positions)
    loss = (projected * cosine + sine).square().mean()
    loss.backward()
    for parameter in model.parameters():
        gathered = [torch.empty_like(parameter.grad) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, parameter.grad.contiguous())
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
    trace = {
        "cos": cosine,
        "sin": sine,
        "loss": loss,
        "grad": {name: parameter.grad for name, parameter in model.named_parameters()},
    }
    trace = _cpu_copy(trace)
    optimizer.step()
    optimizer.zero_grad()
    trace["parameters"] = _cpu_copy(dict(model.named_parameters()))
    return trace


def _worker(rank, backend, phase, root, endpoint):
    if backend == "hccl":
        torch.npu.set_device(rank)
        device = torch.device("npu", rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend, store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        torch.manual_seed(13)
        raw = _TinyModel().to(device)
        model = DistributedDataParallel(raw, device_ids=[rank] if backend == "hccl" else None, broadcast_buffers=False)
        leaf = torch.optim.AdamW(model.parameters(), lr=0.01)
        optimizer = MultiOptimizer(model, {"model": leaf}, ["model"])
        state = {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None}
        checkpoint = str(Path(root) / "checkpoint")
        reference = Path(root) / f"reference-{rank}.pt"
        parallel = SimpleNamespace(dp_mode="ddp")
        if phase == "save":
            _step(model, optimizer, device, 8)
            if rank == 0:
                raw.rope(torch.zeros((1, 1, 8), device=device), torch.arange(32, device=device).reshape(1, -1))
                raw.rope(torch.zeros((1, 1, 8), device=device), torch.arange(4, device=device).reshape(1, -1))
            assert (raw.rope.inv_freq is raw.rope.original_inv_freq) == (rank == 0)
            DistributedCheckpointer.save(checkpoint, state, parallel_state=parallel)
        else:
            DistributedCheckpointer.load(checkpoint, state, parallel_state=parallel)
        traces = [_step(model, optimizer, device, length) for length in (8, 32, 24, 4)]
        actual = _cpu_copy({"traces": traces, "model": model.state_dict(), "optimizer": optimizer.state_dict()})
        if phase == "save":
            torch.save(actual, reference)
        else:
            _equal(actual, torch.load(reference, map_location="cpu", weights_only=True))
        value = torch.ones(1, device=device)
        dist.all_reduce(value)
        assert value.item() == 2
    finally:
        dist.destroy_process_group()


def _run(backend, root):
    spawn_with_rendezvous(_worker, args=(backend, "save", str(root)), nprocs=2)
    spawn_with_rendezvous(_worker, args=(backend, "load", str(root)), nprocs=2)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_qwen3_reset_ddp_gloo(tmp_path):
    _run("gloo", tmp_path)


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator", nproc=2))
def test_qwen3_reset_ddp_hccl(tmp_path):
    available = get_device_type() == "npu" and torch.npu.is_available() and torch.npu.device_count() >= 2
    if not available:
        if os.environ.get("VEOMNI_REQUIRE_NPU_TESTS") == "1":
            pytest.fail("This acceptance gate requires two usable Ascend NPUs")
        pytest.skip("requires two Ascend NPUs")
    _run("hccl", tmp_path)
