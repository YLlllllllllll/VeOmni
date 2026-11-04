# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Native Qwen3 registry selection with the unchanged RoPE recovery gate."""

import importlib
import os
from unittest.mock import patch

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.manual.qwen38_native_evidence import test_dynamic_rope_reset as baseline
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.models.registry import MODELING_REGISTRY
from veomni.models.transformers import qwen3
from veomni.utils.device import get_device_type


class _RegistryTinyModel(baseline._TinyModel):
    def __init__(self):
        super().__init__()
        with patch.object(qwen3, "IS_NPU_AVAILABLE", True):
            model_class = MODELING_REGISTRY["qwen3"]("Qwen3Model")
        assert model_class.__module__.endswith("patched_modeling_qwen3_npu")
        rope_class = importlib.import_module(model_class.__module__).Qwen3RotaryEmbedding
        self.rope = rope_class(self.rope.config)


def _worker(rank, backend, phase, root, endpoint):
    baseline._TinyModel = _RegistryTinyModel
    baseline._worker(rank, backend, phase, root, endpoint)


def _run(backend, root):
    spawn_with_rendezvous(_worker, args=(backend, "save", str(root)), nprocs=2)
    spawn_with_rendezvous(_worker, args=(backend, "load", str(root)), nprocs=2)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_registry_qwen3_reset_ddp_gloo(tmp_path):
    _run("gloo", tmp_path)


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator", nproc=2))
def test_registry_qwen3_reset_ddp_hccl(tmp_path):
    available = get_device_type() == "npu" and torch.npu.is_available() and torch.npu.device_count() >= 2
    if not available:
        if os.environ.get("VEOMNI_REQUIRE_NPU_TESTS") == "1":
            pytest.fail("This acceptance gate requires two usable Ascend NPUs")
        pytest.skip("requires two Ascend NPUs")
    _run("hccl", tmp_path)
