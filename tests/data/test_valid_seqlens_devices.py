"""Tail-segment extraction must agree on CPU and Ascend NPU."""

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.utils.device import IS_NPU_AVAILABLE
from veomni.utils.seqlen_pos_transform_utils import valid_seqlens_from_cu_seqlens


_CASES = [
    ([0, 4, 7, 8, 9, 10], None, [4, 3]),
    ([0, 4, 5, 6, 7, 8], 3, [4, 1]),
    ([0, 4, 5], 0, [4, 1]),
    ([0, 1, 3], None, [1, 2]),
    ([0, 1, 2], None, []),
    ([0], None, []),
]


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("cu,tail,expected", _CASES)
def test_valid_seqlens_cpu(cu, tail, expected):
    value = torch.tensor(cu, dtype=torch.int32)
    result = valid_seqlens_from_cu_seqlens(value, tail_padding_length=tail)
    assert result.tolist() == expected
    assert result.dtype == torch.int32


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator"))
@pytest.mark.skipif(not IS_NPU_AVAILABLE or not torch.npu.is_available(), reason="requires an Ascend NPU")
@pytest.mark.parametrize("cu,tail,expected", _CASES)
def test_valid_seqlens_ascend(cu, tail, expected):
    value = torch.tensor(cu, dtype=torch.int32, device="npu")
    result = valid_seqlens_from_cu_seqlens(value, tail_padding_length=tail)
    assert result.cpu().tolist() == expected
    assert result.dtype == torch.int32
