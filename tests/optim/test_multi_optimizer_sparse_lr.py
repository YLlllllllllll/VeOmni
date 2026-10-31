"""An unused optimizer leaf still has a scheduler-controlled learning rate."""

from types import SimpleNamespace

import pytest
import torch

from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.optim.optimizer import MultiOptimizer


def _build():
    model = torch.nn.Linear(2, 2)
    weight = torch.optim.AdamW([model.weight], lr=0.01)
    bias = torch.optim.AdamW([model.bias], lr=0.01)
    optimizer = MultiOptimizer(model, {"weight": weight, "bias": bias}, ["weight", "bias"])
    return model, weight, bias, optimizer


def test_sparse_leaf_preserves_learning_rate_without_creating_state():
    model, weight, bias, optimizer = _build()
    model.weight.grad = torch.ones_like(model.weight)
    weight.step()
    weight.zero_grad()
    bias.param_groups[0]["lr"] = 0.0025
    result = optimizer._sparse_state_dict()
    assert result["param_groups.bias.lr"] == 0.0025
    assert not bias.state


def test_sparse_leaf_uses_optimizer_state_dict_post_hook():
    _, _, bias, optimizer = _build()

    def change_serialized_lr(_optimizer, state):
        state["param_groups"][0]["lr"] = 0.037
        return state

    bias.register_state_dict_post_hook(change_serialized_lr)
    result = optimizer._sparse_state_dict()
    assert result["param_groups.bias.lr"] == 0.037
    assert bias.param_groups[0]["lr"] == 0.01
    assert not bias.state


def test_sparse_leaf_preserves_tensor_learning_rate():
    model = torch.nn.Linear(2, 2)
    learning_rate = torch.tensor(0.0025)
    leaf = torch.optim.AdamW([model.bias], lr=learning_rate)
    optimizer = MultiOptimizer(model, {"bias": leaf}, ["bias"])
    result = optimizer._sparse_state_dict()
    torch.testing.assert_close(result["param_groups.bias.lr"], learning_rate, rtol=0, atol=0)
    assert not leaf.state


def test_sparse_leaf_serializes_state_created_by_pre_hook():
    model = torch.nn.Linear(2, 2)
    leaf = torch.optim.AdamW([model.bias], lr=0.01)
    optimizer = MultiOptimizer(model, {"bias": leaf}, ["bias"])

    def add_state(opt):
        opt.state[model.bias]["custom"] = torch.ones_like(model.bias)

    leaf.register_state_dict_pre_hook(add_state)
    result = optimizer._sparse_state_dict()
    torch.testing.assert_close(result["state.bias.custom"], torch.ones_like(model.bias), rtol=0, atol=0)


def test_sparse_leaf_respects_post_hook_parameter_reordering():
    model = torch.nn.Linear(2, 2)
    leaf = torch.optim.AdamW([{"params": [model.weight], "lr": 0.01}, {"params": [model.bias], "lr": 0.02}])
    optimizer = MultiOptimizer(model, {"leaf": leaf}, ["leaf"])

    def swap_serialized_groups(_optimizer, state):
        first, second = state["param_groups"]
        first["params"], second["params"] = second["params"], first["params"]
        return state

    leaf.register_state_dict_post_hook(swap_serialized_groups)
    result = optimizer._sparse_state_dict()
    assert result["param_groups.weight.lr"] == 0.02
    assert result["param_groups.bias.lr"] == 0.01
    assert not leaf.state


def test_sparse_leaf_respects_live_group_reordering_in_post_hook():
    model = torch.nn.Linear(2, 2)
    leaf = torch.optim.AdamW([{"params": [model.weight], "lr": 0.01}, {"params": [model.bias], "lr": 0.02}])
    optimizer = MultiOptimizer(model, {"leaf": leaf}, ["leaf"])

    def reverse_live_groups(opt, state):
        opt.param_groups.reverse()
        return state

    leaf.register_state_dict_post_hook(reverse_live_groups)
    result = optimizer._sparse_state_dict()
    assert result["param_groups.weight.lr"] == 0.01
    assert result["param_groups.bias.lr"] == 0.02
    assert len(leaf._optimizer_state_dict_pre_hooks) == 0
    assert len(leaf._optimizer_state_dict_post_hooks) == 1


def test_sparse_leaf_rejects_custom_state_dict():
    class CustomAdamW(torch.optim.AdamW):
        def state_dict(self):
            return super().state_dict()

    model = torch.nn.Linear(2, 2)
    leaf = CustomAdamW(model.parameters())
    optimizer = MultiOptimizer(model, {"leaf": leaf}, ["leaf"])
    with pytest.raises(RuntimeError, match="custom state_dict"):
        optimizer._sparse_state_dict()


def test_sparse_leaf_rejects_instance_patched_state_dict_before_mutation():
    from torch.distributed.checkpoint.state_dict import _patch_optimizer_state_dict

    model = torch.nn.Linear(2, 2)
    leaf = torch.optim.AdamW(model.parameters())
    optimizer = MultiOptimizer(model, {"leaf": leaf}, ["leaf"])
    _patch_optimizer_state_dict(model, optimizers=(leaf,))
    before = [param.detach().clone() for param in model.parameters()]
    with pytest.raises(RuntimeError, match="custom state_dict"):
        optimizer._sparse_state_dict()
    assert not leaf.state
    for actual, expected in zip(model.parameters(), before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sparse_leaf_reentrant_hook_fails_without_leaking_temporary_hooks():
    model = torch.nn.Linear(2, 2)
    leaf = torch.optim.AdamW(model.parameters())
    optimizer = MultiOptimizer(model, {"leaf": leaf}, ["leaf"])
    entered = False

    def reenter(opt):
        nonlocal entered
        if not entered:
            entered = True
            try:
                opt.state_dict()
            finally:
                entered = False

    leaf.register_state_dict_pre_hook(reenter)
    with pytest.raises(RuntimeError, match="Cannot reenter"):
        optimizer._sparse_state_dict()
    assert not leaf.state
    assert len(leaf._optimizer_state_dict_pre_hooks) == 1
    assert not leaf._optimizer_state_dict_post_hooks


def test_sparse_leaf_learning_rate_survives_dcp_roundtrip(tmp_path):
    model, weight, bias, optimizer = _build()
    model.weight.grad = torch.ones_like(model.weight)
    weight.step()
    weight.zero_grad()
    bias.param_groups[0]["lr"] = 0.0025
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    assert not bias.state
    restored_model, _, restored_bias, restored = _build()
    DistributedCheckpointer.load(
        str(tmp_path), {"model": restored_model, "optimizer": restored}, parallel_state=parallel
    )
    assert restored_bias.param_groups[0]["lr"] == 0.0025


def test_sparse_leaf_post_hook_live_reorder_survives_dcp_roundtrip(tmp_path):
    def build():
        model = torch.nn.Linear(2, 2)
        leaf = torch.optim.AdamW([{"params": [model.weight], "lr": 0.01}, {"params": [model.bias], "lr": 0.02}])
        return model, leaf, MultiOptimizer(model, {"leaf": leaf}, ["leaf"])

    model, leaf, optimizer = build()
    leaf.register_state_dict_post_hook(lambda opt, state: (opt.param_groups.reverse(), state)[1])
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    restored_model, restored_leaf, restored = build()
    DistributedCheckpointer.load(
        str(tmp_path), {"model": restored_model, "optimizer": restored}, parallel_state=parallel
    )
    assert restored_leaf.param_groups[0]["lr"] == 0.01
    assert restored_leaf.param_groups[1]["lr"] == 0.02
