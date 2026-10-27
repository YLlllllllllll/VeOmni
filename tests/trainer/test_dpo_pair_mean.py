# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Check actual DPO optimizer gradients against the same global pair mean."""

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel
from transformers.utils import ModelOutput

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_device_count
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.data.data_collator import MainCollator, PostCollator
from veomni.data.data_transform import process_dpo_example
from veomni.data.dpo_collator import DPOProvenanceCollator, DPOZeroTailCollator
from veomni.distributed.clip_grad_norm import _allreduce_ddp_sp_grads
from veomni.trainer import text_dpo_trainer as dpo
from veomni.utils.device import IS_NPU_AVAILABLE


@dataclass
class _AuxOutput(ModelOutput):
    log_probs: torch.Tensor | None = None


@dataclass
class _ModelOutput(ModelOutput):
    fused_linear_aux: _AuxOutput | None = None
    marker: int | None = None


class _TinyModel(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def forward(self, input_ids, **kwargs):
        return _ModelOutput(fused_linear_aux=_AuxOutput(log_probs=input_ids.float() * self.weight), marker=0)


class _FSDPTinyModel(_TinyModel):
    def __init__(self, value):
        torch.nn.Module.__init__(self)
        self.weight = torch.nn.Parameter(torch.tensor([value, 0.0, 0.0, 0.0], dtype=torch.float32))

    def forward(self, input_ids, **kwargs):
        return _ModelOutput(fused_linear_aux=_AuxOutput(log_probs=input_ids.float() * self.weight[0]), marker=0)


class _CaptureOptimizer:
    def __init__(self, model, *, require_dtensor=False):
        self.model = model
        self.require_dtensor = require_dtensor
        self.gradient = None
        self.gradient_vector = None
        self.steps = 0

    def step(self):
        gradient = self.model.weight.grad
        if self.require_dtensor:
            assert isinstance(gradient, DTensor), "FSDP test requires an actually sharded DTensor gradient"
        if isinstance(gradient, DTensor):
            gradient = gradient.full_tensor()
        if gradient.ndim:
            self.gradient_vector = gradient.detach().clone()
            gradient = gradient[0]
        self.gradient = gradient.detach().clone()
        self.steps += 1

    def zero_grad(self):
        pass


class _Runtime:
    def __init__(self, value, state, *, policy, distributed, fsdp_mesh=None, device="cpu"):
        self.parallel_state = state
        self.model_config = SimpleNamespace(vocab_size=256)
        self.module = _FSDPTinyModel(value) if policy and fsdp_mesh is not None else _TinyModel(value)
        self.module.to(device)
        if policy and fsdp_mesh is not None:
            from torch.distributed._composable.fsdp import fully_shard

            self.model = fully_shard(self.module, mesh=fsdp_mesh)
        elif policy and distributed:
            self.model = DistributedDataParallel(self.module, process_group=getattr(state, "dp_group", None))
        else:
            self.model = self.module
        self.fsdp = fsdp_mesh is not None and policy
        if policy:
            self.optimizer = _CaptureOptimizer(self.module, require_dtensor=self.fsdp)
            self.lr_scheduler = SimpleNamespace(step=lambda: None)

    def __call__(self, **kwargs):
        return self.model(**kwargs)

    def clip_grad_norm(self):
        if self.fsdp:
            return torch.tensor(0.0)
        if self.parallel_state.sp_enabled:
            _allreduce_ddp_sp_grads(self.model, self.parallel_state)
        return self.module.weight.grad.detach().norm()


def _microbatch(coefficients, sp_size=1, sp_rank=0):
    ids, labels = [], []
    for value in coefficients:
        ids.extend((max(value, 0), 0, max(-value, 0), 0))
        labels.extend((-100, 1, -100, 1))
    ids = torch.tensor([ids], dtype=torch.long)
    labels = torch.tensor([labels], dtype=torch.long)
    cu = torch.arange(0, ids.shape[-1] + 1, 2, dtype=torch.int32)
    if sp_size > 1:
        labels = F.pad(labels[..., 1:], (0, 1), value=-100)
        width = ids.shape[-1] // sp_size
        ids = ids[:, sp_rank * width : (sp_rank + 1) * width]
        labels = labels[:, sp_rank * width : (sp_rank + 1) * width]
    return {"input_ids": ids, "labels": labels, "cu_seq_lens_q": cu}


def _build_trainer(state, *, distributed, monkeypatch=None, fsdp_mesh=None, device="cpu"):
    set_attribute = setattr if monkeypatch is None else monkeypatch.setattr
    set_attribute(dpo, "get_parallel_state", lambda: state)
    set_attribute(dpo, "use_parallel_state", lambda *_: nullcontext())
    set_attribute(dpo, "set_batch_invariant_mode", lambda *_: nullcontext())
    set_attribute(dpo, "mark_compile_step_begin", lambda *_: None)
    trainer = dpo.TextDPOTrainer.__new__(dpo.TextDPOTrainer)
    cfg = SimpleNamespace(
        explicit_segment_provenance=False,
        average_log_prob=False,
        beta=0.1,
        label_smoothing=0.0,
        loss_type="sigmoid",
        reference_free=False,
    )
    captured = {}
    trainer.base = SimpleNamespace(
        args=SimpleNamespace(dpo_config=cfg, train=SimpleNamespace(enable_batch_invariant_mode=False)),
        state=SimpleNamespace(global_step=0, epoch=0),
        _callbacks=[],
        model_fwd_context=nullcontext(),
        model_bwd_context=nullcontext(),
        preforward=lambda batch: {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()},
        _reset_async_activation_offload_if_enabled=lambda *_: None,
        on_step_begin=lambda **kwargs: None,
        sync_before_train_step=lambda: None,
        model_reshard=lambda *args: None,
        _configure_hsdp_allreduce=lambda *args: None,
        on_step_end=lambda **kwargs: captured.update(kwargs),
    )
    trainer.policy_model = _Runtime(
        -0.7, state, policy=True, distributed=distributed, fsdp_mesh=fsdp_mesh, device=device
    )
    trainer.reference_model = _Runtime(0.0, state, policy=False, distributed=distributed, device=device)
    trainer.post_forward = PostCollator()
    return trainer, captured


def _expected(coefficients):
    weight = torch.tensor(-0.7, requires_grad=True)
    rewards = torch.tensor(coefficients, dtype=torch.float32) * weight * 0.1
    loss = -F.logsigmoid(rewards).mean()
    loss.backward()
    return loss.detach(), weight.grad.detach(), rewards.detach().mean()


def _run(trainer, groups, state):
    batches = [_microbatch(group, state.sp_size, state.sp_rank if state.sp_enabled else 0) for group in groups]
    trainer.train_step(iter([batches]))


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("groups", [[[1, 3, 3, 3]], [[1], [3, 3, 3]]])
def test_single_process_pair_weighted_gradient_and_metrics(groups, monkeypatch):
    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    trainer, metrics = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
    _run(trainer, groups, state)
    expected_loss, expected_grad, expected_chosen = _expected([1, 3, 3, 3])
    torch.testing.assert_close(trainer.policy_model.optimizer.gradient, expected_grad)
    assert metrics["loss"] == pytest.approx(expected_loss.item(), abs=1e-7)
    assert metrics["loss_dict"]["chosen_rewards"] == pytest.approx(expected_chosen.item(), abs=1e-7)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_reward_accuracy_is_pair_weighted_not_microbatch_weighted(monkeypatch):
    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    trainer, metrics = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
    _run(trainer, [[-2], [1, 3, 3]], state)
    assert metrics["loss_dict"]["reward_accuracy"] == pytest.approx(0.25, abs=1e-7)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("accepts_kwargs", [False, True])
def test_legacy_backward_override_fails_before_optimizer(monkeypatch, accepts_kwargs):
    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    trainer, _ = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
    if accepts_kwargs:
        trainer.forward_backward_step = lambda micro_batch, **kwargs: None
    else:
        trainer.forward_backward_step = lambda micro_batch: None
    with pytest.raises(RuntimeError, match="must accept pair_normalization"):
        _run(trainer, [[1]], state)
    assert trainer.policy_model.optimizer.steps == 0


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_unknown_tail_and_malformed_labels_fail_before_optimizer(monkeypatch):
    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    trainer, _ = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
    batch = _microbatch([1])
    batch["labels"] = batch["labels"][:, :-1]
    with pytest.raises(RuntimeError, match="same nonempty"):
        trainer.train_step(iter([[batch]]))
    assert trainer.policy_model.optimizer.steps == 0
    batch = _microbatch([1])
    batch["input_ids"] = F.pad(batch["input_ids"], (0, 2))
    batch["labels"] = F.pad(batch["labels"], (0, 2), value=-100)
    batch["cu_seq_lens_q"] = torch.tensor([0, 2, 4, 5, 6], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="trailing one-token segments are ambiguous"):
        trainer.train_step(iter([[batch]]))
    assert trainer.policy_model.optimizer.steps == 0


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize(
    "case", ["float_ids", "negative_ids", "negative_labels", "ids_outside_vocab", "labels_outside_vocab"]
)
def test_invalid_token_values_fail_before_optimizer(monkeypatch, case):
    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    trainer, _ = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
    batch = _microbatch([1])
    if case == "float_ids":
        batch["input_ids"] = batch["input_ids"].float()
    elif case == "negative_ids":
        batch["input_ids"][0, 0] = -1
    elif case == "negative_labels":
        batch["labels"][0, 0] = -2
    elif case == "ids_outside_vocab":
        batch["input_ids"][0, 0] = 256
    else:
        batch["labels"][0, 0] = 256
    with pytest.raises(RuntimeError, match="token ID"):
        trainer.train_step(iter([[batch]]))
    assert trainer.policy_model.optimizer.steps == 0


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("case", ["key_boundaries", "linear_boundaries", "maximum", "attention_shape"])
def test_inconsistent_attention_metadata_fails_before_optimizer(monkeypatch, case):
    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    trainer, _ = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
    batch = _microbatch([1])
    if case == "key_boundaries":
        batch["cu_seq_lens_k"] = torch.tensor([0, 1, 4], dtype=torch.int32)
    elif case == "linear_boundaries":
        batch["linear_attn_cu_seq_lens_q"] = torch.tensor([0, 1, 4], dtype=torch.int32)
    elif case == "maximum":
        batch["max_length_q"] = 3
    else:
        batch["attention_mask"] = torch.ones((1, 3), dtype=torch.long)
    with pytest.raises(RuntimeError):
        trainer.train_step(iter([[batch]]))
    assert trainer.policy_model.optimizer.steps == 0


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_standard_zero_tail_preserves_one_token_pair(monkeypatch):
    import veomni.data.data_collator as collator_module

    class Tokenizer:
        def encode(self, text, add_special_tokens=True):
            return [ord(value) for value in text]

    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    monkeypatch.setattr(collator_module, "get_parallel_state", lambda: state)
    features = [
        process_dpo_example({"chosen": chosen, "rejected": rejected}, tokenizer=Tokenizer())[0]
        for chosen, rejected in [("Bx", "Ax"), ("B", "A")]
    ]
    trainer, metrics = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
    batch = DPOZeroTailCollator(MainCollator())(features)
    assert batch["cu_seq_lens_q"].tolist() == [0, 2, 4, 5, 6]
    assert batch["tail_padding_length"].item() == 0
    trainer.train_step(iter([[batch]]))
    expected_weight = torch.tensor(-0.7, requires_grad=True)
    expected_loss = (-F.logsigmoid(expected_weight * 0.1) - F.logsigmoid(expected_weight * 0)) / 2
    expected_loss.backward()
    torch.testing.assert_close(trainer.policy_model.optimizer.gradient, expected_weight.grad)
    assert metrics["loss"] == pytest.approx(expected_loss.item(), abs=1e-7)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_known_nonzero_collator_tail_is_trimmed_without_sp(monkeypatch):
    import veomni.data.data_collator as collator_module

    class Tokenizer:
        def encode(self, text, add_special_tokens=True):
            return [ord(value) for value in text]

    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    monkeypatch.setattr(collator_module, "get_parallel_state", lambda: state)
    features = [process_dpo_example({"chosen": "Bx", "rejected": "Ax"}, tokenizer=Tokenizer())[0]]
    results = []
    for collator in (MainCollator(), MainCollator(pad_to_length=8)):
        trainer, metrics = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
        trainer.train_step(iter([[collator(features)]]))
        results.append((trainer.policy_model.optimizer.gradient.clone(), metrics["loss"]))
    torch.testing.assert_close(results[0][0], results[1][0])
    assert results[0][1] == pytest.approx(results[1][1], abs=1e-7)


def _real_collator_sp_worker(rank, world_size, endpoint):
    import veomni.data.data_collator as collator_module

    class Tokenizer:
        def encode(self, text, add_special_tokens=True):
            return [ord(value) for value in text]

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=world_size)
    try:
        dp_groups = [dist.new_group([0]), dist.new_group([1])]
        state = SimpleNamespace(
            sp_enabled=True,
            sp_size=2,
            sp_rank=rank,
            sp_group=dist.group.WORLD,
            dp_group=dp_groups[rank],
            fsdp_group=dist.group.WORLD,
        )
        collator_module.get_parallel_state = lambda: state
        trainer, metrics = _build_trainer(state, distributed=True)
        features = [
            process_dpo_example({"chosen": chosen, "rejected": rejected}, tokenizer=Tokenizer())[0]
            for chosen, rejected in [("Bx", "Ax"), ("B", "A")]
        ]
        batch = DPOZeroTailCollator(MainCollator())(features)
        assert batch["cu_seq_lens_q"].tolist() == [0, 2, 4, 5, 6]
        assert batch["tail_padding_length"].item() == 0
        trainer.train_step(iter([[batch]]))
        expected_weight = torch.tensor(-0.7, requires_grad=True)
        expected_loss = (-F.logsigmoid(expected_weight * 0.1) - F.logsigmoid(expected_weight * 0)) / 2
        expected_loss.backward()
        torch.testing.assert_close(trainer.policy_model.optimizer.gradient, expected_weight.grad)
        assert metrics["loss"] == pytest.approx(expected_loss.item(), abs=1e-7)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_standard_zero_tail_one_token_pair_with_sp():
    spawn_with_rendezvous(_real_collator_sp_worker, args=(2,), nprocs=2)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_legacy_callback_can_replace_valid_batches_before_weighting(monkeypatch):
    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    trainer, metrics = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
    trainer.base.on_step_begin = lambda micro_batches, **kwargs: micro_batches.__setitem__(
        0, _microbatch([1, 3, 3, 3])
    )
    _run(trainer, [[1]], state)
    expected_loss, expected_grad, _ = _expected([1, 3, 3, 3])
    torch.testing.assert_close(trainer.policy_model.optimizer.gradient, expected_grad)
    assert metrics["loss"] == pytest.approx(expected_loss.item(), abs=1e-7)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_provenance_enabled_uneven_microbatches_match_pair_mean(monkeypatch):
    import veomni.data.data_collator as collator_module

    class Tokenizer:
        def encode(self, text, add_special_tokens=True):
            return [ord(value) for value in text]

    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    monkeypatch.setattr(collator_module, "get_parallel_state", lambda: state)
    trainer, metrics = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
    trainer.base.args.dpo_config.explicit_segment_provenance = True
    trainer.base.args.data = SimpleNamespace(enable_multisource=False)
    trainer.policy_model.model_config = SimpleNamespace(vocab_size=256)
    collator = DPOProvenanceCollator(MainCollator(pad_to_length=16))

    def feature(coefficient):
        return process_dpo_example(
            {"chosen": chr(ord("A") + coefficient) + "x", "rejected": "Ax"},
            tokenizer=Tokenizer(),
            record_segment_provenance=True,
        )[0]

    batches = [collator([feature(1)]), collator([feature(3), feature(3), feature(3)])]
    trainer.train_step(iter([batches]))
    expected_loss, expected_grad, _ = _expected([1, 3, 3, 3])
    torch.testing.assert_close(trainer.policy_model.optimizer.gradient, expected_grad, rtol=1e-6, atol=1e-7)
    assert metrics["loss"] == pytest.approx(expected_loss.item(), abs=1e-6)


def _distributed_worker(rank, world_size, scenario, native_npu, endpoint):
    if native_npu:
        import torch_npu  # noqa: F401

        torch.npu.set_device(rank)
    device = f"npu:{rank}" if native_npu else "cpu"
    dist.init_process_group(
        "hccl" if native_npu else "gloo",
        store=endpoint.connect(),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    try:
        if scenario == "dp2_sp2":
            groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
            dp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
            state = SimpleNamespace(
                sp_enabled=True,
                sp_size=2,
                sp_rank=rank % 2,
                sp_group=groups[rank // 2],
                dp_group=dp_groups[rank % 2],
                fsdp_group=dist.group.WORLD,
            )
            batches = [[1], [2]] if rank < 2 else [[3, 3], [4, 4, 4]]
            expected = [1, 2, 3, 3, 4, 4, 4]
        else:
            state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0, fsdp_group=dist.group.WORLD)
            batches = [[1], [2]] if rank == 0 else [[3, 3], [4, 4, 4]]
            expected = [1, 2, 3, 3, 4, 4, 4]
        trainer, metrics = _build_trainer(state, distributed=True, device=device)
        _run(trainer, batches, state)
        expected_loss, expected_grad, expected_chosen = _expected(expected)
        torch.testing.assert_close(trainer.policy_model.optimizer.gradient.cpu(), expected_grad, rtol=1e-6, atol=1e-7)
        assert metrics["loss"] == pytest.approx(expected_loss.item(), abs=1e-6)
        assert metrics["loss_dict"]["chosen_rewards"] == pytest.approx(expected_chosen.item(), abs=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=lambda context: LaunchSpec(device="cpu", nproc=context.params["world_size"]))
@pytest.mark.parametrize(("world_size", "scenario"), [(2, "dp2"), (4, "dp2_sp2")])
def test_distributed_unequal_pair_counts_match_global_reference(world_size, scenario):
    spawn_with_rendezvous(_distributed_worker, args=(world_size, scenario, False), nprocs=world_size)


def _provenance_dp_sp_worker(rank, world_size, native_npu, endpoint):
    import veomni.data.data_collator as collator_module

    class Tokenizer:
        def encode(self, text, add_special_tokens=True):
            return [ord(value) for value in text]

    if native_npu:
        import torch_npu  # noqa: F401

        torch.npu.set_device(rank)
    device = f"npu:{rank}" if native_npu else "cpu"
    dist.init_process_group(
        "hccl" if native_npu else "gloo",
        store=endpoint.connect(),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    try:
        groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        dp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
        dp_rank = rank // 2
        state = SimpleNamespace(
            sp_enabled=True,
            sp_size=2,
            sp_rank=rank % 2,
            sp_group=groups[dp_rank],
            dp_group=dp_groups[rank % 2],
            fsdp_group=dist.group.WORLD,
        )
        collator_module.get_parallel_state = lambda: state
        trainer, metrics = _build_trainer(state, distributed=True, device=device)
        trainer.base.args.dpo_config.explicit_segment_provenance = True
        trainer.base.args.data = SimpleNamespace(enable_multisource=False)
        collator = DPOProvenanceCollator(MainCollator())

        def feature(coefficient):
            return process_dpo_example(
                {"chosen": chr(ord("A") + coefficient) + "x", "rejected": "Ax"},
                tokenizer=Tokenizer(),
                record_segment_provenance=True,
            )[0]

        microbatches = [[1], [3, 3]] if dp_rank == 0 else [[2], [4, 5, 6]]
        trainer.train_step(iter([[collator([feature(value) for value in group]) for group in microbatches]]))
        expected_loss, expected_grad, _ = _expected([1, 3, 3, 2, 4, 5, 6])
        torch.testing.assert_close(trainer.policy_model.optimizer.gradient.cpu(), expected_grad, rtol=1e-5, atol=1e-6)
        assert metrics["loss"] == pytest.approx(expected_loss.item(), abs=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=4))
def test_provenance_enabled_unequal_pairs_with_dp_and_sp():
    spawn_with_rendezvous(_provenance_dp_sp_worker, args=(4, False), nprocs=4)


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator", nproc=4))
@pytest.mark.skipif(not IS_NPU_AVAILABLE or runner_device_count() < 4, reason="requires four Ascend NPUs and HCCL")
def test_native_hccl_provenance_dp_sp_pair_weighting():
    spawn_with_rendezvous(_provenance_dp_sp_worker, args=(4, True), nprocs=4)


def _fsdp_worker(rank, world_size, hsdp, sp, native_npu, endpoint):
    from torch.distributed.device_mesh import init_device_mesh

    if native_npu:
        import torch_npu  # noqa: F401

        torch.npu.set_device(rank)
    device = f"npu:{rank}" if native_npu else "cpu"
    dist.init_process_group(
        "hccl" if native_npu else "gloo",
        store=endpoint.connect(),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    try:
        shape = (2, world_size // 2) if hsdp else (world_size,)
        names = ("replicate", "shard") if hsdp else ("shard",)
        mesh = init_device_mesh("npu" if native_npu else "cpu", shape, mesh_dim_names=names)
        if sp:
            sp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
            state = SimpleNamespace(
                sp_enabled=True,
                sp_size=2,
                sp_rank=rank % 2,
                sp_group=sp_groups[rank // 2],
                fsdp_group=dist.group.WORLD,
            )
        else:
            state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0, fsdp_group=dist.group.WORLD)
        trainer, metrics = _build_trainer(state, distributed=True, fsdp_mesh=mesh, device=device)
        if hsdp:

            def configure_hsdp(micro_step, num_micro_steps, runtime):
                if micro_step == 0:
                    runtime.model.set_requires_all_reduce(False)
                elif micro_step == num_micro_steps - 1:
                    runtime.model.set_requires_all_reduce(True)

            trainer.base._configure_hsdp_allreduce = configure_hsdp
        if sp:
            groups = [[1], [2]] if rank < 2 else [[3, 3], [4, 4, 4]]
            all_values = [1, 2, 3, 3, 4, 4, 4]
        else:
            groups = [[rank + 1], [rank + 2] * (rank + 1)]
            all_values = [
                item
                for owner in range(world_size)
                for group in ([owner + 1], [owner + 2] * (owner + 1))
                for item in group
            ]
        _run(trainer, groups, state)
        expected_loss, expected_grad, expected_chosen = _expected(all_values)
        torch.testing.assert_close(trainer.policy_model.optimizer.gradient.cpu(), expected_grad, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(
            trainer.policy_model.optimizer.gradient_vector[1:].cpu(), torch.zeros(3), rtol=0, atol=0
        )
        assert metrics["loss"] == pytest.approx(expected_loss.item(), abs=1e-6)
        assert metrics["loss_dict"]["chosen_rewards"] == pytest.approx(expected_chosen.item(), abs=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=lambda context: LaunchSpec(device="cpu", nproc=context.params["world_size"]))
@pytest.mark.parametrize(
    ("world_size", "hsdp", "sp"), [(2, False, False), (4, True, False), (4, False, True), (4, True, True)]
)
def test_fsdp2_and_hsdp_pair_weighted_gradients(world_size, hsdp, sp):
    spawn_with_rendezvous(_fsdp_worker, args=(world_size, hsdp, sp, False), nprocs=world_size)


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator", nproc=4))
@pytest.mark.skipif(not IS_NPU_AVAILABLE or runner_device_count() < 4, reason="requires four Ascend NPUs and HCCL")
def test_native_hccl_ddp_sp_pair_weighting():
    assert torch.npu.device_count() >= 4
    spawn_with_rendezvous(_distributed_worker, args=(4, "dp2_sp2", True), nprocs=4)


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator", nproc=4))
@pytest.mark.skipif(not IS_NPU_AVAILABLE or runner_device_count() < 4, reason="requires four Ascend NPUs and HCCL")
@pytest.mark.parametrize("hsdp", [False, True])
def test_native_hccl_fsdp_sp_pair_weighting(hsdp):
    assert torch.npu.device_count() >= 4
    spawn_with_rendezvous(_fsdp_worker, args=(4, hsdp, True, True), nprocs=4)


def _replace_with_adamw(trainer):
    trainer.policy_model.optimizer = torch.optim.AdamW(trainer.policy_model.module.parameters(), lr=0.01)
    trainer.policy_model.lr_scheduler = torch.optim.lr_scheduler.StepLR(trainer.policy_model.optimizer, step_size=1)


def _cold_resume_worker(rank, checkpoint_path, result_path):
    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    trainer, metrics = _build_trainer(state, distributed=False)
    _replace_with_adamw(trainer)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    trainer.validate_dpo_provenance_checkpoint(saved["dpo_protocol"])
    trainer.policy_model.module.load_state_dict(saved["model"])
    trainer.policy_model.optimizer.load_state_dict(saved["optimizer"])
    trainer.policy_model.lr_scheduler.load_state_dict(saved["scheduler"])
    trainer.state.global_step = saved["step"]
    _run(trainer, [[2], [4, 5]], state)
    torch.save(
        {
            "model": trainer.policy_model.module.state_dict(),
            "optimizer": trainer.policy_model.optimizer.state_dict(),
            "metrics": metrics,
        },
        result_path,
    )


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_pair_mean_checkpoint_fresh_process_matches_uninterrupted(tmp_path, monkeypatch):
    state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0)
    trainer, metrics = _build_trainer(state, distributed=False, monkeypatch=monkeypatch)
    _replace_with_adamw(trainer)
    _run(trainer, [[1], [2, 3]], state)
    checkpoint = tmp_path / "step1.pt"
    torch.save(
        {
            "model": trainer.policy_model.module.state_dict(),
            "optimizer": trainer.policy_model.optimizer.state_dict(),
            "scheduler": trainer.policy_model.lr_scheduler.state_dict(),
            "dpo_protocol": trainer.dpo_provenance_checkpoint_state(),
            "step": trainer.state.global_step,
        },
        checkpoint,
    )
    _run(trainer, [[2], [4, 5]], state)
    result = tmp_path / "cold.pt"
    mp.spawn(_cold_resume_worker, args=(str(checkpoint), str(result)), nprocs=1, join=True)
    restored = torch.load(result, map_location="cpu", weights_only=False)
    torch.testing.assert_close(restored["model"]["weight"], trainer.policy_model.module.weight, rtol=0, atol=0)
    expected_state = next(iter(trainer.policy_model.optimizer.state_dict()["state"].values()))
    restored_state = next(iter(restored["optimizer"]["state"].values()))
    for key in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(restored_state[key], expected_state[key], rtol=0, atol=0)
    assert restored["metrics"]["loss"] == metrics["loss"]


def _invalid_worker(rank, world_size, scenario, endpoint):
    dist.init_process_group(
        "gloo", store=endpoint.connect(), rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        if scenario in ("sp_boundaries", "sp_tail_divergence"):
            dp_groups = [dist.new_group([0]), dist.new_group([1])]
            state = SimpleNamespace(
                sp_enabled=True,
                sp_size=2,
                sp_rank=rank,
                sp_group=dist.group.WORLD,
                dp_group=dp_groups[rank],
                fsdp_group=dist.group.WORLD,
            )
        else:
            state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0, fsdp_group=dist.group.WORLD)
        trainer, _ = _build_trainer(state, distributed=True)
        if scenario == "mismatched_microsteps":
            batches = [_microbatch([1])] * (1 if rank == 0 else 2)
        elif scenario in ("one_eof", "all_eof", "loader_error", "eof_and_loader_error", "callback_error"):
            batches = [_microbatch([1])]
            if scenario == "callback_error" and rank == 1:

                def fail_callback(**kwargs):
                    raise ValueError("injected callback error")

                trainer.base.on_step_begin = fail_callback
        else:
            if scenario == "sp_tail_divergence":
                local_tokens = 3 if rank == 0 else 4
                batch = {
                    "input_ids": torch.zeros((1, local_tokens), dtype=torch.long),
                    "labels": torch.full((1, local_tokens), -100, dtype=torch.long),
                    "cu_seq_lens_q": torch.tensor([0, 2, 4, 2 * local_tokens], dtype=torch.int32),
                    "tail_padding_length": torch.tensor(2 * local_tokens - 4, dtype=torch.int32),
                }
            else:
                batch = _microbatch([1], state.sp_size, state.sp_rank)
            if rank == 1 and scenario == "sp_boundaries":
                batch["cu_seq_lens_q"] = torch.tensor([0, 1, 4], dtype=torch.int32)
            if rank == 1 and scenario == "bad_labels":
                batch["labels"] = batch["labels"][:, :-1]
            if rank == 1 and scenario == "bad_token":
                batch["input_ids"][0, 0] = 256
            batches = [batch]
        if (
            scenario == "all_eof"
            or (scenario == "one_eof" and rank == 1)
            or (scenario == "eof_and_loader_error" and rank == 0)
        ):
            iterator = iter([])
        elif scenario in ("loader_error", "eof_and_loader_error") and rank == 1:

            def fail_loader():
                raise ValueError("injected loader error")
                yield  # pragma: no cover

            iterator = fail_loader()
        else:
            iterator = iter([batches])
        exception = StopIteration if scenario == "all_eof" else RuntimeError
        with pytest.raises(exception, match="injected loader error" if scenario == "eof_and_loader_error" else None):
            trainer.train_step(iterator)
        assert trainer.policy_model.optimizer.steps == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
@pytest.mark.parametrize(
    "scenario",
    [
        "mismatched_microsteps",
        "sp_boundaries",
        "sp_tail_divergence",
        "one_eof",
        "all_eof",
        "loader_error",
        "eof_and_loader_error",
        "callback_error",
        "bad_labels",
        "bad_token",
    ],
)
def test_incompatible_distributed_batches_fail_before_optimizer(scenario):
    spawn_with_rendezvous(_invalid_worker, args=(2, scenario), nprocs=2)
