"""Cross-rank failure paths for weighted SFT and DPO input handling."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import spawn_with_rendezvous


def _worker(rank, endpoint):
    from tests.trainer.test_dpo_pair_mean import PostCollator, _build_trainer, _microbatch
    from veomni.data import data_collator
    from veomni.trainer import text_dpo_trainer as dpo
    from veomni.trainer.base import BaseTrainer
    from veomni.utils import loss_utils

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        data_collator.get_parallel_state = lambda: SimpleNamespace(sp_enabled=False)
        loss_utils.get_parallel_state = lambda: SimpleNamespace(sp_enabled=False)
        base = object.__new__(BaseTrainer)
        base.state = SimpleNamespace(global_step=3)
        base.model = SimpleNamespace()
        base._reset_async_activation_offload_if_enabled = lambda _model: None
        base.on_step_begin = lambda **kwargs: None
        feature = {
            "input_ids": torch.tensor([1, 2]),
            "labels": torch.tensor([1, 2]),
            "attention_mask": torch.tensor([1, 1]),
            "position_ids": torch.tensor([0, 1]),
            "loss_weights": torch.tensor([0.5] if rank == 1 else [0.5, 0.5]),
        }
        packer = data_collator.PackingCollator()
        loader = DataLoader([feature], batch_size=1, collate_fn=lambda samples: [packer(samples)])
        with pytest.raises(RuntimeError, match="data loading failed"):
            base._fetch_sft_micro_batches(iter(loader))
        assert base.state.global_step == 3
        with pytest.raises(RuntimeError, match="end of the data at different times"):
            base._fetch_sft_micro_batches(iter([] if rank == 0 else [[{"labels": torch.tensor([1])}]]))
        with pytest.raises(StopIteration):
            base._fetch_sft_micro_batches(iter([]))
        for malformed in ("padding", "field"):
            batch = {"labels": torch.tensor([1, 2])}
            if rank == 1 and malformed == "padding":
                batch["padding_flag"] = torch.tensor([True, False])
            if rank == 1 and malformed == "field":
                batch[3] = torch.tensor([1])
            with pytest.raises(RuntimeError, match="malformed batch metadata"):
                base._fetch_sft_micro_batches(iter([[batch]]))
            completed = torch.tensor(1)
            dist.all_reduce(completed)
            assert completed.item() == 2
        objective_batch = {"labels": torch.tensor([1, 2])}
        if rank == 1:
            objective_batch["padding_flag"] = torch.tensor([True, False])
        with pytest.raises(ValueError, match="does not support padded micro-batches"):
            loss_utils.reject_objective_padding([objective_batch], "test objective")
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2

        base.model = SimpleNamespace(
            model_config=SimpleNamespace(model_type="qwen4_exp"),
            unwrapped_module=SimpleNamespace(veomni_ce=SimpleNamespace(impl="chunk_loss")),
        )
        batches = [{"labels": torch.tensor([1, 2]), "loss_weights": torch.tensor([0.5, 0.5])}]

        def mutate(micro_batches):
            if rank == 1:
                micro_batches[0].pop("loss_weights")

        base.on_step_begin = mutate
        with pytest.raises(RuntimeError, match="callback or loss-token"):
            base._sft_step_preflight(batches, base)
        assert base.state.global_step == 3

        def add_padding(micro_batches):
            if rank == 1:
                micro_batches[0]["padding_flag"] = True

        base.on_step_begin = add_padding
        batches = [{"labels": torch.tensor([1, 2]), "loss_weights": torch.tensor([0.5, 0.5])}]
        with pytest.raises(RuntimeError, match="callback or loss-token"):
            base._sft_step_preflight(batches, base)
        assert base.state.global_step == 3

        batch = _microbatch([1])
        if rank == 1:
            batch["loss_weights"] = torch.ones_like(batch["labels"], dtype=torch.float32)
        state = SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0, fsdp_group=dist.group.WORLD)
        with pytest.raises(RuntimeError, match="DPO does not support token loss_weights"):
            dpo._dpo_pair_weighting([batch], PostCollator(), state, False)

        trainer, _ = _build_trainer(state, distributed=True)

        def invalidate(micro_batches, **kwargs):
            if rank == 1:
                micro_batches[0] = None

        trainer.base.on_step_begin = invalidate
        with pytest.raises(RuntimeError, match="DPO pair-count preflight failed"):
            trainer.train_step(iter([[_microbatch([1])]]))
        assert trainer.state.global_step == 0
        assert trainer.policy_model.optimizer.steps == 0
        auxiliary = {"labels": torch.tensor([[-100, 1]])}
        auxiliary["image_labels" if rank == 0 else "audio_labels"] = torch.tensor([1, 2, 3] if rank == 0 else [1])
        _, _, totals = loss_utils.prepare_loss_token_counts([auxiliary])
        assert totals["image_tokens"] == 3 and totals["audio_tokens"] == 1
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_weighted_cross_rank_errors_fail_before_forward():
    spawn_with_rendezvous(_worker, nprocs=2)
