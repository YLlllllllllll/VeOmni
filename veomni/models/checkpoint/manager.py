# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpoint/resume for one :class:`~veomni.models.model_runtime.VeOmniModelRuntime`.

On-disk layout is owned by :mod:`veomni.checkpoint.layout` (see
``docs/usage/checkpoint.md``). This class only binds that layout to one model's
runtime — the module, optimizer, scheduler, assets, and the ParallelState they
were built under.

Two blobs, two owners:

* **lr_scheduler** — this model's scheduler. Passed to DCP like the optimizer;
  the checkpointer pickles ``state_dict`` as a single ``lr_scheduler.pt`` beside
  the two DCP directories.
* **global_state** — the job cursor (dataloader in ``loader/``; step, rng and
  meters in ``extra_state/``), written per rank by
  :class:`~veomni.trainer.callbacks.global_state_callback.GlobalStateCallback`.
"""

import hashlib
import marshal
import math
import os
import types
import weakref
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR, StepLR

from ...checkpoint import CheckpointerBase, build_checkpointer, layout
from ...utils import helper
from ...utils.dist_utils import raise_if_any_rank_failed


def _scheduler_recipe(scheduler, optimizer):
    """Stable schedule identity; LambdaLR state_dict omits its callable."""
    from ...optim.lr_scheduler import MultiLRScheduler

    if scheduler is None:
        return None
    if type(scheduler) is MultiLRScheduler:
        if set(scheduler) != set(optimizer.optimizers_dict):
            raise RuntimeError("MultiOptimizer scheduler leaves differ")
        return {
            name: _scheduler_recipe(scheduler[name], optimizer.optimizers_dict[name]) for name in sorted(scheduler)
        }
    if getattr(optimizer, "_is_multi_optimizer", False) is True:
        raise RuntimeError("MultiOptimizer requires a scheduler for each optimizer leaf")
    if type(scheduler) not in (LambdaLR, StepLR) or scheduler.optimizer is not optimizer:
        raise RuntimeError("MultiOptimizer scheduler implementation or binding is unsupported")
    if type(scheduler) is StepLR:
        return {"type": "StepLR", "step_size": scheduler.step_size, "gamma": scheduler.gamma}

    def immutable(value):
        if value is None or type(value) in (bool, int, float, str):
            return value
        if isinstance(value, tuple):
            return [immutable(item) for item in value]
        if isinstance(value, dict) and all(isinstance(key, str) for key in value):
            return {key: immutable(item) for key, item in sorted(value.items())}
        raise RuntimeError("MultiOptimizer scheduler captures unsupported mutable state")

    def normalize(code):
        constants = tuple(normalize(item) if isinstance(item, types.CodeType) else item for item in code.co_consts)
        return code.replace(co_filename="", co_firstlineno=0, co_consts=constants)

    functions = []
    for function in scheduler.lr_lambdas:
        if not isinstance(function, types.FunctionType):
            raise RuntimeError("MultiOptimizer scheduler callable is unsupported")
        if any(
            name in function.__globals__ and (name != "math" or function.__globals__[name] is not math)
            for name in function.__code__.co_names
        ):
            raise RuntimeError("MultiOptimizer scheduler depends on unsupported global state")
        functions.append(
            {
                "code": hashlib.sha256(marshal.dumps(normalize(function.__code__))).hexdigest(),
                "defaults": immutable(function.__defaults__),
                "kwdefaults": immutable(function.__kwdefaults__),
                "closure": [immutable(cell.cell_contents) for cell in function.__closure__ or ()],
            }
        )
    return {"type": "LambdaLR", "functions": functions}


def _validate_scheduler_checkpoint(saved, current, global_step):
    """Require every scheduler leaf and its step to match a fresh runtime."""

    def validate(left, right):
        if not isinstance(left, dict) or not isinstance(right, dict) or set(left) != set(right):
            raise RuntimeError("MultiOptimizer checkpoint scheduler layout differs")
        found = "last_epoch" in right
        if found:
            initial, restored = right["last_epoch"], left["last_epoch"]
            if type(initial) is not int or type(restored) is not int or restored != initial + global_step:
                raise RuntimeError("MultiOptimizer checkpoint scheduler step differs from trainer step")
        if "_step_count" in right:
            initial_count, saved_count = right["_step_count"], left["_step_count"]
            if (
                type(initial_count) is not int
                or type(saved_count) is not int
                or saved_count != initial_count + global_step
            ):
                raise RuntimeError("MultiOptimizer checkpoint scheduler call count differs from trainer step")
        for name, value in right.items():
            if isinstance(value, dict):
                child_found = validate(left[name], value)
                if "last_epoch" not in right and not child_found:
                    raise RuntimeError("MultiOptimizer checkpoint scheduler leaf has no provable step")
                found = child_found or found
            elif isinstance(value, (list, tuple)):
                if not isinstance(left[name], (list, tuple)) or len(left[name]) != len(value):
                    raise RuntimeError("MultiOptimizer checkpoint scheduler state shape differs")
                if name != "_last_lr" and left[name] != value:
                    raise RuntimeError("MultiOptimizer checkpoint scheduler configuration differs")
            elif name not in {"last_epoch", "_step_count", "_get_lr_called_within_step"} and left[name] != value:
                raise RuntimeError("MultiOptimizer checkpoint scheduler configuration differs")
        return found

    if not validate(saved, current):
        raise RuntimeError("MultiOptimizer checkpoint scheduler has no provable step")


def _validate_scheduler_optimizer_lrs(saved, optimizer):
    """A restored scheduler and optimizer must agree on the next-step LR."""
    for name, leaf in optimizer.optimizers_dict.items():
        state = saved.get(name)
        if not isinstance(state, dict) or not isinstance(state.get("_last_lr"), (list, tuple)):
            raise RuntimeError("MultiOptimizer checkpoint scheduler LR state is missing")
        last_lrs = state["_last_lr"]
        if len(last_lrs) != len(leaf.param_groups):
            raise RuntimeError("MultiOptimizer checkpoint scheduler LR groups differ")
        for value, group in zip(last_lrs, leaf.param_groups, strict=True):
            actual = group["lr"]
            if hasattr(value, "numel") and value.numel() == 1:
                value = value.item()
            if hasattr(actual, "numel") and actual.numel() == 1:
                actual = actual.item()
            if type(value) not in (int, float) or type(actual) not in (int, float) or value != actual:
                raise RuntimeError("MultiOptimizer checkpoint scheduler and optimizer LRs differ")


if TYPE_CHECKING:
    from ...arguments import CheckpointConfig
    from ...trainer.callbacks import TrainerState
    from ..model_runtime import VeOmniModelRuntime


logger = helper.create_logger(__name__)


class ModelCheckpointManager:
    """Own DCP / HF / LoRA save-load for one model runtime.

    The runtime supplies the module, optimizer, scheduler, assets and mesh; this
    class owns the *ordering* around them — when to drain an in-flight async
    save, where the ``empty_cache`` and ``barrier`` calls go, and which
    directory each artifact lands in.

    That ordering is load-bearing rather than incidental. The two ``empty_cache``
    calls bracketing a DCP save keep the save from competing with the training
    step for HBM: without the pre-save one, DCP's NCCL gather buffers can fail to
    allocate (seen as ``NCCL WARN Cuda failure 2 'out of memory'`` inside
    ``dcp.save`` on a Qwen3.5-35B-a3b VL h100x16 run).

    On-disk layout for a single-model job::

        <save_path>/global_step_{N}/
        ├── model/
        │   ├── ckpt/          # DCP shards: weights
        │   ├── optimizer/     # DCP shards: optimizer state
        │   └── lr_scheduler.pt
        ├── loader/            # dataloader cursor, per rank
        ├── extra_state/       # step, rng, meters, per rank
        ├── hf_ckpt/           # full-model HF safetensors export
        └── lora_ckpt/         # PEFT adapter export

    ``loader/`` and ``extra_state/`` are written by
    :class:`~veomni.trainer.callbacks.global_state_callback.GlobalStateCallback`,
    which also writes the step's ``checkpoint_manifest.json``.

    Paths are never built here — they all come from
    :mod:`veomni.checkpoint.layout`, so a save and the load that follows it
    cannot drift apart.
    """

    def __init__(self, runtime: "VeOmniModelRuntime"):
        self.runtime = runtime
        self.config: "CheckpointConfig" = runtime.train_args.checkpoint
        self._last_saved_step: int = -1
        self._loaded_optimizer_global_step = None
        self._loaded_optimizer_generation = None
        self._last_saved_optimizer_generation = None
        self._last_saved_step_path = None
        self._saved_step_paths = []
        self._invalidated_saved_step_paths = []
        self._last_legacy_export_dcp_step = None
        self._last_saved_optimizer_ref = None
        self._loaded_optimizer_ref = None
        self._loaded_checkpoint_path = None
        self._loaded_checkpointer = None
        self._last_saved_without_optimizer = False
        self._last_export_validated_step = None
        self._last_export_released_optimizer = False
        self.checkpointer: CheckpointerBase = build_checkpointer(
            ckpt_manager=self.config.manager,
            dist_backend=runtime.args.accelerator.fsdp_config.fsdp_mode,
        )
        if getattr(self.config, "save_hf_weights", False) and not bool(
            getattr(self.runtime.args, "lora_config", None)
        ):
            from ...utils.save_safetensor_utils import should_use_distributed_hf_safetensor_save

            if should_use_distributed_hf_safetensor_save(getattr(self.config, "manager", None)):
                from .convert import get_checkpoint_tensor_converter

                model = getattr(runtime, "model", None)
                converter = get_checkpoint_tensor_converter(model) if model is not None else None
                validate_export = getattr(converter, "validate_export_support", None)
                if callable(validate_export):
                    validate_export(model, fqn_to_index_mapping=getattr(runtime.args, "fqn_to_index_mapping", None))

    @property
    def parallel_state(self):
        """This model's mesh, via the runtime's by-name registry lookup.

        Not the ambient ``get_parallel_state()``: ``build_checkpoint()`` runs
        outside the runtime's ``use_parallel_state`` scope, so ambient is still
        ``"base"`` while a DPO policy or an Omni module is registered under its
        own name. A property rather than a cached object, so a re-registered
        mesh is picked up the same way ``VeOmniModelRuntime.parallel_state`` is.
        """
        return self.runtime.parallel_state

    @property
    def last_saved_step(self) -> int:
        """Last step this run handed to the checkpointer, for same-step dedupe.

        Says nothing about what is on disk -- under ``save_async`` the shards are
        still being written when ``save_dcp`` returns. It answers the one
        question the filesystem cannot: whether *this* run wrote the step. A
        complete checkpoint left at the same step by an earlier run looks
        identical from the outside but holds different weights, so
        :meth:`prepare_export` overwrites rather than trusting what it finds.
        """
        return self._last_saved_step

    @property
    def trainable_only(self) -> bool:
        return bool(self.runtime.args.lora_config)

    def step_dir(self, state: "TrainerState") -> str:
        """Root of this step's checkpoint, shared by every module of the job."""
        return layout.step_dir(self.config.save_path, state.global_step)

    def save_dir(self, state: "TrainerState") -> str:
        """Where this step's model state lives: weights, optimizer, scheduler."""
        return layout.model_dir(self.step_dir(state))

    def weights_dir(self, state: "TrainerState") -> str:
        """DCP directory holding this step's weights alone."""
        return layout.weights_dir(self.step_dir(state))

    def hf_export_dir(self, state: "TrainerState") -> str:
        """Where this step's full-model safetensors export lives."""
        return layout.hf_export_dir(self.step_dir(state))

    def lora_export_dir(self, state: "TrainerState") -> str:
        """Where this step's PEFT adapter export lives."""
        return layout.lora_export_dir(self.step_dir(state))

    def assets_dir(self) -> str:
        """Where this model's config/tokenizer/processor sidecars live.

        Once per run, at the output root — not inside a step.
        """
        return layout.assets_dir(self.config.model_assets_dir)

    def load_dir(self) -> Optional[str]:
        """Step directory to resume from."""
        return self.config.load_path

    def _checkpointer_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments shared by ``checkpointer.save`` and ``checkpointer.load``."""
        return {
            "trainable_only": self.trainable_only,
            "parallel_state": self.parallel_state,
        }

    def _invalidate_saved_receipt(self) -> None:
        self._last_saved_step = -1
        self._last_saved_step_path = None
        self._last_legacy_export_dcp_step = None
        self._last_saved_optimizer_ref = None
        self._last_saved_without_optimizer = False
        self._last_export_validated_step = None
        self._last_export_released_optimizer = False
        if hasattr(self, "_last_saved_optimizer_generation"):
            self._last_saved_optimizer_generation = None

    def wait_for_pending_save(self) -> None:
        """Block until the in-flight async save is on disk, if there is one."""
        try:
            self.checkpointer.wait_for_pending_save()
        except BaseException:
            self._invalidate_saved_receipt()
            raise

    def _extra_state(self, state: "TrainerState") -> Dict[str, Any]:
        """Model-bound state to store beside the weights.

        The lr scheduler plus whatever the runtime contributes via
        ``extra_state()`` (e.g. the DiT condition model's generator).
        """
        lr_scheduler = self.runtime.lr_scheduler
        extra_state = {
            "lr_scheduler": None if lr_scheduler is None else lr_scheduler.state_dict(),
        }
        extra_state.update(self.runtime.extra_state())
        if getattr(self.runtime.optimizer, "_is_multi_optimizer", False) is True:
            if any(key in extra_state for key in ("_veomni_optimizer_global_step", "_veomni_scheduler_recipe")):
                raise RuntimeError("Reserved MultiOptimizer checkpoint step key is already in use")
            extra_state["_veomni_optimizer_global_step"] = int(state.global_step)
            try:
                extra_state["_veomni_scheduler_recipe"] = {
                    "status": "supported",
                    "value": _scheduler_recipe(lr_scheduler, self.runtime.optimizer),
                }
            except RuntimeError as exc:
                extra_state["_veomni_scheduler_recipe"] = {"status": "unsupported", "reason": str(exc)}
        return extra_state

    def _load_extra_state(self, extra_state: Dict[str, Any]) -> None:
        lr_state = extra_state.get("lr_scheduler")
        lr_scheduler = self.runtime.lr_scheduler
        if lr_state is not None and lr_scheduler is not None:
            lr_scheduler.load_state_dict(lr_state)

        self.runtime.load_extra_state(extra_state)

    def load(self) -> None:
        """Restore model, optimizer and model-bound extra state from ``load_path``."""
        self._invalidated_saved_step_paths.extend(self._saved_step_paths)
        self._saved_step_paths.clear()
        self._invalidate_saved_receipt()
        self._loaded_optimizer_ref = None
        self._loaded_checkpoint_path = None
        self._loaded_checkpointer = None
        self._loaded_optimizer_global_step = None
        self._loaded_optimizer_generation = None
        load_dir, error = None, None
        try:
            load_dir = self.load_dir()
            if load_dir is not None:
                load_dir = os.fspath(load_dir)
                if not isinstance(load_dir, str):
                    raise TypeError("Checkpoint load path must be a string")
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "preparing checkpoint load path")
        if load_dir is None:
            return

        optimizer, scheduler, checkpointer, state, kwargs, error = None, None, None, None, None, None
        try:
            optimizer = self.runtime.optimizer
            scheduler = self.runtime.lr_scheduler
            checkpointer = self.checkpointer
            state = {
                "model": self.runtime.model,
                "optimizer": optimizer,
                "extra_state": {},
                "_veomni_scheduler_object": scheduler,
            }
            kwargs = self._checkpointer_kwargs()
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "preparing checkpoint load arguments")
        error = None
        try:
            self.wait_for_pending_save()
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "waiting before checkpoint load")
        multi = getattr(optimizer, "_is_multi_optimizer", False) is True
        step = generation = None
        error = None
        try:
            checkpointer.load(load_dir, state, **kwargs)
            if multi:
                if state.get("_veomni_optimizer_identity_verified") is not True:
                    raise RuntimeError("MultiOptimizer checkpoint identity has not been proven")
                step = state["extra_state"].pop("_veomni_optimizer_global_step", None)
                if type(step) is not int or step < 0:
                    raise RuntimeError("MultiOptimizer checkpoint is missing its proven trainer global step")
                recipe = state["extra_state"].pop("_veomni_scheduler_recipe", None)
                if (
                    not isinstance(recipe, dict)
                    or recipe.get("status") != "supported"
                    or set(recipe) != {"status", "value"}
                ):
                    raise RuntimeError("MultiOptimizer checkpoint scheduler identity is missing or unsupported")
                if recipe["value"] != _scheduler_recipe(scheduler, optimizer):
                    raise RuntimeError("MultiOptimizer checkpoint scheduler identity differs")
                saved_scheduler = state["extra_state"].get("lr_scheduler")
                if (scheduler is None) != (saved_scheduler is None) or (
                    scheduler is not None and not isinstance(saved_scheduler, dict)
                ):
                    raise RuntimeError("MultiOptimizer checkpoint scheduler state is missing or incompatible")
                if scheduler is not None:
                    _validate_scheduler_checkpoint(saved_scheduler, scheduler.state_dict(), step)
                    _validate_scheduler_optimizer_lrs(saved_scheduler, optimizer)
                generation = state.get("_veomni_optimizer_checkpoint_generation")
                if not isinstance(generation, str) or not generation:
                    raise RuntimeError("MultiOptimizer checkpoint generation is missing")
            self._load_extra_state(state["extra_state"])
            if self.runtime.optimizer is not optimizer or self.runtime.lr_scheduler is not scheduler:
                raise RuntimeError("Optimizer or scheduler changed while restoring the checkpoint")
            if self.checkpointer is not checkpointer:
                raise RuntimeError("Checkpoint provider changed while restoring the checkpoint")
        except BaseException as exc:
            error = exc

        def abort_file_backed():
            leaves = getattr(optimizer, "optimizers_dict", {})
            for leaf in [optimizer, *(leaves.values() if isinstance(leaves, dict) else ())]:
                abort = getattr(leaf, "checkpoint_abort_file_backed_state", None)
                if callable(abort):
                    try:
                        abort()
                    except Exception:
                        pass

        try:
            raise_if_any_rank_failed(error, "checking restored optimizer identity")
            dist.barrier()
        except BaseException:
            abort_file_backed()
            raise
        if multi:
            self._loaded_optimizer_global_step = step
            self._loaded_optimizer_generation = generation
        if optimizer is not None:
            try:
                self._loaded_optimizer_ref = weakref.ref(optimizer)
            except TypeError:
                self._loaded_optimizer_ref = None
        self._loaded_checkpoint_path = load_dir
        self._loaded_checkpointer = checkpointer
        logger.info_rank0(f"Load distributed checkpoint from {load_dir} successfully!")

    def save_dcp(self, state: "TrainerState") -> None:
        """Write model, optimizer and model-bound extra state for this step."""
        self._invalidate_saved_receipt()
        optimizer, optimizer_ref, error, extra_state, kwargs, step_path = None, None, None, None, None, None
        try:
            step_path = self.step_dir(state)
            if any(layout.same_checkpoint_path(step_path, path) for path in self._invalidated_saved_step_paths):
                raise RuntimeError("Cannot overwrite a checkpoint after loading without a new verified receipt")
            optimizer = self.runtime.optimizer
            try:
                optimizer_ref = weakref.ref(optimizer) if optimizer is not None else None
            except TypeError:
                optimizer_ref = None
            extra_state = self._extra_state(state)
            kwargs = self._checkpointer_kwargs()
            helper.empty_cache()
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "preparing checkpoint save arguments")
        multi = getattr(optimizer, "_is_multi_optimizer", False) is True
        generation = self.checkpointer.save(
            self.config.save_path,
            {
                "model": self.runtime.model,
                "optimizer": optimizer,
                "extra_state": extra_state,
                "_veomni_scheduler_object": self.runtime.lr_scheduler,
            },
            global_steps=state.global_step,
            save_async=self.config.save_async,
            save_to_lowest_rank=self.config.dcp_save_to_lowest_rank,
            stage_dir=self.config.stage_dir,
            save_timeout_seconds=self.config.save_timeout_seconds,
            **kwargs,
        )
        error = None
        try:
            if multi and (not isinstance(generation, str) or not generation):
                raise RuntimeError("MultiOptimizer save did not return its checkpoint generation")
            helper.empty_cache()
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "checking saved checkpoint generation")
        dist.barrier()
        if multi:
            self._last_saved_optimizer_generation = generation
        self._last_saved_step = state.global_step
        self._last_saved_step_path = step_path
        self._saved_step_paths.append(step_path)
        self._last_saved_optimizer_ref = optimizer_ref
        self._last_saved_without_optimizer = optimizer is None
        logger.info_rank0(f"Distributed checkpoint saved at {self.save_dir(state)} successfully!")

    def prepare_export(self, state: "TrainerState", stage: str) -> str:
        """Make sure this step's DCP exists, then return its weights directory.

        Returns the weights directory rather than ``save_dir`` because that is
        what the legacy (non-distributed) export path converts from, and it now
        holds the weights alone.

        The DCP written here can land on a step the save cadence never reaches.
        ``GlobalStateCallback`` finishes such a step off with the cursor files
        and the manifest, so it resumes like any other.
        """
        self._last_legacy_export_dcp_step = None
        self._last_export_validated_step = None
        self._last_export_released_optimizer = False
        error = None
        try:
            if self._last_saved_step != state.global_step:
                dist.barrier()
                self.save_dcp(state)
            self.wait_for_pending_save()
        except BaseException as exc:
            # Never retain a same-step receipt after a real save or wait failure.
            self._invalidate_saved_receipt()
            error = exc
        try:
            raise_if_any_rank_failed(error, "preparing the exported checkpoint")
        except BaseException:
            self._invalidate_saved_receipt()
            raise

        if self._last_saved_step == state.global_step:
            self._last_export_validated_step = state.global_step
            optimizer = self.runtime.optimizer
            if optimizer is not None and getattr(optimizer, "_is_multi_optimizer", False) is False:
                self._last_legacy_export_dcp_step = state.global_step

        if stage == "train_end":
            self._last_export_released_optimizer = self._last_export_validated_step == state.global_step
            self.runtime.optimizer = None
            self.runtime.lr_scheduler = None

        return self.weights_dir(state)

    def checkpoint_receipt_is_current(self, step: int) -> bool:
        """Check that a same-step DCP still represents this optimizer."""
        if self._last_saved_step != step or getattr(self.checkpointer, "_async_save_failed", False) is True:
            return False
        optimizer = self.runtime.optimizer
        if optimizer is None:
            return self._last_saved_without_optimizer or self._last_export_released_optimizer
        return self._last_saved_optimizer_ref is not None and self._last_saved_optimizer_ref() is optimizer

    def confirm_loaded_checkpoint_for_reuse(self, step: int, generation: str) -> None:
        """Bind a proven same-output resume to the freshly restored optimizer."""
        optimizer = self.runtime.optimizer
        if (
            type(step) is not int
            or not isinstance(generation, str)
            or not generation
            or getattr(self, "_loaded_optimizer_global_step", None) != step
            or getattr(self, "_loaded_optimizer_generation", None) != generation
            or getattr(optimizer, "_is_multi_optimizer", False) is not True
            or self._loaded_optimizer_ref is None
            or self._loaded_optimizer_ref() is not optimizer
            or (self._invalidated_saved_step_paths and self._loaded_checkpoint_path is None)
            or getattr(self.checkpointer, "_async_save_failed", False) is True
        ):
            raise RuntimeError("Cannot reuse an unproven optimizer checkpoint receipt")
        try:
            optimizer_ref = weakref.ref(optimizer)
        except TypeError as exc:
            raise RuntimeError("Restored optimizer cannot be bound to the checkpoint receipt") from exc
        self._last_saved_step = step
        self._last_saved_step_path = self._loaded_checkpoint_path
        if self._loaded_checkpoint_path is not None:
            self._saved_step_paths.append(self._loaded_checkpoint_path)
        self._invalidated_saved_step_paths = [
            path
            for path in self._invalidated_saved_step_paths
            if not layout.same_checkpoint_path(path, self._loaded_checkpoint_path)
        ]
        self._last_saved_optimizer_ref = optimizer_ref
        self._last_saved_without_optimizer = False
        self._last_saved_optimizer_generation = generation
        self._last_legacy_export_dcp_step = None
        self._last_export_validated_step = None
        self._last_export_released_optimizer = False

    def confirm_legacy_loaded_checkpoint_for_reuse(self, step: int, load_path: str) -> None:
        """Bind a legacy DCP only after its optimizer and trainer cursor restore."""
        from ...checkpoint.dcp_checkpointer import DistributedCheckpointer

        optimizer = self.runtime.optimizer
        if (
            type(step) is not int
            or step < 0
            or optimizer is None
            or getattr(optimizer, "_is_multi_optimizer", False) is not False
            or self._loaded_optimizer_ref is None
            or self._loaded_optimizer_ref() is not optimizer
            or self._loaded_checkpoint_path is None
            or not layout.same_checkpoint_path(self._loaded_checkpoint_path, load_path)
            or self._loaded_checkpointer is not DistributedCheckpointer
            or self.checkpointer is not self._loaded_checkpointer
            or getattr(self.checkpointer, "_async_save_failed", False) is True
        ):
            raise RuntimeError("Cannot reuse an unproven legacy optimizer checkpoint receipt")
        self._last_saved_step = step
        self._last_saved_step_path = load_path
        self._saved_step_paths.append(load_path)
        self._invalidated_saved_step_paths = [
            path for path in self._invalidated_saved_step_paths if not layout.same_checkpoint_path(path, load_path)
        ]
        self._last_saved_optimizer_ref = self._loaded_optimizer_ref
        self._last_saved_without_optimizer = False
        self._last_legacy_export_dcp_step = step
        self._last_export_validated_step = step
        self._last_export_released_optimizer = False

    def save_hf(self, state: "TrainerState", stage: str = "step_end") -> None:
        from ...utils.save_safetensor_utils import save_hf_safetensor

        weights_path = self.prepare_export(state, stage)

        save_hf_safetensor(
            save_hf_safetensor_path=self.hf_export_dir(state),
            model_assets=self.runtime.model_assets,
            ckpt_manager=self.config.manager,
            output_dir=self.config.output_dir,
            save_checkpoint_path=weights_path,
            model=self.runtime.model,
            fqn_to_index_mapping=self.runtime.args.fqn_to_index_mapping,
            is_rank_0=self.parallel_state.global_rank == 0,
            parallel_state=self.parallel_state,
        )
        helper.empty_cache()
        dist.barrier()

    def save_lora(self, state: "TrainerState", stage: str = "step_end", adapter_name: str = "default") -> None:
        from ...utils.save_safetensor_utils import save_lora_adapter_with_dcp

        self.prepare_export(state, stage)
        save_lora_adapter_with_dcp(
            model=self.runtime.model,
            save_path=self.lora_export_dir(state),
            adapter_name=adapter_name,
            parallel_state=self.parallel_state,
        )
        helper.empty_cache()
        dist.barrier()

    def save_hf_or_lora(self, state: "TrainerState", stage: str = "step_end") -> None:
        if self.trainable_only:
            self.save_lora(state, stage=stage)
        else:
            self.save_hf(state, stage=stage)


__all__ = ["ModelCheckpointManager"]
