# Qwen3 DPO training guide

DPO training with **Qwen3-0.6B** on the
[trl-lib/ultrafeedback_binarized](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized) dataset.

Config: [`configs/text/qwen3_dpo.yaml`](../../configs/text/qwen3_dpo.yaml)

---

## Step 1 — Prepare the dataset

```shell
python scripts/download_hf_data.py \
    --repo_id trl-lib/ultrafeedback_binarized \
    --local_dir ./ultrafeedback_binarized \
    --allow_patterns "*.parquet"
```

This downloads all train and test Parquet files from the Hub into `./ultrafeedback_binarized/`.

---

## Step 2 — Prepare the model

```shell
python scripts/download_hf_model.py \
    --repo_id Qwen/Qwen3-0.6B \
    --local_dir .
```

The script appends the model name to `--local_dir`, saving to `./Qwen3-0.6B`.

---

## Step 3 — Start DPO training

```shell
bash train.sh tasks/train_text_dpo.py configs/text/qwen3_dpo.yaml
```

Key config values (full DPO argument reference: [arguments.md — DPOConfig](../usage/arguments.md#dpoconfig)):

| Parameter | Value |
|---|---|
| `model.model_path` | `./Qwen3-0.6B` |
| `data.train_path` | `./ultrafeedback_binarized` |
| `data.max_seq_len` | `2048` |
| `train.global_batch_size` | `8` |
| `train.micro_batch_size` | `1` |
| `train.num_train_epochs` | `1` |
| `dpo_config.beta` | `0.1` |
| `dpo_config.loss_type` | `sigmoid` |
| frozen reference | copies `model`; custom `reference_model` config is not supported |
| `train.checkpoint.output_dir` | `Qwen3-0.6B-dpo-ultrafeedback` |
| `train.wandb.project` | `VeOmni` |
| `train.wandb.name` | `Qwen3-0.6B-dpo-ultrafeedback` |

---

## Step 4 — Monitor training

Training outputs (DPO loss, chosen/rejected rewards, reward accuracy, grad norm) are printed every
step and logged to Weights & Biases.

---

## Checkpoints

Checkpoints are saved under `train.checkpoint.output_dir` every `save_steps` steps.
With `save_hf_weights: true`, a HuggingFace-compatible checkpoint is also written:

```
Qwen3-0.6B-dpo-ultrafeedback/
└── checkpoints/
    └── global_step_200/
        ├── checkpoint_manifest.json ← written last; marks the step resumable
        ├── model/
        │   ├── ckpt/                ← DCP shards: weights
        │   ├── optimizer/           ← DCP shards: optimizer state
        │   └── lr_scheduler.pt
        ├── loader/rank_{R}.pt       ← dataloader cursor
        ├── extra_state/rank_{R}.pt  ← step, rng, meters
        └── hf_ckpt/                 ← HuggingFace safetensors (when save_hf_weights)
```

File-by-file contract: [Checkpoint layout](../usage/checkpoint.md).

Each optimizer step uses the global mean over preference pairs, including when
microbatches or data-parallel ranks contain different numbers of pairs. The
reported loss and reward metrics use the same pair weights. Sequence-parallel
ranks must have the same segment boundaries, and all data-parallel ranks must
execute the same number of microbatches for an optimizer step. For full input
identity checking across sequence-parallel ranks, enable the experimental
provenance mode described below. Validate expert and dense gradients separately
on the target EP/HSDP runtime; their distributed reduction implementations
can differ. Callback-internal collective failures still rely on the launcher's
failure handling.
Without provenance, legacy tail-padding inference can mistake a real final
one-token segment for padding; use the proven mode or supply unambiguous tail
metadata before relying on its pair count.

Custom DPO trainers overriding `forward_backward_step` must explicitly declare
the `pair_normalization` keyword and apply it to the sum of per-pair losses before
backward. A catch-all `**kwargs` alone is insufficient because it can silently
discard the scale. An old override without that argument is rejected before model forward;
using its unweighted microbatch mean would silently change the objective.

For standard DPO packing, the collator now records an exact zero tail when it
would otherwise be confused with a final one-token segment, and a known nonzero
collator tail is excluded from loss and gradients. Custom collators must provide
an exact `tail_padding_length`; if it is missing and the final packed segment is
one token, the trainer rejects the batch rather than guessing. Enable explicit
segment provenance below to validate producer segment boundaries and SP sample
identity in addition to the collator tail. Without it, legacy SP validation
checks packed layout and tensor shapes but cannot establish that ranks received
the same sample contents. DPO preflight validates integer token IDs against the
policy model's configured text vocabulary before running the model.

Checkpoints now record the objective normalization. A trainer-state checkpoint
from the earlier per-microbatch normalization is rejected by default; use model
weights only for a clean comparison, or explicitly set
`dpo_config.allow_legacy_objective_resume: true` to retain the old optimizer and
scheduler while switching to the corrected normalization. That continuation is
not numerically equivalent to the old trajectory. A checkpoint without a data
schema also requires the separate `allow_legacy_data_resume` option and has the
additional restrictions below.

## Experimental: explicit segment provenance

Set `dpo_config.explicit_segment_provenance: true` to retain the physical
chosen/rejected lengths produced by the built-in DPO transform. This distinguishes
a real one-token response from padding appended by the standard collator and
attributes packed source-token counts per original DPO example. It does not
infer or remove padding already present in input examples.

The option defaults to `false`; the plaintext path uses tokenizer character
offsets to mask prompt-only tokens, including when BPE merges cross the boundary.
If offsets are unavailable, an exact token-prefix boundary is required.
When a fast tokenizer returns a zero-width token at the prompt boundary,
the boundary cannot be attributed safely; use untrimmed offsets or a
tokenizer that provides an unambiguous tokenization.
A plaintext pair with a non-empty prompt is rejected when its prompt leaves
either completion with no scoreable tokens after truncation; shorten the prompt
or increase `max_seq_len`. With an empty prompt, a one-token segment retains
the existing zero-log-probability behavior because no causal target exists.
Enable the option
with the same value on every rank, using the built-in DPO transform and standard
`MainCollator`. Each step validates policy/reference SP membership and a small
fingerprint of each SP group's transformed tensors and source IDs before model
execution. Identical transformed inputs may originate from different raw
records; this check does not certify raw dataset identity or cursor alignment. Unproven or
inconsistent samples are rejected on all ranks before model forward; with
multisource loading, every example must have one ordered integer `ds_idx`.
Source names are checked against their IDs only when the loader supplies them.
Token and position tensors must use 32- or 64-bit integers, the packed attention
mask must contain only ones, and input/label token IDs must fit the policy model's
configured text vocabulary. Pre-padded attention masks containing zeros require a
separately validated packed unpadding implementation and are rejected here.
This experimental path accepts only the built-in text tensors, `cur_token_num`,
and configured source metadata on input features; unknown extra feature keys are
rejected. Source and bookkeeping metadata are never sent to the model in this
mode. Callbacks may remove these fields but may not add or mutate them.
Local packed tensors and channel-loss metadata are fingerprinted at preflight
to detect mutation after collation. Model inputs are checked again after
step-begin callbacks; callbacks should transform data before collation instead.
Input validation and ordinary rank-local callback exceptions after callbacks
return are coordinated before model execution. Failures inside a callback's own
collective still rely on the launcher's failure handler or timeout.
Use the standard trainer loop: direct `concatenated_forward` calls require a
microbatch already approved by both rank-agreement phases. The provenance path
uses three WORLD-level and one FSDP-group object collective per step; measure
their overhead and validate the target hardware before using the option for production. CPU/Gloo tests do not establish
Ascend HCCL behavior, full-model numerical equivalence, or checkpoint recovery.

Resume only with the same data-semantics version and provenance option recorded
in trainer state. An old or missing trainer state is rejected by default in both
modes: plaintext prompt-masking semantics changed, so a legacy loader buffer can
mix old labels with newly transformed labels. Start a fresh run from model weights
without restoring optimizer, scheduler, or loader state for a consistent objective.
When provenance is disabled, setting both `dpo_config.allow_legacy_data_resume: true`
and `dpo_config.allow_legacy_objective_resume: true` explicitly allows an
unversioned legacy checkpoint, but the continuation is not exact and may
mix buffered labels with new labels and existing optimizer/scheduler state. This
escape hatch does not permit changing provenance mode or bypass a versioned,
incompatible protocol; an enabled run cannot use it.
