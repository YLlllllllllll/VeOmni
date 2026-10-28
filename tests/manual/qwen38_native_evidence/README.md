# Qwen3.8 native unit evidence

These scripts reproduce two synthetic distributed checks for integration commit
`e26bf6d0b28a4a284fc5797513237a16ee374796` (containing #1006 head
`9ad2c2f0ec2299f193621936a78f1de7c2d612c1`, #1378 head
`a4a618059599113f93b8f572b32b7ac4d10ec446`, and hybrid CP support).
Only the scripts in this directory are added on top of that integration commit;
no training implementation is changed. The `.py.txt` files preserve the exact
executed byte contents and can be passed directly to Python. They are archival
fixtures on an evidence-only branch, not CI-discovered Python tests; preserving
the executed bytes takes precedence over applying the repository formatter.

Run from the repository root in a compatible VeOmni Ascend environment:

```bash
PYTHONPATH="$PWD:$PYTHONPATH" OMP_NUM_THREADS=1 \
  python -m torch.distributed.run --standalone --nproc-per-node=16 \
  tests/manual/qwen38_native_evidence/channel_collator_unit.py.txt \
  --backend npu --ulysses 8 --cp 2

PYTHONPATH="$PWD:$PYTHONPATH" OMP_NUM_THREADS=1 \
  python -m torch.distributed.run --standalone --nproc-per-node=16 \
  tests/manual/qwen38_native_evidence/dpo_sp_unit.py.txt \
  --backend npu --ulysses 4 --cp 2
```

Both passed on one host with 16 Ascend 910B2C NPUs, CANN 9.0.0,
PyTorch `2.10.0+cpu`, torch_npu `2.10.0.post2`, and HCCL. The private
runtime image digest was
`sha256:b12ca400100c5cb1ab9cb490fd21dd60fab3d9bd2ee87e935f6c8aedff9f7014`;
the image itself is not publicly available. Sanitized terminal markers:

```text
COLLATOR_UNIT_PASS {"world":16,"ulysses":8,"cp":2,"source_counts":[2,20,36],"loss":3.516782760620117}
DPO_UNIT_PASS {"world":16,"dp":2,"ulysses":4,"cp":2,"pair_counts":[1,2,1,3],"loss":0.8218554258346558,"gradient":-0.1961914300918579}
```

The script SHA-256 digests are `8c72773dab2c2aae88accdb1e21f6a5b0ce83400340644ba016064f446603447`
(collator) and `884176f47739fc7766a82e4cbe75109798ac8ac6a6ab572a54e8923370fb2e35`
(DPO).
The first uses the actual `MainCollator` and checks packed provenance,
fixed and automatic padding, input and attention boundaries, per-source
statistics, FP32 eager cross-entropy and gradients against an independent
reference. The second uses a scalar toy DPO policy and checks pair-weighted
loss and gradients with unequal pair counts across a DP2/U4/CP2 mesh.

These checks do not validate a full model, BF16 model arithmetic, EP expert
gradients, a real optimizer update, throughput, DCP save, or cold recovery.
They are not a substitute for the repository's CI and end-to-end tests.
