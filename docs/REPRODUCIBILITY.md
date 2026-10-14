# Reproduction procedure

## Fixed inputs

- Model: Qwen/Qwen3.5-9B, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Torch 2.7.0+cu126, Transformers 5.2.0, FLA 0.4.1, Triton 3.3.0,
  causal-conv1d 1.7.0.
- AppWorld 0.1.3.post1, with its installed data and an isolated environment server.
- Packaged `assets/test_normal.json`: 168 IDs in the historical order.
- Model initialization seed 20260914; task seed = 20260912 + index in the full
  packaged selection, including when using task subsets or worker sharding.
- Greedy decoding; at most 50 executed actions, 768 generated tokens per action,
  and 32,768 logical context tokens. No context-window extension is claimed.
- All four audited runtime optimizations enabled: zero-output branch elision,
  empty-decode signal bypass, fixed-reader preparation, and intra-forward RoPE reuse.
- FLA autotuning uses the deterministic first-valid policy
  `fla-first-valid-triton33-v1`; each run writes the observed choices to
  `kernel_choices.json`.
- Only Stage 3 is enabled. API catalogs and contracts are protected.

The runtime hashes calibration tensors, checks the model config and exact Agent
protocol, and records code hashes, configuration, seeds, and official task scores.
Numerical differences across hardware/dependency versions can change greedy
trajectories; use the pinned environment for close reproduction.

## Data and environment

Obtain model weights and AppWorld data yourself; neither is redistributed here.
Follow the [official AppWorld guide](https://github.com/StonyBrookNLP/appworld)
for installation, data download, and containerized serving. Pin the client and
server to 0.1.3.post1 for this experiment.

This evaluator uses `remote_docker=True`. The server's current working directory
must be its AppWorld root because the remote protocol can reset the root to `.`.
The helper below changes directory first, binds to loopback, and uses the
official AppWorld environment application:

```bash
# Run this inside an isolated AppWorld runtime with data installed under this root.
python tools/serve_environment.py --root /path/to/appworld --port 37752
```

Use one server per active evaluator. The worker and server must see the same
task/data version.

## Evaluation and sharding

Use the README's two evaluate commands for the uncompressed base and the method.
They share the prompt, model, limits and task seeds. Base disables adapters and
compression and uses the native GDN path; the calibration is only a frozen
loading scaffold for that baseline.

For two workers, start two servers and run separate processes, for example:

```bash
CUDA_VISIBLE_DEVICES=0 stable-interfaces evaluate \
  --model /path/to/Qwen3.5-9B --appworld-root /path/to/appworld \
  --environment-url http://127.0.0.1:37752 --output runs/stable-0 \
  --num-shards 2 --shard-index 0

CUDA_VISIBLE_DEVICES=1 stable-interfaces evaluate \
  --model /path/to/Qwen3.5-9B --appworld-root /path/to/appworld \
  --environment-url http://127.0.0.1:37753 --output runs/stable-1 \
  --num-shards 2 --shard-index 1

stable-interfaces table --run base=runs/base \
  --run stable_interfaces=runs/stable-0 --run stable_interfaces=runs/stable-1
```

Run shard commands concurrently in separate terminals. Use new output
directories. To continue an interrupted study, run the missing task IDs into a
new directory and combine only complete, nonoverlapping shards. Do not count
an infrastructure exception as an official task failure.

## Calibration

The included Stage-3 calibration uses 22 records from 12 Train/Dev tasks.
Normal inference does not require those records or another repository.

The included calibration is loaded directly for evaluation. Its JSON includes
the Train/Dev task IDs and input digest. The portable bundle does not depend
on private raw trajectories.

## Results provenance

The reference CSV contains task IDs and numeric results only, not instructions,
credentials, or Tool text. It contains 168 Base rows and 168 stable_interfaces
rows. It supports exact reproduction of the reported table arithmetic without
rerunning the model. The runtime enforces the published single Stage `[3]`.

The 168 tasks were used during development; they are not a new blind test set.

## Package verification

```bash
pytest -q -m 'not cuda'
pytest -q -m cuda  # needs the CUDA/FLA stack
python -m pip install 'setuptools>=77' wheel
python -m build --no-isolation
```

The wheel includes source, both small asset directories, the calibration, the
prompt, task selection and numeric reference table. No local paths, environment
archives, base-model tensors, raw trajectories or GPU caches are needed by the
installed wheel.
