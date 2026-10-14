# Contributing

Use [GitHub Issues](https://github.com/Hxnng/Agentic-Soft-Token/issues) for reproducibility questions and bug reports. Include the commit, Python version, `stable-interfaces doctor` output, the failing command, and a small example. Remove credentials and private task/tool content from reports.

## Development setup

Use Python 3.11 or 3.12. For the CPU reference tests on Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
pytest -q -m 'not cuda'
stable-interfaces inspect
stable-interfaces table --reference
```

On macOS, install `torch==2.7.0` from the default PyPI index instead of the Linux CPU index. CPU tests use small reference models; full 9B inference requires the pinned CUDA stack in the [README](README.md).

## Check a change

Keep published thresholds, task order, seeds, calibration, and metric definitions fixed unless the change explicitly introduces a new experiment. Changes to pooling, signal extraction, or caching should include a focused regression check and preserve the native Base path.

Run the existing CPU suite and verify both package formats:

```bash
ruff check src tests tools
ruff format --check src tests tools
pytest -q -m 'not cuda'
python -m build
```

For runtime changes, also run `pytest -q -m cuda` and the real-model smoke check on a CUDA host:

```bash
stable-interfaces smoke --model /path/to/Qwen3.5-9B --output runs/smoke-check
```

The [CI workflow](.github/workflows/ci.yml) runs CPU tests on Python 3.11/3.12, validates the calibration and reference table, builds the source distribution and wheel, and checks the installed wheel outside the checkout. CUDA checks require a separate GPU environment.

Do not commit model weights, datasets, raw trajectories, environment files, or credentials. Keep generated runs and local environments in the ignored directories. Contributions to the original code follow the existing [MIT License](LICENSE).
