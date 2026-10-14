"""Portable entry points for Stage-3 compression and the uncompressed baseline."""

import argparse
import importlib.metadata
import json
from pathlib import Path


def doctor():
    import torch

    packages = {}
    for name in (
        "torch",
        "transformers",
        "flash-linear-attention",
        "safetensors",
        "accelerate",
        "appworld",
        "causal-conv1d",
        "triton",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="report installed dependency versions")
    inspect = sub.add_parser("inspect", help="validate bundled calibration and Stage-3 policy")
    inspect.add_argument("--bundle", type=Path)
    inspect.add_argument("--config", type=Path)
    evaluation = sub.add_parser(
        "evaluate", help="run official test-normal tasks with fresh task caches"
    )
    evaluation.add_argument("--model", required=True, type=Path)
    evaluation.add_argument("--appworld-root", required=True, type=Path)
    evaluation.add_argument("--environment-url", required=True)
    evaluation.add_argument("--output", required=True, type=Path)
    evaluation.add_argument(
        "--variant", choices=("base", "stable_interfaces"), default="stable_interfaces"
    )
    evaluation.add_argument("--bundle", type=Path)
    evaluation.add_argument("--config", type=Path)
    evaluation.add_argument("--device", default="cuda:0")
    evaluation.add_argument("--tasks", nargs="+")
    evaluation.add_argument("--limit", type=int)
    evaluation.add_argument("--shard-index", type=int, default=0)
    evaluation.add_argument("--num-shards", type=int, default=1)
    smoke = sub.add_parser(
        "smoke", help="check native parity and persistent Stage-3 cache on the real model"
    )
    smoke.add_argument("--model", required=True, type=Path)
    smoke.add_argument("--output", required=True, type=Path)
    smoke.add_argument("--device", default="cuda:0")
    table = sub.add_parser("table", help="print the five agreed main-table metrics")
    table.add_argument(
        "--reference", action="store_true", help="use packaged paired 168-task numeric results"
    )
    table.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="NAME=DIRECTORY",
        help="repeat for methods/shards; one method must be named base",
    )
    table.add_argument(
        "--allow-partial",
        action="store_true",
        help="explicitly allow paired subsets instead of all 168 tasks",
    )
    table.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        print(json.dumps(doctor(), indent=2))
    elif args.command == "inspect":
        from dataclasses import asdict
        from .configuration import load_plan, load_selection
        from .runtime.artifacts import read_bundle

        config, weights, meta = read_bundle(args.bundle)
        print(
            json.dumps(
                {
                    "plan": asdict(load_plan(args.config)),
                    "calibration": meta,
                    "tasks": len(load_selection()["task_ids"]),
                    "stages": config.stages,
                    "stored_scalars": sum(v.numel() for v in weights.values()),
                    "trainable_parameters": 0,
                },
                indent=2,
            )
        )
    elif args.command == "evaluate":
        from .evaluation import evaluate

        evaluate(args)
    elif args.command == "smoke":
        from .smoke import smoke

        smoke(args)
    elif args.command == "table":
        from .report import report

        report(args)
