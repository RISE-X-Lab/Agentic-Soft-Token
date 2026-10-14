"""One sequential worker, one remote AppWorld environment, fresh cache per task."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch
from . import load_model
from .algorithm.stable_contract_merge import stable_scope
from .configuration import DEFAULT_PLAN, DEFAULT_SELECTION, load_plan, load_selection, select_tasks
from .metrics import record_kv, validate_native, task_row, aggregate
from .runtime.adapters import disabled
from .runtime.agent import Actor, RolloutConfig
from .runtime.artifacts import file_sha, read_bundle, weight_digest
from .runtime.io import new_directory, write_json
from .runtime.protocol import Protocol


@torch.no_grad()
def evaluate(args):
    from appworld import AppWorld, load_task_ids, update_root
    from transformers import AutoTokenizer

    selection = load_selection()
    tasks = select_tasks(selection, args.tasks, args.limit, args.shard_index, args.num_shards)
    update_root(str(args.appworld_root))
    if set(load_task_ids("test_normal")) != set(selection["task_ids"]):
        raise ValueError("AppWorld test-normal differs from the packaged 168-task split")
    plan = load_plan(args.config) if args.variant == "stable_interfaces" else None
    native = plan is None
    generation = RolloutConfig(
        max_steps=selection["max_steps"],
        max_action_tokens=selection["max_action_tokens"],
        max_context=selection["max_context"],
        greedy=selection["greedy"],
    )
    _, _, metadata = read_bundle(args.bundle)
    output = new_directory(args.output)
    try:
        write_json(output / "status.json", dict(phase="loading", completed=0, total=len(tasks)))
        from .deterministic_kernels import POLICY, choices, install

        install()
        torch.manual_seed(20260914)
        model = load_model(args.model, bundle=args.bundle, device=args.device)
        if plan is not None and plan.base.data.stages != model.config.stages:
            raise ValueError("plan Stage differs from calibrated bundle Stage")
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        identity = weight_digest(model)
        package = Path(__file__).parent
        write_json(
            output / "run.json",
            dict(
                variant=args.variant,
                tasks=tasks,
                selection=selection,
                selection_sha256=file_sha(DEFAULT_SELECTION),
                plan=None if plan is None else asdict(plan),
                plan_sha256=None if native else file_sha(args.config or DEFAULT_PLAN),
                generation=asdict(generation),
                runtime=asdict(model.runtime),
                model_seed=20260914,
                task_seeds={
                    task: selection["seed"] + selection["task_ids"].index(task) for task in tasks
                },
                bundle_tensor_sha256=metadata["tensor_sha256"],
                model_config_sha256=metadata["model_config_sha256"],
                source_sha256={
                    str(path.relative_to(package)): file_sha(path)
                    for path in sorted(package.rglob("*.py"))
                },
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                gradient_updates=0,
                final_holdout=False,
                device=str(args.device),
                torch_version=torch.__version__,
                gpu=torch.cuda.get_device_name(model.device),
                kernel_policy=POLICY,
                compression_stage=None if native else model.config.stages[0],
            ),
        )
        rows = []
        with disabled(model) if native else stable_scope(model, plan):
            # Warmup is outside task timing and uses a fresh cache.
            from .smoke import fixture

            ids, kwargs = fixture(model.device)
            for _ in range(2):
                result = model(ids, **kwargs, compression=not native, use_cache=True)
                cache = result.past_key_values
                for token in ids[:16]:
                    _, result = model.next_logits(
                        token[None], compression=not native, past_key_values=cache
                    )
                    cache = result.past_key_values
                del cache, result
            for task in tasks:
                actor = Actor(model, tokenizer, Protocol(), generation, compression=not native)
                generate = actor.generate
                progress = dict(phase="rollout", task_id=task, calls=0, generated_tokens=0)

                def tracked(*args, **kwargs):
                    action, metrics = generate(*args, **kwargs)
                    progress["calls"] += 1
                    progress["generated_tokens"] += metrics["generated_tokens"]
                    progress.update(last_stop=metrics["stop"], updated_at=time.time())
                    write_json(output / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
                    return action, metrics

                actor.generate = tracked
                suffix = hashlib.sha256(str(output).encode()).hexdigest()[:10]
                experiment = f"stable-{args.variant}-{suffix}-{task}"
                start = time.perf_counter()
                with AppWorld(
                    task_id=task,
                    experiment_name=experiment,
                    remote_environment_url=args.environment_url,
                    remote_docker=True,
                ) as app:
                    record = actor.collect(
                        app, "test_normal", selection["seed"] + selection["task_ids"].index(task)
                    )
                record.update(
                    task_wall_seconds=time.perf_counter() - start,
                    variant=args.variant,
                    experiment_name=experiment,
                    native=native,
                    architecture=asdict(model.config),
                    merge_plan=None if plan is None else asdict(plan),
                )
                record["kv_cost"] = validate_native(record) if native else record_kv(record)
                write_json(output / "trajectories" / f"test_normal-{task}.json", record)
                rows.append(task_row(record))
                write_json(
                    output / "status.json",
                    dict(
                        phase="evaluating",
                        completed=len(rows),
                        total=len(tasks),
                        success=sum(row["success"] for row in rows),
                    ),
                )
                print(
                    json.dumps(
                        dict(
                            phase="task_complete",
                            task=task,
                            success=record["official_score"]["success"],
                        )
                    ),
                    flush=True,
                )
        if weight_digest(model) != identity:
            raise RuntimeError("frozen auxiliary weights changed")
        write_json(output / "summary.json", aggregate(rows))
        write_json(output / "kernel_choices.json", choices())
        write_json(
            output / "status.json",
            dict(
                phase="complete",
                completed=len(rows),
                total=len(tasks),
                success=sum(row["success"] for row in rows),
                weights_unchanged=True,
            ),
        )
    except BaseException as error:
        write_json(
            output / "status.json",
            dict(
                phase="interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                error_type=type(error).__name__,
                message=str(error),
            ),
        )
        raise
