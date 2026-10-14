"""One published method and one immutable, ordered test-normal selection."""

import json
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"
DEFAULT_PLAN = ASSETS / "stable_interfaces.json"
DEFAULT_SELECTION = ASSETS / "test_normal.json"


def load_plan(path=None):
    from .algorithm.stable_contract_merge import StablePlan, plans_from

    plans = plans_from(Path(path) if path else DEFAULT_PLAN)
    if set(plans) != {"stable_interfaces"}:
        raise ValueError("configuration must contain exactly stable_interfaces")
    plan = plans["stable_interfaces"]
    if not isinstance(plan, StablePlan) or not plan.native_signals or not plan.protect_catalog:
        raise ValueError("native signals and interface protection must both be enabled")
    for branch in (plan.base.catalog, plan.base.data):
        if branch.stages != (3,) or not branch.protect_memory or not branch.one_stage_per_tool:
            raise ValueError("only Stage 3, with retained memory protection, is supported")
        if branch.merge_error is None:
            raise ValueError("bounded hidden pooling is required")
    return plan


def load_selection():
    value = json.loads(DEFAULT_SELECTION.read_text())
    if (
        value["split"] != "test_normal"
        or len(value["task_ids"]) != 168
        or len(set(value["task_ids"])) != 168
    ):
        raise ValueError("the packaged test-normal selection is invalid")
    return value


def select_tasks(selection, tasks=None, limit=None, shard_index=0, num_shards=1):
    official = selection["task_ids"]
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("require 0 <= shard-index < num-shards")
    if tasks is not None and (
        not tasks or len(set(tasks)) != len(tasks) or not set(tasks) <= set(official)
    ):
        raise ValueError("tasks must be unique IDs from the packaged test-normal selection")
    chosen = [task for task in official if tasks is None or task in tasks]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        chosen = chosen[:limit]
    result = chosen[shard_index::num_shards]
    if not result:
        raise ValueError("empty task shard")
    return result
