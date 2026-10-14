from dataclasses import replace
import json
import pytest
import torch
from helpers import tiny
from test_operation_guard import Tokenizer
from stable_interfaces.configuration import load_plan, load_selection, select_tasks, DEFAULT_PLAN
from stable_interfaces.runtime.agent import Actor, Tape
from stable_interfaces.runtime.config import Config
from stable_interfaces.runtime.protocol import Protocol
from stable_interfaces.runtime.artifacts import read_bundle
from stable_interfaces.algorithm.stable_contract_merge import stable_scope
from stable_interfaces.algorithm.pooling import adaptive_pool


def test_only_stage3_is_loadable(tmp_path):
    plan = load_plan()
    assert plan.base.data.stages == plan.base.catalog.stages == (3,)
    assert plan.native_signals and plan.protect_catalog
    config, weights, meta = read_bundle()
    assert config.stages == (3,) and config.slots == 16
    assert meta["gradient_updates"] == 0
    assert {key.split(".")[1] for key in weights if key.startswith("readers.")} == {"10"}
    with pytest.raises(ValueError, match="Stage 3"):
        Config(stages=(2, 4, 6))
    document = json.loads(DEFAULT_PLAN.read_text())
    document["stable_interfaces"]["base"]["data"]["rules"][0]["stage"] = 4
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="unsupported stage"):
        load_plan(path)


def test_shards_partition_selection_without_changing_task_seeds():
    selection = load_selection()
    shards = [select_tasks(selection, shard_index=i, num_shards=4) for i in range(4)]
    assert sum(map(len, shards)) == len(set(sum(shards, []))) == 168
    assert set(sum(shards, [])) == set(selection["task_ids"])
    tasks = select_tasks(selection, tasks=selection["task_ids"][3:6][::-1])
    assert tasks == selection["task_ids"][3:6]
    with pytest.raises(ValueError):
        select_tasks(selection, tasks=["unknown"])
    with pytest.raises(ValueError):
        select_tasks(selection, shard_index=4, num_shards=4)


def test_interface_scope_protects_catalog_and_restores_after_exception(monkeypatch):
    model = tiny(monkeypatch)
    plan = replace(load_plan(), native_signals=False)  # CPU oracle checks masks/hooks only.
    action = "print(apis.api_docs.show_api_descriptions(app_name='venmo'))"
    monkeypatch.setattr(Actor, "generate", lambda self, tape, gen: (action, {}))
    original = Actor.generate, Tape.observation
    actor = object.__new__(Actor)
    tape = Tape(Tokenizer(), Protocol(), "fixture", 13)
    with pytest.raises(RuntimeError, match="interrupt"), stable_scope(model, plan):
        _, metrics = actor.generate(tape, None)
        tape.observation('{"list_items": "retrieve items using pagination"}')
        assert metrics["protect_tool_catalog"]
        assert not any(e for s, e in zip(tape.spans, tape.eligible) if s == 0)
        action = "print(apis.venmo.show_transactions())"
        actor.generate(tape, None)
        tape.observation('{"description": "ordinary repeated words", "item_id": 394}')
        assert any(e for s, e in zip(tape.spans, tape.eligible) if s == 1)
        raise RuntimeError("interrupt")
    assert (Actor.generate, Tape.observation) == original
    assert not hasattr(model, "_merge_policy")
    assert not hasattr(model, "_compression_bypass")


@torch.no_grad()
def test_scope_lifetime_and_policy_changes_invalidate_cache(monkeypatch):
    model = tiny(monkeypatch)
    plan = replace(load_plan(), native_signals=False)
    with stable_scope(model, plan):
        cache = model(torch.tensor([1, 2]), use_cache=True).past_key_values
        _, output = model.next_logits(torch.tensor([3]), past_key_values=cache)
        assert output.past_key_values.logical_length == 3
    with pytest.raises(ValueError, match="stale"):
        model.next_logits(torch.tensor([4]), past_key_values=cache)


def test_pool_retains_outlier_and_bounds_actual_bfloat16_centers():
    hidden = torch.tensor([[1.0, 1.0]] * 15 + [[30.0, -30.0]], dtype=torch.bfloat16)
    position = torch.arange(16)
    ids, groups, soft, endpoints, worst = adaptive_pool(hidden, position, 4, 0.5, 2)
    assert 15 not in ids.tolist()
    assert len(ids) > len(soft) and 0 < len(soft) <= 4
    actual = (hidden[ids].float() - soft[groups].float()).norm(dim=-1) / hidden[ids].float().norm(
        dim=-1
    )
    assert bool((actual <= 0.5).all()) and bool((worst <= 0.5).all())
    for group in range(len(soft)):
        assert endpoints[group] == position[ids[groups == group]].max()


@torch.no_grad()
def test_scoped_soft_tokens_preserve_protected_positions_and_cached_decode(monkeypatch):
    from helpers import sample

    model, source = tiny(monkeypatch), sample()
    plan = load_plan()

    def permissive(branch):
        rule = replace(branch.rules[0], chi=100.0, psi=100.0)
        return replace(
            branch, rules=(rule,), min_layer_savings=0, min_tool_fraction=0.0, merge_error=10.0
        )

    plan = replace(
        plan,
        native_signals=False,
        base=replace(
            plan.base, catalog=permissive(plan.base.catalog), data=permissive(plan.base.data)
        ),
    )
    with stable_scope(model, plan):
        output = model(source.ids, **source.model_kwargs(), use_cache=True)
        assert len(output.sequence.hidden) < len(source.ids)
        retained = set(output.sequence.source[output.sequence.source >= 0].tolist())
        assert set(source.protected.nonzero().flatten().tolist()) <= retained
        cache = output.past_key_values
        before = list(cache.layer_lengths)
        logits, output = model.next_logits(torch.tensor([7]), past_key_values=cache)
        assert torch.isfinite(logits).all()
        assert output.past_key_values.layer_lengths == [length + 1 for length in before]
        assert output.past_key_values.layer_lengths[11] < output.past_key_values.logical_length
