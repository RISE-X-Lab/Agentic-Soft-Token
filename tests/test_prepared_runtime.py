from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from helpers import tiny
from stable_interfaces.runtime.runtime import RuntimeOptions


@pytest.mark.parametrize("change", ["inplace", "replace", "ridge", "epsilon", "dtype"])
@torch.no_grad()
def test_prepared_terms_refresh_after_dependency_changes(monkeypatch, change):
    model = tiny(monkeypatch)
    model.configure_runtime(RuntimeOptions(True, True, True, True))
    reader = model.readers["10"]
    old = reader.fixed_terms()
    assert reader.fixed_terms()[0] is old[0]
    if change == "inplace":
        reader.queries.add_(0.1)
    elif change == "replace":
        reader.queries = nn.Parameter(reader.queries.clone(), requires_grad=False)
    elif change == "ridge":
        reader.config = replace(reader.config, ridge=0.1)
    elif change == "epsilon":
        reader.config = replace(reader.config, epsilon=0.1)
    elif change == "dtype":
        reader.to(dtype=torch.float64)
    q, gram = reader.fixed_terms()
    assert q is not old[0]
    expected = F.normalize(reader.queries.float(), dim=1, eps=reader.config.epsilon)
    assert torch.equal(q, expected)
    assert torch.equal(
        gram, expected.transpose(1, 2) @ expected + reader.config.ridge * torch.eye(q.shape[-1])
    )
    assert not any("fixed_terms" in key for key in model.state_dict())


@torch.no_grad()
def test_autocast_does_not_reuse_incompatible_preparation(monkeypatch):
    model = tiny(monkeypatch)
    model.configure_runtime(RuntimeOptions(prepare_memory_reader=True))
    reader = model.readers["10"]
    ordinary = reader.fixed_terms()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cast = reader.fixed_terms()
        assert cast[0] is not ordinary[0]
    assert reader.fixed_terms()[0] is not cast[0]


def test_preparation_does_not_reuse_an_autograd_graph(monkeypatch):
    model = tiny(monkeypatch)
    reader = model.readers["10"]
    reader.cache_fixed_terms = True
    reader.queries.requires_grad_(True)
    for _ in range(2):
        q, gram = reader.fixed_terms()
        (q.sum() + gram.sum()).backward()
        assert reader.queries.grad is not None
        reader.queries.grad = None
    assert reader._fixed_terms_cache is None


def test_rope_reuse_is_per_forward_and_dynamic_type_rejected(monkeypatch):
    model = tiny(monkeypatch)
    model.configure_runtime(RuntimeOptions(reuse_rope=True))
    original = model.backbone.rotary_emb.forward
    positions = []

    def observe(x, ids):
        positions.append(ids.clone())
        return original(x, ids)

    monkeypatch.setattr(model.backbone.rotary_emb, "forward", observe)
    cache = model(torch.tensor([1, 2]), use_cache=True).past_key_values
    for i in range(2):
        _, out = model.next_logits(torch.tensor([i + 3]), past_key_values=cache)
        cache = out.past_key_values
    assert [int(t[0, 0, -1]) for t in positions] == [1, 2, 3]
    monkeypatch.setattr(model.backbone.rotary_emb, "rope_type", "dynamic")
    with pytest.raises(ValueError, match="default rotary"):
        model.configure_runtime(RuntimeOptions(reuse_rope=True))


def test_runtime_change_invalidates_existing_cache_and_old_metadata_still_loads(monkeypatch):
    model = tiny(monkeypatch)
    cache = model(torch.tensor([1, 2]), use_cache=True).past_key_values
    model.configure_runtime(RuntimeOptions(prepare_memory_reader=True, reuse_rope=True))
    with pytest.raises(ValueError, match="stale"):
        model.next_logits(torch.tensor([3]), past_key_values=cache)
    old = RuntimeOptions(**{"elide_zero_branches": True, "skip_empty_decode_signals": True})
    assert not old.prepare_memory_reader and not old.reuse_rope


def test_reuse_supports_inference_mode_position_tensors(monkeypatch):
    model = tiny(monkeypatch)
    model.configure_runtime(RuntimeOptions(True, True, True, True))
    with torch.inference_mode():
        cache = model(torch.tensor([1, 2]), use_cache=True).past_key_values
        logits, out = model.next_logits(torch.tensor([3]), past_key_values=cache)
        assert torch.isfinite(logits).all() and out.past_key_values.logical_length == 3
