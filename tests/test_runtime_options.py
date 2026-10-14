import pytest
import torch

from helpers import tiny, sample
from stable_interfaces.runtime.artifacts import weight_digest
from stable_interfaces.runtime.runtime import RuntimeOptions


def snapshot(model, s):
    full = model(s.ids, **s.model_kwargs())
    cache, chunks = None, []
    for a, b in ((0, 24), (24, 25), (25, 32), (32, 54), (54, 55), (55, 64)):
        logits, out = model.next_logits(
            s.ids[a:b],
            span=s.span[a:b],
            eligible=s.eligible[a:b],
            protected=s.protected[a:b],
            past_key_values=cache,
        )
        cache = out.past_key_values
        chunks.append((out, logits))
    return full, chunks, cache


@pytest.mark.parametrize(
    "options",
    [
        RuntimeOptions(True, False),
        RuntimeOptions(True, True),
        RuntimeOptions(True, True, True, False),
        RuntimeOptions(True, True, True, True),
    ],
)
def test_runtime_options_preserve_active_memory_and_cached_decode(monkeypatch, options):
    model, s = tiny(monkeypatch), sample()
    before = weight_digest(model)
    reference = snapshot(model, s)
    assert [e["stage"] for e in reference[0].events] == [3]
    assert len(reference[0].sequence.hidden) < len(s.ids)
    model.configure_runtime(options)
    candidate = snapshot(model, s)
    for left, right in [(reference[0], candidate[0])] + [
        (a[0], b[0]) for a, b in zip(reference[1], candidate[1])
    ]:
        assert left.lengths == right.lengths
        assert left.events == right.events
        for name in vars(left.sequence):
            assert torch.equal(getattr(left.sequence, name), getattr(right.sequence, name)), name
    for (_, a), (_, b) in zip(reference[1], candidate[1]):
        assert torch.equal(a, b)
    assert reference[2].layer_lengths == candidate[2].layer_lengths
    assert reference[2].compression_metrics() == candidate[2].compression_metrics()
    for name in ("key_cache", "value_cache", "conv_states", "recurrent_states"):
        for a, b in zip(getattr(reference[2].native, name), getattr(candidate[2].native, name)):
            assert (a is None and b is None) or (
                a is not None and b is not None and torch.equal(a, b)
            )
    assert weight_digest(model) == before


def test_runtime_switch_invalidates_existing_cache(monkeypatch):
    model, s = tiny(monkeypatch), sample()
    cache = model(s.ids[:2], use_cache=True).past_key_values
    model.configure_runtime(RuntimeOptions(True, True))
    with pytest.raises(ValueError, match="stale"):
        model(s.ids[2:3], use_cache=True, past_key_values=cache)


@pytest.mark.parametrize("corruption", ["nonzero_B", "nonfinite_A"])
def test_elision_rejects_nonzero_or_nonfinite_interface(monkeypatch, corruption):
    model = tiny(monkeypatch)
    if corruption == "nonzero_B":
        model.readers["10"].B.fill_(1)
    else:
        model.readers["10"].A.fill_(float("inf"))
    with pytest.raises(ValueError, match="trained adapters|finite frozen"):
        model.configure_runtime(RuntimeOptions(True, False))
    assert model.runtime == RuntimeOptions()


def test_fast_single_token_path_retains_geometry_validation(monkeypatch):
    model, s = tiny(monkeypatch), sample()
    model.configure_runtime(RuntimeOptions(True, True))
    with pytest.raises(ValueError, match="complete Tool"):
        model(s.ids[:1], span=torch.tensor([0]), use_cache=True)
    with pytest.raises(ValueError, match="protected position"):
        model(
            s.ids[:1],
            span=torch.tensor([-1]),
            eligible=torch.tensor([True]),
            protected=torch.tensor([False]),
            use_cache=True,
        )
    with pytest.raises(ValueError, match="inconsistent sequence metadata"):
        model(s.ids[:1], span=torch.tensor([-1, -1]), use_cache=True)


def test_runtime_options_require_boolean_switches():
    with pytest.raises(ValueError, match="bool"):
        RuntimeOptions(elide_zero_branches="false")
