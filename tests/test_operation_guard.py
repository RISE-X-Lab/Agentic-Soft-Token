import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from stable_interfaces.algorithm.operation_guard import (
    contract_action,
    contract_result,
    operation_scope,
    protect_sample_contracts,
)
from stable_interfaces.algorithm.observation_guard import observation_scope
from stable_interfaces.runtime.agent import Actor, Tape
from stable_interfaces.runtime.protocol import Protocol


class Tokenizer:
    all_special_ids = []

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def __call__(self, text, **kwargs):
        return dict(
            input_ids=self.encode(text), offset_mapping=[(i, i + 1) for i in range(len(text))]
        )


@pytest.mark.parametrize(
    "action,expected",
    [
        ("print(apis.api_docs.show_api_doc(app_name='phone',api_name='show_alarms'))", True),
        (
            "a=apis.api_docs.show_api_doc(app_name='x',api_name='y'); print(a,apis.x.get_data())",
            True,
        ),
        ("print(apis.api_docs.show_api_descriptions(app_name='phone'))", False),
        ("print('apis.api_docs.show_api_doc()')", False),
        ("print(apis.phone.show_alarms())", False),
    ],
)
def test_contract_provenance(action, expected):
    assert contract_action(action) is expected


def test_contract_atomicity_literal_parity_and_scope_restoration(monkeypatch):
    action = "print(apis.api_docs.show_api_doc(app_name='x',api_name='list_items'))"
    monkeypatch.setattr(Actor, "generate", lambda self, tape, gen: (action, {}))
    original_generate, original_observation = Actor.generate, Tape.observation
    actor = object.__new__(Actor)
    reference = Tape(Tokenizer(), Protocol(), "example", 7)
    candidate = Tape(Tokenizer(), Protocol(), "example", 7)
    doc = '{"parameters": [{"name": "page_limit", "default": 5, "description": "return at most five items per page"}], "response_schemas": {"success": ["item"]}}'
    data = '{"name": "ordinary observation", "alarm_id": 394}'
    with observation_scope("literals"):
        actor.generate(reference, None)
        reference.observation(doc)
        actor.generate(reference, None)
        reference.observation(data)
    with operation_scope():
        _, metrics = actor.generate(candidate, None)
        candidate.observation(doc)
        assert metrics["protect_operational_contract"]
        assert not any(e for s, e in zip(candidate.spans, candidate.eligible) if s == 0)
        action = "print(apis.phone.show_alarms())"
        _, metrics = actor.generate(candidate, None)
        candidate.observation(data)
        assert not metrics["protect_operational_contract"]
    assert candidate.ids == reference.ids and candidate.spans == reference.spans
    assert [e for s, e in zip(candidate.spans, candidate.eligible) if s == 1] == [
        e for s, e in zip(reference.spans, reference.eligible) if s == 1
    ]
    assert any(e for s, e in zip(candidate.spans, candidate.eligible) if s == 1)
    assert Actor.generate is original_generate and Tape.observation is original_observation


def test_offline_mask_matches_online_and_keeps_original_tokens(monkeypatch):
    action = "print(apis.api_docs.show_api_doc(app_name='x',api_name='y'))"
    monkeypatch.setattr(Actor, "generate", lambda self, tape, gen: (action, {}))
    actor = object.__new__(Actor)
    tape = Tape(Tokenizer(), Protocol(), "example", 3)
    text = "parameters and defaults must remain explicit: limit=5"
    with observation_scope("literals"):
        actor.generate(tape, None)
        tape.observation(text)
    sample = tape.sample()
    before = sample.ids.clone()
    protect_sample_contracts({"turns": [{"action": action, "result": text}]}, sample)
    assert torch.equal(sample.ids, before) and not bool(sample.eligible.any())


def test_exception_restores_both_wrappers():
    old = (Actor.generate, Tape.observation)
    with pytest.raises(RuntimeError), operation_scope():
        raise RuntimeError("stop")
    assert old == (Actor.generate, Tape.observation)


def test_returned_schema_protects_aliased_doc_calls(monkeypatch):
    monkeypatch.setattr(Actor, "generate", lambda self, tape, gen: ("print(get_doc())", {}))
    actor = object.__new__(Actor)
    tape = Tape(Tokenizer(), Protocol(), "example", 3)
    text = '{"app_name":"x","api_name":"y","parameters":[],"response_schemas":{}}'
    with operation_scope():
        _, metrics = actor.generate(tape, None)
        tape.observation(text)
        assert metrics["protect_operational_contract"] and not any(tape.eligible)


def test_ordinary_records_and_catalogs_are_not_contracts():
    assert not contract_result('{"parameters":[1,2],"description":"ordinary data"}')
    assert not contract_result('[{"name":"list_items","description":"List items."}]')
    assert not contract_result("not JSON")
