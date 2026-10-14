import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from stable_interfaces.algorithm.observation_guard import (
    observation_scope,
    protect_action,
    literal_token_mask,
)
from stable_interfaces.runtime.agent import Actor, Tape
from stable_interfaces.runtime.protocol import Protocol


@pytest.mark.parametrize(
    "action,expected",
    [
        ("print(apis.api_docs.show_api_descriptions(app_name='spotify'))", True),
        ("passwords = apis.supervisor.show_account_passwords(); print(passwords)", True),
        ("print(apis.spotify.show_playlist_library(access_token=token))", False),
        ("print('apis.api_docs.show_api_doc()')", False),
        ("print(apis.supervisor.show_profile())", False),
    ],
)
def test_guard_uses_executed_code_structure(action, expected):
    assert protect_action(action) is expected


def test_guard_preserves_text_and_ids_and_restores_scope(monkeypatch):
    class Tokenizer:
        all_special_ids = []

        def encode(self, text, **kwargs):
            return [ord(char) for char in text]

    actor = object.__new__(Actor)
    action = "print(apis.api_docs.show_api_descriptions(app_name='phone'))"
    monkeypatch.setattr(Actor, "generate", lambda self, tape, gen: (action, {}))
    original_generate, original_observation = Actor.generate, Tape.observation
    normal = Tape(Tokenizer(), Protocol(), "example", 7)
    guarded = Tape(Tokenizer(), Protocol(), "example", 7)
    normal.observation("Exact method name: update_alarm")
    with observation_scope("interfaces"):
        actor.generate(guarded, None)
        guarded.observation("Exact method name: update_alarm")
        assert guarded.ids == normal.ids and guarded.spans == normal.spans
        assert any(normal.eligible) and not any(guarded.eligible)
        guarded.sample().validate()
        # A later data observation retains its compression eligibility.
        guarded.observation("ordinary data")
        assert any(guarded.eligible)
    assert Actor.generate is original_generate and Tape.observation is original_observation


def test_scope_restores_on_error():
    original_generate, original_observation = Actor.generate, Tape.observation
    with pytest.raises(RuntimeError), observation_scope("interfaces"):
        raise RuntimeError("interrupted")
    assert Actor.generate is original_generate and Tape.observation is original_observation


def test_literals_protect_complete_numbers_and_api_names():
    text = "ordinary text: show_playlist_library alarm_id=394 value=-3.14e+2"
    offsets = [(i, i + 1) for i in range(len(text))]
    mask = literal_token_mask(text, offsets)
    for literal in ("show_playlist_library", "alarm_id", "394", "-3.14e+2"):
        start = text.index(literal)
        assert all(mask[start : start + len(literal)])
    assert not any(mask[: len("ordinary text: ")])
    # A BPE piece overlapping part of a literal is protected as a whole.
    assert literal_token_mask("x 394 y", [(0, 2), (2, 4), (4, 6), (6, 7)]) == [
        False,
        True,
        True,
        False,
    ]


def test_literal_scope_keeps_original_ids_and_protects_login(monkeypatch):
    class Tokenizer:
        all_special_ids = []

        def encode(self, text, **kwargs):
            return [ord(char) for char in text]

        def __call__(self, text, **kwargs):
            return {
                "input_ids": self.encode(text),
                "offset_mapping": [(i, i + 1) for i in range(len(text))],
            }

    action = "print(apis.phone.show_alarms(access_token=token))"
    monkeypatch.setattr(Actor, "generate", lambda self, tape, gen: (action, {}))
    actor = object.__new__(Actor)
    tape = Tape(Tokenizer(), Protocol(), "example", 7)
    with observation_scope("literals"):
        actor.generate(tape, None)
        tape.observation("ordinary alarm_id=394")
        content = [i for i, s in enumerate(tape.spans) if s == 0]
        assert "".join(chr(tape.ids[i]) for i in content) == "ordinary alarm_id=394"
        assert all(tape.eligible[i] for i in content[:8])
        assert all(not tape.eligible[i] for i in content[-3:])
        action = "print(apis.phone.login(username=user, password=password))"
        actor.generate(tape, None)
        tape.observation("opaque authentication token")
        assert not any(e for s, e in zip(tape.spans, tape.eligible) if s == 1)
        tape.sample().validate()
