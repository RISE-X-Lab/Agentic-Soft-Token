"""Provenance guard for exact API interfaces and credentials.

Changes only compression eligibility, never tool text, prompts, or token IDs.
An entire mixed observation is protected if its action calls a protected API.
Direct ``apis.app.method`` calls are recognized; dynamically aliased calls are
not classified. The scope is for a single sequential evaluation worker.
"""

import ast
import re
from contextlib import contextmanager


def api_calls(action):
    if not action:
        return []
    try:
        tree = ast.parse(action)
    except SyntaxError:
        return []
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (
            isinstance(fn, ast.Attribute)
            and isinstance(fn.value, ast.Attribute)
            and isinstance(fn.value.value, ast.Name)
            and fn.value.value.id == "apis"
        ):
            continue
        app, method = fn.value.attr, fn.attr
        calls.append((app, method))
    return calls


def protect_action(action):
    return any(
        app == "api_docs" or (app == "supervisor" and method == "show_account_passwords")
        for app, method in api_calls(action)
    )


# Conservatively protect entire token pieces overlapping numbers or identifiers
# containing digits/underscores. This does not claim to identify all critical text.
LITERALS = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\b[A-Za-z_]\w*[_\d]\w*\b")


def literal_token_mask(text, offsets):
    intervals = [match.span() for match in LITERALS.finditer(text)]
    result, cursor = [], 0
    for start, end in offsets:
        while cursor < len(intervals) and intervals[cursor][1] <= start:
            cursor += 1
        result.append(
            bool(
                end > start
                and cursor < len(intervals)
                and intervals[cursor][0] < end
                and intervals[cursor][1] > start
            )
        )
    return result


@contextmanager
def observation_scope(mode="none"):
    if mode not in ("none", "interfaces", "literals"):
        raise ValueError("unknown observation guard")
    if mode == "none":
        yield
        return
    from stable_interfaces.runtime.agent import Actor, Tape

    generate, observation = Actor.generate, Tape.observation

    def guarded_generate(self, tape, generator):
        action, metrics = generate(self, tape, generator)
        tape._guard_next_observation = (
            protect_action(action)
            if mode == "interfaces"
            else any(
                (app == "supervisor" and method == "show_account_passwords") or method == "login"
                for app, method in api_calls(action)
            )
        )
        metrics["protect_observation"] = tape._guard_next_observation
        return action, metrics

    def guarded_observation(self, result):
        start = len(self.ids)
        tool_span = self.span_id
        observation(self, result)
        if getattr(self, "_guard_next_observation", False):
            self.eligible[start:] = [False] * (len(self.ids) - start)
        elif mode == "literals":
            text = str(result)
            encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            positions = [i for i in range(start, len(self.ids)) if self.spans[i] == tool_span]
            if [self.ids[i] for i in positions] != encoded["input_ids"]:
                raise ValueError("literal guard tokenizer offsets must match original Tool IDs")
            for index, protect in zip(
                positions, literal_token_mask(text, encoded["offset_mapping"]), strict=True
            ):
                if protect:
                    self.eligible[index] = False
        self._guard_next_observation = False

    Actor.generate, Tape.observation = guarded_generate, guarded_observation
    try:
        yield
    finally:
        Actor.generate, Tape.observation = generate, observation
