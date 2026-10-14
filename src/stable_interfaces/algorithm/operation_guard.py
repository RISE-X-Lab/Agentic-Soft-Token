"""Keep explicit Tool contracts; preserve the frozen literal baseline unchanged.

A detailed api_docs.show_api_doc return is one indivisible operational contract:
description, parameters/defaults/constraints, and response schema remain explicit.
Catalog summaries and ordinary data continue through literal-aware GDN selection.
Mixed calls containing a detailed contract conservatively protect the whole return.
This is provenance-based, not a task-ID rule or a rewritten prompt.
"""

from contextlib import contextmanager


import json


from .observation_guard import api_calls, observation_scope as literal_scope


def contract_action(action):
    return ("api_docs", "show_api_doc") in api_calls(action)


def contract_result(result):
    """Recognize a returned API schema even when the action used a Python alias."""
    try:
        value = json.loads(str(result))
    except (ValueError, TypeError):
        return False
    return (
        isinstance(value, dict)
        and isinstance(value.get("app_name"), str)
        and isinstance(value.get("api_name"), str)
        and isinstance(value.get("parameters"), list)
        and isinstance(value.get("response_schemas"), dict)
    )


def protect_sample_contracts(record, sample):
    """Apply the same contract guard to an already literal-guarded replay sample."""
    for span, turn in enumerate(record["turns"]):
        if contract_action(turn["action"]) or contract_result(turn["result"]):
            sample.eligible[sample.span == span] = False
    sample.protected = ~sample.eligible
    return sample.validate()


@contextmanager
def operation_scope(mode="contracts"):
    if mode != "contracts":
        raise ValueError("operation guard requires contracts mode")
    from stable_interfaces.runtime.agent import Actor, Tape

    with literal_scope("literals"):
        generate, observation = Actor.generate, Tape.observation

        def guarded_generate(self, tape, generator):
            action, metrics = generate(self, tape, generator)
            contract = contract_action(action)
            tape._guard_next_observation = (
                getattr(tape, "_guard_next_observation", False) or contract
            )
            metrics["protect_observation"] = tape._guard_next_observation
            metrics["protect_operational_contract"] = contract
            tape._operation_metrics = metrics
            return action, metrics

        def guarded_observation(self, result):
            if contract_result(result):
                self._guard_next_observation = True
                metrics = getattr(self, "_operation_metrics", None)
                if metrics is not None:
                    metrics.update(protect_observation=True, protect_operational_contract=True)
            try:
                return observation(self, result)
            finally:
                self._operation_metrics = None

        Actor.generate, Tape.observation = guarded_generate, guarded_observation
        try:
            yield
        finally:
            Actor.generate, Tape.observation = generate, observation
