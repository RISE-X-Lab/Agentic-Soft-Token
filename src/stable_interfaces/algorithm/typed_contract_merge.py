"""Compose frozen merge policies by Tool role without changing cache policy mid-task.

Detailed contracts/authentication stay explicit through operation_scope. Catalogs
use the retained conservative plan; only ordinary data use the broader plan.
Both plans and the dispatcher belong to one immutable MemoryCache policy.
Mixed/unknown span provenance falls back to the conservative plan.
"""

from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import json
from .observation_guard import api_calls
from .operation_guard import operation_scope, contract_result, contract_action
from .policy import Plan, Rule
from .selective_merge import merge_scope


@dataclass(frozen=True)
class TypedPlan:
    catalog: Plan
    data: Plan
    policy: str = "contract_catalog_data_v1"

    def __post_init__(self):
        if self.policy != "contract_catalog_data_v1":
            raise ValueError("unknown typed policy")
        if self.catalog.stages != self.data.stages:
            raise ValueError("typed plans must share stages")
        for p in (self.catalog, self.data):
            if p.signal != "reference" or p.mode != "cascade" or not p.preflight:
                raise ValueError("typed dispatch requires reference cascade with preflight")


def parse_plan(value):
    if "catalog" in value:
        return TypedPlan(
            parse_plan(value["catalog"]),
            parse_plan(value["data"]),
            value.get("policy", "contract_catalog_data_v1"),
        )
    return Plan(
        tuple(Rule(**r) for r in value["rules"]), **{k: v for k, v in value.items() if k != "rules"}
    )


def plans_from(path):
    return {name: parse_plan(value) for name, value in json.loads(path.read_text()).items()}


def action_kind(action):
    calls = api_calls(action)
    if contract_action(action):
        return "contract"
    if any(a == "api_docs" for a, m in calls):
        return "catalog"
    if any(m == "login" or (a == "supervisor" and m == "show_account_passwords") for a, m in calls):
        return "authentication"
    # Python computations over previously returned data also produce data observations.
    return "data"


def choose_data(seq, kinds):
    ranges = seq.tool_ranges()
    return bool(ranges) and all(kinds.get(tool, "catalog") == "data" for tool, start, end in ranges)


@contextmanager
def typed_scope(model, plan, audit=None, kinds=None):
    if not isinstance(plan, TypedPlan):
        with merge_scope(model, plan, audit), operation_scope():
            yield
        return
    from stable_interfaces.runtime.agent import Actor, Tape
    import stable_interfaces.runtime.model as implementation

    kinds = {} if kinds is None else kinds
    owner = None
    with ExitStack() as stack:
        stack.enter_context(merge_scope(model, plan.catalog, audit))
        catalog_merge, catalog_gate = implementation.consolidate, model._compression_bypass
        stack.enter_context(merge_scope(model, plan.data, audit))
        data_merge, data_gate = implementation.consolidate, model._compression_bypass
        policy = model._merge_policy
        stack.enter_context(operation_scope())
        generate, observation = Actor.generate, Tape.observation

        def generate_typed(self, tape, generator):
            action, metrics = generate(self, tape, generator)
            tape._next_tool_role = action_kind(action)
            metrics["observation_role"] = tape._next_tool_role
            return action, metrics

        def observation_typed(self, result):
            nonlocal owner
            if owner is not self:
                kinds.clear()
                owner = self
            role = (
                "contract"
                if contract_result(result)
                else getattr(self, "_next_tool_role", "catalog")
            )
            kinds[self.span_id] = role
            try:
                return observation(self, result)
            finally:
                self._next_tool_role = "catalog"

        def gate(seq, index):
            return (data_gate if choose_data(seq, kinds) else catalog_gate)(seq, index)

        def consolidate(seq, *args, **kwargs):
            return (data_merge if choose_data(seq, kinds) else catalog_merge)(seq, *args, **kwargs)

        implementation.consolidate = consolidate
        model._compression_bypass = gate
        model._merge_policy = plan
        Actor.generate, Tape.observation = generate_typed, observation_typed
        try:
            yield
        finally:
            Actor.generate, Tape.observation = generate, observation
            implementation.consolidate = data_merge
            model._compression_bypass = data_gate
            model._merge_policy = policy
