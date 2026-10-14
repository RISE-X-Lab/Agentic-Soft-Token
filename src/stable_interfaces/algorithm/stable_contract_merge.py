"""Read routing signals from an unchanged native GDN forward.

The native output and recurrent cache are never rebuilt from split Tool scans.
Intermediate FLA tensors are borrowed from the original call. A prefix-state
kernel supplies each causal Tool-end state; its extra work is not called free.
This scope is inference-only and belongs to one sequential worker.
"""

from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import json

from .typed_contract_merge import TypedPlan, parse_plan as parse_base, typed_scope


@dataclass(frozen=True)
class StablePlan:
    base: TypedPlan
    native_signals: bool = True
    protect_catalog: bool = False

    def __post_init__(self):
        if not isinstance(self.base, TypedPlan):
            raise ValueError("a frozen typed baseline is required")
        if type(self.native_signals) is not bool or type(self.protect_catalog) is not bool:
            raise ValueError("boolean switches required")


def plans_from(path):
    values = json.loads(path.read_text())
    return {
        name: StablePlan(
            parse_base(v["base"]), v.get("native_signals", True), v.get("protect_catalog", False)
        )
        if "base" in v
        else parse_base(v)
        for name, v in values.items()
    }


@contextmanager
def native_signal_scope():
    import torch
    import stable_interfaces.runtime.model as implementation
    from stable_interfaces.runtime.bridge import Signals
    from stable_interfaces.runtime.dynamics import api, prior_read

    original = implementation.gdn_forward

    def gdn_forward(gdn, hidden, config, ranges=(), cache=None, collect=False):
        if not collect:
            return original(gdn, hidden, config, ranges, cache, False)
        if torch.is_grad_enabled() or config.backend != "fla" or hidden.shape[1] <= 1:
            raise ValueError("native signals require multi-token frozen FLA prefill")
        f = api()
        h_kernel, native_kernel = f.chunk_gated_delta_rule_fwd_h, gdn.chunk_gated_delta_rule
        captured = {}

        def state_kernel(*args, **kw):
            result = h_kernel(*args, **kw)
            captured.update(kw)
            captured.update(h=result[0], write=result[1], final=result[2])
            return result

        def native(q, k, v, **kw):
            captured.update(value=v, raw_g=kw["g"])
            return native_kernel(q, k, v, **kw)

        f.chunk_gated_delta_rule_fwd_h = state_kernel
        gdn.chunk_gated_delta_rule = native
        try:
            mixed, _ = original(gdn, hidden, config, (), cache, False)
        finally:
            f.chunk_gated_delta_rule_fwd_h = h_kernel
            gdn.chunk_gated_delta_rule = native_kernel
        if "h" not in captured or captured.get("cu_seqlens") is not None:
            raise RuntimeError("native FLA intermediates unavailable")
        k, v, cg = captured["k"], captured["value"], captured["g"]
        h0 = captured["initial_state"]
        if h0 is None:
            h0 = torch.zeros(
                (k.shape[0], k.shape[2], k.shape[3], v.shape[-1]),
                dtype=torch.float32,
                device=k.device,
            )
        delta = v.float() - captured["raw_g"].exp()[..., None] * prior_read(
            k, h0, captured["write"], captured["h"], cg
        )
        states = []
        for _, _, end in ranges:
            if end == k.shape[1]:
                state = captured["final"]
            else:
                # Triangular WY factors are causal: prefix slices need no
                # recomputation of projections, convolution, or WY factors.
                _, _, state = h_kernel(
                    k=k[:, :end].contiguous(),
                    w=captured["w"][:, :end].contiguous(),
                    u=captured["u"][:, :end].contiguous(),
                    g=cg[:, :end].contiguous(),
                    initial_state=captured["initial_state"],
                    output_final_state=True,
                    cu_seqlens=None,
                )
            states.append(state[0])
        return mixed, Signals(delta[0], v[0], k[0], tuple(states))

    implementation.gdn_forward = gdn_forward
    try:
        yield
    finally:
        implementation.gdn_forward = original


@contextmanager
def stable_scope(model, plan, audit=None, kinds=None):
    if not isinstance(plan, StablePlan):
        with typed_scope(model, plan, audit, kinds):
            yield
        return
    from stable_interfaces.runtime.agent import Actor

    with ExitStack() as stack:
        stack.enter_context(typed_scope(model, plan.base, audit, kinds))
        previous = model._merge_policy
        model._merge_policy = plan
        if plan.native_signals:
            stack.enter_context(native_signal_scope())
        generate = Actor.generate

        def guarded_generate(self, tape, generator):
            action, metrics = generate(self, tape, generator)
            if plan.protect_catalog and getattr(tape, "_next_tool_role", None) == "catalog":
                tape._guard_next_observation = True
                metrics.update(protect_observation=True, protect_tool_catalog=True)
            return action, metrics

        Actor.generate = guarded_generate
        try:
            yield
        finally:
            Actor.generate = generate
            model._merge_policy = previous
