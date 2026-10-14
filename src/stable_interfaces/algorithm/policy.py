"""Immutable Stage-3 policy and its scoped reader configuration."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class Rule:
    stage: int
    chi: float
    psi: float
    min_net: int = 1

    def __post_init__(self):
        if self.stage != 3:
            raise ValueError("unsupported stage")
        if any(not math.isfinite(x) or x <= 0 for x in (self.chi, self.psi)):
            raise ValueError("positive finite thresholds required")
        if type(self.min_net) is not int or self.min_net < 1:
            raise ValueError("positive integer min_net required")


@dataclass(frozen=True)
class Plan:
    rules: tuple[Rule, ...]
    protect_memory: bool = True
    mode: str = "cascade"
    reader: str = "projection"
    min_layer_savings: int = 256
    merge_error: float = 0.5
    members_per_slot: int = 2
    preflight: bool = True
    one_stage_per_tool: bool = True
    min_tool_fraction: float = 0.05
    signal: str = "reference"
    beta_floor: float = 1e-6

    def __post_init__(self):
        if len(self.rules) != 1 or self.rules[0].stage != 3:
            raise ValueError("exactly one Stage 3 rule required")
        if self.mode != "cascade" or self.reader != "projection" or self.signal != "reference":
            raise ValueError("requires native reference signals and projected cascade selection")
        if not all(
            flag is True for flag in (self.protect_memory, self.preflight, self.one_stage_per_tool)
        ):
            raise ValueError("memory protection, preflight and one commit per Tool are required")
        if type(self.min_layer_savings) is not int or self.min_layer_savings < 0:
            raise ValueError("invalid savings")
        if self.merge_error is None or not math.isfinite(self.merge_error) or self.merge_error <= 0:
            raise ValueError("positive finite merge error required")
        if type(self.members_per_slot) is not int or self.members_per_slot < 2:
            raise ValueError("invalid group size")
        if not 0 <= self.min_tool_fraction < 1:
            raise ValueError("invalid minimum fraction")
        if not math.isfinite(self.beta_floor) or not 0 < self.beta_floor < 1:
            raise ValueError("invalid beta floor")

    @property
    def stages(self):
        return (3,)


@contextmanager
def configured_scope(model, plan):
    """Restore model, reader and cache policy settings even after an interruption."""
    if "10" not in model.readers:
        raise ValueError("load the calibrated Stage-3 reader first")
    old_config = model.config
    old_readers = {name: reader.config for name, reader in model.readers.items()}
    had_policy = hasattr(model, "_merge_policy")
    old_policy = getattr(model, "_merge_policy", None)
    rule = plan.rules[0]
    model.config = replace(old_config, stages=plan.stages)
    model.readers["10"].config = replace(
        old_readers["10"],
        stages=plan.stages,
        chi_threshold=rule.chi,
        readability_threshold=rule.psi,
    )
    model._merge_policy = plan
    try:
        yield
    finally:
        model.config = old_config
        for name, reader in model.readers.items():
            reader.config = old_readers[name]
        if had_policy:
            model._merge_policy = old_policy
        else:
            del model._merge_policy
