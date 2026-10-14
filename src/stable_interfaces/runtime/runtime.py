"""Explicit inference execution options, separate from calibrated routing."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeOptions:
    elide_zero_branches: bool = False
    skip_empty_decode_signals: bool = False
    prepare_memory_reader: bool = False
    reuse_rope: bool = False

    def __post_init__(self):
        if any(
            type(x) is not bool
            for x in (
                self.elide_zero_branches,
                self.skip_empty_decode_signals,
                self.prepare_memory_reader,
                self.reuse_rope,
            )
        ):
            raise ValueError("runtime switches must be bool")
