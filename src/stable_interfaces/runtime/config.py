from dataclasses import dataclass, asdict
import math

FORMAT = "gdn-ast-frozen-v1"


@dataclass(frozen=True)
class Config:
    backend: str = "fla"
    stages: tuple[int, ...] = (3,)
    slots: int = 16
    chi_threshold: float = 0.58
    readability_threshold: float = 0.58
    epsilon: float = 1e-6
    ridge: float = 1e-4
    lora_rank: int = 8
    lora_alpha: float = 8.0
    reader_rank: int = 16
    reader_alpha: float = 16.0
    readability_check: bool = True

    def __post_init__(self):
        if self.backend not in ("fla", "oracle"):
            raise ValueError("backend must be fla or oracle")
        if self.stages != (3,):
            raise ValueError("this release compresses only at Stage 3")
        if min(self.slots, self.lora_rank, self.reader_rank) < 1:
            raise ValueError("slots and ranks must be positive")
        for x in (
            self.chi_threshold,
            self.readability_threshold,
            self.epsilon,
            self.ridge,
            self.lora_alpha,
            self.reader_alpha,
        ):
            if not math.isfinite(x) or x <= 0:
                raise ValueError("calibration values must be finite and positive")

    @property
    def signal_layers(self):
        return tuple(4 * s - 2 for s in self.stages)

    @property
    def adaptation_layers(self):
        return tuple(i for s in self.stages for i in range(4 * s - 4, 4 * s - 1))

    def dictionary(self):
        return asdict(self)
