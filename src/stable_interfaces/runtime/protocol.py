"""Structured Tool spans and portable Agent transcripts and official score validation."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import torch


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass
class Sample:
    ids: torch.Tensor
    labels: torch.Tensor
    span: torch.Tensor
    eligible: torch.Tensor
    protected: torch.Tensor
    route_seed: int = 0

    def validate(self):
        if self.ids.ndim != 1 or self.ids.dtype != torch.long or len(self.ids) < 2:
            raise ValueError("invalid original token IDs")
        for name in ("labels", "span", "eligible", "protected"):
            field = getattr(self, name)
            if field.shape != self.ids.shape or field.device != self.ids.device:
                raise ValueError("sample geometry mismatch")
        if (
            self.labels.dtype != torch.long
            or self.span.dtype != torch.long
            or self.eligible.dtype != torch.bool
            or self.protected.dtype != torch.bool
        ):
            raise ValueError("metadata dtype mismatch")
        supervised = self.labels != -100
        if not torch.equal(self.labels[supervised], self.ids[supervised]):
            raise ValueError("labels must be the actual continuation IDs")
        if bool((self.eligible & (self.protected | (self.span < 0))).any()):
            raise ValueError("protected token eligible for compression")
        predictors = (self.labels[1:] != -100).nonzero().flatten()
        if bool(self.eligible[predictors].any()):
            raise ValueError("action predictor may not be compressed")
        return self

    def model_kwargs(self):
        return {
            "span": self.span,
            "eligible": self.eligible,
            "protected": self.protected,
            "route_seed": self.route_seed,
        }

    def to(self, device):
        return Sample(
            *(
                getattr(self, n).to(device)
                for n in ("ids", "labels", "span", "eligible", "protected")
            ),
            self.route_seed,
        )

    def dictionary(self):
        return {
            **{
                n: getattr(self, n).tolist()
                for n in ("ids", "labels", "span", "eligible", "protected")
            },
            "route_seed": self.route_seed,
        }

    @classmethod
    def from_dict(cls, record, device="cpu"):
        args = [
            torch.tensor(
                record[n],
                device=device,
                dtype=torch.bool if n in ("eligible", "protected") else torch.long,
            )
            for n in ("ids", "labels", "span", "eligible", "protected")
        ]
        return cls(*args, int(record["route_seed"])).validate()


class Protocol:
    def __init__(self, path=None):
        path = (
            Path(__file__).parent / "assets/appworld_protocol.json" if path is None else Path(path)
        )
        self.config = json.loads(Path(path).read_text())

    def header(self, role):
        return "<|im_start|>" + role + "\n"

    def initial(self, instruction):
        return (
            self.header("system")
            + self.config["system_prompt"]
            + "<|im_end|>\n"
            + self.header("user")
            + instruction
            + "<|im_end|>\n"
        )

    def assistant(self):
        return self.header("assistant") + self.config["assistant_prefix"]

    def encode(self, trajectory, tokenizer, max_tokens=32768, *, skip_first_action=False):
        # Same append boundaries, special-token ownership and Tool masks as
        # real Agent inference. Do not retokenize a concatenated transcript.
        from .agent import Tape

        tape = Tape(tokenizer, self, trajectory["instruction"], 0)
        for index, turn in enumerate(trajectory["turns"]):
            tape.append(self.assistant())
            tape.append(
                self.config["action_open"] + turn["action"] + self.config["action_close"],
                action=not skip_first_action or index > 0,
            )
            tape.observation(turn["result"])
        if len(tape.ids) > max_tokens:
            raise ValueError("context_limit: whole trajectories must not be silently truncated")
        return tape.sample()


def validate_score(score, split):
    if (
        score.get("split") != split
        or type(score.get("success")) is not bool
        or type(score.get("passed")) is not int
        or type(score.get("total")) is not int
        or not 0 <= score["passed"] <= score["total"]
        or score["total"] == 0
        or score["success"] != (score["passed"] == score["total"])
        or score.get("source")
        not in ("appworld.world.evaluate", "appworld.evaluator.evaluate_task")
    ):
        raise ValueError("invalid official evaluator evidence")


def severe_loop(turns):
    # Repeated identical (action,result), not repeated API names or valid paging.
    signatures = [(t["action"].strip(), str(t["result"]).strip()) for t in turns]
    if any(count >= 4 for count in Counter(signatures).values()):
        return "repeated_identical_action_observation"
    for width in (1, 2, 3):
        if (
            len(signatures) >= 3 * width
            and signatures[-width:]
            == signatures[-2 * width : -width]
            == signatures[-3 * width : -2 * width]
        ):
            return "repeated_cycle"
    return None
