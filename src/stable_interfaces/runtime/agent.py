"""Frozen-model Agent execution with official evaluation and a persistent cache."""

from dataclasses import dataclass, asdict
import ast
import time
import torch
from .protocol import Sample, validate_score
from .sampling import sample_token
from .artifacts import weight_digest


@dataclass(frozen=True)
class RolloutConfig:
    max_steps: int = 50
    max_action_tokens: int = 768
    max_context: int = 32768
    temperature: float = 0.6
    top_k: int = 20
    greedy: bool = True

    def __post_init__(self):
        if (
            self.temperature != 0.6
            or self.top_k != 20
            or min(self.max_steps, self.max_action_tokens, self.max_context) <= 0
        ):
            raise ValueError("rollout uses temperature=.6, top-k=20 and positive limits")


class Tape:
    def __init__(self, tokenizer, protocol, instruction, seed):
        self.tokenizer, self.protocol, self.seed = tokenizer, protocol, seed
        self.ids, self.labels, self.spans, self.eligible = [], [], [], []
        self.special = set(tokenizer.all_special_ids)
        self.span_id = 0
        self.cache = None
        self.append(protocol.initial(instruction))

    def append_ids(self, ids, *, action=False, tool=False):
        self.ids.extend(ids)
        self.labels.extend(ids if action else [-100] * len(ids))
        self.spans.extend([self.span_id if tool else -1] * len(ids))
        self.eligible.extend([tool and token not in self.special for token in ids])

    def append(self, text, **kwargs):
        self.append_ids(self.tokenizer.encode(text, add_special_tokens=False), **kwargs)

    def observation(self, result):
        end_ids = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if self.ids[-len(end_ids) :] != end_ids:
            self.append("<|im_end|>")
        self.append("\n" + self.protocol.header("user") + "<tool_result>\n")
        self.append(str(result), tool=True)
        self.append("\n</tool_result><|im_end|>\n")
        self.span_id += 1

    def sample(self, device="cpu"):
        ids, labels, span = [
            torch.tensor(x, dtype=torch.long, device=device)
            for x in (self.ids, self.labels, self.spans)
        ]
        eligible = torch.tensor(self.eligible, dtype=torch.bool, device=device)
        return Sample(ids, labels, span, eligible, ~eligible, self.seed).validate()

    def uncached(self, device):
        offset = 0 if self.cache is None else self.cache.logical_length
        if not 0 <= offset < len(self.ids):
            raise ValueError("generation needs new tokens after the cached prefix")
        ids = torch.tensor(self.ids[offset:], dtype=torch.long, device=device)
        span = torch.tensor(self.spans[offset:], dtype=torch.long, device=device)
        eligible = torch.tensor(self.eligible[offset:], dtype=torch.bool, device=device)
        return ids, {
            "span": span,
            "eligible": eligible,
            "protected": ~eligible,
            "route_seed": self.seed,
        }


def parse_action(text, protocol):
    opening, closing = protocol.config["action_open"], protocol.config["action_close"]
    if opening not in text or closing not in text.split(opening, 1)[1]:
        return None
    code = text.split(opening, 1)[1].split(closing, 1)[0].strip()
    try:
        ast.parse(code)
    except SyntaxError:
        return None
    return code or None


class Actor:
    def __init__(
        self, model, tokenizer, protocol, config=None, *, routing="greedy", compression=True
    ):
        self.model, self.tokenizer, self.protocol = model, tokenizer, protocol
        self.config = RolloutConfig() if config is None else config
        self.routing, self.compression = routing, compression

    @torch.no_grad()
    def generate(self, tape, generator):
        tape.append(self.protocol.assistant())
        generated, cost, forward_tokens, compression_metrics = [], 0, 0, []
        initial = len(tape.ids)
        start = time.perf_counter()
        ttft, stop = None, "action_token_limit"
        for _ in range(self.config.max_action_tokens):
            if len(tape.ids) >= self.config.max_context:
                stop = "context_limit"
                break
            new_ids, metadata = tape.uncached(self.model.device)
            logits, output = self.model.next_logits(
                new_ids,
                **metadata,
                routing=self.routing,
                compression=self.compression,
                use_cache=True,
                past_key_values=tape.cache,
            )
            tape.cache = output.past_key_values
            if tape.cache is None:
                raise RuntimeError("Agent generation must return a reusable hybrid cache")
            token = (
                int(logits.argmax())
                if self.config.greedy
                else sample_token(logits, generator, self.config.temperature, self.config.top_k)
            )
            cost += sum(output.lengths)
            forward_tokens += len(new_ids)
            compression_metrics = tape.cache.compression_metrics()
            del logits, output, new_ids, metadata
            tape.append_ids([token], action=True)
            generated.append(token)
            if ttft is None:
                ttft = time.perf_counter() - start
            text = self.tokenizer.decode(generated, skip_special_tokens=False)
            if parse_action(text, self.protocol) is not None:
                stop = "action_complete"
                break
            if token in (
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            ):
                stop = "eos_without_complete_action"
                break
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        return parse_action(text, self.protocol), {
            "stop": stop,
            "generated_ids": generated,
            "generated_tokens": len(generated),
            "logical_prompt_tokens": initial,
            "actual_forward_input_tokens": forward_tokens,
            "token_layer_positions": cost,
            "wall_seconds": time.perf_counter() - start,
            "ttft_seconds": ttft,
            "use_cache": True,
            "cache_bytes": tape.cache.bytes() if tape.cache is not None else 0,
            "cache_layer_lengths": list(tape.cache.layer_lengths) if tape.cache is not None else [],
            "last_forward_compression": compression_metrics,
        }

    def collect(self, world, split, seed):
        # AppWorld Task exposes .id; AppWorld itself exposes .task_id. Resolve
        # BEFORE expensive generation, not while serializing the final result.
        task_id = task_identifier(world)
        self.model.eval()
        identity = weight_digest(self.model)
        instruction = task_instruction(world.task)
        tape = Tape(self.tokenizer, self.protocol, instruction, seed)
        generator = torch.Generator(device=self.model.device).manual_seed(seed)
        turns, measurements, reason = [], [], None
        for _ in range(self.config.max_steps):
            action, metrics = self.generate(tape, generator)
            measurements.append(metrics)
            if action is None:
                reason = metrics["stop"]
                break
            # Infrastructure errors propagate, never count as official failure.
            result = str(world.execute(action))
            turns.append({"action": action, "result": result})
            tape.observation(result)
            if len(tape.ids) > self.config.max_context:
                reason = "context_limit"
                break
            reason = None
            if reason or world.task_completed():
                break
        completed = bool(world.task_completed())
        reason = reason or (None if completed else "step_limit")
        tracker = world.evaluate()
        score = {
            "success": bool(tracker.success),
            "passed": int(tracker.pass_count),
            "total": int(tracker.num_tests),
            "source": "appworld.world.evaluate",
            "split": split,
        }
        validate_score(score, split)
        if weight_digest(self.model) != identity:
            raise RuntimeError("weights changed during rollout; not a well-defined behavior policy")
        record = {
            "task_id": task_id,
            "split": split,
            "instruction": instruction,
            "turns": turns,
            "official_score": score,
            "completed": completed,
            "accepted": completed and reason is None,
            "rejection_reason": reason,
            "sample": tape.sample().dictionary(),
            "metrics": measurements,
            "behavior": {
                "version": 0,
                "digest": identity,
                "routing": "deterministic_memory_thresholds",
                "temperature": 0.6,
                "top_k": 20,
                "top_p": None,
                "use_cache": True,
                "runtime": asdict(self.model.runtime),
            },
        }
        # Greedy evaluation does not apply temperature or top-k sampling.
        record["behavior"].update(
            decoding="greedy" if self.config.greedy else "sample",
            temperature=None if self.config.greedy else self.config.temperature,
            top_k=None if self.config.greedy else self.config.top_k,
        )
        return record


def task_identifier(world):
    task_id = getattr(world.task, "id", None)
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("AppWorld Task.id must be a nonempty string")
    if getattr(world, "task_id", task_id) != task_id:
        raise ValueError("AppWorld.task_id differs from Task.id")
    return task_id


def task_instruction(task):
    supervisor = task.supervisor
    if not isinstance(supervisor, dict):
        return "Task: " + str(task.instruction)
    name = " ".join(str(supervisor.get(k, "")).strip() for k in ("first_name", "last_name")).strip()
    facts = [f"My name is {name}."] if name else []
    for field, label in (("email", "personal email"), ("phone_number", "phone number")):
        if supervisor.get(field):
            facts.append(f"My {label} is {supervisor[field]}.")
    return (" ".join(facts) + "\n\n" if facts else "") + "Task: " + str(task.instruction)
