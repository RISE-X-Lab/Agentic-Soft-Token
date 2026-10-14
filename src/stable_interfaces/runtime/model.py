from dataclasses import dataclass
import torch
from torch import nn
from .config import Config
from .adapters import install, disabled, RowLoRA
from .geometry import Sequence, causal_mask
from .bridge import gdn_forward, Signals
from .memory import MemoryReader, consolidate
from .runtime import RuntimeOptions


def identity(model):
    return (
        id(model),
        tuple((id(p), p._version) for p in model.auxiliary_state().values()),
        tuple((m.enabled, m.elide_zero_output) for m in model.modules() if isinstance(m, RowLoRA)),
        tuple((r.elide_zero_output, r.cache_fixed_terms) for r in model.readers.values()),
    )


def cache_policy(model, compression):
    return (
        model.config,
        tuple(r.config for r in model.readers.values()),
        compression,
        model.runtime,
        getattr(model, "_merge_policy", None),
    )


class MemoryCache:
    def __init__(self, model, compression):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

        self.native = Qwen3_5DynamicCache(model.backbone.config)
        self.identity, self.policy = identity(model), cache_policy(model, compression)
        self.logical_length, self.layer_lengths = 0, [0] * 32
        self.seen_spans, self.totals, self.valid = set(), {}, True

    def validate(self, model, compression):
        if (
            not self.valid
            or self.identity != identity(model)
            or self.policy != cache_policy(model, compression)
        ):
            raise ValueError("cache is invalid, stale, or belongs to different weights/policy")

    def bytes(self):
        return sum(
            x.numel() * x.element_size()
            for name in ("key_cache", "value_cache", "conv_states", "recurrent_states")
            for x in getattr(self.native, name)
            if x is not None
        )

    def compression_metrics(self):
        return [
            {**v, "rho_actual": v["eligible_after"] / max(1, v["eligible_before"])}
            for _, v in sorted(self.totals.items())
        ]


@dataclass
class Output:
    sequence: Sequence
    lengths: list[int]
    events: list[dict]
    past_key_values: MemoryCache | None = None


class MemoryInnovation(nn.Module):
    def __init__(self, backbone, lm_head, config=None):
        super().__init__()
        self.config = Config() if config is None else config
        self.runtime = RuntimeOptions()
        if list(backbone.config.layer_types) != (["linear_attention"] * 3 + ["full_attention"]) * 8:
            raise ValueError("requires native GGGA x 8 topology")
        self.backbone, self.lm_head = backbone, lm_head.requires_grad_(False)
        install(backbone, self.config)
        self.readers = nn.ModuleDict(
            {
                str(i): MemoryReader(backbone.layers[i].linear_attn, self.config)
                for i in self.config.signal_layers
            }
        )
        self.requires_grad_(False)
        self.eval()
        self.assert_scope()

    @property
    def device(self):
        return self.backbone.embed_tokens.weight.device

    def auxiliary_state(self):
        # Frozen calibration + zero-update compatibility branches, not trainable weights.
        return {
            n: p
            for n, p in self.named_parameters()
            if n.startswith("readers.") or n.endswith((".A", ".B"))
        }

    def configure_runtime(self, options):
        if not isinstance(options, RuntimeOptions):
            raise TypeError("requires RuntimeOptions")
        self.assert_scope()
        if options.reuse_rope and getattr(self.backbone.rotary_emb, "rope_type", None) != "default":
            raise ValueError("RoPE reuse requires the validated default rotary embedding")
        if options.elide_zero_branches and any(
            not bool(torch.isfinite(p).all()) for p in self.auxiliary_state().values()
        ):
            raise ValueError("zero-branch elision requires finite frozen interface weights")
        self.runtime = options
        for module in self.modules():
            if isinstance(module, (RowLoRA, MemoryReader)):
                module.elide_zero_output = options.elide_zero_branches
            if isinstance(module, MemoryReader):
                module.cache_fixed_terms = options.prepare_memory_reader
                module._fixed_terms_cache = None

    def assert_scope(self):
        if any(p.requires_grad for p in self.parameters()):
            raise ValueError("this runtime is inference-only; all parameters must be frozen")
        if any(
            torch.count_nonzero(p) for n, p in self.auxiliary_state().items() if n.endswith(".B")
        ):
            raise ValueError("trained adapters are prohibited in the training-free runtime")
        expected = {
            f"backbone.layers.{i}.linear_attn.{proj}.{matrix}"
            for i in self.config.adaptation_layers
            for proj in ("in_proj_qkv", "in_proj_a", "in_proj_b")
            for matrix in ("A", "B")
        }
        expected |= {
            f"readers.{i}.{name}"
            for i in self.config.signal_layers
            for name in ("queries", "gate", "A", "B")
        }
        if set(self.auxiliary_state()) != expected:
            raise ValueError(
                "trainable scope must be native GDN K/V/a/b plus memory interface only"
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        *,
        span=None,
        eligible=None,
        protected=None,
        compression=True,
        use_cache=False,
        past_key_values=None,
        debug=False,
        route_seed=0,
        routing="greedy",
    ):
        if self.training:
            raise ValueError("inference-only runtime requires model.eval()")
        if input_ids.ndim != 1 or input_ids.dtype != torch.long or not len(input_ids):
            raise ValueError("use one nonempty unpadded int64 trajectory per worker")
        if routing != "greedy":
            raise ValueError("unknown routing policy")
        # Online randomness belongs to action sampling, not artificial route noise.
        if use_cache and (torch.is_grad_enabled() or self.training):
            raise ValueError("cache requires eval/no_grad and unmodified production routing")
        if past_key_values is not None and not use_cache:
            raise ValueError("cache passed with use_cache=False")
        cache = (
            (MemoryCache(self, compression) if past_key_values is None else past_key_values)
            if use_cache
            else None
        )
        if cache is not None:
            if not isinstance(cache, MemoryCache):
                raise TypeError("requires MemoryCache, not legacy merge/native cache")
            cache.validate(self, compression)
        seq = Sequence.create(
            self.backbone.embed_tokens(input_ids),
            span,
            eligible,
            protected,
            offset=0 if cache is None else cache.logical_length,
        )
        spans = {s for s, _, _ in seq.tool_ranges()} if compression else set()
        # Sequence.create has already validated eligible => span >= 0. With a
        # single non-Tool token, both selected stages would only synchronize to
        # discover eligible.any() == False. Keep all geometry/cache validation.
        skip_signals = (
            self.runtime.skip_empty_decode_signals
            and cache is not None
            and len(input_ids) == 1
            and not spans
        )
        if cache is not None:
            if spans & cache.seen_spans:
                raise ValueError("cannot reopen an already cached Tool observation")
            cache.valid = False  # interrupted partial appends cannot be reused
        lengths, events = [], []
        rope_cached = None
        for index, layer in enumerate(self.backbone.layers):
            n = len(seq.hidden)
            lengths.append(n)
            collect = (
                compression
                and index in self.config.signal_layers
                and not skip_signals
                and bool(seq.eligible.any())
            )
            gate = getattr(self, "_compression_bypass", None)
            bypass = gate(seq, index) if collect and gate is not None else None
            if bypass is not None:
                collect = False
            ranges = seq.tool_ranges() if collect else ()
            x = seq.hidden[None]
            if layer.layer_type == "linear_attention":

                def run(x, layer=layer, ranges=ranges, collect=collect):
                    mixed, signals = gdn_forward(
                        layer.linear_attn,
                        layer.input_layernorm(x),
                        self.config,
                        ranges,
                        None if cache is None else cache.native,
                        collect,
                    )
                    y = x + mixed
                    y = y + layer.mlp(layer.post_attention_layernorm(y))
                    return (
                        (y, signals.delta, signals.value, signals.key, *signals.states)
                        if collect
                        else y
                    )
            else:
                rope_ids = seq.position[None, None].expand(3, 1, -1)
                if self.runtime.reuse_rope:
                    version = None if seq.position.is_inference() else seq.position._version
                    key = (id(seq.position), version, x.dtype, x.device)
                    if rope_cached is None or rope_cached[1] != key:
                        rope_cached = (seq.position, key, self.backbone.rotary_emb(x, rope_ids))
                    rope = rope_cached[2]
                else:
                    rope = self.backbone.rotary_emb(x, rope_ids)
                previous = 0 if cache is None else cache.layer_lengths[index]
                mask = causal_mask(n, previous, x, self.backbone.config._attn_implementation)
                physical = torch.arange(previous, previous + n, device=x.device)

                def run(x, layer=layer, rope=rope, ids=rope_ids, mask=mask, physical=physical):
                    return layer(
                        x,
                        position_embeddings=rope,
                        position_ids=ids,
                        attention_mask=mask,
                        cache_position=physical,
                        past_key_values=None if cache is None else cache.native,
                    )

            result = run(x)
            if collect:
                hidden, delta, value, key, *states = result
                signals = Signals(delta, value, key, tuple(states))
            else:
                hidden = result
            seq = seq.with_hidden(hidden[0])
            if cache is not None:
                cache.layer_lengths[index] += n
                if (
                    layer.layer_type == "full_attention"
                    and cache.native.get_seq_length(index) != cache.layer_lengths[index]
                ):
                    raise RuntimeError("physical KV length mismatch")
            if collect or bypass is not None:
                if bypass is None:
                    seq, event = consolidate(
                        seq, signals, self.readers[str(index)], layer, (index + 2) // 4
                    )
                else:
                    event = bypass
                events.append(event)
                if cache is not None:
                    totals = cache.totals.setdefault(
                        event["stage"],
                        {
                            "stage": event["stage"],
                            "eligible_before": 0,
                            "eligible_after": 0,
                            "migrated": 0,
                            "memory_tokens": 0,
                        },
                    )
                    for key in ("eligible_before", "eligible_after", "migrated", "memory_tokens"):
                        totals[key] += event[key]
        seq = seq.with_hidden(self.backbone.norm(seq.hidden))
        if debug:
            seq.validate()
        if cache is not None:
            cache.logical_length += len(input_ids)
            cache.seen_spans.update(spans)
            cache.valid = True
        return Output(seq, lengths, events, cache)

    @torch.no_grad()
    def native_hidden(self, ids):
        flags = [(m, m.training) for m in self.backbone.modules()]
        try:
            self.backbone.eval()
            with disabled(self.backbone):
                return self.backbone(input_ids=ids[None], use_cache=False).last_hidden_state[0]
        finally:
            for m, flag in flags:
                m.training = flag

    def next_logits(self, input_ids, **kwargs):
        kwargs.setdefault("use_cache", True)
        result = self(input_ids, **kwargs)
        if int(result.sequence.source[-1]) < 0:
            raise ValueError("generation predictor must be an original protected boundary")
        return self.lm_head(result.sequence.hidden[-1]), result


def load_model(path, config=None, device="cuda:0"):
    import transformers

    if transformers.__version__ != "5.2.0":
        raise RuntimeError("model bridge requires validated Transformers 5.2.0")
    transformers.utils.logging.disable_progress_bar()
    native = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
        path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": device},
    )
    return MemoryInnovation(native.model.language_model, native.lm_head, config)
