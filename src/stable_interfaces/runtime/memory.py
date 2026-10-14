"""Native residual + actual state-read coverage, with no pruning quota."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from .geometry import Sequence


def rms(x):
    # vector_norm has a defined zero subgradient; sqrt(mean(x*x)) does not.
    return torch.linalg.vector_norm(x.float().flatten(-2), dim=-1) / math.sqrt(
        x.shape[-2] * x.shape[-1]
    )


class MemoryReader(nn.Module):
    def __init__(self, gdn, config):
        super().__init__()
        h, dk, dv, m = gdn.num_v_heads, gdn.head_k_dim, gdn.head_v_dim, config.slots
        if m > dk:
            raise ValueError("memory slots cannot exceed per-head key dimension")
        self.config = config
        self.elide_zero_output = False
        self.cache_fixed_terms = False
        self._fixed_terms_cache = None
        q = torch.linalg.qr(
            torch.randn(h, dk, m, device=gdn.out_proj.weight.device), mode="reduced"
        ).Q
        self.queries = nn.Parameter(q.float())
        self.gate = nn.Parameter(torch.ones(m, h, 1, device=q.device))
        self.A = nn.Parameter(torch.empty(config.reader_rank, h * dv, device=q.device))
        self.B = nn.Parameter(
            torch.zeros(gdn.out_proj.out_features, config.reader_rank, device=q.device)
        )
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def fixed_terms(self):
        # Keep a strong reference to queries; replacements, in-place updates,
        # device/dtype/config and autocast changes must not reuse stale math.
        cacheable = (
            self.cache_fixed_terms
            and not torch.is_grad_enabled()
            and not self.queries.requires_grad
        )
        device_type = self.queries.device.type
        autocast = torch.is_autocast_enabled(device_type)
        signature = (
            id(self.queries),
            self.queries._version,
            self.queries.device,
            self.queries.dtype,
            self.config.epsilon,
            self.config.ridge,
            autocast,
            torch.get_autocast_dtype(device_type) if autocast else None,
        )
        cached = self._fixed_terms_cache
        if cacheable and cached is not None and cached[1] == signature:
            return cached[2], cached[3]
        q = F.normalize(self.queries.float(), dim=1, eps=self.config.epsilon)
        gram = q.transpose(1, 2) @ q
        gram = gram + self.config.ridge * torch.eye(gram.shape[-1], device=gram.device)
        self._fixed_terms_cache = (self.queries, signature, q, gram) if cacheable else None
        return q, gram

    def read(self, state, key, value):
        # Every memory slot reads the SAME actual state; there is no hidden copy.
        if self.cache_fixed_terms:
            q, gram = self.fixed_terms()
            raw = torch.einsum("hkv,hkm->hmv", state.float(), q)
        else:
            q = F.normalize(self.queries.float(), dim=1, eps=self.config.epsilon)
            raw = torch.einsum("hkv,hkm->hmv", state.float(), q)
            gram = q.transpose(1, 2) @ q
            gram = gram + self.config.ridge * torch.eye(gram.shape[-1], device=gram.device)
        coefficients = torch.linalg.solve(gram, torch.einsum("hkm,nhk->hmn", q, key.float()))
        prediction = torch.einsum("hmv,hmn->nhv", raw, coefficients)
        # Global head/value RMS avoids division by near-zero individual heads.
        denominator = rms(value) + self.config.epsilon
        error = rms(value.float() - prediction) / denominator
        return raw.transpose(0, 1), error

    def express(self, raw, layer):
        gdn = layer.linear_attn
        dtype = gdn.out_proj.weight.dtype
        # Reuse native frozen gated norm and output projection only on reader branch.
        gate = self.gate.expand_as(raw)
        normed = gdn.norm(
            raw.to(dtype).reshape(-1, raw.shape[-1]), gate.to(dtype).reshape(-1, raw.shape[-1])
        ).reshape(len(raw), -1)
        if self.elide_zero_output:
            z = gdn.out_proj(normed)
        else:
            update = F.linear(F.linear(normed.float(), self.A), self.B) * (
                self.config.reader_alpha / self.config.reader_rank
            )
            z = gdn.out_proj(normed) + update.to(dtype)
        return z + layer.mlp(layer.post_attention_layernorm(z))


def consolidate(seq, signals, reader, layer, stage):
    cfg = reader.config
    ranges = seq.tool_ranges()
    if len(ranges) != len(signals.states):
        raise ValueError("Tool boundary state alignment mismatch")
    fields = {
        name: []
        for name in (
            "hidden",
            "position",
            "source",
            "span",
            "eligible",
            "protected",
            "kind",
            "slot",
        )
    }
    traces, plans, counts = (
        [],
        {},
        dict(eligible_before=0, eligible_after=0, migrated=0, memory_tokens=0),
    )

    def original(indices):
        for name in fields:
            fields[name].append(getattr(seq, name)[indices])

    cursor = 0
    for (tool, start, end), state in zip(ranges, signals.states):
        original(slice(cursor, start))
        indices = torch.arange(start, end, device=seq.hidden.device)
        value = signals.value[start:end]
        denominator = rms(value) + cfg.epsilon
        chi = rms(signals.delta[start:end]) / denominator
        raw, psi = reader.read(state, signals.key[start:end], value)
        score = chi / cfg.chi_threshold
        if cfg.readability_check:
            score = torch.maximum(score, psi / cfg.readability_threshold)
        if not bool(torch.isfinite(score).all()):
            raise FloatingPointError("nonfinite memory-routing score")
        eligible = seq.eligible[start:end]
        migrate = (score.detach() < 1) & eligible
        proposed = int(migrate.sum())
        # The interface must pay for itself; never silently expand a short Tool.
        execute = proposed > cfg.slots
        actual = migrate if execute else torch.zeros_like(migrate)
        counts["eligible_before"] += int(eligible.sum())
        counts["eligible_after"] += int(eligible.sum()) - (proposed - cfg.slots if execute else 0)
        counts["migrated"] += proposed if execute else 0
        counts["memory_tokens"] += cfg.slots if execute else 0
        original(indices[~actual])
        if execute:
            z = reader.express(raw, layer)
            m, pos = len(z), seq.position[end - 1]
            fields["hidden"].append(z)
            fields["position"].append(pos.expand(m))
            fields["source"].append(seq.source.new_full((m,), -1))
            fields["span"].append(seq.span.new_full((m,), tool))
            fields["eligible"].append(eligible.new_ones(m))
            fields["protected"].append(eligible.new_zeros(m))
            fields["kind"].append(seq.kind.new_full((m,), stage))
            fields["slot"].append(torch.arange(m, device=z.device))
        cursor = end
    original(slice(cursor, None))
    output = Sequence(**{name: torch.cat(chunks, 0) for name, chunks in fields.items()})
    event = dict(
        stage=stage,
        tokens_before=len(seq.hidden),
        tokens_after=len(output.hidden),
        **counts,
        traces=traces,
        plan=plans,
    )
    return output, event
