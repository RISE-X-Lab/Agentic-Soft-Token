"""Physical order is causal; logical positions are only RoPE coordinates.

Memory tokens have source=-1 and may share a logical position. There is no
fictional contiguous source interval or one-token/one-position assumption.
"""

from dataclasses import dataclass, replace
import torch


@dataclass
class Sequence:
    hidden: torch.Tensor
    position: torch.Tensor
    source: torch.Tensor
    span: torch.Tensor
    eligible: torch.Tensor
    protected: torch.Tensor
    kind: torch.Tensor  # 0 original; otherwise the compression stage
    slot: torch.Tensor  # -1 original; memory bundle slot ordinal

    @classmethod
    def create(cls, hidden, span=None, eligible=None, protected=None, offset=0):
        n = len(hidden)
        pos = torch.arange(offset, offset + n, device=hidden.device)
        span = torch.full_like(pos, -1) if span is None else span
        eligible = span >= 0 if eligible is None else eligible
        protected = ~eligible if protected is None else protected
        seq = cls(
            hidden,
            pos,
            pos.clone(),
            span,
            eligible,
            protected,
            torch.zeros_like(pos),
            torch.full_like(pos, -1),
        )
        seq.validate()
        return seq

    def with_hidden(self, hidden):
        return replace(self, hidden=hidden)

    def validate(self):
        n = len(self.hidden)
        for field in ("position", "source", "span", "eligible", "protected", "kind", "slot"):
            t = getattr(self, field)
            if t.shape != (n,) or t.device != self.hidden.device:
                raise ValueError("inconsistent sequence metadata")
        if self.eligible.dtype != torch.bool or self.protected.dtype != torch.bool:
            raise ValueError("invalid mask dtype")
        if bool((self.eligible & (self.protected | (self.span < 0))).any()):
            raise ValueError("protected position cannot migrate")
        if bool((self.position[1:] < self.position[:-1]).any()):
            raise ValueError("logical positions must be monotone, not necessarily unique")
        originals = self.source[self.source >= 0]
        if len(originals) and bool((originals[1:] <= originals[:-1]).any()):
            raise ValueError("original token order was changed")
        if not torch.equal(self.source < 0, self.kind != 0):
            raise ValueError("memory/original identity mismatch")
        return self

    def tool_ranges(self):
        values = self.span.detach().cpu().tolist()
        seen, spans, i = set(), [], 0
        while i < len(values):
            if values[i] < 0:
                i += 1
                continue
            span, start = values[i], i
            while i < len(values) and values[i] == span:
                i += 1
            if span in seen:
                raise ValueError("a Tool span cannot reopen")
            seen.add(span)
            if i == len(values):
                raise ValueError("complete Tool observation must include its closing boundary")
            spans.append((span, start, i))
        return spans


def causal_mask(n, previous, hidden, implementation="sdpa"):
    if implementation == "sdpa" and (previous == 0 or n == 1):
        return None
    q = previous + torch.arange(n, device=hidden.device)
    k = torch.arange(previous + n, device=hidden.device)
    allowed = k[None, :] <= q[:, None]
    mask = torch.zeros((n, previous + n), dtype=hidden.dtype, device=hidden.device)
    return mask.masked_fill(~allowed, torch.finfo(hidden.dtype).min)[None, None]
