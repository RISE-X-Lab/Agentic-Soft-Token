"""Optional stochastic decoding; released Test20 uses greedy decoding."""

import torch


def sample_token(logits, generator, temperature=0.6, top_k=20):
    if logits.ndim != 1 or not torch.isfinite(logits).all() or temperature <= 0:
        raise ValueError("invalid next-token logits/temperature")
    if not 1 <= top_k <= logits.numel():
        raise ValueError("invalid top-k")
    probabilities = (logits.float() / temperature).softmax(-1)
    values, ids = probabilities.topk(top_k)
    return int(ids[torch.multinomial(values / values.sum(), 1, generator=generator)])
