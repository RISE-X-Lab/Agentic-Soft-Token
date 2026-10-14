"""Differentiable reference only, for tiny correctness/gradient tests."""

import torch


def normalize(x):
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)


def scan(q, k, v, g, beta, initial_state=None, normalized=False):
    if not normalized:
        q, k = normalize(q), normalize(k)
    state = (
        v.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
        if initial_state is None
        else initial_state
    )
    outputs, residuals = [], []
    for t in range(q.shape[1]):
        prior = state * g[:, t].exp()[..., None, None]
        delta = v[:, t] - torch.einsum("bhkv,bhk->bhv", prior, k[:, t])
        state = prior + k[:, t, :, :, None] * (beta[:, t, :, None] * delta)[..., None, :]
        outputs.append(torch.einsum("bhkv,bhk->bhv", state, q[:, t]) / q.shape[-1] ** 0.5)
        residuals.append(delta)
    return torch.stack(outputs, 1), torch.stack(residuals, 1), state, k
