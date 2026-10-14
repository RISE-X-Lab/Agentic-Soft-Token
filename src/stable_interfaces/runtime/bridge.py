"""Native projections/causal convolution; scans split ONLY at Tool ends."""

from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass
class Signals:
    delta: torch.Tensor  # [N,H,V]
    value: torch.Tensor
    key: torch.Tensor
    states: tuple[torch.Tensor, ...]  # matching tool_ranges order


def gdn_forward(gdn, hidden, config, ranges=(), cache=None, collect=False):
    b, n, _ = hidden.shape
    index, kernel = gdn.layer_idx, gdn.conv_kernel_size
    raw = gdn.in_proj_qkv(hidden).transpose(1, 2)
    previous_conv = None if cache is None else cache.conv_states[index]
    state = None if cache is None else cache.recurrent_states[index]
    if (previous_conv is None) != (state is None):
        raise ValueError("incomplete GDN convolution/state cache")
    if previous_conv is not None and n == 1:
        mixed = gdn.causal_conv1d_update(
            raw, previous_conv, gdn.conv1d.weight.squeeze(1), gdn.conv1d.bias, gdn.activation
        )
        final_conv = previous_conv
    else:
        history = (
            raw
            if previous_conv is None
            else torch.cat(
                (
                    previous_conv[:, :, -(kernel - 1) :] if kernel > 1 else previous_conv[:, :, :0],
                    raw,
                ),
                -1,
            )
        )
        final_conv = F.pad(history, (kernel - history.shape[-1], 0)).contiguous()
        if gdn.causal_conv1d_fn is not None:
            mixed = gdn.causal_conv1d_fn(
                x=history,
                weight=gdn.conv1d.weight.squeeze(1),
                bias=gdn.conv1d.bias,
                activation=gdn.activation,
                seq_idx=None,
            )
        else:
            mixed = F.silu(gdn.conv1d(history)[:, :, : history.shape[-1]])
        mixed = mixed[:, :, -n:]
    q, k, v = mixed.transpose(1, 2).split((gdn.key_dim, gdn.key_dim, gdn.value_dim), -1)
    q, k = [x.reshape(b, n, gdn.num_k_heads, gdn.head_k_dim) for x in (q, k)]
    copies = gdn.num_v_heads // gdn.num_k_heads
    q, k = [x.repeat_interleave(copies, 2) if copies > 1 else x for x in (q, k)]
    v = v.reshape(b, n, gdn.num_v_heads, gdn.head_v_dim)
    z = gdn.in_proj_z(hidden).reshape(b, n, gdn.num_v_heads, gdn.head_v_dim)
    beta = gdn.in_proj_b(hidden).sigmoid()
    g = -gdn.A_log.float().exp() * F.softplus(gdn.in_proj_a(hidden).float() + gdn.dt_bias)
    signals = None
    if collect:
        if config.backend != "oracle":
            raise RuntimeError("CUDA signal collection requires the native-signal scope")
        from .recurrence import scan

        q, k, v, g, beta = [x.float() for x in (q, k, v, g, beta)]
        endpoints = [end for _, _, end in ranges]
        ends = endpoints + ([n] if not endpoints or endpoints[-1] != n else [])
        outputs, deltas, keys, states, start = [], [], [], [], 0
        for end in ends:
            out, delta, state, key = scan(
                q[:, start:end],
                k[:, start:end],
                v[:, start:end],
                g[:, start:end],
                beta[:, start:end],
                state,
            )
            outputs.append(out)
            deltas.append(delta)
            keys.append(key)
            if end in endpoints:
                states.append(state[0])
            start = end
        out = torch.cat(outputs, 1).to(hidden.dtype)
        signals = Signals(torch.cat(deltas, 1)[0], v[0], torch.cat(keys, 1)[0], tuple(states))
    elif config.backend == "oracle":
        from .recurrence import scan

        out, _, state, _ = scan(q.float(), k.float(), v.float(), g.float(), beta.float(), state)
        out = out.to(hidden.dtype)
    else:
        fn = (
            gdn.recurrent_gated_delta_rule
            if cache is not None and n == 1
            else gdn.chunk_gated_delta_rule
        )
        out, state = fn(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    if cache is not None:
        cache.conv_states[index] = final_conv
        cache.recurrent_states[index] = state
    normed = gdn.norm(out.reshape(-1, gdn.head_v_dim), z.reshape(-1, gdn.head_v_dim))
    return gdn.out_proj(normed.reshape(b, n, gdn.value_dim)), signals
