"""Read native FLA intermediates without rebuilding the model forward."""

import torch


def api():
    import fla

    if fla.__version__ != "0.4.1":
        raise RuntimeError("native signal reads require audited FLA 0.4.1")
    from fla.ops.gated_delta_rule import chunk

    return chunk


def shifted_query(k):
    return torch.cat((k[:, 1:], torch.zeros_like(k[:, :1])), 1).contiguous()


def read_fp32(query, key, write, states, g):
    import triton
    from fla.ops.common.chunk_o import chunk_fwd_kernel_o

    b, n, heads, dk = query.shape
    result = torch.empty(write.shape, dtype=torch.float32, device=write.device)

    def grid(meta):
        return (triton.cdiv(write.shape[-1], meta["BV"]), triton.cdiv(n, 64), b * heads)

    chunk_fwd_kernel_o[grid](
        q=query,
        k=key,
        v=write,
        h=states,
        g=g,
        g_gamma=None,
        o=result,
        cu_seqlens=None,
        chunk_indices=None,
        scale=1.0,
        T=n,
        H=heads,
        K=dk,
        V=write.shape[-1],
        BT=64,
    )
    return result


def prior_read(k, h0, write, states, cumulative_g):
    first = torch.einsum("bhkv,bhk->bhv", h0.float(), k[:, 0].float())[:, None]
    return torch.cat(
        (first, read_fp32(shifted_query(k), k, write, states, cumulative_g)[:, :-1]), 1
    )
