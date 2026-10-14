"""State-readability projection and bounded pooling of full hidden states."""

import torch
from stable_interfaces.runtime.memory import rms


def projected_error(reader, state, key, value):
    """Same ridge projection algebra; floating-point reassociation is tested separately."""
    q, gram = reader.fixed_terms()
    signature = (
        id(reader.queries),
        reader.queries._version,
        reader.queries.device,
        reader.queries.dtype,
        reader.config.epsilon,
        reader.config.ridge,
    )
    previous = getattr(reader, "_cascade_projection_cache", None)
    if previous is None or previous[0] != signature:
        projection = torch.linalg.solve(gram, q.transpose(1, 2))
        reader._cascade_projection_cache = (signature, projection)
    else:
        projection = previous[1]
    raw = torch.einsum("hkv,hkm->hmv", state.float(), q)
    coefficients = torch.einsum("hmk,nhk->hmn", projection, key.float())
    prediction = torch.einsum("hmv,hmn->nhv", raw, coefficients)
    return rms(value.float() - prediction) / (rms(value) + reader.config.epsilon)


def adaptive_pool(hidden, position, slots, tolerance, members_per_slot):
    """Trim outliers once, then accept only groups passing the ACTUAL BF16 mean error bound.

    Return original member indices, their compact groups, actual soft states/positions.
    At most slots groups; singletons and rejected/outlier positions remain explicit.
    This is a hidden reconstruction bound, not an Attention or decision guarantee.
    """
    n = len(hidden)
    m = min(slots, max(1, n // members_per_slot))
    if n <= m:
        raise ValueError("candidate count must exceed slots")
    labels = torch.arange(n, device=hidden.device) * m // n
    x = hidden.float()
    den = torch.linalg.vector_norm(x, dim=-1).clamp_min(1e-6)

    def center(ids, groups):
        mass = torch.bincount(groups, minlength=m)
        z = x.new_zeros((m, x.shape[-1])).index_add_(0, groups, x[ids]) / mass.clamp_min(1)[:, None]
        return z.to(hidden.dtype).float(), mass

    ids = torch.arange(n, device=x.device)
    # Use an existing member nearest the mean direction as the first center.
    # A large-norm outlier must not drag the preliminary arithmetic mean away
    # from a coherent majority and cause that majority to be discarded.
    unit = x / den[:, None]
    direction = x.new_zeros((m, x.shape[-1])).index_add_(0, labels, unit)
    affinity = (unit * direction[labels]).sum(-1)
    best = x.new_full((m,), -float("inf")).scatter_reduce_(
        0, labels, affinity, reduce="amax", include_self=True
    )
    possible = torch.where(affinity == best[labels], ids, n)
    anchors = ids.new_full((m,), n).scatter_reduce_(
        0, labels, possible, reduce="amin", include_self=True
    )
    z = x[anchors]
    err = torch.linalg.vector_norm(x - z[labels], dim=-1) / den
    ids = ids[err <= tolerance]
    groups = labels[ids]
    z, mass = center(ids, groups)
    err = torch.linalg.vector_norm(x[ids] - z[groups], dim=-1) / den[ids]
    worst = x.new_zeros(m).scatter_reduce_(0, groups, err, reduce="amax", include_self=True)
    accepted = (mass >= 2) & (worst <= tolerance)
    keep = accepted[groups]
    ids = ids[keep]
    groups = groups[keep]
    active = accepted.nonzero().flatten()
    remap = torch.full((m,), -1, device=x.device, dtype=torch.long)
    remap[active] = torch.arange(len(active), device=x.device)
    endpoint = position.new_full((m,), -1).scatter_reduce_(
        0, groups, position[ids], reduce="amax", include_self=True
    )
    return ids, remap[groups], z[active].to(hidden.dtype), endpoint[active], worst[active]
