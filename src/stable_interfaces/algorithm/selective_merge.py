"""Native-signal selection and profitable Stage-3 soft-token commits."""

from contextlib import contextmanager
import math
import torch
from stable_interfaces.runtime.geometry import Sequence
from stable_interfaces.runtime.memory import rms
from .policy import configured_scope
from .pooling import projected_error, adaptive_pool


@contextmanager
def merge_scope(model, plan, audit=None):
    if plan is None:
        yield
        return
    with configured_scope(model, plan):
        import stable_interfaces.runtime.model as implementation

        original = implementation.consolidate
        old_policy = model._merge_policy
        had_gate = hasattr(model, "_compression_bypass")
        old_gate = getattr(model, "_compression_bypass", None)
        rules = {r.stage: r for r in plan.rules}
        owned = {id(r) for r in model.readers.values()}

        def bypass(seq, index):
            stage = (index + 2) // 4
            rule = rules[stage]
            minimum = max(rule.min_net, math.ceil(plan.min_layer_savings / (33 - 4 * stage)))
            before = 0
            for _, start, end in seq.tool_ranges():
                eligible = seq.eligible[start:end]
                before += int(eligible.sum())
                selectable = (
                    eligible & (seq.source[start:end] >= 0) if plan.protect_memory else eligible
                )
                closed = plan.one_stage_per_tool and bool((seq.source[start:end] < 0).any())
                required = max(minimum, math.ceil(plan.min_tool_fraction * int(eligible.sum()))) + 1
                if not closed and int(selectable.sum()) >= required:
                    return None
            event = dict(
                stage=stage,
                tokens_before=len(seq.hidden),
                tokens_after=len(seq.hidden),
                eligible_before=before,
                eligible_after=before,
                migrated=0,
                memory_tokens=0,
                traces=[],
                plan={},
            )
            if audit is not None:
                audit.append(
                    dict(
                        **event,
                        signal_bypass=True,
                        psi_positions=0,
                        chi_positions=0,
                        skipped_psi=0,
                        rejected_by_group=0,
                        tools=[],
                        min_net=minimum,
                    )
                )
            return event

        def consolidate(seq, signals, reader, layer, stage, **kw):
            if id(reader) not in owned:
                return original(seq, signals, reader, layer, stage, **kw)
            rule = rules[stage]
            slots = reader.config.slots
            minimum = max(rule.min_net, math.ceil(plan.min_layer_savings / (33 - 4 * stage)))
            fields = {name: [] for name in Sequence.__dataclass_fields__}
            traces = []
            counts = dict(eligible_before=0, eligible_after=0, migrated=0, memory_tokens=0)
            stats = dict(psi_positions=0, chi_positions=0, skipped_psi=0, rejected_by_group=0)
            ranges = seq.tool_ranges()
            cursor = 0
            if len(ranges) != len(signals.states):
                raise ValueError("Tool boundary mismatch")

            def keep(ids):
                for name in fields:
                    fields[name].append(getattr(seq, name)[ids])

            for (tool, start, end), state in zip(ranges, signals.states):
                keep(slice(cursor, start))
                cursor = end
                eligible = seq.eligible[start:end]
                before = int(eligible.sum())
                counts["eligible_before"] += before
                selectable = (
                    eligible & (seq.source[start:end] >= 0) if plan.protect_memory else eligible
                )
                local_minimum = max(minimum, math.ceil(plan.min_tool_fraction * before))
                if plan.one_stage_per_tool and bool((seq.source[start:end] < 0).any()):
                    counts["eligible_after"] += before
                    keep(slice(start, end))
                    if audit is not None:
                        traces.append(
                            dict(
                                tool=tool,
                                closed_by_earlier_stage=True,
                                members=[],
                                soft_tokens=0,
                                max_merge_error=None,
                            )
                        )
                    continue
                value = signals.value[start:end]
                chi = rms(signals.delta[start:end]) / (rms(value) + reader.config.epsilon)
                if not bool(torch.isfinite(chi).all()):
                    raise FloatingPointError("nonfinite innovation")
                candidate = ((chi < rule.chi) & selectable).nonzero().flatten()
                stats["chi_positions"] += len(value)
                members = seq.source.new_empty(0)
                z = seq.hidden[:0]
                endpoint = seq.position[:0]
                worst = chi[:0]
                needed = local_minimum + 1
                if len(candidate) >= needed:
                    key = signals.key[start:end][candidate]
                    v = value[candidate]
                    psi = projected_error(reader, state, key, v)
                    stats["psi_positions"] += len(candidate)
                    if not bool(torch.isfinite(psi).all()):
                        raise FloatingPointError("nonfinite readability")
                    chosen = candidate[psi < rule.psi] + start
                    if len(chosen) >= needed:
                        ids, labels, z, endpoint, worst = adaptive_pool(
                            seq.hidden[chosen],
                            seq.position[chosen],
                            slots,
                            plan.merge_error,
                            plan.members_per_slot,
                        )
                        members = chosen[ids]
                        stats["rejected_by_group"] += len(chosen) - len(members)
                        if len(members) - len(z) < local_minimum:
                            members = members[:0]
                            z = z[:0]
                            endpoint = endpoint[:0]
                else:
                    stats["skipped_psi"] += 1
                counts["migrated"] += len(members)
                counts["memory_tokens"] += len(z)
                counts["eligible_after"] += before - len(members) + len(z)
                if not len(z):
                    keep(slice(start, end))
                else:
                    mask = torch.ones(end - start, device=seq.hidden.device, dtype=torch.bool)
                    mask[members - start] = False
                    keep(torch.arange(start, end, device=seq.hidden.device)[mask])
                    fields["hidden"].append(z)
                    fields["position"].append(endpoint)
                    for name, val in (("source", -1), ("span", tool), ("kind", stage)):
                        fields[name].append(getattr(seq, name).new_full((len(z),), val))
                    fields["eligible"].append(eligible.new_ones(len(z)))
                    fields["protected"].append(eligible.new_zeros(len(z)))
                    fields["slot"].append(torch.arange(len(z), device=z.device))
                if audit is not None:
                    traces.append(
                        dict(
                            tool=tool,
                            chi_candidates=len(candidate),
                            members=seq.source[members].cpu().tolist(),
                            soft_tokens=len(z),
                            max_merge_error=float(worst.max()) if len(z) and len(worst) else None,
                        )
                    )
            keep(slice(cursor, None))
            # Nothing moved: retain original tensor identity and reusable RoPE.
            if counts["migrated"]:
                data = {k: torch.cat(v) for k, v in fields.items()}
                order = data["position"].argsort(stable=True)
                result = Sequence(**{k: v[order] for k, v in data.items()})
            else:
                result = seq
            event = dict(
                stage=stage,
                tokens_before=len(seq.hidden),
                tokens_after=len(result.hidden),
                **counts,
                traces=[],
                plan={},
            )
            if audit is not None:
                result.validate()
                audit.append(dict(**event, **stats, tools=traces, min_net=minimum))
            return result, event

        implementation.consolidate = consolidate
        model._merge_policy = plan
        if plan.preflight:
            model._compression_bypass = bypass
        try:
            yield
        finally:
            implementation.consolidate = original
            model._merge_policy = old_policy
            if had_gate:
                model._compression_bypass = old_gate
            elif hasattr(model, "_compression_bypass"):
                del model._compression_bypass
