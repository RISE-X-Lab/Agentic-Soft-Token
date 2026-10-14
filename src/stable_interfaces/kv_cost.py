"""Exact causal-pair geometry proxies; these are not measured HBM transactions."""

ATTENTION_LAYERS = tuple(range(3, 32, 4))


def trajectory_steps(record):
    previous = [0] * len(record["metrics"][0]["cache_layer_lengths"])
    logical = 0
    steps = []
    for index, m in enumerate(record["metrics"]):
        d = m["generated_tokens"] - 1
        if d < 0:
            raise ValueError("complete nonempty generation calls required")
        after = m["cache_layer_lengths"]
        prefill = [x - d for x in after]
        end = m["logical_prompt_tokens"] + d
        if m["actual_forward_input_tokens"] != end - logical:
            raise ValueError("input boundary mismatch")
        steps.append(
            dict(
                index=index,
                previous=previous,
                prefill=prefill,
                after=after,
                logical_previous=logical,
                logical_prefill=m["logical_prompt_tokens"],
                decode_tokens=d,
            )
        )
        previous = after
        logical = end
    return steps


def summarize_kv(steps, *, layers=ATTENTION_LAYERS, bytes_per_kv_position=4096):
    per_layer = {
        i: dict(
            native_decode_kv_visits=0,
            actual_decode_kv_visits=0,
            native_prefill_causal_pairs=0,
            actual_prefill_causal_pairs=0,
        )
        for i in layers
    }
    trace = []
    for s in steps:
        d = s["decode_tokens"]
        native_previous = s["logical_previous"]
        native_prefill = s["logical_prefill"]
        native_q = native_prefill - native_previous
        if d < 0 or native_q <= 0:
            raise ValueError("invalid logical append")
        saved = 0
        for i in layers:
            previous = s["previous"][i]
            prefill = s["prefill"][i]
            q = prefill - previous
            if q <= 0 or prefill > native_prefill or s["after"][i] != prefill + d:
                raise ValueError("invalid physical cache append")
            row = per_layer[i]
            native_visits = d * native_prefill + d * (d + 1) // 2
            actual_visits = d * prefill + d * (d + 1) // 2
            row["native_decode_kv_visits"] += native_visits
            row["actual_decode_kv_visits"] += actual_visits
            row["native_prefill_causal_pairs"] += (
                native_q * native_previous + native_q * (native_q + 1) // 2
            )
            row["actual_prefill_causal_pairs"] += q * previous + q * (q + 1) // 2
            saved += native_visits - actual_visits
        trace.append(
            dict(
                step=s["index"],
                decode_tokens=d,
                logical_prefill=native_prefill,
                actual_kv_lengths={str(i): s["prefill"][i] for i in layers},
                native_kv_bytes_after=(native_prefill + d) * len(layers) * bytes_per_kv_position,
                actual_kv_bytes_after=sum(s["after"][i] for i in layers) * bytes_per_kv_position,
                saved_decode_kv_visits=saved,
            )
        )
    totals = {k: sum(r[k] for r in per_layer.values()) for k in next(iter(per_layer.values()))}
    for phase, unit in (("decode", "kv_visits"), ("prefill", "causal_pairs")):
        base = totals[f"native_{phase}_{unit}"]
        actual = totals[f"actual_{phase}_{unit}"]
        totals[f"{phase}_reduction"] = 1 - actual / base if base else None
    totals["saved_logical_decode_kv_bytes"] = (
        totals["native_decode_kv_visits"] - totals["actual_decode_kv_visits"]
    ) * bytes_per_kv_position
    totals["native_kv_bytes_peak"] = max(x["native_kv_bytes_after"] for x in trace)
    totals["actual_kv_bytes_peak"] = max(x["actual_kv_bytes_after"] for x in trace)
    totals["kv_peak_reduction"] = (
        1 - totals["actual_kv_bytes_peak"] / totals["native_kv_bytes_peak"]
    )
    return dict(
        totals=totals,
        by_layer={str(k): v for k, v in per_layer.items()},
        steps=trace,
        bytes_per_kv_position=bytes_per_kv_position,
        note="Decode counts logical KV positions presented to each query, including its own key. Bytes are a geometry proxy, not measured HBM traffic. Prefill counts causally valid pairs, not padded kernel work.",
    )
