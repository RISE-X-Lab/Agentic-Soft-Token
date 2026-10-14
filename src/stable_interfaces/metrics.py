"""Task-level geometry ledgers. These counts are not measured FLOPs or latency."""

from .kv_cost import trajectory_steps, summarize_kv


def record_kv(record):
    measurements, empty = [], []
    for i, metric in enumerate(record["metrics"]):
        if metric["generated_tokens"]:
            measurements.append(metric)
        else:
            if metric["actual_forward_input_tokens"] or metric["token_layer_positions"]:
                raise ValueError("zero-output generation performed unaccounted model work")
            empty.append(i)
    if not measurements:
        return dict(no_forward_calls=True, zero_forward_generation_calls=empty)
    cost = summarize_kv(trajectory_steps({"metrics": measurements}))
    cost["zero_forward_generation_calls"] = empty
    cost["nonempty_generation_indices"] = [
        i for i, m in enumerate(record["metrics"]) if m["generated_tokens"]
    ]
    return cost


def validate_native(record):
    for metric in record["metrics"]:
        if metric["token_layer_positions"] != 32 * metric["actual_forward_input_tokens"]:
            raise ValueError("native token stream shortened inside the model")
        if metric["last_forward_compression"]:
            raise ValueError("native model produced compression events")
        if metric["generated_tokens"]:
            expected = metric["logical_prompt_tokens"] + metric["generated_tokens"] - 1
            if metric["cache_layer_lengths"] != [expected] * 32:
                raise ValueError("native cache does not contain the complete logical prefix")
    return record_kv(record)


def task_row(record):
    cost = validate_native(record) if record.get("native") else record_kv(record)
    if record.get("kv_cost") != cost:
        raise ValueError("stored KV ledger differs from the recorded cache geometry")
    totals = cost.get("totals", {})
    return dict(
        task=record["task_id"],
        success=record["official_score"]["success"],
        steps=len(record["turns"]),
        generated_tokens=sum(m["generated_tokens"] for m in record["metrics"]),
        actual_decode_kv=totals.get("actual_decode_kv_visits", 0),
        actual_prefill_pairs=totals.get("actual_prefill_causal_pairs", 0),
    )


def aggregate(rows):
    if not rows or len({row["task"] for row in rows}) != len(rows):
        raise ValueError("empty or duplicated task results")
    success = sum(row["success"] for row in rows)
    kv = sum(row["actual_decode_kv"] for row in rows)
    prefill = sum(row["actual_prefill_pairs"] for row in rows)
    return dict(
        tasks=len(rows),
        success=success,
        success_rate=success / len(rows),
        steps=sum(row["steps"] for row in rows),
        mean_steps=sum(row["steps"] for row in rows) / len(rows),
        generated_tokens=sum(row["generated_tokens"] for row in rows),
        decode_kv=kv,
        prefill_pairs=prefill,
        attention_pairs=kv + prefill,
        kv_per_success=kv / success if success else None,
    )


def compare(groups):
    if "base" not in groups:
        raise ValueError("one method must be named base")
    expected = {row["task"] for row in groups["base"]}
    if any({row["task"] for row in rows} != expected for rows in groups.values()):
        raise ValueError("all methods must have exactly the same task IDs")
    totals = {name: aggregate(rows) for name, rows in groups.items()}
    base = totals["base"]
    if not base["decode_kv"] or not base["attention_pairs"]:
        raise ValueError("baseline must have nonzero measured work")
    for row in totals.values():
        row["decode_kv_vs_base"] = row["decode_kv"] / base["decode_kv"]
        row["attention_pairs_vs_base"] = row["attention_pairs"] / base["attention_pairs"]
        row["kv_per_success_reduction"] = (
            1 - row["kv_per_success"] / base["kv_per_success"]
            if row["kv_per_success"] is not None and base["kv_per_success"] is not None
            else None
        )
    return totals
