import pytest
from stable_interfaces.metrics import aggregate, compare, record_kv
from stable_interfaces.kv_cost import summarize_kv
from stable_interfaces.report import reference_rows, markdown


def row(task, success, kv, steps=1):
    return dict(
        task=task,
        success=success,
        actual_decode_kv=kv,
        actual_prefill_pairs=100,
        steps=steps,
        generated_tokens=2,
    )


def test_amortization_includes_failures_and_zero_success_is_undefined():
    result = aggregate([row("a", True, 10), row("b", False, 90, 7)])
    assert result["kv_per_success"] == 100 and result["mean_steps"] == 4
    assert aggregate([row("a", False, 10)])["kv_per_success"] is None
    with pytest.raises(ValueError, match="duplicated"):
        aggregate([row("a", True, 10), row("a", False, 20)])
    with pytest.raises(ValueError, match="same task"):
        compare({"base": [row("a", True, 10)], "ours": [row("b", True, 5)]})


def test_decode_counts_each_new_query_and_prefill_separately():
    steps = [
        dict(
            index=0,
            previous=[0] * 32,
            prefill=[3] * 32,
            after=[5] * 32,
            logical_previous=0,
            logical_prefill=3,
            decode_tokens=2,
        ),
        dict(
            index=1,
            previous=[5] * 32,
            prefill=[7] * 32,
            after=[8] * 32,
            logical_previous=5,
            logical_prefill=8,
            decode_tokens=1,
        ),
    ]
    totals = summarize_kv(steps)["totals"]
    assert totals["actual_decode_kv_visits"] == 8 * (4 + 5 + 8)
    assert totals["native_decode_kv_visits"] == 8 * (4 + 5 + 9)
    assert totals["actual_prefill_causal_pairs"] == 8 * (1 + 2 + 3 + 6 + 7)
    assert totals["native_prefill_causal_pairs"] == 8 * (1 + 2 + 3 + 6 + 7 + 8)


def test_empty_generation_is_not_charged_or_silently_discarded():
    metric = dict(generated_tokens=0, actual_forward_input_tokens=0, token_layer_positions=0)
    assert record_kv({"metrics": [metric]})["no_forward_calls"]
    metric["token_layer_positions"] = 1
    with pytest.raises(ValueError, match="unaccounted"):
        record_kv({"metrics": [metric]})


def test_packaged_reference_reproduces_the_full_168_task_table():
    result = compare(reference_rows())
    base, ours = result["base"], result["stable_interfaces"]
    assert base["tasks"] == ours["tasks"] == 168
    assert (base["success"], ours["success"]) == (54, 56)
    assert base["decode_kv"] == 28569303272 and ours["decode_kv"] == 23297469080
    assert ours["kv_per_success"] == pytest.approx(416026233.5714286)
    assert ours["kv_per_success_reduction"] == pytest.approx(0.21365192567104396)
    assert "81.55%" in markdown(result) and "80.70%" in markdown(result)
