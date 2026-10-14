# Main-table definitions

Every method must use the same tasks, model, prompt, generation limits, and
official evaluator. Include failures and time/step-limit exits in all totals.
Infrastructure errors are incomplete runs, not benchmark failures.

1. **Success**: official all-requirements success count / task count. This is not
   a count of individual passed assertions.
2. **Steps**: total executed Agent actions divided by task count. A final generation
   call that does not yield executable action still incurs model cost but adds no
   executed step.
3. **Decode KV / Base**: all tasks' cumulative decode KV position queries divided
   by Base's total. Sum over every decode forward and every Full Attention layer.
4. **KV / Success**: all tasks' cumulative decode KV queries divided by successes,
   including failed-task cost. Undefined when there are no successes.
5. **Attention Pairs / Base**: total effective causal prefill pairs plus decode KV
   queries, divided by Base's corresponding total.

Relative costs use Base=100%, so **smaller is better**. If displaying a positive
reduction instead, compute `100% - cost/Base` and use an **up** arrow.

For a layer with p cached positions after prefill and d subsequent decode
forwards, decode queries visit `d*p + d*(d+1)/2` KV positions. The model's own
key is included. In an action generating G tokens, d=G-1 because the first token
is predicted by the prefill. The last generated token is consumed in the next
step's prefill. This matches the Actor implementation; it avoids double counting.

If q positions are appended by prefill to c old cached positions, its valid
causal pairs are `q*c + q*(q+1)/2`. Counts are summed over the eight Full Attention
layers [3,7,11,15,19,23,27,31] (zero-based). GDN recurrence, MLP, signal collection,
selection, pooling, padding, and kernel launch overhead are not measured by these
counts. Consequently they cannot be labeled total-model FLOPs or speedups.

The runtime preserves original logical token positions. Do not substitute the
shortened physical length for every layer, or pretend the early layers shrink.

Paired 168-task reference totals:

| Method | Decode KV | Prefill pairs | Executed steps | Successes |
|---|---:|---:|---:|---:|
| Base | 28,569,303,272 | 128,897,088,616 | 4,217 | 54 |
| Stage-3 stable_interfaces | 23,297,469,080 | 103,770,978,816 | 4,203 | 56 |

The main table measures realized task trajectories, so savings include both
compression and changed behavior. A separate same-trajectory uncompressed
geometry reference isolates structural savings (7.16% decode KV for the Stage-3 run).
It is a counterfactual geometry calculation, not a second free-running rollout.

For ACON or another compressor, retain this Agent-side table and separately
account for compressor inference and any invalidated prefix recomputation.
Our zero extra LLM calls do not imply zero local compression overhead.
