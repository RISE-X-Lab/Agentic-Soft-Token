# Algorithm and frozen configuration

Only Stage 3 is enabled. One stage contains three GDN layers and one Full
Attention layer. Compression runs after the third GDN block (zero-based layer
10, including its residual and MLP), before Full Attention at layer 11.

The native FLA call supplies the output, final recurrent state, and intermediate
tensors. A side read computes the unweighted innovation
`delta = value - exp(raw_g) * prior_read(...)`. Native output and recurrent cache
are retained. Tool-end states are causal; prefix-state reads do add work.

For each eligible token, `chi = rms(delta)/(rms(value)+epsilon)`. Tokens with
chi below threshold are tested by `projected_error`: a ridge projection onto the
16 calibrated GDN read directions reconstructs the value. This normalized
reconstruction error is psi. The reader uses the state at the end of the Tool
observation, which is available before the subsequent Agent action.

The actual soft token is a mean of **full hidden states**, not a GDN recurrent
state or a memory-readout vector. Selected tokens are assigned to up to 16
ordered groups. A representative near each group's dominant direction is used
to reject outliers. The remaining members are averaged in FP32 and rounded to
the runtime dtype. Each group's actual rounded center must reconstruct every
accepted member within the relative L2 error limit. Groups with fewer than two
members stay explicit. Soft tokens use the greatest original logical position
of their members; the combined sequence is stably ordered by position.

The released AppWorld policy is fixed. Ordinary Tool data uses `chi < 0.60`,
`psi < 0.80`, and hidden merge error `<= 0.50`. Each Tool has at most 16 soft
tokens, at least two members per preliminary group, and must save at least 256
downstream token-layer positions and 5% of its eligible positions. API catalogs
are protected wholesale. The same published policy uses conservative bounds
(`chi < 0.55`, `psi < 0.74`, merge error `<= 0.55`) when a span has mixed or
unknown provenance; this is a safety branch, not a second experiment.

At Stage 3 the absolute minimum net reduction
is `ceil(256 / 21) = 13`, further bounded by 5% of eligible positions in that
Tool. This is not 5% of the whole context. Preflight avoids signal collection
when the eligible count cannot meet that minimum. Existing soft tokens are
protected from further merging. There is no reader at another Stage in this release.

Protection recognizes direct `apis.app.method(...)` calls in generated Python.
Detailed schemas are additionally recognized from returned JSON. API catalogs,
detailed docs, login returns and password-access returns remain explicit.
Literal protection matches numbers and identifiers containing digits or
underscores. This is a conservative heuristic, not an exhaustive detector of
important words, passwords or aliased API calls.

The first eleven layers process the original input positions. The remaining
layers process the merged sequence. Full Attention cache lengths are tracked
per layer while RoPE keeps original logical positions. GDN recurrent-state
capacity is fixed; it is not shortened as a token list. Subsequent decode
appends one token to each layer's own cache. Later Agent steps append only the
uncached suffix and reuse the already-compressed history.

The 0.58 values in `runtime/assets/calibration.json` describe the calibration
scaffold. Runtime routing uses the thresholds in
`assets/stable_interfaces.json` above. Both files are checksum/provenance checked.

Core implementation map:

- `algorithm/stable_contract_merge.py`: unchanged native GDN output and signal capture.
- `algorithm/observation_guard.py`, `operation_guard.py`: protected Tool content.
- `algorithm/typed_contract_merge.py`: provenance dispatch.
- `algorithm/selective_merge.py`: preflight, chi/psi selection and net saving check.
- `algorithm/pooling.py`: projected error and bounded hidden pooling.
- `runtime/model.py`: layer-specific caches and policy lifetime validation.

`algorithm/policy.py` defines the immutable Stage-3 policy. The public runtime
uses native signal capture, projected selection, and bounded hidden-state pooling.
