"""Real FLA checks for the native-output/side-signal separation."""

from types import SimpleNamespace
import pytest
import torch

from stable_interfaces.algorithm.stable_contract_merge import native_signal_scope

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FLA"),
]


@torch.no_grad()
def test_capture_keeps_native_output_and_recurrent_cache_exact():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    from stable_interfaces.runtime.config import Config
    import stable_interfaces.runtime.model as implementation

    torch.manual_seed(76)
    cfg = Qwen3_5TextConfig(
        hidden_size=128,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
    )
    gdn = Qwen3_5GatedDeltaNet(cfg, layer_idx=0).eval().to(device="cuda", dtype=torch.bfloat16)
    for p in gdn.parameters():
        p.requires_grad_(False)
    old = implementation.gdn_forward
    caches = [SimpleNamespace(conv_states=[None], recurrent_states=[None]) for _ in range(2)]
    for n, end in ((137, 120), (85, 78)):
        x = torch.randn((1, n, 128), device="cuda", dtype=torch.bfloat16)
        native, _ = old(gdn, x, Config(), (), caches[0], False)
        with native_signal_scope():
            output, signals = implementation.gdn_forward(
                gdn, x, Config(), [(0, 3, end)], caches[1], True
            )
        assert implementation.gdn_forward is old
        assert torch.equal(native, output)
        assert torch.equal(caches[0].conv_states[0], caches[1].conv_states[0])
        assert torch.equal(caches[0].recurrent_states[0], caches[1].recurrent_states[0])
        assert len(signals.states) == 1 and signals.delta.shape[0] == n


@torch.no_grad()
def test_reused_wy_prefix_state_matches_native_prefix_scan():
    from stable_interfaces.runtime.dynamics import api

    f = api()
    torch.manual_seed(92)
    q, k, v = [torch.randn((1, 137, 4, 64), device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    g = -torch.rand((1, 137, 4), device="cuda") * 0.3
    beta = torch.rand((1, 137, 4), device="cuda", dtype=torch.bfloat16)
    h0 = torch.randn((1, 4, 64, 64), device="cuda") * 0.01
    original = f.chunk_gated_delta_rule_fwd_h
    captured = {}

    def capture(*args, **kw):
        captured.update(kw)
        return original(*args, **kw)

    f.chunk_gated_delta_rule_fwd_h = capture
    try:
        f.chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=h0,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    finally:
        f.chunk_gated_delta_rule_fwd_h = original
    for end in (37, 64, 79, 128):
        _, reference = f.chunk_gated_delta_rule(
            q[:, :end],
            k[:, :end],
            v[:, :end],
            g[:, :end],
            beta[:, :end],
            initial_state=h0,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        _, _, state = original(
            k=captured["k"][:, :end].contiguous(),
            w=captured["w"][:, :end].contiguous(),
            u=captured["u"][:, :end].contiguous(),
            g=captured["g"][:, :end].contiguous(),
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=None,
        )
        torch.testing.assert_close(state, reference, rtol=1e-5, atol=1e-5)
