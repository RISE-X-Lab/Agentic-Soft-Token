import torch
from stable_interfaces.runtime.config import Config
from stable_interfaces.runtime.model import MemoryInnovation
from stable_interfaces.runtime.protocol import Sample
from stable_interfaces.runtime.recurrence import scan


def tiny(monkeypatch, **options):
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    def kernel(q, k, v, g, beta, **kw):
        out, _, state, _ = scan(
            q,
            k,
            v,
            g,
            beta,
            kw.get("initial_state"),
            normalized=not kw.get("use_qk_l2norm_in_kernel", False),
        )
        return out, state if kw.get("output_final_state", False) else None

    monkeypatch.setattr(qwen, "FusedRMSNormGated", None)
    monkeypatch.setattr(qwen, "causal_conv1d_fn", None)
    monkeypatch.setattr(qwen, "causal_conv1d_update", qwen.torch_causal_conv1d_update)
    monkeypatch.setattr(qwen, "chunk_gated_delta_rule", kernel)
    monkeypatch.setattr(qwen, "fused_recurrent_gated_delta_rule", kernel)
    torch.manual_seed(14)
    cfg = Qwen3_5TextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        vocab_size=101,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [2, 2, 4],
        },
    )
    cfg._attn_implementation = "sdpa"
    native = qwen.Qwen3_5ForCausalLM(cfg).eval()
    recipe = dict(
        backend="oracle",
        slots=2,
        lora_rank=2,
        lora_alpha=2.0,
        reader_rank=2,
        reader_alpha=2.0,
        chi_threshold=100.0,
        readability_threshold=100.0,
    )
    recipe.update(options)
    return MemoryInnovation(native.model, native.lm_head, Config(**recipe)).eval()


def sample(n=64):
    torch.manual_seed(15)
    ids = torch.randint(1, 100, (n,))
    span = torch.full_like(ids, -1)
    span[4:22] = 0
    span[34:52] = 1
    labels = torch.full_like(ids, -100)
    labels[25:30], labels[55:60] = ids[25:30], ids[55:60]
    return Sample(ids, labels, span, span >= 0, span < 0).validate()
