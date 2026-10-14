"""Scope-limited native GDN adaptation; all Attention weights stay native."""

from contextlib import contextmanager
import math
import torch
from torch import nn
from torch.nn import functional as F


class RowLoRA(nn.Module):
    def __init__(self, base, rank, alpha, start_row=0):
        super().__init__()
        if not isinstance(base, nn.Linear) or not 0 <= start_row < base.out_features:
            raise ValueError("LoRA requires an unwrapped Linear and a valid output slice")
        self.base = base.requires_grad_(False)
        self.start_row = start_row
        self.scale = alpha / rank
        self.enabled = True
        self.elide_zero_output = False
        self.A = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32)
        )
        self.B = nn.Parameter(
            torch.zeros(
                base.out_features - start_row, rank, device=base.weight.device, dtype=torch.float32
            )
        )
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        output = self.base(x)
        if not self.enabled or self.elide_zero_output:
            return output
        with torch.autocast(device_type=x.device.type, enabled=False):
            update = F.linear(F.linear(x.float(), self.A), self.B) * self.scale
        update = F.pad(update, (self.start_row, 0))
        return output + update.to(output.dtype)


def install(backbone, config):
    backbone.requires_grad_(False)
    for index in config.adaptation_layers:
        gdn = backbone.layers[index].linear_attn
        # Native fused layout = [Q, K, V]. The Q rows have NO adapter parameters.
        gdn.in_proj_qkv = RowLoRA(gdn.in_proj_qkv, config.lora_rank, config.lora_alpha, gdn.key_dim)
        for name in ("in_proj_a", "in_proj_b"):
            setattr(gdn, name, RowLoRA(getattr(gdn, name), config.lora_rank, config.lora_alpha))


@contextmanager
def disabled(module):
    adapters = [m for m in module.modules() if isinstance(m, RowLoRA)]
    previous = [m.enabled for m in adapters]
    try:
        for adapter in adapters:
            adapter.enabled = False
        yield
    finally:
        for adapter, flag in zip(adapters, previous):
            adapter.enabled = flag
