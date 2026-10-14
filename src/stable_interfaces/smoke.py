"""Small real-backbone engineering checks; this is not an accuracy benchmark."""

import json
import torch
from . import load_model, compression
from .runtime.artifacts import weight_digest
from .runtime.io import new_directory, write_json


def fixture(device="cpu"):
    generator = torch.Generator().manual_seed(20260913)
    ids = torch.randint(100, 3000, (256,), generator=generator).to(device)
    span = torch.full_like(ids, -1)
    span[8:96], span[144:232] = 0, 1
    return ids, dict(span=span, eligible=span >= 0, protected=span < 0)


@torch.no_grad()
def smoke(args):
    output = new_directory(args.output)
    report = dict(engineering_only=True, passed=False, stages=[3])
    try:
        torch.manual_seed(20260914)
        model = load_model(args.model, device=args.device)
        before = weight_digest(model)
        ids, kwargs = fixture(model.device)
        native = model.native_hidden(ids)
        noop = model(ids, **kwargs, compression=False)
        error = float((native.float() - noop.sequence.hidden.float()).abs().max())
        if error > 1e-5:
            raise RuntimeError(f"native no-compression forward differs: {error}")
        del native, noop
        snapshots, cache = [], None
        with compression(model):
            for start, end in ((0, 100), (100, 101), (101, 236), (236, 237), (237, 256)):
                _, result = model.next_logits(
                    ids[start:end],
                    **{name: value[start:end] for name, value in kwargs.items()},
                    past_key_values=cache,
                )
                cache = result.past_key_values
                if any(event["stage"] != 3 for event in cache.compression_metrics()):
                    raise RuntimeError("compression occurred outside Stage 3")
                if any(n != end for n in cache.layer_lengths[:11]):
                    raise RuntimeError("layers before the compression boundary shrank")
                snapshots.append(
                    dict(
                        logical_length=cache.logical_length,
                        layer_lengths=list(cache.layer_lengths),
                        compression=cache.compression_metrics(),
                    )
                )
        for previous, current in ((snapshots[0], snapshots[1]), (snapshots[2], snapshots[3])):
            if current["compression"] != previous["compression"] or any(
                b != a + 1 for a, b in zip(previous["layer_lengths"], current["layer_lengths"])
            ):
                raise RuntimeError(
                    "single-token decode did not preserve and append to the compressed cache"
                )
        if weight_digest(model) != before or any(p.requires_grad for p in model.parameters()):
            raise RuntimeError("model was not frozen")
        report.update(
            passed=True,
            native_noop_max_abs_error=error,
            snapshots=snapshots,
            weights_unchanged=True,
            trainable_parameters=0,
        )
    finally:
        write_json(output / "smoke.json", report)
    print(json.dumps(report, indent=2))
