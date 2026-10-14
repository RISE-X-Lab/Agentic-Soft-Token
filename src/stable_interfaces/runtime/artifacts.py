"""Portable safetensors calibration bundles; no pickle or optimizer in runtime."""

import hashlib
import json
from pathlib import Path
import torch
from .config import Config, FORMAT

ASSETS = Path(__file__).parent / "assets"


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def weight_digest(model):
    h = hashlib.sha256()
    for name, p in model.auxiliary_state().items():
        h.update(name.encode() + p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def read_bundle(bundle=None):
    from safetensors.torch import load_file

    root = ASSETS if bundle is None else Path(bundle)
    metadata = json.loads((root / "calibration.json").read_text())
    if metadata.get("format") != FORMAT or metadata.get("gradient_updates") != 0:
        raise ValueError("only a zero-update calibration bundle is supported")
    if file_sha(root / "calibration.safetensors") != metadata["tensor_sha256"]:
        raise ValueError("calibration tensor checksum mismatch")
    if metadata.get("protocol_sha256") != file_sha(ASSETS / "appworld_protocol.json"):
        raise ValueError("bundle was calibrated with a different Agent protocol")
    architecture = {**metadata["config"], "stages": tuple(metadata["config"]["stages"])}
    config = Config(**architecture)
    weights = load_file(str(root / "calibration.safetensors"))
    for name, value in weights.items():
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite calibration tensor")
        if name.endswith(".B") and torch.count_nonzero(value):
            raise ValueError("trained adapters are not a training-free calibration")
    return config, weights, metadata


def load_into(model, weights):
    current = model.auxiliary_state()
    if current.keys() != weights.keys():
        raise ValueError("calibration parameter whitelist mismatch")
    for n, p in current.items():
        if p.shape != weights[n].shape or p.dtype != weights[n].dtype:
            raise ValueError(f"calibration shape/dtype mismatch: {n}")
    with torch.no_grad():
        for n, p in current.items():
            p.copy_(weights[n])
    model.requires_grad_(False).eval()
    model.assert_scope()


def load_frozen_model(model_path, *, bundle=None, device="cuda:0"):
    from .model import load_model
    from .execution import install_fixed_kernels

    config, weights, metadata = read_bundle(bundle)
    model_path = Path(model_path).resolve()
    if file_sha(model_path / "config.json") != metadata["model_config_sha256"]:
        raise ValueError("base model config differs from the calibrated model")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "the released 9B runtime requires CUDA; use CPU oracle tests for development"
        )
    install_fixed_kernels()
    model = load_model(str(model_path), config=config, device=device)
    load_into(model, weights)
    return model
