"""The Stage-3 stable_interfaces method. Importing this module does not load CUDA."""

__version__ = "0.2.0"


def load_model(model_path, *, bundle=None, device="cuda:0"):
    """Load the frozen backbone and packaged Stage-3 calibration."""
    from .runtime.artifacts import load_frozen_model
    from .runtime.runtime import RuntimeOptions

    model = load_frozen_model(model_path, bundle=bundle, device=device)
    if model.config.stages != (3,) or model.config.slots != 16:
        raise ValueError("stable_interfaces requires Stage 3 and 16 maximum slots")
    model.configure_runtime(RuntimeOptions(True, True, True, True))
    return model


def compression(model, *, config=None, audit=None):
    """Keep this scope active for an entire task, including all cached decoding."""
    from .configuration import load_plan
    from .algorithm.stable_contract_merge import stable_scope

    return stable_scope(model, load_plan(config), audit=audit)
