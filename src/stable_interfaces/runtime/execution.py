"""Explicit, process-local FLA launch policy; no package files or math changes.

FLA 0.4.1 uses NB=ceil(flattened_rows/2048) as an autotune key AND
constexpr in its small-feature normalization kernels, although the bodies
never read NB. Dynamic AST lengths therefore compile/tune many copies of the
same computation. Launch the ORIGINAL kernel with a fixed tile and NB=1.
Real T, all masks, input/output dtypes and backward formulas are unchanged.

Only four normalization entrypoints are affected. This is an execution option,
not an architecture parameter; keep the installation alive through backward.
"""

import ast
from contextlib import contextmanager
import importlib


RECIPE = "fla041-normalization-bt32-w4-nb1-v1"
_installed = None


class FixedLaunch:
    def __init__(self, original):
        self.original = original
        self.heuristics = []
        kernel = original
        while hasattr(kernel, "fn"):
            # A JITFunction's .fn is a Python function: stop BEFORE unwrapping it.
            if hasattr(kernel, "src"):
                break
            if hasattr(kernel, "values"):
                self.heuristics.append(kernel.values)
            kernel = kernel.fn
        if not hasattr(kernel, "src"):
            raise RuntimeError("unexpected FLA/Triton decorator layout")
        tree = ast.parse(kernel.src)
        if "NB" not in kernel.arg_names or any(
            isinstance(n, ast.Name) and n.id == "NB" for n in ast.walk(tree)
        ):
            raise RuntimeError(
                "NB must be an unused launch-calibration argument, never a tensor dimension"
            )
        self.kernel = kernel
        self.launches = 0

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            if args:
                raise ValueError("validated FLA normalization calls must use keyword arguments")
            values = {**kwargs, "NB": 1, "BT": 32, "num_warps": 4}
            for rules in self.heuristics:
                for name, rule in rules.items():
                    values[name] = rule(values)
            self.launches += 1
            return self.kernel[grid](**values)

        return launch


def install_fixed_kernels():
    """Opt-in process-local configuration, idempotent. Does not alter any files."""
    global _installed
    if _installed is not None:
        return
    import fla

    if fla.__version__ != "0.4.1":
        raise RuntimeError("fixed normalization launches require the validated FLA 0.4.1")
    plan = []
    for module_name, names in (
        ("fla.modules.l2norm", ("l2norm_fwd_kernel", "l2norm_bwd_kernel")),
        (
            "fla.modules.fused_norm_gate",
            ("layer_norm_gated_fwd_kernel", "layer_norm_gated_bwd_kernel"),
        ),
    ):
        module = importlib.import_module(module_name)
        for name in names:
            original = getattr(module, name)
            plan.append((module, name, original, FixedLaunch(original)))
    # Validate every entry before changing any runtime binding.
    for module, name, _, replacement in plan:
        setattr(module, name, replacement)
    _installed = plan


def restore_default_kernels():
    global _installed
    if _installed is not None:
        for module, name, original, replacement in _installed:
            if getattr(module, name) is not replacement:
                raise RuntimeError("another component changed this process's FLA launch policy")
            setattr(module, name, original)
        _installed = None


@contextmanager
def fixed_kernels():
    was_installed = _installed is not None
    install_fixed_kernels()
    try:
        yield
    finally:
        if not was_installed:
            restore_default_kernels()


def execution_info():
    return {
        "normalization": RECIPE if _installed is not None else "upstream_autotune",
        "package_files_modified": False,
        "launches": {}
        if _installed is None
        else {name: replacement.launches for _, name, _, replacement in _installed},
    }
