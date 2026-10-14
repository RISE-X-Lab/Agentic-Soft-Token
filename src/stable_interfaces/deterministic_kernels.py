"""Deterministic first-valid launch selection for audited FLA kernels."""

POLICY = "fla-first-valid-triton33-v1"
_original = None
_choices = {}


def install():
    """Replace FLA autotuning with a deterministic first-valid policy."""
    global _original
    import triton
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.errors import OutOfResources, PTXASError
    from triton.compiler.errors import CompileTimeAssertionFailure

    if triton.__version__ != "3.3.0":
        raise RuntimeError("deterministic launch policy requires Triton 3.3.0")
    if _original is not None:
        return
    _original = Autotuner.run

    def run(self, *args, **kwargs):
        if not self.base_fn.__module__.startswith("fla.") or len(self.configs) == 1:
            return _original(self, *args, **kwargs)
        self.nargs = dict(zip(self.arg_names, args))
        supplied = {**self.nargs, **kwargs}
        arguments = {key: value for key, value in supplied.items() if key in self.arg_names}
        key = tuple(
            [arguments[name] for name in self.keys if name in arguments]
            + [str(value.dtype) for value in arguments.values() if hasattr(value, "dtype")]
        )
        cache = getattr(self, "_stage_fixed_cache", None)
        if cache is None:
            cache = self._stage_fixed_cache = {}
        candidates = [cache[key]] if key in cache else self.prune_configs(kwargs)
        try:
            for config in candidates:
                meta = config.all_kwargs()
                if kwargs.keys() & meta.keys():
                    raise ValueError("conflicting fixed-launch meta parameters")
                full = {**self.nargs, **kwargs, **meta}
                if key not in cache:
                    self.pre_hook(full, reset_only=True)
                if config.pre_hook is not None:
                    config.pre_hook(full)
                try:
                    result = self.fn.run(*args, **kwargs, **meta)
                except (OutOfResources, PTXASError, CompileTimeAssertionFailure):
                    if key in cache:
                        raise
                    continue
                cache[key] = config
                self.best_config = config
                name = self.base_fn.__module__ + "." + self.base_fn.__name__
                _choices[name + " " + str(key)] = meta
                return result
            raise RuntimeError("no compilable deterministic FLA configuration")
        finally:
            self.nargs = None

    Autotuner.run = run


def choices():
    """Return the FLA launch choices observed in this process."""
    return dict(sorted(_choices.items()))
