"""Small, atomic experiment records; user outputs are never overwritten implicitly."""

import json
import os
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def new_directory(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path
