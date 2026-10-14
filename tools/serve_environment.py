"""Serve one local AppWorld worker, bound to loopback only.

Run inside an isolated AppWorld runtime, not in a sensitive host environment.
The working directory matters: remote_docker resets the server's root to '.'.
"""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("use an unprivileged port in 1024..65535")
    root = Path(args.root).resolve()
    if not (root / "data/tasks").is_dir():
        parser.error("root must contain installed AppWorld data/tasks")
    os.chdir(root)
    from appworld import AppWorld, update_root
    from appworld.serve.environment import app
    import uvicorn

    update_root(str(root))
    try:
        uvicorn.run(app, host="127.0.0.1", port=args.port)
    finally:
        AppWorld.close_all()


if __name__ == "__main__":
    main()
