#!/usr/bin/env python3
"""Launch the wifisim web planner.

Usage::

    python run_web.py [--host 127.0.0.1] [--port 5000] [--engine auto]

Open http://127.0.0.1:5000 in a browser.  The layout autosaves to the cache
directory and computed layers persist there across restarts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from web.app import create_app  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="wifisim 5 GHz coverage planner")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--engine", default="sionna_rt", choices=["sionna_rt"])
    p.add_argument("--cache", default=".wifisim_cache")
    args = p.parse_args()

    app = create_app(cache_dir=args.cache, engine=args.engine)
    print(f"wifisim planner on http://{args.host}:{args.port}  (engine={args.engine})")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
