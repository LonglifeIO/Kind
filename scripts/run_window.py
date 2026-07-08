"""Phase 11.5 CLI — serve Window, the read-only record viewer.

Usage::

    python scripts/run_window.py --run-id phase_13_calibration

Window serves a small HTML interface on ``<host>:<port>``; the host's
Tailscale setup handles remote access. Once it is running, point a
browser at ``http://<mac-mini-tailscale-name>:8765/`` from any device
on the Tailnet.

Window's GET routes are read-only — they open the run's on-disk
records for reading and write nowhere. The one exception is the
builder's hello button (``POST /hello``), which writes a manual
perturbation request into the run's ``perturbation_inbox/`` for the
live runner to drain (the plan's DP2 GUI-button convenience). It makes
no LLM calls and does not touch Io's process. Pointing it at a run
that is mid-biography is safe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kind.window import create_app

_DEFAULT_PORT = 8765
_DEFAULT_HOST = "0.0.0.0"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Window — a read-only viewer over a Kind run's on-disk "
            "mirror and telemetry records."
        )
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="the run to view; its records live under runs/<run-id>/.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_DEFAULT_PORT,
        help=f"local port to serve on (default {_DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--host",
        default=_DEFAULT_HOST,
        help=(
            f"bind address (default {_DEFAULT_HOST!r}); the default lets "
            f"Tailscale reach the server, 127.0.0.1 restricts to localhost."
        ),
    )
    args = parser.parse_args(argv)

    run_id: str = args.run_id
    run_dir = Path("runs") / run_id
    if not run_dir.is_dir():
        print(
            f"run_window: run directory not found: {run_dir}",
            file=sys.stderr,
        )
        return 1

    app = create_app(run_id, run_dir)
    print(
        f"run_window: serving run {run_id!r} on "
        f"http://{args.host}:{args.port}/",
        file=sys.stderr,
    )
    # threaded: the live view polls every 500 ms while other pages load;
    # a single-threaded server would serialize them.
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
