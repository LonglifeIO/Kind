#!/usr/bin/env python3
"""Phase 6 mirror call against a completed run's telemetry.

The mirror is not called from inside the runner at Probe 1 (plan §2.10:
no in-loop prompting). This script is how it gets called: a single
Gemini read of the most recent N episodes of ``agent_step`` records,
written as a structured :class:`~kind.mirror.caller.MirrorReading` to
``runs/{run_id}/mirror/readings.jsonl``.

Usage:
    python scripts/call_mirror.py probe1-20260503-123926

Or against the latest ``probe1-*`` directory under ``runs/``:
    python scripts/call_mirror.py

Requires ``GEMINI_API_KEY`` in the environment. The default model is
``gemini-2.5-pro`` (see :class:`~kind.mirror.caller.MirrorCaller`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from kind.mirror.caller import MirrorCaller, MirrorReading

# python-dotenv is a declared dependency in pyproject.toml; load the
# project-root .env so GEMINI_API_KEY (and any other secrets) are
# available to MirrorCaller's constructor without requiring the operator
# to source the file into their shell first. Silent if no .env exists.
load_dotenv()

_RUN_ID_GLOB: str = "probe1-*"
_RUNS_DIR_NAME: str = "runs"
_N_EPISODES: int = 3


def _latest_run_id(runs_dir: Path) -> str:
    """Return the newest ``probe1-*`` directory name under ``runs_dir``.

    Sort is lexical; the run-id format ``probe1-YYYYMMDD-HHMMSS`` makes
    lexical and chronological order coincide, so a plain string sort
    suffices. Raises :class:`FileNotFoundError` if the runs directory is
    absent or contains no matching subdirectories — the caller renders
    that into a clean stderr exit, not a traceback.
    """
    if not runs_dir.is_dir():
        raise FileNotFoundError(
            f"runs directory not found at {runs_dir.resolve()} — "
            f"no run to read against."
        )
    candidates = sorted(p for p in runs_dir.glob(_RUN_ID_GLOB) if p.is_dir())
    if not candidates:
        raise FileNotFoundError(
            f"no probe1-* run directories under {runs_dir.resolve()}."
        )
    return candidates[-1].name


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Call the Probe 1 mirror against a finished run's telemetry."
        ),
    )
    parser.add_argument(
        "run_id",
        nargs="?",
        default=None,
        help=(
            "run id under runs/ (e.g. probe1-20260503-123926). "
            "Defaults to the most recent probe1-* directory."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Gemini model id (e.g. gemini-2.5-flash). Omit to use the "
            "MirrorCaller default (gemini-2.5-pro). The caller's "
            "docstring names flash as the documented fallback when pro "
            "returns 503/rate-limit errors."
        ),
    )
    return parser.parse_args(argv)


def _print_reading(reading: MirrorReading, target_path: Path) -> None:
    print()
    print("=" * 60)
    print("Mirror reading")
    print("=" * 60)
    print(f"  model:            {reading.model_used}")
    print(f"  n_episodes_read:  {reading.n_episodes_read}")
    print(f"  agent_step_range: {reading.agent_step_range}")
    print()
    print("---- summary ----")
    print(reading.summary)
    print()
    print("---- flagged_observations ----")
    if reading.flagged_observations:
        for note in reading.flagged_observations:
            print(f"  - {note}")
    else:
        print("  (none)")
    print()
    print(f"reading written to: {target_path}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    runs_dir = Path(_RUNS_DIR_NAME)

    try:
        run_id = (
            _latest_run_id(runs_dir)
            if args.run_id is None
            else str(args.run_id)
        )
    except FileNotFoundError as exc:
        print(f"[call_mirror] {exc}", file=sys.stderr)
        return 2

    telemetry_dir = runs_dir / run_id / "telemetry"
    mirror_dir = runs_dir / run_id / "mirror"

    print(f"[call_mirror] run_id={run_id}")
    print(f"[call_mirror] telemetry_dir={telemetry_dir}")

    try:
        caller = (
            MirrorCaller(model=str(args.model))
            if args.model is not None
            else MirrorCaller()
        )
    except ValueError as exc:
        print(f"[call_mirror] {exc}", file=sys.stderr)
        return 2

    print(
        f"[call_mirror] calling {caller.model} "
        f"(last {_N_EPISODES} episodes)..."
    )

    reading = caller.read_recent(
        telemetry_dir=telemetry_dir,
        n_episodes=_N_EPISODES,
        run_id=run_id,
    )
    caller.write_reading(reading, mirror_dir=mirror_dir)

    _print_reading(reading, mirror_dir / "readings.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
