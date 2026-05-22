"""Phase 12 smoke CLI — invokes :func:`run_phase_12_smoke` against the
real Gemini API.

Usage::

    python scripts/run_phase_12_smoke.py \\
        --probe-1-run-id <probe-1 run id> \\
        --probe-1-run-dir runs/probe1-20260503-123926 \\
        --probe-1-checkpoint <checkpoint id> \\
        --probe-1-5-run-id <probe-1.5 run id> \\
        --probe-1-5-run-dir runs/probe1.5-20260507 \\
        --probe-1-5-checkpoint <checkpoint id> \\
        --output-dir runs/phase_12_smoke/

The script consumes :envvar:`GEMINI_API_KEY` from the environment
(override the env var name with ``--llm-api-key-env-var``). It prints a
one-line summary on completion: total passes, total LLM calls, total
retries, total failures, total wallclock, total tokens, count of sham
admissions.

**This is the project's first command-line invocation that hits a real
LLM API.** Running this consumes Gemini API quota and may take several
minutes. Phase 12's journal entry records the run; subsequent rounds
read the resulting :class:`Phase12SmokeResult` for cross-round analysis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from kind.mirror.calibration.round import CheckpointSpec
from kind.mirror.calibration.smoke import (
    Phase12SmokeResult,
    run_phase_12_smoke,
)

# Load the project-root .env so GEMINI_API_KEY (and any other secrets)
# are available to the smoke harness without requiring the operator to
# pre-set them in the shell. Mirrors :mod:`scripts.call_mirror`'s
# pattern; the LLM caller reads from os.environ at call time.
load_dotenv()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 12 smoke harness — runs the full mirror execution "
            "plane against the real Gemini API, two rounds (one Probe "
            "1 checkpoint + one Probe 1.5 checkpoint), 5 passes each."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--probe-1-run-id",
        required=True,
        help="The run id the Probe 1 checkpoint belongs to (e.g. "
        "probe1-20260503-123926).",
    )
    parser.add_argument(
        "--probe-1-run-dir",
        required=True,
        type=Path,
        help="The on-disk run directory for the Probe 1 checkpoint. "
        "Typically runs/{probe-1-run-id}/.",
    )
    parser.add_argument(
        "--probe-1-checkpoint",
        required=True,
        help="The Probe 1 checkpoint id to run Round 1 against.",
    )
    parser.add_argument(
        "--probe-1-5-run-id",
        required=True,
        help="The run id the Probe 1.5 checkpoint belongs to.",
    )
    parser.add_argument(
        "--probe-1-5-run-dir",
        required=True,
        type=Path,
        help="The on-disk run directory for the Probe 1.5 checkpoint.",
    )
    parser.add_argument(
        "--probe-1-5-checkpoint",
        required=True,
        help="The Probe 1.5 checkpoint id to run Round 2 against.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Where to write the smoke output (mirror/, pre_reg/, "
        "rounds/). Created if missing.",
    )
    parser.add_argument(
        "--llm-api-key-env-var",
        default="GEMINI_API_KEY",
        help="Environment variable carrying the Gemini API key "
        "(default: GEMINI_API_KEY).",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Free-text notes attached to the smoke result. The "
        "journal entry quotes these.",
    )
    return parser


def _print_summary(result: Phase12SmokeResult) -> None:
    """Print the one-line summary the plan calls for on completion."""
    total_passes = len(result.probe_1_round.pass_results) + len(
        result.probe_1_5_round.pass_results
    )
    audit = result.llm_call_audit
    sham_total = (
        result.probe_1_round.sham_findings_summary.total
        + result.probe_1_5_round.sham_findings_summary.total
    )
    summary = (
        f"phase_12_smoke: passes={total_passes} "
        f"calls={audit.total_calls} "
        f"retries={audit.total_retries} "
        f"failures={audit.total_failures} "
        f"wallclock_ms={result.wallclock_ms} "
        f"tokens_in={audit.total_tokens_in} "
        f"tokens_out={audit.total_tokens_out} "
        f"sham_admissions={sham_total}"
    )
    print(summary)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    probe_1_spec = CheckpointSpec(
        run_id=args.probe_1_run_id,
        checkpoint_id=args.probe_1_checkpoint,
        run_dir=args.probe_1_run_dir,
    )
    probe_1_5_spec = CheckpointSpec(
        run_id=args.probe_1_5_run_id,
        checkpoint_id=args.probe_1_5_checkpoint,
        run_dir=args.probe_1_5_run_dir,
    )

    result = run_phase_12_smoke(
        probe_1_spec,
        probe_1_5_spec,
        output_dir=args.output_dir,
        llm_api_key_env_var=args.llm_api_key_env_var,
        notes=args.notes,
    )
    _print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
