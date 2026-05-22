"""Phase 13 calibration CLI — invokes :func:`run_phase_13_calibration`
against the real Gemini API.

Usage::

    python scripts/run_phase_13_calibration.py \\
        --probe-1-run-id <probe-1 run id> \\
        --probe-1-run-dir runs/probe1-20260503-123926 \\
        --probe-1-checkpoint <checkpoint id> \\
        --probe-1-5-run-id <probe-1.5 run id> \\
        --probe-1-5-run-dir runs/probe1.5-20260507 \\
        --probe-1-5-checkpoint <checkpoint id> \\
        --output-dir runs/phase_13_calibration/

The script consumes :envvar:`GEMINI_API_KEY` from the environment
(override the env var name with ``--llm-api-key-env-var``). It prints a
one-line summary on completion: total passes, total LLM calls, total
retries, total failures, total wallclock, total tokens, sham
admissions, synthetic admissions, and the isolation-study finding flag
(reading-a or reading-b).

**This consumes Gemini API quota.** Phase 13 makes ~100 LLM calls
end-to-end (40 inside the two rounds + 20 inside the isolation study,
plus retries). Wallclock ~30-50 minutes at Phase 12's per-call
latency. Plan the run; the journal entry records the audit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from kind.mirror.calibration.phase_13 import (
    Phase13CalibrationResult,
    run_phase_13_calibration,
)
from kind.mirror.calibration.round import CheckpointSpec

# Load the project-root .env so GEMINI_API_KEY (and any other secrets)
# are available without requiring the operator to pre-set them in the
# shell.
load_dotenv()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 13 calibration harness — runs the synthetic-real-"
            "perturbation injection plus the held-out isolation study "
            "against the real Gemini API."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--probe-1-run-id",
        required=True,
        help="The run id the Probe 1 checkpoint belongs to "
        "(e.g. probe1-20260503-123926).",
    )
    parser.add_argument(
        "--probe-1-run-dir",
        required=True,
        type=Path,
        help="The on-disk run directory for the Probe 1 checkpoint.",
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
        help="Where to write the calibration output (mirror/, pre_reg/, "
        "rounds/, phase_13_calibration_result.json). Created if missing.",
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
        help="Free-text notes attached to the calibration result. The "
        "journal entry quotes these.",
    )
    return parser


def _print_summary(result: Phase13CalibrationResult) -> None:
    """Print the one-line summary on completion."""
    total_passes = len(result.probe_1_round.pass_results) + len(
        result.probe_1_5_round.pass_results
    )
    audit = result.llm_call_audit
    sham_total = (
        result.probe_1_round.sham_findings_summary.total
        + result.probe_1_5_round.sham_findings_summary.total
    )
    synth_total = (
        result.probe_1_round.synthetic_findings_summary.total_admissions
        + result.probe_1_5_round.synthetic_findings_summary.total_admissions
    )
    iso_flag = (
        "reading_a"
        if "Reading (a) supported" in result.held_out_isolation_study.findings
        else "reading_b"
    )
    summary = (
        f"phase_13_calibration: passes={total_passes} "
        f"calls={audit.total_calls} "
        f"retries={audit.total_retries} "
        f"failures={audit.total_failures} "
        f"wallclock_ms={result.wallclock_ms} "
        f"tokens_in={audit.total_tokens_in} "
        f"tokens_out={audit.total_tokens_out} "
        f"sham_admissions={sham_total} "
        f"synthetic_admissions={synth_total} "
        f"isolation_finding={iso_flag}"
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

    result = run_phase_13_calibration(
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
