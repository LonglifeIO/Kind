"""Phase 9 judge smoke CLI — invokes :func:`run_phase_9_judge_smoke`
against the real Gemini API.

Usage::

    python scripts/run_phase_9_judge_smoke.py \\
        --phase-13-calibration-path \\
            runs/phase_13_calibration/mirror/phase_13_calibration_result.json \\
        --output-dir runs/phase_9_judge_smoke/

The script consumes :envvar:`GEMINI_API_KEY` from the environment
(override the env var name with ``--llm-api-key-env-var``). It prints
a one-line summary on completion: total judgments, total calls, total
retries, total failures, total wallclock.

**This consumes Gemini API quota.** Phase 9 makes 2 batched judge
LLM calls end-to-end (one per round, each batched across the three
criteria). At Phase 12/13's per-call latency (~30 s plus the judge
fragment is larger than the primary/adversarial fragments), expect
~5 minutes wallclock total. The journal entry records the audit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from kind.mirror.calibration.phase_9_judge_smoke import (
    Phase9JudgeSmokeResult,
    run_phase_9_judge_smoke,
)

# Load the project-root .env so GEMINI_API_KEY (and any other secrets)
# are available without requiring the operator to pre-set them in the
# shell.
load_dotenv()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 9 judge smoke harness — runs the judge driver "
            "against the two Phase 13 round results, producing one "
            "RoundJudgment per round."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase-13-calibration-path",
        required=True,
        type=Path,
        help="Path to the Phase 13 calibration result JSON file "
        "(typically runs/phase_13_calibration/mirror/"
        "phase_13_calibration_result.json). The two round JSON files "
        "are discovered from this path's sibling rounds/ directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Where to write the judge smoke output (mirror/judgments/, "
        "phase_9_judge_smoke_result.json). Created if missing.",
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
        help="Free-text notes attached to the judge smoke result. The "
        "journal entry quotes these.",
    )
    return parser


def _print_summary(result: Phase9JudgeSmokeResult) -> None:
    """Print the one-line summary on completion."""
    total_judgments = len(result.probe_1_judgment.criterion_judgments) + len(
        result.probe_1_5_judgment.criterion_judgments
    )
    audit = result.llm_call_audit
    summary = (
        f"phase_9_judge_smoke: judgments={total_judgments} "
        f"calls={audit.total_calls} "
        f"retries={audit.total_retries} "
        f"failures={audit.total_failures} "
        f"wallclock_ms={result.wallclock_ms} "
        f"tokens_in={audit.total_tokens_in} "
        f"tokens_out={audit.total_tokens_out}"
    )
    print(summary)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = run_phase_9_judge_smoke(
        phase_13_calibration_path=args.phase_13_calibration_path,
        output_dir=args.output_dir,
        llm_api_key_env_var=args.llm_api_key_env_var,
        notes=args.notes,
    )
    _print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
