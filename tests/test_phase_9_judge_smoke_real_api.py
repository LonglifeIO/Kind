"""Phase 9 end-to-end judge smoke against the REAL Gemini API.

This test is opt-in: pytest skips it by default, running only when
``pytest --run-real-api`` is passed or when :envvar:`GEMINI_API_KEY`
is set in the environment.

Running this test consumes Gemini API quota (2 batched judge LLM
calls — one per round, each batched across 3 criteria; ~5 minutes
wallclock typical). The test asserts:

- the smoke completes (no uncaught exceptions);
- the result is well-formed (Pydantic round-trip succeeds);
- every criterion in both rounds has a non-empty rationale;
- the LLM audit shows at least one success per round.

Substantive interpretation is journal-side. This test is the
engineering gate.

The test takes the Phase 13 calibration result path from an
environment variable:

- :envvar:`KIND_PHASE_13_CALIBRATION_PATH` — path to the Phase 13
  calibration result JSON (e.g. ``runs/phase_13_calibration/mirror/
  phase_13_calibration_result.json``)

If the env var is unset the test skips with a clear message.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kind.mirror.calibration.phase_9_judge_smoke import (
    Phase9JudgeSmokeResult,
    run_phase_9_judge_smoke,
)


_REQUIRED_ENV_VARS = ("KIND_PHASE_13_CALIBRATION_PATH",)


def _calibration_path() -> Path:
    """Read the Phase 13 calibration result path from the env;
    skip if unset."""
    missing = [v for v in _REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        pytest.skip(
            f"missing required env vars for real-API judge smoke: {missing}"
        )
    return Path(os.environ["KIND_PHASE_13_CALIBRATION_PATH"])


@pytest.mark.real_api
def test_phase_9_judge_smoke_runs_end_to_end_against_real_gemini(
    tmp_path: Path,
) -> None:
    """Run the full Phase 9 judge smoke against the real Gemini API.

    The test consumes API quota; conftest.py's collection-modifier
    skips this by default unless ``--run-real-api`` or
    :envvar:`GEMINI_API_KEY` is set.
    """
    calibration_path = _calibration_path()
    if not calibration_path.is_file():
        pytest.skip(
            f"Phase 13 calibration result not found at {calibration_path}; "
            f"the Phase 9 smoke needs the Phase 13 rounds on disk first"
        )
    result = run_phase_9_judge_smoke(
        phase_13_calibration_path=calibration_path,
        output_dir=tmp_path / "phase_9_judge_smoke",
        notes="real-API judge smoke test invocation",
    )
    assert isinstance(result, Phase9JudgeSmokeResult)

    # Well-formed: round-trip serializes.
    redumped = Phase9JudgeSmokeResult.model_validate_json(
        result.model_dump_json()
    )
    assert redumped == result

    # Every criterion in both rounds has a non-empty rationale.
    for judgment in (result.probe_1_judgment, result.probe_1_5_judgment):
        for cj in judgment.criterion_judgments:
            assert cj.rationale.strip(), (
                f"criterion {cj.criterion_id!r} in round "
                f"{judgment.round_id!r} has an empty rationale; the "
                f"judge's load-bearing audit trail is missing"
            )

    # At least one success record per round in the audit.
    audit = result.llm_call_audit
    rounds_with_success = {
        r.round_id for r in audit.records if r.outcome == "success"
    }
    assert result.probe_1_judgment.round_id in rounds_with_success
    assert result.probe_1_5_judgment.round_id in rounds_with_success
