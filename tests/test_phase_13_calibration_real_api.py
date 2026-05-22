"""Phase 13 end-to-end calibration against the REAL Gemini API.

This test is opt-in: pytest skips it by default, running only when
``pytest --run-real-api`` is passed or when :envvar:`GEMINI_API_KEY`
is set in the environment.

Running this test consumes Gemini API quota (4 LLM calls per pass × 5
passes × 2 rounds + 10 isolation calls × 2 checkpoints = ~60 calls
minimum, plus retries; ~30-50 minutes wallclock typical). The test
asserts:

- the calibration completes (no uncaught exceptions);
- the result is well-formed (Pydantic round-trip succeeds);
- the LLM audit shows ≥ 60 call records (the structural floor);
- the held-out isolation study produces ≥ 10 readings per checkpoint
  (Phase 13 commits 10);
- the synthetic findings summary's ``total_synthetic_events`` matches
  the expected count (5 passes × 2 synthetics × 2 criteria × 2 roles
  × 2 checkpoints = 80).

Substantive interpretation is journal-side. This test is the
engineering gate.

The test takes paths from environment variables (same convention as
the Phase 12 real-API test):

- :envvar:`KIND_PHASE_12_PROBE_1_RUN_DIR` — Probe 1 source run dir
- :envvar:`KIND_PHASE_12_PROBE_1_5_RUN_DIR` — Probe 1.5 source run dir
- :envvar:`KIND_PHASE_12_PROBE_1_CHECKPOINT_ID` — Probe 1 checkpoint
- :envvar:`KIND_PHASE_12_PROBE_1_5_CHECKPOINT_ID` — Probe 1.5 checkpoint
- :envvar:`KIND_PHASE_12_PROBE_1_RUN_ID` — Probe 1 run id
- :envvar:`KIND_PHASE_12_PROBE_1_5_RUN_ID` — Probe 1.5 run id
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kind.mirror.calibration.phase_13 import (
    PHASE_13_PASSES_PER_CHECKPOINT,
    PHASE_13_SYNTHETICS_PER_PASS,
    Phase13CalibrationResult,
    run_phase_13_calibration,
)
from kind.mirror.calibration.round import CheckpointSpec


_REQUIRED_ENV_VARS = (
    "KIND_PHASE_12_PROBE_1_RUN_DIR",
    "KIND_PHASE_12_PROBE_1_5_RUN_DIR",
    "KIND_PHASE_12_PROBE_1_CHECKPOINT_ID",
    "KIND_PHASE_12_PROBE_1_5_CHECKPOINT_ID",
    "KIND_PHASE_12_PROBE_1_RUN_ID",
    "KIND_PHASE_12_PROBE_1_5_RUN_ID",
)


def _checkpoint_specs() -> tuple[CheckpointSpec, CheckpointSpec]:
    missing = [v for v in _REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        pytest.skip(
            f"missing required env vars for real-API calibration: {missing}"
        )
    probe_1 = CheckpointSpec(
        run_id=os.environ["KIND_PHASE_12_PROBE_1_RUN_ID"],
        checkpoint_id=os.environ["KIND_PHASE_12_PROBE_1_CHECKPOINT_ID"],
        run_dir=Path(os.environ["KIND_PHASE_12_PROBE_1_RUN_DIR"]),
    )
    probe_1_5 = CheckpointSpec(
        run_id=os.environ["KIND_PHASE_12_PROBE_1_5_RUN_ID"],
        checkpoint_id=os.environ["KIND_PHASE_12_PROBE_1_5_CHECKPOINT_ID"],
        run_dir=Path(os.environ["KIND_PHASE_12_PROBE_1_5_RUN_DIR"]),
    )
    return probe_1, probe_1_5


@pytest.mark.real_api
def test_phase_13_calibration_runs_end_to_end_against_real_gemini(
    tmp_path: Path,
) -> None:
    """Run the full Phase 13 calibration against the real Gemini API.

    The test consumes API quota; conftest.py's collection-modifier
    skips this by default unless ``--run-real-api`` or
    :envvar:`GEMINI_API_KEY` is set.
    """
    probe_1, probe_1_5 = _checkpoint_specs()
    result = run_phase_13_calibration(
        probe_1,
        probe_1_5,
        output_dir=tmp_path / "phase_13_calibration",
        notes="real-API Phase 13 calibration invocation",
    )
    assert isinstance(result, Phase13CalibrationResult)

    # Well-formed: round-trip serializes.
    redumped = Phase13CalibrationResult.model_validate_json(
        result.model_dump_json()
    )
    assert redumped == result

    # Structural floor on the LLM audit: 4 calls per pass × 5 passes ×
    # 2 rounds = 40; plus 10 isolation calls × 2 checkpoints = 20;
    # total ≥ 60.
    audit = result.llm_call_audit
    assert audit.total_calls >= 60, (
        f"expected >= 60 LLM call records (5 passes × 4 calls × 2 "
        f"checkpoints + 10 isolation × 2 checkpoints); got "
        f"{audit.total_calls}"
    )

    # Synthetic findings: 5 passes × 2 synthetics × 2 active criteria
    # × 2 roles × 2 rounds = 80 findings total across the two rounds.
    expected_synthetic_events = (
        PHASE_13_PASSES_PER_CHECKPOINT * PHASE_13_SYNTHETICS_PER_PASS * 2 * 2
    )
    probe_1_synth = result.probe_1_round.synthetic_findings_summary
    probe_1_5_synth = result.probe_1_5_round.synthetic_findings_summary
    assert (
        probe_1_synth.total_synthetic_events
        + probe_1_5_synth.total_synthetic_events
    ) == expected_synthetic_events

    # Isolation study: 10 readings per checkpoint (Phase 13 commit).
    iso = result.held_out_isolation_study
    assert len(iso.probe_1_readings) == 10
    assert len(iso.probe_1_5_readings) == 10
    # Findings string is non-empty (the model validator enforces this,
    # but the assertion is the smoke-test surface for the
    # interpretive output).
    assert iso.findings.strip() != ""
