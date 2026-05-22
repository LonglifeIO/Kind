"""Phase 12 end-to-end smoke against the REAL Gemini API.

**This is the first test in the project that makes real LLM API calls.**
It is opt-in: pytest skips it by default, running only when
``pytest --run-real-api`` is passed or when :envvar:`GEMINI_API_KEY` is
set in the environment.

Running this test consumes Gemini API quota (4 LLM calls per pass × 5
passes × 2 rounds = 40 calls, plus any retries; ~1-2 minutes wallclock
typical). The test asserts:

- the smoke completes (no uncaught exceptions);
- the result is well-formed (Pydantic round-trip succeeds);
- the LLM audit shows at least one success per role per checkpoint
  (4 successes per checkpoint × 2 checkpoints = 8 success records
  minimum; in practice the audit will have many more if retries fire).

Anything more substantive than these structural checks is journal-side
analysis. This test is the engineering gate: it verifies the Phase 12
wiring is sound end-to-end. The journal entry interprets the readings.

The test takes paths from environment variables to avoid hardcoding
the user's run-directory layout:

- :envvar:`KIND_PHASE_12_PROBE_1_RUN_DIR` — Probe 1 source run dir
- :envvar:`KIND_PHASE_12_PROBE_1_5_RUN_DIR` — Probe 1.5 source run dir
- :envvar:`KIND_PHASE_12_PROBE_1_CHECKPOINT_ID` — Probe 1 checkpoint
- :envvar:`KIND_PHASE_12_PROBE_1_5_CHECKPOINT_ID` — Probe 1.5 checkpoint
- :envvar:`KIND_PHASE_12_PROBE_1_RUN_ID` — Probe 1 run id (for record
  envelope)
- :envvar:`KIND_PHASE_12_PROBE_1_5_RUN_ID` — Probe 1.5 run id

If any of these are unset the test skips with a clear message naming
the missing variable.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kind.mirror.calibration.round import CheckpointSpec
from kind.mirror.calibration.smoke import (
    Phase12SmokeResult,
    run_phase_12_smoke,
)


_REQUIRED_ENV_VARS = (
    "KIND_PHASE_12_PROBE_1_RUN_DIR",
    "KIND_PHASE_12_PROBE_1_5_RUN_DIR",
    "KIND_PHASE_12_PROBE_1_CHECKPOINT_ID",
    "KIND_PHASE_12_PROBE_1_5_CHECKPOINT_ID",
    "KIND_PHASE_12_PROBE_1_RUN_ID",
    "KIND_PHASE_12_PROBE_1_5_RUN_ID",
)


def _checkpoint_specs() -> tuple[CheckpointSpec, CheckpointSpec]:
    """Read source-checkpoint specs from env vars; raise SkipTest if
    any are missing."""
    missing = [v for v in _REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        pytest.skip(
            f"missing required env vars for real-API smoke: {missing}"
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
def test_phase_12_smoke_runs_end_to_end_against_real_gemini(
    tmp_path: Path,
) -> None:
    """Run the full Phase 12 smoke against the real Gemini API.

    The test consumes API quota; conftest.py's collection-modifier
    skips this by default unless ``--run-real-api`` or
    :envvar:`GEMINI_API_KEY` is set.
    """
    probe_1, probe_1_5 = _checkpoint_specs()
    result = run_phase_12_smoke(
        probe_1,
        probe_1_5,
        output_dir=tmp_path / "phase_12_smoke",
        notes="real-API smoke test invocation",
    )
    assert isinstance(result, Phase12SmokeResult)

    # Well-formed: round-trip serializes.
    redumped = Phase12SmokeResult.model_validate_json(result.model_dump_json())
    assert redumped == result

    # At least one successful call per role per checkpoint. The Phase
    # 12 commitment is 5 passes per checkpoint × 4 LLM calls per pass
    # = 20 calls per checkpoint = 40 total. A "successful" call is
    # outcome == "success". With any non-zero retry budget, every pass
    # should produce at least one success record.
    audit = result.llm_call_audit
    assert audit.total_calls >= 40, (
        f"expected >= 40 LLM call records (5 passes × 4 calls × 2 "
        f"checkpoints); got {audit.total_calls}"
    )
    # The Phase 12 plan says calibration may surface
    # structured-output / token-budget issues; total_failures > 0 is a
    # finding to journal, not a test failure. The assertion here is the
    # weaker "the wiring at least delivered some success records".
    successes_by_checkpoint = {
        probe_1.checkpoint_id: 0,
        probe_1_5.checkpoint_id: 0,
    }
    for r in audit.records:
        if r.outcome == "success":
            successes_by_checkpoint[r.checkpoint_id] = (
                successes_by_checkpoint.get(r.checkpoint_id, 0) + 1
            )
    assert successes_by_checkpoint[probe_1.checkpoint_id] >= 4, (
        f"expected >= 4 success records for Probe 1 checkpoint; got "
        f"{successes_by_checkpoint[probe_1.checkpoint_id]}"
    )
    assert successes_by_checkpoint[probe_1_5.checkpoint_id] >= 4, (
        f"expected >= 4 success records for Probe 1.5 checkpoint; got "
        f"{successes_by_checkpoint[probe_1_5.checkpoint_id]}"
    )
