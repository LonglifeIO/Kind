"""Phase 9 judge smoke harness — :func:`run_phase_9_judge_smoke` and
:class:`Phase9JudgeSmokeResult`.

The Phase 9 smoke runs the judge driver against the two
:class:`~kind.mirror.calibration.round.RoundResult` artifacts the
Phase 13 calibration already produced. The two rounds (Probe 1, Probe
1.5) carry the multi-pass primary + adversarial readings the judge
consumes; the judge produces one
:class:`~kind.mirror.judge.RoundJudgment` per round.

**Phase 9 commitments (journaled at module-level constants).**

- Source: the two
  :class:`~kind.mirror.calibration.round.RoundResult` files at
  ``{phase_13_calibration_path}/../rounds/phase_13_probe_1_round.json``
  and ``.../phase_13_probe_1_5_round.json``. The harness accepts the
  parent
  :class:`~kind.mirror.calibration.phase_13.Phase13CalibrationResult`
  JSON path; the two round files are discovered from its
  sibling ``rounds/`` subdirectory. (The
  :class:`Phase13CalibrationResult` carries the rounds inline, so an
  alternative load path could pull them directly; the on-disk-rounds
  path is the one Phase 9 commits because it matches what
  :mod:`kind.mirror.judge_driver` expects.)
- Same :class:`~kind.mirror.llm_caller.LLMConfig` as Phases 12/13
  (``gemini-2.5-pro``).
- The judge runs once per round; each round produces three criterion
  judgments (2 active + 1 held-out). Total LLM calls for the Phase 9
  smoke: 2 (one batched judge call per round).

The judge batches all three criteria into one call per round (see
:func:`~kind.mirror.judge_llm_caller.call_judge_llm`'s batching
discipline); the plan's "6 calls" prose is reconciled here as "6
criterion judgments produced across 2 batched calls". The audit
records reflect the batched call count.

**The mirror is one-way.** The smoke writes only to
``output_dir/mirror/judgments/``. The source round results on disk
are untouched.

Out of scope: aggregation across Phase 9's two judgments (Phase 10+);
real-API tests (those live at
``tests/test_phase_9_judge_smoke_real_api.py``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict

from kind.mirror.calibration.llm_audit import LLMCallAudit
from kind.mirror.calibration.phase_13 import (
    PHASE_13_PROBE_1_5_ROUND_ID,
    PHASE_13_PROBE_1_ROUND_ID,
)
from kind.mirror.judge import RoundJudgment
from kind.mirror.judge_driver import judge_round
from kind.mirror.judge_llm_caller import JudgeLLMClient
from kind.mirror.llm_caller import LLMConfig

__all__ = [
    "PHASE_9_PROBE_1_ROUND_FILENAME",
    "PHASE_9_PROBE_1_5_ROUND_FILENAME",
    "PHASE_9_JUDGE_MAX_OUTPUT_TOKENS",
    "Phase9JudgeSmokeResult",
    "run_phase_9_judge_smoke",
]


# ---------------------------------------------------------------------------
# Phase 9 commitments.
# ---------------------------------------------------------------------------

PHASE_9_PROBE_1_ROUND_FILENAME: Final[str] = (
    f"{PHASE_13_PROBE_1_ROUND_ID}.json"
)
PHASE_9_PROBE_1_5_ROUND_FILENAME: Final[str] = (
    f"{PHASE_13_PROBE_1_5_ROUND_ID}.json"
)
# Phase 9 finding (journaled in the first-smoke entry): the batched
# judge call carries ~60 ClaimPolarityAssignmentPayload records (3
# criteria × ~5 passes × ~2 roles × ~2 claims) plus three rationales
# plus three FalsifierVerdictPayload records — structurally larger
# than the Phase 8 primary/adversarial response. The 8192-token
# default truncated every attempt across the retry budget on the
# first smoke run; 32768 carries comfortably and stays well under
# gemini-2.5-pro's 65536 output ceiling.
PHASE_9_JUDGE_MAX_OUTPUT_TOKENS: Final[int] = 32768


# ---------------------------------------------------------------------------
# Result.
# ---------------------------------------------------------------------------


class Phase9JudgeSmokeResult(BaseModel):
    """The full Phase 9 judge smoke output.

    Frozen, ``extra="forbid"``. Carries both rounds' judgments, the
    cross-round LLM-call audit, the wallclock, and free-text notes.

    On-disk form:
    ``output_dir/mirror/phase_9_judge_smoke_result.json``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "0.1.0"
    probe_1_judgment: RoundJudgment
    probe_1_5_judgment: RoundJudgment
    llm_call_audit: LLMCallAudit
    wallclock_ms: int
    notes: str


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _resolve_round_path(
    phase_13_calibration_path: Path, filename: str
) -> Path:
    """Resolve the on-disk path to a Phase 13 round JSON file.

    The input is the Phase 13 calibration result JSON path (e.g.
    ``runs/phase_13_calibration/mirror/phase_13_calibration_result.json``);
    the round files live as siblings at
    ``runs/phase_13_calibration/mirror/rounds/{round_id}.json``.
    """
    rounds_dir = phase_13_calibration_path.parent / "rounds"
    return rounds_dir / filename


def _write_smoke_result(
    result: Phase9JudgeSmokeResult, mirror_dir: Path
) -> None:
    """Write the :class:`Phase9JudgeSmokeResult` to disk atomically.

    Same write-temp-then-rename pattern as
    :class:`~kind.mirror.calibration.phase_13.Phase13CalibrationResult`.
    """
    mirror_dir.mkdir(parents=True, exist_ok=True)
    final_path = mirror_dir / "phase_9_judge_smoke_result.json"
    tmp_path = mirror_dir / ".phase_9_judge_smoke_result.json.tmp"
    payload = result.model_dump(mode="json")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    tmp_path.replace(final_path)


# ---------------------------------------------------------------------------
# Public entrypoint.
# ---------------------------------------------------------------------------


def run_phase_9_judge_smoke(
    *,
    phase_13_calibration_path: Path,
    output_dir: Path,
    llm_api_key_env_var: str = "GEMINI_API_KEY",
    llm_client: JudgeLLMClient | None = None,
    notes: str = "",
) -> Phase9JudgeSmokeResult:
    """Run the Phase 9 judge smoke against the real Gemini API.

    Sequence:

    1. Resolve the two Phase 13 round JSON paths from
       ``phase_13_calibration_path``.
    2. Build an :class:`LLMConfig` with the same defaults as Phase
       13's (``gemini-2.5-pro``, 8192 max output tokens, 3 max
       retries).
    3. Call :func:`~kind.mirror.judge_driver.judge_round` on the
       Probe 1 round → produces Round 1 judgment.
    4. Call :func:`~kind.mirror.judge_driver.judge_round` on the
       Probe 1.5 round → produces Round 2 judgment.
    5. Aggregate the cross-round LLM-call audit from the two
       judgments' ``judge_llm_call_records``.
    6. Pack and write the :class:`Phase9JudgeSmokeResult` atomically.

    ``llm_client``: when ``None``, the judge LLM caller constructs a
    default Gemini client via the
    :attr:`~kind.mirror.llm_caller.LLMConfig.api_key_env_var`. Tests
    inject a :class:`~kind.mirror.judge_llm_caller.MockJudgeLLMClient`.
    """
    t0 = int(time.time() * 1000)

    probe_1_path = _resolve_round_path(
        phase_13_calibration_path, PHASE_9_PROBE_1_ROUND_FILENAME
    )
    probe_1_5_path = _resolve_round_path(
        phase_13_calibration_path, PHASE_9_PROBE_1_5_ROUND_FILENAME
    )
    if not probe_1_path.is_file():
        raise FileNotFoundError(
            f"run_phase_9_judge_smoke: Probe 1 round file not found at "
            f"{probe_1_path}. The Phase 13 calibration must have run "
            f"first and produced its round artifacts."
        )
    if not probe_1_5_path.is_file():
        raise FileNotFoundError(
            f"run_phase_9_judge_smoke: Probe 1.5 round file not found "
            f"at {probe_1_5_path}. The Phase 13 calibration must have "
            f"run first."
        )

    llm_config = LLMConfig(
        api_key_env_var=llm_api_key_env_var,
        max_output_tokens=PHASE_9_JUDGE_MAX_OUTPUT_TOKENS,
    )

    # Run the judge on each round sequentially. The two rounds use
    # distinct round_ids so their judgment files don't collide on
    # disk.
    probe_1_judgment = judge_round(
        probe_1_path,
        output_dir=output_dir,
        llm_config=llm_config,
        llm_client=llm_client,
        notes=(
            "Phase 9 judge smoke Round 1 — Probe 1 checkpoint; judge "
            "reads multi-pass primary + adversarial readings; produces "
            "criterion judgments for the two active criteria + the "
            "held-out criterion."
        ),
    )
    probe_1_5_judgment = judge_round(
        probe_1_5_path,
        output_dir=output_dir,
        llm_config=llm_config,
        llm_client=llm_client,
        notes=(
            "Phase 9 judge smoke Round 2 — Probe 1.5 checkpoint; "
            "same shape as Round 1, against the Probe 1.5 round's "
            "multi-pass readings."
        ),
    )

    # Cross-round audit.
    all_records = (
        probe_1_judgment.judge_llm_call_records
        + probe_1_5_judgment.judge_llm_call_records
    )
    audit = LLMCallAudit.from_records(all_records)

    t1 = int(time.time() * 1000)
    result = Phase9JudgeSmokeResult(
        probe_1_judgment=probe_1_judgment,
        probe_1_5_judgment=probe_1_5_judgment,
        llm_call_audit=audit,
        wallclock_ms=t1 - t0,
        notes=notes,
    )

    _write_smoke_result(result, output_dir / "mirror")
    return result
