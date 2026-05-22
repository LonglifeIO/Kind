"""Phase 13 calibration harness — :func:`run_phase_13_calibration`
and :class:`Phase13CalibrationResult`.

Phase 13 is the discriminative-case extension to Phase 12's calibration.
The Phase 12 sham-zero result was vacuous because the source telemetry
contained zero real perturbations; Phase 13 closes the two newly-open
findings that gate Phase 9:

- §1: synthetic-real-perturbation injection gives the recovery
  statistics real values to compute against, so the mirror has a
  discriminative case to read. Two synthetic perturbations per pass;
  the calibration check inspects whether the primary and adversarial
  readers admit at the synthetic timestamps.
- §7: the held-out isolation study runs the Phase 12 pass-0 held-out
  prompt fragment without active-set context, ten times per
  checkpoint. The output distributions determine whether Phase 12's
  perfectly stable held-out readings were prompt-context interference
  (Reading (b)) or Phase 7's prose being sharp (Reading (a)).

**Phase 13 commitments (journaled at module-level constants).** Two
rounds: one Probe 1 + one Probe 1.5. 5 passes each. Sham seeds reuse
Phase 12's 42 / 43 (so the sham schedule is exactly comparable). The
synthetic seeds are 142 / 143 — distinct from Phase 12's; the
synthetic schedule is per-round deterministic.

**LLM cost.** Phase 13 makes Phase 12's 40 calls (the round driver
loop) + 20 isolation calls (10 per checkpoint × 2 checkpoints) = 60
calls minimum, more if the retry budget fires. At Phase 12's ~30 s
per call this is ~30 min wallclock. The journal entry records the
audit.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Final

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from kind.mirror.calibration.held_out_isolation import (
    HeldOutIsolationConfig,
    HeldOutIsolationStudy,
    run_held_out_isolation_study,
)
from kind.mirror.calibration.llm_audit import LLMCallAudit
from kind.mirror.calibration.round import (
    CheckpointSpec,
    RoundConfig,
    RoundResult,
    run_round,
)
from kind.mirror.calibration.round_diff import RoundDiff, compute_round_diff
from kind.mirror.calibration.sham_schedule import generate_sham_schedule
from kind.mirror.calibration.synthetic_perturbation import (
    generate_synthetic_perturbation_schedule,
)
from kind.mirror.llm_caller import LLMClient, LLMConfig
from kind.mirror.statistics import StatisticConfig

__all__ = [
    "PHASE_13_PASSES_PER_CHECKPOINT",
    "PHASE_13_REAL_PERTURBATIONS_PER_PASS",
    "PHASE_13_SHAMS_PER_PASS",
    "PHASE_13_SYNTHETICS_PER_PASS",
    "PHASE_13_PROBE_1_SHAM_SEED",
    "PHASE_13_PROBE_1_5_SHAM_SEED",
    "PHASE_13_PROBE_1_SYNTHETIC_SEED",
    "PHASE_13_PROBE_1_5_SYNTHETIC_SEED",
    "PHASE_13_PROBE_1_ROUND_ID",
    "PHASE_13_PROBE_1_5_ROUND_ID",
    "Phase13CalibrationResult",
    "run_phase_13_calibration",
]


# ---------------------------------------------------------------------------
# Phase 13 commitments.
# ---------------------------------------------------------------------------

PHASE_13_PASSES_PER_CHECKPOINT: Final[int] = 5
PHASE_13_REAL_PERTURBATIONS_PER_PASS: Final[int] = 2
PHASE_13_SHAMS_PER_PASS: Final[int] = 1
PHASE_13_SYNTHETICS_PER_PASS: Final[int] = 2
PHASE_13_PROBE_1_SHAM_SEED: Final[int] = 42
PHASE_13_PROBE_1_5_SHAM_SEED: Final[int] = 43
PHASE_13_PROBE_1_SYNTHETIC_SEED: Final[int] = 142
PHASE_13_PROBE_1_5_SYNTHETIC_SEED: Final[int] = 143
PHASE_13_PROBE_1_ROUND_ID: Final[str] = "phase_13_probe_1_round"
PHASE_13_PROBE_1_5_ROUND_ID: Final[str] = "phase_13_probe_1_5_round"

_AGENT_STEP_SUBDIR: Final[str] = "agent_step"


# ---------------------------------------------------------------------------
# Result.
# ---------------------------------------------------------------------------


class Phase13CalibrationResult(BaseModel):
    """The full Phase 13 calibration output.

    Frozen, ``extra="forbid"``. Carries both rounds' results, the diff
    record between them (with synthetic-schedule rationale because
    the seeds differ per checkpoint), the cross-round LLM-call audit,
    the held-out isolation study, and free-text notes.

    On-disk form:
    ``output_dir/mirror/phase_13_calibration_result.json``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "0.1.0"
    probe_1_round: RoundResult
    probe_1_5_round: RoundResult
    round_diff: RoundDiff
    llm_call_audit: LLMCallAudit
    held_out_isolation_study: HeldOutIsolationStudy
    wallclock_ms: int
    notes: str


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _count_agent_step_rows(run_dir: Path) -> int:
    """Mirrors the helper in smoke.py — total rows across every
    parquet shard under ``run_dir/telemetry/agent_step/``. The count
    is the ``telemetry_length`` the schedulers use for placement-window
    bounds."""
    shard_dir = run_dir / "telemetry" / _AGENT_STEP_SUBDIR
    if not shard_dir.is_dir():
        return 0
    total = 0
    for shard in sorted(shard_dir.glob("shard-*.parquet")):
        table = pq.read_table(str(shard))  # type: ignore[no-untyped-call]
        total += table.num_rows
    return total


def _build_round_config(
    *,
    round_id: str,
    checkpoint: CheckpointSpec,
    sham_seed: int,
    synthetic_seed: int,
    telemetry_length: int,
    llm_config: LLMConfig,
    statistic_config: StatisticConfig,
    notes: str,
) -> RoundConfig:
    """Construct a Phase-13-shaped :class:`RoundConfig` for one round.

    The shared commitments (``passes_per_checkpoint``,
    ``shams_per_pass``, ``synthetics_per_pass``,
    ``real_perturbations_per_pass``) come from the module-level
    constants. The seeds and checkpoint differ between Round 1 and
    Round 2.

    The sham schedule is built first, then the synthetic schedule is
    built with ``sham_schedule=`` passed in so the generator
    constructively avoids slot collisions.
    """
    sham_schedule = generate_sham_schedule(
        checkpoint_ids=(checkpoint.checkpoint_id,),
        passes_per_checkpoint=PHASE_13_PASSES_PER_CHECKPOINT,
        real_perturbations_per_pass=PHASE_13_REAL_PERTURBATIONS_PER_PASS,
        shams_per_pass=PHASE_13_SHAMS_PER_PASS,
        telemetry_length=telemetry_length,
        seed=sham_seed,
    )
    synthetic_schedule = generate_synthetic_perturbation_schedule(
        checkpoint_ids=(checkpoint.checkpoint_id,),
        passes_per_checkpoint=PHASE_13_PASSES_PER_CHECKPOINT,
        synthetics_per_pass=PHASE_13_SYNTHETICS_PER_PASS,
        telemetry_length=telemetry_length,
        recovery_window=statistic_config.recovery_window_W,
        seed=synthetic_seed,
        sham_schedule=sham_schedule,
    )
    return RoundConfig(
        round_id=round_id,
        checkpoints=(checkpoint,),
        passes_per_checkpoint=PHASE_13_PASSES_PER_CHECKPOINT,
        statistic_config=statistic_config,
        llm_config=llm_config,
        sham_schedule=sham_schedule,
        synthetic_schedule=synthetic_schedule,
        notes=notes,
    )


def _write_calibration_result(
    result: Phase13CalibrationResult, mirror_dir: Path
) -> None:
    """Write the :class:`Phase13CalibrationResult` to disk atomically.

    Same write-temp-then-rename pattern as :class:`RoundResult` and
    :class:`Phase12SmokeResult`: a crash mid-write doesn't leave a
    half-finished JSON on disk.
    """
    mirror_dir.mkdir(parents=True, exist_ok=True)
    final_path = mirror_dir / "phase_13_calibration_result.json"
    tmp_path = mirror_dir / ".phase_13_calibration_result.json.tmp"
    payload = result.model_dump(mode="json")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    tmp_path.replace(final_path)


# ---------------------------------------------------------------------------
# Public entrypoint.
# ---------------------------------------------------------------------------


def run_phase_13_calibration(
    probe_1_checkpoint: CheckpointSpec,
    probe_1_5_checkpoint: CheckpointSpec,
    *,
    output_dir: Path,
    llm_api_key_env_var: str = "GEMINI_API_KEY",
    llm_client: LLMClient | None = None,
    notes: str = "",
) -> Phase13CalibrationResult:
    """Run the Phase 13 calibration end-to-end against the real Gemini
    API.

    Sequence:

    1. Count telemetry rows per checkpoint (the schedulers' placement
       window).
    2. Build Round 1's :class:`RoundConfig` (Probe 1 checkpoint, sham
       seed 42, synthetic seed 142) and Round 2's (Probe 1.5
       checkpoint, sham seed 43, synthetic seed 143).
    3. Run Round 1, then Round 2 — sequentially. The synthetic
       calibration check fires after each pass inside the round
       driver; the round result carries the
       :class:`SyntheticFindingsSummary` aggregate.
    4. Compute the :class:`RoundDiff`. The two rounds share a
       :class:`StatisticConfig` and :class:`LLMConfig`; they differ
       on sham seed, synthetic seed, and checkpoint. The diff
       requires a ``synthetic_schedule`` rationale (the seeds differ
       by design).
    5. Aggregate the cross-round
       :class:`LLMCallAudit` from the concatenated per-round records.
    6. Run the held-out isolation study against the two rounds'
       pass-0 references. The isolation study makes its own LLM calls
       (10 per checkpoint × 2 checkpoints = 20 additional calls).
    7. Pack and write the :class:`Phase13CalibrationResult` atomically.

    ``llm_client``: when ``None``, the LLM caller constructs a default
    Gemini client via :envvar:`GEMINI_API_KEY` (or
    ``llm_api_key_env_var``). Tests inject a
    :class:`~kind.mirror.llm_caller.MockLLMClient`.
    """
    t0 = int(time.time() * 1000)

    # 1. Telemetry-length per checkpoint.
    probe_1_length = _count_agent_step_rows(probe_1_checkpoint.run_dir)
    probe_1_5_length = _count_agent_step_rows(probe_1_5_checkpoint.run_dir)
    statistic_config = StatisticConfig()
    min_length = 3 + statistic_config.recovery_window_W
    if probe_1_length < min_length:
        raise ValueError(
            f"run_phase_13_calibration: Probe 1 checkpoint at "
            f"{probe_1_checkpoint.run_dir} has telemetry_length="
            f"{probe_1_length}, which is below the synthetic "
            f"scheduler's minimum ({min_length} = 3 + recovery_window_W "
            f"= 3 + {statistic_config.recovery_window_W})."
        )
    if probe_1_5_length < min_length:
        raise ValueError(
            f"run_phase_13_calibration: Probe 1.5 checkpoint at "
            f"{probe_1_5_checkpoint.run_dir} has telemetry_length="
            f"{probe_1_5_length}, which is below the synthetic "
            f"scheduler's minimum ({min_length})."
        )

    llm_config = LLMConfig(api_key_env_var=llm_api_key_env_var)

    # 2. Build the two round configs.
    probe_1_config = _build_round_config(
        round_id=PHASE_13_PROBE_1_ROUND_ID,
        checkpoint=probe_1_checkpoint,
        sham_seed=PHASE_13_PROBE_1_SHAM_SEED,
        synthetic_seed=PHASE_13_PROBE_1_SYNTHETIC_SEED,
        telemetry_length=probe_1_length,
        llm_config=llm_config,
        statistic_config=statistic_config,
        notes=(
            "Phase 13 calibration Round 1 — Probe 1 checkpoint; 5 "
            "passes; default StatisticConfig; default LLMConfig "
            "(gemini-2.5-pro); 2 real perturbations + 1 sham + 2 "
            "synthetic per pass; sham seed 42, synthetic seed 142."
        ),
    )
    probe_1_5_config = _build_round_config(
        round_id=PHASE_13_PROBE_1_5_ROUND_ID,
        checkpoint=probe_1_5_checkpoint,
        sham_seed=PHASE_13_PROBE_1_5_SHAM_SEED,
        synthetic_seed=PHASE_13_PROBE_1_5_SYNTHETIC_SEED,
        telemetry_length=probe_1_5_length,
        llm_config=llm_config,
        statistic_config=statistic_config,
        notes=(
            "Phase 13 calibration Round 2 — Probe 1.5 checkpoint; 5 "
            "passes; default StatisticConfig; default LLMConfig "
            "(gemini-2.5-pro); 2 real perturbations + 1 sham + 2 "
            "synthetic per pass; sham seed 43, synthetic seed 143."
        ),
    )

    # 3. Run the two rounds sequentially.
    probe_1_round = run_round(
        probe_1_config, output_dir=output_dir, llm_client=llm_client
    )
    probe_1_5_round = run_round(
        probe_1_5_config, output_dir=output_dir, llm_client=llm_client
    )

    # 4. Cross-round diff. The synthetic-schedule rationale is
    # required (the seeds differ per checkpoint by design).
    round_diff = compute_round_diff(
        probe_1_config,
        probe_1_5_config,
        rationales={
            "synthetic_schedule": (
                "Phase 13 commits distinct synthetic seeds per "
                "checkpoint (142 / 143) so each round's schedule is "
                "independent. The seeds are journaled at the "
                "phase_13 module-level constants; the schedule "
                "differences are by design, not drift."
            )
        },
        notes=(
            "Phase 13 calibration: Round 1 vs Round 2 structural "
            "diff. The configs share StatisticConfig and LLMConfig "
            "by construction; the synthetic schedules differ by "
            "seed."
        ),
    )

    # 5. Run the held-out isolation study against the two rounds'
    # pass-0 references. The audit aggregation happens *after* this so
    # the isolation study's per-run records can be folded in alongside
    # the round records (the audit covers the full Phase 13 LLM-call
    # surface, not just the round driver's calls).
    iso_config = HeldOutIsolationConfig(
        probe_1_checkpoint_id=probe_1_checkpoint.checkpoint_id,
        probe_1_5_checkpoint_id=probe_1_5_checkpoint.checkpoint_id,
        llm_config=llm_config,
    )
    isolation_study = run_held_out_isolation_study(
        iso_config,
        reference_rounds=(probe_1_round, probe_1_5_round),
        llm_client=llm_client,
    )

    # 6. Aggregate the cross-round audit, including the isolation
    # study's per-run records. The Phase 13 audit reflects every LLM
    # call the calibration made — round-driver calls plus
    # isolation-study calls — so cost / retry / latency analyses see
    # the full surface in one place.
    isolation_records = tuple(
        rec
        for reading in (
            *isolation_study.probe_1_readings,
            *isolation_study.probe_1_5_readings,
        )
        for rec in reading.call_records
    )
    all_records = (
        probe_1_round.llm_call_records
        + probe_1_5_round.llm_call_records
        + isolation_records
    )
    audit = LLMCallAudit.from_records(all_records)

    t1 = int(time.time() * 1000)
    result = Phase13CalibrationResult(
        probe_1_round=probe_1_round,
        probe_1_5_round=probe_1_5_round,
        round_diff=round_diff,
        llm_call_audit=audit,
        held_out_isolation_study=isolation_study,
        wallclock_ms=t1 - t0,
        notes=notes,
    )

    # 7. Write to disk.
    _write_calibration_result(result, output_dir / "mirror")
    return result
