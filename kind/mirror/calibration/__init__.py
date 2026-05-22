"""Phase 12 calibration plane — round driver, sham-schedule injection,
round-diff record, LLM-call audit, and the Phase 12 smoke harness.

The calibration plane stress-tests the Phase 8 mirror execution against
the real Gemini API for the first time. Components live in submodules:

- :mod:`kind.mirror.calibration.llm_audit` — per-attempt LLM-call records
  and the per-round audit aggregate. Records flow from the modified
  :mod:`kind.mirror.llm_caller` through the orchestrator to the round
  driver.
- :mod:`kind.mirror.calibration.sham_schedule` — deterministic
  sham-perturbation scheduling and timeline injection. The schedule is a
  pre-registration commitment; the mirror does not see sham/real labels
  in its prompt.
- :mod:`kind.mirror.calibration.round_diff` — cross-round diff record
  that names every changed configuration field with a required rationale.
  Two rounds with byte-identical ``StatisticConfig`` are comparable;
  rounds that differ need the diff record to be interpretable later.
- :mod:`kind.mirror.calibration.round` — :class:`RoundConfig`,
  :class:`RoundResult`, and :func:`run_round`. The round drives one or
  more checkpoints through ``passes_per_checkpoint`` adversarial passes
  apiece, emits the round-level pre-registration before the first pass
  runs, and aggregates sham findings across passes.
- :mod:`kind.mirror.calibration.smoke` —
  :func:`run_phase_12_smoke` and :class:`Phase12SmokeResult`. The Phase
  12 commitments (two rounds against one Probe 1 + one Probe 1.5
  checkpoint, five passes each, the default ``StatisticConfig`` from
  Phase 8, the default ``LLMConfig`` from Phase 8, seeds 42 and 43)
  are journaled at the function defaults.

**The round as pre-registration carrier.** The plan's load-bearing
constraint says "StatisticConfig commits at pre-registration time." The
implementation here interprets "the pre-registration record" as the
on-disk pre-registration artifact set — both Phase 0's
:class:`~kind.observer.pre_reg.PreRegistration` JSONL (per-pass
criterion-shape commitment) AND Phase 12's :class:`RoundConfig` JSON
(per-round statistic + sham commitment) — written before the first pass
runs. Pydantic ``frozen=True`` on :class:`RoundConfig` structurally
enforces "no mutation after the round starts"; the round driver writes
:class:`RoundConfig` to ``runs/{run_id}/mirror/pre_reg/round_{round_id}/
round_config.json`` before opening the per-pass loop. The plan's
alternative wording about extending :class:`PreRegistration` is honored
in spirit (the round's commitments are on disk pre-pass, frozen, and
load-bearing for any cross-round comparability) without churning the
Phase 0 schema surface.

**Out of scope.** Phases 9–11 (judges, aggregation protocols, larger
sweeps). Phase 12 is the calibration; later phases consume its findings.
"""

from kind.mirror.calibration.held_out_isolation import (
    HeldOutIsolationConfig,
    HeldOutIsolationReading,
    HeldOutIsolationStudy,
    PHASE_13_HELD_OUT_ISOLATION_N_RUNS,
    PHASE_13_HELD_OUT_ISOLATION_PASS_INDEX,
    PHASE_13_HELD_OUT_ISOLATION_SEED,
    load_reference_rounds_from_disk,
    run_held_out_isolation_study,
)
from kind.mirror.calibration.llm_audit import (
    LLMCallAudit,
    LLMCallRecord,
    LLMCallRecordCollector,
)
from kind.mirror.calibration.phase_13 import (
    PHASE_13_PASSES_PER_CHECKPOINT,
    PHASE_13_PROBE_1_5_SHAM_SEED,
    PHASE_13_PROBE_1_5_SYNTHETIC_SEED,
    PHASE_13_PROBE_1_SHAM_SEED,
    PHASE_13_PROBE_1_SYNTHETIC_SEED,
    PHASE_13_REAL_PERTURBATIONS_PER_PASS,
    PHASE_13_SHAMS_PER_PASS,
    PHASE_13_SYNTHETICS_PER_PASS,
    Phase13CalibrationResult,
    run_phase_13_calibration,
)
from kind.mirror.calibration.round import (
    EMPTY_SYNTHETIC_SCHEDULE,
    RoundConfig,
    RoundResult,
    ShamFindingsSummary,
    run_round,
)
from kind.mirror.calibration.round_diff import (
    ConfigFieldChange,
    CriterionSetChange,
    RoundDiff,
    SyntheticScheduleChange,
    compute_round_diff,
)
from kind.mirror.calibration.sham_schedule import (
    ShamSchedule,
    ShamScheduleEntry,
    generate_sham_schedule,
    inject_sham_events,
)
from kind.mirror.calibration.smoke import (
    PHASE_12_PASSES_PER_CHECKPOINT,
    PHASE_12_PROBE_1_SEED,
    PHASE_12_PROBE_1_5_SEED,
    PHASE_12_REAL_PERTURBATIONS_PER_PASS,
    PHASE_12_SHAMS_PER_PASS,
    Phase12SmokeResult,
    run_phase_12_smoke,
)
from kind.mirror.calibration.synthetic_calibration_check import (
    SyntheticCalibrationFinding,
    SyntheticFindingsSummary,
    aggregate_synthetic_findings,
    check_synthetic_calibration,
)
from kind.mirror.calibration.synthetic_perturbation import (
    SyntheticPerturbationEntry,
    SyntheticPerturbationSchedule,
    generate_synthetic_perturbation_schedule,
    inject_synthetic_events,
)

__all__ = [
    # LLM audit.
    "LLMCallAudit",
    "LLMCallRecord",
    "LLMCallRecordCollector",
    # Sham schedule.
    "ShamSchedule",
    "ShamScheduleEntry",
    "generate_sham_schedule",
    "inject_sham_events",
    # Round diff.
    "ConfigFieldChange",
    "CriterionSetChange",
    "RoundDiff",
    "SyntheticScheduleChange",
    "compute_round_diff",
    # Round driver.
    "EMPTY_SYNTHETIC_SCHEDULE",
    "RoundConfig",
    "RoundResult",
    "ShamFindingsSummary",
    "run_round",
    # Smoke harness.
    "PHASE_12_PASSES_PER_CHECKPOINT",
    "PHASE_12_PROBE_1_SEED",
    "PHASE_12_PROBE_1_5_SEED",
    "PHASE_12_REAL_PERTURBATIONS_PER_PASS",
    "PHASE_12_SHAMS_PER_PASS",
    "Phase12SmokeResult",
    "run_phase_12_smoke",
    # Phase 13 — synthetic perturbation schedule.
    "SyntheticPerturbationEntry",
    "SyntheticPerturbationSchedule",
    "generate_synthetic_perturbation_schedule",
    "inject_synthetic_events",
    # Phase 13 — synthetic calibration check.
    "SyntheticCalibrationFinding",
    "SyntheticFindingsSummary",
    "aggregate_synthetic_findings",
    "check_synthetic_calibration",
    # Phase 13 — held-out isolation study.
    "HeldOutIsolationConfig",
    "HeldOutIsolationReading",
    "HeldOutIsolationStudy",
    "PHASE_13_HELD_OUT_ISOLATION_N_RUNS",
    "PHASE_13_HELD_OUT_ISOLATION_PASS_INDEX",
    "PHASE_13_HELD_OUT_ISOLATION_SEED",
    "load_reference_rounds_from_disk",
    "run_held_out_isolation_study",
    # Phase 13 — calibration harness.
    "PHASE_13_PASSES_PER_CHECKPOINT",
    "PHASE_13_PROBE_1_5_SHAM_SEED",
    "PHASE_13_PROBE_1_5_SYNTHETIC_SEED",
    "PHASE_13_PROBE_1_SHAM_SEED",
    "PHASE_13_PROBE_1_SYNTHETIC_SEED",
    "PHASE_13_REAL_PERTURBATIONS_PER_PASS",
    "PHASE_13_SHAMS_PER_PASS",
    "PHASE_13_SYNTHETICS_PER_PASS",
    "Phase13CalibrationResult",
    "run_phase_13_calibration",
]
