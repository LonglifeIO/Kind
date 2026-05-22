"""Mirror: interpretive layer that translates Io's processing into something legible."""

from kind.mirror.calibration import (
    EMPTY_SYNTHETIC_SCHEDULE,
    PHASE_12_PASSES_PER_CHECKPOINT,
    PHASE_12_PROBE_1_5_SEED,
    PHASE_12_PROBE_1_SEED,
    PHASE_12_REAL_PERTURBATIONS_PER_PASS,
    PHASE_12_SHAMS_PER_PASS,
    PHASE_13_HELD_OUT_ISOLATION_N_RUNS,
    PHASE_13_HELD_OUT_ISOLATION_PASS_INDEX,
    PHASE_13_HELD_OUT_ISOLATION_SEED,
    PHASE_13_PASSES_PER_CHECKPOINT,
    PHASE_13_PROBE_1_5_SHAM_SEED,
    PHASE_13_PROBE_1_5_SYNTHETIC_SEED,
    PHASE_13_PROBE_1_SHAM_SEED,
    PHASE_13_PROBE_1_SYNTHETIC_SEED,
    PHASE_13_REAL_PERTURBATIONS_PER_PASS,
    PHASE_13_SHAMS_PER_PASS,
    PHASE_13_SYNTHETICS_PER_PASS,
    ConfigFieldChange,
    CriterionSetChange,
    HeldOutIsolationConfig,
    HeldOutIsolationReading,
    HeldOutIsolationStudy,
    LLMCallAudit,
    LLMCallRecord,
    LLMCallRecordCollector,
    Phase12SmokeResult,
    Phase13CalibrationResult,
    RoundConfig,
    RoundDiff,
    RoundResult,
    ShamFindingsSummary,
    ShamSchedule,
    ShamScheduleEntry,
    SyntheticCalibrationFinding,
    SyntheticFindingsSummary,
    SyntheticPerturbationEntry,
    SyntheticPerturbationSchedule,
    SyntheticScheduleChange,
    aggregate_synthetic_findings,
    check_synthetic_calibration,
    compute_round_diff,
    generate_sham_schedule,
    generate_synthetic_perturbation_schedule,
    inject_sham_events,
    inject_synthetic_events,
    load_reference_rounds_from_disk,
    run_held_out_isolation_study,
    run_phase_12_smoke,
    run_phase_13_calibration,
    run_round,
)
from kind.mirror.calibration.phase_9_judge_smoke import (
    Phase9JudgeSmokeResult,
    run_phase_9_judge_smoke,
)
from kind.mirror.admissibility import (
    AdmissibilityBatchResult,
    AdmissibilityVerdict,
    compute_admissibility,
    load_admissibility_inputs,
)
from kind.mirror.citation_canonical import canonicalize_scalar_field
from kind.mirror.faithfulness import (
    FAITHFULNESS_THRESHOLD,
    FAITHFULNESS_VALUE_TOLERANCE,
    FaithfulnessAssignment,
    FaithfulnessResult,
    FaithfulnessStatus,
    verify_reading,
)
from kind.mirror.criteria_v2 import (
    EQUANIMITY_PERTURBATION_RECOVERY,
    REFLEXIVE_ATTENTION,
    SECOND_ORDER_VOLITION,
    V2_REGISTRY,
)
from kind.mirror.judge import (
    ClaimPolarity,
    ClaimPolarityAssignment,
    CriterionJudgment,
    FalsifierVerdict,
    RoundJudgment,
    Verdict,
)
from kind.mirror.judge_driver import (
    JUDGMENTS_SUBDIR,
    judge_round,
    load_round_result_from_disk,
)
from kind.mirror.judge_llm_caller import (
    JUDGE_SYSTEM_PROMPT,
    ClaimPolarityAssignmentPayload,
    FalsifierVerdictPayload,
    JudgeBatchPayload,
    JudgePayload,
    MockJudgeLLMClient,
    call_judge_llm,
)
from kind.mirror.judge_prompt_builder import (
    JudgePromptFragment,
    build_judge_fragment,
)
from kind.mirror.llm_caller import (
    CallOutcome,
    LLMClient,
    LLMConfig,
    LLMRecordSink,
    MirrorLLMError,
    MirrorReading,
    MockLLMClient,
    PassRole,
    call_mirror_llm,
)
from kind.mirror.orchestrator import (
    PassConfig,
    PassResult,
    ShamCalibrationFinding,
    run_adversarial_pass,
)
from kind.mirror.perturbation_align import (
    PerturbationAlignmentError,
    PerturbationEvent,
    PerturbationTimeline,
    align_perturbations,
)
from kind.mirror.prompt_builder import (
    EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE,
    SECOND_ORDER_VOLITION_EXCLUSIONS,
    SHAM_PERTURBATION_NOTICE,
    PromptFragment,
    build_fragment,
)
from kind.mirror.registry import (
    EMPTY_REGISTRY,
    Criterion,
    CriterionRegistry,
    ReadingSurface,
    SignalMapping,
    TelemetrySurface,
)
from kind.mirror.stability import (
    PARAPHRASE_THRESHOLDS,
    PARAPHRASE_VARIANTS_PER_SURFACE,
    RESEED_THRESHOLDS,
    STABILITY_N_PARAPHRASES_DEFAULT,
    STABILITY_N_RESEEDS_DEFAULT,
    STABILITY_SEED_BASE,
    STABILITY_TEMPERATURE,
    StabilityResult,
    stability_check,
)
from kind.mirror.statistics import (
    StatisticConfig,
    StatisticResult,
    TelemetryBatch,
    compute_statistic,
)

__all__ = [
    # Phases 6 & 7 — the frozen criteria registry.
    "EMPTY_REGISTRY",
    "Criterion",
    "CriterionRegistry",
    "ReadingSurface",
    "SignalMapping",
    "TelemetrySurface",
    "REFLEXIVE_ATTENTION",
    "EQUANIMITY_PERTURBATION_RECOVERY",
    "SECOND_ORDER_VOLITION",
    "V2_REGISTRY",
    # Phase 8 — statistics.
    "StatisticConfig",
    "StatisticResult",
    "TelemetryBatch",
    "compute_statistic",
    # Phase 8 — perturbation alignment.
    "PerturbationAlignmentError",
    "PerturbationEvent",
    "PerturbationTimeline",
    "align_perturbations",
    # Phase 8 — prompt building.
    "EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE",
    "SECOND_ORDER_VOLITION_EXCLUSIONS",
    "SHAM_PERTURBATION_NOTICE",
    "PromptFragment",
    "build_fragment",
    # Phase 8 — LLM caller.
    "CallOutcome",
    "LLMClient",
    "LLMConfig",
    "LLMRecordSink",
    "MirrorLLMError",
    "MirrorReading",
    "MockLLMClient",
    "PassRole",
    "call_mirror_llm",
    # Phase 8 — orchestrator.
    "PassConfig",
    "PassResult",
    "ShamCalibrationFinding",
    "run_adversarial_pass",
    # Phase 12 — LLM audit.
    "LLMCallAudit",
    "LLMCallRecord",
    "LLMCallRecordCollector",
    # Phase 12 — sham schedule.
    "ShamSchedule",
    "ShamScheduleEntry",
    "generate_sham_schedule",
    "inject_sham_events",
    # Phase 12 — round diff.
    "ConfigFieldChange",
    "CriterionSetChange",
    "RoundDiff",
    "compute_round_diff",
    # Phase 12 — round driver.
    "RoundConfig",
    "RoundResult",
    "ShamFindingsSummary",
    "run_round",
    # Phase 12 — smoke harness.
    "PHASE_12_PASSES_PER_CHECKPOINT",
    "PHASE_12_PROBE_1_SEED",
    "PHASE_12_PROBE_1_5_SEED",
    "PHASE_12_REAL_PERTURBATIONS_PER_PASS",
    "PHASE_12_SHAMS_PER_PASS",
    "Phase12SmokeResult",
    "run_phase_12_smoke",
    # Phase 13 — synthetic perturbation schedule.
    "EMPTY_SYNTHETIC_SCHEDULE",
    "SyntheticPerturbationEntry",
    "SyntheticPerturbationSchedule",
    "generate_synthetic_perturbation_schedule",
    "inject_synthetic_events",
    # Phase 13 — synthetic calibration check.
    "SyntheticCalibrationFinding",
    "SyntheticFindingsSummary",
    "aggregate_synthetic_findings",
    "check_synthetic_calibration",
    # Phase 13 — round-diff dimension.
    "SyntheticScheduleChange",
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
    # Phase 9 — judge data plane.
    "ClaimPolarity",
    "ClaimPolarityAssignment",
    "FalsifierVerdict",
    "CriterionJudgment",
    "RoundJudgment",
    "Verdict",
    # Phase 9 — judge prompt builder.
    "JudgePromptFragment",
    "build_judge_fragment",
    # Phase 9 — judge LLM caller.
    "JUDGE_SYSTEM_PROMPT",
    "JudgePayload",
    "JudgeBatchPayload",
    "ClaimPolarityAssignmentPayload",
    "FalsifierVerdictPayload",
    "MockJudgeLLMClient",
    "call_judge_llm",
    # Phase 9 — judge driver.
    "JUDGMENTS_SUBDIR",
    "judge_round",
    "load_round_result_from_disk",
    # Phase 9 — judge smoke harness.
    "Phase9JudgeSmokeResult",
    "run_phase_9_judge_smoke",
    # Phase 10 — stability runner.
    "PARAPHRASE_THRESHOLDS",
    "PARAPHRASE_VARIANTS_PER_SURFACE",
    "RESEED_THRESHOLDS",
    "STABILITY_N_PARAPHRASES_DEFAULT",
    "STABILITY_N_RESEEDS_DEFAULT",
    "STABILITY_SEED_BASE",
    "STABILITY_TEMPERATURE",
    "StabilityResult",
    "stability_check",
    # Phase 11 — faithfulness verifier.
    "FAITHFULNESS_THRESHOLD",
    "FAITHFULNESS_VALUE_TOLERANCE",
    "FaithfulnessAssignment",
    "FaithfulnessResult",
    "FaithfulnessStatus",
    "canonicalize_scalar_field",
    "verify_reading",
    # Phase 12 — admissibility consumer.
    "AdmissibilityBatchResult",
    "AdmissibilityVerdict",
    "compute_admissibility",
    "load_admissibility_inputs",
]
