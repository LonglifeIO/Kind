"""Phase 12 round driver — :class:`RoundConfig`, :class:`RoundResult`,
:func:`run_round`.

A *round* is a set of adversarial passes that share a
:class:`~kind.mirror.statistics.StatisticConfig` and a
:class:`~kind.mirror.calibration.sham_schedule.ShamSchedule`. The plan
calls out the round as a first-class object because round-to-round
comparability is the central calibration discipline that lands as
executable code in Phase 12.

**On-disk layout.** A round writes to::

    runs/phase_12_smoke/                          (output_dir; from the caller)
    └─ mirror/
       ├─ pre_reg/round_{round_id}/
       │   ├─ round_config.json                   (the round-level pre-registration)
       │   └─ pre_reg.jsonl                       (Phase 0's per-pass records)
       └─ rounds/{round_id}.json                  (the RoundResult)

The pass-level artifacts continue to land under each
checkpoint's own ``runs/{run_id}/mirror/`` per Phase 8's contract; the
round-level files live under the round driver's ``output_dir`` so a
single calibration run has one self-contained directory.

**The frozen-after-pre-registration invariant.** :class:`RoundConfig` is
Pydantic ``frozen=True``; once constructed, its fields cannot be
mutated. The round driver writes :class:`RoundConfig` to disk *before*
the per-pass loop opens. A test
(:func:`tests.test_round.test_round_config_frozen_after_pre_registration`)
asserts that attempting to mutate ``config.statistic_config`` raises
``ValidationError``. Phase 12's structural commitment lands here.

**The mirror does not see sham labels.** The sham injection runs
*orchestrator-side* (inside :func:`run_adversarial_pass`); the prompt
fragments cite the merged
:class:`~kind.mirror.perturbation_align.PerturbationTimeline` for
recovery-window step ranges only — ``is_sham`` is never surfaced to
the LLM. The post-call sham calibration check sees the flag; the LLM
does not.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from kind.mirror.calibration.llm_audit import (
    LLMCallRecord,
    LLMCallRecordCollector,
)
from kind.mirror.calibration.sham_schedule import (
    ShamSchedule,
    ShamScheduleEntry,
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
)
from kind.mirror.criteria_v2 import V2_REGISTRY
from kind.mirror.llm_caller import (
    LLMClient,
    LLMConfig,
)
from kind.mirror.orchestrator import (
    MIRROR_SUBDIR,
    PRE_REG_SUBDIR,
    PassConfig,
    PassResult,
    ShamCalibrationFinding,
    run_adversarial_pass,
)
from kind.mirror.registry import CriterionRegistry
from kind.mirror.statistics import StatisticConfig

# Phase-12 partition defaults — the V2 registry's
# ``active()`` / ``held_out()`` views, wrapped as sub-:class:`CriterionRegistry`
# instances for use as :class:`RoundConfig` defaults. These are
# constructed at module-load time and re-used as Pydantic field
# defaults across :class:`RoundConfig` instances; both objects are
# frozen, so the sharing is safe.
_V2_ACTIVE_REGISTRY: Final[CriterionRegistry] = CriterionRegistry(
    criteria=V2_REGISTRY.active()
)
_V2_HELD_OUT_REGISTRY: Final[CriterionRegistry] = CriterionRegistry(
    criteria=V2_REGISTRY.held_out()
)

# Phase 13: an empty synthetic-perturbation schedule. Phase 12 rounds
# (and any Phase 8/12 caller that didn't know about synthetics) default
# to this — backwards compatibility. Phase 13 rounds always supply a
# populated schedule.
EMPTY_SYNTHETIC_SCHEDULE: Final[SyntheticPerturbationSchedule] = (
    SyntheticPerturbationSchedule(
        entries=(), synthetics_per_pass=0, seed=0
    )
)
from kind.observer.pre_reg import (
    ColumnInit,
    PreRegistration,
    PreRegSink,
)

__all__ = [
    "CheckpointSpec",
    "EMPTY_SYNTHETIC_SCHEDULE",
    "RoundConfig",
    "RoundResult",
    "ShamFindingsSummary",
    "ROUNDS_SUBDIR",
    "ROUND_CONFIG_FILENAME",
    "run_round",
]


# ---------------------------------------------------------------------------
# Output-directory contract.
# ---------------------------------------------------------------------------

ROUNDS_SUBDIR: Final[str] = "rounds"
ROUND_CONFIG_FILENAME: Final[str] = "round_config.json"

# Matches :data:`~kind.mirror.registry._SNAKE_CASE_RE` — keep this in
# sync with the ``Criterion.id`` convention.
_ROUND_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]*$")
_ROUND_ID_MAX_LEN: Final[int] = 60


# ---------------------------------------------------------------------------
# Checkpoint spec.
# ---------------------------------------------------------------------------


class CheckpointSpec(BaseModel):
    """One checkpoint a round runs against.

    Frozen, ``extra="forbid"``. The plan's ``checkpoint_ids: tuple[str,
    ...]`` field is implemented here as a tuple of
    :class:`CheckpointSpec` so the round driver knows where each
    checkpoint's telemetry lives (its source ``run_dir``). The plan's
    string-only form was a simplification; the implementation needs the
    ``run_id`` and ``run_dir`` to find the parquet shards and
    ``world_event.jsonl`` log.

    Fields:

    - ``run_id``: the run the checkpoint belongs to
      (e.g. ``"probe1-20260503-123926"``).
    - ``checkpoint_id``: the checkpoint name within that run.
    - ``run_dir``: the on-disk root the run was written to
      (``runs/{run_id}/`` typically). The round driver reads
      ``run_dir/telemetry/`` for parquet shards and the world-event
      log; the orchestrator continues to write under
      ``run_dir/mirror/`` per the Phase 8 contract.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    checkpoint_id: str
    run_dir: Path

    @field_validator("run_id", "checkpoint_id")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty.")
        return value


# ---------------------------------------------------------------------------
# Records.
# ---------------------------------------------------------------------------


class RoundConfig(BaseModel):
    """The Phase 12 round-level pre-registration.

    Frozen, ``extra="forbid"``. Carries every choice the round commits
    to before the first pass runs:

    - ``round_id``: snake_case, regex-validated like a
      :class:`~kind.mirror.registry.Criterion` id.
    - ``checkpoints``: tuple of :class:`CheckpointSpec`. Phase 12's
      smoke commits two rounds with one checkpoint each (one Probe 1,
      one Probe 1.5); subsequent rounds may run multiple checkpoints.
    - ``passes_per_checkpoint``: how many adversarial passes to run on
      each checkpoint. Phase 12 commits ``5``.
    - ``statistic_config``: the per-round
      :class:`~kind.mirror.statistics.StatisticConfig`. Shared across
      all passes. The frozen invariant means the config cannot drift
      between passes within a round.
    - ``llm_config``: shared
      :class:`~kind.mirror.llm_caller.LLMConfig`.
    - ``sham_schedule``: the
      :class:`~kind.mirror.calibration.sham_schedule.ShamSchedule`
      that names every sham injection across the round. Pre-registered
      alongside the criterion-shape commitments.
    - ``synthetic_schedule`` (Phase 13): the
      :class:`~kind.mirror.calibration.synthetic_perturbation.SyntheticPerturbationSchedule`
      that names every synthetic-real-perturbation injection across the
      round. Independent of ``sham_schedule``; cross-disjoint at the
      ``(checkpoint_id, pass_index, t)`` slot level (enforced by the
      ``_enforce_synthetic_schedule_consistency`` model validator).
      Defaults to :data:`EMPTY_SYNTHETIC_SCHEDULE` so Phase 12 callers
      keep working unchanged; Phase 13 rounds populate it explicitly.
    - ``active_registry`` / ``held_out_registry``: the criterion-set
      partition for the round. Default to Phase 7's V2 partition (the
      active set has the three frozen criteria; held-out is empty for
      Phase 12 — the plan's
      ``behavior_side_scalar_conditioning`` held-out criterion is
      journaled but the criterion-set entry hasn't been added to the
      registry yet; the round still emits a held-out pre-registration
      record with an empty active list).
    - ``pre_registration_template_path``: optional path to a
      hand-filled prose template (the Phase 5 template).
      ``None`` means no template; the round still pre-registers via
      the auto-generated :class:`PreRegistration` records, but the
      journal entry's prose carries the freeze.
    - ``column_init`` / ``builder_mode``: surface-level fields the
      orchestrator's per-pass :class:`PassConfig` reads. Defaults match
      Phase 8.
    - ``perturbation_tolerance_ms``: passed through to the alignment.
    - ``notes``: free-text for the journal entry.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    round_id: str
    checkpoints: tuple[CheckpointSpec, ...]
    passes_per_checkpoint: int
    statistic_config: StatisticConfig
    llm_config: LLMConfig
    sham_schedule: ShamSchedule
    synthetic_schedule: SyntheticPerturbationSchedule = EMPTY_SYNTHETIC_SCHEDULE
    active_registry: CriterionRegistry = _V2_ACTIVE_REGISTRY
    held_out_registry: CriterionRegistry = _V2_HELD_OUT_REGISTRY
    pre_registration_template_path: Path | None = None
    column_init: ColumnInit = "unknown"
    builder_mode: str = "skeptic"
    perturbation_tolerance_ms: int = 1000
    notes: str = ""

    @field_validator("round_id")
    @classmethod
    def _validate_round_id(cls, value: str) -> str:
        if not _ROUND_ID_RE.fullmatch(value):
            raise ValueError(
                f"round_id must match {_ROUND_ID_RE.pattern!r} "
                f"(snake_case identifier); got {value!r}."
            )
        if len(value) > _ROUND_ID_MAX_LEN:
            raise ValueError(
                f"round_id must be <= {_ROUND_ID_MAX_LEN} chars; got "
                f"{len(value)}."
            )
        return value

    @field_validator("checkpoints")
    @classmethod
    def _validate_checkpoints_non_empty(
        cls, value: tuple[CheckpointSpec, ...]
    ) -> tuple[CheckpointSpec, ...]:
        if not value:
            raise ValueError(
                "RoundConfig: checkpoints tuple must be non-empty — a "
                "round with no checkpoints has nothing to read against."
            )
        return value

    @field_validator("passes_per_checkpoint")
    @classmethod
    def _validate_passes_per_checkpoint(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(
                f"passes_per_checkpoint must be > 0; got {value}."
            )
        return value

    @field_validator("builder_mode")
    @classmethod
    def _validate_builder_mode(cls, value: str) -> str:
        if value not in {"proponent", "skeptic"}:
            raise ValueError(
                f"builder_mode must be 'proponent' or 'skeptic'; got "
                f"{value!r}."
            )
        return value

    @field_validator("perturbation_tolerance_ms")
    @classmethod
    def _validate_tolerance_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(
                f"perturbation_tolerance_ms must be >= 0; got {value}."
            )
        return value

    @model_validator(mode="after")
    def _enforce_sham_schedule_consistency(self) -> "RoundConfig":
        """Every sham schedule entry's ``(checkpoint_id, pass_index)``
        must reference a checkpoint that this round will actually run.
        Entries scheduled against a checkpoint the round doesn't visit
        would be dead schedule entries; the calibration discipline
        requires the schedule to be tight.
        """
        valid_ckpts = {c.checkpoint_id for c in self.checkpoints}
        for entry in self.sham_schedule.entries:
            if entry.checkpoint_id not in valid_ckpts:
                raise ValueError(
                    f"RoundConfig: sham schedule entry references "
                    f"checkpoint_id={entry.checkpoint_id!r}, which is "
                    f"not in this round's checkpoints "
                    f"{sorted(valid_ckpts)}. Either add the checkpoint "
                    f"to the round or regenerate the schedule against "
                    f"the round's actual checkpoint list."
                )
            if entry.pass_index >= self.passes_per_checkpoint:
                raise ValueError(
                    f"RoundConfig: sham schedule entry has pass_index="
                    f"{entry.pass_index} but this round only runs "
                    f"{self.passes_per_checkpoint} passes per "
                    f"checkpoint."
                )
        return self

    @model_validator(mode="after")
    def _enforce_synthetic_schedule_consistency(self) -> "RoundConfig":
        """Every synthetic schedule entry's ``(checkpoint_id,
        pass_index)`` must reference a checkpoint/pass the round runs;
        and no synthetic entry may share a ``(checkpoint_id, pass_index,
        t)`` slot with a sham entry. Cross-schedule disjointness is the
        load-bearing calibration invariant — a single slot carrying
        both a sham and a synthetic would force the orchestrator to
        choose which calibration check to run, and either choice is
        wrong.
        """
        valid_ckpts = {c.checkpoint_id for c in self.checkpoints}
        for entry in self.synthetic_schedule.entries:
            if entry.checkpoint_id not in valid_ckpts:
                raise ValueError(
                    f"RoundConfig: synthetic schedule entry references "
                    f"checkpoint_id={entry.checkpoint_id!r}, which is "
                    f"not in this round's checkpoints "
                    f"{sorted(valid_ckpts)}. Either add the checkpoint "
                    f"to the round or regenerate the schedule against "
                    f"the round's actual checkpoint list."
                )
            if entry.pass_index >= self.passes_per_checkpoint:
                raise ValueError(
                    f"RoundConfig: synthetic schedule entry has "
                    f"pass_index={entry.pass_index} but this round only "
                    f"runs {self.passes_per_checkpoint} passes per "
                    f"checkpoint."
                )

        # Cross-schedule disjointness: a (checkpoint, pass, t) slot
        # can carry at most one of sham / synthetic.
        sham_slots = {
            (s.checkpoint_id, s.pass_index, s.sham_t)
            for s in self.sham_schedule.entries
        }
        for entry in self.synthetic_schedule.entries:
            slot = (entry.checkpoint_id, entry.pass_index, entry.synthetic_t)
            if slot in sham_slots:
                raise ValueError(
                    f"RoundConfig: synthetic schedule entry at slot "
                    f"(checkpoint_id={entry.checkpoint_id!r}, "
                    f"pass_index={entry.pass_index}, "
                    f"t={entry.synthetic_t}) collides with a sham "
                    f"schedule entry at the same slot. The calibration "
                    f"cannot tolerate the ambiguity — regenerate one "
                    f"of the schedules with a different seed, or pass "
                    f"sham_schedule= to the synthetic generator so the "
                    f"collision is avoided constructively."
                )
        return self


class ShamFindingsSummary(BaseModel):
    """Aggregated sham-calibration findings across all passes in a round.

    Frozen, ``extra="forbid"``. An "admission" is one
    :class:`~kind.mirror.orchestrator.ShamCalibrationFinding` whose
    ``overlapping_primary_claim_indices`` tuple is non-empty (the
    primary equanimity reading cited a step range covering the sham
    timestamp). Three breakdowns are recorded per the plan:

    - ``by_criterion``: count of admissions keyed by criterion id. At
      Phase 12 only the equanimity criterion is sham-checked by the
      orchestrator (see :func:`_sham_calibration_check`); the dict's
      shape generalizes to the future-extended check.
    - ``by_checkpoint``: count keyed by checkpoint id.
    - ``by_role``: count keyed by reader role. Only ``"primary"`` is
      checked at Phase 12; the dict shape carries the structure for
      later phases that may extend the check.
    - ``total``: total admissions across the round.
    - ``total_sham_events``: total sham events scheduled (admissions
      + clean shams). The admission rate is
      ``total / total_sham_events`` when ``total_sham_events > 0``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    by_criterion: dict[str, int]
    by_checkpoint: dict[str, int]
    by_role: dict[str, int]
    total: int
    total_sham_events: int

    @field_validator("total", "total_sham_events")
    @classmethod
    def _validate_total_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"count must be >= 0; got {value}.")
        return value


class RoundResult(BaseModel):
    """The full Phase 12 round's output.

    Frozen, ``extra="forbid"``. Carries the round config (so the
    journal entry can re-read what was committed), every per-pass
    :class:`PassResult`, the aggregated sham findings, the round's
    LLM-call records, the wallclock, and free-text notes.

    The on-disk form lives at
    ``output_dir/mirror/rounds/{round_id}.json`` and is written
    atomically (write-temp-then-rename).

    The ``llm_call_records`` tuple is the raw audit material; the
    smoke harness aggregates it (across rounds) into
    :class:`~kind.mirror.calibration.llm_audit.LLMCallAudit` for
    :class:`~kind.mirror.calibration.smoke.Phase12SmokeResult`. Keeping
    the raw records on :class:`RoundResult` (rather than the aggregate)
    matches the plan's RoundResult shape while preserving the data
    flow.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "0.1.0"
    round_id: str
    round_config: RoundConfig
    pass_results: tuple[PassResult, ...]
    sham_findings_summary: ShamFindingsSummary
    synthetic_findings_summary: SyntheticFindingsSummary
    llm_call_records: tuple[LLMCallRecord, ...]
    round_wallclock_ms: int
    notes: str


# ---------------------------------------------------------------------------
# Sham aggregation.
# ---------------------------------------------------------------------------


_EQUANIMITY_ID: Final[str] = "equanimity_perturbation_recovery"


def _aggregate_sham_findings(
    pass_results: tuple[PassResult, ...],
    *,
    sham_schedule: ShamSchedule,
) -> ShamFindingsSummary:
    """Aggregate per-pass sham findings into a round-level summary.

    Walks every :attr:`PassResult.sham_calibration_findings`, counts
    admissions (non-empty ``overlapping_primary_claim_indices``), and
    breaks the count down by criterion / checkpoint / role.

    The criterion is hardcoded to
    ``"equanimity_perturbation_recovery"`` at Phase 12 because the
    orchestrator's sham check only inspects the primary equanimity
    reading. The dict structure is preserved as a hook for later
    phases.

    ``total_sham_events`` is the number of sham schedule entries the
    round actually had (across all checkpoints and passes); used as
    the denominator for an admission-rate analysis at journal time.
    """
    by_criterion: dict[str, int] = {}
    by_checkpoint: dict[str, int] = {}
    by_role: dict[str, int] = {}
    total = 0
    for pr in pass_results:
        for finding in pr.sham_calibration_findings:
            if not finding.overlapping_primary_claim_indices:
                continue
            total += 1
            by_criterion[_EQUANIMITY_ID] = (
                by_criterion.get(_EQUANIMITY_ID, 0) + 1
            )
            by_checkpoint[pr.checkpoint_id] = (
                by_checkpoint.get(pr.checkpoint_id, 0) + 1
            )
            by_role["primary"] = by_role.get("primary", 0) + 1
    return ShamFindingsSummary(
        by_criterion=by_criterion,
        by_checkpoint=by_checkpoint,
        by_role=by_role,
        total=total,
        total_sham_events=len(sham_schedule.entries),
    )


# ---------------------------------------------------------------------------
# Pre-registration emission.
# ---------------------------------------------------------------------------


def _build_round_pre_registration(
    config: RoundConfig,
    checkpoint: CheckpointSpec,
    *,
    target_registry: CriterionRegistry,
    other_registry: CriterionRegistry,
) -> PreRegistration:
    """Build one :class:`PreRegistration` record for the round at a
    given checkpoint, with ``target_registry`` as the active partition.

    The record is the per-checkpoint shape commitment; the round-level
    statistic/sham commitments live in :class:`RoundConfig`'s on-disk
    JSON. The two together form the pre-registration artifact set for
    the round.
    """
    from kind.mirror.registry import ReadingSurface

    target_ids = [c.id for c in target_registry.criteria]
    other_ids = [c.id for c in other_registry.criteria]
    # Avoid double-listing — if target == other (e.g. V2_REGISTRY for
    # both partitions at Phase 12), the held-out list is empty per the
    # active/held_out disjoint invariant.
    if target_registry is other_registry:
        other_ids = []

    signal_mappings: dict[str, list[str]] = {}
    falsifiers: dict[str, str] = {}
    scalar_checks: dict[str, list[str]] = {}
    reading_surfaces_per_criterion: dict[str, list[ReadingSurface]] = {}
    for crit in target_registry.criteria:
        signal_mappings[crit.id] = [m.name for m in crit.signal_mappings]
        falsifiers[crit.id] = crit.falsifier
        scalar_checks[crit.id] = [
            f"{m.name}::estimator_committed_at_round_{config.round_id}"
            for m in crit.signal_mappings
        ]
        reading_surfaces_per_criterion[crit.id] = sorted(
            crit.reading_surfaces, key=lambda s: s.value
        )

    expected_outcome_per_surface = {
        surface: (
            f"round {config.round_id}: per-surface expected outcome is "
            f"empirical at this pass; the pre-registration freezes the "
            f"claim shape, not the predicted value"
        )
        for surface in ReadingSurface
    }

    return PreRegistration(
        run_id=checkpoint.run_id,
        timestamp_ms=int(time.time() * 1000),
        criteria_active=target_ids,
        criteria_held_out=other_ids,
        signal_mappings=signal_mappings,
        falsifiers=falsifiers,
        scalar_checks=scalar_checks,
        reading_surfaces_per_criterion=reading_surfaces_per_criterion,
        asymmetry_of_access=(
            "Io reads PolicyView (action, action_logprob, "
            "policy_entropy, obs_hash, self_prediction_error scalar); "
            "the mirror reads TelemetryView (h_t, z_t, q/p params, KL "
            "per-dim and aggregate, recon_loss, encoder_embedding, "
            "intrinsic_signal, full self_prediction vector) plus "
            "orchestrator-side alignment of world_event timestamps to "
            "AgentStep.t — the membrane discipline."
        ),
        builder_mode=config.builder_mode,  # type: ignore[arg-type]
        expected_outcome=(
            f"round {config.round_id}: calibration round committing "
            f"to StatisticConfig and ShamSchedule before passes run; "
            f"substantive prediction lives in the journal prose, not "
            f"here"
        ),
        expected_outcome_per_surface=expected_outcome_per_surface,
        substrate_decisions_off_table=[
            "the actor's column initialization (the init is what it "
            "is for the source runs)",
            "the self-prediction head's structure (single-scalar Watts "
            "exception on PolicyView)",
            "the four telemetry-stream layout (agent_step, "
            "dream_rollout, replay_meta, world_event)",
            "the StatisticConfig (Phase 8 commitments — see the round "
            "config's statistic_config field)",
            "the ShamSchedule (committed at round pre-registration — "
            "see the round config's sham_schedule field)",
        ],
        column_init=config.column_init,
        new_actor_readable_interfaces_added=[],
    )


def _write_round_config(
    config: RoundConfig, pre_reg_round_dir: Path
) -> Path:
    """Write the round config to disk atomically before the per-pass
    loop opens. The path is the round-level pre-registration carrier
    for ``StatisticConfig`` and ``ShamSchedule``.

    Returns the final path on success.
    """
    pre_reg_round_dir.mkdir(parents=True, exist_ok=True)
    final_path = pre_reg_round_dir / ROUND_CONFIG_FILENAME
    tmp_path = pre_reg_round_dir / f".{ROUND_CONFIG_FILENAME}.tmp"
    payload = config.model_dump(mode="json")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    tmp_path.replace(final_path)
    return final_path


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------


def _validate_round_config_on_disk(config: RoundConfig) -> None:
    """Phase 12's "validate config is well-formed" step.

    Checks each checkpoint has a ``telemetry/`` subdirectory under its
    ``run_dir`` (the orchestrator's preconditions). Anything stricter
    (parquet shape checks, world-event log existence) lives in the
    orchestrator's per-pass flow — at the round level we just sanity-
    check the directory layout.
    """
    for ckpt in config.checkpoints:
        if not ckpt.run_dir.is_dir():
            raise ValueError(
                f"run_round: checkpoint {ckpt.checkpoint_id!r}'s "
                f"run_dir does not exist: {ckpt.run_dir}"
            )
        telemetry_dir = ckpt.run_dir / "telemetry"
        if not telemetry_dir.is_dir():
            raise ValueError(
                f"run_round: checkpoint {ckpt.checkpoint_id!r}'s "
                f"run_dir has no telemetry/ subdir: {telemetry_dir}"
            )


def _sham_entries_for_pass(
    config: RoundConfig,
    checkpoint_id: str,
    pass_index: int,
) -> tuple[ShamScheduleEntry, ...]:
    """Filter the round's sham schedule to the entries that apply to
    one ``(checkpoint_id, pass_index)`` pair."""
    return tuple(
        e
        for e in config.sham_schedule.entries
        if e.checkpoint_id == checkpoint_id and e.pass_index == pass_index
    )


def _synthetic_entries_for_pass(
    config: RoundConfig,
    checkpoint_id: str,
    pass_index: int,
) -> tuple[SyntheticPerturbationEntry, ...]:
    """Filter the round's synthetic schedule to the entries that apply
    to one ``(checkpoint_id, pass_index)`` pair."""
    return tuple(
        e
        for e in config.synthetic_schedule.entries
        if e.checkpoint_id == checkpoint_id and e.pass_index == pass_index
    )


# ---------------------------------------------------------------------------
# Public entrypoint.
# ---------------------------------------------------------------------------


def run_round(
    config: RoundConfig,
    *,
    output_dir: Path,
    llm_client: LLMClient | None = None,
) -> RoundResult:
    """Run one Phase 12 round end-to-end.

    Sequence:

    1. Validate ``config`` (each checkpoint's run_dir / telemetry
       subdir exist on disk).
    2. Emit round-level pre-registration: write
       :class:`RoundConfig` to
       ``output_dir/mirror/pre_reg/round_{round_id}/round_config.json``
       and one :class:`PreRegistration` JSONL line per checkpoint into
       the sibling ``pre_reg.jsonl``. Both are on disk before the
       per-pass loop opens.
    3. For each checkpoint, for each pass index in ``range(
       config.passes_per_checkpoint)``: build a :class:`PassConfig`
       from the round's config, filter the sham schedule, construct an
       :class:`LLMCallRecordCollector`, and call
       :func:`~kind.mirror.orchestrator.run_adversarial_pass`.
    4. Collect every pass's :class:`PassResult` and every collector's
       :class:`LLMCallRecord` instances.
    5. Aggregate sham findings via :func:`_aggregate_sham_findings`.
    6. Write :class:`RoundResult` to
       ``output_dir/mirror/rounds/{round_id}.json`` atomically.

    The frozen-after-pre-registration invariant lives at the
    :class:`RoundConfig` Pydantic level: any attempt to mutate the
    config raises ``ValidationError``. The
    :func:`tests.test_round.test_round_config_frozen_after_pre_registration`
    test asserts this.
    """
    _validate_round_config_on_disk(config)

    output_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir = output_dir / MIRROR_SUBDIR
    pre_reg_round_dir = mirror_dir / PRE_REG_SUBDIR / f"round_{config.round_id}"
    pre_reg_round_dir.mkdir(parents=True, exist_ok=True)

    # 1. Round-level pre-registration: write the RoundConfig FIRST so
    # the on-disk artifact is in place before any per-pass record
    # appears. The driver does not re-read this file; the journal does.
    _write_round_config(config, pre_reg_round_dir)

    # 2. Per-checkpoint shape pre-registrations.
    with PreRegSink(pre_reg_round_dir) as sink:
        for ckpt in config.checkpoints:
            record = _build_round_pre_registration(
                config,
                ckpt,
                target_registry=config.active_registry,
                other_registry=config.held_out_registry,
            )
            sink.write(record)

    # 3. Per-checkpoint per-pass loop.
    round_t0 = int(time.time() * 1000)
    pass_results: list[PassResult] = []
    pass_synthetic_findings: list[tuple[PassResult, tuple[SyntheticCalibrationFinding, ...]]] = []
    all_records: list[LLMCallRecord] = []
    for ckpt in config.checkpoints:
        for pass_index in range(config.passes_per_checkpoint):
            pass_cfg = PassConfig(
                run_id=ckpt.run_id,
                checkpoint_id=ckpt.checkpoint_id,
                run_dir=ckpt.run_dir,
                active_registry=config.active_registry,
                held_out_registry=config.held_out_registry,
                statistic_config=config.statistic_config,
                llm_config=config.llm_config,
                column_init=config.column_init,
                builder_mode=config.builder_mode,
                perturbation_tolerance_ms=config.perturbation_tolerance_ms,
            )
            sham_entries = _sham_entries_for_pass(
                config, ckpt.checkpoint_id, pass_index
            )
            synthetic_entries = _synthetic_entries_for_pass(
                config, ckpt.checkpoint_id, pass_index
            )
            collector = LLMCallRecordCollector(
                round_id=config.round_id,
                pass_index=pass_index,
                checkpoint_id=ckpt.checkpoint_id,
            )
            pass_result = run_adversarial_pass(
                pass_cfg,
                llm_client=llm_client,
                injected_sham_entries=sham_entries,
                injected_synthetic_entries=synthetic_entries,
                record_sink=collector,
            )
            pass_results.append(pass_result)
            all_records.extend(collector.records)
            # Phase 13: run the synthetic calibration check
            # post-call, against this pass's readings. Even when the
            # synthetic schedule is empty (Phase 12 callers), the check
            # returns an empty tuple — the cost is negligible and the
            # uniform call path keeps the round driver shape stable.
            synthetic_findings = check_synthetic_calibration(
                pass_result, active_registry=config.active_registry
            )
            pass_synthetic_findings.append(
                (pass_result, synthetic_findings)
            )

    round_t1 = int(time.time() * 1000)

    # 4. Aggregate sham findings.
    sham_summary = _aggregate_sham_findings(
        tuple(pass_results), sham_schedule=config.sham_schedule
    )

    # 4b. Aggregate synthetic findings.
    synthetic_summary = aggregate_synthetic_findings(
        tuple(pass_synthetic_findings)
    )

    # 5. Pack and write the result.
    result = RoundResult(
        round_id=config.round_id,
        round_config=config,
        pass_results=tuple(pass_results),
        sham_findings_summary=sham_summary,
        synthetic_findings_summary=synthetic_summary,
        llm_call_records=tuple(all_records),
        round_wallclock_ms=round_t1 - round_t0,
        notes=config.notes,
    )
    _write_round_result(result, mirror_dir / ROUNDS_SUBDIR)
    return result


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _write_round_result(result: RoundResult, rounds_dir: Path) -> None:
    """Write the :class:`RoundResult` to
    ``rounds_dir/{round_id}.json`` atomically.

    Uses the same write-temp-then-rename pattern as
    :func:`~kind.mirror.orchestrator._write_pass_result` so a crash
    mid-write doesn't leave a half-finished JSON on disk.
    """
    rounds_dir.mkdir(parents=True, exist_ok=True)
    final_path = rounds_dir / f"{result.round_id}.json"
    tmp_path = rounds_dir / f".{result.round_id}.json.tmp"
    payload = result.model_dump(mode="json")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    tmp_path.replace(final_path)


# Compatibility re-exports.
_SHAM_CALIBRATION_FINDING: Final[type[ShamCalibrationFinding]] = (
    ShamCalibrationFinding
)
