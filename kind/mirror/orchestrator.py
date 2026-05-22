"""Phase 8 adversarial-pass orchestrator.

This module composes Parts 1–4 into a single :func:`run_adversarial_pass`
that, given a checkpoint, runs *one* adversarial pass: it loads
telemetry, aligns perturbations, computes the committed statistic
results, builds the prompt fragments, calls the LLM for primary and
adversarial readings (active set), then repeats for the held-out
partition, runs a sham-perturbation calibration check, and writes
everything to ``runs/{run_id}/mirror/``.

**The one-way invariant.** The orchestrator's outputs all live under
``runs/{run_id}/mirror/`` (sibling of ``telemetry_dir``,
``checkpoints_dir``). No path writes to ``runs/{run_id}/telemetry/``,
``runs/{run_id}/checkpoints/``, or any agent-readable location. The
orchestrator does **not** construct an
:class:`~kind.agents.actor.Actor`, a
:class:`~kind.agents.world_model.WorldModel`, or a
:class:`~kind.training.runner.Runner`; it does not call
:meth:`~kind.training.runner.Runner.run`. The pre-registration goes
through :class:`~kind.observer.pre_reg.PreRegSink` *directly*, writing to
``runs/{run_id}/mirror/pre_reg/pre_reg.jsonl`` — the on-disk shape
matches Phase 5's contract (a directory with a single ``pre_reg.jsonl``
file) but the directory lives under ``mirror/`` to satisfy the
write-only-to-mirror-side invariant. The Phase 8 build spec's prose
about "the orchestrator constructs the runner with ``pre_reg_dir`` set
per Phase 5's contract" reconciled here with the semantic
:func:`tests.test_orchestrator_one_way_invariant.test_orchestrator_does_not_construct_actor_or_world_model`
check: the on-disk format is preserved; the construct-a-runner phrasing
is not. The decision is journaled in the Phase 8 entry.

**The two-pass structure.** The active-set readings are produced first
(pre-registered, then primary + adversarial LLM calls), then the
held-out readings are produced second against the same telemetry with
their own pre-registration record. The held-out pass's prompts contain
no reference to the active-set readings — the held-out partition is
the structural adversarial check on the active set.

**The sham-perturbation calibration check.** Any sham events in the
:class:`~kind.mirror.perturbation_align.PerturbationTimeline` are
walked after the readings come back; for each sham event, the
orchestrator looks for claims in the primary equanimity reading whose
``cited_step_range`` covers the sham ``t``. A non-empty list of such
claims is recorded as a calibration failure in
:attr:`PassResult.notes`; the sham check does not raise on failure
(the round records the outcome; future rounds interpret it).

Out of scope: any change to Io, the actor, the world model, the dream
state, the runner; multi-round analysis (Phase 12); real LLM API calls
in tests (tests inject a :class:`~kind.mirror.llm_caller.MockLLMClient`).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Final

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, field_validator

from kind.mirror.criteria_v2 import V2_REGISTRY
from kind.mirror.llm_caller import (
    LLMClient,
    LLMConfig,
    LLMRecordSink,
    MirrorReading,
    call_mirror_llm,
)
from kind.mirror.perturbation_align import (
    PerturbationEvent,
    PerturbationTimeline,
    align_perturbations,
)
from kind.mirror.prompt_builder import PromptFragment, build_fragment
from kind.mirror.registry import (
    Criterion,
    CriterionRegistry,
    ReadingSurface,
)
from kind.mirror.statistics import (
    StatisticConfig,
    StatisticResult,
    TelemetryBatch,
    compute_statistic,
)
from kind.observer.pre_reg import (
    ColumnInit,
    PRE_REG_FILE,
    PreRegistration,
    PreRegSink,
)

__all__ = [
    "PassConfig",
    "PassResult",
    "ShamCalibrationFinding",
    "run_adversarial_pass",
    "MIRROR_SUBDIR",
    "PRE_REG_SUBDIR",
    "PASSES_SUBDIR",
]


# ---------------------------------------------------------------------------
# Output-directory contract.
# ---------------------------------------------------------------------------

MIRROR_SUBDIR: Final[str] = "mirror"
PRE_REG_SUBDIR: Final[str] = "pre_reg"
PASSES_SUBDIR: Final[str] = "passes"

_AGENT_STEP_SUBDIR: Final[str] = "agent_step"
_DREAM_ROLLOUT_SUBDIR: Final[str] = "dream_rollout"
_WORLD_EVENT_FILE: Final[str] = "world_event.jsonl"


# ---------------------------------------------------------------------------
# Result records.
# ---------------------------------------------------------------------------


class ShamCalibrationFinding(BaseModel):
    """Per-sham-event record of the orchestrator's calibration check.

    Frozen. A claim is considered "potentially admitting at the sham
    timestamp" if it appears in the primary equanimity reading and its
    ``cited_step_range`` covers the sham ``t``. The check is conservative
    (any citation overlap, not just supportive claims) on the principle
    that the prompt explicitly tells the LLM not to admit equanimity at
    sham timestamps; any citation overlap is informationally worth
    surfacing for the round's interpretation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sham_t: int
    sham_wallclock_ms: int
    overlapping_primary_claim_indices: tuple[int, ...]
    note: str


class PassConfig(BaseModel):
    """Per-pass configuration for :func:`run_adversarial_pass`.

    Frozen. Fields:

    - ``run_id``: the run the pass reads from / writes under.
    - ``checkpoint_id``: the checkpoint this pass evaluates.
    - ``run_dir``: ``runs/{run_id}/`` root path. Subdirectories
      ``telemetry/``, ``checkpoints/``, and (orchestrator-written)
      ``mirror/`` live under this path.
    - ``active_registry``: registry of criteria the active-set readings
      cover; defaults to
      ``CriterionRegistry(criteria=V2_REGISTRY.active())``.
    - ``held_out_registry``: registry of criteria the held-out readings
      cover; defaults to
      ``CriterionRegistry(criteria=V2_REGISTRY.held_out())``.
    - ``statistic_config``: per-round statistic choices; defaults to
      :class:`StatisticConfig`'s defaults.
    - ``llm_config``: LLM-caller config; defaults to
      :class:`LLMConfig`'s defaults.
    - ``column_init``: the column-init the run's actor was constructed
      with. Surfaces in the pre-registration record.
    - ``builder_mode``: ``"proponent"`` or ``"skeptic"`` — the builder's
      mode for this pass; surfaces in the pre-registration record.
    - ``asymmetry_of_access``: free-text describing what Io reads vs
      what the mirror reads at this round.
    - ``perturbation_tolerance_ms``: alignment tolerance for the
      perturbation aligner.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    run_id: str
    checkpoint_id: str
    run_dir: Path
    active_registry: CriterionRegistry
    held_out_registry: CriterionRegistry
    statistic_config: StatisticConfig = StatisticConfig()
    llm_config: LLMConfig = LLMConfig()
    column_init: ColumnInit = "unknown"
    builder_mode: str = "proponent"
    asymmetry_of_access: str = (
        "Io reads PolicyView (action, action_logprob, policy_entropy, "
        "obs_hash, self_prediction_error scalar); the mirror reads "
        "TelemetryView (h_t, z_t, q/p params, KL per-dim and aggregate, "
        "recon_loss, encoder_embedding, intrinsic_signal, full "
        "self_prediction vector) plus orchestrator-side alignment of "
        "world_event timestamps to AgentStep.t — the membrane discipline."
    )
    perturbation_tolerance_ms: int = 1000

    @field_validator("run_id", "checkpoint_id")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty.")
        return value

    @field_validator("builder_mode")
    @classmethod
    def _validate_builder_mode(cls, value: str) -> str:
        if value not in {"proponent", "skeptic"}:
            raise ValueError(
                f"builder_mode must be 'proponent' or 'skeptic'; got {value!r}."
            )
        return value


class PassResult(BaseModel):
    """The output of one Phase 8 adversarial pass.

    Frozen. Carries every load-bearing artifact: both pre-registration
    records, the four reading tuples (active-primary,
    active-adversarial, held-out-primary, held-out-adversarial), the
    computed statistic results, the aligned perturbation timeline, and
    a free-text notes block carrying the sham-calibration findings.

    The on-disk form lives at
    ``runs/{run_id}/mirror/passes/{checkpoint_id}.json``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "0.1.0"
    run_id: str
    checkpoint_id: str
    timestamp_ms: int
    active_pre_registration: PreRegistration
    active_primary_readings: tuple[MirrorReading, ...]
    active_adversarial_readings: tuple[MirrorReading, ...]
    held_out_pre_registration: PreRegistration
    held_out_primary_readings: tuple[MirrorReading, ...]
    held_out_adversarial_readings: tuple[MirrorReading, ...]
    statistic_results: tuple[StatisticResult, ...]
    perturbation_timeline: PerturbationTimeline
    sham_calibration_findings: tuple[ShamCalibrationFinding, ...]
    notes: str


# ---------------------------------------------------------------------------
# Telemetry loading.
# ---------------------------------------------------------------------------


def _load_agent_step_rows(telemetry_dir: Path) -> tuple[dict[str, Any], ...]:
    """Load every :class:`~kind.observer.schemas.AgentStep` row in
    ``telemetry_dir/agent_step/`` (parquet shards). Returns rows sorted
    by ``t``."""
    shard_dir = telemetry_dir / _AGENT_STEP_SUBDIR
    if not shard_dir.is_dir():
        return tuple()
    rows: list[dict[str, Any]] = []
    for shard in sorted(shard_dir.glob("shard-*.parquet")):
        table = pq.read_table(str(shard))  # type: ignore[no-untyped-call]
        rows.extend(table.to_pylist())
    rows.sort(key=lambda r: int(r["t"]))
    return tuple(rows)


def _load_dream_rollout_rows(
    telemetry_dir: Path,
) -> tuple[dict[str, Any], ...]:
    shard_dir = telemetry_dir / _DREAM_ROLLOUT_SUBDIR
    if not shard_dir.is_dir():
        return tuple()
    rows: list[dict[str, Any]] = []
    for shard in sorted(shard_dir.glob("shard-*.parquet")):
        table = pq.read_table(str(shard))  # type: ignore[no-untyped-call]
        rows.extend(table.to_pylist())
    return tuple(rows)


def _build_telemetry_batch(
    telemetry_dir: Path,
    perturbation_timeline: PerturbationTimeline,
) -> TelemetryBatch:
    return TelemetryBatch(
        agent_step_rows=_load_agent_step_rows(telemetry_dir),
        dream_rollout_rows=_load_dream_rollout_rows(telemetry_dir),
        replay_meta_rows=tuple(),
        perturbation_step_indices=tuple(
            e.t for e in perturbation_timeline.events
        ),
        perturbation_is_sham=tuple(
            e.is_sham for e in perturbation_timeline.events
        ),
    )


def _build_agent_step_wallclock_lookup(
    telemetry_dir: Path,
) -> dict[int, int]:
    """Build a ``{t: wallclock_ms}`` lookup over the run's
    ``agent_step`` shards. Phase 12's
    :func:`~kind.mirror.calibration.sham_schedule.inject_sham_events`
    consumes this so each injected sham event's ``wallclock_ms`` is
    sourced from the orchestrator-side log (not synthesized)."""
    rows = _load_agent_step_rows(telemetry_dir)
    return {int(r["t"]): int(r["wallclock_ms"]) for r in rows}


# ---------------------------------------------------------------------------
# Per-criterion statistic + fragment building.
# ---------------------------------------------------------------------------


def _compute_results_for_registry(
    registry: CriterionRegistry,
    batch: TelemetryBatch,
    statistic_config: StatisticConfig,
) -> tuple[StatisticResult, ...]:
    """Compute every signal's :class:`StatisticResult` across a registry.

    Ordering: criteria in registry order; within a criterion, signals
    in the criterion's ``signal_mappings`` order.
    """
    results: list[StatisticResult] = []
    for criterion in registry.criteria:
        for mapping in criterion.signal_mappings:
            results.append(compute_statistic(batch, mapping, statistic_config))
    return tuple(results)


def _build_fragments_for_registry(
    registry: CriterionRegistry,
    results: tuple[StatisticResult, ...],
    perturbation_timeline: PerturbationTimeline,
) -> tuple[PromptFragment, ...]:
    """Build one :class:`PromptFragment` per criterion in the registry.

    Walks ``results`` in registry-criterion order, grouping by criterion
    so each fragment sees its own signals. The walk index advances
    through ``results`` as each criterion's signals are consumed.
    """
    fragments: list[PromptFragment] = []
    cursor = 0
    for criterion in registry.criteria:
        n_signals = len(criterion.signal_mappings)
        criterion_results = results[cursor : cursor + n_signals]
        cursor += n_signals
        fragments.append(
            build_fragment(
                criterion=criterion,
                statistic_results=criterion_results,
                perturbation_timeline=perturbation_timeline,
            )
        )
    return tuple(fragments)


# ---------------------------------------------------------------------------
# Pre-registration building.
# ---------------------------------------------------------------------------


def _build_pre_registration(
    *,
    run_id: str,
    active_registry: CriterionRegistry,
    held_out_registry: CriterionRegistry,
    target_registry: CriterionRegistry,
    column_init: ColumnInit,
    builder_mode: str,
    asymmetry_of_access: str,
) -> PreRegistration:
    """Build a :class:`PreRegistration` whose ``criteria_active`` is the
    ``target_registry``'s criterion ids and whose ``criteria_held_out``
    is the *other* partition's criterion ids.

    For an active-set pre-registration, ``target_registry`` is the
    active set; for the held-out pre-registration, ``target_registry``
    is the held-out set. The per-criterion dicts (``signal_mappings``,
    ``falsifiers``, ``scalar_checks``, ``reading_surfaces_per_criterion``)
    are filled only for the target registry's criteria — the
    :class:`PreRegistration` model's ``_enforce_per_criterion_completeness``
    validator requires entries for every active criterion only.
    """
    target_ids: list[str] = [c.id for c in target_registry.criteria]
    # The "other partition" is whichever of (active, held_out) is NOT
    # the target. Used to fill ``criteria_held_out`` in the model
    # (which from the *target* pre-registration's perspective is
    # everything else in the round).
    other_registry: CriterionRegistry
    if target_registry is active_registry:
        other_registry = held_out_registry
    else:
        other_registry = active_registry
    other_ids: list[str] = [c.id for c in other_registry.criteria]

    signal_mappings: dict[str, list[str]] = {}
    falsifiers: dict[str, str] = {}
    scalar_checks: dict[str, list[str]] = {}
    reading_surfaces_per_criterion: dict[str, list[ReadingSurface]] = {}
    expected_outcome_per_surface: dict[ReadingSurface, str] = {}

    for criterion in target_registry.criteria:
        signal_mappings[criterion.id] = [
            m.name for m in criterion.signal_mappings
        ]
        falsifiers[criterion.id] = criterion.falsifier
        # Phase 8 scalar_checks default: the estimator-name per signal.
        # A future round that adds explicit thresholds amends this list.
        scalar_checks[criterion.id] = [
            f"{m.name}::estimator_committed_at_phase8"
            for m in criterion.signal_mappings
        ]
        reading_surfaces_per_criterion[criterion.id] = sorted(
            criterion.reading_surfaces, key=lambda s: s.value
        )

    for surface in ReadingSurface:
        expected_outcome_per_surface[surface] = (
            f"to be determined empirically at this pass; the "
            f"pre-registration freezes the claim shape, not the "
            f"predicted value"
        )

    return PreRegistration(
        run_id=run_id,
        timestamp_ms=int(time.time() * 1000),
        criteria_active=target_ids,
        criteria_held_out=other_ids,
        signal_mappings=signal_mappings,
        falsifiers=falsifiers,
        scalar_checks=scalar_checks,
        reading_surfaces_per_criterion=reading_surfaces_per_criterion,
        asymmetry_of_access=asymmetry_of_access,
        builder_mode=builder_mode,  # type: ignore[arg-type]
        expected_outcome=(
            "the criterion's reading is computed honestly from the "
            "committed statistics; the pre-registration's job is to make "
            "any later drift visible — substantive prediction lives in the "
            "journal prose, not here"
        ),
        expected_outcome_per_surface=expected_outcome_per_surface,
        substrate_decisions_off_table=[
            "the actor's column initialization (small_gaussian default; "
            "the init is what it is for this run)",
            "the self-prediction head's structure (single-scalar Watts "
            "exception on PolicyView)",
            "the four telemetry-stream layout (agent_step, dream_rollout, "
            "replay_meta, world_event)",
        ],
        column_init=column_init,
        new_actor_readable_interfaces_added=[],
    )


# ---------------------------------------------------------------------------
# Sham calibration.
# ---------------------------------------------------------------------------


_EQUANIMITY_ID: Final[str] = "equanimity_perturbation_recovery"


def _sham_calibration_check(
    timeline: PerturbationTimeline,
    primary_active_readings: tuple[MirrorReading, ...],
    active_registry: CriterionRegistry,
) -> tuple[ShamCalibrationFinding, ...]:
    """For each sham event, find primary-equanimity claims whose
    ``cited_step_range`` covers the sham ``t``.

    The check looks only in the active partition's primary readings —
    the held-out partition is the adversarial check, not the
    calibration check; the adversarial partition is by construction
    arguing against admissions. Returns one finding per sham event;
    empty events list yields an empty tuple.
    """
    sham_events: list[PerturbationEvent] = [
        e for e in timeline.events if e.is_sham
    ]
    if not sham_events:
        return tuple()
    # Locate the primary equanimity reading, if present in this active set.
    equanimity_position = None
    for i, criterion in enumerate(active_registry.criteria):
        if criterion.id == _EQUANIMITY_ID:
            equanimity_position = i
            break
    findings: list[ShamCalibrationFinding] = []
    if equanimity_position is None:
        # Equanimity is not in the active set this round; no calibration
        # check to run — sham events still surface in the notes via the
        # caller, but no per-claim overlap analysis applies.
        for e in sham_events:
            findings.append(
                ShamCalibrationFinding(
                    sham_t=e.t,
                    sham_wallclock_ms=e.wallclock_ms,
                    overlapping_primary_claim_indices=tuple(),
                    note=(
                        "equanimity is not in this round's active set; "
                        "no primary-equanimity claims to check against"
                    ),
                )
            )
        return tuple(findings)
    if equanimity_position >= len(primary_active_readings):
        # Defensive — the active-readings tuple should have a slot per
        # active criterion, but if it doesn't, surface that explicitly.
        for e in sham_events:
            findings.append(
                ShamCalibrationFinding(
                    sham_t=e.t,
                    sham_wallclock_ms=e.wallclock_ms,
                    overlapping_primary_claim_indices=tuple(),
                    note=(
                        "primary-active-readings tuple has no slot for the "
                        "equanimity criterion's position; calibration check "
                        "could not run"
                    ),
                )
            )
        return tuple(findings)

    equanimity_reading = primary_active_readings[equanimity_position]
    for e in sham_events:
        overlapping: list[int] = []
        for i, claim in enumerate(equanimity_reading.claims):
            rng = claim.cited_step_range
            if rng is None:
                continue
            lo, hi = rng
            if lo <= e.t <= hi:
                overlapping.append(i)
        note = (
            "no overlapping claims — sham timestamp is structurally "
            "non-admitted at this pass"
            if not overlapping
            else (
                f"{len(overlapping)} primary-equanimity claim(s) cite a "
                f"step range covering the sham t={e.t}; an equanimity "
                f"admission at a sham timestamp is a calibration failure "
                f"— inspect the claims explicitly"
            )
        )
        findings.append(
            ShamCalibrationFinding(
                sham_t=e.t,
                sham_wallclock_ms=e.wallclock_ms,
                overlapping_primary_claim_indices=tuple(overlapping),
                note=note,
            )
        )
    return tuple(findings)


# ---------------------------------------------------------------------------
# Public entrypoint.
# ---------------------------------------------------------------------------


def run_adversarial_pass(
    config: PassConfig,
    *,
    llm_client: LLMClient | None = None,
    injected_sham_entries: tuple[Any, ...] = (),
    injected_synthetic_entries: tuple[Any, ...] = (),
    record_sink: LLMRecordSink | None = None,
) -> PassResult:
    """Run one Phase 8 adversarial pass.

    See the module docstring for the full step list. The flow in short:
    load telemetry → align perturbations → compute statistics →
    pre-register and call LLM (active set) → pre-register and call LLM
    (held-out set) → sham calibration check → write result JSON →
    return.

    ``llm_client``: if not provided, the default Gemini client is
    constructed inside :func:`~kind.mirror.llm_caller.call_mirror_llm`.
    Tests inject a :class:`~kind.mirror.llm_caller.MockLLMClient`.

    Phase 12 additions (both default-empty / ``None`` so Phase 8 callers
    keep working unchanged):

    - ``injected_sham_entries``: a tuple of
      :class:`~kind.mirror.calibration.sham_schedule.ShamScheduleEntry`
      records (typed as :class:`Any` here to avoid an import-cycle with
      the calibration package). When non-empty, the orchestrator
      synthesizes sham :class:`PerturbationEvent` records from the
      entries and merges them into the aligned timeline before
      statistics run. The merge respects the
      :class:`~kind.mirror.perturbation_align.PerturbationTimeline`'s
      sorted-unique invariant; a sham whose ``t`` collides with a real
      perturbation raises (the calibration cannot tolerate the
      ambiguity).
    - ``record_sink``: a Phase 12
      :class:`~kind.mirror.llm_caller.LLMRecordSink` that receives one
      :class:`~kind.mirror.calibration.llm_audit.LLMCallRecord` per
      LLM-call attempt across the four LLM calls this pass makes
      (active primary, active adversarial, held-out primary, held-out
      adversarial). The round driver constructs one collector per pass
      and reads its records after the pass returns.

    Phase 13 addition (default-empty so Phase 8/12 callers keep
    working):

    - ``injected_synthetic_entries``: a tuple of
      :class:`~kind.mirror.calibration.synthetic_perturbation.SyntheticPerturbationEntry`
      records. When non-empty, the orchestrator synthesizes
      :class:`PerturbationEvent` records with ``is_sham=False`` from
      the entries and merges them into the aligned timeline alongside
      any sham injections. The injected events route through the
      recovery-statistic codepath as real perturbations (the
      ``is_sham=False`` flag is what controls the dispatch); their
      ``is_synthetic=True`` payload flag is inspected by the round
      driver's synthetic calibration check. The two schedules are
      cross-disjoint at the schedule layer; collisions at injection
      time raise via the timeline's sorted-unique validator.
    """
    telemetry_dir = config.run_dir / "telemetry"
    mirror_dir = config.run_dir / MIRROR_SUBDIR
    pre_reg_dir = mirror_dir / PRE_REG_SUBDIR
    passes_dir = mirror_dir / PASSES_SUBDIR
    world_event_path = telemetry_dir / _WORLD_EVENT_FILE

    # 1. Align perturbations against the world-event log.
    aligned_timeline = align_perturbations(
        world_event_path,
        telemetry_dir,
        run_id=config.run_id,
        checkpoint_id=config.checkpoint_id,
        tolerance_ms=config.perturbation_tolerance_ms,
    )
    # 1a. Phase 12: inject scheduled sham events into the timeline.
    # 1b. Phase 13: inject scheduled synthetic events alongside the
    #     shams. The two injections compose: shams first (so the
    #     wallclock lookup is built once for both), then synthetics
    #     against the sham-augmented timeline. Cross-schedule
    #     disjointness is enforced at the schedule layer; a collision
    #     here would surface from the timeline's sorted-unique
    #     validator.
    wallclock_lookup: dict[int, int] | None = None
    if injected_sham_entries or injected_synthetic_entries:
        wallclock_lookup = _build_agent_step_wallclock_lookup(telemetry_dir)
    timeline = aligned_timeline
    if injected_sham_entries:
        # Import locally to avoid a circular import at module load time
        # (kind.mirror.calibration imports from kind.mirror.orchestrator
        # indirectly via the round driver).
        from kind.mirror.calibration.sham_schedule import inject_sham_events
        assert wallclock_lookup is not None  # narrowed by the if above
        timeline = inject_sham_events(
            timeline,
            injected_sham_entries,
            agent_step_wallclock_lookup=wallclock_lookup,
        )
    if injected_synthetic_entries:
        from kind.mirror.calibration.synthetic_perturbation import (
            inject_synthetic_events,
        )
        assert wallclock_lookup is not None  # narrowed by the if above
        timeline = inject_synthetic_events(
            timeline,
            injected_synthetic_entries,
            agent_step_wallclock_lookup=wallclock_lookup,
        )
    # 2. Build the in-memory batch.
    batch = _build_telemetry_batch(telemetry_dir, timeline)

    # 3. Compute statistic results for both partitions. Held-out and
    # active are computed against the same telemetry; the held-out
    # readings are produced second so the held-out pass cannot influence
    # the active pass (the readings are sequential, and the active LLM
    # call closes before the held-out call opens).
    active_results = _compute_results_for_registry(
        config.active_registry, batch, config.statistic_config
    )
    held_out_results = _compute_results_for_registry(
        config.held_out_registry, batch, config.statistic_config
    )
    all_results = active_results + held_out_results

    # 4. Build pre-registrations; emit them via PreRegSink. Active first,
    # then held-out, two appended records in the same JSONL.
    active_pre_reg = _build_pre_registration(
        run_id=config.run_id,
        active_registry=config.active_registry,
        held_out_registry=config.held_out_registry,
        target_registry=config.active_registry,
        column_init=config.column_init,
        builder_mode=config.builder_mode,
        asymmetry_of_access=config.asymmetry_of_access,
    )
    held_out_pre_reg = _build_pre_registration(
        run_id=config.run_id,
        active_registry=config.active_registry,
        held_out_registry=config.held_out_registry,
        target_registry=config.held_out_registry,
        column_init=config.column_init,
        builder_mode=config.builder_mode,
        asymmetry_of_access=config.asymmetry_of_access,
    )
    with PreRegSink(pre_reg_dir) as sink:
        sink.write(active_pre_reg)
        sink.write(held_out_pre_reg)

    # 5. Build prompt fragments.
    active_fragments = _build_fragments_for_registry(
        config.active_registry, active_results, timeline
    )
    held_out_fragments = _build_fragments_for_registry(
        config.held_out_registry, held_out_results, timeline
    )

    # 6. LLM calls — active primary + active adversarial.
    paired_id_active = f"{config.checkpoint_id}-active"
    active_primary = (
        call_mirror_llm(
            active_fragments,
            role="primary",
            config=config.llm_config,
            run_id=config.run_id,
            digest_run_id=config.run_id,
            digest_episode_range=_episode_range(batch),
            paired_reading_id=paired_id_active,
            client=llm_client,
            record_sink=record_sink,
        )
        if active_fragments
        else tuple()
    )
    active_adversarial = (
        call_mirror_llm(
            active_fragments,
            role="adversarial",
            config=config.llm_config,
            run_id=config.run_id,
            digest_run_id=config.run_id,
            digest_episode_range=_episode_range(batch),
            paired_reading_id=paired_id_active,
            client=llm_client,
            record_sink=record_sink,
        )
        if active_fragments
        else tuple()
    )

    # 7. Held-out LLM calls.
    paired_id_held_out = f"{config.checkpoint_id}-held_out"
    held_out_primary = (
        call_mirror_llm(
            held_out_fragments,
            role="primary",
            config=config.llm_config,
            run_id=config.run_id,
            digest_run_id=config.run_id,
            digest_episode_range=_episode_range(batch),
            paired_reading_id=paired_id_held_out,
            client=llm_client,
            record_sink=record_sink,
        )
        if held_out_fragments
        else tuple()
    )
    held_out_adversarial = (
        call_mirror_llm(
            held_out_fragments,
            role="adversarial",
            config=config.llm_config,
            run_id=config.run_id,
            digest_run_id=config.run_id,
            digest_episode_range=_episode_range(batch),
            paired_reading_id=paired_id_held_out,
            client=llm_client,
            record_sink=record_sink,
        )
        if held_out_fragments
        else tuple()
    )

    # 8. Sham calibration check.
    sham_findings = _sham_calibration_check(
        timeline, active_primary, config.active_registry
    )
    sham_notes = _format_sham_notes(sham_findings)

    # 9. Pack the result and write it to disk.
    result = PassResult(
        run_id=config.run_id,
        checkpoint_id=config.checkpoint_id,
        timestamp_ms=int(time.time() * 1000),
        active_pre_registration=active_pre_reg,
        active_primary_readings=active_primary,
        active_adversarial_readings=active_adversarial,
        held_out_pre_registration=held_out_pre_reg,
        held_out_primary_readings=held_out_primary,
        held_out_adversarial_readings=held_out_adversarial,
        statistic_results=all_results,
        perturbation_timeline=timeline,
        sham_calibration_findings=sham_findings,
        notes=sham_notes,
    )
    _write_pass_result(result, passes_dir)
    return result


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _episode_range(batch: TelemetryBatch) -> tuple[int, int]:
    if not batch.agent_step_rows:
        return (0, 0)
    eps = [int(row["episode_id"]) for row in batch.agent_step_rows]
    return (min(eps), max(eps))


def _format_sham_notes(findings: tuple[ShamCalibrationFinding, ...]) -> str:
    if not findings:
        return "no sham events in timeline"
    lines = ["sham-perturbation calibration findings:"]
    for f in findings:
        lines.append(
            f"  - sham_t={f.sham_t} "
            f"(wallclock_ms={f.sham_wallclock_ms}): "
            f"{len(f.overlapping_primary_claim_indices)} overlapping "
            f"primary-equanimity claim(s); {f.note}"
        )
    return "\n".join(lines)


def _write_pass_result(result: PassResult, passes_dir: Path) -> None:
    """Write the :class:`PassResult` to
    ``passes_dir/{checkpoint_id}.json``. The directory is created if
    needed; the file is atomically replaced (write-temp-then-rename) so
    a crash mid-write doesn't leave a half-finished JSON on disk."""
    passes_dir.mkdir(parents=True, exist_ok=True)
    final_path = passes_dir / f"{result.checkpoint_id}.json"
    tmp_path = passes_dir / f".{result.checkpoint_id}.json.tmp"
    payload = result.model_dump(mode="json")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    tmp_path.replace(final_path)


# Compatibility re-exports for callers that referenced the registry
# constants directly from the orchestrator namespace.
_V2_REGISTRY: Final[CriterionRegistry] = V2_REGISTRY
