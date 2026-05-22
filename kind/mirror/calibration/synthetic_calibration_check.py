"""Phase 13 synthetic-perturbation calibration check.

The orchestrator-side post-call check that verifies the mirror's
discriminative behavior on synthetic real perturbations. Mirrors the
shape of :func:`kind.mirror.orchestrator._sham_calibration_check` from
Phase 8 but operates over a broader matrix: per synthetic event, per
criterion in the active set, per reader role (primary + adversarial).

The discriminative-case question Phase 13 closes: at a synthetic-real-
perturbation timestamp the recovery statistics have *real values* (not
empty-baseline sentinels), so the mirror has something to discriminate
from. Phase 12's sham-zero result was vacuous in the absence of any
real perturbations; Phase 13's synthetic admissions are the
calibration's discriminative signal.

**An "admission" is a claim whose ``cited_step_range`` covers a
synthetic event's ``t``.** Same convention as the sham check — any
citation overlap is informationally worth surfacing, not just
admit-rather-than-refute claims. The journal entry interprets the
finding; the orchestrator records it.

**The recovery-lag cross-reference.** Each synthetic event's
:attr:`SyntheticCalibrationFinding.recovery_lag_at_synthetic` carries
the per-perturbation recovery lag the Phase 8 ``recovery_lag_steps``
estimator computed at this synthetic event's position. This is the
load-bearing number for the journal entry's "admits-at-synthetic-with-
short-lag" vs "admits-at-synthetic-with-non-recovery-sentinel" reading.

Out of scope: changing the sham check (Phase 8); writing the on-disk
artifact (the round driver does that); journal-side analysis (the
journal entry interprets, this module produces).
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, field_validator

from kind.mirror.llm_caller import MirrorReading, PassRole
from kind.mirror.orchestrator import PassResult
from kind.mirror.perturbation_align import PerturbationEvent
from kind.mirror.registry import CriterionRegistry
from kind.mirror.statistics import StatisticResult

__all__ = [
    "SyntheticCalibrationFinding",
    "SyntheticFindingsSummary",
    "check_synthetic_calibration",
    "aggregate_synthetic_findings",
]


# The signal name the recovery-lag cross-reference targets. Phase 8's
# ``compute_recovery_lag_steps`` emits a :class:`StatisticResult` whose
# ``signal_name`` is this constant; the finding reaches into it by name.
_RECOVERY_LAG_SIGNAL_NAME: Final[str] = "recovery_lag_steps"


# ---------------------------------------------------------------------------
# Per-event finding.
# ---------------------------------------------------------------------------


class SyntheticCalibrationFinding(BaseModel):
    """Per-(synthetic_event, criterion, role) record of the
    orchestrator's post-call calibration check.

    Frozen, ``extra="forbid"``. A finding's :attr:`admitted` flag is
    ``True`` iff at least one claim in the named reading's claim list
    cites a step range covering the synthetic event's ``t``. The
    overlapping claim indices are recorded explicitly so the journal
    entry can quote the claims verbatim.

    Fields:

    - ``synthetic_t``: the ``AgentStep`` step index the synthetic event
      was injected at.
    - ``synthetic_wallclock_ms``: the synthetic event's
      ``wallclock_ms`` (sourced from the agent-step lookup at
      injection time).
    - ``criterion_id``: which criterion's reading was checked.
    - ``reader_role``: ``"primary"`` or ``"adversarial"``.
    - ``admitted``: ``True`` iff any claim's ``cited_step_range``
      overlaps ``synthetic_t``.
    - ``overlapping_claim_indices``: the 0-based positions of the
      overlapping claims within the reading's claim list. Empty when
      ``admitted`` is ``False``.
    - ``recovery_lag_at_synthetic``: the per-perturbation recovery lag
      Phase 8's ``recovery_lag_steps`` computed at this synthetic
      event's index in the timeline's non-sham ordering. ``None`` when
      the recovery-lag statistic is not present in the pass's
      statistic results (e.g., the active set didn't include
      equanimity) or when the synthetic event's position couldn't be
      resolved.
    - ``note``: free-text contextual comment. The summary aggregator
      and the journal entry quote this.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    synthetic_t: int
    synthetic_wallclock_ms: int
    criterion_id: str
    reader_role: PassRole
    admitted: bool
    overlapping_claim_indices: tuple[int, ...]
    recovery_lag_at_synthetic: float | None
    note: str

    @field_validator("criterion_id")
    @classmethod
    def _validate_criterion_id_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("criterion_id must be non-empty.")
        return value


# ---------------------------------------------------------------------------
# Aggregated summary.
# ---------------------------------------------------------------------------


class SyntheticFindingsSummary(BaseModel):
    """Aggregated synthetic-calibration findings across all passes in a
    round.

    Frozen, ``extra="forbid"``. An "admission" is one
    :class:`SyntheticCalibrationFinding` whose ``admitted`` flag is
    ``True``. The summary's role-keyed dicts carry both ``"primary"``
    and ``"adversarial"`` — the synthetic check walks both readers (the
    sham check at Phase 8 walked only primary; the synthetic check
    widens the matrix because the discriminative case warrants
    inspection from both stances).

    Fields:

    - ``total_synthetic_events``: total findings produced across the
      round (one per ``(synthetic_event, criterion, role)`` triple).
    - ``total_admissions``: count of findings with ``admitted=True``.
    - ``admissions_by_criterion``: keyed by criterion id.
    - ``admissions_by_checkpoint``: keyed by checkpoint id.
    - ``admissions_by_role``: keyed by reader role.
    - ``mean_recovery_lag_at_admissions``: arithmetic mean of
      ``recovery_lag_at_synthetic`` over the admitted findings whose
      lag is non-``None``. ``None`` when no admissions carry a lag.
    - ``mean_recovery_lag_at_non_admissions``: same shape, over the
      non-admitted findings. ``None`` when no non-admissions carry a
      lag.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_synthetic_events: int
    total_admissions: int
    admissions_by_criterion: dict[str, int]
    admissions_by_checkpoint: dict[str, int]
    admissions_by_role: dict[str, int]
    mean_recovery_lag_at_admissions: float | None
    mean_recovery_lag_at_non_admissions: float | None

    @field_validator("total_synthetic_events", "total_admissions")
    @classmethod
    def _validate_total_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"count must be >= 0; got {value}.")
        return value


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _recovery_lag_for_synthetic(
    pass_result: PassResult,
    synthetic_event: PerturbationEvent,
) -> float | None:
    """Look up the per-perturbation recovery lag at ``synthetic_event``.

    Phase 8's :func:`~kind.mirror.statistics.compute_recovery_lag_steps`
    emits a single :class:`StatisticResult` whose ``value`` is a
    ``list[float]`` of per-non-sham-perturbation lags in
    ``batch.perturbation_step_indices`` order (matching the timeline's
    sorted-by-``t`` order, filtered to ``is_sham=False`` events). The
    synthetic event's index in that filtered list is its position in
    the lag-value list.

    Returns ``None`` when:

    - the pass's statistic_results don't include ``recovery_lag_steps``
      (e.g., the active set didn't include equanimity);
    - the recovery_lag_steps result's ``value`` is not a list (the
      empty-batch path returns ``[]`` but other paths could return a
      float; defensive);
    - the synthetic event's position resolves past the end of the lag
      list (shouldn't happen given a correctly-built timeline, but
      surfaced as ``None`` rather than raising).
    """
    recovery_result: StatisticResult | None = None
    for result in pass_result.statistic_results:
        if result.signal_name == _RECOVERY_LAG_SIGNAL_NAME:
            recovery_result = result
            break
    if recovery_result is None:
        return None
    value = recovery_result.value
    if not isinstance(value, list):
        return None
    # The synthetic event's position in the non-sham filtered timeline.
    non_sham_events = [
        e for e in pass_result.perturbation_timeline.events if not e.is_sham
    ]
    try:
        position = non_sham_events.index(synthetic_event)
    except ValueError:
        return None
    if position >= len(value):
        return None
    lag = value[position]
    # Defensive: the list elements should be floats (Phase 8's emit), but
    # the StatisticResult's value union admits non-float list elements
    # in theory; cast to float if numeric, else return None.
    if isinstance(lag, (int, float)):
        return float(lag)
    return None


def _readings_by_role(
    pass_result: PassResult, role: PassRole
) -> tuple[MirrorReading, ...]:
    """Return the per-criterion active-set readings for ``role``. Mirrors
    the sham check's read of ``primary_active_readings``; the synthetic
    check walks both roles."""
    if role == "primary":
        return pass_result.active_primary_readings
    return pass_result.active_adversarial_readings


def _overlapping_claim_indices(
    reading: MirrorReading, synthetic_t: int
) -> tuple[int, ...]:
    """Return the 0-based indices of claims whose ``cited_step_range``
    covers ``synthetic_t``. Claims without a step range are skipped."""
    overlapping: list[int] = []
    for i, claim in enumerate(reading.claims):
        rng = claim.cited_step_range
        if rng is None:
            continue
        lo, hi = rng
        if lo <= synthetic_t <= hi:
            overlapping.append(i)
    return tuple(overlapping)


# ---------------------------------------------------------------------------
# Public entrypoint.
# ---------------------------------------------------------------------------


def check_synthetic_calibration(
    pass_result: PassResult,
    *,
    active_registry: CriterionRegistry,
) -> tuple[SyntheticCalibrationFinding, ...]:
    """Walk every synthetic event in ``pass_result.perturbation_timeline``
    and emit one finding per ``(synthetic_event, criterion, role)``
    triple.

    A synthetic event is identified by ``payload.get("is_synthetic",
    False)`` — the explicit category flag the
    :class:`~kind.mirror.calibration.synthetic_perturbation.SyntheticPerturbationEntry`
    validator pins to ``True``. ``is_sham=False`` is also required (the
    three categories are mutually exclusive); a timeline event with both
    flags would be a programming error upstream and is silently
    treated as non-synthetic here (the upstream entry validator
    prevents this case).

    For each criterion in ``active_registry``, the function reads the
    primary and adversarial readings for that criterion (looked up by
    position in the active-set tuples), finds overlapping claims, and
    emits a finding. If a criterion's position runs past the available
    readings (defensive case — shouldn't happen given Phase 8's pass
    structure), the finding records ``admitted=False`` with an
    explanatory note.

    The recovery-lag cross-reference is best-effort: per-synthetic, the
    function reaches into ``pass_result.statistic_results`` for the
    ``recovery_lag_steps`` signal and indexes by the synthetic event's
    position in the non-sham timeline ordering. When the lookup
    succeeds, the finding's ``recovery_lag_at_synthetic`` carries the
    per-perturbation lag value; otherwise ``None``.

    Returns an empty tuple when no synthetic events are present.
    """
    synthetic_events = [
        e
        for e in pass_result.perturbation_timeline.events
        if e.payload.get("is_synthetic", False) and not e.is_sham
    ]
    if not synthetic_events:
        return tuple()

    findings: list[SyntheticCalibrationFinding] = []
    for synthetic_event in synthetic_events:
        recovery_lag = _recovery_lag_for_synthetic(
            pass_result, synthetic_event
        )
        for position, criterion in enumerate(active_registry.criteria):
            roles: tuple[PassRole, ...] = ("primary", "adversarial")
            for role_typed in roles:
                readings = _readings_by_role(pass_result, role_typed)
                if position >= len(readings):
                    findings.append(
                        SyntheticCalibrationFinding(
                            synthetic_t=synthetic_event.t,
                            synthetic_wallclock_ms=synthetic_event.wallclock_ms,
                            criterion_id=criterion.id,
                            reader_role=role_typed,
                            admitted=False,
                            overlapping_claim_indices=tuple(),
                            recovery_lag_at_synthetic=recovery_lag,
                            note=(
                                f"active-{role_typed} readings tuple has no "
                                f"slot for criterion position {position} "
                                f"({criterion.id!r}); no per-claim "
                                f"overlap analysis available"
                            ),
                        )
                    )
                    continue
                reading = readings[position]
                # The reading's claims may concern a different criterion
                # if upstream code reordered them. Defensive identity
                # check is left to the caller's structural test suite;
                # at this layer we trust the orchestrator's contract
                # that active-set readings are one-per-criterion in
                # registry order.
                overlapping = _overlapping_claim_indices(
                    reading, synthetic_event.t
                )
                admitted = len(overlapping) > 0
                if admitted:
                    note = (
                        f"{len(overlapping)} {role_typed} {criterion.id} "
                        f"claim(s) cite a step range covering the "
                        f"synthetic t={synthetic_event.t}; the "
                        f"discriminative case admitted"
                    )
                else:
                    note = (
                        f"no overlapping {role_typed} {criterion.id} claims "
                        f"— the synthetic timestamp is structurally "
                        f"non-admitted at this surface for this role"
                    )
                findings.append(
                    SyntheticCalibrationFinding(
                        synthetic_t=synthetic_event.t,
                        synthetic_wallclock_ms=synthetic_event.wallclock_ms,
                        criterion_id=criterion.id,
                        reader_role=role_typed,
                        admitted=admitted,
                        overlapping_claim_indices=overlapping,
                        recovery_lag_at_synthetic=recovery_lag,
                        note=note,
                    )
                )
    return tuple(findings)


# ---------------------------------------------------------------------------
# Aggregation.
# ---------------------------------------------------------------------------


def aggregate_synthetic_findings(
    pass_results_with_findings: tuple[
        tuple[PassResult, tuple[SyntheticCalibrationFinding, ...]], ...
    ],
) -> SyntheticFindingsSummary:
    """Aggregate per-pass findings into a round-level summary.

    Walks every ``(pass_result, findings)`` tuple. Counts admissions by
    criterion, checkpoint, and role. Computes the mean recovery-lag-at-
    admissions / non-admissions over the findings whose
    ``recovery_lag_at_synthetic`` is non-``None``.

    The aggregator takes the per-pass results paired with their findings
    so the checkpoint id (which lives on ``PassResult.checkpoint_id``
    but not on individual findings) can be threaded through the
    ``admissions_by_checkpoint`` dict without re-deriving it.
    """
    total_synthetic_events = 0
    total_admissions = 0
    admissions_by_criterion: dict[str, int] = {}
    admissions_by_checkpoint: dict[str, int] = {}
    admissions_by_role: dict[str, int] = {}
    lags_at_admissions: list[float] = []
    lags_at_non_admissions: list[float] = []

    for pass_result, findings in pass_results_with_findings:
        for finding in findings:
            total_synthetic_events += 1
            if finding.admitted:
                total_admissions += 1
                admissions_by_criterion[finding.criterion_id] = (
                    admissions_by_criterion.get(finding.criterion_id, 0)
                    + 1
                )
                admissions_by_checkpoint[pass_result.checkpoint_id] = (
                    admissions_by_checkpoint.get(
                        pass_result.checkpoint_id, 0
                    )
                    + 1
                )
                admissions_by_role[finding.reader_role] = (
                    admissions_by_role.get(finding.reader_role, 0) + 1
                )
                if finding.recovery_lag_at_synthetic is not None:
                    lags_at_admissions.append(
                        finding.recovery_lag_at_synthetic
                    )
            else:
                if finding.recovery_lag_at_synthetic is not None:
                    lags_at_non_admissions.append(
                        finding.recovery_lag_at_synthetic
                    )

    mean_lag_admissions = (
        sum(lags_at_admissions) / len(lags_at_admissions)
        if lags_at_admissions
        else None
    )
    mean_lag_non_admissions = (
        sum(lags_at_non_admissions) / len(lags_at_non_admissions)
        if lags_at_non_admissions
        else None
    )
    return SyntheticFindingsSummary(
        total_synthetic_events=total_synthetic_events,
        total_admissions=total_admissions,
        admissions_by_criterion=admissions_by_criterion,
        admissions_by_checkpoint=admissions_by_checkpoint,
        admissions_by_role=admissions_by_role,
        mean_recovery_lag_at_admissions=mean_lag_admissions,
        mean_recovery_lag_at_non_admissions=mean_lag_non_admissions,
    )
