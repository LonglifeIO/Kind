"""Phase 13 gate test —
:mod:`kind.mirror.calibration.synthetic_calibration_check`.

Covers the orchestrator-side calibration check that walks each synthetic
event in the pass's perturbation timeline and produces one finding per
``(synthetic_event, criterion, role)`` triple.

- Empty timeline → empty findings tuple.
- A reading whose claim cites the synthetic ``t`` → admitted finding,
  ``overlapping_claim_indices`` non-empty.
- A reading whose claim cites elsewhere → non-admitted finding,
  ``overlapping_claim_indices=()``.
- Recovery-lag cross-reference resolves from the pass's
  ``recovery_lag_steps`` :class:`StatisticResult`.
- :func:`aggregate_synthetic_findings` rolls per-pass findings into the
  round-level :class:`SyntheticFindingsSummary` with correct breakdowns.
- Round-trip serialization of both record types.
"""

from __future__ import annotations

from typing import Any

from kind.mirror.calibration.synthetic_calibration_check import (
    SyntheticCalibrationFinding,
    SyntheticFindingsSummary,
    aggregate_synthetic_findings,
    check_synthetic_calibration,
)
from kind.mirror.criteria_v2 import V2_REGISTRY
from kind.mirror.llm_caller import MirrorReading
from kind.mirror.orchestrator import PassResult
from kind.mirror.perturbation_align import (
    PerturbationEvent,
    PerturbationTimeline,
)
from kind.mirror.registry import CriterionRegistry
from kind.mirror.statistics import StatisticResult
from kind.mirror.structured import StructuredClaim
from kind.observer.pre_reg import PreRegistration


# ---------------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------------


def _claim(
    cited_step_range: tuple[int, int] | None,
    cited_value: float = 0.0,
) -> StructuredClaim:
    return StructuredClaim(
        claim="example claim",
        cited_stream="agent_step",
        cited_run_id="r",
        cited_episode_range=(0, 1),
        cited_step_range=cited_step_range,
        cited_scalar_field="h_t",
        cited_value=cited_value,
        falsifier="f",
        paraphrase_stability=None,
        reseed_stability=None,
        faithfulness_status="not_checked",
        judge_ruling="not_judged",
        reading_surface="substrate_side",
        masked_steps_handling="n/a",
    )


def _reading(criterion_id: str, claims: list[StructuredClaim]) -> MirrorReading:
    return MirrorReading(
        run_id="r",
        timestamp_ms=0,
        reader_role="advocate",
        paired_reading_id=None,
        framework_anchor="buddhist_phenomenology",
        baseline_flag="genuine",
        digest_run_id="r",
        digest_episode_range=(0, 1),
        claims=claims,
        free_text_notes="",
    )


def _adversarial_reading(
    criterion_id: str, claims: list[StructuredClaim]
) -> MirrorReading:
    reading = _reading(criterion_id, claims)
    return reading.model_copy(update={"reader_role": "skeptic"})


def _pre_reg() -> PreRegistration:
    return PreRegistration(
        run_id="r",
        timestamp_ms=0,
        criteria_active=[c.id for c in V2_REGISTRY.active()],
        criteria_held_out=[c.id for c in V2_REGISTRY.held_out()],
        signal_mappings={
            c.id: [m.name for m in c.signal_mappings]
            for c in V2_REGISTRY.active()
        },
        falsifiers={c.id: c.falsifier for c in V2_REGISTRY.active()},
        scalar_checks={
            c.id: [f"{m.name}::e" for m in c.signal_mappings]
            for c in V2_REGISTRY.active()
        },
        reading_surfaces_per_criterion={
            c.id: sorted(c.reading_surfaces, key=lambda s: s.value)
            for c in V2_REGISTRY.active()
        },
        asymmetry_of_access="x",
        builder_mode="skeptic",
        expected_outcome="x",
        expected_outcome_per_surface={},
        substrate_decisions_off_table=[],
        column_init="unknown",
        new_actor_readable_interfaces_added=[],
    )


def _recovery_lag_result(lags: list[float]) -> StatisticResult:
    return StatisticResult(
        signal_name="recovery_lag_steps",
        value=lags,
        estimator="mahalanobis_recovery_lag_p95_3step_streak",
        n_samples=len(lags),
        notes="test fixture",
    )


def _build_pass_result(
    *,
    timeline_events: tuple[PerturbationEvent, ...],
    primary_readings: tuple[MirrorReading, ...],
    adversarial_readings: tuple[MirrorReading, ...],
    statistic_results: tuple[StatisticResult, ...] = (),
    checkpoint_id: str = "ckpt-1",
) -> PassResult:
    timeline = PerturbationTimeline(
        events=timeline_events,
        run_id="r",
        checkpoint_id=checkpoint_id,
    )
    return PassResult(
        run_id="r",
        checkpoint_id=checkpoint_id,
        timestamp_ms=0,
        active_pre_registration=_pre_reg(),
        active_primary_readings=primary_readings,
        active_adversarial_readings=adversarial_readings,
        held_out_pre_registration=_pre_reg(),
        held_out_primary_readings=tuple(),
        held_out_adversarial_readings=tuple(),
        statistic_results=statistic_results,
        perturbation_timeline=timeline,
        sham_calibration_findings=tuple(),
        notes="",
    )


def _active_registry() -> CriterionRegistry:
    return CriterionRegistry(criteria=V2_REGISTRY.active())


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_empty_timeline_produces_no_findings() -> None:
    """A pass result with no synthetic events in its timeline produces
    an empty findings tuple."""
    pass_result = _build_pass_result(
        timeline_events=tuple(),
        primary_readings=tuple(),
        adversarial_readings=tuple(),
    )
    findings = check_synthetic_calibration(
        pass_result, active_registry=_active_registry()
    )
    assert findings == tuple()


def test_timeline_with_only_real_perturbations_produces_no_findings() -> None:
    """Real perturbations (no ``is_synthetic`` flag) are not picked up
    by the synthetic check."""
    pass_result = _build_pass_result(
        timeline_events=(
            PerturbationEvent(
                t=50,
                wallclock_ms=5000,
                payload={"kind": "real"},
                is_sham=False,
            ),
        ),
        primary_readings=(
            _reading("reflexive_attention", [_claim((0, 100))]),
            _reading("equanimity_perturbation_recovery", [_claim((0, 100))]),
        ),
        adversarial_readings=(
            _adversarial_reading(
                "reflexive_attention", [_claim((0, 100))]
            ),
            _adversarial_reading(
                "equanimity_perturbation_recovery", [_claim((0, 100))]
            ),
        ),
    )
    findings = check_synthetic_calibration(
        pass_result, active_registry=_active_registry()
    )
    assert findings == tuple()


def test_synthetic_event_with_overlapping_claim_produces_admitted_finding() -> None:
    """A synthetic event whose ``t`` falls inside an equanimity claim's
    ``cited_step_range`` produces an ``admitted=True`` finding."""
    pass_result = _build_pass_result(
        timeline_events=(
            PerturbationEvent(
                t=50,
                wallclock_ms=5000,
                payload={"is_synthetic": True, "is_sham": False},
                is_sham=False,
            ),
        ),
        primary_readings=(
            _reading("reflexive_attention", [_claim((200, 300))]),
            # The equanimity claim's range covers t=50.
            _reading(
                "equanimity_perturbation_recovery", [_claim((40, 60))]
            ),
        ),
        adversarial_readings=(
            _adversarial_reading(
                "reflexive_attention", [_claim((200, 300))]
            ),
            _adversarial_reading(
                "equanimity_perturbation_recovery", [_claim((200, 300))]
            ),
        ),
    )
    findings = check_synthetic_calibration(
        pass_result, active_registry=_active_registry()
    )
    # 1 synthetic × 2 criteria × 2 roles = 4 findings.
    assert len(findings) == 4
    # The equanimity primary finding admits; the others don't.
    by_key = {(f.criterion_id, f.reader_role): f for f in findings}
    eq_primary = by_key[("equanimity_perturbation_recovery", "primary")]
    eq_adversarial = by_key[
        ("equanimity_perturbation_recovery", "adversarial")
    ]
    rf_primary = by_key[("reflexive_attention", "primary")]
    assert eq_primary.admitted is True
    assert eq_primary.overlapping_claim_indices == (0,)
    assert eq_adversarial.admitted is False
    assert rf_primary.admitted is False


def test_synthetic_event_without_overlapping_claim_produces_non_admitted_finding() -> None:
    """Synthetic at t=50 with a claim covering only (200, 300) produces
    ``admitted=False``."""
    pass_result = _build_pass_result(
        timeline_events=(
            PerturbationEvent(
                t=50,
                wallclock_ms=5000,
                payload={"is_synthetic": True, "is_sham": False},
                is_sham=False,
            ),
        ),
        primary_readings=(
            _reading("reflexive_attention", [_claim((200, 300))]),
            _reading(
                "equanimity_perturbation_recovery", [_claim((200, 300))]
            ),
        ),
        adversarial_readings=(
            _adversarial_reading(
                "reflexive_attention", [_claim((200, 300))]
            ),
            _adversarial_reading(
                "equanimity_perturbation_recovery", [_claim((200, 300))]
            ),
        ),
    )
    findings = check_synthetic_calibration(
        pass_result, active_registry=_active_registry()
    )
    assert all(f.admitted is False for f in findings)
    assert all(f.overlapping_claim_indices == tuple() for f in findings)


def test_recovery_lag_cross_reference_resolves_correctly() -> None:
    """The synthetic event's ``recovery_lag_at_synthetic`` is the
    per-perturbation lag at its position in the non-sham timeline
    ordering."""
    # Two synthetic events at t=20 and t=80 (sorted). Their positions in
    # the non-sham timeline ordering are 0 and 1; the
    # ``recovery_lag_steps`` value list aligns by position.
    pass_result = _build_pass_result(
        timeline_events=(
            PerturbationEvent(
                t=20,
                wallclock_ms=2000,
                payload={"is_synthetic": True, "is_sham": False},
                is_sham=False,
            ),
            PerturbationEvent(
                t=80,
                wallclock_ms=8000,
                payload={"is_synthetic": True, "is_sham": False},
                is_sham=False,
            ),
        ),
        primary_readings=(
            _reading("reflexive_attention", []),
            _reading("equanimity_perturbation_recovery", []),
        ),
        adversarial_readings=(
            _adversarial_reading("reflexive_attention", []),
            _adversarial_reading(
                "equanimity_perturbation_recovery", []
            ),
        ),
        statistic_results=(
            _recovery_lag_result([3.0, 51.0]),  # t=20 → 3.0; t=80 → 51.0
        ),
    )
    findings = check_synthetic_calibration(
        pass_result, active_registry=_active_registry()
    )
    # Find a finding per synthetic event (any criterion/role; the lag
    # is the same across the (criterion, role) matrix per synthetic).
    by_t: dict[int, SyntheticCalibrationFinding] = {}
    for f in findings:
        by_t.setdefault(f.synthetic_t, f)
    assert by_t[20].recovery_lag_at_synthetic == 3.0
    assert by_t[80].recovery_lag_at_synthetic == 51.0


def test_recovery_lag_cross_reference_skipped_when_statistic_absent() -> None:
    """Without a ``recovery_lag_steps`` result in the pass's
    statistic_results, the finding's ``recovery_lag_at_synthetic`` is
    ``None``."""
    pass_result = _build_pass_result(
        timeline_events=(
            PerturbationEvent(
                t=50,
                wallclock_ms=5000,
                payload={"is_synthetic": True, "is_sham": False},
                is_sham=False,
            ),
        ),
        primary_readings=(
            _reading("reflexive_attention", []),
            _reading("equanimity_perturbation_recovery", []),
        ),
        adversarial_readings=(
            _adversarial_reading("reflexive_attention", []),
            _adversarial_reading(
                "equanimity_perturbation_recovery", []
            ),
        ),
        statistic_results=(),  # no recovery_lag_steps result
    )
    findings = check_synthetic_calibration(
        pass_result, active_registry=_active_registry()
    )
    assert all(f.recovery_lag_at_synthetic is None for f in findings)


def test_synthetic_event_mixed_with_sham_only_synthetic_is_checked() -> None:
    """A timeline containing both a sham (is_sham=True) and a synthetic
    (is_sham=False, is_synthetic=True) emits findings only for the
    synthetic — the sham is the sham-check's domain, not the
    synthetic-check's."""
    pass_result = _build_pass_result(
        timeline_events=(
            PerturbationEvent(
                t=30,
                wallclock_ms=3000,
                payload={"is_sham": True},
                is_sham=True,
            ),
            PerturbationEvent(
                t=50,
                wallclock_ms=5000,
                payload={"is_synthetic": True, "is_sham": False},
                is_sham=False,
            ),
        ),
        primary_readings=(
            _reading("reflexive_attention", []),
            _reading("equanimity_perturbation_recovery", []),
        ),
        adversarial_readings=(
            _adversarial_reading("reflexive_attention", []),
            _adversarial_reading(
                "equanimity_perturbation_recovery", []
            ),
        ),
    )
    findings = check_synthetic_calibration(
        pass_result, active_registry=_active_registry()
    )
    # Only the t=50 synthetic produces findings (2 criteria × 2 roles).
    assert len(findings) == 4
    assert all(f.synthetic_t == 50 for f in findings)


def test_finding_serialization_round_trip() -> None:
    finding = SyntheticCalibrationFinding(
        synthetic_t=50,
        synthetic_wallclock_ms=5000,
        criterion_id="equanimity_perturbation_recovery",
        reader_role="primary",
        admitted=True,
        overlapping_claim_indices=(0, 2),
        recovery_lag_at_synthetic=3.5,
        note="test",
    )
    redumped = SyntheticCalibrationFinding.model_validate_json(
        finding.model_dump_json()
    )
    assert redumped == finding


def test_aggregate_synthetic_findings_zero_admissions() -> None:
    """No admissions across passes → zero totals, empty breakdowns,
    None lags."""
    pass_result = _build_pass_result(
        timeline_events=(
            PerturbationEvent(
                t=50,
                wallclock_ms=5000,
                payload={"is_synthetic": True, "is_sham": False},
                is_sham=False,
            ),
        ),
        primary_readings=(
            _reading("reflexive_attention", [_claim((200, 300))]),
            _reading(
                "equanimity_perturbation_recovery", [_claim((200, 300))]
            ),
        ),
        adversarial_readings=(
            _adversarial_reading(
                "reflexive_attention", [_claim((200, 300))]
            ),
            _adversarial_reading(
                "equanimity_perturbation_recovery", [_claim((200, 300))]
            ),
        ),
    )
    findings = check_synthetic_calibration(
        pass_result, active_registry=_active_registry()
    )
    summary = aggregate_synthetic_findings(((pass_result, findings),))
    assert summary.total_admissions == 0
    assert summary.total_synthetic_events == 4
    assert summary.admissions_by_criterion == {}
    assert summary.admissions_by_checkpoint == {}
    assert summary.admissions_by_role == {}
    assert summary.mean_recovery_lag_at_admissions is None


def test_aggregate_synthetic_findings_with_admissions() -> None:
    """Two passes, both admit at the synthetic event for the equanimity
    primary reading. Aggregated summary reflects 2 admissions broken
    down by criterion / checkpoint / role."""
    # Build a pass that admits.
    admitted_pass = _build_pass_result(
        timeline_events=(
            PerturbationEvent(
                t=50,
                wallclock_ms=5000,
                payload={"is_synthetic": True, "is_sham": False},
                is_sham=False,
            ),
        ),
        primary_readings=(
            _reading("reflexive_attention", [_claim((200, 300))]),
            _reading(
                "equanimity_perturbation_recovery", [_claim((40, 60))]
            ),
        ),
        adversarial_readings=(
            _adversarial_reading(
                "reflexive_attention", [_claim((200, 300))]
            ),
            _adversarial_reading(
                "equanimity_perturbation_recovery", [_claim((200, 300))]
            ),
        ),
        statistic_results=(_recovery_lag_result([3.0]),),
    )
    f_admitted = check_synthetic_calibration(
        admitted_pass, active_registry=_active_registry()
    )
    summary = aggregate_synthetic_findings(
        ((admitted_pass, f_admitted), (admitted_pass, f_admitted))
    )
    # Two passes × 1 admission each = 2 total admissions.
    assert summary.total_admissions == 2
    assert summary.admissions_by_criterion == {
        "equanimity_perturbation_recovery": 2
    }
    assert summary.admissions_by_role == {"primary": 2}
    assert summary.admissions_by_checkpoint == {"ckpt-1": 2}
    # The admissions all had recovery_lag = 3.0.
    assert summary.mean_recovery_lag_at_admissions == 3.0
    # Non-admissions also carry the recovery lag (it's per-event, not
    # per-finding); the mean is the same.
    assert summary.mean_recovery_lag_at_non_admissions == 3.0


def test_summary_serialization_round_trip() -> None:
    summary = SyntheticFindingsSummary(
        total_synthetic_events=10,
        total_admissions=3,
        admissions_by_criterion={"equanimity_perturbation_recovery": 3},
        admissions_by_checkpoint={"ckpt-1": 3},
        admissions_by_role={"primary": 2, "adversarial": 1},
        mean_recovery_lag_at_admissions=4.5,
        mean_recovery_lag_at_non_admissions=51.0,
    )
    redumped = SyntheticFindingsSummary.model_validate_json(
        summary.model_dump_json()
    )
    assert redumped == summary


def test_check_synthetic_calibration_payload_with_is_sham_true_excluded() -> None:
    """A timeline event with both ``is_synthetic=True`` *and*
    ``is_sham=True`` (a programming error upstream — the entry validator
    prevents constructing it via the schedule, but a hand-built timeline
    could) is treated as non-synthetic by the check. The is_sham=True
    filter routes it to the sham check, not the synthetic check."""
    event_payload: dict[str, Any] = {
        "is_synthetic": True,
        "is_sham": True,
    }
    pass_result = _build_pass_result(
        timeline_events=(
            PerturbationEvent(
                t=50,
                wallclock_ms=5000,
                payload=event_payload,
                is_sham=True,
            ),
        ),
        primary_readings=(
            _reading("reflexive_attention", [_claim((40, 60))]),
            _reading(
                "equanimity_perturbation_recovery", [_claim((40, 60))]
            ),
        ),
        adversarial_readings=(
            _adversarial_reading(
                "reflexive_attention", [_claim((40, 60))]
            ),
            _adversarial_reading(
                "equanimity_perturbation_recovery", [_claim((40, 60))]
            ),
        ),
    )
    findings = check_synthetic_calibration(
        pass_result, active_registry=_active_registry()
    )
    assert findings == tuple()
