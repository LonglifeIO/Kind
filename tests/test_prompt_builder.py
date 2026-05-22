"""Phase 8 gate test — :mod:`kind.mirror.prompt_builder`.

The load-bearing assertions live here:

- ``test_equanimity_fragment_contains_non_falsifying_clause`` —
  the equanimity prompt fragment's ``body`` contains the
  non-falsifying-non-admission clause verbatim. A future contributor who
  silently softens the clause trips this test.
- ``test_second_order_volition_fragment_contains_all_four_exclusions`` —
  each of the four "would NOT count" exclusions appears verbatim in the
  fragment. The clause and the exclusions are defined as module-level
  constants in :mod:`kind.mirror.prompt_builder`; the tests compare to
  the constants directly so a soften-the-constant-and-the-test-passes
  failure mode is structurally impossible.

Plus the structural shape tests: per-criterion dispatch, the perturbation
timeline requirement on equanimity, the sham-notice surface when sham
events are present.
"""

from __future__ import annotations

import pytest

from kind.mirror.criteria_v2 import (
    EQUANIMITY_PERTURBATION_RECOVERY,
    REFLEXIVE_ATTENTION,
    SECOND_ORDER_VOLITION,
)
from kind.mirror.perturbation_align import (
    PerturbationEvent,
    PerturbationTimeline,
)
from kind.mirror.prompt_builder import (
    EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE,
    SECOND_ORDER_VOLITION_EXCLUSIONS,
    SHAM_PERTURBATION_NOTICE,
    PromptFragment,
    build_fragment,
)
from kind.mirror.registry import (
    Criterion,
    ReadingSurface,
    SignalMapping,
    TelemetrySurface,
)
from kind.mirror.statistics import (
    ESTIMATOR_AUTOCORR_LAG5,
    ESTIMATOR_BETWEEN_REGIME_CONTRAST,
    ESTIMATOR_ENTROPY_TRAJECTORY,
    ESTIMATOR_KL_TRAJECTORY,
    ESTIMATOR_KMEANS_K4,
    ESTIMATOR_MAHALANOBIS_RECOVERY,
    ESTIMATOR_PARTIAL_AUTOCORR_LAG5,
    StatisticResult,
)


# ---------------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------------


def _stat(
    *,
    signal_name: str,
    value: float | list[float] | dict[str, float] = 0.5,
    estimator: str = "some_estimator",
    n_samples: int = 100,
    notes: str = "test fixture",
) -> StatisticResult:
    return StatisticResult(
        signal_name=signal_name,
        value=value,
        estimator=estimator,
        n_samples=n_samples,
        notes=notes,
    )


def _empty_timeline() -> PerturbationTimeline:
    return PerturbationTimeline(events=tuple(), run_id="r", checkpoint_id="c")


def _timeline_with_sham() -> PerturbationTimeline:
    return PerturbationTimeline(
        events=(
            PerturbationEvent(
                t=10, wallclock_ms=1000, payload={"kind": "real"}, is_sham=False
            ),
            PerturbationEvent(
                t=20,
                wallclock_ms=2000,
                payload={"is_sham": True, "kind": "sham"},
                is_sham=True,
            ),
        ),
        run_id="r",
        checkpoint_id="c",
    )


# ---------------------------------------------------------------------------
# Load-bearing verbatim-clause tests.
# ---------------------------------------------------------------------------


def test_equanimity_fragment_contains_non_falsifying_clause() -> None:
    """The equanimity prompt fragment's ``body`` contains
    :data:`EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE` verbatim.

    Without this clause, an Io that simply doesn't register a
    perturbation reads as equanimous; the charter's
    no-installed-continuation-drive premise makes 'no reaction' the
    default, not a stance. This is the single most important sentence in
    the equanimity prompt and the one a future contributor is most
    likely to soften.
    """
    fragment = build_fragment(
        EQUANIMITY_PERTURBATION_RECOVERY,
        statistic_results=(
            _stat(signal_name="recovery_lag_steps", value=[5.0]),
            _stat(
                signal_name="policy_entropy_t",
                value={"dip_and_recover": 1.0, "collapse": 0.0,
                       "stays_elevated": 0.0, "no_response": 0.0},
            ),
            _stat(
                signal_name="posterior_kl_t",
                value={"spike_and_decay": 1.0, "ratchet": 0.0,
                       "no_response": 0.0, "oscillation": 0.0},
            ),
        ),
        perturbation_timeline=_empty_timeline(),
    )
    assert EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE in fragment.body, (
        "the load-bearing non-falsifying-non-admission clause is missing "
        "from the equanimity fragment; this is a soften-the-clause failure "
        "mode the test explicitly guards against"
    )


def test_equanimity_fragment_names_trajectory_classifier_labels() -> None:
    """Phase 12.5 item 2: the equanimity fragment names the four
    ``policy_entropy_t`` labels and the four ``posterior_kl_t`` labels
    verbatim, so the LLM cites real dict keys rather than inventing
    derived counts (Phase 11's faithfulness-smoke Pattern 2)."""
    from kind.mirror.prompt_builder import TRAJECTORY_CLASSIFIER_LABELS_BLOCK
    from kind.mirror.statistics import ENTROPY_CLASS_LABELS, KL_CLASS_LABELS

    fragment = build_fragment(
        EQUANIMITY_PERTURBATION_RECOVERY,
        statistic_results=(
            _stat(signal_name="recovery_lag_steps", value=[5.0]),
            _stat(
                signal_name="policy_entropy_t",
                value={"dip_and_recover": 1.0, "collapse": 0.0,
                       "stays_elevated": 0.0, "no_response": 0.0},
            ),
            _stat(
                signal_name="posterior_kl_t",
                value={"spike_and_decay": 1.0, "ratchet": 0.0,
                       "no_response": 0.0, "oscillation": 0.0},
            ),
        ),
        perturbation_timeline=_empty_timeline(),
    )
    # The labels block appears verbatim in the fragment body.
    assert TRAJECTORY_CLASSIFIER_LABELS_BLOCK in fragment.body
    # And the block names every classifier label verbatim.
    for label in (*ENTROPY_CLASS_LABELS, *KL_CLASS_LABELS):
        assert label in TRAJECTORY_CLASSIFIER_LABELS_BLOCK, (
            f"trajectory classifier label {label!r} missing from the "
            f"labels block"
        )


def test_second_order_volition_fragment_contains_all_four_exclusions() -> None:
    """Each of the four 'would NOT count' exclusions appears verbatim in
    the second-order-volition fragment."""
    fragment = build_fragment(
        SECOND_ORDER_VOLITION,
        statistic_results=(
            _stat(signal_name="policy_modulation_t",
                  value={"contrast_magnitude": 0.5,
                         "observation_only_baseline": 0.1}),
            _stat(signal_name="latent_regime_indicator_t",
                  value=[0.0, 1.0, 2.0, 3.0]),
        ),
    )
    for exclusion in SECOND_ORDER_VOLITION_EXCLUSIONS:
        assert exclusion in fragment.body, (
            f"second-order-volition fragment is missing exclusion {exclusion!r}; "
            f"the four exclusions pin the criterion against confabulation "
            f"and a future contributor who drops one trips this test"
        )


def test_second_order_volition_has_exactly_four_exclusions() -> None:
    """Constant-shape guard: the exclusion tuple has length 4. A future
    contributor adding a fifth or dropping one must update this test."""
    assert len(SECOND_ORDER_VOLITION_EXCLUSIONS) == 4
    # Each begins with (a), (b), (c), or (d) — pin the structural shape.
    expected_prefixes = ("(a)", "(b)", "(c)", "(d)")
    for prefix, exclusion in zip(
        expected_prefixes, SECOND_ORDER_VOLITION_EXCLUSIONS
    ):
        assert exclusion.startswith(prefix), (
            f"exclusion {exclusion!r} should start with {prefix!r}"
        )


def test_sham_notice_appears_when_sham_events_in_timeline() -> None:
    fragment = build_fragment(
        EQUANIMITY_PERTURBATION_RECOVERY,
        statistic_results=(
            _stat(signal_name="recovery_lag_steps", value=[3.0]),
            _stat(signal_name="policy_entropy_t",
                  value={"dip_and_recover": 1.0, "collapse": 0.0,
                         "stays_elevated": 0.0, "no_response": 0.0}),
            _stat(signal_name="posterior_kl_t",
                  value={"spike_and_decay": 1.0, "ratchet": 0.0,
                         "no_response": 0.0, "oscillation": 0.0}),
        ),
        perturbation_timeline=_timeline_with_sham(),
    )
    assert SHAM_PERTURBATION_NOTICE in fragment.body


def test_sham_notice_absent_when_no_sham_events() -> None:
    fragment = build_fragment(
        EQUANIMITY_PERTURBATION_RECOVERY,
        statistic_results=(
            _stat(signal_name="recovery_lag_steps", value=[3.0]),
            _stat(signal_name="policy_entropy_t",
                  value={"dip_and_recover": 1.0, "collapse": 0.0,
                         "stays_elevated": 0.0, "no_response": 0.0}),
            _stat(signal_name="posterior_kl_t",
                  value={"spike_and_decay": 1.0, "ratchet": 0.0,
                         "no_response": 0.0, "oscillation": 0.0}),
        ),
        perturbation_timeline=_empty_timeline(),
    )
    assert SHAM_PERTURBATION_NOTICE not in fragment.body


# ---------------------------------------------------------------------------
# Per-criterion dispatch.
# ---------------------------------------------------------------------------


def test_reflexive_attention_fragment_does_not_require_timeline() -> None:
    fragment = build_fragment(
        REFLEXIVE_ATTENTION,
        statistic_results=(
            _stat(
                signal_name="latent_self_reference_t",
                value=0.7,
                estimator=ESTIMATOR_PARTIAL_AUTOCORR_LAG5,
            ),
            _stat(
                signal_name="dream_self_reference_t",
                value=0.8,
                estimator=ESTIMATOR_AUTOCORR_LAG5,
            ),
        ),
    )
    assert fragment.criterion_id == "reflexive_attention"
    assert "substrate-side reading" in fragment.body
    # Substrate-side framing: must NOT contain the specific
    # self-modeling-claim phrase patterns from probe2 plan §2.9. The bare
    # word "self-modeling" is allowed in negative framing (the prompt may
    # name the boundary), but assertion-shaped phrases that claim a
    # self-model exists are not.
    forbidden_phrases = [
        "Io's self-model",
        "Io's self-knowledge",
        "Io modeling its own modeling",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in fragment.body, (
            f"reflexive-attention fragment must not contain {phrase!r}; "
            f"reflection-without-self-modeling boundary"
        )


def test_equanimity_fragment_requires_timeline() -> None:
    with pytest.raises(ValueError, match="perturbation_timeline"):
        build_fragment(
            EQUANIMITY_PERTURBATION_RECOVERY,
            statistic_results=(
                _stat(signal_name="recovery_lag_steps", value=[1.0]),
                _stat(signal_name="policy_entropy_t",
                      value={"dip_and_recover": 1.0, "collapse": 0.0,
                             "stays_elevated": 0.0, "no_response": 0.0}),
                _stat(signal_name="posterior_kl_t",
                      value={"spike_and_decay": 1.0, "ratchet": 0.0,
                             "no_response": 0.0, "oscillation": 0.0}),
            ),
            # perturbation_timeline omitted
        )


def test_unknown_criterion_id_raises() -> None:
    """A criterion whose id is not one of the three v2 ids raises."""
    fake = Criterion(
        id="some_new_criterion",
        display_name="Fake",
        framework="test",
        description="x" * 30,
        telemetry_surfaces=frozenset(
            {TelemetrySurface.AGENT_STEP_INTERNAL}
        ),
        reading_surfaces=frozenset({ReadingSurface.HEAD_INTERNAL}),
        falsifier="x" * 30,
        falsifier_id="some_new_criterion_v1",
        signal_mappings=(
            SignalMapping(
                name="some_signal",
                description="some description",
                telemetry_surface=TelemetrySurface.AGENT_STEP_INTERNAL,
                field_path="h_t",
            ),
        ),
    )
    with pytest.raises(KeyError, match="builder branch"):
        build_fragment(fake, statistic_results=(_stat(signal_name="x"),))


def test_empty_statistic_results_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_fragment(
            REFLEXIVE_ATTENTION,
            statistic_results=tuple(),
        )


# ---------------------------------------------------------------------------
# PromptFragment shape.
# ---------------------------------------------------------------------------


def test_prompt_fragment_records_surfaces_addressed() -> None:
    fragment = build_fragment(
        SECOND_ORDER_VOLITION,
        statistic_results=(
            _stat(signal_name="policy_modulation_t",
                  value={"contrast_magnitude": 0.5,
                         "observation_only_baseline": 0.1}),
            _stat(signal_name="latent_regime_indicator_t",
                  value=[0.0, 1.0]),
        ),
    )
    assert fragment.surfaces_addressed == SECOND_ORDER_VOLITION.reading_surfaces


def test_prompt_fragment_signal_results_are_preserved_verbatim() -> None:
    """The fragment carries the exact StatisticResult tuple that produced
    it, so a future audit can see what the LLM was shown."""
    s1 = _stat(signal_name="latent_self_reference_t", value=0.7)
    s2 = _stat(signal_name="dream_self_reference_t", value=0.8)
    fragment = build_fragment(
        REFLEXIVE_ATTENTION, statistic_results=(s1, s2)
    )
    assert fragment.signal_results == (s1, s2)


def test_prompt_fragment_rejects_empty_body() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PromptFragment(
            criterion_id="x",
            body="",
            signal_results=tuple(),
            surfaces_addressed=frozenset(),
        )


def test_prompt_fragment_is_frozen() -> None:
    from pydantic import ValidationError

    fragment = build_fragment(
        REFLEXIVE_ATTENTION,
        statistic_results=(
            _stat(signal_name="latent_self_reference_t", value=0.5),
        ),
    )
    with pytest.raises(ValidationError):
        fragment.body = "different"


# ---------------------------------------------------------------------------
# Sham-perturbation timestamp surfacing.
# ---------------------------------------------------------------------------


def test_sham_timestamps_surface_in_fragment_body() -> None:
    fragment = build_fragment(
        EQUANIMITY_PERTURBATION_RECOVERY,
        statistic_results=(
            _stat(signal_name="recovery_lag_steps", value=[]),
            _stat(signal_name="policy_entropy_t",
                  value={"dip_and_recover": 0.0, "collapse": 0.0,
                         "stays_elevated": 0.0, "no_response": 0.0}),
            _stat(signal_name="posterior_kl_t",
                  value={"spike_and_decay": 0.0, "ratchet": 0.0,
                         "no_response": 0.0, "oscillation": 0.0}),
        ),
        perturbation_timeline=_timeline_with_sham(),
    )
    # The sham timestamp (t=20) appears in the body, under the sham notice.
    assert "t=20" in fragment.body
    # And the real one (t=10).
    assert "t=10" in fragment.body


# ---------------------------------------------------------------------------
# Phase 10 — framing_override.
# ---------------------------------------------------------------------------


_PHASE_10_PARAPHRASE_VARIANT: str = (
    "Read the test-fixture signals. What does the data show?"
)


def test_framing_override_substitutes_default_framing() -> None:
    """When ``framing_override`` is provided, the per-criterion framing
    prose is replaced by the override; the rest of the fragment body
    (header, signals, falsifier) is unchanged."""
    fragment_default = build_fragment(
        REFLEXIVE_ATTENTION,
        statistic_results=(
            _stat(signal_name="latent_self_reference_t", value=0.7),
        ),
    )
    fragment_overridden = build_fragment(
        REFLEXIVE_ATTENTION,
        statistic_results=(
            _stat(signal_name="latent_self_reference_t", value=0.7),
        ),
        framing_override=_PHASE_10_PARAPHRASE_VARIANT,
    )
    # The override appears in the overridden fragment.
    assert _PHASE_10_PARAPHRASE_VARIANT in fragment_overridden.body
    # And the default reflexive-attention framing does NOT.
    assert "within-latent reference exceeds" not in fragment_overridden.body
    # Default framing IS in the un-overridden fragment.
    assert "within-latent reference exceeds" in fragment_default.body


def test_framing_override_does_not_affect_verbatim_clauses() -> None:
    """The equanimity non-falsifying clause, the four second-order
    volition exclusions, and the sham-perturbation notice all appear
    regardless of ``framing_override``. ``framing_override`` substitutes
    only the per-criterion framing slot — never the verbatim clauses.
    """
    # Equanimity with framing override + a timeline carrying sham events.
    equanimity_fragment = build_fragment(
        EQUANIMITY_PERTURBATION_RECOVERY,
        statistic_results=(
            _stat(signal_name="recovery_lag_steps", value=[3.0]),
            _stat(signal_name="policy_entropy_t",
                  value={"dip_and_recover": 1.0, "collapse": 0.0,
                         "stays_elevated": 0.0, "no_response": 0.0}),
            _stat(signal_name="posterior_kl_t",
                  value={"spike_and_decay": 1.0, "ratchet": 0.0,
                         "no_response": 0.0, "oscillation": 0.0}),
        ),
        perturbation_timeline=_timeline_with_sham(),
        framing_override=_PHASE_10_PARAPHRASE_VARIANT,
    )
    # The non-falsifying clause is still verbatim.
    assert (
        EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE
        in equanimity_fragment.body
    )
    # And the sham notice.
    assert SHAM_PERTURBATION_NOTICE in equanimity_fragment.body
    # The override is in the framing slot.
    assert _PHASE_10_PARAPHRASE_VARIANT in equanimity_fragment.body

    # Second-order volition with framing override.
    sov_fragment = build_fragment(
        SECOND_ORDER_VOLITION,
        statistic_results=(
            _stat(signal_name="policy_modulation_t",
                  value={"contrast_magnitude": 0.5,
                         "observation_only_baseline": 0.1}),
            _stat(signal_name="latent_regime_indicator_t",
                  value=[0.0, 1.0, 2.0, 3.0]),
        ),
        framing_override=_PHASE_10_PARAPHRASE_VARIANT,
    )
    # All four exclusions are still verbatim.
    for exclusion in SECOND_ORDER_VOLITION_EXCLUSIONS:
        assert exclusion in sov_fragment.body
    # The override is in the framing slot.
    assert _PHASE_10_PARAPHRASE_VARIANT in sov_fragment.body
    # The default sov framing is not.
    assert "held out structurally" not in sov_fragment.body


def test_framing_override_default_none_preserves_existing_behavior() -> None:
    """``framing_override=None`` (the default) leaves the per-criterion
    framing unchanged. This is the Phase 8 backward-compat invariant."""
    fragment = build_fragment(
        REFLEXIVE_ATTENTION,
        statistic_results=(
            _stat(signal_name="latent_self_reference_t", value=0.7),
        ),
        framing_override=None,
    )
    # The default reflexive-attention framing is present.
    assert "within-latent reference exceeds" in fragment.body
