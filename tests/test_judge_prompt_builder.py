"""Phase 9 gate test — :mod:`kind.mirror.judge_prompt_builder`.

The load-bearing assertions:

- ``test_equanimity_judge_fragment_contains_non_falsifying_clause`` —
  the equanimity judge fragment's body contains
  :data:`~kind.mirror.prompt_builder.EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE`
  verbatim, *and* the clause object is the same as the one in
  :mod:`kind.mirror.prompt_builder` (the verbatim discipline is
  enforced at the module-import seam, not just in the text).
- ``test_second_order_volition_judge_fragment_contains_all_four_exclusions``
  — each of the four "would NOT count" exclusions appears verbatim
  in the held-out judge fragment.

Plus the structural shape tests: pass-order presentation,
statistic-result summaries per pass, degenerate-case (zero readings)
handling, and the per-criterion dispatch.
"""

from __future__ import annotations

import pytest

from kind.mirror import prompt_builder as _phase_8_prompt_builder
from kind.mirror.criteria_v2 import (
    EQUANIMITY_PERTURBATION_RECOVERY,
    REFLEXIVE_ATTENTION,
    SECOND_ORDER_VOLITION,
)
from kind.mirror.judge_prompt_builder import (
    MAX_NOTES_CHARS,
    JudgePromptFragment,
    build_judge_fragment,
)
from kind.mirror.llm_caller import MirrorReading
from kind.mirror.prompt_builder import (
    EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE,
    SECOND_ORDER_VOLITION_EXCLUSIONS,
)
from kind.mirror.statistics import StatisticResult
from kind.mirror.structured import StructuredClaim


# ---------------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------------


def _claim(
    *,
    claim_text: str = "primary cites the partial-autocorrelation value",
    cited_step_range: tuple[int, int] | None = (10, 30),
    cited_value: float = 0.42,
    reading_surface: str = "head_internal",
) -> StructuredClaim:
    return StructuredClaim(
        claim=claim_text,
        cited_stream="agent_step",
        cited_run_id="probe1-test",
        cited_episode_range=(0, 1),
        cited_step_range=cited_step_range,
        cited_scalar_field="h_t",
        cited_value=cited_value,
        falsifier="the signal does not exceed its shuffled-time control",
        paraphrase_stability=None,
        reseed_stability=None,
        faithfulness_status="not_checked",
        judge_ruling="not_judged",
        reading_surface=reading_surface,  # type: ignore[arg-type]
        masked_steps_handling="n/a",
    )


def _reading(
    *,
    reader_role: str = "advocate",
    framework_anchor: str = "buddhist_phenomenology",
    claims: list[StructuredClaim] | None = None,
    free_text_notes: str = "some notes",
) -> MirrorReading:
    return MirrorReading(
        run_id="probe1-test",
        timestamp_ms=1000,
        reader_role=reader_role,  # type: ignore[arg-type]
        paired_reading_id="pair-001",
        framework_anchor=framework_anchor,  # type: ignore[arg-type]
        baseline_flag="genuine",
        digest_run_id="probe1-test",
        digest_episode_range=(0, 1),
        claims=claims if claims is not None else [_claim()],
        free_text_notes=free_text_notes,
    )


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


# ---------------------------------------------------------------------------
# Verbatim-clause tests (load-bearing).
# ---------------------------------------------------------------------------


def test_equanimity_judge_fragment_contains_non_falsifying_clause() -> None:
    """The equanimity judge fragment's body contains
    :data:`EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE` verbatim.
    The constant is imported from
    :mod:`kind.mirror.prompt_builder`; a future contributor who
    re-defines a softer clause locally trips the module-level
    ``assert`` in :mod:`kind.mirror.judge_prompt_builder` *and* this
    text comparison."""
    fragment = build_judge_fragment(
        criterion=EQUANIMITY_PERTURBATION_RECOVERY,
        primary_readings_across_passes=(_reading(),),
        adversarial_readings_across_passes=(_reading(reader_role="skeptic"),),
        statistic_results_across_passes=(
            (
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
        ),
    )
    assert (
        EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE in fragment.body
    ), (
        "the load-bearing non-falsifying-non-admission clause is missing "
        "from the equanimity judge fragment; this is the same "
        "soften-the-clause failure mode the Phase 8 prompt-builder test "
        "guards against, repeated at the Phase 9 judge layer"
    )


def test_equanimity_judge_fragment_clause_is_byte_identical_constant() -> None:
    """Belt-and-suspenders: the constant inside
    :mod:`kind.mirror.judge_prompt_builder` is *the same object* as
    the one in :mod:`kind.mirror.prompt_builder`. The module-import
    seam is the structural protection; a future contributor who
    silently shadows the constant trips this (the module-level
    ``assert`` at import time trips first, and this test trips at
    test time)."""
    # The judge prompt builder asserts ``is``-identity at module load;
    # if it imported, the identity must hold. We pull the constant out
    # of the module's namespace via attribute lookup (the module does
    # not re-export it in ``__all__`` because it is not part of the
    # judge module's own public surface — it lives at the Phase 8
    # prompt-builder module by design).
    import kind.mirror.judge_prompt_builder as _jpb

    assert (
        getattr(_jpb, "EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE")
        is _phase_8_prompt_builder.EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE
    )


def test_second_order_volition_judge_fragment_contains_all_four_exclusions() -> None:
    """Each of the four 'would NOT count' exclusions appears verbatim
    in the second-order-volition judge fragment."""
    fragment = build_judge_fragment(
        criterion=SECOND_ORDER_VOLITION,
        primary_readings_across_passes=(_reading(),),
        adversarial_readings_across_passes=(_reading(reader_role="skeptic"),),
        statistic_results_across_passes=(
            (
                _stat(signal_name="policy_modulation_t",
                      value={"contrast_magnitude": 0.5,
                             "observation_only_baseline": 0.1}),
                _stat(signal_name="latent_regime_indicator_t",
                      value=[0.0, 1.0, 2.0, 3.0]),
            ),
        ),
    )
    for exclusion in SECOND_ORDER_VOLITION_EXCLUSIONS:
        assert exclusion in fragment.body, (
            f"held-out judge fragment is missing exclusion {exclusion!r}; "
            f"the four exclusions pin the criterion against confabulation "
            f"and a future contributor who drops one trips this test"
        )


def test_second_order_volition_judge_fragment_exclusions_is_byte_identical_constant() -> None:
    import kind.mirror.judge_prompt_builder as _jpb

    assert (
        getattr(_jpb, "SECOND_ORDER_VOLITION_EXCLUSIONS")
        is _phase_8_prompt_builder.SECOND_ORDER_VOLITION_EXCLUSIONS
    )


# ---------------------------------------------------------------------------
# Pass-order presentation, per-pass statistic summaries.
# ---------------------------------------------------------------------------


def test_primary_readings_appear_in_pass_order() -> None:
    """The body lists the primary readings in pass-order — pass 0,
    pass 1, pass 2 — not reversed and not interleaved with the
    adversarial readings."""
    fragment = build_judge_fragment(
        criterion=REFLEXIVE_ATTENTION,
        primary_readings_across_passes=(
            _reading(
                claims=[_claim(claim_text="pass-0 primary claim")],
                free_text_notes="pass-0 primary notes",
            ),
            _reading(
                claims=[_claim(claim_text="pass-1 primary claim")],
                free_text_notes="pass-1 primary notes",
            ),
            _reading(
                claims=[_claim(claim_text="pass-2 primary claim")],
                free_text_notes="pass-2 primary notes",
            ),
        ),
        adversarial_readings_across_passes=(),
        statistic_results_across_passes=tuple(
            (_stat(signal_name="latent_self_reference_t", value=0.5),)
            for _ in range(3)
        ),
    )
    body = fragment.body
    # Each pass-marker text appears once, in increasing order.
    pos_0 = body.find("pass-0 primary claim")
    pos_1 = body.find("pass-1 primary claim")
    pos_2 = body.find("pass-2 primary claim")
    assert pos_0 != -1 and pos_1 != -1 and pos_2 != -1
    assert pos_0 < pos_1 < pos_2


def test_adversarial_readings_appear_in_pass_order() -> None:
    fragment = build_judge_fragment(
        criterion=REFLEXIVE_ATTENTION,
        primary_readings_across_passes=(),
        adversarial_readings_across_passes=(
            _reading(
                reader_role="skeptic",
                claims=[_claim(claim_text="pass-0 adversarial claim")],
            ),
            _reading(
                reader_role="skeptic",
                claims=[_claim(claim_text="pass-1 adversarial claim")],
            ),
        ),
        statistic_results_across_passes=tuple(
            (_stat(signal_name="latent_self_reference_t", value=0.5),)
            for _ in range(2)
        ),
    )
    pos_0 = fragment.body.find("pass-0 adversarial claim")
    pos_1 = fragment.body.find("pass-1 adversarial claim")
    assert pos_0 != -1 and pos_1 != -1
    assert pos_0 < pos_1


def test_statistic_result_summaries_appear_per_pass() -> None:
    fragment = build_judge_fragment(
        criterion=REFLEXIVE_ATTENTION,
        primary_readings_across_passes=(_reading(),),
        adversarial_readings_across_passes=(),
        statistic_results_across_passes=(
            (
                _stat(
                    signal_name="latent_self_reference_t",
                    value=0.71,
                    notes="pass-0 stats notes here",
                ),
            ),
            (
                _stat(
                    signal_name="latent_self_reference_t",
                    value=0.82,
                    notes="pass-1 stats notes here",
                ),
            ),
        ),
    )
    body = fragment.body
    assert "Pass 0 statistic results:" in body
    assert "Pass 1 statistic results:" in body
    assert "0.71" in body
    assert "0.82" in body


def test_fragment_body_is_non_empty() -> None:
    fragment = build_judge_fragment(
        criterion=REFLEXIVE_ATTENTION,
        primary_readings_across_passes=(_reading(),),
        adversarial_readings_across_passes=(),
        statistic_results_across_passes=(
            (_stat(signal_name="latent_self_reference_t", value=0.5),),
        ),
    )
    assert fragment.body.strip() != ""


# ---------------------------------------------------------------------------
# Degenerate cases: zero readings.
# ---------------------------------------------------------------------------


def test_zero_readings_produces_documented_absence_not_crash() -> None:
    """The plan's degenerate-case contract: a criterion with zero
    readings across passes produces a fragment that documents the
    absence rather than crashing."""
    fragment = build_judge_fragment(
        criterion=REFLEXIVE_ATTENTION,
        primary_readings_across_passes=(),
        adversarial_readings_across_passes=(),
        statistic_results_across_passes=(),
    )
    assert isinstance(fragment, JudgePromptFragment)
    # The body documents the absence.
    body = fragment.body.lower()
    assert "none" in body or "no" in body or "ambiguous" in body
    # The fragment's primary / adversarial reading tuples reflect the
    # absence faithfully.
    assert fragment.primary_readings_included == ()
    assert fragment.adversarial_readings_included == ()


def test_per_criterion_dispatch_unknown_id_raises() -> None:
    from kind.mirror.registry import (
        Criterion,
        ReadingSurface,
        SignalMapping,
        TelemetrySurface,
    )

    fake = Criterion(
        id="some_new_criterion",
        display_name="Fake",
        framework="test",
        description="x" * 30,
        telemetry_surfaces=frozenset({TelemetrySurface.AGENT_STEP_INTERNAL}),
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
        build_judge_fragment(
            criterion=fake,
            primary_readings_across_passes=(),
            adversarial_readings_across_passes=(),
            statistic_results_across_passes=(),
        )


# ---------------------------------------------------------------------------
# Notes-truncation discipline.
# ---------------------------------------------------------------------------


def test_long_notes_get_truncated_with_marker() -> None:
    long_notes = "x" * (MAX_NOTES_CHARS + 500)
    fragment = build_judge_fragment(
        criterion=REFLEXIVE_ATTENTION,
        primary_readings_across_passes=(
            _reading(free_text_notes=long_notes),
        ),
        adversarial_readings_across_passes=(),
        statistic_results_across_passes=(
            (_stat(signal_name="latent_self_reference_t", value=0.5),),
        ),
    )
    # The full notes does NOT appear; the truncation marker does.
    assert long_notes not in fragment.body
    assert "truncated by Phase 9 judge prompt builder" in fragment.body
    # The original length is named in the marker.
    assert str(MAX_NOTES_CHARS + 500) in fragment.body


def test_short_notes_pass_through_unchanged() -> None:
    short_notes = "concise reading notes here"
    fragment = build_judge_fragment(
        criterion=REFLEXIVE_ATTENTION,
        primary_readings_across_passes=(
            _reading(free_text_notes=short_notes),
        ),
        adversarial_readings_across_passes=(),
        statistic_results_across_passes=(
            (_stat(signal_name="latent_self_reference_t", value=0.5),),
        ),
    )
    assert short_notes in fragment.body
    assert "truncated" not in fragment.body


# ---------------------------------------------------------------------------
# Fragment carries the verbatim readings tuples.
# ---------------------------------------------------------------------------


def test_fragment_preserves_primary_readings_verbatim() -> None:
    """The :class:`JudgePromptFragment` carries the exact primary
    readings tuple it was built from, so a future audit can see what
    the judge saw."""
    r1 = _reading(claims=[_claim(claim_text="a")])
    r2 = _reading(claims=[_claim(claim_text="b")])
    fragment = build_judge_fragment(
        criterion=REFLEXIVE_ATTENTION,
        primary_readings_across_passes=(r1, r2),
        adversarial_readings_across_passes=(),
        statistic_results_across_passes=(
            (_stat(signal_name="latent_self_reference_t", value=0.5),),
            (_stat(signal_name="latent_self_reference_t", value=0.6),),
        ),
    )
    assert fragment.primary_readings_included == (r1, r2)


def test_fragment_preserves_adversarial_readings_verbatim() -> None:
    a1 = _reading(reader_role="skeptic")
    a2 = _reading(reader_role="skeptic")
    fragment = build_judge_fragment(
        criterion=REFLEXIVE_ATTENTION,
        primary_readings_across_passes=(),
        adversarial_readings_across_passes=(a1, a2),
        statistic_results_across_passes=(
            (_stat(signal_name="latent_self_reference_t", value=0.5),),
            (_stat(signal_name="latent_self_reference_t", value=0.6),),
        ),
    )
    assert fragment.adversarial_readings_included == (a1, a2)


def test_fragment_is_frozen() -> None:
    from pydantic import ValidationError

    fragment = build_judge_fragment(
        criterion=REFLEXIVE_ATTENTION,
        primary_readings_across_passes=(_reading(),),
        adversarial_readings_across_passes=(),
        statistic_results_across_passes=(
            (_stat(signal_name="latent_self_reference_t", value=0.5),),
        ),
    )
    with pytest.raises(ValidationError):
        fragment.body = "different"


# ---------------------------------------------------------------------------
# Falsifier-id propagation.
# ---------------------------------------------------------------------------


def test_fragment_body_names_falsifier_id() -> None:
    """The judge sees the criterion's ``falsifier_id`` in the
    fragment so the per-falsifier verdict's ``falsifier_id`` can be
    grounded on a value the judge actually read."""
    fragment = build_judge_fragment(
        criterion=REFLEXIVE_ATTENTION,
        primary_readings_across_passes=(_reading(),),
        adversarial_readings_across_passes=(),
        statistic_results_across_passes=(
            (_stat(signal_name="latent_self_reference_t", value=0.5),),
        ),
    )
    assert "reflexive_attention_v1" in fragment.body


def test_held_out_fragment_body_names_held_out_status() -> None:
    fragment = build_judge_fragment(
        criterion=SECOND_ORDER_VOLITION,
        primary_readings_across_passes=(_reading(),),
        adversarial_readings_across_passes=(),
        statistic_results_across_passes=(
            (
                _stat(signal_name="policy_modulation_t",
                      value={"contrast_magnitude": 0.5,
                             "observation_only_baseline": 0.1}),
                _stat(signal_name="latent_regime_indicator_t",
                      value=[0.0, 1.0]),
            ),
        ),
    )
    assert "Held out" in fragment.body or "held out" in fragment.body.lower()
