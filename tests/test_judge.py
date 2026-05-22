"""Phase 9 gate test — :mod:`kind.mirror.judge` data plane.

Pins the structural shape of the five new Phase 9 records:
:class:`ClaimPolarity` (enum), :class:`ClaimPolarityAssignment`,
:class:`FalsifierVerdict` (with the four-way partition invariant),
:class:`CriterionJudgment` (with the falsifier_verdicts /
claim_polarity_assignments cross-criterion guards), and
:class:`RoundJudgment` (with the unique criterion id invariant).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kind.mirror.judge import (
    ClaimPolarity,
    ClaimPolarityAssignment,
    CriterionJudgment,
    FalsifierVerdict,
    RoundJudgment,
)


# ---------------------------------------------------------------------------
# Fixture helpers.
# ---------------------------------------------------------------------------


def _polarity_assignment(
    *,
    pass_index: int = 0,
    criterion_id: str = "reflexive_attention",
    reader_role: str = "primary",
    claim_index: int = 0,
    cited_step_range: tuple[int, int] | None = (10, 20),
    polarity: ClaimPolarity = ClaimPolarity.SUPPORTIVE,
    polarity_rationale: str = "the claim cites a value above the control",
) -> ClaimPolarityAssignment:
    return ClaimPolarityAssignment(
        pass_index=pass_index,
        criterion_id=criterion_id,
        reader_role=reader_role,  # type: ignore[arg-type]
        claim_index=claim_index,
        cited_step_range=cited_step_range,
        polarity=polarity,
        polarity_rationale=polarity_rationale,
    )


def _falsifier_verdict(
    *,
    criterion_id: str = "reflexive_attention",
    falsifier_id: str = "reflexive_attention_v1",
    passes_supporting: tuple[int, ...] = (0, 2),
    passes_refuting: tuple[int, ...] = (1,),
    passes_non_falsifying: tuple[int, ...] = (3,),
    passes_ambiguous: tuple[int, ...] = (4,),
) -> FalsifierVerdict:
    return FalsifierVerdict(
        criterion_id=criterion_id,
        falsifier_id=falsifier_id,
        passes_supporting=passes_supporting,
        passes_refuting=passes_refuting,
        passes_non_falsifying=passes_non_falsifying,
        passes_ambiguous=passes_ambiguous,
    )


# ---------------------------------------------------------------------------
# ClaimPolarity — the four exhaustive members.
# ---------------------------------------------------------------------------


def test_claim_polarity_has_four_values() -> None:
    members = {p.value for p in ClaimPolarity}
    assert members == {
        "supportive",
        "refutational",
        "non_falsifying",
        "ambiguous",
    }


def test_claim_polarity_members_are_str_subclass() -> None:
    """The enum is str-valued so JSON-serialization stays
    string-stable. A future contributor who drops the str base class
    would break the on-disk JSON shape silently — this test trips."""
    for p in ClaimPolarity:
        assert isinstance(p, str)


# ---------------------------------------------------------------------------
# ClaimPolarityAssignment — frozen, validates fields.
# ---------------------------------------------------------------------------


def test_claim_polarity_assignment_is_frozen() -> None:
    ca = _polarity_assignment()
    with pytest.raises(ValidationError):
        ca.pass_index = 1


def test_claim_polarity_assignment_extra_forbidden() -> None:
    with pytest.raises(ValidationError):
        ClaimPolarityAssignment(
            pass_index=0,
            criterion_id="x",
            reader_role="primary",
            claim_index=0,
            cited_step_range=None,
            polarity=ClaimPolarity.AMBIGUOUS,
            polarity_rationale="test",
            unknown_field="oops",  # type: ignore[call-arg]
        )


def test_claim_polarity_assignment_rejects_empty_rationale() -> None:
    with pytest.raises(ValidationError, match="rationale"):
        _polarity_assignment(polarity_rationale="")


def test_claim_polarity_assignment_rejects_negative_indices() -> None:
    with pytest.raises(ValidationError):
        _polarity_assignment(pass_index=-1)
    with pytest.raises(ValidationError):
        _polarity_assignment(claim_index=-1)


def test_claim_polarity_assignment_accepts_optional_cited_step_range() -> None:
    """A claim that scopes to an episode range only (no specific
    steps) has ``cited_step_range=None``; the assignment must accept
    this case."""
    ca = _polarity_assignment(cited_step_range=None)
    assert ca.cited_step_range is None


# ---------------------------------------------------------------------------
# FalsifierVerdict — the four-way partition invariant.
# ---------------------------------------------------------------------------


def test_falsifier_verdict_accepts_disjoint_partition() -> None:
    fv = _falsifier_verdict()
    assert fv.passes_supporting == (0, 2)
    assert fv.passes_refuting == (1,)
    assert fv.passes_non_falsifying == (3,)
    assert fv.passes_ambiguous == (4,)


def test_falsifier_verdict_rejects_overlap_supporting_refuting() -> None:
    with pytest.raises(ValidationError, match="passes_supporting"):
        _falsifier_verdict(
            passes_supporting=(0, 1),
            passes_refuting=(1, 2),
        )


def test_falsifier_verdict_rejects_overlap_any_pair() -> None:
    """Every pair of the four partition tuples must be disjoint —
    the model validator checks all six pairs."""
    # supporting ∩ non_falsifying
    with pytest.raises(ValidationError):
        _falsifier_verdict(
            passes_supporting=(0,), passes_non_falsifying=(0,)
        )
    # refuting ∩ ambiguous
    with pytest.raises(ValidationError):
        _falsifier_verdict(passes_refuting=(2,), passes_ambiguous=(2,))


def test_falsifier_verdict_rejects_unsorted_tuple() -> None:
    with pytest.raises(ValidationError, match="sorted"):
        _falsifier_verdict(passes_supporting=(2, 0))


def test_falsifier_verdict_rejects_duplicate_within_tuple() -> None:
    with pytest.raises(ValidationError, match="unique"):
        _falsifier_verdict(passes_supporting=(0, 0, 1))


def test_falsifier_verdict_rejects_negative_pass_index() -> None:
    with pytest.raises(ValidationError, match=">= 0"):
        _falsifier_verdict(passes_supporting=(-1, 0))


def test_falsifier_verdict_accepts_empty_partition_tuples() -> None:
    """All four tuples can be empty — a criterion the judge could
    not classify at all (every pass ambiguous, or zero passes total)
    is a legal degenerate state."""
    fv = FalsifierVerdict(
        criterion_id="x",
        falsifier_id="x_v1",
        passes_supporting=(),
        passes_refuting=(),
        passes_non_falsifying=(),
        passes_ambiguous=(),
    )
    assert fv.passes_supporting == ()


def test_falsifier_verdict_is_frozen() -> None:
    fv = _falsifier_verdict()
    with pytest.raises(ValidationError):
        fv.criterion_id = "other"


# ---------------------------------------------------------------------------
# CriterionJudgment — confidence range, cross-criterion guards.
# ---------------------------------------------------------------------------


def _criterion_judgment(
    *,
    criterion_id: str = "reflexive_attention",
    framework: str = "buddhist_phenomenology",
    falsifier_verdicts: tuple[FalsifierVerdict, ...] | None = None,
    polarity_assignments: tuple[ClaimPolarityAssignment, ...] | None = None,
    verdict: str = "satisfied",
    confidence: float = 0.85,
    rationale: str = "primary citations are supported across all passes",
) -> CriterionJudgment:
    return CriterionJudgment(
        criterion_id=criterion_id,
        framework=framework,
        falsifier_verdicts=falsifier_verdicts
        if falsifier_verdicts is not None
        else (
            _falsifier_verdict(
                criterion_id=criterion_id,
                passes_supporting=(0,),
                passes_refuting=(),
                passes_non_falsifying=(),
                passes_ambiguous=(),
            ),
        ),
        verdict=verdict,  # type: ignore[arg-type]
        confidence=confidence,
        rationale=rationale,
        claim_polarity_assignments=polarity_assignments
        if polarity_assignments is not None
        else (_polarity_assignment(criterion_id=criterion_id),),
    )


def test_criterion_judgment_accepts_valid_record() -> None:
    cj = _criterion_judgment()
    assert cj.criterion_id == "reflexive_attention"
    assert cj.verdict == "satisfied"
    assert 0.0 <= cj.confidence <= 1.0


def test_criterion_judgment_rejects_confidence_above_one() -> None:
    with pytest.raises(ValidationError, match="confidence"):
        _criterion_judgment(confidence=1.5)


def test_criterion_judgment_rejects_confidence_below_zero() -> None:
    with pytest.raises(ValidationError, match="confidence"):
        _criterion_judgment(confidence=-0.1)


def test_criterion_judgment_rejects_empty_rationale() -> None:
    with pytest.raises(ValidationError, match="rationale"):
        _criterion_judgment(rationale="")


def test_criterion_judgment_rejects_falsifier_verdict_for_wrong_criterion() -> None:
    """A :class:`FalsifierVerdict` whose ``criterion_id`` doesn't
    match the judgment's id is rejected — the judge cannot attribute
    a verdict across criteria."""
    wrong_fv = _falsifier_verdict(
        criterion_id="equanimity_perturbation_recovery",
        passes_supporting=(0,),
        passes_refuting=(),
        passes_non_falsifying=(),
        passes_ambiguous=(),
    )
    with pytest.raises(ValidationError, match="does not match"):
        _criterion_judgment(
            criterion_id="reflexive_attention",
            falsifier_verdicts=(wrong_fv,),
        )


def test_criterion_judgment_rejects_polarity_for_wrong_criterion() -> None:
    """A :class:`ClaimPolarityAssignment` whose ``criterion_id``
    doesn't match the judgment's id is rejected."""
    wrong_pa = _polarity_assignment(criterion_id="other_id")
    with pytest.raises(ValidationError, match="does not match"):
        _criterion_judgment(polarity_assignments=(wrong_pa,))


def test_criterion_judgment_accepts_each_verdict_value() -> None:
    for v in ("satisfied", "not_satisfied", "non_falsifying", "mixed", "ambiguous"):
        cj = _criterion_judgment(verdict=v)
        assert cj.verdict == v


def test_criterion_judgment_is_frozen() -> None:
    cj = _criterion_judgment()
    with pytest.raises(ValidationError):
        cj.confidence = 0.5


# ---------------------------------------------------------------------------
# RoundJudgment — unique criterion-id invariant, serialization round-trip.
# ---------------------------------------------------------------------------


def test_round_judgment_accepts_unique_criterion_judgments() -> None:
    rj = RoundJudgment(
        round_id="r1",
        round_config_summary="passes=5; criteria=[reflexive_attention]",
        criterion_judgments=(
            _criterion_judgment(criterion_id="reflexive_attention"),
            _criterion_judgment(
                criterion_id="equanimity_perturbation_recovery",
                falsifier_verdicts=(
                    _falsifier_verdict(
                        criterion_id="equanimity_perturbation_recovery",
                        falsifier_id="equanimity_perturbation_recovery_v1",
                        passes_supporting=(),
                        passes_refuting=(),
                        passes_non_falsifying=(0,),
                        passes_ambiguous=(),
                    ),
                ),
                polarity_assignments=(
                    _polarity_assignment(
                        criterion_id="equanimity_perturbation_recovery",
                        polarity=ClaimPolarity.NON_FALSIFYING,
                    ),
                ),
                verdict="non_falsifying",
            ),
        ),
        judge_llm_call_records=(),
        wallclock_ms=42,
        notes="test",
    )
    assert len(rj.criterion_judgments) == 2


def test_round_judgment_rejects_duplicate_criterion_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        RoundJudgment(
            round_id="r1",
            round_config_summary="x",
            criterion_judgments=(
                _criterion_judgment(criterion_id="reflexive_attention"),
                _criterion_judgment(criterion_id="reflexive_attention"),
            ),
            judge_llm_call_records=(),
            wallclock_ms=0,
            notes="",
        )


def test_round_judgment_rejects_empty_round_id() -> None:
    with pytest.raises(ValidationError, match="round_id"):
        RoundJudgment(
            round_id="",
            round_config_summary="x",
            criterion_judgments=(),
            judge_llm_call_records=(),
            wallclock_ms=0,
            notes="",
        )


def test_round_judgment_rejects_empty_summary() -> None:
    with pytest.raises(ValidationError, match="summary"):
        RoundJudgment(
            round_id="r1",
            round_config_summary="",
            criterion_judgments=(),
            judge_llm_call_records=(),
            wallclock_ms=0,
            notes="",
        )


def test_round_judgment_round_trips_through_json() -> None:
    """Serialization round-trip: dump → validate equals the original.
    The on-disk JSON form must be self-consistent."""
    rj = RoundJudgment(
        round_id="r1",
        round_config_summary="passes=5",
        criterion_judgments=(_criterion_judgment(),),
        judge_llm_call_records=(),
        wallclock_ms=100,
        notes="round-trip test",
    )
    reloaded = RoundJudgment.model_validate_json(rj.model_dump_json())
    assert reloaded == rj


def test_round_judgment_is_frozen() -> None:
    rj = RoundJudgment(
        round_id="r1",
        round_config_summary="x",
        criterion_judgments=(),
        judge_llm_call_records=(),
        wallclock_ms=0,
        notes="",
    )
    with pytest.raises(ValidationError):
        rj.round_id = "other"
