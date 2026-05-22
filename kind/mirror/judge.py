"""Phase 9 judge data plane — :class:`ClaimPolarity`,
:class:`ClaimPolarityAssignment`, :class:`FalsifierVerdict`,
:class:`CriterionJudgment`, :class:`RoundJudgment`.

Phase 9 builds the third interpretive role in the mirror's adversarial
structure. Phases 6–8 produced primary and adversarial readings against
frozen criteria; Phase 12 verified the calibration discipline in the
empty-baseline case; Phase 13 verified it in the discriminative case
and produced the multi-pass record the judge consumes.

The judge reads :class:`~kind.mirror.calibration.round.RoundResult`
artifacts. For each criterion, across all passes in the round, it
aggregates the primary and adversarial readings at the *falsifier*
level (not the claim level — Phase 13's stance-drift findings rule out
claim-level aggregation), assigns claim polarity to each citation
(supportive / refutational / non-falsifying / ambiguous), and produces
a structured judgment with a confidence score and a free-text
explanation.

The judge is Phase 9's third LLM role. Phases 6–8 committed two:
primary (the Phenomenological Advocate) and adversarial (the
Statistical Skeptic). The judge is a separate call with its own system
prompt — the Methodological Arbiter — and its own structured output
schema. The judge does not produce findings about Io; it produces
findings about the readings.

**Load-bearing constraints.**

*The mirror is one-way.* Inherited from all prior phases. The judge
writes only under ``runs/{run_id}/mirror/judgments/``. No new write
surface on Io's side.

*The judge reads readings, not telemetry.* The judge does not recompute
statistics; it does not call :func:`~kind.mirror.statistics.compute_statistic`;
it does not load :class:`~kind.observer.schemas.AgentStep` files. It
reads the :class:`~kind.mirror.calibration.round.RoundResult` artifact
and operates on the :class:`~kind.mirror.llm_caller.MirrorReading`
records and the :class:`~kind.mirror.statistics.StatisticResult`
records the orchestrator already wrote.

*Phase 7's criteria are frozen.* The judge can disagree with the
primary, the adversarial, or both. It cannot change the criterion's
prose, the falsifier, the signal mappings, or any committed shape. If
the judge consistently finds the criterion's frozen falsifier
insufficient, that's a finding to journal — and an input to a future
external-framework-prompted Phase 7 amendment — not a license to
amend.

*Claim polarity is judge-local.* Phase 9 assigns polarity as part of
the judge's output, not as a field on
:class:`~kind.mirror.structured.StructuredClaim`. The polarity enum
lives in *this* module; the existing :class:`StructuredClaim` model is
unchanged. If Phase 13's evidence that polarity is unambiguous on
inspection holds up across Phase 9's larger sample, a future Phase 14+
may justify amending :class:`StructuredClaim` with a structured
polarity field. Phase 9 produces that empirical case.

Out of scope here: the judge prompt builder
(:mod:`kind.mirror.judge_prompt_builder`); the LLM caller
(:mod:`kind.mirror.judge_llm_caller`); the driver
(:mod:`kind.mirror.judge_driver`); the Phase 9 smoke harness.
"""

from __future__ import annotations

from enum import Enum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from kind.mirror.calibration.llm_audit import LLMCallRecord
from kind.mirror.llm_caller import PassRole

__all__ = [
    "ClaimPolarity",
    "ClaimPolarityAssignment",
    "FalsifierVerdict",
    "CriterionJudgment",
    "RoundJudgment",
    "Verdict",
]


# ---------------------------------------------------------------------------
# Per-claim polarity enum.
# ---------------------------------------------------------------------------


class ClaimPolarity(str, Enum):
    """The judge's per-claim polarity classification.

    Frozen (str-valued enum members). The four values exhaustively
    partition the space of citation-shaped claims at this judge layer:

    - :attr:`SUPPORTIVE`: the claim affirms the criterion is satisfied
      at the cited evidence.
    - :attr:`REFUTATIONAL`: the claim explicitly argues against the
      criterion's satisfaction at the cited evidence.
    - :attr:`NON_FALSIFYING`: the claim cites evidence but invokes a
      non-falsifying-non-admission clause — the data couldn't evaluate
      the criterion either way. The equanimity criterion has this
      clause baked into its prose (no detectable response is not an
      admission of equanimity); reflexive attention and second-order
      volition lack an explicit clause but can still be flagged here
      when a claim cites data but explicitly disclaims a verdict.
    - :attr:`AMBIGUOUS`: the claim cites evidence but the polarity
      isn't clear from the claim text. Flagged for review. Phase 9's
      journal entry counts ambiguous classifications as the
      cost-of-LLM-parsing signal: many ambiguous → the structured
      polarity field on :class:`StructuredClaim` earns its way for
      Phase 14+.
    """

    SUPPORTIVE = "supportive"
    REFUTATIONAL = "refutational"
    NON_FALSIFYING = "non_falsifying"
    AMBIGUOUS = "ambiguous"


# Verdict literal for criterion-level judgments. ``mixed`` carries the
# case where some passes admit and some refute (Phase 13's
# stance-drift case); ``ambiguous`` is the structural fallback when
# the judge cannot tell from the evidence either way.
Verdict = Literal[
    "satisfied",
    "not_satisfied",
    "non_falsifying",
    "mixed",
    "ambiguous",
]


_VERDICT_VALUES: Final[frozenset[str]] = frozenset(
    {"satisfied", "not_satisfied", "non_falsifying", "mixed", "ambiguous"}
)


# ---------------------------------------------------------------------------
# Per-claim polarity assignment record.
# ---------------------------------------------------------------------------


class ClaimPolarityAssignment(BaseModel):
    """The judge's polarity verdict on a single claim within a reading.

    Frozen, ``extra="forbid"``. Produced by the judge for every claim
    across every primary and adversarial reading the criterion's
    judgment is over. The ``cited_step_range`` is copied from the
    underlying claim for the judge's later reference; the
    ``polarity_rationale`` is the judge's free-text explanation drawn
    from the claim text.

    Fields:

    - ``pass_index``: 0-based pass index within the round
      (``0 .. passes_per_checkpoint - 1`` per checkpoint).
    - ``criterion_id``: which criterion this claim was made about.
    - ``reader_role``: ``"primary"`` or ``"adversarial"``
      (:data:`~kind.mirror.llm_caller.PassRole`). The judge sees both
      roles; the polarity assignment names which one.
    - ``claim_index``: 0-based position in the reading's ``claims``
      tuple. With ``pass_index`` + ``reader_role`` + ``criterion_id``
      this fully identifies the claim.
    - ``cited_step_range``: copied from the underlying
      :class:`~kind.mirror.structured.StructuredClaim.cited_step_range`
      (``None`` for claims that scoped to an episode range only). The
      copy lets the judge's downstream consumer correlate polarity
      assignments to perturbation timestamps without re-loading the
      reading.
    - ``polarity``: one of :class:`ClaimPolarity`.
    - ``polarity_rationale``: non-empty; the judge's explanation for
      the polarity assignment. The judge cites the claim text it
      classified against in this field.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    pass_index: int
    criterion_id: str
    reader_role: PassRole
    claim_index: int
    cited_step_range: tuple[int, int] | None
    polarity: ClaimPolarity
    polarity_rationale: str

    @field_validator("pass_index", "claim_index")
    @classmethod
    def _validate_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"index must be >= 0; got {value}.")
        return value

    @field_validator("criterion_id")
    @classmethod
    def _validate_criterion_id_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("criterion_id must be non-empty.")
        return value

    @field_validator("polarity_rationale")
    @classmethod
    def _validate_rationale_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "ClaimPolarityAssignment.polarity_rationale must be "
                "non-empty: the judge's rationale is the load-bearing "
                "audit trail for the polarity classification. An empty "
                "rationale defeats the structured-grounding mandate."
            )
        return value


# ---------------------------------------------------------------------------
# Per-falsifier verdict record.
# ---------------------------------------------------------------------------


class FalsifierVerdict(BaseModel):
    """Per-falsifier roll-up of pass-level admissions / refutations /
    non-falsifying outcomes across one criterion in one round.

    Frozen, ``extra="forbid"``. Phase 9 commits one
    :class:`FalsifierVerdict` per falsifier on the criterion. Each of
    the three v2 criteria commits a single falsifier (the
    :attr:`~kind.mirror.registry.Criterion.falsifier_id` value); the
    tuple shape on :class:`CriterionJudgment` supports future criteria
    with multi-falsifier structures.

    The four partition tuples cover the pass-index space exhaustively:
    every pass that the criterion's judgment is over appears in
    exactly one of (supporting, refuting, non-falsifying, ambiguous).
    A validator enforces the partition.

    Fields:

    - ``criterion_id``: which criterion this verdict is for.
    - ``falsifier_id``: the
      :attr:`~kind.mirror.registry.Criterion.falsifier_id` value
      (snake_case; the criterion's stable falsifier identifier).
    - ``passes_supporting``: sorted, unique pass indices whose primary
      readings produced supportive citations.
    - ``passes_refuting``: sorted, unique pass indices whose primary
      readings produced refutational citations OR whose adversarial
      readings produced supportive refutations. The OR is structural:
      the adversarial role's job is to argue against the criterion, so
      an adversarial-supportive citation is a primary-refutational
      citation from the verdict's perspective.
    - ``passes_non_falsifying``: sorted, unique pass indices whose
      primary readings invoked the non-falsifying clause.
    - ``passes_ambiguous``: sorted, unique pass indices the judge
      could not classify cleanly.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion_id: str
    falsifier_id: str
    passes_supporting: tuple[int, ...]
    passes_refuting: tuple[int, ...]
    passes_non_falsifying: tuple[int, ...]
    passes_ambiguous: tuple[int, ...]

    @field_validator("criterion_id", "falsifier_id")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty.")
        return value

    @field_validator(
        "passes_supporting",
        "passes_refuting",
        "passes_non_falsifying",
        "passes_ambiguous",
    )
    @classmethod
    def _validate_sorted_unique_non_negative(
        cls, value: tuple[int, ...]
    ) -> tuple[int, ...]:
        for v in value:
            if v < 0:
                raise ValueError(
                    f"pass index must be >= 0; got {v} in tuple {value}."
                )
        if list(value) != sorted(value):
            raise ValueError(
                f"pass-index tuple must be sorted ascending; got {value}."
            )
        if len(set(value)) != len(value):
            raise ValueError(
                f"pass-index tuple must have unique values; got {value} "
                f"with duplicates."
            )
        return value

    @model_validator(mode="after")
    def _validate_partition_disjoint(self) -> FalsifierVerdict:
        """The four tuples partition the pass-index space — every
        pass appears in exactly one of supporting / refuting /
        non-falsifying / ambiguous. A pass that fell into more than
        one bucket would mean the judge double-counted; a pass that
        fell into none would mean the judge silently dropped it. Both
        are calibration failures the journal entry should surface, so
        we enforce structurally."""
        all_sets = (
            set(self.passes_supporting),
            set(self.passes_refuting),
            set(self.passes_non_falsifying),
            set(self.passes_ambiguous),
        )
        # Pairwise-disjointness.
        pairs = (
            ("passes_supporting", "passes_refuting", all_sets[0], all_sets[1]),
            ("passes_supporting", "passes_non_falsifying", all_sets[0], all_sets[2]),
            ("passes_supporting", "passes_ambiguous", all_sets[0], all_sets[3]),
            ("passes_refuting", "passes_non_falsifying", all_sets[1], all_sets[2]),
            ("passes_refuting", "passes_ambiguous", all_sets[1], all_sets[3]),
            ("passes_non_falsifying", "passes_ambiguous", all_sets[2], all_sets[3]),
        )
        for name_a, name_b, set_a, set_b in pairs:
            overlap = set_a & set_b
            if overlap:
                raise ValueError(
                    f"FalsifierVerdict for {self.criterion_id!r}: pass "
                    f"indices {sorted(overlap)} appear in both "
                    f"{name_a} and {name_b}. The four partition tuples "
                    f"must be pairwise disjoint — a pass cannot be both "
                    f"e.g. supportive and refuting in the same verdict. "
                    f"The judge double-counted; this is a calibration "
                    f"failure to surface, not absorb."
                )
        return self


# ---------------------------------------------------------------------------
# Per-criterion judgment record.
# ---------------------------------------------------------------------------


class CriterionJudgment(BaseModel):
    """The judge's structured verdict on one criterion across the round.

    Frozen, ``extra="forbid"``. The ``verdict`` and ``confidence`` are
    the load-bearing output; the ``rationale`` is for human
    inspection; the ``claim_polarity_assignments`` carry the per-claim
    audit trail.

    Fields:

    - ``criterion_id``: the criterion this judgment is for.
    - ``framework``: copied from the criterion's
      :attr:`~kind.mirror.registry.Criterion.framework`. The judge
      sees the framework via the prompt fragment; preserving it on
      the judgment makes cross-framework analyses easier without
      re-loading the registry.
    - ``falsifier_verdicts``: one
      :class:`FalsifierVerdict` per falsifier on the criterion.
      Phase 7's criteria each commit one falsifier, so the tuple is
      length-1 for the three v2 criteria; the tuple shape supports
      future multi-falsifier structures.
    - ``verdict``: one of the five :data:`Verdict` values.
    - ``confidence``: the judge's confidence in the verdict in
      ``[0.0, 1.0]``. The journal entry inspects the confidence
      distribution across the six Phase 9 judgments (2 rounds × 3
      criteria).
    - ``rationale``: non-empty; the judge's free-text explanation.
    - ``claim_polarity_assignments``: every claim across all passes
      this judgment is over, with its assigned polarity.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion_id: str
    framework: str
    falsifier_verdicts: tuple[FalsifierVerdict, ...]
    verdict: Verdict
    confidence: float
    rationale: str
    claim_polarity_assignments: tuple[ClaimPolarityAssignment, ...]

    @field_validator("criterion_id", "framework")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty.")
        return value

    @field_validator("confidence")
    @classmethod
    def _validate_confidence_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"CriterionJudgment.confidence must be in [0.0, 1.0]; "
                f"got {value}."
            )
        return value

    @field_validator("rationale")
    @classmethod
    def _validate_rationale_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "CriterionJudgment.rationale must be non-empty: the "
                "judge's free-text explanation is what the human reader "
                "is for. A verdict without a rationale is opaque."
            )
        return value

    @field_validator("verdict")
    @classmethod
    def _validate_verdict_value(cls, value: str) -> str:
        # Pydantic's Literal validator already handles this; the
        # explicit check gives a friendlier error for typos.
        if value not in _VERDICT_VALUES:
            raise ValueError(
                f"verdict must be one of {sorted(_VERDICT_VALUES)}; "
                f"got {value!r}."
            )
        return value

    @model_validator(mode="after")
    def _validate_falsifier_verdicts_match_criterion(self) -> CriterionJudgment:
        """Every :class:`FalsifierVerdict` in the tuple references this
        criterion's ``criterion_id``. The judge cannot attach a
        falsifier verdict to one criterion under another's
        :class:`CriterionJudgment`."""
        for index, fv in enumerate(self.falsifier_verdicts):
            if fv.criterion_id != self.criterion_id:
                raise ValueError(
                    f"CriterionJudgment for {self.criterion_id!r}: "
                    f"falsifier_verdicts[{index}].criterion_id is "
                    f"{fv.criterion_id!r}, which does not match. The "
                    f"judge cannot attribute a falsifier verdict across "
                    f"criteria."
                )
        return self

    @model_validator(mode="after")
    def _validate_claim_assignments_match_criterion(self) -> CriterionJudgment:
        """Every :class:`ClaimPolarityAssignment` in the tuple
        references this criterion's ``criterion_id``."""
        for index, ca in enumerate(self.claim_polarity_assignments):
            if ca.criterion_id != self.criterion_id:
                raise ValueError(
                    f"CriterionJudgment for {self.criterion_id!r}: "
                    f"claim_polarity_assignments[{index}].criterion_id "
                    f"is {ca.criterion_id!r}, which does not match. "
                    f"Polarity assignments are scoped to one criterion."
                )
        return self


# ---------------------------------------------------------------------------
# Round-level judgment record.
# ---------------------------------------------------------------------------


class RoundJudgment(BaseModel):
    """The judge's full output for one round.

    Frozen, ``extra="forbid"``. The on-disk form lives at
    ``output_dir/mirror/judgments/{round_id}.json`` and is written
    atomically by :func:`~kind.mirror.judge_driver.judge_round`.

    Fields:

    - ``round_id``: the :class:`~kind.mirror.calibration.round.RoundConfig.round_id`
      this judgment is for. The judgment's round_id matches the
      source round's round_id; the judgment is keyed to one round.
    - ``round_config_summary``: human-readable summary of the round's
      :class:`~kind.mirror.statistics.StatisticConfig`,
      :class:`~kind.mirror.llm_caller.LLMConfig`,
      :attr:`~kind.mirror.calibration.sham_schedule.ShamSchedule.shams_per_pass`,
      :attr:`~kind.mirror.calibration.synthetic_perturbation.SyntheticPerturbationSchedule.synthetics_per_pass`,
      etc. — what the judge needs to know about the round's context
      without serializing the full config. The judge builder
      constructs this summary; the round's full config is preserved on
      the source :class:`~kind.mirror.calibration.round.RoundResult`.
    - ``criterion_judgments``: one
      :class:`CriterionJudgment` per active criterion plus one per
      held-out criterion — the judge judges both partitions,
      separately.
    - ``judge_llm_call_records``: the
      :class:`~kind.mirror.calibration.llm_audit.LLMCallRecord`
      tuple covering every judge LLM-call attempt. Phase 9's
      cost-trajectory analysis reads through this.
    - ``wallclock_ms``: end-to-end wallclock of the judge run.
    - ``notes``: free-text for the journal entry.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "0.1.0"
    round_id: str
    round_config_summary: str
    criterion_judgments: tuple[CriterionJudgment, ...]
    judge_llm_call_records: tuple[LLMCallRecord, ...]
    wallclock_ms: int
    notes: str

    @field_validator("round_id")
    @classmethod
    def _validate_round_id_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("round_id must be non-empty.")
        return value

    @field_validator("round_config_summary")
    @classmethod
    def _validate_summary_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "RoundJudgment.round_config_summary must be non-empty: "
                "the summary is the judge's view of the round's context; "
                "an empty summary makes the on-disk artifact non-"
                "interpretable without re-loading the source round."
            )
        return value

    @field_validator("wallclock_ms")
    @classmethod
    def _validate_wallclock_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(
                f"wallclock_ms must be >= 0; got {value}."
            )
        return value

    @model_validator(mode="after")
    def _validate_unique_criterion_ids(self) -> RoundJudgment:
        """Each criterion appears at most once in
        :attr:`criterion_judgments` — the active and held-out
        partitions are disjoint at registry level, so the judgment
        tuple shouldn't double-count a criterion either."""
        ids = [j.criterion_id for j in self.criterion_judgments]
        if len(set(ids)) != len(ids):
            raise ValueError(
                f"RoundJudgment.criterion_judgments has duplicate "
                f"criterion_id values; got {ids}. Each criterion is "
                f"judged at most once per round."
            )
        return self
