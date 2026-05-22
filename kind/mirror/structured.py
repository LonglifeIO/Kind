"""Probe 2 mirror-side structured-reading schemas.

The Probe 1 mirror at ``kind/mirror/caller.py`` produced free-text
``MirrorReading`` records (``schema_version == "0.1.0"``). Probe 2 needs
structured readings the parallel-with-arbiter pipeline can compose: the
Phenomenological Advocate and the Statistical Skeptic each emit a
:class:`StructuredReading` against the same hierarchical digest; the
:class:`Judge` consumes the pair and emits a :class:`JudgeRuling` with
per-claim per-surface outcomes; the faithfulness verifier resolves each
:class:`StructuredClaim`'s citation against the underlying telemetry.

This module is Phase 0's pure-declaration surface. No reader, no judge, no
faithfulness verifier here — just the schemas. The Probe 1 free-text
record stays at ``MIRROR_READING_SCHEMA_VERSION = "0.1.0"`` in
``kind/mirror/caller.py`` for backward-readability of any earlier
readings; Probe 2's structured stream lives at
``MIRROR_READING_V2_VERSION = "0.2.0"`` here.

**The reading-surface stratification (Probe 2 v2).** Probe 2 v2's
synthesis §2.2 distributes reflexive-attention's evidential-weight across
three reading surfaces: substrate-side (KL allocation, ensemble
disagreement, per-dim trajectories — weakest because Phase 8's KS-D
shows substrate-shaping is largely auxiliary-target-shape-independent);
head-internal (``self_prediction_error_t`` distribution, per-dim
``self_prediction_t`` allocation — strongest for self-specificity per
Phase 7-vs-frozen-target ``sp_err`` KS-D = 0.284); behavior-side (Io's
policy conditioning on the scalar — most confabulation-susceptible
because the actor's new column is byte-identical to its initialization
throughout training; the conditioning is fixed-by-init not
developed-through-training). Every :class:`StructuredClaim` carries an
explicit ``reading_surface`` value; the Judge's per-claim rulings carry
the same value; the faithfulness verifier dispatches on
``(cited_stream, reading_surface)``.

**The masked-steps-handling discipline (Probe 1.5 v2 §10 item 3).** The
``self_prediction_error_t`` scalar takes a sentinel value (0.0) on the
first step of every episode, when no actual ``h_{t+1}`` exists; the
``self_prediction_error_masked_t`` boolean is the discriminator. Any
claim that aggregates ``self_prediction_error_t`` over a step range must
declare ``masked_steps_handling``; the verifier rejects citations that
leave it ``"n/a"`` for an aggregation citation. Substrate-side and
behavior-side citations that don't touch the scalar set
``masked_steps_handling`` to ``"n/a"``.

**The cited_stream extension to ``"conditioning_analysis"``.** Behavior-side
claims cite the conditioning analysis module's output cache, not
``agent_step`` directly: the per-state action-distribution-under-perturbation
tables are themselves an analysis artifact (per Probe 1.5 v2 §10 item 11;
formalized in :mod:`kind.observer.conditioning`). The cited
``run_id`` plus ``checkpoint_id`` plus a ``regime``/``perturbation``
encoding in the ``cited_scalar_field`` resolves the citation against the
cached :class:`~kind.observer.conditioning.ConditioningResult` JSONL.

**The frozen+forbid discipline.** Every model is
``ConfigDict(extra="forbid", frozen=True)`` matching
``RecordEnvelope`` in :mod:`kind.observer.schemas`. Once a reading is
produced it is what it is; the reader does not mutate it; downstream
consumers can rely on ``__hash__`` / ``__eq__`` semantics for byte-stable
round-trip checks.

Out of scope at Phase 0:
- The readers themselves (Phase 8 — :mod:`kind.mirror.adversarial`).
- The Judge implementation (Phase 9 — :mod:`kind.mirror.judge`).
- The stability-test runner (Phase 10 — :mod:`kind.mirror.stability`).
- The faithfulness verifier extension (Phase 11 — extension to
  :mod:`kind.observer.eyeball`).
- Any LLM call. Phase 0 is schemas only.
"""

from __future__ import annotations

from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, model_validator

__all__ = [
    "MIRROR_READING_V2_VERSION",
    "ReadingSurface",
    "StructuredClaim",
    "StructuredReading",
    "JudgeRuling",
]


MIRROR_READING_V2_VERSION: Final[str] = "0.2.0"


# Probe 2 v2's three reading surfaces. The ordering — substrate-side first,
# head-internal second, behavior-side third — matches the synthesis's
# evidential-weight ordering (substrate-side weakest, head-internal
# strongest for self-specificity, behavior-side most
# confabulation-susceptible). The order is informational only; consumers
# must not rely on enumeration order for any logic.
ReadingSurface: TypeAlias = Literal[
    "substrate_side",
    "head_internal",
    "behavior_side",
]


# Cited-stream literal extended in v2 with ``"conditioning_analysis"``.
# Probe 1's free-text reading had no concept of cited streams; v2's
# structured citation form names exactly which telemetry stream the claim
# resolves against, so the faithfulness verifier dispatches correctly.
CitedStream: TypeAlias = Literal[
    "agent_step",
    "dream_rollout",
    "replay_meta",
    "world_event",
    "conditioning_analysis",
]


# Faithfulness-status literal. ``"not_checked"`` is the writer-side
# default before the verifier runs; the verifier flips it to one of the
# other three. ``"resolved"`` means the cited value matched within
# tolerance; ``"off_by_tolerance"`` means the cited value missed within
# the verifier's configured tolerance band; ``"unresolved"`` means the
# citation could not be resolved (missing field, out-of-range step,
# malformed cited_scalar_field for the surface, masked_steps_handling
# left at "n/a" for an aggregation citation, etc.).
FaithfulnessStatus: TypeAlias = Literal[
    "resolved",
    "off_by_tolerance",
    "unresolved",
    "not_checked",
]


# Judge-ruling literal. ``"not_judged"`` is the writer-side default
# before the Judge runs.
JudgeClaimRuling: TypeAlias = Literal[
    "supported",
    "absent",
    "unresolved",
    "not_judged",
]


# Masked-steps handling literal. The discipline: any claim that aggregates
# ``self_prediction_error_t`` over a step range must declare whether
# masked steps are included or excluded; ``"n/a"`` is the only acceptable
# value for citations that do not touch the scalar's aggregation. The
# verifier rejects ``"n/a"`` citations that DO touch ``self_prediction_error_t``
# aggregation.
MaskedStepsHandling: TypeAlias = Literal["included", "excluded", "n/a"]


# Reader-role literal. ``"single"`` carries Probe 1's single-mirror form
# forward; ``"advocate"`` and ``"skeptic"`` are Probe 2's parallel pair.
ReaderRole: TypeAlias = Literal["advocate", "skeptic", "single"]


# Framework-anchor literal. The Advocate anchors in
# ``"buddhist_phenomenology"``; the Skeptic anchors in
# ``"null_statistics"`` (substrate-side surface),
# ``"predictive_processing"`` (head-internal surface, where the
# auxiliary-target-tracking refutation lives), or ``"null_statistics"``
# again at behavior-side. ``"neutral"`` is the Judge's framing.
FrameworkAnchor: TypeAlias = Literal[
    "buddhist_phenomenology",
    "predictive_processing",
    "null_statistics",
    "neutral",
]


# Baseline-flag literal. ``"genuine"`` is the actual Probe 1.5 telemetry;
# the four ``"shuffled_*"`` flags name the four shuffle protocols (the
# fourth — ``"shuffled_scalar_within_trajectory"`` — is new in Probe 2
# v2 per synthesis §2.4 element 2 and lands in Phase 3); the five
# ``"lesion_*"`` flags name the five lesion arms (two carried from v1
# ensemble candidates, three new in v2 — ``disable_self_prediction``,
# ``init_zero_scalar_column``, ``zero_or_randomize_scalar`` — per
# synthesis §2.4 element 4 and Phase 4); ``"sham_aligned"`` names a
# reading on telemetry that contains a sham builder-perturbation event
# (per synthesis §2.4 element 3).
BaselineFlag: TypeAlias = Literal[
    "genuine",
    "shuffled_within_episode",
    "shuffled_across_episodes",
    "decoupled_action_state",
    "shuffled_scalar_within_trajectory",
    "lesion_k1",
    "lesion_constant",
    "lesion_disable_self_prediction",
    "lesion_init_zero_scalar_column",
    "lesion_zero_or_randomize_scalar",
    "sham_aligned",
]


# Per-claim Judge ruling without the writer-side ``"not_judged"`` slot.
# This is the literal the Judge actually populates into the ``rulings``
# tuple. The :class:`StructuredClaim`'s ``judge_ruling`` field uses the
# writer-side variant (with ``"not_judged"``) because claims exist before
# the Judge runs.
JudgeRulingOutcome: TypeAlias = Literal["supported", "absent", "unresolved"]


class StructuredClaim(BaseModel):
    """A single claim within a :class:`StructuredReading`.

    Each claim must be grounded in
    ``(stream, run_id, episode/step_range, scalar_field, value)`` per the
    synthesis's structured-grounding mandate. The two new v2 fields
    (``reading_surface``, ``masked_steps_handling``) carry the
    per-surface stratification and the masked-steps discipline.

    The ``cited_episode_range`` and ``cited_step_range`` are individually
    optional — a claim may scope to an episode range without naming
    specific steps, or to a step range that crosses episodes. The
    faithfulness verifier handles either case.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim: str
    cited_stream: CitedStream
    cited_run_id: str
    cited_episode_range: tuple[int, int] | None
    cited_step_range: tuple[int, int] | None
    cited_scalar_field: str
    cited_value: float
    falsifier: str
    paraphrase_stability: float | None
    reseed_stability: float | None
    # ``faithfulness_status`` is the faithfulness verifier's field, not
    # the reading LLM's. It carries the writer-side default
    # ``"not_checked"`` so a claim parsed from a structured-output
    # response that omits the field (Phase 12.5 item 1 drops it from the
    # schema the LLM sees) lands at ``"not_checked"``; the verifier then
    # overwrites the verdict on its own assignment record.
    faithfulness_status: FaithfulnessStatus = "not_checked"
    judge_ruling: JudgeClaimRuling
    reading_surface: ReadingSurface
    masked_steps_handling: MaskedStepsHandling


class StructuredReading(BaseModel):
    """A complete structured reading from one reader on one digest.

    Probe 2's adversarial pair produces two of these per high-confidence
    checkpoint, sharing a ``paired_reading_id``. Probe 1's
    free-text-only reading lives at the older
    :class:`~kind.mirror.caller.MirrorReading`; this is its v2 successor.

    The ``digest_*`` fields name the digest the reader read against; the
    digest itself is not embedded (it can be reconstructed from the run's
    telemetry). The ``baseline_flag`` distinguishes a genuine reading
    from one against shuffled telemetry, lesion telemetry, or
    sham-perturbation-aligned telemetry — the Judge and the smoke
    pipeline use it to compute per-surface contrast ratios.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = MIRROR_READING_V2_VERSION
    run_id: str
    timestamp_ms: int
    reader_role: ReaderRole
    paired_reading_id: str | None
    framework_anchor: FrameworkAnchor
    baseline_flag: BaselineFlag
    digest_run_id: str
    digest_episode_range: tuple[int, int]
    claims: list[StructuredClaim]
    free_text_notes: str


class JudgeRuling(BaseModel):
    """The Judge's per-claim per-surface verdict on a paired reading.

    The ``rulings`` field is a list of ``(claim_index, reading_surface,
    ruling, ground_text)`` tuples — one tuple per Advocate claim per
    surface. The Judge sees both the Advocate's and the Skeptic's
    readings plus the digest, and rules on each Advocate claim at its
    declared surface.

    ``agreement_without_evidence_unresolved`` lists claim indices that
    both readers agreed on but where neither's citation resolved under
    the faithfulness check; the Judge's mandate is to flag these as
    ``"unresolved"`` rather than treating cross-reader agreement as
    additional evidence.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = MIRROR_READING_V2_VERSION
    run_id: str
    timestamp_ms: int
    paired_reading_id: str
    advocate_id: str
    skeptic_id: str
    digest_run_id: str
    rulings: list[tuple[int, ReadingSurface, JudgeRulingOutcome, str]]
    agreement_without_evidence_unresolved: list[int]

    @model_validator(mode="after")
    def _enforce_agreement_indices_in_range(self) -> "JudgeRuling":
        """Indices in ``agreement_without_evidence_unresolved`` are
        non-negative integers — the Judge cannot flag a non-existent
        claim. The list may reference indices outside the current
        ``rulings`` range only if the agreement spans claims the Judge
        chose not to rule on, so the upper-bound check is left to the
        consumer; what we enforce here is the lower bound and unique
        membership."""
        for idx in self.agreement_without_evidence_unresolved:
            if idx < 0:
                raise ValueError(
                    f"agreement_without_evidence_unresolved contains a "
                    f"negative claim index {idx}; indices must be "
                    f"non-negative."
                )
        if len(set(self.agreement_without_evidence_unresolved)) != len(
            self.agreement_without_evidence_unresolved
        ):
            raise ValueError(
                "agreement_without_evidence_unresolved contains duplicate "
                "claim indices; each flagged claim should appear at most "
                "once."
            )
        return self
