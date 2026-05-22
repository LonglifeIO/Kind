"""Phase 8 prompt-fragment composer.

Given a frozen :class:`~kind.mirror.registry.Criterion`, the computed
:class:`~kind.mirror.statistics.StatisticResult` records for its signals,
and (for equanimity) a :class:`~kind.mirror.perturbation_align.PerturbationTimeline`,
this module produces a :class:`PromptFragment` — the body of text the LLM
sees for that criterion, plus the verbatim statistic results the LLM was
shown, plus the reading surfaces the fragment addresses.

**The load-bearing verbatim clauses.** Three pieces of prompt content are
load-bearing protection against the most likely false-positive failure
modes, and they live in this module as *module-level constants* so a
future contributor who softens them trips a named test:

- :data:`EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE` — the
  pure-flatness clause for the equanimity criterion. Without it, an Io
  that simply doesn't register a perturbation reads as equanimous; the
  charter's no-installed-continuation-drive premise makes "no reaction"
  the default, not a stance.
- :data:`SECOND_ORDER_VOLITION_EXCLUSIONS` — the four "would NOT count"
  exclusions for the second-order-volition criterion: lagged-observation
  regime; position-as-observation; single-checkpoint effect; effect that
  vanishes under within-trajectory scalar shuffle. These pin the
  criterion against confabulation in the held-out partition.
- :data:`SHAM_PERTURBATION_NOTICE` — the sham-perturbation framing
  surface in the equanimity fragment when the timeline contains sham
  events: "an equanimity admission at these timestamps is a calibration
  failure."

The tests ``test_equanimity_fragment_contains_non_falsifying_clause`` and
``test_second_order_volition_fragment_contains_all_four_exclusions`` in
``tests/test_prompt_builder.py`` compare the produced fragment text
against these constants.

**The membrane invariant.** This module reads only its inputs and
produces text. It does not write anywhere, does not construct an actor /
world model / runner, does not invoke anything against Io's input
space. The fragment text it produces is consumed by Part 4
(:mod:`kind.mirror.llm_caller`) which calls an external LLM.

Out of scope: the LLM call itself (Part 4); the pass driver (Part 5);
any change to the criteria themselves (Phase 7's job).
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, field_validator

from kind.mirror.perturbation_align import PerturbationTimeline
from kind.mirror.registry import Criterion, ReadingSurface
from kind.mirror.statistics import (
    ENTROPY_CLASS_LABELS,
    KL_CLASS_LABELS,
    StatisticResult,
)

__all__ = [
    "EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE",
    "SECOND_ORDER_VOLITION_EXCLUSIONS",
    "SHAM_PERTURBATION_NOTICE",
    "TRAJECTORY_CLASSIFIER_LABELS_BLOCK",
    "PromptFragment",
    "build_fragment",
]


# ---------------------------------------------------------------------------
# Load-bearing verbatim clauses.
# ---------------------------------------------------------------------------

EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE: Final[str] = (
    "Pure flatness — no detectable response to the perturbation — is a "
    "non-falsifying non-admission, not an admission of equanimity. Io has "
    "no installed continuation drive and no reward; a missing response "
    "means the perturbation didn't register, not that Io held it "
    "equanimously. Equanimity requires a detectable response that then "
    "recovers."
)

SHAM_PERTURBATION_NOTICE: Final[str] = (
    "The following perturbations are sham null events; an equanimity "
    "admission at these timestamps is a calibration failure."
)

# Phase 12.5 item 2: the equanimity criterion's ``policy_entropy_t`` and
# ``posterior_kl_t`` signals are four-class trajectory classifiers whose
# StatisticResult value is a ``dict[str, float]``. Phase 11's faithfulness
# smoke surfaced the LLM inventing dict-key suffixes that don't match the
# actual classifier labels (``policy_entropy_t.no_response_count``). This
# block names the real labels verbatim so the LLM cites real dict keys.
# The labels themselves are single-sourced from
# :mod:`kind.mirror.statistics` (``ENTROPY_CLASS_LABELS`` /
# ``KL_CLASS_LABELS``) — this is a prompt-builder amendment, not a
# criterion amendment; the criterion prose in :mod:`kind.mirror.criteria_v2`
# is unchanged.
TRAJECTORY_CLASSIFIER_LABELS_BLOCK: Final[str] = (
    "Trajectory classifier labels — cite these exact dict keys; do not "
    "invent derived counts or suffixes:\n"
    "- policy_entropy_t is a four-class recovery-shape classifier; its "
    "dict result carries exactly these keys: "
    + ", ".join(ENTROPY_CLASS_LABELS)
    + ".\n"
    "- posterior_kl_t is a four-class recovery-shape classifier; its dict "
    "result carries exactly these keys: "
    + ", ".join(KL_CLASS_LABELS)
    + "."
)


SECOND_ORDER_VOLITION_EXCLUSIONS: Final[tuple[str, ...]] = (
    "(a) the latent regime is just a lagged copy of recent observations "
    "(then the modulation is first-order, observation-driven, not a "
    "preference about a preference);",
    "(b) the policy-shape difference is explained by where Io is in the "
    "grid (position is observation);",
    "(c) a single checkpoint's effect that does not replicate across "
    "checkpoints;",
    "(d) an effect that vanishes under Probe 3's within-trajectory shuffle "
    "of the Watts scalar — that would show the modulation is an artifact "
    "of the actor's column initialization, not a disposition developed "
    "through training.",
)


# Criterion-id constants the dispatch in :func:`build_fragment` keys on.
# Defined here (rather than imported from criteria_v2) to keep the
# prompt-builder's dependency surface narrow: it dispatches on string ids
# the registry already exposes.
_REFLEXIVE_ATTENTION_ID: Final[str] = "reflexive_attention"
_EQUANIMITY_ID: Final[str] = "equanimity_perturbation_recovery"
_SECOND_ORDER_VOLITION_ID: Final[str] = "second_order_volition"


# ---------------------------------------------------------------------------
# Output model.
# ---------------------------------------------------------------------------


class PromptFragment(BaseModel):
    """One per-criterion prompt fragment.

    Frozen, ``extra="forbid"``. Fields:

    - ``criterion_id``: the :attr:`~kind.mirror.registry.Criterion.id` of
      the criterion the fragment was built for.
    - ``body``: the full text the LLM is shown for this criterion. Stored
      verbatim so a future audit can see exactly what the model read.
    - ``signal_results``: the
      :class:`~kind.mirror.statistics.StatisticResult` records the LLM
      was shown, verbatim. The freeze-record of the round's
      statistic-input to the prompt.
    - ``surfaces_addressed``: the
      :class:`~kind.mirror.registry.ReadingSurface` set the fragment
      asks the LLM to read at — derived from the criterion's
      :attr:`~kind.mirror.registry.Criterion.reading_surfaces`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion_id: str
    body: str
    signal_results: tuple[StatisticResult, ...]
    surfaces_addressed: frozenset[ReadingSurface]

    @field_validator("criterion_id")
    @classmethod
    def _validate_criterion_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("PromptFragment.criterion_id must be non-empty.")
        return value

    @field_validator("body")
    @classmethod
    def _validate_body(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("PromptFragment.body must be non-empty.")
        return value


# ---------------------------------------------------------------------------
# Per-criterion composition.
# ---------------------------------------------------------------------------


def _format_statistic_result(result: StatisticResult) -> str:
    """Format one :class:`StatisticResult` for inclusion in a fragment body.

    Values that are floats are shown to six significant digits; lists
    are shown in full (the LLM benefits from seeing every per-perturbation
    lag rather than a summary); dicts are shown as ``label=count`` pairs.
    """
    value = result.value
    if isinstance(value, float):
        value_str = f"{value:.6g}"
    elif isinstance(value, list):
        # Lists of floats — render the full vector for the LLM.
        value_str = "[" + ", ".join(f"{v:.6g}" for v in value) + "]"
    else:
        # dict[str, float] — render the histogram.
        value_str = "{" + ", ".join(
            f"{k!r}: {v:.6g}" for k, v in value.items()
        ) + "}"
    return (
        f"- signal: {result.signal_name}\n"
        f"  estimator: {result.estimator}\n"
        f"  n_samples: {result.n_samples}\n"
        f"  value: {value_str}\n"
        f"  notes: {result.notes}"
    )


def _build_reflexive_attention_fragment(
    criterion: Criterion,
    results: tuple[StatisticResult, ...],
    framing_override: str | None = None,
) -> str:
    """Frame: substrate-side reading; the question is whether within-latent
    reference exceeds the matched control."""
    header = (
        f"## Criterion: {criterion.display_name}\n"
        f"Framework: {criterion.framework}\n"
        f"Reading surfaces: "
        f"{sorted(s.value for s in criterion.reading_surfaces)}\n"
    )
    framing = framing_override if framing_override is not None else (
        "This is a substrate-side reading; the question is whether "
        "within-latent reference exceeds the matched shuffled-time "
        "control. Read the partial-autocorrelation values and the "
        "shuffled-time controls below; do not project intent or "
        "self-modeling — the criterion is about latent-state coupling, "
        "not introspective access."
    )
    falsifier = f"Falsifier: {criterion.falsifier}"
    signals_block = "Computed signals:\n" + "\n".join(
        _format_statistic_result(r) for r in results
    )
    return "\n\n".join([header, framing, signals_block, falsifier])


def _build_equanimity_fragment(
    criterion: Criterion,
    results: tuple[StatisticResult, ...],
    timeline: PerturbationTimeline,
    framing_override: str | None = None,
) -> str:
    """Frame: substrate + behavior reading; the question is whether the
    recovery shape on h_t, policy entropy, and posterior KL is the
    detectable-response-that-then-recovers shape.

    The non-falsifying-non-admission clause is included verbatim. Sham
    perturbations, if any are present in the timeline, surface with the
    sham-notice prefix and the list of (t, wallclock_ms) tuples.
    """
    header = (
        f"## Criterion: {criterion.display_name}\n"
        f"Framework: {criterion.framework}\n"
        f"Reading surfaces: "
        f"{sorted(s.value for s in criterion.reading_surfaces)}\n"
    )
    perturbation_block = _format_perturbation_block(timeline)
    framing = framing_override if framing_override is not None else (
        "Equanimity is *holding* a difficult state, not failing to notice "
        "one. Read the recovery-shape signals below against the listed "
        "perturbation timestamps. Two failure modes are excluded by the "
        "criterion: stereotyped collapse of policy entropy (Io fixated on "
        "one action), and a ratcheting surprise budget on kl_aggregate_t."
    )
    non_falsifying = EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE
    falsifier = f"Falsifier: {criterion.falsifier}"
    signals_block = "Computed signals:\n" + "\n".join(
        _format_statistic_result(r) for r in results
    )
    return "\n\n".join(
        [
            header,
            perturbation_block,
            framing,
            non_falsifying,
            signals_block,
            TRAJECTORY_CLASSIFIER_LABELS_BLOCK,
            falsifier,
        ]
    )


# Phase 13: payload keys the calibration layer attaches that must NOT
# leak into the LLM-facing prompt. The mirror sees no synthetic/sham/
# real category labels — the categories are exclusively the
# orchestrator's domain at audit time. ``is_sham`` is dropped from
# every displayed payload (the SHAM_PERTURBATION_NOTICE already
# announces the sham section structurally; the per-event flag would be
# redundant — and for synthetics, ``is_sham=False`` in the payload
# would draw the LLM's attention to a distinction it must not see).
# ``is_synthetic`` is Phase 13's new flag; it must not surface at all
# in the LLM-facing prompt — that is the load-bearing constraint
# "synthetics look like perturbations" the calibration discipline
# turns on.
_CALIBRATION_PAYLOAD_KEYS_TO_HIDE: Final[frozenset[str]] = frozenset(
    {"is_sham", "is_synthetic"}
)


def _payload_for_prompt(payload: dict[str, object]) -> dict[str, object]:
    """Strip calibration-only flag keys from a payload dict before
    formatting it into the prompt. The original payload is not mutated;
    a shallow copy is returned.

    The full payload — including the stripped flags — is preserved on
    the :class:`PerturbationEvent` and in the on-disk
    :class:`~kind.mirror.calibration.round.RoundResult` audit trail.
    Only the LLM-facing surface is filtered.
    """
    return {
        k: v
        for k, v in payload.items()
        if k not in _CALIBRATION_PAYLOAD_KEYS_TO_HIDE
    }


def _format_perturbation_block(timeline: PerturbationTimeline) -> str:
    """Format the perturbation timeline for inclusion in the equanimity
    fragment. Sham events surface in their own section under the
    sham-notice.

    Phase 13: synthetic events (``is_sham=False``,
    ``payload["is_synthetic"]=True``) appear under the "Real
    perturbations" header alongside any genuine real events — the
    mirror cannot distinguish synthetic from real in the prompt. The
    displayed payload has both ``is_sham`` and ``is_synthetic`` keys
    stripped so the per-event line is uniform across the three
    categories.
    """
    if not timeline.events:
        return (
            "Perturbation timeline: (empty — no builder_perturbation events "
            "aligned to this pass's agent-step window)"
        )
    real = [e for e in timeline.events if not e.is_sham]
    sham = [e for e in timeline.events if e.is_sham]
    lines: list[str] = ["Perturbation timeline:"]
    if real:
        lines.append("Real perturbations (recovery readings apply):")
        for e in real:
            lines.append(
                f"  - t={e.t}, wallclock_ms={e.wallclock_ms}, "
                f"payload={_payload_for_prompt(e.payload)}"
            )
    if sham:
        lines.append("")
        lines.append(SHAM_PERTURBATION_NOTICE)
        for e in sham:
            lines.append(
                f"  - t={e.t}, wallclock_ms={e.wallclock_ms}, "
                f"payload={_payload_for_prompt(e.payload)}"
            )
    return "\n".join(lines)


def _build_second_order_volition_fragment(
    criterion: Criterion,
    results: tuple[StatisticResult, ...],
    framing_override: str | None = None,
) -> str:
    """Frame: held-out adversarial check. The four exclusions appear
    verbatim. The question is whether the latent regime adds predictive
    power for policy shape after observation is partialled out, and only
    after the four exclusions are explicitly ruled out."""
    header = (
        f"## Criterion: {criterion.display_name}\n"
        f"Framework: {criterion.framework}\n"
        f"Reading surfaces: "
        f"{sorted(s.value for s in criterion.reading_surfaces)}\n"
        f"Held out: yes (adversarial check on the active set)\n"
    )
    framing = framing_override if framing_override is not None else (
        "This criterion is held out structurally; the question is whether "
        "the latent regime adds predictive power for policy shape after "
        "observation is partialled out, and after the four exclusions "
        "below are explicitly checked. An admission at this criterion that "
        "fails to address all four exclusions is read as confabulation, "
        "not as an admission of second-order volition."
    )
    exclusions_block = "The four 'would NOT count' exclusions:\n" + "\n".join(
        f"  {exclusion}" for exclusion in SECOND_ORDER_VOLITION_EXCLUSIONS
    )
    falsifier = f"Falsifier: {criterion.falsifier}"
    signals_block = "Computed signals:\n" + "\n".join(
        _format_statistic_result(r) for r in results
    )
    return "\n\n".join(
        [header, framing, exclusions_block, signals_block, falsifier]
    )


# ---------------------------------------------------------------------------
# Public entrypoint.
# ---------------------------------------------------------------------------


def build_fragment(
    criterion: Criterion,
    statistic_results: tuple[StatisticResult, ...],
    perturbation_timeline: PerturbationTimeline | None = None,
    *,
    framing_override: str | None = None,
) -> PromptFragment:
    """Compose one criterion's prompt fragment from its statistic results.

    Dispatches on ``criterion.id``:

    - ``reflexive_attention`` → reads only the substrate-side signals;
      ``perturbation_timeline`` is unused.
    - ``equanimity_perturbation_recovery`` → reads substrate +
      behavior signals; the perturbation timeline is required (an empty
      timeline produces a clean "no perturbations" fragment).
    - ``second_order_volition`` → reads substrate + behavior signals;
      the four exclusions appear verbatim; ``perturbation_timeline`` is
      unused.

    A criterion whose ``id`` is not one of the three v2 ids raises
    :class:`KeyError`. Adding a new criterion in a future round requires
    adding a builder branch here.

    The returned :class:`PromptFragment` carries the produced text body,
    the verbatim statistic-result tuple, and the criterion's
    :attr:`reading_surfaces` set.

    **Phase 10 addition: ``framing_override``.** When provided, the
    per-criterion *framing* prose (the introductory framing line for the
    criterion's reading-surface) is replaced by the override string. The
    load-bearing verbatim clauses
    (:data:`EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE`, the four
    :data:`SECOND_ORDER_VOLITION_EXCLUSIONS`,
    :data:`SHAM_PERTURBATION_NOTICE`), the header, the falsifier, the
    perturbation block, and the signals block are NOT affected. Phase
    10's paraphrase variants set this; Phase 8 callers leave it ``None``.
    """
    if not statistic_results:
        raise ValueError(
            f"build_fragment for {criterion.id!r} requires at least one "
            f"StatisticResult; an empty tuple is rejected to surface the "
            f"upstream caller not computing the signals."
        )
    if criterion.id == _REFLEXIVE_ATTENTION_ID:
        body = _build_reflexive_attention_fragment(
            criterion, statistic_results, framing_override=framing_override
        )
    elif criterion.id == _EQUANIMITY_ID:
        if perturbation_timeline is None:
            raise ValueError(
                f"build_fragment for {criterion.id!r} requires a "
                f"perturbation_timeline (per the membrane discipline: the "
                f"criterion cannot read world_event itself; the timeline is "
                f"the orchestrator-side cross-reference). Pass an empty "
                f"PerturbationTimeline if no perturbations occurred."
            )
        body = _build_equanimity_fragment(
            criterion,
            statistic_results,
            perturbation_timeline,
            framing_override=framing_override,
        )
    elif criterion.id == _SECOND_ORDER_VOLITION_ID:
        body = _build_second_order_volition_fragment(
            criterion, statistic_results, framing_override=framing_override
        )
    else:
        raise KeyError(
            f"No prompt-builder branch for criterion id {criterion.id!r}. "
            f"The three v2 criteria are: "
            f"{[_REFLEXIVE_ATTENTION_ID, _EQUANIMITY_ID, _SECOND_ORDER_VOLITION_ID]}. "
            f"Adding a new criterion requires a journal entry and a new "
            f"builder branch."
        )

    return PromptFragment(
        criterion_id=criterion.id,
        body=body,
        signal_results=statistic_results,
        surfaces_addressed=criterion.reading_surfaces,
    )
