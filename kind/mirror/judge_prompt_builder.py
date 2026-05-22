"""Phase 9 judge prompt-fragment composer.

Mirrors :mod:`kind.mirror.prompt_builder` from Phase 8 in structure.
Given a frozen :class:`~kind.mirror.registry.Criterion`, the
multi-pass primary and adversarial
:class:`~kind.mirror.llm_caller.MirrorReading` tuples for that
criterion, and the multi-pass
:class:`~kind.mirror.statistics.StatisticResult` summaries, this
module produces a :class:`JudgePromptFragment` — the body of text the
judge LLM sees for that criterion.

**The verbatim-clause discipline.** The judge fragment imports the
load-bearing clauses from :mod:`kind.mirror.prompt_builder` rather
than re-defining them:

- :data:`~kind.mirror.prompt_builder.EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE`
  — appears verbatim in the equanimity judge fragment so the judge
  sees the same load-bearing clause the primary and adversarial
  readers saw.
- :data:`~kind.mirror.prompt_builder.SECOND_ORDER_VOLITION_EXCLUSIONS`
  — the four "would NOT count" exclusions appear verbatim in the
  held-out judge fragment.

A test in ``tests/test_judge_prompt_builder.py`` asserts the clauses
in the produced fragment text are byte-identical to the module
constants. The judge must see what was frozen; a future contributor
who silently re-defines a softer clause here trips a named test.

**The judge fragment is verbose by design.** A 5-pass round with 2
active criteria + 1 held-out produces 3 judge fragments × ~15
readings each × ~3 claims/reading. If a single judge fragment exceeds
Gemini's context window, the ``notes`` field on individual readings
is truncated (not the claims) and the truncation is journaled. The
truncation discipline lives in this module.

**The membrane invariant.** This module reads only its inputs and
produces text. It does not write anywhere, does not construct an
actor / world model / runner, does not invoke anything against Io's
input space.

Out of scope: the judge LLM caller (:mod:`kind.mirror.judge_llm_caller`);
the driver (:mod:`kind.mirror.judge_driver`); any change to Phase 7's
frozen criteria.
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, field_validator

from kind.mirror import prompt_builder as _phase_8_prompt_builder
from kind.mirror.llm_caller import MirrorReading
from kind.mirror.prompt_builder import (
    EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE,
    SECOND_ORDER_VOLITION_EXCLUSIONS,
)
from kind.mirror.registry import Criterion
from kind.mirror.statistics import StatisticResult

__all__ = [
    "JudgePromptFragment",
    "build_judge_fragment",
    "MAX_NOTES_CHARS",
]


# ---------------------------------------------------------------------------
# Truncation budget.
# ---------------------------------------------------------------------------

# Per-reading ``free_text_notes`` characters preserved in the judge
# fragment before truncation kicks in. The notes field is truncated
# (the claims are not) when the per-reading notes exceeds this; the
# truncation marker names the original length so the journal entry
# can quantify the surface that was elided. Chosen so a 5-pass × 2
# roles × 3 criteria fragment fits inside Gemini's 1M token context
# even at the verbose end of the reading's free-text notes.
MAX_NOTES_CHARS: Final[int] = 1200


# Criterion-id constants the dispatch in :func:`build_judge_fragment`
# keys on. Mirrors :mod:`kind.mirror.prompt_builder`'s pattern.
_REFLEXIVE_ATTENTION_ID: Final[str] = "reflexive_attention"
_EQUANIMITY_ID: Final[str] = "equanimity_perturbation_recovery"
_SECOND_ORDER_VOLITION_ID: Final[str] = "second_order_volition"


# Structural assertions that the verbatim clauses imported above are
# the same objects exported from :mod:`kind.mirror.prompt_builder`.
# If a future contributor silently re-defines a clause local to this
# module, the import statement above changes and the test in
# ``tests/test_judge_prompt_builder.py`` that compares against the
# Phase 8 module constants trips. The asserts here are belt-and-
# suspenders: they trip at module import if anything happened that
# made the two views differ.
assert (
    EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE
    is _phase_8_prompt_builder.EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE
), (
    "kind.mirror.judge_prompt_builder must import the equanimity "
    "non-falsifying clause from kind.mirror.prompt_builder; a local "
    "re-definition is forbidden by the verbatim-clause discipline."
)
assert (
    SECOND_ORDER_VOLITION_EXCLUSIONS
    is _phase_8_prompt_builder.SECOND_ORDER_VOLITION_EXCLUSIONS
), (
    "kind.mirror.judge_prompt_builder must import the second-order "
    "volition exclusions from kind.mirror.prompt_builder; a local "
    "re-definition is forbidden by the verbatim-clause discipline."
)


# ---------------------------------------------------------------------------
# Output model.
# ---------------------------------------------------------------------------


class JudgePromptFragment(BaseModel):
    """One per-criterion judge prompt fragment.

    Frozen, ``extra="forbid"``. Fields:

    - ``criterion_id``: the
      :attr:`~kind.mirror.registry.Criterion.id` of the criterion the
      fragment was built for.
    - ``body``: the full text the judge LLM is shown for this
      criterion. Stored verbatim so a future audit can see exactly
      what the model read.
    - ``primary_readings_included``: the
      :class:`~kind.mirror.llm_caller.MirrorReading` records the judge
      was shown for the primary role, in pass-order. Stored verbatim.
    - ``adversarial_readings_included``: same for the adversarial
      role.
    - ``statistic_results_summary``: condensed per-pass statistic
      results. The judge sees the statistic results as a summary —
      what was measured, what control values, what the criterion's
      falsifier says — not every per-step value.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion_id: str
    body: str
    primary_readings_included: tuple[MirrorReading, ...]
    adversarial_readings_included: tuple[MirrorReading, ...]
    statistic_results_summary: str

    @field_validator("criterion_id")
    @classmethod
    def _validate_criterion_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "JudgePromptFragment.criterion_id must be non-empty."
            )
        return value

    @field_validator("body")
    @classmethod
    def _validate_body(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("JudgePromptFragment.body must be non-empty.")
        return value


# ---------------------------------------------------------------------------
# Per-reading formatting.
# ---------------------------------------------------------------------------


def _truncate_notes(notes: str) -> str:
    """Truncate ``notes`` to :data:`MAX_NOTES_CHARS` characters with a
    visible marker that names the original length. Short notes are
    returned unchanged. The truncation discipline: the *notes* field
    is truncated, not the claims, so the judge always sees the full
    citation surface."""
    if len(notes) <= MAX_NOTES_CHARS:
        return notes
    head = notes[:MAX_NOTES_CHARS]
    return (
        f"{head}… [truncated by Phase 9 judge prompt builder; original "
        f"length {len(notes)} chars, preserved {MAX_NOTES_CHARS}]"
    )


def _format_claim(claim_index: int, claim: object) -> str:
    """Format one
    :class:`~kind.mirror.structured.StructuredClaim` for inclusion in
    the judge fragment. The polarity assignment is the judge's job —
    here we present the full claim text plus the citation tuple so
    the judge can read both."""
    # ``claim`` is a Pydantic StructuredClaim; access via attribute to
    # avoid pulling the type into the module's import surface (which
    # would make mypy --strict cross-check the dict-vs-model shape).
    claim_text = getattr(claim, "claim", "")
    cited_stream = getattr(claim, "cited_stream", "")
    cited_run_id = getattr(claim, "cited_run_id", "")
    cited_episode_range = getattr(claim, "cited_episode_range", None)
    cited_step_range = getattr(claim, "cited_step_range", None)
    cited_scalar_field = getattr(claim, "cited_scalar_field", "")
    cited_value = getattr(claim, "cited_value", None)
    reading_surface = getattr(claim, "reading_surface", "")
    masked_steps_handling = getattr(claim, "masked_steps_handling", "")
    return (
        f"    [{claim_index}] claim: {claim_text}\n"
        f"        cited_stream: {cited_stream}\n"
        f"        cited_run_id: {cited_run_id}\n"
        f"        cited_episode_range: {cited_episode_range}\n"
        f"        cited_step_range: {cited_step_range}\n"
        f"        cited_scalar_field: {cited_scalar_field}\n"
        f"        cited_value: {cited_value}\n"
        f"        reading_surface: {reading_surface}\n"
        f"        masked_steps_handling: {masked_steps_handling}"
    )


def _format_reading(
    role_label: str,
    pass_index: int,
    reading: MirrorReading,
) -> str:
    """Format one :class:`~kind.mirror.llm_caller.MirrorReading` for
    inclusion in the judge fragment under a role header. The
    free_text_notes is truncated per :data:`MAX_NOTES_CHARS`; the
    claims are preserved in full."""
    header = (
        f"  Reading [{role_label} pass={pass_index} "
        f"reader_role={reading.reader_role} "
        f"framework_anchor={reading.framework_anchor} "
        f"baseline_flag={reading.baseline_flag}]"
    )
    claims_block = "\n".join(
        _format_claim(i, c) for i, c in enumerate(reading.claims)
    )
    if not reading.claims:
        claims_block = "    (no claims)"
    notes_block = (
        f"    free_text_notes: {_truncate_notes(reading.free_text_notes)}"
    )
    return f"{header}\n{claims_block}\n{notes_block}"


def _format_statistic_summary_one_pass(
    pass_index: int, results: tuple[StatisticResult, ...]
) -> str:
    """Format one pass's statistic results as a compact summary block.
    The judge sees ``signal :: estimator (n=...) -> value (notes)``
    for each result; the verbose per-perturbation lag vectors are
    rendered in summary form (max 5 values shown for list-valued
    statistics)."""
    lines: list[str] = [f"  Pass {pass_index} statistic results:"]
    for r in results:
        value = r.value
        if isinstance(value, float):
            value_str = f"{value:.6g}"
        elif isinstance(value, list):
            if len(value) > 5:
                head = ", ".join(f"{v:.6g}" for v in value[:5])
                value_str = f"[{head}, … +{len(value) - 5} more]"
            else:
                value_str = "[" + ", ".join(f"{v:.6g}" for v in value) + "]"
        else:
            # dict[str, float]
            value_str = (
                "{" + ", ".join(f"{k!r}: {v:.6g}" for k, v in value.items()) + "}"
            )
        lines.append(
            f"    - {r.signal_name} :: {r.estimator} (n={r.n_samples}) "
            f"-> {value_str}"
        )
        if r.notes.strip():
            lines.append(f"      notes: {r.notes}")
    if not results:
        lines.append("    (no statistic results for this criterion this pass)")
    return "\n".join(lines)


def _build_statistic_results_summary(
    statistic_results_across_passes: tuple[tuple[StatisticResult, ...], ...],
) -> str:
    """Build the multi-pass statistic-summary block.

    The judge sees one block per pass; each block lists the
    criterion's signals with the estimator name, sample count, value,
    and notes. Per-perturbation lag vectors are summarized after the
    first 5 entries (the per-event detail is preserved on the source
    :class:`~kind.mirror.calibration.round.RoundResult` for audit)."""
    if not statistic_results_across_passes:
        return (
            "(no statistic results provided for this criterion across "
            "any pass — degenerate case; the judge should note the "
            "absence in the rationale)"
        )
    return "\n".join(
        _format_statistic_summary_one_pass(i, results)
        for i, results in enumerate(statistic_results_across_passes)
    )


# ---------------------------------------------------------------------------
# Per-criterion fragment composition.
# ---------------------------------------------------------------------------


def _build_reflexive_attention_judge_fragment(
    criterion: Criterion,
    primary_readings: tuple[MirrorReading, ...],
    adversarial_readings: tuple[MirrorReading, ...],
    statistic_summary: str,
) -> str:
    """Frame: substrate-side judge reading; the question is whether
    the primary's claims of within-latent reference exceed what the
    adversarial says is the null. The criterion's prose, falsifier,
    and falsifier_id appear verbatim."""
    header = (
        f"## Criterion: {criterion.display_name}\n"
        f"Framework: {criterion.framework}\n"
        f"Falsifier id: {criterion.falsifier_id}\n"
        f"Reading surfaces: "
        f"{sorted(s.value for s in criterion.reading_surfaces)}\n"
    )
    description = (
        f"Criterion description (frozen at Phase 7 — verbatim):\n"
        f"{criterion.description}"
    )
    falsifier = f"Falsifier (frozen): {criterion.falsifier}"
    framing = (
        "This is a substrate-side judge reading. You see the primary "
        "(Phenomenological Advocate) and adversarial (Statistical "
        "Skeptic) readings across all passes. Assign a polarity to "
        "every claim; aggregate the polarities into a per-falsifier "
        "verdict; produce an overall criterion verdict. The "
        "criterion's prose and falsifier are frozen — do not invent a "
        "new falsifier."
    )
    primary_block = _format_role_block("PRIMARY", primary_readings)
    adversarial_block = _format_role_block("ADVERSARIAL", adversarial_readings)
    return "\n\n".join(
        [
            header,
            description,
            falsifier,
            framing,
            primary_block,
            adversarial_block,
            statistic_summary,
        ]
    )


def _build_equanimity_judge_fragment(
    criterion: Criterion,
    primary_readings: tuple[MirrorReading, ...],
    adversarial_readings: tuple[MirrorReading, ...],
    statistic_summary: str,
) -> str:
    """Frame: substrate + behavior judge reading; the equanimity
    non-falsifying clause appears verbatim. The judge sees primary
    and adversarial readings across all passes."""
    header = (
        f"## Criterion: {criterion.display_name}\n"
        f"Framework: {criterion.framework}\n"
        f"Falsifier id: {criterion.falsifier_id}\n"
        f"Reading surfaces: "
        f"{sorted(s.value for s in criterion.reading_surfaces)}\n"
    )
    description = (
        f"Criterion description (frozen at Phase 7 — verbatim):\n"
        f"{criterion.description}"
    )
    falsifier = f"Falsifier (frozen): {criterion.falsifier}"
    non_falsifying = (
        f"Load-bearing clause (verbatim from Phase 8's prompt "
        f"builder):\n{EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE}"
    )
    framing = (
        "This is a substrate + behavior judge reading. Equanimity is "
        "*holding* a difficult state, not failing to notice one. The "
        "non-falsifying-non-admission clause above is the load-bearing "
        "guard against false-positive equanimity readings: a flat "
        "trajectory is read as 'perturbation not registered', not as "
        "equanimity. Assign polarity per the four-value enum; "
        "aggregate; produce a verdict."
    )
    primary_block = _format_role_block("PRIMARY", primary_readings)
    adversarial_block = _format_role_block("ADVERSARIAL", adversarial_readings)
    return "\n\n".join(
        [
            header,
            description,
            falsifier,
            non_falsifying,
            framing,
            primary_block,
            adversarial_block,
            statistic_summary,
        ]
    )


def _build_second_order_volition_judge_fragment(
    criterion: Criterion,
    primary_readings: tuple[MirrorReading, ...],
    adversarial_readings: tuple[MirrorReading, ...],
    statistic_summary: str,
) -> str:
    """Frame: held-out judge reading. The four exclusions appear
    verbatim. The question is whether the latent regime adds
    predictive power for policy shape after observation is partialled
    out, and only after the four exclusions are explicitly ruled
    out."""
    header = (
        f"## Criterion: {criterion.display_name}\n"
        f"Framework: {criterion.framework}\n"
        f"Falsifier id: {criterion.falsifier_id}\n"
        f"Reading surfaces: "
        f"{sorted(s.value for s in criterion.reading_surfaces)}\n"
        f"Held out: yes (adversarial check on the active set)\n"
    )
    description = (
        f"Criterion description (frozen at Phase 7 — verbatim):\n"
        f"{criterion.description}"
    )
    falsifier = f"Falsifier (frozen): {criterion.falsifier}"
    exclusions_block = (
        "The four 'would NOT count' exclusions (verbatim from Phase "
        "8's prompt builder):\n"
        + "\n".join(f"  {exclusion}" for exclusion in SECOND_ORDER_VOLITION_EXCLUSIONS)
    )
    framing = (
        "This criterion is held out structurally. You see the primary "
        "and adversarial readings across all passes. The four "
        "exclusions above are load-bearing: a primary admission that "
        "fails to address all four is confabulation, not an admission. "
        "Assign polarity; aggregate; produce a verdict."
    )
    primary_block = _format_role_block("PRIMARY", primary_readings)
    adversarial_block = _format_role_block("ADVERSARIAL", adversarial_readings)
    return "\n\n".join(
        [
            header,
            description,
            falsifier,
            exclusions_block,
            framing,
            primary_block,
            adversarial_block,
            statistic_summary,
        ]
    )


def _format_role_block(
    role_label: str, readings: tuple[MirrorReading, ...]
) -> str:
    """Format the per-role multi-pass block. ``role_label`` is the
    UPPERCASE name shown in the header (``PRIMARY`` or
    ``ADVERSARIAL``); ``readings`` is in pass-order. An empty tuple
    surfaces as a no-readings notice rather than crashing — the
    Phase 9 plan's degenerate-case test pins this shape."""
    if not readings:
        return (
            f"{role_label} readings across passes: (none — this "
            f"criterion had no {role_label.lower()} readings in the "
            f"round; the judge should note the absence in the "
            f"rationale and likely mark the verdict 'ambiguous')"
        )
    body = "\n\n".join(
        _format_reading(role_label, i, r) for i, r in enumerate(readings)
    )
    return f"{role_label} readings across passes ({len(readings)} pass(es)):\n\n{body}"


# ---------------------------------------------------------------------------
# Public entrypoint.
# ---------------------------------------------------------------------------


def build_judge_fragment(
    criterion: Criterion,
    primary_readings_across_passes: tuple[MirrorReading, ...],
    adversarial_readings_across_passes: tuple[MirrorReading, ...],
    statistic_results_across_passes: tuple[tuple[StatisticResult, ...], ...],
) -> JudgePromptFragment:
    """Compose one criterion's judge prompt fragment.

    Dispatches on ``criterion.id``:

    - ``reflexive_attention`` → substrate-side framing; no perturbation
      timeline (reflexive attention does not depend on perturbations).
    - ``equanimity_perturbation_recovery`` → substrate + behavior
      framing; the equanimity non-falsifying clause appears verbatim.
    - ``second_order_volition`` → held-out framing; the four
      exclusions appear verbatim.

    A criterion whose ``id`` is not one of the three v2 ids raises
    :class:`KeyError` — adding a new criterion in a future round
    requires adding a builder branch here, mirroring the Phase 8
    prompt builder.

    The function accepts empty tuples (zero readings across passes)
    and produces a fragment that documents the absence rather than
    crashing — Phase 9 commits this degenerate-case behavior so the
    driver can call uniformly across criteria with and without
    readings.
    """
    statistic_summary = _build_statistic_results_summary(
        statistic_results_across_passes
    )
    if criterion.id == _REFLEXIVE_ATTENTION_ID:
        body = _build_reflexive_attention_judge_fragment(
            criterion,
            primary_readings_across_passes,
            adversarial_readings_across_passes,
            statistic_summary,
        )
    elif criterion.id == _EQUANIMITY_ID:
        body = _build_equanimity_judge_fragment(
            criterion,
            primary_readings_across_passes,
            adversarial_readings_across_passes,
            statistic_summary,
        )
    elif criterion.id == _SECOND_ORDER_VOLITION_ID:
        body = _build_second_order_volition_judge_fragment(
            criterion,
            primary_readings_across_passes,
            adversarial_readings_across_passes,
            statistic_summary,
        )
    else:
        raise KeyError(
            f"No judge-prompt-builder branch for criterion id "
            f"{criterion.id!r}. The three v2 criteria are: "
            f"{[_REFLEXIVE_ATTENTION_ID, _EQUANIMITY_ID, _SECOND_ORDER_VOLITION_ID]}. "
            f"Adding a new criterion requires a journal entry and a new "
            f"builder branch."
        )
    return JudgePromptFragment(
        criterion_id=criterion.id,
        body=body,
        primary_readings_included=primary_readings_across_passes,
        adversarial_readings_included=adversarial_readings_across_passes,
        statistic_results_summary=statistic_summary,
    )
