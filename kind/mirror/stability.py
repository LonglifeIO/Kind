"""Phase 10 stability-test runner.

Phase 10 wraps Phase 8's :func:`~kind.mirror.llm_caller.call_mirror_llm`
in a paraphrase-and-reseed loop, computes per-surface structured-field
agreement across the resulting readings, and produces an admissibility
verdict per :class:`~kind.mirror.registry.ReadingSurface`. The runner
exercises what Phase 12's smoke explicitly did NOT test: stability
across paraphrases and reseeds.

**The synthesis's three commitments Phase 10 honors structurally.**

- **The paraphrase set is checked in.** Per the synthesis's §7
  calibration-protocol commitment, the paraphrase set is committed at
  module level as :data:`PARAPHRASE_VARIANTS_PER_SURFACE` and revisited
  only with a journaled reason. The structural test
  ``test_paraphrase_variants_do_not_modify_verbatim_clauses`` asserts
  that no variant contains or alters the load-bearing clauses imported
  from :mod:`kind.mirror.prompt_builder`
  (:data:`~kind.mirror.prompt_builder.EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE`,
  the four
  :data:`~kind.mirror.prompt_builder.SECOND_ORDER_VOLITION_EXCLUSIONS`,
  :data:`~kind.mirror.prompt_builder.SHAM_PERTURBATION_NOTICE`). The
  variants vary framing prose, not substantive content.

- **Reseeds are independent LLM calls with varied seeds.** Phase 10's
  reseed pass uses :data:`STABILITY_TEMPERATURE` (0.7, the
  synthesis-default exploratory setting) and varies the seed across
  reseed calls: the n-th reseed call receives seed
  :data:`STABILITY_SEED_BASE` ``+ n``. Both temperature and seed flow
  through :class:`~kind.mirror.llm_caller.LLMConfig` via the Phase 10
  extension to ``call_mirror_llm`` (which derives a copy of the config
  with the seed/temperature overrides at call time).

- **Per-surface aggregation.** The synthesis specifies stability per
  reading surface (substrate-side, head-internal, behavior-side); the
  per-surface agreement scores are the load-bearing output. The
  per-claim
  ``structured_field_agreement_per_claim`` tuple is informational —
  consumers can derive per-claim scores from the paraphrase readings
  in the result.

**Reconciliation with Phase 9's per-criterion judge shape.** Phase 9's
judge operates on the per-criterion partition. Phase 10's stability
runner gates on the per-surface axis per the synthesis. The two
verifiers run independently; the join between per-surface stability
scores and per-criterion verdicts happens at consumption time (a Phase
11+ verifier, or a Phase 9 amendment that wires Phase 10's scores
in). Phase 10 produces the per-surface record and exposes it for
consumption.

**Reconciliation with Phase 8's prompt-fragment contract.** The
synthesis's stability-runner signature names ``telemetry_batch`` as an
input. Phase 8's prompt-builder consumes pre-computed
:class:`~kind.mirror.statistics.StatisticResult` records, not raw
telemetry batches. Phase 10's signature takes ``statistic_results``
matching Phase 8's actual contract; the orchestrator (or a smoke
driver) computes statistics upstream and threads them through. This is
a mechanical reconciliation, not a design choice — the journal entry
records it as such.

**Load-bearing constraints inherited from prior phases.**

The mirror is one-way. Phase 10 writes only under
``runs/{run_id}/mirror/stability.jsonl`` when invoked with an
``audit_jsonl_path``. No path writes to ``runs/{run_id}/telemetry/``,
``runs/{run_id}/checkpoints/``, or any Io-readable surface. The driver
does not construct an :class:`~kind.agents.actor.Actor`, a
:class:`~kind.agents.world_model.WorldModel`, or a
:class:`~kind.training.runner.Runner`.

The verbatim clauses appear unchanged in every paraphrase variant. The
paraphrase variants substitute only the per-criterion framing prose;
the header, the falsifier, the perturbation block, the signals block,
the non-falsifying-non-admission clause, the four exclusions, and the
sham-perturbation notice flow through unchanged (the prompt-builder
extension's ``framing_override`` parameter affects only the framing
slot).

Out of scope for Phase 10: any change to Io, the actor, the world
model, the dream state, or the runner; the consumption layer that
wires per-surface stability into per-criterion verdicts (Phase 11+);
automatic paraphrase generation (variants are checked in);
per-claim adaptive thresholding (per-surface is the granularity);
agreement metrics beyond Jaccard on structured-field tuples (a richer
metric is a Phase 11+ amendment).
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from kind.mirror.citation_canonical import canonicalize_scalar_field
from kind.mirror.llm_caller import (
    LLMClient,
    LLMConfig,
    LLMRecordSink,
    MirrorReading,
    PassRole,
    call_mirror_llm,
)
from kind.mirror.perturbation_align import PerturbationTimeline
from kind.mirror.prompt_builder import build_fragment
from kind.mirror.registry import Criterion, ReadingSurface
from kind.mirror.statistics import StatisticResult

__all__ = [
    "PARAPHRASE_VARIANTS_PER_SURFACE",
    "PARAPHRASE_THRESHOLDS",
    "RESEED_THRESHOLDS",
    "STABILITY_TEMPERATURE",
    "STABILITY_SEED_BASE",
    "STABILITY_N_PARAPHRASES_DEFAULT",
    "STABILITY_N_RESEEDS_DEFAULT",
    "StabilityResult",
    "stability_check",
]


# ---------------------------------------------------------------------------
# Module-level commitments.
# ---------------------------------------------------------------------------

#: The Gemini temperature for reseed-pass variation. The synthesis-default
#: exploratory setting; a future tune is journaled and changes this
#: single-source. Paraphrase-pass calls leave temperature at the LLM
#: caller's default (typically ≈ 0 on the structured-output path); only
#: reseeds drive temperature up.
STABILITY_TEMPERATURE: Final[float] = 0.7

#: The base seed for the reseed pass. The n-th reseed receives
#: ``STABILITY_SEED_BASE + n``. Deterministic and journaled. A future
#: tune to a different base is a single-source change here.
STABILITY_SEED_BASE: Final[int] = 1000

#: Default paraphrase count per surface. Matches the variant count in
#: :data:`PARAPHRASE_VARIANTS_PER_SURFACE`; a request for more variants
#: than committed raises at call time.
STABILITY_N_PARAPHRASES_DEFAULT: Final[int] = 3

#: Default reseed count. Per the synthesis. A future tune is journaled.
STABILITY_N_RESEEDS_DEFAULT: Final[int] = 3


#: Per-surface paraphrase variants. The three variants per surface are
#: conservative reframings: they vary the surrounding prose, not
#: substantive content. None of the variants contains or alters the
#: load-bearing verbatim clauses imported from
#: :mod:`kind.mirror.prompt_builder`. The structural test
#: ``test_paraphrase_variants_do_not_modify_verbatim_clauses`` enforces
#: this byte-by-byte against the committed clause text.
#:
#: A future amendment to the variant set requires a journal entry naming
#: the reason. Adding a fourth variant per surface, or replacing a
#: variant, both qualify as journaled amendments.
PARAPHRASE_VARIANTS_PER_SURFACE: Final[dict[ReadingSurface, tuple[str, str, str]]] = {
    ReadingSurface.SUBSTRATE_SIDE: (
        "Read the substrate-side telemetry directly. What does the data show?",
        "Approach the substrate-side reading as an observer of latent "
        "dynamics. What is present in the data?",
        "Consider the substrate-side measurements. What can be concluded "
        "from them?",
    ),
    ReadingSurface.HEAD_INTERNAL: (
        "Read the head-internal signals. What is the pattern?",
        "Inspect the head-internal latent structure. What does it reveal?",
        "Consider the head-internal measurements. What do they indicate?",
    ),
    ReadingSurface.BEHAVIOR_SIDE: (
        "Read the behavior-side traces. What does the action distribution "
        "show?",
        "Approach the behavior-side reading through observed actions. What "
        "is the pattern?",
        "Consider the behavior-side measurements. What does the policy "
        "indicate?",
    ),
}


#: Per-surface paraphrase-agreement thresholds. Per the synthesis §7
#: defaults (substrate-side 0.80, head-internal 0.80, behavior-side
#: 0.75). The synthesis flags these as "open during build" — the build
#: phase can tune empirically. Phase 10 commits the defaults; the smoke
#: result indicates whether they're right.
PARAPHRASE_THRESHOLDS: Final[dict[ReadingSurface, float]] = {
    ReadingSurface.SUBSTRATE_SIDE: 0.80,
    ReadingSurface.HEAD_INTERNAL: 0.80,
    ReadingSurface.BEHAVIOR_SIDE: 0.75,
}

#: Per-surface reseed-agreement thresholds. Same defaults as
#: :data:`PARAPHRASE_THRESHOLDS`. The synthesis flags both as "open
#: during build"; Phase 10 commits the defaults.
RESEED_THRESHOLDS: Final[dict[ReadingSurface, float]] = {
    ReadingSurface.SUBSTRATE_SIDE: 0.80,
    ReadingSurface.HEAD_INTERNAL: 0.80,
    ReadingSurface.BEHAVIOR_SIDE: 0.75,
}


# ---------------------------------------------------------------------------
# Result record.
# ---------------------------------------------------------------------------


class StabilityResult(BaseModel):
    """One stability-check outcome for one (criterion, role, checkpoint).

    Frozen, ``extra="forbid"``. The load-bearing outputs are
    :attr:`paraphrase_agreement_per_surface`,
    :attr:`reseed_agreement_per_surface`, and
    :attr:`admissible_per_surface`. The reading tuples carry the
    underlying readings for audit; the per-claim agreement tuple is
    informational. The wallclock duration is end-to-end.

    The per-surface dicts are keyed by :class:`ReadingSurface` enum
    members; the keys are exactly the criterion's
    :attr:`~kind.mirror.registry.Criterion.reading_surfaces`. A surface
    the criterion does not declare does not appear in either dict.

    Fields:

    - ``paraphrase_agreement_per_surface``: per-surface mean pairwise
      Jaccard across the paraphrase readings at that surface.
    - ``reseed_agreement_per_surface``: per-surface mean pairwise
      Jaccard across the reseed readings.
    - ``n_paraphrases``: paraphrase count per surface (matches the
      kwarg passed to :func:`stability_check`).
    - ``n_reseeds``: reseed count (matches the kwarg).
    - ``structured_field_agreement_per_claim``: a flat tuple of
      per-claim agreement scores derived from the paraphrase readings,
      in deterministic (surface-order, tuple-order) sequence. Each
      score is the fraction of paraphrase readings at the surface that
      contained the corresponding ``(cited_step_range,
      cited_scalar_field, cited_value)`` tuple. Informational.
    - ``admissible_per_surface``: per-surface admissibility verdict —
      ``True`` iff both paraphrase and reseed agreement at the surface
      meet their respective thresholds.
    - ``paraphrase_readings``: every paraphrase reading produced, in
      (surface-order, variant-index) sequence.
    - ``reseed_readings``: every reseed reading produced, in seed-order.
    - ``criterion_id``: the criterion this stability check was for.
    - ``reader_role``: ``"primary"`` or ``"adversarial"``.
    - ``run_id``: the run the readings were produced against.
    - ``checkpoint_id``: the checkpoint id within the run.
    - ``wallclock_ms``: end-to-end duration of the stability check.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    paraphrase_agreement_per_surface: dict[ReadingSurface, float]
    reseed_agreement_per_surface: dict[ReadingSurface, float]
    n_paraphrases: int
    n_reseeds: int
    structured_field_agreement_per_claim: tuple[float, ...]
    admissible_per_surface: dict[ReadingSurface, bool]
    paraphrase_readings: tuple[MirrorReading, ...]
    reseed_readings: tuple[MirrorReading, ...]
    criterion_id: str
    reader_role: PassRole
    run_id: str
    checkpoint_id: str
    wallclock_ms: int

    @field_validator("n_paraphrases", "n_reseeds")
    @classmethod
    def _validate_count_at_least_two(cls, value: int) -> int:
        if value < 2:
            raise ValueError(
                f"n_paraphrases / n_reseeds must be >= 2 for pairwise "
                f"agreement to be defined; got {value}."
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

    @field_validator("criterion_id", "run_id", "checkpoint_id")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty.")
        return value

    @model_validator(mode="after")
    def _validate_surface_keys_aligned(self) -> "StabilityResult":
        """The three per-surface dicts share the same key set — every
        surface appearing in ``paraphrase_agreement_per_surface`` also
        appears in ``reseed_agreement_per_surface`` and
        ``admissible_per_surface``. The driver constructs them together;
        the validator pins the alignment so a future refactor that
        forgets one of the three trips here."""
        p_keys = set(self.paraphrase_agreement_per_surface.keys())
        r_keys = set(self.reseed_agreement_per_surface.keys())
        a_keys = set(self.admissible_per_surface.keys())
        if not (p_keys == r_keys == a_keys):
            raise ValueError(
                f"StabilityResult per-surface dict keys are misaligned: "
                f"paraphrase={sorted(s.value for s in p_keys)}, "
                f"reseed={sorted(s.value for s in r_keys)}, "
                f"admissible={sorted(s.value for s in a_keys)}. The three "
                f"dicts should share the same key set (the criterion's "
                f"reading_surfaces)."
            )
        return self


# ---------------------------------------------------------------------------
# Helpers — per-claim tuple extraction, pairwise Jaccard.
# ---------------------------------------------------------------------------


# Type alias for the structured-field tuple used as the unit of agreement.
# Order matches the synthesis's "structured-field tuple" naming:
# (cited_step_range, cited_scalar_field, cited_value). ``cited_step_range``
# is ``None`` for claims that scoped to an episode range only; ``None``
# hashes and compares cleanly inside the tuple.
_ClaimFieldTuple = tuple[tuple[int, int] | None, str, float]


def _claim_tuples_at_surface(
    reading: MirrorReading,
    surface: ReadingSurface,
    known_signal_names: frozenset[str],
) -> frozenset[_ClaimFieldTuple]:
    """Extract the structured-field tuples for one reading's claims at
    one surface. The ``reading_surface`` field on
    :class:`~kind.mirror.structured.StructuredClaim` is the Phase 0
    Literal whose string values match the
    :class:`~kind.mirror.registry.ReadingSurface` enum's str values;
    ``c.reading_surface == surface`` therefore matches via the
    str-valued-enum equality contract.

    Phase 12.5 item 4: the ``cited_scalar_field`` component is
    canonicalized via
    :func:`~kind.mirror.citation_canonical.canonicalize_scalar_field`
    before it enters the tuple, so the same signal cited bare in one
    reading and compound in another contributes the same tuple to the
    Jaccard set. ``known_signal_names`` comes from the criterion's
    :class:`~kind.mirror.registry.SignalMapping` declarations.
    """
    return frozenset(
        (
            c.cited_step_range,
            canonicalize_scalar_field(c.cited_scalar_field, known_signal_names),
            c.cited_value,
        )
        for c in reading.claims
        if c.reading_surface == surface
    )


def _pairwise_jaccard_mean(
    tuple_sets: list[frozenset[_ClaimFieldTuple]],
) -> float:
    """Mean Jaccard across all unique unordered pairs of tuple sets.

    Convention: empty-vs-empty pairs contribute 1.0 (no claims at this
    surface, in either reading — consistent emptiness, not
    disagreement); one-empty-one-nonempty pairs contribute 0; the
    standard ``|A ∩ B| / |A ∪ B|`` applies elsewhere.

    Pre: ``len(tuple_sets) >= 2``. The driver enforces this via the
    ``n_paraphrases`` / ``n_reseeds`` validator on
    :class:`StabilityResult`.
    """
    if len(tuple_sets) < 2:  # pragma: no cover — driver guards
        raise ValueError(
            f"_pairwise_jaccard_mean requires >= 2 sets; got "
            f"{len(tuple_sets)}."
        )
    scores: list[float] = []
    for a, b in itertools.combinations(tuple_sets, 2):
        union = a | b
        if not union:
            scores.append(1.0)
            continue
        intersection = a & b
        scores.append(len(intersection) / len(union))
    return sum(scores) / len(scores)


def _structured_field_agreement_for_surface(
    tuple_sets: list[frozenset[_ClaimFieldTuple]],
) -> list[float]:
    """Per-tuple agreement = fraction of readings containing the tuple.

    For each unique tuple across the readings, count how many readings
    contain it; divide by the number of readings. The result is one
    agreement score per unique tuple. Order: ``sorted(all_tuples,
    key=str)`` for determinism (tuple comparison with ``None``
    components is awkward without a key function).
    """
    if not tuple_sets:
        return []
    all_tuples: set[_ClaimFieldTuple] = set()
    for ts in tuple_sets:
        all_tuples |= ts
    if not all_tuples:
        return []
    sorted_tuples = sorted(all_tuples, key=str)
    n = len(tuple_sets)
    return [sum(1 for ts in tuple_sets if t in ts) / n for t in sorted_tuples]


def _surfaces_for(criterion: Criterion) -> tuple[ReadingSurface, ...]:
    """Iterate the criterion's declared surfaces in the enum's
    definition order (substrate → head-internal → behavior). The
    enum's definition order is deterministic across Python runs."""
    return tuple(s for s in ReadingSurface if s in criterion.reading_surfaces)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def stability_check(
    role: PassRole,
    criterion: Criterion,
    statistic_results: tuple[StatisticResult, ...],
    perturbation_timeline: PerturbationTimeline | None = None,
    *,
    run_id: str,
    checkpoint_id: str,
    digest_run_id: str,
    digest_episode_range: tuple[int, int],
    llm_config: LLMConfig,
    n_paraphrases: int = STABILITY_N_PARAPHRASES_DEFAULT,
    n_reseeds: int = STABILITY_N_RESEEDS_DEFAULT,
    seed_base: int = STABILITY_SEED_BASE,
    temperature: float = STABILITY_TEMPERATURE,
    llm_client: LLMClient | None = None,
    record_sink: LLMRecordSink | None = None,
    audit_jsonl_path: Path | None = None,
) -> StabilityResult:
    """Run a stability check for one (criterion, role) at one checkpoint.

    Logic:

    1. **Paraphrase pass.** For each surface ``s`` in the criterion's
       :attr:`~kind.mirror.registry.Criterion.reading_surfaces`, issue
       ``n_paraphrases`` calls to
       :func:`~kind.mirror.llm_caller.call_mirror_llm`, each with
       ``framing_override`` set to one of
       :data:`PARAPHRASE_VARIANTS_PER_SURFACE` ``[s]``. Reading-surfaces
       are iterated in :class:`ReadingSurface` definition order
       (substrate-side, head-internal, behavior-side).

    2. **Reseed pass.** Issue ``n_reseeds`` calls at the default
       framing (no ``framing_override``) with
       ``temperature=temperature`` and ``seed=seed_base + i`` for
       ``i in range(n_reseeds)``.

    3. **Per-surface agreement.** For each declared surface, filter
       each reading's claims to claims with ``reading_surface == s``;
       compute the mean pairwise Jaccard across the per-surface
       paraphrase readings (n_paraphrases readings) and separately
       across the reseed readings (n_reseeds readings).

    4. **Admissibility.** A surface is admissible iff its paraphrase
       agreement meets :data:`PARAPHRASE_THRESHOLDS` ``[s]`` AND its
       reseed agreement meets :data:`RESEED_THRESHOLDS` ``[s]``.

    5. **Audit.** If ``audit_jsonl_path`` is provided, append the
       :class:`StabilityResult` as a single JSONL line to that path.
       The caller is responsible for the path's containing directory
       living under ``runs/{run_id}/mirror/``; the driver does not
       enforce this so the one-way-write invariant is preserved at the
       call site (the directory choice is the caller's commitment).

    **Total LLM call count.** ``n_paraphrases × |reading_surfaces| +
    n_reseeds``. For the default 3 / 3 with a three-surface criterion
    that's 12 calls. For a one-surface criterion (reflexive_attention,
    declaring only ``HEAD_INTERNAL``) that's 6 calls.

    Raises :class:`ValueError` if ``n_paraphrases`` or ``n_reseeds`` is
    < 2 (pairwise comparison requires at least two readings), or if
    ``n_paraphrases`` exceeds the committed variant count at any of
    the criterion's surfaces.
    """
    if n_paraphrases < 2:
        raise ValueError(
            f"stability_check: n_paraphrases must be >= 2 for pairwise "
            f"agreement; got {n_paraphrases}."
        )
    if n_reseeds < 2:
        raise ValueError(
            f"stability_check: n_reseeds must be >= 2 for pairwise "
            f"agreement; got {n_reseeds}."
        )
    surfaces = _surfaces_for(criterion)
    for s in surfaces:
        if n_paraphrases > len(PARAPHRASE_VARIANTS_PER_SURFACE[s]):
            raise ValueError(
                f"stability_check: n_paraphrases={n_paraphrases} exceeds "
                f"the {len(PARAPHRASE_VARIANTS_PER_SURFACE[s])} variants "
                f"committed for surface {s.value!r} in "
                f"PARAPHRASE_VARIANTS_PER_SURFACE. Either commit more "
                f"variants (journaled amendment) or lower n_paraphrases."
            )

    start_ms = int(time.time() * 1000)

    # 1. Paraphrase pass.
    paraphrase_readings_per_surface: dict[ReadingSurface, list[MirrorReading]] = {}
    for s in surfaces:
        readings_at_s: list[MirrorReading] = []
        for variant in PARAPHRASE_VARIANTS_PER_SURFACE[s][:n_paraphrases]:
            fragment = build_fragment(
                criterion,
                statistic_results,
                perturbation_timeline=perturbation_timeline,
                framing_override=variant,
            )
            batch = call_mirror_llm(
                (fragment,),
                role=role,
                config=llm_config,
                run_id=run_id,
                digest_run_id=digest_run_id,
                digest_episode_range=digest_episode_range,
                client=llm_client,
                record_sink=record_sink,
            )
            readings_at_s.append(batch[0])
        paraphrase_readings_per_surface[s] = readings_at_s

    # 2. Reseed pass.
    reseed_readings_list: list[MirrorReading] = []
    for i in range(n_reseeds):
        fragment = build_fragment(
            criterion,
            statistic_results,
            perturbation_timeline=perturbation_timeline,
        )
        batch = call_mirror_llm(
            (fragment,),
            role=role,
            config=llm_config,
            run_id=run_id,
            digest_run_id=digest_run_id,
            digest_episode_range=digest_episode_range,
            client=llm_client,
            record_sink=record_sink,
            seed=seed_base + i,
            temperature=temperature,
        )
        reseed_readings_list.append(batch[0])

    # 3. Per-surface agreement + 4. admissibility.
    paraphrase_agreement: dict[ReadingSurface, float] = {}
    reseed_agreement: dict[ReadingSurface, float] = {}
    admissible: dict[ReadingSurface, bool] = {}
    sf_agreement_flat: list[float] = []

    # Phase 12.5 item 4: the known signal names for the iterated
    # canonical-form rule come from the criterion's SignalMapping
    # declarations — the canonical form is well-defined whether or not
    # a statistic was actually computed for every declared signal.
    known_signal_names = frozenset(
        m.name for m in criterion.signal_mappings
    )

    for s in surfaces:
        p_tuple_sets = [
            _claim_tuples_at_surface(r, s, known_signal_names)
            for r in paraphrase_readings_per_surface[s]
        ]
        r_tuple_sets = [
            _claim_tuples_at_surface(r, s, known_signal_names)
            for r in reseed_readings_list
        ]
        paraphrase_agreement[s] = _pairwise_jaccard_mean(p_tuple_sets)
        reseed_agreement[s] = _pairwise_jaccard_mean(r_tuple_sets)
        admissible[s] = (
            paraphrase_agreement[s] >= PARAPHRASE_THRESHOLDS[s]
            and reseed_agreement[s] >= RESEED_THRESHOLDS[s]
        )
        sf_agreement_flat.extend(
            _structured_field_agreement_for_surface(p_tuple_sets)
        )

    # Flatten paraphrase readings in surface-order, variant-order.
    paraphrase_readings_flat: list[MirrorReading] = []
    for s in surfaces:
        paraphrase_readings_flat.extend(paraphrase_readings_per_surface[s])

    end_ms = int(time.time() * 1000)
    wallclock_ms = max(end_ms - start_ms, 0)

    result = StabilityResult(
        paraphrase_agreement_per_surface=paraphrase_agreement,
        reseed_agreement_per_surface=reseed_agreement,
        n_paraphrases=n_paraphrases,
        n_reseeds=n_reseeds,
        structured_field_agreement_per_claim=tuple(sf_agreement_flat),
        admissible_per_surface=admissible,
        paraphrase_readings=tuple(paraphrase_readings_flat),
        reseed_readings=tuple(reseed_readings_list),
        criterion_id=criterion.id,
        reader_role=role,
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        wallclock_ms=wallclock_ms,
    )

    if audit_jsonl_path is not None:
        # Append one JSONL line. The caller chose the path; the driver
        # writes only what it was told to. The directory is the
        # caller's commitment.
        line = result.model_dump_json()
        with audit_jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.write("\n")

    return result
