"""Phase 11 faithfulness-verifier module.

Phase 11 is the second half of the admissibility-gate substrate. Phase
10's :mod:`~kind.mirror.stability` asks *is this reading stable under
paraphrase and reseed variation?* Phase 11's :func:`verify_reading`
asks *does this reading actually trace back to the data?* The two
verifiers measure different axes and produce independent records; a
future admissibility consumer combines them.

**What the verifier does.** For each :class:`~kind.mirror.structured.StructuredClaim`
in a :class:`~kind.mirror.llm_caller.MirrorReading`, the verifier
resolves the citation — ``cited_step_range``, ``cited_scalar_field``,
``cited_value`` — against the
:class:`~kind.mirror.statistics.StatisticResult` records the LLM was
shown by the prompt-builder. The verdict per claim is one of:

- :attr:`FaithfulnessStatus.RESOLVED` — the citation traces.
- :attr:`FaithfulnessStatus.UNRESOLVED_FIELD` — the cited scalar field
  (after canonical-form normalization) does not match any statistic
  result's :attr:`~kind.mirror.statistics.StatisticResult.signal_name`.
- :attr:`FaithfulnessStatus.UNRESOLVED_VALUE` — the field matches but
  the cited value does not match the statistic result within
  ``abs(diff) < 1e-6`` tolerance.
- :attr:`FaithfulnessStatus.UNRESOLVED_RANGE` — the cited step range
  falls outside the statistic result's representable range. Only fires
  for trajectory-valued (``list[float]``) statistics with non-``None``
  cited step ranges; for scalar and dict-valued statistics the cited
  step range is informational and never trips this verdict.

The verifier overwrites whatever the LLM wrote into the claim's
``faithfulness_status`` field; the verifier owns the verdict, not the
reader. The prompt-builder amendment to prevent the LLM from filling
the field at all is journaled separately and not Phase 11's job.

**The canonical-form rule.** Phase 10's stability smoke surfaced the
LLM constructing ``cited_scalar_field`` in multiple equally-faithful
forms — ``latent_self_reference_t`` (bare) and
``latent_self_reference_t.partial_autocorr_lag5`` (compound with an
estimator-name suffix) for the same signal. Phase 13's production
calibration data adds another compound form — ``policy_entropy_t.no_response``
(compound with a dict-key suffix), where the signal's
:class:`~kind.mirror.statistics.StatisticResult` carries a
``dict[str, float]`` value and the suffix names a specific dict key.

Phase 12.5 lifted :func:`~kind.mirror.citation_canonical.canonicalize_scalar_field`
into the shared :mod:`kind.mirror.citation_canonical` module and made
it iterated: it strips suffixes until a prefix matches a known
statistic signal name, falling back to the bare (no-dot) form. The
verifier passes the signal names of its
:class:`~kind.mirror.statistics.StatisticResult` records as the known
set, and additionally remembers the original trailing suffix (see
:func:`_suffix_after_canonical`) so compound citations into
``dict[str, float]`` results resolve against the named dict key.

**Threshold commitment.** :data:`FAITHFULNESS_THRESHOLD` is committed
at ``0.80``. The reasoning: faithfulness is stricter than stability —
a reading where 20% of the LLM's citations don't resolve is genuinely
suspect, because a citation that doesn't trace back to the prompt's
signals block is fabricated rather than just paraphrased. The
synthesis convention for per-surface stability thresholds is also
``0.80`` (substrate-side, head-internal); the verifier inherits the
convention here at the per-reading granularity.

**One-way invariant.** The verifier reads
:class:`~kind.mirror.statistics.StatisticResult` records and a
:class:`~kind.mirror.llm_caller.MirrorReading`, both already on the
mirror's side of the membrane. It writes only to its return value and
(optionally) to a caller-provided audit JSONL path. No path writes to
``runs/{run_id}/telemetry/``, ``runs/{run_id}/checkpoints/``, or any
other Io-readable surface.

Out of scope for Phase 11: the admissibility consumer that joins
faithfulness with Phase 10's per-surface stability (future phase); the
prompt-builder amendment that removes the LLM's
``faithfulness_status`` write surface (deferred, journaled); a richer
canonical-form rule beyond the single-suffix strip (a Phase 12+
amendment with empirical evidence in hand); any change to Io, the
actor, the world model, the dream state, or the runner.
"""

from __future__ import annotations

import time
from enum import Enum
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from kind.mirror.citation_canonical import canonicalize_scalar_field
from kind.mirror.llm_caller import MirrorReading, PassRole
from kind.mirror.statistics import StatisticResult

__all__ = [
    "FAITHFULNESS_THRESHOLD",
    "FAITHFULNESS_VALUE_TOLERANCE",
    "FaithfulnessAssignment",
    "FaithfulnessResult",
    "FaithfulnessStatus",
    "canonicalize_scalar_field",
    "verify_reading",
]


# ---------------------------------------------------------------------------
# Module-level commitments.
# ---------------------------------------------------------------------------

#: Per-reading admissibility threshold on the faithfulness rate. A
#: reading is :attr:`FaithfulnessResult.admissible` iff its
#: :attr:`FaithfulnessResult.faithfulness_rate` meets or exceeds this
#: constant. Phase 11 commits ``0.80`` — matching the synthesis-default
#: convention for per-surface stability thresholds, and stricter than a
#: looser ``0.70`` would be because a fabricated citation is a different
#: kind of failure than a paraphrase-drift.
FAITHFULNESS_THRESHOLD: Final[float] = 0.80

#: Float-comparison tolerance for resolving
#: :attr:`StructuredClaim.cited_value` against a statistic result. Phase
#: 11 commits ``1e-6`` per the spec; a future tune is journaled and
#: single-source. The tolerance applies to both scalar
#: (:class:`float`) and list-membership (``list[float]``) comparisons.
FAITHFULNESS_VALUE_TOLERANCE: Final[float] = 1e-6


# ---------------------------------------------------------------------------
# The verdict enum.
# ---------------------------------------------------------------------------


class FaithfulnessStatus(str, Enum):
    """One of four verifier verdicts on a single
    :class:`~kind.mirror.structured.StructuredClaim`.

    The enum is intentionally distinct from
    :data:`kind.mirror.structured.FaithfulnessStatus` (the writer-side
    Literal the LLM may pre-fill into the claim's
    ``faithfulness_status`` field). The Phase 0 writer-side literal
    carries the four values ``{"resolved", "off_by_tolerance",
    "unresolved", "not_checked"}``; Phase 11's verifier verdict is more
    specific — it distinguishes *why* an unresolved citation didn't
    resolve, which is what the downstream admissibility consumer needs.

    The values are lowercase strings; ``(str, Enum)`` gives JSON-friendly
    serialization (``model_dump_json`` emits the string form) without
    relying on Pydantic's enum-value mode setting.
    """

    RESOLVED = "resolved"
    UNRESOLVED_FIELD = "unresolved_field"
    UNRESOLVED_VALUE = "unresolved_value"
    UNRESOLVED_RANGE = "unresolved_range"


# ---------------------------------------------------------------------------
# Result records.
# ---------------------------------------------------------------------------


class FaithfulnessAssignment(BaseModel):
    """Per-claim faithfulness verdict.

    Frozen, ``extra="forbid"``. One assignment per
    :class:`~kind.mirror.structured.StructuredClaim` in the source
    reading; the assignments tuple on
    :class:`FaithfulnessResult` is the same length as the source
    reading's claims list.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    pass_index: int
    criterion_id: str
    reader_role: PassRole
    claim_index: int
    cited_scalar_field_original: str
    cited_scalar_field_canonical: str
    cited_value: float
    cited_step_range: tuple[int, int] | None
    status: FaithfulnessStatus
    resolution_notes: str

    @field_validator("pass_index", "claim_index")
    @classmethod
    def _validate_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"value must be >= 0; got {value}.")
        return value

    @field_validator("criterion_id", "cited_scalar_field_original")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty.")
        return value

    @field_validator("resolution_notes")
    @classmethod
    def _validate_resolution_notes_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "FaithfulnessAssignment.resolution_notes must be non-empty: "
                "the verifier always names which statistic result was "
                "matched (or which wasn't) and why the status landed where "
                "it did."
            )
        return value


class FaithfulnessResult(BaseModel):
    """Per-reading faithfulness verdict.

    Frozen, ``extra="forbid"``. The load-bearing outputs are
    :attr:`faithfulness_rate` and :attr:`admissible`; the assignments
    tuple carries the per-claim record for audit and for the downstream
    admissibility consumer.

    The aggregate counts (:attr:`n_resolved`,
    :attr:`n_unresolved_field`, :attr:`n_unresolved_value`,
    :attr:`n_unresolved_range`) are validator-pinned to sum to
    :attr:`n_claims_total`. A reading with zero claims is allowed —
    :attr:`n_claims_total` is ``0``, all the counts are ``0``,
    :attr:`faithfulness_rate` is ``1.0`` (vacuously faithful), and
    :attr:`admissible` is ``True``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion_id: str
    reader_role: PassRole
    pass_index: int
    run_id: str
    checkpoint_id: str
    assignments: tuple[FaithfulnessAssignment, ...]
    n_claims_total: int
    n_resolved: int
    n_unresolved_field: int
    n_unresolved_value: int
    n_unresolved_range: int
    faithfulness_rate: float
    admissible: bool
    wallclock_ms: int
    notes: str

    @field_validator("pass_index")
    @classmethod
    def _validate_pass_index_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"pass_index must be >= 0; got {value}.")
        return value

    @field_validator(
        "n_claims_total",
        "n_resolved",
        "n_unresolved_field",
        "n_unresolved_value",
        "n_unresolved_range",
        "wallclock_ms",
    )
    @classmethod
    def _validate_count_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"value must be >= 0; got {value}.")
        return value

    @field_validator("criterion_id", "run_id", "checkpoint_id")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty.")
        return value

    @field_validator("faithfulness_rate")
    @classmethod
    def _validate_rate_in_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"faithfulness_rate must be in [0.0, 1.0]; got {value}."
            )
        return value

    @model_validator(mode="after")
    def _validate_counts_sum(self) -> "FaithfulnessResult":
        total = (
            self.n_resolved
            + self.n_unresolved_field
            + self.n_unresolved_value
            + self.n_unresolved_range
        )
        if total != self.n_claims_total:
            raise ValueError(
                f"FaithfulnessResult counts do not sum to n_claims_total: "
                f"n_resolved={self.n_resolved}, "
                f"n_unresolved_field={self.n_unresolved_field}, "
                f"n_unresolved_value={self.n_unresolved_value}, "
                f"n_unresolved_range={self.n_unresolved_range}, "
                f"sum={total}, n_claims_total={self.n_claims_total}."
            )
        if len(self.assignments) != self.n_claims_total:
            raise ValueError(
                f"FaithfulnessResult.assignments has length "
                f"{len(self.assignments)} but n_claims_total is "
                f"{self.n_claims_total}; they must match."
            )
        return self


# ---------------------------------------------------------------------------
# Canonical-form normalization.
# ---------------------------------------------------------------------------


def _suffix_after_canonical(field: str) -> str | None:
    """Return the last dot-separated segment of ``field`` — the dict-key
    candidate — or ``None`` if the input has no dot.

    Used by the verifier to look up a specific key in a
    ``dict[str, float]`` statistic value when the LLM cited a compound
    form like ``"policy_entropy_t.no_response"``. The dict keys are
    single tokens (``no_response``, ``collapse``); taking the last
    segment yields the key even for an iterated two-dot citation like
    ``"policy_entropy_t.classification.collapse"`` (→ ``"collapse"``).
    """
    if "." not in field:
        return None
    return field.rsplit(".", 1)[1]


# ---------------------------------------------------------------------------
# The verifier.
# ---------------------------------------------------------------------------


def _find_statistic_by_signal_name(
    statistic_results: tuple[StatisticResult, ...] | list[StatisticResult],
    canonical_field: str,
) -> StatisticResult | None:
    """Return the first statistic result whose
    :attr:`~kind.mirror.statistics.StatisticResult.signal_name` equals
    ``canonical_field``, or ``None`` if none match.

    The orchestrator emits one
    :class:`~kind.mirror.statistics.StatisticResult` per
    :class:`~kind.mirror.registry.SignalMapping`; the signal names are
    unique across the tuple in practice. The "first match" semantics is
    deterministic regardless.
    """
    for result in statistic_results:
        if result.signal_name == canonical_field:
            return result
    return None


def _resolve_against_float(
    statistic_value: float,
    cited_value: float,
    tolerance: float,
) -> tuple[FaithfulnessStatus, str]:
    """Resolve a citation against a scalar (:class:`float`)
    statistic value. Returns ``(status, notes)``.
    """
    diff = abs(statistic_value - cited_value)
    if diff < tolerance:
        return (
            FaithfulnessStatus.RESOLVED,
            f"scalar statistic value {statistic_value!r} matches cited value "
            f"{cited_value!r} within tolerance {tolerance} (|diff|={diff!r}).",
        )
    return (
        FaithfulnessStatus.UNRESOLVED_VALUE,
        f"scalar statistic value {statistic_value!r} does not match cited "
        f"value {cited_value!r} within tolerance {tolerance} "
        f"(|diff|={diff!r}).",
    )


def _resolve_against_list(
    statistic_value: list[float],
    cited_value: float,
    cited_step_range: tuple[int, int] | None,
    tolerance: float,
) -> tuple[FaithfulnessStatus, str]:
    """Resolve a citation against a list (``list[float]``) statistic
    value. Returns ``(status, notes)``.

    If ``cited_step_range`` is non-``None``, the verifier extracts the
    sub-range ``statistic_value[start:end+1]`` and checks whether
    ``cited_value`` appears within it (within tolerance). If the cited
    step range falls outside the list's length the verdict is
    :attr:`FaithfulnessStatus.UNRESOLVED_RANGE`. If the range is
    ``None`` the verifier checks membership across the whole list.
    """
    if cited_step_range is not None:
        start, end = cited_step_range
        if start < 0 or end < start or end >= len(statistic_value):
            return (
                FaithfulnessStatus.UNRESOLVED_RANGE,
                f"cited_step_range={cited_step_range!r} falls outside the "
                f"trajectory's [0, {len(statistic_value) - 1}] index range "
                f"(list length {len(statistic_value)}).",
            )
        sub = statistic_value[start : end + 1]
    else:
        sub = list(statistic_value)
    for v in sub:
        if abs(v - cited_value) < tolerance:
            return (
                FaithfulnessStatus.RESOLVED,
                f"cited value {cited_value!r} appears in the trajectory's "
                f"step-range slice (length {len(sub)}) within tolerance "
                f"{tolerance}.",
            )
    return (
        FaithfulnessStatus.UNRESOLVED_VALUE,
        f"cited value {cited_value!r} does not appear in the trajectory's "
        f"step-range slice (length {len(sub)}) within tolerance {tolerance}.",
    )


def _resolve_against_dict(
    statistic_value: dict[str, float],
    cited_value: float,
    suffix: str | None,
    tolerance: float,
) -> tuple[FaithfulnessStatus, str]:
    """Resolve a citation against a dict (``dict[str, float]``)
    statistic value. Returns ``(status, notes)``.

    The expected production pattern is the LLM citing a compound
    ``cited_scalar_field`` like ``"policy_entropy_t.no_response"``,
    where the suffix names a specific dict key. The verifier looks up
    ``statistic_value[suffix]`` and compares to ``cited_value`` with
    tolerance.

    When no suffix is present (the LLM cited the bare field name with a
    dict-valued statistic), the verifier falls back to membership
    across all dict values — a permissive resolution because the LLM
    may have aggregated across keys in some way.
    """
    keys_repr = ", ".join(sorted(statistic_value.keys()))
    if suffix is not None:
        if suffix not in statistic_value:
            return (
                FaithfulnessStatus.UNRESOLVED_VALUE,
                f"dict statistic does not contain key {suffix!r}; available "
                f"keys: {{{keys_repr}}}. Cited value {cited_value!r}.",
            )
        v = statistic_value[suffix]
        if abs(v - cited_value) < tolerance:
            return (
                FaithfulnessStatus.RESOLVED,
                f"dict statistic key {suffix!r} value {v!r} matches cited "
                f"value {cited_value!r} within tolerance {tolerance}.",
            )
        return (
            FaithfulnessStatus.UNRESOLVED_VALUE,
            f"dict statistic key {suffix!r} value {v!r} does not match "
            f"cited value {cited_value!r} within tolerance {tolerance}.",
        )
    for v in statistic_value.values():
        if abs(v - cited_value) < tolerance:
            return (
                FaithfulnessStatus.RESOLVED,
                f"cited value {cited_value!r} appears among the dict "
                f"statistic's values (keys: {{{keys_repr}}}) within "
                f"tolerance {tolerance}; the citation used the bare signal "
                f"name without a key suffix.",
            )
    return (
        FaithfulnessStatus.UNRESOLVED_VALUE,
        f"cited value {cited_value!r} does not appear among the dict "
        f"statistic's values (keys: {{{keys_repr}}}) within tolerance "
        f"{tolerance}; no key suffix on the citation.",
    )


def verify_reading(
    reading: MirrorReading,
    statistic_results: tuple[StatisticResult, ...] | list[StatisticResult],
    *,
    criterion_id: str,
    pass_index: int,
    run_id: str,
    checkpoint_id: str,
    audit_jsonl_path: Path | None = None,
) -> FaithfulnessResult:
    """Verify each claim in ``reading`` against ``statistic_results``.

    For each :class:`~kind.mirror.structured.StructuredClaim` in
    ``reading.claims``:

    1. Compute the canonical form of the claim's ``cited_scalar_field``
       via :func:`canonicalize_scalar_field`.
    2. Look up the matching
       :class:`~kind.mirror.statistics.StatisticResult` by
       :attr:`~kind.mirror.statistics.StatisticResult.signal_name`. If
       none match → :attr:`FaithfulnessStatus.UNRESOLVED_FIELD`.
    3. If matched, dispatch on the statistic value's runtime type:

       - :class:`float` → tolerance check on cited value;
       - ``list[float]`` → if ``cited_step_range`` is non-``None`` and
         falls outside ``[0, len-1]``, the verdict is
         :attr:`FaithfulnessStatus.UNRESOLVED_RANGE`; otherwise
         membership check (within tolerance) over the sub-range;
       - ``dict[str, float]`` → look up the suffix key if the original
         citation was compound; otherwise membership check across dict
         values.

    The verifier overwrites whatever the LLM wrote into the claim's
    ``faithfulness_status`` field; the verdict in
    :attr:`FaithfulnessAssignment.status` is the canonical record.

    ``criterion_id`` is taken as a kwarg because the
    :class:`~kind.mirror.structured.StructuredReading` envelope does
    not declare a ``criterion_id`` field. The Phase 8 per-criterion
    payload does carry the id, but it's stripped when the payload is
    wrapped into the envelope. The orchestrator (which built the
    per-criterion prompt fragment in the first place) is the
    authoritative source; the verifier requires the caller to pass it
    along.

    ``reader_role`` is derived from the reading's own
    :attr:`~kind.mirror.structured.StructuredReading.reader_role`
    (``"advocate"`` / ``"skeptic"`` / ``"single"``) via the
    :data:`~kind.mirror.llm_caller.PassRole` mapping
    (``"advocate"`` and ``"single"`` → ``"primary"``;
    ``"skeptic"`` → ``"adversarial"``).

    If ``audit_jsonl_path`` is provided, the :class:`FaithfulnessResult`
    is appended as a single JSONL line to that path. The caller is
    responsible for the path's containing directory living under
    ``runs/{run_id}/mirror/``; the one-way-write invariant is preserved
    at the call site by the caller's directory choice.

    The verifier is pure aside from the optional audit emission. No LLM
    calls; no statistic recomputation; no reads against
    :class:`~kind.observer.schemas.AgentStep` or telemetry files. The
    input ``reading`` and ``statistic_results`` are not modified.
    """
    start_ms = int(time.time() * 1000)
    reader_role = _pass_role_for(reading)

    # The known signal names for the iterated canonical-form rule come
    # from the statistic results the verifier was handed (Phase 12.5
    # item 3).
    known_signal_names = frozenset(
        r.signal_name for r in statistic_results
    )

    assignments: list[FaithfulnessAssignment] = []
    n_resolved = 0
    n_unresolved_field = 0
    n_unresolved_value = 0
    n_unresolved_range = 0

    for claim_index, claim in enumerate(reading.claims):
        original = claim.cited_scalar_field
        canonical = canonicalize_scalar_field(original, known_signal_names)
        suffix = _suffix_after_canonical(original)

        statistic = _find_statistic_by_signal_name(
            statistic_results, canonical
        )
        if statistic is None:
            available = ", ".join(
                sorted(r.signal_name for r in statistic_results)
            )
            status = FaithfulnessStatus.UNRESOLVED_FIELD
            notes = (
                f"no statistic result has signal_name matching the canonical "
                f"form {canonical!r} (original cited_scalar_field "
                f"{original!r}). Available signal names: {{{available}}}."
            )
        else:
            value = statistic.value
            if isinstance(value, dict):
                status, notes = _resolve_against_dict(
                    value, claim.cited_value, suffix,
                    FAITHFULNESS_VALUE_TOLERANCE,
                )
            elif isinstance(value, list):
                status, notes = _resolve_against_list(
                    value, claim.cited_value, claim.cited_step_range,
                    FAITHFULNESS_VALUE_TOLERANCE,
                )
            else:
                status, notes = _resolve_against_float(
                    float(value), claim.cited_value,
                    FAITHFULNESS_VALUE_TOLERANCE,
                )

        if status is FaithfulnessStatus.RESOLVED:
            n_resolved += 1
        elif status is FaithfulnessStatus.UNRESOLVED_FIELD:
            n_unresolved_field += 1
        elif status is FaithfulnessStatus.UNRESOLVED_VALUE:
            n_unresolved_value += 1
        else:
            n_unresolved_range += 1

        assignments.append(
            FaithfulnessAssignment(
                pass_index=pass_index,
                criterion_id=criterion_id,
                reader_role=reader_role,
                claim_index=claim_index,
                cited_scalar_field_original=original,
                cited_scalar_field_canonical=canonical,
                cited_value=claim.cited_value,
                cited_step_range=claim.cited_step_range,
                status=status,
                resolution_notes=notes,
            )
        )

    n_total = len(reading.claims)
    if n_total == 0:
        rate = 1.0
    else:
        rate = n_resolved / n_total

    admissible = rate >= FAITHFULNESS_THRESHOLD

    end_ms = int(time.time() * 1000)
    wallclock_ms = max(end_ms - start_ms, 0)

    result = FaithfulnessResult(
        criterion_id=criterion_id,
        reader_role=reader_role,
        pass_index=pass_index,
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        assignments=tuple(assignments),
        n_claims_total=n_total,
        n_resolved=n_resolved,
        n_unresolved_field=n_unresolved_field,
        n_unresolved_value=n_unresolved_value,
        n_unresolved_range=n_unresolved_range,
        faithfulness_rate=rate,
        admissible=admissible,
        wallclock_ms=wallclock_ms,
        notes=(
            f"verified {n_total} claim(s) against "
            f"{len(statistic_results)} statistic result(s); "
            f"rate={rate:.4f}; threshold={FAITHFULNESS_THRESHOLD}."
        ),
    )

    if audit_jsonl_path is not None:
        line = result.model_dump_json()
        with audit_jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.write("\n")

    return result


# ---------------------------------------------------------------------------
# Reader-role mapping.
# ---------------------------------------------------------------------------


def _pass_role_for(reading: MirrorReading) -> PassRole:
    """Map the reading's
    :attr:`~kind.mirror.structured.StructuredReading.reader_role`
    (``"advocate"`` / ``"skeptic"`` / ``"single"``) to the
    :data:`~kind.mirror.llm_caller.PassRole` literal the result carries
    (``"primary"`` / ``"adversarial"``). The mapping mirrors
    :data:`kind.mirror.llm_caller._ROLE_TO_READER_ROLE` in reverse;
    ``"single"`` (Probe 1's legacy slot) maps to ``"primary"`` so a
    legacy reading still produces a well-formed verdict.
    """
    if reading.reader_role == "skeptic":
        return "adversarial"
    return "primary"
