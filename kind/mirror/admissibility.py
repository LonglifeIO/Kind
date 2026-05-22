"""Phase 12 admissibility-consumer module.

Phases 10 and 11 built the two halves of the admissibility-gate
substrate as *independent producers*. Phase 10's
:mod:`~kind.mirror.stability` asks *is this reading stable under
paraphrase and reseed variation?* and emits a
:class:`~kind.mirror.stability.StabilityResult` per (criterion, role,
checkpoint). Phase 11's :mod:`~kind.mirror.faithfulness` asks *do this
reading's citations trace back to the data?* and emits a
:class:`~kind.mirror.faithfulness.FaithfulnessResult` per reading. The
synthesis named both verifiers feeding a future consumer; Phase 12 is
that consumer.

**The join rule — A (AND-conjunction at the reading level).** A reading
is :attr:`AdmissibilityVerdict.admissible` iff its faithfulness rate
cleared :data:`~kind.mirror.faithfulness.FAITHFULNESS_THRESHOLD` AND
every declared reading surface cleared both
:data:`~kind.mirror.stability.PARAPHRASE_THRESHOLDS` and
:data:`~kind.mirror.stability.RESEED_THRESHOLDS`. The consumer reads
the two verifiers' already-computed admissibility verdicts
(``FaithfulnessResult.admissible`` and
``StabilityResult.admissible_per_surface``) and computes the
conjunction. It does not recompute either statistic, does not call the
LLM, and does not amend either verifier — if a verifier's record shape
changes in a future phase, the consumer inherits the change via the
Pydantic models; the consumer owns the join, not either verifier's
contract.

**The granularity resolution.** Phase 11's
:class:`~kind.mirror.faithfulness.FaithfulnessResult` is per-reading;
Phase 10's :class:`~kind.mirror.stability.StabilityResult` is
per-(criterion, role, checkpoint) and covers N readings. The consumer
indexes the stability results by ``(criterion_id, reader_role,
checkpoint_id)`` and applies the matched per-surface stability scores
to each faithfulness result that shares the envelope. The verdict is
per-reading: one :class:`AdmissibilityVerdict` per
:class:`~kind.mirror.faithfulness.FaithfulnessResult`.

**The no-stability-result case.** The stability runner is opt-in per
(criterion, role, checkpoint); not every reading with a faithfulness
result has a matching stability result. Phase 12 commits the
*permissive* reading: when no stability result matches, the verdict's
:attr:`AdmissibilityVerdict.stability_admissible_per_surface` is empty,
:attr:`AdmissibilityVerdict.stability_admissible_all_surfaces` is
``True`` by vacuous-case convention, and the verdict is gated by
faithfulness alone. The verdict's ``notes`` records the missing-stability
case explicitly so a downstream consumer can choose to treat it
differently. The candidate amendment — requiring stability for all
readings ("both verifiers must have run") — is journaled and not made.

**One-way invariant.** The consumer reads
:class:`~kind.mirror.faithfulness.FaithfulnessResult` and
:class:`~kind.mirror.stability.StabilityResult` records, both already
on the mirror's side of the membrane. It writes only to its return
value and (optionally) to a caller-provided audit JSONL path. No path
writes to ``runs/{run_id}/telemetry/``, ``runs/{run_id}/checkpoints/``,
or any other Io-readable surface.

Out of scope for Phase 12: any change to Io, the actor, the world
model, the dream state, or the runner; any change to Phase 10's
stability runner or Phase 11's faithfulness verifier; LLM calls of any
kind (the consumer is verifier-only); the prompt-builder amendments,
the iterated canonicalization rule, and the per-statistic-type
list-shape table all still deferred from Phases 10 and 11.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from kind.mirror.faithfulness import FaithfulnessResult
from kind.mirror.llm_caller import PassRole
from kind.mirror.registry import ReadingSurface
from kind.mirror.stability import StabilityResult

__all__ = [
    "AdmissibilityVerdict",
    "AdmissibilityBatchResult",
    "compute_admissibility",
    "load_admissibility_inputs",
]


_ModelT = TypeVar("_ModelT", bound=BaseModel)


# ---------------------------------------------------------------------------
# Result records.
# ---------------------------------------------------------------------------


class AdmissibilityVerdict(BaseModel):
    """Per-reading admissibility verdict — the join of one
    :class:`~kind.mirror.faithfulness.FaithfulnessResult` with the
    matching :class:`~kind.mirror.stability.StabilityResult`.

    Frozen, ``extra="forbid"``. The load-bearing output is
    :attr:`admissible`; the per-axis fields carry the contributing
    evidence for audit and for Window's display.

    Fields:

    - ``pass_index`` / ``criterion_id`` / ``reader_role`` / ``run_id``
      / ``checkpoint_id``: the reading's envelope, copied from the
      source :class:`~kind.mirror.faithfulness.FaithfulnessResult`.
    - ``faithfulness_admissible``: copied from
      :attr:`~kind.mirror.faithfulness.FaithfulnessResult.admissible`.
    - ``faithfulness_rate``: copied from
      :attr:`~kind.mirror.faithfulness.FaithfulnessResult.faithfulness_rate`.
    - ``stability_admissible_per_surface``: copied from the matched
      :attr:`~kind.mirror.stability.StabilityResult.admissible_per_surface`;
      an empty dict when no stability result matched (the
      no-stability-result case).
    - ``stability_admissible_all_surfaces``: ``True`` iff every value
      in ``stability_admissible_per_surface`` is ``True``; ``True`` by
      vacuous-case convention when the dict is empty.
    - ``admissible``: the conjunction —
      ``faithfulness_admissible AND stability_admissible_all_surfaces``.
    - ``notes``: non-empty free text recording which source records
      were joined, which surfaces contributed, and any anomalies.
    - ``wallclock_ms``: the per-verdict join duration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    pass_index: int
    criterion_id: str
    reader_role: PassRole
    run_id: str
    checkpoint_id: str
    faithfulness_admissible: bool
    faithfulness_rate: float
    stability_admissible_per_surface: dict[ReadingSurface, bool]
    stability_admissible_all_surfaces: bool
    admissible: bool
    notes: str
    wallclock_ms: int

    @field_validator("pass_index", "wallclock_ms")
    @classmethod
    def _validate_non_negative(cls, value: int) -> int:
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

    @field_validator("notes")
    @classmethod
    def _validate_notes_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "AdmissibilityVerdict.notes must be non-empty: the consumer "
                "always records which FaithfulnessResult and which "
                "StabilityResult were joined into the verdict."
            )
        return value


class AdmissibilityBatchResult(BaseModel):
    """Aggregate admissibility verdict over a full round or run.

    Frozen, ``extra="forbid"``. One :class:`AdmissibilityBatchResult`
    per :func:`compute_admissibility` call; it carries every per-reading
    :class:`AdmissibilityVerdict` plus the inadmissibility breakdown.

    The four ``n_inadmissible_*`` count fields partition the inadmissible
    readings by cause; together with :attr:`n_admissible` they sum to
    :attr:`n_readings_total` (the model validator pins this). A reading
    is counted in exactly one bucket:

    - ``n_inadmissible_faithfulness``: faithfulness alone failed
      (stability matched and cleared every surface).
    - ``n_inadmissible_stability``: stability alone failed
      (faithfulness cleared).
    - ``n_inadmissible_both``: both faithfulness and stability failed
      (stability matched).
    - ``n_inadmissible_no_stability``: the reading is inadmissible and
      had no matching stability result — faithfulness alone gated it
      and faithfulness failed. A reading with no stability result that
      *is* admissible counts in :attr:`n_admissible`, not here.

    A run with zero readings yields :attr:`n_readings_total` ``0``, all
    counts ``0``, and :attr:`admissibility_rate` ``1.0`` (vacuous).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdicts: tuple[AdmissibilityVerdict, ...]
    n_readings_total: int
    n_admissible: int
    n_inadmissible_faithfulness: int
    n_inadmissible_stability: int
    n_inadmissible_both: int
    n_inadmissible_no_stability: int
    admissibility_rate: float
    notes: str
    wallclock_ms: int

    @field_validator(
        "n_readings_total",
        "n_admissible",
        "n_inadmissible_faithfulness",
        "n_inadmissible_stability",
        "n_inadmissible_both",
        "n_inadmissible_no_stability",
        "wallclock_ms",
    )
    @classmethod
    def _validate_count_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"value must be >= 0; got {value}.")
        return value

    @field_validator("admissibility_rate")
    @classmethod
    def _validate_rate_in_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"admissibility_rate must be in [0.0, 1.0]; got {value}."
            )
        return value

    @field_validator("notes")
    @classmethod
    def _validate_notes_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("AdmissibilityBatchResult.notes must be non-empty.")
        return value

    @model_validator(mode="after")
    def _validate_counts_sum(self) -> "AdmissibilityBatchResult":
        total = (
            self.n_admissible
            + self.n_inadmissible_faithfulness
            + self.n_inadmissible_stability
            + self.n_inadmissible_both
            + self.n_inadmissible_no_stability
        )
        if total != self.n_readings_total:
            raise ValueError(
                f"AdmissibilityBatchResult counts do not sum to "
                f"n_readings_total: n_admissible={self.n_admissible}, "
                f"n_inadmissible_faithfulness={self.n_inadmissible_faithfulness}, "
                f"n_inadmissible_stability={self.n_inadmissible_stability}, "
                f"n_inadmissible_both={self.n_inadmissible_both}, "
                f"n_inadmissible_no_stability={self.n_inadmissible_no_stability}, "
                f"sum={total}, n_readings_total={self.n_readings_total}."
            )
        if len(self.verdicts) != self.n_readings_total:
            raise ValueError(
                f"AdmissibilityBatchResult.verdicts has length "
                f"{len(self.verdicts)} but n_readings_total is "
                f"{self.n_readings_total}; they must match."
            )
        return self


# ---------------------------------------------------------------------------
# The consumer.
# ---------------------------------------------------------------------------


def _stability_index(
    stability_results: tuple[StabilityResult, ...] | list[StabilityResult],
) -> dict[tuple[str, PassRole, str], StabilityResult]:
    """Index the stability results by ``(criterion_id, reader_role,
    checkpoint_id)``. Each :class:`~kind.mirror.stability.StabilityResult`
    covers N readings; the index lets the consumer find the stability
    record matching a given faithfulness result. The signal names are
    unique across the tuple in practice; if two stability results share
    an envelope the later one wins, deterministically."""
    index: dict[tuple[str, PassRole, str], StabilityResult] = {}
    for result in stability_results:
        index[(result.criterion_id, result.reader_role, result.checkpoint_id)] = (
            result
        )
    return index


def compute_admissibility(
    *,
    faithfulness_results: tuple[FaithfulnessResult, ...] | list[FaithfulnessResult],
    stability_results: tuple[StabilityResult, ...] | list[StabilityResult],
    run_id: str,
    audit_jsonl_path: Path | None = None,
) -> AdmissibilityBatchResult:
    """Join the two verifiers' records into one verdict per reading.

    For each :class:`~kind.mirror.faithfulness.FaithfulnessResult`:

    1. Look up the matching
       :class:`~kind.mirror.stability.StabilityResult` by
       ``(criterion_id, reader_role, checkpoint_id)``.
    2. Read ``faithfulness_admissible`` and ``faithfulness_rate`` from
       the faithfulness result.
    3. Read ``stability_admissible_per_surface`` from the stability
       result (or the empty dict if no match — the no-stability-result
       case).
    4. ``stability_admissible_all_surfaces`` is the conjunction across
       the per-surface dict's values; vacuously ``True`` if empty.
    5. ``admissible = faithfulness_admissible AND
       stability_admissible_all_surfaces``.
    6. Construct an :class:`AdmissibilityVerdict` with notes describing
       the join.

    The per-reading verdicts are aggregated into an
    :class:`AdmissibilityBatchResult` whose four ``n_inadmissible_*``
    fields partition the inadmissible readings by cause.

    If ``audit_jsonl_path`` is provided, each verdict is appended as a
    single JSONL line to that path (the batch is the in-memory
    aggregate; the per-verdict lines are the durable record). The caller
    is responsible for the path's containing directory living under
    ``runs/{run_id}/mirror/``; the one-way-write invariant is preserved
    at the call site by the caller's directory choice.

    The consumer is pure aside from the optional audit emission. No LLM
    calls; no statistic recomputation; no telemetry reads. The input
    tuples are not modified.
    """
    batch_start_ms = int(time.time() * 1000)
    index = _stability_index(stability_results)

    verdicts: list[AdmissibilityVerdict] = []
    n_admissible = 0
    n_inadmissible_faithfulness = 0
    n_inadmissible_stability = 0
    n_inadmissible_both = 0
    n_inadmissible_no_stability = 0

    for faithfulness in faithfulness_results:
        verdict_start_ms = int(time.time() * 1000)
        envelope = (
            faithfulness.criterion_id,
            faithfulness.reader_role,
            faithfulness.checkpoint_id,
        )
        stability = index.get(envelope)

        faithfulness_admissible = faithfulness.admissible
        if stability is None:
            per_surface: dict[ReadingSurface, bool] = {}
        else:
            per_surface = dict(stability.admissible_per_surface)
        stability_admissible_all_surfaces = all(per_surface.values())
        admissible = faithfulness_admissible and stability_admissible_all_surfaces

        if stability is None:
            stability_note = (
                f"no StabilityResult matched envelope "
                f"(criterion_id={faithfulness.criterion_id!r}, "
                f"reader_role={faithfulness.reader_role!r}, "
                f"checkpoint_id={faithfulness.checkpoint_id!r}); stability is "
                f"vacuously admissible (no declared surfaces) — faithfulness "
                f"alone gates this reading"
            )
        else:
            surface_pairs = ", ".join(
                f"{surface.value}={per_surface[surface]}"
                for surface in sorted(per_surface, key=lambda s: s.value)
            )
            stability_note = (
                f"joined StabilityResult(criterion_id="
                f"{stability.criterion_id!r}, reader_role="
                f"{stability.reader_role!r}, checkpoint_id="
                f"{stability.checkpoint_id!r}) — admissible_per_surface: "
                f"{{{surface_pairs}}}; "
                f"stability_admissible_all_surfaces="
                f"{stability_admissible_all_surfaces}"
            )
        notes = (
            f"joined FaithfulnessResult(criterion_id="
            f"{faithfulness.criterion_id!r}, reader_role="
            f"{faithfulness.reader_role!r}, pass_index="
            f"{faithfulness.pass_index}, checkpoint_id="
            f"{faithfulness.checkpoint_id!r}, faithfulness_rate="
            f"{faithfulness.faithfulness_rate:.4f}, admissible="
            f"{faithfulness_admissible}); {stability_note}. Verdict "
            f"admissible={admissible} (AND-conjunction)."
        )

        verdict_end_ms = int(time.time() * 1000)
        verdict = AdmissibilityVerdict(
            pass_index=faithfulness.pass_index,
            criterion_id=faithfulness.criterion_id,
            reader_role=faithfulness.reader_role,
            run_id=faithfulness.run_id,
            checkpoint_id=faithfulness.checkpoint_id,
            faithfulness_admissible=faithfulness_admissible,
            faithfulness_rate=faithfulness.faithfulness_rate,
            stability_admissible_per_surface=per_surface,
            stability_admissible_all_surfaces=stability_admissible_all_surfaces,
            admissible=admissible,
            notes=notes,
            wallclock_ms=max(verdict_end_ms - verdict_start_ms, 0),
        )
        verdicts.append(verdict)

        if admissible:
            n_admissible += 1
        elif stability is None:
            # Inadmissible with no stability record: faithfulness alone
            # gated it (and failed). Counted here, not in
            # n_inadmissible_faithfulness — the missing-stability case
            # is recorded distinctly so a downstream consumer can treat
            # it differently.
            n_inadmissible_no_stability += 1
        else:
            faithfulness_failed = not faithfulness_admissible
            stability_failed = not stability_admissible_all_surfaces
            if faithfulness_failed and stability_failed:
                n_inadmissible_both += 1
            elif faithfulness_failed:
                n_inadmissible_faithfulness += 1
            else:
                n_inadmissible_stability += 1

    n_total = len(verdicts)
    admissibility_rate = n_admissible / n_total if n_total > 0 else 1.0

    batch_end_ms = int(time.time() * 1000)
    batch = AdmissibilityBatchResult(
        verdicts=tuple(verdicts),
        n_readings_total=n_total,
        n_admissible=n_admissible,
        n_inadmissible_faithfulness=n_inadmissible_faithfulness,
        n_inadmissible_stability=n_inadmissible_stability,
        n_inadmissible_both=n_inadmissible_both,
        n_inadmissible_no_stability=n_inadmissible_no_stability,
        admissibility_rate=admissibility_rate,
        notes=(
            f"joined {len(faithfulness_results)} faithfulness result(s) with "
            f"{len(stability_results)} stability result(s) for run "
            f"{run_id!r}; {n_admissible}/{n_total} reading(s) admissible "
            f"(rate={admissibility_rate:.4f}) under the AND-conjunction join."
        ),
        wallclock_ms=max(batch_end_ms - batch_start_ms, 0),
    )

    if audit_jsonl_path is not None:
        with audit_jsonl_path.open("a", encoding="utf-8") as fh:
            for verdict in batch.verdicts:
                fh.write(verdict.model_dump_json())
                fh.write("\n")

    return batch


# ---------------------------------------------------------------------------
# Convenience loader.
# ---------------------------------------------------------------------------


def _read_jsonl_models(path: Path, model: type[_ModelT]) -> list[_ModelT]:
    """Read every non-blank line of a JSONL file through ``model``. A
    missing file yields an empty list. A malformed line raises (this is
    a thin wrapper for scripts and tests — Window's loaders carry the
    error-tolerant per-line variant)."""
    if not path.is_file():
        return []
    records: list[_ModelT] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(model.model_validate_json(stripped))
    return records


def load_admissibility_inputs(
    run_id: str, run_dir: Path
) -> tuple[tuple[FaithfulnessResult, ...], tuple[StabilityResult, ...]]:
    """Load the consumer's two inputs from their committed on-disk
    locations.

    Reads faithfulness records from ``{run_dir}/mirror/faithfulness.jsonl``
    and stability records from ``{run_dir}/mirror/stability.jsonl`` —
    the locations Phases 11 and 10 committed for their respective audit
    JSONL files. ``run_dir`` is the run's directory (``runs/{run_id}/``
    typically); ``run_id`` is the run's label, carried for the caller's
    clarity and for symmetry with the other ``runs/{run_id}/`` loaders.

    Returns the two tuples ready for :func:`compute_admissibility`. A
    missing JSONL file yields an empty tuple — a run that ran neither
    verifier still loads cleanly. This is a thin wrapper; the
    verifier-side phases (10, 11) own the on-disk shapes.
    """
    mirror_dir = run_dir / "mirror"
    faithfulness = _read_jsonl_models(
        mirror_dir / "faithfulness.jsonl", FaithfulnessResult
    )
    stability = _read_jsonl_models(
        mirror_dir / "stability.jsonl", StabilityResult
    )
    return tuple(faithfulness), tuple(stability)
