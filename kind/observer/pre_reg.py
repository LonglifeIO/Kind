"""Probe 2 pre-registration schema and JSONL sink.

The Probe 2 v2 calibration protocol's first element (synthesis §2.4
element 1) is *pre-registration*: before each adversarial pass, the
builder writes down the criteria-active set, the criteria-held-out set,
the per-criterion signal mappings, falsifiers, scalar checks, the
builder's mode (proponent vs skeptic), the expected outcome, plus the
v2 additions: which reading surface each criterion is read against, the
asymmetry of access between Io and the mirror, the per-surface expected
outcomes, the substrate decisions off the table for this round, the
column-init the run was constructed with, and any new actor-readable
interfaces the round adds.

The schema-bump from v1's ``"0.1.0"`` to v2's ``"0.2.0"``
(:data:`PRE_REG_SCHEMA_VERSION`) carries the new fields. No migration is
needed — Probe 1.5 did not run the calibration protocol, so no v1 records
exist on disk.

**The garden-of-forking-paths discipline (Gelman & Loken 2013).**
Pre-registration as a constraint, not a formality: the structural counter
against the mirror's reading drifting toward what the builder hopes for
in retrospect. The :class:`PreRegistration` record is append-only, written
before the reading runs, and is what the journal entry's prose expansion
embeds.

**The Watts-default-applied-to-builder discipline (synthesis §2.5
element 8; Probe 1.5 v2 §2(b) sub-clause).** The
``substrate_decisions_off_table`` field carries the per-round commitment
not to revise the substrate based on the reading. The
``new_actor_readable_interfaces_added`` field carries the Probe 1.5 v2
§2(b) sub-clause: any new actor-readable interface the round adds must
be journaled with the four-part discipline ((i) which affordance, (ii)
minimum form, (iii) alternatives considered, (iv) failure-mode controls).
For Probe 2 specifically the field is typically empty (the substrate is
settled; no new actor-readable interface is added); the field is the
structural hook for Probes 3 and beyond.

**Per-criterion completeness validation.** A ``model_validator`` enforces
that every criterion id in ``criteria_active`` has corresponding entries
in ``signal_mappings`` and ``falsifiers``; missing entries raise
``ValidationError`` at construction. The ``scalar_checks`` and
``reading_surfaces_per_criterion`` dicts are also keyed by criterion id;
their absence for an active criterion raises. ``criteria_held_out`` need
not have entries in any of these dicts — held-out criteria are by design
not in the prompt at this round.

Out of scope at Phase 0:
- A CLI for filling in a pre-registration.
- Automatic comparison of pre-registration to the actual reading after
  the fact (synthesis §2.4 element 1 is journaled, not code-enforced).
- Any reader of the JSONL file. :class:`PreRegSink` writes; downstream
  phases that read are Phase 12's smoke and Phase 13's gate tests.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import TracebackType
from typing import IO, Final, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, model_validator

# Phase 7: the surface keys on ``expected_outcome_per_surface`` and the
# surface values on ``reading_surfaces_per_criterion`` use the
# :class:`~kind.observer.reading_surface.ReadingSurface` *enum*, not the Phase 0
# ``Literal`` at :data:`kind.mirror.structured.ReadingSurface`. The enum's
# values match the Literal's strings exactly, so this is a type tightening
# with no value change and no JSON-data-serialization change (str-valued
# enum members serialize identically to their string values); a typo in a
# surface key now trips at construction rather than reaching a validator
# late. :mod:`kind.mirror.structured`'s ``StructuredClaim`` /
# ``StructuredReading`` / ``JudgeRuling`` still use the Phase 0 Literal.
#
# Phase 8a: imported from the observer-level leaf (re-exported by
# ``kind.mirror.registry`` for back-compat) so the observer layer no longer
# reaches up into the mirror layer — breaking the cold-import cycle.
from kind.observer.reading_surface import ReadingSurface

__all__ = [
    "PRE_REG_SCHEMA_VERSION",
    "PRE_REG_FILE",
    "ColumnInit",
    "BuilderMode",
    "PreRegistration",
    "PreRegSink",
    "PreRegSinkClosedError",
]


# v1 was ``"0.1.0"``; v2 bumps to ``"0.2.0"`` for the new fields'
# addition. Probe 1.5 did not run the calibration protocol, so no v1
# pre-reg records exist on disk; backward-readability is structural
# (the v2 model can be relaxed to read any v1 records that materialize
# later) but unexercised.
PRE_REG_SCHEMA_VERSION: Final[str] = "0.2.0"

# By convention the sink writes to ``runs/{run_id}/pre_reg/pre_reg.jsonl``.
# The directory layout is the runner's concern; the sink takes a directory
# and uses this filename inside it.
PRE_REG_FILE: Final[str] = "pre_reg.jsonl"


# Column-init carrier (Phase 8 column-init confound; synthesis §2.4 +
# §3 (l)). Pre-registration records the column_init the run's actor was
# constructed with, so the Skeptic's substrate-side and behavior-side
# refutations can cite it (the column-init-determination refutation at
# behavior-side is the load-bearing case). ``"unknown"`` is the
# fallback for runs whose construction did not record the choice
# explicitly; the build phase's discipline is to avoid this state.
ColumnInit: TypeAlias = Literal["zero", "small_gaussian", "unknown"]


# Builder-mode literal carries the two-mode discipline (synthesis §2.5
# element 6): the builder declares whether they are arguing FOR the
# criterion's presence at this round (proponent) or AGAINST (skeptic).
# The mode is journaled to make the framing visible to future-the-builder.
BuilderMode: TypeAlias = Literal["proponent", "skeptic"]


class PreRegistration(BaseModel):
    """One pre-registration record per adversarial pass.

    The ``criteria_active`` and ``criteria_held_out`` lists name criterion
    ids from the registry at :mod:`kind.mirror.criteria` (Phase 7 — this
    module does not import the registry to avoid a circular dependency at
    Phase 0; the registry's id strings are the contract).

    The ``signal_mappings``, ``falsifiers``, ``scalar_checks``, and
    ``reading_surfaces_per_criterion`` dicts are keyed by criterion id.
    A ``model_validator`` enforces that every active criterion has
    entries in each of these dicts; missing entries fail validation. This
    is the structural counter against an active criterion that has no
    falsifier (the discipline element synthesis §2.4 element 1 names as
    load-bearing).

    ``asymmetry_of_access`` is free-text describing what Io reads vs
    what the mirror reads at this round. The Probe 1.5 v2 case: Io
    reads the scalar ``self_prediction_error_t`` on PolicyView; the
    mirror reads the full prediction vector ``self_prediction_t``,
    per-dimension allocation, perturbation-recovery dynamics,
    behavioral conditioning, the masked flag, all longitudinal
    cross-run analysis. The asymmetry is load-bearing: it determines
    what the readers can claim.

    ``expected_outcome`` is free-text the builder's overall expectation;
    ``expected_outcome_per_surface`` is the per-surface refinement
    (substrate-side / head-internal / behavior-side).

    ``new_actor_readable_interfaces_added`` carries the Probe 1.5 v2
    §2(b) sub-clause: each entry is a free-text description of an
    interface the round adds (typically empty at Probe 2). The four-part
    journal discipline lives in the prose journal entry; the schema
    field is the structural carrier.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = PRE_REG_SCHEMA_VERSION
    run_id: str
    timestamp_ms: int
    criteria_active: list[str]
    criteria_held_out: list[str]
    signal_mappings: dict[str, list[str]]
    falsifiers: dict[str, str]
    scalar_checks: dict[str, list[str]]
    reading_surfaces_per_criterion: dict[str, list[ReadingSurface]]
    asymmetry_of_access: str
    builder_mode: BuilderMode
    expected_outcome: str
    expected_outcome_per_surface: dict[ReadingSurface, str]
    substrate_decisions_off_table: list[str]
    column_init: ColumnInit
    new_actor_readable_interfaces_added: list[str]

    @model_validator(mode="after")
    def _enforce_per_criterion_completeness(self) -> "PreRegistration":
        """Every active criterion id must have an entry in
        ``signal_mappings``, ``falsifiers``, ``scalar_checks``, and
        ``reading_surfaces_per_criterion``. Held-out criteria are not
        checked — a held-out criterion is by design not in the prompt
        at this round, so its mappings may be deferred."""
        for criterion_id in self.criteria_active:
            missing: list[str] = []
            if criterion_id not in self.signal_mappings:
                missing.append("signal_mappings")
            if criterion_id not in self.falsifiers:
                missing.append("falsifiers")
            if criterion_id not in self.scalar_checks:
                missing.append("scalar_checks")
            if criterion_id not in self.reading_surfaces_per_criterion:
                missing.append("reading_surfaces_per_criterion")
            if missing:
                raise ValueError(
                    f"PreRegistration: active criterion "
                    f"{criterion_id!r} is missing entries in "
                    f"{missing}. Synthesis §2.4 element 1 requires "
                    f"every active criterion carry a falsifier, signal "
                    f"mapping, scalar check, and reading-surface "
                    f"assignment before the reading runs."
                )
        # Active and held-out sets must be disjoint — a criterion is
        # one or the other at this round, not both.
        overlap = set(self.criteria_active) & set(self.criteria_held_out)
        if overlap:
            raise ValueError(
                f"PreRegistration: criteria_active and criteria_held_out "
                f"overlap on {sorted(overlap)}. A criterion is either in "
                f"the prompt this round or held out; not both."
            )
        return self


class PreRegSinkClosedError(RuntimeError):
    """Raised when ``write()`` is called on a sink that has been closed."""


class PreRegSink:
    """Append-only JSONL writer for :class:`PreRegistration` records.

    Mirrors the shape of :class:`kind.observer.sinks.JsonlSink` but takes
    a *directory* (by convention ``runs/{run_id}/pre_reg/``) and writes to
    a fixed filename (``pre_reg.jsonl``) inside it. The constructor creates
    the directory if it does not exist.

    The sink is append-only: each :meth:`write` call adds one
    JSON-encoded line; the file is flushed and ``os.fsync``-ed on
    :meth:`close` so a clean shutdown leaves the data on disk. The sink
    supports ``with`` syntax via :meth:`__enter__` / :meth:`__exit__`.

    There is no read API on the sink — downstream consumers read
    ``runs/{run_id}/pre_reg/pre_reg.jsonl`` directly via standard JSONL
    parsing plus :meth:`PreRegistration.model_validate_json`. This keeps
    the sink a pure write-side concern.
    """

    def __init__(self, dir: Path) -> None:
        self._dir = dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = dir / PRE_REG_FILE
        self._file: IO[str] | None = self._path.open("a", encoding="utf-8")
        self._closed = False

    @property
    def path(self) -> Path:
        return self._path

    def write(self, record: PreRegistration) -> None:
        if self._closed or self._file is None:
            raise PreRegSinkClosedError(
                f"PreRegSink at {self._path} is closed"
            )
        self._file.write(record.model_dump_json() + "\n")

    def close(self) -> None:
        if self._closed:
            return
        if self._file is not None:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            self._file = None
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
