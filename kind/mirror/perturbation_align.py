"""Phase 8 perturbation-log reader and timeline aligner.

The equanimity criterion needs to know *when* a perturbation happened to
compute ``recovery_lag_steps`` and the post-perturbation entropy / KL
trajectory shapes — but per the membrane discipline the criterion cannot
read ``world_event`` directly (``world_event`` is walled off from the
agent process; the mirror reading it would be the cross-membrane
dependency the design notes prohibit).

The cross-reference therefore happens at prompt-build, *orchestrator-side*,
on the mirror's side of the membrane. This module is the load-bearing
piece of that glue: it reads the runner's emitted ``world_event.jsonl``
plus the ``agent_step`` parquet shards, matches each
``builder_perturbation`` event to the closest-by-``wallclock_ms``
:class:`~kind.observer.schemas.AgentStep` record (within a configurable
tolerance), and produces a :class:`PerturbationTimeline` of typed
records keyed on ``AgentStep.t``. The timeline is what the equanimity
prompt-builder fragment cites for its perturbation-aware computations.

**The sham-event seam.** A sham builder-perturbation event is a flag-only
``world_event`` (no actual env mutation) emitted by the env-server's
sham-perturbation path with ``payload["is_sham"]=True`` per
:mod:`kind.observer.schemas`. The aligner reads the flag and surfaces
it on the corresponding :class:`PerturbationEvent`; the orchestrator's
sham-perturbation calibration check (Part 5) walks the timeline and
asserts no equanimity admission at sham timestamps at any surface.

**The membrane invariant.** This module reads only. It does not write,
does not construct an actor / world model / runner, does not invoke
anything against Io's input space. It is on the mirror's side of the
membrane; the world-event log it reads has been written there by the
runner's transport-client handler, not pulled back into the agent's
read surfaces.

Out of scope: any change to the env or the runner. Phase 8's aligner
consumes whatever the runner already writes.
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any, Final

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, model_validator

__all__ = [
    "PERTURBATION_EVENT_TYPE",
    "DEFAULT_ALIGNMENT_TOLERANCE_MS",
    "PerturbationAlignmentError",
    "PerturbationEvent",
    "PerturbationTimeline",
    "align_perturbations",
]


PERTURBATION_EVENT_TYPE: Final[str] = "builder_perturbation"
DEFAULT_ALIGNMENT_TOLERANCE_MS: Final[int] = 1000
_AGENT_STEP_SUBDIR: Final[str] = "agent_step"


class PerturbationAlignmentError(RuntimeError):
    """Raised when a ``builder_perturbation`` event cannot be matched to
    an :class:`~kind.observer.schemas.AgentStep` record within the
    configured ``wallclock_ms`` tolerance.

    The error message names the perturbation's ``wallclock_ms``, the
    nearest agent-step's ``wallclock_ms``, the delta, and the tolerance,
    so the operator can decide whether to widen the tolerance or to
    investigate a clock-skew / dropped-shard issue.
    """


class PerturbationEvent(BaseModel):
    """One ``builder_perturbation`` event aligned to the agent timeline.

    Frozen, ``extra="forbid"``. Fields:

    - ``t``: the :class:`~kind.observer.schemas.AgentStep` ``t`` the
      perturbation aligned to (closest-by-``wallclock_ms``).
    - ``wallclock_ms``: the perturbation event's own ``wallclock_ms`` from
      the world-event log (NOT the matched agent-step's wallclock; the
      raw event clock is preserved so the alignment can be re-audited).
    - ``payload``: the original world-event payload, untouched.
    - ``is_sham``: read from ``payload["is_sham"]`` (defaults to ``False``
      when the key is absent — real perturbations don't carry the flag).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    t: int
    wallclock_ms: int
    payload: dict[str, Any]
    is_sham: bool = False


class PerturbationTimeline(BaseModel):
    """The ordered, agent-step-aligned perturbation log for one pass.

    Frozen, ``extra="forbid"``. The ``events`` tuple is sorted by ``t``
    (validated); two events at the same ``t`` are rejected (the alignment
    can't tell them apart and the equanimity statistics would
    double-count). Empty timelines are valid and represent
    "no perturbations occurred (or aligned) during the pass window".
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    events: tuple[PerturbationEvent, ...]
    run_id: str
    checkpoint_id: str

    @model_validator(mode="after")
    def _validate_sorted_unique(self) -> PerturbationTimeline:
        seen_t: set[int] = set()
        prev_t: int | None = None
        for event in self.events:
            if event.t in seen_t:
                raise ValueError(
                    f"PerturbationTimeline: two events at the same agent-step "
                    f"t={event.t}; the aligner cannot distinguish them and "
                    f"the equanimity statistics would double-count. Drop "
                    f"duplicates upstream."
                )
            if prev_t is not None and event.t < prev_t:
                raise ValueError(
                    f"PerturbationTimeline: events must be sorted by t; got "
                    f"t={event.t} after t={prev_t}."
                )
            seen_t.add(event.t)
            prev_t = event.t
        return self


# ---------------------------------------------------------------------------
# Log reading.
# ---------------------------------------------------------------------------


def _read_world_event_log(path: Path) -> list[dict[str, Any]]:
    """Read every line of ``path`` (a JSONL world-event log) into a list
    of dicts. Missing file returns an empty list; malformed lines raise.

    The schema validation happens upstream — Phase 1's
    :class:`~kind.observer.sinks.JsonlSink` is the producer; this reader
    treats records as plain dicts to avoid a Pydantic round-trip on every
    line.
    """
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _read_agent_step_rows(path: Path) -> list[dict[str, Any]]:
    """Read every parquet shard under ``path/agent_step/`` (or
    ``path`` if it already names a shard directory) into a list of
    dicts, ordered by row order within shards and shards sorted by name.
    """
    if path.is_dir():
        # ``path`` may be either the run's telemetry_dir (contains
        # ``agent_step/`` subdir) or the agent_step subdir itself.
        if (path / _AGENT_STEP_SUBDIR).is_dir():
            shard_dir = path / _AGENT_STEP_SUBDIR
        else:
            shard_dir = path
    else:
        return []
    shards = sorted(shard_dir.glob("shard-*.parquet"))
    rows: list[dict[str, Any]] = []
    for shard in shards:
        table = pq.read_table(str(shard))  # type: ignore[no-untyped-call]
        rows.extend(table.to_pylist())
    return rows


def align_perturbations(
    world_event_log_path: Path,
    agent_step_log_path: Path,
    *,
    run_id: str,
    checkpoint_id: str,
    tolerance_ms: int = DEFAULT_ALIGNMENT_TOLERANCE_MS,
) -> PerturbationTimeline:
    """Build a :class:`PerturbationTimeline` from the runner's emitted logs.

    Reads ``world_event_log_path`` (a JSONL file at
    ``runs/{run_id}/telemetry/world_event.jsonl``), filters to
    ``event_type == "builder_perturbation"``, and matches each event to
    the closest :class:`~kind.observer.schemas.AgentStep` by
    ``wallclock_ms`` within ``tolerance_ms``. If no agent-step record
    falls within tolerance, :class:`PerturbationAlignmentError` is raised
    naming the perturbation, the nearest record, the delta, and the
    tolerance.

    ``run_id`` and ``checkpoint_id`` are taken as explicit arguments
    (rather than read from the world-event records) because a pass is
    structurally tied to a single ``(run_id, checkpoint_id)`` pair and
    the orchestrator chooses the checkpoint, not the world-event log.
    A future contributor who wants to assert the world-event records
    match the orchestrator's ``run_id`` can do so explicitly after
    calling; the aligner itself doesn't gate on it (a mirror-side
    consistency check is the right place for that, not this reader).

    Sham events (``payload["is_sham"]==True``) align the same way as
    real events; the ``is_sham`` flag is preserved on the
    :class:`PerturbationEvent` and the orchestrator's sham calibration
    check (Part 5) walks the timeline to find them.

    Returns an empty timeline if there are no ``builder_perturbation``
    events in the log or if either log is empty. Empty logs are valid
    — a pass against a checkpoint that ran without any perturbations
    produces a clean empty timeline, not an error.
    """
    if tolerance_ms < 0:
        raise ValueError(
            f"tolerance_ms must be non-negative; got {tolerance_ms}."
        )

    world_events = _read_world_event_log(world_event_log_path)
    perturbations = [
        we for we in world_events
        if we.get("event_type") == PERTURBATION_EVENT_TYPE
    ]
    if not perturbations:
        return PerturbationTimeline(
            events=tuple(),
            run_id=run_id,
            checkpoint_id=checkpoint_id,
        )

    agent_rows = _read_agent_step_rows(agent_step_log_path)
    if not agent_rows:
        # Perturbations exist on the log but the agent-step parquet is
        # empty — this is a hard inconsistency the orchestrator should
        # surface, not silently swallow.
        raise PerturbationAlignmentError(
            f"world_event.jsonl at {world_event_log_path} contains "
            f"{len(perturbations)} builder_perturbation event(s) but "
            f"the agent_step shards at {agent_step_log_path} are empty; "
            f"cannot align."
        )

    # Build a wallclock-sorted index. ``wallclock_ms`` is monotonic-ish
    # within a run but parquet shard order is by row append order, which
    # for the runner is also wallclock-monotonic — defensive sort for
    # the day that invariant changes.
    sorted_rows = sorted(agent_rows, key=lambda r: int(r["wallclock_ms"]))
    wallclocks = [int(r["wallclock_ms"]) for r in sorted_rows]
    ts = [int(r["t"]) for r in sorted_rows]

    aligned: list[PerturbationEvent] = []
    for pert in perturbations:
        wc_pert = int(pert["wallclock_ms"])
        # Binary search for the closest wallclock_ms.
        idx = bisect.bisect_left(wallclocks, wc_pert)
        candidates: list[int] = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(wallclocks):
            candidates.append(idx)
        if not candidates:  # pragma: no cover — len(agent_rows) > 0 above
            raise PerturbationAlignmentError(
                f"empty agent-step index; cannot align perturbation at "
                f"wallclock_ms={wc_pert}"
            )
        best = min(candidates, key=lambda i: abs(wallclocks[i] - wc_pert))
        delta = abs(wallclocks[best] - wc_pert)
        if delta > tolerance_ms:
            raise PerturbationAlignmentError(
                f"builder_perturbation event at wallclock_ms={wc_pert} "
                f"has no agent_step within tolerance_ms={tolerance_ms} "
                f"(nearest wallclock_ms={wallclocks[best]}, delta={delta} "
                f"ms, agent_step t={ts[best]}). Widen tolerance_ms or "
                f"investigate clock skew / dropped shard."
            )
        payload = pert.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}
        is_sham = bool(payload.get("is_sham", False))
        aligned.append(
            PerturbationEvent(
                t=ts[best],
                wallclock_ms=wc_pert,
                payload=dict(payload),
                is_sham=is_sham,
            )
        )

    # Sort by t for the timeline contract; reject collisions per the
    # PerturbationTimeline validator (two events at the same t cannot
    # be distinguished).
    aligned.sort(key=lambda e: e.t)
    return PerturbationTimeline(
        events=tuple(aligned),
        run_id=run_id,
        checkpoint_id=checkpoint_id,
    )
