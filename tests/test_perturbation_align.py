"""Phase 8 gate test — :mod:`kind.mirror.perturbation_align`.

Covers the load-bearing alignment behaviors:

- empty logs produce an empty timeline (not an error);
- matched perturbations align to the closest agent-step by wallclock_ms;
- unmatched perturbations (no agent-step within tolerance) raise
  :class:`PerturbationAlignmentError` naming the delta and the tolerance;
- sham events propagate ``is_sham=True`` on the
  :class:`PerturbationEvent`;
- ordering by ``t`` is preserved on the
  :class:`PerturbationTimeline.events` tuple;
- two events at the same ``t`` are rejected at timeline construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kind.mirror.perturbation_align import (
    DEFAULT_ALIGNMENT_TOLERANCE_MS,
    PERTURBATION_EVENT_TYPE,
    PerturbationAlignmentError,
    PerturbationEvent,
    PerturbationTimeline,
    align_perturbations,
)


# ---------------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------------


def _world_event_line(
    *,
    t_event: int,
    wallclock_ms: int,
    event_type: str = PERTURBATION_EVENT_TYPE,
    payload: dict[str, Any] | None = None,
    run_id: str = "probe2-test",
    checkpoint_id: str | None = "ckpt-000001",
    source: str = "builder",
) -> str:
    payload = payload if payload is not None else {}
    record = {
        "schema_version": "0.2.0",
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "t_event": t_event,
        "event_type": event_type,
        "source": source,
        "payload": payload,
        "wallclock_ms": wallclock_ms,
    }
    return json.dumps(record)


def _write_world_event_log(
    path: Path, lines: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")


def _write_agent_step_shard(
    shard_dir: Path,
    rows: list[dict[str, Any]],
    index: int = 0,
) -> None:
    """Write a minimal AgentStep parquet shard with just the fields the
    aligner reads (``t``, ``wallclock_ms``). The aligner does not
    deserialize through the full AgentStep model, so a stub schema
    suffices for the test."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            pa.field("t", pa.int64()),
            pa.field("wallclock_ms", pa.int64()),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    shard_path = shard_dir / f"shard-{index:06d}.parquet"
    with shard_path.open("wb") as fh:
        pq.write_table(table, fh)  # type: ignore[no-untyped-call]


# ---------------------------------------------------------------------------
# Empty / degenerate cases.
# ---------------------------------------------------------------------------


def test_empty_world_event_log_produces_empty_timeline(tmp_path: Path) -> None:
    we_path = tmp_path / "telemetry" / "world_event.jsonl"
    _write_world_event_log(we_path, [])
    # No agent-step shards either.
    timeline = align_perturbations(
        we_path,
        tmp_path / "telemetry",
        run_id="r",
        checkpoint_id="c",
    )
    assert timeline.events == tuple()
    assert timeline.run_id == "r"
    assert timeline.checkpoint_id == "c"


def test_missing_world_event_log_produces_empty_timeline(tmp_path: Path) -> None:
    we_path = tmp_path / "telemetry" / "world_event.jsonl"
    # File does not exist — treated as no events.
    timeline = align_perturbations(
        we_path,
        tmp_path / "telemetry",
        run_id="r",
        checkpoint_id="c",
    )
    assert timeline.events == tuple()


def test_non_perturbation_events_are_filtered(tmp_path: Path) -> None:
    """``env_reset`` events should not appear in the timeline."""
    we_path = tmp_path / "telemetry" / "world_event.jsonl"
    _write_world_event_log(
        we_path,
        [
            _world_event_line(
                t_event=0,
                wallclock_ms=100,
                event_type="env_reset",
            ),
        ],
    )
    # No agent-step shards needed because there are no perturbations.
    timeline = align_perturbations(
        we_path,
        tmp_path / "telemetry",
        run_id="r",
        checkpoint_id="c",
    )
    assert timeline.events == tuple()


# ---------------------------------------------------------------------------
# Alignment.
# ---------------------------------------------------------------------------


def test_matched_perturbations_align_to_closest_wallclock(tmp_path: Path) -> None:
    we_path = tmp_path / "telemetry" / "world_event.jsonl"
    _write_world_event_log(
        we_path,
        [
            _world_event_line(t_event=10, wallclock_ms=1050),
            _world_event_line(t_event=30, wallclock_ms=3010),
        ],
    )
    rows = [
        {"t": i, "wallclock_ms": i * 100}
        for i in range(0, 50)
    ]
    _write_agent_step_shard(tmp_path / "telemetry" / "agent_step", rows)
    timeline = align_perturbations(
        we_path,
        tmp_path / "telemetry",
        run_id="r",
        checkpoint_id="c",
    )
    assert len(timeline.events) == 2
    # First perturbation: wallclock 1050 → closest agent_step is t=10
    # (wallclock 1000) or t=11 (wallclock 1100); delta 50 vs 50 — bisect
    # returns the right-hand one by convention but min(...) takes either;
    # the test asserts only that the alignment is within tolerance.
    assert timeline.events[0].t in {10, 11}
    assert timeline.events[0].wallclock_ms == 1050
    assert timeline.events[1].t in {30, 31}
    assert timeline.events[1].wallclock_ms == 3010
    assert all(not e.is_sham for e in timeline.events)


def test_unmatched_perturbation_raises(tmp_path: Path) -> None:
    we_path = tmp_path / "telemetry" / "world_event.jsonl"
    # Perturbation at wallclock 9999, far from any agent_step.
    _write_world_event_log(
        we_path,
        [_world_event_line(t_event=99, wallclock_ms=9999)],
    )
    rows = [
        {"t": i, "wallclock_ms": i * 100}
        for i in range(0, 10)
    ]
    _write_agent_step_shard(tmp_path / "telemetry" / "agent_step", rows)
    with pytest.raises(PerturbationAlignmentError, match="tolerance_ms"):
        align_perturbations(
            we_path,
            tmp_path / "telemetry",
            run_id="r",
            checkpoint_id="c",
            tolerance_ms=500,
        )


def test_unmatched_with_widened_tolerance_passes(tmp_path: Path) -> None:
    we_path = tmp_path / "telemetry" / "world_event.jsonl"
    _write_world_event_log(
        we_path,
        [_world_event_line(t_event=99, wallclock_ms=2000)],
    )
    rows = [
        {"t": i, "wallclock_ms": i * 100}
        for i in range(0, 10)
    ]
    _write_agent_step_shard(tmp_path / "telemetry" / "agent_step", rows)
    timeline = align_perturbations(
        we_path,
        tmp_path / "telemetry",
        run_id="r",
        checkpoint_id="c",
        tolerance_ms=2000,
    )
    assert len(timeline.events) == 1
    assert timeline.events[0].t == 9  # closest available


def test_empty_agent_step_with_perturbations_raises(tmp_path: Path) -> None:
    we_path = tmp_path / "telemetry" / "world_event.jsonl"
    _write_world_event_log(
        we_path,
        [_world_event_line(t_event=10, wallclock_ms=1000)],
    )
    # No agent_step shards written.
    with pytest.raises(PerturbationAlignmentError, match="empty"):
        align_perturbations(
            we_path,
            tmp_path / "telemetry",
            run_id="r",
            checkpoint_id="c",
        )


# ---------------------------------------------------------------------------
# Sham flag propagation.
# ---------------------------------------------------------------------------


def test_sham_event_propagates_is_sham_true(tmp_path: Path) -> None:
    we_path = tmp_path / "telemetry" / "world_event.jsonl"
    _write_world_event_log(
        we_path,
        [
            _world_event_line(
                t_event=10,
                wallclock_ms=1000,
                payload={"is_sham": True, "kind": "sham"},
            ),
            _world_event_line(
                t_event=20,
                wallclock_ms=2000,
                payload={"kind": "real"},
            ),
        ],
    )
    rows = [
        {"t": i, "wallclock_ms": i * 100}
        for i in range(0, 30)
    ]
    _write_agent_step_shard(tmp_path / "telemetry" / "agent_step", rows)
    timeline = align_perturbations(
        we_path,
        tmp_path / "telemetry",
        run_id="r",
        checkpoint_id="c",
    )
    assert len(timeline.events) == 2
    assert timeline.events[0].is_sham is True
    assert timeline.events[1].is_sham is False
    # The payload is preserved verbatim.
    assert timeline.events[0].payload == {"is_sham": True, "kind": "sham"}
    assert timeline.events[1].payload == {"kind": "real"}


# ---------------------------------------------------------------------------
# Ordering invariants.
# ---------------------------------------------------------------------------


def test_events_sorted_by_t_even_when_log_unordered(tmp_path: Path) -> None:
    we_path = tmp_path / "telemetry" / "world_event.jsonl"
    _write_world_event_log(
        we_path,
        [
            _world_event_line(t_event=30, wallclock_ms=3000),
            _world_event_line(t_event=10, wallclock_ms=1000),
            _world_event_line(t_event=20, wallclock_ms=2000),
        ],
    )
    rows = [
        {"t": i, "wallclock_ms": i * 100}
        for i in range(0, 50)
    ]
    _write_agent_step_shard(tmp_path / "telemetry" / "agent_step", rows)
    timeline = align_perturbations(
        we_path,
        tmp_path / "telemetry",
        run_id="r",
        checkpoint_id="c",
    )
    ts = [e.t for e in timeline.events]
    assert ts == sorted(ts)


def test_timeline_rejects_two_events_at_same_t() -> None:
    with pytest.raises(ValueError, match="same agent-step"):
        PerturbationTimeline(
            events=(
                PerturbationEvent(
                    t=10, wallclock_ms=1000, payload={}, is_sham=False
                ),
                PerturbationEvent(
                    t=10, wallclock_ms=1001, payload={}, is_sham=False
                ),
            ),
            run_id="r",
            checkpoint_id="c",
        )


def test_timeline_rejects_unsorted_events() -> None:
    with pytest.raises(ValueError, match="sorted by t"):
        PerturbationTimeline(
            events=(
                PerturbationEvent(
                    t=20, wallclock_ms=2000, payload={}, is_sham=False
                ),
                PerturbationEvent(
                    t=10, wallclock_ms=1000, payload={}, is_sham=False
                ),
            ),
            run_id="r",
            checkpoint_id="c",
        )


# ---------------------------------------------------------------------------
# Defaults and validation.
# ---------------------------------------------------------------------------


def test_default_tolerance_is_one_second() -> None:
    assert DEFAULT_ALIGNMENT_TOLERANCE_MS == 1000


def test_negative_tolerance_rejected(tmp_path: Path) -> None:
    we_path = tmp_path / "telemetry" / "world_event.jsonl"
    _write_world_event_log(we_path, [])
    with pytest.raises(ValueError, match="non-negative"):
        align_perturbations(
            we_path,
            tmp_path / "telemetry",
            run_id="r",
            checkpoint_id="c",
            tolerance_ms=-1,
        )


def test_perturbation_event_rejects_extras() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PerturbationEvent(
            t=0,
            wallclock_ms=0,
            payload={},
            is_sham=False,
            extra="not allowed",  # type: ignore[call-arg]
        )
