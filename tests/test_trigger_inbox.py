"""Probe 4 Phase 2 — trigger tagging + manual-trigger inbox (plan §S-PERT).

Covers: the ``trigger`` kwarg on the env-server mutator surface (tagging,
vocabulary validation, legacy byte-identity), the spool-drain semantics
(fire, archive, result sidecars, never-crash), and the pixel-equality
no-marker property for generator-tagged events (the plan's Phase 2 gate
requirement that the observation-marker-free gate holds for generator
events).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import CellType, GridWorldConfig
from kind.env.trigger_inbox import (
    drain_trigger_inbox,
    write_trigger_request,
)
from kind.observer.schemas import WorldEvent
from kind.observer.sinks import JsonlSink


def _quiet_config() -> GridWorldConfig:
    return GridWorldConfig(
        initial_regrowth_p=0.0,
        drift_p_min=0.0,
        drift_p_max=0.0,
        drift_magnitude_per_step=0.0,
        n_initial_resources=0,
        start_cell=(3, 3),
    )


@contextmanager
def _make_env_server(
    tmp_path: Path, *, sink_name: str = "world_event.jsonl"
) -> Iterator[tuple[EnvServer, Path]]:
    sink_path = tmp_path / sink_name
    sink = JsonlSink(sink_path, WorldEvent)
    server = EnvServer(
        EnvServerConfig(
            grid_world_config=_quiet_config(),
            seed=42,
            world_event_handler=sink.write,
            run_id="trigger-test",
        )
    )
    try:
        server.start()
        yield server, sink_path
    finally:
        try:
            server.close()
        finally:
            sink.close()


def _read_builder_events(path: Path) -> list[WorldEvent]:
    return [
        record
        for record in (
            WorldEvent.model_validate_json(line)
            for line in path.read_text().splitlines()
            if line.strip()
        )
        if record.event_type == "builder_perturbation"
    ]


# ---- trigger tagging on the mutator surface --------------------------------


def test_trigger_tag_lands_in_payload(tmp_path: Path) -> None:
    with _make_env_server(tmp_path) as (server, sink_path):
        server.add_resource((1, 1), trigger="generator")
        server.add_resource((1, 2), trigger="manual")

    events = _read_builder_events(sink_path)
    assert [e.payload["trigger"] for e in events] == ["generator", "manual"]
    for event in events:
        assert event.source == "builder"


def test_all_four_mutators_accept_trigger(tmp_path: Path) -> None:
    with _make_env_server(tmp_path) as (server, sink_path):
        server.add_resource((1, 1), trigger="manual")
        server.remove_object((1, 1), CellType.RESOURCE, trigger="manual")
        server.set_cell_state((2, 2), CellType.WALL, trigger="manual")
        server.add_resource((5, 5), trigger="manual")
        server.move_object((5, 5), (5, 6), trigger="manual")

    events = _read_builder_events(sink_path)
    assert len(events) == 5
    assert all(e.payload["trigger"] == "manual" for e in events)


def test_no_trigger_keeps_legacy_payload(tmp_path: Path) -> None:
    """The default emits the exact pre-Probe-4 payload — no trigger key."""
    with _make_env_server(tmp_path) as (server, sink_path):
        server.add_resource((1, 1))

    events = _read_builder_events(sink_path)
    assert len(events) == 1
    assert "trigger" not in events[0].payload
    assert set(events[0].payload.keys()) == {
        "mutator",
        "cell",
        "pre_state",
        "post_state",
    }


def test_invalid_trigger_raises(tmp_path: Path) -> None:
    with _make_env_server(tmp_path) as (server, _sink_path):
        with pytest.raises(ValueError, match="trigger"):
            server.add_resource((1, 1), trigger="automatic")


# ---- the no-marker property for generator events ---------------------------


def test_generator_tagged_event_is_pixel_equal_to_natural(
    tmp_path: Path,
) -> None:
    """The gate property, replicated for the generator path: a
    generator-tagged ``add_resource`` and a direct grid write (what
    regrowth would have produced) yield pixel-identical observations —
    the trigger tag lives only in the world_event stream."""
    cell = (3, 4)
    with _make_env_server(tmp_path, sink_name="gen.jsonl") as (
        server_a,
        path_a,
    ), _make_env_server(tmp_path, sink_name="nat.jsonl") as (
        server_b,
        _path_b,
    ):
        server_a.add_resource(cell, trigger="generator")
        server_b._grid_world._grid[cell] = CellType.RESOURCE.value  # type: ignore[union-attr]
        step_a = server_a.step(4)
        step_b = server_b.step(4)

    assert np.array_equal(step_a.observation, step_b.observation)
    events = _read_builder_events(path_a)
    assert len(events) == 1
    assert events[0].payload["trigger"] == "generator"


# ---- inbox drain ------------------------------------------------------------


def test_drain_fires_and_archives_with_result(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    request = write_trigger_request(inbox, "add_resource", {"cell": [2, 5]})
    with _make_env_server(tmp_path) as (server, sink_path):
        results = drain_trigger_inbox(inbox, server)
        assert server.grid_world_state.grid[2, 5] == CellType.RESOURCE.value

    assert len(results) == 1
    assert results[0].ok and results[0].error is None
    # Request archived; sidecar written.
    assert not request.exists()
    archived = inbox / "processed" / request.name
    assert archived.exists()
    sidecar = inbox / "processed" / (request.name + ".result.json")
    assert json.loads(sidecar.read_text())["ok"] is True
    # The fired event is manual-tagged.
    events = _read_builder_events(sink_path)
    assert len(events) == 1
    assert events[0].payload["trigger"] == "manual"
    assert events[0].payload["cell"] == [2, 5]


def test_drain_processes_requests_in_submission_order(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    write_trigger_request(inbox, "add_resource", {"cell": [1, 1]})
    write_trigger_request(inbox, "add_resource", {"cell": [1, 2]})
    write_trigger_request(inbox, "add_resource", {"cell": [1, 3]})
    with _make_env_server(tmp_path) as (server, sink_path):
        results = drain_trigger_inbox(inbox, server)

    assert [r.ok for r in results] == [True, True, True]
    events = _read_builder_events(sink_path)
    assert [e.payload["cell"] for e in events] == [[1, 1], [1, 2], [1, 3]]


def test_drain_survives_invalid_requests(tmp_path: Path) -> None:
    """Malformed JSON, unknown mutators, and mutator validation errors
    all become archived error results — the run never crashes."""
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True)
    (inbox / "0-broken.json").write_text("{not json", encoding="utf-8")
    write_trigger_request(inbox, "teleport_agent", {"cell": [1, 1]})
    write_trigger_request(inbox, "add_resource", {"cell": [99, 99]})
    write_trigger_request(inbox, "add_resource", {"cell": [4, 4]})

    with _make_env_server(tmp_path) as (server, sink_path):
        results = drain_trigger_inbox(inbox, server)
        assert server.grid_world_state.grid[4, 4] == CellType.RESOURCE.value

    assert [r.ok for r in results] == [False, False, False, True]
    assert results[0].error is not None  # malformed JSON
    assert "unknown mutator" in str(results[1].error)
    assert "out of grid bounds" in str(results[2].error)
    # Exactly one real event fired.
    assert len(_read_builder_events(sink_path)) == 1
    # Inbox is empty; everything archived.
    assert list(inbox.glob("*.json")) == []


def test_drain_decodes_cell_types(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    write_trigger_request(
        inbox, "set_cell_state", {"cell": [2, 2], "state": "wall"}
    )
    with _make_env_server(tmp_path) as (server, _sink_path):
        results = drain_trigger_inbox(inbox, server)
        assert server.grid_world_state.grid[2, 2] == CellType.WALL.value
    assert results[0].ok


def test_drain_on_missing_inbox_is_noop(tmp_path: Path) -> None:
    with _make_env_server(tmp_path) as (server, _sink_path):
        assert drain_trigger_inbox(tmp_path / "nope", server) == []
