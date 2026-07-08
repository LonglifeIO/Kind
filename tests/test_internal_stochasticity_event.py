"""Probe 4 Phase 1 — per-event internal-stochasticity logging (plan §S-CTRL).

The ENVIRONMENT class of the three-way matched control: each regrowth
resource-addition emits one granular ``internal_stochasticity_event``
``WorldEvent`` when ``EnvServerConfig.emit_internal_stochasticity_events``
is on, with a payload whose comparison keys exactly match a builder
``add_resource`` payload. Default off keeps legacy emission byte-identical
(the house opt-in pattern); the pixel-equality gate and every existing
env-server test run against the default and stay green.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import CellType, GridWorld, GridWorldConfig
from kind.observer.schemas import (
    PROBE_4_WORLD_EVENT_SCHEMA_VERSION,
    WorldEvent,
    export_json_schema_v0_7_0,
    export_json_schema_v0_8_0,
)
from kind.observer.sinks import JsonlSink


def _regrowth_heavy_config() -> GridWorldConfig:
    """High regrowth so short runs produce visible per-event emission."""
    return GridWorldConfig(
        episode_length=30,
        initial_regrowth_p=0.3,
        drift_p_min=0.1,
        drift_p_max=0.5,
        drift_magnitude_per_step=0.001,
        n_initial_resources=0,
        start_cell=(3, 3),
    )


def _quiet_config() -> GridWorldConfig:
    return GridWorldConfig(
        initial_regrowth_p=0.0,
        drift_p_min=0.0,
        drift_p_max=0.0,
        drift_magnitude_per_step=0.0,
        n_initial_resources=0,
    )


@contextmanager
def _make_env_server(
    tmp_path: Path,
    *,
    grid_world_config: GridWorldConfig,
    emit_internal_stochasticity_events: bool,
    seed: int = 42,
    sink_name: str = "world_event.jsonl",
) -> Iterator[tuple[EnvServer, Path]]:
    sink_path = tmp_path / sink_name
    sink = JsonlSink(sink_path, WorldEvent)
    config = EnvServerConfig(
        grid_world_config=grid_world_config,
        seed=seed,
        world_event_handler=sink.write,
        run_id="granular-test",
        emit_internal_stochasticity_events=emit_internal_stochasticity_events,
    )
    server = EnvServer(config)
    try:
        yield server, sink_path
    finally:
        try:
            server.close()
        finally:
            sink.close()


def _read_world_events(path: Path) -> list[WorldEvent]:
    return [
        WorldEvent.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


# ---- default off: legacy emission unchanged --------------------------------


def test_default_off_emits_no_granular_events(tmp_path: Path) -> None:
    """With the flag at its default, a regrowth-heavy run emits no
    ``internal_stochasticity_event`` records — legacy emission preserved."""
    with _make_env_server(
        tmp_path,
        grid_world_config=_regrowth_heavy_config(),
        emit_internal_stochasticity_events=False,
    ) as (server, sink_path):
        server.start()
        for _ in range(20):
            server.step(4)

    events = _read_world_events(sink_path)
    assert all(
        e.event_type != "internal_stochasticity_event" for e in events
    )
    # The aggregate path is untouched and counted regrowth.
    aggs = [
        e for e in events if e.event_type == "internal_stochasticity_aggregate"
    ]
    assert not aggs or aggs[0].payload["regrowth_events"] >= 0


# ---- flag on: one granular event per regrowth addition ---------------------


def test_granular_events_match_independent_grid_diff(tmp_path: Path) -> None:
    """Every EMPTY→RESOURCE transition the world produces on non-boundary
    steps appears as exactly one granular event with the right cell and
    ``t_event``, verified against an independent same-seed GridWorld
    replica diffed step by step."""
    config = _regrowth_heavy_config()
    seed = 7
    n_steps = 20  # inside one episode (episode_length=30)

    # Independent replica: diff pre/post grids per step.
    replica = GridWorld(config, seed)
    replica.reset()
    expected: list[tuple[int, tuple[int, int]]] = []
    for _ in range(n_steps):
        pre = replica.state.grid.copy()
        step = replica.step(4)
        post = replica.state.grid
        mask = (pre == CellType.EMPTY.value) & (
            post == CellType.RESOURCE.value
        )
        for row, col in np.argwhere(mask):
            expected.append((step.env_step, (int(row), int(col))))

    with _make_env_server(
        tmp_path,
        grid_world_config=config,
        emit_internal_stochasticity_events=True,
        seed=seed,
    ) as (server, sink_path):
        server.start()
        for _ in range(n_steps):
            server.step(4)

    events = _read_world_events(sink_path)
    granular = [
        e for e in events if e.event_type == "internal_stochasticity_event"
    ]
    assert len(granular) > 0, "regrowth-heavy run must produce events"
    emitted = [
        (e.t_event, (e.payload["cell"][0], e.payload["cell"][1]))
        for e in granular
    ]
    assert emitted == expected


def test_granular_event_record_shape(tmp_path: Path) -> None:
    """Envelope and payload of a granular event: source, version, and the
    builder-comparable payload shape (pre-registration §1)."""
    with _make_env_server(
        tmp_path,
        grid_world_config=_regrowth_heavy_config(),
        emit_internal_stochasticity_events=True,
    ) as (server, sink_path):
        server.start()
        for _ in range(20):
            server.step(4)

    granular = [
        e
        for e in _read_world_events(sink_path)
        if e.event_type == "internal_stochasticity_event"
    ]
    assert len(granular) > 0
    for event in granular:
        assert event.source == "environment"
        assert event.schema_version == PROBE_4_WORLD_EVENT_SCHEMA_VERSION
        assert set(event.payload.keys()) == {
            "process",
            "cell",
            "pre_state",
            "post_state",
        }
        assert event.payload["process"] == "regrowth"
        assert event.payload["pre_state"] == "empty"
        assert event.payload["post_state"] == "resource"
        row, col = event.payload["cell"]
        assert 0 <= row < 8 and 0 <= col < 8


def test_granular_payload_comparable_with_builder_add_resource(
    tmp_path: Path,
) -> None:
    """The comparison keys (cell / pre_state / post_state) of the granular
    ENVIRONMENT event exactly match a builder ``add_resource`` payload —
    the direct per-event comparability the matched control requires."""
    with _make_env_server(
        tmp_path,
        grid_world_config=_regrowth_heavy_config(),
        emit_internal_stochasticity_events=True,
    ) as (server, sink_path):
        server.start()
        server.add_resource((0, 0))
        for _ in range(20):
            server.step(4)

    events = _read_world_events(sink_path)
    builder = [e for e in events if e.event_type == "builder_perturbation"]
    granular = [
        e for e in events if e.event_type == "internal_stochasticity_event"
    ]
    assert builder and granular
    builder_keys = set(builder[0].payload.keys()) - {"mutator"}
    granular_keys = set(granular[0].payload.keys()) - {"process"}
    assert builder_keys == granular_keys == {"cell", "pre_state", "post_state"}
    # Same value vocabulary on the comparison keys.
    assert builder[0].payload["pre_state"] == granular[0].payload["pre_state"]
    assert (
        builder[0].payload["post_state"] == granular[0].payload["post_state"]
    )


def test_builder_mutation_not_misattributed_as_granular_event(
    tmp_path: Path,
) -> None:
    """A builder ``add_resource`` between steps must not surface as an
    ENVIRONMENT-class granular event (the pre-step snapshot discipline)."""
    with _make_env_server(
        tmp_path,
        grid_world_config=_quiet_config(),
        emit_internal_stochasticity_events=True,
    ) as (server, sink_path):
        server.start()
        server.step(4)
        server.add_resource((2, 5))
        for _ in range(4):
            server.step(4)

    events = _read_world_events(sink_path)
    assert all(
        e.event_type != "internal_stochasticity_event" for e in events
    )
    builder = [e for e in events if e.event_type == "builder_perturbation"]
    assert len(builder) == 1  # the builder event itself is logged as builder


def test_granular_count_equals_aggregate_count(tmp_path: Path) -> None:
    """Within one episode, the number of granular events equals the
    per-episode aggregate's ``regrowth_events`` — the aggregate path is
    unchanged and the two views of the same diff agree."""
    config = _regrowth_heavy_config()
    with _make_env_server(
        tmp_path,
        grid_world_config=config,
        emit_internal_stochasticity_events=True,
    ) as (server, sink_path):
        server.start()
        # Cross exactly one episode boundary (episode_length=30).
        for _ in range(30):
            server.step(4)

    events = _read_world_events(sink_path)
    aggs = [
        e for e in events if e.event_type == "internal_stochasticity_aggregate"
    ]
    assert len(aggs) == 1
    granular_episode_0 = [
        e
        for e in events
        if e.event_type == "internal_stochasticity_event"
        and e.t_event <= aggs[0].t_event
    ]
    assert aggs[0].payload["regrowth_events"] == len(granular_episode_0)


def test_granular_events_deterministic(tmp_path: Path) -> None:
    """Same seed, same actions → identical granular event sequences."""

    def run(sink_name: str) -> list[tuple[int, list[int]]]:
        with _make_env_server(
            tmp_path,
            grid_world_config=_regrowth_heavy_config(),
            emit_internal_stochasticity_events=True,
            seed=99,
            sink_name=sink_name,
        ) as (server, sink_path):
            server.start()
            for i in range(20):
                server.step(i % 5)
        return [
            (e.t_event, list(e.payload["cell"]))
            for e in _read_world_events(sink_path)
            if e.event_type == "internal_stochasticity_event"
        ]

    assert run("a.jsonl") == run("b.jsonl")


# ---- schema discipline ------------------------------------------------------


def test_granular_event_round_trips_through_jsonl(tmp_path: Path) -> None:
    with _make_env_server(
        tmp_path,
        grid_world_config=_regrowth_heavy_config(),
        emit_internal_stochasticity_events=True,
    ) as (server, sink_path):
        server.start()
        for _ in range(20):
            server.step(4)

    for line in sink_path.read_text().splitlines():
        record = WorldEvent.model_validate_json(line)
        assert record.event_type in {
            "env_reset",
            "internal_stochasticity_aggregate",
            "internal_stochasticity_event",
        }


def test_validator_rejects_granular_event_at_legacy_version() -> None:
    """Mixed-version writer rejection: the new event type must stamp the
    Probe 4 WorldEvent record version."""
    with pytest.raises(ValueError, match="internal_stochasticity_event"):
        WorldEvent(
            schema_version="0.1.0",
            run_id="r",
            checkpoint_id=None,
            t_event=1,
            event_type="internal_stochasticity_event",
            source="environment",
            payload={
                "process": "regrowth",
                "cell": [1, 2],
                "pre_state": "empty",
                "post_state": "resource",
            },
            wallclock_ms=0,
        )


def test_validator_rejects_granular_event_missing_payload_keys() -> None:
    with pytest.raises(ValueError, match="matched control"):
        WorldEvent(
            schema_version=PROBE_4_WORLD_EVENT_SCHEMA_VERSION,
            run_id="r",
            checkpoint_id=None,
            t_event=1,
            event_type="internal_stochasticity_event",
            source="environment",
            payload={"cell": [1, 2]},
            wallclock_ms=0,
        )


def test_validator_leaves_other_event_types_untouched() -> None:
    """Legacy records at legacy versions still construct (backward read)."""
    record = WorldEvent(
        schema_version="0.1.0",
        run_id="r",
        checkpoint_id=None,
        t_event=1,
        event_type="builder_perturbation",
        source="builder",
        payload={"mutator": "add_resource", "cell": [1, 2]},
        wallclock_ms=0,
    )
    assert record.event_type == "builder_perturbation"


# ---- exports ----------------------------------------------------------------


def test_v0_7_0_export_is_frozen_and_matches_disk() -> None:
    """The superseded export reads its checked-in bytes (house pattern)."""
    disk = Path(__file__).resolve().parents[1] / "schemas" / "v0.7.0.json"
    assert export_json_schema_v0_7_0() == disk.read_bytes()


def test_v0_8_0_export_byte_stable_matches_disk_and_carries_the_literal() -> (
    None
):
    """The Probe 4 export is byte-stable, matches ``schemas/v0.8.0.json``,
    and its WorldEvent model carries the granular event type. If a model
    intentionally changes, regenerate via export_json_schema_v0_8_0 and
    commit."""
    import json

    first = export_json_schema_v0_8_0()
    assert first == export_json_schema_v0_8_0()
    disk = Path(__file__).resolve().parents[1] / "schemas" / "v0.8.0.json"
    assert disk.read_bytes() == first, (
        "schemas/v0.8.0.json no longer matches the live models — regenerate "
        "via export_json_schema_v0_8_0() and commit the result."
    )
    document = json.loads(first)
    world_event = document["models"]["telemetry"]["WorldEvent"]
    assert (
        "internal_stochasticity_event"
        in world_event["properties"]["event_type"]["enum"]
    )
    assert document["schema_version"] == "0.8.0"
    assert document["world_event_schema_version"] == (
        PROBE_4_WORLD_EVENT_SCHEMA_VERSION
    )
