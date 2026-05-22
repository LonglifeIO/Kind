"""Phase 3a gate test for ``kind/env/env_server.py`` and ``mutators.py``.

Plan §4 test #3: invoke ``EnvServer.add_resource((3, 4))``, confirm a
``WorldEvent`` record appears in the world-event sink with the right fields
and that the agent's observation contains no marker that distinguishes the
builder-added resource from a regrowth-added resource. Plus the smaller
unit tests the user spec calls out: lifecycle, the four mutators, episode-
boundary emission ordering, envelope fields, determinism, error cases,
no-op behavior, wallclock monotonicity.

CPU only; no PyTorch is imported here. The harness is synchronous and
in-process — Phase 4 wraps TCP around it via the ``world_event_handler``
seam in :class:`EnvServerConfig` — so these tests also run without any
network or async machinery. Each test owns its own
:class:`~kind.observer.sinks.JsonlSink`; the harness emits records to a
caller-supplied handler (``sink.write`` here), and the test closes the
sink before reading its file. The Phase 4 transport server replaces the
handler with a wire-shipping callable; the tests in
``test_transport.py`` exercise that path.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import (
    CellType,
    EnvStep,
    GridWorldConfig,
)
from kind.observer.schemas import PROBE_1_SCHEMA_VERSION, WorldEvent
from kind.observer.sinks import JsonlSink


# ---- shared helpers --------------------------------------------------------


def _make_quiet_config() -> GridWorldConfig:
    """A grid config with regrowth and drift disabled.

    Used by tests that need the underlying world to stay still so they can
    isolate the harness's behavior from the world's stochasticity.
    """
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
    grid_world_config: GridWorldConfig | None = None,
    seed: int = 42,
    run_id: str = "test-run-001",
    sink_name: str = "world_event.jsonl",
) -> Iterator[tuple[EnvServer, Path]]:
    """Yield ``(server, sink_path)``; manage both EnvServer and JsonlSink.

    The ``world_event_handler`` is bound to the sink's ``write`` so every
    ``WorldEvent`` the harness emits is written to the JSONL file at
    ``sink_path``. On exit, the harness is closed first (idempotent state
    transition) and the sink second (flushes and ``fsync`` to disk). Tests
    read ``sink_path`` after the ``with`` block exits; the sink is closed
    by then so the file is durable.
    """
    sink_path = tmp_path / sink_name
    sink = JsonlSink(sink_path, WorldEvent)
    config = EnvServerConfig(
        grid_world_config=grid_world_config or GridWorldConfig(),
        seed=seed,
        world_event_handler=sink.write,
        run_id=run_id,
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


def _read_world_event_dicts(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


# ---- gate test (plan §4 test #3) ------------------------------------------


def test_gate_perturbation_hook_logged(tmp_path: Path) -> None:
    """Plan's named gate: ``add_resource`` logs the right ``WorldEvent``
    and the agent's observation contains no marker distinguishing the
    builder-added cell from a regrowth-added cell.

    The no-marker check is the parallel-env approach the user spec names:
    construct two envs from the same seed, in one call ``add_resource``, in
    the other set the same cell to ``RESOURCE`` directly via the underlying
    grid (the same state change a regrowth event would have produced), step
    both, and compare observations pixel-for-pixel.
    """
    quiet = _make_quiet_config()
    cell = (3, 4)

    with _make_env_server(
        tmp_path,
        grid_world_config=quiet,
        sink_name="builder.jsonl",
        run_id="builder-run",
    ) as (server_a, path_a), _make_env_server(
        tmp_path,
        grid_world_config=quiet,
        sink_name="natural.jsonl",
        run_id="natural-run",
    ) as (server_b, path_b):
        server_a.start()
        server_b.start()

        # Pre-condition: same seed and same config means both grids start
        # identical, in particular the target cell is empty in both.
        assert server_a.grid_world_state.grid[cell] == CellType.EMPTY.value
        assert server_b.grid_world_state.grid[cell] == CellType.EMPTY.value

        # In server_a: builder-side mutator. This is what Phase 4 onwards
        # eventually wires to a network message.
        server_a.add_resource(cell)
        # In server_b: simulate the same resulting world state via direct
        # grid mutation — the same end-state the natural regrowth process
        # would have produced if it had fired on this cell.
        server_b._grid_world._grid[cell] = CellType.RESOURCE.value  # type: ignore[union-attr]

        # Step both with stay; the worlds advance identically, no regrowth.
        step_a = server_a.step(4)
        step_b = server_b.step(4)

    # Agent's observation contains NO marker distinguishing the two paths.
    assert np.array_equal(step_a.observation, step_b.observation)

    # The world-event stream, by contrast, has full ground truth on the
    # builder side: env_a's sink contains the builder_perturbation record;
    # env_b's does not.
    events_a = _read_world_events(path_a)
    events_b = _read_world_events(path_b)

    # env_a: env_reset (start) + builder_perturbation (add_resource).
    perturbation_records = [
        e for e in events_a if e.event_type == "builder_perturbation"
    ]
    assert len(perturbation_records) == 1
    rec = perturbation_records[0]
    assert rec.source == "builder"
    assert rec.payload["mutator"] == "add_resource"
    assert rec.payload["cell"] == [3, 4]
    assert rec.payload["pre_state"] == "empty"
    assert rec.payload["post_state"] == "resource"

    # env_b: only the env_reset on start (no builder events).
    assert all(e.event_type != "builder_perturbation" for e in events_b)


# ---- lifecycle ------------------------------------------------------------


def test_start_emits_initial_env_reset_and_returns_first_env_step(
    tmp_path: Path,
) -> None:
    with _make_env_server(tmp_path) as (server, sink_path):
        first = server.start()
        assert isinstance(first, EnvStep)
        assert first.env_step == 0
        assert first.episode_id == 0
        assert first.step_in_episode == 0

    events = _read_world_events(sink_path)
    assert len(events) == 1
    initial = events[0]
    assert initial.event_type == "env_reset"
    assert initial.source == "environment"
    assert initial.t_event == 0
    assert initial.payload["episode_id"] == 0
    assert "resource_positions" in initial.payload
    assert "regrowth_p" in initial.payload


def test_step_advances_env_and_returns_env_step(tmp_path: Path) -> None:
    with _make_env_server(
        tmp_path, grid_world_config=_make_quiet_config()
    ) as (server, _sink_path):
        first = server.start()
        next_step = server.step(4)  # stay
        assert next_step.env_step == 1
        assert next_step.episode_id == 0
        assert next_step.step_in_episode == 1
        assert first.observation.shape == next_step.observation.shape


def test_close_flushes_sink_to_disk(tmp_path: Path) -> None:
    with _make_env_server(tmp_path) as (server, sink_path):
        server.start()
    # File exists, has at least one record, and parses back via the schema.
    events = _read_world_events(sink_path)
    assert len(events) >= 1
    assert all(isinstance(e, WorldEvent) for e in events)


def test_close_is_idempotent(tmp_path: Path) -> None:
    with _make_env_server(tmp_path) as (server, _sink_path):
        server.start()
        server.close()
        server.close()  # second call is a no-op


def test_step_before_start_raises(tmp_path: Path) -> None:
    with _make_env_server(tmp_path) as (server, _sink_path):
        with pytest.raises(RuntimeError, match="start"):
            server.step(0)


def test_start_called_twice_raises(tmp_path: Path) -> None:
    with _make_env_server(tmp_path) as (server, _sink_path):
        server.start()
        with pytest.raises(RuntimeError, match="already"):
            server.start()


def test_step_after_close_raises(tmp_path: Path) -> None:
    with _make_env_server(tmp_path) as (server, _sink_path):
        server.start()
        server.close()
        with pytest.raises(RuntimeError, match="closed"):
            server.step(0)


def test_context_manager_calls_close(tmp_path: Path) -> None:
    with _make_env_server(tmp_path) as (server, sink_path):
        server.start()
        server.step(4)
    # On exit, close was called; file is flushed.
    events = _read_world_events(sink_path)
    assert len(events) >= 1


# ---- the four mutators ---------------------------------------------------


def test_add_resource_emits_builder_perturbation_event(tmp_path: Path) -> None:
    with _make_env_server(
        tmp_path, grid_world_config=_make_quiet_config()
    ) as (server, sink_path):
        server.start()
        cell = (2, 5)
        assert server.grid_world_state.grid[cell] == CellType.EMPTY.value
        server.add_resource(cell)
        assert server.grid_world_state.grid[cell] == CellType.RESOURCE.value

    events = _read_world_events(sink_path)
    perts = [e for e in events if e.event_type == "builder_perturbation"]
    assert len(perts) == 1
    p = perts[0]
    assert p.source == "builder"
    assert p.payload["mutator"] == "add_resource"
    assert p.payload["cell"] == [2, 5]
    assert p.payload["pre_state"] == "empty"
    assert p.payload["post_state"] == "resource"


def test_remove_object_emits_builder_perturbation_event(
    tmp_path: Path,
) -> None:
    with _make_env_server(
        tmp_path, grid_world_config=_make_quiet_config()
    ) as (server, sink_path):
        server.start()
        # Set a resource directly so the test's removal is the only
        # perturbation in the stream.
        server._grid_world._grid[1, 2] = CellType.RESOURCE.value  # type: ignore[union-attr]
        server.remove_object((1, 2), CellType.RESOURCE)
        assert server.grid_world_state.grid[1, 2] == CellType.EMPTY.value

    events = _read_world_events(sink_path)
    perts = [e for e in events if e.event_type == "builder_perturbation"]
    assert len(perts) == 1
    p = perts[0]
    assert p.payload["mutator"] == "remove_object"
    assert p.payload["cell"] == [1, 2]
    assert p.payload["object_type"] == "resource"
    assert p.payload["pre_state"] == "resource"
    assert p.payload["post_state"] == "empty"


def test_set_cell_state_emits_builder_perturbation_event(
    tmp_path: Path,
) -> None:
    with _make_env_server(
        tmp_path, grid_world_config=_make_quiet_config()
    ) as (server, sink_path):
        server.start()
        server.set_cell_state((4, 4), CellType.WALL)
        assert server.grid_world_state.grid[4, 4] == CellType.WALL.value

    events = _read_world_events(sink_path)
    perts = [e for e in events if e.event_type == "builder_perturbation"]
    assert len(perts) == 1
    p = perts[0]
    assert p.payload["mutator"] == "set_cell_state"
    assert p.payload["cell"] == [4, 4]
    assert p.payload["pre_state"] == "empty"
    assert p.payload["post_state"] == "wall"


def test_move_object_emits_builder_perturbation_event(
    tmp_path: Path,
) -> None:
    with _make_env_server(
        tmp_path, grid_world_config=_make_quiet_config()
    ) as (server, sink_path):
        server.start()
        server._grid_world._grid[1, 1] = CellType.RESOURCE.value  # type: ignore[union-attr]
        server.move_object((1, 1), (2, 2))
        assert server.grid_world_state.grid[1, 1] == CellType.EMPTY.value
        assert server.grid_world_state.grid[2, 2] == CellType.RESOURCE.value

    events = _read_world_events(sink_path)
    perts = [e for e in events if e.event_type == "builder_perturbation"]
    assert len(perts) == 1
    p = perts[0]
    assert p.payload["mutator"] == "move_object"
    assert p.payload["cell_from"] == [1, 1]
    assert p.payload["cell_to"] == [2, 2]
    assert p.payload["pre_state_from"] == "resource"
    assert p.payload["pre_state_to"] == "empty"
    assert p.payload["post_state_from"] == "empty"
    assert p.payload["post_state_to"] == "resource"


# ---- mutator no-op behavior ----------------------------------------------


def test_add_resource_idempotent_on_existing_resource(tmp_path: Path) -> None:
    """Calling add_resource on a cell that is already a resource succeeds and
    emits a WorldEvent with pre_state == post_state == "resource"."""
    with _make_env_server(
        tmp_path, grid_world_config=_make_quiet_config()
    ) as (server, sink_path):
        server.start()
        server._grid_world._grid[3, 3] = CellType.RESOURCE.value  # type: ignore[union-attr]
        server.add_resource((3, 3))
        assert server.grid_world_state.grid[3, 3] == CellType.RESOURCE.value

    events = _read_world_events(sink_path)
    perts = [e for e in events if e.event_type == "builder_perturbation"]
    assert len(perts) == 1
    assert perts[0].payload["pre_state"] == "resource"
    assert perts[0].payload["post_state"] == "resource"


def test_remove_object_idempotent_on_empty_cell(tmp_path: Path) -> None:
    with _make_env_server(
        tmp_path, grid_world_config=_make_quiet_config()
    ) as (server, sink_path):
        server.start()
        # (1, 1) is empty; remove_object(RESOURCE) is a no-op.
        assert server.grid_world_state.grid[1, 1] == CellType.EMPTY.value
        server.remove_object((1, 1), CellType.RESOURCE)

    events = _read_world_events(sink_path)
    perts = [e for e in events if e.event_type == "builder_perturbation"]
    assert len(perts) == 1
    assert perts[0].payload["pre_state"] == "empty"
    assert perts[0].payload["post_state"] == "empty"


def test_set_cell_state_idempotent_on_matching_state(tmp_path: Path) -> None:
    with _make_env_server(
        tmp_path, grid_world_config=_make_quiet_config()
    ) as (server, sink_path):
        server.start()
        # (1, 1) is empty; set_cell_state(EMPTY) is a no-op but emits.
        server.set_cell_state((1, 1), CellType.EMPTY)

    events = _read_world_events(sink_path)
    perts = [e for e in events if e.event_type == "builder_perturbation"]
    assert len(perts) == 1
    assert perts[0].payload["pre_state"] == "empty"
    assert perts[0].payload["post_state"] == "empty"


# ---- mutator error cases -------------------------------------------------


@pytest.mark.parametrize("cell", [(99, 0), (0, 99), (-1, 0), (0, -1)])
def test_add_resource_out_of_bounds_raises(
    tmp_path: Path, cell: tuple[int, int]
) -> None:
    with _make_env_server(tmp_path) as (server, _sink_path):
        server.start()
        with pytest.raises(ValueError, match="out of grid bounds"):
            server.add_resource(cell)


def test_add_resource_on_wall_raises(tmp_path: Path) -> None:
    with _make_env_server(
        tmp_path, grid_world_config=GridWorldConfig(walls=((4, 4),))
    ) as (server, _sink_path):
        server.start()
        with pytest.raises(ValueError, match="wall"):
            server.add_resource((4, 4))


def test_remove_object_invalid_object_type_raises(tmp_path: Path) -> None:
    with _make_env_server(tmp_path) as (server, _sink_path):
        server.start()
        # Plain ints are rejected even when they happen to match a CellType
        # value; the public API requires CellType instances.
        with pytest.raises(TypeError, match="CellType"):
            server.remove_object((0, 0), 2)  # type: ignore[arg-type]


def test_remove_object_empty_object_type_raises(tmp_path: Path) -> None:
    with _make_env_server(tmp_path) as (server, _sink_path):
        server.start()
        with pytest.raises(ValueError, match="non-empty"):
            server.remove_object((0, 0), CellType.EMPTY)


def test_remove_object_type_mismatch_raises(tmp_path: Path) -> None:
    with _make_env_server(
        tmp_path, grid_world_config=_make_quiet_config()
    ) as (server, _sink_path):
        server.start()
        server._grid_world._grid[2, 3] = CellType.WALL.value  # type: ignore[union-attr]
        with pytest.raises(ValueError, match="not"):
            server.remove_object((2, 3), CellType.RESOURCE)


def test_set_cell_state_invalid_state_raises(tmp_path: Path) -> None:
    with _make_env_server(tmp_path) as (server, _sink_path):
        server.start()
        with pytest.raises(TypeError, match="CellType"):
            server.set_cell_state((0, 0), 2)  # type: ignore[arg-type]


def test_move_object_same_cell_raises(tmp_path: Path) -> None:
    with _make_env_server(
        tmp_path, grid_world_config=_make_quiet_config()
    ) as (server, _sink_path):
        server.start()
        server._grid_world._grid[2, 2] = CellType.RESOURCE.value  # type: ignore[union-attr]
        with pytest.raises(ValueError, match="differ"):
            server.move_object((2, 2), (2, 2))


def test_move_object_empty_source_raises(tmp_path: Path) -> None:
    with _make_env_server(
        tmp_path, grid_world_config=_make_quiet_config()
    ) as (server, _sink_path):
        server.start()
        with pytest.raises(ValueError, match="empty"):
            server.move_object((1, 1), (2, 2))


def test_move_object_nonempty_destination_raises(tmp_path: Path) -> None:
    with _make_env_server(
        tmp_path, grid_world_config=_make_quiet_config()
    ) as (server, _sink_path):
        server.start()
        server._grid_world._grid[1, 1] = CellType.RESOURCE.value  # type: ignore[union-attr]
        server._grid_world._grid[2, 2] = CellType.RESOURCE.value  # type: ignore[union-attr]
        with pytest.raises(ValueError, match="not empty"):
            server.move_object((1, 1), (2, 2))


def test_move_object_out_of_bounds_raises(tmp_path: Path) -> None:
    with _make_env_server(
        tmp_path, grid_world_config=_make_quiet_config()
    ) as (server, _sink_path):
        server.start()
        server._grid_world._grid[1, 1] = CellType.RESOURCE.value  # type: ignore[union-attr]
        with pytest.raises(ValueError, match="out of grid bounds"):
            server.move_object((1, 1), (99, 0))


# ---- envelope fields -----------------------------------------------------


def test_world_event_envelope_fields(tmp_path: Path) -> None:
    """schema_version, run_id, checkpoint_id are populated on every record."""
    with _make_env_server(
        tmp_path,
        grid_world_config=_make_quiet_config(),
        run_id="envelope-test-007",
    ) as (server, sink_path):
        server.start()
        server.add_resource((2, 2))

    events = _read_world_events(sink_path)
    assert len(events) >= 2
    for e in events:
        assert e.schema_version == PROBE_1_SCHEMA_VERSION
        assert e.schema_version == "0.1.0"
        assert e.run_id == "envelope-test-007"
        assert e.checkpoint_id is None  # Probe 1 default


def test_set_checkpoint_id_propagates_to_subsequent_events(
    tmp_path: Path,
) -> None:
    with _make_env_server(
        tmp_path, grid_world_config=_make_quiet_config()
    ) as (server, sink_path):
        server.start()  # initial env_reset has checkpoint_id=None
        server.add_resource((1, 1))  # also None
        server.set_checkpoint_id("ckpt-42")
        server.set_cell_state((2, 2), CellType.WALL)  # has ckpt-42
        server.set_checkpoint_id(None)
        server.add_resource((3, 3))  # back to None

    events = _read_world_events(sink_path)
    # First two events are pre-checkpoint (env_reset, add_resource)
    assert events[0].checkpoint_id is None
    assert events[1].checkpoint_id is None
    assert events[2].checkpoint_id == "ckpt-42"
    assert events[3].checkpoint_id is None


# ---- episode boundary ----------------------------------------------------


def test_episode_boundary_emits_aggregate_then_env_reset(
    tmp_path: Path,
) -> None:
    """At the boundary, the closing-episode aggregate is emitted before the
    new-episode env_reset — the user spec's defined order: close the old
    before opening the new."""
    with _make_env_server(
        tmp_path, grid_world_config=GridWorldConfig(episode_length=5)
    ) as (server, sink_path):
        server.start()
        # 5 steps: the 5th crosses the boundary.
        for _ in range(5):
            server.step(4)

    events = _read_world_events(sink_path)
    # Expected: env_reset (start), agg (boundary), env_reset (boundary)
    types = [e.event_type for e in events]
    assert types == [
        "env_reset",
        "internal_stochasticity_aggregate",
        "env_reset",
    ]
    # Both boundary events have the same t_event = episode_length.
    assert events[1].t_event == 5
    assert events[2].t_event == 5
    # Aggregate references the closing episode (0); env_reset references new (1).
    assert events[1].payload["episode_id"] == 0
    assert events[2].payload["episode_id"] == 1


def test_internal_stochasticity_aggregate_payload_keys(
    tmp_path: Path,
) -> None:
    """The aggregate payload has the expected keys with finite values."""
    with _make_env_server(
        tmp_path,
        grid_world_config=GridWorldConfig(
            episode_length=10,
            initial_regrowth_p=0.3,  # high so we get visible regrowth
            drift_p_min=0.1,
            drift_p_max=0.5,
            drift_magnitude_per_step=0.001,
            n_initial_resources=0,
        ),
    ) as (server, sink_path):
        server.start()
        for _ in range(10):
            server.step(4)

    events = _read_world_events(sink_path)
    aggs = [
        e for e in events if e.event_type == "internal_stochasticity_aggregate"
    ]
    assert len(aggs) == 1
    p = aggs[0].payload
    assert set(p.keys()) == {
        "episode_id",
        "regrowth_events",
        "mean_drift_step_magnitude",
        "final_p",
    }
    assert isinstance(p["episode_id"], int)
    assert isinstance(p["regrowth_events"], int)
    assert p["regrowth_events"] >= 0
    assert isinstance(p["mean_drift_step_magnitude"], float)
    assert np.isfinite(p["mean_drift_step_magnitude"])
    assert isinstance(p["final_p"], float)
    assert np.isfinite(p["final_p"])
    assert 0.1 <= p["final_p"] <= 0.5  # within drift bounds


def test_aggregate_counts_regrowth_only_not_builder_mutations(
    tmp_path: Path,
) -> None:
    """A builder ``add_resource`` between steps must not be counted as a
    regrowth event in the per-episode aggregate."""
    with _make_env_server(
        tmp_path,
        grid_world_config=GridWorldConfig(
            episode_length=5,
            initial_regrowth_p=0.0,
            drift_p_min=0.0,
            drift_p_max=0.0,
            drift_magnitude_per_step=0.0,
            n_initial_resources=0,
        ),
    ) as (server, sink_path):
        server.start()
        # Step 1: stay. No regrowth (p=0).
        server.step(4)
        # Builder adds a resource.
        server.add_resource((2, 5))
        # Step 2..5: stay. No regrowth.
        for _ in range(4):
            server.step(4)

    events = _read_world_events(sink_path)
    aggs = [
        e for e in events if e.event_type == "internal_stochasticity_aggregate"
    ]
    assert len(aggs) == 1
    # The builder add must not bleed into the regrowth count.
    assert aggs[0].payload["regrowth_events"] == 0


# ---- determinism --------------------------------------------------------


def test_determinism_with_mutators(tmp_path: Path) -> None:
    """Two EnvServer instances with the same config and seed, the same
    action sequence, and the same mutator invocations at the same env steps,
    produce identical observation streams."""
    actions = [0, 1, 2, 3, 4] * 10

    def run(sink_name: str) -> list[np.ndarray]:
        observations: list[np.ndarray] = []
        with _make_env_server(tmp_path, sink_name=sink_name) as (server, _path):
            first = server.start()
            observations.append(first.observation)
            for i, action in enumerate(actions):
                if i == 7:
                    server.add_resource((5, 5))
                if i == 13:
                    server.set_cell_state((6, 6), CellType.WALL)
                step = server.step(action)
                observations.append(step.observation)
        return observations

    obs_a = run("a.jsonl")
    obs_b = run("b.jsonl")
    assert len(obs_a) == len(obs_b)
    for o_a, o_b in zip(obs_a, obs_b):
        assert np.array_equal(o_a, o_b)


# ---- wallclock monotonicity ---------------------------------------------


def test_wallclock_monotonic_across_events(tmp_path: Path) -> None:
    """Wallclocks across all WorldEvent records are non-decreasing.

    This is the property the harness's monotonic clock guarantees; the test
    exercises a run with mutators interleaved between steps and across one
    episode boundary so the event list is varied.
    """
    with _make_env_server(
        tmp_path, grid_world_config=GridWorldConfig(episode_length=5)
    ) as (server, sink_path):
        server.start()
        for i in range(12):
            if i % 3 == 0:
                # Builder events scattered through the run.
                cell = (i % 8, (i + 1) % 8)
                grid_value = server.grid_world_state.grid[cell]
                if grid_value == CellType.EMPTY.value:
                    server.add_resource(cell)
            server.step(i % 5)

    events = _read_world_events(sink_path)
    wallclocks = [e.wallclock_ms for e in events]
    for prev, curr in zip(wallclocks, wallclocks[1:]):
        assert curr >= prev, (
            f"wallclock went backwards: {prev} → {curr}"
        )


def test_env_step_wallclock_uses_harness_clock(tmp_path: Path) -> None:
    """The harness overrides EnvStep.wallclock_ms with its monotonic clock,
    so EnvStep wallclocks are also non-decreasing across calls."""
    with _make_env_server(
        tmp_path, grid_world_config=_make_quiet_config()
    ) as (server, _sink_path):
        first = server.start()
        wallclocks = [first.wallclock_ms]
        for _ in range(20):
            wallclocks.append(server.step(4).wallclock_ms)
    for prev, curr in zip(wallclocks, wallclocks[1:]):
        assert curr >= prev


# ---- observation stream invariants -----------------------------------


def test_mutator_does_not_emit_record_into_observation(tmp_path: Path) -> None:
    """The agent's observation contains only the four legal pixel values
    {0, 64, 128, 255} regardless of whether builder mutators have fired —
    the no-marker commitment at the rendering level."""
    with _make_env_server(
        tmp_path, grid_world_config=_make_quiet_config()
    ) as (server, _sink_path):
        server.start()
        server.add_resource((2, 2))
        server.set_cell_state((3, 3), CellType.WALL)
        server._grid_world._grid[4, 4] = CellType.RESOURCE.value  # type: ignore[union-attr]
        server.move_object((4, 4), (5, 5))
        step = server.step(4)
    valid = {0, 64, 128, 255}
    assert set(np.unique(step.observation).tolist()).issubset(valid)


# ---- mutator effects vs. observation rendering ---------------------------


def test_mutator_effect_visible_in_next_observation_when_in_view(
    tmp_path: Path,
) -> None:
    """A builder add_resource within the agent's ego-centric view shows up
    as a resource pixel in the next observation."""
    # Pin start_cell so the in-view assertion is deterministic post the
    # Probe 2 default flip from (3, 3) to None (random non-wall sample).
    # With start_cell=(3, 3) and view_size=7 the view spans rows 0..6 and
    # cols 0..6; (3, 4) is one cell right of center.
    quiet = replace(_make_quiet_config(), start_cell=(3, 3))
    with _make_env_server(tmp_path, grid_world_config=quiet) as (server, _path):
        before = server.start().observation.copy()
        server.add_resource((3, 4))
        after = server.step(4).observation
    # Center of obs = agent position. One view-cell to the right is the
    # changed cell. Specifically:
    #   center pixel is at (obs_resolution // 2, obs_resolution // 2)
    # After add_resource, that view cell renders as RESOURCE (pixel 255).
    # The two observations must differ at the changed cell.
    assert not np.array_equal(before, after)
    # Specifically, somewhere in the observation a pixel changed from
    # EMPTY (128) to RESOURCE (255).
    assert (after == 255).sum() > (before == 255).sum()


# ---- cleanup of state on context manager exit --------------------------


def test_close_writes_complete_record_set(tmp_path: Path) -> None:
    """After close, every line in the JSONL is parseable and the records
    are in the order they were emitted."""
    with _make_env_server(tmp_path) as (server, sink_path):
        server.start()
        for i in range(5):
            server.step(i % 5)
        server.add_resource((4, 7))

    raw_dicts = _read_world_event_dicts(sink_path)
    parsed = [WorldEvent.model_validate(d) for d in raw_dicts]
    # Every record has an envelope and the right payload shape.
    assert all(e.schema_version == PROBE_1_SCHEMA_VERSION for e in parsed)
    # The last record is the builder_perturbation we just emitted.
    assert parsed[-1].event_type == "builder_perturbation"
    assert parsed[-1].payload["mutator"] == "add_resource"


# ---- handler indirection -------------------------------------------------


def test_world_event_handler_is_called_directly(tmp_path: Path) -> None:
    """The handler in :class:`EnvServerConfig` receives every emitted
    ``WorldEvent``. Tests collect into a list to verify the harness does
    not buffer or drop events; this is the seam Phase 4's transport server
    overrides via :meth:`EnvServer.set_world_event_handler`.
    """
    captured: list[WorldEvent] = []
    config = EnvServerConfig(
        grid_world_config=_make_quiet_config(),
        seed=42,
        world_event_handler=captured.append,
        run_id="handler-test",
    )
    server = EnvServer(config)
    server.start()
    server.add_resource((2, 2))
    server.close()

    assert len(captured) == 2
    assert captured[0].event_type == "env_reset"
    assert captured[1].event_type == "builder_perturbation"


def test_set_world_event_handler_replaces_handler_mid_run(
    tmp_path: Path,
) -> None:
    """:meth:`EnvServer.set_world_event_handler` redirects subsequent
    emissions to the new handler. Phase 4's transport server uses this to
    take over the harness's emissions when a client connects.
    """
    captured_a: list[WorldEvent] = []
    captured_b: list[WorldEvent] = []
    config = EnvServerConfig(
        grid_world_config=_make_quiet_config(),
        seed=42,
        world_event_handler=captured_a.append,
        run_id="set-handler-test",
    )
    server = EnvServer(config)
    server.start()  # → captured_a
    server.set_world_event_handler(captured_b.append)
    server.add_resource((2, 2))  # → captured_b
    server.close()

    assert len(captured_a) == 1
    assert captured_a[0].event_type == "env_reset"
    assert len(captured_b) == 1
    assert captured_b[0].event_type == "builder_perturbation"
