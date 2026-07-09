"""World v2 W5 — E4: the mover pilot (plan W5; synthesis E4, DP3).

Pins the mover both ways. Default off is byte-identical (including the
RNG-stream contract: the fourth spawned stream leaves the original
three untouched). Enabled: the mover wanders on its cadence with a
persistent heading (deterministic per seed), bounces off walls / edges
/ objects / Io, never tramples anything (moves only into EMPTY), and is
displaced one cell by Io's contact — blocking like the wall it renders
as when the push target is blocked. Autonomous steps emit granular
``process="mover_step"`` events; displacements deliberately do NOT
(Io-caused, visible in AgentStep, no self class in WorldEvent.source —
the trail-stamping precedent), staying exposed mirror-side instead.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import CellType, GridWorld, GridWorldConfig
from kind.env.world_stages import (
    MOVER_START,
    MOVER_STEP_EVERY,
    MOVER_TURN_HAZARD,
    apply_world_stage,
)
from kind.observer.schemas import WorldEvent
from kind.observer.sinks import JsonlSink

UP, DOWN, LEFT, RIGHT, STAY = range(5)


def _mover_config(**overrides: object) -> GridWorldConfig:
    """Still world; mover on at (0,7) heading down; agent far away."""
    base = GridWorldConfig(
        initial_regrowth_p=0.0,
        drift_p_min=0.0,
        drift_p_max=0.0,
        drift_magnitude_per_step=0.0,
        n_initial_resources=0,
        start_cell=(7, 0),
        episode_resample=False,
        mover_enabled=True,
        mover_start=(0, 7),
        mover_step_every=2,
        mover_turn_hazard=0.0,  # pure inertial motion unless a test opts in
    )
    return dataclasses.replace(base, **overrides)  # type: ignore[arg-type]


def _walls(world: GridWorld) -> set[tuple[int, int]]:
    grid = world.state.grid
    return {
        (int(r), int(c))
        for r, c in np.argwhere(grid == CellType.WALL.value)
    }


# ---- default off: byte-identity ----------------------------------------------


def test_default_off_is_byte_identical() -> None:
    world_a = GridWorld(GridWorldConfig(), seed=42)
    world_b = GridWorld(GridWorldConfig(mover_enabled=False), seed=42)
    world_a.reset()
    world_b.reset()
    rng = np.random.default_rng(3)
    for action in (int(a) for a in rng.integers(0, 5, size=250)):
        step_a = world_a.step(action)
        step_b = world_b.step(action)
        assert np.array_equal(step_a.observation, step_b.observation)
    assert world_a.mover_pos is None


# ---- autonomous motion ---------------------------------------------------------


def test_inertial_motion_and_bounce() -> None:
    """Zero hazard: the mover runs its heading down the column, bounces
    at the wall edge, and runs back — deterministic."""
    world = GridWorld(_mover_config(), seed=42)
    world.reset()
    assert world.mover_pos == (0, 7)
    positions = []
    for _ in range(32):  # 16 moves at cadence 2
        world.step(STAY)
        positions.append(world.mover_pos)
    # Down the column to row 7, bounce, back up to row 1 (row 0 vacated).
    rows = [p[0] for p in positions[1::2]]  # after each move step
    assert rows[:7] == [1, 2, 3, 4, 5, 6, 7]
    assert rows[7:14] == [6, 5, 4, 3, 2, 1, 0]
    walls = _walls(world)
    assert walls == {world.mover_pos}, "the mover left wall debris behind"


def test_moves_on_cadence_only() -> None:
    world = GridWorld(_mover_config(mover_step_every=3), seed=42)
    world.reset()
    moved_at = []
    for step_index in range(1, 10):
        world.step(STAY)
        if world.last_mover_step is not None:
            moved_at.append(step_index)
    assert moved_at == [3, 6, 9]


def test_same_seed_same_trajectory_with_hazard() -> None:
    trajectories = []
    for _ in range(2):
        world = GridWorld(_mover_config(mover_turn_hazard=0.5), seed=99)
        world.reset()
        trajectory = []
        for _ in range(60):
            world.step(STAY)
            trajectory.append(world.mover_pos)
        trajectories.append(trajectory)
    assert trajectories[0] == trajectories[1]


def test_mover_never_tramples_or_steps_on_io() -> None:
    """400 hazard-on steps among walls, resources, trail, and a parked
    Io: every cell the mover leaves is EMPTY, it never overlaps Io, and
    walls/resources/trail survive it."""
    config = _mover_config(
        mover_turn_hazard=0.3,
        walls=((3, 3), (3, 4), (4, 3)),
        initial_regrowth_p=0.1,
        drift_p_min=0.05,
        drift_p_max=0.2,
        n_initial_resources=6,
        trail_enabled=True,
        trail_decay_steps=8,
        start_cell=(4, 4),
    )
    world = GridWorld(config, seed=7)
    world.reset()
    rng = np.random.default_rng(11)
    for action in (int(a) for a in rng.integers(0, 5, size=400)):
        world.step(action)
        state = world.state
        mover = world.mover_pos
        assert mover is not None
        assert mover != state.agent_pos
        assert int(state.grid[mover]) == CellType.WALL.value
        for wall in ((3, 3), (3, 4), (4, 3)):
            assert int(state.grid[wall]) == CellType.WALL.value


# ---- contact displacement -------------------------------------------------------


def test_push_displaces_mover_and_io_advances() -> None:
    """Io at (0,5) pushes right: mover (0,6)→(0,7), Io →(0,6)."""
    world = GridWorld(
        _mover_config(
            mover_start=(0, 6), start_cell=(0, 5), mover_step_every=1000
        ),
        seed=42,
    )
    world.reset()
    world.step(RIGHT)
    assert world.state.agent_pos == (0, 6)
    assert world.mover_pos == (0, 7)
    assert world.last_mover_displacement == ((0, 6), (0, 7))
    assert world.last_mover_step is None


def test_blocked_push_blocks_io_like_a_wall() -> None:
    """Mover against the edge: Io's push fails, both stay put."""
    world = GridWorld(
        _mover_config(
            mover_start=(0, 7), start_cell=(0, 6), mover_step_every=1000
        ),
        seed=42,
    )
    world.reset()
    world.step(RIGHT)
    assert world.state.agent_pos == (0, 6)
    assert world.mover_pos == (0, 7)
    assert world.last_mover_displacement is None


def test_push_into_resource_is_blocked() -> None:
    """The mover cannot be shoved onto food."""
    world = GridWorld(
        _mover_config(
            mover_start=(3, 5),
            start_cell=(3, 4),
            mover_step_every=1000,
            initial_regrowth_p=1.0,
            drift_p_min=0.5,
            drift_p_max=1.0,
            n_initial_resources=0,
        ),
        seed=42,
    )
    world.reset()
    world.step(STAY)  # p=1 regrowth fills every EMPTY cell, incl. (3,6)
    assert int(world.state.grid[3, 6]) == CellType.RESOURCE.value
    world.step(RIGHT)
    assert world.state.agent_pos == (3, 4)
    assert world.mover_pos == (3, 5)


# ---- events ---------------------------------------------------------------------


def test_mover_step_events_and_no_displacement_events(
    tmp_path: Path,
) -> None:
    sink_path = tmp_path / "world_event.jsonl"
    sink = JsonlSink(sink_path, WorldEvent)
    server = EnvServer(
        EnvServerConfig(
            grid_world_config=_mover_config(
                mover_start=(0, 6), start_cell=(0, 5), mover_step_every=2
            ),
            seed=42,
            world_event_handler=sink.write,
            run_id="mover-test",
            emit_internal_stochasticity_events=True,
        )
    )
    try:
        server.start()
        server.step(RIGHT)  # displacement: (0,6)→(0,7); Io →(0,6)
        server.step(STAY)  # move step: mover heads down, (0,7)→(1,7)
    finally:
        server.close()
        sink.close()

    events = [
        json.loads(line) for line in sink_path.read_text().splitlines()
    ]
    steps = [
        e for e in events if e["payload"].get("process") == "mover_step"
    ]
    assert len(steps) == 1
    assert steps[0]["payload"]["cell_from"] == [0, 7]
    assert steps[0]["payload"]["cell"] == [1, 7]
    assert steps[0]["payload"]["pre_state"] == "empty"
    assert steps[0]["payload"]["post_state"] == "wall"
    assert not any(
        "displace" in str(e["payload"].get("process", "")) for e in events
    ), "a displacement leaked into the ENVIRONMENT event stream"


# ---- placement, validation, preset ------------------------------------------------


def test_random_agent_start_avoids_mover() -> None:
    for seed in range(20):
        world = GridWorld(_mover_config(start_cell=None), seed=seed)
        world.reset()
        assert world.state.agent_pos != (0, 7)


def test_initial_resources_avoid_mover() -> None:
    world = GridWorld(
        _mover_config(n_initial_resources=61),  # every available cell
        seed=42,
    )
    world.reset()
    assert int(world.state.grid[0, 7]) == CellType.WALL.value


def test_mover_validation() -> None:
    with pytest.raises(ValueError, match="out of bounds"):
        GridWorld(_mover_config(mover_start=(8, 0)), seed=1)
    with pytest.raises(ValueError, match="collides with a wall"):
        GridWorld(
            _mover_config(mover_start=(2, 2), walls=((2, 2),)), seed=1
        )
    with pytest.raises(ValueError, match="agent's start_cell"):
        GridWorld(
            _mover_config(mover_start=(7, 0), start_cell=(7, 0)), seed=1
        )
    with pytest.raises(ValueError, match="mover_turn_hazard"):
        GridWorld(_mover_config(mover_turn_hazard=1.5), seed=1)


def test_stage_e4_is_cumulative() -> None:
    staged = apply_world_stage(GridWorldConfig(), "e4")
    assert staged.episode_resample is False
    assert staged.trail_enabled is True
    assert staged.bloom_cell is not None
    assert staged.regrowth_mode == "patch"
    assert staged.mover_enabled is True
    assert staged.mover_start == MOVER_START
    assert staged.mover_step_every == MOVER_STEP_EVERY
    assert staged.mover_turn_hazard == MOVER_TURN_HAZARD
