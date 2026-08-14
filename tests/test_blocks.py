"""World v3 E5: pushable blocks (synthesis DP1/DP2, ratified 2026-08-14).

Pins the blocks both ways. Default off (empty ``block_cells``) is
byte-identical — blocks add no RNG stream, so this is exact, not
stream-discipline-dependent. Enabled: a block is displaced one cell by
Io's contact under the mover's exact rule (into EMPTY only; otherwise
it blocks like the wall it renders as) and then STAYS — no self-motion,
no decay. The mover cannot push blocks (its moves require EMPTY); Io is
the only author. Block pushes emit NO environment events (Io-caused —
the mover-displacement precedent); ground truth is the
``block_positions`` / ``last_block_displacement`` surfaces, GridState,
and the ``block_pos.jsonl`` run-script sidecar. Also pins the
reachable-set §7 monitor (``kind.env.reachability``) the e5 arrival is
gated on.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import CellType, GridWorld, GridWorldConfig
from kind.env.reachability import (
    reachable_set_size,
    reachable_set_size_from_layout,
)
from kind.env.world_stages import E5_BLOCK_CELLS, apply_world_stage
from kind.observer.schemas import WorldEvent
from kind.observer.sinks import JsonlSink
from kind.window.live import LiveStateWriter

UP, DOWN, LEFT, RIGHT, STAY = range(5)


def _block_config(**overrides: object) -> GridWorldConfig:
    """Still world; one block at (3,5); agent right beside it."""
    base = GridWorldConfig(
        initial_regrowth_p=0.0,
        drift_p_min=0.0,
        drift_p_max=0.0,
        drift_magnitude_per_step=0.0,
        n_initial_resources=0,
        start_cell=(3, 4),
        episode_resample=False,
        block_cells=((3, 5),),
    )
    return dataclasses.replace(base, **overrides)  # type: ignore[arg-type]


# ---- default off: byte-identity -------------------------------------------


def test_default_off_is_byte_identical() -> None:
    world_a = GridWorld(GridWorldConfig(), seed=42)
    world_b = GridWorld(GridWorldConfig(block_cells=()), seed=42)
    world_a.reset()
    world_b.reset()
    rng = np.random.default_rng(3)
    for action in (int(a) for a in rng.integers(0, 5, size=250)):
        step_a = world_a.step(action)
        step_b = world_b.step(action)
        assert np.array_equal(step_a.observation, step_b.observation)
    assert world_a.block_positions == ()
    assert world_a.state.block_positions == ()


# ---- push mechanics --------------------------------------------------------


def test_push_displaces_block_and_io_advances() -> None:
    """Io at (3,4) pushes right: block (3,5)→(3,6), Io →(3,5)."""
    world = GridWorld(_block_config(), seed=42)
    world.reset()
    assert world.block_positions == ((3, 5),)
    assert int(world.state.grid[3, 5]) == CellType.WALL.value
    world.step(RIGHT)
    assert world.state.agent_pos == (3, 5)
    assert world.block_positions == ((3, 6),)
    assert world.state.block_positions == ((3, 6),)
    assert world.last_block_displacement == ((3, 5), (3, 6))
    assert int(world.state.grid[3, 6]) == CellType.WALL.value
    assert int(world.state.grid[3, 5]) != CellType.WALL.value


def test_pushed_block_stays() -> None:
    """No self-motion, no decay: the authored layout persists."""
    world = GridWorld(_block_config(), seed=42)
    world.reset()
    world.step(RIGHT)
    assert world.block_positions == ((3, 6),)
    for _ in range(100):
        world.step(STAY)
    assert world.block_positions == ((3, 6),)
    assert int(world.state.grid[3, 6]) == CellType.WALL.value
    assert world.last_block_displacement is None  # cleared per step


def test_push_at_edge_blocks_io_like_a_wall() -> None:
    world = GridWorld(
        _block_config(block_cells=((3, 7),), start_cell=(3, 6)), seed=42
    )
    world.reset()
    world.step(RIGHT)
    assert world.state.agent_pos == (3, 6)
    assert world.block_positions == ((3, 7),)
    assert world.last_block_displacement is None


def test_push_into_wall_is_blocked() -> None:
    world = GridWorld(_block_config(walls=((3, 6),)), seed=42)
    world.reset()
    world.step(RIGHT)
    assert world.state.agent_pos == (3, 4)
    assert world.block_positions == ((3, 5),)


def test_push_into_resource_is_blocked() -> None:
    """A block cannot be shoved onto food (the mover's rule)."""
    world = GridWorld(
        _block_config(
            initial_regrowth_p=1.0,
            drift_p_min=0.5,
            drift_p_max=1.0,
        ),
        seed=42,
    )
    world.reset()
    world.step(STAY)  # p=1 regrowth fills every EMPTY cell, incl. (3,6)
    assert int(world.state.grid[3, 6]) == CellType.RESOURCE.value
    world.step(RIGHT)
    assert world.state.agent_pos == (3, 4)
    assert world.block_positions == ((3, 5),)


def test_push_into_second_block_is_blocked() -> None:
    """Two blocks in a row do not compress or chain-push."""
    world = GridWorld(
        _block_config(block_cells=((3, 5), (3, 6))), seed=42
    )
    world.reset()
    world.step(RIGHT)
    assert world.state.agent_pos == (3, 4)
    assert world.block_positions == ((3, 5), (3, 6))


def test_push_into_mover_is_blocked_and_vice_versa() -> None:
    """Block↔mover: neither can displace the other; Io alone authors."""
    world = GridWorld(
        _block_config(
            block_cells=((3, 5),),
            mover_enabled=True,
            mover_start=(3, 6),
            mover_step_every=1000,
        ),
        seed=42,
    )
    world.reset()
    world.step(RIGHT)  # push target (3,6) is the mover: blocked
    assert world.state.agent_pos == (3, 4)
    assert world.block_positions == ((3, 5),)
    assert world.mover_pos == (3, 6)


def test_mover_cannot_push_block() -> None:
    """The mover bounces off a block exactly as off a wall."""
    world = GridWorld(
        _block_config(
            block_cells=((1, 7),),
            start_cell=(7, 0),
            mover_enabled=True,
            mover_start=(0, 7),  # heading (1,0): straight into the block
            mover_step_every=1,
        ),
        seed=42,
    )
    world.reset()
    for _ in range(6):
        world.step(STAY)
    # Forward blocked by the block, reverse blocked by the edge: parked.
    assert world.mover_pos == (0, 7)
    assert world.block_positions == ((1, 7),)


def test_same_seed_same_block_trajectory() -> None:
    trajectories = []
    for _ in range(2):
        world = GridWorld(
            _block_config(
                initial_regrowth_p=0.05,
                drift_p_min=0.01,
                drift_p_max=0.1,
                n_initial_resources=4,
            ),
            seed=99,
        )
        world.reset()
        rng = np.random.default_rng(5)
        trajectory = []
        for action in (int(a) for a in rng.integers(0, 5, size=200)):
            world.step(action)
            trajectory.append(world.block_positions)
        trajectories.append(trajectory)
    assert trajectories[0] == trajectories[1]


# ---- events: blocks are silent in the ENVIRONMENT stream -------------------


def test_no_block_events_in_environment_stream(tmp_path: Path) -> None:
    sink_path = tmp_path / "world_event.jsonl"
    sink = JsonlSink(sink_path, WorldEvent)
    server = EnvServer(
        EnvServerConfig(
            grid_world_config=_block_config(),
            seed=42,
            world_event_handler=sink.write,
            run_id="block-test",
            emit_internal_stochasticity_events=True,
        )
    )
    try:
        server.start()
        server.step(RIGHT)  # push: (3,5)→(3,6)
        server.step(STAY)
    finally:
        server.close()
        sink.close()

    events = [
        json.loads(line) for line in sink_path.read_text().splitlines()
    ]
    assert not any(
        "block" in str(e["payload"].get("process", "")) for e in events
    ), "a block push leaked into the ENVIRONMENT event stream"


# ---- placement, validation, preset -----------------------------------------


def test_random_agent_start_avoids_blocks() -> None:
    for seed in range(20):
        world = GridWorld(_block_config(start_cell=None), seed=seed)
        world.reset()
        assert world.state.agent_pos != (3, 5)


def test_initial_resources_avoid_blocks() -> None:
    world = GridWorld(
        _block_config(n_initial_resources=63),  # every available cell
        seed=42,
    )
    world.reset()
    assert int(world.state.grid[3, 5]) == CellType.WALL.value
    assert world.block_positions == ((3, 5),)


def test_block_validation() -> None:
    with pytest.raises(ValueError, match="out of bounds"):
        GridWorld(_block_config(block_cells=((8, 0),)), seed=1)
    with pytest.raises(ValueError, match="collides with a wall"):
        GridWorld(
            _block_config(block_cells=((2, 2),), walls=((2, 2),)), seed=1
        )
    with pytest.raises(ValueError, match="duplicate block cell"):
        GridWorld(_block_config(block_cells=((3, 5), (3, 5))), seed=1)
    with pytest.raises(ValueError, match="collides with mover_start"):
        GridWorld(
            _block_config(
                block_cells=((0, 7),),
                mover_enabled=True,
                mover_start=(0, 7),
            ),
            seed=1,
        )
    with pytest.raises(ValueError, match="agent's start_cell"):
        GridWorld(
            _block_config(block_cells=((3, 4),), start_cell=(3, 4)), seed=1
        )
    with pytest.raises(ValueError, match="bloom cell"):
        GridWorld(
            _block_config(block_cells=((4, 4),), bloom_cell=(4, 4)), seed=1
        )


def test_stage_e5_is_cumulative() -> None:
    staged = apply_world_stage(GridWorldConfig(), "e5")
    assert staged.episode_resample is False
    assert staged.trail_enabled is True
    assert staged.bloom_cell is not None
    assert staged.regrowth_mode == "patch"
    assert staged.mover_enabled is True
    assert staged.block_cells == E5_BLOCK_CELLS
    # Spawn cells clear of every standing feature (S3 sketch): walls,
    # the bloom cell and its Moore ring, and the mover's spawn corner.
    bloom_r, bloom_c = staged.bloom_cell
    ring = {
        (bloom_r + dr, bloom_c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
    }
    for cell in staged.block_cells:
        assert cell not in staged.walls
        assert cell not in ring
        assert cell != staged.mover_start


def test_e5_world_boots_and_runs() -> None:
    world = GridWorld(
        apply_world_stage(GridWorldConfig(), "e5"), seed=42
    )
    world.reset()
    assert world.block_positions == E5_BLOCK_CELLS
    rng = np.random.default_rng(1)
    for action in (int(a) for a in rng.integers(0, 5, size=300)):
        world.step(action)
        for cell in world.block_positions:
            assert int(world.state.grid[cell]) == CellType.WALL.value


# ---- the reachable-set §7 monitor ------------------------------------------


def test_reachable_set_full_empty_grid() -> None:
    grid = np.zeros((8, 8), dtype=np.uint8)
    assert reachable_set_size(grid, (0, 0)) == 64


def test_reachable_set_counts_io_side_of_a_partition() -> None:
    grid = np.zeros((8, 8), dtype=np.uint8)
    grid[:, 3] = CellType.WALL.value  # full column: two regions
    assert reachable_set_size(grid, (0, 0)) == 24  # 8 x 3
    assert reachable_set_size(grid, (0, 5)) == 32  # 8 x 4


def test_reachable_set_treats_trail_and_resources_as_traversable() -> None:
    grid = np.zeros((4, 4), dtype=np.uint8)
    grid[1, 1] = CellType.TRAIL.value
    grid[2, 2] = CellType.RESOURCE.value
    assert reachable_set_size(grid, (0, 0)) == 16


def test_reachable_set_sealed_pocket() -> None:
    """The self-walling signature: a one-cell pocket reads as 1."""
    grid = np.zeros((8, 8), dtype=np.uint8)
    for cell in ((0, 1), (1, 0), (1, 1)):
        grid[cell] = CellType.WALL.value
    assert reachable_set_size(grid, (0, 0)) == 1


def test_reachable_set_rejects_impossible_states() -> None:
    grid = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="out of bounds"):
        reachable_set_size(grid, (8, 0))
    grid[2, 2] = CellType.WALL.value
    with pytest.raises(ValueError, match="WALL"):
        reachable_set_size(grid, (2, 2))


def test_reachable_set_from_layout_matches_grid_path() -> None:
    walls = ((2, 2), (3, 2), (4, 2), (5, 2), (5, 3), (5, 4))
    blocks = ((3, 6), (6, 1))
    grid = np.zeros((8, 8), dtype=np.uint8)
    for r, c in walls + blocks:
        grid[r, c] = CellType.WALL.value
    assert reachable_set_size_from_layout(
        8, walls + blocks, (7, 5)
    ) == reachable_set_size(grid, (7, 5))


# ---- the block_pos.jsonl sidecar -------------------------------------------


def _step_info(env_step: int) -> SimpleNamespace:
    return SimpleNamespace(
        env_step=env_step, episode_id=0, step_in_episode=env_step
    )


def test_block_sidecar_written_on_change_only(tmp_path: Path) -> None:
    server = EnvServer(
        EnvServerConfig(
            grid_world_config=_block_config(),
            seed=42,
            world_event_handler=lambda event: None,
            run_id="sidecar-test",
        )
    )
    writer = LiveStateWriter(
        server, tmp_path, run_id="sidecar-test", print_every=0
    )
    try:
        server.start()
        writer(_step_info(0))  # initial layout line
        server.step(STAY)
        writer(_step_info(1))  # unchanged: no line
        server.step(RIGHT)  # push: (3,5)→(3,6)
        writer(_step_info(2))  # changed: one line
    finally:
        server.close()

    lines = [
        json.loads(line)
        for line in (tmp_path / "block_pos.jsonl")
        .read_text()
        .splitlines()
    ]
    assert lines == [
        {"t": 0, "blocks": [[3, 5]]},
        {"t": 2, "blocks": [[3, 6]]},
    ]


def test_block_sidecar_absent_without_blocks(tmp_path: Path) -> None:
    server = EnvServer(
        EnvServerConfig(
            grid_world_config=_block_config(block_cells=()),
            seed=42,
            world_event_handler=lambda event: None,
            run_id="sidecar-test",
        )
    )
    writer = LiveStateWriter(
        server, tmp_path, run_id="sidecar-test", print_every=0
    )
    try:
        server.start()
        writer(_step_info(0))
    finally:
        server.close()
    assert not (tmp_path / "block_pos.jsonl").exists()
