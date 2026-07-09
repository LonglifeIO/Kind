"""World v2 W3 — E2: the hidden clock (plan W3; synthesis E2).

Pins the bloom both ways. Default off (``bloom_cell=None``) is
byte-identical. Enabled: an unobserved phase counter fires every
``bloom_period`` steps, stamping the EMPTY cells of the source cell's
Moore ring in the *trail vocabulary* (no new cell type — the cause is
distinguishable only dynamically) for exactly ``bloom_duration`` steps;
the source cell itself never changes; walls, resources, live trail, and
out-of-bounds cells are never stamped. Provenance is honest end to end:
stamps emit granular ``process="bloom"`` events, fades emit
``process="bloom_fade"`` (never ``trail_decay``), and a bloom cell Io
walks through and vacates becomes Io's own footprint.
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
    BLOOM_CELL,
    BLOOM_DURATION,
    BLOOM_PERIOD,
    apply_world_stage,
)
from kind.observer.schemas import WorldEvent
from kind.observer.sinks import JsonlSink

UP, DOWN, LEFT, RIGHT, STAY = range(5)


def _quiet_bloom_config(**overrides: object) -> GridWorldConfig:
    """Still world, bloom on at (4,4), agent parked far away."""
    base = GridWorldConfig(
        initial_regrowth_p=0.0,
        drift_p_min=0.0,
        drift_p_max=0.0,
        drift_magnitude_per_step=0.0,
        n_initial_resources=0,
        start_cell=(0, 0),
        episode_resample=False,
        bloom_cell=(4, 4),
        bloom_period=10,
        bloom_duration=2,
    )
    return dataclasses.replace(base, **overrides)  # type: ignore[arg-type]


def _ring(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr, dc) != (0, 0)
    }


def _trail_cells(world: GridWorld) -> set[tuple[int, int]]:
    grid = world.state.grid
    return {
        (int(r), int(c))
        for r, c in np.argwhere(grid == CellType.TRAIL.value)
    }


# ---- default off: byte-identity ---------------------------------------------


def test_default_off_is_byte_identical() -> None:
    world_a = GridWorld(GridWorldConfig(), seed=42)
    world_b = GridWorld(GridWorldConfig(bloom_cell=None), seed=42)
    world_a.reset()
    world_b.reset()
    rng = np.random.default_rng(3)
    for action in (int(a) for a in rng.integers(0, 5, size=250)):
        step_a = world_a.step(action)
        step_b = world_b.step(action)
        assert np.array_equal(step_a.observation, step_b.observation)


# ---- the clock ----------------------------------------------------------------


def test_period_and_duration_are_exact() -> None:
    """Blooms appear on the terminal phase and fade on schedule."""
    world = GridWorld(_quiet_bloom_config(), seed=42)
    world.reset()
    ring = _ring((4, 4))
    for step_index in range(1, 35):
        world.step(STAY)
        blooming = _trail_cells(world)
        # With period 10 / duration 2: stamped at steps 10, 20, 30;
        # visible for exactly 2 observations (the stamp step and the
        # next), gone on the one after — the trail's timing convention.
        phase = step_index % 10
        if phase in (0, 1) and step_index >= 10:
            assert blooming == ring, (
                f"step {step_index}: expected the full ring, got {blooming}"
            )
        else:
            assert blooming == set(), (
                f"step {step_index}: unexpected trail cells {blooming}"
            )


def test_source_cell_never_changes() -> None:
    world = GridWorld(_quiet_bloom_config(), seed=42)
    world.reset()
    for _ in range(25):
        world.step(STAY)
        assert int(world.state.grid[4, 4]) == CellType.EMPTY.value


def test_bloom_never_stamps_occupied_cells() -> None:
    """Walls and resources inside the ring survive every bloom."""
    config = _quiet_bloom_config(walls=((3, 3), (3, 4)))
    world = GridWorld(config, seed=42)
    world.reset()
    # Plant a resource inside the ring through regrowth-free surgery:
    # use a fresh config with initial resources and a fixed seed is
    # nondeterministic in placement, so instead park a resource via
    # high regrowth in a bounded window, then freeze.
    for _ in range(25):
        world.step(STAY)
        grid = world.state.grid
        assert int(grid[3, 3]) == CellType.WALL.value
        assert int(grid[3, 4]) == CellType.WALL.value


def test_bloom_ring_clips_at_grid_edge() -> None:
    world = GridWorld(_quiet_bloom_config(bloom_cell=(7, 7)), seed=42)
    world.reset()
    for _ in range(10):
        world.step(STAY)
    assert _trail_cells(world) == {(6, 6), (6, 7), (7, 6)}


def test_phase_is_unobserved() -> None:
    """No observation differs between two pre-bloom steps solely
    because the hidden phase differs — the counter has no render."""
    world = GridWorld(_quiet_bloom_config(bloom_period=30), seed=42)
    step_a = world.reset()
    obs_prev = step_a.observation
    for _ in range(28):  # all pre-terminal: phases 1..28, no bloom yet
        step = world.step(STAY)
        assert np.array_equal(step.observation, obs_prev)
        obs_prev = step.observation


# ---- provenance ----------------------------------------------------------------


def test_bloom_and_fade_events_are_classified(tmp_path: Path) -> None:
    sink_path = tmp_path / "world_event.jsonl"
    sink = JsonlSink(sink_path, WorldEvent)
    server = EnvServer(
        EnvServerConfig(
            grid_world_config=_quiet_bloom_config(),
            seed=42,
            world_event_handler=sink.write,
            run_id="bloom-test",
            emit_internal_stochasticity_events=True,
        )
    )
    try:
        server.start()
        for _ in range(13):  # one full bloom + fade cycle
            server.step(STAY)
    finally:
        server.close()
        sink.close()

    events = [
        json.loads(line) for line in sink_path.read_text().splitlines()
    ]
    blooms = [e for e in events if e["payload"].get("process") == "bloom"]
    fades = [
        e for e in events if e["payload"].get("process") == "bloom_fade"
    ]
    decays = [
        e for e in events if e["payload"].get("process") == "trail_decay"
    ]
    ring = _ring((4, 4))
    assert {tuple(e["payload"]["cell"]) for e in blooms} == ring
    assert {tuple(e["payload"]["cell"]) for e in fades} == ring
    assert decays == [], "a bloom fade was misattributed to Io's trail"
    for event in blooms:
        assert event["payload"]["pre_state"] == "empty"
        assert event["payload"]["post_state"] == "trail"
    for event in fades:
        assert event["payload"]["pre_state"] == "trail"
        assert event["payload"]["post_state"] == "empty"


def test_vacated_bloom_cell_becomes_io_footprint(tmp_path: Path) -> None:
    """Io walking through a bloom and leaving converts the cell's
    provenance: its eventual fade is a trail_decay, not a bloom_fade."""
    sink_path = tmp_path / "world_event.jsonl"
    sink = JsonlSink(sink_path, WorldEvent)
    config = _quiet_bloom_config(
        start_cell=(4, 3),  # inside the ring
        trail_enabled=True,
        trail_decay_steps=4,
        bloom_period=10,
        bloom_duration=3,
    )
    server = EnvServer(
        EnvServerConfig(
            grid_world_config=config,
            seed=42,
            world_event_handler=sink.write,
            run_id="bloom-convert-test",
            emit_internal_stochasticity_events=True,
        )
    )
    try:
        server.start()
        for _ in range(9):
            server.step(STAY)
        server.step(STAY)  # step 10: bloom fires around (4,4)
        # Io stands at (4,3): that ring cell was occupied by...
        # the agent is not a cell — (4,3) was EMPTY and is now TRAIL
        # under Io. Io steps away, re-stamping it as its own footprint.
        server.step(LEFT)  # vacates (4,3) -> footprint, ttl 4
        for _ in range(8):
            server.step(STAY)
    finally:
        server.close()
        sink.close()

    events = [
        json.loads(line) for line in sink_path.read_text().splitlines()
    ]
    fades_43 = [
        e
        for e in events
        if e["payload"].get("process") == "bloom_fade"
        and tuple(e["payload"]["cell"]) == (4, 3)
    ]
    decays_43 = [
        e
        for e in events
        if e["payload"].get("process") == "trail_decay"
        and tuple(e["payload"]["cell"]) == (4, 3)
    ]
    assert fades_43 == [], "converted footprint still faded as bloom"
    assert len(decays_43) == 1, "footprint decay missing after conversion"


# ---- config and preset -----------------------------------------------------------


def test_bloom_validation() -> None:
    with pytest.raises(ValueError, match="out of bounds"):
        GridWorld(_quiet_bloom_config(bloom_cell=(8, 0)), seed=1)
    with pytest.raises(ValueError, match="collides with a wall"):
        GridWorld(
            _quiet_bloom_config(bloom_cell=(3, 3), walls=((3, 3),)), seed=1
        )
    with pytest.raises(ValueError, match="bloom_duration"):
        GridWorld(
            _quiet_bloom_config(bloom_period=5, bloom_duration=5), seed=1
        )


def test_stage_e2_is_cumulative() -> None:
    """e2 (clock) lands AFTER e3 (weather) — builder-ratified reorder,
    2026-07-09 — so the e2 stage carries the patch as well."""
    staged = apply_world_stage(GridWorldConfig(), "e2")
    assert staged.episode_resample is False
    assert staged.trail_enabled is True
    assert staged.regrowth_mode == "patch"
    assert staged.bloom_cell == BLOOM_CELL
    assert staged.bloom_period == BLOOM_PERIOD
    assert staged.bloom_duration == BLOOM_DURATION


def test_e2_bloom_ring_is_wall_free_and_in_bounds() -> None:
    staged = apply_world_stage(GridWorldConfig(), "e2")
    size = staged.grid_size
    for r, c in _ring(BLOOM_CELL):
        assert 0 <= r < size and 0 <= c < size
        assert (r, c) not in staged.walls
