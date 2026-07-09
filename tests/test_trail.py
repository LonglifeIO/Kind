"""World v2 W2 — E1: the somatic trail (plan W2; synthesis E1, C1-cautious).

Pins the trail both ways. Default off is byte-identical (no TRAIL value
ever appears; trajectories match a pre-trail world exactly). Enabled:
vacated cells are stamped and decay back on a deterministic clock; food
and walls are never overwritten; trail blocks regrowth while present;
re-vacating refreshes the clock; decay emits granular
``process="trail_decay"`` events through the validated matched-control
payload shape; builder mutators may pave over or remove trail but never
fabricate or move it. TRAIL renders at its own gray level and the OOB
sentinel's render contract (value 3 → 64) is untouched — which is why
``CellType.TRAIL`` is 4.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import (
    CellType,
    GridWorld,
    GridWorldConfig,
    _OOB_SENTINEL,
    _RENDER_TABLE,
)
from kind.env.world_stages import (
    E0_WALLS,
    TRAIL_DECAY_STEPS,
    apply_world_stage,
)
from kind.observer.schemas import WorldEvent
from kind.observer.sinks import JsonlSink

# Actions (grid_world contract): 0=up, 1=down, 2=left, 3=right, 4=stay.
UP, DOWN, LEFT, RIGHT, STAY = range(5)


def _quiet_trail_config(**overrides: object) -> GridWorldConfig:
    """Still world (no regrowth, no drift), trail on, fixed start."""
    base = GridWorldConfig(
        initial_regrowth_p=0.0,
        drift_p_min=0.0,
        drift_p_max=0.0,
        drift_magnitude_per_step=0.0,
        n_initial_resources=0,
        start_cell=(3, 3),
        trail_enabled=True,
        trail_decay_steps=10,
        episode_resample=False,
    )
    return dataclasses.replace(base, **overrides)  # type: ignore[arg-type]


def _cell(world: GridWorld, r: int, c: int) -> int:
    return int(world.state.grid[r, c])


# ---- default off: byte-identity --------------------------------------------


def test_default_off_is_byte_identical() -> None:
    """A default world and an explicit trail_enabled=False world produce
    identical trajectories, and no TRAIL value ever appears."""
    world_a = GridWorld(GridWorldConfig(), seed=42)
    world_b = GridWorld(GridWorldConfig(trail_enabled=False), seed=42)
    world_a.reset()
    world_b.reset()
    rng = np.random.default_rng(3)
    for action in (int(a) for a in rng.integers(0, 5, size=300)):
        step_a = world_a.step(action)
        step_b = world_b.step(action)
        assert np.array_equal(step_a.observation, step_b.observation)
        grid = world_a.state.grid
        assert not bool((grid == CellType.TRAIL.value).any())


# ---- stamping ----------------------------------------------------------------


def test_stamp_on_vacate() -> None:
    world = GridWorld(_quiet_trail_config(), seed=42)
    world.reset()
    world.step(RIGHT)  # (3,3) -> (3,4); vacated (3,3)
    assert _cell(world, 3, 3) == CellType.TRAIL.value
    assert world.state.agent_pos == (3, 4)


def test_no_stamp_on_stay() -> None:
    world = GridWorld(_quiet_trail_config(), seed=42)
    world.reset()
    world.step(STAY)
    assert not bool((world.state.grid == CellType.TRAIL.value).any())


def test_no_stamp_on_blocked_move() -> None:
    """A wall collision or off-grid move vacates nothing."""
    config = _quiet_trail_config(start_cell=(0, 0), walls=((0, 1),))
    world = GridWorld(config, seed=42)
    world.reset()
    world.step(UP)  # off-grid
    world.step(RIGHT)  # wall at (0,1)
    assert not bool((world.state.grid == CellType.TRAIL.value).any())
    assert world.state.agent_pos == (0, 0)


def test_vacated_resource_is_not_stamped() -> None:
    """A resource that regrew under the stationary agent survives the
    agent's departure — food is never overwritten by a footprint."""
    config = _quiet_trail_config(
        initial_regrowth_p=0.3,
        drift_p_min=0.1,
        drift_p_max=0.5,
        drift_magnitude_per_step=0.0,
    )
    world = GridWorld(config, seed=42)
    world.reset()
    for _ in range(200):
        world.step(STAY)
        if _cell(world, 3, 3) == CellType.RESOURCE.value:
            break
    else:
        pytest.fail("regrowth never landed under the agent at p=0.3")
    world.step(RIGHT)
    assert _cell(world, 3, 3) == CellType.RESOURCE.value


# ---- decay -------------------------------------------------------------------


def test_decay_schedule_is_exact() -> None:
    """A footprint lasts exactly trail_decay_steps steps after stamping."""
    world = GridWorld(_quiet_trail_config(trail_decay_steps=10), seed=42)
    world.reset()
    world.step(RIGHT)  # stamp (3,3) at the end of this step
    for _ in range(9):
        world.step(STAY)
        assert _cell(world, 3, 3) == CellType.TRAIL.value
    world.step(STAY)  # 10th tick after the stamp
    assert _cell(world, 3, 3) == CellType.EMPTY.value


def test_revacate_refreshes_the_clock() -> None:
    world = GridWorld(_quiet_trail_config(trail_decay_steps=10), seed=42)
    world.reset()
    world.step(RIGHT)  # stamp (3,3)
    for _ in range(5):
        world.step(STAY)
    world.step(LEFT)  # re-enter (3,3); stamps (3,4)
    world.step(RIGHT)  # re-vacate (3,3): fresh clock
    for _ in range(9):
        world.step(STAY)
        assert _cell(world, 3, 3) == CellType.TRAIL.value
    world.step(STAY)
    assert _cell(world, 3, 3) == CellType.EMPTY.value


def test_trail_blocks_regrowth_and_decayed_cell_regrows() -> None:
    """No TRAIL cell ever converts directly to RESOURCE; once decayed to
    EMPTY the cell is regrowth-eligible again."""
    config = _quiet_trail_config(
        initial_regrowth_p=0.3,
        drift_p_min=0.1,
        drift_p_max=0.5,
        drift_magnitude_per_step=0.0,
        trail_decay_steps=6,
    )
    world = GridWorld(config, seed=42)
    world.reset()
    prev = world.state.grid
    saw_decay_then_regrow = False
    rng = np.random.default_rng(5)
    was_trail = np.zeros_like(prev, dtype=bool)
    for action in (int(a) for a in rng.integers(0, 4, size=400)):
        world.step(action)
        grid = world.state.grid
        trail_prev = prev == CellType.TRAIL.value
        resource_now = grid == CellType.RESOURCE.value
        assert not bool((trail_prev & resource_now).any()), (
            "a trail cell regrew without decaying first"
        )
        if bool((was_trail & resource_now).any()):
            saw_decay_then_regrow = True
        was_trail |= trail_prev
        prev = grid
    assert saw_decay_then_regrow, (
        "no decayed footprint ever regrew — exclusion test saw nothing"
    )


def test_trail_is_passable_and_inedible() -> None:
    world = GridWorld(_quiet_trail_config(), seed=42)
    world.reset()
    energy_before = world.state.true_energy
    world.step(RIGHT)  # stamp (3,3)
    world.step(LEFT)  # walk back onto own trail
    assert world.state.agent_pos == (3, 3)
    # Entering trail replenishes nothing (energy only ever decayed).
    assert world.state.true_energy < energy_before


# ---- rendering ---------------------------------------------------------------


def test_render_contract() -> None:
    """TRAIL has its own gray level; the OOB sentinel contract holds."""
    assert CellType.TRAIL.value == 4
    assert _OOB_SENTINEL == 3
    assert int(_RENDER_TABLE[_OOB_SENTINEL]) == 64
    assert int(_RENDER_TABLE[CellType.TRAIL.value]) == 192
    assert len(set(int(v) for v in _RENDER_TABLE)) == 5


def test_trail_is_visible_in_observation() -> None:
    world = GridWorld(_quiet_trail_config(), seed=42)
    world.reset()
    step = world.step(RIGHT)  # trail at (3,3), one cell left of Io
    assert bool((step.observation == 192).any())


# ---- granular events ---------------------------------------------------------


def test_trail_decay_emits_validated_granular_events(tmp_path: Path) -> None:
    """Each decay emits one ``process="trail_decay"`` event through the
    writer-side validator (JsonlSink write is the validation path);
    stamping emits nothing (self-caused, visible in AgentStep)."""
    sink_path = tmp_path / "world_event.jsonl"
    sink = JsonlSink(sink_path, WorldEvent)
    server = EnvServer(
        EnvServerConfig(
            grid_world_config=_quiet_trail_config(trail_decay_steps=5),
            seed=42,
            world_event_handler=sink.write,
            run_id="trail-test",
            emit_internal_stochasticity_events=True,
        )
    )
    try:
        server.start()
        server.step(RIGHT)  # stamp (3,3)
        server.step(RIGHT)  # stamp (3,4)
        for _ in range(6):
            server.step(STAY)
    finally:
        server.close()
        sink.close()

    events = [
        json.loads(line) for line in sink_path.read_text().splitlines()
    ]
    decays = [
        e
        for e in events
        if e["payload"].get("process") == "trail_decay"
    ]
    assert len(decays) == 2
    cells = sorted(tuple(e["payload"]["cell"]) for e in decays)
    assert cells == [(3, 3), (3, 4)]
    for event in decays:
        assert event["event_type"] == "internal_stochasticity_event"
        assert event["source"] == "environment"
        assert event["payload"]["pre_state"] == "trail"
        assert event["payload"]["post_state"] == "empty"


# ---- builder mutators vs trail ------------------------------------------------


def test_builder_can_pave_trail_and_decay_never_stomps_it() -> None:
    """Paving a trail cell with a wall works, names the pre-state
    honestly, and the stale decay clock never overwrites the wall."""
    server = EnvServer(
        EnvServerConfig(
            grid_world_config=_quiet_trail_config(trail_decay_steps=5),
            seed=42,
            world_event_handler=lambda _r: None,
            run_id="trail-pave-test",
        )
    )
    try:
        server.start()
        server.step(RIGHT)  # trail at (3,3)
        server.set_cell_state((3, 3), CellType.WALL)
        for _ in range(8):  # run well past the stale clock
            server.step(STAY)
        assert (
            int(server.grid_world_state.grid[3, 3]) == CellType.WALL.value
        )
    finally:
        server.close()


def test_builder_can_remove_trail() -> None:
    server = EnvServer(
        EnvServerConfig(
            grid_world_config=_quiet_trail_config(),
            seed=42,
            world_event_handler=lambda _r: None,
            run_id="trail-remove-test",
        )
    )
    try:
        server.start()
        server.step(RIGHT)
        server.remove_object((3, 3), CellType.TRAIL)
        assert (
            int(server.grid_world_state.grid[3, 3]) == CellType.EMPTY.value
        )
    finally:
        server.close()


def test_builder_cannot_fabricate_or_move_trail() -> None:
    server = EnvServer(
        EnvServerConfig(
            grid_world_config=_quiet_trail_config(),
            seed=42,
            world_event_handler=lambda _r: None,
            run_id="trail-fabricate-test",
        )
    )
    try:
        server.start()
        with pytest.raises(ValueError, match="footprints"):
            server.set_cell_state((0, 0), CellType.TRAIL)
        server.step(RIGHT)  # real trail at (3,3)
        with pytest.raises(ValueError, match="not a movable object"):
            server.move_object((3, 3), (0, 0))
    finally:
        server.close()


# ---- the e1 stage preset -------------------------------------------------------


def test_stage_e1_is_cumulative() -> None:
    staged = apply_world_stage(GridWorldConfig(), "e1")
    assert staged.episode_resample is False
    assert staged.walls == E0_WALLS
    assert staged.trail_enabled is True
    assert staged.trail_decay_steps == TRAIL_DECAY_STEPS


def test_window_live_template_styles_trail() -> None:
    template = (
        Path("kind/window/templates/live.html").read_text(encoding="utf-8")
    )
    assert "v === 4" in template, "live grid has no style for TRAIL"
