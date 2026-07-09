"""World v2 W4 — E3: food becomes weather (plan W4; synthesis E3).

Pins patch-mode regrowth both ways. Default ``"uniform"`` is
byte-identical. In ``"patch"`` mode: the 3×3 patch drifts on a
deterministic bounce law (no RNG — pinned against a hand-computed
trajectory), regrowth stratifies inside/outside the patch, every patch
move emits one granular ``process="patch_drift"`` event through the
validated payload shape (with ``center_from``/``center_to`` extras),
``process="regrowth"`` events keep their unchanged shape, and the
boundary analyzer's occupancy-share diagnostic (C4 crowd-out watch)
computes the in-patch share from the position sidecar + event stream.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import CellType, GridWorld, GridWorldConfig
from kind.env.world_stages import (
    PATCH_EXPIRY_P,
    PATCH_P_INSIDE,
    PATCH_P_OUTSIDE,
    PATCH_STEP_EVERY,
    apply_world_stage,
)
from kind.observer.schemas import WorldEvent
from kind.observer.sinks import JsonlSink

UP, DOWN, LEFT, RIGHT, STAY = range(5)

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "analyze_boundary.py"
)


def _load_analyzer():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "analyze_boundary_for_test", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_boundary_for_test"] = module
    spec.loader.exec_module(module)
    return module


def _patch_config(**overrides: object) -> GridWorldConfig:
    base = GridWorldConfig(
        initial_regrowth_p=0.0,
        drift_p_min=0.0,
        drift_p_max=0.0,
        drift_magnitude_per_step=0.0,
        n_initial_resources=0,
        start_cell=(0, 0),
        episode_resample=False,
        regrowth_mode="patch",
        patch_step_every=20,
        patch_p_inside=0.06,
        patch_p_outside=0.001,
    )
    return dataclasses.replace(base, **overrides)  # type: ignore[arg-type]


# ---- default mode: byte-identity ---------------------------------------------


def test_default_uniform_is_byte_identical() -> None:
    world_a = GridWorld(GridWorldConfig(), seed=42)
    world_b = GridWorld(GridWorldConfig(regrowth_mode="uniform"), seed=42)
    world_a.reset()
    world_b.reset()
    rng = np.random.default_rng(3)
    for action in (int(a) for a in rng.integers(0, 5, size=250)):
        step_a = world_a.step(action)
        step_b = world_b.step(action)
        assert np.array_equal(step_a.observation, step_b.observation)


# ---- the drift law -------------------------------------------------------------


def test_bounce_trajectory_is_exact() -> None:
    """Hand-computed zigzag on the 8×8 grid: center range [1,6] for a
    3×3 patch; start (3,3) heading (1,1)."""
    world = GridWorld(_patch_config(patch_step_every=1), seed=42)
    world.reset()
    expected = [
        (4, 4),
        (5, 5),
        (6, 6),
        (5, 5),  # both components reflect at 6
        (4, 4),
        (3, 3),
        (2, 2),
        (1, 1),
        (2, 2),  # both components reflect at 1
        (3, 3),
    ]
    seen: list[tuple[int, int]] = []
    for _ in expected:
        world.step(STAY)
        move = world.last_patch_move
        assert move is not None, "patch failed to move on its cadence"
        seen.append(move[1])
    assert seen == expected


def test_patch_moves_on_cadence_only() -> None:
    world = GridWorld(_patch_config(patch_step_every=5), seed=42)
    world.reset()
    moves = []
    for step_index in range(1, 16):
        world.step(STAY)
        if world.last_patch_move is not None:
            moves.append(step_index)
    assert moves == [5, 10, 15]


def test_uniform_mode_never_moves_patch() -> None:
    world = GridWorld(
        _patch_config(regrowth_mode="uniform", initial_regrowth_p=0.0),
        seed=42,
    )
    world.reset()
    for _ in range(30):
        world.step(STAY)
        assert world.last_patch_move is None


# ---- rate stratification --------------------------------------------------------


def test_rate_stratification_inside_vs_outside() -> None:
    """With p_inside=1 and p_outside=0, exactly the patch square's
    EMPTY cells regrow, every step."""
    world = GridWorld(
        _patch_config(
            patch_p_inside=1.0, patch_p_outside=0.0, patch_step_every=1000
        ),
        seed=42,
    )
    world.reset()
    world.step(STAY)
    grid = world.state.grid
    resources = {
        (int(r), int(c))
        for r, c in np.argwhere(grid == CellType.RESOURCE.value)
    }
    patch_square = {(r, c) for r in (2, 3, 4) for c in (2, 3, 4)}
    assert resources == patch_square


def test_stratification_follows_the_patch() -> None:
    """After the patch moves, new regrowth lands under the new square."""
    world = GridWorld(
        _patch_config(
            patch_p_inside=1.0, patch_p_outside=0.0, patch_step_every=1
        ),
        seed=42,
    )
    world.reset()
    world.step(STAY)  # patch moves to (4,4) before regrowth
    grid = world.state.grid
    resources = {
        (int(r), int(c))
        for r, c in np.argwhere(grid == CellType.RESOURCE.value)
    }
    assert resources == {(r, c) for r in (3, 4, 5) for c in (3, 4, 5)}


# ---- granular events -------------------------------------------------------------


def test_patch_drift_and_regrowth_events(tmp_path: Path) -> None:
    sink_path = tmp_path / "world_event.jsonl"
    sink = JsonlSink(sink_path, WorldEvent)
    server = EnvServer(
        EnvServerConfig(
            grid_world_config=_patch_config(
                patch_step_every=3,
                patch_p_inside=1.0,
                patch_p_outside=0.0,
            ),
            seed=42,
            world_event_handler=sink.write,
            run_id="weather-test",
            emit_internal_stochasticity_events=True,
        )
    )
    try:
        server.start()
        for _ in range(7):
            server.step(STAY)
    finally:
        server.close()
        sink.close()

    events = [
        json.loads(line) for line in sink_path.read_text().splitlines()
    ]
    drifts = [
        e for e in events if e["payload"].get("process") == "patch_drift"
    ]
    regrowths = [
        e for e in events if e["payload"].get("process") == "regrowth"
    ]
    assert len(drifts) == 2  # steps 3 and 6
    assert drifts[0]["payload"]["center_from"] == [3, 3]
    assert drifts[0]["payload"]["center_to"] == [4, 4]
    assert drifts[0]["payload"]["cell"] == [4, 4]
    assert drifts[1]["payload"]["center_from"] == [4, 4]
    assert drifts[1]["payload"]["center_to"] == [5, 5]
    # Regrowth events keep the unchanged matched-control shape.
    assert regrowths, "no regrowth events under p_inside=1.0"
    for event in regrowths:
        assert event["payload"]["pre_state"] == "empty"
        assert event["payload"]["post_state"] == "resource"


# ---- validation and preset --------------------------------------------------------


def test_patch_validation() -> None:
    with pytest.raises(ValueError, match="regrowth_mode"):
        GridWorld(_patch_config(regrowth_mode="fog"), seed=1)
    with pytest.raises(ValueError, match="positive and odd"):
        GridWorld(_patch_config(patch_size=2), seed=1)
    with pytest.raises(ValueError, match="patch_step_every"):
        GridWorld(_patch_config(patch_step_every=0), seed=1)
    with pytest.raises(ValueError, match="patch_p_inside"):
        GridWorld(_patch_config(patch_p_inside=1.5), seed=1)


def test_stage_e3_is_cumulative() -> None:
    """e3 lands BEFORE e2 (builder-ratified reorder, 2026-07-09): the
    weather stage carries continuity + terrain + trail but NO clock."""
    staged = apply_world_stage(GridWorldConfig(), "e3")
    assert staged.episode_resample is False
    assert staged.trail_enabled is True
    assert staged.bloom_cell is None, "the clock must not land with e3"
    assert staged.regrowth_mode == "patch"
    assert staged.patch_step_every == PATCH_STEP_EVERY
    assert staged.patch_p_inside == PATCH_P_INSIDE
    assert staged.patch_p_outside == PATCH_P_OUTSIDE
    assert staged.patch_expiry_p == PATCH_EXPIRY_P


# ---- the off-patch expiry amendment (ratified 2026-07-09) ---------------------


def test_expiry_default_off_is_byte_identical() -> None:
    """patch mode without expiry keeps the pre-amendment stream: same
    seed, same trajectory, including full-board no-draw steps."""
    world_a = GridWorld(_patch_config(), seed=42)
    world_b = GridWorld(_patch_config(patch_expiry_p=0.0), seed=42)
    world_a.reset()
    world_b.reset()
    rng = np.random.default_rng(3)
    for action in (int(a) for a in rng.integers(0, 5, size=200)):
        step_a = world_a.step(action)
        step_b = world_b.step(action)
        assert np.array_equal(step_a.observation, step_b.observation)


def test_expiry_only_off_patch() -> None:
    """With expiry_p=1, every off-patch resource vanishes each step
    while the patch square's food survives."""
    world = GridWorld(
        _patch_config(
            patch_p_inside=1.0,
            patch_p_outside=0.0,
            patch_expiry_p=1.0,
            patch_step_every=1000,
            n_initial_resources=20,
        ),
        seed=42,
    )
    world.reset()
    world.step(STAY)
    grid = world.state.grid
    resources = {
        (int(r), int(c))
        for r, c in np.argwhere(grid == CellType.RESOURCE.value)
    }
    patch_square = {(r, c) for r in (2, 3, 4) for c in (2, 3, 4)}
    agent_cell = {world.state.agent_pos}
    assert resources == patch_square - agent_cell


def test_regrown_cell_does_not_expire_same_step() -> None:
    """Pre-state discipline: a cell regrowing this step survives this
    step even at expiry_p=1 (it expires from the next step on)."""
    world = GridWorld(
        _patch_config(
            patch_p_inside=1.0,
            patch_p_outside=1.0,  # off-patch cells regrow every step...
            patch_expiry_p=1.0,  # ...and expire every following step
            patch_step_every=1000,
            n_initial_resources=0,
        ),
        seed=42,
    )
    world.reset()
    world.step(STAY)
    grid = world.state.grid
    # Everything empty regrew this step and nothing had existed before,
    # so nothing could expire: the board (minus agent cell) is full.
    off_patch_resources = int(
        np.count_nonzero(grid == CellType.RESOURCE.value)
    )
    assert off_patch_resources >= 60  # 63 non-agent cells, no walls
    world.step(STAY)
    # Now the off-patch cells (pre-state resources) expire — and regrow
    # again in the same step? No: regrowth reads pre-state EMPTY, so
    # expired-this-step cells stay empty until the next step.
    grid = world.state.grid
    patch_square = {(r, c) for r in (2, 3, 4) for c in (2, 3, 4)}
    resources = {
        (int(r), int(c))
        for r, c in np.argwhere(grid == CellType.RESOURCE.value)
    }
    assert resources <= patch_square


def test_expiry_events_are_world_reported(tmp_path: Path) -> None:
    sink_path = tmp_path / "world_event.jsonl"
    sink = JsonlSink(sink_path, WorldEvent)
    server = EnvServer(
        EnvServerConfig(
            grid_world_config=_patch_config(
                patch_p_inside=0.0,
                patch_p_outside=0.0,
                patch_expiry_p=1.0,
                patch_step_every=1000,
                n_initial_resources=4,
                start_cell=(2, 3),  # inside the patch: never expires
            ),
            seed=42,
            world_event_handler=sink.write,
            run_id="expiry-test",
            emit_internal_stochasticity_events=True,
        )
    )
    try:
        server.start()
        initial = server.grid_world_state.grid
        off_patch = [
            (int(r), int(c))
            for r, c in np.argwhere(initial == CellType.RESOURCE.value)
            if not (2 <= r <= 4 and 2 <= c <= 4)
        ]
        server.step(STAY)
    finally:
        server.close()
        sink.close()

    events = [
        json.loads(line) for line in sink_path.read_text().splitlines()
    ]
    expiries = [
        e
        for e in events
        if e["payload"].get("process") == "resource_expiry"
    ]
    assert sorted(tuple(e["payload"]["cell"]) for e in expiries) == sorted(
        off_patch
    )
    for event in expiries:
        assert event["payload"]["pre_state"] == "resource"
        assert event["payload"]["post_state"] == "empty"
        assert event["source"] == "environment"


# ---- the occupancy-share diagnostic -------------------------------------------------


def test_occupancy_share_from_sidecar_and_events(tmp_path: Path) -> None:
    analyzer = _load_analyzer()
    telemetry = tmp_path / "telemetry"
    telemetry.mkdir(parents=True)
    # Patch at (3,3) until t=10, then at (5,5).
    events = [
        {
            "record_version": "0.4.0",
            "run_id": "occ",
            "t_event": 10,
            "event_type": "internal_stochasticity_event",
            "source": "environment",
            "payload": {
                "process": "patch_drift",
                "cell": [5, 5],
                "pre_state": "patch_absent",
                "post_state": "patch_present",
                "center_from": [3, 3],
                "center_to": [5, 5],
            },
        }
    ]
    (telemetry / "world_event.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n"
    )
    # Io inside the (3,3) square for t<10, far away after.
    lines = [
        json.dumps({"t": t, "pos": [3, 3] if t < 10 else [0, 0]})
        for t in range(20)
    ]
    (tmp_path / "agent_pos.jsonl").write_text("\n".join(lines) + "\n")

    positions = analyzer._positions(tmp_path)
    centers = analyzer._patch_centers(telemetry)
    share_before = analyzer._occupancy_share(positions, centers, 0, 10)
    share_after = analyzer._occupancy_share(positions, centers, 10, 20)
    assert share_before == 1.0
    assert share_after == 0.0
    # Missing records → None, never a crash.
    assert analyzer._occupancy_share({}, centers, 0, 10) is None
    assert analyzer._occupancy_share(positions, [], 0, 10) is None
