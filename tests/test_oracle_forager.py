"""Probe 3.5 — oracle feasibility instrument (Amendment 02 §6).

Tests the pre-committed oracle policy (forage-below-setpoint via BFS, stay
otherwise) and the feasibility harness. The oracle is an observer-side
instrument — not Io, no learning — so these tests touch no Io code path; the
verdict on the *default physics* is rendered by
``scripts/run_probe3_5_oracle_feasibility.py``, not asserted here (asserting a
pass would fit a test to an empirical outcome).
"""

from __future__ import annotations

import numpy as np
import pytest

from kind.env.grid_world import CellType, GridWorldConfig
from kind.observer.oracle_forager import oracle_action, run_oracle_feasibility

_E = CellType.EMPTY.value
_W = CellType.WALL.value
_R = CellType.RESOURCE.value
_STAY = 4


def _grid(rows: list[list[int]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.uint8)


def test_oracle_stays_when_at_or_above_setpoint() -> None:
    g = _grid([[_E, _R], [_E, _E]])
    assert oracle_action(g, (0, 0), 0.6, setpoint=0.6) == _STAY
    assert oracle_action(g, (0, 0), 0.95, setpoint=0.6) == _STAY


def test_oracle_steps_toward_adjacent_resource_when_below_setpoint() -> None:
    # Resource directly right of the agent → action 3 (0,+1).
    g = _grid([[_E, _R], [_E, _E]])
    assert oracle_action(g, (0, 0), 0.45, setpoint=0.6) == 3
    # Resource directly below → action 1 (+1,0).
    g2 = _grid([[_E, _E], [_R, _E]])
    assert oracle_action(g2, (0, 0), 0.45, setpoint=0.6) == 1


def test_oracle_bfs_routes_around_walls() -> None:
    # Wall between agent (0,0) and resource (0,2): direct row blocked at (0,1);
    # the shortest path goes down and around, so the first step is down (1).
    g = _grid(
        [
            [_E, _W, _R],
            [_E, _E, _E],
        ]
    )
    assert oracle_action(g, (0, 0), 0.3, setpoint=0.6) == 1


def test_oracle_stays_when_no_resource_exists() -> None:
    g = _grid([[_E, _E], [_E, _E]])
    assert oracle_action(g, (0, 0), 0.1, setpoint=0.6) == _STAY


def test_oracle_first_step_of_shortest_path_multistep() -> None:
    # Resource two steps right; nearest first action must be right (3).
    g = _grid([[_E, _E, _R], [_E, _E, _E]])
    assert oracle_action(g, (0, 0), 0.4, setpoint=0.6) == 3


def test_run_oracle_feasibility_is_deterministic_and_well_formed() -> None:
    cfg = GridWorldConfig()
    r1 = run_oracle_feasibility(cfg, n_seeds=2, episodes_per_seed=2)
    r2 = run_oracle_feasibility(cfg, n_seeds=2, episodes_per_seed=2)
    assert r1.per_seed_occupancy == r2.per_seed_occupancy
    assert r1.pooled_occupancy == r2.pooled_occupancy
    assert len(r1.per_seed_occupancy) == 2
    assert all(0.0 <= o <= 1.0 for o in r1.per_seed_occupancy)
    # Band is the fixed B0a′ band by default.
    assert r1.band_low == pytest.approx(0.45)
    assert r1.band_high == pytest.approx(0.75)
    assert r1.passed == (r1.pooled_occupancy >= r1.threshold)
