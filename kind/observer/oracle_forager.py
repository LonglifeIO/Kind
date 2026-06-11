"""Probe 3.5 — environment-feasibility oracle (Amendment 02 §6).

A **scripted nearest-resource forager**: an observer-side instrument that
demonstrates the world is *winnable by competence* under the operative physics —
oracle in-band occupancy ≥ F1 over F2 seeds × episodes — **before** Phase 2
trains anything. With feasibility established, Phase 2 measures whether the
preference *produces* the competence, not whether the world *permits* it.

This is **not Io and involves no learning**: the oracle reads
``GridState.true_energy`` directly (it is an instrument — no opacity constraint
applies to it), and its policy is pre-committed in the amendment so it cannot be
fitted later:

    when ``true_energy <`` setpoint → step along a BFS shortest path to the
    nearest resource cell; otherwise → stay (entry-triggered consumption means
    staying never replenishes).

Deterministic given the env seed. Touches no Io code path: imports only the env.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from kind.env.grid_world import (
    _ACTION_DELTAS,
    CellType,
    GridWorld,
    GridWorldConfig,
)

__all__ = [
    "OracleFeasibilityReport",
    "oracle_action",
    "run_oracle_feasibility",
]

_STAY = 4
_MOVE_ACTIONS = (0, 1, 2, 3)


def oracle_action(
    grid: NDArray[np.uint8],
    agent_pos: tuple[int, int],
    true_energy: float,
    *,
    setpoint: float = 0.6,
) -> int:
    """The pre-committed oracle policy (Amendment 02 §6) — a pure function.

    Returns the first action of a BFS shortest path from ``agent_pos`` to the
    nearest RESOURCE cell when ``true_energy < setpoint``; ``stay`` otherwise,
    and ``stay`` when no resource exists (waiting for regrowth costs only base
    decay). BFS expands neighbours in fixed action order (up, down, left,
    right), so tie-breaking is deterministic.
    """
    if true_energy >= setpoint:
        return _STAY

    n_rows, n_cols = grid.shape
    # BFS from the agent over non-wall cells; first action recorded per cell.
    visited = np.zeros((n_rows, n_cols), dtype=bool)
    visited[agent_pos] = True
    queue: deque[tuple[tuple[int, int], int]] = deque()
    for a in _MOVE_ACTIONS:
        dr, dc = _ACTION_DELTAS[a]
        nr, nc = agent_pos[0] + dr, agent_pos[1] + dc
        if not (0 <= nr < n_rows and 0 <= nc < n_cols):
            continue
        if grid[nr, nc] == CellType.WALL.value:
            continue
        if grid[nr, nc] == CellType.RESOURCE.value:
            return a
        visited[nr, nc] = True
        queue.append(((nr, nc), a))

    while queue:
        (r, c), first_action = queue.popleft()
        for a in _MOVE_ACTIONS:
            dr, dc = _ACTION_DELTAS[a]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                continue
            if visited[nr, nc]:
                continue
            if grid[nr, nc] == CellType.WALL.value:
                continue
            if grid[nr, nc] == CellType.RESOURCE.value:
                return first_action
            visited[nr, nc] = True
            queue.append(((nr, nc), first_action))

    # No resource reachable — wait for regrowth at minimum (stay) cost.
    return _STAY


@dataclass(frozen=True)
class OracleFeasibilityReport:
    """The §6 feasibility verdict: per-seed and pooled in-band occupancy."""

    per_seed_occupancy: tuple[float, ...]
    pooled_occupancy: float
    band_low: float
    band_high: float
    n_seeds: int
    episodes_per_seed: int
    threshold: float

    @property
    def passed(self) -> bool:
        return self.pooled_occupancy >= self.threshold


def run_oracle_feasibility(
    grid_cfg: GridWorldConfig,
    *,
    n_seeds: int = 8,
    episodes_per_seed: int = 20,
    setpoint: float = 0.6,
    band_halfwidth: float = 0.15,
    threshold: float = 0.70,
    seed_base: int = 0,
) -> OracleFeasibilityReport:
    """Run the oracle per the confirmed F1/F2 protocol; report in-band occupancy.

    F2 = ``n_seeds`` × ``episodes_per_seed`` (200-step episodes, energy carrying
    across the soft boundary, resources resampling per episode — the frozen env
    mechanics). In-band = ``true_energy`` within ``setpoint ± band_halfwidth``
    (the fixed B0a′ band), sampled pre-step as in the existing eval harnesses.
    Pass = pooled occupancy ≥ ``threshold`` (F1).
    """
    band_low = setpoint - band_halfwidth
    band_high = setpoint + band_halfwidth
    n_steps = grid_cfg.episode_length * episodes_per_seed

    per_seed: list[float] = []
    for s in range(n_seeds):
        world = GridWorld(grid_cfg, seed=seed_base + s)
        world.reset()
        in_band = 0
        for _ in range(n_steps):
            state = world.state
            e = state.true_energy
            if band_low <= e <= band_high:
                in_band += 1
            action = oracle_action(
                state.grid, state.agent_pos, e, setpoint=setpoint
            )
            world.step(action)
        per_seed.append(in_band / n_steps)

    pooled = float(np.mean(per_seed))
    return OracleFeasibilityReport(
        per_seed_occupancy=tuple(per_seed),
        pooled_occupancy=pooled,
        band_low=band_low,
        band_high=band_high,
        n_seeds=n_seeds,
        episodes_per_seed=episodes_per_seed,
        threshold=threshold,
    )
