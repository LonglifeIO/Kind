"""Reachable-set size — the world-v3 §7 monitor (synthesis S3, ratified
2026-08-14).

With e5's pushable blocks, Io can for the first time construct its own
confinement (self-walling: pushing blocks so its reachable set shrinks
while single-cell occupancy rises). This module computes the size of the
4-connected region of non-WALL cells containing Io — cheap, torch-free,
computable live (from ``GridState.grid``) or by replay (walls from
config + block positions from the ``block_pos.jsonl`` sidecar).

Two deliberate reading notes, journaled with the S4 pre-registration:

* The count treats every WALL-vocabulary cell as an obstacle — the
  mover included. The mover shifts the live count by at most its own
  blocking geometry and moves every 2 steps, so persistent shrinkage is
  attributable to walls-plus-blocks, not to it.
* A pushed block *may* legitimately shrink the reachable set without
  pathology (e.g. sealing a dead-end corridor cell). The §7 tripwire is
  the pre-registered *conjunction* — reachable-set shrinking while
  single-cell occupancy rises — not any single reading of this number.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from kind.env.grid_world import CellType

__all__ = ["reachable_set_size", "reachable_set_size_from_layout"]


def reachable_set_size(
    grid: NDArray[np.uint8], agent_pos: tuple[int, int]
) -> int:
    """Count cells 4-connected-reachable from ``agent_pos``.

    ``grid`` is a ``GridState.grid``-shaped array; every cell whose
    value is not ``CellType.WALL`` is traversable (EMPTY, TRAIL,
    RESOURCE — Io can enter all of them). The count includes Io's own
    cell. ``agent_pos`` on a WALL value raises — that state is
    impossible in a live world and a replay reaching it is corrupt.
    """
    rows, cols = grid.shape
    r0, c0 = agent_pos
    if not (0 <= r0 < rows and 0 <= c0 < cols):
        raise ValueError(
            f"agent_pos {agent_pos} is out of bounds for grid "
            f"shape {grid.shape}"
        )
    if int(grid[r0, c0]) == CellType.WALL.value:
        raise ValueError(
            f"agent_pos {agent_pos} sits on a WALL cell — impossible "
            f"live state; a replay reaching it has diverged"
        )
    seen = {(r0, c0)}
    frontier: deque[tuple[int, int]] = deque(seen)
    while frontier:
        r, c = frontier.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if (nr, nc) in seen:
                continue
            if int(grid[nr, nc]) == CellType.WALL.value:
                continue
            seen.add((nr, nc))
            frontier.append((nr, nc))
    return len(seen)


def reachable_set_size_from_layout(
    grid_size: int,
    obstacles: tuple[tuple[int, int], ...],
    agent_pos: tuple[int, int],
) -> int:
    """Replay-side convenience: walls + block positions, no grid array.

    ``obstacles`` is the union of static walls and current block cells
    (and the mover's cell, if the replay tracks it). Builds the boolean
    layout and delegates to :func:`reachable_set_size`.
    """
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    for r, c in obstacles:
        grid[r, c] = CellType.WALL.value
    return reachable_set_size(grid, agent_pos)
