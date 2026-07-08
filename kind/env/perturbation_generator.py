"""Probe 4 Phase 2 — stochastic perturbation generator (plan §S-PERT).

The generator is a **pure decision engine**: given its seeded RNG and the
post-step world state, :meth:`PerturbationGenerator.poll` decides at each
step boundary whether a builder perturbation fires now and where. It
mutates nothing itself — the runner drains it inside ``_step_once``
(between the env step and the ``AgentStep`` emit) and fires the returned
cells through the co-located env-server's tested mutator surface with
``trigger="generator"``, so every generator event flows through the same
``source="builder"`` emission path (and the same pixel-equality
no-marker property) as a manual event.

**The statistical signature (plan open item 3 — proposed here, journaled,
revisitable before the Phase 4 run; a stimulus-design knob, not a frozen
success criterion).** The generator's events stay inside the
environment's own vocabulary-in-use (resource additions — exactly what
regrowth produces), so the builder class is **never content-identified**
(synthesis T5: a reserved tile/type would be a de facto observation
marker and the probe would pass trivially for the wrong reason). The
separable axes are:

- **Timing** — an under-dispersed renewal process: inter-event gap =
  ``min_spacing_steps + U{0..spacing_jitter_steps}``. The floor is the
  Phase-1 measured envelope (raw h-trace horizon ≈ 40 steps: traces
  never overlap), and the bounded gap is far from regrowth's memoryless
  per-cell Bernoulli. Jitter keeps timing genuinely stochastic so the
  category cannot be carried by clock-prediction alone (the
  marginal-rate deflation needs something to fail on).
- **Magnitude / within-event structure** — ``cells_per_event`` adjacent
  cells appear in one step (default 2). Regrowth produces simultaneous
  adjacent pairs only at ~p² per adjacency (~1 per hundred steps at
  p=0.01) — rare enough to be a signature, common enough that
  magnitude-matched ENVIRONMENT controls exist in a long run (the T1
  matched-control requirement cuts both ways).
- **Placement** — otherwise matched to regrowth's law (uniform over
  eligible EMPTY cells), *except* the not-self exclusion below.

**Legibly not-self (synthesis §3.4 / T5).** Decoupled from Io's actions
in place and time: no cell within Chebyshev distance
``exclusion_radius`` of Io's position (default 1 — never at Io's cell or
its 8 neighbours, so the event can never coincide with where Io's own
action is resolving), and — when ``defer_on_consumption`` is on — never
on a step boundary where Io just consumed (the runner detects the
consumption from the true-energy jump and the generator defers to the
next boundary). This is the *opposite* of a contingency design: it
removes coincidences with Io's actions rather than creating them.

**Determinism.** Given the same seed and the same world trajectory
(grids, agent positions, consumption flags), the generator produces the
identical event schedule and placements — RNG draws happen only inside
``poll``, whose call sequence is fixed by the trajectory. Deferrals
(consumption, no eligible cluster) re-attempt at the next boundary and
are part of that deterministic schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from kind.env.grid_world import CellType

__all__ = [
    "PerturbationGeneratorConfig",
    "PlannedPerturbation",
    "PerturbationGenerator",
]


# 4-adjacency, fixed order (up, down, left, right) so cluster growth is
# deterministic given the anchor.
_NEIGHBOR_DELTAS: Final[tuple[tuple[int, int], ...]] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)


@dataclass(frozen=True)
class PerturbationGeneratorConfig:
    """Builder-set generator parameters (plan §S-PERT).

    ``min_spacing_steps`` defaults to the Phase-1 measured envelope floor
    (raw h-trace horizon ≈ 40 steps on the Step-0 instance — journal,
    Phase 1); re-measure on the actual Phase-4 instance before fixing the
    final rate. The Phase-3 positive control expresses its deliberately
    blatant planted category through this same config (small spacing,
    large clusters) — no separate machinery.
    """

    seed: int
    min_spacing_steps: int = 40
    spacing_jitter_steps: int = 40
    cells_per_event: int = 2
    exclusion_radius: int = 1
    defer_on_consumption: bool = True

    def __post_init__(self) -> None:
        if self.min_spacing_steps < 1:
            raise ValueError(
                f"min_spacing_steps must be >= 1, got {self.min_spacing_steps}"
            )
        if self.spacing_jitter_steps < 0:
            raise ValueError(
                f"spacing_jitter_steps must be >= 0, got "
                f"{self.spacing_jitter_steps}"
            )
        if self.cells_per_event < 1:
            raise ValueError(
                f"cells_per_event must be >= 1, got {self.cells_per_event}"
            )
        if self.exclusion_radius < 0:
            raise ValueError(
                f"exclusion_radius must be >= 0, got {self.exclusion_radius}"
            )


@dataclass(frozen=True)
class PlannedPerturbation:
    """One generator firing: the adjacent cell cluster to add resources at.

    The runner fires each cell through
    ``EnvServer.add_resource(cell, trigger="generator")`` in the given
    order (anchor first, then adjacency-order growth), all within one
    step boundary — one *event*, ``len(cells)`` mutator records sharing
    one ``t_event``.
    """

    cells: tuple[tuple[int, int], ...]


class PerturbationGenerator:
    """Seeded step-boundary decision engine. See module docstring."""

    def __init__(self, config: PerturbationGeneratorConfig) -> None:
        self._config = config
        self._rng = np.random.default_rng(config.seed)
        # The first event becomes due this many steps into the run; the
        # same gap law as every subsequent inter-event gap.
        self._next_due_step: int = self._draw_gap()

    @property
    def next_due_step(self) -> int:
        """The env step at/after which the next event fires (read-only,
        for tests and telemetry)."""
        return self._next_due_step

    def _draw_gap(self) -> int:
        jitter = int(
            self._rng.integers(0, self._config.spacing_jitter_steps + 1)
        )
        return self._config.min_spacing_steps + jitter

    def poll(
        self,
        *,
        env_step: int,
        grid: NDArray[np.uint8],
        agent_pos: tuple[int, int],
        consumed_last_step: bool,
    ) -> PlannedPerturbation | None:
        """Decide whether an event fires at this step boundary.

        ``grid`` / ``agent_pos`` are the post-step world state (the
        mutation applies to that state and becomes visible in the *next*
        observation — the documented builder-event timing convention).
        Returns ``None`` when not yet due, or on a deferral (consumption
        co-timing; no eligible not-self cluster) — deferrals re-attempt
        at the next boundary.
        """
        if env_step < self._next_due_step:
            return None
        if consumed_last_step and self._config.defer_on_consumption:
            # Not-self in time: never co-timed with Io's own consumption.
            return None
        cluster = self._select_cluster(grid, agent_pos)
        if cluster is None:
            return None
        self._next_due_step = env_step + self._draw_gap()
        return PlannedPerturbation(cells=cluster)

    # ---- placement -------------------------------------------------------

    def _eligible_mask(
        self, grid: NDArray[np.uint8], agent_pos: tuple[int, int]
    ) -> NDArray[np.bool_]:
        """EMPTY cells outside the Chebyshev exclusion radius around Io."""
        eligible = grid == CellType.EMPTY.value
        ar, ac = agent_pos
        radius = self._config.exclusion_radius
        rows = np.arange(grid.shape[0])[:, None]
        cols = np.arange(grid.shape[1])[None, :]
        chebyshev = np.maximum(np.abs(rows - ar), np.abs(cols - ac))
        mask: NDArray[np.bool_] = eligible & (chebyshev > radius)
        return mask

    def _select_cluster(
        self, grid: NDArray[np.uint8], agent_pos: tuple[int, int]
    ) -> tuple[tuple[int, int], ...] | None:
        """Pick an adjacent cluster of ``cells_per_event`` eligible cells.

        Anchors are tried in RNG-permuted order (this is the placement
        randomness — uniform over eligible anchors); cluster growth from
        each anchor is deterministic (fixed adjacency order, breadth-
        first over already-chosen cells). The first anchor that yields a
        full cluster wins; if none does, the event defers.
        """
        eligible = self._eligible_mask(grid, agent_pos)
        candidates = [
            (int(r), int(c)) for r, c in np.argwhere(eligible)
        ]
        if not candidates:
            return None
        needed = self._config.cells_per_event
        order = self._rng.permutation(len(candidates))
        grid_size = grid.shape[0]
        for index in order:
            anchor = candidates[int(index)]
            cluster: list[tuple[int, int]] = [anchor]
            chosen = {anchor}
            # Grow: scan cluster members in order, their neighbours in
            # fixed adjacency order, taking eligible unchosen cells.
            cursor = 0
            while len(cluster) < needed and cursor < len(cluster):
                base_r, base_c = cluster[cursor]
                for dr, dc in _NEIGHBOR_DELTAS:
                    if len(cluster) >= needed:
                        break
                    nr, nc = base_r + dr, base_c + dc
                    if not (0 <= nr < grid_size and 0 <= nc < grid.shape[1]):
                        continue
                    if (nr, nc) in chosen:
                        continue
                    if not bool(eligible[nr, nc]):
                        continue
                    cluster.append((nr, nc))
                    chosen.add((nr, nc))
                cursor += 1
            if len(cluster) == needed:
                return tuple(cluster)
        return None
