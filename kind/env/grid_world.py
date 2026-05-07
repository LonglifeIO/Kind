"""Phase 2a env-server core.

A bounded 8×8 grid with three cell types (empty, wall, resource), a single
agent (Io) at a fixed start cell by default, two independent stochastic
processes — per-cell regrowth and aperiodic random-walk drift in the
regrowth rate — and fixed-length 200-step episodes that auto-reset with no
terminal-state signal in the observation. The agent sees a 7×7 ego-centric
partial view of the grid rendered to a 32×32 grayscale image; the underlying
``GridState`` is exposed mirror-side for ground-truth context (Probe 4 will
use this) but never reaches the agent.

The environment lives alone in this module. There is no harness, no
mutators, no TCP transport, no telemetry emission — those are Phase 3 and
Phase 4. Phase 2a is the env in isolation. NumPy only; PyTorch is not
imported.

Structural choices recorded in ``docs/workingjournal/probe1.md`` rather
than here (drift-magnitude arithmetic, initial-resource count, start cell,
walls, p-carries-across-episodes). The numbers in ``GridWorldConfig``
reflect the journal's documented defaults.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Final

import numpy as np
from numpy.typing import NDArray


# ---- constants -------------------------------------------------------------


class CellType(IntEnum):
    """Underlying-grid cell vocabulary.

    Stored as ``uint8`` in the grid array. The integer values are part of
    the contract: rendering and out-of-bounds sentinel logic indexes a
    fixed lookup table by cell value.
    """

    EMPTY = 0
    WALL = 1
    RESOURCE = 2


# Out-of-bounds sentinel value used when an ego-centric view extends past the
# grid edge. Distinct from every ``CellType`` value so the renderer can map
# it to a distinguishable grayscale level.
_OOB_SENTINEL: Final[int] = 3


# Grayscale rendering table indexed by cell-or-sentinel value (0..3).
# EMPTY → 128 (mid gray), WALL → 0 (black), RESOURCE → 255 (white),
# OOB → 64 (dark gray, distinct from every in-bounds cell).
_RENDER_TABLE: Final[NDArray[np.uint8]] = np.array(
    [128, 0, 255, 64], dtype=np.uint8
)


# Action displacement vectors: (dr, dc). Order is part of the public contract:
# 0=up, 1=down, 2=left, 3=right, 4=stay.
_ACTION_DELTAS: Final[tuple[tuple[int, int], ...]] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (0, 0),
)


NUM_ACTIONS: Final[int] = 5


# ---- configuration and step records ---------------------------------------


@dataclass(frozen=True)
class GridWorldConfig:
    """Static configuration for the grid world.

    Defaults reflect Probe 1 starting points per the implementation plan
    §2.3 and the environment synthesis §3. The drift-magnitude derivation
    (±10% over 50 episodes ⇒ σ_step ≈ 1e-5) is recorded in the journal.
    ``walls`` is empty at Probe 1; the cell type exists in the vocabulary
    but no walls are placed by default.
    """

    grid_size: int = 8
    view_size: int = 7
    obs_resolution: int = 32
    episode_length: int = 200
    initial_regrowth_p: float = 0.01
    drift_magnitude_per_step: float = 1e-5
    drift_p_min: float = 0.001
    drift_p_max: float = 0.05
    n_initial_resources: int = 4
    start_cell: tuple[int, int] | None = (3, 3)
    walls: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class EnvStep:
    """One environment step's agent-visible record.

    ``observation`` is a read-only ``uint8`` array shaped
    ``(obs_resolution, obs_resolution)``. Phase 3a's harness will wrap this
    into the ``AgentStep`` Pydantic record; Phase 2a just produces the
    structured tuple.
    """

    observation: NDArray[np.uint8]
    env_step: int
    episode_id: int
    step_in_episode: int
    wallclock_ms: int


@dataclass(frozen=True)
class GridState:
    """Mirror-side ground-truth view of the underlying world.

    Exposed by ``GridWorld.state`` for Probe 4's eventual
    builder-vs-internal-stochasticity distinguishability test and for
    test assertions. Io's actor never sees this — the only agent-visible
    surface is ``EnvStep.observation``.
    """

    grid: NDArray[np.uint8]
    agent_pos: tuple[int, int]
    regrowth_p: float


# ---- the environment ------------------------------------------------------


class GridWorld:
    """Bounded grid with regrowth dynamics and aperiodic drift in p.

    Determinism: a single integer seed feeds ``numpy.random.SeedSequence``
    which spawns two child sequences, one per stochastic process (regrowth,
    drift). Two ``GridWorld`` instances constructed with the same seed and
    stepped through the same action sequence produce identical observations
    and identical ``GridState`` trajectories.

    Episode boundary: the 200th step after a ``reset()`` (i.e. the step at
    which ``step_in_episode`` would equal ``episode_length``) re-samples
    fresh resources and replaces the agent at the start cell, increments
    ``episode_id``, and emits an ``EnvStep`` with ``step_in_episode=0`` of
    the new episode. The drift parameter ``p`` carries across episodes —
    the random walk does not reset.
    """

    def __init__(self, config: GridWorldConfig, seed: int) -> None:
        self.config = config
        self._validate_config()

        # Two independent streams from a single integer seed.
        seed_sequence = np.random.SeedSequence(seed)
        regrowth_seq, drift_seq = seed_sequence.spawn(2)
        self._regrowth_rng: np.random.Generator = np.random.default_rng(
            regrowth_seq
        )
        self._drift_rng: np.random.Generator = np.random.default_rng(drift_seq)

        # Mutable state (initialized by reset()).
        self._grid: NDArray[np.uint8] = np.zeros(
            (config.grid_size, config.grid_size), dtype=np.uint8
        )
        self._agent_pos: tuple[int, int] = (0, 0)
        self._regrowth_p: float = config.initial_regrowth_p
        self._env_step: int = 0
        self._episode_id: int = 0
        self._step_in_episode: int = 0
        self._initialized: bool = False

    # ---- public API -------------------------------------------------------

    def reset(self) -> EnvStep:
        """Reset to a fresh first episode and return its initial ``EnvStep``.

        Drift is also reset to ``initial_regrowth_p``; this is the only path
        that resets ``p``. Subsequent within-run episode boundaries carry
        ``p`` across.
        """
        self._regrowth_p = self.config.initial_regrowth_p
        self._env_step = 0
        self._episode_id = 0
        self._step_in_episode = 0
        self._reset_episode_world()
        self._initialized = True
        return self._make_env_step()

    def step(self, action: int) -> EnvStep:
        """Apply ``action``, advance dynamics, return the resulting ``EnvStep``.

        Off-grid moves and wall collisions leave the agent's position
        unchanged, but the clock still ticks and both stochastic processes
        still advance. Resource consumption is a state change triggered by
        entering a resource cell — there is no separate ``consume`` verb.
        """
        if not self._initialized:
            raise RuntimeError("GridWorld.reset() must be called before step()")
        if not isinstance(action, int) or isinstance(action, bool):
            raise TypeError(f"action must be int, got {type(action).__name__}")
        if not 0 <= action < NUM_ACTIONS:
            raise ValueError(
                f"action must be in [0, {NUM_ACTIONS}); got {action}"
            )

        self._apply_action(action)
        self._update_drift()
        self._update_regrowth()

        self._env_step += 1
        self._step_in_episode += 1

        if self._step_in_episode >= self.config.episode_length:
            # Episode boundary. Drift carries; resources are resampled fresh;
            # agent is replaced at the start cell. The returned EnvStep
            # reflects the new (post-reset) world — there is no terminal-state
            # signal in the observation.
            self._reset_episode_world()
            self._step_in_episode = 0
            self._episode_id += 1

        return self._make_env_step()

    @property
    def state(self) -> GridState:
        """Return a mirror-side snapshot of the underlying world state."""
        grid_copy = self._grid.copy()
        grid_copy.setflags(write=False)
        return GridState(
            grid=grid_copy,
            agent_pos=self._agent_pos,
            regrowth_p=self._regrowth_p,
        )

    # ---- validation -------------------------------------------------------

    def _validate_config(self) -> None:
        c = self.config
        if c.grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {c.grid_size}")
        if c.view_size <= 0 or c.view_size % 2 == 0:
            raise ValueError(
                f"view_size must be positive and odd (so it has a center); "
                f"got {c.view_size}"
            )
        if c.obs_resolution <= 0:
            raise ValueError(
                f"obs_resolution must be positive, got {c.obs_resolution}"
            )
        if c.episode_length <= 0:
            raise ValueError(
                f"episode_length must be positive, got {c.episode_length}"
            )
        if not 0.0 <= c.drift_p_min <= c.drift_p_max <= 1.0:
            raise ValueError(
                "must satisfy 0 <= drift_p_min <= drift_p_max <= 1; got "
                f"({c.drift_p_min}, {c.drift_p_max})"
            )
        if not c.drift_p_min <= c.initial_regrowth_p <= c.drift_p_max:
            raise ValueError(
                f"initial_regrowth_p={c.initial_regrowth_p} must be in "
                f"[{c.drift_p_min}, {c.drift_p_max}]"
            )
        if c.drift_magnitude_per_step < 0:
            raise ValueError(
                "drift_magnitude_per_step must be non-negative, got "
                f"{c.drift_magnitude_per_step}"
            )
        if c.n_initial_resources < 0:
            raise ValueError(
                f"n_initial_resources must be non-negative, got "
                f"{c.n_initial_resources}"
            )

        wall_set: set[tuple[int, int]] = set()
        for wall in c.walls:
            wr, wc = wall
            if not (0 <= wr < c.grid_size and 0 <= wc < c.grid_size):
                raise ValueError(
                    f"wall {wall!r} is out of grid bounds "
                    f"[0, {c.grid_size})²"
                )
            wall_set.add(wall)

        if c.start_cell is not None:
            sr, sc = c.start_cell
            if not (0 <= sr < c.grid_size and 0 <= sc < c.grid_size):
                raise ValueError(
                    f"start_cell {c.start_cell!r} out of grid bounds"
                )
            if c.start_cell in wall_set:
                raise ValueError(
                    f"start_cell {c.start_cell!r} cannot be a wall"
                )

        # The agent occupies one non-wall cell; resources cannot share it on
        # placement, so the available pool is grid_size² − walls − 1.
        n_cells = c.grid_size * c.grid_size
        n_available = n_cells - len(wall_set)
        if c.start_cell is not None:
            n_available -= 1
        if c.n_initial_resources > n_available:
            raise ValueError(
                f"n_initial_resources={c.n_initial_resources} exceeds "
                f"available non-wall non-agent cells ({n_available})"
            )

    # ---- internal mechanics ----------------------------------------------

    def _reset_episode_world(self) -> None:
        """Place the agent at the start cell, sample fresh resources.

        Drift parameter ``p`` is preserved by this method; only ``reset()``
        re-initializes ``p``. Walls are restored from config.
        """
        # Place agent. Random start cell draws from the regrowth stream so
        # there is no third RNG to seed — spatial events on cells share one
        # source of randomness.
        if self.config.start_cell is None:
            r = int(self._regrowth_rng.integers(0, self.config.grid_size))
            c = int(self._regrowth_rng.integers(0, self.config.grid_size))
            # If random landed on a wall, draw again until it doesn't.
            while self._is_wall(r, c):
                r = int(self._regrowth_rng.integers(0, self.config.grid_size))
                c = int(self._regrowth_rng.integers(0, self.config.grid_size))
            self._agent_pos = (r, c)
        else:
            self._agent_pos = self.config.start_cell

        # Initialize the grid: empty everywhere, walls written in.
        self._grid[:] = CellType.EMPTY.value
        for wr, wc in self.config.walls:
            self._grid[wr, wc] = CellType.WALL.value

        # Sample initial-resource cells from non-wall, non-agent cells.
        # Using an exact count rather than per-cell Bernoulli at p — the
        # synthesis's "regrowth distribution at the current p" wording is
        # interpreted as a *count*, see the journal entry. This keeps the
        # initial sparsity controllable and makes the smoke test legible.
        ar, ac = self._agent_pos
        wall_set = set(self.config.walls)
        available: list[tuple[int, int]] = []
        for r in range(self.config.grid_size):
            for c in range(self.config.grid_size):
                if (r, c) == (ar, ac):
                    continue
                if (r, c) in wall_set:
                    continue
                available.append((r, c))

        n_to_place = min(self.config.n_initial_resources, len(available))
        if n_to_place > 0:
            permutation = self._regrowth_rng.permutation(len(available))
            for idx in permutation[:n_to_place]:
                r, c = available[int(idx)]
                self._grid[r, c] = CellType.RESOURCE.value

    def _is_wall(self, r: int, c: int) -> bool:
        return bool(self._grid[r, c] == CellType.WALL.value)

    def _apply_action(self, action: int) -> None:
        dr, dc = _ACTION_DELTAS[action]
        # Stay action is a true no-op for position. Resource consumption is
        # *triggered by entering* (synthesis §Q2), so a stay over a resource
        # — including a resource that just regrew under the agent — does
        # not consume. Returning early here also ensures no consumption
        # logic runs against the agent's own cell.
        if dr == 0 and dc == 0:
            return

        r, c = self._agent_pos
        new_r, new_c = r + dr, c + dc

        # Off-grid: stay in place. Wall collision: stay in place. Either
        # case still advances the clock and runs regrowth/drift via the
        # caller's ordering.
        gs = self.config.grid_size
        if not (0 <= new_r < gs and 0 <= new_c < gs):
            return
        if self._is_wall(new_r, new_c):
            return

        self._agent_pos = (new_r, new_c)
        if self._grid[new_r, new_c] == CellType.RESOURCE.value:
            self._grid[new_r, new_c] = CellType.EMPTY.value

    def _update_drift(self) -> None:
        """Bounded random walk on the regrowth rate parameter ``p``.

        Per-step Gaussian step with standard deviation
        ``drift_magnitude_per_step``. The result is clipped into
        ``[drift_p_min, drift_p_max]`` to prevent the pathological extremes
        the synthesis warns about (e.g. p saturating to 0 or 1 over a
        long run). Clipping at the edge means the walk does not "reflect" —
        it just stops at the boundary until a step pushes it back inside.
        """
        delta = float(
            self._drift_rng.normal(
                loc=0.0, scale=self.config.drift_magnitude_per_step
            )
        )
        clipped = float(
            np.clip(
                self._regrowth_p + delta,
                self.config.drift_p_min,
                self.config.drift_p_max,
            )
        )
        self._regrowth_p = clipped

    def _update_regrowth(self) -> None:
        """Per-cell Bernoulli with the current ``p`` over all empty cells.

        Action-independent and per-cell. Regrowth applies to every empty
        cell regardless of whether the agent is standing on it — the
        synthesis specifies "Each empty cell has a per-step probability p
        of becoming a resource", with no exception for the agent's cell.
        If a resource regrows under the agent, the agent is not "entering"
        it and so does not consume it; the agent must move away and back
        to consume.
        """
        empty_mask = self._grid == CellType.EMPTY.value
        if not bool(empty_mask.any()):
            return
        coin_flips = self._regrowth_rng.random(size=self._grid.shape)
        regrowth_mask = empty_mask & (coin_flips < self._regrowth_p)
        self._grid[regrowth_mask] = CellType.RESOURCE.value

    # ---- observation rendering -------------------------------------------

    def _extract_view(self) -> NDArray[np.uint8]:
        """7×7 ego-centric view of the underlying grid, padded with OOB.

        Padding with ``_OOB_SENTINEL`` ensures every cell in the returned
        view is one of {EMPTY, WALL, RESOURCE, OOB}. The agent's position
        is always at the geometric center of the view; cells beyond the
        grid boundary in the view are rendered as OOB, distinct from
        every in-bounds cell.
        """
        view_radius = self.config.view_size // 2
        padded = np.pad(
            self._grid,
            pad_width=view_radius,
            mode="constant",
            constant_values=_OOB_SENTINEL,
        )
        r, c = self._agent_pos
        pr = r + view_radius
        pc = c + view_radius
        view = padded[
            pr - view_radius : pr + view_radius + 1,
            pc - view_radius : pc + view_radius + 1,
        ]
        # The slice is a view into ``padded``; copy so the caller can hold a
        # standalone array without aliasing the padded buffer.
        return view.copy()

    def _render_observation(self) -> NDArray[np.uint8]:
        """Render the 7×7 view into a 32×32 grayscale ``uint8`` image.

        Cell values 0..3 are mapped through ``_RENDER_TABLE`` to grayscale
        intensities. The 7×7 → 32×32 expansion uses nearest-neighbor
        repetition with cell sizes derived from ``np.linspace(0, 32, 8)``
        so the boundaries of the 7×7 view align exactly with pixel
        boundaries in the 32×32 output (no anti-aliasing, no resampling
        artifacts that the encoder would have to learn around).
        """
        view = self._extract_view()
        rendered_cells = _RENDER_TABLE[view]
        target = self.config.obs_resolution
        source = self.config.view_size
        edges = np.linspace(0, target, source + 1).astype(np.int64)
        repeats = np.diff(edges)
        expanded = np.repeat(
            np.repeat(rendered_cells, repeats, axis=0), repeats, axis=1
        )
        out = expanded.astype(np.uint8, copy=False)
        out.setflags(write=False)
        return out

    def _make_env_step(self) -> EnvStep:
        return EnvStep(
            observation=self._render_observation(),
            env_step=self._env_step,
            episode_id=self._episode_id,
            step_in_episode=self._step_in_episode,
            wallclock_ms=int(time.time() * 1000),
        )


__all__ = [
    "CellType",
    "EnvStep",
    "GridState",
    "GridWorld",
    "GridWorldConfig",
    "NUM_ACTIONS",
]
