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
from collections import deque
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

    Probe 2 env revision (synthesis §2.1; implementation plan §2.2):
    ``start_cell`` defaults to ``None``, which selects a random non-wall
    in-bounds cell at every episode reset, drawn from the regrowth RNG
    stream. The previous Probe 1 default ``(3, 3)`` is still accepted as
    an explicit override; the post-Probe-1 journal entry documented the
    fixed-start as the cause of late-episode trajectory collapse onto a
    degenerate two-cell loop, and §2.1 of the synthesis settled the
    revision. Reproducibility commitment: given the seed, every
    episode's sampled start cell can be recovered from the
    ``env_reset`` event in the ``world_event`` stream — see
    :meth:`kind.env.env_server.EnvServer._emit_env_reset`.
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
    start_cell: tuple[int, int] | None = None
    walls: tuple[tuple[int, int], ...] = ()

    # Probe 3.5 energy economy (implementation plan §S-ENV; pre-registration
    # 2026-06-10 §"Energy units"). Energy is **Io's own homeostatic variable**,
    # distinct from the world: it depletes per step (a base decay plus an
    # action-magnitude cost — movement costs more than ``stay``) and replenishes
    # on resource *entry* (reusing the existing entry-triggered consumption). It
    # floors at ``energy_norm_min`` and caps at ``energy_norm_max`` and **never
    # terminates the episode** — there is no death, no absorbing state. Energy
    # *carries across* the soft 200-step episode boundary (it is Io's internal
    # state, not the world's; resampling it would make the boundary a stealth
    # survival refill). Only ``reset()`` re-initialises it.
    #
    # Raw energy lives in ``[energy_norm_min, energy_norm_max]``; ``true_energy``
    # and ``sensed_energy`` are the **normalized** value in ``[0, 1]`` via these
    # fixed config constants (not data-dependent rescaling). The setpoint 0.6 of
    # the frozen pre-registration is in this normalized 0–1 space; ``energy_init``
    # defaults to ``0.6 * energy_norm_max`` so the run starts at setpoint.
    #
    # ``sensed_energy`` is the **coarse, noisy, lagged** scalar Io observes (T2:
    # "like hunger, not introspection" — noise forces inference, protecting
    # opacity): the normalized true value ``energy_obs_lag`` steps ago, plus
    # additive Gaussian noise (σ = ``energy_obs_noise_sigma``), clipped to [0, 1]
    # and lightly quantized to ``energy_obs_quantization_levels`` levels. A third
    # RNG stream (spawned from the env ``SeedSequence``) drives the sensing noise
    # so determinism is preserved. ``true_energy`` (the un-noised normalized
    # value) goes only to ``GridState`` — mirror/telemetry ground truth — and
    # **never enters any training loss** (plan S-ENV rule; the WM reconstruction
    # trains on ``sensed_energy``).
    #
    # Defaults are build-time fixed structural choices (journaled, not swept);
    # only σ / lag are swept (Phase 4). The decay/replenish balance is chosen so
    # energy *varies* over an episode (a dead, floored channel would be
    # unlearnable) while staying non-terminal.
    energy_norm_min: float = 0.0
    energy_norm_max: float = 10.0
    energy_init: float = 6.0
    energy_base_decay: float = 0.08
    energy_move_cost: float = 0.04
    energy_replenish_per_resource: float = 0.8
    energy_obs_noise_sigma: float = 0.05
    energy_obs_lag: int = 1
    energy_obs_quantization_levels: int = 16


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
    # Probe 3.5: the coarse, noisy, lagged, normalized energy scalar Io
    # observes (in [0, 1]). This is the *sensed* value — never the true one.
    sensed_energy: float = 0.0


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
    # Probe 3.5: the normalized true energy in [0, 1] — mirror/telemetry
    # ground truth only. Io never sees this; it never enters any training
    # loss (eval probes use it, but as targets, never as training inputs).
    true_energy: float = 0.0


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

        # Three independent streams from a single integer seed. The third
        # (sensing) stream is the Probe 3.5 addition — spawned from the same
        # ``SeedSequence`` so two ``GridWorld`` instances with the same seed and
        # action sequence produce identical ``sensed_energy`` trajectories.
        seed_sequence = np.random.SeedSequence(seed)
        regrowth_seq, drift_seq, sensing_seq = seed_sequence.spawn(3)
        self._regrowth_rng: np.random.Generator = np.random.default_rng(
            regrowth_seq
        )
        self._drift_rng: np.random.Generator = np.random.default_rng(drift_seq)
        self._sensing_rng: np.random.Generator = np.random.default_rng(
            sensing_seq
        )

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

        # Probe 3.5 energy state (raw, in [energy_norm_min, energy_norm_max]).
        # ``_norm_history`` buffers recent *normalized* true-energy values so the
        # sensed scalar can read the value ``energy_obs_lag`` steps in the past.
        self._energy: float = config.energy_init
        self._norm_history: deque[float] = deque(maxlen=config.energy_obs_lag + 1)

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
        # Energy is re-initialised here — this is the *only* path that resets
        # it. The soft 200-step episode boundary (in ``step``) carries it.
        # The sensing history is cleared; ``_compute_sensed_energy`` (called by
        # ``_make_env_step`` below) seeds it with the first normalized value.
        self._energy = self.config.energy_init
        self._norm_history.clear()
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

        consumed_resource = self._apply_action(action)
        self._update_energy(action, consumed_resource)
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
            true_energy=self._normalize_energy(self._energy),
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

        # Probe 3.5 energy economy validation.
        if not c.energy_norm_min < c.energy_norm_max:
            raise ValueError(
                "must satisfy energy_norm_min < energy_norm_max; got "
                f"({c.energy_norm_min}, {c.energy_norm_max})"
            )
        if not c.energy_norm_min <= c.energy_init <= c.energy_norm_max:
            raise ValueError(
                f"energy_init={c.energy_init} must be in "
                f"[{c.energy_norm_min}, {c.energy_norm_max}]"
            )
        if c.energy_base_decay < 0.0:
            raise ValueError(
                f"energy_base_decay must be non-negative, got {c.energy_base_decay}"
            )
        if c.energy_move_cost < 0.0:
            raise ValueError(
                f"energy_move_cost must be non-negative, got {c.energy_move_cost}"
            )
        if c.energy_replenish_per_resource < 0.0:
            raise ValueError(
                "energy_replenish_per_resource must be non-negative, got "
                f"{c.energy_replenish_per_resource}"
            )
        if c.energy_obs_noise_sigma < 0.0:
            raise ValueError(
                "energy_obs_noise_sigma must be non-negative, got "
                f"{c.energy_obs_noise_sigma}"
            )
        if c.energy_obs_lag < 0:
            raise ValueError(
                f"energy_obs_lag must be non-negative, got {c.energy_obs_lag}"
            )
        if c.energy_obs_quantization_levels < 1:
            raise ValueError(
                "energy_obs_quantization_levels must be >= 1, got "
                f"{c.energy_obs_quantization_levels}"
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
        wall_set = set(self.config.walls)

        # Place agent. Random start cell draws from the regrowth stream so
        # there is no third RNG to seed — spatial events on cells share one
        # source of randomness. Wall rejection consults ``wall_set`` from
        # config rather than the (possibly stale or freshly-zeroed) grid
        # buffer, so the first reset with non-empty walls also rejects
        # them correctly.
        if self.config.start_cell is None:
            r = int(self._regrowth_rng.integers(0, self.config.grid_size))
            c = int(self._regrowth_rng.integers(0, self.config.grid_size))
            while (r, c) in wall_set:
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

    def _apply_action(self, action: int) -> bool:
        """Apply the action; return whether a resource was *entered* this step.

        The return value drives Probe 3.5 energy replenishment — replenishment
        is triggered by *entering* a resource cell (the same event that flips
        RESOURCE→EMPTY), so a ``stay`` over a resource (or a resource that
        regrew under a stationary agent) does not replenish.
        """
        dr, dc = _ACTION_DELTAS[action]
        # Stay action is a true no-op for position. Resource consumption is
        # *triggered by entering* (synthesis §Q2), so a stay over a resource
        # — including a resource that just regrew under the agent — does
        # not consume. Returning early here also ensures no consumption
        # logic runs against the agent's own cell.
        if dr == 0 and dc == 0:
            return False

        r, c = self._agent_pos
        new_r, new_c = r + dr, c + dc

        # Off-grid: stay in place. Wall collision: stay in place. Either
        # case still advances the clock and runs regrowth/drift via the
        # caller's ordering.
        gs = self.config.grid_size
        if not (0 <= new_r < gs and 0 <= new_c < gs):
            return False
        if self._is_wall(new_r, new_c):
            return False

        self._agent_pos = (new_r, new_c)
        if self._grid[new_r, new_c] == CellType.RESOURCE.value:
            self._grid[new_r, new_c] = CellType.EMPTY.value
            return True
        return False

    def _update_energy(self, action: int, consumed_resource: bool) -> None:
        """Advance Io's homeostatic energy for one step (Probe 3.5).

        Depletion = per-step base decay + an action-magnitude cost (movement,
        actions 0–3, costs more than ``stay``, action 4). Replenishment is
        added on resource *entry* (``consumed_resource``). The result is
        clamped to ``[energy_norm_min, energy_norm_max]`` — it floors at the
        minimum and **never terminates the episode**. There is no death and no
        absorbing state; depletion just bottoms out and the env keeps running.
        """
        delta = -self.config.energy_base_decay
        is_move = _ACTION_DELTAS[action] != (0, 0)
        if is_move:
            delta -= self.config.energy_move_cost
        if consumed_resource:
            delta += self.config.energy_replenish_per_resource
        self._energy = float(
            np.clip(
                self._energy + delta,
                self.config.energy_norm_min,
                self.config.energy_norm_max,
            )
        )

    def _normalize_energy(self, raw: float) -> float:
        """Map raw energy → normalized [0, 1] via the fixed config constants.

        Not data-dependent rescaling — ``energy_norm_min`` / ``energy_norm_max``
        are fixed structural choices, so ``true_energy`` and ``sensed_energy``
        share one stable scale across runs and across the sweep.
        """
        span = self.config.energy_norm_max - self.config.energy_norm_min
        if span <= 0.0:
            return 0.0
        return float(
            np.clip((raw - self.config.energy_norm_min) / span, 0.0, 1.0)
        )

    def _quantize_unit(self, value: float) -> float:
        """Light quantization of a [0, 1] value to a fixed number of levels.

        Coarse interoception (T2): the sensed scalar is not a high-resolution
        readout. ``energy_obs_quantization_levels <= 1`` disables quantization.
        """
        levels = self.config.energy_obs_quantization_levels
        if levels <= 1:
            return value
        step = round(value * (levels - 1))
        return float(step) / float(levels - 1)

    def _compute_sensed_energy(self) -> float:
        """Produce the coarse, noisy, lagged sensed scalar for emission.

        Appends the current normalized true energy to the lag buffer, reads the
        value ``energy_obs_lag`` steps in the past (the oldest in the
        ``maxlen = lag + 1`` deque; during the lag-length warm-up it is the
        earliest available), adds one Gaussian noise draw from the dedicated
        sensing RNG, clips to [0, 1], and quantizes. Called exactly once per
        emitted ``EnvStep`` (one noise draw per env step → determinism).
        """
        norm = self._normalize_energy(self._energy)
        self._norm_history.append(norm)
        lagged = self._norm_history[0]
        sigma = self.config.energy_obs_noise_sigma
        noise = (
            float(self._sensing_rng.normal(loc=0.0, scale=sigma))
            if sigma > 0.0
            else 0.0
        )
        noisy = float(np.clip(lagged + noise, 0.0, 1.0))
        return self._quantize_unit(noisy)

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
            sensed_energy=self._compute_sensed_energy(),
        )


__all__ = [
    "CellType",
    "EnvStep",
    "GridState",
    "GridWorld",
    "GridWorldConfig",
    "NUM_ACTIONS",
]
