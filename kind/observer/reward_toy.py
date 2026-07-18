"""Probe 4.5 S-TOY — the reward-equipped positive control. NEVER touches Io.

A small, separate, explicitly reward-driven agent (frozen prereg §6):
tabular Q-learning on ``(cell, energy decile)``, per-step reward
``−max(0, |e − setpoint| − halfwidth)``, γ = 0.95, ε-greedy with decay,
trained on the same fault-on physics — engineered to *have* a foreground,
so the §2 fixed-surprise allocation discriminator can be validated against
a system known to allocate by stakes. Reward is **allowed here and only
here**: the toy lives observer-side, imports nothing into any Io code path,
and a structural lint (``tests/test_reward_toy.py``) pins that no
``kind/agents/`` or ``kind/training/`` module imports it.

The toy reads mirror-side ground truth (``GridState.true_energy``, its own
cell) for both its Q-state and its reward — it is scaffolding outside Io's
epistemic constraints, not a model of anything. "Trained to convergence"
is realized as a fixed pre-committed step budget with the TD-error and
reward trends recorded (never a behavior-tuned stopping rule).

Deterministic given ``config.seed``: the env, the ε-greedy draws, and the
argmax tie-break (first index) are all fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from kind.agents.preference import BAND_HALFWIDTH, SETPOINT
from kind.agents.views import PolicyView
from kind.env.grid_world import GridState, GridWorld, GridWorldConfig

__all__ = [
    "RewardToyConfig",
    "RewardToyTrainStats",
    "RewardToy",
    "toy_reward",
]

_NUM_ACTIONS = 5


def toy_reward(true_energy: float) -> float:
    """The §6 reward: negative band deviation, zero anywhere in-band."""
    return -max(0.0, abs(true_energy - SETPOINT) - BAND_HALFWIDTH)


@dataclass(frozen=True)
class RewardToyConfig:
    """The §6 toy protocol. ``grid_world_config`` is the fault-on world the
    real validation uses; tests shrink the budget, runs keep the defaults."""

    grid_world_config: GridWorldConfig
    seed: int
    train_steps: int = 300_000
    gamma: float = 0.95
    learning_rate: float = 0.2
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 200_000
    n_energy_bins: int = 10


@dataclass(frozen=True)
class RewardToyTrainStats:
    """The convergence record (reported, never a stopping rule)."""

    train_steps: int
    mean_abs_td_first_block: float
    mean_abs_td_final_block: float
    mean_reward_first_block: float
    mean_reward_final_block: float
    states_visited: int
    block_size: int


class RewardToy:
    """Tabular Q on ``(cell, energy decile)`` over the 5 grid actions."""

    def __init__(self, config: RewardToyConfig) -> None:
        self.config = config
        size = config.grid_world_config.grid_size
        self._q: NDArray[np.float64] = np.zeros(
            (size * size, config.n_energy_bins, _NUM_ACTIONS)
        )
        self._rng = np.random.default_rng(config.seed)

    def _state_index(self, state: GridState) -> tuple[int, int]:
        size = self.config.grid_world_config.grid_size
        cell = state.agent_pos[0] * size + state.agent_pos[1]
        # Decile of the normalized [0, 1] energy; 1.0 folds into the top bin.
        bins = self.config.n_energy_bins
        decile = min(int(float(state.true_energy) * bins), bins - 1)
        return cell, decile

    def greedy_action(self, state: GridState) -> int:
        cell, decile = self._state_index(state)
        return int(np.argmax(self._q[cell, decile]))  # first-index tie-break

    def action(self, *, view: PolicyView, state: GridState) -> int:
        """The ``AllocationPolicy`` protocol (the harness's one interface).
        The view is Io machinery; the toy ignores it."""
        del view
        return self.greedy_action(state)

    def train(self) -> RewardToyTrainStats:
        """Q-learning on the configured world for the fixed budget."""
        c = self.config
        world = GridWorld(c.grid_world_config, seed=c.seed)
        world.reset()
        state = world.state
        cell, decile = self._state_index(state)

        block = max(1, c.train_steps // 10)
        td_first: list[float] = []
        td_final: list[float] = []
        r_first: list[float] = []
        r_final: list[float] = []
        visited: set[tuple[int, int]] = set()

        for step in range(c.train_steps):
            frac = min(1.0, step / max(1, c.epsilon_decay_steps))
            epsilon = c.epsilon_start + frac * (c.epsilon_end - c.epsilon_start)
            if float(self._rng.random()) < epsilon:
                action = int(self._rng.integers(0, _NUM_ACTIONS))
            else:
                action = int(np.argmax(self._q[cell, decile]))

            world.step(action)
            next_state = world.state
            next_cell, next_decile = self._state_index(next_state)
            reward = toy_reward(float(next_state.true_energy))

            target = reward + c.gamma * float(
                np.max(self._q[next_cell, next_decile])
            )
            td = target - float(self._q[cell, decile, action])
            self._q[cell, decile, action] += c.learning_rate * td

            visited.add((cell, decile))
            if step < block:
                td_first.append(abs(td))
                r_first.append(reward)
            elif step >= c.train_steps - block:
                td_final.append(abs(td))
                r_final.append(reward)
            cell, decile = next_cell, next_decile

        return RewardToyTrainStats(
            train_steps=c.train_steps,
            mean_abs_td_first_block=float(np.mean(td_first)),
            mean_abs_td_final_block=float(np.mean(td_final)),
            mean_reward_first_block=float(np.mean(r_first)),
            mean_reward_final_block=float(np.mean(r_final)),
            states_visited=len(visited),
            block_size=block,
        )
