"""Probe 4.5 Phase 3 — S-TOY tests: determinism, learning signal recorded,
protocol conformance, and the no-toy-import structural lint (the toy's
reward may exist observer-side and nowhere else)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from kind.env.grid_world import GridWorldConfig
from kind.observer.allocation import AllocationPolicy
from kind.observer.reward_toy import (
    RewardToy,
    RewardToyConfig,
    toy_reward,
)


def _tiny_config(seed: int = 7) -> RewardToyConfig:
    return RewardToyConfig(
        grid_world_config=GridWorldConfig(energy_fault_enabled=True),
        seed=seed,
        train_steps=20_000,
        epsilon_decay_steps=12_000,
    )


def test_reward_is_negative_band_deviation() -> None:
    assert toy_reward(0.6) == 0.0
    assert toy_reward(0.45) == pytest.approx(0.0, abs=1e-12)
    assert toy_reward(0.75) == pytest.approx(0.0, abs=1e-12)
    assert toy_reward(0.30) == pytest.approx(-0.15)
    assert toy_reward(1.0) == pytest.approx(-0.25)
    assert toy_reward(0.0) == pytest.approx(-0.45)


def test_toy_training_deterministic() -> None:
    stats = []
    qs = []
    for _ in range(2):
        toy = RewardToy(_tiny_config())
        stats.append(toy.train())
        qs.append(toy._q.copy())
    assert stats[0] == stats[1]
    assert np.array_equal(qs[0], qs[1])


def test_toy_learns_and_records_convergence_trend() -> None:
    toy = RewardToy(_tiny_config())
    stats = toy.train()
    # The convergence record exists (reported, never a stopping rule)...
    assert stats.train_steps == 20_000
    assert stats.states_visited > 50
    # ...and under the fixed seed the reward trend improves — the toy is
    # engineered to have a foreground, and it measurably acquires one.
    assert stats.mean_reward_final_block > stats.mean_reward_first_block
    assert stats.mean_abs_td_final_block < stats.mean_abs_td_first_block


def test_toy_satisfies_the_allocation_policy_protocol() -> None:
    toy = RewardToy(_tiny_config())
    policy: AllocationPolicy = toy  # structural typing must hold
    from kind.env.grid_world import GridWorld

    world = GridWorld(_tiny_config().grid_world_config, seed=1)
    world.reset()
    view_free_action = toy.greedy_action(world.state)
    assert 0 <= view_free_action < 5
    assert policy is toy


def test_no_io_code_path_imports_the_toy() -> None:
    """Structural lint (plan §S-TOY): reward machinery exists observer-side
    and nowhere else — no module under kind/agents, kind/training, kind/env,
    or kind/window may import it. Positive control: this test file does."""
    kind_root = Path(__file__).resolve().parents[1] / "kind"
    offenders: list[str] = []
    for package in ("agents", "training", "env", "window"):
        for source in (kind_root / package).rglob("*.py"):
            if "reward_toy" in source.read_text(encoding="utf-8"):
                offenders.append(str(source))
    assert not offenders, f"Io code paths import the toy: {offenders}"
    # Positive control: the lint's needle is findable at all.
    assert "reward_toy" in Path(__file__).read_text(encoding="utf-8")


def test_toy_ignores_the_io_view() -> None:
    """The protocol's view argument is Io machinery; the toy must not read
    it (it acts on mirror-side state alone)."""
    from kind.agents.views import PolicyView
    from kind.env.grid_world import GridWorld

    toy = RewardToy(_tiny_config())
    world = GridWorld(_tiny_config().grid_world_config, seed=2)
    world.reset()
    state = world.state
    views = [
        PolicyView(
            h=torch.zeros(1, 16),
            z=torch.zeros(1, 4),
            self_prediction_error=torch.zeros(()),
        ),
        PolicyView(
            h=torch.ones(1, 16) * 100,
            z=torch.ones(1, 4) * -100,
            self_prediction_error=torch.ones(()),
        ),
    ]
    actions = {toy.action(view=v, state=state) for v in views}
    assert len(actions) == 1
