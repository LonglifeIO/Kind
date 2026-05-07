"""Phase 2a gate test for ``kind/env/grid_world.py``.

Plan §4 test #1: env step shape and basic mechanics. Specifically the
``reset()`` / ``step()`` shape contract, episode reset at 200 steps,
deterministic replay from a single seed, action validity, wall collision,
resource consumption, regrowth dynamics under high-p, drift dynamics under
amplified magnitude, and independence of the two RNG streams.

Smaller unit tests cover OOB rendering distinguishability, the ``state``
property's full surface, and wallclock-ms monotonicity.

CPU only; no PyTorch is imported by the env, so these tests run fast and
deterministic without device concerns.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from kind.env.grid_world import (
    NUM_ACTIONS,
    CellType,
    EnvStep,
    GridState,
    GridWorld,
    GridWorldConfig,
)


# ---- shared fixtures -------------------------------------------------------


def default_config() -> GridWorldConfig:
    """The Probe 1 production defaults — used by the gate test."""
    return GridWorldConfig()


def static_p_config(p: float, n_initial_resources: int = 0) -> GridWorldConfig:
    """Config with drift bound to a single value of ``p`` and amplified or
    zeroed dynamics suitable for behavioral isolation in tests."""
    return GridWorldConfig(
        initial_regrowth_p=p,
        drift_p_min=p,
        drift_p_max=p,
        drift_magnitude_per_step=0.0,
        n_initial_resources=n_initial_resources,
    )


# ---- gate test (plan §4 test #1) ------------------------------------------


def test_gate_env_step_shape_and_basic_mechanics() -> None:
    """The plan's named gate: ``reset()`` and ``step()`` shape contract.

    Verifies the observation dimensions, dtype, metadata fields, monotonic
    counters, and that consecutive steps populate distinct ``EnvStep``
    records that pass downstream consumers' shape expectations.
    """
    config = default_config()
    env = GridWorld(config, seed=42)

    # reset() — initial step is at env_step=0, episode_id=0, step_in_episode=0.
    step_0 = env.reset()
    assert isinstance(step_0, EnvStep)
    assert step_0.observation.shape == (config.obs_resolution, config.obs_resolution)
    assert step_0.observation.dtype == np.uint8
    assert step_0.env_step == 0
    assert step_0.episode_id == 0
    assert step_0.step_in_episode == 0
    assert step_0.wallclock_ms > 0

    # The observation is read-only; downstream consumers cannot accidentally
    # mutate it and have the mutation bleed into the next render.
    assert not step_0.observation.flags.writeable

    # step() — counters advance by 1, shape and dtype unchanged.
    step_1 = env.step(0)
    assert step_1.observation.shape == (config.obs_resolution, config.obs_resolution)
    assert step_1.observation.dtype == np.uint8
    assert step_1.env_step == 1
    assert step_1.episode_id == 0
    assert step_1.step_in_episode == 1


# ---- episode reset semantics ----------------------------------------------


def test_episode_resets_after_episode_length_steps() -> None:
    """200 step() calls after reset() crosses the episode boundary cleanly.

    The 200th step() returns ``episode_id=1, step_in_episode=0`` with the
    fresh-world observation. There is no terminal signal in the
    observation; the only difference from step #199 is that the world has
    been re-sampled.
    """
    config = default_config()
    env = GridWorld(config, seed=42)
    env.reset()

    # Step 199 times — all stay within episode 0.
    for i in range(config.episode_length - 1):
        result = env.step(4)  # stay
        assert result.episode_id == 0
        assert result.step_in_episode == i + 1
        assert result.env_step == i + 1

    # The 200th step crosses the boundary.
    boundary = env.step(4)
    assert boundary.episode_id == 1
    assert boundary.step_in_episode == 0
    assert boundary.env_step == config.episode_length  # global step is continuous


def test_no_terminal_signal_in_observation_at_episode_boundary() -> None:
    """The episode-boundary observation has the same shape and value range as
    any other observation — no flag, no special marker, no NaN.
    """
    config = GridWorldConfig(episode_length=5)
    env = GridWorld(config, seed=42)
    env.reset()
    for _ in range(config.episode_length - 1):
        env.step(4)
    boundary = env.step(4)
    assert boundary.observation.shape == (config.obs_resolution, config.obs_resolution)
    assert boundary.observation.dtype == np.uint8
    # All observation values must come from the rendering table {128, 0, 255, 64}.
    valid_values = {128, 0, 255, 64}
    assert set(np.unique(boundary.observation).tolist()).issubset(valid_values)


def test_drift_carries_across_episode_boundaries() -> None:
    """``p`` continues drifting across episodes; only ``reset()`` resets it."""
    config = GridWorldConfig(
        episode_length=10,
        drift_magnitude_per_step=0.001,  # large enough to drift visibly
    )
    env = GridWorld(config, seed=42)
    env.reset()
    # Run through one full episode boundary; capture p before/after.
    for _ in range(config.episode_length):
        env.step(4)
    p_after_first_episode = env.state.regrowth_p
    # If drift were reset between episodes, p would be the initial value.
    # It almost certainly isn't, after 10 random steps with non-zero drift.
    assert p_after_first_episode != config.initial_regrowth_p


# ---- determinism ----------------------------------------------------------


def test_deterministic_from_single_integer_seed() -> None:
    """Two GridWorlds with the same seed and action sequence produce
    identical observations and identical state trajectories."""
    config = default_config()
    env_a = GridWorld(config, seed=42)
    env_b = GridWorld(config, seed=42)

    s_a0 = env_a.reset()
    s_b0 = env_b.reset()
    assert np.array_equal(s_a0.observation, s_b0.observation)
    assert env_a.state.agent_pos == env_b.state.agent_pos
    assert env_a.state.regrowth_p == env_b.state.regrowth_p
    assert np.array_equal(env_a.state.grid, env_b.state.grid)

    actions = [0, 1, 2, 3, 4] * 50
    for action in actions:
        s_a = env_a.step(action)
        s_b = env_b.step(action)
        assert np.array_equal(s_a.observation, s_b.observation)
        assert s_a.env_step == s_b.env_step
        assert s_a.episode_id == s_b.episode_id
        assert s_a.step_in_episode == s_b.step_in_episode
        # wallclock_ms is real time; the two envs are running in slightly
        # different milliseconds, so we don't compare it.
        assert env_a.state.regrowth_p == env_b.state.regrowth_p
        assert np.array_equal(env_a.state.grid, env_b.state.grid)


def test_different_seeds_produce_different_trajectories() -> None:
    """Sanity check on the seed: different seeds → different worlds."""
    config = default_config()
    env_a = GridWorld(config, seed=42)
    env_b = GridWorld(config, seed=43)
    env_a.reset()
    env_b.reset()
    # Initial resource placements are drawn from regrowth; different seeds
    # should put resources in different cells.
    assert not np.array_equal(env_a.state.grid, env_b.state.grid)


# ---- action mechanics ----------------------------------------------------


def test_action_validity_in_range() -> None:
    env = GridWorld(default_config(), seed=42)
    env.reset()
    # All valid actions are accepted without raising.
    for a in range(NUM_ACTIONS):
        env.step(a)


def test_action_validity_out_of_range_raises() -> None:
    env = GridWorld(default_config(), seed=42)
    env.reset()
    with pytest.raises(ValueError, match="action must be in"):
        env.step(NUM_ACTIONS)
    with pytest.raises(ValueError, match="action must be in"):
        env.step(-1)


def test_step_before_reset_raises() -> None:
    env = GridWorld(default_config(), seed=42)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)


def test_each_action_moves_in_correct_direction() -> None:
    """0=up, 1=down, 2=left, 3=right, 4=stay."""
    config = static_p_config(p=0.005, n_initial_resources=0)
    config = replace(config, start_cell=(4, 4))
    env = GridWorld(config, seed=42)
    env.reset()
    assert env.state.agent_pos == (4, 4)

    env.step(0)  # up
    assert env.state.agent_pos == (3, 4)
    env.step(1)  # down
    assert env.state.agent_pos == (4, 4)
    env.step(2)  # left
    assert env.state.agent_pos == (4, 3)
    env.step(3)  # right
    assert env.state.agent_pos == (4, 4)
    env.step(4)  # stay
    assert env.state.agent_pos == (4, 4)


def test_off_grid_move_leaves_position_unchanged_but_advances_clock() -> None:
    """Walking into the boundary is a no-op for position but the clock ticks
    and the stochastic processes still advance."""
    config = static_p_config(p=0.005, n_initial_resources=0)
    config = replace(config, start_cell=(0, 0))
    env = GridWorld(config, seed=42)
    env.reset()
    pos_before = env.state.agent_pos

    step = env.step(0)  # up — would leave the grid
    assert env.state.agent_pos == pos_before
    assert step.env_step == 1
    assert step.step_in_episode == 1


def test_wall_collision_blocks_movement() -> None:
    """Moving into a wall cell is a no-op for position; clock ticks anyway."""
    config = GridWorldConfig(
        walls=((3, 4),),
        start_cell=(3, 3),
        n_initial_resources=0,
        drift_magnitude_per_step=0.0,
    )
    env = GridWorld(config, seed=42)
    env.reset()
    assert env.state.agent_pos == (3, 3)

    step = env.step(3)  # right — into the wall at (3, 4)
    assert env.state.agent_pos == (3, 3)
    assert step.env_step == 1


def test_resource_consumption_clears_cell() -> None:
    """Entering a resource cell empties it (consumption-on-entry).

    Setup uses ``n_initial_resources=0`` so the test can place exactly one
    resource at a known location via the private grid attribute. This is
    not part of the public API but is a clean way to isolate the behavior.
    """
    config = static_p_config(p=0.005, n_initial_resources=0)
    config = replace(config, start_cell=(3, 3))
    env = GridWorld(config, seed=42)
    env.reset()
    # Place a resource directly to the right of the agent.
    env._grid[3, 4] = CellType.RESOURCE.value

    env.step(3)  # right
    assert env.state.agent_pos == (3, 4)
    assert env.state.grid[3, 4] == CellType.EMPTY.value


def test_stay_on_resource_does_not_consume() -> None:
    """Resource consumption is *triggered by entering*. Standing on a
    resource (e.g. one that regrew under the agent) does not consume."""
    config = static_p_config(p=0.005, n_initial_resources=0)
    config = replace(config, start_cell=(3, 3))
    env = GridWorld(config, seed=42)
    env.reset()
    env._grid[3, 3] = CellType.RESOURCE.value  # under the agent

    env.step(4)  # stay
    assert env.state.grid[3, 3] == CellType.RESOURCE.value


# ---- regrowth dynamics ---------------------------------------------------


def test_no_regrowth_when_p_is_zero() -> None:
    """With ``p=0``, no regrowth events occur over many steps."""
    config = GridWorldConfig(
        initial_regrowth_p=0.0,
        drift_p_min=0.0,
        drift_p_max=0.0,
        drift_magnitude_per_step=0.0,
        n_initial_resources=0,
    )
    env = GridWorld(config, seed=42)
    env.reset()

    # Empty grid initially; no regrowth → grid stays empty.
    for _ in range(200):
        env.step(4)  # stay; no consumption either
    n_resources = int((env.state.grid == CellType.RESOURCE.value).sum())
    assert n_resources == 0


def test_high_p_produces_many_regrowth_events() -> None:
    """With ``p`` set high (0.5), many empty cells regrow within a single step."""
    config = GridWorldConfig(
        initial_regrowth_p=0.5,
        drift_p_min=0.5,
        drift_p_max=0.5,
        drift_magnitude_per_step=0.0,
        n_initial_resources=0,
    )
    env = GridWorld(config, seed=42)
    env.reset()
    n_resources_before = int((env.state.grid == CellType.RESOURCE.value).sum())
    assert n_resources_before == 0

    env.step(4)  # stay
    n_resources_after = int((env.state.grid == CellType.RESOURCE.value).sum())
    # On an 8×8 grid with 64 cells and p=0.5, expectation is 32 regrowth events.
    # Stochastic, but should be far more than zero.
    assert n_resources_after > 10


# ---- drift dynamics ------------------------------------------------------


def test_drift_changes_p_across_many_steps() -> None:
    """With non-zero drift magnitude, ``p`` changes over time (not constant)
    yet stays bounded inside ``[drift_p_min, drift_p_max]``."""
    config = GridWorldConfig(
        initial_regrowth_p=0.025,  # in the middle of the bounds
        drift_p_min=0.001,
        drift_p_max=0.05,
        drift_magnitude_per_step=0.001,  # amplified for visibility
        n_initial_resources=0,
    )
    env = GridWorld(config, seed=42)
    env.reset()

    p_values: list[float] = [env.state.regrowth_p]
    for _ in range(500):
        env.step(4)
        p_values.append(env.state.regrowth_p)

    p_array = np.array(p_values)
    # p must change at least sometimes.
    assert (np.diff(p_array) != 0).any()
    # p must stay inside the bounds at every step.
    assert (p_array >= config.drift_p_min).all()
    assert (p_array <= config.drift_p_max).all()


def test_drift_clamps_at_bounds() -> None:
    """A very large drift magnitude saturates against the bounds; ``p``
    never escapes ``[drift_p_min, drift_p_max]``."""
    config = GridWorldConfig(
        initial_regrowth_p=0.025,
        drift_p_min=0.02,
        drift_p_max=0.03,
        drift_magnitude_per_step=0.5,  # huge — far larger than the bound width
        n_initial_resources=0,
    )
    env = GridWorld(config, seed=42)
    env.reset()
    for _ in range(200):
        env.step(4)
        p = env.state.regrowth_p
        assert config.drift_p_min <= p <= config.drift_p_max


# ---- two RNG streams independence ----------------------------------------


def test_two_rng_streams_are_separate_generator_instances() -> None:
    """Structural test: the env holds two distinct ``Generator`` instances
    derived from a single ``SeedSequence`` via ``spawn(2)``."""
    env = GridWorld(default_config(), seed=42)
    assert env._regrowth_rng is not env._drift_rng
    a = env._regrowth_rng.random()
    b = env._drift_rng.random()
    # Two independently-spawned streams almost surely produce different values.
    assert a != b


def test_changing_regrowth_seed_does_not_change_drift_trajectory() -> None:
    """Behavioural test: replacing the regrowth stream after construction
    leaves the drift trajectory unchanged.

    Setup: build two envs with seed=42 (so both spawn the same drift
    stream). Replace one env's regrowth stream with one derived from a
    different seed. The drift stream — and hence the ``p`` trajectory —
    must remain identical, because the drift stream is wholly separate.
    """
    config = GridWorldConfig(
        initial_regrowth_p=0.025,
        drift_p_min=0.001,
        drift_p_max=0.05,
        drift_magnitude_per_step=0.001,
        n_initial_resources=4,
    )
    env_a = GridWorld(config, seed=42)
    env_b = GridWorld(config, seed=42)
    env_b._regrowth_rng = np.random.default_rng(np.random.SeedSequence(999))

    env_a.reset()
    env_b.reset()

    drifts_a: list[float] = []
    drifts_b: list[float] = []
    for _ in range(50):
        env_a.step(4)
        env_b.step(4)
        drifts_a.append(env_a.state.regrowth_p)
        drifts_b.append(env_b.state.regrowth_p)

    assert drifts_a == drifts_b
    # And the regrowth events differ — the grids should not be identical
    # after enough regrowth coin flips.
    assert not np.array_equal(env_a.state.grid, env_b.state.grid)


def test_changing_drift_seed_does_not_change_regrowth_at_fixed_p() -> None:
    """The reverse: replacing the drift stream after construction leaves the
    regrowth coin-flip trajectory unchanged when ``p`` is held constant.

    Setup: lock ``p`` by setting ``drift_p_min == drift_p_max`` and
    ``drift_magnitude_per_step=0`` so the drift stream's draws never change
    ``p``. Replace one env's drift stream; the grid trajectories must
    match exactly because both envs are flipping the same regrowth coins
    at the same ``p``.
    """
    config = static_p_config(p=0.05, n_initial_resources=0)
    env_a = GridWorld(config, seed=42)
    env_b = GridWorld(config, seed=42)
    env_b._drift_rng = np.random.default_rng(np.random.SeedSequence(999))

    env_a.reset()
    env_b.reset()

    for _ in range(30):
        env_a.step(4)
        env_b.step(4)

    assert np.array_equal(env_a.state.grid, env_b.state.grid)


# ---- observation rendering -----------------------------------------------


def test_oob_cells_are_distinguishable_from_in_bounds_cells() -> None:
    """An agent at the corner sees out-of-bounds cells in the partial view;
    these must render with a value distinct from every in-bounds cell type.
    """
    config = GridWorldConfig(start_cell=(0, 0), n_initial_resources=0)
    env = GridWorld(config, seed=42)
    step = env.reset()

    obs = step.observation
    # The four legal pixel values are: empty=128, wall=0, resource=255, OOB=64.
    valid_values = {0, 64, 128, 255}
    assert set(np.unique(obs).tolist()).issubset(valid_values)

    # At (0, 0), the upper-left of the 7×7 view extends 3 cells past the
    # grid boundary. Those cells render as OOB (64) — distinct from empty
    # (128) and from wall (0) and from resource (255).
    # Top-left pixel of the rendered observation maps to the top-left view cell.
    assert obs[0, 0] == 64

    # The center cell of the view is the agent's cell, which here is empty.
    cy = obs.shape[0] // 2
    cx = obs.shape[1] // 2
    assert obs[cy, cx] == 128


def test_observation_has_only_legal_render_values() -> None:
    """Every pixel in every observation comes from the four-element render
    table. No anti-aliasing, no interpolation artifacts."""
    env = GridWorld(default_config(), seed=42)
    step = env.reset()
    valid = {0, 64, 128, 255}
    assert set(np.unique(step.observation).tolist()).issubset(valid)
    for _ in range(50):
        s = env.step(4)
        assert set(np.unique(s.observation).tolist()).issubset(valid)


# ---- state property -----------------------------------------------------


def test_state_exposes_full_grid_and_agent_position_and_p() -> None:
    """The mirror-side ``state`` property carries the underlying grid,
    the agent's position, and the current drift parameter ``p``."""
    env = GridWorld(default_config(), seed=42)
    env.reset()

    state = env.state
    assert isinstance(state, GridState)
    assert state.grid.shape == (8, 8)
    assert state.grid.dtype == np.uint8
    # The grid is read-only; downstream consumers cannot mutate it.
    assert not state.grid.flags.writeable

    assert isinstance(state.agent_pos, tuple)
    assert len(state.agent_pos) == 2
    assert all(isinstance(x, int) for x in state.agent_pos)

    assert isinstance(state.regrowth_p, float)
    assert state.regrowth_p > 0


def test_state_grid_only_contains_known_cell_types() -> None:
    """The underlying grid carries only EMPTY/WALL/RESOURCE values —
    never the OOB sentinel, which only appears in the rendered view."""
    env = GridWorld(default_config(), seed=42)
    env.reset()
    legal = {CellType.EMPTY.value, CellType.WALL.value, CellType.RESOURCE.value}
    assert set(np.unique(env.state.grid).tolist()).issubset(legal)


# ---- wallclock metadata --------------------------------------------------


def test_wallclock_ms_is_monotonically_non_decreasing() -> None:
    """Wallclock_ms must never go backwards across consecutive records."""
    env = GridWorld(default_config(), seed=42)
    step = env.reset()
    prev = step.wallclock_ms
    for _ in range(50):
        s = env.step(4)
        assert s.wallclock_ms >= prev
        prev = s.wallclock_ms


# ---- config validation --------------------------------------------------


def test_config_rejects_out_of_bounds_walls() -> None:
    with pytest.raises(ValueError, match="out of grid bounds"):
        GridWorld(GridWorldConfig(walls=((10, 10),)), seed=42)


def test_config_rejects_out_of_bounds_start_cell() -> None:
    with pytest.raises(ValueError, match="out of grid bounds"):
        GridWorld(GridWorldConfig(start_cell=(99, 0)), seed=42)


def test_config_rejects_initial_p_outside_drift_bounds() -> None:
    with pytest.raises(ValueError, match="initial_regrowth_p"):
        GridWorld(
            GridWorldConfig(
                initial_regrowth_p=0.99, drift_p_min=0.0, drift_p_max=0.5
            ),
            seed=42,
        )


def test_config_rejects_too_many_initial_resources() -> None:
    with pytest.raises(ValueError, match="exceeds available"):
        GridWorld(
            GridWorldConfig(grid_size=4, n_initial_resources=100),
            seed=42,
        )


def test_config_rejects_even_view_size() -> None:
    with pytest.raises(ValueError, match="odd"):
        GridWorld(GridWorldConfig(view_size=8), seed=42)
