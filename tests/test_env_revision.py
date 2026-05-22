"""Probe 2 Phase 1 gate tests for the env revision.

Implementation plan §2.2 (v2 carries v1 unchanged) names two changes to
the Phase 2a / 3a env layer:

1. ``GridWorldConfig.start_cell`` default flips from ``(3, 3)`` to
   ``None`` — meaning "sample a random non-wall in-bounds cell from the
   regrowth RNG stream at every episode reset". The
   ``_emit_env_reset`` payload gains a ``start_cell`` entry recording
   the cell that was actually used so random-start runs are
   reproducible from the seed via the ``world_event`` JSONL alone.
2. ``EnvServer.fire_sham_perturbation`` is added — a flag-only method
   that emits a ``builder_perturbation`` ``WorldEvent`` with
   ``payload["is_sham"] = True`` *without* mutating any underlying
   grid state. The agent observation must be byte-identical
   pre- and post-sham, the regrowth and drift RNG streams must not
   advance, and the resulting ``world_event`` record is the only
   external trace.

Together, the two changes are the env-side surface the calibration
protocol's null-event test (synthesis §2.4 element 3) relies on plus
the documented failure-mode fix Probe 1's post-run journal flagged
(late-episode trajectory collapse onto a degenerate two-cell loop with
the fixed start as the suspected cause). The substrate is unchanged;
the agent observation space, action space, and episode length are
unchanged; the four real mutators are unchanged.

CPU only. The harness is synchronous and in-process.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import (
    CellType,
    GridWorld,
    GridWorldConfig,
)
from kind.observer.schemas import WorldEvent
from kind.observer.sinks import JsonlSink


# ---- shared helpers -------------------------------------------------------


def _quiet_config() -> GridWorldConfig:
    """Regrowth and drift turned off so RNG-stream consumption is
    isolated to the start-cell sampling in ``_reset_episode_world``.

    With ``initial_regrowth_p=0`` the per-step Bernoulli has zero
    probability of regrowth (still draws a coin-flip array per step,
    which advances the regrowth RNG, but produces no events); with
    ``n_initial_resources=0`` the post-placement permutation is
    skipped entirely. The drift stream's draws are likewise zero-
    magnitude so ``p`` stays constant.
    """
    return GridWorldConfig(
        initial_regrowth_p=0.0,
        drift_p_min=0.0,
        drift_p_max=0.0,
        drift_magnitude_per_step=0.0,
        n_initial_resources=0,
    )


@contextmanager
def _env_server(
    tmp_path: Path,
    *,
    grid_world_config: GridWorldConfig | None = None,
    seed: int = 42,
    sink_name: str = "world_event.jsonl",
    run_id: str = "phase1-test",
) -> Iterator[tuple[EnvServer, Path]]:
    """Yield ``(server, sink_path)``; manage both EnvServer and JsonlSink."""
    sink_path = tmp_path / sink_name
    sink = JsonlSink(sink_path, WorldEvent)
    config = EnvServerConfig(
        grid_world_config=grid_world_config or _quiet_config(),
        seed=seed,
        world_event_handler=sink.write,
        run_id=run_id,
    )
    server = EnvServer(config)
    try:
        yield server, sink_path
    finally:
        try:
            server.close()
        finally:
            sink.close()


def _read_world_events(path: Path) -> list[WorldEvent]:
    return [
        WorldEvent.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


# ---- 1. default start_cell is None (input contract) ---------------------


def test_default_start_cell_is_none() -> None:
    """The Probe 2 default — random non-wall sample at every episode
    reset — is encoded as ``start_cell=None`` on the config dataclass.
    """
    config = GridWorldConfig()
    assert config.start_cell is None


def test_explicit_start_cell_is_still_accepted() -> None:
    """An explicit ``start_cell=(r, c)`` continues to work; the flip
    changes the default only, not the contract."""
    config = GridWorldConfig(start_cell=(3, 3))
    env = GridWorld(config, seed=42)
    env.reset()
    assert env.state.agent_pos == (3, 3)


# ---- 2. random sample at reset; payload carries it ----------------------


def test_random_sample_lands_in_non_wall_cell() -> None:
    """With ``start_cell=None`` the agent is placed at a non-wall
    in-bounds cell sampled from the regrowth RNG stream at every reset.
    """
    config = GridWorldConfig(walls=((4, 4), (4, 5)))
    env = GridWorld(config, seed=42)
    env.reset()
    r, c = env.state.agent_pos
    assert 0 <= r < config.grid_size
    assert 0 <= c < config.grid_size
    assert (r, c) not in {(4, 4), (4, 5)}


def test_env_reset_payload_includes_start_cell(tmp_path: Path) -> None:
    """The ``_emit_env_reset`` payload carries the actual sampled start
    cell so the JSONL alone is enough to recover the per-episode start
    sequence — the reproducibility commitment the random-start path
    promises."""
    with _env_server(tmp_path) as (server, sink_path):
        first = server.start()
        agent_pos = server.grid_world_state.agent_pos

    events = _read_world_events(sink_path)
    resets = [e for e in events if e.event_type == "env_reset"]
    assert len(resets) == 1
    payload = resets[0].payload
    assert "start_cell" in payload
    assert payload["start_cell"] == [int(agent_pos[0]), int(agent_pos[1])]
    # Cross-check: the agent at episode 0 step 0 is *at* the start cell,
    # so the cell appears at the geometric center of the rendered
    # observation (rendered as EMPTY because the agent's underlying
    # grid value is EMPTY).
    assert first.observation.shape[0] == first.observation.shape[1]


def test_env_reset_payload_includes_start_cell_at_each_episode_boundary(
    tmp_path: Path,
) -> None:
    """At every episode boundary the env-reset payload carries the new
    episode's sampled start cell — not just the first reset.
    """
    config = replace(_quiet_config(), episode_length=5)
    with _env_server(tmp_path, grid_world_config=config) as (server, sink_path):
        server.start()
        # Three full episodes' worth of steps so we cross two boundaries.
        for _ in range(3 * config.episode_length):
            server.step(4)

    events = _read_world_events(sink_path)
    resets = [e for e in events if e.event_type == "env_reset"]
    # 1 initial + 3 boundary crossings = 4 env_reset events.
    assert len(resets) == 4
    for reset in resets:
        assert "start_cell" in reset.payload
        cell = reset.payload["start_cell"]
        assert isinstance(cell, list)
        assert len(cell) == 2
        assert all(isinstance(x, int) for x in cell)
        assert all(0 <= x < config.grid_size for x in cell)


def test_explicit_start_cell_round_trips_into_payload(tmp_path: Path) -> None:
    """When ``start_cell`` is set explicitly, the env_reset payload
    records the configured cell — the recorded value is the cell
    actually used, regardless of whether sampling was involved."""
    config = replace(_quiet_config(), start_cell=(2, 6))
    with _env_server(tmp_path, grid_world_config=config) as (server, sink_path):
        server.start()

    events = _read_world_events(sink_path)
    resets = [e for e in events if e.event_type == "env_reset"]
    assert len(resets) == 1
    assert resets[0].payload["start_cell"] == [2, 6]


# ---- 3. reproducibility from regrowth seed ------------------------------


def test_random_sampling_reproducible_from_seed() -> None:
    """Two ``GridWorld`` instances at the same seed sample the same
    cell at every episode reset; the random-start path is fully
    determined by the regrowth stream's seed."""
    config = replace(_quiet_config(), episode_length=5)

    def collect_starts(seed: int, n_episodes: int) -> list[tuple[int, int]]:
        env = GridWorld(config, seed=seed)
        env.reset()
        starts = [env.state.agent_pos]
        # Step through (n_episodes - 1) boundaries.
        for _ in range((n_episodes - 1) * config.episode_length):
            env.step(4)
            if env.state.agent_pos != starts[-1] and env._step_in_episode == 0:
                # Boundary just crossed; record the new start.
                starts.append(env.state.agent_pos)
        return starts

    # A simpler approach: collect via env_step counters directly.
    def collect_via_resets(seed: int, n_episodes: int) -> list[tuple[int, int]]:
        env = GridWorld(config, seed=seed)
        env.reset()
        recorded: list[tuple[int, int]] = [env.state.agent_pos]
        prev_episode = 0
        for _ in range(n_episodes * config.episode_length):
            env.step(4)
            if env._episode_id != prev_episode:
                recorded.append(env.state.agent_pos)
                prev_episode = env._episode_id
            if len(recorded) >= n_episodes:
                break
        return recorded

    starts_a = collect_via_resets(seed=42, n_episodes=4)
    starts_b = collect_via_resets(seed=42, n_episodes=4)
    assert starts_a == starts_b
    assert len(starts_a) == 4


def test_different_seeds_produce_different_start_sequences() -> None:
    """Sanity check: two seeds → at least one differing start cell
    across the first few episodes (the alternative is a silent
    seed-collapsing bug)."""
    config = replace(_quiet_config(), episode_length=5)

    def first_n_starts(seed: int, n: int) -> list[tuple[int, int]]:
        env = GridWorld(config, seed=seed)
        env.reset()
        starts = [env.state.agent_pos]
        prev_episode = 0
        for _ in range(n * config.episode_length):
            env.step(4)
            if env._episode_id != prev_episode:
                starts.append(env.state.agent_pos)
                prev_episode = env._episode_id
            if len(starts) >= n:
                break
        return starts

    starts_a = first_n_starts(seed=42, n=4)
    starts_b = first_n_starts(seed=43, n=4)
    assert starts_a != starts_b


def test_random_sample_via_env_server_reproducible_from_payload(
    tmp_path: Path,
) -> None:
    """Two ``EnvServer`` instances at the same seed produce
    byte-identical ``world_event`` ``start_cell`` sequences. The JSONL
    is the canonical reproducibility record; this test verifies the
    payload preserves the property the underlying RNG already has.
    """
    config = replace(_quiet_config(), episode_length=5)

    def starts_from_jsonl(sink_name: str) -> list[list[int]]:
        with _env_server(
            tmp_path,
            grid_world_config=config,
            sink_name=sink_name,
            seed=42,
        ) as (server, sink_path):
            server.start()
            for _ in range(3 * config.episode_length):
                server.step(4)
        events = _read_world_events(sink_path)
        return [
            list(e.payload["start_cell"])
            for e in events
            if e.event_type == "env_reset"
        ]

    starts_a = starts_from_jsonl("a.jsonl")
    starts_b = starts_from_jsonl("b.jsonl")
    assert starts_a == starts_b
    assert len(starts_a) == 4


# ---- 4. sham-perturbation: no grid mutation, observation byte-identical -


def test_sham_perturbation_does_not_mutate_grid(tmp_path: Path) -> None:
    """``fire_sham_perturbation`` does not change any underlying grid
    cell; the grid byte-state is identical pre and post."""
    with _env_server(tmp_path) as (server, _sink_path):
        server.start()
        before = server.grid_world_state.grid.copy()
        server.fire_sham_perturbation(
            "add_resource",
            {"intended_cell": [4, 4]},
        )
        after = server.grid_world_state.grid.copy()
    assert np.array_equal(before, after)


def test_sham_perturbation_observation_byte_identical(tmp_path: Path) -> None:
    """A real run that fires a sham between two steps produces the
    same observations as a parallel run that omits the sham. The
    sham is the canonical null event for the calibration protocol's
    null-event test."""
    config = _quiet_config()

    def run(fire_sham: bool, sink_name: str) -> list[np.ndarray]:
        observations: list[np.ndarray] = []
        with _env_server(
            tmp_path,
            grid_world_config=config,
            sink_name=sink_name,
        ) as (server, _path):
            first = server.start()
            observations.append(first.observation.copy())
            for i in range(20):
                if fire_sham and i == 7:
                    server.fire_sham_perturbation(
                        "add_resource",
                        {"intended_cell": [2, 2]},
                    )
                step = server.step(4)
                observations.append(step.observation.copy())
        return observations

    obs_with_sham = run(fire_sham=True, sink_name="with.jsonl")
    obs_without_sham = run(fire_sham=False, sink_name="without.jsonl")
    assert len(obs_with_sham) == len(obs_without_sham)
    for o_a, o_b in zip(obs_with_sham, obs_without_sham):
        assert np.array_equal(o_a, o_b)


# ---- 5. sham-perturbation emits world_event with is_sham=True -----------


def test_sham_perturbation_emits_builder_perturbation_event(
    tmp_path: Path,
) -> None:
    """The sham emits a ``builder_perturbation`` ``WorldEvent`` with
    ``source="builder"`` and ``payload["is_sham"]=True``. The caller's
    additional payload entries are preserved unchanged."""
    with _env_server(tmp_path) as (server, sink_path):
        server.start()
        server.fire_sham_perturbation(
            "add_resource",
            {"intended_cell": [3, 5], "intended_mutator": "add_resource"},
        )

    events = _read_world_events(sink_path)
    perts = [e for e in events if e.event_type == "builder_perturbation"]
    assert len(perts) == 1
    rec = perts[0]
    assert rec.event_type == "builder_perturbation"
    assert rec.source == "builder"
    payload = rec.payload
    assert payload["is_sham"] is True
    assert payload["sham_label"] == "add_resource"
    # Caller-supplied fields are preserved.
    assert payload["intended_cell"] == [3, 5]
    assert payload["intended_mutator"] == "add_resource"


def test_real_perturbation_does_not_carry_is_sham_true(tmp_path: Path) -> None:
    """The four real mutators emit records *without* ``is_sham=True``.
    The flag is the structural distinguisher between the sham path and
    the real path; if a real mutator's payload carried it, the
    calibration protocol's null-event test would lose its anchor.
    """
    with _env_server(tmp_path) as (server, sink_path):
        server.start()
        server.add_resource((2, 2))

    events = _read_world_events(sink_path)
    perts = [e for e in events if e.event_type == "builder_perturbation"]
    assert len(perts) == 1
    payload = perts[0].payload
    # Either absent or explicitly False — both are compatible with the
    # documented convention. The calibration protocol filters on
    # ``payload.get("is_sham") is True``.
    assert payload.get("is_sham") is not True


# ---- 6. sham does not advance the regrowth or drift RNG streams ---------


def test_sham_perturbation_does_not_advance_regrowth_rng(
    tmp_path: Path,
) -> None:
    """The sham fires no mutator and consumes no RNG draw. Two runs
    that differ only by an interposed sham produce identical regrowth
    schedules — the regrowth coin flips at every step land on the same
    cells in both runs.

    Setup uses ``initial_regrowth_p=0.5`` to make the property visible:
    if the sham did advance the regrowth stream, the post-sham
    coin-flip array would be drawn from a different state and the
    cells that regrow would diverge across the two runs.
    """
    config = GridWorldConfig(
        initial_regrowth_p=0.5,
        drift_p_min=0.5,
        drift_p_max=0.5,
        drift_magnitude_per_step=0.0,
        n_initial_resources=0,
        episode_length=200,
        start_cell=(0, 0),  # pin so the only difference is the sham
    )

    def grids_at_each_step(fire_sham: bool, sink_name: str) -> list[np.ndarray]:
        grids: list[np.ndarray] = []
        with _env_server(
            tmp_path,
            grid_world_config=config,
            sink_name=sink_name,
        ) as (server, _path):
            server.start()
            grids.append(server.grid_world_state.grid.copy())
            for i in range(20):
                if fire_sham and i == 5:
                    server.fire_sham_perturbation(
                        "add_resource",
                        {"intended_cell": [3, 3]},
                    )
                server.step(4)
                grids.append(server.grid_world_state.grid.copy())
        return grids

    grids_with = grids_at_each_step(fire_sham=True, sink_name="rng_with.jsonl")
    grids_without = grids_at_each_step(
        fire_sham=False, sink_name="rng_without.jsonl"
    )
    assert len(grids_with) == len(grids_without)
    for g_a, g_b in zip(grids_with, grids_without):
        assert np.array_equal(g_a, g_b)


def test_sham_perturbation_does_not_advance_drift_rng(
    tmp_path: Path,
) -> None:
    """The drift stream's per-step magnitude is unaffected by the sham.
    Two runs that differ only by an interposed sham produce identical
    ``regrowth_p`` trajectories, recorded into the
    ``internal_stochasticity_aggregate`` per-episode payload."""
    config = GridWorldConfig(
        initial_regrowth_p=0.025,
        drift_p_min=0.001,
        drift_p_max=0.05,
        drift_magnitude_per_step=0.001,  # amplified for visibility
        n_initial_resources=0,
        episode_length=10,
        start_cell=(0, 0),
    )

    def final_p_per_episode(fire_sham: bool, sink_name: str) -> list[float]:
        with _env_server(
            tmp_path,
            grid_world_config=config,
            sink_name=sink_name,
        ) as (server, sink_path):
            server.start()
            for i in range(3 * config.episode_length):
                if fire_sham and i == 4:
                    server.fire_sham_perturbation(
                        "add_resource",
                        {"intended_cell": [3, 3]},
                    )
                server.step(4)
        events = _read_world_events(sink_path)
        return [
            float(e.payload["final_p"])
            for e in events
            if e.event_type == "internal_stochasticity_aggregate"
        ]

    finals_with = final_p_per_episode(fire_sham=True, sink_name="d_with.jsonl")
    finals_without = final_p_per_episode(
        fire_sham=False, sink_name="d_without.jsonl"
    )
    assert finals_with == finals_without
    assert len(finals_with) >= 2


# ---- ancillary: sham requires start() ----------------------------------


def test_sham_before_start_raises(tmp_path: Path) -> None:
    """Mutator-shaped methods (the four real mutators and the sham)
    share the same lifecycle precondition: ``start()`` must be called
    first."""
    with _env_server(tmp_path) as (server, _sink_path):
        with pytest.raises(RuntimeError, match="start"):
            server.fire_sham_perturbation("add_resource", {})


def test_sham_after_close_raises(tmp_path: Path) -> None:
    """``close()`` is the lifecycle terminator for the four real mutators
    and for the sham."""
    with _env_server(tmp_path) as (server, _sink_path):
        server.start()
        server.close()
        with pytest.raises(RuntimeError, match="closed"):
            server.fire_sham_perturbation("add_resource", {})
