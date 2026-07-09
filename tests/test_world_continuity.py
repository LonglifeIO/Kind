"""World v2 W1 — E0: the world stops forgetting (plan W1; synthesis DP1/DP6).

Pins the ``episode_resample`` flag both ways: default ``True`` is
byte-identical to today's world (every existing episode-semantics test
doubles as a pin, plus a direct trajectory-identity test here); ``False``
makes the world persist indefinitely — no resample, no ``episode_id``
increment, no ``step_in_episode`` zeroing, no ``env_reset`` or
per-episode aggregate emission after session start, walls immortal,
consumption/regrowth/drift/energy continuous. The e0 stage preset
(``kind/env/world_stages.py``) is pinned here too, including the
S3 trivial-loop constraint that the terrain must not partition the grid.
"""

from __future__ import annotations

import dataclasses
import json
import math
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import CellType, GridWorld, GridWorldConfig
from kind.env.transport import EnvTransportClient, EnvTransportServer
from kind.env.world_stages import (
    E0_WALLS,
    WORLD_STAGES,
    apply_world_stage,
)
from kind.observer.schemas import WorldEvent
from kind.observer.sinks import JsonlSink
from kind.training.resume import (
    continuation_counters,
    resume_from_latest_checkpoint,
)
from kind.training.runner import Runner

from tests.test_integration_smoke import (  # the house tiny-config helpers
    _make_runner_config,
    _quiet_grid_world_config,
    _transport_pair,
)


def _actions(n: int, seed: int = 7) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(a) for a in rng.integers(0, 5, size=n)]


def _persistent_config(**overrides: object) -> GridWorldConfig:
    """A small continuing world: resample off, boundary would land at 50."""
    base = GridWorldConfig(
        episode_length=50,
        episode_resample=False,
        initial_regrowth_p=0.05,
        n_initial_resources=4,
        start_cell=(3, 3),
    )
    return dataclasses.replace(base, **overrides)  # type: ignore[arg-type]


# ---- the flag, default on: byte-identity ----------------------------------


def test_default_flag_is_byte_identical() -> None:
    """``episode_resample=True`` (the default) is today's world exactly."""
    default_world = GridWorld(GridWorldConfig(), seed=42)
    explicit_world = GridWorld(
        GridWorldConfig(episode_resample=True), seed=42
    )
    default_world.reset()
    explicit_world.reset()
    for action in _actions(450):  # crosses two 200-step boundaries
        step_a = default_world.step(action)
        step_b = explicit_world.step(action)
        assert step_a.episode_id == step_b.episode_id
        assert step_a.step_in_episode == step_b.step_in_episode
        assert step_a.sensed_energy == step_b.sensed_energy
        assert np.array_equal(step_a.observation, step_b.observation)
    assert default_world.state.true_energy == explicit_world.state.true_energy
    assert np.array_equal(default_world.state.grid, explicit_world.state.grid)


# ---- the flag, off: the world persists -------------------------------------


def test_resample_off_counters_and_persistence() -> None:
    """500+ steps: episode_id frozen, step_in_episode grows, walls immortal,
    the agent is never teleported (no boundary replacement)."""
    world = GridWorld(_persistent_config(walls=E0_WALLS), seed=42)
    step = world.reset()
    assert step.episode_id == 0
    prev_pos = world.state.agent_pos
    for i, action in enumerate(_actions(520)):
        step = world.step(action)
        assert step.episode_id == 0
        assert step.step_in_episode == i + 1  # grows without zeroing
        # Walls are immortal at every step.
        grid = world.state.grid
        for wr, wc in E0_WALLS:
            assert grid[wr, wc] == CellType.WALL.value
        # No teleport: one cardinal step (or stay) per env step — the
        # boundary's agent replacement never happens.
        r, c = world.state.agent_pos
        assert abs(r - prev_pos[0]) + abs(c - prev_pos[1]) <= 1
        prev_pos = (r, c)
    assert step.step_in_episode == 520


def test_resample_off_initial_episode_id_stays_seeded() -> None:
    """The continuation counters freeze at their seeded values."""
    config = _persistent_config(initial_env_step=1000, initial_episode_id=77)
    world = GridWorld(config, seed=42)
    world.reset()
    for action in _actions(120):
        step = world.step(action)
        assert step.episode_id == 77
    assert step.env_step == 1000 + 120


def test_resample_off_regrowth_and_energy_continue() -> None:
    """Consumption, regrowth, and energy dynamics keep running past where
    the boundary used to be; nothing re-initializes."""
    world = GridWorld(
        _persistent_config(
            initial_regrowth_p=0.3,
            drift_p_min=0.1,
            drift_p_max=0.5,
            drift_magnitude_per_step=0.001,
            n_initial_resources=0,
        ),
        seed=42,
    )
    world.reset()
    energies: list[float] = []
    resource_counts: list[int] = []
    for action in _actions(150):
        world.step(action)
        state = world.state
        energies.append(state.true_energy)
        resource_counts.append(
            int(np.count_nonzero(state.grid == CellType.RESOURCE.value))
        )
    # Regrowth is alive well past step 50 (the inert episode_length).
    assert max(resource_counts[60:]) > 0
    # Energy was never re-initialized mid-run: with base decay 0.08 and
    # replenish 0.8 per resource, a silent reset to energy_init=6.0
    # (normalized 0.6) would show as a jump > the largest legal delta.
    max_legal_delta = (0.8 + 0.08 + 0.04) / 10.0  # replenish + decay, normed
    for before, after in zip(energies, energies[1:]):
        assert abs(after - before) <= max_legal_delta + 1e-9


def test_resample_off_no_env_reset_events(tmp_path: Path) -> None:
    """EnvServer emits exactly one env_reset (session start) and zero
    per-episode aggregates when the world never resamples."""
    sink_path = tmp_path / "world_event.jsonl"
    sink = JsonlSink(sink_path, WorldEvent)
    server = EnvServer(
        EnvServerConfig(
            grid_world_config=_persistent_config(),
            seed=42,
            world_event_handler=sink.write,
            run_id="continuity-test",
        )
    )
    try:
        server.start()
        for action in _actions(160):  # 3× the (inert) episode_length
            server.step(action)
    finally:
        server.close()
        sink.close()

    event_types = [
        json.loads(line)["event_type"]
        for line in sink_path.read_text().splitlines()
    ]
    assert event_types.count("env_reset") == 1
    assert event_types.count("internal_stochasticity_aggregate") == 0


# ---- the e0 stage preset ----------------------------------------------------


def test_stage_default_is_unchanged() -> None:
    config = GridWorldConfig()
    assert apply_world_stage(config, "default") is config


def test_stage_e0_sets_continuity_and_terrain() -> None:
    config = GridWorldConfig(initial_env_step=5, initial_episode_id=3)
    staged = apply_world_stage(config, "e0")
    assert staged.episode_resample is False
    assert staged.walls == E0_WALLS
    # Everything else carries through untouched.
    assert staged.initial_env_step == 5
    assert staged.initial_episode_id == 3
    assert staged.episode_length == config.episode_length


def test_stage_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown world stage"):
        apply_world_stage(GridWorldConfig(), "e99")
    # Stages accumulate in order; later phases append, never reorder.
    assert WORLD_STAGES[:2] == ("default", "e0")


def test_e0_walls_do_not_partition_the_grid() -> None:
    """S3's trivial-loop confound: every non-wall cell must stay reachable
    from every other (4-connectivity flood fill)."""
    size = GridWorldConfig().grid_size
    walls = set(E0_WALLS)
    assert len(walls) == len(E0_WALLS), "duplicate wall cells"
    for r, c in walls:
        assert 0 <= r < size and 0 <= c < size
    open_cells = {
        (r, c)
        for r in range(size)
        for c in range(size)
        if (r, c) not in walls
    }
    start = next(iter(open_cells))
    seen = {start}
    frontier = [start]
    while frontier:
        r, c = frontier.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (r + dr, c + dc)
            if neighbor in open_cells and neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    assert seen == open_cells, "e0 terrain partitions the grid"


# ---- training smoke over the continuing world -------------------------------


def test_training_smoke_resample_off(tmp_path: Path) -> None:
    """Tiny-config training over 3× the old episode length in a world that
    never resamples: the run completes and every logged loss is finite."""
    config = _make_runner_config(tmp_path=tmp_path, run_id="smoke-e0")
    grid_config = dataclasses.replace(
        _quiet_grid_world_config(),
        episode_resample=False,
        walls=E0_WALLS,
        initial_regrowth_p=0.02,
        drift_p_max=0.05,
    )
    env_server = EnvServer(
        EnvServerConfig(
            grid_world_config=grid_config,
            seed=42,
            world_event_handler=lambda _r: None,
            run_id="smoke-e0-env",
        )
    )
    transport_server = EnvTransportServer(env_server, host="127.0.0.1", port=0)
    thread = threading.Thread(
        target=transport_server.serve_forever, daemon=True
    )
    thread.start()
    client = EnvTransportClient(
        host="127.0.0.1",
        port=transport_server.actual_port,
        world_event_handler=lambda _r: None,
    )
    try:
        runner = Runner(config, client, env_server=env_server)
        try:
            runner.run(total_env_steps=150)  # 3× episode_length=50
        finally:
            runner.close()
    finally:
        client.close()
        transport_server.shutdown()
        thread.join(timeout=5.0)

    rows: dict[str, list[float]] = {"kl_aggregate_t": [], "recon_loss_t": []}
    episode_ids: list[int] = []
    for shard in sorted(
        (config.telemetry_dir / "agent_step").glob("*.parquet")
    ):
        table = pq.read_table(
            shard, columns=["episode_id", "kl_aggregate_t", "recon_loss_t"]
        ).to_pydict()
        episode_ids.extend(int(e) for e in table["episode_id"])
        for key in rows:
            rows[key].extend(float(v) for v in table[key])
    assert len(episode_ids) == 150
    assert set(episode_ids) == {0}, "episode_id moved in a continuing world"
    for key, values in rows.items():
        assert all(math.isfinite(v) for v in values), f"non-finite {key}"


# ---- resume into e0 ---------------------------------------------------------


@contextmanager
def _e0_transport_pair(
    *,
    run_id: str,
    initial_env_step: int,
    initial_episode_id: int,
    seed: int = 43,
) -> Iterator[tuple[EnvTransportClient, EnvServer]]:
    """The continuation transport pair with the e0 stage applied — the
    launcher's ``--resume --world-stage e0`` composition, in miniature."""
    grid_config = apply_world_stage(
        dataclasses.replace(
            _quiet_grid_world_config(),
            initial_env_step=initial_env_step,
            initial_episode_id=initial_episode_id,
        ),
        "e0",
    )
    env_server = EnvServer(
        EnvServerConfig(
            grid_world_config=grid_config,
            seed=seed,
            world_event_handler=lambda _r: None,
            run_id=run_id,
        )
    )
    server = EnvTransportServer(env_server, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = EnvTransportClient(
        host="127.0.0.1",
        port=server.actual_port,
        world_event_handler=lambda _r: None,
    )
    try:
        yield client, env_server
    finally:
        client.close()
        server.shutdown()
        thread.join(timeout=5.0)


def test_resume_into_e0_continues_the_biography(tmp_path: Path) -> None:
    """Session 1 in the default (resampling) world; session 2 resumes the
    same mind into a persistent e0 world. Telemetry continues, the resume
    marker records the stage shape, and session 2's episode_id is frozen
    at its seeded value."""
    run_dir = tmp_path / "biography"
    config_a = _make_runner_config(
        tmp_path=run_dir,
        checkpoint_every_n_env_steps=20,
        warmup_env_steps=5,
        dream_cadence_env_steps=10_000,
        run_id="resume-e0",
    )
    with _transport_pair(run_id="resume-e0-env-a") as (
        client,
        _server,
        env_server,
        _thread,
    ):
        runner_a = Runner(config_a, client, env_server=env_server)
        try:
            runner_a.run(total_env_steps=21)
        finally:
            runner_a.close()

    initial_env_step, initial_episode_id = continuation_counters(
        config_a.telemetry_dir
    )
    config_b = _make_runner_config(
        tmp_path=run_dir,
        checkpoint_every_n_env_steps=20,
        warmup_env_steps=5,
        dream_cadence_env_steps=10_000,
        run_id="resume-e0",
    )
    with _e0_transport_pair(
        run_id="resume-e0-env-b",
        initial_env_step=initial_env_step,
        initial_episode_id=initial_episode_id,
    ) as (client, env_server):
        runner_b = Runner(config_b, client, env_server=env_server)
        try:
            resumed_from = resume_from_latest_checkpoint(
                runner_b,
                client,
                env_server,
                marker_extra={"world_stage": "e0"},
            )
            assert resumed_from == "ckpt-000001"
            runner_b.run(total_env_steps=60)  # crosses the inert boundary
        finally:
            runner_b.close()

    # Session 2's rows: t continues monotonically; episode_id frozen at
    # the seeded value for the whole session (the continuing world). The
    # first resumed row stamps from the checkpointed pending state — the
    # paused episode's id — per the documented resume convention
    # (``continuation_counters`` docstring); every row after it carries
    # the fresh world's frozen id.
    ts: list[int] = []
    session2_episode_ids: set[int] = set()
    for shard in sorted(
        (config_a.telemetry_dir / "agent_step").glob("*.parquet")
    ):
        table = pq.read_table(shard, columns=["t", "episode_id"]).to_pydict()
        for t, episode_id in zip(table["t"], table["episode_id"]):
            ts.append(int(t))
            if int(t) > initial_env_step:  # past the stitching row
                session2_episode_ids.add(int(episode_id))
    assert len(ts) == 21 + 60
    assert len(set(ts)) == len(ts)
    assert session2_episode_ids == {initial_episode_id}

    # The resume marker landed and carries the stage.
    markers = [
        json.loads(line)
        for line in (config_a.telemetry_dir / "world_event.jsonl")
        .read_text()
        .splitlines()
        if json.loads(line)["payload"].get("sham_label") == "resume_marker"
    ]
    assert len(markers) == 1
    assert markers[0]["payload"]["world_stage"] == "e0"
