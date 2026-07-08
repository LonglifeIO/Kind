"""Continuation plan C1 — resume-and-CONTINUE across a pause.

The existing integration smokes pin load-state equality
(weights/RNG/optimizer round-trip). This file pins the part the
biography actually needs and nothing had tested: a fresh process
(fresh env server + transport + runner) loads the latest checkpoint
via the resume helper and **keeps running** — telemetry continues in
the same streams, the step counter does not restart, and the resume
marker lands in the world_event record (flag-only, excluded from
Phase-3 anchor extraction by the existing sham test).
"""

from __future__ import annotations

import dataclasses
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.transport import EnvTransportClient, EnvTransportServer
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


@contextmanager
def _continuation_transport_pair(
    *,
    run_id: str,
    initial_env_step: int,
    initial_episode_id: int,
    seed: int = 43,
) -> Iterator[tuple[EnvTransportClient, EnvServer]]:
    """The smoke `_transport_pair`, with the session-2 counter seeds."""
    grid_config = dataclasses.replace(
        _quiet_grid_world_config(),
        initial_env_step=initial_env_step,
        initial_episode_id=initial_episode_id,
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


def _agent_step_ts(telemetry_dir: Path) -> list[int]:
    ts: list[int] = []
    for shard in sorted((telemetry_dir / "agent_step").glob("*.parquet")):
        ts.extend(
            int(t)
            for t in pq.read_table(shard, columns=["t"]).to_pydict()["t"]
        )
    return sorted(ts)


def test_resume_continues_same_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "biography"
    config_a = _make_runner_config(
        tmp_path=run_dir,
        checkpoint_every_n_env_steps=20,
        warmup_env_steps=5,
        dream_cadence_env_steps=10_000,
        run_id="resume-continue",
    )
    with _transport_pair(run_id="resume-continue-env-a") as (
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

    session1_ts = _agent_step_ts(config_a.telemetry_dir)
    assert len(session1_ts) == 21

    # Session 2: fresh everything, same run dir (the biography contract);
    # the fresh world's counters seeded from the run's own record.
    initial_env_step, initial_episode_id = continuation_counters(
        config_a.telemetry_dir
    )
    assert initial_env_step == max(session1_ts) + 1
    config_b = _make_runner_config(
        tmp_path=run_dir,
        checkpoint_every_n_env_steps=20,
        warmup_env_steps=5,
        dream_cadence_env_steps=10_000,
        run_id="resume-continue",
    )
    with _continuation_transport_pair(
        run_id="resume-continue-env-b",
        initial_env_step=initial_env_step,
        initial_episode_id=initial_episode_id,
    ) as (client, env_server):
        runner_b = Runner(config_b, client, env_server=env_server)
        try:
            resumed_from = resume_from_latest_checkpoint(
                runner_b, client, env_server, marker_extra={"session": 2}
            )
            assert resumed_from == "ckpt-000001"
            runner_b.run(total_env_steps=10)
        finally:
            runner_b.close()

    # Telemetry continued: session 1's shards intact, more steps after,
    # t strictly monotonic across the boundary, no duplicates.
    all_ts = _agent_step_ts(config_a.telemetry_dir)
    assert len(all_ts) == len(session1_ts) + 10
    assert len(set(all_ts)) == len(all_ts), "duplicate t in agent_step"
    assert all_ts[: len(session1_ts)] == session1_ts, (
        "session 1 telemetry was modified by the resumed session"
    )
    assert min(all_ts[len(session1_ts) :]) > max(session1_ts)

    # The resume marker landed, flag-only, with the payload contract.
    markers = []
    world_events = (config_a.telemetry_dir / "world_event.jsonl").read_text()
    for line in world_events.splitlines():
        record = json.loads(line)
        if record["payload"].get("sham_label") == "resume_marker":
            markers.append(record)
    assert len(markers) == 1
    payload = markers[0]["payload"]
    assert payload["is_sham"] is True
    assert payload["resumed_from_checkpoint"] == "ckpt-000001"
    assert payload["session"] == 2


def test_resume_without_checkpoint_raises(tmp_path: Path) -> None:
    config = _make_runner_config(
        tmp_path=tmp_path / "fresh",
        run_id="resume-fresh",
    )
    with _transport_pair(run_id="resume-fresh-env") as (
        client,
        _server,
        env_server,
        _thread,
    ):
        runner = Runner(config, client, env_server=env_server)
        try:
            with pytest.raises(FileNotFoundError):
                resume_from_latest_checkpoint(runner, client, env_server)
        finally:
            runner.close()
