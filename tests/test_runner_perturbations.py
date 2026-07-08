"""Probe 4 Phase 2 — runner-level step-boundary perturbation drain
(plan §S-PERT).

The generator poll and manual-trigger inbox drain live inside
``_step_once`` between the env step and the AgentStep emit. These tests
run a tiny real runner (the integration-smoke pattern) and pin: opt-in
default (no builder events without config), generator events in the
world_event stream with ``trigger="generator"`` at the configured
spacing, inbox requests fired mid-run with ``trigger="manual"``, and the
co-located env_server requirement.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from kind.agents.world_model import WorldModelConfig
from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import GridWorldConfig
from kind.env.perturbation_generator import PerturbationGeneratorConfig
from kind.env.transport import EnvTransportClient, EnvTransportServer
from kind.env.trigger_inbox import write_trigger_request
from kind.training.runner import Runner, RunnerConfig


def _quiet_grid_world_config() -> GridWorldConfig:
    return GridWorldConfig(
        episode_length=50,
        initial_regrowth_p=0.0,
        drift_p_min=0.0,
        drift_p_max=0.0,
        drift_magnitude_per_step=0.0,
        n_initial_resources=2,
    )


def _tiny_world_model_config() -> WorldModelConfig:
    return WorldModelConfig(
        obs_channels=1,
        obs_size=32,
        h_dim=32,
        z_dim=4,
        embed_dim=32,
        num_actions=5,
        action_emb_dim=8,
        mlp_hidden=32,
        free_bits_per_dim=1.0,
    )


def _make_runner_config(
    tmp_path: Path,
    *,
    perturbation_generator: PerturbationGeneratorConfig | None = None,
    perturbation_inbox_dir: Path | None = None,
) -> RunnerConfig:
    return RunnerConfig(
        world_model_config=_tiny_world_model_config(),
        run_id="perturbation-smoke",
        telemetry_dir=tmp_path / "telemetry",
        checkpoints_dir=tmp_path / "checkpoints",
        action_dim=5,
        ensemble_k=2,
        imagination_horizon=4,
        replay_capacity=200,
        replay_sequence_length=4,
        replay_batch_size=2,
        train_every_n_env_steps=1,
        warmup_env_steps=10,
        dream_cadence_env_steps=50,
        dream_horizon=4,
        checkpoint_every_n_env_steps=100,
        parquet_rows_per_shard=10,
        device="cpu",
        perturbation_generator=perturbation_generator,
        perturbation_inbox_dir=perturbation_inbox_dir,
    )


@contextmanager
def _transport_pair() -> Iterator[tuple[EnvTransportClient, EnvServer]]:
    env_server = EnvServer(
        EnvServerConfig(
            grid_world_config=_quiet_grid_world_config(),
            seed=42,
            world_event_handler=lambda _r: None,
            run_id="perturbation-smoke-env",
        )
    )
    transport_server = EnvTransportServer(
        env_server, host="127.0.0.1", port=0
    )
    server_thread = threading.Thread(
        target=transport_server.serve_forever,
        name="PerturbationSmokeServer",
        daemon=True,
    )
    server_thread.start()
    client = EnvTransportClient(
        host="127.0.0.1",
        port=transport_server.actual_port,
        world_event_handler=lambda _r: None,
    )
    try:
        yield client, env_server
    finally:
        try:
            client.close()
        finally:
            transport_server.shutdown()
            server_thread.join(timeout=5.0)


def _read_builder_events(telemetry_dir: Path) -> list[dict[str, object]]:
    path = telemetry_dir / "world_event.jsonl"
    if not path.exists():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    return [r for r in rows if r["event_type"] == "builder_perturbation"]


# ---- config validation ------------------------------------------------------


def test_generator_requires_co_located_env_server(tmp_path: Path) -> None:
    config = _make_runner_config(
        tmp_path,
        perturbation_generator=PerturbationGeneratorConfig(seed=0),
    )
    with _transport_pair() as (client, _env_server):
        with pytest.raises(ValueError, match="co-located env_server"):
            Runner(config, client, env_server=None)


def test_inbox_requires_co_located_env_server(tmp_path: Path) -> None:
    config = _make_runner_config(
        tmp_path, perturbation_inbox_dir=tmp_path / "inbox"
    )
    with _transport_pair() as (client, _env_server):
        with pytest.raises(ValueError, match="co-located env_server"):
            Runner(config, client, env_server=None)


# ---- opt-in default ---------------------------------------------------------


def test_default_config_fires_no_builder_events(tmp_path: Path) -> None:
    config = _make_runner_config(tmp_path)
    with _transport_pair() as (client, env_server):
        runner = Runner(config, client, env_server=env_server)
        try:
            runner.run(20)
        finally:
            runner.close()
    assert _read_builder_events(config.telemetry_dir) == []


# ---- generator drain --------------------------------------------------------


def test_generator_events_fire_with_tag_and_spacing(tmp_path: Path) -> None:
    generator_config = PerturbationGeneratorConfig(
        seed=5,
        min_spacing_steps=5,
        spacing_jitter_steps=2,
        cells_per_event=2,
        exclusion_radius=1,
    )
    config = _make_runner_config(
        tmp_path, perturbation_generator=generator_config
    )
    with _transport_pair() as (client, env_server):
        runner = Runner(config, client, env_server=env_server)
        try:
            runner.run(40)
        finally:
            runner.close()

    events = _read_builder_events(config.telemetry_dir)
    assert len(events) >= 4  # spacing <= 7 over 40 steps, 2 cells/event
    for event in events:
        payload = event["payload"]
        assert isinstance(payload, dict)
        assert payload["trigger"] == "generator"
        assert payload["mutator"] == "add_resource"
        assert payload["pre_state"] == "empty"
        assert payload["post_state"] == "resource"

    # Events arrive in same-t_event clusters of cells_per_event; the
    # cluster start steps respect the spacing floor.
    t_events = sorted({int(e["t_event"]) for e in events})  # type: ignore[arg-type]
    gaps = [b - a for a, b in zip(t_events, t_events[1:])]
    assert all(gap >= 5 for gap in gaps)
    per_boundary = {
        t: sum(1 for e in events if e["t_event"] == t) for t in t_events
    }
    assert all(count == 2 for count in per_boundary.values())


def test_generator_run_is_reproducible(tmp_path: Path) -> None:
    """Same seeds end-to-end → identical builder event streams (the
    step-boundary placement determinism gate)."""

    def run_once(base: Path) -> list[tuple[int, list[int]]]:
        generator_config = PerturbationGeneratorConfig(
            seed=5, min_spacing_steps=5, spacing_jitter_steps=2
        )
        config = _make_runner_config(
            base, perturbation_generator=generator_config
        )
        import torch

        torch.manual_seed(1234)
        with _transport_pair() as (client, env_server):
            runner = Runner(config, client, env_server=env_server)
            try:
                runner.run(30)
            finally:
                runner.close()
        return [
            (int(e["t_event"]), list(e["payload"]["cell"]))  # type: ignore[index, arg-type, call-overload]
            for e in _read_builder_events(config.telemetry_dir)
        ]

    run_a = run_once(tmp_path / "a")
    run_b = run_once(tmp_path / "b")
    assert run_a == run_b
    assert len(run_a) > 0


# ---- inbox drain mid-run ----------------------------------------------------


def test_inbox_request_fires_mid_run_with_manual_tag(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    write_trigger_request(inbox, "add_resource", {"cell": [6, 6]})
    config = _make_runner_config(tmp_path, perturbation_inbox_dir=inbox)
    with _transport_pair() as (client, env_server):
        runner = Runner(config, client, env_server=env_server)
        try:
            runner.run(5)
        finally:
            runner.close()

    events = _read_builder_events(config.telemetry_dir)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert isinstance(payload, dict)
    assert payload["trigger"] == "manual"
    assert payload["cell"] == [6, 6]
    # Archived with a result sidecar.
    processed = list((inbox / "processed").glob("*.result.json"))
    assert len(processed) == 1
    assert json.loads(processed[0].read_text())["ok"] is True
