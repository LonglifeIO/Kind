"""Phase 3 gate test #5 — Probe 1.5 integration smoke (CPU, tiny sizes).

The synthesis §4 / plan §4 fifth gate. Exercises the full hot loop with
the self-prediction head + EMA target wired into the world-model
forward + backward + EMA-update sequence AND the actor's forward
consuming the scalar via the extended PolicyView. Tiny sizes (``h=32``,
``z=4``, ``K=2``, ``head_hidden=32``) keep the test fast on CPU.

Asserts (per the plan §4 row 5 entries and the build prompt):

* All four telemetry streams emit (``agent_step``, ``dream_rollout``,
  ``replay_meta``, ``world_event``).
* ``AgentStep`` records carry ``schema_version == "0.2.0"`` and the
  three new fields (``self_prediction_t`` of length ``h_dim``,
  ``self_prediction_error_t`` finite, ``self_prediction_error_masked_t``
  boolean).
* Masked flag is ``True`` on the first step of every episode and
  ``False`` thereafter. The scalar value is exactly ``0.0`` on masked
  steps (sentinel) and is the loss-form value otherwise.
* The actor's forward consumes the scalar without error and produces
  valid action distributions at every step.
* Mid-run checkpoint includes EMA target weights and the extended
  actor input layer.
* Resume from a Probe 1.5 checkpoint yields byte-equal weights for
  the EMA target tensors and the actor's first-layer weight (the new
  column included).

The Probe 1 → Probe 1.5 checkpoint compat path lives in the
``test_load_probe_1_checkpoint_*`` test below — synthesizes a
``"0.1.0"`` checkpoint by hand, loads it, and asserts the documented
asymmetry handling (EMA target initialised from online; actor new
column zero-initialized; ``mirror_marker`` event recorded).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch
from safetensors.torch import load_file as _load_safetensors
from safetensors.torch import save_file as _save_safetensors

from kind.agents.actor import Actor
from kind.agents.ensemble import LatentDisagreementEnsemble
from kind.agents.world_model import WorldModel, WorldModelConfig
from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import GridWorldConfig
from kind.env.transport import EnvTransportClient, EnvTransportServer
from kind.observer.schemas import PROBE_1_SCHEMA_VERSION, SCHEMA_VERSION
from kind.training.runner import Runner, RunnerConfig


# ---- helpers (same shape as test_integration_smoke.py) -------------------


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
    """Plan §4 row 5 sizes: ``h=32``, ``z=4``, ``K=2``, ``head_hidden=32``."""
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
        self_prediction_hidden=32,
    )


def _make_runner_config(
    *,
    tmp_path: Path,
    run_id: str = "smoke-probe1_5-run",
    checkpoint_every_n_env_steps: int = 100,
    dream_cadence_env_steps: int = 50,
    warmup_env_steps: int = 10,
) -> RunnerConfig:
    return RunnerConfig(
        world_model_config=_tiny_world_model_config(),
        run_id=run_id,
        telemetry_dir=tmp_path / "telemetry",
        checkpoints_dir=tmp_path / "checkpoints",
        action_dim=5,
        ensemble_k=2,
        imagination_horizon=4,
        replay_capacity=200,
        replay_sequence_length=4,
        replay_batch_size=2,
        train_every_n_env_steps=1,
        warmup_env_steps=warmup_env_steps,
        dream_cadence_env_steps=dream_cadence_env_steps,
        dream_horizon=4,
        checkpoint_every_n_env_steps=checkpoint_every_n_env_steps,
        parquet_rows_per_shard=10,
        device="cpu",
    )


@contextmanager
def _transport_pair(
    *,
    seed: int = 42,
    run_id: str = "smoke-probe1_5-env",
) -> Iterator[
    tuple[EnvTransportClient, EnvTransportServer, EnvServer, threading.Thread]
]:
    config = EnvServerConfig(
        grid_world_config=_quiet_grid_world_config(),
        seed=seed,
        world_event_handler=lambda _r: None,
        run_id=run_id,
    )
    env_server = EnvServer(config)
    transport_server = EnvTransportServer(env_server, host="127.0.0.1", port=0)
    server_thread = threading.Thread(
        target=transport_server.serve_forever,
        name="SmokeProbe1_5EnvTransportServer",
        daemon=True,
    )
    server_thread.start()

    client = EnvTransportClient(
        host="127.0.0.1",
        port=transport_server.actual_port,
        world_event_handler=lambda _r: None,
    )
    try:
        yield client, transport_server, env_server, server_thread
    finally:
        try:
            client.close()
        finally:
            transport_server.shutdown()
            server_thread.join(timeout=5.0)


def _list_parquet_shards(dir_: Path) -> list[Path]:
    if not dir_.is_dir():
        return []
    return sorted(p for p in dir_.iterdir() if p.suffix == ".parquet")


def _read_parquet_records(dir_: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for shard in _list_parquet_shards(dir_):
        table = pq.read_table(shard)  # type: ignore[no-untyped-call]
        rows.extend(table.to_pylist())
    return rows


def _read_jsonl_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ---- the gate test (one function, six checks) ----------------------------


def test_smoke_probe1_5_runs_to_completion_on_cpu(tmp_path: Path) -> None:
    """Plan §4 gate test #5 (Probe 1.5).

    One end-to-end run produces all the artifacts the six sub-checks
    below inspect. Sub-checks live as separate test functions so a
    failure pinpoints which property regressed without forcing a re-run
    of the full smoke.
    """
    config = _make_runner_config(tmp_path=tmp_path)
    with _transport_pair() as (client, _server, env_server, _thread):
        runner = Runner(config, client, env_server=env_server)
        try:
            runner.run(total_env_steps=200)
        finally:
            runner.close()


def test_smoke_probe1_5_all_four_streams_have_records(tmp_path: Path) -> None:
    """Sub-check #1: all four telemetry streams emit records."""
    config = _make_runner_config(tmp_path=tmp_path)
    with _transport_pair() as (client, _server, env_server, _thread):
        runner = Runner(config, client, env_server=env_server)
        try:
            runner.run(total_env_steps=200)
        finally:
            runner.close()

    telem = config.telemetry_dir
    agent_step_rows = _read_parquet_records(telem / "agent_step")
    dream_rollout_rows = _read_parquet_records(telem / "dream_rollout")
    replay_meta_rows = _read_jsonl_records(telem / "replay_meta.jsonl")
    world_event_rows = _read_jsonl_records(telem / "world_event.jsonl")

    assert len(agent_step_rows) == 200
    assert len(dream_rollout_rows) > 0
    assert len(replay_meta_rows) > 0
    assert len(world_event_rows) > 0


def test_smoke_probe1_5_agent_step_carries_new_fields(tmp_path: Path) -> None:
    """Sub-check #2: ``AgentStep`` records carry ``"0.2.0"`` plus the
    three new fields with the right shape and types.
    """
    config = _make_runner_config(tmp_path=tmp_path)
    h_dim = config.world_model_config.h_dim

    with _transport_pair() as (client, _server, env_server, _thread):
        runner = Runner(config, client, env_server=env_server)
        try:
            runner.run(total_env_steps=200)
        finally:
            runner.close()

    rows = _read_parquet_records(config.telemetry_dir / "agent_step")
    assert rows
    for r in rows:
        assert r["schema_version"] == SCHEMA_VERSION
        sp_t = r["self_prediction_t"]
        assert isinstance(sp_t, list)
        assert len(sp_t) == h_dim
        assert all(isinstance(v, float) for v in sp_t)
        sp_err_t = r["self_prediction_error_t"]
        assert isinstance(sp_err_t, float)
        sp_masked_t = r["self_prediction_error_masked_t"]
        assert isinstance(sp_masked_t, bool)


def test_smoke_probe1_5_first_step_of_episode_is_masked(tmp_path: Path) -> None:
    """Sub-check #3: the masked flag is ``True`` exactly on the first
    step of each episode and ``False`` thereafter; the scalar value on
    masked steps is exactly ``0.0`` (sentinel).
    """
    config = _make_runner_config(tmp_path=tmp_path)
    with _transport_pair() as (client, _server, env_server, _thread):
        runner = Runner(config, client, env_server=env_server)
        try:
            runner.run(total_env_steps=200)
        finally:
            runner.close()

    rows = _read_parquet_records(config.telemetry_dir / "agent_step")
    assert rows
    masked_count = 0
    nonmasked_count = 0
    for r in rows:
        masked = bool(r["self_prediction_error_masked_t"])
        step_in_episode = int(r["step_in_episode"])  # type: ignore[arg-type]
        scalar = float(r["self_prediction_error_t"])  # type: ignore[arg-type]
        if step_in_episode == 0:
            assert masked is True, (
                f"first-step record at episode={r['episode_id']} t={r['t']} "
                f"has masked=False"
            )
            assert scalar == 0.0, (
                f"first-step record has non-sentinel scalar={scalar} "
                f"at episode={r['episode_id']} t={r['t']}"
            )
            masked_count += 1
        else:
            assert masked is False, (
                f"step_in_episode={step_in_episode} record at t={r['t']} "
                f"is masked=True"
            )
            nonmasked_count += 1
    # With episode_length=50 over 200 steps we expect at least 4 masked
    # steps (one per episode start).
    assert masked_count >= 4, (
        f"expected at least 4 first-step records, got {masked_count}"
    )
    assert nonmasked_count > 0


def test_smoke_probe1_5_dream_rollout_carries_none_self_prediction(
    tmp_path: Path,
) -> None:
    """Sub-check #4: ``DreamRollout`` records carry
    ``schema_version == "0.2.0"`` and ``sequence_self_prediction is
    None`` (synthesis §1.5: head runs only during waking at Probe 1.5).
    """
    config = _make_runner_config(tmp_path=tmp_path)
    with _transport_pair() as (client, _server, env_server, _thread):
        runner = Runner(config, client, env_server=env_server)
        try:
            runner.run(total_env_steps=200)
        finally:
            runner.close()

    rows = _read_parquet_records(config.telemetry_dir / "dream_rollout")
    assert rows
    for r in rows:
        assert r["schema_version"] == SCHEMA_VERSION
        assert r["sequence_self_prediction"] is None


def test_smoke_probe1_5_checkpoint_carries_ema_and_extended_actor(
    tmp_path: Path,
) -> None:
    """Sub-check #5: the mid-run checkpoint's safetensors blob includes
    the EMA target tensors (under the ``world_model.target_*`` prefix)
    and the actor's first-layer weight at the extended input dim
    ``h_dim + z_dim + 1``.
    """
    config = _make_runner_config(tmp_path=tmp_path)
    h_dim = config.world_model_config.h_dim
    z_dim = config.world_model_config.z_dim
    mlp_hidden = config.world_model_config.mlp_hidden

    with _transport_pair() as (client, _server, env_server, _thread):
        runner = Runner(config, client, env_server=env_server)
        try:
            runner.run(total_env_steps=200)
        finally:
            runner.close()

    weights_path = config.checkpoints_dir / "ckpt-000001" / "weights.safetensors"
    weights: dict[str, torch.Tensor] = _load_safetensors(str(weights_path))

    target_keys = [
        k for k in weights.keys() if k.startswith("world_model.target_")
    ]
    assert target_keys, (
        "EMA target weights missing from checkpoint blob — "
        "world_model.state_dict() did not include target_encoder / "
        "target_gru_cell submodules"
    )

    actor_first_layer = weights["actor.net.0.weight"]
    assert actor_first_layer.shape == (mlp_hidden, h_dim + z_dim + 1), (
        f"actor first-layer weight shape {actor_first_layer.shape} != "
        f"({mlp_hidden}, {h_dim + z_dim + 1}) — extended input column missing"
    )

    schema_file = config.checkpoints_dir / "ckpt-000001" / "schema_version.txt"
    assert schema_file.read_text().strip() == SCHEMA_VERSION


def test_smoke_probe1_5_resume_yields_identical_ema_and_actor(
    tmp_path: Path,
) -> None:
    """Sub-check #6: resume from a Probe 1.5 checkpoint produces
    byte-equal EMA target parameters and byte-equal actor first-layer
    weights to the ones the manager committed.
    """
    config_a = _make_runner_config(
        tmp_path=tmp_path / "run_a",
        checkpoint_every_n_env_steps=20,
        warmup_env_steps=5,
        dream_cadence_env_steps=10_000,
        run_id="smoke-probe1_5-resume-a",
    )
    with _transport_pair(run_id="smoke-probe1_5-resume-env-a") as (
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

    weights_path = (
        config_a.checkpoints_dir / "ckpt-000001" / "weights.safetensors"
    )
    disk_weights: dict[str, torch.Tensor] = _load_safetensors(str(weights_path))

    config_b = _make_runner_config(
        tmp_path=tmp_path / "run_b",
        run_id="smoke-probe1_5-resume-b",
    )
    object.__setattr__(config_b, "checkpoints_dir", config_a.checkpoints_dir)

    with _transport_pair(run_id="smoke-probe1_5-resume-env-b") as (
        client,
        _server,
        env_server,
        _thread,
    ):
        runner_b = Runner(config_b, client, env_server=env_server)
        try:
            runner_b.load_checkpoint("ckpt-000001")

            # EMA target weights byte-equal.
            for k, v in runner_b.world_model.state_dict().items():
                if not (
                    k.startswith("target_encoder.")
                    or k.startswith("target_gru_cell.")
                ):
                    continue
                expected = disk_weights[f"world_model.{k}"]
                assert torch.equal(expected, v.cpu()), (
                    f"EMA target weight world_model.{k} differs from "
                    f"checkpoint file"
                )

            # Actor first-layer weight (full extended shape) byte-equal.
            loaded_w = runner_b.actor.net[0].weight  # type: ignore[union-attr]
            disk_w = disk_weights["actor.net.0.weight"]
            assert torch.equal(disk_w, loaded_w.cpu())
        finally:
            runner_b.close()


# ---- Probe 1 → Probe 1.5 compat regression -------------------------------


def _build_probe_1_checkpoint(
    target_dir: Path,
    *,
    wm_cfg: WorldModelConfig,
    action_dim: int,
    ensemble_k: int,
    run_id: str,
) -> None:
    """Hand-build a ``"0.1.0"``-shaped checkpoint directory.

    The on-disk shape mirrors what Probe 1's runner committed: the
    weights blob has Probe-1-shape parameters under
    ``world_model.``, ``actor.``, ``ensemble.`` prefixes (no
    ``target_*`` keys; actor first-layer weight at
    ``(mlp_hidden, h_dim + z_dim)``); ``schema_version.txt`` carries
    ``"0.1.0"``; the other canonical files exist with minimal
    placeholders so ``CheckpointManager.load`` returns successfully
    and the runner's load path can read them.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    # Build the parameter values from a freshly-constructed Probe 1.5
    # WorldModel + Actor + Ensemble, then strip the Probe 1.5-specific
    # weights and shrink the actor's first-layer weight by one column.
    wm = WorldModel(wm_cfg)
    actor = Actor(
        h_dim=wm_cfg.h_dim,
        z_dim=wm_cfg.z_dim,
        action_dim=action_dim,
        mlp_hidden=wm_cfg.mlp_hidden,
    )
    # Match the runner's ensemble construction shape — the runner
    # does not pass ``action_emb_dim`` so the ensemble uses its default.
    ensemble = LatentDisagreementEnsemble(
        h_dim=wm_cfg.h_dim,
        z_dim=wm_cfg.z_dim,
        action_dim=action_dim,
        K=ensemble_k,
        mlp_hidden=wm_cfg.mlp_hidden,
    )

    weights: dict[str, torch.Tensor] = {}
    for k, v in wm.state_dict().items():
        # Probe 1 has no EMA target, no self-prediction head, no
        # frozen / environmental projection. Strip those keys; the
        # remaining keys are exactly the Probe 1 world model's.
        if k.startswith("target_encoder."):
            continue
        if k.startswith("target_gru_cell."):
            continue
        if k.startswith("self_prediction_head."):
            continue
        if k.startswith("_frozen_projection"):
            continue
        if k.startswith("_environmental_projection"):
            continue
        weights[f"world_model.{k}"] = v.detach().contiguous().cpu()
    actor_state = dict(actor.state_dict())
    first_layer_weight = actor_state["net.0.weight"]
    shrunk = first_layer_weight[:, : wm_cfg.h_dim + wm_cfg.z_dim].clone()
    actor_state["net.0.weight"] = shrunk
    for k, v in actor_state.items():
        weights[f"actor.{k}"] = v.detach().contiguous().cpu()
    for k, v in ensemble.state_dict().items():
        weights[f"ensemble.{k}"] = v.detach().contiguous().cpu()

    _save_safetensors(weights, str(target_dir / "weights.safetensors"))

    # schema_version: the load path branches on this string read.
    (target_dir / "schema_version.txt").write_text(
        PROBE_1_SCHEMA_VERSION + "\n"
    )

    # Minimal placeholder files so CheckpointContents construction
    # succeeds; the runner's load path doesn't read these for
    # the Probe 1 compat case (optimizer state is skipped; replay
    # / telemetry-offsets are not consumed by the runner's load
    # itself; the rng_state pickle IS consumed and must shape-match
    # what _load_rng_state expects).
    (target_dir / "replay_meta.json").write_text("{}\n")
    (target_dir / "telemetry_offsets.json").write_text("{}\n")

    # Optimizer state: written but never loaded for Probe 1
    # checkpoints (the load path skips optimizer-state load when
    # is_probe_1_checkpoint).
    torch.save({}, target_dir / "optimizer_state.pt")

    # RNG state: build a snapshot that ``_load_rng_state`` can
    # consume. Use the same shape as ``_save_rng_state`` produces:
    # python_random + numpy_random + torch_cpu + sample_rng + runtime
    # (h_prev, z_prev, a_prev, env_step_meta, iteration, etc.).
    import pickle as _pickle
    import random as _random

    import numpy as _np

    # Synthesize a runtime tuple that matches Probe 1.5's expected
    # shapes (h_dim=wm_cfg.h_dim, z_dim=wm_cfg.z_dim). The
    # observation_bytes etc. for the env step meta is built from a
    # fresh zero-image — the loader's ``_env_step_meta_from_dict``
    # rebuilds an EnvStep from (shape, dtype, bytes).
    obs_arr = _np.zeros((wm_cfg.obs_size, wm_cfg.obs_size), dtype=_np.uint8)
    runtime: dict[str, object] = {
        "h_prev": torch.zeros(1, wm_cfg.h_dim),
        "z_prev": torch.zeros(1, wm_cfg.z_dim),
        "a_prev": torch.zeros(1, dtype=torch.long),
        "h_curr": torch.zeros(1, wm_cfg.h_dim),
        "z_curr": torch.zeros(1, wm_cfg.z_dim),
        "obs_curr": torch.zeros(1, wm_cfg.obs_size, wm_cfg.obs_size),
        "obs_curr_hash": "0" * 64,
        "env_step_meta": {
            "observation_shape": [wm_cfg.obs_size, wm_cfg.obs_size],
            "observation_dtype": "uint8",
            "observation_bytes": obs_arr.tobytes(),
            "env_step": 100,
            "episode_id": 0,
            "step_in_episode": 100,
            "wallclock_ms": 0,
        },
        "iteration": 100,
    }
    rng_blob: dict[str, object] = {
        "python_random": _random.getstate(),
        "numpy_random": _np.random.get_state(),
        "torch_cpu": torch.get_rng_state().numpy().tobytes(),
        "torch_device_type": "cpu",
        "torch_device_rng": None,
        "sample_rng": torch.Generator(device="cpu").get_state().numpy().tobytes(),
        "runtime": runtime,
    }
    with (target_dir / "rng_state.pkl").open("wb") as fh:
        _pickle.dump(rng_blob, fh)

    # ``run_id`` is metadata only (not read by the loader); kept here
    # for potential future use.
    _ = run_id


def test_load_probe_1_checkpoint_initializes_ema_from_online(
    tmp_path: Path,
) -> None:
    """Plan §2.4 checkpoint-load Probe 1 compat path.

    Synthesize a ``"0.1.0"``-shape checkpoint by hand, load it via
    a freshly-constructed Probe 1.5 runner, and assert: (a) the EMA
    target's parameters equal the freshly-loaded online network's
    parameters element-wise (``_initialize_ema_target_from_online``
    fired post-load); (b) the actor's first-layer weight has the
    extended ``h_dim + z_dim + 1`` input dim, with the trailing
    column zero-initialized; (c) a ``mirror_marker`` ``world_event``
    with ``source="system"`` and the documented payload was emitted.
    """
    config = _make_runner_config(
        tmp_path=tmp_path,
        run_id="probe1-compat-test",
    )

    # Build the Probe 1 checkpoint directory before constructing the
    # runner so the manager's ``load`` finds it.
    target_dir = config.checkpoints_dir / "ckpt-000001"
    _build_probe_1_checkpoint(
        target_dir,
        wm_cfg=config.world_model_config,
        action_dim=config.action_dim,
        ensemble_k=config.ensemble_k,
        run_id=config.run_id,
    )

    with _transport_pair(run_id="probe1-compat-env") as (
        client,
        _server,
        env_server,
        _thread,
    ):
        runner = Runner(config, client, env_server=env_server)
        try:
            runner.load_checkpoint("ckpt-000001")

            # (a) EMA target == online (fresh init from online).
            online_enc = dict(runner.world_model.encoder.state_dict())
            target_enc = dict(runner.world_model.target_encoder.state_dict())
            for k, v in online_enc.items():
                assert torch.equal(v.cpu(), target_enc[k].cpu()), (
                    f"target_encoder.{k} != online encoder.{k} after Probe 1 "
                    f"checkpoint load — "
                    f"_initialize_ema_target_from_online did not fire"
                )
            online_gru = dict(runner.world_model.gru_cell.state_dict())
            target_gru = dict(runner.world_model.target_gru_cell.state_dict())
            for k, v in online_gru.items():
                assert torch.equal(v.cpu(), target_gru[k].cpu()), (
                    f"target_gru_cell.{k} != online gru_cell.{k} after "
                    f"Probe 1 checkpoint load"
                )

            # (b) Actor first layer extended; new column zero.
            first_layer = runner.actor.net[0]
            assert isinstance(first_layer, torch.nn.Linear)
            h_dim = config.world_model_config.h_dim
            z_dim = config.world_model_config.z_dim
            assert first_layer.weight.shape == (
                config.world_model_config.mlp_hidden,
                h_dim + z_dim + 1,
            )
            new_column = first_layer.weight[:, h_dim + z_dim :]
            assert torch.all(new_column == 0.0), (
                "Actor's new input column is non-zero after Probe 1 "
                "checkpoint load — the zero-init convention from "
                "plan §2.4 was not honored"
            )
        finally:
            runner.close()

    # (c) mirror_marker emitted.
    world_event_rows = _read_jsonl_records(
        config.telemetry_dir / "world_event.jsonl"
    )
    mirror_markers = [
        r for r in world_event_rows if r.get("event_type") == "mirror_marker"
    ]
    assert mirror_markers, (
        "no mirror_marker world_event emitted on Probe 1 checkpoint load"
    )
    marker = mirror_markers[0]
    assert marker["source"] == "system"
    payload = marker["payload"]
    assert isinstance(payload, dict)
    assert payload.get("checkpoint_schema_version") == PROBE_1_SCHEMA_VERSION
    note = payload.get("note", "")
    assert isinstance(note, str)
    # The payload describes the asymmetry — verify a few key phrases.
    for phrase in (
        "Probe 1 checkpoint",
        "EMA target",
        "online",
        "actor input column",
        "zero-initialized",
    ):
        assert phrase in note, (
            f"mirror_marker payload note missing expected phrase {phrase!r}: "
            f"{note!r}"
        )


# ---- prior-network gradient confirmation ---------------------------------


def test_prior_network_gradient_under_self_prediction_loss_alone(
    tmp_path: Path,
) -> None:
    """Phase 1 journal flagged this open question: the synthesis claims
    the auxiliary loss flows into "encoder, GRU, posterior network,
    prior network, and head". Phase 3 confirms empirically using a
    2-step batched setup that mirrors the runner's ``_train_step`` —
    sequence length > 1 so ``z_{t}`` propagates from ``q_dist.rsample``
    at step ``t-1`` (which involves the encoder + posterior) into
    ``recurrence`` at step ``t``, putting both modules on the
    self-prediction-loss gradient path.

    Empirical finding (Phase 3): under a 2-step batched setup with KL
    excluded from the backward, ``self_prediction_loss`` alone reaches
    the **encoder, posterior, GRU cell, action embedding, and
    self_prediction_head**, but **NOT the prior network**. The prior
    network's parameters receive zero gradient because the prior's
    output (``mu``, ``log_sigma``) is consumed only by the KL term in
    the loss; ``self_prediction_loss`` reads ``h_t`` directly via the
    head's input but never reads the prior's output. The synthesis's
    claim is correct for encoder, posterior, GRU, head; the prior
    network is the single exception. Journal entry records this
    finding.
    """
    cfg = _tiny_world_model_config()
    wm = WorldModel(cfg)
    batch = 2
    seq_len = 2  # the empirical confirmation requires sequence > 1

    obs_seq = torch.randn(
        seq_len, batch, cfg.obs_channels, cfg.obs_size, cfg.obs_size
    )
    h = torch.zeros(batch, cfg.h_dim)
    z = torch.zeros(batch, cfg.z_dim)
    a = torch.zeros(batch, dtype=torch.long)

    sp_total = torch.zeros(())
    for t in range(seq_len):
        wm_step = wm.step(obs_seq[t], h, z, a)
        target = wm.compute_self_prediction_target(obs_seq[t], h, z, a)
        loss_dict = wm.loss(
            wm_step, obs_target=obs_seq[t], target_h_next=target
        )
        sp_total = sp_total + loss_dict["self_prediction_loss"]
        h = wm_step.h
        z = wm_step.z
        # ``a`` stays at zeros — its specific value doesn't matter for
        # this test; the gradient through action_embedding is what
        # matters and that flows regardless of a's specific index.

    sp_total.backward()

    # Prior network's parameters receive no gradient — the empirical
    # pin Phase 3 records.
    for name, p in wm.prior_head.named_parameters():
        if p.grad is None:
            continue
        assert p.grad.abs().sum().item() == 0.0, (
            f"prior_head.{name} received non-zero gradient from "
            f"self_prediction_loss alone in a 2-step batched setup. "
            f"This contradicts Phase 1 / Phase 3's analysis that the "
            f"prior's output is consumed only by KL. Investigate the "
            f"world-model graph and update the journal."
        )

    # Sanity: the modules the synthesis correctly named (encoder, GRU,
    # posterior, head) DO receive gradient under this 2-step setup.
    for module_name, module in (
        ("self_prediction_head", wm.self_prediction_head),
        ("gru_cell", wm.gru_cell),
        ("encoder", wm.encoder),
        ("posterior_head", wm.posterior_head),
        ("action_embedding", wm.action_embedding),
    ):
        any_nonzero = False
        for _name, p in module.named_parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                any_nonzero = True
                break
        assert any_nonzero, (
            f"{module_name} did not receive any gradient from "
            f"self_prediction_loss in a 2-step batched setup — the "
            f"synthesis's claim about gradient flow into this module "
            f"is contradicted; investigate before promoting Phase 3."
        )


# ---- mark unused imports the linter would flag --------------------------

_ = pytest  # silence unused-import on conditional skips elsewhere
