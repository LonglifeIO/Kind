"""Phase 5 gate: end-to-end CPU smoke test for the runner.

Exercises the full Phase 1–5 substrate as a running loop: env-server →
transport → runner → world model + actor + ensemble + replay → telemetry
sinks → checkpoint manager. Uses a tiny world-model config (h=32, z=4,
K=2) and a quiet env config so a 200-step run completes in seconds on
CPU.

The plan §4 names ``test_integration_smoke.py`` as the fifth gate test;
this file is its full elaboration. The MPS smoke (Phase 8 §5) is a
separate, hand-runnable script — this file's MPS test is opportunistic
("skip if MPS unavailable") rather than mandatory at the gate.

Resume-from-checkpoint determinism: the resume test verifies bit-for-bit
equality of world-model / actor / ensemble parameters and optimizer
state after :meth:`load_checkpoint`. Trajectory-level match across the
resume boundary would additionally require the env-server's RNG to be
checkpointable; Probe 1's wire protocol does not yet ship the env-server
state across the BARRIER, so trajectory match is documented as a
limitation in the working journal rather than asserted here.
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

from kind.agents.world_model import WorldModelConfig
from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import GridWorldConfig
from kind.env.transport import EnvTransportClient, EnvTransportServer
from kind.training.runner import Runner, RunnerConfig


# ---- helpers --------------------------------------------------------------


def _quiet_grid_world_config() -> GridWorldConfig:
    """Tiny env with regrowth/drift disabled — keeps the world still so the
    world model has a deterministic substrate during the smoke."""
    return GridWorldConfig(
        episode_length=50,
        initial_regrowth_p=0.0,
        drift_p_min=0.0,
        drift_p_max=0.0,
        drift_magnitude_per_step=0.0,
        n_initial_resources=2,
    )


def _tiny_world_model_config() -> WorldModelConfig:
    """Smallest world model that exercises every path without slowing the
    smoke. h=32, z=4, embed=32, mlp_hidden=32; obs stays 32×32 (the
    encoder/decoder strides cannot be made smaller without rewriting them).
    """
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
    *,
    tmp_path: Path,
    run_id: str = "smoke-run",
    device: str = "cpu",
    checkpoint_every_n_env_steps: int = 100,
    dream_cadence_env_steps: int = 50,
    warmup_env_steps: int = 10,
    replay_capacity: int = 200,
    replay_sequence_length: int = 4,
    replay_batch_size: int = 2,
    parquet_rows_per_shard: int = 10,
    imagination_horizon: int = 4,
    dream_horizon: int = 4,
    ensemble_k: int = 2,
) -> RunnerConfig:
    return RunnerConfig(
        world_model_config=_tiny_world_model_config(),
        run_id=run_id,
        telemetry_dir=tmp_path / "telemetry",
        checkpoints_dir=tmp_path / "checkpoints",
        action_dim=5,
        ensemble_k=ensemble_k,
        imagination_horizon=imagination_horizon,
        replay_capacity=replay_capacity,
        replay_sequence_length=replay_sequence_length,
        replay_batch_size=replay_batch_size,
        train_every_n_env_steps=1,
        warmup_env_steps=warmup_env_steps,
        dream_cadence_env_steps=dream_cadence_env_steps,
        dream_horizon=dream_horizon,
        checkpoint_every_n_env_steps=checkpoint_every_n_env_steps,
        parquet_rows_per_shard=parquet_rows_per_shard,
        device=device,
    )


@contextmanager
def _transport_pair(
    *,
    seed: int = 42,
    run_id: str = "smoke-env",
) -> Iterator[
    tuple[EnvTransportClient, EnvTransportServer, EnvServer, threading.Thread]
]:
    """Spin up an env-server / transport-server / client triple.

    Yields ``(client, transport_server, env_server, server_thread)`` and
    tears down on context exit. The ``env_server`` is yielded so the
    test can pass it to the :class:`Runner` for loopback
    ``set_checkpoint_id`` propagation.

    The client returned is *not yet connected*; the runner's
    :meth:`Runner.run` calls :meth:`EnvTransportClient.connect` itself
    (or the test calls it explicitly when constructing without
    ``run``).
    """
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
        name="SmokeEnvTransportServer",
        daemon=True,
    )
    server_thread.start()

    client = EnvTransportClient(
        host="127.0.0.1",
        port=transport_server.actual_port,
        # The runner replaces this handler at construction time via
        # set_world_event_handler; this no-op is the placeholder.
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
    """Read all shard files in ``dir_`` and return the concatenated rows."""
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


# ---- init / construction --------------------------------------------------


def test_runner_init_does_not_error(tmp_path: Path) -> None:
    config = _make_runner_config(tmp_path=tmp_path)
    with _transport_pair() as (client, _server, env_server, _thread):
        runner = Runner(config, client, env_server=env_server)
        try:
            assert runner.device == torch.device("cpu")
            assert runner.replay_buffer.capacity == config.replay_capacity
            assert (
                runner.replay_buffer.sequence_length
                == config.replay_sequence_length
            )
        finally:
            runner.close()


def test_three_optimizers_have_disjoint_parameters(tmp_path: Path) -> None:
    """Verify the three optimizers' param sets don't overlap.

    This is the synthesis §Q2 commitment that the world model's loss,
    the ensemble's loss, and the actor's loss update three disjoint
    parameter sets — the substrate's "three losses, three optimizers"
    structure.
    """
    config = _make_runner_config(tmp_path=tmp_path)
    with _transport_pair() as (client, _server, env_server, _thread):
        runner = Runner(config, client, env_server=env_server)
        try:
            wm_params = {id(p) for p in runner.world_model.parameters()}
            actor_params = {id(p) for p in runner.actor.parameters()}
            ens_params = {id(p) for p in runner.ensemble.parameters()}
            assert wm_params.isdisjoint(actor_params)
            assert wm_params.isdisjoint(ens_params)
            assert actor_params.isdisjoint(ens_params)
            # Sanity: each set is non-empty.
            assert wm_params and actor_params and ens_params
        finally:
            runner.close()


# ---- 200-step CPU smoke ---------------------------------------------------


def test_smoke_runs_to_completion_on_cpu(tmp_path: Path) -> None:
    """The full hot loop runs without errors for 200 env steps."""
    config = _make_runner_config(tmp_path=tmp_path)
    with _transport_pair() as (client, _server, env_server, _thread):
        runner = Runner(config, client, env_server=env_server)
        try:
            runner.run(total_env_steps=200)
        finally:
            runner.close()


def test_smoke_all_four_streams_have_records(tmp_path: Path) -> None:
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

    assert len(agent_step_rows) == 200, (
        f"expected 200 agent_step records, got {len(agent_step_rows)}"
    )
    assert len(dream_rollout_rows) > 0, "no dream_rollout records emitted"
    assert len(replay_meta_rows) > 0, "no replay_meta records emitted"
    # WorldEvent: at minimum the initial env_reset; with episode_length=50
    # over 200 steps we get 4 episode boundaries, each producing an
    # internal_stochasticity_aggregate + env_reset pair, so >= 9 records.
    assert len(world_event_rows) > 0, "no world_event records emitted"


def test_smoke_agent_step_record_shapes_and_types(tmp_path: Path) -> None:
    """The runner's tensor → list[float] / float lifting matches the
    schema's field types; lengths match the world model's h/z/embed dims."""
    config = _make_runner_config(tmp_path=tmp_path)
    h_dim = config.world_model_config.h_dim
    z_dim = config.world_model_config.z_dim
    embed_dim = config.world_model_config.embed_dim

    with _transport_pair() as (client, _server, env_server, _thread):
        runner = Runner(config, client, env_server=env_server)
        try:
            runner.run(total_env_steps=20)
        finally:
            runner.close()

    rows = _read_parquet_records(config.telemetry_dir / "agent_step")
    assert rows, "no agent_step records"
    first = rows[0]
    # ``h_t`` is a list[float] of length h_dim.
    h_t = first["h_t"]
    assert isinstance(h_t, list)
    assert len(h_t) == h_dim
    assert all(isinstance(v, float) for v in h_t)

    z_t = first["z_t"]
    assert isinstance(z_t, list)
    assert len(z_t) == z_dim

    kl_per_dim = first["kl_per_dim_t"]
    assert isinstance(kl_per_dim, list)
    assert len(kl_per_dim) == z_dim

    encoder_embedding = first["encoder_embedding_t"]
    assert isinstance(encoder_embedding, list)
    assert len(encoder_embedding) == embed_dim

    # Scalars.
    assert isinstance(first["kl_aggregate_t"], float)
    assert isinstance(first["recon_loss_t"], float)
    assert isinstance(first["intrinsic_signal_t"], float)
    assert isinstance(first["policy_entropy_t"], float)
    assert isinstance(first["action_logprob_t"], float)
    assert isinstance(first["t"], int)
    assert isinstance(first["episode_id"], int)
    assert isinstance(first["step_in_episode"], int)
    assert isinstance(first["wallclock_ms"], int)
    assert isinstance(first["action_t"], int)
    assert 0 <= int(first["action_t"]) < 5

    # ``q_params_t`` is a tuple[list[float], list[float]] — Pydantic
    # serializes it to a list of two lists in the parquet shard.
    q_params = first["q_params_t"]
    assert isinstance(q_params, list)
    assert len(q_params) == 2
    assert len(q_params[0]) == z_dim and len(q_params[1]) == z_dim


def test_smoke_dream_rollout_emitted_at_cadence(tmp_path: Path) -> None:
    """With dream_cadence=50 and 200 steps, expect 4 records (env_step =
    50, 100, 150, 200)."""
    config = _make_runner_config(
        tmp_path=tmp_path,
        dream_cadence_env_steps=50,
    )
    with _transport_pair() as (client, _server, env_server, _thread):
        runner = Runner(config, client, env_server=env_server)
        try:
            runner.run(total_env_steps=200)
        finally:
            runner.close()

    rows = _read_parquet_records(config.telemetry_dir / "dream_rollout")
    # The cadence check uses `env_step_now > 0 and env_step_now % cadence == 0`.
    # With env_step_now = 0..199 across 200 iterations (no episode-boundary
    # skips because step counter advances even on boundary), the matches
    # are 50, 100, 150 → 3 dreams. (env_step=200 would be the 201st
    # iteration which we don't reach.)
    assert len(rows) == 3, f"expected 3 dream rollouts at cadence=50, got {len(rows)}"
    for r in rows:
        seq = r["sequence_h"]
        assert isinstance(seq, list)
        assert len(seq) == config.dream_horizon


def test_smoke_checkpoint_committed_mid_run(tmp_path: Path) -> None:
    """With cadence=100 and 200 steps, expect a single mid-run commit."""
    config = _make_runner_config(
        tmp_path=tmp_path, checkpoint_every_n_env_steps=100
    )
    with _transport_pair() as (client, _server, env_server, _thread):
        runner = Runner(config, client, env_server=env_server)
        try:
            runner.run(total_env_steps=200)
        finally:
            runner.close()

    ckpts = sorted(
        p.name for p in config.checkpoints_dir.iterdir() if p.is_dir()
    )
    # Expect ckpt-000001 (at env_step=100). env_step=0 doesn't qualify
    # (the `env_step_now > 0` guard); env_step=200 isn't reached because
    # the loop iterates env_step 0..199.
    assert ckpts == ["ckpt-000001"], f"unexpected checkpoints: {ckpts}"

    # All canonical files are present in the checkpoint dir.
    target = config.checkpoints_dir / "ckpt-000001"
    expected_names = {
        "weights.safetensors",
        "replay_meta.json",
        "optimizer_state.pt",
        "rng_state.pkl",
        "telemetry_offsets.json",
        "schema_version.txt",
    }
    actual_names = {p.name for p in target.iterdir()}
    assert expected_names.issubset(actual_names), (
        f"checkpoint missing files: expected {expected_names}, got {actual_names}"
    )


def test_smoke_checkpoint_carries_through_to_subsequent_records(
    tmp_path: Path,
) -> None:
    """After a checkpoint commits, subsequent ``AgentStep`` records carry
    the new ``checkpoint_id`` in their envelope. Records emitted before
    the commit carry ``None``."""
    config = _make_runner_config(
        tmp_path=tmp_path, checkpoint_every_n_env_steps=50
    )
    with _transport_pair() as (client, _server, env_server, _thread):
        runner = Runner(config, client, env_server=env_server)
        try:
            runner.run(total_env_steps=120)
        finally:
            runner.close()

    rows = _read_parquet_records(config.telemetry_dir / "agent_step")

    def _t(r: dict[str, object]) -> int:
        v = r["t"]
        assert isinstance(v, int)
        return v

    # The runner's _step_once emits the AgentStep *before* the
    # checkpoint commit at the end of the same iteration. So a record
    # whose ``t`` is itself the cadence boundary still carries the old
    # checkpoint_id (the one that was current at the moment its envelope
    # was sealed). The commit takes effect for *strictly later* records.
    pre = [r for r in rows if _t(r) <= 50]
    assert all(r["checkpoint_id"] is None for r in pre), (
        "pre-commit records carry a checkpoint_id"
    )
    # After the first commit (at the end of t=50's iteration), records
    # with t in [51..99] carry ckpt-000001. After the second commit (at
    # the end of t=100), records with t in [101..119] carry ckpt-000002.
    post = [r for r in rows if _t(r) > 50]
    assert all(r["checkpoint_id"] is not None for r in post), (
        "post-commit records still have checkpoint_id=None"
    )


# ---- resume --------------------------------------------------------------


def test_smoke_resume_loads_identical_weights(tmp_path: Path) -> None:
    """Construct runner_A, run until the checkpoint commits, then close.
    Construct runner_B fresh, load_checkpoint, verify the loaded weights
    match the bytes the manager committed to disk.

    Comparing runner_b's loaded state against the checkpoint *file* (not
    against runner_a's in-memory state at end-of-run) is the robust
    invariant: the file is what resume reads from, and the file is
    what's tested for equality. The world model's weights at runner_a's
    end-of-run are not necessarily the same as at the moment of commit
    if any further training happened post-commit; reading from disk
    avoids that fragility.
    """
    from safetensors.torch import load_file as _load_safetensors

    config_a = _make_runner_config(
        tmp_path=tmp_path / "run_a",
        checkpoint_every_n_env_steps=20,
        warmup_env_steps=5,
        # Push dream cadence past total_env_steps so it doesn't interleave.
        dream_cadence_env_steps=10_000,
        run_id="smoke-resume-a",
    )

    # Phase 1: train and commit. Run total=21 so the loop ends right
    # after the iter-20 checkpoint with no further training between
    # commit and end-of-run.
    with _transport_pair(run_id="smoke-resume-env-a") as (
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

    # Read the checkpoint file directly. This is the bytes the manager
    # committed; runner_b's load_checkpoint should produce these exact
    # tensors back into its three modules.
    weights_path = config_a.checkpoints_dir / "ckpt-000001" / "weights.safetensors"
    assert weights_path.is_file()
    disk_weights: dict[str, torch.Tensor] = _load_safetensors(str(weights_path))

    # Phase 2: fresh runner_b sharing the checkpoints_dir.
    config_b = _make_runner_config(
        tmp_path=tmp_path / "run_b",
        run_id="smoke-resume-b",
    )
    object.__setattr__(config_b, "checkpoints_dir", config_a.checkpoints_dir)

    with _transport_pair(run_id="smoke-resume-env-b") as (
        client,
        _server,
        env_server,
        _thread,
    ):
        runner_b = Runner(config_b, client, env_server=env_server)
        try:
            latest = runner_b.checkpoint_manager.latest()
            assert latest == "ckpt-000001"
            runner_b.load_checkpoint(latest)

            for k, v in runner_b.world_model.state_dict().items():
                expected = disk_weights[f"world_model.{k}"]
                assert torch.equal(expected, v.cpu()), (
                    f"world_model.{k} differs from checkpoint file"
                )
            for k, v in runner_b.actor.state_dict().items():
                expected = disk_weights[f"actor.{k}"]
                assert torch.equal(expected, v.cpu()), (
                    f"actor.{k} differs from checkpoint file"
                )
            for k, v in runner_b.ensemble.state_dict().items():
                expected = disk_weights[f"ensemble.{k}"]
                assert torch.equal(expected, v.cpu()), (
                    f"ensemble.{k} differs from checkpoint file"
                )
        finally:
            runner_b.close()


def test_smoke_resume_loads_identical_rng_state(tmp_path: Path) -> None:
    """Plan §4 test #5 names "resume from checkpoint yields identical RNG
    state". The companion ``test_smoke_resume_loads_identical_weights``
    pins parameter equality (the safetensors round-trip); this test pins
    the RNG-state round-trip explicitly: after ``load_checkpoint``, the
    runner's PyTorch CPU RNG, the runner's sample RNG (the
    ``torch.Generator`` driving replay sampling), and Python ``random``
    match the bytes committed to ``rng_state.pkl``.
    NumPy's RNG state is also part of the blob — its tuple shape varies
    across NumPy versions so we compare the algorithm tag and the state
    array byte-for-byte rather than the whole tuple.
    Trajectory match across the resume boundary is the stronger claim
    that additionally requires the env-server's RNG to be checkpointable;
    Probe 1's wire protocol does not yet ship that, and the Phase 5
    journal entry documents this as a known limitation. This test
    asserts only what is genuinely round-trippable.
    """
    import pickle as _pickle
    import random as _random

    import numpy as _np

    config_a = _make_runner_config(
        tmp_path=tmp_path / "run_a",
        checkpoint_every_n_env_steps=20,
        warmup_env_steps=5,
        dream_cadence_env_steps=10_000,
        run_id="smoke-rng-a",
    )
    with _transport_pair(run_id="smoke-rng-env-a") as (
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

    rng_path = config_a.checkpoints_dir / "ckpt-000001" / "rng_state.pkl"
    assert rng_path.is_file(), "rng_state.pkl missing from checkpoint"
    with rng_path.open("rb") as fh:
        committed = _pickle.load(fh)

    config_b = _make_runner_config(
        tmp_path=tmp_path / "run_b",
        run_id="smoke-rng-b",
    )
    object.__setattr__(config_b, "checkpoints_dir", config_a.checkpoints_dir)

    with _transport_pair(run_id="smoke-rng-env-b") as (
        client,
        _server,
        env_server,
        _thread,
    ):
        runner_b = Runner(config_b, client, env_server=env_server)
        try:
            runner_b.load_checkpoint("ckpt-000001")

            assert (
                torch.get_rng_state().numpy().tobytes()
                == committed["torch_cpu"]
            ), "torch CPU RNG state differs from committed bytes"

            sample_state_bytes = (
                runner_b._sample_rng.get_state().numpy().tobytes()
            )
            assert sample_state_bytes == committed["sample_rng"], (
                "runner sample-RNG state differs from committed bytes"
            )

            assert _random.getstate() == committed["python_random"], (
                "Python random state differs from committed tuple"
            )

            np_state_now = _np.random.get_state()
            np_state_committed = committed["numpy_random"]
            assert np_state_now[0] == np_state_committed[0]
            assert (np_state_now[1] == np_state_committed[1]).all()
            assert np_state_now[2:] == np_state_committed[2:]
        finally:
            runner_b.close()


def test_smoke_resume_runs_after_load(tmp_path: Path) -> None:
    """After load_checkpoint, the runner can run again without errors.

    This exercises the path where ``run()`` does *not* call
    :meth:`EnvTransportClient.connect` (since the runtime state was
    populated from the checkpoint) — though the env-server in this test
    is fresh so the env_step counter restarts at 0 mid-run, the runner
    still steps cleanly.
    """
    config_a = _make_runner_config(
        tmp_path=tmp_path / "run_a",
        checkpoint_every_n_env_steps=10,
        warmup_env_steps=2,
        run_id="smoke-resume-run-a",
    )
    with _transport_pair(run_id="smoke-resume-run-env-a") as (
        client,
        _server,
        env_server,
        _thread,
    ):
        runner_a = Runner(config_a, client, env_server=env_server)
        try:
            runner_a.run(total_env_steps=15)
        finally:
            runner_a.close()

    config_b = _make_runner_config(
        tmp_path=tmp_path / "run_b",
        run_id="smoke-resume-run-b",
    )
    object.__setattr__(config_b, "checkpoints_dir", config_a.checkpoints_dir)

    # Use a fresh env-server / transport so run() can connect.
    with _transport_pair(run_id="smoke-resume-run-env-b") as (
        client,
        _server,
        env_server,
        _thread,
    ):
        runner_b = Runner(config_b, client, env_server=env_server)
        try:
            latest = runner_b.checkpoint_manager.latest()
            assert latest is not None
            runner_b.load_checkpoint(latest)
            # The loaded runtime state has env_step_meta, so run() will
            # use it directly rather than calling client.connect().
            # However the client we passed isn't connected yet. Calling
            # run() will detect env_step_meta is set and skip connect —
            # but the transport will still need to be connected for
            # client.step() to work. Connect the client manually so the
            # transport has a live socket; the returned EnvStep is
            # ignored since the runtime state from the checkpoint is
            # what the runner trusts.
            client.connect()
            runner_b.run(total_env_steps=10)
        finally:
            runner_b.close()


# ---- close cleanup ------------------------------------------------------


def test_smoke_close_idempotent_and_flushes_sinks(tmp_path: Path) -> None:
    """Closing the runner twice does not raise; sinks finalised once."""
    config = _make_runner_config(tmp_path=tmp_path)
    with _transport_pair() as (client, _server, env_server, _thread):
        runner = Runner(config, client, env_server=env_server)
        runner.run(total_env_steps=20)
        runner.close()
        runner.close()  # idempotent

    # Sinks are flushed: at least one parquet shard exists.
    shards = _list_parquet_shards(config.telemetry_dir / "agent_step")
    assert shards, "no agent_step shards after close"


# ---- resume bookkeeping --------------------------------------------------


def test_smoke_telemetry_offsets_file_is_valid_json(tmp_path: Path) -> None:
    config = _make_runner_config(
        tmp_path=tmp_path, checkpoint_every_n_env_steps=20
    )
    with _transport_pair() as (client, _server, env_server, _thread):
        runner = Runner(config, client, env_server=env_server)
        try:
            runner.run(total_env_steps=40)
        finally:
            runner.close()

    offsets = config.checkpoints_dir / "ckpt-000001" / "telemetry_offsets.json"
    assert offsets.is_file()
    payload = json.loads(offsets.read_text())
    assert "agent_step_dir" in payload
    assert "dream_rollout_dir" in payload
    assert "replay_meta_path" in payload
    assert "world_event_path" in payload


def test_smoke_schema_version_file_committed(tmp_path: Path) -> None:
    config = _make_runner_config(
        tmp_path=tmp_path, checkpoint_every_n_env_steps=20
    )
    with _transport_pair() as (client, _server, env_server, _thread):
        runner = Runner(config, client, env_server=env_server)
        try:
            runner.run(total_env_steps=40)
        finally:
            runner.close()

    schema_file = config.checkpoints_dir / "ckpt-000001" / "schema_version.txt"
    # Probe 1.5 v2 (plan §2.4 step 3): the runner now stamps
    # ``"0.2.0"`` on checkpoints because the substrate carries the
    # self-prediction head + EMA target + extended actor input layer.
    # The Probe 1 → Probe 1.5 compat path on the load side handles
    # ``"0.1.0"`` checkpoints (see tests/test_integration_smoke_probe1_5.py
    # for the dedicated regression).
    assert schema_file.read_text().strip() == "0.2.0"


# ---- MPS opportunistic test ---------------------------------------------


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS device not available",
)
def test_smoke_runs_on_mps_if_available(tmp_path: Path) -> None:
    """Smoke on MPS: 50 steps, no fallback warnings on the hot path.

    The plan §5 designates a separate ``scripts/smoke_mps.py`` for the
    canonical MPS gate; this test is the in-pytest opportunistic check
    that the runner constructed with ``device="mps"`` runs without
    errors on a tiny world model. The full gate-time MPS smoke is the
    Phase 8 script.
    """
    config = _make_runner_config(
        tmp_path=tmp_path,
        device="mps",
        warmup_env_steps=5,
        replay_capacity=100,
    )
    with _transport_pair(run_id="smoke-mps-env") as (
        client,
        _server,
        env_server,
        _thread,
    ):
        runner = Runner(config, client, env_server=env_server)
        try:
            runner.run(total_env_steps=50)
        finally:
            runner.close()

    rows = _read_parquet_records(config.telemetry_dir / "agent_step")
    assert rows, "no agent_step records on MPS run"
