"""Phase 4 gate test #7 — Probe 2 v2 lesion scaffold (CPU smokes +
checkpoint mutation script).

The synthesis §2.4 element 4 / plan §2.5 / plan §4 gate test 7. Five
lesion mechanisms operate at five different points in the substrate:

* ``"disable_self_prediction"`` — substrate-side; ``WorldModel.step``
  emits zeros from the head, ``_update_ema_target`` is a no-op, the
  auxiliary loss is identically zero. Tests below verify the head's
  parameters do not move during training (a regression check between
  two checkpoints) and ``self_prediction_t`` records identically zero
  across all dims.
* ``"zero_or_randomize_scalar"`` (zero) — behavior-side; ``views.split``
  forces ``PolicyView.self_prediction_error = 0.0`` and the masked
  flag True. The head and EMA target continue to train normally so
  the substrate-side reads exactly like the un-lesioned run.
* ``"zero_or_randomize_scalar"`` (randomize) — behavior-side; same as
  above except the scalar is drawn from
  ``Uniform(empirical_min, empirical_max)``. Per-step values differ;
  masked flag is still True.
* ``"init_zero_scalar_column"`` — capacity-as-init-shape; not a runtime
  flag but a checkpoint mutation produced by
  ``scripts/probe2_lesion_init_zero_scalar_column.py``. The lesioned
  checkpoint has the actor's scalar column zeroed; every other tensor
  is byte-identical to the source.
* ``"ensemble_k1"`` / ``"ensemble_constant"`` — substrate-side; v1
  candidates preserved as alternatives. Both produce
  ``intrinsic_signal_t == 0`` on every step.

Tiny CPU sizes throughout (``h=32``, ``z=4``, ``K=2``) so the
integration smokes run fast.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
import torch
from safetensors.torch import load_file as _load_safetensors
from safetensors.torch import save_file as _save_safetensors

from kind.agents.actor import Actor
from kind.agents.world_model import WorldModelConfig
from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import GridWorldConfig
from kind.env.transport import EnvTransportClient, EnvTransportServer
from kind.observer.schemas import SCHEMA_VERSION
from kind.training.runner import Runner, RunnerConfig

# Module under test for the script: imported directly so the test can
# call :func:`mutate` (the side-effect-free programmatic entry) and
# :func:`main` (the CLI entry) without round-tripping through subprocess.
_SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPT_DIR))
import probe2_lesion_init_zero_scalar_column as lesion_script  # noqa: E402


# ---- shared fixtures (parallels test_integration_smoke_probe1_5.py) ------


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
    """Phase 4 gate test sizes: ``h=32``, ``z=4``, ``K=2``,
    ``head_hidden=32``. Small enough for a 60-step CPU smoke; large
    enough that the actor's scalar column is non-trivial under
    small-Gaussian init."""
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
    run_id: str = "smoke-lesion-run",
    checkpoint_every_n_env_steps: int = 60,
    dream_cadence_env_steps: int = 50,
    warmup_env_steps: int = 5,
    lesion_kind: Any = None,
    lesion_zero_or_randomize_variant: Any = "zero",
    lesion_zero_or_randomize_empirical_min: float = 0.0,
    lesion_zero_or_randomize_empirical_max: float = 1.0,
    ensemble_k: int = 2,
) -> RunnerConfig:
    return RunnerConfig(
        world_model_config=_tiny_world_model_config(),
        run_id=run_id,
        telemetry_dir=tmp_path / "telemetry",
        checkpoints_dir=tmp_path / "checkpoints",
        action_dim=5,
        ensemble_k=ensemble_k,
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
        lesion_kind=lesion_kind,
        lesion_zero_or_randomize_variant=lesion_zero_or_randomize_variant,
        lesion_zero_or_randomize_empirical_min=(
            lesion_zero_or_randomize_empirical_min
        ),
        lesion_zero_or_randomize_empirical_max=(
            lesion_zero_or_randomize_empirical_max
        ),
    )


@contextmanager
def _transport_pair(
    *,
    seed: int = 42,
    run_id: str = "smoke-lesion-env",
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
        name="LesionTestEnvTransportServer",
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


def _run_smoke(
    config: RunnerConfig, *, total_env_steps: int = 60
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run a short integration smoke and return (agent_step rows,
    world_event rows). The transport pair seed is the same across every
    invocation so two un-lesioned runs at the same config produce
    byte-identical telemetry; the lesion test cases compare against
    the un-lesioned reference at this granularity."""
    with _transport_pair(run_id=f"{config.run_id}-env") as (
        client,
        _server,
        env_server,
        _thread,
    ):
        torch.manual_seed(0)
        runner = Runner(config, client, env_server=env_server)
        try:
            runner.run(total_env_steps=total_env_steps)
        finally:
            runner.close()

    agent_step_rows = _read_parquet_records(config.telemetry_dir / "agent_step")
    world_event_rows = _read_jsonl_records(
        config.telemetry_dir / "world_event.jsonl"
    )
    return agent_step_rows, world_event_rows


# ---- 1. disable_self_prediction smoke ------------------------------------


def test_disable_self_prediction_zero_head_output(tmp_path: Path) -> None:
    """``self_prediction_t`` is identically zero across all dims under
    the ``"disable_self_prediction"`` lesion."""
    config = _make_runner_config(
        tmp_path=tmp_path,
        run_id="lesion-disable-sp",
        lesion_kind="disable_self_prediction",
    )
    rows, _ = _run_smoke(config)
    assert rows
    h_dim = config.world_model_config.h_dim
    for r in rows:
        sp_t = r["self_prediction_t"]
        assert isinstance(sp_t, list)
        assert len(sp_t) == h_dim
        # Every dim is exactly 0.0 — the head emits a zero tensor in
        # ``WorldModel.step`` under the lesion.
        assert all(v == 0.0 for v in sp_t), (
            f"self_prediction_t at t={r['t']} contains non-zero values "
            f"under disable_self_prediction lesion"
        )


def test_disable_self_prediction_zero_pair_scalar(tmp_path: Path) -> None:
    """``self_prediction_error_t`` is the loss-form's zero-pair value
    on non-masked steps: ``cosine: 1.0``, ``mse: 0.0``."""
    # Cosine variant.
    config_cos = _make_runner_config(
        tmp_path=tmp_path / "cosine",
        run_id="lesion-disable-sp-cos",
        lesion_kind="disable_self_prediction",
    )
    rows_cos, _ = _run_smoke(config_cos)
    assert rows_cos
    nonmasked_cos = [
        r for r in rows_cos if not bool(r["self_prediction_error_masked_t"])
    ]
    assert nonmasked_cos
    for r in nonmasked_cos:
        assert float(r["self_prediction_error_t"]) == 1.0, (
            f"non-masked self_prediction_error_t under cosine lesion is "
            f"{r['self_prediction_error_t']!r}, expected 1.0"
        )

    # MSE variant.
    config_mse_base = _make_runner_config(
        tmp_path=tmp_path / "mse",
        run_id="lesion-disable-sp-mse",
        lesion_kind="disable_self_prediction",
    )
    config_mse = dataclasses.replace(
        config_mse_base, self_prediction_loss_form="mse"
    )
    rows_mse, _ = _run_smoke(config_mse)
    assert rows_mse
    nonmasked_mse = [
        r for r in rows_mse if not bool(r["self_prediction_error_masked_t"])
    ]
    assert nonmasked_mse
    for r in nonmasked_mse:
        assert float(r["self_prediction_error_t"]) == 0.0, (
            f"non-masked self_prediction_error_t under mse lesion is "
            f"{r['self_prediction_error_t']!r}, expected 0.0"
        )


def test_disable_self_prediction_head_params_do_not_move(
    tmp_path: Path,
) -> None:
    """The head's parameters and the EMA target's parameters are
    byte-identical at end-of-training to their construction-time
    initialisation under the disable lesion. This is the regression
    check the prompt names: the auxiliary loss is identically zero so
    ``self_prediction_head`` receives no gradient, and
    ``_update_ema_target`` is a no-op so ``target_encoder`` /
    ``target_gru_cell`` retain their construction-time values (which
    are copies of the online network's at construction)."""
    config = _make_runner_config(
        tmp_path=tmp_path,
        run_id="lesion-disable-sp-static",
        lesion_kind="disable_self_prediction",
        # Run long enough to commit a checkpoint we can introspect.
        checkpoint_every_n_env_steps=60,
        warmup_env_steps=5,
    )
    # Capture the construction-time head + EMA target weights by
    # building a runner and snapshotting before run().
    with _transport_pair(run_id="lesion-disable-sp-static-env") as (
        client,
        _server,
        env_server,
        _thread,
    ):
        torch.manual_seed(0)
        runner = Runner(config, client, env_server=env_server)
        head_keys = [
            k
            for k in runner.world_model.state_dict()
            if k.startswith("self_prediction_head.")
        ]
        ema_keys = [
            k
            for k in runner.world_model.state_dict()
            if k.startswith("target_encoder.") or k.startswith("target_gru_cell.")
        ]
        head_before = {
            k: runner.world_model.state_dict()[k].clone() for k in head_keys
        }
        ema_before = {
            k: runner.world_model.state_dict()[k].clone() for k in ema_keys
        }
        try:
            runner.run(total_env_steps=60)
            head_after = {
                k: runner.world_model.state_dict()[k].clone() for k in head_keys
            }
            ema_after = {
                k: runner.world_model.state_dict()[k].clone() for k in ema_keys
            }
        finally:
            runner.close()

    assert head_keys, "self_prediction_head weights missing from state_dict"
    assert ema_keys, "EMA target weights missing from state_dict"
    for k in head_keys:
        assert torch.equal(head_before[k], head_after[k]), (
            f"head parameter {k!r} moved under disable_self_prediction "
            f"lesion (max abs diff "
            f"{(head_before[k] - head_after[k]).abs().max().item()!r})"
        )
    for k in ema_keys:
        assert torch.equal(ema_before[k], ema_after[k]), (
            f"EMA target parameter {k!r} moved under disable_self_"
            f"prediction lesion (max abs diff "
            f"{(ema_before[k] - ema_after[k]).abs().max().item()!r})"
        )


# ---- 2. zero_or_randomize_scalar smokes ----------------------------------


def test_zero_or_randomize_scalar_zero_variant(tmp_path: Path) -> None:
    """Under the zero variant: every recorded
    ``self_prediction_error_t`` is exactly ``0.0``;
    ``self_prediction_error_masked_t`` is True on every step. The
    head and EMA target continue to train (the substrate-side cohort
    reads the same as un-lesioned)."""
    config = _make_runner_config(
        tmp_path=tmp_path,
        run_id="lesion-z0-zero",
        lesion_kind="zero_or_randomize_scalar",
        lesion_zero_or_randomize_variant="zero",
    )
    rows, _ = _run_smoke(config)
    assert rows
    for r in rows:
        assert float(r["self_prediction_error_t"]) == 0.0, (
            f"self_prediction_error_t under zero lesion is "
            f"{r['self_prediction_error_t']!r}, expected 0.0 "
            f"(t={r['t']})"
        )
        assert bool(r["self_prediction_error_masked_t"]) is True, (
            f"masked flag under zero lesion is False (t={r['t']})"
        )


def test_zero_or_randomize_scalar_randomize_variant_distinct_values(
    tmp_path: Path,
) -> None:
    """Under the randomize variant: per-step ``self_prediction_error_t``
    values differ across steps (drawn from
    ``Uniform(empirical_min, empirical_max)``);
    ``self_prediction_error_masked_t`` is True on every step. The
    bounds are configured wide so the uniform draw is observably
    non-trivial (``[0.1, 0.9]``)."""
    config = _make_runner_config(
        tmp_path=tmp_path,
        run_id="lesion-z0-randomize",
        lesion_kind="zero_or_randomize_scalar",
        lesion_zero_or_randomize_variant="randomize",
        lesion_zero_or_randomize_empirical_min=0.1,
        lesion_zero_or_randomize_empirical_max=0.9,
    )
    rows, _ = _run_smoke(config)
    assert rows

    # Every step's masked flag is True under the lesion.
    for r in rows:
        assert bool(r["self_prediction_error_masked_t"]) is True, (
            f"masked flag under randomize lesion is False (t={r['t']})"
        )
        # The scalar lies inside the configured bounds.
        v = float(r["self_prediction_error_t"])
        assert 0.1 <= v <= 0.9, (
            f"self_prediction_error_t={v} under randomize lesion outside "
            f"configured bounds [0.1, 0.9] (t={r['t']})"
        )

    # Per-step values are not all identical — Uniform(0.1, 0.9) over
    # 60 steps with the runner's sample-RNG produces a meaningful
    # spread. The check is "at least 50% unique values" which is far
    # above what a degenerate constant override would produce.
    values = [float(r["self_prediction_error_t"]) for r in rows]
    assert len(set(values)) >= len(values) // 2, (
        f"randomize-variant scalars too clustered: "
        f"{len(set(values))} unique values out of {len(values)} steps"
    )


def test_zero_or_randomize_scalar_substrate_side_unchanged(
    tmp_path: Path,
) -> None:
    """Under the zero_or_randomize_scalar lesion, the substrate-side
    telemetry (``kl_aggregate_t``, ``recon_loss_t``,
    ``intrinsic_signal_t``, ``self_prediction_t`` — the head's full
    vector) reads close to an un-lesioned run on the substrate-side
    surface. The lesion only overrides the actor's behavior-side input;
    the head and EMA target still train, so the substrate-side cohort's
    statistics differ only via the actor's policy effect on state
    visitation, not via any direct head-side lesion.
    """
    # We don't assert byte-identity here — the actor's policy change
    # (forced 0.0 scalar input) propagates into different action
    # selection over time, which propagates into different states
    # visited. What we assert is that the head's full vector
    # ``self_prediction_t`` is *not* identically zero (the head still
    # produces non-trivial output), distinguishing this lesion from
    # ``disable_self_prediction``.
    config = _make_runner_config(
        tmp_path=tmp_path,
        run_id="lesion-z0-substrate-unchanged",
        lesion_kind="zero_or_randomize_scalar",
        lesion_zero_or_randomize_variant="zero",
    )
    rows, _ = _run_smoke(config)
    assert rows
    has_nonzero_sp_dim = False
    for r in rows:
        sp_t = r["self_prediction_t"]
        assert isinstance(sp_t, list)
        if any(v != 0.0 for v in sp_t):
            has_nonzero_sp_dim = True
            break
    assert has_nonzero_sp_dim, (
        "under zero_or_randomize_scalar lesion, self_prediction_t is "
        "all-zero everywhere — the head's output should still be "
        "non-trivial since only the actor's PolicyView is lesioned"
    )


# ---- 3. init_zero_scalar_column script tests -----------------------------


def _build_synthetic_phase7_5_checkpoint(
    target_dir: Path,
    *,
    wm_cfg: WorldModelConfig,
    action_dim: int,
    ensemble_k: int,
) -> None:
    """Hand-build a Probe-1.5 (``"0.2.0"``) checkpoint for the script
    tests. Mirrors what the runner's ``_commit_checkpoint`` writes: a
    weights blob with ``world_model.``/``actor.``/``ensemble.``
    prefixes, plus the five sidecar files. The actor's first-layer
    weight has the small-Gaussian column at index ``h_dim+z_dim``
    (Phase 7.5's init).
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    # Build models with the right shapes; small-Gaussian column is
    # set by the Actor constructor.
    from kind.agents.ensemble import LatentDisagreementEnsemble
    from kind.agents.world_model import WorldModel

    torch.manual_seed(123)
    world_model = WorldModel(wm_cfg)
    actor = Actor(
        h_dim=wm_cfg.h_dim,
        z_dim=wm_cfg.z_dim,
        action_dim=action_dim,
        mlp_hidden=wm_cfg.mlp_hidden,
    )
    ensemble = LatentDisagreementEnsemble(
        h_dim=wm_cfg.h_dim,
        z_dim=wm_cfg.z_dim,
        action_dim=action_dim,
        K=ensemble_k,
        mlp_hidden=wm_cfg.mlp_hidden,
    )

    combined: dict[str, torch.Tensor] = {}
    for k, v in world_model.state_dict().items():
        combined[f"world_model.{k}"] = v.detach().contiguous().cpu()
    for k, v in actor.state_dict().items():
        combined[f"actor.{k}"] = v.detach().contiguous().cpu()
    for k, v in ensemble.state_dict().items():
        combined[f"ensemble.{k}"] = v.detach().contiguous().cpu()
    _save_safetensors(combined, str(target_dir / "weights.safetensors"))

    (target_dir / "schema_version.txt").write_text(SCHEMA_VERSION + "\n")
    (target_dir / "replay_meta.json").write_text(
        json.dumps(
            {
                "buffer_size": 0,
                "capacity": 200,
                "sequence_length": 4,
            },
            indent=2,
        )
        + "\n"
    )
    (target_dir / "telemetry_offsets.json").write_text(
        json.dumps(
            {
                "agent_step_dir": "irrelevant",
                "dream_rollout_dir": "irrelevant",
                "replay_meta_path": "irrelevant",
                "world_event_path": "irrelevant",
            },
            indent=2,
        )
        + "\n"
    )

    # Construct minimal but legitimate Adam state dicts so the
    # runner's ``load_checkpoint`` (which calls ``Adam.load_state_dict``
    # for ``"0.2.0"`` checkpoints) does not raise on a degenerate
    # placeholder. The optimizers carry zero gradient steps; their
    # state's per-parameter ``exp_avg`` / ``exp_avg_sq`` tensors are
    # absent and PyTorch tolerates that for an Adam optimizer that
    # has not yet stepped.
    wm_opt = torch.optim.Adam(world_model.parameters(), lr=1e-4)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=4e-5)
    ens_opt = torch.optim.Adam(ensemble.parameters(), lr=4e-5)
    torch.save(
        {
            "world_model": wm_opt.state_dict(),
            "actor": actor_opt.state_dict(),
            "ensemble": ens_opt.state_dict(),
        },
        target_dir / "optimizer_state.pt",
    )

    # rng_state.pkl mirrors the runner's ``_save_rng_state`` shape
    # closely enough for ``_load_rng_state`` to round-trip without
    # raising. The runtime tensors are None — ``load_checkpoint``
    # tolerates None entries by leaving the runner's runtime state at
    # its construction-time defaults, and the test path either
    # short-circuits before run() (the dry-run check) or runs the
    # runner just long enough to verify the actor-column state.
    import pickle
    import random as _random

    import numpy as np

    rng_blob: dict[str, Any] = {
        "python_random": _random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().numpy().tobytes(),
        "torch_device_type": "cpu",
        "torch_device_rng": None,
        "sample_rng": torch.Generator(device="cpu").get_state().numpy().tobytes(),
        "runtime": {
            "h_prev": None,
            "z_prev": None,
            "a_prev": None,
            "h_curr": None,
            "z_curr": None,
            "obs_curr": None,
            "obs_curr_hash": None,
            "env_step_meta": None,
            "iteration": 0,
        },
    }
    with open(target_dir / "rng_state.pkl", "wb") as fh:
        pickle.dump(rng_blob, fh)


def test_init_zero_scalar_column_real_run(tmp_path: Path) -> None:
    """The mutation script produces a valid lesioned checkpoint:
    the actor's scalar column is zero; existing columns and bias
    are byte-identical to the source; the world model and EMA
    target tensors are byte-identical to the source; the world_event
    JSONL carries the correct mirror_marker payload."""
    wm_cfg = _tiny_world_model_config()
    source_dir = tmp_path / "source" / "checkpoints" / "ckpt-000001"
    output_dir = tmp_path / "lesioned" / "checkpoints" / "ckpt-000001"
    _build_synthetic_phase7_5_checkpoint(
        source_dir,
        wm_cfg=wm_cfg,
        action_dim=5,
        ensemble_k=2,
    )

    result = lesion_script.mutate(
        source_checkpoint=source_dir,
        output_dir=output_dir,
        dry_run=False,
    )

    assert result.dry_run is False
    # The lesioned actor column is exactly zero.
    out_weights = _load_safetensors(str(output_dir / "weights.safetensors"))
    in_weights = _load_safetensors(str(source_dir / "weights.safetensors"))
    actor_w_out = out_weights["actor.net.0.weight"]
    actor_w_in = in_weights["actor.net.0.weight"]
    column_idx = wm_cfg.h_dim + wm_cfg.z_dim
    assert actor_w_out.shape == actor_w_in.shape
    assert torch.equal(
        actor_w_out[:, column_idx:],
        torch.zeros_like(actor_w_out[:, column_idx:]),
    ), "lesioned actor scalar column is not all-zero"

    # Existing actor columns (the h+z block) are byte-identical to the
    # source.
    assert torch.equal(
        actor_w_out[:, :column_idx], actor_w_in[:, :column_idx]
    ), "existing actor columns differ from source after mutation"

    # The actor's other parameters (bias on net.0, all other layers)
    # are byte-identical to the source.
    for k in in_weights:
        if k == "actor.net.0.weight":
            continue
        assert torch.equal(out_weights[k], in_weights[k]), (
            f"non-actor-first-layer tensor {k!r} differs from source "
            f"after mutation"
        )

    # World model + EMA target tensors are byte-identical to the source
    # (subset of the above by-key check; explicit assertion for
    # readability).
    wm_keys = [k for k in in_weights if k.startswith("world_model.")]
    ema_keys = [
        k
        for k in wm_keys
        if k.startswith("world_model.target_encoder.")
        or k.startswith("world_model.target_gru_cell.")
    ]
    assert wm_keys
    assert ema_keys
    for k in wm_keys:
        assert torch.equal(out_weights[k], in_weights[k]), (
            f"world_model tensor {k!r} differs from source after mutation"
        )

    # Sidecar files are byte-identical.
    assert sorted(result.sidecar_files_copied) == sorted(
        lesion_script._SIDECAR_FILES
    )
    for name in lesion_script._SIDECAR_FILES:
        src_bytes = (source_dir / name).read_bytes()
        out_bytes = (output_dir / name).read_bytes()
        assert src_bytes == out_bytes, (
            f"sidecar file {name!r} not byte-identical between source "
            f"and lesioned"
        )

    # world_event.jsonl carries the mirror_marker payload.
    assert result.world_event_path is not None
    assert result.world_event_path.exists()
    we_records = _read_jsonl_records(result.world_event_path)
    assert len(we_records) == 1
    record = we_records[0]
    assert record["event_type"] == "mirror_marker"
    assert record["source"] == "system"
    payload = record["payload"]
    assert isinstance(payload, dict)
    assert payload["lesion_kind"] == "init_zero_scalar_column"
    assert payload["source_checkpoint"] == str(source_dir)
    assert payload["actor_first_layer_key"] == "actor.net.0.weight"
    assert payload["actor_input_dim"] == wm_cfg.h_dim + wm_cfg.z_dim + 1
    assert payload["column_start_index"] == column_idx

    # Moments before are non-trivial (small-Gaussian); after are zero.
    moments_before = result.column_moments_before
    moments_after = result.column_moments_after
    assert moments_before.abs_max > 0.0, (
        f"source column abs_max is 0.0 — Phase 7.5 init should be "
        f"non-zero small-Gaussian (got {moments_before})"
    )
    assert moments_after.abs_max == 0.0
    assert moments_after.mean == 0.0
    assert moments_after.std == 0.0


def test_init_zero_scalar_column_dry_run(tmp_path: Path) -> None:
    """``--dry-run`` short-circuits before any file write, prints the
    planned mutation summary, and returns 0."""
    wm_cfg = _tiny_world_model_config()
    source_dir = tmp_path / "source" / "checkpoints" / "ckpt-000001"
    output_dir = tmp_path / "lesioned" / "checkpoints" / "ckpt-000001"
    _build_synthetic_phase7_5_checkpoint(
        source_dir,
        wm_cfg=wm_cfg,
        action_dim=5,
        ensemble_k=2,
    )

    # Programmatic dry-run via mutate().
    result = lesion_script.mutate(
        source_checkpoint=source_dir,
        output_dir=output_dir,
        dry_run=True,
    )
    assert result.dry_run is True
    assert result.world_event_path is None
    assert result.sidecar_files_copied == ()
    # The output directory is *not* created — dry-run short-circuits
    # before any file system write.
    assert not output_dir.exists()
    # The moments are computed even under dry-run.
    assert result.column_moments_before.abs_max > 0.0
    assert result.column_moments_after.abs_max == 0.0

    # CLI dry-run via main(): exits 0, prints summary lines.
    rc = lesion_script.main(
        [
            "--source-checkpoint",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )
    assert rc == 0
    # Still no file system write from the CLI path.
    assert not output_dir.exists()


def test_init_zero_scalar_column_loadable_into_runner(tmp_path: Path) -> None:
    """The lesioned checkpoint loads into a fresh runner without
    error and the actor's scalar column reads back as zero."""
    wm_cfg = _tiny_world_model_config()
    source_dir = tmp_path / "source" / "checkpoints" / "ckpt-000001"
    output_dir = tmp_path / "lesioned" / "checkpoints" / "ckpt-000001"
    _build_synthetic_phase7_5_checkpoint(
        source_dir,
        wm_cfg=wm_cfg,
        action_dim=5,
        ensemble_k=2,
    )
    lesion_script.mutate(
        source_checkpoint=source_dir,
        output_dir=output_dir,
        dry_run=False,
    )

    # Load the lesioned checkpoint into a fresh runner. We point the
    # runner's ``checkpoints_dir`` at the lesioned dir's parent (the
    # CheckpointManager looks for ckpt-NNNNNN children inside).
    config = _make_runner_config(
        tmp_path=tmp_path / "runner_target",
        run_id="lesion-init-zero-loaded",
    )
    object.__setattr__(
        config, "checkpoints_dir", output_dir.parent
    )
    with _transport_pair(run_id="lesion-init-zero-loaded-env") as (
        client,
        _server,
        env_server,
        _thread,
    ):
        runner = Runner(config, client, env_server=env_server)
        try:
            runner.load_checkpoint("ckpt-000001")
            actor_w = runner.actor.net[0].weight  # type: ignore[union-attr]
            column_idx = wm_cfg.h_dim + wm_cfg.z_dim
            assert torch.equal(
                actor_w[:, column_idx:].cpu(),
                torch.zeros_like(actor_w[:, column_idx:]).cpu(),
            ), "loaded lesioned actor scalar column is not zero"
        finally:
            runner.close()


# ---- 4. non-lesion behavior unchanged -----------------------------------


def test_non_lesion_run_byte_identical_to_reference(tmp_path: Path) -> None:
    """Two runs at ``lesion_kind=None`` with the same RNG seed produce
    byte-identical agent_step telemetry. This is the regression check
    that the lesion plumbing introduced no behavioral change on the
    un-lesioned path."""
    config_a = _make_runner_config(
        tmp_path=tmp_path / "ref_a",
        run_id="lesion-none-ref-a",
        lesion_kind=None,
    )
    rows_a, _ = _run_smoke(config_a)

    config_b = _make_runner_config(
        tmp_path=tmp_path / "ref_b",
        run_id="lesion-none-ref-b",
        lesion_kind=None,
    )
    rows_b, _ = _run_smoke(config_b)

    assert len(rows_a) == len(rows_b) > 0
    # Compare the substrate-side scalar fields which are deterministic
    # under fixed seed (the env transport seed is the same across runs;
    # the runner's RNG-state is reset via ``torch.manual_seed(0)``
    # inside ``_run_smoke``). The run_ids differ, so we filter that.
    deterministic_fields = (
        "kl_aggregate_t",
        "recon_loss_t",
        "self_prediction_error_t",
        "self_prediction_error_masked_t",
        "intrinsic_signal_t",
        "action_t",
        "policy_entropy_t",
        "step_in_episode",
        "episode_id",
    )
    for ra, rb in zip(rows_a, rows_b, strict=True):
        for f in deterministic_fields:
            assert ra[f] == rb[f], (
                f"non-lesion run telemetry differs at t={ra['t']} "
                f"field={f!r}: a={ra[f]!r} b={rb[f]!r}"
            )


def test_non_lesion_run_no_mirror_marker_emitted(tmp_path: Path) -> None:
    """No ``mirror_marker`` ``world_event`` is emitted on an
    un-lesioned run."""
    config = _make_runner_config(
        tmp_path=tmp_path,
        run_id="lesion-none-no-marker",
        lesion_kind=None,
    )
    _, world_events = _run_smoke(config)
    mirror_markers = [
        e for e in world_events if e.get("event_type") == "mirror_marker"
    ]
    assert mirror_markers == [], (
        f"un-lesioned run emitted mirror_marker events: {mirror_markers}"
    )


# ---- 5. lesion kind recorded in world_event ------------------------------


def test_lesion_kind_recorded_in_world_event_disable(
    tmp_path: Path,
) -> None:
    """The ``mirror_marker`` ``world_event`` carries the correct
    ``lesion_kind`` for ``disable_self_prediction``."""
    config = _make_runner_config(
        tmp_path=tmp_path,
        run_id="lesion-marker-disable",
        lesion_kind="disable_self_prediction",
    )
    _, world_events = _run_smoke(config, total_env_steps=10)
    mirror_markers = [
        e for e in world_events if e.get("event_type") == "mirror_marker"
    ]
    assert len(mirror_markers) == 1
    payload = mirror_markers[0]["payload"]
    assert isinstance(payload, dict)
    assert payload["lesion_kind"] == "disable_self_prediction"
    assert mirror_markers[0]["source"] == "system"
    # No variant for this lesion kind.
    assert "lesion_variant" not in payload


def test_lesion_kind_recorded_in_world_event_zero_or_randomize(
    tmp_path: Path,
) -> None:
    """The ``mirror_marker`` carries ``lesion_kind`` AND
    ``lesion_variant`` for ``zero_or_randomize_scalar``, plus the
    empirical bounds."""
    config = _make_runner_config(
        tmp_path=tmp_path,
        run_id="lesion-marker-randomize",
        lesion_kind="zero_or_randomize_scalar",
        lesion_zero_or_randomize_variant="randomize",
        lesion_zero_or_randomize_empirical_min=0.05,
        lesion_zero_or_randomize_empirical_max=0.95,
    )
    _, world_events = _run_smoke(config, total_env_steps=10)
    mirror_markers = [
        e for e in world_events if e.get("event_type") == "mirror_marker"
    ]
    assert len(mirror_markers) == 1
    payload = mirror_markers[0]["payload"]
    assert isinstance(payload, dict)
    assert payload["lesion_kind"] == "zero_or_randomize_scalar"
    assert payload["lesion_variant"] == "randomize"
    assert payload["lesion_empirical_min"] == 0.05
    assert payload["lesion_empirical_max"] == 0.95


# ---- 6. v1 ensemble lesions (regression checks) --------------------------


def test_lesion_ensemble_k1_smoke(tmp_path: Path) -> None:
    """``"ensemble_k1"`` runs end-to-end without errors; the
    ``intrinsic_signal_t`` is identically zero (variance over a single
    head is 0)."""
    config = _make_runner_config(
        tmp_path=tmp_path,
        run_id="lesion-ensemble-k1",
        lesion_kind="ensemble_k1",
        ensemble_k=5,
    )
    rows, world_events = _run_smoke(config, total_env_steps=20)
    assert rows
    for r in rows:
        assert float(r["intrinsic_signal_t"]) == 0.0, (
            f"ensemble_k1 produced non-zero intrinsic_signal_t="
            f"{r['intrinsic_signal_t']!r} at t={r['t']}"
        )
    mirror_markers = [
        e for e in world_events if e.get("event_type") == "mirror_marker"
    ]
    assert len(mirror_markers) == 1
    assert mirror_markers[0]["payload"]["lesion_kind"] == "ensemble_k1"


def test_lesion_ensemble_constant_smoke(tmp_path: Path) -> None:
    """``"ensemble_constant"`` runs end-to-end without errors; the
    ``intrinsic_signal_t`` is identically zero (the disagreement
    method short-circuits to zeros regardless of head outputs)."""
    config = _make_runner_config(
        tmp_path=tmp_path,
        run_id="lesion-ensemble-constant",
        lesion_kind="ensemble_constant",
        ensemble_k=5,
    )
    rows, world_events = _run_smoke(config, total_env_steps=20)
    assert rows
    for r in rows:
        assert float(r["intrinsic_signal_t"]) == 0.0, (
            f"ensemble_constant produced non-zero intrinsic_signal_t="
            f"{r['intrinsic_signal_t']!r} at t={r['t']}"
        )
    mirror_markers = [
        e for e in world_events if e.get("event_type") == "mirror_marker"
    ]
    assert len(mirror_markers) == 1
    assert mirror_markers[0]["payload"]["lesion_kind"] == "ensemble_constant"
