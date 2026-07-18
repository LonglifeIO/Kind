"""Probe 4.5 Phase 1 — maintenance-refit harness + runner hook tests (S-DEC).

Authority: the frozen pre-registration §3
(``docs/decisions/probe4_5_preregistration_2026-07-13.md``) and the
implementation plan's Phase 1 test list: refit touches only
``energy_decoder`` parameters (bit-identity on everything else — the F1
discipline as a test); refit determinism given seeds; honesty-gate report
round-trip; the honesty-STOP branch exercised; the cadence hook fires at the
configured checkpoint-aligned boundary and never mid-step.

Margin math is validated on synthetic tables (the decode-honesty test
discipline: an honest decoder must pass, a lying one must fail, coverage gaps
are skipped-and-reported, an unmeasurable keystone readout fails). No test
asserts anything about a *trained* checkpoint's honesty — margins here are
deliberately loosened or made impossible to exercise both gate branches; the
frozen §3 values stay the dataclass defaults.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from kind.agents.actor import Actor
from kind.agents.world_model import WorldModel, WorldModelConfig
from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import GridWorldConfig
from kind.env.transport import EnvTransportClient, EnvTransportServer
from kind.observer.decode_honesty import (
    DecodeHonestyReport,
    HonestyTable,
    TeacherForcedTrajectory,
    honesty_table,
    region_edges_from_config,
)
from kind.observer.maintenance_refit import (
    MAINTENANCE_REFIT_SCHEMA_VERSION,
    HonestyMargins,
    HonestyStopError,
    MaintenanceRefitConfig,
    evaluate_honesty_margins,
    maintenance_report_path,
    refit_energy_decoder_head,
    report_to_jsonable,
    run_maintenance_refit,
    run_scheduled_maintenance,
)
from kind.training.runner import Runner, RunnerConfig

_EDGES = region_edges_from_config(GridWorldConfig())

#: Margins loose enough that any random-init head passes — the pass branch,
#: without fitting a test to a trained outcome. ``region_min_n=1`` keeps the
#: coverage-qualification path live (n == 0 regions are skipped, not graded).
_LOOSE_MARGINS = HonestyMargins(
    oracle_in_band_abs_error_max=1e6,
    pooled_slope_min=-1e6,
    pooled_out_of_range_mass_max=1.0,
    oracle_region_abs_bias_max=1e6,
    region_min_n=1,
)

#: Margins no decoder can meet (slope ≥ 1e6) — the STOP branch.
_IMPOSSIBLE_MARGINS = HonestyMargins(pooled_slope_min=1e6)


# ---- tiny fixtures --------------------------------------------------------


def _tiny_wm_config() -> WorldModelConfig:
    return WorldModelConfig(
        h_dim=16, z_dim=4, embed_dim=8, mlp_hidden=16, energy_embed_dim=4
    )


def _tiny_instances(
    seed: int = 7,
) -> tuple[WorldModel, Actor, GridWorldConfig]:
    torch.manual_seed(seed)
    wm = WorldModel(_tiny_wm_config())
    actor = Actor(h_dim=16, z_dim=4, action_dim=5, mlp_hidden=16)
    grid_cfg = GridWorldConfig(episode_length=40)
    return wm, actor, grid_cfg


def _synthetic_refit_trajectories(
    n: int = 600, seed: int = 0
) -> list[TeacherForcedTrajectory]:
    """Refit-input pairs with the tiny model's latent dims; the refit reads
    only ``h`` / ``z`` / ``sensed_energy``."""
    rng = np.random.default_rng(seed)
    sensed = rng.uniform(0.0, 1.0, n)
    return [
        TeacherForcedTrajectory(
            true_energy=sensed.copy(),
            sensed_energy=sensed,
            decode_energy=np.zeros(n),
            h=rng.normal(size=(n, 16)).astype(np.float32),
            z=rng.normal(size=(n, 4)).astype(np.float32),
        )
    ]


def _tiny_maintenance_config(
    out_dir: Path,
    grid_cfg: GridWorldConfig,
    *,
    margins: HonestyMargins,
    refit_every_n_env_steps: int = 10_000,
) -> MaintenanceRefitConfig:
    return MaintenanceRefitConfig(
        grid_world_config=grid_cfg,
        out_dir=out_dir,
        refit_every_n_env_steps=refit_every_n_env_steps,
        margins=margins,
        train_seeds_per_source=1,
        train_episodes_per_seed=1,
        diagnostic_seeds_per_source=2,
        refit_epochs=2,
        refit_batch_size=64,
        honesty_seeds_per_source=1,
        honesty_episodes_per_seed=1,
    )


# ---- refit mechanics ------------------------------------------------------


def test_refit_touches_only_energy_decoder_head() -> None:
    wm, _actor, _grid = _tiny_instances()
    before = {name: p.detach().clone() for name, p in wm.named_parameters()}
    refit_energy_decoder_head(
        wm,
        _synthetic_refit_trajectories(),
        epochs=2,
        batch_size=64,
        learning_rate=1e-3,
        torch_seed=4321,
    )
    changed = {
        name
        for name, p in wm.named_parameters()
        if not torch.equal(p.detach(), before[name])
    }
    assert changed, "refit changed nothing"
    assert all(name.startswith("energy_decoder.") for name in changed), changed


def test_refit_restores_requires_grad_and_clears_grads() -> None:
    # The live-model discipline the F1 script (dead-copy) didn't need: the
    # EMA target siblings carry requires_grad=False by design — a blanket
    # re-enable would hand them to the runner's optimizer.
    wm, _actor, _grid = _tiny_instances()
    wm.train()
    flags_before = {name: p.requires_grad for name, p in wm.named_parameters()}
    assert not any(
        p.requires_grad
        for name, p in wm.named_parameters()
        if name.startswith(("target_encoder.", "target_gru_cell."))
    )
    refit_energy_decoder_head(
        wm,
        _synthetic_refit_trajectories(),
        epochs=1,
        batch_size=64,
        learning_rate=1e-3,
        torch_seed=4321,
    )
    assert {
        name: p.requires_grad for name, p in wm.named_parameters()
    } == flags_before
    assert all(p.grad is None for p in wm.energy_decoder.parameters())
    assert wm.training  # mode restored


def test_refit_deterministic_given_seeds() -> None:
    trajectories = _synthetic_refit_trajectories()
    heads: list[dict[str, torch.Tensor]] = []
    for _ in range(2):
        wm, _actor, _grid = _tiny_instances(seed=11)  # identical init
        refit_energy_decoder_head(
            wm,
            trajectories,
            epochs=3,
            batch_size=64,
            learning_rate=1e-3,
            torch_seed=4321,
        )
        heads.append(
            {
                name: p.detach().clone()
                for name, p in wm.energy_decoder.named_parameters()
            }
        )
    assert heads[0].keys() == heads[1].keys()
    for name in heads[0]:
        assert torch.equal(heads[0][name], heads[1][name]), name


# ---- the §3 margin verdict ------------------------------------------------


def _synthetic_table(
    source: str,
    true: np.ndarray,
    decode: np.ndarray,
) -> HonestyTable:
    n = true.shape[0]
    traj = TeacherForcedTrajectory(
        true_energy=true.astype(np.float64),
        sensed_energy=true.astype(np.float64),
        decode_energy=decode.astype(np.float64),
        h=np.zeros((n, 1), dtype=np.float32),
        z=np.zeros((n, 1), dtype=np.float32),
    )
    return honesty_table(source, [traj], _EDGES)


def _report_from_tables(
    oracle: HonestyTable, pooled: HonestyTable
) -> DecodeHonestyReport:
    return DecodeHonestyReport(
        schema_version="0.1.0",
        checkpoint_label="synthetic",
        edges=_EDGES,
        per_source=(oracle,),
        pooled=pooled,
        seeds_per_source=1,
        episodes_per_seed=1,
        seed_bases=(("oracle", 0),),
    )


def test_honest_decoder_passes_frozen_margins() -> None:
    true = np.linspace(0.0, 1.0, 4001)
    table = _synthetic_table("oracle", true, true)
    verdict = evaluate_honesty_margins(
        _report_from_tables(table, _synthetic_table("pooled", true, true)),
        HonestyMargins(),  # the FROZEN §3 values
    )
    assert verdict.all_passed
    names = {c.name for c in verdict.checks}
    assert "oracle_in_band_abs_error" in names
    assert "pooled_slope_decode_vs_true" in names
    assert "pooled_out_of_range_mass" in names


def test_lying_decoder_fails_frozen_margins() -> None:
    # The bin-1 shape: constant impossible reading above the ceiling
    # (classification §2 — mean 1.124 on oracle in-band states).
    true = np.linspace(0.0, 1.0, 4001)
    decode = np.full_like(true, 1.124)
    table = _synthetic_table("oracle", true, decode)
    verdict = evaluate_honesty_margins(
        _report_from_tables(table, _synthetic_table("pooled", true, decode)),
        HonestyMargins(),
    )
    assert not verdict.all_passed
    failed = {c.name for c in verdict.checks if not c.passed}
    assert "oracle_in_band_abs_error" in failed
    assert "pooled_slope_decode_vs_true" in failed
    assert "pooled_out_of_range_mass" in failed


def test_under_covered_regions_skipped_and_reported() -> None:
    # In-band-only coverage: the other regions fall below the §3 floor of
    # 500 samples and are reported-not-graded (coverage-qualified).
    true = np.full(2000, 0.6)
    with_spread = np.concatenate([true, np.linspace(0.46, 0.74, 500)])
    table = _synthetic_table("oracle", with_spread, with_spread)
    verdict = evaluate_honesty_margins(
        _report_from_tables(
            table, _synthetic_table("pooled", with_spread, with_spread)
        ),
        HonestyMargins(),
    )
    assert set(verdict.coverage_skipped_regions) == {
        "floor_adjacent",
        "below_band",
        "above_band",
        "ceiling_adjacent",
    }
    region_checks = [
        c for c in verdict.checks if c.name.startswith("oracle_region_abs_bias:")
    ]
    assert [c.name for c in region_checks] == ["oracle_region_abs_bias:in_band"]


def test_unmeasurable_keystone_fails() -> None:
    # No oracle in-band coverage at all → the in-band readout is None →
    # honesty that cannot be measured is not demonstrated.
    true = np.full(2000, 0.01)
    table = _synthetic_table("oracle", true, true)
    verdict = evaluate_honesty_margins(
        _report_from_tables(table, _synthetic_table("pooled", true, true)),
        HonestyMargins(),
    )
    in_band = next(
        c for c in verdict.checks if c.name == "oracle_in_band_abs_error"
    )
    assert in_band.value is None
    assert not in_band.passed
    assert not verdict.all_passed


# ---- one full cycle: report + round-trip + STOP ---------------------------


def test_maintenance_cycle_writes_round_trippable_report(
    tmp_path: Path,
) -> None:
    wm, actor, grid_cfg = _tiny_instances()
    wm.train()  # the runner's modules arrive in train mode
    config = _tiny_maintenance_config(
        tmp_path / "maintenance", grid_cfg, margins=_LOOSE_MARGINS
    )
    report = run_maintenance_refit(
        wm, actor, config, env_step=123, occasion="burn_in_close"
    )
    assert report.schema_version == MAINTENANCE_REFIT_SCHEMA_VERSION
    assert report.verdict.all_passed
    assert report.occasion == "burn_in_close"
    # Machine-written report round-trips.
    path = maintenance_report_path(
        config.out_dir, env_step=123, occasion="burn_in_close"
    )
    assert path.exists()
    assert json.loads(path.read_text()) == json.loads(
        json.dumps(report_to_jsonable(report))
    )
    # Live-model state restored: mode, and the EMA siblings still frozen.
    assert wm.training
    assert not any(
        p.requires_grad
        for name, p in wm.named_parameters()
        if name.startswith(("target_encoder.", "target_gru_cell."))
    )


def test_scheduled_maintenance_stop_after_diagnostic_recollection(
    tmp_path: Path,
) -> None:
    wm, actor, grid_cfg = _tiny_instances()
    config = _tiny_maintenance_config(
        tmp_path / "maintenance", grid_cfg, margins=_IMPOSSIBLE_MARGINS
    )
    with pytest.raises(HonestyStopError) as excinfo:
        run_scheduled_maintenance(wm, actor, config, env_step=200)
    # §3: exactly one diagnostic re-collection before the STOP; both cycles'
    # reports are on disk and on the error.
    assert len(excinfo.value.reports) == 2
    assert [r.occasion for r in excinfo.value.reports] == [
        "scheduled",
        "diagnostic_recollection",
    ]
    assert maintenance_report_path(
        config.out_dir, env_step=200, occasion="scheduled"
    ).exists()
    assert maintenance_report_path(
        config.out_dir, env_step=200, occasion="diagnostic_recollection"
    ).exists()
    # The diagnostic used the larger mixture (per-source step counts doubled).
    first, second = excinfo.value.reports
    assert (
        second.mixture_counts["own_policy"]
        == 2 * first.mixture_counts["own_policy"]
    )


# ---- runner hook: checkpoint-aligned cadence, never mid-step --------------


def _quiet_grid_world_config() -> GridWorldConfig:
    return GridWorldConfig(
        episode_length=50,
        initial_regrowth_p=0.0,
        drift_p_min=0.0,
        drift_p_max=0.0,
        drift_magnitude_per_step=0.0,
        n_initial_resources=2,
    )


def _tiny_runner_wm_config() -> WorldModelConfig:
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


def _runner_config(
    tmp_path: Path,
    *,
    maintenance: MaintenanceRefitConfig | None,
    checkpoint_every_n_env_steps: int,
) -> RunnerConfig:
    return RunnerConfig(
        world_model_config=_tiny_runner_wm_config(),
        run_id="maintenance-smoke",
        telemetry_dir=tmp_path / "telemetry",
        checkpoints_dir=tmp_path / "checkpoints",
        ensemble_k=2,
        imagination_horizon=4,
        replay_capacity=200,
        replay_sequence_length=4,
        replay_batch_size=2,
        warmup_env_steps=10,
        dream_cadence_env_steps=50,
        dream_horizon=4,
        checkpoint_every_n_env_steps=checkpoint_every_n_env_steps,
        parquet_rows_per_shard=10,
        maintenance_refit=maintenance,
    )


@contextmanager
def _transport_pair() -> Iterator[tuple[EnvTransportClient, EnvServer]]:
    config = EnvServerConfig(
        grid_world_config=_quiet_grid_world_config(),
        seed=42,
        world_event_handler=lambda _r: None,
        run_id="maintenance-smoke-env",
    )
    env_server = EnvServer(config)
    transport_server = EnvTransportServer(env_server, host="127.0.0.1", port=0)
    server_thread = threading.Thread(
        target=transport_server.serve_forever,
        name="MaintenanceSmokeEnvTransportServer",
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


def test_runner_rejects_cadence_not_checkpoint_aligned(tmp_path: Path) -> None:
    maintenance = _tiny_maintenance_config(
        tmp_path / "maintenance",
        _quiet_grid_world_config(),
        margins=_LOOSE_MARGINS,
        refit_every_n_env_steps=75,  # not a multiple of 50
    )
    config = _runner_config(
        tmp_path, maintenance=maintenance, checkpoint_every_n_env_steps=50
    )
    with _transport_pair() as (client, env_server):
        with pytest.raises(ValueError, match="checkpoint-aligned"):
            Runner(config, client, env_server=env_server)


def test_runner_hook_fires_at_boundary_only_and_before_commit(
    tmp_path: Path,
) -> None:
    # Checkpoints every 50, refits every 100: the refit must fire exactly
    # once in a 101-step run (env_step 100) — not at 50 (a checkpoint
    # boundary that is not a refit boundary), never mid-step — and the
    # checkpoint committed at 100 must carry the POST-refit head (refit
    # before commit: a resume continues from post-refit state).
    maintenance = _tiny_maintenance_config(
        tmp_path / "maintenance",
        _quiet_grid_world_config(),
        margins=_LOOSE_MARGINS,
        refit_every_n_env_steps=100,
    )
    config = _runner_config(
        tmp_path, maintenance=maintenance, checkpoint_every_n_env_steps=50
    )
    with _transport_pair() as (client, env_server):
        runner = Runner(config, client, env_server=env_server)
        try:
            runner.run(total_env_steps=101)
        finally:
            runner.close()

    reports = sorted((tmp_path / "maintenance").glob("*.json"))
    assert [p.name for p in reports] == ["refit_00000100_scheduled.json"]

    # Ordering: the ckpt-000002 weights (committed at env_step 100, after
    # the refit at the same boundary) equal the runner's post-refit head.
    weights = load_file(
        str(tmp_path / "checkpoints" / "ckpt-000002" / "weights.safetensors")
    )
    head_now = {
        f"world_model.energy_decoder.{name}": p.detach().cpu()
        for name, p in runner._world_model.energy_decoder.named_parameters()
    }
    for name, expected in head_now.items():
        assert torch.equal(weights[name], expected), name


def test_runner_honesty_stop_propagates_and_blocks_commit(
    tmp_path: Path,
) -> None:
    # §3 honesty-STOP through the runner: the run raises; the checkpoint at
    # the same boundary is never committed (the run does not proceed — and
    # does not snapshot — on a lying belief).
    maintenance = _tiny_maintenance_config(
        tmp_path / "maintenance",
        _quiet_grid_world_config(),
        margins=_IMPOSSIBLE_MARGINS,
        refit_every_n_env_steps=50,
    )
    config = _runner_config(
        tmp_path, maintenance=maintenance, checkpoint_every_n_env_steps=50
    )
    with _transport_pair() as (client, env_server):
        runner = Runner(config, client, env_server=env_server)
        try:
            with pytest.raises(HonestyStopError):
                runner.run(total_env_steps=101)
        finally:
            runner.close()

    reports = sorted(p.name for p in (tmp_path / "maintenance").glob("*.json"))
    assert reports == [
        "refit_00000050_diagnostic_recollection.json",
        "refit_00000050_scheduled.json",
    ]
    assert not (tmp_path / "checkpoints" / "ckpt-000001").exists()
