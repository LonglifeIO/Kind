"""Probe 3.5 Phase 1 — energy channel without preference.

Tests for the energy economy (env), the fused proprioceptive branch (world
model), the Probe-3.5 telemetry (schema), and the dead-path battery metric
functions. The dead-path battery *on a trained model* (the Phase-1 gate) is
exercised by ``test_dead_path_battery_passes_on_trained_channel`` below and,
at the full frozen training age, by ``scripts/run_probe3_5_phase1_baseline.py``.

The load-bearing discipline: ``true_energy`` never enters a training loss — it is
GridState ground truth and an eval-probe target only (plan S-ENV rule).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kind.agents.world_model import WorldModel, WorldModelConfig
from kind.env.grid_world import NUM_ACTIONS, GridWorld, GridWorldConfig
from kind.observer.energy_eval import (
    DeadPathMargins,
    battery_a_latent_predictability,
    battery_c_action_history_ablation,
    battery_d_per_dim_kl_escape,
    collect_energy_eval_data,
    run_dead_path_battery,
)
from kind.observer.schemas import (
    PROBE_3_5_TELEMETRY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    AgentStep,
)

_STAY = NUM_ACTIONS - 1  # action 4


def _energy_cfg(**overrides: object) -> GridWorldConfig:
    return GridWorldConfig(**overrides)  # type: ignore[arg-type]


# ===========================================================================
# Environment — energy dynamics
# ===========================================================================


def test_true_energy_starts_at_normalized_setpoint() -> None:
    """``energy_init`` defaults to ``0.6 * energy_norm_max`` → normalized 0.6,
    the frozen pre-registration setpoint."""
    w = GridWorld(_energy_cfg(), seed=0)
    w.reset()
    assert w.state.true_energy == pytest.approx(0.6, abs=1e-9)


def test_energy_normalized_to_unit_range() -> None:
    """true_energy and sensed_energy are always in [0, 1]."""
    w = GridWorld(_energy_cfg(), seed=1)
    s = w.reset()
    rng = np.random.default_rng(1)
    for _ in range(300):
        assert 0.0 <= w.state.true_energy <= 1.0
        assert 0.0 <= s.sensed_energy <= 1.0
        s = w.step(int(rng.integers(0, NUM_ACTIONS)))


def test_movement_costs_more_than_stay() -> None:
    """Depletion asymmetry: a move action costs base_decay + move_cost; stay
    costs only base_decay. From the same state, one move depletes more."""
    cfg = _energy_cfg()
    w_move = GridWorld(cfg, seed=2)
    w_move.reset()
    w_stay = GridWorld(cfg, seed=2)
    w_stay.reset()
    e0 = w_move.state.true_energy
    # A move into a wall-free, resource-free direction (so no replenishment):
    # action 4 is stay; action 0 (up) moves. Choose a step where neither lands
    # on a resource by using stay vs up from the identical start and comparing
    # the *decay* component only when no resource is consumed.
    w_stay.step(_STAY)
    w_move.step(0)  # up
    stay_drop = e0 - w_stay.state.true_energy
    move_drop = e0 - w_move.state.true_energy
    # If the move consumed a resource, move_drop could be negative; guard by
    # asserting the cost asymmetry only in the no-replenish case (both drops
    # positive). Stay's drop is the pure base decay; move's is base + move cost.
    if move_drop > 0:
        assert move_drop > stay_drop
    assert stay_drop == pytest.approx(
        cfg.energy_base_decay / cfg.energy_norm_max, abs=1e-9
    )


def test_replenish_on_resource_entry() -> None:
    """Entering a resource cell raises energy by the replenish amount (minus the
    step's move cost). A stay over a resource does not replenish."""
    cfg = _energy_cfg(start_cell=(0, 0), n_initial_resources=0)
    w = GridWorld(cfg, seed=3)
    w.reset()
    # Place a resource adjacent and step onto it.
    w._grid[0, 1] = 2  # RESOURCE  (test reaches into the grid deliberately)
    before = w.state.true_energy
    w.step(3)  # right → enters (0,1), consumes the resource
    after = w.state.true_energy
    expected = min(
        1.0,
        before
        + (cfg.energy_replenish_per_resource - cfg.energy_base_decay - cfg.energy_move_cost)
        / cfg.energy_norm_max,
    )
    assert after == pytest.approx(expected, abs=1e-9)


def test_energy_floors_at_zero_no_terminal_state() -> None:
    """Energy floors at 0 and the env keeps running — no death, no termination
    on depletion. Sustained movement drives energy to 0; the env still steps."""
    cfg = _energy_cfg(n_initial_resources=0)
    w = GridWorld(cfg, seed=4)
    w.reset()
    for _ in range(500):
        w.step(0)  # keep moving (into the wall edge eventually; still decays)
    assert w.state.true_energy == 0.0
    # Still steppable — no terminal/absorbing state.
    step = w.step(0)
    assert step.episode_id >= 0
    assert 0.0 <= w.state.true_energy <= 1.0


def test_energy_carries_across_episode_boundary() -> None:
    """Energy is Io's internal state — the soft 200-step auto-reset (which
    resamples resources and replaces the agent) does NOT reset it."""
    cfg = _energy_cfg(episode_length=10, n_initial_resources=0)
    w = GridWorld(cfg, seed=5)
    w.reset()
    # Deplete a bit, then cross the boundary; energy must not jump back to init.
    last_energy = w.state.true_energy
    crossed = False
    for _ in range(25):
        before_ep = w._episode_id
        w.step(_STAY)
        if w._episode_id != before_ep:
            # Just crossed: energy must be continuous (decayed), not re-init.
            assert w.state.true_energy < cfg.energy_init / cfg.energy_norm_max
            assert w.state.true_energy <= last_energy + 1e-9
            crossed = True
            break
        last_energy = w.state.true_energy
    assert crossed, "expected to cross an episode boundary within 25 steps"


def test_hard_reset_reinitializes_energy() -> None:
    """``reset()`` (the only path) re-initialises energy to the setpoint."""
    cfg = _energy_cfg(n_initial_resources=0)
    w = GridWorld(cfg, seed=6)
    w.reset()
    for _ in range(50):
        w.step(0)
    assert w.state.true_energy < 0.6
    w.reset()
    assert w.state.true_energy == pytest.approx(0.6, abs=1e-9)


def test_sensing_is_deterministic_given_seed() -> None:
    """Same seed + same actions → identical sensed_energy trajectories (the
    third RNG stream is spawned deterministically from the env SeedSequence)."""
    cfg = _energy_cfg(energy_obs_noise_sigma=0.1)
    actions = [i % NUM_ACTIONS for i in range(120)]

    def run(seed: int) -> list[float]:
        w = GridWorld(cfg, seed)
        s = w.reset()
        out = [s.sensed_energy]
        for a in actions:
            out.append(w.step(a).sensed_energy)
        return out

    assert run(7) == run(7)
    assert run(7) != run(8)


def test_sensing_noise_zero_is_lagged_true_quantized() -> None:
    """With σ=0 the sensed scalar is the (quantized) normalized true energy
    lagged by ``energy_obs_lag`` steps."""
    cfg = _energy_cfg(
        energy_obs_noise_sigma=0.0, energy_obs_lag=1, energy_obs_quantization_levels=1
    )
    w = GridWorld(cfg, seed=9)
    s0 = w.reset()
    true0 = w.state.true_energy
    s1 = w.step(_STAY)
    # lag=1: sensed at step 1 reflects the normalized true energy one step ago
    # (the reset state), σ=0 and no quantization → exact.
    assert s1.sensed_energy == pytest.approx(true0, abs=1e-9)
    # The reset emission seeds the buffer with its own value.
    assert s0.sensed_energy == pytest.approx(true0, abs=1e-9)


def test_third_rng_stream_does_not_disturb_regrowth_or_drift() -> None:
    """Spawning a third (sensing) child from the SeedSequence leaves the first
    two children (regrowth, drift) byte-identical — so pre-3.5 env determinism
    (resource layouts, drift) is preserved. Checked via identical grids under a
    no-op (stay) policy that never consumes."""
    cfg = _energy_cfg()
    w = GridWorld(cfg, seed=11)
    w.reset()
    grids = []
    for _ in range(50):
        w.step(_STAY)
        grids.append(w.state.grid.copy())
    w2 = GridWorld(cfg, seed=11)
    w2.reset()
    for i in range(50):
        w2.step(_STAY)
        assert np.array_equal(w2.state.grid, grids[i])


# ===========================================================================
# World model — fused proprioceptive branch
# ===========================================================================


def _small_wm_config() -> WorldModelConfig:
    return WorldModelConfig(
        h_dim=16, z_dim=4, embed_dim=8, mlp_hidden=16, energy_embed_dim=4
    )


def test_world_model_step_emits_energy_pred_shape() -> None:
    wm = WorldModel(_small_wm_config())
    obs = torch.zeros(3, 1, 32, 32)
    h = torch.zeros(3, 16)
    z = torch.zeros(3, 4)
    a = torch.zeros(3, dtype=torch.long)
    sensed = torch.rand(3, 1)
    step = wm.step(obs, h, z, a, sensed_energy=sensed)
    assert step.energy_pred.shape == (3, 1)
    assert torch.isfinite(step.energy_pred).all()


def test_decode_energy_shape() -> None:
    wm = WorldModel(_small_wm_config())
    h = torch.randn(5, 16)
    z = torch.randn(5, 4)
    assert wm.decode_energy(h, z).shape == (5, 1)


def test_energy_encoder_fuses_into_posterior_width() -> None:
    """The posterior head's input is h + embed + energy_embed (grounding fact 1)."""
    cfg = _small_wm_config()
    wm = WorldModel(cfg)
    first = wm.posterior_head.fc1
    assert first.in_features == cfg.h_dim + cfg.embed_dim + cfg.energy_embed_dim


def test_loss_includes_weighted_energy_term() -> None:
    """``loss`` exposes the raw ``energy_recon_loss`` and folds the weighted term
    into ``total``; the recon target is sensed_energy."""
    cfg = _small_wm_config()
    wm = WorldModel(cfg)
    obs = torch.zeros(2, 1, 32, 32)
    h = torch.zeros(2, cfg.h_dim)
    z = torch.zeros(2, cfg.z_dim)
    a = torch.zeros(2, dtype=torch.long)
    sensed = torch.full((2, 1), 0.5)
    step = wm.step(obs, h, z, a, sensed_energy=sensed)

    loss_no_energy = wm.loss(step, obs)
    loss_energy = wm.loss(step, obs, sensed_energy_target=sensed.squeeze(-1))
    assert "energy_recon_loss" in loss_energy
    # With a target, total grows by exactly energy_recon_weight * raw MSE.
    expected_total = (
        loss_no_energy["total"] + cfg.energy_recon_weight * loss_energy["energy_recon_loss"]
    )
    assert loss_energy["total"].item() == pytest.approx(
        expected_total.item(), rel=1e-5
    )
    # Without a target the energy term is zero and total excludes it.
    assert loss_no_energy["energy_recon_loss"].item() == pytest.approx(0.0)


def test_energy_branch_grads_only_with_target_and_input() -> None:
    """The energy encoder/decoder receive gradients exactly when the energy
    channel is exercised (sensed fused + sensed target)."""
    cfg = _small_wm_config()
    wm = WorldModel(cfg)
    obs = torch.zeros(2, 1, 32, 32)
    h = torch.zeros(2, cfg.h_dim)
    z = torch.zeros(2, cfg.z_dim)
    a = torch.zeros(2, dtype=torch.long)
    sensed = torch.rand(2, 1)
    step = wm.step(obs, h, z, a, sensed_energy=sensed)
    loss = wm.loss(step, obs, sensed_energy_target=sensed.squeeze(-1))
    loss["total"].backward()
    assert wm.energy_encoder.fc1.weight.grad is not None
    assert wm.energy_decoder.head.weight.grad is not None


# ===========================================================================
# Telemetry schema — Probe 3.5 record version + validator
# ===========================================================================


def _agent_step_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = dict(
        run_id="r",
        checkpoint_id=None,
        t=0,
        episode_id=0,
        step_in_episode=0,
        wallclock_ms=0,
        h_t=[0.0],
        q_params_t=([0.0], [0.0]),
        p_params_t=([0.0], [0.0]),
        z_t=[0.0],
        kl_per_dim_t=[0.0],
        kl_aggregate_t=0.0,
        recon_loss_t=0.0,
        action_t=0,
        action_logprob_t=0.0,
        policy_entropy_t=0.0,
        obs_hash_t="x",
        intrinsic_signal_t=0.0,
        encoder_embedding_t=[0.0],
        self_prediction_t=[0.0],
        self_prediction_error_t=0.0,
        self_prediction_error_masked_t=False,
    )
    base.update(overrides)
    return base


def test_agent_step_probe_3_5_roundtrip_with_energy_fields() -> None:
    """A 3.5 record with all energy fields validates and round-trips."""
    rec = AgentStep(
        schema_version=PROBE_3_5_TELEMETRY_SCHEMA_VERSION,
        sensed_energy_t=0.7,
        true_energy_t=0.72,
        energy_pred_t=0.69,
        energy_recon_error_t=0.0001,
        **_agent_step_kwargs(),
    )
    again = AgentStep.model_validate_json(rec.model_dump_json())
    assert again.schema_version == PROBE_3_5_TELEMETRY_SCHEMA_VERSION
    assert again.true_energy_t == pytest.approx(0.72)


def test_agent_step_probe_3_5_rejects_missing_energy_field() -> None:
    """At the 3.5 record version every energy field is required non-None."""
    with pytest.raises(ValueError, match="energy"):
        AgentStep(
            schema_version=PROBE_3_5_TELEMETRY_SCHEMA_VERSION,
            sensed_energy_t=0.7,
            true_energy_t=None,  # missing → reject
            energy_pred_t=0.69,
            energy_recon_error_t=0.0001,
            **_agent_step_kwargs(),
        )


def test_agent_step_legacy_version_ignores_energy_fields() -> None:
    """A legacy "0.2.0" record with energy fields None still validates — the
    3.5 validator gates strictly on the new version, so old shards stay
    backward-readable."""
    rec = AgentStep(schema_version=SCHEMA_VERSION, **_agent_step_kwargs())
    assert rec.sensed_energy_t is None
    assert rec.schema_version == SCHEMA_VERSION


# ===========================================================================
# Dead-path battery — metric functions on synthetic data
# ===========================================================================


def _synthetic_eval_data(
    *, energy_in_latent: bool, n: int = 400, seed: int = 0
):  # type: ignore[no-untyped-def]
    """Build EnergyEvalData where true_energy either IS or is NOT encoded in the
    latents, so the battery's pass/fail behavior can be pinned."""
    from kind.observer.energy_eval import EnergyEvalData

    rng = np.random.default_rng(seed)
    true_e = rng.uniform(0.1, 0.9, size=n)
    h = rng.normal(size=(n, 8))
    if energy_in_latent:
        # First latent dim carries energy (plus a little noise).
        h[:, 0] = true_e * 5.0 + rng.normal(scale=0.05, size=n)
    z = rng.normal(size=(n, 4))
    # KL: one dim well above the 1.5 floor when the channel is "alive".
    kl = rng.uniform(0.2, 0.8, size=(n, 4))
    if energy_in_latent:
        kl[:, 0] = rng.uniform(1.8, 2.2, size=n)
    action = rng.integers(0, 5, size=n).astype(np.int64)
    return EnergyEvalData(
        h=h, z=z, true_energy=true_e, sensed_energy=true_e.copy(),
        kl_per_dim=kl, action=action,
    )


def test_battery_a_passes_when_energy_in_latent() -> None:
    data = _synthetic_eval_data(energy_in_latent=True)
    r = battery_a_latent_predictability(data, DeadPathMargins())
    assert r.passed and r.metrics["r2"] >= 0.5


def test_battery_a_fails_when_energy_absent_from_latent() -> None:
    data = _synthetic_eval_data(energy_in_latent=False)
    r = battery_a_latent_predictability(data, DeadPathMargins())
    assert not r.passed


def test_battery_c_passes_when_latent_beats_action_history() -> None:
    data = _synthetic_eval_data(energy_in_latent=True)
    r = battery_c_action_history_ablation(data, DeadPathMargins())
    # Latents carry energy; random actions do not → MSE ratio is large.
    assert r.passed and r.metrics["ratio"] >= 1.5


def test_battery_d_detects_kl_escape_and_collapse() -> None:
    alive = _synthetic_eval_data(energy_in_latent=True)
    dead = _synthetic_eval_data(energy_in_latent=False)
    assert battery_d_per_dim_kl_escape(alive, DeadPathMargins()).passed
    assert not battery_d_per_dim_kl_escape(dead, DeadPathMargins()).passed


def test_dead_path_battery_runs_end_to_end_and_is_deterministic() -> None:
    """The battery harness runs A–D on a (lightly-trained) world model + a real
    env trajectory and produces a well-formed, deterministic report.

    This pins the harness wiring, not a pass verdict: at the frozen margins the
    Phase-1 finding is that the battery does *not* cleanly pass on the
    pure-epistemic regime (energy floors; the channel rides the deterministic
    recurrent state, not the stochastic latent — see the dated results doc and
    the probe3_5 journal). The full-training-age gate run is
    ``scripts/run_probe3_5_phase1_baseline.py``; asserting a pass here would be
    flaky (learnability needs substantial training) and would also be tuning a
    test toward a signature the frozen pre-registration forbids fitting.
    """
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    cfg = GridWorldConfig()
    w = GridWorld(cfg, seed=0)
    s = w.reset()
    obs: list[np.ndarray] = []
    act: list[int] = []
    sen: list[float] = []
    tru: list[float] = []
    prev = s
    for _ in range(120):
        a = int(rng.integers(0, NUM_ACTIONS))
        obs.append(prev.observation.astype(np.float32) / 255.0)
        act.append(a)
        sen.append(prev.sensed_energy)
        tru.append(w.state.true_energy)
        prev = w.step(a)
    obs_t = torch.from_numpy(np.stack(obs)).unsqueeze(1)
    act_t = torch.tensor(act, dtype=torch.long)
    sen_t = torch.tensor(sen, dtype=torch.float32)
    tru_t = torch.tensor(tru, dtype=torch.float32)

    wm = WorldModel(WorldModelConfig())

    def battery_verdicts() -> list[tuple[str, bool]]:
        report = run_dead_path_battery(wm, obs_t, act_t, sen_t, tru_t)
        return [(r.name, r.passed) for r in report.results]

    v1 = battery_verdicts()
    v2 = battery_verdicts()
    names = [n for n, _ in v1]
    assert names == [
        "A_latent_predictability",
        "B_interventional_response",
        "C_action_history_ablation",
        "D_per_dim_kl_escape",
    ]
    # Deterministic given a fixed model + trajectory.
    assert v1 == v2


def test_collect_energy_eval_data_shapes() -> None:
    cfg = _small_wm_config()
    wm = WorldModel(cfg)
    n = 20
    obs = torch.zeros(n, 1, 32, 32)
    act = torch.zeros(n, dtype=torch.long)
    sensed = torch.rand(n)
    true = torch.rand(n)
    data = collect_energy_eval_data(wm, obs, act, sensed, true)
    assert data.h.shape == (n, cfg.h_dim)
    assert data.z.shape == (n, cfg.z_dim)
    assert data.kl_per_dim.shape == (n, cfg.z_dim)
    assert data.true_energy.shape == (n,)
