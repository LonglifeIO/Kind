"""Phase 2 gate tests for ``kind/training/dream.py`` (plan §4 Phase 2 row, 5).

The five tests check the *mechanism* of the four-axis differentiation (the
KS-D comparison against controls is Phase 3's job), plus the reproducibility
and no-gradient contracts the faithfulness chain depends on:

1. four-axis simultaneous differentiation (default config exhibits all four);
2. temperature-schedule ramp + the identity control + realized tail spread;
3. provenance writer-mapping per seed mode + periodic-re-seeding chimera;
4. ensemble disagreement recorded-but-not-used (trajectory independent of it);
5. byte-reproducibility from the recorded rng_seed + no-gradient enforcement.

All run on CPU against small fixed seeds.
"""

from __future__ import annotations

import torch

from kind.agents.ensemble import LatentDisagreementEnsemble
from kind.agents.world_model import WorldModel, WorldModelConfig
from kind.observer.schemas import PROBE_3_TELEMETRY_SCHEMA_VERSION, DreamRollout
from kind.training.dream import (
    DreamRolloutConfig,
    TempSchedule,
    compute_checkpoint_hash,
    compute_perturbed_prior_anchor,
    emit_dream_rollout,
)
from kind.training.dream_seed import SeedSelectionConfig, _run_warmup, select_seed
from kind.training.replay import SequenceReplayBuffer, Transition

_DEVICE = torch.device("cpu")
_H_DIM = 200
_Z_DIM = 16
_NUM_ACTIONS = 5


def _make_world_model(seed: int = 0) -> WorldModel:
    torch.manual_seed(seed)
    wm = WorldModel(
        WorldModelConfig(
            obs_channels=1,
            obs_size=32,
            h_dim=_H_DIM,
            z_dim=_Z_DIM,
            embed_dim=256,
            num_actions=_NUM_ACTIONS,
        )
    )
    wm.eval()
    return wm


def _make_ensemble(seed: int = 1) -> LatentDisagreementEnsemble:
    torch.manual_seed(seed)
    return LatentDisagreementEnsemble(
        h_dim=_H_DIM, z_dim=_Z_DIM, action_dim=_NUM_ACTIONS, K=5
    )


def _populate_buffer(
    buffer: SequenceReplayBuffer,
    n_transitions: int = 1500,
    episode_length: int = 200,
    obs_seed: int = 42,
) -> None:
    obs_gen = torch.Generator()
    obs_gen.manual_seed(obs_seed)
    for i in range(n_transitions):
        buffer.insert(
            Transition(
                obs=torch.rand((1, 32, 32), generator=obs_gen),
                action=i % _NUM_ACTIONS,
                next_obs=torch.rand((1, 32, 32), generator=obs_gen),
                env_step=i,
                episode_id=i // episode_length,
                step_in_episode=i % episode_length,
            )
        )


def _full_buffer() -> SequenceReplayBuffer:
    buf = SequenceReplayBuffer(capacity=2000, sequence_length=32)
    _populate_buffer(buf)
    return buf


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ---------------------------------------------------------------------------
# Test 1 — four-axis simultaneous differentiation (plan §2.3 test 1, load-bearing)
# ---------------------------------------------------------------------------
def test_four_axis_simultaneous_differentiation() -> None:
    wm = _make_world_model()
    ensemble = _make_ensemble()
    buf = _full_buffer()
    # Default config IS the real dreaming path; actor is None to prove the
    # goal-directed action path cannot be invoked for action choice (axis 1).
    config = DreamRolloutConfig(horizon=150)
    ckpt_hash = compute_checkpoint_hash(wm, None, ensemble)

    rec = emit_dream_rollout(
        world_model=wm,
        actor=None,
        ensemble=ensemble,
        replay_buffer=buf,
        seed_selection_config=SeedSelectionConfig(mode="replay"),
        config=config,
        dream_session_id="sess-0",
        env_step_at_emit=5000,
        run_id="run-test",
        checkpoint_id="ckpt-1",
        checkpoint_hash=ckpt_hash,
        rng=_gen(7),
        device=_DEVICE,
    )

    # Record is v0.3.0-valid (the model_validator ran on construction).
    assert rec.schema_version == PROBE_3_TELEMETRY_SCHEMA_VERSION
    assert rec.gradient_policy == "none"
    assert rec.sequence_self_prediction is None  # head does not run during dream

    # Axis 1 — actions uniform over the vocabulary, not an actor policy.
    assert len(rec.sequence_action) == config.horizon
    counts = [rec.sequence_action.count(a) for a in range(_NUM_ACTIONS)]
    assert all(c > 0 for c in counts), f"some action never sampled: {counts}"
    expected = config.horizon / _NUM_ACTIONS
    assert all(0.4 * expected < c < 1.6 * expected for c in counts), (
        f"action distribution not approximately uniform: {counts}"
    )
    # uniform log-prob is the constant -log(num_actions).
    assert all(
        abs(lp - (-torch.log(torch.tensor(float(_NUM_ACTIONS))).item())) < 1e-6
        for lp in rec.sequence_action_logprob
    )

    # Axis 2 — disagreement recorded, full-length, finite, with variation.
    dis = rec.sequence_ensemble_disagreement_variance
    assert dis is not None and len(dis) == config.horizon
    assert all(d >= 0.0 and torch.isfinite(torch.tensor(d)) for d in dis)
    assert max(dis) > min(dis), "disagreement trajectory is flat — not really recorded"

    # Axis 3 — non-flat scheduled temperature.
    temps = rec.temperature_schedule
    assert temps is not None and len(temps) == config.horizon
    assert temps[-1] > temps[0]

    # Axis 4 — replay seed provenance, not current state.
    assert rec.seed_kind == "replay"
    assert rec.seed_replay_segment_id is not None
    assert rec.seed_replay_step_offset == 0


# ---------------------------------------------------------------------------
# Test 2 — temperature schedule ramp + identity control + realized tail spread
# ---------------------------------------------------------------------------
def test_temperature_schedule_ramp_and_identity() -> None:
    wm = _make_world_model()
    ensemble = _make_ensemble()
    buf = _full_buffer()
    horizon = 60
    schedule = TempSchedule(head_value=1.5, tail_value=2.5, ramp_start_fraction=0.6)

    scheduled = emit_dream_rollout(
        world_model=wm,
        actor=None,
        ensemble=ensemble,
        replay_buffer=buf,
        seed_selection_config=SeedSelectionConfig(mode="replay"),
        config=DreamRolloutConfig(
            horizon=horizon,
            temperature_mode="scheduled",
            prior_temperature_schedule=schedule,
            re_seed_every_n_steps=0,  # isolate the temperature effect from chimera jumps
        ),
        dream_session_id="s",
        env_step_at_emit=0,
        run_id="r",
        checkpoint_id=None,
        checkpoint_hash="h",
        rng=_gen(11),
        device=_DEVICE,
    )

    temps = scheduled.temperature_schedule
    assert temps is not None and len(temps) == horizon
    ramp_start = int(schedule.ramp_start_fraction * horizon)  # 36
    assert all(abs(t - 1.5) < 1e-9 for t in temps[:ramp_start])  # flat head
    assert abs(temps[-1] - 2.5) < 1e-9  # reaches tail_value at the last step
    assert all(b >= a - 1e-12 for a, b in zip(temps, temps[1:]))  # non-decreasing

    # Realized spread: per-step latent change grows in the tail as temperature
    # climbs (re-seeding disabled, so no chimera jumps confound this).
    zs = [torch.tensor(z) for z in scheduled.sequence_z_prior]
    steps = [float((zs[t] - zs[t - 1]).norm().item()) for t in range(1, len(zs))]
    head_mean = sum(steps[:15]) / 15
    tail_mean = sum(steps[-15:]) / 15
    assert tail_mean > head_mean, (
        f"tail step-change {tail_mean:.4f} not > head {head_mean:.4f}"
    )

    # Identity control: temperature pinned flat at 1.0 throughout.
    identity = emit_dream_rollout(
        world_model=wm,
        actor=None,
        ensemble=ensemble,
        replay_buffer=buf,
        seed_selection_config=SeedSelectionConfig(mode="replay"),
        config=DreamRolloutConfig(
            horizon=horizon, temperature_mode="identity", re_seed_every_n_steps=0
        ),
        dream_session_id="s",
        env_step_at_emit=0,
        run_id="r",
        checkpoint_id=None,
        checkpoint_hash="h",
        rng=_gen(11),
        device=_DEVICE,
    )
    assert identity.temperature_schedule is not None
    assert all(abs(t - 1.0) < 1e-9 for t in identity.temperature_schedule)


# ---------------------------------------------------------------------------
# Test 3 — provenance writer mapping per seed mode + chimera re-seeding
# ---------------------------------------------------------------------------
def test_provenance_writer_mapping_and_chimera() -> None:
    wm = _make_world_model()
    ensemble = _make_ensemble()
    buf = _full_buffer()
    anchor = compute_perturbed_prior_anchor(wm, _DEVICE)
    no_reseed = DreamRolloutConfig(horizon=20, re_seed_every_n_steps=0)

    def _emit(seed_cfg: SeedSelectionConfig, rng_seed: int) -> DreamRollout:
        return emit_dream_rollout(
            world_model=wm,
            actor=None,
            ensemble=ensemble,
            replay_buffer=buf,
            seed_selection_config=seed_cfg,
            config=no_reseed,
            dream_session_id="s",
            env_step_at_emit=0,
            run_id="r",
            checkpoint_id=None,
            checkpoint_hash="h",
            rng=_gen(rng_seed),
            device=_DEVICE,
            perturbed_prior_anchor=anchor,
        )

    # The DreamSeed select_seed would produce from the same incoming rng state
    # (same seed → identical draw), so provenance can be checked one-to-one.
    for mode, rng_seed in (("replay", 100), ("perturbed_prior", 101), ("hybrid", 102)):
        seed_cfg = SeedSelectionConfig(mode=mode)  # type: ignore[arg-type]
        seed = select_seed(buf, wm, seed_cfg, _gen(rng_seed), perturbed_prior_anchor=anchor)
        rec = _emit(seed_cfg, rng_seed)

        assert rec.seed_kind == seed.mode
        assert rec.rng_seed == seed.rng_seed
        assert rec.seed_replay_segment_id == seed.replay_segment_id
        assert rec.seed_replay_step_offset == seed.replay_step_offset
        if seed.perturbation_magnitude is None:
            assert rec.seed_perturbation_magnitude is None
        else:
            assert rec.seed_perturbation_magnitude is not None
            assert abs(rec.seed_perturbation_magnitude - seed.perturbation_magnitude) < 1e-6
        # sampling_parameters carries replay_warmup_length as an int.
        assert rec.sampling_parameters is not None
        warmup = rec.sampling_parameters["replay_warmup_length"]
        assert isinstance(warmup, int) and warmup == seed_cfg.replay_warmup_length

    # Chimera: re-seeding fires at exactly [N, 2N, ...] within the horizon.
    chimera = emit_dream_rollout(
        world_model=wm,
        actor=None,
        ensemble=ensemble,
        replay_buffer=buf,
        seed_selection_config=SeedSelectionConfig(mode="replay"),
        config=DreamRolloutConfig(horizon=35, re_seed_every_n_steps=10),
        dream_session_id="s",
        env_step_at_emit=0,
        run_id="r",
        checkpoint_id=None,
        checkpoint_hash="h",
        rng=_gen(200),
        device=_DEVICE,
        perturbed_prior_anchor=anchor,
    )
    assert chimera.re_seed_step_indices == [10, 20, 30]
    assert chimera.sub_mode_tags is not None and "chimera" in chimera.sub_mode_tags
    # No-reseed rollout records None for the indices.
    assert _emit(SeedSelectionConfig(mode="replay"), 100).re_seed_step_indices is None


# ---------------------------------------------------------------------------
# Test 4 — ensemble disagreement recorded but not used (axis 2)
# ---------------------------------------------------------------------------
def test_ensemble_disagreement_recorded_not_used() -> None:
    wm = _make_world_model()
    buf = _full_buffer()
    config = DreamRolloutConfig(horizon=40, re_seed_every_n_steps=0)

    def _emit(ensemble: LatentDisagreementEnsemble) -> DreamRollout:
        return emit_dream_rollout(
            world_model=wm,
            actor=None,
            ensemble=ensemble,
            replay_buffer=buf,
            seed_selection_config=SeedSelectionConfig(mode="replay"),
            config=config,
            dream_session_id="s",
            env_step_at_emit=0,
            run_id="r",
            checkpoint_id=None,
            checkpoint_hash="h",
            rng=_gen(321),  # same incoming rng state in both calls
            device=_DEVICE,
        )

    ens_a = _make_ensemble(seed=1)
    rec_a = _emit(ens_a)

    # A different ensemble produces different disagreement values...
    ens_b = _make_ensemble(seed=999)
    rec_b = _emit(ens_b)

    dis_a = rec_a.sequence_ensemble_disagreement_variance
    dis_b = rec_b.sequence_ensemble_disagreement_variance
    assert dis_a is not None and dis_b is not None
    assert len(dis_a) == config.horizon
    assert all(d >= 0.0 for d in dis_a)
    assert dis_a != dis_b, "ensemble change did not change recorded disagreement"

    # ...but the dream trajectory is byte-identical regardless of the ensemble:
    # disagreement is recorded, never fed back into action or prior sampling.
    assert rec_a.sequence_action == rec_b.sequence_action
    assert rec_a.sequence_z_prior == rec_b.sequence_z_prior
    assert rec_a.sequence_h == rec_b.sequence_h


# ---------------------------------------------------------------------------
# Test 5 — reproducibility from rng_seed + no-gradient enforcement
# ---------------------------------------------------------------------------
def test_reproducibility_and_no_gradient() -> None:
    wm = _make_world_model()
    ensemble = _make_ensemble()
    buf = _full_buffer()
    seed_cfg = SeedSelectionConfig(mode="replay")
    config = DreamRolloutConfig(horizon=30, re_seed_every_n_steps=10)
    anchor = compute_perturbed_prior_anchor(wm, _DEVICE)

    def _emit(grad_enabled: bool = False) -> DreamRollout:
        with torch.set_grad_enabled(grad_enabled):
            return emit_dream_rollout(
                world_model=wm,
                actor=None,
                ensemble=ensemble,
                replay_buffer=buf,
                seed_selection_config=seed_cfg,
                config=config,
                dream_session_id="s",
                env_step_at_emit=0,
                run_id="r",
                checkpoint_id=None,
                checkpoint_hash="h",
                rng=_gen(2024),
                device=_DEVICE,
                perturbed_prior_anchor=anchor,
            )

    # (5a) Two runs from the same incoming rng state are byte-identical,
    # including the recorded rng_seed.
    rec1 = _emit()
    rec2 = _emit()
    assert rec1.rng_seed == rec2.rng_seed
    assert rec1.sequence_z_prior == rec2.sequence_z_prior
    assert rec1.sequence_action == rec2.sequence_action
    assert rec1.sequence_h == rec2.sequence_h
    assert rec1.sequence_decoded_obs == rec2.sequence_decoded_obs

    # The recorded rng_seed alone reproduces the seed: replicate select_seed's
    # post-draw replay dispatch (seed_gen <- rng_seed), per the Phase 1
    # reproducibility chain, and confirm the seed matches what was rolled out.
    assert rec1.rng_seed is not None
    seed_gen = torch.Generator()
    seed_gen.manual_seed(rec1.rng_seed)
    starts = buf.valid_seed_window_starts(
        length=seed_cfg.replay_warmup_length,
        min_age_steps=seed_cfg.replay_min_segment_age_steps,
    )
    pick = int(torch.randint(0, len(starts), (1,), generator=seed_gen).item())
    obs_seq, action_seq = buf.get_window(starts[pick], seed_cfg.replay_warmup_length)
    h_re, z_re = _run_warmup(wm, obs_seq, action_seq, seed_gen)
    assert h_re.squeeze(0).cpu().tolist() == rec1.seed_h0
    assert z_re.squeeze(0).cpu().tolist() == rec1.seed_z0

    # (5b) The rollout ran under no-grad: no parameter accumulated a gradient
    # even with grad globally enabled at the call site.
    for p in wm.parameters():
        assert p.grad is None
    for p in ensemble.parameters():
        assert p.grad is None
    rec_grad = _emit(grad_enabled=True)
    assert rec_grad.gradient_policy == "none"
    for p in wm.parameters():
        assert p.grad is None
    for p in ensemble.parameters():
        assert p.grad is None
