"""Phase 2b gate test for ``kind/agents/world_model.py``.

Plan §4 test #2 (world-model side; the actor and ensemble half lands in
Phase 3b): given dummy observations and a prior ``(h, z)``, the world model
returns a ``WorldModelStep`` with correct dtypes, shapes, finiteness; the
loss method returns finite scalars on all keys; backward populates gradients
on every parameter.

Smaller unit tests cover the public component methods (encode, decode,
recurrence, prior, posterior), the per-dim free-bits floor in ``loss()``,
and forward determinism under a fixed RNG seed.

CPU only at this phase. The MPS smoke is Phase 8.
"""

from __future__ import annotations

import pytest
import torch

from kind.agents.world_model import WorldModel, WorldModelConfig, WorldModelStep


# ---- shared fixtures -------------------------------------------------------


def small_config() -> WorldModelConfig:
    """Small but real config for fast CPU tests.

    All sizes scaled down from Probe 1 defaults; ratios kept intact so the
    forward pass exercises all the same conduits.
    """
    return WorldModelConfig(
        obs_channels=1,
        obs_size=32,
        h_dim=32,
        z_dim=4,
        embed_dim=64,
        num_actions=5,
        action_emb_dim=8,
        mlp_hidden=32,
        free_bits_per_dim=1.0,
    )


def fresh_inputs(
    config: WorldModelConfig, batch: int = 2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = torch.randn(batch, config.obs_channels, config.obs_size, config.obs_size)
    h = torch.zeros(batch, config.h_dim)
    z = torch.zeros(batch, config.z_dim)
    a = torch.zeros(batch, dtype=torch.long)
    return obs, h, z, a


# ---- gate test (plan §4 test #2, world-model side) -------------------------


def test_gate_world_model_step_and_loss_and_backward() -> None:
    config = small_config()
    wm = WorldModel(config)
    batch = 3
    obs, h, z, a = fresh_inputs(config, batch=batch)

    # Probe 3.5: exercise the energy channel too — fuse a sensed scalar so the
    # energy encoder/decoder are in the graph and the gate's backward populates
    # their gradients alongside the rest of the substrate.
    sensed_energy = torch.rand(batch, 1)
    step = wm.step(obs, h, z, a, sensed_energy=sensed_energy)

    # All eight WorldModelStep fields, with the shapes the plan specifies.
    # Probe 1.5 added ``self_prediction`` as field eight.
    assert isinstance(step, WorldModelStep)
    assert step.h.shape == (batch, config.h_dim)
    assert step.z.shape == (batch, config.z_dim)
    assert isinstance(step.q_params, tuple) and len(step.q_params) == 2
    assert step.q_params[0].shape == (batch, config.z_dim)
    assert step.q_params[1].shape == (batch, config.z_dim)
    assert isinstance(step.p_params, tuple) and len(step.p_params) == 2
    assert step.p_params[0].shape == (batch, config.z_dim)
    assert step.p_params[1].shape == (batch, config.z_dim)
    assert step.kl_per_dim.shape == (batch, config.z_dim)
    assert step.recon.shape == (
        batch,
        config.obs_channels,
        config.obs_size,
        config.obs_size,
    )
    assert step.embed.shape == (batch, config.embed_dim)
    assert step.self_prediction.shape == (batch, config.h_dim)

    # Dtype: float32 (default) on CPU.
    for tensor in (
        step.h,
        step.z,
        step.kl_per_dim,
        step.recon,
        step.embed,
        step.self_prediction,
    ):
        assert tensor.dtype == torch.float32

    # Finiteness — no NaN/Inf in either KL, reconstruction, or the head's
    # prediction.
    assert torch.isfinite(step.kl_per_dim).all()
    assert torch.isfinite(step.recon).all()
    assert torch.isfinite(step.h).all()
    assert torch.isfinite(step.z).all()
    assert torch.isfinite(step.self_prediction).all()

    # Loss dict — exactly the five keys (Probe 1.5 added
    # ``self_prediction_loss`` per plan §2.2), every value a finite 0-dim
    # tensor. The auxiliary target is computed via the world model's own
    # ``compute_self_prediction_target`` so the gate exercises the full
    # Phase 1 path.
    target_h = wm.compute_self_prediction_target(obs, h, z, a)
    loss = wm.loss(
        step, obs, target_h_next=target_h, sensed_energy_target=sensed_energy
    )
    expected_keys = {
        "total",
        "recon",
        "kl",
        "kl_aggregate_unclipped",
        "self_prediction_loss",
        "energy_recon_loss",
    }
    assert set(loss.keys()) == expected_keys
    for key, value in loss.items():
        assert value.dim() == 0, f"{key} is not a scalar (dim={value.dim()})"
        assert torch.isfinite(value), f"{key} is not finite ({value.item()})"

    # Backward over the combined Probe 1.5 objective populates gradients
    # on every *trainable* parameter (the EMA target's encoder + GRU
    # carry ``requires_grad=False`` and never receive gradients; the
    # frozen projection — when allocated — likewise). The runner
    # combines ``total + λ_self * self_prediction_loss``; here we use
    # λ=1 to drive the head's gradient unambiguously.
    (loss["total"] + loss["self_prediction_loss"]).backward()
    for name, param in wm.named_parameters():
        if not param.requires_grad:
            assert param.grad is None, (
                f"requires_grad=False parameter {name} unexpectedly has a "
                f"populated .grad — backward should not write into the EMA "
                f"target or the frozen projection"
            )
            continue
        assert param.grad is not None, f"no gradient on {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient on {name}"


# ---- component-level shape and behavior tests ------------------------------


def test_encode_returns_embed_dim_vector() -> None:
    config = small_config()
    wm = WorldModel(config)
    obs = torch.randn(4, config.obs_channels, config.obs_size, config.obs_size)
    embed = wm.encode(obs)
    assert embed.shape == (4, config.embed_dim)
    assert torch.isfinite(embed).all()


def test_decode_returns_observation_shaped_tensor() -> None:
    config = small_config()
    wm = WorldModel(config)
    h = torch.randn(4, config.h_dim)
    z = torch.randn(4, config.z_dim)
    recon = wm.decode(h, z)
    assert recon.shape == (
        4,
        config.obs_channels,
        config.obs_size,
        config.obs_size,
    )
    assert torch.isfinite(recon).all()


def test_recurrence_advances_h_and_keeps_shape() -> None:
    config = small_config()
    wm = WorldModel(config)
    h = torch.zeros(4, config.h_dim)
    z = torch.randn(4, config.z_dim)
    a = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    h_new = wm.recurrence(h, z, a)
    assert h_new.shape == (4, config.h_dim)
    # GRU should mix in z + action embedding; from a zero h the new h should
    # differ for non-trivial inputs.
    assert not torch.equal(h_new, h)


def test_prior_outputs_two_z_dim_tensors() -> None:
    config = small_config()
    wm = WorldModel(config)
    h = torch.randn(4, config.h_dim)
    mu, log_sigma = wm.prior(h)
    assert mu.shape == (4, config.z_dim)
    assert log_sigma.shape == (4, config.z_dim)


def test_posterior_outputs_two_z_dim_tensors() -> None:
    config = small_config()
    wm = WorldModel(config)
    h = torch.randn(4, config.h_dim)
    embed = torch.randn(4, config.embed_dim)
    mu, log_sigma = wm.posterior(h, embed)
    assert mu.shape == (4, config.z_dim)
    assert log_sigma.shape == (4, config.z_dim)


# ---- free-bits behavior ----------------------------------------------------


def test_free_bits_floor_applied_per_dim_below_threshold() -> None:
    """When per-dim KL is below the floor, kl uses the floor; when above, KL.

    With ``free_bits_per_dim=1.0`` and per-dim KL ``[0.1, 5.0, 0.5, 2.0]``,
    the clipped per-dim values are ``[1.0, 5.0, 1.0, 2.0]`` summing to 9.0;
    the unclipped sum is 7.6. ``recon`` is forced to zero by passing
    ``recon == obs_target``, so ``total`` equals the clipped KL.
    """
    config = WorldModelConfig(
        obs_channels=1,
        obs_size=32,
        h_dim=8,
        z_dim=4,
        embed_dim=16,
        num_actions=5,
        action_emb_dim=4,
        mlp_hidden=8,
        free_bits_per_dim=1.0,
    )
    wm = WorldModel(config)

    fake_obs = torch.zeros(1, 1, 32, 32)
    fake_step = WorldModelStep(
        h=torch.zeros(1, config.h_dim),
        z=torch.zeros(1, config.z_dim),
        q_params=(torch.zeros(1, config.z_dim), torch.zeros(1, config.z_dim)),
        p_params=(torch.zeros(1, config.z_dim), torch.zeros(1, config.z_dim)),
        kl_per_dim=torch.tensor([[0.1, 5.0, 0.5, 2.0]]),
        recon=fake_obs.clone(),  # recon == obs_target → recon_loss = 0
        embed=torch.zeros(1, config.embed_dim),
        self_prediction=torch.zeros(1, config.h_dim),
        energy_pred=torch.zeros(1, 1),
    )

    loss = wm.loss(fake_step, fake_obs)
    assert loss["recon"].item() == pytest.approx(0.0)
    assert loss["kl"].item() == pytest.approx(9.0)
    assert loss["kl_aggregate_unclipped"].item() == pytest.approx(7.6)
    assert loss["total"].item() == pytest.approx(9.0)


def test_free_bits_floor_does_not_clip_when_kl_above_floor() -> None:
    config = WorldModelConfig(
        obs_channels=1,
        obs_size=32,
        h_dim=8,
        z_dim=4,
        embed_dim=16,
        num_actions=5,
        action_emb_dim=4,
        mlp_hidden=8,
        free_bits_per_dim=1.0,
    )
    wm = WorldModel(config)

    fake_obs = torch.zeros(1, 1, 32, 32)
    fake_step = WorldModelStep(
        h=torch.zeros(1, config.h_dim),
        z=torch.zeros(1, config.z_dim),
        q_params=(torch.zeros(1, config.z_dim), torch.zeros(1, config.z_dim)),
        p_params=(torch.zeros(1, config.z_dim), torch.zeros(1, config.z_dim)),
        kl_per_dim=torch.tensor([[2.0, 3.0, 4.0, 5.0]]),
        recon=fake_obs.clone(),
        embed=torch.zeros(1, config.embed_dim),
        self_prediction=torch.zeros(1, config.h_dim),
        energy_pred=torch.zeros(1, 1),
    )
    loss = wm.loss(fake_step, fake_obs)
    # All values are above the floor; unclipped == clipped.
    assert loss["kl"].item() == pytest.approx(14.0)
    assert loss["kl_aggregate_unclipped"].item() == pytest.approx(14.0)


# ---- determinism -----------------------------------------------------------


def test_step_is_deterministic_given_same_seed_and_inputs() -> None:
    config = small_config()

    torch.manual_seed(7)
    wm1 = WorldModel(config)
    torch.manual_seed(7)
    wm2 = WorldModel(config)

    obs, h, z, a = fresh_inputs(config, batch=2)

    torch.manual_seed(123)
    step1 = wm1.step(obs, h, z, a)
    torch.manual_seed(123)
    step2 = wm2.step(obs, h, z, a)

    assert torch.equal(step1.h, step2.h)
    assert torch.equal(step1.z, step2.z)
    assert torch.equal(step1.recon, step2.recon)
    assert torch.equal(step1.kl_per_dim, step2.kl_per_dim)
    assert torch.equal(step1.embed, step2.embed)
    # The Probe 1.5 head's forward is deterministic in the online ``h``,
    # so identical seed + input must produce identical predictions.
    assert torch.equal(step1.self_prediction, step2.self_prediction)


# ---- output sanity ---------------------------------------------------------


def test_kl_per_dim_is_non_negative() -> None:
    """Analytical KL between two Gaussians is always ≥ 0."""
    config = small_config()
    wm = WorldModel(config)
    obs, h, z, a = fresh_inputs(config, batch=4)
    step = wm.step(obs, h, z, a)
    assert (step.kl_per_dim >= 0).all()


def test_recurrence_is_action_sensitive() -> None:
    """Different actions must produce different recurrent states."""
    config = small_config()
    wm = WorldModel(config)
    h = torch.zeros(2, config.h_dim)
    z = torch.randn(2, config.z_dim)
    a0 = torch.tensor([0, 0], dtype=torch.long)
    a1 = torch.tensor([1, 1], dtype=torch.long)
    h_a0 = wm.recurrence(h, z, a0)
    h_a1 = wm.recurrence(h, z, a1)
    assert not torch.equal(h_a0, h_a1)
