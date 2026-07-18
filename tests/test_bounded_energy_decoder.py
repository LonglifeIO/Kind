"""Probe 4.5 Phase 1 — F2 bounded energy-decoder head (config-gated).

Authority: ``docs/decisions/probe4_5_f2_bounded_decoder_2026-07-13.md``
(ADOPTED 2026-07-13) and the frozen pre-registration §3. When
``WorldModelConfig.energy_decoder_bounded`` is False (the default), the head
is the existing unbounded linear — legacy instances byte-identical. When True
(all Probe 4.5 instances), the output passes through a sigmoid, bounding
``decode_energy`` to the physical [0, 1] — removing the impossible >1 regime
that inverted the Probe 3.5 preference geometry (seek classification §2).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from kind.agents.world_model import WorldModel, WorldModelConfig


def _small_cfg(**overrides: object) -> WorldModelConfig:
    base: dict[str, object] = dict(
        h_dim=16, z_dim=4, embed_dim=8, mlp_hidden=16, energy_embed_dim=4
    )
    base.update(overrides)
    return WorldModelConfig(**base)  # type: ignore[arg-type]


def test_default_config_is_unbounded() -> None:
    """Default (and explicit False) keeps the raw linear head — the legacy
    surface is byte-identical: forward == head(elu(fc1(x))), no sigmoid."""
    for cfg in (_small_cfg(), _small_cfg(energy_decoder_bounded=False)):
        wm = WorldModel(cfg)
        assert wm.energy_decoder.bounded is False
        latent = torch.randn(7, cfg.h_dim + cfg.z_dim)
        with torch.no_grad():
            manual = wm.energy_decoder.head(
                F.elu(wm.energy_decoder.fc1(latent))
            )
            out = wm.energy_decoder(latent)
        assert torch.equal(out, manual)


def test_default_head_can_read_outside_unit_range() -> None:
    """The unbounded default can produce impossible readings under extreme
    latents (the bin-1 regime exists structurally) — this is the property F2
    removes, pinned here so the contrast is explicit."""
    wm = WorldModel(_small_cfg())
    # Push the linear head far along its weight direction.
    latent = torch.randn(64, 20) * 100.0
    with torch.no_grad():
        out = wm.decode_energy(latent[:, :16], latent[:, 16:])
    assert (out.max() > 1.0) or (out.min() < 0.0)


def test_bounded_config_gates_sigmoid() -> None:
    """energy_decoder_bounded=True → sigmoid(head(elu(fc1(x)))), exactly."""
    cfg = _small_cfg(energy_decoder_bounded=True)
    wm = WorldModel(cfg)
    assert wm.energy_decoder.bounded is True
    latent = torch.randn(7, cfg.h_dim + cfg.z_dim)
    with torch.no_grad():
        manual = torch.sigmoid(
            wm.energy_decoder.head(F.elu(wm.energy_decoder.fc1(latent)))
        )
        out = wm.energy_decoder(latent)
    assert torch.equal(out, manual)


def test_bounded_output_stays_in_unit_range_under_extreme_latents() -> None:
    """The bounded head cannot read outside [0, 1] — the >1 regime that
    inverted the preference geometry is unrepresentable."""
    cfg = _small_cfg(energy_decoder_bounded=True)
    wm = WorldModel(cfg)
    h = torch.randn(64, cfg.h_dim) * 100.0
    z = torch.randn(64, cfg.z_dim) * 100.0
    with torch.no_grad():
        out = wm.decode_energy(h, z)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_bounded_head_is_differentiable() -> None:
    """The sigmoid is smooth everywhere — gradients flow to both layers (the
    reason F2 chose sigmoid over a clamp, decision doc §'why this form')."""
    cfg = _small_cfg(energy_decoder_bounded=True)
    wm = WorldModel(cfg)
    h = torch.randn(5, cfg.h_dim, requires_grad=True)
    z = torch.randn(5, cfg.z_dim)
    out = wm.decode_energy(h, z)
    out.sum().backward()
    assert h.grad is not None
    assert torch.isfinite(h.grad).all()
    grads = [p.grad for p in wm.energy_decoder.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_bounded_recon_path_unchanged_in_form() -> None:
    """``WorldModel.loss`` computes the same weighted MSE against
    ``sensed_energy`` through the bounded head — no loss-form change."""
    cfg = _small_cfg(energy_decoder_bounded=True)
    wm = WorldModel(cfg)
    obs = torch.rand(3, 1, 32, 32)
    h = torch.zeros(3, cfg.h_dim)
    z = torch.zeros(3, cfg.z_dim)
    a = torch.zeros(3, dtype=torch.long)
    sensed = torch.rand(3, 1)
    step = wm.step(obs, h, z, a, sensed_energy=sensed)
    losses = wm.loss(step, obs, sensed_energy_target=sensed)
    energy_recon = losses["energy_recon_loss"]
    expected = F.mse_loss(step.energy_pred, sensed)
    assert torch.allclose(energy_recon, expected)
    assert torch.isfinite(losses["total"])


def test_bounded_flag_composes_with_dp9_dedicated_dims() -> None:
    """The DP9 escalation branch (decoder reads only dedicated z-dims) and the
    F2 bound are orthogonal gates — both apply when both are set."""
    cfg = _small_cfg(energy_decoder_bounded=True, energy_dedicated_dims=2)
    wm = WorldModel(cfg)
    h = torch.randn(4, cfg.h_dim) * 100.0
    z = torch.randn(4, cfg.z_dim) * 100.0
    with torch.no_grad():
        out = wm.decode_energy(h, z)
    assert out.shape == (4, 1)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
