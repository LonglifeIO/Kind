"""Probe 3.5 Phase 2 — integration tests for the filled scaffold.

The mechanism gate's integration layer (Step 4b), waking side:

- the pragmatic term's gradient **flows to the policy** through the imagined
  trajectory (waking path; the dream-side unreachability lives in
  ``tests/test_pragmatic_guards.py``);
- ``precision = 0`` reproduces the Phase-1 actor behavior **exactly** on a
  fixed-seed comparison (bit-identical loss and gradients — stronger than
  the pre-registered "statistically indistinguishable");
- the composition is **coefficient-free**: ``actor_loss ==
  −mean(sum_disagreement + pragmatic_value)`` with no outer factor (DP5);
- **no viability→capacity coupling**: the preference changes only the loss
  value, never the imagined trajectory, its length, or the epistemic term
  at loss-computation time (behavior changes only through training).
"""

from __future__ import annotations

import pytest
import torch

from kind.agents.actor import Actor
from kind.agents.ensemble import LatentDisagreementEnsemble
from kind.agents.preference import EnergyPreferenceConfig
from kind.agents.world_model import WorldModel, WorldModelConfig

H_DIM = 32
Z_DIM = 4
ACTION_DIM = 5
EMBED_DIM = 64
ACTION_EMB_DIM = 8
MLP_HIDDEN = 32
K = 5
BATCH = 4
HORIZON = 8


def _fresh_triple(seed: int) -> tuple[Actor, WorldModel, LatentDisagreementEnsemble]:
    torch.manual_seed(seed)
    wm = WorldModel(
        WorldModelConfig(
            obs_channels=1,
            obs_size=32,
            h_dim=H_DIM,
            z_dim=Z_DIM,
            embed_dim=EMBED_DIM,
            num_actions=ACTION_DIM,
            action_emb_dim=ACTION_EMB_DIM,
            mlp_hidden=MLP_HIDDEN,
        )
    )
    ensemble = LatentDisagreementEnsemble(
        h_dim=H_DIM,
        z_dim=Z_DIM,
        action_dim=ACTION_DIM,
        K=K,
        action_emb_dim=ACTION_EMB_DIM,
        mlp_hidden=MLP_HIDDEN,
    )
    actor = Actor(
        h_dim=H_DIM, z_dim=Z_DIM, action_dim=ACTION_DIM, mlp_hidden=MLP_HIDDEN
    )
    return actor, wm, ensemble


def _loss_dict(
    seed: int, energy_preference: EnergyPreferenceConfig | None
) -> dict[str, torch.Tensor]:
    """Fixed-seed imagination loss: same models, same starting state, same
    RNG stream for the Gumbel/rsample draws."""
    actor, wm, ensemble = _fresh_triple(seed)
    torch.manual_seed(seed + 1)
    h_0 = torch.randn(BATCH, H_DIM)
    z_0 = torch.randn(BATCH, Z_DIM)
    torch.manual_seed(seed + 2)  # the imagination loop's sampling stream
    return actor.imagine_and_compute_loss(
        wm, ensemble, h_0, z_0, horizon=HORIZON, energy_preference=energy_preference
    )


def test_precision_zero_reproduces_phase1_actor_exactly() -> None:
    """precision = 0 and the Phase-1 scaffold (no preference) are the same
    point: identical loss, identical decomposition. The preference at zero
    precision computes decode_energy but contributes exactly zero with
    exactly zero gradient — and consumes no RNG, so the fixed-seed
    trajectories are identical."""
    base = _loss_dict(7, None)
    zeroed = _loss_dict(7, EnergyPreferenceConfig(precision=0.0))
    for key in ("actor_loss", "mean_disagreement", "policy_entropy"):
        assert torch.equal(base[key], zeroed[key]), key
    assert zeroed["mean_pragmatic_value"].item() == 0.0
    assert zeroed["pragmatic_share"].item() == 0.0


def test_precision_zero_gradients_match_phase1_exactly() -> None:
    """Bit-identical gradients on every actor parameter — the Phase-1 actor
    training step is reproduced, not approximated."""
    grads: list[list[torch.Tensor]] = []
    for pref in (None, EnergyPreferenceConfig(precision=0.0)):
        actor, wm, ensemble = _fresh_triple(11)
        torch.manual_seed(12)
        h_0 = torch.randn(BATCH, H_DIM)
        z_0 = torch.randn(BATCH, Z_DIM)
        torch.manual_seed(13)
        out = actor.imagine_and_compute_loss(
            wm, ensemble, h_0, z_0, horizon=HORIZON, energy_preference=pref
        )
        out["actor_loss"].backward()  # type: ignore[no-untyped-call]
        grads.append(
            [
                p.grad.detach().clone()
                for p in actor.parameters()
                if p.grad is not None
            ]
        )
    assert len(grads[0]) == len(grads[1]) > 0
    for g_none, g_zero in zip(grads[0], grads[1], strict=True):
        assert torch.equal(g_none, g_zero)


def test_pragmatic_gradient_flows_to_the_policy() -> None:
    """The pragmatic component alone produces non-zero gradient on the
    actor's parameters — the pull reaches the policy through the imagined
    trajectory (via decode_energy over (h_τ, z_τ), both downstream of the
    policy's action choices)."""
    actor, wm, ensemble = _fresh_triple(23)
    torch.manual_seed(24)
    h_0 = torch.randn(BATCH, H_DIM)
    z_0 = torch.randn(BATCH, Z_DIM)
    torch.manual_seed(25)
    out = actor.imagine_and_compute_loss(
        wm,
        ensemble,
        h_0,
        z_0,
        horizon=HORIZON,
        energy_preference=EnergyPreferenceConfig(precision=50.0),
    )
    pragmatic_grads = torch.autograd.grad(
        out["mean_pragmatic_value"],
        list(actor.parameters()),
        retain_graph=True,
        allow_unused=True,
    )
    total = sum(float(g.abs().sum()) for g in pragmatic_grads if g is not None)
    assert total > 0.0, (
        "the pragmatic term produced zero gradient on every actor parameter "
        "— the pull does not reach the policy"
    )


def test_pragmatic_term_is_nonzero_out_of_band() -> None:
    """With a preference and a randomly-initialized decoder (whose outputs
    are generically out-of-band), the pragmatic term is non-zero and ≤ 0 —
    the term is live, not a dead path."""
    out = _loss_dict(31, EnergyPreferenceConfig(precision=50.0))
    assert out["mean_pragmatic_value"].item() < 0.0
    assert 0.0 < out["pragmatic_share"].item() < 1.0


def test_composition_is_coefficient_free() -> None:
    """actor_loss == −mean(sum_disagreement + pragmatic_value), reconstructed
    from the returned per-step means with no extra factor. Precision is the
    weight; there is no β (DP5/DP6, synthesis §8.3)."""
    out = _loss_dict(43, EnergyPreferenceConfig(precision=10.0))
    reconstructed = -(
        out["mean_disagreement"] + out["mean_pragmatic_value"]
    ) * HORIZON
    assert float(out["actor_loss"].detach()) == pytest.approx(
        float(reconstructed.detach()), rel=1e-5
    )


def test_no_viability_capacity_coupling_at_loss_time() -> None:
    """The preference does not alter the imagined trajectory, the horizon,
    or the epistemic term at loss-computation time — same fixed seed, wildly
    different precisions, identical disagreement and entropy. Low decoded
    energy degrades nothing about the computation (no vulnerability spiral,
    synthesis T1); the preference acts on behavior only through training."""
    weak = _loss_dict(57, EnergyPreferenceConfig(precision=0.1))
    strong = _loss_dict(57, EnergyPreferenceConfig(precision=1000.0))
    assert torch.equal(weak["mean_disagreement"], strong["mean_disagreement"])
    assert torch.equal(weak["policy_entropy"], strong["policy_entropy"])
    assert weak["mean_pragmatic_value"].item() != strong[
        "mean_pragmatic_value"
    ].item()
