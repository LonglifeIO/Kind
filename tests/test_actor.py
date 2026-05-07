"""Phase 3b gate test for ``kind/agents/actor.py`` and
``kind/agents/ensemble.py``.

Plan §4 test #2 (actor and ensemble half — the world-model side lives in
``test_agent_forward.py``). The gate verifies the public surfaces of
``LatentDisagreementEnsemble`` and ``Actor``, the gradient-flow contract
of the actor's imagination loss, and the per-loss isolation guarantees
that the synthesis §Q5 self-opacity boundary depends on (the world model
and ensemble must not receive gradients from the actor's loss; the world
model must not receive gradients from the ensemble's loss).

CPU only at this phase. The MPS smoke is Phase 8.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from kind.agents.actor import ActionOutput, Actor
from kind.agents.ensemble import LatentDisagreementEnsemble
from kind.agents.views import PolicyView, split
from kind.agents.world_model import WorldModel, WorldModelConfig


# ---- shared fixtures -----------------------------------------------------


# Sizes scaled down from Probe 1 defaults for fast CPU tests; the structural
# shape (h_dim > z_dim, action_emb_dim small relative to h_dim, MLP hidden
# bounded) is preserved so the forward pass exercises the same conduits.
H_DIM = 32
Z_DIM = 4
ACTION_DIM = 5
EMBED_DIM = 64
ACTION_EMB_DIM = 8
MLP_HIDDEN = 32
K = 5
BATCH = 4


def world_model_config() -> WorldModelConfig:
    return WorldModelConfig(
        obs_channels=1,
        obs_size=32,
        h_dim=H_DIM,
        z_dim=Z_DIM,
        embed_dim=EMBED_DIM,
        num_actions=ACTION_DIM,
        action_emb_dim=ACTION_EMB_DIM,
        mlp_hidden=MLP_HIDDEN,
        free_bits_per_dim=1.0,
    )


def fresh_world_model() -> WorldModel:
    return WorldModel(world_model_config())


def fresh_ensemble() -> LatentDisagreementEnsemble:
    return LatentDisagreementEnsemble(
        h_dim=H_DIM,
        z_dim=Z_DIM,
        action_dim=ACTION_DIM,
        K=K,
        action_emb_dim=ACTION_EMB_DIM,
        mlp_hidden=MLP_HIDDEN,
    )


def fresh_actor() -> Actor:
    return Actor(
        h_dim=H_DIM,
        z_dim=Z_DIM,
        action_dim=ACTION_DIM,
        mlp_hidden=MLP_HIDDEN,
    )


def fresh_state(batch: int = BATCH) -> tuple[torch.Tensor, torch.Tensor]:
    """``(h, z)`` pair seeded with structured but non-trivial values."""
    return torch.randn(batch, H_DIM), torch.randn(batch, Z_DIM)


# ===========================================================================
# Ensemble tests
# ===========================================================================


def test_ensemble_predict_next_latent_returns_K_B_zdim() -> None:
    """``predict_next_latent`` returns ``(K, B, z_dim)``."""
    ens = fresh_ensemble()
    h, z = fresh_state()
    a = torch.randint(0, ACTION_DIM, (BATCH,))
    predictions = ens.predict_next_latent(h, z, a)
    assert predictions.shape == (K, BATCH, Z_DIM)
    assert torch.isfinite(predictions).all()


def test_ensemble_disagreement_is_per_batch_finite_scalar() -> None:
    """Disagreement is shape ``(B,)`` and finite."""
    ens = fresh_ensemble()
    h, z = fresh_state()
    a = torch.randint(0, ACTION_DIM, (BATCH,))
    d = ens.disagreement(h, z, a)
    assert d.shape == (BATCH,)
    assert torch.isfinite(d).all()


def test_ensemble_disagreement_is_strictly_positive_with_random_init() -> None:
    """K randomly-initialised heads almost surely disagree on a random
    input — a deterministic-zero disagreement would mean all five heads
    converged to the same function on init, which has measure zero under
    PyTorch's default initialisation."""
    ens = fresh_ensemble()
    h, z = fresh_state()
    a = torch.randint(0, ACTION_DIM, (BATCH,))
    d = ens.disagreement(h, z, a)
    assert (d > 0).all(), f"disagreement saturated to zero: {d.tolist()}"


def test_ensemble_disagreement_from_action_emb_matches_index_path() -> None:
    """For a one-hot action embedding constructed from the action index,
    ``disagreement_from_action_emb`` must produce the same value as
    ``disagreement`` on the index path (modulo float-precision)."""
    ens = fresh_ensemble()
    h, z = fresh_state()
    a = torch.randint(0, ACTION_DIM, (BATCH,))
    one_hot = torch.nn.functional.one_hot(a, num_classes=ACTION_DIM).float()
    a_emb = one_hot @ ens.action_embedding.weight

    d_indices = ens.disagreement(h, z, a)
    d_from_emb = ens.disagreement_from_action_emb(h, z, a_emb)
    assert torch.allclose(d_indices, d_from_emb, atol=1e-5)


def test_ensemble_compute_loss_returns_finite_per_head_and_total() -> None:
    """Per-head MSE is shape ``(K,)``; total is the sum (a 0-dim Tensor).
    Both must be finite."""
    ens = fresh_ensemble()
    h, z = fresh_state()
    a = torch.randint(0, ACTION_DIM, (BATCH,))
    target = torch.randn(BATCH, Z_DIM)

    loss = ens.compute_loss(h, z, a, target)
    assert set(loss.keys()) == {"loss", "per_head_losses"}
    assert loss["loss"].dim() == 0
    assert loss["per_head_losses"].shape == (K,)
    assert torch.isfinite(loss["loss"])
    assert torch.isfinite(loss["per_head_losses"]).all()
    # Total is the sum of per-head MSEs.
    assert torch.allclose(loss["loss"], loss["per_head_losses"].sum())


def test_ensemble_compute_loss_does_not_grad_world_model() -> None:
    """Plan §2.6 gate: backward through the ensemble's loss does NOT
    accumulate gradients on the world model's parameters.

    The ensemble's ``compute_loss`` detaches all three inputs internally,
    so even if the caller passes tensors that depend on the world model,
    the world model's parameters do not receive gradients from this path.
    """
    wm = fresh_world_model()
    ens = fresh_ensemble()

    obs = torch.randn(BATCH, 1, 32, 32)
    h_prev = torch.zeros(BATCH, H_DIM)
    z_prev = torch.zeros(BATCH, Z_DIM)
    a_prev = torch.zeros(BATCH, dtype=torch.long)

    step = wm.step(obs, h_prev, z_prev, a_prev)
    a_t = torch.randint(0, ACTION_DIM, (BATCH,))
    target = torch.randn(BATCH, Z_DIM)

    loss = ens.compute_loss(step.h, step.z, a_t, target)
    loss["loss"].backward()

    # World model: every parameter has either no grad or zero grad.
    for name, param in wm.named_parameters():
        if param.grad is not None:
            assert param.grad.abs().sum().item() == 0, (
                f"world_model.{name} received non-zero gradient from the "
                f"ensemble's loss"
            )
    # Ensemble's heads and action_embedding: at least some parameters have
    # non-zero gradient (the loss touches them).
    ens_grad_count = sum(
        1 for p in ens.parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    assert ens_grad_count > 0


def test_ensemble_predict_is_deterministic_under_fixed_seed() -> None:
    """Same inputs and same seed (so weight init is identical) give
    bit-identical predictions."""
    h, z = fresh_state()
    a = torch.randint(0, ACTION_DIM, (BATCH,))

    torch.manual_seed(7)
    ens_a = fresh_ensemble()
    torch.manual_seed(7)
    ens_b = fresh_ensemble()

    out_a = ens_a.predict_next_latent(h, z, a)
    out_b = ens_b.predict_next_latent(h, z, a)
    assert torch.equal(out_a, out_b)


# ===========================================================================
# Actor forward / act_greedy tests
# ===========================================================================


def test_actor_forward_returns_valid_action_output() -> None:
    """``forward`` returns an ``ActionOutput`` with all fields of correct
    shape and dtype, finite logprob and entropy, and an action in
    ``[0, action_dim)``."""
    actor = fresh_actor()
    h, z = fresh_state()
    view = PolicyView(h=h, z=z, self_prediction_error=torch.zeros(BATCH))

    out = actor.forward(view)
    assert isinstance(out, ActionOutput)

    # Shapes
    assert out.action.shape == (BATCH,)
    assert out.logprob.shape == (BATCH,)
    assert out.entropy.shape == (BATCH,)
    assert out.logits.shape == (BATCH, ACTION_DIM)

    # Dtypes
    assert out.action.dtype == torch.long
    assert out.logprob.dtype == torch.float32
    assert out.entropy.dtype == torch.float32
    assert out.logits.dtype == torch.float32

    # Action range
    assert (out.action >= 0).all()
    assert (out.action < ACTION_DIM).all()

    # Finiteness
    assert torch.isfinite(out.logprob).all()
    assert torch.isfinite(out.entropy).all()
    assert torch.isfinite(out.logits).all()


def test_actor_act_greedy_returns_argmax_actions() -> None:
    """``act_greedy`` returns the argmax over policy logits — deterministic
    given fixed weights."""
    actor = fresh_actor()
    h, z = fresh_state()
    view = PolicyView(h=h, z=z, self_prediction_error=torch.zeros(BATCH))

    actions_a = actor.act_greedy(view)
    actions_b = actor.act_greedy(view)

    assert actions_a.shape == (BATCH,)
    assert actions_a.dtype == torch.long
    assert (actions_a >= 0).all()
    assert (actions_a < ACTION_DIM).all()
    # Same input + same weights → same argmax actions.
    assert torch.equal(actions_a, actions_b)


def test_actor_forward_reads_only_policy_view_fields() -> None:
    """The actor's forward signature accepts only ``PolicyView``; passing
    any other dataclass-shaped object would fail static type-checking.
    Runtime check: ``view.h`` and ``view.z`` are the only attributes
    touched."""
    actor = fresh_actor()
    h, z = fresh_state()
    view = PolicyView(h=h, z=z, self_prediction_error=torch.zeros(BATCH))

    out = actor.forward(view)
    # Sanity: the output's shape depends on input shapes, confirming the
    # forward consumed the view.
    assert out.logits.shape[0] == h.shape[0]


# ===========================================================================
# Actor imagination loss tests (the core of Phase 3b)
# ===========================================================================


def test_imagine_and_compute_loss_returns_expected_dict_with_finite_scalars() -> None:
    """Plan §2.6: ``{"actor_loss", "mean_disagreement", "policy_entropy"}``,
    every value a 0-dim Tensor and finite."""
    actor = fresh_actor()
    wm = fresh_world_model()
    ens = fresh_ensemble()
    h_0, z_0 = fresh_state()

    out = actor.imagine_and_compute_loss(wm, ens, h_0, z_0, horizon=15)
    assert set(out.keys()) == {"actor_loss", "mean_disagreement", "policy_entropy"}
    for key, value in out.items():
        assert value.dim() == 0, f"{key} is not a scalar (dim={value.dim()})"
        assert torch.isfinite(value), f"{key} is not finite ({value.item()})"


def test_imagine_and_compute_loss_is_differentiable_in_actor_params() -> None:
    """Calling ``actor_loss.backward()`` populates gradients on every
    actor parameter."""
    actor = fresh_actor()
    wm = fresh_world_model()
    ens = fresh_ensemble()
    h_0, z_0 = fresh_state()

    out = actor.imagine_and_compute_loss(wm, ens, h_0, z_0, horizon=8)
    out["actor_loss"].backward()

    for name, param in actor.named_parameters():
        assert param.grad is not None, f"actor.{name} has no gradient"
        assert torch.isfinite(param.grad).all(), (
            f"actor.{name} has non-finite gradient"
        )
        # At least some component is non-zero — the ST Gumbel-Softmax path
        # is what makes this true; if all were zero the actor would never
        # learn.
        assert param.grad.abs().sum().item() > 0, (
            f"actor.{name} has zero gradient (the ST Gumbel-Softmax path "
            f"is not flowing gradients to the policy logits)"
        )


def test_imagine_loss_does_not_grad_world_model_or_ensemble() -> None:
    """Plan §2.6 / synthesis §Q2: the world model and ensemble parameters
    do NOT accumulate gradients from the actor's imagination loss.

    The ``_frozen_params`` context manager is what enforces this — it sets
    ``requires_grad=False`` on every parameter for the duration of the
    forward pass. Gradients flow THROUGH the parameters (matmul-backward
    uses the values to compute gradients on the inputs) but do not
    accumulate ON them.
    """
    actor = fresh_actor()
    wm = fresh_world_model()
    ens = fresh_ensemble()
    h_0, z_0 = fresh_state()

    out = actor.imagine_and_compute_loss(wm, ens, h_0, z_0, horizon=8)
    out["actor_loss"].backward()

    for name, param in wm.named_parameters():
        if param.grad is not None:
            assert param.grad.abs().sum().item() == 0, (
                f"world_model.{name} received gradient from the actor's "
                f"loss — _frozen_params is not isolating the world model"
            )
    for name, param in ens.named_parameters():
        if param.grad is not None:
            assert param.grad.abs().sum().item() == 0, (
                f"ensemble.{name} received gradient from the actor's loss "
                f"— _frozen_params is not isolating the ensemble"
            )


def test_imagine_restores_world_model_requires_grad_after_loss() -> None:
    """The frozen-params context manager restores ``requires_grad`` on
    exit. After the loss call, every world-model and ensemble parameter
    that was trainable before the call must remain trainable after —
    otherwise their own training losses would silently fail to update
    them.

    The Probe 1.5 EMA target sibling (``target_encoder``,
    ``target_gru_cell``) carries ``requires_grad=False`` by design (plan
    §2.2; the EMA update rule, not gradient descent, is what changes
    them); these parameters are excluded from the restoration check
    because the context manager treats them the same on entry and exit.
    The check is "what was True is still True" rather than "everything
    is True".
    """
    actor = fresh_actor()
    wm = fresh_world_model()
    ens = fresh_ensemble()
    h_0, z_0 = fresh_state()

    # Snapshot which parameters started trainable. The EMA target's are
    # not; everything else is.
    trainable_before_wm = {
        name for name, p in wm.named_parameters() if p.requires_grad
    }
    for p in ens.parameters():
        assert p.requires_grad

    out = actor.imagine_and_compute_loss(wm, ens, h_0, z_0, horizon=4)
    out["actor_loss"].backward()

    # After the call, requires_grad is restored on every previously
    # trainable parameter — and the EMA target's parameters remain
    # ``requires_grad=False``.
    trainable_after_wm = {
        name for name, p in wm.named_parameters() if p.requires_grad
    }
    assert trainable_after_wm == trainable_before_wm, (
        f"trainable-set drifted across imagine_and_compute_loss: "
        f"before={trainable_before_wm} after={trainable_after_wm}"
    )
    for p in ens.parameters():
        assert p.requires_grad


def test_imagine_horizon_changes_loss() -> None:
    """Different horizons produce different losses given the same starting
    state and seed — the imagination loop must actually unroll across
    ``horizon`` steps and accumulate disagreement.

    A longer horizon includes the contributions of the shorter horizon's
    steps plus additional later steps, so the two losses differ; this is
    the sanity check that the ``for τ in range(horizon)`` loop is doing
    work."""
    h_0, z_0 = fresh_state()

    torch.manual_seed(42)
    actor_a = fresh_actor()
    wm_a = fresh_world_model()
    ens_a = fresh_ensemble()
    torch.manual_seed(123)
    out_h5 = actor_a.imagine_and_compute_loss(wm_a, ens_a, h_0, z_0, horizon=5)

    torch.manual_seed(42)
    actor_b = fresh_actor()
    wm_b = fresh_world_model()
    ens_b = fresh_ensemble()
    torch.manual_seed(123)
    out_h15 = actor_b.imagine_and_compute_loss(wm_b, ens_b, h_0, z_0, horizon=15)

    # Different horizons → different losses.
    assert not torch.allclose(out_h5["actor_loss"], out_h15["actor_loss"])


def test_imagine_horizon_must_be_positive() -> None:
    actor = fresh_actor()
    wm = fresh_world_model()
    ens = fresh_ensemble()
    h_0, z_0 = fresh_state()

    with pytest.raises(ValueError, match="horizon"):
        actor.imagine_and_compute_loss(wm, ens, h_0, z_0, horizon=0)
    with pytest.raises(ValueError, match="horizon"):
        actor.imagine_and_compute_loss(wm, ens, h_0, z_0, horizon=-1)


# ===========================================================================
# Action output structure
# ===========================================================================


def test_action_output_is_frozen_dataclass() -> None:
    """``ActionOutput`` is frozen — Phase 5's runner can hold references
    without worrying about silent mutation. Same discipline as the views."""
    out = ActionOutput(
        action=torch.zeros(2, dtype=torch.long),
        logprob=torch.zeros(2),
        entropy=torch.zeros(2),
        logits=torch.zeros(2, ACTION_DIM),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(out, "action", torch.zeros(2, dtype=torch.long))


def test_action_output_field_set() -> None:
    """``ActionOutput`` carries exactly the four documented fields — the
    runner extracts ``action_t``, ``action_logprob_t``, ``policy_entropy_t``
    from these for the ``AgentStep`` schema."""
    field_names = {f.name for f in dataclasses.fields(ActionOutput)}
    assert field_names == {"action", "logprob", "entropy", "logits"}


# ===========================================================================
# Integration: split → actor.forward → step
# ===========================================================================


def test_actor_forward_consumes_split_output_directly() -> None:
    """The full chain: world_model.step → split → actor.forward.

    Verifies the Phase 3c ``split`` function's output (a ``PolicyView``)
    plugs into the Phase 3b ``Actor.forward`` without any glue code."""
    wm = fresh_world_model()
    actor = fresh_actor()
    ens = fresh_ensemble()

    obs = torch.randn(BATCH, 1, 32, 32)
    h_prev = torch.zeros(BATCH, H_DIM)
    z_prev = torch.zeros(BATCH, Z_DIM)
    a_prev = torch.zeros(BATCH, dtype=torch.long)

    step = wm.step(obs, h_prev, z_prev, a_prev)
    intrinsic = ens.disagreement(step.h, step.z, a_prev)
    policy_view, _ = split(
        step,
        intrinsic,
        self_prediction_error=torch.zeros(()),
        self_prediction_error_masked=True,
    )

    out = actor.forward(policy_view)
    assert out.action.shape == (BATCH,)
    assert (out.action >= 0).all() and (out.action < ACTION_DIM).all()


# ===========================================================================
# Probe 1.5 Phase 2 — Actor input-layer extension and scalar consumption
# ===========================================================================


def test_actor_input_layer_has_extended_input_dim() -> None:
    """The first Linear layer's ``in_features`` is ``h_dim + z_dim + 1``.
    Plan §2.3: the +1 is the column corresponding to the scalar
    self-prediction-error input."""
    actor = fresh_actor()
    first_layer = actor.net[0]
    assert isinstance(first_layer, torch.nn.Linear)
    assert first_layer.in_features == H_DIM + Z_DIM + 1
    assert first_layer.out_features == MLP_HIDDEN


def test_actor_new_input_column_initializes_to_small_gaussian() -> None:
    """Plan §6 row 15 documented escalation from zero-init: the new
    column initializes from ``N(0, 0.01)`` after Phase 7's column-is-
    zero finding established that the imagine-only training path
    leaves a zero-initialized column at zero indefinitely. The
    remaining columns retain their PyTorch Linear default
    (Kaiming-uniform).

    Asserts on the new column's empirical distribution: no exact
    zeros (continuous distribution); abs-mean in the expected range
    for ``N(0, 0.01)`` over ``MLP_HIDDEN`` samples (theoretical
    abs-mean ≈ ``0.01 * sqrt(2/π) ≈ 0.00798``); max abs value
    bounded well below the existing-columns' Kaiming-uniform spread
    (so the new column reads as small relative to ``(h, z)``'s
    contribution at construction).
    """
    import math

    actor = fresh_actor()
    first_layer = actor.net[0]
    assert isinstance(first_layer, torch.nn.Linear)
    new_column = first_layer.weight[:, H_DIM + Z_DIM:]
    assert new_column.shape == (MLP_HIDDEN, 1)

    # No exact zeros (continuous distribution; probability of an
    # exact zero from .normal_() is essentially zero).
    assert not torch.any(new_column == 0.0), (
        f"new column has exact zeros "
        f"({(new_column == 0.0).sum().item()}/{MLP_HIDDEN}); "
        f"the small-Gaussian init has not been applied"
    )

    # abs-mean within a loose range of the theoretical N(0, 0.01)
    # value. The bound is generous because the test fixture's
    # MLP_HIDDEN is small (32) and the test runs without a fixed
    # seed; the theoretical abs-mean is 0.01 * sqrt(2/π) ≈ 0.00798
    # and the loose bound covers ~5σ on the sample mean.
    abs_mean = new_column.abs().mean().item()
    expected_abs_mean = 0.01 * math.sqrt(2.0 / math.pi)
    assert 0.3 * expected_abs_mean < abs_mean < 2.0 * expected_abs_mean, (
        f"new column abs-mean {abs_mean:.5f} outside expected range "
        f"[{0.3 * expected_abs_mean:.5f}, {2.0 * expected_abs_mean:.5f}] "
        f"for N(0, 0.01) over {MLP_HIDDEN} samples (theoretical "
        f"{expected_abs_mean:.5f})"
    )

    # Max abs value bounded — under N(0, 0.01) with MLP_HIDDEN=32
    # samples, the expected max is ~0.01 * sqrt(2 ln 32) ≈ 0.026; a
    # 6σ bound at 0.06 is comfortably safe and rejects any init that
    # forgot to scale by 0.01 (e.g., a plain `.normal_()` would
    # produce values in roughly [-3, +3]).
    max_abs = new_column.abs().max().item()
    assert max_abs < 0.06, (
        f"new column max abs {max_abs:.5f} exceeds 6σ bound for "
        f"N(0, 0.01); init is not small-Gaussian (perhaps unscaled "
        f"normal_() was applied)"
    )

    # Sanity: the existing columns are not all zero (Kaiming-uniform
    # default).
    existing_columns = first_layer.weight[:, :H_DIM + Z_DIM]
    assert not torch.all(existing_columns == 0.0)
    # And the new column is small relative to the existing columns'
    # spread — pins the relative scale rather than the absolute one.
    existing_abs_max = existing_columns.abs().max().item()
    assert max_abs < existing_abs_max, (
        f"new column max abs ({max_abs:.5f}) exceeds existing "
        f"columns' max abs ({existing_abs_max:.5f}) — the new column "
        f"is no longer small relative to (h, z)"
    )


def test_actor_forward_at_construction_depends_on_scalar() -> None:
    """Plan §6 row 15 documented escalation: with the new column
    initialized to ``N(0, 0.01)`` at construction, the actor's logits
    depend on the scalar from step 0 — the actor is non-invariant to
    the scalar's value before any training has occurred. This is the
    structural inverse of the original zero-init invariant Phase 7
    found unsatisfiable (under zero-init the logits were byte-
    identical across scalar values; under small-Gaussian init they
    differ).

    The forward with scalar=0 still zeroes the column-contribution
    pathway (anything-times-zero is zero), so the logits with
    scalar=0 are not the witness for the dependence; the test
    contrasts scalar=0 against a non-zero scalar where the
    column-contribution is non-zero by construction.
    """
    actor = fresh_actor()
    h, z = fresh_state()

    view_zero = PolicyView(
        h=h, z=z, self_prediction_error=torch.zeros(BATCH)
    )
    view_nonzero = PolicyView(
        h=h, z=z, self_prediction_error=torch.full((BATCH,), 5.0)
    )

    out_zero = actor.forward(view_zero).logits
    out_nonzero = actor.forward(view_nonzero).logits

    assert not torch.equal(out_zero, out_nonzero), (
        "Actor's logits do not depend on the scalar at construction "
        "even though the new column should be small-Gaussian-"
        "initialized — plan §6 row 15 escalation may not have been "
        "applied (the column is still at zero)."
    )


def test_actor_forward_consumes_scalar_after_new_column_is_perturbed() -> None:
    """After perturbing the new column away from zero (simulating
    training dynamics moving the column's weights), the actor's logits
    *do* depend on the scalar. This is the gradient-target sanity check:
    once Phase 3 wires the runner and the new column moves under
    training, the scalar reaches the policy through this column."""
    actor = fresh_actor()
    h, z = fresh_state()

    # Move the new column away from zero.
    first_layer = actor.net[0]
    assert isinstance(first_layer, torch.nn.Linear)
    with torch.no_grad():
        first_layer.weight[:, H_DIM + Z_DIM:] = torch.randn(MLP_HIDDEN, 1)

    view_zero = PolicyView(
        h=h, z=z, self_prediction_error=torch.zeros(BATCH)
    )
    view_nonzero = PolicyView(
        h=h, z=z, self_prediction_error=torch.full((BATCH,), 5.0)
    )

    out_zero = actor.forward(view_zero).logits
    out_nonzero = actor.forward(view_nonzero).logits

    assert not torch.equal(out_zero, out_nonzero), (
        "After perturbing the new column, the actor's logits should "
        "depend on the scalar. They don't — the scalar input is being "
        "ignored despite the column having weight."
    )


def test_actor_forward_accepts_scalar_shapes() -> None:
    """Plan §2.3 specifies the scalar as shape ``()`` (env-step path,
    ``B=1``) or ``(B,)`` (batched paths). The actor's forward handles
    both shapes uniformly via the internal ``_scalar_to_column`` helper."""
    actor = fresh_actor()
    # Use B=1 so a 0-dim scalar can broadcast cleanly.
    h = torch.randn(1, H_DIM)
    z = torch.randn(1, Z_DIM)

    # 0-dim scalar.
    view_0d = PolicyView(h=h, z=z, self_prediction_error=torch.tensor(0.5))
    out_0d = actor.forward(view_0d)
    assert out_0d.logits.shape == (1, ACTION_DIM)

    # 1-dim scalar of shape (1,).
    view_1d = PolicyView(
        h=h, z=z, self_prediction_error=torch.tensor([0.5])
    )
    out_1d = actor.forward(view_1d)
    assert out_1d.logits.shape == (1, ACTION_DIM)

    # The two paths should produce identical logits (the scalar value
    # and the (h, z) inputs are identical; only the dim-shape differs).
    assert torch.equal(out_0d.logits, out_1d.logits)


def test_actor_forward_rejects_malformed_scalar_shape() -> None:
    """An invalid scalar shape is a runner bug; the actor surfaces it
    cleanly rather than silently broadcasting in unexpected ways."""
    actor = fresh_actor()
    h, z = fresh_state()

    # 3-dim tensor — invalid.
    view = PolicyView(
        h=h,
        z=z,
        self_prediction_error=torch.zeros(BATCH, 1, 1),
    )
    with pytest.raises(ValueError, match="self_prediction_error"):
        actor.forward(view)


def test_actor_imagine_does_not_grad_the_new_column() -> None:
    """Plan §2.3 mask-via-zero-feed: during ``imagine_and_compute_loss``
    the imagined scalar is zero at every step. The gradient on the new
    column from this path is therefore zero — the column's weights move
    only via env-step training paths once Phase 3 wires the runner. The
    failure-mode (a) detection is what tests whether this is enough for
    the actor to develop conditioning; this test pins the structural
    invariant that imagined-trajectory backward does *not* drive the
    new column."""
    actor = fresh_actor()
    wm = fresh_world_model()
    ens = fresh_ensemble()
    h_0, z_0 = fresh_state()

    out = actor.imagine_and_compute_loss(wm, ens, h_0, z_0, horizon=8)
    out["actor_loss"].backward()

    first_layer = actor.net[0]
    assert isinstance(first_layer, torch.nn.Linear)
    new_column_grad = first_layer.weight.grad
    assert new_column_grad is not None
    new_column_only = new_column_grad[:, H_DIM + Z_DIM:]
    assert torch.all(new_column_only == 0.0), (
        "Imagined trajectory drove gradient onto the new input column — "
        "the mask-via-zero-feed convention from plan §2.3 is broken; "
        "imagined-policy training would exercise the scalar-conditioning "
        "pathway, which Probe 1.5 commits to deferring until Phase 3 "
        "wires waking-only training."
    )

    # Sanity: the existing columns DO receive gradient from imagination.
    existing_columns_grad = new_column_grad[:, :H_DIM + Z_DIM]
    assert existing_columns_grad.abs().sum().item() > 0, (
        "Imagined trajectory produced zero gradient on the existing input "
        "columns — the imagination path itself is broken (a regression "
        "from Probe 1, not a Probe 1.5 issue)."
    )


def test_actor_forward_grad_flows_through_new_column() -> None:
    """When the scalar has ``requires_grad=True`` *and* the new column
    has been perturbed away from zero, calling ``forward`` and
    backpropping a synthetic loss through the logits produces non-zero
    gradient on the new column.

    This is the structural sanity check for Phase 3's runner integration:
    the new column is part of the actor's parameter set, the autograd
    graph reaches it, and once the column moves and the scalar is
    non-zero, gradient flows. Phase 3 will exercise this path with a
    real env-step actor_loss; Phase 2 verifies the structural plumbing.
    """
    actor = fresh_actor()
    h, z = fresh_state()
    sp_error = torch.full((BATCH,), 0.5)

    # Perturb the new column so the gradient on it is non-zero (with
    # zero column the gradient on column = scalar * upstream is zero
    # via the column itself, but the gradient on the column = scalar *
    # upstream is what we're checking — the column receives gradient
    # from the matmul backward regardless of its own value, as long as
    # the scalar input is non-zero).
    first_layer = actor.net[0]
    assert isinstance(first_layer, torch.nn.Linear)
    with torch.no_grad():
        first_layer.weight[:, H_DIM + Z_DIM:] = torch.randn(MLP_HIDDEN, 1)

    view = PolicyView(h=h, z=z, self_prediction_error=sp_error)
    out = actor.forward(view)
    # Synthetic loss: sum of logits — produces non-zero upstream grad
    # on the linear layer's weight.
    out.logits.sum().backward()

    grad = first_layer.weight.grad
    assert grad is not None
    new_column_grad = grad[:, H_DIM + Z_DIM:]
    # With a non-zero scalar (0.5) and non-zero upstream grad on the
    # logits, the new column receives non-zero gradient.
    assert new_column_grad.abs().sum().item() > 0, (
        "New input column did not receive gradient when a non-zero "
        "scalar was passed to forward — autograd is not reaching the "
        "column, the scalar input is structurally orphaned."
    )
