"""Phase 3b actor — policy and analytic-gradient training loss.

The actor reads ``PolicyView`` only — the synthesis §Q5 self-opacity
boundary is enforced at the import level (``TelemetryView`` is never
imported here), at the type-signature level (``forward(view: PolicyView)``),
and at the dataclass-frozen-ness level. The actor's intrinsic reward is
the ensemble disagreement variance (synthesis §Q2 — disagreement, *not*
posterior-prior KL, on Watts-intuition grounds); training is via
DreamerV1-style analytic gradients through the differentiable latent
dynamics, with the world model and ensemble parameters frozen during the
actor's loss so neither receives gradients from this path.

Discrete action sampling in imagination uses straight-through
Gumbel-Softmax: forward returns a hard one-hot, backward uses the soft
softmax path. The action one-hot is multiplied through the world model's
and the ensemble's action embedding matrices to produce differentiable
embeddings, which is what makes the imagined trajectory differentiable
end-to-end in the actor's parameters. Sampling at env-step time uses a
plain ``Categorical`` — non-differentiable, but env-step doesn't need
gradients.

The pragmatic prior was uniform (zero) from Probe 1 through Probe 3.5
Phase 1; the formula structure ``-mean(sum_τ epistemic + sum_τ pragmatic)``
was kept as scaffolding for that span. **Probe 3.5 Phase 2 fills the
scaffold**: when an ``EnergyPreferenceConfig`` is passed to
``imagine_and_compute_loss``, the pragmatic term is the saturating
Gaussian log-preference over ``world_model.decode_energy`` along the
imagined rollout (``kind/agents/preference.py``; synthesis DP5/DP6,
Amendment 02 geometry). The composition stays coefficient-free —
precision is the dominance-relevant weight; there is no β and none is
to be added. With no preference passed, the Probe-1 zero scaffold is
preserved exactly.

**Probe 1.5 v2 self-attention affordance.** The actor reads a scalar
``self_prediction_error`` field on ``PolicyView`` alongside ``(h, z)``.
The input layer's weight matrix gains one additional column (the
``+1`` in ``input_dim = h_dim + z_dim + 1``); the new column is
small-Gaussian-initialized at construction (``N(0, 0.01)``; plan §6
row 15 documented escalation from the original zero-init default,
triggered by Phase 7's column-is-zero finding — the imagine-only
training path leaves a zero-initialized column at zero indefinitely).
Forward at env-step time concatenates the scalar with ``(h, z)``
before projection. Forward during imagination
(``imagine_and_compute_loss``) feeds zero for the scalar at every
imagined step (the mask-via-zero-feed convention from plan §2.3);
the gradient on the new column from this path is mathematically
zero, so the column's values stay at their initialization. The
small-Gaussian draw is what makes the actor's policy non-invariant
to the scalar from step 0.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kind.agents.ensemble import LatentDisagreementEnsemble
from kind.agents.preference import EnergyPreferenceConfig, energy_log_preference
from kind.agents.views import PolicyView
from kind.agents.world_model import WorldModel


@dataclass(frozen=True)
class ActionOutput:
    """One forward pass's per-batch-element action information.

    The schema's ``AgentStep`` consumes ``action_t``, ``action_logprob_t``,
    and ``policy_entropy_t`` directly from these fields; ``logits`` is
    kept for downstream use (e.g. dream-rollout telemetry, off-policy
    importance weighting if Phase 5+ ever needs it).
    """

    action: Tensor
    logprob: Tensor
    entropy: Tensor
    logits: Tensor


@contextmanager
def _frozen_params(*modules: nn.Module) -> Iterator[None]:
    """Set ``requires_grad=False`` on every parameter of the given modules
    for the duration of the context, then restore the original state.

    This is the synthesis-§Q2 / plan-§2.6 detachment pattern for the
    actor's imagination loss: the world model and ensemble forward
    operations remain in the autograd graph (their values are needed to
    compute gradients on the actor's parameters), but the parameters
    themselves do not receive gradients. Setting ``requires_grad=False``
    on a parameter does not stop gradient *flow* through the operation —
    it only stops gradient *accumulation* on that parameter — which is
    exactly the semantics required.

    Restoration happens in a ``finally`` so an exception during the
    actor's loss computation does not leave the world model permanently
    frozen.
    """
    saved: list[tuple[nn.Parameter, bool]] = []
    try:
        for module in modules:
            for parameter in module.parameters():
                saved.append((parameter, parameter.requires_grad))
                parameter.requires_grad_(False)
        yield
    finally:
        for parameter, was_grad in saved:
            parameter.requires_grad_(was_grad)


def _scalar_to_column(scalar: Tensor, batch_size: int) -> Tensor:
    """Normalize a self-prediction-error scalar to shape ``(batch_size, 1)``
    for concatenation with ``(h, z)``.

    Plan §2.3 specifies the scalar as shape ``()`` (env-step path,
    ``B=1``) or ``(B,)`` (batched paths). This helper handles both
    uniformly:

    - ``()`` → expand to ``(batch_size,)`` then unsqueeze to
      ``(batch_size, 1)``.
    - ``(B,)`` → unsqueeze to ``(B, 1)``; if ``B == 1`` and
      ``batch_size > 1``, expand to ``(batch_size, 1)``.
    - ``(B, 1)`` → already correct.

    Any other shape raises ``ValueError`` (the runner is what knows the
    right shape; a malformed scalar is a runner bug, not an actor
    fallback case).
    """
    if scalar.dim() == 0:
        return scalar.expand(batch_size).unsqueeze(-1)
    if scalar.dim() == 1:
        if scalar.shape[0] == 1 and batch_size > 1:
            return scalar.expand(batch_size).unsqueeze(-1)
        return scalar.unsqueeze(-1)
    if scalar.dim() == 2 and scalar.shape[-1] == 1:
        return scalar
    raise ValueError(
        f"self_prediction_error must be shape () or (B,) or (B, 1); "
        f"got shape {tuple(scalar.shape)}"
    )


class Actor(nn.Module):
    """Two-hidden-layer MLP policy over discrete actions.

    Reads only ``PolicyView`` — the concat of ``h``, ``z``, and (Probe
    1.5 v2) the scalar ``self_prediction_error`` is performed inside
    ``forward`` per the synthesis §Q5 / plan §7 convention that the
    view stays a pure data container. Actions are sampled categorically
    at env-step time; for training, the ``imagine_and_compute_loss``
    method unrolls the world model in latent space and produces a
    differentiable loss in the actor's parameters.

    The input layer's ``in_features`` is ``h_dim + z_dim + 1``. The
    final column (corresponding to the scalar input) is small-Gaussian-
    initialized at construction (``N(0, 0.01)``; plan §6 row 15
    documented escalation from the original zero-init default after
    Phase 7's column-is-zero finding established that the imagine-only
    training path leaves a zero-initialized column at zero
    indefinitely). The remaining columns initialize via PyTorch's
    Linear default (Kaiming-uniform). The ``+1`` shifts the Kaiming-
    uniform bound by ``1/sqrt(fan_in)``'s dependence on ``fan_in``,
    so the existing columns' init values differ slightly from a
    Probe-1-shaped actor's; this is acceptable per plan §2.3 (no
    determinism contract pinned specific actor output values from a
    seed).
    """

    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        action_dim: int,
        mlp_hidden: int = 200,
    ) -> None:
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.action_dim = action_dim

        input_dim = h_dim + z_dim + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, action_dim),
        )

        # Small-Gaussian init on the final column of the first layer's
        # weight matrix (the column corresponding to the scalar
        # self-prediction-error input). Plan §6 row 15 documented
        # escalation from zero-init, triggered by Phase 7's column-is-
        # zero finding: the imagine path's mask-via-zero-feed leaves
        # the gradient on the new column mathematically zero, so a
        # zero-initialized column stays at zero across training; the
        # small-Gaussian draw is what makes the actor's policy
        # non-invariant to the scalar from step 0. The bias term and
        # the existing columns retain their Kaiming-uniform / uniform
        # init.
        first_layer = cast(nn.Linear, self.net[0])
        with torch.no_grad():
            first_layer.weight[:, h_dim + z_dim:].normal_(mean=0.0, std=0.01)

    # ---- env-step API -----------------------------------------------

    def forward(self, view: PolicyView) -> ActionOutput:
        """Sample an action from the categorical policy over ``view``.

        The runner calls this at env-step time. Sampling is a plain
        ``Categorical.sample`` — non-differentiable, but env-step does
        not need gradients. For training, see ``imagine_and_compute_loss``.

        Forward concatenates ``(view.h, view.z,
        view.self_prediction_error)`` (the scalar normalized to shape
        ``(B, 1)``) before projection through the input layer.
        """
        scalar_col = _scalar_to_column(
            view.self_prediction_error, batch_size=view.h.shape[0]
        )
        x = torch.cat([view.h, view.z, scalar_col], dim=-1)
        logits = self.net(x)
        dist = torch.distributions.Categorical(logits=logits)
        # Categorical's sample/log_prob/entropy are typed as untyped in the
        # stubs; casts give us a Tensor at the call site, ignores silence
        # the `no-untyped-call` warning that the stubs trigger.
        action = cast(Tensor, dist.sample())  # type: ignore[no-untyped-call]
        logprob = cast(Tensor, dist.log_prob(action))  # type: ignore[no-untyped-call]
        entropy = cast(Tensor, dist.entropy())  # type: ignore[no-untyped-call]
        return ActionOutput(
            action=action,
            logprob=logprob,
            entropy=entropy,
            logits=logits,
        )

    def act_greedy(self, view: PolicyView) -> Tensor:
        """Argmax over the policy logits — deterministic given fixed weights.

        For evaluation paths where stochastic sampling would add unwanted
        variance (e.g. running a fixed checkpoint repeatedly to compare
        trajectories). Not used in the training hot loop.
        """
        scalar_col = _scalar_to_column(
            view.self_prediction_error, batch_size=view.h.shape[0]
        )
        x = torch.cat([view.h, view.z, scalar_col], dim=-1)
        logits = self.net(x)
        return cast(Tensor, logits.argmax(dim=-1))

    # ---- training-time imagination loss -----------------------------

    def imagine_and_compute_loss(
        self,
        world_model: WorldModel,
        ensemble: LatentDisagreementEnsemble,
        h_0: Tensor,
        z_0: Tensor,
        horizon: int = 15,
        energy_preference: EnergyPreferenceConfig | None = None,
    ) -> dict[str, Tensor]:
        """Roll the world model forward for ``horizon`` steps in latent
        space and return the differentiable actor loss.

        At each imagined step ``τ``:

        1. Logits = ``self.net(concat(h_τ, z_τ, 0_τ))`` where ``0_τ`` is
           the mask-via-zero-feed convention for the scalar
           self-prediction-error during imagination (plan §2.3). At
           Probe 1.5 the scalar is fixed at zero throughout the
           imagined trajectory; the gradient through the new column is
           therefore zero from this path. The column's weights move
           only via env-step training paths once Phase 3 wires the
           runner — the synthesis §1.7(a) failure-mode (a) detection
           is what tests whether this is enough for the actor to
           develop conditioning on the scalar.
        2. Action one-hot = ``F.gumbel_softmax(logits, hard=True)`` —
           one-hot in forward, soft (straight-through) in backward.
        3. Action embeddings into the world model and the ensemble are
           computed as ``one_hot @ embedding.weight`` so the gradient
           flows from the policy logits through the trajectory.
        4. Disagreement at ``(h_τ, z_τ, ens_action_emb)`` is the
           per-step intrinsic reward (synthesis §Q2).
        5. The world model's recurrence advances ``h``; the prior
           produces the next ``z`` via reparameterised sampling
           (Normal.rsample is differentiable).

        The world model's and ensemble's parameters are frozen for the
        duration of the loss via ``_frozen_params`` — gradients flow
        through their operations but not onto their parameters. The
        actor's parameters are the only ones that receive gradients
        from ``actor_loss.backward()``.

        **Probe 3.5 Phase 2 — the scaffold is filled.** When
        ``energy_preference`` is provided, the pragmatic value is the
        saturating Gaussian log-preference over the **decoded energy
        along the imagined rollout**::

            pragmatic_value = Σ_τ energy_log_preference(
                world_model.decode_energy(h_τ, z_τ), energy_preference)

        differentiable through the (frozen) energy decoder, so the
        gradient reaches the policy through the imagined trajectory —
        the waking path only (the dream regime structurally never calls
        this method; ``tests/test_pragmatic_guards.py``). The actor
        never reads an energy quantity directly: the preference operates
        on the world model's *belief* (``decode_energy``), and
        PolicyView stays frozen at ``{h, z, self_prediction_error}``.

        **Precision is the dominance-relevant weight; do not add a β or
        any outer coefficient — the additive form is load-bearing
        (DP5/DP6, synthesis §8.3).** ``total_return = sum_disagreement
        + pragmatic_value``, coefficient-free.

        When ``energy_preference`` is ``None`` (Probe 1 → Phase 1
        behavior, and the default), ``pragmatic_value`` is identically
        zero — the original scaffold, byte-for-byte. A provided config
        with ``precision = 0`` computes the same zero with the same
        gradients (the preference is exactly zero with exactly zero
        gradient at zero precision), so the degenerate-null
        configuration is a point on the same surface.

        Args:
            world_model: the substrate's recurrent generative model.
            ensemble: the disagreement signal source.
            h_0: shape ``(B, h_dim)``; starting deterministic state.
                Detached internally to prevent gradient flow upstream.
            z_0: shape ``(B, z_dim)``; starting stochastic latent.
                Detached internally.
            horizon: number of imagined steps. Default 15 per plan §6.
            energy_preference: the fixed pragmatic preference (Probe 3.5
                Phase 2). ``None`` preserves the zero scaffold.

        Returns:
            ``{"actor_loss": scalar, "mean_disagreement": scalar,
            "policy_entropy": scalar, "mean_pragmatic_value": scalar,
            "pragmatic_share": scalar}`` — all 0-dim Tensors.
            ``mean_disagreement``, ``policy_entropy``, and
            ``mean_pragmatic_value`` are averaged over both batch and
            time, for telemetry. ``pragmatic_share`` is the
            per-training-step pragmatic/epistemic decomposition the
            pre-registration's A2b / §8.4 share signatures read:
            ``|P| / (|E| + |P|)`` over the batch-mean absolute summed
            terms (0 when both are 0). Telemetry-only — nothing in the
            loss reads it.
        """
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")

        h = h_0.detach()
        z = z_0.detach()
        batch_size = h.shape[0]
        device = h.device

        sum_disagreement = torch.zeros(batch_size, device=device)
        sum_entropy = torch.zeros(batch_size, device=device)
        sum_pragmatic = torch.zeros(batch_size, device=device)

        # Mask-via-zero-feed: the scalar self-prediction-error is zero
        # throughout imagination (plan §2.3). Allocated once outside
        # the loop and reused per step — the column it lands in has
        # zero gradient from this path regardless.
        imagined_scalar_col = torch.zeros(batch_size, 1, device=device)

        with _frozen_params(world_model, ensemble):
            for _ in range(horizon):
                # Policy at the current imagined state.
                logits = self.net(
                    torch.cat([h, z, imagined_scalar_col], dim=-1)
                )

                # Straight-through Gumbel-Softmax: hard one-hot in forward,
                # soft (reparameterised) gradient in backward.
                action_one_hot = F.gumbel_softmax(logits, tau=1.0, hard=True)

                # Differentiable action embeddings via the soft path.
                wm_action_emb = (
                    action_one_hot @ world_model.action_embedding.weight
                )
                ens_action_emb = (
                    action_one_hot @ ensemble.action_embedding.weight
                )

                # Per-step intrinsic reward: ensemble disagreement.
                disagreement = ensemble.disagreement_from_action_emb(
                    h, z, ens_action_emb
                )
                sum_disagreement = sum_disagreement + disagreement

                # Probe 3.5 Phase 2: the pragmatic preference over the
                # decoded energy *belief* at the current imagined state.
                # Differentiable through the frozen decoder; the actor
                # reads no energy quantity directly (DP4 — PolicyView is
                # unchanged). Skipped entirely under the Phase-1 scaffold
                # (energy_preference=None).
                if energy_preference is not None:
                    energy_belief = world_model.decode_energy(h, z).squeeze(-1)
                    sum_pragmatic = sum_pragmatic + energy_log_preference(
                        energy_belief, energy_preference
                    )

                # Per-step policy entropy (telemetry only — never enters
                # the loss directly). Cast + ignore as in `forward` above.
                dist = torch.distributions.Categorical(logits=logits)
                step_entropy = cast(Tensor, dist.entropy())  # type: ignore[no-untyped-call]
                sum_entropy = sum_entropy + step_entropy

                # Advance the world model: GRU recurrence, prior sample.
                # Calling gru_cell directly (rather than world_model.recurrence)
                # because we already have the action embedding; recurrence
                # would re-look-up the embedding from a long index.
                gru_input = torch.cat([z, wm_action_emb], dim=-1)
                h_next = world_model.gru_cell(gru_input, h)
                mu, log_sigma = world_model.prior(h_next)
                sigma = torch.exp(log_sigma)
                z_next = torch.distributions.Normal(mu, sigma).rsample()

                h = h_next
                z = z_next

        # Probe 3.5 Phase 2: the scaffold is filled. With a preference,
        # pragmatic_value is the summed saturating log-preference over
        # decoded energy along the imagined rollout; without one it is
        # identically zero (the Probe-1 scaffold preserved). The
        # composition is coefficient-free — precision is the weight; no
        # β exists and none is to be added (DP5/DP6, synthesis §8.3).
        pragmatic_value = (
            sum_pragmatic
            if energy_preference is not None
            else torch.zeros_like(sum_disagreement)
        )
        total_return = sum_disagreement + pragmatic_value

        actor_loss = -total_return.mean()
        mean_disagreement = sum_disagreement.mean() / horizon
        policy_entropy = sum_entropy.mean() / horizon
        mean_pragmatic_value = pragmatic_value.mean() / horizon

        # Per-training-step pragmatic/epistemic share (telemetry only —
        # the §8.4 "pragmatic share → 1" falsification signature and the
        # A2b mid-band share band read this). |P| / (|E| + |P|), 0 when
        # both terms are 0.
        epistemic_abs = sum_disagreement.abs().mean()
        pragmatic_abs = pragmatic_value.abs().mean()
        share_denominator = epistemic_abs + pragmatic_abs
        pragmatic_share = torch.where(
            share_denominator > 0,
            pragmatic_abs / share_denominator,
            torch.zeros_like(share_denominator),
        )

        return {
            "actor_loss": actor_loss,
            "mean_disagreement": mean_disagreement,
            "policy_entropy": policy_entropy,
            "mean_pragmatic_value": mean_pragmatic_value,
            "pragmatic_share": pragmatic_share,
        }


__all__ = ["ActionOutput", "Actor"]
