"""Phase 3b ensemble of one-step latent predictors (Plan2Explore-lite).

K=5 small MLPs, each predicting the next stochastic latent ``z_{t+1}``
given the current state ``(h_t, z_t)`` and action ``a_t``. The heads are
deterministic point predictors with independent random initialisation;
the disagreement signal that drives Io's actor (synthesis §Q2) is the
biased variance across the K heads, summed over the ``z`` dimensions.

Two design choices land in this module:

1. **The ensemble has its own action embedding.** It is *not* shared with
   the world model's recurrence action embedding. The synthesis §Q2 spec
   is explicit about this: the ensemble rides on top of the world model's
   latent, but its representation choices are its own. Sharing would
   couple the ensemble's gradients to the world model's representation
   learning in a way the synthesis rules out.

2. **Inputs are detached internally in ``compute_loss``.** The ensemble's
   training step computes MSE per head against the world model's actual
   posterior latent at the next step (the target). Detaching all three
   inputs (h_t, z_t, target) ensures backward through the ensemble's loss
   only accumulates gradients on the ensemble's own parameters — never on
   the world model. This is the synthesis §Q2 commitment that "the
   ensemble rides on top of the world model's latent without modifying
   it".

The disagreement is computed two ways: ``disagreement(h, z, a_indices)``
takes a ``long`` tensor of action indices, used at env-step time and at
ensemble-training time; ``disagreement_from_action_emb(h, z, a_emb)``
takes a soft action embedding directly, used by Phase 3b's actor in
imagination where the action one-hot comes from a straight-through
Gumbel-Softmax sample (the soft path is what makes the imagined
trajectory differentiable end-to-end). Both share a private
``_heads_forward`` so the variance computation is identical in either
mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class EnsembleConfig:
    """Static configuration for ``LatentDisagreementEnsemble``.

    Defaults reflect Probe 1 starting points (synthesis §Q2, plan §6):
    K=5 heads, action embedding dim 16, MLP hidden 200. The action
    embedding dim is bounded above by 32 per the user's spec — keeping
    it small relative to the world model's representation prevents the
    ensemble from spending its capacity on action features.
    """

    h_dim: int
    z_dim: int
    action_dim: int
    K: int = 5
    action_emb_dim: int = 16
    mlp_hidden: int = 200


class LatentDisagreementEnsemble(nn.Module):
    """K independent one-step latent predictors with a disagreement variance.

    Each head is a 2-hidden-layer MLP (200 units, ELU activation, matching
    the world model's prior/posterior heads' style) that maps
    ``concat(h, z, action_embedding(a))`` to a point prediction of the
    next ``z``. The K heads share no weights and receive different random
    initialisations via PyTorch's default scheme — that initialisation
    diversity is the source of the disagreement signal.
    """

    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        action_dim: int,
        K: int = 5,
        action_emb_dim: int = 16,
        mlp_hidden: int = 200,
        lesion_constant_disagreement: bool = False,
    ) -> None:
        super().__init__()
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}")
        if action_emb_dim > 32:
            raise ValueError(
                f"action_emb_dim must be <= 32 per spec; got {action_emb_dim}"
            )
        self.K = K
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.action_emb_dim = action_emb_dim
        # Probe 2 v2 ``ensemble_constant`` lesion (plan §2.5): when True,
        # ``disagreement`` and ``disagreement_from_action_emb`` short-circuit
        # to a zero per-batch tensor, regardless of head outputs. The heads
        # still train normally so the ensemble's internal state remains
        # consistent across resume; the lesion's observable effect is
        # ``intrinsic_signal_t == 0`` in the agent_step stream and an actor
        # whose only intrinsic signal is constant. The companion
        # ``ensemble_k1`` lesion is implemented at construction time by
        # passing ``K=1`` (variance over a single head is identically zero
        # via :func:`torch.var`'s biased reduction at K=1).
        self._lesion_constant_disagreement = lesion_constant_disagreement

        self.action_embedding = nn.Embedding(action_dim, action_emb_dim)

        input_dim = h_dim + z_dim + action_emb_dim
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, mlp_hidden),
                    nn.ELU(),
                    nn.Linear(mlp_hidden, mlp_hidden),
                    nn.ELU(),
                    nn.Linear(mlp_hidden, z_dim),
                )
                for _ in range(K)
            ]
        )

    # ---- public prediction API ---------------------------------------

    def predict_next_latent(self, h: Tensor, z: Tensor, a: Tensor) -> Tensor:
        """K next-latent predictions from action *indices*.

        Args:
            h: shape ``(B, h_dim)``.
            z: shape ``(B, z_dim)``.
            a: shape ``(B,)``, dtype ``long``; integer action indices.

        Returns:
            Tensor of shape ``(K, B, z_dim)``.
        """
        a_emb = self.action_embedding(a)
        return self._heads_forward(h, z, a_emb)

    def disagreement(self, h: Tensor, z: Tensor, a: Tensor) -> Tensor:
        """Per-batch-element disagreement signal: variance across K heads,
        summed over the ``z_dim`` dimensions.

        Returns:
            Tensor of shape ``(B,)``.
        """
        predictions = self.predict_next_latent(h, z, a)
        raw = self._variance(predictions)
        if self._lesion_constant_disagreement:
            # Probe 2 v2 ``ensemble_constant`` lesion (plan §2.5): the
            # actor's intrinsic signal is forced constant-zero regardless
            # of head outputs. The lesion targets the substrate-side
            # surface by removing the actor's intrinsic motivation; the
            # head outputs themselves are untouched (they still receive
            # gradient through ``compute_loss``) so the substrate-side
            # latent dynamics are preserved. Multiplying ``raw`` by 0.0
            # keeps the autograd chain intact (backward through this
            # path computes a zero gradient on the actor's parameters)
            # rather than detaching, which would break the actor's
            # ``actor_loss.backward()`` when the only signal is the
            # lesioned one.
            return raw * 0.0
        return raw

    def disagreement_from_action_emb(
        self, h: Tensor, z: Tensor, a_emb: Tensor
    ) -> Tensor:
        """Variant of ``disagreement`` taking a soft action embedding.

        Used by Phase 3b's actor imagination loop, where the action
        one-hot comes from a straight-through Gumbel-Softmax sample and
        the embedding is computed via ``one_hot @
        self.action_embedding.weight`` to keep the gradient path through
        the policy logits intact. Distinct from ``disagreement``: that
        method takes ``long`` indices and goes through ``nn.Embedding``,
        which is *not* differentiable in the index.

        Args:
            h: shape ``(B, h_dim)``.
            z: shape ``(B, z_dim)``.
            a_emb: shape ``(B, action_emb_dim)``; a (possibly soft) action
                embedding.

        Returns:
            Tensor of shape ``(B,)``.
        """
        predictions = self._heads_forward(h, z, a_emb)
        raw = self._variance(predictions)
        if self._lesion_constant_disagreement:
            # Probe 2 v2 ``ensemble_constant`` lesion (plan §2.5): see
            # ``disagreement`` above. The actor imagines under a
            # constant-zero intrinsic signal; multiplying ``raw`` by 0.0
            # keeps the autograd chain intact (the actor's
            # ``actor_loss.backward()`` runs cleanly with zero gradient
            # contribution from this path) while pinning the value at
            # zero. A bare ``torch.zeros(...)`` would detach the chain
            # and break backward when the lesioned signal is the only
            # contribution to the loss.
            return raw * 0.0
        return raw

    # ---- training loss ----------------------------------------------

    def compute_loss(
        self,
        h_t: Tensor,
        z_t: Tensor,
        a_t: Tensor,
        z_target: Tensor,
    ) -> dict[str, Tensor]:
        """MSE loss per head against the world model's next-step latent.

        Each head trains on the same target ``z_target`` (the world
        model's actual posterior at the next step). The total loss is the
        sum over heads of the per-head MSE; per-head MSE is the mean over
        batch and ``z_dim`` of the squared error.

        All three inputs are detached internally so backward through this
        loss only accumulates gradients on the ensemble's own parameters.
        The synthesis §Q2 commitment is that the ensemble rides on top of
        the world model's latent without modifying it; detaching is the
        cleanest way to enforce that, regardless of whether the caller
        also remembered to detach.

        Args:
            h_t: shape ``(B, h_dim)``.
            z_t: shape ``(B, z_dim)``.
            a_t: shape ``(B,)``, dtype ``long``; action indices.
            z_target: shape ``(B, z_dim)``; the world model's actual
                posterior latent at the next step.

        Returns:
            A dict with ``"loss"`` (scalar Tensor — sum over heads of
            per-head MSE) and ``"per_head_losses"`` (shape ``(K,)`` —
            one MSE per head, useful for telemetry).
        """
        h_det = h_t.detach()
        z_det = z_t.detach()
        target_det = z_target.detach()

        predictions = self.predict_next_latent(h_det, z_det, a_t)
        sq_err = (predictions - target_det.unsqueeze(0)) ** 2
        # Mean over batch and z_dim → per-head scalar MSE.
        per_head_losses = sq_err.mean(dim=(1, 2))
        total_loss = per_head_losses.sum()

        return {"loss": total_loss, "per_head_losses": per_head_losses}

    # ---- internals --------------------------------------------------

    def _heads_forward(self, h: Tensor, z: Tensor, a_emb: Tensor) -> Tensor:
        """Run every head on the same ``concat(h, z, a_emb)`` input.

        Stacks the K outputs along a leading dim so the variance
        computation can reduce across heads with ``dim=0``.
        """
        x = torch.cat([h, z, a_emb], dim=-1)
        per_head: list[Tensor] = [head(x) for head in self.heads]
        return torch.stack(per_head, dim=0)

    @staticmethod
    def _variance(predictions: Tensor) -> Tensor:
        """Biased variance across K heads, summed over the ``z_dim`` axis.

        The user's spec calls for ``unbiased=False`` because Plan2Explore
        uses biased variance and K=5 is small enough that the unbiased
        correction adds noise from the correction without buying
        meaningful accuracy.
        """
        return torch.var(predictions, dim=0, unbiased=False).sum(dim=-1)


__all__ = ["EnsembleConfig", "LatentDisagreementEnsemble"]
