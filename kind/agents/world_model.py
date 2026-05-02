"""Phase 2b RSSM world model.

A custom minimal RSSM modeled on PlaNet's state factorization with
DreamerV1-style latent imagination. Continuous Gaussian stochastic latent.
Free bits is the only DreamerV3 borrowing — no symlog, twohot, KL balancing,
percentile normalization, or unimix. No reward predictor, no continuation
head, no return-to-go conditioning. The decoder is kept (mirror legibility
for dream rollouts, per Probe 1 synthesis §Q1).

The world model exposes the substrate conduits later probes need:
deterministic recurrent state ``h``, sampled posterior latent ``z``,
posterior/prior parameters for KL, encoder embedding for the mirror, and
decoded reconstruction for dream rollouts. Everything is held together in a
single frozen ``WorldModelStep`` dataclass; Phase 3c's ``views`` module is
what splits this into ``PolicyView`` (what Io's actor reads — concat of h
and z, nothing else) and ``TelemetryView`` (everything, for the mirror).
The split is enforced one phase up; the world model just exposes the full
conduit here.

CPU only at this phase. The MPS smoke (Phase 8) is the platform-correctness
gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Normal, kl_divergence


@dataclass(frozen=True)
class WorldModelConfig:
    """Static sizes for the world model.

    Defaults are Probe 1 starting points per implementation plan §6:
    ``h_dim=200``, ``z_dim=16``, ``free_bits_per_dim=1.0`` nat. Tests use
    smaller sizes for speed; the smoke test (Phase 8) uses these defaults
    on the target platform.
    """

    obs_channels: int = 1
    obs_size: int = 32
    h_dim: int = 200
    z_dim: int = 16
    embed_dim: int = 256
    num_actions: int = 5
    action_emb_dim: int = 16
    mlp_hidden: int = 200
    free_bits_per_dim: float = 1.0


@dataclass(frozen=True)
class WorldModelStep:
    """One forward step's full output, held together as one immutable record.

    ``kl_per_dim`` is the *unclipped* per-dimension KL between posterior and
    prior — the free-bits floor is applied inside ``WorldModel.loss()``, not
    here. ``q_params`` and ``p_params`` are ``(μ, log-σ)`` tuples; downstream
    code that wants a ``Normal`` must ``exp`` the second element.

    This dataclass is purely in-memory plumbing — it never leaves the
    process. Persistent telemetry uses the ``AgentStep`` Pydantic model in
    ``kind/observer/schemas.py`` (Phase 0).
    """

    h: Tensor
    z: Tensor
    q_params: tuple[Tensor, Tensor]
    p_params: tuple[Tensor, Tensor]
    kl_per_dim: Tensor
    recon: Tensor
    embed: Tensor


class _ConvEncoder(nn.Module):
    """3-layer stride-2 conv encoder, 32×32 grayscale → flat embedding.

    No BatchNorm (small batches and MPS make it fragile, per plan §2.5).
    ELU activations to match the prior/posterior MLPs.
    """

    def __init__(self, obs_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(obs_channels, 16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        # 32 → 16 → 8 → 4
        self._flat_dim = 64 * 4 * 4
        self.proj = nn.Linear(self._flat_dim, embed_dim)

    def forward(self, obs: Tensor) -> Tensor:
        x = F.elu(self.conv1(obs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = x.flatten(start_dim=1)
        return F.elu(self.proj(x))


class _ConvDecoder(nn.Module):
    """Mirror of the encoder: latent → 32×32 grayscale reconstruction.

    Final layer has no activation — output is raw pixels for MSE loss.
    """

    def __init__(self, latent_dim: int, obs_channels: int) -> None:
        super().__init__()
        self._flat_dim = 64 * 4 * 4
        self.proj = nn.Linear(latent_dim, self._flat_dim)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(
            16, obs_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, latent: Tensor) -> Tensor:
        x = F.elu(self.proj(latent))
        x = x.view(-1, 64, 4, 4)
        x = F.elu(self.deconv1(x))
        x = F.elu(self.deconv2(x))
        return cast(Tensor, self.deconv3(x))


class _GaussianHead(nn.Module):
    """MLP that outputs ``(μ, log-σ)`` for a diagonal Gaussian.

    Two ELU-activated hidden layers; final linear produces ``2 * output_dim``
    values that ``chunk`` into μ and log-σ. log-σ is *not* bounded — free
    bits is the only stability mechanism per plan §2.5; bounding log-σ would
    be a DreamerV3-style trick the synthesis explicitly rules out.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 2 * output_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = F.elu(self.fc1(x))
        h = F.elu(self.fc2(h))
        out = self.head(h)
        mu, log_sigma = torch.chunk(out, 2, dim=-1)
        return mu, log_sigma


class WorldModel(nn.Module):
    """The minimal RSSM. See module docstring for what's in and what's out."""

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self._free_bits_per_dim = config.free_bits_per_dim

        self.action_embedding = nn.Embedding(config.num_actions, config.action_emb_dim)
        self.gru_cell = nn.GRUCell(
            input_size=config.z_dim + config.action_emb_dim,
            hidden_size=config.h_dim,
        )
        self.encoder = _ConvEncoder(config.obs_channels, config.embed_dim)
        self.decoder = _ConvDecoder(
            latent_dim=config.h_dim + config.z_dim,
            obs_channels=config.obs_channels,
        )
        self.prior_head = _GaussianHead(
            input_dim=config.h_dim,
            hidden_dim=config.mlp_hidden,
            output_dim=config.z_dim,
        )
        self.posterior_head = _GaussianHead(
            input_dim=config.h_dim + config.embed_dim,
            hidden_dim=config.mlp_hidden,
            output_dim=config.z_dim,
        )

    def encode(self, obs: Tensor) -> Tensor:
        return cast(Tensor, self.encoder(obs))

    def prior(self, h: Tensor) -> tuple[Tensor, Tensor]:
        return cast("tuple[Tensor, Tensor]", self.prior_head(h))

    def posterior(self, h: Tensor, embed: Tensor) -> tuple[Tensor, Tensor]:
        return cast(
            "tuple[Tensor, Tensor]",
            self.posterior_head(torch.cat([h, embed], dim=-1)),
        )

    def recurrence(self, h: Tensor, z: Tensor, a: Tensor) -> Tensor:
        a_emb = self.action_embedding(a)
        gru_input = torch.cat([z, a_emb], dim=-1)
        return cast(Tensor, self.gru_cell(gru_input, h))

    def decode(self, h: Tensor, z: Tensor) -> Tensor:
        return cast(Tensor, self.decoder(torch.cat([h, z], dim=-1)))

    def step(
        self,
        obs: Tensor,
        h_prev: Tensor,
        z_prev: Tensor,
        a_prev: Tensor,
    ) -> WorldModelStep:
        """One forward step.

        Composition order per plan §2.5:
        ``recurrence → prior → encode → posterior → sample → decode → KL``.

        Shapes (B = batch dim):
        ``obs: (B, obs_channels, obs_size, obs_size)``,
        ``h_prev: (B, h_dim)``, ``z_prev: (B, z_dim)``,
        ``a_prev: (B,)`` (long; index into the action embedding).
        """
        h = self.recurrence(h_prev, z_prev, a_prev)
        p_params = self.prior(h)
        embed = self.encode(obs)
        q_params = self.posterior(h, embed)

        mu_q, log_sigma_q = q_params
        sigma_q = torch.exp(log_sigma_q)
        q_dist = Normal(mu_q, sigma_q)
        z = q_dist.rsample()

        recon = self.decode(h, z)

        mu_p, log_sigma_p = p_params
        sigma_p = torch.exp(log_sigma_p)
        p_dist = Normal(mu_p, sigma_p)
        kl_per_dim = kl_divergence(q_dist, p_dist)

        return WorldModelStep(
            h=h,
            z=z,
            q_params=q_params,
            p_params=p_params,
            kl_per_dim=kl_per_dim,
            recon=recon,
            embed=embed,
        )

    def loss(
        self, step: WorldModelStep, obs_target: Tensor
    ) -> dict[str, Tensor]:
        """ELBO with per-dimension free-bits floor on the KL term.

        Returns a dict with:

        * ``total`` — the trainable scalar (recon + free-bits-clipped KL).
        * ``recon`` — MSE summed over pixel dims, mean over batch.
        * ``kl`` — KL after per-dim free-bits clipping, summed over latent
          dims, mean over batch. This is what enters ``total``.
        * ``kl_aggregate_unclipped`` — raw posterior-prior KL summed over
          latent dims, mean over batch. **Telemetry only** — this is the
          quantity the synthesis §Q5 rules out of Io's reward stream
          (routing posterior-prior KL into the actor's signal would be a
          structural form of self-readout). The actor never sees this; the
          mirror does.
        """
        diff = step.recon - obs_target
        recon_loss = (diff**2).flatten(start_dim=1).sum(dim=1).mean()

        kl_aggregate_unclipped = step.kl_per_dim.sum(dim=-1).mean()

        kl_clipped = torch.clamp(step.kl_per_dim, min=self._free_bits_per_dim)
        kl_loss = kl_clipped.sum(dim=-1).mean()

        total = recon_loss + kl_loss
        return {
            "total": total,
            "recon": recon_loss,
            "kl": kl_loss,
            "kl_aggregate_unclipped": kl_aggregate_unclipped,
        }


__all__ = [
    "WorldModel",
    "WorldModelConfig",
    "WorldModelStep",
]
