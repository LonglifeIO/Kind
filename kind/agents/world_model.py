"""Phase 2b RSSM world model + Probe 1.5 self-prediction head.

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
what splits this into ``PolicyView`` (what Io's actor reads) and
``TelemetryView`` (everything, for the mirror).

**Probe 1.5 addition (synthesis §1.2 v2; implementation plan §2.2).** A
``SelfPredictionHead`` (small MLP ``h_dim → h_dim``) reads the deterministic
recurrent state ``h_t`` and emits ``ĥ_{t+1}``, exposed on
``WorldModelStep.self_prediction``. EMA-tracked sibling copies of the
encoder and the GRU cell (``target_encoder``, ``target_gru_cell``) live on
the world model with ``requires_grad=False``; ``compute_self_prediction_target``
produces the supervisory target ``bar{h}_{t+1}`` via the configured
``self_prediction_target_mode`` (``"online"``: EMA GRU on the same
recurrence inputs; ``"frozen"``: fixed random-orthogonal projection of the
online ``h_t``; ``"environmental"``: EMA encoder of the next observation,
projected to ``h_dim`` if ``embed_dim`` differs). ``WorldModel.loss``
returns ``self_prediction_loss`` in its dict when the runner passes
``target_h_next``; the term is *not* folded into ``total`` — the runner is
what sums ``wm_total + λ_self * self_prediction_loss`` (plan §2.4). The
single mutating side-effect specific to the head is
``_update_ema_target()``, which the runner calls after the world-model
optimizer step (plan §2.4).

CPU only at this phase. The MPS smoke (Phase 6) is the platform-correctness
gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Normal, kl_divergence


@dataclass(frozen=True)
class WorldModelConfig:
    """Static sizes for the world model.

    Probe 1 fields unchanged. Probe 1.5 adds four (synthesis §1.2 / §3 v2;
    plan §2.2): ``self_prediction_hidden`` (head MLP width),
    ``ema_decay`` (BYOL/SPR convention; Tian/Chen/Ganguli 2021),
    ``self_prediction_target_mode`` (``"online"`` for the real Probe 1.5
    path; ``"frozen"`` and ``"environmental"`` for the failure-mode
    controls Phase 5 wires — the substrate honors all three modes here so
    the controls are submodes on a single class), and
    ``self_prediction_loss_form`` (``"cosine"`` is the BYOL convention;
    MSE is the documented fallback per plan §6 row 3).
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
    self_prediction_hidden: int = 200
    ema_decay: float = 0.99
    self_prediction_target_mode: Literal["online", "frozen", "environmental"] = "online"
    self_prediction_loss_form: Literal["cosine", "mse"] = "cosine"
    # Probe 2 v2 lesion plumbing (plan §2.5; synthesis §2.4 element 4).
    # Only ``"disable_self_prediction"`` affects WorldModel behavior; the
    # other Probe 2 lesion kinds operate at views.split (zero_or_randomize_
    # scalar), at the runner's ensemble construction (ensemble_k1,
    # ensemble_constant), or as a checkpoint mutation (init_zero_scalar_
    # column). When ``"disable_self_prediction"``: ``WorldModel.step`` emits
    # zeros from the head, ``_update_ema_target`` is a no-op, and
    # ``WorldModel.loss``'s ``self_prediction_loss`` is identically zero so
    # the runner's ``wm_total + λ_self * self_prediction_loss`` backward
    # contributes nothing on the head/EMA axis.
    lesion_kind: Literal[None, "disable_self_prediction"] = None


@dataclass(frozen=True)
class WorldModelStep:
    """One forward step's full output, held together as one immutable record.

    ``kl_per_dim`` is the *unclipped* per-dimension KL between posterior and
    prior — the free-bits floor is applied inside ``WorldModel.loss()``, not
    here. ``q_params`` and ``p_params`` are ``(μ, log-σ)`` tuples; downstream
    code that wants a ``Normal`` must ``exp`` the second element.

    Probe 1.5 adds ``self_prediction``: the head's output ``ĥ_{t+1}`` of
    shape ``(B, h_dim)``, computed inside ``step`` from ``h_t`` only (the
    head's input does *not* include ``z_t`` per synthesis §3 default — keeps
    the head's input distinct from the ensemble's input and reduces
    entanglement). Phase 1 surfaces the value here; Phase 2 plumbs it
    through ``views.split`` into ``TelemetryView`` (full vector for the
    mirror) and the scalar derivative onto ``PolicyView`` (the Watts-
    heuristic exception per synthesis §1.3 v2 / §2(b) v2).

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
    self_prediction: Tensor


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


class SelfPredictionHead(nn.Module):
    """Probe 1.5 self-prediction head — small MLP ``h_dim → h_dim``.

    Predicts ``ĥ_{t+1}`` from the online deterministic recurrent state
    ``h_t``. Architecture per plan §6 row 5 default: two ELU-activated
    hidden layers at width ``hidden_dim`` (defaults to 200, matching the
    other MLPs in this module). The head sits parallel to the prior
    network as a sibling MLP off the shared GRU; its training-time loss
    (cosine or MSE against the EMA target's ``bar{h}_{t+1}``) flows
    gradient into the encoder, GRU, posterior, and prior — but not into
    the actor's parameters (synthesis §1.2 v2; the actor only reads the
    *scalar derivative* of the head's loss via PolicyView at Phase 2).
    """

    def __init__(self, h_dim: int, hidden_dim: int = 200) -> None:
        super().__init__()
        self.fc1 = nn.Linear(h_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, h_dim)

    def forward(self, h: Tensor) -> Tensor:
        x = F.elu(self.fc1(h))
        x = F.elu(self.fc2(x))
        return cast(Tensor, self.head(x))


class WorldModel(nn.Module):
    """The minimal RSSM, extended with the Probe 1.5 self-prediction head
    and EMA-tracked target sibling.

    Probe 1.5 additions (synthesis §1.2 v2; plan §2.2):

    * ``self_prediction_head``: a small MLP from ``h_dim`` to ``h_dim``.
    * ``target_encoder``, ``target_gru_cell``: EMA-tracked sibling copies
      of the online encoder and GRU cell, initialised from the online
      parameters at construction; their parameters carry
      ``requires_grad=False`` so backward never populates them and the
      runner's optimizer never updates them. ``_update_ema_target()`` is
      the only path through which they change.
    * ``_frozen_projection``: a fixed random-orthogonal ``(h_dim, h_dim)``
      ``nn.Parameter`` allocated only when
      ``self_prediction_target_mode == "frozen"``; ``None`` otherwise.
      Carries ``requires_grad=False``.
    * ``_environmental_projection``: a fixed (non-trainable)
      ``nn.Linear(embed_dim, h_dim)`` allocated only when
      ``self_prediction_target_mode == "environmental"`` *and*
      ``embed_dim != h_dim``; ``None`` otherwise.
    """

    target_encoder: _ConvEncoder
    target_gru_cell: nn.GRUCell
    _frozen_projection: nn.Parameter | None
    _environmental_projection: nn.Linear | None

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

        # Probe 1.5: self-prediction head + EMA-tracked target siblings.
        self.self_prediction_head = SelfPredictionHead(
            h_dim=config.h_dim,
            hidden_dim=config.self_prediction_hidden,
        )

        # The EMA target is a *separate* copy of encoder + GRU. Same
        # architecture, same init-time arguments; the per-parameter init
        # values are then overwritten with the online network's values via
        # ``_initialize_ema_target_from_online`` so the two networks start
        # identical (BYOL convention; Tian/Chen/Ganguli 2021's
        # collapse-prevention analysis assumes this initialisation). The
        # parameters then carry ``requires_grad=False`` — they do not
        # appear in the optimizer's parameter set, and backward leaves
        # their ``.grad`` ``None``. The only path that changes them is
        # ``_update_ema_target``.
        self.target_encoder = _ConvEncoder(config.obs_channels, config.embed_dim)
        self.target_gru_cell = nn.GRUCell(
            input_size=config.z_dim + config.action_emb_dim,
            hidden_size=config.h_dim,
        )
        self._initialize_ema_target_from_online()
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        for p in self.target_gru_cell.parameters():
            p.requires_grad_(False)

        # Frozen-projection only allocated when the constructor flag asks
        # for it. ``nn.init.orthogonal_`` produces a rank-preserving
        # semantic-alignment-broken projection (plan §2.2 / §8.1
        # discussion); ``requires_grad=False`` keeps backward from
        # populating ``.grad``.
        if config.self_prediction_target_mode == "frozen":
            proj = torch.empty(config.h_dim, config.h_dim)
            nn.init.orthogonal_(proj)
            self._frozen_projection = nn.Parameter(proj, requires_grad=False)
        else:
            self._frozen_projection = None

        # Environmental projection — only allocated when the constructor
        # asks for environmental mode *and* the encoder's output
        # dimensionality does not already match ``h_dim``. The layer is
        # frozen (not EMA-tracked, not trained); its weights are fixed at
        # construction time so the environmental control's reproducibility
        # is preserved across runs.
        if (
            config.self_prediction_target_mode == "environmental"
            and config.embed_dim != config.h_dim
        ):
            self._environmental_projection = nn.Linear(
                config.embed_dim, config.h_dim
            )
            for p in self._environmental_projection.parameters():
                p.requires_grad_(False)
        else:
            self._environmental_projection = None

    # ---- core RSSM components (Probe 1) -----------------------------------

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

    # ---- Probe 1.5 EMA target machinery -----------------------------------

    def _initialize_ema_target_from_online(self) -> None:
        """Copy the online encoder + GRU's parameter and buffer state into
        the EMA target siblings. Called by ``__init__`` (so the EMA target
        starts identical to the online network — BYOL convention) and by
        the runner's checkpoint-load path when loading a Probe 1
        checkpoint that has no EMA target weights on disk (plan §2.4).

        The copy preserves the ``requires_grad=False`` flag the constructor
        sets on the EMA target's parameters; ``data.copy_`` mutates only
        the underlying tensor storage and does not touch ``requires_grad``.
        """
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_gru_cell.load_state_dict(self.gru_cell.state_dict())

    def _update_ema_target(self) -> None:
        """Apply the EMA update to the target encoder + GRU.

        For each ``(target_param, online_param)`` pair on
        ``target_encoder`` and ``target_gru_cell``::

            target.data.mul_(ema_decay).add_(online.data, alpha=1 - ema_decay)

        Called by the runner after the world-model optimizer step (plan
        §2.4). The arithmetic is in-place on the EMA target's underlying
        storage so the gradient graph is not entered. The action embedding
        is *not* EMA-tracked (it is shared with the online network);
        plan §2.2's "EMA-tracked sibling copies of encoder + gru_cell"
        names exactly two modules.

        Probe 2 v2 ``disable_self_prediction`` lesion (plan §2.5): when
        the lesion is in effect, the EMA target stops tracking the online
        network — the substrate-side lesion targets the head's gradient
        flow, and freezing the target is what makes the head's auxiliary
        loss path do no work. The target's parameters retain their
        construction-time values (from
        ``_initialize_ema_target_from_online``).
        """
        if self.config.lesion_kind == "disable_self_prediction":
            return
        ema_decay = self.config.ema_decay
        with torch.no_grad():
            for online_p, target_p in zip(
                self.encoder.parameters(),
                self.target_encoder.parameters(),
                strict=True,
            ):
                target_p.data.mul_(ema_decay).add_(
                    online_p.data, alpha=1.0 - ema_decay
                )
            for online_p, target_p in zip(
                self.gru_cell.parameters(),
                self.target_gru_cell.parameters(),
                strict=True,
            ):
                target_p.data.mul_(ema_decay).add_(
                    online_p.data, alpha=1.0 - ema_decay
                )

    def compute_self_prediction_target(
        self,
        obs: Tensor,
        h_prev: Tensor,
        z_prev: Tensor,
        a_prev: Tensor,
        *,
        next_obs: Tensor | None = None,
    ) -> Tensor:
        """Produce ``bar{h}_{t+1}`` via the configured ``target_mode``.

        Returns a tensor of shape ``(B, h_dim)`` for any of the three
        modes. The returned tensor is fully detached: no gradient flows
        back into the EMA target parameters (which carry
        ``requires_grad=False`` regardless), the action embedding, the
        online recurrence, or the input tensors. The runner's
        ``self_prediction_loss`` computation calls ``.detach()`` on the
        result as a defensive belt-and-braces; the no-gradient discipline
        is *primarily* enforced here.

        This same routine handles per-step (B=1) and batched (B=batch)
        call sites; the batched call path is what the Phase 3 runner's
        ``_train_step`` will use, the per-step path is what the runner's
        ``_step_once`` calls to compute the scalar that flows into
        PolicyView (Phase 2 / Phase 3 plumbing). Plan §2.2.
        """
        mode = self.config.self_prediction_target_mode
        if mode == "online":
            with torch.no_grad():
                a_emb = self.action_embedding(a_prev)
                gru_input = torch.cat([z_prev, a_emb], dim=-1)
                target_h = self.target_gru_cell(gru_input, h_prev)
            return cast(Tensor, target_h)
        if mode == "frozen":
            assert self._frozen_projection is not None, (
                "frozen-target mode requires _frozen_projection to be allocated"
            )
            with torch.no_grad():
                a_emb = self.action_embedding(a_prev)
                gru_input = torch.cat([z_prev, a_emb], dim=-1)
                # The synthesis specifies projection of the online ``h_t``
                # (post-recurrence under online params) — rank-preserved,
                # semantic-alignment-broken (plan §2.2 / §8.1).
                h_t_online = self.gru_cell(gru_input, h_prev)
                target_h = F.linear(h_t_online, self._frozen_projection)
            return target_h
        if mode == "environmental":
            if next_obs is None:
                raise ValueError(
                    "compute_self_prediction_target requires next_obs when "
                    "self_prediction_target_mode='environmental'"
                )
            with torch.no_grad():
                embed = self.target_encoder(next_obs)
                if self._environmental_projection is not None:
                    embed = self._environmental_projection(embed)
            return cast(Tensor, embed)
        # mypy treats the Literal exhaustive; this guard is defensive.
        raise ValueError(
            f"unknown self_prediction_target_mode: {mode!r}"
        )  # pragma: no cover

    # ---- forward + loss ---------------------------------------------------

    def step(
        self,
        obs: Tensor,
        h_prev: Tensor,
        z_prev: Tensor,
        a_prev: Tensor,
        *,
        next_obs: Tensor | None = None,
    ) -> WorldModelStep:
        """One forward step.

        Composition order per plan §2.5:
        ``recurrence → prior → encode → posterior → sample → decode → KL``,
        with the Probe 1.5 self-prediction head fired off ``h`` after
        recurrence (its output is a pure passive emission on
        ``WorldModelStep.self_prediction``; nothing in the existing RSSM
        pipeline reads it).

        Shapes (B = batch dim):
        ``obs: (B, obs_channels, obs_size, obs_size)``,
        ``h_prev: (B, h_dim)``, ``z_prev: (B, z_dim)``,
        ``a_prev: (B,)`` (long; index into the action embedding).

        ``next_obs`` is consumed only by ``compute_self_prediction_target``
        in environmental mode (plan §2.2). It is accepted on this signature
        for forward compatibility with the Phase 3 runner integration —
        the runner calls ``world_model.step`` and
        ``world_model.compute_self_prediction_target`` separately, so
        ``step`` does not need ``next_obs`` itself. The argument is kept on
        the signature so the runner can pass the same kwarg-set to both
        without branching.
        """
        del next_obs  # consumed only by compute_self_prediction_target
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

        # Probe 2 v2 ``disable_self_prediction`` lesion (plan §2.5): replace
        # the head's output with a fixed zero tensor of shape ``(B, h_dim)``;
        # the head is structurally present (its parameters serialise
        # normally into checkpoints) but does no work in the forward, and
        # the auxiliary loss path will be identically zero. The runner
        # continues to populate ``self_prediction_t`` from this zero tensor
        # so AgentStep's writer-side discipline (``"0.2.0"`` requires
        # non-None) is satisfied with a sentinel that downstream readers
        # can identify as the lesion's signature.
        if self.config.lesion_kind == "disable_self_prediction":
            self_prediction = torch.zeros(
                h.shape[0], self.config.h_dim, device=h.device, dtype=h.dtype
            )
        else:
            self_prediction = self.self_prediction_head(h)

        return WorldModelStep(
            h=h,
            z=z,
            q_params=q_params,
            p_params=p_params,
            kl_per_dim=kl_per_dim,
            recon=recon,
            embed=embed,
            self_prediction=self_prediction,
        )

    def loss(
        self,
        step: WorldModelStep,
        obs_target: Tensor,
        *,
        target_h_next: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """ELBO with per-dimension free-bits floor on the KL term, plus the
        Probe 1.5 self-prediction auxiliary.

        Returns a dict with:

        * ``total`` — the trainable scalar (recon + free-bits-clipped KL).
          Note that ``self_prediction_loss`` is *not* folded in here; the
          runner is responsible for ``wm_total + λ_self * self_prediction_loss``
          (plan §2.4). Keeping the auxiliary out of ``total`` here means
          callers that don't care about the head (e.g. the existing Probe 1
          loss-shape unit tests) pass ``target_h_next=None`` and get a
          zero-valued ``self_prediction_loss`` they can ignore.
        * ``recon`` — MSE summed over pixel dims, mean over batch.
        * ``kl`` — KL after per-dim free-bits clipping, summed over latent
          dims, mean over batch. This is what enters ``total``.
        * ``kl_aggregate_unclipped`` — raw posterior-prior KL summed over
          latent dims, mean over batch. **Telemetry only** — this is the
          quantity the synthesis §Q5 rules out of Io's reward stream
          (routing posterior-prior KL into the actor's signal would be a
          structural form of self-readout). The actor never sees this; the
          mirror does.
        * ``self_prediction_loss`` — when ``target_h_next`` is provided,
          ``1 - cos_sim(step.self_prediction, target_h_next.detach())`` for
          ``self_prediction_loss_form="cosine"`` (BYOL convention) or
          ``F.mse_loss(step.self_prediction, target_h_next.detach())`` for
          ``"mse"``. When ``target_h_next is None``, returns a zero scalar
          on the same device / dtype as ``step.self_prediction`` —
          callers that don't pass it (Phase 1 internal tests; Phase 2
          unit tests for unrelated paths) get a no-op auxiliary.
        """
        diff = step.recon - obs_target
        recon_loss = (diff**2).flatten(start_dim=1).sum(dim=1).mean()

        kl_aggregate_unclipped = step.kl_per_dim.sum(dim=-1).mean()

        kl_clipped = torch.clamp(step.kl_per_dim, min=self._free_bits_per_dim)
        kl_loss = kl_clipped.sum(dim=-1).mean()

        total = recon_loss + kl_loss

        if target_h_next is None:
            self_prediction_loss = torch.zeros(
                (), device=step.self_prediction.device, dtype=step.self_prediction.dtype
            )
        elif self.config.lesion_kind == "disable_self_prediction":
            # Probe 2 v2 lesion (plan §2.5): the auxiliary loss is
            # identically zero so the runner's
            # ``wm_total + λ_self * self_prediction_loss`` backward
            # contributes nothing on the head/EMA axis. Skipping the
            # computation also avoids a degenerate cosine_similarity over
            # the zero ``step.self_prediction`` vector (which PyTorch
            # treats as 0 via the eps guard but is semantically
            # ill-defined).
            self_prediction_loss = torch.zeros(
                (), device=step.self_prediction.device, dtype=step.self_prediction.dtype
            )
        elif self.config.self_prediction_loss_form == "cosine":
            self_prediction_loss = (
                1.0
                - F.cosine_similarity(
                    step.self_prediction, target_h_next.detach(), dim=-1
                ).mean()
            )
        else:  # "mse"
            self_prediction_loss = F.mse_loss(
                step.self_prediction, target_h_next.detach()
            )

        return {
            "total": total,
            "recon": recon_loss,
            "kl": kl_loss,
            "kl_aggregate_unclipped": kl_aggregate_unclipped,
            "self_prediction_loss": self_prediction_loss,
        }


__all__ = [
    "SelfPredictionHead",
    "WorldModel",
    "WorldModelConfig",
    "WorldModelStep",
]
