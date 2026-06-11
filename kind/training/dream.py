"""Phase 2 dream-rollout module — the four-axis-differentiated dream regime.

This module is where "dream ≠ waking planning" stops being four properties in
prose (synthesis §3) and becomes four simultaneous code paths the Phase 3
visibility smoke can attribute. A dream rollout differs from the preserved
waking-planning rollout (the renamed Probe 1.5 ``_emit_dream``) on all four
axes *jointly*:

  1. **Goal-coupling absent (axis 1).** Default ``action_policy="uniform_random"``
     draws actions uniformly over the action vocabulary; the actor is never
     consulted for action choice (``actor`` is ``None`` in the default config).
     ``"temperature_modified_actor"`` is the only non-default alternative the
     smoke can compare against — it *softens* the actor's policy by an
     independent ``actor_action_temperature`` but still does not couple action
     back to the environment.
  2. **Ensemble disagreement recorded-but-not-used (axis 2).** The K=5 latent
     disagreement is evaluated at every step and written into
     ``sequence_ensemble_disagreement_variance`` — but never read to choose an
     action, set a temperature, or terminate. It is a quantity that
     constitutively *exists but does nothing* during dream, whereas in waking
     it drives the actor's only intrinsic signal.
  3. **Distinct temperature/sampling regime (axis 3).** The prior's sigma is
     multiplied by a per-step temperature following ``prior_temperature_schedule``
     (default: 1.5 for the head 60%, ramping to 2.5 over the tail 40% — the
     associative-drift tail). This is kept *separate* from the axis-1 actor
     temperature so Phase 3 can attribute distributional shifts to one axis at
     a time.
  4. **Initial conditions from replay / perturbed prior (axis 4).** The seed is
     produced by Phase 1's ``select_seed`` (replay / perturbed_prior / hybrid),
     not the current waking state. Periodic re-seeding
     (``re_seed_every_n_steps``) produces the chimera structure: the rollout's
     effective trajectory is a concatenation of sub-trajectories, each from a
     fresh seed.

**Prior-sampling / temperature / RNG contract.** Dreaming has no observation
input, so it samples the *prior* ``p(z|h)`` — not the posterior, and not via
``WorldModel.step`` (which samples through the global PyTorch RNG and takes no
generator). The prior is sampled generator-driven and temperature-scaled,
reusing the Phase 1 pattern: noise is drawn on a CPU ``torch.Generator`` and
moved to the model's device, so reproducibility is device-independent::

    z_t = prior_mu + temperature_t * prior_sigma * randn(generator=dream_gen)

The rollout's ``dream_gen`` is derived *deterministically and solely* from
``DreamSeed.rng_seed`` (a fixed offset; see ``_DREAM_ROLLOUT_GEN_OFFSET``), and
``DreamRollout.rng_seed`` is set to ``DreamSeed.rng_seed``. Because Phase 1
guarantees the seed ``(h_init, z_init)`` is a pure function of
``(rng_seed, world_model weights, buffer obs window)``, and the trajectory is a
pure function of ``(rng_seed, world_model weights, seed)``, the single recorded
int reproduces seed-selection-through-trajectory given the pinned checkpoint
(``checkpoint_hash``) and buffer. Mid-rollout re-seeds draw from ``dream_gen``,
so they stay inside that single-int reproduction.

**No gradient flow — type-level and runtime (synthesis §1).**
``gradient_policy`` is a *closed* ``Literal["none"]``: widening it to
world-model or full Dreamer updates requires a separate synthesis. That is the
type-level half. The runtime half is that every rollout runs under
``torch.no_grad()``, so no graph builds from dream into any parameter.

**Out of scope at Probe 3** (plan §2.3 / §9): no lucid control; no
mirror-driven action policy or termination (the mirror is logs-only); no
self-prediction head during dream — ``sequence_self_prediction`` stays ``None``
(the head runs only during waking per Probe 1.5 synthesis §1.5); no gradient
flow (the closed Literal enforces it); no hidden-state-write affordance.

This module imports observer schemas (read-only writer target), Phase 1's
``select_seed``, and the agent modules it rolls forward. It does *not* import
``kind.training.state_machine`` (Phase 4) or ``kind.mirror.*`` (Phase 5).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Final, Literal, Protocol, cast

import torch
from torch import Tensor

from kind.agents.actor import Actor
from kind.agents.ensemble import LatentDisagreementEnsemble
from kind.agents.views import PolicyView
from kind.agents.world_model import WorldModel
from kind.observer.dream_session import (
    DREAM_SESSION_META_SCHEMA_VERSION,
    DreamSessionMeta,
    DreamSessionSink,
)
from kind.observer.schemas import (
    PROBE_3_5_PHASE3_DREAM_SCHEMA_VERSION,
    PROBE_3_TELEMETRY_SCHEMA_VERSION,
    DreamRollout,
)
from kind.training.dream_seed import SeedSelectionConfig, select_seed
from kind.training.replay import SequenceReplayBuffer

__all__ = [
    "TempSchedule",
    "DreamRolloutConfig",
    "DreamRolloutSink",
    "emit_dream_rollout",
    "run_dream_session",
    "compute_checkpoint_hash",
    "compute_perturbed_prior_anchor",
]

# Fixed offset deriving the rollout's prior-sampling generator from the seed's
# rng_seed. A constant (not external entropy) so that one recorded rng_seed
# reproduces the whole dream — seed selection through prior trajectory. See the
# module docstring's RNG contract.
_DREAM_ROLLOUT_GEN_OFFSET: int = 0x9E3779B1
_GEN_SEED_MODULUS: int = 2**63 - 1

# The Phase 0 sink model pins schema_version to a Literal; carry the same
# literal here (asserted equal to the exported constant) for the writer.
_SESSION_META_VERSION: Final[Literal["0.1.0"]] = "0.1.0"
assert _SESSION_META_VERSION == DREAM_SESSION_META_SCHEMA_VERSION


@dataclass(frozen=True)
class TempSchedule:
    """A per-step temperature trajectory on the prior's sigma (plan §2.3).

    ``head_value`` holds for the head of the rollout; the temperature ramps
    linearly to ``tail_value`` over the tail, starting at
    ``ramp_start_fraction`` of the horizon. The tail ramp is the
    associative-drift parametric regime (synthesis §2) — a parameter
    trajectory within a single rollout, not a separate mode.
    """

    head_value: float
    tail_value: float
    ramp_start_fraction: float


@dataclass(frozen=True)
class DreamRolloutConfig:
    """Per-rollout configuration (plan §2.3).

    The default config is the real dreaming path: the four axes are realized
    as the conjunction of ``action_policy="uniform_random"`` (axis 1) +
    ``record_ensemble_disagreement=True`` (axis 2) + a non-flat
    ``prior_temperature_schedule`` (axis 3) + a ``select_seed`` seed (axis 4).
    The control switches (``temperature_mode="identity"``,
    ``seed_strategy_for_control="prior_only"``) are the *mechanism* Phase 3
    configures to build its degenerate controls; the KS-D comparison itself is
    Phase 3's, not this module's.
    """

    horizon: int = 30
    # Off-switch for the Phase 3 control: "identity" forces multiplier 1.0
    # every step (the degenerate sampling regime), "scheduled" applies the
    # prior_temperature_schedule.
    temperature_mode: Literal["scheduled", "identity"] = "scheduled"
    prior_temperature_schedule: TempSchedule = TempSchedule(
        head_value=1.5, tail_value=2.5, ramp_start_fraction=0.6
    )
    re_seed_every_n_steps: int = 10
    re_seed_mode: Literal["resample_seed", "perturb_in_place"] = "resample_seed"
    action_policy: Literal["uniform_random", "temperature_modified_actor"] = (
        "uniform_random"
    )
    # Used only when action_policy == "temperature_modified_actor"; *independent*
    # of prior_temperature_schedule so axis 1 and axis 3 stay separately
    # attributable (plan §2.3 axis-1/axis-3 flag).
    actor_action_temperature: float = 1.5
    record_ensemble_disagreement: bool = True
    # Off-switch for the Phase 3 pure-prior control: "prior_only" ignores
    # select_seed and samples (h_init, z_init) from the prior at a zero h.
    seed_strategy_for_control: Literal["normal", "prior_only"] = "normal"
    # Probe 3.5 §7 dream passive-decode (pre-registered resolved sub-decision
    # #1; built at Phase 3). When True, each dream step also records
    # ``decode_energy(h_next, z_next)`` — observer-side telemetry only, under
    # the rollout's existing ``no_grad``, alongside ``sequence_decoded_obs``.
    # The preference term has NO code path into dreaming (F5 intact; the
    # guards in tests/test_pragmatic_guards.py and the byte-identity test in
    # tests/test_dream_energy_monitor.py are the proof): the mirror gets to
    # watch whether offline processing touches the energy belief; Io's dream
    # gains nothing to optimize. Default off → legacy emission byte-identical.
    record_decoded_energy: bool = False
    # Closed at first build; widening requires a separate synthesis (the
    # type-level half of the synthesis §1 no-gradient-flow commitment).
    gradient_policy: Literal["none"] = "none"


class DreamRolloutSink(Protocol):
    """Structural type for the DreamRollout telemetry sink (e.g. ParquetSink)."""

    def write(self, record: DreamRollout) -> None: ...


def _temperature_at_step(
    t: int, horizon: int, schedule: TempSchedule, mode: Literal["scheduled", "identity"]
) -> float:
    """The prior-sigma multiplier at step ``t`` (axis 3)."""
    if mode == "identity":
        return 1.0
    if horizon <= 1:
        return schedule.tail_value
    ramp_start = int(schedule.ramp_start_fraction * horizon)
    if t < ramp_start:
        return schedule.head_value
    span = (horizon - 1) - ramp_start
    if span <= 0:
        return schedule.tail_value
    frac = (t - ramp_start) / span
    return schedule.head_value + frac * (schedule.tail_value - schedule.head_value)


def _diag_gaussian_kl(
    mu1: Tensor, log_sigma1: Tensor, mu2: Tensor, log_sigma2: Tensor
) -> float:
    """Analytic KL(N(mu1, sigma1) || N(mu2, sigma2)), summed over the latent
    dim and mean over the (singleton) batch — one scalar."""
    var1 = torch.exp(2.0 * log_sigma1)
    var2 = torch.exp(2.0 * log_sigma2)
    kl = log_sigma2 - log_sigma1 + (var1 + (mu1 - mu2) ** 2) / (2.0 * var2) - 0.5
    return float(kl.sum(dim=-1).mean().item())


def _diag_gaussian_entropy(log_sigma: Tensor, z_dim: int) -> float:
    """Differential entropy of a diagonal Gaussian, summed over the latent dim.

    H = sum(log_sigma) + 0.5 * z_dim * log(2*pi*e). Computed analytically to
    keep the path free of torch.distributions' untyped calls.
    """
    const = 0.5 * z_dim * math.log(2.0 * math.pi * math.e)
    return float((log_sigma.sum(dim=-1) + const).mean().item())


def compute_checkpoint_hash(
    world_model: WorldModel,
    actor: Actor | None,
    ensemble: LatentDisagreementEnsemble,
) -> str:
    """SHA256 over the (world_model [+ actor] + ensemble) weights.

    Deterministic in sorted-key order so faithfulness can verify a dream
    rollout's run identity against the source checkpoint (plan §3.1).
    """
    hasher = hashlib.sha256()
    modules: list[tuple[str, torch.nn.Module]] = [("world_model", world_model)]
    if actor is not None:
        modules.append(("actor", actor))
    modules.append(("ensemble", ensemble))
    for name, module in modules:
        state = module.state_dict()
        for key in sorted(state.keys()):
            tensor = cast(Tensor, state[key])
            hasher.update(name.encode("utf-8"))
            hasher.update(key.encode("utf-8"))
            hasher.update(tensor.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def compute_perturbed_prior_anchor(
    world_model: WorldModel, device: torch.device
) -> Tensor:
    """The anchor for ``select_seed(mode="perturbed_prior")`` (Phase 2 owns this).

    Phase 1 deferred the anchor choice to Phase 2. The choice: the substrate's
    *unconditioned one-step expectation* — one recurrence step from a zero
    ``(h=0, z=0, a=0)`` start, then the prior mean at that ``h``. It is
    deterministic (no RNG), content-free, and not the current waking state, so
    axis 4 ("initial conditions not from current state") is preserved. Returns
    a ``(1, h_dim + z_dim)`` tensor — the h-concat-z shape ``select_seed``
    expects.
    """
    cfg = world_model.config
    dtype = next(world_model.parameters()).dtype
    with torch.no_grad():
        h0 = torch.zeros(1, cfg.h_dim, device=device, dtype=dtype)
        z0 = torch.zeros(1, cfg.z_dim, device=device, dtype=dtype)
        a0 = torch.zeros(1, dtype=torch.long, device=device)
        h_anchor = world_model.recurrence(h0, z0, a0)
        mu_p, _ = world_model.prior(h_anchor)
    return torch.cat([h_anchor, mu_p], dim=-1)


def _perturb_in_place(
    h: Tensor,
    z: Tensor,
    sigma: float,
    gen: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Add a sigma-scaled Gaussian to (h, z) for the perturb_in_place re-seed."""
    noise_h = (torch.randn(h.shape, generator=gen) * sigma).to(device=device, dtype=dtype)
    noise_z = (torch.randn(z.shape, generator=gen) * sigma).to(device=device, dtype=dtype)
    return h + noise_h, z + noise_z


def _sample_action(
    *,
    config: DreamRolloutConfig,
    actor: Actor | None,
    h: Tensor,
    z: Tensor,
    num_actions: int,
    uniform_logprob: float,
    dream_gen: torch.Generator,
    device: torch.device,
) -> tuple[Tensor, float]:
    """Axis 1: choose an action without goal-coupling.

    ``uniform_random`` draws uniformly over the vocabulary and never touches
    the actor. ``temperature_modified_actor`` softens the actor's softmax by
    ``actor_action_temperature`` and samples generator-driven. Returns the
    action (shape ``(1,)`` long, on device) and its log-probability.
    """
    if config.action_policy == "uniform_random":
        a_cpu = torch.randint(0, num_actions, (1,), generator=dream_gen)
        return a_cpu.to(device), uniform_logprob

    # temperature_modified_actor: soften the policy, sample generator-driven on
    # CPU (the Phase 1 device-independence pattern). The actor's choice never
    # couples back to the environment — this is still axis-1 goal-coupling
    # *softened*, not present.
    assert actor is not None, (
        "action_policy='temperature_modified_actor' requires a non-None actor"
    )
    zero_scalar = torch.zeros((), device=device)
    view = PolicyView(h=h, z=z, self_prediction_error=zero_scalar)
    logits = actor.forward(view).logits
    probs = torch.softmax(logits / config.actor_action_temperature, dim=-1).cpu()
    a_cpu = torch.multinomial(probs, num_samples=1, generator=dream_gen).squeeze(-1)
    logprob = float(torch.log(probs.gather(-1, a_cpu.unsqueeze(-1))).item())
    return a_cpu.to(device), logprob


def emit_dream_rollout(
    *,
    world_model: WorldModel,
    actor: Actor | None,
    ensemble: LatentDisagreementEnsemble,
    replay_buffer: SequenceReplayBuffer,
    seed_selection_config: SeedSelectionConfig,
    config: DreamRolloutConfig,
    dream_session_id: str,
    env_step_at_emit: int,
    run_id: str,
    checkpoint_id: str | None,
    checkpoint_hash: str,
    rng: torch.Generator,
    device: torch.device,
    perturbed_prior_anchor: Tensor | None = None,
) -> DreamRollout:
    """Run one four-axis-differentiated dream rollout and return its record.

    Standalone-runnable for Phase 3: given a world model, replay buffer,
    ensemble, config, and rng, it produces a ``DreamRollout`` with no
    dependency on the Phase 4 state machine. The whole rollout runs under
    ``torch.no_grad()`` (the runtime half of ``gradient_policy="none"``).

    ``rng`` is the session generator; ``select_seed`` draws this rollout's
    ``rng_seed`` from it. The prior-sampling generator is then derived from
    that ``rng_seed`` (see the module RNG contract), so ``DreamRollout.rng_seed``
    is the single statistic that reproduces seed + trajectory.
    """
    wm_cfg = world_model.config
    h_dim = wm_cfg.h_dim
    z_dim = wm_cfg.z_dim
    num_actions = wm_cfg.num_actions
    dtype = next(world_model.parameters()).dtype
    uniform_logprob = -math.log(num_actions)
    horizon = config.horizon

    with torch.no_grad():
        # ---- seed (axis 4), or the pure-prior control override -------------
        control_tag: str | None = None
        if config.seed_strategy_for_control == "prior_only":
            # Phase 3 pure-prior control: ignore the replay seed and sample
            # (h_init, z_init) from the prior at a zero h. Records as
            # perturbed_prior with zero magnitude (it *is* a prior-derived seed
            # with no perturbation) plus a "prior_only_control" sub_mode_tag, so
            # the record stays v0.3.0-valid and self-describing.
            rng_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
            dream_gen = torch.Generator()
            dream_gen.manual_seed((rng_seed + _DREAM_ROLLOUT_GEN_OFFSET) % _GEN_SEED_MODULUS)
            h = torch.zeros(1, h_dim, device=device, dtype=dtype)
            mu_p0, log_sigma_p0 = world_model.prior(h)
            noise0 = torch.randn(mu_p0.shape, generator=dream_gen).to(
                device=device, dtype=dtype
            )
            z = mu_p0 + torch.exp(log_sigma_p0) * noise0
            seed_kind: Literal["replay", "perturbed_prior", "hybrid"] = "perturbed_prior"
            seed_replay_segment_id: int | None = None
            seed_replay_step_offset: int | None = None
            seed_perturbation_magnitude: float | None = 0.0
            control_tag = "prior_only_control"
        else:
            seed = select_seed(
                replay_buffer,
                world_model,
                seed_selection_config,
                rng,
                perturbed_prior_anchor=perturbed_prior_anchor,
            )
            rng_seed = seed.rng_seed
            dream_gen = torch.Generator()
            dream_gen.manual_seed((rng_seed + _DREAM_ROLLOUT_GEN_OFFSET) % _GEN_SEED_MODULUS)
            h = seed.h_init
            z = seed.z_init
            seed_kind = seed.mode
            seed_replay_segment_id = seed.replay_segment_id
            seed_replay_step_offset = seed.replay_step_offset
            seed_perturbation_magnitude = seed.perturbation_magnitude

        seed_h0 = h.squeeze(0).detach().cpu().tolist()
        seed_z0 = z.squeeze(0).detach().cpu().tolist()

        sequence_h: list[list[float]] = []
        sequence_z_prior: list[list[float]] = []
        sequence_action: list[int] = []
        sequence_action_logprob: list[float] = []
        sequence_prior_entropy: list[float] = []
        sequence_decoded_obs: list[bytes] = []
        # §7 passive-decode monitor: populated only when opted in; stays None
        # on the record otherwise (legacy byte-identical).
        sequence_decoded_energy: list[float] | None = (
            [] if config.record_decoded_energy else None
        )
        disagreement_seq: list[float] = []
        temperature_schedule: list[float] = []
        re_seed_step_indices: list[int] = []

        cumulative_prior_entropy = 0.0
        kls: list[float] = []
        prev_mu: Tensor | None = None
        prev_log_sigma: Tensor | None = None
        max_norm_change = 0.0
        prev_z = z

        re_seed_n = config.re_seed_every_n_steps

        for t in range(horizon):
            # ---- periodic re-seed (axis 4 chimera) -------------------------
            # The jump registers naturally in max_step_latent_norm_change: the
            # next z_prior is sampled from a freshly-seeded h, far from the
            # prior step's z. prev_z is intentionally *not* reset.
            if re_seed_n > 0 and t > 0 and t % re_seed_n == 0:
                re_seed_step_indices.append(t)
                if config.seed_strategy_for_control == "prior_only":
                    mu_pr, log_sigma_pr = world_model.prior(h)
                    noise_pr = torch.randn(mu_pr.shape, generator=dream_gen).to(
                        device=device, dtype=dtype
                    )
                    z = mu_pr + torch.exp(log_sigma_pr) * noise_pr
                elif config.re_seed_mode == "resample_seed":
                    # Re-seed draws from dream_gen so it stays inside the single
                    # rng_seed reproduction (RNG contract).
                    re_seed = select_seed(
                        replay_buffer,
                        world_model,
                        seed_selection_config,
                        dream_gen,
                        perturbed_prior_anchor=perturbed_prior_anchor,
                    )
                    h = re_seed.h_init
                    z = re_seed.z_init
                else:  # perturb_in_place
                    h, z = _perturb_in_place(
                        h, z, seed_selection_config.perturbation_sigma, dream_gen, device, dtype
                    )

            # ---- action (axis 1): no goal-coupling -------------------------
            a, logprob = _sample_action(
                config=config,
                actor=actor,
                h=h,
                z=z,
                num_actions=num_actions,
                uniform_logprob=uniform_logprob,
                dream_gen=dream_gen,
                device=device,
            )
            sequence_action.append(int(a.item()))
            sequence_action_logprob.append(logprob)

            # ---- ensemble disagreement (axis 2): recorded, never used ------
            if config.record_ensemble_disagreement:
                disagreement_seq.append(float(ensemble.disagreement(h, z, a).item()))
            else:
                disagreement_seq.append(0.0)

            # ---- temperature (axis 3) --------------------------------------
            temp_t = _temperature_at_step(
                t, horizon, config.prior_temperature_schedule, config.temperature_mode
            )
            temperature_schedule.append(temp_t)

            # ---- recurrence + temperature-scaled prior sampling ------------
            # Dreaming has no obs, so we sample the *prior* p(z|h), generator-
            # driven on CPU then moved to device (Phase 1 device-independence
            # pattern). We cannot use WorldModel.step (posterior + global RNG).
            h_next = world_model.recurrence(h, z, a)
            mu_p, log_sigma_p = world_model.prior(h_next)
            sigma_p = torch.exp(log_sigma_p)
            noise = torch.randn(mu_p.shape, generator=dream_gen).to(
                device=mu_p.device, dtype=mu_p.dtype
            )
            z_next = mu_p + temp_t * sigma_p * noise

            decoded = world_model.decode(h_next, z_next)
            decoded_uint8 = (
                decoded.squeeze(0).squeeze(0).clamp(0.0, 1.0).mul(255.0).to(torch.uint8).cpu().numpy()
            )
            sequence_decoded_obs.append(decoded_uint8.tobytes())

            # §7 dream passive-decode (Probe 3.5, built at Phase 3): record
            # the energy belief alongside the decoded obs. A passive read of
            # ``decode_energy`` under the rollout's no_grad — observer-side
            # only, never a dream driver; the preference term is not computed
            # here or anywhere on this path (F5).
            if sequence_decoded_energy is not None:
                sequence_decoded_energy.append(
                    float(
                        world_model.decode_energy(h_next, z_next)
                        .reshape(-1)[0]
                        .item()
                    )
                )

            # Base-prior entropy (the model's prior; the temperature regime is
            # recorded separately in temperature_schedule, so this field stays
            # comparable to the waking distribution).
            step_entropy = _diag_gaussian_entropy(log_sigma_p, z_dim)
            sequence_prior_entropy.append(step_entropy)
            cumulative_prior_entropy += step_entropy

            if prev_mu is not None and prev_log_sigma is not None:
                kls.append(
                    _diag_gaussian_kl(prev_mu, prev_log_sigma, mu_p, log_sigma_p)
                )
            prev_mu = mu_p
            prev_log_sigma = log_sigma_p

            norm_change = float((z_next - prev_z).norm().item())
            if norm_change > max_norm_change:
                max_norm_change = norm_change
            prev_z = z_next

            sequence_h.append(h_next.squeeze(0).detach().cpu().tolist())
            sequence_z_prior.append(z_next.squeeze(0).detach().cpu().tolist())

            h = h_next
            z = z_next

    mean_kl = sum(kls) / len(kls) if kls else 0.0

    tags: list[str] = []
    if config.temperature_mode == "scheduled":
        tags.append("associative_drift_tail")
    if re_seed_step_indices:
        tags.append("chimera")
    if control_tag is not None:
        tags.append(control_tag)

    sampling_parameters: dict[str, float | int | str | bool] = {
        "replay_warmup_length": int(seed_selection_config.replay_warmup_length),
        "horizon": int(horizon),
        "re_seed_every_n_steps": int(config.re_seed_every_n_steps),
        "actor_action_temperature": float(config.actor_action_temperature),
        "action_policy": config.action_policy,
        "temperature_mode": config.temperature_mode,
        "re_seed_mode": config.re_seed_mode,
        "seed_strategy_for_control": config.seed_strategy_for_control,
    }

    return DreamRollout(
        # §7 monitor on → the Phase-3 dream record version (which requires the
        # monitor field non-None); off → the Probe-3 version, byte-identical
        # to pre-monitor emission.
        schema_version=(
            PROBE_3_5_PHASE3_DREAM_SCHEMA_VERSION
            if config.record_decoded_energy
            else PROBE_3_TELEMETRY_SCHEMA_VERSION
        ),
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        seed_step=env_step_at_emit,
        seed_h0=seed_h0,
        seed_z0=seed_z0,
        sequence_h=sequence_h,
        sequence_z_prior=sequence_z_prior,
        sequence_action=sequence_action,
        sequence_action_logprob=sequence_action_logprob,
        sequence_prior_entropy=sequence_prior_entropy,
        sequence_decoded_obs=sequence_decoded_obs,
        cumulative_prior_entropy=cumulative_prior_entropy,
        mean_step_kl_successive_priors=mean_kl,
        max_step_latent_norm_change=max_norm_change,
        # Synthesis §1.5: the self-prediction head runs only during waking; the
        # reserved slot stays None during dream.
        sequence_self_prediction=None,
        dream_session_id=dream_session_id,
        seed_kind=seed_kind,
        seed_replay_segment_id=seed_replay_segment_id,
        seed_replay_step_offset=seed_replay_step_offset,
        seed_perturbation_magnitude=seed_perturbation_magnitude,
        temperature_schedule=temperature_schedule,
        sub_mode_tags=tags or None,
        sampling_parameters=sampling_parameters,
        gradient_policy=config.gradient_policy,
        rng_seed=rng_seed,
        termination_reason="horizon_complete",
        re_seed_step_indices=re_seed_step_indices or None,
        sequence_ensemble_disagreement_variance=disagreement_seq,
        checkpoint_hash=checkpoint_hash,
        sequence_decoded_energy=sequence_decoded_energy,
    )


def run_dream_session(
    *,
    world_model: WorldModel,
    actor: Actor | None,
    ensemble: LatentDisagreementEnsemble,
    replay_buffer: SequenceReplayBuffer,
    seed_selection_config: SeedSelectionConfig,
    rollout_config: DreamRolloutConfig,
    num_rollouts: int,
    dream_session_id: str,
    run_id: str,
    checkpoint_id: str | None,
    started_at_env_step: int,
    started_at_wallclock_ms: int,
    rng: torch.Generator,
    device: torch.device,
    dream_rollout_sink: DreamRolloutSink,
    dream_session_sink: DreamSessionSink,
    envelope_config_snapshot: dict[str, Any] | None = None,
    checkpoint_hash: str | None = None,
    ended_at_env_step: int | None = None,
    ended_at_wallclock_ms: int | None = None,
    end_trigger: str | None = None,
) -> DreamSessionMeta:
    """Run a dream session: loop ``emit_dream_rollout`` and double-write the meta.

    The state machine (Phase 4) owns *when* a session starts and the real
    ``end_trigger``; Phase 2 owns the session machinery so Phase 3 can run
    sessions without Phase 4. The ``DreamSessionMeta`` double-write is the
    canonical pattern (plan §2.1): a start record (``ended_*`` None) the mirror
    can find in-flight, then an end record the faithfulness verifier resolves
    against. Returns the end record.
    """
    if checkpoint_hash is None:
        checkpoint_hash = compute_checkpoint_hash(world_model, actor, ensemble)

    # The anchor is deterministic and ignored by select_seed except in
    # perturbed_prior mode, so computing it unconditionally is cheap and keeps
    # the call site uniform.
    anchor = compute_perturbed_prior_anchor(world_model, device)
    seed_config_snapshot = asdict(seed_selection_config)

    start_record = DreamSessionMeta(
        schema_version=_SESSION_META_VERSION,
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        dream_session_id=dream_session_id,
        started_at_env_step=started_at_env_step,
        started_at_wallclock_ms=started_at_wallclock_ms,
        ended_at_env_step=None,
        ended_at_wallclock_ms=None,
        end_trigger=None,
        rollout_count=0,
        envelope_config_snapshot=envelope_config_snapshot or {},
        seed_selection_config_snapshot=seed_config_snapshot,
    )
    dream_session_sink.write(start_record)

    for i in range(num_rollouts):
        record = emit_dream_rollout(
            world_model=world_model,
            actor=actor,
            ensemble=ensemble,
            replay_buffer=replay_buffer,
            seed_selection_config=seed_selection_config,
            config=rollout_config,
            dream_session_id=dream_session_id,
            env_step_at_emit=started_at_env_step + i,
            run_id=run_id,
            checkpoint_id=checkpoint_id,
            checkpoint_hash=checkpoint_hash,
            rng=rng,
            device=device,
            perturbed_prior_anchor=anchor,
        )
        dream_rollout_sink.write(record)

    end_record = DreamSessionMeta(
        schema_version=_SESSION_META_VERSION,
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        dream_session_id=dream_session_id,
        started_at_env_step=started_at_env_step,
        started_at_wallclock_ms=started_at_wallclock_ms,
        ended_at_env_step=(
            ended_at_env_step
            if ended_at_env_step is not None
            else started_at_env_step + num_rollouts
        ),
        ended_at_wallclock_ms=(
            ended_at_wallclock_ms
            if ended_at_wallclock_ms is not None
            else started_at_wallclock_ms
        ),
        end_trigger=end_trigger,
        rollout_count=num_rollouts,
        envelope_config_snapshot=envelope_config_snapshot or {},
        seed_selection_config_snapshot=seed_config_snapshot,
    )
    dream_session_sink.write(end_record)
    return end_record
