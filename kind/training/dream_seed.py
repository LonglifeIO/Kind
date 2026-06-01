"""Phase 1 replay seed selection for dream rollouts.

`DreamSeed` is the typed initial-conditions object the Phase 2 dream-rollout
module consumes at each dream rollout's start. Phase 1's job is to produce it
in the three modes the Probe 3 plan §2.2 (amended 2026-05-27) names: replay,
perturbed_prior, and hybrid.

**Option-1 re-encode semantics.** Under the Phase 0.5 decision
(`docs/decisions/phase0_5_replay_seed_source_2026-05-27.md`), replay-mode
seeds are produced by re-encoding a short obs/action window read from
`SequenceReplayBuffer`. The buffer's storage shape is unchanged; no `(h, z)`
is stored alongside transitions (the rejected Option 2 stays rejected). The
seed's `(h_init, z_init)` is the endpoint of running the world model forward
over the warmup window from a zero `(h, z, a)` start. Provenance is the
window's start `env_step` plus the recorded `rng_seed`; the source
`checkpoint_hash` lives on the emitted `DreamRollout` (Phase 2).

`DreamRollout` provenance fields under Option-1 semantics (plan §3.1's
amendment note):

  `seed_replay_segment_id` = `env_step` of the warmup window's start
  `seed_replay_step_offset` = 0 (first build always reads the window from
                              its start; the offset slot is reserved)

The warmup length itself is recorded in `DreamRollout.sampling_parameters`
under the key `"replay_warmup_length"` at Phase 2's writer.

**Out of scope at Probe 3** (per plan §2.2):
  - No prioritized replay sampling — uniform over valid window starts.
  - No learned seed selection.
  - No mirror-driven seed selection (mirror is logs-only per synthesis §6).
  - No storage of historical `(h, z)` in the replay buffer (Option 2).

**Closed-Literal discipline.** `SeedSelectionConfig.mode` is closed at
`{"replay", "perturbed_prior", "hybrid"}` and `hybrid_alpha_distribution` at
`{"uniform_0_1", "fixed_0_5", "beta_2_2"}`. Widening either requires a
separate synthesis. A `"current_waking"` mode is deliberately absent here —
the Phase 3 waking-planning control lives on `DreamRolloutConfig`'s
`seed_strategy_for_control` knob (Phase 2), not on `SeedSelectionConfig`.

This module imports from `kind.training.replay` and `kind.agents.world_model`
only. It does not import `kind.training.dream`, `kind.training.state_machine`,
or `kind.mirror.*`; those are downstream phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from kind.agents.world_model import WorldModel
from kind.training.replay import SequenceReplayBuffer

__all__ = [
    "DreamSeed",
    "SeedSelectionConfig",
    "select_seed",
]


@dataclass(frozen=True)
class DreamSeed:
    """The Phase 2 dream-rollout consumer's initial-conditions input.

    `h_init` has shape `(1, h_dim)`; `z_init` has shape `(1, z_dim)`. Both
    live on the world model's device.

    Provenance is populated per mode:
      - `replay`: `replay_segment_id` (= window-start `env_step`),
        `replay_step_offset` (= 0).
      - `perturbed_prior`: `perturbation_magnitude` (= L2 of the realized
        perturbation tensor — the concatenated h and z noise, computed not
        theoretical).
      - `hybrid`: all of the above plus `hybrid_mixture_alpha` (in [0, 1];
        1.0 = pure replay, 0.0 = pure perturbed_prior).

    `rng_seed` is the seeded state that drove window selection, posterior
    sampling during warmup, perturbation noise, and alpha sampling.
    Reproducibility chain: `(rng_seed, world_model weights, buffer obs
    window) → (h_init, z_init)`.

    In-memory state only; not telemetry. The DreamRollout schema (Phase 0)
    is where the provenance is persisted at Phase 2's writer.
    """

    mode: Literal["replay", "perturbed_prior", "hybrid"]
    h_init: Tensor
    z_init: Tensor

    replay_segment_id: int | None
    replay_step_offset: int | None
    perturbation_magnitude: float | None
    hybrid_mixture_alpha: float | None

    rng_seed: int


@dataclass(frozen=True)
class SeedSelectionConfig:
    """Probe-4-perturbable knobs for seed construction.

    `replay_warmup_length=8` is the Phase 0.5 floor: shorter warmups leave
    `h_final` dominated by GRU bias rather than obs structure, which would
    make the visibility-smoke gating (Phase 3) noisier on a parameter
    Probe 3 controls.
    """

    mode: Literal["replay", "perturbed_prior", "hybrid"] = "replay"
    perturbation_sigma: float = 0.1
    hybrid_alpha_distribution: Literal["uniform_0_1", "fixed_0_5", "beta_2_2"] = (
        "uniform_0_1"
    )
    replay_min_segment_age_steps: int = 1000
    replay_warmup_length: int = 8


def select_seed(
    replay_buffer: SequenceReplayBuffer,
    world_model: WorldModel,
    config: SeedSelectionConfig,
    rng: torch.Generator,
    *,
    perturbed_prior_anchor: Tensor | None = None,
) -> DreamSeed:
    """Produce a `DreamSeed` for a Phase 2 dream rollout.

    Modes (per plan §2.2 amended):
      - `replay`: sample a valid window of `replay_warmup_length` from the
        buffer (uniform over valid starts, respecting
        `replay_min_segment_age_steps` and episode boundaries); read obs and
        action sequences; re-encode through the world model from a zero
        `(h, z, a)` start; the final `(h, z)` is the seed.
      - `perturbed_prior`: split `perturbed_prior_anchor` into `(h_anchor,
        z_anchor)` (h_dim-prefix, z_dim-suffix); add a Gaussian perturbation
        with std `perturbation_sigma` independently to each side. The
        argument is required in this mode.
      - `hybrid`: run the replay procedure to obtain `(h_replay, z_replay)`;
        perturb that anchor by a `perturbation_sigma`-scaled Gaussian; sample
        `alpha` per `hybrid_alpha_distribution`; return the convex
        combination `alpha * (h_replay, z_replay) + (1 - alpha) *
        (h_perturbed, z_perturbed)`. `perturbed_prior_anchor` is ignored in
        hybrid mode — the anchor is the re-encoded replay endpoint.

    `rng` is consumed once to derive `rng_seed`, which seeds a fresh
    generator that drives every downstream stochastic step. The full seed is
    therefore byte-reproducible against `(rng_seed, world_model weights,
    buffer obs window)`.

    `perturbed_prior_anchor` shape, when used: `(1, h_dim + z_dim)` —
    h-concat-z. The split point is taken from `world_model.config.h_dim`.
    """
    # Derive a per-seed RNG state from the incoming generator. The incoming
    # rng advances by exactly one int draw; the subsequent seed_gen is then
    # used for every downstream stochastic step so the full seed is
    # reproducible from rng_seed alone.
    rng_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
    seed_gen = torch.Generator()
    seed_gen.manual_seed(rng_seed)

    h_dim = world_model.config.h_dim
    z_dim = world_model.config.z_dim

    if config.mode == "replay":
        h_init, z_init, segment_id = _replay_warmup(
            replay_buffer, world_model, config, seed_gen
        )
        return DreamSeed(
            mode="replay",
            h_init=h_init,
            z_init=z_init,
            replay_segment_id=segment_id,
            replay_step_offset=0,
            perturbation_magnitude=None,
            hybrid_mixture_alpha=None,
            rng_seed=rng_seed,
        )

    if config.mode == "perturbed_prior":
        if perturbed_prior_anchor is None:
            raise ValueError(
                "select_seed(mode='perturbed_prior') requires "
                "perturbed_prior_anchor (no replay step runs in pure "
                "perturbed_prior mode)"
            )
        expected_shape = (1, h_dim + z_dim)
        if tuple(perturbed_prior_anchor.shape) != expected_shape:
            raise ValueError(
                f"perturbed_prior_anchor must have shape {expected_shape}, "
                f"got {tuple(perturbed_prior_anchor.shape)}"
            )
        anchor_h = perturbed_prior_anchor[:, :h_dim]
        anchor_z = perturbed_prior_anchor[:, h_dim:]
        h_init, z_init, magnitude = _perturb_anchor(
            anchor_h, anchor_z, config.perturbation_sigma, seed_gen
        )
        return DreamSeed(
            mode="perturbed_prior",
            h_init=h_init,
            z_init=z_init,
            replay_segment_id=None,
            replay_step_offset=None,
            perturbation_magnitude=magnitude,
            hybrid_mixture_alpha=None,
            rng_seed=rng_seed,
        )

    # mode == "hybrid": replay re-encoding is the anchor, perturbed_prior_anchor
    # is ignored per plan §2.2.
    h_replay, z_replay, segment_id = _replay_warmup(
        replay_buffer, world_model, config, seed_gen
    )
    h_perturbed, z_perturbed, magnitude = _perturb_anchor(
        h_replay, z_replay, config.perturbation_sigma, seed_gen
    )
    alpha = _sample_alpha(config.hybrid_alpha_distribution, seed_gen)
    h_init = alpha * h_replay + (1.0 - alpha) * h_perturbed
    z_init = alpha * z_replay + (1.0 - alpha) * z_perturbed
    return DreamSeed(
        mode="hybrid",
        h_init=h_init,
        z_init=z_init,
        replay_segment_id=segment_id,
        replay_step_offset=0,
        perturbation_magnitude=magnitude,
        hybrid_mixture_alpha=alpha,
        rng_seed=rng_seed,
    )


# ---- internals (named with leading underscore but importable for tests) -----


def _replay_warmup(
    replay_buffer: SequenceReplayBuffer,
    world_model: WorldModel,
    config: SeedSelectionConfig,
    seed_gen: torch.Generator,
) -> tuple[Tensor, Tensor, int]:
    """Sample a valid window, read its obs/action, and re-encode from zero.

    Returns `(h_final, z_final, start_env_step)`. The first stochastic draw
    on `seed_gen` is the window-start pick; subsequent draws are the
    posterior-sampling noise inside `_run_warmup`.
    """
    L = config.replay_warmup_length
    starts = replay_buffer.valid_seed_window_starts(
        length=L, min_age_steps=config.replay_min_segment_age_steps
    )
    if not starts:
        raise RuntimeError(
            f"no valid window of length {L} satisfying "
            f"replay_min_segment_age_steps={config.replay_min_segment_age_steps} "
            f"in buffer (buffer_size={len(replay_buffer)})"
        )
    pick = int(torch.randint(0, len(starts), (1,), generator=seed_gen).item())
    start_env_step = starts[pick]
    obs_seq, action_seq = replay_buffer.get_window(start_env_step, L)
    h_final, z_final = _run_warmup(world_model, obs_seq, action_seq, seed_gen)
    return h_final, z_final, start_env_step


def _run_warmup(
    world_model: WorldModel,
    obs_seq: Tensor,
    action_seq: Tensor,
    seed_gen: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Run the world model forward from zero `(h, z, a)` over the window.

    `obs_seq` has shape `(L, *obs_shape)` (e.g. `(L, 1, 32, 32)`);
    `action_seq` has shape `(L,)` long. Each step matches the runner's
    training-time pattern (`runner.py` `_train_step`): apply `recurrence`
    then `encode → posterior → sample z`, then advance `a_prev` to the
    current action. After the loop the final `(h, z)` is the seed.

    Posterior sampling uses `seed_gen` (a CPU generator) to draw standard
    normal noise; the noise is then moved to the world model's device. This
    keeps the byte-equal reproducibility independent of the device the
    model is on (the runner's MPS / CUDA path produces the same noise
    pattern given the same `rng_seed`).
    """
    cfg = world_model.config
    param = next(world_model.parameters())
    device = param.device
    dtype = param.dtype
    L = obs_seq.shape[0]

    h = torch.zeros(1, cfg.h_dim, device=device, dtype=dtype)
    z = torch.zeros(1, cfg.z_dim, device=device, dtype=dtype)
    a_prev = torch.zeros(1, dtype=torch.long, device=device)

    obs_seq_dev = obs_seq.to(device=device, dtype=dtype)
    action_seq_dev = action_seq.to(device=device, dtype=torch.long)

    with torch.no_grad():
        for t in range(L):
            obs_t = obs_seq_dev[t : t + 1]
            h = world_model.recurrence(h, z, a_prev)
            embed = world_model.encode(obs_t)
            mu_q, log_sigma_q = world_model.posterior(h, embed)
            sigma_q = torch.exp(log_sigma_q)
            noise_cpu = torch.randn(mu_q.shape, generator=seed_gen)
            noise = noise_cpu.to(device=mu_q.device, dtype=mu_q.dtype)
            z = mu_q + sigma_q * noise
            a_prev = action_seq_dev[t : t + 1]

    return h, z


def _perturb_anchor(
    anchor_h: Tensor,
    anchor_z: Tensor,
    sigma: float,
    seed_gen: torch.Generator,
) -> tuple[Tensor, Tensor, float]:
    """Add a Gaussian perturbation (std `sigma`) independently to h and z.

    Returns `(h_perturbed, z_perturbed, magnitude)`, where `magnitude` is
    the L2 norm of the *concatenated realized* perturbation (the noise
    actually applied), not the theoretical expected magnitude. The noise
    is sampled on CPU with `seed_gen` and moved to the anchor's device so
    reproducibility is independent of device.
    """
    noise_h_cpu = torch.randn(anchor_h.shape, generator=seed_gen) * sigma
    noise_z_cpu = torch.randn(anchor_z.shape, generator=seed_gen) * sigma
    noise_h = noise_h_cpu.to(device=anchor_h.device, dtype=anchor_h.dtype)
    noise_z = noise_z_cpu.to(device=anchor_z.device, dtype=anchor_z.dtype)
    combined = torch.cat([noise_h.flatten(), noise_z.flatten()])
    magnitude = float(torch.linalg.norm(combined).item())
    return anchor_h + noise_h, anchor_z + noise_z, magnitude


def _sample_alpha(
    distribution: Literal["uniform_0_1", "fixed_0_5", "beta_2_2"],
    seed_gen: torch.Generator,
) -> float:
    """Draw `alpha ∈ [0, 1]` for the hybrid convex combination."""
    if distribution == "fixed_0_5":
        return 0.5
    if distribution == "uniform_0_1":
        return float(torch.rand((1,), generator=seed_gen).item())
    # beta_2_2: construct two Gamma(2, 1) samples from four uniforms
    # (Gamma(2, 1) = -log(U1) - log(U2) = -log(U1 * U2)); Beta(2, 2)
    # = X / (X + Y). Routed through torch.rand so seed_gen drives the
    # entire draw, which torch.distributions.Beta cannot do.
    u = torch.rand((4,), generator=seed_gen)
    x = -torch.log(u[0]) - torch.log(u[1])
    y = -torch.log(u[2]) - torch.log(u[3])
    return float((x / (x + y)).item())
