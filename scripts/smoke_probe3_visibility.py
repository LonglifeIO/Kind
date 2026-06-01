#!/usr/bin/env python3
"""Probe 3 Phase 3 — the visibility smoke (the gate).

This is the load-bearing early empirical test (plan §1 Phase 3 row, §5
sequencing rationale; the Probe-2-Phase-12 / Probe-1.5-Phase-7 discipline):
*before* building the state machine (Phase 4), mirror integration (Phase 5),
and runtime protection (Phase 6), confirm the four-axis dream regime drives
non-degenerate, distinguishable variation in ``DreamRollout`` telemetry
relative to its two controls. If it does not, the four-axis differentiation
is structurally coded (Phase 2) but not *empirically* real, and Phases 4–8
stay gated until the dream regime is revised.

Two controls, two named concerns (synthesis §3):
  - **pure-prior control** — ``emit_dream_rollout`` with
    ``seed_strategy_for_control="prior_only"`` + ``temperature_mode="identity"``.
    Tests Concern A: "a pure prior rollout looks like the GRU baseline."
  - **waking-planning control** — the preserved Probe 1.5 rollout
    (current-state seed, actor-driven actions, T=1.0, ensemble not stepped),
    replicated here by :func:`waking_planning_rollout` (a faithful standalone
    copy of the Probe 1.5 ``Runner._emit_dream`` calibration handshake — see
    the runner-integration note below). Tests Concern B: "planning already
    does world-model rollouts; env-off doesn't differentiate."
The dream regime must be distinguishable from *both*.

**Unit of comparison (interpretive call — the gate's defensibility).** Per-step
values pooled across rollouts are autocorrelated within a trajectory (a
rollout is not iid), which inflates effective N and can make KS-D spuriously
large. Per-rollout aggregates (one scalar per rollout) are iid across rollouts
and are the *primary gate metric* here. KS-D ≥ 0.15 is a descriptive
distinguishability floor (plan §5; calibrated just below Probe 1.5 Phase 8's
cross-family 0.22–0.34 band), **not** a significance test — so we run enough
rollouts (default 50) that the per-rollout KS-D estimate is stable, and report
the sample size. A per-step pooled view is reported as a *secondary* read
only.

**Horizon is matched across all three regimes** for the comparison
(``--horizon``, default 30). ``cumulative_prior_entropy`` and
``max_step_latent_norm_change`` grow with horizon; matching horizon removes it
as a confound so KS-D reflects the four axes, not the rollout length. (The
preserved waking-planning method hardcodes ``dream_horizon=15``; the smoke runs
its *logic* at the matched horizon.)

**Bucketing is by ``sub_mode_tags``, not ``seed_kind``** (Phase 2 flag): the
pure-prior control records as ``seed_kind="perturbed_prior"`` with
``seed_perturbation_magnitude=0.0`` and a ``"prior_only_control"`` tag, so
bucketing by ``seed_kind`` would silently contaminate the perturbed_prior
dream arm with controls.

**Cold-start non-degeneracy pre-flight runs first** (deliverable 3 / Phase 1
flag 2 / Probe-1.5-Phase-7 lesson): on the real buffer, replay seeds must vary
across windows and must carry obs structure beyond the GRU's prior-only
trajectory. A degenerate replay arm makes the gate meaningless, so the smoke
aborts ("raise replay_warmup_length") rather than computing KS-D over
degenerate seeds.

**Watts default (plan §10).** This script reads telemetry and runs rollouts; it
adds **no** Io-readable surface. The smoke is pure analysis/observation — Io
never reads ``DreamRollout`` records, the KS-D table, or any output here.

**Runner-integration note (plan §2.3 / prompt pitfall).** The waking-planning
arm is produced by a standalone replica of the Probe 1.5 ``Runner._emit_dream``
calibration handshake rather than by instantiating a full ``Runner`` (whose
env-server coupling and sink wiring are heavier than a self-contained gate
warrants). The replica mirrors that handshake's body exactly — current-state
seed, actor every step, T=1.0 prior sampling, no ensemble step, ``"0.2.0"``
record. The pure-prior arm is the cleaner fully-self-contained comparison (both
arms via ``emit_dream_rollout``); both controls are in scope and both are run.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from safetensors.torch import load_file
from torch import Tensor

from kind.agents.actor import Actor
from kind.agents.ensemble import LatentDisagreementEnsemble
from kind.agents.views import PolicyView
from kind.agents.world_model import WorldModel, WorldModelConfig
from kind.env.grid_world import GridWorld, GridWorldConfig
from kind.observer.schemas import SCHEMA_VERSION, DreamRollout
from kind.training.dream import (
    DreamRolloutConfig,
    TempSchedule,
    compute_checkpoint_hash,
    compute_perturbed_prior_anchor,
    emit_dream_rollout,
)
from kind.training.dream_seed import SeedSelectionConfig, _run_warmup
from kind.training.replay import SequenceReplayBuffer, Transition

# The five KS fields the plan names (§1 Phase 3 row / §5). All are computed as
# one scalar per rollout (the primary, iid unit of comparison). The first three
# are recorded directly on DreamRollout; the last two are aggregated by the
# smoke (the ensemble trajectory's mean; the decoded frames' mean Shannon
# entropy — Phase 2 records the raw frames but no entropy).
KS_FIELDS: tuple[str, ...] = (
    "mean_step_kl_successive_priors",
    "cumulative_prior_entropy",
    "max_step_latent_norm_change",
    "ensemble_disagreement_mean",
    "decoded_obs_entropy_mean",
)

# Bucket controls by this tag, never by seed_kind (prior_only shares
# seed_kind="perturbed_prior" with real perturbed-prior dreams).
PRIOR_ONLY_CONTROL_TAG: str = "prior_only_control"

KS_D_THRESHOLD_DEFAULT: float = 0.15
MIN_FIELDS_TO_PASS: int = 3  # plan §5: "at least three of {…}"


# =====================================================================
# Measurement machinery (pure, importable — unit-tested in
# tests/test_dream_visibility_smoke.py)
# =====================================================================


def ks_d(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Two-sample Kolmogorov–Smirnov statistic: max |CDF_a − CDF_b|.

    Pooled-sort form (matches ``scripts/probe1_5_compare_controls._ks_two_sample``
    to numerical precision): sort both samples, evaluate each empirical CDF at
    the union of sample points, take the max absolute gap. The D statistic only
    — no p-value — because the gate is a descriptive distinguishability floor,
    not a significance test (module docstring).
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(f"KS expects 1-D arrays; got a.ndim={a.ndim}, b.ndim={b.ndim}")
    if a.size == 0 or b.size == 0:
        raise ValueError(
            f"KS requires non-empty samples; got len(a)={a.size}, len(b)={b.size}"
        )
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    union = np.concatenate([a_sorted, b_sorted])
    cdf_a = np.searchsorted(a_sorted, union, side="right") / a_sorted.size
    cdf_b = np.searchsorted(b_sorted, union, side="right") / b_sorted.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def decoded_obs_entropy(frame_bytes: bytes) -> float:
    """Shannon entropy (bits) of one decoded-obs frame's uint8 histogram.

    Phase 2 records ``sequence_decoded_obs`` as quantised uint8 bytes per step
    but no entropy; the smoke computes it as the per-frame intensity-histogram
    entropy. A flat / collapsed reconstruction (every pixel one value) has
    entropy 0; a richly varied frame approaches 8 bits.
    """
    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    if arr.size == 0:
        return 0.0
    counts = np.bincount(arr, minlength=256).astype(np.float64)
    probs = counts / counts.sum()
    nz = probs[probs > 0.0]
    return float(-np.sum(nz * np.log2(nz)))


def _ensemble_disagreement_mean(record: DreamRollout) -> float | None:
    seq = record.sequence_ensemble_disagreement_variance
    if seq is None:
        return None  # structural absence (axis 2): waking-planning never steps the ensemble
    if len(seq) == 0:
        return 0.0
    return float(np.mean(np.asarray(seq, dtype=np.float64)))


def _decoded_obs_entropy_mean(record: DreamRollout) -> float | None:
    frames = record.sequence_decoded_obs
    if frames is None or len(frames) == 0:
        return None
    per_step = np.asarray([decoded_obs_entropy(f) for f in frames], dtype=np.float64)
    return float(np.mean(per_step))


def per_rollout_field(record: DreamRollout, field_name: str) -> float | None:
    """Extract one KS field from one record as a single per-rollout scalar.

    Cross-version aware: the same semantic field is pulled from a ``"0.2.0"``
    waking-planning record and a ``"0.3.0"`` dream/pure-prior record. The three
    base fields (``mean_step_kl_successive_priors``, ``cumulative_prior_entropy``,
    ``max_step_latent_norm_change``) and ``sequence_decoded_obs`` exist and mean
    the same thing on both versions (Probe-1-era fields on the shared
    ``DreamRollout`` model). ``ensemble_disagreement_mean`` is ``0.3.0``-only;
    ``None`` on a ``"0.2.0"`` record signals the axis-2 structural absence.
    """
    if field_name == "mean_step_kl_successive_priors":
        return float(record.mean_step_kl_successive_priors)
    if field_name == "cumulative_prior_entropy":
        return float(record.cumulative_prior_entropy)
    if field_name == "max_step_latent_norm_change":
        return float(record.max_step_latent_norm_change)
    if field_name == "ensemble_disagreement_mean":
        return _ensemble_disagreement_mean(record)
    if field_name == "decoded_obs_entropy_mean":
        return _decoded_obs_entropy_mean(record)
    raise ValueError(f"unknown KS field {field_name!r}")


def extract_per_rollout(
    records: list[DreamRollout],
) -> dict[str, NDArray[np.float64]]:
    """Per-rollout scalar series per KS field for one regime.

    A field is omitted from the returned dict when *every* record returns
    ``None`` for it (structural absence — e.g. ensemble disagreement on the
    ``"0.2.0"`` waking-planning control). Within a present field, ``None``
    contributions are dropped; a ragged field would surface as a shorter array.
    """
    out: dict[str, NDArray[np.float64]] = {}
    for name in KS_FIELDS:
        vals = [per_rollout_field(r, name) for r in records]
        present = [v for v in vals if v is not None]
        if not present:
            continue
        out[name] = np.asarray(present, dtype=np.float64)
    return out


def per_step_pooled(
    records: list[DreamRollout], field_name: str
) -> NDArray[np.float64]:
    """Secondary view: per-step values pooled across rollouts.

    Autocorrelated within a rollout (not iid) — reported only as a secondary
    read, never the gate. Defined for the per-step fields whose pooling is
    meaningful: successive-prior latent change (from ``sequence_z_prior``
    diffs), per-step prior entropy, per-step ensemble disagreement, per-step
    decoded-obs entropy.
    """
    pooled: list[float] = []
    for r in records:
        if field_name == "latent_step_change":
            zp = np.asarray(r.sequence_z_prior, dtype=np.float64)
            if zp.shape[0] >= 2:
                diffs = np.linalg.norm(np.diff(zp, axis=0), axis=1)
                pooled.extend(diffs.tolist())
        elif field_name == "prior_entropy":
            pooled.extend(float(x) for x in r.sequence_prior_entropy)
        elif field_name == "ensemble_disagreement":
            seq = r.sequence_ensemble_disagreement_variance
            if seq is not None:
                pooled.extend(float(x) for x in seq)
        elif field_name == "decoded_obs_entropy":
            frames = r.sequence_decoded_obs
            if frames is not None:
                pooled.extend(decoded_obs_entropy(f) for f in frames)
        else:
            raise ValueError(f"unknown per-step field {field_name!r}")
    return np.asarray(pooled, dtype=np.float64)


def is_degenerate(
    series: NDArray[np.float64], *, rel_std_floor: float = 1e-3
) -> bool:
    """True if a per-rollout series carries no within-regime variation.

    A regime collapsed to a fixed trajectory can produce a high *between*-regime
    KS-D that means nothing (prompt pitfall: assert within-regime variation
    before trusting between-regime distance). Degenerate iff zero std or
    coefficient of variation below ``rel_std_floor``.
    """
    if series.size < 2:
        return True
    std = float(np.std(series))
    if std == 0.0:
        return True
    scale = float(np.abs(np.mean(series))) + 1e-12
    return (std / scale) < rel_std_floor


# =====================================================================
# Regime → config mapping (unit-tested)
# =====================================================================


def is_prior_only_control(record: DreamRollout) -> bool:
    """True iff a record is a pure-prior control, bucketed by ``sub_mode_tags``.

    Never by ``seed_kind``: the pure-prior control records as
    ``seed_kind="perturbed_prior"`` (a prior-derived seed with zero
    perturbation), so bucketing on ``seed_kind`` would fold controls into the
    perturbed_prior dream arm and contaminate it.
    """
    return bool(record.sub_mode_tags) and PRIOR_ONLY_CONTROL_TAG in (
        record.sub_mode_tags or []
    )


def dream_config(horizon: int) -> DreamRolloutConfig:
    """The real dream regime: all four axes active (Phase 2 defaults)."""
    return DreamRolloutConfig(horizon=horizon)


def pure_prior_config(horizon: int) -> DreamRolloutConfig:
    """Concern-A control: pure-prior rollout, no seed perturbation, no schedule.

    ``seed_strategy_for_control="prior_only"`` ignores the replay seed and
    samples ``(h_init, z_init)`` from the prior at a zero ``h``;
    ``temperature_mode="identity"`` forces the per-step multiplier to 1.0. The
    record self-describes via the ``"prior_only_control"`` ``sub_mode_tag``
    (and ``seed_perturbation_magnitude=0.0``) — which is what the smoke buckets
    on (never ``seed_kind``).
    """
    return DreamRolloutConfig(
        horizon=horizon,
        temperature_mode="identity",
        seed_strategy_for_control="prior_only",
    )


# =====================================================================
# Cold-start non-degeneracy pre-flight (deliverable 3 — runs before KS-D)
# =====================================================================


@dataclass(frozen=True)
class ColdStartResult:
    passed: bool
    seed_variation: float  # mean pairwise relative L2 distance between window seeds
    obs_structure: float  # mean relative L2 gap (replay h_init vs prior-only warmup)
    seed_variation_floor: float
    obs_structure_floor: float
    n_windows: int
    message: str


def _prior_only_warmup(
    world_model: WorldModel,
    action_seq: Tensor,
    seed_gen: torch.Generator,
) -> Tensor:
    """The GRU-bias baseline: the warmup recurrence run *without* obs.

    Identical recurrence + action sequence to the replay warmup, but ``z`` is
    sampled from the *prior* ``p(z|h)`` each step instead of the posterior
    ``q(z|h, obs)``. The endpoint ``h`` is therefore the GRU's trajectory
    driven by actions and its own bias, with the observation conditioning
    removed — the thing the replay seed must measurably exceed for the warmup
    to be carrying obs structure (Probe-1.5-Phase-7 lesson).
    """
    cfg = world_model.config
    param = next(world_model.parameters())
    device = param.device
    dtype = param.dtype
    h = torch.zeros(1, cfg.h_dim, device=device, dtype=dtype)
    z = torch.zeros(1, cfg.z_dim, device=device, dtype=dtype)
    a_prev = torch.zeros(1, dtype=torch.long, device=device)
    action_dev = action_seq.to(device=device, dtype=torch.long)
    with torch.no_grad():
        for t in range(action_dev.shape[0]):
            h = world_model.recurrence(h, z, a_prev)
            mu_p, log_sigma_p = world_model.prior(h)
            noise = torch.randn(mu_p.shape, generator=seed_gen).to(
                device=mu_p.device, dtype=mu_p.dtype
            )
            z = mu_p + torch.exp(log_sigma_p) * noise
            a_prev = action_dev[t : t + 1]
    return h


def cold_start_preflight(
    world_model: WorldModel,
    replay_buffer: SequenceReplayBuffer,
    seed_config: SeedSelectionConfig,
    rng: torch.Generator,
    *,
    n_windows: int = 8,
    seed_variation_floor: float = 0.01,
    obs_structure_floor: float = 0.05,
) -> ColdStartResult:
    """Assert the replay conditioning surface is non-zero on the real buffer.

    Two checks (deliverable 3):
      1. **Seed variation** — replay seeds vary across windows (not collapsed
         to one ``(h, z)`` regardless of window). Mean pairwise relative L2
         distance between ``n_windows`` distinct-window ``h_init`` vectors must
         exceed ``seed_variation_floor``.
      2. **Obs structure** — replay-seeded ``h_init`` differs meaningfully from
         the prior-only (no-obs) warmup baseline over the same action sequence:
         the warmup carries obs structure, not just GRU bias. Mean relative L2
         gap must exceed ``obs_structure_floor``.

    Either failing aborts the smoke (caller raises) — a degenerate replay arm
    makes the gate meaningless; the fix is a longer ``replay_warmup_length``.
    """
    starts = replay_buffer.valid_seed_window_starts(
        length=seed_config.replay_warmup_length,
        min_age_steps=seed_config.replay_min_segment_age_steps,
    )
    if len(starts) < 2:
        return ColdStartResult(
            passed=False,
            seed_variation=0.0,
            obs_structure=0.0,
            seed_variation_floor=seed_variation_floor,
            obs_structure_floor=obs_structure_floor,
            n_windows=len(starts),
            message=(
                f"only {len(starts)} valid replay window(s) of length "
                f"{seed_config.replay_warmup_length} at min_age="
                f"{seed_config.replay_min_segment_age_steps}; cannot assess "
                f"seed variation. Fill a larger buffer or lower min_age."
            ),
        )

    # Distinct windows, spread across the available starts.
    k = min(n_windows, len(starts))
    idxs = np.linspace(0, len(starts) - 1, k).round().astype(int)
    chosen = sorted({starts[int(i)] for i in idxs})

    h_seeds: list[Tensor] = []
    obs_gaps: list[float] = []
    for start in chosen:
        obs_seq, action_seq = replay_buffer.get_window(
            start, seed_config.replay_warmup_length
        )
        # One generator per window, derived from rng, so the posterior and
        # prior-only warmups share the same noise stream (the gap is then the
        # posterior-vs-prior difference, not noise).
        win_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
        g_post = torch.Generator()
        g_post.manual_seed(win_seed)
        h_replay, _z_replay = _run_warmup(world_model, obs_seq, action_seq, g_post)
        g_prior = torch.Generator()
        g_prior.manual_seed(win_seed)
        h_baseline = _prior_only_warmup(world_model, action_seq, g_prior)
        gap = float((h_replay - h_baseline).norm().item())
        denom = float(h_replay.norm().item()) + 1e-12
        obs_gaps.append(gap / denom)
        h_seeds.append(h_replay.detach().reshape(-1))

    stacked = torch.stack(h_seeds, dim=0)  # (k, h_dim)
    pair_rel: list[float] = []
    for i in range(stacked.shape[0]):
        for j in range(i + 1, stacked.shape[0]):
            d = float((stacked[i] - stacked[j]).norm().item())
            scale = float(stacked[i].norm().item()) + 1e-12
            pair_rel.append(d / scale)
    seed_variation = float(np.mean(pair_rel)) if pair_rel else 0.0
    obs_structure = float(np.mean(obs_gaps)) if obs_gaps else 0.0

    ok_var = seed_variation >= seed_variation_floor
    ok_obs = obs_structure >= obs_structure_floor
    passed = ok_var and ok_obs
    if passed:
        message = "cold-start pre-flight passed"
    else:
        reasons = []
        if not ok_var:
            reasons.append(
                f"seed variation {seed_variation:.4f} < floor {seed_variation_floor}"
            )
        if not ok_obs:
            reasons.append(
                f"obs structure {obs_structure:.4f} < floor {obs_structure_floor}"
            )
        message = (
            "cold-start pre-flight FAILED ("
            + "; ".join(reasons)
            + "). Raise replay_warmup_length (the warmup is dominated by GRU "
            "bias rather than obs structure) and/or fill a larger buffer."
        )
    return ColdStartResult(
        passed=passed,
        seed_variation=seed_variation,
        obs_structure=obs_structure,
        seed_variation_floor=seed_variation_floor,
        obs_structure_floor=obs_structure_floor,
        n_windows=stacked.shape[0],
        message=message,
    )


# =====================================================================
# Waking-planning control (standalone replica of the renamed runner method)
# =====================================================================


def _diag_gaussian_kl(
    mu1: Tensor, log_sigma1: Tensor, mu2: Tensor, log_sigma2: Tensor
) -> float:
    var1 = torch.exp(2.0 * log_sigma1)
    var2 = torch.exp(2.0 * log_sigma2)
    kl = log_sigma2 - log_sigma1 + (var1 + (mu1 - mu2) ** 2) / (2.0 * var2) - 0.5
    return float(kl.sum(dim=-1).mean().item())


def waking_planning_rollout(
    *,
    world_model: WorldModel,
    actor: Actor,
    h_curr: Tensor,
    z_curr: Tensor,
    horizon: int,
    run_id: str,
    checkpoint_id: str | None,
    device: torch.device,
) -> DreamRollout:
    """Goal-coupled, current-state-seeded prior rollout — the waking-planning
    control.

    A faithful standalone copy of the Probe 1.5 ``Runner._emit_dream``
    calibration handshake: seed = current waking ``(h, z)``; action =
    ``actor.forward`` every step (axis 1 present); prior sampled at the model's
    training-time temperature via ``Normal(mu, sigma).sample()`` (axis 3 at
    T=1.0); ensemble not stepped (axis 2 absent → ``"0.2.0"`` record with
    ``sequence_ensemble_disagreement_variance`` left as ``None``). The four-axis
    opposite of the dream regime.
    """
    with torch.no_grad():
        h = h_curr.detach()
        z = z_curr.detach()
        seed_h0 = h.squeeze(0).detach().cpu().tolist()
        seed_z0 = z.squeeze(0).detach().cpu().tolist()

        sequence_h: list[list[float]] = []
        sequence_z_prior: list[list[float]] = []
        sequence_action: list[int] = []
        sequence_action_logprob: list[float] = []
        sequence_prior_entropy: list[float] = []
        sequence_decoded_obs: list[bytes] = []

        cumulative_prior_entropy = 0.0
        kls: list[float] = []
        prev_mu: Tensor | None = None
        prev_log_sigma: Tensor | None = None
        max_norm_change = 0.0
        prev_z = z

        sp_zero = torch.zeros((), device=device)
        for _ in range(horizon):
            view = PolicyView(h=h, z=z, self_prediction_error=sp_zero)
            action_output = actor.forward(view)
            a = action_output.action
            sequence_action.append(int(a.item()))
            sequence_action_logprob.append(float(action_output.logprob.item()))

            h_next = world_model.recurrence(h, z, a)
            mu, log_sigma = world_model.prior(h_next)
            sigma = torch.exp(log_sigma)
            dist = torch.distributions.Normal(mu, sigma)
            z_next = cast(Tensor, dist.sample())  # type: ignore[no-untyped-call]

            decoded = world_model.decode(h_next, z_next)
            decoded_uint8 = (
                decoded.squeeze(0)
                .squeeze(0)
                .clamp(0.0, 1.0)
                .mul(255.0)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            sequence_decoded_obs.append(decoded_uint8.tobytes())

            step_entropy = float(
                cast(Tensor, dist.entropy()).sum(dim=-1).mean().item()  # type: ignore[no-untyped-call]
            )
            sequence_prior_entropy.append(step_entropy)
            cumulative_prior_entropy += step_entropy

            if prev_mu is not None and prev_log_sigma is not None:
                kls.append(_diag_gaussian_kl(prev_mu, prev_log_sigma, mu, log_sigma))
            prev_mu = mu
            prev_log_sigma = log_sigma

            norm_change = float((z_next - prev_z).norm().item())
            if norm_change > max_norm_change:
                max_norm_change = norm_change
            prev_z = z_next

            sequence_h.append(h_next.squeeze(0).detach().cpu().tolist())
            sequence_z_prior.append(z_next.squeeze(0).detach().cpu().tolist())
            h = h_next
            z = z_next

    mean_kl = sum(kls) / len(kls) if kls else 0.0
    return DreamRollout(
        schema_version=SCHEMA_VERSION,  # "0.2.0": axis-2 disagreement stays None
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        seed_step=0,
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
        sequence_self_prediction=None,
    )


# =====================================================================
# Checkpoint loading + buffer fill (the real-run path)
# =====================================================================


@dataclass
class LoadedCheckpoint:
    world_model: WorldModel
    actor: Actor
    ensemble: LatentDisagreementEnsemble
    checkpoint_id: str


def load_checkpoint(checkpoint_dir: Path, device: torch.device) -> LoadedCheckpoint:
    """Load a Probe 1.5 ``"0.2.0"`` checkpoint into fresh default-config models.

    The checkpoint is exact-byte (``"0.2.0"`` resume contract): split the
    safetensors blob by ``world_model.`` / ``actor.`` / ``ensemble.`` prefix
    and ``load_state_dict`` strictly. Default ``WorldModelConfig`` matches the
    Probe 1.5 substrate (h=200, z=16, embed=256, online target mode → no frozen
    / environmental projection tensors).
    """
    weights_path = checkpoint_dir / "weights.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"no weights.safetensors under {checkpoint_dir}")
    weights = load_file(str(weights_path), device=str(device))

    def sub(prefix: str) -> dict[str, Tensor]:
        return {k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)}

    wm_cfg = WorldModelConfig()
    world_model = WorldModel(wm_cfg).to(device)
    actor = Actor(wm_cfg.h_dim, wm_cfg.z_dim, wm_cfg.num_actions).to(device)
    ensemble = LatentDisagreementEnsemble(
        wm_cfg.h_dim, wm_cfg.z_dim, wm_cfg.num_actions
    ).to(device)
    world_model.load_state_dict(sub("world_model."))
    actor.load_state_dict(sub("actor."))
    ensemble.load_state_dict(sub("ensemble."))
    world_model.eval()
    actor.eval()
    ensemble.eval()
    return LoadedCheckpoint(
        world_model=world_model,
        actor=actor,
        ensemble=ensemble,
        checkpoint_id=checkpoint_dir.name,
    )


def _obs_to_cpu(observation: NDArray[np.uint8]) -> Tensor:
    """uint8 (32, 32) → float32 (1, 32, 32) in [0, 1], matching the runner."""
    return torch.from_numpy(observation.astype(np.float32) / 255.0).unsqueeze(0)


@dataclass
class WakingState:
    h: Tensor
    z: Tensor


def fill_buffer(
    loaded: LoadedCheckpoint,
    *,
    n_steps: int,
    capacity: int,
    env_seed: int,
    device: torch.device,
    waking_state_stride: int,
) -> tuple[SequenceReplayBuffer, list[WakingState]]:
    """Regenerate a real Probe 1.5 buffer by stepping GridWorld under the
    loaded checkpoint.

    The buffer is not persisted to disk (runner: "in-memory only"), so the
    smoke fills one from real transitions: obs → ``world_model.step`` →
    ``actor.forward`` → ``env.step`` → ``buffer.insert``, mirroring the runner's
    waking step minus telemetry/training. Snapshots of the live waking ``(h, z)``
    are captured every ``waking_state_stride`` steps and returned for the
    waking-planning control's current-state seeds.
    """
    wm = loaded.world_model
    actor = loaded.actor
    env = GridWorld(GridWorldConfig(), seed=env_seed)
    buffer = SequenceReplayBuffer(capacity=capacity, sequence_length=32)
    waking_states: list[WakingState] = []

    es = env.reset()
    h = torch.zeros(1, wm.config.h_dim, device=device)
    z = torch.zeros(1, wm.config.z_dim, device=device)
    a = torch.zeros(1, dtype=torch.long, device=device)
    sp_zero = torch.zeros((), device=device)
    with torch.no_grad():
        for _ in range(n_steps):
            obs_cpu = _obs_to_cpu(es.observation)
            obs_dev = obs_cpu.unsqueeze(0).to(device)
            step = wm.step(obs_dev, h, z, a)
            action = actor.forward(
                PolicyView(h=step.h, z=step.z, self_prediction_error=sp_zero)
            ).action
            action_int = int(action.item())
            nxt = env.step(action_int)
            buffer.insert(
                Transition(
                    obs=obs_cpu,
                    action=action_int,
                    next_obs=_obs_to_cpu(nxt.observation),
                    env_step=es.env_step,
                    episode_id=es.episode_id,
                    step_in_episode=es.step_in_episode,
                )
            )
            if es.env_step % waking_state_stride == 0:
                waking_states.append(
                    WakingState(h=step.h.detach().clone(), z=step.z.detach().clone())
                )
            h, z, a = step.h, step.z, action
            es = nxt
    return buffer, waking_states


# =====================================================================
# Gate computation + report
# =====================================================================


@dataclass
class FieldComparison:
    field: str
    ks_d: float | None  # None == structural absence (field unavailable in a regime)
    dream_degenerate: bool
    control_degenerate: bool
    available: bool
    passes: bool


@dataclass
class ControlComparison:
    control_name: str
    fields: list[FieldComparison] = field(default_factory=list)
    n_fields_passing: int = 0
    passed: bool = False


@dataclass
class SmokeReport:
    threshold: float
    n_rollouts: int
    horizon: int
    cold_start: ColdStartResult
    comparisons: list[ControlComparison] = field(default_factory=list)
    per_step_secondary: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def proceed(self) -> bool:
        return self.cold_start.passed and all(c.passed for c in self.comparisons)


def compare_regimes(
    dream: dict[str, NDArray[np.float64]],
    control: dict[str, NDArray[np.float64]],
    control_name: str,
    *,
    threshold: float,
) -> ControlComparison:
    """KS-D per field between the dream regime and one control.

    A field counts toward the pass only if it is *available in both regimes*
    (present in both dicts — ensemble disagreement is absent for the
    waking-planning control by construction, axis 2) and *non-degenerate in
    both* (within-regime variation, asserted before trusting between-regime
    distance). Pass = at least ``MIN_FIELDS_TO_PASS`` available fields clear
    ``threshold``.
    """
    cmp = ControlComparison(control_name=control_name)
    for name in KS_FIELDS:
        d = dream.get(name)
        c = control.get(name)
        available = d is not None and c is not None
        if not available:
            cmp.fields.append(
                FieldComparison(
                    field=name,
                    ks_d=None,
                    dream_degenerate=False,
                    control_degenerate=False,
                    available=False,
                    passes=False,
                )
            )
            continue
        assert d is not None and c is not None
        d_deg = is_degenerate(d)
        c_deg = is_degenerate(c)
        distance = ks_d(d, c)
        passes = (not d_deg) and (not c_deg) and distance >= threshold
        cmp.fields.append(
            FieldComparison(
                field=name,
                ks_d=distance,
                dream_degenerate=d_deg,
                control_degenerate=c_deg,
                available=True,
                passes=passes,
            )
        )
    cmp.n_fields_passing = sum(1 for f in cmp.fields if f.passes)
    cmp.passed = cmp.n_fields_passing >= MIN_FIELDS_TO_PASS
    return cmp


def run_smoke(
    checkpoint_dir: Path,
    *,
    n_rollouts: int,
    horizon: int,
    threshold: float,
    buffer_steps: int,
    env_seed: int,
    rng_seed: int,
    replay_min_segment_age_steps: int,
    replay_warmup_length: int,
    device: torch.device,
) -> SmokeReport:
    """Load a checkpoint, fill a real buffer, run the three regimes, gate."""
    loaded = load_checkpoint(checkpoint_dir, device)
    seed_config = SeedSelectionConfig(
        replay_min_segment_age_steps=replay_min_segment_age_steps,
        replay_warmup_length=replay_warmup_length,
    )

    buffer, waking_states = fill_buffer(
        loaded,
        n_steps=buffer_steps,
        capacity=buffer_steps + 1,
        env_seed=env_seed,
        device=device,
        waking_state_stride=max(1, buffer_steps // (n_rollouts + 2)),
    )

    rng = torch.Generator()
    rng.manual_seed(rng_seed)

    # --- cold-start pre-flight (must pass before any KS-D) ---------------
    cold = cold_start_preflight(loaded.world_model, buffer, seed_config, rng)
    if not cold.passed:
        # Abort before computing KS-D over degenerate seeds.
        return SmokeReport(
            threshold=threshold,
            n_rollouts=n_rollouts,
            horizon=horizon,
            cold_start=cold,
        )

    checkpoint_hash = compute_checkpoint_hash(
        loaded.world_model, loaded.actor, loaded.ensemble
    )
    anchor = compute_perturbed_prior_anchor(loaded.world_model, device)

    def run_emit_regime(cfg: DreamRolloutConfig) -> list[DreamRollout]:
        out: list[DreamRollout] = []
        for i in range(n_rollouts):
            out.append(
                emit_dream_rollout(
                    world_model=loaded.world_model,
                    actor=None,
                    ensemble=loaded.ensemble,
                    replay_buffer=buffer,
                    seed_selection_config=seed_config,
                    config=cfg,
                    dream_session_id="smoke",
                    env_step_at_emit=i,
                    run_id="smoke",
                    checkpoint_id=loaded.checkpoint_id,
                    checkpoint_hash=checkpoint_hash,
                    rng=rng,
                    device=device,
                    perturbed_prior_anchor=anchor,
                )
            )
        return out

    dream_records = run_emit_regime(dream_config(horizon))
    pure_prior_records = run_emit_regime(pure_prior_config(horizon))

    # Bucket by tag, never by seed_kind. Sanity: pure_prior must carry the tag,
    # dream must not — getting this wrong contaminates the perturbed_prior arm.
    dream_records = [r for r in dream_records if not is_prior_only_control(r)]
    pure_prior_records = [r for r in pure_prior_records if is_prior_only_control(r)]

    # Waking-planning control: one rollout per captured current waking state.
    waking_records: list[DreamRollout] = []
    n_wp = min(n_rollouts, len(waking_states))
    for ws in waking_states[:n_wp]:
        waking_records.append(
            waking_planning_rollout(
                world_model=loaded.world_model,
                actor=loaded.actor,
                h_curr=ws.h,
                z_curr=ws.z,
                horizon=horizon,
                run_id="smoke",
                checkpoint_id=loaded.checkpoint_id,
                device=device,
            )
        )

    dream_scalars = extract_per_rollout(dream_records)
    pure_prior_scalars = extract_per_rollout(pure_prior_records)
    waking_scalars = extract_per_rollout(waking_records)

    report = SmokeReport(
        threshold=threshold,
        n_rollouts=n_rollouts,
        horizon=horizon,
        cold_start=cold,
    )
    report.comparisons.append(
        compare_regimes(
            dream_scalars, pure_prior_scalars, "pure_prior", threshold=threshold
        )
    )
    report.comparisons.append(
        compare_regimes(
            dream_scalars, waking_scalars, "waking_planning", threshold=threshold
        )
    )

    # Secondary per-step pooled view (reported, never gated).
    for field_name in ("latent_step_change", "prior_entropy", "decoded_obs_entropy"):
        d_pool = per_step_pooled(dream_records, field_name)
        pp_pool = per_step_pooled(pure_prior_records, field_name)
        wp_pool = per_step_pooled(waking_records, field_name)
        entry: dict[str, float] = {}
        if d_pool.size and pp_pool.size:
            entry["vs_pure_prior"] = ks_d(d_pool, pp_pool)
        if d_pool.size and wp_pool.size:
            entry["vs_waking_planning"] = ks_d(d_pool, wp_pool)
        report.per_step_secondary[field_name] = entry

    return report


# =====================================================================
# Reporting + CLI
# =====================================================================


def format_report(report: SmokeReport) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("Probe 3 Phase 3 — visibility smoke (the gate)")
    lines.append("=" * 72)
    lines.append(
        f"sample size: {report.n_rollouts} rollouts/regime | horizon: "
        f"{report.horizon} (matched across regimes) | "
        f"KS-D threshold: {report.threshold} | "
        f"unit of comparison: per-rollout aggregates (iid; primary gate metric)"
    )
    lines.append("")
    cs = report.cold_start
    lines.append("-- cold-start non-degeneracy pre-flight (real buffer) --")
    lines.append(
        f"  seed variation:  {cs.seed_variation:.4f}  (floor {cs.seed_variation_floor}) "
        f"[windows={cs.n_windows}]"
    )
    lines.append(
        f"  obs structure:   {cs.obs_structure:.4f}  (floor {cs.obs_structure_floor})"
    )
    lines.append(f"  -> {'PASS' if cs.passed else 'ABORT'}: {cs.message}")
    lines.append("")

    if not cs.passed:
        lines.append("KS-D not computed — cold-start aborted (degenerate replay arm).")
        lines.append("RECOMMENDATION: do NOT proceed to Phases 4-8; fix the replay seed.")
        return "\n".join(lines)

    lines.append("-- KS-D table (per field, per control; per-rollout aggregates) --")
    header = f"  {'field':<38}"
    for c in report.comparisons:
        header += f"{c.control_name:>18}"
    lines.append(header)
    for fname in KS_FIELDS:
        row = f"  {fname:<38}"
        for c in report.comparisons:
            fc = next(f for f in c.fields if f.field == fname)
            if not fc.available:
                cell = "n/a(absent)"
            elif fc.dream_degenerate or fc.control_degenerate:
                cell = f"{fc.ks_d:.3f}DEGEN"
            else:
                mark = "*" if fc.passes else " "
                cell = f"{fc.ks_d:.3f}{mark}"
            row += f"{cell:>18}"
        lines.append(row)
    lines.append("  (* = field clears threshold; n/a(absent) = structural axis-2 "
                 "absence in waking-planning)")
    lines.append("")

    lines.append("-- per-control call --")
    for c in report.comparisons:
        lines.append(
            f"  {c.control_name:<18} fields passing: {c.n_fields_passing} "
            f"(need >= {MIN_FIELDS_TO_PASS})  -> {'PASS' if c.passed else 'FAIL'}"
        )
    lines.append("")

    lines.append("-- secondary per-step pooled KS-D (autocorrelated; NOT the gate) --")
    for fname, entry in report.per_step_secondary.items():
        parts = ", ".join(f"{k}={v:.3f}" for k, v in entry.items())
        lines.append(f"  {fname:<22} {parts}")
    lines.append("")

    lines.append("=" * 72)
    if report.proceed:
        lines.append("GATE: PASS — dream regime is distinguishable from BOTH controls.")
        lines.append("RECOMMENDATION: proceed to Phases 4-8.")
    else:
        failed = [c.control_name for c in report.comparisons if not c.passed]
        lines.append(f"GATE: FAIL — dream collapses to control(s): {failed}.")
        lines.append("RECOMMENDATION: do NOT proceed to Phases 4-8; revise dream regime.")
        if "pure_prior" in failed:
            lines.append(
                "  pure_prior fail (Concern A): dream ~ default prior rollout. "
                "Knobs: prior temperature head/tail, re_seed_every_n_steps, "
                "action_policy."
            )
        if "waking_planning" in failed:
            lines.append(
                "  waking_planning fail (Concern B): dream ~ goal-directed "
                "imagination. Knobs: seed mode (replay vs perturbed_prior), "
                "replay_min_segment_age_steps, temperature schedule."
            )
    lines.append("=" * 72)
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(
            "runs/probe1_5_phase7_5-20260507-101800/checkpoints/ckpt-000001"
        ),
        help="Probe 1.5 checkpoint directory (contains weights.safetensors).",
    )
    parser.add_argument("--n-rollouts", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=KS_D_THRESHOLD_DEFAULT)
    parser.add_argument(
        "--buffer-steps",
        type=int,
        default=2500,
        help="Env steps to fill the real buffer (must exceed warmup + min-age).",
    )
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument(
        "--replay-min-age",
        type=int,
        default=1000,
        help="SeedSelectionConfig.replay_min_segment_age_steps (default 1000).",
    )
    parser.add_argument(
        "--replay-warmup-length",
        type=int,
        default=8,
        help="SeedSelectionConfig.replay_warmup_length (default 8).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    device = torch.device("cpu")
    report = run_smoke(
        args.checkpoint_dir,
        n_rollouts=args.n_rollouts,
        horizon=args.horizon,
        threshold=args.threshold,
        buffer_steps=args.buffer_steps,
        env_seed=args.env_seed,
        rng_seed=args.rng_seed,
        replay_min_segment_age_steps=args.replay_min_age,
        replay_warmup_length=args.replay_warmup_length,
        device=device,
    )
    print(format_report(report))
    return 0 if report.proceed else 1


if __name__ == "__main__":
    sys.exit(main())
