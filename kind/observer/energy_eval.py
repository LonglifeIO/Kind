"""Probe 3.5 dead-path assertion battery (the Phase-1 gate).

The energy channel must be demonstrably **world-grounded** before any preference
exists (synthesis T6; the project has been burned by silent dead paths three
times — CLAUDE.md / sink-routing lesson). This module is the runnable battery the
frozen pre-registration (`docs/decisions/probe3_5_preregistration_2026-06-10.md`
§2, §8) defines, plus the small amount of plumbing needed to collect the latents
the battery reads.

All probes are **eval-only**: ``true_energy`` is used here as a probe *target*
(battery A / C) but never enters a training loss — that boundary is the plan's
S-ENV rule, enforced upstream (the world model trains its energy reconstruction on
``sensed_energy``). This module reads a trained world model and a buffer of real
transitions; it never trains anything.

Batteries (frozen margins from the pre-registration §8, mirrored in
:class:`DeadPathMargins`):

* **A — latent-predictability.** A linear probe ``[h, z] → true_energy`` beats a
  mean-predictor baseline by margin ``Bm`` (MSE reduction ≥ 50% / R² ≥ 0.5).
* **B — interventional responsiveness.** Holding a real context fixed and sweeping
  the fused ``sensed_energy`` input low→high, ``decode_energy`` rises by ≥ ``Bδ1``
  of the input change (the decoder reads the energy observation); with the input
  held constant the response is ≤ ``Bδ2`` (no rise without the intervention). This
  is the realization of the S2 action-lesion "predicts replenishment only given
  the [energy] coincidence, not from nothing" for the amortized model.
* **C — action-history ablation.** Predicting ``true_energy`` from action history
  alone is worse than from the latents by ≥ ``Bg`` (``MSE_hist ≥ Bg · MSE_latent``)
  — the channel is grounded in the world (resource coincidence, visible in the
  latents), not derivable from the action sequence.
* **D — per-dim KL escape.** At least ``Bd1`` energy-correlated latent dim sustains
  per-dim KL ≥ ``Bd2`` nats (above the ``free_bits_per_dim = 1.0`` floor) over the
  evaluation window — the channel's dims are not collapsed.

Battery E (energy-scramble degradation) is a Phase-2+ behavioral check and lives
with the assay harness, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from kind.agents.world_model import WorldModel

__all__ = [
    "DeadPathMargins",
    "EnergyEvalData",
    "BatteryResult",
    "DeadPathBatteryReport",
    "collect_energy_eval_data",
    "battery_a_latent_predictability",
    "battery_b_interventional_response",
    "battery_c_action_history_ablation",
    "battery_d_per_dim_kl_escape",
    "run_dead_path_battery",
    # Amendment 01 (CONFIRMED 2026-06-10): B → B′ imagination intervention;
    # D demoted to a monitor; A, C, B′ are the amended Phase-1 gate.
    "BPrimeMargins",
    "BPrimeResult",
    "battery_b_prime_imagination_intervention",
    "AmendedGateReport",
    "run_amended_gate",
]


@dataclass(frozen=True)
class DeadPathMargins:
    """The frozen pre-registration §8 battery margins (Bm/Bδ1/Bδ2/Bg/Bd1/Bd2).

    These mirror `docs/decisions/probe3_5_preregistration_2026-06-10.md` §2/§8
    exactly. They are *not* tunable here — the doc is the authority; this is a
    code-side copy so the battery is runnable. Changing a value is editing the
    frozen pre-registration and requires a new dated doc.
    """

    # Bm: battery A latent-probe margin — probe R² over the mean baseline.
    a_min_r2: float = 0.5
    # Bδ1: battery B responsiveness floor — decoded rise as a fraction of the
    # swept input change (the decoder passes ≥ 80% of the energy observation).
    b_min_response_fraction: float = 0.8
    # Bδ2: battery B control ceiling — decoded change with the input held fixed,
    # as a fraction of the input sweep span (must be ~0).
    b_max_control_fraction: float = 0.2
    # Bg: battery C ablation gap — MSE(action-history) ≥ Bg × MSE(latent).
    c_min_mse_ratio: float = 1.5
    # Bd1 / Bd2: battery D — at least this many dims sustain per-dim KL ≥ this
    # many nats (the free_bits floor is 1.0).
    d_min_dims: int = 1
    d_min_kl_nats: float = 1.5


@dataclass(frozen=True)
class EnergyEvalData:
    """Collected per-step eval arrays from a teacher-forced rollout.

    ``h`` / ``z`` are ``(T, h_dim)`` / ``(T, z_dim)``; ``true_energy`` and
    ``sensed_energy`` are ``(T,)``; ``kl_per_dim`` is ``(T, z_dim)``;
    ``action`` is ``(T,)`` long. All numpy, on CPU — the battery is pure
    eval analysis.
    """

    h: NDArray[np.float64]
    z: NDArray[np.float64]
    true_energy: NDArray[np.float64]
    sensed_energy: NDArray[np.float64]
    kl_per_dim: NDArray[np.float64]
    action: NDArray[np.int64]


@dataclass(frozen=True)
class BatteryResult:
    """One battery's verdict plus the metric(s) it was judged on."""

    name: str
    passed: bool
    metrics: dict[str, float]
    detail: str


@dataclass(frozen=True)
class DeadPathBatteryReport:
    """The full A–D report. ``all_passed`` is the Phase-1 gate."""

    results: tuple[BatteryResult, ...]

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)


# ---- collection ----------------------------------------------------------


@torch.no_grad()
def collect_energy_eval_data(
    world_model: WorldModel,
    obs_seq: Tensor,
    action_seq: Tensor,
    sensed_energy_seq: Tensor,
    true_energy_seq: Tensor,
    *,
    device: torch.device | None = None,
) -> EnergyEvalData:
    """Teacher-force the world model along a real trajectory and collect latents.

    ``obs_seq`` is ``(T, C, H, W)``; ``action_seq``/``sensed_energy_seq``/
    ``true_energy_seq`` are ``(T,)``. At each step the posterior is fed the real
    observation *and* the real sensed energy (the channel as Io infers it), the
    recurrence is advanced with the real action, and ``(h, z, kl_per_dim)`` are
    recorded. ``true_energy`` is carried through untouched as the eval target.
    """
    dev = device if device is not None else next(world_model.parameters()).device
    world_model.eval()
    t_steps = int(obs_seq.shape[0])
    cfg = world_model.config

    h = torch.zeros(1, cfg.h_dim, device=dev)
    z = torch.zeros(1, cfg.z_dim, device=dev)
    a_prev = torch.zeros(1, dtype=torch.long, device=dev)

    h_rows: list[NDArray[np.float64]] = []
    z_rows: list[NDArray[np.float64]] = []
    kl_rows: list[NDArray[np.float64]] = []
    for t in range(t_steps):
        obs_t = obs_seq[t : t + 1].to(dev)
        sensed_t = sensed_energy_seq[t : t + 1].to(dev).reshape(1, 1)
        step = world_model.step(obs_t, h, z, a_prev, sensed_energy=sensed_t)
        h_rows.append(step.h.squeeze(0).double().cpu().numpy())
        z_rows.append(step.z.squeeze(0).double().cpu().numpy())
        kl_rows.append(step.kl_per_dim.squeeze(0).double().cpu().numpy())
        h, z = step.h, step.z
        a_prev = action_seq[t : t + 1].to(dev).long()

    return EnergyEvalData(
        h=np.stack(h_rows, axis=0),
        z=np.stack(z_rows, axis=0),
        true_energy=true_energy_seq.double().cpu().numpy(),
        sensed_energy=sensed_energy_seq.double().cpu().numpy(),
        kl_per_dim=np.stack(kl_rows, axis=0),
        action=action_seq.long().cpu().numpy(),
    )


# ---- linear-probe helper -------------------------------------------------


def _ridge_fit_predict_r2(
    features_train: NDArray[np.float64],
    target_train: NDArray[np.float64],
    features_eval: NDArray[np.float64],
    target_eval: NDArray[np.float64],
    *,
    ridge: float = 1e-3,
) -> tuple[float, float]:
    """Fit a ridge linear probe on the train split, return ``(r2, mse)`` on eval.

    ``r2`` is computed against the *eval* target's mean (so the mean-predictor
    baseline is R² = 0 by construction — battery A's "beats a mean baseline").
    Ridge keeps the fit well-posed when the latent dimension exceeds the sample
    count or features are collinear (the tiny-tensor regime).
    """
    x_train = np.concatenate(
        [features_train, np.ones((features_train.shape[0], 1))], axis=1
    )
    x_eval = np.concatenate(
        [features_eval, np.ones((features_eval.shape[0], 1))], axis=1
    )
    d = x_train.shape[1]
    reg = ridge * np.eye(d)
    reg[-1, -1] = 0.0  # do not regularize the bias
    gram = x_train.T @ x_train + reg
    weights = np.linalg.solve(gram, x_train.T @ target_train)
    pred = x_eval @ weights
    resid = target_eval - pred
    mse = float(np.mean(resid**2))
    denom = float(np.sum((target_eval - target_eval.mean()) ** 2))
    if denom <= 1e-12:
        # Degenerate target (no variance) — R² undefined; report 0.0.
        return 0.0, mse
    r2 = 1.0 - float(np.sum(resid**2)) / denom
    return r2, mse


def _split_index(n: int) -> int:
    return max(1, n // 2)


# ---- battery A — latent predictability -----------------------------------


def battery_a_latent_predictability(
    data: EnergyEvalData, margins: DeadPathMargins
) -> BatteryResult:
    """Probe ``[h, z] → true_energy``; pass if R² ≥ ``a_min_r2`` (vs mean baseline).

    Eval-only use of ``true_energy`` (the probe target). The probe is fit on the
    first half of the trajectory and evaluated on the second half, so a high R²
    reflects genuine latent information rather than in-sample overfit.
    """
    features = np.concatenate([data.h, data.z], axis=1)
    n = features.shape[0]
    s = _split_index(n)
    r2, mse = _ridge_fit_predict_r2(
        features[:s], data.true_energy[:s], features[s:], data.true_energy[s:]
    )
    passed = r2 >= margins.a_min_r2
    return BatteryResult(
        name="A_latent_predictability",
        passed=bool(passed),
        metrics={"r2": r2, "mse": mse, "threshold_r2": margins.a_min_r2},
        detail=(
            f"[h,z]→true_energy probe R²={r2:.3f} "
            f"(threshold ≥ {margins.a_min_r2}); MSE={mse:.5f}"
        ),
    )


# ---- battery B — interventional responsiveness ---------------------------


@torch.no_grad()
def battery_b_interventional_response(
    world_model: WorldModel,
    obs_seq: Tensor,
    action_seq: Tensor,
    sensed_energy_seq: Tensor,
    margins: DeadPathMargins,
    *,
    low: float = 0.2,
    high: float = 0.8,
    device: torch.device | None = None,
) -> BatteryResult:
    """Sweep the fused ``sensed_energy`` input on fixed contexts; measure decode.

    For a sample of real ``(obs, h_prev, z_prev, a_prev)`` contexts, run the world
    model twice — once with ``sensed_energy = low`` and once with ``high`` — and
    read ``decode_energy``. The mean rise as a fraction of the input span
    ``(high − low)`` must be ≥ ``b_min_response_fraction`` (the decoder reads the
    energy observation). The control — input held at ``low`` for both forwards —
    must change decode by ≤ ``b_max_control_fraction`` of the span (no rise without
    the intervention). This is the amortized-model realization of the S2
    action-lesion: decoded energy responds to the energy coincidence, not to
    nothing.
    """
    dev = device if device is not None else next(world_model.parameters()).device
    world_model.eval()
    t_steps = int(obs_seq.shape[0])
    cfg = world_model.config

    # Re-derive the per-step (h_prev, z_prev, a_prev) contexts by teacher-forcing.
    h = torch.zeros(1, cfg.h_dim, device=dev)
    z = torch.zeros(1, cfg.z_dim, device=dev)
    a_prev = torch.zeros(1, dtype=torch.long, device=dev)
    contexts: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
    for t in range(t_steps):
        obs_t = obs_seq[t : t + 1].to(dev)
        contexts.append((obs_t, h.clone(), z.clone(), a_prev.clone()))
        sensed_t = sensed_energy_seq[t : t + 1].to(dev).reshape(1, 1)
        step = world_model.step(obs_t, h, z, a_prev, sensed_energy=sensed_t)
        h, z = step.h, step.z
        a_prev = action_seq[t : t + 1].to(dev).long()

    span = high - low
    rises: list[float] = []
    controls: list[float] = []
    for obs_t, h_c, z_c, a_c in contexts:
        low_t = torch.full((1, 1), low, device=dev)
        high_t = torch.full((1, 1), high, device=dev)
        step_low = world_model.step(obs_t, h_c, z_c, a_c, sensed_energy=low_t)
        step_high = world_model.step(obs_t, h_c, z_c, a_c, sensed_energy=high_t)
        dec_low = float(step_low.energy_pred.reshape(-1)[0].item())
        dec_high = float(step_high.energy_pred.reshape(-1)[0].item())
        rises.append(dec_high - dec_low)
        # Control: same (low) input on both forwards — z is resampled from the
        # posterior, so any nonzero delta here is sampling noise, not response.
        step_low2 = world_model.step(obs_t, h_c, z_c, a_c, sensed_energy=low_t)
        dec_low2 = float(step_low2.energy_pred.reshape(-1)[0].item())
        controls.append(abs(dec_low2 - dec_low))

    mean_rise = float(np.mean(rises))
    response_fraction = mean_rise / span if span > 0 else 0.0
    control_fraction = float(np.mean(controls)) / span if span > 0 else 0.0
    passed = (
        response_fraction >= margins.b_min_response_fraction
        and control_fraction <= margins.b_max_control_fraction
    )
    return BatteryResult(
        name="B_interventional_response",
        passed=bool(passed),
        metrics={
            "response_fraction": response_fraction,
            "control_fraction": control_fraction,
            "mean_rise": mean_rise,
            "threshold_response": margins.b_min_response_fraction,
            "threshold_control": margins.b_max_control_fraction,
        },
        detail=(
            f"decode_energy rise/Δinput={response_fraction:.3f} "
            f"(≥ {margins.b_min_response_fraction}); "
            f"control={control_fraction:.3f} (≤ {margins.b_max_control_fraction})"
        ),
    )


# ---- battery C — action-history ablation ---------------------------------


def _action_history_features(
    action: NDArray[np.int64], num_actions: int, window: int
) -> NDArray[np.float64]:
    """One-hot action history: row ``t`` is the concatenated one-hots of actions
    ``t-window+1 .. t`` (zero-padded at the start). Shape ``(T, window*A)``."""
    t_steps = action.shape[0]
    feats = np.zeros((t_steps, window * num_actions), dtype=np.float64)
    for t in range(t_steps):
        for k in range(window):
            idx = t - k
            if idx < 0:
                break
            a = int(action[idx])
            feats[t, k * num_actions + a] = 1.0
    return feats


def battery_c_action_history_ablation(
    data: EnergyEvalData,
    margins: DeadPathMargins,
    *,
    num_actions: int = 5,
    history_window: int = 8,
) -> BatteryResult:
    """Latents must beat action-history-alone: ``MSE_hist ≥ Bg · MSE_latent``.

    If energy were a deterministic function of the action sequence, the GRU could
    predict it from action history without grounding it in observed resources
    (the "predictable from action history alone" trap, T6). Replenishment depends
    on *resource coincidence* — visible in the latents (which see the grid + the
    fused sensed energy), not in actions alone — so the latent probe must predict
    materially better.
    """
    features_latent = np.concatenate([data.h, data.z], axis=1)
    features_hist = _action_history_features(
        data.action, num_actions, history_window
    )
    n = features_latent.shape[0]
    s = _split_index(n)
    _, mse_latent = _ridge_fit_predict_r2(
        features_latent[:s],
        data.true_energy[:s],
        features_latent[s:],
        data.true_energy[s:],
    )
    _, mse_hist = _ridge_fit_predict_r2(
        features_hist[:s],
        data.true_energy[:s],
        features_hist[s:],
        data.true_energy[s:],
    )
    ratio = mse_hist / mse_latent if mse_latent > 1e-12 else float("inf")
    passed = ratio >= margins.c_min_mse_ratio
    return BatteryResult(
        name="C_action_history_ablation",
        passed=bool(passed),
        metrics={
            "mse_latent": mse_latent,
            "mse_history": mse_hist,
            "ratio": ratio,
            "threshold_ratio": margins.c_min_mse_ratio,
        },
        detail=(
            f"MSE(history)/MSE(latent)={ratio:.3f} "
            f"(≥ {margins.c_min_mse_ratio}); "
            f"latent={mse_latent:.5f}, history={mse_hist:.5f}"
        ),
    )


# ---- battery D — per-dim KL escape ---------------------------------------


def battery_d_per_dim_kl_escape(
    data: EnergyEvalData,
    margins: DeadPathMargins,
    *,
    window: int | None = None,
) -> BatteryResult:
    """At least ``d_min_dims`` latent dim sustains per-dim KL ≥ ``d_min_kl_nats``.

    Averaged over the final ``window`` steps (defaults to the whole eval
    trajectory; the frozen Bd3 window applies to the *training-step* KL during
    the run). A channel whose dims all sit at the ``free_bits_per_dim = 1.0``
    floor is collapsed (posterior-collapse failure, T6); the energy fusion should
    keep at least one dim's KL above the floor.
    """
    kl = data.kl_per_dim
    if window is not None and window < kl.shape[0]:
        kl = kl[-window:]
    mean_kl_per_dim = kl.mean(axis=0)
    n_escaping = int(np.sum(mean_kl_per_dim >= margins.d_min_kl_nats))
    max_kl = float(mean_kl_per_dim.max())
    passed = n_escaping >= margins.d_min_dims
    return BatteryResult(
        name="D_per_dim_kl_escape",
        passed=bool(passed),
        metrics={
            "n_dims_escaping": float(n_escaping),
            "max_mean_kl": max_kl,
            "threshold_kl": margins.d_min_kl_nats,
            "threshold_dims": float(margins.d_min_dims),
        },
        detail=(
            f"{n_escaping} dim(s) sustain KL ≥ {margins.d_min_kl_nats} nats "
            f"(need ≥ {margins.d_min_dims}); max mean per-dim KL = {max_kl:.3f}"
        ),
    )


# ---- the full battery ----------------------------------------------------


def run_dead_path_battery(
    world_model: WorldModel,
    obs_seq: Tensor,
    action_seq: Tensor,
    sensed_energy_seq: Tensor,
    true_energy_seq: Tensor,
    *,
    margins: DeadPathMargins | None = None,
    num_actions: int = 5,
    history_window: int = 8,
    kl_window: int | None = None,
    device: torch.device | None = None,
) -> DeadPathBatteryReport:
    """Run batteries A–D on a trained world model + a real eval trajectory."""
    m = margins if margins is not None else DeadPathMargins()
    data = collect_energy_eval_data(
        world_model,
        obs_seq,
        action_seq,
        sensed_energy_seq,
        true_energy_seq,
        device=device,
    )
    results = (
        battery_a_latent_predictability(data, m),
        battery_b_interventional_response(
            world_model,
            obs_seq,
            action_seq,
            sensed_energy_seq,
            m,
            device=device,
        ),
        battery_c_action_history_ablation(
            data, m, num_actions=num_actions, history_window=history_window
        ),
        battery_d_per_dim_kl_escape(data, m, window=kl_window),
    )
    return DeadPathBatteryReport(results=results)


# ==========================================================================
# Amendment 01 (CONFIRMED 2026-06-10) — battery B′ + the amended gate.
#
# The frozen battery above (A–D, `run_dead_path_battery`) is preserved verbatim
# for provenance — it is the battery the Phase-1 results record ran. The
# amendment re-aims the gate to the substrate's actual route (energy is model-led
# via `h`, not sensor-led via `z`): battery B (observation input-sweep) → B′
# (imagination intervention, the synthesis S2 action-lesion properly realized);
# battery D (per-dim KL escape) demoted from gate to monitor. A, C, B′ are the
# amended Phase-1 gate. Authority:
# `docs/decisions/probe3_5_preregistration_amendment01_2026-06-10.md`.
# ==========================================================================


@dataclass(frozen=True)
class BPrimeMargins:
    """Amendment-01 §2.1 / §6 confirmed margins for battery B′ (B′1, B′2).

    Not tunable here — the CONFIRMED amendment is the authority; this is a
    code-side copy so the battery is runnable.
    """

    # B′1: fraction of matched pairs where the coincident rollout's decoded
    # energy exceeds its non-coincident control.
    min_pair_fraction: float = 0.8
    # B′2: floor on the mean (coincident − control) decoded delta, as a fraction
    # of the *normalized* per-resource replenishment increment.
    min_mean_delta_frac_replenish: float = 0.5


@dataclass(frozen=True)
class BPrimeResult:
    """Battery B′ verdict plus the metrics it was judged on."""

    name: str
    passed: bool
    metrics: dict[str, float]
    detail: str


@torch.no_grad()
def battery_b_prime_imagination_intervention(
    world_model: WorldModel,
    h: Tensor,
    z: Tensor,
    action_coincident: Tensor,
    action_control: Tensor,
    *,
    replenish_norm: float,
    margins: BPrimeMargins | None = None,
    device: torch.device | None = None,
) -> BPrimeResult:
    """B′ — roll matched latents forward in imagination on paired actions.

    From each matched latent state ``(h, z)`` (real posterior states drawn from a
    rollout where the agent is one step from a resource), roll the world model
    forward **one imagined step** on a **coincident** action (steps onto the
    resource) and a **control** action (does not), reading ``decode_energy`` after
    each. The imagined step uses the **prior mean** for ``z'`` — deterministic,
    isolating the modelled action effect from sampling noise::

        h' = recurrence(h, z, a);  z' = E[prior(h')];  e = decode_energy(h', z')

    This is the synthesis S2 action-lesion realized for the model-led substrate:
    *does the world model predict replenishment from the modelled resource
    coincidence (not from nothing)?* — the route the channel actually uses (energy
    rides ``h``), rather than the redundant observation the frozen battery B swept.

    Pass (Amendment 01 §2.1): decoded energy on the coincident rollout exceeds its
    control in ≥ ``min_pair_fraction`` of pairs (B′1) **and** the mean delta is ≥
    ``min_mean_delta_frac_replenish × replenish_norm`` (B′2). ``replenish_norm`` is
    the per-resource replenishment expressed in normalized [0, 1] energy units
    (``energy_replenish_per_resource / energy_norm_max``), matching the units
    ``decode_energy`` is trained in.
    """
    m = margins if margins is not None else BPrimeMargins()
    dev = device if device is not None else next(world_model.parameters()).device
    world_model.eval()
    h_dev = h.to(dev)
    z_dev = z.to(dev)
    a_coin = action_coincident.to(dev).long()
    a_ctrl = action_control.to(dev).long()

    def _imagine(a: Tensor) -> Tensor:
        h_next = world_model.recurrence(h_dev, z_dev, a)
        z_mean, _ = world_model.prior(h_next)
        return world_model.decode_energy(h_next, z_mean).reshape(-1)

    delta = (_imagine(a_coin) - _imagine(a_ctrl)).double().cpu().numpy()
    n = int(delta.shape[0])
    pair_fraction = float(np.mean(delta > 0.0)) if n > 0 else 0.0
    mean_delta = float(np.mean(delta)) if n > 0 else 0.0
    min_mean_delta = m.min_mean_delta_frac_replenish * replenish_norm
    passed = (
        n > 0
        and pair_fraction >= m.min_pair_fraction
        and mean_delta >= min_mean_delta
    )
    return BPrimeResult(
        name="B_prime_imagination_intervention",
        passed=bool(passed),
        metrics={
            "n_pairs": float(n),
            "pair_fraction": pair_fraction,
            "mean_delta": mean_delta,
            "min_mean_delta": min_mean_delta,
            "replenish_norm": replenish_norm,
            "threshold_pair_fraction": m.min_pair_fraction,
        },
        detail=(
            f"coincident>control in {pair_fraction:.3f} of {n} pairs "
            f"(≥ {m.min_pair_fraction}); mean delta {mean_delta:.4f} "
            f"(≥ {min_mean_delta:.4f} = "
            f"{m.min_mean_delta_frac_replenish}×replenish_norm)"
        ),
    )


@dataclass(frozen=True)
class AmendedGateReport:
    """The Amendment-01 §3 gate: A, C, B′ gate; D retained as a monitor.

    ``gate_passed`` is the amended Phase-1 gate (D is **not** part of it —
    ``d_monitor`` is reported as a permanent monitor, read not gated).
    """

    a: BatteryResult
    c: BatteryResult
    b_prime: BPrimeResult
    d_monitor: BatteryResult

    @property
    def gate_passed(self) -> bool:
        return self.a.passed and self.c.passed and self.b_prime.passed


def run_amended_gate(
    world_model: WorldModel,
    obs_seq: Tensor,
    action_seq: Tensor,
    sensed_energy_seq: Tensor,
    true_energy_seq: Tensor,
    *,
    b_prime_h: Tensor,
    b_prime_z: Tensor,
    b_prime_action_coincident: Tensor,
    b_prime_action_control: Tensor,
    replenish_norm: float,
    margins: DeadPathMargins | None = None,
    b_prime_margins: BPrimeMargins | None = None,
    num_actions: int = 5,
    history_window: int = 8,
    kl_window: int | None = None,
    device: torch.device | None = None,
) -> AmendedGateReport:
    """Run the amended Phase-1 gate (A, C, B′) + the D monitor on a trained model.

    A and C read a teacher-forced eval trajectory (as in the frozen battery); B′
    reads matched-latent intervention contexts (collected env-side, where the
    agent is one step from a resource); D is computed and reported as a monitor.
    """
    m = margins if margins is not None else DeadPathMargins()
    data = collect_energy_eval_data(
        world_model,
        obs_seq,
        action_seq,
        sensed_energy_seq,
        true_energy_seq,
        device=device,
    )
    a = battery_a_latent_predictability(data, m)
    c = battery_c_action_history_ablation(
        data, m, num_actions=num_actions, history_window=history_window
    )
    d = battery_d_per_dim_kl_escape(data, m, window=kl_window)
    b_prime = battery_b_prime_imagination_intervention(
        world_model,
        b_prime_h,
        b_prime_z,
        b_prime_action_coincident,
        b_prime_action_control,
        replenish_norm=replenish_norm,
        margins=b_prime_margins,
        device=device,
    )
    return AmendedGateReport(a=a, c=c, b_prime=b_prime, d_monitor=d)
