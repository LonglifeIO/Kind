"""Probe 3.5 Phase 2 — the pragmatic preference (fills the F7 scaffold).

A **stateless, fixed, saturating Gaussian log-preference** over the world
model's decoded energy along imagined rollouts (synthesis DP5/DP6; Amendment
02 geometry). This is the entire pragmatic machinery: one frozen config
dataclass and one pure function. No learned value head, no critic, no reward
predictor, no state — the ingredients-only requirement in code form.

Functional form (constants confirmed by the builder, 2026-06-11)::

    d(e) = relu(|e − setpoint| − band_halfwidth)        # 0 inside the band
    v(e) = −S · tanh( precision · d(e)² / (2S) )

- **Flat in-band** (Amendment 02 B0a′: [0.45, 0.75]): ``v ≡ 0`` with exactly
  zero gradient — no hoarding reward, Keramati drive-reduction (T5).
- **Gaussian log-preference just outside the band**: for small ``d``,
  ``v ≈ −(precision/2)·d²`` — the log of a Gaussian over band-edge distance,
  ``precision`` = inverse variance (T4).
- **Bounded and saturating at large deviation**: ``|v| ≤ S`` always — the
  structural dominance bound (DP6 saturation; the deferred
  clip-to-epistemic-scale is *not* implemented). ``S = 1.0``
  [SAT-1 | pre, builder-confirmed 2026-06-11]: inert (<3% distortion) in the
  S1 0.1×–1× operating regime, binding at the extreme grid points — a guard,
  not a shaper.

**Precision is the dominance-relevant weight. Do not add a β or any outer
coefficient — the additive, coefficient-free form is load-bearing (DP5/DP6,
synthesis §8.3).** The term enters the actor objective as
``total_return = sum_disagreement + pragmatic_value`` with no weighting;
``precision`` is what the frozen S1 grid sweeps.

``precision = 0`` makes the function identically zero with identically zero
gradient — the degenerate-null configuration is a point on the same surface,
not a separate code path.

Opacity: this module is imported by the **waking** actor objective only
(``kind/agents/actor.py``). The dream/offline regime must never reach it —
enforced structurally by the import lint and behavioral backstop in
``tests/test_pragmatic_guards.py`` (F5: dreams are not for anything). The
actor never reads any energy quantity directly; the preference operates on
``decode_energy`` over imagined latents inside the imagination loss, and
PolicyView stays frozen at ``{h, z, self_prediction_error}``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

#: Fixed setpoint in the normalized [0, 1] energy space (frozen B0b).
SETPOINT: float = 0.6

#: Fixed band halfwidth, absolute (Amendment 02 B0a′ — band [0.45, 0.75]).
BAND_HALFWIDTH: float = 0.15

#: Saturation scale S — the structural bound on the per-step pragmatic value
#: ([SAT-1: 1.0 | pre], builder-confirmed 2026-06-11).
SATURATION: float = 1.0


@dataclass(frozen=True)
class EnergyPreferenceConfig:
    """Static parameters of the saturating Gaussian log-preference.

    ``precision`` is the only intended degree of freedom (the S1 sweep knob);
    ``setpoint`` / ``band_halfwidth`` / ``saturation`` default to the frozen /
    amended / confirmed constants and exist as fields so eval harnesses (e.g.
    a Phase-3 positive control narrowing the band) can construct variants
    without editing this module. The dataclass is frozen: the preference is
    stateless and fixed for the lifetime of a run — never learned, never
    self-optimized (F1/F2).
    """

    precision: float
    setpoint: float = SETPOINT
    band_halfwidth: float = BAND_HALFWIDTH
    saturation: float = SATURATION

    def __post_init__(self) -> None:
        if self.precision < 0.0:
            raise ValueError(f"precision must be ≥ 0, got {self.precision}")
        if self.band_halfwidth <= 0.0:
            raise ValueError(
                f"band_halfwidth must be > 0, got {self.band_halfwidth}"
            )
        if self.saturation <= 0.0:
            raise ValueError(f"saturation must be > 0, got {self.saturation}")


def energy_log_preference(energy: Tensor, config: EnergyPreferenceConfig) -> Tensor:
    """The saturating Gaussian log-preference, elementwise.

    Args:
        energy: decoded energy values, any shape (the actor passes the
            squeezed ``decode_energy`` output along imagined rollouts).
            Values outside [0, 1] are legitimate inputs — the decoder is an
            unconstrained MLP and the saturation bound is what keeps decoder
            extrapolation from manufacturing unbounded value.
        config: the fixed preference parameters.

    Returns:
        A tensor of ``energy``'s shape: 0 in-band, negative outside, bounded
        below by ``−config.saturation``. Differentiable; the gradient is 0
        in-band, points toward the band outside it, and vanishes only in the
        deeply saturated tail.
    """
    deviation = torch.relu((energy - config.setpoint).abs() - config.band_halfwidth)
    s = config.saturation
    return -s * torch.tanh(config.precision * deviation.pow(2) / (2.0 * s))


__all__ = [
    "BAND_HALFWIDTH",
    "SATURATION",
    "SETPOINT",
    "EnergyPreferenceConfig",
    "energy_log_preference",
]
