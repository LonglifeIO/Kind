"""Probe 3.5 Phase 2 — unit tests for the pragmatic preference function.

The mechanism gate's unit layer (Step 4a): correct pragmatic values on
synthetic energy values — in-band → flat with exactly zero gradient;
out-of-band → correctly signed pull toward the band; saturation bounds
respected; precision the only weight, with ``precision = 0`` identically
zero. The functional form's constants: setpoint 0.6 (frozen B0b), band
halfwidth 0.15 absolute (Amendment 02 B0a′ — band [0.45, 0.75]),
saturation S = 1.0 ([SAT-1 | pre], builder-confirmed 2026-06-11).
"""

from __future__ import annotations

import pytest
import torch

from kind.agents.preference import (
    BAND_HALFWIDTH,
    SATURATION,
    SETPOINT,
    EnergyPreferenceConfig,
    energy_log_preference,
)


def _cfg(precision: float = 2.0) -> EnergyPreferenceConfig:
    return EnergyPreferenceConfig(precision=precision)


def test_constants_match_the_frozen_and_confirmed_values() -> None:
    """Setpoint and band are the pre-registered fixed constants; S is the
    builder-confirmed saturation scale. Pin them so a silent edit trips."""
    assert SETPOINT == 0.6
    assert BAND_HALFWIDTH == 0.15
    assert SATURATION == 1.0
    cfg = _cfg()
    assert cfg.setpoint == SETPOINT
    assert cfg.band_halfwidth == BAND_HALFWIDTH
    assert cfg.saturation == SATURATION


def test_in_band_value_is_exactly_zero() -> None:
    """Flat in-band (Keramati drive-reduction): every energy strictly inside
    [0.45, 0.75] yields exactly 0 — no hoarding reward, no pull at the
    setpoint. At the exact band edges float32 representation leaves an
    O(1e-8) residual deviation, so the boundary is asserted to float
    precision rather than exact zero (a measure-zero rounding artifact,
    not a flat-band violation)."""
    energy = torch.linspace(0.451, 0.749, 31)
    values = energy_log_preference(energy, _cfg())
    assert torch.equal(values, torch.zeros_like(values))
    edges = energy_log_preference(torch.tensor([0.45, 0.75]), _cfg())
    assert (edges.abs() < 1e-12).all()


def test_in_band_gradient_is_exactly_zero() -> None:
    """The in-band flat is flat in gradient too — the actor receives no
    signal from in-band beliefs (the indifference plateau)."""
    energy = torch.tensor([0.46, 0.6, 0.74], requires_grad=True)
    energy_log_preference(energy, _cfg()).sum().backward()
    assert energy.grad is not None
    assert torch.equal(energy.grad, torch.zeros_like(energy))


def test_out_of_band_pull_is_correctly_signed() -> None:
    """Below the band the gradient of the value is positive (higher energy =
    higher value → pull up); above the band it is negative (pull down).
    Both pull *toward* the band."""
    below = torch.tensor([0.0, 0.2, 0.40], requires_grad=True)
    energy_log_preference(below, _cfg()).sum().backward()
    assert below.grad is not None
    assert (below.grad > 0).all(), f"below-band gradient {below.grad}"

    above = torch.tensor([0.80, 0.9, 1.0], requires_grad=True)
    energy_log_preference(above, _cfg()).sum().backward()
    assert above.grad is not None
    assert (above.grad < 0).all(), f"above-band gradient {above.grad}"


def test_out_of_band_value_is_negative_and_monotone_in_deviation() -> None:
    """Value strictly decreases as energy moves away from the band on either
    side — the pull is graded, not a cliff."""
    cfg = _cfg()
    below = energy_log_preference(torch.tensor([0.40, 0.30, 0.20, 0.10]), cfg)
    assert (below < 0).all()
    assert (below[1:] < below[:-1]).all()
    above = energy_log_preference(torch.tensor([0.80, 0.85, 0.90, 0.95]), cfg)
    assert (above < 0).all()
    assert (above[1:] < above[:-1]).all()


def test_saturation_bounds_respected() -> None:
    """|v| ≤ S everywhere — including far outside [0, 1], where an
    unconstrained decoder could extrapolate. The bound is approached, not
    crossed, at extreme deviation."""
    cfg = _cfg(precision=1000.0)
    extreme = torch.tensor([-10.0, -1.0, 0.0, 1.0, 2.0, 10.0])
    values = energy_log_preference(extreme, cfg)
    assert (values.abs() <= cfg.saturation).all()
    assert float(values[0]) == pytest.approx(-cfg.saturation, abs=1e-4)


def test_gaussian_log_preference_in_the_unsaturated_regime() -> None:
    """Just outside the band the form is the Gaussian log-preference in
    band-edge distance: v ≈ −(precision/2)·d². Precision is the inverse
    variance (T4) and the only weight (DP5)."""
    cfg = _cfg(precision=2.0)
    e = torch.tensor([0.35])  # d = 0.10
    expected = -(2.0 / 2.0) * 0.10**2
    assert float(energy_log_preference(e, cfg)) == pytest.approx(
        expected, rel=1e-3
    )


def test_precision_scales_magnitude() -> None:
    """Doubling precision (in the unsaturated regime) doubles the pull —
    precision is the dominance-relevant weight; there is no other."""
    e = torch.tensor([0.30])
    v1 = float(energy_log_preference(e, _cfg(precision=1.0)))
    v2 = float(energy_log_preference(e, _cfg(precision=2.0)))
    assert v2 == pytest.approx(2.0 * v1, rel=1e-2)


def test_precision_zero_is_identically_zero_with_zero_gradient() -> None:
    """The degenerate-null configuration: precision = 0 yields exactly zero
    value and exactly zero gradient everywhere — a point on the same
    surface, not a separate code path."""
    energy = torch.linspace(-1.0, 2.0, 61, requires_grad=True)
    values = energy_log_preference(energy, _cfg(precision=0.0))
    assert torch.equal(values, torch.zeros_like(values))
    values.sum().backward()
    assert energy.grad is not None
    assert torch.equal(energy.grad, torch.zeros_like(energy))


def test_elementwise_over_arbitrary_shapes() -> None:
    """The function is elementwise: batched / trailing-singleton shapes pass
    through unchanged (the actor squeezes ``decode_energy``'s (B, 1))."""
    cfg = _cfg()
    flat = energy_log_preference(torch.full((6,), 0.2), cfg)
    assert flat.shape == (6,)
    col = energy_log_preference(torch.full((6, 1), 0.2), cfg)
    assert col.shape == (6, 1)
    assert torch.equal(col.squeeze(-1), flat)


def test_config_validation_rejects_bad_constants() -> None:
    with pytest.raises(ValueError):
        EnergyPreferenceConfig(precision=-1.0)
    with pytest.raises(ValueError):
        EnergyPreferenceConfig(precision=1.0, band_halfwidth=0.0)
    with pytest.raises(ValueError):
        EnergyPreferenceConfig(precision=1.0, saturation=0.0)
