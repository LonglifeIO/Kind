"""Probe 4 Phase 1 — memory-horizon harness (plan Phase 1; synthesis T4).

The harness is observer-side and eval-only; these tests pin its
determinism, its self-checks (pulse visibility, stream purity, episode
fit), and the h-timing convention (the pulse first reaches ``h`` one
recurrence after the perturbed observation is consumed — curve index 0).
The *measured* horizon on the real frozen instance is a run artifact
journaled by the phase, not a test assertion — a test must not pin a
number that is a property of particular weights.
"""

from __future__ import annotations

import pytest
import torch

from kind.agents.world_model import WorldModel, WorldModelConfig
from kind.observer.memory_horizon import (
    MemoryHorizonConfig,
    measure_memory_horizon,
)


def _make_model(seed: int = 0) -> WorldModel:
    torch.manual_seed(seed)
    model = WorldModel(WorldModelConfig())
    model.eval()
    return model


def _small_config(**overrides: object) -> MemoryHorizonConfig:
    defaults: dict[str, object] = {
        "seed": 3,
        "injection_steps": (5,),
        "window_steps": 20,
    }
    defaults.update(overrides)
    return MemoryHorizonConfig(**defaults)  # type: ignore[arg-type]


def test_harness_is_deterministic() -> None:
    """Same model, same config → identical reports (the phase gate's
    horizon-harness determinism requirement)."""
    model = _make_model()
    report_a = measure_memory_horizon(model, _small_config())
    report_b = measure_memory_horizon(model, _small_config())
    assert report_a == report_b


def test_pulse_reaches_h_and_curves_have_window_length() -> None:
    model = _make_model()
    config = _small_config()
    report = measure_memory_horizon(model, config)
    assert len(report.measurements) == 1
    measurement = report.measurements[0]
    # Curve index 0 is the first h that can carry the pulse; the pulse
    # must actually reach it (a visible one-step observation difference
    # must displace h through the posterior→recurrence path).
    assert measurement.l2_curve[0] > 0.0
    assert measurement.kl_peak > 0.0
    assert measurement.l2_peak > 0.0
    # Curve spans the post-injection window (obs stream has
    # injection+window+1 entries; h-curve starts at injection+1).
    assert len(measurement.l2_curve) == config.window_steps
    assert len(measurement.kl_curve) == config.window_steps


def test_multiple_injection_contexts() -> None:
    model = _make_model()
    report = measure_memory_horizon(
        model, _small_config(injection_steps=(5, 10))
    )
    assert len(report.measurements) == 2
    assert report.measurements[0].context.injection_step == 5
    assert report.measurements[1].context.injection_step == 10


def test_out_of_view_injection_raises() -> None:
    """A pulse the agent cannot see is not a measurement — the harness
    refuses rather than reporting a vacuous zero-divergence horizon."""
    model = _make_model()
    config = _small_config(injection_cell=(7, 7))  # agent at (3,3), view 0..6
    with pytest.raises(ValueError, match="not visible"):
        measure_memory_horizon(model, config)


def test_window_must_fit_inside_one_episode() -> None:
    model = _make_model()
    config = _small_config(injection_steps=(150,), window_steps=120)
    with pytest.raises(ValueError, match="episode"):
        measure_memory_horizon(model, config)


def test_report_carries_credit_assignment_ceilings() -> None:
    """The BPTT window (replay_sequence_length) and imagination horizon
    ride the report — the synthesis's binding limits, mirrored from the
    RunnerConfig defaults."""
    model = _make_model()
    report = measure_memory_horizon(model, _small_config())
    assert report.replay_sequence_length == 32
    assert report.imagination_horizon == 15


def test_horizon_semantics_sustained_below_threshold() -> None:
    """If a horizon is reported (not censored), the curve is below
    threshold from that index through the end of the window."""
    model = _make_model()
    config = _small_config(window_steps=30)
    report = measure_memory_horizon(model, config)
    measurement = report.measurements[0]
    for curve, horizon, peak in (
        (measurement.kl_curve, measurement.horizon_kl_steps, measurement.kl_peak),
        (measurement.l2_curve, measurement.horizon_l2_steps, measurement.l2_peak),
    ):
        if horizon is None:
            continue
        threshold = report.decay_fraction * peak
        assert all(value < threshold for value in curve[horizon:])
        assert horizon == 0 or curve[horizon - 1] >= threshold
