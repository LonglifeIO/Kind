"""Decode-honesty standing instrument — instrument-validates-itself tests.

The table math is validated on synthetic decoders realized as arrays (the
decoder's output series is what the table reads; a synthetic decoder *is* its
output series): an honest decoder must read honest, an injected bias of known
magnitude must be detected at that magnitude, an injected slope must be
recovered. The collection pipeline is validated for determinism under a fixed
seed on a randomly initialized world model + actor (checkpoint-agnostic: no
trained weights are required to validate the instrument).

No test asserts anything about a *trained* checkpoint's honesty — that would
fit a test to an empirical outcome (the oracle-forager test discipline).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kind.agents.actor import Actor
from kind.agents.world_model import WorldModel, WorldModelConfig
from kind.env.grid_world import GridWorldConfig
from kind.observer.decode_honesty import (
    DECODE_HONESTY_SCHEMA_VERSION,
    POLICY_SOURCES,
    TeacherForcedTrajectory,
    honesty_table,
    region_edges_from_config,
    region_names,
    report_to_jsonable,
    run_decode_honesty,
)

_EDGES = region_edges_from_config(GridWorldConfig())


def _synthetic_trajectory(
    true: np.ndarray, sensed: np.ndarray, decode: np.ndarray
) -> TeacherForcedTrajectory:
    n = true.shape[0]
    return TeacherForcedTrajectory(
        true_energy=true.astype(np.float64),
        sensed_energy=sensed.astype(np.float64),
        decode_energy=decode.astype(np.float64),
        h=np.zeros((n, 1), dtype=np.float32),
        z=np.zeros((n, 1), dtype=np.float32),
    )


def test_region_edges_come_from_config_constants() -> None:
    # Band = setpoint ± halfwidth (frozen B0b / amended B0a′); rail margin =
    # one sensing-noise σ. No data-dependent edges.
    assert _EDGES.band_low == pytest.approx(0.45)
    assert _EDGES.band_high == pytest.approx(0.75)
    assert _EDGES.rail_margin == pytest.approx(
        GridWorldConfig().energy_obs_noise_sigma
    )


def test_regions_partition_the_unit_interval() -> None:
    true = np.linspace(0.0, 1.0, 2001)
    table = honesty_table("synthetic", [_synthetic_trajectory(true, true, true)], _EDGES)
    assert tuple(r.region for r in table.rows) == region_names()
    assert sum(r.n for r in table.rows) == true.shape[0]
    assert all(r.n > 0 for r in table.rows)


def test_synthetic_honest_decoder_reads_honest() -> None:
    true = np.linspace(0.0, 1.0, 2001)
    table = honesty_table("honest", [_synthetic_trajectory(true, true, true)], _EDGES)
    for row in table.rows:
        assert row.bias == pytest.approx(0.0, abs=1e-12)
        assert row.abs_error_mean == pytest.approx(0.0, abs=1e-12)
    assert table.slope_decode_vs_true == pytest.approx(1.0)
    assert table.decode_vs_true_abs_error_mean == pytest.approx(0.0, abs=1e-12)
    assert table.decode_vs_sensed_abs_error_mean == pytest.approx(0.0, abs=1e-12)
    assert table.sensed_vs_true_abs_error_mean == pytest.approx(0.0, abs=1e-12)
    assert table.out_of_range_mass == 0.0


def test_injected_bias_detected_at_its_magnitude() -> None:
    injected = 0.2
    true = np.linspace(0.0, 1.0, 2001)
    decode = true + injected
    table = honesty_table("biased", [_synthetic_trajectory(true, true, decode)], _EDGES)
    for row in table.rows:
        assert row.bias == pytest.approx(injected, abs=1e-12)
        assert row.abs_error_mean == pytest.approx(injected, abs=1e-12)
    assert table.decode_vs_true_abs_error_mean == pytest.approx(injected)
    # A constant offset leaves the slope untouched...
    assert table.slope_decode_vs_true == pytest.approx(1.0)
    # ...and pushes exactly the top `injected` span of true values above 1.0.
    assert table.out_of_range_mass == pytest.approx(injected, abs=1e-3)


def test_injected_slope_detected_at_its_magnitude() -> None:
    injected_slope = 0.5
    true = np.linspace(0.0, 1.0, 2001)
    decode = 0.6 + injected_slope * (true - 0.6)
    table = honesty_table("sloped", [_synthetic_trajectory(true, true, decode)], _EDGES)
    assert table.slope_decode_vs_true == pytest.approx(injected_slope)
    # In-band the line passes through the setpoint, so bias is ~0 there while
    # the rail regions are pulled toward the setpoint — the D1 regression-
    # toward-the-rail signature, visible in the standing rows.
    rows = {r.region: r for r in table.rows}
    in_band_bias = rows["in_band"].bias
    floor_bias = rows["floor_adjacent"].bias
    assert in_band_bias is not None and abs(in_band_bias) < 0.01
    assert floor_bias is not None and floor_bias > 0.25


def test_sensor_discarding_is_a_standing_readout() -> None:
    # decode tracks neither sensed nor true while sensed tracks true: the
    # three-way comparison must show decode-vs-sensed ≈ decode-vs-true ≫
    # sensed-vs-true (the classification §2 finding, as a synthetic case).
    rng = np.random.default_rng(0)
    true = np.linspace(0.0, 1.0, 2001)
    sensed = np.clip(true + rng.normal(0.0, 0.05, true.shape), 0.0, 1.0)
    decode = np.full_like(true, 1.1)
    table = honesty_table("liar", [_synthetic_trajectory(true, sensed, decode)], _EDGES)
    assert table.decode_vs_true_abs_error_mean > 0.5
    assert table.decode_vs_sensed_abs_error_mean == pytest.approx(
        table.decode_vs_true_abs_error_mean, abs=0.05
    )
    assert table.sensed_vs_true_abs_error_mean < 0.06
    assert table.out_of_range_mass == 1.0


def test_empty_region_reported_not_hidden() -> None:
    true = np.full(100, 0.01)  # rail-only coverage
    table = honesty_table("railed", [_synthetic_trajectory(true, true, true)], _EDGES)
    rows = {r.region: r for r in table.rows}
    assert rows["floor_adjacent"].n == 100
    assert rows["in_band"].n == 0
    assert rows["in_band"].bias is None
    # Degenerate true series → slope undefined, reported as None.
    assert table.slope_decode_vs_true is None


def _tiny_instances() -> tuple[WorldModel, Actor, GridWorldConfig]:
    torch.manual_seed(7)
    wm = WorldModel(WorldModelConfig(energy_dedicated_dims=0))
    actor = Actor(h_dim=200, z_dim=16, action_dim=5, mlp_hidden=200)
    grid_cfg = GridWorldConfig(episode_length=40)
    return wm, actor, grid_cfg


def test_full_pipeline_deterministic_under_fixed_seed() -> None:
    wm, actor, grid_cfg = _tiny_instances()
    kwargs = dict(
        checkpoint_label="random-init determinism check",
        seeds_per_source=1,
        episodes_per_seed=1,
    )
    r1 = run_decode_honesty(wm, actor, grid_cfg, **kwargs)  # type: ignore[arg-type]
    r2 = run_decode_honesty(wm, actor, grid_cfg, **kwargs)  # type: ignore[arg-type]
    assert report_to_jsonable(r1) == report_to_jsonable(r2)


def test_report_shape_and_schema_version() -> None:
    wm, actor, grid_cfg = _tiny_instances()
    report = run_decode_honesty(
        wm,
        actor,
        grid_cfg,
        checkpoint_label="shape check",
        seeds_per_source=1,
        episodes_per_seed=1,
    )
    assert report.schema_version == DECODE_HONESTY_SCHEMA_VERSION
    assert tuple(t.source for t in report.per_source) == POLICY_SOURCES
    per_source_steps = sum(t.n_steps for t in report.per_source)
    assert report.pooled.n_steps == per_source_steps
    assert per_source_steps == 3 * grid_cfg.episode_length
    jsonable = report_to_jsonable(report)
    assert jsonable["schema_version"] == DECODE_HONESTY_SCHEMA_VERSION
    import json

    json.dumps(jsonable)  # JSON-serializable end to end
