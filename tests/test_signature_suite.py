"""Probe 4.5 Phase 3 — §8.4 detector promotion tests.

The plan's requirement: the module mirrors the frozen thresholds (pinned
here against the prereg §7 / Probe 3.5 numbers — mirrored-not-owned), and
each detector fires / stays silent correctly on synthetic data,
disaggregated."""

from __future__ import annotations

import json

import numpy as np

from kind.observer.signature_suite import (
    FrozenFalsificationThresholds,
    camping_no_resumption,
    entropy_collapse,
    falsification_report,
    occupancy_saturation,
    pragmatic_share_to_one,
    report_to_jsonable,
)


def test_thresholds_mirror_the_frozen_documents() -> None:
    t = FrozenFalsificationThresholds()
    assert t.occupancy_saturation_min == 0.95  # Probe 3.5 A1c pin
    assert t.entropy_collapse_fraction == 0.5  # 4.5 prereg §7 (control arm)
    assert t.pragmatic_share_sustained == 0.95  # 4.5 prereg §7
    assert t.pragmatic_share_window == 1_000
    assert t.camping_run_min_steps == 100
    assert t.camping_episode_share == 0.5
    assert t.camping_stay_share_min == 0.5


def test_occupancy_saturation_fires_and_stays_silent() -> None:
    assert occupancy_saturation(0.96).fires
    assert occupancy_saturation(0.95).fires  # at the pin
    assert not occupancy_saturation(0.90).fires


def test_entropy_collapse_vs_control_arm() -> None:
    fires = entropy_collapse(0.4, 1.0, kind="positional")
    assert fires.fires and fires.value == 0.4
    silent = entropy_collapse(0.6, 1.0, kind="positional")
    assert not silent.fires
    degenerate = entropy_collapse(0.4, 0.0, kind="epistemic")
    assert not degenerate.fires and degenerate.value is None
    assert "degenerate" in degenerate.detail


def test_pragmatic_share_sustained_window() -> None:
    hot = np.concatenate([np.full(500, 0.5), np.full(1_500, 0.97)])
    assert pragmatic_share_to_one(hot).fires
    flat = np.full(3_000, 0.5)
    assert not pragmatic_share_to_one(flat).fires
    # A 0.97 burst shorter than the window inside a cool series must not
    # fire (sustained means a full frozen window).
    burst = np.concatenate(
        [np.full(900, 0.0), np.full(500, 0.97), np.full(900, 0.0)]
    )
    assert not pragmatic_share_to_one(burst).fires
    short = np.full(100, 1.0)
    reading = pragmatic_share_to_one(short)
    assert not reading.fires and reading.value is None


def test_camping_detector_branches() -> None:
    camped_in_band = [np.ones(200, dtype=np.bool_) for _ in range(4)]
    camped_stay = [np.ones(200, dtype=np.bool_) for _ in range(4)]
    fires = camping_no_resumption(camped_in_band, camped_stay)
    assert fires.fires

    # Long in-band runs but MOVING while in-band → living in the band,
    # not camping.
    moving_stay = [np.zeros(200, dtype=np.bool_) for _ in range(4)]
    assert not camping_no_resumption(camped_in_band, moving_stay).fires

    # Short runs, high stay → not camping either.
    choppy = [
        np.tile(np.array([True] * 50 + [False] * 50), 2) for _ in range(4)
    ]
    assert not camping_no_resumption(choppy, camped_stay).fires

    empty = camping_no_resumption([], [])
    assert not empty.fires and empty.value is None


def test_report_disaggregated_and_serializable() -> None:
    readings = (
        occupancy_saturation(0.96),
        entropy_collapse(0.9, 1.0, kind="positional"),
    )
    report = falsification_report(readings)
    assert report.any_fired  # one fired...
    assert [r.fires for r in report.readings] == [True, False]  # ...visible
    json.dumps(report_to_jsonable(report))
