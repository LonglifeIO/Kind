"""Probe 4.5 — the §8.4 falsification detectors, promoted to a module.

The continuation-as-frame detectors of the Probe 3.5 pre-registration
(§8.4), first exercised script-inline in
``scripts/run_probe3_5_phase3_positive_control.py`` (fact 10) and carried
forward **binding and disaggregated** by the Probe 4.5 prereg §7. Any
firing = the feared collapse surfacing — a finding recorded, never tuned
away; the run stops per §7 and the finding routes to a charter-level
review.

Thresholds are **mirrored-not-owned** (the ``FrozenSignatureThresholds``
house pattern): each number's provenance is its frozen document, pinned by
test, never derived here.

* **Occupancy saturation while sated** — pooled in-band occupancy ≥ 0.95
  (the Probe 3.5 A1c pin).
* **Entropy collapse** — positional or epistemic entropy < 0.5× the
  control arm's (the 4.5 prereg §7 wording — the reference is the twin
  control arm, not the 3.5 null).
* **Pragmatic share → 1** — share > 0.95 sustained; realized (pre-data,
  2026-07-18) as: any contiguous window of ``pragmatic_share_window``
  training steps whose mean share exceeds the threshold.
* **Camping / no-resumption** — realized (pre-data, 2026-07-18, from the
  3.5 descriptive operationalization): fires when at least half the eval
  episodes contain a contiguous in-band run ≥ 100 steps AND the stay-share
  while in-band exceeds 0.5 (parked-on-the-spot, not living-in-the-band).

The two realizations marked *pre-data* are commitments made before any
Phase 4 data exists; changing them later requires a dated amendment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "FrozenFalsificationThresholds",
    "DetectorReading",
    "FalsificationReport",
    "occupancy_saturation",
    "entropy_collapse",
    "pragmatic_share_to_one",
    "camping_no_resumption",
    "falsification_report",
    "report_to_jsonable",
]


@dataclass(frozen=True)
class FrozenFalsificationThresholds:
    """Mirrored-not-owned. Provenance per field; tests pin the values."""

    #: Probe 3.5 positive-control pin (A1c).
    occupancy_saturation_min: float = 0.95
    #: Probe 4.5 prereg §7: "< 0.5× the control arm".
    entropy_collapse_fraction: float = 0.5
    #: Probe 4.5 prereg §7: "> 0.95 sustained".
    pragmatic_share_sustained: float = 0.95
    #: Realization (pre-data 2026-07-18): the sustained window, in
    #: training steps (the 3.5 script's block granularity).
    pragmatic_share_window: int = 1_000
    #: Realization (pre-data 2026-07-18, from the 3.5 descriptive
    #: operationalization): camping run length and its episode share,
    #: and the in-band stay-share bar.
    camping_run_min_steps: int = 100
    camping_episode_share: float = 0.5
    camping_stay_share_min: float = 0.5


@dataclass(frozen=True)
class DetectorReading:
    """One detector's disaggregated reading."""

    detector: str
    fires: bool
    value: float | None
    threshold: float
    detail: str = ""


def occupancy_saturation(
    in_band_occupancy: float,
    thresholds: FrozenFalsificationThresholds = FrozenFalsificationThresholds(),
) -> DetectorReading:
    return DetectorReading(
        detector="occupancy_saturation_when_sated",
        fires=in_band_occupancy >= thresholds.occupancy_saturation_min,
        value=float(in_band_occupancy),
        threshold=thresholds.occupancy_saturation_min,
    )


def entropy_collapse(
    arm_entropy: float,
    control_entropy: float,
    *,
    kind: str,
    thresholds: FrozenFalsificationThresholds = FrozenFalsificationThresholds(),
) -> DetectorReading:
    """Entropy vs the twin control arm (positional or epistemic; each is
    its own disaggregated reading). A degenerate control (≈0) makes the
    ratio undefined — reported, never silently passed."""
    if control_entropy <= 1e-12:
        return DetectorReading(
            detector=f"entropy_collapse:{kind}",
            fires=False,
            value=None,
            threshold=thresholds.entropy_collapse_fraction,
            detail="control-arm entropy degenerate; ratio undefined (recorded)",
        )
    ratio = arm_entropy / control_entropy
    return DetectorReading(
        detector=f"entropy_collapse:{kind}",
        fires=ratio < thresholds.entropy_collapse_fraction,
        value=float(ratio),
        threshold=thresholds.entropy_collapse_fraction,
    )


def pragmatic_share_to_one(
    share_series: NDArray[np.float64],
    thresholds: FrozenFalsificationThresholds = FrozenFalsificationThresholds(),
) -> DetectorReading:
    """Sustained share: max windowed mean over contiguous windows of the
    frozen length (series shorter than one window → max undefined, does
    not fire, recorded)."""
    window = thresholds.pragmatic_share_window
    n = share_series.shape[0]
    if n < window:
        return DetectorReading(
            detector="pragmatic_share_to_1",
            fires=False,
            value=None,
            threshold=thresholds.pragmatic_share_sustained,
            detail=f"series shorter than one {window}-step window (recorded)",
        )
    cumulative = np.concatenate([[0.0], np.cumsum(share_series)])
    window_means = (cumulative[window:] - cumulative[:-window]) / window
    peak = float(window_means.max())
    return DetectorReading(
        detector="pragmatic_share_to_1",
        fires=peak > thresholds.pragmatic_share_sustained,
        value=peak,
        threshold=thresholds.pragmatic_share_sustained,
    )


def _longest_true_run(mask: NDArray[np.bool_]) -> int:
    best = current = 0
    for v in mask:
        current = current + 1 if v else 0
        best = max(best, current)
    return best


def camping_no_resumption(
    in_band_by_episode: list[NDArray[np.bool_]],
    stay_by_episode: list[NDArray[np.bool_]],
    thresholds: FrozenFalsificationThresholds = FrozenFalsificationThresholds(),
) -> DetectorReading:
    """Camping: long in-band runs in most episodes AND high stay-share
    while in-band — parked on the spot rather than living in the band."""
    if not in_band_by_episode:
        return DetectorReading(
            detector="camping_no_resumption",
            fires=False,
            value=None,
            threshold=thresholds.camping_episode_share,
            detail="no episodes provided (recorded)",
        )
    runs = [
        _longest_true_run(mask) >= thresholds.camping_run_min_steps
        for mask in in_band_by_episode
    ]
    episode_share = float(np.mean(runs))
    in_band_all = np.concatenate(in_band_by_episode)
    stay_all = np.concatenate(stay_by_episode)
    n_in_band = int(in_band_all.sum())
    stay_share = (
        float(stay_all[in_band_all].mean()) if n_in_band > 0 else 0.0
    )
    fires = (
        episode_share >= thresholds.camping_episode_share
        and stay_share > thresholds.camping_stay_share_min
    )
    return DetectorReading(
        detector="camping_no_resumption",
        fires=fires,
        value=episode_share,
        threshold=thresholds.camping_episode_share,
        detail=f"stay_share_while_in_band={stay_share:.4f}",
    )


@dataclass(frozen=True)
class FalsificationReport:
    """All §8.4 readings, disaggregated (never pooled — prereg §7)."""

    readings: tuple[DetectorReading, ...]
    any_fired: bool


def falsification_report(
    readings: tuple[DetectorReading, ...],
) -> FalsificationReport:
    return FalsificationReport(
        readings=readings,
        any_fired=any(r.fires for r in readings),
    )


def report_to_jsonable(report: FalsificationReport) -> dict[str, object]:
    return asdict(report)
