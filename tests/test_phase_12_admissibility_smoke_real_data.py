"""Phase 12 Part 5 — smoke test of the admissibility consumer against
real Phase 13 calibration data.

This is a *data-shapes* smoke, not an LLM smoke. It runs no LLM calls
and is part of the standard ``pytest`` suite (no ``--run-real-api``
flag). It exercises :func:`~kind.mirror.admissibility.compute_admissibility`
against whatever faithfulness and stability records exist on disk under
``runs/phase_13_calibration/mirror/`` so any mismatch between the
consumer's expected record shapes and the verifiers' actual on-disk
output surfaces here.

The test skips when neither record type is present — which is the
current state: Phase 13's calibration ran before Phases 10 and 11
existed, so it wrote neither ``faithfulness.jsonl`` nor
``stability.jsonl``. The skip-when-absent behavior keeps a fresh
checkout's ``pytest`` clean. It parallels Phase 11's data-shapes smoke
at :mod:`tests.test_phase_11_faithfulness_smoke_real_data`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kind.mirror.admissibility import (
    AdmissibilityBatchResult,
    compute_admissibility,
    load_admissibility_inputs,
)

_PHASE_13_RUN_DIR = (
    Path(__file__).resolve().parents[1] / "runs" / "phase_13_calibration"
)
_FAITHFULNESS_JSONL = _PHASE_13_RUN_DIR / "mirror" / "faithfulness.jsonl"
_STABILITY_JSONL = _PHASE_13_RUN_DIR / "mirror" / "stability.jsonl"


@pytest.mark.skipif(
    not _FAITHFULNESS_JSONL.exists() and not _STABILITY_JSONL.exists(),
    reason=(
        "Neither faithfulness.jsonl nor stability.jsonl is present under "
        "runs/phase_13_calibration/mirror/ — Phase 13 ran before Phases 10 "
        "and 11 existed. This data-shapes smoke requires the real verifier "
        "artifacts."
    ),
)
def test_compute_admissibility_runs_against_phase_13_records() -> None:
    faithfulness_results, stability_results = load_admissibility_inputs(
        "phase_13_calibration", _PHASE_13_RUN_DIR
    )
    batch = compute_admissibility(
        faithfulness_results=faithfulness_results,
        stability_results=stability_results,
        run_id="phase_13_calibration",
    )

    # Well-formedness: the batch is a frozen AdmissibilityBatchResult,
    # the counts partition n_readings_total, and the verdicts tuple
    # matches the count.
    assert isinstance(batch, AdmissibilityBatchResult)
    assert batch.n_readings_total == len(faithfulness_results)
    assert len(batch.verdicts) == batch.n_readings_total
    assert (
        batch.n_admissible
        + batch.n_inadmissible_faithfulness
        + batch.n_inadmissible_stability
        + batch.n_inadmissible_both
        + batch.n_inadmissible_no_stability
        == batch.n_readings_total
    )
    assert 0.0 <= batch.admissibility_rate <= 1.0

    # Every verdict carries a non-empty notes field and a per-surface
    # dict whose conjunction matches stability_admissible_all_surfaces.
    for verdict in batch.verdicts:
        assert verdict.notes.strip() != ""
        assert verdict.stability_admissible_all_surfaces == all(
            verdict.stability_admissible_per_surface.values()
        )
        assert verdict.admissible == (
            verdict.faithfulness_admissible
            and verdict.stability_admissible_all_surfaces
        )
