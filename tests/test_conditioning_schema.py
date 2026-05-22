"""Probe 2 Phase 0 schema test for ``ConditioningResult`` / ``RegimeStats``.

Phase 0's scope: the schemas exist, round-trip through JSONL, validate on
field types. Full ``compute_conditioning`` analysis lands in Phase 6;
this test exercises only the data-shape contract Phase 6 must honor.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from kind.observer.conditioning import (
    CONDITIONING_RESULT_SCHEMA_VERSION,
    ConditioningResult,
    RegimeBucket,
    RegimeStats,
    compute_conditioning,
)


def _stats(**overrides: Any) -> RegimeStats:
    base: dict[str, Any] = {
        "n_states": 12,
        "kl_mean": 2.1e-7,
        "kl_std": 8.4e-8,
        "kl_p50": 1.9e-7,
        "kl_p90": 4.3e-7,
    }
    base.update(overrides)
    return RegimeStats(**base)


def _bucket(
    *,
    regime: str = "high_disagreement",
    perturbation: str = "gaussian",
    stats: RegimeStats | None = None,
) -> RegimeBucket:
    return RegimeBucket(
        regime=regime,  # type: ignore[arg-type]
        perturbation=perturbation,  # type: ignore[arg-type]
        stats=stats if stats is not None else _stats(),
    )


def _result(**overrides: Any) -> ConditioningResult:
    base: dict[str, Any] = {
        "run_id": "probe1_5_phase7_5-20260507-101800",
        "checkpoint_id": "ckpt-000001",
        "timestamp_ms": 1_730_000_000_000,
        "n_states_sampled": 200,
        "perturbation_distributions": ["gaussian", "zero"],
        "regimes": [
            "perturbation_window",
            "high_disagreement",
            "high_kl",
            "steady_state",
        ],
        "empirical_scalar_mean": 0.0091,
        "empirical_scalar_sigma": 0.012,
        "empirical_scalar_range": (0.0, 0.082),
        "per_regime_per_perturbation": [
            _bucket(regime="high_disagreement", perturbation="gaussian"),
            _bucket(
                regime="steady_state",
                perturbation="gaussian",
                stats=_stats(n_states=180, kl_mean=4.2e-6, kl_p90=8.1e-6),
            ),
        ],
        "masked_steps_excluded": 5,
    }
    base.update(overrides)
    return ConditioningResult(**base)


# ---- (1) RegimeStats round-trip -----------------------------------------


def test_regime_stats_jsonl_round_trip_byte_stable() -> None:
    original = _stats()
    line_a = original.model_dump_json()
    parsed = RegimeStats.model_validate_json(line_a)
    line_b = parsed.model_dump_json()
    assert line_a == line_b
    assert parsed == original


def test_regime_stats_extra_field_rejected() -> None:
    payload: dict[str, Any] = _stats().model_dump()
    payload["unexpected"] = 0.0
    with pytest.raises(ValidationError):
        RegimeStats.model_validate(payload)


# ---- (2) RegimeBucket round-trip ----------------------------------------


def test_regime_bucket_jsonl_round_trip_byte_stable() -> None:
    original = _bucket()
    line_a = original.model_dump_json()
    parsed = RegimeBucket.model_validate_json(line_a)
    line_b = parsed.model_dump_json()
    assert line_a == line_b
    assert parsed == original


@pytest.mark.parametrize(
    "regime",
    ["perturbation_window", "high_disagreement", "high_kl", "steady_state"],
)
def test_regime_bucket_accepts_all_four_regimes(regime: str) -> None:
    bucket = _bucket(regime=regime)
    assert bucket.regime == regime


@pytest.mark.parametrize("perturbation", ["gaussian", "zero", "uniform"])
def test_regime_bucket_accepts_all_three_perturbations(perturbation: str) -> None:
    bucket = _bucket(perturbation=perturbation)
    assert bucket.perturbation == perturbation


def test_regime_bucket_rejects_invalid_regime() -> None:
    with pytest.raises(ValidationError):
        _bucket(regime="not_a_regime")


def test_regime_bucket_rejects_invalid_perturbation() -> None:
    with pytest.raises(ValidationError):
        _bucket(perturbation="not_a_perturbation")


# ---- (3) ConditioningResult round-trip ----------------------------------


def test_conditioning_result_jsonl_round_trip_byte_stable() -> None:
    original = _result()
    line_a = original.model_dump_json()
    parsed = ConditioningResult.model_validate_json(line_a)
    line_b = parsed.model_dump_json()
    assert line_a == line_b
    assert parsed == original


def test_conditioning_result_default_schema_version() -> None:
    result = _result()
    assert result.schema_version == CONDITIONING_RESULT_SCHEMA_VERSION
    assert CONDITIONING_RESULT_SCHEMA_VERSION == "0.1.0"


def test_conditioning_result_empty_buckets_allowed() -> None:
    """A run that had no qualifying states (e.g., Probe 1 records that
    did not include the self-prediction fields) produces a result with
    an empty bucket list — the graceful-degraded path."""
    result = _result(n_states_sampled=0, per_regime_per_perturbation=[])
    assert result.per_regime_per_perturbation == []


def test_conditioning_result_extra_field_rejected() -> None:
    payload: dict[str, Any] = _result().model_dump()
    payload["unexpected_field"] = "x"
    with pytest.raises(ValidationError):
        ConditioningResult.model_validate(payload)


def test_conditioning_result_is_frozen() -> None:
    result = _result()
    with pytest.raises(ValidationError):
        result.run_id = "different"  # type: ignore[misc]


# ---- (4) field-level validation -----------------------------------------


def test_perturbation_distributions_field_rejects_invalid_label() -> None:
    with pytest.raises(ValidationError):
        _result(perturbation_distributions=["gaussian", "imaginary"])


def test_regimes_field_rejects_invalid_label() -> None:
    with pytest.raises(ValidationError):
        _result(regimes=["perturbation_window", "imaginary_regime"])


def test_empirical_scalar_range_must_be_two_floats() -> None:
    with pytest.raises(ValidationError):
        _result(empirical_scalar_range=(0.0, 0.1, 0.2))  # type: ignore[arg-type]


# ---- (5) compute_conditioning stub raises NotImplementedError ----------


def test_compute_conditioning_phase_0_stub_raises(tmp_path: Any) -> None:
    """Phase 0 ships the schemas plus the function signature; the body
    raises NotImplementedError. Phase 6 will populate it."""
    with pytest.raises(NotImplementedError) as exc_info:
        compute_conditioning(tmp_path)
    assert "Phase 6" in str(exc_info.value)
