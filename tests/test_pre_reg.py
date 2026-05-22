"""Probe 2 Phase 0 gate test 4 — pre-registration sink with v2 fields.

Per the Probe 2 implementation plan §4 row 4: ``PreRegSink`` writes/reads
cleanly; ``PreRegistration`` validates required v2 fields
(``reading_surfaces_per_criterion``, ``asymmetry_of_access``,
``expected_outcome_per_surface``, ``column_init``,
``new_actor_readable_interfaces_added``); the ``column_init`` validator
accepts ``"zero"`` / ``"small_gaussian"`` / ``"unknown"`` and rejects
others; missing falsifiers for an active criterion raises;
``new_actor_readable_interfaces_added`` accepts an empty list (the
typical Probe 2 case).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from kind.mirror.registry import ReadingSurface
from kind.observer.pre_reg import (
    PRE_REG_FILE,
    PRE_REG_SCHEMA_VERSION,
    PreRegistration,
    PreRegSink,
    PreRegSinkClosedError,
)


def _full_record(**overrides: Any) -> PreRegistration:
    """Build a complete v2 PreRegistration with all required fields
    populated. Overrides replace specific fields for negative tests."""
    base: dict[str, Any] = {
        "run_id": "probe2-round1-20260508",
        "timestamp_ms": 1_730_000_000_000,
        "criteria_active": [
            "equanimity_perturbation_recovery",
            "head_internal_sp_err_distribution",
        ],
        "criteria_held_out": ["behavior_side_scalar_conditioning"],
        "signal_mappings": {
            "equanimity_perturbation_recovery": [
                "kl_aggregate_t",
                "ensemble_disagreement",
                "self_prediction_error_t",
            ],
            "head_internal_sp_err_distribution": [
                "self_prediction_error_t",
                "self_prediction_t per-dim",
            ],
        },
        "falsifiers": {
            "equanimity_perturbation_recovery": (
                "if recovery shape is oscillatory or absent within "
                "20 steps post-perturbation"
            ),
            "head_internal_sp_err_distribution": (
                "if sp_err distribution is indistinguishable from a "
                "frozen-target run at KS-D < 0.10"
            ),
        },
        "scalar_checks": {
            "equanimity_perturbation_recovery": ["kl_aggregate_t", "sp_err mean"],
            "head_internal_sp_err_distribution": ["sp_err KS-D"],
        },
        "reading_surfaces_per_criterion": {
            "equanimity_perturbation_recovery": [
                ReadingSurface.SUBSTRATE_SIDE,
                ReadingSurface.HEAD_INTERNAL,
            ],
            "head_internal_sp_err_distribution": [ReadingSurface.HEAD_INTERNAL],
        },
        "asymmetry_of_access": (
            "Io reads scalar self_prediction_error_t on PolicyView; the "
            "mirror reads the full self_prediction_t vector, per-dim "
            "allocation, perturbation-recovery dynamics, behavioral "
            "conditioning, masked flag, longitudinal cross-run analysis."
        ),
        "builder_mode": "skeptic",
        "expected_outcome": (
            "first round expects equanimity at substrate-side and "
            "head-internal to admit at moderate strength; held-out "
            "behavior-side-conditioning criterion to be neutral until "
            "introduced at the late checkpoint."
        ),
        "expected_outcome_per_surface": {
            ReadingSurface.SUBSTRATE_SIDE: "equanimity recovery shape admits weakly",
            ReadingSurface.HEAD_INTERNAL: (
                "sp_err recovery shape admits at moderate strength"
            ),
            ReadingSurface.BEHAVIOR_SIDE: (
                "no admission this round (held-out criterion)"
            ),
        },
        "substrate_decisions_off_table": [
            "RSSM lineage choice",
            "PolicyView field set",
            "K=5 ensemble cardinality",
        ],
        "column_init": "small_gaussian",
        "new_actor_readable_interfaces_added": [],
    }
    base.update(overrides)
    return PreRegistration(**base)


# ---- (1) all v2 required fields populate cleanly -------------------------


def test_pre_registration_round_trips_with_all_v2_fields(tmp_path: Path) -> None:
    """Write a fully-populated v2 PreRegistration through the sink; read
    the JSONL back; validate it; equality holds."""
    record = _full_record()

    with PreRegSink(tmp_path) as sink:
        sink.write(record)

    path = tmp_path / PRE_REG_FILE
    assert path.exists()

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = PreRegistration.model_validate_json(lines[0])
    assert parsed == record
    assert parsed.schema_version == PRE_REG_SCHEMA_VERSION
    assert PRE_REG_SCHEMA_VERSION == "0.2.0"


def test_pre_registration_default_schema_version() -> None:
    record = _full_record()
    assert record.schema_version == PRE_REG_SCHEMA_VERSION


# ---- (2) PreRegSink append-only behavior --------------------------------


def test_pre_reg_sink_appends_multiple_records(tmp_path: Path) -> None:
    sink = PreRegSink(tmp_path)
    r1 = _full_record(timestamp_ms=1_730_000_000_001)
    r2 = _full_record(
        timestamp_ms=1_730_000_000_002,
        criteria_active=["self_prediction_quadruplet"],
        criteria_held_out=[],
        signal_mappings={"self_prediction_quadruplet": ["sp_err", "kl_per_dim_t"]},
        falsifiers={"self_prediction_quadruplet": "if no spike-recovery shape"},
        scalar_checks={"self_prediction_quadruplet": ["sp_err"]},
        reading_surfaces_per_criterion={
            "self_prediction_quadruplet": [
                ReadingSurface.SUBSTRATE_SIDE,
                ReadingSurface.HEAD_INTERNAL,
            ]
        },
    )
    sink.write(r1)
    sink.write(r2)
    sink.close()

    lines = (tmp_path / PRE_REG_FILE).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    parsed1 = PreRegistration.model_validate_json(lines[0])
    parsed2 = PreRegistration.model_validate_json(lines[1])
    assert parsed1 == r1
    assert parsed2 == r2


def test_pre_reg_sink_raises_after_close(tmp_path: Path) -> None:
    sink = PreRegSink(tmp_path)
    sink.close()
    with pytest.raises(PreRegSinkClosedError):
        sink.write(_full_record())


def test_pre_reg_sink_close_is_idempotent(tmp_path: Path) -> None:
    sink = PreRegSink(tmp_path)
    sink.close()
    sink.close()  # second call is a no-op, not a raise


def test_pre_reg_sink_creates_parent_directory(tmp_path: Path) -> None:
    """The sink takes a directory and creates it (and its parents) if
    missing — the runner's ``runs/{run_id}/pre_reg/`` is created on first
    write."""
    target = tmp_path / "runs" / "probe2-round1" / "pre_reg"
    assert not target.exists()
    with PreRegSink(target) as sink:
        sink.write(_full_record())
    assert (target / PRE_REG_FILE).exists()


# ---- (3) column_init validator ------------------------------------------


@pytest.mark.parametrize("init", ["zero", "small_gaussian", "unknown"])
def test_column_init_accepts_three_values(init: str) -> None:
    record = _full_record(column_init=init)
    assert record.column_init == init


@pytest.mark.parametrize(
    "bad", ["xavier", "kaiming", "small-gaussian", "Zero", "", "random"]
)
def test_column_init_rejects_other_values(bad: str) -> None:
    with pytest.raises(ValidationError):
        _full_record(column_init=bad)


# ---- (4) missing falsifiers / signal_mappings raises --------------------


def test_missing_falsifier_for_active_criterion_raises() -> None:
    """A criterion in criteria_active without a falsifier entry fails
    validation. Synthesis §2.4 element 1: every active criterion carries
    a falsifier before the reading runs."""
    with pytest.raises(ValidationError) as exc_info:
        _full_record(
            falsifiers={
                "equanimity_perturbation_recovery": "...",
                # missing entry for "head_internal_sp_err_distribution"
            }
        )
    assert "head_internal_sp_err_distribution" in str(exc_info.value)
    assert "falsifiers" in str(exc_info.value)


def test_missing_signal_mapping_for_active_criterion_raises() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _full_record(
            signal_mappings={"equanimity_perturbation_recovery": ["kl_aggregate_t"]},
        )
    assert "head_internal_sp_err_distribution" in str(exc_info.value)
    assert "signal_mappings" in str(exc_info.value)


def test_missing_scalar_checks_for_active_criterion_raises() -> None:
    with pytest.raises(ValidationError):
        _full_record(scalar_checks={"equanimity_perturbation_recovery": ["x"]})


def test_missing_reading_surfaces_for_active_criterion_raises() -> None:
    with pytest.raises(ValidationError):
        _full_record(
            reading_surfaces_per_criterion={
                "equanimity_perturbation_recovery": [ReadingSurface.SUBSTRATE_SIDE]
            }
        )


def test_active_held_out_overlap_raises() -> None:
    """A criterion cannot be both active and held-out at the same round."""
    with pytest.raises(ValidationError) as exc_info:
        _full_record(
            criteria_active=["equanimity_perturbation_recovery"],
            criteria_held_out=["equanimity_perturbation_recovery"],
            signal_mappings={
                "equanimity_perturbation_recovery": ["kl_aggregate_t"],
            },
            falsifiers={"equanimity_perturbation_recovery": "..."},
            scalar_checks={"equanimity_perturbation_recovery": ["..."]},
            reading_surfaces_per_criterion={
                "equanimity_perturbation_recovery": [ReadingSurface.SUBSTRATE_SIDE]
            },
        )
    assert "overlap" in str(exc_info.value).lower()


# ---- (5) new_actor_readable_interfaces_added field shape ----------------


def test_new_actor_readable_interfaces_added_accepts_empty_list() -> None:
    """Empty list is the typical Probe 2 case — Probe 2 adds no new
    actor-readable interfaces; the field is the structural hook for
    Probes 3 and beyond."""
    record = _full_record(new_actor_readable_interfaces_added=[])
    assert record.new_actor_readable_interfaces_added == []


def test_new_actor_readable_interfaces_added_accepts_strings() -> None:
    """When a future probe does add an interface, each entry is free-text
    describing it. The four-part Probe 1.5 v2 §2(b) discipline lives in
    the prose journal entry; the schema field is the structural carrier."""
    record = _full_record(
        new_actor_readable_interfaces_added=[
            "PolicyView gains scalar dream_engagement (Probe 3 hypothesis; affordance: dream-state head training)"
        ]
    )
    assert len(record.new_actor_readable_interfaces_added) == 1


# ---- (6) empty criteria_active is allowed -------------------------------


def test_empty_criteria_active_is_allowed() -> None:
    """A round with no active criteria is structurally valid — for
    instance, a baseline-only smoke or a calibration round purely on
    shuffled telemetry. The validator only requires per-criterion
    completeness on what IS active."""
    record = _full_record(
        criteria_active=[],
        signal_mappings={},
        falsifiers={},
        scalar_checks={},
        reading_surfaces_per_criterion={},
    )
    assert record.criteria_active == []


# ---- (7) frozen + extra=forbid disciplines -------------------------------


def test_pre_registration_extra_field_rejected() -> None:
    payload: dict[str, Any] = _full_record().model_dump()
    payload["unexpected_field"] = "x"
    with pytest.raises(ValidationError):
        PreRegistration.model_validate(payload)


def test_pre_registration_is_frozen() -> None:
    record = _full_record()
    with pytest.raises(ValidationError):
        record.run_id = "different"


# ---- (8) builder_mode validator -----------------------------------------


@pytest.mark.parametrize("mode", ["proponent", "skeptic"])
def test_builder_mode_accepts_two_values(mode: str) -> None:
    record = _full_record(builder_mode=mode)
    assert record.builder_mode == mode


def test_builder_mode_rejects_other_values() -> None:
    with pytest.raises(ValidationError):
        _full_record(builder_mode="neutral")


# ---- (9) ReadingSurface enum in expected_outcome_per_surface ------------


def test_expected_outcome_per_surface_accepts_three_surface_keys() -> None:
    record = _full_record(
        expected_outcome_per_surface={
            ReadingSurface.SUBSTRATE_SIDE: "weak admit",
            ReadingSurface.HEAD_INTERNAL: "moderate admit",
            ReadingSurface.BEHAVIOR_SIDE: "no admit (held out)",
        }
    )
    assert set(record.expected_outcome_per_surface) == {
        ReadingSurface.SUBSTRATE_SIDE,
        ReadingSurface.HEAD_INTERNAL,
        ReadingSurface.BEHAVIOR_SIDE,
    }
    # Every key is an enum member after the Phase 7 migration (Pydantic
    # coerces bare strings on the way in, but the stored keys are enums).
    assert all(isinstance(k, ReadingSurface) for k in record.expected_outcome_per_surface)


def test_expected_outcome_per_surface_coerces_bare_strings() -> None:
    """Backward-compat: a caller passing bare strings still works — Pydantic
    coerces them to ReadingSurface members at construction."""
    record = _full_record(
        expected_outcome_per_surface={
            "substrate_side": "weak admit",
            "head_internal": "moderate admit",
            "behavior_side": "no admit",
        }
    )
    assert set(record.expected_outcome_per_surface) == {
        ReadingSurface.SUBSTRATE_SIDE,
        ReadingSurface.HEAD_INTERNAL,
        ReadingSurface.BEHAVIOR_SIDE,
    }


def test_expected_outcome_per_surface_rejects_invalid_surface_key() -> None:
    with pytest.raises(ValidationError):
        _full_record(
            expected_outcome_per_surface={"not_a_surface": "x"},
        )
