"""Phase 12 gate test — :mod:`kind.mirror.calibration.round_diff`.

Covers:

- :class:`ConfigFieldChange` non-empty rationale validator;
- :class:`CriterionSetChange` non-empty rationale validator;
- :func:`compute_round_diff` empty-when-identical;
- :func:`compute_round_diff` raises when changed-field has no rationale;
- :func:`compute_round_diff` populates the diff when rationales supplied;
- :func:`compute_round_diff` distinguishes statistic_config vs llm_config
  changes by field-path prefix;
- :func:`compute_round_diff` criterion-set change detection;
- round-trip serialization.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from kind.mirror.calibration.round import CheckpointSpec, RoundConfig
from kind.mirror.calibration.round_diff import (
    ConfigFieldChange,
    RoundDiff,
    SyntheticScheduleChange,
    compute_round_diff,
)
from kind.mirror.calibration.sham_schedule import generate_sham_schedule
from kind.mirror.calibration.synthetic_perturbation import (
    generate_synthetic_perturbation_schedule,
)
from kind.mirror.criteria_v2 import V2_REGISTRY
from kind.mirror.llm_caller import LLMConfig
from kind.mirror.registry import CriterionRegistry
from kind.mirror.statistics import StatisticConfig


def _checkpoint() -> CheckpointSpec:
    return CheckpointSpec(
        run_id="run-1",
        checkpoint_id="ckpt-000001",
        run_dir=Path("/tmp/run-1"),
    )


def _make_round_config(
    *,
    round_id: str = "round_a",
    statistic_config: StatisticConfig | None = None,
    llm_config: LLMConfig | None = None,
    active_registry: CriterionRegistry | None = None,
    held_out_registry: CriterionRegistry | None = None,
) -> RoundConfig:
    return RoundConfig(
        round_id=round_id,
        checkpoints=(_checkpoint(),),
        passes_per_checkpoint=5,
        statistic_config=statistic_config or StatisticConfig(),
        llm_config=llm_config or LLMConfig(),
        sham_schedule=generate_sham_schedule(
            checkpoint_ids=("ckpt-000001",),
            passes_per_checkpoint=5,
            real_perturbations_per_pass=2,
            shams_per_pass=1,
            telemetry_length=200,
            seed=42,
        ),
        active_registry=active_registry
        or CriterionRegistry(criteria=V2_REGISTRY.active()),
        held_out_registry=held_out_registry
        or CriterionRegistry(criteria=V2_REGISTRY.held_out()),
    )


# ---------------------------------------------------------------------------
# ConfigFieldChange.
# ---------------------------------------------------------------------------


def test_config_field_change_construction_succeeds() -> None:
    change = ConfigFieldChange(
        field_path="statistic_config.kmeans_k",
        prior_value=4,
        current_value=8,
        rationale="round 2 has wider latent space; bumping k to 8 to "
        "test cluster recovery",
    )
    assert change.field_path == "statistic_config.kmeans_k"


def test_config_field_change_rejects_empty_rationale() -> None:
    with pytest.raises(ValidationError):
        ConfigFieldChange(
            field_path="x.y",
            prior_value=1,
            current_value=2,
            rationale="",
        )


def test_config_field_change_rejects_whitespace_rationale() -> None:
    with pytest.raises(ValidationError):
        ConfigFieldChange(
            field_path="x.y",
            prior_value=1,
            current_value=2,
            rationale="   \n  ",
        )


# ---------------------------------------------------------------------------
# compute_round_diff — empty-when-identical.
# ---------------------------------------------------------------------------


def test_identical_configs_produce_empty_diff() -> None:
    """Two rounds with the same StatisticConfig / LLMConfig / criterion
    partition produce a diff with empty tuples."""
    c1 = _make_round_config(round_id="round_a")
    c2 = _make_round_config(round_id="round_b")
    diff = compute_round_diff(c1, c2)
    assert diff.prior_round_id == "round_a"
    assert diff.current_round_id == "round_b"
    assert diff.statistic_config_changes == ()
    assert diff.llm_config_changes == ()
    assert diff.criterion_set_changes is None


def test_compute_round_diff_with_notes() -> None:
    c1 = _make_round_config(round_id="round_a")
    c2 = _make_round_config(round_id="round_b")
    diff = compute_round_diff(
        c1, c2, notes="Phase 12 smoke: bit-identical configs expected"
    )
    assert "Phase 12 smoke" in diff.notes


# ---------------------------------------------------------------------------
# compute_round_diff — changes without rationales raise.
# ---------------------------------------------------------------------------


def test_changed_statistic_config_field_without_rationale_raises() -> None:
    c1 = _make_round_config(
        round_id="round_a", statistic_config=StatisticConfig(kmeans_k=4)
    )
    c2 = _make_round_config(
        round_id="round_b", statistic_config=StatisticConfig(kmeans_k=8)
    )
    with pytest.raises(ValueError, match="kmeans_k"):
        compute_round_diff(c1, c2, rationales=None)


def test_changed_llm_config_field_without_rationale_raises() -> None:
    c1 = _make_round_config(
        round_id="round_a", llm_config=LLMConfig(max_retries=3)
    )
    c2 = _make_round_config(
        round_id="round_b", llm_config=LLMConfig(max_retries=5)
    )
    with pytest.raises(ValueError, match="max_retries"):
        compute_round_diff(c1, c2)


def test_changed_field_with_rationale_appears_in_diff() -> None:
    c1 = _make_round_config(
        round_id="round_a", statistic_config=StatisticConfig(kmeans_k=4)
    )
    c2 = _make_round_config(
        round_id="round_b", statistic_config=StatisticConfig(kmeans_k=8)
    )
    diff = compute_round_diff(
        c1,
        c2,
        rationales={
            "statistic_config.kmeans_k": "round 2 tests wider clustering"
        },
    )
    assert len(diff.statistic_config_changes) == 1
    change = diff.statistic_config_changes[0]
    assert change.field_path == "statistic_config.kmeans_k"
    assert change.prior_value == 4
    assert change.current_value == 8
    assert "wider clustering" in change.rationale


def test_changes_separated_by_config_prefix() -> None:
    """Changes in StatisticConfig go to statistic_config_changes;
    changes in LLMConfig go to llm_config_changes."""
    c1 = _make_round_config(
        round_id="round_a",
        statistic_config=StatisticConfig(kmeans_k=4),
        llm_config=LLMConfig(max_retries=3),
    )
    c2 = _make_round_config(
        round_id="round_b",
        statistic_config=StatisticConfig(kmeans_k=8),
        llm_config=LLMConfig(max_retries=5),
    )
    diff = compute_round_diff(
        c1,
        c2,
        rationales={
            "statistic_config.kmeans_k": "test wider clustering",
            "llm_config.max_retries": "test increased robustness",
        },
    )
    assert len(diff.statistic_config_changes) == 1
    assert len(diff.llm_config_changes) == 1
    assert diff.statistic_config_changes[0].field_path == "statistic_config.kmeans_k"
    assert diff.llm_config_changes[0].field_path == "llm_config.max_retries"


# ---------------------------------------------------------------------------
# Criterion set changes.
# ---------------------------------------------------------------------------


def test_criterion_set_change_without_rationale_raises() -> None:
    """Swapping active and held-out partitions raises without
    rationale."""
    c1 = _make_round_config(
        round_id="round_a",
        active_registry=CriterionRegistry(criteria=V2_REGISTRY.active()),
        held_out_registry=CriterionRegistry(
            criteria=V2_REGISTRY.held_out()
        ),
    )
    c2 = _make_round_config(
        round_id="round_b",
        active_registry=CriterionRegistry(criteria=V2_REGISTRY.held_out()),
        held_out_registry=CriterionRegistry(criteria=V2_REGISTRY.active()),
    )
    with pytest.raises(ValueError, match="criterion-set"):
        compute_round_diff(c1, c2, rationales={})


def test_criterion_set_change_with_rationale_appears() -> None:
    c1 = _make_round_config(
        round_id="round_a",
        active_registry=CriterionRegistry(criteria=V2_REGISTRY.active()),
        held_out_registry=CriterionRegistry(
            criteria=V2_REGISTRY.held_out()
        ),
    )
    c2 = _make_round_config(
        round_id="round_b",
        active_registry=CriterionRegistry(criteria=V2_REGISTRY.held_out()),
        held_out_registry=CriterionRegistry(criteria=V2_REGISTRY.active()),
    )
    diff = compute_round_diff(
        c1,
        c2,
        rationales={
            "criterion_set": "Round 2 promotes the held-out criterion to "
            "active for the active-set audit"
        },
    )
    assert diff.criterion_set_changes is not None
    assert "promotes" in diff.criterion_set_changes.rationale


# ---------------------------------------------------------------------------
# Serialization.
# ---------------------------------------------------------------------------


def test_round_diff_serialization_round_trip() -> None:
    c1 = _make_round_config(
        round_id="round_a", statistic_config=StatisticConfig(kmeans_k=4)
    )
    c2 = _make_round_config(
        round_id="round_b", statistic_config=StatisticConfig(kmeans_k=8)
    )
    diff = compute_round_diff(
        c1,
        c2,
        rationales={"statistic_config.kmeans_k": "wider cluster test"},
        notes="round_a → round_b parameter sweep",
    )
    redumped = RoundDiff.model_validate_json(diff.model_dump_json())
    assert redumped == diff


# ---------------------------------------------------------------------------
# Phase 13: synthetic-schedule diff dimension.
# ---------------------------------------------------------------------------


def _make_round_config_with_synthetic(
    *,
    round_id: str,
    synthetics_per_pass: int,
    synthetic_seed: int,
) -> RoundConfig:
    """Build a RoundConfig matching the Phase-12 _make_round_config
    helper, plus a non-default synthetic schedule. Sham seed stays 42
    so the cross-schedule disjointness check has a populated sham
    schedule to evaluate against."""
    sham = generate_sham_schedule(
        checkpoint_ids=("ckpt-000001",),
        passes_per_checkpoint=5,
        real_perturbations_per_pass=2,
        shams_per_pass=1,
        telemetry_length=200,
        seed=42,
    )
    synthetic = generate_synthetic_perturbation_schedule(
        checkpoint_ids=("ckpt-000001",),
        passes_per_checkpoint=5,
        synthetics_per_pass=synthetics_per_pass,
        telemetry_length=200,
        recovery_window=50,
        seed=synthetic_seed,
        sham_schedule=sham,
    )
    return RoundConfig(
        round_id=round_id,
        checkpoints=(_checkpoint(),),
        passes_per_checkpoint=5,
        statistic_config=StatisticConfig(),
        llm_config=LLMConfig(),
        sham_schedule=sham,
        synthetic_schedule=synthetic,
        active_registry=CriterionRegistry(criteria=V2_REGISTRY.active()),
        held_out_registry=CriterionRegistry(criteria=V2_REGISTRY.held_out()),
    )


def test_compute_round_diff_identical_synthetic_schedules_produce_none() -> None:
    """Two rounds with byte-identical synthetic schedules produce
    ``synthetic_schedule_changes=None`` (no rationale needed)."""
    c1 = _make_round_config_with_synthetic(
        round_id="round_a",
        synthetics_per_pass=2,
        synthetic_seed=142,
    )
    c2 = _make_round_config_with_synthetic(
        round_id="round_b",
        synthetics_per_pass=2,
        synthetic_seed=142,
    )
    diff = compute_round_diff(c1, c2)
    assert diff.synthetic_schedule_changes is None


def test_compute_round_diff_different_synthetic_seed_requires_rationale() -> None:
    """Distinct synthetic seeds between rounds → schedules differ →
    rationale required."""
    c1 = _make_round_config_with_synthetic(
        round_id="round_a",
        synthetics_per_pass=2,
        synthetic_seed=142,
    )
    c2 = _make_round_config_with_synthetic(
        round_id="round_b",
        synthetics_per_pass=2,
        synthetic_seed=143,
    )
    with pytest.raises(ValueError, match="synthetic_schedule"):
        compute_round_diff(c1, c2)


def test_compute_round_diff_different_synthetic_with_rationale_produces_change() -> None:
    """Synthetic schedules differ + rationale supplied → diff carries
    a SyntheticScheduleChange record with the scalar fields."""
    c1 = _make_round_config_with_synthetic(
        round_id="round_a",
        synthetics_per_pass=2,
        synthetic_seed=142,
    )
    c2 = _make_round_config_with_synthetic(
        round_id="round_b",
        synthetics_per_pass=2,
        synthetic_seed=143,
    )
    diff = compute_round_diff(
        c1,
        c2,
        rationales={
            "synthetic_schedule": "Phase 13 uses distinct seeds per "
            "checkpoint so the schedules are independent."
        },
    )
    assert diff.synthetic_schedule_changes is not None
    change = diff.synthetic_schedule_changes
    assert change.prior_seed == 142
    assert change.current_seed == 143
    assert change.prior_synthetics_per_pass == 2
    assert change.current_synthetics_per_pass == 2
    assert change.prior_entry_count == change.current_entry_count == 10
    assert "distinct seeds" in change.rationale


def test_synthetic_schedule_change_rejects_empty_rationale() -> None:
    with pytest.raises(ValidationError):
        SyntheticScheduleChange(
            prior_seed=142,
            current_seed=143,
            prior_synthetics_per_pass=2,
            current_synthetics_per_pass=2,
            prior_entry_count=10,
            current_entry_count=10,
            rationale="",
        )


def test_round_diff_with_synthetic_change_serializes_round_trip() -> None:
    c1 = _make_round_config_with_synthetic(
        round_id="round_a",
        synthetics_per_pass=2,
        synthetic_seed=142,
    )
    c2 = _make_round_config_with_synthetic(
        round_id="round_b",
        synthetics_per_pass=2,
        synthetic_seed=143,
    )
    diff = compute_round_diff(
        c1,
        c2,
        rationales={
            "synthetic_schedule": "Phase 13 distinct seed per checkpoint"
        },
    )
    redumped = RoundDiff.model_validate_json(diff.model_dump_json())
    assert redumped == diff
