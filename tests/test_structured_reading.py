"""Probe 2 Phase 0 gate test 3 — structured-reading schema with v2 fields.

Per the Probe 2 implementation plan §4 row 3: ``StructuredReading``,
``StructuredClaim``, and ``JudgeRuling`` round-trip through JSONL
byte-stable; all enum values including the three ``reading_surface``
values, the new ``baseline_flag`` values, and the new
``cited_stream="conditioning_analysis"`` value validate. Synthesis §2.2
calls out the per-surface stratification as the load-bearing v2
addition; this test pins the schema-level shape it depends on.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from kind.mirror.structured import (
    MIRROR_READING_V2_VERSION,
    JudgeRuling,
    StructuredClaim,
    StructuredReading,
)


def _claim(
    *,
    reading_surface: str = "head_internal",
    cited_stream: str = "agent_step",
    cited_scalar_field: str = "self_prediction_error_t",
    masked_steps_handling: str = "excluded",
    cited_value: float = 0.0091,
    judge_ruling: str = "not_judged",
    faithfulness_status: str = "not_checked",
) -> StructuredClaim:
    """Build a StructuredClaim with sensible defaults; tests override
    specific fields to exercise the per-enum surfaces."""
    return StructuredClaim(
        claim="placeholder claim text",
        cited_stream=cited_stream,  # type: ignore[arg-type]
        cited_run_id="probe1_5_phase7_5-20260507-101800",
        cited_episode_range=(20, 25),
        cited_step_range=(4000, 5000),
        cited_scalar_field=cited_scalar_field,
        cited_value=cited_value,
        falsifier="if the cited mean shifts by >2σ under within-episode shuffle",
        paraphrase_stability=None,
        reseed_stability=None,
        faithfulness_status=faithfulness_status,  # type: ignore[arg-type]
        judge_ruling=judge_ruling,  # type: ignore[arg-type]
        reading_surface=reading_surface,  # type: ignore[arg-type]
        masked_steps_handling=masked_steps_handling,  # type: ignore[arg-type]
    )


def _reading(
    *,
    baseline_flag: str = "genuine",
    reader_role: str = "advocate",
    framework_anchor: str = "buddhist_phenomenology",
    claims: list[StructuredClaim] | None = None,
) -> StructuredReading:
    return StructuredReading(
        run_id="probe2_smoke-20260508-090000",
        timestamp_ms=1_730_000_000_000,
        reader_role=reader_role,  # type: ignore[arg-type]
        paired_reading_id="paired-0001",
        framework_anchor=framework_anchor,  # type: ignore[arg-type]
        baseline_flag=baseline_flag,  # type: ignore[arg-type]
        digest_run_id="probe1_5_phase7_5-20260507-101800",
        digest_episode_range=(0, 25),
        claims=claims if claims is not None else [_claim()],
        free_text_notes="placeholder notes",
    )


# ---- (1) StructuredClaim round-trips through JSONL byte-stable -----------


def test_structured_claim_jsonl_round_trip_is_byte_stable() -> None:
    """One claim → JSONL line → reparsed claim → re-serialized JSONL line.
    The bytes match across the round-trip; the model-level equality
    holds."""
    original = _claim()
    line_a = original.model_dump_json()
    parsed = StructuredClaim.model_validate_json(line_a)
    line_b = parsed.model_dump_json()
    assert line_a == line_b
    assert parsed == original


# ---- (2) StructuredReading round-trips with claims -----------------------


def test_structured_reading_jsonl_round_trip_byte_stable() -> None:
    original = _reading(claims=[_claim(), _claim(reading_surface="substrate_side")])
    line_a = original.model_dump_json()
    parsed = StructuredReading.model_validate_json(line_a)
    line_b = parsed.model_dump_json()
    assert line_a == line_b
    assert parsed == original


def test_structured_reading_schema_version_default_is_v2() -> None:
    reading = _reading()
    assert reading.schema_version == MIRROR_READING_V2_VERSION
    assert MIRROR_READING_V2_VERSION == "0.2.0"


# ---- (3) all three reading_surface values validate -----------------------


@pytest.mark.parametrize(
    "surface", ["substrate_side", "head_internal", "behavior_side"]
)
def test_all_three_reading_surface_values_validate(surface: str) -> None:
    """Each of the three reading surfaces is a valid value for
    StructuredClaim.reading_surface."""
    claim = _claim(reading_surface=surface)
    assert claim.reading_surface == surface


def test_invalid_reading_surface_raises() -> None:
    with pytest.raises(ValidationError):
        _claim(reading_surface="not_a_real_surface")


# ---- (4) all baseline_flag enum values including v2 additions ------------


@pytest.mark.parametrize(
    "flag",
    [
        "genuine",
        "shuffled_within_episode",
        "shuffled_across_episodes",
        "decoupled_action_state",
        "shuffled_scalar_within_trajectory",
        "lesion_k1",
        "lesion_constant",
        "lesion_disable_self_prediction",
        "lesion_init_zero_scalar_column",
        "lesion_zero_or_randomize_scalar",
        "sham_aligned",
    ],
)
def test_all_baseline_flag_values_validate(flag: str) -> None:
    """The baseline_flag enum carries v1's seven values plus four v2
    additions: shuffled_scalar_within_trajectory (synthesis §2.4 element
    2) and three lesion variants (synthesis §2.4 element 4)."""
    reading = _reading(baseline_flag=flag)
    assert reading.baseline_flag == flag


def test_invalid_baseline_flag_raises() -> None:
    with pytest.raises(ValidationError):
        _reading(baseline_flag="lesion_imaginary")


# ---- (5) cited_stream "conditioning_analysis" is admitted ----------------


@pytest.mark.parametrize(
    "stream",
    [
        "agent_step",
        "dream_rollout",
        "replay_meta",
        "world_event",
        "conditioning_analysis",
    ],
)
def test_cited_stream_values_validate(stream: str) -> None:
    """The cited_stream literal carries Probe 1's four telemetry streams
    plus Probe 2 v2's behavior-side ``conditioning_analysis`` stream."""
    claim = _claim(cited_stream=stream)
    assert claim.cited_stream == stream


def test_invalid_cited_stream_raises() -> None:
    with pytest.raises(ValidationError):
        _claim(cited_stream="agent_step_extra")


# ---- (6) masked_steps_handling enum values ------------------------------


@pytest.mark.parametrize("handling", ["included", "excluded", "n/a"])
def test_all_masked_steps_handling_values_validate(handling: str) -> None:
    claim = _claim(masked_steps_handling=handling)
    assert claim.masked_steps_handling == handling


# ---- (7) JudgeRuling round-trips with per-surface tuples -----------------


def test_judge_ruling_jsonl_round_trip_byte_stable() -> None:
    """The rulings field is a list of (claim_index, reading_surface,
    ruling, ground_text) tuples — the v2 per-surface tuple shape."""
    ruling = JudgeRuling(
        run_id="probe2_smoke-20260508-090000",
        timestamp_ms=1_730_000_000_001,
        paired_reading_id="paired-0001",
        advocate_id="advocate-gemini-2.5-pro",
        skeptic_id="skeptic-claude-opus-4-7",
        digest_run_id="probe1_5_phase7_5-20260507-101800",
        rulings=[
            (0, "substrate_side", "supported", "kl_aggregate_t mean=12.4 at ep 22"),
            (1, "head_internal", "absent", "sp_err shows no recovery shape"),
            (2, "behavior_side", "unresolved", "conditioning citation off-tolerance"),
        ],
        agreement_without_evidence_unresolved=[3],
    )
    line_a = ruling.model_dump_json()
    parsed = JudgeRuling.model_validate_json(line_a)
    line_b = parsed.model_dump_json()
    assert line_a == line_b
    assert parsed == ruling


def test_judge_ruling_rejects_invalid_surface_in_tuple() -> None:
    with pytest.raises(ValidationError):
        JudgeRuling(
            run_id="r",
            timestamp_ms=0,
            paired_reading_id="p",
            advocate_id="a",
            skeptic_id="s",
            digest_run_id="d",
            rulings=[(0, "not_a_surface", "supported", "g")],  # type: ignore[list-item]
            agreement_without_evidence_unresolved=[],
        )


def test_judge_ruling_rejects_invalid_outcome_in_tuple() -> None:
    """The Judge cannot rule "not_judged" — that's a writer-side default
    on the StructuredClaim before the Judge runs. Once the Judge runs,
    every per-surface ruling is one of three outcomes."""
    with pytest.raises(ValidationError):
        JudgeRuling(
            run_id="r",
            timestamp_ms=0,
            paired_reading_id="p",
            advocate_id="a",
            skeptic_id="s",
            digest_run_id="d",
            rulings=[(0, "substrate_side", "not_judged", "g")],  # type: ignore[list-item]
            agreement_without_evidence_unresolved=[],
        )


def test_judge_ruling_rejects_negative_agreement_index() -> None:
    with pytest.raises(ValidationError):
        JudgeRuling(
            run_id="r",
            timestamp_ms=0,
            paired_reading_id="p",
            advocate_id="a",
            skeptic_id="s",
            digest_run_id="d",
            rulings=[],
            agreement_without_evidence_unresolved=[-1, 2],
        )


def test_judge_ruling_rejects_duplicate_agreement_indices() -> None:
    with pytest.raises(ValidationError):
        JudgeRuling(
            run_id="r",
            timestamp_ms=0,
            paired_reading_id="p",
            advocate_id="a",
            skeptic_id="s",
            digest_run_id="d",
            rulings=[],
            agreement_without_evidence_unresolved=[1, 1, 2],
        )


# ---- (8) frozen + extra=forbid disciplines -------------------------------


def test_structured_claim_extra_field_rejected() -> None:
    payload: dict[str, Any] = StructuredClaim.model_validate_json(
        _claim().model_dump_json()
    ).model_dump()
    payload["unexpected_field"] = "x"
    with pytest.raises(ValidationError):
        StructuredClaim.model_validate(payload)


def test_structured_reading_extra_field_rejected() -> None:
    payload: dict[str, Any] = StructuredReading.model_validate_json(
        _reading().model_dump_json()
    ).model_dump()
    payload["unexpected_field"] = "x"
    with pytest.raises(ValidationError):
        StructuredReading.model_validate(payload)


def test_structured_reading_is_frozen() -> None:
    """frozen=True: a reading cannot be mutated after construction. The
    same discipline RecordEnvelope carries — once produced, the reading
    is what it is."""
    reading = _reading()
    with pytest.raises(ValidationError):
        reading.run_id = "different"  # type: ignore[misc]
