"""Phase 11 gate test — :mod:`kind.mirror.faithfulness`.

No LLM calls. The verifier is deterministic; the suite pins:

- :func:`canonicalize_scalar_field` cases (bare, compound, multi-dot,
  empty);
- the four-member :class:`FaithfulnessStatus` enum;
- the frozen invariant on :class:`FaithfulnessAssignment` and
  :class:`FaithfulnessResult` (Pydantic ``frozen=True``);
- the resolution-notes non-empty validator;
- the counts-sum-to-n_claims_total model-level invariant;
- the module-level :data:`FAITHFULNESS_THRESHOLD` constant at ``0.80``;
- the verifier's four-outcome decision tree across the citation cases
  (resolved, unresolved-field, unresolved-value, unresolved-range);
- the canonical-form match between a compound citation and a bare
  ``signal_name``;
- the LLM-prefill-overwrite contract — the verifier never reads or
  trusts whatever the LLM wrote into the claim's
  ``faithfulness_status`` field;
- the admissibility-at-threshold boundary (rate >= 0.80 → admissible
  True; rate < 0.80 → admissible False);
- the audit JSONL emission when ``audit_jsonl_path`` is provided;
- the input-immutability invariant — the verifier does not modify
  ``reading`` or ``statistic_results``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from kind.mirror.faithfulness import (
    FAITHFULNESS_THRESHOLD,
    FAITHFULNESS_VALUE_TOLERANCE,
    FaithfulnessAssignment,
    FaithfulnessResult,
    FaithfulnessStatus,
    canonicalize_scalar_field,
    verify_reading,
)
from kind.mirror.llm_caller import MirrorReading
from kind.mirror.statistics import StatisticResult
from kind.mirror.structured import StructuredClaim


# ---------------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------------


def _claim(
    *,
    cited_scalar_field: str = "latent_self_reference_t",
    cited_value: float = 0.5,
    cited_step_range: tuple[int, int] | None = None,
    faithfulness_status: str = "not_checked",
    reading_surface: str = "head_internal",
) -> StructuredClaim:
    return StructuredClaim(
        claim="example",
        cited_stream="agent_step",
        cited_run_id="r",
        cited_episode_range=None,
        cited_step_range=cited_step_range,
        cited_scalar_field=cited_scalar_field,
        cited_value=cited_value,
        falsifier="x",
        paraphrase_stability=None,
        reseed_stability=None,
        faithfulness_status=faithfulness_status,  # type: ignore[arg-type]
        judge_ruling="not_judged",
        reading_surface=reading_surface,  # type: ignore[arg-type]
        masked_steps_handling="n/a",
    )


def _reading(
    *,
    claims: list[StructuredClaim] | None = None,
    reader_role: str = "advocate",
) -> MirrorReading:
    return MirrorReading(
        run_id="r",
        timestamp_ms=1,
        reader_role=reader_role,  # type: ignore[arg-type]
        paired_reading_id="pair",
        framework_anchor="buddhist_phenomenology",
        baseline_flag="genuine",
        digest_run_id="r",
        digest_episode_range=(0, 1),
        claims=list(claims) if claims is not None else [_claim()],
        free_text_notes="",
    )


def _stat(name: str, value: Any = 0.5) -> StatisticResult:
    return StatisticResult(
        signal_name=name,
        value=value,
        estimator="x",
        n_samples=10,
        notes="",
    )


def _assignment(
    *,
    status: FaithfulnessStatus = FaithfulnessStatus.RESOLVED,
    resolution_notes: str = "ok",
) -> FaithfulnessAssignment:
    return FaithfulnessAssignment(
        pass_index=0,
        criterion_id="reflexive_attention",
        reader_role="primary",
        claim_index=0,
        cited_scalar_field_original="latent_self_reference_t",
        cited_scalar_field_canonical="latent_self_reference_t",
        cited_value=0.5,
        cited_step_range=None,
        status=status,
        resolution_notes=resolution_notes,
    )


def _result(
    *,
    n_total: int = 1,
    n_resolved: int = 1,
    assignments: tuple[FaithfulnessAssignment, ...] | None = None,
) -> FaithfulnessResult:
    if assignments is None:
        assignments = tuple(_assignment() for _ in range(n_total))
    rate = 1.0 if n_total == 0 else n_resolved / n_total
    return FaithfulnessResult(
        criterion_id="reflexive_attention",
        reader_role="primary",
        pass_index=0,
        run_id="r",
        checkpoint_id="ck",
        assignments=assignments,
        n_claims_total=n_total,
        n_resolved=n_resolved,
        n_unresolved_field=n_total - n_resolved,
        n_unresolved_value=0,
        n_unresolved_range=0,
        faithfulness_rate=rate,
        admissible=rate >= FAITHFULNESS_THRESHOLD,
        wallclock_ms=0,
        notes="",
    )


# ---------------------------------------------------------------------------
# canonicalize_scalar_field — the suffix-stripping rule.
# ---------------------------------------------------------------------------


def test_canonicalize_scalar_field_bare_form_unchanged() -> None:
    assert (
        canonicalize_scalar_field("latent_self_reference_t", frozenset())
        == "latent_self_reference_t"
    )


def test_canonicalize_scalar_field_compound_form_stripped() -> None:
    assert (
        canonicalize_scalar_field(
            "latent_self_reference_t.partial_autocorr_lag5", frozenset()
        )
        == "latent_self_reference_t"
    )


def test_canonicalize_scalar_field_empty_string_raises() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        canonicalize_scalar_field("", frozenset())


def test_canonicalize_scalar_field_iterated_strips_to_known_signal() -> None:
    # Phase 12.5 item 3: the LLM emits a two-dot form. One strip yields
    # `policy_entropy_t.classification`, which matches no signal name;
    # the iterated rule strips again to reach the known signal name.
    assert (
        canonicalize_scalar_field(
            "policy_entropy_t.classification.collapse",
            frozenset({"policy_entropy_t"}),
        )
        == "policy_entropy_t"
    )


def test_canonicalize_scalar_field_iterated_fallback_to_bare() -> None:
    # The same two-dot input with no known signal names falls back to
    # the bare form (all suffixes stripped).
    assert (
        canonicalize_scalar_field(
            "policy_entropy_t.classification.collapse", frozenset()
        )
        == "policy_entropy_t"
    )


# ---------------------------------------------------------------------------
# Enum and frozen-record invariants.
# ---------------------------------------------------------------------------


def test_faithfulness_status_enum_has_four_members() -> None:
    members = {m.name for m in FaithfulnessStatus}
    assert members == {
        "RESOLVED",
        "UNRESOLVED_FIELD",
        "UNRESOLVED_VALUE",
        "UNRESOLVED_RANGE",
    }


def test_faithfulness_assignment_is_frozen() -> None:
    a = _assignment()
    with pytest.raises(ValidationError):
        a.status = FaithfulnessStatus.UNRESOLVED_FIELD


def test_faithfulness_assignment_resolution_notes_non_empty() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        FaithfulnessAssignment(
            pass_index=0,
            criterion_id="x",
            reader_role="primary",
            claim_index=0,
            cited_scalar_field_original="f",
            cited_scalar_field_canonical="f",
            cited_value=0.0,
            cited_step_range=None,
            status=FaithfulnessStatus.RESOLVED,
            resolution_notes="   ",
        )


def test_faithfulness_result_is_frozen() -> None:
    r = _result()
    with pytest.raises(ValidationError):
        r.admissible = False


def test_faithfulness_result_counts_sum_to_n_claims_total() -> None:
    # Two claims, but the counts only sum to 1 — the validator should
    # reject this.
    with pytest.raises(ValidationError, match="counts do not sum"):
        FaithfulnessResult(
            criterion_id="x",
            reader_role="primary",
            pass_index=0,
            run_id="r",
            checkpoint_id="ck",
            assignments=(_assignment(), _assignment()),
            n_claims_total=2,
            n_resolved=1,
            n_unresolved_field=0,
            n_unresolved_value=0,
            n_unresolved_range=0,
            faithfulness_rate=0.5,
            admissible=False,
            wallclock_ms=0,
            notes="",
        )


def test_faithfulness_threshold_is_module_constant() -> None:
    assert FAITHFULNESS_THRESHOLD == 0.80


def test_faithfulness_value_tolerance_is_module_constant() -> None:
    assert FAITHFULNESS_VALUE_TOLERANCE == 1e-6


# ---------------------------------------------------------------------------
# verify_reading — the four outcomes.
# ---------------------------------------------------------------------------


def test_verify_reading_resolved_case() -> None:
    reading = _reading(
        claims=[
            _claim(
                cited_scalar_field="latent_self_reference_t",
                cited_value=0.7,
            )
        ]
    )
    stats = (_stat("latent_self_reference_t", 0.7),)
    out = verify_reading(
        reading,
        stats,
        criterion_id="reflexive_attention",
        pass_index=0,
        run_id="r",
        checkpoint_id="ck",
    )
    assert out.n_resolved == 1
    assert out.n_unresolved_field == 0
    assert out.n_unresolved_value == 0
    assert out.n_unresolved_range == 0
    assert out.faithfulness_rate == 1.0
    assert out.admissible is True
    assert out.assignments[0].status is FaithfulnessStatus.RESOLVED


def test_verify_reading_unresolved_field_case() -> None:
    reading = _reading(
        claims=[
            _claim(cited_scalar_field="not_a_real_signal", cited_value=0.7)
        ]
    )
    stats = (_stat("latent_self_reference_t", 0.7),)
    out = verify_reading(
        reading,
        stats,
        criterion_id="reflexive_attention",
        pass_index=0,
        run_id="r",
        checkpoint_id="ck",
    )
    assert out.n_unresolved_field == 1
    assert out.n_resolved == 0
    assignment = out.assignments[0]
    assert assignment.status is FaithfulnessStatus.UNRESOLVED_FIELD
    # The resolution notes name the available signal names so a
    # downstream reader can see what the verifier was given.
    assert "latent_self_reference_t" in assignment.resolution_notes


def test_verify_reading_unresolved_value_case() -> None:
    reading = _reading(
        claims=[
            _claim(
                cited_scalar_field="latent_self_reference_t",
                cited_value=0.7,
            )
        ]
    )
    stats = (_stat("latent_self_reference_t", 0.50),)
    out = verify_reading(
        reading,
        stats,
        criterion_id="reflexive_attention",
        pass_index=0,
        run_id="r",
        checkpoint_id="ck",
    )
    assert out.n_unresolved_value == 1
    assert out.assignments[0].status is FaithfulnessStatus.UNRESOLVED_VALUE


def test_verify_reading_canonical_form_matches_compound_to_bare() -> None:
    # The Phase 10 smoke pattern: LLM cites the compound form
    # `latent_self_reference_t.partial_autocorr_lag5`; the statistic
    # carries the bare `signal_name`. Canonical form must align them.
    reading = _reading(
        claims=[
            _claim(
                cited_scalar_field=(
                    "latent_self_reference_t.partial_autocorr_lag5"
                ),
                cited_value=0.123,
            )
        ]
    )
    stats = (_stat("latent_self_reference_t", 0.123),)
    out = verify_reading(
        reading,
        stats,
        criterion_id="reflexive_attention",
        pass_index=0,
        run_id="r",
        checkpoint_id="ck",
    )
    assert out.assignments[0].status is FaithfulnessStatus.RESOLVED
    assert (
        out.assignments[0].cited_scalar_field_canonical
        == "latent_self_reference_t"
    )
    assert (
        out.assignments[0].cited_scalar_field_original
        == "latent_self_reference_t.partial_autocorr_lag5"
    )


def test_verify_reading_dict_value_resolves_via_key_suffix() -> None:
    # The Phase 13 production pattern: LLM cites
    # `policy_entropy_t.no_response` (compound with a dict-key suffix)
    # against a dict-valued statistic.
    reading = _reading(
        claims=[
            _claim(
                cited_scalar_field="policy_entropy_t.no_response",
                cited_value=2.0,
            )
        ]
    )
    stats = (
        _stat(
            "policy_entropy_t",
            {"dip_and_recover": 5.0, "no_response": 2.0},
        ),
    )
    out = verify_reading(
        reading,
        stats,
        criterion_id="equanimity_perturbation_recovery",
        pass_index=0,
        run_id="r",
        checkpoint_id="ck",
    )
    assert out.assignments[0].status is FaithfulnessStatus.RESOLVED


def test_verify_reading_list_value_with_step_range_resolves() -> None:
    reading = _reading(
        claims=[
            _claim(
                cited_scalar_field="latent_regime_indicator_t",
                cited_value=2.0,
                cited_step_range=(0, 3),
            )
        ]
    )
    stats = (
        _stat(
            "latent_regime_indicator_t",
            [1.0, 2.0, 1.0, 3.0, 0.0, 0.0],
        ),
    )
    out = verify_reading(
        reading,
        stats,
        criterion_id="second_order_volition",
        pass_index=0,
        run_id="r",
        checkpoint_id="ck",
    )
    assert out.assignments[0].status is FaithfulnessStatus.RESOLVED


def test_verify_reading_list_value_with_step_range_out_of_bounds_trips_range() -> None:
    reading = _reading(
        claims=[
            _claim(
                cited_scalar_field="latent_regime_indicator_t",
                cited_value=2.0,
                cited_step_range=(0, 100),
            )
        ]
    )
    stats = (
        _stat("latent_regime_indicator_t", [1.0, 2.0, 1.0]),
    )
    out = verify_reading(
        reading,
        stats,
        criterion_id="second_order_volition",
        pass_index=0,
        run_id="r",
        checkpoint_id="ck",
    )
    assert out.n_unresolved_range == 1
    assert (
        out.assignments[0].status is FaithfulnessStatus.UNRESOLVED_RANGE
    )


def test_verify_reading_overwrites_llm_prefill() -> None:
    # The LLM pre-fills `faithfulness_status="resolved"` on a claim whose
    # citation actually doesn't resolve (UNRESOLVED_FIELD). The
    # verifier's verdict must be UNRESOLVED_FIELD regardless of the
    # LLM's pre-fill.
    reading = _reading(
        claims=[
            _claim(
                cited_scalar_field="not_a_real_signal",
                cited_value=0.0,
                faithfulness_status="resolved",
            )
        ]
    )
    stats = (_stat("latent_self_reference_t", 0.7),)
    out = verify_reading(
        reading,
        stats,
        criterion_id="reflexive_attention",
        pass_index=0,
        run_id="r",
        checkpoint_id="ck",
    )
    assert out.assignments[0].status is FaithfulnessStatus.UNRESOLVED_FIELD


def test_verify_reading_admissibility_at_threshold() -> None:
    # Construct a reading with exactly 80% resolution: 4 resolved + 1
    # unresolved out of 5 claims = 0.80 → admissible True.
    claims = [
        _claim(cited_scalar_field="latent_self_reference_t", cited_value=0.7),
        _claim(cited_scalar_field="latent_self_reference_t", cited_value=0.7),
        _claim(cited_scalar_field="latent_self_reference_t", cited_value=0.7),
        _claim(cited_scalar_field="latent_self_reference_t", cited_value=0.7),
        _claim(cited_scalar_field="missing_signal", cited_value=0.7),
    ]
    reading = _reading(claims=claims)
    stats = (_stat("latent_self_reference_t", 0.7),)
    out = verify_reading(
        reading,
        stats,
        criterion_id="reflexive_attention",
        pass_index=0,
        run_id="r",
        checkpoint_id="ck",
    )
    assert out.faithfulness_rate == pytest.approx(0.80)
    assert out.admissible is True

    # 3 of 5 resolved = 0.60 < 0.80 → admissible False.
    claims2 = claims[:3] + [
        _claim(cited_scalar_field="missing_signal_a", cited_value=0.7),
        _claim(cited_scalar_field="missing_signal_b", cited_value=0.7),
    ]
    reading2 = _reading(claims=claims2)
    out2 = verify_reading(
        reading2,
        stats,
        criterion_id="reflexive_attention",
        pass_index=0,
        run_id="r",
        checkpoint_id="ck",
    )
    assert out2.faithfulness_rate == pytest.approx(0.60)
    assert out2.admissible is False


def test_verify_reading_below_threshold_at_3_of_4_is_not_admissible() -> None:
    # 3 resolved of 4 claims = 0.75 < 0.80 → admissible False.
    claims = [
        _claim(cited_scalar_field="latent_self_reference_t", cited_value=0.7),
        _claim(cited_scalar_field="latent_self_reference_t", cited_value=0.7),
        _claim(cited_scalar_field="latent_self_reference_t", cited_value=0.7),
        _claim(cited_scalar_field="missing_signal", cited_value=0.7),
    ]
    reading = _reading(claims=claims)
    stats = (_stat("latent_self_reference_t", 0.7),)
    out = verify_reading(
        reading,
        stats,
        criterion_id="reflexive_attention",
        pass_index=0,
        run_id="r",
        checkpoint_id="ck",
    )
    assert out.faithfulness_rate == pytest.approx(0.75)
    assert out.admissible is False


def test_verify_reading_emits_audit_jsonl_when_path_provided(
    tmp_path: Path,
) -> None:
    reading = _reading()
    stats = (_stat("latent_self_reference_t", 0.5),)
    out_path = tmp_path / "faithfulness.jsonl"

    result1 = verify_reading(
        reading,
        stats,
        criterion_id="reflexive_attention",
        pass_index=0,
        run_id="r",
        checkpoint_id="ck",
        audit_jsonl_path=out_path,
    )
    # Append a second call to verify the JSONL is line-oriented.
    result2 = verify_reading(
        reading,
        stats,
        criterion_id="reflexive_attention",
        pass_index=1,
        run_id="r",
        checkpoint_id="ck",
        audit_jsonl_path=out_path,
    )

    lines = out_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    parsed1 = json.loads(lines[0])
    parsed2 = json.loads(lines[1])
    assert parsed1["pass_index"] == 0
    assert parsed2["pass_index"] == 1
    # The serialized status is the enum's string value.
    assert parsed1["assignments"][0]["status"] in {
        "resolved",
        "unresolved_field",
        "unresolved_value",
        "unresolved_range",
    }
    # Round-trip the result back through the model to assert the
    # serialization is the canonical FaithfulnessResult form.
    rehydrated = FaithfulnessResult.model_validate(parsed1)
    assert rehydrated.criterion_id == result1.criterion_id
    assert rehydrated.faithfulness_rate == result1.faithfulness_rate
    _ = result2


def test_verify_reading_does_not_modify_inputs() -> None:
    claims = [
        _claim(
            cited_scalar_field="latent_self_reference_t",
            cited_value=0.7,
            faithfulness_status="resolved",
        )
    ]
    reading = _reading(claims=claims)
    stats = (_stat("latent_self_reference_t", 0.5),)

    reading_before = reading.model_dump()
    stats_before = [s.model_dump() for s in stats]
    # Snapshot the claim instances so we can compare reference-equal
    # field values too.
    claim_snapshot = copy.deepcopy(reading.claims[0])

    _ = verify_reading(
        reading,
        stats,
        criterion_id="reflexive_attention",
        pass_index=0,
        run_id="r",
        checkpoint_id="ck",
    )

    assert reading.model_dump() == reading_before
    assert [s.model_dump() for s in stats] == stats_before
    # The LLM's pre-filled `faithfulness_status` on the source claim
    # is unchanged — the verifier's verdict lives on the assignment,
    # not on the claim.
    assert reading.claims[0].faithfulness_status == "resolved"
    assert reading.claims[0] == claim_snapshot


def test_verify_reading_empty_claims_is_vacuously_admissible() -> None:
    reading = _reading(claims=[])
    stats = (_stat("latent_self_reference_t", 0.5),)
    out = verify_reading(
        reading,
        stats,
        criterion_id="reflexive_attention",
        pass_index=0,
        run_id="r",
        checkpoint_id="ck",
    )
    assert out.n_claims_total == 0
    assert out.faithfulness_rate == 1.0
    assert out.admissible is True
    assert out.assignments == ()


def test_verify_reading_assignment_pass_index_propagates() -> None:
    reading = _reading()
    stats = (_stat("latent_self_reference_t", 0.5),)
    out = verify_reading(
        reading,
        stats,
        criterion_id="reflexive_attention",
        pass_index=7,
        run_id="r",
        checkpoint_id="ck",
    )
    assert out.pass_index == 7
    assert out.assignments[0].pass_index == 7
    assert out.assignments[0].criterion_id == "reflexive_attention"


def test_verify_reading_reader_role_maps_skeptic_to_adversarial() -> None:
    reading = _reading(reader_role="skeptic")
    stats = (_stat("latent_self_reference_t", 0.5),)
    out = verify_reading(
        reading,
        stats,
        criterion_id="reflexive_attention",
        pass_index=0,
        run_id="r",
        checkpoint_id="ck",
    )
    assert out.reader_role == "adversarial"
    assert out.assignments[0].reader_role == "adversarial"
