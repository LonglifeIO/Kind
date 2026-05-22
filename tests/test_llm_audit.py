"""Phase 12 gate test — :mod:`kind.mirror.calibration.llm_audit`.

Unit-level. No real LLM API calls; uses
:class:`~kind.mirror.llm_caller.MockLLMClient` for end-to-end record-
emission tests through :func:`~kind.mirror.llm_caller.call_mirror_llm`.

Covers:

- :class:`LLMCallRecord` field validation (non-empty ids, positive
  attempt_number, valid outcome, non-negative pass_index);
- :class:`LLMCallAudit.from_records` totals (count, retries, failures,
  wallclock, token sums; ``None`` when no records carry tokens);
- :class:`LLMCallRecordCollector` accumulation order, record snapshot
  immutability, negative-latency handling (clock skew → None);
- end-to-end: ``call_mirror_llm`` emits one ``success`` record on the
  happy path, two failure records + one success on a retry-recovered
  pass, three failure records + one synthetic ``max_retries_exceeded``
  record when the budget is exhausted.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kind.mirror.calibration.llm_audit import (
    LLMCallAudit,
    LLMCallRecord,
    LLMCallRecordCollector,
)
from kind.mirror.criteria_v2 import (
    EQUANIMITY_PERTURBATION_RECOVERY,
    REFLEXIVE_ATTENTION,
    SECOND_ORDER_VOLITION,
)
from kind.mirror.llm_caller import (
    BatchPayload,
    LLMConfig,
    MirrorLLMError,
    MockLLMClient,
    _PerCriterionReadingPayload,
    call_mirror_llm,
)
from kind.mirror.perturbation_align import PerturbationTimeline
from kind.mirror.prompt_builder import PromptFragment, build_fragment
from kind.mirror.statistics import StatisticResult
from kind.mirror.structured import StructuredClaim


# ---------------------------------------------------------------------------
# Fixture helpers.
# ---------------------------------------------------------------------------


def _stat(name: str, value: float | list[float] | dict[str, float] = 0.5) -> StatisticResult:
    return StatisticResult(
        signal_name=name, value=value, estimator="x", n_samples=100, notes="n"
    )


def _claim() -> StructuredClaim:
    return StructuredClaim(
        claim="x",
        cited_stream="agent_step",
        cited_run_id="r",
        cited_episode_range=(0, 1),
        cited_step_range=(0, 100),
        cited_scalar_field="h_t",
        cited_value=0.0,
        falsifier="f",
        paraphrase_stability=None,
        reseed_stability=None,
        faithfulness_status="not_checked",
        judge_ruling="not_judged",
        reading_surface="head_internal",
        masked_steps_handling="n/a",
    )


def _per_criterion(criterion_id: str) -> _PerCriterionReadingPayload:
    return _PerCriterionReadingPayload(
        criterion_id=criterion_id,
        framework_anchor="buddhist_phenomenology",
        claims=[_claim()],
        free_text_notes="n",
    )


def _make_fragments() -> tuple[PromptFragment, ...]:
    timeline = PerturbationTimeline(events=(), run_id="r", checkpoint_id="c")
    return (
        build_fragment(
            REFLEXIVE_ATTENTION,
            statistic_results=(
                _stat("latent_self_reference_t", 0.7),
                _stat("dream_self_reference_t", 0.6),
            ),
        ),
        build_fragment(
            EQUANIMITY_PERTURBATION_RECOVERY,
            statistic_results=(
                _stat("recovery_lag_steps", [3.0]),
                _stat(
                    "policy_entropy_t",
                    {
                        "dip_and_recover": 1.0,
                        "collapse": 0.0,
                        "stays_elevated": 0.0,
                        "no_response": 0.0,
                    },
                ),
                _stat(
                    "posterior_kl_t",
                    {
                        "spike_and_decay": 1.0,
                        "ratchet": 0.0,
                        "no_response": 0.0,
                        "oscillation": 0.0,
                    },
                ),
            ),
            perturbation_timeline=timeline,
        ),
        build_fragment(
            SECOND_ORDER_VOLITION,
            statistic_results=(
                _stat(
                    "policy_modulation_t",
                    {"contrast_magnitude": 0.5, "observation_only_baseline": 0.1},
                ),
                _stat("latent_regime_indicator_t", [0.0, 1.0, 2.0, 3.0]),
            ),
        ),
    )


def _good_payload() -> BatchPayload:
    return BatchPayload(
        per_criterion=[
            _per_criterion("reflexive_attention"),
            _per_criterion("equanimity_perturbation_recovery"),
            _per_criterion("second_order_volition"),
        ]
    )


# ---------------------------------------------------------------------------
# LLMCallRecord field validation.
# ---------------------------------------------------------------------------


def _record(**overrides: object) -> LLMCallRecord:
    base: dict[str, object] = {
        "round_id": "round_1",
        "pass_index": 0,
        "checkpoint_id": "ckpt-000001",
        "role": "primary",
        "attempt_number": 1,
        "request_timestamp_ms": 1_700_000_000_000,
        "response_timestamp_ms": 1_700_000_000_500,
        "latency_ms": 500,
        "model_name": "gemini-2.5-pro",
        "prompt_token_count": 1200,
        "response_token_count": 800,
        "outcome": "success",
        "error_message": None,
    }
    base.update(overrides)
    return LLMCallRecord(**base)  # type: ignore[arg-type]


def test_record_construction_with_full_fields_succeeds() -> None:
    r = _record()
    assert r.outcome == "success"
    assert r.latency_ms == 500


def test_record_rejects_empty_round_id() -> None:
    with pytest.raises(ValidationError):
        _record(round_id="")


def test_record_rejects_empty_model_name() -> None:
    with pytest.raises(ValidationError):
        _record(model_name="   ")


def test_record_rejects_invalid_outcome() -> None:
    with pytest.raises(ValidationError):
        _record(outcome="success_ish")


def test_record_rejects_zero_attempt_number() -> None:
    with pytest.raises(ValidationError):
        _record(attempt_number=0)


def test_record_rejects_negative_pass_index() -> None:
    with pytest.raises(ValidationError):
        _record(pass_index=-1)


def test_record_failure_carries_error_message() -> None:
    r = _record(
        outcome="runtime_error",
        response_timestamp_ms=None,
        latency_ms=None,
        prompt_token_count=None,
        response_token_count=None,
        error_message="RuntimeError('SDK failed')",
    )
    assert r.outcome == "runtime_error"
    assert r.error_message == "RuntimeError('SDK failed')"
    assert r.latency_ms is None


def test_record_frozen() -> None:
    r = _record()
    with pytest.raises(ValidationError):
        r.round_id = "different"


# ---------------------------------------------------------------------------
# LLMCallAudit.from_records aggregation.
# ---------------------------------------------------------------------------


def test_audit_from_records_aggregates_totals() -> None:
    records = (
        _record(attempt_number=1, outcome="validation_error", error_message="e1"),
        _record(attempt_number=2, outcome="success"),
        _record(attempt_number=1, outcome="success", prompt_token_count=600, response_token_count=400),
    )
    audit = LLMCallAudit.from_records(records)
    assert audit.total_calls == 3
    assert audit.total_retries == 1
    assert audit.total_failures == 0
    # 500 + 500 + 500 = 1500 ms
    assert audit.total_wallclock_ms == 1500
    assert audit.total_tokens_in == 1200 + 1200 + 600
    assert audit.total_tokens_out == 800 + 800 + 400


def test_audit_from_records_empty() -> None:
    audit = LLMCallAudit.from_records(())
    assert audit.total_calls == 0
    assert audit.total_retries == 0
    assert audit.total_failures == 0
    assert audit.total_wallclock_ms == 0
    assert audit.total_tokens_in is None
    assert audit.total_tokens_out is None


def test_audit_from_records_failure_path() -> None:
    """A failed pass: three retries then max_retries_exceeded."""
    records = (
        _record(attempt_number=1, outcome="validation_error", error_message="e1"),
        _record(attempt_number=2, outcome="runtime_error", error_message="e2"),
        _record(
            attempt_number=3,
            outcome="max_retries_exceeded",
            response_timestamp_ms=None,
            latency_ms=None,
            prompt_token_count=None,
            response_token_count=None,
            error_message="last",
        ),
    )
    audit = LLMCallAudit.from_records(records)
    assert audit.total_calls == 3
    assert audit.total_retries == 2  # attempt_number 2 and 3
    assert audit.total_failures == 1
    # max_retries_exceeded carries no latency; first two contribute 500 each.
    assert audit.total_wallclock_ms == 1000


def test_audit_tokens_none_when_no_records_carry_them() -> None:
    """Every record has token counts == None → audit totals are None."""
    records = (
        _record(prompt_token_count=None, response_token_count=None),
        _record(prompt_token_count=None, response_token_count=None),
    )
    audit = LLMCallAudit.from_records(records)
    assert audit.total_tokens_in is None
    assert audit.total_tokens_out is None


def test_audit_round_trip_serializes() -> None:
    records = (
        _record(),
        _record(attempt_number=2, outcome="validation_error", error_message="e"),
    )
    audit = LLMCallAudit.from_records(records)
    redumped = LLMCallAudit.model_validate_json(audit.model_dump_json())
    assert redumped == audit


# ---------------------------------------------------------------------------
# LLMCallRecordCollector.
# ---------------------------------------------------------------------------


def test_collector_appends_records_in_order() -> None:
    coll = LLMCallRecordCollector(
        round_id="round_1", pass_index=0, checkpoint_id="ckpt-000001"
    )
    coll.record(
        role="primary",
        attempt_number=1,
        request_timestamp_ms=1_000,
        response_timestamp_ms=1_500,
        model_name="gemini-2.5-pro",
        prompt_token_count=100,
        response_token_count=50,
        outcome="success",
        error_message=None,
    )
    coll.record(
        role="adversarial",
        attempt_number=1,
        request_timestamp_ms=2_000,
        response_timestamp_ms=2_300,
        model_name="gemini-2.5-pro",
        prompt_token_count=None,
        response_token_count=None,
        outcome="success",
        error_message=None,
    )
    assert len(coll) == 2
    assert coll.records[0].role == "primary"
    assert coll.records[1].role == "adversarial"
    assert coll.records[0].latency_ms == 500
    assert coll.records[1].latency_ms == 300


def test_collector_rejects_negative_clock_skew_latency() -> None:
    coll = LLMCallRecordCollector(
        round_id="r", pass_index=0, checkpoint_id="c"
    )
    rec = coll.record(
        role="primary",
        attempt_number=1,
        request_timestamp_ms=2_000,
        response_timestamp_ms=1_500,  # response before request — clock skew
        model_name="gemini-2.5-pro",
        prompt_token_count=None,
        response_token_count=None,
        outcome="success",
        error_message=None,
    )
    assert rec.latency_ms is None


def test_collector_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError):
        LLMCallRecordCollector(
            round_id="", pass_index=0, checkpoint_id="c"
        )
    with pytest.raises(ValueError):
        LLMCallRecordCollector(
            round_id="r", pass_index=-1, checkpoint_id="c"
        )
    with pytest.raises(ValueError):
        LLMCallRecordCollector(
            round_id="r", pass_index=0, checkpoint_id=""
        )


def test_collector_snapshot_is_tuple() -> None:
    """`collector.records` is a tuple — mutating the returned snapshot
    does not affect the collector's internal state."""
    coll = LLMCallRecordCollector(
        round_id="r", pass_index=0, checkpoint_id="c"
    )
    coll.record(
        role="primary",
        attempt_number=1,
        request_timestamp_ms=1_000,
        response_timestamp_ms=1_100,
        model_name="m",
        prompt_token_count=None,
        response_token_count=None,
        outcome="success",
        error_message=None,
    )
    snapshot1 = coll.records
    coll.record(
        role="adversarial",
        attempt_number=1,
        request_timestamp_ms=1_200,
        response_timestamp_ms=1_300,
        model_name="m",
        prompt_token_count=None,
        response_token_count=None,
        outcome="success",
        error_message=None,
    )
    snapshot2 = coll.records
    assert len(snapshot1) == 1
    assert len(snapshot2) == 2
    assert isinstance(snapshot1, tuple)


# ---------------------------------------------------------------------------
# End-to-end through call_mirror_llm.
# ---------------------------------------------------------------------------


def test_call_mirror_llm_emits_success_record() -> None:
    fragments = _make_fragments()
    coll = LLMCallRecordCollector(
        round_id="r", pass_index=0, checkpoint_id="ckpt"
    )
    client = MockLLMClient(_good_payload())
    call_mirror_llm(
        fragments,
        role="primary",
        config=LLMConfig(),
        run_id="r",
        digest_run_id="d",
        digest_episode_range=(0, 10),
        client=client,
        record_sink=coll,
    )
    assert len(coll) == 1
    assert coll.records[0].outcome == "success"
    assert coll.records[0].attempt_number == 1
    assert coll.records[0].role == "primary"


def test_call_mirror_llm_emits_retry_then_success_records() -> None:
    fragments = _make_fragments()
    coll = LLMCallRecordCollector(
        round_id="r", pass_index=0, checkpoint_id="ckpt"
    )
    client = MockLLMClient(
        [RuntimeError("malformed"), _good_payload()]
    )
    call_mirror_llm(
        fragments,
        role="primary",
        config=LLMConfig(max_retries=3),
        run_id="r",
        digest_run_id="d",
        digest_episode_range=(0, 10),
        client=client,
        record_sink=coll,
    )
    assert len(coll) == 2
    assert coll.records[0].outcome == "runtime_error"
    assert coll.records[0].attempt_number == 1
    assert coll.records[1].outcome == "success"
    assert coll.records[1].attempt_number == 2


def test_call_mirror_llm_emits_max_retries_exceeded_synthetic_record() -> None:
    fragments = _make_fragments()
    coll = LLMCallRecordCollector(
        round_id="r", pass_index=0, checkpoint_id="ckpt"
    )
    client = MockLLMClient(
        [
            RuntimeError("e1"),
            RuntimeError("e2"),
            RuntimeError("e3"),
        ]
    )
    with pytest.raises(MirrorLLMError):
        call_mirror_llm(
            fragments,
            role="primary",
            config=LLMConfig(max_retries=1),
            run_id="r",
            digest_run_id="d",
            digest_episode_range=(0, 10),
            client=client,
            record_sink=coll,
        )
    # max_retries=1 means 2 attempts. Both fail, then 1 synthetic
    # max_retries_exceeded record. Total: 3 records.
    assert len(coll) == 3
    assert coll.records[0].outcome == "runtime_error"
    assert coll.records[1].outcome == "runtime_error"
    assert coll.records[2].outcome == "max_retries_exceeded"
    assert coll.records[2].response_timestamp_ms is None
    assert coll.records[2].latency_ms is None


def test_call_mirror_llm_record_sink_optional() -> None:
    """No record_sink: the call still works; no records collected."""
    fragments = _make_fragments()
    client = MockLLMClient(_good_payload())
    readings = call_mirror_llm(
        fragments,
        role="primary",
        config=LLMConfig(),
        run_id="r",
        digest_run_id="d",
        digest_episode_range=(0, 10),
        client=client,
        record_sink=None,
    )
    assert len(readings) == 3
