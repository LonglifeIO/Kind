"""Phase 9 gate test — :mod:`kind.mirror.judge_llm_caller`.

All tests inject a :class:`MockJudgeLLMClient`. No real API calls.

Covers:

- payload validation (``JudgePayload`` confidence range);
- the judge system prompt selection (every call uses
  :data:`JUDGE_SYSTEM_PROMPT`, never the primary / adversarial
  prompts);
- the per-criterion-count check (LLM returning a different number
  raises :class:`MirrorLLMError`);
- the per-criterion id mismatch check (LLM reordering raises);
- the malformed-output retry path: success after a transient
  failure; failure beyond the budget raises;
- the schema munger compatibility: the JudgeBatchPayload schema
  rounds-trips through :func:`_to_gemini_schema` cleanly;
- envelope-field stamping: the judgment's ``framework`` is filled
  from the criterion record, not the LLM output;
- per-criterion record emission via the record sink.
"""

from __future__ import annotations

from typing import Any

import pytest

from kind.mirror.calibration.llm_audit import LLMCallRecordCollector
from kind.mirror.criteria_v2 import (
    EQUANIMITY_PERTURBATION_RECOVERY,
    REFLEXIVE_ATTENTION,
    SECOND_ORDER_VOLITION,
)
from kind.mirror.judge import ClaimPolarity, CriterionJudgment
from kind.mirror.judge_llm_caller import (
    JUDGE_SYSTEM_PROMPT,
    ClaimPolarityAssignmentPayload,
    FalsifierVerdictPayload,
    JudgeBatchPayload,
    JudgePayload,
    MockJudgeLLMClient,
    call_judge_llm,
)
from kind.mirror.judge_prompt_builder import (
    JudgePromptFragment,
    build_judge_fragment,
)
from kind.mirror.llm_caller import (
    ADVERSARIAL_SYSTEM_PROMPT,
    PRIMARY_SYSTEM_PROMPT,
    LLMConfig,
    MirrorLLMError,
    MirrorReading,
)
from kind.mirror.statistics import StatisticResult
from kind.mirror.structured import StructuredClaim


# ---------------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------------


def _claim() -> StructuredClaim:
    return StructuredClaim(
        claim="example",
        cited_stream="agent_step",
        cited_run_id="r",
        cited_episode_range=(0, 1),
        cited_step_range=(10, 20),
        cited_scalar_field="h_t",
        cited_value=0.5,
        falsifier="x",
        paraphrase_stability=None,
        reseed_stability=None,
        faithfulness_status="not_checked",
        judge_ruling="not_judged",
        reading_surface="head_internal",
        masked_steps_handling="n/a",
    )


def _reading(reader_role: str = "advocate") -> MirrorReading:
    return MirrorReading(
        run_id="r",
        timestamp_ms=1,
        reader_role=reader_role,  # type: ignore[arg-type]
        paired_reading_id="pair",
        framework_anchor="buddhist_phenomenology",
        baseline_flag="genuine",
        digest_run_id="r",
        digest_episode_range=(0, 1),
        claims=[_claim()],
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


def _make_three_fragments() -> tuple[JudgePromptFragment, ...]:
    """Three judge fragments — one per active+held-out criterion."""
    return (
        build_judge_fragment(
            criterion=REFLEXIVE_ATTENTION,
            primary_readings_across_passes=(_reading(),),
            adversarial_readings_across_passes=(_reading("skeptic"),),
            statistic_results_across_passes=(
                (_stat("latent_self_reference_t", 0.7),),
            ),
        ),
        build_judge_fragment(
            criterion=EQUANIMITY_PERTURBATION_RECOVERY,
            primary_readings_across_passes=(_reading(),),
            adversarial_readings_across_passes=(_reading("skeptic"),),
            statistic_results_across_passes=(
                (
                    _stat("recovery_lag_steps", [5.0]),
                    _stat(
                        "policy_entropy_t",
                        {"dip_and_recover": 1.0, "collapse": 0.0,
                         "stays_elevated": 0.0, "no_response": 0.0},
                    ),
                    _stat(
                        "posterior_kl_t",
                        {"spike_and_decay": 1.0, "ratchet": 0.0,
                         "no_response": 0.0, "oscillation": 0.0},
                    ),
                ),
            ),
        ),
        build_judge_fragment(
            criterion=SECOND_ORDER_VOLITION,
            primary_readings_across_passes=(_reading(),),
            adversarial_readings_across_passes=(_reading("skeptic"),),
            statistic_results_across_passes=(
                (
                    _stat(
                        "policy_modulation_t",
                        {"contrast_magnitude": 0.5,
                         "observation_only_baseline": 0.1},
                    ),
                    _stat("latent_regime_indicator_t", [0.0, 1.0, 2.0, 3.0]),
                ),
            ),
        ),
    )


def _judge_payload(
    criterion_id: str,
    verdict: str = "satisfied",
    confidence: float = 0.8,
) -> JudgePayload:
    """Build a minimal JudgePayload that passes validation."""
    return JudgePayload(
        criterion_id=criterion_id,
        claim_polarity_assignments=(
            ClaimPolarityAssignmentPayload(
                pass_index=0,
                reader_role="primary",
                claim_index=0,
                cited_step_range=(10, 20),
                polarity=ClaimPolarity.SUPPORTIVE,
                polarity_rationale="claim cites the signal value",
            ),
        ),
        falsifier_verdicts=(
            FalsifierVerdictPayload(
                falsifier_id=f"{criterion_id}_v1",
                passes_supporting=(0,),
                passes_refuting=(),
                passes_non_falsifying=(),
                passes_ambiguous=(),
            ),
        ),
        verdict=verdict,  # type: ignore[arg-type]
        confidence=confidence,
        rationale="the cited evidence supports the criterion",
    )


def _batch_payload_for_three() -> JudgeBatchPayload:
    return JudgeBatchPayload(
        per_criterion=[
            _judge_payload("reflexive_attention"),
            _judge_payload("equanimity_perturbation_recovery"),
            _judge_payload("second_order_volition"),
        ]
    )


# ---------------------------------------------------------------------------
# Payload validation.
# ---------------------------------------------------------------------------


def test_judge_payload_rejects_confidence_above_one() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="confidence"):
        _judge_payload("reflexive_attention", confidence=1.5)


def test_judge_payload_rejects_confidence_below_zero() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="confidence"):
        _judge_payload("reflexive_attention", confidence=-0.1)


def test_judge_payload_accepts_each_verdict_value() -> None:
    for v in ("satisfied", "not_satisfied", "non_falsifying", "mixed", "ambiguous"):
        p = _judge_payload("reflexive_attention", verdict=v)
        assert p.verdict == v


# ---------------------------------------------------------------------------
# System prompt selection.
# ---------------------------------------------------------------------------


def test_judge_call_uses_judge_system_prompt() -> None:
    fragments = _make_three_fragments()
    client = MockJudgeLLMClient(_batch_payload_for_three())
    call_judge_llm(
        fragments,
        LLMConfig(),
        fragment_criteria=(
            REFLEXIVE_ATTENTION,
            EQUANIMITY_PERTURBATION_RECOVERY,
            SECOND_ORDER_VOLITION,
        ),
        run_id="r",
        round_id="r1",
        digest_run_id="r",
        digest_episode_range=(0, 10),
        client=client,
    )
    assert len(client.calls) == 1
    system_prompt, _ = client.calls[0]
    assert system_prompt == JUDGE_SYSTEM_PROMPT
    # And explicitly NOT one of the other two role prompts.
    assert system_prompt != PRIMARY_SYSTEM_PROMPT
    assert system_prompt != ADVERSARIAL_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Per-criterion count + id matching.
# ---------------------------------------------------------------------------


def test_wrong_number_of_judgments_raises() -> None:
    fragments = _make_three_fragments()
    payload = JudgeBatchPayload(
        per_criterion=[
            _judge_payload("reflexive_attention"),
            _judge_payload("equanimity_perturbation_recovery"),
        ]
    )
    client = MockJudgeLLMClient(payload)
    with pytest.raises(MirrorLLMError, match="per-criterion judgment"):
        call_judge_llm(
            fragments,
            LLMConfig(),
            fragment_criteria=(
                REFLEXIVE_ATTENTION,
                EQUANIMITY_PERTURBATION_RECOVERY,
                SECOND_ORDER_VOLITION,
            ),
            run_id="r",
            round_id="r1",
            digest_run_id="r",
            digest_episode_range=(0, 10),
            client=client,
        )


def test_reordered_criteria_raises() -> None:
    fragments = _make_three_fragments()
    payload = JudgeBatchPayload(
        per_criterion=[
            _judge_payload("equanimity_perturbation_recovery"),
            _judge_payload("reflexive_attention"),
            _judge_payload("second_order_volition"),
        ]
    )
    client = MockJudgeLLMClient(payload)
    with pytest.raises(MirrorLLMError, match="reordered or relabeled"):
        call_judge_llm(
            fragments,
            LLMConfig(),
            fragment_criteria=(
                REFLEXIVE_ATTENTION,
                EQUANIMITY_PERTURBATION_RECOVERY,
                SECOND_ORDER_VOLITION,
            ),
            run_id="r",
            round_id="r1",
            digest_run_id="r",
            digest_episode_range=(0, 10),
            client=client,
        )


def test_fragment_criteria_length_mismatch_raises() -> None:
    fragments = _make_three_fragments()
    client = MockJudgeLLMClient(_batch_payload_for_three())
    with pytest.raises(ValueError, match="fragment_criteria"):
        call_judge_llm(
            fragments,
            LLMConfig(),
            fragment_criteria=(REFLEXIVE_ATTENTION,),  # only one!
            run_id="r",
            round_id="r1",
            digest_run_id="r",
            digest_episode_range=(0, 10),
            client=client,
        )


def test_fragment_criteria_id_mismatch_raises() -> None:
    fragments = _make_three_fragments()
    client = MockJudgeLLMClient(_batch_payload_for_three())
    # Pair the equanimity fragment with the reflexive_attention criterion.
    with pytest.raises(ValueError, match="out of order"):
        call_judge_llm(
            fragments,
            LLMConfig(),
            fragment_criteria=(
                EQUANIMITY_PERTURBATION_RECOVERY,
                REFLEXIVE_ATTENTION,
                SECOND_ORDER_VOLITION,
            ),
            run_id="r",
            round_id="r1",
            digest_run_id="r",
            digest_episode_range=(0, 10),
            client=client,
        )


# ---------------------------------------------------------------------------
# Retry path.
# ---------------------------------------------------------------------------


def test_retry_recovers_within_budget() -> None:
    fragments = _make_three_fragments()
    good_payload = _batch_payload_for_three()
    client = MockJudgeLLMClient(
        [RuntimeError("transient malformed JSON"), good_payload]
    )
    judgments = call_judge_llm(
        fragments,
        LLMConfig(max_retries=3),
        fragment_criteria=(
            REFLEXIVE_ATTENTION,
            EQUANIMITY_PERTURBATION_RECOVERY,
            SECOND_ORDER_VOLITION,
        ),
        run_id="r",
        round_id="r1",
        digest_run_id="r",
        digest_episode_range=(0, 10),
        client=client,
    )
    assert len(judgments) == 3
    assert len(client.calls) == 2


def test_retry_exhaustion_raises_mirror_llm_error() -> None:
    fragments = _make_three_fragments()
    client = MockJudgeLLMClient(
        [
            RuntimeError("malformed 1"),
            RuntimeError("malformed 2"),
            RuntimeError("malformed 3"),
        ]
    )
    with pytest.raises(MirrorLLMError, match="max_retries"):
        call_judge_llm(
            fragments,
            LLMConfig(max_retries=1),
            fragment_criteria=(
                REFLEXIVE_ATTENTION,
                EQUANIMITY_PERTURBATION_RECOVERY,
                SECOND_ORDER_VOLITION,
            ),
            run_id="r",
            round_id="r1",
            digest_run_id="r",
            digest_episode_range=(0, 10),
            client=client,
        )


# ---------------------------------------------------------------------------
# Schema munger compatibility.
# ---------------------------------------------------------------------------


def test_judge_batch_payload_rounds_through_schema_munger_cleanly() -> None:
    """The Phase 12 finding (Gemini OpenAPI 3.0 rejects
    ``prefixItems``) applies to the judge schema too: the
    ``ClaimPolarityAssignmentPayload.cited_step_range`` is a
    ``tuple[int, int] | None`` that Pydantic emits as ``prefixItems``.
    Phase 9 reuses :func:`_to_gemini_schema` from Phase 8/12; this
    test pins the round-trip."""
    from kind.mirror.llm_caller import _to_gemini_schema

    src = JudgeBatchPayload.model_json_schema()
    munged = _to_gemini_schema(src)

    def _walk(o: Any) -> bool:
        if isinstance(o, dict):
            if "prefixItems" in o:
                return True
            return any(_walk(v) for v in o.values())
        if isinstance(o, list):
            return any(_walk(x) for x in o)
        return False

    # Source contains prefixItems (the tuple[int, int] cited_step_range).
    assert _walk(src), (
        "test premise broken: JudgeBatchPayload.model_json_schema() no "
        "longer contains prefixItems; the schema munger may now be "
        "unnecessary"
    )
    # Munged is clean.
    assert not _walk(munged)


def test_judge_batch_payload_munger_inlines_refs_and_drops_defs() -> None:
    from kind.mirror.llm_caller import _to_gemini_schema

    munged = _to_gemini_schema(JudgeBatchPayload.model_json_schema())
    assert "$defs" not in munged


# ---------------------------------------------------------------------------
# Envelope stamping.
# ---------------------------------------------------------------------------


def test_judgment_framework_comes_from_criterion_record() -> None:
    """The Phase 9 caller fills the judgment's ``framework`` field
    from the criterion record (not from the LLM output) so the
    framework is the frozen value, not whatever the judge said."""
    fragments = _make_three_fragments()
    client = MockJudgeLLMClient(_batch_payload_for_three())
    judgments = call_judge_llm(
        fragments,
        LLMConfig(),
        fragment_criteria=(
            REFLEXIVE_ATTENTION,
            EQUANIMITY_PERTURBATION_RECOVERY,
            SECOND_ORDER_VOLITION,
        ),
        run_id="r",
        round_id="r1",
        digest_run_id="r",
        digest_episode_range=(0, 10),
        client=client,
    )
    assert judgments[0].framework == REFLEXIVE_ATTENTION.framework
    assert (
        judgments[1].framework
        == EQUANIMITY_PERTURBATION_RECOVERY.framework
    )
    assert judgments[2].framework == SECOND_ORDER_VOLITION.framework


def test_judgment_polarity_assignments_carry_criterion_id() -> None:
    """The caller stamps ``criterion_id`` onto each
    :class:`ClaimPolarityAssignment` from the payload's envelope-less
    form."""
    fragments = _make_three_fragments()
    client = MockJudgeLLMClient(_batch_payload_for_three())
    judgments = call_judge_llm(
        fragments,
        LLMConfig(),
        fragment_criteria=(
            REFLEXIVE_ATTENTION,
            EQUANIMITY_PERTURBATION_RECOVERY,
            SECOND_ORDER_VOLITION,
        ),
        run_id="r",
        round_id="r1",
        digest_run_id="r",
        digest_episode_range=(0, 10),
        client=client,
    )
    for j in judgments:
        for assignment in j.claim_polarity_assignments:
            assert assignment.criterion_id == j.criterion_id


def test_empty_fragments_rejected() -> None:
    with pytest.raises(ValueError, match="empty tuple"):
        call_judge_llm(
            (),
            LLMConfig(),
            fragment_criteria=(),
            run_id="r",
            round_id="r1",
            digest_run_id="r",
            digest_episode_range=(0, 10),
            client=MockJudgeLLMClient(JudgeBatchPayload(per_criterion=[])),
        )


# ---------------------------------------------------------------------------
# Record sink emission.
# ---------------------------------------------------------------------------


def test_record_sink_receives_success_record_on_first_attempt() -> None:
    fragments = _make_three_fragments()
    client = MockJudgeLLMClient(_batch_payload_for_three())
    collector = LLMCallRecordCollector(
        round_id="r1", pass_index=0, checkpoint_id="ckpt"
    )
    call_judge_llm(
        fragments,
        LLMConfig(),
        fragment_criteria=(
            REFLEXIVE_ATTENTION,
            EQUANIMITY_PERTURBATION_RECOVERY,
            SECOND_ORDER_VOLITION,
        ),
        run_id="r",
        round_id="r1",
        digest_run_id="r",
        digest_episode_range=(0, 10),
        client=client,
        record_sink=collector,
    )
    records = collector.records
    assert len(records) == 1
    assert records[0].outcome == "success"
    assert records[0].attempt_number == 1


def test_record_sink_receives_retry_records_on_recovery() -> None:
    fragments = _make_three_fragments()
    client = MockJudgeLLMClient(
        [RuntimeError("transient"), _batch_payload_for_three()]
    )
    collector = LLMCallRecordCollector(
        round_id="r1", pass_index=0, checkpoint_id="ckpt"
    )
    call_judge_llm(
        fragments,
        LLMConfig(max_retries=3),
        fragment_criteria=(
            REFLEXIVE_ATTENTION,
            EQUANIMITY_PERTURBATION_RECOVERY,
            SECOND_ORDER_VOLITION,
        ),
        run_id="r",
        round_id="r1",
        digest_run_id="r",
        digest_episode_range=(0, 10),
        client=client,
        record_sink=collector,
    )
    records = collector.records
    assert len(records) == 2
    assert records[0].outcome == "runtime_error"
    assert records[1].outcome == "success"


def test_record_sink_receives_max_retries_exceeded_synthetic_record() -> None:
    fragments = _make_three_fragments()
    client = MockJudgeLLMClient(
        [RuntimeError("a"), RuntimeError("b"), RuntimeError("c")]
    )
    collector = LLMCallRecordCollector(
        round_id="r1", pass_index=0, checkpoint_id="ckpt"
    )
    with pytest.raises(MirrorLLMError):
        call_judge_llm(
            fragments,
            LLMConfig(max_retries=1),
            fragment_criteria=(
                REFLEXIVE_ATTENTION,
                EQUANIMITY_PERTURBATION_RECOVERY,
                SECOND_ORDER_VOLITION,
            ),
            run_id="r",
            round_id="r1",
            digest_run_id="r",
            digest_episode_range=(0, 10),
            client=client,
            record_sink=collector,
        )
    outcomes = [r.outcome for r in collector.records]
    # max_retries=1 → 2 attempts (1 + 1 retry); both fail; then one
    # synthetic max_retries_exceeded record.
    assert outcomes[-1] == "max_retries_exceeded"


# ---------------------------------------------------------------------------
# Round-trip serialization through the full Phase 9 stack.
# ---------------------------------------------------------------------------


def test_full_judgment_round_trips_through_json() -> None:
    """The :class:`CriterionJudgment` records produced by
    :func:`call_judge_llm` round-trip through Pydantic's
    JSON serialization."""
    fragments = _make_three_fragments()
    client = MockJudgeLLMClient(_batch_payload_for_three())
    judgments = call_judge_llm(
        fragments,
        LLMConfig(),
        fragment_criteria=(
            REFLEXIVE_ATTENTION,
            EQUANIMITY_PERTURBATION_RECOVERY,
            SECOND_ORDER_VOLITION,
        ),
        run_id="r",
        round_id="r1",
        digest_run_id="r",
        digest_episode_range=(0, 10),
        client=client,
    )
    for j in judgments:
        reloaded = CriterionJudgment.model_validate_json(j.model_dump_json())
        assert reloaded == j
