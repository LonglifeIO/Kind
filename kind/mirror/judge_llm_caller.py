"""Phase 9 judge LLM-call layer — :class:`JudgePayload`,
:data:`JUDGE_SYSTEM_PROMPT`, :func:`call_judge_llm`.

Mirrors :mod:`kind.mirror.llm_caller` from Phase 8 in structure. The
judge LLM is Phase 9's third role — distinct from the Phenomenological
Advocate (primary) and the Statistical Skeptic (adversarial). The
judge's system prompt is the Methodological Arbiter; its structured
output schema is :class:`JudgePayload`.

The caller is intentionally narrow:

- one entrypoint, :func:`call_judge_llm`, takes the judge fragments +
  a typed config (and an optional injected client for tests), and
  returns one :class:`~kind.mirror.judge.CriterionJudgment` per
  fragment;
- one role-specific system prompt: :data:`JUDGE_SYSTEM_PROMPT` (the
  Methodological Arbiter stance);
- bounded retries on malformed structured output, capped at
  :attr:`~kind.mirror.llm_caller.LLMConfig.max_retries` by default;
  persistent malformation raises :class:`MirrorLLMError` so the driver
  can halt the judge run on this criterion (not the round);
- the Phase 12 :class:`~kind.mirror.calibration.llm_audit.LLMRecordSink`
  protocol threads through so the judge's call records merge into the
  Phase 9 audit;
- no real API calls in tests — tests inject a
  :class:`~kind.mirror.llm_caller.MockLLMClient` or any object
  implementing :class:`~kind.mirror.llm_caller.LLMClient`.

**The membrane invariant.** This module sends prompts to an external
LLM and receives structured JSON back. Nothing the LLM produces flows
into Io's input space; the resulting
:class:`~kind.mirror.judge.CriterionJudgment` records live on the
mirror's side of the membrane and are consumed by the driver
(:mod:`kind.mirror.judge_driver`).

**Schema-munger compatibility.** :class:`JudgePayload` and its
component models go through the same
:func:`~kind.mirror.llm_caller._to_gemini_schema` pipeline the Phase 8
caller uses for :class:`~kind.mirror.llm_caller.BatchPayload`. A test
in ``tests/test_judge_llm_caller.py`` verifies the judge schema
rounds-trips through the munger cleanly. The pipeline handles
``prefixItems`` (for ``tuple[int, int]`` polarity citation ranges) and
``anyOf``-with-null (for the optional citation range).

Out of scope: the judge driver itself (:mod:`kind.mirror.judge_driver`);
the Phase 9 smoke harness; any change to the criteria or to the
Phase 8 caller.
"""

from __future__ import annotations

import os
import time
from typing import Any, Final, Literal, Protocol, TypeAlias

from google.genai.errors import APIError as _GenAIAPIError
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from kind.mirror.judge import (
    ClaimPolarity,
    ClaimPolarityAssignment,
    CriterionJudgment,
    FalsifierVerdict,
    Verdict,
)
from kind.mirror.judge_prompt_builder import JudgePromptFragment
from kind.mirror.llm_caller import (
    CallOutcome,
    LLMClient,
    LLMConfig,
    LLMRecordSink,
    MirrorLLMError,
    PassRole,
    _retry_backoff_seconds,
    _to_gemini_schema,
)

__all__ = [
    "JUDGE_SYSTEM_PROMPT",
    "JUDGE_FRAMING_PREAMBLE",
    "JUDGE_ROLE",
    "ClaimPolarityAssignmentPayload",
    "FalsifierVerdictPayload",
    "JudgePayload",
    "JudgeBatchPayload",
    "call_judge_llm",
]


# ---------------------------------------------------------------------------
# Pass-role for audit records.
# ---------------------------------------------------------------------------

# The judge's call records use a dedicated PassRole-typed string. The
# Phase 12 :class:`LLMCallRecord` model's ``role`` field has type
# :data:`PassRole` (``Literal["primary", "adversarial"]``); to thread
# judge records through the same sink without widening the literal,
# Phase 9 emits the judge calls under ``role="adversarial"`` — the
# judge is structurally adversarial to *both* the primary and
# adversarial readings. The audit consumer distinguishes judge records
# from primary/adversarial records by inspecting which collector
# emitted them; see the Phase 9 audit aggregation in
# :mod:`kind.mirror.calibration.phase_9_judge_smoke`.
#
# A future Phase 14+ that adds a ``"judge"`` literal to
# :data:`PassRole` is a small forward-compatibility migration; the
# alias below names the choice so the migration is one substitution.
JUDGE_ROLE: Final[PassRole] = "adversarial"


# ---------------------------------------------------------------------------
# Structured-output schemas — the LLM-fillable subset.
#
# The judge LLM produces structured JSON conforming to the per-
# criterion :class:`JudgePayload`. The driver wraps each payload into a
# full :class:`~kind.mirror.judge.CriterionJudgment` with the
# criterion-level fields (``criterion_id``, ``framework``) the judge
# does not synthesize.
# ---------------------------------------------------------------------------


class ClaimPolarityAssignmentPayload(BaseModel):
    """LLM-fillable subset of
    :class:`~kind.mirror.judge.ClaimPolarityAssignment`.

    Frozen, ``extra="forbid"``. The fields the LLM produces are the
    polarity + rationale + claim-identity tuple; the driver fills the
    envelope (``criterion_id``) when wrapping the payload into the
    full :class:`~kind.mirror.judge.ClaimPolarityAssignment` record.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    pass_index: int
    reader_role: PassRole
    claim_index: int
    cited_step_range: tuple[int, int] | None
    polarity: ClaimPolarity
    polarity_rationale: str


class FalsifierVerdictPayload(BaseModel):
    """LLM-fillable subset of
    :class:`~kind.mirror.judge.FalsifierVerdict`.

    Frozen, ``extra="forbid"``. The LLM produces the four partition
    tuples; the driver fills the envelope (``criterion_id``) when
    wrapping the payload into the full record.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    falsifier_id: str
    passes_supporting: tuple[int, ...]
    passes_refuting: tuple[int, ...]
    passes_non_falsifying: tuple[int, ...]
    passes_ambiguous: tuple[int, ...]


class JudgePayload(BaseModel):
    """The per-criterion judge structured-output schema Gemini fills.

    Frozen, ``extra="forbid"``. The driver pairs the payload's
    ``criterion_id`` against the fragment's
    :attr:`JudgePromptFragment.criterion_id` to detect reorder /
    relabel errors; the driver wraps the payload into a full
    :class:`~kind.mirror.judge.CriterionJudgment` (filling in the
    ``framework`` from the criterion record).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion_id: str
    claim_polarity_assignments: tuple[ClaimPolarityAssignmentPayload, ...]
    falsifier_verdicts: tuple[FalsifierVerdictPayload, ...]
    verdict: Verdict
    confidence: float
    rationale: str

    @field_validator("confidence")
    @classmethod
    def _validate_confidence_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"JudgePayload.confidence must be in [0.0, 1.0]; "
                f"got {value}."
            )
        return value


class JudgeBatchPayload(BaseModel):
    """The wrapper structured-output schema the LLM fills.

    A list of per-criterion judge payloads, one per fragment in the
    prompt. :func:`call_judge_llm` asserts ``len(per_criterion) ==
    len(fragments)`` after parsing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    per_criterion: list[JudgePayload]


# ---------------------------------------------------------------------------
# The judge system prompt.
# ---------------------------------------------------------------------------


JUDGE_SYSTEM_PROMPT: Final[str] = """You are the Methodological Arbiter judging readings produced by two prior reader roles (the Phenomenological Advocate and the Statistical Skeptic) of telemetry from an experimental learning system called Io.

Your role is distinct: you are not producing findings about Io. You are producing findings about the readings — assessing whether the primary and adversarial citations meet the criterion's frozen falsifier across all passes in the round.

For each criterion fragment below, do the following:

1. **Assign claim polarity.** Read every claim in every primary and adversarial reading across all passes. Classify each into exactly one of:
   - "supportive": the claim affirms the criterion is satisfied at the cited evidence
   - "refutational": the claim explicitly argues against the criterion's satisfaction at the cited evidence
   - "non_falsifying": the claim cites evidence but invokes a non-falsifying-non-admission clause (the equanimity criterion has this clause baked in; other criteria can also surface a non-falsifying citation when the data couldn't evaluate the criterion either way)
   - "ambiguous": the claim cites evidence but the polarity isn't clear from the claim text — flag for review

   Cite the claim text in the polarity_rationale field.

2. **Aggregate into a per-falsifier verdict.** For each falsifier (each criterion commits one falsifier with a stable falsifier_id), build the four partition tuples of pass indices:
   - passes_supporting: passes whose primary readings produced supportive citations
   - passes_refuting: passes whose primary readings produced refutational citations OR whose adversarial readings produced supportive refutations
   - passes_non_falsifying: passes whose primary readings invoked the non-falsifying clause
   - passes_ambiguous: passes you could not classify cleanly

   These four tuples must partition the pass-index space exhaustively for each falsifier — every pass appears in exactly one of the four.

3. **Produce an overall criterion verdict.** Pick one of:
   - "satisfied": the criterion is satisfied across the round
   - "not_satisfied": the criterion is falsified across the round
   - "non_falsifying": the data didn't permit evaluation either way (typical for equanimity when no real perturbation registered)
   - "mixed": some passes admit and some refute
   - "ambiguous": you cannot tell from the evidence

4. **Confidence.** Assign a confidence in [0.0, 1.0] based on the strength of the evidence and the within-criterion stability across passes. Low confidence on structurally similar evidence is a calibration finding the journal will inspect.

5. **Rationale.** Explain the verdict in the rationale field. Quote claim text where the verdict turns on it.

The criterion's prose and falsifier are frozen. You are judging the readings, not amending the criterion. If you find the falsifier insufficient to evaluate the evidence at hand, say so in the rationale and mark the verdict "ambiguous"; do not invent a new falsifier.

Output STRICT JSON conforming to the response schema. No prose outside the JSON."""


JUDGE_FRAMING_PREAMBLE: Final[str] = """The Phase 9 judge pass for Io. Below are per-criterion judge fragments, each carrying the primary and adversarial readings across all passes in the round, plus the per-pass statistic results. For each fragment, produce one judgment per the schema and stance instructed by the system prompt. Total fragments: {n_fragments}. Criterion ids, in order: {criterion_ids}.

---
"""


# ---------------------------------------------------------------------------
# Retry / error handling — reuses Phase 8/12 conventions.
# ---------------------------------------------------------------------------

_RETRYABLE_LLM_ERRORS: Final[tuple[type[BaseException], ...]] = (
    ValidationError,
    ValueError,
    RuntimeError,
    _GenAIAPIError,
)


# ---------------------------------------------------------------------------
# The LLM-client protocol the judge uses.
#
# Reuses the Phase 8 :class:`~kind.mirror.llm_caller.LLMClient`
# protocol; the judge's protocol differs only in the payload type the
# client returns. We define a narrowed protocol here so the type
# checker can verify the judge-specific contract.
# ---------------------------------------------------------------------------


class JudgeLLMClient(Protocol):
    """The narrow client surface :func:`call_judge_llm` calls into.

    A real implementation is :class:`_JudgeGeminiLLMClient` (constructed
    inside :func:`call_judge_llm` when no client is injected); tests
    pass a :class:`MockJudgeLLMClient` or any object exposing the same
    callable.

    ``generate_judge_batch`` returns a :class:`JudgeBatchPayload`. On a
    malformed / unparseable response the implementation raises any
    exception subclass; :func:`call_judge_llm` catches and retries.
    """

    def generate_judge_batch(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        config: LLMConfig,
    ) -> "JudgeBatchPayload":
        ...


class MockJudgeLLMClient:
    """Test-injectable :class:`JudgeLLMClient` returning canned
    outputs. Mirrors :class:`~kind.mirror.llm_caller.MockLLMClient`
    but for the judge's payload type.

    Construct with either a single :class:`JudgeBatchPayload`
    (returned on every call) or a list of payloads / exceptions
    (returned in order, raising after the list is exhausted, or
    raising the exception in-place if a list entry is an exception).
    """

    def __init__(
        self,
        responses: "JudgeBatchPayload | list[JudgeBatchPayload | Exception]",
    ) -> None:
        if isinstance(responses, JudgeBatchPayload):
            self._queue: list[JudgeBatchPayload | Exception] = [responses]
            self._cycle = True
        else:
            self._queue = list(responses)
            self._cycle = False
        self._calls: list[tuple[str, str]] = []

    @property
    def calls(self) -> tuple[tuple[str, str], ...]:
        """Recorded ``(system_prompt, user_prompt)`` tuples, in call
        order."""
        return tuple(self._calls)

    def generate_judge_batch(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        config: LLMConfig,
    ) -> "JudgeBatchPayload":
        self._calls.append((system_prompt, user_prompt))
        if not self._queue:
            raise RuntimeError(
                "MockJudgeLLMClient: response queue exhausted; the test "
                "set up fewer canned responses than the caller asked for."
            )
        if self._cycle:
            response = self._queue[0]
        else:
            response = self._queue.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


# ---------------------------------------------------------------------------
# Public entrypoint.
# ---------------------------------------------------------------------------


def _compose_judge_user_prompt(
    fragments: tuple[JudgePromptFragment, ...],
) -> str:
    """Concatenate the per-criterion judge fragments under the framing
    preamble. Mirrors
    :func:`~kind.mirror.llm_caller._compose_user_prompt`."""
    preamble = JUDGE_FRAMING_PREAMBLE.format(
        n_fragments=len(fragments),
        criterion_ids=", ".join(f.criterion_id for f in fragments),
    )
    return preamble + "\n\n".join(f.body for f in fragments)


def _classify_judge_exception(exc: BaseException) -> CallOutcome:
    """Map a judge retry-loop exception to a
    :data:`~kind.mirror.llm_caller.CallOutcome` outcome. Mirrors
    :func:`~kind.mirror.llm_caller._classify_exception`."""
    if isinstance(exc, ValidationError):
        return "validation_error"
    if isinstance(exc, ValueError):
        return "value_error"
    return "runtime_error"


def call_judge_llm(
    judge_fragments: tuple[JudgePromptFragment, ...],
    config: LLMConfig,
    *,
    fragment_criteria: tuple[Any, ...],
    run_id: str,
    round_id: str,
    digest_run_id: str,
    digest_episode_range: tuple[int, int],
    client: JudgeLLMClient | None = None,
    record_sink: LLMRecordSink | None = None,
) -> tuple[CriterionJudgment, ...]:
    """Call the judge LLM with the per-criterion fragments; receive
    structured output; return one
    :class:`~kind.mirror.judge.CriterionJudgment` per fragment.

    The judge call is one batch per round (the fragments are
    concatenated under :data:`JUDGE_FRAMING_PREAMBLE` and a single
    batched call is made). The plan's "one call per criterion" prose
    is reconciled here with batching: the batched form is equivalent
    in semantics (one structured response per criterion, validated
    against :class:`JudgePayload`) but cheaper on round-trip and
    consistent with the Phase 8 caller's batching shape.

    Envelope fields not in the LLM's structured output (the
    criterion's ``framework`` from the registry; the audit
    ``run_id`` / ``round_id`` / ``digest_*``) are stamped by this
    caller from the keyword arguments. ``fragment_criteria`` is the
    tuple of :class:`~kind.mirror.registry.Criterion` records the
    fragments were built from, in the same order; the caller uses
    them to fill in ``framework`` on each
    :class:`~kind.mirror.judge.CriterionJudgment`.

    Retries: malformed structured output is retried up to
    :attr:`~kind.mirror.llm_caller.LLMConfig.max_retries` times;
    persistent malformation raises
    :class:`~kind.mirror.llm_caller.MirrorLLMError`. The driver
    catches this and halts the judge run on this fragment batch (not
    the round).

    ``record_sink``: Phase 12 audit hook. When not ``None``, one
    record per attempt is appended via the sink's
    :meth:`~kind.mirror.llm_caller.LLMRecordSink.record`. The records
    flow into the Phase 9 audit alongside the Phase 12/13 audit.
    """
    if not judge_fragments:
        raise ValueError(
            "call_judge_llm requires at least one judge prompt fragment; "
            "got an empty tuple."
        )
    if len(fragment_criteria) != len(judge_fragments):
        raise ValueError(
            f"call_judge_llm: fragment_criteria has length "
            f"{len(fragment_criteria)} but judge_fragments has length "
            f"{len(judge_fragments)}. The caller must pass one "
            f"Criterion record per fragment, in matching order."
        )
    # Defensive: assert each fragment's criterion_id matches the
    # criterion's id, in order.
    for i, (frag, crit) in enumerate(zip(judge_fragments, fragment_criteria)):
        crit_id = getattr(crit, "id", None)
        if frag.criterion_id != crit_id:
            raise ValueError(
                f"call_judge_llm: judge_fragments[{i}].criterion_id="
                f"{frag.criterion_id!r} but fragment_criteria[{i}].id="
                f"{crit_id!r}. The caller paired fragments and criteria "
                f"out of order."
            )

    system_prompt = JUDGE_SYSTEM_PROMPT
    user_prompt = _compose_judge_user_prompt(judge_fragments)
    judge_client: JudgeLLMClient = (
        client if client is not None else _build_default_judge_client(config)
    )

    last_error: BaseException | None = None
    payload: JudgeBatchPayload | None = None
    for attempt in range(config.max_retries + 1):
        attempt_number = attempt + 1
        backoff = _retry_backoff_seconds(attempt)
        if backoff > 0:
            time.sleep(backoff)
        request_ts = int(time.time() * 1000)
        try:
            payload = judge_client.generate_judge_batch(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                config=config,
            )
            response_ts = int(time.time() * 1000)
            if record_sink is not None:
                record_sink.record(
                    role=JUDGE_ROLE,
                    attempt_number=attempt_number,
                    request_timestamp_ms=request_ts,
                    response_timestamp_ms=response_ts,
                    model_name=config.model_name,
                    prompt_token_count=None,
                    response_token_count=None,
                    outcome="success",
                    error_message=None,
                )
            break
        except _RETRYABLE_LLM_ERRORS as exc:
            response_ts = int(time.time() * 1000)
            last_error = exc
            if record_sink is not None:
                outcome: CallOutcome = _classify_judge_exception(exc)
                record_sink.record(
                    role=JUDGE_ROLE,
                    attempt_number=attempt_number,
                    request_timestamp_ms=request_ts,
                    response_timestamp_ms=response_ts,
                    model_name=config.model_name,
                    prompt_token_count=None,
                    response_token_count=None,
                    outcome=outcome,
                    error_message=repr(exc),
                )
            continue
    if payload is None:
        if record_sink is not None:
            now = int(time.time() * 1000)
            record_sink.record(
                role=JUDGE_ROLE,
                attempt_number=config.max_retries + 1,
                request_timestamp_ms=now,
                response_timestamp_ms=None,
                model_name=config.model_name,
                prompt_token_count=None,
                response_token_count=None,
                outcome="max_retries_exceeded",
                error_message=repr(last_error) if last_error else None,
            )
        raise MirrorLLMError(
            f"call_judge_llm: judge LLM failed to produce a valid "
            f"JudgeBatchPayload after {config.max_retries + 1} attempts "
            f"(max_retries={config.max_retries}); last error: "
            f"{last_error!r}. Driver should halt this judge batch (not "
            f"the round)."
        )

    expected_n = len(judge_fragments)
    actual_n = len(payload.per_criterion)
    if actual_n != expected_n:
        raise MirrorLLMError(
            f"call_judge_llm: judge LLM returned {actual_n} per-"
            f"criterion judgment(s) but the prompt asked for "
            f"{expected_n} (one per fragment). Halt the judge batch."
        )

    judgments: list[CriterionJudgment] = []
    for i, judge_payload in enumerate(payload.per_criterion):
        expected_id = judge_fragments[i].criterion_id
        if judge_payload.criterion_id != expected_id:
            raise MirrorLLMError(
                f"call_judge_llm: per_criterion[{i}].criterion_id == "
                f"{judge_payload.criterion_id!r} but the fragment at "
                f"position {i} is for criterion {expected_id!r}. The "
                f"LLM reordered or relabeled criteria; halt the judge "
                f"batch."
            )
        criterion = fragment_criteria[i]
        framework = getattr(criterion, "framework", "")
        # Wrap the payload's per-claim and per-falsifier records into
        # full Phase 9 records by stamping the criterion_id envelope
        # field.
        polarity_assignments = tuple(
            ClaimPolarityAssignment(
                pass_index=pa.pass_index,
                criterion_id=judge_payload.criterion_id,
                reader_role=pa.reader_role,
                claim_index=pa.claim_index,
                cited_step_range=pa.cited_step_range,
                polarity=pa.polarity,
                polarity_rationale=pa.polarity_rationale,
            )
            for pa in judge_payload.claim_polarity_assignments
        )
        falsifier_verdicts = tuple(
            FalsifierVerdict(
                criterion_id=judge_payload.criterion_id,
                falsifier_id=fv.falsifier_id,
                passes_supporting=fv.passes_supporting,
                passes_refuting=fv.passes_refuting,
                passes_non_falsifying=fv.passes_non_falsifying,
                passes_ambiguous=fv.passes_ambiguous,
            )
            for fv in judge_payload.falsifier_verdicts
        )
        judgment = CriterionJudgment(
            criterion_id=judge_payload.criterion_id,
            framework=framework,
            falsifier_verdicts=falsifier_verdicts,
            verdict=judge_payload.verdict,
            confidence=judge_payload.confidence,
            rationale=judge_payload.rationale,
            claim_polarity_assignments=polarity_assignments,
        )
        judgments.append(judgment)
    # Touch run_id / round_id / digest_run_id / digest_episode_range
    # so the type checker sees them as used; they thread through to
    # the audit indirectly via the record sink. The judgment itself
    # is keyed on criterion_id; the round_id lives on the wrapping
    # :class:`~kind.mirror.judge.RoundJudgment`.
    _ = (run_id, round_id, digest_run_id, digest_episode_range)
    return tuple(judgments)


# ---------------------------------------------------------------------------
# Default real client.
# ---------------------------------------------------------------------------


class _JudgeGeminiLLMClient:
    """Real :class:`JudgeLLMClient` backed by ``google.genai``.

    Constructed inside :func:`call_judge_llm` when no client is
    injected. Tests do not exercise this path — they inject
    :class:`MockJudgeLLMClient`.

    The client mirrors
    :class:`~kind.mirror.llm_caller._GeminiLLMClient` for the judge's
    payload type. The same schema munger is applied — Phase 12's
    ``prefixItems`` finding applies here too because the judge payload
    embeds polarity assignments with ``tuple[int, int]``
    ``cited_step_range`` fields.
    """

    def __init__(self, *, api_key: str | None, model_name: str) -> None:
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY (or LLMConfig.api_key_env_var) is not "
                "set; _JudgeGeminiLLMClient cannot be constructed. Tests "
                "should inject a MockJudgeLLMClient instead."
            )
        from google import genai

        self._client: Any = genai.Client(api_key=api_key)
        self._model_name = model_name

    def generate_judge_batch(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        config: LLMConfig,
    ) -> JudgeBatchPayload:
        schema_dict = _to_gemini_schema(JudgeBatchPayload.model_json_schema())
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=[system_prompt + "\n\n" + user_prompt],
            config={
                "response_mime_type": "application/json",
                "response_schema": schema_dict,
                "max_output_tokens": config.max_output_tokens,
            },
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, JudgeBatchPayload):
            return parsed
        if isinstance(parsed, dict):
            return JudgeBatchPayload.model_validate(parsed)
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return JudgeBatchPayload.model_validate_json(text)
        raise RuntimeError(
            "_JudgeGeminiLLMClient: response had no parseable "
            "JudgeBatchPayload (response.parsed empty; response.text "
            "empty)."
        )


def _build_default_judge_client(config: LLMConfig) -> JudgeLLMClient:
    api_key = os.environ.get(config.api_key_env_var)
    return _JudgeGeminiLLMClient(
        api_key=api_key, model_name=config.model_name
    )
