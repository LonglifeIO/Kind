"""Phase 8 LLM-call layer — the narrow surface between Phase 8's
prompt fragments and the external model.

Phase 8 commits the LLM choice: **gemini-2.5-pro** is the default, the
same lineage Phase 0's :mod:`kind.mirror.caller` already uses. The choice
is methodological-independence from the Anthropic-heavy research workflow
the project leans on elsewhere; the journal entry records the call and
the structured-output schema version.

The caller is intentionally narrow:

- one entrypoint, :func:`call_mirror_llm`, takes the prompt fragments + a
  role + a typed config (and an optional injected client for tests), and
  returns one :class:`MirrorReading` per criterion;
- two role-specific system prompts: ``"primary"`` (the Phenomenological
  Advocate stance) and ``"adversarial"`` (the Statistical Skeptic stance);
- bounded retries on malformed structured output, capped at 3 by default;
  persistent malformation raises :class:`MirrorLLMError` so the
  orchestrator can halt the pass (not the run);
- no real API calls in tests — tests inject a :class:`MockLLMClient` or
  any object implementing the :class:`LLMClient` protocol.

**The membrane invariant.** This module sends prompts to an external LLM
and receives structured JSON back. Nothing the LLM produces flows into
Io's input space; the resulting :class:`MirrorReading` records live on
the mirror's side of the membrane and are consumed by the pass driver
(Part 5).

Out of scope: any change to Phase 0's
:class:`~kind.mirror.structured.StructuredReading` schema; the pass
driver itself (Part 5); aggregation across passes.
"""

from __future__ import annotations

import os
import time
from typing import Any, Final, Literal, Protocol, TypeAlias

from google.genai.errors import APIError as _GenAIAPIError
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from kind.mirror.prompt_builder import PromptFragment
from kind.mirror.structured import (
    MIRROR_READING_V2_VERSION,
    BaselineFlag,
    FrameworkAnchor,
    ReaderRole,
    StructuredClaim,
    StructuredReading,
)

__all__ = [
    "MirrorLLMError",
    "MirrorReading",
    "PassRole",
    "CallOutcome",
    "LLMConfig",
    "LLMClient",
    "LLMRecordSink",
    "MockLLMClient",
    "PrimaryReadingPayload",
    "AdversarialReadingPayload",
    "BatchPayload",
    "PRIMARY_SYSTEM_PROMPT",
    "ADVERSARIAL_SYSTEM_PROMPT",
    "FRAMING_PREAMBLE",
    "call_mirror_llm",
]


# ---------------------------------------------------------------------------
# Type aliases and the role literal.
# ---------------------------------------------------------------------------

# Per the Phase 8 spec: ``MirrorReading`` is an alias for Phase 0's
# :class:`~kind.mirror.structured.StructuredReading`. The LLM produces
# structured JSON conforming to a per-criterion subset (see
# :class:`PrimaryReadingPayload` / :class:`AdversarialReadingPayload` below);
# the caller wraps each into a full :class:`MirrorReading` with envelope
# fields from the orchestrator-supplied context.
MirrorReading: TypeAlias = StructuredReading

# The role names used at this module's API surface. Mapped to
# :class:`~kind.mirror.structured.ReaderRole` values internally
# (``primary → "advocate"``; ``adversarial → "skeptic"``) — see the
# discussion in the module docstring.
PassRole: TypeAlias = Literal["primary", "adversarial"]


# The five outcomes a Phase 12 :class:`LLMRecordSink` may receive. Lives
# here (rather than in :mod:`kind.mirror.calibration.llm_audit`) so the
# protocol type signature avoids a back-import; the audit module
# re-exports the alias for callers that import from there.
CallOutcome: TypeAlias = Literal[
    "success",
    "validation_error",
    "value_error",
    "runtime_error",
    "max_retries_exceeded",
]


_ROLE_TO_READER_ROLE: Final[dict[PassRole, ReaderRole]] = {
    "primary": "advocate",
    "adversarial": "skeptic",
}


# Phase 12 finding (journaled): Gemini's SDK occasionally surfaces
# transient errors past its own tenacity-backed retry layer (5xx
# UNAVAILABLE, 429 rate-limited). These subclass
# :class:`google.genai.errors.APIError` (→ ``Exception``), not
# ``RuntimeError``, so the caller's retry loop must catch APIError
# explicitly to retry within its budget. Programming bugs (TypeError,
# AttributeError, etc.) are *not* in the retryable set — they should
# surface immediately.
_RETRYABLE_LLM_ERRORS: Final[tuple[type[BaseException], ...]] = (
    ValidationError,
    ValueError,
    RuntimeError,
    _GenAIAPIError,
)


# Backoff schedule between retries. Phase 12 finding: hammering on a
# 503 doesn't help — exponential up to a 30s cap matches the SDK's
# own tenacity defaults and gives sustained outages a chance to clear.
def _retry_backoff_seconds(attempt_index: int) -> float:
    """Backoff before retry ``attempt_index`` (0-indexed).

    The first retry (index 1) waits 2s; each subsequent retry doubles
    up to a 30s cap. Index 0 (the original call) returns 0 — no wait.
    """
    if attempt_index <= 0:
        return 0.0
    return min(2.0 ** attempt_index, 30.0)


# ---------------------------------------------------------------------------
# Errors.
# ---------------------------------------------------------------------------


class MirrorLLMError(RuntimeError):
    """Raised when the LLM call cannot produce a valid structured reading
    within :attr:`LLMConfig.max_retries`. The orchestrator halts the
    *pass* (not the run) on this error; the run continues, the pass
    result records the failure, and a future pass on the same checkpoint
    can be retried.
    """


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------


class LLMConfig(BaseModel):
    """Phase 8 LLM caller configuration.

    Frozen. The Phase 8 commitment is journaled here at the field
    defaults:

    - ``model_name``: ``"gemini-2.5-pro"`` is the project's settled
      mirror-layer choice (Phase 0's caller uses the same lineage).
    - ``max_output_tokens``: ``8192`` budget per call. The structured
      output for ~3 criteria of ~5 claims each typically fits in ~2-3K
      tokens; the headroom is for the Skeptic's longer free-text notes.
    - ``max_retries``: ``3`` retries on malformed structured output; the
      orchestrator halts the pass on a fourth failure.
    - ``api_key_env_var``: ``"GEMINI_API_KEY"``; tests bypass entirely by
      injecting a :class:`MockLLMClient`.

    **Phase 10 addition: ``temperature`` and ``seed``.** Both optional;
    when ``None`` the SDK's defaults apply (Gemini's structured-output
    path uses a low default temperature ≈ 0). When set, they flow into
    the Gemini ``generate_content`` config dict. Phase 10's stability
    runner sets ``temperature=0.7`` and varies ``seed`` to produce
    independent reseed calls; the rest of the system uses the defaults.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str = "gemini-2.5-pro"
    max_output_tokens: int = 8192
    max_retries: int = 3
    api_key_env_var: str = "GEMINI_API_KEY"
    temperature: float | None = None
    seed: int | None = None

    @field_validator("model_name", "api_key_env_var")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty.")
        return value

    @field_validator("max_output_tokens", "max_retries")
    @classmethod
    def _validate_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"value must be positive; got {value}.")
        return value

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not 0.0 <= value <= 2.0:
            raise ValueError(
                f"LLMConfig.temperature must be in [0.0, 2.0]; got {value}."
            )
        return value

    @field_validator("seed")
    @classmethod
    def _validate_seed(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value < 0:
            raise ValueError(
                f"LLMConfig.seed must be non-negative; got {value}."
            )
        return value


# ---------------------------------------------------------------------------
# Structured-output schemas — the LLM-fillable subset.
#
# Each per-criterion reading the LLM produces fills these. The full
# :class:`MirrorReading` (=:class:`StructuredReading`) wraps the payload
# with envelope fields (``run_id``, ``timestamp_ms``, ``reader_role``,
# ``paired_reading_id``, ``baseline_flag``, ``digest_run_id``,
# ``digest_episode_range``, ``schema_version``) set by :func:`call_mirror_llm`.
# ---------------------------------------------------------------------------


class _PerCriterionReadingPayload(BaseModel):
    """Common subset shared by primary and adversarial per-criterion payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion_id: str
    framework_anchor: FrameworkAnchor
    claims: list[StructuredClaim]
    free_text_notes: str


class PrimaryReadingPayload(_PerCriterionReadingPayload):
    """The primary (Phenomenological Advocate) per-criterion payload."""


class AdversarialReadingPayload(_PerCriterionReadingPayload):
    """The adversarial (Statistical Skeptic) per-criterion payload."""


class BatchPayload(BaseModel):
    """The wrapper structured-output schema the LLM fills.

    A list of per-criterion payloads, one per fragment in the prompt.
    The caller asserts ``len(per_criterion) == len(fragments)`` after
    parsing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    per_criterion: list[_PerCriterionReadingPayload]


# ---------------------------------------------------------------------------
# Prompts.
# ---------------------------------------------------------------------------


PRIMARY_SYSTEM_PROMPT: Final[str] = """You are reading telemetry from an experimental learning system called Io, for a per-criterion structured reading.

For each criterion fragment below, evaluate honestly whether the criterion is satisfied at the named reading surfaces. Your stance is the Phenomenological Advocate: you take seriously the possibility that the criterion is satisfied, and you ground every claim in the computed statistic results the fragment names.

For each criterion produce one reading containing:
- criterion_id: the criterion's id
- framework_anchor: "buddhist_phenomenology" for reflexive_attention and equanimity_perturbation_recovery; "predictive_processing" or "null_statistics" only if the criterion specifies it
- claims: a list of StructuredClaim entries, each grounded in a (cited_stream, cited_run_id, cited_episode_range or cited_step_range, cited_scalar_field, cited_value) tuple. Set reading_surface to the surface this claim reads at. Set masked_steps_handling to "n/a" unless the citation aggregates self_prediction_error_t over a step range.
- free_text_notes: any relevant context, anomalies, or framing that doesn't fit cleanly into a single StructuredClaim.

Discipline:
- Do not project intent or self-modeling. Reflexive attention is about latent-state coupling, not introspective access.
- For equanimity: the non-falsifying-non-admission clause is load-bearing. A flat trajectory is NOT equanimity.
- Do not make claims you cannot ground in the computed signals or the telemetry the fragment names.

Output STRICT JSON conforming to the response schema. No prose outside the JSON."""


ADVERSARIAL_SYSTEM_PROMPT: Final[str] = """You are reading telemetry from an experimental learning system called Io, as the adversarial Statistical Skeptic to a primary reading.

For each criterion fragment below, argue against the criterion being satisfied at the named reading surfaces. Your stance is the null-hypothesis-first reader: the most plausible explanation for any observed structure is the null, and the criterion is satisfied only when the null is overwhelmed by specific cited evidence.

For each criterion produce one reading containing:
- criterion_id: the criterion's id
- framework_anchor: "null_statistics" at substrate-side and behavior-side; "predictive_processing" at head-internal (where the auxiliary-target-tracking refutation lives)
- claims: a list of StructuredClaim entries that cite the most plausible null hypothesis the fragment's signals admit. Each claim grounds in a (cited_stream, cited_run_id, cited_episode_range or cited_step_range, cited_scalar_field, cited_value) tuple. Set reading_surface; set masked_steps_handling per the same rule.
- free_text_notes: name the specific refutation candidates the fragment surfaces (the shuffled-time control, the observation-only baseline, the four exclusions for second-order volition, the non-falsifying-non-admission clause for equanimity).

Discipline:
- Pure flatness on the equanimity signals is the most likely null, not an admission of equanimity.
- A latent-regime contrast that does not exceed the observation-only baseline is the null at second-order volition.
- A within-h_t autocorrelation that does not exceed the shuffled-time control is the null at reflexive attention.
- Sham perturbations: an equanimity admission at a sham timestamp is a calibration failure; flag it.

Output STRICT JSON conforming to the response schema. No prose outside the JSON."""


FRAMING_PREAMBLE: Final[str] = """The Phase 8 adversarial pass for Io. Below are per-criterion prompt fragments. For each fragment, produce one reading per the schema and stance instructed by the system prompt. Total fragments: {n_fragments}. Criterion ids, in order: {criterion_ids}.

---
"""


# ---------------------------------------------------------------------------
# The LLM-client protocol and the in-test mock.
# ---------------------------------------------------------------------------


class LLMClient(Protocol):
    """The narrow client surface :func:`call_mirror_llm` calls into.

    A real implementation is :class:`GeminiLLMClient` (constructed inside
    :func:`call_mirror_llm` when no client is injected); tests pass a
    :class:`MockLLMClient` or any object exposing the same callable.

    ``generate_batch`` returns a :class:`BatchPayload`. On a malformed /
    unparseable response the implementation raises any exception
    subclass; :func:`call_mirror_llm` catches and retries.

    Phase 12 addition: ``generate_batch`` may also return a
    :class:`BatchResponse` (or any object exposing the ``payload`` and
    ``usage_metadata`` attributes via duck-typing) — the caller checks
    ``isinstance(result, BatchPayload)`` first and falls back to
    treating the result as an attribute-bearing container so it can
    extract token counts for the :class:`LLMRecordSink`. Implementations
    that don't track tokens (the mock client) keep returning a plain
    :class:`BatchPayload`; the audit records' token-count fields stay
    ``None``.
    """

    def generate_batch(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        config: LLMConfig,
    ) -> BatchPayload:
        ...


class LLMRecordSink(Protocol):
    """The Phase 12 audit-collector protocol :func:`call_mirror_llm`
    emits records into.

    Implemented by
    :class:`~kind.mirror.calibration.llm_audit.LLMCallRecordCollector`
    in the calibration package; defined here as a structural protocol so
    :mod:`kind.mirror.llm_caller` doesn't pull the calibration package
    in. The caller emits one record per attempt: success, retry, or
    final ``max_retries_exceeded`` synthetic record. The protocol's
    return type is :class:`object` so an implementation can return the
    appended record (the collector does) or ``None`` — the caller
    discards the return value.
    """

    def record(
        self,
        *,
        role: PassRole,
        attempt_number: int,
        request_timestamp_ms: int,
        response_timestamp_ms: int | None,
        model_name: str,
        prompt_token_count: int | None,
        response_token_count: int | None,
        outcome: CallOutcome,
        error_message: str | None,
    ) -> object:
        ...


class MockLLMClient:
    """Test-injectable :class:`LLMClient` returning canned outputs.

    Construct with either a single :class:`BatchPayload` (returned on
    every call) or a list of payloads (returned in order, raising after
    the list is exhausted). The :class:`Exception` slot supports the
    malformed-output retry-path test: if a position holds an exception
    instance, the mock raises it from that call instead of returning a
    payload; the next call advances to the next position.
    """

    def __init__(
        self,
        responses: BatchPayload | list[BatchPayload | Exception],
    ) -> None:
        if isinstance(responses, BatchPayload):
            self._queue: list[BatchPayload | Exception] = [responses]
            self._cycle = True
        else:
            self._queue = list(responses)
            self._cycle = False
        self._calls: list[tuple[str, str]] = []
        self._configs: list[LLMConfig] = []

    @property
    def calls(self) -> tuple[tuple[str, str], ...]:
        """Recorded ``(system_prompt, user_prompt)`` tuples, in call order.

        Tests use this to assert role-specific system prompt selection
        (one of :data:`PRIMARY_SYSTEM_PROMPT` /
        :data:`ADVERSARIAL_SYSTEM_PROMPT` shows up under the right role).
        """
        return tuple(self._calls)

    @property
    def configs(self) -> tuple[LLMConfig, ...]:
        """Recorded :class:`LLMConfig` per call, in call order.

        Phase 10 stability tests use this to verify seed/temperature
        propagation: a reseed call should arrive with the expected
        ``seed`` and ``temperature`` set on the derived config.
        """
        return tuple(self._configs)

    def generate_batch(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        config: LLMConfig,
    ) -> BatchPayload:
        self._calls.append((system_prompt, user_prompt))
        self._configs.append(config)
        if not self._queue:
            raise RuntimeError(
                "MockLLMClient: response queue exhausted; the test set up "
                "fewer canned responses than the caller asked for."
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


def _compose_user_prompt(fragments: tuple[PromptFragment, ...]) -> str:
    """Concatenate the per-criterion fragments under the framing preamble."""
    preamble = FRAMING_PREAMBLE.format(
        n_fragments=len(fragments),
        criterion_ids=", ".join(f.criterion_id for f in fragments),
    )
    return preamble + "\n\n".join(f.body for f in fragments)


def _select_system_prompt(role: PassRole) -> str:
    if role == "primary":
        return PRIMARY_SYSTEM_PROMPT
    if role == "adversarial":
        return ADVERSARIAL_SYSTEM_PROMPT
    raise ValueError(  # pragma: no cover — Literal exhausts; defensive
        f"unknown role {role!r}; expected 'primary' or 'adversarial'."
    )


def call_mirror_llm(
    fragments: tuple[PromptFragment, ...],
    role: PassRole,
    config: LLMConfig,
    *,
    run_id: str,
    digest_run_id: str,
    digest_episode_range: tuple[int, int],
    paired_reading_id: str | None = None,
    baseline_flag: BaselineFlag = "genuine",
    client: LLMClient | None = None,
    record_sink: LLMRecordSink | None = None,
    seed: int | None = None,
    temperature: float | None = None,
) -> tuple[MirrorReading, ...]:
    """Call the LLM with the concatenated per-criterion fragments; receive
    structured output; return one :class:`MirrorReading` per criterion.

    Role selection: ``"primary"`` uses :data:`PRIMARY_SYSTEM_PROMPT` and
    sets the reading's ``reader_role`` to ``"advocate"``;
    ``"adversarial"`` uses :data:`ADVERSARIAL_SYSTEM_PROMPT` and sets
    ``reader_role`` to ``"skeptic"``. The two system prompts are
    explicitly different documents — argue-against-primary is structural,
    not a paraphrase.

    Retries: malformed structured output is retried up to
    :attr:`LLMConfig.max_retries` times; persistent malformation raises
    :class:`MirrorLLMError`. The orchestrator catches this and halts the
    pass (not the run).

    Envelope fields not in the LLM's structured output (``run_id``,
    ``timestamp_ms``, ``reader_role``, ``paired_reading_id``,
    ``baseline_flag``, ``digest_run_id``, ``digest_episode_range``,
    ``schema_version``) are stamped by this caller from the keyword
    arguments. The LLM produces only the
    ``(criterion_id, framework_anchor, claims, free_text_notes)`` part of
    each reading; everything else is orchestration-side.

    ``client``: if not provided, a :class:`GeminiLLMClient` is
    constructed from :attr:`LLMConfig.model_name` /
    :attr:`LLMConfig.api_key_env_var`. Tests inject a
    :class:`MockLLMClient` (no real API call).

    ``record_sink``: Phase 12 audit hook. When not ``None``, one record
    per attempt is appended (success, validation_error, value_error,
    runtime_error) plus one synthetic ``max_retries_exceeded`` record
    on retry-budget exhaustion. The sink's protocol is
    :class:`LLMRecordSink`; the calibration package's
    :class:`~kind.mirror.calibration.llm_audit.LLMCallRecordCollector`
    is the production implementation.

    **Phase 10 addition: ``seed`` and ``temperature``.** When set, they
    derive a copy of ``config`` with the corresponding field overridden
    and pass that derived config to the client. Either parameter left
    as ``None`` leaves the config's existing value (which itself
    defaults to ``None`` — the SDK's defaults). Phase 10's stability
    runner threads independent ``seed`` values plus a fixed elevated
    ``temperature`` through reseed calls; Phase 8 callers never set
    either.
    """
    if not fragments:
        raise ValueError(
            "call_mirror_llm requires at least one prompt fragment; "
            "got an empty tuple."
        )
    if seed is not None or temperature is not None:
        overrides: dict[str, Any] = {}
        if seed is not None:
            overrides["seed"] = seed
        if temperature is not None:
            overrides["temperature"] = temperature
        config = config.model_copy(update=overrides)
    system_prompt = _select_system_prompt(role)
    user_prompt = _compose_user_prompt(fragments)
    llm_client: LLMClient = (
        client if client is not None else _build_default_client(config)
    )

    last_error: BaseException | None = None
    payload: BatchPayload | None = None
    for attempt in range(config.max_retries + 1):
        attempt_number = attempt + 1
        # Phase 12: backoff before retry attempts (not before the
        # original call). The SDK's own tenacity layer already backs
        # off; this is the second-line defense for errors that pass
        # through it.
        backoff = _retry_backoff_seconds(attempt)
        if backoff > 0:
            time.sleep(backoff)
        request_ts = int(time.time() * 1000)
        try:
            payload = llm_client.generate_batch(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                config=config,
            )
            response_ts = int(time.time() * 1000)
            if record_sink is not None:
                tokens_in, tokens_out = _extract_token_counts(payload)
                record_sink.record(
                    role=role,
                    attempt_number=attempt_number,
                    request_timestamp_ms=request_ts,
                    response_timestamp_ms=response_ts,
                    model_name=config.model_name,
                    prompt_token_count=tokens_in,
                    response_token_count=tokens_out,
                    outcome="success",
                    error_message=None,
                )
            break
        except _RETRYABLE_LLM_ERRORS as exc:
            response_ts = int(time.time() * 1000)
            last_error = exc
            if record_sink is not None:
                outcome: CallOutcome = _classify_exception(exc)
                record_sink.record(
                    role=role,
                    attempt_number=attempt_number,
                    request_timestamp_ms=request_ts,
                    response_timestamp_ms=response_ts,
                    model_name=config.model_name,
                    prompt_token_count=None,
                    response_token_count=None,
                    outcome=outcome,
                    error_message=repr(exc),
                )
            # Retry; the next iteration sends the same prompts after
            # the backoff above.
            continue
    if payload is None:
        if record_sink is not None:
            now = int(time.time() * 1000)
            record_sink.record(
                role=role,
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
            f"call_mirror_llm: LLM failed to produce a valid BatchPayload "
            f"after {config.max_retries + 1} attempts (max_retries="
            f"{config.max_retries}); last error: {last_error!r}. Orchestrator "
            f"should halt this pass (not the run)."
        )

    expected_n = len(fragments)
    actual_n = len(payload.per_criterion)
    if actual_n != expected_n:
        raise MirrorLLMError(
            f"call_mirror_llm: LLM returned {actual_n} per-criterion "
            f"reading(s) but the prompt asked for {expected_n} (one per "
            f"fragment). Halt the pass."
        )

    reader_role = _ROLE_TO_READER_ROLE[role]
    now_ms = int(time.time() * 1000)
    readings: list[MirrorReading] = []
    for i, per_criterion in enumerate(payload.per_criterion):
        expected_id = fragments[i].criterion_id
        if per_criterion.criterion_id != expected_id:
            raise MirrorLLMError(
                f"call_mirror_llm: per_criterion[{i}].criterion_id == "
                f"{per_criterion.criterion_id!r} but the fragment at "
                f"position {i} is for criterion {expected_id!r}. The LLM "
                f"reordered or relabeled criteria; halt the pass."
            )
        reading = MirrorReading(
            schema_version=MIRROR_READING_V2_VERSION,
            run_id=run_id,
            timestamp_ms=now_ms,
            reader_role=reader_role,
            paired_reading_id=paired_reading_id,
            framework_anchor=per_criterion.framework_anchor,
            baseline_flag=baseline_flag,
            digest_run_id=digest_run_id,
            digest_episode_range=digest_episode_range,
            claims=list(per_criterion.claims),
            free_text_notes=per_criterion.free_text_notes,
        )
        readings.append(reading)
    return tuple(readings)


# ---------------------------------------------------------------------------
# Default real client.
# ---------------------------------------------------------------------------


class _GeminiLLMClient:
    """Real :class:`LLMClient` backed by ``google.genai``.

    Constructed inside :func:`call_mirror_llm` when no client is
    injected. Tests do not exercise this path — they inject
    :class:`MockLLMClient`.

    The client is intentionally minimal: one method, one call, structured
    output via the SDK's ``response_schema`` mechanism (same pattern as
    Phase 0's :mod:`kind.mirror.caller`).
    """

    def __init__(self, *, api_key: str | None, model_name: str) -> None:
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY (or LLMConfig.api_key_env_var) is not set; "
                "GeminiLLMClient cannot be constructed. Tests should inject "
                "a MockLLMClient instead."
            )
        # Imported lazily so a test that injects a MockLLMClient never
        # pulls in google.genai's import side effects.
        from google import genai

        self._client: Any = genai.Client(api_key=api_key)
        self._model_name = model_name

    def generate_batch(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        config: LLMConfig,
    ) -> BatchPayload:
        # Phase 12 finding (journaled): Gemini's structured-output Schema
        # validator implements OpenAPI 3.0, not JSON Schema 2020-12, and
        # rejects ``prefixItems`` with "Extra inputs are not permitted".
        # Pydantic emits ``prefixItems`` for ``tuple[int, int]``
        # (``StructuredClaim.cited_episode_range`` and
        # ``cited_step_range``). Rather than relax the Phase-0
        # ``StructuredClaim`` contract — the "do not relax the schema
        # beyond what the criterion's prose justifies" plan constraint
        # — we munge the schema dict at SDK-call time:
        # ``prefixItems`` → ``items`` (fixed length is preserved by the
        # ``minItems`` / ``maxItems`` Pydantic already emits). On
        # response parse, ``BatchPayload.model_validate`` coerces the
        # arriving list of ints back to a tuple per Pydantic's normal
        # collection coercion. The schema sent to Gemini differs from
        # the schema the response is validated against; the difference
        # is the prefixItems → items conversion only.
        schema_dict = _to_gemini_schema(BatchPayload.model_json_schema())
        generation_config: dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": schema_dict,
            "max_output_tokens": config.max_output_tokens,
        }
        if config.temperature is not None:
            generation_config["temperature"] = config.temperature
        if config.seed is not None:
            generation_config["seed"] = config.seed
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=[system_prompt + "\n\n" + user_prompt],
            config=generation_config,
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, BatchPayload):
            return parsed
        if isinstance(parsed, dict):
            return BatchPayload.model_validate(parsed)
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return BatchPayload.model_validate_json(text)
        raise RuntimeError(
            "GeminiLLMClient: response had no parseable BatchPayload "
            "(response.parsed empty; response.text empty)."
        )


def _build_default_client(config: LLMConfig) -> LLMClient:
    api_key = os.environ.get(config.api_key_env_var)
    return _GeminiLLMClient(api_key=api_key, model_name=config.model_name)


# JSON Schema keys Gemini's response_schema validator does NOT accept,
# per https://ai.google.dev/gemini-api/docs/structured-output. The set
# is the residual after we apply the structural transformations
# (prefixItems → items, anyOf-with-null → nullable, $ref inlining); any
# remaining key in this set is dropped by the munger.
_GEMINI_DROP_KEYS: Final[frozenset[str]] = frozenset(
    {
        "$schema",
        "$id",
        "title",
        "additionalProperties",
        "definitions",
        "default",
        "examples",
        "const",
        # ``allOf``/``not``/``oneOf`` aren't in our schemas at present,
        # but the drop-list includes them defensively for forward
        # compatibility with future Pydantic emission changes.
        "allOf",
        "not",
        "oneOf",
        "discriminator",
    }
)


def _resolve_ref(ref: str, defs: dict[str, Any]) -> Any:
    """Resolve a ``$ref`` like ``"#/$defs/Foo"`` into the dict at
    ``defs["Foo"]``. Raises ``KeyError`` if the ref isn't a local
    ``$defs`` reference (Phase 12 doesn't need wider ref support)."""
    prefix = "#/$defs/"
    if not ref.startswith(prefix):
        raise KeyError(
            f"_to_gemini_schema: unsupported $ref form {ref!r}; only "
            f"local $defs refs ({prefix}…) are inlined."
        )
    name = ref[len(prefix):]
    return defs[name]


def _inline_refs(node: Any, defs: dict[str, Any]) -> Any:
    """Recursively replace every ``{"$ref": "#/$defs/X"}`` with a deep
    copy of ``defs["X"]``, walking into the resolved sub-tree so its
    own refs also resolve. Cycles aren't expected in our schema; if one
    appeared this would recurse forever, by design (the upstream
    Pydantic models are acyclic)."""
    if isinstance(node, dict):
        if "$ref" in node and isinstance(node["$ref"], str):
            resolved = _resolve_ref(node["$ref"], defs)
            return _inline_refs(resolved, defs)
        return {k: _inline_refs(v, defs) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline_refs(item, defs) for item in node]
    return node


def _convert_anyof_nullable(node: Any) -> Any:
    """Convert ``anyOf: [<schema>, {"type": "null"}]`` to ``<schema>``
    with ``nullable: true``. Pydantic emits this anyOf shape for
    ``Optional[X]`` fields; Gemini's Schema doesn't accept ``anyOf``
    but does accept ``nullable``.

    A two-element anyOf with one null branch is converted in place; an
    anyOf with three+ branches or no null branch is left alone (and
    will likely be rejected by Gemini, but the build's current schemas
    don't have such cases — flag if a future field introduces one).
    """
    if isinstance(node, dict):
        if "anyOf" in node and isinstance(node["anyOf"], list):
            branches = node["anyOf"]
            null_branches = [
                b for b in branches if isinstance(b, dict) and b.get("type") == "null"
            ]
            non_null = [
                b for b in branches if not (isinstance(b, dict) and b.get("type") == "null")
            ]
            if len(null_branches) == 1 and len(non_null) == 1:
                # Optional[X] form — fold into nullable.
                merged: dict[str, Any] = {}
                for k, v in node.items():
                    if k == "anyOf":
                        continue
                    merged[k] = v
                inner = non_null[0]
                if isinstance(inner, dict):
                    for k, v in inner.items():
                        merged.setdefault(k, v)
                merged["nullable"] = True
                return _convert_anyof_nullable(merged)
        return {k: _convert_anyof_nullable(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_convert_anyof_nullable(item) for item in node]
    return node


def _convert_prefix_items(node: Any) -> Any:
    """Recursively convert ``prefixItems`` (JSON Schema 2020-12
    fixed-length tuple typing) → ``items`` (single-type-per-array
    OpenAPI 3.0 form). The minItems / maxItems Pydantic emits already
    pin the length; the loss is the per-position type heterogeneity,
    which our schemas don't use (every prefixItems is a homogeneous
    int pair from ``StructuredClaim.cited_episode_range`` /
    ``cited_step_range``)."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k == "prefixItems" and isinstance(v, list) and v:
                out["items"] = _convert_prefix_items(v[0])
                continue
            out[k] = _convert_prefix_items(v)
        return out
    if isinstance(node, list):
        return [_convert_prefix_items(item) for item in node]
    return node


def _strip_unsupported_keys(node: Any) -> Any:
    """Drop the schema keys in :data:`_GEMINI_DROP_KEYS` from every
    nested dict. The drop is silent — Pydantic emits these keys for
    correctness or for OpenAPI tooling, but Gemini's structured-output
    Schema validator rejects unknown fields hard."""
    if isinstance(node, dict):
        return {
            k: _strip_unsupported_keys(v)
            for k, v in node.items()
            if k not in _GEMINI_DROP_KEYS
        }
    if isinstance(node, list):
        return [_strip_unsupported_keys(item) for item in node]
    return node


def _drop_faithfulness_status(node: Any) -> Any:
    """Recursively remove the ``faithfulness_status`` property from every
    object schema.

    Phase 12.5 item 1: ``faithfulness_status`` is the faithfulness
    verifier's field, not the reading LLM's. Phase 10's smoke saw the
    LLM pre-fill it (``"resolved"`` on 4 of 6 readings); the verifier
    overwrites the verdict regardless, but the LLM should not be asked
    to write a field it does not own. Dropping the property from the
    structured-output JSON schema is structural enforcement — Gemini's
    validator constrains against the schema, so a dropped property is
    one the model is never asked to produce. Only
    :class:`~kind.mirror.structured.StructuredClaim` carries this field,
    so the removal is effectively targeted even though the walk is
    generic. On response parse, ``StructuredClaim.model_validate``
    supplies the field's writer-side default (``"not_checked"``).

    The property is removed from every ``properties`` dict and from
    every ``required`` list (the field carries a default, so it is
    normally absent from ``required`` already — the list-strip is
    defensive)."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k == "properties" and isinstance(v, dict):
                out[k] = {
                    pk: _drop_faithfulness_status(pv)
                    for pk, pv in v.items()
                    if pk != "faithfulness_status"
                }
                continue
            if k == "required" and isinstance(v, list):
                out[k] = [r for r in v if r != "faithfulness_status"]
                continue
            out[k] = _drop_faithfulness_status(v)
        return out
    if isinstance(node, list):
        return [_drop_faithfulness_status(item) for item in node]
    return node


def _to_gemini_schema(schema: Any) -> Any:
    """Convert a Pydantic-emitted JSON Schema dict to a form Gemini's
    ``response_schema`` validator accepts.

    The pipeline is sequential and idempotent:

    0. **Drop ``faithfulness_status``** (Phase 12.5 item 1): the
       faithfulness verifier owns that field; the reading LLM is not
       asked to fill it, so the property is removed from every object
       schema before anything else runs.
    1. **Inline ``$ref``**: Pydantic emits ``$defs`` + ``$ref`` for
       shared / nested models. Gemini's Schema does not support refs;
       inline each ref with a deep copy of its ``$defs`` target. After
       this pass ``$defs`` is dropped.
    2. **Convert ``anyOf`` with null branch → ``nullable``**: Pydantic
       emits ``anyOf: [X, {"type":"null"}]`` for ``Optional[X]``;
       Gemini accepts ``nullable: true`` instead.
    3. **Convert ``prefixItems`` → ``items``**: Phase 12 finding #1.
    4. **Strip unsupported top-level keys**:
       :data:`_GEMINI_DROP_KEYS` (``$schema``, ``title``,
       ``additionalProperties``, etc.).

    The conversion is lossy in three ways the build accepts: the loss
    of ``prefixItems`` heterogeneity (length still pinned), the loss
    of ``additionalProperties: false`` (Gemini doesn't add unsolicited
    fields anyway), and the loss of ``title`` (informational only).
    The response parser ``BatchPayload.model_validate`` re-asserts the
    full Phase-0 contract, so any divergence between what Gemini was
    told and what the response must satisfy surfaces as a
    ``ValidationError`` and trips the retry loop — the conversion's
    looseness can only cost extra retries, not silent correctness
    failures.
    """
    if not isinstance(schema, dict):
        return schema
    fields_dropped = _drop_faithfulness_status(schema)
    defs_raw = fields_dropped.get("$defs")
    defs: dict[str, Any] = (
        defs_raw if isinstance(defs_raw, dict) else {}
    )
    inlined = _inline_refs(fields_dropped, defs)
    if isinstance(inlined, dict) and "$defs" in inlined:
        inlined = {k: v for k, v in inlined.items() if k != "$defs"}
    nullable = _convert_anyof_nullable(inlined)
    prefix_converted = _convert_prefix_items(nullable)
    stripped = _strip_unsupported_keys(prefix_converted)
    return stripped


# ---------------------------------------------------------------------------
# Phase 12 helpers: exception → outcome classification, token-count read.
# ---------------------------------------------------------------------------


def _classify_exception(exc: BaseException) -> CallOutcome:
    """Map a retry-loop exception to a :data:`CallOutcome` outcome.

    The classification mirrors the ``except`` clause's ordering: a
    :class:`pydantic.ValidationError` is "validation_error" (the
    structured-output schema rejected the response); a
    :class:`ValueError` is "value_error" (the caller's own checks or
    SDK input validation); a :class:`google.genai.errors.APIError` (or
    any other catch-all) is "runtime_error" — Phase 12 finding: the
    SDK's APIError subclasses (ServerError 5xx, ClientError 4xx) all
    classify here. The synthetic ``max_retries_exceeded`` outcome is
    emitted by the caller, not this helper.
    """
    if isinstance(exc, ValidationError):
        return "validation_error"
    if isinstance(exc, ValueError):
        return "value_error"
    return "runtime_error"


def _extract_token_counts(
    payload: BatchPayload,
) -> tuple[int | None, int | None]:
    """Best-effort read of ``(prompt_tokens, response_tokens)`` from a
    payload that may or may not carry SDK ``usage_metadata``.

    A plain :class:`BatchPayload` (the mock-client path) returns
    ``(None, None)``. A real Gemini response would arrive wrapped, but
    the Phase 8 client at this module's tail returns the plain
    :class:`BatchPayload` (it discards the SDK envelope) — Phase 12's
    follow-up at :class:`_GeminiLLMClient` is to thread the usage
    metadata through; for now this helper returns ``(None, None)`` and
    the audit records' token counts stay ``None``. The journal entry
    records the Phase 12 follow-up.

    The function is defensive: an attribute that doesn't exist or isn't
    an int returns ``None`` for that slot rather than raising.
    """
    prompt_tokens: int | None = None
    response_tokens: int | None = None
    usage = getattr(payload, "usage_metadata", None)
    if usage is not None:
        pt = getattr(usage, "prompt_token_count", None)
        rt = getattr(usage, "candidates_token_count", None)
        if isinstance(pt, int):
            prompt_tokens = pt
        if isinstance(rt, int):
            response_tokens = rt
    return prompt_tokens, response_tokens
