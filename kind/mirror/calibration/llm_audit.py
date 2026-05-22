"""Phase 12 LLM-call audit — per-attempt records and per-round aggregate.

The Phase 8 LLM caller already has the safety net: bounded retries and
:class:`~kind.mirror.llm_caller.MirrorLLMError` on exhaustion. What the
caller does *not* have at Phase 8 is a diagnostic record of what each
call attempt actually did — latency, token counts, outcome, error
message. That diagnostic surface is what Phase 12's calibration needs
to answer Phase 8's newly-open §3 (LLM-caller drift, structured-output
schema fragility, role-prompt interference).

This module declares the record shape and the per-round aggregate. The
modified :mod:`kind.mirror.llm_caller` emits records into a collector
the orchestrator threads through; the round driver collects the
per-pass collector contents into the round's
:class:`LLMCallAudit`.

**One record per attempt, not per call.** A single ``call_mirror_llm``
invocation that succeeds on attempt 3 of 4 produces three records: two
failed attempts (validation_error / runtime_error / etc.) and one
success. The audit's :attr:`total_calls` counts records;
:attr:`total_retries` counts records with ``attempt_number > 1``;
:attr:`total_failures` counts ``max_retries_exceeded`` records. This
shape is the one the Phase 12 journal entry inspects.

**Token counts are best-effort.** The Gemini SDK exposes them via the
response's ``usage_metadata`` field, but the field's presence is not
guaranteed (and is absent in the mock-client path). The record's
``prompt_token_count`` and ``response_token_count`` are
``int | None``; ``None`` means the SDK did not return a count for this
attempt. The audit's :attr:`total_tokens_in` / :attr:`total_tokens_out`
sum over the non-None records and are themselves ``int | None`` —
``None`` if every record's token count is ``None``.
"""

from __future__ import annotations

from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, field_validator

from kind.mirror.llm_caller import PassRole

__all__ = [
    "CallOutcome",
    "LLMCallRecord",
    "LLMCallAudit",
    "LLMCallRecordCollector",
]


# ---------------------------------------------------------------------------
# The outcome literal.
# ---------------------------------------------------------------------------

# Five outcomes per the Phase 12 plan Part 5. The order is the order
# the Phase 8 ``call_mirror_llm`` retry loop encounters them: success on
# a parseable BatchPayload; validation_error on a Pydantic ValidationError
# from a malformed structured response; value_error on a ValueError from
# the SDK or the caller's own checks; runtime_error on any other
# RuntimeError (the SDK's generic transport failures land here);
# max_retries_exceeded is the synthetic outcome the caller emits one
# extra record under when the retry budget is gone and
# ``MirrorLLMError`` is about to be raised.
CallOutcome: TypeAlias = Literal[
    "success",
    "validation_error",
    "value_error",
    "runtime_error",
    "max_retries_exceeded",
]


_OUTCOME_VALUES: Final[frozenset[str]] = frozenset(
    {
        "success",
        "validation_error",
        "value_error",
        "runtime_error",
        "max_retries_exceeded",
    }
)


# ---------------------------------------------------------------------------
# Per-attempt record.
# ---------------------------------------------------------------------------


class LLMCallRecord(BaseModel):
    """One record per LLM call attempt.

    Frozen, ``extra="forbid"``. A single ``call_mirror_llm`` invocation
    that succeeds on the third attempt produces three records: two
    ``validation_error`` / ``runtime_error`` attempts and one
    ``success`` attempt. The orchestrator's collector accumulates the
    records; the round driver packs them into :class:`LLMCallAudit`.

    Fields:

    - ``round_id``: the round this attempt belongs to (snake_case id from
      :class:`~kind.mirror.calibration.round.RoundConfig`).
    - ``pass_index``: 0-based index within the round (0 ..
      ``passes_per_checkpoint`` − 1 per checkpoint).
    - ``checkpoint_id``: which checkpoint this attempt was reading.
    - ``role``: ``"primary"`` or ``"adversarial"`` from
      :data:`~kind.mirror.llm_caller.PassRole`.
    - ``attempt_number``: 1-indexed; 1 is the first attempt, 2 is the
      first retry, etc. ``LLMConfig.max_retries + 1`` is the maximum
      ``attempt_number`` a record can carry before
      ``max_retries_exceeded`` is emitted.
    - ``request_timestamp_ms`` / ``response_timestamp_ms``: epoch ms.
      ``response_timestamp_ms`` is ``None`` for the synthetic
      ``max_retries_exceeded`` record (no actual response).
    - ``latency_ms``: ``response_timestamp_ms - request_timestamp_ms``
      when both are set; ``None`` otherwise.
    - ``model_name``: the ``LLMConfig.model_name`` used for this attempt
      (so a round that intentionally varied the model can be audited).
    - ``prompt_token_count`` / ``response_token_count``: best-effort
      from the SDK's ``usage_metadata``; ``None`` if unavailable.
    - ``outcome``: one of the five :data:`CallOutcome` values.
    - ``error_message``: verbatim ``repr(exc)`` for failures; ``None``
      on success. The audit's later analysis greps these for patterns.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    round_id: str
    pass_index: int
    checkpoint_id: str
    role: PassRole
    attempt_number: int
    request_timestamp_ms: int
    response_timestamp_ms: int | None
    latency_ms: int | None
    model_name: str
    prompt_token_count: int | None
    response_token_count: int | None
    outcome: CallOutcome
    error_message: str | None

    @field_validator("round_id", "checkpoint_id", "model_name")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty.")
        return value

    @field_validator("pass_index")
    @classmethod
    def _validate_pass_index_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"pass_index must be >= 0; got {value}.")
        return value

    @field_validator("attempt_number")
    @classmethod
    def _validate_attempt_number_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError(
                f"attempt_number is 1-indexed; got {value}. The first "
                f"attempt is 1; a value below 1 indicates a logic error in "
                f"the caller's record emission."
            )
        return value

    @field_validator("outcome")
    @classmethod
    def _validate_outcome(cls, value: str) -> str:
        # Pydantic's Literal validator handles this, but a custom check
        # gives a friendlier error message for the case where a future
        # contributor passes a typo'd outcome string.
        if value not in _OUTCOME_VALUES:
            raise ValueError(
                f"outcome must be one of {sorted(_OUTCOME_VALUES)}; "
                f"got {value!r}."
            )
        return value


# ---------------------------------------------------------------------------
# Per-round aggregate.
# ---------------------------------------------------------------------------


class LLMCallAudit(BaseModel):
    """Per-round aggregate of every LLM call attempt.

    Frozen, ``extra="forbid"``. Built by the round driver from the
    flattened per-pass record collectors. The summary totals are
    materialized at construction (not computed on the fly) so the audit
    is self-contained on disk; :meth:`from_records` is the helper that
    derives them.

    The audit is intentionally narrow — five totals plus the record
    tuple. Richer analyses (per-role latency distribution, per-criterion
    error breakdown, drift over time within a round) live on the
    journal-side as separate analyses; the audit is the input to those,
    not the place to compute them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    records: tuple[LLMCallRecord, ...]
    total_calls: int
    total_retries: int
    total_failures: int
    total_wallclock_ms: int
    total_tokens_in: int | None
    total_tokens_out: int | None

    @field_validator("total_calls", "total_retries", "total_failures")
    @classmethod
    def _validate_totals_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"total must be >= 0; got {value}.")
        return value

    @field_validator("total_wallclock_ms")
    @classmethod
    def _validate_wallclock_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"total_wallclock_ms must be >= 0; got {value}.")
        return value

    @classmethod
    def from_records(cls, records: tuple[LLMCallRecord, ...]) -> "LLMCallAudit":
        """Derive the audit's totals from the record tuple.

        ``total_calls``: ``len(records)`` — one record per attempt.
        ``total_retries``: count of records with ``attempt_number > 1``.
        ``total_failures``: count of records with
        ``outcome == "max_retries_exceeded"``.
        ``total_wallclock_ms``: sum of ``latency_ms`` over records that
        have it (skips ``None`` — the ``max_retries_exceeded`` records
        contribute nothing).
        ``total_tokens_in`` / ``total_tokens_out``: sum over non-None
        token counts; ``None`` if every record's count is ``None``.

        The records tuple is preserved in attempt-order — the caller is
        responsible for the order; this method does not sort.
        """
        total_calls = len(records)
        total_retries = sum(1 for r in records if r.attempt_number > 1)
        total_failures = sum(
            1 for r in records if r.outcome == "max_retries_exceeded"
        )
        total_wallclock_ms = sum(
            r.latency_ms for r in records if r.latency_ms is not None
        )
        in_tokens = [
            r.prompt_token_count
            for r in records
            if r.prompt_token_count is not None
        ]
        out_tokens = [
            r.response_token_count
            for r in records
            if r.response_token_count is not None
        ]
        total_tokens_in = sum(in_tokens) if in_tokens else None
        total_tokens_out = sum(out_tokens) if out_tokens else None
        return cls(
            records=records,
            total_calls=total_calls,
            total_retries=total_retries,
            total_failures=total_failures,
            total_wallclock_ms=total_wallclock_ms,
            total_tokens_in=total_tokens_in,
            total_tokens_out=total_tokens_out,
        )


# ---------------------------------------------------------------------------
# The collector.
# ---------------------------------------------------------------------------


class LLMCallRecordCollector:
    """Per-pass mutable accumulator for :class:`LLMCallRecord` instances.

    The modified :func:`~kind.mirror.llm_caller.call_mirror_llm` accepts
    an optional ``record_collector`` keyword argument; when provided, the
    caller appends one :class:`LLMCallRecord` per attempt. The round
    driver constructs one collector, passes it down through the
    orchestrator, and reads ``collector.records`` after the pass returns
    to build the audit.

    This is *not* a Pydantic model — it's a thin mutable container.
    Frozen records are appended to a list; tests assert the list's
    contents. The orchestrator threads the collector through the
    Phase 8 ``run_adversarial_pass`` via an additional keyword argument
    (see :mod:`kind.mirror.orchestrator`'s Phase 12 addition).

    Construction takes the round-level context the records all share
    (``round_id``, ``pass_index``, ``checkpoint_id``); each
    :meth:`record` call only needs the per-attempt fields. The collector
    fills in the shared fields automatically. This keeps the
    record-emission call sites in the LLM caller small.
    """

    def __init__(
        self,
        *,
        round_id: str,
        pass_index: int,
        checkpoint_id: str,
    ) -> None:
        if not round_id.strip():
            raise ValueError("round_id must be non-empty.")
        if pass_index < 0:
            raise ValueError(
                f"pass_index must be >= 0; got {pass_index}."
            )
        if not checkpoint_id.strip():
            raise ValueError("checkpoint_id must be non-empty.")
        self._round_id = round_id
        self._pass_index = pass_index
        self._checkpoint_id = checkpoint_id
        self._records: list[LLMCallRecord] = []

    @property
    def records(self) -> tuple[LLMCallRecord, ...]:
        """Tuple snapshot of the recorded attempts, in append order."""
        return tuple(self._records)

    @property
    def round_id(self) -> str:
        return self._round_id

    @property
    def pass_index(self) -> int:
        return self._pass_index

    @property
    def checkpoint_id(self) -> str:
        return self._checkpoint_id

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
    ) -> LLMCallRecord:
        """Append one record to the collector. Returns the record for
        the caller's convenience (the LLM caller's tests assert on the
        returned record's fields)."""
        latency_ms: int | None
        if response_timestamp_ms is not None:
            latency_ms = response_timestamp_ms - request_timestamp_ms
            if latency_ms < 0:
                # A negative latency means clock skew on the response
                # side; the audit records the raw values and surfaces
                # the latency as ``None`` rather than a negative number
                # (which the validator would also accept but mislead).
                latency_ms = None
        else:
            latency_ms = None
        rec = LLMCallRecord(
            round_id=self._round_id,
            pass_index=self._pass_index,
            checkpoint_id=self._checkpoint_id,
            role=role,
            attempt_number=attempt_number,
            request_timestamp_ms=request_timestamp_ms,
            response_timestamp_ms=response_timestamp_ms,
            latency_ms=latency_ms,
            model_name=model_name,
            prompt_token_count=prompt_token_count,
            response_token_count=response_token_count,
            outcome=outcome,
            error_message=error_message,
        )
        self._records.append(rec)
        return rec

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return (
            f"LLMCallRecordCollector(round_id={self._round_id!r}, "
            f"pass_index={self._pass_index}, "
            f"checkpoint_id={self._checkpoint_id!r}, "
            f"records={len(self._records)})"
        )
