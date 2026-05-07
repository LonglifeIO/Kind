"""Phase 6 mirror caller — a single Gemini call reading recent telemetry.

The mirror is what the charter names as the third layer of the project:
the interpretive process that translates what Io is doing into something a
human can reason about. Probe 1's mirror is the *plumbing* version of
that layer — a single LLM call that reads recent ``agent_step`` records
and produces a structured reading. It is not yet the adversarial
two-interpreter structure the design notes describe (that lives at Probe
2), and it is not yet driven by frozen criteria from
``Kind_frameworks.md`` (also Probe 2). What this module verifies is that
the conduit is alive: telemetry can be summarised, an external model can
read it, and a Pydantic record of the reading lands on disk where future
probes can re-read it.

**Model choice.** Gemini, not Anthropic — for methodological independence
from the Anthropic-heavy research workflow the project already uses. The
default is ``gemini-2.5-pro``; the implementation is parameterised so a
caller can drop to ``gemini-2.5-flash`` if rate limits become an issue.
The plan §2.10 mentions Anthropic as a default; this module supersedes
that on the broader project commitment to a different model lineage at
the mirror layer.

**API key handling.** The constructor reads ``GEMINI_API_KEY`` from the
environment unless an explicit ``api_key`` is passed. No key is hardcoded.

**Self-opacity.** The mirror reads ``TelemetryView``-equivalent data via
the parquet shards under ``telemetry_dir/agent_step/`` — i.e. it sees
posteriors, priors, KL, recon loss, intrinsic signal — everything Io
itself does *not* see. The mirror's view is the synthesis §Q5 pole the
actor's view is *not*: full transparency, not opacity. The conduit is
already wired by Phase 5's runner; this module just consumes it.

**Stance for Probe 1's prompt.** The system prompt frames Io minimally
as "an experimental learning system" and asks for an *observational*
summary, not an interpretation. Probe 1 is a calibration check — does
the data carry signal — not an evaluation of whether anything resembling
inner experience is forming. The frameworks document's frozen criteria
are deliberately absent from this prompt; they enter at Probe 2 alongside
the adversarial structure.

**Inputs the LLM gets.** Not the raw records — the high-dimensional
fields (``h_t``, ``z_t``, ``q_params_t``, ``p_params_t``, ``kl_per_dim_t``,
``encoder_embedding_t``) are not legible to a language model and would
crowd out the parts that are. The digest emits per-episode aggregates
(mean / std for ``kl_aggregate_t``, means for ``recon_loss_t`` /
``intrinsic_signal_t`` / ``policy_entropy_t``, action distribution),
flagged outliers (``kl_aggregate_t`` > 3σ, top recon-loss steps), and a
small set of sample records (first / middle / last per episode) shown
with scalar fields only. The full records remain in parquet for any
later mirror that wants them; Phase 6 just produces the readable summary.

**Structured output.** The Gemini SDK's ``response_schema`` is given
:class:`MirrorReadingPayload` (a two-field Pydantic model: ``summary``
and ``flagged_observations``); the SDK handles JSON-mode constraint and
parsing. The returned payload is then wrapped with envelope fields
(``run_id``, ``timestamp_ms``, ``agent_step_range``, etc.) into the full
:class:`MirrorReading`. The split between payload and envelope is what
makes the tests doable without a live API: the LLM-fillable surface is
just two fields, mockable via a fake response object.

**Out of scope at Probe 1** (per the user brief and plan §2.10):

- No adversarial second-LLM check.
- No frozen mirror criteria.
- No in-loop prompting; the runner does not call the mirror.
- No tool use in the LLM call.
- No memory/conversation history across calls.
- No emission to ``world_event`` (the ``mirror_marker`` slot is reserved
  for Probe 2).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Final

import pyarrow.parquet as pq
from google import genai
from pydantic import BaseModel, ConfigDict

from kind.observer.digest import build_digest

__all__ = [
    "MIRROR_READING_SCHEMA_VERSION",
    "MirrorCaller",
    "MirrorReading",
    "MirrorReadingPayload",
]


# Mirror readings are their own stream with their own version. The
# AgentStep schema is "0.1.0" (kind/observer/schemas.py); the mirror's
# reading record is independent so schema bumps in either don't drag the
# other along.
MIRROR_READING_SCHEMA_VERSION: Final[str] = "0.1.0"

_DEFAULT_MODEL: Final[str] = "gemini-2.5-pro"
_DEFAULT_MAX_TOKENS: Final[int] = 4096
_AGENT_STEP_SUBDIR: Final[str] = "agent_step"
_READINGS_FILE: Final[str] = "readings.jsonl"


_PROMPT_PREAMBLE: Final[str] = """You are reading telemetry from an experimental learning system called Io.
Io is an agent learning to navigate a small grid environment. The telemetry
below summarizes recent episodes of Io's behavior.

This is Probe 1 of a longer investigation. Probe 1's purpose is to confirm
that the telemetry pipeline produces legible signal, not to evaluate whether
Io is developing meaningfully or whether anything resembling inner
experience is present. Your job here is to read the data and produce a
brief, observational summary.

Look for:
- Whether the telemetry shows coherent learning signal (loss trends, KL
  behavior, exploration patterns).
- Whether anything in the data is unexpected or worth flagging for human
  review.
- Whether the data is dense enough to support meaningful future readings.

Respond in JSON with exactly two fields:
- summary: 2-4 paragraphs of observational free text.
- flagged_observations: a list of short bullet-style notes (may be empty).

Do not make claims about Io's inner experience. Do not use consciousness
vocabulary or projections. The Probe 1 mirror is a calibration check, not
an interpretive instrument.

Telemetry below.
---
"""


# ---- Pydantic models ------------------------------------------------------


class MirrorReadingPayload(BaseModel):
    """The Gemini-fillable subset of a mirror reading.

    Two fields only — ``summary`` and ``flagged_observations`` — passed to
    the SDK as the ``response_schema``. Everything else (run id, timestamp,
    agent-step range, model used) is envelope information the
    :class:`MirrorCaller` adds after the call returns. This split keeps the
    LLM-constrained surface narrow and makes mocking trivial in tests.
    """

    summary: str
    flagged_observations: list[str]


class MirrorReading(BaseModel):
    """Structured Pydantic output from a single mirror call.

    Frozen and ``extra="forbid"`` — the same discipline as
    :class:`~kind.observer.schemas.RecordEnvelope`. The reading is a record,
    not a workspace; once produced, it is what it is.

    ``agent_step_range`` is a ``(first_t, last_t)`` tuple of global env
    steps covered by the records the mirror read. Probe 2's mirror can
    use this to align its reading against the timeline the agent was on
    when the data was captured, without re-reading the parquet shards.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = MIRROR_READING_SCHEMA_VERSION
    run_id: str
    timestamp_ms: int
    n_episodes_read: int
    agent_step_range: tuple[int, int]
    summary: str
    flagged_observations: list[str]
    model_used: str


# ---- Mirror caller --------------------------------------------------------


class MirrorCaller:
    """Reads ``agent_step`` parquet shards, calls Gemini, returns a reading.

    Construction takes the model name, a max-output-tokens budget, and an
    optional API key. If ``api_key`` is ``None`` the constructor reads
    ``GEMINI_API_KEY`` from the environment; if neither is set,
    ``ValueError`` is raised before any network resources are bound.

    The constructor instantiates a :class:`google.genai.Client` immediately
    (no network call is made) and stores it on ``self._client``. Tests
    monkeypatch this attribute with a fake to avoid live API calls.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        api_key: str | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Pass api_key= to MirrorCaller or "
                "export the environment variable."
            )
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        self._model = model
        self._max_tokens = max_tokens
        # The client is typed as Any rather than genai.Client because the
        # SDK does not ship strict-mode type stubs for every call surface
        # (notably models.generate_content's keyword args). The interaction
        # surface is narrow — one method call — so the loss of static
        # checking is bounded and the alternative is a forest of typing
        # suppressions.
        self._client: Any = genai.Client(api_key=key)

    # ---- public surface ------------------------------------------------

    @property
    def model(self) -> str:
        return self._model

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def read_recent(
        self,
        telemetry_dir: Path,
        n_episodes: int = 3,
        run_id: str = "",
    ) -> MirrorReading:
        """Read the last ``n_episodes`` of ``agent_step`` and call Gemini.

        Raises :class:`ValueError` if no records exist under
        ``telemetry_dir/agent_step/`` — the mirror does not call the API
        on empty data. The error message names the directory checked so
        the operator can verify the runner actually wrote there.

        Returns a :class:`MirrorReading` with the LLM-produced summary
        wrapped in the envelope fields. The reading is *not* automatically
        written to disk — see :meth:`write_reading`.
        """
        if n_episodes <= 0:
            raise ValueError(f"n_episodes must be positive, got {n_episodes}")
        rows = _read_recent_agent_step_records(telemetry_dir, n_episodes)
        if not rows:
            raise ValueError(
                f"No agent_step records found under "
                f"{telemetry_dir / _AGENT_STEP_SUBDIR}. The mirror cannot "
                f"read empty telemetry; check the runner wrote shards."
            )
        digest_text = build_digest(rows)
        prompt = _PROMPT_PREAMBLE + digest_text

        response = self._client.models.generate_content(
            model=self._model,
            contents=[prompt],
            config={
                "response_mime_type": "application/json",
                "response_schema": MirrorReadingPayload,
                "max_output_tokens": self._max_tokens,
            },
        )
        payload = _extract_payload(response)

        episode_ids: list[int] = []
        for r in rows:
            eid = int(r["episode_id"])
            if eid not in episode_ids:
                episode_ids.append(eid)
        t_first = int(rows[0]["t"])
        t_last = int(rows[-1]["t"])

        return MirrorReading(
            schema_version=MIRROR_READING_SCHEMA_VERSION,
            run_id=run_id,
            timestamp_ms=int(time.time() * 1000),
            n_episodes_read=len(episode_ids),
            agent_step_range=(t_first, t_last),
            summary=payload.summary,
            flagged_observations=list(payload.flagged_observations),
            model_used=self._model,
        )

    def write_reading(self, reading: MirrorReading, mirror_dir: Path) -> None:
        """Append ``reading`` to ``mirror_dir/readings.jsonl``.

        Creates ``mirror_dir`` if it does not exist. Each line is one
        :meth:`~pydantic.BaseModel.model_dump_json` of the reading; no
        envelope wrapping (the reading is already a complete envelope).
        """
        mirror_dir.mkdir(parents=True, exist_ok=True)
        path = mirror_dir / _READINGS_FILE
        with path.open("a", encoding="utf-8") as fh:
            fh.write(reading.model_dump_json() + "\n")


# ---- record reading -------------------------------------------------------


def _read_recent_agent_step_records(
    telemetry_dir: Path, n_episodes: int
) -> list[dict[str, Any]]:
    """Load all parquet shards under ``telemetry_dir/agent_step/`` and
    filter to the last ``n_episodes`` distinct episode ids in record order.

    Returns an empty list if the subdirectory does not exist or contains
    no shards. Probe 1's runner writes shards lazily; an in-progress run
    will have at least one shard once the buffer flushes, but a brand-new
    run with zero records is a real case worth handling cleanly.
    """
    agent_step_dir = telemetry_dir / _AGENT_STEP_SUBDIR
    if not agent_step_dir.is_dir():
        return []
    shards = sorted(agent_step_dir.glob("shard-*.parquet"))
    if not shards:
        return []
    all_rows: list[dict[str, Any]] = []
    for shard in shards:
        table = pq.read_table(str(shard))  # type: ignore[no-untyped-call]
        all_rows.extend(table.to_pylist())
    if not all_rows:
        return []
    seen: list[int] = []
    for r in all_rows:
        eid = int(r["episode_id"])
        if eid not in seen:
            seen.append(eid)
    last_n = set(seen[-n_episodes:])
    return [r for r in all_rows if int(r["episode_id"]) in last_n]


# ---- response payload extraction ------------------------------------------


def _extract_payload(response: Any) -> MirrorReadingPayload:
    """Pull a :class:`MirrorReadingPayload` from a Gemini response.

    The SDK's behaviour with ``response_schema=PydanticModel`` is to
    populate ``response.parsed`` with either an instance of the model or a
    dict that matches the model's JSON schema. The exact shape depends on
    the SDK version; this function handles both, plus a ``response.text``
    JSON fallback for environments where ``parsed`` is not populated.
    """
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, MirrorReadingPayload):
        return parsed
    if isinstance(parsed, dict):
        return MirrorReadingPayload.model_validate(parsed)
    if isinstance(parsed, list) and parsed:
        first = parsed[0]
        if isinstance(first, MirrorReadingPayload):
            return first
        if isinstance(first, dict):
            return MirrorReadingPayload.model_validate(first)
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return MirrorReadingPayload.model_validate_json(text)
    raise RuntimeError(
        "Mirror response had no parseable content "
        "(response.parsed was empty and response.text was empty)."
    )
