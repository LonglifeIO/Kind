"""Phase 6 tests for ``kind/mirror/caller.py``.

The mirror caller is not in the five gating tests (plan §2.10: the smoke
verifies it is callable but does not assert on content). These tests
cover the structural surface — construction, parquet reading, digest
shape, payload extraction, response wrapping, JSONL append, and the
empty-telemetry edge — using a fake :class:`google.genai.Client` so no
network calls happen and outputs are deterministic.

A live integration test exists at the bottom of this file, gated on the
``GEMINI_API_KEY`` environment variable. It is skipped by default and is
intended for manual verification when the operator wants to confirm the
real conduit is alive.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kind.mirror.caller import (
    MIRROR_READING_SCHEMA_VERSION,
    MirrorCaller,
    MirrorReading,
    MirrorReadingPayload,
)
from kind.mirror.caller import (
    _extract_payload,
    _read_recent_agent_step_records,
)
from kind.observer.digest import build_digest, compact_record_repr
from kind.observer.schemas import PROBE_1_SCHEMA_VERSION, AgentStep
from kind.observer.sinks import ParquetSink


# ---- fake Gemini client ---------------------------------------------------


class _FakeResponse:
    """Stand-in for ``google.genai.types.GenerateContentResponse``.

    The real response exposes ``parsed`` (set when ``response_schema`` is
    used) and a ``text`` property. We mimic the duck type with the two
    attributes the caller's ``_extract_payload`` reads.
    """

    def __init__(
        self,
        parsed: Any = None,
        text: str | None = None,
    ) -> None:
        self.parsed = parsed
        self._text = text

    @property
    def text(self) -> str | None:
        return self._text


class _FakeModelsAPI:
    """Captures call kwargs and returns a configured response.

    A test asserts on ``last_call_kwargs`` to verify the caller invoked
    ``generate_content`` with the right model, contents, and config
    (specifically that ``response_schema`` is :class:`MirrorReadingPayload`).
    """

    def __init__(self, response: Any) -> None:
        self._response = response
        self.last_call_kwargs: dict[str, Any] | None = None
        self.call_count: int = 0

    def generate_content(self, **kwargs: Any) -> Any:
        self.last_call_kwargs = kwargs
        self.call_count += 1
        return self._response


class _FakeClient:
    """Mirrors the ``self._client.models.generate_content(...)`` shape."""

    def __init__(self, response: Any) -> None:
        self.models = _FakeModelsAPI(response)


def _install_fake_client(
    caller: MirrorCaller, response: Any
) -> _FakeClient:
    fake = _FakeClient(response)
    caller._client = fake  # noqa: SLF001 — replacing the only network seam
    return fake


# ---- test data factories --------------------------------------------------


def _make_agent_step(
    *,
    t: int,
    episode_id: int,
    step_in_episode: int,
    action_t: int = 0,
    kl_aggregate_t: float = 0.5,
    recon_loss_t: float = 1.0,
    intrinsic_signal_t: float = 0.3,
    policy_entropy_t: float = 1.5,
    h_dim: int = 4,
    z_dim: int = 2,
    embed_dim: int = 4,
) -> AgentStep:
    """Construct an ``AgentStep`` with shape fields large enough to be
    representative but small enough to keep tests fast."""
    return AgentStep(
        schema_version=PROBE_1_SCHEMA_VERSION,
        run_id="test-run",
        checkpoint_id=None,
        t=t,
        episode_id=episode_id,
        step_in_episode=step_in_episode,
        wallclock_ms=1_000_000 + t,
        h_t=[0.1] * h_dim,
        q_params_t=([0.0] * z_dim, [0.0] * z_dim),
        p_params_t=([0.0] * z_dim, [0.0] * z_dim),
        z_t=[0.5] * z_dim,
        kl_per_dim_t=[kl_aggregate_t / z_dim] * z_dim,
        kl_aggregate_t=kl_aggregate_t,
        recon_loss_t=recon_loss_t,
        action_t=action_t,
        action_logprob_t=-1.6,
        policy_entropy_t=policy_entropy_t,
        obs_hash_t=f"hash-{t}",
        intrinsic_signal_t=intrinsic_signal_t,
        encoder_embedding_t=[0.0] * embed_dim,
    )


def _write_records_to_parquet(
    records: list[AgentStep], telemetry_dir: Path, rows_per_shard: int = 10
) -> None:
    """Write the records to ``telemetry_dir/agent_step/`` via the
    real Phase 1 :class:`ParquetSink` so reads exercise the real on-disk
    format."""
    sink = ParquetSink(
        telemetry_dir / "agent_step", AgentStep, rows_per_shard=rows_per_shard
    )
    for r in records:
        sink.write(r)
    sink.close()


def _records_for_episodes(
    *,
    n_episodes: int,
    steps_per_episode: int,
    starting_t: int = 0,
    starting_episode_id: int = 0,
    action_pattern: Callable[[int, int], int] | None = None,
) -> list[AgentStep]:
    """Generate ``n_episodes * steps_per_episode`` records.

    ``action_pattern(episode_index, step_in_episode) -> action_t`` lets a
    test parameterise the action distribution; default is uniform 0-4
    (cycling) so action skewness checks see a non-skewed distribution.
    """
    records: list[AgentStep] = []
    t = starting_t
    for ep_ix in range(n_episodes):
        for step in range(steps_per_episode):
            action = (
                action_pattern(ep_ix, step)
                if action_pattern is not None
                else t % 5
            )
            records.append(
                _make_agent_step(
                    t=t,
                    episode_id=starting_episode_id + ep_ix,
                    step_in_episode=step,
                    action_t=action,
                )
            )
            t += 1
    return records


# ---- construction --------------------------------------------------------


def test_init_with_explicit_api_key_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit ``api_key`` is used directly; env var is not consulted."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    caller = MirrorCaller(api_key="fake-key-explicit")
    assert caller.model == "gemini-2.5-pro"
    assert caller.max_tokens == 4096


def test_init_reads_env_var_when_no_explicit_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-from-env")
    caller = MirrorCaller()
    assert caller is not None


def test_init_raises_without_any_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        MirrorCaller()


def test_init_rejects_empty_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty string is treated the same as missing — no false-positive key."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        MirrorCaller()


def test_init_rejects_zero_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        MirrorCaller(api_key="fake", max_tokens=0)


def test_init_rejects_negative_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        MirrorCaller(api_key="fake", max_tokens=-1)


def test_init_accepts_custom_model_and_max_tokens() -> None:
    caller = MirrorCaller(
        model="gemini-2.5-flash", max_tokens=2048, api_key="fake"
    )
    assert caller.model == "gemini-2.5-flash"
    assert caller.max_tokens == 2048


# ---- parquet reading -----------------------------------------------------


def test_read_recent_returns_empty_for_missing_subdir(tmp_path: Path) -> None:
    rows = _read_recent_agent_step_records(tmp_path, n_episodes=3)
    assert rows == []


def test_read_recent_returns_empty_for_empty_subdir(tmp_path: Path) -> None:
    (tmp_path / "agent_step").mkdir()
    rows = _read_recent_agent_step_records(tmp_path, n_episodes=3)
    assert rows == []


def test_read_recent_filters_to_last_n_episodes(tmp_path: Path) -> None:
    """Five episodes written; ask for last 3; verify episode ids 2, 3, 4."""
    records = _records_for_episodes(n_episodes=5, steps_per_episode=4)
    _write_records_to_parquet(records, tmp_path)
    rows = _read_recent_agent_step_records(tmp_path, n_episodes=3)
    episode_ids = sorted({int(r["episode_id"]) for r in rows})
    assert episode_ids == [2, 3, 4]
    assert len(rows) == 3 * 4  # 3 episodes × 4 steps each


def test_read_recent_returns_all_records_when_n_geq_total(tmp_path: Path) -> None:
    records = _records_for_episodes(n_episodes=2, steps_per_episode=3)
    _write_records_to_parquet(records, tmp_path)
    rows = _read_recent_agent_step_records(tmp_path, n_episodes=99)
    assert len(rows) == 6


def test_read_recent_concatenates_across_shards(tmp_path: Path) -> None:
    """rows_per_shard < total ensures multiple shards exist; reader stitches them."""
    records = _records_for_episodes(n_episodes=3, steps_per_episode=5)
    _write_records_to_parquet(records, tmp_path, rows_per_shard=4)
    shards = sorted((tmp_path / "agent_step").glob("shard-*.parquet"))
    assert len(shards) >= 2  # we forced multi-shard
    rows = _read_recent_agent_step_records(tmp_path, n_episodes=3)
    assert len(rows) == 15


# ---- digest function -----------------------------------------------------


def test_digest_returns_no_records_marker_for_empty_input() -> None:
    text = build_digest([])
    assert "no records" in text


def test_digest_includes_per_episode_summary_fields() -> None:
    records = _records_for_episodes(n_episodes=2, steps_per_episode=10)
    rows = [r.model_dump() for r in records]
    text = build_digest(rows)
    assert "## episode 0" in text
    assert "## episode 1" in text
    assert "kl_aggregate_t" in text
    assert "recon_loss_t" in text
    assert "intrinsic_signal_t" in text
    assert "policy_entropy_t" in text
    assert "action_t distribution" in text
    assert "step count: 10" in text


def test_digest_action_distribution_counts_each_action() -> None:
    """One episode of 10 steps with action 0 every step; counts should be [10,0,0,0,0]."""
    records = _records_for_episodes(
        n_episodes=1,
        steps_per_episode=10,
        action_pattern=lambda ep, step: 0,
    )
    rows = [r.model_dump() for r in records]
    text = build_digest(rows)
    assert "[10, 0, 0, 0, 0]" in text
    # And it should be flagged as skewed (10/10 > 0.7).
    assert "(skewed)" in text


def test_digest_unskewed_distribution_does_not_flag_skew() -> None:
    """Equal share across 5 actions; not skewed."""
    records = _records_for_episodes(
        n_episodes=1,
        steps_per_episode=10,
        action_pattern=lambda ep, step: step % 5,
    )
    rows = [r.model_dump() for r in records]
    text = build_digest(rows)
    assert "[2, 2, 2, 2, 2]" in text
    assert "(skewed)" not in text


def test_digest_flags_kl_outliers_above_3_sigma() -> None:
    """Manually construct rows with one outlier in kl_aggregate_t."""
    base_records = []
    for i in range(20):
        base_records.append(
            _make_agent_step(
                t=i,
                episode_id=0,
                step_in_episode=i,
                kl_aggregate_t=0.5,
            )
        )
    base_records[10] = _make_agent_step(
        t=10,
        episode_id=0,
        step_in_episode=10,
        kl_aggregate_t=50.0,  # far outlier
    )
    rows = [r.model_dump() for r in base_records]
    text = build_digest(rows)
    assert "flagged outliers:" in text
    assert "step_in_episode=10" in text


def test_digest_marks_no_outliers_when_kl_is_constant() -> None:
    records = []
    for i in range(20):
        records.append(
            _make_agent_step(
                t=i, episode_id=0, step_in_episode=i, kl_aggregate_t=0.5
            )
        )
    rows = [r.model_dump() for r in records]
    text = build_digest(rows)
    # Recon-loss outliers will still fire since recon_loss_t is constant
    # and the `top recon` code path picks 1 anyway when n_steps>=5; verify
    # the constant-kl case does NOT produce a kl-outlier line.
    assert "(z=" not in text
    assert "flagged outliers:" in text or "flagged outliers: none" in text


def test_digest_handles_episode_with_one_step() -> None:
    """A 1-step episode should not crash the std calculation."""
    records = [_make_agent_step(t=0, episode_id=0, step_in_episode=0)]
    rows = [r.model_dump() for r in records]
    text = build_digest(rows)
    assert "## episode 0" in text
    assert "step count: 1" in text


def test_digest_includes_sample_records_first_middle_last() -> None:
    records = _records_for_episodes(n_episodes=1, steps_per_episode=10)
    rows = [r.model_dump() for r in records]
    text = build_digest(rows)
    assert '"__pos": "first"' in text
    assert '"__pos": "middle"' in text
    assert '"__pos": "last"' in text


def test_digest_includes_envelope_fields() -> None:
    records = _records_for_episodes(n_episodes=1, steps_per_episode=3)
    rows = [r.model_dump() for r in records]
    text = build_digest(rows)
    assert f"schema_version: {PROBE_1_SCHEMA_VERSION}" in text
    assert "run_id: test-run" in text
    assert "env_step range:" in text
    assert "wallclock span:" in text


def test_compact_record_repr_excludes_high_dim_fields() -> None:
    """The high-dim vectors must NOT appear in the compact repr — they
    bloat the prompt and aren't legible to a language model."""
    record = _make_agent_step(t=0, episode_id=0, step_in_episode=0)
    repr_str = compact_record_repr(record.model_dump(), position="first")
    parsed = json.loads(repr_str)
    forbidden = {
        "h_t",
        "z_t",
        "q_params_t",
        "p_params_t",
        "kl_per_dim_t",
        "encoder_embedding_t",
    }
    assert forbidden.isdisjoint(set(parsed.keys()))
    # Scalars must be present.
    assert {
        "t",
        "episode_id",
        "step_in_episode",
        "kl_aggregate_t",
        "recon_loss_t",
        "action_t",
    }.issubset(parsed.keys())


# ---- payload extraction --------------------------------------------------


def test_extract_payload_handles_pydantic_instance() -> None:
    instance = MirrorReadingPayload(
        summary="summary text", flagged_observations=["a", "b"]
    )
    response = _FakeResponse(parsed=instance)
    out = _extract_payload(response)
    assert out is instance


def test_extract_payload_handles_dict() -> None:
    response = _FakeResponse(
        parsed={"summary": "from dict", "flagged_observations": []}
    )
    out = _extract_payload(response)
    assert out.summary == "from dict"
    assert out.flagged_observations == []


def test_extract_payload_handles_list_of_one_instance() -> None:
    """Some Gemini SDK versions return parsed as a list; the first element
    is what we want."""
    instance = MirrorReadingPayload(
        summary="from list", flagged_observations=["x"]
    )
    response = _FakeResponse(parsed=[instance])
    out = _extract_payload(response)
    assert out is instance


def test_extract_payload_handles_text_fallback() -> None:
    """If parsed is empty but text contains valid JSON, parse the JSON."""
    text = json.dumps({"summary": "fallback", "flagged_observations": ["f"]})
    response = _FakeResponse(parsed=None, text=text)
    out = _extract_payload(response)
    assert out.summary == "fallback"
    assert out.flagged_observations == ["f"]


def test_extract_payload_raises_when_response_is_empty() -> None:
    response = _FakeResponse(parsed=None, text=None)
    with pytest.raises(RuntimeError, match="no parseable content"):
        _extract_payload(response)


def test_extract_payload_raises_when_text_is_only_whitespace() -> None:
    response = _FakeResponse(parsed=None, text="   \n\t")
    with pytest.raises(RuntimeError, match="no parseable content"):
        _extract_payload(response)


# ---- read_recent end-to-end (mocked) --------------------------------------


def _fake_caller_with_payload(
    summary: str = "ok",
    flags: list[str] | None = None,
) -> tuple[MirrorCaller, _FakeClient]:
    """Build a caller with a fake client wired to return the given payload."""
    caller = MirrorCaller(api_key="fake")
    payload = MirrorReadingPayload(
        summary=summary, flagged_observations=flags or []
    )
    fake = _install_fake_client(caller, _FakeResponse(parsed=payload))
    return caller, fake


def test_read_recent_raises_on_missing_telemetry_dir(tmp_path: Path) -> None:
    caller, _ = _fake_caller_with_payload()
    with pytest.raises(ValueError, match="No agent_step records found"):
        caller.read_recent(tmp_path / "does-not-exist", n_episodes=3)


def test_read_recent_raises_on_empty_agent_step_subdir(tmp_path: Path) -> None:
    (tmp_path / "agent_step").mkdir()
    caller, _ = _fake_caller_with_payload()
    with pytest.raises(ValueError, match="cannot read empty telemetry"):
        caller.read_recent(tmp_path, n_episodes=3)


def test_read_recent_does_not_call_api_when_telemetry_empty(tmp_path: Path) -> None:
    """The empty-telemetry guard fires before the API is reached."""
    caller, fake = _fake_caller_with_payload()
    with pytest.raises(ValueError):
        caller.read_recent(tmp_path, n_episodes=3)
    assert fake.models.call_count == 0


def test_read_recent_rejects_zero_n_episodes(tmp_path: Path) -> None:
    caller, _ = _fake_caller_with_payload()
    with pytest.raises(ValueError, match="n_episodes must be positive"):
        caller.read_recent(tmp_path, n_episodes=0)


def test_read_recent_calls_generate_content_with_response_schema(
    tmp_path: Path,
) -> None:
    """Verify the SDK call's config kwarg includes the Pydantic response
    schema and the JSON mime type."""
    records = _records_for_episodes(n_episodes=2, steps_per_episode=3)
    _write_records_to_parquet(records, tmp_path)

    caller, fake = _fake_caller_with_payload()
    caller.read_recent(tmp_path, n_episodes=2)

    kwargs = fake.models.last_call_kwargs
    assert kwargs is not None
    assert kwargs["model"] == "gemini-2.5-pro"
    assert isinstance(kwargs["contents"], list)
    assert len(kwargs["contents"]) == 1
    assert "Telemetry below" in kwargs["contents"][0]
    config = kwargs["config"]
    assert config["response_mime_type"] == "application/json"
    assert config["response_schema"] is MirrorReadingPayload
    assert config["max_output_tokens"] == 4096


def test_read_recent_passes_max_tokens_through(tmp_path: Path) -> None:
    records = _records_for_episodes(n_episodes=1, steps_per_episode=2)
    _write_records_to_parquet(records, tmp_path)
    caller = MirrorCaller(api_key="fake", max_tokens=512)
    fake = _install_fake_client(
        caller,
        _FakeResponse(
            parsed=MirrorReadingPayload(summary="x", flagged_observations=[])
        ),
    )
    caller.read_recent(tmp_path, n_episodes=1)
    assert fake.models.last_call_kwargs is not None
    assert fake.models.last_call_kwargs["config"]["max_output_tokens"] == 512


def test_read_recent_wraps_payload_into_full_reading(tmp_path: Path) -> None:
    records = _records_for_episodes(n_episodes=2, steps_per_episode=3)
    _write_records_to_parquet(records, tmp_path)

    caller, _ = _fake_caller_with_payload(
        summary="the summary", flags=["one", "two"]
    )
    reading = caller.read_recent(
        tmp_path, n_episodes=2, run_id="run-xyz"
    )

    assert isinstance(reading, MirrorReading)
    assert reading.schema_version == MIRROR_READING_SCHEMA_VERSION
    assert reading.run_id == "run-xyz"
    assert reading.timestamp_ms > 0
    assert reading.n_episodes_read == 2
    # 2 episodes × 3 steps = 6 records, t in [0, 5]
    assert reading.agent_step_range == (0, 5)
    assert reading.summary == "the summary"
    assert reading.flagged_observations == ["one", "two"]
    assert reading.model_used == "gemini-2.5-pro"


def test_read_recent_uses_only_last_n_episodes_in_envelope(tmp_path: Path) -> None:
    records = _records_for_episodes(n_episodes=5, steps_per_episode=4)
    _write_records_to_parquet(records, tmp_path)

    caller, _ = _fake_caller_with_payload()
    reading = caller.read_recent(tmp_path, n_episodes=2, run_id="r")

    assert reading.n_episodes_read == 2
    # Last 2 episodes are 3 and 4. With 4 steps each, t ranges 12..19.
    assert reading.agent_step_range == (12, 19)


def test_read_recent_reading_is_frozen() -> None:
    """The reading must reject in-place mutation — it is a record."""
    reading = MirrorReading(
        schema_version=MIRROR_READING_SCHEMA_VERSION,
        run_id="r",
        timestamp_ms=1,
        n_episodes_read=1,
        agent_step_range=(0, 0),
        summary="s",
        flagged_observations=[],
        model_used="m",
    )
    with pytest.raises(Exception):  # ValidationError under Pydantic v2 frozen
        reading.summary = "tampered"  # type: ignore[misc]


def test_mirror_reading_rejects_extra_fields() -> None:
    """``extra="forbid"`` — schema discipline matches the AgentStep envelope."""
    with pytest.raises(Exception):
        MirrorReading(
            schema_version="0.1.0",
            run_id="r",
            timestamp_ms=1,
            n_episodes_read=1,
            agent_step_range=(0, 0),
            summary="s",
            flagged_observations=[],
            model_used="m",
            extra_field="not allowed",  # type: ignore[call-arg]
        )


# ---- write_reading -------------------------------------------------------


def _make_reading(summary: str = "test summary", run_id: str = "r") -> MirrorReading:
    return MirrorReading(
        schema_version=MIRROR_READING_SCHEMA_VERSION,
        run_id=run_id,
        timestamp_ms=12345,
        n_episodes_read=2,
        agent_step_range=(0, 9),
        summary=summary,
        flagged_observations=["obs"],
        model_used="gemini-2.5-pro",
    )


def test_write_reading_creates_directory_if_missing(tmp_path: Path) -> None:
    caller = MirrorCaller(api_key="fake")
    mirror_dir = tmp_path / "mirror" / "deep" / "missing"
    caller.write_reading(_make_reading(), mirror_dir)
    assert (mirror_dir / "readings.jsonl").is_file()


def test_write_reading_appends_one_line(tmp_path: Path) -> None:
    caller = MirrorCaller(api_key="fake")
    mirror_dir = tmp_path / "mirror"
    caller.write_reading(_make_reading(summary="first"), mirror_dir)
    content = (mirror_dir / "readings.jsonl").read_text()
    assert content.count("\n") == 1
    parsed = json.loads(content.strip())
    assert parsed["summary"] == "first"
    assert parsed["schema_version"] == MIRROR_READING_SCHEMA_VERSION


def test_write_reading_accumulates_across_calls(tmp_path: Path) -> None:
    caller = MirrorCaller(api_key="fake")
    mirror_dir = tmp_path / "mirror"
    caller.write_reading(_make_reading(summary="alpha"), mirror_dir)
    caller.write_reading(_make_reading(summary="beta"), mirror_dir)
    caller.write_reading(_make_reading(summary="gamma"), mirror_dir)

    lines = (mirror_dir / "readings.jsonl").read_text().splitlines()
    assert len(lines) == 3
    parsed_summaries = [json.loads(line)["summary"] for line in lines]
    assert parsed_summaries == ["alpha", "beta", "gamma"]


def test_write_reading_round_trips_through_pydantic(tmp_path: Path) -> None:
    caller = MirrorCaller(api_key="fake")
    mirror_dir = tmp_path / "mirror"
    original = _make_reading(summary="round-trip")
    caller.write_reading(original, mirror_dir)
    line = (mirror_dir / "readings.jsonl").read_text().strip()
    restored = MirrorReading.model_validate_json(line)
    assert restored == original


# ---- dependency lint regression ------------------------------------------


def test_actor_module_still_does_not_import_telemetry_view() -> None:
    """Phase 3c's lint must continue to hold after Phase 6 adds the
    mirror caller. The mirror caller is allowed to import TelemetryView
    (it's a TelemetryView consumer); only the actor is restricted. This
    sanity test re-runs the dependency lint here so a Phase 6 regression
    that accidentally touched the actor module would be caught even if
    test_views.py wasn't run."""
    import ast

    actor_path = Path(__file__).resolve().parent.parent / "kind" / "agents" / "actor.py"
    tree = ast.parse(actor_path.read_text())
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
    assert "TelemetryView" not in names
    assert "kind.agents.views" not in names


# ---- live API integration (manual; gated on env var) ----------------------


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set; skipping live mirror call",
)
def test_live_mirror_call_returns_reading(tmp_path: Path) -> None:
    """Manual verification only — confirms a real Gemini call works.

    Skipped by default. Set ``GEMINI_API_KEY`` and run::

        pytest tests/test_mirror_caller.py::test_live_mirror_call_returns_reading -s

    The test only asserts the call returned a :class:`MirrorReading` with
    a non-empty summary; specific content is the LLM's prerogative."""
    records = _records_for_episodes(n_episodes=2, steps_per_episode=4)
    _write_records_to_parquet(records, tmp_path)
    caller = MirrorCaller()  # reads env var
    reading = caller.read_recent(
        tmp_path, n_episodes=2, run_id="live-test-run"
    )
    assert isinstance(reading, MirrorReading)
    assert len(reading.summary) > 0
    assert reading.n_episodes_read == 2
    assert reading.model_used == "gemini-2.5-pro"
