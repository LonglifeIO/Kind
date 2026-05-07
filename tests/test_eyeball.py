"""Phase 7 tests for ``kind/observer/eyeball.py`` and ``kind/observer/digest.py``.

Coverage:

* Smoke test for each of the five public eyeball functions — construct
  dummy telemetry, call the helper, capture stdout, verify the output
  carries the right key terms.
* :func:`count_world_events` correctly counts records by event_type
  against a known JSONL fixture.
* Regression check on the digest lift: ``build_digest`` (now in
  ``kind/observer/digest.py``) produces the same shape it did when it
  was ``_build_digest`` inside the mirror caller.
* CLI entry points: invoke each subcommand via the in-process ``main``
  function and verify it returns ``0`` and produces expected output.
* The journal scaffolding README exists at the expected path with the
  documented sections present.

The tests use the real :class:`~kind.observer.sinks.ParquetSink` and
:class:`~kind.observer.sinks.JsonlSink` to construct fixtures so the
on-disk format the helpers read is exactly what the runner writes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kind.observer import eyeball
from kind.observer.digest import build_digest, compact_record_repr
from kind.observer.eyeball import (
    _format_action_distribution,
    _format_duration_ms,
    _format_scalar,
    _render_ascii_frame,
    _summarise_vector,
    count_world_events,
    main,
    show_dream_rollout,
    show_episode_summary,
    show_recent_agent_steps,
    show_run_summary,
)
from kind.observer.schemas import (
    PROBE_1_SCHEMA_VERSION,
    AgentStep,
    DreamRollout,
    WorldEvent,
)
from kind.observer.sinks import JsonlSink, ParquetSink


REPO_ROOT = Path(__file__).resolve().parent.parent


# ---- factories -----------------------------------------------------------


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
    wallclock_ms: int | None = None,
) -> AgentStep:
    return AgentStep(
        schema_version=PROBE_1_SCHEMA_VERSION,
        run_id="test-run",
        checkpoint_id=None,
        t=t,
        episode_id=episode_id,
        step_in_episode=step_in_episode,
        wallclock_ms=wallclock_ms if wallclock_ms is not None else 1_000_000 + t,
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


def _write_agent_steps(records: list[AgentStep], telemetry_dir: Path) -> None:
    sink = ParquetSink(telemetry_dir / "agent_step", AgentStep, rows_per_shard=10_000)
    for r in records:
        sink.write(r)
    sink.close()


def _make_dream_rollout(
    *,
    seed_step: int,
    horizon: int = 3,
    h_dim: int = 4,
    z_dim: int = 2,
    with_decoded_obs: bool = False,
) -> DreamRollout:
    decoded: list[bytes] | None
    if with_decoded_obs:
        # Each frame is 32x32 uint8 = 1024 bytes. Use a gradient pattern
        # so the ASCII renderer produces something non-trivial: row r, col
        # c → (r * 8 + c) mod 256.
        frames: list[bytes] = []
        for _step in range(horizon):
            buf = bytearray(32 * 32)
            for r in range(32):
                for c in range(32):
                    buf[r * 32 + c] = (r * 8 + c) % 256
            frames.append(bytes(buf))
        decoded = frames
    else:
        decoded = None
    return DreamRollout(
        schema_version=PROBE_1_SCHEMA_VERSION,
        run_id="test-run",
        checkpoint_id=None,
        seed_step=seed_step,
        seed_h0=[0.0] * h_dim,
        seed_z0=[0.0] * z_dim,
        sequence_h=[[0.0] * h_dim for _ in range(horizon)],
        sequence_z_prior=[[0.0] * z_dim for _ in range(horizon)],
        sequence_action=[i % 5 for i in range(horizon)],
        sequence_action_logprob=[-1.6] * horizon,
        sequence_prior_entropy=[1.0 + 0.1 * i for i in range(horizon)],
        sequence_decoded_obs=decoded,
        cumulative_prior_entropy=sum(1.0 + 0.1 * i for i in range(horizon)),
        mean_step_kl_successive_priors=0.05,
        max_step_latent_norm_change=0.5,
    )


def _write_dream_rollouts(records: list[DreamRollout], telemetry_dir: Path) -> None:
    sink = ParquetSink(
        telemetry_dir / "dream_rollout", DreamRollout, rows_per_shard=10_000
    )
    for r in records:
        sink.write(r)
    sink.close()


def _make_world_event(
    *,
    t_event: int,
    event_type: str,
    source: str,
    payload: dict[str, Any] | None = None,
    wallclock_ms: int = 0,
) -> WorldEvent:
    # Cast through the schema; the Literal types are checked at runtime.
    return WorldEvent(
        schema_version=PROBE_1_SCHEMA_VERSION,
        run_id="test-run",
        checkpoint_id=None,
        t_event=t_event,
        event_type=event_type,  # type: ignore[arg-type]
        source=source,  # type: ignore[arg-type]
        payload=payload or {},
        wallclock_ms=wallclock_ms,
    )


def _write_world_events(records: list[WorldEvent], telemetry_dir: Path) -> None:
    sink = JsonlSink(telemetry_dir / "world_event.jsonl", WorldEvent)
    for r in records:
        sink.write(r)
    sink.close()


def _episode_records(
    *,
    episode_id: int,
    n_steps: int,
    starting_t: int = 0,
    action_pattern: list[int] | None = None,
) -> list[AgentStep]:
    records: list[AgentStep] = []
    for step in range(n_steps):
        action = (
            action_pattern[step % len(action_pattern)]
            if action_pattern is not None
            else step % 5
        )
        records.append(
            _make_agent_step(
                t=starting_t + step,
                episode_id=episode_id,
                step_in_episode=step,
                action_t=action,
            )
        )
    return records


# ---- show_recent_agent_steps ---------------------------------------------


def test_show_recent_agent_steps_prints_default_scalar_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = _episode_records(episode_id=0, n_steps=5)
    _write_agent_steps(records, tmp_path)
    show_recent_agent_steps(tmp_path, n=3)
    out = capsys.readouterr().out
    # Header carries the truncation count.
    assert "last 3 of 5 records" in out
    # Default scalar fields are present.
    for field in ("t:", "episode_id:", "kl_aggregate_t:", "action_t:", "obs_hash_t:"):
        assert field in out


def test_show_recent_agent_steps_default_excludes_high_dim_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = _episode_records(episode_id=0, n_steps=2)
    _write_agent_steps(records, tmp_path)
    show_recent_agent_steps(tmp_path, n=2)
    out = capsys.readouterr().out
    # Match the rendered field prefix exactly (two-space indent + key + colon)
    # so this test isn't tripped by substring matches like ``obs_hash_t``
    # containing ``h_t``.
    forbidden = (
        "  h_t:",
        "  z_t:",
        "  q_params_t:",
        "  p_params_t:",
        "  kl_per_dim_t:",
        "  encoder_embedding_t:",
    )
    for field in forbidden:
        assert field not in out, f"high-dim field '{field}' leaked into default output"


def test_show_recent_agent_steps_custom_fields_summarise_vectors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = _episode_records(episode_id=0, n_steps=1)
    _write_agent_steps(records, tmp_path)
    show_recent_agent_steps(tmp_path, n=1, fields=["t", "h_t", "z_t"])
    out = capsys.readouterr().out
    assert "t:" in out
    assert "h_t:" in out
    assert "z_t:" in out
    # Vector summary form: "shape=(N,) mean=... std=... min=... max=... [...]"
    assert "shape=(" in out
    assert "mean=" in out


def test_show_recent_agent_steps_missing_directory_prints_helpful_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    show_recent_agent_steps(tmp_path / "does-not-exist", n=5)
    out = capsys.readouterr().out
    assert "no agent_step records" in out


def test_show_recent_agent_steps_rejects_zero_n(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="n must be positive"):
        show_recent_agent_steps(tmp_path, n=0)


def test_show_recent_agent_steps_n_larger_than_total(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = _episode_records(episode_id=0, n_steps=3)
    _write_agent_steps(records, tmp_path)
    show_recent_agent_steps(tmp_path, n=99)
    out = capsys.readouterr().out
    assert "last 3 of 3 records" in out


# ---- count_world_events --------------------------------------------------


def test_count_world_events_returns_correct_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    events = [
        _make_world_event(t_event=0, event_type="env_reset", source="environment"),
        _make_world_event(t_event=200, event_type="env_reset", source="environment"),
        _make_world_event(t_event=200, event_type="env_reset", source="environment"),
        _make_world_event(t_event=50, event_type="builder_perturbation", source="builder"),
        _make_world_event(t_event=150, event_type="builder_perturbation", source="builder"),
        _make_world_event(
            t_event=199,
            event_type="internal_stochasticity_aggregate",
            source="environment",
        ),
    ]
    _write_world_events(events, tmp_path)
    counts = count_world_events(tmp_path)
    assert counts == {
        "env_reset": 3,
        "builder_perturbation": 2,
        "internal_stochasticity_aggregate": 1,
    }
    out = capsys.readouterr().out
    assert "env_reset: 3" in out
    assert "builder_perturbation: 2" in out
    assert "internal_stochasticity_aggregate: 1" in out


def test_count_world_events_missing_file_returns_empty_dict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    counts = count_world_events(tmp_path)
    assert counts == {}
    out = capsys.readouterr().out
    assert "no world_event file" in out


def test_count_world_events_empty_file_returns_empty_dict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "world_event.jsonl").write_text("")
    counts = count_world_events(tmp_path)
    assert counts == {}
    out = capsys.readouterr().out
    assert "no events recorded" in out


def test_count_world_events_skips_malformed_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Defensive: a half-flushed JSONL line should not crash the counter.

    The runner's :class:`~kind.observer.sinks.JsonlSink` flushes on
    ``close``, but a crashed run can leave a partial last line. The
    counter ignores it and reports the count of bad lines.
    """
    valid = _make_world_event(t_event=0, event_type="env_reset", source="environment")
    text = valid.model_dump_json() + "\n" + "{not valid json"
    (tmp_path / "world_event.jsonl").write_text(text)
    counts = count_world_events(tmp_path)
    assert counts == {"env_reset": 1}
    out = capsys.readouterr().out
    assert "skipped 1 malformed line" in out


# ---- show_episode_summary -----------------------------------------------


def test_show_episode_summary_defaults_to_most_recent_episode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ep0 = _episode_records(episode_id=0, n_steps=3)
    ep1 = _episode_records(episode_id=1, n_steps=4, starting_t=3)
    _write_agent_steps(ep0 + ep1, tmp_path)
    show_episode_summary(tmp_path)
    out = capsys.readouterr().out
    assert "episode 1" in out
    assert "step count: 4" in out


def test_show_episode_summary_with_specific_episode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ep0 = _episode_records(episode_id=0, n_steps=3)
    ep1 = _episode_records(episode_id=1, n_steps=4, starting_t=3)
    _write_agent_steps(ep0 + ep1, tmp_path)
    show_episode_summary(tmp_path, episode_id=0)
    out = capsys.readouterr().out
    assert "episode 0" in out
    assert "step count: 3" in out


def test_show_episode_summary_includes_action_distribution_with_labels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = _episode_records(
        episode_id=0,
        n_steps=10,
        action_pattern=[0, 0, 1, 2, 3, 4, 0, 0, 1, 2],  # 4 up, 2 down, 2 left, 1 right, 1 stay
    )
    _write_agent_steps(records, tmp_path)
    show_episode_summary(tmp_path, episode_id=0)
    out = capsys.readouterr().out
    assert "up=4" in out
    assert "down=2" in out
    assert "left=2" in out
    assert "right=1" in out
    assert "stay=1" in out


def test_show_episode_summary_missing_episode_prints_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = _episode_records(episode_id=0, n_steps=2)
    _write_agent_steps(records, tmp_path)
    show_episode_summary(tmp_path, episode_id=999)
    out = capsys.readouterr().out
    assert "episode 999 has no agent_step records" in out


def test_show_episode_summary_no_records(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    show_episode_summary(tmp_path)
    out = capsys.readouterr().out
    assert "no agent_step records" in out


# ---- show_dream_rollout --------------------------------------------------


def test_show_dream_rollout_default_picks_most_recent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rollouts = [
        _make_dream_rollout(seed_step=100, horizon=3),
        _make_dream_rollout(seed_step=200, horizon=3),
        _make_dream_rollout(seed_step=300, horizon=3),
    ]
    _write_dream_rollouts(rollouts, tmp_path)
    show_dream_rollout(tmp_path)
    out = capsys.readouterr().out
    assert "seed_step=300" in out
    # Index resolution should print "index 2 of 3".
    assert "index 2 of 3" in out


def test_show_dream_rollout_with_decoded_obs_renders_ascii(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rollouts = [_make_dream_rollout(seed_step=42, horizon=2, with_decoded_obs=True)]
    _write_dream_rollouts(rollouts, tmp_path)
    show_dream_rollout(tmp_path)
    out = capsys.readouterr().out
    assert "decoded observations" in out
    # Two steps → two rendered frames.
    assert "step 1" in out
    assert "step 2" in out
    # Each frame has 16 lines of 16 characters using the ramp.
    # Look for at least one ramp character.
    assert any(c in out for c in " .:-=+*#%@")


def test_show_dream_rollout_without_decoded_obs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rollouts = [_make_dream_rollout(seed_step=42, horizon=3, with_decoded_obs=False)]
    _write_dream_rollouts(rollouts, tmp_path)
    show_dream_rollout(tmp_path)
    out = capsys.readouterr().out
    assert "(none recorded)" in out


def test_show_dream_rollout_index_out_of_range(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rollouts = [_make_dream_rollout(seed_step=10, horizon=2)]
    _write_dream_rollouts(rollouts, tmp_path)
    show_dream_rollout(tmp_path, rollout_index=99)
    out = capsys.readouterr().out
    assert "out of range" in out


def test_show_dream_rollout_no_records(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    show_dream_rollout(tmp_path)
    out = capsys.readouterr().out
    assert "no dream_rollout records" in out


# ---- show_run_summary ----------------------------------------------------


def test_show_run_summary_full_telemetry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ep0 = _episode_records(episode_id=0, n_steps=4)
    ep1 = _episode_records(episode_id=1, n_steps=4, starting_t=4)
    _write_agent_steps(ep0 + ep1, tmp_path)
    rollouts = [_make_dream_rollout(seed_step=4, horizon=3)]
    _write_dream_rollouts(rollouts, tmp_path)
    events = [
        _make_world_event(t_event=0, event_type="env_reset", source="environment"),
        _make_world_event(t_event=4, event_type="env_reset", source="environment"),
    ]
    _write_world_events(events, tmp_path)

    show_run_summary(tmp_path)
    out = capsys.readouterr().out

    assert "total agent_step records: 8" in out
    assert "total episodes: 2" in out
    assert "total dream_rollout records: 1" in out
    assert "world_event total: 2" in out
    # Action labels should appear at the run level too.
    assert "up=" in out
    # KL trend section should show early/late means.
    assert "kl_aggregate_t early/late" in out


def test_show_run_summary_with_no_telemetry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    show_run_summary(tmp_path)
    out = capsys.readouterr().out
    assert "agent_step: 0 records" in out
    assert "total dream_rollout records: 0" in out


# ---- digest regression ---------------------------------------------------


def test_build_digest_lift_preserves_structure(
    tmp_path: Path,
) -> None:
    """The digest function lifted from mirror/caller to observer/digest must
    produce the same shape on the same input. This test calls
    ``build_digest`` directly and verifies the markers the mirror caller's
    own digest tests pin. A regression here means the lift altered behaviour.
    """
    records = [
        r.model_dump()
        for r in _episode_records(episode_id=0, n_steps=10)
    ]
    text = build_digest(records)
    # Top-level structure markers.
    assert "# Telemetry Digest (agent_step records)" in text
    assert f"schema_version: {PROBE_1_SCHEMA_VERSION}" in text
    assert "## episode 0" in text
    # Per-episode summary sections (mirror caller test pins these by name).
    assert "step count: 10" in text
    assert "kl_aggregate_t" in text
    assert "recon_loss_t" in text
    assert "intrinsic_signal_t" in text
    assert "policy_entropy_t" in text
    assert "action_t distribution" in text
    # Sample records use the first/middle/last positions.
    assert '"__pos": "first"' in text
    assert '"__pos": "middle"' in text
    assert '"__pos": "last"' in text


def test_build_digest_empty_returns_no_records_marker() -> None:
    assert build_digest([]) == "(no records)"


def test_compact_record_repr_excludes_high_dim_fields() -> None:
    """The lifted ``compact_record_repr`` must continue to exclude the
    high-dim vectors. Same property the mirror caller's pre-lift tests
    pinned; re-pinned here so a regression in observer/digest.py is
    caught directly."""
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
    assert {"t", "kl_aggregate_t", "action_t", "obs_hash_t"}.issubset(parsed.keys())


# ---- formatting helpers --------------------------------------------------


def test_format_scalar_rounds_floats_to_four_decimals() -> None:
    assert _format_scalar(1.234567) == "1.2346"
    assert _format_scalar(0.0) == "0.0000"
    assert _format_scalar(42) == "42"
    assert _format_scalar("hi") == "hi"
    assert _format_scalar(None) == "None"


def test_format_scalar_uses_scientific_for_extreme_magnitudes() -> None:
    # Very small values switch to scientific notation.
    assert "e" in _format_scalar(1e-7)
    # Very large values switch to scientific notation.
    assert "e" in _format_scalar(1e8)


def test_format_scalar_handles_nan_and_inf() -> None:
    assert _format_scalar(float("nan")) == "nan"
    assert _format_scalar(float("inf")) == "inf"
    assert _format_scalar(float("-inf")) == "-inf"


def test_format_action_distribution_uses_named_actions() -> None:
    # 3 ups, 2 stays, nothing else.
    s = _format_action_distribution([0, 0, 0, 4, 4])
    assert "up=3" in s
    assert "stay=2" in s
    assert "down=0" in s


def test_format_duration_ms_handles_units() -> None:
    assert _format_duration_ms(0) == "0ms"
    assert _format_duration_ms(500) == "500ms"
    assert _format_duration_ms(1_500).endswith("s")
    assert _format_duration_ms(120_000).endswith("m")
    assert _format_duration_ms(7_200_000).endswith("h")


def test_format_duration_ms_handles_negative() -> None:
    assert _format_duration_ms(-500).startswith("-")


def test_summarise_vector_includes_shape_and_stats() -> None:
    out = _summarise_vector([0.0, 1.0, 2.0, 3.0, 4.0])
    assert "shape=(5,)" in out
    assert "mean=" in out
    assert "min=" in out
    assert "max=" in out


def test_summarise_vector_handles_pair_of_lists() -> None:
    """``q_params_t`` is a tuple of two lists (mu, log_sigma); the vector
    summariser recurses one level so both summaries print side by side."""
    out = _summarise_vector(([0.0, 1.0, 2.0], [3.0, 4.0, 5.0]))
    # Two summaries separated by " | " in one parenthesised group.
    assert out.startswith("(")
    assert " | " in out


# ---- ASCII rendering -----------------------------------------------------


def test_render_ascii_frame_produces_16x16() -> None:
    """A 32×32 uint8 frame downsamples to 16 lines of 16 characters."""
    buf = bytes(b * 0 for b in range(32 * 32))  # all zeros
    lines = _render_ascii_frame(buf)
    assert len(lines) == 16
    for line in lines:
        assert len(line) == 16


def test_render_ascii_frame_intensity_extremes() -> None:
    """All-black bytes should map to the lowest ramp char (' '); all-white
    bytes to the highest ('@')."""
    black = bytes([0] * 1024)
    white = bytes([255] * 1024)
    black_lines = _render_ascii_frame(black)
    white_lines = _render_ascii_frame(white)
    assert black_lines[0][0] == " "
    assert white_lines[0][0] == "@"


def test_render_ascii_frame_rejects_wrong_size() -> None:
    out = _render_ascii_frame(bytes(100))
    assert len(out) == 1
    assert "unexpected size 100" in out[0]


# ---- CLI -----------------------------------------------------------------


def test_cli_recent_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = _episode_records(episode_id=0, n_steps=4)
    _write_agent_steps(records, tmp_path)
    rc = main(["recent", str(tmp_path), "-n", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "last 2 of 4 records" in out


def test_cli_events_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    events = [
        _make_world_event(t_event=0, event_type="env_reset", source="environment"),
        _make_world_event(t_event=10, event_type="builder_perturbation", source="builder"),
    ]
    _write_world_events(events, tmp_path)
    rc = main(["events", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "env_reset: 1" in out
    assert "builder_perturbation: 1" in out


def test_cli_episode_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = _episode_records(episode_id=2, n_steps=3)
    _write_agent_steps(records, tmp_path)
    rc = main(["episode", str(tmp_path), "-e", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "episode 2" in out


def test_cli_dream_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rollouts = [_make_dream_rollout(seed_step=99, horizon=2)]
    _write_dream_rollouts(rollouts, tmp_path)
    rc = main(["dream", str(tmp_path), "-i", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "seed_step=99" in out


def test_cli_summary_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = _episode_records(episode_id=0, n_steps=4)
    _write_agent_steps(records, tmp_path)
    rc = main(["summary", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "run summary" in out
    assert "total agent_step records: 4" in out


def test_cli_help_does_not_crash(capsys: pytest.CaptureFixture[str]) -> None:
    """``--help`` exits with SystemExit(0); the parser has all five commands."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for command in ("recent", "events", "episode", "dream", "summary"):
        assert command in out


def test_cli_unknown_command_fails(capsys: pytest.CaptureFixture[str]) -> None:
    """argparse rejects unknown commands with exit 2."""
    with pytest.raises(SystemExit) as exc:
        main(["nonsense", "/tmp"])
    assert exc.value.code != 0


# ---- journal scaffolding -------------------------------------------------


JOURNAL_README = REPO_ROOT / "docs" / "workingjournal" / "README.md"


def test_journal_readme_exists() -> None:
    """Phase 7 journal scaffolding: docs/workingjournal/README.md exists."""
    assert JOURNAL_README.is_file(), f"missing journal README at {JOURNAL_README}"


def test_journal_readme_documents_structure_and_conventions() -> None:
    """The README should explicitly name the per-probe / pre-probe layout
    and the entry conventions used across the existing journal entries."""
    text = JOURNAL_README.read_text()
    # Title and the three sections the user brief named.
    assert "# Kind" in text
    assert "Working Journal" in text
    assert "## Structure" in text
    assert "## Entry conventions" in text
    # Names the existing files so future builders can find them.
    assert "probe1.md" in text
    assert "pre-probe1.md" in text


# ---- dependency lint regression ------------------------------------------


def test_actor_module_still_does_not_import_telemetry_view() -> None:
    """Phase 3c's lint must continue to hold after Phase 7 lifts the
    digest into ``kind/observer/digest.py`` and adds the eyeball helpers.
    Neither change touches the actor module; this test re-runs the lint
    here so a regression that does is caught even when test_views.py
    isn't in the run."""
    import ast

    actor_path = REPO_ROOT / "kind" / "agents" / "actor.py"
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


def test_eyeball_module_exposes_named_helpers() -> None:
    """Public surface check: the seven helpers are in the module's
    ``__all__`` so ``from kind.observer.eyeball import *`` is the
    documented surface. Probe 1.5 v2 (plan §2.7) added
    ``show_self_prediction`` and ``show_self_prediction_conditioning``
    to the original Probe 1 five."""
    expected = {
        "show_recent_agent_steps",
        "count_world_events",
        "show_episode_summary",
        "show_dream_rollout",
        "show_run_summary",
        "show_self_prediction",
        "show_self_prediction_conditioning",
    }
    assert set(eyeball.__all__) == expected
