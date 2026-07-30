"""Windowed-mode vetting for the V2 mirror round (2026-07-30).

The continuing-world biography (frozen ``episode_id``, ~470k rows)
broke three assumptions the Phase-8 orchestrator was built on: telemetry
small enough to load whole, per-episode analysis slices, and a
perturbation log with no dream/pause wallclock gaps. The windowed mode
(``PassConfig.window_steps`` + ``StatisticConfig.chunk_steps`` +
skip-unmatched alignment) is the dated fix
(``docs/decisions/mirror_v2_windowed_mode_2026-07-30.md``). These tests
pin: chunked analysis slices, the windowed loaders (dedupe-by-t, dream
seed filtering), the aligner's window/skip extensions, and an
end-to-end windowed pass on a continuing-world-shaped run dir.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kind.mirror.orchestrator import (
    PassConfig,
    _load_agent_step_rows,
    _load_dream_rollout_rows,
    run_adversarial_pass,
)
from kind.mirror.perturbation_align import (
    PerturbationAlignmentError,
    align_perturbations,
)
from kind.mirror.registry import CriterionRegistry
from kind.mirror.criteria_v2 import V2_REGISTRY
from kind.mirror.llm_caller import LLMConfig
from kind.mirror.statistics import (
    StatisticConfig,
    _analysis_slices,
    _episode_slices,
)
from tests.test_orchestrator import (
    _build_agent_step_rows,
    _build_mock_client_for_round,
    _write_agent_step_shards,
    _write_world_event_log,
)

# ---------------------------------------------------------------------------
# _analysis_slices
# ---------------------------------------------------------------------------


def _rows_with_episodes(spec: list[tuple[int, int]]) -> list[dict[str, Any]]:
    """``spec`` = [(episode_id, n_rows), ...] → minimal sorted rows."""
    rows: list[dict[str, Any]] = []
    t = 0
    for eid, n in spec:
        for _ in range(n):
            rows.append({"t": t, "episode_id": eid})
            t += 1
    return rows


def test_analysis_slices_none_is_episode_slices() -> None:
    rows = _rows_with_episodes([(0, 5), (1, 7)])
    assert _analysis_slices(rows, None) == _episode_slices(rows)


def test_analysis_slices_chunks_within_episodes() -> None:
    rows = _rows_with_episodes([(0, 5), (1, 7)])
    assert _analysis_slices(rows, 3) == [
        (0, 3),
        (3, 5),
        (5, 8),
        (8, 11),
        (11, 12),
    ]


def test_analysis_slices_never_bridges_episode_boundary() -> None:
    rows = _rows_with_episodes([(0, 4), (1, 4)])
    for start, end in _analysis_slices(rows, 3):
        eids = {int(rows[i]["episode_id"]) for i in range(start, end)}
        assert len(eids) == 1


def test_analysis_slices_rejects_nonpositive_chunk() -> None:
    rows = _rows_with_episodes([(0, 4)])
    with pytest.raises(ValueError, match="chunk_steps must be positive"):
        _analysis_slices(rows, 0)


# ---------------------------------------------------------------------------
# Windowed loaders
# ---------------------------------------------------------------------------


def test_windowed_agent_step_load_dedupes_by_t(tmp_path: Path) -> None:
    telemetry_dir = tmp_path / "telemetry"
    rows = _build_agent_step_rows(n_steps=100, n_episodes=1)
    # A stray duplicated row (the biography's resume-buffer quirk):
    # same t as rows[90], different wallclock so keep-first is
    # observable.
    stray = dict(rows[90])
    stray["wallclock_ms"] = 999_999
    _write_agent_step_shards(telemetry_dir, rows + [stray])

    loaded = _load_agent_step_rows(telemetry_dir, window_steps=50)
    ts = [int(r["t"]) for r in loaded]
    assert ts == sorted(set(ts)), "duplicate t survived the windowed load"
    assert ts[-1] == 99 and ts[0] >= 50
    by_t = {int(r["t"]): r for r in loaded}
    assert int(by_t[90]["wallclock_ms"]) == 90 * 100, "keep-first violated"


def test_dream_rollout_load_filters_by_seed_step(tmp_path: Path) -> None:
    shard_dir = tmp_path / "telemetry" / "dream_rollout"
    shard_dir.mkdir(parents=True)
    rows = [{"seed_step": s, "dream_session_id": f"d{s}"} for s in (10, 60, 90)]
    with (shard_dir / "shard-000000.parquet").open("wb") as fh:
        pq.write_table(pa.Table.from_pylist(rows), fh)  # type: ignore[no-untyped-call]

    loaded = _load_dream_rollout_rows(tmp_path / "telemetry", t_range=(50, 95))
    assert [int(r["seed_step"]) for r in loaded] == [60, 90]
    all_rows = _load_dream_rollout_rows(tmp_path / "telemetry")
    assert len(all_rows) == 3


# ---------------------------------------------------------------------------
# Aligner window/skip extensions
# ---------------------------------------------------------------------------


def _perturbation(t_event: int, wallclock_ms: int) -> dict[str, Any]:
    return {
        "schema_version": "0.2.0",
        "run_id": "probe2-orch-test",
        "checkpoint_id": "ckpt-000001",
        "t_event": t_event,
        "event_type": "builder_perturbation",
        "source": "builder",
        "payload": {"kind": "test"},
        "wallclock_ms": wallclock_ms,
    }


def _aligner_fixture(tmp_path: Path) -> tuple[Path, Path]:
    telemetry_dir = tmp_path / "telemetry"
    rows = _build_agent_step_rows(n_steps=200, n_episodes=1)
    _write_agent_step_shards(telemetry_dir, rows)
    _write_world_event_log(
        telemetry_dir,
        [
            _perturbation(20, 20 * 100),        # outside the window below
            _perturbation(150, 150 * 100),      # aligns cleanly
            _perturbation(151, 150 * 100 + 3),  # collides onto t=150
            _perturbation(170, 10_000_000),     # beyond any tolerance
        ],
    )
    return telemetry_dir / "world_event.jsonl", telemetry_dir


def test_aligner_t_range_filters_events(tmp_path: Path) -> None:
    world_path, telemetry_dir = _aligner_fixture(tmp_path)
    timeline = align_perturbations(
        world_path,
        telemetry_dir,
        run_id="probe2-orch-test",
        checkpoint_id="ckpt-000001",
        t_range=(100, 199),
        skip_unmatched=True,
    )
    assert [e.t for e in timeline.events] == [150]
    assert timeline.skipped_unmatched == 1   # the 10_000_000 ms event
    assert timeline.skipped_collisions == 1  # the t=151 event onto t=150


def test_aligner_default_mode_still_raises(tmp_path: Path) -> None:
    world_path, telemetry_dir = _aligner_fixture(tmp_path)
    with pytest.raises(PerturbationAlignmentError):
        align_perturbations(
            world_path,
            telemetry_dir,
            run_id="probe2-orch-test",
            checkpoint_id="ckpt-000001",
        )


def test_aligner_accepts_preloaded_rows(tmp_path: Path) -> None:
    world_path, telemetry_dir = _aligner_fixture(tmp_path)
    rows = _load_agent_step_rows(telemetry_dir, window_steps=100)
    timeline = align_perturbations(
        world_path,
        telemetry_dir / "nonexistent",  # would fail if actually read
        run_id="probe2-orch-test",
        checkpoint_id="ckpt-000001",
        agent_step_rows=rows,
        t_range=(int(rows[0]["t"]), int(rows[-1]["t"])),
        skip_unmatched=True,
    )
    assert [e.t for e in timeline.events] == [150]


# ---------------------------------------------------------------------------
# End-to-end windowed pass on a continuing-world-shaped run dir
# ---------------------------------------------------------------------------


def test_windowed_pass_end_to_end(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "probe2-orch-test"
    telemetry_dir = run_dir / "telemetry"
    # Continuing-world shape: one frozen episode_id over 400 rows.
    rows = _build_agent_step_rows(n_steps=400, n_episodes=1)
    _write_agent_step_shards(telemetry_dir, rows)
    _write_world_event_log(
        telemetry_dir,
        [
            _perturbation(50, 50 * 100),    # outside the 200-step window
            _perturbation(300, 300 * 100),  # inside
            _perturbation(390, 10_000_000), # inside but unalignable
        ],
    )
    config = PassConfig(
        run_id="probe2-orch-test",
        checkpoint_id="ckpt-000001",
        run_dir=run_dir,
        active_registry=CriterionRegistry(criteria=V2_REGISTRY.active()),
        held_out_registry=CriterionRegistry(criteria=V2_REGISTRY.held_out()),
        statistic_config=StatisticConfig(chunk_steps=50),
        llm_config=LLMConfig(),
        window_steps=200,
    )
    client = _build_mock_client_for_round()
    result = run_adversarial_pass(config, llm_client=client)

    # The batch was the window, not the run.
    for stat in result.statistic_results:
        assert stat.n_samples <= 200, stat.signal_name
    # Only the in-window, alignable perturbation survived.
    assert [e.t for e in result.perturbation_timeline.events] == [300]
    assert result.perturbation_timeline.skipped_unmatched == 1
    # The window metadata is recorded on the pass.
    assert "windowed pass: window_steps=200" in result.notes
    assert "chunk_steps=50" in result.notes
    # Pass artifacts land under mirror/ as in the legacy mode.
    passes_dir = run_dir / "mirror" / "passes"
    assert any(passes_dir.glob("*.json"))


def test_full_mode_unchanged_by_default(tmp_path: Path) -> None:
    """window_steps=None keeps the legacy full-stream behavior."""
    run_dir = tmp_path / "runs" / "probe2-orch-test"
    telemetry_dir = run_dir / "telemetry"
    rows = _build_agent_step_rows(n_steps=200, n_episodes=2)
    _write_agent_step_shards(telemetry_dir, rows)
    _write_world_event_log(telemetry_dir, [_perturbation(60, 60 * 100)])
    config = PassConfig(
        run_id="probe2-orch-test",
        checkpoint_id="ckpt-000001",
        run_dir=run_dir,
        active_registry=CriterionRegistry(criteria=V2_REGISTRY.active()),
        held_out_registry=CriterionRegistry(criteria=V2_REGISTRY.held_out()),
        llm_config=LLMConfig(),
    )
    client = _build_mock_client_for_round()
    result = run_adversarial_pass(config, llm_client=client)
    assert "windowed pass" not in result.notes
    assert result.perturbation_timeline.skipped_unmatched == 0
    max_n = max(s.n_samples for s in result.statistic_results)
    assert max_n == 200
