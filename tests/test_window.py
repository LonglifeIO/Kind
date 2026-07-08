"""Phase 11.5 — Window tests.

Coverage: the loaders deserialize each on-disk record type through its
Pydantic model; a malformed record surfaces as an error rather than a
crash; the state-inference heuristic resolves each of the four states
plus ``unknown``; the per-hour state-time breakdown and the pace
estimate compute correctly against synthetic activity; the HTTP server
answers 200 on its routes and 404 cleanly on a missing round; the
server opens no file for write across a full pass through every route;
and Window imports none of the mirror's production code-path modules.

The on-disk-fixture tests skip cleanly when the run artifacts are
absent, so a fresh checkout without the ``runs/`` tree still passes.
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from kind.mirror import (
    AdmissibilityVerdict,
    FaithfulnessResult,
    PassResult,
    ReadingSurface,
    RoundJudgment,
    RoundResult,
    StabilityResult,
)
from kind.window import loaders, state
from kind.window.server import create_app
from kind.window.state import IoState, StreamActivity

# ---------------------------------------------------------------------------
# On-disk fixtures.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUNS = _REPO_ROOT / "runs"
_PHASE_13 = _RUNS / "phase_13_calibration"
_PHASE_9 = _RUNS / "phase_9_judge_smoke"
_PHASE_10_STABILITY = _RUNS / "phase_10_stability_smoke" / "stability.jsonl"
_PROBE_1 = _RUNS / "probe1-20260503-123926"

_PHASE_13_ROUND_ID = "phase_13_probe_1_round"


def _require(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"on-disk fixture not present: {path}")


# ---------------------------------------------------------------------------
# Loaders — deserialization through the Pydantic models.
# ---------------------------------------------------------------------------


def test_load_round_results_deserializes() -> None:
    _require(_PHASE_13)
    outcomes = loaders.load_round_results(_PHASE_13)
    assert outcomes, "expected at least one round under mirror/rounds/"
    for outcome in outcomes:
        assert outcome.ok, f"round failed to load: {outcome.error}"
        assert isinstance(outcome.value, RoundResult)
        assert outcome.value.round_id


def test_load_round_judgments_deserializes() -> None:
    _require(_PHASE_9)
    outcomes = loaders.load_round_judgments(_PHASE_9)
    assert outcomes, "expected judgments under phase_9_judge_smoke"
    for outcome in outcomes:
        assert outcome.ok, f"judgment failed to load: {outcome.error}"
        assert isinstance(outcome.value, RoundJudgment)


def test_load_pass_results_deserializes() -> None:
    _require(_PROBE_1)
    outcomes = loaders.load_pass_results(_PROBE_1)
    assert outcomes, "expected a PassResult under probe1 mirror/passes/"
    for outcome in outcomes:
        assert outcome.ok, f"pass failed to load: {outcome.error}"
        assert isinstance(outcome.value, PassResult)


def test_load_stability_results_deserializes() -> None:
    _require(_PHASE_10_STABILITY)
    outcomes = loaders.load_stability_results(_PHASE_10_STABILITY)
    assert outcomes, "expected stability records in the JSONL"
    for outcome in outcomes:
        assert outcome.ok, f"stability record failed: {outcome.error}"
        assert isinstance(outcome.value, StabilityResult)


def test_load_agent_steps_deserializes() -> None:
    _require(_PROBE_1 / "telemetry" / "agent_step")
    outcomes = loaders.load_agent_steps(_PROBE_1)
    assert outcomes, "expected agent_step rows in the probe1 telemetry"
    assert all(o.ok for o in outcomes), "agent_step rows must deserialize"
    first = outcomes[0].value
    assert first is not None
    assert first.episode_id >= 0


def test_load_faithfulness_results_deserializes(tmp_path: Path) -> None:
    """No run writes a faithfulness JSONL on disk yet, so this exercises
    the loader against a synthetic record round-tripped through the
    FaithfulnessResult model — the same model the verifier emits."""
    record = FaithfulnessResult(
        criterion_id="reflexive_attention",
        reader_role="primary",
        pass_index=0,
        run_id="window-test",
        checkpoint_id="ckpt-000001",
        assignments=(),
        n_claims_total=0,
        n_resolved=0,
        n_unresolved_field=0,
        n_unresolved_value=0,
        n_unresolved_range=0,
        faithfulness_rate=1.0,
        admissible=True,
        wallclock_ms=0,
        notes="synthetic window-test record",
    )
    jsonl = tmp_path / "faithfulness.jsonl"
    jsonl.write_text(record.model_dump_json() + "\n", encoding="utf-8")
    outcomes = loaders.load_faithfulness_results(jsonl)
    assert len(outcomes) == 1
    assert outcomes[0].ok
    assert isinstance(outcomes[0].value, FaithfulnessResult)


def test_loader_surfaces_error_on_malformed_record(tmp_path: Path) -> None:
    """A round file that does not deserialize is folded into an error
    outcome — Window surfaces the failure rather than crashing."""
    rounds_dir = tmp_path / "mirror" / "rounds"
    rounds_dir.mkdir(parents=True)
    (rounds_dir / "broken.json").write_text("{not valid json", encoding="utf-8")
    outcomes = loaders.load_round_results(tmp_path)
    assert len(outcomes) == 1
    assert not outcomes[0].ok
    assert outcomes[0].value is None
    assert outcomes[0].error is not None
    assert "broken.json" in outcomes[0].error


# ---------------------------------------------------------------------------
# State inference.
# ---------------------------------------------------------------------------


_NOW = 1_000_000_000


def test_infer_state_waking() -> None:
    activity = {
        "agent_step": _NOW - 1_000,
        "dream_rollout": None,
        "replay_meta": None,
        "world_event": _NOW - 1_000,
    }
    assert state.infer_current_state(activity, now_ms=_NOW) is IoState.WAKING


def test_infer_state_dreaming() -> None:
    activity = {
        "agent_step": _NOW - 10 * 60 * 1000,
        "dream_rollout": _NOW - 1_000,
        "replay_meta": None,
        "world_event": None,
    }
    assert state.infer_current_state(activity, now_ms=_NOW) is IoState.DREAMING


def test_infer_state_dormant() -> None:
    activity = {
        "agent_step": None,
        "dream_rollout": None,
        "replay_meta": _NOW - 2_000,
        "world_event": None,
    }
    assert state.infer_current_state(activity, now_ms=_NOW) is IoState.DORMANT


def test_infer_state_paused() -> None:
    activity = {
        "agent_step": _NOW - 10 * 60 * 1000,
        "dream_rollout": None,
        "replay_meta": None,
        "world_event": None,
    }
    assert state.infer_current_state(activity, now_ms=_NOW) is IoState.PAUSED


def test_infer_state_unknown_when_heuristic_does_not_resolve() -> None:
    """dream_rollout and replay_meta both active without agent_step is
    not 'dreaming' and not 'dormant' — the heuristic surfaces unknown
    rather than guessing."""
    activity = {
        "agent_step": None,
        "dream_rollout": _NOW - 1_000,
        "replay_meta": _NOW - 1_000,
        "world_event": None,
    }
    assert state.infer_current_state(activity, now_ms=_NOW) is IoState.UNKNOWN


# ---------------------------------------------------------------------------
# State-time breakdown and pace.
# ---------------------------------------------------------------------------


def test_state_time_breakdown_matches_constructed() -> None:
    """A 24-hour window with one waking hour, one dreaming hour, one
    dormant hour, and 21 idle hours produces exactly that breakdown."""
    hour = 60 * 60 * 1000
    now = 24 * hour
    activity = StreamActivity(
        agent_step=(now - hour + 1,),       # hour bucket 0  -> waking
        dream_rollout=(now - 2 * hour + 1,),  # hour bucket 1 -> dreaming
        replay_meta=(now - 3 * hour + 1,),    # hour bucket 2 -> dormant
    )
    breakdown = state.bucket_activity_by_hour(
        activity, now_ms=now, window_hours=24
    )
    assert breakdown[IoState.WAKING] == 1
    assert breakdown[IoState.DREAMING] == 1
    assert breakdown[IoState.DORMANT] == 1
    assert breakdown[IoState.PAUSED] == 21
    assert breakdown[IoState.UNKNOWN] == 0
    assert sum(breakdown.values()) == 24


def test_pace_estimate() -> None:
    hour = 60 * 60 * 1000
    now = 100 * hour
    window_ms = 24 * hour
    events = [
        (0, now - 2 * hour),          # episode 0, in window
        (0, now - 3 * hour),          # episode 0 again, still in window
        (1, now - hour),              # episode 1, in window
        (2, now - window_ms - hour),  # episode 2, outside the window
    ]
    pace = state.pace_estimate(events, now_ms=now, window_hours=24)
    assert pace == pytest.approx(2 / 24)


def test_parse_run_start() -> None:
    parsed = state.parse_run_start("probe1-20260503-123926")
    assert parsed is not None
    assert (parsed.year, parsed.month, parsed.day) == (2026, 5, 3)
    assert (parsed.hour, parsed.minute, parsed.second) == (12, 39, 26)
    assert state.parse_run_start("phase_13_calibration") is None


# ---------------------------------------------------------------------------
# HTTP server.
# ---------------------------------------------------------------------------


def test_server_routes_respond_200() -> None:
    _require(_PHASE_13)
    app = create_app("phase_13_calibration", _PHASE_13)
    client = app.test_client()
    for route in ("/", "/rounds", "/audit", "/live"):
        response = client.get(route)
        assert response.status_code == 200, f"{route} -> {response.status_code}"


def test_live_route_renders_written_state(tmp_path: Path) -> None:
    """A snapshot written by the runner-side producer renders on /live:
    grid, Io marker, event feed, energy."""
    from kind.window.live import (
        LiveEventRow,
        LiveWindowState,
        write_live_state,
    )

    grid = [[0] * 8 for _ in range(8)]
    grid[2][3] = 2  # a resource
    grid[5][5] = 1  # a wall
    write_live_state(
        tmp_path,
        LiveWindowState(
            run_id="live-test",
            env_step=123,
            episode_id=4,
            step_in_episode=56,
            wallclock_ms=0,
            grid=grid,
            agent_pos=(1, 1),
            true_energy=0.5,
            recent_events=[
                LiveEventRow(
                    t_event=120,
                    source="builder",
                    event_type="builder_perturbation",
                    detail="add_resource [0, 7]",
                )
            ],
        ),
    )
    app = create_app("live-test", tmp_path)
    response = app.test_client().get("/live")
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "env step" in page and "123" in page
    assert "builder_perturbation" in page
    assert "0.500" in page


def test_live_route_surfaces_malformed_state(tmp_path: Path) -> None:
    from kind.window.live import live_state_path

    path = live_state_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    response = create_app("live-test", tmp_path).test_client().get("/live")
    assert response.status_code == 200
    assert "unreadable" in response.get_data(as_text=True)


def test_server_round_detail_responds_200() -> None:
    _require(_PHASE_13)
    app = create_app("phase_13_calibration", _PHASE_13)
    client = app.test_client()
    response = client.get(f"/rounds/{_PHASE_13_ROUND_ID}")
    assert response.status_code == 200


def test_server_404_on_nonexistent_round() -> None:
    _require(_PHASE_13)
    app = create_app("phase_13_calibration", _PHASE_13)
    client = app.test_client()
    assert client.get("/rounds/no_such_round").status_code == 404
    assert client.get("/judgments/no_such_round").status_code == 404


def test_server_opens_no_file_for_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full pass through every route opens no file for write.

    Mirrors Phase 8's read-only invariant test: wrap ``Path.open`` and
    assert no recorded open used a write mode.
    """
    _require(_PHASE_13)
    write_paths: list[Path] = []
    real_open = Path.open

    def tracking_open(
        self: Path, mode: str = "r", *args: Any, **kwargs: Any
    ) -> Any:
        if any(ch in mode for ch in ("w", "a", "x", "+")):
            write_paths.append(Path(str(self)))
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)

    app = create_app("phase_13_calibration", _PHASE_13)
    client = app.test_client()
    for route in (
        "/",
        "/live",
        "/rounds",
        f"/rounds/{_PHASE_13_ROUND_ID}",
        "/audit",
        "/judgments/no_such_round",
    ):
        client.get(route)

    assert write_paths == [], (
        f"Window opened files for write during a route pass: "
        f"{write_paths}. The viewer is read-only."
    )


def test_aggregate_llm_audit_sums_round_records() -> None:
    _require(_PHASE_13)
    audit = loaders.aggregate_llm_audit(_PHASE_13)
    expected = sum(
        len(o.value.llm_call_records)
        for o in loaders.load_round_results(_PHASE_13)
        if o.value is not None
    )
    assert audit.total_calls == expected


# ---------------------------------------------------------------------------
# Phase 12 — admissibility integration.
# ---------------------------------------------------------------------------


def _faithfulness_record(
    *,
    criterion_id: str = "reflexive_attention",
    checkpoint_id: str = "ckpt-1",
    pass_index: int = 0,
    admissible: bool = True,
    rate: float = 1.0,
) -> FaithfulnessResult:
    return FaithfulnessResult(
        criterion_id=criterion_id,
        reader_role="primary",
        pass_index=pass_index,
        run_id="window-test",
        checkpoint_id=checkpoint_id,
        assignments=(),
        n_claims_total=0,
        n_resolved=0,
        n_unresolved_field=0,
        n_unresolved_value=0,
        n_unresolved_range=0,
        faithfulness_rate=rate,
        admissible=admissible,
        wallclock_ms=0,
        notes="window-test faithfulness record",
    )


def _stability_record(
    *,
    criterion_id: str = "reflexive_attention",
    checkpoint_id: str = "ckpt-1",
) -> StabilityResult:
    surfaces = {ReadingSurface.HEAD_INTERNAL: True}
    return StabilityResult(
        paraphrase_agreement_per_surface={s: 1.0 for s in surfaces},
        reseed_agreement_per_surface={s: 1.0 for s in surfaces},
        n_paraphrases=2,
        n_reseeds=2,
        structured_field_agreement_per_claim=(),
        admissible_per_surface=dict(surfaces),
        paraphrase_readings=(),
        reseed_readings=(),
        criterion_id=criterion_id,
        reader_role="primary",
        run_id="window-test",
        checkpoint_id=checkpoint_id,
        wallclock_ms=0,
    )


def test_load_admissibility_records_deserializes_correctly(
    tmp_path: Path,
) -> None:
    """No run writes an admissibility JSONL on disk yet, so this
    exercises the loader against a synthetic record round-tripped
    through the AdmissibilityVerdict model."""
    mirror_dir = tmp_path / "mirror"
    mirror_dir.mkdir(parents=True)
    verdict = AdmissibilityVerdict(
        pass_index=0,
        criterion_id="reflexive_attention",
        reader_role="primary",
        run_id="window-test",
        checkpoint_id="ckpt-1",
        faithfulness_admissible=True,
        faithfulness_rate=1.0,
        stability_admissible_per_surface={ReadingSurface.HEAD_INTERNAL: True},
        stability_admissible_all_surfaces=True,
        admissible=True,
        notes="window-test verdict",
        wallclock_ms=0,
    )
    (mirror_dir / "admissibility.jsonl").write_text(
        verdict.model_dump_json() + "\n", encoding="utf-8"
    )
    outcomes = loaders.load_admissibility_records(tmp_path)
    assert len(outcomes) == 1
    assert outcomes[0].ok
    assert isinstance(outcomes[0].value, AdmissibilityVerdict)
    # A run with no admissibility.jsonl yields an empty list.
    assert loaders.load_admissibility_records(tmp_path / "nope") == []


def test_admissibility_route_responds_200() -> None:
    _require(_PHASE_13)
    app = create_app("phase_13_calibration", _PHASE_13)
    client = app.test_client()
    response = client.get("/admissibility")
    assert response.status_code == 200


def test_admissibility_route_shows_inadmissibility_breakdowns(
    tmp_path: Path,
) -> None:
    """A faithfulness-inadmissible reading with a stability-admissible
    match lands in the faithfulness-only inadmissibility bucket; the
    /admissibility view surfaces the breakdown and the per-verdict
    row."""
    mirror_dir = tmp_path / "mirror"
    mirror_dir.mkdir(parents=True)
    (mirror_dir / "faithfulness.jsonl").write_text(
        _faithfulness_record(admissible=False, rate=0.5).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    (mirror_dir / "stability.jsonl").write_text(
        _stability_record().model_dump_json() + "\n", encoding="utf-8"
    )
    app = create_app("window-test", tmp_path)
    client = app.test_client()
    response = client.get("/admissibility")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "inadmissible — faithfulness only" in body
    assert "reflexive_attention" in body
    assert "inadmissible" in body


def test_rounds_route_surfaces_per_reading_verdict_when_present(
    tmp_path: Path,
) -> None:
    """When an admissibility record matches a pass, the round-detail
    view surfaces the verdict inline alongside that pass's readings."""
    source_round = (
        _PHASE_13 / "mirror" / "rounds" / f"{_PHASE_13_ROUND_ID}.json"
    )
    _require(source_round)

    rounds_dir = tmp_path / "mirror" / "rounds"
    rounds_dir.mkdir(parents=True)
    shutil.copy(source_round, rounds_dir / f"{_PHASE_13_ROUND_ID}.json")

    # Read the copied round to learn a real (pass_index, checkpoint_id)
    # to anchor the synthetic verdict to.
    round_result = loaders.load_round_results(tmp_path)[0].value
    assert isinstance(round_result, RoundResult)
    assert round_result.pass_results, "expected at least one pass"
    checkpoint_id = round_result.pass_results[0].checkpoint_id

    verdict = AdmissibilityVerdict(
        pass_index=0,
        criterion_id="reflexive_attention",
        reader_role="primary",
        run_id="window-test",
        checkpoint_id=checkpoint_id,
        faithfulness_admissible=True,
        faithfulness_rate=0.92,
        stability_admissible_per_surface={ReadingSurface.HEAD_INTERNAL: True},
        stability_admissible_all_surfaces=True,
        admissible=True,
        notes="window-test inline verdict",
        wallclock_ms=0,
    )
    (tmp_path / "mirror" / "admissibility.jsonl").write_text(
        verdict.model_dump_json() + "\n", encoding="utf-8"
    )

    app = create_app("window-test", tmp_path)
    client = app.test_client()
    response = client.get(f"/rounds/{_PHASE_13_ROUND_ID}")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "admissibility verdicts" in body
    assert "window-test inline verdict" in body


# ---------------------------------------------------------------------------
# Import discipline.
# ---------------------------------------------------------------------------


def test_window_does_not_import_mirror_production_modules() -> None:
    """Window reads records; it does not construct or trigger any of the
    mirror's production code paths. It must not import
    ``kind.mirror.orchestrator``, ``kind.mirror.judge_driver``, or
    ``kind.mirror.calibration.smoke`` — the models it needs are
    re-exported from the ``kind.mirror`` package surface.

    The check parses each Window module's AST and inspects import
    statements only — a docstring that *names* a forbidden module (the
    loaders module's docstring does, to explain the discipline) is not
    a violation.
    """
    forbidden = (
        "kind.mirror.orchestrator",
        "kind.mirror.judge_driver",
        "kind.mirror.calibration.smoke",
    )
    window_dir = _REPO_ROOT / "kind" / "window"
    for module_path in sorted(window_dir.glob("*.py")):
        tree = ast.parse(
            module_path.read_text(encoding="utf-8"), filename=str(module_path)
        )
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.append(node.module)
        for name in imported:
            for bad in forbidden:
                assert not name.startswith(bad), (
                    f"{module_path.name} imports {name!r}, which is a "
                    f"mirror production module Window must not import."
                )
