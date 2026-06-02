"""Phase 8a load-bearing test — the live four-state cycle, end to end.

The pre-8a integration smokes (``test_integration_smoke.py``) only ever exercise
*waking* (desktop on throughout). This smoke drives the supervisor through a full
``waking → dreaming → dormant → waking`` cycle via an injected ``ScriptedDesktop``
source (desktop on → off → on) and a deterministic clock, and asserts the
coverage the waking-only smokes lack:

* entering dreaming on ``desktop_off`` runs a **four-axis** dream session — a
  ``"0.3.0"`` ``DreamSessionMeta`` (double-written) plus ``DreamRollout`` records
  carrying a ``dream_session_id``, a replay seed, and a non-flat temperature
  schedule (distinct from the waking-planning ``"0.2.0"`` handshake);
* the session is **capped** by the Phase 6 composite (it terminates — here on the
  rollout-count cap — rather than running unbounded);
* the state transitions fire **in order** (waking→dreaming→dormant→waking) on the
  ``world_event`` stream, and dormant emits **heartbeats**;
* ``desktop_on`` **resumes waking**, and exactly ``total_env_steps`` *waking*
  steps run despite the dream interruption (the budget counts waking steps).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from kind.training.dream import DreamRolloutConfig
from kind.training.dream_seed import SeedSelectionConfig
from kind.training.runner import Runner, RunnerConfig
from kind.training.state_machine import DreamEnvelopeConfig, ScriptedDesktop
from tests.test_integration_smoke import (
    _quiet_grid_world_config,  # noqa: F401  (imported for parity / fixtures)
    _read_jsonl_records,
    _read_parquet_records,
    _tiny_world_model_config,
    _transport_pair,
)

_HEARTBEAT_MS = 100


def _stepping_clock(step_ms: int = _HEARTBEAT_MS) -> Callable[[], int]:
    """A monotonic clock advancing ``step_ms`` per call (one call per supervisor
    tick), so dormant heartbeats fire deterministically."""
    state = {"t": 0}

    def clock() -> int:
        value = state["t"]
        state["t"] += step_ms
        return value

    return clock


def _dream_cycle_config(tmp_path: Path) -> RunnerConfig:
    return RunnerConfig(
        world_model_config=_tiny_world_model_config(),
        run_id="probe3-cycle",
        telemetry_dir=tmp_path / "telemetry",
        checkpoints_dir=tmp_path / "checkpoints",
        action_dim=5,
        ensemble_k=2,
        imagination_horizon=4,
        replay_capacity=200,
        replay_sequence_length=4,
        replay_batch_size=2,
        train_every_n_env_steps=1,
        warmup_env_steps=5,
        # No waking-planning rollout in this short run — the dream_rollout stream
        # carries ONLY the four-axis records, so the assertions are unambiguous.
        dream_cadence_env_steps=10_000,
        dream_horizon=4,
        checkpoint_every_n_env_steps=10_000,
        parquet_rows_per_shard=10,
        device="cpu",
        # Live Phase 7 surface: small caps so the session terminates fast, a
        # short warmup window and zero min-age so the four-axis replay seed finds
        # valid windows in a short run, a small heartbeat interval.
        dream_envelope=DreamEnvelopeConfig(
            hard_cap_rollout_count=3,
            hard_cap_wallclock_ms=10**12,
            compute_budget_seconds_per_hour=10**9,
            dormant_heartbeat_interval_ms=_HEARTBEAT_MS,
        ),
        seed_selection=SeedSelectionConfig(
            mode="replay",
            replay_min_segment_age_steps=0,
            replay_warmup_length=2,
        ),
        dream_rollout_config=DreamRolloutConfig(horizon=6),
        # No real sleep during dormant (the scripted source + clock drive it).
        dormant_tick_interval_s=0.0,
    )


def test_full_waking_dreaming_dormant_waking_cycle(tmp_path: Path) -> None:
    config = _dream_cycle_config(tmp_path)
    total_env_steps = 24

    # Desktop: on for the first 12 waking steps (populate the buffer), then off
    # for 4 supervisor ticks (1 dreaming entry + 1 dormant entry + 2 dormant
    # heartbeats), then sticks on True so waking resumes and finishes the budget.
    desktop = ScriptedDesktop([True] * 12 + [False] * 4 + [True])

    with _transport_pair() as (client, _server, env_server, _thread):
        runner = Runner(
            config,
            client,
            env_server=env_server,
            host_signal_source=desktop,
            host_clock_ms=_stepping_clock(),
        )
        try:
            runner.run(total_env_steps=total_env_steps)
        finally:
            runner.close()

    telem = config.telemetry_dir
    agent_rows = _read_parquet_records(telem / "agent_step")
    dream_rows = _read_parquet_records(telem / "dream_rollout")
    session_rows = _read_jsonl_records(telem / "dream_session.jsonl")
    world_events = _read_jsonl_records(telem / "world_event.jsonl")

    # The budget counts WAKING steps — the dream interruption did not consume it.
    assert len(agent_rows) == total_env_steps, (
        f"expected {total_env_steps} waking agent_step records, got {len(agent_rows)}"
    )

    # --- The four-axis dream session ran and was double-written. ---
    assert len(session_rows) == 2, (
        f"expected a start + end DreamSessionMeta, got {len(session_rows)}"
    )
    start, end = session_rows
    assert start["dream_session_id"] == end["dream_session_id"]
    assert start["ended_at_env_step"] is None  # the in-flight start marker
    assert end["ended_at_env_step"] is not None  # the closure
    # Capped by the composite — it terminated, on the rollout-count cap.
    assert end["rollout_count"] == 3
    assert end["end_trigger"] == "hard_cap_rollout_count"
    # The live Phase 7 surface is the provenance snapshot.
    assert end["envelope_config_snapshot"]["hard_cap_rollout_count"] == 3
    assert end["seed_selection_config_snapshot"]["mode"] == "replay"

    # --- The dream rollouts are four-axis "0.3.0", not the waking handshake. ---
    four_axis = [r for r in dream_rows if r["schema_version"] == "0.3.0"]
    assert len(four_axis) == 3, f"expected 3 four-axis rollouts, got {len(four_axis)}"
    session_id = end["dream_session_id"]
    for r in four_axis:
        assert r["dream_session_id"] == session_id
        assert r["seed_kind"] == "replay"  # replay-seeded (axis 4)
        temp = r["temperature_schedule"]
        assert isinstance(temp, list) and len(temp) == 6  # the configured horizon
        assert max(temp) > min(temp)  # non-flat (axis 3) — not the identity regime
        # Ensemble disagreement recorded (axis 2) — length == horizon, finite.
        disagreement = r["sequence_ensemble_disagreement_variance"]
        assert isinstance(disagreement, list) and len(disagreement) == 6
        assert all(v >= 0.0 for v in disagreement)

    # --- The state transitions fired in order. ---
    transitions = [
        (e["payload"]["from_state"], e["payload"]["to_state"], e["payload"]["trigger"])
        for e in world_events
        if e["event_type"] == "state_transition"
    ]
    assert transitions == [
        ("waking", "dreaming", "desktop_off"),
        ("dreaming", "dormant", "hard_cap_rollout_count"),
        ("dormant", "waking", "desktop_on"),
    ], transitions

    # --- Dormant emitted heartbeats (the "resting, not paused" signal). ---
    heartbeats = [e for e in world_events if e["event_type"] == "dormant_heartbeat"]
    assert len(heartbeats) >= 1, "dormant emitted no heartbeats"
    assert all(e["payload"]["mac_alive"] is True for e in heartbeats)


def test_desktop_on_throughout_runs_pure_waking(tmp_path: Path) -> None:
    """The supervisor is transparent when the desktop never goes off: no dream
    session, no state-transition events, exactly the waking budget — the property
    that keeps the pre-8a smokes behaviorally unchanged, asserted directly."""
    config = _dream_cycle_config(tmp_path)
    with _transport_pair() as (client, _server, env_server, _thread):
        runner = Runner(config, client, env_server=env_server)  # default: always awake
        try:
            runner.run(total_env_steps=15)
        finally:
            runner.close()

    telem = config.telemetry_dir
    assert len(_read_parquet_records(telem / "agent_step")) == 15
    assert _read_jsonl_records(telem / "dream_session.jsonl") == []
    transitions = [
        e
        for e in _read_jsonl_records(telem / "world_event.jsonl")
        if e["event_type"] == "state_transition"
    ]
    assert transitions == []
