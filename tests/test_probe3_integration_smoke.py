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

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from kind.training.dream import DreamRolloutConfig
from kind.training.dream_seed import SeedSelectionConfig
from kind.training.runner import Runner, RunnerConfig
from kind.training.state_machine import (
    AlwaysAwakeDesktop,
    DreamEnvelopeConfig,
    ScriptedDesktop,
)
from tests.test_integration_smoke import (
    _quiet_grid_world_config,  # noqa: F401  (imported for parity / fixtures)
    _read_jsonl_records,
    _read_parquet_records,
    _tiny_world_model_config,
    _transport_pair,
)

_HEARTBEAT_MS = 100
# The deterministic clock advances this much per supervisor tick; the controller
# measures each dream session at exactly one step (two consecutive clock reads
# around the synchronous driver call). 1000 ms/session keeps the recorded session
# comfortably above the cold-ledger seed projection (rollout_count × the ~110 ms
# DEFAULT seed = 0.33 s), so the rollout-count cap fires cleanly on session 1.
_SESSION_STEP_MS = 1000


def _stepping_clock(step_ms: int = _SESSION_STEP_MS) -> Callable[[], int]:
    """A monotonic clock advancing ``step_ms`` per call (one call per supervisor
    tick), so dormant heartbeats fire deterministically."""
    state = {"t": 0}

    def clock() -> int:
        value = state["t"]
        state["t"] += step_ms
        return value

    return clock


def _dream_cycle_config(
    tmp_path: Path,
    *,
    metabolic_capacity_seconds: float = 1.2,
    metabolic_refill_rate: float = 0.0,
) -> RunnerConfig:
    # The controller measures each dream session at one clock step = 1.0 s and
    # spends it from the metabolic token bucket. Defaults give the single-session
    # case: C=1.2 s spends down past a full session (≈1.0 s) after one 3-rollout
    # session, and R=0 never refills ⇒ no re-entry (rest). A larger C + positive R
    # produce a periodic dream-burst / rest rhythm (the rhythm-shape test).
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
            metabolic_capacity_seconds=metabolic_capacity_seconds,
            metabolic_refill_rate=metabolic_refill_rate,
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
    # The *bucket-depleted → rest* case: the default C=1.2 s bucket drops below a
    # full session (≈1.0 s) after one 3-rollout session, and R=0 never refills, so
    # no metabolic re-entry fires and the absence stays single-session — the 8a
    # cycle, now with the token-bucket pacer present but depleted. (The periodic
    # re-dream rhythm is the next test.)
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


def _longest_small_run(gaps: list[int], rest_ms: int) -> int:
    """Longest run of consecutive within-burst (small) re-entry gaps — the size
    (minus 1) of the largest dream burst. A token bucket bounds this by capacity;
    the 8b rolling-window pacer front-loaded it to ~100."""
    longest = 0
    run = 0
    for g in gaps:
        if g < rest_ms:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return longest


def test_metabolic_rhythm_shape_periodic_bursts_no_trickle(tmp_path: Path) -> None:
    """Phase 8 load-bearing — the token-bucket rhythm SHAPE (catches burst/trickle).

    Over a long absence the metabolic loop must produce **periodic** dream-bursts
    and dormant rests, every session **full** (no 1-rollout trickle), each burst
    **bounded** (not all sessions front-loaded). This is the test 8b's session-
    count-only assertion could not make — it cannot see the burst/rest/trickle
    pathology the rolling-window pacer produced; this asserts the shape."""
    # C=3 s ⇒ a burst of a few ~1 s sessions; R=0.5 ⇒ rests of a few ticks while
    # tokens refill — several cycles over a long off-span.
    config = _dream_cycle_config(
        tmp_path, metabolic_capacity_seconds=2.5, metabolic_refill_rate=0.1
    )
    total_env_steps = 12
    desktop = ScriptedDesktop([True] * 6 + [False] * 60 + [True])

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
    session_rows = [
        r for r in _read_jsonl_records(telem / "dream_session.jsonl")
        if r["ended_at_env_step"] is not None  # end records (one per session)
    ]
    transitions = [
        (e["payload"]["from_state"], e["payload"]["to_state"], e["payload"]["trigger"])
        for e in _read_jsonl_records(telem / "world_event.jsonl")
        if e["event_type"] == "state_transition"
    ]
    # Re-entry rhythm: the dormant duration before each metabolic re-entry. Small
    # gaps (≈1 tick) are *within a burst* (the bucket still affords a session
    # back-to-back); large gaps are *rests* (the bucket depleted and refilled over
    # several dormant ticks — which emit no transitions, so rests are visible only
    # here, not in the transition log). The token bucket produces a clean rhythm:
    # an initial capacity-bounded burst, then regular full sessions separated by
    # refill rests — e.g. gaps [1000, 1000, 7000, 7000, …].
    reentry_gaps = [
        e["payload"]["wallclock_ms_in_prev_state"]
        for e in _read_jsonl_records(telem / "world_event.jsonl")
        if e["event_type"] == "state_transition"
        and e["payload"]["trigger"] == "metabolic_replenished"
    ]
    _REST_MS = 2500  # gap ≥ this ⇒ a rest (bucket refilled); below ⇒ within-burst

    # (1) NO TRICKLE — every session ran a FULL session (the rollout-count cap),
    # never a 1-rollout degenerate. Structural: the compute cap is gone (so no
    # mid-session compute truncation), and the full-session re-entry hysteresis
    # only fires with room for a real session. (The 8b rolling-window pacer
    # produced rollout_count==1 trickle sessions here.)
    assert len(session_rows) >= 4, f"expected several sessions, got {len(session_rows)}"
    assert all(r["rollout_count"] == 3 for r in session_rows), (
        f"trickle: not every session is full — {[r['rollout_count'] for r in session_rows]}"
    )
    assert all(r["end_trigger"] == "hard_cap_rollout_count" for r in session_rows)

    # (2) PERIODIC — ≥2 rests (the rhythm repeats: dream → rest → dream → rest),
    # not one front-loaded burst followed by a single long rest. Every re-entry is
    # the content-blind metabolic edge from dormant.
    rests = [g for g in reentry_gaps if g >= _REST_MS]
    assert len(rests) >= 2, f"not periodic (rests={rests}) gaps={reentry_gaps}"
    reentries = [t for t in transitions if t[2] == "metabolic_replenished"]
    assert all((f, to) == ("dormant", "dreaming") for f, to, _ in reentries)

    # (3) BOUNDED BURST — the longest run of back-to-back (within-burst) sessions
    # is bounded by capacity, NOT front-loading all sessions at the absence start
    # (the 8b pathology: ~100 sessions back-to-back). Longest run of consecutive
    # small (within-burst) gaps:
    longest_burst_run = _longest_small_run(reentry_gaps, _REST_MS)
    assert longest_burst_run <= 5, (
        f"unbounded front-loaded burst (run={longest_burst_run}) gaps={reentry_gaps}"
    )

    # desktop_on preempts to waking, and the waking budget completed.
    assert transitions[-1][1] == "waking" and transitions[-1][2] == "desktop_on"
    assert len(_read_parquet_records(telem / "agent_step")) == total_env_steps


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


def test_offline_checkpoint_crash_and_resume(tmp_path: Path) -> None:
    """Phase 8c load-bearing — crash-safe long absences. Drive an offline period
    with offline checkpoints, simulate a crash (discard the live process), resume
    from the last offline checkpoint, and assert the FULL state is restored
    (bucket, controller state, weights identity), the loss is bounded, the
    checkpoint landed at a between-sessions boundary, and the metabolic rhythm
    continues frozen across the gap (no spurious refill credit for the pause)."""
    config = dataclasses.replace(
        _dream_cycle_config(
            tmp_path, metabolic_capacity_seconds=3.0, metabolic_refill_rate=0.1
        ),
        # Small offline-checkpoint interval so checkpoints fire during the test.
        offline_checkpoint_interval_ms=5000,
        # No waking checkpoints in this short run (so every checkpoint is offline).
        checkpoint_every_n_env_steps=10_000,
    )
    total_env_steps = 12
    # Waking (populate buffer), a long off-span (dreaming + offline checkpoints),
    # then on so the run terminates.
    desktop = ScriptedDesktop([True] * 6 + [False] * 40 + [True])

    with _transport_pair() as (client, _server, env_server, _thread):
        runner_a = Runner(
            config,
            client,
            env_server=env_server,
            host_signal_source=desktop,
            host_clock_ms=_stepping_clock(),
        )
        try:
            runner_a.run(total_env_steps=total_env_steps)
        finally:
            runner_a.close()

    # The last offline checkpoint = the latest checkpoint dir carrying an
    # offline_state.json (waking checkpoints don't write one).
    ckpts_dir = config.checkpoints_dir
    offline_ckpts = sorted(
        d.name
        for d in ckpts_dir.iterdir()
        if d.is_dir() and (d / "offline_state.json").is_file()
    )
    assert len(offline_ckpts) >= 1, "no offline checkpoint was written"
    last_offline = offline_ckpts[-1]
    saved = json.loads((ckpts_dir / last_offline / "offline_state.json").read_text())
    saved_tokens = saved["metabolic_budget"]["tokens"]
    capacity = config.dream_envelope.metabolic_capacity_seconds

    # Boundary yield: the offline checkpoint landed at a between-sessions boundary
    # (dormant), never mid-session — the CheckpointWindowCap held.
    assert saved["controller_state"] == "dormant"
    # The bucket was depleted by dreaming (so the frozen-pause assertion is
    # meaningful — a credited pause would have refilled it to capacity).
    assert saved_tokens < capacity

    # --- "Crash": runner_a is gone; the on-disk checkpoint is behind the live
    # state. Resume runner_b from the last offline checkpoint. ---
    with _transport_pair() as (client2, _s2, env_server2, _t2):
        runner_b = Runner(
            config,
            client2,
            env_server=env_server2,
            host_signal_source=AlwaysAwakeDesktop(),
            host_clock_ms=_stepping_clock(),
        )
        try:
            runner_b.load_checkpoint(last_offline)
            runner_b._build_state_controller()  # applies the offline-state restore

            # Controller state restored (the resuming mind is the same mind).
            assert runner_b._controller is not None
            assert runner_b._controller.current_state == "dormant"

            # Weights identity: runner_b's world model == the checkpoint's weights.
            disk = load_file(str(ckpts_dir / last_offline / "weights.safetensors"))
            live_wm = runner_b._world_model.state_dict()
            sample_key = next(iter(live_wm))
            assert torch.equal(
                live_wm[sample_key].cpu(), disk[f"world_model.{sample_key}"].cpu()
            )

            # Bucket restored FROZEN across the pause: the first observation after
            # resume returns exactly the saved tokens (the pause — checkpoint→resume
            # — credited NO refill), not capacity.
            assert runner_b._metabolic_budget is not None
            resumed = runner_b._metabolic_budget.current_tokens(now_ms=1_000_000)
            assert resumed == pytest.approx(saved_tokens)
            assert resumed < capacity  # a credited pause would have filled it

            # The rhythm continues from the restored bucket: refill resumes from
            # the resume time forward (here a long post-resume span fills it).
            later = runner_b._metabolic_budget.current_tokens(now_ms=1_000_000 + 10**8)
            assert later > resumed
        finally:
            runner_b.close()
