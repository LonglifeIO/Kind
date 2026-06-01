"""Probe 3 Phase 4 — state-machine tests (plan §4 Phase 4 row).

The load-bearing test is :func:`test_host_signals_carry_no_io_state` /
:func:`test_tick_params_carry_no_io_state` — the synthesis §5 exogenous-trigger
commitment enforced at the type/signature level, the third structural guard of
Probe 3 (after Phase 3's non-degeneracy and empirical gate). It catches an
Io-state-derived trigger at import/collection time rather than at runtime.

The remaining tests cover the transition table, dreaming-vs-dormant telemetry
distinguishability, ``DormantHeartbeat`` cadence, ``state_transition`` payload
round-trips, and one heavier dream-driving integration test that wires the real
:func:`kind.training.dream.run_dream_session` against a real Probe 1.5
checkpoint (the wiring the Phase 3 replica stood in for). The fast
transition-logic tests inject a stub driver so they need no checkpoint.
"""

from __future__ import annotations

import inspect
import typing
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

from kind.agents.views import PolicyView
from kind.observer.dream_session import DreamSessionMeta, DreamSessionSink
from kind.observer.schemas import (
    PROBE_3_TELEMETRY_SCHEMA_VERSION,
    AgentStep,
    DreamRollout,
    WorldEvent,
)
from kind.observer.sinks import ParquetSink, read_parquet_dir
from kind.training.dream import DreamRolloutConfig
from kind.training.dream_seed import SeedSelectionConfig
from kind.training.state_machine import (
    DreamEnvelopeConfig,
    DreamSessionContext,
    DreamSessionOutcome,
    DormantHeartbeat,
    HostSignals,
    ProtectionVerdict,
    RunDreamSessionDriver,
    StateController,
    StubRolloutCountProtection,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_CKPT = (
    REPO_ROOT
    / "runs"
    / "probe1_5_phase7_5-20260507-101800"
    / "checkpoints"
    / "ckpt-000001"
)


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------


class _Collector:
    """A ``WorldEventEmit`` that records every written event."""

    def __init__(self) -> None:
        self.events: list[WorldEvent] = []

    def write(self, record: WorldEvent) -> None:
        self.events.append(record)


class _StubDriver:
    """A ``DreamDriver`` that runs no rollouts; records that it was invoked.

    Stands in for the real ``run_dream_session`` so the fast transition tests
    need no checkpoint. ``end_trigger`` is the cap it reports back, which the
    controller uses for the subsequent ``dreaming → dormant`` transition.
    """

    def __init__(self, end_trigger: str = "hard_cap_rollout_count") -> None:
        self.sessions: list[str] = []
        self._end_trigger = end_trigger

    def run_session(
        self,
        *,
        dream_session_id: str,
        started_at_env_step: int,
        started_at_wallclock_ms: int,
        protection: Any,
        envelope: Any,
    ) -> DreamSessionOutcome:
        self.sessions.append(dream_session_id)
        return DreamSessionOutcome(
            end_trigger=self._end_trigger,  # type: ignore[arg-type]
            rollout_count=3,
        )


def _fixed_id(value: str) -> Callable[[], str]:
    return lambda: value


def _controller(
    *,
    driver: _StubDriver | None = None,
    emit: _Collector | None = None,
    envelope: DreamEnvelopeConfig | None = None,
    session_id: str = "s1",
    initial_state: str = "waking",
) -> StateController:
    return StateController(
        envelope if envelope is not None else DreamEnvelopeConfig(),
        SeedSelectionConfig(),
        dream_driver=driver,
        world_event_emit=emit,
        dream_session_id_factory=_fixed_id(session_id),
        initial_state=initial_state,  # type: ignore[arg-type]
    )


_OFF = HostSignals(desktop_alive=False)
_ON = HostSignals(desktop_alive=True)


# --------------------------------------------------------------------------
# Load-bearing: exogenous-trigger type-level enforcement (synthesis §5)
# --------------------------------------------------------------------------

# The host-observable primitive types a trigger field/parameter may carry. The
# allowlist is intentionally tight: widening it (e.g. to admit a host-side
# float like CPU temperature) is a deliberate, reviewed act, because the same
# tightness is what catches an Io-state-derived field. A float, a Tensor, a
# PolicyView, an AgentStep — anything sourced from Io's data plane — is not in
# this set and trips the guard.
_HOST_OBSERVABLE_TYPES = {bool, int}

# Explicit Io data-plane types that must never appear on the trigger surface.
_FORBIDDEN_IO_TYPES = {torch.Tensor, PolicyView, AgentStep, DreamRollout}


def test_host_signals_carry_no_io_state() -> None:
    """Every ``HostSignals`` field is a host-observable primitive — load-bearing.

    No field derives from ``agent_step``, Io's latents/policy, or dream content.
    Adding an Io-state-derived field (e.g. ``latent_norm: float`` or
    ``last_disagreement: torch.Tensor``) fails this assertion at collection
    time — the structural guard the plan specified, the trigger-side face of the
    Watts default-to-no on self-access.
    """
    hints = typing.get_type_hints(HostSignals)
    assert hints, "HostSignals must declare at least one host-observable field"
    for name, annotation in hints.items():
        assert annotation in _HOST_OBSERVABLE_TYPES, (
            f"HostSignals.{name} is typed {annotation!r}, which is not a "
            f"host-observable primitive {_HOST_OBSERVABLE_TYPES}. A trigger "
            f"field must be observable without reading Io (synthesis §5)."
        )
        assert annotation not in _FORBIDDEN_IO_TYPES


def test_tick_params_carry_no_io_state() -> None:
    """``StateController.tick``'s only state-bearing input is ``HostSignals``.

    The other parameters are host-observable counters (``env_step``,
    ``wallclock_ms``). No parameter is typed as an Io data-plane type, so an
    Io-state-derived trigger cannot be wired through ``tick`` without being
    caught here.
    """
    sig = inspect.signature(StateController.tick)
    params = [p for p in sig.parameters if p != "self"]
    assert params == ["host_signals", "env_step", "wallclock_ms"]

    hints = typing.get_type_hints(StateController.tick)
    assert hints["host_signals"] is HostSignals
    assert hints["env_step"] is int
    assert hints["wallclock_ms"] is int
    for name, annotation in hints.items():
        assert annotation not in _FORBIDDEN_IO_TYPES, (
            f"tick parameter {name} is typed {annotation!r}, an Io data-plane "
            f"type — that would make whether Io dreams depend on Io's state."
        )


# --------------------------------------------------------------------------
# Transition table coverage
# --------------------------------------------------------------------------


def test_waking_stays_waking_when_desktop_alive() -> None:
    c = _controller(driver=_StubDriver())
    assert c.tick(_ON, env_step=1, wallclock_ms=100) is None
    assert c.current_state == "waking"


def test_waking_to_dreaming_on_desktop_off() -> None:
    driver = _StubDriver()
    c = _controller(driver=driver, session_id="s1")
    t = c.tick(_OFF, env_step=10, wallclock_ms=1000)
    assert t is not None
    assert (t.from_state, t.to_state, t.trigger) == ("waking", "dreaming", "desktop_off")
    assert t.dream_session_id == "s1"
    assert c.current_state == "dreaming"
    assert driver.sessions == ["s1"]  # the session was driven on entry


def test_dreaming_to_dormant_on_cap() -> None:
    driver = _StubDriver(end_trigger="hard_cap_rollout_count")
    c = _controller(driver=driver)
    c.tick(_OFF, 10, 1000)  # -> dreaming; driver reports the cap
    t = c.tick(_OFF, 11, 2000)  # -> dormant on the pending cap
    assert t is not None
    assert (t.from_state, t.to_state, t.trigger) == (
        "dreaming",
        "dormant",
        "hard_cap_rollout_count",
    )
    assert c.current_state == "dormant"


def test_dreaming_to_waking_on_desktop_on_takes_priority_over_cap() -> None:
    driver = _StubDriver(end_trigger="hard_cap_rollout_count")
    c = _controller(driver=driver)
    c.tick(_OFF, 10, 1000)  # -> dreaming (pending cap set)
    t = c.tick(_ON, 11, 1500)  # desktop back before settling -> waking
    assert t is not None
    assert (t.from_state, t.to_state, t.trigger) == ("dreaming", "waking", "desktop_on")
    assert c.current_state == "waking"


def test_dormant_to_waking_on_desktop_on() -> None:
    driver = _StubDriver()
    c = _controller(driver=driver)
    c.tick(_OFF, 10, 1000)  # dreaming
    c.tick(_OFF, 11, 2000)  # dormant
    t = c.tick(_ON, 12, 3000)
    assert t is not None
    assert (t.from_state, t.to_state, t.trigger) == ("dormant", "waking", "desktop_on")
    assert c.current_state == "waking"


def test_tick_never_produces_paused() -> None:
    """Paused is supervisor-mediated, never reachable from ``tick`` (§2.4 cmt 3)."""
    driver = _StubDriver()
    c = _controller(driver=driver)
    signals = [_OFF, _OFF, _ON, _OFF, _OFF, _ON, _OFF]
    for i, hs in enumerate(signals):
        t = c.tick(hs, env_step=i, wallclock_ms=i * 1000)
        if t is not None:
            assert t.to_state != "paused"
    assert c.current_state != "paused"


# --------------------------------------------------------------------------
# Supervisor-mediated paused transitions (outside tick)
# --------------------------------------------------------------------------


def test_on_shutdown_records_paused() -> None:
    emit = _Collector()
    c = _controller(driver=_StubDriver(), emit=emit)
    t = c.on_shutdown(env_step=5, wallclock_ms=500)
    assert (t.from_state, t.to_state, t.trigger) == ("waking", "paused", "mac_off")
    assert c.current_state == "paused"
    assert emit.events[-1].payload["to_state"] == "paused"


def test_on_startup_resumes_waking_when_desktop_on() -> None:
    c = _controller(driver=_StubDriver(), initial_state="paused")
    t = c.on_startup(_ON, env_step=0, wallclock_ms=0)
    assert (t.from_state, t.to_state, t.trigger) == ("paused", "waking", "mac_on")
    assert c.current_state == "waking"


def test_on_startup_resumes_dormant_when_desktop_off() -> None:
    c = _controller(driver=_StubDriver(), initial_state="paused")
    t = c.on_startup(_OFF, env_step=0, wallclock_ms=0)
    assert (t.from_state, t.to_state, t.trigger) == ("paused", "dormant", "mac_on")
    assert c.current_state == "dormant"


# --------------------------------------------------------------------------
# Dreaming is operationally distinguishable from dormant (synthesis §10)
# --------------------------------------------------------------------------


def test_dreaming_and_dormant_have_disjoint_telemetry() -> None:
    emit = _Collector()
    driver = _StubDriver()
    env = DreamEnvelopeConfig(dormant_heartbeat_interval_ms=100)
    c = _controller(driver=driver, emit=emit, envelope=env, session_id="s1")

    c.tick(_OFF, 10, 1000)  # -> dreaming: the driver runs (dream work)
    assert driver.sessions == ["s1"]
    assert {e.event_type for e in emit.events} == {"state_transition"}

    c.tick(_OFF, 11, 2000)  # -> dormant
    sessions_at_dormant = len(driver.sessions)

    c.tick(_OFF, 12, 2500)  # dormant: only heartbeats fire
    heartbeats = [e for e in emit.events if e.event_type == "dormant_heartbeat"]
    assert len(heartbeats) >= 1
    assert all(e.payload["mac_alive"] is True for e in heartbeats)
    # Dormant drives no dream work — the telemetry signatures are disjoint.
    assert len(driver.sessions) == sessions_at_dormant


# --------------------------------------------------------------------------
# DormantHeartbeat cadence (injected clock, no real waiting)
# --------------------------------------------------------------------------


def test_dormant_heartbeat_cadence_with_injected_clock() -> None:
    emit = _Collector()
    now = {"ms": 0}
    hb = DormantHeartbeat(
        interval_ms=100,
        emit=emit,
        run_id="r",
        checkpoint_id=None,
        clock=lambda: now["ms"],
    )
    hb.start(0)
    now["ms"] = 1000  # one second elapsed
    fired = hb.poll()
    assert fired == 10
    assert len(emit.events) == 10
    assert all(e.event_type == "dormant_heartbeat" for e in emit.events)
    assert all(e.payload["mac_alive"] is True for e in emit.events)
    elapsed = [e.payload["dormant_wallclock_ms_elapsed"] for e in emit.events]
    assert elapsed == [100 * (i + 1) for i in range(10)]


def test_dormant_heartbeat_emits_incrementally_across_polls() -> None:
    emit = _Collector()
    hb = DormantHeartbeat(
        interval_ms=100, emit=emit, run_id="r", checkpoint_id=None
    )
    hb.start(0)
    assert hb.poll(now_ms=250) == 2  # 2 full intervals
    assert hb.poll(now_ms=550) == 3  # 3 more (intervals 3,4,5)
    assert len(emit.events) == 5


# --------------------------------------------------------------------------
# state_transition payload round-trips
# --------------------------------------------------------------------------


def test_state_transition_payload_round_trips() -> None:
    emit = _Collector()
    driver = _StubDriver()
    c = _controller(driver=driver, emit=emit, session_id="sX")

    c.tick(_OFF, env_step=42, wallclock_ms=5000)  # waking -> dreaming
    ev = next(e for e in emit.events if e.event_type == "state_transition")
    assert ev.source == "environment"
    assert ev.schema_version == PROBE_3_TELEMETRY_SCHEMA_VERSION
    assert ev.payload["from_state"] == "waking"
    assert ev.payload["to_state"] == "dreaming"
    assert ev.payload["trigger"] == "desktop_off"
    assert ev.payload["dream_session_id"] == "sX"
    assert ev.payload["env_step_at_transition"] == 42
    # State entered at 0 (constructor), left at 5000.
    assert ev.payload["wallclock_ms_in_prev_state"] == 5000

    # Pydantic round-trip preserves the record exactly.
    again = WorldEvent.model_validate_json(ev.model_dump_json())
    assert again == ev


def test_wallclock_in_prev_state_tracks_dwell() -> None:
    emit = _Collector()
    driver = _StubDriver()
    c = _controller(driver=driver, emit=emit)
    c.tick(_OFF, 10, 5000)  # waking(entered@0) -> dreaming
    c.tick(_OFF, 11, 6000)  # dreaming(entered@5000) -> dormant
    dormant_ev = next(
        e
        for e in emit.events
        if e.event_type == "state_transition" and e.payload["to_state"] == "dormant"
    )
    assert dormant_ev.payload["wallclock_ms_in_prev_state"] == 1000


# --------------------------------------------------------------------------
# Protection hook surface (Phase 4 builds the hook; Phase 6 fills the policy)
# --------------------------------------------------------------------------


def test_stub_protection_caps_on_rollout_count_only() -> None:
    policy = StubRolloutCountProtection()
    env = DreamEnvelopeConfig(hard_cap_rollout_count=3)

    def ctx(k: int) -> DreamSessionContext:
        return DreamSessionContext(
            envelope=env,
            rollouts_completed=k,
            session_wallclock_ms_elapsed=10**9,  # huge: a wallclock policy would fire
            checkpoint_in_progress=True,  # a checkpoint policy would fire
        )

    # Content-blind, rollout-count-only: ignores wallclock and checkpoint (those
    # are Phase 6 policies), fires exactly at the rollout cap.
    assert policy.should_stop(ctx(0)) is None
    assert policy.should_stop(ctx(2)) is None
    verdict = policy.should_stop(ctx(3))
    assert verdict == ProtectionVerdict(trigger="hard_cap_rollout_count")


# --------------------------------------------------------------------------
# Dream-driving integration test (real run_dream_session, real checkpoint)
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not (REAL_CKPT / "weights.safetensors").exists(),
    reason="Probe 1.5 checkpoint not present",
)
def test_dream_driving_emits_real_dream_rollout(tmp_path: Path) -> None:
    """The dreaming handler drives a real ``run_dream_session`` (the wiring the
    Phase 3 replica stood in for) and a valid ``DreamRollout`` is emitted
    through the (now-fixed) sink. Then the cap exit lands in dormant — a rest,
    not an error (dormant ≠ failure)."""
    from scripts.smoke_probe3_visibility import fill_buffer, load_checkpoint

    device = torch.device("cpu")
    loaded = load_checkpoint(REAL_CKPT, device)
    buffer, _ = fill_buffer(
        loaded,
        n_steps=400,
        capacity=401,
        env_seed=0,
        device=device,
        waking_state_stride=50,
    )
    seed_cfg = SeedSelectionConfig(
        replay_min_segment_age_steps=50, replay_warmup_length=8
    )
    rollout_cfg = DreamRolloutConfig(horizon=4, re_seed_every_n_steps=2)

    rollout_dir = tmp_path / "dream_rollout"
    telemetry_dir = tmp_path / "telemetry"
    rollout_sink = ParquetSink(rollout_dir, DreamRollout)
    session_sink = DreamSessionSink(telemetry_dir)
    rng = torch.Generator()
    rng.manual_seed(0)

    driver = RunDreamSessionDriver(
        world_model=loaded.world_model,
        actor=None,
        ensemble=loaded.ensemble,
        replay_buffer=buffer,
        seed_selection_config=seed_cfg,
        rollout_config=rollout_cfg,
        run_id="it",
        checkpoint_id=loaded.checkpoint_id,
        rng=rng,
        device=device,
        dream_rollout_sink=rollout_sink,
        dream_session_sink=session_sink,
    )

    emit = _Collector()
    env = DreamEnvelopeConfig(hard_cap_rollout_count=2, dormant_heartbeat_interval_ms=100)
    c = StateController(
        env,
        seed_cfg,
        dream_driver=driver,
        world_event_emit=emit,
        run_id="it",
        checkpoint_id=loaded.checkpoint_id,
        dream_session_id_factory=_fixed_id("s-int"),
    )

    t1 = c.tick(_OFF, env_step=100, wallclock_ms=1000)  # -> dreaming, run session
    assert t1 is not None and t1.to_state == "dreaming"

    rollout_sink.close()
    records = read_parquet_dir(rollout_dir, DreamRollout)
    assert len(records) == 2  # hard_cap_rollout_count=2
    r = records[0]
    assert r.schema_version == PROBE_3_TELEMETRY_SCHEMA_VERSION
    assert r.dream_session_id == "s-int"
    assert r.seed_kind == "replay"
    # The sink fix is what lets the 0.3.0 dict field serialize without raising.
    assert r.sampling_parameters is not None
    assert r.sequence_ensemble_disagreement_variance is not None
    assert len(r.sequence_ensemble_disagreement_variance) == 4

    # DreamSessionMeta was double-written (start + end), same session id.
    session_sink.close()
    meta_lines = (telemetry_dir / "dream_session.jsonl").read_text().splitlines()
    metas = [DreamSessionMeta.model_validate_json(line) for line in meta_lines]
    assert len(metas) == 2
    assert {m.dream_session_id for m in metas} == {"s-int"}
    assert metas[-1].end_trigger == "hard_cap_rollout_count"

    # Cap exit -> dormant (a rest, not an error path): no exception, heartbeats.
    t2 = c.tick(_OFF, env_step=101, wallclock_ms=2000)
    assert t2 is not None
    assert t2.to_state == "dormant"
    assert t2.trigger == "hard_cap_rollout_count"
