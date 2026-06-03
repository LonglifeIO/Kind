"""Phase 4 state machine — waking / dreaming / dormant / paused.

This module is where dreaming becomes a *first-class state* rather than a
waking-cadence side effect. The ``StateController`` governs Io's transitions
between the four states of the design notes' four-state model, drives a dream
session while dreaming, and structurally enforces the two commitments the
synthesis settled (``docs/decisions/synthesis_probe3_dream_foundational_2026-05-27.md``):

  1. **The exogenous trigger (synthesis §5).** Whether Io dreams is decided by
     host-side facts Io cannot observe — desktop on/off — carried on
     :class:`HostSignals`. *No field of ``HostSignals`` and no parameter of
     ``StateController.tick`` derives from ``agent_step``, Io's latents/policy,
     dream content, or any data-plane quantity.* This purity is load-bearing:
     wiring an Io-state-derived trigger ("dream when latent-norm is high") must
     fail to typecheck / be caught by the type-level test at import time, the
     way ``gradient_policy: Literal["none"]`` made no-gradient-flow unmissable.
     It is the same self-opacity commitment as the Watts default-to-no, seen
     from the trigger side: Io does not observe its own state, and the state is
     not in Io's observation/PolicyView, so the state machine never reads Io to
     decide what Io does.

     **Refined reading (Phase 8b, Fork B = B2; decision-doc Fork B amendment).**
     "Io does not decide whether to dream from its own state" is read as
     *"whether-to-dream is gated by nothing Io-state-derived"* — not as
     "whether-to-dream is gated by HostSignals only." The B2 dormant→dreaming
     re-entry edge is gated by a **content-blind compute ledger** (durations and
     timestamps; no Io latents, policy, or dream content), which is not Io's
     state but a metabolic/resource constraint — the same *category* as the
     desktop being off. The commitment's load-bearing purpose (forbid an
     installed self-continuation drive — Io's own state deciding to keep itself
     dreaming) is preserved; the trigger surface extends from HostSignals-only to
     {HostSignals, content-blind runtime pacer}. The structural guarantee is made
     unrepresentable-to-violate by the re-entry input type (:class:`MetabolicState`
     in ``kind.training.protection``, all content-blind primitives) and its
     type-level test, exactly as ``HostSignals`` is guarded here.

  2. **Dormant ≠ failure (synthesis §10; plan §2.5).** A dream session that
     hits a content-blind cap transitions to *dormant* — a legitimate rest, the
     capacity-over-exercise stance applied to dreaming — not to an error path.
     Dormant emits ``dormant_heartbeat`` world events as the positive
     "resting, not paused" signal, and is operationally distinguishable from
     dreaming (which drives ``DreamRollout`` records) by construction.

**Paused is supervisor-mediated, not tick-reachable (plan §2.4 commitment 3).**
``tick()`` runs only while the process is alive, so it can never observe the Mac
being off. ``paused`` is therefore recorded by the supervisor on shutdown
(:meth:`StateController.on_shutdown`) and resumed on startup
(:meth:`StateController.on_startup`) — never inferred inside ``tick``. Modelling
paused as a tick-reachable state is the error plan review caught; the structure
here makes it unrepresentable.

**The Phase 4 / Phase 6 boundary (patched plan).** Phase 4 builds the state
machine, the transition semantics, and the *protection hooks* — the
exit-condition interface (:class:`ProtectionPolicy`) where the four
content-blind caps plug in, plus the trigger enum they emit. The cap *policies*
themselves (the wallclock cap, the rollout-count cap, the checkpoint-window
forced transition, and the rolling one-hour compute-budget ledger) are Phase 6.
A trivial stub policy (:class:`StubRolloutCountProtection`) ships here so the
hook surface is exercised; Phase 6 fills it with the real content-blind logic.

**This module does not touch the runner.** The live state-machine ↔ runner
wiring — and whether the waking-cadence handshake coexists with or yields to
state-machine-driven dreaming — is Phase 8. Phase 4 is a standalone, in-isolation
unit-testable consumer of :func:`kind.training.dream.run_dream_session` and the
Phase 0 ``world_event`` payload conventions.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import torch

from kind.agents.actor import Actor
from kind.agents.ensemble import LatentDisagreementEnsemble
from kind.agents.world_model import WorldModel
from kind.observer.dream_session import DreamSessionSink
from kind.observer.schemas import PROBE_3_TELEMETRY_SCHEMA_VERSION, WorldEvent
from kind.training.dream import (
    DreamRolloutConfig,
    DreamRolloutSink,
    run_dream_session,
)
from kind.training.dream_seed import SeedSelectionConfig
from kind.training.replay import SequenceReplayBuffer

__all__ = [
    "State",
    "Trigger",
    "CapTrigger",
    "HostSignals",
    "DesktopSignalSource",
    "AlwaysAwakeDesktop",
    "ScriptedDesktop",
    "DesktopWatcher",
    "StateTransition",
    "DreamEnvelopeConfig",
    "DreamSessionContext",
    "ProtectionVerdict",
    "ProtectionPolicy",
    "StubRolloutCountProtection",
    "DreamSessionOutcome",
    "DreamDriver",
    "RunDreamSessionDriver",
    "WorldEventEmit",
    "DormantHeartbeat",
    "StateController",
]

State = Literal["waking", "dreaming", "dormant", "paused"]

# The full trigger vocabulary the state_transition payload may carry (plan §2.4
# / the Phase 0 schema convention). Exogenous host signals plus the four
# content-blind caps plus the supervisor mac on/off edges.
Trigger = Literal[
    "desktop_off",
    "desktop_on",
    "mac_off",
    "mac_on",
    "hard_cap_wallclock",
    "hard_cap_rollout_count",
    "checkpoint_window",
    "compute_budget",
    # Phase 8b — the dormant→dreaming re-entry edge (Fork B = B2). Not a cap
    # (it is not in CapTrigger): it is the *complement* of the compute-budget
    # cap, fired by the content-blind metabolic pacer while the desktop is off.
    "metabolic_replenished",
]

# The subset of triggers a content-blind protection cap may raise (the Phase 4
# hook surface's output; Phase 6 fills the policies that decide which fires).
CapTrigger = Literal[
    "hard_cap_wallclock",
    "hard_cap_rollout_count",
    "checkpoint_window",
    "compute_budget",
]


@dataclass(frozen=True)
class HostSignals:
    """The exogenous trigger type — host-side facts Io cannot observe.

    Every field must be host-observable *without reading Io*: nothing here
    derives from ``agent_step``, Io's latents/policy, dream content, or any
    data-plane quantity (synthesis §5; the no-internal-trigger commitment). The
    type-level test in ``tests/test_state_machine.py`` is the guard — it asserts
    at the signature level that no Io-state-derived field can be added without
    being caught at import time.

    ``mac_alive`` is deliberately *absent*: ``tick`` only runs while the Mac is
    on, so a mac-off signal is unrepresentable here. Mac on/off is handled by
    the supervisor entry points (:meth:`StateController.on_shutdown` /
    :meth:`StateController.on_startup`), keeping paused un-tick-reachable.
    """

    desktop_alive: bool = True
    checkpoint_in_progress: bool = False


class DesktopSignalSource(Protocol):
    """Produces the exogenous ``desktop_alive`` bit (plan §2.4 ``poll() -> bool``).

    Deliberately produces *only* ``desktop_alive`` — a host-observable fact Io
    cannot see. ``checkpoint_in_progress`` is *not* sourced here: it is the
    runner's own checkpoint-commit state, assembled into :class:`HostSignals` by
    the supervisor. Keeping this source desktop-only preserves the exogenous
    trigger commitment at the type level (nothing here is Io-derived) and the
    Phase 4 ``HostSignals`` type-test.
    """

    def desktop_alive(self) -> bool: ...


class AlwaysAwakeDesktop:
    """The loopback / smoke default: the desktop is always on.

    With this source the supervisor never leaves the waking state, so the
    pre-Phase-8a integration smokes run pure waking and stay behaviorally
    unchanged (the dream-state machinery only activates on a ``desktop_off``
    edge, which this source never produces).
    """

    def desktop_alive(self) -> bool:
        return True


class ScriptedDesktop:
    """A test source returning a programmed sequence of ``desktop_alive`` bits.

    Each :meth:`desktop_alive` call advances through ``sequence`` and *sticks on
    the last value* once exhausted — so a sequence ending in ``True`` guarantees
    the supervisor eventually resumes waking (the smoke drives desktop
    on→off→on by ending on ``True``).
    """

    def __init__(self, sequence: Sequence[bool]) -> None:
        if not sequence:
            raise ValueError("ScriptedDesktop requires a non-empty sequence")
        self._sequence = list(sequence)
        self._i = 0

    def desktop_alive(self) -> bool:
        value = self._sequence[self._i]
        if self._i < len(self._sequence) - 1:
            self._i += 1
        return value


class DesktopWatcher:
    """Plan §2.4 sentinel-file poll: ``desktop_alive`` iff the sentinel exists.

    The desktop writes/deletes a sentinel file (e.g. on a shared/synced path);
    its presence is the exogenous on/off signal. A pure filesystem read —
    nothing Io-derived. Probe 4 may replace this with a network heartbeat.
    """

    def __init__(self, sentinel_path: Path) -> None:
        self._sentinel_path = sentinel_path

    def desktop_alive(self) -> bool:
        return self._sentinel_path.exists()


@dataclass(frozen=True)
class StateTransition:
    """One state-machine edge (plan §2.4).

    ``dream_session_id`` is set on edges that bound a dream session (the
    ``waking → dreaming`` start carries the new session's id; the
    ``dreaming → *`` exit carries the same id). ``wallclock_ms_at_transition``
    is the absolute tick wallclock; the ``world_event`` payload also records
    ``wallclock_ms_in_prev_state`` (time spent in the state being left).
    """

    from_state: State
    to_state: State
    dream_session_id: str | None
    trigger: Trigger
    env_step_at_transition: int
    wallclock_ms_at_transition: int


@dataclass(frozen=True)
class DreamEnvelopeConfig:
    """The Probe-4-perturbable *when* / *how-long* of dreaming (plan §2.4 / §7).

    Phase 4 consumes these as the parameters the protection hooks read; Phase 6
    implements the content-blind policies that enforce them. All are
    content-blind quantities (wallclock, rollout counts, a checkpoint boolean, a
    compute ledger budget) — none reads ``DreamRollout`` content.
    """

    hard_cap_wallclock_ms: int = 30 * 60 * 1000  # 30 minutes per dream session
    hard_cap_rollout_count: int = 50  # max DreamRollouts per session
    checkpoint_window_force_dormant: bool = True  # checkpoint boundary → dormant
    dormant_heartbeat_interval_ms: int = 60_000  # one heartbeat per minute
    # Phase 8b (Fork B = B2): this is no longer only a Phase-6 *protection cap* —
    # it is now the dream/rest **duty cycle** during an absence (the metabolic
    # pacer). 600 s/hour ⇒ ≤10 min dreaming / ≥50 min rest per rolling hour ⇒ a
    # ~17% wall-clock duty cycle, the rest-majority lean closer to the charter's
    # capacity-over-exercise / dormant-≠-failure stance than the Phase-6
    # protection default's 50% (1800 s). **This is the builder's knob — the
    # rhythm parameter that sets how much Io dreams while the desktop is off — to
    # tune by observation, NOT a settled value.**
    compute_budget_seconds_per_hour: float = 600.0  # ~17% dream duty cycle


@dataclass(frozen=True)
class DreamSessionContext:
    """Content-blind facts a :class:`ProtectionPolicy` may read to decide a stop.

    Deliberately carries *only* content-blind quantities (synthesis §6; plan
    §2.4 commitment 2): the envelope, the rollout count so far, the session's
    elapsed wallclock, the checkpoint barrier boolean, and (Phase 6) the two
    rolling-ledger-derived scalars below. It carries no ``DreamRollout``
    content, no mirror reading, and nothing derived from Io's state. Every
    field is a content-blind primitive (or the content-blind
    :class:`DreamEnvelopeConfig`); the type-level test in
    ``tests/test_runtime_protection.py`` enforces this and must trip if a
    content-bearing field is ever added.

    The two Phase 6 fields are populated by the composite
    :class:`~kind.training.protection.DreamProtectionPolicy` from its rolling
    compute ledger before it delegates to the per-cap sub-policies — so the
    sub-policies consume *only* this content-blind context and cannot reach the
    ledger (or anything else) directly. Both default to ``0.0`` so the Phase 4
    construction sites (``_plan_session_rollouts``, ``tick``) that don't carry a
    ledger still build a valid context; the composite overwrites them with the
    ledger's real values.
    """

    envelope: DreamEnvelopeConfig
    rollouts_completed: int
    session_wallclock_ms_elapsed: int
    checkpoint_in_progress: bool
    # Ledger-derived, content-blind. The wallclock cap reads the running
    # rollout-duration estimate (a projection input, not actual-elapsed); the
    # compute-budget cap reads the rolling-window compute already accrued.
    rollout_duration_estimate_ms: float = 0.0
    window_compute_seconds: float = 0.0


@dataclass(frozen=True)
class ProtectionVerdict:
    """A protection hook's answer: which content-blind cap ends the session."""

    trigger: CapTrigger


class ProtectionPolicy(Protocol):
    """The protection-hook interface — the dreaming-state exit + (8b) re-entry.

    Phase 4 defines this surface and where it plugs into the dreaming-state
    exit check; Phase 6 implements the policies (wallclock, rollout-count,
    checkpoint-window, and the rolling one-hour compute ledger) behind it.
    ``should_stop`` returns the cap that should end the session, or ``None`` to
    continue. Phase 8b adds two methods for the metabolic loop:
    ``record_session`` (the controller reports a completed session's real compute
    so the rolling-hour ledger accumulates actuals) and ``metabolic_reentry``
    (the content-blind dormant→dreaming re-entry decision — the complement of the
    compute-budget cap). The stub no-ops both (it never re-dreams); the Phase 6
    composite implements them against its ledger.
    """

    def should_stop(self, ctx: DreamSessionContext) -> ProtectionVerdict | None: ...

    def record_session(
        self, *, num_rollouts: int, session_wallclock_ms: int, now_ms: int
    ) -> None: ...

    def metabolic_reentry(
        self, envelope: DreamEnvelopeConfig, *, now_ms: int
    ) -> bool: ...


class StubRolloutCountProtection:
    """Phase 4 stub policy — caps on rollout count only; never re-dreams.

    A trivial content-blind policy so the hook surface is exercised end-to-end
    (deliverable 6: "a trivial/stub policy is fine for Phase 4 as long as the
    hook surface is what Phase 6 fills"). It reads only
    ``envelope.hard_cap_rollout_count`` and the rollout count so far — no
    content. Phase 6 replaces it with the real four-cap policy. The Phase 8b
    metabolic methods are no-ops: the stub keeps no ledger, so it records no
    compute and never grants metabolic re-entry (single-session-per-absence,
    the pre-8b behavior — used by the transition-coverage and type-level tests).
    """

    def should_stop(self, ctx: DreamSessionContext) -> ProtectionVerdict | None:
        if ctx.rollouts_completed >= ctx.envelope.hard_cap_rollout_count:
            return ProtectionVerdict(trigger="hard_cap_rollout_count")
        return None

    def record_session(
        self, *, num_rollouts: int, session_wallclock_ms: int, now_ms: int
    ) -> None:
        return None

    def metabolic_reentry(
        self, envelope: DreamEnvelopeConfig, *, now_ms: int
    ) -> bool:
        return False


@dataclass(frozen=True)
class DreamSessionOutcome:
    """What a :class:`DreamDriver` reports back after running one session."""

    end_trigger: CapTrigger
    rollout_count: int


class DreamDriver(Protocol):
    """Runs one dream session for the controller's dreaming span.

    The controller invokes this when it enters dreaming. The driver runs the
    session (in the real driver, via :func:`kind.training.dream.run_dream_session`),
    consulting the supplied :class:`ProtectionPolicy` to decide when the session
    stops, and reports the realized cap trigger so the controller can transition
    ``dreaming → dormant`` with a consistent trigger.
    """

    def run_session(
        self,
        *,
        dream_session_id: str,
        started_at_env_step: int,
        started_at_wallclock_ms: int,
        protection: ProtectionPolicy,
        envelope: DreamEnvelopeConfig,
        checkpoint_in_progress: bool = False,
    ) -> DreamSessionOutcome: ...


def _plan_session_rollouts(
    protection: ProtectionPolicy,
    envelope: DreamEnvelopeConfig,
    *,
    checkpoint_in_progress: bool = False,
) -> tuple[int, CapTrigger]:
    """Ask the protection hook, rollout by rollout, where the session stops.

    This *is* "the handler asks 'should this session stop?' through the hook
    interface" (deliverable 4) realized against the settled, one-shot
    :func:`run_dream_session` (which takes a fixed ``num_rollouts``): poll
    ``should_stop`` with an increasing ``rollouts_completed`` until it returns a
    verdict, giving the rollout count and the cap trigger. The
    ``hard_cap_rollout_count`` is a hard ceiling so a misbehaving Phase 6 policy
    cannot loop forever. Phase 6's richer policies (wallclock, compute ledger)
    may want :func:`run_dream_session` to become interruptible mid-loop; that is
    Phase 6/8's concern — the Phase 4 surface is the hook, not the loop shape.
    """
    ceiling = max(0, envelope.hard_cap_rollout_count)
    for k in range(ceiling + 1):
        verdict = protection.should_stop(
            DreamSessionContext(
                envelope=envelope,
                rollouts_completed=k,
                session_wallclock_ms_elapsed=0,
                checkpoint_in_progress=checkpoint_in_progress,
            )
        )
        if verdict is not None:
            return k, verdict.trigger
    return ceiling, "hard_cap_rollout_count"


class RunDreamSessionDriver:
    """The real :class:`DreamDriver` — wraps :func:`run_dream_session`.

    Phase 4 is the first place the real ``run_dream_session`` is wired (the
    Phase 3 smoke used a standalone replica). The driver determines the
    session's rollout count via the protection hook
    (:func:`_plan_session_rollouts`), runs the session once with that count,
    double-writes the ``DreamSessionMeta`` (handled inside ``run_dream_session``),
    and reports the realized cap trigger.
    """

    def __init__(
        self,
        *,
        world_model: WorldModel,
        actor: Actor | None,
        ensemble: LatentDisagreementEnsemble,
        replay_buffer: SequenceReplayBuffer,
        seed_selection_config: SeedSelectionConfig,
        rollout_config: DreamRolloutConfig,
        run_id: str,
        checkpoint_id: str | None,
        rng: torch.Generator,
        device: torch.device,
        dream_rollout_sink: DreamRolloutSink,
        dream_session_sink: DreamSessionSink,
        envelope_config_snapshot: dict[str, Any] | None = None,
        checkpoint_hash: str | None = None,
    ) -> None:
        self._world_model = world_model
        self._actor = actor
        self._ensemble = ensemble
        self._replay_buffer = replay_buffer
        self._seed_selection_config = seed_selection_config
        self._rollout_config = rollout_config
        self._run_id = run_id
        self._checkpoint_id = checkpoint_id
        self._rng = rng
        self._device = device
        self._dream_rollout_sink = dream_rollout_sink
        self._dream_session_sink = dream_session_sink
        self._envelope_config_snapshot = envelope_config_snapshot
        self._checkpoint_hash = checkpoint_hash

    def run_session(
        self,
        *,
        dream_session_id: str,
        started_at_env_step: int,
        started_at_wallclock_ms: int,
        protection: ProtectionPolicy,
        envelope: DreamEnvelopeConfig,
        checkpoint_in_progress: bool = False,
    ) -> DreamSessionOutcome:
        num_rollouts, end_trigger = _plan_session_rollouts(
            protection, envelope, checkpoint_in_progress=checkpoint_in_progress
        )
        run_dream_session(
            world_model=self._world_model,
            actor=self._actor,
            ensemble=self._ensemble,
            replay_buffer=self._replay_buffer,
            seed_selection_config=self._seed_selection_config,
            rollout_config=self._rollout_config,
            num_rollouts=num_rollouts,
            dream_session_id=dream_session_id,
            run_id=self._run_id,
            checkpoint_id=self._checkpoint_id,
            started_at_env_step=started_at_env_step,
            started_at_wallclock_ms=started_at_wallclock_ms,
            rng=self._rng,
            device=self._device,
            dream_rollout_sink=self._dream_rollout_sink,
            dream_session_sink=self._dream_session_sink,
            envelope_config_snapshot=self._envelope_config_snapshot,
            checkpoint_hash=self._checkpoint_hash,
            ended_at_env_step=started_at_env_step + num_rollouts,
            ended_at_wallclock_ms=started_at_wallclock_ms,
            end_trigger=end_trigger,
        )
        return DreamSessionOutcome(end_trigger=end_trigger, rollout_count=num_rollouts)


class WorldEventEmit(Protocol):
    """Structural type for the ``world_event`` writer (e.g. ``JsonlSink.write``)."""

    def write(self, record: WorldEvent) -> None: ...


def _default_clock_ms() -> int:
    return time.monotonic_ns() // 1_000_000


class DormantHeartbeat:
    """Emits ``dormant_heartbeat`` world events at the configured cadence.

    The positive "resting, not paused" signal (plan §2.5): dormant gaps with no
    records would be indistinguishable from ``paused``; heartbeats assert
    presence. The clock is injectable so cadence is testable without real
    waiting. :meth:`poll` emits every heartbeat that has come due since the last
    poll (so a coarse tick cadence still produces the right count).
    """

    def __init__(
        self,
        *,
        interval_ms: int,
        emit: WorldEventEmit | None,
        run_id: str,
        checkpoint_id: str | None,
        clock: Callable[[], int] | None = None,
    ) -> None:
        if interval_ms <= 0:
            raise ValueError(f"interval_ms must be positive, got {interval_ms}")
        self._interval_ms = interval_ms
        self._emit = emit
        self._run_id = run_id
        self._checkpoint_id = checkpoint_id
        # clock is a zero-arg callable returning int ms; default monotonic.
        self._clock: Callable[[], int] = clock if clock is not None else _default_clock_ms
        self._started_at_ms: int | None = None
        self._emitted = 0

    def start(self, started_at_ms: int) -> None:
        self._started_at_ms = started_at_ms
        self._emitted = 0

    def poll(self, now_ms: int | None = None) -> int:
        """Emit any heartbeats due since the last poll; return how many fired."""
        if self._started_at_ms is None:
            return 0
        now = now_ms if now_ms is not None else self._clock()
        elapsed = now - self._started_at_ms
        if elapsed < 0:
            return 0
        due = elapsed // self._interval_ms
        fired = 0
        while self._emitted < due:
            self._emitted += 1
            beat_elapsed_ms = self._emitted * self._interval_ms
            if self._emit is not None:
                self._emit.write(
                    WorldEvent(
                        schema_version=PROBE_3_TELEMETRY_SCHEMA_VERSION,
                        run_id=self._run_id,
                        checkpoint_id=self._checkpoint_id,
                        t_event=0,
                        event_type="dormant_heartbeat",
                        source="environment",
                        payload={
                            "dormant_started_at_ms": self._started_at_ms,
                            "dormant_wallclock_ms_elapsed": beat_elapsed_ms,
                            "mac_alive": True,
                        },
                        wallclock_ms=self._started_at_ms + beat_elapsed_ms,
                    )
                )
            fired += 1
        return fired

    def stop(self) -> None:
        self._started_at_ms = None
        self._emitted = 0


class StateController:
    """Governs Io's waking ↔ dreaming ↔ dormant transitions.

    Constructor shape follows plan §2.4 (``envelope`` and ``seed_selection``
    first). The dream-driving, world-event emission, protection policy, clock,
    and session-id factory are keyword-only injected dependencies — a documented
    extension of the §2.4 signature so the transition logic is unit-testable in
    isolation (inject stubs) while the integration path injects the real
    :class:`RunDreamSessionDriver` and a real ``world_event`` sink. With
    ``dream_driver=None`` the controller still performs the full transition
    logic and emits transitions; it simply runs no dream rollouts (used by the
    transition-coverage and type-level tests).

    **Paused is not handled by ``tick``** (plan §2.4 commitment 3): ``tick``
    only runs while the process is alive, so it never sees the Mac off. Use
    :meth:`on_shutdown` / :meth:`on_startup` for the supervisor-mediated paused
    transitions.
    """

    def __init__(
        self,
        envelope: DreamEnvelopeConfig,
        seed_selection: SeedSelectionConfig,
        *,
        dream_driver: DreamDriver | None = None,
        protection: ProtectionPolicy | None = None,
        world_event_emit: WorldEventEmit | None = None,
        run_id: str = "state-machine",
        checkpoint_id: str | None = None,
        dream_session_id_factory: Callable[[], str] | None = None,
        clock: Callable[[], int] | None = None,
        initial_state: State = "waking",
    ) -> None:
        self._envelope = envelope
        self._seed_selection = seed_selection
        self._dream_driver = dream_driver
        self._protection: ProtectionPolicy = (
            protection if protection is not None else StubRolloutCountProtection()
        )
        self._emit = world_event_emit
        self._run_id = run_id
        self._checkpoint_id = checkpoint_id
        # Phase 8b: clock used to *measure* each dream session's wall-time (two
        # reads around the synchronous driver call) so the real compute can be
        # recorded into the metabolic ledger. Injectable for deterministic tests;
        # default monotonic. Distinct from the ``wallclock_ms`` passed to tick
        # (a single snapshot, which cannot measure a duration).
        self._clock: Callable[[], int] = clock if clock is not None else _default_clock_ms
        self._id_factory: Callable[[], str] = (
            dream_session_id_factory
            if dream_session_id_factory is not None
            else _default_session_id
        )

        self._state: State = initial_state
        self._current_session_id: str | None = None
        # The cap that ended the current dream session, pending the
        # dreaming → dormant transition on the next tick (dormant ≠ failure:
        # a cap exit is a rest, recorded with the cap trigger).
        self._pending_dormant_trigger: CapTrigger | None = None
        # Wallclock at which the current state was entered, for the payload's
        # wallclock_ms_in_prev_state.
        self._state_entered_wallclock_ms = 0
        self._heartbeat = DormantHeartbeat(
            interval_ms=envelope.dormant_heartbeat_interval_ms,
            emit=world_event_emit,
            run_id=run_id,
            checkpoint_id=checkpoint_id,
        )

    @property
    def current_state(self) -> State:
        return self._state

    @property
    def current_dream_session_id(self) -> str | None:
        return self._current_session_id

    def tick(
        self, host_signals: HostSignals, env_step: int, wallclock_ms: int
    ) -> StateTransition | None:
        """Advance the state machine one step on exogenous host signals.

        Returns the transition that fired this tick, or ``None`` if the state is
        unchanged. ``tick`` is the trigger surface: its only state-bearing input
        is :class:`HostSignals` (exogenous; nothing Io-derived) plus the
        host-observable ``env_step`` and ``wallclock_ms`` counters.
        """
        if self._state == "waking":
            if not host_signals.desktop_alive:
                return self._enter_dreaming(
                    env_step,
                    wallclock_ms,
                    checkpoint_in_progress=host_signals.checkpoint_in_progress,
                )
            return None

        if self._state == "dreaming":
            # desktop_on takes priority over a pending cap: the builder came
            # back before the session settled into dormant.
            if host_signals.desktop_alive:
                return self._transition(
                    "dreaming", "waking", "desktop_on", env_step, wallclock_ms
                )
            if self._pending_dormant_trigger is not None:
                trigger = self._pending_dormant_trigger
                return self._transition(
                    "dreaming", "dormant", trigger, env_step, wallclock_ms
                )
            # No driver ran a session (dream_driver=None) — consult the
            # protection hook directly so a cap can still end the (empty)
            # dreaming span. With the stub rollout-count policy at
            # rollouts_completed=0 this is None unless the cap is 0.
            verdict = self._protection.should_stop(
                DreamSessionContext(
                    envelope=self._envelope,
                    rollouts_completed=0,
                    session_wallclock_ms_elapsed=wallclock_ms
                    - self._state_entered_wallclock_ms,
                    checkpoint_in_progress=host_signals.checkpoint_in_progress,
                )
            )
            if verdict is not None:
                return self._transition(
                    "dreaming", "dormant", verdict.trigger, env_step, wallclock_ms
                )
            return None

        if self._state == "dormant":
            # desktop_on takes priority over re-entry: the builder came back.
            if host_signals.desktop_alive:
                return self._transition(
                    "dormant", "waking", "desktop_on", env_step, wallclock_ms
                )
            # Phase 8b (Fork B = B2): the metabolic re-entry edge. While the
            # desktop is off, re-dream when the content-blind ledger says a
            # projected session fits within the rolling-hour budget (the
            # complement of the compute-budget cap). The re-entry decision reads
            # *nothing Io-derived* — its input is the typed content-blind
            # ``MetabolicState`` (structurally enforced). This extends the trigger
            # surface from HostSignals-only to {HostSignals, content-blind runtime
            # pacer}; the exogenous-trigger commitment's load-bearing sense — no
            # Io-authored dream schedule — is preserved (decision-doc Fork B, the
            # ratified reinterpretation).
            if self._protection.metabolic_reentry(self._envelope, now_ms=wallclock_ms):
                return self._enter_dreaming(
                    env_step,
                    wallclock_ms,
                    checkpoint_in_progress=host_signals.checkpoint_in_progress,
                    from_state="dormant",
                    trigger="metabolic_replenished",
                )
            self._heartbeat.poll(wallclock_ms)
            return None

        # paused is not tick-reachable (plan §2.4 commitment 3). If we somehow
        # tick while paused, do nothing — the supervisor owns resumption.
        return None

    # ---- supervisor-mediated paused transitions (not tick-reachable) -------

    def on_shutdown(self, env_step: int, wallclock_ms: int) -> StateTransition:
        """Record ``* → paused`` on process shutdown (trigger ``mac_off``).

        Paused is the only state entered outside ``tick`` (plan §2.4 commitment
        3): the supervisor calls this as the process goes down. Any in-flight
        dream session id is carried onto the transition for traceability.
        """
        return self._transition(
            self._state, "paused", "mac_off", env_step, wallclock_ms
        )

    def on_startup(
        self, host_signals: HostSignals, env_step: int, wallclock_ms: int
    ) -> StateTransition:
        """Resume from ``paused`` on process startup (trigger ``mac_on``).

        Resume target (plan §2.4 paused row): ``desktop_on`` → ``waking``;
        ``desktop_off`` → ``dormant``. Resuming to *dormant* rather than diving
        straight into ``dreaming`` on boot is the startup-resume default chosen
        here (a design choice within §2.4's "or dreaming if envelope permits"
        latitude) — the conservative reading of capacity-over-exercise: the
        machine does not auto-enter dreaming without an observed desktop-off
        edge.
        """
        target: State = "waking" if host_signals.desktop_alive else "dormant"
        return self._transition("paused", target, "mac_on", env_step, wallclock_ms)

    # ---- internals --------------------------------------------------------

    def _enter_dreaming(
        self,
        env_step: int,
        wallclock_ms: int,
        *,
        checkpoint_in_progress: bool = False,
        from_state: State = "waking",
        trigger: Trigger = "desktop_off",
    ) -> StateTransition:
        """Enter a dream session. ``from_state`` / ``trigger`` distinguish the
        two entry edges: ``waking``/``desktop_off`` (8a) and (8b)
        ``dormant``/``metabolic_replenished`` (the re-entry edge); the body is
        identical — the session runs synchronously here."""
        session_id = str(self._id_factory())
        transition = self._transition(
            from_state, "dreaming", trigger, env_step, wallclock_ms, session_id
        )
        # Drive the session. The driver consults the protection hook to decide
        # where the session stops and reports the realized cap, which the next
        # tick uses to transition dreaming → dormant. With dream_driver=None no
        # rollouts run — the transition logic is exercised on its own.
        if self._dream_driver is not None:
            # Phase 8b: measure the session's real wall-time (two clock reads
            # around the synchronous driver call) and record it into the
            # metabolic ledger via the protection hook, so the rolling-hour
            # budget accumulates actuals — the prerequisite that makes B2's
            # re-entry pacing non-fiction.
            t0 = self._clock()
            outcome = self._dream_driver.run_session(
                dream_session_id=session_id,
                started_at_env_step=env_step,
                started_at_wallclock_ms=wallclock_ms,
                protection=self._protection,
                envelope=self._envelope,
                checkpoint_in_progress=checkpoint_in_progress,
            )
            t1 = self._clock()
            self._pending_dormant_trigger = outcome.end_trigger
            self._protection.record_session(
                num_rollouts=outcome.rollout_count,
                session_wallclock_ms=max(0, t1 - t0),
                now_ms=t1,
            )
        return transition

    def _transition(
        self,
        from_state: State,
        to_state: State,
        trigger: Trigger,
        env_step: int,
        wallclock_ms: int,
        session_id: str | None = None,
    ) -> StateTransition:
        # Resolve the dream_session_id carried on the edge: the new id on a
        # dreaming entry, the current id on a dreaming exit, else None.
        if to_state == "dreaming":
            self._current_session_id = session_id
            dream_session_id = session_id
        elif from_state == "dreaming":
            dream_session_id = self._current_session_id
        else:
            dream_session_id = None

        wallclock_in_prev = wallclock_ms - self._state_entered_wallclock_ms

        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            dream_session_id=dream_session_id,
            trigger=trigger,
            env_step_at_transition=env_step,
            wallclock_ms_at_transition=wallclock_ms,
        )
        self._emit_state_transition(transition, wallclock_in_prev)

        # Post-transition bookkeeping.
        self._state = to_state
        self._state_entered_wallclock_ms = wallclock_ms
        if from_state == "dreaming" and to_state != "dreaming":
            # Session closed: clear the pending cap and the session id.
            self._pending_dormant_trigger = None
            self._current_session_id = None
        if to_state == "dormant":
            self._heartbeat.start(wallclock_ms)
        else:
            self._heartbeat.stop()
        return transition

    def _emit_state_transition(
        self, transition: StateTransition, wallclock_ms_in_prev_state: int
    ) -> None:
        if self._emit is None:
            return
        self._emit.write(
            WorldEvent(
                schema_version=PROBE_3_TELEMETRY_SCHEMA_VERSION,
                run_id=self._run_id,
                checkpoint_id=self._checkpoint_id,
                t_event=transition.env_step_at_transition,
                event_type="state_transition",
                source="environment",
                payload={
                    "from_state": transition.from_state,
                    "to_state": transition.to_state,
                    "dream_session_id": transition.dream_session_id,
                    "trigger": transition.trigger,
                    "wallclock_ms_in_prev_state": wallclock_ms_in_prev_state,
                    "env_step_at_transition": transition.env_step_at_transition,
                },
                wallclock_ms=transition.wallclock_ms_at_transition,
            )
        )


_SESSION_COUNTER = {"n": 0}


def _default_session_id() -> str:
    """A deterministic-ish session id factory (no external entropy in tests).

    Production wiring (Phase 8) can inject a UUID factory; the default keeps the
    module importable and the tests reproducible.
    """
    _SESSION_COUNTER["n"] += 1
    return f"dream-session-{_SESSION_COUNTER['n']:06d}"
