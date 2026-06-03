"""Probe 3 Phase 6 — content-blind runtime protection policies + the rolling ledger.

Phase 4 (``kind/training/state_machine.py``) shipped the protection *hook
surface*: the :class:`~kind.training.state_machine.ProtectionPolicy` Protocol,
the content-blind :class:`~kind.training.state_machine.DreamSessionContext`, the
:class:`~kind.training.state_machine.ProtectionVerdict`, the
:data:`~kind.training.state_machine.CapTrigger` vocabulary, and a trivial
:class:`~kind.training.state_machine.StubRolloutCountProtection`. Phase 6 fills
that surface with the four real content-blind caps and the rolling one-hour
compute ledger they read, composed into :class:`DreamProtectionPolicy` — the
policy injected into the :class:`~kind.training.state_machine.StateController`.

**Why content-blind is load-bearing (synthesis §6).** Envelope control *is*
selection pressure — that is precisely why the mirror gets
``continue_and_log_uncertainty`` and never terminates a dream on content
(Phase 5). Phase 6 is the other half of that resolution: the runtime *does*
bound dreams, but **blind to content** — it stops on time, count, compute, and
checkpoint state, never on *what* Io is dreaming. A policy that stopped a dream
because of what it contained would make the runtime's dream-length bounding a
hidden content filter, reintroducing exactly the selection pressure §6 ruled
out. The structural guarantee is the diet: each cap sees only a
:class:`DreamSessionContext`, which carries only content-blind quantities; the
per-cap sub-policies cannot reach the ledger (or anything else) directly. The
composite is the single integration point that holds the ledger, and the ledger
itself is content-blind (it counts rollout durations and compute, never reads
``DreamRollout`` content).

**The four caps compose (earliest cap wins) and the rollout-count cap is the
absolute ceiling.** :class:`DreamProtectionPolicy` polls its sub-policies in a
fixed priority order and returns the first verdict; the
:func:`~kind.training.state_machine._plan_session_rollouts` poll loop calls it
with an increasing ``rollouts_completed`` and stops at the smallest count any
cap fires — so the earliest cap (by count) wins. The rollout-count cap fires at
``hard_cap_rollout_count`` regardless of the other (estimate-based, hence
fallible) caps, so a coarse wallclock estimate or a stale ledger can never run
the session away.

**The wallclock cap is a projection, not actual-elapsed (interruptibility fork,
option a).** ``run_dream_session`` is one-shot (fixed ``num_rollouts``), and the
planning poll happens with ``session_wallclock_ms_elapsed == 0`` (the session
has not run yet). So the wallclock cap converts its budget to a rollout-count
bound by *projecting* ``rollouts_completed × rollout_duration_estimate_ms``
(from the ledger's running average) rather than measuring elapsed time. This is
coarse by design: the cap's job is runaway-prevention during long builder
absences (the system wakes when the desktop comes on, not on a timer — the cap
is the backstop), and the rollout-count ceiling bounds the coarseness. Precise
mid-session wallclock enforcement would require making ``run_dream_session``
interruptible (option b) — a settled-Phase-2-surface change, the builder's call,
not a Phase 6 drive-by.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Final

from kind.training.state_machine import (
    DreamEnvelopeConfig,
    DreamSessionContext,
    ProtectionVerdict,
)

__all__ = [
    "ONE_HOUR_MS",
    "DEFAULT_ROLLOUT_DURATION_MS",
    "MetabolicState",
    "has_metabolic_room",
    "RollingComputeLedger",
    "RolloutCountCap",
    "WallclockCap",
    "CheckpointWindowCap",
    "ComputeBudgetCap",
    "DreamProtectionPolicy",
]


#: The rolling compute ledger's window length (plan §2.4 "rolling one-hour
#: ledger"). Compute older than this has aged out of the budget.
ONE_HOUR_MS: Final[int] = 60 * 60 * 1000

#: Seed rollout-duration estimate used until the ledger has measured rollouts.
#: A design choice within §2.4's range: §2.4 fixes the budgets (30 min/hour
#: compute, 30 min/session wallclock) but not a per-rollout duration; §2.7's
#: reference is a *compute* figure (~150 K=5 head evals/rollout), not a time.
#:
#: **Measurement (un-seeds the prior 1000.0 placeholder).** Real four-axis dream
#: sessions run against the Probe-1.5 checkpoint
#: (``runs/probe1_5_phase7_5-20260507-101800/``, h=200 z=16 K=5, horizon=30),
#: including the per-rollout replay seed re-encoding:
#:   - Phase 8a, **CPU**: ~11.4 ms/rollout.
#:   - Phase 8b, **MPS** (the deployment device — the mind lives on the Mac):
#:     ~108 ms/rollout — *~9.5× slower than CPU*. At this small scale (h=200,
#:     z=16) the per-kernel MPS dispatch overhead dominates the tiny-tensor
#:     compute, so the GPU loses to CPU. (Worth noting for the deployment: at
#:     this scale, dreaming on CPU is faster; the seed below reflects MPS, the
#:     conservative direction.)
#: Seeded at 110.0 ms (the MPS figure rounded up). This is a *bootstrap* estimate
#: only: Phase 8b wires the ledger to record real session durations
#: (``record_session``), so after the first session the window and the per-rollout
#: estimate self-correct to actuals — the seed matters only for the first
#: session's (and the first re-entry's) look-ahead projection. Over-estimating is
#: the safe direction (caps/the metabolic projection fire earlier). The §2.4
#: ordering holds: the rollout-count ceiling (50) still binds long before the
#: wallclock cap (30 min ⇒ ~16k rollouts at 110 ms).
DEFAULT_ROLLOUT_DURATION_MS: Final[float] = 110.0


def _default_clock_ms() -> int:
    return time.monotonic_ns() // 1_000_000


# ---------------------------------------------------------------------------
# The B2 metabolic re-entry decision (Phase 8b) — content-blind by construction.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetabolicState:
    """Content-blind inputs to the dormant→dreaming re-entry decision (8b).

    Every field is a content-blind primitive — the rolling-hour windowed compute,
    a projected session's compute, the per-hour budget, and wall-clock. There is
    **no** tensor, latent, policy, dream-content, or Io-state-derived field, and
    (the structural guard) there *can be* none: the re-entry decision cannot read
    Io's state because its input type cannot carry it. This is the Phase 8b analog
    of Phase 4's ``HostSignals`` type-test and Phase 6's ``DreamSessionContext``
    content-blindness — "nothing Io-derived gates dreaming" made unrepresentable-
    to-violate rather than conventional. The type-level test in
    ``tests/test_metabolic_reentry.py`` enforces it and is pinned with a positive
    control.

    **Premise (shared with Phase 6).** The ledger's durations are a legitimate
    content-blind compute measure because the four-axis rollout's compute is
    data-independent — a fixed horizon × K-ensemble forward, the same regardless
    of *what* Io dreams. This holds only while rollouts stay fixed-compute; a
    future variable-compute rollout would reopen the question.
    """

    window_compute_seconds: float
    projected_session_seconds: float
    compute_budget_seconds_per_hour: float
    wallclock_ms: int


def has_metabolic_room(state: MetabolicState) -> bool:
    """True iff a projected session fits within the rolling-hour budget.

    The **exact complement of the ``ComputeBudgetCap`` stop condition**: the cap
    stops a session when ``window + projected > budget``; re-entry fires when
    ``window + projected <= budget`` — the rolling-hour window has drained enough
    that another (here: one-rollout-minimum) session's compute fits. The ledger is
    the shared state for both stop (the cap ended the prior session) and restart
    (this complement), which is the metabolic loop: dream until the budget is
    spent, rest while the window drains, re-dream when it has room.
    """
    return (
        state.window_compute_seconds + state.projected_session_seconds
        <= state.compute_budget_seconds_per_hour
    )


class RollingComputeLedger:
    """Content-blind rolling-window accounting of dream-compute.

    Records per-rollout *durations* — the content-blind compute measure: dream
    wall-time, the in-process proxy for the ~150 K=5 head evals/rollout §2.7
    names — each stamped with its completion time, and exposes (a) the compute
    accrued within the trailing window (:meth:`window_compute_seconds`, read by
    the compute-budget cap) and (b) a running rollout-duration estimate
    (:meth:`rollout_duration_estimate_ms`, projected from by the wallclock cap).
    It never sees ``DreamRollout`` content — only durations (numbers) and
    timestamps (numbers).

    **In-process for first build.** A restart empties it; a pause longer than
    the window empties it naturally (correct — old compute has aged out); a
    pause shorter than the window under-counts (the rollout-count ceiling bounds
    that). Across-pause persistence is a flagged refinement, not a Phase 6
    requirement.

    The estimate is an all-time running average (stable), distinct from the
    windowed compute sum (which ages out) — a design choice within §2.4's range.
    """

    def __init__(
        self,
        *,
        window_ms: int = ONE_HOUR_MS,
        default_rollout_duration_ms: float = DEFAULT_ROLLOUT_DURATION_MS,
        clock: Callable[[], int] | None = None,
    ) -> None:
        if window_ms <= 0:
            raise ValueError(f"window_ms must be positive, got {window_ms}")
        if default_rollout_duration_ms <= 0:
            raise ValueError(
                f"default_rollout_duration_ms must be positive, got "
                f"{default_rollout_duration_ms}"
            )
        self._window_ms = window_ms
        self._default_estimate_ms = default_rollout_duration_ms
        self._clock: Callable[[], int] = clock if clock is not None else _default_clock_ms
        # (completed_at_ms, duration_ms) within the trailing window.
        self._entries: deque[tuple[int, float]] = deque()
        # All-time accumulators for the running estimate.
        self._alltime_sum_ms = 0.0
        self._alltime_count = 0

    def record_rollout(self, duration_ms: float, *, now_ms: int | None = None) -> None:
        """Record one completed rollout's wall-time. Content-blind: a duration
        and a timestamp, never any dream content."""
        if duration_ms < 0:
            raise ValueError(f"duration_ms must be non-negative, got {duration_ms}")
        now = now_ms if now_ms is not None else self._clock()
        self._entries.append((now, float(duration_ms)))
        self._alltime_sum_ms += float(duration_ms)
        self._alltime_count += 1
        self._evict(now)

    def record_session(
        self,
        *,
        session_wallclock_ms: float,
        num_rollouts: int,
        now_ms: int | None = None,
    ) -> None:
        """Record a completed session's *real* compute (Phase 8b).

        The rolling-hour window gets the session total (so the metabolic budget
        reflects actual dream wall-time, not the seed estimate), and the all-time
        per-rollout estimate is refined by ``total / num_rollouts``. Content-blind:
        a duration and a count, never any dream content. A zero-rollout session
        contributes nothing to the estimate (no division) but its (zero) duration
        is still windowed."""
        if session_wallclock_ms < 0:
            raise ValueError(
                f"session_wallclock_ms must be non-negative, got {session_wallclock_ms}"
            )
        if num_rollouts < 0:
            raise ValueError(f"num_rollouts must be non-negative, got {num_rollouts}")
        now = now_ms if now_ms is not None else self._clock()
        self._entries.append((now, float(session_wallclock_ms)))
        if num_rollouts > 0:
            self._alltime_sum_ms += float(session_wallclock_ms)
            self._alltime_count += num_rollouts
        self._evict(now)

    def _evict(self, now_ms: int) -> None:
        cutoff = now_ms - self._window_ms
        while self._entries and self._entries[0][0] <= cutoff:
            self._entries.popleft()

    def window_compute_seconds(self, *, now_ms: int | None = None) -> float:
        """Dream-compute (seconds) accrued within the trailing window."""
        now = now_ms if now_ms is not None else self._clock()
        self._evict(now)
        return sum(duration for _, duration in self._entries) / 1000.0

    def rollout_duration_estimate_ms(self) -> float:
        """Running rollout-duration estimate (ms); the seed default until at
        least one rollout has been recorded."""
        if self._alltime_count == 0:
            return self._default_estimate_ms
        return self._alltime_sum_ms / self._alltime_count


# ---------------------------------------------------------------------------
# The four content-blind caps. Each consumes ONLY a DreamSessionContext.
# ---------------------------------------------------------------------------


class RolloutCountCap:
    """The absolute ceiling — fires at ``hard_cap_rollout_count``.

    This is the cap that bounds the other, estimate-based caps: it fires on an
    exact integer count, so no coarse projection or stale ledger can run the
    session past it.
    """

    def should_stop(self, ctx: DreamSessionContext) -> ProtectionVerdict | None:
        if ctx.rollouts_completed >= ctx.envelope.hard_cap_rollout_count:
            return ProtectionVerdict(trigger="hard_cap_rollout_count")
        return None


class WallclockCap:
    """Option (a) — projection, not actual-elapsed.

    Projects ``rollouts_completed × rollout_duration_estimate_ms`` and fires
    when that reaches ``hard_cap_wallclock_ms``. The estimate is the ledger's
    running average (enriched onto the context by the composite). Deliberately
    does *not* read ``session_wallclock_ms_elapsed`` — the planning poll happens
    at elapsed 0 — and is coarse by design: a safety backstop the rollout-count
    ceiling bounds.
    """

    def should_stop(self, ctx: DreamSessionContext) -> ProtectionVerdict | None:
        estimate = ctx.rollout_duration_estimate_ms
        if estimate <= 0:
            return None
        projected_ms = ctx.rollouts_completed * estimate
        if projected_ms >= ctx.envelope.hard_cap_wallclock_ms:
            return ProtectionVerdict(trigger="hard_cap_wallclock")
        return None


class CheckpointWindowCap:
    """Fires when a checkpoint write is in progress, so a dream session does not
    run across the atomic state-sync boundary (the persistence design needs the
    boundary clear). Content-blind: reads only the checkpoint barrier boolean.
    """

    def should_stop(self, ctx: DreamSessionContext) -> ProtectionVerdict | None:
        if ctx.envelope.checkpoint_window_force_dormant and ctx.checkpoint_in_progress:
            return ProtectionVerdict(trigger="checkpoint_window")
        return None


class ComputeBudgetCap:
    """Fires when the rolling-window compute already accrued, plus the projected
    compute of the rollouts this session would add, would exceed the per-hour
    budget. Content-blind: reads the windowed compute sum and the per-rollout
    estimate (both ledger-derived, enriched onto the context), never content.
    """

    def should_stop(self, ctx: DreamSessionContext) -> ProtectionVerdict | None:
        per_rollout_seconds = ctx.rollout_duration_estimate_ms / 1000.0
        if per_rollout_seconds <= 0:
            return None
        projected_seconds = (
            ctx.window_compute_seconds + ctx.rollouts_completed * per_rollout_seconds
        )
        if projected_seconds > ctx.envelope.compute_budget_seconds_per_hour:
            return ProtectionVerdict(trigger="compute_budget")
        return None


# ---------------------------------------------------------------------------
# The composite — the production protection policy.
# ---------------------------------------------------------------------------


class DreamProtectionPolicy:
    """The composite content-blind protection policy injected into
    :class:`~kind.training.state_machine.StateController` — the real policy that
    supersedes :class:`~kind.training.state_machine.StubRolloutCountProtection`.

    Holds the rolling compute ledger (the single ledger-integration point),
    enriches the incoming content-blind :class:`DreamSessionContext` with the
    ledger's two content-blind scalars, then polls the four sub-policies in a
    fixed priority order and returns the first verdict. Because
    :func:`~kind.training.state_machine._plan_session_rollouts` polls with an
    increasing ``rollouts_completed``, the smallest count any cap fires wins —
    the earliest cap. At an equal count the priority order below breaks the tie:

    1. **checkpoint-window** — a checkpoint write must not be crossed; highest
       priority and independent of count.
    2. **wallclock** — the projected-duration backstop.
    3. **compute-budget** — the rolling-window compute backstop.
    4. **rollout-count** — the absolute ceiling, fired last because the other
       caps should bind first when their (smaller) bounds are reached; when none
       of them does, this guarantees the session still stops.

    The per-cap sub-policies consume *only* the (content-blind) context; the
    ledger is reachable only through this composite. That is the structural
    content-blindness guarantee: no content can enter a stop decision because
    neither the context nor the ledger carries any.
    """

    def __init__(self, *, ledger: RollingComputeLedger | None = None) -> None:
        self._ledger = ledger if ledger is not None else RollingComputeLedger()
        self._rollout_count_cap = RolloutCountCap()
        self._wallclock_cap = WallclockCap()
        self._checkpoint_cap = CheckpointWindowCap()
        self._compute_cap = ComputeBudgetCap()
        # Priority order for equal-count ties (see class docstring).
        self._subpolicies = (
            self._checkpoint_cap,
            self._wallclock_cap,
            self._compute_cap,
            self._rollout_count_cap,
        )

    @property
    def ledger(self) -> RollingComputeLedger:
        return self._ledger

    def record_session(
        self, *, num_rollouts: int, session_wallclock_ms: int, now_ms: int
    ) -> None:
        """Record a completed session's real compute into the rolling ledger
        (Phase 8b) — the actuals that make the metabolic budget non-fiction."""
        self._ledger.record_session(
            session_wallclock_ms=session_wallclock_ms,
            num_rollouts=num_rollouts,
            now_ms=now_ms,
        )

    def metabolic_reentry(
        self, envelope: DreamEnvelopeConfig, *, now_ms: int
    ) -> bool:
        """Decide dormant→dreaming re-entry (Phase 8b) — content-blind, the
        complement of the ``ComputeBudgetCap``.

        Projects a one-rollout-minimum session (``rollout_duration_estimate_ms``)
        against the drained rolling-hour window: re-enter iff that fits within
        ``compute_budget_seconds_per_hour``. Reads *only* content-blind ledger
        scalars and the envelope budget — assembled into the typed
        :class:`MetabolicState`, whose content-blindness is structurally
        enforced. Nothing Io-state-derived can enter this decision."""
        estimate_ms = self._ledger.rollout_duration_estimate_ms()
        state = MetabolicState(
            window_compute_seconds=self._ledger.window_compute_seconds(now_ms=now_ms),
            projected_session_seconds=estimate_ms / 1000.0,
            compute_budget_seconds_per_hour=envelope.compute_budget_seconds_per_hour,
            wallclock_ms=now_ms,
        )
        return has_metabolic_room(state)

    def should_stop(self, ctx: DreamSessionContext) -> ProtectionVerdict | None:
        enriched = replace(
            ctx,
            rollout_duration_estimate_ms=self._ledger.rollout_duration_estimate_ms(),
            window_compute_seconds=self._ledger.window_compute_seconds(),
        )
        for sub in self._subpolicies:
            verdict = sub.should_stop(enriched)
            if verdict is not None:
                return verdict
        return None
