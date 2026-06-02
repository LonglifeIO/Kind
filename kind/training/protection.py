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
from dataclasses import replace
from typing import Final

from kind.training.state_machine import (
    DreamSessionContext,
    ProtectionVerdict,
)

__all__ = [
    "ONE_HOUR_MS",
    "DEFAULT_ROLLOUT_DURATION_MS",
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
#: **Phase 8a measurement (un-seeds the prior 1000.0 placeholder).** A real
#: four-axis dream session run against the Probe-1.5 checkpoint
#: (``runs/probe1_5_phase7_5-20260507-101800/``, h=200 z=16 K=5, horizon=30, CPU)
#: measured ~11.4 ms/rollout — including the per-rollout replay seed re-encoding.
#: Seeded here at 15.0 ms (the measured figure rounded up for a small conservative
#: margin and device-variance headroom; the real deployment is MPS). The ordering
#: §2.4 intends is preserved: the rollout-count ceiling (default 50) still binds
#: long before the wallclock cap (default 30 min ⇒ ~120k rollouts at 15 ms),
#: rollout-count being the working bound and wallclock the slow-rollout backstop.
#: NB: the live driver does not yet feed measured durations back into the ledger
#: (it stays cold, so this seed is the operative estimate); wiring
#: ``record_rollout`` into the session loop is a flagged refinement (and, under the
#: as-built single-session-per-absence model, the ledger's cross-session role is
#: dormant until Fork B's re-dreaming edge — Phase 8b).
DEFAULT_ROLLOUT_DURATION_MS: Final[float] = 15.0


def _default_clock_ms() -> int:
    return time.monotonic_ns() // 1_000_000


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
