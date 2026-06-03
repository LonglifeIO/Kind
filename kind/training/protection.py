"""Probe 3 runtime protection — per-session caps + the metabolic token-bucket pacer.

Phase 4 (``kind/training/state_machine.py``) shipped the protection *hook
surface*: the :class:`~kind.training.state_machine.ProtectionPolicy` Protocol,
the content-blind :class:`~kind.training.state_machine.DreamSessionContext`, the
:class:`~kind.training.state_machine.ProtectionVerdict`, the
:data:`~kind.training.state_machine.CapTrigger` vocabulary, and a trivial
:class:`~kind.training.state_machine.StubRolloutCountProtection`. Phase 6 filled
it with content-blind **per-session caps** (rollout-count, wallclock,
checkpoint-window); Phase 8b added a cross-session metabolic pacer; this phase
**replaces that pacer's mechanism** with a token bucket.

**Two roles, two mechanisms.**

* **Per-session caps** end an *individual* dream session: ``RolloutCountCap``
  (the absolute ceiling), ``WallclockCap`` (a projection backstop), and
  ``CheckpointWindowCap`` (don't cross an atomic state-sync). Each consumes only
  a content-blind :class:`~kind.training.state_machine.DreamSessionContext`. These
  are unchanged.

* **The metabolic pacer** gates *between* sessions during a desktop-off absence
  (the B2 re-entry edge). It is now a :class:`MetabolicBudget` **token bucket**:
  ``tokens`` (compute-seconds) are *spent* by dreaming (``record_session``, real
  compute) and *refill continuously* at ``refill_rate`` toward ``capacity`` as
  wall-time passes. Re-entry fires when the bucket holds a **full session's**
  worth — the hysteresis that produces clean periodic dream/rest cycles.

**Why the token bucket (the mechanism fix this phase makes).** Phase 8b paced
re-entry off a *rolling-hour window* (re-enter when ``window + projected ≤
budget``). That window holds spent compute for a full hour, so with a budget
large relative to a session it produced **burst → long rest → degenerate
1-rollout trickle**: ~100 sessions back-to-back at the start of an absence
(window fills), then no re-entry for ~48 min (window full), then 1-rollout
"dreams" trickling out the instant a single rollout's worth frees up. Root
cause: the window *returns all spent compute at once an hour later* rather than
replenishing continuously. The token bucket fixes both pathologies: **continuous
refill** (vs all-at-once-an-hour-later) yields clean periodic cycles; **capacity
C** bounds the burst; and the **full-session hysteresis** kills the trickle —
re-entry only fires with room for a real session, so every re-dream runs to the
rollout-count cap, never a 1-rollout degenerate.

**Content-blindness is preserved and the structural guard transfers.** The
bucket's state is compute-units and time — ``tokens``, ``capacity``,
``refill_rate``, a projected-session duration, wall-clock — never any
``DreamRollout`` content or Io state. The re-entry decision reads only the typed
content-blind :class:`MetabolicState`, whose content-blindness is enforced by
``tests/test_metabolic_reentry.py`` (two belts + a positive control). "Nothing
Io-derived gates dreaming" stays unrepresentable-to-violate. **Premise (shared
with Phase 6):** the bucket's durations are a legitimate content-blind compute
measure because the four-axis rollout's compute is *data-independent* (fixed
horizon × K-ensemble — the same regardless of *what* Io dreams); holds only while
rollouts stay fixed-compute.

**Interruptibility unchanged (option a).** ``run_dream_session`` stays one-shot;
the bucket gates *between* sessions and never interrupts one. The ``WallclockCap``
projects ``rollouts_completed × rollout_duration_estimate_ms`` at plan time; the
estimate now lives on the bucket (refined by ``record_session``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Final

from kind.training.state_machine import (
    DEFAULT_METABOLIC_CAPACITY_SECONDS,
    DEFAULT_METABOLIC_REFILL_RATE,
    DreamEnvelopeConfig,
    DreamSessionContext,
    ProtectionVerdict,
)

__all__ = [
    "DEFAULT_ROLLOUT_DURATION_MS",
    "MetabolicState",
    "has_metabolic_room",
    "MetabolicBudget",
    "RolloutCountCap",
    "WallclockCap",
    "CheckpointWindowCap",
    "DreamProtectionPolicy",
]


#: Seed rollout-duration estimate used until the bucket has recorded a session.
#:
#: **Measurement (un-seeds the prior 1000.0 placeholder).** Real four-axis dream
#: sessions run against the Probe-1.5 checkpoint
#: (``runs/probe1_5_phase7_5-20260507-101800/``, h=200 z=16 K=5, horizon=30),
#: including the per-rollout replay seed re-encoding:
#:   - Phase 8a, **CPU**: ~11.4 ms/rollout.
#:   - Phase 8b, **MPS** (the deployment device — the mind lives on the Mac):
#:     ~108 ms/rollout — *~9.5× slower than CPU*. At this small scale (h=200,
#:     z=16) the per-kernel MPS dispatch overhead dominates the tiny-tensor
#:     compute, so the GPU loses to CPU. (At this scale, dreaming on CPU is
#:     faster; the seed below reflects MPS, the conservative direction.)
#: Seeded at 110.0 ms (the MPS figure rounded up). A *bootstrap* only: the bucket
#: records real session durations (``record_session``), so after the first session
#: the per-rollout estimate self-corrects to actuals — the seed matters only for
#: the first session's (and the first re-entry's) look-ahead projection.
#: Over-estimating is the safe direction (the wallclock cap / the metabolic
#: projection fire earlier).
DEFAULT_ROLLOUT_DURATION_MS: Final[float] = 110.0


def _default_clock_ms() -> int:
    return time.monotonic_ns() // 1_000_000


# ---------------------------------------------------------------------------
# The metabolic re-entry decision — content-blind by construction.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetabolicState:
    """Content-blind inputs to the dormant→dreaming re-entry decision.

    Every field is a content-blind primitive — the token-bucket ``tokens`` /
    ``capacity`` / ``refill_rate``, a projected session's compute, and wall-clock.
    There is **no** tensor, latent, policy, dream-content, or Io-state-derived
    field, and (the structural guard) there *can be* none: the re-entry decision
    cannot read Io's state because its input type cannot carry it. The Phase 8b/8
    analog of Phase 4's ``HostSignals`` type-test and Phase 6's
    ``DreamSessionContext`` content-blindness — "nothing Io-derived gates dreaming"
    made unrepresentable-to-violate. ``tests/test_metabolic_reentry.py`` enforces
    it (forbidden-types + Io-derived-name belts) with a positive control.
    """

    tokens: float
    capacity: float
    refill_rate: float
    projected_session_seconds: float
    wallclock_ms: int


def has_metabolic_room(state: MetabolicState) -> bool:
    """True iff the bucket holds a **full projected session's** worth of tokens.

    The full-session threshold (not one rollout) is the **hysteresis** that kills
    the trickle: re-entry only fires with room for a real session, so every
    re-dream runs a full session, never a 1-rollout degenerate. (``capacity`` /
    ``refill_rate`` ride on the state for the content-blind contract and for
    callers that want to reason about the cycle; the decision itself is the token
    comparison.)
    """
    return state.tokens >= state.projected_session_seconds


class MetabolicBudget:
    """The cross-session metabolic pacer — a continuous-refill token bucket.

    ``tokens`` (compute-seconds) deplete when Io dreams (``record_session`` spends
    the session's real compute) and **refill continuously** at ``refill_rate``
    (compute-s per wall-s) toward ``capacity`` as wall-time passes. The bucket
    also tracks the per-rollout duration estimate (all-time average of recorded
    sessions, seeded with :data:`DEFAULT_ROLLOUT_DURATION_MS`) used for the
    look-ahead projection before a session runs.

    **In-process for first build.** A restart resets it to full; across-pause
    persistence is a flagged refinement. The bucket starts full so Io can dream at
    the start of an absence.
    """

    def __init__(
        self,
        *,
        capacity_seconds: float = DEFAULT_METABOLIC_CAPACITY_SECONDS,
        refill_rate: float = DEFAULT_METABOLIC_REFILL_RATE,
        seed_rollout_duration_ms: float = DEFAULT_ROLLOUT_DURATION_MS,
        initial_tokens: float | None = None,
    ) -> None:
        if capacity_seconds <= 0:
            raise ValueError(f"capacity_seconds must be positive, got {capacity_seconds}")
        if refill_rate < 0:
            raise ValueError(f"refill_rate must be non-negative, got {refill_rate}")
        if seed_rollout_duration_ms <= 0:
            raise ValueError(
                f"seed_rollout_duration_ms must be positive, got {seed_rollout_duration_ms}"
            )
        self._capacity = capacity_seconds
        self._refill_rate = refill_rate
        self._seed_ms = seed_rollout_duration_ms
        self._tokens = capacity_seconds if initial_tokens is None else initial_tokens
        self._alltime_sum_ms = 0.0
        self._alltime_count = 0
        self._last_ms: int | None = None

    @property
    def capacity_seconds(self) -> float:
        return self._capacity

    @property
    def refill_rate(self) -> float:
        return self._refill_rate

    def _refill(self, now_ms: int) -> None:
        """Credit continuous refill up to ``now_ms`` (clamped to capacity)."""
        if self._last_ms is None:
            self._last_ms = now_ms
            return
        elapsed_s = max(0, now_ms - self._last_ms) / 1000.0
        self._tokens = min(self._capacity, self._tokens + self._refill_rate * elapsed_s)
        self._last_ms = now_ms

    def current_tokens(self, *, now_ms: int) -> float:
        """The refilled token balance at ``now_ms``."""
        self._refill(now_ms)
        return self._tokens

    def rollout_duration_estimate_ms(self) -> float:
        """Per-rollout estimate (ms): the seed until a session is recorded, then
        the all-time per-rollout average."""
        if self._alltime_count == 0:
            return self._seed_ms
        return self._alltime_sum_ms / self._alltime_count

    def record_session(
        self,
        *,
        session_wallclock_ms: float,
        num_rollouts: int,
        now_ms: int,
    ) -> None:
        """Spend a completed session's *real* compute from the bucket and refine
        the per-rollout estimate. Content-blind: a duration and a count, never any
        dream content. Refills up to ``now_ms`` first (crediting the wall-time the
        session spanned), then spends — net per session is ``−compute × (1 −
        refill_rate)``, the metabolic drain. Tokens may go transiently negative
        (over-draw debt); continuous refill recovers them."""
        if session_wallclock_ms < 0:
            raise ValueError(
                f"session_wallclock_ms must be non-negative, got {session_wallclock_ms}"
            )
        if num_rollouts < 0:
            raise ValueError(f"num_rollouts must be non-negative, got {num_rollouts}")
        self._refill(now_ms)
        self._tokens = min(self._capacity, self._tokens) - session_wallclock_ms / 1000.0
        if num_rollouts > 0:
            self._alltime_sum_ms += float(session_wallclock_ms)
            self._alltime_count += num_rollouts

    def affords_session(self, projected_session_seconds: float, *, now_ms: int) -> bool:
        """True iff a full projected session fits in the (refilled) bucket."""
        return has_metabolic_room(
            MetabolicState(
                tokens=self.current_tokens(now_ms=now_ms),
                capacity=self._capacity,
                refill_rate=self._refill_rate,
                projected_session_seconds=projected_session_seconds,
                wallclock_ms=now_ms,
            )
        )


# ---------------------------------------------------------------------------
# The per-session caps. Each consumes ONLY a DreamSessionContext.
# ---------------------------------------------------------------------------


class RolloutCountCap:
    """The absolute ceiling — fires at ``hard_cap_rollout_count``.

    Fires on an exact integer count, so no coarse projection can run the session
    past it. With the compute-budget cap gone (now the cross-session bucket's
    job), this is the working per-session bound: every session runs to it, which
    is *why* the metabolic re-entry's full-session hysteresis produces full
    sessions rather than a trickle.
    """

    def should_stop(self, ctx: DreamSessionContext) -> ProtectionVerdict | None:
        if ctx.rollouts_completed >= ctx.envelope.hard_cap_rollout_count:
            return ProtectionVerdict(trigger="hard_cap_rollout_count")
        return None


class WallclockCap:
    """Option (a) — projection, not actual-elapsed.

    Projects ``rollouts_completed × rollout_duration_estimate_ms`` and fires when
    that reaches ``hard_cap_wallclock_ms``. The estimate is the bucket's running
    average (enriched onto the context by the composite). A coarse safety backstop
    the rollout-count ceiling bounds.
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
    run across the atomic state-sync boundary. Content-blind: reads only the
    checkpoint barrier boolean.
    """

    def should_stop(self, ctx: DreamSessionContext) -> ProtectionVerdict | None:
        if ctx.envelope.checkpoint_window_force_dormant and ctx.checkpoint_in_progress:
            return ProtectionVerdict(trigger="checkpoint_window")
        return None


# ---------------------------------------------------------------------------
# The composite — the production protection policy.
# ---------------------------------------------------------------------------


class DreamProtectionPolicy:
    """The composite content-blind protection policy injected into
    :class:`~kind.training.state_machine.StateController` — superseding
    :class:`~kind.training.state_machine.StubRolloutCountProtection`.

    Holds the metabolic token bucket (the cross-session pacer + the per-rollout
    estimate). For *per-session* stops it enriches the incoming content-blind
    :class:`DreamSessionContext` with the bucket's estimate, then polls the three
    per-session caps in priority order:

    1. **checkpoint-window** — a checkpoint write must not be crossed.
    2. **wallclock** — the projected-duration backstop.
    3. **rollout-count** — the absolute ceiling (the working bound).

    For *between-session* pacing it exposes ``record_session`` (spend real compute)
    and ``metabolic_reentry`` (re-enter iff a full session is affordable). The
    per-cap sub-policies consume *only* the content-blind context; the bucket is
    reachable only through this composite — the structural content-blindness
    guarantee.
    """

    def __init__(self, *, budget: MetabolicBudget | None = None) -> None:
        self._budget = budget if budget is not None else MetabolicBudget()
        self._rollout_count_cap = RolloutCountCap()
        self._wallclock_cap = WallclockCap()
        self._checkpoint_cap = CheckpointWindowCap()
        self._subpolicies = (
            self._checkpoint_cap,
            self._wallclock_cap,
            self._rollout_count_cap,
        )

    @property
    def budget(self) -> MetabolicBudget:
        return self._budget

    def record_session(
        self, *, num_rollouts: int, session_wallclock_ms: int, now_ms: int
    ) -> None:
        """Spend a completed session's real compute from the metabolic bucket."""
        self._budget.record_session(
            session_wallclock_ms=session_wallclock_ms,
            num_rollouts=num_rollouts,
            now_ms=now_ms,
        )

    def metabolic_reentry(self, envelope: DreamEnvelopeConfig, *, now_ms: int) -> bool:
        """Decide dream entry/re-entry — content-blind, the token-bucket
        affordability check with full-session hysteresis.

        Projects a **full** session (``hard_cap_rollout_count ×
        rollout_duration_estimate``) and re-enters iff the refilled bucket holds
        that many tokens. The full-session threshold is the hysteresis that kills
        the 1-rollout trickle. Reads *only* content-blind bucket scalars + the
        envelope rollout cap (assembled into the typed :class:`MetabolicState`)."""
        estimate_s = self._budget.rollout_duration_estimate_ms() / 1000.0
        projected_full = max(0, envelope.hard_cap_rollout_count) * estimate_s
        return self._budget.affords_session(projected_full, now_ms=now_ms)

    def should_stop(self, ctx: DreamSessionContext) -> ProtectionVerdict | None:
        enriched = replace(
            ctx,
            rollout_duration_estimate_ms=self._budget.rollout_duration_estimate_ms(),
        )
        for sub in self._subpolicies:
            verdict = sub.should_stop(enriched)
            if verdict is not None:
                return verdict
        return None
