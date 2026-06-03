"""Phase 6 gate tests — content-blind runtime protection (plan §4 Phase 6 row).

The load-bearing test is the content-blindness type-level guard: every
``DreamSessionContext`` field is a content-blind primitive (no ``DreamRollout``,
mirror reading, latent/tensor, or Io-derived type), and the per-cap policies
consume only that context. It is sanity-checked to genuinely trip on an injected
content-bearing field.

The structural tests: each cap fires at the right point; the caps compose
(earliest wins, rollout-count is the absolute ceiling that bounds a bad
estimate); the ledger does rolling-window accounting content-blind; and the
composite drives the right cap → dormant through the ``StateController`` (the
dormant-≠-failure path). One test confirms ``run_dream_session`` is untouched
(option a held, not option b).
"""

from __future__ import annotations

import dataclasses
import inspect
import subprocess
import typing
from dataclasses import dataclass
from pathlib import Path

import torch

from kind.agents.views import PolicyView
from kind.mirror.dream_reading import DreamReading
from kind.mirror.structured import StructuredClaim
from kind.observer.schemas import AgentStep, DreamRollout
from kind.training.protection import (
    DEFAULT_ROLLOUT_DURATION_MS,
    CheckpointWindowCap,
    ComputeBudgetCap,
    DreamProtectionPolicy,
    RollingComputeLedger,
    RolloutCountCap,
    WallclockCap,
)
from kind.training.state_machine import (
    DreamEnvelopeConfig,
    DreamSessionContext,
    DreamSessionOutcome,
    HostSignals,
    ProtectionVerdict,
    StateController,
    _plan_session_rollouts,
)
from kind.training.dream_seed import SeedSelectionConfig

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _ctx(
    *,
    rollouts_completed: int = 0,
    session_wallclock_ms_elapsed: int = 0,
    checkpoint_in_progress: bool = False,
    rollout_duration_estimate_ms: float = 0.0,
    window_compute_seconds: float = 0.0,
    envelope: DreamEnvelopeConfig | None = None,
) -> DreamSessionContext:
    return DreamSessionContext(
        envelope=envelope if envelope is not None else DreamEnvelopeConfig(),
        rollouts_completed=rollouts_completed,
        session_wallclock_ms_elapsed=session_wallclock_ms_elapsed,
        checkpoint_in_progress=checkpoint_in_progress,
        rollout_duration_estimate_ms=rollout_duration_estimate_ms,
        window_compute_seconds=window_compute_seconds,
    )


class _PlanningDriver:
    """A ``DreamDriver`` that exercises the real planning path with the injected
    composite policy, without the heavy ``run_dream_session``.

    ``RunDreamSessionDriver`` determines ``num_rollouts``/``end_trigger`` via
    :func:`_plan_session_rollouts` and then runs the session once with that
    count; this driver does exactly that planning step (the part the composite
    governs) and reports the realized cap, so the cap → dormant path is verified
    end-to-end through the controller with the real composite.
    """

    def __init__(self) -> None:
        self.sessions: list[str] = []

    def run_session(
        self,
        *,
        dream_session_id: str,
        started_at_env_step: int,
        started_at_wallclock_ms: int,
        protection: object,
        envelope: DreamEnvelopeConfig,
        checkpoint_in_progress: bool = False,
    ) -> DreamSessionOutcome:
        self.sessions.append(dream_session_id)
        count, trigger = _plan_session_rollouts(
            protection,  # type: ignore[arg-type]
            envelope,
            checkpoint_in_progress=checkpoint_in_progress,
        )
        return DreamSessionOutcome(end_trigger=trigger, rollout_count=count)


def _controller(
    *,
    envelope: DreamEnvelopeConfig,
    protection: DreamProtectionPolicy,
    driver: _PlanningDriver | None,
) -> StateController:
    return StateController(
        envelope,
        SeedSelectionConfig(),
        dream_driver=driver,
        protection=protection,
        dream_session_id_factory=lambda: "s1",
    )


_OFF = HostSignals(desktop_alive=False)
_OFF_CHECKPOINT = HostSignals(desktop_alive=False, checkpoint_in_progress=True)


# ---------------------------------------------------------------------------
# Content-blindness (LOAD-BEARING, type-level).
# ---------------------------------------------------------------------------

_CONTENT_BLIND_SCALARS = {int, float, bool, str}

# Types that must never appear on the protection surface: anything sourced from
# Io's data plane or carrying dream content.
_FORBIDDEN_CONTENT_TYPES = {
    torch.Tensor,
    DreamRollout,
    AgentStep,
    PolicyView,
    DreamReading,
    StructuredClaim,
}


def _content_blind_offenders(dc: type) -> list[str]:
    """Return the names of any fields on a dataclass that are not content-blind.

    A field is content-blind if it is a primitive scalar (int/float/bool/str) or
    the content-blind :class:`DreamEnvelopeConfig` (recursively all-primitive).
    Anything else — a tensor, a DreamRollout, a mirror reading, a latent — is an
    offender.
    """
    hints = typing.get_type_hints(dc)
    offenders: list[str] = []
    for name, annotation in hints.items():
        if annotation in _CONTENT_BLIND_SCALARS:
            continue
        if annotation is DreamEnvelopeConfig:
            if _content_blind_offenders(DreamEnvelopeConfig):
                offenders.append(name)
            continue
        offenders.append(name)
    return offenders


def test_dream_session_context_is_content_blind() -> None:
    """LOAD-BEARING. Every ``DreamSessionContext`` field is a content-blind
    primitive (or the content-blind envelope). No field is a ``DreamRollout``,
    mirror reading, latent/tensor, or Io-derived type — so no content can enter
    a stop decision through the context. The structural face of the synthesis
    §6 "envelope control must not become a content filter" commitment, the
    Phase 6 analog of Phase 4's ``HostSignals`` type-level guard."""
    offenders = _content_blind_offenders(DreamSessionContext)
    assert offenders == [], (
        f"DreamSessionContext has non-content-blind field(s) {offenders}; the "
        f"protection surface must carry only content-blind quantities "
        f"(synthesis §6)."
    )
    # Explicit belt-and-braces: no forbidden Io/content type on any field.
    hints = typing.get_type_hints(DreamSessionContext)
    for name, annotation in hints.items():
        assert annotation not in _FORBIDDEN_CONTENT_TYPES, (
            f"DreamSessionContext.{name} is a content/Io-derived type "
            f"{annotation!r} — that would let dream content drive a stop."
        )


def test_content_blindness_check_trips_on_injected_content_field() -> None:
    """Sanity-check the guard genuinely fails when a content-bearing field is
    added — the load-bearing test is only meaningful if it can fail."""

    @dataclass(frozen=True)
    class _BadContextWithDreamRollout:
        envelope: DreamEnvelopeConfig
        rollouts_completed: int
        smuggled_content: DreamRollout  # a content-bearing field

    @dataclass(frozen=True)
    class _BadContextWithTensor:
        rollouts_completed: int
        latent: torch.Tensor  # a latent tensor

    assert _content_blind_offenders(_BadContextWithDreamRollout) == [
        "smuggled_content"
    ]
    assert _content_blind_offenders(_BadContextWithTensor) == ["latent"]


def test_policies_consume_only_dream_session_context() -> None:
    """Each per-cap policy's ``should_stop`` takes only a ``DreamSessionContext``
    — they cannot see content because the only thing they consume doesn't carry
    any. (The composite additionally holds the content-blind ledger; the
    sub-policies reach nothing beyond the context.)"""
    for policy in (
        RolloutCountCap(),
        WallclockCap(),
        CheckpointWindowCap(),
        ComputeBudgetCap(),
        DreamProtectionPolicy(),
    ):
        sig = inspect.signature(policy.should_stop)
        params = [p for p in sig.parameters if p != "self"]
        assert params == ["ctx"], f"{type(policy).__name__}.should_stop params {params}"
        hints = typing.get_type_hints(policy.should_stop)
        assert hints["ctx"] is DreamSessionContext


def test_ledger_is_content_blind() -> None:
    """The rolling ledger records and exposes only numbers — durations and
    timestamps — never any dream content. No ledger method touches a forbidden
    Io/content type."""
    for method_name in (
        "record_rollout",
        "window_compute_seconds",
        "rollout_duration_estimate_ms",
    ):
        method = getattr(RollingComputeLedger, method_name)
        hints = typing.get_type_hints(method)
        for name, annotation in hints.items():
            assert annotation not in _FORBIDDEN_CONTENT_TYPES, (
                f"RollingComputeLedger.{method_name} param/return {name} is a "
                f"content/Io type {annotation!r}."
            )
    # Recorded entries are numeric tuples, not content.
    ledger = RollingComputeLedger(clock=lambda: 0)
    ledger.record_rollout(123.0, now_ms=0)
    completed_at, duration = ledger._entries[0]
    assert isinstance(completed_at, int)
    assert isinstance(duration, float)


def test_no_mirror_verdict_can_short_circuit_the_runtime() -> None:
    """The §4 "mirror uncertain never short-circuits the runtime" guard: the
    protection surface carries no mirror field, so a mirror reading/verdict
    cannot enter a stop decision. The runtime bounds dreams blind to the
    mirror's content rulings (Phase 5's logs-only mirror is structurally unable
    to shorten a dream)."""
    hints = typing.get_type_hints(DreamSessionContext)
    for annotation in hints.values():
        assert annotation not in {DreamReading, StructuredClaim}
    # A policy's verdict is a pure function of the content-blind context: two
    # contexts equal in their content-blind fields yield the same verdict.
    policy = DreamProtectionPolicy()
    a = _ctx(rollouts_completed=2, envelope=DreamEnvelopeConfig(hard_cap_rollout_count=5))
    b = _ctx(rollouts_completed=2, envelope=DreamEnvelopeConfig(hard_cap_rollout_count=5))
    assert policy.should_stop(a) == policy.should_stop(b)


# ---------------------------------------------------------------------------
# Each cap fires at the right point.
# ---------------------------------------------------------------------------


def test_rollout_count_cap_fires_at_the_count() -> None:
    cap = RolloutCountCap()
    env = DreamEnvelopeConfig(hard_cap_rollout_count=3)
    assert cap.should_stop(_ctx(rollouts_completed=2, envelope=env)) is None
    assert cap.should_stop(_ctx(rollouts_completed=3, envelope=env)) == ProtectionVerdict(
        trigger="hard_cap_rollout_count"
    )


def test_wallclock_cap_fires_at_projected_count() -> None:
    """Option (a): fires at the projected count estimate × count ≥ budget — a
    projection, not actual-elapsed."""
    cap = WallclockCap()
    env = DreamEnvelopeConfig(hard_cap_wallclock_ms=5000)
    # estimate 1000 ms/rollout ⇒ projected reaches 5000 at the 5th rollout.
    assert cap.should_stop(
        _ctx(rollouts_completed=4, rollout_duration_estimate_ms=1000.0, envelope=env)
    ) is None
    assert cap.should_stop(
        _ctx(rollouts_completed=5, rollout_duration_estimate_ms=1000.0, envelope=env)
    ) == ProtectionVerdict(trigger="hard_cap_wallclock")
    # A zero estimate never fires (the rollout-count ceiling bounds instead).
    assert cap.should_stop(
        _ctx(rollouts_completed=10_000, rollout_duration_estimate_ms=0.0, envelope=env)
    ) is None


def test_checkpoint_window_cap_fires_on_checkpoint_in_progress() -> None:
    cap = CheckpointWindowCap()
    assert cap.should_stop(_ctx(checkpoint_in_progress=False)) is None
    assert cap.should_stop(_ctx(checkpoint_in_progress=True)) == ProtectionVerdict(
        trigger="checkpoint_window"
    )
    # Respects the envelope toggle.
    env_off = DreamEnvelopeConfig(checkpoint_window_force_dormant=False)
    assert cap.should_stop(_ctx(checkpoint_in_progress=True, envelope=env_off)) is None


def test_compute_budget_cap_fires_when_window_would_exceed() -> None:
    cap = ComputeBudgetCap()
    env = DreamEnvelopeConfig(compute_budget_seconds_per_hour=10.0)
    # 8 s already in the window, 1 s/rollout: projected 8 + k exceeds 10 at k=3.
    assert cap.should_stop(
        _ctx(
            rollouts_completed=2,
            rollout_duration_estimate_ms=1000.0,
            window_compute_seconds=8.0,
            envelope=env,
        )
    ) is None
    assert cap.should_stop(
        _ctx(
            rollouts_completed=3,
            rollout_duration_estimate_ms=1000.0,
            window_compute_seconds=8.0,
            envelope=env,
        )
    ) == ProtectionVerdict(trigger="compute_budget")


# ---------------------------------------------------------------------------
# Composition — earliest cap wins; rollout-count is the absolute ceiling.
# ---------------------------------------------------------------------------


def test_composition_rollout_count_ceiling_bounds_a_bad_wallclock_estimate() -> None:
    """A pathologically small rollout-duration estimate would defer the
    wallclock cap far past any sane count; the rollout-count ceiling must still
    bound the session."""
    # estimate 1 ms ⇒ wallclock projection reaches the 30-min budget only at
    # ~1.8M rollouts — far past the ceiling.
    ledger = RollingComputeLedger(default_rollout_duration_ms=1.0)
    composite = DreamProtectionPolicy(ledger=ledger)
    env = DreamEnvelopeConfig(
        hard_cap_rollout_count=50, hard_cap_wallclock_ms=30 * 60 * 1000
    )
    count, trigger = _plan_session_rollouts(composite, env)
    assert (count, trigger) == (50, "hard_cap_rollout_count")


def test_composition_wallclock_wins_when_estimate_is_large() -> None:
    """When rollouts are slow (large estimate), the wallclock backstop fires
    before the rollout-count ceiling — the earliest cap wins."""
    ledger = RollingComputeLedger(default_rollout_duration_ms=60_000.0)  # 60 s/rollout
    composite = DreamProtectionPolicy(ledger=ledger)
    env = DreamEnvelopeConfig(
        hard_cap_rollout_count=50,
        hard_cap_wallclock_ms=30 * 60 * 1000,
        # High budget so the (8b) pacer default (600 s) doesn't fire first — this
        # test isolates the wallclock cap winning over rollout-count.
        compute_budget_seconds_per_hour=10**6,
    )  # 30 min ⇒ projection hits the budget at 30 rollouts < 50
    count, trigger = _plan_session_rollouts(composite, env)
    assert (count, trigger) == (30, "hard_cap_wallclock")


def test_composition_compute_budget_can_win() -> None:
    """A nearly-spent compute window makes the compute cap fire before the
    rollout-count ceiling."""
    clock = {"t": 1_000_000}
    ledger = RollingComputeLedger(
        default_rollout_duration_ms=1000.0, clock=lambda: clock["t"]
    )
    # 7 prior rollouts of 1 s each within the window ⇒ 7 s accrued.
    for _ in range(7):
        ledger.record_rollout(1000.0, now_ms=clock["t"])
    composite = DreamProtectionPolicy(ledger=ledger)
    env = DreamEnvelopeConfig(
        hard_cap_rollout_count=50,
        hard_cap_wallclock_ms=30 * 60 * 1000,
        compute_budget_seconds_per_hour=10.0,
    )
    # 7 s window + 1 s/rollout ⇒ projected exceeds 10 s at 4 rollouts.
    count, trigger = _plan_session_rollouts(composite, env)
    assert (count, trigger) == (4, "compute_budget")


# ---------------------------------------------------------------------------
# The ledger — rolling-window accounting, content-blind.
# ---------------------------------------------------------------------------


def test_ledger_rolling_window_ages_out_old_compute() -> None:
    clock = {"t": 0}
    ledger = RollingComputeLedger(window_ms=1000, clock=lambda: clock["t"])
    ledger.record_rollout(500.0, now_ms=0)
    ledger.record_rollout(500.0, now_ms=500)
    # Both within the 1 s window at t=900 ⇒ 1.0 s.
    assert ledger.window_compute_seconds(now_ms=900) == 1.0
    # At t=1400 (cutoff 400) the first (t=0) rollout has aged out; the second
    # (t=500) is still within the window ⇒ 0.5 s.
    assert ledger.window_compute_seconds(now_ms=1400) == 0.5
    # At t=1600 (cutoff 600) the t=500 rollout has also aged out ⇒ 0.0 s.
    assert ledger.window_compute_seconds(now_ms=1600) == 0.0


def test_ledger_running_estimate_is_all_time_average() -> None:
    ledger = RollingComputeLedger(default_rollout_duration_ms=1000.0, clock=lambda: 0)
    # Cold ledger returns the seed default.
    assert ledger.rollout_duration_estimate_ms() == 1000.0
    ledger.record_rollout(200.0, now_ms=0)
    ledger.record_rollout(400.0, now_ms=0)
    # All-time average, stable even as windowed compute ages out.
    assert ledger.rollout_duration_estimate_ms() == 300.0


# ---------------------------------------------------------------------------
# Through the StateController — the right cap fires → dormant (dormant ≠ failure).
# ---------------------------------------------------------------------------


def test_controller_rollout_count_cap_to_dormant() -> None:
    composite = DreamProtectionPolicy()  # cold ledger, default estimate
    env = DreamEnvelopeConfig(hard_cap_rollout_count=3, hard_cap_wallclock_ms=10**12)
    driver = _PlanningDriver()
    c = _controller(envelope=env, protection=composite, driver=driver)
    t1 = c.tick(_OFF, env_step=1, wallclock_ms=100)  # → dreaming
    assert t1 is not None and t1.to_state == "dreaming"
    t = c.tick(_OFF, env_step=2, wallclock_ms=200)  # → dormant on the pending cap
    assert t is not None
    assert (t.from_state, t.to_state, t.trigger) == (
        "dreaming",
        "dormant",
        "hard_cap_rollout_count",
    )
    assert c.current_state == "dormant"


def test_controller_wallclock_cap_to_dormant() -> None:
    ledger = RollingComputeLedger(default_rollout_duration_ms=60_000.0)
    composite = DreamProtectionPolicy(ledger=ledger)
    env = DreamEnvelopeConfig(
        hard_cap_rollout_count=50,
        hard_cap_wallclock_ms=30 * 60 * 1000,
        compute_budget_seconds_per_hour=10**6,  # high so the pacer doesn't fire first
    )
    c = _controller(envelope=env, protection=composite, driver=_PlanningDriver())
    c.tick(_OFF, 1, 100)  # → dreaming, driver plans via the composite
    t = c.tick(_OFF, 2, 200)  # → dormant
    assert t is not None
    assert (t.from_state, t.to_state, t.trigger) == ("dreaming", "dormant", "hard_cap_wallclock")


def test_controller_checkpoint_window_cap_to_dormant() -> None:
    """The checkpoint cap fires via the dreaming-state tick where the real
    ``checkpoint_in_progress`` host signal flows (driver=None: no driven
    session ran, so the tick consults the policy directly)."""
    composite = DreamProtectionPolicy()
    env = DreamEnvelopeConfig()
    c = _controller(envelope=env, protection=composite, driver=None)
    t1 = c.tick(_OFF, 1, 100)  # → dreaming (no driver ⇒ no session)
    assert t1 is not None and t1.to_state == "dreaming"
    t = c.tick(_OFF_CHECKPOINT, 2, 200)  # checkpoint in progress → dormant
    assert t is not None
    assert (t.from_state, t.to_state, t.trigger) == ("dreaming", "dormant", "checkpoint_window")
    assert c.current_state == "dormant"


def test_controller_cap_to_dormant_raises_nothing() -> None:
    """Dormant ≠ failure: a capped dream rests, it does not error."""
    composite = DreamProtectionPolicy()
    env = DreamEnvelopeConfig(hard_cap_rollout_count=1)
    c = _controller(envelope=env, protection=composite, driver=_PlanningDriver())
    c.tick(_OFF, 1, 100)
    t = c.tick(_OFF, 2, 200)
    assert t is not None and t.to_state == "dormant"


# ---------------------------------------------------------------------------
# run_dream_session unchanged (option a held, not option b).
# ---------------------------------------------------------------------------


def test_run_dream_session_module_unchanged() -> None:
    """Option (a) keeps ``kind/training/dream.py`` untouched; reopening it is
    option (b), out of Phase 6 scope. Assert no working-tree diff against HEAD
    for that file."""
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", "kind/training/dream.py"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", (
        f"kind/training/dream.py has uncommitted changes — Phase 6 must not "
        f"reopen run_dream_session (option b is the builder's call):\n{result.stdout}"
    )


def test_envelope_defaults_match_plan_2_4() -> None:
    """The §2.4 cap values the policies enforce."""
    env = DreamEnvelopeConfig()
    assert env.hard_cap_wallclock_ms == 30 * 60 * 1000
    assert env.hard_cap_rollout_count == 50
    assert env.checkpoint_window_force_dormant is True
    # Phase 8b (Fork B = B2): compute_budget became the dream/rest pacer; the
    # default dropped from the Phase-6 protection value (1800 s) to a rest-
    # majority ~17%-duty-cycle pacer (600 s). The builder's knob to tune.
    assert env.compute_budget_seconds_per_hour == 600.0
    # Phase 8b re-tuned the seed to the MPS (deployment-device) per-rollout cost
    # (~108 ms on the Probe-1.5 checkpoint — ~9.5× the CPU figure, MPS dispatch
    # overhead dominating at this small scale), rounded up to 110.0.
    assert DEFAULT_ROLLOUT_DURATION_MS == 110.0


def test_dataclasses_replace_round_trips_new_context_fields() -> None:
    """The composite enriches via ``dataclasses.replace``; confirm the two new
    fields default to 0.0 and are overwritten correctly."""
    base = _ctx(rollouts_completed=5)
    assert base.rollout_duration_estimate_ms == 0.0
    assert base.window_compute_seconds == 0.0
    enriched = dataclasses.replace(
        base, rollout_duration_estimate_ms=42.0, window_compute_seconds=3.0
    )
    assert enriched.rollout_duration_estimate_ms == 42.0
    assert enriched.window_compute_seconds == 3.0
    assert enriched.rollouts_completed == 5
