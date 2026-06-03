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
    DreamProtectionPolicy,
    MetabolicBudget,
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
        DreamProtectionPolicy(),
    ):
        sig = inspect.signature(policy.should_stop)
        params = [p for p in sig.parameters if p != "self"]
        assert params == ["ctx"], f"{type(policy).__name__}.should_stop params {params}"
        hints = typing.get_type_hints(policy.should_stop)
        assert hints["ctx"] is DreamSessionContext


def test_budget_is_content_blind() -> None:
    """The metabolic token bucket records and exposes only numbers — durations,
    counts, tokens, timestamps — never any dream content. No bucket method touches
    a forbidden Io/content type."""
    for method_name in (
        "record_session",
        "current_tokens",
        "rollout_duration_estimate_ms",
        "affords_session",
    ):
        method = getattr(MetabolicBudget, method_name)
        hints = typing.get_type_hints(method)
        for name, annotation in hints.items():
            assert annotation not in _FORBIDDEN_CONTENT_TYPES, (
                f"MetabolicBudget.{method_name} param/return {name} is a "
                f"content/Io type {annotation!r}."
            )
    # Recorded state is numeric, not content.
    budget = MetabolicBudget(capacity_seconds=10.0, refill_rate=1.0)
    budget.record_session(session_wallclock_ms=123.0, num_rollouts=1, now_ms=0)
    assert isinstance(budget.current_tokens(now_ms=0), float)


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


# ---------------------------------------------------------------------------
# Composition — earliest cap wins; rollout-count is the absolute ceiling.
# (The compute-budget cap is gone — cross-session pacing is the token bucket's
# job now, not a per-session cap.)
# ---------------------------------------------------------------------------


def test_composition_rollout_count_ceiling_bounds_a_bad_wallclock_estimate() -> None:
    """A pathologically small rollout-duration estimate would defer the
    wallclock cap far past any sane count; the rollout-count ceiling must still
    bound the session."""
    # estimate 1 ms ⇒ wallclock projection reaches the 30-min budget only at
    # ~1.8M rollouts — far past the ceiling.
    budget = MetabolicBudget(seed_rollout_duration_ms=1.0)
    composite = DreamProtectionPolicy(budget=budget)
    env = DreamEnvelopeConfig(
        hard_cap_rollout_count=50, hard_cap_wallclock_ms=30 * 60 * 1000
    )
    count, trigger = _plan_session_rollouts(composite, env)
    assert (count, trigger) == (50, "hard_cap_rollout_count")


def test_composition_wallclock_wins_when_estimate_is_large() -> None:
    """When rollouts are slow (large estimate), the wallclock backstop fires
    before the rollout-count ceiling — the earliest cap wins."""
    budget = MetabolicBudget(seed_rollout_duration_ms=60_000.0)  # 60 s/rollout
    composite = DreamProtectionPolicy(budget=budget)
    env = DreamEnvelopeConfig(
        hard_cap_rollout_count=50,
        hard_cap_wallclock_ms=30 * 60 * 1000,
    )  # 30 min ⇒ projection hits the budget at 30 rollouts < 50
    count, trigger = _plan_session_rollouts(composite, env)
    assert (count, trigger) == (30, "hard_cap_wallclock")


# ---------------------------------------------------------------------------
# The metabolic token bucket — continuous refill, spend, content-blind estimate.
# ---------------------------------------------------------------------------


def test_budget_refills_continuously_toward_capacity() -> None:
    budget = MetabolicBudget(capacity_seconds=10.0, refill_rate=1.0, initial_tokens=0.0)
    # First observation sets the baseline (no refill yet).
    assert budget.current_tokens(now_ms=0) == 0.0
    # +3 s of wall-time at 1 token/s ⇒ +3 tokens.
    assert budget.current_tokens(now_ms=3000) == 3.0
    # Clamped to capacity, never above.
    assert budget.current_tokens(now_ms=100_000) == 10.0


def test_budget_spend_drains_and_refill_recovers() -> None:
    budget = MetabolicBudget(capacity_seconds=10.0, refill_rate=1.0)
    assert budget.current_tokens(now_ms=0) == 10.0  # starts full
    budget.record_session(session_wallclock_ms=5000.0, num_rollouts=3, now_ms=0)
    assert budget.current_tokens(now_ms=0) == 5.0  # spent 5 s of compute
    assert budget.current_tokens(now_ms=2000) == 7.0  # +2 s refill


def test_budget_estimate_is_all_time_average() -> None:
    budget = MetabolicBudget(seed_rollout_duration_ms=1000.0)
    # Cold bucket returns the seed default.
    assert budget.rollout_duration_estimate_ms() == 1000.0
    budget.record_session(session_wallclock_ms=600.0, num_rollouts=2, now_ms=0)
    budget.record_session(session_wallclock_ms=600.0, num_rollouts=2, now_ms=0)
    # All-time per-rollout average: (600 + 600) / (2 + 2) = 300.
    assert budget.rollout_duration_estimate_ms() == 300.0


def test_budget_affords_full_session_is_the_reentry_hysteresis() -> None:
    """Affordability is full-session, not one-rollout — the anti-trickle
    hysteresis."""
    budget = MetabolicBudget(capacity_seconds=10.0, refill_rate=0.0, initial_tokens=3.0)
    assert budget.affords_session(2.5, now_ms=0) is True  # 3 ≥ 2.5
    assert budget.affords_session(3.5, now_ms=0) is False  # 3 < 3.5


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
    budget = MetabolicBudget(seed_rollout_duration_ms=60_000.0)
    composite = DreamProtectionPolicy(budget=budget)
    env = DreamEnvelopeConfig(
        hard_cap_rollout_count=50,
        hard_cap_wallclock_ms=30 * 60 * 1000,
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
    # Phase 8 token-bucket pacer (superseding 8b's compute_budget_seconds_per_hour):
    # C bounds the burst, R (~17% duty) is the long-run dream/rest ratio. Both are
    # the builder's rhythm knobs.
    assert env.metabolic_capacity_seconds == 150.0
    assert env.metabolic_refill_rate == 0.17
    # The seed is the MPS (deployment-device) per-rollout cost (~108 ms on the
    # Probe-1.5 checkpoint — ~9.5× the CPU figure), rounded up to 110.0.
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
