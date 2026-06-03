"""Phase 8b load-bearing test — the metabolic re-entry decision is content-blind.

The B2 dormant→dreaming re-entry edge must read *nothing Io-state-derived* — the
analog of the Phase 4 ``HostSignals`` type-test and the Phase 6
``DreamSessionContext`` content-blindness test. The re-entry decision's only input
is the typed :class:`~kind.training.protection.MetabolicState`, whose fields are
content-blind primitives. The checker has two belts: a **forbidden-types** belt
(no tensor / dream-content / Io-model type) *and* a **Io-derived-name** belt (a
``float``-typed field named for an Io quantity — ``latent_norm``,
``intrinsic_signal`` — is still forbidden, because type alone cannot catch it).
The meta-test pins the checker by feeding it synthetic dataclasses with injected
Io-derived fields and asserting it trips.

Plus: the decision logic (``has_metabolic_room`` is the exact complement of the
``ComputeBudgetCap``) and the policy plumbing (the stub never re-dreams; the
composite re-enters iff a projected session fits the budget).
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

import torch

from kind.agents.views import PolicyView
from kind.mirror.dream_reading import DreamReading
from kind.mirror.structured import StructuredClaim
from kind.observer.schemas import AgentStep, DreamRollout
from kind.training.protection import (
    DreamProtectionPolicy,
    MetabolicBudget,
    MetabolicState,
    has_metabolic_room,
)
from kind.training.state_machine import DreamEnvelopeConfig, StubRolloutCountProtection

_CONTENT_BLIND_SCALARS = {int, float, bool, str}

_FORBIDDEN_TYPES = {
    torch.Tensor,
    DreamRollout,
    AgentStep,
    PolicyView,
    DreamReading,
    StructuredClaim,
}

# Substrings marking a field name as Io-state-derived even when its type is a
# content-blind scalar (a ``latent_norm: float`` is type-clean but Io-derived).
_IO_DERIVED_NAME_MARKERS = (
    "latent",
    "intrinsic",
    "policy",
    "self_prediction",
    "disagreement",
    "action",
    "reward",
    "h_t",
    "z_t",
    "agent_step",
    "dream_content",
)


def _io_derived_offenders(dc: type) -> list[str]:
    """Return names of any fields that are not content-blind: a forbidden type,
    a non-scalar type, or a content-blind scalar with an Io-derived name."""
    hints = typing.get_type_hints(dc)
    offenders: list[str] = []
    for name, annotation in hints.items():
        if annotation in _FORBIDDEN_TYPES:
            offenders.append(name)
            continue
        if annotation not in _CONTENT_BLIND_SCALARS:
            offenders.append(name)
            continue
        low = name.lower()
        if any(marker in low for marker in _IO_DERIVED_NAME_MARKERS):
            offenders.append(name)
            continue
    return offenders


def test_metabolic_state_is_content_blind() -> None:
    """LOAD-BEARING. Every ``MetabolicState`` field is a content-blind primitive
    with a non-Io-derived name — so the re-entry decision cannot read Io's state
    through it. The structural face of the 8b exogenous-trigger reinterpretation:
    'nothing Io-derived gates dreaming' is unrepresentable to violate."""
    assert _io_derived_offenders(MetabolicState) == []
    hints = typing.get_type_hints(MetabolicState)
    for name, annotation in hints.items():
        assert annotation not in _FORBIDDEN_TYPES, (name, annotation)


def test_reentry_content_blindness_check_trips_on_injected_io_field() -> None:
    """Pin the checker: it must fail on an injected Io-derived field — whether
    type-forbidden (a tensor) or merely Io-named (a ``float``)."""

    @dataclass(frozen=True)
    class _BadWithLatentFloat:
        window_compute_seconds: float
        latent_norm: float  # Io-derived by name, content-blind by type

    @dataclass(frozen=True)
    class _BadWithIntrinsic:
        intrinsic_signal: float

    @dataclass(frozen=True)
    class _BadWithTensor:
        window_compute_seconds: float
        latents: torch.Tensor

    assert _io_derived_offenders(_BadWithLatentFloat) == ["latent_norm"]
    assert _io_derived_offenders(_BadWithIntrinsic) == ["intrinsic_signal"]
    assert _io_derived_offenders(_BadWithTensor) == ["latents"]


def test_has_metabolic_room_is_full_session_token_check() -> None:
    """Re-enter iff the bucket holds a full projected session's worth of tokens —
    the anti-trickle hysteresis (full session, not one rollout)."""
    # Room: 3 tokens ≥ a 2.5 s projected session.
    assert has_metabolic_room(
        MetabolicState(
            tokens=3.0,
            capacity=10.0,
            refill_rate=0.17,
            projected_session_seconds=2.5,
            wallclock_ms=0,
        )
    )
    # No room: 3 tokens < a 3.5 s projected session.
    assert not has_metabolic_room(
        MetabolicState(
            tokens=3.0,
            capacity=10.0,
            refill_rate=0.17,
            projected_session_seconds=3.5,
            wallclock_ms=0,
        )
    )


def test_stub_never_redreams_and_records_nothing() -> None:
    """The stub keeps no ledger: it grants no metabolic re-entry and recording is
    a no-op (single-session-per-absence, the pre-8b behavior)."""
    stub = StubRolloutCountProtection()
    assert stub.metabolic_reentry(DreamEnvelopeConfig(), now_ms=0) is False
    stub.record_session(num_rollouts=3, session_wallclock_ms=100, now_ms=0)  # no raise


def test_composite_reentry_tracks_the_token_bucket() -> None:
    """The composite re-enters while the bucket affords a FULL session, stops once
    spending drains it below one, then continuous refill recovers it — the
    depletion→replenishment metabolic loop end to end (no rolling-window trickle)."""
    budget = MetabolicBudget(
        capacity_seconds=10.0,
        refill_rate=1.0,  # 1 compute-s per wall-s
        seed_rollout_duration_ms=1000.0,
        initial_tokens=10.0,
    )
    composite = DreamProtectionPolicy(budget=budget)
    env = DreamEnvelopeConfig(hard_cap_rollout_count=3)  # full session ≈ 3 × 1 s = 3 s

    # Full bucket affords a 3 s session.
    assert composite.metabolic_reentry(env, now_ms=0) is True
    # Spend three 3 s sessions at the same instant (no refill): 10 → 7 → 4 → 1.
    composite.record_session(num_rollouts=3, session_wallclock_ms=3000, now_ms=0)
    assert composite.metabolic_reentry(env, now_ms=0) is True  # 7 ≥ 3
    composite.record_session(num_rollouts=3, session_wallclock_ms=3000, now_ms=0)
    assert composite.metabolic_reentry(env, now_ms=0) is True  # 4 ≥ 3
    composite.record_session(num_rollouts=3, session_wallclock_ms=3000, now_ms=0)
    assert composite.metabolic_reentry(env, now_ms=0) is False  # 1 < 3 → rest
    # Continuous refill (vs the old all-at-once-an-hour-later return) recovers:
    # +1 token/s, so 2 s later tokens = 3 → affordable again.
    assert composite.metabolic_reentry(env, now_ms=2000) is True
