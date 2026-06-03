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
    MetabolicState,
    RollingComputeLedger,
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


def test_has_metabolic_room_is_the_budget_cap_complement() -> None:
    """Re-enter iff ``window + projected <= budget`` — the exact complement of the
    ComputeBudgetCap's stop condition (``window + projected > budget``)."""
    # Room: 8 s window + 1 s projected = 9 ≤ 10.
    assert has_metabolic_room(
        MetabolicState(
            window_compute_seconds=8.0,
            projected_session_seconds=1.0,
            compute_budget_seconds_per_hour=10.0,
            wallclock_ms=0,
        )
    )
    # No room: 9.5 + 1.0 = 10.5 > 10.
    assert not has_metabolic_room(
        MetabolicState(
            window_compute_seconds=9.5,
            projected_session_seconds=1.0,
            compute_budget_seconds_per_hour=10.0,
            wallclock_ms=0,
        )
    )


def test_stub_never_redreams_and_records_nothing() -> None:
    """The stub keeps no ledger: it grants no metabolic re-entry and recording is
    a no-op (single-session-per-absence, the pre-8b behavior)."""
    stub = StubRolloutCountProtection()
    assert stub.metabolic_reentry(DreamEnvelopeConfig(), now_ms=0) is False
    stub.record_session(num_rollouts=3, session_wallclock_ms=100, now_ms=0)  # no raise


def test_composite_reentry_tracks_the_ledger() -> None:
    """The composite re-enters while the rolling window has room and stops once a
    recorded session fills it past the budget — the metabolic loop end to end."""
    clock = {"t": 0}
    ledger = RollingComputeLedger(
        default_rollout_duration_ms=15.0, clock=lambda: clock["t"]
    )
    composite = DreamProtectionPolicy(ledger=ledger)
    env = DreamEnvelopeConfig(compute_budget_seconds_per_hour=0.25)  # 250 ms budget

    # Cold ledger: window 0 + ~15 ms projected ≤ 250 ms → room.
    assert composite.metabolic_reentry(env, now_ms=0) is True
    # Record sessions of 100 ms each; after the window passes the budget, no room.
    composite.record_session(num_rollouts=3, session_wallclock_ms=100, now_ms=0)
    assert composite.metabolic_reentry(env, now_ms=0) is True  # 100 + 33 ≤ 250
    composite.record_session(num_rollouts=3, session_wallclock_ms=100, now_ms=0)
    composite.record_session(num_rollouts=3, session_wallclock_ms=100, now_ms=0)
    # Window now 300 ms > 250 ms budget → no room → rest.
    assert composite.metabolic_reentry(env, now_ms=0) is False
