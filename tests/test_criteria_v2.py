"""Probe 2 Phase 7 gate test — the three concrete v2 criteria and
``V2_REGISTRY``.

Phase 7 puts the three books on Phase 6's shelf: reflexive attention and
equanimity perturbation recovery (Buddhist phenomenology, active),
second-order volition (Frankfurt, held out). The operational mapping from
framework prose to criterion code is recorded in the Phase 7 entry of
``docs/workingjournal/probe2.md``; this file pins the structural
commitments those criteria carry so a future schema change can't break the
registry silently and a future edit can't drift the registry composition
away from the module constants.
"""

from __future__ import annotations

from kind.mirror.criteria_v2 import (
    EQUANIMITY_PERTURBATION_RECOVERY,
    REFLEXIVE_ATTENTION,
    SECOND_ORDER_VOLITION,
    V2_REGISTRY,
)
from kind.mirror.registry import Criterion, ReadingSurface, SignalMapping, TelemetrySurface


def _resolves(criterion: Criterion) -> None:
    """Re-run construction-time validation on every SignalMapping of the
    criterion — round-tripping through ``model_dump`` / ``model_validate``
    re-runs ``_validate_field_path_resolves``, so a schema field rename or
    surface change that broke the v2 registry trips here even if the
    SignalMapping-model tests didn't notice."""
    for mapping in criterion.signal_mappings:
        rebuilt = SignalMapping.model_validate(mapping.model_dump())
        assert rebuilt == mapping


# ---------------------------------------------------------------------------
# (1)–(3) registry composition: three criteria, two active, one held out
# ---------------------------------------------------------------------------


def test_v2_registry_has_three_criteria() -> None:
    assert len(V2_REGISTRY.criteria) == 3
    assert V2_REGISTRY.all_ids() == frozenset(
        {"reflexive_attention", "equanimity_perturbation_recovery", "second_order_volition"}
    )


def test_v2_registry_active_set_has_two() -> None:
    active = V2_REGISTRY.active()
    assert [c.id for c in active] == [
        "reflexive_attention",
        "equanimity_perturbation_recovery",
    ]
    assert REFLEXIVE_ATTENTION in active
    assert EQUANIMITY_PERTURBATION_RECOVERY in active
    assert SECOND_ORDER_VOLITION not in active


def test_v2_registry_held_out_set_has_one() -> None:
    held = V2_REGISTRY.held_out()
    assert [c.id for c in held] == ["second_order_volition"]
    assert SECOND_ORDER_VOLITION.held_out is True
    assert REFLEXIVE_ATTENTION.held_out is False
    assert EQUANIMITY_PERTURBATION_RECOVERY.held_out is False


# ---------------------------------------------------------------------------
# (4)–(6) each criterion's signal mappings resolve against the schemas
# ---------------------------------------------------------------------------


def test_reflexive_attention_signal_mappings_resolve() -> None:
    c = REFLEXIVE_ATTENTION
    assert c.framework == "buddhist_phenomenology"
    assert c.reading_surfaces == frozenset({ReadingSurface.HEAD_INTERNAL})
    assert c.telemetry_surfaces == frozenset(
        {TelemetrySurface.AGENT_STEP_INTERNAL, TelemetrySurface.DREAM_ROLLOUT}
    )
    by_name = {m.name: m for m in c.signal_mappings}
    assert set(by_name) == {"latent_self_reference_t", "dream_self_reference_t"}
    assert by_name["latent_self_reference_t"].field_path == "h_t"
    assert (
        by_name["latent_self_reference_t"].telemetry_surface
        is TelemetrySurface.AGENT_STEP_INTERNAL
    )
    assert by_name["dream_self_reference_t"].field_path == "sequence_h"
    assert (
        by_name["dream_self_reference_t"].telemetry_surface
        is TelemetrySurface.DREAM_ROLLOUT
    )
    assert c.falsifier.strip()
    _resolves(c)


def test_equanimity_perturbation_recovery_signal_mappings_resolve() -> None:
    c = EQUANIMITY_PERTURBATION_RECOVERY
    assert c.framework == "buddhist_phenomenology"
    assert c.reading_surfaces == frozenset(
        {ReadingSurface.SUBSTRATE_SIDE, ReadingSurface.BEHAVIOR_SIDE}
    )
    assert c.telemetry_surfaces == frozenset(
        {TelemetrySurface.AGENT_STEP_INTERNAL, TelemetrySurface.AGENT_STEP_OBSERVABLE}
    )
    by_name = {m.name: m for m in c.signal_mappings}
    assert set(by_name) == {"recovery_lag_steps", "policy_entropy_t", "posterior_kl_t"}
    assert by_name["recovery_lag_steps"].field_path == "h_t"
    assert (
        by_name["recovery_lag_steps"].telemetry_surface
        is TelemetrySurface.AGENT_STEP_INTERNAL
    )
    assert by_name["policy_entropy_t"].field_path == "policy_entropy_t"
    assert (
        by_name["policy_entropy_t"].telemetry_surface
        is TelemetrySurface.AGENT_STEP_OBSERVABLE
    )
    assert by_name["posterior_kl_t"].field_path == "kl_aggregate_t"
    assert (
        by_name["posterior_kl_t"].telemetry_surface
        is TelemetrySurface.AGENT_STEP_INTERNAL
    )
    assert c.falsifier.strip()
    _resolves(c)


def test_second_order_volition_signal_mappings_resolve() -> None:
    c = SECOND_ORDER_VOLITION
    assert c.framework == "frankfurt"
    assert c.reading_surfaces == frozenset(
        {ReadingSurface.SUBSTRATE_SIDE, ReadingSurface.BEHAVIOR_SIDE}
    )
    assert c.telemetry_surfaces == frozenset(
        {TelemetrySurface.AGENT_STEP_INTERNAL, TelemetrySurface.AGENT_STEP_OBSERVABLE}
    )
    by_name = {m.name: m for m in c.signal_mappings}
    assert set(by_name) == {"policy_modulation_t", "latent_regime_indicator_t"}
    assert by_name["policy_modulation_t"].field_path == "action_t"
    assert (
        by_name["policy_modulation_t"].telemetry_surface
        is TelemetrySurface.AGENT_STEP_OBSERVABLE
    )
    assert by_name["latent_regime_indicator_t"].field_path == "h_t"
    assert (
        by_name["latent_regime_indicator_t"].telemetry_surface
        is TelemetrySurface.AGENT_STEP_INTERNAL
    )
    assert c.falsifier.strip()
    _resolves(c)


# ---------------------------------------------------------------------------
# (7)–(8) by_framework
# ---------------------------------------------------------------------------


def test_v2_registry_by_framework_buddhist() -> None:
    buddhist = V2_REGISTRY.by_framework("buddhist_phenomenology")
    assert [c.id for c in buddhist] == [
        "reflexive_attention",
        "equanimity_perturbation_recovery",
    ]


def test_v2_registry_by_framework_frankfurt() -> None:
    frankfurt = V2_REGISTRY.by_framework("frankfurt")
    assert [c.id for c in frankfurt] == ["second_order_volition"]


# ---------------------------------------------------------------------------
# (9)–(11) by_reading_surface
# ---------------------------------------------------------------------------


def test_v2_registry_by_reading_surface_substrate_side() -> None:
    substrate = V2_REGISTRY.by_reading_surface(ReadingSurface.SUBSTRATE_SIDE)
    assert {c.id for c in substrate} == {
        "equanimity_perturbation_recovery",
        "second_order_volition",
    }


def test_v2_registry_by_reading_surface_head_internal() -> None:
    head_internal = V2_REGISTRY.by_reading_surface(ReadingSurface.HEAD_INTERNAL)
    assert [c.id for c in head_internal] == ["reflexive_attention"]


def test_v2_registry_by_reading_surface_behavior_side() -> None:
    behavior = V2_REGISTRY.by_reading_surface(ReadingSurface.BEHAVIOR_SIDE)
    assert {c.id for c in behavior} == {
        "equanimity_perturbation_recovery",
        "second_order_volition",
    }


# ---------------------------------------------------------------------------
# (12) the registry composition matches the module constants
# ---------------------------------------------------------------------------


def test_v2_registry_ids_match_module_constants() -> None:
    assert V2_REGISTRY.get("reflexive_attention") is REFLEXIVE_ATTENTION
    assert V2_REGISTRY.get("equanimity_perturbation_recovery") is EQUANIMITY_PERTURBATION_RECOVERY
    assert V2_REGISTRY.get("second_order_volition") is SECOND_ORDER_VOLITION
    assert tuple(c.id for c in V2_REGISTRY.criteria) == (
        REFLEXIVE_ATTENTION.id,
        EQUANIMITY_PERTURBATION_RECOVERY.id,
        SECOND_ORDER_VOLITION.id,
    )


# ---------------------------------------------------------------------------
# (12b) Phase 9: each v2 criterion commits a stable falsifier_id
# ---------------------------------------------------------------------------


def test_v2_criteria_have_committed_falsifier_ids() -> None:
    """Phase 9 commits a ``falsifier_id`` per v2 criterion the judge
    layer references in its per-falsifier verdicts. The ``_v1`` suffix
    is for forward versioning if a criterion's falsifier prose is
    amended in a later Phase 7 revision. These values are journaled in
    the Phase 9 entry — any change here is a Phase 7 revision the
    journal records explicitly."""
    assert REFLEXIVE_ATTENTION.falsifier_id == "reflexive_attention_v1"
    assert (
        EQUANIMITY_PERTURBATION_RECOVERY.falsifier_id
        == "equanimity_perturbation_recovery_v1"
    )
    assert SECOND_ORDER_VOLITION.falsifier_id == "second_order_volition_v1"


def test_v2_falsifier_ids_are_unique_across_registry() -> None:
    """No two criteria share a falsifier_id — the judge's
    ``FalsifierVerdict.falsifier_id`` field has to disambiguate
    cross-criterion. Phase 9's verdict-aggregation reads through this
    invariant; if a future Phase 7 revision adds a fourth criterion
    that shares an existing falsifier_id, this trips and the new
    criterion needs its own ``_v1`` (or ``_v2`` if it's a re-versioning
    of an existing one)."""
    ids = [c.falsifier_id for c in V2_REGISTRY.criteria]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# (13) extra: the v2 criteria are frozen — the canonical commitment can't be
#      mutated at runtime
# ---------------------------------------------------------------------------


def test_v2_criteria_are_frozen() -> None:
    import pytest
    from pydantic import ValidationError

    for c in (REFLEXIVE_ATTENTION, EQUANIMITY_PERTURBATION_RECOVERY, SECOND_ORDER_VOLITION):
        with pytest.raises(ValidationError):
            c.held_out = True
    with pytest.raises(ValidationError):
        V2_REGISTRY.criteria = ()
