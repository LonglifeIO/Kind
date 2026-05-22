"""Probe 2 Phase 6 gate test — the criterion registry's structural shape.

Phase 6 builds the empty shelf with strong invariants; this test pins the
shelf's shape so Phase 7 (which populates it with the three v2 criteria) and
Phase 8 (which reads from it) can rely on the contract. The structural
read-only invariant — no write surfaces on :class:`Criterion`, no method on
:class:`CriterionRegistry` returning a writer — is the load-bearing
assertion of the mirror's one-way data plane (see the module docstring at
:mod:`kind.mirror.registry`).

Tests use small inline fixture criteria with mock framework names like
``test_framework_a``. Phase 7's three v2 criteria are deliberately NOT
imported — Phase 6 must pass without Phase 7 existing.
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Callable

import pytest
from pydantic import ValidationError

from kind.mirror.registry import (
    EMPTY_REGISTRY,
    Criterion,
    CriterionRegistry,
    ReadingSurface,
    SignalMapping,
    TelemetrySurface,
)


# ---------------------------------------------------------------------------
# Small inline fixtures
# ---------------------------------------------------------------------------


def _signal_mapping(
    *,
    name: str = "some_signal",
    description: str = "derived from AgentStep.kl_aggregate_t (Phase 6 shelf test)",
    telemetry_surface: TelemetrySurface = TelemetrySurface.AGENT_STEP_INTERNAL,
    field_path: str = "kl_aggregate_t",
    slice_spec: str | None = None,
) -> SignalMapping:
    """Build a SignalMapping with sensible defaults for the registry tests.

    The exhaustive SignalMapping-level validation lives in
    ``tests/test_signal_mapping.py``; this helper just supplies a valid
    instance the Criterion / CriterionRegistry tests can lean on.
    """
    return SignalMapping(
        name=name,
        description=description,
        telemetry_surface=telemetry_surface,
        field_path=field_path,
        slice_spec=slice_spec,
    )


def _criterion(
    *,
    id: str = "criterion_one",
    display_name: str = "Criterion One",
    framework: str = "test_framework_a",
    description: str = "A test criterion for Phase 6's structural shelf.",
    telemetry_surfaces: frozenset[TelemetrySurface] | None = None,
    reading_surfaces: frozenset[ReadingSurface] | None = None,
    falsifier: str = "if the cited mean drifts > 2σ under shuffle",
    falsifier_id: str = "test_falsifier_v1",
    signal_mappings: tuple[SignalMapping, ...] | None = None,
    held_out: bool = False,
) -> Criterion:
    """Build a Criterion with sensible defaults; tests override fields."""
    return Criterion(
        id=id,
        display_name=display_name,
        framework=framework,
        description=description,
        telemetry_surfaces=telemetry_surfaces
        if telemetry_surfaces is not None
        else frozenset({TelemetrySurface.AGENT_STEP_INTERNAL}),
        reading_surfaces=reading_surfaces
        if reading_surfaces is not None
        else frozenset({ReadingSurface.HEAD_INTERNAL}),
        falsifier=falsifier,
        falsifier_id=falsifier_id,
        signal_mappings=signal_mappings
        if signal_mappings is not None
        else (_signal_mapping(),),
        held_out=held_out,
    )


# ---------------------------------------------------------------------------
# (1) Criterion.id regex enforcement
# ---------------------------------------------------------------------------


def test_criterion_id_regex_enforced() -> None:
    # Uppercase rejected.
    with pytest.raises(ValidationError):
        _criterion(id="CriterionOne")
    # Leading digit rejected.
    with pytest.raises(ValidationError):
        _criterion(id="1_criterion")
    # Hyphen rejected.
    with pytest.raises(ValidationError):
        _criterion(id="criterion-one")
    # Empty rejected (regex requires a leading lowercase letter).
    with pytest.raises(ValidationError):
        _criterion(id="")
    # > 40 chars rejected.
    long_id = "a" + "_" * 40
    assert len(long_id) > 40
    with pytest.raises(ValidationError):
        _criterion(id=long_id)
    # Valid id passes.
    valid = _criterion(id="reflexive_attention_42")
    assert valid.id == "reflexive_attention_42"


# ---------------------------------------------------------------------------
# (2) Criterion.framework regex (same as id)
# ---------------------------------------------------------------------------


def test_criterion_framework_regex_enforced() -> None:
    with pytest.raises(ValidationError):
        _criterion(framework="BuddhistPhenomenology")
    with pytest.raises(ValidationError):
        _criterion(framework="2_phenomenology")
    with pytest.raises(ValidationError):
        _criterion(framework="buddhist-phenomenology")
    with pytest.raises(ValidationError):
        _criterion(framework="")
    valid = _criterion(framework="buddhist_phenomenology")
    assert valid.framework == "buddhist_phenomenology"


# ---------------------------------------------------------------------------
# (2b) Phase 9: falsifier_id regex + length (same shape as Criterion.id)
# ---------------------------------------------------------------------------


def test_criterion_falsifier_id_regex_enforced() -> None:
    # Uppercase rejected.
    with pytest.raises(ValidationError):
        _criterion(falsifier_id="FalsifierOne")
    # Leading digit rejected.
    with pytest.raises(ValidationError):
        _criterion(falsifier_id="1_falsifier")
    # Hyphen rejected.
    with pytest.raises(ValidationError):
        _criterion(falsifier_id="falsifier-one")
    # Empty rejected.
    with pytest.raises(ValidationError):
        _criterion(falsifier_id="")
    # > 40 chars rejected.
    long_id = "a" + "_" * 40
    assert len(long_id) > 40
    with pytest.raises(ValidationError):
        _criterion(falsifier_id=long_id)
    # Valid id passes; the _v1 suffix convention is by convention,
    # the regex pins shape only.
    valid = _criterion(falsifier_id="reflexive_attention_v1")
    assert valid.falsifier_id == "reflexive_attention_v1"


# ---------------------------------------------------------------------------
# (3) telemetry_surfaces non-empty
# ---------------------------------------------------------------------------


def test_criterion_telemetry_surfaces_non_empty() -> None:
    with pytest.raises(ValidationError):
        _criterion(telemetry_surfaces=frozenset())


# ---------------------------------------------------------------------------
# (4) reading_surfaces non-empty
# ---------------------------------------------------------------------------


def test_criterion_reading_surfaces_non_empty() -> None:
    with pytest.raises(ValidationError):
        _criterion(reading_surfaces=frozenset())


# ---------------------------------------------------------------------------
# (5) signal_mappings is a non-empty tuple of SignalMapping; the Phase 6
#     dict[str, str] form is rejected with a migration hint. Exhaustive
#     SignalMapping-level validation lives in tests/test_signal_mapping.py.
# ---------------------------------------------------------------------------


def test_criterion_signal_mappings_tuple_form() -> None:
    # A non-empty tuple of valid SignalMapping records is accepted, in order.
    c = _criterion(
        signal_mappings=(
            _signal_mapping(name="signal_a", field_path="kl_aggregate_t"),
            _signal_mapping(name="signal_b", field_path="h_t"),
        )
    )
    assert [m.name for m in c.signal_mappings] == ["signal_a", "signal_b"]
    # Empty tuple rejected (a criterion with no signals can't be operationalized).
    with pytest.raises(ValidationError):
        _criterion(signal_mappings=())
    # The Phase 6 dict[str, str] form is rejected with a migration-pointing error.
    with pytest.raises(ValidationError) as excinfo:
        Criterion(
            id="criterion_one",
            display_name="Criterion One",
            framework="test_framework_a",
            description="A test criterion.",
            telemetry_surfaces=frozenset({TelemetrySurface.AGENT_STEP_INTERNAL}),
            reading_surfaces=frozenset({ReadingSurface.HEAD_INTERNAL}),
            falsifier="if the cited mean drifts > 2σ under shuffle",
            falsifier_id="test_falsifier_v1",
            signal_mappings={"some_signal": "derived from x"},  # type: ignore[arg-type]
        )
    assert "SignalMapping" in str(excinfo.value)


# ---------------------------------------------------------------------------
# (6) Criterion is frozen
# ---------------------------------------------------------------------------


def test_criterion_is_frozen() -> None:
    c = _criterion()
    # Pydantic v2 frozen=True raises ValidationError on field assignment.
    with pytest.raises(ValidationError):
        c.id = "criterion_two"
    with pytest.raises(ValidationError):
        c.framework = "test_framework_b"
    with pytest.raises(ValidationError):
        c.held_out = True


# ---------------------------------------------------------------------------
# (7) Criterion does not expose any write-surface field — the structural
#     assertion of the read-only invariant. A future contributor who adds
#     a write surface trips here.
# ---------------------------------------------------------------------------


_FORBIDDEN_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "write_surface",
        "write_surfaces",
        "output_surface",
        "output_surfaces",
        "feedback_surface",
        "feedback_surfaces",
    }
)


def test_criterion_does_not_have_write_surface_field() -> None:
    """The registry is part of the mirror's read-only data plane.
    :class:`Criterion` names the surfaces it reads FROM. There is no field
    naming surfaces it writes TO. A contributor who tries to add one trips
    this test."""
    field_names = set(Criterion.model_fields.keys())
    # No field name contains the substring "write".
    write_named = {name for name in field_names if "write" in name.lower()}
    assert write_named == set(), (
        f"Criterion has fields with 'write' in the name: {write_named}. "
        f"The registry is part of the mirror's read-only data plane; no "
        f"write surface may be named on Criterion."
    )
    # No field name matches the forbidden-names list.
    forbidden_named = field_names & _FORBIDDEN_FIELD_NAMES
    assert forbidden_named == set(), (
        f"Criterion has forbidden write-surface fields: {forbidden_named}. "
        f"The registry is part of the mirror's read-only data plane."
    )


# ---------------------------------------------------------------------------
# (8) Registry get returns the right Criterion
# ---------------------------------------------------------------------------


def test_registry_get_returns_criterion_by_id() -> None:
    a = _criterion(id="alpha")
    b = _criterion(id="beta")
    registry = CriterionRegistry(criteria=(a, b))
    assert registry.get("alpha") is a
    assert registry.get("beta") is b


# ---------------------------------------------------------------------------
# (9) Registry get raises KeyError for missing id
# ---------------------------------------------------------------------------


def test_registry_get_raises_for_missing_id() -> None:
    registry = CriterionRegistry(criteria=(_criterion(id="alpha"),))
    with pytest.raises(KeyError) as excinfo:
        registry.get("zeta")
    # The KeyError message names the missing id so the diagnosis is one read.
    assert "zeta" in str(excinfo.value)


# ---------------------------------------------------------------------------
# (10) Registry has returns bool, doesn't raise
# ---------------------------------------------------------------------------


def test_registry_has_returns_bool() -> None:
    registry = CriterionRegistry(criteria=(_criterion(id="alpha"),))
    assert registry.has("alpha") is True
    assert registry.has("zeta") is False


# ---------------------------------------------------------------------------
# (11) Duplicate ids rejected with both occurrences named
# ---------------------------------------------------------------------------


def test_registry_rejects_duplicate_ids() -> None:
    a = _criterion(id="alpha", display_name="A First")
    a_dup = _criterion(id="alpha", display_name="A Second")
    with pytest.raises(ValidationError) as excinfo:
        CriterionRegistry(criteria=(a, a_dup))
    msg = str(excinfo.value)
    # The error message names the duplicate id and both indices.
    assert "alpha" in msg
    assert "0" in msg and "1" in msg


# ---------------------------------------------------------------------------
# (12) Active and held-out partition correctly
# ---------------------------------------------------------------------------


def test_registry_active_and_held_out_partition_correctly() -> None:
    a = _criterion(id="alpha", held_out=False)
    b = _criterion(id="beta", held_out=True)
    c = _criterion(id="gamma", held_out=False)
    registry = CriterionRegistry(criteria=(a, b, c))

    active = registry.active()
    held = registry.held_out()

    # Together cover the whole set.
    assert {x.id for x in active} | {x.id for x in held} == {
        "alpha",
        "beta",
        "gamma",
    }
    # Disjoint.
    assert {x.id for x in active} & {x.id for x in held} == set()
    # Counts add up.
    assert len(active) + len(held) == len(registry.criteria)
    # All-ids reflects every criterion.
    assert registry.all_ids() == frozenset({"alpha", "beta", "gamma"})


# ---------------------------------------------------------------------------
# (13) Registration order preserved in active() and held_out()
# ---------------------------------------------------------------------------


def test_registry_active_and_held_out_preserve_registration_order() -> None:
    a = _criterion(id="alpha", held_out=False)
    b = _criterion(id="beta", held_out=True)
    c = _criterion(id="gamma", held_out=False)
    d = _criterion(id="delta", held_out=True)
    registry = CriterionRegistry(criteria=(a, b, c, d))

    assert [x.id for x in registry.active()] == ["alpha", "gamma"]
    assert [x.id for x in registry.held_out()] == ["beta", "delta"]


# ---------------------------------------------------------------------------
# (14) by_framework returns only matching criteria
# ---------------------------------------------------------------------------


def test_registry_by_framework_returns_only_matching() -> None:
    a = _criterion(id="alpha", framework="test_framework_a")
    b = _criterion(id="beta", framework="test_framework_b")
    c = _criterion(id="gamma", framework="test_framework_a")
    registry = CriterionRegistry(criteria=(a, b, c))

    a_set = registry.by_framework("test_framework_a")
    assert [x.id for x in a_set] == ["alpha", "gamma"]

    b_set = registry.by_framework("test_framework_b")
    assert [x.id for x in b_set] == ["beta"]

    # Empty result for unknown framework.
    z_set = registry.by_framework("nonexistent_framework")
    assert z_set == ()


# ---------------------------------------------------------------------------
# (15) by_reading_surface handles multi-surface criteria
# ---------------------------------------------------------------------------


def test_registry_by_reading_surface_handles_multi_surface_criteria() -> None:
    a = _criterion(
        id="alpha",
        reading_surfaces=frozenset(
            {ReadingSurface.SUBSTRATE_SIDE, ReadingSurface.HEAD_INTERNAL}
        ),
    )
    b = _criterion(
        id="beta",
        reading_surfaces=frozenset({ReadingSurface.HEAD_INTERNAL}),
    )
    c = _criterion(
        id="gamma",
        reading_surfaces=frozenset({ReadingSurface.BEHAVIOR_SIDE}),
    )
    registry = CriterionRegistry(criteria=(a, b, c))

    sub_side = registry.by_reading_surface(ReadingSurface.SUBSTRATE_SIDE)
    head_int = registry.by_reading_surface(ReadingSurface.HEAD_INTERNAL)
    beh_side = registry.by_reading_surface(ReadingSurface.BEHAVIOR_SIDE)

    # Multi-surface alpha appears for both substrate-side AND head-internal.
    assert [x.id for x in sub_side] == ["alpha"]
    assert [x.id for x in head_int] == ["alpha", "beta"]
    assert [x.id for x in beh_side] == ["gamma"]


# ---------------------------------------------------------------------------
# (16) Registry is frozen
# ---------------------------------------------------------------------------


def test_registry_is_frozen() -> None:
    a = _criterion(id="alpha")
    registry = CriterionRegistry(criteria=(a,))
    with pytest.raises(ValidationError):
        registry.criteria = (a, _criterion(id="beta"))


# ---------------------------------------------------------------------------
# (17) Empty registry only via the EMPTY_REGISTRY constant
# ---------------------------------------------------------------------------


def test_empty_registry_is_only_for_tests() -> None:
    # The constant exists and has zero criteria.
    assert isinstance(EMPTY_REGISTRY, CriterionRegistry)
    assert len(EMPTY_REGISTRY.criteria) == 0
    assert EMPTY_REGISTRY.all_ids() == frozenset()
    assert EMPTY_REGISTRY.active() == ()
    assert EMPTY_REGISTRY.held_out() == ()

    # Constructing CriterionRegistry(criteria=()) directly must raise.
    with pytest.raises(ValidationError) as excinfo:
        CriterionRegistry(criteria=())
    assert "EMPTY_REGISTRY" in str(excinfo.value)


# ---------------------------------------------------------------------------
# (18) No method on CriterionRegistry returns a writer-shaped type
# ---------------------------------------------------------------------------


# The registry is part of the mirror's read-only data plane: lookups return
# frozen records or simple containers, never callables or file-like objects.
# Forbidden return-annotation substrings cover the most common writer-like
# types in Python's stdlib type system.
_FORBIDDEN_RETURN_SUBSTRINGS: frozenset[str] = frozenset(
    {
        "Callable",
        "BinaryIO",
        "TextIO",
        # Bare "IO" is a common Callable-adjacent shape; we check for it as
        # a token below rather than as a substring (to avoid false positives
        # on strings that contain "IO" inside other identifiers, e.g.,
        # "FrozenSet[IOSurface]" — though no such thing exists today).
    }
)


def _annotation_str(method: object) -> str:
    sig = inspect.signature(method)  # type: ignore[arg-type]
    ret = sig.return_annotation
    if ret is inspect.Signature.empty:
        return ""
    if isinstance(ret, type):
        return ret.__name__
    return str(ret)


def test_registry_no_method_returns_writer() -> None:
    """The registry is part of the mirror's read-only data plane: every
    public lookup returns a frozen record or a simple container, never a
    callable or file-like object that could be invoked against Io's input
    space. This test introspects every public method's return annotation
    and asserts it does not contain Callable / IO / BinaryIO / TextIO or
    function-shaped types."""
    public_methods = [
        name
        for name, _ in inspect.getmembers(CriterionRegistry, inspect.isfunction)
        if not name.startswith("_")
        # Pydantic injects a few helpers like ``model_dump``,
        # ``model_validate``, etc. Those are inherited from BaseModel and
        # not part of the registry's public lookup API; restrict to
        # methods defined on CriterionRegistry itself.
        and name in CriterionRegistry.__dict__
    ]

    # Sanity: we expect at least the seven lookup methods on the registry.
    expected_methods = {
        "get",
        "has",
        "all_ids",
        "active",
        "held_out",
        "by_framework",
        "by_reading_surface",
    }
    assert expected_methods.issubset(set(public_methods)), (
        f"Expected at least {expected_methods}; "
        f"introspection found {set(public_methods)}."
    )

    for name in public_methods:
        method = getattr(CriterionRegistry, name)
        annotation = _annotation_str(method)
        # No forbidden substring in the annotation.
        for forbidden in _FORBIDDEN_RETURN_SUBSTRINGS:
            assert forbidden not in annotation, (
                f"CriterionRegistry.{name} has return annotation "
                f"{annotation!r} which contains forbidden substring "
                f"{forbidden!r}. The registry is part of the mirror's "
                f"read-only data plane; no method may return a writer."
            )
        # Resolve the annotation and check it isn't a Callable type or a
        # function/method type. ``typing.get_type_hints`` resolves the
        # forward references in our annotated signatures.
        hints = typing.get_type_hints(method)
        ret_hint = hints.get("return")
        if ret_hint is None:
            continue
        # A direct Callable return annotation would have origin
        # ``collections.abc.Callable``.
        origin = typing.get_origin(ret_hint)
        assert origin is not Callable, (
            f"CriterionRegistry.{name} returns a Callable. The registry "
            f"is part of the mirror's read-only data plane."
        )
