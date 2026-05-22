"""Probe 2 Phase 7 gate test — the typed ``SignalMapping`` model and the
refactored ``Criterion.signal_mappings``.

Phase 7 refactors ``Criterion.signal_mappings`` from a raw ``dict[str,
str]`` (Phase 6) into a ``tuple[SignalMapping, ...]`` whose field-path
references resolve at construction against the actual telemetry schemas
(:class:`~kind.observer.schemas.AgentStep`,
:class:`~kind.observer.schemas.DreamRollout`,
:class:`~kind.observer.schemas.ReplayMeta`). This file pins the
:class:`~kind.mirror.registry.SignalMapping` contract and the
criterion-level invariants that go with it; ``tests/test_criteria_v2.py``
exercises the three concrete v2 criteria.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kind.mirror.registry import (
    Criterion,
    ReadingSurface,
    SignalMapping,
    TelemetrySurface,
)

# ---------------------------------------------------------------------------
# Structural-no-writer pattern — third instance (Phase 6 had two:
# ``tests.test_criterion_registry._FORBIDDEN_FIELD_NAMES`` and
# ``..._FORBIDDEN_RETURN_SUBSTRINGS``). Kept local to the test file per the
# Phase 7 journal's decision not (yet) to promote it to a shared mirror-side
# helper; documented here as the recurring shape so a future contributor
# recognizes it.
# ---------------------------------------------------------------------------
_FORBIDDEN_WRITER_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "write_surface",
        "write_surfaces",
        "output_surface",
        "output_surfaces",
        "feedback_surface",
        "feedback_surfaces",
        "writer",
        "callback",
        "sink",
        "emit",
        "send",
        "publish",
    }
)


# ---------------------------------------------------------------------------
# Inline fixtures
# ---------------------------------------------------------------------------


def _mapping(
    *,
    name: str = "some_signal",
    description: str = "a within-h_t dependence measure (test fixture)",
    telemetry_surface: TelemetrySurface = TelemetrySurface.AGENT_STEP_INTERNAL,
    field_path: str = "h_t",
    slice_spec: str | None = None,
) -> SignalMapping:
    return SignalMapping(
        name=name,
        description=description,
        telemetry_surface=telemetry_surface,
        field_path=field_path,
        slice_spec=slice_spec,
    )


def _criterion(
    *,
    telemetry_surfaces: frozenset[TelemetrySurface] | None = None,
    signal_mappings: tuple[SignalMapping, ...] | None = None,
) -> Criterion:
    return Criterion(
        id="criterion_one",
        display_name="Criterion One",
        framework="test_framework",
        description="A test criterion for the Phase 7 signal-mapping shape.",
        telemetry_surfaces=telemetry_surfaces
        if telemetry_surfaces is not None
        else frozenset({TelemetrySurface.AGENT_STEP_INTERNAL}),
        reading_surfaces=frozenset({ReadingSurface.HEAD_INTERNAL}),
        falsifier="if the signal does not exceed its shuffled-time control",
        falsifier_id="test_falsifier_v1",
        signal_mappings=signal_mappings
        if signal_mappings is not None
        else (_mapping(),),
    )


# ---------------------------------------------------------------------------
# (1) name regex
# ---------------------------------------------------------------------------


def test_signal_mapping_name_regex_enforced() -> None:
    with pytest.raises(ValidationError):
        _mapping(name="BadName")  # uppercase
    with pytest.raises(ValidationError):
        _mapping(name="1bad")  # leading digit
    with pytest.raises(ValidationError):
        _mapping(name="bad-name")  # hyphen
    with pytest.raises(ValidationError):
        _mapping(name="")  # empty
    long_name = "a" + "_" * 40  # > 40 chars
    assert len(long_name) > 40
    with pytest.raises(ValidationError):
        _mapping(name=long_name)
    # Valid name passes.
    m = _mapping(name="latent_self_reference_t")
    assert m.name == "latent_self_reference_t"


# ---------------------------------------------------------------------------
# (2) description non-empty
# ---------------------------------------------------------------------------


def test_signal_mapping_description_non_empty() -> None:
    with pytest.raises(ValidationError):
        _mapping(description="")
    with pytest.raises(ValidationError):
        _mapping(description="   ")
    # Multi-line free text is fine.
    m = _mapping(description="line one\nline two\n\nline four")
    assert "line four" in m.description


# ---------------------------------------------------------------------------
# (3) slice_spec regex
# ---------------------------------------------------------------------------


def test_signal_mapping_slice_spec_regex() -> None:
    # None means "no slice".
    assert _mapping(slice_spec=None).slice_spec is None
    # Valid NumPy-style slice expressions: digits, colons, commas only.
    for ok in (":32", "64:128", ":", "0,1,2", "16:32,48:64"):
        assert _mapping(slice_spec=ok).slice_spec == ok
    # Invalid: empty string, letters, whitespace, brackets.
    for bad in ("", "abc", ":32x", "1 2", "[:32]", "h_t[:32]"):
        with pytest.raises(ValidationError):
            _mapping(slice_spec=bad)


# ---------------------------------------------------------------------------
# (4) frozen
# ---------------------------------------------------------------------------


def test_signal_mapping_is_frozen() -> None:
    m = _mapping()
    with pytest.raises(ValidationError):
        m.name = "other"
    with pytest.raises(ValidationError):
        m.field_path = "z_t"
    # extra="forbid".
    with pytest.raises(ValidationError):
        SignalMapping(  # type: ignore[call-arg]
            name="x",
            description="y",
            telemetry_surface=TelemetrySurface.AGENT_STEP_INTERNAL,
            field_path="h_t",
            unexpected="z",
        )


# ---------------------------------------------------------------------------
# (5) field_path resolves against AGENT_STEP_INTERNAL
# ---------------------------------------------------------------------------


def test_signal_mapping_field_path_resolves_against_agent_step_internal() -> None:
    for ok in ("h_t", "z_t", "q_params_t", "p_params_t", "kl_aggregate_t"):
        m = _mapping(
            telemetry_surface=TelemetrySurface.AGENT_STEP_INTERNAL, field_path=ok
        )
        assert m.field_path == ok
    # An unknown field name is rejected with a clear error naming it.
    with pytest.raises(ValidationError) as excinfo:
        _mapping(
            telemetry_surface=TelemetrySurface.AGENT_STEP_INTERNAL,
            field_path="nonexistent_field",
        )
    assert "nonexistent_field" in str(excinfo.value)


# ---------------------------------------------------------------------------
# (6) field_path resolves against DREAM_ROLLOUT
# ---------------------------------------------------------------------------


def test_signal_mapping_field_path_resolves_against_dream_rollout() -> None:
    for ok in ("sequence_h", "sequence_z_prior", "seed_h0", "sequence_self_prediction"):
        m = _mapping(
            telemetry_surface=TelemetrySurface.DREAM_ROLLOUT, field_path=ok
        )
        assert m.field_path == ok
    with pytest.raises(ValidationError) as excinfo:
        _mapping(
            telemetry_surface=TelemetrySurface.DREAM_ROLLOUT,
            field_path="h_t",  # AgentStep field, not a DreamRollout field
        )
    assert "h_t" in str(excinfo.value)


# ---------------------------------------------------------------------------
# (7) AGENT_STEP_OBSERVABLE may not declare an internal field path
# ---------------------------------------------------------------------------


def test_signal_mapping_observable_internal_split_enforced() -> None:
    with pytest.raises(ValidationError) as excinfo:
        _mapping(
            telemetry_surface=TelemetrySurface.AGENT_STEP_OBSERVABLE,
            field_path="h_t",
        )
    msg = str(excinfo.value)
    assert "h_t" in msg
    assert "asymmetry-of-access" in msg
    # Other internal fields are likewise rejected on the observable surface.
    for internal_field in ("z_t", "q_params_t", "self_prediction_t", "kl_aggregate_t"):
        with pytest.raises(ValidationError):
            _mapping(
                telemetry_surface=TelemetrySurface.AGENT_STEP_OBSERVABLE,
                field_path=internal_field,
            )
    # A genuinely observable field is fine on the observable surface.
    m = _mapping(
        telemetry_surface=TelemetrySurface.AGENT_STEP_OBSERVABLE,
        field_path="action_t",
    )
    assert m.field_path == "action_t"


# ---------------------------------------------------------------------------
# (8) AGENT_STEP_INTERNAL may not declare an observable field path
# ---------------------------------------------------------------------------


def test_signal_mapping_internal_observable_split_enforced() -> None:
    with pytest.raises(ValidationError) as excinfo:
        _mapping(
            telemetry_surface=TelemetrySurface.AGENT_STEP_INTERNAL,
            field_path="action_t",
        )
    msg = str(excinfo.value)
    assert "action_t" in msg
    assert "asymmetry-of-access" in msg
    for observable_field in (
        "action_logprob_t",
        "policy_entropy_t",
        "obs_hash_t",
        "self_prediction_error_t",
    ):
        with pytest.raises(ValidationError):
            _mapping(
                telemetry_surface=TelemetrySurface.AGENT_STEP_INTERNAL,
                field_path=observable_field,
            )


# ---------------------------------------------------------------------------
# (9) no SignalMapping field is writer-shaped — structural assertion
# ---------------------------------------------------------------------------


def test_signal_mapping_no_writer_shape() -> None:
    """SignalMapping is part of the mirror's read-only data plane: no field
    is a writable handle, callback, sink, or anything invokable against Io's
    input space. This is the same structural pattern as Phase 6's
    ``test_criterion_does_not_have_write_surface_field`` /
    ``test_registry_no_method_returns_writer`` — the third instance."""
    field_names = set(SignalMapping.model_fields.keys())
    write_named = {name for name in field_names if "write" in name.lower()}
    assert write_named == set(), (
        f"SignalMapping has fields with 'write' in the name: {write_named}."
    )
    forbidden_named = field_names & _FORBIDDEN_WRITER_FIELD_NAMES
    assert forbidden_named == set(), (
        f"SignalMapping has forbidden writer-shaped fields: {forbidden_named}."
    )


# ---------------------------------------------------------------------------
# (10) Criterion.signal_mappings non-empty
# ---------------------------------------------------------------------------


def test_criterion_signal_mappings_non_empty() -> None:
    with pytest.raises(ValidationError):
        _criterion(signal_mappings=())


# ---------------------------------------------------------------------------
# (11) Criterion signal-mapping names unique
# ---------------------------------------------------------------------------


def test_criterion_signal_mapping_names_unique() -> None:
    with pytest.raises(ValidationError) as excinfo:
        _criterion(
            signal_mappings=(
                _mapping(name="dup_signal", field_path="h_t"),
                _mapping(name="dup_signal", field_path="z_t"),
            )
        )
    msg = str(excinfo.value)
    assert "dup_signal" in msg
    assert "0" in msg and "1" in msg


# ---------------------------------------------------------------------------
# (12) every SignalMapping surface is in the criterion's telemetry_surfaces
# ---------------------------------------------------------------------------


def test_criterion_signal_mapping_surfaces_subset_of_criterion_surfaces() -> None:
    # The criterion declares only AGENT_STEP_INTERNAL, but a mapping reads
    # from DREAM_ROLLOUT — rejected.
    with pytest.raises(ValidationError) as excinfo:
        _criterion(
            telemetry_surfaces=frozenset({TelemetrySurface.AGENT_STEP_INTERNAL}),
            signal_mappings=(
                _mapping(name="ok_signal", field_path="h_t"),
                _mapping(
                    name="off_surface_signal",
                    telemetry_surface=TelemetrySurface.DREAM_ROLLOUT,
                    field_path="sequence_h",
                ),
            ),
        )
    msg = str(excinfo.value)
    assert "off_surface_signal" in msg
    assert "dream_rollout" in msg
    # Declaring both surfaces makes it valid.
    c = _criterion(
        telemetry_surfaces=frozenset(
            {TelemetrySurface.AGENT_STEP_INTERNAL, TelemetrySurface.DREAM_ROLLOUT}
        ),
        signal_mappings=(
            _mapping(name="ok_signal", field_path="h_t"),
            _mapping(
                name="dream_signal",
                telemetry_surface=TelemetrySurface.DREAM_ROLLOUT,
                field_path="sequence_h",
            ),
        ),
    )
    assert {m.name for m in c.signal_mappings} == {"ok_signal", "dream_signal"}


# ---------------------------------------------------------------------------
# (13) tuple order preserved
# ---------------------------------------------------------------------------


def test_criterion_signal_mappings_preserve_order() -> None:
    c = _criterion(
        signal_mappings=(
            _mapping(name="first", field_path="h_t"),
            _mapping(name="second", field_path="z_t"),
            _mapping(name="third", field_path="kl_aggregate_t"),
        )
    )
    assert [m.name for m in c.signal_mappings] == ["first", "second", "third"]
    assert isinstance(c.signal_mappings, tuple)


# ---------------------------------------------------------------------------
# (14) dotted-path resolution
# ---------------------------------------------------------------------------


def test_signal_mapping_dotted_path_resolves() -> None:
    # A single-component path resolves.
    assert _mapping(field_path="h_t").field_path == "h_t"
    # An unknown component on a surface with no allowlist (DREAM_ROLLOUT) is
    # rejected with an error naming the bad component AND the available
    # fields at that level.
    with pytest.raises(ValidationError) as excinfo:
        _mapping(
            telemetry_surface=TelemetrySurface.DREAM_ROLLOUT,
            field_path="posterior",
        )
    msg = str(excinfo.value)
    assert "posterior" in msg
    assert "sequence_h" in msg  # one of DreamRollout's actual fields, listed
    # A dotted path whose intermediate component is not a nested model (the
    # telemetry schemas have none) is rejected naming that component and the
    # remaining path.
    with pytest.raises(ValidationError) as excinfo:
        _mapping(
            telemetry_surface=TelemetrySurface.AGENT_STEP_INTERNAL,
            field_path="kl_per_dim_t.mean",
        )
    msg = str(excinfo.value)
    assert "kl_per_dim_t" in msg
    assert "mean" in msg


# ---------------------------------------------------------------------------
# (15) the Phase 6 dict[str, str] form is rejected with a migration hint
# ---------------------------------------------------------------------------


def test_criterion_with_old_dict_signal_mappings_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Criterion(
            id="criterion_one",
            display_name="Criterion One",
            framework="test_framework",
            description="A test criterion.",
            telemetry_surfaces=frozenset({TelemetrySurface.AGENT_STEP_INTERNAL}),
            reading_surfaces=frozenset({ReadingSurface.HEAD_INTERNAL}),
            falsifier="if the cited mean drifts > 2σ under shuffle",
            falsifier_id="test_falsifier_v1",
            signal_mappings={  # type: ignore[arg-type]
                "some_signal": "derived from agent_step.kl_aggregate_t"
            },
        )
    msg = str(excinfo.value)
    assert "SignalMapping" in msg
    assert "Phase 7" in msg


# ---------------------------------------------------------------------------
# (16) extra: REPLAY_META field paths resolve; no allowlist applies there
# ---------------------------------------------------------------------------


def test_signal_mapping_field_path_resolves_against_replay_meta() -> None:
    for ok in ("priority", "buffer_size", "segment_id", "event_type"):
        m = _mapping(
            telemetry_surface=TelemetrySurface.REPLAY_META, field_path=ok
        )
        assert m.field_path == ok
    with pytest.raises(ValidationError):
        _mapping(
            telemetry_surface=TelemetrySurface.REPLAY_META, field_path="h_t"
        )


# ---------------------------------------------------------------------------
# (17) extra: slice_spec is stored verbatim for Phase 8 to interpret
# ---------------------------------------------------------------------------


def test_signal_mapping_slice_spec_stored_verbatim() -> None:
    m = _mapping(field_path="h_t", slice_spec="64:128")
    assert m.slice_spec == "64:128"
    # round-trips through model_dump / model_validate
    rebuilt = SignalMapping.model_validate(m.model_dump())
    assert rebuilt == m


# ---------------------------------------------------------------------------
# (18) extra: field_path shape is checked before semantic resolution
# ---------------------------------------------------------------------------


def test_signal_mapping_field_path_shape_rejected() -> None:
    for bad in ("", "h t", "h_t.", ".h_t", "h_t..z_t", "h-t", "h_t[:32]"):
        with pytest.raises(ValidationError):
            _mapping(field_path=bad)
