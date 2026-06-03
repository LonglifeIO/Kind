"""Phase 7 gate tests — the cross-probe surface (plan §4 Phase 7 row).

The load-bearing test is the **no-hidden-state-write restriction**: the typed
perturbation delta can address *only* dream-envelope + seed-selection fields, and
a delta carrying any hidden-state key (a latent/tensor field, ``h`` / ``z``,
weights, an actor/policy field, dream-internal-state) is **rejected at parse
time** rather than silently applied — the way Phase 6's content-blindness checker
is pinned with a positive control. It is sanity-checked to genuinely trip: a
delta with a forbidden key raises; the same delta without it applies. The point
is that a hidden-state-targeting perturbation is *unrepresentable*, not merely
discouraged (synthesis §8: "no hidden-state writes").

The structural tests: the surface round-trips a valid envelope/seed-selection
delta (only the named fields change); the application is staged and takes effect
at a simulated checkpoint boundary, not live mid-state; and the new
``RunnerConfig`` fields are additive and inert (the waking loop is unchanged).
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest
from pydantic import ValidationError

from kind.training.cross_probe_surface import (
    DreamEnvelopeDelta,
    DreamPerturbation,
    SeedSelectionDelta,
    apply_at_checkpoint_boundary,
    stage_perturbation,
)
from kind.training.dream_seed import SeedSelectionConfig
from kind.training.runner import RunnerConfig
from kind.training.state_machine import DreamEnvelopeConfig

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# The no-hidden-state-write restriction (LOAD-BEARING).
# ---------------------------------------------------------------------------

# Keys that target Io's interior (the data plane) — none of them is an envelope
# or seed-selection field, so none is addressable. Each must be rejected.
_HIDDEN_STATE_KEYS = [
    "h",
    "z",
    "h_t",
    "z_t",
    "latents",
    "weights",
    "actor",
    "policy",
    "dream_internal_state",
    "hidden_state",
    "world_model",
]


@pytest.mark.parametrize("forbidden_key", _HIDDEN_STATE_KEYS)
def test_top_level_hidden_state_key_is_rejected(forbidden_key: str) -> None:
    """LOAD-BEARING. A perturbation carrying a top-level hidden-state key is
    rejected at parse time. The delta model has no field that addresses Io's
    interior and forbids extras, so the key is unrepresentable."""
    delta = f'{{"dream_envelope": {{"hard_cap_rollout_count": 30}}, "{forbidden_key}": 1}}'
    with pytest.raises(ValidationError):
        stage_perturbation(delta)


@pytest.mark.parametrize("forbidden_key", _HIDDEN_STATE_KEYS)
def test_nested_hidden_state_key_is_rejected(forbidden_key: str) -> None:
    """LOAD-BEARING. A hidden-state key smuggled *inside* either sub-surface is
    rejected too — the nested deltas forbid extras, so the interior cannot be
    addressed through the envelope or the seed-selection object either."""
    in_envelope = (
        f'{{"dream_envelope": {{"hard_cap_rollout_count": 30, "{forbidden_key}": 1}}}}'
    )
    in_seed = f'{{"seed_selection": {{"mode": "hybrid", "{forbidden_key}": 1}}}}'
    with pytest.raises(ValidationError):
        stage_perturbation(in_envelope)
    with pytest.raises(ValidationError):
        stage_perturbation(in_seed)


def test_restriction_trips_genuinely_positive_control() -> None:
    """Sanity-check the load-bearing guard: a delta with a forbidden hidden-state
    key raises, and the *same* delta without it parses and applies. The guard is
    only meaningful if it can both fail (forbidden) and pass (allowed)."""
    forbidden = (
        '{"dream_envelope": {"hard_cap_rollout_count": 30}, "latents": [0.1, 0.2]}'
    )
    allowed = '{"dream_envelope": {"hard_cap_rollout_count": 30}}'

    with pytest.raises(ValidationError):
        stage_perturbation(forbidden)

    # The same delta, hidden-state key removed, applies cleanly.
    perturbation = stage_perturbation(allowed)
    result = apply_at_checkpoint_boundary(
        perturbation,
        dream_envelope=DreamEnvelopeConfig(),
        seed_selection=SeedSelectionConfig(),
    )
    assert result.dream_envelope.hard_cap_rollout_count == 30


def test_invalid_enum_value_is_rejected() -> None:
    """A bonus the Literal types buy: a seed-selection field with an invalid enum
    value (e.g. an unauthorized ``mode``) is rejected at parse time too."""
    with pytest.raises(ValidationError):
        stage_perturbation('{"seed_selection": {"mode": "lucid"}}')


def test_delta_addresses_exactly_envelope_and_seed_fields() -> None:
    """The delta surface is *exactly* the two config surfaces — no more. The
    envelope/seed deltas declare exactly their config's fields, and the
    perturbation declares exactly the two sub-surfaces. This pins "address only":
    there is no addressable field beyond the envelope and seed-selection, and a
    future config field would have to be added to the delta deliberately (it
    cannot leak in, because extras are forbidden)."""
    envelope_fields = {f.name for f in dataclasses.fields(DreamEnvelopeConfig)}
    seed_fields = {f.name for f in dataclasses.fields(SeedSelectionConfig)}

    assert set(DreamEnvelopeDelta.model_fields) == envelope_fields
    assert set(SeedSelectionDelta.model_fields) == seed_fields
    assert set(DreamPerturbation.model_fields) == {"dream_envelope", "seed_selection"}


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    return imported


def test_surface_cannot_fire_a_world_event() -> None:
    """Phase 7 fires no ``builder_perturbation`` (reserved for Probe 4) — and the
    guarantee is structural: neither the surface module nor the script imports a
    ``WorldEvent`` schema or a ``world_event`` sink, so neither can construct or
    emit *any* world event. Provenance is the next ``DreamSessionMeta`` snapshot,
    not an event. (Docstrings explaining the reservation are fine; the point is
    that no event can be fired, which the absence of the import enforces.)"""
    for rel in (
        Path("kind") / "training" / "cross_probe_surface.py",
        Path("scripts") / "perturb_dream_envelope.py",
    ):
        imported = _imported_modules(REPO_ROOT / rel)
        assert not any("world_event" in m.lower() for m in imported), rel
        # ``kind.observer.schemas`` is where ``WorldEvent`` lives; not importing
        # it means no event can be built or fired.
        assert not any(m.endswith("observer.schemas") for m in imported), rel


# ---------------------------------------------------------------------------
# The surface round-trips — only the named fields change.
# ---------------------------------------------------------------------------


def test_surface_round_trips_only_named_fields() -> None:
    """A valid envelope/seed-selection delta applies → the resulting configs
    reflect exactly the overridden fields and only those → the delta round-trips
    through the model."""
    base_env = DreamEnvelopeConfig()
    base_seed = SeedSelectionConfig()

    delta_json = (
        '{"dream_envelope": {"hard_cap_rollout_count": 7, '
        '"metabolic_refill_rate": 0.25}, '
        '"seed_selection": {"mode": "hybrid", "replay_warmup_length": 16}}'
    )
    perturbation = stage_perturbation(delta_json)
    result = apply_at_checkpoint_boundary(
        perturbation, dream_envelope=base_env, seed_selection=base_seed
    )

    # Exactly the named envelope fields changed; the rest are untouched.
    assert result.dream_envelope.hard_cap_rollout_count == 7
    assert result.dream_envelope.metabolic_refill_rate == 0.25
    assert (
        result.dream_envelope.hard_cap_wallclock_ms == base_env.hard_cap_wallclock_ms
    )
    assert (
        result.dream_envelope.dormant_heartbeat_interval_ms
        == base_env.dormant_heartbeat_interval_ms
    )
    assert (
        result.dream_envelope.checkpoint_window_force_dormant
        == base_env.checkpoint_window_force_dormant
    )

    # Exactly the named seed fields changed; the rest are untouched.
    assert result.seed_selection.mode == "hybrid"
    assert result.seed_selection.replay_warmup_length == 16
    assert result.seed_selection.perturbation_sigma == base_seed.perturbation_sigma
    assert (
        result.seed_selection.replay_min_segment_age_steps
        == base_seed.replay_min_segment_age_steps
    )

    # The delta round-trips through the model (JSON → model → JSON → model).
    reparsed = DreamPerturbation.model_validate_json(perturbation.model_dump_json())
    assert reparsed == perturbation


def test_partial_delta_overrides_only_set_fields() -> None:
    """An envelope-only delta leaves seed-selection entirely unchanged, and vice
    versa — only the set (non-None) fields are applied."""
    base_env = DreamEnvelopeConfig(hard_cap_rollout_count=50)
    base_seed = SeedSelectionConfig(mode="replay")

    env_only = stage_perturbation('{"dream_envelope": {"hard_cap_rollout_count": 5}}')
    r1 = apply_at_checkpoint_boundary(
        env_only, dream_envelope=base_env, seed_selection=base_seed
    )
    assert r1.dream_envelope.hard_cap_rollout_count == 5
    assert r1.seed_selection == base_seed  # untouched

    seed_only = stage_perturbation('{"seed_selection": {"mode": "perturbed_prior"}}')
    r2 = apply_at_checkpoint_boundary(
        seed_only, dream_envelope=base_env, seed_selection=base_seed
    )
    assert r2.seed_selection.mode == "perturbed_prior"
    assert r2.dream_envelope == base_env  # untouched


# ---------------------------------------------------------------------------
# Checkpoint-boundary application — staged, not live mid-state.
# ---------------------------------------------------------------------------


def test_application_is_staged_not_live() -> None:
    """Staging (parse/validate) does not touch any config; only the boundary
    apply produces new configs, and it returns *new* objects rather than mutating
    the inputs — the atomic-state-sync "swap at the boundary, not live mid-state"
    semantics, testable without a live run."""
    base_env = DreamEnvelopeConfig()
    base_seed = SeedSelectionConfig()

    # Stage: the perturbation exists, but nothing has been applied yet.
    perturbation = stage_perturbation('{"dream_envelope": {"hard_cap_rollout_count": 3}}')
    assert base_env.hard_cap_rollout_count == 50  # original config untouched by staging

    # Apply at the (simulated) boundary: the result is a NEW object; the inputs
    # are unchanged (frozen → not mutated in place).
    result = apply_at_checkpoint_boundary(
        perturbation, dream_envelope=base_env, seed_selection=base_seed
    )
    assert result.dream_envelope is not base_env
    assert result.dream_envelope.hard_cap_rollout_count == 3
    assert base_env.hard_cap_rollout_count == 50  # input still untouched post-apply
    # The seed config was not in the delta — the same object flows through.
    assert result.seed_selection is base_seed


def test_applied_perturbation_snapshot_is_provenance() -> None:
    """The applied perturbation's resulting config is exactly what the next
    ``DreamSessionMeta.envelope_config_snapshot`` would record — the provenance
    path, no new event type needed."""
    perturbation = stage_perturbation(
        '{"dream_envelope": {"dormant_heartbeat_interval_ms": 30000}}'
    )
    result = apply_at_checkpoint_boundary(
        perturbation,
        dream_envelope=DreamEnvelopeConfig(),
        seed_selection=SeedSelectionConfig(),
    )
    snapshot = dataclasses.asdict(result.dream_envelope)
    assert snapshot["dormant_heartbeat_interval_ms"] == 30000
    # The snapshot is a plain JSON-serializable dict (the DreamSessionMeta field
    # type), carrying exactly the envelope config's fields.
    assert set(snapshot) == {f.name for f in dataclasses.fields(DreamEnvelopeConfig)}


# ---------------------------------------------------------------------------
# The surface on RunnerConfig is additive and inert (smokes stay green).
# ---------------------------------------------------------------------------


def test_runner_config_exposes_inert_surface() -> None:
    """The two new ``RunnerConfig`` fields are additive (settled defaults) and
    inert (the waking loop reads neither). A config built without mentioning them
    gets the settled defaults — so existing construction sites (the integration
    smokes) are unchanged."""
    fields = RunnerConfig.__dataclass_fields__
    assert "dream_envelope" in fields
    assert "seed_selection" in fields
    assert fields["dream_envelope"].default == DreamEnvelopeConfig()
    assert fields["seed_selection"].default == SeedSelectionConfig()


def test_runner_config_surface_is_perturbable_from_outside() -> None:
    """The §4 "perturbable from outside" check: a builder can construct a
    ``RunnerConfig`` carrying a perturbed envelope/seed-selection (the configs the
    cross-probe surface produces), and they round-trip onto the config."""
    perturbation = stage_perturbation(
        '{"dream_envelope": {"hard_cap_rollout_count": 9}, '
        '"seed_selection": {"mode": "hybrid"}}'
    )
    result = apply_at_checkpoint_boundary(
        perturbation,
        dream_envelope=DreamEnvelopeConfig(),
        seed_selection=SeedSelectionConfig(),
    )
    cfg = RunnerConfig(
        world_model_config=_minimal_world_model_config(),
        run_id="phase7-surface",
        telemetry_dir=Path("/tmp/phase7/telemetry"),
        checkpoints_dir=Path("/tmp/phase7/checkpoints"),
        dream_envelope=result.dream_envelope,
        seed_selection=result.seed_selection,
    )
    assert cfg.dream_envelope.hard_cap_rollout_count == 9
    assert cfg.seed_selection.mode == "hybrid"


def _minimal_world_model_config() -> object:
    from kind.agents.world_model import WorldModelConfig

    return WorldModelConfig()
