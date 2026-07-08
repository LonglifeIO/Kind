"""Probe 4 Phase 1 — the SELF-class extraction contract (plan §S-CTRL).

``extract_self_action_effects`` derives Io's own consumption events from
``AgentStep`` records (``action_t`` + the ``true_energy`` jump between
consecutive from-state samples + the ``h_t`` transition). These tests pin
the contract: detection threshold semantics, episode-boundary and gap
exclusions, movement-action attribution, the h-transition orientation,
and the loud failure on runs without energy telemetry.
"""

from __future__ import annotations

import pytest

from kind.observer.schemas import (
    PROBE_3_5_TELEMETRY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    AgentStep,
)
from kind.observer.source_events import (
    CONSUMPTION_JUMP_THRESHOLD,
    SelfActionEffect,
    extract_self_action_effects,
)

# The settled physics, normalized (grid_world defaults / span 10):
# replenish +0.08, base decay −0.008, move cost −0.004.
_ORDINARY_MOVE_DELTA = -0.012
_CONSUMPTION_MOVE_DELTA = 0.068


def _make_step(
    *,
    t: int,
    episode_id: int,
    action: int,
    true_energy: float | None,
    h_seed: float,
    schema_version: str = PROBE_3_5_TELEMETRY_SCHEMA_VERSION,
) -> AgentStep:
    """A minimal valid AgentStep; energy fields None only at legacy
    versions (the validator enforces the version discipline)."""
    energy_fields: dict[str, float | None] = (
        {
            "sensed_energy_t": true_energy,
            "true_energy_t": true_energy,
            "energy_pred_t": true_energy,
            "energy_recon_error_t": 0.0,
        }
        if true_energy is not None
        else {
            "sensed_energy_t": None,
            "true_energy_t": None,
            "energy_pred_t": None,
            "energy_recon_error_t": None,
        }
    )
    return AgentStep(
        schema_version=schema_version,
        run_id="self-class-test",
        checkpoint_id=None,
        t=t,
        episode_id=episode_id,
        step_in_episode=t % 200,
        wallclock_ms=t,
        h_t=[h_seed, h_seed + 1.0],
        q_params_t=([0.0], [0.0]),
        p_params_t=([0.0], [0.0]),
        z_t=[0.0],
        kl_per_dim_t=[0.0],
        kl_aggregate_t=0.0,
        recon_loss_t=0.0,
        action_t=action,
        action_logprob_t=0.0,
        policy_entropy_t=0.0,
        obs_hash_t="hash",
        intrinsic_signal_t=0.0,
        encoder_embedding_t=[0.0],
        self_prediction_t=[0.0, 0.0],
        self_prediction_error_t=0.0,
        self_prediction_error_masked_t=False,
        **energy_fields,  # type: ignore[arg-type]
    )


def _trajectory_with_one_consumption() -> list[AgentStep]:
    """Four contiguous steps; the action at t=11 (a move) consumes, so
    the jump appears between the t=11 and t=12 from-state samples."""
    return [
        _make_step(t=10, episode_id=0, action=1, true_energy=0.500, h_seed=10.0),
        _make_step(
            t=11,
            episode_id=0,
            action=2,
            true_energy=0.500 + _ORDINARY_MOVE_DELTA,
            h_seed=11.0,
        ),
        _make_step(
            t=12,
            episode_id=0,
            action=4,
            true_energy=0.488 + _CONSUMPTION_MOVE_DELTA,
            h_seed=12.0,
        ),
        _make_step(
            t=13,
            episode_id=0,
            action=4,
            true_energy=0.556 - 0.008,
            h_seed=13.0,
        ),
    ]


def test_detects_consumption_and_orients_h_transition() -> None:
    steps = _trajectory_with_one_consumption()
    events = extract_self_action_effects(steps)
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, SelfActionEffect)
    # Attributed to the earlier record: the from-state the action was
    # taken in (AgentStep semantics — t/h_t/action_t/true_energy_t are
    # all keyed to the from state).
    assert event.t == 11
    assert event.action == 2
    assert event.true_energy_before == pytest.approx(0.488)
    assert event.true_energy_after == pytest.approx(0.556)
    # h_before is the last h computed before the consumption's
    # observation arrived; h_after the first computed from it.
    assert event.h_before == (11.0, 12.0)
    assert event.h_after == (12.0, 13.0)


def test_ordinary_steps_produce_no_events() -> None:
    steps = [
        _make_step(
            t=t,
            episode_id=0,
            action=1,
            true_energy=0.5 + _ORDINARY_MOVE_DELTA * (t - 10),
            h_seed=float(t),
        )
        for t in range(10, 15)
    ]
    assert extract_self_action_effects(steps) == []


def test_episode_boundary_pair_excluded() -> None:
    """A jump across an episode boundary is excluded: the runner
    zero-resets h at the boundary, so the h-transition there is not a
    world-model transition."""
    steps = [
        _make_step(t=10, episode_id=0, action=1, true_energy=0.5, h_seed=1.0),
        _make_step(t=11, episode_id=1, action=4, true_energy=0.58, h_seed=2.0),
    ]
    assert extract_self_action_effects(steps) == []


def test_non_contiguous_pair_skipped() -> None:
    steps = [
        _make_step(t=10, episode_id=0, action=1, true_energy=0.5, h_seed=1.0),
        _make_step(t=12, episode_id=0, action=4, true_energy=0.58, h_seed=2.0),
    ]
    assert extract_self_action_effects(steps) == []


def test_disordered_records_raise() -> None:
    steps = [
        _make_step(t=11, episode_id=0, action=1, true_energy=0.5, h_seed=1.0),
        _make_step(t=10, episode_id=0, action=1, true_energy=0.5, h_seed=2.0),
    ]
    with pytest.raises(ValueError, match="ordered"):
        extract_self_action_effects(steps)


def test_jump_attributed_to_stay_raises() -> None:
    """Consumption is entry-triggered; a jump on a stay step is a data
    inconsistency, not an event."""
    steps = [
        _make_step(t=10, episode_id=0, action=4, true_energy=0.5, h_seed=1.0),
        _make_step(t=11, episode_id=0, action=4, true_energy=0.58, h_seed=2.0),
    ]
    with pytest.raises(ValueError, match="not a movement"):
        extract_self_action_effects(steps)


def test_records_without_energy_telemetry_raise() -> None:
    """The SELF class needs true_energy_t; a run without energy telemetry
    fails loudly, naming the flag."""
    steps = [
        _make_step(
            t=10,
            episode_id=0,
            action=1,
            true_energy=None,
            h_seed=1.0,
            schema_version=SCHEMA_VERSION,
        ),
        _make_step(
            t=11,
            episode_id=0,
            action=1,
            true_energy=None,
            h_seed=2.0,
            schema_version=SCHEMA_VERSION,
        ),
    ]
    with pytest.raises(ValueError, match="energy_telemetry"):
        extract_self_action_effects(steps)


def test_threshold_splits_consumption_from_ordinary_noise() -> None:
    """The house threshold sits between the ordinary-step delta and the
    consumption jump at the settled physics (seek-classification §1)."""
    assert abs(_ORDINARY_MOVE_DELTA) < CONSUMPTION_JUMP_THRESHOLD
    assert _CONSUMPTION_MOVE_DELTA > CONSUMPTION_JUMP_THRESHOLD
