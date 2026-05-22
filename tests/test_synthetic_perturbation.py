"""Phase 13 gate test —
:mod:`kind.mirror.calibration.synthetic_perturbation`.

Covers the calibration's synthetic-real-perturbation injection protocol
— the discriminative-case layer that closes Phase 12's newly-open §1:

- :class:`SyntheticPerturbationEntry` validation: forces
  ``is_synthetic=True`` *and* ``is_sham=False`` on the payload; rejects
  payloads missing either flag, or with the flags in disagreement;
  rejects empty checkpoint_id / negative pass_index / negative
  synthetic_t.
- :class:`SyntheticPerturbationSchedule` no-collision invariant: two
  entries sharing ``(checkpoint_id, pass_index, synthetic_t)`` raise at
  construction.
- :func:`generate_synthetic_perturbation_schedule`: determinism (same
  seed → same schedule); placement-window bounds respecting
  ``recovery_window``; argument validation; cross-schedule disjointness
  against a supplied :class:`ShamSchedule`.
- :func:`inject_synthetic_events`: immutability of the original
  timeline; sorted-order preservation across mixed real + synthetic
  events; empty-entries fast path; missing-wallclock lookup raises;
  collision with an existing event raises via the timeline's
  sorted-unique validator; round-trip serialization of the augmented
  timeline.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kind.mirror.calibration.sham_schedule import (
    ShamScheduleEntry,
    generate_sham_schedule,
)
from kind.mirror.calibration.synthetic_perturbation import (
    SyntheticPerturbationEntry,
    SyntheticPerturbationSchedule,
    generate_synthetic_perturbation_schedule,
    inject_synthetic_events,
)
from kind.mirror.perturbation_align import (
    PerturbationEvent,
    PerturbationTimeline,
)


# ---------------------------------------------------------------------------
# SyntheticPerturbationEntry validation.
# ---------------------------------------------------------------------------


def test_entry_with_both_flags_set_correctly_succeeds() -> None:
    entry = SyntheticPerturbationEntry(
        checkpoint_id="ckpt-1",
        pass_index=0,
        synthetic_t=50,
        synthetic_payload={
            "is_synthetic": True,
            "is_sham": False,
            "source": "test",
        },
    )
    assert entry.synthetic_payload["is_synthetic"] is True
    assert entry.synthetic_payload["is_sham"] is False


def test_entry_rejects_payload_missing_is_synthetic_key() -> None:
    with pytest.raises(ValidationError, match="is_synthetic"):
        SyntheticPerturbationEntry(
            checkpoint_id="ckpt-1",
            pass_index=0,
            synthetic_t=50,
            synthetic_payload={"is_sham": False, "kind": "test"},
        )


def test_entry_rejects_payload_with_is_synthetic_false() -> None:
    with pytest.raises(ValidationError, match="is_synthetic"):
        SyntheticPerturbationEntry(
            checkpoint_id="ckpt-1",
            pass_index=0,
            synthetic_t=50,
            synthetic_payload={"is_synthetic": False, "is_sham": False},
        )


def test_entry_rejects_payload_missing_is_sham_key() -> None:
    with pytest.raises(ValidationError, match="is_sham"):
        SyntheticPerturbationEntry(
            checkpoint_id="ckpt-1",
            pass_index=0,
            synthetic_t=50,
            synthetic_payload={"is_synthetic": True, "kind": "test"},
        )


def test_entry_rejects_payload_with_is_sham_true() -> None:
    """A synthetic entry cannot also claim to be a sham — the three
    categories are mutually exclusive at the payload level."""
    with pytest.raises(ValidationError, match="mutually exclusive"):
        SyntheticPerturbationEntry(
            checkpoint_id="ckpt-1",
            pass_index=0,
            synthetic_t=50,
            synthetic_payload={"is_synthetic": True, "is_sham": True},
        )


def test_entry_rejects_empty_checkpoint_id() -> None:
    with pytest.raises(ValidationError):
        SyntheticPerturbationEntry(
            checkpoint_id="",
            pass_index=0,
            synthetic_t=50,
            synthetic_payload={"is_synthetic": True, "is_sham": False},
        )


def test_entry_rejects_negative_pass_index() -> None:
    with pytest.raises(ValidationError):
        SyntheticPerturbationEntry(
            checkpoint_id="c",
            pass_index=-1,
            synthetic_t=50,
            synthetic_payload={"is_synthetic": True, "is_sham": False},
        )


def test_entry_rejects_negative_synthetic_t() -> None:
    with pytest.raises(ValidationError):
        SyntheticPerturbationEntry(
            checkpoint_id="c",
            pass_index=0,
            synthetic_t=-1,
            synthetic_payload={"is_synthetic": True, "is_sham": False},
        )


# ---------------------------------------------------------------------------
# SyntheticPerturbationSchedule invariants.
# ---------------------------------------------------------------------------


def test_schedule_with_no_collisions_succeeds() -> None:
    SyntheticPerturbationSchedule(
        entries=(
            SyntheticPerturbationEntry(
                checkpoint_id="c",
                pass_index=0,
                synthetic_t=10,
                synthetic_payload={"is_synthetic": True, "is_sham": False},
            ),
            SyntheticPerturbationEntry(
                checkpoint_id="c",
                pass_index=0,
                synthetic_t=20,
                synthetic_payload={"is_synthetic": True, "is_sham": False},
            ),
            SyntheticPerturbationEntry(
                checkpoint_id="c",
                pass_index=1,
                synthetic_t=10,  # same t but different pass — no collision
                synthetic_payload={"is_synthetic": True, "is_sham": False},
            ),
        ),
        synthetics_per_pass=2,
        seed=142,
    )


def test_schedule_rejects_collision() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        SyntheticPerturbationSchedule(
            entries=(
                SyntheticPerturbationEntry(
                    checkpoint_id="c",
                    pass_index=0,
                    synthetic_t=10,
                    synthetic_payload={
                        "is_synthetic": True,
                        "is_sham": False,
                    },
                ),
                SyntheticPerturbationEntry(
                    checkpoint_id="c",
                    pass_index=0,
                    synthetic_t=10,
                    synthetic_payload={
                        "is_synthetic": True,
                        "is_sham": False,
                    },
                ),
            ),
            synthetics_per_pass=2,
            seed=142,
        )


def test_schedule_serialization_round_trip() -> None:
    sched = generate_synthetic_perturbation_schedule(
        checkpoint_ids=("c1", "c2"),
        passes_per_checkpoint=3,
        synthetics_per_pass=2,
        telemetry_length=200,
        recovery_window=50,
        seed=142,
    )
    redumped = SyntheticPerturbationSchedule.model_validate_json(
        sched.model_dump_json()
    )
    assert redumped == sched


def test_schedule_rejects_negative_synthetics_per_pass() -> None:
    with pytest.raises(ValidationError):
        SyntheticPerturbationSchedule(
            entries=(), synthetics_per_pass=-1, seed=142
        )


# ---------------------------------------------------------------------------
# generate_synthetic_perturbation_schedule.
# ---------------------------------------------------------------------------


def test_generate_synthetic_perturbation_schedule_determinism() -> None:
    """Same seed → byte-identical schedules."""
    a = generate_synthetic_perturbation_schedule(
        checkpoint_ids=("c1", "c2"),
        passes_per_checkpoint=5,
        synthetics_per_pass=2,
        telemetry_length=200,
        recovery_window=50,
        seed=142,
    )
    b = generate_synthetic_perturbation_schedule(
        checkpoint_ids=("c1", "c2"),
        passes_per_checkpoint=5,
        synthetics_per_pass=2,
        telemetry_length=200,
        recovery_window=50,
        seed=142,
    )
    assert a == b


def test_generate_synthetic_perturbation_schedule_different_seeds_differ() -> None:
    a = generate_synthetic_perturbation_schedule(
        checkpoint_ids=("c1",),
        passes_per_checkpoint=5,
        synthetics_per_pass=2,
        telemetry_length=200,
        recovery_window=50,
        seed=142,
    )
    b = generate_synthetic_perturbation_schedule(
        checkpoint_ids=("c1",),
        passes_per_checkpoint=5,
        synthetics_per_pass=2,
        telemetry_length=200,
        recovery_window=50,
        seed=143,
    )
    assert a != b
    assert a.seed == 142
    assert b.seed == 143


def test_generate_synthetic_perturbation_schedule_entry_count() -> None:
    """Total entries == checkpoint_count × passes × synthetics_per_pass."""
    sched = generate_synthetic_perturbation_schedule(
        checkpoint_ids=("c1", "c2"),
        passes_per_checkpoint=5,
        synthetics_per_pass=2,
        telemetry_length=200,
        recovery_window=50,
        seed=142,
    )
    assert len(sched.entries) == 2 * 5 * 2


def test_generate_synthetic_perturbation_schedule_all_entries_have_flags() -> None:
    sched = generate_synthetic_perturbation_schedule(
        checkpoint_ids=("c1",),
        passes_per_checkpoint=3,
        synthetics_per_pass=2,
        telemetry_length=200,
        recovery_window=50,
        seed=142,
    )
    for e in sched.entries:
        assert e.synthetic_payload["is_synthetic"] is True
        assert e.synthetic_payload["is_sham"] is False


def test_generate_synthetic_perturbation_schedule_placement_window_bounds() -> None:
    """Every synthetic_t must fall in [1, telemetry_length - recovery_window - 2].

    With telemetry_length=200, recovery_window=50 → window is [1, 148].
    The upper bound guarantees a full recovery_window of post-perturbation
    samples is available for the equanimity statistics.
    """
    sched = generate_synthetic_perturbation_schedule(
        checkpoint_ids=("c",),
        passes_per_checkpoint=3,
        synthetics_per_pass=5,
        telemetry_length=200,
        recovery_window=50,
        seed=142,
    )
    for e in sched.entries:
        assert 1 <= e.synthetic_t <= 148, (
            f"synthetic_t={e.synthetic_t} outside placement window "
            f"[1, 148]"
        )


def test_generate_synthetic_perturbation_schedule_rejects_too_small_window() -> None:
    """telemetry_length too short to leave recovery_window samples raises."""
    with pytest.raises(ValueError, match="placement window"):
        generate_synthetic_perturbation_schedule(
            checkpoint_ids=("c",),
            passes_per_checkpoint=3,
            synthetics_per_pass=1,
            telemetry_length=20,
            recovery_window=50,
            seed=142,
        )


def test_generate_synthetic_perturbation_schedule_rejects_too_many_synthetics() -> None:
    """synthetics_per_pass exceeding placement window raises."""
    with pytest.raises(ValueError, match="placement window"):
        generate_synthetic_perturbation_schedule(
            checkpoint_ids=("c",),
            passes_per_checkpoint=1,
            synthetics_per_pass=200,
            telemetry_length=100,
            recovery_window=20,
            seed=142,
        )


def test_generate_synthetic_perturbation_schedule_rejects_empty_checkpoint_ids() -> None:
    with pytest.raises(ValueError):
        generate_synthetic_perturbation_schedule(
            checkpoint_ids=(),
            passes_per_checkpoint=3,
            synthetics_per_pass=2,
            telemetry_length=200,
            recovery_window=50,
            seed=142,
        )


def test_generate_synthetic_perturbation_schedule_rejects_zero_passes() -> None:
    with pytest.raises(ValueError):
        generate_synthetic_perturbation_schedule(
            checkpoint_ids=("c",),
            passes_per_checkpoint=0,
            synthetics_per_pass=2,
            telemetry_length=200,
            recovery_window=50,
            seed=142,
        )


def test_generate_synthetic_perturbation_schedule_disjoint_from_sham_schedule() -> None:
    """When a sham schedule is supplied, no synthetic shares a
    ``(checkpoint_id, pass_index, t)`` slot with a sham."""
    sham = generate_sham_schedule(
        checkpoint_ids=("c1",),
        passes_per_checkpoint=5,
        real_perturbations_per_pass=2,
        shams_per_pass=1,
        telemetry_length=200,
        seed=42,
    )
    synthetic = generate_synthetic_perturbation_schedule(
        checkpoint_ids=("c1",),
        passes_per_checkpoint=5,
        synthetics_per_pass=2,
        telemetry_length=200,
        recovery_window=50,
        seed=142,
        sham_schedule=sham,
    )
    # Collisions per (checkpoint, pass) must not exist between schedules.
    sham_slots = {(s.checkpoint_id, s.pass_index, s.sham_t) for s in sham.entries}
    synthetic_slots = {
        (e.checkpoint_id, e.pass_index, e.synthetic_t)
        for e in synthetic.entries
    }
    assert sham_slots.isdisjoint(synthetic_slots), (
        "synthetic schedule contains a slot also held by the sham "
        "schedule; cross-schedule disjointness was not enforced"
    )


# ---------------------------------------------------------------------------
# inject_synthetic_events.
# ---------------------------------------------------------------------------


def _wallclock_lookup(n: int) -> dict[int, int]:
    """Build a {t: wallclock_ms} lookup for steps 0..n-1 with
    wallclock = t*100."""
    return {t: t * 100 for t in range(n)}


def test_inject_synthetic_events_empty_entries_returns_input() -> None:
    timeline = PerturbationTimeline(
        events=(
            PerturbationEvent(
                t=10,
                wallclock_ms=1000,
                payload={"kind": "real"},
                is_sham=False,
            ),
        ),
        run_id="r",
        checkpoint_id="c",
    )
    out = inject_synthetic_events(
        timeline,
        synthetic_entries=(),
        agent_step_wallclock_lookup=_wallclock_lookup(100),
    )
    assert out is timeline  # same object — fast path


def test_inject_synthetic_events_preserves_sorted_order_mixed_real_and_synthetic() -> None:
    timeline = PerturbationTimeline(
        events=(
            PerturbationEvent(
                t=30, wallclock_ms=3000, payload={"kind": "real"}
            ),
            PerturbationEvent(
                t=70, wallclock_ms=7000, payload={"kind": "real"}
            ),
        ),
        run_id="r",
        checkpoint_id="c",
    )
    entries = (
        SyntheticPerturbationEntry(
            checkpoint_id="c",
            pass_index=0,
            synthetic_t=50,
            synthetic_payload={"is_synthetic": True, "is_sham": False},
        ),
        SyntheticPerturbationEntry(
            checkpoint_id="c",
            pass_index=0,
            synthetic_t=10,
            synthetic_payload={"is_synthetic": True, "is_sham": False},
        ),
    )
    out = inject_synthetic_events(
        timeline,
        synthetic_entries=entries,
        agent_step_wallclock_lookup=_wallclock_lookup(100),
    )
    ts = [e.t for e in out.events]
    assert ts == sorted(ts)
    assert ts == [10, 30, 50, 70]


def test_inject_synthetic_events_does_not_mutate_original() -> None:
    timeline = PerturbationTimeline(
        events=(
            PerturbationEvent(t=30, wallclock_ms=3000, payload={}),
        ),
        run_id="r",
        checkpoint_id="c",
    )
    entries = (
        SyntheticPerturbationEntry(
            checkpoint_id="c",
            pass_index=0,
            synthetic_t=50,
            synthetic_payload={"is_synthetic": True, "is_sham": False},
        ),
    )
    inject_synthetic_events(
        timeline,
        synthetic_entries=entries,
        agent_step_wallclock_lookup=_wallclock_lookup(100),
    )
    assert len(timeline.events) == 1
    assert timeline.events[0].t == 30


def test_inject_synthetic_events_routes_through_real_perturbation_codepath() -> None:
    """Synthetic events must land on the timeline with ``is_sham=False``
    so the recovery-statistic codepath processes them as real
    perturbations (the recovery_lag / trajectory classifiers skip
    is_sham=True events)."""
    timeline = PerturbationTimeline(
        events=(),
        run_id="r",
        checkpoint_id="c",
    )
    entries = (
        SyntheticPerturbationEntry(
            checkpoint_id="c",
            pass_index=0,
            synthetic_t=50,
            synthetic_payload={"is_synthetic": True, "is_sham": False},
        ),
    )
    out = inject_synthetic_events(
        timeline,
        synthetic_entries=entries,
        agent_step_wallclock_lookup=_wallclock_lookup(100),
    )
    assert len(out.events) == 1
    event = out.events[0]
    assert event.is_sham is False  # routed through real-perturbation path
    assert event.payload["is_synthetic"] is True  # category preserved
    assert event.payload["is_sham"] is False
    assert event.wallclock_ms == 5000  # sourced from the lookup


def test_inject_synthetic_events_missing_wallclock_lookup_raises() -> None:
    timeline = PerturbationTimeline(
        events=(), run_id="r", checkpoint_id="c"
    )
    entries = (
        SyntheticPerturbationEntry(
            checkpoint_id="c",
            pass_index=0,
            synthetic_t=50,
            synthetic_payload={"is_synthetic": True, "is_sham": False},
        ),
    )
    with pytest.raises(ValueError, match="wallclock"):
        inject_synthetic_events(
            timeline,
            synthetic_entries=entries,
            agent_step_wallclock_lookup={},  # missing 50
        )


def test_inject_synthetic_events_collision_with_real_raises() -> None:
    """If a synthetic_t collides with an existing real perturbation t,
    the timeline's sorted-unique validator rejects."""
    timeline = PerturbationTimeline(
        events=(
            PerturbationEvent(
                t=50, wallclock_ms=5000, payload={"kind": "real"}
            ),
        ),
        run_id="r",
        checkpoint_id="c",
    )
    entries = (
        SyntheticPerturbationEntry(
            checkpoint_id="c",
            pass_index=0,
            synthetic_t=50,
            synthetic_payload={"is_synthetic": True, "is_sham": False},
        ),
    )
    with pytest.raises(ValidationError, match="t=50"):
        inject_synthetic_events(
            timeline,
            synthetic_entries=entries,
            agent_step_wallclock_lookup=_wallclock_lookup(100),
        )


def test_inject_synthetic_events_collision_with_sham_raises() -> None:
    """If a synthetic_t collides with an existing sham event in the
    timeline (already injected upstream), the timeline's validator
    rejects."""
    timeline = PerturbationTimeline(
        events=(
            PerturbationEvent(
                t=50,
                wallclock_ms=5000,
                payload={"is_sham": True},
                is_sham=True,
            ),
        ),
        run_id="r",
        checkpoint_id="c",
    )
    entries = (
        SyntheticPerturbationEntry(
            checkpoint_id="c",
            pass_index=0,
            synthetic_t=50,
            synthetic_payload={"is_synthetic": True, "is_sham": False},
        ),
    )
    with pytest.raises(ValidationError, match="t=50"):
        inject_synthetic_events(
            timeline,
            synthetic_entries=entries,
            agent_step_wallclock_lookup=_wallclock_lookup(100),
        )


def test_inject_synthetic_events_round_trip_serialization() -> None:
    """The augmented timeline serializes and round-trips."""
    timeline = PerturbationTimeline(
        events=(),
        run_id="r",
        checkpoint_id="c",
    )
    entries = (
        SyntheticPerturbationEntry(
            checkpoint_id="c",
            pass_index=0,
            synthetic_t=50,
            synthetic_payload={"is_synthetic": True, "is_sham": False},
        ),
        SyntheticPerturbationEntry(
            checkpoint_id="c",
            pass_index=0,
            synthetic_t=80,
            synthetic_payload={"is_synthetic": True, "is_sham": False},
        ),
    )
    out = inject_synthetic_events(
        timeline,
        synthetic_entries=entries,
        agent_step_wallclock_lookup=_wallclock_lookup(100),
    )
    redumped = PerturbationTimeline.model_validate_json(out.model_dump_json())
    assert redumped == out


def test_sham_entry_and_synthetic_entry_slot_keys_are_disjoint_by_construction() -> None:
    """A sham entry and a synthetic entry constructed against the same
    ``(checkpoint_id, pass_index, t)`` would each individually validate,
    so the cross-schedule disjointness must live above the per-entry
    layer. This test confirms the asymmetry: the two entry models exist
    independently; the round-driver/generator is the layer that bridges
    them."""
    sham = ShamScheduleEntry(
        checkpoint_id="c",
        pass_index=0,
        sham_t=50,
        sham_payload={"is_sham": True},
    )
    synthetic = SyntheticPerturbationEntry(
        checkpoint_id="c",
        pass_index=0,
        synthetic_t=50,
        synthetic_payload={"is_synthetic": True, "is_sham": False},
    )
    # Both constructed cleanly. The keys collide, but the entry models
    # don't enforce cross-schedule disjointness on their own.
    assert (sham.checkpoint_id, sham.pass_index, sham.sham_t) == (
        synthetic.checkpoint_id,
        synthetic.pass_index,
        synthetic.synthetic_t,
    )
