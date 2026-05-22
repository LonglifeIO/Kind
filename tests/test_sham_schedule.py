"""Phase 12 gate test — :mod:`kind.mirror.calibration.sham_schedule`.

Covers the calibration's sham-injection protocol:

- :class:`ShamScheduleEntry` validation: forces ``is_sham=True`` on the
  payload; rejects payloads with is_sham absent or False; rejects empty
  checkpoint_id / negative pass_index / negative sham_t.
- :class:`ShamSchedule` no-collision invariant: two entries sharing
  ``(checkpoint_id, pass_index, sham_t)`` raise at construction.
- :func:`generate_sham_schedule`: determinism (same seed → same
  schedule); placement-window bounds; argument validation.
- :func:`inject_sham_events`: immutability of the original timeline;
  sorted-order preservation; empty-entries fast path; missing-wallclock
  lookup raises; round-trip serialization of the augmented timeline.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kind.mirror.calibration.sham_schedule import (
    ShamSchedule,
    ShamScheduleEntry,
    generate_sham_schedule,
    inject_sham_events,
)
from kind.mirror.perturbation_align import (
    PerturbationEvent,
    PerturbationTimeline,
)


# ---------------------------------------------------------------------------
# ShamScheduleEntry validation.
# ---------------------------------------------------------------------------


def test_entry_with_is_sham_true_payload_succeeds() -> None:
    entry = ShamScheduleEntry(
        checkpoint_id="ckpt-1",
        pass_index=0,
        sham_t=50,
        sham_payload={"is_sham": True, "kind": "test"},
    )
    assert entry.sham_payload["is_sham"] is True


def test_entry_rejects_payload_without_is_sham_key() -> None:
    with pytest.raises(ValidationError, match="is_sham"):
        ShamScheduleEntry(
            checkpoint_id="ckpt-1",
            pass_index=0,
            sham_t=50,
            sham_payload={"kind": "test"},
        )


def test_entry_rejects_payload_with_is_sham_false() -> None:
    with pytest.raises(ValidationError, match="is_sham"):
        ShamScheduleEntry(
            checkpoint_id="ckpt-1",
            pass_index=0,
            sham_t=50,
            sham_payload={"is_sham": False},
        )


def test_entry_rejects_empty_checkpoint_id() -> None:
    with pytest.raises(ValidationError):
        ShamScheduleEntry(
            checkpoint_id="",
            pass_index=0,
            sham_t=50,
            sham_payload={"is_sham": True},
        )


def test_entry_rejects_negative_pass_index() -> None:
    with pytest.raises(ValidationError):
        ShamScheduleEntry(
            checkpoint_id="c",
            pass_index=-1,
            sham_t=50,
            sham_payload={"is_sham": True},
        )


def test_entry_rejects_negative_sham_t() -> None:
    with pytest.raises(ValidationError):
        ShamScheduleEntry(
            checkpoint_id="c",
            pass_index=0,
            sham_t=-1,
            sham_payload={"is_sham": True},
        )


# ---------------------------------------------------------------------------
# ShamSchedule invariants.
# ---------------------------------------------------------------------------


def test_schedule_with_no_collisions_succeeds() -> None:
    ShamSchedule(
        entries=(
            ShamScheduleEntry(
                checkpoint_id="c",
                pass_index=0,
                sham_t=10,
                sham_payload={"is_sham": True},
            ),
            ShamScheduleEntry(
                checkpoint_id="c",
                pass_index=0,
                sham_t=20,
                sham_payload={"is_sham": True},
            ),
            ShamScheduleEntry(
                checkpoint_id="c",
                pass_index=1,
                sham_t=10,  # same t but different pass — no collision
                sham_payload={"is_sham": True},
            ),
        ),
        real_perturbations_per_pass=2,
        shams_per_pass=1,
        seed=42,
    )


def test_schedule_rejects_collision() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        ShamSchedule(
            entries=(
                ShamScheduleEntry(
                    checkpoint_id="c",
                    pass_index=0,
                    sham_t=10,
                    sham_payload={"is_sham": True},
                ),
                ShamScheduleEntry(
                    checkpoint_id="c",
                    pass_index=0,
                    sham_t=10,
                    sham_payload={"is_sham": True},
                ),
            ),
            real_perturbations_per_pass=2,
            shams_per_pass=1,
            seed=42,
        )


def test_schedule_serialization_round_trip() -> None:
    sched = generate_sham_schedule(
        checkpoint_ids=("c1", "c2"),
        passes_per_checkpoint=3,
        real_perturbations_per_pass=2,
        shams_per_pass=1,
        telemetry_length=200,
        seed=42,
    )
    redumped = ShamSchedule.model_validate_json(sched.model_dump_json())
    assert redumped == sched


# ---------------------------------------------------------------------------
# generate_sham_schedule.
# ---------------------------------------------------------------------------


def test_generate_sham_schedule_determinism() -> None:
    """Same seed → byte-identical schedules."""
    a = generate_sham_schedule(
        checkpoint_ids=("c1", "c2"),
        passes_per_checkpoint=5,
        real_perturbations_per_pass=2,
        shams_per_pass=1,
        telemetry_length=200,
        seed=42,
    )
    b = generate_sham_schedule(
        checkpoint_ids=("c1", "c2"),
        passes_per_checkpoint=5,
        real_perturbations_per_pass=2,
        shams_per_pass=1,
        telemetry_length=200,
        seed=42,
    )
    assert a == b


def test_generate_sham_schedule_different_seeds_differ() -> None:
    a = generate_sham_schedule(
        checkpoint_ids=("c1",),
        passes_per_checkpoint=5,
        real_perturbations_per_pass=2,
        shams_per_pass=1,
        telemetry_length=200,
        seed=42,
    )
    b = generate_sham_schedule(
        checkpoint_ids=("c1",),
        passes_per_checkpoint=5,
        real_perturbations_per_pass=2,
        shams_per_pass=1,
        telemetry_length=200,
        seed=43,
    )
    assert a != b
    assert a.seed == 42
    assert b.seed == 43


def test_generate_sham_schedule_entry_count() -> None:
    """Total entries == checkpoint_count × passes × shams_per_pass."""
    sched = generate_sham_schedule(
        checkpoint_ids=("c1", "c2"),
        passes_per_checkpoint=5,
        real_perturbations_per_pass=2,
        shams_per_pass=1,
        telemetry_length=200,
        seed=42,
    )
    assert len(sched.entries) == 2 * 5 * 1


def test_generate_sham_schedule_all_entries_carry_is_sham_true() -> None:
    sched = generate_sham_schedule(
        checkpoint_ids=("c1",),
        passes_per_checkpoint=3,
        real_perturbations_per_pass=2,
        shams_per_pass=2,
        telemetry_length=200,
        seed=42,
    )
    assert all(e.sham_payload["is_sham"] is True for e in sched.entries)


def test_generate_sham_schedule_placement_window() -> None:
    """No sham at step 0 or step telemetry_length-1."""
    sched = generate_sham_schedule(
        checkpoint_ids=("c",),
        passes_per_checkpoint=3,
        real_perturbations_per_pass=2,
        shams_per_pass=5,
        telemetry_length=100,
        seed=42,
    )
    for e in sched.entries:
        assert 1 <= e.sham_t <= 98


def test_generate_sham_schedule_rejects_too_small_telemetry_length() -> None:
    with pytest.raises(ValueError, match="telemetry_length"):
        generate_sham_schedule(
            checkpoint_ids=("c",),
            passes_per_checkpoint=3,
            real_perturbations_per_pass=2,
            shams_per_pass=1,
            telemetry_length=2,
            seed=42,
        )


def test_generate_sham_schedule_rejects_shams_exceeding_window() -> None:
    """shams_per_pass > placement_window_size raises."""
    with pytest.raises(ValueError, match="placement window"):
        generate_sham_schedule(
            checkpoint_ids=("c",),
            passes_per_checkpoint=1,
            real_perturbations_per_pass=2,
            shams_per_pass=20,
            telemetry_length=10,  # window is size 8 (steps 1..8)
            seed=42,
        )


def test_generate_sham_schedule_rejects_empty_checkpoint_ids() -> None:
    with pytest.raises(ValueError):
        generate_sham_schedule(
            checkpoint_ids=(),
            passes_per_checkpoint=3,
            real_perturbations_per_pass=2,
            shams_per_pass=1,
            telemetry_length=100,
            seed=42,
        )


def test_generate_sham_schedule_rejects_zero_passes() -> None:
    with pytest.raises(ValueError):
        generate_sham_schedule(
            checkpoint_ids=("c",),
            passes_per_checkpoint=0,
            real_perturbations_per_pass=2,
            shams_per_pass=1,
            telemetry_length=100,
            seed=42,
        )


# ---------------------------------------------------------------------------
# inject_sham_events.
# ---------------------------------------------------------------------------


def _wallclock_lookup(n: int) -> dict[int, int]:
    """Build a {t: wallclock_ms} lookup for steps 0..n-1 with
    wallclock = t*100."""
    return {t: t * 100 for t in range(n)}


def test_inject_sham_events_empty_entries_returns_input() -> None:
    timeline = PerturbationTimeline(
        events=(
            PerturbationEvent(
                t=10, wallclock_ms=1000, payload={"kind": "real"}, is_sham=False
            ),
        ),
        run_id="r",
        checkpoint_id="c",
    )
    out = inject_sham_events(
        timeline, sham_entries=(), agent_step_wallclock_lookup=_wallclock_lookup(100)
    )
    assert out is timeline  # same object — fast path


def test_inject_sham_events_preserves_sorted_order() -> None:
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
        ShamScheduleEntry(
            checkpoint_id="c",
            pass_index=0,
            sham_t=50,
            sham_payload={"is_sham": True},
        ),
        ShamScheduleEntry(
            checkpoint_id="c",
            pass_index=0,
            sham_t=10,
            sham_payload={"is_sham": True},
        ),
    )
    out = inject_sham_events(
        timeline,
        sham_entries=entries,
        agent_step_wallclock_lookup=_wallclock_lookup(100),
    )
    ts = [e.t for e in out.events]
    assert ts == sorted(ts)
    assert ts == [10, 30, 50, 70]


def test_inject_sham_events_does_not_mutate_original() -> None:
    timeline = PerturbationTimeline(
        events=(
            PerturbationEvent(t=30, wallclock_ms=3000, payload={}),
        ),
        run_id="r",
        checkpoint_id="c",
    )
    entries = (
        ShamScheduleEntry(
            checkpoint_id="c",
            pass_index=0,
            sham_t=50,
            sham_payload={"is_sham": True},
        ),
    )
    inject_sham_events(
        timeline,
        sham_entries=entries,
        agent_step_wallclock_lookup=_wallclock_lookup(100),
    )
    assert len(timeline.events) == 1
    assert timeline.events[0].t == 30


def test_inject_sham_events_missing_wallclock_lookup_raises() -> None:
    timeline = PerturbationTimeline(
        events=(),
        run_id="r",
        checkpoint_id="c",
    )
    entries = (
        ShamScheduleEntry(
            checkpoint_id="c",
            pass_index=0,
            sham_t=50,
            sham_payload={"is_sham": True},
        ),
    )
    with pytest.raises(ValueError, match="wallclock"):
        inject_sham_events(
            timeline,
            sham_entries=entries,
            agent_step_wallclock_lookup={},  # missing 50
        )


def test_inject_sham_events_collision_with_real_raises() -> None:
    """If a sham_t collides with an existing real perturbation t, the
    timeline's sorted-unique validator rejects."""
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
        ShamScheduleEntry(
            checkpoint_id="c",
            pass_index=0,
            sham_t=50,
            sham_payload={"is_sham": True},
        ),
    )
    with pytest.raises(ValidationError, match="t=50"):
        inject_sham_events(
            timeline,
            sham_entries=entries,
            agent_step_wallclock_lookup=_wallclock_lookup(100),
        )
