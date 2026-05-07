"""Phase 5 tests for ``kind/training/replay.py``.

Exercises the FIFO sequence-replay buffer's structural properties:
insert/sample/evict mechanics, episode-boundary respect on sampling, the
ReplayMeta envelope, the handler-callback contract, and the schema-side
fields the buffer populates per :class:`~kind.observer.schemas.ReplayMeta`.

Tests use small int "observation" tensors (one float per element) so the
focus stays on the buffer's logic; the runner-side test
(``test_integration_smoke.py``) exercises real-shape ``(1, 32, 32)``
observations end-to-end.
"""

from __future__ import annotations

import pytest
import torch

from kind.observer.schemas import ReplayMeta
from kind.training.replay import (
    Batch,
    SequenceReplayBuffer,
    Transition,
)


# ---- helpers --------------------------------------------------------------


def _make_transition(
    *,
    env_step: int,
    episode_id: int,
    step_in_episode: int = 0,
    action: int = 0,
    obs_value: float | None = None,
) -> Transition:
    """Build a minimal :class:`Transition`.

    The ``obs`` / ``next_obs`` tensors are shape ``(1, 4, 4)`` with all
    values equal to ``obs_value`` (defaults to ``env_step``) so a sampled
    batch can be visually inspected against the expected env-step
    sequence.
    """
    value = float(env_step) if obs_value is None else obs_value
    obs = torch.full((1, 4, 4), value, dtype=torch.float32)
    next_obs = torch.full((1, 4, 4), value + 0.5, dtype=torch.float32)
    return Transition(
        obs=obs,
        action=action,
        next_obs=next_obs,
        env_step=env_step,
        episode_id=episode_id,
        step_in_episode=step_in_episode,
    )


def _new_buffer(
    capacity: int = 100,
    sequence_length: int = 4,
    run_id: str = "test-run",
) -> tuple[SequenceReplayBuffer, list[ReplayMeta]]:
    """Construct a buffer with an attached records list.

    Returns ``(buffer, records)`` where ``records`` is the list the
    buffer's handler appends to. Tests inspect ``records`` to verify
    insert/sample/evict ordering and field correctness.
    """
    records: list[ReplayMeta] = []
    buffer = SequenceReplayBuffer(
        capacity=capacity,
        sequence_length=sequence_length,
        run_id=run_id,
        replay_meta_handler=records.append,
    )
    return buffer, records


# ---- construction validation ---------------------------------------------


def test_buffer_rejects_non_positive_capacity() -> None:
    with pytest.raises(ValueError, match="capacity must be positive"):
        SequenceReplayBuffer(capacity=0, sequence_length=4)


def test_buffer_rejects_non_positive_sequence_length() -> None:
    with pytest.raises(ValueError, match="sequence_length must be positive"):
        SequenceReplayBuffer(capacity=100, sequence_length=0)


def test_empty_buffer_has_zero_length() -> None:
    buffer, _ = _new_buffer()
    assert len(buffer) == 0


def test_can_sample_false_when_below_sequence_length() -> None:
    buffer, _ = _new_buffer(sequence_length=4)
    # Two transitions are not enough for one window of length 4.
    for i in range(2):
        buffer.insert(_make_transition(env_step=i, episode_id=0, step_in_episode=i))
    assert not buffer.can_sample(1)


# ---- insert mechanics ----------------------------------------------------


def test_insert_appends_and_returns_meta() -> None:
    buffer, records = _new_buffer()
    meta = buffer.insert(_make_transition(env_step=0, episode_id=0))
    assert len(buffer) == 1
    assert meta.event_type == "insert"
    assert meta.t_event == 0
    assert meta.buffer_size == 1
    # Handler also received the same record.
    assert records == [meta]


def test_insert_meta_has_envelope_fields() -> None:
    buffer, _ = _new_buffer(run_id="envelope-test")
    meta = buffer.insert(_make_transition(env_step=42, episode_id=3))
    assert meta.run_id == "envelope-test"
    assert meta.checkpoint_id is None
    assert meta.schema_version == "0.1.0"
    assert meta.priority is None


def test_insert_meta_segment_id_increments() -> None:
    buffer, records = _new_buffer()
    for i in range(5):
        buffer.insert(_make_transition(env_step=i, episode_id=0))
    insert_records = [r for r in records if r.event_type == "insert"]
    assert [r.segment_id for r in insert_records] == [1, 2, 3, 4, 5]


def test_insert_meta_t_event_matches_transition_env_step() -> None:
    buffer, records = _new_buffer()
    buffer.insert(_make_transition(env_step=17, episode_id=0))
    assert records[-1].t_event == 17


def test_set_checkpoint_id_propagates_to_subsequent_records() -> None:
    buffer, records = _new_buffer()
    buffer.insert(_make_transition(env_step=0, episode_id=0))
    assert records[-1].checkpoint_id is None
    buffer.set_checkpoint_id("ckpt-000123")
    buffer.insert(_make_transition(env_step=1, episode_id=0))
    assert records[-1].checkpoint_id == "ckpt-000123"


# ---- eviction ------------------------------------------------------------


def test_evict_emits_record_and_removes_oldest() -> None:
    buffer, records = _new_buffer(capacity=3, sequence_length=2)
    for i in range(3):
        buffer.insert(_make_transition(env_step=i, episode_id=0))
    assert len(buffer) == 3
    records.clear()

    # Now insert a fourth — the buffer is at capacity, so eviction fires.
    buffer.insert(_make_transition(env_step=3, episode_id=0))
    assert len(buffer) == 3

    # Two records emitted: the evict (env_step=0 transition) then the insert.
    assert [r.event_type for r in records] == ["evict", "insert"]
    assert records[0].t_event == 0  # evicted transition
    assert records[1].t_event == 3  # inserted transition


def test_evict_segment_id_increments_independently_from_inserts() -> None:
    buffer, records = _new_buffer(capacity=2, sequence_length=2)
    # Fill, then evict twice.
    for i in range(4):
        buffer.insert(_make_transition(env_step=i, episode_id=0))
    evict_records = [r for r in records if r.event_type == "evict"]
    assert [r.segment_id for r in evict_records] == [1, 2]


def test_evict_buffer_size_after_event() -> None:
    """``buffer_size`` on the evict record reflects state *after* eviction
    (the buffer has just shrunk by one). Then the insert that follows
    raises buffer_size back."""
    buffer, records = _new_buffer(capacity=3, sequence_length=2)
    for i in range(3):
        buffer.insert(_make_transition(env_step=i, episode_id=0))
    records.clear()
    buffer.insert(_make_transition(env_step=3, episode_id=0))
    assert records[0].buffer_size == 2  # after evict
    assert records[1].buffer_size == 3  # after insert


# ---- sample mechanics ----------------------------------------------------


def test_sample_returns_batch_of_correct_shape() -> None:
    buffer, _ = _new_buffer(sequence_length=4)
    for i in range(20):
        buffer.insert(_make_transition(env_step=i, episode_id=0, step_in_episode=i))
    batch, meta = buffer.sample(batch_size=3)
    assert isinstance(batch, Batch)
    assert batch.obs.shape == (3, 4, 1, 4, 4)
    assert batch.next_obs.shape == (3, 4, 1, 4, 4)
    assert batch.action.shape == (3, 4)
    assert batch.action.dtype == torch.long
    assert meta.event_type == "sample"


def test_sample_emits_meta_with_segment_bounds() -> None:
    buffer, records = _new_buffer(sequence_length=4)
    for i in range(20):
        buffer.insert(_make_transition(env_step=i, episode_id=0, step_in_episode=i))
    records.clear()
    _, meta = buffer.sample(batch_size=2)
    assert meta.event_type == "sample"
    # Sample meta is emitted via the handler too.
    sample_records = [r for r in records if r.event_type == "sample"]
    assert sample_records == [meta]
    # Segment bounds are within the buffer's env_step range.
    assert 0 <= meta.segment_start <= meta.segment_end < 20


def test_sample_segment_id_increments() -> None:
    buffer, records = _new_buffer(sequence_length=4)
    for i in range(20):
        buffer.insert(_make_transition(env_step=i, episode_id=0))
    records.clear()
    buffer.sample(batch_size=1)
    buffer.sample(batch_size=1)
    sample_records = [r for r in records if r.event_type == "sample"]
    assert [r.segment_id for r in sample_records] == [1, 2]


def test_sample_raises_when_not_enough_valid_windows() -> None:
    buffer, _ = _new_buffer(sequence_length=4)
    # Only 3 transitions — no full windows of length 4.
    for i in range(3):
        buffer.insert(_make_transition(env_step=i, episode_id=0))
    with pytest.raises(RuntimeError, match="not enough valid windows"):
        buffer.sample(batch_size=1)


def test_sample_window_observations_are_consecutive() -> None:
    """Each sampled sequence is L consecutive transitions from the buffer.

    Since we set ``obs_value=env_step`` in ``_make_transition``, the
    sampled obs sequence values should be consecutive integers.
    """
    buffer, _ = _new_buffer(sequence_length=4)
    for i in range(30):
        buffer.insert(_make_transition(env_step=i, episode_id=0))
    batch, _ = buffer.sample(batch_size=5)
    # Each row's obs values should be consecutive (e.g. [3, 4, 5, 6]).
    for row in range(5):
        values = batch.obs[row].view(4, -1)[:, 0].tolist()
        diffs = [values[i + 1] - values[i] for i in range(3)]
        assert all(d == 1.0 for d in diffs), (
            f"row {row} not consecutive: {values}"
        )


# ---- episode-boundary respect --------------------------------------------


def test_can_sample_excludes_boundary_crossing_windows() -> None:
    """A window of length 4 starting at index i is valid only if all 4
    transitions share the same episode_id. With episodes of length 3 and
    L=4, no window can fit inside one episode."""
    buffer, _ = _new_buffer(sequence_length=4)
    # Two episodes of length 3 each = 6 transitions, but no length-4
    # window fits inside any single episode.
    for ep in range(2):
        for s in range(3):
            buffer.insert(
                _make_transition(
                    env_step=ep * 3 + s,
                    episode_id=ep,
                    step_in_episode=s,
                )
            )
    assert not buffer.can_sample(1)


def test_sampled_window_is_single_episode() -> None:
    """Stress: many small episodes adjacent; sampling must never produce
    a window that crosses an episode boundary."""
    buffer, _ = _new_buffer(sequence_length=4)
    # 5 episodes of length 5 → 25 transitions; valid windows are
    # the [s, s+4) slices that don't cross a boundary. With L=4 and
    # ep_length=5, valid starts within episode E are E*5, E*5+1.
    for ep in range(5):
        for s in range(5):
            buffer.insert(
                _make_transition(
                    env_step=ep * 5 + s,
                    episode_id=ep,
                    step_in_episode=s,
                )
            )
    # Exhaustively sample a fair-sized batch and inspect every row.
    batch, _ = buffer.sample(batch_size=8)
    for row in range(8):
        values = batch.obs[row].view(4, -1)[:, 0].tolist()
        # Episode for each transition is value // 5 (since env_step=ep*5+s).
        episodes = [int(v) // 5 for v in values]
        assert len(set(episodes)) == 1, (
            f"row {row} crossed episode boundary: episodes={episodes}, "
            f"values={values}"
        )


def test_total_segments_reflects_valid_window_count() -> None:
    buffer, records = _new_buffer(sequence_length=4)
    # Episode 0: 5 transitions → 2 valid windows.
    for s in range(5):
        buffer.insert(_make_transition(env_step=s, episode_id=0, step_in_episode=s))
    # Episode 1: 4 transitions → 1 valid window.
    for s in range(4):
        buffer.insert(
            _make_transition(env_step=5 + s, episode_id=1, step_in_episode=s)
        )
    # Total valid windows = 2 + 1 = 3.
    assert records[-1].total_segments == 3


# ---- handler contract -----------------------------------------------------


def test_handler_called_for_every_event_type() -> None:
    buffer, records = _new_buffer(capacity=3, sequence_length=2)
    # 3 inserts (no eviction).
    for i in range(3):
        buffer.insert(_make_transition(env_step=i, episode_id=0))
    # 1 insert (evict + insert = 2 records).
    buffer.insert(_make_transition(env_step=3, episode_id=0))
    # 1 sample.
    buffer.sample(batch_size=1)

    types = [r.event_type for r in records]
    assert types.count("insert") == 4
    assert types.count("evict") == 1
    assert types.count("sample") == 1


def test_handler_is_optional() -> None:
    """Constructing the buffer without a handler is supported; insert /
    sample still return the primary record."""
    buffer = SequenceReplayBuffer(capacity=5, sequence_length=2)
    meta = buffer.insert(_make_transition(env_step=0, episode_id=0))
    assert meta.event_type == "insert"


# ---- determinism via the rng generator -----------------------------------


def test_sampling_with_seeded_generator_is_deterministic() -> None:
    """Two buffers with the same seed and the same insert sequence sample
    the same windows. This is the property the runner's resume relies on."""
    rng_a = torch.Generator(device="cpu")
    rng_a.manual_seed(2026)
    buffer_a = SequenceReplayBuffer(
        capacity=100, sequence_length=4, rng=rng_a
    )
    rng_b = torch.Generator(device="cpu")
    rng_b.manual_seed(2026)
    buffer_b = SequenceReplayBuffer(
        capacity=100, sequence_length=4, rng=rng_b
    )
    for i in range(20):
        t = _make_transition(env_step=i, episode_id=0)
        buffer_a.insert(t)
        buffer_b.insert(t)
    batch_a, _ = buffer_a.sample(batch_size=4)
    batch_b, _ = buffer_b.sample(batch_size=4)
    assert torch.equal(batch_a.obs, batch_b.obs)
    assert torch.equal(batch_a.action, batch_b.action)


# ---- properties read at construction --------------------------------------


def test_capacity_and_sequence_length_properties() -> None:
    buffer = SequenceReplayBuffer(capacity=42, sequence_length=7)
    assert buffer.capacity == 42
    assert buffer.sequence_length == 7
