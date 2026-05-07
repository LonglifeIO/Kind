"""Phase 5 sequence replay buffer.

A FIFO ring of single-step transitions, sampled in fixed-length windows
that respect episode boundaries. The implementation plan §2.9 and the
synthesis §Q3 specify this minimal shape at Probe 1; prioritized and
curious replay are deferred to Probe 3+ via the nullable ``priority``
field on :class:`~kind.observer.schemas.ReplayMeta`.

The buffer constructs and emits a :class:`ReplayMeta` record on every
insert (``event_type="insert"``), every sample (``event_type="sample"``),
and every eviction (``event_type="evict"``). The runner (Phase 5)
supplies a callable handler at construction (typically a
:class:`~kind.observer.sinks.JsonlSink`'s ``write`` method); the buffer
invokes it synchronously for each event. ``insert`` and ``sample`` also
return the primary record so callers that want the meta inline can read
it without re-routing through the handler.

**Episode-boundary respect.** Each :class:`Transition` carries the
``episode_id`` of the *from* observation (the obs the action was based
on). A sequence-length-``L`` window starting at buffer index ``s`` is
valid iff every transition in ``[s, s+L)`` shares the same
``episode_id``. With Probe 1's defaults (``episode_length=200``,
``sequence_length=32``), the rejection rate is roughly
``L / episode_length`` ≈ 16% — a small constant overhead, well-behaved.

**Tensors stored as-is.** The buffer does not copy or move the
:class:`Transition`'s ``obs``/``next_obs`` tensors. The runner is
responsible for keeping them on a memory-budgeted device (CPU at Probe 1
since the trainer's full-scale capacity is 100k transitions). The
runner converts numpy uint8 observations from the transport into
float32 ``(1, 32, 32)`` tensors at runner-side, so a sample's
:class:`Batch` is already in the dtype the world model expects; the
caller moves to the training device.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from kind.observer.schemas import PROBE_1_SCHEMA_VERSION, ReplayMeta

__all__ = [
    "Batch",
    "ReplayMetaHandler",
    "SequenceReplayBuffer",
    "Transition",
]


ReplayMetaHandler = Callable[[ReplayMeta], None]
"""Callable that consumes a single :class:`ReplayMeta` record.

The buffer invokes the configured handler synchronously for every insert,
sample, and eviction event, in chronological order. Phase 5's runner
wires this to a :class:`~kind.observer.sinks.JsonlSink`'s ``write``
method so the records land in the run's ``replay_meta`` stream.
"""


@dataclass(frozen=True)
class Transition:
    """One env-step transition the runner inserts into the buffer.

    The ``episode_id`` and ``step_in_episode`` fields refer to the *from*
    observation — the one ``action`` was sampled against. The buffer
    uses ``episode_id`` to reject boundary-crossing sample windows.
    """

    obs: Tensor
    action: int
    next_obs: Tensor
    env_step: int
    episode_id: int
    step_in_episode: int


@dataclass(frozen=True)
class Batch:
    """A batched sequence sample.

    Shapes: ``obs``/``next_obs`` are ``(B, L, *obs_shape)``; ``action`` is
    ``(B, L)`` long. The runner stacks transitions inside ``sample`` —
    the buffer is responsible for shape consistency, the caller for any
    device transfer.
    """

    obs: Tensor
    action: Tensor
    next_obs: Tensor


class SequenceReplayBuffer:
    """FIFO buffer of single-step transitions with sequence-length sampling.

    The buffer is parameterized by ``capacity`` (max stored transitions)
    and ``sequence_length`` (the ``L`` of each sampled window). It is
    not parameterized by ``batch_size`` — that is a per-call argument to
    ``sample`` so the runner can vary it without rebuilding the buffer.

    Determinism is controlled by the optional ``rng`` argument: a
    :class:`torch.Generator` whose state is driven by the runner's
    checkpointable RNG. When ``rng`` is ``None``, sampling uses
    PyTorch's default generator (whose state is also checkpointed by the
    runner via ``torch.get_rng_state``).
    """

    def __init__(
        self,
        capacity: int,
        sequence_length: int = 32,
        run_id: str = "",
        replay_meta_handler: ReplayMetaHandler | None = None,
        rng: torch.Generator | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if sequence_length <= 0:
            raise ValueError(
                f"sequence_length must be positive, got {sequence_length}"
            )
        self._capacity = capacity
        self._L = sequence_length
        self._run_id = run_id
        self._handler = replay_meta_handler
        self._rng = rng
        self._checkpoint_id: str | None = None

        # The deque gives O(1) append + popleft; conversion to a list at
        # sample time gives O(N) one-time access for the K=batch_size
        # window slices.
        self._transitions: deque[Transition] = deque()
        self._total_inserts: int = 0
        self._total_evicts: int = 0
        self._total_samples: int = 0

    # ---- size queries ---------------------------------------------------

    def __len__(self) -> int:
        return len(self._transitions)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def sequence_length(self) -> int:
        return self._L

    def can_sample(self, batch_size: int) -> bool:
        """``True`` if at least ``batch_size`` valid windows are available.

        A "valid window" is a sequence-length-``L`` contiguous slice that
        does not cross an episode boundary.
        """
        return self._count_valid_windows() >= batch_size

    # ---- envelope plumbing ----------------------------------------------

    def set_checkpoint_id(self, checkpoint_id: str | None) -> None:
        """Set the ``checkpoint_id`` envelope on subsequently-emitted records.

        The runner calls this after a successful checkpoint commit so all
        future insert/sample/evict records carry the new checkpoint id.
        Mirrors the symmetric setter on :class:`~kind.env.env_server.EnvServer`.
        """
        self._checkpoint_id = checkpoint_id

    # ---- main API -------------------------------------------------------

    def insert(self, transition: Transition) -> ReplayMeta:
        """Append ``transition``; evict the oldest if at capacity.

        If eviction occurs, an ``"evict"`` record is emitted via the
        handler before the new transition is appended; then the
        ``"insert"`` record is emitted (and returned). Both records'
        ``buffer_size`` and ``total_segments`` reflect the buffer state
        *after* the operation each describes.
        """
        if len(self._transitions) >= self._capacity:
            evicted = self._transitions.popleft()
            self._total_evicts += 1
            evict_meta = self._build_meta(
                event_type="evict",
                t_event=evicted.env_step,
                segment_id=self._total_evicts,
                segment_start=evicted.env_step,
                segment_end=evicted.env_step,
            )
            self._dispatch(evict_meta)

        self._transitions.append(transition)
        self._total_inserts += 1
        insert_meta = self._build_meta(
            event_type="insert",
            t_event=transition.env_step,
            segment_id=self._total_inserts,
            segment_start=transition.env_step,
            segment_end=transition.env_step,
        )
        self._dispatch(insert_meta)
        return insert_meta

    def sample(self, batch_size: int) -> tuple[Batch, ReplayMeta]:
        """Sample ``batch_size`` valid windows and emit a ``"sample"`` record.

        Sampling is uniform-without-replacement over the set of valid
        window starts. Raises :class:`RuntimeError` if fewer than
        ``batch_size`` valid windows exist; the runner is expected to
        gate the call with :meth:`can_sample`.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        valid_starts = self._valid_window_starts()
        if len(valid_starts) < batch_size:
            raise RuntimeError(
                "not enough valid windows to sample "
                f"batch_size={batch_size}; have {len(valid_starts)} "
                f"(buffer_size={len(self._transitions)})"
            )

        # Materialize the deque as a list once for O(1) indexed access.
        snapshot = list(self._transitions)

        # Sample without replacement: take the first ``batch_size``
        # entries of a random permutation.
        if self._rng is None:
            perm = torch.randperm(len(valid_starts))
        else:
            perm = torch.randperm(len(valid_starts), generator=self._rng)
        chosen = [valid_starts[int(perm[i].item())] for i in range(batch_size)]

        L = self._L
        obs_seqs: list[Tensor] = []
        next_obs_seqs: list[Tensor] = []
        action_seqs: list[Tensor] = []
        for s in chosen:
            window = snapshot[s : s + L]
            obs_seqs.append(torch.stack([t.obs for t in window], dim=0))
            next_obs_seqs.append(torch.stack([t.next_obs for t in window], dim=0))
            action_seqs.append(
                torch.tensor([t.action for t in window], dtype=torch.long)
            )

        batch = Batch(
            obs=torch.stack(obs_seqs, dim=0),
            action=torch.stack(action_seqs, dim=0),
            next_obs=torch.stack(next_obs_seqs, dim=0),
        )

        starts_env = [snapshot[s].env_step for s in chosen]
        ends_env = [snapshot[s + L - 1].env_step for s in chosen]
        latest_env = snapshot[-1].env_step
        self._total_samples += 1
        sample_meta = self._build_meta(
            event_type="sample",
            t_event=latest_env,
            segment_id=self._total_samples,
            segment_start=min(starts_env),
            segment_end=max(ends_env),
        )
        self._dispatch(sample_meta)
        return batch, sample_meta

    # ---- internals ------------------------------------------------------

    def _valid_window_starts(self) -> list[int]:
        """Indices ``s`` in ``[0, n-L]`` whose window is single-episode.

        O(N) via a forward scan that records episode-boundary positions
        and then admits ``s`` iff no boundary lies in ``(s, s+L)``.
        """
        n = len(self._transitions)
        L = self._L
        if n < L:
            return []

        episodes = [t.episode_id for t in self._transitions]
        # Boundary positions: indices where episode_id flips.
        boundaries: list[int] = [
            i for i in range(1, n) if episodes[i] != episodes[i - 1]
        ]

        starts: list[int] = []
        bi = 0
        max_start = n - L
        for s in range(max_start + 1):
            # Advance bi past any boundary at index <= s (those don't
            # affect the window starting at s+1).
            while bi < len(boundaries) and boundaries[bi] <= s:
                bi += 1
            # If the next boundary is >= s+L, the window [s, s+L) has
            # no internal boundary and is valid.
            if bi == len(boundaries) or boundaries[bi] >= s + L:
                starts.append(s)
        return starts

    def _count_valid_windows(self) -> int:
        return len(self._valid_window_starts())

    def _build_meta(
        self,
        *,
        event_type: str,
        t_event: int,
        segment_id: int,
        segment_start: int,
        segment_end: int,
    ) -> ReplayMeta:
        # ``total_segments`` and ``buffer_size`` reflect post-event state.
        return ReplayMeta(
            schema_version=PROBE_1_SCHEMA_VERSION,
            run_id=self._run_id,
            checkpoint_id=self._checkpoint_id,
            event_type=event_type,  # type: ignore[arg-type]
            t_event=t_event,
            segment_id=segment_id,
            segment_start=segment_start,
            segment_end=segment_end,
            priority=None,
            buffer_size=len(self._transitions),
            total_segments=self._count_valid_windows(),
        )

    def _dispatch(self, record: ReplayMeta) -> None:
        if self._handler is not None:
            self._handler(record)
