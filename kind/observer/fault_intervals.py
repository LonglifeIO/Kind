"""Probe 4.5 S-TEL — the observer-side fault join.

Reconstructs per-step fault state from the granular ``energy_fault_event``
records (the fault's *only* ground truth besides ``GridState`` — no
observation marker exists by design, prereg §4). Semantics pinned to the
emission convention (``GridWorld._update_fault`` + env-server emission at
``t_event = env_step``): an ``"onset"`` at ``t`` means step ``t`` itself
decayed at the fault rate; an ``"offset"`` at ``t`` means step ``t`` decayed
at the base rate again. A fault interval is therefore the half-open step
range ``[t_onset, t_offset)``.

Eval-only, deterministic, pure: consumes event dicts (as read from the
``world_event`` sink) or typed records, produces intervals and masks.
Nothing in any Io code path imports this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "FaultInterval",
    "fault_intervals_from_events",
    "fault_mask",
]


@dataclass(frozen=True)
class FaultInterval:
    """One fault interval: active for steps ``t_onset <= t < t_offset``.

    ``t_offset`` is ``None`` for an interval still open at the end of the
    event stream (the run stopped mid-fault) — consumers treat it as
    extending to the end of whatever step range they analyze.
    """

    t_onset: int
    t_offset: int | None


def fault_intervals_from_events(
    events: Iterable[Mapping[str, Any]],
) -> tuple[FaultInterval, ...]:
    """Fold ``energy_fault_event`` records into ordered fault intervals.

    Accepts any iterable of mapping-shaped records (JSONL rows, Parquet
    rows, or ``WorldEvent.model_dump()`` outputs); non-fault event types are
    ignored, so a whole ``world_event`` stream can be passed unfiltered.
    Raises on malformed sequences (offset before onset, double onset,
    non-monotonic edges) — a broken edge stream is an analysis-stopping
    defect, not something to squint past: one dropped edge inverts every
    step's reconstructed state until the next edge.
    """
    intervals: list[FaultInterval] = []
    open_onset: int | None = None
    last_t: int | None = None
    for event in events:
        if event.get("event_type") != "energy_fault_event":
            continue
        t = int(event["t_event"])
        transition = event["payload"]["transition"]
        if last_t is not None and t < last_t:
            raise ValueError(
                f"energy_fault_event stream is not time-ordered: t={t} after "
                f"t={last_t}"
            )
        last_t = t
        if transition == "onset":
            if open_onset is not None:
                raise ValueError(
                    f"double onset: t={t} while the interval opened at "
                    f"t={open_onset} is still open"
                )
            open_onset = t
        elif transition == "offset":
            if open_onset is None:
                raise ValueError(f"offset without onset at t={t}")
            intervals.append(FaultInterval(t_onset=open_onset, t_offset=t))
            open_onset = None
        else:  # pragma: no cover — writer-side validator forbids this
            raise ValueError(f"unknown fault transition {transition!r}")
    if open_onset is not None:
        intervals.append(FaultInterval(t_onset=open_onset, t_offset=None))
    return tuple(intervals)


def fault_mask(
    intervals: Iterable[FaultInterval], *, t_start: int, t_end: int
) -> NDArray[np.bool_]:
    """Boolean per-step fault state over ``[t_start, t_end)``.

    ``mask[i]`` is the fault state of env step ``t_start + i`` — the join
    key for stratifying any per-step telemetry (the §8
    reliability-conditioning stratification and the Phase-2 gate's
    belief-error-during-fault profile both consume this).
    """
    if t_end < t_start:
        raise ValueError(f"t_end {t_end} < t_start {t_start}")
    mask = np.zeros(t_end - t_start, dtype=np.bool_)
    for interval in intervals:
        lo = max(interval.t_onset, t_start)
        hi = t_end if interval.t_offset is None else min(interval.t_offset, t_end)
        if hi > lo:
            mask[lo - t_start : hi - t_start] = True
    return mask
