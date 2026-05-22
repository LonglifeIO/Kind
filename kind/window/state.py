"""Window's view-state derivation.

The functions here turn loaded records into the small derived
structures the templates render: the four-state inference
(waking / dreaming / dormant / paused), the per-hour state-time
breakdown, the pace estimate, the run-uptime breakdown, and the
``/rounds`` and ``/audit`` row structures.

**The state-inference heuristic.** Io's state is inferred from
*telemetry write activity*, not from any explicit state-transition
event — Probe 3 has not landed, so no such events exist yet. The rule:
look at the last ``STATE_WINDOW_MS`` of write activity across the four
streams. ``agent_step`` activity means waking. ``dream_rollout``
activity *alone* means dreaming. ``replay_meta`` activity *alone* means
dormant. No activity means paused. Anything the rule cannot cleanly
resolve (``dream_rollout`` and ``replay_meta`` both active without
``agent_step``; ``world_event`` active alone) is surfaced as
``unknown`` rather than guessed. When Probe 3 lands and emits explicit
state-transition events, this presence-based heuristic can be replaced.

**The breakdown's coarseness.** The per-hour state-time breakdown
buckets per-record timestamps. ``agent_step`` and ``world_event``
carry a ``wallclock_ms`` field; ``dream_rollout`` and ``replay_meta``
do not. So a real run's breakdown distinguishes waking hours from
paused hours but cannot place dreaming or dormant hours on the
timeline. The breakdown is documented as coarse; Probe 3's
state-transition events would make it exact.

Window is interpretation-neutral: the derivations here count, bucket,
and aggregate; they do not rank, highlight, weight, or filter.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Mapping

from kind.mirror import (
    AdmissibilityVerdict,
    LLMCallRecord,
    RoundJudgment,
    RoundResult,
)
from kind.window import loaders
from kind.window.loaders import Loaded

__all__ = [
    "STATE_WINDOW_MS",
    "CriterionVerdict",
    "IoState",
    "LatencyRow",
    "OverviewState",
    "RoundRow",
    "StreamActivity",
    "bucket_activity_by_hour",
    "build_overview",
    "build_round_rows",
    "decide_state",
    "format_duration",
    "group_admissibility_for_round",
    "infer_current_state",
    "latency_distribution",
    "pace_estimate",
    "parse_run_start",
]


#: The window the state-inference heuristic looks back over — five
#: minutes, per the Phase 11.5 spec. A stream with a write more recent
#: than this is "active".
STATE_WINDOW_MS: int = 5 * 60 * 1000

_HOUR_MS: int = 60 * 60 * 1000

_AGENT_STEP = "agent_step"
_DREAM_ROLLOUT = "dream_rollout"
_REPLAY_META = "replay_meta"
_WORLD_EVENT = "world_event"

#: The three streams that determine a state. ``world_event`` is
#: auxiliary — it accompanies waking but does not name a state on its
#: own.
_STATE_STREAMS: frozenset[str] = frozenset(
    {_AGENT_STEP, _DREAM_ROLLOUT, _REPLAY_META}
)


class IoState(str, Enum):
    """The four states the heuristic resolves, plus ``UNKNOWN``.

    ``(str, Enum)`` gives JSON-friendly, template-friendly values.
    """

    WAKING = "waking"
    DREAMING = "dreaming"
    DORMANT = "dormant"
    PAUSED = "paused"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# State inference.
# ---------------------------------------------------------------------------


def decide_state(active_streams: frozenset[str]) -> IoState:
    """Resolve a state from the set of streams with recent write
    activity. Pure — the disk-touching parts live in the callers.

    The precedence: ``agent_step`` present → waking (it dominates;
    waking writes the other streams too). Then ``dream_rollout`` alone
    → dreaming, ``replay_meta`` alone → dormant. No state-stream
    activity and no activity at all → paused. No state-stream activity
    but *some* activity (a lone ``world_event``) → unknown. Both
    ``dream_rollout`` and ``replay_meta`` without ``agent_step`` →
    unknown — the heuristic cannot cleanly resolve it.
    """
    state_streams = active_streams & _STATE_STREAMS
    if _AGENT_STEP in state_streams:
        return IoState.WAKING
    if state_streams == frozenset({_DREAM_ROLLOUT}):
        return IoState.DREAMING
    if state_streams == frozenset({_REPLAY_META}):
        return IoState.DORMANT
    if not state_streams:
        return IoState.PAUSED if not active_streams else IoState.UNKNOWN
    return IoState.UNKNOWN


def infer_current_state(
    last_write_ms: Mapping[str, int | None],
    *,
    now_ms: int,
    window_ms: int = STATE_WINDOW_MS,
) -> IoState:
    """Infer Io's current state from each stream's last-write time.

    ``last_write_ms`` maps a stream name to the epoch-ms of its most
    recent write, or ``None`` for a stream with no files. A stream is
    "active" when its last write is within ``window_ms`` of ``now_ms``.
    """
    active = frozenset(
        name
        for name, ts in last_write_ms.items()
        if ts is not None and now_ms - ts <= window_ms
    )
    return decide_state(active)


# ---------------------------------------------------------------------------
# Per-hour state-time breakdown.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamActivity:
    """Per-stream record-write timestamps (epoch ms) for the per-hour
    breakdown. ``dream_rollout`` and ``replay_meta`` carry no per-record
    wallclock, so a real run leaves those tuples empty (see the module
    docstring on the breakdown's coarseness)."""

    agent_step: tuple[int, ...] = ()
    dream_rollout: tuple[int, ...] = ()
    replay_meta: tuple[int, ...] = ()
    world_event: tuple[int, ...] = ()


def bucket_activity_by_hour(
    activity: StreamActivity,
    *,
    now_ms: int,
    window_hours: int,
) -> dict[IoState, int]:
    """Bucket the last ``window_hours`` into per-hour states.

    Hour ``h`` (``0`` is the most recent) covers
    ``[now_ms - (h + 1) * hour, now_ms - h * hour)``. Each hour's state
    is :func:`decide_state` applied to the streams with at least one
    timestamp in that hour. Returns a count per :class:`IoState`; the
    counts sum to ``window_hours``.
    """
    counts: dict[IoState, int] = {state: 0 for state in IoState}
    streams: dict[str, tuple[int, ...]] = {
        _AGENT_STEP: activity.agent_step,
        _DREAM_ROLLOUT: activity.dream_rollout,
        _REPLAY_META: activity.replay_meta,
        _WORLD_EVENT: activity.world_event,
    }
    for h in range(window_hours):
        hi = now_ms - h * _HOUR_MS
        lo = hi - _HOUR_MS
        active = frozenset(
            name
            for name, timestamps in streams.items()
            if any(lo <= t < hi for t in timestamps)
        )
        counts[decide_state(active)] += 1
    return counts


# ---------------------------------------------------------------------------
# Pace.
# ---------------------------------------------------------------------------


def pace_estimate(
    episode_events: Iterable[tuple[int, int]],
    *,
    now_ms: int,
    window_hours: int = 24,
) -> float:
    """Episodes per hour over the last ``window_hours``.

    ``episode_events`` is an iterable of ``(episode_id, wallclock_ms)``
    pairs — one per ``agent_step`` row. The estimate is the count of
    distinct episode ids with at least one row inside the window,
    divided by ``window_hours``.
    """
    window_ms = window_hours * _HOUR_MS
    seen: set[int] = {
        episode_id
        for episode_id, wallclock_ms in episode_events
        if now_ms - window_ms <= wallclock_ms <= now_ms
    }
    return len(seen) / window_hours


# ---------------------------------------------------------------------------
# Run-start parsing and duration formatting.
# ---------------------------------------------------------------------------


_RUN_TIMESTAMP_RE = re.compile(r"(\d{8})-(\d{6})$")


def parse_run_start(run_dir_name: str) -> datetime | None:
    """Parse a run's start time from the trailing ``YYYYMMDD-HHMMSS`` in
    its directory name (e.g. ``probe1-20260503-123926``). A name with no
    such suffix (e.g. ``phase_13_calibration``) yields ``None``."""
    match = _RUN_TIMESTAMP_RE.search(run_dir_name)
    if match is None:
        return None
    try:
        return datetime.strptime(
            f"{match.group(1)}-{match.group(2)}", "%Y%m%d-%H%M%S"
        )
    except ValueError:
        return None


def format_duration(ms: int) -> str:
    """Render a millisecond duration as a coarse ``Nd Nh Nm Ns`` string.
    A negative input yields ``"unknown"``."""
    if ms < 0:
        return "unknown"
    total_seconds = ms // 1000
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Overview.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OverviewState:
    """The derived state the ``/`` overview renders."""

    run_id: str
    run_dir: Path
    start_time: datetime | None
    has_telemetry: bool
    current_state: IoState
    uptime_ms: int | None
    uptime_human: str
    total_steps: int
    total_episodes: int
    pace_episodes_per_hour: float
    breakdown_24h: dict[IoState, int]
    breakdown_7d: dict[IoState, int]
    telemetry_errors: tuple[str, ...]


def build_overview(
    run_dir: Path, run_id: str, *, now_ms: int
) -> OverviewState:
    """Load the run's telemetry and derive the overview state.

    A run with no ``telemetry/`` directory (a mirror-only calibration
    run) has its current state surfaced as ``unknown`` — the heuristic
    has nothing to read, and ``paused`` would falsely imply an Io
    process that simply stopped.
    """
    start_time = parse_run_start(run_dir.name)
    has_telemetry = loaders.telemetry_dir_exists(run_dir)

    if has_telemetry:
        current_state = infer_current_state(
            loaders.stream_last_write_ms(run_dir), now_ms=now_ms
        )
    else:
        current_state = IoState.UNKNOWN

    step_outcomes = loaders.load_agent_steps(run_dir)
    world_outcomes = loaders.load_world_events(run_dir)

    steps = [o.value for o in step_outcomes if o.value is not None]
    world_events = [o.value for o in world_outcomes if o.value is not None]
    telemetry_errors = tuple(
        o.error for o in step_outcomes if o.error is not None
    ) + tuple(
        o.error for o in world_outcomes if o.error is not None
    )

    total_steps = len(steps)
    total_episodes = len({s.episode_id for s in steps})
    episode_events = [(s.episode_id, s.wallclock_ms) for s in steps]
    pace = pace_estimate(episode_events, now_ms=now_ms)

    activity = StreamActivity(
        agent_step=tuple(s.wallclock_ms for s in steps),
        world_event=tuple(e.wallclock_ms for e in world_events),
    )
    breakdown_24h = bucket_activity_by_hour(
        activity, now_ms=now_ms, window_hours=24
    )
    breakdown_7d = bucket_activity_by_hour(
        activity, now_ms=now_ms, window_hours=24 * 7
    )

    uptime_ms: int | None = None
    if start_time is not None:
        uptime_ms = now_ms - int(start_time.timestamp() * 1000)
    elif steps:
        uptime_ms = now_ms - min(s.wallclock_ms for s in steps)
    uptime_human = (
        format_duration(uptime_ms) if uptime_ms is not None else "unknown"
    )

    return OverviewState(
        run_id=run_id,
        run_dir=run_dir,
        start_time=start_time,
        has_telemetry=has_telemetry,
        current_state=current_state,
        uptime_ms=uptime_ms,
        uptime_human=uptime_human,
        total_steps=total_steps,
        total_episodes=total_episodes,
        pace_episodes_per_hour=pace,
        breakdown_24h=breakdown_24h,
        breakdown_7d=breakdown_7d,
        telemetry_errors=telemetry_errors,
    )


# ---------------------------------------------------------------------------
# Rounds.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CriterionVerdict:
    """One criterion's verdict from a matching :class:`RoundJudgment`."""

    criterion_id: str
    verdict: str
    confidence: float


@dataclass(frozen=True)
class RoundRow:
    """One row of the ``/rounds`` list."""

    round_id: str
    file_name: str
    mtime_ms: int
    checkpoint_ids: tuple[str, ...]
    n_passes: int
    has_judgment: bool
    verdicts: tuple[CriterionVerdict, ...]
    error: str | None


def build_round_rows(
    round_outcomes: list[Loaded[RoundResult]],
    judgment_outcomes: list[Loaded[RoundJudgment]],
) -> list[RoundRow]:
    """Derive the ``/rounds`` rows from loaded rounds and judgments.

    A round is joined to a judgment by ``round_id``. A round that failed
    to deserialize becomes a row carrying its error (and its file stem
    as a stand-in ``round_id``) so the failure is visible rather than
    silently dropped.
    """
    judgments_by_round: dict[str, RoundJudgment] = {
        o.value.round_id: o.value
        for o in judgment_outcomes
        if o.value is not None
    }
    rows: list[RoundRow] = []
    for outcome in round_outcomes:
        if outcome.value is None:
            rows.append(
                RoundRow(
                    round_id=outcome.path.stem,
                    file_name=outcome.path.name,
                    mtime_ms=outcome.mtime_ms,
                    checkpoint_ids=(),
                    n_passes=0,
                    has_judgment=False,
                    verdicts=(),
                    error=outcome.error,
                )
            )
            continue
        result = outcome.value
        judgment = judgments_by_round.get(result.round_id)
        verdicts: tuple[CriterionVerdict, ...] = ()
        if judgment is not None:
            verdicts = tuple(
                CriterionVerdict(
                    criterion_id=cj.criterion_id,
                    verdict=cj.verdict,
                    confidence=cj.confidence,
                )
                for cj in judgment.criterion_judgments
            )
        rows.append(
            RoundRow(
                round_id=result.round_id,
                file_name=outcome.path.name,
                mtime_ms=outcome.mtime_ms,
                checkpoint_ids=tuple(
                    c.checkpoint_id for c in result.round_config.checkpoints
                ),
                n_passes=len(result.pass_results),
                has_judgment=judgment is not None,
                verdicts=verdicts,
                error=None,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Per-reading admissibility verdicts, joined to a round's passes.
# ---------------------------------------------------------------------------


def group_admissibility_for_round(
    round_result: RoundResult,
    admissibility_outcomes: list[Loaded[AdmissibilityVerdict]],
) -> list[tuple[AdmissibilityVerdict, ...]]:
    """Group the run's admissibility verdicts by the round's passes.

    Returns a list parallel to ``round_result.pass_results``: index
    ``i`` carries the :class:`AdmissibilityVerdict` records whose
    ``(pass_index, checkpoint_id)`` matches pass ``i``. The verdict's
    ``pass_index`` is matched against the pass's position in the round;
    the ``checkpoint_id`` disambiguates verdicts from other rounds in
    the same run's ``admissibility.jsonl``.

    The grouping is non-prescriptive — it joins, it does not rank or
    filter. A pass with no matching verdict gets an empty tuple. Verdict
    records that failed to deserialize are skipped (their error surfaces
    via the loader's :class:`Loaded` outcome elsewhere)."""
    verdicts = [
        o.value for o in admissibility_outcomes if o.value is not None
    ]
    per_pass: list[tuple[AdmissibilityVerdict, ...]] = []
    for index, pass_result in enumerate(round_result.pass_results):
        per_pass.append(
            tuple(
                v
                for v in verdicts
                if v.pass_index == index
                and v.checkpoint_id == pass_result.checkpoint_id
            )
        )
    return per_pass


# ---------------------------------------------------------------------------
# LLM-call audit.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencyRow:
    """Latency distribution for one ``(role, checkpoint_id)`` group."""

    role: str
    checkpoint_id: str
    n: int
    min_ms: int
    max_ms: int
    median_ms: float
    mean_ms: float


def latency_distribution(
    records: Iterable[LLMCallRecord],
) -> list[LatencyRow]:
    """Per-``(role, checkpoint)`` latency distribution across LLM-call
    records. Records with no ``latency_ms`` (the synthetic
    ``max_retries_exceeded`` records) are skipped. Rows are returned in
    sorted ``(role, checkpoint_id)`` order."""
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for record in records:
        if record.latency_ms is not None:
            groups[(record.role, record.checkpoint_id)].append(
                record.latency_ms
            )
    rows: list[LatencyRow] = []
    for (role, checkpoint_id), latencies in sorted(groups.items()):
        rows.append(
            LatencyRow(
                role=role,
                checkpoint_id=checkpoint_id,
                n=len(latencies),
                min_ms=min(latencies),
                max_ms=max(latencies),
                median_ms=float(median(latencies)),
                mean_ms=float(mean(latencies)),
            )
        )
    return rows
