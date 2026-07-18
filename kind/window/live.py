"""Window's live-state surface — the "watch the experiment" view's data.

A long run's driver script (the Probe 4 biography) writes a single
small JSON snapshot, ``runs/<run-id>/window/live_state.json``,
atomically on every waking step via :func:`write_live_state`. Window's
``/live`` route reads it back through :class:`LiveWindowState` (the
house rule: records are read through Pydantic models, never ad-hoc
dicts) and renders the grid, Io's position, and the recent world-event
feed.

Roles are deliberately split: :func:`write_live_state` is the
**runner-side producer utility** — colocated here so writer and reader
share one model — and is never called by the Flask app, which stays
read-only. The snapshot is ephemeral view-state (overwritten in place,
never appended); it is *not* telemetry and carries no analysis
authority — the four telemetry streams remain the record.

The event feed shows ``source`` ground truth (builder / environment)
because the Window is the *builder's* eye. Io's observation carries no
such marker; the asymmetry is the Probe 4 affordance, and displaying
it here changes nothing Io can see.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:  # torch-free at runtime: the Window app never imports the
    # runner; the callback is duck-typed against RunnerStepInfo's fields.
    from kind.env.env_server import EnvServer
    from kind.training.runner import RunnerStepInfo

__all__ = [
    "LiveEventRow",
    "LiveWindowState",
    "LiveStateWriter",
    "live_state_path",
    "load_live_state",
    "write_live_state",
]


class LiveEventRow(BaseModel):
    """One recent world event, as the builder's eye sees it."""

    t_event: int
    source: str
    event_type: str
    detail: str = ""


class LiveWindowState(BaseModel):
    """One waking-step snapshot of the observable run state."""

    run_id: str
    env_step: int
    episode_id: int
    step_in_episode: int
    wallclock_ms: int
    grid: list[list[int]]
    agent_pos: tuple[int, int]
    true_energy: float | None = None
    recent_events: list[LiveEventRow] = []


def live_state_path(run_dir: Path) -> Path:
    return run_dir / "window" / "live_state.json"


def load_live_state(run_dir: Path) -> LiveWindowState | str | None:
    """Read the snapshot. ``None`` when absent (no live run has written
    one); an error string when present but unreadable (surfaced in the
    page, per the house malformed-record rule)."""
    path = live_state_path(run_dir)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return LiveWindowState.model_validate(json.load(handle))
    except (OSError, ValueError, ValidationError) as error:
        return f"{type(error).__name__}: {error}"


def write_live_state(run_dir: Path, state: LiveWindowState) -> None:
    """Runner-side producer: atomic tmp-write + rename so the reader
    never sees a torn snapshot. Not called by the Window app."""
    path = live_state_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=".live_state_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(state.model_dump_json())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


class LiveStateWriter:
    """Step callback: one atomic ``window/live_state.json`` snapshot per
    waking step — promoted from the Probe 4 biography script (2026-07-18)
    so every run script gets the ``/live`` view in three lines. The
    recent-event feed is tailed from the run's own ``world_event.jsonl``
    (line-flushed by the telemetry sink) — the transport layer owns the
    in-process event handler chain (``EnvTransportServer`` replaces the
    config-time handler), so the flushed record IS the reliable source.
    Builder-eye only — writes outside telemetry, touches nothing Io sees.

    Also writes the per-step position sidecar ``agent_pos.jsonl`` (a
    run-script record, not telemetry — the derived-feed precedent; feeds
    the boundary analyzer's occupancy-share diagnostic).
    """

    _TAIL_BYTES = 16_384
    _CONSUMPTION_JUMP = 0.03  # house threshold (seek-classification §1)

    def __init__(
        self,
        env_server: "EnvServer",
        run_dir: Path,
        *,
        run_id: str,
        event_feed_len: int = 20,
        print_every: int = 1_000,
    ) -> None:
        self._env_server = env_server
        self._run_dir = run_dir
        self._run_id = run_id
        self._event_feed_len = event_feed_len
        self._print_every = print_every
        self._started = time.monotonic()
        self._prev_energy: float | None = None
        self._derived_rows: list[LiveEventRow] = []
        run_dir.mkdir(parents=True, exist_ok=True)  # fresh runs: dir first
        self._pos_log = (run_dir / "agent_pos.jsonl").open(
            "a", buffering=1, encoding="utf-8"
        )

    def _recent_events(self) -> list[LiveEventRow]:
        path = self._run_dir / "telemetry" / "world_event.jsonl"
        try:
            size = path.stat().st_size
        except OSError:
            return []
        with path.open("rb") as handle:
            handle.seek(max(0, size - self._TAIL_BYTES))
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
        if size > self._TAIL_BYTES and lines:
            lines = lines[1:]  # drop the partial first line
        from kind.observer.schemas import WorldEvent  # torch-free

        rows: list[LiveEventRow] = []
        for line in lines[-self._event_feed_len :]:
            try:
                record = WorldEvent.model_validate_json(line)
            except ValueError:
                continue
            payload = record.payload
            parts = [
                f"{key}={payload[key]}"
                for key in (
                    "mutator",
                    "process",
                    "cell",
                    "trigger",
                    "transition",
                )
                if key in payload
            ]
            rows.append(
                LiveEventRow(
                    t_event=record.t_event,
                    source=record.source,
                    event_type=record.event_type,
                    detail="  ".join(parts),
                )
            )
        return rows

    def _track_consumption(self, env_step: int, energy: float) -> None:
        """Derived builder-eye row: a consumption's energy jump made
        visible in the feed (the grid pixel often lives < one poll).
        View-state only — never telemetry."""
        prev = self._prev_energy
        self._prev_energy = energy
        if prev is not None and energy - prev > self._CONSUMPTION_JUMP:
            self._derived_rows.append(
                LiveEventRow(
                    t_event=env_step,
                    source="io",
                    event_type="consumption (derived)",
                    detail=f"energy +{energy - prev:.3f}",
                )
            )
            del self._derived_rows[: -self._event_feed_len]

    def __call__(self, info: "RunnerStepInfo") -> None:
        state = self._env_server.grid_world_state
        self._pos_log.write(
            f'{{"t": {info.env_step}, "pos": [{int(state.agent_pos[0])}, '
            f"{int(state.agent_pos[1])}]}}\n"
        )
        self._track_consumption(info.env_step, float(state.true_energy))
        feed = sorted(
            self._recent_events() + self._derived_rows,
            key=lambda row: row.t_event,
        )[-self._event_feed_len :]
        write_live_state(
            self._run_dir,
            LiveWindowState(
                run_id=self._run_id,
                env_step=info.env_step,
                episode_id=info.episode_id,
                step_in_episode=info.step_in_episode,
                wallclock_ms=int(time.time() * 1000),
                grid=[[int(v) for v in row] for row in state.grid],
                agent_pos=(int(state.agent_pos[0]), int(state.agent_pos[1])),
                true_energy=float(state.true_energy),
                recent_events=feed,
            ),
        )
        if (
            self._print_every > 0
            and info.env_step > 0
            and info.env_step % self._print_every == 0
        ):
            elapsed = time.monotonic() - self._started
            print(
                f"step {info.env_step}  episode {info.episode_id}  "
                f"{elapsed:.0f}s  ({elapsed / info.env_step * 1000:.0f} ms/step)",
                flush=True,
            )
