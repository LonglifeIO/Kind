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
from pathlib import Path

from pydantic import BaseModel, ValidationError

__all__ = [
    "LiveEventRow",
    "LiveWindowState",
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
