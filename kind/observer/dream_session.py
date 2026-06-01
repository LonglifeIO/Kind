"""Probe 3 dream-session metadata model and JSONL sink.

Dream-session metadata is append-only, one JSON object per line at
``runs/{run_id}/telemetry/dream_session.jsonl``. The same
``dream_session_id`` is written twice per session: once on session start with
closure fields left ``None``, then again on session end with the closure fields
populated. That double-write pattern keeps in-flight sessions observable without
forcing a mutating update protocol onto the telemetry directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Literal, Self

from pydantic import BaseModel, ConfigDict

DREAM_SESSION_META_SCHEMA_VERSION: str = "0.1.0"
DREAM_SESSION_FILE: str = "dream_session.jsonl"


class DreamSessionMeta(BaseModel):
    """One record per dream-session lifecycle edge (start, then end)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1.0"]
    run_id: str
    checkpoint_id: str | None
    dream_session_id: str
    started_at_env_step: int
    started_at_wallclock_ms: int
    ended_at_env_step: int | None
    ended_at_wallclock_ms: int | None
    end_trigger: str | None
    rollout_count: int
    envelope_config_snapshot: dict[str, Any]
    seed_selection_config_snapshot: dict[str, Any]


class DreamSessionSinkClosedError(RuntimeError):
    """Raised when ``write()`` is called after the sink has been closed."""


class DreamSessionSink:
    """Append-only JSONL writer for :class:`DreamSessionMeta` records."""

    def __init__(self, dir: Path) -> None:
        self._dir = dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = dir / DREAM_SESSION_FILE
        self._file: IO[str] | None = self._path.open("a", encoding="utf-8")
        self._closed = False

    @property
    def path(self) -> Path:
        return self._path

    def write(self, record: DreamSessionMeta) -> None:
        if self._closed or self._file is None:
            raise DreamSessionSinkClosedError(
                f"DreamSessionSink at {self._path} is closed"
            )
        self._file.write(record.model_dump_json() + "\n")

    def close(self) -> None:
        if self._closed:
            return
        if self._file is not None:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            self._file = None
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


__all__ = [
    "DREAM_SESSION_META_SCHEMA_VERSION",
    "DREAM_SESSION_FILE",
    "DreamSessionMeta",
    "DreamSessionSink",
    "DreamSessionSinkClosedError",
]
