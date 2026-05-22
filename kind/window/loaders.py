"""Window's record loaders.

Every function here reads on-disk records *through their Pydantic
models* — Window never parses JSON into ad-hoc dicts and renders those.
If a future phase changes a model's shape, Window inherits the change
for free. If a record fails to deserialize, the loader surfaces the
error (wrapped in a :class:`Loaded` outcome) rather than papering over
it; the views render the error.

The loaders are read-only. Every file is opened in a read mode; no
loader writes, appends, or truncates anything. The parquet reads go
through :mod:`pyarrow`, which opens files in its own C++ layer — the
Phase 8 ``Path.open`` write-tracking test does not see those, and they
are read-only regardless.

The models Window reads through are imported from the ``kind.mirror``
package surface and from :mod:`kind.observer.schemas`. Window
deliberately does *not* import from ``kind.mirror.orchestrator``,
``kind.mirror.judge_driver``, or ``kind.mirror.calibration.smoke`` — it
reads records, it does not construct or trigger any of the mirror's
production code paths. The models it needs (:class:`PassResult` lives
in the orchestrator module; :class:`RoundResult` in the round driver;
:class:`RoundJudgment` in the judge module) are all re-exported from
``kind.mirror``, so Window reaches them without naming the production
modules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import pyarrow.parquet as pq
from pydantic import BaseModel, ValidationError

from kind.mirror import (
    AdmissibilityVerdict,
    FaithfulnessResult,
    LLMCallAudit,
    LLMCallRecord,
    PassResult,
    RoundJudgment,
    RoundResult,
    StabilityResult,
)
from kind.observer.schemas import AgentStep, DreamRollout, ReplayMeta, WorldEvent

__all__ = [
    "Loaded",
    "LoaderError",
    "aggregate_llm_audit",
    "load_admissibility_records",
    "load_agent_steps",
    "load_dream_rollouts",
    "load_faithfulness_results",
    "load_pass_results",
    "load_replay_meta",
    "load_round_judgments",
    "load_round_results",
    "load_stability_results",
    "load_world_events",
    "stream_last_write_ms",
    "telemetry_dir_exists",
]


ModelT = TypeVar("ModelT", bound=BaseModel)

#: The four telemetry streams, in the synthesis's stream order.
TELEMETRY_STREAMS: tuple[str, ...] = (
    "agent_step",
    "dream_rollout",
    "replay_meta",
    "world_event",
)


class LoaderError(RuntimeError):
    """Raised when an on-disk record cannot be deserialized.

    The directory- and JSONL-level loaders catch this and fold it into a
    :class:`Loaded` outcome so one bad file does not break the view; the
    single-file ``_read_json_model`` helper raises it.
    """


@dataclass(frozen=True)
class Loaded(Generic[ModelT]):
    """One load outcome — either a deserialized model or an error.

    ``value`` is the model on success and ``None`` on failure; ``error``
    is ``None`` on success and a human-readable message on failure.
    Exactly one of the two is set. ``mtime_ms`` is the source file's
    modification time (epoch ms) — the ``/rounds`` view sorts by it.
    """

    path: Path
    mtime_ms: int
    value: ModelT | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Low-level readers.
# ---------------------------------------------------------------------------


def _mtime_ms(path: Path) -> int:
    return int(path.stat().st_mtime * 1000)


def _read_json_model(path: Path, model: type[ModelT]) -> ModelT:
    """Read one JSON file and deserialize it through ``model``.

    Raises :class:`LoaderError` on a read failure or a deserialization
    failure — the caller folds it into a :class:`Loaded` outcome.
    """
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise LoaderError(
            f"{path.name}: could not read JSON ({exc!r})."
        ) from exc
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise LoaderError(
            f"{path.name}: does not deserialize as {model.__name__} "
            f"({exc.error_count()} validation error(s))."
        ) from exc


def _load_json_dir(
    directory: Path, model: type[ModelT]
) -> list[Loaded[ModelT]]:
    """Load every ``*.json`` file in ``directory`` through ``model``,
    sorted by file modification time descending (most recent first).
    A missing directory yields an empty list."""
    if not directory.is_dir():
        return []
    outcomes: list[Loaded[ModelT]] = []
    for path in sorted(directory.glob("*.json")):
        mtime_ms = _mtime_ms(path)
        try:
            value = _read_json_model(path, model)
        except LoaderError as exc:
            outcomes.append(
                Loaded(path=path, mtime_ms=mtime_ms, value=None, error=str(exc))
            )
        else:
            outcomes.append(
                Loaded(path=path, mtime_ms=mtime_ms, value=value, error=None)
            )
    outcomes.sort(key=lambda o: o.mtime_ms, reverse=True)
    return outcomes


def _load_jsonl(path: Path, model: type[ModelT]) -> list[Loaded[ModelT]]:
    """Load every non-blank line of a JSONL file through ``model``. One
    :class:`Loaded` outcome per line; a malformed line becomes an error
    outcome and the rest of the file still loads. A missing file yields
    an empty list."""
    if not path.is_file():
        return []
    mtime_ms = _mtime_ms(path)
    outcomes: list[Loaded[ModelT]] = []
    with path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    for index, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            value = model.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            outcomes.append(
                Loaded(
                    path=path,
                    mtime_ms=mtime_ms,
                    value=None,
                    error=f"line {index + 1}: {exc!r}",
                )
            )
        else:
            outcomes.append(
                Loaded(path=path, mtime_ms=mtime_ms, value=value, error=None)
            )
    return outcomes


def _load_parquet_dir(
    directory: Path, model: type[ModelT]
) -> list[Loaded[ModelT]]:
    """Load every row of every ``shard-*.parquet`` file in ``directory``
    through ``model``. One :class:`Loaded` outcome per row; a row that
    does not deserialize becomes an error outcome. A missing directory
    yields an empty list."""
    if not directory.is_dir():
        return []
    outcomes: list[Loaded[ModelT]] = []
    for shard in sorted(directory.glob("shard-*.parquet")):
        mtime_ms = _mtime_ms(shard)
        try:
            table = pq.read_table(str(shard))  # type: ignore[no-untyped-call]
            rows = table.to_pylist()
        except Exception as exc:  # noqa: BLE001 — surface any pyarrow fault
            outcomes.append(
                Loaded(
                    path=shard,
                    mtime_ms=mtime_ms,
                    value=None,
                    error=f"{shard.name}: parquet read failed ({exc!r}).",
                )
            )
            continue
        for index, row in enumerate(rows):
            try:
                value = model.model_validate(row)
            except ValidationError as exc:
                outcomes.append(
                    Loaded(
                        path=shard,
                        mtime_ms=mtime_ms,
                        value=None,
                        error=(
                            f"{shard.name} row {index}: does not "
                            f"deserialize as {model.__name__} "
                            f"({exc.error_count()} error(s))."
                        ),
                    )
                )
            else:
                outcomes.append(
                    Loaded(
                        path=shard,
                        mtime_ms=mtime_ms,
                        value=value,
                        error=None,
                    )
                )
    return outcomes


# ---------------------------------------------------------------------------
# Mirror-side record loaders.
# ---------------------------------------------------------------------------


def load_round_results(run_dir: Path) -> list[Loaded[RoundResult]]:
    """Load ``runs/{run_id}/mirror/rounds/*.json`` as :class:`RoundResult`
    records, most-recently-modified first."""
    return _load_json_dir(run_dir / "mirror" / "rounds", RoundResult)


def load_round_judgments(run_dir: Path) -> list[Loaded[RoundJudgment]]:
    """Load ``runs/{run_id}/mirror/judgments/*.json`` as
    :class:`RoundJudgment` records."""
    return _load_json_dir(run_dir / "mirror" / "judgments", RoundJudgment)


def load_pass_results(run_dir: Path) -> list[Loaded[PassResult]]:
    """Load ``runs/{run_id}/mirror/passes/*.json`` as :class:`PassResult`
    records. Phase 12+ rounds embed their passes inside the
    :class:`RoundResult`; this loader covers the Phase 8 single-pass
    artifacts written directly under ``passes/``."""
    return _load_json_dir(run_dir / "mirror" / "passes", PassResult)


def load_stability_results(path: Path) -> list[Loaded[StabilityResult]]:
    """Load a stability-audit JSONL file as :class:`StabilityResult`
    records — one per line. The path is explicit because Phase 10's
    ``audit_jsonl_path`` is caller-chosen and not at a fixed location."""
    return _load_jsonl(path, StabilityResult)


def load_faithfulness_results(path: Path) -> list[Loaded[FaithfulnessResult]]:
    """Load a faithfulness-audit JSONL file as :class:`FaithfulnessResult`
    records — one per line. The path is explicit for the same reason as
    :func:`load_stability_results`."""
    return _load_jsonl(path, FaithfulnessResult)


def load_admissibility_records(
    run_dir: Path,
) -> list[Loaded[AdmissibilityVerdict]]:
    """Load ``runs/{run_id}/mirror/admissibility.jsonl`` as
    :class:`AdmissibilityVerdict` records — one per line.

    Unlike Phase 10's stability JSONL (whose ``audit_jsonl_path`` is
    caller-chosen), Phase 12 commits the admissibility consumer's audit
    file to a fixed location under the run's ``mirror/`` directory, so
    this loader takes ``run_dir`` and builds the path itself. A missing
    file yields an empty list."""
    return _load_jsonl(run_dir / "mirror" / "admissibility.jsonl",
                       AdmissibilityVerdict)


def aggregate_llm_audit(run_dir: Path) -> LLMCallAudit:
    """Aggregate every :class:`~kind.mirror.LLMCallRecord` across every
    :class:`RoundResult` in the run into a single :class:`LLMCallAudit`.

    Records from rounds that failed to deserialize are skipped (their
    error surfaces separately in the ``/rounds`` view). An empty run
    yields a zero-totals audit.
    """
    records: list[LLMCallRecord] = []
    for outcome in load_round_results(run_dir):
        if outcome.value is not None:
            records.extend(outcome.value.llm_call_records)
    return LLMCallAudit.from_records(tuple(records))


# ---------------------------------------------------------------------------
# Telemetry record loaders.
# ---------------------------------------------------------------------------


def load_agent_steps(run_dir: Path) -> list[Loaded[AgentStep]]:
    """Load ``runs/{run_id}/telemetry/agent_step/shard-*.parquet`` as
    :class:`~kind.observer.schemas.AgentStep` records — one per row."""
    return _load_parquet_dir(run_dir / "telemetry" / "agent_step", AgentStep)


def load_dream_rollouts(run_dir: Path) -> list[Loaded[DreamRollout]]:
    """Load ``runs/{run_id}/telemetry/dream_rollout/shard-*.parquet`` as
    :class:`~kind.observer.schemas.DreamRollout` records."""
    return _load_parquet_dir(
        run_dir / "telemetry" / "dream_rollout", DreamRollout
    )


def load_world_events(run_dir: Path) -> list[Loaded[WorldEvent]]:
    """Load ``runs/{run_id}/telemetry/world_event.jsonl`` as
    :class:`~kind.observer.schemas.WorldEvent` records."""
    return _load_jsonl(run_dir / "telemetry" / "world_event.jsonl", WorldEvent)


def load_replay_meta(run_dir: Path) -> list[Loaded[ReplayMeta]]:
    """Load ``runs/{run_id}/telemetry/replay_meta.jsonl`` as
    :class:`~kind.observer.schemas.ReplayMeta` records."""
    return _load_jsonl(run_dir / "telemetry" / "replay_meta.jsonl", ReplayMeta)


# ---------------------------------------------------------------------------
# Telemetry write-activity probe (for state inference).
# ---------------------------------------------------------------------------


def telemetry_dir_exists(run_dir: Path) -> bool:
    """True iff ``runs/{run_id}/telemetry/`` is present — a run with no
    telemetry directory never ran Io (it is a mirror-only calibration
    run), and the overview surfaces ``unknown`` rather than guessing."""
    return (run_dir / "telemetry").is_dir()


def stream_last_write_ms(run_dir: Path) -> dict[str, int | None]:
    """Return the most-recent write time (epoch ms) of each telemetry
    stream, or ``None`` for a stream with no files.

    The write time is the file modification time — the genuine
    "write activity" signal the state-inference heuristic keys off.
    ``agent_step`` and ``dream_rollout`` are sharded directories (the
    newest shard's mtime is used); ``replay_meta`` and ``world_event``
    are single JSONL files.
    """
    telemetry = run_dir / "telemetry"
    result: dict[str, int | None] = {}
    for name in ("agent_step", "dream_rollout"):
        shard_dir = telemetry / name
        shard_mtimes = (
            [s.stat().st_mtime for s in shard_dir.glob("shard-*.parquet")]
            if shard_dir.is_dir()
            else []
        )
        result[name] = (
            int(max(shard_mtimes) * 1000) if shard_mtimes else None
        )
    for name in ("replay_meta", "world_event"):
        jsonl = telemetry / f"{name}.jsonl"
        result[name] = (
            int(jsonl.stat().st_mtime * 1000) if jsonl.is_file() else None
        )
    return result
