"""Phase 1 telemetry sinks.

Two append-only writers consuming Pydantic v2 models from
``kind/observer/schemas.py``:

* ``JsonlSink`` writes line-delimited JSON to a single file.
* ``ParquetSink`` writes Arrow-backed Parquet files sharded by record count.

Per the implementation plan §2.2: ``replay_meta`` and ``world_event`` go to
``JsonlSink`` (low-volume, human-inspectable); ``agent_step`` and
``dream_rollout`` go to ``ParquetSink`` (columnar, downstream-friendly).
The runner (Phase 5) is responsible for the ``runs/{run_id}/telemetry/``
directory layout; sinks just write to the path or directory they are given.

The PyArrow column schema is derived from the Pydantic model's field info
upfront (rather than inferred from the first batch). This guarantees every
shard a sink writes shares the same column layout even if early batches
contain only ``None`` for an ``Optional`` field — the field is declared
nullable from the start.
"""

from __future__ import annotations

import json
import os
import types
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Literal, Self, TypeVar, Union, get_args, get_origin

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


class SinkClosedError(RuntimeError):
    """Raised when ``write()`` is called on a sink that has been closed."""


class SchemaMismatchError(TypeError):
    """Raised when a record's type does not match the sink's configured schema."""


class JsonlSink:
    """Append-only JSON-Lines writer for a single Pydantic record type.

    One record per line — ``record.model_dump_json()`` followed by ``"\\n"``,
    **flushed per write** so concurrent readers (the live window's event
    feed, check-in greps against a running biography) see each record as
    soon as it is written — Python's default 8 KiB text buffer otherwise
    lags the stream by minutes at biography event rates (observed
    2026-07-08). ``os.fsync`` still happens only on ``close()`` (flush
    reaches the OS page cache, which is what a same-host reader needs;
    per-write fsync would buy nothing but latency).
    """

    def __init__(self, path: Path, schema: type[BaseModel]) -> None:
        self._path = path
        self._schema = schema
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file: IO[str] | None = path.open("a", encoding="utf-8")
        self._closed = False

    def write(self, record: BaseModel) -> None:
        if self._closed or self._file is None:
            raise SinkClosedError(f"JsonlSink at {self._path} is closed")
        if not isinstance(record, self._schema):
            raise SchemaMismatchError(
                f"JsonlSink for {self._schema.__name__} cannot write a "
                f"{type(record).__name__}"
            )
        self._file.write(record.model_dump_json() + "\n")
        self._file.flush()

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


class ParquetSink:
    """Sharded Parquet writer for a single Pydantic record type.

    Records are buffered in memory; each ``rows_per_shard`` records are
    flushed as a single Parquet file under the sink's directory, and a final
    partial shard is flushed on ``close()``. Shards are named
    ``shard-{n:06d}.parquet``. The PyArrow schema is derived once from the
    Pydantic model so every shard shares one column layout (and so reads can
    concatenate freely across shards).

    Each shard file is fsync-ed individually after its bytes are written.
    """

    SHARD_NAME_FORMAT = "shard-{index:06d}.parquet"

    def __init__(
        self,
        dir: Path,
        schema: type[BaseModel],
        rows_per_shard: int = 10_000,
    ) -> None:
        if rows_per_shard <= 0:
            raise ValueError(
                f"rows_per_shard must be positive, got {rows_per_shard}"
            )
        self._dir = dir
        self._schema = schema
        self._rows_per_shard = rows_per_shard
        dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict[str, Any]] = []
        self._arrow_schema: pa.Schema = _arrow_schema_from_pydantic(schema)
        self._json_fields: frozenset[str] = json_encoded_field_names(schema)
        # Append semantics for a continued run (biography resume): a
        # fresh sink over a directory with existing shards continues the
        # numbering instead of overwriting shard-000000 (C1 finding
        # 2026-07-08 — a resumed session silently clobbered session 1's
        # first shard).
        existing = sorted(dir.glob("shard-*.parquet"))
        self._next_shard_index = (
            int(existing[-1].stem.split("-")[1]) + 1 if existing else 0
        )
        self._closed = False

    def write(self, record: BaseModel) -> None:
        if self._closed:
            raise SinkClosedError(f"ParquetSink at {self._dir} is closed")
        if not isinstance(record, self._schema):
            raise SchemaMismatchError(
                f"ParquetSink for {self._schema.__name__} cannot write a "
                f"{type(record).__name__}"
            )
        row = record.model_dump()
        # JSON-encode dict-typed fields into their string columns (None stays
        # None → a null cell). Symmetric with the decode in
        # model_validate_parquet_row / read_parquet_dir.
        for name in self._json_fields:
            value = row.get(name)
            if value is not None:
                row[name] = json.dumps(value, sort_keys=True)
        self._buffer.append(row)
        if len(self._buffer) >= self._rows_per_shard:
            self._flush_shard()

    def _flush_shard(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=self._arrow_schema)
        shard_path = self._dir / self.SHARD_NAME_FORMAT.format(
            index=self._next_shard_index
        )
        with shard_path.open("wb") as fh:
            pq.write_table(table, fh)  # type: ignore[no-untyped-call]
            fh.flush()
            os.fsync(fh.fileno())
        self._next_shard_index += 1
        self._buffer = []

    def close(self) -> None:
        if self._closed:
            return
        self._flush_shard()
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


# ---- Pydantic → Arrow schema derivation ------------------------------------


def _is_optional_annotation(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        return type(None) in get_args(annotation)
    return False


def _arrow_type_for(annotation: Any) -> pa.DataType:
    """Map a Python / typing annotation to a PyArrow data type.

    Supports the type set used by ``AgentStep`` and ``DreamRollout`` (the two
    record models routed to ``ParquetSink``). Raises ``ValueError`` for
    unsupported types — most importantly ``dict[str, Any]``, which appears
    only in ``WorldEvent.payload`` and is correctly routed to ``JsonlSink``.
    """
    origin = get_origin(annotation)

    if origin in (Union, types.UnionType):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) != 1:
            raise ValueError(
                "ParquetSink only supports Optional unions "
                f"(one non-None arm); got {annotation!r}"
            )
        return _arrow_type_for(non_none[0])

    if origin is Literal:
        values = get_args(annotation)
        if all(isinstance(v, str) for v in values):
            return pa.string()
        if all(isinstance(v, bool) for v in values):
            return pa.bool_()
        if all(isinstance(v, int) for v in values):
            return pa.int64()
        raise ValueError(
            f"ParquetSink: unsupported Literal element types in {annotation!r}"
        )

    if origin is list:
        (inner,) = get_args(annotation)
        return pa.list_(_arrow_type_for(inner))

    if origin is tuple:
        # Collapse a homogeneous fixed-arity tuple to a list<inner>. This
        # matches Pydantic's serialization (tuple → sequence) and how PyArrow
        # accepts tuples of equal-typed elements when given an explicit schema.
        # Heterogeneous tuples would need a struct, which Probe 1's schemas
        # do not use.
        args = get_args(annotation)
        if not args:
            raise ValueError(
                f"ParquetSink: empty tuple annotation {annotation!r}"
            )
        inners = [_arrow_type_for(a) for a in args]
        first = inners[0]
        if not all(t == first for t in inners):
            raise ValueError(
                "ParquetSink: heterogeneous tuple annotations are not "
                f"supported; got {annotation!r}"
            )
        return pa.list_(first)

    if origin is dict:
        # dict-typed fields (e.g. DreamRollout.sampling_parameters, whose value
        # union dict[str, float | int | str | bool] is heterogeneous and so has
        # no clean native Arrow column type) are stored JSON-encoded in a
        # string column. The write path encodes the dict to a JSON string; the
        # read path (model_validate_parquet_row / read_parquet_dir) decodes it
        # back to a dict before model_validate, so consumers see a dict, not a
        # string. Keeps the columnar streams together (no JsonlSink split) and
        # leaves the Pydantic model — and its JSON Schema export — unchanged.
        return pa.string()

    if annotation is str:
        return pa.string()
    if annotation is bool:
        return pa.bool_()
    if annotation is int:
        return pa.int64()
    if annotation is float:
        return pa.float64()
    if annotation is bytes:
        return pa.binary()

    raise ValueError(
        f"ParquetSink: unsupported field annotation {annotation!r}"
    )


def _arrow_schema_from_pydantic(model: type[BaseModel]) -> pa.Schema:
    fields: list[pa.Field] = []
    for name, info in model.model_fields.items():
        annotation = info.annotation
        nullable = _is_optional_annotation(annotation)
        arrow_type = _arrow_type_for(annotation)
        fields.append(pa.field(name, arrow_type, nullable=nullable))
    return pa.schema(fields)


def _unwrap_optional(annotation: Any) -> Any:
    """Return the single non-None arm of an Optional union, else the annotation."""
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return annotation


def json_encoded_field_names(model: type[BaseModel]) -> frozenset[str]:
    """Field names ``ParquetSink`` stores JSON-encoded (dict-typed fields).

    A field is JSON-encoded iff its annotation, after unwrapping ``Optional``,
    has ``dict`` origin. Shared by the write path (encode) and the read path
    (``model_validate_parquet_row`` / ``read_parquet_dir`` decode) so the two
    halves stay symmetric — the single source of truth for which columns are
    JSON strings on disk.
    """
    names: set[str] = set()
    for name, info in model.model_fields.items():
        if get_origin(_unwrap_optional(info.annotation)) is dict:
            names.add(name)
    return frozenset(names)


def decode_parquet_row(model: type[BaseModel], row: dict[str, Any]) -> dict[str, Any]:
    """Decode a ``ParquetSink``-written row back to model-ready values.

    JSON-decodes the model's dict-typed string columns (see
    :func:`json_encoded_field_names`) so the row is ready for
    ``model.model_validate``. A no-op for models with no dict fields, so it is
    safe to route every parquet-row reconstruction through it. Does not mutate
    the input.
    """
    json_fields = json_encoded_field_names(model)
    if not json_fields:
        return row
    decoded = dict(row)
    for name in json_fields:
        value = decoded.get(name)
        if isinstance(value, str):
            decoded[name] = json.loads(value)
    return decoded


def model_validate_parquet_row(model: type[ModelT], row: dict[str, Any]) -> ModelT:
    """Reconstruct a model from a ``ParquetSink``-written row (decode + validate).

    The symmetric counterpart to ``ParquetSink.write``'s JSON encoding: decodes
    dict-typed string columns back to dicts, then ``model_validate``s. Use this
    (or :func:`read_parquet_dir`) instead of bare ``model.model_validate(row)``
    when a model may carry dict fields, so consumers transparently get a dict.
    """
    return model.model_validate(decode_parquet_row(model, row))


def read_parquet_dir(directory: Path, model: type[ModelT]) -> list[ModelT]:
    """Read every shard under ``directory`` and reconstruct ``model`` records.

    Concatenates ``shard-*.parquet`` in name order and reconstructs each row via
    :func:`model_validate_parquet_row` (dict-field-aware). The canonical
    ``ParquetSink`` read counterpart.
    """
    records: list[ModelT] = []
    for shard in sorted(directory.glob("shard-*.parquet")):
        table = pq.read_table(shard)  # type: ignore[no-untyped-call]
        for row in table.to_pylist():
            records.append(model_validate_parquet_row(model, row))
    return records


__all__ = [
    "JsonlSink",
    "ParquetSink",
    "SchemaMismatchError",
    "SinkClosedError",
    "decode_parquet_row",
    "json_encoded_field_names",
    "model_validate_parquet_row",
    "read_parquet_dir",
]
