"""Probe 4 Phase 2 — manual perturbation trigger inbox (plan §S-PERT,
sharpened).

**Why an inbox and not a live wire client (a plan discrepancy, recorded
here and in the journal).** The plan scoped the builder trigger as "a
thin CLI over the tested ``EnvTransportClient.mutate()`` wire path". That
collides with two live constraints the plan's grounding pass did not
surface: :class:`~kind.env.transport.EnvTransportServer` is
**single-connection**, and the client enforces a **one-outstanding-
request** contract (``_acquire_request_lock_or_raise``: "issue MUTATEs
from the same loop that issues STEPs, or serialize externally"). During
a live run the runner owns the sole connection and the sole request
loop, so a second process cannot reach the mutators over the wire at
all.

The realization that preserves the plan's intent: the builder's CLI
(``scripts/fire_perturbation.py``) writes a small JSON request file into
a spool directory; the runner drains the spool at the same step boundary
where the generator fires, calling the **same tested env-server mutator
surface** with ``trigger="manual"``. The builder's real-time timing is
preserved to within one env step; manual events gain the same
deterministic step-boundary placement, the same ``source="builder"``
emission, and the same no-observation-marker property as every other
builder event. Each request is archived with a result sidecar so the
CLI can report success/failure.

**Robustness rule:** a malformed or invalid request must never crash the
run (Phase 4 is a biography — a builder typo cannot be allowed to kill
it). Every failure mode becomes an archived error result.

The argument vocabulary mirrors the transport's MUTATE codec
(``transport._decode_mutate_args``): cells are ``[row, col]`` lists,
cell types are lowercase names. The two codecs are twins; the tests
hold them to the same vocabulary.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from kind.env.grid_world import CellType

if TYPE_CHECKING:
    from kind.env.env_server import EnvServer

__all__ = [
    "TriggerResult",
    "write_trigger_request",
    "drain_trigger_inbox",
]


_VALID_MUTATORS: Final[frozenset[str]] = frozenset(
    {"add_resource", "remove_object", "set_cell_state", "move_object"}
)
_PROCESSED_DIR_NAME: Final[str] = "processed"


@dataclass(frozen=True)
class TriggerResult:
    """Outcome of one drained request (also written as the sidecar)."""

    request_name: str
    ok: bool
    error: str | None


def write_trigger_request(
    inbox_dir: Path, mutator: str, args: dict[str, Any]
) -> Path:
    """Write one request file atomically (tmp + rename); return its path.

    The filename is monotonic-nanosecond-prefixed so the runner's sorted
    drain processes requests in submission order. Used by the CLI; tests
    call it directly.
    """
    inbox_dir.mkdir(parents=True, exist_ok=True)
    name = f"{time.monotonic_ns()}-{mutator}.json"
    final_path = inbox_dir / name
    tmp_path = inbox_dir / (name + ".tmp")
    tmp_path.write_text(
        json.dumps({"mutator": mutator, "args": args}, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.rename(final_path)
    return final_path


def _decode_args(mutator: str, args: dict[str, Any]) -> dict[str, Any]:
    """Restore Python-side types from the JSON request (the transport
    codec's twin — see module docstring)."""
    if mutator == "add_resource":
        return {"cell": _decode_cell(args["cell"])}
    if mutator == "remove_object":
        return {
            "cell": _decode_cell(args["cell"]),
            "object_type": _decode_cell_type(args["object_type"]),
        }
    if mutator == "set_cell_state":
        return {
            "cell": _decode_cell(args["cell"]),
            "state": _decode_cell_type(args["state"]),
        }
    if mutator == "move_object":
        return {
            "cell_from": _decode_cell(args["cell_from"]),
            "cell_to": _decode_cell(args["cell_to"]),
        }
    raise ValueError(f"unknown mutator: {mutator!r}")


def _decode_cell(value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(isinstance(v, int) for v in value)
    ):
        raise ValueError(f"cell must be a [row, col] int pair, got {value!r}")
    return (int(value[0]), int(value[1]))


def _decode_cell_type(value: Any) -> CellType:
    if not isinstance(value, str) or value.upper() not in CellType.__members__:
        raise ValueError(
            f"cell type must be one of "
            f"{[m.lower() for m in CellType.__members__]}, got {value!r}"
        )
    return CellType[value.upper()]


def drain_trigger_inbox(
    inbox_dir: Path, env_server: "EnvServer"
) -> list[TriggerResult]:
    """Process every pending request file, oldest first.

    Each request fires through the env-server's mutator surface with
    ``trigger="manual"``; the request file is then moved to
    ``processed/`` and a ``<name>.result.json`` sidecar records the
    outcome. Failures (malformed JSON, unknown mutator, mutator
    validation errors) become error results — the run never crashes on a
    bad request.
    """
    if not inbox_dir.is_dir():
        return []
    processed_dir = inbox_dir / _PROCESSED_DIR_NAME
    results: list[TriggerResult] = []
    for request_path in sorted(inbox_dir.glob("*.json")):
        ok = False
        error: str | None = None
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            mutator = str(request["mutator"])
            if mutator not in _VALID_MUTATORS:
                raise ValueError(f"unknown mutator: {mutator!r}")
            decoded = _decode_args(mutator, request.get("args") or {})
            method = getattr(env_server, mutator)
            method(**decoded, trigger="manual")
            ok = True
        except (
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            error = f"{type(exc).__name__}: {exc}"
        processed_dir.mkdir(parents=True, exist_ok=True)
        archived = processed_dir / request_path.name
        request_path.rename(archived)
        result = TriggerResult(
            request_name=request_path.name, ok=ok, error=error
        )
        (processed_dir / (request_path.name + ".result.json")).write_text(
            json.dumps(
                {"ok": result.ok, "error": result.error}, sort_keys=True
            ),
            encoding="utf-8",
        )
        results.append(result)
    return results
