"""Phase 3a builder-perturbation mutators.

The four named mutators specified by the environment synthesis §Q5 and the
implementation plan §2.4: ``add_resource``, ``remove_object``,
``set_cell_state``, ``move_object``. Each is a pure function that mutates a
``GridWorld``'s underlying grid in place and returns a payload ``dict`` ready
to drop into a ``WorldEvent`` record's ``payload`` field.

These are local Python functions; they do not emit ``WorldEvent`` records
themselves. ``EnvServer`` (in ``env_server.py``) is the only caller and is
responsible for wrapping each return value in a ``builder_perturbation``
``WorldEvent`` and writing it to the ``world_event`` sink.

**Validation policy.** Out-of-bounds cells and invalid ``object_type`` /
``state`` values raise immediately. No-op mutations (e.g. ``add_resource``
on a cell that is already a resource, ``remove_object`` on a cell that is
already empty) succeed and emit a payload with ``pre_state == post_state``,
preserving the auditing trail per the user's recommendation. ``move_object``
is the exception: ``cell_from == cell_to`` raises (a self-move is not a
no-op of the same character — internal stochasticity does not produce
self-moves), an empty source raises (cannot move what is not there), and a
non-empty destination raises (the simplest semantics for an atomic move).

**The same-vocabulary commitment.** The environment synthesis specifies that
builder mutators deliberately use the same vocabulary of state changes as
internal stochasticity — ``add_resource`` does what regrowth does,
``remove_object`` does what consumption does, ``set_cell_state`` and
``move_object`` are within the WALL/RESOURCE/EMPTY world. The probability
distribution over (when, where, what) is what Probe 4 will eventually use to
distinguish builder events from regrowth; that difference does not live in
the mutator semantics here.
"""

from __future__ import annotations

from typing import Any

from kind.env.grid_world import CellType, GridWorld

__all__ = [
    "CELL_TYPE_NAMES",
    "cell_type_name",
    "add_resource",
    "remove_object",
    "set_cell_state",
    "move_object",
]


# ---- helpers --------------------------------------------------------------


CELL_TYPE_NAMES: dict[int, str] = {
    CellType.EMPTY.value: "empty",
    CellType.WALL.value: "wall",
    CellType.RESOURCE.value: "resource",
    CellType.TRAIL.value: "trail",
}


def cell_type_name(value: int) -> str:
    """Map an underlying-grid cell value (0/1/2/4) to its lowercase name.

    ``WorldEvent`` payloads encode pre/post state as strings, not enum
    instances, so JSONL roundtrip stays clean.
    """
    if value not in CELL_TYPE_NAMES:
        raise ValueError(f"unknown cell-type value: {value}")
    return CELL_TYPE_NAMES[value]


def _validate_cell(grid_world: GridWorld, cell: tuple[int, int]) -> None:
    """Raise ``ValueError`` if ``cell`` is out of grid bounds."""
    if not isinstance(cell, tuple) or len(cell) != 2:
        raise ValueError(f"cell must be a (row, col) tuple, got {cell!r}")
    r, c = cell
    if not isinstance(r, int) or not isinstance(c, int):
        raise ValueError(f"cell coords must be ints, got {cell!r}")
    gs = grid_world.config.grid_size
    if not (0 <= r < gs and 0 <= c < gs):
        raise ValueError(
            f"cell {cell!r} out of grid bounds [0, {gs})²"
        )


def _validate_cell_type(value: object, name: str) -> CellType:
    """Coerce ``value`` to ``CellType`` or raise a clear ``TypeError``.

    Accepts only ``CellType`` instances. Plain ints are rejected: the public
    surface of the mutators types these parameters as ``CellType``, and
    silently coercing ints would hide bugs in callers.
    """
    if not isinstance(value, CellType):
        raise TypeError(
            f"{name} must be a CellType (got {type(value).__name__}: {value!r})"
        )
    return value


# ---- the four mutators ---------------------------------------------------


def add_resource(
    grid_world: GridWorld, cell: tuple[int, int]
) -> dict[str, Any]:
    """Set ``cell`` to ``RESOURCE`` and return the ``WorldEvent`` payload.

    Idempotent on cells that are already resources (no-op success, payload
    has ``pre_state == post_state == "resource"``). Raises ``ValueError`` if
    ``cell`` is out of bounds or currently a wall — the latter because
    regrowth never overwrites walls, so allowing the mutator to do so would
    break the same-vocabulary commitment the synthesis is built on.
    """
    _validate_cell(grid_world, cell)
    r, c = cell
    pre_value = int(grid_world._grid[r, c])
    if pre_value == CellType.WALL.value:
        raise ValueError(
            f"add_resource at {cell}: cannot add a resource on a wall cell"
        )
    grid_world._grid[r, c] = CellType.RESOURCE.value
    post_value = int(grid_world._grid[r, c])
    return {
        "mutator": "add_resource",
        "cell": [r, c],
        "pre_state": cell_type_name(pre_value),
        "post_state": cell_type_name(post_value),
    }


def remove_object(
    grid_world: GridWorld,
    cell: tuple[int, int],
    object_type: CellType,
) -> dict[str, Any]:
    """Remove the object at ``cell`` if it matches ``object_type``.

    Sets the cell to ``EMPTY``. If the cell is already ``EMPTY``, this is a
    no-op success (idempotent removal); the payload records
    ``pre_state == post_state == "empty"``. If the cell holds a different
    non-empty type than ``object_type``, raises ``ValueError`` (a clearer
    failure mode than silently overwriting).

    Raises ``TypeError`` if ``object_type`` is not a ``CellType``;
    ``ValueError`` if ``object_type`` is ``CellType.EMPTY`` (removing
    "empty" has no defined meaning under the mutator vocabulary).
    """
    _validate_cell(grid_world, cell)
    target = _validate_cell_type(object_type, "object_type")
    if target == CellType.EMPTY:
        raise ValueError(
            "remove_object: object_type must be a non-empty cell type "
            "(WALL, RESOURCE, or TRAIL)"
        )
    r, c = cell
    pre_value = int(grid_world._grid[r, c])
    if pre_value == CellType.EMPTY.value:
        # Idempotent: cell is already empty, the requested object is not
        # there to remove. Succeed with pre == post == "empty".
        post_value = pre_value
    elif pre_value == target.value:
        grid_world._grid[r, c] = CellType.EMPTY.value
        post_value = int(grid_world._grid[r, c])
    else:
        raise ValueError(
            f"remove_object at {cell}: cell holds "
            f"{cell_type_name(pre_value)}, not {cell_type_name(target.value)}"
        )
    return {
        "mutator": "remove_object",
        "cell": [r, c],
        "object_type": cell_type_name(target.value),
        "pre_state": cell_type_name(pre_value),
        "post_state": cell_type_name(post_value),
    }


def set_cell_state(
    grid_world: GridWorld,
    cell: tuple[int, int],
    state: CellType,
) -> dict[str, Any]:
    """Set ``cell`` to ``state``, regardless of current value.

    The most general of the four mutators: any cell-type can be written to
    any cell. No-op on ``pre == post`` (both equal to the requested state)
    succeeds and emits the payload with matching pre/post.
    """
    _validate_cell(grid_world, cell)
    new_state = _validate_cell_type(state, "state")
    if new_state == CellType.TRAIL:
        # World v2 E1: trail is Io's own footprint by definition — a
        # builder-written trail cell would put SELF-attributable state
        # into the world from the BUILDER class (and, having no decay
        # clock, would never fade). Builders may pave over or remove
        # trail; they may not fabricate it.
        raise ValueError(
            "set_cell_state: TRAIL cannot be written by a builder "
            "mutator — trail cells are Io's own footprints"
        )
    r, c = cell
    pre_value = int(grid_world._grid[r, c])
    grid_world._grid[r, c] = new_state.value
    post_value = int(grid_world._grid[r, c])
    return {
        "mutator": "set_cell_state",
        "cell": [r, c],
        "pre_state": cell_type_name(pre_value),
        "post_state": cell_type_name(post_value),
    }


def move_object(
    grid_world: GridWorld,
    cell_from: tuple[int, int],
    cell_to: tuple[int, int],
) -> dict[str, Any]:
    """Move the object at ``cell_from`` to ``cell_to``.

    The single non-idempotent mutator. Raises ``ValueError`` if either cell
    is out of bounds, if ``cell_from == cell_to`` (a self-move has no
    counterpart in internal stochasticity and is rejected to keep the
    mutator vocabulary clean), if ``cell_from`` is empty (nothing to move),
    or if ``cell_to`` is non-empty (the destination must be empty so the
    move is atomic and the post-state is unambiguous).
    """
    _validate_cell(grid_world, cell_from)
    _validate_cell(grid_world, cell_to)
    if cell_from == cell_to:
        raise ValueError(
            f"move_object: cell_from and cell_to must differ "
            f"(both are {cell_from!r})"
        )
    rf, cf = cell_from
    rt, ct = cell_to
    pre_from = int(grid_world._grid[rf, cf])
    pre_to = int(grid_world._grid[rt, ct])
    if pre_from == CellType.EMPTY.value:
        raise ValueError(
            f"move_object: cell_from {cell_from!r} is empty (nothing to move)"
        )
    if pre_from == CellType.TRAIL.value:
        # World v2 E1: a moved trail cell would be a builder-fabricated
        # footprint at the destination (and would detach from its decay
        # clock). Trail is not a movable object; pave or remove instead.
        raise ValueError(
            "move_object: TRAIL is not a movable object — trail cells "
            "are Io's own footprints"
        )
    if pre_to != CellType.EMPTY.value:
        raise ValueError(
            f"move_object: cell_to {cell_to!r} is not empty "
            f"(holds {cell_type_name(pre_to)})"
        )
    grid_world._grid[rt, ct] = pre_from
    grid_world._grid[rf, cf] = CellType.EMPTY.value
    post_from = int(grid_world._grid[rf, cf])
    post_to = int(grid_world._grid[rt, ct])
    return {
        "mutator": "move_object",
        "cell_from": [rf, cf],
        "cell_to": [rt, ct],
        "pre_state_from": cell_type_name(pre_from),
        "pre_state_to": cell_type_name(pre_to),
        "post_state_from": cell_type_name(post_from),
        "post_state_to": cell_type_name(post_to),
    }
