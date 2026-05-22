"""One-time migration — add ``falsifier_id`` to every ``Criterion``
embedded in the Phase 13 calibration artifacts on disk.

Phase 9 added a required ``falsifier_id: str`` field to
:class:`kind.mirror.registry.Criterion`. The Phase 13 calibration
artifacts on disk were serialized before Phase 9 and therefore do not
carry the field. Re-loading them with the post-Phase-9 schema raises
``ValidationError`` for every embedded :class:`Criterion`.

This migration fills in ``falsifier_id`` using the ``id + "_v1"``
convention — the same suffix Phase 7's three v2 criteria commit. The
substitution is mechanical: each Criterion's id maps to its
falsifier_id deterministically. The migration is idempotent: a
Criterion that already carries ``falsifier_id`` is left untouched.

Files migrated:

- ``runs/phase_13_calibration/mirror/rounds/phase_13_probe_1_round.json``
- ``runs/phase_13_calibration/mirror/rounds/phase_13_probe_1_5_round.json``
- ``runs/phase_13_calibration/mirror/phase_13_calibration_result.json``
- ``runs/phase_13_calibration/mirror/pre_reg/round_*/round_config.json``

The migration journals the change in the Phase 9 findings entry. No
schema or code changes follow from this migration; it is a one-time
data fix to bring pre-Phase-9 artifacts forward.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _walk_and_fix_criteria(node: Any, *, count: list[int]) -> None:
    """Recursively walk ``node`` and add ``falsifier_id`` to every
    dict that looks like a :class:`Criterion` (has an ``id`` field
    that's a snake_case string, ``signal_mappings`` field that's a
    list, ``falsifier`` field, ``held_out`` field, etc.) and is
    missing ``falsifier_id``."""
    if isinstance(node, dict):
        looks_like_criterion = (
            "id" in node
            and "signal_mappings" in node
            and "falsifier" in node
            and "held_out" in node
            and "framework" in node
            and isinstance(node.get("id"), str)
        )
        if looks_like_criterion and "falsifier_id" not in node:
            node["falsifier_id"] = f"{node['id']}_v1"
            count[0] += 1
        for v in node.values():
            _walk_and_fix_criteria(v, count=count)
    elif isinstance(node, list):
        for item in node:
            _walk_and_fix_criteria(item, count=count)


def _migrate_file(path: Path) -> tuple[bool, int]:
    """Migrate one JSON file in-place atomically. Returns
    ``(changed, n_criteria_patched)``."""
    if not path.is_file():
        return (False, 0)
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    count = [0]
    _walk_and_fix_criteria(payload, count=count)
    if count[0] == 0:
        return (False, 0)
    tmp = path.with_suffix(path.suffix + ".tmp_migrate")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    tmp.replace(path)
    return (True, count[0])


def main() -> int:
    base = Path("runs/phase_13_calibration/mirror")
    paths = [
        base / "rounds" / "phase_13_probe_1_round.json",
        base / "rounds" / "phase_13_probe_1_5_round.json",
        base / "phase_13_calibration_result.json",
    ]
    paths.extend(
        p / "round_config.json"
        for p in (base / "pre_reg").glob("round_*")
    )
    total = 0
    for path in paths:
        changed, n = _migrate_file(path)
        marker = "patched" if changed else "(skipped/idempotent)"
        print(f"  {path}: {marker} ({n} criteria updated)")
        total += n
    print(f"Total criteria patched: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
