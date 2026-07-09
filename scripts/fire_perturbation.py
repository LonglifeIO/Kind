"""Probe 4 Phase 2 — the builder's manual perturbation trigger (plan §S-PERT).

Thin CLI over the trigger inbox: writes one JSON request file into the
run's spool directory; the live runner drains the spool at its next step
boundary and fires the mutation through the tested env-server mutator
surface with ``trigger="manual"`` (see ``kind/env/trigger_inbox.py`` for
why this is a spool and not a live wire connection — the transport is
single-connection and the runner owns it).

Usage (the inbox dir is whatever ``RunnerConfig.perturbation_inbox_dir``
the run was started with):

    python scripts/fire_perturbation.py --inbox RUN_DIR/perturbation_inbox \\
        add_resource --cell 3 4
    python scripts/fire_perturbation.py --inbox ... \\
        remove_object --cell 3 4 --object-type resource
    python scripts/fire_perturbation.py --inbox ... \\
        set_cell_state --cell 3 4 --state wall
    python scripts/fire_perturbation.py --inbox ... \\
        move_object --cell-from 1 1 --cell-to 2 2

``--wait SECONDS`` polls for the result sidecar the runner writes on
drain and prints the outcome (default: fire-and-forget).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from kind.env.trigger_inbox import write_trigger_request


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fire a manual builder perturbation into a live run."
    )
    parser.add_argument(
        "--inbox",
        type=Path,
        required=True,
        help="The run's perturbation inbox directory "
        "(RunnerConfig.perturbation_inbox_dir).",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=0.0,
        help="Seconds to wait for the runner's result sidecar (0 = don't).",
    )
    sub = parser.add_subparsers(dest="mutator", required=True)

    add_resource = sub.add_parser("add_resource")
    add_resource.add_argument("--cell", type=int, nargs=2, required=True)

    remove_object = sub.add_parser("remove_object")
    remove_object.add_argument("--cell", type=int, nargs=2, required=True)
    remove_object.add_argument(
        "--object-type",
        choices=["wall", "resource", "trail"],
        required=True,
    )

    set_cell_state = sub.add_parser("set_cell_state")
    set_cell_state.add_argument("--cell", type=int, nargs=2, required=True)
    set_cell_state.add_argument(
        "--state", choices=["empty", "wall", "resource"], required=True
    )

    move_object = sub.add_parser("move_object")
    move_object.add_argument("--cell-from", type=int, nargs=2, required=True)
    move_object.add_argument("--cell-to", type=int, nargs=2, required=True)

    return parser


def _args_payload(namespace: argparse.Namespace) -> dict[str, Any]:
    mutator = str(namespace.mutator)
    if mutator == "add_resource":
        return {"cell": list(namespace.cell)}
    if mutator == "remove_object":
        return {
            "cell": list(namespace.cell),
            "object_type": str(namespace.object_type),
        }
    if mutator == "set_cell_state":
        return {"cell": list(namespace.cell), "state": str(namespace.state)}
    if mutator == "move_object":
        return {
            "cell_from": list(namespace.cell_from),
            "cell_to": list(namespace.cell_to),
        }
    raise ValueError(f"unknown mutator: {mutator!r}")


def main(argv: list[str] | None = None) -> int:
    namespace = _build_parser().parse_args(argv)
    request_path = write_trigger_request(
        namespace.inbox, str(namespace.mutator), _args_payload(namespace)
    )
    print(f"queued: {request_path.name}")
    if namespace.wait <= 0.0:
        return 0
    sidecar = (
        namespace.inbox / "processed" / (request_path.name + ".result.json")
    )
    deadline = time.monotonic() + float(namespace.wait)
    while time.monotonic() < deadline:
        if sidecar.exists():
            result = json.loads(sidecar.read_text(encoding="utf-8"))
            print(f"result: {json.dumps(result, sort_keys=True)}")
            return 0 if result.get("ok") else 1
        time.sleep(0.1)
    print("result: not drained within --wait window (run still queued)")
    return 2


if __name__ == "__main__":
    sys.exit(main())
