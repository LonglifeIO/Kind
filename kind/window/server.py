"""Window's HTTP server — a small Flask app: read-only views plus one
builder action.

The server serves on ``localhost:<port>``; the host's Tailscale setup
handles remote access from other devices on the Tailnet without any
code here. Window does not authenticate, encrypt, or restrict access —
it assumes any caller reaching the port is authorized, because the
Tailscale ACL is what gates that.

**The one write surface: ``POST /hello``** — the builder's manual
perturbation button (the plan's DP2 "GUI button" optional convenience,
realized). It writes exactly one request file into the run's
``perturbation_inbox/`` via the tested Phase-2 spool
(:func:`kind.env.trigger_inbox.write_trigger_request`); the live
runner drains it at the next step boundary with ``trigger="manual"``.
Every GET route remains read-only (the invariant test sweeps them);
the write scope of the POST is pinned by its own test to the inbox
directory alone.

The read-only GET routes:

- ``/`` — overview (current state, uptime, totals, pace, breakdowns).
- ``/rounds`` — the round list, most-recent first.
- ``/rounds/<round_id>`` — one round's full readings, with the
  per-reading admissibility verdict surfaced inline per pass.
- ``/judgments/<round_id>`` — one round's judgment detail.
- ``/audit`` — the aggregated LLM-call audit.
- ``/admissibility`` — the joined faithfulness+stability admissibility
  verdict for the run.

The server opens files only for reading (the loaders do all I/O). The
Phase 11.5 read-only test monkeypatches ``Path.open`` and asserts no
write-mode open happens across a full pass through every route.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request

from kind.env.trigger_inbox import write_trigger_request
from kind.mirror import compute_admissibility
from kind.window import loaders
from kind.window.live import LiveWindowState, load_live_state
from kind.window.state import (
    IoState,
    build_overview,
    build_round_rows,
    group_admissibility_for_round,
    latency_distribution,
)

__all__ = ["create_app"]


def _now_ms() -> int:
    return int(time.time() * 1000)


def create_app(run_id: str, run_dir: Path) -> Flask:
    """Build the Window Flask app for one run.

    ``run_id`` is the run's identifier (shown in the chrome);
    ``run_dir`` is its on-disk root (``runs/{run_id}/`` typically). The
    app reads from ``run_dir`` and writes nowhere.
    """
    app = Flask(__name__)

    @app.context_processor
    def _inject_run() -> dict[str, str]:
        return {"run_id": run_id}

    @app.route("/")
    def overview() -> str:
        view = build_overview(run_dir, run_id, now_ms=_now_ms())
        return render_template(
            "overview.html", ov=view, states=list(IoState)
        )

    @app.route("/rounds")
    def rounds() -> str:
        round_outcomes = loaders.load_round_results(run_dir)
        judgment_outcomes = loaders.load_round_judgments(run_dir)
        rows = build_round_rows(round_outcomes, judgment_outcomes)
        return render_template("rounds.html", rows=rows)

    @app.route("/rounds/<round_id>")
    def round_detail(round_id: str) -> str:
        loaded = [
            o.value
            for o in loaders.load_round_results(run_dir)
            if o.value is not None
        ]
        match = next(
            (r for r in loaded if r.round_id == round_id), None
        )
        if match is None:
            abort(404)
        verdicts_per_pass = group_admissibility_for_round(
            match, loaders.load_admissibility_records(run_dir)
        )
        return render_template(
            "round_detail.html",
            result=match,
            verdicts_per_pass=verdicts_per_pass,
        )

    @app.route("/judgments/<round_id>")
    def judgment_detail(round_id: str) -> str:
        loaded = [
            o.value
            for o in loaders.load_round_judgments(run_dir)
            if o.value is not None
        ]
        match = next(
            (j for j in loaded if j.round_id == round_id), None
        )
        if match is None:
            abort(404)
        return render_template("judgment.html", judgment=match)

    @app.route("/live")
    def live() -> str:
        return render_template("live.html")

    @app.route("/live.json")
    def live_json() -> Response:
        loaded = load_live_state(run_dir)
        state: LiveWindowState | None = None
        error: str | None = None
        if isinstance(loaded, LiveWindowState):
            state = loaded
        elif isinstance(loaded, str):
            error = loaded
        age_s: float | None = None
        if state is not None:
            age_s = max(0.0, (_now_ms() - state.wallclock_ms) / 1000.0)
        return jsonify(
            {
                "state": None if state is None else state.model_dump(),
                "error": error,
                "age_s": age_s,
            }
        )

    # The wall-motif hello: a fixed 3-cell "L", anchored at the chosen
    # cell — the builder's inedible gesture (yields nothing, changes
    # nothing Io can use; only something to be modeled). Walls persist
    # until the next episode reshuffle wipes the board.
    _WALL_MOTIF = ((0, 0), (1, 0), (1, 1))

    def _hello_cells(
        loaded: LiveWindowState, kind: str, anchor: tuple[int, int]
    ) -> list[tuple[int, int]] | str:
        """Cells the hello touches, or a refusal string."""
        size = len(loaded.grid)
        agent = tuple(loaded.agent_pos)
        offsets = _WALL_MOTIF if kind == "wall_motif" else ((0, 0),)
        cells = [(anchor[0] + dr, anchor[1] + dc) for dr, dc in offsets]
        for r, c in cells:
            if not (0 <= r < size and 0 <= c < size):
                return "cell out of bounds"
            if (r, c) == agent:
                return "that is Io's own cell"
            if kind == "wall_motif" and loaded.grid[r][c] == 1:
                return "there is already a wall there"
        return cells

    @app.route("/hello", methods=["POST"])
    def hello() -> Response:
        """The builder's hello into the run's perturbation inbox
        (drained by the live runner, tagged ``trigger="manual"``).
        Body: optional ``{"row": r, "col": c}`` for a clicked anchor
        (random legal spot otherwise) and optional ``"kind"`` —
        ``"resource"`` (one ``add_resource``, default) or
        ``"wall_motif"`` (three ``set_cell_state`` walls in a fixed
        L). The only writes this app performs, and only into
        ``perturbation_inbox/``.
        """
        loaded = load_live_state(run_dir)
        if not isinstance(loaded, LiveWindowState):
            return jsonify(
                {"ok": False, "error": "no live run state to aim at"}
            )
        body = request.get_json(silent=True) or {}
        kind = str(body.get("kind", "resource"))
        if kind not in ("resource", "wall_motif"):
            return jsonify({"ok": False, "error": f"unknown kind {kind!r}"})
        size = len(loaded.grid)
        agent_row, agent_col = loaded.agent_pos
        row = body.get("row")
        col = body.get("col")
        if row is not None and col is not None:
            outcome = _hello_cells(loaded, kind, (int(row), int(col)))
            if isinstance(outcome, str):
                return jsonify({"ok": False, "error": outcome})
            cells = outcome
        else:
            anchors = [
                (r, c)
                for r in range(size)
                for c in range(size)
                if max(abs(r - agent_row), abs(c - agent_col)) > 1
                and loaded.grid[r][c] == 0
            ]
            random.shuffle(anchors)
            cells = []
            for anchor in anchors:
                outcome = _hello_cells(loaded, kind, anchor)
                if not isinstance(outcome, str) and all(
                    max(abs(r - agent_row), abs(c - agent_col)) > 1
                    for r, c in outcome
                ):
                    cells = outcome
                    break
            if not cells:
                return jsonify(
                    {"ok": False, "error": "no legal spot available"}
                )
        inbox = run_dir / "perturbation_inbox"
        if kind == "resource":
            write_trigger_request(
                inbox, "add_resource", {"cell": list(cells[0])}
            )
        else:
            for r, c in cells:
                write_trigger_request(
                    inbox, "set_cell_state", {"cell": [r, c], "state": "wall"}
                )
        return jsonify(
            {"ok": True, "kind": kind, "cells": [list(c) for c in cells]}
        )

    @app.route("/audit")
    def audit() -> str:
        audit_record = loaders.aggregate_llm_audit(run_dir)
        latency = latency_distribution(audit_record.records)
        return render_template(
            "audit.html", audit=audit_record, latency=latency
        )

    @app.route("/admissibility")
    def admissibility() -> str:
        faithfulness_results = [
            o.value
            for o in loaders.load_faithfulness_results(
                run_dir / "mirror" / "faithfulness.jsonl"
            )
            if o.value is not None
        ]
        stability_results = [
            o.value
            for o in loaders.load_stability_results(
                run_dir / "mirror" / "stability.jsonl"
            )
            if o.value is not None
        ]
        batch = compute_admissibility(
            faithfulness_results=faithfulness_results,
            stability_results=stability_results,
            run_id=run_id,
        )
        return render_template("admissibility.html", batch=batch)

    @app.errorhandler(404)
    def not_found(error: Exception) -> tuple[str, int]:
        return (
            render_template(
                "error.html", message="No such page or record."
            ),
            404,
        )

    return app
