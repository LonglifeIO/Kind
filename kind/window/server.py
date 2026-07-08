"""Window's HTTP server — a small read-only Flask app.

The server serves on ``localhost:<port>``; the host's Tailscale setup
handles remote access from other devices on the Tailnet without any
code here. Window does not authenticate, encrypt, or restrict access —
it assumes any caller reaching the port is authorized, because the
Tailscale ACL is what gates that.

Six routes, each read-only:

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

import time
from pathlib import Path

from flask import Flask, abort, render_template

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
        return render_template(
            "live.html", state=state, error=error, age_s=age_s
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
