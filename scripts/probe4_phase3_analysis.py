"""Probe 4 Phase 3 — positive-control analysis / GO-NO-GO renderer.

Loads a run's telemetry, builds the three-way event windows, runs the
frozen-prereg detectors (§2a basin separation, §2b dream
over-representation, §3c per-event divergence), and renders the §6
positive-control verdict mechanically. Usage:

    python scripts/probe4_phase3_analysis.py runs/probe4_phase3_positive_control
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray

from kind.observer.schemas import AgentStep, WorldEvent
from kind.observer.source_separation import (
    basin_separation,
    collect_event_windows,
    dream_over_representation,
    extract_event_anchors,
    per_event_divergence,
    positive_control_verdict,
)


def _load_agent_steps(telemetry_dir: Path) -> list[AgentStep]:
    rows: list[dict[str, object]] = []
    for shard in sorted((telemetry_dir / "agent_step").glob("*.parquet")):
        rows.extend(pq.read_table(shard).to_pylist())  # type: ignore[no-untyped-call]
    rows.sort(key=lambda r: int(r["t"]))  # type: ignore[arg-type]
    return [AgentStep.model_validate(row) for row in rows]


def _load_world_events(telemetry_dir: Path) -> list[WorldEvent]:
    path = telemetry_dir / "world_event.jsonl"
    return [
        WorldEvent.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _load_dream_h_states(telemetry_dir: Path) -> NDArray[np.float64]:
    """Dream-session h-states only (dream_session_id non-null); the
    waking-planning calibration rollouts are excluded (T2b reads the
    offline machinery). Raw dict extraction — the analysis needs only
    ``sequence_h`` and the session id."""
    rows: list[list[float]] = []
    dream_dir = telemetry_dir / "dream_rollout"
    for shard in sorted(dream_dir.glob("*.parquet")):
        for record in pq.read_table(shard).to_pylist():  # type: ignore[no-untyped-call]
            if record.get("dream_session_id") is None:
                continue
            sequence_h = record.get("sequence_h") or []
            rows.extend([list(map(float, h)) for h in sequence_h])
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def main(run_dir: Path) -> int:
    telemetry = run_dir / "telemetry"
    agent_steps = _load_agent_steps(telemetry)
    world_events = _load_world_events(telemetry)
    dream_states = _load_dream_h_states(telemetry)

    anchors = extract_event_anchors(agent_steps, world_events)
    windows = collect_event_windows(agent_steps, anchors)
    counts = {
        name: sum(1 for w in windows if w.anchor.source_class == name)
        for name in ("self", "environment", "builder")
    }
    print(f"agent_steps={len(agent_steps)}  world_events={len(world_events)}")
    print(f"event windows: {counts}  dream_h_states={dream_states.shape}")

    basin = basin_separation(windows)
    print(
        f"§2a basin: S(b,s)={basin.s_builder_self:.3f}  "
        f"S(b,e)={basin.s_builder_environment:.3f}  "
        f"baseline S(e,s)={basin.s_environment_self:.3f}  "
        f"(rule: ≥{basin.required_factor}× baseline → passes={basin.passes})"
    )

    dream = dream_over_representation(dream_states, windows)
    print(
        f"§2b dream: ratio={dream.ratio}  hits B/E={dream.hits_builder}/"
        f"{dream.hits_environment}  n={dream.n_builder}/"
        f"{dream.n_environment_matched}  states={dream.n_dream_states}  "
        f"(rule: ≥{dream.threshold_r} → passes={dream.passes})"
    )

    divergence = per_event_divergence(windows)
    print(f"§3c per-event PE means: { {k: round(v, 2) for k, v in divergence.mean_pe.items()} }")
    print(
        f"§3c intrinsic-after means: { {k: round(v, 5) for k, v in divergence.mean_intrinsic_after.items()} }"
    )

    verdict = positive_control_verdict(basin, dream)
    print(
        f"§6 positive control: basin factors "
        f"({verdict.basin_factor_builder_self:.2f}, "
        f"{verdict.basin_factor_builder_environment:.2f}) "
        f"required ≥{verdict.required_basin_factor}; dream {verdict.dream_ratio} "
        f"required ≥{verdict.required_dream_ratio}"
    )
    print(f"VERDICT: {'GO' if verdict.go else 'NO-GO (STOP)'}")
    for reason in verdict.reasons:
        print(f"  - {reason}")

    (run_dir / "positive_control_verdict.json").write_text(
        json.dumps(
            {
                "go": verdict.go,
                "basin": {
                    "s_builder_self": basin.s_builder_self,
                    "s_builder_environment": basin.s_builder_environment,
                    "s_environment_self": basin.s_environment_self,
                    "passes": basin.passes,
                },
                "dream": {
                    "ratio": dream.ratio,
                    "hits_builder": dream.hits_builder,
                    "hits_environment": dream.hits_environment,
                    "n_dream_states": dream.n_dream_states,
                    "passes": dream.passes,
                },
                "counts": counts,
                "reasons": list(verdict.reasons),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0 if verdict.go else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
