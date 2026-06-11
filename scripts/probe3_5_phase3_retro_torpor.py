"""Probe 3.5 Phase 3 Step 2 — retro torpor check. No new runs.

Tests the conserve-without-seek hypothesis (the Phase-2 journal's torpor
watch-note) on the data that generated it: from the **preserved Phase-2
telemetry** (training-time AgentStep records — the actions Io actually took
while training), the stay-share (action 4) of the smoke run (1.0× precision)
versus the Step-0 degenerate null. Mechanism predicted, if torpor: the
out-of-band penalty taxes movement it cannot convert into foraging, so the
cheapest policy response is to stop paying the movement cost — stay-share
rises without any band gain.

One number each, plus per-1000-step blocks for spread/trajectory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def stay_stats(path: str) -> dict[str, Any]:
    table = pq.read_table(path, columns=["action_t", "t"])  # type: ignore[no-untyped-call]
    rows = sorted(table.to_pylist(), key=lambda r: r["t"])
    acts = np.array([r["action_t"] for r in rows])
    stay = acts == 4
    n = len(acts)
    blocks = [float(stay[i : i + 1000].mean()) for i in range(0, n, 1000)]
    histogram = {str(a): int((acts == a).sum()) for a in range(5)}
    return {
        "n_steps": n,
        "stay_share": float(stay.mean()),
        "stay_share_per_1000_step_blocks": blocks,
        "stay_share_final_1000": blocks[-1],
        "action_histogram": histogram,
    }


def main() -> None:
    null = stay_stats("runs/probe3_5_phase2/telemetry_step0/shard-000000.parquet")
    smoke = stay_stats("runs/probe3_5_phase2/telemetry_smoke/shard-000000.parquet")
    out = {
        "LABEL": (
            "retro torpor check (Phase 3 Step 2) — training-time action "
            "distribution from preserved Phase-2 telemetry; no new runs"
        ),
        "step0_null_training_actions": null,
        "smoke_1x_training_actions": smoke,
    }
    Path("runs/probe3_5_phase3").mkdir(parents=True, exist_ok=True)
    Path("runs/probe3_5_phase3/retro_torpor_check.json").write_text(
        json.dumps(out, indent=2)
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
