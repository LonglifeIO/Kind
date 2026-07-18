"""Probe 4.5 Phase 2 gate — oracle feasibility under fault-on physics.

Prereg §4: fault intervals change the physics envelope, so the world must be
re-shown winnable by competence before any training run — the scripted
regulator must still hold the band (in-band occupancy ≥ 70%, 8 seeds × 20
episodes; the Amendment-02 bar under the new envelope). If it cannot, the
band is amended by dated doc before any run — Phase 4 failures must stay
attributable to the agent-side pathway, never the world.

Runs the standing instrument twice: default physics (the reference the
Amendment-02 record established) and fault-on physics at the frozen §4 band
(the gate). Prints a JSON report; the journal entry is written from it.

Usage::

    python scripts/run_probe4_5_phase2_oracle_gate.py \
        [--out runs/probe4_5_phase2/oracle_gate.json]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from kind.env.grid_world import GridWorldConfig
from kind.observer.oracle_forager import run_oracle_feasibility


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=str, default="runs/probe4_5_phase2/oracle_gate.json"
    )
    args = parser.parse_args()

    # The frozen §4 band rides in as the config defaults; only the enable
    # flag is set (no value re-chosen here).
    reference = run_oracle_feasibility(GridWorldConfig())
    gate = run_oracle_feasibility(
        GridWorldConfig(energy_fault_enabled=True)
    )

    report: dict[str, Any] = {
        "LABEL": (
            "Probe 4.5 Phase 2 oracle-feasibility gate (prereg §4): the "
            "world must stay winnable by competence under fault-on physics."
        ),
        "reference_default_physics": asdict(reference),
        "gate_fault_on_physics": asdict(gate),
        "occupancy_cost_of_faults": round(
            reference.pooled_occupancy - gate.pooled_occupancy, 4
        ),
        "gate_passed": gate.passed,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\n[out] wrote {out_path}")
    if not gate.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
