"""Probe 3.5 — run the Amendment-02 §6 oracle feasibility check.

Runs the pre-committed scripted nearest-resource forager under the **current
default physics** per the confirmed F1/F2 protocol (in-band occupancy ≥ 70%
over 8 seeds × 20 episodes, fixed B0a′ band 0.6 ± 0.15) and reports pass/fail.

If the default physics fails, candidate `(decay, move_cost, replenish)` triples
are evaluated **by the oracle criterion** (the pre-committed disposition rule)
and the best is reported with its feasibility numbers **for builder adoption —
never written into `GridWorldConfig`**.

Usage::

    python scripts/run_probe3_5_oracle_feasibility.py [--seeds 8] [--episodes 20]
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace
from typing import Any

from kind.env.grid_world import GridWorldConfig
from kind.observer.oracle_forager import run_oracle_feasibility

F1_THRESHOLD = 0.70


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    default_cfg = GridWorldConfig()
    report = run_oracle_feasibility(
        default_cfg,
        n_seeds=args.seeds,
        episodes_per_seed=args.episodes,
        threshold=F1_THRESHOLD,
    )
    out: dict[str, Any] = {
        "default_physics": {
            "energy_base_decay": default_cfg.energy_base_decay,
            "energy_move_cost": default_cfg.energy_move_cost,
            "energy_replenish_per_resource": default_cfg.energy_replenish_per_resource,
        },
        "band": [report.band_low, report.band_high],
        "protocol": {"F1": F1_THRESHOLD, "F2_seeds": args.seeds, "F2_episodes": args.episodes},
        "per_seed_occupancy": list(report.per_seed_occupancy),
        "pooled_occupancy": report.pooled_occupancy,
        "passed": report.passed,
    }

    if not report.passed:
        # Pre-committed disposition: select physics by the oracle criterion;
        # report for builder adoption — never self-applied.
        candidates = []
        for decay, move, repl in itertools.product(
            [0.08, 0.06, 0.04, 0.02], [0.04, 0.02, 0.01], [0.8, 1.5, 2.5]
        ):
            cfg = replace(
                default_cfg,
                energy_base_decay=decay,
                energy_move_cost=move,
                energy_replenish_per_resource=repl,
            )
            r = run_oracle_feasibility(
                cfg,
                n_seeds=args.seeds,
                episodes_per_seed=args.episodes,
                threshold=F1_THRESHOLD,
            )
            candidates.append(
                {
                    "decay": decay,
                    "move_cost": move,
                    "replenish": repl,
                    "pooled_occupancy": r.pooled_occupancy,
                    "passed": r.passed,
                }
            )
        best = max(candidates, key=lambda c: float(c["pooled_occupancy"]))
        out["oracle_selected_candidate"] = best
        out["candidates"] = candidates

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
