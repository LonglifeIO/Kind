"""Probe 4.5 Phase 3 — instrument validation: the GO/NO-GO.

Prereg §6: before the real question is asked, the fixed-surprise allocation
discriminator must (a) **fire** on a system engineered to have a foreground
— the reward toy, trained to its fixed budget on the same fault-on physics,
scored for surprise through the frozen pilot instrument (one lens for toy
and Io) — with Δ_alloc ≥ 0.20 under matching; and (b) **stay silent** on
the flat engine — the precision-0 pilot's own greedy eval — with
|Δ_alloc| < 0.05 under matching. Both land → GO (Phase 4). Either failure
→ STOP: instrument failure, not evidence about Io; at most one amendment
cycle, as a new dated doc.

Eval protocol: the §5 pattern — eval seeds 9500–9507 × 20 greedy-eval
episodes each, fault-on worlds, one allocation block per policy.

Usage::

    python scripts/run_probe4_5_phase3_validation.py \
        [--pilot-dir runs/probe4_5_phase3_pilot] \
        [--toy-seed 4503] [--toy-steps 300000] \
        [--out runs/probe4_5_phase3_validation/verdict.json]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from kind.agents.actor import Actor
from kind.agents.ensemble import LatentDisagreementEnsemble
from kind.agents.world_model import WorldModel, WorldModelConfig
from kind.env.grid_world import GridWorldConfig
from kind.observer.allocation import (
    AllocationStep,
    GreedyActorPolicy,
    allocation_block,
    block_report_to_jsonable,
    collect_allocation_steps,
    pilot_gate,
    toy_gate,
)
from kind.observer.reward_toy import RewardToy, RewardToyConfig

#: The §5 eval seed series.
EVAL_SEEDS = tuple(range(9500, 9508))
EVAL_EPISODES_PER_SEED = 20


def _latest_checkpoint(checkpoints_dir: Path) -> Path:
    candidates = sorted(
        p
        for p in checkpoints_dir.iterdir()
        if p.is_dir() and not p.name.endswith(".staging")
    )
    if not candidates:
        raise FileNotFoundError(f"no checkpoints under {checkpoints_dir}")
    return candidates[-1]


def _load_pilot(
    checkpoint_dir: Path,
) -> tuple[WorldModel, LatentDisagreementEnsemble, Actor]:
    weights = load_file(str(checkpoint_dir / "weights.safetensors"))

    def split(prefix: str) -> dict[str, Any]:
        return {
            k.removeprefix(prefix): v
            for k, v in weights.items()
            if k.startswith(prefix)
        }

    wm = WorldModel(WorldModelConfig(energy_decoder_bounded=True))
    wm.load_state_dict(split("world_model."))
    wm.eval()
    ensemble = LatentDisagreementEnsemble(
        h_dim=200, z_dim=16, action_dim=5, K=5
    )
    ensemble.load_state_dict(split("ensemble."))
    ensemble.eval()
    actor = Actor(h_dim=200, z_dim=16, action_dim=5, mlp_hidden=200)
    actor.load_state_dict(split("actor."))
    actor.eval()
    return wm, ensemble, actor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-dir", type=str, default="runs/probe4_5_phase3_pilot"
    )
    parser.add_argument("--toy-seed", type=int, default=4503)
    parser.add_argument("--toy-steps", type=int, default=300_000)
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES_PER_SEED)
    parser.add_argument(
        "--out",
        type=str,
        default="runs/probe4_5_phase3_validation/verdict.json",
    )
    args = parser.parse_args()

    checkpoint_dir = _latest_checkpoint(Path(args.pilot_dir) / "checkpoints")
    wm, ensemble, actor = _load_pilot(checkpoint_dir)
    grid_cfg = GridWorldConfig(energy_fault_enabled=True)

    print(f"[toy] training ({args.toy_steps} steps, seed {args.toy_seed})")
    toy = RewardToy(
        RewardToyConfig(
            grid_world_config=grid_cfg,
            seed=args.toy_seed,
            train_steps=args.toy_steps,
        )
    )
    toy_stats = toy.train()
    print(
        f"[toy] reward {toy_stats.mean_reward_first_block:.4f} -> "
        f"{toy_stats.mean_reward_final_block:.4f}; "
        f"|td| {toy_stats.mean_abs_td_first_block:.4f} -> "
        f"{toy_stats.mean_abs_td_final_block:.4f}"
    )

    toy_steps: list[AllocationStep] = []
    pilot_steps: list[AllocationStep] = []
    greedy = GreedyActorPolicy(actor)
    for seed in EVAL_SEEDS:
        print(f"[collect] eval seed {seed}")
        toy_steps += collect_allocation_steps(
            wm, ensemble, toy, grid_cfg, seed=seed, episodes=args.episodes
        )
        pilot_steps += collect_allocation_steps(
            wm, ensemble, greedy, grid_cfg, seed=seed, episodes=args.episodes
        )

    toy_block = allocation_block(toy_steps, label="reward_toy")
    pilot_block = allocation_block(pilot_steps, label="precision_0_pilot")
    positive = toy_gate(toy_block)
    negative = pilot_gate(pilot_block)
    go = positive.passed and negative.passed

    verdict: dict[str, Any] = {
        "LABEL": (
            "Probe 4.5 Phase 3 GO/NO-GO — prereg §6 instrument validation. "
            "GO requires the toy to trip the discriminator at 2x headroom "
            "AND the precision-0 pilot to stay silent."
        ),
        "lens_checkpoint": str(checkpoint_dir),
        "eval": (
            f"{len(EVAL_SEEDS)} seeds x {args.episodes} episodes, fault-on"
        ),
        "toy_training": asdict(toy_stats),
        "positive_control": asdict(positive),
        "negative_control": asdict(negative),
        "toy_block": block_report_to_jsonable(toy_block),
        "pilot_block": block_report_to_jsonable(pilot_block),
        "GO": go,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(verdict, indent=2))
    print(
        f"\n[verdict] toy Δ={positive.delta} (needs ≥ {positive.threshold}) "
        f"passed={positive.passed}"
    )
    print(
        f"[verdict] pilot Δ={negative.delta} (needs |Δ| < "
        f"{negative.threshold}) passed={negative.passed}"
    )
    print(f"[verdict] GO = {go}")
    print(f"[out] wrote {out_path}")
    if not go:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
