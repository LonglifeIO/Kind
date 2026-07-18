"""Probe 4.5 Phase 2 gate — the belief-error profile during faults.

Prereg §4 / plan Phase 2 gate: measure, through the honest Phase-1 decoder
(eval-only), whether the fault dynamic actually opens a belief-truth gap at
the frozen band — the h-led belief's learned average dynamics should
over-read true energy while a fault drains it faster than the model
expects. If faults open no gap, the dynamic is a dead column at the
physics level and the §4 two-sided reading applies (a dated amendment
question, never a mid-run tune).

Method: teacher-forced collection (the standing ``decode_honesty``
collector, unmodified) on **fault-on** worlds at the instrument's own eval
seed bases; each trajectory's per-step fault mask is reconstructed exactly
by replaying the schedule on a dummy world with the same seed — sound
because the fault process is action-independent and a pure function of
seed + step count (``tests/test_energy_fault.py`` pins this). Profile =
decode-vs-true bias and |error| on fault steps vs non-fault steps, per
source and pooled.

Usage::

    python scripts/run_probe4_5_phase2_belief_error_profile.py \
        [--checkpoint runs/probe4_5_phase1/checkpoints/<latest>] \
        [--out runs/probe4_5_phase2/belief_error_profile.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from kind.agents.actor import Actor
from kind.agents.world_model import WorldModel, WorldModelConfig
from kind.env.grid_world import GridWorld, GridWorldConfig
from kind.observer.decode_honesty import (
    DEFAULT_SEED_BASES,
    collect_teacher_forced_trajectory,
)


def _latest_checkpoint(checkpoints_dir: Path) -> Path:
    candidates = sorted(
        p for p in checkpoints_dir.iterdir()
        if p.is_dir() and not p.name.endswith(".staging")
    )
    if not candidates:
        raise FileNotFoundError(f"no checkpoints under {checkpoints_dir}")
    return candidates[-1]


def _load_phase1_modules(checkpoint_dir: Path) -> tuple[WorldModel, Actor]:
    """Load the Phase-1 instance (F2 bounded head ON) from a checkpoint."""
    weights = load_file(str(checkpoint_dir / "weights.safetensors"))
    wm = WorldModel(WorldModelConfig(energy_decoder_bounded=True))
    wm.load_state_dict(
        {
            k.removeprefix("world_model."): v
            for k, v in weights.items()
            if k.startswith("world_model.")
        }
    )
    wm.eval()
    actor = Actor(h_dim=200, z_dim=16, action_dim=5, mlp_hidden=200)
    actor.load_state_dict(
        {
            k.removeprefix("actor."): v
            for k, v in weights.items()
            if k.startswith("actor.")
        }
    )
    actor.eval()
    return wm, actor


def _fault_mask_for_seed(
    grid_cfg: GridWorldConfig, seed: int, n_steps: int
) -> np.ndarray:
    """Replay the (action-independent) fault schedule on a dummy world."""
    world = GridWorld(grid_cfg, seed=seed)
    world.reset()
    mask = np.zeros(n_steps, dtype=np.bool_)
    for i in range(n_steps):
        world.step(4)
        mask[i] = world.fault_active
    return mask


def _profile(
    decode: np.ndarray, true: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    def stats(sel: np.ndarray) -> dict[str, float | int]:
        if int(sel.sum()) == 0:
            return {"n": 0}
        d, t = decode[sel], true[sel]
        return {
            "n": int(sel.sum()),
            "bias_decode_minus_true": float((d - t).mean()),
            "abs_error_mean": float(np.abs(d - t).mean()),
            "true_mean": float(t.mean()),
            "decode_mean": float(d.mean()),
        }

    fault = stats(mask)
    clear = stats(~mask)
    gap = (
        round(
            float(fault["bias_decode_minus_true"])
            - float(clear["bias_decode_minus_true"]),
            4,
        )
        if fault.get("n", 0) and clear.get("n", 0)
        else None
    )
    return {
        "fault_steps": fault,
        "clear_steps": clear,
        # The gate quantity: extra over-read during faults (positive =
        # the belief lags the faster drain, as the dynamic intends).
        "fault_minus_clear_bias": gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint dir (default: latest under runs/probe4_5_phase1/checkpoints)",
    )
    parser.add_argument("--seeds-per-source", type=int, default=4)
    parser.add_argument("--episodes-per-seed", type=int, default=10)
    parser.add_argument(
        "--out",
        type=str,
        default="runs/probe4_5_phase2/belief_error_profile.json",
    )
    args = parser.parse_args()

    checkpoint_dir = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else _latest_checkpoint(Path("runs/probe4_5_phase1/checkpoints"))
    )
    wm, actor = _load_phase1_modules(checkpoint_dir)

    # Fault-on at the frozen §4 band (defaults; only the flag is set).
    grid_cfg = GridWorldConfig(energy_fault_enabled=True)
    n_steps = grid_cfg.episode_length * args.episodes_per_seed

    per_source: dict[str, Any] = {}
    all_decode: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    all_mask: list[np.ndarray] = []
    for source, base in DEFAULT_SEED_BASES:
        decode_l: list[np.ndarray] = []
        true_l: list[np.ndarray] = []
        mask_l: list[np.ndarray] = []
        for s in range(args.seeds_per_source):
            seed = base + s
            with torch.no_grad():
                traj = collect_teacher_forced_trajectory(
                    wm,
                    actor,
                    grid_cfg,
                    policy_source=source,
                    seed=seed,
                    episodes=args.episodes_per_seed,
                )
            decode_l.append(traj.decode_energy)
            true_l.append(traj.true_energy)
            mask_l.append(_fault_mask_for_seed(grid_cfg, seed, n_steps))
        decode = np.concatenate(decode_l)
        true = np.concatenate(true_l)
        mask = np.concatenate(mask_l)
        per_source[source] = _profile(decode, true, mask)
        all_decode.append(decode)
        all_true.append(true)
        all_mask.append(mask)

    report: dict[str, Any] = {
        "LABEL": (
            "Probe 4.5 Phase 2 belief-error profile (prereg §4 / plan Phase "
            "2 gate): does the fault dynamic open a belief-truth gap, read "
            "through the honest Phase-1 decoder?"
        ),
        "checkpoint": str(checkpoint_dir),
        "per_source": per_source,
        "pooled": _profile(
            np.concatenate(all_decode),
            np.concatenate(all_true),
            np.concatenate(all_mask),
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\n[out] wrote {out_path}")


if __name__ == "__main__":
    main()
