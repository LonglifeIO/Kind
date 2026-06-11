"""Probe 3.5 Phase 2 Step 0 — instantiate the degenerate baseline, formally.

Amendment 02 §1 (CONFIRMED 2026-06-11): the pure-epistemic baseline *is* the
degenerate distribution — the energy-blind, rail-pinned ``true_energy``
distribution of the trained epistemic-only actor (``pragmatic_value`` ≡ 0),
measured under the **operative (default) physics** per the **frozen §3
mechanics, unchanged**: P1 = 8 seeds × P2 = 20 episodes (200 steps each) at
training age P3 = 5000 env-steps. The rail identity (floor vs ceiling) is
**recorded as a measurement, not assumed**.

This run records, into the dated results doc:

- rail identity + ``true_energy`` mean/std/floor/ceiling fractions;
- in-band occupancy on the **fixed** B0a′ band [0.45, 0.75] (expected ≈ 0%),
  plus the O1 steady-state window variant (final 50% of each episode);
- the behavioral-entropy reference values the A1b ratios denominate against:
  per-episode positional entropy (frozen Shared definition: Shannon entropy of
  the grid-cell visitation histogram *over an episode*) and per-episode
  epistemic activity (mean K=5 ensemble disagreement encountered per step);
- the typical epistemic-term magnitude (mean / p90 per-step disagreement) —
  the S1 precision-grid input, measured on the Phase-2 **pre-preference
  burn-in instance** (this instance: trained with ``pragmatic_value`` still
  zero), per Amendment 02 §2.

The Phase-1 eval artifacts (one 1500-step contiguous rollout, one seed) do
**not** satisfy the frozen 8 × 20 × 200 mechanics, so this measurement runs
fresh. Seeds follow the amended-gate harness precedent: P1 means 8 *eval* env
seeds on the single P3-trained instance (the baseline is a property of the
trained-indifferent actor, not of training-seed variance).

Usage::

    python scripts/run_probe3_5_phase2_step0_baseline.py \
        [--steps 5000] [--seed 0] [--eval-seeds 8] [--episodes 20] \
        [--out runs/probe3_5_phase2/step0_baseline.json]
"""

from __future__ import annotations

import argparse
import json
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from kind.agents.views import PolicyView
from kind.agents.world_model import WorldModelConfig
from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import GridWorld, GridWorldConfig
from kind.env.transport import EnvTransportClient, EnvTransportServer
from kind.training.runner import Runner, RunnerConfig

SETPOINT = 0.6  # B0b (frozen)
BAND_HALFWIDTH = 0.15  # B0a' (Amendment 02 §2 — fixed, absolute)
FLOOR_THRESHOLD = 0.05  # R5 convention (Amendment 01 §4)
CEILING_THRESHOLD = 0.95  # recalibration-record convention


@contextmanager
def _transport_pair(
    grid_cfg: GridWorldConfig, seed: int, run_id: str
) -> Iterator[tuple[EnvTransportClient, EnvServer]]:
    config = EnvServerConfig(
        grid_world_config=grid_cfg,
        seed=seed,
        world_event_handler=lambda _r: None,
        run_id=run_id,
    )
    env_server = EnvServer(config)
    server = EnvTransportServer(env_server, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = EnvTransportClient(
        host="127.0.0.1",
        port=server.actual_port,
        world_event_handler=lambda _r: None,
    )
    try:
        yield client, env_server
    finally:
        try:
            client.close()
        finally:
            server.shutdown()
            thread.join(timeout=5.0)


def _shannon_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def _train(
    grid_cfg: GridWorldConfig, steps: int, warmup: int, seed: int
) -> Runner:
    torch.manual_seed(seed)
    np.random.seed(seed)
    wm_cfg = WorldModelConfig(energy_dedicated_dims=0)
    tmp = tempfile.mkdtemp(prefix="probe3_5_p2_step0_")
    tmp_path = Path(tmp)
    with _transport_pair(grid_cfg, seed=seed, run_id="probe3_5_p2_step0") as (
        client,
        env_server,
    ):
        run_cfg = RunnerConfig(
            world_model_config=wm_cfg,
            run_id="probe3_5_p2_step0",
            telemetry_dir=tmp_path / "telemetry",
            checkpoints_dir=tmp_path / "checkpoints",
            warmup_env_steps=warmup,
            dream_cadence_env_steps=10_000_000,  # suppress waking-planning rollout
            checkpoint_every_n_env_steps=10_000_000,
            energy_telemetry=True,
            device="cpu",
        )
        runner = Runner(run_cfg, client, env_server=env_server)
        runner.run(total_env_steps=steps)
    return runner


@torch.no_grad()
def _eval_seed(
    runner: Runner,
    grid_cfg: GridWorldConfig,
    seed: int,
    episodes: int,
) -> dict[str, Any]:
    """One eval seed: ``episodes`` × 200-step episodes, contiguous (the env
    auto-resets at the soft boundary; energy carries across it — frozen
    resolved sub-decision #2). Greedy actor, frozen PolicyView."""
    device = runner.device
    wm = runner.world_model
    actor = runner.actor
    ep_len = grid_cfg.episode_length
    n_steps = ep_len * episodes
    world = GridWorld(grid_cfg, seed=seed)
    step = world.reset()

    h = torch.zeros(1, wm.config.h_dim, device=device)
    z = torch.zeros(1, wm.config.z_dim, device=device)
    a_prev = torch.zeros(1, dtype=torch.long, device=device)
    zero_scalar = torch.zeros((), device=device)

    true_list: list[float] = []
    disagreements: list[float] = []
    per_episode_entropy: list[float] = []
    per_episode_epistemic: list[float] = []
    visit = np.zeros((grid_cfg.grid_size, grid_cfg.grid_size), dtype=np.int64)

    for t in range(n_steps):
        obs_np = step.observation.astype(np.float32) / 255.0
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).unsqueeze(0).to(device)
        sensed_t = torch.tensor(
            [[step.sensed_energy]], device=device, dtype=torch.float32
        )
        wm_step = wm.step(obs_t, h, z, a_prev, sensed_energy=sensed_t)
        intrinsic = runner.ensemble.disagreement(h, z, a_prev)
        disagreements.append(float(intrinsic.reshape(-1)[0].item()))
        view = PolicyView(
            h=wm_step.h, z=wm_step.z, self_prediction_error=zero_scalar
        )
        action = int(actor.act_greedy(view).reshape(-1)[0].item())

        gs = world.state
        visit[gs.agent_pos] += 1
        true_list.append(float(gs.true_energy))

        step = world.step(action)
        h, z, a_prev = (
            wm_step.h,
            wm_step.z,
            torch.tensor([action], dtype=torch.long, device=device),
        )

        if (t + 1) % ep_len == 0:
            per_episode_entropy.append(_shannon_entropy(visit))
            per_episode_epistemic.append(
                float(np.mean(disagreements[t + 1 - ep_len : t + 1]))
            )
            visit = np.zeros_like(visit)

    true_arr = np.asarray(true_list).reshape(episodes, ep_len)
    in_band = np.abs(true_arr - SETPOINT) <= BAND_HALFWIDTH
    final_half = in_band[:, ep_len // 2 :]
    return {
        "true": true_arr,
        "in_band_occupancy": float(in_band.mean()),
        "in_band_occupancy_final_half": float(final_half.mean()),
        "per_episode_entropy": per_episode_entropy,
        "per_episode_epistemic": per_episode_epistemic,
        "disagreements": np.asarray(disagreements),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000, help="training age P3")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seeds", type=int, default=8, help="P1")
    parser.add_argument("--episodes", type=int, default=20, help="P2")
    parser.add_argument(
        "--out", type=str, default="runs/probe3_5_phase2/step0_baseline.json"
    )
    parser.add_argument(
        "--checkpoint-out",
        type=str,
        default="runs/probe3_5_phase2/step0_burnin_checkpoint.pt",
    )
    args = parser.parse_args()

    grid_cfg = GridWorldConfig()  # default physics (oracle-feasible, Amendment 02 §6)
    runner = _train(grid_cfg, args.steps, args.warmup, seed=args.seed)
    try:
        per_seed: list[dict[str, Any]] = []
        pooled_true: list[np.ndarray] = []
        pooled_entropy: list[float] = []
        pooled_epistemic: list[float] = []
        pooled_disagreement: list[np.ndarray] = []
        for s in range(args.eval_seeds):
            ev = _eval_seed(
                runner, grid_cfg, seed=9000 + s, episodes=args.episodes
            )
            true_arr = ev["true"].reshape(-1)
            seed_summary = {
                "eval_seed": 9000 + s,
                "true_mean": float(true_arr.mean()),
                "true_std": float(true_arr.std()),
                "floor_frac": float(np.mean(true_arr < FLOOR_THRESHOLD)),
                "ceiling_frac": float(np.mean(true_arr > CEILING_THRESHOLD)),
                "in_band_occupancy": ev["in_band_occupancy"],
                "in_band_occupancy_final_half": ev["in_band_occupancy_final_half"],
                "pos_entropy_per_episode_mean": float(
                    np.mean(ev["per_episode_entropy"])
                ),
                "pos_entropy_per_episode_sd": float(
                    np.std(ev["per_episode_entropy"])
                ),
                "epistemic_per_episode_mean": float(
                    np.mean(ev["per_episode_epistemic"])
                ),
                "epistemic_per_episode_sd": float(
                    np.std(ev["per_episode_epistemic"])
                ),
            }
            per_seed.append(seed_summary)
            pooled_true.append(true_arr)
            pooled_entropy.extend(ev["per_episode_entropy"])
            pooled_epistemic.extend(ev["per_episode_epistemic"])
            pooled_disagreement.append(ev["disagreements"])
            print(f"[seed {9000 + s}] {json.dumps(seed_summary)}")
    finally:
        runner.close()

    pooled = np.concatenate(pooled_true)
    disagreement_pooled = np.concatenate(pooled_disagreement)
    floor_frac = float(np.mean(pooled < FLOOR_THRESHOLD))
    ceiling_frac = float(np.mean(pooled > CEILING_THRESHOLD))
    rail = (
        "floor"
        if floor_frac > ceiling_frac
        else ("ceiling" if ceiling_frac > floor_frac else "indeterminate")
    )

    def _ms(key: str) -> dict[str, float]:
        vals = np.asarray([p[key] for p in per_seed])
        return {"mean": float(vals.mean()), "sd": float(vals.std())}

    out: dict[str, Any] = {
        "config": {
            "training_age_P3": args.steps,
            "train_seed": args.seed,
            "P1_eval_seeds": args.eval_seeds,
            "P2_episodes_per_seed": args.episodes,
            "physics": "default (oracle-feasible per Amendment 02 §6)",
            "grid_energy": {
                k: v for k, v in asdict(grid_cfg).items() if k.startswith("energy_")
            },
            "band_B0a_prime": [SETPOINT - BAND_HALFWIDTH, SETPOINT + BAND_HALFWIDTH],
            "actor": "trained epistemic-only (pragmatic_value ≡ 0), greedy eval",
        },
        "degenerate_null": {
            "rail_identity": rail,
            "true_energy_mean": float(pooled.mean()),
            "true_energy_std": float(pooled.std()),
            "floor_frac_lt_0.05": floor_frac,
            "ceiling_frac_gt_0.95": ceiling_frac,
            "in_band_occupancy": _ms("in_band_occupancy"),
            "in_band_occupancy_final_half_O1_window": _ms(
                "in_band_occupancy_final_half"
            ),
            "pooled_in_band_occupancy": float(
                np.mean(np.abs(pooled - SETPOINT) <= BAND_HALFWIDTH)
            ),
        },
        "entropy_reference_A1b": {
            "positional_entropy_per_episode_nats": {
                "mean": float(np.mean(pooled_entropy)),
                "sd": float(np.std(pooled_entropy)),
                "n_episodes": len(pooled_entropy),
            },
            "positional_entropy_max_nats": float(np.log(grid_cfg.grid_size**2)),
            "epistemic_activity_per_episode": {
                "mean": float(np.mean(pooled_epistemic)),
                "sd": float(np.std(pooled_epistemic)),
                "n_episodes": len(pooled_epistemic),
            },
            "per_seed": per_seed,
        },
        "epistemic_magnitude_S1_input": {
            "mean_per_step_disagreement": float(disagreement_pooled.mean()),
            "p90_per_step_disagreement": float(
                np.percentile(disagreement_pooled, 90)
            ),
            "provenance": (
                "measured on this instance — the Phase-2 pre-preference burn-in "
                "(trained preference-off to P3), per Amendment 02 §2 S1 "
                "instantiation point"
            ),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[out] wrote {out_path}")

    ckpt_path = Path(args.checkpoint_out)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "world_model": runner.world_model.state_dict(),
            "ensemble": runner.ensemble.state_dict(),
            "actor": runner.actor.state_dict(),
            "grid_config": asdict(grid_cfg),
            "training_age": args.steps,
            "train_seed": args.seed,
            "note": "Probe 3.5 Phase 2 Step 0 burn-in instance (pragmatic ≡ 0)",
        },
        ckpt_path,
    )
    print(f"[checkpoint] wrote {ckpt_path}")


if __name__ == "__main__":
    main()
