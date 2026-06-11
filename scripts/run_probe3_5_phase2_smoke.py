"""Probe 3.5 Phase 2 Step 4c — mechanism smoke. NON-VERDICT.

One short run with the preference live: S1-baseline precision (the 1.0× grid
point, instantiated from the burn-in measurement), σ = 0.075 (S2′ 0.5× band
halfwidth), lag 1, trained preference-on to P3 (age-matched to the
baseline/assay discipline). Reports **directional movement only** —

- does energy leave the rail at all?
- does behavioral entropy survive at all?

**This is mechanism evidence, not the probe's answer.** The pass / dominant /
inert verdict belongs to Phase 4, rendered mechanically from the frozen
signatures over the full pre-committed sweep. Nothing here is tuned, and
nothing here is to be read against O1/A1b as a verdict.

Usage::

    python scripts/run_probe3_5_phase2_smoke.py \
        [--steps 5000] [--seed 0] [--eval-seeds 8] [--episodes 20] \
        [--out runs/probe3_5_phase2/smoke_s1_baseline.json]
"""

from __future__ import annotations

import argparse
import json
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from kind.agents.preference import EnergyPreferenceConfig
from kind.agents.views import PolicyView
from kind.agents.world_model import WorldModelConfig
from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import GridWorld, GridWorldConfig
from kind.env.transport import EnvTransportClient, EnvTransportServer
from kind.training.runner import Runner, RunnerConfig

SETPOINT = 0.6  # B0b (frozen)
BAND_HALFWIDTH = 0.15  # B0a' (Amendment 02)
SMOKE_SIGMA = 0.075  # S2' 0.5x band halfwidth — the S4a-analog mid operating value
FLOOR_THRESHOLD = 0.05
CEILING_THRESHOLD = 0.95


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


@torch.no_grad()
def _eval_seed(
    runner: Runner, grid_cfg: GridWorldConfig, seed: int, episodes: int
) -> dict[str, Any]:
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
            visit = np.zeros_like(visit)

    true_arr = np.asarray(true_list)
    return {
        "true": true_arr,
        "per_episode_entropy": per_episode_entropy,
        "disagreements": np.asarray(disagreements),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000, help="training age P3")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seeds", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--out", type=str, default="runs/probe3_5_phase2/smoke_s1_baseline.json"
    )
    args = parser.parse_args()

    s1 = json.loads(
        Path("runs/probe3_5_phase2/s1_instantiation.json").read_text()
    )
    precision = float(s1["s1_baseline_precision_1x"])
    baseline = json.loads(
        Path("runs/probe3_5_phase2/step0_baseline.json").read_text()
    )

    grid_cfg = replace(GridWorldConfig(), energy_obs_noise_sigma=SMOKE_SIGMA)
    preference = EnergyPreferenceConfig(precision=precision)
    print(
        f"[smoke] S1-baseline precision={precision:.4f}, sigma={SMOKE_SIGMA}, "
        f"lag={grid_cfg.energy_obs_lag}, P3={args.steps}"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tmp = tempfile.mkdtemp(prefix="probe3_5_p2_smoke_")
    tmp_path = Path(tmp)
    with _transport_pair(grid_cfg, seed=args.seed, run_id="probe3_5_p2_smoke") as (
        client,
        env_server,
    ):
        run_cfg = RunnerConfig(
            world_model_config=WorldModelConfig(energy_dedicated_dims=0),
            run_id="probe3_5_p2_smoke",
            telemetry_dir=tmp_path / "telemetry",
            checkpoints_dir=tmp_path / "checkpoints",
            warmup_env_steps=args.warmup,
            dream_cadence_env_steps=10_000_000,
            checkpoint_every_n_env_steps=10_000_000,
            energy_telemetry=True,
            energy_preference=preference,
            device="cpu",
        )
        runner = Runner(run_cfg, client, env_server=env_server)
        runner.run(total_env_steps=args.steps)

    try:
        pooled_true: list[np.ndarray] = []
        pooled_entropy: list[float] = []
        pooled_epistemic: list[np.ndarray] = []
        for s in range(args.eval_seeds):
            ev = _eval_seed(runner, grid_cfg, seed=9000 + s, episodes=args.episodes)
            pooled_true.append(ev["true"])
            pooled_entropy.extend(ev["per_episode_entropy"])
            pooled_epistemic.append(ev["disagreements"])
        final_share = runner._last_pragmatic_share
        final_pragmatic = runner._last_pragmatic_value
        final_epistemic = runner._last_epistemic_value
    finally:
        runner.close()

    pooled = np.concatenate(pooled_true)
    ep_len = grid_cfg.episode_length
    per_ep = pooled.reshape(-1, ep_len)
    in_band = np.abs(per_ep - SETPOINT) <= BAND_HALFWIDTH
    entropy_mean = float(np.mean(pooled_entropy))
    epistemic_mean = float(np.concatenate(pooled_epistemic).mean())

    base_entropy = baseline["entropy_reference_A1b"][
        "positional_entropy_per_episode_nats"
    ]["mean"]
    base_epistemic = baseline["entropy_reference_A1b"][
        "epistemic_activity_per_episode"
    ]["mean"]
    base_in_band = baseline["degenerate_null"]["pooled_in_band_occupancy"]
    base_floor = baseline["degenerate_null"]["floor_frac_lt_0.05"]

    out: dict[str, Any] = {
        "LABEL": (
            "NON-VERDICT — mechanism evidence only (Step 4c smoke). "
            "Directional movement at one grid point, one training seed. The "
            "probe's verdict belongs to Phase 4 under the frozen signatures."
        ),
        "config": {
            "precision_S1_1x": precision,
            "sigma": SMOKE_SIGMA,
            "lag": grid_cfg.energy_obs_lag,
            "training_age_P3": args.steps,
            "train_seed": args.seed,
            "eval": f"{args.eval_seeds} seeds x {args.episodes} episodes",
            "grid_energy": {
                k: v for k, v in asdict(grid_cfg).items() if k.startswith("energy_")
            },
        },
        "directional": {
            "true_energy_mean": float(pooled.mean()),
            "true_energy_std": float(pooled.std()),
            "floor_frac_lt_0.05": float(np.mean(pooled < FLOOR_THRESHOLD)),
            "ceiling_frac_gt_0.95": float(np.mean(pooled > CEILING_THRESHOLD)),
            "in_band_occupancy": float(in_band.mean()),
            "in_band_occupancy_final_half": float(in_band[:, ep_len // 2 :].mean()),
            "positional_entropy_per_episode_nats": entropy_mean,
            "epistemic_activity_per_step": epistemic_mean,
        },
        "vs_degenerate_null_directional_only": {
            "energy_left_the_rail": {
                "null_floor_frac": base_floor,
                "smoke_floor_frac": float(np.mean(pooled < FLOOR_THRESHOLD)),
                "null_in_band": base_in_band,
                "smoke_in_band": float(in_band.mean()),
            },
            "entropy_survived": {
                "null_entropy_nats": base_entropy,
                "smoke_entropy_nats": entropy_mean,
                "ratio": entropy_mean / base_entropy if base_entropy > 0 else None,
                "null_epistemic": base_epistemic,
                "smoke_epistemic": epistemic_mean,
                "epistemic_ratio": (
                    epistemic_mean / base_epistemic if base_epistemic > 0 else None
                ),
            },
        },
        "final_training_step_decomposition": {
            "pragmatic_value": final_pragmatic,
            "epistemic_value": final_epistemic,
            "pragmatic_share": final_share,
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["directional"], indent=2))
    print(json.dumps(out["final_training_step_decomposition"], indent=2))
    print(f"[out] wrote {out_path}")


if __name__ == "__main__":
    main()
