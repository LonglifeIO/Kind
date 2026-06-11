"""Probe 3.5 Phase 3 — positive control at the top of the instantiated grid.

One question: can the mechanism as built — a preference over a learned belief,
acting through imagined rollouts and policy gradients — produce detectable
displacement at the strong end of the pre-registered grid? This is
instrument-and-pathway validation, **not the verdict** (the verdict belongs to
Phase 4's frozen raise-from-low order). Both outcomes are informative:
displacement validates the sweep's interpretability; non-displacement at 10×
is a substrate finding (belief-mediated regulation insufficient at any
pre-registered strength) that gates Phase 4's design.

Configuration (builder-confirmed 2026-06-11): precision = the instantiated S1
10× point (26.2335); σ = [P3-σ: 0.075 | pre — operative baseline noise]; lag 1;
frozen mechanics — 8 eval seeds × 20 episodes × 200 steps at P3 = 5000,
age-matched against the Step-0 null. Telemetry persistent (share trajectory +
D monitor read from it). **Throwaway instance — the checkpoint is not carried
forward (plan Phase 3).**

Also runs the §8.4 signature suite on the eval data — the first exercise of
the falsification detectors on real data. Pinned thresholds are used where
they exist (A1c 95% occupancy, A1d 40% entropy ceiling); the share→1 and
camping signatures have no pinned numeric thresholds and are reported
descriptively, labeled as detector-exercise operationalizations.

Usage::

    python scripts/run_probe3_5_phase3_positive_control.py \
        [--steps 5000] [--seed 0] [--sigma 0.075] \
        [--precision-key 10.0x] [--out runs/probe3_5_phase3/positive_control.json]

The σ = 0 diagnostic (the single permitted not-green follow-up) is
``--sigma 0 --out runs/probe3_5_phase3/positive_control_sigma0.json``.
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
import pyarrow.parquet as pq
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
FLOOR_THRESHOLD = 0.05
CEILING_THRESHOLD = 0.95
STAY_ACTION = 4

# §8.4 pinned thresholds (frozen §1/§8): A1c dominance occupancy, A1d entropy
# ceiling. The other two signatures (share → 1, camping) carry no pinned
# numeric thresholds and are reported descriptively.
A1C_OCCUPANCY = 0.95
A1D_ENTROPY_FRACTION = 0.40
A3C_CAMP_WINDOW = 100  # steps (frozen A3c window, used descriptively here)


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
    act_list: list[int] = []
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
        act_list.append(action)

        step = world.step(action)
        h, z, a_prev = (
            wm_step.h,
            wm_step.z,
            torch.tensor([action], dtype=torch.long, device=device),
        )

        if (t + 1) % ep_len == 0:
            per_episode_entropy.append(_shannon_entropy(visit))
            visit = np.zeros_like(visit)

    return {
        "true": np.asarray(true_list).reshape(episodes, ep_len),
        "actions": np.asarray(act_list).reshape(episodes, ep_len),
        "per_episode_entropy": np.asarray(per_episode_entropy),
        "disagreements": np.asarray(disagreements).reshape(episodes, ep_len),
    }


def _longest_true_run(mask: np.ndarray) -> int:
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000, help="training age P3")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=0.075)
    parser.add_argument("--eval-seeds", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--precision-key", type=str, default="10.0x")
    parser.add_argument(
        "--out", type=str, default="runs/probe3_5_phase3/positive_control.json"
    )
    args = parser.parse_args()

    s1 = json.loads(Path("runs/probe3_5_phase2/s1_instantiation.json").read_text())
    precision = float(s1["precision_grid"][args.precision_key])
    null = json.loads(Path("runs/probe3_5_phase2/step0_baseline.json").read_text())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_dir = out_path.parent / (out_path.stem + "_telemetry")

    grid_cfg = replace(GridWorldConfig(), energy_obs_noise_sigma=args.sigma)
    preference = EnergyPreferenceConfig(precision=precision)
    print(
        f"[positive-control] precision={precision:.4f} ({args.precision_key}), "
        f"sigma={args.sigma}, lag={grid_cfg.energy_obs_lag}, P3={args.steps}"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tmp = tempfile.mkdtemp(prefix="probe3_5_p3_ckpt_")
    with _transport_pair(grid_cfg, seed=args.seed, run_id="probe3_5_p3_pc") as (
        client,
        env_server,
    ):
        run_cfg = RunnerConfig(
            world_model_config=WorldModelConfig(energy_dedicated_dims=0),
            run_id="probe3_5_p3_pc",
            telemetry_dir=telemetry_dir,
            checkpoints_dir=Path(tmp) / "checkpoints",  # throwaway (plan Phase 3)
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
        per_seed: list[dict[str, Any]] = []
        pooled_true: list[np.ndarray] = []
        pooled_entropy: list[float] = []
        pooled_epistemic: list[np.ndarray] = []
        pooled_actions: list[np.ndarray] = []
        ep_len = grid_cfg.episode_length
        o1_lo = ep_len // 2
        for s in range(args.eval_seeds):
            ev = _eval_seed(runner, grid_cfg, seed=9000 + s, episodes=args.episodes)
            true_arr = ev["true"]
            in_band = np.abs(true_arr - SETPOINT) <= BAND_HALFWIDTH
            o1 = float(in_band[:, o1_lo:].mean())
            flat = true_arr.reshape(-1)
            per_seed.append(
                {
                    "eval_seed": 9000 + s,
                    "o1_window_occupancy": o1,
                    "in_band_occupancy": float(in_band.mean()),
                    "true_mean": float(flat.mean()),
                    "true_std": float(flat.std()),
                    "floor_frac": float(np.mean(flat < FLOOR_THRESHOLD)),
                    "ceiling_frac": float(np.mean(flat > CEILING_THRESHOLD)),
                    "stay_share_eval": float(np.mean(ev["actions"] == STAY_ACTION)),
                    "pos_entropy_mean": float(ev["per_episode_entropy"].mean()),
                    "pos_entropy_sd": float(ev["per_episode_entropy"].std()),
                    "epistemic_mean": float(ev["disagreements"].mean()),
                }
            )
            pooled_true.append(true_arr)
            pooled_entropy.extend(ev["per_episode_entropy"].tolist())
            pooled_epistemic.append(ev["disagreements"].reshape(-1))
            pooled_actions.append(ev["actions"].reshape(-1))
            print(f"  seed {9000 + s}: {json.dumps(per_seed[-1])}")
    finally:
        runner.close()

    true_all = np.concatenate(pooled_true)  # (8*20, 200)
    in_band_all = np.abs(true_all - SETPOINT) <= BAND_HALFWIDTH
    o1_pooled = float(in_band_all[:, o1_lo:].mean())
    flat_all = true_all.reshape(-1)
    epistemic_all = np.concatenate(pooled_epistemic)
    actions_all = np.concatenate(pooled_actions)
    entropy_arr = np.asarray(pooled_entropy)

    # ---- telemetry-side: share trajectory, train stay-share, D monitor ----
    shard_paths = sorted((telemetry_dir / "agent_step").glob("*.parquet"))
    rows: list[dict[str, Any]] = []
    for shard in shard_paths:
        rows.extend(
            pq.read_table(  # type: ignore[no-untyped-call]
                shard, columns=["t", "action_t", "pragmatic_share_t", "kl_per_dim_t"]
            ).to_pylist()
        )
    rows.sort(key=lambda r: r["t"])
    share_seq = np.asarray(
        [r["pragmatic_share_t"] for r in rows if r["pragmatic_share_t"] is not None]
    )
    share_blocks = [
        float(share_seq[i : i + 1000].mean()) for i in range(0, len(share_seq), 1000)
    ]
    train_actions = np.asarray([r["action_t"] for r in rows])
    kl_final = np.asarray([r["kl_per_dim_t"] for r in rows[-1000:]])
    d_monitor_max_mean_kl = float(kl_final.mean(axis=0).max())

    # ---- null references (Step-0 record) ----
    null_entropy_mean = null["entropy_reference_A1b"][
        "positional_entropy_per_episode_nats"
    ]["mean"]
    null_entropy_sd = null["entropy_reference_A1b"][
        "positional_entropy_per_episode_nats"
    ]["sd"]
    null_epistemic_mean = null["entropy_reference_A1b"][
        "epistemic_activity_per_episode"
    ]["mean"]
    null_epistemic_sd = null["entropy_reference_A1b"][
        "epistemic_activity_per_episode"
    ]["sd"]

    # ---- §8.4 detector suite (first exercise on real data) ----
    pooled_occupancy = float(in_band_all.mean())
    entropy_ratio = float(entropy_arr.mean()) / null_entropy_mean
    epistemic_ratio = float(epistemic_all.mean()) / null_epistemic_mean
    # Camping (descriptive): episodes with a contiguous in-band run ≥ the A3c
    # window, and the eval stay-share inside in-band steps.
    camp_episodes = sum(
        1 for ep in in_band_all if _longest_true_run(ep) >= A3C_CAMP_WINDOW
    )
    in_band_stay_share = (
        float(np.mean(actions_all[in_band_all.reshape(-1)] == STAY_ACTION))
        if in_band_all.any()
        else None
    )
    signature_suite = {
        "framing": (
            "first exercise of the §8.4 falsification detectors on real "
            "data; firing at 10× is information about the grid's top, not "
            "the probe's verdict"
        ),
        "1_occupancy_saturation_when_sated": {
            "pinned_threshold_A1c": A1C_OCCUPANCY,
            "pooled_in_band_occupancy": pooled_occupancy,
            "fires": bool(pooled_occupancy >= A1C_OCCUPANCY),
        },
        "2_entropy_collapse": {
            "pinned_threshold_A1d_fraction": A1D_ENTROPY_FRACTION,
            "positional_ratio_vs_null": entropy_ratio,
            "positional_fires": bool(entropy_ratio < A1D_ENTROPY_FRACTION),
            "epistemic_ratio_vs_null": epistemic_ratio,
            "epistemic_fires": bool(epistemic_ratio < A1D_ENTROPY_FRACTION),
        },
        "3_pragmatic_share_to_1": {
            "note": (
                "no pinned numeric threshold in the pre-registration "
                "('share → 1'); reported descriptively"
            ),
            "share_trajectory_per_1000_train_steps": share_blocks,
            "share_final_1000_mean": share_blocks[-1] if share_blocks else None,
            "share_max_block": max(share_blocks) if share_blocks else None,
        },
        "4_camping_no_resumption": {
            "note": (
                "descriptive operationalization (the A3c criterion is "
                "defined on the recovery assay, which is Phase 4): episodes "
                "with a contiguous in-band run ≥ 100 steps, and stay-share "
                "while in-band"
            ),
            "episodes_with_in_band_run_ge_100": camp_episodes,
            "n_episodes": int(in_band_all.shape[0]),
            "stay_share_while_in_band": in_band_stay_share,
        },
    }

    # ---- the confirmed gate ----
    nonzero_seeds = sum(1 for p in per_seed if p["o1_window_occupancy"] > 0.0)
    gate = {
        "margin": (
            "[P3-G | pre, builder-confirmed 2026-06-11]: pooled O1-window "
            "occupancy ≥ 0.25 AND > 0 on ≥ 7/8 seeds. Provenance class: "
            "arbitrary-but-pre-committed (half the verdict bar), chosen for "
            "the interpretability function, not derived. Governs this "
            "session's branch only (proceed vs diagnostic-then-stop); lives "
            "outside the frozen §8.4 verdict machinery and is not a "
            "signature."
        ),
        "pooled_o1_window_occupancy": o1_pooled,
        "seeds_with_nonzero_o1": nonzero_seeds,
        "green": bool(o1_pooled >= 0.25 and nonzero_seeds >= 7),
    }

    out: dict[str, Any] = {
        "LABEL": (
            "Phase 3 positive control — instrument-and-pathway validation at "
            "the top of the pre-registered grid. NOT the probe's verdict "
            "(Phase 4, frozen raise-from-low order)."
        ),
        "config": {
            "precision": precision,
            "precision_grid_point": args.precision_key,
            "sigma": args.sigma,
            "lag": grid_cfg.energy_obs_lag,
            "training_age_P3": args.steps,
            "train_seed": args.seed,
            "eval": f"{args.eval_seeds} seeds x {args.episodes} episodes x {ep_len}",
            "grid_energy": {
                k: v for k, v in asdict(grid_cfg).items() if k.startswith("energy_")
            },
            "instance": "throwaway — checkpoint not carried forward (plan Phase 3)",
        },
        "displacement": {
            "o1_window_occupancy_pooled": o1_pooled,
            "o1_window_occupancy_per_seed": [
                p["o1_window_occupancy"] for p in per_seed
            ],
            "null_o1_reference": "0.00% exactly (Step-0 record)",
            "in_band_occupancy_pooled": pooled_occupancy,
            "true_energy_mean": float(flat_all.mean()),
            "true_energy_std": float(flat_all.std()),
            "floor_frac_lt_0.05": float(np.mean(flat_all < FLOOR_THRESHOLD)),
            "ceiling_frac_gt_0.95": float(np.mean(flat_all > CEILING_THRESHOLD)),
        },
        "entropy_vs_null_with_spread": {
            "positional_control": {
                "mean": float(entropy_arr.mean()),
                "sd": float(entropy_arr.std()),
            },
            "positional_null": {"mean": null_entropy_mean, "sd": null_entropy_sd},
            "epistemic_control": {
                "mean": float(epistemic_all.mean()),
                "sd": float(epistemic_all.std()),
            },
            "epistemic_null": {
                "mean": null_epistemic_mean,
                "sd": null_epistemic_sd,
            },
        },
        "stay_share": {
            "eval_greedy": float(np.mean(actions_all == STAY_ACTION)),
            "train_sampled": float(np.mean(train_actions == STAY_ACTION)),
            "train_final_1000": float(
                np.mean(train_actions[-1000:] == STAY_ACTION)
            ),
            "null_train_reference": 0.0392,
            "smoke_1x_train_reference": 0.0406,
        },
        "d_monitor": {
            "max_mean_per_dim_kl_final_1000_train_steps": d_monitor_max_mean_kl,
            "note": "monitor only (Amendment 01 — not gated)",
        },
        "per_seed": per_seed,
        "signature_suite_8_4": signature_suite,
        "gate": gate,
    }

    out_path.write_text(json.dumps(out, indent=2))
    np.savez_compressed(
        out_path.with_suffix(".npz"),
        true=true_all,
        actions=np.concatenate(pooled_actions).reshape(true_all.shape),
        entropy=entropy_arr,
    )
    print(json.dumps({"gate": gate, "displacement": out["displacement"]}, indent=2))
    print(f"[out] wrote {out_path}")


if __name__ == "__main__":
    main()
