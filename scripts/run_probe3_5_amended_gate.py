"""Probe 3.5 — amended Phase-1 gate + baseline instantiation (task Step 4).

Reads the recalibration result (``recal_summary.json``: the chosen
``(decay, move_cost, replenish)`` triple), **retrains a fresh instance at the
chosen physics** so the world model learns the new energy dynamics, then runs the
**amended gate** (Amendment 01 §3): batteries A, C, B′ at the frozen/amended
margins, with D logged as a monitor only.

- Gate distribution: a **variance-rich uniform-random** eval rollout from the
  retrained model — the distribution on which Phase 1 established the channel is
  learnable/world-grounded (A, C). B′ intervention contexts (matched latents one
  step from a resource) are collected from the same adjacency-rich rollout.
- If the gate is green, ``--baseline-seeds`` fresh epistemic instances are trained
  at the chosen physics and evaluated over ``--baseline-episodes`` episodes each
  (the pure-epistemic baseline, ``pragmatic_value`` = 0), instantiating the frozen
  baseline-relative formulas (band, σ-grid, S1 precision) from the measured
  references — per the standing rule, *not* threshold editing.

Usage::

    python scripts/run_probe3_5_amended_gate.py --recal runs/recal_probe3_5 \
        [--steps 5000] [--baseline-seeds 8] [--baseline-episodes 20] [--gate-only]
"""

from __future__ import annotations

import argparse
import json
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from kind.agents.views import PolicyView
from kind.agents.world_model import WorldModelConfig
from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import (
    _ACTION_DELTAS,
    CellType,
    GridWorld,
    GridWorldConfig,
    NUM_ACTIONS,
)
from kind.observer.energy_eval import (
    BPrimeMargins,
    DeadPathMargins,
    run_amended_gate,
)
from kind.training.runner import Runner, RunnerConfig

SETPOINT = 0.6  # B0b (frozen)


@contextmanager
def _transport_pair(
    grid_cfg: GridWorldConfig, seed: int, run_id: str
) -> Iterator[tuple[Any, EnvServer]]:
    from kind.env.transport import EnvTransportClient, EnvTransportServer

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


def _train(grid_cfg: GridWorldConfig, steps: int, warmup: int, seed: int) -> Runner:
    torch.manual_seed(seed)
    np.random.seed(seed)
    wm_cfg = WorldModelConfig(energy_dedicated_dims=0)
    tmp = tempfile.mkdtemp(prefix="probe3_5_gate_")
    tmp_path = Path(tmp)
    with _transport_pair(grid_cfg, seed=seed, run_id="probe3_5_gate") as (
        client,
        env_server,
    ):
        run_cfg = RunnerConfig(
            world_model_config=wm_cfg,
            run_id="probe3_5_gate",
            telemetry_dir=tmp_path / "telemetry",
            checkpoints_dir=tmp_path / "checkpoints",
            warmup_env_steps=warmup,
            dream_cadence_env_steps=10_000_000,
            checkpoint_every_n_env_steps=10_000_000,
            energy_telemetry=True,
            device="cpu",
        )
        runner = Runner(run_cfg, client, env_server=env_server)
        runner.run(total_env_steps=steps)
    return runner


def _neighbors(grid: np.ndarray, pos: tuple[int, int]) -> dict[int, int]:
    """Map action → destination cell-type for the 4 moves (in-bounds, non-wall).

    Returns only passable destinations (off-grid / wall moves omitted)."""
    out: dict[int, int] = {}
    gs = grid.shape[0]
    r, c = pos
    for a in range(4):  # moves only (0-3); 4 is stay
        dr, dc = _ACTION_DELTAS[a]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < gs and 0 <= nc < gs):
            continue
        if grid[nr, nc] == CellType.WALL.value:
            continue
        out[a] = int(grid[nr, nc])
    return out


@torch.no_grad()
def _gate_rollout(
    runner: Runner,
    grid_cfg: GridWorldConfig,
    seed: int,
    n_steps: int,
    max_contexts: int = 400,
) -> dict[str, Any]:
    """Trained-epistemic-actor rollout; teacher-force the WM; record the eval
    arrays for A/C/D **and** B′ intervention contexts.

    Post-recalibration the **epistemic actor** is the non-degenerate distribution
    (energy spans the range; std ≈ 0.28), so A/C have target variance and B′
    contexts have decode headroom — unlike a uniform-random policy, which the
    recalibrated gentle physics saturates at the ceiling (logged separately as a
    diagnostic). This mirrors the Phase-1 choice (run decodability where the
    target varies); recalibration inverted *which* policy that is.

    A B′ context is a step where the agent is one move from ≥1 RESOURCE cell and
    ≥1 non-resource passable cell — giving a coincident action (onto the resource)
    and a matched control action (onto empty), from the same real latent (h, z).
    """
    device = runner.device
    wm = runner.world_model
    actor = runner.actor
    world = GridWorld(grid_cfg, seed=seed)
    step = world.reset()

    h = torch.zeros(1, wm.config.h_dim, device=device)
    z = torch.zeros(1, wm.config.z_dim, device=device)
    a_prev = torch.zeros(1, dtype=torch.long, device=device)
    zero_scalar = torch.zeros((), device=device)

    obs_list: list[np.ndarray] = []
    act_list: list[int] = []
    sensed_list: list[float] = []
    true_list: list[float] = []
    ctx_h: list[np.ndarray] = []
    ctx_z: list[np.ndarray] = []
    ctx_coin: list[int] = []
    ctx_ctrl: list[int] = []

    for _ in range(n_steps):
        obs_np = step.observation.astype(np.float32) / 255.0
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).unsqueeze(0).to(device)
        sensed_t = torch.tensor(
            [[step.sensed_energy]], device=device, dtype=torch.float32
        )
        wm_step = wm.step(obs_t, h, z, a_prev, sensed_energy=sensed_t)
        view = PolicyView(
            h=wm_step.h, z=wm_step.z, self_prediction_error=zero_scalar
        )
        action = int(actor.act_greedy(view).reshape(-1)[0].item())

        gs = world.state
        # B′ context detection on the pre-step world.
        if len(ctx_h) < max_contexts:
            nbrs = _neighbors(gs.grid, gs.agent_pos)
            res = [a for a, t in nbrs.items() if t == CellType.RESOURCE.value]
            non = [a for a, t in nbrs.items() if t != CellType.RESOURCE.value]
            if res and non:
                ctx_h.append(wm_step.h.squeeze(0).cpu().numpy())
                ctx_z.append(wm_step.z.squeeze(0).cpu().numpy())
                ctx_coin.append(int(res[0]))
                ctx_ctrl.append(int(non[0]))

        obs_list.append(obs_np)
        act_list.append(action)
        sensed_list.append(float(step.sensed_energy))
        true_list.append(float(gs.true_energy))

        step = world.step(action)
        h, z, a_prev = (
            wm_step.h,
            wm_step.z,
            torch.tensor([action], dtype=torch.long, device=device),
        )

    return {
        "obs": torch.from_numpy(np.stack(obs_list)).unsqueeze(1),
        "action": torch.tensor(act_list, dtype=torch.long),
        "sensed": torch.tensor(sensed_list, dtype=torch.float32),
        "true": torch.tensor(true_list, dtype=torch.float32),
        "ctx_h": torch.tensor(np.stack(ctx_h), dtype=torch.float32)
        if ctx_h
        else torch.zeros(0, wm.config.h_dim),
        "ctx_z": torch.tensor(np.stack(ctx_z), dtype=torch.float32)
        if ctx_z
        else torch.zeros(0, wm.config.z_dim),
        "ctx_coin": torch.tensor(ctx_coin, dtype=torch.long),
        "ctx_ctrl": torch.tensor(ctx_ctrl, dtype=torch.long),
    }


def _gate_dict(report: Any) -> dict[str, Any]:
    return {
        "A": {"passed": report.a.passed, "detail": report.a.detail},
        "C": {"passed": report.c.passed, "detail": report.c.detail},
        "B_prime": {
            "passed": report.b_prime.passed,
            "detail": report.b_prime.detail,
            "metrics": report.b_prime.metrics,
        },
        "D_monitor": {
            "passed": report.d_monitor.passed,
            "detail": report.d_monitor.detail,
        },
        "gate_passed": report.gate_passed,
    }


def _shannon_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


@torch.no_grad()
def _epistemic_baseline_rollout(
    runner: Runner,
    grid_cfg: GridWorldConfig,
    seed: int,
    n_steps: int,
) -> dict[str, Any]:
    """Pure-epistemic baseline rollout (trained actor, ``pragmatic_value`` = 0).

    Records the frozen §3 references: true_energy mean/std, in-band occupancy,
    positional entropy, epistemic-term magnitude. Energy is the env's own
    true_energy under the chosen physics (the instance was trained at it).
    """
    device = runner.device
    wm = runner.world_model
    actor = runner.actor
    world = GridWorld(grid_cfg, seed=seed)
    step = world.reset()

    h = torch.zeros(1, wm.config.h_dim, device=device)
    z = torch.zeros(1, wm.config.z_dim, device=device)
    a_prev = torch.zeros(1, dtype=torch.long, device=device)
    zero_scalar = torch.zeros((), device=device)

    true_list: list[float] = []
    disagreements: list[float] = []
    visit = np.zeros((grid_cfg.grid_size, grid_cfg.grid_size), dtype=np.int64)

    for _ in range(n_steps):
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

    return {
        "true": np.asarray(true_list),
        "disagreements": np.asarray(disagreements),
        "visit": visit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recal", type=str, default="runs/recal_probe3_5")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--gate-seed", type=int, default=100)
    parser.add_argument("--gate-eval-steps", type=int, default=8000)
    parser.add_argument("--baseline-seeds", type=int, default=8)
    parser.add_argument("--baseline-episodes", type=int, default=20)
    parser.add_argument("--diag-seeds", type=int, default=4)
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--out", type=str, default="runs/recal_probe3_5/gate_baseline.json")
    args = parser.parse_args()

    recal = json.loads((Path(args.recal) / "recal_summary.json").read_text())
    chosen = recal["chosen"]
    if chosen is None:
        raise SystemExit("recalibration found no passing triple — stop-and-report")
    decay = float(chosen["decay"])
    move_cost = float(chosen["move_cost"])
    repl = float(chosen["replenish"])
    base_cfg = GridWorldConfig()
    grid_cfg = replace(
        base_cfg,
        energy_base_decay=decay,
        energy_move_cost=move_cost,
        energy_replenish_per_resource=repl,
    )
    replenish_norm = repl / grid_cfg.energy_norm_max
    print(
        f"[physics] chosen triple decay={decay} move_cost={move_cost} "
        f"replenish={repl} (replenish_norm={replenish_norm:.3f})"
    )

    # ---- retrain one fresh instance at the chosen physics; run the gate ------
    # The single retrained instance is kept alive through the baseline: per the
    # builder's Step-3 passivity framing (the epistemic actor reads no energy
    # quantity, so its trajectory distribution is ~invariant to energy physics)
    # and Step-4 "on the retrained instance", the baseline is measured over
    # P1=8 *eval* seeds × P2=20 episodes on this instance — not 8 retrainings.
    runner = _train(grid_cfg, args.steps, args.warmup, seed=args.gate_seed)
    out: dict[str, Any] = {
        "chosen_physics": {"decay": decay, "move_cost": move_cost, "replenish": repl},
        "replenish_norm": replenish_norm,
    }
    try:
        # Diagnostic: uniform-random saturates energy at the ceiling under the
        # recalibrated gentle physics (the inverse of the Phase-1 floor) — record
        # it so the gate's distribution choice is auditable, not silent.
        rng = np.random.default_rng(args.gate_seed + 3000)
        dw = GridWorld(grid_cfg, seed=args.gate_seed + 3000)
        dw.reset()
        ur = []
        for _ in range(4000):
            ur.append(dw.state.true_energy)
            dw.step(int(rng.integers(0, NUM_ACTIONS)))
        ur_arr = np.asarray(ur)
        out["uniform_random_diagnostic"] = {
            "mean": float(ur_arr.mean()),
            "std": float(ur_arr.std()),
            "ceil_frac_gt_0.95": float(np.mean(ur_arr > 0.95)),
            "note": "uniform-random saturates at the ceiling; A/C/B′ run on the "
            "epistemic distribution instead (non-degenerate, std≈0.28)",
        }
        print("[diag-uniform-random]", json.dumps(out["uniform_random_diagnostic"]))

        roll = _gate_rollout(
            runner, grid_cfg, seed=args.gate_seed + 7000, n_steps=args.gate_eval_steps
        )
        print(f"[gate] collected {int(roll['ctx_coin'].shape[0])} B′ contexts")
        report = run_amended_gate(
            runner.world_model,
            roll["obs"],
            roll["action"],
            roll["sensed"],
            roll["true"],
            b_prime_h=roll["ctx_h"],
            b_prime_z=roll["ctx_z"],
            b_prime_action_coincident=roll["ctx_coin"],
            b_prime_action_control=roll["ctx_ctrl"],
            replenish_norm=replenish_norm,
            margins=DeadPathMargins(),
            b_prime_margins=BPrimeMargins(),
        )

        gate = _gate_dict(report)
        print("[gate]", json.dumps(gate, indent=2))
        out["gate"] = gate

        # Always-on DIAGNOSTIC (runs regardless of verdict): does the
        # retrained-at-chosen-physics instance's epistemic actor actually
        # produce the R3-R5 distribution the *default-trained* analytic search
        # predicted? Distinguishes "channel not grounded in this regime" from
        # "recalibration did not transfer to the trained instance" (the
        # physics-invariance premise breaking under live-energy training).
        ep_len = grid_cfg.episode_length
        diag_true: list[np.ndarray] = []
        for s in range(args.diag_seeds):
            ev = _epistemic_baseline_rollout(
                runner, grid_cfg, seed=args.gate_seed + 13000 + s,
                n_steps=ep_len * 5,
            )
            diag_true.append(ev["true"])
        dt = np.concatenate(diag_true)
        out["instance_energy_diagnostic"] = {
            "diag_seeds": args.diag_seeds,
            "mean": float(dt.mean()),
            "std": float(dt.std()),
            "floor_frac_lt_0.05": float(np.mean(dt < 0.05)),
            "ceil_frac_gt_0.95": float(np.mean(dt > 0.95)),
            "R3_std_ge_0.10": bool(dt.std() >= 0.10),
            "R4_mean_in_0.30_0.70": bool(0.30 <= dt.mean() <= 0.70),
            "R5_floor_frac_le_0.10": bool(np.mean(dt < 0.05) <= 0.10),
            "note": "measured on the retrained instance's own epistemic actor "
            "(vs the analytic search's prediction std≈0.284 from the "
            "default-trained actor's trajectories)",
        }
        print("[diag-instance-energy]", json.dumps(out["instance_energy_diagnostic"]))

        if args.gate_only or not report.gate_passed:
            if not report.gate_passed:
                print("[disposition] gate NOT green — stopping before baseline.")
            Path(args.out).write_text(json.dumps(out, indent=2))
            print(f"[out] wrote {args.out}")
            return

        # ---- baseline: P1 eval seeds × P2 episodes on the retrained instance --
        print(
            f"[baseline] {args.baseline_seeds} eval seeds × "
            f"{args.baseline_episodes} episodes on the retrained instance"
        )
        per_seed: list[dict[str, float]] = []
        pooled_true: list[np.ndarray] = []
        for s in range(args.baseline_seeds):
            ev = _epistemic_baseline_rollout(
                runner, grid_cfg, seed=args.gate_seed + 9000 + s,
                n_steps=ep_len * args.baseline_episodes,
            )
            true_arr = ev["true"]
            std = float(true_arr.std())
            in_band = (
                float(np.mean(np.abs(true_arr - SETPOINT) <= std)) if std > 0 else 0.0
            )
            per_seed.append(
                {
                    "true_mean": float(true_arr.mean()),
                    "true_std": std,
                    "floor_frac": float(np.mean(true_arr < 0.05)),
                    "in_band_occ": in_band,
                    "pos_entropy": _shannon_entropy(ev["visit"]),
                    "epistemic_mean": float(ev["disagreements"].mean()),
                    "epistemic_p90": float(np.percentile(ev["disagreements"], 90)),
                }
            )
            pooled_true.append(true_arr)
            print(f"  seed {s}: {per_seed[-1]}")
    finally:
        runner.close()

    def _ms(key: str) -> dict[str, float]:
        vals = np.asarray([p[key] for p in per_seed])
        return {"mean": float(vals.mean()), "sd": float(vals.std())}

    pooled = np.concatenate(pooled_true)
    baseline_std = float(np.mean([p["true_std"] for p in per_seed]))
    epi_mean = float(np.mean([p["epistemic_mean"] for p in per_seed]))
    out["baseline"] = {
        "n_eval_seeds": args.baseline_seeds,
        "episodes_per_seed": args.baseline_episodes,
        "interpretation": (
            "8 eval seeds on the single retrained instance (physics-invariant "
            "trajectory; frozen PolicyView) — not 8 independent trainings"
        ),
        "true_energy_mean": _ms("true_mean"),
        "true_energy_std": _ms("true_std"),
        "floor_frac": _ms("floor_frac"),
        "in_band_occupancy": _ms("in_band_occ"),
        "positional_entropy_nats": _ms("pos_entropy"),
        "epistemic_magnitude_mean": _ms("epistemic_mean"),
        "epistemic_magnitude_p90": _ms("epistemic_p90"),
        "target_check": {
            "R3_std_ge_0.10": baseline_std >= 0.10,
            "R4_mean_in_0.30_0.70": 0.30 <= float(pooled.mean()) <= 0.70,
            "R5_floor_frac_le_0.10": float(np.mean(pooled < 0.05)) <= 0.10,
        },
        "instantiated_formulas": {
            "setpoint_B0b": SETPOINT,
            "band_B0a": {
                "half_width": baseline_std,
                "low": SETPOINT - baseline_std,
                "high": SETPOINT + baseline_std,
            },
            "sigma_grid_S2": {
                "0x_diagnostic": 0.0,
                "0.5x": 0.5 * baseline_std,
                "1.0x": 1.0 * baseline_std,
            },
            "precision_grid_S1_at_epistemic_mag": {
                "epistemic_magnitude": epi_mean,
                "values": [round(f * epi_mean, 5) for f in (0.1, 0.32, 1.0, 3.2, 10.0)],
            },
        },
    }
    print("[baseline]", json.dumps(out["baseline"], indent=2))
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[out] wrote {args.out}")
    return


if __name__ == "__main__":
    main()
