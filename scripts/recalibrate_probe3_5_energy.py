"""Probe 3.5 — energy-physics recalibration (Amendment 01 §4).

Pre-committed target (CONFIRMED 2026-06-10): under the trained epistemic-only
actor at age P3, over R1=8 seeds × R2=20 episodes —
    R3  true_energy std   ≥ 0.10
    R4  true_energy mean   ∈ [0.30, 0.70]
    R5  fraction of steps at floor (< 0.05) ≤ 10%
Tunable surface: ONLY (energy_base_decay, energy_move_cost,
energy_replenish_per_resource). Everything else fixed (defines the assay
conditions).

Method (the passivity exploit, Amendment 01 §4 "Method"): the epistemic-only
actor reads no energy quantity, so its trajectory distribution is **invariant to
energy physics**. We train ONE epistemic instance, record the physics-invariant
per-step flags (is_move, consumed_resource) over many episodes, and **re-simulate
energy analytically** for each candidate triple. No retraining is needed to
search. The analytic re-simulation is validated to reproduce the env's own
true_energy series exactly at the default triple before any candidate is trusted.

This script does NOT retrain at the chosen physics or run the gate — those are
the next steps, on a fresh instance (Amendment 01 §4 / task Step 3-4). It writes
its trajectory record + chosen triple to ``--out`` for the gate/baseline steps.
"""

from __future__ import annotations

import argparse
import itertools
import json
import tempfile
import threading
from contextlib import contextmanager
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
)
from kind.training.runner import Runner, RunnerConfig

# Frozen / confirmed targets (Amendment 01 §4).
R3_STD_MIN = 0.10
R4_MEAN_LO, R4_MEAN_HI = 0.30, 0.70
R5_FLOOR_FRAC_MAX = 0.10
FLOOR_LEVEL = 0.05
R1_SEEDS = 8
R2_EPISODES = 20


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


def _consumed(grid: np.ndarray, pos: tuple[int, int], action: int) -> bool:
    """Did applying ``action`` from ``pos`` on ``grid`` enter a RESOURCE cell?

    Mirrors ``GridWorld._apply_action`` exactly: a move (action 0-3) into an
    in-bounds, non-wall RESOURCE cell consumes; stay (4), off-grid, and wall
    collisions do not. This is the replenishment trigger, computed observer-side
    from the pre-step world so the analytic re-simulation matches the env.
    """
    dr, dc = _ACTION_DELTAS[action]
    if dr == 0 and dc == 0:
        return False
    r, c = pos
    nr, nc = r + dr, c + dc
    gs = grid.shape[0]
    if not (0 <= nr < gs and 0 <= nc < gs):
        return False
    if grid[nr, nc] == CellType.WALL.value:
        return False
    return bool(grid[nr, nc] == CellType.RESOURCE.value)


@torch.no_grad()
def _eval_record(
    runner: Runner,
    grid_cfg: GridWorldConfig,
    seed: int,
    n_steps: int,
) -> dict[str, Any]:
    """Run the trained epistemic actor; record physics-invariant per-step flags.

    Records, per step (sampled exactly as ``run_probe3_5_phase1_baseline`` does —
    pre-step ``true_energy``): ``is_move``, ``consumed``, the env's own
    ``true_energy`` (for the analytic cross-check at default physics), the agent
    position (positional entropy), and the ensemble disagreement (epistemic
    magnitude). Also returns the (obs, action, sensed, true) arrays for the gate.
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

    is_move = np.zeros(n_steps, dtype=bool)
    consumed = np.zeros(n_steps, dtype=bool)
    env_true = np.zeros(n_steps, dtype=np.float64)
    disagreements = np.zeros(n_steps, dtype=np.float64)
    visit = np.zeros((grid_cfg.grid_size, grid_cfg.grid_size), dtype=np.int64)
    obs_list: list[np.ndarray] = []
    act_list: list[int] = []
    sensed_list: list[float] = []
    true_list: list[float] = []

    for t in range(n_steps):
        obs_np = step.observation.astype(np.float32) / 255.0
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).unsqueeze(0).to(device)
        sensed_t = torch.tensor(
            [[step.sensed_energy]], device=device, dtype=torch.float32
        )
        wm_step = wm.step(obs_t, h, z, a_prev, sensed_energy=sensed_t)
        intrinsic = runner.ensemble.disagreement(h, z, a_prev)
        disagreements[t] = float(intrinsic.reshape(-1)[0].item())
        view = PolicyView(
            h=wm_step.h, z=wm_step.z, self_prediction_error=zero_scalar
        )
        action = int(actor.act_greedy(view).reshape(-1)[0].item())

        gs = world.state
        visit[gs.agent_pos] += 1
        env_true[t] = float(gs.true_energy)
        is_move[t] = _ACTION_DELTAS[action] != (0, 0)
        consumed[t] = _consumed(gs.grid, gs.agent_pos, action)

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
        "is_move": is_move,
        "consumed": consumed,
        "env_true": env_true,
        "disagreements": disagreements,
        "visit": visit,
        "obs": torch.from_numpy(np.stack(obs_list)).unsqueeze(1),
        "action": torch.tensor(act_list, dtype=torch.long),
        "sensed": torch.tensor(sensed_list, dtype=torch.float32),
        "true": torch.tensor(true_list, dtype=torch.float32),
    }


def resim_energy(
    is_move: np.ndarray,
    consumed: np.ndarray,
    *,
    decay: float,
    move_cost: float,
    replenish: float,
    norm_max: float = 10.0,
    init: float = 6.0,
) -> np.ndarray:
    """Analytic re-simulation of normalized true_energy for one trajectory.

    Reproduces ``GridWorld._update_energy`` + ``_normalize_energy`` exactly: the
    per-step series is sampled **pre-step** (like the env), so element t is the
    energy *before* applying step t's action. Energy carries across the soft
    episode boundary (no reset mid-rollout — resolved decision #7).
    """
    n = is_move.shape[0]
    out = np.zeros(n, dtype=np.float64)
    e = init
    for t in range(n):
        out[t] = min(max(e, 0.0), norm_max) / norm_max
        delta = -decay
        if is_move[t]:
            delta -= move_cost
        if consumed[t]:
            delta += replenish
        e = min(max(e + delta, 0.0), norm_max)
    return out


def _metrics(series_list: list[np.ndarray]) -> dict[str, float]:
    """R3-R5 aggregates over a list of per-trajectory normalized-energy series.

    std/mean are pooled over all steps of all trajectories; floor-fraction is the
    pooled fraction of steps below ``FLOOR_LEVEL``.
    """
    allv = np.concatenate(series_list)
    return {
        "std": float(allv.std()),
        "mean": float(allv.mean()),
        "floor_frac": float(np.mean(allv < FLOOR_LEVEL)),
    }


def _meets_target(m: dict[str, float]) -> bool:
    return (
        m["std"] >= R3_STD_MIN
        and R4_MEAN_LO <= m["mean"] <= R4_MEAN_HI
        and m["floor_frac"] <= R5_FLOOR_FRAC_MAX
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000, help="training age P3")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record-seeds", type=int, default=R1_SEEDS)
    parser.add_argument("--record-episodes", type=int, default=R2_EPISODES)
    parser.add_argument("--out", type=str, default="runs/recal_probe3_5")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    grid_cfg = GridWorldConfig()  # default (floor-producing) physics for training
    wm_cfg = WorldModelConfig(energy_dedicated_dims=0)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ep_len = grid_cfg.episode_length

    with tempfile.TemporaryDirectory(prefix="probe3_5_recal_") as tmp:
        tmp_path = Path(tmp)
        with _transport_pair(grid_cfg, seed=args.seed, run_id="probe3_5_recal") as (
            client,
            env_server,
        ):
            run_cfg = RunnerConfig(
                world_model_config=wm_cfg,
                run_id="probe3_5_recal",
                telemetry_dir=tmp_path / "telemetry",
                checkpoints_dir=tmp_path / "checkpoints",
                warmup_env_steps=args.warmup,
                dream_cadence_env_steps=10_000_000,
                checkpoint_every_n_env_steps=10_000_000,
                energy_telemetry=True,
                device="cpu",
            )
            runner = Runner(run_cfg, client, env_server=env_server)
            try:
                runner.run(total_env_steps=args.steps)
                # Save the trained epistemic instance for reuse / B′ dev.
                torch.save(
                    runner.world_model.state_dict(), out_dir / "search_wm.pt"
                )
                records = []
                for s in range(args.record_seeds):
                    rec = _eval_record(
                        runner,
                        grid_cfg,
                        seed=args.seed + 5000 + s,
                        n_steps=ep_len * args.record_episodes,
                    )
                    records.append(rec)
            finally:
                runner.close()

    # ---- validate the analytic re-simulation against env ground truth -------
    default = dict(
        decay=grid_cfg.energy_base_decay,
        move_cost=grid_cfg.energy_move_cost,
        replenish=grid_cfg.energy_replenish_per_resource,
    )
    max_abs_err = 0.0
    for rec in records:
        resim = resim_energy(rec["is_move"], rec["consumed"], **default)
        max_abs_err = max(max_abs_err, float(np.max(np.abs(resim - rec["env_true"]))))
    print(f"[validate] analytic-vs-env max abs err at default physics: {max_abs_err:.2e}")
    assert max_abs_err < 1e-9, "analytic re-simulation does not match the env"

    # ---- report incidental foraging rate (the load-bearing quantity) --------
    total_steps = sum(r["is_move"].shape[0] for r in records)
    total_consumed = sum(int(r["consumed"].sum()) for r in records)
    total_moves = sum(int(r["is_move"].sum()) for r in records)
    print(
        f"[forage] consume rate = {total_consumed/total_steps:.4f}/step "
        f"({total_consumed} over {total_steps} steps); "
        f"move rate = {total_moves/total_steps:.3f}/step"
    )
    # Energy under default physics (reproduces the Phase-1 floor finding).
    default_m = _metrics(
        [resim_energy(r["is_move"], r["consumed"], **default) for r in records]
    )
    print(f"[default physics] {default_m}")

    # ---- candidate grid (only the three tunable magnitudes) -----------------
    decay_grid = [0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01]
    move_grid = [0.04, 0.02, 0.01, 0.005]
    repl_grid = [0.8, 1.5, 2.5, 4.0, 6.0]

    candidates: list[dict[str, Any]] = []
    for decay, move_cost, repl in itertools.product(decay_grid, move_grid, repl_grid):
        series = [
            resim_energy(
                r["is_move"], r["consumed"],
                decay=decay, move_cost=move_cost, replenish=repl,
            )
            for r in records
        ]
        m = _metrics(series)
        candidates.append(
            {
                "decay": decay,
                "move_cost": move_cost,
                "replenish": repl,
                **m,
                "meets": _meets_target(m),
            }
        )

    passing = [c for c in candidates if c["meets"]]
    print(f"[sweep] {len(passing)}/{len(candidates)} candidate triples meet R3-R5")

    chosen = None
    if passing:
        # Selection rule (pre-committed, deterministic): among triples meeting
        # R3-R5, prefer mean closest to the setpoint (0.6), then largest margin
        # above the std floor, then gentlest dynamics (smallest decay+move_cost).
        def _key(c: dict[str, Any]) -> tuple[float, float, float]:
            return (
                abs(c["mean"] - 0.6),
                -(c["std"] - R3_STD_MIN),
                c["decay"] + c["move_cost"],
            )

        chosen = sorted(passing, key=_key)[0]
        print(f"[chosen] {json.dumps({k: chosen[k] for k in ('decay','move_cost','replenish','std','mean','floor_frac')})}")
    else:
        print("[chosen] NONE — no triple in the candidate grid meets R3-R5 (stop-and-report)")

    # ---- persist for the gate / baseline steps ------------------------------
    np.savez(
        out_dir / "trajectories.npz",
        **{f"is_move_{i}": r["is_move"] for i, r in enumerate(records)},
        **{f"consumed_{i}": r["consumed"] for i, r in enumerate(records)},
        **{f"visit_{i}": r["visit"] for i, r in enumerate(records)},
        **{f"disagreements_{i}": r["disagreements"] for i, r in enumerate(records)},
        n_records=np.array([len(records)]),
        episode_length=np.array([ep_len]),
    )
    summary = {
        "training_age_P3": args.steps,
        "record_seeds": args.record_seeds,
        "record_episodes": args.record_episodes,
        "analytic_max_abs_err": max_abs_err,
        "consume_rate_per_step": total_consumed / total_steps,
        "default_physics": default,
        "default_metrics": default_m,
        "target": {
            "R3_std_min": R3_STD_MIN,
            "R4_mean": [R4_MEAN_LO, R4_MEAN_HI],
            "R5_floor_frac_max": R5_FLOOR_FRAC_MAX,
        },
        "n_passing": len(passing),
        "chosen": chosen,
        "candidates": candidates,
    }
    (out_dir / "recal_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[out] wrote {out_dir/'recal_summary.json'} and trajectories.npz")


if __name__ == "__main__":
    main()
