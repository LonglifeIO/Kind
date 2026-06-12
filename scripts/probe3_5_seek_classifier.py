"""Probe 3.5 post-close — seek-mechanism classifier (eval and analysis only).

Implements diagnostics D1–D4 of the pre-stated decision rule in
`docs/decisions/probe3_5_seek_classification_2026-06-12.md` §1 (written before
this script ran). Bin 1: instrument defect (the belief lies). Bin 2:
architectural absence (belief honest; credit-assignment path missing).

All model-dependent work runs on the Step-0 instance — the only persisted
weight set. No training, no env changes, no physics changes; the env is used
exactly as the existing eval harnesses use it.

Usage::

    python scripts/probe3_5_seek_classifier.py \
        [--out runs/probe3_5_seek_classification/results.json]
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray

from kind.agents.actor import Actor
from kind.agents.views import PolicyView
from kind.agents.world_model import WorldModel, WorldModelConfig
from kind.env.grid_world import _ACTION_DELTAS, CellType, GridWorld, GridWorldConfig
from kind.observer.oracle_forager import oracle_action

SETPOINT = 0.6
BAND_HALFWIDTH = 0.15
STAY = 4
MOVE_ACTIONS = (0, 1, 2, 3)
JUMP_THRESHOLD = 0.03  # realized consumption jump ≈ +0.068; ordinary ≈ −0.012
HORIZON = 15  # the actor's imagination horizon (RunnerConfig default)
ANALYSIS_TORCH_SEED = 1234


# ---------------------------------------------------------------------------
# BFS helpers (env-instrument side, mirroring oracle_forager's expansion order)
# ---------------------------------------------------------------------------


def bfs_path_to_nearest_resource(
    grid: NDArray[np.uint8], agent_pos: tuple[int, int]
) -> list[int] | None:
    """Full action sequence of a BFS shortest path onto the nearest RESOURCE
    cell (fixed up/down/left/right expansion order, as the oracle). None if no
    resource is reachable."""
    n_rows, n_cols = grid.shape
    visited = np.zeros((n_rows, n_cols), dtype=bool)
    visited[agent_pos] = True
    queue: deque[tuple[tuple[int, int], list[int]]] = deque()
    queue.append((agent_pos, []))
    while queue:
        (r, c), path = queue.popleft()
        for a in MOVE_ACTIONS:
            dr, dc = _ACTION_DELTAS[a]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                continue
            if visited[nr, nc]:
                continue
            if grid[nr, nc] == CellType.WALL.value:
                continue
            if grid[nr, nc] == CellType.RESOURCE.value:
                return path + [a]
            visited[nr, nc] = True
            queue.append(((nr, nc), path + [a]))
    return None


def avoidance_control_path(
    grid: NDArray[np.uint8], agent_pos: tuple[int, int], length: int
) -> list[int]:
    """A same-length move sequence that never enters a RESOURCE cell: at each
    simulated position take the first move (fixed order) onto an in-bounds,
    non-wall, non-resource cell; stay if none exists (staying never consumes).
    Matched movement cost isolates the replenishment contrast in D3."""
    n_rows, n_cols = grid.shape
    pos = agent_pos
    path: list[int] = []
    for _ in range(length):
        chosen = STAY
        for a in MOVE_ACTIONS:
            dr, dc = _ACTION_DELTAS[a]
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                continue
            if grid[nr, nc] in (CellType.WALL.value, CellType.RESOURCE.value):
                continue
            chosen = a
            pos = (nr, nc)
            break
        path.append(chosen)
    return path


# ---------------------------------------------------------------------------
# Unified trajectory collector: env rollout + teacher-forced world model
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    """Per-step records for one seed's rollout (T = episodes × 200)."""

    true: NDArray[np.float64]  # (T,) pre-step true energy
    sensed: NDArray[np.float64]  # (T,) pre-step sensed energy
    action: NDArray[np.int64]  # (T,) action taken at t
    decode: NDArray[np.float64]  # (T,) energy_pred at t (post obs_t update)
    h: NDArray[np.float32]  # (T, h_dim) post obs_t update
    z: NDArray[np.float32]  # (T, z_dim)
    bfs_dist: NDArray[np.int64]  # (T,) BFS distance to nearest resource (-1 none)
    grids: list[bytes]  # (T,) grid snapshots (uint8 bytes, 8x8)
    positions: list[tuple[int, int]]  # (T,) agent positions
    consumed: NDArray[np.bool_]  # (T,) consumption occurred during step t


@torch.no_grad()
def collect(
    wm: WorldModel,
    actor: Actor,
    grid_cfg: GridWorldConfig,
    *,
    policy: Literal["oracle", "greedy", "sample"],
    seed: int,
    episodes: int,
) -> Trajectory:
    device = torch.device("cpu")
    ep_len = grid_cfg.episode_length
    n_steps = ep_len * episodes
    world = GridWorld(grid_cfg, seed=seed)
    step = world.reset()

    h = torch.zeros(1, wm.config.h_dim, device=device)
    z = torch.zeros(1, wm.config.z_dim, device=device)
    a_prev = torch.zeros(1, dtype=torch.long, device=device)
    zero_scalar = torch.zeros((), device=device)

    true_l: list[float] = []
    sensed_l: list[float] = []
    act_l: list[int] = []
    dec_l: list[float] = []
    h_l: list[NDArray[np.float32]] = []
    z_l: list[NDArray[np.float32]] = []
    dist_l: list[int] = []
    grids: list[bytes] = []
    positions: list[tuple[int, int]] = []

    for _ in range(n_steps):
        obs_np = step.observation.astype(np.float32) / 255.0
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).unsqueeze(0)
        sensed_t = torch.tensor([[step.sensed_energy]], dtype=torch.float32)
        wm_step = wm.step(obs_t, h, z, a_prev, sensed_energy=sensed_t)

        gs = world.state
        path = bfs_path_to_nearest_resource(gs.grid, gs.agent_pos)
        if policy == "oracle":
            action = oracle_action(gs.grid, gs.agent_pos, gs.true_energy)
        else:
            view = PolicyView(
                h=wm_step.h, z=wm_step.z, self_prediction_error=zero_scalar
            )
            if policy == "greedy":
                action = int(actor.act_greedy(view).reshape(-1)[0].item())
            else:
                action = int(actor.forward(view).action.reshape(-1)[0].item())

        true_l.append(float(gs.true_energy))
        sensed_l.append(float(step.sensed_energy))
        act_l.append(action)
        dec_l.append(float(wm_step.energy_pred.reshape(-1)[0].item()))
        h_l.append(wm_step.h.squeeze(0).numpy().astype(np.float32))
        z_l.append(wm_step.z.squeeze(0).numpy().astype(np.float32))
        dist_l.append(len(path) if path is not None else -1)
        grids.append(gs.grid.astype(np.uint8).tobytes())
        positions.append((int(gs.agent_pos[0]), int(gs.agent_pos[1])))

        step = world.step(action)
        h, z = wm_step.h, wm_step.z
        a_prev = torch.tensor([action], dtype=torch.long)

    true = np.asarray(true_l)
    consumed = np.zeros(n_steps, dtype=bool)
    consumed[:-1] = np.diff(true) > JUMP_THRESHOLD
    return Trajectory(
        true=true,
        sensed=np.asarray(sensed_l),
        action=np.asarray(act_l, dtype=np.int64),
        decode=np.asarray(dec_l),
        h=np.stack(h_l),
        z=np.stack(z_l),
        bfs_dist=np.asarray(dist_l, dtype=np.int64),
        grids=grids,
        positions=positions,
        consumed=consumed,
    )


# ---------------------------------------------------------------------------
# D1 — decoder honesty
# ---------------------------------------------------------------------------

BANDS: tuple[tuple[str, float, float], ...] = (
    ("floor[0,0.05)", 0.0, 0.05),
    ("low[0.05,0.45)", 0.05, 0.45),
    ("band[0.45,0.75]", 0.45, 0.7500001),
    ("high(0.75,0.95]", 0.7500001, 0.95),
    ("ceiling(0.95,1]", 0.95, 1.0000001),
)


def d1_stats(trajs: list[Trajectory]) -> dict[str, Any]:
    true = np.concatenate([t.true for t in trajs])
    sensed = np.concatenate([t.sensed for t in trajs])
    decode = np.concatenate([t.decode for t in trajs])
    per_band: dict[str, Any] = {}
    for name, lo, hi in BANDS:
        m = (true >= lo) & (true < hi)
        if m.sum() == 0:
            per_band[name] = {"n": 0}
            continue
        per_band[name] = {
            "n": int(m.sum()),
            "true_mean": float(true[m].mean()),
            "decode_mean": float(decode[m].mean()),
            "signed_error_mean": float((decode[m] - true[m]).mean()),
            "abs_error_mean": float(np.abs(decode[m] - true[m]).mean()),
        }
    # OLS slope decode ~ true over the occupied range.
    var = float(np.var(true))
    slope = float(np.cov(true, decode)[0, 1] / var) if var > 1e-12 else None
    return {
        "n_steps": int(true.shape[0]),
        "per_band": per_band,
        "slope_decode_vs_true": slope,
        "decode_vs_sensed_abs_error_mean": float(np.abs(decode - sensed).mean()),
        "decode_vs_true_abs_error_mean": float(np.abs(decode - true).mean()),
        "true_range": [float(true.min()), float(true.max())],
        "decode_range": [float(decode.min()), float(decode.max())],
    }


# ---------------------------------------------------------------------------
# D2 — consumption-transition modeling (one imagined step, B′ pattern)
# ---------------------------------------------------------------------------


Tensor_ = torch.Tensor


@torch.no_grad()
def one_step_imagined_decode(
    wm: WorldModel, h: Tensor_, z: Tensor_, action: int
) -> float:
    a = torch.tensor([action], dtype=torch.long)
    h_next = wm.recurrence(h, z, a)
    mu_p, _ = wm.prior(h_next)
    return float(wm.decode_energy(h_next, mu_p).reshape(-1)[0].item())


@torch.no_grad()
def d2_stats(
    wm: WorldModel, trajs: list[Trajectory], rng: np.random.Generator
) -> dict[str, Any]:
    cons_pred: list[float] = []
    cons_real: list[float] = []
    ord_pred: list[float] = []
    ord_real: list[float] = []
    for tr in trajs:
        t_max = len(tr.true) - 1
        cons_idx = np.where(tr.consumed[:t_max])[0]
        n = len(cons_idx)
        if n == 0:
            continue
        ord_pool = np.where(~tr.consumed[:t_max])[0]
        ord_idx = rng.choice(ord_pool, size=min(n, len(ord_pool)), replace=False)
        for idx_set, pred_out, real_out in (
            (cons_idx, cons_pred, cons_real),
            (ord_idx, ord_pred, ord_real),
        ):
            for t in idx_set:
                h = torch.from_numpy(tr.h[t]).unsqueeze(0)
                z = torch.from_numpy(tr.z[t]).unsqueeze(0)
                e_next = one_step_imagined_decode(wm, h, z, int(tr.action[t]))
                pred_out.append(e_next - tr.decode[t])
                real_out.append(float(tr.true[t + 1] - tr.true[t]))
    return {
        "n_consumption_steps": len(cons_pred),
        "predicted_jump_mean": float(np.mean(cons_pred)) if cons_pred else None,
        "realized_jump_mean": float(np.mean(cons_real)) if cons_real else None,
        "n_ordinary_steps": len(ord_pred),
        "ordinary_predicted_mean": float(np.mean(ord_pred)) if ord_pred else None,
        "ordinary_realized_mean": float(np.mean(ord_real)) if ord_real else None,
        "predicted_over_realized": (
            float(np.mean(cons_pred) / np.mean(cons_real))
            if cons_pred and np.mean(cons_real) > 0
            else None
        ),
    }


# ---------------------------------------------------------------------------
# D3 — imagination representability on scripted resource-reaching paths
# ---------------------------------------------------------------------------


@torch.no_grad()
def imagine_scripted(
    wm: WorldModel,
    h0: Tensor_,
    z0: Tensor_,
    actions: list[int],
    *,
    n_samples: int,
    use_prior_mean: bool,
    gen: torch.Generator,
) -> NDArray[np.float64]:
    """Roll the world model on a scripted action sequence from (h0, z0);
    return decoded energy per imagined step, averaged over the batch of
    samples. Mirrors the actor's imagination stepping (gru on (z, action_emb);
    z' from the prior) with scripted actions instead of policy samples."""
    batch = 1 if use_prior_mean else n_samples
    h = h0.expand(batch, -1).contiguous()
    z = z0.expand(batch, -1).contiguous()
    out: list[float] = []
    for a in actions:
        a_t = torch.full((batch,), a, dtype=torch.long)
        emb = wm.action_embedding(a_t)
        h = wm.gru_cell(torch.cat([z, emb], dim=-1), h)
        mu_p, log_sigma_p = wm.prior(h)
        if use_prior_mean:
            z = mu_p
        else:
            noise = torch.randn(mu_p.shape, generator=gen)
            z = mu_p + torch.exp(log_sigma_p) * noise
        out.append(float(wm.decode_energy(h, z).reshape(-1).mean().item()))
    return np.asarray(out)


@torch.no_grad()
def d3_stats(
    wm: WorldModel,
    trajs: list[Trajectory],
    *,
    max_contexts: int,
    delta_threshold: float,
    gen: torch.Generator,
) -> dict[str, Any]:
    grid_side = 8
    contexts: list[tuple[Trajectory, int, list[int]]] = []
    for tr in trajs:
        for t in range(len(tr.true)):
            d = int(tr.bfs_dist[t])
            if not (1 <= d <= 8):
                continue
            grid = np.frombuffer(tr.grids[t], dtype=np.uint8).reshape(
                grid_side, grid_side
            )
            path = bfs_path_to_nearest_resource(grid, tr.positions[t])
            if path is None:
                continue
            contexts.append((tr, t, path))
    if len(contexts) > max_contexts:
        stride = len(contexts) / max_contexts
        contexts = [contexts[int(i * stride)] for i in range(max_contexts)]

    rows: list[dict[str, float]] = []
    for tr, t, path in contexts:
        grid = np.frombuffer(tr.grids[t], dtype=np.uint8).reshape(
            grid_side, grid_side
        )
        d = len(path)
        scripted = path + [STAY, STAY]  # 2 post-coincidence steps
        control = avoidance_control_path(grid, tr.positions[t], len(scripted))
        h0 = torch.from_numpy(tr.h[t]).unsqueeze(0)
        z0 = torch.from_numpy(tr.z[t]).unsqueeze(0)
        dec_path = imagine_scripted(
            wm, h0, z0, scripted, n_samples=16, use_prior_mean=False, gen=gen
        )
        dec_ctrl = imagine_scripted(
            wm, h0, z0, control, n_samples=16, use_prior_mean=False, gen=gen
        )
        dec_path_mu = imagine_scripted(
            wm, h0, z0, scripted, n_samples=1, use_prior_mean=True, gen=gen
        )
        dec_ctrl_mu = imagine_scripted(
            wm, h0, z0, control, n_samples=1, use_prior_mean=True, gen=gen
        )
        post = slice(d - 1, len(scripted))  # decode indices after entering action
        rows.append(
            {
                "bfs_dist": float(d),
                "delta_rsample": float(dec_path[post].mean() - dec_ctrl[post].mean()),
                "delta_prior_mean": float(
                    dec_path_mu[post].mean() - dec_ctrl_mu[post].mean()
                ),
                "true_energy_at_context": float(tr.true[t]),
            }
        )
    if not rows:
        return {"n_contexts": 0}
    deltas = np.asarray([r["delta_rsample"] for r in rows])
    deltas_mu = np.asarray([r["delta_prior_mean"] for r in rows])
    return {
        "n_contexts": len(rows),
        "delta_rsample_mean": float(deltas.mean()),
        "delta_rsample_frac_ge_threshold": float(
            np.mean(deltas >= delta_threshold)
        ),
        "delta_prior_mean_mean": float(deltas_mu.mean()),
        "delta_prior_mean_frac_ge_threshold": float(
            np.mean(deltas_mu >= delta_threshold)
        ),
        "threshold": delta_threshold,
        "bfs_dist_mean": float(np.mean([r["bfs_dist"] for r in rows])),
    }


# ---------------------------------------------------------------------------
# D4 — horizon arithmetic
# ---------------------------------------------------------------------------


def d4_stats(trajs: list[Trajectory], horizon: int) -> dict[str, Any]:
    dists = np.concatenate([t.bfs_dist for t in trajs])
    dists = dists[dists >= 0]
    consumed = [t.consumed for t in trajs]
    within: list[bool] = []
    for c in consumed:
        t_steps = len(c)
        for t in range(t_steps - horizon):
            within.append(bool(c[t + 1 : t + 1 + horizon].any()))
    hist = {str(d): int((dists == d).sum()) for d in range(int(dists.max()) + 1)}
    return {
        "bfs_distance_mean": float(dists.mean()),
        "bfs_distance_p90": float(np.percentile(dists, 90)),
        "bfs_distance_max": int(dists.max()),
        "frac_within_horizon_distance": float(np.mean(dists <= horizon)),
        "empirical_P_consume_within_horizon": float(np.mean(within)),
        "distance_histogram": hist,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=str, default="runs/probe3_5_seek_classification/results.json"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/probe3_5_phase2/step0_burnin_checkpoint.pt",
    )
    args = parser.parse_args()

    torch.manual_seed(ANALYSIS_TORCH_SEED)
    rng = np.random.default_rng(ANALYSIS_TORCH_SEED)
    gen = torch.Generator()
    gen.manual_seed(ANALYSIS_TORCH_SEED)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    wm = WorldModel(WorldModelConfig(energy_dedicated_dims=0))
    wm.load_state_dict(ckpt["world_model"])
    wm.eval()
    actor = Actor(h_dim=200, z_dim=16, action_dim=5, mlp_hidden=200)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    grid_cfg = GridWorldConfig()  # default physics — the Step-0 regime

    print("[collect] oracle 8 seeds x 10 episodes (teacher-forced)")
    oracle_trajs = [
        collect(wm, actor, grid_cfg, policy="oracle", seed=9100 + s, episodes=10)
        for s in range(8)
    ]
    print("[collect] greedy 8 seeds x 20 episodes (own distribution)")
    greedy_trajs = [
        collect(wm, actor, grid_cfg, policy="greedy", seed=9000 + s, episodes=20)
        for s in range(8)
    ]
    print("[collect] sampled-policy 4 seeds x 10 episodes (D4)")
    sampled_trajs = [
        collect(wm, actor, grid_cfg, policy="sample", seed=9200 + s, episodes=10)
        for s in range(4)
    ]

    out: dict[str, Any] = {
        "LABEL": (
            "seek-mechanism classifier diagnostics (post-close; eval/analysis "
            "only; Step-0 instance). Decision rule pre-stated in "
            "docs/decisions/probe3_5_seek_classification_2026-06-12.md §1."
        ),
        "instance": {
            "checkpoint": args.checkpoint,
            "note": str(ckpt["note"]),
            "training_age": int(ckpt["training_age"]),
        },
        "D1_decoder_honesty": {
            "oracle_OOD": d1_stats(oracle_trajs),
            "own_greedy_in_distribution": d1_stats(greedy_trajs),
        },
        "D2_consumption_transitions": {
            "own_greedy_has_data_for": d2_stats(wm, greedy_trajs, rng),
            "oracle_OOD_complement": d2_stats(wm, oracle_trajs, rng),
        },
        "D3_imagination_representability": {
            "own_greedy_contexts": d3_stats(
                wm, greedy_trajs, max_contexts=100, delta_threshold=0.04, gen=gen
            ),
            "oracle_contexts": d3_stats(
                wm, oracle_trajs, max_contexts=100, delta_threshold=0.04, gen=gen
            ),
        },
        "D4_horizon_arithmetic": {
            "greedy": d4_stats(greedy_trajs, HORIZON),
            "sampled_policy": d4_stats(sampled_trajs, HORIZON),
            "oracle_reference": d4_stats(oracle_trajs, HORIZON),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[out] wrote {out_path}")


if __name__ == "__main__":
    main()
