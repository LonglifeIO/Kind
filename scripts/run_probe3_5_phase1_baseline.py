"""Probe 3.5 Phase 1 — train the energy channel (no preference), run the
dead-path battery, and measure the pure-epistemic baseline under the frozen
protocol.

This is the Phase-1 *gate* run, not a unit test. It:

1. Trains a runner to the frozen training age ``P3`` (energy telemetry on,
   ``pragmatic_value`` identically zero — the actor is untouched), with the
   world model fusing the sensed energy channel.
2. Runs the dead-path battery A–D (``kind.observer.energy_eval``) at the frozen
   pre-registration margins on a fresh eval rollout. **Battery D failing is the
   pre-registered DP9 escalation trigger** — pass ``--energy-dedicated-dims k``
   to take the escalation (weaker z-only decoder + raised free-bits floor).
3. Measures the pure-epistemic baseline (positional entropy, in-band occupancy,
   true_energy std, typical epistemic-term magnitude) — the references the
   frozen baseline-relative thresholds resolve against. Recording them is *not*
   threshold editing (pre-registration standing rule).

Usage::

    python scripts/run_probe3_5_phase1_baseline.py [--steps 5000] \
        [--energy-dedicated-dims 2] [--seed 0] [--eval-steps 1500]

Prints a JSON report; the dated results doc is written by hand from it (the
"baseline instantiation" appendix references the frozen formulas).
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
from kind.observer.energy_eval import DeadPathMargins, run_dead_path_battery
from kind.training.runner import Runner, RunnerConfig


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
        host="127.0.0.1", port=server.actual_port, world_event_handler=lambda _r: None
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
def _eval_rollout(
    runner: Runner,
    grid_cfg: GridWorldConfig,
    seed: int,
    n_steps: int,
) -> dict[str, Any]:
    """Run the trained actor in a fresh in-process env; collect the eval arrays
    (obs/action/sensed/true) plus the pure-epistemic baseline metrics."""
    device = runner.device
    wm = runner.world_model
    actor = runner.actor
    world = GridWorld(grid_cfg, seed=seed)
    step = world.reset()

    h = torch.zeros(1, wm.config.h_dim, device=device)
    z = torch.zeros(1, wm.config.z_dim, device=device)
    a_prev = torch.zeros(1, dtype=torch.long, device=device)

    obs_list: list[np.ndarray] = []
    act_list: list[int] = []
    sensed_list: list[float] = []
    true_list: list[float] = []
    disagreements: list[float] = []
    visit = np.zeros((grid_cfg.grid_size, grid_cfg.grid_size), dtype=np.int64)
    zero_scalar = torch.zeros((), device=device)

    for _ in range(n_steps):
        obs_np = step.observation.astype(np.float32) / 255.0
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).unsqueeze(0).to(device)
        sensed_t = torch.tensor(
            [[step.sensed_energy]], device=device, dtype=torch.float32
        )
        wm_step = wm.step(obs_t, h, z, a_prev, sensed_energy=sensed_t)
        intrinsic = runner.ensemble.disagreement(h, z, a_prev)
        disagreements.append(float(intrinsic.reshape(-1)[0].item()))
        view = PolicyView(h=wm_step.h, z=wm_step.z, self_prediction_error=zero_scalar)
        action = int(actor.act_greedy(view).reshape(-1)[0].item())

        gs = world.state
        visit[gs.agent_pos] += 1
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

    true_arr = np.asarray(true_list)
    setpoint = grid_cfg.energy_init / grid_cfg.energy_norm_max
    true_std = float(true_arr.std())
    # In-band = within ±1× the baseline std of the setpoint (frozen B0a/B0b).
    in_band = float(np.mean(np.abs(true_arr - setpoint) <= true_std)) if true_std > 0 else 0.0
    return {
        "obs": torch.from_numpy(np.stack(obs_list)).unsqueeze(1),
        "action": torch.tensor(act_list, dtype=torch.long),
        "sensed": torch.tensor(sensed_list, dtype=torch.float32),
        "true": torch.tensor(true_list, dtype=torch.float32),
        "baseline": {
            "setpoint": setpoint,
            "true_energy_mean": float(true_arr.mean()),
            "true_energy_std": true_std,
            "in_band_occupancy_at_1std": in_band,
            "positional_entropy_nats": _shannon_entropy(visit),
            "positional_entropy_max_nats": float(
                np.log((grid_cfg.grid_size**2))
            ),
            "epistemic_magnitude_mean_disagreement": float(np.mean(disagreements)),
            "epistemic_magnitude_disagreement_p90": float(
                np.percentile(disagreements, 90)
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000, help="training age P3")
    parser.add_argument("--energy-dedicated-dims", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-steps", type=int, default=1500)
    parser.add_argument("--warmup", type=int, default=1000)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    grid_cfg = GridWorldConfig()
    wm_cfg = WorldModelConfig(energy_dedicated_dims=args.energy_dedicated_dims)

    with tempfile.TemporaryDirectory(prefix="probe3_5_phase1_") as tmp:
        tmp_path = Path(tmp)
        with _transport_pair(grid_cfg, seed=args.seed, run_id="probe3_5_p1") as (
            client,
            env_server,
        ):
            run_cfg = RunnerConfig(
                world_model_config=wm_cfg,
                run_id="probe3_5_p1",
                telemetry_dir=tmp_path / "telemetry",
                checkpoints_dir=tmp_path / "checkpoints",
                warmup_env_steps=args.warmup,
                dream_cadence_env_steps=10_000_000,  # suppress waking-planning rollout
                checkpoint_every_n_env_steps=10_000_000,
                energy_telemetry=True,
                device="cpu",
            )
            runner = Runner(run_cfg, client, env_server=env_server)
            try:
                runner.run(total_env_steps=args.steps)
                evald = _eval_rollout(runner, grid_cfg, seed=args.seed + 1000, n_steps=args.eval_steps)
            finally:
                runner.close()

    margins = DeadPathMargins()
    report = run_dead_path_battery(
        runner.world_model,
        evald["obs"],
        evald["action"],
        evald["sensed"],
        evald["true"],
        margins=margins,
    )
    out: dict[str, Any] = {
        "config": {
            "training_age_P3": args.steps,
            "energy_dedicated_dims": args.energy_dedicated_dims,
            "seed": args.seed,
            "grid_energy": {
                k: v
                for k, v in asdict(grid_cfg).items()
                if k.startswith("energy_")
            },
        },
        "battery": {
            r.name: {"passed": r.passed, "metrics": r.metrics, "detail": r.detail}
            for r in report.results
        },
        "battery_all_passed": report.all_passed,
        "baseline_instantiation": evald["baseline"],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
