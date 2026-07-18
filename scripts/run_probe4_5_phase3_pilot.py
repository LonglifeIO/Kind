"""Probe 4.5 Phase 3 — the precision-0 pilot (negative control + toy lens).

The prereg §6 "short control-arm run": a fresh throwaway instance on
**fault-on physics from step 0** (the frozen §4 band) with the preference
engaged at **precision = 0** — the degenerate null on the same surface
(identically zero value, identically zero gradient; the §2 control-arm
config exercised for real, not approximated by ``None``). Honesty
maintenance runs at the frozen §3 cadence with its STOP live — which also
answers the Phase-1 "newly open" question of whether the margins hold
under fault-on physics.

Two Phase-3 roles for the finished instance (checkpoint throwaway after):

1. **Negative control** — its greedy-eval allocation must read inert
   (|Δ_alloc| < 0.05 under matching; the validation script computes this).
2. **The lens** — its frozen world model + ensemble score the reward toy's
   trajectories for surprise (one lens for toy and Io, plan discrepancy 5).

Usage::

    python scripts/run_probe4_5_phase3_pilot.py \
        [--steps 20000] [--seed 4502] [--device mps] \
        [--run-dir runs/probe4_5_phase3_pilot]
"""

from __future__ import annotations

import argparse
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from kind.agents.preference import EnergyPreferenceConfig
from kind.agents.world_model import WorldModelConfig
from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import GridWorldConfig
from kind.env.transport import EnvTransportClient, EnvTransportServer
from kind.observer.maintenance_refit import (
    HonestyStopError,
    MaintenanceRefitConfig,
    run_scheduled_maintenance,
)
from kind.training.runner import Runner, RunnerConfig
from kind.window.live import LiveStateWriter


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
    transport_server = EnvTransportServer(env_server, host="127.0.0.1", port=0)
    server_thread = threading.Thread(
        target=transport_server.serve_forever,
        name="Probe45Phase3PilotServer",
        daemon=True,
    )
    server_thread.start()
    client = EnvTransportClient(
        host="127.0.0.1",
        port=transport_server.actual_port,
        world_event_handler=lambda _r: None,
    )
    try:
        yield client, env_server
    finally:
        try:
            client.close()
        finally:
            transport_server.shutdown()
            server_thread.join(timeout=5.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=4502)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument(
        "--run-dir", type=str, default="runs/probe4_5_phase3_pilot"
    )
    parser.add_argument("--refit-every", type=int, default=10_000)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = Path(args.run_dir)
    # Fault-on at the frozen §4 band (defaults; only the flag is set).
    grid_cfg = GridWorldConfig(energy_fault_enabled=True)

    maintenance = MaintenanceRefitConfig(
        grid_world_config=grid_cfg,
        out_dir=run_dir / "maintenance",
        refit_every_n_env_steps=args.refit_every,
    )
    run_cfg = RunnerConfig(
        world_model_config=WorldModelConfig(energy_decoder_bounded=True),
        run_id="probe4_5_phase3_pilot",
        telemetry_dir=run_dir / "telemetry",
        checkpoints_dir=run_dir / "checkpoints",
        checkpoint_every_n_env_steps=args.refit_every,
        energy_telemetry=True,
        # The degenerate null on the same surface (§5): value and gradient
        # identically zero, the control-arm config for real.
        energy_preference=EnergyPreferenceConfig(precision=0.0),
        maintenance_refit=maintenance,
        device=args.device,
    )

    summary: dict[str, Any] = {
        "LABEL": (
            "Probe 4.5 Phase 3 pilot — precision-0 control-arm instance on "
            "fault-on physics; throwaway; lens + negative control."
        ),
        "steps": args.steps,
        "seed": args.seed,
        "device": args.device,
        "fault_on": True,
        "precision": 0.0,
    }

    stop: HonestyStopError | None = None
    with _transport_pair(
        grid_cfg, seed=args.seed, run_id="probe4_5_phase3_pilot"
    ) as (client, env_server):
        live_writer = LiveStateWriter(
            env_server, run_dir, run_id=run_cfg.run_id
        )
        runner = Runner(
            run_cfg, client, env_server=env_server, step_callback=live_writer
        )
        try:
            runner.run(total_env_steps=args.steps)
            # Burn-in-close refit + final checkpoint (the attempt-1 lesson:
            # the loop covers env steps 0..N-1; the close always runs
            # explicitly, then the post-close state commits).
            run_scheduled_maintenance(
                runner._world_model,  # noqa: SLF001 — pause point; loop over
                runner._actor,  # noqa: SLF001
                maintenance,
                env_step=args.steps,
                occasion="burn_in_close",
                run_label=run_cfg.run_id,
            )
            runner._commit_checkpoint(args.steps)  # noqa: SLF001
        except HonestyStopError as e:
            stop = e
        finally:
            runner.close()

    reports = []
    for path in sorted((run_dir / "maintenance").glob("refit_*.json")):
        r = json.loads(path.read_text())
        reports.append(
            {
                "env_step": r["env_step"],
                "occasion": r["occasion"],
                "all_passed": r["verdict"]["all_passed"],
            }
        )
    summary["refits"] = reports
    summary["honesty_stop"] = None if stop is None else str(stop)
    summary["completed"] = stop is None and all(
        r["all_passed"] for r in reports
    )

    out_path = run_dir / "pilot_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\n[out] wrote {out_path}")
    if not summary["completed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
