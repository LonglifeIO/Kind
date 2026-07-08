"""Probe 4 Phase 2 — interleaved three-class smoke (plan Phase 2 deliverable).

A short CPU run demonstrating all three event classes live in one
telemetry stream at per-event granularity, with the generator firing at
the Phase-1-derived envelope and one manual event injected mid-run
through the trigger inbox:

- SELF — Io's own consumptions (extracted from AgentStep,
  ``observer.source_events``);
- ENVIRONMENT — regrowth additions (granular
  ``internal_stochasticity_event`` records, Phase 1);
- BUILDER — generator events (``trigger="generator"``) + one manual
  event (``trigger="manual"``).

This is plumbing demonstration, not a measurement of Io: the world model
is the tiny smoke config, and nothing here feeds any probe verdict. The
statistical-signature comparison (generator timing/magnitude vs
regrowth's) is computed by the companion analysis inline in the journal
entry; the run artifacts live under ``runs/probe4_phase2_smoke/``.
"""

from __future__ import annotations

import threading
from pathlib import Path

import torch

from kind.agents.world_model import WorldModelConfig
from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import GridWorldConfig
from kind.env.perturbation_generator import PerturbationGeneratorConfig
from kind.env.transport import EnvTransportClient, EnvTransportServer
from kind.env.trigger_inbox import write_trigger_request
from kind.training.runner import Runner, RunnerConfig

RUN_DIR = Path("runs/probe4_phase2_smoke")
TOTAL_STEPS = 3000
MANUAL_AT_STEP = 1500


def main() -> None:
    torch.manual_seed(1234)
    run_dir = RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=True)
    inbox = run_dir / "perturbation_inbox"

    env_server = EnvServer(
        EnvServerConfig(
            # Default world: regrowth p=0.01 with drift, 4 resources —
            # the ENVIRONMENT class must be the real process, not a
            # quiet stand-in.
            grid_world_config=GridWorldConfig(),
            seed=20260707,
            world_event_handler=lambda _r: None,
            run_id="probe4-phase2-smoke",
            emit_internal_stochasticity_events=True,
        )
    )
    transport_server = EnvTransportServer(env_server, host="127.0.0.1", port=0)
    server_thread = threading.Thread(
        target=transport_server.serve_forever,
        name="Probe4Phase2SmokeServer",
        daemon=True,
    )
    server_thread.start()
    client = EnvTransportClient(
        host="127.0.0.1",
        port=transport_server.actual_port,
        world_event_handler=lambda _r: None,
    )

    config = RunnerConfig(
        world_model_config=WorldModelConfig(
            obs_channels=1,
            obs_size=32,
            h_dim=32,
            z_dim=4,
            embed_dim=32,
            num_actions=5,
            action_emb_dim=8,
            mlp_hidden=32,
            free_bits_per_dim=1.0,
        ),
        run_id="probe4-phase2-smoke",
        telemetry_dir=run_dir / "telemetry",
        checkpoints_dir=run_dir / "checkpoints",
        action_dim=5,
        ensemble_k=2,
        imagination_horizon=4,
        replay_capacity=5000,
        replay_sequence_length=8,
        replay_batch_size=4,
        train_every_n_env_steps=4,
        warmup_env_steps=100,
        dream_cadence_env_steps=1000,
        dream_horizon=4,
        checkpoint_every_n_env_steps=100_000,  # no checkpoints needed
        parquet_rows_per_shard=1000,
        device="cpu",
        energy_telemetry=True,
        # The Phase-1-derived envelope: spacing floor = the measured raw
        # h-trace horizon (~40 steps); jitter keeps timing stochastic.
        perturbation_generator=PerturbationGeneratorConfig(
            seed=7,
            min_spacing_steps=40,
            spacing_jitter_steps=40,
            cells_per_event=2,
            exclusion_radius=1,
        ),
        perturbation_inbox_dir=inbox,
    )

    runner = Runner(config, client, env_server=env_server)
    try:
        runner.run(MANUAL_AT_STEP)
        write_trigger_request(inbox, "add_resource", {"cell": [0, 7]})
        runner.run(TOTAL_STEPS - MANUAL_AT_STEP)
    finally:
        runner.close()
        client.close()
        transport_server.shutdown()
        server_thread.join(timeout=5.0)
    print(f"done: {TOTAL_STEPS} steps → {run_dir}")


if __name__ == "__main__":
    main()
