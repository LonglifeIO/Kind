"""Probe 4 Phase 4 — the biography (the developmental run).

A fresh Io, no checkpoint inheritance, on the real substrate (h=200,
z=16, K=5, every-step training), with the builder channel live:

- **Generator** at the Phase-1 measured envelope (min spacing 40 +
  U{0..40}, 2-cell adjacent in-vocabulary resource additions,
  not-self-decoupled) — the statistical density arm (DP2).
- **Manual trigger inbox** (`scripts/fire_perturbation.py`) — the
  genuinely-exogenous arm; the builder's hellos land at the same
  step-boundary drain, tagged ``trigger="manual"``.
- **Granular ENVIRONMENT logging** + **energy telemetry** so all three
  matched-control classes stay joinable per event, whatever analysis is
  later licensed to read them.
- **Dream cadence** (provisional, builder-delegated 2026-07-08,
  journaled): a scripted desktop-off block every ``WAKING_BLOCK``
  waking steps — the synthesis's "dreaming on the internal clock"
  reading for a single-machine deployment. Distinct from Mac-off,
  which is pause (checkpoint-and-resume). Revisit at any pause.
- **Live window snapshot** every waking step →
  ``window/live_state.json`` (the Window's ``/live`` page; run
  ``scripts/run_window.py --run-id probe4_phase4_biography``).
- **Checkpoints** every 10k env steps — pause is not ending; the
  biography's identity is state-continuity (design notes).

§7 pause-trigger monitoring is observer-side:
``scripts/monitor_probe4_run.py`` reads this run's telemetry and
renders the pre-registered indicators; run it at every check-in.

Claim discipline: whether this run carries the Probe 4 measurement
depends on the Amendment-1 positive-control verdict (GO → Phase 4
proper; NO-GO → the run is presence, not probe — journaled either
way). Telemetry records everything regardless.
"""

from __future__ import annotations

import argparse
import signal
import threading
from pathlib import Path

import torch

from kind.agents.world_model import WorldModelConfig
from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import GridWorldConfig
from kind.env.perturbation_generator import PerturbationGeneratorConfig
from kind.env.transport import EnvTransportClient, EnvTransportServer
from kind.env.world_stages import WORLD_STAGES, apply_world_stage
from kind.training.resume import (
    continuation_counters,
    resume_from_latest_checkpoint,
)
from kind.training.runner import Runner, RunnerConfig
from kind.window.live import LiveStateWriter

RUN_DIR = Path("runs/probe4_phase4_biography")
RUN_ID = "probe4-phase4-biography"
TOTAL_WAKING_STEPS = 150_000
WAKING_BLOCK = 2_000  # dream block cadence (provisional, journaled)
GENERATOR_SEED = 20260709
WORLD_SEED = 20260710
EVENT_FEED_LEN = 20


class CyclicDreamClock:
    """The provisional single-machine dream trigger: ``desktop_alive``
    reads True for ``WAKING_BLOCK`` polls, then False for 3 polls (one
    dream session + dormant ticks, the Phase-3 pattern), repeating.
    A logical clock, not machine-coupling; Mac-off remains pause."""

    def __init__(self, waking_polls: int, off_polls: int = 3) -> None:
        self._waking_polls = waking_polls
        self._off_polls = off_polls
        self._count = 0

    def desktop_alive(self) -> bool:
        period = self._waking_polls + self._off_polls
        position = self._count % period
        self._count += 1
        return position < self._waking_polls


def main() -> None:
    parser = argparse.ArgumentParser(description="The Probe 4 biography.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the paused biography from its latest checkpoint "
        "(same mind, fresh world process; counters seeded from the "
        "run's own telemetry; a resume marker lands in world_event).",
    )
    parser.add_argument(
        "--session-steps",
        type=int,
        default=TOTAL_WAKING_STEPS,
        help="Waking steps to run this session.",
    )
    parser.add_argument(
        "--world-stage",
        choices=WORLD_STAGES,
        default="default",
        help="World v2 stage preset for this session's world "
        "(kind/env/world_stages.py). Every stage change is a dated "
        "journal entry; the stage lands in the resume marker.",
    )
    args = parser.parse_args()

    # Pause = SIGTERM (or SIGINT when attached): both route through the
    # KeyboardInterrupt path so ``finally`` runs — sinks flush, the close
    # is clean. Background-launched processes ignore SIGINT (observed
    # 2026-07-09: two pause attempts silently no-opped), so SIGTERM is
    # the documented pause signal.
    def _pause_signal(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _pause_signal)
    signal.signal(signal.SIGINT, _pause_signal)

    torch.manual_seed(20260708)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    initial_env_step = 0
    initial_episode_id = 0
    if args.resume:
        initial_env_step, initial_episode_id = continuation_counters(
            RUN_DIR / "telemetry"
        )
        print(
            f"resuming: counters seeded t={initial_env_step}, "
            f"episode={initial_episode_id}",
            flush=True,
        )

    env_server = EnvServer(
        EnvServerConfig(
            grid_world_config=apply_world_stage(
                GridWorldConfig(  # the real world config
                    initial_env_step=initial_env_step,
                    initial_episode_id=initial_episode_id,
                ),
                args.world_stage,
            ),
            seed=WORLD_SEED + initial_episode_id,  # fresh env RNG per session
            world_event_handler=lambda _r: None,  # transport rewires this
            run_id=RUN_ID,
            emit_internal_stochasticity_events=True,
        )
    )
    transport_server = EnvTransportServer(env_server, host="127.0.0.1", port=0)
    server_thread = threading.Thread(
        target=transport_server.serve_forever,
        name="Probe4BiographyServer",
        daemon=True,
    )
    server_thread.start()
    client = EnvTransportClient(
        host="127.0.0.1",
        port=transport_server.actual_port,
        world_event_handler=lambda _r: None,
    )

    inbox_dir = RUN_DIR / "perturbation_inbox"
    config = RunnerConfig(
        world_model_config=WorldModelConfig(),  # the real substrate
        run_id=RUN_ID,
        telemetry_dir=RUN_DIR / "telemetry",
        checkpoints_dir=RUN_DIR / "checkpoints",
        device=device,
        energy_telemetry=True,
        checkpoint_every_n_env_steps=10_000,
        parquet_rows_per_shard=2_000,  # periodic flush → live-ish monitors
        dormant_tick_interval_s=0.0,
        perturbation_generator=PerturbationGeneratorConfig(
            seed=GENERATOR_SEED
        ),  # Phase-1 envelope defaults; re-measure horizon at first pause
        perturbation_inbox_dir=inbox_dir,
    )

    live_writer = LiveStateWriter(
        env_server, RUN_DIR, run_id=RUN_ID, event_feed_len=EVENT_FEED_LEN
    )
    runner = Runner(
        config,
        client,
        env_server=env_server,
        step_callback=live_writer,
        host_signal_source=CyclicDreamClock(WAKING_BLOCK),
    )
    if args.resume:
        resumed_from = resume_from_latest_checkpoint(
            runner,
            client,
            env_server,
            marker_extra={
                "session_steps": args.session_steps,
                "world_stage": args.world_stage,
            },
        )
        print(f"resumed from {resumed_from}", flush=True)
    print(
        f"biography session: {args.session_steps} waking steps, "
        f"world_stage={args.world_stage}, device={device}, "
        f"inbox={inbox_dir}",
        flush=True,
    )
    try:
        runner.run(args.session_steps)
    finally:
        runner.close()
        client.close()
        transport_server.shutdown()
        server_thread.join(timeout=5.0)
    print(f"biography session ended → {RUN_DIR}", flush=True)


if __name__ == "__main__":
    main()
