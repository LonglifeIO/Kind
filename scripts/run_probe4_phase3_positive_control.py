"""Probe 4 Phase 3 — positive control (instrument validation, GO/NO-GO).

Plan Phase 3 / frozen prereg §6: before the real question is asked, a
deliberately **blatant** planted source-category must make the §2
detectors fire with 2× headroom. This run validates the instrument, not
Io; the instance is **throwaway by construction** (checkpoints not
carried forward).

**The planted category.** Transient 2×2 WALL blocks placed *in Io's
view* every ``PULSE_PERIOD`` waking steps and removed ``PULSE_HOLD``
steps later, fired through the tested mutator surface
(``trigger="generator"``). Prereg §6 explicitly allows the planted
channel to be content-blatant ("an object type/placement the world never
produces, injected at an unmistakable cadence") — walls are in the
world's legal vocabulary but no internal process ever produces one, and
their appearance/disappearance is guaranteed-distinct from regrowth. The
*real* Phase-4 generator stays in-vocabulary (resource additions); only
the instrument test gets this blunt.

**The three classes in-run:** SELF = consumptions (energy telemetry on);
ENVIRONMENT = regrowth (granular per-event logging on, default p=0.01);
BUILDER = the wall pulses. Dream sessions are produced by a
``ScriptedDesktop`` schedule (a desktop-off block every
``WAKING_BLOCK`` waking steps) — scripted machinery for instrument
validation; the biography's single-machine dream trigger remains the
open Phase-4 ``[BUILDER]`` decision, unpre-empted.

Substrate: the real config (h=200, z=16, K=5, every-step training) — the
instrument must be validated on the substrate Phase 4 will use.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import torch

from kind.agents.world_model import WorldModelConfig
from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import CellType, GridWorldConfig
from kind.env.transport import EnvTransportClient, EnvTransportServer
from kind.training.runner import Runner, RunnerConfig, RunnerStepInfo
from kind.training.state_machine import ScriptedDesktop

RUN_DIR = Path("runs/probe4_phase3_positive_control")
TOTAL_WAKING_STEPS = 10_000
WAKING_BLOCK = 2_000  # a dream block after every this-many waking steps
PULSE_PERIOD = 25
PULSE_HOLD = 5
EXCLUSION_RADIUS = 1  # never on/adjacent-to Io (the not-self floor)


class WallPulseInjector:
    """The blatant planted channel, fired from the runner's step callback.

    Places a 2×2 wall block at Chebyshev ≥ 2 from Io (inside the 7×7
    view, so every pulse is observed) on EMPTY cells only; removes it
    ``PULSE_HOLD`` steps later, guarding against episode resets having
    wiped it (removal fires only on cells still WALL). All mutations are
    ``trigger="generator"`` builder events through the tested surface.
    """

    _ORIENTATIONS = (
        ((2, 1), (2, 2), (3, 1), (3, 2)),
        ((2, -2), (2, -1), (3, -2), (3, -1)),
        ((-3, 1), (-3, 2), (-2, 1), (-2, 2)),
        ((-3, -2), (-3, -1), (-2, -2), (-2, -1)),
    )

    def __init__(self, env_server: EnvServer) -> None:
        self._env_server = env_server
        self._pending_removals: list[tuple[int, tuple[int, int]]] = []
        self._pulses_placed = 0
        self._started = time.monotonic()

    def __call__(self, info: RunnerStepInfo) -> None:
        state = self._env_server.grid_world_state
        grid = state.grid
        ar, ac = state.agent_pos
        step = info.env_step

        # Remove expired blocks (only cells still WALL — an episode
        # reset may have wiped them already).
        due = [cell for due_at, cell in self._pending_removals if step >= due_at]
        self._pending_removals = [
            (due_at, cell)
            for due_at, cell in self._pending_removals
            if step < due_at
        ]
        for cell in due:
            if grid[cell] == CellType.WALL.value:
                self._env_server.remove_object(
                    cell, CellType.WALL, trigger="generator"
                )

        if step > 0 and step % PULSE_PERIOD == 0:
            grid = self._env_server.grid_world_state.grid  # post-removal
            size = grid.shape[0]
            for offsets in self._ORIENTATIONS:
                cells = [(ar + dr, ac + dc) for dr, dc in offsets]
                # Blatant-channel eligibility: in-bounds, outside the
                # not-self exclusion, and not already WALL. Overwriting a
                # RESOURCE is allowed — the default world saturates with
                # resources and an all-EMPTY guard starves the planted
                # cadence; destroying the odd resource is acceptable on a
                # throwaway instrument-validation instance (journaled).
                if all(
                    0 <= r < size
                    and 0 <= c < size
                    and grid[r, c] != CellType.WALL.value
                    and max(abs(r - ar), abs(c - ac)) > EXCLUSION_RADIUS
                    for r, c in cells
                ):
                    for cell in cells:
                        self._env_server.set_cell_state(
                            cell, CellType.WALL, trigger="generator"
                        )
                        self._pending_removals.append(
                            (step + PULSE_HOLD, cell)
                        )
                    self._pulses_placed += 1
                    break

        if step > 0 and step % 500 == 0:
            elapsed = time.monotonic() - self._started
            print(
                f"step {step}  pulses {self._pulses_placed}  "
                f"{elapsed:.0f}s  ({elapsed / step * 1000:.0f} ms/step)",
                flush=True,
            )


def main() -> None:
    torch.manual_seed(42)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    env_server = EnvServer(
        EnvServerConfig(
            grid_world_config=GridWorldConfig(),  # the real world config
            seed=20260708,
            world_event_handler=lambda _r: None,
            run_id="probe4-phase3-positive-control",
            emit_internal_stochasticity_events=True,
        )
    )
    transport_server = EnvTransportServer(env_server, host="127.0.0.1", port=0)
    server_thread = threading.Thread(
        target=transport_server.serve_forever,
        name="Probe4PositiveControlServer",
        daemon=True,
    )
    server_thread.start()
    client = EnvTransportClient(
        host="127.0.0.1",
        port=transport_server.actual_port,
        world_event_handler=lambda _r: None,
    )

    config = RunnerConfig(
        world_model_config=WorldModelConfig(),  # the real substrate
        run_id="probe4-phase3-positive-control",
        telemetry_dir=RUN_DIR / "telemetry",
        checkpoints_dir=RUN_DIR / "checkpoints",
        device=device,
        energy_telemetry=True,
        checkpoint_every_n_env_steps=10_000_000,  # throwaway: no checkpoints
        dormant_tick_interval_s=0.0,  # scripted absences are ticks, not wallclock
    )

    # A desktop-off block (3 ticks: dream session, →dormant, dormant)
    # after every WAKING_BLOCK waking steps; ends True so waking resumes.
    schedule: list[bool] = []
    for _ in range(TOTAL_WAKING_STEPS // WAKING_BLOCK):
        schedule += [True] * WAKING_BLOCK + [False] * 3
    schedule += [True]
    desktop = ScriptedDesktop(schedule)

    injector = WallPulseInjector(env_server)
    runner = Runner(
        config,
        client,
        env_server=env_server,
        step_callback=injector,
        host_signal_source=desktop,
    )
    try:
        runner.run(TOTAL_WAKING_STEPS)
    finally:
        runner.close()
        client.close()
        transport_server.shutdown()
        server_thread.join(timeout=5.0)
    print(
        f"done: {TOTAL_WAKING_STEPS} waking steps, "
        f"{injector._pulses_placed} wall pulses → {RUN_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()
