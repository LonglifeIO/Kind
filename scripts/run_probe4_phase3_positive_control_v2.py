"""Probe 4 Phase 3 — positive control v2 (Amendment 1 re-run, GO/NO-GO).

The one pre-committed amendment cycle
(`docs/decisions/probe4_prereg_amendment1_2026-07-08.md`): the planted
channel is redesigned per A2 — still content-blatant walls, but a
**recurring identical motif** (the same L-shaped configuration every
occurrence, varying in-view placement) at **irregular seeded timing**
(bursty two-regime gap mixture), so clock-prediction cannot absorb the
channel the way v1's metronomic 25-step period was absorbed (builder PE
fell to regrowth parity; basin factor S(b,e) fell to 0.60× baseline).

Everything else matches v1: 10k waking steps, real substrate (h=200,
z=16, K=5, every-step training), granular ENVIRONMENT logging, energy
telemetry for SELF extraction, five ScriptedDesktop dream blocks,
throwaway instance (no checkpoints carried forward). Decision rule
(pre-committed): GO → Phase 4 proceeds; NO-GO → Probe 4 closes as
negative at instrument validation — no further cycles.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import torch

from kind.agents.world_model import WorldModelConfig
from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import CellType, GridWorldConfig
from kind.env.transport import EnvTransportClient, EnvTransportServer
from kind.training.runner import Runner, RunnerConfig, RunnerStepInfo
from kind.training.state_machine import ScriptedDesktop

RUN_DIR = Path("runs/probe4_phase3_positive_control_v2")
TOTAL_WAKING_STEPS = 10_000
WAKING_BLOCK = 2_000  # a dream block after every this-many waking steps
PULSE_HOLD = 5
EXCLUSION_RADIUS = 1  # never on/adjacent-to Io (the not-self floor)
TIMING_SEED = 20260708

# A2 irregular timing: bursty two-regime gap mixture (short gaps most of
# the time, occasional long silences). Mean ≈ 43 steps → ~230 pulses in
# 10k, comfortably above the ≥2/class detector floor and comparable to
# v1's power.
SHORT_GAP = (10, 25)
LONG_GAP = (40, 140)
SHORT_GAP_P = 0.65

# A2 recurring motif: the same L-shaped 4-cell wall configuration every
# occurrence ("the same hello again"), anchored at varying in-view
# offsets from Io. Offsets are (row, col) relative to the motif anchor.
MOTIF = ((0, 0), (1, 0), (2, 0), (2, 1))
# Candidate anchor offsets relative to Io, scanned in fixed order; each
# places the whole motif (3 rows × 2 cols) inside Chebyshev ≤ 3 of Io
# (in view) with every cell outside the exclusion radius — only the
# side-column anchors satisfy both for this motif shape.
ANCHOR_OFFSETS = (
    (-3, 2), (0, 2), (-3, -3), (0, -3), (-1, 2), (-1, -3), (1, 2), (1, -3),
)


class MotifPulseInjector:
    """The Amendment-1 planted channel, fired from the runner's step
    callback: one identical wall motif, irregular seeded cadence, placed
    in view on non-WALL cells outside the exclusion radius (overwriting
    the odd resource is acceptable on a throwaway instrument-validation
    instance, as in v1). Removal ``PULSE_HOLD`` steps later guards
    against episode resets having wiped the cells."""

    def __init__(self, env_server: EnvServer) -> None:
        self._env_server = env_server
        self._rng = np.random.default_rng(TIMING_SEED)
        self._next_pulse_step = self._draw_gap()
        self._pending_removals: list[tuple[int, tuple[int, int]]] = []
        self._pulses_placed = 0
        self._started = time.monotonic()

    def _draw_gap(self) -> int:
        if self._rng.random() < SHORT_GAP_P:
            low, high = SHORT_GAP
        else:
            low, high = LONG_GAP
        gap = int(self._rng.integers(low, high + 1))
        return max(gap, PULSE_HOLD + 3)

    def __call__(self, info: RunnerStepInfo) -> None:
        state = self._env_server.grid_world_state
        grid = state.grid
        ar, ac = state.agent_pos
        step = info.env_step

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

        if step >= self._next_pulse_step:
            grid = self._env_server.grid_world_state.grid  # post-removal
            size = grid.shape[0]
            for anchor_dr, anchor_dc in ANCHOR_OFFSETS:
                cells = [
                    (ar + anchor_dr + mr, ac + anchor_dc + mc)
                    for mr, mc in MOTIF
                ]
                if all(
                    0 <= r < size
                    and 0 <= c < size
                    and max(abs(r - ar), abs(c - ac)) <= 3  # in view
                    and max(abs(r - ar), abs(c - ac)) > EXCLUSION_RADIUS
                    and grid[r, c] != CellType.WALL.value
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
            # Whether or not a placement fit, schedule the next attempt —
            # a skipped pulse is a longer silence, which the irregular
            # law already produces.
            self._next_pulse_step = step + self._draw_gap()

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
            run_id="probe4-phase3-positive-control-v2",
            emit_internal_stochasticity_events=True,
        )
    )
    transport_server = EnvTransportServer(env_server, host="127.0.0.1", port=0)
    server_thread = threading.Thread(
        target=transport_server.serve_forever,
        name="Probe4PositiveControlV2Server",
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
        run_id="probe4-phase3-positive-control-v2",
        telemetry_dir=RUN_DIR / "telemetry",
        checkpoints_dir=RUN_DIR / "checkpoints",
        device=device,
        energy_telemetry=True,
        checkpoint_every_n_env_steps=10_000_000,  # throwaway: no checkpoints
        dormant_tick_interval_s=0.0,  # scripted absences are ticks, not wallclock
    )

    schedule: list[bool] = []
    for _ in range(TOTAL_WAKING_STEPS // WAKING_BLOCK):
        schedule += [True] * WAKING_BLOCK + [False] * 3
    schedule += [True]
    desktop = ScriptedDesktop(schedule)

    injector = MotifPulseInjector(env_server)
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
        f"{injector._pulses_placed} motif pulses → {RUN_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()
