#!/usr/bin/env python3
"""Probe 1 first end-to-end run.

Drives the full Probe 1 substrate end-to-end on Mac MPS:

* :class:`~kind.env.env_server.EnvServer` over the production grid config
  (8x8, three cell types, two stochastic processes, 200-step episodes),
  exposed via TCP on loopback.
* :class:`~kind.env.transport.EnvTransportServer` and
  :class:`~kind.env.transport.EnvTransportClient` paired across loopback —
  the same shape Probe 3 will eventually drive across two machines.
* :class:`~kind.training.runner.Runner` driving the world model + actor +
  ensemble + replay + telemetry sinks + checkpoint manager. Production
  sizes per plan §6 except for the deltas listed under "deltas from
  defaults" in the script body.

The smoke (``scripts/smoke_mps.py``) verified the substrate trains
operationally on this machine. This script is the next step: real
loopback transport, real telemetry accumulation against the actual
grid world, real training kicking in after warmup. The mirror is *not*
called here — the mirror is a separate manual step against the
parquet shards this run leaves on disk.

Run:
    .venv/bin/python scripts/run_probe1.py

Output lands under ``runs/probe1-<timestamp>/``:
    telemetry/agent_step/        — parquet shards
    telemetry/dream_rollout/     — parquet shards (cadence: 1000 env steps)
    telemetry/replay_meta.jsonl
    telemetry/world_event.jsonl
    checkpoints/ckpt-NNNNNN/     — atomic-rename committed
    run.log                      — INFO-level structured event log
    summary.txt                  — captured eyeball summary

Progress is shown via ``tqdm`` with a postfix carrying running 100-step
means of KL / recon / intrinsic / policy entropy. The postfix updates
every ``_PROGRESS_POSTFIX_INTERVAL`` env steps so the bar stays
readable; the bar itself ticks every step.

If MPS is not available the script refuses to run with a clear stderr
message and a non-zero exit. SIGINT (Ctrl+C) is treated as a graceful
early termination — partial telemetry stays on disk, the runner is
closed, the script exits 0. Any uncaught exception is logged with a
traceback to ``run.log`` and the script exits 1; the partial run's
telemetry is still inspectable with the eyeball helpers.
"""

from __future__ import annotations

import contextlib
import io
import logging
import socket
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from kind.agents.world_model import WorldModelConfig
from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import GridWorldConfig
from kind.env.transport import EnvTransportClient, EnvTransportServer
from kind.observer.eyeball import show_run_summary
from kind.training.runner import (
    Runner,
    RunnerConfig,
    RunnerStepInfo,
)

# ---- run parameters -------------------------------------------------------
#
# Hardcoded rather than argparse-driven for the first run, per the user's
# brief: reproducibility is more valuable than configurability for the
# initial Probe 1 invocation. Edit these constants if you want to vary
# any of them in subsequent runs.

_TOTAL_ENV_STEPS: int = 5000
_SEED: int = 42
_DEVICE: str = "mps"

# Deltas from plan §6 defaults:
#   warmup_env_steps=200       — default 1000; lower so training kicks in
#                                  during the run (after ~1 episode), not
#                                  near the end.
#   checkpoint_every_n=2500    — default 10_000; fires once mid-run instead
#                                  of never-firing in a 5000-step run.
#   dream_cadence_env_steps=1000 — keeps the §6 default; ~5 dream rollouts.
_WARMUP_ENV_STEPS: int = 200
_CHECKPOINT_EVERY_N_ENV_STEPS: int = 2500
_DREAM_CADENCE_ENV_STEPS: int = 1000

# Progress display.
_ROLLING_WINDOW: int = 100
_PROGRESS_POSTFIX_INTERVAL: int = 50

# Run output root. Each invocation creates a timestamped subdirectory.
_RUNS_DIR_NAME: str = "runs"


# ---- run-id and paths -----------------------------------------------------


def _make_run_id(now_seconds: float | None = None) -> str:
    """Generate a deterministic-from-clock run id like ``probe1-YYYYMMDD-HHMMSS``."""
    epoch = time.time() if now_seconds is None else now_seconds
    return time.strftime("probe1-%Y%m%d-%H%M%S", time.localtime(epoch))


def _make_run_paths(run_id: str) -> tuple[Path, Path, Path, Path]:
    """Return ``(run_dir, telemetry_dir, checkpoints_dir, run_log_path)``."""
    run_dir = Path(_RUNS_DIR_NAME) / run_id
    return (
        run_dir,
        run_dir / "telemetry",
        run_dir / "checkpoints",
        run_dir / "run.log",
    )


# ---- progress bookkeeping -------------------------------------------------


@dataclass
class _Progress:
    """Lightweight rolling-window stats fed to :data:`tqdm` and ``run.log``.

    All fields are mutable; the runner's ``step_callback`` invokes
    :meth:`update` on each iteration, the main thread reads them through
    the rolling-mean accessors when refreshing the postfix.
    """

    env_step: int = 0
    last_dream_at: int | None = None
    last_checkpoint_at: int | None = None
    current_checkpoint_id: str | None = None
    last_episode_id: int = 0

    _kl: deque[float] = field(default_factory=lambda: deque(maxlen=_ROLLING_WINDOW))
    _recon: deque[float] = field(
        default_factory=lambda: deque(maxlen=_ROLLING_WINDOW)
    )
    _intrinsic: deque[float] = field(
        default_factory=lambda: deque(maxlen=_ROLLING_WINDOW)
    )
    _entropy: deque[float] = field(
        default_factory=lambda: deque(maxlen=_ROLLING_WINDOW)
    )

    def update(self, info: RunnerStepInfo) -> None:
        self.env_step = info.env_step
        self.last_episode_id = info.episode_id
        self._kl.append(info.kl_aggregate)
        self._recon.append(info.recon_loss)
        self._intrinsic.append(info.intrinsic_signal)
        self._entropy.append(info.policy_entropy)
        if info.dream_emitted:
            self.last_dream_at = info.env_step
        if info.checkpoint_committed:
            self.last_checkpoint_at = info.env_step
            self.current_checkpoint_id = info.checkpoint_id

    def postfix_dict(self) -> dict[str, str]:
        if not self._kl:
            return {}
        return {
            "ep": f"{self.last_episode_id}",
            "kl": f"{_mean(self._kl):.3f}",
            "recon": f"{_mean(self._recon):.3f}",
            "intr": f"{_mean(self._intrinsic):.4f}",
            "entH": f"{_mean(self._entropy):.3f}",
        }


def _mean(buf: deque[float]) -> float:
    return sum(buf) / len(buf) if buf else 0.0


# ---- logging --------------------------------------------------------------


def _setup_logging(run_log_path: Path) -> logging.Logger:
    """Configure the run logger.

    File handler at INFO; stderr stays quiet (WARNING+) so the tqdm bar
    doesn't fight a parallel stream of INFO lines. Returns the configured
    logger.
    """
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("kind.probe1.run")
    logger.setLevel(logging.INFO)
    # If this script is re-imported (e.g. by a test), avoid duplicate handlers.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    file_fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    file_handler = logging.FileHandler(run_log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(file_fmt)
    logger.addHandler(stderr_handler)

    logger.propagate = False
    return logger


# ---- transport stack helpers ---------------------------------------------


def _detect_mps_or_exit(logger: logging.Logger) -> None:
    if not torch.backends.mps.is_available():
        msg = (
            "MPS device not available — this script targets Mac MPS. "
            "Run on a Mac, or change _DEVICE in this script."
        )
        print(f"[run_probe1] {msg}", file=sys.stderr)
        logger.error(msg)
        sys.exit(2)


def _start_transport_server(
    env_server: EnvServer, logger: logging.Logger
) -> tuple[EnvTransportServer, threading.Thread, int]:
    """Start the transport server on an ephemeral loopback port.

    Returns ``(transport_server, server_thread, actual_port)``. The
    thread is daemon so a hung shutdown still lets the process exit;
    callers should still call :meth:`EnvTransportServer.shutdown` to
    drain cleanly.
    """
    transport_server = EnvTransportServer(env_server, host="127.0.0.1", port=0)
    server_thread = threading.Thread(
        target=transport_server.serve_forever,
        name="Probe1EnvTransportServer",
        daemon=True,
    )
    server_thread.start()
    port = transport_server.actual_port
    logger.info(
        "transport server listening on 127.0.0.1:%d (loopback, ephemeral port)",
        port,
    )
    return transport_server, server_thread, port


def _make_runner_config(
    *,
    run_id: str,
    telemetry_dir: Path,
    checkpoints_dir: Path,
) -> RunnerConfig:
    """Build :class:`RunnerConfig` for the Probe 1 run.

    World model + ensemble are at production sizes via
    :class:`WorldModelConfig`'s defaults (h=200, z=16, embed=256, K=5
    via :class:`RunnerConfig.ensemble_k` default). Three cadence
    parameters override §6 defaults; everything else is the §6
    starting point.
    """
    return RunnerConfig(
        world_model_config=WorldModelConfig(),
        run_id=run_id,
        telemetry_dir=telemetry_dir,
        checkpoints_dir=checkpoints_dir,
        warmup_env_steps=_WARMUP_ENV_STEPS,
        checkpoint_every_n_env_steps=_CHECKPOINT_EVERY_N_ENV_STEPS,
        dream_cadence_env_steps=_DREAM_CADENCE_ENV_STEPS,
        device=_DEVICE,
    )


def _make_env_server(run_id: str) -> EnvServer:
    """Construct the env-server.

    The env-server's ``world_event_handler`` is a placeholder no-op; the
    transport server replaces it with a wire-shipping callable when a
    client connects. The runner then routes the incoming ``WorldEvent``
    records into its own JSONL sink via
    :meth:`EnvTransportClient.set_world_event_handler`.
    """
    return EnvServer(
        EnvServerConfig(
            grid_world_config=GridWorldConfig(),
            seed=_SEED,
            world_event_handler=lambda _record: None,
            run_id=run_id,
        )
    )


# ---- training-progress reporter -----------------------------------------


class _ProgressReporter:
    """Closure-style callback bundling :class:`_Progress` with a tqdm bar.

    The runner calls :meth:`__call__` every iteration; the reporter
    updates the rolling stats, increments the bar, and refreshes the
    postfix every :data:`_PROGRESS_POSTFIX_INTERVAL` env steps. The
    logger captures lifecycle events (warmup complete, dream emitted,
    checkpoint committed) at the same callback site so the run.log
    timeline aligns with the bar.
    """

    def __init__(
        self,
        bar: "tqdm[Any]",
        progress: _Progress,
        logger: logging.Logger,
        warmup: int,
        run_start_monotonic: float,
    ) -> None:
        self._bar = bar
        self._progress = progress
        self._logger = logger
        self._warmup = warmup
        self._run_start = run_start_monotonic
        self._first_step_logged: bool = False
        self._warmup_logged: bool = False
        self._training_logged: bool = False

    def __call__(self, info: RunnerStepInfo) -> None:
        if not self._first_step_logged:
            self._logger.info(
                "first agent_step received (env_step=%d, episode=%d) — "
                "transport handshake complete, hot loop running",
                info.env_step,
                info.episode_id,
            )
            self._first_step_logged = True

        self._progress.update(info)
        self._bar.update(1)

        if (
            self._progress.env_step % _PROGRESS_POSTFIX_INTERVAL == 0
            or self._progress.env_step == _TOTAL_ENV_STEPS - 1
        ):
            self._bar.set_postfix(self._progress.postfix_dict(), refresh=True)

        if not self._warmup_logged and info.env_step >= self._warmup:
            self._logger.info(
                "warmup complete at env_step=%d (%.1fs since run start) — "
                "training step now firing every iteration",
                info.env_step,
                time.monotonic() - self._run_start,
            )
            self._warmup_logged = True
            self._training_logged = True

        if info.dream_emitted:
            self._logger.info(
                "dream rollout emitted at env_step=%d (%.1fs since run start)",
                info.env_step,
                time.monotonic() - self._run_start,
            )

        if info.checkpoint_committed:
            self._logger.info(
                "checkpoint committed: %s at env_step=%d (%.1fs since run start)",
                info.checkpoint_id,
                info.env_step,
                time.monotonic() - self._run_start,
            )


# ---- main entry point ----------------------------------------------------


@contextlib.contextmanager
def _transport_stack(
    env_server: EnvServer, logger: logging.Logger
) -> Iterator[tuple[EnvTransportClient, EnvTransportServer, threading.Thread]]:
    """Yield ``(client, transport_server, server_thread)``; tear down on exit.

    The client is *not* connected here — the runner calls ``connect()``
    inside its own ``run()``. Cleanup is best-effort: any individual
    close that raises is logged at WARNING and swallowed so the rest of
    the teardown still runs.
    """
    transport_server, server_thread, port = _start_transport_server(
        env_server, logger
    )
    client = EnvTransportClient(
        host="127.0.0.1",
        port=port,
        # Placeholder — the runner replaces this via
        # set_world_event_handler at construction.
        world_event_handler=lambda _record: None,
    )
    try:
        yield client, transport_server, server_thread
    finally:
        try:
            client.close()
        except Exception as exc:  # pragma: no cover — best-effort cleanup
            logger.warning("transport client close raised: %r", exc)
        try:
            transport_server.shutdown()
        except Exception as exc:  # pragma: no cover
            logger.warning("transport server shutdown raised: %r", exc)
        server_thread.join(timeout=5.0)
        if server_thread.is_alive():
            logger.warning(
                "transport server thread did not exit within 5s — daemon "
                "thread will die with the process"
            )


def main() -> int:
    run_id = _make_run_id()
    run_dir, telemetry_dir, checkpoints_dir, run_log_path = _make_run_paths(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = _setup_logging(run_log_path)
    logger.info("=" * 60)
    logger.info("Probe 1 run starting")
    logger.info("run_id: %s", run_id)
    logger.info("host: %s", socket.gethostname())
    logger.info(
        "torch %s | mps available=%s",
        torch.__version__,
        torch.backends.mps.is_available(),
    )
    logger.info(
        "config: total_steps=%d seed=%d device=%s warmup=%d "
        "checkpoint_cadence=%d dream_cadence=%d",
        _TOTAL_ENV_STEPS,
        _SEED,
        _DEVICE,
        _WARMUP_ENV_STEPS,
        _CHECKPOINT_EVERY_N_ENV_STEPS,
        _DREAM_CADENCE_ENV_STEPS,
    )
    logger.info("run_dir: %s", run_dir.resolve())

    _detect_mps_or_exit(logger)

    env_server = _make_env_server(run_id)
    runner: Runner | None = None
    progress = _Progress()
    run_start_monotonic = time.monotonic()
    exit_code = 0

    bar = tqdm(
        total=_TOTAL_ENV_STEPS,
        desc=f"run_probe1 {run_id}",
        unit="step",
        dynamic_ncols=True,
        smoothing=0.1,
        mininterval=0.5,
    )

    try:
        with _transport_stack(env_server, logger) as (
            transport_client,
            _transport_server,
            _server_thread,
        ):
            config = _make_runner_config(
                run_id=run_id,
                telemetry_dir=telemetry_dir,
                checkpoints_dir=checkpoints_dir,
            )
            reporter = _ProgressReporter(
                bar=bar,
                progress=progress,
                logger=logger,
                warmup=_WARMUP_ENV_STEPS,
                run_start_monotonic=run_start_monotonic,
            )
            runner = Runner(
                config,
                transport_client,
                env_server=env_server,
                step_callback=reporter,
            )
            try:
                runner.run(total_env_steps=_TOTAL_ENV_STEPS)
                logger.info(
                    "runner.run completed: env_step=%d wall=%.1fs",
                    progress.env_step,
                    time.monotonic() - run_start_monotonic,
                )
            except KeyboardInterrupt:
                logger.warning(
                    "interrupted (SIGINT) at env_step=%d (%.1fs since run start) — "
                    "graceful shutdown",
                    progress.env_step,
                    time.monotonic() - run_start_monotonic,
                )
            finally:
                bar.close()
                runner.close()
    except Exception as exc:
        bar.close()
        if runner is not None:
            with contextlib.suppress(Exception):
                runner.close()
        logger.error(
            "run failed at env_step=%d: %r", progress.env_step, exc
        )
        logger.error("traceback:\n%s", traceback.format_exc())
        print(
            f"[run_probe1] FAILED at env_step={progress.env_step}: {exc!r}",
            file=sys.stderr,
        )
        exit_code = 1

    # ---- post-run summary ----------------------------------------------
    # Even on partial-failure runs, emit the summary so the journal
    # entry has something to read against.
    summary_text = _capture_run_summary(telemetry_dir)
    summary_path = run_dir / "summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    logger.info("summary.txt written (%d chars)", len(summary_text))

    total_wall = time.monotonic() - run_start_monotonic
    logger.info(
        "Probe 1 run finished: env_step=%d total_wall=%.1fs run_dir=%s",
        progress.env_step,
        total_wall,
        run_dir.resolve(),
    )
    logger.info("=" * 60)

    print()  # tqdm sometimes leaves the cursor mid-line
    print("=" * 60)
    print("Probe 1 run complete")
    print(f"  run_id:    {run_id}")
    print(f"  env_steps: {progress.env_step + 1} of {_TOTAL_ENV_STEPS}")
    print(f"  wall:      {total_wall:.1f}s")
    print(f"  run_dir:   {run_dir.resolve()}")
    print(f"  log:       {run_log_path.resolve()}")
    print(f"  summary:   {summary_path.resolve()}")
    print("=" * 60)
    print()
    print("---- run summary ----")
    print(summary_text)

    return exit_code


def _capture_run_summary(telemetry_dir: Path) -> str:
    """Capture :func:`show_run_summary`'s stdout output as a string.

    ``show_run_summary`` prints rather than returns; the script wants
    both the print-to-stdout-at-end behaviour *and* a copy on disk in
    ``summary.txt``. ``contextlib.redirect_stdout`` is the cleanest way
    to capture without changing eyeball.py's contract.
    """
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            show_run_summary(telemetry_dir)
    except Exception as exc:  # pragma: no cover — defensive
        return f"(eyeball.show_run_summary raised: {exc!r})\n"
    return buffer.getvalue()


if __name__ == "__main__":  # pragma: no cover — manual entry point
    sys.exit(main())
