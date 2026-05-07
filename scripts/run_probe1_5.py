#!/usr/bin/env python3
"""Probe 1.5 first env-coupled run (plan §1 Phase 7).

Drives the full Probe 1.5 substrate end-to-end on Mac MPS — the real
``self_prediction_target_mode='online'`` path, the actor's per-step
consumption of the scalar via the extended PolicyView, the EMA target's
in-place update after the world-model optimizer step, the three new
``AgentStep`` fields (``self_prediction_t``, ``self_prediction_error_t``,
``self_prediction_error_masked_t``) populated on every emission. Same
shape, seed, and cadence as ``scripts/run_probe1.py`` so the run is
directly comparable against ``runs/probe1-20260503-123926/`` per
plan §1's Phase 7 dependency graph.

Deltas from ``scripts/run_probe1.py``:

* :class:`RunnerConfig` is constructed with
  ``self_prediction_target_mode='online'`` (and the same on the nested
  :class:`WorldModelConfig`), explicit at the script level so
  Phase 7's test can verify the lesion-free path is wired without
  parsing dry-run stdout.
* Run output lives under ``runs/probe1_5-<timestamp>/``.
* A ``--dry-run`` flag short-circuits before any side effect (no env
  server, no transport, no MPS detection, no ``runs/`` directory
  creation), prints a side-effect-free summary, and returns 0 — the
  Phase 5 / Phase 6 testing pattern carried forward.
* :func:`make_runner_config` is exposed public-by-name (no leading
  underscore) so the test can call it directly without standing up the
  full transport / training stack — the Phase 5 pattern carried
  forward.

Plan §6 row 14: the first mirror call at Probe 1.5 uses the existing
Probe 1-style calibration prompt (synthesis §3 default 14 / §1.4 v2
default for the first call); ``scripts/call_mirror.py`` is invoked
against the new run with no prompt changes — Probe 2's frozen-criteria
prompt is what introduces the self-prediction quadruplet at both
reading surfaces, and is deferred per the synthesis discipline.

Run:
    .venv/bin/python scripts/run_probe1_5.py
    .venv/bin/python scripts/run_probe1_5.py --dry-run

Output lands under ``runs/probe1_5-<timestamp>/``:
    telemetry/agent_step/        — parquet shards (0.2.0 records)
    telemetry/dream_rollout/     — parquet shards (0.2.0 records,
                                   sequence_self_prediction=None per
                                   synthesis §1.5)
    telemetry/replay_meta.jsonl
    telemetry/world_event.jsonl
    checkpoints/ckpt-NNNNNN/     — atomic-rename committed; EMA target
                                   weights ride inside weights.safetensors
                                   under the world_model.target_* prefixes;
                                   the actor's extended input layer rides
                                   inside the actor.* prefix
    run.log                      — INFO-level structured event log
    summary.txt                  — captured eyeball summary (now includes
                                   the Phase 4 self-prediction lines)
"""

from __future__ import annotations

import argparse
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
from typing import Any, Literal

import torch
from tqdm import tqdm

from kind.agents.world_model import WorldModelConfig
from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import GridWorldConfig
from kind.env.transport import EnvTransportClient, EnvTransportServer
from kind.observer.eyeball import show_run_summary
from kind.observer.schemas import SCHEMA_VERSION
from kind.training.runner import (
    Runner,
    RunnerConfig,
    RunnerStepInfo,
)

# ---- run parameters -------------------------------------------------------
#
# Match ``scripts/run_probe1.py`` exactly so the Probe 1.5 main run is
# directly comparable to Probe 1's reference run on every dimension
# except the self-prediction target mode (which is ``"online"`` here vs
# the absent affordance in Probe 1). Plan §1 Phase 7: "5000 env steps,
# seed=42 for direct comparability with Probe 1's run".

_TOTAL_ENV_STEPS: int = 5000
_SEED: int = 42
_DEVICE: str = "mps"

# Cadence deltas from plan §6 defaults — same as ``run_probe1.py`` so
# the Probe 1.5 main run's checkpoint and dream rollout cadences align
# with Probe 1's reference run.
_WARMUP_ENV_STEPS: int = 200
_CHECKPOINT_EVERY_N_ENV_STEPS: int = 2500
_DREAM_CADENCE_ENV_STEPS: int = 1000

# Probe 1.5 main path: the real EMA-bootstrapped self-prediction target
# (synthesis §1.2 / §6 row 1; plan §6 row 1 / row 2 / row 3 defaults).
# Annotated as the same ``Literal`` ``RunnerConfig.self_prediction_target_mode``
# and ``WorldModelConfig.self_prediction_target_mode`` declare so mypy
# sees the value flowing into both constructors as the right type
# (Phase 5 journal flagged this gotcha).
_TARGET_MODE: Literal["online", "frozen", "environmental"] = "online"

# Progress display.
_ROLLING_WINDOW: int = 100
_PROGRESS_POSTFIX_INTERVAL: int = 50

# Run output root.
_RUNS_DIR_NAME: str = "runs"
_RUN_ID_PREFIX: str = "probe1_5"


# ---- run-id and paths -----------------------------------------------------


def _make_run_id(
    now_seconds: float | None = None, prefix: str = _RUN_ID_PREFIX
) -> str:
    """Generate a deterministic-from-clock run id like
    ``probe1_5-YYYYMMDD-HHMMSS``.

    ``prefix`` defaults to :data:`_RUN_ID_PREFIX` so existing tests that
    pin ``"probe1_5-"`` carry through unchanged. The optional override
    is what ``--run-tag`` consumes (Phase 7.5's escalation runs to
    ``runs/probe1_5_phase7_5-<timestamp>/`` per plan §6 row 15's
    documented escalation; the journal records why the tag changed).
    """
    epoch = time.time() if now_seconds is None else now_seconds
    return time.strftime(
        f"{prefix}-%Y%m%d-%H%M%S", time.localtime(epoch)
    )


def _make_run_paths(run_id: str) -> tuple[Path, Path, Path, Path]:
    """Return ``(run_dir, telemetry_dir, checkpoints_dir, run_log_path)``."""
    run_dir = Path(_RUNS_DIR_NAME) / run_id
    return (
        run_dir,
        run_dir / "telemetry",
        run_dir / "checkpoints",
        run_dir / "run.log",
    )


# ---- runner config (public helper for tests) ------------------------------


def make_runner_config(
    *,
    run_id: str,
    telemetry_dir: Path,
    checkpoints_dir: Path,
) -> RunnerConfig:
    """Build the :class:`RunnerConfig` for the Probe 1.5 main run.

    Public-by-name (no leading underscore) so the Phase 7 test can call
    this directly to verify
    ``self_prediction_target_mode == "online"`` without needing to
    invoke :func:`main` or stand up the env server. Symmetric with
    ``scripts/probe1_5_control_frozen_target.py``'s
    ``make_runner_config`` (Phase 5).

    World model + ensemble are at production sizes via
    :class:`WorldModelConfig`'s defaults (h=200, z=16, embed=256,
    head_hidden=200, ema_decay=0.99, target_mode='online',
    loss_form='cosine', K=5 via ``RunnerConfig.ensemble_k`` default).
    Three cadence parameters override §6 defaults; everything else is
    the §6 starting point.

    The runner's ``__init__`` does ``dataclasses.replace`` on the
    world-model config using the runner-level field as the source of
    truth (``runner.py`` §"models on device"); both fields are set to
    ``"online"`` here so the dry-run summary surfaces the wiring on
    both surfaces, the same shape Phase 5's frozen-target control
    helper uses for its lesion.
    """
    return RunnerConfig(
        world_model_config=WorldModelConfig(
            self_prediction_target_mode=_TARGET_MODE,
        ),
        run_id=run_id,
        telemetry_dir=telemetry_dir,
        checkpoints_dir=checkpoints_dir,
        warmup_env_steps=_WARMUP_ENV_STEPS,
        checkpoint_every_n_env_steps=_CHECKPOINT_EVERY_N_ENV_STEPS,
        dream_cadence_env_steps=_DREAM_CADENCE_ENV_STEPS,
        device=_DEVICE,
        self_prediction_target_mode=_TARGET_MODE,
    )


# ---- progress bookkeeping -------------------------------------------------


@dataclass
class _Progress:
    """Lightweight rolling-window stats fed to :data:`tqdm` and ``run.log``.

    Same shape as ``run_probe1.py``'s ``_Progress`` — duplicated here so
    the script is auditable as a single self-contained file (the
    lean-revision discipline carried from Phase 5: each run script reads
    top-to-bottom without needing to follow imports into Probe 1's
    runner).
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
    logger = logging.getLogger("kind.probe1_5.run")
    logger.setLevel(logging.INFO)
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
        print(f"[run_probe1_5] {msg}", file=sys.stderr)
        logger.error(msg)
        sys.exit(2)


def _start_transport_server(
    env_server: EnvServer, logger: logging.Logger
) -> tuple[EnvTransportServer, threading.Thread, int]:
    """Start the transport server on an ephemeral loopback port."""
    transport_server = EnvTransportServer(env_server, host="127.0.0.1", port=0)
    server_thread = threading.Thread(
        target=transport_server.serve_forever,
        name="Probe1_5EnvTransportServer",
        daemon=True,
    )
    server_thread.start()
    port = transport_server.actual_port
    logger.info(
        "transport server listening on 127.0.0.1:%d (loopback, ephemeral port)",
        port,
    )
    return transport_server, server_thread, port


def _make_env_server(run_id: str) -> EnvServer:
    """Construct the env-server.

    Same as Probe 1's: the env-server's ``world_event_handler`` is a
    placeholder no-op; the transport server replaces it with a
    wire-shipping callable when a client connects. The runner then
    routes incoming ``WorldEvent`` records into its own JSONL sink via
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
    """Closure-style callback bundling :class:`_Progress` with a tqdm bar."""

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


# ---- transport stack -----------------------------------------------------


@contextlib.contextmanager
def _transport_stack(
    env_server: EnvServer, logger: logging.Logger
) -> Iterator[tuple[EnvTransportClient, EnvTransportServer, threading.Thread]]:
    """Yield ``(client, transport_server, server_thread)``; tear down on exit."""
    transport_server, server_thread, port = _start_transport_server(
        env_server, logger
    )
    client = EnvTransportClient(
        host="127.0.0.1",
        port=port,
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


# ---- argparse + dry-run summary -----------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_probe1_5",
        description=(
            "Probe 1.5 first env-coupled run (plan §1 Phase 7). Runs the "
            "full Probe 1.5 substrate end-to-end on Mac MPS for 5000 env "
            "steps at seed=42 with self_prediction_target_mode='online' "
            "(the real Probe 1.5 path)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build the RunnerConfig and print the run plan without "
            "starting the env server, constructing the runner, "
            "detecting MPS, or creating the runs/ directory. Used by "
            "the Phase 7 test to verify the config has "
            "self_prediction_target_mode='online' without standing up "
            "the full transport / training stack."
        ),
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help=(
            "Optional override for the run-id prefix. Default is "
            f"{_RUN_ID_PREFIX!r} (the run lands at "
            f"runs/{_RUN_ID_PREFIX}-<timestamp>/). Pass e.g. "
            "'probe1_5_phase7_5' to land Phase 7.5's escalation run "
            "(plan §6 row 15) at a distinct directory while leaving "
            "Phase 7's run intact on disk."
        ),
    )
    return parser.parse_args(argv)


def _print_dry_run_summary(
    *, run_id: str, run_dir: Path, config: RunnerConfig
) -> None:
    print("=" * 60)
    print("Probe 1.5 main run — DRY RUN")
    print("=" * 60)
    print(f"  run_id:                       {run_id}")
    print(f"  run_dir (would be written):   {run_dir.resolve()}")
    print(f"  schema_version:               {SCHEMA_VERSION}")
    print(f"  seed:                         {_SEED}")
    print(f"  total_env_steps:              {_TOTAL_ENV_STEPS}")
    print(f"  device:                       {_DEVICE}")
    print(f"  warmup_env_steps:             {config.warmup_env_steps}")
    print(
        f"  checkpoint_every_n_env_steps: "
        f"{config.checkpoint_every_n_env_steps}"
    )
    print(f"  dream_cadence_env_steps:      {config.dream_cadence_env_steps}")
    print(
        f"  self_prediction_target_mode:  "
        f"{config.self_prediction_target_mode}"
    )
    print(
        f"  world_model_config.self_prediction_target_mode:  "
        f"{config.world_model_config.self_prediction_target_mode}"
    )
    print(f"  lambda_self:                  {config.lambda_self}")
    print(f"  ema_decay:                    {config.ema_decay}")
    print(
        f"  self_prediction_loss_form:    "
        f"{config.self_prediction_loss_form}"
    )
    print("=" * 60)
    print("[dry-run] no run executed; no telemetry/checkpoints written")


# ---- main entry point ----------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    prefix = args.run_tag if args.run_tag else _RUN_ID_PREFIX
    run_id = _make_run_id(prefix=prefix)
    run_dir, telemetry_dir, checkpoints_dir, run_log_path = _make_run_paths(run_id)

    config = make_runner_config(
        run_id=run_id,
        telemetry_dir=telemetry_dir,
        checkpoints_dir=checkpoints_dir,
    )

    if args.dry_run:
        _print_dry_run_summary(run_id=run_id, run_dir=run_dir, config=config)
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)

    logger = _setup_logging(run_log_path)
    logger.info("=" * 60)
    logger.info("Probe 1.5 main run starting")
    logger.info("run_id: %s", run_id)
    logger.info("host: %s", socket.gethostname())
    logger.info(
        "torch %s | mps available=%s",
        torch.__version__,
        torch.backends.mps.is_available(),
    )
    logger.info(
        "config: total_steps=%d seed=%d device=%s warmup=%d "
        "checkpoint_cadence=%d dream_cadence=%d target_mode=%s "
        "loss_form=%s lambda_self=%g ema_decay=%g",
        _TOTAL_ENV_STEPS,
        _SEED,
        _DEVICE,
        _WARMUP_ENV_STEPS,
        _CHECKPOINT_EVERY_N_ENV_STEPS,
        _DREAM_CADENCE_ENV_STEPS,
        config.self_prediction_target_mode,
        config.self_prediction_loss_form,
        config.lambda_self,
        config.ema_decay,
    )
    logger.info("schema_version: %s", SCHEMA_VERSION)
    logger.info("run_dir: %s", run_dir.resolve())

    _detect_mps_or_exit(logger)

    env_server = _make_env_server(run_id)
    runner: Runner | None = None
    progress = _Progress()
    run_start_monotonic = time.monotonic()
    exit_code = 0

    bar = tqdm(
        total=_TOTAL_ENV_STEPS,
        desc=f"run_probe1_5 {run_id}",
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
                    "interrupted (SIGINT) at env_step=%d (%.1fs since run "
                    "start) — graceful shutdown",
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
            f"[run_probe1_5] FAILED at env_step={progress.env_step}: {exc!r}",
            file=sys.stderr,
        )
        exit_code = 1

    # ---- post-run summary ----------------------------------------------
    summary_text = _capture_run_summary(telemetry_dir)
    summary_path = run_dir / "summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    logger.info("summary.txt written (%d chars)", len(summary_text))

    total_wall = time.monotonic() - run_start_monotonic
    logger.info(
        "Probe 1.5 main run finished: env_step=%d total_wall=%.1fs run_dir=%s",
        progress.env_step,
        total_wall,
        run_dir.resolve(),
    )
    logger.info("=" * 60)

    print()  # tqdm sometimes leaves the cursor mid-line
    print("=" * 60)
    print("Probe 1.5 main run complete")
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
    """Capture :func:`show_run_summary`'s stdout output as a string."""
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            show_run_summary(telemetry_dir)
    except Exception as exc:  # pragma: no cover — defensive
        return f"(eyeball.show_run_summary raised: {exc!r})\n"
    return buffer.getvalue()


if __name__ == "__main__":  # pragma: no cover — manual entry point
    sys.exit(main())
