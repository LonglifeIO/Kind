#!/usr/bin/env python3
"""Probe 1.5 frozen-target control variant — substrate-side test of
failure mode (a) (inert affordance) per synthesis §1.7(a) v2 / plan
§2.8 / plan §8.1.

This script extends ``scripts/run_probe1.py``'s structure with a single
substantive delta: the ``RunnerConfig.self_prediction_target_mode`` is
set to ``"frozen"`` so the world model's :meth:`compute_self_prediction_target`
returns a fixed random-orthogonal projection of the online ``h_t``
instead of the EMA target's ``bar{h}_{t+1}``. The projection preserves
rank but breaks semantic alignment (plan §8.1 / §2.2 discussion); the
head learns a fixed function of its own input rather than the actual
next state.

If the run's structural metrics — per-dimension KL distribution, weight-
distribution moments, ``kl_aggregate_t`` trajectories — turn out
statistically *indistinguishable* from Probe 1's no-affordance baseline,
the head is being driven by trivial means and the Probe 1.5 affordance
is dead at the substrate-side reading. If the metrics are
distinguishable from both Probe 1 and the frozen-target run, the
affordance is alive at the substrate-side. The behavior-side test
(whether Io's policy actually conditions on the scalar) is the
counterfactual probe's job (plan §2.9 / §8.3, deferred per the lean
revision §13).

A ``mirror_marker`` ``world_event`` is emitted at run start with
``source="system"`` and a payload naming the lesion shape and
rationale, so the journal entry and any later mirror reading land on
an explicit timestamped record of "this run carries this specific
lesion" rather than having to infer the variant from the run id.

**Lean revision (plan §13).** This is the only Phase 5 control script
built. The environmental-auxiliary control variant (plan §2.8 second
half) and the multi-run comparison driver
``scripts/probe1_5_compare_controls.py`` (plan §2.8 third script) are
deferred to a later phase if Phase 8's reading shows the deferred
apparatus is needed.

Run:
    .venv/bin/python scripts/probe1_5_control_frozen_target.py
    .venv/bin/python scripts/probe1_5_control_frozen_target.py --dry-run

Output lands under ``runs/probe1_5_control_frozen_target-<timestamp>/``
with the same telemetry / checkpoint / log / summary layout as
``run_probe1.py``.
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
from kind.observer.schemas import SCHEMA_VERSION, WorldEvent
from kind.training.runner import (
    Runner,
    RunnerConfig,
    RunnerStepInfo,
)

# ---- run parameters -------------------------------------------------------
#
# These match ``scripts/run_probe1.py`` exactly so the frozen-target run
# is directly comparable to the Probe 1 baseline and the (future) Probe
# 1.5 main run on every dimension except the self-prediction target
# mode. Plan §8.4: "Each training-run control starts from the same
# fresh init as the Probe 1.5 main run: same seed (42), same total env
# steps (5000), same RunnerConfig except the self_prediction_target_mode
# flag."

_TOTAL_ENV_STEPS: int = 5000
_SEED: int = 42
_DEVICE: str = "mps"

# Cadence deltas from plan §6 defaults — same as ``run_probe1.py`` so
# the frozen-target run's checkpoint and dream rollout cadences align
# with Probe 1's reference run.
_WARMUP_ENV_STEPS: int = 200
_CHECKPOINT_EVERY_N_ENV_STEPS: int = 2500
_DREAM_CADENCE_ENV_STEPS: int = 1000

# The lesion this script installs. ``_TARGET_MODE`` is annotated as
# the same ``Literal`` ``RunnerConfig.self_prediction_target_mode`` and
# ``WorldModelConfig.self_prediction_target_mode`` declare so mypy sees
# the value flowing into both constructors as the right type.
_TARGET_MODE: Literal["online", "frozen", "environmental"] = "frozen"
_LESION_KIND: str = "frozen_target"
_LESION_RATIONALE: str = (
    "Frozen-target ablation: the self-prediction head's supervisory "
    "target is a fixed random-orthogonal projection of the online h_t "
    "instead of the EMA target's bar{h}_{t+1}. Rank-preserving, "
    "semantic-alignment-broken (synthesis §1.7(a) v2 / plan §8.1). "
    "Substrate-side test of failure mode (a): if structural metrics "
    "across episodes are statistically indistinguishable from Probe 1's "
    "no-affordance baseline, the affordance is inert at the "
    "substrate-side reading."
)

# Progress display.
_ROLLING_WINDOW: int = 100
_PROGRESS_POSTFIX_INTERVAL: int = 50

# Run output root.
_RUNS_DIR_NAME: str = "runs"
_RUN_ID_PREFIX: str = "probe1_5_control_frozen_target"


# ---- run-id and paths -----------------------------------------------------


def _make_run_id(now_seconds: float | None = None) -> str:
    """Generate a deterministic-from-clock run id like
    ``probe1_5_control_frozen_target-YYYYMMDD-HHMMSS``."""
    epoch = time.time() if now_seconds is None else now_seconds
    return time.strftime(
        f"{_RUN_ID_PREFIX}-%Y%m%d-%H%M%S", time.localtime(epoch)
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
    """Build the :class:`RunnerConfig` for the frozen-target control run.

    Public-by-name (no leading underscore) so the Phase 5 test can call
    this directly to verify ``self_prediction_target_mode == "frozen"``
    without needing to invoke :func:`main` or stand up the env server.
    The ``WorldModelConfig`` is the production default plus
    ``self_prediction_target_mode="frozen"``; the runner's
    ``__init__`` applies ``dataclasses.replace`` on the world-model
    config so the runner-side ``self_prediction_target_mode="frozen"``
    is the source of truth (runner.py §"models on device").
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
    the control script is auditable as a single self-contained file
    (the lean-revision discipline: each control script reads top-to-
    bottom without needing to follow imports into Probe 1's runner).
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
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("kind.probe1_5.control_frozen_target.run")
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
        print(f"[probe1_5_control_frozen_target] {msg}", file=sys.stderr)
        logger.error(msg)
        sys.exit(2)


def _start_transport_server(
    env_server: EnvServer, logger: logging.Logger
) -> tuple[EnvTransportServer, threading.Thread, int]:
    transport_server = EnvTransportServer(env_server, host="127.0.0.1", port=0)
    server_thread = threading.Thread(
        target=transport_server.serve_forever,
        name="Probe1_5ControlFrozenTargetEnvTransportServer",
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


# ---- mirror_marker emission ----------------------------------------------


def _emit_lesion_mirror_marker(
    runner: Runner, run_id: str, logger: logging.Logger
) -> None:
    """Write a ``mirror_marker`` ``world_event`` naming the lesion this
    run installs.

    Goes through the runner's already-open ``_world_event_sink`` so the
    JSONL write shares the file handle with the runner's other emissions
    — same pattern the runner itself uses when emitting the Probe 1 →
    Probe 1.5 checkpoint-load mirror_marker (``runner.py`` §"mirror_marker
    for Probe 1 → Probe 1.5 transitions"). Accessing the private sink
    attribute is the in-process-script equivalent of an
    ``emit_world_event`` method that the runner does not expose; the
    alternative — opening a second JsonlSink against the same file —
    risks file-handle interleaving inside the JsonlSink's append-mode
    open. The runner is in the same process and the sink's
    ``write`` is the only public interface.
    """
    t_event = (
        runner._env_step_meta.env_step
        if runner._env_step_meta is not None
        else 0
    )
    runner._world_event_sink.write(
        WorldEvent(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            checkpoint_id=runner.latest_checkpoint_id,
            t_event=t_event,
            event_type="mirror_marker",
            source="system",
            payload={
                "lesion_kind": _LESION_KIND,
                "rationale": _LESION_RATIONALE,
                "self_prediction_target_mode": _TARGET_MODE,
                "seed": _SEED,
                "total_env_steps": _TOTAL_ENV_STEPS,
                "run_id": run_id,
            },
            wallclock_ms=time.monotonic_ns() // 1_000_000,
        )
    )
    logger.info(
        "mirror_marker emitted at run start: lesion_kind=%s "
        "self_prediction_target_mode=%s",
        _LESION_KIND,
        _TARGET_MODE,
    )


# ---- main entry point ----------------------------------------------------


@contextlib.contextmanager
def _transport_stack(
    env_server: EnvServer, logger: logging.Logger
) -> Iterator[tuple[EnvTransportClient, EnvTransportServer, threading.Thread]]:
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


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="probe1_5_control_frozen_target",
        description=(
            "Probe 1.5 frozen-target control variant. Runs the same "
            "5000 env-step Probe 1.5 substrate with "
            "self_prediction_target_mode='frozen' so the head's "
            "supervisory target is a fixed random-orthogonal projection "
            "of the online h_t. Substrate-side test of failure mode "
            "(a) per synthesis §1.7(a) v2 / plan §8.1."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build the RunnerConfig and print the run plan without "
            "starting the env server or constructing the runner. "
            "Used by the Phase 5 test to verify the config has "
            "self_prediction_target_mode='frozen' without standing up "
            "the full transport / training stack."
        ),
    )
    return parser.parse_args(argv)


def _print_dry_run_summary(
    *, run_id: str, run_dir: Path, config: RunnerConfig
) -> None:
    print("=" * 60)
    print("Probe 1.5 frozen-target control — DRY RUN")
    print("=" * 60)
    print(f"  run_id:                       {run_id}")
    print(f"  run_dir (would be written):   {run_dir.resolve()}")
    print(f"  seed:                         {_SEED}")
    print(f"  total_env_steps:              {_TOTAL_ENV_STEPS}")
    print(f"  device:                       {_DEVICE}")
    print(f"  warmup_env_steps:             {config.warmup_env_steps}")
    print(
        f"  self_prediction_target_mode:  "
        f"{config.self_prediction_target_mode}"
    )
    print(
        f"  world_model_config.self_prediction_target_mode:  "
        f"{config.world_model_config.self_prediction_target_mode}"
    )
    print(f"  lesion_kind (mirror_marker):  {_LESION_KIND}")
    print("=" * 60)
    print("[dry-run] no run executed; no telemetry/checkpoints written")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    run_id = _make_run_id()
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
    logger.info("Probe 1.5 frozen-target control run starting")
    logger.info("run_id: %s", run_id)
    logger.info("host: %s", socket.gethostname())
    logger.info(
        "torch %s | mps available=%s",
        torch.__version__,
        torch.backends.mps.is_available(),
    )
    logger.info(
        "config: total_steps=%d seed=%d device=%s warmup=%d "
        "checkpoint_cadence=%d dream_cadence=%d target_mode=%s",
        _TOTAL_ENV_STEPS,
        _SEED,
        _DEVICE,
        _WARMUP_ENV_STEPS,
        _CHECKPOINT_EVERY_N_ENV_STEPS,
        _DREAM_CADENCE_ENV_STEPS,
        _TARGET_MODE,
    )
    logger.info("lesion_kind=%s", _LESION_KIND)
    logger.info("run_dir: %s", run_dir.resolve())

    _detect_mps_or_exit(logger)

    env_server = _make_env_server(run_id)
    runner: Runner | None = None
    progress = _Progress()
    run_start_monotonic = time.monotonic()
    exit_code = 0

    bar = tqdm(
        total=_TOTAL_ENV_STEPS,
        desc=f"probe1_5_control_frozen_target {run_id}",
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
            # Emit the lesion's mirror_marker before the hot loop starts so
            # the world_event JSONL has the lesion-naming record at
            # t_event=0, ahead of any builder_perturbation / env_reset
            # records the env-server's reader thread will deliver as the
            # transport handshake completes.
            _emit_lesion_mirror_marker(runner, run_id, logger)
            try:
                runner.run(total_env_steps=_TOTAL_ENV_STEPS)
                logger.info(
                    "runner.run completed: env_step=%d wall=%.1fs",
                    progress.env_step,
                    time.monotonic() - run_start_monotonic,
                )
            except KeyboardInterrupt:
                logger.warning(
                    "interrupted (SIGINT) at env_step=%d (%.1fs since "
                    "run start) — graceful shutdown",
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
            f"[probe1_5_control_frozen_target] FAILED at "
            f"env_step={progress.env_step}: {exc!r}",
            file=sys.stderr,
        )
        exit_code = 1

    summary_text = _capture_run_summary(telemetry_dir)
    summary_path = run_dir / "summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    logger.info("summary.txt written (%d chars)", len(summary_text))

    total_wall = time.monotonic() - run_start_monotonic
    logger.info(
        "Probe 1.5 frozen-target control run finished: env_step=%d "
        "total_wall=%.1fs run_dir=%s",
        progress.env_step,
        total_wall,
        run_dir.resolve(),
    )
    logger.info("=" * 60)

    print()
    print("=" * 60)
    print("Probe 1.5 frozen-target control run complete")
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
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            show_run_summary(telemetry_dir)
    except Exception as exc:  # pragma: no cover — defensive
        return f"(eyeball.show_run_summary raised: {exc!r})\n"
    return buffer.getvalue()


if __name__ == "__main__":  # pragma: no cover — manual entry point
    sys.exit(main())
