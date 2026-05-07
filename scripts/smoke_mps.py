#!/usr/bin/env python3
"""Day-one MPS smoke test for Probe 1 (plan §5).

Runs 100 RSSM training steps at production sizes (h=200, z=16, K=5)
on Mac MPS to confirm the substrate trains operationally on the
canonical platform. This is the platform-correctness gate the
implementation plan §5 names; the in-pytest CPU smoke
(``tests/test_integration_smoke.py``) covers gate-time correctness on
the test runner. The MPS smoke is run manually by the human builder
after Phase 8's audit lands — it is not part of the pytest suite.

Run:
    python scripts/smoke_mps.py

What passes (plan §5):
  - no MPS-fallback warnings on the hot path (during the 100 training
    steps; init-time warnings are tolerated)
  - all four telemetry sinks write valid records
  - world-model forward produces finite KL and finite recon loss for
    every step
  - backward populates gradients on the world-model and ensemble
    parameters; optimizer.step runs without error
  - all 100 iterations complete without an exception

What fails:
  - any MPS-fallback warning during the training loop
  - any NaN/Inf in KL or recon
  - any sink that fails to write a valid record
  - any exception during the 100-step loop

Failure means: the substrate decision is fine, the implementation has
a platform-specific gap. The fix is to investigate the specific
failure (e.g. add a ``.cpu()`` for an op that doesn't support MPS, or
restructure the operation), not to broad-stroke around it with
``PYTORCH_ENABLE_MPS_FALLBACK=1``.

Wall time is a *soft* bar (default 60 s for 100 training steps). If the
smoke exceeds the bar, the script prints a warning but does not fail.
Whether to drop sizes (h=128, z=8 per plan §6 "Revisit when") or to
revisit the substrate variant (per pre-probe1.md's "On stalls" framing)
is a post-smoke journal entry the human builder writes after reading
what surfaced.

On a non-Mac (or Mac without MPS), the script prints a clear message and
exits cleanly with a non-zero status. It does not silently fall back to
CPU — the smoke's purpose is platform correctness on Mac, and a CPU run
proves nothing about that.
"""

from __future__ import annotations

import sys
import tempfile
import time
import warnings
from pathlib import Path

import torch

from kind.agents.ensemble import LatentDisagreementEnsemble
from kind.agents.world_model import WorldModel, WorldModelConfig
from kind.observer.schemas import (
    PROBE_1_SCHEMA_VERSION,
    AgentStep,
    DreamRollout,
    ReplayMeta,
    WorldEvent,
)
from kind.observer.sinks import JsonlSink, ParquetSink

# ---- smoke parameters (plan §5 + §6 defaults) -----------------------------

_NUM_TRAINING_STEPS: int = 100
_BATCH_SIZE: int = 16
_SEQUENCE_LENGTH: int = 32
_WALL_TIME_SOFT_BAR_SECONDS: float = 60.0
_K_ENSEMBLE: int = 5
_PROGRESS_INTERVAL_STEPS: int = 10

# Probe 1 production world-model defaults (plan §6).
_H_DIM: int = 200
_Z_DIM: int = 16
_EMBED_DIM: int = 256
_NUM_ACTIONS: int = 5
_OBS_SIZE: int = 32
_OBS_CHANNELS: int = 1
_FREE_BITS_PER_DIM: float = 1.0

# Optimizer learning rates (RunnerConfig defaults).
_WORLD_MODEL_LR: float = 1e-4
_ENSEMBLE_LR: float = 4e-5

# Patterns we screen the warning stream for. PyTorch's MPS fallback
# emits warnings shaped like "The operator '...' is not currently
# supported on the MPS backend and will fall back to run on the CPU."
# We match conservatively on substrings any such warning would contain.
_MPS_FALLBACK_PATTERNS: tuple[str, ...] = (
    "MPS backend",
    "fall back",
    "fallback",
    "PYTORCH_ENABLE_MPS_FALLBACK",
)


# ---- detection ------------------------------------------------------------


def _detect_mps_or_exit() -> torch.device:
    """Return ``torch.device('mps')`` or print a message and exit non-zero."""
    if not torch.backends.mps.is_available():
        print(
            "[smoke] MPS device not available — this script is for Mac MPS."
            " On a non-Mac (or a Mac without MPS), run pytest's CPU smoke"
            " in tests/test_integration_smoke.py instead.",
            file=sys.stderr,
        )
        sys.exit(2)
    return torch.device("mps")


def _is_mps_fallback_warning(message: str) -> bool:
    return any(pattern in message for pattern in _MPS_FALLBACK_PATTERNS)


# ---- training step --------------------------------------------------------


def _make_world_model(device: torch.device) -> WorldModel:
    config = WorldModelConfig(
        obs_channels=_OBS_CHANNELS,
        obs_size=_OBS_SIZE,
        h_dim=_H_DIM,
        z_dim=_Z_DIM,
        embed_dim=_EMBED_DIM,
        num_actions=_NUM_ACTIONS,
        free_bits_per_dim=_FREE_BITS_PER_DIM,
    )
    return WorldModel(config).to(device)


def _make_ensemble(device: torch.device) -> LatentDisagreementEnsemble:
    return LatentDisagreementEnsemble(
        h_dim=_H_DIM,
        z_dim=_Z_DIM,
        action_dim=_NUM_ACTIONS,
        K=_K_ENSEMBLE,
    ).to(device)


def _run_one_training_step(
    world_model: WorldModel,
    ensemble: LatentDisagreementEnsemble,
    wm_opt: torch.optim.Optimizer,
    ens_opt: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one full sequence-pass training step.

    Returns ``(kl_aggregate, recon_loss)`` as Python floats so the
    caller can assert finiteness without holding tensor references on
    the device past the step.
    """
    obs_seq = torch.rand(
        _BATCH_SIZE,
        _SEQUENCE_LENGTH,
        _OBS_CHANNELS,
        _OBS_SIZE,
        _OBS_SIZE,
        device=device,
    )
    action_seq = torch.randint(
        low=0,
        high=_NUM_ACTIONS,
        size=(_BATCH_SIZE, _SEQUENCE_LENGTH),
        device=device,
        dtype=torch.long,
    )

    h = torch.zeros(_BATCH_SIZE, _H_DIM, device=device)
    z = torch.zeros(_BATCH_SIZE, _Z_DIM, device=device)
    a_prev = torch.zeros(_BATCH_SIZE, dtype=torch.long, device=device)

    h_outs: list[torch.Tensor] = []
    z_outs: list[torch.Tensor] = []
    wm_total = torch.zeros((), device=device)
    last_kl: torch.Tensor | None = None
    last_recon: torch.Tensor | None = None

    for t in range(_SEQUENCE_LENGTH):
        obs_t = obs_seq[:, t]
        wm_step = world_model.step(obs_t, h, z, a_prev)
        loss_dict = world_model.loss(wm_step, obs_target=obs_t)
        wm_total = wm_total + loss_dict["total"]
        last_kl = loss_dict["kl_aggregate_unclipped"]
        last_recon = loss_dict["recon"]
        h_outs.append(wm_step.h)
        z_outs.append(wm_step.z)
        h = wm_step.h
        z = wm_step.z
        a_prev = action_seq[:, t]

    assert last_kl is not None and last_recon is not None

    wm_opt.zero_grad(set_to_none=True)
    wm_total.backward()  # type: ignore[no-untyped-call]
    wm_opt.step()

    h_stack = torch.stack(h_outs, dim=1).detach()
    z_stack = torch.stack(z_outs, dim=1).detach()

    h_t = h_stack[:, :-1].reshape(-1, _H_DIM)
    z_t = z_stack[:, :-1].reshape(-1, _Z_DIM)
    a_t = action_seq[:, :-1].reshape(-1)
    z_target = z_stack[:, 1:].reshape(-1, _Z_DIM)

    ens_loss = ensemble.compute_loss(h_t, z_t, a_t, z_target)
    ens_opt.zero_grad(set_to_none=True)
    ens_loss["loss"].backward()  # type: ignore[no-untyped-call]
    ens_opt.step()

    return float(last_kl.item()), float(last_recon.item())


# ---- sinks ---------------------------------------------------------------


def _exercise_sinks(telemetry_dir: Path) -> None:
    """Open all four sinks, write one synthetic record each, read back valid."""
    run_id = "smoke-mps"
    h_dim = _H_DIM
    z_dim = _Z_DIM
    embed_dim = _EMBED_DIM

    agent_step_dir = telemetry_dir / "agent_step"
    dream_dir = telemetry_dir / "dream_rollout"
    replay_path = telemetry_dir / "replay_meta.jsonl"
    world_event_path = telemetry_dir / "world_event.jsonl"

    sample_agent_step = AgentStep(
        schema_version=PROBE_1_SCHEMA_VERSION,
        run_id=run_id,
        checkpoint_id=None,
        t=0,
        episode_id=0,
        step_in_episode=0,
        wallclock_ms=int(time.monotonic_ns() // 1_000_000),
        h_t=[0.0] * h_dim,
        q_params_t=([0.0] * z_dim, [0.0] * z_dim),
        p_params_t=([0.0] * z_dim, [0.0] * z_dim),
        z_t=[0.0] * z_dim,
        kl_per_dim_t=[0.0] * z_dim,
        kl_aggregate_t=0.0,
        recon_loss_t=0.0,
        action_t=0,
        action_logprob_t=0.0,
        policy_entropy_t=0.0,
        obs_hash_t="0" * 64,
        intrinsic_signal_t=0.0,
        encoder_embedding_t=[0.0] * embed_dim,
    )
    sample_dream = DreamRollout(
        schema_version=PROBE_1_SCHEMA_VERSION,
        run_id=run_id,
        checkpoint_id=None,
        seed_step=0,
        seed_h0=[0.0] * h_dim,
        seed_z0=[0.0] * z_dim,
        sequence_h=[[0.0] * h_dim],
        sequence_z_prior=[[0.0] * z_dim],
        sequence_action=[0],
        sequence_action_logprob=[0.0],
        sequence_prior_entropy=[0.0],
        sequence_decoded_obs=None,
        cumulative_prior_entropy=0.0,
        mean_step_kl_successive_priors=0.0,
        max_step_latent_norm_change=0.0,
    )
    sample_replay_meta = ReplayMeta(
        schema_version=PROBE_1_SCHEMA_VERSION,
        run_id=run_id,
        checkpoint_id=None,
        event_type="insert",
        t_event=0,
        segment_id=0,
        segment_start=0,
        segment_end=_SEQUENCE_LENGTH,
        priority=None,
        buffer_size=1,
        total_segments=1,
    )
    sample_world_event = WorldEvent(
        schema_version=PROBE_1_SCHEMA_VERSION,
        run_id=run_id,
        checkpoint_id=None,
        t_event=0,
        event_type="env_reset",
        source="environment",
        payload={"episode_id": 0},
        wallclock_ms=int(time.monotonic_ns() // 1_000_000),
    )

    with ParquetSink(agent_step_dir, AgentStep) as sink:
        sink.write(sample_agent_step)
    with ParquetSink(dream_dir, DreamRollout) as sink:
        sink.write(sample_dream)
    with JsonlSink(replay_path, ReplayMeta) as sink:
        sink.write(sample_replay_meta)
    with JsonlSink(world_event_path, WorldEvent) as sink:
        sink.write(sample_world_event)

    # Read-back validates the round-trip — the smoke's "all four sinks
    # write valid records" criterion (plan §5).
    import pyarrow.parquet as pq

    agent_shards = sorted(agent_step_dir.glob("shard-*.parquet"))
    if not agent_shards:
        raise RuntimeError("ParquetSink wrote no shard for AgentStep")
    rt_agent = AgentStep.model_validate(
        pq.read_table(agent_shards[0]).to_pylist()[0]  # type: ignore[no-untyped-call]
    )
    if rt_agent.schema_version != PROBE_1_SCHEMA_VERSION:
        raise RuntimeError(
            f"AgentStep schema_version round-trip failed: {rt_agent.schema_version!r}"
        )

    dream_shards = sorted(dream_dir.glob("shard-*.parquet"))
    if not dream_shards:
        raise RuntimeError("ParquetSink wrote no shard for DreamRollout")
    rt_dream = DreamRollout.model_validate(
        pq.read_table(dream_shards[0]).to_pylist()[0]  # type: ignore[no-untyped-call]
    )
    if rt_dream.schema_version != PROBE_1_SCHEMA_VERSION:
        raise RuntimeError("DreamRollout schema_version round-trip failed")

    rt_replay = ReplayMeta.model_validate_json(
        replay_path.read_text().splitlines()[0]
    )
    if rt_replay.schema_version != PROBE_1_SCHEMA_VERSION:
        raise RuntimeError("ReplayMeta schema_version round-trip failed")

    rt_world = WorldEvent.model_validate_json(
        world_event_path.read_text().splitlines()[0]
    )
    if rt_world.schema_version != PROBE_1_SCHEMA_VERSION:
        raise RuntimeError("WorldEvent schema_version round-trip failed")


# ---- entry point ----------------------------------------------------------


def main() -> int:
    """Run the smoke; return a process exit status (0 on success).

    The function returns rather than ``sys.exit``-ing so test code that
    imports the script can call ``main()`` directly. The ``__main__``
    block at the bottom routes the return value into ``sys.exit``.
    """
    device = _detect_mps_or_exit()

    print(
        f"[smoke] mps detected — running {_NUM_TRAINING_STEPS} training "
        f"steps at h={_H_DIM}, z={_Z_DIM}, K={_K_ENSEMBLE}, "
        f"batch={_BATCH_SIZE}, seq={_SEQUENCE_LENGTH}",
        flush=True,
    )

    # Init-time fallbacks are tolerated; the hot-path warning capture is
    # opened just before the training loop.
    world_model = _make_world_model(device)
    ensemble = _make_ensemble(device)
    wm_opt = torch.optim.Adam(world_model.parameters(), lr=_WORLD_MODEL_LR)
    ens_opt = torch.optim.Adam(ensemble.parameters(), lr=_ENSEMBLE_LR)

    # Hot-path warning capture.
    fallback_warnings: list[str] = []
    per_step_seconds: list[float] = []
    finite_kl_seen = True
    finite_recon_seen = True

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        loop_start = time.monotonic()
        for step_index in range(_NUM_TRAINING_STEPS):
            step_start = time.monotonic()
            kl_value, recon_value = _run_one_training_step(
                world_model, ensemble, wm_opt, ens_opt, device
            )
            per_step_seconds.append(time.monotonic() - step_start)

            steps_done = step_index + 1
            if (
                steps_done % _PROGRESS_INTERVAL_STEPS == 0
                or steps_done == _NUM_TRAINING_STEPS
            ):
                rolling_mean_ms = (
                    1000.0
                    * sum(per_step_seconds)
                    / len(per_step_seconds)
                )
                elapsed = time.monotonic() - loop_start
                print(
                    f"[smoke] step {steps_done}/{_NUM_TRAINING_STEPS} | "
                    f"per-step={rolling_mean_ms:.1f}ms | "
                    f"elapsed={elapsed:.1f}s | "
                    f"kl={kl_value:.3f} recon={recon_value:.3f}",
                    flush=True,
                )

            if not _is_finite(kl_value):
                finite_kl_seen = False
                print(
                    f"[smoke] non-finite KL at step {step_index}: {kl_value!r}",
                    file=sys.stderr,
                )
                break
            if not _is_finite(recon_value):
                finite_recon_seen = False
                print(
                    f"[smoke] non-finite recon at step {step_index}: "
                    f"{recon_value!r}",
                    file=sys.stderr,
                )
                break
        loop_total_seconds = time.monotonic() - loop_start

        for w in captured:
            text = str(w.message)
            if _is_mps_fallback_warning(text):
                fallback_warnings.append(text)

    # Sink exercise — covered after the hot loop so a sink failure
    # doesn't poison the warning capture above.
    sinks_ok = True
    sink_error: str | None = None
    with tempfile.TemporaryDirectory(prefix="kind_smoke_mps_") as tmp:
        try:
            _exercise_sinks(Path(tmp))
        except Exception as exc:  # pragma: no cover — surfaces in stderr
            sinks_ok = False
            sink_error = repr(exc)

    # Verify the optimizer actually populated gradients on at least one
    # parameter in each module (defensive — backward already ran above
    # so any missing-grad would point at a fundamental wiring issue).
    grads_ok = (
        any(p.grad is not None for p in world_model.parameters())
        and any(p.grad is not None for p in ensemble.parameters())
    )

    # Decide overall status.
    hot_path_clean = not fallback_warnings
    finiteness_ok = finite_kl_seen and finite_recon_seen
    overall_ok = (
        hot_path_clean and finiteness_ok and sinks_ok and grads_ok
    )

    # Soft bar warning (print, don't fail).
    if loop_total_seconds > _WALL_TIME_SOFT_BAR_SECONDS:
        print(
            f"[smoke] WARNING: wall time {loop_total_seconds:.1f}s exceeds "
            f"soft bar {_WALL_TIME_SOFT_BAR_SECONDS:.0f}s — read what "
            "surfaced and decide whether to drop sizes per plan §6 or "
            "revisit the substrate variant per pre-probe1.md 'On stalls'.",
            file=sys.stderr,
        )

    # Diagnostic detail before the one-line summary.
    if fallback_warnings:
        print(
            f"[smoke] FAIL: {len(fallback_warnings)} MPS-fallback warning(s) "
            "on the hot path:",
            file=sys.stderr,
        )
        for text in fallback_warnings[:5]:
            print(f"  - {text}", file=sys.stderr)
    if not finiteness_ok:
        print("[smoke] FAIL: non-finite KL or recon", file=sys.stderr)
    if not sinks_ok:
        print(f"[smoke] FAIL: sinks errored: {sink_error}", file=sys.stderr)
    if not grads_ok:
        print(
            "[smoke] FAIL: backward did not populate gradients on the "
            "world model or ensemble",
            file=sys.stderr,
        )

    # One-line summary per plan §5.
    mean_per_step_ms = (
        1000.0 * sum(per_step_seconds) / len(per_step_seconds)
        if per_step_seconds
        else float("nan")
    )
    status_word = "ok" if overall_ok else "FAIL"
    sinks_word = "ok" if sinks_ok else "FAIL"
    shapes_word = "ok" if finiteness_ok and grads_ok else "FAIL"
    print(
        f"[smoke] mps {status_word} | wall={loop_total_seconds:.2f}s | "
        f"per-step={mean_per_step_ms:.1f}ms | sinks {sinks_word} | "
        f"shapes {shapes_word}",
        flush=True,
    )

    return 0 if overall_ok else 1


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


if __name__ == "__main__":  # pragma: no cover — manual entry point
    sys.exit(main())
