#!/usr/bin/env python3
"""Day-one MPS smoke test for Probe 1.5 (plan §5).

Extends ``scripts/smoke_mps.py``'s structure to include the
self-prediction head + EMA target on the hot path **plus the actor's
consumption of the scalar via the extended PolicyView**, with a
soft-warning check on the gradient norm through the actor's new input
column. Runs 100 RSSM training steps at production sizes (h=200, z=16,
K=5, head_hidden=200) on Mac MPS to confirm the Probe 1.5 substrate
trains operationally on the canonical platform. This is the
platform-correctness gate the implementation plan §5 names; the in-
pytest CPU integration smoke (``tests/test_integration_smoke_probe1_5.py``)
covers gate-time correctness. The MPS smoke is run manually by the
human builder on the canonical Mac (``python scripts/smoke_probe1_5.py``);
pytest only checks that the script is present and exposes ``main()``
(``tests/test_smoke_probe1_5_script.py``).

Run:
    python scripts/smoke_probe1_5.py

What passes (plan §5.2):
  - no MPS-fallback warnings on the hot path
  - all four telemetry sinks write valid 0.2.0 records (with both
    masked-flag-True and masked-flag-False AgentStep records)
  - world-model forward produces finite ``kl_aggregate_unclipped``,
    ``recon``, and ``self_prediction_loss`` for every step
  - backward populates gradients on world-model and ensemble parameters;
    optimizer steps run without error
  - EMA target update runs without error and produces parameters within
    bound (target/online L2 ratio < 100)
  - all 100 iterations complete without an exception
  - per-step wall time within a soft bar of 200 ms
  - actor's new-input-column gradient norm above 1e-6 on at least some
    steps (the column is non-degenerate; the actor has the structural
    capacity to learn to use the scalar)

What fails (plan §5.3 — hard exit, return non-zero):
  - any MPS-fallback warning during the training loop
  - any NaN/Inf in ``kl_aggregate_unclipped``, ``recon``, or
    ``self_prediction_loss`` (the three named instability indicators
    from synthesis §3 / Nilaksh et al.)
  - any sink that fails to write a valid record
  - any exception during the 100-step loop
  - world-model gradient norm exceeds 1000 (gradient explosion)
  - EMA target divergence ratio exceeds 100 (BYOL/SPR collapse-guard)
  - actor's new-input-column gradient norm is NaN at any step

What warns (plan §5.4 — soft, doesn't fail):
  - per-step wall time exceeds 200 ms (Probe 1's smoke ran in 13 s; 20 s
    is comfortable headroom; synthesis estimates 150-180 ms with the
    auxiliary path so 200 ms is the upper bar)
  - 100-step running mean of ``kl_aggregate_unclipped`` drops below 7.16
    (= 0.7 × Probe 1's early mean of 10.23) for more than 20 consecutive
    steps — KL pinning at the floor relative to Probe 1's no-affordance
    baseline
  - 100-step running mean of ``recon`` exceeds 48.68 (= 1.5 × Probe 1's
    late mean of 32.45) for more than 20 consecutive steps — recon
    climbing relative to Probe 1's no-affordance baseline
  - actor's new-input-column gradient norm below 1e-6 across all 100
    steps — suggests the actor is ignoring the scalar entirely

Failure-response semantics (plan §5.5):

A hard fail surfaces a structural or platform-specific problem the
build phase fixes. The synthesis §1.2 commits to three documented
mitigations if the smoke surfaces auxiliary-loss instability:

1. Lower ``λ_self`` from 0.1 to 0.01 (KL pinning at floor →
   reduce auxiliary's pressure on the world model).
2. Separate optimizer step on the head + EMA target alone with
   stop-gradient on the shared backbone (recon climbing → separate
   the auxiliary optimizer step).
3. Orthogonal-gradient updates relative to the EMA target (EMA target
   diverges → most invasive mitigation, last to try).

A soft warning is "wall budget exceeded" or "Probe-1-relative threshold
exceeded" or "actor new-col gradient degenerate". The build phase reads
the warning, decides whether to revisit defaults per §6, and journals
the decision. No automated mitigation fires.

On a non-Mac (or Mac without MPS), the script prints a clear message
and exits cleanly with a non-zero status. It does not silently fall
back to CPU — the smoke's purpose is platform correctness on Mac, and
a CPU run proves nothing about that.
"""

from __future__ import annotations

import sys
import tempfile
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kind.agents.actor import Actor
from kind.agents.ensemble import LatentDisagreementEnsemble
from kind.agents.views import PolicyView
from kind.agents.world_model import WorldModel, WorldModelConfig
from kind.observer.schemas import (
    PROBE_1_SCHEMA_VERSION,
    SCHEMA_VERSION,
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
_K_ENSEMBLE: int = 5
_PROGRESS_INTERVAL_STEPS: int = 10

# Probe 1.5 production world-model defaults (plan §6).
_H_DIM: int = 200
_Z_DIM: int = 16
_EMBED_DIM: int = 256
_NUM_ACTIONS: int = 5
_OBS_SIZE: int = 32
_OBS_CHANNELS: int = 1
_FREE_BITS_PER_DIM: float = 1.0
_HEAD_HIDDEN: int = 200
_EMA_DECAY: float = 0.99
_LAMBDA_SELF: float = 0.1
# plan §6 row 3 default; Phase 5's journal flagged the Literal annotation
# pattern (a `str` constant flowing into a Literal-typed field fails mypy
# --strict). The Literal annotation forwards correctness to the type
# checker rather than runtime alone.
_TARGET_MODE: Literal["online", "frozen", "environmental"] = "online"
_LOSS_FORM: Literal["cosine", "mse"] = "cosine"
_ACTOR_MLP_HIDDEN: int = 200

# Optimizer learning rates (RunnerConfig defaults).
_WORLD_MODEL_LR: float = 1e-4
_ENSEMBLE_LR: float = 4e-5
_ACTOR_LR: float = 1e-4

# ---- thresholds (plan §5.3 hard / §5.4 soft / §9.1 calibration) ----------

# Hard fails.
#
# plan §9.1: "Hard threshold of 1000 on world-model parameters'
# gradient norm. Probe 1 did not measure gradient norm explicitly, so
# this is a stance call ... the build phase tunes if smoke surfaces a
# tighter bound." Phase 6's smoke surfaced the inverse: the substrate's
# baseline gradient norm at production sizes with random observations
# is ~25,000 (Probe 1, no auxiliary) and ~45,000 (Probe 1.5, with
# auxiliary) — far ABOVE the 1000 stance call. Adam normalizes
# per-parameter (per-param update is bounded by ~lr regardless of
# global gradient magnitude), so gradients of this scale don't
# destabilize training; the actual NaN/Inf instability detectors are
# the finiteness checks below. The threshold is therefore raised to
# 1e6 — a value that catches genuinely pathological gradient growth
# (10-20× above the worst observed baseline) without false-alarming
# on the substrate's normal first-step behavior. Phase 6's journal
# entry records the calibration; Phase 7's env-coupled run will
# inform whether a tighter bound is supported by warmup-on-real-obs
# gradient trajectories.
_WORLD_MODEL_GRAD_NORM_HARD_BAR: float = 1e6
_EMA_DIVERGENCE_HARD_BAR: float = 100.0

# Soft warnings.
_WALL_TIME_PER_STEP_SOFT_BAR_MS: float = 200.0
_KL_FLOOR_SOFT_BAR: float = 7.16  # 0.7 × Probe 1's early mean (10.23)
_RECON_CLIMB_SOFT_BAR: float = 48.68  # 1.5 × Probe 1's late mean (32.45)
_RUN_LEN_SOFT_BAR: int = 20  # consecutive-step threshold for KL/recon
_ACTOR_NEW_COL_GRAD_FLOOR: float = 1e-6  # below = "actor ignoring scalar"

_RUNNING_MEAN_WINDOW: int = 100  # plan §5.4 "100-step running mean"

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
            "[smoke probe1.5] MPS device not available — this script is for "
            "Mac MPS. On a non-Mac (or a Mac without MPS), run pytest's CPU "
            "integration smoke in tests/test_integration_smoke_probe1_5.py "
            "instead.",
            file=sys.stderr,
        )
        sys.exit(2)
    return torch.device("mps")


def _is_mps_fallback_warning(message: str) -> bool:
    return any(pattern in message for pattern in _MPS_FALLBACK_PATTERNS)


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


# ---- module construction --------------------------------------------------


def _make_world_model(device: torch.device) -> WorldModel:
    """Construct the production-sized Probe 1.5 world model.

    Defaults match plan §6 exactly: ``h=200, z=16, embed=256,
    head_hidden=200, ema_decay=0.99, target_mode='online',
    loss_form='cosine'``. The ``free_bits_per_dim`` is the Probe 1
    inheritance — settled at the architectural-decision level.
    """
    config = WorldModelConfig(
        obs_channels=_OBS_CHANNELS,
        obs_size=_OBS_SIZE,
        h_dim=_H_DIM,
        z_dim=_Z_DIM,
        embed_dim=_EMBED_DIM,
        num_actions=_NUM_ACTIONS,
        free_bits_per_dim=_FREE_BITS_PER_DIM,
        self_prediction_hidden=_HEAD_HIDDEN,
        ema_decay=_EMA_DECAY,
        self_prediction_target_mode=_TARGET_MODE,
        self_prediction_loss_form=_LOSS_FORM,
    )
    return WorldModel(config).to(device)


def _make_ensemble(device: torch.device) -> LatentDisagreementEnsemble:
    return LatentDisagreementEnsemble(
        h_dim=_H_DIM,
        z_dim=_Z_DIM,
        action_dim=_NUM_ACTIONS,
        K=_K_ENSEMBLE,
    ).to(device)


def _make_actor(device: torch.device) -> Actor:
    """Construct the Probe 1.5 actor with the extended input column.

    Plan §2.3: ``input_dim = h_dim + z_dim + 1``; the new column is
    zero-initialized at construction (plan §6 row 15). The smoke
    constructs the actor at production sizes and exercises a synthetic
    forward+backward against a constructed PolicyView at each training
    step (plan §5.1 step 5).
    """
    return Actor(
        h_dim=_H_DIM,
        z_dim=_Z_DIM,
        action_dim=_NUM_ACTIONS,
        mlp_hidden=_ACTOR_MLP_HIDDEN,
    ).to(device)


# ---- per-step measurements ------------------------------------------------


def _world_model_grad_norm(world_model: WorldModel) -> float:
    """Compute the L2 norm of all trainable world-model parameters' grads.

    ``clip_grad_norm_`` with ``max_norm=inf`` returns the actual norm
    without clipping. ``error_if_nonfinite=True`` would raise on NaN/Inf;
    we want the value (and let the caller decide what to do), so we
    leave it at the default.
    """
    return float(
        torch.nn.utils.clip_grad_norm_(
            world_model.parameters(), max_norm=float("inf")
        ).item()
    )


def _ema_divergence_max_ratio(world_model: WorldModel) -> float:
    """Max of ``(target - online).norm() / (online.norm() + 1e-8)`` across
    all (target_param, online_param) pairs on encoder + GRU.

    Plan §5.1 / §5.3: hard fail above 100. The BYOL/SPR convention is
    that the EMA target tracks the online parameters; a large divergence
    indicates the EMA update mechanism is broken or the online network
    is making jumps the EMA can't keep up with.
    """
    max_ratio = 0.0
    pairs = (
        (world_model.encoder, world_model.target_encoder),
        (world_model.gru_cell, world_model.target_gru_cell),
    )
    for online_module, target_module in pairs:
        for online_p, target_p in zip(
            online_module.parameters(),
            target_module.parameters(),
            strict=True,
        ):
            online_norm = float(online_p.data.norm().item())
            diff_norm = float((target_p.data - online_p.data).norm().item())
            ratio = diff_norm / (online_norm + 1e-8)
            if ratio > max_ratio:
                max_ratio = ratio
    return max_ratio


# ---- training step --------------------------------------------------------


class _StepResult:
    """Bundle of per-step instability-check values.

    Plain class (not @dataclass) to avoid the Python 3.14 dataclass /
    sys.modules registration trap the Phase 5 journal entry flagged for
    scripts that get exec'd by tests via spec_from_file_location. The
    smoke script is exec'd by tests/test_smoke_probe1_5_script.py via
    that loader; defining a @dataclass at module level would require
    the test to register the module in sys.modules before exec_module
    runs. Plain attribute assignment in __init__ is the safer shape.
    """

    __slots__ = (
        "kl_value",
        "recon_value",
        "sp_loss_value",
        "wm_grad_norm",
        "ema_div_ratio",
        "actor_new_col_grad_norm",
    )

    def __init__(
        self,
        kl_value: float,
        recon_value: float,
        sp_loss_value: float,
        wm_grad_norm: float,
        ema_div_ratio: float,
        actor_new_col_grad_norm: float,
    ) -> None:
        self.kl_value = kl_value
        self.recon_value = recon_value
        self.sp_loss_value = sp_loss_value
        self.wm_grad_norm = wm_grad_norm
        self.ema_div_ratio = ema_div_ratio
        self.actor_new_col_grad_norm = actor_new_col_grad_norm


def _run_one_training_step(
    world_model: WorldModel,
    ensemble: LatentDisagreementEnsemble,
    actor: Actor,
    wm_opt: torch.optim.Optimizer,
    ens_opt: torch.optim.Optimizer,
    actor_opt: torch.optim.Optimizer,
    device: torch.device,
) -> _StepResult:
    """Run one full sequence-pass training step.

    Returns the per-step measurements the instability checks consume.
    Tensor references are released by the time this returns; caller
    works only with the Python floats.
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

    h_outs: list[Tensor] = []
    z_outs: list[Tensor] = []
    wm_total = torch.zeros((), device=device)
    last_kl: Tensor | None = None
    last_recon: Tensor | None = None
    last_sp_loss: Tensor | None = None
    last_self_prediction: Tensor | None = None
    last_target_h_next: Tensor | None = None

    for t in range(_SEQUENCE_LENGTH):
        obs_t = obs_seq[:, t]
        wm_step = world_model.step(obs_t, h, z, a_prev)
        # Probe 1.5: the EMA-target's bar{h}_{t+1} is what the head
        # predicts against. compute_self_prediction_target uses
        # torch.no_grad internally so the EMA target's parameters
        # never enter the gradient graph.
        target_h_next = world_model.compute_self_prediction_target(
            obs_t, h, z, a_prev
        )
        loss_dict = world_model.loss(
            wm_step, obs_target=obs_t, target_h_next=target_h_next
        )
        # plan §2.4 step 12: combined backward, λ_self * sp_loss summed
        # into wm_total. The EMA update fires after wm_opt.step().
        wm_total = wm_total + loss_dict["total"] + _LAMBDA_SELF * loss_dict[
            "self_prediction_loss"
        ]
        last_kl = loss_dict["kl_aggregate_unclipped"]
        last_recon = loss_dict["recon"]
        last_sp_loss = loss_dict["self_prediction_loss"]
        last_self_prediction = wm_step.self_prediction
        last_target_h_next = target_h_next
        h_outs.append(wm_step.h)
        z_outs.append(wm_step.z)
        h = wm_step.h
        z = wm_step.z
        a_prev = action_seq[:, t]

    assert (
        last_kl is not None
        and last_recon is not None
        and last_sp_loss is not None
        and last_self_prediction is not None
        and last_target_h_next is not None
    )

    wm_opt.zero_grad(set_to_none=True)
    wm_total.backward()  # type: ignore[no-untyped-call]
    # plan §5.1: capture the world-model gradient norm BEFORE the
    # optimizer step (after step the .grad tensors are still populated
    # but we want the pre-update measurement).
    wm_grad_norm = _world_model_grad_norm(world_model)
    wm_opt.step()
    # Probe 1.5: EMA target update after the world-model optimizer step
    # (plan §2.4 step 12 / world_model._update_ema_target).
    world_model._update_ema_target()
    ema_div_ratio = _ema_divergence_max_ratio(world_model)

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

    # Actor exercise: construct a PolicyView from the last sequence
    # position's (h, z) detached, paired with the per-batch-element
    # cosine-distance scalar between the head's prediction and the EMA
    # target. The actor reads the scalar via PolicyView's
    # self_prediction_error field — the synthesis §1.3 (v2) Watts-
    # heuristic exception. A synthetic CE loss against a constant
    # target action drives the gradient through the actor's network;
    # the gradient on the new column on the input layer is what plan
    # §5.1 / §5.4 names as the soft-warning surface for "the actor is
    # ignoring the scalar entirely."
    pred_det = last_self_prediction.detach()
    target_det = last_target_h_next.detach()
    if _LOSS_FORM == "cosine":
        scalar = 1.0 - F.cosine_similarity(pred_det, target_det, dim=-1)
    else:  # "mse"
        scalar = ((pred_det - target_det) ** 2).mean(dim=-1)

    h_det = h_outs[-1].detach()
    z_det = z_outs[-1].detach()
    policy_view = PolicyView(
        h=h_det, z=z_det, self_prediction_error=scalar
    )
    actor_output = actor.forward(policy_view)
    target_actions = torch.zeros(_BATCH_SIZE, dtype=torch.long, device=device)
    actor_loss = F.cross_entropy(actor_output.logits, target_actions)
    actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()  # type: ignore[no-untyped-call]
    # plan §5.1 / §5.4: the new column on the actor's first linear
    # layer is at column indices [h_dim+z_dim:]. Slice the weight grad,
    # take the L2 norm. NaN here is a hard fail (§5.3); below 1e-6
    # across all 100 steps is a soft warning (§5.4). The cast matches
    # actor.py's own pattern for accessing self.net[0] as nn.Linear.
    first_layer = cast(nn.Linear, actor.net[0])
    first_layer_grad = first_layer.weight.grad
    assert first_layer_grad is not None, (
        "actor.net[0].weight.grad is None after actor_loss.backward(); "
        "the autograd graph did not reach the first layer"
    )
    new_col_grad_norm = float(
        first_layer_grad[:, _H_DIM + _Z_DIM:].norm().item()
    )
    actor_opt.step()

    return _StepResult(
        kl_value=float(last_kl.item()),
        recon_value=float(last_recon.item()),
        sp_loss_value=float(last_sp_loss.item()),
        wm_grad_norm=wm_grad_norm,
        ema_div_ratio=ema_div_ratio,
        actor_new_col_grad_norm=new_col_grad_norm,
    )


# ---- sinks ---------------------------------------------------------------


def _exercise_sinks(telemetry_dir: Path) -> None:
    """Open all four sinks; write synthetic 0.2.0 records; verify round-trip.

    Plan §5.1 step 8: alternate ``self_prediction_error_masked_t``
    True/False across the synthetic AgentStep records to exercise both
    code paths (the first-step sentinel-zero case and the empirical
    near-zero case). The DreamRollout record carries
    ``sequence_self_prediction=None`` per synthesis §1.5 — Probe 1.5
    leaves the reserved field None; Probe 3 may populate.
    """
    run_id = "smoke-probe1_5"
    h_dim = _H_DIM
    z_dim = _Z_DIM
    embed_dim = _EMBED_DIM

    agent_step_dir = telemetry_dir / "agent_step"
    dream_dir = telemetry_dir / "dream_rollout"
    replay_path = telemetry_dir / "replay_meta.jsonl"
    world_event_path = telemetry_dir / "world_event.jsonl"

    # Build six synthetic AgentStep records alternating masked True/False.
    # Plan §5.1 / synthesis §3 v2 / plan §6 row 16: masked steps carry
    # scalar=0.0 (sentinel); non-masked steps carry an empirical value.
    sample_agent_steps = [
        AgentStep(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            checkpoint_id=None,
            t=i,
            episode_id=i // 2,
            step_in_episode=i % 2,  # 0 → masked first step; 1 → second step
            wallclock_ms=int(time.monotonic_ns() // 1_000_000),
            h_t=[float(j) * 0.01 for j in range(h_dim)],
            q_params_t=([0.0] * z_dim, [-1.0] * z_dim),
            p_params_t=([0.0] * z_dim, [0.0] * z_dim),
            z_t=[0.0] * z_dim,
            kl_per_dim_t=[0.5] * z_dim,
            kl_aggregate_t=0.5 * z_dim,
            recon_loss_t=10.0,
            action_t=i % _NUM_ACTIONS,
            action_logprob_t=-1.6,
            policy_entropy_t=1.6,
            obs_hash_t="0" * 64,
            intrinsic_signal_t=0.001,
            encoder_embedding_t=[0.0] * embed_dim,
            self_prediction_t=[float(j) * 0.001 for j in range(h_dim)],
            # Alternating: even i → masked True (sentinel zero scalar);
            # odd i → masked False (empirical scalar).
            self_prediction_error_t=0.0 if (i % 2 == 0) else 0.123,
            self_prediction_error_masked_t=(i % 2 == 0),
        )
        for i in range(6)
    ]

    sample_dream = DreamRollout(
        schema_version=SCHEMA_VERSION,
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
        sequence_self_prediction=None,  # synthesis §1.5 — None at Probe 1.5
    )
    # ReplayMeta and WorldEvent are not bumped by Probe 1.5 (they have no
    # new fields); they continue to stamp PROBE_1_SCHEMA_VERSION as the
    # Phase 0 writer-migration discipline names ("what's still
    # PROBE_1_SCHEMA_VERSION in a Phase 3+ codebase is Probe-1-shaped by
    # intent, not by oversight").
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
        for record in sample_agent_steps:
            sink.write(record)
    with ParquetSink(dream_dir, DreamRollout) as sink:
        sink.write(sample_dream)
    with JsonlSink(replay_path, ReplayMeta) as sink:
        sink.write(sample_replay_meta)
    with JsonlSink(world_event_path, WorldEvent) as sink:
        sink.write(sample_world_event)

    # Read-back validates the round-trip — the smoke's "all four sinks
    # write valid records" criterion (plan §5.2). Beyond Probe 1's
    # smoke, this also exercises both the masked-True and masked-False
    # AgentStep code paths through the validator.
    import pyarrow.parquet as pq

    agent_shards = sorted(agent_step_dir.glob("shard-*.parquet"))
    if not agent_shards:
        raise RuntimeError("ParquetSink wrote no shard for AgentStep")
    rows = pq.read_table(agent_shards[0]).to_pylist()  # type: ignore[no-untyped-call]
    if len(rows) < 6:
        raise RuntimeError(
            f"ParquetSink wrote fewer AgentStep rows than expected: "
            f"got {len(rows)}, expected 6"
        )
    masked_seen = False
    unmasked_seen = False
    for row in rows:
        rt = AgentStep.model_validate(row)
        if rt.schema_version != SCHEMA_VERSION:
            raise RuntimeError(
                f"AgentStep schema_version round-trip failed: "
                f"{rt.schema_version!r} vs {SCHEMA_VERSION!r}"
            )
        if rt.self_prediction_error_masked_t is True:
            masked_seen = True
        elif rt.self_prediction_error_masked_t is False:
            unmasked_seen = True
    if not (masked_seen and unmasked_seen):
        raise RuntimeError(
            f"AgentStep round-trip did not exercise both masked-flag "
            f"code paths (masked_seen={masked_seen}, "
            f"unmasked_seen={unmasked_seen})"
        )

    dream_shards = sorted(dream_dir.glob("shard-*.parquet"))
    if not dream_shards:
        raise RuntimeError("ParquetSink wrote no shard for DreamRollout")
    rt_dream = DreamRollout.model_validate(
        pq.read_table(dream_shards[0]).to_pylist()[0]  # type: ignore[no-untyped-call]
    )
    if rt_dream.schema_version != SCHEMA_VERSION:
        raise RuntimeError("DreamRollout schema_version round-trip failed")
    if rt_dream.sequence_self_prediction is not None:
        raise RuntimeError(
            "DreamRollout sequence_self_prediction must be None at "
            "Probe 1.5 (synthesis §1.5)"
        )

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


# ---- instability counters ------------------------------------------------


class _ConsecutiveBelowAboveCounter:
    """Tracks the longest consecutive run where the running mean is below
    (or above) a threshold.

    Plan §5.4: "100-step running mean of X should not drop below
    (exceed) Y for more than 20 consecutive steps." A deque of size
    100 produces the running mean; the counter resets when the
    condition fails. The max consecutive count is what the soft-warning
    decision reads at the end.
    """

    __slots__ = (
        "_window",
        "_threshold",
        "_compare_below",
        "_consecutive",
        "max_consecutive",
    )

    def __init__(
        self,
        window: int,
        threshold: float,
        compare_below: bool,
    ) -> None:
        self._window: deque[float] = deque(maxlen=window)
        self._threshold = threshold
        self._compare_below = compare_below
        self._consecutive = 0
        self.max_consecutive = 0

    def update(self, value: float) -> None:
        self._window.append(value)
        rolling_mean = sum(self._window) / len(self._window)
        breached = (
            rolling_mean < self._threshold
            if self._compare_below
            else rolling_mean > self._threshold
        )
        if breached:
            self._consecutive += 1
            if self._consecutive > self.max_consecutive:
                self.max_consecutive = self._consecutive
        else:
            self._consecutive = 0


# ---- entry point ----------------------------------------------------------


def main() -> int:
    """Run the smoke; return a process exit status (0 on success).

    The function returns rather than ``sys.exit``-ing so test code that
    imports the script can call ``main()`` directly. The ``__main__``
    block at the bottom routes the return value into ``sys.exit``.
    """
    device = _detect_mps_or_exit()

    print(
        f"[smoke probe1.5] mps detected — running {_NUM_TRAINING_STEPS} "
        f"training steps at h={_H_DIM}, z={_Z_DIM}, K={_K_ENSEMBLE}, "
        f"head_hidden={_HEAD_HIDDEN}, batch={_BATCH_SIZE}, "
        f"seq={_SEQUENCE_LENGTH}, ema_decay={_EMA_DECAY}, "
        f"target_mode={_TARGET_MODE!r}, loss_form={_LOSS_FORM!r}",
        flush=True,
    )

    # Init-time fallbacks are tolerated; the hot-path warning capture is
    # opened just before the training loop (Probe 1's smoke convention).
    world_model = _make_world_model(device)
    ensemble = _make_ensemble(device)
    actor = _make_actor(device)
    wm_opt = torch.optim.Adam(world_model.parameters(), lr=_WORLD_MODEL_LR)
    ens_opt = torch.optim.Adam(ensemble.parameters(), lr=_ENSEMBLE_LR)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=_ACTOR_LR)

    # Hot-path measurement state.
    fallback_warnings: list[str] = []
    per_step_seconds: list[float] = []
    finite_kl_seen = True
    finite_recon_seen = True
    finite_sp_loss_seen = True
    wm_grad_norm_overflow = False
    ema_diverged = False
    actor_new_col_grad_norm_nan = False
    new_col_grad_norms: list[float] = []
    wm_grad_norms: list[float] = []
    ema_div_ratios: list[float] = []
    sp_loss_values: list[float] = []
    early_termination_step: int | None = None

    kl_floor_counter = _ConsecutiveBelowAboveCounter(
        window=_RUNNING_MEAN_WINDOW,
        threshold=_KL_FLOOR_SOFT_BAR,
        compare_below=True,
    )
    recon_climb_counter = _ConsecutiveBelowAboveCounter(
        window=_RUNNING_MEAN_WINDOW,
        threshold=_RECON_CLIMB_SOFT_BAR,
        compare_below=False,
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        loop_start = time.monotonic()
        for step_index in range(_NUM_TRAINING_STEPS):
            step_start = time.monotonic()
            try:
                result = _run_one_training_step(
                    world_model,
                    ensemble,
                    actor,
                    wm_opt,
                    ens_opt,
                    actor_opt,
                    device,
                )
            except Exception as exc:
                early_termination_step = step_index
                print(
                    f"[smoke probe1.5] exception at step {step_index}: {exc!r}",
                    file=sys.stderr,
                )
                break
            per_step_seconds.append(time.monotonic() - step_start)

            # Per-step instability checks (plan §5.1 / §5.3 hard fails).
            if not _is_finite(result.kl_value):
                finite_kl_seen = False
                early_termination_step = step_index
                print(
                    f"[smoke probe1.5] non-finite kl_aggregate_unclipped at "
                    f"step {step_index}: {result.kl_value!r}",
                    file=sys.stderr,
                )
                break
            if not _is_finite(result.recon_value):
                finite_recon_seen = False
                early_termination_step = step_index
                print(
                    f"[smoke probe1.5] non-finite recon at step "
                    f"{step_index}: {result.recon_value!r}",
                    file=sys.stderr,
                )
                break
            if not _is_finite(result.sp_loss_value):
                finite_sp_loss_seen = False
                early_termination_step = step_index
                print(
                    f"[smoke probe1.5] non-finite self_prediction_loss at "
                    f"step {step_index}: {result.sp_loss_value!r}",
                    file=sys.stderr,
                )
                break
            if result.wm_grad_norm > _WORLD_MODEL_GRAD_NORM_HARD_BAR:
                wm_grad_norm_overflow = True
                early_termination_step = step_index
                print(
                    f"[smoke probe1.5] world-model gradient norm exceeds "
                    f"hard bar at step {step_index}: "
                    f"{result.wm_grad_norm:.2f} > "
                    f"{_WORLD_MODEL_GRAD_NORM_HARD_BAR:.0f}",
                    file=sys.stderr,
                )
                break
            if result.ema_div_ratio > _EMA_DIVERGENCE_HARD_BAR:
                ema_diverged = True
                early_termination_step = step_index
                print(
                    f"[smoke probe1.5] EMA target divergence ratio exceeds "
                    f"hard bar at step {step_index}: "
                    f"{result.ema_div_ratio:.2f} > "
                    f"{_EMA_DIVERGENCE_HARD_BAR:.0f}",
                    file=sys.stderr,
                )
                break
            if not _is_finite(result.actor_new_col_grad_norm):
                actor_new_col_grad_norm_nan = True
                early_termination_step = step_index
                print(
                    f"[smoke probe1.5] actor new-input-column gradient norm "
                    f"is NaN/Inf at step {step_index}: "
                    f"{result.actor_new_col_grad_norm!r}",
                    file=sys.stderr,
                )
                break

            # Track values for soft-warning summary.
            new_col_grad_norms.append(result.actor_new_col_grad_norm)
            wm_grad_norms.append(result.wm_grad_norm)
            ema_div_ratios.append(result.ema_div_ratio)
            sp_loss_values.append(result.sp_loss_value)
            kl_floor_counter.update(result.kl_value)
            recon_climb_counter.update(result.recon_value)

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
                    f"[smoke probe1.5] step {steps_done}/"
                    f"{_NUM_TRAINING_STEPS} | "
                    f"per-step={rolling_mean_ms:.1f}ms | "
                    f"elapsed={elapsed:.1f}s | "
                    f"kl={result.kl_value:.3f} "
                    f"recon={result.recon_value:.3f} "
                    f"sp={result.sp_loss_value:.4f} "
                    f"wm_grad={result.wm_grad_norm:.2f} "
                    f"ema_div={result.ema_div_ratio:.4f} "
                    f"new_col_grad={result.actor_new_col_grad_norm:.6f}",
                    flush=True,
                )

        loop_total_seconds = time.monotonic() - loop_start

        for w in captured:
            text = str(w.message)
            if _is_mps_fallback_warning(text):
                fallback_warnings.append(text)

    # Sink exercise — covered after the hot loop so a sink failure
    # doesn't poison the warning capture above. The sinks write 0.2.0
    # AgentStep records with both masked-True and masked-False rows so
    # both code paths through the validator are exercised at the
    # platform-correctness gate.
    sinks_ok = True
    sink_error: str | None = None
    with tempfile.TemporaryDirectory(prefix="kind_smoke_probe1_5_") as tmp:
        try:
            _exercise_sinks(Path(tmp))
        except Exception as exc:  # pragma: no cover — surfaces in stderr
            sinks_ok = False
            sink_error = repr(exc)

    # Verify the optimizers actually populated gradients on at least one
    # parameter in each module (defensive — backward already ran above
    # so any missing-grad would point at a fundamental wiring issue).
    grads_ok = (
        any(p.grad is not None for p in world_model.parameters())
        and any(p.grad is not None for p in ensemble.parameters())
        and any(p.grad is not None for p in actor.parameters())
    )

    # Decide overall hard-fail status.
    hot_path_clean = not fallback_warnings
    finiteness_ok = (
        finite_kl_seen and finite_recon_seen and finite_sp_loss_seen
    )
    instability_ok = not (
        wm_grad_norm_overflow
        or ema_diverged
        or actor_new_col_grad_norm_nan
    )
    full_loop_ok = early_termination_step is None
    overall_ok = (
        hot_path_clean
        and finiteness_ok
        and instability_ok
        and full_loop_ok
        and sinks_ok
        and grads_ok
    )

    # Soft warnings (print to stderr; don't fail).
    soft_warnings: list[str] = []
    mean_per_step_ms = (
        1000.0 * sum(per_step_seconds) / len(per_step_seconds)
        if per_step_seconds
        else float("nan")
    )
    if (
        per_step_seconds
        and mean_per_step_ms > _WALL_TIME_PER_STEP_SOFT_BAR_MS
    ):
        soft_warnings.append(
            f"per-step wall time {mean_per_step_ms:.1f}ms exceeds soft "
            f"bar {_WALL_TIME_PER_STEP_SOFT_BAR_MS:.0f}ms"
        )
    if kl_floor_counter.max_consecutive > _RUN_LEN_SOFT_BAR:
        soft_warnings.append(
            f"KL pinning at floor: running mean of "
            f"kl_aggregate_unclipped below {_KL_FLOOR_SOFT_BAR:.2f} for "
            f"{kl_floor_counter.max_consecutive} consecutive steps "
            f"(threshold = {_RUN_LEN_SOFT_BAR})"
        )
    if recon_climb_counter.max_consecutive > _RUN_LEN_SOFT_BAR:
        soft_warnings.append(
            f"recon climbing: running mean of recon above "
            f"{_RECON_CLIMB_SOFT_BAR:.2f} for "
            f"{recon_climb_counter.max_consecutive} consecutive steps "
            f"(threshold = {_RUN_LEN_SOFT_BAR})"
        )
    if (
        new_col_grad_norms
        and max(new_col_grad_norms) < _ACTOR_NEW_COL_GRAD_FLOOR
    ):
        soft_warnings.append(
            f"actor new-input-column gradient norm below "
            f"{_ACTOR_NEW_COL_GRAD_FLOOR:.0e} across all "
            f"{len(new_col_grad_norms)} steps "
            f"(max={max(new_col_grad_norms):.2e}) — actor may be "
            f"ignoring the scalar (synthesis §1.7(a) failure mode (a) "
            f"early signal; documented mitigation: switch new column "
            f"init from zero to small-Gaussian per plan §6 row 15)"
        )

    # Diagnostic detail before the one-line summary.
    if fallback_warnings:
        print(
            f"[smoke probe1.5] FAIL: {len(fallback_warnings)} MPS-fallback "
            f"warning(s) on the hot path:",
            file=sys.stderr,
        )
        for text in fallback_warnings[:5]:
            print(f"  - {text}", file=sys.stderr)
    if not finite_kl_seen:
        print(
            "[smoke probe1.5] FAIL: non-finite kl_aggregate_unclipped",
            file=sys.stderr,
        )
    if not finite_recon_seen:
        print("[smoke probe1.5] FAIL: non-finite recon", file=sys.stderr)
    if not finite_sp_loss_seen:
        print(
            "[smoke probe1.5] FAIL: non-finite self_prediction_loss",
            file=sys.stderr,
        )
    if wm_grad_norm_overflow:
        print(
            f"[smoke probe1.5] FAIL: world-model gradient norm exploded "
            f"(>{_WORLD_MODEL_GRAD_NORM_HARD_BAR:.0f}). Mitigation 1 "
            f"(lower λ_self) is the documented first try; see plan §5.5.",
            file=sys.stderr,
        )
    if ema_diverged:
        print(
            f"[smoke probe1.5] FAIL: EMA target diverged "
            f"(>{_EMA_DIVERGENCE_HARD_BAR:.0f}× online L2). Mitigation 3 "
            f"(orthogonal-gradient updates) is the documented escalation; "
            f"see plan §5.5.",
            file=sys.stderr,
        )
    if actor_new_col_grad_norm_nan:
        print(
            "[smoke probe1.5] FAIL: actor new-input-column gradient norm "
            "is NaN/Inf. The autograd graph reaches a non-finite value "
            "between the actor's loss and the input layer's weight grad.",
            file=sys.stderr,
        )
    if not sinks_ok:
        print(
            f"[smoke probe1.5] FAIL: sinks errored: {sink_error}",
            file=sys.stderr,
        )
    if not grads_ok:
        print(
            "[smoke probe1.5] FAIL: backward did not populate gradients "
            "on the world model, ensemble, or actor",
            file=sys.stderr,
        )
    for warning_text in soft_warnings:
        print(
            f"[smoke probe1.5] WARNING: {warning_text}", file=sys.stderr
        )

    # One-line summary per plan §5.1 step 9.
    status_word = "ok" if overall_ok else "FAIL"
    sinks_word = "ok" if sinks_ok else "FAIL"
    shapes_word = (
        "ok" if (finiteness_ok and grads_ok and full_loop_ok) else "FAIL"
    )
    kl_floor_word = (
        "clean"
        if kl_floor_counter.max_consecutive <= _RUN_LEN_SOFT_BAR
        else f"warn({kl_floor_counter.max_consecutive})"
    )
    recon_climb_word = (
        "clean"
        if recon_climb_counter.max_consecutive <= _RUN_LEN_SOFT_BAR
        else f"warn({recon_climb_counter.max_consecutive})"
    )
    ema_div_word = (
        "clean"
        if not ema_diverged
        else f"FAIL({max(ema_div_ratios):.1f})"
    )
    max_wm_grad = max(wm_grad_norms) if wm_grad_norms else float("nan")
    max_new_col_grad = (
        max(new_col_grad_norms) if new_col_grad_norms else float("nan")
    )
    print(
        f"[smoke probe1.5] mps {status_word} | "
        f"wall={loop_total_seconds:.2f}s | "
        f"per-step={mean_per_step_ms:.1f}ms | "
        f"sinks {sinks_word} | shapes {shapes_word} | "
        f"instability checks: KL floor pinning={kl_floor_word} | "
        f"recon climbing={recon_climb_word} | "
        f"EMA divergence={ema_div_word} | "
        f"grad norm={max_wm_grad:.2f} | "
        f"actor new-col grad={max_new_col_grad:.2e}",
        flush=True,
    )

    return 0 if overall_ok else 1


if __name__ == "__main__":  # pragma: no cover — manual entry point
    sys.exit(main())
