"""Phase 5 runner — the integration loop.

Ties together the substrate components Phases 1–4 built: the transport
client drives the env-server; the world model encodes observations and
exposes the substrate conduit; the actor decides actions; the disagreement
ensemble produces the intrinsic signal; transitions accumulate in the
:class:`~kind.training.replay.SequenceReplayBuffer`; training samples drive
three independent losses; telemetry emits through all four sinks; the
checkpoint manager commits periodically.

**One process, one loop.** Probe 1 runs the runner on the Mac with the
env-server reachable via loopback. The runner is single-threaded; the
transport client's reader thread is the only auxiliary thread, and it
delivers ``WorldEvent`` records into the runner's JSONL sink
synchronously.

**Three optimizers, three losses.** The world model's optimizer steps on
the ELBO over the sampled sequence. The ensemble's optimizer steps on
per-head MSE against the world model's posterior at the next step (with
detached targets so the world model receives no gradient from this
path). The actor's optimizer steps on the actor's analytic-gradient
imagination loss; ``imagine_and_compute_loss`` freezes world-model and
ensemble parameters during its forward, so the gradient flows only into
the actor's parameters. The three :meth:`backward` calls are independent;
the three :meth:`step` calls update three disjoint parameter sets.

**The four-state operational model.** Probe 1 only exercises *waking*
(the design notes' four states: waking / dreaming / dormant / paused).
The runner's structure does not foreclose the others — Phase 3 will plug
dreaming into the same conduits — but neither does it pre-build them.
Dream rollouts at Probe 1 are *telemetry-only* periodic emissions; they
do not drive behaviour.

**Self-opacity boundary.** The actor reads ``PolicyView`` only — that
contract is enforced by Phase 3c's import-level lint, the type signature
on :meth:`Actor.forward`, and the frozen ``PolicyView`` dataclass. The
runner respects the boundary structurally: it routes
``WorldModelStep`` through :func:`views.split` and only ever passes the
resulting ``PolicyView`` to the actor's forward.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pickle
import random
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from safetensors.torch import load_file, save_file
from torch import Tensor

from kind.agents.actor import ActionOutput, Actor
from kind.agents.ensemble import LatentDisagreementEnsemble
from kind.agents.preference import EnergyPreferenceConfig
from kind.agents.views import PolicyView, split
from kind.agents.world_model import WorldModel, WorldModelConfig, WorldModelStep
from kind.env.env_server import EnvServer
from kind.env.grid_world import EnvStep
from kind.env.perturbation_generator import (
    PerturbationGenerator,
    PerturbationGeneratorConfig,
)
from kind.env.transport import EnvTransportClient
from kind.env.trigger_inbox import drain_trigger_inbox
from kind.observer.source_events import CONSUMPTION_JUMP_THRESHOLD
from kind.observer.schemas import (
    PROBE_1_SCHEMA_VERSION,
    PROBE_3_5_PHASE2_TELEMETRY_SCHEMA_VERSION,
    PROBE_3_5_TELEMETRY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    AgentStep,
    DreamRollout,
    ReplayMeta,
    WorldEvent,
)
from kind.observer.dream_session import DreamSessionSink
from kind.observer.maintenance_refit import (
    MaintenanceRefitConfig,
    run_scheduled_maintenance,
)
from kind.observer.pre_reg import PreRegistration, PreRegSink
from kind.observer.sinks import JsonlSink, ParquetSink
from kind.training.checkpoint import CheckpointContents, CheckpointManager
from kind.training.dream import DreamRolloutConfig
from kind.training.dream_seed import SeedSelectionConfig
from kind.training.protection import DreamProtectionPolicy, MetabolicBudget
from kind.training.replay import (
    Batch,
    SequenceReplayBuffer,
    Transition,
)
from kind.training.state_machine import (
    AlwaysAwakeDesktop,
    DesktopSignalSource,
    DreamEnvelopeConfig,
    HostSignals,
    RunDreamSessionDriver,
    State,
    StateController,
)

__all__ = ["Runner", "RunnerConfig", "RunnerStepInfo", "StepCallback"]


# ---- per-iteration callback -----------------------------------------------


@dataclass(frozen=True)
class RunnerStepInfo:
    """Snapshot of one ``Runner._step_once`` iteration.

    Passed to the optional :data:`StepCallback` the runner invokes at the
    end of every iteration. All fields reflect post-step state — the
    :class:`AgentStep` that was just emitted, plus flags indicating
    whether a dream rollout or checkpoint commit fired during this
    iteration. The flags let an external observer (a progress bar, a
    logger, a future in-loop mirror) react to the runner's lifecycle
    events without having to subclass.

    Field semantics:

    * ``env_step`` / ``episode_id`` / ``step_in_episode`` — the indexing
      fields the matching :class:`AgentStep` carries. ``env_step`` is
      the *FROM*-side step (the one the action was taken at), to match
      the AgentStep schema's ``t`` field.
    * ``kl_aggregate`` / ``recon_loss`` / ``intrinsic_signal`` /
      ``policy_entropy`` — scalar telemetry values from the AgentStep.
    * ``dream_emitted`` — true iff this iteration fired a dream rollout.
    * ``checkpoint_committed`` — true iff this iteration fired a
      checkpoint commit (post-commit, the new ``checkpoint_id`` is
      already in ``checkpoint_id``).
    * ``checkpoint_id`` — current checkpoint id (most recently committed
      one, or ``None`` if no commit has happened yet).
    """

    env_step: int
    episode_id: int
    step_in_episode: int
    kl_aggregate: float
    recon_loss: float
    intrinsic_signal: float
    policy_entropy: float
    dream_emitted: bool
    checkpoint_committed: bool
    checkpoint_id: str | None


StepCallback = Callable[[RunnerStepInfo], None]


# ---- canonical telemetry layout -----------------------------------------


_AGENT_STEP_DIR: Final[str] = "agent_step"
_DREAM_ROLLOUT_DIR: Final[str] = "dream_rollout"
_REPLAY_META_FILE: Final[str] = "replay_meta.jsonl"
_WORLD_EVENT_FILE: Final[str] = "world_event.jsonl"


# ---- config -------------------------------------------------------------


@dataclass(frozen=True)
class RunnerConfig:
    """Configuration for one :class:`Runner` instance.

    Required: ``world_model_config``, ``run_id``, ``telemetry_dir``,
    ``checkpoints_dir``. Everything else has a Probe 1 default per the
    implementation plan §6.

    Optimizer learning rates default to DreamerV1-ish values:
    ``world_model_lr=1e-4``, ``actor_lr=4e-5``, ``ensemble_lr=4e-5``.
    These are starting points; the smoke test (Phase 8) is the first
    revision opportunity.

    The ``warmup_env_steps`` default of 1000 is chosen so the buffer has
    several episode-aligned windows before training begins; with
    ``replay_sequence_length=32`` and ``episode_length=200``, a 1000-step
    warmup gives ~5 episodes worth of transitions, which is more than
    enough valid windows for a 16-batch sample.
    """

    world_model_config: WorldModelConfig
    run_id: str
    telemetry_dir: Path
    checkpoints_dir: Path

    # Actor / ensemble.
    action_dim: int = 5
    ensemble_k: int = 5
    imagination_horizon: int = 15

    # Replay.
    replay_capacity: int = 100_000
    replay_sequence_length: int = 32
    replay_batch_size: int = 16

    # Training cadence.
    train_every_n_env_steps: int = 1
    warmup_env_steps: int = 1000

    # Dream rollouts (telemetry-only at Probe 1).
    dream_cadence_env_steps: int = 1000
    dream_horizon: int = 15

    # Checkpoints.
    checkpoint_every_n_env_steps: int = 10_000

    # Optimizer learning rates.
    world_model_lr: float = 1e-4
    actor_lr: float = 4e-5
    ensemble_lr: float = 4e-5

    # Device.
    device: str = "cpu"

    # Telemetry shard size for Parquet sinks.
    parquet_rows_per_shard: int = 10_000

    # Probe 1.5 self-prediction (synthesis §6 / plan §6 defaults; plan
    # §2.4 RunnerConfig additions). The runner uses ``lambda_self``
    # directly to scale the auxiliary loss into the world-model
    # backward; the other three are runner-authoritative copies of the
    # corresponding ``WorldModelConfig`` fields and are written into the
    # constructed ``WorldModel`` via ``dataclasses.replace`` at runner
    # init so RunnerConfig is the single source of truth.
    lambda_self: float = 0.1
    ema_decay: float = 0.99

    # Probe 3.5 (implementation plan §S-TEL). When True, AgentStep records are
    # stamped at ``PROBE_3_5_TELEMETRY_SCHEMA_VERSION`` ("0.4.0") and carry the
    # four energy fields (sensed/true/pred/recon-error); requires a co-located
    # ``env_server`` so ``true_energy`` is reachable. When False (the default),
    # records stay at the legacy ``SCHEMA_VERSION`` ("0.2.0") with the energy
    # fields None — byte-identical to pre-3.5 telemetry, so every existing
    # runner / smoke is unchanged. The Phase-1 baseline run sets this True. The
    # ``sensed_energy`` scalar is fused into the world model on every step
    # regardless of this flag — it governs telemetry emission, not the substrate.
    energy_telemetry: bool = False

    # Probe 3.5 Phase 2 (implementation plan §S-ACT). When non-None, the
    # saturating Gaussian log-preference over decoded energy is passed to the
    # **waking** actor-training call (``imagine_and_compute_loss``) — the
    # scaffold filled. ``None`` (the default) preserves the Phase-1 zero
    # scaffold exactly, so every existing runner / smoke is unchanged. With
    # both this and ``energy_telemetry`` set, AgentStep records are stamped at
    # ``PROBE_3_5_PHASE2_TELEMETRY_SCHEMA_VERSION`` ("0.5.0") and carry the
    # per-training-step pragmatic/epistemic decomposition fields; before the
    # first training step the decomposition is genuinely (0.0, 0.0, 0.0).
    # Precision is the dominance-relevant weight (no β anywhere); the dream
    # path has no route to this config (tests/test_pragmatic_guards.py).
    energy_preference: EnergyPreferenceConfig | None = None

    # Probe 4 Phase 2 (implementation plan §S-PERT). When non-None, a seeded
    # stochastic perturbation generator is polled at every step boundary
    # (inside ``_step_once``, between the env step and the AgentStep emit) and
    # its planned cells are fired through the co-located env-server's mutator
    # surface with ``trigger="generator"`` — deterministic step-boundary
    # placement relative to the logged transitions. ``None`` (the default)
    # changes nothing — every existing runner is byte-identical. Requires a
    # co-located ``env_server`` (placement reads GridState; the not-self
    # exclusion needs Io's position), same pattern as ``energy_telemetry``.
    perturbation_generator: PerturbationGeneratorConfig | None = None
    # Probe 4 Phase 2: the manual-trigger spool directory (the builder's CLI
    # writes request files; the runner drains them at the same step boundary,
    # firing with ``trigger="manual"``). ``None`` (default) disables the
    # drain. Requires a co-located ``env_server`` when set.
    perturbation_inbox_dir: Path | None = None

    # Probe 4.5 Phase 1 (implementation plan §S-DEC; prereg §3). When
    # non-None, the scheduled decoder-head maintenance refit + honesty gate
    # (``kind.observer.maintenance_refit``) runs at the pre-registered
    # checkpoint-aligned cadence — at the step boundary, immediately BEFORE
    # the checkpoint commit, so every committed checkpoint carries the
    # refit head and a resume continues from post-refit state. The cadence
    # must be a multiple of ``checkpoint_every_n_env_steps`` (validated at
    # construction). A §3 honesty-STOP raises ``HonestyStopError`` out of
    # ``run()`` — the orchestrating script records the finding; the run does
    # not proceed on a lying belief. ``None`` (the default) changes nothing —
    # every existing runner is byte-identical.
    maintenance_refit: MaintenanceRefitConfig | None = None
    self_prediction_target_mode: Literal[
        "online", "frozen", "environmental"
    ] = "online"
    self_prediction_loss_form: Literal["cosine", "mse"] = "cosine"

    # Probe 2 v2 lesion plumbing (plan §2.5; synthesis §2.4 element 4).
    # The five named lesions target three reading surfaces:
    #
    # * ``"ensemble_k1"`` / ``"ensemble_constant"`` — substrate-side; v1
    #   candidates preserved here. ``"ensemble_k1"`` overrides
    #   ``ensemble_k`` to 1 at construction; ``"ensemble_constant"``
    #   passes ``lesion_constant_disagreement=True`` to the ensemble so
    #   ``disagreement`` returns zeros.
    # * ``"disable_self_prediction"`` — substrate-side; threaded into the
    #   constructed ``WorldModelConfig.lesion_kind`` so ``WorldModel.step``
    #   emits zeros from the head and ``_update_ema_target`` is a no-op.
    # * ``"zero_or_randomize_scalar"`` — behavior-side; threaded into
    #   ``views.split`` at every env-step so the actor's PolicyView's
    #   ``self_prediction_error`` is overridden. ``lesion_zero_or_randomize_
    #   variant`` selects the specific override; ``lesion_zero_or_
    #   randomize_empirical_min`` / ``..._max`` are the bounds for the
    #   randomize variant. The substrate-side reads exactly like the
    #   un-lesioned run.
    # * ``"init_zero_scalar_column"`` — capacity-as-init-shape; not a
    #   runtime flag but a checkpoint mutation produced by
    #   ``scripts/probe2_lesion_init_zero_scalar_column.py``. Setting this
    #   value on a ``RunnerConfig`` is purely informational at run start
    #   (it triggers the same ``mirror_marker`` emission convention the
    #   other kinds use); the actual column-zeroing happens before the
    #   runner loads the lesioned checkpoint.
    #
    # ``None`` (the default) is the un-lesioned path; downstream code
    # treats it as the no-op carrier.
    lesion_kind: Literal[
        None,
        "ensemble_k1",
        "ensemble_constant",
        "disable_self_prediction",
        "zero_or_randomize_scalar",
        "init_zero_scalar_column",
    ] = None
    lesion_zero_or_randomize_variant: Literal["zero", "randomize"] = "zero"
    # Empirical bounds for the ``"randomize"`` variant; the defaults cover
    # the cosine loss form's range. Real Probe 2 lesion runs journal the
    # bounds derived from the source run's empirical
    # ``self_prediction_error_t`` distribution.
    lesion_zero_or_randomize_empirical_min: float = 0.0
    lesion_zero_or_randomize_empirical_max: float = 1.0

    # Probe 2 v2 pre-registration sink (plan §2.5; synthesis §2.4 element 1).
    # When set, the runner constructs a :class:`PreRegSink` at this directory
    # and :meth:`Runner.emit_pre_registration` writes records to
    # ``<pre_reg_dir>/pre_reg.jsonl``. When ``None`` (the default), the
    # runner has no pre-reg sink, no directory is created, and
    # ``emit_pre_registration`` raises. The Probe 2 convention is
    # ``runs/{run_id}/pre_reg/`` as a sibling of ``telemetry_dir``;
    # adversarial-pass orchestration (Phase 8) sets this explicitly. Probe 1
    # and Probe 1.5 runners leave it ``None``.
    pre_reg_dir: Path | None = None

    # Probe 3 Phase 7 — the cross-probe surface (plan §7). The two
    # Probe-4-perturbable configs: the dream *envelope* (the when / how-long
    # of dreaming — caps, heartbeat, compute budget) and the *seed-selection*
    # (the from-what — seed mode, perturbation sigma, replay window). They are
    # exposed here as the typed control surface a Probe 4 builder perturbation
    # (via ``scripts/perturb_dream_envelope.py`` →
    # ``kind.training.cross_probe_surface``) overrides at a checkpoint boundary.
    #
    # **Additive and inert at Phase 7.** The waking loop reads neither field;
    # live state-machine ↔ runner consumption is Phase 8. They are
    # content-blind from Io's side (the actor and world model never read them),
    # and they add no Io-readable interface — this is a builder-side control.
    # Both defaults are the settled §2.2 / §2.4 configs.
    dream_envelope: DreamEnvelopeConfig = DreamEnvelopeConfig()
    seed_selection: SeedSelectionConfig = SeedSelectionConfig()

    # Phase 8a — the four-axis dream-rollout config the live state-machine-driven
    # dream session uses (distinct from ``dream_horizon``, which is the *waking-
    # planning* rollout's horizon, the Phase 3 control). Default is the settled
    # four-axis dreaming regime (``kind/training/dream.py`` §2.3). Read only on the
    # dreaming path; the waking loop never touches it.
    dream_rollout_config: DreamRolloutConfig = DreamRolloutConfig()

    # Phase 8a — the supervisor's dormant-state poll cadence (Fork A / the
    # decision doc's tick-cadence ratification: ~1 Hz wall-clock while dormant,
    # responsive to desktop_on within ~1 s, sub-heartbeat granularity). Only the
    # dormant branch sleeps; the waking branch ticks once per ``_step_once``.
    dormant_tick_interval_s: float = 1.0

    # Phase 8c — offline-period checkpoint cadence (the *minimum* wall-clock gap
    # between offline checkpoints, evaluated at dormant between-sessions
    # boundaries). A crash during a long absence loses at most one such interval
    # of dreaming. Default 5 minutes. Offline checkpoints are barrier-free (the
    # env is off) and capture the full resumable state incl. the metabolic bucket
    # + controller state.
    offline_checkpoint_interval_ms: int = 5 * 60 * 1000

    # Pickled extra fields are stored as a Tensor-friendly dict via the
    # rng_state pickle so checkpoint resume can restore them.
    _checkpoint_id_zero_pad: int = field(default=6, repr=False)


# ---- runner -------------------------------------------------------------


class Runner:
    """Drives the env loop, training, dream emissions, and checkpoints.

    Lifecycle: construct, optionally :meth:`load_checkpoint`, then
    :meth:`run` for the desired number of env steps. :meth:`close`
    flushes sinks and shuts down the transport client. The runner is
    not re-entrant; :meth:`run` is intended to be called once per
    instance.

    Construction takes:

    * ``config`` — the :class:`RunnerConfig`.
    * ``transport_client`` — an :class:`~kind.env.transport.EnvTransportClient`
      that has *not* yet been connected. The runner calls
      :meth:`~kind.env.transport.EnvTransportClient.set_world_event_handler`
      at construction time so the client's reader thread routes
      ``WorldEvent`` records to the runner's JSONL sink.
    * ``env_server`` — optional :class:`~kind.env.env_server.EnvServer`
      reference. Under Probe 1's loopback deployment, both ends are in
      the same process, so the runner can call
      :meth:`~kind.env.env_server.EnvServer.set_checkpoint_id` directly
      after each commit so subsequent ``WorldEvent`` records carry the
      new checkpoint id. For a real desktop split this reference is
      ``None`` and the desktop's records keep ``checkpoint_id=None``
      between commits — a documented limitation of Probe 1's wire
      protocol (no ``SET_CHECKPOINT_ID`` message yet).
    """

    def __init__(
        self,
        config: RunnerConfig,
        transport_client: EnvTransportClient,
        env_server: EnvServer | None = None,
        step_callback: StepCallback | None = None,
        host_signal_source: DesktopSignalSource | None = None,
        host_clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._config = config
        self._transport = transport_client
        self._env_server = env_server
        self._step_callback = step_callback
        self._device = torch.device(config.device)
        # Phase 8a supervisor inputs. The desktop on/off source is the
        # exogenous trigger (default: always-on, so the pre-8a smokes run pure
        # waking). The clock drives the controller tick's wallclock (heartbeat
        # cadence, transition timestamps); injectable so tests are deterministic.
        self._host_signal_source: DesktopSignalSource = (
            host_signal_source if host_signal_source is not None else AlwaysAwakeDesktop()
        )
        self._host_clock_ms: Callable[[], int] = (
            host_clock_ms
            if host_clock_ms is not None
            else (lambda: time.monotonic_ns() // 1_000_000)
        )
        # Set True only while a checkpoint commit is staging (item 6 plumbing).
        # In the single-process synchronous supervisor the tick never runs
        # *during* a commit, so the dreaming path reads this as False; the path
        # is wired live (no hardcoded constant) and becomes meaningful in a real
        # Mac/desktop split where a commit could overlap dreaming.
        self._checkpoint_in_progress: bool = False
        # The state-machine supervisor pieces are built lazily at the top of
        # run() (so checkpoint_id reflects any load_checkpoint), once.
        self._controller: StateController | None = None
        self._dream_session_sink: DreamSessionSink | None = None
        self._dream_rng: torch.Generator | None = None
        # Phase 8c offline-period checkpointing. The metabolic bucket reference
        # (for snapshotting); the pending offline-state to restore on resume; and
        # the wall-clock of the last offline checkpoint (cadence throttle).
        self._metabolic_budget: MetabolicBudget | None = None
        self._pending_offline_state: dict[str, Any] | None = None
        self._last_offline_checkpoint_ms: int | None = None

        # ---- models on device -----------------------------------------
        # RunnerConfig is authoritative for the four Probe 1.5 self-
        # prediction fields (plan §2.4): override the user-supplied
        # ``world_model_config`` so the constructed WorldModel honors
        # whatever the runner config specifies. By default both share
        # the same defaults, so this is a no-op for unmodified configs.
        # Probe 2 v2 (plan §2.5): only the ``"disable_self_prediction"``
        # kind affects WorldModel; other kinds are absorbed at the
        # ensemble construction (``"ensemble_k1"``, ``"ensemble_constant"``)
        # or at views.split (``"zero_or_randomize_scalar"``) or via a
        # pre-run checkpoint mutation (``"init_zero_scalar_column"``).
        wm_lesion_kind: Literal[None, "disable_self_prediction"] = (
            "disable_self_prediction"
            if config.lesion_kind == "disable_self_prediction"
            else None
        )
        wm_cfg = dataclasses.replace(
            config.world_model_config,
            ema_decay=config.ema_decay,
            self_prediction_target_mode=config.self_prediction_target_mode,
            self_prediction_loss_form=config.self_prediction_loss_form,
            lesion_kind=wm_lesion_kind,
        )
        self._world_model = WorldModel(wm_cfg).to(self._device)
        self._actor = Actor(
            h_dim=wm_cfg.h_dim,
            z_dim=wm_cfg.z_dim,
            action_dim=config.action_dim,
            mlp_hidden=wm_cfg.mlp_hidden,
        ).to(self._device)
        # Probe 2 v2 ``ensemble_k1`` / ``ensemble_constant`` lesions
        # (plan §2.5): the v1 candidates' implementation point is the
        # ensemble's constructor. ``ensemble_k1`` shrinks K to 1 (single
        # head → variance over a length-1 axis is identically 0;
        # disagreement is structurally absent). ``ensemble_constant``
        # keeps K at config.ensemble_k but flips the
        # ``lesion_constant_disagreement`` flag so ``disagreement(...)``
        # short-circuits to zeros — the heads still train normally so
        # resume / checkpoint shape is unaffected.
        ensemble_k_effective = (
            1 if config.lesion_kind == "ensemble_k1" else config.ensemble_k
        )
        ensemble_constant = config.lesion_kind == "ensemble_constant"
        self._ensemble = LatentDisagreementEnsemble(
            h_dim=wm_cfg.h_dim,
            z_dim=wm_cfg.z_dim,
            action_dim=config.action_dim,
            K=ensemble_k_effective,
            mlp_hidden=wm_cfg.mlp_hidden,
            lesion_constant_disagreement=ensemble_constant,
        ).to(self._device)

        # ---- optimizers ------------------------------------------------
        self._wm_opt = torch.optim.Adam(
            self._world_model.parameters(), lr=config.world_model_lr
        )
        self._actor_opt = torch.optim.Adam(
            self._actor.parameters(), lr=config.actor_lr
        )
        self._ens_opt = torch.optim.Adam(
            self._ensemble.parameters(), lr=config.ensemble_lr
        )

        # ---- telemetry sinks -------------------------------------------
        # Phase 1's sinks: agent_step + dream_rollout to Parquet (columnar,
        # downstream-friendly); replay_meta + world_event to JSONL
        # (low-volume, human-inspectable). All four live under
        # ``telemetry_dir`` per the plan §2.2 layout.
        config.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self._agent_step_sink = ParquetSink(
            config.telemetry_dir / _AGENT_STEP_DIR,
            AgentStep,
            rows_per_shard=config.parquet_rows_per_shard,
        )
        self._dream_sink = ParquetSink(
            config.telemetry_dir / _DREAM_ROLLOUT_DIR,
            DreamRollout,
            rows_per_shard=config.parquet_rows_per_shard,
        )
        self._replay_meta_sink = JsonlSink(
            config.telemetry_dir / _REPLAY_META_FILE, ReplayMeta
        )
        self._world_event_sink = JsonlSink(
            config.telemetry_dir / _WORLD_EVENT_FILE, WorldEvent
        )

        # Probe 2 v2 pre-registration sink (plan §2.5; synthesis §2.4
        # element 1). Constructed only when ``pre_reg_dir`` is set so
        # Probe 1 / Probe 1.5 runners create no extra artifact. The sink's
        # constructor creates the directory if missing and opens the file
        # for append.
        self._pre_reg_sink: PreRegSink | None = (
            PreRegSink(config.pre_reg_dir)
            if config.pre_reg_dir is not None
            else None
        )

        # Wire the WorldEvent stream from the transport client into our
        # JSONL sink. The client's reader thread invokes the handler
        # synchronously on each WORLD_EVENT message.
        self._transport.set_world_event_handler(self._world_event_sink.write)

        # ---- replay buffer ---------------------------------------------
        # The buffer's RNG is the runner's torch.Generator so sampling
        # stays deterministic across resume from checkpoint (see
        # _save_rng_state / _load_rng_state).
        self._sample_rng = torch.Generator(device="cpu")
        self._sample_rng.manual_seed(0)
        self._replay = SequenceReplayBuffer(
            capacity=config.replay_capacity,
            sequence_length=config.replay_sequence_length,
            run_id=config.run_id,
            replay_meta_handler=self._replay_meta_sink.write,
            rng=self._sample_rng,
        )

        # ---- checkpoint manager ---------------------------------------
        self._checkpoint_manager = CheckpointManager(
            config.checkpoints_dir, transport_client
        )
        self._checkpoint_id: str | None = None
        self._checkpoint_counter: int = 0

        # ---- runtime state (initialised in run() / load_checkpoint) ---
        # h_prev / z_prev / a_prev are the (h, z, a) inputs to the next
        # iteration's world_model.step. They live on the model device.
        self._h_prev: Tensor | None = None
        self._z_prev: Tensor | None = None
        self._a_prev: Tensor | None = None
        # The *last* world_model.step's h, z — used to seed dream rollouts.
        self._h_curr: Tensor | None = None
        self._z_curr: Tensor | None = None
        # The current observation (pre-step), kept as a CPU float32 tensor
        # ready for buffer storage. The runner converts numpy uint8 →
        # float32 / 255 once and reuses for both the model forward and
        # the buffer.
        self._obs_curr: Tensor | None = None
        self._obs_curr_hash: str | None = None
        # The latest EnvStep metadata (env_step, episode_id, step_in_episode).
        self._env_step_meta: EnvStep | None = None
        # The runner's local "iteration" counter — distinct from the
        # env-server's env_step. Equals env_step at start, advances by 1
        # per loop iteration. Persisted across checkpoints.
        self._iteration: int = 0

        # Probe 2 v2 lesion bookkeeping (plan §2.5; synthesis §2.4
        # element 4): a single ``mirror_marker`` is emitted at the top of
        # ``run()`` if ``lesion_kind`` is non-None; the flag below
        # guarantees one emission per Runner instance regardless of
        # ``run()`` re-entry edge cases. Pre-computed lesion routing
        # decisions for ``views.split``: only the
        # ``"zero_or_randomize_scalar"`` kind reaches split; other kinds
        # have already taken effect at construction time.
        self._lesion_marker_emitted: bool = False
        self._views_lesion_kind: Literal["zero", "randomize"] | None = (
            config.lesion_zero_or_randomize_variant
            if config.lesion_kind == "zero_or_randomize_scalar"
            else None
        )

        # Probe 3.5: energy telemetry is an explicit opt-in (default off →
        # existing runners emit "0.2.0" byte-identically). When on, it requires
        # the mirror-side ground truth (``true_energy`` on ``GridState``),
        # reachable only through a co-located ``env_server`` (the loopback
        # Phase-1 deployment) — so an env_server-less runner with the flag on is
        # a misconfiguration. ``sensed_energy`` is fused into the world model on
        # every step regardless of this flag.
        if config.energy_telemetry and env_server is None:
            raise ValueError(
                "RunnerConfig.energy_telemetry=True requires a co-located "
                "env_server (true_energy is read from GridState); got None."
            )
        self._energy_telemetry_enabled: bool = config.energy_telemetry

        # Probe 4 Phase 2 (plan §S-PERT): the step-boundary perturbation
        # surfaces. Both are opt-in (default None → byte-identical); both
        # need the co-located env_server (placement reads GridState; firing
        # goes through the tested mutator surface).
        if (
            config.perturbation_generator is not None
            or config.perturbation_inbox_dir is not None
        ) and env_server is None:
            raise ValueError(
                "RunnerConfig.perturbation_generator / perturbation_inbox_dir "
                "require a co-located env_server (placement reads GridState; "
                "mutations fire through EnvServer); got None."
            )
        self._perturbation_generator: PerturbationGenerator | None = (
            PerturbationGenerator(config.perturbation_generator)
            if config.perturbation_generator is not None
            else None
        )

        # Probe 4.5 S-DEC: the maintenance cadence is checkpoint-aligned by
        # pre-registration (§3) — enforce the alignment structurally so the
        # refit can only ever fire at a checkpoint boundary.
        if config.maintenance_refit is not None and (
            config.maintenance_refit.refit_every_n_env_steps
            % config.checkpoint_every_n_env_steps
            != 0
        ):
            raise ValueError(
                "RunnerConfig.maintenance_refit.refit_every_n_env_steps must "
                "be a multiple of checkpoint_every_n_env_steps (prereg §3: "
                "checkpoint-aligned cadence); got "
                f"{config.maintenance_refit.refit_every_n_env_steps} vs "
                f"{config.checkpoint_every_n_env_steps}."
            )
        # Consumption co-timing detector for the generator's not-self
        # deferral: consecutive true-energy samples, the house jump
        # threshold (observer.source_events). Independent of the
        # energy_telemetry flag — read directly from the co-located
        # GridState at drain time.
        self._perturbation_prev_true_energy: float | None = None

        # Probe 3.5 Phase 2: the most recent imagination-training step's
        # pragmatic/epistemic decomposition (per-training-step quantities —
        # grounding fact 3). Stamped onto AgentStep records when the Phase-2
        # record version is in effect. Before the first training step the
        # values are genuinely zero: no preference gradient has existed yet.
        self._last_pragmatic_value: float = 0.0
        self._last_epistemic_value: float = 0.0
        self._last_pragmatic_share: float = 0.0

        self._closed: bool = False

    # ---- public API -----------------------------------------------------

    def run(self, total_env_steps: int) -> None:
        """Drive the loop for ``total_env_steps`` iterations.

        Calls :meth:`~kind.env.transport.EnvTransportClient.connect` if
        the runner has not yet acquired an initial observation (i.e.
        :meth:`load_checkpoint` was not called). Otherwise uses the
        loaded state.

        The loop's per-iteration sequence is recorded in
        ``docs/plans/Kind_probe1_implementation_plan.md`` §2.9; the
        implementation here is its literal translation.
        """
        if total_env_steps <= 0:
            raise ValueError(
                f"total_env_steps must be positive, got {total_env_steps}"
            )
        if self._closed:
            raise RuntimeError("Runner is closed; cannot run.")

        if self._env_step_meta is None:
            # No checkpoint loaded — connect and seed initial state.
            initial = self._transport.connect()
            self._absorb_env_step(initial)
            self._init_runtime_zero_state()

        # Probe 2 v2 (plan §2.5; synthesis §2.4 element 4): emit a single
        # ``mirror_marker`` ``world_event`` at run start if a lesion is
        # configured, so downstream digests / readers can identify the
        # run's lesion shape from the world_event stream alone. The flag
        # guarantees one emission per Runner instance even if ``run()``
        # is re-entered. Convention matches Probe 1.5 Phase 5's
        # mirror_marker: ``event_type="mirror_marker"``, ``source="system"``,
        # ``payload.lesion_kind=<the kind>``, plus
        # ``payload.lesion_variant`` for ``"zero_or_randomize_scalar"``.
        self._maybe_emit_lesion_mirror_marker()
        self._build_state_controller()

        # Phase 8a supervisor loop (Fork A, A-shape-1). Each iteration ticks the
        # StateController on the current host signals, then runs the body of the
        # resulting state. ``total_env_steps`` counts *waking* steps (env steps),
        # so dreaming/dormant iterations don't consume the budget — and when the
        # desktop is on throughout (the pre-8a smokes), every iteration is a
        # waking ``_step_once`` and behavior is byte-identical to the old
        # ``for _ in range(total_env_steps): self._step_once()`` loop (the
        # controller starts in waking and tick returns None each time, emitting
        # nothing). The dream-state machinery activates only on a desktop_off
        # edge, which the always-on default source never produces.
        controller = self._controller
        assert controller is not None
        waking_steps_done = 0
        interval_ms = self._config.offline_checkpoint_interval_ms
        while waking_steps_done < total_env_steps:
            now_ms = self._host_clock_ms()
            # Phase 8c: an offline-period checkpoint is *due* when we are offline
            # (dreaming/dormant) and a full interval has elapsed since the last
            # one (or the offline-period start). Setting ``checkpoint_in_progress``
            # gates a dream re-entry so the save lands at a clean dormant
            # between-sessions boundary, never mid-session — the
            # ``CheckpointWindowCap`` made live. A session in flight is never
            # interrupted (option a); the checkpoint waits at most one session.
            prev_state = controller.current_state
            offline_ckpt_due = (
                prev_state in ("dreaming", "dormant")
                and self._last_offline_checkpoint_ms is not None
                and now_ms - self._last_offline_checkpoint_ms >= interval_ms
            )
            self._checkpoint_in_progress = offline_ckpt_due
            host = self._current_host_signals()
            env_step_now = (
                self._env_step_meta.env_step
                if self._env_step_meta is not None
                else self._iteration
            )
            transition = controller.tick(host, env_step_now, now_ms)
            state: State = controller.current_state
            # Mark the offline-period start so the cadence counts from when the
            # desktop went off (not from process start / the previous absence).
            if (
                transition is not None
                and transition.from_state == "waking"
                and state in ("dreaming", "dormant")
            ):
                self._last_offline_checkpoint_ms = now_ms
            if state == "waking":
                self._checkpoint_in_progress = False  # waking owns its own flag
                self._step_once()
                waking_steps_done += 1
            elif state == "dreaming":
                # The dream session already ran synchronously inside tick()
                # (RunDreamSessionDriver runs run_dream_session in
                # _enter_dreaming); the next tick transitions dreaming → dormant
                # on the pending cap. Nothing to do here.
                continue
            elif state == "dormant":
                if offline_ckpt_due:
                    # Clean between-sessions boundary: commit the offline
                    # checkpoint (barrier-free; full resumable state incl. bucket).
                    self._commit_offline_checkpoint()
                    self._last_offline_checkpoint_ms = now_ms
                    self._checkpoint_in_progress = False
                elif self._config.dormant_tick_interval_s > 0:
                    # Heartbeat-idle: poll at the ratified ~1 Hz dormant cadence,
                    # responsive to desktop_on. The heartbeat fires inside tick().
                    time.sleep(self._config.dormant_tick_interval_s)
            else:  # "paused" — supervisor-mediated; not entered by this loop.
                break

    def _build_state_controller(self) -> None:
        """Construct the supervisor's StateController + dream driver once.

        Built lazily at the top of run() (after any load_checkpoint) so the
        driver's ``checkpoint_id`` reflects the loaded checkpoint. Wires the
        *live* Phase 7 surface (``config.dream_envelope`` / ``config.seed_selection``
        — the caps and seed-selection the entering session reads, and the
        ``DreamSessionMeta`` provenance snapshots), the Phase 6 protection
        composite (not the stub), and the four-axis dream config. The dream
        session writes four-axis ``"0.3.0"`` ``DreamRollout`` records to the same
        ``dream_rollout`` stream as the waking-planning rollout (disambiguated by
        ``schema_version`` + ``dream_session_id``, Fork A), and state transitions
        to the existing ``world_event`` stream.

        ``checkpoint_hash=None`` so ``run_dream_session`` recomputes the hash from
        the *live* weights each session. (Known 8a limitation: the driver's
        ``checkpoint_id`` *label* is captured here at run-start; a checkpoint that
        commits during waking before a later dream session leaves the label
        stale — cosmetic, since the hash is live. Refreshing the label per
        session is a later refinement.)
        """
        if self._controller is not None:
            return
        self._dream_session_sink = DreamSessionSink(self._config.telemetry_dir)
        self._dream_rng = torch.Generator()
        self._dream_rng.manual_seed(0)
        driver = RunDreamSessionDriver(
            world_model=self._world_model,
            actor=self._actor,
            ensemble=self._ensemble,
            replay_buffer=self._replay,
            seed_selection_config=self._config.seed_selection,
            rollout_config=self._config.dream_rollout_config,
            run_id=self._config.run_id,
            checkpoint_id=self._checkpoint_id,
            rng=self._dream_rng,
            device=self._device,
            dream_rollout_sink=self._dream_sink,
            dream_session_sink=self._dream_session_sink,
            envelope_config_snapshot=dataclasses.asdict(self._config.dream_envelope),
            checkpoint_hash=None,
        )
        # Phase 8 token-bucket metabolic pacer, built from the live Phase 7
        # envelope's capacity/refill knobs (the dream/rest duty cycle).
        metabolic_budget = MetabolicBudget(
            capacity_seconds=self._config.dream_envelope.metabolic_capacity_seconds,
            refill_rate=self._config.dream_envelope.metabolic_refill_rate,
        )
        self._metabolic_budget = metabolic_budget
        self._controller = StateController(
            self._config.dream_envelope,
            self._config.seed_selection,
            dream_driver=driver,
            protection=DreamProtectionPolicy(budget=metabolic_budget),
            world_event_emit=self._world_event_sink,
            run_id=self._config.run_id,
            checkpoint_id=self._checkpoint_id,
            dream_session_id_factory=lambda: str(uuid.uuid4()),
            # Phase 8b: the same host clock the supervisor ticks with, so the
            # controller measures each dream session's real wall-time and records
            # it into the metabolic ledger (the B2 re-entry pacer's actuals).
            clock=self._host_clock_ms,
        )
        # Phase 8c: resume from an offline-period checkpoint, if one was loaded.
        # The bucket is restored *frozen across the pause* (its refill clock
        # resets — see MetabolicBudget.restore); the controller resumes into its
        # checkpointed state (waking/dormant). Resuming from a waking checkpoint
        # (no offline_state) correctly keeps the defaults: waking + full bucket.
        if self._pending_offline_state is not None:
            now_ms = self._host_clock_ms()
            metabolic_budget.restore(self._pending_offline_state["metabolic_budget"])
            self._controller.restore_resume_state(
                self._pending_offline_state["controller_state"], now_ms=now_ms
            )
            self._last_offline_checkpoint_ms = now_ms
            self._pending_offline_state = None

    def _current_host_signals(self) -> HostSignals:
        """Assemble the exogenous ``HostSignals`` for one supervisor tick.

        ``desktop_alive`` comes from the injected desktop source (the exogenous
        trigger); ``checkpoint_in_progress`` is the runner's *own* commit state
        (item 6 plumbing). Nothing here is Io-derived.
        """
        return HostSignals(
            desktop_alive=self._host_signal_source.desktop_alive(),
            checkpoint_in_progress=self._checkpoint_in_progress,
        )

    def load_checkpoint(self, checkpoint_id: str) -> None:
        """Restore weights, optimizer state, RNG state, and runtime state.

        Must be called *before* :meth:`run` (it sets the runner's
        runtime state, which run() consumes). Raises :class:`FileNotFoundError`
        if the checkpoint id is unknown.

        Probe 1.5 v2 (plan §2.4): reads ``schema_version.txt`` and
        branches on the source schema. For ``"0.2.0"`` checkpoints the
        path is unchanged from Probe 1's resume contract — exact byte
        equality on weights, optimizers, RNG, runtime state. For
        ``"0.1.0"`` (Probe 1) checkpoints the runner initialises the
        EMA target as a fresh copy of the freshly-loaded online
        network, expands the actor's first-layer weight to the new
        ``h_dim + z_dim + 1`` input dimension with the extra column
        zero-initialized (preserves the Probe 1 actor's behaviour
        exactly when the scalar is zero — i.e., on the first env step
        the actor's logits are byte-identical to what the Probe 1
        actor would have produced), re-initializes
        ``_frozen_projection`` from a fresh
        ``torch.nn.init.orthogonal_`` if the configured target_mode is
        ``"frozen"``, skips optimizer-state load (the saved Adam state
        was keyed against the Probe 1 parameter set; the optimizers
        become fresh in the Probe 1.5 process), and emits a
        ``world_event`` with ``event_type="mirror_marker"`` describing
        the asymmetry. The plan picks "initialise from online when
        missing" rather than "refuse to load" so a future investigation
        running the failure-mode controls from a Probe 1 starting state
        is not blocked.
        """
        contents = self._checkpoint_manager.load(checkpoint_id)
        # ---- read schema version BEFORE splitting weights ----
        checkpoint_schema_version = contents.schema_version_path.read_text().strip()
        is_probe_1_checkpoint = checkpoint_schema_version == PROBE_1_SCHEMA_VERSION

        # ---- weights ----
        weights = load_file(str(contents.weights_path), device=str(self._device))
        wm_state_disk: dict[str, Tensor] = {}
        actor_state_disk: dict[str, Tensor] = {}
        ens_state_disk: dict[str, Tensor] = {}
        for k, v in weights.items():
            if k.startswith("world_model."):
                wm_state_disk[k[len("world_model.") :]] = v
            elif k.startswith("actor."):
                actor_state_disk[k[len("actor.") :]] = v
            elif k.startswith("ensemble."):
                ens_state_disk[k[len("ensemble.") :]] = v

        if is_probe_1_checkpoint:
            # World model: load with strict=False so the missing
            # ``target_encoder.*`` / ``target_gru_cell.*`` /
            # optionally ``_frozen_projection`` / ``_environmental_projection``
            # keys do not raise. The constructor has already populated
            # those tensors with reasonable initial values; the
            # ``_initialize_ema_target_from_online`` call below
            # overwrites the EMA target tensors so they start identical
            # to the freshly-loaded online network (BYOL convention,
            # synthesis §1.2 v2).
            self._world_model.load_state_dict(wm_state_disk, strict=False)
            self._world_model._initialize_ema_target_from_online()
            if (
                self._world_model.config.self_prediction_target_mode
                == "frozen"
                and self._world_model._frozen_projection is not None
            ):
                with torch.no_grad():
                    fresh = torch.empty_like(
                        self._world_model._frozen_projection.data
                    )
                    torch.nn.init.orthogonal_(fresh)
                    self._world_model._frozen_projection.data.copy_(fresh)

            # Actor: expand the saved first-layer weight by one
            # zero-initialized column so the loaded actor behaves
            # byte-identically to the Probe 1 actor when the scalar
            # input is zero (which it is on the first env step under
            # the masking convention, and remains the practical
            # starting condition until env-step training drives the
            # column away from zero).
            actor_state_full = dict(actor_state_disk)
            first_layer_key = "net.0.weight"
            if first_layer_key in actor_state_full:
                disk_weight = actor_state_full[first_layer_key]
                expected_in = (
                    self._world_model.config.h_dim
                    + self._world_model.config.z_dim
                    + 1
                )
                if disk_weight.shape[-1] != expected_in:
                    expanded = torch.zeros(
                        disk_weight.shape[0],
                        expected_in,
                        dtype=disk_weight.dtype,
                        device=disk_weight.device,
                    )
                    expanded[:, : disk_weight.shape[-1]] = disk_weight
                    # The trailing column stays zero (zeros allocator
                    # above); this matches the constructor's zero-init
                    # convention for the new column (plan §6 row 15).
                    actor_state_full[first_layer_key] = expanded
            self._actor.load_state_dict(actor_state_full)

            # Ensemble: shape unchanged from Probe 1.
            self._ensemble.load_state_dict(ens_state_disk)
        else:
            # ``"0.2.0"`` resume: exact byte equality.
            self._world_model.load_state_dict(wm_state_disk)
            self._actor.load_state_dict(actor_state_disk)
            self._ensemble.load_state_dict(ens_state_disk)

        # ---- optimizers ----
        # For Probe 1 checkpoints the saved optimizer state was keyed
        # against the Probe 1 parameter set (no self_prediction_head,
        # no EMA target, smaller actor first-layer weight). PyTorch's
        # ``Adam.load_state_dict`` requires shape-matching state to the
        # current parameter set, so loading the Probe-1-shaped Adam
        # state into the Probe-1.5-shaped optimizers raises. We skip
        # the optimizer-state load in that case; the optimizers stay
        # fresh in the resumed process. This loses Adam momentum
        # accumulation for the Probe 1 parameters but preserves
        # parameter values; the asymmetry is recorded in the
        # ``mirror_marker`` ``world_event`` below so the journal entry
        # captures it. ``"0.2.0"`` checkpoints continue to round-trip
        # exactly.
        if not is_probe_1_checkpoint:
            opt_states = torch.load(
                contents.optimizer_state_path,
                map_location=self._device,
                weights_only=False,
            )
            self._wm_opt.load_state_dict(opt_states["world_model"])
            self._actor_opt.load_state_dict(opt_states["actor"])
            self._ens_opt.load_state_dict(opt_states["ensemble"])

        # ---- RNG + runtime state ----
        with open(contents.rng_state_path, "rb") as fh:
            rng_blob: dict[str, Any] = pickle.load(fh)
        self._load_rng_state(rng_blob)

        # ---- Phase 8c: offline-period state (bucket + controller resume) ----
        # Present only on offline checkpoints; stashed here and applied by
        # _build_state_controller (which builds the bucket + controller). A waking
        # checkpoint has no offline_state → resume keeps the defaults (waking +
        # full bucket), which is exactly the waking state.
        if (
            contents.offline_state_path is not None
            and contents.offline_state_path.is_file()
        ):
            self._pending_offline_state = json.loads(
                contents.offline_state_path.read_text()
            )

        # ---- checkpoint id (for envelope on subsequent records) ----
        self._checkpoint_id = checkpoint_id
        self._replay.set_checkpoint_id(checkpoint_id)
        if self._env_server is not None:
            self._env_server.set_checkpoint_id(checkpoint_id)
        # Bump the counter so the next auto-id doesn't collide (best-effort:
        # if the loaded id was ckpt-NNNNNN, parse it back).
        self._checkpoint_counter = self._parse_checkpoint_counter(checkpoint_id)

        # ---- mirror_marker for Probe 1 → Probe 1.5 transitions ----
        if is_probe_1_checkpoint:
            t_event = (
                self._env_step_meta.env_step
                if self._env_step_meta is not None
                else 0
            )
            self._world_event_sink.write(
                WorldEvent(
                    schema_version=SCHEMA_VERSION,
                    run_id=self._config.run_id,
                    checkpoint_id=checkpoint_id,
                    t_event=t_event,
                    event_type="mirror_marker",
                    source="system",
                    payload={
                        "note": (
                            "loaded Probe 1 checkpoint; EMA target "
                            "initialised from online; actor input "
                            "column zero-initialized; optimizer state "
                            "skipped (fresh Adam in Probe 1.5 process)"
                        ),
                        "checkpoint_id": checkpoint_id,
                        "checkpoint_schema_version": checkpoint_schema_version,
                    },
                    wallclock_ms=time.monotonic_ns() // 1_000_000,
                )
            )

    def close(self) -> None:
        """Flush sinks, close the transport client. Idempotent."""
        if self._closed:
            return
        self._closed = True
        # Sinks first: their close() fsyncs and finalises shards.
        for sink in (
            self._agent_step_sink,
            self._dream_sink,
            self._replay_meta_sink,
            self._world_event_sink,
        ):
            try:
                sink.close()
            except Exception:
                # Best-effort cleanup; don't mask earlier exceptions.
                pass
        if self._pre_reg_sink is not None:
            try:
                self._pre_reg_sink.close()
            except Exception:
                pass
        if self._dream_session_sink is not None:
            try:
                self._dream_session_sink.close()
            except Exception:
                pass
        try:
            self._transport.close()
        except Exception:
            pass

    def emit_pre_registration(self, record: PreRegistration) -> None:
        """Write a :class:`PreRegistration` record through the runner's
        :class:`PreRegSink`.

        Probe 2 v2 calibration protocol's first element (plan §2.5;
        synthesis §2.4 element 1): pre-registration is the structural
        counter against the reading drifting toward what the builder
        hopes for in retrospect. Append-only, written before the
        adversarial-pass reading runs. Multiple calls within a single
        runner instance append additional records to the same JSONL.

        Raises :class:`RuntimeError` if the runner has been closed or if
        no ``pre_reg_dir`` was configured on the :class:`RunnerConfig`.
        """
        if self._closed:
            raise RuntimeError("Runner is closed; cannot emit pre-registration.")
        if self._pre_reg_sink is None:
            raise RuntimeError(
                "Runner.emit_pre_registration called but no pre_reg_dir is "
                "configured on RunnerConfig. Set RunnerConfig.pre_reg_dir to "
                "enable pre-registration emission."
            )
        self._pre_reg_sink.write(record)

    # ---- lesion mirror_marker emission ---------------------------------

    def _maybe_emit_lesion_mirror_marker(self) -> None:
        """Emit a single ``mirror_marker`` ``world_event`` at run start
        if a lesion is configured.

        Probe 2 v2 (plan §2.5; synthesis §2.4 element 4). The emission's
        shape matches Probe 1.5 Phase 5's settled convention:
        ``event_type="mirror_marker"``, ``source="system"``, and a
        payload carrying the lesion identification fields. For the
        ``"zero_or_randomize_scalar"`` kind the payload also carries
        ``lesion_variant`` (``"zero"`` or ``"randomize"``) plus the
        empirical bounds that drove the randomize variant; the
        ``"init_zero_scalar_column"`` kind is normally emitted by the
        mutation script (which runs before the runner loads the lesioned
        checkpoint), but if a runner is constructed with that kind set
        explicitly, the marker fires here for consistency. The flag
        ``self._lesion_marker_emitted`` guarantees one emission per
        Runner instance.
        """
        if self._lesion_marker_emitted:
            return
        if self._config.lesion_kind is None:
            self._lesion_marker_emitted = True
            return

        t_event = (
            self._env_step_meta.env_step
            if self._env_step_meta is not None
            else 0
        )
        payload: dict[str, Any] = {
            "lesion_kind": self._config.lesion_kind,
        }
        if self._config.lesion_kind == "zero_or_randomize_scalar":
            payload["lesion_variant"] = (
                self._config.lesion_zero_or_randomize_variant
            )
            payload["lesion_empirical_min"] = (
                self._config.lesion_zero_or_randomize_empirical_min
            )
            payload["lesion_empirical_max"] = (
                self._config.lesion_zero_or_randomize_empirical_max
            )
        self._world_event_sink.write(
            WorldEvent(
                schema_version=SCHEMA_VERSION,
                run_id=self._config.run_id,
                checkpoint_id=self._checkpoint_id,
                t_event=t_event,
                event_type="mirror_marker",
                source="system",
                payload=payload,
                wallclock_ms=time.monotonic_ns() // 1_000_000,
            )
        )
        self._lesion_marker_emitted = True

    # ---- one iteration --------------------------------------------------

    def _step_once(self) -> None:
        """One iteration of the hot loop."""
        assert self._obs_curr is not None
        assert self._h_prev is not None
        assert self._z_prev is not None
        assert self._a_prev is not None
        assert self._env_step_meta is not None
        assert self._obs_curr_hash is not None

        env_step_meta = self._env_step_meta
        obs_t_cpu = self._obs_curr  # (1, 32, 32) float32 on CPU
        obs_t_dev = obs_t_cpu.unsqueeze(0).to(self._device, non_blocking=True)

        # Probe 3.5: the energy paired with the *from* observation. ``sensed`` is
        # what Io observes (carried over the wire on the EnvStep); ``true`` is the
        # GridState ground truth, read here (before stepping the env, so it
        # matches the from-state) and used for telemetry only — never fused, never
        # a training target. The sensed scalar is fused into the world model below.
        sensed_energy_from = float(env_step_meta.sensed_energy)
        true_energy_from: float | None = None
        if self._energy_telemetry_enabled:
            assert self._env_server is not None
            true_energy_from = float(self._env_server.grid_world_state.true_energy)
        sensed_energy_t = torch.tensor(
            [[sensed_energy_from]], device=self._device, dtype=torch.float32
        )

        # 1. World model forward. ``next_obs`` is consumed only by
        #    ``compute_self_prediction_target`` in environmental mode;
        #    for online / frozen modes the kwarg is ignored. Per plan
        #    §2.4 step 1 we pass ``obs_t_dev`` (the observation that
        #    arrived from the *previous* env step's response) as the
        #    environmental-mode target source.
        target_mode = self._config.self_prediction_target_mode
        next_obs_for_target: Tensor | None
        if target_mode == "environmental":
            next_obs_for_target = obs_t_dev
        else:
            next_obs_for_target = None
        wm_step = self._world_model.step(
            obs_t_dev,
            self._h_prev,
            self._z_prev,
            self._a_prev,
            next_obs=next_obs_for_target,
            sensed_energy=sensed_energy_t,
        )

        # 2. Compute intrinsic from (h_prev, z_prev, a_prev). This is
        #    telemetry-only at env-step time (the actor reads only
        #    PolicyView; intrinsic enters the actor's loss in training).
        with torch.no_grad():
            intrinsic = self._ensemble.disagreement(
                self._h_prev, self._z_prev, self._a_prev
            )

        # 3. Compute the per-step self-prediction scalar against the EMA
        #    target ``bar{h}_{t+1}``. Plan §2.4 steps 2–4: target via
        #    ``compute_self_prediction_target`` honoring the configured
        #    target_mode; scalar via ``self_prediction_loss_form``
        #    arithmetic; first-step-of-episode override forces 0.0 with
        #    masked flag True so the actor's input is dimensionally
        #    consistent across all steps and downstream analyzers can
        #    discriminate the sentinel from an empirical near-zero.
        loss_form = self._config.self_prediction_loss_form
        if self._config.lesion_kind == "disable_self_prediction":
            # Probe 2 v2 lesion (plan §2.5): the head emits zeros from
            # ``WorldModel.step`` and the auxiliary loss is identically
            # zero; the per-step scalar is the loss-form's zero-pair
            # value (``cosine: 1.0`` since 1 − cos(0, 0) is 1 under
            # PyTorch's eps-guarded definition; ``mse: 0.0`` since
            # ``||0 − 0||² = 0``). Skipping the actual computation
            # avoids the degenerate cosine-similarity-of-zero-vectors
            # call and pins the sentinel deterministically.
            sp_zero_pair = 1.0 if loss_form == "cosine" else 0.0
            sp_scalar_computed = torch.tensor(
                sp_zero_pair, device=self._device
            )
        else:
            target_h_next = self._world_model.compute_self_prediction_target(
                obs_t_dev,
                self._h_prev,
                self._z_prev,
                self._a_prev,
                next_obs=next_obs_for_target,
            )
            with torch.no_grad():
                if loss_form == "cosine":
                    sp_scalar_computed = (
                        1.0
                        - F.cosine_similarity(
                            wm_step.self_prediction,
                            target_h_next.detach(),
                            dim=-1,
                        ).mean()
                    )
                else:  # "mse"
                    sp_scalar_computed = F.mse_loss(
                        wm_step.self_prediction, target_h_next.detach()
                    )
        is_first_step_of_episode = env_step_meta.step_in_episode == 0
        if is_first_step_of_episode:
            sp_scalar_for_view = torch.zeros((), device=self._device)
            sp_masked = True
        else:
            sp_scalar_for_view = sp_scalar_computed.detach()
            sp_masked = False

        # 4. Split views; actor decides. The scalar lands on PolicyView
        #    (the synthesis §1.3 v2 single-scalar Watts-heuristic
        #    exception); the full prediction vector + masked flag land
        #    on TelemetryView (mirror-side reading).
        # Probe 2 v2 ``zero_or_randomize_scalar`` lesion (plan §2.5):
        # ``self._views_lesion_kind`` is the precomputed routing decision;
        # ``views.split`` overrides the scalar (and forces the masked
        # flag True) for the lesioned step. The substrate-side
        # telemetry path is untouched — ``self_prediction_t`` /
        # ``self_prediction_error_t`` continue to record the head's
        # actual output and the empirical scalar respectively, so the
        # mirror's substrate-side / head-internal cohorts read the
        # un-lesioned signals; only the actor's input is lesioned.
        policy_view, telemetry_view = split(
            wm_step,
            intrinsic,
            self_prediction_error=sp_scalar_for_view,
            self_prediction_error_masked=sp_masked,
            lesion_zero_or_randomize=self._views_lesion_kind,
            lesion_empirical_min=(
                self._config.lesion_zero_or_randomize_empirical_min
            ),
            lesion_empirical_max=(
                self._config.lesion_zero_or_randomize_empirical_max
            ),
            lesion_rng=self._sample_rng,
        )
        action_output = self._actor.forward(policy_view)
        action_int = int(action_output.action.item())

        # 4. Step env.
        next_meta = self._transport.step(action_int)

        # 4b. Probe 4 Phase 2 (plan §S-PERT): step-boundary perturbation
        #     placement. The generator poll and the manual-trigger inbox
        #     drain both run here — after the env step (so the mutation
        #     applies to the post-step world, stamps t_event = the step
        #     just taken, and becomes visible in the *next* observation,
        #     the documented builder-event timing convention) and before
        #     the AgentStep emit (so placement is deterministic relative
        #     to the logged transitions). Direct env-server calls are safe
        #     here for the same reason the energy-telemetry read is: the
        #     transport server's reader thread is blocked awaiting the
        #     next message between our synchronous step() calls.
        if (
            self._perturbation_generator is not None
            or self._config.perturbation_inbox_dir is not None
        ):
            self._drain_perturbations(next_meta.env_step)

        # 5. Build & insert transition. The transition's obs/episode_id
        #    are the *from* observation's (the one this action was based
        #    on); next_obs is the env's response.
        next_obs_cpu = self._obs_to_cpu_tensor(next_meta.observation)
        transition = Transition(
            obs=obs_t_cpu,
            action=action_int,
            next_obs=next_obs_cpu,
            env_step=env_step_meta.env_step,
            episode_id=env_step_meta.episode_id,
            step_in_episode=env_step_meta.step_in_episode,
            # Probe 3.5: the from-step sensed energy, so the world model can
            # train its energy reconstruction on replayed sequences.
            sensed_energy=sensed_energy_from,
        )
        self._replay.insert(transition)

        # 6. Emit AgentStep telemetry. Probe 2 v2 (plan §2.5): the
        #    scalar/masked fields recorded into AgentStep come *post-
        #    split* so the ``"zero_or_randomize_scalar"`` lesion's
        #    override on PolicyView's scalar (and the True masked flag
        #    on TelemetryView) lands in the telemetry stream. Under no
        #    lesion the post-split values equal the pre-split values
        #    exactly (split passes them through), so non-lesioned runs
        #    produce telemetry byte-identical to the un-lesioned
        #    reference. The head's full vector (``self_prediction_t``)
        #    is *not* lesioned at split — substrate-side / head-internal
        #    cohorts continue to see the head's actual output.
        agent_step_record = self._emit_agent_step(
            env_step_meta=env_step_meta,
            wm_step=wm_step,
            obs_t_dev=obs_t_dev,
            action_output=action_output,
            intrinsic=intrinsic,
            obs_hash=self._obs_curr_hash,
            self_prediction_error=policy_view.self_prediction_error,
            self_prediction_error_masked=(
                telemetry_view.self_prediction_error_masked
            ),
            sensed_energy_value=sensed_energy_from,
            true_energy_value=true_energy_from,
        )

        # 7. Update runtime state for next iteration.
        episode_changed = next_meta.episode_id != env_step_meta.episode_id
        if episode_changed:
            self._init_runtime_zero_state_keep_obs()
        else:
            self._h_prev = wm_step.h.detach()
            self._z_prev = wm_step.z.detach()
            self._a_prev = action_output.action.detach()
        # h_curr / z_curr always reflect the most recent forward — used as
        # the dream seed regardless of episode boundary (the latest world
        # model output is always meaningful).
        self._h_curr = wm_step.h.detach()
        self._z_curr = wm_step.z.detach()
        self._absorb_env_step(next_meta)

        # 8. Training (after warmup).
        env_step_now = env_step_meta.env_step
        if (
            env_step_now >= self._config.warmup_env_steps
            and env_step_now % self._config.train_every_n_env_steps == 0
        ):
            self._train_step()

        # 9. Waking-planning rollout at cadence. This is the Probe-1.5
        #    calibration handshake (current-state-seeded, actor-driven, "0.2.0",
        #    sequence_self_prediction None) — the Phase 3 visibility-smoke's
        #    waking-planning *control*, NOT the four-axis offline dream (which is
        #    state-machine-driven, fires only on a desktop_off edge, and lives in
        #    the dreaming branch of the supervisor). Reframed in Phase 8a (Fork A
        #    option iii): renamed to mark it as waking-planning, behavior
        #    unchanged. It fires only in the waking state, which is the only
        #    state that runs _step_once.
        dream_emitted = (
            env_step_now > 0
            and env_step_now % self._config.dream_cadence_env_steps == 0
        )
        if dream_emitted:
            self._emit_waking_planning_rollout_for_phase3_control(env_step_now)

        # 10. Probe 4.5 S-DEC — scheduled decoder-head maintenance at the
        #     pre-registered checkpoint-aligned cadence (prereg §3), at the
        #     step boundary and BEFORE the checkpoint commit so the committed
        #     checkpoint carries the refit head (a resume continues from
        #     post-refit state). The cadence is never behavior-triggered; a
        #     §3 honesty-STOP propagates as ``HonestyStopError`` and ends the
        #     run — the orchestrating script records the finding.
        maintenance = self._config.maintenance_refit
        if (
            maintenance is not None
            and env_step_now > 0
            and env_step_now % maintenance.refit_every_n_env_steps == 0
        ):
            run_scheduled_maintenance(
                self._world_model,
                self._actor,
                maintenance,
                env_step=env_step_now,
                run_label=self._config.run_id,
            )

        # 11. Checkpoint at cadence.
        checkpoint_committed = (
            env_step_now > 0
            and env_step_now % self._config.checkpoint_every_n_env_steps == 0
        )
        if checkpoint_committed:
            self._commit_checkpoint(env_step_now)

        # 12. Optional external observer hook. Last so the callback sees
        #     the post-dream / post-checkpoint state — the same envelope
        #     that just got attached to the next AgentStep record.
        if self._step_callback is not None:
            self._step_callback(
                RunnerStepInfo(
                    env_step=agent_step_record.t,
                    episode_id=agent_step_record.episode_id,
                    step_in_episode=agent_step_record.step_in_episode,
                    kl_aggregate=agent_step_record.kl_aggregate_t,
                    recon_loss=agent_step_record.recon_loss_t,
                    intrinsic_signal=agent_step_record.intrinsic_signal_t,
                    policy_entropy=agent_step_record.policy_entropy_t,
                    dream_emitted=dream_emitted,
                    checkpoint_committed=checkpoint_committed,
                    checkpoint_id=self._checkpoint_id,
                )
            )

        self._iteration += 1

    # ---- training -------------------------------------------------------

    def _train_step(self) -> None:
        """Sample a batch and step the three optimizers.

        Returns silently if the buffer doesn't have enough valid windows
        (which can happen during warmup if episodes are short or windows
        long). The training cadence is gated by ``warmup_env_steps`` in
        :meth:`_step_once`, but the buffer's window count is the harder
        constraint at small scales.
        """
        bs = self._config.replay_batch_size
        if not self._replay.can_sample(bs):
            return
        batch, _meta = self._replay.sample(bs)

        wm_cfg = self._config.world_model_config
        device = self._device
        L = self._config.replay_sequence_length

        obs_seq = batch.obs.to(device, non_blocking=True)  # (B, L, 1, 32, 32)
        action_seq = batch.action.to(device, non_blocking=True)  # (B, L) long
        # Probe 3.5: per-step sensed energy (B, L) — fused into the posterior
        # and the energy reconstruction target at each step below.
        sensed_energy_seq = batch.sensed_energy.to(device, non_blocking=True)
        B = obs_seq.shape[0]

        h = torch.zeros(B, wm_cfg.h_dim, device=device)
        z = torch.zeros(B, wm_cfg.z_dim, device=device)
        a_prev = torch.zeros(B, dtype=torch.long, device=device)

        h_outs: list[Tensor] = []
        z_outs: list[Tensor] = []
        wm_total = torch.zeros((), device=device)

        target_mode = self._config.self_prediction_target_mode
        lambda_self = self._config.lambda_self

        for t in range(L):
            obs_t = obs_seq[:, t]  # (B, 1, 32, 32)
            # Environmental mode needs the next observation as the
            # supervisory target. For the last position in the sampled
            # sequence we fall back to ``obs_t`` (the same observation
            # the head's prediction was paired against at env-step
            # time): a clean closed-form is held for Phase 5 / future
            # passes that have access to the buffer's true (t+1)
            # transition.
            if target_mode == "environmental":
                if t + 1 < L:
                    next_obs_for_target_b: Tensor | None = obs_seq[:, t + 1]
                else:
                    next_obs_for_target_b = obs_t
            else:
                next_obs_for_target_b = None
            sensed_energy_t_b = sensed_energy_seq[:, t]  # (B,)
            wm_step = self._world_model.step(
                obs_t,
                h,
                z,
                a_prev,
                next_obs=next_obs_for_target_b,
                sensed_energy=sensed_energy_t_b,
            )
            target_h_next_b = self._world_model.compute_self_prediction_target(
                obs_t, h, z, a_prev, next_obs=next_obs_for_target_b
            )
            # Plan §2.4 batched-target masking decision (Phase 1 journal
            # guess + synthesis §3 v2 silence): the batched auxiliary
            # loss includes first-step targets *unmasked*. The head's
            # training shouldn't be hampered by mask-the-scalar being a
            # downstream concern; the mask is for the actor-readable
            # PolicyView scalar only. Phase 3 journal records the
            # decision and its rationale.
            loss_dict = self._world_model.loss(
                wm_step,
                obs_target=obs_t,
                target_h_next=target_h_next_b,
                sensed_energy_target=sensed_energy_t_b,
            )
            wm_total = (
                wm_total
                + loss_dict["total"]
                + lambda_self * loss_dict["self_prediction_loss"]
            )
            h_outs.append(wm_step.h)
            z_outs.append(wm_step.z)
            h = wm_step.h
            z = wm_step.z
            a_prev = action_seq[:, t]

        # World model backward.
        self._wm_opt.zero_grad(set_to_none=True)
        wm_total.backward()  # type: ignore[no-untyped-call]
        self._wm_opt.step()
        # EMA target update — plan §2.4: after the world-model optimizer
        # step, advance the EMA-tracked encoder and GRU parameters by
        # one convex-combination step toward the freshly-updated online
        # parameters. The arithmetic is in-place on the EMA target's
        # underlying storage (no gradient graph entered).
        self._world_model._update_ema_target()

        # Stack and detach for the ensemble + actor losses (so neither
        # back-propagates into the world model's parameters).
        h_stack = torch.stack(h_outs, dim=1).detach()  # (B, L, h_dim)
        z_stack = torch.stack(z_outs, dim=1).detach()  # (B, L, z_dim)

        # ---- ensemble loss ----
        if L >= 2:
            h_t = h_stack[:, :-1].reshape(-1, wm_cfg.h_dim)
            z_t = z_stack[:, :-1].reshape(-1, wm_cfg.z_dim)
            a_t = action_seq[:, :-1].reshape(-1)
            z_target = z_stack[:, 1:].reshape(-1, wm_cfg.z_dim)
            ens_loss_dict = self._ensemble.compute_loss(h_t, z_t, a_t, z_target)
            self._ens_opt.zero_grad(set_to_none=True)
            ens_loss_dict["loss"].backward()  # type: ignore[no-untyped-call]
            self._ens_opt.step()

        # ---- actor loss (analytic gradients through imagination) ----
        h_0 = h_stack[:, 0]
        z_0 = z_stack[:, 0]
        actor_loss_dict = self._actor.imagine_and_compute_loss(
            self._world_model,
            self._ensemble,
            h_0=h_0,
            z_0=z_0,
            horizon=self._config.imagination_horizon,
            energy_preference=self._config.energy_preference,
        )
        self._actor_opt.zero_grad(set_to_none=True)
        actor_loss_dict["actor_loss"].backward()  # type: ignore[no-untyped-call]
        self._actor_opt.step()

        # Probe 3.5 Phase 2: cache the per-training-step pragmatic/epistemic
        # decomposition for the AgentStep writer (telemetry only).
        self._last_pragmatic_value = float(
            actor_loss_dict["mean_pragmatic_value"].detach().item()
        )
        self._last_epistemic_value = float(
            actor_loss_dict["mean_disagreement"].detach().item()
        )
        self._last_pragmatic_share = float(
            actor_loss_dict["pragmatic_share"].detach().item()
        )

    # ---- dream rollout --------------------------------------------------

    def _emit_waking_planning_rollout_for_phase3_control(self, env_step: int) -> None:
        """Emit one waking-planning ``DreamRollout`` (Probe-1.5 handshake).

        Renamed in Phase 8a (Fork A option iii) from ``_emit_dream`` — the name
        ``dream`` was a Probe-1 misnomer for what is, on inspection, a *waking*
        rollout: current-state-seeded (``_h_curr`` / ``_z_curr``), actor-driven
        (goal-coupled), ``"0.2.0"``, ``sequence_self_prediction is None``, with
        the environment live. It is the Phase 3 visibility-smoke's
        *waking-planning control* (the four-axis offline dream is differentiated
        *from* it). Behavior is unchanged from the pre-8a ``_emit_dream``; only
        the name and call-site framing changed. The four-axis dream lives in the
        supervisor's dreaming branch.
        """
        if self._h_curr is None or self._z_curr is None:
            return
        horizon = self._config.dream_horizon
        wm_cfg = self._config.world_model_config

        with torch.no_grad():
            h = self._h_curr.detach()
            z = self._z_curr.detach()
            seed_h0 = h.squeeze(0).detach().cpu().tolist()
            seed_z0 = z.squeeze(0).detach().cpu().tolist()

            sequence_h: list[list[float]] = []
            sequence_z_prior: list[list[float]] = []
            sequence_action: list[int] = []
            sequence_action_logprob: list[float] = []
            sequence_prior_entropy: list[float] = []
            sequence_decoded_obs: list[bytes] = []

            cumulative_prior_entropy = 0.0
            kls: list[float] = []
            prev_mu: Tensor | None = None
            prev_log_sigma: Tensor | None = None
            max_norm_change = 0.0
            prev_z = z

            # Probe 1.5 v2: dream-rollout PolicyView gets a zero scalar
            # (the synthesis §1.5 commits Probe 1.5 to running the head
            # only during waking; dream rollouts feed zero through the
            # actor's new column, the same mask-via-zero-feed convention
            # the imagination loss uses). Allocated once and reused across
            # the dream horizon.
            dream_sp_error = torch.zeros((), device=self._device)
            for _ in range(horizon):
                view = PolicyView(
                    h=h, z=z, self_prediction_error=dream_sp_error
                )
                action_output = self._actor.forward(view)
                a = action_output.action  # (1,) long
                sequence_action.append(int(a.item()))
                sequence_action_logprob.append(float(action_output.logprob.item()))

                h_next = self._world_model.recurrence(h, z, a)
                mu, log_sigma = self._world_model.prior(h_next)
                sigma = torch.exp(log_sigma)
                dist = torch.distributions.Normal(mu, sigma)
                z_next = cast(Tensor, dist.sample())  # type: ignore[no-untyped-call]

                # Decode for telemetry: clamp [0, 1] (recon is unbounded
                # but the world model trains against [0, 1] obs targets);
                # quantise to uint8 for compact bytes.
                decoded = self._world_model.decode(h_next, z_next)
                decoded_uint8 = (
                    decoded.squeeze(0)
                    .squeeze(0)
                    .clamp(0.0, 1.0)
                    .mul(255.0)
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
                sequence_decoded_obs.append(decoded_uint8.tobytes())

                # Per-step prior entropy of the diagonal Gaussian over z_dim.
                step_entropy = float(
                    cast(Tensor, dist.entropy()).sum(dim=-1).mean().item()  # type: ignore[no-untyped-call]
                )
                sequence_prior_entropy.append(step_entropy)
                cumulative_prior_entropy += step_entropy

                # KL between successive priors (analytic for diagonal Gaussian).
                if prev_mu is not None and prev_log_sigma is not None:
                    kl = self._diag_gaussian_kl(
                        prev_mu, prev_log_sigma, mu, log_sigma
                    )
                    kls.append(float(kl.item()))
                prev_mu = mu
                prev_log_sigma = log_sigma

                norm_change = float((z_next - prev_z).norm().item())
                if norm_change > max_norm_change:
                    max_norm_change = norm_change
                prev_z = z_next

                sequence_h.append(h_next.squeeze(0).detach().cpu().tolist())
                sequence_z_prior.append(z_next.squeeze(0).detach().cpu().tolist())

                h = h_next
                z = z_next

        mean_kl = sum(kls) / len(kls) if kls else 0.0
        record = DreamRollout(
            schema_version=SCHEMA_VERSION,
            run_id=self._config.run_id,
            checkpoint_id=self._checkpoint_id,
            seed_step=env_step,
            seed_h0=seed_h0,
            seed_z0=seed_z0,
            sequence_h=sequence_h,
            sequence_z_prior=sequence_z_prior,
            sequence_action=sequence_action,
            sequence_action_logprob=sequence_action_logprob,
            sequence_prior_entropy=sequence_prior_entropy,
            sequence_decoded_obs=sequence_decoded_obs,
            cumulative_prior_entropy=cumulative_prior_entropy,
            mean_step_kl_successive_priors=mean_kl,
            max_step_latent_norm_change=max_norm_change,
            # Synthesis §1.5 v2: the head runs only during waking at
            # Probe 1.5; dream rollouts emit ``None`` here. The slot is
            # the Probe 3 forward-compatibility hook (per the
            # framing journaled at ``docs/workingjournal/pre-probe3.md``).
            sequence_self_prediction=None,
        )
        self._dream_sink.write(record)
        # Suppress unused-variable lint on the wm_cfg pulled for symmetry
        # with other branches; the dream rollout loop reads h_dim only via
        # tensor shapes, so the local isn't accessed by name.
        _ = wm_cfg

    @staticmethod
    def _diag_gaussian_kl(
        mu1: Tensor, log_sigma1: Tensor, mu2: Tensor, log_sigma2: Tensor
    ) -> Tensor:
        """Analytic KL(N(mu1, sigma1) || N(mu2, sigma2)) for diagonal Gaussians.

        Result is the sum over the latent dim, mean over the (singleton)
        batch — a single scalar.
        """
        var1 = torch.exp(2 * log_sigma1)
        var2 = torch.exp(2 * log_sigma2)
        kl_per_dim = (
            log_sigma2
            - log_sigma1
            + (var1 + (mu1 - mu2) ** 2) / (2 * var2)
            - 0.5
        )
        return kl_per_dim.sum(dim=-1).mean()

    # ---- AgentStep emission --------------------------------------------

    def _drain_perturbations(self, env_step: int) -> None:
        """Probe 4 Phase 2 (plan §S-PERT): fire due perturbations at the
        step boundary.

        The generator is polled with the post-step world state (grid,
        Io's position) plus the consumption co-timing flag, detected from
        consecutive true-energy samples with the house jump threshold
        (``observer.source_events``) — a consumption at this boundary
        defers the generator's event (the not-self-in-time rule; the
        generator re-attempts next boundary). Planned cells fire through
        the co-located env-server's ``add_resource`` with
        ``trigger="generator"``. The manual-trigger inbox drains at the
        same boundary with ``trigger="manual"``; inbox failures become
        archived error results, never runner exceptions.
        """
        assert self._env_server is not None
        state = self._env_server.grid_world_state
        current_true_energy = float(state.true_energy)
        previous = self._perturbation_prev_true_energy
        consumed_last_step = (
            previous is not None
            and (current_true_energy - previous) > CONSUMPTION_JUMP_THRESHOLD
        )
        self._perturbation_prev_true_energy = current_true_energy

        if self._perturbation_generator is not None:
            planned = self._perturbation_generator.poll(
                env_step=env_step,
                grid=state.grid,
                agent_pos=state.agent_pos,
                consumed_last_step=consumed_last_step,
            )
            if planned is not None:
                for cell in planned.cells:
                    self._env_server.add_resource(cell, trigger="generator")

        if self._config.perturbation_inbox_dir is not None:
            drain_trigger_inbox(
                self._config.perturbation_inbox_dir, self._env_server
            )

    def _emit_agent_step(
        self,
        *,
        env_step_meta: EnvStep,
        wm_step: WorldModelStep,
        obs_t_dev: Tensor,
        action_output: ActionOutput,
        intrinsic: Tensor,
        obs_hash: str,
        self_prediction_error: Tensor,
        self_prediction_error_masked: bool,
        sensed_energy_value: float,
        true_energy_value: float | None,
    ) -> AgentStep:
        """Build and write one AgentStep record. Returns the record so
        :meth:`_step_once` can feed scalar fields into ``step_callback``
        without recomputing them.

        Tensors are detached and moved to CPU before ``.tolist()`` /
        ``.item()`` so the telemetry path holds no gradient or device
        memory and can be flushed independently of the training step.
        """
        # Strip batch dim (B=1) and lift to plain Python types.
        h_t_list = wm_step.h.squeeze(0).detach().cpu().tolist()
        z_t_list = wm_step.z.squeeze(0).detach().cpu().tolist()
        q_mu, q_log_sigma = wm_step.q_params
        p_mu, p_log_sigma = wm_step.p_params
        q_params_pair = (
            q_mu.squeeze(0).detach().cpu().tolist(),
            q_log_sigma.squeeze(0).detach().cpu().tolist(),
        )
        p_params_pair = (
            p_mu.squeeze(0).detach().cpu().tolist(),
            p_log_sigma.squeeze(0).detach().cpu().tolist(),
        )
        kl_per_dim_list = wm_step.kl_per_dim.squeeze(0).detach().cpu().tolist()
        kl_aggregate = float(sum(kl_per_dim_list))
        # Recon loss: MSE between the world model's reconstruction and
        # the actual observation, summed over pixels (B=1 so .mean() is
        # just an identity scaling).
        with torch.no_grad():
            diff = wm_step.recon - obs_t_dev
            recon_loss = float((diff**2).flatten().sum().item())

        # Probe 1.5 v2 (plan §2.4): the three new fields are required-
        # non-None for ``"0.2.0"`` records (the AgentStep validator
        # enforces this). The runner is the sole AgentStep writer at
        # this phase; the scalar arrives as a 0-dim tensor (or shape
        # ``(1,)`` after batched-path expansion), which ``.item()``
        # lifts to a plain float regardless.
        self_prediction_t = (
            wm_step.self_prediction.squeeze(0).detach().cpu().tolist()
        )
        self_prediction_error_t = float(
            self_prediction_error.detach().cpu().reshape(-1)[0].item()
        )

        # Probe 3.5 energy telemetry. When enabled (a co-located env_server is
        # present so ``true_energy`` is reachable), stamp the new record version
        # and populate all four energy fields; otherwise emit the legacy
        # "0.2.0" record with the energy fields None (byte-identical to pre-3.5).
        # ``energy_pred_t`` is the world model's decoded scalar; the recon error
        # is the per-step squared error against the *sensed* target (never true).
        energy_pred_t: float | None = None
        energy_recon_error_t: float | None = None
        sensed_energy_t: float | None = None
        true_energy_t: float | None = None
        pragmatic_value_t: float | None = None
        epistemic_value_t: float | None = None
        pragmatic_share_t: float | None = None
        schema_version = SCHEMA_VERSION
        if self._energy_telemetry_enabled:
            assert true_energy_value is not None
            energy_pred_val = float(
                wm_step.energy_pred.detach().cpu().reshape(-1)[0].item()
            )
            sensed_energy_t = sensed_energy_value
            true_energy_t = true_energy_value
            energy_pred_t = energy_pred_val
            energy_recon_error_t = (energy_pred_val - sensed_energy_value) ** 2
            schema_version = PROBE_3_5_TELEMETRY_SCHEMA_VERSION
            # Probe 3.5 Phase 2: with a preference configured, stamp the
            # Phase-2 record version and the per-training-step
            # pragmatic/epistemic decomposition (the most recent training
            # step's values; genuinely zero before the first one).
            if self._config.energy_preference is not None:
                pragmatic_value_t = self._last_pragmatic_value
                epistemic_value_t = self._last_epistemic_value
                pragmatic_share_t = self._last_pragmatic_share
                schema_version = PROBE_3_5_PHASE2_TELEMETRY_SCHEMA_VERSION

        record = AgentStep(
            schema_version=schema_version,
            run_id=self._config.run_id,
            checkpoint_id=self._checkpoint_id,
            t=env_step_meta.env_step,
            episode_id=env_step_meta.episode_id,
            step_in_episode=env_step_meta.step_in_episode,
            wallclock_ms=time.monotonic_ns() // 1_000_000,
            h_t=h_t_list,
            q_params_t=q_params_pair,
            p_params_t=p_params_pair,
            z_t=z_t_list,
            kl_per_dim_t=kl_per_dim_list,
            kl_aggregate_t=kl_aggregate,
            recon_loss_t=recon_loss,
            action_t=int(action_output.action.item()),
            action_logprob_t=float(action_output.logprob.item()),
            policy_entropy_t=float(action_output.entropy.item()),
            obs_hash_t=obs_hash,
            intrinsic_signal_t=float(intrinsic.squeeze().item()),
            encoder_embedding_t=wm_step.embed.squeeze(0).detach().cpu().tolist(),
            self_prediction_t=self_prediction_t,
            self_prediction_error_t=self_prediction_error_t,
            self_prediction_error_masked_t=bool(self_prediction_error_masked),
            sensed_energy_t=sensed_energy_t,
            true_energy_t=true_energy_t,
            energy_pred_t=energy_pred_t,
            energy_recon_error_t=energy_recon_error_t,
            pragmatic_value_t=pragmatic_value_t,
            epistemic_value_t=epistemic_value_t,
            pragmatic_share_t=pragmatic_share_t,
        )
        self._agent_step_sink.write(record)
        return record

    # ---- checkpoint commit ---------------------------------------------

    def _commit_checkpoint(self, env_step: int) -> None:
        """Stage runtime artifacts to a temp dir, then commit atomically."""
        self._checkpoint_counter += 1
        checkpoint_id = self._format_checkpoint_id(self._checkpoint_counter)

        # Item 6: mark the checkpoint-window open while staging/committing, so a
        # supervisor tick that observes it during the window would route a dream
        # session away from the atomic state-sync boundary. In the single-process
        # synchronous supervisor the tick never runs *during* this method (it is
        # called from within _step_once, between ticks), so the dreaming path
        # reads this as False — the path is live (no hardcoded constant), and the
        # flag becomes meaningful in a real Mac/desktop split.
        self._checkpoint_in_progress = True
        try:
            self._commit_checkpoint_inner(checkpoint_id, env_step)
        finally:
            self._checkpoint_in_progress = False

    def _commit_offline_checkpoint(self) -> None:
        """Commit an offline-period checkpoint (Phase 8c) — barrier-free, with the
        full resumable state incl. the metabolic bucket + controller state.

        Called from the supervisor loop's dormant branch at a clean between-
        sessions boundary. The env is off, so no env-server barrier (the canonical
        *mind* state is all that needs saving); the write is atomic temp-then-
        rename so a crash mid-write cannot corrupt the committed checkpoint.
        """
        self._checkpoint_counter += 1
        checkpoint_id = self._format_checkpoint_id(self._checkpoint_counter)
        with tempfile.TemporaryDirectory(prefix="kind_ckpt_stage_") as staging_str:
            staging = Path(staging_str)
            contents = self._build_checkpoint_contents(
                staging, include_offline_state=True
            )
            self._checkpoint_manager.commit_offline(checkpoint_id, contents)
        self._after_commit(checkpoint_id)

    def _offline_state_dict(self) -> dict[str, Any]:
        """The offline-period resume state: the metabolic bucket snapshot (frozen-
        across-pause on restore) + the controller's current state. Both content-
        blind (numbers + a state name)."""
        assert self._metabolic_budget is not None
        assert self._controller is not None
        return {
            "metabolic_budget": self._metabolic_budget.snapshot(),
            "controller_state": self._controller.current_state,
        }

    def _after_commit(self, checkpoint_id: str) -> None:
        """Post-commit: propagate the new checkpoint id into envelope producers."""
        self._checkpoint_id = checkpoint_id
        self._replay.set_checkpoint_id(checkpoint_id)
        if self._env_server is not None:
            self._env_server.set_checkpoint_id(checkpoint_id)

    def _commit_checkpoint_inner(self, checkpoint_id: str, env_step: int) -> None:
        with tempfile.TemporaryDirectory(prefix="kind_ckpt_stage_") as staging_str:
            staging = Path(staging_str)
            contents = self._build_checkpoint_contents(
                staging, include_offline_state=False
            )
            self._checkpoint_manager.commit(checkpoint_id, contents)
        self._after_commit(checkpoint_id)
        # `env_step` is a Probe 1 informational input to the journal / eyeball
        # helpers; the runner doesn't persist it separately.
        _ = env_step

    def _build_checkpoint_contents(
        self, staging: Path, *, include_offline_state: bool
    ) -> CheckpointContents:
        """Stage all checkpoint artifacts into ``staging`` and return the contents.

        Shared by the waking commit (``include_offline_state=False`` — resuming
        from waking correctly uses the defaults: waking + full bucket) and the
        offline commit (``include_offline_state=True`` — captures the metabolic
        bucket + controller state so the offline rhythm survives a pause)."""
        if True:
            # Combined weights file: one prefix per module so load_checkpoint
            # can split back into three state dicts.
            combined: dict[str, Tensor] = {}
            for k, v in self._world_model.state_dict().items():
                combined[f"world_model.{k}"] = v.detach().contiguous().cpu()
            for k, v in self._actor.state_dict().items():
                combined[f"actor.{k}"] = v.detach().contiguous().cpu()
            for k, v in self._ensemble.state_dict().items():
                combined[f"ensemble.{k}"] = v.detach().contiguous().cpu()
            weights_path = staging / "weights.safetensors"
            save_file(combined, str(weights_path))

            # Optimizer states: a single torch.save with one dict per opt.
            optimizer_state_path = staging / "optimizer_state.pt"
            torch.save(
                {
                    "world_model": self._wm_opt.state_dict(),
                    "actor": self._actor_opt.state_dict(),
                    "ensemble": self._ens_opt.state_dict(),
                },
                optimizer_state_path,
            )

            # RNG state + agent runtime state, all in one pickle so resume
            # can restore both atomically. The runtime state covers
            # h_prev/z_prev/a_prev, the latest EnvStep meta, and the
            # buffer's event counters.
            rng_state_path = staging / "rng_state.pkl"
            with open(rng_state_path, "wb") as fh:
                pickle.dump(self._save_rng_state(), fh)

            # Telemetry offsets: descriptive only at Probe 1 — no resume
            # of telemetry continuity is implemented yet.
            telemetry_offsets_path = staging / "telemetry_offsets.json"
            telemetry_offsets_path.write_text(
                json.dumps(self._telemetry_offsets(), indent=2) + "\n"
            )

            # Schema version file. Probe 1.5 v2 (plan §2.4 step 3 of
            # checkpoint commit deltas): migrates from
            # ``PROBE_1_SCHEMA_VERSION`` to ``SCHEMA_VERSION``. The EMA
            # target weights and the actor's extended input layer ride
            # inside ``weights.safetensors`` automatically — the EMA
            # target is part of ``world_model.state_dict()`` (via the
            # ``target_encoder.*`` / ``target_gru_cell.*`` /
            # optionally ``_frozen_projection`` / ``_environmental_projection``
            # prefixes), and the extended actor input layer is one
            # tensor inside ``actor.net.0.weight`` with one additional
            # column — both prefixed and serialised by the existing
            # ``world_model.`` and ``actor.`` blob-prefix code above.
            schema_version_path = staging / "schema_version.txt"
            schema_version_path.write_text(SCHEMA_VERSION + "\n")

            # Replay meta file: descriptive snapshot of buffer counters.
            # The buffer itself is not persisted at Probe 1 (in-memory only).
            replay_meta_path = staging / "replay_meta.json"
            replay_meta_path.write_text(
                json.dumps(self._replay_snapshot(), indent=2) + "\n"
            )

            # Phase 8c — the optional offline-period state (bucket + controller
            # state), captured only on offline checkpoints.
            offline_state_path: Path | None = None
            if include_offline_state:
                offline_state_path = staging / "offline_state.json"
                offline_state_path.write_text(
                    json.dumps(self._offline_state_dict(), indent=2) + "\n"
                )

            return CheckpointContents(
                weights_path=weights_path,
                replay_meta_path=replay_meta_path,
                optimizer_state_path=optimizer_state_path,
                rng_state_path=rng_state_path,
                telemetry_offsets_path=telemetry_offsets_path,
                schema_version_path=schema_version_path,
                offline_state_path=offline_state_path,
            )

    # ---- runtime helpers -----------------------------------------------

    def _absorb_env_step(self, env_step: EnvStep) -> None:
        """Update obs_curr / env_step_meta from a transport response."""
        self._obs_curr = self._obs_to_cpu_tensor(env_step.observation)
        self._obs_curr_hash = self._hash_observation(env_step.observation)
        self._env_step_meta = env_step

    def _init_runtime_zero_state(self) -> None:
        """Initialise (h_prev, z_prev, a_prev) to zeros for episode 0 / start."""
        wm_cfg = self._config.world_model_config
        self._h_prev = torch.zeros(1, wm_cfg.h_dim, device=self._device)
        self._z_prev = torch.zeros(1, wm_cfg.z_dim, device=self._device)
        self._a_prev = torch.zeros(1, dtype=torch.long, device=self._device)
        self._h_curr = self._h_prev.clone()
        self._z_curr = self._z_prev.clone()

    def _init_runtime_zero_state_keep_obs(self) -> None:
        """Reset (h_prev, z_prev, a_prev) on episode boundary; leave obs alone."""
        wm_cfg = self._config.world_model_config
        self._h_prev = torch.zeros(1, wm_cfg.h_dim, device=self._device)
        self._z_prev = torch.zeros(1, wm_cfg.z_dim, device=self._device)
        self._a_prev = torch.zeros(1, dtype=torch.long, device=self._device)

    @staticmethod
    def _obs_to_cpu_tensor(obs: NDArray[np.uint8]) -> Tensor:
        """Convert numpy uint8 (H, W) → float32 (1, H, W) tensor in [0, 1].

        The ``np.array(obs)`` makes the array writable so ``torch.from_numpy``
        does not warn about non-writable backing buffers (the transport
        decode returns a read-only view).
        """
        arr = np.array(obs, dtype=np.uint8, copy=True)
        tensor = torch.from_numpy(arr).float() / 255.0
        return tensor.unsqueeze(0)  # (1, H, W)

    @staticmethod
    def _hash_observation(obs: NDArray[np.uint8]) -> str:
        return hashlib.sha256(obs.tobytes()).hexdigest()

    # ---- RNG / runtime state save / load -------------------------------

    def _save_rng_state(self) -> dict[str, Any]:
        """Snapshot all RNG sources + the runner's runtime state.

        Includes Python ``random``, NumPy default RNG, PyTorch CPU RNG,
        the runner's sample-RNG (the ``torch.Generator`` that drives
        replay sampling), the device-specific RNG when running on MPS or
        CUDA, plus the runtime tuple (h_prev / z_prev / a_prev / latest
        EnvStep / iteration counter).
        """
        device = self._device
        device_rng_state: bytes | None = None
        if device.type == "cuda":
            device_rng_state = torch.cuda.get_rng_state().numpy().tobytes()
        elif device.type == "mps":
            device_rng_state = torch.mps.get_rng_state().numpy().tobytes()

        # The runtime tensors live on `device`; persist as CPU tensors for
        # portability across resume to a different device.
        runtime: dict[str, Any] = {
            "h_prev": (
                self._h_prev.detach().cpu().clone()
                if self._h_prev is not None
                else None
            ),
            "z_prev": (
                self._z_prev.detach().cpu().clone()
                if self._z_prev is not None
                else None
            ),
            "a_prev": (
                self._a_prev.detach().cpu().clone()
                if self._a_prev is not None
                else None
            ),
            "h_curr": (
                self._h_curr.detach().cpu().clone()
                if self._h_curr is not None
                else None
            ),
            "z_curr": (
                self._z_curr.detach().cpu().clone()
                if self._z_curr is not None
                else None
            ),
            "obs_curr": (
                self._obs_curr.detach().cpu().clone()
                if self._obs_curr is not None
                else None
            ),
            "obs_curr_hash": self._obs_curr_hash,
            "env_step_meta": self._env_step_meta_to_dict(),
            "iteration": self._iteration,
        }

        return {
            "python_random": random.getstate(),
            "numpy_random": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().numpy().tobytes(),
            "torch_device_type": device.type,
            "torch_device_rng": device_rng_state,
            "sample_rng": self._sample_rng.get_state().numpy().tobytes(),
            "runtime": runtime,
        }

    def _load_rng_state(self, blob: dict[str, Any]) -> None:
        random.setstate(blob["python_random"])
        np.random.set_state(blob["numpy_random"])
        torch.set_rng_state(
            torch.frombuffer(bytearray(blob["torch_cpu"]), dtype=torch.uint8)
        )
        if blob.get("sample_rng") is not None:
            self._sample_rng.set_state(
                torch.frombuffer(
                    bytearray(blob["sample_rng"]), dtype=torch.uint8
                )
            )
        device_type = blob.get("torch_device_type")
        device_rng = blob.get("torch_device_rng")
        if device_rng is not None:
            tensor = torch.frombuffer(bytearray(device_rng), dtype=torch.uint8)
            if device_type == "cuda" and torch.cuda.is_available():
                torch.cuda.set_rng_state(tensor)
            elif device_type == "mps" and torch.backends.mps.is_available():
                torch.mps.set_rng_state(tensor)

        runtime = blob["runtime"]
        h_prev_cpu = runtime["h_prev"]
        z_prev_cpu = runtime["z_prev"]
        a_prev_cpu = runtime["a_prev"]
        h_curr_cpu = runtime["h_curr"]
        z_curr_cpu = runtime["z_curr"]
        obs_curr_cpu = runtime["obs_curr"]

        self._h_prev = (
            h_prev_cpu.to(self._device) if h_prev_cpu is not None else None
        )
        self._z_prev = (
            z_prev_cpu.to(self._device) if z_prev_cpu is not None else None
        )
        self._a_prev = (
            a_prev_cpu.to(self._device) if a_prev_cpu is not None else None
        )
        self._h_curr = (
            h_curr_cpu.to(self._device) if h_curr_cpu is not None else None
        )
        self._z_curr = (
            z_curr_cpu.to(self._device) if z_curr_cpu is not None else None
        )
        self._obs_curr = obs_curr_cpu  # CPU is correct for the runner state.
        self._obs_curr_hash = runtime["obs_curr_hash"]
        meta_dict = runtime["env_step_meta"]
        self._env_step_meta = (
            self._env_step_meta_from_dict(meta_dict)
            if meta_dict is not None
            else None
        )
        self._iteration = int(runtime["iteration"])

    def _env_step_meta_to_dict(self) -> dict[str, Any] | None:
        if self._env_step_meta is None:
            return None
        return {
            "observation_shape": list(self._env_step_meta.observation.shape),
            "observation_dtype": str(self._env_step_meta.observation.dtype),
            "observation_bytes": self._env_step_meta.observation.tobytes(),
            "env_step": self._env_step_meta.env_step,
            "episode_id": self._env_step_meta.episode_id,
            "step_in_episode": self._env_step_meta.step_in_episode,
            "wallclock_ms": self._env_step_meta.wallclock_ms,
            "sensed_energy": self._env_step_meta.sensed_energy,
        }

    @staticmethod
    def _env_step_meta_from_dict(d: dict[str, Any]) -> EnvStep:
        shape = tuple(int(x) for x in d["observation_shape"])
        dtype = np.dtype(d["observation_dtype"])
        arr = np.frombuffer(d["observation_bytes"], dtype=dtype).reshape(shape)
        return EnvStep(
            observation=arr,
            env_step=int(d["env_step"]),
            episode_id=int(d["episode_id"]),
            step_in_episode=int(d["step_in_episode"]),
            wallclock_ms=int(d["wallclock_ms"]),
            sensed_energy=float(d.get("sensed_energy", 0.0)),
        )

    # ---- bookkeeping ---------------------------------------------------

    def _telemetry_offsets(self) -> dict[str, Any]:
        """Snapshot byte offsets / shard counts for each telemetry stream."""
        return {
            "agent_step_dir": str(self._config.telemetry_dir / _AGENT_STEP_DIR),
            "dream_rollout_dir": str(
                self._config.telemetry_dir / _DREAM_ROLLOUT_DIR
            ),
            "replay_meta_path": str(
                self._config.telemetry_dir / _REPLAY_META_FILE
            ),
            "world_event_path": str(
                self._config.telemetry_dir / _WORLD_EVENT_FILE
            ),
        }

    def _replay_snapshot(self) -> dict[str, Any]:
        return {
            "buffer_size": len(self._replay),
            "capacity": self._replay.capacity,
            "sequence_length": self._replay.sequence_length,
        }

    def _format_checkpoint_id(self, counter: int) -> str:
        return f"ckpt-{counter:0{self._config._checkpoint_id_zero_pad}d}"

    @staticmethod
    def _parse_checkpoint_counter(checkpoint_id: str) -> int:
        # Best-effort: strip "ckpt-" and parse digits. Anything unexpected
        # falls back to 0.
        if not checkpoint_id.startswith("ckpt-"):
            return 0
        suffix = checkpoint_id[len("ckpt-") :]
        try:
            return int(suffix)
        except ValueError:
            return 0

    # ---- introspection (test-side only) --------------------------------

    @property
    def world_model(self) -> WorldModel:
        return self._world_model

    @property
    def actor(self) -> Actor:
        return self._actor

    @property
    def ensemble(self) -> LatentDisagreementEnsemble:
        return self._ensemble

    @property
    def replay_buffer(self) -> SequenceReplayBuffer:
        return self._replay

    @property
    def checkpoint_manager(self) -> CheckpointManager:
        return self._checkpoint_manager

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def latest_checkpoint_id(self) -> str | None:
        return self._checkpoint_id

    # ---- context manager ----------------------------------------------

    def __enter__(self) -> "Runner":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        self.close()
        # Module-level imports kept above so static type checkers see the
        # call sites cleanly; suppress unused-import diagnostics for those
        # that turn out to be referenced only by string in docstrings.
        _ = (Callable, Final, os, shutil)
