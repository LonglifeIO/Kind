"""Phase 3a env-server harness.

Wraps Phase 2a's :class:`~kind.env.grid_world.GridWorld` with the harness
layer the Probe 1 implementation plan §2.4 specifies: the four named
mutators exposed as methods, ``WorldEvent`` records dispatched to a
caller-supplied handler, ``env_reset`` events at episode boundaries, and
per-episode aggregate records of internal stochasticity. The harness is a
synchronous, in-process Python interface — Phase 4 wraps it with TCP
transport. There is no network code, no async, no threading, no
scheduling, and no perturbation generator here. The four mutators are the
manual trigger affordance Probe 1 exposes; Probe 4's research will revisit
whether (and how) a stochastic generator complements the manual triggers.

**Structural intent.** The harness is sink-agnostic at Probe 1: every
``WorldEvent`` is dispatched to the ``world_event_handler`` callable from
:class:`EnvServerConfig`. Phase 1's :class:`~kind.observer.sinks.JsonlSink`
is a natural choice for the handler in Probe 1's loopback deployment (the
test passes ``sink.write`` as the handler). Phase 4's transport server
replaces the handler with one that ships ``WorldEvent`` records over the
wire to the Mac trainer; the handler indirection is the seam that lets the
desktop env-server stay sink-free, with the Mac trainer owning the
``world_event`` sink end-to-end.

**Wallclock.** Both the wrapped :class:`EnvStep` records and every
``WorldEvent`` capture ``wallclock_ms`` via ``time.monotonic_ns() //
1_000_000``. The harness overrides the ``EnvStep.wallclock_ms`` field
produced by ``GridWorld`` (which uses ``time.time()``) so all telemetry
emitted by the harness shares a single monotonic clock — non-decreasing
across consecutive records is the property the smoke and gate tests rely
on. Phase 4's transport preserves the wallclock as set by the harness; the
Mac side does not re-stamp on receipt.

**Builder vs. internal stochasticity.** The four mutators emit
``WorldEvent`` records with ``source="builder"`` and
``event_type="builder_perturbation"``. Internal stochasticity at Probe 1 is
logged at coarse aggregate per episode — one
``internal_stochasticity_aggregate`` record per episode boundary, carrying
the per-episode regrowth-event count, the mean drift step magnitude, and
the final ``p`` value. The asymmetry — full ground truth in the
``world_event`` stream, no marker in the agent's observation — is the
Probe 4 affordance the harness exists to provide.

**Per-event internal stochasticity (Probe 4 Phase 1, plan §S-CTRL).** When
``EnvServerConfig.emit_internal_stochasticity_events`` is on (default off —
legacy emission byte-identical), each regrowth resource-addition
(EMPTY→RESOURCE) additionally emits one granular ``WorldEvent``
(``source="environment"``, ``event_type="internal_stochasticity_event"``,
record version ``PROBE_4_WORLD_EVENT_SCHEMA_VERSION``) whose payload's
comparison keys (``cell`` / ``pre_state`` / ``post_state``) exactly match a
builder ``add_resource`` payload — the ENVIRONMENT class of the three-way
matched control (pre-registration §1), directly comparable per-event with
the BUILDER class. The per-episode aggregate is emitted unchanged either
way. Two inherited scope notes carry over from the aggregate's
diff-counting: builder mutations between steps are rolled into the pre-step
snapshot (never misattributed as internal events), and the episode-boundary
closing step's regrowth is not counted (grid reset before the diff; ~0.6
events per ~120-event episode at p=0.01, inside the noise floor).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from types import TracebackType
from typing import Any, Final, Self

import numpy as np
from numpy.typing import NDArray

from kind.env import mutators
from kind.env.grid_world import (
    CellType,
    EnvStep,
    GridState,
    GridWorld,
    GridWorldConfig,
)
from kind.observer.schemas import (
    PROBE_1_SCHEMA_VERSION,
    PROBE_4_WORLD_EVENT_SCHEMA_VERSION,
    WorldEvent,
)

__all__ = [
    "EnvServer",
    "EnvServerConfig",
    "WorldEventHandler",
]


# ---- handler type --------------------------------------------------------


WorldEventHandler = Callable[[WorldEvent], None]
"""Callable that consumes a single ``WorldEvent``.

The harness invokes the configured handler synchronously in the same
thread that called :meth:`EnvServer.start`, :meth:`EnvServer.step`, or one
of the four mutator methods. The handler is responsible for whatever
persistence the caller wants — Phase 1's ``JsonlSink.write`` is the
natural fit at Probe 1's loopback deployment; Phase 4's transport server
replaces the handler with one that ships records over the wire.
"""


# ---- configuration --------------------------------------------------------


@dataclass(frozen=True)
class EnvServerConfig:
    """Static configuration for an :class:`EnvServer` run.

    ``run_id`` and ``world_event_handler`` come from the runner's run
    layout; ``grid_world_config`` and ``seed`` are forwarded to the wrapped
    :class:`GridWorld`. Frozen so the harness cannot have its identity
    silently rewritten mid-run; the handler is mutable on the
    :class:`EnvServer` itself via :meth:`EnvServer.set_world_event_handler`,
    which Phase 4's transport server uses to redirect emissions over the
    wire after a client connects.
    """

    grid_world_config: GridWorldConfig
    seed: int
    world_event_handler: WorldEventHandler
    run_id: str
    # Probe 4 Phase 1 (plan §S-CTRL): when True, each regrowth
    # resource-addition emits one granular ``internal_stochasticity_event``
    # record (see module docstring). Default False keeps legacy emission
    # byte-identical — the house opt-in pattern.
    emit_internal_stochasticity_events: bool = False


# ---- the harness ---------------------------------------------------------


_BUILDER_PERTURBATION: Final[str] = "builder_perturbation"
_ENV_RESET: Final[str] = "env_reset"
_INTERNAL_STOCHASTICITY_AGGREGATE: Final[str] = "internal_stochasticity_aggregate"
_INTERNAL_STOCHASTICITY_EVENT: Final[str] = "internal_stochasticity_event"
_SOURCE_BUILDER: Final[str] = "builder"
_SOURCE_ENVIRONMENT: Final[str] = "environment"

# Probe 4 Phase 2 (plan §S-PERT / prereg §1): the stratification tag on
# builder perturbations — payload["trigger"] ∈ {"generator", "manual"}.
# ``None`` (the default on every mutator method) emits the legacy payload
# with no tag, keeping all pre-Probe-4 callers byte-identical.
_VALID_TRIGGERS: Final[frozenset[str]] = frozenset({"generator", "manual"})


class EnvServer:
    """Synchronous, in-process harness around a :class:`GridWorld`.

    Lifecycle: construct, then call :meth:`start` to emit the initial
    ``env_reset`` ``WorldEvent`` and obtain the first :class:`EnvStep`;
    call :meth:`step` to advance the env (with ``WorldEvent`` records
    emitted at episode boundaries); call any of the four mutator methods
    between :meth:`step` calls to inject builder perturbations; call
    :meth:`close` to release resources. The harness is also a context
    manager; the ``with`` block calls :meth:`close` on exit but does not
    implicitly call :meth:`start` — tests and the runner call
    :meth:`start` explicitly to get the first ``EnvStep``.

    All wallclocks (the wrapped :class:`EnvStep` and every emitted
    ``WorldEvent``) come from a single ``time.monotonic_ns()`` source, which
    guarantees non-decreasing values across the harness's telemetry.
    """

    def __init__(self, config: EnvServerConfig) -> None:
        self._config = config
        self._world_event_handler: WorldEventHandler = config.world_event_handler
        self._grid_world: GridWorld | None = None
        self._checkpoint_id: str | None = None

        self._started = False
        self._closed = False

        # Per-episode aggregate accumulators. Reset on episode boundary.
        self._regrowth_events_this_episode: int = 0
        self._drift_step_magnitudes_this_episode: list[float] = []

        # Per-step state for diff-counting regrowth events. The grid
        # snapshot is taken before each call to ``GridWorld.step`` and
        # compared with the post-step grid to count
        # ``EMPTY → RESOURCE`` transitions. The previous-step ``p`` lets
        # the harness compute ``|Δp|`` per step for the drift aggregate.
        self._pre_step_grid: NDArray[np.uint8] | None = None
        self._pre_step_p: float | None = None

        # The current episode id, updated on episode boundary.
        self._current_episode_id: int = 0
        # The most recent env_step counter the harness has observed,
        # used as ``t_event`` for builder-perturbation events emitted
        # between steps.
        self._latest_env_step: int = 0

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> EnvStep:
        """Emit the initial ``env_reset`` and return the first ``EnvStep``.

        Constructs the underlying :class:`GridWorld`, calls
        ``GridWorld.reset()``, dispatches a single ``env_reset``
        ``WorldEvent`` for episode 0 to the handler, and returns the first
        :class:`EnvStep`. Callable exactly once per :class:`EnvServer`
        instance; subsequent calls raise.
        """
        if self._started:
            raise RuntimeError("EnvServer.start() has already been called")
        if self._closed:
            raise RuntimeError("EnvServer is closed and cannot be re-started")

        self._grid_world = GridWorld(
            self._config.grid_world_config, self._config.seed
        )
        raw_step = self._grid_world.reset()
        env_step = replace(raw_step, wallclock_ms=self._wallclock_ms())

        self._started = True
        self._current_episode_id = env_step.episode_id
        self._latest_env_step = env_step.env_step

        self._emit_env_reset(t_event=env_step.env_step)
        self._reset_aggregate_accumulators()

        return env_step

    def step(self, action: int) -> EnvStep:
        """Advance the underlying env by one step; emit boundary events.

        On a normal step, accumulates the regrowth-event count and the
        drift-step magnitude into the per-episode aggregates. On an episode
        boundary, emits an ``internal_stochasticity_aggregate`` record for
        the closing episode followed by an ``env_reset`` record for the new
        episode (in that defined order), then resets the accumulators.
        """
        grid_world = self._require_started()
        # Snapshot the pre-step state immediately before advancing the env.
        # Snapshotting here (rather than at the end of the previous call)
        # ensures any builder mutators called between calls to ``step`` are
        # rolled into the snapshot, so the regrowth-event diff does not
        # mis-attribute builder mutations as internal-stochasticity events.
        self._snapshot_pre_step()
        raw_step = grid_world.step(action)
        env_step = replace(raw_step, wallclock_ms=self._wallclock_ms())
        self._latest_env_step = env_step.env_step

        if env_step.episode_id != self._current_episode_id:
            # Boundary: the GridWorld has already reset the world internally.
            # The pre/post diff for this step covers the closing-step's
            # regrowth in a way that no longer reflects reality (the grid
            # was reset between regrowth and the post-step snapshot), so
            # the closing step's regrowth events are intentionally not
            # counted here — the aggregate covers steps 1..N-1 of the
            # episode. With p=0.01 the under-count is ~0.6 events per
            # episode of ~120, well inside the noise floor for the
            # aggregate's intended use.
            self._emit_internal_stochasticity_aggregate(
                t_event=env_step.env_step,
                episode_id=self._current_episode_id,
            )
            self._current_episode_id = env_step.episode_id
            self._emit_env_reset(t_event=env_step.env_step)
            self._reset_aggregate_accumulators()
        else:
            self._accumulate_step_stats()

        return env_step

    def close(self) -> None:
        """Mark the harness closed. Idempotent.

        Phase 3a's harness held a :class:`~kind.observer.sinks.JsonlSink`
        directly and closed it here; after the Phase 4 refactor the sink
        is owned by whoever supplied the handler (Probe 1's tests; Phase
        4's transport client; eventually the trainer process). Closing
        the harness no longer touches that sink — the caller flushes its
        own sink via whatever lifecycle it manages.
        """
        if self._closed:
            return
        self._closed = True

    # ---- mutators (the four named) ---------------------------------------

    def add_resource(
        self, cell: tuple[int, int], *, trigger: str | None = None
    ) -> None:
        """Builder mutator: set ``cell`` to ``RESOURCE``."""
        grid_world = self._require_started()
        payload = mutators.add_resource(grid_world, cell)
        self._emit_builder_perturbation(self._tag_trigger(payload, trigger))

    def remove_object(
        self,
        cell: tuple[int, int],
        object_type: CellType,
        *,
        trigger: str | None = None,
    ) -> None:
        """Builder mutator: remove the matching object at ``cell``."""
        grid_world = self._require_started()
        payload = mutators.remove_object(grid_world, cell, object_type)
        self._emit_builder_perturbation(self._tag_trigger(payload, trigger))

    def set_cell_state(
        self,
        cell: tuple[int, int],
        state: CellType,
        *,
        trigger: str | None = None,
    ) -> None:
        """Builder mutator: write ``state`` to ``cell`` unconditionally."""
        grid_world = self._require_started()
        payload = mutators.set_cell_state(grid_world, cell, state)
        self._emit_builder_perturbation(self._tag_trigger(payload, trigger))

    def move_object(
        self,
        cell_from: tuple[int, int],
        cell_to: tuple[int, int],
        *,
        trigger: str | None = None,
    ) -> None:
        """Builder mutator: move the object from ``cell_from`` to ``cell_to``."""
        grid_world = self._require_started()
        payload = mutators.move_object(grid_world, cell_from, cell_to)
        self._emit_builder_perturbation(self._tag_trigger(payload, trigger))

    @staticmethod
    def _tag_trigger(
        payload: dict[str, Any], trigger: str | None
    ) -> dict[str, Any]:
        """Merge the Probe 4 stratification tag into a mutator payload.

        ``trigger=None`` (every pre-Probe-4 caller) returns the payload
        untouched — legacy emission byte-identical. Non-None values are
        validated against the prereg §1 vocabulary so a typo cannot
        silently produce an unstratifiable event class.
        """
        if trigger is None:
            return payload
        if trigger not in _VALID_TRIGGERS:
            raise ValueError(
                f"trigger must be one of {sorted(_VALID_TRIGGERS)}, got "
                f"{trigger!r}"
            )
        return {**payload, "trigger": trigger}

    # ---- sham perturbation (Probe 2 calibration protocol) --------------

    def fire_sham_perturbation(
        self,
        mutator_label: str,
        payload: dict[str, Any],
    ) -> None:
        """Emit a flag-only ``builder_perturbation`` ``WorldEvent``.

        Probe 2 implementation plan §2.2 (carried unchanged from v1) and
        synthesis §2.4 element 3: the calibration protocol's null-event
        test fires this method to generate a builder-perturbation entry
        in the ``world_event`` stream that is *not* paired with any
        actual grid mutation. Concretely, the emitted record carries
        ``payload['is_sham'] = True`` and
        ``payload['sham_label'] = mutator_label``; the caller-supplied
        ``payload`` (which may name the intended mutator and cell so the
        sham looks structurally like a real perturbation in the JSONL)
        is merged in unchanged. Two side-effects are explicitly absent:
        no underlying-grid value is modified, and the regrowth and drift
        RNG streams are not advanced. The agent's next observation is
        therefore byte-equal to the observation that would have been
        produced if the sham had not fired.

        The mirror's reading at the sham timestamp must not flag any
        manifestation it would flag for a real perturbation at any
        reading surface; finding a flag at a sham timestamp is what the
        protocol catches as confabulation.
        """
        # Require started — same precondition as the four real mutators
        # — so the harness lifecycle is consistent across mutator-shaped
        # methods. The grid world is fetched but never consulted.
        self._require_started()
        emitted_payload: dict[str, Any] = {
            **payload,
            "is_sham": True,
            "sham_label": mutator_label,
        }
        self._emit_builder_perturbation(emitted_payload)

    # ---- runner-side wiring ---------------------------------------------

    def set_world_event_handler(self, handler: WorldEventHandler) -> None:
        """Replace the ``WorldEvent`` handler.

        Used by Phase 4's :class:`~kind.env.transport.EnvTransportServer`
        to redirect ``WorldEvent`` emissions over the wire once a client
        has connected. The replacement is safe at any point in the
        harness's lifecycle; subsequent emissions go to the new handler.
        """
        self._world_event_handler = handler

    def set_checkpoint_id(self, checkpoint_id: str | None) -> None:
        """Set the ``checkpoint_id`` envelope field on subsequent records.

        At Probe 1 the harness defaults to ``None``; the runner (Phase 5)
        will call this when a checkpoint is committed so subsequent
        ``WorldEvent`` records carry the new checkpoint identifier.
        """
        self._checkpoint_id = checkpoint_id

    @property
    def grid_world_state(self) -> GridState:
        """Read-only mirror-side view of the underlying world state.

        Convenience accessor for tests and for the eventual mirror caller
        (Phase 6) — the runner does not need this; it sees the rendered
        observation through the wrapped :class:`EnvStep`.
        """
        grid_world = self._require_started()
        return grid_world.state

    # ---- context manager -------------------------------------------------

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ---- private helpers -------------------------------------------------

    def _require_started(self) -> GridWorld:
        if self._closed:
            raise RuntimeError("EnvServer is closed")
        if not self._started:
            raise RuntimeError("EnvServer.start() must be called first")
        # The assertion narrows the optional type for mypy; it holds by
        # construction once ``_started`` is True and ``_closed`` is False.
        assert self._grid_world is not None
        return self._grid_world

    def _wallclock_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000

    # ---- WorldEvent emission --------------------------------------------

    def _emit_world_event(
        self,
        *,
        t_event: int,
        event_type: str,
        source: str,
        payload: dict[str, Any],
        schema_version: str = PROBE_1_SCHEMA_VERSION,
    ) -> None:
        record = WorldEvent(
            schema_version=schema_version,
            run_id=self._config.run_id,
            checkpoint_id=self._checkpoint_id,
            t_event=t_event,
            event_type=event_type,  # type: ignore[arg-type]
            source=source,  # type: ignore[arg-type]
            payload=payload,
            wallclock_ms=self._wallclock_ms(),
        )
        self._world_event_handler(record)

    def _emit_builder_perturbation(self, payload: dict[str, Any]) -> None:
        self._emit_world_event(
            t_event=self._latest_env_step,
            event_type=_BUILDER_PERTURBATION,
            source=_SOURCE_BUILDER,
            payload=payload,
        )

    def _emit_env_reset(self, *, t_event: int) -> None:
        grid_world = self._require_started()
        state = grid_world.state
        resource_mask = state.grid == CellType.RESOURCE.value
        resource_positions: list[list[int]] = [
            [int(r), int(c)] for r, c in zip(*np.where(resource_mask))
        ]
        # Probe 2 env revision (synthesis §2.1; implementation plan §2.2):
        # the actual cell the agent was placed at this reset is recorded
        # so random-start runs are reproducible from the regrowth seed
        # via the ``world_event`` JSONL alone. With a fixed
        # ``GridWorldConfig.start_cell`` the recorded value equals the
        # configured cell; with the Probe-2 default ``None`` it is the
        # cell sampled from the regrowth stream.
        ar, ac = state.agent_pos
        payload: dict[str, Any] = {
            "episode_id": self._current_episode_id,
            "resource_positions": resource_positions,
            "regrowth_p": float(state.regrowth_p),
            "start_cell": [int(ar), int(ac)],
        }
        self._emit_world_event(
            t_event=t_event,
            event_type=_ENV_RESET,
            source=_SOURCE_ENVIRONMENT,
            payload=payload,
        )

    def _emit_internal_stochasticity_aggregate(
        self,
        *,
        t_event: int,
        episode_id: int,
    ) -> None:
        grid_world = self._require_started()
        magnitudes = self._drift_step_magnitudes_this_episode
        mean_drift = float(np.mean(magnitudes)) if magnitudes else 0.0
        payload: dict[str, Any] = {
            "episode_id": episode_id,
            "regrowth_events": self._regrowth_events_this_episode,
            "mean_drift_step_magnitude": mean_drift,
            "final_p": float(grid_world.state.regrowth_p),
        }
        self._emit_world_event(
            t_event=t_event,
            event_type=_INTERNAL_STOCHASTICITY_AGGREGATE,
            source=_SOURCE_ENVIRONMENT,
            payload=payload,
        )

    # ---- per-step bookkeeping -------------------------------------------

    def _snapshot_pre_step(self) -> None:
        grid_world = self._require_started()
        state = grid_world.state
        # ``state.grid`` is already a copy with write=False; copy again to a
        # writable buffer is unnecessary — we only need to read from it next
        # step.
        self._pre_step_grid = state.grid
        self._pre_step_p = float(state.regrowth_p)

    def _accumulate_step_stats(self) -> None:
        """Update the per-episode aggregate accumulators from the last step.

        Called only on non-boundary steps. Counts regrowth events as cells
        where the pre-step grid was ``EMPTY`` and the post-step grid is
        ``RESOURCE``; this misses the rare case where a cell is consumed
        and then regrows in the same step (pre==RESOURCE, post==RESOURCE),
        but at Probe 1's regrowth rate that under-count is a fraction of an
        event per episode.

        Probe 4 Phase 1 (plan §S-CTRL): when
        ``emit_internal_stochasticity_events`` is on, the same diff that
        feeds the aggregate also emits one granular
        ``internal_stochasticity_event`` per regrowth cell, in row-major
        cell order (deterministic given the seed). ``t_event`` is the
        env-step whose ``EnvStep`` first reflects the regrowth — the
        regrowth happened *during* that step. (Builder events, by the
        existing convention, stamp the env-step *before* their mutation is
        applied and become visible at ``t_event + 1``; the analysis joins
        each class at its documented visibility step.) The pre-step
        snapshot discipline in :meth:`step` already excludes builder
        mutations from this diff, so the granular ENVIRONMENT class can
        never contain a misattributed BUILDER event.
        """
        grid_world = self._require_started()
        post_state = grid_world.state
        pre_grid = self._pre_step_grid
        pre_p = self._pre_step_p
        if pre_grid is not None:
            empty_pre = pre_grid == CellType.EMPTY.value
            resource_post = post_state.grid == CellType.RESOURCE.value
            regrowth_mask = empty_pre & resource_post
            regrowth_count = int(regrowth_mask.sum())
            self._regrowth_events_this_episode += regrowth_count
            if (
                self._config.emit_internal_stochasticity_events
                and regrowth_count > 0
            ):
                # np.argwhere is row-major: deterministic emission order.
                for row, col in np.argwhere(regrowth_mask):
                    self._emit_world_event(
                        t_event=self._latest_env_step,
                        event_type=_INTERNAL_STOCHASTICITY_EVENT,
                        source=_SOURCE_ENVIRONMENT,
                        payload={
                            "process": "regrowth",
                            "cell": [int(row), int(col)],
                            "pre_state": "empty",
                            "post_state": "resource",
                        },
                        schema_version=PROBE_4_WORLD_EVENT_SCHEMA_VERSION,
                    )
        if pre_grid is not None and self._config.emit_internal_stochasticity_events:
            # World v2 E1 (plan W2): trail decays are world dynamics
            # (the stamping is self-caused and visible in AgentStep; the
            # decay is the environment's process), logged granularly with
            # the same matched-control payload shape under a new process
            # tag. TRAIL→EMPTY is unambiguous in the pre/post diff: decay
            # is the only transition that produces it (consumption is
            # RESOURCE→EMPTY; a resample never leaves TRAIL behind), and
            # decay running after regrowth in ``GridWorld.step`` means a
            # decayed cell can never regrow in the same step.
            trail_pre = pre_grid == CellType.TRAIL.value
            empty_post = post_state.grid == CellType.EMPTY.value
            decay_mask = trail_pre & empty_post
            if bool(decay_mask.any()):
                for row, col in np.argwhere(decay_mask):
                    self._emit_world_event(
                        t_event=self._latest_env_step,
                        event_type=_INTERNAL_STOCHASTICITY_EVENT,
                        source=_SOURCE_ENVIRONMENT,
                        payload={
                            "process": "trail_decay",
                            "cell": [int(row), int(col)],
                            "pre_state": "trail",
                            "post_state": "empty",
                        },
                        schema_version=PROBE_4_WORLD_EVENT_SCHEMA_VERSION,
                    )
        if pre_p is not None:
            magnitude = abs(float(post_state.regrowth_p) - pre_p)
            self._drift_step_magnitudes_this_episode.append(magnitude)

    def _reset_aggregate_accumulators(self) -> None:
        self._regrowth_events_this_episode = 0
        self._drift_step_magnitudes_this_episode = []
