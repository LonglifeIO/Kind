"""Probe 4 Phase 1 — the SELF event class: self-action-effect extraction.

The three-way matched control (pre-registration §1; implementation plan
S-CTRL) needs three event classes, all resource-cell state transitions
distinguished only by cause:

- **SELF** — Io consumes a resource (RESOURCE → EMPTY at Io's cell,
  caused by Io's own movement; consumption is entry-triggered,
  ``grid_world._apply_action``). Extracted **observer-side from
  AgentStep** by this module — no new emission.
- **ENVIRONMENT** — regrowth adds a resource; logged per-event as
  ``internal_stochasticity_event`` (``env_server``).
- **BUILDER** — the four mutators; logged per-event as
  ``builder_perturbation`` (``env_server``).

The extraction contract (this is the analysis contract the plan's Phase 1
names; every consumer of the SELF class goes through it):

1. **Input**: ``AgentStep`` records from a single run, ordered by ``t``,
   with the energy telemetry enabled (``RunnerConfig.energy_telemetry=
   True`` so ``true_energy_t`` is non-None; the Probe 4 run configuration
   carries this flag — it is observer-side telemetry only and touches no
   preference or actor surface).
2. **Detection**: a consumption at record ``t`` is a rise in the
   noiseless ground truth between consecutive records,
   ``true_energy_{t+1} − true_energy_t > jump_threshold`` (default 0.03 —
   the house value from the Probe 3.5 seek-classification §1: the
   realized consumption jump is ≈ +0.068 normalized against an ordinary
   step's ≈ −0.012, so 0.03 splits the two distributions cleanly).
   ``AgentStep`` field semantics (``runner._emit_agent_step``): ``t``,
   ``h_t``, ``action_t``, and ``true_energy_t`` are all keyed to the
   *from* state — the action chosen at ``t`` produces the world state
   whose ground truth appears on record ``t+1``.
3. **Attribution**: the consumption is attributed to ``action_t`` on the
   *earlier* record, which must be a movement action (0–3): consumption
   is entry-triggered, so ``stay`` (4) cannot consume
   (``grid_world.py`` — a stay over a resource does not consume). A
   detected jump on a ``stay`` step is a data inconsistency and raises.
4. **The h-transition**: the event carries ``(h_t, h_{t+1})``. Timing
   precision (``WorldModel.step`` computes the recurrence *before* the
   posterior consumes the observation): ``h_{t+1}`` is the first state
   that carries the consuming **action** (``a_t`` enters the recurrence)
   — the action route is exactly what makes a self-effect *self* — while
   the consumption's **observation** (first visible at ``obs_{t+1}``)
   enters ``z_{t+1}`` and reaches ``h`` only at ``h_{t+2}``. The
   three-way analysis harness draws its own uniform transition windows
   from these anchors; this extractor identifies the events.
5. **Exclusions**: pairs that are not step-contiguous (``t+1`` gaps, e.g.
   shard boundaries) are skipped; pairs straddling a soft episode
   boundary are excluded because the runner zero-resets ``h`` at the
   boundary (``_init_runtime_zero_state_keep_obs``) so the h-transition
   there is not a world-model transition.
6. **Detection envelope, documented not silently handled**: energy is
   clamped at the ceiling, so a consumption with
   ``true_energy_t ≳ 1 − (replenish_norm − decay_norm)`` (≈ 0.93 at the
   settled physics) produces a truncated jump that may under-read the
   threshold and be missed. At the settled physics the run lives far
   below the ceiling (Probe 3.5 finding 1: the indifferent agent floors),
   so the envelope is wide; any analysis operating near the ceiling must
   re-derive it.

Eval-only, observer-side: reads telemetry records, computes nothing with
gradients, emits nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from kind.observer.schemas import AgentStep

__all__ = [
    "CONSUMPTION_JUMP_THRESHOLD",
    "MOVE_ACTIONS",
    "SelfActionEffect",
    "extract_self_action_effects",
]


# House value (Probe 3.5 seek-classification §1): realized consumption jump
# ≈ +0.068 normalized; ordinary step ≈ −0.012; 0.03 splits them cleanly.
CONSUMPTION_JUMP_THRESHOLD: float = 0.03

# Actions 0–3 are moves (up/down/left/right); 4 is stay. Consumption is
# entry-triggered, so only a move can consume (grid_world._apply_action).
MOVE_ACTIONS: tuple[int, ...] = (0, 1, 2, 3)


@dataclass(frozen=True)
class SelfActionEffect:
    """One SELF-class event: Io's own consumption and its h-transition.

    ``t`` / ``episode_id`` / ``action`` / ``true_energy_before`` come from
    the earlier record of the detected pair (the *from* state the action
    was taken in); ``true_energy_after`` / ``h_after`` come from the later
    record. ``h_before → h_after`` is the action-entry transition (module
    docstring point 4); the attractor-displacement analysis derives its
    own uniform per-class windows from the event anchors.
    """

    t: int
    episode_id: int
    action: int
    true_energy_before: float
    true_energy_after: float
    h_before: tuple[float, ...]
    h_after: tuple[float, ...]


def extract_self_action_effects(
    steps: Sequence[AgentStep],
    *,
    jump_threshold: float = CONSUMPTION_JUMP_THRESHOLD,
) -> list[SelfActionEffect]:
    """Extract the SELF event class from an ordered AgentStep sequence.

    See the module docstring for the full contract. Raises ``ValueError``
    if the records lack ``true_energy_t`` (run without
    ``energy_telemetry=True``), if they are not ordered by ``t``, or if a
    detected jump is attributed to a non-movement action (impossible under
    the entry-triggered consumption physics — a data inconsistency).
    """
    events: list[SelfActionEffect] = []
    for earlier, later in zip(steps, steps[1:]):
        if later.t <= earlier.t:
            raise ValueError(
                f"AgentStep records must be ordered by t; got {earlier.t} "
                f"followed by {later.t}"
            )
        if later.t != earlier.t + 1:
            # Non-contiguous pair (shard gap): no consecutive-sample jump
            # is computable here; skip per contract point 5.
            continue
        if later.episode_id != earlier.episode_id:
            # Soft episode boundary: h is zero-reset by the runner, so the
            # h-transition is not a world-model transition. Excluded.
            continue
        if earlier.true_energy_t is None or later.true_energy_t is None:
            raise ValueError(
                "AgentStep records carry true_energy_t=None — the SELF-class "
                "extraction requires a run with RunnerConfig.energy_telemetry"
                "=True (observer-side telemetry only; see module docstring)"
            )
        jump = later.true_energy_t - earlier.true_energy_t
        if jump <= jump_threshold:
            continue
        if earlier.action_t not in MOVE_ACTIONS:
            raise ValueError(
                f"true_energy jump {jump:.4f} at t={earlier.t} is attributed "
                f"to action_t={earlier.action_t}, which is not a movement "
                f"action — consumption is entry-triggered, so this is a data "
                f"inconsistency in the input records"
            )
        events.append(
            SelfActionEffect(
                t=earlier.t,
                episode_id=earlier.episode_id,
                action=earlier.action_t,
                true_energy_before=earlier.true_energy_t,
                true_energy_after=later.true_energy_t,
                h_before=tuple(earlier.h_t),
                h_after=tuple(later.h_t),
            )
        )
    return events
