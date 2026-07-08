"""Probe 4 Phase 1 — memory-horizon eval harness (observer-side, eval-only).

Measures the **functional forward horizon** of a frozen world model's
deterministic recurrent state ``h``: inject a distinct one-step event into
the observation stream, teacher-force the perturbed and an un-perturbed
counterfactual stream through the same frozen model, and track the
per-step divergence of the two ``h``-trajectories until it decays into
noise (synthesis T4 [S2]; implementation plan Phase 1). The horizon this
reports is the number that sets the Probe 4 **event rate** (DP3) — it is a
design knob, not a success threshold (pre-registration §8).

Design (one measurement pair per injection context):

1. Two ``GridWorld`` instances from the same seed, stepped through the
   same scripted action sequence, produce identical observation streams —
   except that in the perturbed world one cell is set to ``RESOURCE``
   immediately before the injection step and reverted to ``EMPTY``
   immediately after it. Direct grid writes advance no RNG stream, so with
   a **quiet** world config (regrowth and drift disabled — the default
   here) the two streams are pixel-identical everywhere except the single
   injection step. The harness asserts exactly this before trusting a
   measurement: a pulse that is not visible (injected cell out of view) or
   a stream that diverges elsewhere (stochastic world config) raises
   rather than silently producing a wrong horizon.
2. Both streams are teacher-forced through the frozen model with the
   runner's zero-state init (``h₀ = 0``, ``z₀ = 0``, ``a₀ = 0``;
   ``runner._init_runtime_zero_state``) and the runner's observation
   scaling (uint8 → float32 / 255). ``z`` is taken as the **posterior
   mean** rather than an ``rsample`` so the comparison is deterministic —
   the divergence measured is the input pulse's, not sampling noise's
   (the B′ pattern from the Probe 3.5 energy eval).
3. Per step, two divergence measures between the trajectories:

   - ``kl_divergence_nats`` — KL(prior(h_perturbed) ‖ prior(h_control)),
     summed over the ``z`` dims. The prior head is a function of ``h``
     only, so this is the model's own belief-space reading of how
     different the two ``h`` states are — "the h-trajectory KL" of the
     plan.
   - ``h_l2_distance`` — ‖h_perturbed − h_control‖₂, the raw state-space
     reading. Secondary/corroborating.

4. The horizon is the first post-injection step at which the divergence
   falls below ``decay_fraction`` × its peak **and stays below it** for
   the rest of the window (GRU transients can be non-monotonic; a dip is
   not a decay). If the divergence never sustains below the threshold the
   horizon is censored (``None``) and the window length is the lower
   bound.

The report also carries the two **credit-assignment ceilings** the
synthesis names (T4): the world-model truncated-BPTT window
(``replay_sequence_length``, runner default 32 — the window within which
a category must *form*) and the imagination horizon (runner default 15).
Both are reported numbers, not measured here — they mirror
``RunnerConfig`` and are runner-authoritative; callers measuring against
a differently-configured runner pass the actual values.

Eval-only discipline: everything runs under ``torch.no_grad()`` on a
model switched to ``eval()``; nothing here trains, mutates, or emits
telemetry. The harness is instance-conditional — the horizon is a
property of the *weights it is handed*, and re-measuring on the actual
Phase 4 instance is cheap and expected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from kind.agents.world_model import WorldModel, WorldModelConfig
from kind.env.grid_world import CellType, GridWorld, GridWorldConfig

__all__ = [
    "MemoryHorizonConfig",
    "InjectionContext",
    "HorizonMeasurement",
    "MemoryHorizonReport",
    "load_world_model_from_checkpoint",
    "measure_memory_horizon",
]


# Runner-authoritative ceilings, mirrored from ``RunnerConfig`` defaults
# (kind/training/runner.py). Reported alongside the measured horizon so the
# journal entry carries both the passive-retention number and the binding
# credit-assignment windows (synthesis T4: the BPTT window is the binding
# limit — the category must *form* within it, not merely be *held* by h).
_DEFAULT_REPLAY_SEQUENCE_LENGTH: int = 32
_DEFAULT_IMAGINATION_HORIZON: int = 15


def _quiet_grid_config() -> GridWorldConfig:
    """A stochasticity-free world so the injected pulse is the only
    difference between the paired streams.

    Regrowth and drift are disabled; the start cell is pinned so the
    injected cell can be chosen deterministically in view. Resources are
    static (n_initial_resources default) — a static world is exactly what
    makes the one-step pulse a clean memory probe rather than a
    world-divergence probe.
    """
    return GridWorldConfig(
        initial_regrowth_p=0.0,
        drift_p_min=0.0,
        drift_p_max=0.0,
        drift_magnitude_per_step=0.0,
        start_cell=(3, 3),
    )


@dataclass(frozen=True)
class MemoryHorizonConfig:
    """Static configuration for one harness run.

    ``injection_steps`` are the env steps at which the one-step pulse is
    injected — one measurement pair per entry (each shares the control
    trajectory). ``window_steps`` is how far past the *last* injection the
    trajectories are rolled; the horizon is censored at the window. The
    per-context measurement window must fit inside one episode (the soft
    episode boundary resets the agent and resamples resources, which would
    contaminate the pulse with a world change).
    """

    seed: int = 0
    injection_steps: tuple[int, ...] = (20,)
    window_steps: int = 120
    decay_fraction: float = 0.01
    actions: tuple[int, ...] | None = None
    injection_cell: tuple[int, int] | None = None
    grid_world_config: GridWorldConfig = field(default_factory=_quiet_grid_config)
    replay_sequence_length: int = _DEFAULT_REPLAY_SEQUENCE_LENGTH
    imagination_horizon: int = _DEFAULT_IMAGINATION_HORIZON


@dataclass(frozen=True)
class InjectionContext:
    """Where and when one pulse was injected."""

    injection_step: int
    injection_cell: tuple[int, int]


@dataclass(frozen=True)
class HorizonMeasurement:
    """One injection context's divergence curves and horizons.

    ``kl_curve`` / ``l2_curve`` are indexed from the **first ``h`` that can
    carry the pulse**: the recurrence computes ``h_t`` *before* the
    posterior consumes ``obs_t``, so the perturbed observation at the
    injection step first divergences ``z`` there and first reaches ``h``
    one recurrence later — curve index 0 is that step. ``horizon_*`` is
    therefore directly "the number of steps the pulse persists in ``h``".
    ``horizon_*`` is the first index at which the curve falls below
    ``decay_fraction`` × its peak and stays there through the end of the
    window; ``None`` means censored (never sustained below threshold
    within the window).
    """

    context: InjectionContext
    kl_curve: tuple[float, ...]
    l2_curve: tuple[float, ...]
    kl_peak: float
    l2_peak: float
    horizon_kl_steps: int | None
    horizon_l2_steps: int | None


@dataclass(frozen=True)
class MemoryHorizonReport:
    """The harness's full output for one (model, config) pair."""

    measurements: tuple[HorizonMeasurement, ...]
    decay_fraction: float
    window_steps: int
    replay_sequence_length: int
    imagination_horizon: int


def load_world_model_from_checkpoint(
    checkpoint_path: Path,
    world_model_config: WorldModelConfig | None = None,
) -> WorldModel:
    """Load a frozen world model from a runner checkpoint's
    ``"world_model"`` state dict (the seek-classifier loading pattern).

    The returned model is in ``eval()`` mode with gradients untouched —
    the harness never backpropagates through it.
    """
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    world_model = WorldModel(world_model_config or WorldModelConfig())
    world_model.load_state_dict(checkpoint["world_model"])
    world_model.eval()
    return world_model


def _obs_to_tensor(obs: NDArray[np.uint8]) -> Tensor:
    """uint8 (H, W) → float32 (1, 1, H, W) in [0, 1] — the runner's
    observation scaling (``runner._obs_to_cpu_tensor``) plus batch dim."""
    arr = np.array(obs, dtype=np.uint8, copy=True)
    return (torch.from_numpy(arr).float() / 255.0).unsqueeze(0).unsqueeze(0)


def _pick_injection_cell(
    grid_world: GridWorld, config: MemoryHorizonConfig
) -> tuple[int, int]:
    """Choose the cell the pulse is written to.

    Honors ``config.injection_cell`` when given; otherwise scans row-major
    for the first EMPTY cell inside the agent's ego-centric view that is
    not the agent's own cell (deterministic, and guarantees the pulse is
    visible in the rendered observation).
    """
    state = grid_world.state
    if config.injection_cell is not None:
        return config.injection_cell
    view_radius = grid_world.config.view_size // 2
    ar, ac = state.agent_pos
    grid_size = grid_world.config.grid_size
    for r in range(max(0, ar - view_radius), min(grid_size, ar + view_radius + 1)):
        for c in range(
            max(0, ac - view_radius), min(grid_size, ac + view_radius + 1)
        ):
            if (r, c) == (ar, ac):
                continue
            if state.grid[r, c] == CellType.EMPTY.value:
                return (r, c)
    raise ValueError(
        "no EMPTY in-view cell available for injection; pass "
        "MemoryHorizonConfig.injection_cell explicitly"
    )


def _collect_streams(
    config: MemoryHorizonConfig, injection_step: int
) -> tuple[
    list[NDArray[np.uint8]],
    list[NDArray[np.uint8]],
    list[float],
    list[int],
    tuple[int, int],
]:
    """Roll the control and perturbed worlds; return the paired streams.

    Returns ``(control_obs, perturbed_obs, sensed_energy, actions, cell)``
    where the observation lists have ``total_steps + 1`` entries (index 0
    is the reset observation) and are asserted pixel-identical everywhere
    except index ``injection_step``. ``sensed_energy`` is identical across
    the pair by construction (same sensing RNG, same actions) so one list
    serves both.
    """
    total_steps = injection_step + config.window_steps
    if total_steps + 1 >= config.grid_world_config.episode_length:
        raise ValueError(
            f"injection_step + window_steps ({total_steps}) must fit inside "
            f"one episode (episode_length="
            f"{config.grid_world_config.episode_length}); the soft episode "
            f"boundary resamples the world and would contaminate the pulse"
        )
    if config.actions is not None and len(config.actions) < total_steps:
        raise ValueError(
            f"actions has {len(config.actions)} entries; needs >= {total_steps}"
        )

    control = GridWorld(config.grid_world_config, config.seed)
    perturbed = GridWorld(config.grid_world_config, config.seed)
    control_first = control.reset()
    perturbed_first = perturbed.reset()

    injection_cell = _pick_injection_cell(perturbed, config)

    control_obs: list[NDArray[np.uint8]] = [control_first.observation]
    perturbed_obs: list[NDArray[np.uint8]] = [perturbed_first.observation]
    sensed: list[float] = [control_first.sensed_energy]
    actions: list[int] = []

    for step_index in range(1, total_steps + 1):
        action = (
            config.actions[step_index - 1] if config.actions is not None else 4
        )
        actions.append(action)
        if step_index == injection_step:
            # The pulse: one cell flips EMPTY → RESOURCE for exactly one
            # rendered observation. Direct grid writes advance no RNG
            # stream (the same property the gate test's parallel-env
            # design relies on), so the worlds re-converge exactly after
            # the revert below.
            pre_value = int(perturbed._grid[injection_cell])
            if pre_value != CellType.EMPTY.value:
                raise ValueError(
                    f"injection cell {injection_cell} is not EMPTY at the "
                    f"injection step (holds {pre_value}); choose another cell"
                )
            perturbed._grid[injection_cell] = CellType.RESOURCE.value
        control_step = control.step(action)
        perturbed_step = perturbed.step(action)
        if step_index == injection_step:
            perturbed._grid[injection_cell] = CellType.EMPTY.value
        control_obs.append(control_step.observation)
        perturbed_obs.append(perturbed_step.observation)
        sensed.append(control_step.sensed_energy)
        if control_step.sensed_energy != perturbed_step.sensed_energy:
            raise ValueError(
                "sensed_energy diverged between the paired worlds — the "
                "scripted actions interacted with the injected cell; use "
                "actions that avoid it"
            )

    # Self-check: the pulse is visible at exactly the injection step and
    # nowhere else. Anything else means the measurement would not be a
    # one-step-pulse memory probe.
    for index, (obs_a, obs_b) in enumerate(zip(control_obs, perturbed_obs)):
        equal = bool(np.array_equal(obs_a, obs_b))
        if index == injection_step and equal:
            raise ValueError(
                f"injected pulse at step {injection_step} (cell "
                f"{injection_cell}) is not visible in the observation — "
                f"choose an in-view cell"
            )
        if index != injection_step and not equal:
            raise ValueError(
                f"observation streams diverge at step {index} (expected "
                f"divergence only at {injection_step}) — the world config "
                f"must be stochasticity-free for this measurement"
            )
    return control_obs, perturbed_obs, sensed, actions, injection_cell


def _diagonal_gaussian_kl(
    mu_a: Tensor, log_sigma_a: Tensor, mu_b: Tensor, log_sigma_b: Tensor
) -> float:
    """KL(N(mu_a, sigma_a) ‖ N(mu_b, sigma_b)) for diagonal Gaussians,
    summed over dims. Closed form; inputs are ``(1, z_dim)``."""
    var_a = torch.exp(2.0 * log_sigma_a)
    var_b = torch.exp(2.0 * log_sigma_b)
    kl_per_dim = (
        log_sigma_b
        - log_sigma_a
        + (var_a + (mu_a - mu_b) ** 2) / (2.0 * var_b)
        - 0.5
    )
    return float(kl_per_dim.sum().item())


def _teacher_force(
    world_model: WorldModel,
    observations: list[NDArray[np.uint8]],
    sensed_energy: list[float],
    actions: list[int],
) -> list[Tensor]:
    """Teacher-force one stream; return the per-step ``h`` trajectory.

    Deterministic: ``z`` is the posterior mean (no ``rsample``), init is
    the runner's zero state. Returns ``h`` after consuming each
    observation from index 1 onward (index 0 seeds nothing — the runner's
    first forward consumes the reset observation with zero priors, so we
    do the same and include it).
    """
    h = torch.zeros(1, world_model.config.h_dim)
    z = torch.zeros(1, world_model.config.z_dim)
    a = torch.zeros(1, dtype=torch.long)
    trajectory: list[Tensor] = []
    with torch.no_grad():
        for index, obs in enumerate(observations):
            obs_tensor = _obs_to_tensor(obs)
            sensed_tensor = torch.tensor(
                [[sensed_energy[index]]], dtype=torch.float32
            )
            h = world_model.recurrence(h, z, a)
            embed = world_model.encode(obs_tensor)
            energy_embed = world_model.encode_energy(sensed_tensor)
            q_mu, _q_log_sigma = world_model.posterior(h, embed, energy_embed)
            z = q_mu  # posterior mean: deterministic teacher forcing
            trajectory.append(h.clone())
            if index < len(actions):
                a = torch.tensor([actions[index]], dtype=torch.long)
    return trajectory


def _sustained_decay_index(
    curve: tuple[float, ...], peak: float, decay_fraction: float
) -> int | None:
    """First index at which the curve is below ``decay_fraction * peak``
    and stays below it through the end of the window; None if never."""
    if peak <= 0.0:
        return None
    threshold = decay_fraction * peak
    horizon: int | None = None
    for index, value in enumerate(curve):
        if value < threshold:
            if horizon is None:
                horizon = index
        else:
            horizon = None
    return horizon


def measure_memory_horizon(
    world_model: WorldModel, config: MemoryHorizonConfig
) -> MemoryHorizonReport:
    """Run the full harness: one measurement pair per injection context."""
    world_model.eval()
    measurements: list[HorizonMeasurement] = []
    for injection_step in config.injection_steps:
        control_obs, perturbed_obs, sensed, actions, cell = _collect_streams(
            config, injection_step
        )
        h_control = _teacher_force(world_model, control_obs, sensed, actions)
        h_perturbed = _teacher_force(
            world_model, perturbed_obs, sensed, actions
        )

        kl_values: list[float] = []
        l2_values: list[float] = []
        # h_t is computed before obs_t is consumed, so the pulse first
        # reaches h at injection_step + 1 (see HorizonMeasurement).
        with torch.no_grad():
            for index in range(injection_step + 1, len(h_control)):
                h_a = h_control[index]
                h_b = h_perturbed[index]
                p_mu_a, p_log_sigma_a = world_model.prior(h_a)
                p_mu_b, p_log_sigma_b = world_model.prior(h_b)
                kl_values.append(
                    _diagonal_gaussian_kl(
                        p_mu_b, p_log_sigma_b, p_mu_a, p_log_sigma_a
                    )
                )
                l2_values.append(float(torch.linalg.norm(h_b - h_a).item()))

        kl_curve = tuple(kl_values)
        l2_curve = tuple(l2_values)
        kl_peak = max(kl_curve)
        l2_peak = max(l2_curve)
        measurements.append(
            HorizonMeasurement(
                context=InjectionContext(
                    injection_step=injection_step, injection_cell=cell
                ),
                kl_curve=kl_curve,
                l2_curve=l2_curve,
                kl_peak=kl_peak,
                l2_peak=l2_peak,
                horizon_kl_steps=_sustained_decay_index(
                    kl_curve, kl_peak, config.decay_fraction
                ),
                horizon_l2_steps=_sustained_decay_index(
                    l2_curve, l2_peak, config.decay_fraction
                ),
            )
        )
    return MemoryHorizonReport(
        measurements=tuple(measurements),
        decay_fraction=config.decay_fraction,
        window_steps=config.window_steps,
        replay_sequence_length=config.replay_sequence_length,
        imagination_horizon=config.imagination_horizon,
    )
