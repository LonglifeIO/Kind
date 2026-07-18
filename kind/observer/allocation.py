"""Probe 4.5 S-ALLOC — the fixed-surprise allocation harness (frozen §2).

The T7 discriminator, observer-side, eval-only, no gradient: does behavior
allocate toward energy-relevant states **at matched surprise**? Authority:
the frozen pre-registration §2/§6
(``docs/decisions/probe4_5_preregistration_2026-07-13.md``); every number
here is mirrored-not-owned from that document.

The frozen signature, realized:

* **Approach step** — an action that strictly decreases BFS distance to the
  nearest resource, both distances computed **on the pre-step grid** (the
  post-action position evaluated against the pre-step world, so same-step
  regrowth/expiry cannot contaminate the readout; entering the resource
  itself is distance 0, an approach). Stay never counts. Steps with no
  reachable resource on the pre-step grid are excluded, as are episode
  boundary steps (the post-step position is a resample, not action-caused).
* **Stakes strata** — below-band (``true_energy`` < band_low) vs in-band
  (edges derived from the frozen preference constants, never re-typed).
  Above-band steps carry no stakes contrast and are excluded.
* **Surprise matching** — across-strata greedy nearest-neighbor without
  replacement on the intrinsic signal (the §2b house pattern; the smaller
  stratum anchors, anchors in trajectory order), caliper = 10% of the
  pooled intrinsic-signal std within the eval block. Matching runs within
  seed (per-seed sign stability needs per-seed pairs); the caliper is
  block-pooled. Minimum 500 matched pairs per block, else underpowered —
  recorded, not graded.
* **The lens** — ``intrinsic_signal`` is computed exactly as the runner
  computes ``intrinsic_signal_t``: ``ensemble.disagreement(h_prev, z_prev,
  a_prev)`` at each step, teacher-forced through a frozen Io-lineage world
  model. One lens for every policy the harness evaluates (the toy included),
  so the positive control validates the discriminator as it will be used.
* **The deflation reading, pre-committed** — a below/in-band contrast that
  appears unmatched but vanishes under matching is curiosity (energy-
  relevant states being more surprising), not a foreground; the unmatched
  contrast is always reported beside the matched one.

Nothing in any Io code path imports this module.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from kind.agents.actor import Actor
from kind.agents.ensemble import LatentDisagreementEnsemble
from kind.agents.preference import BAND_HALFWIDTH, SETPOINT
from kind.agents.views import PolicyView
from kind.agents.world_model import WorldModel
from kind.env.grid_world import CellType, GridState, GridWorld, GridWorldConfig

__all__ = [
    "ALLOCATION_SCHEMA_VERSION",
    "FrozenAllocationThresholds",
    "AllocationStep",
    "AllocationPolicy",
    "GreedyActorPolicy",
    "bfs_resource_distance",
    "collect_allocation_steps",
    "match_fixed_surprise",
    "AllocationBlockReport",
    "allocation_block",
    "SingleArmGate",
    "toy_gate",
    "pilot_gate",
    "ArmContrastVerdict",
    "arm_contrast",
    "block_report_to_jsonable",
]

ALLOCATION_SCHEMA_VERSION: str = "0.1.0"

_STAY_ACTION = 4
_MOVE_ACTIONS = (0, 1, 2, 3)
_ACTION_DELTAS: dict[int, tuple[int, int]] = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
    4: (0, 0),
}


@dataclass(frozen=True)
class FrozenAllocationThresholds:
    """The frozen §2/§6 numbers, mirrored-not-owned. Configurable only so
    tests can exercise both branches; runs use the defaults."""

    #: §2 pass: ΔΔ ≥ 0.10 on both verdict blocks.
    pass_delta_delta: float = 0.10
    #: §2 inert: |ΔΔ| < 0.05 on both verdict blocks.
    inert_delta_delta: float = 0.05
    #: §2: sign-consistent in ≥ 6 of the eval seeds per block.
    seed_sign_min: int = 6
    #: §2: minimum matched pairs per block, else underpowered.
    min_matched_pairs: int = 500
    #: §2: caliper = 10% of the block-pooled intrinsic-signal std.
    caliper_std_frac: float = 0.10
    #: §6: the toy must show Δ_alloc ≥ 0.20 under matching (2× the pass bar).
    toy_pass_delta: float = 0.20
    #: §6: the precision-0 pilot must read inert (|Δ_alloc| < 0.05).
    pilot_inert_delta: float = 0.05


@dataclass(frozen=True)
class AllocationStep:
    """One collected step's allocation row."""

    seed: int
    eligible: bool
    stratum: str | None  # "below_band" | "in_band" | None (excluded)
    approach: bool
    resource_adjacent: bool  # secondary readout: pre-step BFS distance <= 1
    intrinsic_signal: float


class AllocationPolicy(Protocol):
    """An action source the harness can evaluate. The greedy actor reads
    the view; the reward toy reads the mirror-side state (it is scaffolding
    outside Io's epistemic constraints — import-linted away from Io)."""

    def action(self, *, view: PolicyView, state: GridState) -> int: ...


class GreedyActorPolicy:
    """Io's greedy eval policy (the decode-honesty own-policy convention:
    zero self-prediction scalar at eval)."""

    def __init__(self, actor: Actor) -> None:
        self._actor = actor
        self._zero = torch.zeros(())

    def action(self, *, view: PolicyView, state: GridState) -> int:
        del state
        return int(self._actor.act_greedy(view).reshape(-1)[0].item())


def bfs_resource_distance(
    grid: NDArray[np.uint8], pos: tuple[int, int]
) -> int | None:
    """BFS shortest-path distance from ``pos`` to the nearest RESOURCE cell
    over non-wall cells (the oracle-forager expansion, fixed action order).
    0 when ``pos`` itself is a resource; ``None`` when no resource is
    reachable (or none exists)."""
    n_rows, n_cols = grid.shape
    if grid[pos] == CellType.RESOURCE.value:
        return 0
    visited = np.zeros((n_rows, n_cols), dtype=bool)
    visited[pos] = True
    queue: deque[tuple[tuple[int, int], int]] = deque([(pos, 0)])
    while queue:
        (r, c), dist = queue.popleft()
        for a in _MOVE_ACTIONS:
            dr, dc = _ACTION_DELTAS[a]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                continue
            if visited[nr, nc]:
                continue
            if grid[nr, nc] == CellType.WALL.value:
                continue
            if grid[nr, nc] == CellType.RESOURCE.value:
                return dist + 1
            visited[nr, nc] = True
            queue.append(((nr, nc), dist + 1))
    return None


def _stratum(true_energy: float) -> str | None:
    band_low = SETPOINT - BAND_HALFWIDTH
    band_high = SETPOINT + BAND_HALFWIDTH
    if true_energy < band_low:
        return "below_band"
    if band_low <= true_energy <= band_high:
        return "in_band"
    return None


@torch.no_grad()
def collect_allocation_steps(
    world_model: WorldModel,
    ensemble: LatentDisagreementEnsemble,
    policy: AllocationPolicy,
    grid_cfg: GridWorldConfig,
    *,
    seed: int,
    episodes: int,
) -> list[AllocationStep]:
    """Roll one seed's eval episodes; emit one allocation row per step.

    Teacher-forcing and step ordering mirror the runner exactly: the world
    model updates on the real observation; the intrinsic signal is
    ``ensemble.disagreement(h_prev, z_prev, a_prev)`` (the runner's
    ``intrinsic_signal_t``), paired with the action chosen this step.
    Deterministic given ``seed`` (env, torch RNG for the posterior sample,
    and any policy randomness are all seeded)."""
    torch.manual_seed(seed)
    device = torch.device("cpu")
    world_model.eval()
    ensemble.eval()

    n_steps = grid_cfg.episode_length * episodes
    world = GridWorld(grid_cfg, seed=seed)
    env_step = world.reset()

    h = torch.zeros(1, world_model.config.h_dim, device=device)
    z = torch.zeros(1, world_model.config.z_dim, device=device)
    a_prev = torch.zeros(1, dtype=torch.long, device=device)
    zero_scalar = torch.zeros((), device=device)

    rows: list[AllocationStep] = []
    for _ in range(n_steps):
        state = world.state
        pre_grid = state.grid
        pre_pos = state.agent_pos
        d0 = bfs_resource_distance(pre_grid, pre_pos)
        stratum = _stratum(float(state.true_energy))

        obs_np = env_step.observation.astype(np.float32) / 255.0
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).unsqueeze(0)
        sensed_t = torch.tensor([[env_step.sensed_energy]], dtype=torch.float32)
        wm_step = world_model.step(obs_t, h, z, a_prev, sensed_energy=sensed_t)
        intrinsic = float(
            ensemble.disagreement(h, z, a_prev).reshape(-1)[0].item()
        )

        view = PolicyView(
            h=wm_step.h, z=wm_step.z, self_prediction_error=zero_scalar
        )
        action = policy.action(view=view, state=state)

        env_step = world.step(action)
        boundary = env_step.step_in_episode == 0  # resample, not action-caused
        post_pos = world.state.agent_pos if not boundary else pre_pos

        eligible = d0 is not None and stratum is not None and not boundary
        approach = False
        if eligible and action != _STAY_ACTION:
            d1 = bfs_resource_distance(pre_grid, post_pos)
            approach = d1 is not None and d0 is not None and d1 < d0
        rows.append(
            AllocationStep(
                seed=seed,
                eligible=eligible,
                stratum=stratum,
                approach=approach,
                resource_adjacent=d0 is not None and d0 <= 1,
                intrinsic_signal=intrinsic,
            )
        )

        h, z = wm_step.h, wm_step.z
        a_prev = torch.tensor([action], dtype=torch.long)
    return rows


def match_fixed_surprise(
    anchor_values: NDArray[np.float64],
    candidate_values: NDArray[np.float64],
    *,
    caliper: float,
) -> list[tuple[int, int]]:
    """Greedy nearest-neighbor matching without replacement (the §2b house
    pattern): anchors in input order, each takes the nearest unused
    candidate by |Δintrinsic|; pairs beyond the caliper are dropped.
    Deterministic given input order."""
    used = np.zeros(candidate_values.shape[0], dtype=bool)
    pairs: list[tuple[int, int]] = []
    for i, value in enumerate(anchor_values):
        if used.all():
            break
        distances = np.abs(candidate_values - value)
        distances[used] = np.inf
        j = int(distances.argmin())
        if distances[j] <= caliper:
            pairs.append((i, j))
            used[j] = True
    return pairs


@dataclass(frozen=True)
class AllocationBlockReport:
    """One eval block's §2 readout for one arm/policy."""

    schema_version: str
    label: str
    n_seeds: int
    seeds: tuple[int, ...]
    caliper: float
    n_matched_pairs: int
    underpowered: bool  # n_matched_pairs < min_matched_pairs
    #: Δ_alloc = P(approach | below-band) − P(approach | in-band), matched.
    pooled_delta: float | None
    per_seed_delta: tuple[float | None, ...]
    per_seed_pairs: tuple[int, ...]
    #: The pre-committed deflation readout: the same contrast on ALL
    #: eligible rows, no matching.
    unmatched_delta: float | None
    #: Secondary §2 readout on matched pairs: resource-adjacent occupancy.
    secondary_adjacency_delta: float | None
    #: Stakes-strata sizes over eligible rows (below, in-band).
    n_below: int
    n_in_band: int


def allocation_block(
    steps: list[AllocationStep],
    *,
    label: str,
    thresholds: FrozenAllocationThresholds = FrozenAllocationThresholds(),
) -> AllocationBlockReport:
    """Compute the frozen §2 statistics for one block of collected steps."""
    eligible = [s for s in steps if s.eligible]
    seeds = tuple(sorted({s.seed for s in steps}))
    below_all = [s for s in eligible if s.stratum == "below_band"]
    in_all = [s for s in eligible if s.stratum == "in_band"]

    pooled_std = (
        float(np.std([s.intrinsic_signal for s in eligible]))
        if eligible
        else 0.0
    )
    caliper = thresholds.caliper_std_frac * pooled_std

    def rate(rows: list[AllocationStep]) -> float | None:
        return float(np.mean([s.approach for s in rows])) if rows else None

    unmatched_delta: float | None = None
    below_rate, in_rate = rate(below_all), rate(in_all)
    if below_rate is not None and in_rate is not None:
        unmatched_delta = below_rate - in_rate

    per_seed_delta: list[float | None] = []
    per_seed_pairs: list[int] = []
    matched_below: list[AllocationStep] = []
    matched_in: list[AllocationStep] = []
    for seed in seeds:
        below = [s for s in below_all if s.seed == seed]
        in_band = [s for s in in_all if s.seed == seed]
        # Smaller stratum anchors (the house pattern).
        anchors, candidates = (
            (below, in_band) if len(below) <= len(in_band) else (in_band, below)
        )
        pairs = match_fixed_surprise(
            np.asarray([s.intrinsic_signal for s in anchors]),
            np.asarray([s.intrinsic_signal for s in candidates]),
            caliper=caliper,
        )
        seed_below: list[AllocationStep] = []
        seed_in: list[AllocationStep] = []
        for i, j in pairs:
            a, c = anchors[i], candidates[j]
            if a.stratum == "below_band":
                seed_below.append(a)
                seed_in.append(c)
            else:
                seed_below.append(c)
                seed_in.append(a)
        matched_below.extend(seed_below)
        matched_in.extend(seed_in)
        per_seed_pairs.append(len(pairs))
        sb, si = rate(seed_below), rate(seed_in)
        per_seed_delta.append(
            None if sb is None or si is None else sb - si
        )

    n_pairs = len(matched_below)
    pooled_delta: float | None = None
    secondary: float | None = None
    if n_pairs > 0:
        mb, mi = rate(matched_below), rate(matched_in)
        assert mb is not None and mi is not None
        pooled_delta = mb - mi
        secondary = float(
            np.mean([s.resource_adjacent for s in matched_below])
        ) - float(np.mean([s.resource_adjacent for s in matched_in]))

    return AllocationBlockReport(
        schema_version=ALLOCATION_SCHEMA_VERSION,
        label=label,
        n_seeds=len(seeds),
        seeds=seeds,
        caliper=caliper,
        n_matched_pairs=n_pairs,
        underpowered=n_pairs < thresholds.min_matched_pairs,
        pooled_delta=pooled_delta,
        per_seed_delta=tuple(per_seed_delta),
        per_seed_pairs=tuple(per_seed_pairs),
        unmatched_delta=unmatched_delta,
        secondary_adjacency_delta=secondary,
        n_below=len(below_all),
        n_in_band=len(in_all),
    )


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SingleArmGate:
    """A §6 control gate on one block (toy positive / pilot negative)."""

    gate: str
    delta: float | None
    threshold: float
    n_matched_pairs: int
    underpowered: bool
    passed: bool


def toy_gate(
    report: AllocationBlockReport,
    thresholds: FrozenAllocationThresholds = FrozenAllocationThresholds(),
) -> SingleArmGate:
    """§6 positive control: the reward toy must show Δ_alloc ≥ 0.20 under
    matching. Underpowered blocks fail — an unvalidatable instrument is a
    failed validation, not a free pass."""
    passed = (
        not report.underpowered
        and report.pooled_delta is not None
        and report.pooled_delta >= thresholds.toy_pass_delta
    )
    return SingleArmGate(
        gate="toy_positive_control",
        delta=report.pooled_delta,
        threshold=thresholds.toy_pass_delta,
        n_matched_pairs=report.n_matched_pairs,
        underpowered=report.underpowered,
        passed=passed,
    )


def pilot_gate(
    report: AllocationBlockReport,
    thresholds: FrozenAllocationThresholds = FrozenAllocationThresholds(),
) -> SingleArmGate:
    """§6 negative control: the precision-0 pilot must read inert
    (|Δ_alloc| < 0.05 under matching)."""
    passed = (
        not report.underpowered
        and report.pooled_delta is not None
        and abs(report.pooled_delta) < thresholds.pilot_inert_delta
    )
    return SingleArmGate(
        gate="pilot_negative_control",
        delta=report.pooled_delta,
        threshold=thresholds.pilot_inert_delta,
        n_matched_pairs=report.n_matched_pairs,
        underpowered=report.underpowered,
        passed=passed,
    )


@dataclass(frozen=True)
class ArmContrastVerdict:
    """The §2 two-arm contrast for one eval block (Phase 4 machinery,
    built and pinned at Phase 3 so nothing is authored mid-run)."""

    delta_delta: float | None
    per_seed_delta_delta: tuple[float | None, ...]
    sign_consistent_seeds: int
    seeds_defined: int
    #: "pass" | "inert" | "residual" | "underpowered" — one block's reading;
    #: the §2 verdict needs BOTH verdict blocks to agree (rendered by the
    #: Phase 4 verdict script, never here).
    classification: str


def arm_contrast(
    preference: AllocationBlockReport,
    control: AllocationBlockReport,
    thresholds: FrozenAllocationThresholds = FrozenAllocationThresholds(),
) -> ArmContrastVerdict:
    if preference.seeds != control.seeds:
        raise ValueError(
            f"arms must share eval seeds; got {preference.seeds} vs "
            f"{control.seeds} (the arms are twins — §5)"
        )
    if (
        preference.underpowered
        or control.underpowered
        or preference.pooled_delta is None
        or control.pooled_delta is None
    ):
        return ArmContrastVerdict(
            delta_delta=None,
            per_seed_delta_delta=tuple(
                None for _ in range(preference.n_seeds)
            ),
            sign_consistent_seeds=0,
            seeds_defined=0,
            classification="underpowered",
        )
    dd = preference.pooled_delta - control.pooled_delta
    per_seed: list[float | None] = []
    for p, c in zip(preference.per_seed_delta, control.per_seed_delta):
        per_seed.append(None if p is None or c is None else p - c)
    defined = [d for d in per_seed if d is not None]
    consistent = sum(1 for d in defined if np.sign(d) == np.sign(dd))
    if dd >= thresholds.pass_delta_delta and consistent >= thresholds.seed_sign_min:
        classification = "pass"
    elif abs(dd) < thresholds.inert_delta_delta:
        classification = "inert"
    else:
        classification = "residual"
    return ArmContrastVerdict(
        delta_delta=dd,
        per_seed_delta_delta=tuple(per_seed),
        sign_consistent_seeds=consistent,
        seeds_defined=len(defined),
        classification=classification,
    )


def block_report_to_jsonable(report: AllocationBlockReport) -> dict[str, object]:
    return asdict(report)
