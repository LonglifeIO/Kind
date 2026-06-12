"""Decode-honesty table — standing observer-side calibration instrument.

Promotes the one-off D1 diagnostic of the seek-mechanism classification
(`docs/decisions/probe3_5_seek_classification_2026-06-12.md` §1/§2) into a
typed, deterministic, checkpoint-agnostic evaluator: given any world-model +
actor checkpoint and an env config, generate teacher-forced evaluation
trajectories from three policy sources — the instance's **own greedy policy**
(on-distribution), the **oracle forager** (in-band coverage by construction),
and **uniform-random** actions (mid coverage) — and produce a per-region
honesty table per source and pooled.

Every future reader of ``decode_energy`` reads it through this table: the
table is the decoder's **calibration envelope**. In particular it is the
calibration reference for §7 dream passive-decode monitor readings (dream
states wander off-distribution by construction; the verdict doc §5 bin-1
contingency) and is expected to become pre-registered Probe 4
instrumentation.

Standing readouts, not one-time findings:

* per-region **bias** (decode − true) and **abs error** over the five named
  energy regions — floor-adjacent / below-band / in-band / above-band /
  ceiling-adjacent — with edges derived from config constants (band =
  setpoint ± halfwidth, the frozen B0b / amended B0a′; rail margin = one
  sensing-noise σ, ``GridWorldConfig.energy_obs_noise_sigma``);
* the pooled **decode~true regression slope** (D1's
  regression-toward-the-rail / anti-correlation detector);
* the **three-way error comparison** — decode-vs-true, decode-vs-sensed,
  sensed-vs-true — so "the decoder ignores the honest sensor"
  (classification §2) is a standing readout: when decode-vs-sensed ≈
  decode-vs-true ≫ sensed-vs-true, the honest sense organ is being
  discarded;
* the **out-of-range mass** (decode outside the physical [0, 1]) — the F2
  gate input (classification §9.2).

Eval-only throughout: ``true_energy`` is read from ``GridState`` as an eval
target (the S-ENV rule binds — it enters no training loss anywhere; this
module trains nothing). The instrument is observer-side: it adds no
actor-readable interface; nothing in any Io code path imports it.

Determinism: the collector seeds the global torch RNG from the per-trajectory
seed before stepping (the posterior ``rsample`` draws from the global RNG),
and uniform-random actions come from a dedicated seeded numpy generator —
fixed seeds give bit-identical reports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray

from kind.agents.actor import Actor
from kind.agents.preference import BAND_HALFWIDTH, SETPOINT
from kind.agents.views import PolicyView
from kind.agents.world_model import WorldModel
from kind.env.grid_world import GridWorld, GridWorldConfig
from kind.observer.oracle_forager import oracle_action

__all__ = [
    "DECODE_HONESTY_SCHEMA_VERSION",
    "POLICY_SOURCES",
    "PolicySource",
    "RegionEdges",
    "RegionRow",
    "HonestyTable",
    "TeacherForcedTrajectory",
    "DecodeHonestyReport",
    "region_edges_from_config",
    "region_names",
    "collect_teacher_forced_trajectory",
    "honesty_table",
    "run_decode_honesty",
    "report_to_jsonable",
]

#: Versioned output schema (house pattern: a literal stamped on every record;
#: readers branch on it, writers never emit anything else).
DECODE_HONESTY_SCHEMA_VERSION: str = "0.1.0"

PolicySource = Literal["own_policy", "oracle", "uniform_random"]

POLICY_SOURCES: tuple[PolicySource, ...] = (
    "own_policy",
    "oracle",
    "uniform_random",
)

#: Default evaluation seed bases per source. Chosen disjoint from the seek-
#: classification's collection seeds (9000/9100/9200 series) and from the F1
#: recalibration training mixture, so the standing table is out-of-sample
#: with respect to both — analysis-instrument choices, not frozen-protocol
#: measurements.
DEFAULT_SEED_BASES: tuple[tuple[PolicySource, int], ...] = (
    ("own_policy", 9700),
    ("oracle", 9800),
    ("uniform_random", 9900),
)

_STAY_ACTION = 4
_NUM_ACTIONS = 5


# ---------------------------------------------------------------------------
# Energy regions — edges from config constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionEdges:
    """The three numbers that fix the five-region partition of [0, 1].

    ``band_low`` / ``band_high`` come from the frozen preference constants
    (setpoint ± halfwidth); ``rail_margin`` is one sensing-noise σ
    (``GridWorldConfig.energy_obs_noise_sigma``) — "rail-adjacent" means
    within one σ of the physical floor or ceiling.
    """

    rail_margin: float
    band_low: float
    band_high: float


def region_edges_from_config(
    grid_cfg: GridWorldConfig,
    *,
    setpoint: float = SETPOINT,
    band_halfwidth: float = BAND_HALFWIDTH,
) -> RegionEdges:
    """Derive the region edges from config constants (no data-dependent edges)."""
    return RegionEdges(
        rail_margin=grid_cfg.energy_obs_noise_sigma,
        band_low=setpoint - band_halfwidth,
        band_high=setpoint + band_halfwidth,
    )


def region_names() -> tuple[str, str, str, str, str]:
    """The five named regions, in increasing-energy order."""
    return (
        "floor_adjacent",
        "below_band",
        "in_band",
        "above_band",
        "ceiling_adjacent",
    )


def _region_masks(
    true_energy: NDArray[np.float64], edges: RegionEdges
) -> tuple[tuple[str, NDArray[np.bool_]], ...]:
    """Partition by true energy: every value falls in exactly one region."""
    e = true_energy
    floor = e < edges.rail_margin
    below = (e >= edges.rail_margin) & (e < edges.band_low)
    in_band = (e >= edges.band_low) & (e <= edges.band_high)
    above = (e > edges.band_high) & (e < 1.0 - edges.rail_margin)
    ceiling = e >= 1.0 - edges.rail_margin
    names = region_names()
    return (
        (names[0], floor),
        (names[1], below),
        (names[2], in_band),
        (names[3], above),
        (names[4], ceiling),
    )


# ---------------------------------------------------------------------------
# Collection — teacher-forced trajectory per policy source
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeacherForcedTrajectory:
    """Per-step records of one seed's env rollout, teacher-forced through the
    world model (the posterior sees the real observation and the real sensed
    energy at every step; the recurrence advances on the real action).

    ``decode_energy`` is the ``energy_pred`` emission at each step. ``h`` /
    ``z`` are the post-update latents — carried so downstream consumers (e.g.
    the F1 head recalibration script) can reuse this collector for
    ``(h, z) → sensed`` pairs without re-deriving the teacher-forcing loop;
    the honesty table itself reads only the three scalar series.
    """

    true_energy: NDArray[np.float64]
    sensed_energy: NDArray[np.float64]
    decode_energy: NDArray[np.float64]
    h: NDArray[np.float32]
    z: NDArray[np.float32]


@torch.no_grad()
def collect_teacher_forced_trajectory(
    world_model: WorldModel,
    actor: Actor,
    grid_cfg: GridWorldConfig,
    *,
    policy_source: PolicySource,
    seed: int,
    episodes: int,
) -> TeacherForcedTrajectory:
    """Roll the env under one policy source, teacher-forcing the world model.

    Deterministic given ``seed``: the env is seeded, the global torch RNG is
    seeded (the posterior ``rsample``), and uniform-random actions draw from
    a numpy generator seeded the same way. The actor's greedy action is
    deterministic given fixed weights.
    """
    torch.manual_seed(seed)
    action_rng = np.random.default_rng(seed)
    device = torch.device("cpu")
    world_model.eval()
    actor.eval()

    n_steps = grid_cfg.episode_length * episodes
    world = GridWorld(grid_cfg, seed=seed)
    env_step = world.reset()

    h = torch.zeros(1, world_model.config.h_dim, device=device)
    z = torch.zeros(1, world_model.config.z_dim, device=device)
    a_prev = torch.zeros(1, dtype=torch.long, device=device)
    zero_scalar = torch.zeros((), device=device)

    true_l: list[float] = []
    sensed_l: list[float] = []
    decode_l: list[float] = []
    h_l: list[NDArray[np.float32]] = []
    z_l: list[NDArray[np.float32]] = []

    for _ in range(n_steps):
        obs_np = env_step.observation.astype(np.float32) / 255.0
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).unsqueeze(0)
        sensed_t = torch.tensor([[env_step.sensed_energy]], dtype=torch.float32)
        wm_step = world_model.step(obs_t, h, z, a_prev, sensed_energy=sensed_t)

        state = world.state
        if policy_source == "oracle":
            action = oracle_action(state.grid, state.agent_pos, state.true_energy)
        elif policy_source == "uniform_random":
            action = int(action_rng.integers(0, _NUM_ACTIONS))
        else:
            view = PolicyView(
                h=wm_step.h, z=wm_step.z, self_prediction_error=zero_scalar
            )
            action = int(actor.act_greedy(view).reshape(-1)[0].item())

        true_l.append(float(state.true_energy))
        sensed_l.append(float(env_step.sensed_energy))
        decode_l.append(float(wm_step.energy_pred.reshape(-1)[0].item()))
        h_l.append(wm_step.h.squeeze(0).numpy().astype(np.float32))
        z_l.append(wm_step.z.squeeze(0).numpy().astype(np.float32))

        env_step = world.step(action)
        h, z = wm_step.h, wm_step.z
        a_prev = torch.tensor([action], dtype=torch.long)

    return TeacherForcedTrajectory(
        true_energy=np.asarray(true_l),
        sensed_energy=np.asarray(sensed_l),
        decode_energy=np.asarray(decode_l),
        h=np.stack(h_l),
        z=np.stack(z_l),
    )


# ---------------------------------------------------------------------------
# The honesty table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionRow:
    """One region's honesty readout. ``None`` statistics mean the source never
    visited the region (``n == 0``) — coverage gaps are reported, not hidden."""

    region: str
    n: int
    decode_mean: float | None
    decode_std: float | None
    true_mean: float | None
    bias: float | None
    abs_error_mean: float | None


@dataclass(frozen=True)
class HonestyTable:
    """One source's (or the pooled) honesty table.

    ``slope_decode_vs_true`` is the OLS slope of decode on true over the
    occupied range (``None`` when the true series is degenerate). The
    three-way error comparison makes sensor-discarding a standing readout;
    ``out_of_range_mass`` is the F2 gate input.
    """

    source: str
    n_steps: int
    rows: tuple[RegionRow, ...]
    slope_decode_vs_true: float | None
    decode_vs_true_abs_error_mean: float
    decode_vs_sensed_abs_error_mean: float
    sensed_vs_true_abs_error_mean: float
    out_of_range_mass: float


def honesty_table(
    source: str,
    trajectories: list[TeacherForcedTrajectory],
    edges: RegionEdges,
) -> HonestyTable:
    """Compute the per-region honesty table from collected trajectories."""
    true = np.concatenate([t.true_energy for t in trajectories])
    sensed = np.concatenate([t.sensed_energy for t in trajectories])
    decode = np.concatenate([t.decode_energy for t in trajectories])

    rows: list[RegionRow] = []
    for name, mask in _region_masks(true, edges):
        n = int(mask.sum())
        if n == 0:
            rows.append(
                RegionRow(
                    region=name,
                    n=0,
                    decode_mean=None,
                    decode_std=None,
                    true_mean=None,
                    bias=None,
                    abs_error_mean=None,
                )
            )
            continue
        d, t = decode[mask], true[mask]
        rows.append(
            RegionRow(
                region=name,
                n=n,
                decode_mean=float(d.mean()),
                decode_std=float(d.std()),
                true_mean=float(t.mean()),
                bias=float((d - t).mean()),
                abs_error_mean=float(np.abs(d - t).mean()),
            )
        )

    true_var = float(np.var(true))
    covariance = float(np.mean((true - true.mean()) * (decode - decode.mean())))
    slope = covariance / true_var if true_var > 1e-12 else None
    return HonestyTable(
        source=source,
        n_steps=int(true.shape[0]),
        rows=tuple(rows),
        slope_decode_vs_true=slope,
        decode_vs_true_abs_error_mean=float(np.abs(decode - true).mean()),
        decode_vs_sensed_abs_error_mean=float(np.abs(decode - sensed).mean()),
        sensed_vs_true_abs_error_mean=float(np.abs(sensed - true).mean()),
        out_of_range_mass=float(np.mean((decode < 0.0) | (decode > 1.0))),
    )


# ---------------------------------------------------------------------------
# The full report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodeHonestyReport:
    """The standing instrument's full output: one table per policy source plus
    the pooled table, stamped with the schema version and the collection
    protocol so any reading is reproducible from the report alone."""

    schema_version: str
    checkpoint_label: str
    edges: RegionEdges
    per_source: tuple[HonestyTable, ...]
    pooled: HonestyTable
    seeds_per_source: int
    episodes_per_seed: int
    seed_bases: tuple[tuple[str, int], ...]


def run_decode_honesty(
    world_model: WorldModel,
    actor: Actor,
    grid_cfg: GridWorldConfig,
    *,
    checkpoint_label: str,
    seeds_per_source: int = 4,
    episodes_per_seed: int = 10,
    seed_bases: tuple[tuple[PolicySource, int], ...] = DEFAULT_SEED_BASES,
    setpoint: float = SETPOINT,
    band_halfwidth: float = BAND_HALFWIDTH,
) -> DecodeHonestyReport:
    """Run the standing instrument on a checkpoint's world model + actor.

    Checkpoint-agnostic: the caller loads whatever weight set it wants to
    audit and passes the modules; the instrument neither loads nor saves
    anything. Eval-only; deterministic given the seed protocol.
    """
    edges = region_edges_from_config(
        grid_cfg, setpoint=setpoint, band_halfwidth=band_halfwidth
    )
    bases = dict(seed_bases)
    per_source: list[HonestyTable] = []
    all_trajectories: list[TeacherForcedTrajectory] = []
    for source in POLICY_SOURCES:
        trajectories = [
            collect_teacher_forced_trajectory(
                world_model,
                actor,
                grid_cfg,
                policy_source=source,
                seed=bases[source] + s,
                episodes=episodes_per_seed,
            )
            for s in range(seeds_per_source)
        ]
        per_source.append(honesty_table(source, trajectories, edges))
        all_trajectories.extend(trajectories)

    return DecodeHonestyReport(
        schema_version=DECODE_HONESTY_SCHEMA_VERSION,
        checkpoint_label=checkpoint_label,
        edges=edges,
        per_source=tuple(per_source),
        pooled=honesty_table("pooled", all_trajectories, edges),
        seeds_per_source=seeds_per_source,
        episodes_per_seed=episodes_per_seed,
        seed_bases=tuple((s, bases[s]) for s in POLICY_SOURCES),
    )


def report_to_jsonable(report: DecodeHonestyReport) -> dict[str, Any]:
    """A JSON-serializable dict of the full report (plain ``asdict``)."""
    return asdict(report)
