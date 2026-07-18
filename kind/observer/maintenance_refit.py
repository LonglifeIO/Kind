"""Probe 4.5 S-DEC — scheduled decoder-head maintenance refit + honesty gate.

Promotes the F1 demonstration (``scripts/probe3_5_f1_decoder_recalibration.py``,
run once on the archived Step-0 copy) into the reusable in-run harness the
implementation plan's Phase 1 names: at a pre-registered cadence, refit **only**
``energy_decoder`` on ``(h, z) → sensed_energy`` coverage pairs teacher-forced
through the current frozen snapshot, then judge the head against the frozen
physics honesty criterion (prereg §3) on the standing ``decode_honesty``
instrument. Authority: ``docs/decisions/probe4_5_preregistration_2026-07-13.md``
§3 and ``docs/plans/Kind_probe4_5_implementation_plan.md`` §S-DEC.

**Calibrate for reading, never for driving** binds as three structural rules:

(i)   the calibration target is ``sensed_energy``-match on a coverage mixture
      (own-policy + oracle + uniform-random, equal thirds — the F1 mixture) —
      never any behavioral outcome; the S-ENV rule binds every refit
      (``true_energy`` enters no training loss, here or anywhere);
(ii)  the cadence is pre-registered (§3: every 10k env steps,
      checkpoint-aligned, plus one at burn-in close) — never
      behavior-triggered;
(iii) validation is the frozen §3 margins on the standing honesty table —
      never whether Io regulates or allocates. "Recalibrate until Io
      regulates" is the named fitted failure mode; this structure forbids it.

Margin realization (the §3 text, mapped to the instrument's tables): the
criterion is judged on the **oracle-source table** (the out-of-distribution
keystone, per D1) — in-band mean ``|decode − true|`` and the per-region
coverage-qualified ``|bias|`` bounds read there — except where §3 names the
pooled table explicitly (the decode~true slope). Out-of-range mass is judged
on the **pooled** table: it covers every evaluated source including oracle, so
this reading is at least as strict as any single-table reading and cannot
weaken the frozen criterion.

**Honesty-STOP** (§3): if any margin fails after a scheduled refit, one
diagnostic re-collection (larger coverage mixture, same margins) is permitted;
a second failure raises :class:`HonestyStopError` — the run stops; the probe
closes as *instrument-cannot-be-made-honest*, a finding. Margins are never
revised against behavior.

Live-model discipline (what the F1 script, running on a dead copy, did not
need): every non-head parameter is asserted bit-identical after the refit;
each parameter's ``requires_grad`` flag is snapshotted and restored **exactly**
(the EMA target siblings ``target_encoder`` / ``target_gru_cell`` carry
``requires_grad=False`` by design — a blanket re-enable would let the runner's
optimizer start updating them); module train/eval modes and device placement
are restored; refit-local gradients are cleared. The refit runs on CPU (the
collector's device) and moves the modules back afterward — callers invoke it
only at a pause point (the runner's checkpoint-aligned boundary), never
mid-step.

Observer-side in the sense that matters: nothing here adds an Io-readable
interface — no observation marker, no PolicyView change, no new head. The one
thing it touches is the read head it exists to repair.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from kind.agents.actor import Actor
from kind.agents.world_model import WorldModel
from kind.env.grid_world import GridWorldConfig
from kind.observer.decode_honesty import (
    DecodeHonestyReport,
    HonestyTable,
    PolicySource,
    TeacherForcedTrajectory,
    collect_teacher_forced_trajectory,
    run_decode_honesty,
)

__all__ = [
    "MAINTENANCE_REFIT_SCHEMA_VERSION",
    "RefitOccasion",
    "HonestyMargins",
    "MarginCheck",
    "MarginVerdict",
    "MaintenanceRefitConfig",
    "RefitStats",
    "MaintenanceRefitReport",
    "HonestyStopError",
    "evaluate_honesty_margins",
    "collect_coverage_mixture",
    "refit_energy_decoder_head",
    "run_maintenance_refit",
    "run_scheduled_maintenance",
    "maintenance_report_path",
    "report_to_jsonable",
]

#: Versioned output schema (house pattern).
MAINTENANCE_REFIT_SCHEMA_VERSION: str = "0.1.0"

RefitOccasion = Literal["scheduled", "burn_in_close", "diagnostic_recollection"]


# ---------------------------------------------------------------------------
# The frozen §3 margins
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HonestyMargins:
    """The prereg §3 physics honesty criterion, as margins on the standing
    ``decode_honesty`` readouts. Defaults are the FROZEN 2026-07-13 values —
    they are configurable only so tests can exercise both gate branches;
    Probe 4.5 runs use the defaults, and revising them against behavior is
    forbidden by the prereg.
    """

    #: Oracle-source in-band mean |decode − true| (D1's bin-1a trigger was
    #: > 0.15; the repair must beat the defect threshold with margin).
    oracle_in_band_abs_error_max: float = 0.10
    #: Pooled decode~true OLS slope (D1 trigger was < 0.5; sign-inversion was
    #: the defect).
    pooled_slope_min: float = 0.7
    #: Pooled decode mass outside the physical [0, 1] (structurally ~0 under
    #: the adopted F2 bounded head; kept so the criterion does not depend on
    #: F2).
    pooled_out_of_range_mass_max: float = 0.01
    #: Oracle-source per-region mean |bias| in every named region with at
    #: least ``region_min_n`` samples (coverage-qualified; the decoder may not
    #: lie by more than one band-halfwidth anywhere it can be measured).
    oracle_region_abs_bias_max: float = 0.15
    region_min_n: int = 500


@dataclass(frozen=True)
class MarginCheck:
    """One margin's verdict. ``value`` is ``None`` when the readout itself is
    undefined (no coverage / degenerate series) — an undefined keystone
    readout fails: honesty that cannot be measured is not demonstrated."""

    name: str
    value: float | None
    limit: float
    passed: bool


@dataclass(frozen=True)
class MarginVerdict:
    """The full §3 judgment on one honesty report. Regions on the oracle
    table below the coverage floor are reported, not hidden — and not
    graded."""

    checks: tuple[MarginCheck, ...]
    coverage_skipped_regions: tuple[str, ...]
    all_passed: bool


def _source_table(report: DecodeHonestyReport, source: str) -> HonestyTable:
    for table in report.per_source:
        if table.source == source:
            return table
    raise KeyError(source)


def evaluate_honesty_margins(
    report: DecodeHonestyReport, margins: HonestyMargins
) -> MarginVerdict:
    """Judge a ``decode_honesty`` report against the frozen §3 margins."""
    oracle = _source_table(report, "oracle")
    checks: list[MarginCheck] = []
    skipped: list[str] = []

    in_band = next(r for r in oracle.rows if r.region == "in_band")
    checks.append(
        MarginCheck(
            name="oracle_in_band_abs_error",
            value=in_band.abs_error_mean,
            limit=margins.oracle_in_band_abs_error_max,
            passed=(
                in_band.abs_error_mean is not None
                and in_band.abs_error_mean <= margins.oracle_in_band_abs_error_max
            ),
        )
    )

    slope = report.pooled.slope_decode_vs_true
    checks.append(
        MarginCheck(
            name="pooled_slope_decode_vs_true",
            value=slope,
            limit=margins.pooled_slope_min,
            passed=slope is not None and slope >= margins.pooled_slope_min,
        )
    )

    checks.append(
        MarginCheck(
            name="pooled_out_of_range_mass",
            value=report.pooled.out_of_range_mass,
            limit=margins.pooled_out_of_range_mass_max,
            passed=(
                report.pooled.out_of_range_mass
                <= margins.pooled_out_of_range_mass_max
            ),
        )
    )

    for row in oracle.rows:
        if row.n < margins.region_min_n:
            skipped.append(row.region)
            continue
        abs_bias = None if row.bias is None else abs(row.bias)
        checks.append(
            MarginCheck(
                name=f"oracle_region_abs_bias:{row.region}",
                value=abs_bias,
                limit=margins.oracle_region_abs_bias_max,
                passed=(
                    abs_bias is not None
                    and abs_bias <= margins.oracle_region_abs_bias_max
                ),
            )
        )

    return MarginVerdict(
        checks=tuple(checks),
        coverage_skipped_regions=tuple(skipped),
        all_passed=all(c.passed for c in checks),
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

#: The F1 coverage-mixture seed bases — disjoint from the honesty instrument's
#: evaluation bases (9700/9800/9900), so every gate table is out-of-sample
#: with respect to the worlds the head was fit on.
DEFAULT_TRAIN_SEED_BASES: tuple[tuple[PolicySource, int], ...] = (
    ("own_policy", 9000),
    ("oracle", 9100),
    ("uniform_random", 9300),
)


@dataclass(frozen=True)
class MaintenanceRefitConfig:
    """The harness's full protocol. Defaults are the F1 pattern / prereg §3
    values; ``grid_world_config`` and ``out_dir`` have no sane defaults and
    must come from the run's own config (the collection worlds are fresh
    instances built from the same ``GridWorldConfig`` the live env uses).
    """

    grid_world_config: GridWorldConfig
    #: Machine-written per-refit reports land here (one JSON per cycle).
    out_dir: Path
    #: Prereg §3 cadence: every 10k env steps, checkpoint-aligned. The runner
    #: validates alignment at construction.
    refit_every_n_env_steps: int = 10_000
    margins: HonestyMargins = HonestyMargins()

    # Coverage mixture (the F1 pattern: equal thirds by seeds-per-source).
    train_seed_bases: tuple[tuple[PolicySource, int], ...] = field(
        default=DEFAULT_TRAIN_SEED_BASES
    )
    train_seeds_per_source: int = 8
    train_episodes_per_seed: int = 10
    #: The §3 diagnostic re-collection's larger mixture (same margins).
    diagnostic_seeds_per_source: int = 16

    # Head-refit hyperparameters (the F1 values; not swept).
    refit_epochs: int = 50
    refit_batch_size: int = 512
    refit_learning_rate: float = 1e-3
    refit_torch_seed: int = 4321

    # Honesty-gate collection protocol (the instrument's own defaults).
    honesty_seeds_per_source: int = 4
    honesty_episodes_per_seed: int = 10

    #: Amendment 1 (2026-07-18, fault-on instances): when False, the
    #: runner-hook's mid-run scheduled refits are non-binding — the refit
    #: runs, the report is written, a margin failure is recorded but
    #: triggers no diagnostic and no STOP. The burn-in-close call binds
    #: explicitly regardless (the run scripts pass ``binding=True`` there).
    #: True (the default) is the original frozen §3 rule.
    stop_binding: bool = True


# ---------------------------------------------------------------------------
# Collection + refit (the F1 mechanics, made reusable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefitStats:
    n_pairs: int
    epochs: int
    batch_size: int
    learning_rate: float
    torch_seed: int
    first_epoch_mse: float
    final_epoch_mse: float
    #: The S-ENV rule, stamped on every record.
    target: str = "sensed_energy (S-ENV: true_energy enters no training loss)"


def collect_coverage_mixture(
    world_model: WorldModel,
    actor: Actor,
    grid_cfg: GridWorldConfig,
    *,
    seed_bases: tuple[tuple[PolicySource, int], ...],
    seeds_per_source: int,
    episodes_per_seed: int,
) -> tuple[list[TeacherForcedTrajectory], dict[str, int]]:
    """The coverage mixture, teacher-forced through the frozen snapshot."""
    trajectories: list[TeacherForcedTrajectory] = []
    counts: dict[str, int] = {}
    for source, base in seed_bases:
        n_before = sum(t.true_energy.shape[0] for t in trajectories)
        for s in range(seeds_per_source):
            trajectories.append(
                collect_teacher_forced_trajectory(
                    world_model,
                    actor,
                    grid_cfg,
                    policy_source=source,
                    seed=base + s,
                    episodes=episodes_per_seed,
                )
            )
        counts[source] = (
            sum(t.true_energy.shape[0] for t in trajectories) - n_before
        )
    return trajectories, counts


def refit_energy_decoder_head(
    world_model: WorldModel,
    trajectories: list[TeacherForcedTrajectory],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    torch_seed: int,
) -> RefitStats:
    """Refit **only** ``energy_decoder`` on ``(h, z) → sensed`` MSE.

    The latents were teacher-forced under the frozen model and the decoder is
    a passive emission (``(h, z)`` do not depend on its weights), so fitting
    on pre-collected latents is exact. Every non-head parameter is asserted
    bit-identical afterward; every parameter's ``requires_grad`` flag is
    restored exactly (the EMA target siblings stay frozen); refit-local
    gradients are cleared so the runner's next ``zero_grad``-free step cannot
    see them.
    """
    h = torch.from_numpy(np.concatenate([t.h for t in trajectories]))
    z = torch.from_numpy(np.concatenate([t.z for t in trajectories]))
    sensed = torch.from_numpy(
        np.concatenate([t.sensed_energy for t in trajectories])
    ).to(torch.float32)
    latent = torch.cat([h, z], dim=-1)
    target = sensed.unsqueeze(-1)
    n = latent.shape[0]

    head_param_names = {
        name
        for name, _ in world_model.named_parameters()
        if name.startswith("energy_decoder.")
    }
    frozen_snapshot = {
        name: p.detach().clone()
        for name, p in world_model.named_parameters()
        if name not in head_param_names
    }
    requires_grad_snapshot = {
        name: p.requires_grad for name, p in world_model.named_parameters()
    }
    was_training = world_model.training
    for name, p in world_model.named_parameters():
        p.requires_grad_(name in head_param_names)

    torch.manual_seed(torch_seed)
    optimizer = torch.optim.Adam(
        world_model.energy_decoder.parameters(), lr=learning_rate
    )
    world_model.energy_decoder.train()
    epoch_mse: list[float] = []
    for _epoch in range(epochs):
        perm = torch.randperm(n)
        losses: list[float] = []
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pred: Tensor = world_model.energy_decoder(latent[idx])
            loss: Tensor = torch.nn.functional.mse_loss(pred, target[idx])
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            losses.append(float(loss.item()))
        epoch_mse.append(float(np.mean(losses)))

    # Restore live-model state exactly: modes, per-parameter requires_grad,
    # and no refit-local gradients left behind.
    world_model.energy_decoder.eval()
    world_model.train(was_training)
    for name, p in world_model.named_parameters():
        p.requires_grad_(requires_grad_snapshot[name])
    for p in world_model.energy_decoder.parameters():
        p.grad = None

    # Head-only discipline, asserted: every non-head parameter bit-identical.
    for name, p in world_model.named_parameters():
        if name not in head_param_names:
            if not torch.equal(p.detach(), frozen_snapshot[name]):
                raise AssertionError(f"non-head parameter changed: {name}")

    return RefitStats(
        n_pairs=int(n),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        torch_seed=torch_seed,
        first_epoch_mse=epoch_mse[0],
        final_epoch_mse=epoch_mse[-1],
    )


# ---------------------------------------------------------------------------
# One full cycle: before-table → refit → after-table → verdict → report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaintenanceRefitReport:
    """One maintenance cycle's machine-written record: before/after honesty
    tables (the Phase 1 journal deliverable — the three-way error comparison
    trend reads off these), the refit stats, and the §3 verdict."""

    schema_version: str
    env_step: int
    occasion: RefitOccasion
    run_label: str
    mixture_counts: dict[str, int]
    refit: RefitStats
    before: DecodeHonestyReport
    after: DecodeHonestyReport
    verdict: MarginVerdict


def report_to_jsonable(report: MaintenanceRefitReport) -> dict[str, object]:
    return asdict(report)


def maintenance_report_path(
    out_dir: Path, *, env_step: int, occasion: RefitOccasion
) -> Path:
    return out_dir / f"refit_{env_step:08d}_{occasion}.json"


def run_maintenance_refit(
    world_model: WorldModel,
    actor: Actor,
    config: MaintenanceRefitConfig,
    *,
    env_step: int,
    occasion: RefitOccasion,
    run_label: str = "live",
) -> MaintenanceRefitReport:
    """One full maintenance cycle at a pause point; writes the report JSON.

    Moves the modules to CPU for collection (the collector's device) and
    restores device placement and train/eval modes afterward — parameter
    object identity is preserved by in-place ``Module.to``, so the runner's
    optimizer state stays valid. Callers must be at a step boundary.
    """
    seeds_per_source = (
        config.diagnostic_seeds_per_source
        if occasion == "diagnostic_recollection"
        else config.train_seeds_per_source
    )
    wm_device = next(world_model.parameters()).device
    actor_device = next(actor.parameters()).device
    wm_training, actor_training = world_model.training, actor.training
    world_model.to("cpu")
    actor.to("cpu")
    try:
        before = run_decode_honesty(
            world_model,
            actor,
            config.grid_world_config,
            checkpoint_label=f"{run_label}@{env_step}:pre_refit",
            seeds_per_source=config.honesty_seeds_per_source,
            episodes_per_seed=config.honesty_episodes_per_seed,
        )
        trajectories, mixture_counts = collect_coverage_mixture(
            world_model,
            actor,
            config.grid_world_config,
            seed_bases=config.train_seed_bases,
            seeds_per_source=seeds_per_source,
            episodes_per_seed=config.train_episodes_per_seed,
        )
        refit = refit_energy_decoder_head(
            world_model,
            trajectories,
            epochs=config.refit_epochs,
            batch_size=config.refit_batch_size,
            learning_rate=config.refit_learning_rate,
            torch_seed=config.refit_torch_seed,
        )
        after = run_decode_honesty(
            world_model,
            actor,
            config.grid_world_config,
            checkpoint_label=f"{run_label}@{env_step}:post_refit",
            seeds_per_source=config.honesty_seeds_per_source,
            episodes_per_seed=config.honesty_episodes_per_seed,
        )
    finally:
        world_model.to(wm_device)
        actor.to(actor_device)
        world_model.train(wm_training)
        actor.train(actor_training)

    report = MaintenanceRefitReport(
        schema_version=MAINTENANCE_REFIT_SCHEMA_VERSION,
        env_step=env_step,
        occasion=occasion,
        run_label=run_label,
        mixture_counts=mixture_counts,
        refit=refit,
        before=before,
        after=after,
        verdict=evaluate_honesty_margins(after, config.margins),
    )
    config.out_dir.mkdir(parents=True, exist_ok=True)
    path = maintenance_report_path(
        config.out_dir, env_step=env_step, occasion=occasion
    )
    path.write_text(json.dumps(report_to_jsonable(report), indent=2))
    return report


# ---------------------------------------------------------------------------
# The scheduled entry with the §3 STOP rule
# ---------------------------------------------------------------------------


class HonestyStopError(RuntimeError):
    """Prereg §3 honesty-STOP: margins failed after a scheduled refit AND
    after the one permitted diagnostic re-collection. The run stops; the
    probe closes as *instrument-cannot-be-made-honest* — a finding, recorded,
    never tuned away. Carries both failing reports."""

    def __init__(self, reports: tuple[MaintenanceRefitReport, ...]) -> None:
        self.reports = reports
        failed = [c.name for c in reports[-1].verdict.checks if not c.passed]
        super().__init__(
            f"honesty-STOP at env_step={reports[-1].env_step}: margins failed "
            f"after refit and diagnostic re-collection ({', '.join(failed)})"
        )


def run_scheduled_maintenance(
    world_model: WorldModel,
    actor: Actor,
    config: MaintenanceRefitConfig,
    *,
    env_step: int,
    occasion: RefitOccasion = "scheduled",
    run_label: str = "live",
    binding: bool = True,
) -> tuple[MaintenanceRefitReport, ...]:
    """One scheduled maintenance event under the §3 STOP rule.

    Binding (the original frozen rule): refit + gate; on a margin failure,
    exactly one diagnostic re-collection (larger coverage mixture, same
    margins); on a second failure, raise :class:`HonestyStopError`.
    Non-binding (Amendment 1, mid-burn-in on fault-on instances): the refit
    runs and its report is written — a failure is recorded, nothing more.
    Returns every report produced.
    """
    report = run_maintenance_refit(
        world_model,
        actor,
        config,
        env_step=env_step,
        occasion=occasion,
        run_label=run_label,
    )
    if report.verdict.all_passed or not binding:
        return (report,)
    diagnostic = run_maintenance_refit(
        world_model,
        actor,
        config,
        env_step=env_step,
        occasion="diagnostic_recollection",
        run_label=run_label,
    )
    if diagnostic.verdict.all_passed:
        return (report, diagnostic)
    raise HonestyStopError((report, diagnostic))
