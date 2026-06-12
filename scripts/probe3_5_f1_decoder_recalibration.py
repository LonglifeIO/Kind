"""Probe 3.5 post-close — F1 demonstration: decoder-head-only recalibration
on the archived Step-0 instance.

Executes fix F1 of the seek-mechanism classification
(`docs/decisions/probe3_5_seek_classification_2026-06-12.md` §9.1) as a
*demonstration on an archived copy* — no live substrate changes. Margins are
pre-stated and builder-confirmed in
`docs/decisions/probe3_5_f1_decode_recalibration_2026-06-12.md` §3 before any
training ran.

What this script does, in order:

1. asserts the archived Step-0 checkpoint's SHA-256 (lineage evidence — the
   original is read, never written; the hash is re-asserted at the end);
2. runs the standing decode-honesty instrument (before);
3. collects the coverage mixture, teacher-forced through the frozen model:
   the instance's own greedy rail trajectories at the archived eval seed
   series (9000–9007 — the archived eval *distribution* reproduced; the
   archived run's z-sampling RNG state is not reproducible, the distribution
   is), oracle-forager in-band trajectories (9100–9107), and uniform-random
   mid-coverage (9300–9307) — equal thirds;
4. refits **only** ``energy_decoder`` on ``(h, z) → sensed_energy`` MSE.
   The S-ENV rule binds recalibration as much as original training: the
   target is ``sensed_energy``, never ``true_energy`` — true energy is
   eval-only, read by the honesty table alone. Every other parameter is
   frozen and asserted bit-identical afterward;
5. runs the instrument (after); reports out-of-range mass explicitly (the F2
   gate input — F2 decision deferred, not taken here);
6. writes the recalibrated variant as a **new artifact alongside** the
   archived original, with provenance fields, and the before/after reports
   to a results JSON.

Provenance note, recorded plainly: the head saw oracle-policy states no
Io-lineage instance ever visited — that is the point of coverage, and it is
recorded, pre-biography, carrying nothing forward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from kind.agents.actor import Actor
from kind.agents.world_model import WorldModel, WorldModelConfig
from kind.env.grid_world import GridWorldConfig
from kind.observer.decode_honesty import (
    DecodeHonestyReport,
    HonestyTable,
    PolicySource,
    TeacherForcedTrajectory,
    collect_teacher_forced_trajectory,
    report_to_jsonable,
    run_decode_honesty,
)

#: The archived Step-0 checkpoint's SHA-256, asserted before and after.
EXPECTED_STEP0_SHA256 = (
    "9bddae31c5c8e51c3b470890337dea01d0a76a5622efafaaf7dbfd150d1de68b"
)

#: Coverage-mixture collection (training data; analysis-instrument choices).
#: Disjoint from the instrument's evaluation seed bases (9700/9800/9900), so
#: the before/after tables are out-of-sample with respect to the env worlds
#: the head was fit on.
TRAIN_SEED_BASES: tuple[tuple[PolicySource, int], ...] = (
    ("own_policy", 9000),
    ("oracle", 9100),
    ("uniform_random", 9300),
)
TRAIN_SEEDS_PER_SOURCE = 8
TRAIN_EPISODES_PER_SEED = 10

#: Head-refit hyperparameters (a 216→64→1 MLP on ~48k pairs; not swept).
TRAIN_TORCH_SEED = 4321
EPOCHS = 50
BATCH_SIZE = 512
LEARNING_RATE = 1e-3

#: Pre-stated demonstration margins ([pre, builder-confirmed 2026-06-12];
#: doc §3). M1: oracle-source in-band |bias|; M2: pooled decode~true slope
#: sign/range; M3: own-policy floor-adjacent abs-error no-regression
#: tolerance (the rail readings were the one thing the broken decoder got
#: approximately right).
M1_IN_BAND_ABS_BIAS_MAX = 0.05
M2_SLOPE_RANGE = (0.5, 1.5)
M3_RAIL_REGRESSION_TOLERANCE = 0.02


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_step0(
    path: Path,
) -> tuple[WorldModel, Actor, GridWorldConfig, dict[str, Any]]:
    ckpt: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    wm = WorldModel(WorldModelConfig(energy_dedicated_dims=0))
    wm.load_state_dict(ckpt["world_model"])
    wm.eval()
    actor = Actor(h_dim=200, z_dim=16, action_dim=5, mlp_hidden=200)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    grid_cfg = GridWorldConfig(**ckpt["grid_config"])
    return wm, actor, grid_cfg, ckpt


def collect_training_mixture(
    wm: WorldModel, actor: Actor, grid_cfg: GridWorldConfig
) -> tuple[list[TeacherForcedTrajectory], dict[str, int]]:
    """The coverage mixture: equal thirds of own-rail / oracle / uniform."""
    trajectories: list[TeacherForcedTrajectory] = []
    counts: dict[str, int] = {}
    for source, base in TRAIN_SEED_BASES:
        n_before = sum(t.true_energy.shape[0] for t in trajectories)
        for s in range(TRAIN_SEEDS_PER_SOURCE):
            print(f"[mixture] {source} seed {base + s}")
            trajectories.append(
                collect_teacher_forced_trajectory(
                    wm,
                    actor,
                    grid_cfg,
                    policy_source=source,
                    seed=base + s,
                    episodes=TRAIN_EPISODES_PER_SEED,
                )
            )
        counts[source] = (
            sum(t.true_energy.shape[0] for t in trajectories) - n_before
        )
    return trajectories, counts


def recalibrate_energy_decoder_head(
    wm: WorldModel, trajectories: list[TeacherForcedTrajectory]
) -> dict[str, Any]:
    """Refit only ``energy_decoder`` on (h, z) → sensed MSE; freeze the rest.

    The latents were teacher-forced under the frozen model and the decoder is
    a passive emission ((h, z) do not depend on its weights), so fitting on
    pre-collected latents is exact, not an approximation.
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
        name for name, _ in wm.named_parameters() if name.startswith("energy_decoder.")
    }
    frozen_snapshot = {
        name: p.detach().clone()
        for name, p in wm.named_parameters()
        if name not in head_param_names
    }
    for name, p in wm.named_parameters():
        p.requires_grad_(name in head_param_names)

    torch.manual_seed(TRAIN_TORCH_SEED)
    optimizer = torch.optim.Adam(wm.energy_decoder.parameters(), lr=LEARNING_RATE)
    wm.energy_decoder.train()
    epoch_mse: list[float] = []
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        losses: list[float] = []
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            pred: Tensor = wm.energy_decoder(latent[idx])
            loss: Tensor = torch.nn.functional.mse_loss(pred, target[idx])
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            losses.append(float(loss.item()))
        epoch_mse.append(float(np.mean(losses)))
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"[refit] epoch {epoch + 1}/{EPOCHS} mse {epoch_mse[-1]:.6f}")
    wm.energy_decoder.eval()
    for p in wm.parameters():
        p.requires_grad_(True)

    # Head-only discipline, asserted: every non-head parameter bit-identical.
    for name, p in wm.named_parameters():
        if name not in head_param_names:
            if not torch.equal(p.detach(), frozen_snapshot[name]):
                raise AssertionError(f"non-head parameter changed: {name}")
    print(f"[refit] {len(frozen_snapshot)} non-head tensors asserted bit-identical")

    return {
        "n_pairs": int(n),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "torch_seed": TRAIN_TORCH_SEED,
        "first_epoch_mse": epoch_mse[0],
        "final_epoch_mse": epoch_mse[-1],
        "target": "sensed_energy (S-ENV: true_energy enters no training loss)",
    }


def _table_lookup(report: DecodeHonestyReport, source: str) -> HonestyTable:
    if source == "pooled":
        return report.pooled
    for table in report.per_source:
        if table.source == source:
            return table
    raise KeyError(source)


def _region_value(
    table: HonestyTable, region: str, field: str
) -> float | None:
    for row in table.rows:
        if row.region == region:
            value = getattr(row, field)
            return None if value is None else float(value)
    raise KeyError(region)


def evaluate_margins(
    before: DecodeHonestyReport, after: DecodeHonestyReport
) -> dict[str, Any]:
    """Judge the after-table against the pre-stated M1–M3."""
    in_band_bias = _region_value(_table_lookup(after, "oracle"), "in_band", "bias")
    m1_value = None if in_band_bias is None else abs(in_band_bias)
    m1_pass = m1_value is not None and m1_value <= M1_IN_BAND_ABS_BIAS_MAX

    slope = _table_lookup(after, "pooled").slope_decode_vs_true
    m2_pass = slope is not None and M2_SLOPE_RANGE[0] <= slope <= M2_SLOPE_RANGE[1]

    rail_before = _region_value(
        _table_lookup(before, "own_policy"), "floor_adjacent", "abs_error_mean"
    )
    rail_after = _region_value(
        _table_lookup(after, "own_policy"), "floor_adjacent", "abs_error_mean"
    )
    m3_pass = (
        rail_before is not None
        and rail_after is not None
        and rail_after <= rail_before + M3_RAIL_REGRESSION_TOLERANCE
    )

    return {
        "M1_in_band_abs_bias": {
            "value": m1_value,
            "max": M1_IN_BAND_ABS_BIAS_MAX,
            "source": "oracle",
            "passed": m1_pass,
        },
        "M2_pooled_slope": {
            "value": slope,
            "range": list(M2_SLOPE_RANGE),
            "passed": m2_pass,
        },
        "M3_rail_no_regression": {
            "before_abs_error": rail_before,
            "after_abs_error": rail_after,
            "tolerance": M3_RAIL_REGRESSION_TOLERANCE,
            "source": "own_policy",
            "region": "floor_adjacent",
            "passed": m3_pass,
        },
        "all_passed": m1_pass and m2_pass and m3_pass,
    }


def out_of_range_summary(report: DecodeHonestyReport) -> dict[str, float]:
    """The F2 gate input: decode mass outside [0, 1], per source and pooled."""
    summary = {t.source: t.out_of_range_mass for t in report.per_source}
    summary["pooled"] = report.pooled.out_of_range_mass
    return summary


def format_table_markdown(table: HonestyTable) -> str:
    lines = [
        f"**{table.source}** (n={table.n_steps}; slope "
        f"{'—' if table.slope_decode_vs_true is None else f'{table.slope_decode_vs_true:.3f}'}; "
        f"|decode−true| {table.decode_vs_true_abs_error_mean:.3f}, "
        f"|decode−sensed| {table.decode_vs_sensed_abs_error_mean:.3f}, "
        f"|sensed−true| {table.sensed_vs_true_abs_error_mean:.3f}; "
        f"out-of-range {table.out_of_range_mass:.4f})",
        "",
        "| region | n | decode mean | decode std | true mean | bias | mean \\|err\\| |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in table.rows:
        if r.n == 0:
            lines.append(f"| {r.region} | 0 | — | — | — | — | — |")
        else:
            lines.append(
                f"| {r.region} | {r.n} | {r.decode_mean:.3f} | {r.decode_std:.3f} "
                f"| {r.true_mean:.3f} | {r.bias:+.3f} | {r.abs_error_mean:.3f} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archived-checkpoint",
        type=str,
        default=(
            "runs/probe3_5-archive-20260612/step0_null/step0_burnin_checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--recalibrated-out",
        type=str,
        default=(
            "runs/probe3_5-archive-20260612/step0_null/"
            "step0_f1_recalibrated_checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--out", type=str, default="runs/probe3_5_f1_recalibration/results.json"
    )
    args = parser.parse_args()

    archived = Path(args.archived_checkpoint)
    sha_before = _sha256(archived)
    if sha_before != EXPECTED_STEP0_SHA256:
        raise AssertionError(
            f"archived Step-0 checkpoint hash mismatch: {sha_before}"
        )
    print(f"[lineage] archived checkpoint sha256 asserted: {sha_before[:16]}…")

    wm, actor, grid_cfg, ckpt = _load_step0(archived)

    print("[instrument] before-recalibration honesty table")
    before = run_decode_honesty(
        wm, actor, grid_cfg, checkpoint_label="step0_burnin_checkpoint.pt (archived)"
    )

    trajectories, mixture_counts = collect_training_mixture(wm, actor, grid_cfg)
    print(f"[mixture] counts per source: {mixture_counts}")
    training = recalibrate_energy_decoder_head(wm, trajectories)

    print("[instrument] after-recalibration honesty table")
    after = run_decode_honesty(
        wm,
        actor,
        grid_cfg,
        checkpoint_label="step0_f1_recalibrated_checkpoint.pt",
    )

    margins = evaluate_margins(before, after)
    f2_gate_input = out_of_range_summary(after)

    sha_after = _sha256(archived)
    if sha_after != sha_before:
        raise AssertionError("archived Step-0 checkpoint was modified")
    print("[lineage] archived checkpoint byte-intact after run")

    recal_path = Path(args.recalibrated_out)
    recal_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **ckpt,
            "world_model": wm.state_dict(),
            "note": (
                "Probe 3.5 F1 demonstration: Step-0 with energy_decoder head "
                "recalibrated on a coverage mixture (own-rail + oracle + "
                "uniform-random), target sensed_energy. All non-head tensors "
                "bit-identical to step0_burnin_checkpoint.pt. Archived-copy "
                "artifact — not a live substrate; nothing carries forward."
            ),
            "f1_provenance": {
                "base_checkpoint_sha256": sha_before,
                "training_data": dict(
                    (s, b) for s, b in TRAIN_SEED_BASES
                ),
                "seeds_per_source": TRAIN_SEEDS_PER_SOURCE,
                "episodes_per_seed": TRAIN_EPISODES_PER_SEED,
                "mixture_counts": mixture_counts,
                "training": training,
                "decision_doc": (
                    "docs/decisions/"
                    "probe3_5_f1_decode_recalibration_2026-06-12.md"
                ),
            },
        },
        recal_path,
    )
    print(f"[artifact] recalibrated variant written: {recal_path}")

    out: dict[str, Any] = {
        "LABEL": (
            "F1 demonstration — decoder-head-only recalibration on the "
            "archived Step-0 instance. Margins pre-stated in "
            "docs/decisions/probe3_5_f1_decode_recalibration_2026-06-12.md §3."
        ),
        "archived_checkpoint_sha256": sha_before,
        "recalibrated_checkpoint": str(recal_path),
        "mixture_counts": mixture_counts,
        "training": training,
        "before": report_to_jsonable(before),
        "after": report_to_jsonable(after),
        "margins": margins,
        "f2_gate_input_out_of_range_mass": f2_gate_input,
        "grid_config": asdict(grid_cfg),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[out] wrote {out_path}")

    print("\n=== BEFORE ===")
    for table in (*before.per_source, before.pooled):
        print(format_table_markdown(table))
        print()
    print("=== AFTER ===")
    for table in (*after.per_source, after.pooled):
        print(format_table_markdown(table))
        print()
    print("=== MARGINS ===")
    print(json.dumps(margins, indent=2))
    print("=== F2 GATE INPUT (out-of-range mass, after) ===")
    print(json.dumps(f2_gate_input, indent=2))


if __name__ == "__main__":
    main()
