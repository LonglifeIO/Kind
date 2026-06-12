"""Probe 3.5 close — archive the pre-biography prototypes' artifacts and
confirm the resting defaults. Copies (does not move — working paths cited in
the results docs stay valid); archived, not deleted.

The positive-control and σ=0-diagnostic model WEIGHTS were never persisted
(throwaway by plan: "checkpoints are not carried forward"); their archived
evidence is records + eval arrays + telemetry. The Step-0 null checkpoint is
the only persisted weight set, and it is archived here.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "runs" / "probe3_5-archive-20260612"
TMP = Path("/var/folders/vs/zc8gc4wd6h98m4xsxwkx1s700000gq/T")


def _copy(src: Path, dst: Path) -> str:
    if not src.exists():
        return f"MISSING (recorded, not copied): {src}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    return f"archived: {src} -> {dst.relative_to(REPO)}"


def main() -> None:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    p2 = REPO / "runs" / "probe3_5_phase2"
    p3 = REPO / "runs" / "probe3_5_phase3"

    # Step-0 degenerate null (the burn-in instance — the only persisted weights).
    log.append(_copy(p2 / "step0_burnin_checkpoint.pt", ARCHIVE / "step0_null" / "step0_burnin_checkpoint.pt"))
    log.append(_copy(p2 / "step0_baseline.json", ARCHIVE / "step0_null" / "step0_baseline.json"))
    log.append(_copy(TMP / "probe3_5_p2_step0_3q25cag8" / "telemetry", ARCHIVE / "step0_null" / "telemetry"))

    # Phase-2 smoke (1.0x, sigma=0.075).
    log.append(_copy(p2 / "smoke_s1_baseline.json", ARCHIVE / "phase2_smoke" / "smoke_s1_baseline.json"))
    log.append(_copy(TMP / "probe3_5_p2_smoke_mc93xsrw" / "telemetry", ARCHIVE / "phase2_smoke" / "telemetry"))

    # Phase-3 positive control (10x, sigma=0.075) — records, arrays, telemetry; no weights (throwaway by plan).
    log.append(_copy(p3 / "positive_control.json", ARCHIVE / "positive_control" / "positive_control.json"))
    log.append(_copy(p3 / "positive_control.npz", ARCHIVE / "positive_control" / "positive_control.npz"))
    log.append(_copy(p3 / "positive_control_telemetry", ARCHIVE / "positive_control" / "telemetry"))

    # Phase-3 sigma=0 diagnostic — likewise no weights.
    log.append(_copy(p3 / "positive_control_sigma0.json", ARCHIVE / "sigma0_diagnostic" / "positive_control_sigma0.json"))
    log.append(_copy(p3 / "positive_control_sigma0.npz", ARCHIVE / "sigma0_diagnostic" / "positive_control_sigma0.npz"))
    log.append(_copy(p3 / "positive_control_sigma0_telemetry", ARCHIVE / "sigma0_diagnostic" / "telemetry"))

    # Cross-cutting instantiation / check artifacts.
    log.append(_copy(p2 / "s1_instantiation.json", ARCHIVE / "s1_instantiation.json"))
    log.append(_copy(p3 / "retro_torpor_check.json", ARCHIVE / "retro_torpor_check.json"))
    log.append(_copy(REPO / "runs" / "oracle_feasibility.json", ARCHIVE / "oracle_feasibility.json"))

    # ---- resting defaults, confirmed programmatically ----
    from kind.env.grid_world import GridWorldConfig
    from kind.training.dream import DreamRolloutConfig
    from kind.training.runner import RunnerConfig

    g = GridWorldConfig()
    physics = {k: v for k, v in asdict(g).items() if k.startswith("energy_")}
    expected = {
        "energy_norm_min": 0.0,
        "energy_norm_max": 10.0,
        "energy_init": 6.0,
        "energy_base_decay": 0.08,
        "energy_move_cost": 0.04,
        "energy_replenish_per_resource": 0.8,
        "energy_obs_noise_sigma": 0.05,
        "energy_obs_lag": 1,
        "energy_obs_quantization_levels": 16,
    }
    assert physics == expected, physics
    rc_fields = {f.name: f.default for f in RunnerConfig.__dataclass_fields__.values()}
    assert rc_fields["energy_preference"] is None
    assert rc_fields["energy_telemetry"] is False
    assert DreamRolloutConfig().record_decoded_energy is False
    resting = {
        "grid_physics_defaults": physics,
        "runner_energy_preference_default": None,
        "runner_energy_telemetry_default": False,
        "dream_record_decoded_energy_default": False,
    }

    manifest = f"""# Probe 3.5 archive — 2026-06-12

Pre-biography prototype artifacts from Probe 3.5 (valence substrate),
archived at probe close per the verdict doc
(`docs/decisions/probe3_5_verdict_2026-06-12.md`). Archived, not deleted;
working copies under `runs/probe3_5_phase2/` and `runs/probe3_5_phase3/`
remain in place. Nothing in this lineage carries forward as Io.

## Contents

- `step0_null/` — the Step-0 degenerate-baseline / burn-in instance
  (epistemic-only, P3=5000, default physics). **The only persisted weights**
  (`step0_burnin_checkpoint.pt`), plus the baseline record and full training
  telemetry.
- `phase2_smoke/` — the Phase-2 mechanism smoke (1.0x precision,
  sigma=0.075): record + training telemetry. No weights persisted.
- `positive_control/` — Phase-3 positive control (10x precision,
  sigma=0.075): record, eval arrays (true energy / actions / entropy),
  training telemetry. **No weights — throwaway by plan** ("checkpoints are
  not carried forward", implementation plan Phase 3).
- `sigma0_diagnostic/` — the single pre-stated sigma=0 diagnostic at the
  same configuration: record, eval arrays, training telemetry. **No weights
  — throwaway by plan.**
- `s1_instantiation.json`, `retro_torpor_check.json`,
  `oracle_feasibility.json` — instantiation and instrument artifacts.

## Resting state at close (asserted by scripts/probe3_5_close_archive.py)

```json
{json.dumps(resting, indent=2)}
```

## Copy log

{chr(10).join("- " + line for line in log)}
"""
    (ARCHIVE / "MANIFEST.md").write_text(manifest)
    print(manifest)


if __name__ == "__main__":
    main()
