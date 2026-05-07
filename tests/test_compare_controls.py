"""Phase 8 sanity tests: ``scripts/probe1_5_compare_controls.py`` exists,
exposes ``main()`` + the public-by-name analysis helpers, and produces the
expected output shape on a synthetic four-directory setup.

The full four-way analysis runs against the real ``runs/`` directories;
that is the load-bearing reading the journal entry interprets. What
pytest does is the lean discipline check: the script's structural
invariants — public surface, KS correctness on synthetic data, output
file shape — hold. The Phase 5 ``test_probe1_5_control_frozen_target``
loader pattern (sys.modules registration before ``exec_module``)
carries through unchanged.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from safetensors.torch import save_file

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "probe1_5_compare_controls.py"
_MODULE_NAME = "kind_probe1_5_compare_controls_module"


def _load_script_module() -> ModuleType:
    """Load the script as a fresh module under a unique name.

    Registration in ``sys.modules`` before ``exec_module`` is the trap
    Phase 5's test journal documented; re-applied here even though the
    script does not currently define ``@dataclass`` classes that would
    re-trigger the failure mode (the @dataclass classes here use plain
    field defaults; the registration is defense-in-depth)."""
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---- structural-existence tests ------------------------------------------


def test_script_file_exists() -> None:
    assert _SCRIPT_PATH.is_file(), (
        f"plan §9.2 (extended four-way per §13 lean revision) expects "
        f"{_SCRIPT_PATH} to exist as a runnable comparison driver"
    )


def test_script_exposes_main_callable() -> None:
    module = _load_script_module()
    assert hasattr(module, "main") and callable(module.main)


def test_script_exposes_public_helpers() -> None:
    """``build_summary_text`` and ``load_run_stats`` are public-by-name
    so the test (and any later eyeball helper) can invoke them
    directly without going through the CLI."""
    module = _load_script_module()
    for name in ("build_summary_text", "load_run_stats"):
        assert hasattr(module, name), f"compare_controls must expose {name}(...)"
        assert callable(getattr(module, name))


# ---- KS-correctness tests (synthetic) ------------------------------------


def test_ks_two_sample_identical_distributions_p_near_one() -> None:
    """Two samples drawn from the same distribution should produce a
    high p-value. Setting both samples to the same array gives D=0
    exactly and p=1 by the Smirnov asymptotic formula."""
    module = _load_script_module()
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, size=500)
    b = a.copy()
    d, p = module._ks_two_sample(a, b)
    assert d == 0.0
    assert p == pytest.approx(1.0, abs=1e-9)


def test_ks_two_sample_distinct_distributions_p_below_threshold() -> None:
    """Two samples from clearly different distributions should produce
    a p-value below the plan's 0.05 threshold."""
    module = _load_script_module()
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, size=500)
    b = rng.normal(2, 1, size=500)
    d, p = module._ks_two_sample(a, b)
    assert d > 0.5  # two normals shifted by 2σ have D > 0.5
    assert p < 1e-10


def test_ks_two_sample_known_value() -> None:
    """A handcrafted case with a known KS statistic: a = uniform(0, 1)
    in 100 evenly-spaced points; b = uniform(0, 1) shifted by +0.5
    truncated to (0, 1). The shift produces D ≈ 0.5 by construction
    (the b-sample's mass below 0.5 vanishes; the a-sample's CDF at
    0.5 is 0.5). This is a numerical anchor for the implementation."""
    module = _load_script_module()
    a = np.linspace(0.005, 0.995, 100)
    b = np.linspace(0.505, 0.995, 50)  # 0.5 of the support, only the upper half
    d, p = module._ks_two_sample(a, b)
    # D ≈ 0.5 exact for this construction.
    assert d == pytest.approx(0.5, abs=0.02)
    assert p < 0.01


# ---- synthetic four-directory setup --------------------------------------


def _write_synthetic_run(
    run_dir: Path,
    *,
    schema_version: str,
    n_episodes: int = 25,
    steps_per_episode: int = 200,
    kl_distribution: tuple[float, float] = (12.0, 1.0),
    recon_distribution: tuple[float, float] = (30.0, 5.0),
    sp_distribution: tuple[float, float] | None = None,
    seed: int = 0,
) -> None:
    """Write one synthetic run directory with parquet telemetry +
    safetensors checkpoint. The schema mirrors the real schema's
    columns: kl_aggregate_t, recon_loss_t, episode_id,
    schema_version, plus (when sp_distribution is not None) the
    self_prediction_error_t and self_prediction_error_masked_t fields.
    """
    rng = np.random.default_rng(seed)
    n_total = n_episodes * steps_per_episode
    episode_id = np.repeat(np.arange(n_episodes), steps_per_episode)
    step_in_ep = np.tile(np.arange(steps_per_episode), n_episodes)
    kl = rng.normal(kl_distribution[0], kl_distribution[1], size=n_total)
    recon = rng.normal(recon_distribution[0], recon_distribution[1], size=n_total)
    sv = np.full(n_total, schema_version, dtype=object)

    columns: dict[str, Any] = {
        "schema_version": sv.tolist(),
        "episode_id": episode_id.astype(np.int64).tolist(),
        "step_in_episode": step_in_ep.astype(np.int64).tolist(),
        "kl_aggregate_t": kl.astype(np.float64).tolist(),
        "recon_loss_t": recon.astype(np.float64).tolist(),
    }
    if sp_distribution is not None:
        sp = rng.normal(sp_distribution[0], sp_distribution[1], size=n_total)
        sp = np.clip(sp, 0.0, None)  # cosine distance is non-negative
        # Mask first step of each episode (matches plan §6 row 16 /
        # synthesis §3 v2 default).
        masked = step_in_ep == 0
        # Sentinel-zero on masked rows (the writer's convention; the
        # reader filters them via the masked flag).
        sp = np.where(masked, 0.0, sp)
        columns["self_prediction_error_t"] = sp.astype(np.float64).tolist()
        columns["self_prediction_error_masked_t"] = masked.tolist()

    table = pa.table(columns)
    agent_step_dir = run_dir / "telemetry" / "agent_step"
    agent_step_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(agent_step_dir / "shard-000000.parquet"))  # type: ignore[no-untyped-call]

    # Checkpoint with one tensor we can compare across runs.
    ckpt_dir = run_dir / "checkpoints" / "ckpt-000001"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights = {
        "world_model.encoder.proj.weight": torch.randn(
            256, 1024, generator=torch.Generator().manual_seed(seed)
        ),
    }
    if schema_version == "0.1.0":
        # Probe-1-shaped actor input layer (no scalar column).
        weights["actor.net.0.weight"] = torch.randn(
            200, 216, generator=torch.Generator().manual_seed(seed + 1)
        )
    else:
        # Probe-1.5-shaped actor input layer (one extra column).
        weights["actor.net.0.weight"] = torch.randn(
            200, 217, generator=torch.Generator().manual_seed(seed + 1)
        )
    save_file(weights, str(ckpt_dir / "weights.safetensors"))


def test_load_run_stats_round_trip(tmp_path: Path) -> None:
    """Synthetic run is loaded; per-step arrays, schema version,
    checkpoint id, and weight moments all populate."""
    module = _load_script_module()

    run_dir = tmp_path / "synthetic_probe1_5"
    _write_synthetic_run(
        run_dir,
        schema_version="0.2.0",
        sp_distribution=(0.05, 0.02),
        seed=7,
    )
    stats = module.load_run_stats("synthetic_probe1_5", run_dir)

    # Episodes 5-25 inclusive = 20 episodes × 200 steps = 4000 rows.
    assert stats.n_filtered_steps == 4000
    assert stats.kl_aggregate.shape == (4000,)
    assert stats.recon_loss.shape == (4000,)
    # sp_err: same 4000 rows, minus the masked first step per episode (20).
    assert stats.self_prediction_error.shape == (3980,)
    assert stats.has_self_prediction is True
    assert stats.checkpoint_id == "ckpt-000001"
    assert stats.schema_version == "0.2.0"

    # Weight moments populate for both tensors written by the helper.
    assert "actor.net.0.weight" in stats.weight_moments
    assert "world_model.encoder.proj.weight" in stats.weight_moments


def test_load_run_stats_probe1_has_no_self_prediction(tmp_path: Path) -> None:
    """Probe-1-shaped synthetic run lacks the sp_err column entirely;
    ``has_self_prediction`` resolves False and the array is empty."""
    module = _load_script_module()
    run_dir = tmp_path / "synthetic_probe1"
    _write_synthetic_run(run_dir, schema_version="0.1.0", sp_distribution=None, seed=0)
    stats = module.load_run_stats("synthetic_probe1", run_dir)
    assert stats.has_self_prediction is False
    assert stats.self_prediction_error.size == 0


def test_build_summary_text_four_way(tmp_path: Path) -> None:
    """Synthetic four-directory setup with one Probe-1-shaped run and
    three Probe-1.5-shaped runs. ``build_summary_text`` should:

    - List all four runs in the overview
    - Produce 6 pairwise KS-blocks (4 choose 2)
    - Emit one-line per-pairing summaries naming all six
    - Mark sp_err as ``n/a`` for any pairing involving the Probe-1-shaped
      run
    - Include the highlighted weight-moment block for ``actor.net.0.weight``
      and ``world_model.encoder.proj.weight``
    """
    module = _load_script_module()

    # Probe 1: tighter KL (no auxiliary), no sp_err.
    p1 = tmp_path / "synthetic_probe1"
    _write_synthetic_run(
        p1, schema_version="0.1.0", kl_distribution=(15.0, 1.0), sp_distribution=None, seed=1
    )

    # Phase 7: sp_err near-zero (well-trained head); KL distinguishable from P1.
    p7 = tmp_path / "synthetic_phase7"
    _write_synthetic_run(
        p7,
        schema_version="0.2.0",
        kl_distribution=(13.0, 1.0),
        sp_distribution=(0.005, 0.001),
        seed=2,
    )

    # Phase 7.5: KL similar to Phase 7 (column-init shouldn't move substrate much).
    p75 = tmp_path / "synthetic_phase7_5"
    _write_synthetic_run(
        p75,
        schema_version="0.2.0",
        kl_distribution=(13.5, 1.0),
        sp_distribution=(0.014, 0.002),
        seed=3,
    )

    # Frozen-target: KL closer to Probe 1 (frozen target → less self-specific shaping).
    ft = tmp_path / "synthetic_frozen_target"
    _write_synthetic_run(
        ft,
        schema_version="0.2.0",
        kl_distribution=(14.5, 1.0),
        sp_distribution=(0.01, 0.005),
        seed=4,
    )

    stats = [
        module.load_run_stats("Probe 1", p1),
        module.load_run_stats("Phase 7", p7),
        module.load_run_stats("Phase 7.5", p75),
        module.load_run_stats("frozen-target", ft),
    ]

    text = module.build_summary_text(stats)

    # Header + run-overview lines.
    assert "Probe 1.5 four-way comparison" in text
    assert "Probe 1" in text and "Phase 7" in text and "Phase 7.5" in text
    assert "frozen-target" in text

    # 6 pairings with one-line summaries.
    expected_pairs = [
        ("Probe 1", "Phase 7"),
        ("Probe 1", "Phase 7.5"),
        ("Probe 1", "frozen-target"),
        ("Phase 7", "Phase 7.5"),
        ("Phase 7", "frozen-target"),
        ("Phase 7.5", "frozen-target"),
    ]
    for a, b in expected_pairs:
        assert f"{a} vs {b}:" in text, f"missing one-line summary for {a} vs {b}"

    # sp_err should be n/a for any pairing involving Probe 1.
    for a, b in expected_pairs:
        if "Probe 1" in (a, b):
            line_marker = f"{a} vs {b}:"
            i = text.find(line_marker)
            assert i >= 0
            line = text[i : text.find("\n", i)]
            assert "sp_err n/a" in line, f"{a} vs {b} should mark sp_err as n/a"

    # Highlighted weight-moment block exists.
    assert "world_model.encoder.proj.weight" in text
    assert "actor.net.0.weight" in text


def test_dry_run_returns_zero_without_writing(tmp_path: Path, capsys: Any) -> None:
    """``--dry-run`` returns 0; prints the resolved run dirs; does not
    create any output directory under ``runs/``.

    The test sets ``--output-dir`` to a path under ``tmp_path`` (the
    dry-run still respects the resolution but does not create the
    directory)."""
    module = _load_script_module()
    out_dir = tmp_path / "would_be_output"
    rc = module.main(
        [
            "--probe1",
            str(tmp_path / "no_such_probe1"),
            "--phase7",
            str(tmp_path / "no_such_phase7"),
            "--phase7-5",
            str(tmp_path / "no_such_phase7_5"),
            "--frozen-target",
            str(tmp_path / "no_such_frozen_target"),
            "--output-dir",
            str(out_dir),
            "--dry-run",
        ]
    )
    # Discovery used the CLI overrides — none of the directories exist,
    # so main returns 1 (under-2 found).
    assert rc == 1
    assert not out_dir.exists()


def test_main_writes_summary_to_output_dir(tmp_path: Path) -> None:
    """End-to-end ``main`` against synthetic dirs: produces summary.txt
    under the explicit ``--output-dir``."""
    module = _load_script_module()

    p1 = tmp_path / "synthetic_probe1"
    p7 = tmp_path / "synthetic_phase7"
    _write_synthetic_run(
        p1, schema_version="0.1.0", kl_distribution=(15.0, 1.0), sp_distribution=None, seed=10
    )
    _write_synthetic_run(
        p7,
        schema_version="0.2.0",
        kl_distribution=(13.0, 1.0),
        sp_distribution=(0.01, 0.001),
        seed=11,
    )

    out_dir = tmp_path / "comparison_output"
    # All four overrides must be set so auto-discovery doesn't reach
    # into the real ``runs/`` directory and load whatever happens to be
    # there (e.g. an in-progress run whose parquet shards aren't yet
    # written). The two unused overrides point to non-existent paths
    # so they're filtered out by ``not p.is_dir()`` in main.
    rc = module.main(
        [
            "--probe1",
            str(p1),
            "--phase7",
            str(p7),
            "--phase7-5",
            str(tmp_path / "no_such_phase7_5"),
            "--frozen-target",
            str(tmp_path / "no_such_frozen_target"),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    summary_path = out_dir / "summary.txt"
    assert summary_path.is_file()
    content = summary_path.read_text(encoding="utf-8")
    assert "Probe 1 vs Phase 7:" in content


@pytest.fixture(autouse=True)
def _restore_sys_modules() -> Any:
    yield
    sys.modules.pop(_MODULE_NAME, None)
