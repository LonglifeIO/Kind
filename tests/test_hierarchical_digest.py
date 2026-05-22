"""Probe 2 Phase 2 — gate test 9: hierarchical digest with three cohorts.

Implementation plan §2.3 + §4 gate test 9. Exercises
:func:`~kind.observer.digest.build_hierarchical_digest` against the
Probe 1.5 Phase 7.5 main run (the primary read-only fixture) and the
Probe 1 baseline (cross-probe schema=0.1.0 fixture for graceful
degradation), plus synthetic conditioning records that exercise the
behavior-side cohort and the drill-down accessor's
:meth:`fetch_conditioning` surface.

Coverage:

- Builds against ``runs/probe1_5_phase7_5-20260507-101800/`` with all
  three cohorts populated when conditioning records are supplied.
- Probe 1 graceful degradation: head-internal + behavior-side cohorts
  collapse to the no-self-prediction-telemetry line; substrate-side is
  full.
- ``conditioning_dir=None`` graceful degradation: behavior-side cohort
  collapses to the no-conditioning-data line; head-internal +
  substrate-side stay full.
- Drill-down accessors: ``fetch_self_prediction`` returns the requested
  window with masked-flag visibility; ``fetch_conditioning`` returns
  the requested (regime, perturbation) pair.
- Sham builder-perturbation events: surfaced with a ``[SHAM]`` prefix
  and ``sham_label``; real perturbations have no prefix.
- Flagged anomalies include head-internal sp_err outliers when
  present (the Phase 7 mirror's ep 23 cluster surfaces against the
  Phase 7.5 fixture's actual sp_err distribution; the test asserts at
  least one head-internal outlier exists rather than pinning a
  specific episode that the run's empirical noise might shift).
- ``n_episodes`` parameter respected: requesting 10 produces 10
  mini-digests.
- ``flagged_only=True`` filters mini-digests to flagged-anomaly
  episodes only.

The conditioning records for the behavior-side cohort and drill-down
fetches are synthesised here — the conditioning analysis module's
actual implementation is Phase 6 territory, so these tests construct
the JSONL records the digest expects to read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kind.observer.conditioning import (
    CONDITIONING_RESULT_SCHEMA_VERSION,
    ConditioningResult,
    RegimeBucket,
    RegimeStats,
)
from kind.observer.digest import (
    DrillDownAccessor,
    FlaggedAnomaly,
    HierarchicalDigest,
    build_flat_digest,
    build_hierarchical_digest,
)
from kind.observer.schemas import WorldEvent

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE_1_5_TELEMETRY_DIR = (
    REPO_ROOT / "runs" / "probe1_5_phase7_5-20260507-101800" / "telemetry"
)
PROBE_1_TELEMETRY_DIR = REPO_ROOT / "runs" / "probe1-20260503-123926" / "telemetry"


_PROBE_1_DEGRADED = (
    "(no self-prediction telemetry — records are Probe 1, schema_version 0.1.0)"
)
_NO_COND_DEGRADED = (
    "(no conditioning data — conditioning_dir not supplied)"
)


# ---- fixtures -------------------------------------------------------------


def _write_synthetic_conditioning(
    conditioning_dir: Path, *, run_id: str, checkpoint_id: str = "ckpt-000001"
) -> ConditioningResult:
    """Write one ConditioningResult JSONL line into ``conditioning_dir``.

    The record carries one bucket per regime ∈ {perturbation_window,
    high_disagreement, high_kl, steady_state} × perturbation ∈
    {gaussian, zero, uniform} = 12 buckets. The KL_p90 values include
    one deliberately large outlier in (steady_state, gaussian) so the
    ``conditioning_kl_p90_outlier`` anomaly surfaces.
    """
    conditioning_dir.mkdir(parents=True, exist_ok=True)
    buckets: list[RegimeBucket] = []
    base_kl_p90 = 1e-7
    for regime in ("perturbation_window", "high_disagreement", "high_kl", "steady_state"):
        for perturbation in ("gaussian", "zero", "uniform"):
            kl_p90 = base_kl_p90
            if regime == "steady_state" and perturbation == "gaussian":
                # Outlier well above the 3σ band of the other 11 buckets.
                kl_p90 = 1e-3
            buckets.append(
                RegimeBucket(
                    regime=regime,
                    perturbation=perturbation,
                    stats=RegimeStats(
                        n_states=20,
                        kl_mean=kl_p90 * 0.5,
                        kl_std=kl_p90 * 0.1,
                        kl_p50=kl_p90 * 0.45,
                        kl_p90=kl_p90,
                    ),
                )
            )
    result = ConditioningResult(
        schema_version=CONDITIONING_RESULT_SCHEMA_VERSION,
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        timestamp_ms=1_700_000_000_000,
        n_states_sampled=240,
        perturbation_distributions=["gaussian", "zero", "uniform"],
        regimes=["perturbation_window", "high_disagreement", "high_kl", "steady_state"],
        empirical_scalar_mean=0.01,
        empirical_scalar_sigma=0.005,
        empirical_scalar_range=(0.0, 0.05),
        per_regime_per_perturbation=buckets,
        masked_steps_excluded=25,
    )
    (conditioning_dir / "conditioning.jsonl").write_text(
        result.model_dump_json() + "\n"
    )
    return result


def _has_real_phase7_5_fixture() -> bool:
    return (
        PROBE_1_5_TELEMETRY_DIR.is_dir()
        and (PROBE_1_5_TELEMETRY_DIR / "agent_step").is_dir()
    )


def _has_probe1_fixture() -> bool:
    return (
        PROBE_1_TELEMETRY_DIR.is_dir()
        and (PROBE_1_TELEMETRY_DIR / "agent_step").is_dir()
    )


# ---- gate-test layer ------------------------------------------------------


@pytest.mark.skipif(
    not _has_real_phase7_5_fixture(),
    reason="Probe 1.5 Phase 7.5 fixture not present",
)
def test_hierarchical_digest_builds_against_phase_7_5_with_three_cohorts(
    tmp_path: Path,
) -> None:
    """The primary fixture builds; all three cohorts populate."""
    cond_dir = tmp_path / "conditioning"
    _write_synthetic_conditioning(
        cond_dir, run_id="probe1_5_phase7_5-20260507-101800"
    )
    hd = build_hierarchical_digest(
        PROBE_1_5_TELEMETRY_DIR,
        n_episodes=25,
        conditioning_dir=cond_dir,
    )
    assert isinstance(hd, HierarchicalDigest)
    assert hd.run_summary.startswith("# Telemetry Run Summary")
    assert "schema_version: 0.2.0" in hd.run_summary
    assert len(hd.episode_mini_digests) == 25
    # Every block has all three cohort labels.
    for ep_id, block in hd.episode_mini_digests.items():
        assert "[substrate-side]" in block, f"episode {ep_id} missing substrate-side"
        assert "[head-internal]" in block, f"episode {ep_id} missing head-internal"
        assert "[behavior-side]" in block, f"episode {ep_id} missing behavior-side"
        # Cohort content (not the degraded line).
        assert _PROBE_1_DEGRADED not in block
        assert _NO_COND_DEGRADED not in block
    # World-event timeline includes the 26 env_resets the fixture has.
    assert "# World Event Timeline" in hd.world_event_timeline
    assert hd.world_event_timeline.count("env_reset") >= 26


@pytest.mark.skipif(
    not _has_probe1_fixture(),
    reason="Probe 1 baseline fixture not present",
)
def test_hierarchical_digest_probe_1_records_degrade_head_and_behavior(
    tmp_path: Path,
) -> None:
    """Probe 1 records carry no self-prediction; head-internal and
    behavior-side cohorts replace their content with the
    no-self-prediction-telemetry line."""
    hd = build_hierarchical_digest(
        PROBE_1_TELEMETRY_DIR,
        n_episodes=5,
        # conditioning_dir omitted; behavior-side would also degrade
        # via the no-conditioning-data path, but the Probe 1 path
        # supersedes (no scalar to condition on at all).
    )
    for ep_id, block in hd.episode_mini_digests.items():
        # Substrate-side cohort still full (Probe 1 has KL, ensemble).
        assert "[substrate-side]" in block
        assert "kl_aggregate_t" in block
        assert "ensemble_disagreement" in block
        # Head-internal + behavior-side carry the Probe 1 graceful line.
        head_section = block.split("[head-internal]")[1].split(
            "[behavior-side]"
        )[0]
        behavior_section = block.split("[behavior-side]")[1]
        assert _PROBE_1_DEGRADED in head_section, (
            f"episode {ep_id}: head-internal missing degraded line"
        )
        assert _PROBE_1_DEGRADED in behavior_section, (
            f"episode {ep_id}: behavior-side missing degraded line"
        )


@pytest.mark.skipif(
    not _has_real_phase7_5_fixture(),
    reason="Probe 1.5 Phase 7.5 fixture not present",
)
def test_hierarchical_digest_no_conditioning_dir_degrades_behavior_only() -> None:
    """When ``conditioning_dir=None`` on a 0.2.0 run, only the
    behavior-side cohort degrades; substrate-side and head-internal
    populate."""
    hd = build_hierarchical_digest(
        PROBE_1_5_TELEMETRY_DIR,
        n_episodes=3,
        conditioning_dir=None,
    )
    assert hd.episode_mini_digests
    for ep_id, block in hd.episode_mini_digests.items():
        # Substrate-side full.
        substrate_section = block.split("[substrate-side]")[1].split(
            "[head-internal]"
        )[0]
        assert "kl_aggregate_t" in substrate_section
        assert _PROBE_1_DEGRADED not in substrate_section
        # Head-internal full (this is a 0.2.0 run).
        head_section = block.split("[head-internal]")[1].split(
            "[behavior-side]"
        )[0]
        assert "self_prediction_error_t" in head_section
        assert _PROBE_1_DEGRADED not in head_section
        # Behavior-side degraded (no conditioning_dir).
        behavior_section = block.split("[behavior-side]")[1]
        assert _NO_COND_DEGRADED in behavior_section


@pytest.mark.skipif(
    not _has_real_phase7_5_fixture(),
    reason="Probe 1.5 Phase 7.5 fixture not present",
)
def test_drill_down_fetch_self_prediction_window_with_masked_flag(
    tmp_path: Path,
) -> None:
    """``fetch_self_prediction`` returns per-step
    ``self_prediction_error_t`` for the requested window with the
    masked flag visible; nothing outside the window."""
    hd = build_hierarchical_digest(
        PROBE_1_5_TELEMETRY_DIR,
        n_episodes=25,
    )
    # Episode 24, steps 0..3 — step 0 is masked under the Phase 0
    # convention; steps 1..3 are non-masked empirical readings.
    text = hd.drill_down.fetch_self_prediction(
        episode_id=24, step_range=(0, 3)
    )
    assert "episode 24" in text
    assert "steps [0, 3]" in text
    # Four steps in the window (0..3 inclusive).
    assert text.count("self_prediction_error_t=") == 4
    # First step's masked flag is True (Phase 0 convention).
    first_line = [
        line for line in text.splitlines() if "step=0:" in line
    ][0]
    assert "self_prediction_error_masked_t=True" in first_line
    # Subsequent steps are non-masked.
    for step in (1, 2, 3):
        line = [
            line for line in text.splitlines() if f"step={step}:" in line
        ][0]
        assert "self_prediction_error_masked_t=False" in line
    # Nothing outside the window.
    assert "step=4:" not in text


def test_drill_down_fetch_conditioning_returns_per_state_kl_distribution(
    tmp_path: Path,
) -> None:
    """``fetch_conditioning`` returns the per-(regime, perturbation)
    bucket from the loaded conditioning records."""
    # Build a fixture-shaped synthetic telemetry directory so the
    # digest can be constructed without the real run.
    telemetry_dir = _build_minimal_synthetic_telemetry(tmp_path / "telem")
    cond_dir = tmp_path / "conditioning"
    _write_synthetic_conditioning(cond_dir, run_id="synthetic")

    hd = build_hierarchical_digest(
        telemetry_dir,
        n_episodes=2,
        conditioning_dir=cond_dir,
    )
    text = hd.drill_down.fetch_conditioning(
        regime="high_disagreement", perturbation="gaussian"
    )
    assert "high_disagreement" in text
    assert "gaussian" in text
    assert "n_states=20" in text
    # KL_p90 should match what _write_synthetic_conditioning wrote.
    assert "1.000e-07" in text or "1e-07" in text


def test_drill_down_fetch_conditioning_no_dir_returns_degraded_line(
    tmp_path: Path,
) -> None:
    telemetry_dir = _build_minimal_synthetic_telemetry(tmp_path / "telem")
    hd = build_hierarchical_digest(
        telemetry_dir,
        n_episodes=2,
        conditioning_dir=None,
    )
    text = hd.drill_down.fetch_conditioning(
        regime="high_disagreement", perturbation="gaussian"
    )
    assert text == _NO_COND_DEGRADED


def test_world_event_timeline_distinguishes_sham_from_real(
    tmp_path: Path,
) -> None:
    """Sham builder_perturbation events appear with ``[SHAM]`` prefix
    and ``sham_label``; real builder_perturbations have no prefix."""
    telemetry_dir = _build_minimal_synthetic_telemetry(
        tmp_path / "telem",
        extra_world_events=[
            {
                "schema_version": "0.1.0",
                "run_id": "synthetic",
                "checkpoint_id": None,
                "t_event": 50,
                "event_type": "builder_perturbation",
                "source": "builder",
                "payload": {"intended_cell": [3, 5]},
                "wallclock_ms": 1_000_050,
            },
            {
                "schema_version": "0.1.0",
                "run_id": "synthetic",
                "checkpoint_id": None,
                "t_event": 75,
                "event_type": "builder_perturbation",
                "source": "builder",
                "payload": {
                    "is_sham": True,
                    "sham_label": "add_resource",
                },
                "wallclock_ms": 1_000_075,
            },
        ],
    )
    hd = build_hierarchical_digest(telemetry_dir, n_episodes=2)
    timeline = hd.world_event_timeline
    real_lines = [
        line
        for line in timeline.splitlines()
        if "builder_perturbation" in line and "[SHAM]" not in line
    ]
    sham_lines = [
        line
        for line in timeline.splitlines()
        if "builder_perturbation" in line and "[SHAM]" in line
    ]
    assert len(real_lines) == 1, f"got {real_lines}"
    assert len(sham_lines) == 1, f"got {sham_lines}"
    assert "sham_label='add_resource'" in sham_lines[0]


@pytest.mark.skipif(
    not _has_real_phase7_5_fixture(),
    reason="Probe 1.5 Phase 7.5 fixture not present",
)
def test_flagged_anomalies_surface_head_internal_sp_err_outliers() -> None:
    """The Phase 7.5 fixture has non-trivial sp_err variance; at least
    one head-internal self_prediction_error_t outlier should surface."""
    hd = build_hierarchical_digest(
        PROBE_1_5_TELEMETRY_DIR,
        n_episodes=25,
    )
    head_outliers = [
        a
        for a in hd.flagged_anomalies
        if a.kind == "self_prediction_outlier"
        and a.reading_surface == "head_internal"
    ]
    assert head_outliers, (
        "no head-internal self_prediction_outlier flagged on Phase 7.5 "
        "fixture; the digest's flagged-anomaly contract requires sp_err "
        "outliers to surface as head-internal anomalies"
    )
    # Each head-internal outlier prefixes its description with the
    # cohort label so downstream readers can attribute without
    # re-deriving from the kind string.
    for a in head_outliers:
        assert a.description.startswith("[head-internal]")


def test_flagged_anomalies_surface_behavior_side_kl_p90_outliers(
    tmp_path: Path,
) -> None:
    """A conditioning record with a deliberately-large KL_p90 in one
    bucket trips the behavior-side ``conditioning_kl_p90_outlier``
    anomaly."""
    telemetry_dir = _build_minimal_synthetic_telemetry(tmp_path / "telem")
    cond_dir = tmp_path / "conditioning"
    _write_synthetic_conditioning(cond_dir, run_id="synthetic")
    hd = build_hierarchical_digest(
        telemetry_dir,
        n_episodes=2,
        conditioning_dir=cond_dir,
    )
    behavior_anomalies = [
        a
        for a in hd.flagged_anomalies
        if a.reading_surface == "behavior_side"
        and a.kind == "conditioning_kl_p90_outlier"
    ]
    assert behavior_anomalies, (
        "no behavior-side KL_p90 outlier flagged; the synthetic "
        "(steady_state, gaussian) bucket at KL_p90=1e-3 is well above "
        "the 3σ band of the other 11 buckets at KL_p90=1e-7 and should "
        "trip the anomaly"
    )
    # The anomaly carries the regime + perturbation in its description
    # so downstream readers do not have to re-derive the bucket.
    found = behavior_anomalies[0]
    assert "steady_state" in found.description
    assert "gaussian" in found.description


@pytest.mark.skipif(
    not _has_real_phase7_5_fixture(),
    reason="Probe 1.5 Phase 7.5 fixture not present",
)
def test_n_episodes_parameter_respected() -> None:
    """Requesting n_episodes=10 produces 10 mini-digests, not 25."""
    hd = build_hierarchical_digest(
        PROBE_1_5_TELEMETRY_DIR, n_episodes=10
    )
    assert len(hd.episode_mini_digests) == 10
    # The selected episodes are the most-recent 10 by episode_id.
    selected = sorted(hd.episode_mini_digests.keys())
    assert selected == list(range(15, 25))


@pytest.mark.skipif(
    not _has_real_phase7_5_fixture(),
    reason="Probe 1.5 Phase 7.5 fixture not present",
)
def test_flagged_only_filters_to_anomaly_episodes() -> None:
    """``flagged_only=True`` keeps only mini-digests for episodes that
    surface at least one flagged anomaly."""
    hd = build_hierarchical_digest(
        PROBE_1_5_TELEMETRY_DIR,
        n_episodes=25,
        flagged_only=True,
    )
    flagged_eps = {
        a.episode_id for a in hd.flagged_anomalies if a.episode_id is not None
    }
    assert flagged_eps, "Phase 7.5 fixture should have at least one flagged episode"
    assert set(hd.episode_mini_digests.keys()) <= flagged_eps
    # And every flagged episode that was selected appears in mini_digests.
    selected_episode_ids = set(range(0, 25))
    expected = flagged_eps & selected_episode_ids
    assert set(hd.episode_mini_digests.keys()) == expected


@pytest.mark.skipif(
    not _has_real_phase7_5_fixture(),
    reason="Probe 1.5 Phase 7.5 fixture not present",
)
def test_flagged_only_false_keeps_all_selected_episodes() -> None:
    hd = build_hierarchical_digest(
        PROBE_1_5_TELEMETRY_DIR,
        n_episodes=25,
        flagged_only=False,
    )
    assert len(hd.episode_mini_digests) == 25


def test_build_flat_digest_alias_preserved(tmp_path: Path) -> None:
    """``build_flat_digest`` is a backward-readable alias for the
    Probe-1-shape flat digest. The alias points to the same callable,
    so callers that consumed ``build_digest(rows)`` continue to work
    under either name."""
    telemetry_dir = _build_minimal_synthetic_telemetry(tmp_path / "telem")
    # Read the rows directly the way build_digest expects.
    import pyarrow.parquet as pq

    shards = sorted((telemetry_dir / "agent_step").glob("shard-*.parquet"))
    rows: list[dict[str, Any]] = []
    for shard in shards:
        rows.extend(pq.read_table(str(shard)).to_pylist())
    assert rows
    text = build_flat_digest(rows)
    assert "# Telemetry Digest (agent_step records)" in text


# ---- helpers --------------------------------------------------------------


def _build_minimal_synthetic_telemetry(
    telemetry_dir: Path,
    *,
    extra_world_events: list[dict[str, Any]] | None = None,
) -> Path:
    """Build a tiny 0.2.0 telemetry directory exercising both cohorts.

    Two episodes of 8 steps each. The first step of each episode is
    masked. The world_event.jsonl carries env_reset events; callers may
    supply additional events via ``extra_world_events``.
    """
    from kind.observer.schemas import AgentStep
    from kind.observer.sinks import ParquetSink

    telemetry_dir.mkdir(parents=True, exist_ok=True)
    h_dim = 8
    z_dim = 4
    n_episodes = 2
    n_steps = 8

    sink = ParquetSink(
        telemetry_dir / "agent_step", AgentStep, rows_per_shard=10_000
    )
    t = 0
    for ep in range(n_episodes):
        for s in range(n_steps):
            masked = s == 0
            scalar = 0.0 if masked else 0.05 + 0.01 * s
            record = AgentStep(
                schema_version="0.2.0",
                run_id="synthetic",
                checkpoint_id=None,
                t=t,
                episode_id=ep,
                step_in_episode=s,
                wallclock_ms=1_000_000 + t,
                h_t=[0.1 * (i + 1) + 0.001 * t for i in range(h_dim)],
                q_params_t=([0.0] * z_dim, [0.0] * z_dim),
                p_params_t=([0.0] * z_dim, [0.0] * z_dim),
                z_t=[0.5] * z_dim,
                kl_per_dim_t=[0.05 + 0.01 * (i + s) for i in range(z_dim)],
                kl_aggregate_t=0.5 + 0.001 * t,
                recon_loss_t=1.0,
                action_t=t % 5,
                action_logprob_t=-1.6,
                policy_entropy_t=1.5,
                obs_hash_t=f"h-{t}",
                intrinsic_signal_t=0.05 + 0.001 * s,
                encoder_embedding_t=[0.0] * h_dim,
                self_prediction_t=[0.05 * (i + 1) for i in range(h_dim)],
                self_prediction_error_t=scalar,
                self_prediction_error_masked_t=masked,
            )
            sink.write(record)
            t += 1
    sink.close()

    world_events: list[dict[str, Any]] = []
    for ep in range(n_episodes):
        world_events.append(
            {
                "schema_version": "0.1.0",
                "run_id": "synthetic",
                "checkpoint_id": None,
                "t_event": ep * n_steps,
                "event_type": "env_reset",
                "source": "environment",
                "payload": {"episode_id": ep, "start_cell": [3, 3]},
                "wallclock_ms": 1_000_000 + ep * n_steps,
            }
        )
    if extra_world_events:
        world_events.extend(extra_world_events)
    # Sort by t_event to mimic real harness ordering.
    world_events.sort(key=lambda e: int(e["t_event"]))
    we_path = telemetry_dir / "world_event.jsonl"
    with we_path.open("w", encoding="utf-8") as fh:
        for ev in world_events:
            # Validate via the WorldEvent model so we round-trip through
            # the schema rather than emitting drifting JSON shapes.
            wm = WorldEvent.model_validate(ev)
            fh.write(wm.model_dump_json() + "\n")
    return telemetry_dir
