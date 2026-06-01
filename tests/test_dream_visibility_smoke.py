"""Probe 3 Phase 3 — visibility-smoke machinery tests.

Unit tests on synthetic / fixture data for the smoke's measurement machinery
(``scripts/smoke_probe3_visibility``). The *gate run* against a real checkpoint
is a script invocation (``python -m scripts.smoke_probe3_visibility``), not a
unit test — these tests verify the pieces the gate's defensibility rests on:

- **KS-D correctness** — identical distributions → ~0; well-separated → ≥ 0.15;
  an analytically-known shifted-uniform case.
- **Non-degeneracy / cold-start detection** — degenerate per-rollout series are
  flagged; varied ones pass; the cold-start pre-flight flags a collapsed replay
  buffer and passes a varied one on a small real world model.
- **Cross-version field extraction** — the same field pulled from a ``"0.2.0"``
  and a ``"0.3.0"`` fixture yields comparable arrays; ensemble disagreement is
  present on ``"0.3.0"`` and absent (structural axis-2) on ``"0.2.0"``.
- **Regime → config mapping** — pure-prior maps to the right Phase 2 control
  knobs and is bucketed by the ``"prior_only_control"`` tag, never ``seed_kind``.
- **Threshold configurability** — the gate call flips on the threshold (plan §4).
- **End-to-end** — ``run_smoke`` produces a comparison report (plan §4 test 1).

The two plan-named Phase 3 tests (end-to-end report; configurable thresholds)
are included; the remaining tests are the machinery coverage the build prompt
requires. The count therefore exceeds the plan §4 "2" — a deliberate deviation
journaled in ``docs/workingjournal/probe3.md`` (the plan named the two
load-bearing checks; the machinery underneath them is unit-tested too).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from kind.agents.world_model import WorldModel, WorldModelConfig
from kind.observer.schemas import (
    PROBE_3_TELEMETRY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    DreamRollout,
)
from kind.training.dream_seed import SeedSelectionConfig
from kind.training.replay import SequenceReplayBuffer, Transition
from scripts.smoke_probe3_visibility import (
    KS_D_THRESHOLD_DEFAULT,
    KS_FIELDS,
    PRIOR_ONLY_CONTROL_TAG,
    cold_start_preflight,
    compare_regimes,
    decoded_obs_entropy,
    dream_config,
    extract_per_rollout,
    is_degenerate,
    is_prior_only_control,
    ks_d,
    per_rollout_field,
    pure_prior_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_CKPT = REPO_ROOT / "runs" / "probe1_5_phase7_5-20260507-101800" / "checkpoints" / "ckpt-000001"


# --------------------------------------------------------------------------
# KS-D correctness
# --------------------------------------------------------------------------


def test_ks_d_identical_is_zero() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    assert ks_d(a, a.copy()) == 0.0


def test_ks_d_well_separated_exceeds_threshold() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(loc=0.0, scale=1.0, size=2000)
    b = rng.normal(loc=10.0, scale=1.0, size=2000)
    # Disjoint supports → CDFs separate completely → D ≈ 1.0, well above 0.15.
    assert ks_d(a, b) >= 0.15
    assert ks_d(a, b) > 0.99


def test_ks_d_known_shift_on_uniforms() -> None:
    # Two large dense uniform grids: U[0,1] vs U[0.5,1.5]. The max CDF gap is
    # the overlap-region separation; analytically D → 0.5 as n → ∞.
    a = np.linspace(0.0, 1.0, 20001)
    b = np.linspace(0.5, 1.5, 20001)
    assert ks_d(a, b) == pytest.approx(0.5, abs=0.01)


def test_ks_d_rejects_empty() -> None:
    with pytest.raises(ValueError):
        ks_d(np.array([]), np.array([1.0]))


# --------------------------------------------------------------------------
# Non-degeneracy detection
# --------------------------------------------------------------------------


def test_is_degenerate_flags_constant_series() -> None:
    assert is_degenerate(np.full(40, 3.14)) is True


def test_is_degenerate_flags_tiny_variation() -> None:
    base = np.full(40, 100.0)
    base[0] += 1e-6  # coefficient of variation far below the 1e-3 floor
    assert is_degenerate(base) is True


def test_is_degenerate_passes_varied_series() -> None:
    rng = np.random.default_rng(2)
    assert is_degenerate(rng.normal(loc=5.0, scale=1.0, size=40)) is False


def test_is_degenerate_flags_singleton() -> None:
    assert is_degenerate(np.array([1.0])) is True


# --------------------------------------------------------------------------
# decoded-obs entropy
# --------------------------------------------------------------------------


def test_decoded_obs_entropy_flat_frame_is_zero() -> None:
    flat = bytes([7] * 1024)  # every pixel the same value
    assert decoded_obs_entropy(flat) == pytest.approx(0.0)


def test_decoded_obs_entropy_uniform_spread_near_eight_bits() -> None:
    # Each of the 256 values appears equally often → 8 bits of entropy.
    spread = bytes(list(range(256)) * 4)
    assert decoded_obs_entropy(spread) == pytest.approx(8.0, abs=1e-9)


# --------------------------------------------------------------------------
# Cross-version field extraction
# --------------------------------------------------------------------------

H_DIM = 4
Z_DIM = 2
HORIZON = 6


def _base_fields() -> dict[str, Any]:
    return {
        "run_id": "run-0000",
        "checkpoint_id": "ckpt-000001",
        "seed_step": 100,
        "seed_h0": [0.0] * H_DIM,
        "seed_z0": [0.0] * Z_DIM,
        "sequence_h": [[0.01 * (j + i) for i in range(H_DIM)] for j in range(HORIZON)],
        "sequence_z_prior": [
            [0.01 * (j + i) for i in range(Z_DIM)] for j in range(HORIZON)
        ],
        "sequence_action": [i % 5 for i in range(HORIZON)],
        "sequence_action_logprob": [-0.1 * i for i in range(HORIZON)],
        "sequence_prior_entropy": [0.05 * i for i in range(HORIZON)],
        # one flat frame per step; flat → entropy 0 (deterministic for the test)
        "sequence_decoded_obs": [bytes([10] * 9) for _ in range(HORIZON)],
        "cumulative_prior_entropy": 0.5,
        "mean_step_kl_successive_priors": 0.1,
        "max_step_latent_norm_change": 0.5,
        "sequence_self_prediction": None,
    }


def _v0_2_0_record() -> DreamRollout:
    """Waking-planning-shaped: no ensemble disagreement (axis-2 absence)."""
    return DreamRollout(schema_version=SCHEMA_VERSION, **_base_fields())


def _v0_3_0_record(*, disagreement: list[float], tags: list[str] | None) -> DreamRollout:
    fields = _base_fields()
    return DreamRollout(
        schema_version=PROBE_3_TELEMETRY_SCHEMA_VERSION,
        dream_session_id="s",
        seed_kind="perturbed_prior",
        seed_perturbation_magnitude=0.0,
        temperature_schedule=[1.0] * HORIZON,
        sub_mode_tags=tags,
        sampling_parameters={"horizon": HORIZON},
        gradient_policy="none",
        rng_seed=7,
        termination_reason="horizon_complete",
        sequence_ensemble_disagreement_variance=disagreement,
        checkpoint_hash="abc",
        **fields,
    )


def test_cross_version_base_fields_comparable() -> None:
    v2 = _v0_2_0_record()
    v3 = _v0_3_0_record(disagreement=[0.2] * HORIZON, tags=None)
    for fld in (
        "mean_step_kl_successive_priors",
        "cumulative_prior_entropy",
        "max_step_latent_norm_change",
        "decoded_obs_entropy_mean",
    ):
        a = per_rollout_field(v2, fld)
        b = per_rollout_field(v3, fld)
        assert a is not None and b is not None
        assert a == pytest.approx(b)


def test_cross_version_ensemble_absent_on_v0_2_0() -> None:
    assert per_rollout_field(_v0_2_0_record(), "ensemble_disagreement_mean") is None
    v3 = _v0_3_0_record(disagreement=[0.1, 0.3, 0.5, 0.1, 0.3, 0.5], tags=None)
    assert per_rollout_field(v3, "ensemble_disagreement_mean") == pytest.approx(0.3)


def test_extract_per_rollout_omits_structurally_absent_field() -> None:
    v2_scalars = extract_per_rollout([_v0_2_0_record() for _ in range(5)])
    assert "ensemble_disagreement_mean" not in v2_scalars
    # the four available base fields are present with one scalar per rollout
    assert v2_scalars["mean_step_kl_successive_priors"].shape == (5,)
    v3_scalars = extract_per_rollout(
        [_v0_3_0_record(disagreement=[0.2] * HORIZON, tags=None) for _ in range(5)]
    )
    assert "ensemble_disagreement_mean" in v3_scalars


# --------------------------------------------------------------------------
# Regime → config mapping + tag bucketing
# --------------------------------------------------------------------------


def test_dream_config_is_real_four_axis_regime() -> None:
    cfg = dream_config(horizon=30)
    assert cfg.horizon == 30
    assert cfg.temperature_mode == "scheduled"
    assert cfg.seed_strategy_for_control == "normal"
    assert cfg.action_policy == "uniform_random"
    assert cfg.record_ensemble_disagreement is True


def test_pure_prior_config_maps_to_control_knobs() -> None:
    cfg = pure_prior_config(horizon=30)
    assert cfg.seed_strategy_for_control == "prior_only"
    assert cfg.temperature_mode == "identity"


def test_prior_only_bucketed_by_tag_not_seed_kind() -> None:
    # A pure-prior control record: seed_kind is "perturbed_prior" (shared with
    # real perturbed-prior dreams) but it carries the control tag.
    control = _v0_3_0_record(
        disagreement=[0.2] * HORIZON, tags=[PRIOR_ONLY_CONTROL_TAG]
    )
    real_dream = _v0_3_0_record(
        disagreement=[0.2] * HORIZON, tags=["associative_drift_tail", "chimera"]
    )
    assert control.seed_kind == real_dream.seed_kind == "perturbed_prior"
    # Bucketing by tag separates them; bucketing by seed_kind would not.
    assert is_prior_only_control(control) is True
    assert is_prior_only_control(real_dream) is False


# --------------------------------------------------------------------------
# Threshold configurability (plan §4 Phase 3 test 2)
# --------------------------------------------------------------------------


def test_gate_threshold_is_configurable() -> None:
    rng = np.random.default_rng(3)
    # Moderately separated per-rollout series: KS-D lands between 0.15 and 0.9.
    dream = {f: rng.normal(loc=0.0, scale=1.0, size=60) for f in KS_FIELDS}
    control = {f: rng.normal(loc=1.0, scale=1.0, size=60) for f in KS_FIELDS}
    lenient = compare_regimes(dream, control, "c", threshold=0.15)
    strict = compare_regimes(dream, control, "c", threshold=0.99)
    assert lenient.n_fields_passing >= strict.n_fields_passing
    assert lenient.passed is True
    assert strict.passed is False


def test_gate_marks_field_unavailable_when_control_absent() -> None:
    rng = np.random.default_rng(4)
    dream = {f: rng.normal(size=40) for f in KS_FIELDS}
    # control missing the ensemble field (waking-planning shape)
    control = {
        f: rng.normal(loc=5.0, size=40)
        for f in KS_FIELDS
        if f != "ensemble_disagreement_mean"
    }
    cmp = compare_regimes(dream, control, "waking_planning", threshold=0.15)
    ens = next(f for f in cmp.fields if f.field == "ensemble_disagreement_mean")
    assert ens.available is False
    assert ens.ks_d is None


def test_gate_excludes_degenerate_field() -> None:
    rng = np.random.default_rng(5)
    dream = {f: rng.normal(loc=0.0, size=40) for f in KS_FIELDS}
    control = {f: rng.normal(loc=10.0, size=40) for f in KS_FIELDS}
    # Collapse one dream field to a constant: a high between-regime KS-D there
    # must not count toward the pass.
    dream["max_step_latent_norm_change"] = np.full(40, 1.0)
    cmp = compare_regimes(dream, control, "c", threshold=0.15)
    deg = next(f for f in cmp.fields if f.field == "max_step_latent_norm_change")
    assert deg.dream_degenerate is True
    assert deg.passes is False


# --------------------------------------------------------------------------
# Cold-start pre-flight on a small real world model
# --------------------------------------------------------------------------


def _tiny_world_model() -> WorldModel:
    torch.manual_seed(0)
    return WorldModel(WorldModelConfig())


def _fill_varied_buffer(n: int) -> SequenceReplayBuffer:
    """A buffer of varied real-shaped obs (distinct random frames per step)."""
    buf = SequenceReplayBuffer(capacity=n + 1, sequence_length=32)
    g = torch.Generator()
    g.manual_seed(0)
    for i in range(n):
        obs = torch.rand((1, 32, 32), generator=g)
        nxt = torch.rand((1, 32, 32), generator=g)
        buf.insert(
            Transition(
                obs=obs,
                action=i % 5,
                next_obs=nxt,
                env_step=i,
                episode_id=i // 50,
                step_in_episode=i % 50,
            )
        )
    return buf


class _ObsBlindWorldModel(WorldModel):
    """A world model whose posterior ignores observations (returns the prior).

    This is the degeneracy the cold-start obs-structure check exists to catch:
    if the warmup's posterior carries no observation information, the
    replay-seeded ``h_init`` is identical to the prior-only (GRU-bias) warmup —
    the warmup is *not* conditioning on obs, only a schema slot (the
    Probe-1.5-Phase-7 lesson). With a shared noise stream the obs-structure gap
    is then exactly 0.
    """

    def posterior(self, h: torch.Tensor, embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return self.prior(h)


def test_cold_start_passes_on_varied_buffer() -> None:
    wm = _tiny_world_model()
    buf = _fill_varied_buffer(300)
    cfg = SeedSelectionConfig(replay_min_segment_age_steps=20, replay_warmup_length=8)
    rng = torch.Generator()
    rng.manual_seed(1)
    result = cold_start_preflight(wm, buf, cfg, rng)
    assert result.passed is True
    assert result.seed_variation >= result.seed_variation_floor
    assert result.obs_structure >= result.obs_structure_floor


def test_cold_start_flags_obs_blind_warmup() -> None:
    torch.manual_seed(0)
    wm = _ObsBlindWorldModel(WorldModelConfig())
    buf = _fill_varied_buffer(300)
    cfg = SeedSelectionConfig(replay_min_segment_age_steps=20, replay_warmup_length=8)
    rng = torch.Generator()
    rng.manual_seed(1)
    result = cold_start_preflight(wm, buf, cfg, rng)
    # Posterior ignores obs → replay warmup == prior-only warmup → obs
    # structure 0 → abort with the "raise replay_warmup_length" guidance.
    assert result.passed is False
    assert result.obs_structure == pytest.approx(0.0, abs=1e-6)
    assert "replay_warmup_length" in result.message


def test_cold_start_flags_insufficient_windows() -> None:
    wm = _tiny_world_model()
    buf = _fill_varied_buffer(60)
    # min_age so high that fewer than two valid windows survive.
    cfg = SeedSelectionConfig(replay_min_segment_age_steps=59, replay_warmup_length=8)
    rng = torch.Generator()
    rng.manual_seed(1)
    result = cold_start_preflight(wm, buf, cfg, rng)
    assert result.passed is False
    assert result.n_windows < 2


# --------------------------------------------------------------------------
# End-to-end (plan §4 Phase 3 test 1) — fast, against the real checkpoint
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not (REAL_CKPT / "weights.safetensors").exists(),
    reason="Probe 1.5 checkpoint not present",
)
def test_run_smoke_produces_report_end_to_end() -> None:
    from scripts.smoke_probe3_visibility import run_smoke

    report = run_smoke(
        REAL_CKPT,
        n_rollouts=8,
        horizon=30,
        threshold=KS_D_THRESHOLD_DEFAULT,
        buffer_steps=400,
        env_seed=0,
        rng_seed=0,
        replay_min_segment_age_steps=50,
        replay_warmup_length=8,
        device=torch.device("cpu"),
    )
    assert report.cold_start.passed is True
    assert len(report.comparisons) == 2
    names = {c.control_name for c in report.comparisons}
    assert names == {"pure_prior", "waking_planning"}
    for c in report.comparisons:
        # every KS field accounted for in the table
        assert {f.field for f in c.fields} == set(KS_FIELDS)
