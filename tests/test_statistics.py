"""Phase 8 gate test — :mod:`kind.mirror.statistics`.

Tests assert the *binding between the Phase 7 SignalMapping and the
Phase 8 statistic*: each computation function returns a
:class:`~kind.mirror.statistics.StatisticResult` whose ``estimator`` field
names the committed Phase 8 choice (the ``ESTIMATOR_*`` module constants).
A future change to a statistic without a journal entry trips the
estimator-name test for that function.

For each computation function the suite covers:

- a *degenerate* input that should produce a null / zero result (with the
  expected estimator string still on the result);
- a *constructed positive* input that should produce a clear non-null
  result (positive autocorrelation for the auto-coupling signals; a
  clean per-perturbation recovery lag for the recovery signal; a tightly
  classified trajectory for the entropy / KL classifiers; cleanly
  separated clusters for the k-means signal; a between-regime contrast
  that exceeds the observation-only baseline for the modulation signal);
- a *shuffled control* whose result is at-or-near the null.

All inputs are synthetic. No telemetry-on-disk fixtures are read; tests
run independently of any prior probe's emitted records.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from kind.mirror.criteria_v2 import (
    EQUANIMITY_PERTURBATION_RECOVERY,
    REFLEXIVE_ATTENTION,
    SECOND_ORDER_VOLITION,
)
from kind.mirror.registry import (
    Criterion,
    SignalMapping,
    TelemetrySurface,
)
from kind.mirror.statistics import (
    ENTROPY_CLASS_LABELS,
    ESTIMATOR_AUTOCORR_LAG5,
    ESTIMATOR_BETWEEN_REGIME_CONTRAST,
    ESTIMATOR_ENTROPY_TRAJECTORY,
    ESTIMATOR_KL_TRAJECTORY,
    ESTIMATOR_KMEANS_K4,
    ESTIMATOR_MAHALANOBIS_RECOVERY,
    ESTIMATOR_PARTIAL_AUTOCORR_LAG5,
    KL_CLASS_LABELS,
    StatisticConfig,
    StatisticResult,
    TelemetryBatch,
    compute_dream_self_reference_t,
    compute_latent_regime_indicator_t,
    compute_latent_self_reference_t,
    compute_policy_entropy_t,
    compute_policy_modulation_t,
    compute_posterior_kl_t,
    compute_recovery_lag_steps,
    compute_statistic,
)


# ---------------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------------


def _mapping_for(criterion: Criterion, name: str) -> SignalMapping:
    for m in criterion.signal_mappings:
        if m.name == name:
            return m
    raise AssertionError(
        f"criterion {criterion.id!r} has no signal mapping named {name!r}"
    )


def _agent_step_row(
    *,
    t: int,
    episode_id: int,
    h_t: list[float],
    z_t: list[float] | None = None,
    encoder_embedding_t: list[float] | None = None,
    policy_entropy_t: float = 1.0,
    kl_aggregate_t: float = 0.5,
    action_t: int = 0,
    action_logprob_t: float = -1.0,
    obs_hash_t: str = "00",
) -> dict[str, Any]:
    return {
        "t": t,
        "episode_id": episode_id,
        "h_t": list(h_t),
        "z_t": list(z_t) if z_t is not None else [0.0, 0.0, 0.0, 0.0],
        "encoder_embedding_t": (
            list(encoder_embedding_t)
            if encoder_embedding_t is not None
            else [0.0, 0.0, 0.0, 0.0]
        ),
        "policy_entropy_t": float(policy_entropy_t),
        "kl_aggregate_t": float(kl_aggregate_t),
        "action_t": int(action_t),
        "action_logprob_t": float(action_logprob_t),
        "obs_hash_t": str(obs_hash_t),
    }


def _empty_batch() -> TelemetryBatch:
    return TelemetryBatch(
        agent_step_rows=tuple(),
        dream_rollout_rows=tuple(),
        replay_meta_rows=tuple(),
        perturbation_step_indices=tuple(),
        perturbation_is_sham=tuple(),
    )


# ---------------------------------------------------------------------------
# Result-shape and estimator-binding tests.
# ---------------------------------------------------------------------------


def test_statistic_result_rejects_extra_fields() -> None:
    """``StatisticResult`` is frozen and forbids extras."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        StatisticResult(
            signal_name="x",
            value=0.0,
            estimator="x",
            n_samples=0,
            notes="",
            extra="not allowed",  # type: ignore[call-arg]
        )


def test_statistic_result_rejects_empty_signal_name() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        StatisticResult(
            signal_name="",
            value=0.0,
            estimator="any",
            n_samples=0,
            notes="",
        )


def test_statistic_config_rejects_zero_lag() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        StatisticConfig(autocorr_lag=0)


# ---------------------------------------------------------------------------
# latent_self_reference_t
# ---------------------------------------------------------------------------


def test_latent_self_reference_empty_batch() -> None:
    mapping = _mapping_for(REFLEXIVE_ATTENTION, "latent_self_reference_t")
    result = compute_latent_self_reference_t(
        _empty_batch(), mapping, StatisticConfig()
    )
    assert result.estimator == ESTIMATOR_PARTIAL_AUTOCORR_LAG5
    assert result.value == 0.0
    assert result.n_samples == 0


def test_latent_self_reference_positive_on_ar1() -> None:
    """A within-h_t AR(1) signal produces a clear positive partial
    autocorrelation at lag 5; the shuffled-time control should sit near
    zero."""
    rng = np.random.default_rng(42)
    n = 200
    rows: list[dict[str, Any]] = []
    h = np.zeros((n, 4))
    h[0] = rng.standard_normal(4)
    for i in range(1, n):
        # Strong autoregressive self-coupling, independent of z/embedding.
        h[i] = 0.95 * h[i - 1] + 0.1 * rng.standard_normal(4)
    for i in range(n):
        rows.append(
            _agent_step_row(
                t=i,
                episode_id=0,
                h_t=h[i].tolist(),
                z_t=rng.standard_normal(4).tolist(),
                encoder_embedding_t=rng.standard_normal(4).tolist(),
            )
        )
    batch = TelemetryBatch(
        agent_step_rows=tuple(rows),
        dream_rollout_rows=tuple(),
        replay_meta_rows=tuple(),
        perturbation_step_indices=tuple(),
        perturbation_is_sham=tuple(),
    )
    mapping = _mapping_for(REFLEXIVE_ATTENTION, "latent_self_reference_t")
    result = compute_latent_self_reference_t(batch, mapping, StatisticConfig())
    assert isinstance(result.value, float)
    assert result.value > 0.4, (
        f"AR(1) signal should produce positive partial autocorrelation; "
        f"got {result.value}"
    )
    assert result.estimator == ESTIMATOR_PARTIAL_AUTOCORR_LAG5
    assert "shuffled-time control" in result.notes


def test_latent_self_reference_iid_noise_is_near_null() -> None:
    """IID Gaussian h_t with no temporal structure produces near-zero
    partial autocorrelation."""
    rng = np.random.default_rng(7)
    n = 200
    rows: list[dict[str, Any]] = [
        _agent_step_row(
            t=i,
            episode_id=0,
            h_t=rng.standard_normal(4).tolist(),
            z_t=rng.standard_normal(4).tolist(),
            encoder_embedding_t=rng.standard_normal(4).tolist(),
        )
        for i in range(n)
    ]
    batch = TelemetryBatch(
        agent_step_rows=tuple(rows),
        dream_rollout_rows=tuple(),
        replay_meta_rows=tuple(),
        perturbation_step_indices=tuple(),
        perturbation_is_sham=tuple(),
    )
    mapping = _mapping_for(REFLEXIVE_ATTENTION, "latent_self_reference_t")
    result = compute_latent_self_reference_t(batch, mapping, StatisticConfig())
    assert isinstance(result.value, float)
    assert abs(result.value) < 0.15, (
        f"IID noise should yield near-null partial autocorrelation; got "
        f"{result.value}"
    )


def test_latent_self_reference_rejects_wrong_surface() -> None:
    """Calling on a mapping with the wrong telemetry_surface raises."""
    bad = SignalMapping(
        name="latent_self_reference_t",
        description="x" * 8,
        telemetry_surface=TelemetrySurface.AGENT_STEP_OBSERVABLE,
        field_path="policy_entropy_t",
    )
    with pytest.raises(ValueError, match="AGENT_STEP_INTERNAL"):
        compute_latent_self_reference_t(_empty_batch(), bad, StatisticConfig())


# ---------------------------------------------------------------------------
# dream_self_reference_t
# ---------------------------------------------------------------------------


def test_dream_self_reference_empty() -> None:
    mapping = _mapping_for(REFLEXIVE_ATTENTION, "dream_self_reference_t")
    result = compute_dream_self_reference_t(
        _empty_batch(), mapping, StatisticConfig()
    )
    assert result.estimator == ESTIMATOR_AUTOCORR_LAG5
    assert result.value == 0.0
    assert result.n_samples == 0


def test_dream_self_reference_positive_on_autocorrelated_rollout() -> None:
    rng = np.random.default_rng(13)
    n_rollouts = 3
    seq_len = 100
    rows: list[dict[str, Any]] = []
    for _ in range(n_rollouts):
        seq = np.zeros((seq_len, 4))
        seq[0] = rng.standard_normal(4)
        for i in range(1, seq_len):
            seq[i] = 0.92 * seq[i - 1] + 0.1 * rng.standard_normal(4)
        rows.append({"sequence_h": seq.tolist()})
    batch = TelemetryBatch(
        agent_step_rows=tuple(),
        dream_rollout_rows=tuple(rows),
        replay_meta_rows=tuple(),
        perturbation_step_indices=tuple(),
        perturbation_is_sham=tuple(),
    )
    mapping = _mapping_for(REFLEXIVE_ATTENTION, "dream_self_reference_t")
    result = compute_dream_self_reference_t(batch, mapping, StatisticConfig())
    assert isinstance(result.value, float)
    assert result.value > 0.4, (
        f"autocorrelated rollout should yield positive autocorrelation; "
        f"got {result.value}"
    )
    assert result.estimator == ESTIMATOR_AUTOCORR_LAG5


def test_dream_self_reference_iid_rollout_is_null() -> None:
    rng = np.random.default_rng(99)
    rows: list[dict[str, Any]] = [
        {"sequence_h": rng.standard_normal((100, 4)).tolist()}
        for _ in range(3)
    ]
    batch = TelemetryBatch(
        agent_step_rows=tuple(),
        dream_rollout_rows=tuple(rows),
        replay_meta_rows=tuple(),
        perturbation_step_indices=tuple(),
        perturbation_is_sham=tuple(),
    )
    mapping = _mapping_for(REFLEXIVE_ATTENTION, "dream_self_reference_t")
    result = compute_dream_self_reference_t(batch, mapping, StatisticConfig())
    assert isinstance(result.value, float)
    assert abs(result.value) < 0.15


# ---------------------------------------------------------------------------
# recovery_lag_steps
# ---------------------------------------------------------------------------


def test_recovery_lag_steps_no_perturbations() -> None:
    mapping = _mapping_for(EQUANIMITY_PERTURBATION_RECOVERY, "recovery_lag_steps")
    result = compute_recovery_lag_steps(
        _empty_batch(), mapping, StatisticConfig()
    )
    assert result.value == []
    assert result.estimator == ESTIMATOR_MAHALANOBIS_RECOVERY


def test_recovery_lag_steps_clean_recovery() -> None:
    """h_t sits at zero for 60 steps, spikes for 5 steps at the
    perturbation, then returns to zero. The recovery lag should be small
    (just a handful of steps)."""
    rng = np.random.default_rng(5)
    rows: list[dict[str, Any]] = []
    h_dim = 4
    # Pre-window: 60 steps of baseline (small noise around 0).
    for i in range(60):
        rows.append(
            _agent_step_row(
                t=i,
                episode_id=0,
                h_t=(0.05 * rng.standard_normal(h_dim)).tolist(),
            )
        )
    # Perturbation at t=60.
    pert_t = 60
    rows.append(
        _agent_step_row(
            t=pert_t,
            episode_id=0,
            h_t=(0.05 * rng.standard_normal(h_dim) + 5.0).tolist(),
        )
    )
    # Post-perturbation: large for 5 steps, then back to baseline.
    for i in range(1, 6):
        rows.append(
            _agent_step_row(
                t=pert_t + i,
                episode_id=0,
                h_t=(0.05 * rng.standard_normal(h_dim) + 5.0 - i).tolist(),
            )
        )
    for i in range(6, 50):
        rows.append(
            _agent_step_row(
                t=pert_t + i,
                episode_id=0,
                h_t=(0.05 * rng.standard_normal(h_dim)).tolist(),
            )
        )
    batch = TelemetryBatch(
        agent_step_rows=tuple(rows),
        dream_rollout_rows=tuple(),
        replay_meta_rows=tuple(),
        perturbation_step_indices=(pert_t,),
        perturbation_is_sham=(False,),
    )
    mapping = _mapping_for(EQUANIMITY_PERTURBATION_RECOVERY, "recovery_lag_steps")
    result = compute_recovery_lag_steps(batch, mapping, StatisticConfig())
    assert isinstance(result.value, list)
    assert len(result.value) == 1
    # Recovery should occur within the 50-step window (well before the
    # sentinel post_W+1 = 51).
    assert result.value[0] < 20, (
        f"clean recovery should be detected quickly; got lag={result.value[0]}"
    )
    assert "recovered_within_W=1/1" in result.notes


def test_recovery_lag_steps_sham_excluded() -> None:
    """Sham perturbations are excluded from the lag list."""
    rng = np.random.default_rng(5)
    rows = [
        _agent_step_row(
            t=i, episode_id=0, h_t=rng.standard_normal(4).tolist()
        )
        for i in range(100)
    ]
    batch = TelemetryBatch(
        agent_step_rows=tuple(rows),
        dream_rollout_rows=tuple(),
        replay_meta_rows=tuple(),
        perturbation_step_indices=(50,),
        perturbation_is_sham=(True,),  # sham — must be excluded
    )
    mapping = _mapping_for(EQUANIMITY_PERTURBATION_RECOVERY, "recovery_lag_steps")
    result = compute_recovery_lag_steps(batch, mapping, StatisticConfig())
    assert result.value == []
    assert result.n_samples == 0


# ---------------------------------------------------------------------------
# policy_entropy_t
# ---------------------------------------------------------------------------


def test_policy_entropy_classification_returns_full_label_set() -> None:
    """Every result has all four ENTROPY_CLASS_LABELS as keys."""
    rows = [
        _agent_step_row(t=i, episode_id=0, h_t=[0.0, 0.0, 0.0, 0.0])
        for i in range(60)
    ]
    batch = TelemetryBatch(
        agent_step_rows=tuple(rows),
        dream_rollout_rows=tuple(),
        replay_meta_rows=tuple(),
        perturbation_step_indices=tuple(),
        perturbation_is_sham=tuple(),
    )
    mapping = _mapping_for(EQUANIMITY_PERTURBATION_RECOVERY, "policy_entropy_t")
    result = compute_policy_entropy_t(batch, mapping, StatisticConfig())
    assert isinstance(result.value, dict)
    assert set(result.value.keys()) == set(ENTROPY_CLASS_LABELS)


def test_policy_entropy_dip_and_recover_detected() -> None:
    """Pre-window stable entropy ≈ 1.0; entropy dips sharply post-perturbation
    then recovers to ≈ 1.0. The classifier should label this
    ``dip_and_recover``."""
    rng = np.random.default_rng(1)
    rows: list[dict[str, Any]] = []
    pert_t = 60
    for i in range(60):
        rows.append(
            _agent_step_row(
                t=i,
                episode_id=0,
                h_t=[0.0, 0.0, 0.0, 0.0],
                policy_entropy_t=1.0 + 0.01 * rng.standard_normal(),
            )
        )
    rows.append(
        _agent_step_row(
            t=pert_t,
            episode_id=0,
            h_t=[0.0, 0.0, 0.0, 0.0],
            policy_entropy_t=0.1,
        )
    )
    for i in range(1, 6):
        rows.append(
            _agent_step_row(
                t=pert_t + i,
                episode_id=0,
                h_t=[0.0, 0.0, 0.0, 0.0],
                policy_entropy_t=0.1 + 0.18 * i,
            )
        )
    for i in range(6, 50):
        rows.append(
            _agent_step_row(
                t=pert_t + i,
                episode_id=0,
                h_t=[0.0, 0.0, 0.0, 0.0],
                policy_entropy_t=1.0 + 0.01 * rng.standard_normal(),
            )
        )
    batch = TelemetryBatch(
        agent_step_rows=tuple(rows),
        dream_rollout_rows=tuple(),
        replay_meta_rows=tuple(),
        perturbation_step_indices=(pert_t,),
        perturbation_is_sham=(False,),
    )
    mapping = _mapping_for(EQUANIMITY_PERTURBATION_RECOVERY, "policy_entropy_t")
    result = compute_policy_entropy_t(batch, mapping, StatisticConfig())
    assert isinstance(result.value, dict)
    assert result.value["dip_and_recover"] == 1.0
    assert sum(result.value.values()) == 1.0  # only one perturbation aligned


# ---------------------------------------------------------------------------
# posterior_kl_t
# ---------------------------------------------------------------------------


def test_posterior_kl_no_response_when_kl_flat() -> None:
    """Pre and post-perturbation kl_aggregate_t both stable around 0.5;
    classifier should label ``no_response``."""
    rng = np.random.default_rng(3)
    rows: list[dict[str, Any]] = []
    pert_t = 60
    for i in range(120):
        rows.append(
            _agent_step_row(
                t=i,
                episode_id=0,
                h_t=[0.0, 0.0, 0.0, 0.0],
                kl_aggregate_t=0.5 + 0.005 * rng.standard_normal(),
            )
        )
    batch = TelemetryBatch(
        agent_step_rows=tuple(rows),
        dream_rollout_rows=tuple(),
        replay_meta_rows=tuple(),
        perturbation_step_indices=(pert_t,),
        perturbation_is_sham=(False,),
    )
    mapping = _mapping_for(EQUANIMITY_PERTURBATION_RECOVERY, "posterior_kl_t")
    result = compute_posterior_kl_t(batch, mapping, StatisticConfig())
    assert isinstance(result.value, dict)
    assert set(result.value.keys()) == set(KL_CLASS_LABELS)
    assert result.value["no_response"] == 1.0


def test_posterior_kl_spike_and_decay_detected() -> None:
    rng = np.random.default_rng(11)
    rows: list[dict[str, Any]] = []
    pert_t = 60
    for i in range(60):
        rows.append(
            _agent_step_row(
                t=i,
                episode_id=0,
                h_t=[0.0, 0.0, 0.0, 0.0],
                kl_aggregate_t=0.5 + 0.01 * rng.standard_normal(),
            )
        )
    rows.append(
        _agent_step_row(
            t=pert_t,
            episode_id=0,
            h_t=[0.0, 0.0, 0.0, 0.0],
            kl_aggregate_t=5.0,
        )
    )
    for i in range(1, 50):
        # Exponential decay back to baseline.
        rows.append(
            _agent_step_row(
                t=pert_t + i,
                episode_id=0,
                h_t=[0.0, 0.0, 0.0, 0.0],
                kl_aggregate_t=0.5 + 4.5 * float(np.exp(-i / 5.0)),
            )
        )
    batch = TelemetryBatch(
        agent_step_rows=tuple(rows),
        dream_rollout_rows=tuple(),
        replay_meta_rows=tuple(),
        perturbation_step_indices=(pert_t,),
        perturbation_is_sham=(False,),
    )
    mapping = _mapping_for(EQUANIMITY_PERTURBATION_RECOVERY, "posterior_kl_t")
    result = compute_posterior_kl_t(batch, mapping, StatisticConfig())
    assert isinstance(result.value, dict)
    assert result.value["spike_and_decay"] == 1.0


# ---------------------------------------------------------------------------
# latent_regime_indicator_t
# ---------------------------------------------------------------------------


def test_latent_regime_indicator_recovers_well_separated_clusters() -> None:
    """h_t built from four well-separated cluster centers; k-means with
    k=4 should recover four distinct labels."""
    rng = np.random.default_rng(0)
    centers = np.array(
        [
            [5.0, 5.0, 0.0, 0.0],
            [-5.0, 5.0, 0.0, 0.0],
            [-5.0, -5.0, 0.0, 0.0],
            [5.0, -5.0, 0.0, 0.0],
        ]
    )
    rows: list[dict[str, Any]] = []
    for i in range(80):
        c = centers[i % 4]
        h = (c + 0.3 * rng.standard_normal(4)).tolist()
        rows.append(_agent_step_row(t=i, episode_id=0, h_t=h))
    batch = TelemetryBatch(
        agent_step_rows=tuple(rows),
        dream_rollout_rows=tuple(),
        replay_meta_rows=tuple(),
        perturbation_step_indices=tuple(),
        perturbation_is_sham=tuple(),
    )
    mapping = _mapping_for(
        SECOND_ORDER_VOLITION, "latent_regime_indicator_t"
    )
    result = compute_latent_regime_indicator_t(batch, mapping, StatisticConfig())
    assert result.estimator == ESTIMATOR_KMEANS_K4
    assert isinstance(result.value, list)
    assert len(result.value) == 80
    unique_labels = {int(x) for x in result.value}
    assert len(unique_labels) == 4, (
        f"expected 4 distinct labels for well-separated clusters; got "
        f"{unique_labels}"
    )


def test_latent_regime_indicator_empty() -> None:
    mapping = _mapping_for(
        SECOND_ORDER_VOLITION, "latent_regime_indicator_t"
    )
    result = compute_latent_regime_indicator_t(
        _empty_batch(), mapping, StatisticConfig()
    )
    assert result.value == []
    assert result.estimator == ESTIMATOR_KMEANS_K4


# ---------------------------------------------------------------------------
# policy_modulation_t
# ---------------------------------------------------------------------------


def test_policy_modulation_contrast_exceeds_baseline_when_regime_drives_action() -> None:
    """Actions are deterministic functions of the latent regime; the
    observation-only baseline (obs_hash bucketed) should yield a much
    smaller contrast."""
    rng = np.random.default_rng(0)
    centers = np.array(
        [
            [5.0, 5.0, 0.0, 0.0],
            [-5.0, 5.0, 0.0, 0.0],
            [-5.0, -5.0, 0.0, 0.0],
            [5.0, -5.0, 0.0, 0.0],
        ]
    )
    rows: list[dict[str, Any]] = []
    for i in range(80):
        c_idx = i % 4
        c = centers[c_idx]
        h = (c + 0.3 * rng.standard_normal(4)).tolist()
        # Action concentrates per regime: regime 0 → action 0, etc.
        action = c_idx
        rows.append(
            _agent_step_row(
                t=i,
                episode_id=0,
                h_t=h,
                action_t=action,
                action_logprob_t=-0.1,
                obs_hash_t=f"obs_{i % 7}",  # uncorrelated with regime
            )
        )
    batch = TelemetryBatch(
        agent_step_rows=tuple(rows),
        dream_rollout_rows=tuple(),
        replay_meta_rows=tuple(),
        perturbation_step_indices=tuple(),
        perturbation_is_sham=tuple(),
    )
    mapping = _mapping_for(SECOND_ORDER_VOLITION, "policy_modulation_t")
    result = compute_policy_modulation_t(batch, mapping, StatisticConfig())
    assert isinstance(result.value, dict)
    assert result.estimator == ESTIMATOR_BETWEEN_REGIME_CONTRAST
    assert result.value["contrast_magnitude"] > result.value[
        "observation_only_baseline"
    ]


def test_policy_modulation_empty() -> None:
    mapping = _mapping_for(SECOND_ORDER_VOLITION, "policy_modulation_t")
    result = compute_policy_modulation_t(
        _empty_batch(), mapping, StatisticConfig()
    )
    assert isinstance(result.value, dict)
    assert result.value["contrast_magnitude"] == 0.0
    assert result.value["observation_only_baseline"] == 0.0


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_compute_statistic_dispatches_by_signal_name() -> None:
    mapping = _mapping_for(REFLEXIVE_ATTENTION, "latent_self_reference_t")
    result = compute_statistic(_empty_batch(), mapping, StatisticConfig())
    assert result.signal_name == "latent_self_reference_t"
    assert result.estimator == ESTIMATOR_PARTIAL_AUTOCORR_LAG5


def test_compute_statistic_unknown_signal_raises() -> None:
    bad = SignalMapping(
        name="not_a_real_signal",
        description="x" * 8,
        telemetry_surface=TelemetrySurface.AGENT_STEP_INTERNAL,
        field_path="h_t",
    )
    with pytest.raises(KeyError, match="not_a_real_signal"):
        compute_statistic(_empty_batch(), bad, StatisticConfig())


def test_estimator_strings_are_committed_constants() -> None:
    """Belt-and-braces: confirm the committed strings match the constants
    every function uses. A drift in either would trip both."""
    assert ESTIMATOR_PARTIAL_AUTOCORR_LAG5 == (
        "partial_autocorr_lag5_controlling_for_z_and_embedding"
    )
    assert ESTIMATOR_AUTOCORR_LAG5 == "autocorr_lag5_on_sequence_h"
    assert ESTIMATOR_MAHALANOBIS_RECOVERY == (
        "mahalanobis_recovery_lag_p95_3step_streak"
    )
    assert ESTIMATOR_ENTROPY_TRAJECTORY == "entropy_trajectory_classifier_4class"
    assert ESTIMATOR_KL_TRAJECTORY == "kl_trajectory_classifier_4class"
    assert ESTIMATOR_KMEANS_K4 == "kmeans_k4_per_episode_standardized"
    assert ESTIMATOR_BETWEEN_REGIME_CONTRAST == (
        "between_regime_contrast_entropy_top2_meanlogprob"
        "_with_obs_only_baseline"
    )
