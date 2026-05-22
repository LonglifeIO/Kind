"""Phase 8 signal-computation module — the specific statistics behind the
seven Phase 7 ``SignalMapping`` records.

Phase 7 committed each :class:`~kind.mirror.registry.SignalMapping` to a
*class* of statistic via its ``description`` ("partial autocorrelation /
conditional MI / partial correlation"; "a return-to-distribution lag on
the recurrent state"; "a recovery-shape classification on the entropy
trajectory"; "a between-regime contrast in policy-shape with an
observation-only control"). Phase 8 commits the *specific* statistic per
signal, the lag, the threshold, the cluster count, the partialling
procedure — and journals each choice.

The seven signals across the three v2 criteria:

- ``latent_self_reference_t`` (reflexive_attention, substrate-internal):
  partial autocorrelation of ``h_t`` at lag ``k=5``, controlling for
  ``z_t`` and ``encoder_embedding_t`` at the same step. Shuffled-time
  control on ``h_t`` within episode is reported alongside as a comparison
  value in :attr:`StatisticResult.notes`.
- ``dream_self_reference_t`` (reflexive_attention, dream): autocorrelation
  of ``sequence_h`` at lag ``k=5`` over each imagined rollout; no
  observation to partial out. Shuffled-time control on ``sequence_h``
  itself.
- ``recovery_lag_steps`` (equanimity, substrate-internal): number of
  steps from each :class:`~kind.mirror.perturbation_align.PerturbationEvent`
  until ``h_t``'s Mahalanobis distance from the pre-perturbation
  distribution falls below threshold ``K`` (the 95th percentile of the
  pre-window Mahalanobis distances) and *stays* below for at least three
  consecutive steps. Pre-window: 50 steps before each perturbation. The
  value is a ``list[float]`` of per-perturbation lags; a perturbation
  that does not recover within ``W`` steps is recorded as ``float(W + 1)``
  (a sentinel value above the window).
- ``policy_entropy_t`` (equanimity, behavior): classify the entropy
  trajectory in the W-step post-perturbation window as one of
  ``{"dip_and_recover", "collapse", "stays_elevated", "no_response"}``.
  Returned as a ``dict[str, float]`` histogram across the perturbations
  (each key's value is the count of perturbations falling in that class).
- ``posterior_kl_t`` (equanimity, substrate-internal): classify the
  ``kl_aggregate_t`` trajectory in the W-step post-perturbation window
  as one of ``{"spike_and_decay", "ratchet", "no_response",
  "oscillation"}``. Same dict-histogram shape.
- ``latent_regime_indicator_t`` (second_order_volition,
  substrate-internal): k-means clustering on ``h_t`` with ``k=4``,
  features standardized per-episode. Returns the regime label sequence
  as a ``list[float]`` (label indices cast to float for the
  :attr:`StatisticResult.value` union).
- ``policy_modulation_t`` (second_order_volition, behavior): between-regime
  contrast in action-distribution shape (entropy, top-2 mass, mean
  log-prob), reported alongside an observation-only baseline. Returned
  as a ``dict[str, float]`` with two keys: ``"contrast_magnitude"`` and
  ``"observation_only_baseline"``.

**The Phase 7 → Phase 8 binding discipline.** Each computation function
implements its committed signal mapping's *class* of statistic. If the
natural implementation of a statistic *narrows* what the criterion
admits beyond the committed class, that's a Phase 7 oversight to
journal and amend (Phase 7 amendment, not a Phase 8 patch to silently
absorb). The committed estimator name lives on the
:attr:`StatisticResult.estimator` field and is asserted in
``tests/test_statistics.py`` — a future change to the statistic without
a journal entry trips a test.

**The membrane discipline.** This module reads telemetry only. It does
not write anywhere, does not construct an actor / world model / runner,
does not invoke anything against Io's input space. It is part of the
mirror's one-way data plane. The structural read-only invariant from
Phases 6 and 7 extends here.

Out of scope here: the perturbation-log reader (Part 2,
:mod:`kind.mirror.perturbation_align`); the prompt-fragment composer
(Part 3, :mod:`kind.mirror.prompt_builder`); the LLM caller (Part 4,
:mod:`kind.mirror.llm_caller`); the pass driver (Part 5,
:mod:`kind.mirror.orchestrator`). All Phase 8.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from kind.mirror.registry import SignalMapping, TelemetrySurface

__all__ = [
    "ESTIMATOR_PARTIAL_AUTOCORR_LAG5",
    "ESTIMATOR_AUTOCORR_LAG5",
    "ESTIMATOR_MAHALANOBIS_RECOVERY",
    "ESTIMATOR_ENTROPY_TRAJECTORY",
    "ESTIMATOR_KL_TRAJECTORY",
    "ESTIMATOR_KMEANS_K4",
    "ESTIMATOR_BETWEEN_REGIME_CONTRAST",
    "RECOVERY_CLASS_LABELS",
    "ENTROPY_CLASS_LABELS",
    "KL_CLASS_LABELS",
    "StatisticResult",
    "StatisticConfig",
    "TelemetryBatch",
    "compute_statistic",
    "compute_latent_self_reference_t",
    "compute_dream_self_reference_t",
    "compute_recovery_lag_steps",
    "compute_policy_entropy_t",
    "compute_posterior_kl_t",
    "compute_latent_regime_indicator_t",
    "compute_policy_modulation_t",
]


# ---------------------------------------------------------------------------
# Committed estimator names. Each function's :attr:`StatisticResult.estimator`
# field carries one of these strings. The test suite asserts the binding so
# that a future change to a statistic without a journal entry trips a test.
# ---------------------------------------------------------------------------

ESTIMATOR_PARTIAL_AUTOCORR_LAG5: Final[str] = (
    "partial_autocorr_lag5_controlling_for_z_and_embedding"
)
ESTIMATOR_AUTOCORR_LAG5: Final[str] = "autocorr_lag5_on_sequence_h"
ESTIMATOR_MAHALANOBIS_RECOVERY: Final[str] = (
    "mahalanobis_recovery_lag_p95_3step_streak"
)
ESTIMATOR_ENTROPY_TRAJECTORY: Final[str] = "entropy_trajectory_classifier_4class"
ESTIMATOR_KL_TRAJECTORY: Final[str] = "kl_trajectory_classifier_4class"
ESTIMATOR_KMEANS_K4: Final[str] = "kmeans_k4_per_episode_standardized"
ESTIMATOR_BETWEEN_REGIME_CONTRAST: Final[str] = (
    "between_regime_contrast_entropy_top2_meanlogprob_with_obs_only_baseline"
)


# ---------------------------------------------------------------------------
# Classification labels, frozen as module-level tuples. A computation
# function's dict-histogram result carries every label as a key (with a 0.0
# count if no perturbation falls in that class), so the consumer can rely
# on the full label set being present in every result.
# ---------------------------------------------------------------------------

RECOVERY_CLASS_LABELS: Final[tuple[str, ...]] = (
    "recovered",
    "did_not_recover",
)
ENTROPY_CLASS_LABELS: Final[tuple[str, ...]] = (
    "dip_and_recover",
    "collapse",
    "stays_elevated",
    "no_response",
)
KL_CLASS_LABELS: Final[tuple[str, ...]] = (
    "spike_and_decay",
    "ratchet",
    "no_response",
    "oscillation",
)


# ---------------------------------------------------------------------------
# Result model.
# ---------------------------------------------------------------------------


class StatisticResult(BaseModel):
    """One computed signal's result.

    Frozen, ``extra="forbid"``. The ``value`` field is a union — recovery-shape
    classifications produce a ``dict[str, float]`` (the histogram of
    per-perturbation labels); scalar partial-correlations produce a
    ``float``; trajectories (the per-step regime label sequence; the
    per-perturbation recovery lag list) produce a ``list[float]``.

    ``estimator`` names the specific Phase 8 commitment (one of the
    ``ESTIMATOR_*`` module constants); tests in
    ``tests/test_statistics.py`` assert the binding for each function.

    ``notes`` is free text the prompt-builder includes verbatim in the
    LLM-facing fragment — confidence intervals, control comparisons, the
    shuffled-time-control value, the n-perturbations-aligned count, the
    pre-window size, the actual lag used. Anything a future reader of the
    pass result needs in order to understand what the number means.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    signal_name: str
    value: float | list[float] | dict[str, float]
    estimator: str
    n_samples: int
    notes: str

    @field_validator("signal_name")
    @classmethod
    def _validate_signal_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "StatisticResult.signal_name must be non-empty: it names "
                "which SignalMapping this result was computed for."
            )
        return value

    @field_validator("estimator")
    @classmethod
    def _validate_estimator(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "StatisticResult.estimator must be non-empty: it names the "
                "specific Phase 8 commitment for the statistic."
            )
        return value

    @field_validator("n_samples")
    @classmethod
    def _validate_n_samples(cls, value: int) -> int:
        if value < 0:
            raise ValueError(
                f"StatisticResult.n_samples must be non-negative; got {value}."
            )
        return value


# ---------------------------------------------------------------------------
# Per-round statistic configuration.
# ---------------------------------------------------------------------------


class StatisticConfig(BaseModel):
    """Per-round configuration of statistic choices.

    Frozen. Default values are the Phase 8 commitments; per-round overrides
    are allowed via the orchestrator's ``PassConfig.statistic_config``. A
    future round that changes a default journals the change and bumps its
    own pass record's notes field.

    Fields:

    - ``autocorr_lag``: the lag ``k`` for both ``latent_self_reference_t``
      and ``dream_self_reference_t``. Default ``5``.
    - ``kmeans_k``: the cluster count for ``latent_regime_indicator_t``.
      Default ``4``.
    - ``kmeans_iters``: Lloyd-algorithm iteration budget. Default ``20``.
    - ``kmeans_seed``: deterministic seed for the k-means init. Default
      ``0``.
    - ``recovery_pre_window``: pre-perturbation step count used to estimate
      the Mahalanobis baseline. Default ``50``.
    - ``recovery_window_W``: post-perturbation step budget within which
      recovery must complete. Default ``50``.
    - ``recovery_streak_required``: number of consecutive
      below-threshold steps required to declare recovery. Default ``3``.
    - ``recovery_threshold_percentile``: percentile of pre-window
      Mahalanobis distances used as the recovery threshold ``K``. Default
      ``95.0``.
    - ``trajectory_window_W``: post-perturbation step budget for the
      entropy and KL trajectory classifiers. Default ``50``.
    - ``trajectory_pre_window``: pre-perturbation step count for the
      classifiers' baseline. Default ``50``.
    - ``bootstrap_seed``: deterministic seed for the shuffled-time controls.
      Default ``0``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    autocorr_lag: int = 5
    kmeans_k: int = 4
    kmeans_iters: int = 20
    kmeans_seed: int = 0
    recovery_pre_window: int = 50
    recovery_window_W: int = 50
    recovery_streak_required: int = 3
    recovery_threshold_percentile: float = 95.0
    trajectory_window_W: int = 50
    trajectory_pre_window: int = 50
    bootstrap_seed: int = 0

    @field_validator("autocorr_lag", "kmeans_k", "kmeans_iters")
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"value must be positive; got {value}.")
        return value

    @field_validator(
        "recovery_pre_window",
        "recovery_window_W",
        "trajectory_window_W",
        "trajectory_pre_window",
    )
    @classmethod
    def _validate_nonneg_window(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"window size must be non-negative; got {value}.")
        return value

    @field_validator("recovery_streak_required")
    @classmethod
    def _validate_streak(cls, value: int) -> int:
        if value < 1:
            raise ValueError(
                f"recovery_streak_required must be >= 1; got {value}."
            )
        return value

    @field_validator("recovery_threshold_percentile")
    @classmethod
    def _validate_percentile(cls, value: float) -> float:
        if not (0.0 < value < 100.0):
            raise ValueError(
                f"recovery_threshold_percentile must be in (0, 100); got {value}."
            )
        return value


# ---------------------------------------------------------------------------
# Telemetry batch — the unit of input to a computation function.
#
# A frozen dataclass holding the loaded telemetry rows for one pass. Rows are
# plain mappings (read from JSONL / Parquet by the orchestrator); the
# computation functions index by field name per the SignalMapping's
# ``field_path``. Fields are tuples (not lists) so the batch is hashable-ish
# and to communicate the read-only intent.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TelemetryBatch:
    """The loaded telemetry for one Phase 8 adversarial pass.

    Carries the four relevant streams as tuples of plain mappings. The
    orchestrator (Part 5) populates this from the runner's emitted
    Parquet shards and JSONL files; tests construct synthetic batches
    directly.

    Fields:

    - ``agent_step_rows``: every :class:`~kind.observer.schemas.AgentStep`
      row in the pass's telemetry window, ordered by ``t``.
    - ``dream_rollout_rows``: every
      :class:`~kind.observer.schemas.DreamRollout` row in the window.
    - ``replay_meta_rows``: every
      :class:`~kind.observer.schemas.ReplayMeta` row in the window. The v2
      criteria do not currently read from ``replay_meta``; the slot is
      carried for forward-compatibility.
    - ``perturbation_step_indices``: the ``t`` values in ``agent_step_rows``
      that align to ``builder_perturbation`` events, computed once by
      :mod:`kind.mirror.perturbation_align` and packaged here so the
      equanimity functions don't repeat the alignment per call.
    - ``perturbation_is_sham``: a parallel tuple of booleans indicating
      whether each aligned perturbation was a sham null-event (per
      ``world_event.payload["is_sham"]`` per
      ``kind.observer.schemas.WorldEvent``). Used by the orchestrator's
      sham-perturbation calibration check.
    """

    agent_step_rows: tuple[Mapping[str, Any], ...]
    dream_rollout_rows: tuple[Mapping[str, Any], ...]
    replay_meta_rows: tuple[Mapping[str, Any], ...]
    perturbation_step_indices: tuple[int, ...]
    perturbation_is_sham: tuple[bool, ...]


# ---------------------------------------------------------------------------
# Numerical helpers.
# ---------------------------------------------------------------------------


def _stack_field(
    rows: Sequence[Mapping[str, Any]], field: str
) -> np.ndarray:
    """Stack a per-row vector / scalar field into a 2-D float array.

    Returns shape ``(n_rows, dim)`` for vector fields; ``(n_rows, 1)``
    for scalar fields. Empty input returns ``(0, 0)``. Missing values
    raise ``KeyError`` with the field name.
    """
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)
    first = rows[0][field]
    if isinstance(first, (list, tuple)):
        dim = len(first)
        arr = np.zeros((len(rows), dim), dtype=np.float64)
        for i, row in enumerate(rows):
            arr[i] = np.asarray(row[field], dtype=np.float64)
        return arr
    arr = np.zeros((len(rows), 1), dtype=np.float64)
    for i, row in enumerate(rows):
        arr[i, 0] = float(row[field])
    return arr


def _pearson_at_lag(x: np.ndarray, lag: int) -> float:
    """Lag-``lag`` Pearson autocorrelation of a 1-D series.

    Returns 0.0 if ``len(x) <= lag`` or if either window has zero variance.
    """
    n = len(x)
    if n <= lag or lag < 1:
        return 0.0
    a = x[:-lag]
    b = x[lag:]
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std == 0.0 or b_std == 0.0:
        return 0.0
    a_centered = a - float(np.mean(a))
    b_centered = b - float(np.mean(b))
    cov = float(np.mean(a_centered * b_centered))
    return cov / (a_std * b_std)


def _residualize(
    target: np.ndarray, regressors: np.ndarray
) -> np.ndarray:
    """Return ``target`` minus its OLS-projection onto ``regressors``.

    ``target`` and ``regressors`` are 2-D ``(n_rows, dim)``; an intercept
    column is prepended to ``regressors`` internally. If the regressor
    design is degenerate (or empty), the unmodified target is returned.
    """
    n = target.shape[0]
    if n == 0 or regressors.shape[1] == 0:
        return target
    design = np.concatenate([np.ones((n, 1)), regressors], axis=1)
    try:
        # lstsq returns the OLS coefficients in residual-minimizing form.
        coef, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return target
    residual: np.ndarray = target - design @ coef
    return residual


def _kmeans_lloyd(
    x: np.ndarray, k: int, iters: int, seed: int
) -> np.ndarray:
    """Minimal Lloyd's-algorithm k-means returning per-row cluster labels.

    Standardizes ``x`` column-wise per-call (subtract column mean, divide
    by column std + 1e-12). Initializes with the first ``k`` rows shuffled
    by the seeded RNG. Returns shape ``(n_rows,)`` of int labels.

    Numerically minimalist: no minibatch, no multiple restarts, no
    cluster-empty re-init. This is fine because the input scale is
    bounded (we always standardize) and the cluster count is small
    (k=4 default). A future caller that needs more sophistication runs
    sklearn separately; the module-internal version is bounded for
    reproducibility and zero-extra-dependency.
    """
    n, d = x.shape
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    if n <= k:
        return np.arange(n, dtype=np.int64)
    means = np.mean(x, axis=0, keepdims=True)
    stds = np.std(x, axis=0, keepdims=True) + 1e-12
    x_std = (x - means) / stds

    rng = np.random.default_rng(seed)
    initial_indices = rng.permutation(n)[:k]
    centroids = x_std[initial_indices].copy()
    labels = np.zeros((n,), dtype=np.int64)

    for _ in range(iters):
        # Assign step.
        dists = np.linalg.norm(
            x_std[:, None, :] - centroids[None, :, :], axis=2
        )
        new_labels = np.argmin(dists, axis=1).astype(np.int64)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        # Update step.
        for j in range(k):
            members = x_std[labels == j]
            if members.shape[0] > 0:
                centroids[j] = np.mean(members, axis=0)
    return labels


def _shuffled_time_indices(
    n: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(n)


def _safe_entropy_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total == 0.0:
        return 0.0
    p = counts / total
    p_nonzero = p[p > 0.0]
    return float(-np.sum(p_nonzero * np.log(p_nonzero)))


def _episode_slices(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[int, int]]:
    """Return ``[(start, end_exclusive), ...]`` per ``episode_id`` run.

    Treats ``rows`` as already sorted by ``t``; an episode boundary is
    detected as an ``episode_id`` change. Empty input returns an empty
    list. ``AgentStep`` rows always carry ``episode_id``; ``DreamRollout``
    rows do not (each rollout is its own unit) and this helper is not
    used on the dream path.
    """
    if not rows:
        return []
    slices: list[tuple[int, int]] = []
    start = 0
    current = int(rows[0]["episode_id"])
    for i, row in enumerate(rows[1:], start=1):
        eid = int(row["episode_id"])
        if eid != current:
            slices.append((start, i))
            start = i
            current = eid
    slices.append((start, len(rows)))
    return slices


# ---------------------------------------------------------------------------
# Computation functions — one per Phase 7 SignalMapping name.
#
# Each function reads ``batch`` per the ``mapping.field_path`` and surface,
# computes the Phase-8-committed statistic per ``config``, and returns a
# :class:`StatisticResult`. The estimator string names the specific Phase 8
# choice.
# ---------------------------------------------------------------------------


def compute_latent_self_reference_t(
    batch: TelemetryBatch,
    mapping: SignalMapping,
    config: StatisticConfig,
) -> StatisticResult:
    """Partial autocorrelation of ``h_t`` at lag ``k``, controlling for
    same-step ``z_t`` and ``encoder_embedding_t``.

    Per-component Pearson autocorrelation at lag ``k`` of the residual
    series (``h_t`` minus its OLS projection onto ``[z_t,
    encoder_embedding_t]``), averaged across components. Computed
    per-episode and then averaged across episodes (so the lag-k
    correlation doesn't bridge an episode boundary). The shuffled-time
    control permutes the time index of ``h_t`` *within episode* and
    recomputes the same statistic; the control value lands in
    :attr:`StatisticResult.notes`.
    """
    if mapping.telemetry_surface is not TelemetrySurface.AGENT_STEP_INTERNAL:
        raise ValueError(
            f"latent_self_reference_t expects telemetry_surface="
            f"AGENT_STEP_INTERNAL; got {mapping.telemetry_surface}."
        )
    rows = batch.agent_step_rows
    lag = config.autocorr_lag
    if not rows:
        return StatisticResult(
            signal_name=mapping.name,
            value=0.0,
            estimator=ESTIMATOR_PARTIAL_AUTOCORR_LAG5,
            n_samples=0,
            notes="empty agent_step batch; statistic undefined",
        )
    h = _stack_field(rows, "h_t")
    z = _stack_field(rows, "z_t")
    e = _stack_field(rows, "encoder_embedding_t")
    regressors = np.concatenate([z, e], axis=1)

    per_episode_values: list[float] = []
    per_episode_shuffled: list[float] = []
    rng_seed = config.bootstrap_seed
    for start, end in _episode_slices(rows):
        if end - start <= lag:
            continue
        h_ep = h[start:end]
        r_ep = regressors[start:end]
        residual = _residualize(h_ep, r_ep)
        per_dim = [
            _pearson_at_lag(residual[:, j], lag)
            for j in range(residual.shape[1])
        ]
        per_episode_values.append(float(np.mean(per_dim)))

        shuffled_idx = _shuffled_time_indices(end - start, rng_seed)
        rng_seed += 1
        shuffled_residual = _residualize(h_ep[shuffled_idx], r_ep)
        per_dim_shuffled = [
            _pearson_at_lag(shuffled_residual[:, j], lag)
            for j in range(shuffled_residual.shape[1])
        ]
        per_episode_shuffled.append(float(np.mean(per_dim_shuffled)))

    if not per_episode_values:
        return StatisticResult(
            signal_name=mapping.name,
            value=0.0,
            estimator=ESTIMATOR_PARTIAL_AUTOCORR_LAG5,
            n_samples=0,
            notes=(
                f"no episode contained > {lag} steps; "
                f"partial autocorrelation at lag {lag} undefined"
            ),
        )
    value = float(np.mean(per_episode_values))
    shuffled = float(np.mean(per_episode_shuffled))
    return StatisticResult(
        signal_name=mapping.name,
        value=value,
        estimator=ESTIMATOR_PARTIAL_AUTOCORR_LAG5,
        n_samples=len(rows),
        notes=(
            f"lag={lag}; n_episodes={len(per_episode_values)}; "
            f"controlling for z_t and encoder_embedding_t at the same step; "
            f"shuffled-time control (within-episode permutation): "
            f"{shuffled:.6f} (signal exceeds control by "
            f"{value - shuffled:.6f}); per-episode partial autocorrelations "
            f"averaged across episodes"
        ),
    )


def compute_dream_self_reference_t(
    batch: TelemetryBatch,
    mapping: SignalMapping,
    config: StatisticConfig,
) -> StatisticResult:
    """Autocorrelation of ``sequence_h`` at lag ``k`` over each dream
    rollout, averaged across rollouts. Shuffled-time control on
    ``sequence_h`` itself."""
    if mapping.telemetry_surface is not TelemetrySurface.DREAM_ROLLOUT:
        raise ValueError(
            f"dream_self_reference_t expects telemetry_surface="
            f"DREAM_ROLLOUT; got {mapping.telemetry_surface}."
        )
    rows = batch.dream_rollout_rows
    lag = config.autocorr_lag
    if not rows:
        return StatisticResult(
            signal_name=mapping.name,
            value=0.0,
            estimator=ESTIMATOR_AUTOCORR_LAG5,
            n_samples=0,
            notes="empty dream_rollout batch; statistic undefined",
        )

    per_rollout_values: list[float] = []
    per_rollout_shuffled: list[float] = []
    rng_seed = config.bootstrap_seed
    for row in rows:
        seq = np.asarray(row["sequence_h"], dtype=np.float64)
        if seq.ndim != 2 or seq.shape[0] <= lag:
            continue
        per_dim = [_pearson_at_lag(seq[:, j], lag) for j in range(seq.shape[1])]
        per_rollout_values.append(float(np.mean(per_dim)))

        shuffled_idx = _shuffled_time_indices(seq.shape[0], rng_seed)
        rng_seed += 1
        shuffled_seq = seq[shuffled_idx]
        per_dim_shuffled = [
            _pearson_at_lag(shuffled_seq[:, j], lag)
            for j in range(shuffled_seq.shape[1])
        ]
        per_rollout_shuffled.append(float(np.mean(per_dim_shuffled)))

    if not per_rollout_values:
        return StatisticResult(
            signal_name=mapping.name,
            value=0.0,
            estimator=ESTIMATOR_AUTOCORR_LAG5,
            n_samples=0,
            notes=(
                f"no dream rollout had sequence_h length > {lag}; "
                f"autocorrelation at lag {lag} undefined"
            ),
        )
    value = float(np.mean(per_rollout_values))
    shuffled = float(np.mean(per_rollout_shuffled))
    return StatisticResult(
        signal_name=mapping.name,
        value=value,
        estimator=ESTIMATOR_AUTOCORR_LAG5,
        n_samples=len(rows),
        notes=(
            f"lag={lag}; n_rollouts={len(per_rollout_values)}; "
            f"no observation to partial out (dream rollouts evolve under "
            f"the prior alone); shuffled-time control on sequence_h: "
            f"{shuffled:.6f} (signal exceeds control by "
            f"{value - shuffled:.6f})"
        ),
    )


def _mahalanobis_distances(
    target: np.ndarray, baseline: np.ndarray
) -> np.ndarray:
    """Squared Mahalanobis distances from rows of ``target`` to the
    centroid of ``baseline``, using ``baseline``'s sample covariance.

    Adds a small ridge to the covariance for numerical stability.
    Returns shape ``(target.shape[0],)``.
    """
    if baseline.shape[0] == 0 or target.shape[0] == 0:
        return np.zeros((target.shape[0],), dtype=np.float64)
    centroid = np.mean(baseline, axis=0)
    centered = baseline - centroid
    cov = (centered.T @ centered) / max(baseline.shape[0] - 1, 1)
    cov = cov + 1e-6 * np.eye(cov.shape[0])
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.zeros((target.shape[0],), dtype=np.float64)
    diffs = target - centroid
    dists: np.ndarray = np.einsum("ni,ij,nj->n", diffs, inv, diffs)
    return dists


def compute_recovery_lag_steps(
    batch: TelemetryBatch,
    mapping: SignalMapping,
    config: StatisticConfig,
) -> StatisticResult:
    """Per-perturbation recovery lag on ``h_t`` via Mahalanobis distance.

    For each aligned non-sham perturbation in
    ``batch.perturbation_step_indices``: build the pre-window of
    ``recovery_pre_window`` steps before the perturbation (within the
    same episode), compute the Mahalanobis-distance threshold ``K`` at
    the ``recovery_threshold_percentile``-th percentile of pre-window
    distances, then walk forward over the ``recovery_window_W``-step
    post window until ``recovery_streak_required`` consecutive distances
    fall at or below ``K``. The lag is the index of the first step in
    that streak (zero-indexed from the perturbation step). Perturbations
    that never recover within ``W`` are recorded as ``W + 1``.

    Sham perturbations are excluded from the lag list — the orchestrator
    routes them to the sham-perturbation calibration check instead.
    """
    if mapping.telemetry_surface is not TelemetrySurface.AGENT_STEP_INTERNAL:
        raise ValueError(
            f"recovery_lag_steps expects telemetry_surface="
            f"AGENT_STEP_INTERNAL; got {mapping.telemetry_surface}."
        )
    rows = batch.agent_step_rows
    if not rows:
        return StatisticResult(
            signal_name=mapping.name,
            value=[],
            estimator=ESTIMATOR_MAHALANOBIS_RECOVERY,
            n_samples=0,
            notes="empty agent_step batch; statistic undefined",
        )
    h = _stack_field(rows, "h_t")
    t_index = {int(row["t"]): i for i, row in enumerate(rows)}
    eps = [int(row["episode_id"]) for row in rows]

    pre_window = config.recovery_pre_window
    post_W = config.recovery_window_W
    streak = config.recovery_streak_required
    percentile = config.recovery_threshold_percentile

    lags: list[float] = []
    n_aligned = 0
    n_recovered = 0
    n_pre_window_short = 0
    for t_pert, is_sham in zip(
        batch.perturbation_step_indices, batch.perturbation_is_sham
    ):
        if is_sham:
            continue
        idx = t_index.get(t_pert)
        if idx is None:
            continue
        n_aligned += 1
        ep = eps[idx]
        # Pre-window: previous up-to-``pre_window`` steps in the same episode.
        pre_start = idx
        while pre_start > 0 and eps[pre_start - 1] == ep:
            pre_start -= 1
            if idx - pre_start >= pre_window:
                break
        pre_h = h[pre_start:idx]
        if pre_h.shape[0] < 2:
            n_pre_window_short += 1
            lags.append(float(post_W + 1))
            continue
        # Post-window: up to ``post_W`` steps in the same episode after idx.
        post_end = idx + 1
        while post_end < len(rows) and eps[post_end] == ep:
            post_end += 1
            if post_end - (idx + 1) >= post_W:
                break
        post_h = h[idx + 1 : post_end]
        if post_h.shape[0] == 0:
            lags.append(float(post_W + 1))
            continue

        pre_dists = _mahalanobis_distances(pre_h, pre_h)
        threshold = float(np.percentile(pre_dists, percentile))
        post_dists = _mahalanobis_distances(post_h, pre_h)

        # Walk forward to find a streak of ``streak`` consecutive distances
        # at or below ``threshold``.
        recovered_at: int | None = None
        run = 0
        for k, d in enumerate(post_dists):
            if d <= threshold:
                run += 1
                if run >= streak:
                    recovered_at = k - streak + 1
                    break
            else:
                run = 0
        if recovered_at is None:
            lags.append(float(post_W + 1))
        else:
            lags.append(float(recovered_at))
            n_recovered += 1

    if not lags:
        return StatisticResult(
            signal_name=mapping.name,
            value=[],
            estimator=ESTIMATOR_MAHALANOBIS_RECOVERY,
            n_samples=0,
            notes=(
                "no non-sham perturbations aligned to the agent_step "
                "timeline; recovery-lag undefined"
            ),
        )
    return StatisticResult(
        signal_name=mapping.name,
        value=lags,
        estimator=ESTIMATOR_MAHALANOBIS_RECOVERY,
        n_samples=n_aligned,
        notes=(
            f"per-perturbation Mahalanobis recovery lags; "
            f"pre_window={pre_window}; post_W={post_W}; "
            f"streak_required={streak}; threshold_percentile={percentile:.1f}; "
            f"recovered_within_W={n_recovered}/{n_aligned}; "
            f"pre_window_too_short={n_pre_window_short}; "
            f"a lag of {post_W + 1} is the non-recovery sentinel"
        ),
    )


def _classify_entropy_trajectory(
    pre: np.ndarray, post: np.ndarray
) -> str:
    """Classify a (pre, post)-perturbation entropy trajectory.

    Heuristic four-class classifier. The boundaries are committed at
    Phase 8 and journaled. ``pre`` is the pre-window entropy trajectory;
    ``post`` is the post-window entropy trajectory; both 1-D float arrays.
    """
    if pre.size == 0 or post.size == 0:
        return "no_response"
    pre_mean = float(np.mean(pre))
    pre_std = float(np.std(pre)) + 1e-12
    post_min = float(np.min(post))
    final_k = max(1, post.size // 4)
    post_final_mean = float(np.mean(post[-final_k:]))

    has_dip = post_min < pre_mean - 2.0 * pre_std
    recovered = abs(post_final_mean - pre_mean) <= 1.0 * pre_std
    sustained_low = post_final_mean < pre_mean - 2.0 * pre_std
    stays_high = float(np.min(post)) > pre_mean + 0.5 * pre_std

    if has_dip and recovered:
        return "dip_and_recover"
    if sustained_low:
        return "collapse"
    if stays_high:
        return "stays_elevated"
    return "no_response"


def compute_policy_entropy_t(
    batch: TelemetryBatch,
    mapping: SignalMapping,
    config: StatisticConfig,
) -> StatisticResult:
    """Classify each non-sham perturbation's post-window
    ``policy_entropy_t`` trajectory into one of
    :data:`ENTROPY_CLASS_LABELS`; return a histogram of counts."""
    if mapping.telemetry_surface is not TelemetrySurface.AGENT_STEP_OBSERVABLE:
        raise ValueError(
            f"policy_entropy_t expects telemetry_surface="
            f"AGENT_STEP_OBSERVABLE; got {mapping.telemetry_surface}."
        )
    rows = batch.agent_step_rows
    if not rows:
        return StatisticResult(
            signal_name=mapping.name,
            value={label: 0.0 for label in ENTROPY_CLASS_LABELS},
            estimator=ESTIMATOR_ENTROPY_TRAJECTORY,
            n_samples=0,
            notes="empty agent_step batch; trajectory classification undefined",
        )
    entropies = _stack_field(rows, "policy_entropy_t")[:, 0]
    t_index = {int(row["t"]): i for i, row in enumerate(rows)}
    eps = [int(row["episode_id"]) for row in rows]

    pre_window = config.trajectory_pre_window
    post_W = config.trajectory_window_W

    counts = {label: 0 for label in ENTROPY_CLASS_LABELS}
    n_aligned = 0
    for t_pert, is_sham in zip(
        batch.perturbation_step_indices, batch.perturbation_is_sham
    ):
        if is_sham:
            continue
        idx = t_index.get(t_pert)
        if idx is None:
            continue
        n_aligned += 1
        ep = eps[idx]
        pre_start = idx
        while pre_start > 0 and eps[pre_start - 1] == ep:
            pre_start -= 1
            if idx - pre_start >= pre_window:
                break
        post_end = idx + 1
        while post_end < len(rows) and eps[post_end] == ep:
            post_end += 1
            if post_end - (idx + 1) >= post_W:
                break
        label = _classify_entropy_trajectory(
            entropies[pre_start:idx], entropies[idx + 1 : post_end]
        )
        counts[label] += 1

    return StatisticResult(
        signal_name=mapping.name,
        value={label: float(counts[label]) for label in ENTROPY_CLASS_LABELS},
        estimator=ESTIMATOR_ENTROPY_TRAJECTORY,
        n_samples=n_aligned,
        notes=(
            f"per-perturbation policy_entropy_t trajectory classification "
            f"into 4 classes; pre_window={pre_window}; post_W={post_W}; "
            f"thresholds at 2*pre_std (dip) and 0.5*pre_std (stays_elevated); "
            f"non-sham perturbations only"
        ),
    )


def _classify_kl_trajectory(
    pre: np.ndarray, post: np.ndarray
) -> str:
    """Classify a (pre, post)-perturbation kl_aggregate_t trajectory.

    Threshold scheme (committed at Phase 8):

    - ``has_spike``: ``max(post) > pre_mean + 3 * pre_std`` — a real
      3-sigma excursion above the pre-window noise floor. The pre-window
      95th percentile alone is too tight when the pre-window is
      well-behaved noise, since ~5% of post values cross it by chance.
    - ``decayed``: ``post_final_mean <= pre_mean + 1 * pre_std`` — final
      quartile of the post window is within one pre-window standard
      deviation of the pre-window mean.
    - ``spike_count``: number of disjoint above-3-sigma excursions
      separated by at least one below-1-sigma sample; distinguishes a
      single spike from a true oscillatory pattern.
    """
    if pre.size == 0 or post.size == 0:
        return "no_response"
    pre_mean = float(np.mean(pre))
    pre_std = float(np.std(pre)) + 1e-12
    post_max = float(np.max(post))
    final_k = max(1, post.size // 4)
    post_final_mean = float(np.mean(post[-final_k:]))

    spike_threshold = pre_mean + 3.0 * pre_std
    has_spike = post_max > spike_threshold
    decayed = post_final_mean <= pre_mean + 1.0 * pre_std

    if not has_spike:
        return "no_response"

    # Count disjoint above-spike-threshold excursions, separated by at
    # least one below-1-sigma sample (a clean return-to-quiescent).
    above = post > spike_threshold
    below_quiescent = post < (pre_mean + 1.0 * pre_std)
    spike_count = 0
    in_spike = False
    quiescent_seen = True
    for a, b in zip(above, below_quiescent):
        if a and quiescent_seen and not in_spike:
            spike_count += 1
            in_spike = True
            quiescent_seen = False
        if b and in_spike:
            in_spike = False
            quiescent_seen = True

    if has_spike and decayed:
        if spike_count >= 2:
            return "oscillation"
        return "spike_and_decay"
    # Has spike and did not decay: ratchet (monotone-non-decreasing past
    # the spike). Tolerate small dips via a sign-count check.
    diffs = np.diff(post)
    n_neg = int(np.sum(diffs < -pre_std))
    if n_neg <= max(1, post.size // 10):
        return "ratchet"
    return "oscillation"


def compute_posterior_kl_t(
    batch: TelemetryBatch,
    mapping: SignalMapping,
    config: StatisticConfig,
) -> StatisticResult:
    """Classify each non-sham perturbation's post-window
    ``kl_aggregate_t`` trajectory into one of :data:`KL_CLASS_LABELS`;
    return a histogram of counts."""
    if mapping.telemetry_surface is not TelemetrySurface.AGENT_STEP_INTERNAL:
        raise ValueError(
            f"posterior_kl_t expects telemetry_surface="
            f"AGENT_STEP_INTERNAL; got {mapping.telemetry_surface}."
        )
    rows = batch.agent_step_rows
    if not rows:
        return StatisticResult(
            signal_name=mapping.name,
            value={label: 0.0 for label in KL_CLASS_LABELS},
            estimator=ESTIMATOR_KL_TRAJECTORY,
            n_samples=0,
            notes="empty agent_step batch; trajectory classification undefined",
        )
    kls = _stack_field(rows, "kl_aggregate_t")[:, 0]
    t_index = {int(row["t"]): i for i, row in enumerate(rows)}
    eps = [int(row["episode_id"]) for row in rows]

    pre_window = config.trajectory_pre_window
    post_W = config.trajectory_window_W

    counts = {label: 0 for label in KL_CLASS_LABELS}
    n_aligned = 0
    for t_pert, is_sham in zip(
        batch.perturbation_step_indices, batch.perturbation_is_sham
    ):
        if is_sham:
            continue
        idx = t_index.get(t_pert)
        if idx is None:
            continue
        n_aligned += 1
        ep = eps[idx]
        pre_start = idx
        while pre_start > 0 and eps[pre_start - 1] == ep:
            pre_start -= 1
            if idx - pre_start >= pre_window:
                break
        post_end = idx + 1
        while post_end < len(rows) and eps[post_end] == ep:
            post_end += 1
            if post_end - (idx + 1) >= post_W:
                break
        label = _classify_kl_trajectory(
            kls[pre_start:idx], kls[idx + 1 : post_end]
        )
        counts[label] += 1

    return StatisticResult(
        signal_name=mapping.name,
        value={label: float(counts[label]) for label in KL_CLASS_LABELS},
        estimator=ESTIMATOR_KL_TRAJECTORY,
        n_samples=n_aligned,
        notes=(
            f"per-perturbation kl_aggregate_t trajectory classification "
            f"into 4 classes; pre_window={pre_window}; post_W={post_W}; "
            f"spike threshold at pre-window 95th percentile; "
            f"ratchet detection via diff sign-count; oscillation detection "
            f"via threshold crossings; non-sham perturbations only"
        ),
    )


def compute_latent_regime_indicator_t(
    batch: TelemetryBatch,
    mapping: SignalMapping,
    config: StatisticConfig,
) -> StatisticResult:
    """K-means clustering on ``h_t`` with ``k=4`` (default), features
    standardized per-episode. Returns the regime label sequence as a
    ``list[float]`` (label indices cast to float for the
    :attr:`StatisticResult.value` union)."""
    if mapping.telemetry_surface is not TelemetrySurface.AGENT_STEP_INTERNAL:
        raise ValueError(
            f"latent_regime_indicator_t expects telemetry_surface="
            f"AGENT_STEP_INTERNAL; got {mapping.telemetry_surface}."
        )
    rows = batch.agent_step_rows
    if not rows:
        return StatisticResult(
            signal_name=mapping.name,
            value=[],
            estimator=ESTIMATOR_KMEANS_K4,
            n_samples=0,
            notes="empty agent_step batch; clustering undefined",
        )
    h = _stack_field(rows, "h_t")
    labels = np.zeros((h.shape[0],), dtype=np.int64)
    for start, end in _episode_slices(rows):
        ep_labels = _kmeans_lloyd(
            h[start:end],
            k=config.kmeans_k,
            iters=config.kmeans_iters,
            seed=config.kmeans_seed,
        )
        labels[start:end] = ep_labels

    return StatisticResult(
        signal_name=mapping.name,
        value=[float(label) for label in labels.tolist()],
        estimator=ESTIMATOR_KMEANS_K4,
        n_samples=len(rows),
        notes=(
            f"k-means clustering on h_t with k={config.kmeans_k}; "
            f"features standardized per-episode; Lloyd's algorithm with "
            f"{config.kmeans_iters} iter budget, seed={config.kmeans_seed}; "
            f"n_episodes={len(_episode_slices(rows))}; label indices "
            f"are episode-local — cross-episode comparison of label "
            f"identity is not meaningful"
        ),
    )


def _action_shape_summary(
    actions: np.ndarray, logprobs: np.ndarray, num_actions: int
) -> tuple[float, float, float]:
    """Return ``(entropy, top2_mass, mean_logprob)`` of a discrete-action
    histogram, treating each row of ``actions`` as one independent sample.
    """
    if actions.size == 0:
        return 0.0, 0.0, 0.0
    counts = np.bincount(actions, minlength=num_actions).astype(np.float64)
    entropy = _safe_entropy_from_counts(counts)
    sorted_counts = np.sort(counts)[::-1]
    total = float(np.sum(counts))
    top2 = float(np.sum(sorted_counts[: min(2, len(sorted_counts))]))
    top2_mass = top2 / total if total > 0.0 else 0.0
    mean_logprob = float(np.mean(logprobs)) if logprobs.size > 0 else 0.0
    return entropy, top2_mass, mean_logprob


def _contrast_magnitude_across_partitions(
    actions: np.ndarray,
    logprobs: np.ndarray,
    partition: np.ndarray,
    num_actions: int,
) -> float:
    """Return the maximum pairwise L2 distance, across partition values,
    of ``(entropy, top2_mass, mean_logprob)`` shape summaries."""
    unique = np.unique(partition)
    if unique.size < 2:
        return 0.0
    shapes: list[tuple[float, float, float]] = []
    for label in unique:
        mask = partition == label
        if not np.any(mask):
            continue
        shapes.append(
            _action_shape_summary(
                actions[mask], logprobs[mask], num_actions=num_actions
            )
        )
    if len(shapes) < 2:
        return 0.0
    arr = np.asarray(shapes, dtype=np.float64)
    max_dist = 0.0
    for i in range(arr.shape[0]):
        for j in range(i + 1, arr.shape[0]):
            d = float(np.linalg.norm(arr[i] - arr[j]))
            if d > max_dist:
                max_dist = d
    return max_dist


def compute_policy_modulation_t(
    batch: TelemetryBatch,
    mapping: SignalMapping,
    config: StatisticConfig,
) -> StatisticResult:
    """Between-regime contrast in action-distribution shape, with an
    observation-only baseline. Phase 8 committed shape summary: a 3-tuple
    ``(entropy, top2_mass, mean_logprob)`` over the action histogram per
    regime; the contrast magnitude is the maximum pairwise L2 distance
    across regimes. The observation-only baseline applies the same
    contrast to a partition built from ``obs_hash_t`` (hashing into
    ``kmeans_k`` buckets via Python's built-in ``hash``) so the latent
    regime's predictive power above observation can be read directly."""
    if mapping.telemetry_surface is not TelemetrySurface.AGENT_STEP_OBSERVABLE:
        raise ValueError(
            f"policy_modulation_t expects telemetry_surface="
            f"AGENT_STEP_OBSERVABLE; got {mapping.telemetry_surface}."
        )
    rows = batch.agent_step_rows
    if not rows:
        return StatisticResult(
            signal_name=mapping.name,
            value={"contrast_magnitude": 0.0, "observation_only_baseline": 0.0},
            estimator=ESTIMATOR_BETWEEN_REGIME_CONTRAST,
            n_samples=0,
            notes="empty agent_step batch; contrast undefined",
        )

    actions = np.asarray(
        [int(row["action_t"]) for row in rows], dtype=np.int64
    )
    logprobs = np.asarray(
        [float(row["action_logprob_t"]) for row in rows], dtype=np.float64
    )
    num_actions = int(np.max(actions)) + 1 if actions.size > 0 else 1

    # Latent regime partition: k-means on h_t per-episode.
    h = _stack_field(rows, "h_t")
    regime_labels = np.zeros((h.shape[0],), dtype=np.int64)
    for start, end in _episode_slices(rows):
        ep_labels = _kmeans_lloyd(
            h[start:end],
            k=config.kmeans_k,
            iters=config.kmeans_iters,
            seed=config.kmeans_seed,
        )
        regime_labels[start:end] = ep_labels

    # Observation-only partition: bucket obs_hash_t by hash mod kmeans_k.
    obs_partition = np.asarray(
        [hash(str(row["obs_hash_t"])) % config.kmeans_k for row in rows],
        dtype=np.int64,
    )

    contrast = _contrast_magnitude_across_partitions(
        actions, logprobs, regime_labels, num_actions=num_actions
    )
    baseline = _contrast_magnitude_across_partitions(
        actions, logprobs, obs_partition, num_actions=num_actions
    )

    return StatisticResult(
        signal_name=mapping.name,
        value={
            "contrast_magnitude": contrast,
            "observation_only_baseline": baseline,
        },
        estimator=ESTIMATOR_BETWEEN_REGIME_CONTRAST,
        n_samples=len(rows),
        notes=(
            f"between-regime contrast: max pairwise L2 distance of "
            f"(entropy, top2_mass, mean_logprob) shape summaries across "
            f"k-means regimes (k={config.kmeans_k}); observation-only "
            f"baseline partitions by hash(obs_hash_t) mod k; the contrast "
            f"adds predictive power above observation only if "
            f"contrast_magnitude > observation_only_baseline by a margin "
            f"the prompt-builder surfaces verbatim for the LLM to judge"
        ),
    )


# ---------------------------------------------------------------------------
# Dispatch.
# ---------------------------------------------------------------------------


_SignalComputer = Callable[
    [TelemetryBatch, SignalMapping, StatisticConfig], StatisticResult
]

_DISPATCH: Final[dict[str, _SignalComputer]] = {
    "latent_self_reference_t": compute_latent_self_reference_t,
    "dream_self_reference_t": compute_dream_self_reference_t,
    "recovery_lag_steps": compute_recovery_lag_steps,
    "policy_entropy_t": compute_policy_entropy_t,
    "posterior_kl_t": compute_posterior_kl_t,
    "latent_regime_indicator_t": compute_latent_regime_indicator_t,
    "policy_modulation_t": compute_policy_modulation_t,
}


def compute_statistic(
    batch: TelemetryBatch,
    mapping: SignalMapping,
    config: StatisticConfig,
) -> StatisticResult:
    """Dispatch to the function committed for ``mapping.name``.

    The seven Phase 8 functions cover the seven Phase 7 ``SignalMapping``
    names across the three v2 criteria. A future criterion adding a new
    signal name without a matching computation function raises
    :class:`KeyError` here, naming the missing function; this is the
    structural counter against a registry-level addition that doesn't
    follow through to a Phase 8 implementation.
    """
    if mapping.name not in _DISPATCH:
        raise KeyError(
            f"No Phase 8 computation registered for signal "
            f"{mapping.name!r}. Known signal names: "
            f"{sorted(_DISPATCH)}. Add a computation function in "
            f"kind/mirror/statistics.py and register it before passing "
            f"the SignalMapping through compute_statistic()."
        )
    return _DISPATCH[mapping.name](batch, mapping, config)


# Re-exports the dispatch-only utility for tests that need to read the
# label list directly; not part of __all__ because it's a leaf-detail.
_DISPATCH_NAMES: Final[tuple[str, ...]] = tuple(sorted(_DISPATCH))
