"""Probe 2 conditioning analysis module — schema-only stub at Phase 0.

Probe 1.5 v2's plan §2.7 introduced a ``show_self_prediction_conditioning``
helper that produced per-state action-distribution-under-perturbation
tables for behavior-side reading; Probe 2 v2's synthesis §2.4 element 7
(faithfulness check) and implementation plan §2.6 formalize that helper as
a first-class analysis module with structured ``ConditioningResult``
records the faithfulness verifier resolves cited claims against. Probe
2's behavior-side reading surface depends on this module: every
behavior-side claim cites
``cited_stream="conditioning_analysis"``; the verifier reads the cached
:class:`ConditioningResult` JSONL and resolves the cited regime /
perturbation distribution / KL statistic against the table.

**Phase 0 scope.** This module defines the schemas (:class:`RegimeStats`,
:class:`RegimeBucket`, :class:`ConditioningResult`) and the
:func:`compute_conditioning` function signature. The signature is a stub
that raises ``NotImplementedError``; the actual analysis lands in Phase
6. Phase 0's concern is that the schemas exist so the
``schemas/v0.3.0.json`` export covers the conditioning model, and so
downstream modules (the digest extension at Phase 2 with the
``conditioning_dir`` argument; the faithfulness verifier at Phase 11)
can import the model without circular dependency.

**Schema version.** ``ConditioningResult`` is a fresh stream at
``"0.1.0"`` — no v0 to bump from, no Probe 1.5 records to migrate.

**Why a list of buckets, not a tuple-keyed dict.** The implementation
plan §2.6 names ``per_regime_per_perturbation: dict[tuple[str, str],
RegimeStats]``. JSON does not support tuple keys, so a faithful
round-trip through JSONL requires either a composite-key string or a
list-of-buckets shape. The list-of-buckets shape — :class:`RegimeBucket`
records carrying ``regime``, ``perturbation``, and ``stats`` — is more
explicit and allows Phase 6 to add helper accessors (``find_bucket``,
``regimes()``, ``perturbations()``) without re-encoding keys. Lookup
helpers land in Phase 6 alongside the actual analysis logic; Phase 0
defines the data shape only.

**The masked-step exclusion contract.** ``compute_conditioning`` excludes
masked steps (records with ``self_prediction_error_masked_t == True``)
from the n_states sample; the exclusion count is recorded in
``masked_steps_excluded``. The discipline carries Probe 1.5 v2 §10 item
3: the scalar's first-step-of-episode sentinel value is not part of the
empirical scalar distribution, so it must not bias the conditioning's
empirical baselines.

Out of scope at Phase 0:
- The actual analysis logic (Phase 6).
- Multi-trajectory or multi-seed averaging within a checkpoint (deferred
  per implementation plan §2.6 "Not at Probe 2").
- Real-time conditioning analysis. The module runs post-hoc against
  committed telemetry and committed checkpoints.
- Any helper accessors beyond the data classes themselves. Phase 6 adds
  ``find_bucket``, etc., when the analysis logic lands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

__all__ = [
    "CONDITIONING_RESULT_SCHEMA_VERSION",
    "PerturbationDistribution",
    "Regime",
    "RegimeStats",
    "RegimeBucket",
    "ConditioningResult",
    "compute_conditioning",
]


CONDITIONING_RESULT_SCHEMA_VERSION: Final[str] = "0.1.0"


# Perturbation distributions sampled when computing the per-state
# action-distribution-under-perturbation tables. Carried from Probe 1.5
# v2 plan §2.7's ``show_self_prediction_conditioning`` helper's choices.
PerturbationDistribution: TypeAlias = Literal["gaussian", "zero", "uniform"]


# Regimes the analysis classifies states into. ``"perturbation_window"``
# states are within ±W steps of a ``builder_perturbation`` world_event
# (the window size W is a Phase 6 configuration). ``"high_disagreement"``
# states are in the top quartile of ``intrinsic_signal_t`` and not in
# ``perturbation_window``. ``"high_kl"`` states are in the top quartile
# of ``kl_aggregate_t`` and not in either of the above.
# ``"steady_state"`` is the residual.
Regime: TypeAlias = Literal[
    "perturbation_window",
    "high_disagreement",
    "high_kl",
    "steady_state",
]


class RegimeStats(BaseModel):
    """Per-(regime, perturbation) summary statistics on the
    action-distribution KL between unperturbed and perturbed PolicyView.

    Fields:

    - ``n_states``: number of states sampled into this bucket.
    - ``kl_mean``, ``kl_std``: mean and standard deviation of the per-state
      action-distribution KL (between unperturbed and perturbed forward
      passes through the actor at the sampled states).
    - ``kl_p50``, ``kl_p90``: 50th and 90th percentile of the same.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_states: int
    kl_mean: float
    kl_std: float
    kl_p50: float
    kl_p90: float


class RegimeBucket(BaseModel):
    """One bucket in :attr:`ConditioningResult.per_regime_per_perturbation`.

    Carries the regime label, the perturbation distribution label, and the
    summary :class:`RegimeStats`. The implementation plan §2.6 names this
    structure as a tuple-keyed dict; this list-of-buckets shape is the
    JSON-roundtrip-friendly alternative. Phase 6 adds ``find_bucket``
    helpers on :class:`ConditioningResult` for tuple-keyed access.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    regime: Regime
    perturbation: PerturbationDistribution
    stats: RegimeStats


class ConditioningResult(BaseModel):
    """One conditioning-analysis run's structured output.

    Produced by :func:`compute_conditioning` and written to
    ``runs/{run_id}/conditioning/conditioning.jsonl`` (one record per
    invocation, append-only across multiple invocations on the same run;
    each record tagged by ``checkpoint_id`` and ``timestamp_ms``).

    The faithfulness verifier resolves a behavior-side claim like
    ``"behavior-side conditioning at high_disagreement under gaussian
    perturbation has KL_p90 = 7.4e-8"`` by finding the matching
    :class:`ConditioningResult` (by ``run_id`` and ``checkpoint_id``),
    finding the matching :class:`RegimeBucket` for
    ``("high_disagreement", "gaussian")``, and comparing the cited
    ``kl_p90`` against the actual ``kl_p90`` within tolerance.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = CONDITIONING_RESULT_SCHEMA_VERSION
    run_id: str
    checkpoint_id: str
    timestamp_ms: int
    n_states_sampled: int
    perturbation_distributions: list[PerturbationDistribution]
    regimes: list[Regime]
    empirical_scalar_mean: float
    empirical_scalar_sigma: float
    empirical_scalar_range: tuple[float, float]
    per_regime_per_perturbation: list[RegimeBucket]
    masked_steps_excluded: int


def compute_conditioning(
    run_dir: Path,
    *,
    checkpoint_id: str | None = None,
    n_states: int = 200,
    perturbation_distributions: list[PerturbationDistribution] | None = None,
    regimes: list[Regime] | None = None,
    output_path: Path | None = None,
) -> ConditioningResult:
    """Phase 0 stub. Full implementation lands in Phase 6.

    Phase 6 will:

    1. Load the run's checkpoint at ``run_dir``, defaulting to the
       latest ``ckpt-*`` if ``checkpoint_id`` is None.
    2. Sample ``n_states`` states from the run's ``agent_step`` parquet,
       excluding states with ``self_prediction_error_masked_t == True``.
    3. Classify each state into one of :data:`Regime` based on its
       ``intrinsic_signal_t``, ``kl_aggregate_t``, and proximity to
       ``builder_perturbation`` world_events.
    4. For each (state, perturbation_distribution) pair: forward the
       actor on the unperturbed PolicyView and on a PolicyView with the
       scalar perturbed; compute the KL between the two action
       distributions.
    5. Aggregate per (regime, perturbation_distribution) into
       :class:`RegimeStats` records, packed into :class:`RegimeBucket`
       entries.
    6. Build a :class:`ConditioningResult` and, if ``output_path`` is
       supplied, append a JSONL line to that path (creating parents as
       needed). Return the result.

    Phase 0's stub raises :class:`NotImplementedError` — the schema
    exists for downstream importers, the function exists so its name
    resolves at import time, and Phase 6 fills in the body. The
    signature here is the contract Phase 6 must honor.
    """

    raise NotImplementedError(
        "compute_conditioning lands in Phase 6. Phase 0 defines the "
        "ConditioningResult schema and this signature only; the analysis "
        "logic — checkpoint loading, state sampling, regime "
        "classification, per-state action-distribution-KL computation, "
        "JSONL emission — is the Phase 6 concern. See the module "
        "docstring and implementation plan §2.6 for the contract."
    )
