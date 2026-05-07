#!/usr/bin/env python3
"""Probe 1.5 four-way comparison driver — substrate-side KS-tests and
per-tensor weight moments across Probe 1 (no-affordance baseline),
Phase 7 (Probe 1.5 main run, zero-init column), Phase 7.5 (Probe 1.5
main run, small-Gaussian column), and the frozen-target control.

Plan §9.2 (extended to four-way per §13 lean revision after Phase 7.5
landed) / Phase 8 of the implementation plan. The substrate-side
question the comparison reads:

- Phase 7.5 vs Probe 1: does the auxiliary loss (with the small-Gaussian
  column on the actor side) shape the substrate measurably vs the
  no-affordance baseline?
- Phase 7.5 vs frozen-target: does the auxiliary's substrate-shaping
  require self-specific targets (the EMA next-state) or does the
  random-orthogonal target produce similar effects?
- Frozen-target vs Probe 1: does the random-target version still differ
  from the no-affordance baseline (sanity check; if no, the auxiliary's
  structural effect requires the self-specific target).
- Phase 7 vs Phase 7.5: does the column-init choice affect the
  substrate-side trajectory? (Substrate trains on real env trajectories
  independent of the actor's column; expected to be small.)

Six pairings total (the four above plus Phase 7 vs Probe 1 and Phase 7
vs frozen-target for completeness). For each pairing, three KS-tests on
per-step distributions over episodes 5-25 (warmup skipped):

- ``kl_aggregate_t`` — substrate posterior-prior KL.
- ``recon_loss_t`` — encoder/decoder reconstruction.
- ``self_prediction_error_t`` — auxiliary head's loss value, masked
  steps excluded; only meaningful for pairings between two Probe 1.5
  runs (Probe 1's parquet does not carry the column).

Per-tensor weight moments at the final checkpoint per named parameter
tensor: mean, abs-mean, std, L2 norm, max abs. For each pairing the
table reports tensor-shape-matched comparison (notably
``actor.net.0.weight`` has shape ``(200, 216)`` for Probe 1 and
``(200, 217)`` for Probe 1.5; the comparison reports the ``[:, :216]``
shared slice for cross-version pairings and the full tensor for
within-version pairings).

The KS p-value uses the asymptotic Smirnov distribution (good for
n_a + n_b >> 100; sample sizes here are ~4000 per run after the
episodes-5-25 filter). Implementation is numpy-only; ``scipy`` is not a
project dependency.

Output lands at ``runs/probe1_5_comparison-<timestamp>/summary.txt``.

Run:
    .venv/bin/python scripts/probe1_5_compare_controls.py
    .venv/bin/python scripts/probe1_5_compare_controls.py --dry-run
    .venv/bin/python scripts/probe1_5_compare_controls.py \\
        --probe1 runs/probe1-20260503-123926 \\
        --phase7 runs/probe1_5-20260506-202458 \\
        --phase7-5 runs/probe1_5_phase7_5-20260507-101800 \\
        --frozen-target runs/probe1_5_control_frozen_target-20260507-112854
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray
from safetensors.torch import load_file


# ---- run-prefix discovery -------------------------------------------------

_RUNS_DIR = Path("runs")
_PROBE_1_PREFIX = "probe1-"
_PHASE_7_PREFIX = "probe1_5-"
_PHASE_7_5_PREFIX = "probe1_5_phase7_5-"
_FROZEN_TARGET_PREFIX = "probe1_5_control_frozen_target-"
_OUTPUT_PREFIX = "probe1_5_comparison-"

# Episodes to compare: 5 through 25 inclusive (i.e. skip the four
# warmup-affected episodes and use the remaining 21). Plan §9.2.
_EPISODE_START = 5
_EPISODE_END_INCLUSIVE = 25  # 25 episodes total (0-24); use 5-24 + boundary

# Significance threshold for the verbal "distinguishable" / "indistinguishable"
# label in the per-pairing one-line summary. The plan §9.2 names p < 0.05 as
# the threshold; the report carries the exact p-value either way.
_P_THRESHOLD = 0.05


# ---- KS two-sample (asymptotic) ------------------------------------------


def _ks_two_sample(a: NDArray[np.float64], b: NDArray[np.float64]) -> tuple[float, float]:
    """Compute the two-sample Kolmogorov–Smirnov statistic and asymptotic
    p-value.

    Both arrays must be 1-D and finite; callers should filter NaN/Inf
    before calling. The implementation is the standard pooled-sort form:
    sort both samples, build their empirical CDFs at the union of
    sample points, take the maximum absolute difference, and convert
    via the Smirnov asymptotic distribution. Equivalent to
    ``scipy.stats.ks_2samp(a, b, alternative="two-sided", method="asymp")``
    to within numerical precision; verified by spot check on the
    distinct-distribution test in ``tests/test_compare_controls.py``.

    Asymptotic p-value:

        p = 2 sum_{j=1..inf} (-1)^{j-1} exp(-2 j^2 lambd^2)

    with ``lambd = (sqrt(en) + 0.12 + 0.11 / sqrt(en)) D``, ``en = n_a
    n_b / (n_a + n_b)``, per Press et al., Numerical Recipes §14.3.4.
    The series converges quickly; ``j ∈ [1, 100]`` is overkill but
    cheap.
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(f"KS expects 1-D arrays; got a.ndim={a.ndim}, b.ndim={b.ndim}")
    if a.size == 0 or b.size == 0:
        raise ValueError(
            f"KS requires non-empty samples; got len(a)={a.size}, len(b)={b.size}"
        )

    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    n_a = a_sorted.size
    n_b = b_sorted.size

    # Empirical CDFs evaluated at the union of sample points.
    union = np.concatenate([a_sorted, b_sorted])
    cdf_a = np.searchsorted(a_sorted, union, side="right") / n_a
    cdf_b = np.searchsorted(b_sorted, union, side="right") / n_b
    d_stat = float(np.max(np.abs(cdf_a - cdf_b)))

    # Numerical safety: D=0 means the empirical CDFs are point-wise
    # identical, which corresponds to p=1 (the asymptotic series
    # alternates ±1 at lambd=0 and degenerates).
    if d_stat == 0.0:
        return 0.0, 1.0

    en = np.sqrt(n_a * n_b / (n_a + n_b))
    lambd = (en + 0.12 + 0.11 / en) * d_stat

    # Smirnov series; clip to [0, 1] for numerical safety.
    j = np.arange(1, 101)
    p = 2.0 * float(np.sum((-1.0) ** (j - 1) * np.exp(-2.0 * j**2 * lambd**2)))
    p = max(0.0, min(1.0, p))
    return d_stat, p


# ---- run loading ----------------------------------------------------------


@dataclass(frozen=True)
class _Moments:
    mean: float
    abs_mean: float
    std: float
    l2_norm: float
    max_abs: float
    shape: tuple[int, ...]


@dataclass
class _RunStats:
    """Per-run extracted arrays and per-tensor weight moments.

    ``label`` is the short name used in the summary report ("Probe 1",
    "Phase 7", etc.). ``run_dir`` is the directory the data was loaded
    from; included for the summary's provenance line.

    Per-step arrays are restricted to episodes ``_EPISODE_START`` through
    ``_EPISODE_END_INCLUSIVE`` inclusive. ``self_prediction_error`` excludes
    masked steps; arrays are empty for runs whose parquet has no
    ``self_prediction_error_t`` column (Probe 1).

    ``weight_moments`` keys are the safetensors named parameter tensor
    names (e.g. ``"actor.net.0.weight"``); values are :class:`_Moments`
    holding per-tensor scalar moments.
    """

    label: str
    run_dir: Path
    schema_version: str
    n_total_steps: int
    n_filtered_steps: int

    kl_aggregate: NDArray[np.float64]
    recon_loss: NDArray[np.float64]
    self_prediction_error: NDArray[np.float64]
    has_self_prediction: bool

    weight_moments: dict[str, _Moments] = field(default_factory=dict)
    checkpoint_id: str | None = None


def _read_agent_step_arrays(
    telemetry_dir: Path,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    bool,
    NDArray[np.int64],
    str,
]:
    """Read every ``agent_step`` parquet shard under ``telemetry_dir`` and
    return per-step arrays.

    Returns a 6-tuple ``(kl, recon, sp_err, has_sp, episode_ids,
    schema_version)``. ``sp_err`` is empty when the column is absent
    (Probe 1) *or* when every value is masked. Masked steps are
    excluded from ``sp_err`` only — ``kl`` and ``recon`` retain all
    rows. ``has_sp`` is True iff the ``self_prediction_error_t`` column
    is present in the parquet schema *and* contains at least one
    non-masked entry.
    """
    agent_step_dir = telemetry_dir / "agent_step"
    if not agent_step_dir.is_dir():
        raise FileNotFoundError(f"no agent_step/ under {telemetry_dir}")

    shards = sorted(agent_step_dir.glob("shard-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {agent_step_dir}")

    kl_chunks: list[NDArray[np.float64]] = []
    recon_chunks: list[NDArray[np.float64]] = []
    sp_chunks: list[tuple[NDArray[np.float64], NDArray[np.bool_]]] = []
    ep_chunks: list[NDArray[np.int64]] = []
    sv_seen: set[str] = set()

    has_sp_column = False
    has_sp_data = False
    for shard in shards:
        table = pq.read_table(str(shard))  # type: ignore[no-untyped-call]
        columns = table.column_names
        if "self_prediction_error_t" in columns:
            has_sp_column = True

        kl = np.asarray(table.column("kl_aggregate_t").to_pylist(), dtype=np.float64)
        recon = np.asarray(table.column("recon_loss_t").to_pylist(), dtype=np.float64)
        ep = np.asarray(table.column("episode_id").to_pylist(), dtype=np.int64)
        kl_chunks.append(kl)
        recon_chunks.append(recon)
        ep_chunks.append(ep)

        if has_sp_column:
            sp_raw = table.column("self_prediction_error_t").to_pylist()
            sp_arr = np.asarray(
                [float(x) if x is not None else np.nan for x in sp_raw],
                dtype=np.float64,
            )
            if "self_prediction_error_masked_t" in columns:
                masked_raw = table.column("self_prediction_error_masked_t").to_pylist()
                masked_arr = np.asarray(
                    [bool(x) if x is not None else False for x in masked_raw],
                    dtype=bool,
                )
            else:
                masked_arr = np.zeros_like(sp_arr, dtype=bool)
            keep = ~masked_arr & np.isfinite(sp_arr)
            if keep.any():
                has_sp_data = True
            sp_chunks.append((sp_arr, keep))

        sv_seen.update(table.column("schema_version").to_pylist())

    kl_all = np.concatenate(kl_chunks) if kl_chunks else np.zeros(0, dtype=np.float64)
    recon_all = (
        np.concatenate(recon_chunks) if recon_chunks else np.zeros(0, dtype=np.float64)
    )
    ep_all = np.concatenate(ep_chunks) if ep_chunks else np.zeros(0, dtype=np.int64)

    # Episode 5-25 (inclusive) filter applied here so callers don't have
    # to re-derive the indexing.
    keep_ep = (ep_all >= _EPISODE_START) & (ep_all <= _EPISODE_END_INCLUSIVE)

    if has_sp_column and sp_chunks:
        # The masked-step exclusion is per-step; for the episode-window
        # filter, rebuild the per-step sp array (with NaN at masked
        # positions), apply the episode filter, then drop NaN.
        sp_full = np.concatenate([sp_arr for sp_arr, _keep in sp_chunks])
        keep_full = np.concatenate([keep_mask for _sp, keep_mask in sp_chunks])
        sp_filtered = sp_full[keep_ep & keep_full & np.isfinite(sp_full)]
    else:
        sp_filtered = np.zeros(0, dtype=np.float64)

    schema_version_str = ",".join(sorted(sv_seen)) if sv_seen else "?"

    return (
        kl_all[keep_ep],
        recon_all[keep_ep],
        sp_filtered,
        has_sp_column and has_sp_data,
        ep_all[keep_ep],
        schema_version_str,
    )


def _latest_checkpoint_dir(checkpoints_root: Path) -> Path | None:
    """Return the highest-numbered ``ckpt-NNNNNN`` subdirectory, or None
    if no checkpoints exist."""
    if not checkpoints_root.is_dir():
        return None
    ckpts = sorted(
        d for d in checkpoints_root.iterdir() if d.is_dir() and d.name.startswith("ckpt-")
    )
    return ckpts[-1] if ckpts else None


def _per_tensor_moments(state_dict: dict[str, Any]) -> dict[str, _Moments]:
    """Compute per-tensor scalar moments for every entry in
    ``state_dict``. Tensor values are converted to numpy via
    ``.detach().cpu().float().numpy()`` before reduction."""
    out: dict[str, _Moments] = {}
    for name, tensor in state_dict.items():
        arr = tensor.detach().cpu().float().numpy()
        flat = arr.reshape(-1)
        if flat.size == 0:
            continue
        out[name] = _Moments(
            mean=float(flat.mean()),
            abs_mean=float(np.abs(flat).mean()),
            std=float(flat.std()),
            l2_norm=float(np.linalg.norm(flat)),
            max_abs=float(np.abs(flat).max()),
            shape=tuple(arr.shape),
        )
    return out


def load_run_stats(
    label: str, run_dir: Path, *, load_weights: bool = True
) -> _RunStats:
    """Public entry point used by both ``main`` and the tests. Reads
    the run's telemetry plus (optionally) the latest checkpoint's
    weights, and returns a :class:`_RunStats` instance."""
    telemetry_dir = run_dir / "telemetry"
    kl, recon, sp_err, has_sp, ep_kept, schema_version = _read_agent_step_arrays(
        telemetry_dir
    )

    weight_moments: dict[str, _Moments] = {}
    checkpoint_id: str | None = None
    if load_weights:
        ckpt_dir = _latest_checkpoint_dir(run_dir / "checkpoints")
        if ckpt_dir is not None:
            checkpoint_id = ckpt_dir.name
            weights_path = ckpt_dir / "weights.safetensors"
            if weights_path.is_file():
                state_dict = load_file(str(weights_path))
                weight_moments = _per_tensor_moments(state_dict)

    return _RunStats(
        label=label,
        run_dir=run_dir,
        schema_version=schema_version,
        n_total_steps=int(ep_kept.size),  # already filtered to episodes 5-25
        n_filtered_steps=int(ep_kept.size),
        kl_aggregate=kl,
        recon_loss=recon,
        self_prediction_error=sp_err,
        has_self_prediction=has_sp,
        weight_moments=weight_moments,
        checkpoint_id=checkpoint_id,
    )


# ---- pairwise comparison --------------------------------------------------


@dataclass(frozen=True)
class _KSResult:
    metric: str
    n_a: int
    n_b: int
    d_stat: float
    p_value: float
    available: bool  # False when one or both runs lack the metric


def _ks_or_unavailable(
    metric: str,
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> _KSResult:
    """Run KS on ``(a, b)`` if both are non-empty; otherwise return an
    "unavailable" result so the report can render the row consistently."""
    if a.size == 0 or b.size == 0:
        return _KSResult(
            metric=metric,
            n_a=int(a.size),
            n_b=int(b.size),
            d_stat=float("nan"),
            p_value=float("nan"),
            available=False,
        )
    d, p = _ks_two_sample(a, b)
    return _KSResult(
        metric=metric, n_a=int(a.size), n_b=int(b.size), d_stat=d, p_value=p, available=True
    )


def _pairwise_ks(stats_a: _RunStats, stats_b: _RunStats) -> list[_KSResult]:
    """Compute KS for the three substrate-side metrics."""
    return [
        _ks_or_unavailable("kl_aggregate_t", stats_a.kl_aggregate, stats_b.kl_aggregate),
        _ks_or_unavailable("recon_loss_t", stats_a.recon_loss, stats_b.recon_loss),
        _ks_or_unavailable(
            "self_prediction_error_t",
            stats_a.self_prediction_error,
            stats_b.self_prediction_error,
        ),
    ]


# ---- formatting -----------------------------------------------------------


def _format_p(p: float) -> str:
    """Render ``p`` as ``0.XXX`` with 3 sig figs in normal range and
    scientific notation below 1e-4. ``nan`` renders as ``nan``."""
    if np.isnan(p):
        return "nan"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _verdict_word(p: float, threshold: float = _P_THRESHOLD) -> str:
    if np.isnan(p):
        return "n/a"
    return "distinguishable" if p < threshold else "indistinguishable"


def _format_one_line_summary(label_a: str, label_b: str, results: list[_KSResult]) -> str:
    """Build the one-line per-pairing summary the plan §9.2 names.

    Example: "Phase 7.5 vs frozen-target: kl distinguishable (p=0.0001);
    recon distinguishable (p=2.34e-08); sp_err distinguishable (p=0.0023)"
    """
    parts: list[str] = []
    for r in results:
        short = {
            "kl_aggregate_t": "kl",
            "recon_loss_t": "recon",
            "self_prediction_error_t": "sp_err",
        }.get(r.metric, r.metric)
        if not r.available:
            parts.append(f"{short} n/a")
            continue
        parts.append(f"{short} {_verdict_word(r.p_value)} (p={_format_p(r.p_value)})")
    return f"{label_a} vs {label_b}: " + "; ".join(parts)


def _format_ks_table(label_a: str, label_b: str, results: list[_KSResult]) -> str:
    """Detailed per-pairing block: one row per metric with D, p, n_a, n_b."""
    lines = [f"  {label_a} vs {label_b}"]
    lines.append(f"    {'metric':<28} {'n_a':>6} {'n_b':>6} {'D':>8} {'p':>10}")
    for r in results:
        if r.available:
            lines.append(
                f"    {r.metric:<28} {r.n_a:>6} {r.n_b:>6} "
                f"{r.d_stat:>8.4f} {_format_p(r.p_value):>10}"
            )
        else:
            lines.append(
                f"    {r.metric:<28} {r.n_a:>6} {r.n_b:>6} "
                f"{'n/a':>8} {'n/a':>10}"
            )
    return "\n".join(lines)


def _format_run_overview(stats: list[_RunStats]) -> str:
    lines = ["## Run overview", ""]
    lines.append(
        f"  {'label':<18} {'schema':<8} {'episodes 5-25 steps':>22} "
        f"{'sp avail':>10} {'ckpt':>14}"
    )
    for s in stats:
        lines.append(
            f"  {s.label:<18} {s.schema_version:<8} {s.n_filtered_steps:>22} "
            f"{('yes' if s.has_self_prediction else 'no'):>10} "
            f"{(s.checkpoint_id or '-'):>14}"
        )
        lines.append(f"    run_dir: {s.run_dir}")
    return "\n".join(lines)


def _format_metric_summary(stats: list[_RunStats]) -> str:
    """Per-run mean/std/min/max for each metric — the descriptive companion
    to the inferential KS-table."""
    lines = ["## Per-run distribution summary (episodes 5-25)", ""]
    metrics = [
        ("kl_aggregate_t", "kl_aggregate"),
        ("recon_loss_t", "recon_loss"),
        ("self_prediction_error_t", "self_prediction_error"),
    ]
    for label, attr in metrics:
        lines.append(f"  {label}")
        lines.append(
            f"    {'run':<18} {'n':>7} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}"
        )
        for s in stats:
            arr: NDArray[np.float64] = getattr(s, attr)
            if arr.size == 0:
                lines.append(f"    {s.label:<18} {0:>7} {'-':>10} {'-':>10} {'-':>10} {'-':>10}")
                continue
            lines.append(
                f"    {s.label:<18} {arr.size:>7} "
                f"{float(arr.mean()):>10.4f} {float(arr.std()):>10.4f} "
                f"{float(arr.min()):>10.4f} {float(arr.max()):>10.4f}"
            )
        lines.append("")
    return "\n".join(lines)


def _format_pairwise_block(stats_pairs: list[tuple[_RunStats, _RunStats]]) -> str:
    """Detailed per-pairing KS table + the one-line summaries."""
    lines = ["## Pairwise KS-tests (episodes 5-25; per-step values)", ""]
    one_liners: list[str] = []
    for a, b in stats_pairs:
        results = _pairwise_ks(a, b)
        lines.append(_format_ks_table(a.label, b.label, results))
        lines.append("")
        one_liners.append(_format_one_line_summary(a.label, b.label, results))

    lines.append("### Per-pairing one-line summary")
    lines.append("")
    lines.extend(f"  {ol}" for ol in one_liners)
    lines.append("")
    return "\n".join(lines)


def _format_weight_moments(stats: list[_RunStats]) -> str:
    """Per-tensor weight moments at the final checkpoint, organized as a
    flat table per run.

    The actor's input layer's first column-set is special: Probe 1 has
    ``actor.net.0.weight`` of shape ``(200, 216)``; Probe 1.5 has
    ``(200, 217)`` (one extra column for the scalar
    ``self_prediction_error`` input). The shared ``[:, :216]`` slice is
    additionally compared (synthetic tensor key
    ``actor.net.0.weight[:, :216]``) so pairings can compare across the
    versions; the new column is reported separately as
    ``actor.net.0.weight[:, 216]``.
    """
    lines = ["## Per-tensor weight moments at final checkpoint", ""]

    # Add synthetic slice keys so cross-version comparisons line up.
    # Mutates each stats.weight_moments to include the slice keys; this
    # is a one-shot analyzer so persistence is fine.
    actor_full_key = "actor.net.0.weight"
    for s in stats:
        if actor_full_key not in s.weight_moments:
            continue
        full_shape = s.weight_moments[actor_full_key].shape
        if len(full_shape) != 2:
            continue
        n_cols = full_shape[1]
        # Slice [:, :216] = (h, z) input; only meaningful at h_dim+z_dim=216.
        # We compute it regardless so the comparison can quote it; load
        # state dict if missing.
        if n_cols >= 216:
            ckpt_dir = (
                _latest_checkpoint_dir(s.run_dir / "checkpoints")
                if s.checkpoint_id is not None
                else None
            )
            if ckpt_dir is not None:
                weights_path = ckpt_dir / "weights.safetensors"
                if weights_path.is_file():
                    sd = load_file(str(weights_path))
                    if actor_full_key in sd:
                        full_t = sd[actor_full_key].detach().cpu().float().numpy()
                        slice_arr = full_t[:, :216].reshape(-1)
                        s.weight_moments[f"{actor_full_key}[:, :216]"] = _Moments(
                            mean=float(slice_arr.mean()),
                            abs_mean=float(np.abs(slice_arr).mean()),
                            std=float(slice_arr.std()),
                            l2_norm=float(np.linalg.norm(slice_arr)),
                            max_abs=float(np.abs(slice_arr).max()),
                            shape=(full_shape[0], 216),
                        )
                        if n_cols >= 217:
                            new_col = full_t[:, 216:].reshape(-1)
                            s.weight_moments[f"{actor_full_key}[:, 216:]"] = _Moments(
                                mean=float(new_col.mean()),
                                abs_mean=float(np.abs(new_col).mean()),
                                std=float(new_col.std()),
                                l2_norm=float(np.linalg.norm(new_col)),
                                max_abs=float(np.abs(new_col).max()),
                                shape=(full_shape[0], n_cols - 216),
                            )

    # Highlight tensors of structural interest first.
    highlight = [
        "actor.net.0.weight",
        "actor.net.0.weight[:, :216]",
        "actor.net.0.weight[:, 216:]",
        "world_model.encoder.proj.weight",
        "world_model.gru_cell.weight_hh",
        "world_model.posterior_head.head.weight",
        "world_model.prior_head.head.weight",
    ]

    lines.append("### Highlighted tensors")
    lines.append("")
    for tname in highlight:
        rows = [(s.label, s.weight_moments.get(tname)) for s in stats]
        if all(m is None for _, m in rows):
            continue
        lines.append(f"  {tname}")
        lines.append(
            f"    {'run':<18} {'shape':<14} {'abs_mean':>10} "
            f"{'L2':>10} {'std':>10} {'max_abs':>10}"
        )
        for label, m in rows:
            if m is None:
                lines.append(f"    {label:<18} {'absent':<14} {'-':>10} {'-':>10} {'-':>10} {'-':>10}")
                continue
            lines.append(
                f"    {label:<18} {str(m.shape):<14} {m.abs_mean:>10.6f} "
                f"{m.l2_norm:>10.4f} {m.std:>10.6f} {m.max_abs:>10.6f}"
            )
        lines.append("")
    return "\n".join(lines)


def build_summary_text(stats: list[_RunStats]) -> str:
    """Assemble the full summary report. Public-by-name so the test
    can import it directly without invoking ``main``."""
    if len(stats) < 2:
        return f"# Probe 1.5 four-way comparison\n\n(only {len(stats)} runs found; nothing to compare)\n"

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
    header = [
        "# Probe 1.5 four-way comparison",
        "",
        f"Generated: {timestamp}",
        "Plan §9.2 (extended four-way per §13 lean revision) / Phase 8.",
        "",
    ]
    sections: list[str] = list(header)
    sections.append(_format_run_overview(stats))
    sections.append("")
    sections.append(_format_metric_summary(stats))

    # All ordered pairs (label_i, label_j) with i < j by stats order.
    pairs: list[tuple[_RunStats, _RunStats]] = [
        (stats[i], stats[j]) for i in range(len(stats)) for j in range(i + 1, len(stats))
    ]
    sections.append(_format_pairwise_block(pairs))
    sections.append(_format_weight_moments(stats))

    return "\n".join(sections) + "\n"


# ---- run discovery --------------------------------------------------------


def _latest_with_prefix(prefix: str, exclude_prefixes: tuple[str, ...] = ()) -> Path | None:
    """Return the most-recently-modified ``runs/<prefix>*`` directory,
    excluding any whose name *also* matches one of ``exclude_prefixes``.

    Sort by directory mtime so a manual ``touch`` can override the
    timestamp embedded in the run id. Falls back to alphabetical order
    when mtimes tie."""
    if not _RUNS_DIR.is_dir():
        return None
    candidates: list[Path] = []
    for d in _RUNS_DIR.iterdir():
        if not d.is_dir():
            continue
        if not d.name.startswith(prefix):
            continue
        if any(d.name.startswith(ex) for ex in exclude_prefixes):
            continue
        candidates.append(d)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return candidates[-1]


def _default_run_dirs() -> dict[str, Path | None]:
    """Discovery defaults — one run per role, latest-mtime within prefix.

    The prefixes overlap (``probe1_5_phase7_5-`` and
    ``probe1_5_control_frozen_target-`` both start with ``probe1_5-``);
    the Phase-7 (main run) discovery excludes the more-specific prefixes."""
    return {
        "probe1": _latest_with_prefix(_PROBE_1_PREFIX),
        "phase7": _latest_with_prefix(
            _PHASE_7_PREFIX,
            exclude_prefixes=(_PHASE_7_5_PREFIX, _FROZEN_TARGET_PREFIX),
        ),
        "phase7_5": _latest_with_prefix(_PHASE_7_5_PREFIX),
        "frozen_target": _latest_with_prefix(_FROZEN_TARGET_PREFIX),
    }


# ---- CLI ------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="probe1_5_compare_controls",
        description=(
            "Probe 1.5 four-way comparison driver: pairwise KS-tests on "
            "per-step distributions over episodes 5-25 across Probe 1, "
            "Phase 7 (Probe 1.5 main), Phase 7.5 (small-Gaussian column), "
            "and the frozen-target control, plus per-tensor weight "
            "moments at the final checkpoint of each run."
        ),
    )
    parser.add_argument(
        "--probe1",
        type=Path,
        help="Probe 1 baseline run dir (default: latest runs/probe1-*).",
    )
    parser.add_argument(
        "--phase7",
        type=Path,
        help=(
            "Phase 7 (Probe 1.5 main, zero-init column) run dir "
            "(default: latest runs/probe1_5-* excluding phase7_5 and control_*)."
        ),
    )
    parser.add_argument(
        "--phase7-5",
        type=Path,
        dest="phase7_5",
        help=(
            "Phase 7.5 (small-Gaussian column) run dir (default: latest "
            "runs/probe1_5_phase7_5-*)."
        ),
    )
    parser.add_argument(
        "--frozen-target",
        type=Path,
        dest="frozen_target",
        help=(
            "Frozen-target control run dir (default: latest "
            "runs/probe1_5_control_frozen_target-*)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Override the output directory. By default a fresh "
            "runs/probe1_5_comparison-<timestamp>/ is created."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print which run dirs would be loaded and return 0 without "
            "reading any data, computing any KS-tests, or writing any "
            "output. Used by the test."
        ),
    )
    return parser.parse_args(argv)


def _resolve_run_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    defaults = _default_run_dirs()
    override = {
        "probe1": args.probe1,
        "phase7": args.phase7,
        "phase7_5": args.phase7_5,
        "frozen_target": args.frozen_target,
    }
    return {key: (override[key] or defaults[key]) for key in defaults}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = _resolve_run_paths(args)

    role_to_label = {
        "probe1": "Probe 1",
        "phase7": "Phase 7",
        "phase7_5": "Phase 7.5",
        "frozen_target": "frozen-target",
    }

    print("Probe 1.5 four-way comparison driver")
    print("=" * 60)
    for role, label in role_to_label.items():
        path = paths[role]
        suffix = "(MISSING)" if path is None or not path.is_dir() else ""
        print(f"  {label:<18} {path or '-'} {suffix}")
    print("=" * 60)

    found = [(role, p) for role, p in paths.items() if p is not None and p.is_dir()]
    if len(found) < 2:
        print(
            f"need at least 2 run directories on disk to compare; got {len(found)}",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        print("[dry-run] no data loaded, no KS-tests run, no output written")
        return 0

    # Output directory
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        out_dir = _RUNS_DIR / time.strftime(
            f"{_OUTPUT_PREFIX}%Y%m%d-%H%M%S", time.localtime()
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print(f"loading runs ({len(found)} of 4 available)...")
    stats: list[_RunStats] = []
    for role, path in found:
        label = role_to_label[role]
        print(f"  loading {label} from {path}")
        stats.append(load_run_stats(label, path))
        print(
            f"    {stats[-1].n_filtered_steps} steps in episodes 5-25; "
            f"sp_err available: {stats[-1].has_self_prediction}; "
            f"checkpoint: {stats[-1].checkpoint_id}"
        )

    print()
    print("computing pairwise KS-tests + per-tensor weight moments...")
    summary_text = build_summary_text(stats)

    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"summary written to {summary_path.resolve()}")
    print()
    print("---- summary ----")
    print(summary_text)
    return 0


if __name__ == "__main__":  # pragma: no cover — manual entry point
    sys.exit(main())
