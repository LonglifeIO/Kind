"""Phase 7 digest — per-episode Markdown summary of ``agent_step`` rows.

Lifted from the Phase 6 mirror caller, where the same function lived
behind a ``_`` prefix. Two consumers want the same shape:

* The mirror caller (``kind/mirror/caller.py``) feeds the digest text
  into Gemini as the readable input alongside the structured-output
  schema; the LLM reads what would otherwise be a 100k-token parquet
  dump.
* The eyeball helpers (``kind/observer/eyeball.py``) print the same
  digest to stdout for the human builder. The principle is that what
  the mirror sees and what the human sees should be the same shape;
  reading the digest directly is how the builder eyeballs without the
  mirror in the loop.

The digest is structured Markdown rather than JSON because Gemini parses
prose context better than dumped JSON when there is no ``response_schema``
constraint on the *input*. The mirror's output is constrained by the
:class:`~kind.mirror.caller.MirrorReadingPayload` schema; the input is
intentionally readable.

The high-dimensional fields (``h_t``, ``z_t``, ``q_params_t``, ``p_params_t``,
``kl_per_dim_t``, ``encoder_embedding_t``) are deliberately excluded — they
are not legible to a language model and would dominate the token budget
without carrying scalar-summary signal. The full records remain in the
parquet shards for any consumer that wants statistical analysis directly.

**Probe 1.5 v2 self-prediction extension** (plan §2.6). For records stamped
``schema_version == "0.2.0"`` (and only those), each per-episode block
gains four extra lines: ``self_prediction_error_t`` mean/std/min/max
excluding masked steps; the per-episode masked-step count; outliers
(non-masked steps whose self-prediction error z-score against the
episode's non-masked mean exceeds ``_KL_OUTLIER_Z_THRESHOLD``); per-
dimension allocation top-5 (the five ``h`` dimensions whose self-
prediction-error variance across the non-masked, non-last steps of the
episode is highest, where the per-step per-dim error is
``self_prediction_t[d] - h_{(t+1)}[d]`` using the next non-masked step's
``h_t`` as the proxy for the EMA target ``bar{h}_{t+1}`` that the runner
trained against). Probe 1 (``"0.1.0"``) records produce no self-prediction
summary lines — the no-affordance baseline stays visibly distinct from a
Probe 1.5 record where the head produced a near-zero value (plan §3.3
"skip" backward-compat approach).

**Probe 2 v2 hierarchical digest** (plan §2.3). The flat
:func:`build_digest` is preserved as :func:`build_flat_digest` for
backward-readability of any callers that consumed the flat shape.
:func:`build_hierarchical_digest` is the Probe 2 entry point: it produces
a :class:`HierarchicalDigest` whose per-episode mini-digest is structured
as three labeled cohorts — ``[substrate-side]`` (KL allocation, ensemble
disagreement, perturbation-recovery), ``[head-internal]`` (self-prediction
error distribution and per-dim allocation), ``[behavior-side]`` (per-state
action-distribution-under-perturbation summary loaded from the
conditioning analysis module's output cache). The cohort labels are
explicit in the digest's prose; the readers' prompts (Phase 8) reference
each cohort by name; the Judge's per-surface rulings (Phase 9) cite the
cohort the claim sits on. Graceful degradation: Probe 1 (``"0.1.0"``)
records produce a substrate-side cohort with full content and
head-internal / behavior-side cohorts replaced by a single
"(no self-prediction telemetry — records are Probe 1, schema_version
0.1.0)" line. Conditioning-dir omitted: behavior-side cohort replaced by
"(no conditioning data — conditioning_dir not supplied)" line; the
substrate-side and head-internal cohorts unchanged. Sham builder
perturbations (``payload["is_sham"]==True``; Probe 2 implementation plan
§2.2) are surfaced in the world-event timeline with a ``[SHAM]`` prefix
and the carrier ``sham_label``.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq

__all__ = [
    "build_digest",
    "build_flat_digest",
    "build_hierarchical_digest",
    "compact_record_repr",
    "DrillDownAccessor",
    "FlaggedAnomaly",
    "HierarchicalDigest",
]


# Per-episode aggregation thresholds. Held as module-level constants so
# the values are visible at the top of the file rather than buried mid
# function — they are the load-bearing knobs of the digest's signal.
_KL_OUTLIER_Z_THRESHOLD: float = 3.0
_ACTION_SKEW_THRESHOLD: float = 0.7
_RECON_TOP_N_DIVISOR: int = 100  # n_top = max(1, n_steps // this)
_RECON_OUTLIER_MIN_STEPS: int = 5  # don't flag recon outliers under this

# Probe 1.5 v2 (plan §2.6).
_SELF_PREDICTION_SCHEMA_VERSION: str = "0.2.0"
_SELF_PREDICTION_TOP_DIMS: int = 5


def build_digest(rows: list[dict[str, Any]]) -> str:
    """Compose the LLM/human-readable summary of ``rows``.

    Per-episode block: step count, ``kl_aggregate_t`` (mean/std/min/max),
    ``recon_loss_t`` mean, ``intrinsic_signal_t`` mean, ``policy_entropy_t``
    mean, action distribution (counts for actions 0-4), flagged outliers
    (``kl_aggregate_t`` z > 3, top recon-loss steps), and three sample
    records (first/middle/last) with scalar fields only.

    Returns ``"(no records)"`` for an empty list.
    """
    if not rows:
        return "(no records)"

    by_ep: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_ep[int(r["episode_id"])].append(r)

    lines: list[str] = []

    total_steps = len(rows)
    total_episodes = len(by_ep)
    wc_first = int(rows[0]["wallclock_ms"])
    wc_last = int(rows[-1]["wallclock_ms"])
    span_ms = wc_last - wc_first
    t_first = int(rows[0]["t"])
    t_last = int(rows[-1]["t"])
    schema_version = str(rows[0].get("schema_version", "?"))
    record_run_id = str(rows[0].get("run_id", "?"))

    lines.append("# Telemetry Digest (agent_step records)")
    lines.append("")
    lines.append(f"- schema_version: {schema_version}")
    lines.append(f"- run_id: {record_run_id}")
    lines.append(f"- episodes covered: {total_episodes}")
    lines.append(f"- total agent_step records: {total_steps}")
    lines.append(f"- env_step range: [{t_first}, {t_last}]")
    lines.append(f"- wallclock span: {span_ms} ms ({wc_first} -> {wc_last})")
    lines.append("")

    for eid in sorted(by_ep.keys()):
        ep_rows = sorted(by_ep[eid], key=lambda r: int(r["step_in_episode"]))
        n_steps = len(ep_rows)
        kl_vals = [float(r["kl_aggregate_t"]) for r in ep_rows]
        recon_vals = [float(r["recon_loss_t"]) for r in ep_rows]
        intrinsic_vals = [float(r["intrinsic_signal_t"]) for r in ep_rows]
        entropy_vals = [float(r["policy_entropy_t"]) for r in ep_rows]
        actions = [int(r["action_t"]) for r in ep_rows]
        action_counts = [actions.count(a) for a in range(5)]

        kl_mean = _mean(kl_vals)
        kl_std = _std(kl_vals, kl_mean)
        recon_mean = _mean(recon_vals)
        intrinsic_mean = _mean(intrinsic_vals)
        entropy_mean = _mean(entropy_vals)

        skew_max = max(action_counts) / n_steps if n_steps else 0.0

        outliers: list[str] = []
        if kl_std > 0:
            for r in ep_rows:
                kl_v = float(r["kl_aggregate_t"])
                z = (kl_v - kl_mean) / kl_std
                if abs(z) > _KL_OUTLIER_Z_THRESHOLD:
                    outliers.append(
                        f"step_in_episode={int(r['step_in_episode'])}: "
                        f"kl_aggregate_t={kl_v:.4f} (z={z:+.2f})"
                    )
        if n_steps >= _RECON_OUTLIER_MIN_STEPS:
            sorted_by_recon = sorted(
                ep_rows, key=lambda r: -float(r["recon_loss_t"])
            )
            n_top = max(1, n_steps // _RECON_TOP_N_DIVISOR)
            for r in sorted_by_recon[:n_top]:
                v = float(r["recon_loss_t"])
                outliers.append(
                    f"step_in_episode={int(r['step_in_episode'])}: "
                    f"recon_loss_t={v:.4f} (top recon)"
                )

        lines.append(f"## episode {eid}")
        lines.append(f"- step count: {n_steps}")
        lines.append(
            f"- kl_aggregate_t: mean={kl_mean:.4f}, std={kl_std:.4f}, "
            f"min={min(kl_vals):.4f}, max={max(kl_vals):.4f}"
        )
        lines.append(f"- recon_loss_t: mean={recon_mean:.4f}")
        lines.append(f"- intrinsic_signal_t: mean={intrinsic_mean:.4f}")
        lines.append(f"- policy_entropy_t: mean={entropy_mean:.4f}")
        skew_note = " (skewed)" if skew_max > _ACTION_SKEW_THRESHOLD else ""
        lines.append(
            f"- action_t distribution (counts for actions 0..4): "
            f"{action_counts}{skew_note}"
        )
        if outliers:
            lines.append("- flagged outliers:")
            for s in outliers:
                lines.append(f"  - {s}")
        else:
            lines.append("- flagged outliers: none")

        # Probe 1.5 v2 self-prediction summary (plan §2.6). Only emitted
        # when the episode contains 0.2.0 records with non-None
        # self_prediction_error_t — Probe 1 episodes skip the block to
        # keep the no-affordance baseline visibly distinct from a Probe
        # 1.5 record where the head produced a near-zero value (plan §3.3
        # "skip" approach over "default to zero").
        _emit_self_prediction_lines(lines, ep_rows)

        if n_steps >= 3:
            sample_indices = sorted({0, n_steps // 2, n_steps - 1})
        else:
            sample_indices = list(range(n_steps))
        lines.append("- sample records:")
        for ix in sample_indices:
            label = _sample_label(ix, n_steps)
            lines.append(f"  - {compact_record_repr(ep_rows[ix], label)}")
        lines.append("")

    return "\n".join(lines)


def _emit_self_prediction_lines(
    lines: list[str], ep_rows: list[dict[str, Any]]
) -> None:
    """Append the Probe 1.5 v2 self-prediction block for one episode.

    Discriminates by schema_version per plan §3.3 "skip" backward-compat
    approach: only 0.2.0 records contribute. Masked steps (sentinel
    scalar=0.0, masked_flag=True; first step of each episode under the
    Phase 0 convention) are excluded from the empirical mean / std /
    min / max and from the per-dimension allocation; the masked-step
    count is logged separately so the mirror sees what was excluded.

    Outlier detection mirrors the kl_aggregate path: any non-masked
    step whose self-prediction-error z-score against the episode's
    non-masked mean exceeds ``_KL_OUTLIER_Z_THRESHOLD`` is flagged.

    Per-dimension allocation (synthesis §1.4 element 3) uses the next
    non-masked step's ``h_t`` as the proxy for the EMA target's
    ``bar{h}_{t+1}`` that the runner trained against (the EMA target
    itself is not in telemetry; the online ``h`` at t+1 is the closest
    available stand-in). Each per-step per-dim error is
    ``self_prediction_t[d] - h_{(t+1)}[d]``, taken over consecutive
    non-masked, in-episode pairs.
    """
    rows_with_scalar = [
        r
        for r in ep_rows
        if str(r.get("schema_version", "")) == _SELF_PREDICTION_SCHEMA_VERSION
        and r.get("self_prediction_error_t") is not None
    ]
    if not rows_with_scalar:
        return

    masked_count = sum(
        1 for r in rows_with_scalar if r.get("self_prediction_error_masked_t") is True
    )
    non_masked = [
        r
        for r in rows_with_scalar
        if r.get("self_prediction_error_masked_t") is not True
    ]
    if not non_masked:
        # Edge case: every 0.2.0 record in the episode is masked. Emit
        # the masked-step count and skip the empirical aggregates so the
        # mirror still sees that the schema bumped, just with no usable
        # readings.
        lines.append(
            f"- self_prediction_error_t masked steps in episode: count={masked_count}"
        )
        return

    sp_vals = [float(r["self_prediction_error_t"]) for r in non_masked]
    sp_mean = _mean(sp_vals)
    sp_std = _std(sp_vals, sp_mean)

    lines.append(
        f"- self_prediction_error_t (excluding masked steps): "
        f"mean={sp_mean:.4f}, std={sp_std:.4f}, "
        f"min={min(sp_vals):.4f}, max={max(sp_vals):.4f}"
    )
    lines.append(
        f"- self_prediction_error_t masked steps in episode: count={masked_count}"
    )

    if sp_std > 0:
        outliers: list[str] = []
        for r in non_masked:
            v = float(r["self_prediction_error_t"])
            z = (v - sp_mean) / sp_std
            if abs(z) > _KL_OUTLIER_Z_THRESHOLD:
                outliers.append(
                    f"step_in_episode={int(r['step_in_episode'])}: "
                    f"self_prediction_error_t={v:.4f} (z={z:+.2f})"
                )
        if outliers:
            lines.append("- self_prediction outliers:")
            for s in outliers:
                lines.append(f"  - {s}")

    per_dim_var = _per_dim_self_prediction_error_variance(rows_with_scalar)
    if per_dim_var:
        ranked = sorted(
            range(len(per_dim_var)), key=lambda d: per_dim_var[d], reverse=True
        )
        top = ranked[: min(_SELF_PREDICTION_TOP_DIMS, len(ranked))]
        if top:
            lines.append(
                f"- self_prediction allocation (per-dim variance across "
                f"episode, top {len(top)} dims):"
            )
            for d in top:
                lines.append(f"  - dim={d}: var={per_dim_var[d]:.4f}")


def _per_dim_self_prediction_error_variance(
    rows: list[dict[str, Any]],
) -> list[float]:
    """Per-dim variance of ``self_prediction_t[d] - h_{(t+1)}[d]``.

    Operates on a single-episode row list pre-filtered to 0.2.0 records
    in step order. Walks consecutive pairs ``(t, t+1)`` where the
    *predicting* row is not masked and the *target* row exists in the
    same window; uses ``ep_rows[i+1]['h_t']`` as the proxy for
    ``bar{h}_{t+1}`` (the actual EMA target from the runner is not in
    telemetry; the next-step online ``h`` is the closest available
    stand-in for behaviour-side analyzers). Returns a list of length
    ``h_dim`` with sample variance per dim, or an empty list if no
    valid pairs exist.

    Two-or-more residuals are required for a non-degenerate variance
    estimate; any dim with fewer pairs returns 0.0 in that slot. The
    list shape stays at ``h_dim`` regardless so the caller's top-k
    sort treats every dim uniformly.
    """
    pairs: list[tuple[list[float], list[float]]] = []
    for i, r in enumerate(rows):
        if r.get("self_prediction_error_masked_t") is True:
            continue
        if i + 1 >= len(rows):
            continue
        sp = r.get("self_prediction_t")
        nxt_h = rows[i + 1].get("h_t")
        if sp is None or nxt_h is None:
            continue
        if not isinstance(sp, list) or not isinstance(nxt_h, list):
            continue
        if len(sp) != len(nxt_h):
            continue
        pairs.append(([float(x) for x in sp], [float(x) for x in nxt_h]))
    if not pairs:
        return []
    h_dim = len(pairs[0][0])
    per_dim_vars: list[float] = []
    for d in range(h_dim):
        residuals = [p[0][d] - p[1][d] for p in pairs]
        if len(residuals) < 2:
            per_dim_vars.append(0.0)
            continue
        m = sum(residuals) / len(residuals)
        v = sum((x - m) ** 2 for x in residuals) / (len(residuals) - 1)
        per_dim_vars.append(v)
    return per_dim_vars


def compact_record_repr(r: dict[str, Any], position: str) -> str:
    """One-line JSON of a record's scalar fields, with floats rounded.

    The high-dimensional fields are dropped — they are unreadable in a
    prompt and bloat the token budget. The scalar fields are the ones a
    Probe 1 calibration reading actually uses.
    """
    scalar_fields = (
        "t",
        "episode_id",
        "step_in_episode",
        "wallclock_ms",
        "kl_aggregate_t",
        "recon_loss_t",
        "action_t",
        "action_logprob_t",
        "policy_entropy_t",
        "intrinsic_signal_t",
        "obs_hash_t",
        # Probe 1.5 v2 (plan §2.6): the scalar Io reads on PolicyView
        # plus the masked-flag the mirror uses to discriminate sentinel
        # zero from empirical near-zero. ``self_prediction_t`` (the
        # full ``ĥ_{t+1}`` vector) is intentionally absent — high-dim
        # vectors stay out of the digest's compact-repr line per the
        # module's "scalar-only" discipline. Probe 1 records lack both
        # fields entirely; the ``if k in r`` guard below silently skips
        # them so 0.1.0 records render with their original Probe 1 shape.
        "self_prediction_error_t",
        "self_prediction_error_masked_t",
    )
    out: dict[str, Any] = {"__pos": position}
    for k in scalar_fields:
        if k not in r:
            continue
        v = r[k]
        # The two Probe 1.5 self-prediction fields are ``T | None`` on
        # the schema; a 0.1.0 record round-tripped through the 0.2.0
        # ParquetSink carries them as ``None``. Skip them in that case
        # so the compact repr stays visibly distinct between Probe 1
        # records (no scalar; no flag) and Probe 1.5 records (both
        # populated — the writer-side validator at synthesis §3.3
        # guarantees non-None on 0.2.0 emissions).
        if (
            v is None
            and k in ("self_prediction_error_t", "self_prediction_error_masked_t")
        ):
            continue
        if isinstance(v, float):
            out[k] = round(v, 4)
        else:
            out[k] = v
    return json.dumps(out, sort_keys=True)


def _sample_label(ix: int, n_steps: int) -> str:
    if ix == 0:
        return "first"
    if ix == n_steps - 1:
        return "last"
    return "middle"


def _mean(vals: list[float]) -> float:
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _std(vals: list[float], mean: float) -> float:
    if len(vals) < 2:
        return 0.0
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))


# ---------------------------------------------------------------------------
# Probe 2 hierarchical digest (plan §2.3).
# ---------------------------------------------------------------------------


# ``build_flat_digest`` is the backward-readable alias for the original
# Probe-1-shape flat digest. The two names point to the same function:
# Probe 2 callers that want the three-cohort hierarchical layout call
# :func:`build_hierarchical_digest`; Probe 1 / Probe 1.5 callers that
# consumed the flat shape (the eyeball helpers, the Probe 1 mirror caller)
# see no behavioural change under either name.
build_flat_digest = build_digest


_HIERARCHICAL_TOP_DIMS: int = 5
_BEHAVIOR_KL_ANOMALY_DEFAULT_Z: float = 3.0
# Phase 7's u=200 monomorphic regime had per-episode mean
# ``intrinsic_signal_t`` near zero; Phase 7.5's healthy runs stay above
# 0.01 throughout. The threshold is the band between them — tunable.
_ENSEMBLE_COLLAPSE_THRESHOLD: float = 1e-3
_PROBE_1_DEGRADED_MESSAGE: str = (
    "(no self-prediction telemetry — records are Probe 1, "
    "schema_version 0.1.0)"
)
_NO_CONDITIONING_DEGRADED_MESSAGE: str = (
    "(no conditioning data — conditioning_dir not supplied)"
)
_DEFAULT_BEHAVIOR_REGIMES: tuple[str, ...] = (
    "perturbation_window",
    "high_disagreement",
    "high_kl",
    "steady_state",
)


ReadingSurface = Literal["substrate_side", "head_internal", "behavior_side"]


@dataclass(frozen=True)
class FlaggedAnomaly:
    """One flagged anomaly attached to a per-episode mini-digest.

    Each anomaly names the reading surface it sits on so the readers'
    prompts (Phase 8) and the Judge's per-surface rulings (Phase 9) can
    cite the cohort the anomaly belongs to without re-deriving the
    surface from the description's prose.
    """

    kind: str
    reading_surface: ReadingSurface
    episode_id: int | None
    description: str


@dataclass(frozen=True)
class HierarchicalDigest:
    """Probe 2's three-cohort digest output (plan §2.3).

    ``run_summary`` is the run-level header (schema version, run_id,
    episode count, env-step range, wallclock span). ``episode_mini_digests``
    maps episode_id to its three-cohort prose block (substrate-side,
    head-internal, behavior-side). ``flagged_anomalies`` is the
    cohort-aware list of anomalies the readers may want to drill into.
    ``world_event_timeline`` carries env_reset, internal-stochasticity,
    builder-perturbation, sham-perturbation, and mirror_marker events in
    order. ``drill_down`` exposes the per-window / per-dream /
    per-conditioning fetchers Phase 8's reader uses for follow-up.
    """

    run_summary: str
    episode_mini_digests: dict[int, str]
    flagged_anomalies: list[FlaggedAnomaly]
    world_event_timeline: str
    drill_down: "DrillDownAccessor"


class DrillDownAccessor:
    """Window / dream / world-event / self-prediction / conditioning fetcher.

    Holds a reference to the loaded telemetry rows so the digest's prose
    surface stays compact while specific windows can be expanded on
    demand. The ``fetch_*`` methods return ready-to-paste prose; the
    Phase 8 reader's prompt cites a window by ``(episode_id,
    step_range)`` and gets back a string it can quote in a structured
    claim's evidence ground.

    The accessor never mutates its inputs and never reads from disk
    after construction — :func:`build_hierarchical_digest` does the
    parquet / JSONL I/O once and hands the rows here. The behavior-side
    drill-down is the one exception: ``fetch_conditioning`` reads the
    pre-loaded :class:`ConditioningResult` records that the builder
    passed in via ``conditioning_dir``; if no conditioning records were
    available, the method returns the graceful-degradation line.
    """

    def __init__(
        self,
        agent_step_rows: list[dict[str, Any]],
        dream_rollout_rows: list[dict[str, Any]],
        world_events: list[dict[str, Any]],
        conditioning_records: list[dict[str, Any]] | None,
    ) -> None:
        self._rows_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in agent_step_rows:
            self._rows_by_episode[int(r["episode_id"])].append(r)
        for ep in self._rows_by_episode:
            self._rows_by_episode[ep].sort(
                key=lambda r: int(r["step_in_episode"])
            )
        self._dream_rows = list(dream_rollout_rows)
        self._world_events = list(world_events)
        self._conditioning_records = (
            list(conditioning_records) if conditioning_records is not None else None
        )

    def fetch_window(
        self, episode_id: int, step_range: tuple[int, int]
    ) -> str:
        """Compact-repr lines for the steps in ``[lo, hi]`` (inclusive)."""
        lo, hi = step_range
        ep_rows = self._rows_by_episode.get(int(episode_id), [])
        if not ep_rows:
            return f"(no agent_step rows for episode {episode_id})"
        selected = [
            r for r in ep_rows if lo <= int(r["step_in_episode"]) <= hi
        ]
        if not selected:
            return (
                f"(no agent_step rows for episode {episode_id} "
                f"in step range [{lo}, {hi}])"
            )
        out: list[str] = [
            f"# drill-down: episode {episode_id}, steps [{lo}, {hi}] "
            f"({len(selected)} records)"
        ]
        for r in selected:
            out.append(
                "  - "
                + compact_record_repr(
                    r, position=f"step={int(r['step_in_episode'])}"
                )
            )
        return "\n".join(out)

    def fetch_dream(self, dream_index: int) -> str:
        """Scalar summary of one dream rollout (no decoded ASCII art)."""
        if dream_index < 0 or dream_index >= len(self._dream_rows):
            return (
                f"(no dream_rollout at index {dream_index}; "
                f"available: {len(self._dream_rows)})"
            )
        r = self._dream_rows[dream_index]
        seq_h = r.get("sequence_h") or []
        seq_action = r.get("sequence_action") or []
        return (
            f"# drill-down: dream_rollout index {dream_index}\n"
            f"  - seed_step: {r.get('seed_step')}\n"
            f"  - n_steps: {len(seq_h)}\n"
            f"  - n_actions: {len(seq_action)}\n"
            f"  - cumulative_prior_entropy: "
            f"{_format_scalar(r.get('cumulative_prior_entropy'))}\n"
            f"  - mean_step_kl_successive_priors: "
            f"{_format_scalar(r.get('mean_step_kl_successive_priors'))}\n"
            f"  - max_step_latent_norm_change: "
            f"{_format_scalar(r.get('max_step_latent_norm_change'))}"
        )

    def fetch_world_event(self, event_index: int) -> str:
        """One world_event record's prose summary."""
        if event_index < 0 or event_index >= len(self._world_events):
            return (
                f"(no world_event at index {event_index}; "
                f"available: {len(self._world_events)})"
            )
        return _format_world_event(self._world_events[event_index], event_index)

    def fetch_self_prediction(
        self, episode_id: int, step_range: tuple[int, int]
    ) -> str:
        """Per-step ``self_prediction_error_t`` with the masked flag.

        Probe 1.5 v2 §10 item 1 names the masked-flag visibility on the
        drill-down window as a load-bearing affordance: aggregations
        elsewhere may drop masked steps for the empirical mean, but the
        drill-down has to surface them so the reader can see why a given
        step's scalar is sentinel zero rather than an empirical reading.
        Probe 1 records (no self-prediction fields) produce the
        graceful-degradation line.
        """
        lo, hi = step_range
        ep_rows = self._rows_by_episode.get(int(episode_id), [])
        if not ep_rows:
            return f"(no agent_step rows for episode {episode_id})"
        selected = [
            r
            for r in ep_rows
            if lo <= int(r["step_in_episode"]) <= hi
            and str(r.get("schema_version", ""))
            == _SELF_PREDICTION_SCHEMA_VERSION
            and r.get("self_prediction_error_t") is not None
        ]
        if not selected:
            # Distinguish "no rows in window" from "no 0.2.0 self-prediction
            # data" — the drill-down's contract per Probe 1.5 v2 §10 item 1
            # is that masked-step visibility is the load-bearing affordance,
            # so a Probe 1 record produces the degraded-line shape rather
            # than an empty window.
            any_in_window = any(
                lo <= int(r["step_in_episode"]) <= hi for r in ep_rows
            )
            if not any_in_window:
                return (
                    f"(no agent_step rows for episode {episode_id} "
                    f"in step range [{lo}, {hi}])"
                )
            return _PROBE_1_DEGRADED_MESSAGE
        out: list[str] = [
            f"# drill-down: self_prediction episode {episode_id}, "
            f"steps [{lo}, {hi}] ({len(selected)} records)"
        ]
        for r in selected:
            sp_err = float(r["self_prediction_error_t"])
            sp_err_masked = bool(r.get("self_prediction_error_masked_t"))
            out.append(
                f"  - step={int(r['step_in_episode'])}: "
                f"self_prediction_error_t={sp_err:.6f}, "
                f"self_prediction_error_masked_t={sp_err_masked}"
            )
        return "\n".join(out)

    def fetch_conditioning(self, regime: str, perturbation: str) -> str:
        """Per-(regime, perturbation) bucket lookup across loaded records.

        Returns the graceful-degradation line if no ``conditioning_dir``
        was supplied at digest construction. With records present, the
        bucket(s) matching the requested (regime, perturbation) pair are
        rendered. If the records carry multiple checkpoints the pairs
        are listed per checkpoint so the reader can see how the
        conditioning evolved.
        """
        if self._conditioning_records is None:
            return _NO_CONDITIONING_DEGRADED_MESSAGE
        if not self._conditioning_records:
            return (
                f"(no conditioning records found for regime={regime!r}, "
                f"perturbation={perturbation!r}; conditioning_dir was "
                f"supplied but the JSONL was empty)"
            )
        out: list[str] = [
            f"# drill-down: conditioning regime={regime}, "
            f"perturbation={perturbation}"
        ]
        any_match = False
        for cond in self._conditioning_records:
            ckpt = cond.get("checkpoint_id", "?")
            buckets = cond.get("per_regime_per_perturbation", []) or []
            for b in buckets:
                if (
                    b.get("regime") == regime
                    and b.get("perturbation") == perturbation
                ):
                    stats = b.get("stats", {}) or {}
                    out.append(
                        f"  - checkpoint={ckpt}: "
                        f"n_states={stats.get('n_states', '?')}, "
                        f"KL_mean={_format_scalar(stats.get('kl_mean'))}, "
                        f"KL_std={_format_scalar(stats.get('kl_std'))}, "
                        f"KL_p50={_format_scalar(stats.get('kl_p50'))}, "
                        f"KL_p90={_format_scalar(stats.get('kl_p90'))}"
                    )
                    any_match = True
        if not any_match:
            return (
                f"(no conditioning bucket matched regime={regime!r}, "
                f"perturbation={perturbation!r})"
            )
        return "\n".join(out)


def build_hierarchical_digest(
    telemetry_dir: Path,
    *,
    n_episodes: int,
    flagged_only: bool = False,
    with_sham: bool = True,
    conditioning_dir: Path | None = None,
    behavior_anomaly_z_threshold: float = _BEHAVIOR_KL_ANOMALY_DEFAULT_Z,
) -> HierarchicalDigest:
    """Build the three-cohort hierarchical digest for a run's telemetry.

    Reads the parquet shards under ``telemetry_dir/agent_step/``, the
    JSONL streams ``telemetry_dir/world_event.jsonl`` and (if present)
    ``telemetry_dir/replay_meta.jsonl``, the parquet shards under
    ``telemetry_dir/dream_rollout/``, and — if ``conditioning_dir`` is
    supplied — the conditioning analysis module's output cache at
    ``conditioning_dir/conditioning.jsonl``. Selects the most recent
    ``n_episodes`` episodes' rows for the per-episode mini-digest.

    Per the plan §2.3 layout, each per-episode mini-digest carries three
    labeled cohorts (``[substrate-side]``, ``[head-internal]``,
    ``[behavior-side]``). Probe 1 (``"0.1.0"``) records produce a
    head-internal and behavior-side cohort replaced by a single
    graceful-degradation line. Conditioning-dir omitted: behavior-side
    cohort replaced by a single graceful-degradation line; substrate-
    side and head-internal cohorts unchanged.

    Args:
        telemetry_dir: Run telemetry root (the dir holding ``agent_step/``
            etc., not the run dir).
        n_episodes: How many of the most recent episodes to include in
            ``episode_mini_digests``. Plan §6 default is 25; the caller
            chooses.
        flagged_only: If True, ``episode_mini_digests`` is filtered to
            episodes with at least one flagged anomaly.
        with_sham: If True, sham builder-perturbation events are
            surfaced with a ``[SHAM]`` prefix in the world-event
            timeline. If False, sham events are filtered out (the flag
            is present so the smoke can build a non-sham comparison
            digest from the same telemetry).
        conditioning_dir: If supplied, the conditioning analysis
            module's output cache directory. The digest reads
            ``conditioning_dir/conditioning.jsonl`` (one record per
            invocation; the digest aggregates all records).
        behavior_anomaly_z_threshold: Per-bucket KL_p90 z-score
            threshold above which the bucket is flagged as a
            behavior-side anomaly. Default 3.0 per the plan §2.3 spec;
            tunable.

    Returns:
        :class:`HierarchicalDigest` with the four prose surfaces and
        the drill-down accessor.
    """
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}")
    if behavior_anomaly_z_threshold < 0:
        raise ValueError(
            f"behavior_anomaly_z_threshold must be non-negative, got "
            f"{behavior_anomaly_z_threshold}"
        )

    agent_rows = _read_parquet_dir(telemetry_dir / "agent_step")
    dream_rows = _read_parquet_dir(telemetry_dir / "dream_rollout")
    world_events = _read_jsonl(telemetry_dir / "world_event.jsonl")
    conditioning_records = (
        _read_jsonl(conditioning_dir / "conditioning.jsonl")
        if conditioning_dir is not None
        else None
    )

    by_ep: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in agent_rows:
        by_ep[int(r["episode_id"])].append(r)
    for ep in by_ep:
        by_ep[ep].sort(key=lambda r: int(r["step_in_episode"]))

    # Select the most recent ``n_episodes`` (by episode_id ordering).
    sorted_episode_ids = sorted(by_ep.keys())
    selected_episode_ids = sorted_episode_ids[-n_episodes:]

    # Pre-compute run-level statistics needed by per-episode anomaly
    # detection (per-episode ``kl_per_dim_t`` allocation top-5 shifts;
    # ``self_prediction_t`` per-dim allocation top-5 shifts).
    prev_kl_top: list[int] | None = None
    prev_sp_top: list[int] | None = None

    flagged: list[FlaggedAnomaly] = []
    mini_digests: dict[int, str] = {}

    # Detect ensemble-disagreement collapse using an absolute
    # monomorphic-regime threshold: an episode whose mean
    # ``intrinsic_signal_t`` falls below
    # ``_ENSEMBLE_COLLAPSE_THRESHOLD`` is flagged. Phase 7's u=200
    # monomorphic regime fired with episode-level intrinsic_signal_t
    # dropping into a near-zero band; Phase 7.5's healthy runs sit
    # comfortably above the threshold and do not trip. The choice of
    # an absolute rather than a relative threshold is deliberate — a
    # relative bottom-percentile cut always flags the lowest episode in
    # a healthy run, which is noise.
    for ep_id in selected_episode_ids:
        ep_rows = by_ep[ep_id]
        intrinsic_vals = [float(r["intrinsic_signal_t"]) for r in ep_rows]
        if not intrinsic_vals:
            continue
        m = _mean(intrinsic_vals)
        if m < _ENSEMBLE_COLLAPSE_THRESHOLD:
            flagged.append(
                FlaggedAnomaly(
                    kind="ensemble_disagreement_collapse",
                    reading_surface="substrate_side",
                    episode_id=ep_id,
                    description=(
                        f"[substrate-side] episode {ep_id}: mean "
                        f"intrinsic_signal_t={m:.6e} below the "
                        f"monomorphic-regime threshold "
                        f"({_ENSEMBLE_COLLAPSE_THRESHOLD:.0e})"
                    ),
                )
            )

    # Behavior-side anomaly detection from conditioning records (Phase
    # 6's output): flag (regime, perturbation) buckets whose KL_p90 is
    # more than ``behavior_anomaly_z_threshold`` standard deviations
    # from the run's empirical mean. Computed over all loaded records'
    # buckets, so a multi-checkpoint conditioning run lets the
    # threshold see the natural KL_p90 dispersion before flagging.
    if conditioning_records:
        kl_p90_vals: list[float] = []
        for cond in conditioning_records:
            for b in cond.get("per_regime_per_perturbation", []) or []:
                stats = b.get("stats", {}) or {}
                if "kl_p90" in stats:
                    kl_p90_vals.append(float(stats["kl_p90"]))
        if len(kl_p90_vals) >= 2:
            mu = _mean(kl_p90_vals)
            sigma = _std(kl_p90_vals, mu)
            if sigma > 0:
                for cond in conditioning_records:
                    ckpt = cond.get("checkpoint_id", "?")
                    for b in cond.get("per_regime_per_perturbation", []) or []:
                        stats = b.get("stats", {}) or {}
                        if "kl_p90" not in stats:
                            continue
                        v = float(stats["kl_p90"])
                        z = (v - mu) / sigma
                        if abs(z) > behavior_anomaly_z_threshold:
                            flagged.append(
                                FlaggedAnomaly(
                                    kind="conditioning_kl_p90_outlier",
                                    reading_surface="behavior_side",
                                    episode_id=None,
                                    description=(
                                        f"checkpoint={ckpt}, "
                                        f"regime={b.get('regime')}, "
                                        f"perturbation={b.get('perturbation')}: "
                                        f"KL_p90={v:.4e} (z={z:+.2f})"
                                    ),
                                )
                            )

    for ep_id in selected_episode_ids:
        ep_rows = by_ep[ep_id]
        block, ep_anomalies, kl_top, sp_top = _build_episode_mini_digest(
            ep_id,
            ep_rows,
            conditioning_records=conditioning_records,
            prev_kl_top=prev_kl_top,
            prev_sp_top=prev_sp_top,
        )
        mini_digests[ep_id] = block
        flagged.extend(ep_anomalies)
        prev_kl_top = kl_top
        prev_sp_top = sp_top

    if flagged_only:
        flagged_episodes = {a.episode_id for a in flagged if a.episode_id is not None}
        mini_digests = {
            ep_id: text
            for ep_id, text in mini_digests.items()
            if ep_id in flagged_episodes
        }

    run_summary = _build_run_summary(
        agent_rows,
        len(selected_episode_ids),
        len(world_events),
    )
    timeline = _build_world_event_timeline(world_events, with_sham=with_sham)
    drill_down = DrillDownAccessor(
        agent_step_rows=agent_rows,
        dream_rollout_rows=dream_rows,
        world_events=world_events,
        conditioning_records=conditioning_records,
    )

    return HierarchicalDigest(
        run_summary=run_summary,
        episode_mini_digests=mini_digests,
        flagged_anomalies=flagged,
        world_event_timeline=timeline,
        drill_down=drill_down,
    )


def _build_run_summary(
    rows: list[dict[str, Any]],
    n_episodes_selected: int,
    n_world_events: int,
) -> str:
    if not rows:
        return "# Telemetry Run Summary\n\n(no agent_step records)"
    schema_version = str(rows[0].get("schema_version", "?"))
    run_id = str(rows[0].get("run_id", "?"))
    t_first = int(rows[0]["t"])
    t_last = int(rows[-1]["t"])
    wc_first = int(rows[0]["wallclock_ms"])
    wc_last = int(rows[-1]["wallclock_ms"])
    n_episodes_total = len({int(r["episode_id"]) for r in rows})
    lines = [
        "# Telemetry Run Summary",
        "",
        f"- schema_version: {schema_version}",
        f"- run_id: {run_id}",
        f"- episodes (total in run): {n_episodes_total}",
        f"- episodes (selected for mini-digests): {n_episodes_selected}",
        f"- agent_step records (total in run): {len(rows)}",
        f"- env_step range: [{t_first}, {t_last}]",
        f"- wallclock span: {wc_last - wc_first} ms ({wc_first} -> {wc_last})",
        f"- world_event records: {n_world_events}",
    ]
    return "\n".join(lines)


def _build_episode_mini_digest(
    ep_id: int,
    ep_rows: list[dict[str, Any]],
    *,
    conditioning_records: list[dict[str, Any]] | None,
    prev_kl_top: list[int] | None,
    prev_sp_top: list[int] | None,
) -> tuple[str, list[FlaggedAnomaly], list[int] | None, list[int] | None]:
    """Compose one episode's three-cohort prose plus the flagged anomalies.

    Returns ``(block_text, anomalies, kl_top, sp_top)`` where ``kl_top``
    and ``sp_top`` are the per-dim allocation top-5 lists this episode
    produced (used to detect allocation shifts when the next episode is
    composed). Either may be ``None`` if the episode has no allocation
    available (Probe 1 records skip ``sp_top``).
    """
    anomalies: list[FlaggedAnomaly] = []
    lines: list[str] = [f"[episode {ep_id}]"]

    # ---- substrate-side cohort ----------------------------------
    lines.append("  [substrate-side]")
    kl_vals = [float(r["kl_aggregate_t"]) for r in ep_rows]
    intrinsic_vals = [float(r["intrinsic_signal_t"]) for r in ep_rows]
    kl_mean = _mean(kl_vals)
    kl_std = _std(kl_vals, kl_mean)
    kl_p90 = _percentile(kl_vals, 90.0)
    intrinsic_mean = _mean(intrinsic_vals)
    intrinsic_std = _std(intrinsic_vals, intrinsic_mean)
    lines.append(
        f"    kl_aggregate_t: mean={kl_mean:.4f}, std={kl_std:.4f}, "
        f"p90={kl_p90:.4f}"
    )
    kl_top = _per_dim_top_indices(
        [r.get("kl_per_dim_t") for r in ep_rows], _HIERARCHICAL_TOP_DIMS
    )
    if kl_top is not None:
        per_dim_var = _per_dim_variance(
            [r.get("kl_per_dim_t") for r in ep_rows]
        )
        if per_dim_var is not None:
            top_strs = [
                f"dim={d} (var={per_dim_var[d]:.4f})" for d in kl_top
            ]
            lines.append(
                f"    per-dim KL allocation top-{len(kl_top)}: "
                + ", ".join(top_strs)
            )
        else:
            lines.append(
                "    per-dim KL allocation top-5: (no kl_per_dim_t)"
            )
    else:
        lines.append("    per-dim KL allocation top-5: (no kl_per_dim_t)")
    intrinsic_regime = _classify_intrinsic_regime(intrinsic_mean, intrinsic_std)
    lines.append(
        f"    ensemble_disagreement: mean={intrinsic_mean:.4f}, "
        f"regime={intrinsic_regime}"
    )

    # Substrate-side anomaly: kl_aggregate_t outliers in episode.
    if kl_std > 0:
        for r in ep_rows:
            v = float(r["kl_aggregate_t"])
            z = (v - kl_mean) / kl_std
            if abs(z) > _KL_OUTLIER_Z_THRESHOLD:
                anomalies.append(
                    FlaggedAnomaly(
                        kind="kl_aggregate_outlier",
                        reading_surface="substrate_side",
                        episode_id=ep_id,
                        description=(
                            f"[substrate-side] episode {ep_id} "
                            f"step={int(r['step_in_episode'])}: "
                            f"kl_aggregate_t={v:.4f} (z={z:+.2f})"
                        ),
                    )
                )

    # Substrate-side anomaly: top-1 recon outlier (carries from
    # ``build_digest``'s convention; per-episode minimum-N = 5).
    if len(ep_rows) >= _RECON_OUTLIER_MIN_STEPS:
        worst = max(ep_rows, key=lambda r: float(r["recon_loss_t"]))
        anomalies.append(
            FlaggedAnomaly(
                kind="recon_outlier",
                reading_surface="substrate_side",
                episode_id=ep_id,
                description=(
                    f"[substrate-side] episode {ep_id} "
                    f"step={int(worst['step_in_episode'])}: "
                    f"recon_loss_t={float(worst['recon_loss_t']):.4f} "
                    f"(top recon)"
                ),
            )
        )

    # Substrate-side anomaly: per-dim KL allocation top-5 shift.
    if (
        prev_kl_top is not None
        and kl_top is not None
        and len(set(prev_kl_top) & set(kl_top)) == 0
    ):
        anomalies.append(
            FlaggedAnomaly(
                kind="kl_allocation_shift",
                reading_surface="substrate_side",
                episode_id=ep_id,
                description=(
                    f"[substrate-side] episode {ep_id}: per-dim KL "
                    f"top-5 set fully reallocated (prev={prev_kl_top}, "
                    f"current={kl_top}; intersection=0)"
                ),
            )
        )

    # ---- head-internal cohort -----------------------------------
    lines.append("  [head-internal]")
    rows_with_scalar = [
        r
        for r in ep_rows
        if str(r.get("schema_version", "")) == _SELF_PREDICTION_SCHEMA_VERSION
        and r.get("self_prediction_error_t") is not None
    ]
    sp_top: list[int] | None = None
    if not rows_with_scalar:
        lines.append(f"    {_PROBE_1_DEGRADED_MESSAGE}")
    else:
        masked_count = sum(
            1
            for r in rows_with_scalar
            if r.get("self_prediction_error_masked_t") is True
        )
        non_masked = [
            r
            for r in rows_with_scalar
            if r.get("self_prediction_error_masked_t") is not True
        ]
        if not non_masked:
            lines.append(
                f"    self_prediction_error_t (excluding masked): "
                f"(every step in episode is masked)"
            )
            lines.append(f"    masked steps in episode: {masked_count}")
        else:
            sp_vals = [float(r["self_prediction_error_t"]) for r in non_masked]
            sp_mean = _mean(sp_vals)
            sp_std = _std(sp_vals, sp_mean)
            sp_p90 = _percentile(sp_vals, 90.0)
            lines.append(
                f"    self_prediction_error_t (excluding masked): "
                f"mean={sp_mean:.6f}, std={sp_std:.6f}, p90={sp_p90:.6f}"
            )
            outlier_lines: list[str] = []
            if sp_std > 0:
                for r in non_masked:
                    v = float(r["self_prediction_error_t"])
                    z = (v - sp_mean) / sp_std
                    if abs(z) > _KL_OUTLIER_Z_THRESHOLD:
                        outlier_lines.append(
                            f"step={int(r['step_in_episode'])} "
                            f"(sp_err={v:.6f}, z={z:+.2f})"
                        )
                        anomalies.append(
                            FlaggedAnomaly(
                                kind="self_prediction_outlier",
                                reading_surface="head_internal",
                                episode_id=ep_id,
                                description=(
                                    f"[head-internal] episode {ep_id} "
                                    f"step={int(r['step_in_episode'])}: "
                                    f"self_prediction_error_t={v:.6f} "
                                    f"(z={z:+.2f})"
                                ),
                            )
                        )
            if outlier_lines:
                lines.append(
                    "    self_prediction outliers (z>3): "
                    + ", ".join(outlier_lines)
                )
            else:
                lines.append("    self_prediction outliers (z>3): none")
            sp_top = _per_dim_self_prediction_top_indices(
                rows_with_scalar, _HIERARCHICAL_TOP_DIMS
            )
            if sp_top is not None:
                per_dim_var = _per_dim_self_prediction_error_variance(
                    rows_with_scalar
                )
                top_strs = [
                    f"dim={d} (var={per_dim_var[d]:.6f})" for d in sp_top
                ]
                lines.append(
                    f"    self_prediction allocation top-{len(sp_top)} "
                    f"dims: " + ", ".join(top_strs)
                )
            else:
                lines.append(
                    "    self_prediction allocation top-5 dims: (no valid pairs)"
                )
            lines.append(f"    masked steps in episode: {masked_count}")

            # Head-internal anomaly: self_prediction allocation top-5
            # shift (Probe 1.5 v2 §10 item 7; Phase 7 mirror surfaced
            # this organically).
            if (
                prev_sp_top is not None
                and sp_top is not None
                and len(set(prev_sp_top) & set(sp_top)) == 0
            ):
                anomalies.append(
                    FlaggedAnomaly(
                        kind="self_prediction_allocation_shift",
                        reading_surface="head_internal",
                        episode_id=ep_id,
                        description=(
                            f"[head-internal] episode {ep_id}: "
                            f"self_prediction top-5 set fully reallocated "
                            f"(prev={prev_sp_top}, current={sp_top}; "
                            f"intersection=0)"
                        ),
                    )
                )

    # ---- behavior-side cohort -----------------------------------
    lines.append("  [behavior-side]")
    if not rows_with_scalar:
        # Probe 1 records: behavior-side cohort degraded the same way
        # as head-internal (per the spec — graceful degradation when
        # the records are 0.1.0).
        lines.append(f"    {_PROBE_1_DEGRADED_MESSAGE}")
    elif conditioning_records is None:
        lines.append(f"    {_NO_CONDITIONING_DEGRADED_MESSAGE}")
    else:
        # Aggregate per-regime stats across all loaded conditioning
        # records (typically one record per checkpoint). The per-regime
        # mini-summary lines aggregate across perturbation
        # distributions: n_states sums, KL_mean / KL_p90 are weighted
        # by n_states. The drill-down accessor exposes the full
        # per-(regime, perturbation) shape.
        regime_summary = _aggregate_conditioning_by_regime(conditioning_records)
        for regime in _DEFAULT_BEHAVIOR_REGIMES:
            agg = regime_summary.get(regime)
            if agg is None:
                lines.append(
                    f"    {regime}: n_states=0, KL_mean=0.0, KL_p90=0.0"
                )
            else:
                lines.append(
                    f"    {regime}: n_states={agg.n_states}, "
                    f"KL_mean={_format_scalar(agg.kl_mean)}, "
                    f"KL_p90={_format_scalar(agg.kl_p90)}"
                )

    return "\n".join(lines), anomalies, kl_top, sp_top


def _build_world_event_timeline(
    events: list[dict[str, Any]], *, with_sham: bool
) -> str:
    if not events:
        return "# World Event Timeline\n\n(no world_event records)"
    out = ["# World Event Timeline", ""]
    for ix, ev in enumerate(events):
        is_sham = bool((ev.get("payload") or {}).get("is_sham"))
        if is_sham and not with_sham:
            continue
        out.append(_format_world_event(ev, ix))
    return "\n".join(out)


def _format_world_event(ev: dict[str, Any], event_index: int) -> str:
    payload = ev.get("payload") or {}
    is_sham = bool(payload.get("is_sham"))
    sham_label = payload.get("sham_label")
    prefix = "[SHAM] " if is_sham else ""
    sham_suffix = f" sham_label={sham_label!r}" if is_sham and sham_label else ""
    extras: list[str] = []
    if "episode_id" in payload:
        extras.append(f"episode_id={payload['episode_id']}")
    if "start_cell" in payload:
        extras.append(f"start_cell={payload['start_cell']}")
    if "lesion_kind" in payload:
        extras.append(f"lesion_kind={payload['lesion_kind']!r}")
    if "source_checkpoint" in payload:
        extras.append(f"source_checkpoint={payload['source_checkpoint']!r}")
    extras_text = (" | " + ", ".join(extras)) if extras else ""
    return (
        f"  - [{event_index:04d}] t_event={ev.get('t_event')} "
        f"{prefix}{ev.get('event_type', '?')} "
        f"(source={ev.get('source', '?')}){sham_suffix}{extras_text}"
    )


# ---------------------------------------------------------------------------
# Helpers for hierarchical digest.
# ---------------------------------------------------------------------------


def _read_parquet_dir(dir_path: Path) -> list[dict[str, Any]]:
    if not dir_path.is_dir():
        return []
    shards = sorted(dir_path.glob("shard-*.parquet"))
    if not shards:
        return []
    rows: list[dict[str, Any]] = []
    for shard in shards:
        table = pq.read_table(str(shard))  # type: ignore[no-untyped-call]
        rows.extend(table.to_pylist())
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # Tolerate trailing-garbage lines rather than failing
                # the whole digest build over one malformed record.
                continue
    return out


def _percentile(vals: list[float], pct: float) -> float:
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    sorted_vals = sorted(vals)
    if pct <= 0:
        return sorted_vals[0]
    if pct >= 100:
        return sorted_vals[-1]
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_vals[lo]
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _per_dim_variance(
    per_step_vectors: list[Any],
) -> list[float] | None:
    """Per-dim sample variance across one episode's per-step vectors.

    Returns ``None`` if no usable vectors exist; otherwise a list of
    length ``dim`` with sample variance per dim. ``per_step_vectors``
    may contain ``None`` entries (Probe 1 records' missing
    ``kl_per_dim_t`` shouldn't exist, but the helper tolerates).
    """
    cleaned: list[list[float]] = []
    for v in per_step_vectors:
        if v is None:
            continue
        if not isinstance(v, list):
            continue
        cleaned.append([float(x) for x in v])
    if not cleaned:
        return None
    dim = len(cleaned[0])
    if dim == 0:
        return None
    out: list[float] = []
    for d in range(dim):
        col = [row[d] for row in cleaned if len(row) > d]
        if len(col) < 2:
            out.append(0.0)
            continue
        m = sum(col) / len(col)
        var = sum((x - m) ** 2 for x in col) / (len(col) - 1)
        out.append(var)
    return out


def _per_dim_top_indices(
    per_step_vectors: list[Any], k: int
) -> list[int] | None:
    per_dim_var = _per_dim_variance(per_step_vectors)
    if per_dim_var is None:
        return None
    ranked = sorted(
        range(len(per_dim_var)), key=lambda d: per_dim_var[d], reverse=True
    )
    return ranked[: min(k, len(ranked))]


def _per_dim_self_prediction_top_indices(
    rows: list[dict[str, Any]], k: int
) -> list[int] | None:
    per_dim_var = _per_dim_self_prediction_error_variance(rows)
    if not per_dim_var:
        return None
    ranked = sorted(
        range(len(per_dim_var)), key=lambda d: per_dim_var[d], reverse=True
    )
    return ranked[: min(k, len(ranked))]


def _classify_intrinsic_regime(mean: float, std: float) -> str:
    """Coarse per-episode regime label for the intrinsic_signal_t mean.

    Three labels: ``high_disagreement`` if mean > 0.05 (the empirical
    high-variance band Phase 7 / 7.5 telemetry produces),
    ``low_disagreement`` if mean < 1e-4 (the monomorphic-regime band
    Phase 7 saw under u=200 collapse), ``mid_disagreement`` otherwise.
    The thresholds are coarse on purpose: the readers' prompt frames
    the regime call against the cohort's other content, not against a
    precise threshold the digest pretends to know.
    """
    if mean > 0.05:
        return "high_disagreement"
    if mean < 1e-4:
        return "low_disagreement"
    return "mid_disagreement"


@dataclass(frozen=True)
class _RegimeAgg:
    n_states: int
    kl_mean: float
    kl_p90: float


def _aggregate_conditioning_by_regime(
    records: list[dict[str, Any]],
) -> dict[str, _RegimeAgg]:
    """Aggregate (regime, perturbation) buckets to per-regime summaries.

    Returns a dict keyed by regime name; each value is a
    :class:`_RegimeAgg` carrying the n_states-weighted KL_mean / KL_p90
    pooled across perturbation distributions and across loaded
    conditioning records (typically one per checkpoint). The
    drill-down accessor's :meth:`fetch_conditioning` exposes the full
    per-(regime, perturbation) detail; this aggregator is for the
    per-episode mini-digest's single-line-per-regime cohort.
    """
    raw: dict[str, list[float]] = defaultdict(list)
    raw_p90: dict[str, list[float]] = defaultdict(list)
    raw_n: dict[str, int] = defaultdict(int)
    for rec in records:
        for b in rec.get("per_regime_per_perturbation", []) or []:
            regime = b.get("regime")
            stats = b.get("stats", {}) or {}
            if not isinstance(regime, str):
                continue
            n = int(stats.get("n_states", 0) or 0)
            if n <= 0:
                # Empty buckets are skipped from the aggregate; the
                # per-episode mini-digest's "regime: n_states=0" line
                # is produced by the regime-not-in-out path in
                # _build_episode_mini_digest, not by aggregating empty
                # buckets here.
                continue
            raw_n[regime] += n
            raw[regime].append(float(stats.get("kl_mean", 0.0) or 0.0) * n)
            raw_p90[regime].append(
                float(stats.get("kl_p90", 0.0) or 0.0) * n
            )
    out: dict[str, _RegimeAgg] = {}
    for regime, n in raw_n.items():
        if n <= 0:
            continue
        kl_mean = sum(raw[regime]) / n
        kl_p90 = sum(raw_p90[regime]) / n
        out[regime] = _RegimeAgg(n_states=n, kl_mean=kl_mean, kl_p90=kl_p90)
    return out


def _format_scalar(v: Any) -> str:
    if v is None:
        return "?"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        f = float(v)
        if not math.isfinite(f):
            return str(f)
        if f == 0.0:
            return "0.0"
        if abs(f) < 1e-3 or abs(f) >= 1e6:
            return f"{f:.3e}"
        return f"{f:.4f}"
    return str(v)
