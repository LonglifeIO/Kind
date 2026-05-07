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
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

__all__ = [
    "build_digest",
    "compact_record_repr",
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
