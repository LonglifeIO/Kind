"""Phase 7 eyeball helpers — human-side reading of telemetry.

The plan §2.11 names this module as the small set of CLI scripts that
let the human builder read what the runner has written. Importable as
``from kind.observer import eyeball``; runnable as
``python -m kind.observer.eyeball <command> [args]``. The plan's
discipline applies: no visualisation beyond plain text and (for dream
observations) ASCII downsampling. No dashboards, no plots, no rich
formatting libraries.

What the helpers expose:

* :func:`show_recent_agent_steps` — pretty-print the last N records.
* :func:`count_world_events` — count records by event_type.
* :func:`show_episode_summary` — summary of one episode.
* :func:`show_dream_rollout` — one dream rollout, optionally with ASCII art.
* :func:`show_run_summary` — high-level summary of the whole run.

Pretty-printing principles (per the user brief and the design notes'
"observer layer (me)" framing):

* Plain text only — no ANSI, no colour, no third-party formatters. Output
  goes to ``stdout`` for ``tail -f`` and copy-paste into the journal.
* Floats round to 4 decimal places; very small / very large slide into
  scientific notation.
* Action distribution shown as ``up=N down=N left=N right=N stay=N``
  rather than as a probability vector — counts are more legible.
* High-dimensional vectors (when explicitly requested via ``fields=``) are
  shown by shape plus mean/std/min/max plus first 3 and last 3 elements.
  By default they are excluded — the same scalar-only discipline the
  digest applies for the LLM mirror.
* Dream rollouts' decoded observations render as 16×16 ASCII (downsampled
  from 32×32) with characters chosen by intensity. The mirror of the
  builder's eye seeing roughly what Io was imagining.

The discipline against confirming prior beliefs (design notes, "Observer
layer (me)") is preserved by keeping these helpers strictly observational.
Nothing here makes a normative claim about whether the data is good or
bad; the helpers print structure, the human reads it, the journal records
what surfaced.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq
import torch

__all__ = [
    "show_recent_agent_steps",
    "count_world_events",
    "show_episode_summary",
    "show_dream_rollout",
    "show_run_summary",
    "show_self_prediction",
    "show_self_prediction_conditioning",
]


# ---- canonical telemetry layout (mirror of kind.training.runner) ----------


_AGENT_STEP_DIR = "agent_step"
_DREAM_ROLLOUT_DIR = "dream_rollout"
_REPLAY_META_FILE = "replay_meta.jsonl"
_WORLD_EVENT_FILE = "world_event.jsonl"


# Action labels follow ``kind/env/grid_world.py``'s ``_ACTION_DELTAS``:
# 0=up, 1=down, 2=left, 3=right, 4=stay. Kept as a tuple so the iteration
# order is part of the contract (changing it would shuffle distribution
# strings in journal entries that copy-paste this output).
_ACTION_LABELS: tuple[str, ...] = ("up", "down", "left", "right", "stay")


# ASCII intensity ramp for dream-rollout decoded observations. Ten
# characters from low to high luminance. The 32×32 grayscale tensor is
# downsampled 2× to 16×16 (one character per source 2×2 block) before
# being mapped through this ramp; legibility comes from the downsample,
# not the ramp width.
_ASCII_RAMP: str = " .:-=+*#%@"


# Default scalar fields shown by ``show_recent_agent_steps`` when no
# explicit ``fields=`` is given. Excludes the high-dim vectors (h_t, z_t,
# q_params_t, p_params_t, kl_per_dim_t, encoder_embedding_t) — the
# scalar-only discipline the digest applies for the mirror also fits a
# human eyeballing parquet rows.
_DEFAULT_SCALAR_FIELDS: tuple[str, ...] = (
    "t",
    "episode_id",
    "step_in_episode",
    "wallclock_ms",
    "kl_aggregate_t",
    "recon_loss_t",
    "intrinsic_signal_t",
    "policy_entropy_t",
    "action_t",
    "action_logprob_t",
    "obs_hash_t",
    "checkpoint_id",
)


_HIGH_DIM_FIELDS: frozenset[str] = frozenset(
    {
        "h_t",
        "z_t",
        "q_params_t",
        "p_params_t",
        "kl_per_dim_t",
        "encoder_embedding_t",
    }
)


# ---- public API: agent_step records ---------------------------------------


def show_recent_agent_steps(
    telemetry_dir: Path,
    n: int = 20,
    fields: list[str] | None = None,
) -> None:
    """Pretty-print the most recent ``n`` ``agent_step`` records.

    If ``fields`` is provided, show only those (in the given order).
    Otherwise show a sensible default subset of scalar fields,
    excluding the high-dimensional vectors. When a high-dim field
    appears in an explicit ``fields=`` list, its value is summarised
    (shape + mean/std/min/max + first 3 + last 3) rather than dumped.

    Reads parquet shards under ``telemetry_dir/agent_step/`` in shard
    order, takes the last ``n`` records, and prints them as a
    ``key=value`` table — one record per group of lines.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    rows = _read_all_agent_step_rows(telemetry_dir)
    if not rows:
        print(f"(no agent_step records under {telemetry_dir / _AGENT_STEP_DIR})")
        return
    last_n = rows[-n:]
    show_fields = list(fields) if fields is not None else list(_DEFAULT_SCALAR_FIELDS)
    print(
        f"# agent_step — last {len(last_n)} of {len(rows)} records "
        f"(from {telemetry_dir / _AGENT_STEP_DIR})"
    )
    print()
    for ix, row in enumerate(last_n):
        print(f"## record {ix + 1} / {len(last_n)} (t={row.get('t', '?')})")
        for key in show_fields:
            if key not in row:
                continue
            value = row[key]
            if key in _HIGH_DIM_FIELDS:
                print(f"  {key}: {_summarise_vector(value)}")
            else:
                print(f"  {key}: {_format_scalar(value)}")
        print()


# ---- public API: world_event counts ---------------------------------------


def count_world_events(telemetry_dir: Path) -> dict[str, int]:
    """Count ``world_event`` records by ``event_type``.

    Returns the count dict. Prints a one-line-per-type summary at the
    same time so the function is useful both interactively (the counts
    print) and programmatically (the dict returns).

    Returns an empty dict if the JSONL file does not exist or is empty.
    Useful for sanity checks — that the ``env_reset`` count matches the
    expected number of episodes, that the ``builder_perturbation`` count
    matches the manual triggers fired during the run, etc.
    """
    path = telemetry_dir / _WORLD_EVENT_FILE
    if not path.is_file():
        print(f"(no world_event file at {path})")
        return {}
    counts: Counter[str] = Counter()
    bad_lines = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            event_type = rec.get("event_type")
            if isinstance(event_type, str):
                counts[event_type] += 1
            else:
                bad_lines += 1
    print(f"# world_event counts (from {path})")
    if not counts:
        print("(no events recorded)")
    else:
        for event_type in sorted(counts):
            print(f"  {event_type}: {counts[event_type]}")
    if bad_lines:
        print(f"  (skipped {bad_lines} malformed line(s))")
    return dict(counts)


# ---- public API: per-episode summary --------------------------------------


def show_episode_summary(
    telemetry_dir: Path, episode_id: int | None = None
) -> None:
    """Pretty-print a summary of one episode.

    If ``episode_id`` is ``None``, the most recent episode (by record
    order) is shown. The summary covers step count, action distribution,
    mean / std for ``kl_aggregate_t``, mean for ``recon_loss_t`` and
    ``intrinsic_signal_t`` and ``policy_entropy_t``, and any flagged
    outliers (``kl_aggregate_t`` |z| > 3, top recon-loss step).
    """
    rows = _read_all_agent_step_rows(telemetry_dir)
    if not rows:
        print(f"(no agent_step records under {telemetry_dir / _AGENT_STEP_DIR})")
        return
    target_id = (
        episode_id if episode_id is not None else _most_recent_episode_id(rows)
    )
    ep_rows = [r for r in rows if int(r.get("episode_id", -1)) == target_id]
    if not ep_rows:
        print(f"(episode {target_id} has no agent_step records)")
        return
    ep_rows.sort(key=lambda r: int(r.get("step_in_episode", 0)))

    n_steps = len(ep_rows)
    kl_vals = [float(r["kl_aggregate_t"]) for r in ep_rows]
    recon_vals = [float(r["recon_loss_t"]) for r in ep_rows]
    intrinsic_vals = [float(r["intrinsic_signal_t"]) for r in ep_rows]
    entropy_vals = [float(r["policy_entropy_t"]) for r in ep_rows]
    actions = [int(r["action_t"]) for r in ep_rows]

    kl_mean = statistics.fmean(kl_vals)
    kl_std = statistics.stdev(kl_vals) if n_steps >= 2 else 0.0
    recon_mean = statistics.fmean(recon_vals)
    intrinsic_mean = statistics.fmean(intrinsic_vals)
    entropy_mean = statistics.fmean(entropy_vals)

    t_first = int(ep_rows[0].get("t", 0))
    t_last = int(ep_rows[-1].get("t", 0))
    wc_first = int(ep_rows[0].get("wallclock_ms", 0))
    wc_last = int(ep_rows[-1].get("wallclock_ms", 0))

    print(f"# agent_step — episode {target_id}")
    print(f"  step count: {n_steps}")
    print(f"  env_step range: [{t_first}, {t_last}]")
    print(f"  duration: {_format_duration_ms(wc_last - wc_first)}")
    print(
        f"  kl_aggregate_t: mean={_format_scalar(kl_mean)}, "
        f"std={_format_scalar(kl_std)}, "
        f"min={_format_scalar(min(kl_vals))}, "
        f"max={_format_scalar(max(kl_vals))}"
    )
    print(f"  recon_loss_t: mean={_format_scalar(recon_mean)}")
    print(f"  intrinsic_signal_t: mean={_format_scalar(intrinsic_mean)}")
    print(f"  policy_entropy_t: mean={_format_scalar(entropy_mean)}")
    print(f"  actions: {_format_action_distribution(actions)}")

    outliers = _episode_outliers(ep_rows, kl_mean, kl_std)
    if outliers:
        print("  flagged outliers:")
        for s in outliers:
            print(f"    - {s}")
    else:
        print("  flagged outliers: none")

    # Probe 1.5 v2 (plan §2.7): when the episode contains 0.2.0 records
    # with non-None ``self_prediction_error_t``, print the self-prediction
    # error mean / std (excluding masked steps) and the per-episode
    # masked-step count. Probe 1 records skip this block entirely so
    # the no-affordance baseline stays visibly distinct from a Probe 1.5
    # record where the head produced a near-zero value (plan §3.3 "skip"
    # backward-compat approach).
    _print_episode_self_prediction_summary(ep_rows)


# ---- public API: dream rollout --------------------------------------------


def show_dream_rollout(
    telemetry_dir: Path, rollout_index: int = -1
) -> None:
    """Pretty-print one dream rollout.

    ``rollout_index`` follows Python list semantics: ``-1`` is the most
    recent, ``0`` the first, ``2`` the third, etc. Prints seed step,
    horizon, cumulative prior entropy, mean step KL between successive
    priors, max latent norm change, the imagined action sequence, and —
    if the runner emitted decoded observations — a 16×16 ASCII rendering
    of each step's imagined observation.
    """
    rows = _read_all_dream_rollout_rows(telemetry_dir)
    if not rows:
        print(
            f"(no dream_rollout records under "
            f"{telemetry_dir / _DREAM_ROLLOUT_DIR})"
        )
        return
    if rollout_index < -len(rows) or rollout_index >= len(rows):
        print(
            f"(rollout index {rollout_index} out of range; "
            f"have {len(rows)} rollouts: valid range "
            f"[{-len(rows)}, {len(rows) - 1}])"
        )
        return
    row = rows[rollout_index]
    seed_step = int(row.get("seed_step", 0))
    sequence_action = list(row.get("sequence_action") or [])
    horizon = len(sequence_action)
    prior_entropy = list(row.get("sequence_prior_entropy") or [])
    cumulative = float(row.get("cumulative_prior_entropy", 0.0))
    mean_kl = float(row.get("mean_step_kl_successive_priors", 0.0))
    max_norm = float(row.get("max_step_latent_norm_change", 0.0))
    decoded = row.get("sequence_decoded_obs")

    resolved_index = rollout_index if rollout_index >= 0 else len(rows) + rollout_index
    print(
        f"# dream_rollout — index {resolved_index} of {len(rows)} "
        f"(seed_step={seed_step}, horizon={horizon})"
    )
    print(f"  cumulative_prior_entropy: {_format_scalar(cumulative)}")
    if prior_entropy:
        print(
            f"  prior_entropy per step: "
            f"mean={_format_scalar(statistics.fmean(prior_entropy))}, "
            f"min={_format_scalar(min(prior_entropy))}, "
            f"max={_format_scalar(max(prior_entropy))}"
        )
    print(f"  mean_step_kl_successive_priors: {_format_scalar(mean_kl)}")
    print(f"  max_step_latent_norm_change: {_format_scalar(max_norm)}")
    if sequence_action:
        action_counts = _action_distribution_counts(sequence_action)
        print(
            f"  imagined action sequence: {sequence_action} "
            f"({_format_action_counts(action_counts)})"
        )

    if decoded:
        print()
        print(
            f"  decoded observations (16x16 ASCII, downsampled from 32x32; "
            f"' .:-=+*#%@' = low to high intensity):"
        )
        for step_ix, frame_bytes in enumerate(decoded):
            if not isinstance(frame_bytes, (bytes, bytearray)):
                continue
            print(f"  -- step {step_ix + 1} (action={sequence_action[step_ix]}):")
            for line in _render_ascii_frame(frame_bytes):
                print(f"  {line}")
            print()
    else:
        print("  decoded observations: (none recorded)")


# ---- public API: run-level summary ----------------------------------------


def show_run_summary(telemetry_dir: Path) -> None:
    """Print a high-level summary of the entire run.

    Counts: env steps (agent_step records), distinct episodes, dream
    rollouts. Action distribution across all records. KL trend
    comparison: first 25% mean vs. last 25% mean of records (a coarse
    "early vs. late" signal). Wallclock duration. Number of world
    events by type.
    """
    rows = _read_all_agent_step_rows(telemetry_dir)
    dream_rows = _read_all_dream_rollout_rows(telemetry_dir)
    print(f"# run summary (from {telemetry_dir})")
    print()

    if not rows:
        print(
            f"  agent_step: 0 records "
            f"(under {telemetry_dir / _AGENT_STEP_DIR})"
        )
    else:
        episode_ids = {int(r.get("episode_id", -1)) for r in rows}
        actions = [int(r["action_t"]) for r in rows if "action_t" in r]
        wc_first = int(rows[0].get("wallclock_ms", 0))
        wc_last = int(rows[-1].get("wallclock_ms", 0))
        t_first = int(rows[0].get("t", 0))
        t_last = int(rows[-1].get("t", 0))
        run_id = str(rows[0].get("run_id", "?"))
        schema_version = str(rows[0].get("schema_version", "?"))
        print(f"  run_id: {run_id}")
        print(f"  schema_version: {schema_version}")
        print(f"  total agent_step records: {len(rows)}")
        print(f"  total episodes: {len(episode_ids)}")
        print(f"  env_step range: [{t_first}, {t_last}]")
        print(f"  wallclock duration: {_format_duration_ms(wc_last - wc_first)}")
        if actions:
            print(f"  actions (overall): {_format_action_distribution(actions)}")

        early_mean, late_mean = _early_late_kl_means(rows)
        if early_mean is not None and late_mean is not None:
            delta = late_mean - early_mean
            sign = "+" if delta >= 0 else ""
            print(
                f"  kl_aggregate_t early/late: "
                f"early_mean={_format_scalar(early_mean)}, "
                f"late_mean={_format_scalar(late_mean)} "
                f"(delta={sign}{_format_scalar(delta)})"
            )

    print()
    print(f"  total dream_rollout records: {len(dream_rows)}")
    if dream_rows:
        seed_steps = [int(r.get("seed_step", 0)) for r in dream_rows]
        print(
            f"  dream seed_step range: "
            f"[{min(seed_steps)}, {max(seed_steps)}]"
        )

    print()
    counts = count_world_events(telemetry_dir)
    # ``count_world_events`` already prints; the returned dict is logged
    # here only as a tail summary so a user reading the run summary sees
    # both the per-type counts (printed above by the call) and the total.
    if counts:
        print(f"  world_event total: {sum(counts.values())}")


# ---- internal: parquet / JSONL reading -----------------------------------


def _read_all_agent_step_rows(telemetry_dir: Path) -> list[dict[str, Any]]:
    """Read every parquet shard under ``agent_step/`` in shard order.

    Returns rows in shard then row order — the same order ``ParquetSink``
    wrote them, which for the runner equals env-step order.
    """
    return _read_parquet_dir(telemetry_dir / _AGENT_STEP_DIR)


def _read_all_dream_rollout_rows(telemetry_dir: Path) -> list[dict[str, Any]]:
    return _read_parquet_dir(telemetry_dir / _DREAM_ROLLOUT_DIR)


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


# ---- internal: episode / outlier helpers ----------------------------------


def _most_recent_episode_id(rows: list[dict[str, Any]]) -> int:
    seen: list[int] = []
    for r in rows:
        eid = int(r.get("episode_id", -1))
        if eid not in seen:
            seen.append(eid)
    return seen[-1] if seen else -1


def _episode_outliers(
    ep_rows: list[dict[str, Any]],
    kl_mean: float,
    kl_std: float,
) -> list[str]:
    """Return formatted outlier strings for one episode.

    KL outliers: any step whose ``kl_aggregate_t`` z-score exceeds 3 in
    absolute value. Recon outliers: the single highest ``recon_loss_t``
    step (only flagged for episodes of 5+ steps so a tiny test episode
    doesn't always nominate "the worst" arbitrarily).
    """
    outliers: list[str] = []
    if kl_std > 0:
        for r in ep_rows:
            kl_v = float(r["kl_aggregate_t"])
            z = (kl_v - kl_mean) / kl_std
            if abs(z) > 3.0:
                outliers.append(
                    f"step_in_episode={int(r['step_in_episode'])}: "
                    f"kl_aggregate_t={_format_scalar(kl_v)} (z={z:+.2f})"
                )
    if len(ep_rows) >= 5:
        worst = max(ep_rows, key=lambda r: float(r["recon_loss_t"]))
        outliers.append(
            f"step_in_episode={int(worst['step_in_episode'])}: "
            f"recon_loss_t={_format_scalar(float(worst['recon_loss_t']))} "
            f"(top recon)"
        )
    return outliers


def _early_late_kl_means(
    rows: list[dict[str, Any]],
) -> tuple[float | None, float | None]:
    """Compute mean ``kl_aggregate_t`` over the first and last quarter.

    Returns ``(None, None)`` if the run is too short to define quarters
    meaningfully (under 4 records); otherwise both means are populated.
    """
    n = len(rows)
    if n < 4:
        return (None, None)
    quarter = max(1, n // 4)
    early = [float(r["kl_aggregate_t"]) for r in rows[:quarter]]
    late = [float(r["kl_aggregate_t"]) for r in rows[-quarter:]]
    return (statistics.fmean(early), statistics.fmean(late))


# ---- internal: action-distribution formatting ----------------------------


def _action_distribution_counts(actions: Sequence[int]) -> list[int]:
    counts = [0] * len(_ACTION_LABELS)
    for a in actions:
        if 0 <= a < len(_ACTION_LABELS):
            counts[a] += 1
    return counts


def _format_action_distribution(actions: Sequence[int]) -> str:
    counts = _action_distribution_counts(actions)
    return _format_action_counts(counts)


def _format_action_counts(counts: Sequence[int]) -> str:
    return " ".join(
        f"{label}={count}" for label, count in zip(_ACTION_LABELS, counts, strict=False)
    )


# ---- internal: scalar / vector / time formatting -------------------------


def _format_scalar(value: Any) -> str:
    """Format a scalar for human reading.

    Floats round to 4 decimal places; values with very small or very
    large magnitude switch to scientific notation. Ints print as-is;
    strings and ``None`` print verbatim.
    """
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        magnitude = abs(value)
        if value == 0.0 or (1e-3 <= magnitude < 1e6):
            return f"{value:.4f}"
        return f"{value:.4e}"
    return str(value)


def _summarise_vector(value: Any) -> str:
    """Summarise a high-dimensional vector — shape + stats + ends.

    Accepts a list of floats (or a 2-element tuple of lists, e.g.
    ``q_params_t``). Recurses one level for the tuple case so a caller
    requesting ``q_params_t`` gets two summaries side by side.
    """
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        # Pair-of-vectors form (q_params, p_params).
        parts = [_summarise_vector(inner) for inner in value]
        return "(" + " | ".join(parts) + ")"
    if not isinstance(value, list):
        return _format_scalar(value)
    if not value:
        return "shape=(0,) (empty)"
    floats = [float(x) for x in value]
    n = len(floats)
    mean = statistics.fmean(floats)
    std = statistics.stdev(floats) if n >= 2 else 0.0
    head = ", ".join(_format_scalar(x) for x in floats[: min(3, n)])
    tail_part = ""
    if n > 3:
        tail_part = ", ..., " + ", ".join(_format_scalar(x) for x in floats[-3:])
    return (
        f"shape=({n},) mean={_format_scalar(mean)} "
        f"std={_format_scalar(std)} "
        f"min={_format_scalar(min(floats))} "
        f"max={_format_scalar(max(floats))} "
        f"[{head}{tail_part}]"
    )


def _format_duration_ms(ms: int) -> str:
    """Format a millisecond duration as a relative time string.

    The user brief: "show wallclock_ms as relative durations (e.g.,
    'episode took 3.2s') rather than as raw milliseconds." Negative
    durations are possible if records are out of order; emit the sign
    rather than swallow it.
    """
    if ms == 0:
        return "0ms"
    sign = "-" if ms < 0 else ""
    abs_ms = abs(ms)
    if abs_ms < 1000:
        return f"{sign}{abs_ms}ms"
    seconds = abs_ms / 1000.0
    if seconds < 60.0:
        return f"{sign}{seconds:.2f}s"
    minutes = seconds / 60.0
    if minutes < 60.0:
        return f"{sign}{minutes:.2f}m"
    hours = minutes / 60.0
    return f"{sign}{hours:.2f}h"


# ---- internal: ASCII rendering for dream observations -------------------


def _render_ascii_frame(frame_bytes: bytes | bytearray) -> list[str]:
    """Render a 32×32 grayscale uint8 frame as 16×16 ASCII.

    Each output character represents a 2×2 block of source pixels, taken
    as the integer mean of the four. The ramp is :data:`_ASCII_RAMP`
    (10 chars from low to high intensity).

    Returns 16 lines, each 16 characters wide. Returns ``["(decoded
    obs has unexpected size {n})"]`` if the byte length is not 1024
    (32 × 32 uint8). The runner-side code in
    :func:`kind.training.runner.Runner._emit_dream` always emits 32×32,
    so this guard is the defensive boundary against shard corruption or
    a future schema migration that changes the resolution.
    """
    n = len(frame_bytes)
    if n != 32 * 32:
        return [f"(decoded obs has unexpected size {n}, expected 1024)"]
    raw = list(frame_bytes)
    lines: list[str] = []
    ramp_len = len(_ASCII_RAMP)
    last_ramp_ix = ramp_len - 1
    for out_row in range(16):
        chars: list[str] = []
        for out_col in range(16):
            r0 = 2 * out_row
            c0 = 2 * out_col
            block_sum = (
                raw[r0 * 32 + c0]
                + raw[r0 * 32 + c0 + 1]
                + raw[(r0 + 1) * 32 + c0]
                + raw[(r0 + 1) * 32 + c0 + 1]
            )
            block_mean = block_sum // 4
            ramp_ix = (block_mean * last_ramp_ix) // 255
            if ramp_ix < 0:
                ramp_ix = 0
            if ramp_ix > last_ramp_ix:
                ramp_ix = last_ramp_ix
            chars.append(_ASCII_RAMP[ramp_ix])
        lines.append("".join(chars))
    return lines


# ---- Probe 1.5 v2 self-prediction surface ---------------------------------


_SELF_PREDICTION_SCHEMA_VERSION: str = "0.2.0"
_SELF_PREDICTION_TOP_DIMS_DEFAULT: int = 10
# Counterfactual probe defaults (plan §2.7). These mirror the digest /
# counterfactual probe's regime-classification machinery — held as
# module-level constants so the values are visible at the top of the
# self-prediction block rather than buried in a function arg default.
_PERTURBATION_WINDOW_W: int = 25  # ± env_steps from a builder_perturbation
_HIGH_REGIME_QUARTILE: float = 0.75
_GAUSSIAN_PERTURB_ALPHA: float = 1.0  # alpha=1.0 is the synthesis canonical
_DEFAULT_PERTURBATION_DISTRIBUTIONS: tuple[str, ...] = (
    "gaussian",
    "zero",
    "uniform",
)
_DEFAULT_REGIMES: tuple[str, ...] = (
    "perturbation_window",
    "high_disagreement",
    "high_kl",
    "steady_state",
)
_PROBE_1_NO_TELEMETRY_MESSAGE: str = (
    "(no self_prediction telemetry — records are Probe 1, "
    "schema_version 0.1.0)"
)
_PROBE_1_NO_SCALAR_MESSAGE: str = (
    "(no scalar to perturb — records are Probe 1, schema_version 0.1.0)"
)


def _has_self_prediction_telemetry(rows: list[dict[str, Any]]) -> bool:
    """True iff at least one row is stamped 0.2.0 with a non-None scalar.

    Plan §3.3 "skip" discriminator. A 0.1.0 record carries the new
    fields as ``None`` (when round-tripped through the 0.2.0
    ``ParquetSink``) or as absent keys (when read from Probe 1's
    pre-bump on-disk shards under ``runs/probe1-20260503-123926/``);
    both cases evaluate to ``False`` here, which is what makes the
    Probe 1 baseline visibly distinct from a Probe 1.5 episode where
    the head produced a near-zero value.
    """
    for r in rows:
        if (
            str(r.get("schema_version", "")) == _SELF_PREDICTION_SCHEMA_VERSION
            and r.get("self_prediction_error_t") is not None
        ):
            return True
    return False


def _print_episode_self_prediction_summary(ep_rows: list[dict[str, Any]]) -> None:
    """Print the per-episode self-prediction summary for ``show_episode_summary``.

    Plan §2.7 surface. Mirrors the digest's lines (plan §2.6) but
    rendered as the eyeball's two-space-indented stdout convention.
    Masked steps (sentinel scalar=0.0, masked_flag=True; first step of
    each episode under the Phase 0 convention) are excluded from
    mean/std/min/max; the masked-step count is logged separately so
    the human builder sees what was excluded. Probe 1 records skip
    this block entirely.
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
        # All 0.2.0 records in the episode are masked (degenerate but
        # possible if a future mask convention extends beyond the first
        # step); emit the count and skip the empirical aggregates.
        print(
            f"  self_prediction_error_t masked steps in episode: "
            f"count={masked_count}"
        )
        return

    sp_vals = [float(r["self_prediction_error_t"]) for r in non_masked]
    mean_v = statistics.fmean(sp_vals)
    std_v = statistics.stdev(sp_vals) if len(sp_vals) >= 2 else 0.0
    print(
        f"  self_prediction_error_t (excluding masked steps): "
        f"mean={_format_scalar(mean_v)}, "
        f"std={_format_scalar(std_v)}, "
        f"min={_format_scalar(min(sp_vals))}, "
        f"max={_format_scalar(max(sp_vals))}"
    )
    print(
        f"  self_prediction_error_t masked steps in episode: count={masked_count}"
    )


def _per_dim_self_prediction_error_variance(
    rows: list[dict[str, Any]],
) -> list[float]:
    """Per-dim variance of ``self_prediction_t[d] - h_{(t+1)}[d]``.

    Mirror of :func:`kind.observer.digest._per_dim_self_prediction_error_variance`
    but kept local to this module so the eyeball does not depend on the
    digest's internals for behaviour-side analysis. Operates on a
    single-episode row list, in step order, of 0.2.0 records. Walks
    consecutive pairs ``(t, t+1)`` where the predicting row is
    non-masked; uses ``rows[i+1]['h_t']`` as the proxy for the EMA
    target's ``bar{h}_{t+1}`` (the actual EMA target is not in
    telemetry; the next-step online ``h`` is the closest available
    stand-in for a behaviour-side analyzer).

    Returns a list of length ``h_dim`` with sample variance per dim,
    or an empty list if no valid pairs exist. Two-or-more residuals
    are required for a non-degenerate variance estimate; any dim with
    fewer pairs returns 0.0 in that slot.
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


def show_self_prediction(
    telemetry_dir: Path,
    *,
    episode_id: int | None = None,
    top_k_dims: int = _SELF_PREDICTION_TOP_DIMS_DEFAULT,
) -> None:
    """Print per-dimension self-prediction-error allocation for one episode.

    Plan §2.7 / synthesis §1.4 element 3. For one episode (or the most
    recent if ``episode_id`` is ``None``), print the top ``top_k_dims``
    dimensions of ``h_t`` whose self-prediction error variance across
    the episode is highest. Masked steps (the first step of each
    episode under the Phase 0 convention) are excluded.

    The per-step per-dim error is ``self_prediction_t[d] -
    h_{(t+1)}[d]`` with the next non-masked step's ``h_t`` standing in
    for the runner's training-time EMA target ``bar{h}_{t+1}`` — see
    :func:`_per_dim_self_prediction_error_variance` for the
    substitution rationale.

    For Probe 1 records (``schema_version == "0.1.0"``), prints the
    no-self-prediction-telemetry message and returns without crashing.
    """
    if top_k_dims <= 0:
        raise ValueError(f"top_k_dims must be positive, got {top_k_dims}")
    rows = _read_all_agent_step_rows(telemetry_dir)
    if not rows:
        print(f"(no agent_step records under {telemetry_dir / _AGENT_STEP_DIR})")
        return
    if not _has_self_prediction_telemetry(rows):
        print(_PROBE_1_NO_TELEMETRY_MESSAGE)
        return

    target_id = (
        episode_id if episode_id is not None else _most_recent_episode_id(rows)
    )
    ep_rows = [r for r in rows if int(r.get("episode_id", -1)) == target_id]
    if not ep_rows:
        print(f"(episode {target_id} has no agent_step records)")
        return
    ep_rows.sort(key=lambda r: int(r.get("step_in_episode", 0)))

    # Restrict to 0.2.0 rows (a synthetic mixed-version episode might
    # exist in test fixtures even if not in production) — the per-dim
    # arithmetic only makes sense for records the head produced.
    rows_with_scalar = [
        r
        for r in ep_rows
        if str(r.get("schema_version", "")) == _SELF_PREDICTION_SCHEMA_VERSION
        and r.get("self_prediction_error_t") is not None
    ]
    if not rows_with_scalar:
        print(_PROBE_1_NO_TELEMETRY_MESSAGE)
        return

    per_dim_var = _per_dim_self_prediction_error_variance(rows_with_scalar)
    masked_count = sum(
        1 for r in rows_with_scalar if r.get("self_prediction_error_masked_t") is True
    )

    print(
        f"# self_prediction allocation — episode {target_id} "
        f"(per-dim variance of (ĥ_t+1 - h_t+1) across non-masked, "
        f"non-last steps; using next-step h_t as the EMA-target proxy)"
    )
    print(f"  records considered: {len(rows_with_scalar)}")
    print(f"  masked steps (excluded): {masked_count}")
    if not per_dim_var:
        print("  (no valid (t, t+1) pairs available; episode too short or all masked)")
        return
    h_dim = len(per_dim_var)
    print(f"  h_dim: {h_dim}")
    ranked = sorted(range(h_dim), key=lambda d: per_dim_var[d], reverse=True)
    top = ranked[: min(top_k_dims, h_dim)]
    print(f"  top {len(top)} dims:")
    for d in top:
        print(f"    - dim={d}: var={_format_scalar(per_dim_var[d])}")


def show_self_prediction_conditioning(
    run_dir: Path,
    *,
    checkpoint_id: str | None = None,
    n_states: int = 200,
    perturbation_distributions: list[str] | None = None,
    regimes: list[str] | None = None,
    perturbation_window_w: int = _PERTURBATION_WINDOW_W,
    seed: int = 0,
) -> None:
    """Print the per-regime KL table for the actor's behavior-side conditioning.

    Plan §2.7 / synthesis §1.4 element 4. Loads the run's checkpoint
    (the named one, or the lexicographically greatest if ``checkpoint_id``
    is ``None``), samples ``n_states`` states from the run's
    ``agent_step`` records (excluding masked steps; the scalar value at
    a masked step is sentinel zero, not an empirical reading), and for
    each sampled state computes the actor's policy distribution under
    several controlled perturbations of the
    ``self_prediction_error`` field on PolicyView. KL divergence
    between unperturbed and perturbed policies is the metric;
    aggregation is per-regime.

    Default perturbation distributions (synthesis §3 v2):

    - ``gaussian``: scalar replaced by ``scalar + N(0, alpha *
      empirical_std)`` with ``alpha = _GAUSSIAN_PERTURB_ALPHA = 1.0``.
    - ``zero``: scalar replaced by ``0.0`` (the same sentinel as the
      masked-first-step case).
    - ``uniform``: scalar replaced by
      ``Uniform(empirical_min, empirical_max)``.

    Default regimes (synthesis §1.7(c) v2):

    - ``perturbation_window``: steps within ``±perturbation_window_w``
      of any ``builder_perturbation`` ``world_event``'s ``t_event``.
    - ``high_disagreement``: top quartile of ``intrinsic_signal_t``.
    - ``high_kl``: top quartile of ``kl_aggregate_t``.
    - ``steady_state``: complement (steps in none of the above).

    Output is a per-regime table:

        regime              n_states  KL_mean  KL_std  KL_p50  KL_p90
        perturbation_window     35     0.42    0.31    0.38    0.78
        ...

    The shape of the table is what the counterfactual probe (plan
    §2.9) consumes as its analysis surface; the threshold on the
    pattern is qualitative and is the journal entry's call.

    For Probe 1 records (or a Probe 1 checkpoint), prints the
    no-scalar-to-perturb message and returns without crashing.
    """
    if n_states <= 0:
        raise ValueError(f"n_states must be positive, got {n_states}")
    if perturbation_window_w < 0:
        raise ValueError(
            f"perturbation_window_w must be non-negative, got "
            f"{perturbation_window_w}"
        )
    perturbation_list = (
        list(perturbation_distributions)
        if perturbation_distributions is not None
        else list(_DEFAULT_PERTURBATION_DISTRIBUTIONS)
    )
    for p in perturbation_list:
        if p not in _DEFAULT_PERTURBATION_DISTRIBUTIONS:
            raise ValueError(
                f"unknown perturbation distribution: {p!r}; "
                f"supported: {_DEFAULT_PERTURBATION_DISTRIBUTIONS}"
            )
    regime_list = (
        list(regimes) if regimes is not None else list(_DEFAULT_REGIMES)
    )
    for rg in regime_list:
        if rg not in _DEFAULT_REGIMES:
            raise ValueError(
                f"unknown regime: {rg!r}; supported: {_DEFAULT_REGIMES}"
            )

    telemetry_dir = run_dir / "telemetry"
    rows = _read_all_agent_step_rows(telemetry_dir)
    if not rows:
        print(f"(no agent_step records under {telemetry_dir / _AGENT_STEP_DIR})")
        return
    if not _has_self_prediction_telemetry(rows):
        print(_PROBE_1_NO_SCALAR_MESSAGE)
        return

    # Restrict to 0.2.0 records with non-None scalar AND non-masked
    # (sentinel zero is not an empirical reading; mask-flagged steps
    # are excluded from the n_states sample per plan §2.7's
    # "Masked steps correctly excluded from n_states sample" test).
    candidate_rows = [
        r
        for r in rows
        if str(r.get("schema_version", "")) == _SELF_PREDICTION_SCHEMA_VERSION
        and r.get("self_prediction_error_t") is not None
        and r.get("self_prediction_error_masked_t") is not True
    ]
    if not candidate_rows:
        print(_PROBE_1_NO_SCALAR_MESSAGE)
        return

    # ---- locate + load checkpoint ----
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        print(f"(no checkpoints/ under {run_dir})")
        return
    if checkpoint_id is None:
        ckpt_dirs = sorted(
            d.name
            for d in checkpoints_dir.iterdir()
            if d.is_dir() and not d.name.endswith(".staging")
        )
        if not ckpt_dirs:
            print(f"(no checkpoints under {checkpoints_dir})")
            return
        checkpoint_id = ckpt_dirs[-1]
    checkpoint_dir = checkpoints_dir / checkpoint_id
    if not checkpoint_dir.is_dir():
        print(f"(checkpoint not found: {checkpoint_id!r} under {checkpoints_dir})")
        return
    schema_version_path = checkpoint_dir / "schema_version.txt"
    if schema_version_path.is_file():
        ckpt_version = schema_version_path.read_text().strip()
        if ckpt_version != _SELF_PREDICTION_SCHEMA_VERSION:
            # A Probe 1 checkpoint has no extended actor input column;
            # the perturbation has nothing to bite on. Per plan §2.7's
            # "no-scalar-to-perturb message" failure mode, surface this
            # clearly rather than silently constructing a Probe-1.5
            # actor and pretending.
            print(_PROBE_1_NO_SCALAR_MESSAGE)
            return

    weights_path = checkpoint_dir / "weights.safetensors"
    if not weights_path.is_file():
        print(f"(no weights.safetensors under {checkpoint_dir})")
        return

    actor = _build_actor_from_checkpoint(weights_path)
    if actor is None:
        print(
            f"(checkpoint {checkpoint_id!r} carries no actor.* weights or "
            f"the weight shapes are not Probe 1.5-compatible)"
        )
        return

    # ---- compute regime classification + empirical statistics ----
    perturbation_t_events = _builder_perturbation_t_events(telemetry_dir)
    intrinsic_q3 = _quartile_threshold(
        [float(r["intrinsic_signal_t"]) for r in candidate_rows],
        _HIGH_REGIME_QUARTILE,
    )
    kl_q3 = _quartile_threshold(
        [float(r["kl_aggregate_t"]) for r in candidate_rows],
        _HIGH_REGIME_QUARTILE,
    )
    sp_vals = [float(r["self_prediction_error_t"]) for r in candidate_rows]
    sp_mean = statistics.fmean(sp_vals)
    sp_sigma = statistics.stdev(sp_vals) if len(sp_vals) >= 2 else 0.0
    sp_min = min(sp_vals)
    sp_max = max(sp_vals)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    sample_size = min(n_states, len(candidate_rows))
    indices = torch.randperm(len(candidate_rows), generator=rng)[:sample_size].tolist()
    sampled_rows = [candidate_rows[i] for i in indices]

    # ---- per-state KL under each perturbation, classify by regime ----
    # Aggregate across all perturbation distributions per regime — the
    # synthesis §3 (v2) default is to sweep all three; the per-regime
    # KL distribution is computed across the union.
    per_regime_kls: dict[str, list[float]] = {rg: [] for rg in regime_list}
    sampled_actions = [int(r["action_t"]) for r in sampled_rows]
    del sampled_actions  # unused but documents the sample's consistency
    for r in sampled_rows:
        regime = _classify_regime(
            r,
            perturbation_t_events,
            perturbation_window_w,
            intrinsic_q3,
            kl_q3,
            regime_list,
        )
        if regime is None:
            continue
        h_vec = torch.tensor(
            [float(x) for x in r["h_t"]], dtype=torch.float32
        ).unsqueeze(0)
        z_vec = torch.tensor(
            [float(x) for x in r["z_t"]], dtype=torch.float32
        ).unsqueeze(0)
        unperturbed_scalar = float(r["self_prediction_error_t"])
        unperturbed_logits = _actor_logits(
            actor, h_vec, z_vec, torch.tensor([unperturbed_scalar])
        )
        for distribution in perturbation_list:
            perturbed_scalar = _apply_perturbation(
                distribution,
                unperturbed_scalar,
                sp_sigma,
                sp_min,
                sp_max,
                rng,
            )
            perturbed_logits = _actor_logits(
                actor, h_vec, z_vec, torch.tensor([perturbed_scalar])
            )
            kl = _kl_divergence_logits(unperturbed_logits, perturbed_logits)
            per_regime_kls[regime].append(kl)

    # ---- print table ----
    print(
        f"# self_prediction_error conditioning (run_dir={run_dir}, "
        f"checkpoint_id={checkpoint_id})"
    )
    print(
        f"  sampled n_states={sample_size}; perturbations="
        f"{perturbation_list}; window_w=±{perturbation_window_w}; seed={seed}"
    )
    print(
        f"  empirical scalar: mean={_format_scalar(sp_mean)} "
        f"sigma={_format_scalar(sp_sigma)} "
        f"range=[{_format_scalar(sp_min)}, {_format_scalar(sp_max)}]"
    )
    print()
    print(f"  {'regime':<22}{'n_states':>10}{'KL_mean':>10}"
          f"{'KL_std':>10}{'KL_p50':>10}{'KL_p90':>10}")
    for rg in regime_list:
        kls = per_regime_kls[rg]
        n = len(kls)
        if n == 0:
            print(f"  {rg:<22}{0:>10}{'-':>10}{'-':>10}{'-':>10}{'-':>10}")
            continue
        m = statistics.fmean(kls)
        s = statistics.stdev(kls) if n >= 2 else 0.0
        p50 = _percentile(kls, 0.50)
        p90 = _percentile(kls, 0.90)
        print(
            f"  {rg:<22}{n:>10}"
            f"{_format_scalar(m):>10}{_format_scalar(s):>10}"
            f"{_format_scalar(p50):>10}{_format_scalar(p90):>10}"
        )


def _builder_perturbation_t_events(telemetry_dir: Path) -> list[int]:
    """Read ``world_event.jsonl`` and return ``t_event`` for builder_perturbation.

    Empty list if the file is absent (Probe 1.5 main run defaults are
    perturbation-free; the regime is then a defensive empty bin per
    plan §2.9 "the perturbation_window regime may have few or no
    states; in that case the comparison reduces to the other three
    regimes plus a flag in the report").
    """
    path = telemetry_dir / _WORLD_EVENT_FILE
    if not path.is_file():
        return []
    t_events: list[int] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event_type") == "builder_perturbation":
                t = rec.get("t_event")
                if isinstance(t, int):
                    t_events.append(t)
    return t_events


def _quartile_threshold(vals: list[float], q: float) -> float:
    """Return the q-quantile of ``vals`` (0.0–1.0). Returns ``+inf`` if empty."""
    if not vals:
        return float("inf")
    s = sorted(vals)
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _percentile(vals: list[float], q: float) -> float:
    """Return the q-quantile of ``vals`` (0.0–1.0). Linear interpolation."""
    if not vals:
        return 0.0
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _classify_regime(
    row: dict[str, Any],
    perturbation_t_events: list[int],
    window_w: int,
    intrinsic_q3: float,
    kl_q3: float,
    regime_list: list[str],
) -> str | None:
    """Pick the first regime in ``regime_list`` order that the row matches.

    Priority order is the order the caller passes in ``regime_list``;
    the default ordering puts ``perturbation_window`` first because
    plan §2.9 names it as the most diagnostic regime when perturbations
    are present, then ``high_disagreement`` and ``high_kl`` because
    those are the within-trajectory regime markers, then
    ``steady_state`` as the catch-all complement. Returns ``None`` if
    the row matches no regime — e.g. a non-default ``regime_list`` that
    excludes ``steady_state`` and the row has no other classification.
    """
    t = int(row.get("t", -1))
    intrinsic = float(row.get("intrinsic_signal_t", 0.0))
    kl = float(row.get("kl_aggregate_t", 0.0))
    in_window = any(
        abs(t - tp) <= window_w for tp in perturbation_t_events
    )
    is_high_disagreement = intrinsic >= intrinsic_q3
    is_high_kl = kl >= kl_q3
    is_steady = not (in_window or is_high_disagreement or is_high_kl)
    for rg in regime_list:
        if rg == "perturbation_window" and in_window:
            return rg
        if rg == "high_disagreement" and is_high_disagreement:
            return rg
        if rg == "high_kl" and is_high_kl:
            return rg
        if rg == "steady_state" and is_steady:
            return rg
    return None


def _apply_perturbation(
    distribution: str,
    scalar: float,
    sigma: float,
    emp_min: float,
    emp_max: float,
    rng: "torch.Generator",
) -> float:
    """Apply one of the three perturbation distributions to a scalar."""
    if distribution == "gaussian":
        noise = torch.randn(1, generator=rng).item() * sigma * _GAUSSIAN_PERTURB_ALPHA
        return scalar + noise
    if distribution == "zero":
        return 0.0
    if distribution == "uniform":
        if emp_max <= emp_min:
            return scalar
        u = torch.rand(1, generator=rng).item()
        return emp_min + u * (emp_max - emp_min)
    raise ValueError(f"unknown perturbation distribution: {distribution!r}")


def _build_actor_from_checkpoint(weights_path: Path) -> "torch.nn.Module | None":
    """Construct an :class:`~kind.agents.actor.Actor` from a safetensors blob.

    Reads only the ``actor.*`` keys; infers ``h_dim``, ``z_dim``,
    ``mlp_hidden``, ``action_dim`` from the weight shapes (the first
    layer's input dim is ``h_dim + z_dim + 1``; the last layer's
    output dim is ``action_dim``; ``mlp_hidden`` is the first layer's
    output dim). Returns ``None`` if the actor weights are absent or
    the shape contract is not Probe 1.5-shaped (i.e. the input dim is
    not at least 3 — minimum ``h=1, z=1, +1``). The split between
    ``h_dim`` and ``z_dim`` is not recoverable from weights alone, so
    we use the ratio ``h_dim:z_dim`` from ``WorldModelConfig``'s
    defaults — but at the level the actor cares about, only the sum
    ``h_dim + z_dim + 1`` matters for the ``forward`` arithmetic. The
    helper splits at the schema's default boundary (h=200, z=16) when
    possible; for tiny test sizes (h=4, z=2) the test fixture supplies
    matching values via construction.

    Implementation note: we don't actually need to know the
    ``h_dim``/``z_dim`` split inside the actor — ``Actor.forward``
    takes ``view.h``, ``view.z``, ``view.self_prediction_error`` and
    concatenates them along ``dim=-1``. As long as the caller supplies
    tensors whose shapes sum to the saved input dim, the forward path
    is identical regardless of where the split is. We therefore
    construct the actor with a default (h, z) pair that sums to the
    correct input dim, and rely on the caller (this module's
    :func:`_actor_logits`) to honor the same split.
    """
    from kind.agents.actor import Actor
    from safetensors.torch import load_file

    weights = load_file(str(weights_path), device="cpu")
    actor_state: dict[str, torch.Tensor] = {}
    for k, v in weights.items():
        if k.startswith("actor."):
            actor_state[k[len("actor.") :]] = v
    if not actor_state:
        return None
    first_layer_key = "net.0.weight"
    last_layer_key = "net.4.weight"
    if first_layer_key not in actor_state or last_layer_key not in actor_state:
        return None
    first_layer = actor_state[first_layer_key]
    last_layer = actor_state[last_layer_key]
    if first_layer.dim() != 2 or last_layer.dim() != 2:
        return None
    mlp_hidden = int(first_layer.shape[0])
    input_dim = int(first_layer.shape[1])
    action_dim = int(last_layer.shape[0])
    if input_dim < 3 or action_dim <= 0 or mlp_hidden <= 0:
        return None
    h_plus_z = input_dim - 1  # the +1 is the scalar
    # The actor's forward is shape-agnostic to the (h_dim, z_dim) split
    # — only the sum matters for the concat. Split at the
    # WorldModelConfig default ratio when possible (h_dim=200, z_dim=16
    # ⇒ h:(h+z) = 200/216) so the Actor's stored ``self.h_dim`` /
    # ``self.z_dim`` are sane for any caller that introspects them;
    # for tiny test sizes we fall back to a 2:1 ratio which keeps both
    # ≥1 for h+z ≥ 3.
    if h_plus_z >= 216 and h_plus_z * 200 % 216 == 0:
        h_dim = h_plus_z * 200 // 216
        z_dim = h_plus_z - h_dim
    else:
        z_dim = max(1, h_plus_z // 3)
        h_dim = h_plus_z - z_dim
    actor = Actor(h_dim=h_dim, z_dim=z_dim, action_dim=action_dim, mlp_hidden=mlp_hidden)
    actor.load_state_dict(actor_state)
    actor.eval()
    return actor


def _actor_logits(
    actor: "torch.nn.Module",
    h: "torch.Tensor",
    z: "torch.Tensor",
    scalar: "torch.Tensor",
) -> "torch.Tensor":
    """Run the actor's first-layer + MLP path; return logits.

    This bypasses :meth:`~kind.agents.actor.Actor.forward`'s sampling
    step (we want the deterministic logits for a KL comparison, not a
    stochastic action). The actor's network is ``self.net`` per the
    Phase 2 build journal; we replicate the concatenation order
    ``(h, z, scalar_col)`` to stay byte-equivalent to the live
    forward.
    """
    from kind.agents.actor import _scalar_to_column

    scalar_col = _scalar_to_column(scalar, batch_size=h.shape[0])
    x = torch.cat([h, z, scalar_col], dim=-1)
    with torch.no_grad():
        logits = actor.net(x)  # type: ignore[operator]
    return cast(torch.Tensor, logits)


def _kl_divergence_logits(
    p_logits: "torch.Tensor", q_logits: "torch.Tensor"
) -> float:
    """KL(p || q) where p, q are categorical distributions from logits.

    Uses the standard ``sum_i p_i (log p_i - log q_i)`` form on the
    softmax-normalized probabilities. Returns a Python float.
    """
    log_p = torch.log_softmax(p_logits, dim=-1)
    log_q = torch.log_softmax(q_logits, dim=-1)
    p = torch.exp(log_p)
    kl = (p * (log_p - log_q)).sum(dim=-1)
    return float(kl.mean().item())


# ---- CLI ------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse hierarchy for ``python -m kind.observer.eyeball``.

    Five subcommands wrap the public functions; each takes the telemetry
    directory as its first positional argument and any function-specific
    options after it.
    """
    parser = argparse.ArgumentParser(
        prog="python -m kind.observer.eyeball",
        description="Read Probe 1 telemetry shards and JSONL streams.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_recent = sub.add_parser("recent", help="Show the most recent agent_step records.")
    p_recent.add_argument("telemetry_dir", type=Path)
    p_recent.add_argument("-n", type=int, default=20, help="Number of records (default 20).")
    p_recent.add_argument(
        "--field",
        action="append",
        default=None,
        dest="fields",
        help="Field to include (repeatable). Default: scalar fields only.",
    )

    p_events = sub.add_parser(
        "events", help="Count world_event records by event_type."
    )
    p_events.add_argument("telemetry_dir", type=Path)

    p_episode = sub.add_parser(
        "episode", help="Summarise one episode's agent_step records."
    )
    p_episode.add_argument("telemetry_dir", type=Path)
    p_episode.add_argument(
        "-e",
        "--episode-id",
        type=int,
        default=None,
        help="Episode id (default: most recent).",
    )

    p_dream = sub.add_parser(
        "dream",
        help="Show one dream rollout (with ASCII art if observations were emitted).",
    )
    p_dream.add_argument("telemetry_dir", type=Path)
    p_dream.add_argument(
        "-i",
        "--index",
        type=int,
        default=-1,
        help="Rollout index (-1 = most recent).",
    )

    p_summary = sub.add_parser("summary", help="Print a high-level run summary.")
    p_summary.add_argument("telemetry_dir", type=Path)

    # Probe 1.5 v2 self-prediction surface (plan §2.7).
    p_selfpred = sub.add_parser(
        "selfpred",
        help=(
            "Show per-dimension self-prediction-error allocation for one "
            "episode (Probe 1.5 v2)."
        ),
    )
    p_selfpred.add_argument("telemetry_dir", type=Path)
    p_selfpred.add_argument(
        "-e",
        "--episode-id",
        type=int,
        default=None,
        help="Episode id (default: most recent).",
    )
    p_selfpred.add_argument(
        "-k",
        "--top-k-dims",
        type=int,
        default=_SELF_PREDICTION_TOP_DIMS_DEFAULT,
        help=f"Top-k dims to show (default {_SELF_PREDICTION_TOP_DIMS_DEFAULT}).",
    )

    p_cond = sub.add_parser(
        "cond",
        help=(
            "Show per-regime KL table for the actor's behavior-side "
            "self-prediction-error conditioning (Probe 1.5 v2)."
        ),
    )
    p_cond.add_argument("run_dir", type=Path)
    p_cond.add_argument(
        "-c",
        "--checkpoint-id",
        type=str,
        default=None,
        help="Checkpoint id (default: most recent).",
    )
    p_cond.add_argument(
        "-n",
        "--n-states",
        type=int,
        default=200,
        help="Number of states to sample (default 200).",
    )
    p_cond.add_argument(
        "--perturbation",
        action="append",
        default=None,
        dest="perturbations",
        choices=list(_DEFAULT_PERTURBATION_DISTRIBUTIONS),
        help=(
            "Perturbation distribution (repeatable). Default: all three "
            f"({list(_DEFAULT_PERTURBATION_DISTRIBUTIONS)})."
        ),
    )
    p_cond.add_argument(
        "--regime",
        action="append",
        default=None,
        dest="regimes",
        choices=list(_DEFAULT_REGIMES),
        help=(
            "Regime to include (repeatable). Default: all four "
            f"({list(_DEFAULT_REGIMES)})."
        ),
    )
    p_cond.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for state sampling and perturbations (default 0).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. ``argv`` defaults to ``sys.argv[1:]``."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    command: str = args.command
    if command == "recent":
        show_recent_agent_steps(args.telemetry_dir, n=args.n, fields=args.fields)
    elif command == "events":
        count_world_events(args.telemetry_dir)
    elif command == "episode":
        show_episode_summary(args.telemetry_dir, episode_id=args.episode_id)
    elif command == "dream":
        show_dream_rollout(args.telemetry_dir, rollout_index=args.index)
    elif command == "summary":
        show_run_summary(args.telemetry_dir)
    elif command == "selfpred":
        show_self_prediction(
            args.telemetry_dir,
            episode_id=args.episode_id,
            top_k_dims=args.top_k_dims,
        )
    elif command == "cond":
        show_self_prediction_conditioning(
            args.run_dir,
            checkpoint_id=args.checkpoint_id,
            n_states=args.n_states,
            perturbation_distributions=args.perturbations,
            regimes=args.regimes,
            seed=args.seed,
        )
    else:  # pragma: no cover — argparse rejects unknown commands first
        parser.error(f"unknown command: {command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
