"""Probe 2 Phase 3 — shuffled-telemetry generator with four protocols.

Implementation plan §2.4 + §6 calibration-protocol element 2;
synthesis §2.4 element 2 ("shuffled-telemetry baseline"). The four
protocols write a structurally indistinguishable telemetry directory
the hierarchical digest reads without modification. Each is the null
baseline against which a specific invariant of a genuine reading is
calibrated:

- :func:`shuffle_within_episode` — per-episode marginals preserved;
  within-episode temporal structure broken. The 16 content fields move
  with one shared per-episode permutation; the indexing fields
  (``schema_version``, ``run_id``, ``checkpoint_id``, ``t``,
  ``episode_id``, ``step_in_episode``, ``wallclock_ms``) stay anchored
  to position so the file's positional indexing remains canonical.
- :func:`shuffle_across_episodes` — within-episode structure preserved
  (each episode's records stay in original step order); the order of
  episodes in the output stream is permuted. Analyses that follow file
  iteration order see a different episode sequence; analyses that
  group by ``episode_id`` see unchanged content per episode.
- :func:`decouple_action_state` — per-episode action distribution
  preserved (multiset of ``action_t`` values per episode is
  byte-stable); the per-step (state, action) coupling is broken by
  permuting ``action_t`` within each episode. ``action_logprob_t`` is
  *not* moved with the action — it stays paired with the original
  state, becoming informational only after the shuffle. This is the
  explicit choice that "everything else stays in place" per the
  protocol's spec; the breakage we calibrate against is the
  state→action conditional, and shuffling only ``action_t`` is the
  minimal intervention that produces it.
- :func:`shuffle_scalar_within_trajectory` — per-trajectory marginal of
  ``self_prediction_error_t`` preserved; within-trajectory regime-
  conditional and temporal-correlation structure broken. The masked
  flag ``self_prediction_error_masked_t`` is preserved byte-identical
  in original positions — the first-step-of-episode masking convention
  is structural, not stochastic; the sentinel scalar 0.0 at masked
  positions stays. Empirical scalars at non-masked positions are
  permuted among themselves, so the (mask, scalar) pair stays
  internally consistent at every step.

Outputs:

- ``output_dir/agent_step/shard-NNNNNN.parquet`` with the same Pydantic-
  derived Arrow schema as the source (Probe 1.5 0.2.0 telemetry).
- ``output_dir/world_event.jsonl`` and ``output_dir/replay_meta.jsonl``
  copied byte-for-byte unchanged.
- ``output_dir/dream_rollout/`` shards copied byte-for-byte unchanged
  (Probe 2 does not shuffle dream telemetry).
- ``output_dir/shuffle_manifest.json`` recording protocol, seed,
  source_run_id, output_run_id, the preservation/break properties, and
  a free-text notes field describing the protocol's contract.

Determinism. Each protocol takes a ``seed`` argument; the same input
plus the same seed produces the same shuffled output. The shuffler
uses :class:`random.Random` instances seeded once per call and
consumes them in deterministic order (sorted episode ids; within each
episode, sorted by ``step_in_episode``). The per-episode iteration
order is fixed by ``sorted(by_ep.keys())`` so episode_id traversal
order is independent of dict-iteration ordering.

Out of scope at Phase 3:

- Shuffling ``dream_rollout``, ``world_event``, or ``replay_meta`` (the
  ``agent_step`` stream is the load-bearing one for criterion readings;
  the other streams are copied through so the output telemetry
  directory is a complete drop-in replacement).
- Partial / probabilistic shuffles (every protocol is a uniform
  permutation over the relevant field set).
- Modifying the schemas, the digest, or the runner. The shuffler
  consumes the schemas Phase 0 produced and produces structurally
  identical output that the Phase 2 digest reads without modification.
"""

from __future__ import annotations

import dataclasses
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import pyarrow as pa
import pyarrow.parquet as pq

from kind.observer.schemas import AgentStep
from kind.observer.sinks import _arrow_schema_from_pydantic

__all__ = [
    "SHUFFLE_MANIFEST_FILE",
    "ShuffleManifest",
    "ShuffleProtocol",
    "decouple_action_state",
    "shuffle_across_episodes",
    "shuffle_scalar_within_trajectory",
    "shuffle_within_episode",
]


SHUFFLE_MANIFEST_FILE: Final[str] = "shuffle_manifest.json"


ShuffleProtocol = Literal[
    "within_episode",
    "across_episodes",
    "decoupled_action_state",
    "scalar_within_trajectory",
]


# Field names whose values are permuted under :func:`shuffle_within_episode`.
# The remaining ``AgentStep`` fields (``schema_version``, ``run_id``,
# ``checkpoint_id``, ``t``, ``episode_id``, ``step_in_episode``,
# ``wallclock_ms``) anchor a row to its position in the file/episode and
# stay at the position-based originals so the digest's per-episode
# iteration still reads sequential ``step_in_episode`` values. The
# ``self_prediction_error_masked_t`` flag is included in the content set
# under the within-episode protocol so the (mask, scalar) pair stays
# internally consistent — the row carries (mask=True, scalar=0.0) or
# (mask=False, scalar=empirical) regardless of where the row lands; the
# step-zero masking convention is what's broken at the position level,
# not the per-row consistency.
_WITHIN_EPISODE_CONTENT_FIELDS: tuple[str, ...] = (
    "h_t",
    "q_params_t",
    "p_params_t",
    "z_t",
    "kl_per_dim_t",
    "kl_aggregate_t",
    "recon_loss_t",
    "action_t",
    "action_logprob_t",
    "policy_entropy_t",
    "obs_hash_t",
    "intrinsic_signal_t",
    "encoder_embedding_t",
    "self_prediction_t",
    "self_prediction_error_t",
    "self_prediction_error_masked_t",
)


@dataclass(frozen=True)
class ShuffleManifest:
    """Provenance record for one shuffle invocation.

    ``episode_marginals_preserved`` is True for protocols whose output
    preserves the per-episode multiset of values for the load-bearing
    field (``kl_aggregate_t`` and friends for within/across episode;
    ``action_t`` for decouple_action_state;
    ``self_prediction_error_t`` for scalar_within_trajectory).
    ``temporal_structure_broken`` is True for protocols whose output
    breaks within-episode ordering of the load-bearing field. Both
    fields are informational tags the smoke and gate tests assert on;
    they do not parameterize the shuffler's behavior.

    ``source_run_id`` is read from the first agent_step row's
    ``run_id`` field (record-level provenance). ``output_run_id`` is
    inferred from the parent directory name of ``output_dir`` under the
    ``runs/{run_id}/telemetry/`` convention; callers that pass a
    different layout get the parent name regardless and may override
    via post-construction inspection if needed.
    """

    protocol: ShuffleProtocol
    seed: int
    source_run_id: str
    output_run_id: str
    episode_marginals_preserved: bool
    temporal_structure_broken: bool
    notes: str

    def to_json(self) -> str:
        """Serialize to a byte-stable JSON string with trailing newline."""
        return (
            json.dumps(dataclasses.asdict(self), sort_keys=True, indent=2)
            + "\n"
        )


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------


def _read_agent_step_rows(telemetry_dir: Path) -> list[dict[str, Any]]:
    agent_step_dir = telemetry_dir / "agent_step"
    if not agent_step_dir.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for shard in sorted(agent_step_dir.glob("shard-*.parquet")):
        table = pq.read_table(str(shard))  # type: ignore[no-untyped-call]
        rows.extend(table.to_pylist())
    return rows


def _write_agent_step_rows(
    rows: list[dict[str, Any]], output_dir: Path
) -> None:
    """Write rows to a single shard at ``output_dir/agent_step/``.

    The Arrow schema is derived from the ``AgentStep`` Pydantic model
    via the same helper :class:`ParquetSink` uses, so the output is
    structurally indistinguishable from a freshly-written sink shard.
    A single shard file is sufficient regardless of the source's shard
    count — the digest reads via the ``shard-*.parquet`` glob and
    concatenates.
    """
    out_agent_step_dir = output_dir / "agent_step"
    out_agent_step_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    arrow_schema = _arrow_schema_from_pydantic(AgentStep)
    table = pa.Table.from_pylist(rows, schema=arrow_schema)
    shard_path = out_agent_step_dir / "shard-000000.parquet"
    with shard_path.open("wb") as fh:
        pq.write_table(table, fh)  # type: ignore[no-untyped-call]


def _copy_unchanged_streams(
    telemetry_dir: Path, output_dir: Path
) -> None:
    """Copy world_event.jsonl, replay_meta.jsonl, and dream_rollout/.

    Probe 2's shuffler is post-hoc and only touches ``agent_step``; the
    other streams are byte-copied so the output telemetry directory is
    a complete drop-in replacement for the input. Missing source files
    are skipped silently — Phase 1 fixtures may omit ``replay_meta``
    legitimately, and synthetic test fixtures may omit ``dream_rollout``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for jsonl_name in ("world_event.jsonl", "replay_meta.jsonl"):
        src = telemetry_dir / jsonl_name
        if src.is_file():
            shutil.copyfile(src, output_dir / jsonl_name)

    src_dream = telemetry_dir / "dream_rollout"
    if src_dream.is_dir():
        dst_dream = output_dir / "dream_rollout"
        dst_dream.mkdir(parents=True, exist_ok=True)
        for shard in sorted(src_dream.glob("shard-*.parquet")):
            shutil.copyfile(shard, dst_dream / shard.name)


def _group_by_episode(
    rows: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    by_ep: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_ep[int(r["episode_id"])].append(r)
    for ep in by_ep:
        by_ep[ep].sort(key=lambda r: int(r["step_in_episode"]))
    return by_ep


def _source_run_id(rows: list[dict[str, Any]], fallback: str) -> str:
    if rows:
        rid = rows[0].get("run_id")
        if isinstance(rid, str) and rid:
            return rid
    return fallback


def _output_run_id(output_dir: Path) -> str:
    """Infer the output run_id from ``output_dir``.

    Convention: ``runs/{run_id}/telemetry/`` is the layout, so the run
    id is the parent directory's name. Falls back to ``output_dir.name``
    when there is no parent (e.g., a relative path supplied as the dir
    itself).
    """
    parent_name = output_dir.parent.name
    if parent_name:
        return parent_name
    return output_dir.name


def _write_manifest(
    output_dir: Path, manifest: ShuffleManifest
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / SHUFFLE_MANIFEST_FILE).write_text(
        manifest.to_json(), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Protocols.
# ---------------------------------------------------------------------------


def shuffle_within_episode(
    telemetry_dir: Path, output_dir: Path, seed: int
) -> ShuffleManifest:
    """Within each episode, randomly permute the rows' content fields.

    The 16 content fields (h_t, z_t, kl_*, action_t, action_logprob_t,
    policy_entropy_t, obs_hash_t, intrinsic_signal_t,
    encoder_embedding_t, self_prediction_*) all move with one shared
    per-episode permutation; the 7 indexing fields (schema_version,
    run_id, checkpoint_id, t, episode_id, step_in_episode,
    wallclock_ms) stay anchored to position. Per-episode marginals (the
    multiset of values for any content field) are preserved; the
    within-episode temporal structure (lag-k correlations between
    content fields) is broken because the per-step (action, kl, sp_err,
    ...) tuples sit at random positions in step_in_episode order.
    """
    rows = _read_agent_step_rows(telemetry_dir)
    by_ep = _group_by_episode(rows)
    rng = random.Random(seed)

    out_rows: list[dict[str, Any]] = []
    for ep_id in sorted(by_ep.keys()):
        ep_rows = by_ep[ep_id]
        n = len(ep_rows)
        perm = list(range(n))
        rng.shuffle(perm)
        for k, src_idx in enumerate(perm):
            anchor = ep_rows[k]
            content = ep_rows[src_idx]
            new_row = dict(anchor)
            for field in _WITHIN_EPISODE_CONTENT_FIELDS:
                new_row[field] = content[field]
            out_rows.append(new_row)

    _write_agent_step_rows(out_rows, output_dir)
    _copy_unchanged_streams(telemetry_dir, output_dir)

    manifest = ShuffleManifest(
        protocol="within_episode",
        seed=seed,
        source_run_id=_source_run_id(rows, telemetry_dir.parent.name),
        output_run_id=_output_run_id(output_dir),
        episode_marginals_preserved=True,
        temporal_structure_broken=True,
        notes=(
            "Within each episode, the 16 content fields are permuted "
            "uniformly at random under one shared per-episode "
            "permutation; the 7 indexing fields stay anchored to "
            "position. Per-episode marginals of every content field "
            "are byte-stable; lag-k correlations within episode are "
            "broken. The step-zero masking convention does not survive "
            "at the position level — a row carrying (mask=True, "
            "scalar=0.0) may now sit at any step_in_episode within its "
            "episode — but the per-row (mask, scalar) pairing stays "
            "internally consistent."
        ),
    )
    _write_manifest(output_dir, manifest)
    return manifest


def shuffle_across_episodes(
    telemetry_dir: Path, output_dir: Path, seed: int
) -> ShuffleManifest:
    """Permute the order in which episodes appear in the output stream.

    Each episode's records stay in original ``step_in_episode`` order
    and keep their original ``episode_id``; only the file-iteration
    order of episodes changes. Analyses that read in file order see a
    different episode sequence; analyses that group by ``episode_id``
    see unchanged content per episode. This is the calibration null
    for any reading that claims an across-episode trajectory or
    progression structure (e.g., "ensemble disagreement collapses by
    episode 24") — under this shuffle the episode that was originally
    last may now be third, etc.
    """
    rows = _read_agent_step_rows(telemetry_dir)
    by_ep = _group_by_episode(rows)
    episode_ids = sorted(by_ep.keys())

    rng = random.Random(seed)
    permuted = list(episode_ids)
    rng.shuffle(permuted)

    out_rows: list[dict[str, Any]] = []
    for ep_id in permuted:
        out_rows.extend(by_ep[ep_id])

    _write_agent_step_rows(out_rows, output_dir)
    _copy_unchanged_streams(telemetry_dir, output_dir)

    manifest = ShuffleManifest(
        protocol="across_episodes",
        seed=seed,
        source_run_id=_source_run_id(rows, telemetry_dir.parent.name),
        output_run_id=_output_run_id(output_dir),
        episode_marginals_preserved=True,
        temporal_structure_broken=False,
        notes=(
            "Per-episode record set is unchanged in content; only the "
            "file-iteration order of episodes is permuted. Analyses "
            "that group by episode_id see the source's per-episode "
            "statistics exactly; analyses that follow file order see a "
            "different episode sequence. Within-episode temporal "
            "structure is preserved (each episode's records stay in "
            "original step order)."
        ),
    )
    _write_manifest(output_dir, manifest)
    return manifest


def decouple_action_state(
    telemetry_dir: Path, output_dir: Path, seed: int
) -> ShuffleManifest:
    """Within each episode, permute ``action_t`` independently.

    All other per-step fields stay in original position. The per-
    episode multiset of ``action_t`` values is preserved (so per-
    episode action distributions are byte-stable); the per-step
    (state, action) coupling is randomized, so the empirical action-
    conditional next-state distribution differs from source.

    ``action_logprob_t`` is *not* moved with ``action_t`` — it stays
    paired with the original state on which the policy computed it.
    After the shuffle the logprob no longer matches the action stored
    at the same step; the field becomes informational only. This is
    the explicit choice that "everything else stays in place" per the
    protocol's spec; the breakage we calibrate against is the
    state→action conditional, and shuffling only ``action_t`` is the
    minimal intervention that produces it.
    """
    rows = _read_agent_step_rows(telemetry_dir)
    by_ep = _group_by_episode(rows)
    rng = random.Random(seed)

    out_rows: list[dict[str, Any]] = []
    for ep_id in sorted(by_ep.keys()):
        ep_rows = by_ep[ep_id]
        actions = [int(r["action_t"]) for r in ep_rows]
        permuted_actions = list(actions)
        rng.shuffle(permuted_actions)
        for r, new_action in zip(ep_rows, permuted_actions):
            new_row = dict(r)
            new_row["action_t"] = new_action
            out_rows.append(new_row)

    _write_agent_step_rows(out_rows, output_dir)
    _copy_unchanged_streams(telemetry_dir, output_dir)

    manifest = ShuffleManifest(
        protocol="decoupled_action_state",
        seed=seed,
        source_run_id=_source_run_id(rows, telemetry_dir.parent.name),
        output_run_id=_output_run_id(output_dir),
        episode_marginals_preserved=True,
        temporal_structure_broken=False,
        notes=(
            "Within each episode, action_t values are uniformly "
            "permuted; per-episode action distribution is byte-stable. "
            "All other per-step fields (h_t, z_t, kl_aggregate_t, "
            "intrinsic_signal_t, self_prediction_*, "
            "action_logprob_t, policy_entropy_t) stay in original "
            "position; the (state, action) coupling is broken. "
            "action_logprob_t no longer matches the action stored at "
            "the same step — the field is informational only after the "
            "shuffle."
        ),
    )
    _write_manifest(output_dir, manifest)
    return manifest


def shuffle_scalar_within_trajectory(
    telemetry_dir: Path, output_dir: Path, seed: int
) -> ShuffleManifest:
    """Within each trajectory, permute non-masked ``self_prediction_error_t``.

    Each episode is one trajectory. Within a trajectory, the scalars
    at non-masked positions are permuted among themselves; the scalars
    at masked positions (the first step of each episode under the
    Probe 1.5 v2 convention) retain their sentinel value 0.0. The
    masked flag ``self_prediction_error_masked_t`` is preserved
    byte-identical against source — the first-step-of-episode masking
    convention is structural, not stochastic. The (mask, scalar) pair
    stays internally consistent at every step (mask=True implies
    scalar=0.0; mask=False implies scalar is one of the permuted
    empirical values).

    All other per-step fields stay in original position. The per-
    trajectory marginal distribution of ``self_prediction_error_t`` is
    preserved (the multiset across non-masked positions is unchanged;
    the sentinel zeros at masked positions are unchanged); the within-
    trajectory dynamics — the regime-conditional mean, the temporal
    correlation with ``kl_per_dim_t`` and the ensemble-disagreement
    variance — is broken.

    Probe 1 (``schema_version="0.1.0"``) records carry None for both
    the scalar and the masked flag; for those records the protocol is
    a no-op on the per-step scalar (the rows are still copied through
    structurally unchanged). The protocol is informative only for
    0.2.0 telemetry; the digest's behavior-side cohort against 0.1.0
    records degrades to the no-self-prediction-telemetry line
    regardless.
    """
    rows = _read_agent_step_rows(telemetry_dir)
    by_ep = _group_by_episode(rows)
    rng = random.Random(seed)

    out_rows: list[dict[str, Any]] = []
    for ep_id in sorted(by_ep.keys()):
        ep_rows = by_ep[ep_id]

        # Identify non-masked positions where the scalar is empirical.
        # Records whose masked flag or scalar is None (Probe 1 schema
        # 0.1.0) are excluded from the shuffle; their rows pass through
        # structurally unchanged.
        non_masked_idx: list[int] = []
        for i, r in enumerate(ep_rows):
            mask = r.get("self_prediction_error_masked_t")
            scalar = r.get("self_prediction_error_t")
            if mask is None or scalar is None:
                continue
            if not bool(mask):
                non_masked_idx.append(i)

        non_masked_scalars = [
            float(ep_rows[i]["self_prediction_error_t"])
            for i in non_masked_idx
        ]
        permuted_scalars = list(non_masked_scalars)
        rng.shuffle(permuted_scalars)

        new_rows = [dict(r) for r in ep_rows]
        for slot, idx in enumerate(non_masked_idx):
            new_rows[idx]["self_prediction_error_t"] = permuted_scalars[slot]
        out_rows.extend(new_rows)

    _write_agent_step_rows(out_rows, output_dir)
    _copy_unchanged_streams(telemetry_dir, output_dir)

    manifest = ShuffleManifest(
        protocol="scalar_within_trajectory",
        seed=seed,
        source_run_id=_source_run_id(rows, telemetry_dir.parent.name),
        output_run_id=_output_run_id(output_dir),
        episode_marginals_preserved=True,
        temporal_structure_broken=True,
        notes=(
            "Within each trajectory (episode), non-masked "
            "self_prediction_error_t values are uniformly permuted "
            "among themselves; self_prediction_error_masked_t is "
            "preserved byte-identical in original positions; sentinel "
            "scalars at masked positions stay 0.0. The per-trajectory "
            "marginal of the scalar is preserved; within-trajectory "
            "regime-conditional means and lag-k correlations with "
            "other fields are broken. All other per-step fields stay "
            "in original position. Probe 1 ('0.1.0') records pass "
            "through unchanged."
        ),
    )
    _write_manifest(output_dir, manifest)
    return manifest
