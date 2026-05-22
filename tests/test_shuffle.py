"""Probe 2 Phase 3 — gate test 5: shuffled-telemetry generator (four protocols).

Implementation plan §2.4 + §4 gate test 5. Exercises the four shuffle
protocols against a synthetic 0.2.0 telemetry directory:

- ``shuffle_within_episode`` preserves per-episode marginals + breaks
  within-episode lag-1 correlation between action_t and kl_aggregate_t.
- ``shuffle_across_episodes`` preserves per-episode content + permutes
  the file-iteration order of episodes.
- ``decouple_action_state`` preserves per-episode action distribution +
  breaks the empirical action-conditional next-state distribution.
- ``shuffle_scalar_within_trajectory`` preserves per-trajectory scalar
  marginal + breaks regime-conditional structure + keeps the masked
  flag byte-identical (and sentinel scalars at masked positions).

Plus determinism (same input + same seed → byte-stable row content)
and end-to-end validity (the Phase 2 hierarchical digest builds
against shuffled output without errors). Manifest provenance round-
trips through JSON.

The synthetic fixture is constructed with action_t and kl_aggregate_t
strongly lag-1-coupled so the within-episode shuffle's correlation-
break is measurable on a small number of episodes; the digest call
exercises the same parquet read path the smoke and Phase 9/12 will
use against real telemetry.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from kind.observer.digest import build_hierarchical_digest
from kind.observer.schemas import AgentStep, WorldEvent
from kind.observer.shuffle import (
    SHUFFLE_MANIFEST_FILE,
    ShuffleManifest,
    decouple_action_state,
    shuffle_across_episodes,
    shuffle_scalar_within_trajectory,
    shuffle_within_episode,
)
from kind.observer.sinks import ParquetSink


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _build_synthetic_telemetry(
    telemetry_dir: Path,
    *,
    n_episodes: int = 8,
    n_steps: int = 12,
    h_dim: int = 4,
    z_dim: int = 4,
    run_id: str = "synthetic-shuffle",
) -> Path:
    """Build a 0.2.0 telemetry directory designed to expose the four
    protocols' break/preserve properties.

    Per-episode action follows a cyclic pattern keyed by episode index
    (so different episodes have different action orderings); per-step
    ``kl_aggregate_t`` is constructed to be strongly correlated with
    the previous step's action so that within-episode lag-1 correlation
    between ``action_t`` and ``kl_aggregate_t`` is non-trivial in the
    source. Step 0 of each episode is masked (``self_prediction_error_t
    = 0.0`` sentinel; ``self_prediction_error_masked_t = True``).
    Non-masked self-prediction errors vary with action_t so a regime-
    conditional mean of the scalar (proxied by the top-half of
    ``intrinsic_signal_t``) is non-trivial in the source.
    """
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    sink = ParquetSink(
        telemetry_dir / "agent_step", AgentStep, rows_per_shard=10_000
    )
    t = 0
    prev_action_per_ep: dict[int, int] = {}
    for ep in range(n_episodes):
        for s in range(n_steps):
            masked = s == 0
            # Per-episode cyclic action; varies by episode so per-
            # episode action distributions differ across episodes.
            action = (s + ep * 2) % 5
            # KL strongly lag-1 coupled to the previous step's action;
            # at s=0 there is no prior, so use a baseline.
            if s == 0:
                kl = 0.5 + 0.001 * ep
            else:
                prev = prev_action_per_ep.get(ep, action)
                kl = 0.5 + 0.05 * prev + 0.001 * ep
            # Intrinsic signal varies within episode so regime
            # classification (top half by intrinsic) splits non-
            # trivially. Coupled to action so regime-conditional sp_err
            # mean is non-trivial.
            intrinsic = 0.05 + 0.005 * action + 0.001 * s
            scalar = 0.0 if masked else 0.05 + 0.02 * action - 0.005 * s
            record = AgentStep(
                schema_version="0.2.0",
                run_id=run_id,
                checkpoint_id=None,
                t=t,
                episode_id=ep,
                step_in_episode=s,
                wallclock_ms=1_000_000 + t,
                h_t=[0.1 * (i + 1) + 0.001 * t for i in range(h_dim)],
                q_params_t=([0.0] * z_dim, [1.0] * z_dim),
                p_params_t=([0.0] * z_dim, [1.0] * z_dim),
                z_t=[0.5 + 0.01 * s] * z_dim,
                kl_per_dim_t=[
                    0.05 + 0.01 * (i + s) for i in range(z_dim)
                ],
                kl_aggregate_t=kl,
                recon_loss_t=1.0 - 0.01 * s,
                action_t=action,
                action_logprob_t=-1.6 - 0.01 * s,
                policy_entropy_t=1.5 - 0.01 * s,
                obs_hash_t=f"h-{ep:02d}-{s:02d}",
                intrinsic_signal_t=intrinsic,
                encoder_embedding_t=[0.0] * h_dim,
                self_prediction_t=[
                    0.05 * (i + 1) + 0.01 * s for i in range(h_dim)
                ],
                self_prediction_error_t=scalar,
                self_prediction_error_masked_t=masked,
            )
            sink.write(record)
            prev_action_per_ep[ep] = action
            t += 1
    sink.close()

    # World-event timeline: env_reset per episode. The shuffle's
    # contract on world_event.jsonl is byte-copy-through; the digest
    # reads these to populate the world-event timeline. Validating
    # through ``WorldEvent`` ensures we emit JSON that round-trips.
    world_events: list[dict[str, Any]] = []
    for ep in range(n_episodes):
        world_events.append(
            {
                "schema_version": "0.1.0",
                "run_id": run_id,
                "checkpoint_id": None,
                "t_event": ep * n_steps,
                "event_type": "env_reset",
                "source": "environment",
                "payload": {"episode_id": ep, "start_cell": [3, 3]},
                "wallclock_ms": 1_000_000 + ep * n_steps,
            }
        )
    we_path = telemetry_dir / "world_event.jsonl"
    with we_path.open("w", encoding="utf-8") as fh:
        for ev in world_events:
            wm = WorldEvent.model_validate(ev)
            fh.write(wm.model_dump_json() + "\n")

    # Replay meta: empty-but-present file matches the convention some
    # runners use (no replay events to log yet); the shuffle copies it.
    (telemetry_dir / "replay_meta.jsonl").write_text("")

    return telemetry_dir


def _read_rows(telemetry_dir: Path) -> list[dict[str, Any]]:
    agent_step_dir = telemetry_dir / "agent_step"
    rows: list[dict[str, Any]] = []
    for shard in sorted(agent_step_dir.glob("shard-*.parquet")):
        rows.extend(pq.read_table(str(shard)).to_pylist())
    return rows


def _group_by_ep(
    rows: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    by_ep: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_ep.setdefault(int(r["episode_id"]), []).append(r)
    for ep in by_ep:
        by_ep[ep].sort(key=lambda r: int(r["step_in_episode"]))
    return by_ep


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def _action_kl_lag1_corr(
    by_ep: dict[int, list[dict[str, Any]]],
) -> float:
    """Lag-1 correlation between action_t and kl_aggregate_t pooled across episodes."""
    a_pairs: list[float] = []
    k_pairs: list[float] = []
    for ep_rows in by_ep.values():
        for k in range(len(ep_rows) - 1):
            a_pairs.append(float(ep_rows[k]["action_t"]))
            k_pairs.append(float(ep_rows[k + 1]["kl_aggregate_t"]))
    return _pearson(a_pairs, k_pairs)


# ---------------------------------------------------------------------------
# within_episode protocol.
# ---------------------------------------------------------------------------


def test_within_episode_preserves_per_episode_marginals(tmp_path: Path) -> None:
    src = _build_synthetic_telemetry(tmp_path / "src" / "telemetry")
    out = tmp_path / "out_within" / "telemetry"
    manifest = shuffle_within_episode(src, out, seed=1234)

    assert manifest.protocol == "within_episode"
    assert manifest.episode_marginals_preserved is True
    assert manifest.temporal_structure_broken is True
    assert manifest.seed == 1234

    src_by_ep = _group_by_ep(_read_rows(src))
    out_by_ep = _group_by_ep(_read_rows(out))

    assert set(src_by_ep.keys()) == set(out_by_ep.keys())

    for ep in src_by_ep:
        # Marginal multiset of every content field is preserved.
        src_kl = sorted(float(r["kl_aggregate_t"]) for r in src_by_ep[ep])
        out_kl = sorted(float(r["kl_aggregate_t"]) for r in out_by_ep[ep])
        assert src_kl == out_kl
        assert _mean(src_kl) == pytest.approx(_mean(out_kl))
        assert _std(src_kl) == pytest.approx(_std(out_kl))

        # Action multiset preserved.
        src_actions = sorted(int(r["action_t"]) for r in src_by_ep[ep])
        out_actions = sorted(int(r["action_t"]) for r in out_by_ep[ep])
        assert src_actions == out_actions

        # Self-prediction error multiset preserved.
        src_sp = sorted(
            float(r["self_prediction_error_t"]) for r in src_by_ep[ep]
        )
        out_sp = sorted(
            float(r["self_prediction_error_t"]) for r in out_by_ep[ep]
        )
        assert src_sp == out_sp


def test_within_episode_breaks_temporal_structure(tmp_path: Path) -> None:
    src = _build_synthetic_telemetry(tmp_path / "src" / "telemetry")
    out = tmp_path / "out_within" / "telemetry"
    shuffle_within_episode(src, out, seed=42)

    src_corr = _action_kl_lag1_corr(_group_by_ep(_read_rows(src)))
    out_corr = _action_kl_lag1_corr(_group_by_ep(_read_rows(out)))

    # Source's action and kl are strongly lag-1-coupled by construction;
    # the shuffle drops the absolute correlation by at least 0.3 (a
    # generous bracket against small-fixture noise; the actual drop is
    # typically > 0.6).
    assert abs(src_corr) - abs(out_corr) > 0.3, (
        f"src_corr={src_corr:.3f}, out_corr={out_corr:.3f}: shuffle did "
        f"not measurably break the lag-1 correlation"
    )


def test_within_episode_preserves_indexing_fields(tmp_path: Path) -> None:
    """``step_in_episode``, ``t``, ``wallclock_ms``, ``episode_id``,
    ``run_id``, ``schema_version``, ``checkpoint_id`` stay anchored to
    position so the file's positional indexing remains canonical."""
    src = _build_synthetic_telemetry(tmp_path / "src" / "telemetry")
    out = tmp_path / "out_within" / "telemetry"
    shuffle_within_episode(src, out, seed=777)

    src_rows = sorted(
        _read_rows(src),
        key=lambda r: (int(r["episode_id"]), int(r["step_in_episode"])),
    )
    out_rows = sorted(
        _read_rows(out),
        key=lambda r: (int(r["episode_id"]), int(r["step_in_episode"])),
    )
    assert len(src_rows) == len(out_rows)
    for s, o in zip(src_rows, out_rows):
        for field in (
            "schema_version",
            "run_id",
            "checkpoint_id",
            "t",
            "episode_id",
            "step_in_episode",
            "wallclock_ms",
        ):
            assert s[field] == o[field], (
                f"indexing field {field!r} drifted at "
                f"episode={s['episode_id']} step={s['step_in_episode']}"
            )


# ---------------------------------------------------------------------------
# across_episodes protocol.
# ---------------------------------------------------------------------------


def test_across_episodes_preserves_per_episode_content(tmp_path: Path) -> None:
    src = _build_synthetic_telemetry(tmp_path / "src" / "telemetry")
    out = tmp_path / "out_across" / "telemetry"
    manifest = shuffle_across_episodes(src, out, seed=2026)

    assert manifest.protocol == "across_episodes"
    assert manifest.episode_marginals_preserved is True
    assert manifest.temporal_structure_broken is False

    src_by_ep = _group_by_ep(_read_rows(src))
    out_by_ep = _group_by_ep(_read_rows(out))

    assert set(src_by_ep.keys()) == set(out_by_ep.keys())
    for ep in src_by_ep:
        # Each episode's records are byte-identical in content (we read
        # back as dicts; comparing dict equality is robust because
        # parquet round-trips Python primitives).
        assert src_by_ep[ep] == out_by_ep[ep], (
            f"episode {ep}: content differs between source and shuffled"
        )


def test_across_episodes_breaks_file_iteration_ordering(tmp_path: Path) -> None:
    src = _build_synthetic_telemetry(tmp_path / "src" / "telemetry")
    out = tmp_path / "out_across" / "telemetry"
    shuffle_across_episodes(src, out, seed=2026)

    src_rows = _read_rows(src)
    out_rows = _read_rows(out)

    # First-occurrence sequence of episode_ids in file iteration order.
    def first_occurrence_seq(rows: list[dict[str, Any]]) -> list[int]:
        seen: list[int] = []
        for r in rows:
            eid = int(r["episode_id"])
            if eid not in seen:
                seen.append(eid)
        return seen

    src_seq = first_occurrence_seq(src_rows)
    out_seq = first_occurrence_seq(out_rows)
    assert src_seq != out_seq, (
        f"episode_id sequence unchanged: src={src_seq}, out={out_seq}"
    )
    assert sorted(src_seq) == sorted(out_seq)


# ---------------------------------------------------------------------------
# decouple_action_state protocol.
# ---------------------------------------------------------------------------


def test_decouple_preserves_per_episode_action_distribution(tmp_path: Path) -> None:
    src = _build_synthetic_telemetry(tmp_path / "src" / "telemetry")
    out = tmp_path / "out_decouple" / "telemetry"
    manifest = decouple_action_state(src, out, seed=7)

    assert manifest.protocol == "decoupled_action_state"

    src_by_ep = _group_by_ep(_read_rows(src))
    out_by_ep = _group_by_ep(_read_rows(out))

    for ep in src_by_ep:
        src_actions = sorted(int(r["action_t"]) for r in src_by_ep[ep])
        out_actions = sorted(int(r["action_t"]) for r in out_by_ep[ep])
        assert src_actions == out_actions


def test_decouple_breaks_action_conditional_next_state(tmp_path: Path) -> None:
    src = _build_synthetic_telemetry(tmp_path / "src" / "telemetry")
    out = tmp_path / "out_decouple" / "telemetry"
    decouple_action_state(src, out, seed=7)

    src_by_ep = _group_by_ep(_read_rows(src))
    out_by_ep = _group_by_ep(_read_rows(out))

    # Empirical mean of next-step kl_aggregate_t conditional on
    # action_t, pooled across episodes. Source has the strong lag-1
    # coupling between action and next-step kl; after decoupling the
    # conditional mean structure should differ.
    def cond_means(by_ep: dict[int, list[dict[str, Any]]]) -> dict[int, float]:
        bucket: dict[int, list[float]] = {}
        for ep_rows in by_ep.values():
            for k in range(len(ep_rows) - 1):
                bucket.setdefault(int(ep_rows[k]["action_t"]), []).append(
                    float(ep_rows[k + 1]["kl_aggregate_t"])
                )
        return {a: _mean(vs) for a, vs in bucket.items() if vs}

    src_cond = cond_means(src_by_ep)
    out_cond = cond_means(out_by_ep)
    common = set(src_cond) & set(out_cond)
    diffs = [abs(src_cond[a] - out_cond[a]) for a in common]
    assert max(diffs) > 1e-6, (
        f"action-conditional next-state mean unchanged: "
        f"src={src_cond}, out={out_cond}"
    )


def test_decouple_leaves_state_fields_in_position(tmp_path: Path) -> None:
    """``decouple_action_state`` only touches ``action_t`` — every
    other per-step field stays in original position."""
    src = _build_synthetic_telemetry(tmp_path / "src" / "telemetry")
    out = tmp_path / "out_decouple" / "telemetry"
    decouple_action_state(src, out, seed=7)

    src_rows = sorted(
        _read_rows(src),
        key=lambda r: (int(r["episode_id"]), int(r["step_in_episode"])),
    )
    out_rows = sorted(
        _read_rows(out),
        key=lambda r: (int(r["episode_id"]), int(r["step_in_episode"])),
    )
    assert len(src_rows) == len(out_rows)
    for s, o in zip(src_rows, out_rows):
        # Every non-action_t field is byte-identical at the same
        # (episode_id, step_in_episode) position.
        for field in s.keys():
            if field == "action_t":
                continue
            assert s[field] == o[field], (
                f"non-action field {field!r} drifted at episode="
                f"{s['episode_id']} step={s['step_in_episode']}"
            )


# ---------------------------------------------------------------------------
# scalar_within_trajectory protocol.
# ---------------------------------------------------------------------------


def test_scalar_shuffle_preserves_per_trajectory_marginal(tmp_path: Path) -> None:
    src = _build_synthetic_telemetry(tmp_path / "src" / "telemetry")
    out = tmp_path / "out_scalar" / "telemetry"
    manifest = shuffle_scalar_within_trajectory(src, out, seed=11)

    assert manifest.protocol == "scalar_within_trajectory"
    assert manifest.episode_marginals_preserved is True
    assert manifest.temporal_structure_broken is True

    src_by_ep = _group_by_ep(_read_rows(src))
    out_by_ep = _group_by_ep(_read_rows(out))

    for ep in src_by_ep:
        src_scalars = sorted(
            float(r["self_prediction_error_t"]) for r in src_by_ep[ep]
        )
        out_scalars = sorted(
            float(r["self_prediction_error_t"]) for r in out_by_ep[ep]
        )
        assert src_scalars == out_scalars
        assert _mean(src_scalars) == pytest.approx(_mean(out_scalars))
        assert _std(src_scalars) == pytest.approx(_std(out_scalars))


def test_scalar_shuffle_breaks_regime_conditional_mean(tmp_path: Path) -> None:
    """The empirical regime-conditional mean of self_prediction_error_t
    differs between source and shuffled. The regime proxy is the top
    half of intrinsic_signal_t per episode (non-masked rows only); the
    scalar in the source is constructed to vary with action which
    correlates with intrinsic, so the source has a non-trivial
    regime-conditional mean. After shuffle, the scalar is independent
    of intrinsic within each trajectory; the regime-conditional mean
    drifts toward the per-trajectory marginal mean."""
    src = _build_synthetic_telemetry(tmp_path / "src" / "telemetry")
    out = tmp_path / "out_scalar" / "telemetry"
    shuffle_scalar_within_trajectory(src, out, seed=11)

    def regime_high_scalar_mean(
        by_ep: dict[int, list[dict[str, Any]]],
    ) -> float:
        vals: list[float] = []
        for ep_rows in by_ep.values():
            non_masked = [
                r
                for r in ep_rows
                if not bool(r.get("self_prediction_error_masked_t"))
            ]
            if len(non_masked) < 2:
                continue
            sorted_by_intrinsic = sorted(
                non_masked,
                key=lambda r: float(r["intrinsic_signal_t"]),
                reverse=True,
            )
            top_half = sorted_by_intrinsic[: len(sorted_by_intrinsic) // 2]
            for r in top_half:
                vals.append(float(r["self_prediction_error_t"]))
        return _mean(vals)

    src_high = regime_high_scalar_mean(_group_by_ep(_read_rows(src)))
    out_high = regime_high_scalar_mean(_group_by_ep(_read_rows(out)))
    assert abs(src_high - out_high) > 1e-6, (
        f"regime-conditional scalar mean unchanged after shuffle: "
        f"src={src_high}, out={out_high}"
    )


def test_scalar_shuffle_preserves_masked_flag_byte_identical(
    tmp_path: Path,
) -> None:
    src = _build_synthetic_telemetry(tmp_path / "src" / "telemetry")
    out = tmp_path / "out_scalar" / "telemetry"
    shuffle_scalar_within_trajectory(src, out, seed=11)

    src_rows = sorted(
        _read_rows(src),
        key=lambda r: (int(r["episode_id"]), int(r["step_in_episode"])),
    )
    out_rows = sorted(
        _read_rows(out),
        key=lambda r: (int(r["episode_id"]), int(r["step_in_episode"])),
    )
    assert len(src_rows) == len(out_rows)
    for s, o in zip(src_rows, out_rows):
        # Masked flag stays byte-identical in original positions.
        assert s["self_prediction_error_masked_t"] == o[
            "self_prediction_error_masked_t"
        ], (
            f"masked flag drifted at episode={s['episode_id']} "
            f"step={s['step_in_episode']}: "
            f"src={s['self_prediction_error_masked_t']} "
            f"out={o['self_prediction_error_masked_t']}"
        )
        # Sentinel scalar at masked positions stays 0.0; only non-
        # masked positions carry permuted values.
        if bool(s["self_prediction_error_masked_t"]):
            assert float(o["self_prediction_error_t"]) == 0.0


def test_scalar_shuffle_leaves_other_fields_in_position(tmp_path: Path) -> None:
    """Only ``self_prediction_error_t`` at non-masked positions moves;
    every other per-step field stays byte-identical."""
    src = _build_synthetic_telemetry(tmp_path / "src" / "telemetry")
    out = tmp_path / "out_scalar" / "telemetry"
    shuffle_scalar_within_trajectory(src, out, seed=11)

    src_rows = sorted(
        _read_rows(src),
        key=lambda r: (int(r["episode_id"]), int(r["step_in_episode"])),
    )
    out_rows = sorted(
        _read_rows(out),
        key=lambda r: (int(r["episode_id"]), int(r["step_in_episode"])),
    )
    for s, o in zip(src_rows, out_rows):
        for field in s.keys():
            if field == "self_prediction_error_t":
                continue
            assert s[field] == o[field], (
                f"non-scalar field {field!r} drifted at "
                f"episode={s['episode_id']} step={s['step_in_episode']}"
            )


# ---------------------------------------------------------------------------
# Determinism + structural validity + manifest provenance.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shuffle_fn,protocol",
    [
        (shuffle_within_episode, "within_episode"),
        (shuffle_across_episodes, "across_episodes"),
        (decouple_action_state, "decoupled_action_state"),
        (shuffle_scalar_within_trajectory, "scalar_within_trajectory"),
    ],
)
def test_determinism_byte_stable_under_same_seed(
    tmp_path: Path, shuffle_fn: Any, protocol: str
) -> None:
    """Same input + same seed → byte-stable row content across two
    invocations. The contract is row-content stability under a stable
    sort by (episode_id, step_in_episode); pyarrow's parquet writer's
    bytes are not part of the public contract (compression/metadata
    may include version-dependent fields), but the row content the
    digest reads is the load-bearing surface."""
    src = _build_synthetic_telemetry(tmp_path / "src" / "telemetry")
    out_a = tmp_path / f"out_{protocol}_a" / "telemetry"
    out_b = tmp_path / f"out_{protocol}_b" / "telemetry"
    shuffle_fn(src, out_a, seed=2024)
    shuffle_fn(src, out_b, seed=2024)

    rows_a = sorted(
        _read_rows(out_a),
        key=lambda r: (int(r["episode_id"]), int(r["step_in_episode"])),
    )
    rows_b = sorted(
        _read_rows(out_b),
        key=lambda r: (int(r["episode_id"]), int(r["step_in_episode"])),
    )
    assert rows_a == rows_b


@pytest.mark.parametrize(
    "shuffle_fn,protocol",
    [
        (shuffle_within_episode, "within_episode"),
        (shuffle_across_episodes, "across_episodes"),
        (decouple_action_state, "decoupled_action_state"),
        (shuffle_scalar_within_trajectory, "scalar_within_trajectory"),
    ],
)
def test_protocol_output_is_digest_buildable(
    tmp_path: Path, shuffle_fn: Any, protocol: str
) -> None:
    """The Phase 2 hierarchical digest builds against shuffled output
    without errors. This is the load-bearing structural-validity gate
    — if the shuffled output's parquet schema or world_event ordering
    drifts, the digest catches it."""
    src = _build_synthetic_telemetry(tmp_path / "src" / "telemetry")
    out = tmp_path / f"out_{protocol}" / "telemetry"
    manifest = shuffle_fn(src, out, seed=99)

    # Manifest landed on disk with the expected protocol tag.
    manifest_path = out / SHUFFLE_MANIFEST_FILE
    assert manifest_path.is_file()
    loaded = json.loads(manifest_path.read_text())
    assert loaded["protocol"] == protocol
    assert loaded["seed"] == 99

    # Digest builds against the shuffled telemetry.
    hd = build_hierarchical_digest(out, n_episodes=4)
    assert hd.run_summary.startswith("# Telemetry Run Summary")
    assert hd.episode_mini_digests
    # The digest should still see the world_event timeline (copied
    # through unchanged) populated with env_reset events.
    assert "env_reset" in hd.world_event_timeline


def test_manifest_records_provenance_and_round_trips(tmp_path: Path) -> None:
    src = _build_synthetic_telemetry(tmp_path / "myrun" / "telemetry")
    out = tmp_path / "myrun_shuffled_within_episode" / "telemetry"
    manifest = shuffle_within_episode(src, out, seed=42)

    assert isinstance(manifest, ShuffleManifest)
    assert manifest.protocol == "within_episode"
    assert manifest.seed == 42
    assert manifest.source_run_id == "synthetic-shuffle"
    assert manifest.output_run_id == "myrun_shuffled_within_episode"
    assert manifest.episode_marginals_preserved is True
    assert manifest.temporal_structure_broken is True
    assert manifest.notes  # non-empty

    # Round-trip through JSONL.
    on_disk = json.loads((out / SHUFFLE_MANIFEST_FILE).read_text())
    assert on_disk["protocol"] == "within_episode"
    assert on_disk["seed"] == 42
    assert on_disk["source_run_id"] == "synthetic-shuffle"
    assert on_disk["output_run_id"] == "myrun_shuffled_within_episode"
    assert on_disk["episode_marginals_preserved"] is True
    assert on_disk["temporal_structure_broken"] is True


def test_world_event_and_replay_meta_copied_unchanged(tmp_path: Path) -> None:
    src = _build_synthetic_telemetry(tmp_path / "src" / "telemetry")
    out = tmp_path / "out" / "telemetry"
    shuffle_within_episode(src, out, seed=1)

    src_we = (src / "world_event.jsonl").read_bytes()
    out_we = (out / "world_event.jsonl").read_bytes()
    assert src_we == out_we

    src_rm = (src / "replay_meta.jsonl").read_bytes()
    out_rm = (out / "replay_meta.jsonl").read_bytes()
    assert src_rm == out_rm


def test_dream_rollout_copied_unchanged(tmp_path: Path) -> None:
    """When a source dream_rollout/ directory exists, the shuffler
    copies its shards byte-for-byte; when it does not exist (the
    synthetic fixture omits it), the output simply does not have a
    dream_rollout/ directory and downstream readers tolerate the
    absence."""
    src = _build_synthetic_telemetry(tmp_path / "src" / "telemetry")
    # Add a synthetic dream_rollout/ shard to the source so the copy
    # path is exercised. The shard's content is opaque to the shuffle
    # — any byte sequence would do; we use a small valid parquet file
    # by way of pyarrow.
    import pyarrow as pa

    dream_dir = src / "dream_rollout"
    dream_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table({"x": [1, 2, 3]})
    shard = dream_dir / "shard-000000.parquet"
    with shard.open("wb") as fh:
        pq.write_table(table, fh)

    out = tmp_path / "out" / "telemetry"
    shuffle_within_episode(src, out, seed=1)

    out_shard = out / "dream_rollout" / "shard-000000.parquet"
    assert out_shard.is_file()
    assert out_shard.read_bytes() == shard.read_bytes()


def test_shuffle_with_no_replay_meta_present(tmp_path: Path) -> None:
    """If the source has no replay_meta.jsonl, the shuffler does not
    fabricate one — the output omits the file. The digest tolerates
    its absence."""
    src = _build_synthetic_telemetry(tmp_path / "src" / "telemetry")
    (src / "replay_meta.jsonl").unlink()
    out = tmp_path / "out" / "telemetry"
    shuffle_within_episode(src, out, seed=1)
    assert not (out / "replay_meta.jsonl").is_file()
    # Digest still builds.
    hd = build_hierarchical_digest(out, n_episodes=2)
    assert hd.run_summary
