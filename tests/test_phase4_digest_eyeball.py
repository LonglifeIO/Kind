"""Phase 4 tests for the Probe 1.5 digest + eyeball self-prediction surface.

Plan §2.6 (digest) and §2.7 (eyeball). Eight specific tests the build
prompt names plus the CLI subcommand checks plus a non-defaulted
``show_episode_summary`` extension check. The Probe 1 reference run at
``runs/probe1-20260503-123926/`` is the on-disk no-self-prediction
fixture; synthetic 0.2.0 / mixed-version rows + a synthetic checkpoint
exercise the new code paths the runner will produce post-Phase-3.

The synthetic 0.2.0 run for the conditioning helper is built by:

* writing 0.2.0 ``AgentStep`` records into a Parquet shard under
  ``<run>/telemetry/agent_step/`` via the real ``ParquetSink``,
* writing a sham ``world_event.jsonl`` with a known
  ``builder_perturbation`` event for the perturbation-window regime,
* constructing a real ``Actor`` (the v2 input layer with the extended
  column) and saving its weights into a synthetic
  ``weights.safetensors`` blob under
  ``<run>/checkpoints/<id>/weights.safetensors`` with the canonical
  ``actor.*`` prefix the runner uses.

The test does not exercise the runner — that's gate test #5's
territory (``tests/test_integration_smoke_probe1_5.py``). Phase 4 is
the read-side analyzer surface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
import torch
from safetensors.torch import save_file as _save_safetensors

from kind.agents.actor import Actor
from kind.observer.digest import build_digest, compact_record_repr
from kind.observer.eyeball import (
    main,
    show_episode_summary,
    show_self_prediction,
    show_self_prediction_conditioning,
)
from kind.observer.schemas import (
    PROBE_1_SCHEMA_VERSION,
    SCHEMA_VERSION,
    AgentStep,
)
from kind.observer.sinks import ParquetSink


REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE_1_RUN_DIR = REPO_ROOT / "runs" / "probe1-20260503-123926"
PROBE_1_TELEMETRY_DIR = PROBE_1_RUN_DIR / "telemetry"


# ---- factories -----------------------------------------------------------


_H_DIM = 4
_Z_DIM = 2


def _agent_step_v0_1_0(*, t: int, episode_id: int, step_in_episode: int) -> AgentStep:
    """Probe 1-shaped record: 0.1.0, all three new fields None."""
    return AgentStep(
        schema_version=PROBE_1_SCHEMA_VERSION,
        run_id="test-run",
        checkpoint_id=None,
        t=t,
        episode_id=episode_id,
        step_in_episode=step_in_episode,
        wallclock_ms=1_000_000 + t,
        h_t=[float(i) + 0.1 * t for i in range(_H_DIM)],
        q_params_t=([0.0] * _Z_DIM, [0.0] * _Z_DIM),
        p_params_t=([0.0] * _Z_DIM, [0.0] * _Z_DIM),
        z_t=[0.5] * _Z_DIM,
        kl_per_dim_t=[0.05] * _Z_DIM,
        kl_aggregate_t=0.5 + 0.01 * t,
        recon_loss_t=1.0,
        action_t=t % 5,
        action_logprob_t=-1.6,
        policy_entropy_t=1.5,
        obs_hash_t=f"hash-{t}",
        intrinsic_signal_t=0.3 + 0.005 * t,
        encoder_embedding_t=[0.0] * _H_DIM,
    )


def _agent_step_v0_2_0(
    *,
    t: int,
    episode_id: int,
    step_in_episode: int,
    masked: bool,
    self_prediction_error: float,
    self_prediction_vector: list[float] | None = None,
    h_t: list[float] | None = None,
    intrinsic_signal_t: float = 0.3,
    kl_aggregate_t: float = 0.5,
) -> AgentStep:
    """Probe 1.5-shaped record: 0.2.0, all three new fields populated.

    Masked=True clamps the scalar to 0.0 (sentinel) per the Phase 0
    masking convention. The vector defaults to a per-dim ramp so the
    per-dim residual variance test has a non-degenerate signal across
    consecutive steps.
    """
    if masked:
        scalar = 0.0
    else:
        scalar = self_prediction_error
    if self_prediction_vector is None:
        self_prediction_vector = [float(i) * 0.1 + 0.01 * t for i in range(_H_DIM)]
    if h_t is None:
        h_t = [float(i) + 0.1 * t for i in range(_H_DIM)]
    return AgentStep(
        schema_version=SCHEMA_VERSION,
        run_id="test-run",
        checkpoint_id=None,
        t=t,
        episode_id=episode_id,
        step_in_episode=step_in_episode,
        wallclock_ms=1_000_000 + t,
        h_t=h_t,
        q_params_t=([0.0] * _Z_DIM, [0.0] * _Z_DIM),
        p_params_t=([0.0] * _Z_DIM, [0.0] * _Z_DIM),
        z_t=[0.5] * _Z_DIM,
        kl_per_dim_t=[0.05] * _Z_DIM,
        kl_aggregate_t=kl_aggregate_t,
        recon_loss_t=1.0,
        action_t=t % 5,
        action_logprob_t=-1.6,
        policy_entropy_t=1.5,
        obs_hash_t=f"hash-{t}",
        intrinsic_signal_t=intrinsic_signal_t,
        encoder_embedding_t=[0.0] * _H_DIM,
        self_prediction_t=self_prediction_vector,
        self_prediction_error_t=scalar,
        self_prediction_error_masked_t=masked,
    )


def _write_agent_steps(records: list[AgentStep], telemetry_dir: Path) -> None:
    sink = ParquetSink(telemetry_dir / "agent_step", AgentStep, rows_per_shard=10_000)
    for r in records:
        sink.write(r)
    sink.close()


def _read_pq_rows(telemetry_dir: Path) -> list[dict[str, Any]]:
    """Read all agent_step shards under telemetry_dir as dicts."""
    out: list[dict[str, Any]] = []
    shards = sorted((telemetry_dir / "agent_step").glob("shard-*.parquet"))
    for shard in shards:
        out.extend(pq.read_table(str(shard)).to_pylist())
    return out


def _build_synthetic_v0_2_0_episode(
    *,
    episode_id: int,
    n_steps: int,
    scalar_pattern: list[float] | None = None,
    starting_t: int = 0,
) -> list[AgentStep]:
    """Construct one episode's worth of 0.2.0 records.

    The first step (``step_in_episode == 0``) is masked per the Phase 0
    convention. Subsequent steps carry the empirical scalar.
    """
    records: list[AgentStep] = []
    for s in range(n_steps):
        masked = s == 0
        scalar = (
            scalar_pattern[s] if scalar_pattern is not None else 0.1 * s
        )
        records.append(
            _agent_step_v0_2_0(
                t=starting_t + s,
                episode_id=episode_id,
                step_in_episode=s,
                masked=masked,
                self_prediction_error=scalar,
            )
        )
    return records


def _build_synthetic_run_dir(
    tmp_path: Path,
    *,
    h_dim: int = 8,
    z_dim: int = 4,
    action_dim: int = 5,
    mlp_hidden: int = 16,
    schema_version: str = SCHEMA_VERSION,
    n_steps_per_episode: int = 30,
    n_episodes: int = 2,
    perturbation_t_event: int | None = None,
) -> tuple[Path, str]:
    """Build a self-contained run directory for ``show_self_prediction_conditioning``.

    Layout matches what ``Runner._commit_checkpoint`` writes:

        <run_dir>/
          telemetry/
            agent_step/shard-*.parquet
            world_event.jsonl
          checkpoints/
            ckpt-000001/
              weights.safetensors  (actor.* keys)
              schema_version.txt

    Returns ``(run_dir, checkpoint_id)``. The actor's weights are a
    fresh Probe 1.5-shaped Actor (input layer ``h_dim + z_dim + 1``;
    new column zero-initialized) — perturbing the scalar therefore
    starts byte-equivalent to ignoring the scalar (zero-init); the
    KL between unperturbed / perturbed will be exactly zero in that
    case. To exercise non-zero KL the test perturbs the new column
    away from zero before saving (so the helper has a column the
    scalar can move).
    """
    run_dir = tmp_path / "synthetic-run"
    telemetry_dir = run_dir / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    # Build the agent_step shards. Two episodes; the first step of
    # each is masked.
    all_records: list[AgentStep] = []
    starting_t = 0
    for ep in range(n_episodes):
        # Use larger h_dim/z_dim than the digest tests' default _H_DIM/_Z_DIM
        # because the conditioning helper reads ``h_t`` / ``z_t`` from the
        # rows and feeds them into the Actor's forward.
        for s in range(n_steps_per_episode):
            masked = s == 0
            scalar = 0.0 if masked else 0.1 + 0.01 * s
            kl = 0.5 + 0.001 * starting_t
            intr = 0.3 + 0.001 * s
            record = AgentStep(
                schema_version=schema_version,
                run_id="synthetic",
                checkpoint_id=None,
                t=starting_t,
                episode_id=ep,
                step_in_episode=s,
                wallclock_ms=1_000_000 + starting_t,
                h_t=[0.1 * (i + 1) + 0.001 * starting_t for i in range(h_dim)],
                q_params_t=([0.0] * z_dim, [0.0] * z_dim),
                p_params_t=([0.0] * z_dim, [0.0] * z_dim),
                z_t=[0.5] * z_dim,
                kl_per_dim_t=[kl / z_dim] * z_dim,
                kl_aggregate_t=kl,
                recon_loss_t=1.0,
                action_t=starting_t % action_dim,
                action_logprob_t=-1.6,
                policy_entropy_t=1.5,
                obs_hash_t=f"hash-{starting_t}",
                intrinsic_signal_t=intr,
                encoder_embedding_t=[0.0] * h_dim,
                self_prediction_t=(
                    None
                    if schema_version == PROBE_1_SCHEMA_VERSION
                    else [0.05 * (i + 1) for i in range(h_dim)]
                ),
                self_prediction_error_t=(
                    None if schema_version == PROBE_1_SCHEMA_VERSION else scalar
                ),
                self_prediction_error_masked_t=(
                    None if schema_version == PROBE_1_SCHEMA_VERSION else masked
                ),
            )
            all_records.append(record)
            starting_t += 1
    _write_agent_steps(all_records, telemetry_dir)

    # world_event.jsonl with one builder_perturbation event (so the
    # perturbation_window regime has at least some classified states).
    we_path = telemetry_dir / "world_event.jsonl"
    if perturbation_t_event is not None:
        we_path.write_text(
            json.dumps(
                {
                    "schema_version": schema_version,
                    "run_id": "synthetic",
                    "checkpoint_id": None,
                    "t_event": perturbation_t_event,
                    "event_type": "builder_perturbation",
                    "source": "builder",
                    "payload": {},
                    "wallclock_ms": 1_000_000 + perturbation_t_event,
                }
            )
            + "\n"
        )
    else:
        we_path.write_text("")

    # Build a Probe 1.5-shaped Actor and perturb the new column away
    # from zero so the conditioning test sees a non-zero KL signal.
    actor = Actor(h_dim=h_dim, z_dim=z_dim, action_dim=action_dim, mlp_hidden=mlp_hidden)
    with torch.no_grad():
        # Move the scalar's column to a small non-zero value so the
        # actor's policy is not invariant to scalar perturbations.
        actor.net[0].weight[:, h_dim + z_dim:] = torch.randn(
            mlp_hidden, 1, generator=torch.Generator().manual_seed(0)
        ) * 0.5

    # Save weights under the canonical actor.* prefix.
    actor_state = {f"actor.{k}": v for k, v in actor.state_dict().items()}
    checkpoint_id = "ckpt-000001"
    ckpt_dir = run_dir / "checkpoints" / checkpoint_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights_path = ckpt_dir / "weights.safetensors"
    _save_safetensors(actor_state, str(weights_path))
    (ckpt_dir / "schema_version.txt").write_text(schema_version + "\n")
    return run_dir, checkpoint_id


# ===========================================================================
# Test 1 — build_digest against runs/probe1-20260503-123926/ produces no
# self-prediction lines (the on-disk Probe 1 baseline is 0.1.0; the digest
# must skip the new block entirely).
# ===========================================================================


@pytest.mark.skipif(
    not PROBE_1_RUN_DIR.is_dir(),
    reason="Probe 1 reference run not present at runs/probe1-20260503-123926/",
)
def test_build_digest_against_probe_1_run_produces_no_self_prediction_lines() -> None:
    rows: list[dict[str, Any]] = []
    shards = sorted((PROBE_1_TELEMETRY_DIR / "agent_step").glob("shard-*.parquet"))
    assert shards, f"no parquet shards under {PROBE_1_TELEMETRY_DIR / 'agent_step'}"
    for shard in shards:
        rows.extend(pq.read_table(str(shard)).to_pylist())
    assert rows, "Probe 1 run has no agent_step records"
    assert all(
        str(r.get("schema_version", "")) == PROBE_1_SCHEMA_VERSION for r in rows
    ), "expected every Probe 1 record to be 0.1.0"

    text = build_digest(rows)
    # Probe 1 records produce no self-prediction summary lines.
    assert "self_prediction_error_t" not in text
    assert "self_prediction allocation" not in text
    assert "self_prediction outliers" not in text
    # The Probe 1 episode block structure is unchanged.
    assert "## episode" in text
    assert "kl_aggregate_t" in text


# ===========================================================================
# Test 2 — build_digest against synthetic mixed-version row list produces
# self-prediction summary lines for 0.2.0 records and skips them for 0.1.0.
# ===========================================================================


def test_build_digest_mixed_version_emits_lines_for_v0_2_0_only() -> None:
    """Episode 0 is Probe 1 (0.1.0); episode 1 is Probe 1.5 (0.2.0)."""
    ep0 = [
        _agent_step_v0_1_0(t=t, episode_id=0, step_in_episode=t)
        for t in range(5)
    ]
    # Pattern: masked sentinel + 7 normal-valued steps. The outlier
    # detection has a separate dedicated test below — small N inflates
    # the sample std so a single spike at z=3 needs many "normal"
    # samples to surface, which is its own check.
    pattern = [0.0, 0.10, 0.12, 0.11, 0.13, 0.14, 0.10, 0.13]
    ep1 = _build_synthetic_v0_2_0_episode(
        episode_id=1,
        n_steps=len(pattern),
        scalar_pattern=pattern,
        starting_t=5,
    )
    rows = [r.model_dump() for r in (*ep0, *ep1)]
    text = build_digest(rows)

    # Probe 1 episode (0): no self-prediction block.
    ep0_block = text.split("## episode 0")[1].split("## episode 1")[0]
    assert "self_prediction_error_t" not in ep0_block
    assert "self_prediction allocation" not in ep0_block

    # Probe 1.5 episode (1): three of the four lines plan §2.6 specifies
    # (mean/std, masked-step count, per-dim allocation top-k).
    ep1_block = text.split("## episode 1")[1]
    assert "self_prediction_error_t (excluding masked steps): mean=" in ep1_block
    # Masked-step count is 1 (the first step of episode 1).
    assert "self_prediction_error_t masked steps in episode: count=1" in ep1_block
    # Per-dim allocation: top-k dims block exists (h_dim=4 here, so it
    # caps at the smaller of 5 and h_dim).
    assert "self_prediction allocation (per-dim variance" in ep1_block
    # Mean is computed over the 7 non-masked scalars
    # (0.10, 0.12, 0.11, 0.13, 0.14, 0.10, 0.13): rounded to 4 decimals.
    expected_mean = sum(pattern[1:]) / (len(pattern) - 1)
    assert f"mean={expected_mean:.4f}" in ep1_block


def test_build_digest_self_prediction_outlier_detection() -> None:
    """A spike at z>3 produces an outlier line. With small N the sample
    std is inflated by the spike itself; we use 20 normal samples + 1
    spike so the spike crosses the z=3 threshold cleanly."""
    pattern = [0.0]  # masked first step
    pattern.extend([0.10] * 20)
    pattern.append(1.0)  # spike at z ≈ 4.4 against the non-masked mean
    ep1 = _build_synthetic_v0_2_0_episode(
        episode_id=0,
        n_steps=len(pattern),
        scalar_pattern=pattern,
    )
    rows = [r.model_dump() for r in ep1]
    text = build_digest(rows)
    assert "self_prediction outliers" in text
    # The outlier is the spike step (step_in_episode == len-1).
    assert f"step_in_episode={len(pattern) - 1}" in text.split(
        "self_prediction outliers"
    )[1]


# ===========================================================================
# Test 3 — compact_record_repr with a 0.2.0 record includes both new
# scalars in the JSON output.
# ===========================================================================


def test_compact_record_repr_v0_2_0_includes_new_scalars() -> None:
    record = _agent_step_v0_2_0(
        t=7,
        episode_id=2,
        step_in_episode=3,
        masked=False,
        self_prediction_error=0.1234,
    )
    repr_str = compact_record_repr(record.model_dump(), position="middle")
    parsed = json.loads(repr_str)
    assert parsed["self_prediction_error_t"] == pytest.approx(0.1234)
    assert parsed["self_prediction_error_masked_t"] is False
    # The full vector stays out of the compact repr (high-dim discipline).
    assert "self_prediction_t" not in parsed


def test_compact_record_repr_v0_1_0_excludes_new_scalars() -> None:
    """Probe 1 records: the two new keys are absent from the compact
    JSON (Plan §3.3 "skip" — new fields are None when round-tripped
    through the 0.2.0 ParquetSink and the digest skips them rather
    than rendering them as ``null``)."""
    record = _agent_step_v0_1_0(t=0, episode_id=0, step_in_episode=0)
    repr_str = compact_record_repr(record.model_dump(), position="first")
    parsed = json.loads(repr_str)
    assert "self_prediction_error_t" not in parsed
    assert "self_prediction_error_masked_t" not in parsed


def test_compact_record_repr_masked_step_renders_masked_true() -> None:
    record = _agent_step_v0_2_0(
        t=0, episode_id=0, step_in_episode=0, masked=True, self_prediction_error=0.0
    )
    repr_str = compact_record_repr(record.model_dump(), position="first")
    parsed = json.loads(repr_str)
    assert parsed["self_prediction_error_t"] == 0.0
    assert parsed["self_prediction_error_masked_t"] is True


# ===========================================================================
# Test 4 — show_self_prediction against runs/probe1-20260503-123926/
# prints the no-self-prediction-telemetry message.
# ===========================================================================


@pytest.mark.skipif(
    not PROBE_1_RUN_DIR.is_dir(),
    reason="Probe 1 reference run not present",
)
def test_show_self_prediction_against_probe_1_prints_no_telemetry_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    show_self_prediction(PROBE_1_TELEMETRY_DIR)
    out = capsys.readouterr().out
    assert "no self_prediction telemetry" in out
    assert "0.1.0" in out


# ===========================================================================
# Test 5 — show_self_prediction against synthetic 0.2.0 telemetry
# directory prints per-dimension lines, excluding masked steps.
# ===========================================================================


def test_show_self_prediction_synthetic_v0_2_0_prints_per_dim_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = _build_synthetic_v0_2_0_episode(
        episode_id=0,
        n_steps=10,
        scalar_pattern=[0.0, 0.1, 0.2, 0.15, 0.5, 0.18, 0.2, 0.1, 0.3, 0.25],
    )
    _write_agent_steps(records, tmp_path)
    show_self_prediction(tmp_path, episode_id=0, top_k_dims=3)
    out = capsys.readouterr().out
    assert "self_prediction allocation" in out
    assert f"h_dim: {_H_DIM}" in out
    # Top-k=3 dims listed; the per-dim residual variance is non-degenerate
    # because the synthetic vectors carry per-step ramps differing from h_t.
    assert out.count("- dim=") == 3
    # Masked-step accounting: episode has one masked step (step 0).
    assert "masked steps (excluded): 1" in out


def test_show_self_prediction_excludes_masked_first_step_from_per_dim_arithmetic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The per-dim residual is computed from non-masked, in-episode
    consecutive (t, t+1) pairs only. If we synthesize an episode whose
    masked step has wildly different self_prediction_t than the rest,
    excluding it must be observable: the variance does not include
    the residual for the masked step.
    """
    # Construct an episode where the masked step's self_prediction_t
    # is anomalously huge — if it leaked into the arithmetic, the
    # per-dim variance would be dominated by it.
    records: list[AgentStep] = []
    for s in range(8):
        masked = s == 0
        if masked:
            sp_vec = [1000.0] * _H_DIM  # would dominate variance if included
        else:
            sp_vec = [0.05 * (i + 1) + 0.01 * s for i in range(_H_DIM)]
        scalar = 0.0 if masked else 0.1 * s
        records.append(
            _agent_step_v0_2_0(
                t=s,
                episode_id=0,
                step_in_episode=s,
                masked=masked,
                self_prediction_error=scalar,
                self_prediction_vector=sp_vec,
            )
        )
    _write_agent_steps(records, tmp_path)
    show_self_prediction(tmp_path, episode_id=0, top_k_dims=4)
    out = capsys.readouterr().out
    # All printed per-dim variances must be small — orders of magnitude
    # below 1000^2 — confirming the masked sentinel is excluded.
    for line in out.splitlines():
        if line.strip().startswith("- dim="):
            v = float(line.split("var=")[1])
            assert v < 1.0, f"masked step residual leaked into variance: {line}"


# ===========================================================================
# Test 6 — show_self_prediction_conditioning against synthetic 0.2.0 run
# directory with a known checkpoint produces the per-regime KL table
# with expected row count and column shape (no assertion on KL magnitudes).
# ===========================================================================


def test_show_self_prediction_conditioning_synthetic_run_emits_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, checkpoint_id = _build_synthetic_run_dir(
        tmp_path,
        h_dim=8,
        z_dim=4,
        action_dim=5,
        mlp_hidden=16,
        n_steps_per_episode=30,
        n_episodes=2,
        perturbation_t_event=10,  # places some states in perturbation_window
    )
    show_self_prediction_conditioning(
        run_dir,
        checkpoint_id=checkpoint_id,
        n_states=40,
        seed=0,
    )
    out = capsys.readouterr().out
    # Header present.
    assert "self_prediction_error conditioning" in out
    assert checkpoint_id in out
    # The four default regimes appear as table rows.
    for regime in (
        "perturbation_window",
        "high_disagreement",
        "high_kl",
        "steady_state",
    ):
        assert regime in out
    # Column header has the expected shape.
    assert "n_states" in out
    assert "KL_mean" in out
    assert "KL_std" in out
    assert "KL_p50" in out
    assert "KL_p90" in out


def test_show_self_prediction_conditioning_picks_latest_checkpoint_when_id_none(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, checkpoint_id = _build_synthetic_run_dir(
        tmp_path,
        h_dim=8,
        z_dim=4,
        action_dim=5,
        mlp_hidden=16,
        n_steps_per_episode=20,
        n_episodes=1,
    )
    show_self_prediction_conditioning(run_dir, checkpoint_id=None, n_states=10)
    out = capsys.readouterr().out
    assert checkpoint_id in out


# ===========================================================================
# Test 7 — show_self_prediction_conditioning against
# runs/probe1-20260503-123926/ prints the no-scalar-to-perturb message.
# ===========================================================================


@pytest.mark.skipif(
    not PROBE_1_RUN_DIR.is_dir(),
    reason="Probe 1 reference run not present",
)
def test_show_self_prediction_conditioning_against_probe_1_prints_no_scalar_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    show_self_prediction_conditioning(PROBE_1_RUN_DIR, n_states=10)
    out = capsys.readouterr().out
    assert "no scalar to perturb" in out
    assert "0.1.0" in out


# ===========================================================================
# Test 8 — masked steps correctly excluded from n_states sample.
# ===========================================================================


def test_show_self_prediction_conditioning_excludes_masked_steps_from_n_states(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Build a run where every record is masked except for one step;
    request n_states larger than the candidate pool and confirm the
    sampled count is bounded by the non-masked count.

    The check is on the printed ``sampled n_states=...`` line, which
    is bounded by ``min(n_states, len(candidate_rows))`` where
    ``candidate_rows`` is the set of 0.2.0 non-masked records.
    """
    run_dir = tmp_path / "all-masked-but-one"
    telemetry_dir = run_dir / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    h_dim, z_dim, action_dim, mlp_hidden = 8, 4, 5, 16

    # Many "first-step-of-episode" records (masked) plus one non-masked
    # record. The Phase 0 convention forces masked=True only on the
    # actual first step; we synthesize an extreme test fixture that
    # masks every step except one to exercise the helper's exclusion.
    records: list[AgentStep] = []
    n_total = 50
    non_masked_t = 25
    for t in range(n_total):
        masked = t != non_masked_t
        scalar = 0.0 if masked else 0.42
        records.append(
            AgentStep(
                schema_version=SCHEMA_VERSION,
                run_id="synthetic",
                checkpoint_id=None,
                t=t,
                episode_id=0,
                step_in_episode=t,
                wallclock_ms=1_000_000 + t,
                h_t=[0.1 * (i + 1) + 0.001 * t for i in range(h_dim)],
                q_params_t=([0.0] * z_dim, [0.0] * z_dim),
                p_params_t=([0.0] * z_dim, [0.0] * z_dim),
                z_t=[0.5] * z_dim,
                kl_per_dim_t=[0.05] * z_dim,
                kl_aggregate_t=0.5,
                recon_loss_t=1.0,
                action_t=t % action_dim,
                action_logprob_t=-1.6,
                policy_entropy_t=1.5,
                obs_hash_t=f"hash-{t}",
                intrinsic_signal_t=0.3,
                encoder_embedding_t=[0.0] * h_dim,
                self_prediction_t=[0.05 * (i + 1) for i in range(h_dim)],
                self_prediction_error_t=scalar,
                self_prediction_error_masked_t=masked,
            )
        )
    _write_agent_steps(records, telemetry_dir)
    (telemetry_dir / "world_event.jsonl").write_text("")

    actor = Actor(h_dim=h_dim, z_dim=z_dim, action_dim=action_dim, mlp_hidden=mlp_hidden)
    with torch.no_grad():
        actor.net[0].weight[:, h_dim + z_dim:] = torch.randn(
            mlp_hidden, 1, generator=torch.Generator().manual_seed(1)
        ) * 0.5
    actor_state = {f"actor.{k}": v for k, v in actor.state_dict().items()}
    ckpt_dir = run_dir / "checkpoints" / "ckpt-000001"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _save_safetensors(actor_state, str(ckpt_dir / "weights.safetensors"))
    (ckpt_dir / "schema_version.txt").write_text(SCHEMA_VERSION + "\n")

    # Ask for 100 states; only 1 candidate exists (the one non-masked
    # row). The helper must clamp.
    show_self_prediction_conditioning(run_dir, n_states=100, seed=0)
    out = capsys.readouterr().out
    assert "sampled n_states=1" in out, out


# ===========================================================================
# Test 9 — CLI accepts the new selfpred and cond subcommands.
# ===========================================================================


def test_cli_help_lists_new_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for command in ("recent", "events", "episode", "dream", "summary", "selfpred", "cond"):
        assert command in out, f"CLI --help missing subcommand {command!r}"


def test_cli_selfpred_against_probe_1_telemetry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Synthetic 0.2.0 telemetry; the CLI subcommand wraps the helper
    and prints the per-dim block."""
    records = _build_synthetic_v0_2_0_episode(episode_id=0, n_steps=8)
    _write_agent_steps(records, tmp_path)
    rc = main(["selfpred", str(tmp_path), "-e", "0", "-k", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "self_prediction allocation" in out


def test_cli_cond_against_synthetic_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, checkpoint_id = _build_synthetic_run_dir(
        tmp_path,
        h_dim=8,
        z_dim=4,
        action_dim=5,
        mlp_hidden=16,
        n_steps_per_episode=20,
        n_episodes=1,
    )
    rc = main(["cond", str(run_dir), "-c", checkpoint_id, "-n", "10", "--seed", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "self_prediction_error conditioning" in out
    # Default regimes printed.
    for regime in (
        "perturbation_window",
        "high_disagreement",
        "high_kl",
        "steady_state",
    ):
        assert regime in out


def test_cli_cond_with_subset_of_perturbations_and_regimes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, _ = _build_synthetic_run_dir(
        tmp_path,
        h_dim=8,
        z_dim=4,
        action_dim=5,
        mlp_hidden=16,
        n_steps_per_episode=15,
        n_episodes=1,
    )
    rc = main(
        [
            "cond",
            str(run_dir),
            "-n",
            "10",
            "--perturbation",
            "zero",
            "--regime",
            "high_kl",
            "--regime",
            "steady_state",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # Subset reflected in header.
    assert "['zero']" in out
    # Only the named regimes appear as rows.
    assert "high_kl" in out
    assert "steady_state" in out
    # The omitted regimes do not appear in the table body.
    table_body = out.split("regime")[-1]
    assert "perturbation_window" not in table_body
    assert "high_disagreement" not in table_body


# ===========================================================================
# Episode summary extension — show_episode_summary picks up the new lines
# for 0.2.0 records and continues to render unchanged for 0.1.0 records.
# ===========================================================================


def test_show_episode_summary_v0_2_0_includes_self_prediction_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = _build_synthetic_v0_2_0_episode(episode_id=0, n_steps=8)
    _write_agent_steps(records, tmp_path)
    show_episode_summary(tmp_path, episode_id=0)
    out = capsys.readouterr().out
    assert "self_prediction_error_t (excluding masked steps)" in out
    assert "self_prediction_error_t masked steps in episode: count=1" in out


def test_show_episode_summary_v0_1_0_excludes_self_prediction_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = [
        _agent_step_v0_1_0(t=t, episode_id=0, step_in_episode=t)
        for t in range(5)
    ]
    _write_agent_steps(records, tmp_path)
    show_episode_summary(tmp_path, episode_id=0)
    out = capsys.readouterr().out
    assert "self_prediction" not in out


# ===========================================================================
# Sanity: show_self_prediction against an empty telemetry dir produces the
# no-records message and does not crash.
# ===========================================================================


def test_show_self_prediction_no_records(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    show_self_prediction(tmp_path)
    out = capsys.readouterr().out
    assert "no agent_step records" in out


def test_show_self_prediction_conditioning_no_records(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "telemetry").mkdir()
    show_self_prediction_conditioning(tmp_path, n_states=10)
    out = capsys.readouterr().out
    assert "no agent_step records" in out


def test_show_self_prediction_conditioning_no_checkpoints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A 0.2.0 telemetry dir with no checkpoints should print a helpful
    message rather than crash."""
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    records = _build_synthetic_v0_2_0_episode(episode_id=0, n_steps=5)
    _write_agent_steps(records, telemetry_dir)
    show_self_prediction_conditioning(tmp_path, n_states=10)
    out = capsys.readouterr().out
    assert "no checkpoints" in out
