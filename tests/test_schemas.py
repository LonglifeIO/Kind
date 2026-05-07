"""Phase 0 gate tests for ``kind/observer/schemas.py``.

Probe 1's three checks (models import; instantiate with valid dummy data;
JSON Schema export is byte-stable and matches the checked-in file) extend
into Probe 1.5's eight (Probe 1.5 implementation plan §2.1):

1. Schema models import.
2. ``AgentStep`` round-trips through Parquet with all three new fields
   populated (``schema_version == "0.2.0"`` writer-side path).
3. ``AgentStep`` round-trips with the new fields absent
   (``schema_version == "0.1.0"`` Probe 1 backward-readability path).
4. ``AgentStep`` round-trips with ``self_prediction_error_masked_t=True``
   (the first-step-of-episode sentinel case).
5. ``DreamRollout`` round-trips with ``sequence_self_prediction=None``
   (the reserved-for-Probe-3 default at Probe 1.5).
6. JSON Schema export is byte-stable across invocations.
7. ``SCHEMA_VERSION`` equals ``"0.2.0"`` and ``PROBE_1_SCHEMA_VERSION``
   equals ``"0.1.0"`` (Probe 1's frozen version, preserved as a named
   constant so writers that have not yet been rewired for Probe 1.5 can
   reference it without scattering string literals).
8. Mixed-version writer rejection: a ``"0.2.0"`` record with any of the
   three new fields ``None`` raises ``pydantic.ValidationError`` at
   construction.

The Parquet round-trip tests use ``ParquetSink`` from Phase 1 — the
same machinery the Probe 1.5 runner will use to emit ``agent_step``
records — so the Phase 0 schema additions are exercised through the
real read/write path rather than only through ``model_validate``.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from kind.observer.schemas import (
    PROBE_1_SCHEMA_VERSION,
    RECORD_MODELS,
    SCHEMA_VERSION,
    AgentStep,
    DreamRollout,
    ReplayMeta,
    WorldEvent,
    export_json_schema,
)
from kind.observer.sinks import ParquetSink

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_FILE_V0_2_0 = REPO_ROOT / "schemas" / "v0.2.0.json"
SCHEMA_FILE_V0_1_0 = REPO_ROOT / "schemas" / "v0.1.0.json"


# ---- shared dimensions mirror the ones the existing tests use -------------

H_DIM = 4
Z_DIM = 2


def _envelope(version: str = SCHEMA_VERSION) -> dict[str, object]:
    return {
        "schema_version": version,
        "run_id": "run-0000",
        "checkpoint_id": None,
    }


def _agent_step_probe1(t: int = 0) -> AgentStep:
    """A Probe-1-shaped AgentStep: ``schema_version == "0.1.0"``, three new
    fields left at their default ``None``."""
    return AgentStep(
        **_envelope(PROBE_1_SCHEMA_VERSION),
        t=t,
        episode_id=0,
        step_in_episode=t,
        wallclock_ms=1_700_000_000_000 + t,
        h_t=[float(i) for i in range(H_DIM)],
        q_params_t=([0.0] * Z_DIM, [0.0] * Z_DIM),
        p_params_t=([0.0] * Z_DIM, [0.0] * Z_DIM),
        z_t=[0.0] * Z_DIM,
        kl_per_dim_t=[0.0] * Z_DIM,
        kl_aggregate_t=0.0,
        recon_loss_t=0.0,
        action_t=0,
        action_logprob_t=0.0,
        policy_entropy_t=0.0,
        obs_hash_t="0" * 16,
        intrinsic_signal_t=0.0,
        encoder_embedding_t=[0.0] * H_DIM,
    )


def _agent_step_probe1_5(
    t: int = 0,
    *,
    masked: bool = False,
    self_prediction_error: float = 0.42,
) -> AgentStep:
    """A Probe-1.5-shaped AgentStep: ``schema_version == "0.2.0"``, with
    all three self-prediction fields populated.

    ``masked=True`` produces the first-step-of-episode sentinel shape
    (scalar=0.0, masked flag True). ``masked=False`` produces a
    subsequent-step shape (scalar=non-zero, masked flag False).
    """
    if masked:
        scalar = 0.0
    else:
        scalar = self_prediction_error
    return AgentStep(
        **_envelope(SCHEMA_VERSION),
        t=t,
        episode_id=0,
        step_in_episode=t,
        wallclock_ms=1_700_000_000_000 + t,
        h_t=[float(i) for i in range(H_DIM)],
        q_params_t=([0.0] * Z_DIM, [0.0] * Z_DIM),
        p_params_t=([0.0] * Z_DIM, [0.0] * Z_DIM),
        z_t=[0.0] * Z_DIM,
        kl_per_dim_t=[0.0] * Z_DIM,
        kl_aggregate_t=0.0,
        recon_loss_t=0.0,
        action_t=0,
        action_logprob_t=0.0,
        policy_entropy_t=0.0,
        obs_hash_t="0" * 16,
        intrinsic_signal_t=0.0,
        encoder_embedding_t=[0.0] * H_DIM,
        self_prediction_t=[float(i) * 0.1 for i in range(H_DIM)],
        self_prediction_error_t=scalar,
        self_prediction_error_masked_t=masked,
    )


# ---- (1) schema models import + structural sanity --------------------------


def test_schema_models_import() -> None:
    """Probe 1's first gate carries: the four record models import as a
    typed tuple. Probe 1.5 adds the named constants for both schema
    versions."""
    assert tuple(m.__name__ for m in RECORD_MODELS) == (
        "AgentStep",
        "DreamRollout",
        "ReplayMeta",
        "WorldEvent",
    )
    # Sanity: every model carries the envelope.
    for model in RECORD_MODELS:
        for key in ("schema_version", "run_id", "checkpoint_id"):
            assert key in model.model_fields, (
                f"{model.__name__} missing envelope field {key!r}"
            )


# ---- (7) named constants ---------------------------------------------------


def test_schema_version_constant_is_v0_2_0() -> None:
    assert SCHEMA_VERSION == "0.2.0"


def test_probe_1_schema_version_constant_is_v0_1_0() -> None:
    """The Probe 1 version is preserved as a named constant — Probe-1-era
    writers that have not yet been rewired for Probe 1.5 reference
    ``PROBE_1_SCHEMA_VERSION`` rather than literal ``"0.1.0"`` strings."""
    assert PROBE_1_SCHEMA_VERSION == "0.1.0"


# ---- (2) Probe-1.5 record round-trips through Parquet ----------------------


def test_agent_step_v0_2_0_round_trips_through_parquet(tmp_path: Path) -> None:
    """A ``"0.2.0"`` record with all three new fields populated writes
    through ``ParquetSink`` and reads back equal under ``model_validate``.

    This is the writer-side path Probe 1.5's runner will exercise from
    Phase 3 forward; verifying it at Phase 0 confirms the schema +
    Arrow-derived Parquet column layout accept the new fields end to
    end.
    """
    parquet_dir = tmp_path / "agent_step_v0_2_0"
    originals = [_agent_step_probe1_5(t, masked=False) for t in range(3)]
    with ParquetSink(parquet_dir, schema=AgentStep, rows_per_shard=10) as sink:
        for record in originals:
            sink.write(record)

    shards = sorted(parquet_dir.glob("shard-*.parquet"))
    assert len(shards) == 1
    rows = pq.read_table(shards[0]).to_pylist()
    reconstructed = [AgentStep.model_validate(r) for r in rows]

    assert reconstructed == originals
    for r in reconstructed:
        assert r.schema_version == SCHEMA_VERSION
        assert r.self_prediction_t is not None
        assert len(r.self_prediction_t) == H_DIM
        assert r.self_prediction_error_t == pytest.approx(0.42)
        assert r.self_prediction_error_masked_t is False


# ---- (3) Probe-1 backward-readability through Parquet ----------------------


def test_agent_step_v0_1_0_round_trips_through_parquet(tmp_path: Path) -> None:
    """A Probe-1-shaped record (``schema_version == "0.1.0"``, the three
    new fields absent → ``None`` after ``model_validate``) writes and
    reads back equal.

    This is the backward-readability case Probe 1's parquet shards under
    ``runs/probe1-20260503-123926/`` rely on. The validator does NOT
    fire on ``"0.1.0"`` records, so the new fields stay ``None`` without
    raising.
    """
    parquet_dir = tmp_path / "agent_step_v0_1_0"
    originals = [_agent_step_probe1(t) for t in range(3)]
    with ParquetSink(parquet_dir, schema=AgentStep, rows_per_shard=10) as sink:
        for record in originals:
            sink.write(record)

    shards = sorted(parquet_dir.glob("shard-*.parquet"))
    assert len(shards) == 1
    rows = pq.read_table(shards[0]).to_pylist()
    reconstructed = [AgentStep.model_validate(r) for r in rows]

    assert reconstructed == originals
    for r in reconstructed:
        assert r.schema_version == PROBE_1_SCHEMA_VERSION
        assert r.self_prediction_t is None
        assert r.self_prediction_error_t is None
        assert r.self_prediction_error_masked_t is None


# ---- (4) first-step-of-episode masked case ---------------------------------


def test_agent_step_v0_2_0_round_trips_with_masked_true(tmp_path: Path) -> None:
    """The first-step-of-episode case: scalar forced to 0.0 (sentinel)
    and the masked flag set ``True``.

    The validator's "non-None" check is satisfied because the scalar
    value 0.0 is non-None — the masked flag is what discriminates "no
    empirical reading available, sentinel zero" from "empirical
    near-zero reading" downstream of writer-side construction.
    """
    parquet_dir = tmp_path / "agent_step_masked"
    masked_record = _agent_step_probe1_5(t=0, masked=True)
    unmasked_record = _agent_step_probe1_5(t=1, masked=False)
    originals = [masked_record, unmasked_record]
    with ParquetSink(parquet_dir, schema=AgentStep, rows_per_shard=10) as sink:
        for record in originals:
            sink.write(record)

    rows = pq.read_table(sorted(parquet_dir.glob("shard-*.parquet"))[0]).to_pylist()
    reconstructed = [AgentStep.model_validate(r) for r in rows]

    assert reconstructed == originals
    # Masked-step semantics survive the round-trip.
    masked_rt, unmasked_rt = reconstructed
    assert masked_rt.self_prediction_error_masked_t is True
    assert masked_rt.self_prediction_error_t == 0.0
    assert masked_rt.self_prediction_t is not None  # vector still emitted
    assert unmasked_rt.self_prediction_error_masked_t is False
    assert unmasked_rt.self_prediction_error_t != 0.0


# ---- (5) DreamRollout reserved field --------------------------------------


def test_dream_rollout_v0_2_0_round_trips_with_sequence_self_prediction_none(
    tmp_path: Path,
) -> None:
    """``sequence_self_prediction`` defaults to ``None`` at Probe 1.5
    (the head does not run during dream rollouts at Probe 1.5; Probe 3
    may populate). The Probe 1.5 writer does not pass the field; the
    record's ``schema_version`` is ``"0.2.0"`` and the validator does
    not fire on ``DreamRollout`` (only ``AgentStep`` carries the
    writer-side enforcement)."""
    parquet_dir = tmp_path / "dream_rollout_v0_2_0"
    horizon = 4
    record = DreamRollout(
        **_envelope(SCHEMA_VERSION),
        seed_step=0,
        seed_h0=[float(i) * 0.01 for i in range(H_DIM)],
        seed_z0=[float(i) * 0.01 for i in range(Z_DIM)],
        sequence_h=[
            [float(j + i) * 0.01 for i in range(H_DIM)] for j in range(horizon)
        ],
        sequence_z_prior=[
            [float(j + i) * 0.01 for i in range(Z_DIM)] for j in range(horizon)
        ],
        sequence_action=[i % 5 for i in range(horizon)],
        sequence_action_logprob=[-float(i) * 0.1 for i in range(horizon)],
        sequence_prior_entropy=[float(i) * 0.05 for i in range(horizon)],
        sequence_decoded_obs=None,
        cumulative_prior_entropy=0.5,
        mean_step_kl_successive_priors=0.1,
        max_step_latent_norm_change=0.5,
    )
    assert record.sequence_self_prediction is None

    with ParquetSink(parquet_dir, schema=DreamRollout, rows_per_shard=10) as sink:
        sink.write(record)
    shards = sorted(parquet_dir.glob("shard-*.parquet"))
    rows = pq.read_table(shards[0]).to_pylist()
    reconstructed = DreamRollout.model_validate(rows[0])
    assert reconstructed == record
    assert reconstructed.sequence_self_prediction is None


# ---- (6) JSON Schema export -----------------------------------------------


def test_json_schema_export_is_byte_stable() -> None:
    first = export_json_schema()
    second = export_json_schema()
    assert first == second


def test_json_schema_export_matches_checked_in_v0_2_0_file() -> None:
    """The current ``export_json_schema()`` output is the bytes on disk at
    ``schemas/v0.2.0.json``. Out-of-sync edits to ``schemas.py`` fail
    here — regenerate via ``export_json_schema()`` and commit the
    result."""
    assert SCHEMA_FILE_V0_2_0.exists(), f"missing schema export at {SCHEMA_FILE_V0_2_0}"
    assert SCHEMA_FILE_V0_2_0.read_bytes() == export_json_schema(), (
        "schemas/v0.2.0.json is out of sync with kind/observer/schemas.py — "
        "regenerate via export_json_schema() and commit the result."
    )


def test_v0_1_0_json_is_preserved_on_disk() -> None:
    """``schemas/v0.1.0.json`` is the frozen Probe 1 JSON Schema export.
    Probe 1.5 leaves it in place per the implementation plan §3.4 ("both
    files remain present; readers external to Kind can pin to either
    version")."""
    assert SCHEMA_FILE_V0_1_0.exists(), (
        f"schemas/v0.1.0.json must remain on disk after the Probe 1.5 "
        f"schema bump (synthesis-frozen historical export); not found at "
        f"{SCHEMA_FILE_V0_1_0}"
    )
    body = SCHEMA_FILE_V0_1_0.read_text()
    assert '"schema_version": "0.1.0"' in body
    # The frozen export must NOT match the current export — it's a
    # historical snapshot of the Probe 1 schema, not the current one.
    assert SCHEMA_FILE_V0_1_0.read_bytes() != export_json_schema()


# ---- (8) mixed-version writer rejection -----------------------------------


@pytest.mark.parametrize(
    "missing_field",
    [
        "self_prediction_t",
        "self_prediction_error_t",
        "self_prediction_error_masked_t",
    ],
)
def test_mixed_version_writer_rejection_raises_validation_error(
    missing_field: str,
) -> None:
    """A record stamped ``schema_version == "0.2.0"`` with any of the
    three new fields ``None`` raises ``pydantic.ValidationError`` at
    construction.

    The validator (``AgentStep._enforce_v2_required_fields``) is the
    writer-side discipline: Probe 1.5 records must populate the
    self-prediction fields. The error message names the missing field
    so writer-side bugs surface specifically.
    """
    full_kwargs = {
        **_envelope(SCHEMA_VERSION),
        "t": 0,
        "episode_id": 0,
        "step_in_episode": 0,
        "wallclock_ms": 1_700_000_000_000,
        "h_t": [0.0] * H_DIM,
        "q_params_t": ([0.0] * Z_DIM, [0.0] * Z_DIM),
        "p_params_t": ([0.0] * Z_DIM, [0.0] * Z_DIM),
        "z_t": [0.0] * Z_DIM,
        "kl_per_dim_t": [0.0] * Z_DIM,
        "kl_aggregate_t": 0.0,
        "recon_loss_t": 0.0,
        "action_t": 0,
        "action_logprob_t": 0.0,
        "policy_entropy_t": 0.0,
        "obs_hash_t": "0" * 16,
        "intrinsic_signal_t": 0.0,
        "encoder_embedding_t": [0.0] * H_DIM,
        "self_prediction_t": [0.0] * H_DIM,
        "self_prediction_error_t": 0.0,
        "self_prediction_error_masked_t": False,
    }
    full_kwargs[missing_field] = None
    with pytest.raises(ValidationError) as exc_info:
        AgentStep(**full_kwargs)
    assert missing_field in str(exc_info.value), (
        f"ValidationError must name the missing field {missing_field!r} for "
        f"writer-side debuggability; got: {exc_info.value}"
    )


def test_mixed_version_writer_rejection_all_three_missing() -> None:
    """A ``"0.2.0"`` record with all three new fields ``None`` (the most
    common writer-side bug — forgot to wire the head) raises
    ``ValidationError`` and names every missing field."""
    with pytest.raises(ValidationError) as exc_info:
        AgentStep(
            **_envelope(SCHEMA_VERSION),
            t=0,
            episode_id=0,
            step_in_episode=0,
            wallclock_ms=1_700_000_000_000,
            h_t=[0.0] * H_DIM,
            q_params_t=([0.0] * Z_DIM, [0.0] * Z_DIM),
            p_params_t=([0.0] * Z_DIM, [0.0] * Z_DIM),
            z_t=[0.0] * Z_DIM,
            kl_per_dim_t=[0.0] * Z_DIM,
            kl_aggregate_t=0.0,
            recon_loss_t=0.0,
            action_t=0,
            action_logprob_t=0.0,
            policy_entropy_t=0.0,
            obs_hash_t="0" * 16,
            intrinsic_signal_t=0.0,
            encoder_embedding_t=[0.0] * H_DIM,
        )
    msg = str(exc_info.value)
    assert "self_prediction_t" in msg
    assert "self_prediction_error_t" in msg
    assert "self_prediction_error_masked_t" in msg


def test_v0_1_0_record_with_new_fields_none_does_not_raise() -> None:
    """The validator is scoped to ``schema_version == "0.2.0"``. A
    ``"0.1.0"`` record with the new fields ``None`` constructs without
    error — that's the Probe 1 backward-readability path."""
    record = _agent_step_probe1(t=0)
    assert record.schema_version == PROBE_1_SCHEMA_VERSION
    assert record.self_prediction_t is None


# ---- per-model envelope sanity (carried from Probe 1) ---------------------


def test_replay_meta_instantiates_under_probe_1_schema_version() -> None:
    """``ReplayMeta`` has no Probe 1.5 fields, so it round-trips at
    either version. This test pins the Probe-1-shaped construction
    used by ``kind/training/replay.py`` until Phase 3 rewires it."""
    record = ReplayMeta(
        **_envelope(PROBE_1_SCHEMA_VERSION),
        event_type="insert",
        t_event=0,
        segment_id=0,
        segment_start=0,
        segment_end=32,
        priority=None,
        buffer_size=1,
        total_segments=1,
    )
    assert record.event_type == "insert"


def test_world_event_instantiates_under_probe_1_schema_version() -> None:
    """``WorldEvent`` has no Probe 1.5 fields. Same Probe-1-shaped
    construction pin as ``ReplayMeta``; the env-server emits these and
    has not been rewired for Probe 1.5 at Phase 0."""
    record = WorldEvent(
        **_envelope(PROBE_1_SCHEMA_VERSION),
        t_event=0,
        event_type="env_reset",
        source="environment",
        payload={"episode_id": 0},
        wallclock_ms=1_700_000_000_000,
    )
    assert record.event_type == "env_reset"
    assert record.payload == {"episode_id": 0}
