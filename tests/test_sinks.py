"""Phase 1 gate test for ``kind/observer/sinks.py``.

Plan §4 test #4 is the gate: write 100 ``AgentStep`` records via
``ParquetSink`` and 100 ``WorldEvent`` records via ``JsonlSink``; read them
back via PyArrow and Pydantic; assert equality after schema validation;
assert ``schema_version`` round-trips.

Smaller unit tests cover small-batch roundtrips, fsync on close,
schema-mismatch rejection, sharding with multiple files, and context-manager
semantics. The dispatch test iterates over Phase 0's ``RECORD_MODELS`` so
adding a fifth schema in a later probe is caught by the loop, not by a
hand-listed enum.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kind.observer.schemas import (
    RECORD_MODELS,
    PROBE_1_SCHEMA_VERSION,
    PROBE_3_TELEMETRY_SCHEMA_VERSION,
    AgentStep,
    DreamRollout,
    RecordEnvelope,
    ReplayMeta,
    WorldEvent,
)
from kind.observer.sinks import (
    JsonlSink,
    ParquetSink,
    SchemaMismatchError,
    SinkClosedError,
    decode_parquet_row,
    json_encoded_field_names,
    model_validate_parquet_row,
    read_parquet_dir,
)


# ---- record factories ------------------------------------------------------

H_DIM = 8
Z_DIM = 4
EMBED_DIM = 6
HORIZON = 5


def make_agent_step(t: int, *, run_id: str = "test-run") -> AgentStep:
    return AgentStep(
        schema_version=PROBE_1_SCHEMA_VERSION,
        run_id=run_id,
        checkpoint_id=f"ckpt-{t // 10:04d}",
        t=t,
        episode_id=t // 200,
        step_in_episode=t % 200,
        wallclock_ms=1_700_000_000_000 + t,
        h_t=[float(t + i) * 0.01 for i in range(H_DIM)],
        q_params_t=(
            [float(i) * 0.1 for i in range(Z_DIM)],
            [float(i) * -0.05 for i in range(Z_DIM)],
        ),
        p_params_t=(
            [0.0] * Z_DIM,
            [-1.0] * Z_DIM,
        ),
        z_t=[float(t + i) * 0.01 for i in range(Z_DIM)],
        kl_per_dim_t=[float(i) * 0.001 for i in range(Z_DIM)],
        kl_aggregate_t=float(t) * 0.001,
        recon_loss_t=float(t) * 0.0005,
        action_t=t % 5,
        action_logprob_t=-float(t) * 0.01,
        policy_entropy_t=0.5 + float(t) * 0.0001,
        obs_hash_t=f"hash-{t:08x}",
        intrinsic_signal_t=float(t) * 0.0002,
        encoder_embedding_t=[float(t + i) * 0.001 for i in range(EMBED_DIM)],
    )


def make_world_event(t: int, *, run_id: str = "test-run") -> WorldEvent:
    return WorldEvent(
        schema_version=PROBE_1_SCHEMA_VERSION,
        run_id=run_id,
        checkpoint_id=None,
        t_event=t,
        event_type=("env_reset" if t % 200 == 0 else "internal_stochasticity_aggregate"),
        source="environment",
        payload={"episode_id": t // 200, "marker": f"e{t}"},
        wallclock_ms=1_700_000_000_000 + t,
    )


def make_replay_meta(t: int, *, run_id: str = "test-run") -> ReplayMeta:
    return ReplayMeta(
        schema_version=PROBE_1_SCHEMA_VERSION,
        run_id=run_id,
        checkpoint_id=None,
        event_type="insert",
        t_event=t,
        segment_id=t,
        segment_start=t,
        segment_end=t + 32,
        priority=None,
        buffer_size=t + 1,
        total_segments=t + 1,
    )


def make_dream_rollout(t: int, *, run_id: str = "test-run") -> DreamRollout:
    return DreamRollout(
        schema_version=PROBE_1_SCHEMA_VERSION,
        run_id=run_id,
        checkpoint_id=None,
        seed_step=t,
        seed_h0=[float(i) * 0.01 for i in range(H_DIM)],
        seed_z0=[float(i) * 0.01 for i in range(Z_DIM)],
        sequence_h=[
            [float(j + i) * 0.01 for i in range(H_DIM)] for j in range(HORIZON)
        ],
        sequence_z_prior=[
            [float(j + i) * 0.01 for i in range(Z_DIM)] for j in range(HORIZON)
        ],
        sequence_action=[i % 5 for i in range(HORIZON)],
        sequence_action_logprob=[-float(i) * 0.1 for i in range(HORIZON)],
        sequence_prior_entropy=[float(i) * 0.05 for i in range(HORIZON)],
        sequence_decoded_obs=None,
        cumulative_prior_entropy=float(HORIZON) * 0.5,
        mean_step_kl_successive_priors=0.1,
        max_step_latent_norm_change=0.5,
    )


def _factory_for(model: type[RecordEnvelope]) -> Any:
    if model is AgentStep:
        return make_agent_step
    if model is DreamRollout:
        return make_dream_rollout
    if model is ReplayMeta:
        return make_replay_meta
    if model is WorldEvent:
        return make_world_event
    raise AssertionError(f"unmapped record model: {model.__name__}")


# ---- gate test (plan §4 test #4) -------------------------------------------


def test_gate_100_record_roundtrip_through_both_sinks(tmp_path: Path) -> None:
    """100 AgentStep via ParquetSink + 100 WorldEvent via JsonlSink, roundtrip."""
    # AgentStep → ParquetSink (single shard, since rows_per_shard > 100)
    parquet_dir = tmp_path / "agent_step"
    originals_a = [make_agent_step(t) for t in range(100)]
    with ParquetSink(parquet_dir, schema=AgentStep, rows_per_shard=200) as sink:
        for record in originals_a:
            sink.write(record)
    shards = sorted(parquet_dir.glob("shard-*.parquet"))
    assert len(shards) == 1, f"expected one shard, got {[p.name for p in shards]}"
    table = pq.read_table(shards[0])
    assert table.num_rows == 100
    rows = table.to_pylist()
    reconstructed_a = [AgentStep.model_validate(r) for r in rows]
    for orig, recon in zip(originals_a, reconstructed_a, strict=True):
        assert recon == orig
        assert recon.schema_version == PROBE_1_SCHEMA_VERSION

    # WorldEvent → JsonlSink (one record per line)
    jsonl_path = tmp_path / "world_event.jsonl"
    originals_w = [make_world_event(t) for t in range(100)]
    with JsonlSink(jsonl_path, schema=WorldEvent) as sink:
        for record in originals_w:
            sink.write(record)
    lines = jsonl_path.read_text().splitlines()
    assert len(lines) == 100
    reconstructed_w = [WorldEvent.model_validate_json(line) for line in lines]
    for orig, recon in zip(originals_w, reconstructed_w, strict=True):
        assert recon == orig
        assert recon.schema_version == PROBE_1_SCHEMA_VERSION


# ---- smaller unit tests ----------------------------------------------------


def test_jsonl_roundtrip_small_batch(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    originals = [make_world_event(t) for t in range(5)]
    with JsonlSink(path, schema=WorldEvent) as sink:
        for r in originals:
            sink.write(r)
    lines = path.read_text().splitlines()
    assert len(lines) == 5
    reconstructed = [WorldEvent.model_validate_json(line) for line in lines]
    assert reconstructed == originals


def test_jsonl_write_is_immediately_readable(tmp_path: Path) -> None:
    """Per-write flush: a concurrent same-host reader (the live window's
    event feed) sees each record without waiting for close() — the
    biography-continuation plan's C2 lag fix."""
    path = tmp_path / "events.jsonl"
    with JsonlSink(path, schema=WorldEvent) as sink:
        sink.write(make_world_event(1))
        lines = path.read_text().splitlines()
        assert len(lines) == 1
        assert WorldEvent.model_validate_json(lines[0]).t_event == 1


def test_parquet_roundtrip_with_column_dtypes(tmp_path: Path) -> None:
    directory = tmp_path / "agent_step"
    originals = [make_agent_step(t) for t in range(8)]
    with ParquetSink(directory, schema=AgentStep, rows_per_shard=100) as sink:
        for r in originals:
            sink.write(r)
    shards = sorted(directory.glob("shard-*.parquet"))
    assert len(shards) == 1
    table = pq.read_table(shards[0])

    # Column-level dtype checks: this is the "Parquet column dtypes match schema"
    # gate from the plan §2.2.
    schema = table.schema
    assert schema.field("intrinsic_signal_t").type == pa.float64()
    assert schema.field("t").type == pa.int64()
    assert schema.field("obs_hash_t").type == pa.string()
    assert schema.field("h_t").type == pa.list_(pa.float64())
    # tuple[list[float], list[float]] → list<list<double>>
    assert schema.field("q_params_t").type == pa.list_(pa.list_(pa.float64()))
    # Optional field stays nullable
    assert schema.field("checkpoint_id").nullable
    # Required fields stay non-nullable
    assert not schema.field("t").nullable

    rows = table.to_pylist()
    reconstructed = [AgentStep.model_validate(r) for r in rows]
    assert reconstructed == originals


def test_jsonl_fsync_on_close(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    with patch("kind.observer.sinks.os.fsync") as fsync_mock:
        sink = JsonlSink(path, schema=WorldEvent)
        sink.write(make_world_event(0))
        sink.close()
    assert fsync_mock.called, "JsonlSink.close() must call os.fsync"
    # And the data is on disk after close (real fsync was bypassed by mock,
    # but write+flush+close still flush page cache to the OS):
    assert path.exists() and path.stat().st_size > 0
    line = path.read_text().splitlines()[0]
    assert WorldEvent.model_validate_json(line).t_event == 0


def test_parquet_fsync_on_close(tmp_path: Path) -> None:
    directory = tmp_path / "agent_step"
    with patch("kind.observer.sinks.os.fsync") as fsync_mock:
        sink = ParquetSink(directory, schema=AgentStep, rows_per_shard=100)
        sink.write(make_agent_step(0))
        sink.close()
    assert fsync_mock.called, "ParquetSink._flush_shard must call os.fsync"
    shards = sorted(directory.glob("shard-*.parquet"))
    assert len(shards) == 1
    assert shards[0].stat().st_size > 0


def test_jsonl_rejects_wrong_schema_type(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    with JsonlSink(path, schema=WorldEvent) as sink:
        with pytest.raises(SchemaMismatchError, match="WorldEvent"):
            sink.write(make_agent_step(0))


def test_parquet_rejects_wrong_schema_type(tmp_path: Path) -> None:
    directory = tmp_path / "agent_step"
    with ParquetSink(directory, schema=AgentStep, rows_per_shard=100) as sink:
        with pytest.raises(SchemaMismatchError, match="AgentStep"):
            sink.write(make_world_event(0))


def test_parquet_sharding_produces_multiple_files(tmp_path: Path) -> None:
    directory = tmp_path / "agent_step"
    with ParquetSink(directory, schema=AgentStep, rows_per_shard=10) as sink:
        for t in range(25):
            sink.write(make_agent_step(t))
    shards = sorted(directory.glob("shard-*.parquet"))
    assert len(shards) == 3, f"expected 3 shards for 25 rows at 10/shard, got {len(shards)}"
    counts = [pq.read_table(p).num_rows for p in shards]
    assert counts == [10, 10, 5]
    # Concatenate across shards and roundtrip
    all_rows: list[dict[str, object]] = []
    for shard in shards:
        all_rows.extend(pq.read_table(shard).to_pylist())
    reconstructed = [AgentStep.model_validate(r) for r in all_rows]
    originals = [make_agent_step(t) for t in range(25)]
    assert reconstructed == originals


def test_write_after_close_raises_for_both_sinks(tmp_path: Path) -> None:
    jpath = tmp_path / "events.jsonl"
    j = JsonlSink(jpath, schema=WorldEvent)
    j.close()
    with pytest.raises(SinkClosedError):
        j.write(make_world_event(0))

    pdir = tmp_path / "agent_step"
    p = ParquetSink(pdir, schema=AgentStep)
    p.close()
    with pytest.raises(SinkClosedError):
        p.write(make_agent_step(0))


def test_close_is_idempotent_for_both_sinks(tmp_path: Path) -> None:
    jpath = tmp_path / "events.jsonl"
    j = JsonlSink(jpath, schema=WorldEvent)
    j.write(make_world_event(0))
    j.close()
    j.close()  # must not raise

    pdir = tmp_path / "agent_step"
    p = ParquetSink(pdir, schema=AgentStep)
    p.write(make_agent_step(0))
    p.close()
    p.close()  # must not raise


def test_parquet_rows_per_shard_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        ParquetSink(tmp_path / "x", schema=AgentStep, rows_per_shard=0)
    with pytest.raises(ValueError, match="positive"):
        ParquetSink(tmp_path / "y", schema=AgentStep, rows_per_shard=-1)


@pytest.mark.parametrize("model", RECORD_MODELS, ids=lambda m: m.__name__)
def test_each_record_model_writes_through_appropriate_sink(
    model: type[RecordEnvelope], tmp_path: Path
) -> None:
    """Iterate over Phase 0's RECORD_MODELS, dispatch each to its sink, roundtrip.

    The ``agent_step`` and ``dream_rollout`` streams go to ``ParquetSink``;
    ``replay_meta`` and ``world_event`` go to ``JsonlSink`` (per plan §2.2).
    """
    parquet_models = (AgentStep, DreamRollout)
    factory = _factory_for(model)
    originals = [factory(t) for t in range(3)]

    if model in parquet_models:
        directory = tmp_path / model.__name__
        with ParquetSink(directory, schema=model, rows_per_shard=10) as sink:
            for r in originals:
                sink.write(r)
        shards = sorted(directory.glob("shard-*.parquet"))
        assert len(shards) == 1
        rows = pq.read_table(shards[0]).to_pylist()
        reconstructed = [model.model_validate(r) for r in rows]
    else:
        path = tmp_path / f"{model.__name__}.jsonl"
        with JsonlSink(path, schema=model) as sink:
            for r in originals:
                sink.write(r)
        lines = path.read_text().splitlines()
        reconstructed = [model.model_validate_json(line) for line in lines]

    assert reconstructed == originals


def test_jsonl_sink_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "subdir" / "events.jsonl"
    with JsonlSink(path, schema=WorldEvent) as sink:
        sink.write(make_world_event(0))
    assert path.exists()


def test_parquet_sink_creates_directory(tmp_path: Path) -> None:
    directory = tmp_path / "nested" / "subdir" / "agent_step"
    with ParquetSink(directory, schema=AgentStep) as sink:
        sink.write(make_agent_step(0))
    assert directory.exists()
    assert sorted(directory.glob("shard-*.parquet"))


# ---- v0.3.0 dict-field (sampling_parameters) round-trip --------------------
# Regression coverage for the sink-routing fix: ParquetSink JSON-encodes
# dict-typed fields into string columns on write and the read helpers decode
# them back to dicts, so a heterogeneous-valued sampling_parameters survives
# write -> read with types intact. (Before the fix, ParquetSink raised at
# construction on the dict annotation — the 52-failure root cause.)


def _make_dream_rollout_v0_3_0(t: int, *, run_id: str = "test-run") -> DreamRollout:
    return DreamRollout(
        schema_version=PROBE_3_TELEMETRY_SCHEMA_VERSION,
        run_id=run_id,
        checkpoint_id="ckpt-000001",
        seed_step=t,
        seed_h0=[float(i) * 0.01 for i in range(H_DIM)],
        seed_z0=[float(i) * 0.01 for i in range(Z_DIM)],
        sequence_h=[
            [float(j + i) * 0.01 for i in range(H_DIM)] for j in range(HORIZON)
        ],
        sequence_z_prior=[
            [float(j + i) * 0.01 for i in range(Z_DIM)] for j in range(HORIZON)
        ],
        sequence_action=[i % 5 for i in range(HORIZON)],
        sequence_action_logprob=[-float(i) * 0.1 for i in range(HORIZON)],
        sequence_prior_entropy=[float(i) * 0.05 for i in range(HORIZON)],
        sequence_decoded_obs=None,
        cumulative_prior_entropy=float(HORIZON) * 0.5,
        mean_step_kl_successive_priors=0.1,
        max_step_latent_norm_change=0.5,
        sequence_self_prediction=None,
        dream_session_id="sess-abc",
        seed_kind="perturbed_prior",
        seed_perturbation_magnitude=0.0,
        temperature_schedule=[1.0] * HORIZON,
        sub_mode_tags=["prior_only_control"],
        # Heterogeneous value union: int, float, str, bool — the case that has
        # no clean native Arrow column type and motivates JSON encoding.
        sampling_parameters={
            "replay_warmup_length": 8,
            "horizon": HORIZON,
            "actor_action_temperature": 1.5,
            "action_policy": "uniform_random",
            "record_ensemble_disagreement": True,
        },
        gradient_policy="none",
        rng_seed=12345,
        termination_reason="horizon_complete",
        re_seed_step_indices=None,
        sequence_ensemble_disagreement_variance=[0.2] * HORIZON,
        checkpoint_hash="deadbeef",
    )


def test_dream_rollout_v0_3_0_dict_field_survives_parquet_roundtrip(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "dream_rollout"
    originals = [_make_dream_rollout_v0_3_0(t) for t in range(3)]
    with ParquetSink(directory, schema=DreamRollout, rows_per_shard=10) as sink:
        for r in originals:
            sink.write(r)

    # Read via the symmetric helper — consumers get a dict back, not a string.
    reconstructed = read_parquet_dir(directory, DreamRollout)
    assert reconstructed == originals

    # The dict is a real dict with value types preserved (int stays int, float
    # stays float, str stays str, bool stays bool — not all coerced to one type).
    sp = reconstructed[0].sampling_parameters
    assert isinstance(sp, dict)
    assert sp["replay_warmup_length"] == 8 and isinstance(sp["replay_warmup_length"], int)
    assert sp["actor_action_temperature"] == 1.5
    assert sp["action_policy"] == "uniform_random"
    assert sp["record_ensemble_disagreement"] is True


def test_dict_field_is_stored_as_json_string_column(tmp_path: Path) -> None:
    directory = tmp_path / "dream_rollout"
    with ParquetSink(directory, schema=DreamRollout, rows_per_shard=10) as sink:
        sink.write(_make_dream_rollout_v0_3_0(0))
    shard = sorted(directory.glob("shard-*.parquet"))[0]
    table = pq.read_table(shard)
    # On disk the column is a string; the raw row carries a JSON string, and
    # the decode helper turns it back into a dict.
    assert table.schema.field("sampling_parameters").type == pa.string()
    raw_row = table.to_pylist()[0]
    assert isinstance(raw_row["sampling_parameters"], str)
    decoded = decode_parquet_row(DreamRollout, raw_row)
    assert isinstance(decoded["sampling_parameters"], dict)


def test_json_encoded_field_names_detects_only_dict_fields() -> None:
    assert json_encoded_field_names(DreamRollout) == frozenset({"sampling_parameters"})
    assert json_encoded_field_names(AgentStep) == frozenset()
    assert json_encoded_field_names(ReplayMeta) == frozenset()


def test_decode_and_validate_are_noop_for_dict_free_model(tmp_path: Path) -> None:
    # For a model with no dict fields, decode_parquet_row returns the row
    # untouched and model_validate_parquet_row == model.model_validate.
    directory = tmp_path / "agent_step"
    original = make_agent_step(0)
    with ParquetSink(directory, schema=AgentStep, rows_per_shard=10) as sink:
        sink.write(original)
    row = pq.read_table(sorted(directory.glob("shard-*.parquet"))[0]).to_pylist()[0]
    assert decode_parquet_row(AgentStep, row) is row
    assert model_validate_parquet_row(AgentStep, row) == original
