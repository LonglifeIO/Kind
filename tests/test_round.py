"""Phase 12 gate test — :mod:`kind.mirror.calibration.round`.

End-to-end test for :func:`run_round` against synthetic telemetry and a
mocked LLM. Mirrors the pattern in :mod:`tests.test_orchestrator` —
synthetic agent_step parquet shards + a synthetic world_event.jsonl per
checkpoint, a :class:`MockLLMClient` that returns canned responses.

Covers:

- :class:`RoundConfig` validation: snake_case round_id, non-empty
  checkpoints, positive passes_per_checkpoint, sham schedule consistency.
- :class:`RoundConfig` ``frozen=True`` invariant — mutating fields after
  construction raises (the structural enforcement of the
  "no-config-drift-after-pre-registration" discipline).
- :func:`run_round` happy path: emits round_config.json + pre_reg.jsonl
  before passes run; runs the per-checkpoint per-pass loop; collects
  records; aggregates sham findings; writes RoundResult atomically.
- :class:`ShamFindingsSummary` aggregation across passes.
- :class:`RoundResult` serialization round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from kind.mirror.calibration.round import (
    ROUND_CONFIG_FILENAME,
    CheckpointSpec,
    RoundConfig,
    RoundResult,
    run_round,
)
from kind.mirror.calibration.sham_schedule import (
    ShamSchedule,
    ShamScheduleEntry,
    generate_sham_schedule,
)
from kind.mirror.calibration.synthetic_perturbation import (
    SyntheticPerturbationEntry,
    SyntheticPerturbationSchedule,
    generate_synthetic_perturbation_schedule,
)
from kind.mirror.criteria_v2 import V2_REGISTRY
from kind.mirror.llm_caller import (
    BatchPayload,
    LLMConfig,
    MockLLMClient,
    _PerCriterionReadingPayload,
)
from kind.mirror.registry import CriterionRegistry
from kind.mirror.statistics import StatisticConfig
from kind.mirror.structured import StructuredClaim


# ---------------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------------


def _build_agent_step_rows(
    n_steps: int = 200, n_episodes: int = 2, h_dim: int = 4
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    steps_per_ep = n_steps // n_episodes
    for i in range(n_steps):
        ep = i // steps_per_ep
        in_ep = i % steps_per_ep
        h = [0.5 * float(in_ep) / steps_per_ep + 0.01 * j for j in range(h_dim)]
        rows.append(
            {
                "schema_version": "0.2.0",
                "run_id": "round-test",
                "checkpoint_id": "ckpt-000001",
                "t": i,
                "episode_id": ep,
                "step_in_episode": in_ep,
                "wallclock_ms": i * 100,
                "h_t": h,
                "z_t": [0.0, 0.0, 0.0, 0.0],
                "encoder_embedding_t": [0.0, 0.0, 0.0, 0.0],
                "policy_entropy_t": 1.0,
                "kl_aggregate_t": 0.5,
                "action_t": i % 5,
                "action_logprob_t": -1.0,
                "obs_hash_t": f"obs_{i % 7}",
                "q_params_t": ([0.0] * 4, [0.0] * 4),
                "p_params_t": ([0.0] * 4, [0.0] * 4),
                "kl_per_dim_t": [0.0] * 4,
                "recon_loss_t": 0.0,
                "intrinsic_signal_t": 0.0,
                "self_prediction_t": [0.0] * 4,
                "self_prediction_error_t": 0.0,
                "self_prediction_error_masked_t": False,
            }
        )
    return rows


def _write_agent_step_shards(
    telemetry_dir: Path, rows: list[dict[str, Any]]
) -> None:
    shard_dir = telemetry_dir / "agent_step"
    shard_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    with (shard_dir / "shard-000000.parquet").open("wb") as fh:
        pq.write_table(table, fh)  # type: ignore[no-untyped-call]


def _write_world_event_log(
    telemetry_dir: Path, perturbations: list[dict[str, Any]]
) -> None:
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    path = telemetry_dir / "world_event.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for p in perturbations:
            fh.write(json.dumps(p) + "\n")


def _set_up_run_dir(tmp_path: Path, run_id: str = "round-test") -> Path:
    """Write a synthetic run dir at ``tmp_path/runs/{run_id}/`` with
    200 agent_step rows and one real perturbation at t=60."""
    run_dir = tmp_path / "runs" / run_id
    telemetry_dir = run_dir / "telemetry"
    _write_agent_step_shards(telemetry_dir, _build_agent_step_rows())
    _write_world_event_log(
        telemetry_dir,
        [
            {
                "schema_version": "0.2.0",
                "run_id": run_id,
                "checkpoint_id": "ckpt-000001",
                "t_event": 60,
                "event_type": "builder_perturbation",
                "source": "builder",
                "payload": {"kind": "real"},
                "wallclock_ms": 60 * 100,
            }
        ],
    )
    return run_dir


def _claim() -> StructuredClaim:
    return StructuredClaim(
        claim="example",
        cited_stream="agent_step",
        cited_run_id="round-test",
        cited_episode_range=(0, 1),
        cited_step_range=(0, 100),
        cited_scalar_field="h_t",
        cited_value=0.0,
        falsifier="f",
        paraphrase_stability=None,
        reseed_stability=None,
        faithfulness_status="not_checked",
        judge_ruling="not_judged",
        reading_surface="head_internal",
        masked_steps_handling="n/a",
    )


def _per_criterion(criterion_id: str) -> _PerCriterionReadingPayload:
    return _PerCriterionReadingPayload(
        criterion_id=criterion_id,
        framework_anchor="buddhist_phenomenology",
        claims=[_claim()],
        free_text_notes="n",
    )


def _active_payload() -> BatchPayload:
    return BatchPayload(
        per_criterion=[
            _per_criterion("reflexive_attention"),
            _per_criterion("equanimity_perturbation_recovery"),
        ]
    )


def _held_out_payload() -> BatchPayload:
    return BatchPayload(
        per_criterion=[_per_criterion("second_order_volition")]
    )


def _build_round_mock_client(passes: int) -> MockLLMClient:
    """Build a mock returning the 4 × passes responses needed for one
    round on one checkpoint with the V2_REGISTRY partition."""
    responses: list[BatchPayload | Exception] = []
    for _ in range(passes):
        responses.append(_active_payload())  # active primary
        responses.append(_active_payload())  # active adversarial
        responses.append(_held_out_payload())  # held-out primary
        responses.append(_held_out_payload())  # held-out adversarial
    return MockLLMClient(responses)


def _make_round_config(
    tmp_path: Path,
    *,
    passes_per_checkpoint: int = 2,
    round_id: str = "round_test_a",
) -> RoundConfig:
    run_dir = _set_up_run_dir(tmp_path)
    checkpoint = CheckpointSpec(
        run_id="round-test",
        checkpoint_id="ckpt-000001",
        run_dir=run_dir,
    )
    sham_schedule = generate_sham_schedule(
        checkpoint_ids=("ckpt-000001",),
        passes_per_checkpoint=passes_per_checkpoint,
        real_perturbations_per_pass=1,
        shams_per_pass=1,
        telemetry_length=200,
        seed=42,
    )
    return RoundConfig(
        round_id=round_id,
        checkpoints=(checkpoint,),
        passes_per_checkpoint=passes_per_checkpoint,
        statistic_config=StatisticConfig(),
        llm_config=LLMConfig(),
        sham_schedule=sham_schedule,
        active_registry=CriterionRegistry(criteria=V2_REGISTRY.active()),
        held_out_registry=CriterionRegistry(criteria=V2_REGISTRY.held_out()),
    )


# ---------------------------------------------------------------------------
# RoundConfig validation.
# ---------------------------------------------------------------------------


def test_round_config_rejects_non_snake_case_round_id(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="round_id"):
        _make_round_config(tmp_path, round_id="Round-1")


def test_round_config_rejects_zero_passes(tmp_path: Path) -> None:
    """Construct a valid sham schedule first, then try to wrap it into
    a RoundConfig with passes_per_checkpoint=0 — RoundConfig's
    validator rejects the zero value (the sham scheduler also rejects
    zero passes, but that's a separate check)."""
    run_dir = _set_up_run_dir(tmp_path)
    checkpoint = CheckpointSpec(
        run_id="round-test", checkpoint_id="ckpt-000001", run_dir=run_dir
    )
    valid_schedule = generate_sham_schedule(
        checkpoint_ids=("ckpt-000001",),
        passes_per_checkpoint=1,
        real_perturbations_per_pass=1,
        shams_per_pass=1,
        telemetry_length=200,
        seed=42,
    )
    with pytest.raises(ValidationError, match="passes_per_checkpoint"):
        RoundConfig(
            round_id="r_test",
            checkpoints=(checkpoint,),
            passes_per_checkpoint=0,
            statistic_config=StatisticConfig(),
            llm_config=LLMConfig(),
            sham_schedule=valid_schedule,
        )


def test_round_config_rejects_empty_checkpoints(tmp_path: Path) -> None:
    sham_schedule = ShamSchedule(
        entries=(), real_perturbations_per_pass=0, shams_per_pass=0, seed=1
    )
    with pytest.raises(ValidationError, match="non-empty"):
        RoundConfig(
            round_id="r",
            checkpoints=(),
            passes_per_checkpoint=1,
            statistic_config=StatisticConfig(),
            llm_config=LLMConfig(),
            sham_schedule=sham_schedule,
        )


def test_round_config_rejects_sham_schedule_for_unknown_checkpoint(
    tmp_path: Path,
) -> None:
    """A sham schedule entry that references a checkpoint not in the
    round's checkpoints raises at RoundConfig construction."""
    run_dir = _set_up_run_dir(tmp_path)
    checkpoint = CheckpointSpec(
        run_id="round-test", checkpoint_id="ckpt-000001", run_dir=run_dir
    )
    # Schedule references "ckpt-other", which isn't in checkpoints.
    sham_schedule = ShamSchedule(
        entries=(
            ShamScheduleEntry(
                checkpoint_id="ckpt-other",
                pass_index=0,
                sham_t=50,
                sham_payload={"is_sham": True},
            ),
        ),
        real_perturbations_per_pass=1,
        shams_per_pass=1,
        seed=42,
    )
    with pytest.raises(ValidationError, match="ckpt-other"):
        RoundConfig(
            round_id="r_test",
            checkpoints=(checkpoint,),
            passes_per_checkpoint=1,
            statistic_config=StatisticConfig(),
            llm_config=LLMConfig(),
            sham_schedule=sham_schedule,
        )


def test_round_config_rejects_sham_schedule_pass_index_out_of_range(
    tmp_path: Path,
) -> None:
    run_dir = _set_up_run_dir(tmp_path)
    checkpoint = CheckpointSpec(
        run_id="round-test", checkpoint_id="ckpt-000001", run_dir=run_dir
    )
    sham_schedule = ShamSchedule(
        entries=(
            ShamScheduleEntry(
                checkpoint_id="ckpt-000001",
                pass_index=5,  # round only does 2 passes
                sham_t=50,
                sham_payload={"is_sham": True},
            ),
        ),
        real_perturbations_per_pass=1,
        shams_per_pass=1,
        seed=42,
    )
    with pytest.raises(ValidationError, match="pass_index"):
        RoundConfig(
            round_id="r_test",
            checkpoints=(checkpoint,),
            passes_per_checkpoint=2,
            statistic_config=StatisticConfig(),
            llm_config=LLMConfig(),
            sham_schedule=sham_schedule,
        )


# ---------------------------------------------------------------------------
# Frozen-after-pre-registration invariant.
# ---------------------------------------------------------------------------


def test_round_config_frozen_after_pre_registration(tmp_path: Path) -> None:
    """The Phase 12 load-bearing invariant: once a RoundConfig is
    constructed (which is *before* run_round writes pre-registration to
    disk), its fields cannot be mutated. The structural enforcement is
    Pydantic ``frozen=True``."""
    config = _make_round_config(tmp_path)
    with pytest.raises(ValidationError):
        config.statistic_config = StatisticConfig(kmeans_k=8)
    with pytest.raises(ValidationError):
        config.passes_per_checkpoint = 10


# ---------------------------------------------------------------------------
# run_round end-to-end.
# ---------------------------------------------------------------------------


def test_run_round_writes_round_config_before_passes(tmp_path: Path) -> None:
    """run_round writes RoundConfig to disk before any per-pass run.
    Verified by inspecting the on-disk output_dir after the call."""
    config = _make_round_config(tmp_path, passes_per_checkpoint=2)
    client = _build_round_mock_client(passes=2)
    output_dir = tmp_path / "phase_12_smoke"
    run_round(config, output_dir=output_dir, llm_client=client)
    round_config_path = (
        output_dir / "mirror" / "pre_reg" / f"round_{config.round_id}"
        / ROUND_CONFIG_FILENAME
    )
    assert round_config_path.is_file()
    loaded = json.loads(round_config_path.read_text())
    assert loaded["round_id"] == config.round_id


def test_run_round_emits_per_checkpoint_pre_registration(
    tmp_path: Path,
) -> None:
    config = _make_round_config(tmp_path, passes_per_checkpoint=2)
    client = _build_round_mock_client(passes=2)
    output_dir = tmp_path / "phase_12_smoke"
    run_round(config, output_dir=output_dir, llm_client=client)
    pre_reg_jsonl = (
        output_dir / "mirror" / "pre_reg" / f"round_{config.round_id}"
        / "pre_reg.jsonl"
    )
    assert pre_reg_jsonl.is_file()
    lines = pre_reg_jsonl.read_text().strip().split("\n")
    # One pre-registration per checkpoint (this round has 1 checkpoint).
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert set(rec["criteria_active"]) == {
        c.id for c in V2_REGISTRY.active()
    }


def test_run_round_produces_pass_results_per_checkpoint_per_pass(
    tmp_path: Path,
) -> None:
    config = _make_round_config(tmp_path, passes_per_checkpoint=2)
    client = _build_round_mock_client(passes=2)
    output_dir = tmp_path / "phase_12_smoke"
    result = run_round(config, output_dir=output_dir, llm_client=client)
    # 1 checkpoint × 2 passes = 2 pass results.
    assert len(result.pass_results) == 2
    assert result.pass_results[0].checkpoint_id == "ckpt-000001"


def test_run_round_collects_llm_records(tmp_path: Path) -> None:
    config = _make_round_config(tmp_path, passes_per_checkpoint=2)
    client = _build_round_mock_client(passes=2)
    output_dir = tmp_path / "phase_12_smoke"
    result = run_round(config, output_dir=output_dir, llm_client=client)
    # 2 passes × 4 LLM calls = 8 records (each call succeeds on first try).
    assert len(result.llm_call_records) == 8
    assert all(r.outcome == "success" for r in result.llm_call_records)
    # Each record carries the round_id and the per-pass pass_index.
    pass_indices = {r.pass_index for r in result.llm_call_records}
    assert pass_indices == {0, 1}


def test_run_round_writes_result_atomically(tmp_path: Path) -> None:
    config = _make_round_config(tmp_path, passes_per_checkpoint=1)
    client = _build_round_mock_client(passes=1)
    output_dir = tmp_path / "phase_12_smoke"
    run_round(config, output_dir=output_dir, llm_client=client)
    result_path = (
        output_dir / "mirror" / "rounds" / f"{config.round_id}.json"
    )
    assert result_path.is_file()
    # No leftover temp file.
    temp = list(result_path.parent.glob(f".*{config.round_id}*.tmp"))
    assert not temp


def test_run_round_result_serialization_round_trip(tmp_path: Path) -> None:
    config = _make_round_config(tmp_path, passes_per_checkpoint=1)
    client = _build_round_mock_client(passes=1)
    output_dir = tmp_path / "phase_12_smoke"
    result = run_round(config, output_dir=output_dir, llm_client=client)
    redumped = RoundResult.model_validate_json(result.model_dump_json())
    assert redumped == result


def test_run_round_validates_missing_telemetry_dir(tmp_path: Path) -> None:
    """A RoundConfig whose checkpoint has no telemetry/ subdir raises at
    run_round time."""
    checkpoint = CheckpointSpec(
        run_id="round-test",
        checkpoint_id="ckpt-000001",
        run_dir=tmp_path / "nowhere",
    )
    (tmp_path / "nowhere").mkdir()  # parent exists, telemetry/ doesn't
    sham_schedule = generate_sham_schedule(
        checkpoint_ids=("ckpt-000001",),
        passes_per_checkpoint=1,
        real_perturbations_per_pass=1,
        shams_per_pass=1,
        telemetry_length=200,
        seed=42,
    )
    config = RoundConfig(
        round_id="r_test",
        checkpoints=(checkpoint,),
        passes_per_checkpoint=1,
        statistic_config=StatisticConfig(),
        llm_config=LLMConfig(),
        sham_schedule=sham_schedule,
        active_registry=CriterionRegistry(criteria=V2_REGISTRY.active()),
        held_out_registry=CriterionRegistry(criteria=V2_REGISTRY.held_out()),
    )
    output_dir = tmp_path / "phase_12_smoke"
    with pytest.raises(ValueError, match="telemetry"):
        run_round(config, output_dir=output_dir, llm_client=MockLLMClient([]))


def test_sham_findings_summary_no_admissions(tmp_path: Path) -> None:
    """With the mock returning claims whose cited_step_range is (0, 100),
    the sham at some t (drawn from generate_sham_schedule with seed=42)
    may or may not overlap. The sham schedule's t is bounded to
    [1, telemetry_length-2] = [1, 198]. With seed=42 and 1 sham per pass
    over 2 passes, the sham_ts are deterministic. Either way, the
    aggregation runs correctly and produces a well-formed
    ShamFindingsSummary."""
    config = _make_round_config(tmp_path, passes_per_checkpoint=2)
    client = _build_round_mock_client(passes=2)
    output_dir = tmp_path / "phase_12_smoke"
    result = run_round(config, output_dir=output_dir, llm_client=client)
    summary = result.sham_findings_summary
    # Round has 2 passes, 1 sham each = 2 sham events.
    assert summary.total_sham_events == 2
    assert summary.total >= 0
    assert summary.total <= 2


# ---------------------------------------------------------------------------
# Phase 13: synthetic-perturbation schedule on RoundConfig + RoundResult.
# ---------------------------------------------------------------------------


def _make_round_config_with_synthetic(
    tmp_path: Path,
    *,
    passes_per_checkpoint: int = 2,
    round_id: str = "round_test_synth",
    synthetics_per_pass: int = 2,
    sham_seed: int = 42,
    synthetic_seed: int = 142,
) -> RoundConfig:
    run_dir = _set_up_run_dir(tmp_path)
    checkpoint = CheckpointSpec(
        run_id="round-test",
        checkpoint_id="ckpt-000001",
        run_dir=run_dir,
    )
    sham_schedule = generate_sham_schedule(
        checkpoint_ids=("ckpt-000001",),
        passes_per_checkpoint=passes_per_checkpoint,
        real_perturbations_per_pass=1,
        shams_per_pass=1,
        telemetry_length=200,
        seed=sham_seed,
    )
    synthetic_schedule = generate_synthetic_perturbation_schedule(
        checkpoint_ids=("ckpt-000001",),
        passes_per_checkpoint=passes_per_checkpoint,
        synthetics_per_pass=synthetics_per_pass,
        telemetry_length=200,
        recovery_window=50,
        seed=synthetic_seed,
        sham_schedule=sham_schedule,
    )
    return RoundConfig(
        round_id=round_id,
        checkpoints=(checkpoint,),
        passes_per_checkpoint=passes_per_checkpoint,
        statistic_config=StatisticConfig(),
        llm_config=LLMConfig(),
        sham_schedule=sham_schedule,
        synthetic_schedule=synthetic_schedule,
        active_registry=CriterionRegistry(criteria=V2_REGISTRY.active()),
        held_out_registry=CriterionRegistry(criteria=V2_REGISTRY.held_out()),
    )


def test_round_config_accepts_synthetic_schedule(tmp_path: Path) -> None:
    config = _make_round_config_with_synthetic(tmp_path)
    # synthetic_schedule made it onto the frozen config.
    assert config.synthetic_schedule.synthetics_per_pass == 2
    assert len(config.synthetic_schedule.entries) == 2 * 2  # passes × per-pass


def test_round_config_default_synthetic_schedule_is_empty(tmp_path: Path) -> None:
    """Phase 12 callers that don't supply synthetic_schedule get an
    empty default — backwards compatibility."""
    config = _make_round_config(tmp_path)
    assert len(config.synthetic_schedule.entries) == 0
    assert config.synthetic_schedule.synthetics_per_pass == 0


def test_round_config_rejects_synthetic_schedule_for_unknown_checkpoint(
    tmp_path: Path,
) -> None:
    run_dir = _set_up_run_dir(tmp_path)
    checkpoint = CheckpointSpec(
        run_id="round-test", checkpoint_id="ckpt-000001", run_dir=run_dir
    )
    sham_schedule = ShamSchedule(
        entries=(), real_perturbations_per_pass=0, shams_per_pass=0, seed=1
    )
    synthetic_schedule = SyntheticPerturbationSchedule(
        entries=(
            SyntheticPerturbationEntry(
                checkpoint_id="ckpt-other",  # not in the round's checkpoints
                pass_index=0,
                synthetic_t=50,
                synthetic_payload={
                    "is_synthetic": True,
                    "is_sham": False,
                },
            ),
        ),
        synthetics_per_pass=1,
        seed=142,
    )
    with pytest.raises(ValidationError, match="ckpt-other"):
        RoundConfig(
            round_id="r_test",
            checkpoints=(checkpoint,),
            passes_per_checkpoint=1,
            statistic_config=StatisticConfig(),
            llm_config=LLMConfig(),
            sham_schedule=sham_schedule,
            synthetic_schedule=synthetic_schedule,
        )


def test_round_config_rejects_synthetic_pass_index_out_of_range(
    tmp_path: Path,
) -> None:
    run_dir = _set_up_run_dir(tmp_path)
    checkpoint = CheckpointSpec(
        run_id="round-test", checkpoint_id="ckpt-000001", run_dir=run_dir
    )
    sham_schedule = ShamSchedule(
        entries=(), real_perturbations_per_pass=0, shams_per_pass=0, seed=1
    )
    synthetic_schedule = SyntheticPerturbationSchedule(
        entries=(
            SyntheticPerturbationEntry(
                checkpoint_id="ckpt-000001",
                pass_index=5,  # round only does 2 passes
                synthetic_t=50,
                synthetic_payload={
                    "is_synthetic": True,
                    "is_sham": False,
                },
            ),
        ),
        synthetics_per_pass=1,
        seed=142,
    )
    with pytest.raises(ValidationError, match="pass_index"):
        RoundConfig(
            round_id="r_test",
            checkpoints=(checkpoint,),
            passes_per_checkpoint=2,
            statistic_config=StatisticConfig(),
            llm_config=LLMConfig(),
            sham_schedule=sham_schedule,
            synthetic_schedule=synthetic_schedule,
        )


def test_round_config_rejects_sham_synthetic_cross_collision(
    tmp_path: Path,
) -> None:
    """A (checkpoint, pass, t) slot held by a sham AND a synthetic
    simultaneously is the load-bearing cross-schedule disjointness
    violation. The model validator rejects."""
    run_dir = _set_up_run_dir(tmp_path)
    checkpoint = CheckpointSpec(
        run_id="round-test", checkpoint_id="ckpt-000001", run_dir=run_dir
    )
    # Hand-build colliding schedules at slot (ckpt, pass=0, t=50).
    sham_schedule = ShamSchedule(
        entries=(
            ShamScheduleEntry(
                checkpoint_id="ckpt-000001",
                pass_index=0,
                sham_t=50,
                sham_payload={"is_sham": True},
            ),
        ),
        real_perturbations_per_pass=0,
        shams_per_pass=1,
        seed=42,
    )
    synthetic_schedule = SyntheticPerturbationSchedule(
        entries=(
            SyntheticPerturbationEntry(
                checkpoint_id="ckpt-000001",
                pass_index=0,
                synthetic_t=50,  # collides with the sham
                synthetic_payload={
                    "is_synthetic": True,
                    "is_sham": False,
                },
            ),
        ),
        synthetics_per_pass=1,
        seed=142,
    )
    with pytest.raises(ValidationError, match="collides with a sham"):
        RoundConfig(
            round_id="r_test",
            checkpoints=(checkpoint,),
            passes_per_checkpoint=1,
            statistic_config=StatisticConfig(),
            llm_config=LLMConfig(),
            sham_schedule=sham_schedule,
            synthetic_schedule=synthetic_schedule,
        )


def _build_round_mock_client_with_synthetics(passes: int) -> MockLLMClient:
    """Same as _build_round_mock_client but with claim ranges that
    overlap the synthetic perturbations the schedule will produce.
    Phase 13 admits at synthetic events from the equanimity reading,
    which exercises the synthetic check's admitted-finding path."""
    responses: list[BatchPayload | Exception] = []
    # Claim ranges (0, 200) cover any synthetic_t in [1, 148].
    wide_claim = StructuredClaim(
        claim="example",
        cited_stream="agent_step",
        cited_run_id="round-test",
        cited_episode_range=(0, 1),
        cited_step_range=(0, 200),
        cited_scalar_field="h_t",
        cited_value=0.0,
        falsifier="f",
        paraphrase_stability=None,
        reseed_stability=None,
        faithfulness_status="not_checked",
        judge_ruling="not_judged",
        reading_surface="substrate_side",
        masked_steps_handling="n/a",
    )

    def per_criterion_wide(criterion_id: str) -> _PerCriterionReadingPayload:
        return _PerCriterionReadingPayload(
            criterion_id=criterion_id,
            framework_anchor="buddhist_phenomenology",
            claims=[wide_claim],
            free_text_notes="n",
        )

    active_payload = BatchPayload(
        per_criterion=[
            per_criterion_wide("reflexive_attention"),
            per_criterion_wide("equanimity_perturbation_recovery"),
        ]
    )
    held_out_payload = BatchPayload(
        per_criterion=[per_criterion_wide("second_order_volition")]
    )
    for _ in range(passes):
        responses.append(active_payload)  # active primary
        responses.append(active_payload)  # active adversarial
        responses.append(held_out_payload)  # held-out primary
        responses.append(held_out_payload)  # held-out adversarial
    return MockLLMClient(responses)


def test_run_round_populates_synthetic_findings_summary(tmp_path: Path) -> None:
    """run_round emits a SyntheticFindingsSummary on the RoundResult.
    With wide-range mock claims that cover every synthetic_t, every
    finding admits — verifies the discriminative path."""
    config = _make_round_config_with_synthetic(
        tmp_path, passes_per_checkpoint=2, synthetics_per_pass=2
    )
    client = _build_round_mock_client_with_synthetics(passes=2)
    output_dir = tmp_path / "phase_13_calibration"
    result = run_round(config, output_dir=output_dir, llm_client=client)
    summary = result.synthetic_findings_summary
    # 2 passes × 2 synthetics × 2 criteria × 2 roles = 16 findings.
    assert summary.total_synthetic_events == 16
    # All admit (the wide claim range covers every synthetic_t).
    assert summary.total_admissions == 16
    # By role: 8 primary + 8 adversarial.
    assert summary.admissions_by_role.get("primary") == 8
    assert summary.admissions_by_role.get("adversarial") == 8


def test_run_round_synthetic_findings_empty_with_no_schedule(tmp_path: Path) -> None:
    """Phase 12 caller (no synthetic schedule) gets an empty
    SyntheticFindingsSummary."""
    config = _make_round_config(tmp_path, passes_per_checkpoint=2)
    client = _build_round_mock_client(passes=2)
    output_dir = tmp_path / "phase_12_smoke"
    result = run_round(config, output_dir=output_dir, llm_client=client)
    summary = result.synthetic_findings_summary
    assert summary.total_synthetic_events == 0
    assert summary.total_admissions == 0


def test_run_round_result_with_synthetic_serialization_round_trip(
    tmp_path: Path,
) -> None:
    config = _make_round_config_with_synthetic(tmp_path, passes_per_checkpoint=1)
    client = _build_round_mock_client_with_synthetics(passes=1)
    output_dir = tmp_path / "phase_13_calibration"
    result = run_round(config, output_dir=output_dir, llm_client=client)
    redumped = RoundResult.model_validate_json(result.model_dump_json())
    assert redumped == result
