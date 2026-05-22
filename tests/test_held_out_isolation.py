"""Phase 13 gate test —
:mod:`kind.mirror.calibration.held_out_isolation`.

Covers the held-out isolation study that closes Phase 12's newly-open
§7:

- :class:`HeldOutIsolationConfig` validation: positive
  ``n_isolation_runs``, non-negative ``pass_index``, non-empty
  checkpoint ids.
- :class:`HeldOutIsolationReading` construction via the mock LLM (the
  primary-role single-fragment call returns one reading).
- Distribution histogram computation across multiple isolation runs.
- ``findings`` string non-empty after running; pattern depends on
  whether the isolation runs diversified.
- :class:`HeldOutIsolationStudy` serialization round-trip.
- The held-out-criterion identity assertion (V2 partition must remain
  ``second_order_volition``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from kind.mirror.calibration.held_out_isolation import (
    HeldOutIsolationConfig,
    HeldOutIsolationReading,
    HeldOutIsolationStudy,
    PHASE_13_HELD_OUT_ISOLATION_N_RUNS,
    PHASE_13_HELD_OUT_ISOLATION_PASS_INDEX,
    PHASE_13_HELD_OUT_ISOLATION_SEED,
    run_held_out_isolation_study,
)
from kind.mirror.calibration.round import (
    CheckpointSpec,
    RoundConfig,
    RoundResult,
    run_round,
)
from kind.mirror.calibration.sham_schedule import generate_sham_schedule
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
# Telemetry fixtures (shared shape with test_round.py).
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
                "run_id": "iso-test",
                "checkpoint_id": "ckpt-iso",
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


def _set_up_run_dir(tmp_path: Path, run_id: str = "iso-test") -> Path:
    run_dir = tmp_path / "runs" / run_id
    telemetry_dir = run_dir / "telemetry"
    _write_agent_step_shards(telemetry_dir, _build_agent_step_rows())
    (telemetry_dir / "world_event.jsonl").write_text("")
    return run_dir


# ---------------------------------------------------------------------------
# Mock LLM helpers.
# ---------------------------------------------------------------------------


def _claim_at_surface(
    surface: str,
    cited_step_range: tuple[int, int] | None = (0, 100),
) -> StructuredClaim:
    return StructuredClaim(
        claim=f"isolation claim at {surface}",
        cited_stream="agent_step",
        cited_run_id="iso-test",
        cited_episode_range=(0, 1),
        cited_step_range=cited_step_range,
        cited_scalar_field="h_t",
        cited_value=0.0,
        falsifier="f",
        paraphrase_stability=None,
        reseed_stability=None,
        faithfulness_status="not_checked",
        judge_ruling="not_judged",
        reading_surface=surface,  # type: ignore[arg-type]
        masked_steps_handling="n/a",
    )


def _held_out_payload(
    n_claims: int = 2, surfaces: list[str] | None = None
) -> BatchPayload:
    if surfaces is None:
        surfaces = ["substrate_side", "behavior_side"][:n_claims]
    return BatchPayload(
        per_criterion=[
            _PerCriterionReadingPayload(
                criterion_id="second_order_volition",
                framework_anchor="buddhist_phenomenology",
                claims=[_claim_at_surface(s) for s in surfaces[:n_claims]],
                free_text_notes="iso",
            )
        ]
    )


def _active_payload() -> BatchPayload:
    return BatchPayload(
        per_criterion=[
            _PerCriterionReadingPayload(
                criterion_id="reflexive_attention",
                framework_anchor="buddhist_phenomenology",
                claims=[_claim_at_surface("substrate_side")],
                free_text_notes="x",
            ),
            _PerCriterionReadingPayload(
                criterion_id="equanimity_perturbation_recovery",
                framework_anchor="buddhist_phenomenology",
                claims=[_claim_at_surface("substrate_side")],
                free_text_notes="x",
            ),
        ]
    )


def _phase_12_like_round(tmp_path: Path) -> RoundResult:
    """Build a Phase-12-shaped RoundResult by running the round driver
    against synthetic telemetry. The result is the per-checkpoint
    reference the isolation study reads from."""
    run_dir = _set_up_run_dir(tmp_path)
    checkpoint = CheckpointSpec(
        run_id="iso-test", checkpoint_id="ckpt-iso", run_dir=run_dir
    )
    sham_schedule = generate_sham_schedule(
        checkpoint_ids=("ckpt-iso",),
        passes_per_checkpoint=1,
        real_perturbations_per_pass=0,
        shams_per_pass=0,
        telemetry_length=200,
        seed=42,
    )
    config = RoundConfig(
        round_id="phase_12_iso_round",
        checkpoints=(checkpoint,),
        passes_per_checkpoint=1,
        statistic_config=StatisticConfig(),
        llm_config=LLMConfig(),
        sham_schedule=sham_schedule,
        active_registry=CriterionRegistry(criteria=V2_REGISTRY.active()),
        held_out_registry=CriterionRegistry(criteria=V2_REGISTRY.held_out()),
    )
    # Build the responses for one pass: active primary + adversarial +
    # held-out primary + adversarial = 4 calls.
    responses: list[BatchPayload | Exception] = [
        _active_payload(),
        _active_payload(),
        _held_out_payload(n_claims=2),
        _held_out_payload(n_claims=2),
    ]
    client = MockLLMClient(responses)
    return run_round(config, output_dir=tmp_path / "phase_12_iso", llm_client=client)


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_config_defaults_match_phase_13_commitments() -> None:
    """The module-level Phase 13 constants match the config defaults."""
    config = HeldOutIsolationConfig(
        probe_1_checkpoint_id="ckpt-1",
        probe_1_5_checkpoint_id="ckpt-1-5",
    )
    assert config.pass_index == PHASE_13_HELD_OUT_ISOLATION_PASS_INDEX
    assert config.n_isolation_runs == PHASE_13_HELD_OUT_ISOLATION_N_RUNS
    assert config.seed == PHASE_13_HELD_OUT_ISOLATION_SEED


def test_config_rejects_empty_checkpoint_id() -> None:
    with pytest.raises(ValidationError):
        HeldOutIsolationConfig(
            probe_1_checkpoint_id="",
            probe_1_5_checkpoint_id="ckpt-1-5",
        )


def test_config_rejects_zero_n_isolation_runs() -> None:
    with pytest.raises(ValidationError, match="n_isolation_runs"):
        HeldOutIsolationConfig(
            probe_1_checkpoint_id="c1",
            probe_1_5_checkpoint_id="c2",
            n_isolation_runs=0,
        )


def test_config_rejects_negative_pass_index() -> None:
    with pytest.raises(ValidationError, match="pass_index"):
        HeldOutIsolationConfig(
            probe_1_checkpoint_id="c1",
            probe_1_5_checkpoint_id="c2",
            pass_index=-1,
        )


def test_isolation_reading_construction_with_mock_llm(tmp_path: Path) -> None:
    """End-to-end: build two Phase-12-like RoundResults, run the
    isolation study with a mock LLM, verify the readings are emitted
    correctly."""
    # Two reference rounds, each with one held-out primary reading.
    probe_1_round = _phase_12_like_round(tmp_path / "p1")
    probe_1_5_round = _phase_12_like_round(tmp_path / "p1_5")
    iso_config = HeldOutIsolationConfig(
        probe_1_checkpoint_id="ckpt-iso",
        probe_1_5_checkpoint_id="ckpt-iso",
        n_isolation_runs=3,
    )
    # 2 checkpoints × 3 runs = 6 LLM calls. The held-out fragment is a
    # single-fragment call (held-out criterion only).
    iso_responses: list[BatchPayload | Exception] = [
        _held_out_payload(n_claims=2) for _ in range(6)
    ]
    iso_client = MockLLMClient(iso_responses)
    study = run_held_out_isolation_study(
        iso_config,
        reference_rounds=(probe_1_round, probe_1_5_round),
        llm_client=iso_client,
    )
    assert len(study.probe_1_readings) == 3
    assert len(study.probe_1_5_readings) == 3
    for reading in (
        *study.probe_1_readings,
        *study.probe_1_5_readings,
    ):
        assert isinstance(reading, HeldOutIsolationReading)
        assert reading.criterion_id == "second_order_volition"
        assert reading.reader_role == "primary"
        assert len(reading.reading.claims) == 2


def test_distribution_histograms_computed_correctly(tmp_path: Path) -> None:
    """When the mock returns varying claim counts and surface mixes,
    the histograms capture the variation."""
    probe_1_round = _phase_12_like_round(tmp_path / "p1")
    probe_1_5_round = _phase_12_like_round(tmp_path / "p1_5")
    iso_config = HeldOutIsolationConfig(
        probe_1_checkpoint_id="ckpt-iso",
        probe_1_5_checkpoint_id="ckpt-iso",
        n_isolation_runs=2,
    )
    # 4 LLM calls. Two produce 2-claim 1-sub-1-beh; two produce
    # 1-claim 1-sub.
    iso_responses: list[BatchPayload | Exception] = [
        _held_out_payload(n_claims=2, surfaces=["substrate_side", "behavior_side"]),
        _held_out_payload(n_claims=1, surfaces=["substrate_side"]),
        _held_out_payload(n_claims=2, surfaces=["substrate_side", "behavior_side"]),
        _held_out_payload(n_claims=1, surfaces=["substrate_side"]),
    ]
    iso_client = MockLLMClient(iso_responses)
    study = run_held_out_isolation_study(
        iso_config,
        reference_rounds=(probe_1_round, probe_1_5_round),
        llm_client=iso_client,
    )
    # 4 readings total: 2 with 2 claims, 2 with 1 claim.
    assert study.claim_count_distribution_isolation == {1: 2, 2: 2}
    # Surface distribution: 2 readings as "sub=1,head=0,beh=1"; 2 as
    # "sub=1,head=0,beh=0".
    assert study.surface_distribution_isolation == {
        "sub=1,head=0,beh=1": 2,
        "sub=1,head=0,beh=0": 2,
    }


def test_findings_string_non_empty_after_running(tmp_path: Path) -> None:
    probe_1_round = _phase_12_like_round(tmp_path / "p1")
    probe_1_5_round = _phase_12_like_round(tmp_path / "p1_5")
    iso_config = HeldOutIsolationConfig(
        probe_1_checkpoint_id="ckpt-iso",
        probe_1_5_checkpoint_id="ckpt-iso",
        n_isolation_runs=2,
    )
    iso_responses: list[BatchPayload | Exception] = [
        _held_out_payload(n_claims=2) for _ in range(4)
    ]
    iso_client = MockLLMClient(iso_responses)
    study = run_held_out_isolation_study(
        iso_config,
        reference_rounds=(probe_1_round, probe_1_5_round),
        llm_client=iso_client,
    )
    assert study.findings.strip() != ""
    # The findings string mentions one of the two readings.
    assert ("Reading (a)" in study.findings) or (
        "Reading (b)" in study.findings
    )


def test_findings_reports_reading_a_when_isolation_perfectly_stable(
    tmp_path: Path,
) -> None:
    """All isolation runs produce identical claim count + surface
    distribution → Reading (a) is supported (Phase 7's prose is sharp)."""
    probe_1_round = _phase_12_like_round(tmp_path / "p1")
    probe_1_5_round = _phase_12_like_round(tmp_path / "p1_5")
    iso_config = HeldOutIsolationConfig(
        probe_1_checkpoint_id="ckpt-iso",
        probe_1_5_checkpoint_id="ckpt-iso",
        n_isolation_runs=3,
    )
    # 6 identical responses.
    iso_responses: list[BatchPayload | Exception] = [
        _held_out_payload(n_claims=2) for _ in range(6)
    ]
    iso_client = MockLLMClient(iso_responses)
    study = run_held_out_isolation_study(
        iso_config,
        reference_rounds=(probe_1_round, probe_1_5_round),
        llm_client=iso_client,
    )
    assert "Reading (a) supported" in study.findings


def test_findings_reports_reading_b_when_isolation_diversifies(
    tmp_path: Path,
) -> None:
    """Isolation runs diversify in claim count → Reading (b) is
    supported (prompt-context interference)."""
    probe_1_round = _phase_12_like_round(tmp_path / "p1")
    probe_1_5_round = _phase_12_like_round(tmp_path / "p1_5")
    iso_config = HeldOutIsolationConfig(
        probe_1_checkpoint_id="ckpt-iso",
        probe_1_5_checkpoint_id="ckpt-iso",
        n_isolation_runs=2,
    )
    iso_responses: list[BatchPayload | Exception] = [
        _held_out_payload(n_claims=2),
        _held_out_payload(n_claims=1, surfaces=["behavior_side"]),
        _held_out_payload(n_claims=3, surfaces=["substrate_side", "behavior_side", "head_internal"]),
        _held_out_payload(n_claims=1, surfaces=["substrate_side"]),
    ]
    iso_client = MockLLMClient(iso_responses)
    study = run_held_out_isolation_study(
        iso_config,
        reference_rounds=(probe_1_round, probe_1_5_round),
        llm_client=iso_client,
    )
    assert "Reading (b) supported" in study.findings


def test_isolation_study_serialization_round_trip(tmp_path: Path) -> None:
    probe_1_round = _phase_12_like_round(tmp_path / "p1")
    probe_1_5_round = _phase_12_like_round(tmp_path / "p1_5")
    iso_config = HeldOutIsolationConfig(
        probe_1_checkpoint_id="ckpt-iso",
        probe_1_5_checkpoint_id="ckpt-iso",
        n_isolation_runs=2,
    )
    iso_responses: list[BatchPayload | Exception] = [
        _held_out_payload(n_claims=2) for _ in range(4)
    ]
    iso_client = MockLLMClient(iso_responses)
    study = run_held_out_isolation_study(
        iso_config,
        reference_rounds=(probe_1_round, probe_1_5_round),
        llm_client=iso_client,
    )
    redumped = HeldOutIsolationStudy.model_validate_json(
        study.model_dump_json()
    )
    assert redumped == study


def test_run_held_out_isolation_study_rejects_mismatched_checkpoint(
    tmp_path: Path,
) -> None:
    """If the config's probe_1_checkpoint_id doesn't match the
    reference round's pass-0 checkpoint, the driver raises."""
    probe_1_round = _phase_12_like_round(tmp_path / "p1")
    probe_1_5_round = _phase_12_like_round(tmp_path / "p1_5")
    iso_config = HeldOutIsolationConfig(
        probe_1_checkpoint_id="ckpt-other",  # mismatch
        probe_1_5_checkpoint_id="ckpt-iso",
        n_isolation_runs=2,
    )
    iso_client = MockLLMClient([])
    with pytest.raises(ValueError, match="checkpoint_id"):
        run_held_out_isolation_study(
            iso_config,
            reference_rounds=(probe_1_round, probe_1_5_round),
            llm_client=iso_client,
        )
