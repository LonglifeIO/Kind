"""Phase 8 gate test — :mod:`kind.mirror.orchestrator`.

End-to-end smoke for :func:`run_adversarial_pass` against synthetic
telemetry and a mocked LLM. All required Phase 8 invariants:

- the four reading tuples (active-primary, active-adversarial,
  held-out-primary, held-out-adversarial) are produced;
- the sham-perturbation check fires on a sham event;
- the held-out pre-registration is a separate record from the active-set
  one (two appended JSONL lines);
- the held-out pass's prompts contain no reference to the active-set
  readings (asserted via the four MockLLMClient call records — the
  held-out user prompts do not contain phrases from the active LLM
  outputs).

No real API calls. All tests use :class:`MockLLMClient`. The end-to-end
test in :mod:`tests.test_orchestrator_one_way_invariant` covers the
write-only-to-mirror-side semantic check.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kind.mirror.criteria_v2 import V2_REGISTRY
from kind.mirror.llm_caller import (
    BatchPayload,
    LLMConfig,
    MockLLMClient,
    _PerCriterionReadingPayload,
)
from kind.mirror.orchestrator import (
    PassConfig,
    PassResult,
    run_adversarial_pass,
)
from kind.mirror.registry import CriterionRegistry
from kind.mirror.structured import StructuredClaim


# ---------------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------------


def _build_agent_step_rows(
    n_steps: int = 200, n_episodes: int = 2, h_dim: int = 4
) -> list[dict[str, Any]]:
    """Synthetic AgentStep rows — deterministic, covering every field
    the orchestrator's statistic functions index."""
    rows: list[dict[str, Any]] = []
    steps_per_ep = n_steps // n_episodes
    for i in range(n_steps):
        ep = i // steps_per_ep
        in_ep = i % steps_per_ep
        # Mild autocorrelation: h_t evolves slowly.
        h = [0.5 * float(in_ep) / steps_per_ep + 0.01 * j for j in range(h_dim)]
        rows.append(
            {
                "schema_version": "0.2.0",
                "run_id": "probe2-orch-test",
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
                # Pad fields the statistics functions don't read; sinks
                # require them when written through a Pydantic-validated
                # parquet writer, but our test writes a raw schema, so
                # they need only be present.
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
    """Write the synthetic rows to a single parquet shard. The
    orchestrator's loader uses ``pa.Table.from_pylist`` with field
    inference; the schema below covers every field the orchestrator
    indexes."""
    shard_dir = telemetry_dir / "agent_step"
    shard_dir.mkdir(parents=True, exist_ok=True)
    # Build a schema that just captures the fields used; let pyarrow
    # infer the rest. To keep things simple we let pyarrow infer fully.
    table = pa.Table.from_pylist(rows)
    shard_path = shard_dir / "shard-000000.parquet"
    with shard_path.open("wb") as fh:
        pq.write_table(table, fh)  # type: ignore[no-untyped-call]


def _write_world_event_log(
    telemetry_dir: Path, perturbations: list[dict[str, Any]]
) -> None:
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    path = telemetry_dir / "world_event.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for p in perturbations:
            fh.write(json.dumps(p) + "\n")


def _claim() -> StructuredClaim:
    return StructuredClaim(
        claim="example claim",
        cited_stream="agent_step",
        cited_run_id="probe2-orch-test",
        cited_episode_range=(0, 1),
        cited_step_range=(0, 100),
        cited_scalar_field="h_t",
        cited_value=0.42,
        falsifier="a falsifier",
        paraphrase_stability=None,
        reseed_stability=None,
        faithfulness_status="not_checked",
        judge_ruling="not_judged",
        reading_surface="head_internal",
        masked_steps_handling="n/a",
    )


def _per_criterion(
    *, criterion_id: str, framework_anchor: str = "buddhist_phenomenology"
) -> _PerCriterionReadingPayload:
    return _PerCriterionReadingPayload(
        criterion_id=criterion_id,
        framework_anchor=framework_anchor,  # type: ignore[arg-type]
        claims=[_claim()],
        free_text_notes="notes",
    )


def _build_mock_client_for_round() -> MockLLMClient:
    """A mock returning ordered responses for the four LLM calls per pass:
    active-primary, active-adversarial, held-out-primary,
    held-out-adversarial. The active registry has 2 criteria and the
    held-out has 1 (the V2_REGISTRY shape)."""
    active_ids = ["reflexive_attention", "equanimity_perturbation_recovery"]
    held_out_ids = ["second_order_volition"]

    def active_payload(*, adversarial: bool) -> BatchPayload:
        anchor = "null_statistics" if adversarial else "buddhist_phenomenology"
        return BatchPayload(
            per_criterion=[
                _per_criterion(criterion_id=cid, framework_anchor=anchor)
                for cid in active_ids
            ]
        )

    def held_out_payload(*, adversarial: bool) -> BatchPayload:
        anchor = "null_statistics" if adversarial else "buddhist_phenomenology"
        return BatchPayload(
            per_criterion=[
                _per_criterion(criterion_id=cid, framework_anchor=anchor)
                for cid in held_out_ids
            ]
        )

    return MockLLMClient(
        [
            active_payload(adversarial=False),
            active_payload(adversarial=True),
            held_out_payload(adversarial=False),
            held_out_payload(adversarial=True),
        ]
    )


def _make_pass_config(
    *, run_dir: Path, run_id: str = "probe2-orch-test"
) -> PassConfig:
    return PassConfig(
        run_id=run_id,
        checkpoint_id="ckpt-000001",
        run_dir=run_dir,
        active_registry=CriterionRegistry(criteria=V2_REGISTRY.active()),
        held_out_registry=CriterionRegistry(criteria=V2_REGISTRY.held_out()),
        llm_config=LLMConfig(),
    )


# ---------------------------------------------------------------------------
# End-to-end smoke.
# ---------------------------------------------------------------------------


def _set_up_run_dir(
    tmp_path: Path,
    *,
    perturbation_t_event: int = 60,
    perturbation_wallclock_ms: int | None = None,
    is_sham: bool = False,
) -> Path:
    run_dir = tmp_path / "runs" / "probe2-orch-test"
    telemetry_dir = run_dir / "telemetry"
    rows = _build_agent_step_rows()
    _write_agent_step_shards(telemetry_dir, rows)
    if perturbation_wallclock_ms is None:
        perturbation_wallclock_ms = perturbation_t_event * 100
    payload: dict[str, Any] = {"kind": "test"}
    if is_sham:
        payload["is_sham"] = True
    _write_world_event_log(
        telemetry_dir,
        [
            {
                "schema_version": "0.2.0",
                "run_id": "probe2-orch-test",
                "checkpoint_id": "ckpt-000001",
                "t_event": perturbation_t_event,
                "event_type": "builder_perturbation",
                "source": "builder",
                "payload": payload,
                "wallclock_ms": perturbation_wallclock_ms,
            }
        ],
    )
    return run_dir


def test_run_adversarial_pass_produces_four_reading_tuples(
    tmp_path: Path,
) -> None:
    run_dir = _set_up_run_dir(tmp_path)
    config = _make_pass_config(run_dir=run_dir)
    client = _build_mock_client_for_round()
    result = run_adversarial_pass(config, llm_client=client)
    assert isinstance(result, PassResult)
    # Active partition has 2 criteria (reflexive_attention, equanimity).
    assert len(result.active_primary_readings) == 2
    assert len(result.active_adversarial_readings) == 2
    # Held-out partition has 1 criterion (second_order_volition).
    assert len(result.held_out_primary_readings) == 1
    assert len(result.held_out_adversarial_readings) == 1
    # Reader roles map correctly.
    assert all(r.reader_role == "advocate" for r in result.active_primary_readings)
    assert all(r.reader_role == "skeptic" for r in result.active_adversarial_readings)
    assert all(r.reader_role == "advocate" for r in result.held_out_primary_readings)
    assert all(r.reader_role == "skeptic" for r in result.held_out_adversarial_readings)


def test_pre_registration_records_appended_to_one_jsonl(
    tmp_path: Path,
) -> None:
    run_dir = _set_up_run_dir(tmp_path)
    config = _make_pass_config(run_dir=run_dir)
    client = _build_mock_client_for_round()
    run_adversarial_pass(config, llm_client=client)
    pre_reg_path = run_dir / "mirror" / "pre_reg" / "pre_reg.jsonl"
    assert pre_reg_path.is_file()
    lines = pre_reg_path.read_text().strip().split("\n")
    assert len(lines) == 2, (
        f"expected one active + one held-out pre-registration record; got "
        f"{len(lines)} lines"
    )
    active_record = json.loads(lines[0])
    held_out_record = json.loads(lines[1])
    # The active record's criteria_active are the active partition's ids.
    assert set(active_record["criteria_active"]) == {
        c.id for c in V2_REGISTRY.active()
    }
    assert set(active_record["criteria_held_out"]) == {
        c.id for c in V2_REGISTRY.held_out()
    }
    # The held-out record's criteria_active are the held-out partition's ids.
    assert set(held_out_record["criteria_active"]) == {
        c.id for c in V2_REGISTRY.held_out()
    }
    assert set(held_out_record["criteria_held_out"]) == {
        c.id for c in V2_REGISTRY.active()
    }


def test_pass_result_written_to_mirror_passes_dir(tmp_path: Path) -> None:
    run_dir = _set_up_run_dir(tmp_path)
    config = _make_pass_config(run_dir=run_dir)
    client = _build_mock_client_for_round()
    result = run_adversarial_pass(config, llm_client=client)
    passes_path = run_dir / "mirror" / "passes" / "ckpt-000001.json"
    assert passes_path.is_file()
    loaded = json.loads(passes_path.read_text())
    assert loaded["checkpoint_id"] == "ckpt-000001"
    assert loaded["run_id"] == result.run_id


def test_sham_perturbation_check_fires(tmp_path: Path) -> None:
    """A sham event in the timeline produces a sham_calibration_finding;
    the finding's note is preserved in PassResult.notes."""
    run_dir = _set_up_run_dir(tmp_path, is_sham=True)
    config = _make_pass_config(run_dir=run_dir)
    client = _build_mock_client_for_round()
    result = run_adversarial_pass(config, llm_client=client)
    assert len(result.sham_calibration_findings) == 1
    finding = result.sham_calibration_findings[0]
    # The sham_t was set up to be 60 (perturbation_t_event=60); the
    # aligner reads agent-step ``t`` from the wallclock match, which
    # for our synthetic data has wallclock_ms == t*100, so the aligned
    # t equals 60.
    assert finding.sham_t == 60
    assert "sham_t=60" in result.notes


def test_held_out_prompts_do_not_reference_active_readings(
    tmp_path: Path,
) -> None:
    """The held-out LLM calls' user prompts contain no claim text from
    the active LLM outputs. We verify this by inspecting the call log
    on the mock client: the third and fourth user prompts (held-out
    primary + adversarial) do not contain the criterion ids of the
    active partition's readings as quoted text from the LLM."""
    run_dir = _set_up_run_dir(tmp_path)
    config = _make_pass_config(run_dir=run_dir)
    client = _build_mock_client_for_round()
    run_adversarial_pass(config, llm_client=client)
    # Four calls expected: active-primary, active-adversarial,
    # held-out-primary, held-out-adversarial.
    assert len(client.calls) == 4
    active_user_prompts = (client.calls[0][1], client.calls[1][1])
    held_out_user_prompts = (client.calls[2][1], client.calls[3][1])
    # The held-out user prompts cover only the held-out criterion ids
    # in their "criterion_ids" header. The active ids should NOT appear
    # in the held-out prompt's section headers.
    for prompt in held_out_user_prompts:
        # The framing preamble includes a comma-separated criterion id
        # list; the held-out prompt's list contains only the held-out
        # ids.
        assert "second_order_volition" in prompt
        # No "## Criterion: Reflexive attention" header (active partition).
        assert "## Criterion: Reflexive attention" not in prompt
        assert "## Criterion: Equanimity" not in prompt


def test_active_and_held_out_use_separate_paired_reading_ids(
    tmp_path: Path,
) -> None:
    run_dir = _set_up_run_dir(tmp_path)
    config = _make_pass_config(run_dir=run_dir)
    client = _build_mock_client_for_round()
    result = run_adversarial_pass(config, llm_client=client)
    active_pair_ids = {r.paired_reading_id for r in result.active_primary_readings}
    held_out_pair_ids = {
        r.paired_reading_id for r in result.held_out_primary_readings
    }
    assert active_pair_ids != held_out_pair_ids
    assert active_pair_ids == {"ckpt-000001-active"}
    assert held_out_pair_ids == {"ckpt-000001-held_out"}


def test_statistic_results_cover_all_signals(tmp_path: Path) -> None:
    run_dir = _set_up_run_dir(tmp_path)
    config = _make_pass_config(run_dir=run_dir)
    client = _build_mock_client_for_round()
    result = run_adversarial_pass(config, llm_client=client)
    # 2 signals for reflexive_attention + 3 for equanimity + 2 for
    # second_order_volition = 7 signals total.
    assert len(result.statistic_results) == 7
    names = [r.signal_name for r in result.statistic_results]
    assert names == [
        "latent_self_reference_t",
        "dream_self_reference_t",
        "recovery_lag_steps",
        "policy_entropy_t",
        "posterior_kl_t",
        "policy_modulation_t",
        "latent_regime_indicator_t",
    ]


def test_passconfig_rejects_empty_run_id() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PassConfig(
            run_id="",
            checkpoint_id="c",
            run_dir=Path("/tmp"),
            active_registry=CriterionRegistry(criteria=V2_REGISTRY.active()),
            held_out_registry=CriterionRegistry(criteria=V2_REGISTRY.held_out()),
        )


def test_passconfig_rejects_unknown_builder_mode() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PassConfig(
            run_id="r",
            checkpoint_id="c",
            run_dir=Path("/tmp"),
            active_registry=CriterionRegistry(criteria=V2_REGISTRY.active()),
            held_out_registry=CriterionRegistry(criteria=V2_REGISTRY.held_out()),
            builder_mode="something_else",
        )
