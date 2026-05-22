"""Phase 9 gate test — :mod:`kind.mirror.judge_driver`.

End-to-end with a mock LLM and a synthetic Phase 13-shaped
:class:`RoundResult`. The test fixture constructs the RoundResult
directly (bypassing the full Phase 12 ``run_round`` pipeline) so the
test stays fast and the assertions are precise.

Covers:

- end-to-end ``judge_round`` with a mock judge client;
- one ``CriterionJudgment`` per registered criterion (active +
  held-out);
- claim polarity assignments cover every claim in every reading;
- atomic write to ``output_dir/mirror/judgments/{round_id}.json``;
- serialization round-trip on the written JSON;
- the judge driver is read-only against the source round file
  (source unchanged after judging);
- judge call records flow into ``judge_llm_call_records``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from kind.mirror.calibration.round import (
    CheckpointSpec,
    RoundConfig,
    RoundResult,
    ShamFindingsSummary,
)
from kind.mirror.calibration.sham_schedule import ShamSchedule
from kind.mirror.calibration.synthetic_calibration_check import (
    SyntheticFindingsSummary,
)
from kind.mirror.calibration.synthetic_perturbation import (
    SyntheticPerturbationSchedule,
)
from kind.mirror.criteria_v2 import (
    EQUANIMITY_PERTURBATION_RECOVERY,
    REFLEXIVE_ATTENTION,
    SECOND_ORDER_VOLITION,
    V2_REGISTRY,
)
from kind.mirror.judge import (
    ClaimPolarity,
    CriterionJudgment,
    RoundJudgment,
)
from kind.mirror.judge_driver import (
    JUDGMENTS_SUBDIR,
    judge_round,
    load_round_result_from_disk,
)
from kind.mirror.judge_llm_caller import (
    ClaimPolarityAssignmentPayload,
    FalsifierVerdictPayload,
    JudgeBatchPayload,
    JudgePayload,
    MockJudgeLLMClient,
)
from kind.mirror.llm_caller import LLMConfig, MirrorReading
from kind.mirror.orchestrator import PassResult
from kind.mirror.perturbation_align import PerturbationTimeline
from kind.mirror.registry import CriterionRegistry, ReadingSurface
from kind.mirror.statistics import StatisticConfig, StatisticResult
from kind.mirror.structured import StructuredClaim
from kind.observer.pre_reg import PreRegistration


# ---------------------------------------------------------------------------
# Synthetic-RoundResult fixture helpers.
# ---------------------------------------------------------------------------


def _claim(
    *,
    cited_step_range: tuple[int, int] | None = (10, 20),
    cited_scalar_field: str = "h_t",
) -> StructuredClaim:
    return StructuredClaim(
        claim="primary cites the partial-autocorrelation value",
        cited_stream="agent_step",
        cited_run_id="test-run",
        cited_episode_range=(0, 1),
        cited_step_range=cited_step_range,
        cited_scalar_field=cited_scalar_field,
        cited_value=0.42,
        falsifier="the signal does not exceed its shuffled-time control",
        paraphrase_stability=None,
        reseed_stability=None,
        faithfulness_status="not_checked",
        judge_ruling="not_judged",
        reading_surface="head_internal",
        masked_steps_handling="n/a",
    )


def _reading(reader_role: str = "advocate") -> MirrorReading:
    return MirrorReading(
        run_id="test-run",
        timestamp_ms=1000,
        reader_role=reader_role,  # type: ignore[arg-type]
        paired_reading_id="pair",
        framework_anchor="buddhist_phenomenology",
        baseline_flag="genuine",
        digest_run_id="test-run",
        digest_episode_range=(0, 5),
        claims=[_claim()],
        free_text_notes="",
    )


def _stat(name: str, value: Any = 0.5) -> StatisticResult:
    return StatisticResult(
        signal_name=name,
        value=value,
        estimator="x",
        n_samples=10,
        notes="",
    )


def _pre_reg(*, run_id: str, target_ids: list[str], other_ids: list[str]) -> PreRegistration:
    """Build a PreRegistration that's well-formed for the registered v2
    criteria."""
    signal_mappings: dict[str, list[str]] = {}
    falsifiers: dict[str, str] = {}
    scalar_checks: dict[str, list[str]] = {}
    reading_surfaces_per_criterion: dict[str, list[ReadingSurface]] = {}
    for cid in target_ids:
        crit = V2_REGISTRY.get(cid)
        signal_mappings[cid] = [m.name for m in crit.signal_mappings]
        falsifiers[cid] = crit.falsifier
        scalar_checks[cid] = [
            f"{m.name}::estimator" for m in crit.signal_mappings
        ]
        reading_surfaces_per_criterion[cid] = sorted(
            crit.reading_surfaces, key=lambda s: s.value
        )
    expected_outcome_per_surface = {
        s: "to be determined" for s in ReadingSurface
    }
    return PreRegistration(
        run_id=run_id,
        timestamp_ms=1000,
        criteria_active=target_ids,
        criteria_held_out=other_ids,
        signal_mappings=signal_mappings,
        falsifiers=falsifiers,
        scalar_checks=scalar_checks,
        reading_surfaces_per_criterion=reading_surfaces_per_criterion,
        asymmetry_of_access="test",
        builder_mode="skeptic",
        expected_outcome="test",
        expected_outcome_per_surface=expected_outcome_per_surface,
        substrate_decisions_off_table=["test"],
        column_init="unknown",
        new_actor_readable_interfaces_added=[],
    )


def _pass_result(
    *,
    run_id: str = "test-run",
    checkpoint_id: str = "ckpt-test",
) -> PassResult:
    """Build a PassResult shaped like Phase 13's: one reading per
    active criterion (two), one reading per held-out criterion (one);
    one statistic result per signal across the V2 registry; an empty
    perturbation timeline; no sham findings."""
    active_ids = [c.id for c in V2_REGISTRY.active()]
    held_out_ids = [c.id for c in V2_REGISTRY.held_out()]
    return PassResult(
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        timestamp_ms=1000,
        active_pre_registration=_pre_reg(
            run_id=run_id, target_ids=active_ids, other_ids=held_out_ids
        ),
        active_primary_readings=tuple(_reading() for _ in active_ids),
        active_adversarial_readings=tuple(
            _reading("skeptic") for _ in active_ids
        ),
        held_out_pre_registration=_pre_reg(
            run_id=run_id, target_ids=held_out_ids, other_ids=active_ids
        ),
        held_out_primary_readings=tuple(_reading() for _ in held_out_ids),
        held_out_adversarial_readings=tuple(
            _reading("skeptic") for _ in held_out_ids
        ),
        statistic_results=(
            _stat("latent_self_reference_t", 0.7),
            _stat("dream_self_reference_t", 0.6),
            _stat("recovery_lag_steps", [5.0]),
            _stat(
                "policy_entropy_t",
                {
                    "dip_and_recover": 1.0, "collapse": 0.0,
                    "stays_elevated": 0.0, "no_response": 0.0,
                },
            ),
            _stat(
                "posterior_kl_t",
                {
                    "spike_and_decay": 1.0, "ratchet": 0.0,
                    "no_response": 0.0, "oscillation": 0.0,
                },
            ),
            _stat(
                "policy_modulation_t",
                {"contrast_magnitude": 0.5, "observation_only_baseline": 0.1},
            ),
            _stat("latent_regime_indicator_t", [0.0, 1.0, 2.0, 3.0]),
        ),
        perturbation_timeline=PerturbationTimeline(
            events=(), run_id=run_id, checkpoint_id=checkpoint_id
        ),
        sham_calibration_findings=(),
        notes="",
    )


def _synth_round_result(
    *,
    round_id: str = "test_round",
    n_passes: int = 2,
    run_id: str = "test-run",
    checkpoint_id: str = "ckpt-test",
    run_dir: Path | None = None,
) -> RoundResult:
    """Construct a synthetic Phase 13-shaped RoundResult."""
    if run_dir is None:
        run_dir = Path("/tmp/test-judge-driver-run")
    config = RoundConfig(
        round_id=round_id,
        checkpoints=(
            CheckpointSpec(
                run_id=run_id,
                checkpoint_id=checkpoint_id,
                run_dir=run_dir,
            ),
        ),
        passes_per_checkpoint=n_passes,
        statistic_config=StatisticConfig(),
        llm_config=LLMConfig(),
        sham_schedule=ShamSchedule(
            entries=(),
            real_perturbations_per_pass=0,
            shams_per_pass=0,
            seed=1,
        ),
        synthetic_schedule=SyntheticPerturbationSchedule(
            entries=(), synthetics_per_pass=0, seed=0
        ),
    )
    pass_results = tuple(
        _pass_result(run_id=run_id, checkpoint_id=checkpoint_id)
        for _ in range(n_passes)
    )
    return RoundResult(
        round_id=round_id,
        round_config=config,
        pass_results=pass_results,
        sham_findings_summary=ShamFindingsSummary(
            by_criterion={},
            by_checkpoint={},
            by_role={},
            total=0,
            total_sham_events=0,
        ),
        synthetic_findings_summary=SyntheticFindingsSummary(
            total_synthetic_events=0,
            total_admissions=0,
            admissions_by_criterion={},
            admissions_by_checkpoint={},
            admissions_by_role={},
            mean_recovery_lag_at_admissions=None,
            mean_recovery_lag_at_non_admissions=None,
        ),
        llm_call_records=(),
        round_wallclock_ms=0,
        notes="",
    )


def _write_round_result_to_disk(
    round_result: RoundResult, mirror_dir: Path
) -> Path:
    """Write the round result to ``mirror_dir/rounds/{round_id}.json``
    (mimicking the on-disk shape the driver expects)."""
    rounds_dir = mirror_dir / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    path = rounds_dir / f"{round_result.round_id}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(round_result.model_dump(mode="json"), fh, indent=2)
    return path


def _judge_payload(
    criterion_id: str,
    verdict: str = "satisfied",
    confidence: float = 0.8,
    polarity_count: int = 1,
) -> JudgePayload:
    return JudgePayload(
        criterion_id=criterion_id,
        claim_polarity_assignments=tuple(
            ClaimPolarityAssignmentPayload(
                pass_index=i,
                reader_role="primary",
                claim_index=0,
                cited_step_range=(10, 20),
                polarity=ClaimPolarity.SUPPORTIVE,
                polarity_rationale="claim cites the signal",
            )
            for i in range(polarity_count)
        ),
        falsifier_verdicts=(
            FalsifierVerdictPayload(
                falsifier_id=f"{criterion_id}_v1",
                passes_supporting=tuple(range(polarity_count)),
                passes_refuting=(),
                passes_non_falsifying=(),
                passes_ambiguous=(),
            ),
        ),
        verdict=verdict,  # type: ignore[arg-type]
        confidence=confidence,
        rationale="rationale",
    )


def _full_judge_batch_for_v2() -> JudgeBatchPayload:
    """Build a JudgeBatchPayload with one per-criterion entry per v2
    criterion (active first, held-out last)."""
    return JudgeBatchPayload(
        per_criterion=[
            _judge_payload("reflexive_attention", polarity_count=2),
            _judge_payload(
                "equanimity_perturbation_recovery",
                verdict="non_falsifying",
                polarity_count=2,
            ),
            _judge_payload(
                "second_order_volition", verdict="ambiguous", polarity_count=2
            ),
        ]
    )


# ---------------------------------------------------------------------------
# End-to-end tests.
# ---------------------------------------------------------------------------


def test_judge_round_end_to_end_returns_round_judgment(tmp_path: Path) -> None:
    round_result = _synth_round_result(round_id="test_round_e2e", n_passes=2)
    round_path = _write_round_result_to_disk(round_result, tmp_path / "mirror")

    client = MockJudgeLLMClient(_full_judge_batch_for_v2())
    judgment = judge_round(
        round_path,
        output_dir=tmp_path,
        llm_config=LLMConfig(),
        llm_client=client,
    )
    assert isinstance(judgment, RoundJudgment)
    assert judgment.round_id == "test_round_e2e"


def test_judge_round_produces_one_judgment_per_v2_criterion(
    tmp_path: Path,
) -> None:
    round_result = _synth_round_result(n_passes=2)
    round_path = _write_round_result_to_disk(round_result, tmp_path / "mirror")
    client = MockJudgeLLMClient(_full_judge_batch_for_v2())
    judgment = judge_round(
        round_path,
        output_dir=tmp_path,
        llm_config=LLMConfig(),
        llm_client=client,
    )
    ids = {cj.criterion_id for cj in judgment.criterion_judgments}
    assert ids == {
        "reflexive_attention",
        "equanimity_perturbation_recovery",
        "second_order_volition",
    }


def test_judge_round_carries_framework_per_criterion(tmp_path: Path) -> None:
    round_result = _synth_round_result(n_passes=2)
    round_path = _write_round_result_to_disk(round_result, tmp_path / "mirror")
    client = MockJudgeLLMClient(_full_judge_batch_for_v2())
    judgment = judge_round(
        round_path,
        output_dir=tmp_path,
        llm_config=LLMConfig(),
        llm_client=client,
    )
    framework_by_id = {cj.criterion_id: cj.framework for cj in judgment.criterion_judgments}
    assert framework_by_id["reflexive_attention"] == REFLEXIVE_ATTENTION.framework
    assert (
        framework_by_id["equanimity_perturbation_recovery"]
        == EQUANIMITY_PERTURBATION_RECOVERY.framework
    )
    assert framework_by_id["second_order_volition"] == SECOND_ORDER_VOLITION.framework


def test_judge_round_polarity_assignments_carry_criterion_id(
    tmp_path: Path,
) -> None:
    """Every :class:`ClaimPolarityAssignment` on a
    :class:`CriterionJudgment` has the matching ``criterion_id``."""
    round_result = _synth_round_result(n_passes=2)
    round_path = _write_round_result_to_disk(round_result, tmp_path / "mirror")
    client = MockJudgeLLMClient(_full_judge_batch_for_v2())
    judgment = judge_round(
        round_path,
        output_dir=tmp_path,
        llm_config=LLMConfig(),
        llm_client=client,
    )
    for cj in judgment.criterion_judgments:
        for pa in cj.claim_polarity_assignments:
            assert pa.criterion_id == cj.criterion_id


def test_judge_round_writes_to_judgments_subdir(tmp_path: Path) -> None:
    round_result = _synth_round_result(round_id="test_write", n_passes=2)
    round_path = _write_round_result_to_disk(round_result, tmp_path / "mirror")
    client = MockJudgeLLMClient(_full_judge_batch_for_v2())
    judge_round(
        round_path,
        output_dir=tmp_path,
        llm_config=LLMConfig(),
        llm_client=client,
    )
    judgments_dir = tmp_path / "mirror" / JUDGMENTS_SUBDIR
    assert judgments_dir.is_dir()
    judgment_file = judgments_dir / "test_write.json"
    assert judgment_file.is_file()


def test_judge_round_written_file_round_trips(tmp_path: Path) -> None:
    round_result = _synth_round_result(round_id="rt", n_passes=2)
    round_path = _write_round_result_to_disk(round_result, tmp_path / "mirror")
    client = MockJudgeLLMClient(_full_judge_batch_for_v2())
    judgment_returned = judge_round(
        round_path,
        output_dir=tmp_path,
        llm_config=LLMConfig(),
        llm_client=client,
    )
    written_path = tmp_path / "mirror" / JUDGMENTS_SUBDIR / "rt.json"
    with written_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    reloaded = RoundJudgment.model_validate(payload)
    assert reloaded == judgment_returned


def test_judge_round_does_not_mutate_source_file(tmp_path: Path) -> None:
    """Read-only contract: the source round file on disk is byte-
    identical before and after the judge runs."""
    round_result = _synth_round_result(n_passes=2)
    round_path = _write_round_result_to_disk(round_result, tmp_path / "mirror")
    before = round_path.read_bytes()
    client = MockJudgeLLMClient(_full_judge_batch_for_v2())
    judge_round(
        round_path,
        output_dir=tmp_path,
        llm_config=LLMConfig(),
        llm_client=client,
    )
    after = round_path.read_bytes()
    assert before == after


def test_judge_round_call_records_flow_into_judgment(tmp_path: Path) -> None:
    """The driver's audit collector threads call records onto the
    :attr:`RoundJudgment.judge_llm_call_records` tuple."""
    round_result = _synth_round_result(n_passes=2)
    round_path = _write_round_result_to_disk(round_result, tmp_path / "mirror")
    client = MockJudgeLLMClient(_full_judge_batch_for_v2())
    judgment = judge_round(
        round_path,
        output_dir=tmp_path,
        llm_config=LLMConfig(),
        llm_client=client,
    )
    # One batched call → one success record.
    assert len(judgment.judge_llm_call_records) == 1
    assert judgment.judge_llm_call_records[0].outcome == "success"


def test_judge_round_load_from_disk_round_trips(tmp_path: Path) -> None:
    round_result = _synth_round_result(round_id="rt2", n_passes=2)
    round_path = _write_round_result_to_disk(round_result, tmp_path / "mirror")
    reloaded = load_round_result_from_disk(round_path)
    assert reloaded.round_id == "rt2"


def test_load_round_result_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_round_result_from_disk(tmp_path / "nonexistent.json")


def test_load_round_result_raises_on_malformed_json(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_round_result_from_disk(bad_path)


def test_judge_round_summary_names_round_context(tmp_path: Path) -> None:
    """The round_config_summary on the judgment names the key round
    fields the journal entry will inspect."""
    round_result = _synth_round_result(round_id="sum_test", n_passes=3)
    round_path = _write_round_result_to_disk(round_result, tmp_path / "mirror")
    client = MockJudgeLLMClient(_full_judge_batch_for_v2())
    judgment = judge_round(
        round_path,
        output_dir=tmp_path,
        llm_config=LLMConfig(),
        llm_client=client,
    )
    summary = judgment.round_config_summary
    assert "sum_test" in summary
    assert "passes_per_checkpoint=3" in summary
    assert "active_criteria" in summary
    assert "held_out_criteria" in summary
