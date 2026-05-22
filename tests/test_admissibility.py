"""Phase 12 gate test — :mod:`kind.mirror.admissibility`.

No LLM calls. The consumer is deterministic; the suite pins:

- the frozen + ``extra="forbid"`` invariants on
  :class:`~kind.mirror.admissibility.AdmissibilityVerdict` and
  :class:`~kind.mirror.admissibility.AdmissibilityBatchResult`;
- the batch-result counts-sum-to-``n_readings_total`` model validator;
- the AND-conjunction join across the four cases (both pass, only
  faithfulness fails, only stability fails, both fail);
- the no-stability-result case (vacuous stability admissibility,
  faithfulness alone gates, the dedicated ``n_inadmissible_no_stability``
  bucket);
- the empty-inputs vacuous case;
- the ``(criterion_id, reader_role, checkpoint_id)`` stability index;
- the per-surface conjunction across a multi-surface criterion;
- the audit-JSONL emission and round-trip;
- the convenience loader against on-disk fixtures;
- the input-immutability invariant;
- the verdict-notes-describe-the-join contract.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from kind.mirror.admissibility import (
    AdmissibilityBatchResult,
    AdmissibilityVerdict,
    compute_admissibility,
    load_admissibility_inputs,
)
from kind.mirror.faithfulness import FaithfulnessResult
from kind.mirror.llm_caller import PassRole
from kind.mirror.registry import ReadingSurface
from kind.mirror.stability import StabilityResult


# ---------------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------------


def _faithfulness(
    *,
    criterion_id: str = "reflexive_attention",
    reader_role: PassRole = "primary",
    pass_index: int = 0,
    checkpoint_id: str = "ckpt-1",
    run_id: str = "r",
    admissible: bool = True,
    rate: float = 1.0,
) -> FaithfulnessResult:
    return FaithfulnessResult(
        criterion_id=criterion_id,
        reader_role=reader_role,
        pass_index=pass_index,
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        assignments=(),
        n_claims_total=0,
        n_resolved=0,
        n_unresolved_field=0,
        n_unresolved_value=0,
        n_unresolved_range=0,
        faithfulness_rate=rate,
        admissible=admissible,
        wallclock_ms=0,
        notes="synthetic faithfulness fixture",
    )


def _stability(
    *,
    criterion_id: str = "reflexive_attention",
    reader_role: PassRole = "primary",
    checkpoint_id: str = "ckpt-1",
    run_id: str = "r",
    admissible_per_surface: dict[ReadingSurface, bool] | None = None,
) -> StabilityResult:
    if admissible_per_surface is None:
        admissible_per_surface = {ReadingSurface.HEAD_INTERNAL: True}
    surfaces = list(admissible_per_surface)
    return StabilityResult(
        paraphrase_agreement_per_surface={s: 1.0 for s in surfaces},
        reseed_agreement_per_surface={s: 1.0 for s in surfaces},
        n_paraphrases=2,
        n_reseeds=2,
        structured_field_agreement_per_claim=(),
        admissible_per_surface=dict(admissible_per_surface),
        paraphrase_readings=(),
        reseed_readings=(),
        criterion_id=criterion_id,
        reader_role=reader_role,
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        wallclock_ms=0,
    )


def _verdict(
    *,
    admissible: bool = True,
    faithfulness_admissible: bool = True,
    stability_admissible_all_surfaces: bool = True,
    stability_admissible_per_surface: dict[ReadingSurface, bool] | None = None,
) -> AdmissibilityVerdict:
    if stability_admissible_per_surface is None:
        stability_admissible_per_surface = {ReadingSurface.HEAD_INTERNAL: True}
    return AdmissibilityVerdict(
        pass_index=0,
        criterion_id="reflexive_attention",
        reader_role="primary",
        run_id="r",
        checkpoint_id="ckpt-1",
        faithfulness_admissible=faithfulness_admissible,
        faithfulness_rate=1.0,
        stability_admissible_per_surface=stability_admissible_per_surface,
        stability_admissible_all_surfaces=stability_admissible_all_surfaces,
        admissible=admissible,
        notes="synthetic verdict fixture",
        wallclock_ms=0,
    )


# ---------------------------------------------------------------------------
# Pydantic invariants.
# ---------------------------------------------------------------------------


def test_admissibility_verdict_is_frozen() -> None:
    verdict = _verdict()
    with pytest.raises(ValidationError):
        verdict.admissible = False


def test_admissibility_verdict_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        AdmissibilityVerdict(
            pass_index=0,
            criterion_id="reflexive_attention",
            reader_role="primary",
            run_id="r",
            checkpoint_id="ckpt-1",
            faithfulness_admissible=True,
            faithfulness_rate=1.0,
            stability_admissible_per_surface={},
            stability_admissible_all_surfaces=True,
            admissible=True,
            notes="x",
            wallclock_ms=0,
            unexpected_field="nope",  # type: ignore[call-arg]
        )


def test_admissibility_verdict_notes_non_empty() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        AdmissibilityVerdict(
            pass_index=0,
            criterion_id="reflexive_attention",
            reader_role="primary",
            run_id="r",
            checkpoint_id="ckpt-1",
            faithfulness_admissible=True,
            faithfulness_rate=1.0,
            stability_admissible_per_surface={},
            stability_admissible_all_surfaces=True,
            admissible=True,
            notes="   ",
            wallclock_ms=0,
        )


def test_admissibility_batch_result_is_frozen() -> None:
    batch = compute_admissibility(
        faithfulness_results=(), stability_results=(), run_id="r"
    )
    with pytest.raises(ValidationError):
        batch.n_readings_total = 5


def test_admissibility_batch_result_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        AdmissibilityBatchResult(
            verdicts=(),
            n_readings_total=0,
            n_admissible=0,
            n_inadmissible_faithfulness=0,
            n_inadmissible_stability=0,
            n_inadmissible_both=0,
            n_inadmissible_no_stability=0,
            admissibility_rate=1.0,
            notes="x",
            wallclock_ms=0,
            unexpected_field="nope",  # type: ignore[call-arg]
        )


def test_admissibility_batch_result_counts_sum_to_n_readings_total() -> None:
    # n_readings_total is 5 but the buckets only sum to 4 — the model
    # validator should reject this.
    with pytest.raises(ValidationError, match="do not sum"):
        AdmissibilityBatchResult(
            verdicts=(_verdict(), _verdict(), _verdict(), _verdict()),
            n_readings_total=5,
            n_admissible=1,
            n_inadmissible_faithfulness=1,
            n_inadmissible_stability=1,
            n_inadmissible_both=1,
            n_inadmissible_no_stability=0,
            admissibility_rate=0.2,
            notes="x",
            wallclock_ms=0,
        )


# ---------------------------------------------------------------------------
# compute_admissibility — the four join cases.
# ---------------------------------------------------------------------------


def test_compute_admissibility_both_pass() -> None:
    faithfulness = _faithfulness(admissible=True, rate=1.0)
    stability = _stability(
        admissible_per_surface={ReadingSurface.HEAD_INTERNAL: True}
    )
    batch = compute_admissibility(
        faithfulness_results=(faithfulness,),
        stability_results=(stability,),
        run_id="r",
    )
    assert batch.n_readings_total == 1
    assert batch.n_admissible == 1
    verdict = batch.verdicts[0]
    assert verdict.admissible is True
    assert verdict.faithfulness_admissible is True
    assert verdict.stability_admissible_all_surfaces is True
    assert batch.admissibility_rate == pytest.approx(1.0)


def test_compute_admissibility_faithfulness_fails() -> None:
    faithfulness = _faithfulness(admissible=False, rate=0.5)
    stability = _stability(
        admissible_per_surface={ReadingSurface.HEAD_INTERNAL: True}
    )
    batch = compute_admissibility(
        faithfulness_results=(faithfulness,),
        stability_results=(stability,),
        run_id="r",
    )
    assert batch.n_readings_total == 1
    assert batch.n_admissible == 0
    assert batch.n_inadmissible_faithfulness == 1
    assert batch.n_inadmissible_stability == 0
    assert batch.n_inadmissible_both == 0
    assert batch.verdicts[0].admissible is False


def test_compute_admissibility_stability_fails() -> None:
    faithfulness = _faithfulness(admissible=True, rate=1.0)
    stability = _stability(
        admissible_per_surface={
            ReadingSurface.SUBSTRATE_SIDE: True,
            ReadingSurface.HEAD_INTERNAL: False,
        }
    )
    batch = compute_admissibility(
        faithfulness_results=(faithfulness,),
        stability_results=(stability,),
        run_id="r",
    )
    assert batch.n_admissible == 0
    assert batch.n_inadmissible_stability == 1
    assert batch.n_inadmissible_faithfulness == 0
    assert batch.n_inadmissible_both == 0
    verdict = batch.verdicts[0]
    assert verdict.admissible is False
    assert verdict.stability_admissible_all_surfaces is False


def test_compute_admissibility_both_fail() -> None:
    faithfulness = _faithfulness(admissible=False, rate=0.3)
    stability = _stability(
        admissible_per_surface={ReadingSurface.HEAD_INTERNAL: False}
    )
    batch = compute_admissibility(
        faithfulness_results=(faithfulness,),
        stability_results=(stability,),
        run_id="r",
    )
    assert batch.n_admissible == 0
    assert batch.n_inadmissible_both == 1
    assert batch.n_inadmissible_faithfulness == 0
    assert batch.n_inadmissible_stability == 0
    assert batch.verdicts[0].admissible is False


# ---------------------------------------------------------------------------
# The no-stability-result case.
# ---------------------------------------------------------------------------


def test_compute_admissibility_no_stability_match() -> None:
    # A faithfulness-admissible reading with no matching stability:
    # stability is vacuously admissible -> verdict admissible.
    faithfulness_ok = _faithfulness(admissible=True, rate=1.0)
    batch_ok = compute_admissibility(
        faithfulness_results=(faithfulness_ok,),
        stability_results=(),
        run_id="r",
    )
    verdict_ok = batch_ok.verdicts[0]
    assert verdict_ok.stability_admissible_per_surface == {}
    assert verdict_ok.stability_admissible_all_surfaces is True
    assert verdict_ok.admissible is True
    assert batch_ok.n_admissible == 1
    assert batch_ok.n_inadmissible_no_stability == 0

    # A faithfulness-inadmissible reading with no matching stability:
    # gated by faithfulness alone -> inadmissible, counted in the
    # dedicated no-stability bucket (not n_inadmissible_faithfulness).
    faithfulness_bad = _faithfulness(admissible=False, rate=0.4)
    batch_bad = compute_admissibility(
        faithfulness_results=(faithfulness_bad,),
        stability_results=(),
        run_id="r",
    )
    verdict_bad = batch_bad.verdicts[0]
    assert verdict_bad.admissible is False
    assert batch_bad.n_inadmissible_no_stability == 1
    assert batch_bad.n_inadmissible_faithfulness == 0


def test_compute_admissibility_empty_inputs() -> None:
    batch = compute_admissibility(
        faithfulness_results=(), stability_results=(), run_id="r"
    )
    assert batch.n_readings_total == 0
    assert batch.verdicts == ()
    assert batch.admissibility_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Stability indexing and per-surface threading.
# ---------------------------------------------------------------------------


def test_compute_admissibility_indexes_stability_correctly() -> None:
    # A faithfulness result for (reflexive_attention, primary, ckpt-1).
    faithfulness = _faithfulness(
        criterion_id="reflexive_attention",
        reader_role="primary",
        checkpoint_id="ckpt-1",
    )
    # The matching stability — same envelope, one surface admissible.
    matching = _stability(
        criterion_id="reflexive_attention",
        reader_role="primary",
        checkpoint_id="ckpt-1",
        admissible_per_surface={ReadingSurface.HEAD_INTERNAL: True},
    )
    # A non-matching stability — different checkpoint; must be ignored.
    non_matching = _stability(
        criterion_id="reflexive_attention",
        reader_role="primary",
        checkpoint_id="ckpt-OTHER",
        admissible_per_surface={ReadingSurface.HEAD_INTERNAL: False},
    )
    batch = compute_admissibility(
        faithfulness_results=(faithfulness,),
        stability_results=(non_matching, matching),
        run_id="r",
    )
    verdict = batch.verdicts[0]
    # The matching stability's surfaces — not the non-matching one's —
    # are on the verdict.
    assert verdict.stability_admissible_per_surface == {
        ReadingSurface.HEAD_INTERNAL: True
    }
    assert verdict.admissible is True


def test_compute_admissibility_threads_per_surface_correctly() -> None:
    # A multi-surface criterion: every declared surface contributes to
    # the conjunction. Two surfaces admissible, one not -> the verdict
    # is inadmissible.
    faithfulness = _faithfulness(admissible=True, rate=1.0)
    stability = _stability(
        admissible_per_surface={
            ReadingSurface.SUBSTRATE_SIDE: True,
            ReadingSurface.HEAD_INTERNAL: True,
            ReadingSurface.BEHAVIOR_SIDE: False,
        }
    )
    batch = compute_admissibility(
        faithfulness_results=(faithfulness,),
        stability_results=(stability,),
        run_id="r",
    )
    verdict = batch.verdicts[0]
    assert len(verdict.stability_admissible_per_surface) == 3
    assert verdict.stability_admissible_all_surfaces is False
    assert verdict.admissible is False

    # All three surfaces admissible -> the conjunction holds.
    stability_all = _stability(
        admissible_per_surface={
            ReadingSurface.SUBSTRATE_SIDE: True,
            ReadingSurface.HEAD_INTERNAL: True,
            ReadingSurface.BEHAVIOR_SIDE: True,
        }
    )
    batch_all = compute_admissibility(
        faithfulness_results=(faithfulness,),
        stability_results=(stability_all,),
        run_id="r",
    )
    assert batch_all.verdicts[0].stability_admissible_all_surfaces is True
    assert batch_all.verdicts[0].admissible is True


# ---------------------------------------------------------------------------
# Audit emission and loader.
# ---------------------------------------------------------------------------


def test_compute_admissibility_audit_jsonl_emission(tmp_path: Path) -> None:
    faithfulness_results = (
        _faithfulness(pass_index=0, checkpoint_id="ckpt-1"),
        _faithfulness(pass_index=1, checkpoint_id="ckpt-2", admissible=False,
                      rate=0.5),
    )
    out_path = tmp_path / "admissibility.jsonl"
    batch = compute_admissibility(
        faithfulness_results=faithfulness_results,
        stability_results=(),
        run_id="r",
        audit_jsonl_path=out_path,
    )
    lines = out_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2 == len(batch.verdicts)
    for line in lines:
        parsed = json.loads(line)
        rehydrated = AdmissibilityVerdict.model_validate_json(
            AdmissibilityVerdict.model_validate(parsed).model_dump_json()
        )
        assert isinstance(rehydrated, AdmissibilityVerdict)
        assert rehydrated.criterion_id == "reflexive_attention"


def test_load_admissibility_inputs_from_disk(tmp_path: Path) -> None:
    mirror_dir = tmp_path / "mirror"
    mirror_dir.mkdir(parents=True)
    faithfulness = _faithfulness()
    stability = _stability()
    (mirror_dir / "faithfulness.jsonl").write_text(
        faithfulness.model_dump_json() + "\n", encoding="utf-8"
    )
    (mirror_dir / "stability.jsonl").write_text(
        stability.model_dump_json() + "\n", encoding="utf-8"
    )
    faithfulness_results, stability_results = load_admissibility_inputs(
        "test-run", tmp_path
    )
    assert len(faithfulness_results) == 1
    assert len(stability_results) == 1
    assert isinstance(faithfulness_results[0], FaithfulnessResult)
    assert isinstance(stability_results[0], StabilityResult)

    # A run that ran neither verifier still loads cleanly.
    empty_faith, empty_stab = load_admissibility_inputs(
        "empty-run", tmp_path / "no_such_run"
    )
    assert empty_faith == ()
    assert empty_stab == ()


# ---------------------------------------------------------------------------
# Read-only invariant and notes contract.
# ---------------------------------------------------------------------------


def test_compute_admissibility_does_not_modify_inputs() -> None:
    faithfulness_results = [_faithfulness(admissible=False, rate=0.5)]
    stability_results = [
        _stability(
            admissible_per_surface={ReadingSurface.HEAD_INTERNAL: False}
        )
    ]
    faith_before = [f.model_dump() for f in faithfulness_results]
    stab_before = [s.model_dump() for s in stability_results]
    faith_snapshot = copy.deepcopy(faithfulness_results)
    stab_snapshot = copy.deepcopy(stability_results)

    _ = compute_admissibility(
        faithfulness_results=faithfulness_results,
        stability_results=stability_results,
        run_id="r",
    )

    assert [f.model_dump() for f in faithfulness_results] == faith_before
    assert [s.model_dump() for s in stability_results] == stab_before
    assert faithfulness_results == faith_snapshot
    assert stability_results == stab_snapshot


def test_admissibility_verdict_notes_describe_join() -> None:
    faithfulness = _faithfulness(
        criterion_id="equanimity_perturbation_recovery",
        reader_role="adversarial",
        checkpoint_id="ckpt-XYZ",
    )
    stability = _stability(
        criterion_id="equanimity_perturbation_recovery",
        reader_role="adversarial",
        checkpoint_id="ckpt-XYZ",
        admissible_per_surface={ReadingSurface.BEHAVIOR_SIDE: True},
    )
    batch = compute_admissibility(
        faithfulness_results=(faithfulness,),
        stability_results=(stability,),
        run_id="r",
    )
    notes = batch.verdicts[0].notes
    assert notes.strip() != ""
    # The notes reference both source records by their envelope fields
    # so the journal trail is traceable.
    assert "FaithfulnessResult" in notes
    assert "StabilityResult" in notes
    assert "equanimity_perturbation_recovery" in notes
    assert "ckpt-XYZ" in notes
    assert "behavior_side" in notes


def test_admissibility_verdict_notes_record_missing_stability() -> None:
    faithfulness = _faithfulness()
    batch = compute_admissibility(
        faithfulness_results=(faithfulness,),
        stability_results=(),
        run_id="r",
    )
    notes = batch.verdicts[0].notes
    assert "no StabilityResult matched" in notes
