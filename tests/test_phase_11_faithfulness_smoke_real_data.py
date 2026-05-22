"""Phase 11 Part 4 — smoke test of the faithfulness verifier against
real Phase 13 calibration data.

This is a *data-shapes* smoke, not an LLM smoke. The test runs no LLM
calls and is part of the standard ``pytest`` suite (no
``--run-real-api`` flag). It exercises :func:`verify_reading` against
the on-disk Phase 13 calibration round result so that any mismatch
between the verifier's expected citation shapes and the LLM's actual
citation behavior surfaces here, not in a Phase 12+ smoke.

The test is skipped if the on-disk round result is not present
(allowing the suite to pass on a fresh checkout without the Phase 13
calibration artifacts).

What the test asserts:

- The verifier returns a well-formed
  :class:`~kind.mirror.faithfulness.FaithfulnessResult`
  (frozen, all counts non-negative, counts sum to ``n_claims_total``).
- At least one claim resolves (the LLM's transcription of the prompt's
  signals block is generally faithful — the substrate journal records
  this and the Phase 13 reading's own ``faithfulness_status`` field
  was largely ``"resolved"``).
- The verifier overwrites the LLM's pre-fill: the source reading's
  ``faithfulness_status`` field on each claim is unchanged, while the
  result's per-claim ``status`` is the verifier's verdict.

Any *finding* (unresolved citations) is journaled — the test itself
just asserts the verifier ran cleanly against real data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kind.mirror.faithfulness import (
    FaithfulnessResult,
    FaithfulnessStatus,
    verify_reading,
)
from kind.mirror.llm_caller import MirrorReading
from kind.mirror.statistics import StatisticResult


PHASE_13_ROUND_PATH = (
    Path(__file__).resolve().parents[1]
    / "runs"
    / "phase_13_calibration"
    / "mirror"
    / "rounds"
    / "phase_13_probe_1_round.json"
)


def _load_round() -> dict[str, object]:
    with PHASE_13_ROUND_PATH.open("r", encoding="utf-8") as fh:
        data: dict[str, object] = json.load(fh)
    return data


def _criterion_id_for_signals(signal_names: set[str]) -> str:
    """Heuristic criterion lookup from the set of cited signal names.

    Phase 13's pre-Phase-11 readings don't carry a ``criterion_id`` on
    the reading envelope; the verifier requires one at call time. The
    Phase 7 criteria register their signal mappings — the heuristic
    here checks which v2 criterion's signals overlap.
    """
    if signal_names & {"latent_self_reference_t", "dream_self_reference_t"}:
        return "reflexive_attention"
    if signal_names & {
        "recovery_lag_steps",
        "policy_entropy_t",
        "posterior_kl_t",
    }:
        return "equanimity_perturbation_recovery"
    if signal_names & {"latent_regime_indicator_t", "policy_modulation_t"}:
        return "second_order_volition"
    return "unknown"


@pytest.mark.skipif(
    not PHASE_13_ROUND_PATH.exists(),
    reason=(
        "Phase 13 calibration round result not present on disk; this is a "
        "data-shapes smoke that requires the real artifacts."
    ),
)
def test_verify_reading_runs_against_phase_13_probe_1_round() -> None:
    round_data = _load_round()
    pass_results = round_data["pass_results"]
    assert isinstance(pass_results, list)

    # Find the first pass with readings.
    target_pass: dict[str, object] | None = None
    for p in pass_results:
        assert isinstance(p, dict)
        readings = p.get("active_primary_readings")
        if isinstance(readings, list) and readings:
            target_pass = p
            break
    assert target_pass is not None, (
        "Phase 13 round has no pass with active_primary_readings; "
        "either the calibration didn't produce readings or the schema "
        "changed."
    )

    readings_raw = target_pass["active_primary_readings"]
    assert isinstance(readings_raw, list) and readings_raw
    stat_results_raw = target_pass["statistic_results"]
    assert isinstance(stat_results_raw, list) and stat_results_raw
    checkpoint_id = target_pass["checkpoint_id"]
    assert isinstance(checkpoint_id, str)
    pass_run_id = target_pass["run_id"]
    assert isinstance(pass_run_id, str)

    reading_dict = readings_raw[0]
    assert isinstance(reading_dict, dict)

    # Rehydrate the StructuredReading + StatisticResult records from JSON.
    reading: MirrorReading = MirrorReading.model_validate(reading_dict)
    statistic_results: tuple[StatisticResult, ...] = tuple(
        StatisticResult.model_validate(s) for s in stat_results_raw
    )

    signal_names = {s.signal_name for s in statistic_results}
    cited_fields = {c.cited_scalar_field for c in reading.claims}
    # Restrict cited_fields to bare forms for the heuristic.
    cited_bare = {f.split(".", 1)[0] for f in cited_fields}
    criterion_id = _criterion_id_for_signals(cited_bare & signal_names)
    assert criterion_id != "unknown", (
        f"Could not heuristically determine criterion_id from cited "
        f"signal names: cited_bare={cited_bare}, "
        f"available_signal_names={signal_names}."
    )

    # Snapshot the source reading for the input-immutability check.
    reading_before = reading.model_dump()

    result = verify_reading(
        reading,
        statistic_results,
        criterion_id=criterion_id,
        pass_index=0,
        run_id=pass_run_id,
        checkpoint_id=checkpoint_id,
    )

    # Well-formedness.
    assert isinstance(result, FaithfulnessResult)
    assert result.n_claims_total == len(reading.claims)
    assert len(result.assignments) == result.n_claims_total
    assert (
        result.n_resolved
        + result.n_unresolved_field
        + result.n_unresolved_value
        + result.n_unresolved_range
        == result.n_claims_total
    )
    assert 0.0 <= result.faithfulness_rate <= 1.0
    # The Phase 13 readings transcribe the prompt's signals block;
    # at least one claim should resolve. If none do, that's a finding
    # the verifier needs to surface — but a complete zero resolution
    # would be a regression in the prompt-builder or the verifier.
    assert result.n_resolved >= 1, (
        f"Expected at least one claim to resolve against the Phase 13 "
        f"statistic results; got {result.n_resolved} of "
        f"{result.n_claims_total}. Per-claim notes: "
        f"{[a.resolution_notes for a in result.assignments]}"
    )

    # The verifier owns the verdict — the source reading's own
    # `faithfulness_status` is unchanged by the call.
    assert reading.model_dump() == reading_before

    # Spot-check: every assignment carries a non-empty resolution_notes
    # and a recognized status.
    for a in result.assignments:
        assert a.resolution_notes.strip() != ""
        assert a.status in {
            FaithfulnessStatus.RESOLVED,
            FaithfulnessStatus.UNRESOLVED_FIELD,
            FaithfulnessStatus.UNRESOLVED_VALUE,
            FaithfulnessStatus.UNRESOLVED_RANGE,
        }
