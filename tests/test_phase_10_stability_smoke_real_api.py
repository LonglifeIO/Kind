"""Phase 10 end-to-end stability smoke against the REAL Gemini API.

This test is opt-in: pytest skips it by default, running only when
``pytest --run-real-api`` is passed or when :envvar:`GEMINI_API_KEY`
is set in the environment.

Running this test consumes Gemini API quota. With
``n_paraphrases=3`` and ``n_reseeds=3`` against the single-surface
``reflexive_attention`` criterion (declared at ``head_internal``), the
smoke makes ``3 + 3 = 6`` LLM calls (small surface, ~$0.30, ~3
minutes wallclock typical). The test asserts:

- the smoke completes (no uncaught exceptions);
- the result is well-formed (Pydantic round-trip succeeds);
- the per-surface agreement dicts contain exactly the criterion's
  declared surfaces;
- the audit JSONL is written with one line for the result;
- the underlying readings tuple lengths match the call structure
  (3 paraphrase readings × 1 surface; 3 reseed readings).

Substantive interpretation (whether the synthesis's per-surface
thresholds 0.80 / 0.80 / 0.75 are well-tuned in this LLM × prompt
combination) is journal-side. This test is the engineering gate.

Statistic input. The smoke uses fixture statistic results modeled
after Phase 1.5's settled telemetry shape — plausible enough that
Gemini produces structured claims, but not tied to a specific
checkpoint. A future amendment can read real Phase 13 statistic
results from disk if the env var :envvar:`KIND_PHASE_13_CALIBRATION_PATH`
is wired through; Phase 10's smoke does not require that integration
to be useful.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kind.mirror.criteria_v2 import REFLEXIVE_ATTENTION
from kind.mirror.llm_caller import LLMConfig
from kind.mirror.registry import ReadingSurface
from kind.mirror.stability import (
    STABILITY_N_PARAPHRASES_DEFAULT,
    STABILITY_N_RESEEDS_DEFAULT,
    StabilityResult,
    stability_check,
)
from kind.mirror.statistics import StatisticResult


def _fixture_reflexive_attention_stats() -> tuple[StatisticResult, ...]:
    """Plausible statistic results for reflexive_attention. The values
    are shaped after Probe 1.5's settled telemetry — modest positive
    within-h_t coupling exceeding the matched shuffled-time control —
    so Gemini has structured content to reference."""
    return (
        StatisticResult(
            signal_name="latent_self_reference_t",
            value=0.42,
            estimator="partial_autocorr_lag5",
            n_samples=5000,
            notes=(
                "Phase 10 smoke fixture: representative within-h_t partial "
                "autocorrelation at lag 5 against the matched shuffled-time "
                "control (control ≈ 0.08)."
            ),
        ),
        StatisticResult(
            signal_name="dream_self_reference_t",
            value=0.51,
            estimator="autocorr_lag5",
            n_samples=512,
            notes=(
                "Phase 10 smoke fixture: within-sequence_h autocorrelation "
                "at lag 5 on the dream rollout (control ≈ 0.10). The dream "
                "version is at least as pronounced as the waking version, "
                "consistent with the reflexive-attention criterion's "
                "expectation."
            ),
        ),
    )


@pytest.mark.real_api
def test_phase_10_stability_smoke_runs_end_to_end_against_real_gemini(
    tmp_path: Path,
) -> None:
    """Run :func:`~kind.mirror.stability.stability_check` for one
    criterion + one role against the real Gemini API, with the
    Phase-10-default ``n_paraphrases=3`` and ``n_reseeds=3``.

    The test consumes API quota; conftest.py's collection-modifier
    skips this by default unless ``--run-real-api`` or
    :envvar:`GEMINI_API_KEY` is set.
    """
    audit_path = tmp_path / "stability.jsonl"

    result = stability_check(
        role="primary",
        criterion=REFLEXIVE_ATTENTION,
        statistic_results=_fixture_reflexive_attention_stats(),
        perturbation_timeline=None,
        run_id="phase_10_smoke",
        checkpoint_id="phase_10_smoke_ckpt",
        digest_run_id="phase_10_smoke_digest",
        digest_episode_range=(0, 10),
        llm_config=LLMConfig(),
        audit_jsonl_path=audit_path,
    )

    # Well-formed: Pydantic round-trip succeeds.
    redumped = StabilityResult.model_validate_json(result.model_dump_json())
    assert redumped == result

    # The criterion declares only HEAD_INTERNAL; per-surface dicts
    # contain exactly that key.
    expected_surfaces = {ReadingSurface.HEAD_INTERNAL}
    assert set(result.paraphrase_agreement_per_surface.keys()) == expected_surfaces
    assert set(result.reseed_agreement_per_surface.keys()) == expected_surfaces
    assert set(result.admissible_per_surface.keys()) == expected_surfaces

    # Reading-tuple shapes.
    # 1 surface × 3 paraphrases = 3 paraphrase readings.
    assert len(result.paraphrase_readings) == 3 == STABILITY_N_PARAPHRASES_DEFAULT
    # 3 reseed readings.
    assert len(result.reseed_readings) == 3 == STABILITY_N_RESEEDS_DEFAULT

    # Agreement scores are in [0, 1].
    for s, score in result.paraphrase_agreement_per_surface.items():
        assert 0.0 <= score <= 1.0, (
            f"paraphrase_agreement_per_surface[{s.value!r}] = {score} is "
            f"outside [0, 1]"
        )
    for s, score in result.reseed_agreement_per_surface.items():
        assert 0.0 <= score <= 1.0, (
            f"reseed_agreement_per_surface[{s.value!r}] = {score} is "
            f"outside [0, 1]"
        )

    # Audit JSONL was written.
    assert audit_path.exists()
    lines = audit_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["criterion_id"] == "reflexive_attention"
    assert parsed["reader_role"] == "primary"
    assert parsed["n_paraphrases"] == STABILITY_N_PARAPHRASES_DEFAULT
    assert parsed["n_reseeds"] == STABILITY_N_RESEEDS_DEFAULT
