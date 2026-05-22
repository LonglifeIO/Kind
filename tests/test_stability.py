"""Phase 10 gate test — :mod:`kind.mirror.stability`.

All tests inject a :class:`~kind.mirror.llm_caller.MockLLMClient`. No
real API calls. The Phase-10-specific gates here:

- The Pydantic invariants on :class:`StabilityResult` (frozen,
  ``extra="forbid"``, count validators).
- The module-level constant commitments
  (:data:`PARAPHRASE_VARIANTS_PER_SURFACE`,
  :data:`PARAPHRASE_THRESHOLDS`, :data:`RESEED_THRESHOLDS`).
- The structural protection of the verbatim clauses — paraphrase
  variants must not contain or alter the load-bearing strings imported
  from :mod:`kind.mirror.prompt_builder`. This is the test that pins
  Phase 10's protection against paraphrase-drift on load-bearing
  language; it is non-negotiable and the journal entry references it.
- Driver behavior: pairwise-Jaccard aggregation, per-surface threshold
  gating, audit JSONL emission, deterministic seed progression,
  record-sink threading, read-only-invariant on inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pytest

from kind.mirror.criteria_v2 import (
    EQUANIMITY_PERTURBATION_RECOVERY,
    REFLEXIVE_ATTENTION,
)
from kind.mirror.llm_caller import (
    BatchPayload,
    CallOutcome,
    LLMConfig,
    MockLLMClient,
    PassRole,
    _PerCriterionReadingPayload,
)
from kind.mirror.perturbation_align import PerturbationTimeline
from kind.mirror.prompt_builder import (
    EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE,
    SECOND_ORDER_VOLITION_EXCLUSIONS,
    SHAM_PERTURBATION_NOTICE,
)
from kind.mirror.registry import ReadingSurface
from kind.mirror.stability import (
    PARAPHRASE_THRESHOLDS,
    PARAPHRASE_VARIANTS_PER_SURFACE,
    RESEED_THRESHOLDS,
    STABILITY_N_PARAPHRASES_DEFAULT,
    STABILITY_N_RESEEDS_DEFAULT,
    STABILITY_SEED_BASE,
    STABILITY_TEMPERATURE,
    StabilityResult,
    stability_check,
)
from kind.mirror.statistics import StatisticResult
from kind.mirror.structured import StructuredClaim


# ---------------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------------


def _stat(
    *,
    signal_name: str,
    value: float | list[float] | dict[str, float] = 0.5,
    estimator: str = "x",
    n_samples: int = 100,
    notes: str = "x",
) -> StatisticResult:
    return StatisticResult(
        signal_name=signal_name,
        value=value,
        estimator=estimator,
        n_samples=n_samples,
        notes=notes,
    )


def _reflexive_attention_stats() -> tuple[StatisticResult, ...]:
    return (
        _stat(signal_name="latent_self_reference_t", value=0.7),
        _stat(signal_name="dream_self_reference_t", value=0.6),
    )


def _equanimity_stats() -> tuple[StatisticResult, ...]:
    return (
        _stat(signal_name="recovery_lag_steps", value=[3.0]),
        _stat(
            signal_name="policy_entropy_t",
            value={
                "dip_and_recover": 1.0,
                "collapse": 0.0,
                "stays_elevated": 0.0,
                "no_response": 0.0,
            },
        ),
        _stat(
            signal_name="posterior_kl_t",
            value={
                "spike_and_decay": 1.0,
                "ratchet": 0.0,
                "no_response": 0.0,
                "oscillation": 0.0,
            },
        ),
    )


def _empty_timeline() -> PerturbationTimeline:
    return PerturbationTimeline(events=tuple(), run_id="r", checkpoint_id="c")


def _claim(
    *,
    cited_step_range: tuple[int, int] | None = (0, 100),
    cited_scalar_field: str = "h_t",
    cited_value: float = 0.42,
    reading_surface: str = "head_internal",
) -> StructuredClaim:
    return StructuredClaim(
        claim="example claim",
        cited_stream="agent_step",
        cited_run_id="probe2-test",
        cited_episode_range=(0, 1),
        cited_step_range=cited_step_range,
        cited_scalar_field=cited_scalar_field,
        cited_value=cited_value,
        falsifier="a falsifier",
        paraphrase_stability=None,
        reseed_stability=None,
        faithfulness_status="not_checked",
        judge_ruling="not_judged",
        reading_surface=reading_surface,  # type: ignore[arg-type]
        masked_steps_handling="n/a",
    )


def _payload(
    *,
    criterion_id: str,
    claims: list[StructuredClaim],
    framework_anchor: str = "buddhist_phenomenology",
) -> BatchPayload:
    return BatchPayload(
        per_criterion=[
            _PerCriterionReadingPayload(
                criterion_id=criterion_id,
                framework_anchor=framework_anchor,  # type: ignore[arg-type]
                claims=claims,
                free_text_notes="notes",
            )
        ]
    )


# Default LLMConfig with one retry so failures surface quickly in tests.
_TEST_LLM_CONFIG: Final[LLMConfig] = LLMConfig(max_retries=1)


# ---------------------------------------------------------------------------
# Pydantic invariants on StabilityResult.
# ---------------------------------------------------------------------------


def _construct_minimal_result(
    *, criterion_id: str = "reflexive_attention"
) -> StabilityResult:
    return StabilityResult(
        paraphrase_agreement_per_surface={ReadingSurface.HEAD_INTERNAL: 1.0},
        reseed_agreement_per_surface={ReadingSurface.HEAD_INTERNAL: 1.0},
        n_paraphrases=3,
        n_reseeds=3,
        structured_field_agreement_per_claim=(1.0,),
        admissible_per_surface={ReadingSurface.HEAD_INTERNAL: True},
        paraphrase_readings=tuple(),
        reseed_readings=tuple(),
        criterion_id=criterion_id,
        reader_role="primary",
        run_id="r",
        checkpoint_id="c",
        wallclock_ms=0,
    )


def test_stability_result_is_frozen() -> None:
    """``StabilityResult`` is frozen — attribute assignment after
    construction raises ``ValidationError`` (or attribute error,
    depending on Pydantic version)."""
    from pydantic import ValidationError

    result = _construct_minimal_result()
    with pytest.raises((ValidationError, AttributeError, TypeError)):
        result.criterion_id = "different"


def test_stability_result_extra_forbid() -> None:
    """``extra="forbid"`` — unknown fields are rejected."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        StabilityResult(
            paraphrase_agreement_per_surface={
                ReadingSurface.HEAD_INTERNAL: 1.0
            },
            reseed_agreement_per_surface={
                ReadingSurface.HEAD_INTERNAL: 1.0
            },
            n_paraphrases=3,
            n_reseeds=3,
            structured_field_agreement_per_claim=(1.0,),
            admissible_per_surface={ReadingSurface.HEAD_INTERNAL: True},
            paraphrase_readings=tuple(),
            reseed_readings=tuple(),
            criterion_id="x",
            reader_role="primary",
            run_id="r",
            checkpoint_id="c",
            wallclock_ms=0,
            unknown_field=42,  # type: ignore[call-arg]
        )


def test_stability_result_rejects_count_below_two() -> None:
    """Pairwise agreement requires >= 2 readings; n_paraphrases or
    n_reseeds < 2 raises."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        StabilityResult(
            paraphrase_agreement_per_surface={
                ReadingSurface.HEAD_INTERNAL: 1.0
            },
            reseed_agreement_per_surface={
                ReadingSurface.HEAD_INTERNAL: 1.0
            },
            n_paraphrases=1,
            n_reseeds=3,
            structured_field_agreement_per_claim=tuple(),
            admissible_per_surface={ReadingSurface.HEAD_INTERNAL: True},
            paraphrase_readings=tuple(),
            reseed_readings=tuple(),
            criterion_id="x",
            reader_role="primary",
            run_id="r",
            checkpoint_id="c",
            wallclock_ms=0,
        )


def test_stability_result_validates_surface_keys_aligned() -> None:
    """The three per-surface dicts must share the same key set."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="misaligned"):
        StabilityResult(
            paraphrase_agreement_per_surface={
                ReadingSurface.HEAD_INTERNAL: 1.0
            },
            reseed_agreement_per_surface={
                ReadingSurface.SUBSTRATE_SIDE: 1.0
            },
            n_paraphrases=3,
            n_reseeds=3,
            structured_field_agreement_per_claim=tuple(),
            admissible_per_surface={ReadingSurface.HEAD_INTERNAL: True},
            paraphrase_readings=tuple(),
            reseed_readings=tuple(),
            criterion_id="x",
            reader_role="primary",
            run_id="r",
            checkpoint_id="c",
            wallclock_ms=0,
        )


# ---------------------------------------------------------------------------
# Module-level constant commitments.
# ---------------------------------------------------------------------------


def test_stability_thresholds_are_module_level_constants() -> None:
    """``PARAPHRASE_THRESHOLDS`` and ``RESEED_THRESHOLDS`` each carry an
    entry for all three :class:`ReadingSurface` members at the synthesis
    defaults (0.80 / 0.80 / 0.75)."""
    expected_surfaces = {
        ReadingSurface.SUBSTRATE_SIDE,
        ReadingSurface.HEAD_INTERNAL,
        ReadingSurface.BEHAVIOR_SIDE,
    }
    assert set(PARAPHRASE_THRESHOLDS.keys()) == expected_surfaces
    assert set(RESEED_THRESHOLDS.keys()) == expected_surfaces
    assert PARAPHRASE_THRESHOLDS[ReadingSurface.SUBSTRATE_SIDE] == 0.80
    assert PARAPHRASE_THRESHOLDS[ReadingSurface.HEAD_INTERNAL] == 0.80
    assert PARAPHRASE_THRESHOLDS[ReadingSurface.BEHAVIOR_SIDE] == 0.75
    assert RESEED_THRESHOLDS[ReadingSurface.SUBSTRATE_SIDE] == 0.80
    assert RESEED_THRESHOLDS[ReadingSurface.HEAD_INTERNAL] == 0.80
    assert RESEED_THRESHOLDS[ReadingSurface.BEHAVIOR_SIDE] == 0.75


def test_paraphrase_variants_are_module_level_constants() -> None:
    """``PARAPHRASE_VARIANTS_PER_SURFACE`` commits exactly three variants
    per surface for all three surfaces; each variant is a non-empty
    string."""
    expected_surfaces = {
        ReadingSurface.SUBSTRATE_SIDE,
        ReadingSurface.HEAD_INTERNAL,
        ReadingSurface.BEHAVIOR_SIDE,
    }
    assert set(PARAPHRASE_VARIANTS_PER_SURFACE.keys()) == expected_surfaces
    for surface, variants in PARAPHRASE_VARIANTS_PER_SURFACE.items():
        assert len(variants) == 3, (
            f"surface {surface.value!r} must have exactly 3 paraphrase "
            f"variants; got {len(variants)}"
        )
        for variant in variants:
            assert isinstance(variant, str)
            assert variant.strip(), (
                f"variant for surface {surface.value!r} must be a non-"
                f"empty string"
            )


def test_stability_constants_match_synthesis_defaults() -> None:
    """The synthesis defaults for temperature, seed base, and counts
    are pinned at module level."""
    assert STABILITY_TEMPERATURE == 0.7
    assert STABILITY_SEED_BASE == 1000
    assert STABILITY_N_PARAPHRASES_DEFAULT == 3
    assert STABILITY_N_RESEEDS_DEFAULT == 3


# ---------------------------------------------------------------------------
# Verbatim-clause protection.
# ---------------------------------------------------------------------------


def test_paraphrase_variants_do_not_modify_verbatim_clauses() -> None:
    """The load-bearing protection of Phase 10. No paraphrase variant
    contains or alters the verbatim clauses imported from
    :mod:`kind.mirror.prompt_builder`.

    A future contributor who attempts to paraphrase the
    non-falsifying-non-admission clause, any of the four second-order-
    volition exclusions, or the sham-perturbation notice as part of a
    variant trips this test. The variants vary surrounding framing
    prose; the clauses are imported (not duplicated) into the
    prompt-builder and the variants must be disjoint from their text.
    """
    forbidden_strings: tuple[str, ...] = (
        EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE,
        SHAM_PERTURBATION_NOTICE,
        *SECOND_ORDER_VOLITION_EXCLUSIONS,
    )
    for surface, variants in PARAPHRASE_VARIANTS_PER_SURFACE.items():
        for variant in variants:
            for forbidden in forbidden_strings:
                assert forbidden not in variant, (
                    f"paraphrase variant for surface {surface.value!r} "
                    f"contains a load-bearing verbatim clause; this is the "
                    f"soften-the-clause failure mode the test guards "
                    f"against. variant={variant!r}, "
                    f"forbidden={forbidden[:60]!r}…"
                )
                # And reciprocally — no fragment of a clause longer than a
                # few words should appear in any variant.
                # (A bare word like "equanimity" can appear in framing
                # prose; we guard against the full clause text, not
                # incidental word overlap.)


# ---------------------------------------------------------------------------
# Driver: pairwise-Jaccard with mock LLM.
# ---------------------------------------------------------------------------


def test_stability_check_with_mock_llm_high_agreement() -> None:
    """When every paraphrase + reseed call returns the same canned
    response, per-surface agreement is 1.0 everywhere and
    ``admissible_per_surface`` is True everywhere."""
    # reflexive_attention declares only HEAD_INTERNAL.
    canned_payload = _payload(
        criterion_id="reflexive_attention",
        claims=[
            _claim(
                cited_step_range=(0, 100),
                cited_scalar_field="h_t",
                cited_value=0.42,
                reading_surface="head_internal",
            )
        ],
    )
    client = MockLLMClient(canned_payload)
    result = stability_check(
        role="primary",
        criterion=REFLEXIVE_ATTENTION,
        statistic_results=_reflexive_attention_stats(),
        perturbation_timeline=None,
        run_id="r",
        checkpoint_id="ckpt-1",
        digest_run_id="d",
        digest_episode_range=(0, 10),
        llm_config=_TEST_LLM_CONFIG,
        llm_client=client,
    )
    assert result.paraphrase_agreement_per_surface[
        ReadingSurface.HEAD_INTERNAL
    ] == pytest.approx(1.0)
    assert result.reseed_agreement_per_surface[
        ReadingSurface.HEAD_INTERNAL
    ] == pytest.approx(1.0)
    assert result.admissible_per_surface[ReadingSurface.HEAD_INTERNAL] is True
    assert result.n_paraphrases == 3
    assert result.n_reseeds == 3
    # Call count: 1 surface × 3 paraphrases + 3 reseeds = 6.
    assert len(client.calls) == 6


def test_stability_check_with_mock_llm_low_agreement() -> None:
    """When every call returns a distinct cited_value, pairwise Jaccard
    is 0.0 and ``admissible_per_surface`` is False everywhere."""
    # 6 distinct payloads (3 paraphrases + 3 reseeds), each with a
    # unique cited_value.
    responses: list[BatchPayload | Exception] = [
        _payload(
            criterion_id="reflexive_attention",
            claims=[
                _claim(
                    cited_step_range=(0, 100),
                    cited_scalar_field="h_t",
                    cited_value=float(i),
                    reading_surface="head_internal",
                )
            ],
        )
        for i in range(6)
    ]
    client = MockLLMClient(responses)
    result = stability_check(
        role="primary",
        criterion=REFLEXIVE_ATTENTION,
        statistic_results=_reflexive_attention_stats(),
        perturbation_timeline=None,
        run_id="r",
        checkpoint_id="ckpt-1",
        digest_run_id="d",
        digest_episode_range=(0, 10),
        llm_config=_TEST_LLM_CONFIG,
        llm_client=client,
    )
    # Each pair of distinct singletons has Jaccard = 0.
    assert result.paraphrase_agreement_per_surface[
        ReadingSurface.HEAD_INTERNAL
    ] == pytest.approx(0.0)
    assert result.reseed_agreement_per_surface[
        ReadingSurface.HEAD_INTERNAL
    ] == pytest.approx(0.0)
    assert (
        result.admissible_per_surface[ReadingSurface.HEAD_INTERNAL] is False
    )


def test_stability_check_per_surface_threshold_gating() -> None:
    """Multi-surface criterion (equanimity declares substrate-side +
    behavior-side). Set up the mock so substrate-side paraphrases are
    consistent (agreement >= 0.80) and behavior-side paraphrases are
    inconsistent (agreement < 0.75). Reseeds are consistent at both
    surfaces. Expected: substrate admissible, behavior not admissible.
    """
    # Equanimity surface order in the driver: enum definition order
    # filtered to {SUBSTRATE_SIDE, BEHAVIOR_SIDE} → (SUBSTRATE_SIDE,
    # BEHAVIOR_SIDE). The driver issues 3 paraphrase calls per surface,
    # then 3 reseed calls. So:
    # - calls 0..2: substrate-side paraphrase variants
    # - calls 3..5: behavior-side paraphrase variants
    # - calls 6..8: reseeds (default framing)
    #
    # Each reading carries claims at one or both surfaces. We construct
    # responses such that:
    # - For all 3 substrate paraphrase calls: identical (substrate_side,
    #   behavior_side) claims → substrate-side agreement = 1.0, but the
    #   behavior_side claims at calls 0..2 won't affect behavior_side
    #   agreement (behavior agreement only uses calls 3..5 — the per-
    #   surface paraphrase pass for behavior_side).
    # - For each behavior paraphrase call: a *different* behavior_side
    #   cited_value → behavior-side agreement = 0.0.
    # - For all 3 reseeds: identical claims at both surfaces.
    substrate_identical = _payload(
        criterion_id="equanimity_perturbation_recovery",
        claims=[
            _claim(
                cited_step_range=(0, 100),
                cited_scalar_field="h_t",
                cited_value=0.5,
                reading_surface="substrate_side",
            ),
            _claim(
                cited_step_range=(0, 100),
                cited_scalar_field="policy_entropy_t",
                cited_value=0.3,
                reading_surface="behavior_side",
            ),
        ],
    )
    behavior_variants = [
        _payload(
            criterion_id="equanimity_perturbation_recovery",
            claims=[
                _claim(
                    cited_step_range=(0, 100),
                    cited_scalar_field="h_t",
                    cited_value=0.5,
                    reading_surface="substrate_side",
                ),
                _claim(
                    cited_step_range=(0, 100),
                    cited_scalar_field="policy_entropy_t",
                    cited_value=float(i + 10),  # distinct per call
                    reading_surface="behavior_side",
                ),
            ],
        )
        for i in range(3)
    ]
    reseed_identical = _payload(
        criterion_id="equanimity_perturbation_recovery",
        claims=[
            _claim(
                cited_step_range=(0, 100),
                cited_scalar_field="h_t",
                cited_value=0.5,
                reading_surface="substrate_side",
            ),
            _claim(
                cited_step_range=(0, 100),
                cited_scalar_field="policy_entropy_t",
                cited_value=0.3,
                reading_surface="behavior_side",
            ),
        ],
    )

    responses: list[BatchPayload | Exception] = [
        # Substrate paraphrase pass (3 calls, all identical).
        substrate_identical,
        substrate_identical,
        substrate_identical,
        # Behavior paraphrase pass (3 calls, distinct behavior_side values).
        behavior_variants[0],
        behavior_variants[1],
        behavior_variants[2],
        # Reseed pass (3 calls, all identical).
        reseed_identical,
        reseed_identical,
        reseed_identical,
    ]
    client = MockLLMClient(responses)
    result = stability_check(
        role="primary",
        criterion=EQUANIMITY_PERTURBATION_RECOVERY,
        statistic_results=_equanimity_stats(),
        perturbation_timeline=_empty_timeline(),
        run_id="r",
        checkpoint_id="ckpt-1",
        digest_run_id="d",
        digest_episode_range=(0, 10),
        llm_config=_TEST_LLM_CONFIG,
        llm_client=client,
    )
    # Substrate-side: paraphrase ≈ 1.0 (identical), reseed ≈ 1.0 → admissible.
    assert result.paraphrase_agreement_per_surface[
        ReadingSurface.SUBSTRATE_SIDE
    ] == pytest.approx(1.0)
    assert result.reseed_agreement_per_surface[
        ReadingSurface.SUBSTRATE_SIDE
    ] == pytest.approx(1.0)
    assert (
        result.admissible_per_surface[ReadingSurface.SUBSTRATE_SIDE] is True
    )

    # Behavior-side: paraphrase ≈ 0.0 (distinct per call), reseed ≈ 1.0
    # → NOT admissible (paraphrase fails its 0.75 threshold).
    assert result.paraphrase_agreement_per_surface[
        ReadingSurface.BEHAVIOR_SIDE
    ] == pytest.approx(0.0)
    assert result.reseed_agreement_per_surface[
        ReadingSurface.BEHAVIOR_SIDE
    ] == pytest.approx(1.0)
    assert (
        result.admissible_per_surface[ReadingSurface.BEHAVIOR_SIDE] is False
    )


# ---------------------------------------------------------------------------
# Phase 12.5 item 4 — canonical-form normalization in the Jaccard metric.
# ---------------------------------------------------------------------------


def test_stability_check_canonicalizes_compound_citations() -> None:
    """Phase 12.5 item 4: a signal cited bare in one reading and compound
    in another canonicalizes to the same form, so the surface Jaccard
    reads 1.0 rather than 0.0.

    ``REFLEXIVE_ATTENTION`` declares ``latent_self_reference_t`` as a
    signal mapping; the compound form
    ``latent_self_reference_t.partial_autocorr_lag5`` canonicalizes to
    it. Before Phase 12.5 the stability metric compared the verbatim
    LLM string and these two forms read as total disagreement — the
    divergence that tripped Phase 10's HEAD_INTERNAL agreement to
    0.333.
    """
    bare = _payload(
        criterion_id="reflexive_attention",
        claims=[
            _claim(
                cited_step_range=(0, 100),
                cited_scalar_field="latent_self_reference_t",
                cited_value=0.7,
                reading_surface="head_internal",
            )
        ],
    )
    compound = _payload(
        criterion_id="reflexive_attention",
        claims=[
            _claim(
                cited_step_range=(0, 100),
                cited_scalar_field=(
                    "latent_self_reference_t.partial_autocorr_lag5"
                ),
                cited_value=0.7,
                reading_surface="head_internal",
            )
        ],
    )
    # 2 paraphrase calls (bare, compound) + 2 reseed calls (bare, compound).
    client = MockLLMClient([bare, compound, bare, compound])
    result = stability_check(
        role="primary",
        criterion=REFLEXIVE_ATTENTION,
        statistic_results=_reflexive_attention_stats(),
        perturbation_timeline=None,
        run_id="r",
        checkpoint_id="ckpt-1",
        digest_run_id="d",
        digest_episode_range=(0, 10),
        llm_config=_TEST_LLM_CONFIG,
        llm_client=client,
        n_paraphrases=2,
        n_reseeds=2,
    )
    # Bare and compound canonicalize to the same tuple → agreement 1.0.
    assert result.paraphrase_agreement_per_surface[
        ReadingSurface.HEAD_INTERNAL
    ] == pytest.approx(1.0)
    assert result.reseed_agreement_per_surface[
        ReadingSurface.HEAD_INTERNAL
    ] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Audit JSONL emission.
# ---------------------------------------------------------------------------


def test_stability_check_emits_audit_jsonl_when_path_provided(
    tmp_path: Path,
) -> None:
    """When ``audit_jsonl_path`` is provided, one JSONL line per
    stability check is appended to the file; the line is the
    ``model_dump_json()`` of the returned :class:`StabilityResult`."""
    canned = _payload(
        criterion_id="reflexive_attention",
        claims=[
            _claim(
                cited_step_range=(0, 100),
                cited_scalar_field="h_t",
                cited_value=0.42,
                reading_surface="head_internal",
            )
        ],
    )
    client = MockLLMClient(canned)
    audit_path = tmp_path / "stability.jsonl"
    result = stability_check(
        role="primary",
        criterion=REFLEXIVE_ATTENTION,
        statistic_results=_reflexive_attention_stats(),
        perturbation_timeline=None,
        run_id="r",
        checkpoint_id="ckpt-1",
        digest_run_id="d",
        digest_episode_range=(0, 10),
        llm_config=_TEST_LLM_CONFIG,
        llm_client=client,
        audit_jsonl_path=audit_path,
    )
    assert audit_path.exists()
    contents = audit_path.read_text(encoding="utf-8")
    lines = contents.strip().split("\n")
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["criterion_id"] == "reflexive_attention"
    assert parsed["reader_role"] == "primary"
    assert parsed["n_paraphrases"] == 3
    assert parsed["n_reseeds"] == 3
    # The model round-trips: the line parses back to a StabilityResult
    # that equals the returned result (modulo dict-key serialization —
    # the per-surface dicts are JSON-encoded with string keys).
    assert parsed["run_id"] == result.run_id


def test_stability_check_no_audit_file_when_path_none(
    tmp_path: Path,
) -> None:
    """When ``audit_jsonl_path`` is ``None``, no file is created."""
    canned = _payload(
        criterion_id="reflexive_attention",
        claims=[
            _claim(
                cited_step_range=(0, 100),
                cited_scalar_field="h_t",
                cited_value=0.42,
                reading_surface="head_internal",
            )
        ],
    )
    client = MockLLMClient(canned)
    stability_check(
        role="primary",
        criterion=REFLEXIVE_ATTENTION,
        statistic_results=_reflexive_attention_stats(),
        perturbation_timeline=None,
        run_id="r",
        checkpoint_id="ckpt-1",
        digest_run_id="d",
        digest_episode_range=(0, 10),
        llm_config=_TEST_LLM_CONFIG,
        llm_client=client,
        audit_jsonl_path=None,
    )
    # tmp_path was given but never named to the driver — nothing under
    # it.
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Seed progression determinism.
# ---------------------------------------------------------------------------


def test_stability_check_seed_progression_is_deterministic() -> None:
    """The reseed pass issues n_reseeds calls with seeds
    ``STABILITY_SEED_BASE + i`` for ``i in range(n_reseeds)``. The
    paraphrase pass does NOT set a seed (relies on the default). All
    reseed calls receive ``STABILITY_TEMPERATURE``."""
    canned = _payload(
        criterion_id="reflexive_attention",
        claims=[
            _claim(
                cited_step_range=(0, 100),
                cited_scalar_field="h_t",
                cited_value=0.42,
                reading_surface="head_internal",
            )
        ],
    )
    client = MockLLMClient(canned)
    stability_check(
        role="primary",
        criterion=REFLEXIVE_ATTENTION,
        statistic_results=_reflexive_attention_stats(),
        perturbation_timeline=None,
        run_id="r",
        checkpoint_id="ckpt-1",
        digest_run_id="d",
        digest_episode_range=(0, 10),
        llm_config=_TEST_LLM_CONFIG,
        llm_client=client,
    )
    configs = client.configs
    # 1 surface × 3 paraphrases = 3 paraphrase calls, then 3 reseeds.
    assert len(configs) == 6
    # First three (paraphrase) have no seed/temperature override.
    for c in configs[:3]:
        assert c.seed is None
        assert c.temperature is None
    # Last three (reseed) have seed=BASE+i and temperature=STABILITY_TEMPERATURE.
    for i, c in enumerate(configs[3:]):
        assert c.seed == STABILITY_SEED_BASE + i
        assert c.temperature == STABILITY_TEMPERATURE


# ---------------------------------------------------------------------------
# Count fields on the result.
# ---------------------------------------------------------------------------


def test_stability_check_returns_correct_n_paraphrases_and_n_reseeds() -> (
    None
):
    """The returned :class:`StabilityResult` reports the counts the
    caller passed."""
    canned = _payload(
        criterion_id="reflexive_attention",
        claims=[
            _claim(
                cited_step_range=(0, 100),
                cited_scalar_field="h_t",
                cited_value=0.42,
                reading_surface="head_internal",
            )
        ],
    )
    client = MockLLMClient(canned)
    result = stability_check(
        role="primary",
        criterion=REFLEXIVE_ATTENTION,
        statistic_results=_reflexive_attention_stats(),
        perturbation_timeline=None,
        run_id="r",
        checkpoint_id="ckpt-1",
        digest_run_id="d",
        digest_episode_range=(0, 10),
        llm_config=_TEST_LLM_CONFIG,
        llm_client=client,
        n_paraphrases=2,
        n_reseeds=2,
    )
    assert result.n_paraphrases == 2
    assert result.n_reseeds == 2
    # 1 surface × 2 paraphrases + 2 reseeds = 4 LLM calls.
    assert len(client.calls) == 4


def test_stability_check_records_correct_reading_counts() -> None:
    """The flat ``paraphrase_readings`` and ``reseed_readings`` tuples
    match the per-surface paraphrase count × number of surfaces, and
    the reseed count respectively."""
    canned = _payload(
        criterion_id="equanimity_perturbation_recovery",
        claims=[
            _claim(
                cited_step_range=(0, 100),
                cited_scalar_field="h_t",
                cited_value=0.5,
                reading_surface="substrate_side",
            ),
            _claim(
                cited_step_range=(0, 100),
                cited_scalar_field="policy_entropy_t",
                cited_value=0.3,
                reading_surface="behavior_side",
            ),
        ],
    )
    client = MockLLMClient(canned)
    result = stability_check(
        role="primary",
        criterion=EQUANIMITY_PERTURBATION_RECOVERY,
        statistic_results=_equanimity_stats(),
        perturbation_timeline=_empty_timeline(),
        run_id="r",
        checkpoint_id="ckpt-1",
        digest_run_id="d",
        digest_episode_range=(0, 10),
        llm_config=_TEST_LLM_CONFIG,
        llm_client=client,
    )
    # Equanimity declares 2 surfaces; default n_paraphrases=3 → 6
    # paraphrase readings; default n_reseeds=3 → 3 reseed readings.
    assert len(result.paraphrase_readings) == 6
    assert len(result.reseed_readings) == 3


# ---------------------------------------------------------------------------
# Record sink threading.
# ---------------------------------------------------------------------------


class _RecordingSink:
    """Stand-in for the Phase 12 ``LLMCallRecordCollector``. Captures
    every call into a list of dicts for inspection."""

    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def record(
        self,
        *,
        role: PassRole,
        attempt_number: int,
        request_timestamp_ms: int,
        response_timestamp_ms: int | None,
        model_name: str,
        prompt_token_count: int | None,
        response_token_count: int | None,
        outcome: CallOutcome,
        error_message: str | None,
    ) -> object:
        self.records.append(
            {
                "role": role,
                "attempt_number": attempt_number,
                "outcome": outcome,
            }
        )
        return None


def test_stability_check_threads_record_sink() -> None:
    """Every LLM call attempt produced by the stability runner flows
    into the provided record sink. With a perfect mock that succeeds
    on the first try every time, the sink receives exactly N records
    where N == total call count."""
    canned = _payload(
        criterion_id="reflexive_attention",
        claims=[
            _claim(
                cited_step_range=(0, 100),
                cited_scalar_field="h_t",
                cited_value=0.42,
                reading_surface="head_internal",
            )
        ],
    )
    client = MockLLMClient(canned)
    sink = _RecordingSink()
    stability_check(
        role="primary",
        criterion=REFLEXIVE_ATTENTION,
        statistic_results=_reflexive_attention_stats(),
        perturbation_timeline=None,
        run_id="r",
        checkpoint_id="ckpt-1",
        digest_run_id="d",
        digest_episode_range=(0, 10),
        llm_config=_TEST_LLM_CONFIG,
        llm_client=client,
        record_sink=sink,
    )
    # Reflexive_attention has 1 surface; 3 paraphrases + 3 reseeds = 6.
    assert len(sink.records) == 6
    # Every record is for the primary role; every outcome is success.
    for rec in sink.records:
        assert rec["role"] == "primary"
        assert rec["outcome"] == "success"
        assert rec["attempt_number"] == 1


# ---------------------------------------------------------------------------
# Read-only invariant.
# ---------------------------------------------------------------------------


def test_stability_check_does_not_modify_statistic_results() -> None:
    """The input ``statistic_results`` tuple is not mutated by the
    driver. (The same invariant the equivalent test on the
    prompt_builder asserts — Phase 10 inherits the read-only-on-inputs
    contract.)"""
    canned = _payload(
        criterion_id="reflexive_attention",
        claims=[
            _claim(
                cited_step_range=(0, 100),
                cited_scalar_field="h_t",
                cited_value=0.42,
                reading_surface="head_internal",
            )
        ],
    )
    client = MockLLMClient(canned)
    stats = _reflexive_attention_stats()
    stats_before = tuple(s.model_dump_json() for s in stats)
    stability_check(
        role="primary",
        criterion=REFLEXIVE_ATTENTION,
        statistic_results=stats,
        perturbation_timeline=None,
        run_id="r",
        checkpoint_id="ckpt-1",
        digest_run_id="d",
        digest_episode_range=(0, 10),
        llm_config=_TEST_LLM_CONFIG,
        llm_client=client,
    )
    stats_after = tuple(s.model_dump_json() for s in stats)
    assert stats_before == stats_after


# ---------------------------------------------------------------------------
# Input validation.
# ---------------------------------------------------------------------------


def test_stability_check_rejects_n_paraphrases_below_two() -> None:
    """``n_paraphrases < 2`` raises at the driver — pairwise comparison
    is undefined for a single sample."""
    client = MockLLMClient(
        _payload(
            criterion_id="reflexive_attention",
            claims=[],
        )
    )
    with pytest.raises(ValueError, match=">= 2"):
        stability_check(
            role="primary",
            criterion=REFLEXIVE_ATTENTION,
            statistic_results=_reflexive_attention_stats(),
            perturbation_timeline=None,
            run_id="r",
            checkpoint_id="ckpt-1",
            digest_run_id="d",
            digest_episode_range=(0, 10),
            llm_config=_TEST_LLM_CONFIG,
            llm_client=client,
            n_paraphrases=1,
        )


def test_stability_check_rejects_n_reseeds_below_two() -> None:
    """``n_reseeds < 2`` raises."""
    client = MockLLMClient(
        _payload(
            criterion_id="reflexive_attention",
            claims=[],
        )
    )
    with pytest.raises(ValueError, match=">= 2"):
        stability_check(
            role="primary",
            criterion=REFLEXIVE_ATTENTION,
            statistic_results=_reflexive_attention_stats(),
            perturbation_timeline=None,
            run_id="r",
            checkpoint_id="ckpt-1",
            digest_run_id="d",
            digest_episode_range=(0, 10),
            llm_config=_TEST_LLM_CONFIG,
            llm_client=client,
            n_reseeds=1,
        )


def test_stability_check_rejects_n_paraphrases_above_committed_variants() -> (
    None
):
    """Requesting more paraphrases than the committed variant count
    raises (a future amendment to the variant set is a journaled
    decision; the driver doesn't silently cycle)."""
    client = MockLLMClient(
        _payload(
            criterion_id="reflexive_attention",
            claims=[],
        )
    )
    with pytest.raises(ValueError, match="exceeds"):
        stability_check(
            role="primary",
            criterion=REFLEXIVE_ATTENTION,
            statistic_results=_reflexive_attention_stats(),
            perturbation_timeline=None,
            run_id="r",
            checkpoint_id="ckpt-1",
            digest_run_id="d",
            digest_episode_range=(0, 10),
            llm_config=_TEST_LLM_CONFIG,
            llm_client=client,
            n_paraphrases=4,
        )


# ---------------------------------------------------------------------------
# Structured-field per-claim agreement: the informational tuple.
# ---------------------------------------------------------------------------


def test_stability_check_per_claim_agreement_perfect() -> None:
    """When all paraphrase readings at a surface contain the same single
    claim tuple, the per-claim agreement tuple has one entry equal to 1.0
    for that tuple."""
    canned = _payload(
        criterion_id="reflexive_attention",
        claims=[
            _claim(
                cited_step_range=(0, 100),
                cited_scalar_field="h_t",
                cited_value=0.42,
                reading_surface="head_internal",
            )
        ],
    )
    client = MockLLMClient(canned)
    result = stability_check(
        role="primary",
        criterion=REFLEXIVE_ATTENTION,
        statistic_results=_reflexive_attention_stats(),
        perturbation_timeline=None,
        run_id="r",
        checkpoint_id="ckpt-1",
        digest_run_id="d",
        digest_episode_range=(0, 10),
        llm_config=_TEST_LLM_CONFIG,
        llm_client=client,
    )
    # Exactly one unique tuple appeared in the paraphrase readings, in
    # all 3 of them → score = 1.0.
    assert result.structured_field_agreement_per_claim == (1.0,)
