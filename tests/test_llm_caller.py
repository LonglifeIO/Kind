"""Phase 8 gate test — :mod:`kind.mirror.llm_caller`.

All tests inject a :class:`MockLLMClient`. No real API calls.

Covers:

- the role-specific system-prompt selection (primary uses
  :data:`PRIMARY_SYSTEM_PROMPT`; adversarial uses
  :data:`ADVERSARIAL_SYSTEM_PROMPT`);
- the per-criterion-count check (LLM returning a different number of
  readings than fragments raises :class:`MirrorLLMError`);
- the per-criterion id mismatch check (LLM reordering criteria raises);
- the malformed-output retry path: a payload that fails validation on
  the first call but succeeds on the second is recovered within the
  retry budget; failure beyond the budget raises;
- envelope-field stamping: the caller fills run_id / timestamp_ms /
  reader_role / paired_reading_id / baseline_flag / digest_run_id /
  digest_episode_range / schema_version from its keyword arguments.
"""

from __future__ import annotations

import pytest

from kind.mirror.criteria_v2 import (
    EQUANIMITY_PERTURBATION_RECOVERY,
    REFLEXIVE_ATTENTION,
    SECOND_ORDER_VOLITION,
)
from kind.mirror.llm_caller import (
    ADVERSARIAL_SYSTEM_PROMPT,
    PRIMARY_SYSTEM_PROMPT,
    BatchPayload,
    LLMConfig,
    MirrorLLMError,
    MockLLMClient,
    _PerCriterionReadingPayload,
    call_mirror_llm,
)
from kind.mirror.perturbation_align import PerturbationTimeline
from kind.mirror.prompt_builder import PromptFragment, build_fragment
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


def _claim() -> StructuredClaim:
    return StructuredClaim(
        claim="example claim",
        cited_stream="agent_step",
        cited_run_id="probe2-test",
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


def _make_fragments() -> tuple[PromptFragment, ...]:
    """Three fragments — one per active+held-out criterion."""
    timeline = PerturbationTimeline(
        events=tuple(), run_id="r", checkpoint_id="c"
    )
    return (
        build_fragment(
            REFLEXIVE_ATTENTION,
            statistic_results=(
                _stat(signal_name="latent_self_reference_t", value=0.7),
                _stat(signal_name="dream_self_reference_t", value=0.6),
            ),
        ),
        build_fragment(
            EQUANIMITY_PERTURBATION_RECOVERY,
            statistic_results=(
                _stat(signal_name="recovery_lag_steps", value=[3.0]),
                _stat(
                    signal_name="policy_entropy_t",
                    value={"dip_and_recover": 1.0, "collapse": 0.0,
                           "stays_elevated": 0.0, "no_response": 0.0},
                ),
                _stat(
                    signal_name="posterior_kl_t",
                    value={"spike_and_decay": 1.0, "ratchet": 0.0,
                           "no_response": 0.0, "oscillation": 0.0},
                ),
            ),
            perturbation_timeline=timeline,
        ),
        build_fragment(
            SECOND_ORDER_VOLITION,
            statistic_results=(
                _stat(signal_name="policy_modulation_t",
                      value={"contrast_magnitude": 0.5,
                             "observation_only_baseline": 0.1}),
                _stat(signal_name="latent_regime_indicator_t",
                      value=[0.0, 1.0, 2.0, 3.0]),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Role-specific system prompts.
# ---------------------------------------------------------------------------


def test_primary_role_uses_primary_system_prompt() -> None:
    fragments = _make_fragments()
    payload = BatchPayload(
        per_criterion=[
            _per_criterion(criterion_id="reflexive_attention"),
            _per_criterion(criterion_id="equanimity_perturbation_recovery"),
            _per_criterion(
                criterion_id="second_order_volition",
                framework_anchor="buddhist_phenomenology",
            ),
        ]
    )
    client = MockLLMClient(payload)
    call_mirror_llm(
        fragments,
        role="primary",
        config=LLMConfig(),
        run_id="r",
        digest_run_id="d",
        digest_episode_range=(0, 10),
        client=client,
    )
    assert len(client.calls) == 1
    system_prompt, _ = client.calls[0]
    assert system_prompt == PRIMARY_SYSTEM_PROMPT


def test_adversarial_role_uses_adversarial_system_prompt() -> None:
    fragments = _make_fragments()
    payload = BatchPayload(
        per_criterion=[
            _per_criterion(
                criterion_id="reflexive_attention",
                framework_anchor="null_statistics",
            ),
            _per_criterion(
                criterion_id="equanimity_perturbation_recovery",
                framework_anchor="null_statistics",
            ),
            _per_criterion(
                criterion_id="second_order_volition",
                framework_anchor="null_statistics",
            ),
        ]
    )
    client = MockLLMClient(payload)
    call_mirror_llm(
        fragments,
        role="adversarial",
        config=LLMConfig(),
        run_id="r",
        digest_run_id="d",
        digest_episode_range=(0, 10),
        client=client,
    )
    assert len(client.calls) == 1
    system_prompt, _ = client.calls[0]
    assert system_prompt == ADVERSARIAL_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Reader-role mapping.
# ---------------------------------------------------------------------------


def test_primary_role_maps_to_advocate_reader_role() -> None:
    fragments = _make_fragments()
    payload = BatchPayload(
        per_criterion=[
            _per_criterion(criterion_id="reflexive_attention"),
            _per_criterion(criterion_id="equanimity_perturbation_recovery"),
            _per_criterion(criterion_id="second_order_volition"),
        ]
    )
    client = MockLLMClient(payload)
    readings = call_mirror_llm(
        fragments,
        role="primary",
        config=LLMConfig(),
        run_id="r",
        digest_run_id="d",
        digest_episode_range=(0, 10),
        client=client,
    )
    assert all(r.reader_role == "advocate" for r in readings)


def test_adversarial_role_maps_to_skeptic_reader_role() -> None:
    fragments = _make_fragments()
    payload = BatchPayload(
        per_criterion=[
            _per_criterion(
                criterion_id="reflexive_attention",
                framework_anchor="null_statistics",
            ),
            _per_criterion(
                criterion_id="equanimity_perturbation_recovery",
                framework_anchor="null_statistics",
            ),
            _per_criterion(
                criterion_id="second_order_volition",
                framework_anchor="null_statistics",
            ),
        ]
    )
    client = MockLLMClient(payload)
    readings = call_mirror_llm(
        fragments,
        role="adversarial",
        config=LLMConfig(),
        run_id="r",
        digest_run_id="d",
        digest_episode_range=(0, 10),
        client=client,
    )
    assert all(r.reader_role == "skeptic" for r in readings)


# ---------------------------------------------------------------------------
# Per-criterion count and id matching.
# ---------------------------------------------------------------------------


def test_wrong_number_of_readings_raises() -> None:
    fragments = _make_fragments()
    # Three fragments but only two readings in payload.
    payload = BatchPayload(
        per_criterion=[
            _per_criterion(criterion_id="reflexive_attention"),
            _per_criterion(criterion_id="equanimity_perturbation_recovery"),
        ]
    )
    client = MockLLMClient(payload)
    with pytest.raises(MirrorLLMError, match="per-criterion reading"):
        call_mirror_llm(
            fragments,
            role="primary",
            config=LLMConfig(),
            run_id="r",
            digest_run_id="d",
            digest_episode_range=(0, 10),
            client=client,
        )


def test_reordered_criteria_raises() -> None:
    fragments = _make_fragments()
    # LLM swapped reflexive_attention and equanimity.
    payload = BatchPayload(
        per_criterion=[
            _per_criterion(criterion_id="equanimity_perturbation_recovery"),
            _per_criterion(criterion_id="reflexive_attention"),
            _per_criterion(criterion_id="second_order_volition"),
        ]
    )
    client = MockLLMClient(payload)
    with pytest.raises(MirrorLLMError, match="reordered or relabeled"):
        call_mirror_llm(
            fragments,
            role="primary",
            config=LLMConfig(),
            run_id="r",
            digest_run_id="d",
            digest_episode_range=(0, 10),
            client=client,
        )


# ---------------------------------------------------------------------------
# Retry path.
# ---------------------------------------------------------------------------


def test_retry_recovers_within_budget() -> None:
    fragments = _make_fragments()
    good_payload = BatchPayload(
        per_criterion=[
            _per_criterion(criterion_id="reflexive_attention"),
            _per_criterion(criterion_id="equanimity_perturbation_recovery"),
            _per_criterion(criterion_id="second_order_volition"),
        ]
    )
    # First call raises; second call returns the good payload.
    client = MockLLMClient(
        [RuntimeError("malformed JSON"), good_payload]
    )
    readings = call_mirror_llm(
        fragments,
        role="primary",
        config=LLMConfig(max_retries=3),
        run_id="r",
        digest_run_id="d",
        digest_episode_range=(0, 10),
        client=client,
    )
    assert len(readings) == 3
    assert len(client.calls) == 2  # one failure + one success


def test_retry_exhaustion_raises_mirror_llm_error() -> None:
    fragments = _make_fragments()
    # max_retries=1 means at most 2 attempts; queue three failures.
    client = MockLLMClient(
        [
            RuntimeError("malformed JSON 1"),
            RuntimeError("malformed JSON 2"),
            RuntimeError("malformed JSON 3"),
        ]
    )
    with pytest.raises(MirrorLLMError, match="max_retries"):
        call_mirror_llm(
            fragments,
            role="primary",
            config=LLMConfig(max_retries=1),
            run_id="r",
            digest_run_id="d",
            digest_episode_range=(0, 10),
            client=client,
        )


# ---------------------------------------------------------------------------
# Envelope stamping.
# ---------------------------------------------------------------------------


def test_envelope_fields_are_stamped_from_kwargs() -> None:
    fragments = _make_fragments()
    payload = BatchPayload(
        per_criterion=[
            _per_criterion(criterion_id="reflexive_attention"),
            _per_criterion(criterion_id="equanimity_perturbation_recovery"),
            _per_criterion(criterion_id="second_order_volition"),
        ]
    )
    client = MockLLMClient(payload)
    readings = call_mirror_llm(
        fragments,
        role="primary",
        config=LLMConfig(),
        run_id="probe2-run-abc",
        digest_run_id="probe2-run-abc",
        digest_episode_range=(5, 15),
        paired_reading_id="pair-001",
        baseline_flag="genuine",
        client=client,
    )
    for r in readings:
        assert r.run_id == "probe2-run-abc"
        assert r.digest_run_id == "probe2-run-abc"
        assert r.digest_episode_range == (5, 15)
        assert r.paired_reading_id == "pair-001"
        assert r.baseline_flag == "genuine"
        assert r.schema_version == "0.2.0"


def test_empty_fragments_rejected() -> None:
    with pytest.raises(ValueError, match="empty tuple"):
        call_mirror_llm(
            tuple(),
            role="primary",
            config=LLMConfig(),
            run_id="r",
            digest_run_id="d",
            digest_episode_range=(0, 10),
            client=MockLLMClient(BatchPayload(per_criterion=[])),
        )


def test_llm_config_rejects_zero_retries() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LLMConfig(max_retries=0)


# ---------------------------------------------------------------------------
# Phase 12 finding regression: Gemini schema munger.
# ---------------------------------------------------------------------------


def test_gemini_schema_munger_converts_prefix_items_to_items() -> None:
    """The Phase 12 smoke first invocation found that Gemini's
    structured-output Schema validator (OpenAPI 3.0) rejects
    ``prefixItems`` (JSON Schema 2020-12). The
    :func:`_to_gemini_schema` helper converts ``prefixItems`` →
    ``items`` so the BatchPayload schema (which embeds
    StructuredClaim's ``cited_episode_range`` / ``cited_step_range``
    int pairs) can be sent to Gemini. This test pins the conversion
    against accidental regression."""
    from kind.mirror.llm_caller import BatchPayload, _to_gemini_schema

    src = BatchPayload.model_json_schema()
    munged = _to_gemini_schema(src)

    def _walk(obj: object) -> bool:
        """True if ``prefixItems`` appears anywhere in the tree."""
        if isinstance(obj, dict):
            if "prefixItems" in obj:
                return True
            return any(_walk(v) for v in obj.values())
        if isinstance(obj, list):
            return any(_walk(item) for item in obj)
        return False

    # Source must contain prefixItems for the test to be meaningful;
    # if a future Pydantic version stops emitting them this test
    # becomes a tautology — flag it.
    assert _walk(src), (
        "test premise broke: BatchPayload.model_json_schema() no longer "
        "contains prefixItems; the schema munger may now be unnecessary"
    )
    # Munged schema has none.
    assert not _walk(munged)


def test_gemini_schema_munger_preserves_min_max_items() -> None:
    """The conversion preserves length pinning — ``minItems`` and
    ``maxItems`` stay on the same node so the int-pair semantics carry
    through."""
    from kind.mirror.llm_caller import _to_gemini_schema

    src = {
        "type": "array",
        "minItems": 2,
        "maxItems": 2,
        "prefixItems": [{"type": "integer"}, {"type": "integer"}],
    }
    munged = _to_gemini_schema(src)
    assert munged["minItems"] == 2
    assert munged["maxItems"] == 2
    assert munged["items"] == {"type": "integer"}
    assert "prefixItems" not in munged


def test_gemini_schema_munger_inlines_refs_and_drops_defs() -> None:
    """The munger inlines ``$ref`` references using ``$defs`` and then
    drops ``$defs``. Gemini's Schema doesn't support refs at all."""
    from kind.mirror.llm_caller import _to_gemini_schema

    src = {
        "type": "object",
        "properties": {"pair": {"$ref": "#/$defs/Pair"}},
        "$defs": {
            "Pair": {
                "type": "array",
                "prefixItems": [{"type": "string"}, {"type": "string"}],
                "minItems": 2,
                "maxItems": 2,
            }
        },
    }
    munged = _to_gemini_schema(src)
    # $defs gone after inlining.
    assert "$defs" not in munged
    # The ref was replaced with the inlined (and prefixItems-converted)
    # subtree.
    pair = munged["properties"]["pair"]
    assert pair["type"] == "array"
    assert pair["items"] == {"type": "string"}
    assert "prefixItems" not in pair


def test_gemini_schema_munger_strips_additional_properties_false() -> None:
    """``additionalProperties: false`` (Pydantic's extra='forbid'
    marker) is dropped — Gemini's Schema doesn't recognize the key."""
    from kind.mirror.llm_caller import _to_gemini_schema

    src = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"x": {"type": "integer"}},
    }
    munged = _to_gemini_schema(src)
    assert "additionalProperties" not in munged
    assert munged["properties"]["x"] == {"type": "integer"}


def test_gemini_schema_munger_converts_anyof_with_null_to_nullable() -> None:
    """``anyOf: [X, {"type":"null"}]`` (Pydantic's Optional[X] form) →
    X with ``nullable: true``."""
    from kind.mirror.llm_caller import _to_gemini_schema

    src = {
        "type": "object",
        "properties": {
            "maybe_int": {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "null"},
                ]
            }
        },
    }
    munged = _to_gemini_schema(src)
    maybe = munged["properties"]["maybe_int"]
    assert maybe.get("type") == "integer"
    assert maybe.get("nullable") is True
    assert "anyOf" not in maybe


def test_gemini_schema_munger_strips_title() -> None:
    from kind.mirror.llm_caller import _to_gemini_schema

    src = {"title": "Foo", "type": "object", "properties": {}}
    munged = _to_gemini_schema(src)
    assert "title" not in munged


def test_gemini_schema_munger_passes_through_non_dict_non_list() -> None:
    """Strings / ints / None pass through unchanged."""
    from kind.mirror.llm_caller import _to_gemini_schema

    assert _to_gemini_schema("foo") == "foo"
    assert _to_gemini_schema(42) == 42
    assert _to_gemini_schema(None) is None


# ---------------------------------------------------------------------------
# Phase 10 extensions — seed, temperature, MockLLMClient.configs.
# ---------------------------------------------------------------------------


def test_llm_config_accepts_optional_seed_and_temperature() -> None:
    """``LLMConfig`` exposes ``seed`` and ``temperature`` as optional
    fields; both default to ``None`` (the SDK's defaults apply)."""
    default_config = LLMConfig()
    assert default_config.seed is None
    assert default_config.temperature is None

    set_config = LLMConfig(seed=42, temperature=0.7)
    assert set_config.seed == 42
    assert set_config.temperature == 0.7


def test_llm_config_validates_temperature_range() -> None:
    """Temperature must be in [0.0, 2.0]."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LLMConfig(temperature=-0.1)
    with pytest.raises(ValidationError):
        LLMConfig(temperature=2.5)
    # Boundaries are accepted.
    LLMConfig(temperature=0.0)
    LLMConfig(temperature=2.0)


def test_llm_config_validates_seed_non_negative() -> None:
    """Seed must be non-negative when set."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LLMConfig(seed=-1)
    LLMConfig(seed=0)
    LLMConfig(seed=999_999)


def test_call_mirror_llm_seed_and_temperature_override_config() -> None:
    """When ``call_mirror_llm`` is invoked with ``seed`` or
    ``temperature`` kwargs, the config the client receives has those
    fields overridden. The original ``config`` argument is unchanged."""
    fragments = _make_fragments()
    payload = BatchPayload(
        per_criterion=[
            _per_criterion(criterion_id="reflexive_attention"),
            _per_criterion(criterion_id="equanimity_perturbation_recovery"),
            _per_criterion(criterion_id="second_order_volition"),
        ]
    )
    client = MockLLMClient(payload)
    config = LLMConfig()
    call_mirror_llm(
        fragments,
        role="primary",
        config=config,
        run_id="r",
        digest_run_id="d",
        digest_episode_range=(0, 10),
        client=client,
        seed=42,
        temperature=0.7,
    )
    # MockLLMClient records the config each call received.
    received = client.configs[0]
    assert received.seed == 42
    assert received.temperature == 0.7
    # The original config is unchanged (frozen + model_copy returns a
    # new instance).
    assert config.seed is None
    assert config.temperature is None


def test_call_mirror_llm_kwargs_default_to_no_override() -> None:
    """Without ``seed`` or ``temperature`` kwargs, the client receives
    the config unchanged (with its own ``None`` defaults)."""
    fragments = _make_fragments()
    payload = BatchPayload(
        per_criterion=[
            _per_criterion(criterion_id="reflexive_attention"),
            _per_criterion(criterion_id="equanimity_perturbation_recovery"),
            _per_criterion(criterion_id="second_order_volition"),
        ]
    )
    client = MockLLMClient(payload)
    call_mirror_llm(
        fragments,
        role="primary",
        config=LLMConfig(),
        run_id="r",
        digest_run_id="d",
        digest_episode_range=(0, 10),
        client=client,
    )
    received = client.configs[0]
    assert received.seed is None
    assert received.temperature is None


def test_call_mirror_llm_preserves_pre_set_config_fields_when_kwargs_omitted() -> (
    None
):
    """If the config already carries ``seed`` / ``temperature``, omitting
    the corresponding kwargs leaves them in place."""
    fragments = _make_fragments()
    payload = BatchPayload(
        per_criterion=[
            _per_criterion(criterion_id="reflexive_attention"),
            _per_criterion(criterion_id="equanimity_perturbation_recovery"),
            _per_criterion(criterion_id="second_order_volition"),
        ]
    )
    client = MockLLMClient(payload)
    call_mirror_llm(
        fragments,
        role="primary",
        config=LLMConfig(seed=99, temperature=0.3),
        run_id="r",
        digest_run_id="d",
        digest_episode_range=(0, 10),
        client=client,
        # no seed/temperature kwargs
    )
    received = client.configs[0]
    assert received.seed == 99
    assert received.temperature == 0.3


def test_mock_llm_client_records_configs_in_call_order() -> None:
    """``MockLLMClient.configs`` is the same length as ``.calls``, in
    call order. Phase 10's stability tests rely on this surface to
    verify seed progression."""
    fragments = _make_fragments()
    payload = BatchPayload(
        per_criterion=[
            _per_criterion(criterion_id="reflexive_attention"),
            _per_criterion(criterion_id="equanimity_perturbation_recovery"),
            _per_criterion(criterion_id="second_order_volition"),
        ]
    )
    client = MockLLMClient(payload)
    call_mirror_llm(
        fragments,
        role="primary",
        config=LLMConfig(),
        run_id="r",
        digest_run_id="d",
        digest_episode_range=(0, 10),
        client=client,
    )
    assert len(client.calls) == len(client.configs) == 1


# ---------------------------------------------------------------------------
# Phase 12.5 item 1 — faithfulness_status is dropped from the LLM schema.
# ---------------------------------------------------------------------------


def test_gemini_schema_munger_drops_faithfulness_status() -> None:
    """Phase 12.5 item 1: ``faithfulness_status`` is the faithfulness
    verifier's field, not the reading LLM's. The schema munger removes
    the property from the structured-output schema so Gemini is never
    asked to fill it. Pinned against accidental regression."""
    from kind.mirror.llm_caller import BatchPayload, _to_gemini_schema

    def _has_property(obj: object, prop: str) -> bool:
        """True if ``prop`` appears in any ``properties`` dict in the tree."""
        if isinstance(obj, dict):
            props = obj.get("properties")
            if isinstance(props, dict) and prop in props:
                return True
            return any(_has_property(v, prop) for v in obj.values())
        if isinstance(obj, list):
            return any(_has_property(item, prop) for item in obj)
        return False

    src = BatchPayload.model_json_schema()
    munged = _to_gemini_schema(src)

    # Premise: the source schema does carry the property (a field with a
    # default still appears under `properties`). If a future change
    # removes the field entirely this test becomes a tautology — flag it.
    assert _has_property(src, "faithfulness_status"), (
        "test premise broke: faithfulness_status is no longer a property "
        "of the StructuredClaim schema"
    )
    # The munged schema does not.
    assert not _has_property(munged, "faithfulness_status")


def test_structured_claim_parses_without_faithfulness_status_to_default() -> (
    None
):
    """A claim payload omitting ``faithfulness_status`` — as Gemini now
    will, since the munger dropped it from the schema — parses into a
    :class:`StructuredClaim` with the writer-side default
    ``"not_checked"``."""
    claim_dict = {
        "claim": "example",
        "cited_stream": "agent_step",
        "cited_run_id": "r",
        "cited_episode_range": None,
        "cited_step_range": [0, 10],
        "cited_scalar_field": "h_t",
        "cited_value": 0.5,
        "falsifier": "f",
        "paraphrase_stability": None,
        "reseed_stability": None,
        "judge_ruling": "not_judged",
        "reading_surface": "head_internal",
        "masked_steps_handling": "n/a",
    }
    claim = StructuredClaim.model_validate(claim_dict)
    assert claim.faithfulness_status == "not_checked"
