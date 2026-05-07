"""Phase 8 gate-summary meta-test.

Plan §4 names exactly five tests as the Probe 1 gate, and the Probe 1.5
implementation plan §4 names five additional gate tests (1: forward
shape; 2: EMA target update mechanics; 3: self-prediction loss
decreases; 4: opacity boundary preserved with the v2 PolicyView field
set; 5: integration smoke). This file imports each gate-test module
and asserts the named test function exists. It is not a substitute for
the gate tests themselves — it is a forward-compatibility check: if a
future refactor renames or removes a gate test, this file fails loudly
rather than letting the gate silently shrink.

The gate-test name table mirrors plan §4 for both Probe 1 and Probe 1.5.
Test #2 (agent forward) is split across two files in the existing test
suite — the world-model half lives in ``test_agent_forward.py``, the
actor and ensemble half in ``test_actor.py``. This file names a
representative test per file so both halves are covered. Probe 1.5
gates 1-3 land in ``test_world_model.py`` (Phase 1); gate 4 lands in
``test_views.py`` and ``test_actor.py`` (Phase 2); gate 5 is held until
Phase 6 builds the Probe 1.5 integration smoke.
"""

from __future__ import annotations

import importlib

from pytest import mark


_GATE_TESTS: tuple[tuple[str, str, str], ...] = (
    (
        "1: env step shape",
        "tests.test_env_step",
        "test_gate_env_step_shape_and_basic_mechanics",
    ),
    (
        "2a: agent forward (world model)",
        "tests.test_agent_forward",
        "test_gate_world_model_step_and_loss_and_backward",
    ),
    (
        "2b: agent forward (actor + ensemble)",
        "tests.test_actor",
        "test_actor_forward_consumes_split_output_directly",
    ),
    (
        "3: perturbation hook logged",
        "tests.test_perturbation_hook",
        "test_gate_perturbation_hook_logged",
    ),
    (
        "4: JSONL/Parquet roundtrip",
        "tests.test_sinks",
        "test_gate_100_record_roundtrip_through_both_sinks",
    ),
    (
        "5a: integration smoke (200 steps)",
        "tests.test_integration_smoke",
        "test_smoke_runs_to_completion_on_cpu",
    ),
    (
        "5b: integration smoke (mid-run checkpoint)",
        "tests.test_integration_smoke",
        "test_smoke_checkpoint_committed_mid_run",
    ),
    (
        "5c: integration smoke (four streams)",
        "tests.test_integration_smoke",
        "test_smoke_all_four_streams_have_records",
    ),
    (
        "5d: integration smoke (resume — parameter equality)",
        "tests.test_integration_smoke",
        "test_smoke_resume_loads_identical_weights",
    ),
    (
        "5e: integration smoke (resume — RNG state equality)",
        "tests.test_integration_smoke",
        "test_smoke_resume_loads_identical_rng_state",
    ),
    # ---- Probe 1.5 gates ----
    (
        "1.5/1: self-prediction forward shape",
        "tests.test_world_model",
        "test_gate_self_prediction_forward_shape",
    ),
    (
        "1.5/2: EMA target update mechanics",
        "tests.test_world_model",
        "test_gate_ema_target_update_mechanics",
    ),
    (
        "1.5/3: self-prediction loss decreases",
        "tests.test_world_model",
        "test_gate_self_prediction_loss_decreases",
    ),
    (
        "1.5/4a: PolicyView field set is exactly h, z, self_prediction_error",
        "tests.test_views",
        "test_policy_view_field_set_is_exactly_h_z_self_prediction_error",
    ),
    (
        "1.5/4b: actor.forward rejects TelemetryView at runtime",
        "tests.test_views",
        "test_actor_forward_rejects_telemetry_view_at_runtime",
    ),
    (
        "1.5/4c: actor.forward(TelemetryView) fails mypy --strict",
        "tests.test_views",
        "test_actor_forward_telemetryview_argument_fails_mypy_strict",
    ),
    (
        "1.5/4d: AST-lint preserved with extended TelemetryView",
        "tests.test_views",
        "test_ast_lint_passes_with_extended_telemetry_view",
    ),
    (
        "1.5/5a: integration smoke (200 steps, all four streams)",
        "tests.test_integration_smoke_probe1_5",
        "test_smoke_probe1_5_all_four_streams_have_records",
    ),
    (
        "1.5/5b: integration smoke (AgentStep new fields populated)",
        "tests.test_integration_smoke_probe1_5",
        "test_smoke_probe1_5_agent_step_carries_new_fields",
    ),
    (
        "1.5/5c: integration smoke (first-step masking)",
        "tests.test_integration_smoke_probe1_5",
        "test_smoke_probe1_5_first_step_of_episode_is_masked",
    ),
    (
        "1.5/5d: integration smoke (DreamRollout sequence_self_prediction None)",
        "tests.test_integration_smoke_probe1_5",
        "test_smoke_probe1_5_dream_rollout_carries_none_self_prediction",
    ),
    (
        "1.5/5e: integration smoke (checkpoint EMA + extended actor layer)",
        "tests.test_integration_smoke_probe1_5",
        "test_smoke_probe1_5_checkpoint_carries_ema_and_extended_actor",
    ),
    (
        "1.5/5f: integration smoke (resume — EMA + actor byte equality)",
        "tests.test_integration_smoke_probe1_5",
        "test_smoke_probe1_5_resume_yields_identical_ema_and_actor",
    ),
    (
        "1.5/5g: Probe 1 → Probe 1.5 checkpoint compat (EMA from online; "
        "actor zero-init; mirror_marker)",
        "tests.test_integration_smoke_probe1_5",
        "test_load_probe_1_checkpoint_initializes_ema_from_online",
    ),
    (
        "1.5/5h: prior-network gradient confirmation under sp_loss alone",
        "tests.test_integration_smoke_probe1_5",
        "test_prior_network_gradient_under_self_prediction_loss_alone",
    ),
)


@mark.parametrize(
    ("gate_label", "module_name", "test_name"),
    _GATE_TESTS,
    ids=[label for label, _module, _test in _GATE_TESTS],
)
def test_gate_test_exists(
    gate_label: str, module_name: str, test_name: str
) -> None:
    """Each named gate test from plan §4 must exist by name in its module."""
    module = importlib.import_module(module_name)
    assert hasattr(module, test_name), (
        f"plan §4 gate '{gate_label}' missing: {module_name}::{test_name}"
    )
    func = getattr(module, test_name)
    assert callable(func), (
        f"plan §4 gate '{gate_label}': {module_name}::{test_name} is not callable"
    )
