"""Probe 3.5 Phase 2 — telemetry: the pragmatic-share record version.

Plan §S-TEL realized for Phase 2: a fresh record-level version
(``PROBE_3_5_PHASE2_TELEMETRY_SCHEMA_VERSION`` = "0.5.0") whose writer-side
validator requires the per-training-step pragmatic decomposition fields
(``pragmatic_value_t`` / ``epistemic_value_t`` / ``pragmatic_share_t`` — the
A2b share band and the §8.4 "pragmatic share → 1" falsification signature
read these); older shards stay backward-readable; the Phase-1 v0.5.0 export
is frozen to bytes; the new live export is v0.6.0; emission is opt-in
(``RunnerConfig.energy_preference`` default None → existing runners
byte-identical). The D monitor (per-dim KL) is retained per Amendment 01 —
``kl_per_dim_t`` has been on every AgentStep since Probe 1 and is asserted
present here so its demotion-to-monitor stays observable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kind.observer.schemas import (
    PROBE_3_5_PHASE2_EXPORT_VERSION,
    PROBE_3_5_PHASE2_TELEMETRY_SCHEMA_VERSION,
    PROBE_3_5_TELEMETRY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    AgentStep,
    export_json_schema_v0_5_0,
    export_json_schema_v0_6_0,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_FILE_V0_5_0 = REPO_ROOT / "schemas" / "v0.5.0.json"
SCHEMA_FILE_V0_6_0 = REPO_ROOT / "schemas" / "v0.6.0.json"


def _agent_step_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = dict(
        run_id="r",
        checkpoint_id=None,
        t=0,
        episode_id=0,
        step_in_episode=0,
        wallclock_ms=0,
        h_t=[0.0],
        q_params_t=([0.0], [0.0]),
        p_params_t=([0.0], [0.0]),
        z_t=[0.0],
        kl_per_dim_t=[0.0],
        kl_aggregate_t=0.0,
        recon_loss_t=0.0,
        action_t=0,
        action_logprob_t=0.0,
        policy_entropy_t=0.0,
        obs_hash_t="x",
        intrinsic_signal_t=0.0,
        encoder_embedding_t=[0.0],
        self_prediction_t=[0.0],
        self_prediction_error_t=0.0,
        self_prediction_error_masked_t=False,
        sensed_energy_t=0.7,
        true_energy_t=0.72,
        energy_pred_t=0.69,
        energy_recon_error_t=0.0001,
    )
    base.update(overrides)
    return base


# ---- record version + validator -------------------------------------------


def test_phase2_record_roundtrip_with_pragmatic_fields() -> None:
    """A Phase-2 record with the decomposition fields validates, round-trips,
    and still carries the D monitor (per-dim KL)."""
    rec = AgentStep(
        schema_version=PROBE_3_5_PHASE2_TELEMETRY_SCHEMA_VERSION,
        pragmatic_value_t=-0.12,
        epistemic_value_t=0.39,
        pragmatic_share_t=0.235,
        **_agent_step_kwargs(),
    )
    again = AgentStep.model_validate_json(rec.model_dump_json())
    assert again.schema_version == PROBE_3_5_PHASE2_TELEMETRY_SCHEMA_VERSION
    assert again.pragmatic_share_t == pytest.approx(0.235)
    assert again.kl_per_dim_t == [0.0]  # D monitor retained (Amendment 01)


def test_phase2_record_rejects_missing_pragmatic_field() -> None:
    """At the Phase-2 record version every decomposition field is required
    non-None — the writer-side discipline."""
    with pytest.raises(ValueError, match="pragmatic"):
        AgentStep(
            schema_version=PROBE_3_5_PHASE2_TELEMETRY_SCHEMA_VERSION,
            pragmatic_value_t=-0.12,
            epistemic_value_t=0.39,
            pragmatic_share_t=None,
            **_agent_step_kwargs(),
        )


def test_phase2_record_rejects_missing_energy_field_too() -> None:
    """The Phase-2 validator subsumes the Phase-1 energy requirements."""
    with pytest.raises(ValueError, match="energy"):
        AgentStep(
            schema_version=PROBE_3_5_PHASE2_TELEMETRY_SCHEMA_VERSION,
            pragmatic_value_t=-0.12,
            epistemic_value_t=0.39,
            pragmatic_share_t=0.235,
            **_agent_step_kwargs(true_energy_t=None),
        )


def test_older_shards_stay_backward_readable() -> None:
    """Phase-1 ("0.4.0") and legacy ("0.2.0") records validate with the
    pragmatic fields absent → None — older shards read through unchanged."""
    phase1 = AgentStep(
        schema_version=PROBE_3_5_TELEMETRY_SCHEMA_VERSION, **_agent_step_kwargs()
    )
    assert phase1.pragmatic_value_t is None
    legacy_kwargs = _agent_step_kwargs(
        sensed_energy_t=None,
        true_energy_t=None,
        energy_pred_t=None,
        energy_recon_error_t=None,
    )
    legacy = AgentStep(schema_version=SCHEMA_VERSION, **legacy_kwargs)
    assert legacy.pragmatic_share_t is None


# ---- frozen exports ---------------------------------------------------------


def test_v0_5_0_export_is_frozen_and_pre_pragmatic() -> None:
    """The Phase-1 export is now a frozen historical artifact: the function
    returns the checked-in bytes, and those bytes predate the pragmatic
    fields."""
    data = export_json_schema_v0_5_0()
    assert data == SCHEMA_FILE_V0_5_0.read_bytes()
    assert b"pragmatic" not in data


def test_v0_6_0_export_byte_stable_and_matches_disk() -> None:
    """The Phase-2 export is byte-stable and matches the checked-in
    ``schemas/v0.6.0.json``. Since Phase 3 (which widened DreamRollout with
    the §7 monitor field) it is a **frozen historical artifact** — the
    function reads the checked-in bytes; the live export is v0.7.0
    (tests/test_dream_energy_monitor.py)."""
    first = export_json_schema_v0_6_0()
    second = export_json_schema_v0_6_0()
    assert first == second
    assert SCHEMA_FILE_V0_6_0.read_bytes() == first
    assert b"sequence_decoded_energy" not in first  # pre-monitor artifact


def test_v0_6_0_export_carries_the_decomposition_fields_and_versions() -> None:
    document = json.loads(export_json_schema_v0_6_0())
    assert document["schema_version"] == PROBE_3_5_PHASE2_EXPORT_VERSION
    assert (
        document["telemetry_schema_version"]
        == PROBE_3_5_PHASE2_TELEMETRY_SCHEMA_VERSION
    )
    agent_step = document["models"]["telemetry"]["AgentStep"]
    for field in ("pragmatic_value_t", "epistemic_value_t", "pragmatic_share_t"):
        assert field in agent_step["properties"], field


# ---- runner emission (opt-in) ----------------------------------------------


def test_runner_stamps_phase2_version_when_preference_configured(
    tmp_path: Path,
) -> None:
    """With ``energy_telemetry=True`` and an ``energy_preference`` configured,
    AgentStep records stamp the Phase-2 version with non-None decomposition
    fields; pre-first-training-step records carry the genuine zero
    decomposition (no preference gradient has existed yet). Opt-in: with
    ``energy_preference=None`` the existing smokes (test_integration_smoke)
    already pin the legacy emission byte-identically."""
    import dataclasses

    from kind.agents.preference import EnergyPreferenceConfig
    from tests.test_integration_smoke import (
        _make_runner_config,
        _read_parquet_records,
        _transport_pair,
    )
    from kind.training.runner import Runner

    config = dataclasses.replace(
        _make_runner_config(tmp_path=tmp_path),
        energy_telemetry=True,
        energy_preference=EnergyPreferenceConfig(precision=2.0),
        warmup_env_steps=10,
    )
    with _transport_pair() as (client, _server, env_server, _thread):
        runner = Runner(config, client, env_server=env_server)
        try:
            runner.run(total_env_steps=40)
        finally:
            runner.close()

    rows = _read_parquet_records(config.telemetry_dir / "agent_step")
    assert len(rows) == 40
    for row in rows:
        assert row["schema_version"] == PROBE_3_5_PHASE2_TELEMETRY_SCHEMA_VERSION
        assert row["pragmatic_value_t"] is not None
        assert row["epistemic_value_t"] is not None
        assert row["pragmatic_share_t"] is not None
        assert row["true_energy_t"] is not None
    # Pre-first-training-step records: the decomposition is genuinely zero.
    first = rows[0]
    assert first["pragmatic_value_t"] == 0.0
    assert first["pragmatic_share_t"] == 0.0
    # After warmup, training steps run and the epistemic term is live.
    last = rows[-1]
    assert last["epistemic_value_t"] != 0.0
