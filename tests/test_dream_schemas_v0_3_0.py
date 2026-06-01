"""Probe 3 Phase 0 — schema additions gate tests.

Six tests per the implementation plan §4 (Phase 0 row):

1. ``test_dream_rollout_v0_3_0_round_trip_all_fields`` — happy-path JSON
   round-trip with all fourteen new fields populated.
2. ``test_dream_rollout_v0_2_0_backward_compat`` — Probe-1.5-shaped JSON
   loads cleanly; the fourteen new fields surface as ``None``.
3. ``test_v3_validator_rejects_missing_seed_kind`` — the
   ``_enforce_v3_required_fields`` validator rejects an incomplete
   ``"0.3.0"`` record.
4. ``test_world_event_state_transition_payload_round_trip`` — the new
   ``"state_transition"`` ``WorldEventType`` value plus its payload
   convention round-trips.
5. ``test_dream_session_meta_double_write`` — the start-then-end
   double-write pattern preserves both records in JSONL.
6. ``test_schemas_v0_4_0_json_byte_stable`` — the v0.4.0 export is
   byte-stable across invocations and matches the file on disk.

The Probe 1.5 ParquetSink path for DreamRollout still lives in
``tests/test_schemas.py``; this file's tests use Pydantic JSON
serialization (``model_dump_json`` / ``model_validate_json``) so the
schema is exercised without coupling to the Arrow column derivation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from kind.observer.dream_session import (
    DREAM_SESSION_FILE,
    DREAM_SESSION_META_SCHEMA_VERSION,
    DreamSessionMeta,
    DreamSessionSink,
)
from kind.observer.schemas import (
    PROBE_3_EXPORT_VERSION,
    PROBE_3_TELEMETRY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    DreamRollout,
    WorldEvent,
    export_json_schema_v0_4_0,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_FILE_V0_4_0 = REPO_ROOT / "schemas" / "v0.4.0.json"

H_DIM = 4
Z_DIM = 2
HORIZON = 6


def _dream_rollout_base_fields() -> dict[str, Any]:
    """Probe 1.5-shaped required fields shared by all DreamRollout
    fixtures in this file. The new fields are layered on top.
    """
    return {
        "run_id": "run-0000",
        "checkpoint_id": "ckpt-000001",
        "seed_step": 100,
        "seed_h0": [float(i) * 0.01 for i in range(H_DIM)],
        "seed_z0": [float(i) * 0.01 for i in range(Z_DIM)],
        "sequence_h": [
            [float(j + i) * 0.01 for i in range(H_DIM)] for j in range(HORIZON)
        ],
        "sequence_z_prior": [
            [float(j + i) * 0.01 for i in range(Z_DIM)] for j in range(HORIZON)
        ],
        "sequence_action": [i % 5 for i in range(HORIZON)],
        "sequence_action_logprob": [-float(i) * 0.1 for i in range(HORIZON)],
        "sequence_prior_entropy": [float(i) * 0.05 for i in range(HORIZON)],
        "sequence_decoded_obs": None,
        "cumulative_prior_entropy": 0.5,
        "mean_step_kl_successive_priors": 0.1,
        "max_step_latent_norm_change": 0.5,
        "sequence_self_prediction": None,
    }


def _dream_rollout_v0_3_0_all_fields() -> DreamRollout:
    """A fully-populated Probe 3 DreamRollout: ``schema_version='0.3.0'``,
    ``seed_kind='hybrid'`` so all three conditional seed-provenance fields
    are required and populated, plus all optional fields populated for
    the strongest round-trip coverage.
    """
    return DreamRollout(
        schema_version=PROBE_3_TELEMETRY_SCHEMA_VERSION,
        **_dream_rollout_base_fields(),
        dream_session_id="session-abc-123",
        seed_kind="hybrid",
        seed_replay_segment_id=42,
        seed_replay_step_offset=7,
        seed_perturbation_magnitude=0.1414,
        temperature_schedule=[1.5, 1.5, 1.5, 1.8, 2.1, 2.5],
        sub_mode_tags=["chimera", "associative_drift_tail"],
        sampling_parameters={"action_temperature": 1.5, "re_seed_every_n_steps": 10.0},
        gradient_policy="none",
        rng_seed=1729,
        termination_reason="horizon_complete",
        re_seed_step_indices=[3],
        sequence_ensemble_disagreement_variance=[
            0.01, 0.02, 0.03, 0.04, 0.05, 0.06
        ],
        checkpoint_hash="a" * 64,
    )


def _dream_rollout_v0_2_0_dict() -> dict[str, Any]:
    """A Probe-1.5-shaped DreamRollout payload (no new fields).
    Returned as a dict to exercise the backward-readability JSON path.
    """
    base = _dream_rollout_base_fields()
    return {"schema_version": SCHEMA_VERSION, **base}


# ---- Test 1: round-trip with all fourteen new fields ----------------------


def test_dream_rollout_v0_3_0_round_trip_all_fields() -> None:
    """A v0.3.0 DreamRollout with all fourteen new fields populated
    serializes to JSON and deserializes back equal to the original.

    Happy-path: confirms the schema additions accept every documented
    field value and round-trip without loss.
    """
    original = _dream_rollout_v0_3_0_all_fields()
    encoded = original.model_dump_json()
    reconstructed = DreamRollout.model_validate_json(encoded)

    assert reconstructed == original

    # Spot-check every new field survives the round-trip with its
    # configured value.
    assert reconstructed.schema_version == PROBE_3_TELEMETRY_SCHEMA_VERSION
    assert reconstructed.dream_session_id == "session-abc-123"
    assert reconstructed.seed_kind == "hybrid"
    assert reconstructed.seed_replay_segment_id == 42
    assert reconstructed.seed_replay_step_offset == 7
    assert reconstructed.seed_perturbation_magnitude == pytest.approx(0.1414)
    assert reconstructed.temperature_schedule == [1.5, 1.5, 1.5, 1.8, 2.1, 2.5]
    assert reconstructed.sub_mode_tags == ["chimera", "associative_drift_tail"]
    assert reconstructed.sampling_parameters == {
        "action_temperature": 1.5,
        "re_seed_every_n_steps": 10.0,
    }
    assert reconstructed.gradient_policy == "none"
    assert reconstructed.rng_seed == 1729
    assert reconstructed.termination_reason == "horizon_complete"
    assert reconstructed.re_seed_step_indices == [3]
    assert reconstructed.sequence_ensemble_disagreement_variance == [
        0.01, 0.02, 0.03, 0.04, 0.05, 0.06
    ]
    assert reconstructed.checkpoint_hash == "a" * 64


# ---- Test 2: backward compat — v0.2.0 record reads with new fields = None --


def test_dream_rollout_v0_2_0_backward_compat() -> None:
    """A Probe-1.5-shaped DreamRollout record (``schema_version='0.2.0'``,
    no new fields populated) reads through the extended model with all
    fourteen new fields surfacing as ``None``.

    This is the backward-readability path the synthesis §1 commitment
    rests on: the schema bump is additive at the model level, so the
    fourteen new fields default to ``None`` for older records, and the
    v3 validator does not fire on ``"0.2.0"`` records.
    """
    record_dict = _dream_rollout_v0_2_0_dict()
    record = DreamRollout.model_validate(record_dict)

    # Probe 1.5 baseline preserved.
    assert record.schema_version == SCHEMA_VERSION
    assert record.sequence_self_prediction is None

    # Every one of the fourteen new fields is None.
    assert record.dream_session_id is None
    assert record.seed_kind is None
    assert record.seed_replay_segment_id is None
    assert record.seed_replay_step_offset is None
    assert record.seed_perturbation_magnitude is None
    assert record.temperature_schedule is None
    assert record.sub_mode_tags is None
    assert record.sampling_parameters is None
    assert record.gradient_policy is None
    assert record.rng_seed is None
    assert record.termination_reason is None
    assert record.re_seed_step_indices is None
    assert record.sequence_ensemble_disagreement_variance is None
    assert record.checkpoint_hash is None

    # Round-trip through JSON also succeeds (no validator surprises).
    round_tripped = DreamRollout.model_validate_json(record.model_dump_json())
    assert round_tripped == record


# ---- Test 3: validator rejects missing seed_kind on v0.3.0 ----------------


def test_v3_validator_rejects_missing_seed_kind() -> None:
    """A record stamped ``schema_version='0.3.0'`` with ``seed_kind=None``
    raises ``ValidationError`` at construction.

    The ``_enforce_v3_required_fields`` validator is the writer-side
    discipline for Probe 3 telemetry: every v3 record must populate the
    nine unconditionally required fields (synthesis §1 commits the
    fields as load-bearing). The error message names the missing field
    so writer-side bugs surface specifically.
    """
    kwargs: dict[str, Any] = {
        "schema_version": PROBE_3_TELEMETRY_SCHEMA_VERSION,
        **_dream_rollout_base_fields(),
        # Populate every other unconditionally required field so that
        # ``seed_kind`` is the sole missing field, and the test asserts
        # that the validator names it specifically.
        "dream_session_id": "session-abc-123",
        "seed_kind": None,
        "temperature_schedule": [1.5] * HORIZON,
        "sampling_parameters": {"action_temperature": 1.5},
        "gradient_policy": "none",
        "rng_seed": 1729,
        "termination_reason": "horizon_complete",
        "sequence_ensemble_disagreement_variance": [0.0] * HORIZON,
        "checkpoint_hash": "a" * 64,
    }
    with pytest.raises(ValidationError) as exc_info:
        DreamRollout(**kwargs)
    assert "seed_kind" in str(exc_info.value), (
        f"ValidationError must name the missing seed_kind field for "
        f"writer-side debuggability; got: {exc_info.value}"
    )


# ---- Test 4: WorldEvent state_transition payload round-trip ---------------


def test_world_event_state_transition_payload_round_trip() -> None:
    """A WorldEvent with ``event_type='state_transition'`` and the payload
    convention named in the plan §2.1 round-trips through JSON.

    The payload is schemaless within the field (per the WorldEvent
    docstring); this test only validates round-trip preservation of the
    new Literal value and the documented payload shape, not Pydantic
    enforcement of the payload's internal structure.
    """
    payload: dict[str, Any] = {
        "from_state": "waking",
        "to_state": "dreaming",
        "dream_session_id": "session-abc-123",
        "trigger": "desktop_off",
        "wallclock_ms_in_prev_state": 60_000,
        "env_step_at_transition": 1500,
    }
    original = WorldEvent(
        schema_version=PROBE_3_TELEMETRY_SCHEMA_VERSION,
        run_id="run-0000",
        checkpoint_id="ckpt-000001",
        t_event=1500,
        event_type="state_transition",
        source="environment",
        payload=payload,
        wallclock_ms=1_700_000_001_500,
    )
    encoded = original.model_dump_json()
    reconstructed = WorldEvent.model_validate_json(encoded)

    assert reconstructed == original
    assert reconstructed.event_type == "state_transition"
    assert reconstructed.payload == payload

    # And the parallel dormant_heartbeat case round-trips.
    dormant_payload: dict[str, Any] = {
        "dormant_started_at_ms": 1_700_000_002_000,
        "dormant_wallclock_ms_elapsed": 60_000,
        "mac_alive": True,
    }
    dormant = WorldEvent(
        schema_version=PROBE_3_TELEMETRY_SCHEMA_VERSION,
        run_id="run-0000",
        checkpoint_id="ckpt-000001",
        t_event=1560,
        event_type="dormant_heartbeat",
        source="environment",
        payload=dormant_payload,
        wallclock_ms=1_700_000_062_000,
    )
    dormant_rt = WorldEvent.model_validate_json(dormant.model_dump_json())
    assert dormant_rt == dormant
    assert dormant_rt.event_type == "dormant_heartbeat"
    assert dormant_rt.payload["mac_alive"] is True


# ---- Test 5: DreamSessionMeta start/end double-write ----------------------


def test_dream_session_meta_double_write(tmp_path: Path) -> None:
    """The start-then-end double-write pattern: emit a start record
    (``ended_*=None``), then an end record (all fields populated), both
    sharing ``dream_session_id``. Read both back from the JSONL stream;
    confirm distinguishable by which has ``ended_*`` populated.
    """
    session_id = "session-abc-123"
    common: dict[str, Any] = {
        "schema_version": DREAM_SESSION_META_SCHEMA_VERSION,
        "run_id": "run-0000",
        "checkpoint_id": "ckpt-000001",
        "dream_session_id": session_id,
        "started_at_env_step": 1000,
        "started_at_wallclock_ms": 1_700_000_000_000,
        "envelope_config_snapshot": {
            "hard_cap_wallclock_ms": 1_800_000,
            "hard_cap_rollout_count": 50,
        },
        "seed_selection_config_snapshot": {
            "mode": "replay",
            "replay_min_segment_age_steps": 1000,
        },
    }
    start = DreamSessionMeta(
        **common,
        ended_at_env_step=None,
        ended_at_wallclock_ms=None,
        end_trigger=None,
        rollout_count=0,
    )
    end = DreamSessionMeta(
        **common,
        ended_at_env_step=1500,
        ended_at_wallclock_ms=1_700_000_001_500,
        end_trigger="desktop_on",
        rollout_count=12,
    )

    sink_dir = tmp_path / "telemetry"
    with DreamSessionSink(sink_dir) as sink:
        sink.write(start)
        sink.write(end)

    path = sink_dir / DREAM_SESSION_FILE
    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, (
        f"expected two records (start + end), got {len(lines)}"
    )

    records = [DreamSessionMeta.model_validate_json(line) for line in lines]
    start_rt, end_rt = records

    # Both records share the same session id.
    assert start_rt.dream_session_id == session_id
    assert end_rt.dream_session_id == session_id

    # The start record's ended_* fields are None.
    assert start_rt.ended_at_env_step is None
    assert start_rt.ended_at_wallclock_ms is None
    assert start_rt.end_trigger is None
    assert start_rt.rollout_count == 0

    # The end record carries the closure fields.
    assert end_rt.ended_at_env_step == 1500
    assert end_rt.ended_at_wallclock_ms == 1_700_000_001_500
    assert end_rt.end_trigger == "desktop_on"
    assert end_rt.rollout_count == 12

    # Config snapshots survive on both records (the mirror reads the
    # start record's snapshot; the verifier resolves against the end
    # record's snapshot — both must be present).
    assert start_rt.envelope_config_snapshot == common["envelope_config_snapshot"]
    assert end_rt.envelope_config_snapshot == common["envelope_config_snapshot"]
    assert start_rt.seed_selection_config_snapshot == common["seed_selection_config_snapshot"]


# ---- Test 6: schemas/v0.4.0.json byte-stable ------------------------------


def test_schemas_v0_4_0_json_byte_stable() -> None:
    """The v0.4.0 export is byte-stable across invocations and matches
    the checked-in file at ``schemas/v0.4.0.json``.

    Same discipline as the v0.2.0 and v0.3.0 exports: out-of-sync edits
    to any contributing model module fail here — regenerate via
    :func:`export_json_schema_v0_4_0` and commit the result.
    """
    first = export_json_schema_v0_4_0()
    second = export_json_schema_v0_4_0()
    assert first == second, (
        "export_json_schema_v0_4_0 is not byte-stable across "
        "invocations — check Pydantic schema generation for "
        "non-deterministic ordering."
    )

    assert SCHEMA_FILE_V0_4_0.exists(), (
        f"missing schema export at {SCHEMA_FILE_V0_4_0} — regenerate "
        f"via export_json_schema_v0_4_0() and commit the result."
    )
    assert SCHEMA_FILE_V0_4_0.read_bytes() == first, (
        f"{SCHEMA_FILE_V0_4_0} is out of sync with "
        "export_json_schema_v0_4_0() — regenerate and commit."
    )

    # Structural sanity: the export covers four model families
    # (telemetry, mirror, conditioning, dream) and stamps the v0.4.0
    # export-file version plus the v0.3.0 telemetry record version.
    document = json.loads(first)
    assert document["schema_version"] == PROBE_3_EXPORT_VERSION
    assert document["telemetry_schema_version"] == PROBE_3_TELEMETRY_SCHEMA_VERSION
    assert "DreamSessionMeta" in document["models"]["dream"]
    assert "DreamRollout" in document["models"]["telemetry"]
    # The two new WorldEventType values are present in the export.
    we_event_type = document["models"]["telemetry"]["WorldEvent"]["properties"][
        "event_type"
    ]
    enum = we_event_type.get("enum")
    if enum is None:
        ref = we_event_type["$ref"].split("/")[-1]
        enum = document["models"]["telemetry"]["WorldEvent"]["$defs"][ref]["enum"]
    assert "state_transition" in enum
    assert "dormant_heartbeat" in enum
    # And the closed gradient_policy Literal is still single-valued.
    dr_props = document["models"]["telemetry"]["DreamRollout"]["properties"]
    gp = dr_props["gradient_policy"]
    # Optional[Literal["none"]] serializes as anyOf with one string enum
    # arm and one null arm. Either form is acceptable; we assert "none"
    # is in the only string-arm enum.
    gp_enums: list[str] = []
    if "enum" in gp:
        gp_enums = list(gp["enum"])
    elif "const" in gp:
        gp_enums = [gp["const"]]
    elif "anyOf" in gp:
        for arm in gp["anyOf"]:
            if arm.get("type") == "string":
                if "enum" in arm:
                    gp_enums.extend(arm["enum"])
                elif "const" in arm:
                    gp_enums.append(arm["const"])
    assert gp_enums == ["none"], (
        f"gradient_policy must be a closed Literal of exactly "
        f'["none"] at Probe 3; saw {gp_enums}'
    )
