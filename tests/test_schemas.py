"""Phase 0 gate test for ``kind/observer/schemas.py``.

Three checks:

1. The four schema models import.
2. Each can be instantiated with valid dummy data.
3. The JSON Schema export is byte-stable across invocations and matches the
   checked-in ``schemas/v0.1.0.json``.

The full JSONL roundtrip test (plan §4 test #4) requires sinks and is deferred
to Phase 1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kind.observer.schemas import (
    RECORD_MODELS,
    SCHEMA_VERSION,
    AgentStep,
    DreamRollout,
    ReplayMeta,
    WorldEvent,
    export_json_schema,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_FILE = REPO_ROOT / "schemas" / "v0.1.0.json"


def test_schema_version_constant() -> None:
    assert SCHEMA_VERSION == "0.1.0"


def test_record_models_collection() -> None:
    assert tuple(m.__name__ for m in RECORD_MODELS) == (
        "AgentStep",
        "DreamRollout",
        "ReplayMeta",
        "WorldEvent",
    )


@pytest.fixture
def envelope() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": "run-0000",
        "checkpoint_id": None,
    }


def test_agent_step_instantiates(envelope: dict[str, object]) -> None:
    record = AgentStep(
        **envelope,
        t=0,
        episode_id=0,
        step_in_episode=0,
        wallclock_ms=1_700_000_000_000,
        h_t=[0.0] * 4,
        q_params_t=([0.0] * 2, [0.0] * 2),
        p_params_t=([0.0] * 2, [0.0] * 2),
        z_t=[0.0] * 2,
        kl_per_dim_t=[0.0] * 2,
        kl_aggregate_t=0.0,
        recon_loss_t=0.0,
        action_t=0,
        action_logprob_t=0.0,
        policy_entropy_t=0.0,
        obs_hash_t="0" * 16,
        intrinsic_signal_t=0.0,
        encoder_embedding_t=[0.0] * 4,
    )
    assert record.schema_version == SCHEMA_VERSION
    assert record.t == 0


def test_dream_rollout_instantiates(envelope: dict[str, object]) -> None:
    record = DreamRollout(
        **envelope,
        seed_step=0,
        seed_h0=[0.0] * 4,
        seed_z0=[0.0] * 2,
        sequence_h=[[0.0] * 4 for _ in range(3)],
        sequence_z_prior=[[0.0] * 2 for _ in range(3)],
        sequence_action=[0, 0, 0],
        sequence_action_logprob=[0.0, 0.0, 0.0],
        sequence_prior_entropy=[0.0, 0.0, 0.0],
        sequence_decoded_obs=None,
        cumulative_prior_entropy=0.0,
        mean_step_kl_successive_priors=0.0,
        max_step_latent_norm_change=0.0,
    )
    assert record.seed_step == 0
    assert len(record.sequence_h) == 3


def test_replay_meta_instantiates(envelope: dict[str, object]) -> None:
    record = ReplayMeta(
        **envelope,
        event_type="insert",
        t_event=0,
        segment_id=0,
        segment_start=0,
        segment_end=32,
        priority=None,
        buffer_size=1,
        total_segments=1,
    )
    assert record.event_type == "insert"


def test_world_event_instantiates(envelope: dict[str, object]) -> None:
    record = WorldEvent(
        **envelope,
        t_event=0,
        event_type="env_reset",
        source="environment",
        payload={"episode_id": 0},
        wallclock_ms=1_700_000_000_000,
    )
    assert record.event_type == "env_reset"
    assert record.payload == {"episode_id": 0}


def test_envelope_fields_present_on_every_model(envelope: dict[str, object]) -> None:
    for model in RECORD_MODELS:
        fields = model.model_fields
        for key in ("schema_version", "run_id", "checkpoint_id"):
            assert key in fields, f"{model.__name__} missing envelope field {key!r}"


def test_json_schema_export_is_byte_stable() -> None:
    first = export_json_schema()
    second = export_json_schema()
    assert first == second


def test_json_schema_export_matches_checked_in_file() -> None:
    assert SCHEMA_FILE.exists(), f"missing schema export at {SCHEMA_FILE}"
    on_disk = SCHEMA_FILE.read_bytes()
    assert on_disk == export_json_schema(), (
        "schemas/v0.1.0.json is out of sync with kind/observer/schemas.py — "
        "regenerate via export_json_schema() and commit the result."
    )
