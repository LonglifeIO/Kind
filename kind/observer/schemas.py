"""Probe 1 telemetry schemas.

Pydantic v2 models for the four telemetry streams named in the Probe 1
implementation plan §3: ``AgentStep``, ``DreamRollout``, ``ReplayMeta``,
``WorldEvent``. Each subclasses ``RecordEnvelope``, which carries the
versioning and run-identification fields common to every record.

These models are pure declarations. They read nothing and write nothing.
Sinks (Phase 1) will consume them; the env-server, agent, and runner
(Phases 2-5) will produce instances of them.

The schema version is frozen at ``"0.1.0"`` for Probe 1. Field additions in
later probes bump the patch or minor and must remain backward-readable;
deprecations are marked, never deleted. ``schemas/v0.1.0.json`` is the
checked-in JSON Schema export.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION: str = "0.1.0"


class RecordEnvelope(BaseModel):
    """Common envelope inherited by every telemetry record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    run_id: str
    checkpoint_id: str | None


class AgentStep(RecordEnvelope):
    """One record per environment timestep, written by the agent process.

    Captures the full per-step substrate state the mirror needs (h_t, posterior
    and prior parameters, sampled latent, KL, recon, action, intrinsic signal,
    encoder embedding) plus indexing fields for downstream alignment.
    """

    t: int
    episode_id: int
    step_in_episode: int
    wallclock_ms: int

    h_t: list[float]
    q_params_t: tuple[list[float], list[float]]
    p_params_t: tuple[list[float], list[float]]
    z_t: list[float]

    kl_per_dim_t: list[float]
    kl_aggregate_t: float
    recon_loss_t: float

    action_t: int
    action_logprob_t: float
    policy_entropy_t: float

    obs_hash_t: str
    intrinsic_signal_t: float
    encoder_embedding_t: list[float]


class DreamRollout(RecordEnvelope):
    """One record per imagination episode (default cadence ~1 per 1k env steps).

    Probe 1 emits dream rollouts as a calibration handshake — nothing trains
    on them — so that the imagination conduit is exercised and visible in
    telemetry from day one.
    """

    seed_step: int
    seed_h0: list[float]
    seed_z0: list[float]

    sequence_h: list[list[float]]
    sequence_z_prior: list[list[float]]
    sequence_action: list[int]
    sequence_action_logprob: list[float]
    sequence_prior_entropy: list[float]
    sequence_decoded_obs: list[bytes] | None

    cumulative_prior_entropy: float
    mean_step_kl_successive_priors: float
    max_step_latent_norm_change: float


class ReplayMeta(RecordEnvelope):
    """One record per replay buffer event (insert / sample / evict)."""

    event_type: Literal["insert", "sample", "evict"]
    t_event: int
    segment_id: int
    segment_start: int
    segment_end: int
    priority: float | None
    buffer_size: int
    total_segments: int


# ``mirror_marker`` is reserved for Probe 2's mirror to mark its own readings
# against the telemetry timeline. No Probe 1 component emits these events;
# the slot is in the literal so Probe 2 can populate it without a schema bump.
WorldEventType = Literal[
    "builder_perturbation",
    "env_reset",
    "internal_stochasticity_aggregate",
    "mirror_marker",
]


class WorldEvent(RecordEnvelope):
    """One record per external event in the world.

    The ``payload`` field is intentionally schemaless within itself (plan §3.4):
    the same model carries per-event records (Probe 4's individual builder
    mutators) and per-episode aggregates (Probe 1's
    ``internal_stochasticity_aggregate`` rollups). Adding new payload shapes
    for future probes does not bump the schema version.
    """

    t_event: int
    event_type: WorldEventType
    source: Literal["builder", "environment", "system"]
    payload: dict[str, Any]
    wallclock_ms: int


RECORD_MODELS: tuple[type[RecordEnvelope], ...] = (
    AgentStep,
    DreamRollout,
    ReplayMeta,
    WorldEvent,
)


def export_json_schema() -> bytes:
    """Return a byte-stable JSON Schema export covering all four record models.

    Stability means: the same Python source produces the same bytes on every
    invocation. ``sort_keys=True`` plus a fixed indent of 2 plus a trailing
    newline pins the textual form; Pydantic's own schema generation is
    deterministic for a given set of inputs.
    """

    document: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Kind Probe 1 Telemetry Schemas",
        "schema_version": SCHEMA_VERSION,
        "models": {model.__name__: model.model_json_schema() for model in RECORD_MODELS},
    }
    text = json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False)
    return (text + "\n").encode("utf-8")


__all__ = [
    "SCHEMA_VERSION",
    "RecordEnvelope",
    "AgentStep",
    "DreamRollout",
    "ReplayMeta",
    "WorldEvent",
    "WorldEventType",
    "RECORD_MODELS",
    "export_json_schema",
]
