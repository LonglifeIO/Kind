"""Telemetry schemas (Probe 1 → Probe 1.5).

Pydantic v2 models for the four telemetry streams named in the Probe 1
implementation plan §3: ``AgentStep``, ``DreamRollout``, ``ReplayMeta``,
``WorldEvent``. Each subclasses ``RecordEnvelope``, which carries the
versioning and run-identification fields common to every record.

These models are pure declarations. They read nothing and write nothing.
Sinks (Phase 1) consume them; the env-server, agent, and runner produce
instances of them.

The schema version is at ``"0.2.0"`` from Probe 1.5 forward. The Probe 1.5
implementation plan §2.1 / §3 names three new optional fields on
``AgentStep`` (``self_prediction_t``, ``self_prediction_error_t``,
``self_prediction_error_masked_t``) and one reserved optional field on
``DreamRollout`` (``sequence_self_prediction``). All four are declared
``Optional[T] = None`` so that records written under ``"0.1.0"`` (Probe 1's
parquet shards under ``runs/probe1-20260503-123926/``) deserialize cleanly
against the new models — the absence of the new fields becomes ``None``.

A custom validator on ``AgentStep`` enforces the writer-side discipline: a
record stamped ``schema_version == "0.2.0"`` must populate all three new
fields with non-None values (the masked flag is a real boolean — ``True``
on the first step of each episode, ``False`` thereafter; the scalar takes
its sentinel value 0.0 when masked but is still non-None). A record
stamped ``"0.1.0"`` is a Probe-1-shaped record and the new fields stay
``None``. This is the "mixed-version writer rejection" check from the
implementation plan §3.3.

Both ``schemas/v0.1.0.json`` and ``schemas/v0.2.0.json`` are checked in.
``v0.1.0.json`` is a frozen historical export and is no longer regenerated
from this module; ``v0.2.0.json`` is the current export from
``export_json_schema()``.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

SCHEMA_VERSION: str = "0.2.0"
PROBE_1_SCHEMA_VERSION: str = "0.1.0"


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

    Probe 1.5 adds three optional fields tied to the self-prediction head
    (synthesis §1.3 v2): ``self_prediction_t`` is the head's output vector
    ``ĥ_{t+1}`` (length ``h_dim``); ``self_prediction_error_t`` is the
    scalar loss between the head's output and the EMA target's
    ``bar{h}_{t+1}`` (the value Io reads on PolicyView via the single-scalar
    Watts-heuristic exception); ``self_prediction_error_masked_t`` is True
    on the first step of each episode (no actual ``h_{t+1}`` available, so
    the scalar is forced to 0.0 as a sentinel) and False on subsequent
    steps. All three are ``None`` on Probe 1 records (``"0.1.0"``); all
    three must be non-None on Probe 1.5 records (``"0.2.0"``) — the
    ``_enforce_v2_required_fields`` validator below is the writer-side
    check.
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

    self_prediction_t: list[float] | None = None
    self_prediction_error_t: float | None = None
    self_prediction_error_masked_t: bool | None = None

    @model_validator(mode="after")
    def _enforce_v2_required_fields(self) -> "AgentStep":
        if self.schema_version == SCHEMA_VERSION:
            missing = [
                name
                for name, value in (
                    ("self_prediction_t", self.self_prediction_t),
                    ("self_prediction_error_t", self.self_prediction_error_t),
                    (
                        "self_prediction_error_masked_t",
                        self.self_prediction_error_masked_t,
                    ),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    f"AgentStep with schema_version=={SCHEMA_VERSION!r} requires "
                    f"non-None values for {missing}. Probe 1.5 writers populate the "
                    f"three self-prediction fields on every emission (synthesis §1.3 "
                    f"v2; implementation plan §3.3 'mixed-version writer rejection'). "
                    f"Probe 1 records use schema_version={PROBE_1_SCHEMA_VERSION!r} and "
                    f"leave the three fields None."
                )
        return self


class DreamRollout(RecordEnvelope):
    """One record per imagination episode (default cadence ~1 per 1k env steps).

    Probe 1 emits dream rollouts as a calibration handshake — nothing trains
    on them — so that the imagination conduit is exercised and visible in
    telemetry from day one.

    Probe 1.5 reserves ``sequence_self_prediction`` (synthesis §1.5 v2):
    the self-prediction head runs only during waking at Probe 1.5, so this
    field stays ``None``; Probe 3 may populate it if its design extends
    self-prediction to imagined trajectories (the framing question is
    journaled at ``docs/workingjournal/pre-probe3.md``). The reserved slot
    is the forward-compatibility hook — no schema bump is needed when
    Probe 3 lands.
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

    sequence_self_prediction: list[list[float]] | None = None


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
        "title": "Kind Telemetry Schemas",
        "schema_version": SCHEMA_VERSION,
        "models": {model.__name__: model.model_json_schema() for model in RECORD_MODELS},
    }
    text = json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False)
    return (text + "\n").encode("utf-8")


__all__ = [
    "SCHEMA_VERSION",
    "PROBE_1_SCHEMA_VERSION",
    "RecordEnvelope",
    "AgentStep",
    "DreamRollout",
    "ReplayMeta",
    "WorldEvent",
    "WorldEventType",
    "RECORD_MODELS",
    "export_json_schema",
]
