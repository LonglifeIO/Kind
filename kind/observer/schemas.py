"""Telemetry schemas (Probe 1 → Probe 1.5 → Probe 3).

Pydantic v2 models for the four telemetry streams named in the Probe 1
implementation plan §3: ``AgentStep``, ``DreamRollout``, ``ReplayMeta``,
``WorldEvent``. Each subclasses ``RecordEnvelope``, which carries the
versioning and run-identification fields common to every record.

These models are pure declarations. They read nothing and write nothing.
Sinks (Phase 1) consume them; the env-server, agent, and runner produce
instances of them.

The schema version is at ``"0.2.0"`` from Probe 1.5 forward, with Probe 3
extending ``DreamRollout`` and ``WorldEvent`` to a ``"0.3.0"`` writer-side
shape (Probe 3 implementation plan §3). The Probe 1.5 implementation plan
§2.1 / §3 names three new optional fields on ``AgentStep``
(``self_prediction_t``, ``self_prediction_error_t``,
``self_prediction_error_masked_t``) and one reserved optional field on
``DreamRollout`` (``sequence_self_prediction``). Probe 3 adds fourteen
optional fields to ``DreamRollout`` (the dream-session id, seed
provenance, sampling regime, ensemble disagreement trajectory,
reproducibility fields, etc.; full table in plan §3.1) and two new
``WorldEventType`` Literal values (``"state_transition"``,
``"dormant_heartbeat"``). All additions are declared
``Optional[T] = None`` so that records written under earlier versions
deserialize cleanly against the new models — the absence of the new
fields becomes ``None``.

Two writer-side validators enforce the version discipline:

- ``AgentStep._enforce_v2_required_fields`` rejects ``"0.2.0"`` records
  with any of the three self-prediction fields ``None``.
- ``DreamRollout._enforce_v3_required_fields`` rejects ``"0.3.0"``
  records that are missing the nine unconditionally required Probe 3
  fields or that fail the conditional seed-provenance requirements
  (``seed_replay_*`` for replay/hybrid seeds, ``seed_perturbation_magnitude``
  for perturbed_prior/hybrid seeds).

Records stamped ``"0.1.0"`` (Probe 1) and ``"0.2.0"`` (Probe 1.5) bypass
the v3 validator's required-field check by virtue of their version
literal, keeping Probe 1 / 1.5 parquet shards under
``runs/probe1-20260503-123926/`` and ``runs/probe1_5_*/`` backward-readable.

Three frozen JSON Schema exports are checked in:

- ``schemas/v0.1.0.json`` — Probe 1 historical export, no longer
  regenerated from this module.
- ``schemas/v0.2.0.json`` — current four-record export (Probe 1.5);
  produced by :func:`export_json_schema`.
- ``schemas/v0.3.0.json`` — Probe 2 multi-family export covering
  telemetry (v0.2.0), mirror-side models (v0.2.0), and conditioning
  models (v0.1.0); produced by :func:`export_json_schema_v0_3_0`.
- ``schemas/v0.4.0.json`` — Probe 3 multi-family export covering
  telemetry (v0.3.0; DreamRollout extended, WorldEvent literal
  extended), mirror-side and conditioning models (unchanged from
  v0.3.0.json), and the new ``DreamSessionMeta`` model (v0.1.0;
  declared in :mod:`kind.observer.dream_session`); produced by
  :func:`export_json_schema_v0_4_0`.

**Version-name collision note (plan §3.3).** Two ``"0.3.0"`` values live
in this module:

- :data:`PROBE_3_TELEMETRY_SCHEMA_VERSION` (= ``"0.3.0"``) is the
  *record-level* schema_version literal Probe 3 writers stamp on
  :class:`DreamRollout` and :class:`WorldEvent` records, and the key
  the ``_enforce_v3_required_fields`` validator gates on.
- :data:`PROBE_2_EXPORT_VERSION` (= ``"0.3.0"``) is the *export-file*
  name version for ``schemas/v0.3.0.json`` — a name on the file Probe 2
  generated, not a record-level version of any single model.

The two values share the same string but mean different things; the
constants are named with disambiguating suffixes so writer-side code
references the right one. Probe 3's export-file version is
:data:`PROBE_3_EXPORT_VERSION` (= ``"0.4.0"``), keeping the file
name and the record-level version mechanically distinct.

**``world_event.payload`` conventions (Probe 1 → Probe 1.5 → Probe 2 → Probe 3).**
The ``WorldEvent.payload`` field is intentionally schemaless within
itself (per the docstring on :class:`WorldEvent`) — adding new payload
shapes for future probes does not bump the ``WorldEvent`` schema
version, only the closed Literal :data:`WorldEventType` gains values
when new event types are added. The conventions accumulated to date:

- ``payload["episode_id"]: int`` — emitted on ``"env_reset"`` events.
- ``payload["start_cell"]: tuple[int, int]`` — emitted on
  ``"env_reset"`` events from Probe 2 forward, when the env's
  start-cell randomization is in effect (per Probe 2 implementation
  plan §2.2).
- ``payload["is_sham"]: bool`` — emitted on ``"builder_perturbation"``
  events fired via the env-server's sham-perturbation flag-only path
  (Probe 2 implementation plan §2.2; synthesis §2.4 element 3). The
  field is set to ``True`` on sham events and is absent (or ``False``)
  on real perturbations. Sham-perturbation events are part of the
  calibration protocol's null-event test: the agent observation is
  byte-equal pre- and post-sham; the mirror's reading must not flag
  reflexive-attention or equanimity at the sham timestamp at any
  reading surface.
- ``payload["lesion_kind"]: str`` and
  ``payload["source_checkpoint"]: str`` — emitted on
  ``"mirror_marker"`` events from ``source="system"`` at the start of
  a lesion run (per Probe 2 implementation plan §2.5; synthesis §2.4
  element 4). The ``lesion_kind`` is one of
  ``"ensemble_k1"``, ``"ensemble_constant"``,
  ``"disable_self_prediction"``,
  ``"zero_or_randomize_scalar"``, or
  ``"init_zero_scalar_column"``. The ``source_checkpoint`` field
  records the checkpoint a mutation-time lesion (the
  ``init_zero_scalar_column`` case) was derived from so the lesion
  run's identity can be traced back to its source.
- ``payload["internal_stochasticity_aggregate"]: dict`` — per-episode
  aggregate emitted on ``"internal_stochasticity_aggregate"`` events
  (Probe 1 plan §3.4).
- ``"internal_stochasticity_event"`` payload (Probe 4 Phase 1, plan
  §S-CTRL): ``{process, cell, pre_state, post_state}`` — one granular
  record per regrowth resource-addition (EMPTY→RESOURCE), emitted with
  ``source="environment"`` when
  ``EnvServerConfig.emit_internal_stochasticity_events`` is on (default
  off — legacy emission byte-identical). The three comparison keys
  ``cell`` / ``pre_state`` / ``post_state`` exactly match a builder
  ``add_resource`` payload so the ENVIRONMENT and BUILDER event classes
  are directly comparable per-event (the three-way matched control,
  pre-registration §1); the discriminator key is ``process`` (value
  ``"regrowth"``) rather than ``mutator`` because no mutator fired — the
  class label lives in ``source``, and the payload names the causal
  process honestly. Unlike other payload conventions this shape is
  **validator-enforced** at the Probe 4 record version (the matched
  control's comparability is load-bearing, so the writer-side check is
  worth the departure from documented-only).
- ``"state_transition"`` payload (Probe 3, plan §2.1): ``{from_state,
  to_state, dream_session_id, trigger, wallclock_ms_in_prev_state,
  env_step_at_transition}``. ``from_state`` and ``to_state`` are one
  of ``"waking"``, ``"dreaming"``, ``"dormant"``, or ``"paused"``;
  ``trigger`` is one of ``"desktop_off"``, ``"desktop_on"``,
  ``"mac_off"``, ``"mac_on"``, ``"hard_cap_wallclock"``,
  ``"hard_cap_rollout_count"``, ``"checkpoint_window"``, or
  ``"compute_budget"``. The payload shape is project convention
  (documented here, not Pydantic-enforced) — Probe 3 phase 4 emits
  these from :class:`kind.training.state_machine.StateController`;
  Probe 3 phase 5 mirror reading consumes them as state-typed
  context. The ``state_transition`` event is environment-sourced (the
  exogenous-trigger commitment in synthesis §5 means state transitions
  carry ``source="environment"``).
- ``"dormant_heartbeat"`` payload (Probe 3, plan §2.1):
  ``{dormant_started_at_ms, dormant_wallclock_ms_elapsed,
  mac_alive: True}``. A periodic ping (default every 60s of dormant
  wallclock) so that the absence of dreaming during dormant is
  observable as a *positive* signal in the telemetry timeline, not as
  silent gaps that would be indistinguishable from ``"paused"`` (plan
  §2.5). ``source="environment"``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

SCHEMA_VERSION: str = "0.2.0"
PROBE_1_SCHEMA_VERSION: str = "0.1.0"
PROBE_3_TELEMETRY_SCHEMA_VERSION: str = "0.3.0"
# Probe 3.5 record-level AgentStep version (the energy-channel generation).
# Distinct from DreamRollout/WorldEvent's "0.3.0" — AgentStep was never bumped
# to 0.3.0 (Probe 3 left it untouched), so 0.4.0 is the natural next AgentStep
# record version. The matching export-file is ``schemas/v0.5.0.json``
# (:data:`PROBE_3_5_EXPORT_VERSION`), preserving the existing +1 offset between
# record-level and export-file versions (Probe 3: record 0.3.0 / export 0.4.0).
PROBE_3_5_TELEMETRY_SCHEMA_VERSION: str = "0.4.0"
# Probe 3.5 *Phase 2* record-level AgentStep version (the pragmatic-preference
# generation): adds the per-training-step pragmatic/epistemic decomposition
# fields (``pragmatic_value_t``, ``epistemic_value_t``, ``pragmatic_share_t``)
# the pre-registration's A2b share band and §8.4 "pragmatic share → 1"
# falsification signature read. **Version-name collision note** (the plan §3.3
# pattern, third instance): the string "0.5.0" is also
# :data:`PROBE_3_5_EXPORT_VERSION` (the *export-file* name of the now-frozen
# ``schemas/v0.5.0.json``); the two values share a string but mean different
# things, and the disambiguating constant names keep writer-side code pointed
# at the right one. The matching export-file for *this* record version is
# ``schemas/v0.6.0.json`` (:data:`PROBE_3_5_PHASE2_EXPORT_VERSION`),
# preserving the +1 offset.
PROBE_3_5_PHASE2_TELEMETRY_SCHEMA_VERSION: str = "0.5.0"
# Probe 3.5 *Phase 3* record-level **DreamRollout** version (the §7
# passive-decode monitor generation): adds ``sequence_decoded_energy``, the
# observer-side energy-belief trace recorded alongside ``sequence_decoded_obs``
# when ``DreamRolloutConfig.record_decoded_energy`` is on (default off →
# writers keep stamping the Probe-3 "0.3.0" and the field stays None,
# byte-identical to pre-monitor emission). **Collision note (fourth
# instance):** the string "0.4.0" is also the AgentStep record version
# :data:`PROBE_3_5_TELEMETRY_SCHEMA_VERSION` and the Probe-3 *export-file*
# version :data:`PROBE_3_EXPORT_VERSION` — different namespaces (DreamRollout
# record vs AgentStep record vs export file); the disambiguating constant
# names keep writer-side code pointed at the right one. The matching
# export-file is ``schemas/v0.7.0.json``
# (:data:`PROBE_3_5_PHASE3_EXPORT_VERSION`).
PROBE_3_5_PHASE3_DREAM_SCHEMA_VERSION: str = "0.4.0"
# Probe 4 Phase 1 record-level **WorldEvent** version (the per-event
# internal-stochasticity generation): the closed ``WorldEventType`` Literal
# gains ``"internal_stochasticity_event"`` — one granular record per regrowth
# resource-addition, the ENVIRONMENT class of the three-way matched control
# (pre-registration §1; plan §S-CTRL). WorldEvent record lineage: "0.1.0"
# (Probe 1 env-server writers) → "0.3.0" (Probe 3 state-machine writers) →
# "0.4.0" (this). **Collision note (fifth instance of the plan §3.3
# pattern):** the string "0.4.0" is also the AgentStep record version
# :data:`PROBE_3_5_TELEMETRY_SCHEMA_VERSION`, the DreamRollout record version
# :data:`PROBE_3_5_PHASE3_DREAM_SCHEMA_VERSION`, and the Probe-3 *export-file*
# version :data:`PROBE_3_EXPORT_VERSION` — four namespaces sharing one
# string; the disambiguating constant names keep writer-side code pointed at
# the right one. The matching export-file is ``schemas/v0.8.0.json``
# (:data:`PROBE_4_EXPORT_VERSION`), preserving the +1 offset.
PROBE_4_WORLD_EVENT_SCHEMA_VERSION: str = "0.4.0"
# Probe 4.5 Phase 2 record-level **WorldEvent** version (implementation plan
# §S-TEL; frozen prereg §4): the closed ``WorldEventType`` Literal gains
# ``"energy_fault_event"`` — one granular record per fault onset/offset, the
# observer-side ground truth for the fallible-honesty dynamic (opacity: the
# fault has no observation marker; these records and
# ``GridState.energy_fault_active`` are the only places it exists). WorldEvent
# record lineage: "0.1.0" → "0.3.0" → "0.4.0" → "0.5.0" (this). The matching
# export-file is ``schemas/v0.9.0.json``
# (:data:`PROBE_4_5_EXPORT_VERSION`), preserving the +1 offset... which the
# lineage now breaks by four: the offset is historical, not load-bearing.
PROBE_4_5_WORLD_EVENT_SCHEMA_VERSION: str = "0.5.0"


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

    # Probe 3.5 energy channel (implementation plan §S-TEL). Optional so older
    # shards ("0.1.0" / "0.2.0") deserialize cleanly (absent → None); required
    # non-None on Probe-3.5 records (the ``_enforce_probe_3_5_required_fields``
    # validator gates on the new record version). ``sensed_energy_t`` is what Io
    # observes (noisy/lagged); ``true_energy_t`` is the GridState ground truth
    # (observer-side only — never a training input); ``energy_pred_t`` is the
    # world model's decoded prediction; ``energy_recon_error_t`` is the per-step
    # squared error ``(energy_pred − sensed_energy)²`` (computed observer-side).
    sensed_energy_t: float | None = None
    true_energy_t: float | None = None
    energy_pred_t: float | None = None
    energy_recon_error_t: float | None = None

    # Probe 3.5 Phase 2 pragmatic/epistemic decomposition (plan §S-TEL;
    # grounding fact 3: the share is a **per-training-step** quantity — the
    # runner stamps the most recent imagination-training decomposition onto
    # subsequent AgentStep records; before the first training step all three
    # are genuinely 0.0 (no preference gradient has existed yet).
    # ``pragmatic_value_t`` / ``epistemic_value_t`` are the per-step batch
    # means of the two summed objective terms; ``pragmatic_share_t`` is
    # |P| / (|E| + |P|). Optional so older shards deserialize cleanly;
    # required non-None at the Phase-2 record version.
    pragmatic_value_t: float | None = None
    epistemic_value_t: float | None = None
    pragmatic_share_t: float | None = None

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

    @model_validator(mode="after")
    def _enforce_probe_3_5_required_fields(self) -> "AgentStep":
        """Probe 3.5 writer-side discipline (implementation plan §S-TEL).

        Records stamped :data:`PROBE_3_5_TELEMETRY_SCHEMA_VERSION` must carry
        non-None values for *both* the Probe 1.5 self-prediction fields **and**
        the four energy fields — the Probe 3.5 runner is the sole writer and
        populates all of them on every emission. Records at older versions
        ("0.1.0" / "0.2.0") bypass this check by their version literal, so
        Probe 1 / 1.5 / 3 shards stay backward-readable (the new fields surface
        as None).
        """
        if self.schema_version != PROBE_3_5_TELEMETRY_SCHEMA_VERSION:
            return self
        missing = [
            name
            for name, value in (
                ("self_prediction_t", self.self_prediction_t),
                ("self_prediction_error_t", self.self_prediction_error_t),
                (
                    "self_prediction_error_masked_t",
                    self.self_prediction_error_masked_t,
                ),
                ("sensed_energy_t", self.sensed_energy_t),
                ("true_energy_t", self.true_energy_t),
                ("energy_pred_t", self.energy_pred_t),
                ("energy_recon_error_t", self.energy_recon_error_t),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"AgentStep with schema_version=="
                f"{PROBE_3_5_TELEMETRY_SCHEMA_VERSION!r} requires non-None "
                f"values for {missing}. Probe 3.5 writers populate the "
                f"self-prediction and energy fields on every emission "
                f"(implementation plan §S-TEL)."
            )
        return self

    @model_validator(mode="after")
    def _enforce_probe_3_5_phase2_required_fields(self) -> "AgentStep":
        """Probe 3.5 Phase 2 writer-side discipline (plan §S-TEL).

        Records stamped :data:`PROBE_3_5_PHASE2_TELEMETRY_SCHEMA_VERSION`
        must carry non-None values for the Probe 1.5 self-prediction fields,
        the four Phase-1 energy fields, **and** the three pragmatic
        decomposition fields — the Phase-2 runner is the sole writer and
        populates all of them on every emission. Records at older versions
        bypass this check by their version literal, so Probe 1 / 1.5 / 3 /
        3.5-Phase-1 shards stay backward-readable (the new fields surface as
        None)."""
        if self.schema_version != PROBE_3_5_PHASE2_TELEMETRY_SCHEMA_VERSION:
            return self
        missing = [
            name
            for name, value in (
                ("self_prediction_t", self.self_prediction_t),
                ("self_prediction_error_t", self.self_prediction_error_t),
                (
                    "self_prediction_error_masked_t",
                    self.self_prediction_error_masked_t,
                ),
                ("sensed_energy_t", self.sensed_energy_t),
                ("true_energy_t", self.true_energy_t),
                ("energy_pred_t", self.energy_pred_t),
                ("energy_recon_error_t", self.energy_recon_error_t),
                ("pragmatic_value_t", self.pragmatic_value_t),
                ("epistemic_value_t", self.epistemic_value_t),
                ("pragmatic_share_t", self.pragmatic_share_t),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"AgentStep with schema_version=="
                f"{PROBE_3_5_PHASE2_TELEMETRY_SCHEMA_VERSION!r} requires "
                f"non-None values for {missing}. Probe 3.5 Phase 2 writers "
                f"populate the self-prediction, energy, and pragmatic "
                f"decomposition fields on every emission (plan §S-TEL)."
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

    dream_session_id: str | None = None
    seed_kind: Literal["replay", "perturbed_prior", "hybrid"] | None = None
    seed_replay_segment_id: int | None = None
    seed_replay_step_offset: int | None = None
    seed_perturbation_magnitude: float | None = None
    temperature_schedule: list[float] | None = None
    sub_mode_tags: list[str] | None = None
    sampling_parameters: dict[str, float | int | str | bool] | None = None
    gradient_policy: Literal["none"] | None = None
    rng_seed: int | None = None
    termination_reason: Literal["horizon_complete", "early_terminate_safety"] | None = None
    re_seed_step_indices: list[int] | None = None
    sequence_ensemble_disagreement_variance: list[float] | None = None
    checkpoint_hash: str | None = None

    # Probe 3.5 Phase 3 — the §7 passive-decode monitor: the world model's
    # decoded energy belief at each dream step, recorded alongside
    # ``sequence_decoded_obs``. Observer-side telemetry only — the dream
    # regime stays preference-free and loss-free with respect to this monitor
    # (tests/test_dream_energy_monitor.py proves the rollout is byte-identical
    # apart from this field). Optional so all older shards deserialize
    # cleanly; required non-None at the Phase-3 dream record version.
    sequence_decoded_energy: list[float] | None = None

    @model_validator(mode="after")
    def _enforce_v3_required_fields(self) -> "DreamRollout":
        # Probe 3 is additive at the model level: older Probe 1 / 1.5 records
        # must continue to read through with the new fields surfacing as None.
        # The required-field discipline is therefore writer-side and keyed
        # strictly on the version stamp. The Probe 3.5 Phase 3 version
        # ("0.4.0", §7 monitor on) requires everything the Probe-3 version
        # does **plus** ``sequence_decoded_energy``.
        if self.schema_version not in (
            PROBE_3_TELEMETRY_SCHEMA_VERSION,
            PROBE_3_5_PHASE3_DREAM_SCHEMA_VERSION,
        ):
            return self

        missing = [
            name
            for name, value in (
                ("dream_session_id", self.dream_session_id),
                ("seed_kind", self.seed_kind),
                ("temperature_schedule", self.temperature_schedule),
                ("sampling_parameters", self.sampling_parameters),
                ("gradient_policy", self.gradient_policy),
                ("rng_seed", self.rng_seed),
                ("termination_reason", self.termination_reason),
                (
                    "sequence_ensemble_disagreement_variance",
                    self.sequence_ensemble_disagreement_variance,
                ),
                ("checkpoint_hash", self.checkpoint_hash),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"DreamRollout with schema_version=={self.schema_version!r} "
                f"requires non-None values for {missing}."
            )

        assert self.seed_kind is not None
        conditional_missing: list[str] = []
        if self.seed_kind in {"replay", "hybrid"}:
            if self.seed_replay_segment_id is None:
                conditional_missing.append("seed_replay_segment_id")
            if self.seed_replay_step_offset is None:
                conditional_missing.append("seed_replay_step_offset")
        if self.seed_kind in {"perturbed_prior", "hybrid"}:
            if self.seed_perturbation_magnitude is None:
                conditional_missing.append("seed_perturbation_magnitude")
        if conditional_missing:
            raise ValueError(
                f"DreamRollout with schema_version=={self.schema_version!r} "
                f"and seed_kind=={self.seed_kind!r} requires non-None values for "
                f"{conditional_missing}."
            )

        # Probe 3.5 Phase 3 (§7 monitor): the Phase-3 dream version is stamped
        # only by monitor-on writers, so the monitor field is required there.
        if (
            self.schema_version == PROBE_3_5_PHASE3_DREAM_SCHEMA_VERSION
            and self.sequence_decoded_energy is None
        ):
            raise ValueError(
                f"DreamRollout with schema_version=="
                f"{PROBE_3_5_PHASE3_DREAM_SCHEMA_VERSION!r} requires a non-None "
                f"sequence_decoded_energy (the §7 passive-decode monitor is "
                f"what this version means)."
            )
        return self


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
    "internal_stochasticity_event",
    "energy_fault_event",
    "mirror_marker",
    "state_transition",
    "dormant_heartbeat",
]

# The validator-enforced payload shape of ``"internal_stochasticity_event"``
# records (module docstring, payload conventions): the three comparison keys
# match a builder ``add_resource`` payload; ``process`` is the honest
# discriminator (no mutator fired).
_INTERNAL_STOCHASTICITY_EVENT_PAYLOAD_KEYS: tuple[str, ...] = (
    "process",
    "cell",
    "pre_state",
    "post_state",
)

# The validator-enforced payload shape of ``"energy_fault_event"`` records
# (Probe 4.5 §S-TEL): ``transition`` is which edge fired; ``decay_multiplier``
# pins the fault physics the record attests to, so the observer-side join
# needs no config lookup.
_ENERGY_FAULT_EVENT_PAYLOAD_KEYS: tuple[str, ...] = (
    "transition",
    "decay_multiplier",
)


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

    @model_validator(mode="after")
    def _enforce_probe_4_granular_event(self) -> "WorldEvent":
        """Probe 4 Phase 1 writer-side discipline (plan §S-TEL).

        ``"internal_stochasticity_event"`` records must stamp
        :data:`PROBE_4_WORLD_EVENT_SCHEMA_VERSION` (the mixed-version
        writer rejection the AgentStep validators established) and must
        carry the matched-control payload shape — the per-event
        comparability with builder ``add_resource`` records is what the
        event type *means* (pre-registration §1), so it is enforced, not
        merely documented. All other event types, at every version,
        bypass this check; older shards deserialize unchanged.
        """
        if self.event_type != "internal_stochasticity_event":
            return self
        if self.schema_version != PROBE_4_WORLD_EVENT_SCHEMA_VERSION:
            raise ValueError(
                f"WorldEvent with event_type='internal_stochasticity_event' "
                f"requires schema_version=="
                f"{PROBE_4_WORLD_EVENT_SCHEMA_VERSION!r} (got "
                f"{self.schema_version!r}) — granular internal-stochasticity "
                f"logging is the Probe 4 WorldEvent generation."
            )
        missing = [
            key
            for key in _INTERNAL_STOCHASTICITY_EVENT_PAYLOAD_KEYS
            if key not in self.payload
        ]
        if missing:
            raise ValueError(
                f"internal_stochasticity_event payload is missing {missing}; "
                f"the matched control requires the builder-comparable shape "
                f"{{process, cell, pre_state, post_state}} (pre-registration "
                f"§1)."
            )
        return self

    @model_validator(mode="after")
    def _enforce_probe_4_5_fault_event(self) -> "WorldEvent":
        """Probe 4.5 Phase 2 writer-side discipline (plan §S-TEL).

        ``"energy_fault_event"`` records must stamp
        :data:`PROBE_4_5_WORLD_EVENT_SCHEMA_VERSION` and carry the
        ``{transition, decay_multiplier}`` payload with ``transition`` one of
        ``"onset"`` / ``"offset"`` — the observer-side fault join
        reconstructs per-step fault state from these edges alone (the fault
        is invisible everywhere else by design), so the shape is enforced,
        not merely documented. All other event types bypass this check.
        """
        if self.event_type != "energy_fault_event":
            return self
        if self.schema_version != PROBE_4_5_WORLD_EVENT_SCHEMA_VERSION:
            raise ValueError(
                f"WorldEvent with event_type='energy_fault_event' requires "
                f"schema_version=={PROBE_4_5_WORLD_EVENT_SCHEMA_VERSION!r} "
                f"(got {self.schema_version!r}) — granular fault logging is "
                f"the Probe 4.5 WorldEvent generation."
            )
        missing = [
            key
            for key in _ENERGY_FAULT_EVENT_PAYLOAD_KEYS
            if key not in self.payload
        ]
        if missing:
            raise ValueError(
                f"energy_fault_event payload is missing {missing}; the fault "
                f"join requires {{transition, decay_multiplier}} (plan "
                f"§S-TEL)."
            )
        if self.payload["transition"] not in ("onset", "offset"):
            raise ValueError(
                f"energy_fault_event transition must be 'onset' or 'offset', "
                f"got {self.payload['transition']!r}."
            )
        return self


RECORD_MODELS: tuple[type[RecordEnvelope], ...] = (
    AgentStep,
    DreamRollout,
    ReplayMeta,
    WorldEvent,
)


def _schemas_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "schemas"


def _read_frozen_schema(filename: str) -> bytes:
    return (_schemas_dir() / filename).read_bytes()


def export_json_schema() -> bytes:
    """Return the frozen Probe 1.5 telemetry export (``schemas/v0.2.0.json``).

    Probe 3 extends the live telemetry models in this module. Keeping the v0.2.0
    export byte-stable therefore means treating it as a frozen historical
    artifact rather than regenerating it from the now-wider models.
    """

    return _read_frozen_schema("v0.2.0.json")


PROBE_2_EXPORT_VERSION: str = "0.3.0"
PROBE_3_EXPORT_VERSION: str = "0.4.0"
PROBE_3_5_EXPORT_VERSION: str = "0.5.0"
PROBE_3_5_PHASE2_EXPORT_VERSION: str = "0.6.0"
PROBE_3_5_PHASE3_EXPORT_VERSION: str = "0.7.0"
PROBE_4_EXPORT_VERSION: str = "0.8.0"
PROBE_4_5_EXPORT_VERSION: str = "0.9.0"


def export_json_schema_v0_3_0() -> bytes:
    """Return the frozen Probe 2 multi-family export (``schemas/v0.3.0.json``).

    The Probe 2 export pins the pre-Probe-3 telemetry surface. Reading the
    checked-in bytes keeps that historical artifact mechanically stable while
    Probe 3 widens only the new v0.4.0 export.
    """

    return _read_frozen_schema("v0.3.0.json")


def export_json_schema_v0_4_0() -> bytes:
    """Return the frozen Probe 3 multi-family export (``schemas/v0.4.0.json``).

    Probe 3.5 widens the live telemetry models (AgentStep gains energy fields),
    so — exactly as the v0.2.0 / v0.3.0 exports already do — the Probe 3 export
    is now treated as a **frozen historical artifact**: reading the checked-in
    bytes keeps it byte-stable rather than regenerating it from the now-wider
    models. Probe 3.5 Phase 1's export is the (now likewise frozen)
    :func:`export_json_schema_v0_5_0`; the live export is
    :func:`export_json_schema_v0_6_0`.
    """

    return _read_frozen_schema("v0.4.0.json")


def export_json_schema_v0_5_0() -> bytes:
    """Return the frozen Probe 3.5 Phase-1 export (``schemas/v0.5.0.json``).

    Probe 3.5 Phase 2 widens the live telemetry models (AgentStep gains the
    pragmatic decomposition fields), so — exactly as the v0.2.0 / v0.3.0 /
    v0.4.0 exports already do — the Phase-1 export is now treated as a
    **frozen historical artifact**: reading the checked-in bytes keeps it
    byte-stable rather than regenerating it from the now-wider models. The
    Phase-2 live export is :func:`export_json_schema_v0_6_0`.
    """

    return _read_frozen_schema("v0.5.0.json")


def export_json_schema_v0_6_0() -> bytes:
    """Return the frozen Probe 3.5 Phase-2 export (``schemas/v0.6.0.json``).

    Probe 3.5 Phase 3 widens the live telemetry models (DreamRollout gains
    ``sequence_decoded_energy``, the §7 passive-decode monitor), so — per the
    house pattern — the Phase-2 export is now a **frozen historical
    artifact**: reading the checked-in bytes keeps it byte-stable rather than
    regenerating it from the now-wider models. The Phase-3 live export is
    :func:`export_json_schema_v0_7_0`.
    """

    return _read_frozen_schema("v0.6.0.json")


def export_json_schema_v0_7_0() -> bytes:
    """Return the frozen Probe 3.5 Phase-3 export (``schemas/v0.7.0.json``).

    Probe 4 Phase 1 widens the live telemetry models (``WorldEventType``
    gains ``"internal_stochasticity_event"``), so — per the house pattern —
    the Phase-3 export is now a **frozen historical artifact**: reading the
    checked-in bytes keeps it byte-stable rather than regenerating it from
    the now-wider models. The Probe 4 live export is
    :func:`export_json_schema_v0_8_0`.
    """

    return _read_frozen_schema("v0.7.0.json")


def export_json_schema_v0_8_0() -> bytes:
    """Return the frozen Probe 4 Phase-1 export (``schemas/v0.8.0.json``).

    Probe 4.5 Phase 2 widens the live telemetry models (``WorldEventType``
    gains ``"energy_fault_event"``; the ``WorldEvent`` model gains the fault
    writer-side validator), so — per the house pattern — the Probe 4 export
    is now a **frozen historical artifact**: reading the checked-in bytes
    keeps it byte-stable rather than regenerating it from the now-wider
    models. The Probe 4.5 live export is :func:`export_json_schema_v0_9_0`.
    """

    return _read_frozen_schema("v0.8.0.json")


def export_json_schema_v0_9_0() -> bytes:
    """Return a byte-stable JSON Schema export covering Probe 4.5 Phase 2.

    Builds on the frozen v0.8.0 export and refreshes the telemetry models
    (``WorldEventType`` gains ``"energy_fault_event"``; the ``WorldEvent``
    model gains the fault-event writer-side validator). The mirror-side,
    conditioning, and dream-session models are carried unchanged. The
    export-file version is :data:`PROBE_4_5_EXPORT_VERSION` (``"0.9.0"``);
    the new record-level version it advertises is
    :data:`PROBE_4_5_WORLD_EVENT_SCHEMA_VERSION` (WorldEvent, ``"0.5.0"``);
    the AgentStep / DreamRollout record versions are unchanged from the
    v0.8.0 export.
    """

    from kind.observer.dream_session import DreamSessionMeta

    base_document: dict[str, Any] = json.loads(_read_frozen_schema("v0.8.0.json"))
    base_document["title"] = "Kind Probe 4.5 Phase 2 Schemas"
    base_document["schema_version"] = PROBE_4_5_EXPORT_VERSION
    base_document["world_event_schema_version"] = (
        PROBE_4_5_WORLD_EVENT_SCHEMA_VERSION
    )
    base_document["models"]["telemetry"] = {
        model.__name__: model.model_json_schema() for model in RECORD_MODELS
    }
    base_document["models"]["dream"] = {
        "DreamSessionMeta": DreamSessionMeta.model_json_schema()
    }
    text = json.dumps(base_document, sort_keys=True, indent=2, ensure_ascii=False)
    return (text + "\n").encode("utf-8")


__all__ = [
    "SCHEMA_VERSION",
    "PROBE_1_SCHEMA_VERSION",
    "PROBE_3_TELEMETRY_SCHEMA_VERSION",
    "PROBE_3_5_TELEMETRY_SCHEMA_VERSION",
    "PROBE_3_5_PHASE2_TELEMETRY_SCHEMA_VERSION",
    "PROBE_3_5_PHASE3_DREAM_SCHEMA_VERSION",
    "PROBE_4_WORLD_EVENT_SCHEMA_VERSION",
    "PROBE_4_5_WORLD_EVENT_SCHEMA_VERSION",
    "PROBE_2_EXPORT_VERSION",
    "PROBE_3_EXPORT_VERSION",
    "PROBE_3_5_EXPORT_VERSION",
    "PROBE_3_5_PHASE2_EXPORT_VERSION",
    "PROBE_3_5_PHASE3_EXPORT_VERSION",
    "PROBE_4_EXPORT_VERSION",
    "PROBE_4_5_EXPORT_VERSION",
    "RecordEnvelope",
    "AgentStep",
    "DreamRollout",
    "ReplayMeta",
    "WorldEvent",
    "WorldEventType",
    "RECORD_MODELS",
    "export_json_schema",
    "export_json_schema_v0_3_0",
    "export_json_schema_v0_4_0",
    "export_json_schema_v0_5_0",
    "export_json_schema_v0_6_0",
    "export_json_schema_v0_7_0",
    "export_json_schema_v0_8_0",
    "export_json_schema_v0_9_0",
]
