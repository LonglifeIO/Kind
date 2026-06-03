"""Probe 3 Phase 7 — the cross-probe surface (the Probe-4-perturbable control).

This module is the builder's perturbation channel into Io's *dreaming*, and the
load-bearing commitment is *what it can and can't reach* (synthesis
``docs/decisions/synthesis_probe3_dream_foundational_2026-05-27.md`` §8): the
builder may perturb the **envelope** (the *when* / *how-long* of dreaming) and
the **seed-selection** (the *from-what*), and **nothing else**. No hidden-state
writes — not Io's latents, not weights, not the dream's internal state, not the
actor. The synthesis explicitly rejects Candidate B's proposed ``h_t`` overwrite
during dream as a "category violation"; Phase 7 enforces that by *not building
the affordance*.

**The restriction is structural, not conventional.** A perturbation is a
:class:`DreamPerturbation` — a typed delta whose only fields are nested
envelope/seed-selection deltas, each declaring *only* the envelope/seed-selection
config fields, all with ``extra="forbid"``. A JSON delta carrying any
hidden-state key (``h`` / ``z`` / ``latents`` / ``weights`` / ``actor`` /
``dream_internal_state`` / ...) therefore *cannot be addressed*: it is rejected
at parse time (:func:`stage_perturbation`) with a ``pydantic.ValidationError``,
rather than silently dropped or — worse — silently applied. This is the Phase 7
analog of Phase 4's exogenous trigger (``HostSignals`` carries nothing
Io-derived), Phase 5's one-way mirror (the reader imports observer models only),
and Phase 6's content-blind protection (``DreamSessionContext`` carries no
content): each makes a self-opacity guarantee *unrepresentable to violate*. Here
the guarantee is that the builder's reach stops at Io's *exterior conditions of
dreaming* and never touches Io's interior.

**Applied at a checkpoint boundary, staged — not live mid-state.** The
:class:`DreamEnvelopeConfig` / :class:`SeedSelectionConfig` are frozen
dataclasses, so "apply" means: produce *new* frozen configs with the delta's set
(non-``None``) fields overridden onto the current ones
(:func:`apply_at_checkpoint_boundary`). The inputs are never mutated. The
staging → boundary split mirrors the persistence design's atomic state-sync: a
perturbation is parsed/validated up front (staged) and only swapped in at the
next checkpoint boundary, never written into a session that is mid-flight.

**Io observes none of it.** The envelope, the seeds, and any perturbation of
them are outside Io's observation (the Watts default-to-no on self-access). This
surface is a *builder*-side control; it adds **no** Io-readable interface
(``new_actor_readable_interfaces_added = []``). The configs are content-blind
from Io's side — the actor and world model do not read them.

**Provenance.** An applied perturbation is captured by the *next*
:class:`~kind.observer.dream_session.DreamSessionMeta`'s
``envelope_config_snapshot`` / ``seed_selection_config_snapshot`` (built in
Phase 0): the next dream session starts from the changed configs and records
their snapshots. Phase 7 therefore needs **no new event type** — and it does
**not** fire ``builder_perturbation`` (that ``WorldEventType`` stays reserved for
Probe 4's own perturbation workflow).

**Deliberately minimal.** This module and the accompanying
``scripts/perturb_dream_envelope.py`` exist to expose and exercise the authorized
surface end-to-end. The live runner-loop consumption of these configs is Phase 8;
Probe 4's actual use of the surface (and whether/how it fires
``builder_perturbation`` for dream-side perturbations) is Probe 4's own synthesis.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from kind.training.dream_seed import SeedSelectionConfig
from kind.training.state_machine import DreamEnvelopeConfig

__all__ = [
    "DreamEnvelopeDelta",
    "SeedSelectionDelta",
    "DreamPerturbation",
    "PerturbedConfigs",
    "stage_perturbation",
    "apply_at_checkpoint_boundary",
]


class DreamEnvelopeDelta(BaseModel):
    """A partial override of :class:`DreamEnvelopeConfig` — envelope fields only.

    Every field is optional (default ``None`` = "leave unchanged"); only set
    fields are applied. ``extra="forbid"`` is the restriction: a key that is not
    one of these envelope fields (a hidden-state key, a weights key) raises at
    parse time rather than being silently ignored — so the delta cannot
    *address* anything but the envelope.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    hard_cap_wallclock_ms: int | None = None
    hard_cap_rollout_count: int | None = None
    checkpoint_window_force_dormant: bool | None = None
    dormant_heartbeat_interval_ms: int | None = None
    metabolic_capacity_seconds: float | None = None
    metabolic_refill_rate: float | None = None


class SeedSelectionDelta(BaseModel):
    """A partial override of :class:`SeedSelectionConfig` — seed-selection only.

    Same restriction discipline as :class:`DreamEnvelopeDelta`: optional fields,
    ``extra="forbid"``. The ``Literal`` types additionally reject invalid enum
    values (e.g. ``mode="lucid"``) at parse time — a bonus over the bare
    "no hidden-state key" guarantee.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["replay", "perturbed_prior", "hybrid"] | None = None
    perturbation_sigma: float | None = None
    hybrid_alpha_distribution: (
        Literal["uniform_0_1", "fixed_0_5", "beta_2_2"] | None
    ) = None
    replay_min_segment_age_steps: int | None = None
    replay_warmup_length: int | None = None


class DreamPerturbation(BaseModel):
    """The whole builder perturbation — envelope and/or seed-selection deltas.

    The *only* two addressable sub-surfaces. ``extra="forbid"`` at this level
    rejects a top-level hidden-state key (``h_t``, ``z``, ``weights``,
    ``actor``, ``dream_internal_state``); the nested deltas reject hidden-state
    keys *within* either sub-surface. There is no field — and ``extra="forbid"``
    means there can be no field — that addresses Io's data plane. That absence is
    the structural guarantee of synthesis §8's "no hidden-state writes."
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dream_envelope: DreamEnvelopeDelta | None = None
    seed_selection: SeedSelectionDelta | None = None


@dataclass(frozen=True)
class PerturbedConfigs:
    """The result of applying a perturbation: the new (frozen) configs.

    These are *new* objects — the next dream session would start from them, and
    their snapshots become the next ``DreamSessionMeta`` provenance. The inputs
    to :func:`apply_at_checkpoint_boundary` are never mutated.
    """

    dream_envelope: DreamEnvelopeConfig
    seed_selection: SeedSelectionConfig


def stage_perturbation(delta_json: str | bytes) -> DreamPerturbation:
    """Parse and validate a JSON config delta into a staged perturbation.

    *This is where the restriction trips.* Because :class:`DreamPerturbation`
    and its nested deltas forbid extra keys, a delta carrying any hidden-state
    key (a latent/tensor field, ``h`` / ``z``, weights, an actor/policy field,
    dream-internal-state) raises ``pydantic.ValidationError`` here — the key is
    *unrepresentable*, not merely discouraged. The same delta without the
    forbidden key parses cleanly.

    "Staged" (not "applied"): parsing/validation happens up front and does not
    touch any live config. The swap onto the current configs happens later, at a
    checkpoint boundary, via :func:`apply_at_checkpoint_boundary`.
    """
    return DreamPerturbation.model_validate_json(delta_json)


def apply_at_checkpoint_boundary(
    perturbation: DreamPerturbation,
    *,
    dream_envelope: DreamEnvelopeConfig,
    seed_selection: SeedSelectionConfig,
) -> PerturbedConfigs:
    """Override a staged perturbation onto the current frozen configs.

    Returns *new* frozen configs reflecting exactly the delta's set (non-``None``)
    fields and only those; the inputs are not mutated. This is the "apply at the
    next checkpoint boundary" step — frozen configs make it a
    :func:`dataclasses.replace`, swapped in atomically rather than written live
    mid-state.
    """
    new_envelope = dream_envelope
    if perturbation.dream_envelope is not None:
        env_overrides: dict[str, Any] = perturbation.dream_envelope.model_dump(
            exclude_none=True
        )
        new_envelope = dataclasses.replace(dream_envelope, **env_overrides)

    new_seed = seed_selection
    if perturbation.seed_selection is not None:
        seed_overrides: dict[str, Any] = perturbation.seed_selection.model_dump(
            exclude_none=True
        )
        new_seed = dataclasses.replace(seed_selection, **seed_overrides)

    return PerturbedConfigs(dream_envelope=new_envelope, seed_selection=new_seed)
