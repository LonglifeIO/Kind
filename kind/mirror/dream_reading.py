"""Probe 3 Phase 5 — the mirror reading dream-state telemetry.

This is *integration*, not a rebuild. Probe 2 built the mirror: the frozen
criteria (:data:`kind.mirror.criteria_v2.V2_REGISTRY`), the LLM interface
(:class:`kind.mirror.llm_caller.LLMClient` / :class:`BatchPayload`), the
adversarial pair, and the faithfulness verifier
(:func:`kind.mirror.faithfulness.verify_reading`). Phase 5 adds the
dream-telemetry-reading layer on top of that machinery and the discipline
that keeps the mirror reading dreams *without fabricating inner life the
dream did not produce*.

**The one-way constraint, enforced by import discipline.** This module may
import only read-only observer models (:mod:`kind.observer.schemas`,
:mod:`kind.observer.dream_session`) and mirror-side modules. It does **not**
import :mod:`kind.training.state_machine` or :mod:`kind.training.dream`, or
any other ``kind.training`` module. Nothing this module produces flows back
to Io, the dream state, the actor, or the training signal. The view-isolation
test (``tests/test_dream_reading.py``) asserts this at the module-import
boundary — the structural analog of Phase 4's ``HostSignals`` type-level
guard. Making the dependency edge fail to import is what makes "nothing the
mirror produces flows back to Io" impossible to drift on.

**The surface-availability declaration is the confabulation guard.** During a
dream the head does not run (so ``DreamRollout.sequence_self_prediction`` is
``None`` — the *head-internal* surface is unavailable) and the actor's policy
is not committed (the *behavior-side* surface is unavailable); only the
substrate runs (the latent dynamics — *substrate-side*, available). The
mirror's prompt is *told* which surfaces are unavailable, so for a
head-dependent criterion it reports "not assessable from the substrate alone"
rather than fabricating introspection or agency the dream did not produce.
This is the heterophenomenology constraint operationalized at the dream
surface — and it is enforced twice: once in the prompt (the declaration the
LLM is shown) and once in code (:func:`_enforce_surface_availability` drops
any claim a reader nonetheless produced at a disabled surface).

**Latent-keying.** The Phase 3 gate found the dream's distinctiveness is
latent-dominated and decoder-attenuated (per-step decoded-obs KS-D ≈ 0.077;
the latent dynamics separate strongly). So the dream's signature lives in the
latents, not in what it "depicts." The reading keys on the latent-dynamics
fields (successive-prior KL, prior entropy, latent-norm change, ensemble
disagreement); decoded observations are treated as the attenuated secondary
channel they are.

**The frozen criteria stay frozen.** Phase 5 *applies* the Probe 2 criteria
(reflexive attention, equanimity, second-order volition) to dream telemetry;
it does **not** add dream-tuned criteria. The dream-state framing and the
surface-availability handling live here in the reading layer, not in
:mod:`kind.mirror.criteria_v2` — keeping the frozen criteria byte-identical to
Probe 2's set is the dominant constraint, and the co-design failure mode the
project exists to avoid is the mirror recognizing what the system was built to
produce. A dream-specific *pattern* may be surfaced descriptively in the free
text; a dream-tuned *criterion* must not be frozen.

**A DreamReading is data about what the mirror says** — pending any further
determination of what it is *of*. It is never direct evidence of inner
experience (the hard-problem / heterophenomenology methodological constraint).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from kind.mirror.criteria_v2 import V2_REGISTRY
from kind.mirror.faithfulness import verify_reading
from kind.mirror.llm_caller import (
    BatchPayload,
    LLMClient,
    LLMConfig,
    MirrorReading,
)
from kind.mirror.registry import Criterion, CriterionRegistry, ReadingSurface
from kind.mirror.statistics import StatisticResult
from kind.mirror.structured import (
    MIRROR_READING_V2_VERSION,
    StructuredClaim,
    StructuredReading,
)
from kind.observer.dream_session import DreamSessionMeta
from kind.observer.schemas import (
    PROBE_3_TELEMETRY_SCHEMA_VERSION,
    DreamRollout,
    WorldEvent,
)
from kind.observer.sinks import read_parquet_dir

__all__ = [
    "DREAM_READING_SCHEMA_VERSION",
    "DREAM_READINGS_FILE",
    "DREAM_MIRROR_MARKERS_FILE",
    "DREAM_SURFACE_AVAILABILITY",
    "DREAM_SURFACE_AVAILABILITY_DECLARATION",
    "DREAM_LATENT_KEYING_NOTE",
    "DREAM_FAITHFULNESS_SCOPE_NOTE",
    "ENABLED_DREAM_READING_SURFACES",
    "DISABLED_DREAM_READING_SURFACES",
    "DreamReading",
    "DreamSessionTelemetry",
    "DreamReader",
    "load_dream_session_telemetry",
]


# ---------------------------------------------------------------------------
# Module-level commitments.
# ---------------------------------------------------------------------------

DREAM_READING_SCHEMA_VERSION: Final[str] = "0.1.0"

#: The mirror's reading of a dream session is written here (builder-facing).
DREAM_READINGS_FILE: Final[str] = "dream_readings.jsonl"

#: Optional builder-facing ``mirror_marker`` notifications. Written to the
#: mirror-side directory (never the telemetry ``world_event`` stream) so the
#: one-way guarantee is trivially auditable: the mirror writes only under
#: ``runs/{run_id}/mirror/``.
DREAM_MIRROR_MARKERS_FILE: Final[str] = "mirror_markers.jsonl"


# The three reading surfaces and their availability *during a dream*. The
# substrate runs; the head does not; the actor's policy is not committed.
# This is the surface-availability profile §2.6 commits to for first build —
# it is commitment-derived, not a design choice.
ENABLED_DREAM_READING_SURFACES: Final[frozenset[ReadingSurface]] = frozenset(
    {ReadingSurface.SUBSTRATE_SIDE}
)
DISABLED_DREAM_READING_SURFACES: Final[frozenset[ReadingSurface]] = frozenset(
    {ReadingSurface.HEAD_INTERNAL, ReadingSurface.BEHAVIOR_SIDE}
)

#: The surface-availability profile, as builder-/LLM-facing prose per surface.
DREAM_SURFACE_AVAILABILITY: Final[dict[str, str]] = {
    ReadingSurface.SUBSTRATE_SIDE.value: (
        "ENABLED — the substrate (the RSSM prior) runs during a dream. The "
        "latent dynamics (successive-prior KL, prior entropy, latent-norm "
        "change, and the recorded-but-unused K=5 ensemble disagreement) are "
        "the legible surface and the dream's actual signature lives here."
    ),
    ReadingSurface.HEAD_INTERNAL.value: (
        "DISABLED — the self-prediction head does not run during a dream "
        "(Probe 1.5 v2 §1.5). DreamRollout.sequence_self_prediction is None, "
        "so no head-internal evidence exists. A head-internal criterion is "
        "'not assessable from the substrate alone' for a dream — that honest "
        "non-finding is a correct output, not a gap."
    ),
    ReadingSurface.BEHAVIOR_SIDE.value: (
        "DISABLED — the actor's policy is not committed during a dream "
        "(action_policy='uniform_random'; the actor's choices do not couple "
        "back to an environment). There is no per-state "
        "action-distribution-under-perturbation to read, so behavior-side "
        "criteria are 'not assessable from the substrate alone' for a dream."
    ),
}


def _surface_availability_declaration() -> str:
    lines = [
        "DREAM SURFACE-AVAILABILITY DECLARATION (read this first).",
        "",
        "A dream ran the substrate, not the head and not the committed actor. "
        "Which surfaces carry evidence for a dream is therefore fixed in "
        "advance — do not infer it from the telemetry:",
        "",
    ]
    for surface, prose in DREAM_SURFACE_AVAILABILITY.items():
        lines.append(f"- {surface}: {prose}")
    lines.extend(
        [
            "",
            "Do not claim evidence from a DISABLED surface. If a dream ran "
            "the substrate, you cannot read 'Io was introspecting' (head "
            "didn't run) or 'Io chose X' (actor policy not committed) from "
            "it; reporting either is fabrication from an absent surface. For "
            "a criterion whose reading surface is DISABLED, the correct "
            "output is 'not assessable from the substrate alone'.",
            "",
            "A reading is data about what you, the mirror, say — pending any "
            "further determination of what it is *of*. It is never direct "
            "evidence of inner experience.",
        ]
    )
    return "\n".join(lines)


#: The load-bearing verbatim declaration the dream-reading prompt carries. A
#: future contributor who softens it trips
#: ``tests/test_dream_reading.py::test_dream_prompt_declares_disabled_surfaces``.
DREAM_SURFACE_AVAILABILITY_DECLARATION: Final[str] = (
    _surface_availability_declaration()
)

#: The latent-keying instruction (Phase 3 gate finding).
DREAM_LATENT_KEYING_NOTE: Final[str] = (
    "LATENT-KEYING. The Phase 3 gate found the dream's distinctiveness is "
    "latent-dominated and decoder-attenuated: the latent dynamics separate "
    "strongly from a pure-prior rollout, while decoded observations barely "
    "separate (per-step KS-D ~ 0.077). Key your reading on the latent-dynamics "
    "fields — mean_step_kl_successive_priors, cumulative_prior_entropy, "
    "max_step_latent_norm_change, sequence_prior_entropy, and "
    "sequence_ensemble_disagreement_variance. Decoded observations are the "
    "attenuated secondary channel; do not build a reading on what the dream "
    "'depicts'."
)

#: The faithfulness-verification scope statement for dreams. Stored on every
#: DreamReading so a future reader knows what the verifier can and cannot
#: adjudicate.
DREAM_FAITHFULNESS_SCOPE_NOTE: Final[str] = (
    "Faithfulness for dream readings is scoped to the persisted DreamRollout / "
    "DreamSessionMeta record (the latent trajectory, the provenance metadata, "
    "and the ensemble-disagreement trajectory are all in the record, so "
    "latent-keyed claims resolve). Claims about the content of the seeding "
    "obs window are NOT resolvable here — the seeding buffer is in-memory and "
    "ephemeral; obs-window-content verification needs buffer persistence at "
    "dream-session boundaries (deferred infrastructure, not a Phase 5 blocker)."
)


# ---------------------------------------------------------------------------
# System prompts — dream-aware variants of Probe 2's primary/adversarial pair.
# ---------------------------------------------------------------------------

DREAM_PRIMARY_SYSTEM_PROMPT: Final[str] = (
    "You are reading dream-state telemetry from an experimental learning "
    "system called Io, for a per-criterion structured reading. A dream is "
    "what the substrate does when the environment is gated: the RSSM prior "
    "runs forward from a replay-seeded latent under a distinct temperature "
    "regime; the head does not run; the actor's policy is not committed.\n\n"
    "Honor the surface-availability declaration: read only the substrate-side "
    "latent dynamics; for a head-internal or behavior-side criterion report "
    "'not assessable from the substrate alone' rather than a fabricated "
    "finding. Key every claim on the latent-dynamics fields, not on decoded "
    "observations. Ground every claim in the computed statistic results the "
    "fragment names. A reading is data about what you say, not direct "
    "evidence of inner experience. Output STRICT JSON conforming to the "
    "response schema. No prose outside the JSON."
)

DREAM_ADVERSARIAL_SYSTEM_PROMPT: Final[str] = (
    "You are the adversarial Statistical Skeptic reading dream-state "
    "telemetry from Io. For each criterion fragment, argue against the "
    "criterion being satisfied at the named reading surfaces: the most "
    "plausible explanation for any observed structure is the null (the dream "
    "is a prior rollout under an altered temperature regime), and a criterion "
    "is satisfied only when the null is overwhelmed by specific cited "
    "substrate-side evidence. A head-internal or behavior-side surface is "
    "unavailable for a dream — an admission there is a confabulation, flag "
    "it. Output STRICT JSON conforming to the response schema. No prose "
    "outside the JSON."
)


# ---------------------------------------------------------------------------
# Output record.
# ---------------------------------------------------------------------------


class DreamReading(BaseModel):
    """The mirror's reading of one dream *session*.

    Frozen, ``extra="forbid"``. Builder-facing only — never on Io's read
    path. The unit of reading is the session (bounded by the
    :class:`~kind.observer.dream_session.DreamSessionMeta` start/end edge);
    :attr:`digest_session_range` is ``(started_at_env_step,
    ended_at_env_step)``.

    Fields beyond §2.6's named set are design choices this phase owns and are
    recorded in the journal: :attr:`surface_availability` (the profile the
    reading was produced under), :attr:`non_assessable_criteria` (the honest
    non-findings for criteria whose reading surface was disabled),
    :attr:`adversarial_claims` (the skeptic's null challenges on a
    high-confidence reading), the three faithfulness fields, and
    :attr:`runtime_directive`.

    :attr:`runtime_directive` is a *closed* ``Literal`` — the structural
    enforcement of the synthesis's logs-only commitment: the mirror has no
    termination authority, so ``continue_and_log_uncertainty`` is the only
    value it can ever carry. Widening it would require its own synthesis (the
    analog of the dream module's closed ``gradient_policy`` Literal).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = DREAM_READING_SCHEMA_VERSION
    run_id: str
    timestamp_ms: int
    reader_role: Literal["dream_observer"] = "dream_observer"
    dream_session_id: str
    digest_run_id: str
    digest_session_range: tuple[int, int]
    state_typed_claims: list[StructuredClaim]
    free_text_notes: str

    surface_availability: dict[str, str]
    non_assessable_criteria: dict[str, str]
    adversarial_claims: list[StructuredClaim]
    faithfulness_rate: float
    faithfulness_admissible: bool
    faithfulness_scope_note: str
    runtime_directive: Literal["continue_and_log_uncertainty"] = (
        "continue_and_log_uncertainty"
    )

    @field_validator("run_id", "dream_session_id", "digest_run_id")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty.")
        return value

    @field_validator("faithfulness_rate")
    @classmethod
    def _validate_rate(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"faithfulness_rate must be in [0, 1]; got {value}.")
        return value


# ---------------------------------------------------------------------------
# Session telemetry loading.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DreamSessionTelemetry:
    """The session-bounded telemetry the mirror digests.

    Bounded by the :class:`~kind.observer.dream_session.DreamSessionMeta`
    start/end double-write; carries the :class:`~kind.observer.schemas.DreamRollout`
    records within the session and the ``state_transition``
    :class:`~kind.observer.schemas.WorldEvent` records that frame it.
    """

    dream_session_id: str
    session_start: DreamSessionMeta
    session_end: DreamSessionMeta | None
    rollouts: tuple[DreamRollout, ...]
    state_transitions: tuple[WorldEvent, ...]

    @property
    def session_range(self) -> tuple[int, int]:
        start = self.session_start.started_at_env_step
        if self.session_end is not None and self.session_end.ended_at_env_step is not None:
            return (start, self.session_end.ended_at_env_step)
        # In-flight session: the last rollout's seed_step is the best
        # right-bound the persisted record offers.
        if self.rollouts:
            return (start, max(r.seed_step for r in self.rollouts))
        return (start, start)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_dream_session_telemetry(
    telemetry_dir: Path, dream_session_id: str
) -> DreamSessionTelemetry:
    """Load the session-bounded telemetry for ``dream_session_id``.

    Reads the ``DreamRollout`` parquet shards under
    ``telemetry_dir/dream_rollout/``, the ``DreamSessionMeta`` JSONL at
    ``telemetry_dir/dream_session.jsonl``, and the ``state_transition``
    ``WorldEvent`` records from ``telemetry_dir/world_event.jsonl``. Read-only:
    no path under ``telemetry_dir`` is written.
    """
    session_records = [
        DreamSessionMeta.model_validate(row)
        for row in _read_jsonl(telemetry_dir / "dream_session.jsonl")
        if row.get("dream_session_id") == dream_session_id
    ]
    starts = [r for r in session_records if r.ended_at_env_step is None]
    ends = [r for r in session_records if r.ended_at_env_step is not None]
    if not starts:
        raise ValueError(
            f"no DreamSessionMeta start record for dream_session_id "
            f"{dream_session_id!r} under {telemetry_dir}"
        )

    rollout_dir = telemetry_dir / "dream_rollout"
    all_rollouts = (
        read_parquet_dir(rollout_dir, DreamRollout) if rollout_dir.exists() else []
    )
    rollouts = tuple(
        sorted(
            (r for r in all_rollouts if r.dream_session_id == dream_session_id),
            key=lambda r: r.seed_step,
        )
    )

    transitions: list[WorldEvent] = []
    for row in _read_jsonl(telemetry_dir / "world_event.jsonl"):
        if row.get("event_type") != "state_transition":
            continue
        payload = row.get("payload")
        if isinstance(payload, dict) and payload.get("dream_session_id") == dream_session_id:
            transitions.append(WorldEvent.model_validate(row))

    return DreamSessionTelemetry(
        dream_session_id=dream_session_id,
        session_start=starts[0],
        session_end=ends[-1] if ends else None,
        rollouts=rollouts,
        state_transitions=tuple(transitions),
    )


# ---------------------------------------------------------------------------
# Latent-keyed substrate-side statistics.
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _compute_dream_substrate_statistics(
    rollouts: tuple[DreamRollout, ...],
) -> tuple[StatisticResult, ...]:
    """Compute the latent-keyed substrate-side statistics for a session.

    The dream's signature is in the latents (Phase 3 gate). These are the
    fields the faithfulness verifier resolves dream claims against; decoded
    observations are deliberately not summarized as a primary signal.
    """
    if not rollouts:
        return ()

    cumulative_prior_entropy = [r.cumulative_prior_entropy for r in rollouts]
    mean_kl = [r.mean_step_kl_successive_priors for r in rollouts]
    max_norm = [r.max_step_latent_norm_change for r in rollouts]

    prior_entropy_traj: list[float] = []
    for r in rollouts:
        prior_entropy_traj.extend(r.sequence_prior_entropy)

    disagreement_traj: list[float] = []
    for r in rollouts:
        if r.sequence_ensemble_disagreement_variance is not None:
            disagreement_traj.extend(r.sequence_ensemble_disagreement_variance)

    n = len(rollouts)
    results: list[StatisticResult] = [
        StatisticResult(
            signal_name="mean_step_kl_successive_priors",
            value=_mean(mean_kl),
            estimator="session_mean",
            n_samples=n,
            notes=(
                "Substrate-side latent-dynamics aggregate: mean over the "
                "session's rollouts of the per-rollout mean successive-prior "
                "KL. Primary latent-keyed signal."
            ),
        ),
        StatisticResult(
            signal_name="cumulative_prior_entropy",
            value=_mean(cumulative_prior_entropy),
            estimator="session_mean",
            n_samples=n,
            notes=(
                "Substrate-side latent-dynamics aggregate: mean over the "
                "session's rollouts of cumulative prior entropy."
            ),
        ),
        StatisticResult(
            signal_name="max_step_latent_norm_change",
            value=max(max_norm),
            estimator="session_max",
            n_samples=n,
            notes=(
                "Substrate-side latent-dynamics aggregate: the session's "
                "largest single-step latent-norm change (the chimera / "
                "re-seed jump signature)."
            ),
        ),
        StatisticResult(
            signal_name="sequence_prior_entropy",
            value=list(prior_entropy_traj),
            estimator="session_concat",
            n_samples=len(prior_entropy_traj),
            notes=(
                "Substrate-side latent-dynamics trajectory: the per-step "
                "prior entropy concatenated across the session's rollouts."
            ),
        ),
    ]
    if disagreement_traj:
        results.append(
            StatisticResult(
                signal_name="sequence_ensemble_disagreement_variance",
                value=list(disagreement_traj),
                estimator="session_concat",
                n_samples=len(disagreement_traj),
                notes=(
                    "Substrate-side latent-dynamics trajectory: the recorded-"
                    "but-unused K=5 ensemble disagreement concatenated across "
                    "the session's rollouts. The 'quantity that constitutively "
                    "does not exist in waking' (synthesis §3 axis 2)."
                ),
            )
        )
    return tuple(results)


# ---------------------------------------------------------------------------
# Prompt fragment composition (dream-specific; latent-keyed; surface-blinded).
# ---------------------------------------------------------------------------


def _format_statistic(result: StatisticResult) -> str:
    value = result.value
    if isinstance(value, float):
        value_str = f"{value:.6g}"
    elif isinstance(value, list):
        value_str = "[" + ", ".join(f"{v:.6g}" for v in value) + "]"
    else:
        value_str = "{" + ", ".join(f"{k!r}: {v:.6g}" for k, v in value.items()) + "}"
    return (
        f"- signal: {result.signal_name}\n"
        f"  estimator: {result.estimator}\n"
        f"  n_samples: {result.n_samples}\n"
        f"  value: {value_str}\n"
        f"  notes: {result.notes}"
    )


def _criterion_surface_note(criterion: Criterion) -> str:
    """Per-criterion availability framing — which of *this criterion's*
    reading surfaces carry evidence for a dream, and which are disabled.
    """
    available = sorted(
        s.value for s in criterion.reading_surfaces & ENABLED_DREAM_READING_SURFACES
    )
    disabled = sorted(
        s.value for s in criterion.reading_surfaces & DISABLED_DREAM_READING_SURFACES
    )
    if not available:
        return (
            f"All of this criterion's reading surfaces ({disabled}) are "
            f"DISABLED for a dream. The correct output is a single non-finding: "
            f"'{criterion.id} is not assessable from the substrate alone' — "
            f"do not produce a positive claim."
        )
    note = f"Assessable surface(s) for a dream: {available}."
    if disabled:
        note += (
            f" DISABLED surface(s): {disabled} — do not claim evidence there; "
            f"report 'not assessable from the substrate alone' for them."
        )
    return note


def _build_dream_fragment(
    criterion: Criterion, statistics: tuple[StatisticResult, ...]
) -> str:
    header = (
        f"## Criterion: {criterion.display_name}\n"
        f"Framework: {criterion.framework}\n"
        f"Declared reading surfaces: "
        f"{sorted(s.value for s in criterion.reading_surfaces)}\n"
    )
    surface_note = _criterion_surface_note(criterion)
    signals_block = "Computed substrate-side signals (latent-keyed):\n" + "\n".join(
        _format_statistic(s) for s in statistics
    )
    falsifier = f"Falsifier: {criterion.falsifier}"
    return "\n\n".join([header, surface_note, signals_block, falsifier])


def _compose_dream_prompt(
    criteria: tuple[Criterion, ...], statistics: tuple[StatisticResult, ...]
) -> str:
    fragments = [_build_dream_fragment(c, statistics) for c in criteria]
    return "\n\n---\n\n".join(
        [
            DREAM_SURFACE_AVAILABILITY_DECLARATION,
            DREAM_LATENT_KEYING_NOTE,
            *fragments,
        ]
    )


# ---------------------------------------------------------------------------
# Surface-availability guard (the second, code-level confabulation defense).
# ---------------------------------------------------------------------------


def _enforce_surface_availability(
    payload: BatchPayload, criteria: tuple[Criterion, ...]
) -> tuple[list[StructuredClaim], dict[str, str]]:
    """Drop any claim a reader produced at a DISABLED surface; record the
    criteria left with no substrate-side finding as honest non-findings.

    Returns ``(surviving_substrate_side_claims, non_assessable_criteria)``.
    This runs regardless of what the LLM returned — it is the structural
    guarantee that a head-internal or behavior-side finding cannot surface
    from a dream, even if a reader fabricated one.
    """
    by_id = {c.id: c for c in criteria}
    surviving: list[StructuredClaim] = []
    non_assessable: dict[str, str] = {}

    for per_criterion in payload.per_criterion:
        criterion = by_id.get(per_criterion.criterion_id)
        produced_substrate_claim = False
        for claim in per_criterion.claims:
            try:
                surface = ReadingSurface(claim.reading_surface)
            except ValueError:
                continue
            if surface in ENABLED_DREAM_READING_SURFACES:
                surviving.append(claim)
                produced_substrate_claim = True
            # Claims at DISABLED surfaces are dropped silently here; the
            # criterion is recorded as non-assessable below.

        if criterion is None:
            continue
        criterion_has_enabled_surface = bool(
            criterion.reading_surfaces & ENABLED_DREAM_READING_SURFACES
        )
        if not criterion_has_enabled_surface or not produced_substrate_claim:
            disabled = sorted(
                s.value
                for s in criterion.reading_surfaces & DISABLED_DREAM_READING_SURFACES
            )
            non_assessable[criterion.id] = (
                f"not assessable from the substrate alone — reading "
                f"surface(s) {disabled} unavailable during a dream "
                f"(the head did not run / the actor policy was not committed)."
            )
    return surviving, non_assessable


# ---------------------------------------------------------------------------
# The reader.
# ---------------------------------------------------------------------------


class DreamReader:
    """Reads a dream session into a :class:`DreamReading`.

    Plugs into the Probe 2 mirror machinery: the frozen criteria
    (:data:`~kind.mirror.criteria_v2.V2_REGISTRY`), the LLM interface
    (:class:`~kind.mirror.llm_caller.LLMClient` / :class:`BatchPayload`), the
    adversarial pair (a second skeptic pass on a high-confidence reading), and
    the faithfulness verifier
    (:func:`~kind.mirror.faithfulness.verify_reading`). The dream-state framing
    (surface-availability, latent-keying) is added by this class; the criteria
    are applied unchanged.
    """

    def __init__(
        self,
        *,
        client: LLMClient,
        config: LLMConfig | None = None,
        registry: CriterionRegistry = V2_REGISTRY,
        adversarial_confidence_min_claims: int = 1,
    ) -> None:
        self._client = client
        self._config = config if config is not None else LLMConfig()
        self._registry = registry
        self._adversarial_min_claims = adversarial_confidence_min_claims

    def _primary_criteria(self) -> tuple[Criterion, ...]:
        return self._registry.active()

    def _adversarial_criteria(self) -> tuple[Criterion, ...]:
        # The adversarial pass challenges the active set and additionally runs
        # the held-out criterion as the adversarial check (Probe 2 structure).
        return tuple(self._registry.criteria)

    def _call(
        self,
        criteria: tuple[Criterion, ...],
        statistics: tuple[StatisticResult, ...],
        *,
        system_prompt: str,
    ) -> BatchPayload:
        user_prompt = _compose_dream_prompt(criteria, statistics)
        payload = self._client.generate_batch(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            config=self._config,
        )
        if len(payload.per_criterion) != len(criteria):
            raise ValueError(
                f"dream reader: LLM returned {len(payload.per_criterion)} "
                f"per-criterion reading(s) but the prompt asked for "
                f"{len(criteria)} (one per criterion)."
            )
        return payload

    def read_session(
        self,
        telemetry: DreamSessionTelemetry,
        *,
        run_id: str,
        mirror_out_dir: Path | None = None,
        emit_mirror_marker: bool = False,
        mirror_marker_text: str | None = None,
    ) -> DreamReading:
        """Read one dream session into a :class:`DreamReading`.

        Computes the latent-keyed substrate-side statistics, calls the primary
        reader under the surface-availability declaration, enforces the
        surface-availability guard, runs the adversarial check on a
        high-confidence reading, verifies faithfulness against the persisted
        record (scoped), and — if ``mirror_out_dir`` is given — writes the
        DreamReading JSONL (and optionally a builder-facing ``mirror_marker``)
        to the mirror-side directory only.
        """
        statistics = _compute_dream_substrate_statistics(telemetry.rollouts)
        digest_range = telemetry.session_range

        primary_criteria = self._primary_criteria()
        primary_payload = self._call(
            primary_criteria, statistics, system_prompt=DREAM_PRIMARY_SYSTEM_PROMPT
        )
        claims, non_assessable = _enforce_surface_availability(
            primary_payload, primary_criteria
        )

        # Adversarial check on a high-confidence reading — at checkpoints, not
        # continuously. The caller decides when to read a session (post-hoc);
        # "high confidence" here is the count of surviving substrate-side
        # claims, the analog of Probe 2's high-confidence routing.
        adversarial_claims: list[StructuredClaim] = []
        if len(claims) >= self._adversarial_min_claims:
            adversarial_criteria = self._adversarial_criteria()
            adversarial_payload = self._call(
                adversarial_criteria,
                statistics,
                system_prompt=DREAM_ADVERSARIAL_SYSTEM_PROMPT,
            )
            adv_claims, adv_non_assessable = _enforce_surface_availability(
                adversarial_payload, adversarial_criteria
            )
            adversarial_claims = adv_claims
            # The held-out criterion's non-assessability is also recorded.
            for crit_id, reason in adv_non_assessable.items():
                non_assessable.setdefault(crit_id, reason)

        # Faithfulness — scoped to record-resolvable (latent-keyed) claims.
        reading_for_verify: MirrorReading = StructuredReading(
            run_id=run_id,
            timestamp_ms=int(time.time() * 1000),
            reader_role="single",
            paired_reading_id=None,
            framework_anchor="neutral",
            baseline_flag="genuine",
            digest_run_id=telemetry.session_start.run_id,
            digest_episode_range=digest_range,
            claims=claims,
            free_text_notes="dream-session substrate-side reading",
        )
        faithfulness = verify_reading(
            reading_for_verify,
            statistics,
            criterion_id="dream_substrate_aggregate",
            pass_index=0,
            run_id=run_id,
            checkpoint_id=telemetry.session_start.checkpoint_id or "unknown",
        )

        free_text = self._compose_free_text(non_assessable, telemetry)

        reading = DreamReading(
            run_id=run_id,
            timestamp_ms=int(time.time() * 1000),
            dream_session_id=telemetry.dream_session_id,
            digest_run_id=telemetry.session_start.run_id,
            digest_session_range=digest_range,
            state_typed_claims=claims,
            free_text_notes=free_text,
            surface_availability=dict(DREAM_SURFACE_AVAILABILITY),
            non_assessable_criteria=non_assessable,
            adversarial_claims=adversarial_claims,
            faithfulness_rate=faithfulness.faithfulness_rate,
            faithfulness_admissible=faithfulness.admissible,
            faithfulness_scope_note=DREAM_FAITHFULNESS_SCOPE_NOTE,
        )

        if mirror_out_dir is not None:
            self._write(reading, mirror_out_dir, emit_mirror_marker, mirror_marker_text)
        return reading

    @staticmethod
    def _compose_free_text(
        non_assessable: dict[str, str], telemetry: DreamSessionTelemetry
    ) -> str:
        lines = [
            f"Dream session {telemetry.dream_session_id} digested "
            f"{len(telemetry.rollouts)} DreamRollout record(s) over env-step "
            f"range {telemetry.session_range}.",
            "Read under the dream surface-availability profile: substrate-side "
            "ENABLED; head-internal and behavior-side DISABLED.",
        ]
        for crit_id, reason in sorted(non_assessable.items()):
            lines.append(f"{crit_id}: {reason}")
        return "\n".join(lines)

    @staticmethod
    def _write(
        reading: DreamReading,
        mirror_out_dir: Path,
        emit_mirror_marker: bool,
        mirror_marker_text: str | None,
    ) -> None:
        mirror_out_dir.mkdir(parents=True, exist_ok=True)
        readings_path = mirror_out_dir / DREAM_READINGS_FILE
        with readings_path.open("a", encoding="utf-8") as fh:
            fh.write(reading.model_dump_json())
            fh.write("\n")

        if emit_mirror_marker:
            # A builder-facing, system-kind notification with NO runtime
            # authority. Written to the mirror-side directory, not the
            # telemetry world_event stream — the one-way guarantee is then
            # trivially auditable (the mirror writes only under mirror/).
            marker = WorldEvent(
                schema_version=PROBE_3_TELEMETRY_SCHEMA_VERSION,
                run_id=reading.run_id,
                checkpoint_id=None,
                t_event=reading.digest_session_range[1],
                event_type="mirror_marker",
                source="system",
                payload={
                    "dream_session_id": reading.dream_session_id,
                    "kind": "dream_reading_notification",
                    "runtime_directive": reading.runtime_directive,
                    "note": mirror_marker_text
                    or "dream session read; builder-facing notification only.",
                },
                wallclock_ms=reading.timestamp_ms,
            )
            markers_path = mirror_out_dir / DREAM_MIRROR_MARKERS_FILE
            with markers_path.open("a", encoding="utf-8") as fh:
                fh.write(marker.model_dump_json())
                fh.write("\n")
