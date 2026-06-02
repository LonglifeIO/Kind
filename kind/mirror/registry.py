"""Probe 2 mirror-side criterion registry.

Phase 6 built the typed module that holds frozen mirror criteria as
queryable records: an empty shelf with strong invariants. Phase 7 puts
the three v2 criteria on the shelf — they live in
:mod:`kind.mirror.criteria_v2` (reflexive attention, equanimity
perturbation recovery, second-order volition) — and, in the same phase,
refactors :class:`Criterion`'s ``signal_mappings`` from a raw
``dict[str, str]`` into a typed :class:`SignalMapping` model whose
field-path references validate at registry-load against the actual
telemetry schemas. Phase 8's adversarial-pass orchestrator reads the
populated registry and emits per-surface prompt fragments.

**Load-bearing invariant: the mirror is one-way.** Io reads its own state
through PolicyView; the mirror reads everything through TelemetryView;
nothing the mirror produces flows back to Io, the dream state, the actor,
the world model, or the training signal. The registry is part of the
mirror's data plane and inherits this asymmetry.

Concretely, this module enforces the asymmetry structurally rather than by
convention:

- :class:`Criterion` names the telemetry surfaces it reads from
  (``telemetry_surfaces``) and the signals it derives
  (``signal_mappings``). It does **not** name write surfaces. There is no
  write-surface field on the type, and no :class:`SignalMapping` field is a
  writable handle, callback, sink, or anything that could be invoked
  against Io's input space. A future contributor who tries to add one
  trips :func:`tests.test_criterion_registry.test_criterion_does_not_have_write_surface_field`
  or :func:`tests.test_signal_mapping.test_signal_mapping_no_writer_shape`.
- :class:`CriterionRegistry`'s lookup methods return frozen records or
  simple containers (``frozenset``, ``tuple``, ``bool``). No method
  returns a writer, a callback, or anything that could be invoked against
  Io's input space. The structural assertion is in
  :func:`tests.test_criterion_registry.test_registry_no_method_returns_writer`.
- The registry's outputs are consumed by mirror-side code (Phase 8) that
  writes to mirror-side disk paths. The registry's shape makes the wrong
  direction structurally impossible.

**The asymmetry-of-access boundary, enforced at registry load.** A
:class:`SignalMapping` names a path into Io's emitted telemetry — a slice
of ``(h_t, z_t)`` on :class:`~kind.observer.schemas.AgentStep`, a field on
:class:`~kind.observer.schemas.DreamRollout`, etc. — and the path is
validated against the actual Pydantic model for the named surface at
*construction* time, not at prompt-build time. A typo or stale reference
trips at module import. The ``agent_step_observable`` vs
``agent_step_internal`` split is enforced by the ``_OBSERVABLE_FIELDS`` /
``_INTERNAL_FIELDS`` allowlists below: a mapping declaring
``telemetry_surface=AGENT_STEP_OBSERVABLE`` may only root its
``field_path`` in a channel Io's PolicyView also reads; a mapping
declaring ``telemetry_surface=AGENT_STEP_INTERNAL`` may only root it in
substrate state the mirror's TelemetryView reads. This is the
asymmetry-of-access boundary the design notes name as load-bearing
(``Kind_design_notes.md`` — the agent/mirror/observer layers section,
"asymmetry of access between Io and the mirror"); it is here a
registry-load invariant rather than a convention.

**The empty-registry exception.** Production registries (Phase 7's
:data:`kind.mirror.criteria_v2.V2_REGISTRY`) must have at least one
criterion; constructing :class:`CriterionRegistry` with ``criteria=()``
directly raises a validation error. The single sanctioned empty form is
the module-level :data:`EMPTY_REGISTRY` constant, which is constructed via
Pydantic's :meth:`~pydantic.BaseModel.model_construct` to bypass the
non-empty check at module load time. Tests that need an empty registry
import :data:`EMPTY_REGISTRY` rather than constructing one.

**Relationship to Phase 0's Literal.** :class:`ReadingSurface` here is a
``str``-valued :class:`enum.Enum`; Phase 0 defined an equivalent
``Literal["substrate_side", "head_internal", "behavior_side"]`` at
:data:`kind.mirror.structured.ReadingSurface`. As of Phase 7,
:class:`~kind.observer.pre_reg.PreRegistration`'s
``expected_outcome_per_surface`` and ``reading_surfaces_per_criterion``
keys use *this* enum; :mod:`kind.mirror.structured`'s ``StructuredClaim``
/ ``StructuredReading`` / ``JudgeRuling`` still use the Phase 0 Literal
(the enum's *values* match the Literal's strings exactly, so the two
interoperate at the string level and JSON serialization is unchanged).

Out of scope at Phase 7 (still later):

- The adversarial-pass orchestrator that reads criteria and emits prompt
  fragments. Phase 8.
- Any prompt-builder or LLM-call code. Phase 8 / later.
- The actual statistical implementations of each criterion's signals
  (Phase 7 commits to a *class* of statistic via each
  :class:`SignalMapping`'s ``description``; Phase 8 commits to the
  specific statistic).
- Any change to :class:`~kind.training.runner.Runner`,
  :class:`~kind.training.runner.RunnerConfig`, or telemetry sinks.
"""

from __future__ import annotations

import re
import types
import typing
from enum import Enum
from typing import Final

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from kind.observer.schemas import AgentStep, DreamRollout, ReplayMeta

__all__ = [
    "TelemetrySurface",
    "ReadingSurface",
    "SignalMapping",
    "Criterion",
    "CriterionRegistry",
    "EMPTY_REGISTRY",
]


_SNAKE_CASE_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]*$")
_ID_MAX_LEN: Final[int] = 40
_SIGNAL_NAME_MAX_LEN: Final[int] = 40

# A permissive NumPy-style slice expression: digits, colons, commas only.
# Examples accepted: ``:32``, ``64:128``, ``:``, ``0,1,2``. Empty string is
# rejected (the regex requires at least one character); ``None`` means "no
# slice — read the whole field" and skips this check.
_SLICE_SPEC_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9:,]+$")

# A dotted-path shape check applied before the semantic resolution against
# the telemetry model: identifier components separated by dots, no
# whitespace, no leading/trailing dot. Catches obvious garbage at
# field-validation time; :func:`_resolve_dotted_path` does the real work.
_FIELD_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$"
)


class TelemetrySurface(str, Enum):
    """Surfaces a criterion may read from.

    Values name the telemetry stream(s) the criterion's signals derive from.
    The four streams below are the read surfaces available to the mirror's
    data plane. ``world_event`` is **deliberately excluded**: ``world_event``
    is architecturally walled off from the agent process, and the mirror
    reading it would create the cross-membrane dependency the design notes
    prohibit (the builder-perturbation events are observed *into* the agent
    process, not back out of it). A future contributor who needs the mirror
    to react to ``world_event`` must amend the design notes' membrane
    decision before adding an enum value here.

    The split between ``agent_step_observable`` and ``agent_step_internal``
    captures the asymmetry of access the synthesis names as load-bearing:
    Io's PolicyView reads the observable channels (action, logprob, policy
    entropy, observation hash, and the single Watts-heuristic scalar);
    the mirror's TelemetryView reads internal state ``(h_t, z_t)``, the
    posterior/prior parameters, the divergences, the encoder embedding, the
    intrinsic signal, the full self-prediction vector, plus those same
    observable channels. A criterion that derives a signal from ``(h_t,
    z_t)`` declares ``agent_step_internal``; a criterion that derives a
    signal from action statistics alone declares ``agent_step_observable``.
    The ``_OBSERVABLE_FIELDS`` / ``_INTERNAL_FIELDS`` allowlists below are
    the enforcement mechanism.
    """

    AGENT_STEP_OBSERVABLE = "agent_step_observable"
    AGENT_STEP_INTERNAL = "agent_step_internal"
    DREAM_ROLLOUT = "dream_rollout"
    REPLAY_META = "replay_meta"
    # Note: ``world_event`` is intentionally NOT a member. The mirror reading
    # ``world_event`` would couple it to the env-process side of the
    # membrane the design notes wall off; the membrane is a substrate
    # decision, not a registry decision. Adding a member here without
    # amending the membrane decision is wrong.


# ``ReadingSurface`` now lives in the observer-level leaf
# ``kind.observer.reading_surface`` (Phase 8a, to break the observer→mirror
# import cycle ``pre_reg`` could trigger on a cold import). It is re-exported
# here unchanged so every ``from kind.mirror.registry import ReadingSurface``
# keeps resolving; the enum's identity, values, and serialization are
# unchanged. See ``kind/observer/reading_surface.py`` for the rationale.
from kind.observer.reading_surface import ReadingSurface


# ---------------------------------------------------------------------------
# Telemetry-model resolution support for SignalMapping.field_path.
#
# A SignalMapping names a path into Io's emitted telemetry. The path is
# validated at construction against the actual Pydantic model for the
# declared surface — a typo or a stale reference trips at module import, not
# when Phase 8 runs an adversarial pass.
# ---------------------------------------------------------------------------

# Each TelemetrySurface maps to the Pydantic model whose fields its
# SignalMappings index. Both AgentStep surfaces share the AgentStep model;
# the OBSERVABLE/INTERNAL distinction is enforced separately by the
# allowlists below.
_SURFACE_TO_MODEL: Final[dict[TelemetrySurface, type[BaseModel]]] = {
    TelemetrySurface.AGENT_STEP_OBSERVABLE: AgentStep,
    TelemetrySurface.AGENT_STEP_INTERNAL: AgentStep,
    TelemetrySurface.DREAM_ROLLOUT: DreamRollout,
    TelemetrySurface.REPLAY_META: ReplayMeta,
}

# AgentStep fields Io's PolicyView can also read — the "observable
# channels": what Io does (the sampled action), how confident it is (the
# action log-probability, the policy entropy), what it saw (the observation
# hash), and the single scalar self-prediction error it reads under the
# Probe 1.5 v2 Watts-heuristic exception (the one sanctioned channel from
# substrate state into Io's input space). A SignalMapping that declares
# ``telemetry_surface=AGENT_STEP_OBSERVABLE`` may only root its
# ``field_path`` here. See ``Kind_design_notes.md`` — the agent/mirror/
# observer layers section, "asymmetry of access between Io and the mirror".
_OBSERVABLE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "action_t",
        "action_logprob_t",
        "policy_entropy_t",
        "obs_hash_t",
        "self_prediction_error_t",
    }
)

# AgentStep fields only the mirror's TelemetryView reads — the substrate
# state proper: the recurrent state, the stochastic latent, the
# posterior/prior parameters and their per-dim and aggregate divergence,
# the reconstruction loss, the encoder embedding, the ensemble-disagreement
# intrinsic signal, and the full self-prediction vector (Io reads only the
# scalar *error* of this vector, never the vector itself). A SignalMapping
# that declares ``telemetry_surface=AGENT_STEP_INTERNAL`` may only root its
# ``field_path`` here. Same design-notes section as ``_OBSERVABLE_FIELDS``.
_INTERNAL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "h_t",
        "z_t",
        "q_params_t",
        "p_params_t",
        "kl_per_dim_t",
        "kl_aggregate_t",
        "recon_loss_t",
        "encoder_embedding_t",
        "intrinsic_signal_t",
        "self_prediction_t",
    }
)
# Note: AgentStep's envelope/indexing fields (``schema_version``,
# ``run_id``, ``checkpoint_id``, ``t``, ``episode_id``, ``step_in_episode``,
# ``wallclock_ms``) are deliberately in *neither* allowlist. A criterion's
# *signal* derives from a substantive channel, not from a record's indices;
# index fields appear in citation ranges (Phase 8), not in SignalMapping
# field paths. A SignalMapping rooting its ``field_path`` in an index field
# is rejected with "not in the {observable,internal} allowlist".


def _as_basemodel(annotation: object) -> type[BaseModel] | None:
    """Return the Pydantic-model class an annotation resolves to, or ``None``.

    Unwraps ``X | None`` / ``Optional[X]`` / ``Union[...]`` and returns the
    first ``BaseModel`` subclass found. Returns ``None`` for non-model
    annotations (``list[float]``, ``int``, ``str``, plain ``tuple[...]``,
    etc.) — the telemetry schemas as of Probe 2 have no nested models, so in
    practice this always returns ``None`` and a dotted path with more than
    one component is rejected as "not a nested model". The unwrap logic is
    kept correct for the day a telemetry model gains a nested submodel.
    """
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        for arg in typing.get_args(annotation):
            inner = _as_basemodel(arg)
            if inner is not None:
                return inner
        return None
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _resolve_dotted_path(model: type[BaseModel], field_path: str) -> None:
    """Walk a dotted path against a Pydantic model's fields.

    Raises :class:`ValueError` naming the bad component and the available
    fields at that level if any component is unknown, or if an intermediate
    component is not a nested model. Succeeds (returns ``None``) if every
    component resolves.
    """
    parts = field_path.split(".")
    current: type[BaseModel] = model
    for index, part in enumerate(parts):
        fields = current.model_fields
        if part not in fields:
            raise ValueError(
                f"field_path {field_path!r}: component {part!r} (position "
                f"{index}) is not a field of {current.__name__}. Available "
                f"fields at this level: {sorted(fields)}."
            )
        if index < len(parts) - 1:
            inner = _as_basemodel(fields[part].annotation)
            if inner is None:
                raise ValueError(
                    f"field_path {field_path!r}: component {part!r} on "
                    f"{current.__name__} has type "
                    f"{fields[part].annotation!r}, which is not a nested "
                    f"model; the remaining path "
                    f"{'.'.join(parts[index + 1:])!r} cannot be resolved."
                )
            current = inner


class SignalMapping(BaseModel):
    """A single named signal a criterion derives from one telemetry surface.

    A criterion's operational definition is, in part, a set of these: each
    names a path into Io's emitted telemetry and commits to a *class* of
    statistic computed from it. The class — not the exact statistic — is the
    Phase 7 commitment; Phase 8's prompt-builder commits the specific
    statistic and computes it. Cross-surface composites are expressed as
    multiple :class:`SignalMapping` records combined at prompt-build time;
    a single :class:`SignalMapping` reads from exactly one surface.

    The model is frozen and forbids extra fields. No field is a writable
    handle, callback, sink, or anything that could be invoked against Io's
    input space — the structural read-only invariant of the mirror's
    one-way data plane (see :func:`tests.test_signal_mapping.test_signal_mapping_no_writer_shape`).

    Validation that runs at construction (so a stale reference trips at
    module import, not at Phase 8 prompt-build):

    - ``name`` matches the snake_case pattern (the same one as
      :attr:`Criterion.id`) and is ≤40 chars.
    - ``description`` is non-empty.
    - ``slice_spec``, if present, is a permissive NumPy-style slice
      expression (digits, colons, commas only); the empty string is
      rejected; ``None`` means "no slice".
    - ``field_path`` is a well-formed dotted path that *resolves* against
      the Pydantic model for ``telemetry_surface``; for the two
      ``AgentStep`` surfaces the root component must additionally be in the
      surface's allowlist (the asymmetry-of-access boundary — a mapping
      declaring ``AGENT_STEP_OBSERVABLE`` with ``field_path="h_t"`` is
      rejected because ``h_t`` is internal).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    telemetry_surface: TelemetrySurface
    field_path: str
    slice_spec: str | None = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if len(value) > _SIGNAL_NAME_MAX_LEN:
            raise ValueError(
                f"SignalMapping.name must be at most {_SIGNAL_NAME_MAX_LEN} "
                f"characters; got {len(value)} for {value!r}."
            )
        if not _SNAKE_CASE_RE.fullmatch(value):
            raise ValueError(
                f"SignalMapping.name must match snake_case "
                f"({_SNAKE_CASE_RE.pattern}); got {value!r}."
            )
        return value

    @field_validator("description")
    @classmethod
    def _validate_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "SignalMapping.description must be non-empty: it is the "
                "operational record of what the signal measures and which "
                "class of statistic Phase 8 will compute from the field path."
            )
        return value

    @field_validator("slice_spec")
    @classmethod
    def _validate_slice_spec(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _SLICE_SPEC_RE.fullmatch(value):
            raise ValueError(
                f"SignalMapping.slice_spec must be a NumPy-style slice "
                f"expression using only digits, colons, and commas (e.g. "
                f"':32', '64:128', '0,1,2'); got {value!r}. The empty string "
                f"is rejected; pass None for 'no slice — read the whole "
                f"field'."
            )
        return value

    @field_validator("field_path")
    @classmethod
    def _validate_field_path_shape(cls, value: str) -> str:
        if not _FIELD_PATH_RE.fullmatch(value):
            raise ValueError(
                f"SignalMapping.field_path must be a dotted path of "
                f"identifier components with no whitespace and no "
                f"leading/trailing dot (e.g. 'h_t', 'q_params_t', "
                f"'posterior.mean'); got {value!r}."
            )
        return value

    @model_validator(mode="after")
    def _validate_field_path_resolves(self) -> SignalMapping:
        model = _SURFACE_TO_MODEL[self.telemetry_surface]
        root = self.field_path.split(".", 1)[0]

        if self.telemetry_surface is TelemetrySurface.AGENT_STEP_OBSERVABLE:
            if root not in _OBSERVABLE_FIELDS:
                detail = (
                    "a substrate-internal AgentStep field — declare "
                    "telemetry_surface=AGENT_STEP_INTERNAL instead"
                    if root in _INTERNAL_FIELDS
                    else "not a substantive AgentStep channel (index/envelope "
                    "fields are signal sources for no criterion)"
                )
                raise ValueError(
                    f"SignalMapping {self.name!r}: field_path "
                    f"{self.field_path!r} roots at {root!r}, which is not an "
                    f"observable AgentStep field. A mapping declaring "
                    f"telemetry_surface=AGENT_STEP_OBSERVABLE may only read "
                    f"the channels Io's PolicyView also reads: "
                    f"{sorted(_OBSERVABLE_FIELDS)}. ({root!r} is {detail}.) "
                    f"This is the asymmetry-of-access boundary enforced at "
                    f"registry load."
                )
        elif self.telemetry_surface is TelemetrySurface.AGENT_STEP_INTERNAL:
            if root not in _INTERNAL_FIELDS:
                detail = (
                    "an observable channel — declare "
                    "telemetry_surface=AGENT_STEP_OBSERVABLE instead"
                    if root in _OBSERVABLE_FIELDS
                    else "not a substantive AgentStep channel (index/envelope "
                    "fields are signal sources for no criterion)"
                )
                raise ValueError(
                    f"SignalMapping {self.name!r}: field_path "
                    f"{self.field_path!r} roots at {root!r}, which is not a "
                    f"substrate-internal AgentStep field. A mapping declaring "
                    f"telemetry_surface=AGENT_STEP_INTERNAL may only read the "
                    f"substrate state the mirror's TelemetryView reads: "
                    f"{sorted(_INTERNAL_FIELDS)}. ({root!r} is {detail}.) "
                    f"This is the asymmetry-of-access boundary enforced at "
                    f"registry load."
                )

        _resolve_dotted_path(model, self.field_path)
        return self


class Criterion(BaseModel):
    """A single frozen mirror criterion.

    Fields: id + display_name + framework + operational description +
    read-from surfaces (``telemetry_surfaces``) + reading-at surfaces
    (``reading_surfaces``) + falsifier prose + falsifier_id + signal
    mappings + held-out flag. As of Phase 7 ``signal_mappings`` is a
    ``tuple[SignalMapping, ...]`` (Phase 6 stored a ``dict[str, str]``;
    the typed model carries the field-path references that resolve
    against the telemetry schemas at registry load). The three v2
    criteria are written as concrete :class:`Criterion` literals in
    :mod:`kind.mirror.criteria_v2`; Phase 8's adversarial-pass
    orchestrator reads the populated registry and emits per-surface
    prompt fragments.

    The model is frozen and forbids extra fields. The absence of any
    write-surface field is structural: the registry is part of the mirror's
    one-way data plane.

    **Phase 9 addition: ``falsifier_id``.** The criterion's prose
    falsifier is paired with a stable string identifier the Phase 9
    judge layer references in its per-falsifier verdicts
    (:class:`~kind.mirror.judge.FalsifierVerdict`). The ``_v1`` suffix
    convention is for future versioning — if a criterion's falsifier
    prose is amended in a future Phase 7 revision, the new revision
    bumps the suffix and the judge's historical verdicts remain
    interpretable. The field is a string (not a writer-shape); the
    Phase 6 structural-no-writer test on :class:`Criterion` continues
    to pass.

    Validators:

    - ``id`` / ``framework`` / ``falsifier_id`` are snake_case;
      ``id`` and ``falsifier_id`` are ≤40 chars.
    - ``display_name`` / ``description`` / ``falsifier`` are non-empty.
    - ``telemetry_surfaces`` / ``reading_surfaces`` are non-empty.
    - ``signal_mappings`` is non-empty, a tuple of :class:`SignalMapping`
      (a raw ``dict`` is rejected with a migration hint), with unique
      ``name`` values across the tuple.
    - Every :class:`SignalMapping`'s ``telemetry_surface`` appears in this
      criterion's ``telemetry_surfaces`` — a criterion cannot reference a
      signal from a surface it did not declare.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    display_name: str
    framework: str
    description: str
    telemetry_surfaces: frozenset[TelemetrySurface]
    reading_surfaces: frozenset[ReadingSurface]
    falsifier: str
    falsifier_id: str
    signal_mappings: tuple[SignalMapping, ...]
    held_out: bool = False

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        if len(value) > _ID_MAX_LEN:
            raise ValueError(
                f"Criterion.id must be at most {_ID_MAX_LEN} characters; "
                f"got {len(value)} for {value!r}."
            )
        if not _SNAKE_CASE_RE.fullmatch(value):
            raise ValueError(
                f"Criterion.id must match snake_case "
                f"({_SNAKE_CASE_RE.pattern}); got {value!r}."
            )
        return value

    @field_validator("framework")
    @classmethod
    def _validate_framework(cls, value: str) -> str:
        if not _SNAKE_CASE_RE.fullmatch(value):
            raise ValueError(
                f"Criterion.framework must match snake_case "
                f"({_SNAKE_CASE_RE.pattern}); got {value!r}."
            )
        return value

    @field_validator("falsifier_id")
    @classmethod
    def _validate_falsifier_id(cls, value: str) -> str:
        # Same shape as Criterion.id — snake_case, ≤40 chars. Phase 9
        # uses this as a stable identifier the judge's
        # ``FalsifierVerdict`` records; the ``_v1`` suffix convention
        # for forward versioning is by convention not by regex (the
        # regex pins shape only).
        if len(value) > _ID_MAX_LEN:
            raise ValueError(
                f"Criterion.falsifier_id must be at most {_ID_MAX_LEN} "
                f"characters; got {len(value)} for {value!r}."
            )
        if not _SNAKE_CASE_RE.fullmatch(value):
            raise ValueError(
                f"Criterion.falsifier_id must match snake_case "
                f"({_SNAKE_CASE_RE.pattern}); got {value!r}."
            )
        return value

    @field_validator("display_name")
    @classmethod
    def _validate_display_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Criterion.display_name must be non-empty.")
        return value

    @field_validator("description")
    @classmethod
    def _validate_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Criterion.description must be non-empty.")
        return value

    @field_validator("falsifier")
    @classmethod
    def _validate_falsifier(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "Criterion.falsifier must be non-empty: a criterion with no "
                "falsifiable condition is not a criterion (the charter's "
                "discipline element)."
            )
        return value

    @field_validator("telemetry_surfaces")
    @classmethod
    def _validate_telemetry_surfaces_non_empty(
        cls, value: frozenset[TelemetrySurface]
    ) -> frozenset[TelemetrySurface]:
        if len(value) == 0:
            raise ValueError(
                "Criterion.telemetry_surfaces must be non-empty: every "
                "criterion derives its signals from at least one telemetry "
                "stream."
            )
        return value

    @field_validator("reading_surfaces")
    @classmethod
    def _validate_reading_surfaces_non_empty(
        cls, value: frozenset[ReadingSurface]
    ) -> frozenset[ReadingSurface]:
        if len(value) == 0:
            raise ValueError(
                "Criterion.reading_surfaces must be non-empty: every "
                "criterion is read at at least one of substrate-side, "
                "head-internal, or behavior-side."
            )
        return value

    @field_validator("signal_mappings", mode="before")
    @classmethod
    def _reject_dict_signal_mappings(cls, value: object) -> object:
        if isinstance(value, dict):
            raise ValueError(
                "Criterion.signal_mappings is now a tuple of SignalMapping "
                "records (Phase 7), not the dict[str, str] of Phase 6. "
                "Migrate each dict entry to a SignalMapping(name=..., "
                "description=..., telemetry_surface=..., field_path=...). "
                "This guard exists so the migration cannot drift back."
            )
        return value

    @field_validator("signal_mappings")
    @classmethod
    def _validate_signal_mappings(
        cls, value: tuple[SignalMapping, ...]
    ) -> tuple[SignalMapping, ...]:
        if len(value) == 0:
            raise ValueError(
                "Criterion.signal_mappings must be non-empty: a criterion "
                "with no signals cannot be operationalized."
            )
        seen: dict[str, int] = {}
        for index, mapping in enumerate(value):
            if mapping.name in seen:
                raise ValueError(
                    f"Criterion.signal_mappings has duplicate signal name "
                    f"{mapping.name!r} at indices {seen[mapping.name]} and "
                    f"{index}; every signal name in a criterion must be "
                    f"unique."
                )
            seen[mapping.name] = index
        return value

    @model_validator(mode="after")
    def _validate_signal_surfaces_subset(self) -> Criterion:
        declared = self.telemetry_surfaces
        for index, mapping in enumerate(self.signal_mappings):
            if mapping.telemetry_surface not in declared:
                raise ValueError(
                    f"Criterion {self.id!r}: signal_mappings[{index}] "
                    f"({mapping.name!r}) reads from telemetry surface "
                    f"{mapping.telemetry_surface.value!r}, which is not in "
                    f"this criterion's declared telemetry_surfaces "
                    f"({sorted(s.value for s in declared)}). A criterion "
                    f"cannot reference a signal from a surface it did not "
                    f"declare."
                )
        return self


class CriterionRegistry(BaseModel):
    """A frozen tuple of :class:`Criterion` records with read-only lookups.

    Production registries (Phase 7's :data:`kind.mirror.criteria_v2.V2_REGISTRY`)
    have at least one criterion; constructing :class:`CriterionRegistry`
    with ``criteria=()`` directly raises a validation error. The single
    sanctioned empty form is the module-level :data:`EMPTY_REGISTRY`
    constant.

    All public methods return immutable values: a :class:`Criterion`,
    a ``bool``, a ``frozenset[str]``, or a ``tuple[Criterion, ...]``. None
    returns a writer, a callback, or anything invokable against Io's input
    space. This is the structural assertion of the read-only invariant.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    criteria: tuple[Criterion, ...]

    @model_validator(mode="after")
    def _validate_non_empty(self) -> CriterionRegistry:
        if len(self.criteria) == 0:
            raise ValueError(
                "CriterionRegistry must have at least one criterion. The "
                "single sanctioned empty form is "
                "kind.mirror.registry.EMPTY_REGISTRY."
            )
        return self

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> CriterionRegistry:
        seen: dict[str, int] = {}
        for index, criterion in enumerate(self.criteria):
            if criterion.id in seen:
                first = seen[criterion.id]
                raise ValueError(
                    f"CriterionRegistry has duplicate criterion id "
                    f"{criterion.id!r} at indices {first} and {index}; "
                    f"every id must be unique."
                )
            seen[criterion.id] = index
        return self

    @model_validator(mode="after")
    def _validate_partition(self) -> CriterionRegistry:
        # Defense against future refactors: by construction, ``held_out``
        # is the partition, so active and held-out sets are disjoint and
        # cover every criterion. The check is structural — if a future
        # change adds a third state, this trips.
        active_count = sum(1 for c in self.criteria if not c.held_out)
        held_out_count = sum(1 for c in self.criteria if c.held_out)
        if active_count + held_out_count != len(self.criteria):
            raise ValueError(
                f"CriterionRegistry partition broken: active={active_count}, "
                f"held_out={held_out_count}, total={len(self.criteria)}. "
                f"The held_out flag is the only sanctioned partition."
            )
        return self

    def get(self, criterion_id: str) -> Criterion:
        """Look up a criterion by id; raise :class:`KeyError` if missing."""
        for criterion in self.criteria:
            if criterion.id == criterion_id:
                return criterion
        raise KeyError(
            f"CriterionRegistry has no criterion with id "
            f"{criterion_id!r}. Known ids: {sorted(self.all_ids())}."
        )

    def has(self, criterion_id: str) -> bool:
        """Non-raising lookup: ``True`` if the id is present."""
        return any(c.id == criterion_id for c in self.criteria)

    def all_ids(self) -> frozenset[str]:
        """Frozen set of every criterion id in the registry."""
        return frozenset(c.id for c in self.criteria)

    def active(self) -> tuple[Criterion, ...]:
        """Criteria with ``held_out=False``, in registration order."""
        return tuple(c for c in self.criteria if not c.held_out)

    def held_out(self) -> tuple[Criterion, ...]:
        """Criteria with ``held_out=True``, in registration order."""
        return tuple(c for c in self.criteria if c.held_out)

    def by_framework(self, framework: str) -> tuple[Criterion, ...]:
        """All criteria from a given framework, in registration order."""
        return tuple(c for c in self.criteria if c.framework == framework)

    def by_reading_surface(
        self, surface: ReadingSurface
    ) -> tuple[Criterion, ...]:
        """All criteria whose ``reading_surfaces`` contains the given
        surface, in registration order. A multi-surface criterion appears
        in the result for each of its surfaces."""
        return tuple(c for c in self.criteria if surface in c.reading_surfaces)


# The single sanctioned empty registry. Constructed via ``model_construct``
# to bypass the non-empty validator at module load time. Tests that need an
# empty registry import this constant; constructing
# ``CriterionRegistry(criteria=())`` directly raises by design.
EMPTY_REGISTRY: Final[CriterionRegistry] = CriterionRegistry.model_construct(
    criteria=(),
)
