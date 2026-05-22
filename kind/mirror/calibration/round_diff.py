"""Phase 12 cross-round diff record.

Two rounds may share :class:`~kind.mirror.statistics.StatisticConfig`
(and be directly comparable as readings against the same statistical
basis) or differ (and need the diff record to be interpretable later).

This module's product is :class:`RoundDiff`, a typed record naming every
changed configuration field between two rounds, plus a required
rationale per change. A round whose config changed without a rationale
is a structural violation — the rationale is the discipline that the
diff doesn't merely surface drift but explains it.

**Required rationale.** :class:`ConfigFieldChange` carries a non-empty
``rationale`` string; the field-level validator rejects empty strings.
:func:`compute_round_diff` takes a ``rationales`` mapping keyed by
dotted field path; a changed field with no entry in the mapping raises
``ValueError`` at compute time. The diff cannot be built dishonestly:
either you supply a rationale for every change, or you don't get a
diff.

**Scope.** Only :class:`~kind.mirror.calibration.round.RoundConfig`'s
``statistic_config`` and ``llm_config`` fields are walked for changes,
plus the criterion-set partitions
(:attr:`~kind.mirror.calibration.round.RoundConfig.active_registry` /
:attr:`held_out_registry` — by their criterion-id tuples, not by deep
equality on the registry models). The ``checkpoint_ids``,
``sham_schedule``, ``round_id``, and
``pre_registration_template_path`` fields are *not* in the diff: those
fields differ between rounds by design (different checkpoints, different
seeds, different templates) and surfacing them as "changes" would drown
the meaningful drift in noise. The journal entry records that these
fields differ; the diff record records configuration choices that
plausibly affect comparability.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from kind.mirror.calibration.round import RoundConfig
from kind.mirror.llm_caller import LLMConfig
from kind.mirror.statistics import StatisticConfig

__all__ = [
    "ConfigFieldChange",
    "CriterionSetChange",
    "RoundDiff",
    "SyntheticScheduleChange",
    "compute_round_diff",
]


# ---------------------------------------------------------------------------
# Records.
# ---------------------------------------------------------------------------


class ConfigFieldChange(BaseModel):
    """One field that changed between two rounds' configs.

    Frozen, ``extra="forbid"``. Fields:

    - ``field_path``: dotted path into the config
      (e.g. ``"statistic_config.kmeans_k"``).
    - ``prior_value`` / ``current_value``: the values that differed.
      Stored as ``Any``; serialized via the value's own
      ``model_dump`` (for nested Pydantic models) or as the primitive
      (for scalars). The journal entry quotes these verbatim.
    - ``rationale``: non-empty string. Explains why the field was
      changed between rounds. The validator rejects empty / whitespace
      strings; a round that changes a config field with no rationale is
      a structural violation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    field_path: str
    prior_value: Any
    current_value: Any
    rationale: str

    @field_validator("field_path", "rationale")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "ConfigFieldChange: field is non-empty. A round that "
                "changes a config field without a rationale is a "
                "structural violation — supply a rationale or do not "
                "change the field."
            )
        return value


class CriterionSetChange(BaseModel):
    """The criterion-set partition's change between two rounds.

    Frozen, ``extra="forbid"``. Each side is a tuple of criterion ids
    (alphabetically sorted for diff stability). The rationale is
    required by the same discipline as :class:`ConfigFieldChange`.

    A round whose active or held-out partition changed without a
    rationale is a stronger structural violation than a single-field
    config change: the partition determines which criteria the prompt
    even mentions. The rationale must explain why the partitioning
    moved.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    prior_active: tuple[str, ...]
    current_active: tuple[str, ...]
    prior_held_out: tuple[str, ...]
    current_held_out: tuple[str, ...]
    rationale: str

    @field_validator("rationale")
    @classmethod
    def _validate_rationale_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "CriterionSetChange: rationale is required and "
                "non-empty. A partition change without rationale is a "
                "structural violation."
            )
        return value


class SyntheticScheduleChange(BaseModel):
    """Phase 13 synthetic-perturbation-schedule diff between two rounds.

    Frozen, ``extra="forbid"``. The synthetic schedule's entries differ
    between rounds by design (different seeds; different checkpoints
    yield different placement-window samples), so an entry-by-entry
    diff would always be non-empty in practice. This record instead
    captures the scalar fields that describe the schedule's *shape*
    (seed, count, synthetics-per-pass) plus a required rationale.

    The diff is non-``None`` only when the two schedules differ on any
    of these scalar fields *or* on entry count. Two rounds whose
    schedules are by-value equal (same entries, same seed, same
    synthetics_per_pass) produce ``synthetic_schedule_changes=None``.

    Fields:

    - ``prior_seed`` / ``current_seed``: the schedule seeds.
    - ``prior_synthetics_per_pass`` / ``current_synthetics_per_pass``:
      the committed count per pass.
    - ``prior_entry_count`` / ``current_entry_count``: the realised
      total entry count (a function of seed, checkpoints, passes,
      synthetics_per_pass).
    - ``rationale``: required, non-empty. Carries the same discipline
      as :class:`ConfigFieldChange`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    prior_seed: int
    current_seed: int
    prior_synthetics_per_pass: int
    current_synthetics_per_pass: int
    prior_entry_count: int
    current_entry_count: int
    rationale: str

    @field_validator("rationale")
    @classmethod
    def _validate_rationale_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "SyntheticScheduleChange: rationale is required and "
                "non-empty. A synthetic-schedule change without "
                "rationale is a structural violation."
            )
        return value


class RoundDiff(BaseModel):
    """The cross-round diff between two :class:`RoundConfig` instances.

    Frozen, ``extra="forbid"``. Empty tuples / ``None`` on every field
    mean the configs are bit-identical on the dimensions that affect
    comparability — the rounds' readings can be compared directly.

    The diff is by-value, not by-reference: two rounds with separately
    constructed but byte-equal ``StatisticConfig`` produce empty
    ``statistic_config_changes``. This is the structural verification
    Phase 12's smoke uses to assert Round 1 and Round 2's configs are
    actually identical (the rounds share the same default
    ``StatisticConfig`` from Phase 8 and the same default
    ``LLMConfig``).

    Phase 13 addition: ``synthetic_schedule_changes`` carries the
    cross-round synthetic-schedule diff. ``None`` means the two
    rounds' synthetic schedules are by-value equal; a
    :class:`SyntheticScheduleChange` record otherwise, with the
    rationale-required invariant.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    prior_round_id: str
    current_round_id: str
    statistic_config_changes: tuple[ConfigFieldChange, ...]
    llm_config_changes: tuple[ConfigFieldChange, ...]
    criterion_set_changes: CriterionSetChange | None
    synthetic_schedule_changes: SyntheticScheduleChange | None = None
    notes: str

    @field_validator("prior_round_id", "current_round_id")
    @classmethod
    def _validate_round_ids(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("round id must be non-empty.")
        return value


# ---------------------------------------------------------------------------
# Diff computation.
# ---------------------------------------------------------------------------


def _diff_pydantic_model(
    prior: BaseModel,
    current: BaseModel,
    *,
    prefix: str,
    rationales: dict[str, str],
) -> tuple[ConfigFieldChange, ...]:
    """Walk two Pydantic models field-by-field, returning changes.

    Both models must be the same class; comparison is by-value via
    ``model_dump()`` of each field. The ``prefix`` is the dotted-path
    namespace (e.g. ``"statistic_config"``); each changed field's
    ``field_path`` is ``f"{prefix}.{field_name}"``.

    A changed field with no entry in ``rationales`` raises a
    ``ValueError`` naming the unrationaled field path. This is the
    structural enforcement: changes without rationales cannot be diffed.
    """
    if type(prior) is not type(current):
        raise TypeError(
            f"_diff_pydantic_model: prior and current must be the same "
            f"class; got {type(prior).__name__} and "
            f"{type(current).__name__}."
        )
    prior_dump = prior.model_dump()
    current_dump = current.model_dump()
    changes: list[ConfigFieldChange] = []
    field_names = set(prior_dump) | set(current_dump)
    for name in sorted(field_names):
        prior_v = prior_dump.get(name)
        current_v = current_dump.get(name)
        if prior_v == current_v:
            continue
        field_path = f"{prefix}.{name}"
        rationale = rationales.get(field_path)
        if rationale is None or not rationale.strip():
            raise ValueError(
                f"compute_round_diff: field {field_path!r} changed "
                f"between rounds ({prior_v!r} → {current_v!r}) but no "
                f"rationale was supplied. Pass rationales="
                f"{{'{field_path}': '...'}} to document the change."
            )
        changes.append(
            ConfigFieldChange(
                field_path=field_path,
                prior_value=prior_v,
                current_value=current_v,
                rationale=rationale,
            )
        )
    return tuple(changes)


def compute_round_diff(
    prior_round_config: RoundConfig,
    current_round_config: RoundConfig,
    *,
    rationales: dict[str, str] | None = None,
    notes: str = "",
) -> RoundDiff:
    """Compute the :class:`RoundDiff` between two rounds.

    Walks the two configs' ``statistic_config`` and ``llm_config`` and
    the criterion-set partitions, naming each change. Every changed
    field must have a rationale in ``rationales`` (keyed by the
    field's dotted path, e.g. ``"statistic_config.kmeans_k"``,
    ``"llm_config.model_name"``, ``"criterion_set"``). A change without
    rationale raises ``ValueError``.

    ``rationales=None`` is allowed *only* if the configs are
    bit-identical on the diffed dimensions; if any change is detected
    with no rationale supplied, the function raises.

    ``notes``: free-text description of what the round-to-round
    comparison was for. Quoted verbatim in the journal entry.
    """
    rationales_map = dict(rationales) if rationales else {}

    statistic_config_changes = _diff_pydantic_model(
        prior_round_config.statistic_config,
        current_round_config.statistic_config,
        prefix="statistic_config",
        rationales=rationales_map,
    )
    llm_config_changes = _diff_pydantic_model(
        prior_round_config.llm_config,
        current_round_config.llm_config,
        prefix="llm_config",
        rationales=rationales_map,
    )

    criterion_set_changes: CriterionSetChange | None = None
    prior_active = tuple(
        sorted(c.id for c in prior_round_config.active_registry.criteria)
    )
    current_active = tuple(
        sorted(c.id for c in current_round_config.active_registry.criteria)
    )
    prior_held_out = tuple(
        sorted(c.id for c in prior_round_config.held_out_registry.criteria)
    )
    current_held_out = tuple(
        sorted(c.id for c in current_round_config.held_out_registry.criteria)
    )
    if (
        prior_active != current_active
        or prior_held_out != current_held_out
    ):
        rationale = rationales_map.get("criterion_set")
        if rationale is None or not rationale.strip():
            raise ValueError(
                "compute_round_diff: criterion-set partition changed "
                "between rounds but no rationale was supplied. Pass "
                "rationales={'criterion_set': '...'} to document the "
                "partition change."
            )
        criterion_set_changes = CriterionSetChange(
            prior_active=prior_active,
            current_active=current_active,
            prior_held_out=prior_held_out,
            current_held_out=current_held_out,
            rationale=rationale,
        )

    # Phase 13: synthetic-schedule diff. Two schedules are equal iff
    # every scalar field matches *and* the entry tuples are by-value
    # equal. The latter is captured indirectly by entry_count + seed
    # equality (since the generator is seed-deterministic given the
    # same inputs) for the common case, but we walk the full equality
    # check for robustness against hand-constructed schedules in tests.
    prior_syn = prior_round_config.synthetic_schedule
    current_syn = current_round_config.synthetic_schedule
    synthetic_schedule_changes: SyntheticScheduleChange | None = None
    schedules_equal = (
        prior_syn.seed == current_syn.seed
        and prior_syn.synthetics_per_pass == current_syn.synthetics_per_pass
        and prior_syn.entries == current_syn.entries
    )
    if not schedules_equal:
        rationale = rationales_map.get("synthetic_schedule")
        if rationale is None or not rationale.strip():
            raise ValueError(
                "compute_round_diff: synthetic_schedule changed between "
                "rounds but no rationale was supplied. Pass rationales="
                "{'synthetic_schedule': '...'} to document the change "
                "(distinct seed per checkpoint, distinct "
                "synthetics_per_pass, etc.)."
            )
        synthetic_schedule_changes = SyntheticScheduleChange(
            prior_seed=prior_syn.seed,
            current_seed=current_syn.seed,
            prior_synthetics_per_pass=prior_syn.synthetics_per_pass,
            current_synthetics_per_pass=current_syn.synthetics_per_pass,
            prior_entry_count=len(prior_syn.entries),
            current_entry_count=len(current_syn.entries),
            rationale=rationale,
        )

    return RoundDiff(
        prior_round_id=prior_round_config.round_id,
        current_round_id=current_round_config.round_id,
        statistic_config_changes=statistic_config_changes,
        llm_config_changes=llm_config_changes,
        criterion_set_changes=criterion_set_changes,
        synthetic_schedule_changes=synthetic_schedule_changes,
        notes=notes,
    )


# Imports below are placed at the bottom to allow the module to be
# imported even if forward-ref-style annotations on
# :class:`ConfigFieldChange.prior_value` / ``current_value`` reference
# concrete config classes. The actual diff machinery only needs the
# classes for type-narrowing in callers; the ``Any`` typed slot accepts
# any serializable value at runtime.
_ = (StatisticConfig, LLMConfig)  # noqa: F841 — keep imports live
