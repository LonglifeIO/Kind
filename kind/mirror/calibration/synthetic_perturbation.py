"""Phase 13 synthetic-real-perturbation injection layer.

A *synthetic real perturbation* is the Phase 13 discriminative case the
Phase 12 calibration could not test: the Probe 1 / Probe 1.5 telemetry
contained zero real ``builder_perturbation`` events, so the sham-zero
result from Phase 12 was vacuous — the mirror had nothing to discriminate
*from*. Phase 13 adds a layer that injects perturbation events into the
aligned timeline at pre-registered timestamps so the recovery statistics
have real numbers to compute and the mirror has a discriminative case to
read.

The structure mirrors :mod:`~kind.mirror.calibration.sham_schedule` from
Phase 12 — pre-registered schedule, deterministic generation, immutable
injection. The two schedules are independent: shams test "does the mirror
admit on no-signal prompts?"; synthetics test "does the mirror admit
when the recovery statistics have real values?". Together they bracket
the calibration's discriminative behavior.

**Three categories, mutually exclusive at the payload level.** A
synthetic perturbation carries ``payload["is_synthetic"] = True`` and
``payload["is_sham"] = False``; a sham carries ``is_sham = True`` and
``is_synthetic = False``; a real Probe 4 perturbation (which doesn't
exist yet) carries both as ``False``. The on-disk record is unambiguous
about what each event was — the orchestrator's calibration checks
dispatch on these flags.

**The mirror does not see the synthetic/sham/real label in its prompt.**
Synthetic events are injected with :attr:`PerturbationEvent.is_sham` =
``False``, so the prompt_builder displays them under "Real perturbations
(recovery readings apply)" alongside any real events — the same surface
a Probe 4 perturbation would land at. The orchestrator's post-call
:func:`~kind.mirror.calibration.synthetic_calibration_check.check_synthetic_calibration`
filters on the payload flag; the LLM cannot.

**The schedule is pre-registered.** The schedule is constructed once,
seed-deterministic, and frozen onto :class:`RoundConfig` *before* the
first pass runs. Round-driver-level cross-disjointness against the sham
schedule is enforced both at generation time (the generator avoids
collisions) and at :class:`RoundConfig` validation (config-load-time
safety net).

Out of scope here: the post-call orchestrator check (lives in
:mod:`~kind.mirror.calibration.synthetic_calibration_check`); any change
to the env/runner/actor; any change to the sham-schedule layer.
"""

from __future__ import annotations

import random
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from kind.mirror.calibration.sham_schedule import ShamSchedule, ShamScheduleEntry
from kind.mirror.perturbation_align import (
    PerturbationEvent,
    PerturbationTimeline,
)

__all__ = [
    "SyntheticPerturbationEntry",
    "SyntheticPerturbationSchedule",
    "generate_synthetic_perturbation_schedule",
    "inject_synthetic_events",
]


# ---------------------------------------------------------------------------
# Records.
# ---------------------------------------------------------------------------


class SyntheticPerturbationEntry(BaseModel):
    """One scheduled synthetic-real-perturbation injection.

    Frozen, ``extra="forbid"``. Fields:

    - ``checkpoint_id``: which checkpoint's telemetry this synthetic is
      scheduled against.
    - ``pass_index``: 0-based index within the round.
    - ``synthetic_t``: the :class:`~kind.observer.schemas.AgentStep` ``t``
      at which the synthetic is injected. The injection looks up the
      corresponding agent-step's ``wallclock_ms`` to populate the
      :class:`~kind.mirror.perturbation_align.PerturbationEvent`.
    - ``synthetic_payload``: the payload dict for the injected event. The
      constructor forces ``is_synthetic=True`` *and* ``is_sham=False`` on
      the payload — a synthetic entry cannot silently relabel itself as
      a sham (or vice versa). The two flags together pin the entry's
      category in the on-disk audit trail.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_id: str
    pass_index: int
    synthetic_t: int
    synthetic_payload: dict[str, Any]

    @field_validator("checkpoint_id")
    @classmethod
    def _validate_checkpoint_id_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("checkpoint_id must be non-empty.")
        return value

    @field_validator("pass_index")
    @classmethod
    def _validate_pass_index_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"pass_index must be >= 0; got {value}.")
        return value

    @field_validator("synthetic_t")
    @classmethod
    def _validate_synthetic_t_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"synthetic_t must be >= 0; got {value}.")
        return value

    @model_validator(mode="after")
    def _enforce_category_flags(self) -> "SyntheticPerturbationEntry":
        """Force ``synthetic_payload["is_synthetic"] = True`` and
        ``synthetic_payload["is_sham"] = False``. The constructor refuses
        to build a synthetic entry whose payload disagrees with its
        category — the calibration discipline requires the on-disk record
        to be unambiguous about whether an event was synthetic, sham, or
        real."""
        is_synth = self.synthetic_payload.get("is_synthetic")
        if is_synth is None:
            raise ValueError(
                f"SyntheticPerturbationEntry: synthetic_payload must "
                f"include is_synthetic=True; got "
                f"payload={self.synthetic_payload!r}. The explicit flag is "
                f"how the orchestrator's calibration check distinguishes "
                f"synthetic from real perturbations at audit time."
            )
        if not is_synth:
            raise ValueError(
                f"SyntheticPerturbationEntry: synthetic_payload has "
                f"is_synthetic=False but the entry is a synthetic. Either "
                f"set is_synthetic=True or do not schedule the entry. "
                f"payload={self.synthetic_payload!r}"
            )
        is_sham = self.synthetic_payload.get("is_sham")
        if is_sham is None:
            raise ValueError(
                f"SyntheticPerturbationEntry: synthetic_payload must "
                f"include is_sham=False (the three categories are "
                f"mutually exclusive — synthetic excludes sham); got "
                f"payload={self.synthetic_payload!r}."
            )
        if is_sham:
            raise ValueError(
                f"SyntheticPerturbationEntry: synthetic_payload has "
                f"is_sham=True but the entry is a synthetic. The three "
                f"categories (synthetic, sham, real) are mutually "
                f"exclusive. payload={self.synthetic_payload!r}"
            )
        return self


class SyntheticPerturbationSchedule(BaseModel):
    """The full round's synthetic-perturbation schedule.

    Frozen, ``extra="forbid"``. The schedule is committed at
    pre-registration time as part of
    :class:`~kind.mirror.calibration.round.RoundConfig`; the seed is on
    the schedule so the schedule is reproducible from
    ``(checkpoint_ids, passes_per_checkpoint, synthetics_per_pass,
    telemetry_length, recovery_window, seed)`` alone.

    Fields:

    - ``entries``: the per-checkpoint-per-pass synthetic entries in
      arbitrary order. :func:`inject_synthetic_events` filters to the
      entries for a given ``(checkpoint_id, pass_index)`` and injects
      them into that pass's timeline.
    - ``synthetics_per_pass``: how many synthetic real perturbations the
      round commits to per pass. Phase 13 commits 2 — same as Phase 12's
      committed real-perturbation count, so the synthetic count
      structurally answers the Phase 12 newly-open §5
      (real-perturbation enforcement).
    - ``seed``: the PRNG seed used to generate the entries'
      ``synthetic_t`` placements. Reproducibility hook.

    Validator: no two entries may share
    ``(checkpoint_id, pass_index, synthetic_t)``. A collision would mean
    two synthetics at the same agent-step, which the alignment cannot
    distinguish and the equanimity statistics would double-count.

    Cross-schedule disjointness against the sham schedule is enforced at
    the :class:`RoundConfig`-level validator (and avoided at generation
    time by the optional ``sham_schedule=`` argument to
    :func:`generate_synthetic_perturbation_schedule`). It is *not*
    enforced on this model directly because the model doesn't carry a
    reference to the sham schedule.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    entries: tuple[SyntheticPerturbationEntry, ...]
    synthetics_per_pass: int
    seed: int

    @field_validator("synthetics_per_pass")
    @classmethod
    def _validate_synthetics_per_pass_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"synthetics_per_pass must be >= 0; got {value}.")
        return value

    @model_validator(mode="after")
    def _enforce_no_collisions(self) -> "SyntheticPerturbationSchedule":
        """No two entries may share the same
        ``(checkpoint_id, pass_index, synthetic_t)``."""
        seen: set[tuple[str, int, int]] = set()
        for e in self.entries:
            key = (e.checkpoint_id, e.pass_index, e.synthetic_t)
            if key in seen:
                raise ValueError(
                    f"SyntheticPerturbationSchedule: duplicate entry at "
                    f"(checkpoint_id={e.checkpoint_id!r}, "
                    f"pass_index={e.pass_index}, "
                    f"synthetic_t={e.synthetic_t}). Each (checkpoint, "
                    f"pass, t) carries at most one scheduled synthetic."
                )
            seen.add(key)
        return self


# ---------------------------------------------------------------------------
# Schedule generation.
# ---------------------------------------------------------------------------


def generate_synthetic_perturbation_schedule(
    checkpoint_ids: tuple[str, ...],
    passes_per_checkpoint: int,
    synthetics_per_pass: int,
    telemetry_length: int,
    recovery_window: int,
    seed: int,
    *,
    sham_schedule: ShamSchedule | None = None,
) -> SyntheticPerturbationSchedule:
    """Generate a deterministic :class:`SyntheticPerturbationSchedule`
    from the round's parameters.

    The schedule is deterministic for a given ``seed``: same inputs in
    → same schedule out, byte-for-byte. The seed is stored on the
    returned schedule.

    Placement: for each ``(checkpoint_id, pass_index)`` pair, the
    ``synthetics_per_pass`` synthetic timestamps are drawn uniformly
    without replacement from ``range(1, telemetry_length - recovery_window
    - 1)`` — the upper bound guarantees ``recovery_window`` post-
    perturbation samples are available so the equanimity statistics
    (Mahalanobis recovery lag, entropy/KL trajectory classification) can
    compute on a full post-window without spilling past telemetry end.
    The lower bound (step 1) avoids the first step, which carries the
    self-prediction-error masked sentinel.

    ``sham_schedule``: if provided, the generator filters the placement
    window per ``(checkpoint_id, pass_index)`` to exclude any timestamps
    already carrying a sham. Cross-schedule disjointness is enforced at
    the :class:`RoundConfig` validator too; the generator-level filter
    is the constructive form (avoid the collision in the first place).

    Raises ``ValueError`` if ``synthetics_per_pass`` exceeds the
    available placement window after sham-collision exclusion, or if
    ``telemetry_length`` is too short to admit any placement.
    """
    if not checkpoint_ids:
        raise ValueError(
            "generate_synthetic_perturbation_schedule: checkpoint_ids "
            "must be non-empty."
        )
    if passes_per_checkpoint <= 0:
        raise ValueError(
            f"generate_synthetic_perturbation_schedule: "
            f"passes_per_checkpoint must be > 0; got {passes_per_checkpoint}."
        )
    if synthetics_per_pass < 0:
        raise ValueError(
            f"generate_synthetic_perturbation_schedule: "
            f"synthetics_per_pass must be >= 0; got {synthetics_per_pass}."
        )
    if recovery_window < 0:
        raise ValueError(
            f"generate_synthetic_perturbation_schedule: recovery_window "
            f"must be >= 0; got {recovery_window}."
        )
    # The placement window is range(1, telemetry_length - recovery_window
    # - 1). It must contain at least one position for any synthetic to
    # land; below the minimum the window is empty and the schedule
    # cannot be built.
    placement_upper_exclusive = telemetry_length - recovery_window - 1
    if placement_upper_exclusive <= 1:
        raise ValueError(
            f"generate_synthetic_perturbation_schedule: "
            f"telemetry_length={telemetry_length} with "
            f"recovery_window={recovery_window} yields an empty "
            f"placement window (upper exclusive bound "
            f"{placement_upper_exclusive} <= lower bound 1). Either "
            f"shrink recovery_window or provide a longer telemetry "
            f"window."
        )
    placement_window = list(range(1, placement_upper_exclusive))
    # Seed a fresh RNG for determinism; do not touch the global RNG.
    rng = random.Random(seed)
    entries: list[SyntheticPerturbationEntry] = []
    for ckpt_id in checkpoint_ids:
        for pass_index in range(passes_per_checkpoint):
            # Per-pass sham-collision exclusion: if a sham schedule was
            # supplied, drop any timestamps it already occupies in this
            # ``(checkpoint_id, pass_index)`` slot from the placement
            # window. The exclusion is intentionally per-pass — sham
            # entries for other passes don't affect this pass's
            # placement window.
            sham_ts_this_pass: set[int] = set()
            if sham_schedule is not None:
                for sham in sham_schedule.entries:
                    if (
                        sham.checkpoint_id == ckpt_id
                        and sham.pass_index == pass_index
                    ):
                        sham_ts_this_pass.add(sham.sham_t)
            available = [
                t for t in placement_window if t not in sham_ts_this_pass
            ]
            if synthetics_per_pass > len(available):
                raise ValueError(
                    f"generate_synthetic_perturbation_schedule: "
                    f"synthetics_per_pass={synthetics_per_pass} exceeds "
                    f"the available placement window size "
                    f"{len(available)} for (checkpoint_id={ckpt_id!r}, "
                    f"pass_index={pass_index}) after excluding "
                    f"{len(sham_ts_this_pass)} sham timestamp(s). Either "
                    f"reduce synthetics_per_pass, widen telemetry_length,"
                    f" or shrink the sham schedule."
                )
            synthetic_ts = rng.sample(available, synthetics_per_pass)
            synthetic_ts.sort()
            for synthetic_t in synthetic_ts:
                entries.append(
                    SyntheticPerturbationEntry(
                        checkpoint_id=ckpt_id,
                        pass_index=pass_index,
                        synthetic_t=synthetic_t,
                        synthetic_payload={
                            "is_synthetic": True,
                            "is_sham": False,
                            "source": "phase_13_calibration",
                            "seed": seed,
                        },
                    )
                )
    return SyntheticPerturbationSchedule(
        entries=tuple(entries),
        synthetics_per_pass=synthetics_per_pass,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Injection.
# ---------------------------------------------------------------------------


def inject_synthetic_events(
    perturbation_timeline: PerturbationTimeline,
    synthetic_entries: tuple[SyntheticPerturbationEntry, ...],
    *,
    agent_step_wallclock_lookup: dict[int, int],
) -> PerturbationTimeline:
    """Produce a new :class:`PerturbationTimeline` with the synthetic
    events added at the scheduled timestamps.

    The original timeline is not mutated (both are Pydantic frozen).
    Each synthetic entry becomes a :class:`PerturbationEvent` with
    ``t=entry.synthetic_t``,
    ``wallclock_ms=agent_step_wallclock_lookup[entry.synthetic_t]``,
    ``payload=entry.synthetic_payload`` (carrying the validator's
    ``is_synthetic=True`` and ``is_sham=False``), and ``is_sham=False``.

    The ``is_sham=False`` flag on the :class:`PerturbationEvent` is
    load-bearing: it routes the synthetic through the same recovery-
    statistic codepath as a real perturbation (the recovery-lag and
    trajectory classifiers skip ``is_sham=True`` events but process
    ``is_sham=False`` events normally). The synthetic's
    ``is_synthetic=True`` flag lives on the payload only and is
    inspected by the orchestrator's calibration check; it does not
    affect the statistic pipeline.

    The merged events list is sorted by ``t`` to preserve the
    :class:`PerturbationTimeline`'s sorted-unique invariant. If a
    synthetic's ``t`` collides with an existing real or sham
    perturbation ``t``, the timeline's validator raises (the
    calibration cannot tolerate the ambiguity — re-generate the
    schedule with a different seed or widen the placement window).

    Out of scope: handling a synthetic whose ``t`` is not in the
    ``agent_step_wallclock_lookup`` (the caller must supply a complete
    lookup over the pass's agent-step ``t`` values; missing keys raise
    ``ValueError`` here, mirroring the sham injection's contract).
    """
    if not synthetic_entries:
        # No-op: return the original timeline unchanged (it's
        # immutable; safe to return the same object).
        return perturbation_timeline
    synthetic_events: list[PerturbationEvent] = []
    for entry in synthetic_entries:
        if entry.synthetic_t not in agent_step_wallclock_lookup:
            raise ValueError(
                f"inject_synthetic_events: synthetic_t={entry.synthetic_t}"
                f" has no corresponding agent-step in the wallclock "
                f"lookup. The caller must supply a complete lookup over "
                f"the pass's agent-step t values."
            )
        synthetic_events.append(
            PerturbationEvent(
                t=entry.synthetic_t,
                wallclock_ms=agent_step_wallclock_lookup[entry.synthetic_t],
                payload=dict(entry.synthetic_payload),
                is_sham=False,
            )
        )
    merged = list(perturbation_timeline.events) + synthetic_events
    merged.sort(key=lambda e: e.t)
    return PerturbationTimeline(
        events=tuple(merged),
        run_id=perturbation_timeline.run_id,
        checkpoint_id=perturbation_timeline.checkpoint_id,
    )


# Imports kept live for downstream callers that import the sham types
# alongside synthetic types and to make the cross-disjointness contract
# legible at module level. The ShamScheduleEntry symbol is used in the
# ``sham_schedule=`` argument's effective dispatch path.
_ = (ShamSchedule, ShamScheduleEntry)  # noqa: F841 — keep imports live
