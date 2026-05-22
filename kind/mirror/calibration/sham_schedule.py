"""Phase 12 sham-perturbation schedule and injection.

A sham perturbation is a flag-only ``world_event`` (no env mutation;
agent observation byte-equal pre/post) emitted by the env-server's
sham path with ``payload["is_sham"]=True``. The Phase 8 orchestrator
already does the sham-aware calibration check: if the primary
equanimity reading admits at a sham timestamp, the calibration has
failed and the finding is journaled.

Phase 12 lifts the sham mechanism from "the orchestrator handles it
when sham events appear in the aligned timeline" to "the round driver
commits to a sham schedule at pre-registration time, then injects the
shams into each pass's timeline before the LLM sees its prompt." This
is the calibration defense against the mirror unconsciously learning to
discount perturbations that match a known distribution: the sham
schedule is randomized within the pass, but the schedule itself is
pre-registered.

**The mirror does not see sham/real labels in its prompt.** The
:func:`~kind.mirror.prompt_builder.build_fragment` machinery cites the
:class:`~kind.mirror.perturbation_align.PerturbationTimeline` for
recovery-window step ranges; it does not surface ``is_sham`` to the
LLM. Only the orchestrator's post-call
:func:`~kind.mirror.orchestrator._sham_calibration_check` has access
to the flag. This means the mirror cannot "learn" from a sham-labeled
event in its own context window; the calibration is what it is.

**The schedule is committed pre-registration.** The
:class:`ShamSchedule` is part of :class:`RoundConfig`; the round driver
writes the round config (including the schedule) to disk before any
pass runs. The seed used to generate the schedule is also part of the
schedule, so the schedule itself is reproducible from the seed alone.

Out of scope: actually executing a sham perturbation (the env-server's
sham path already does that — at Phase 12 the perturbations come from
existing checkpoint telemetry; the "injection" here is
synthetic-additive, layering shams onto an existing timeline for the
mirror to read).
"""

from __future__ import annotations

import random
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from kind.mirror.perturbation_align import (
    PerturbationEvent,
    PerturbationTimeline,
)

__all__ = [
    "ShamScheduleEntry",
    "ShamSchedule",
    "generate_sham_schedule",
    "inject_sham_events",
]


# ---------------------------------------------------------------------------
# Records.
# ---------------------------------------------------------------------------


class ShamScheduleEntry(BaseModel):
    """One scheduled sham injection.

    Frozen, ``extra="forbid"``. Fields:

    - ``checkpoint_id``: which checkpoint's telemetry this sham is
      scheduled against.
    - ``pass_index``: 0-based index within the round.
    - ``sham_t``: the :class:`~kind.observer.schemas.AgentStep` ``t``
      at which the sham is injected. The injection looks up the
      corresponding agent-step's ``wallclock_ms`` to populate the
      :class:`~kind.mirror.perturbation_align.PerturbationEvent`'s
      wallclock; the orchestrator-side log remains the source of truth
      for the time axis.
    - ``sham_payload``: the payload dict for the injected event. The
      constructor forces ``is_sham=True`` on the payload — a Phase 12
      sham is never silently relabeled to a real event.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_id: str
    pass_index: int
    sham_t: int
    sham_payload: dict[str, Any]

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

    @field_validator("sham_t")
    @classmethod
    def _validate_sham_t_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"sham_t must be >= 0; got {value}.")
        return value

    @model_validator(mode="after")
    def _force_is_sham_flag(self) -> "ShamScheduleEntry":
        """Force ``sham_payload["is_sham"] = True``. A sham entry whose
        payload silently sets ``is_sham=False`` would defeat the
        calibration check; the constructor refuses to construct one."""
        # Pydantic frozen models can't reassign the dict reference, but
        # we can validate it has the right shape. If is_sham is present
        # and falsy, raise; if absent, raise with a clear message.
        is_sham = self.sham_payload.get("is_sham")
        if is_sham is None:
            raise ValueError(
                f"ShamScheduleEntry: sham_payload must include "
                f"is_sham=True; got payload={self.sham_payload!r}. The "
                f"explicit flag is the orchestrator's sole signal that "
                f"this event is a sham — leaving it out would route "
                f"the sham through the same code path as a real "
                f"perturbation."
            )
        if not is_sham:
            raise ValueError(
                f"ShamScheduleEntry: sham_payload has is_sham=False "
                f"but the entry is a sham. Either set is_sham=True or "
                f"do not schedule the entry. payload={self.sham_payload!r}"
            )
        return self


class ShamSchedule(BaseModel):
    """The full round's sham-injection schedule.

    Frozen, ``extra="forbid"``. The schedule is committed at
    pre-registration time as part of :class:`RoundConfig`; the seed is
    on the schedule so the schedule is reproducible from
    ``(checkpoint_ids, passes_per_checkpoint, real_perturbations_per_pass,
    shams_per_pass, telemetry_length, seed)`` alone.

    Fields:

    - ``entries``: the per-checkpoint-per-pass sham entries in
      arbitrary order. The orchestrator-side :func:`inject_sham_events`
      filters to the entries for a given ``(checkpoint_id, pass_index)``
      and injects them into that pass's timeline.
    - ``real_perturbations_per_pass``: how many real perturbations the
      round commits to per pass. Phase 12 commits 2. The number is
      pre-registered so the journal entry can verify the calibration
      ratio that was actually run matches what was committed.
    - ``shams_per_pass``: how many shams. Phase 12 commits 1.
    - ``seed``: the PRNG seed used to generate the entries' ``sham_t``
      placements. Reproducibility hook.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    entries: tuple[ShamScheduleEntry, ...]
    real_perturbations_per_pass: int
    shams_per_pass: int
    seed: int

    @field_validator("real_perturbations_per_pass", "shams_per_pass")
    @classmethod
    def _validate_counts_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"count must be >= 0; got {value}.")
        return value

    @model_validator(mode="after")
    def _enforce_no_collisions(self) -> "ShamSchedule":
        """No two entries may share the same
        ``(checkpoint_id, pass_index, sham_t)``. A collision would mean
        two shams at the same agent-step, which the alignment cannot
        distinguish and the equanimity statistics would double-count.
        """
        seen: set[tuple[str, int, int]] = set()
        for e in self.entries:
            key = (e.checkpoint_id, e.pass_index, e.sham_t)
            if key in seen:
                raise ValueError(
                    f"ShamSchedule: duplicate entry at "
                    f"(checkpoint_id={e.checkpoint_id!r}, "
                    f"pass_index={e.pass_index}, sham_t={e.sham_t}). "
                    f"Each (checkpoint, pass, t) carries at most one "
                    f"scheduled sham."
                )
            seen.add(key)
        return self


# ---------------------------------------------------------------------------
# Schedule generation.
# ---------------------------------------------------------------------------


def generate_sham_schedule(
    checkpoint_ids: tuple[str, ...],
    passes_per_checkpoint: int,
    real_perturbations_per_pass: int,
    shams_per_pass: int,
    telemetry_length: int,
    seed: int,
) -> ShamSchedule:
    """Generate a deterministic :class:`ShamSchedule` from the round's
    parameters.

    The schedule is deterministic for a given ``seed``: same inputs in
    → same schedule out, byte-for-byte. The seed is stored on the
    returned schedule (Phase 12's reproducibility hook).

    Placement: for each ``(checkpoint_id, pass_index)`` pair, the
    ``shams_per_pass`` sham timestamps are drawn uniformly without
    replacement from ``range(1, telemetry_length - 1)`` — the bounds
    avoid placing a sham at the very first step (where the
    self-prediction-error masked flag is set and the equanimity signal
    is degenerate by construction) or the very last step (where there
    are no post-perturbation samples for the recovery-window
    statistics).

    The ``real_perturbations_per_pass`` field is *not* used by the
    generator — real perturbations come from the checkpoint's existing
    telemetry, not from the schedule. The schedule's
    ``real_perturbations_per_pass`` is a pre-registration commitment
    (the round expects to find this many real events; the journal
    entry checks the actual count against the commitment).

    Raises ``ValueError`` if ``shams_per_pass`` exceeds the available
    placement window, or if ``telemetry_length`` is too short to admit
    any placement.
    """
    if not checkpoint_ids:
        raise ValueError(
            "generate_sham_schedule: checkpoint_ids must be non-empty."
        )
    if passes_per_checkpoint <= 0:
        raise ValueError(
            f"generate_sham_schedule: passes_per_checkpoint must be > 0; "
            f"got {passes_per_checkpoint}."
        )
    if real_perturbations_per_pass < 0:
        raise ValueError(
            f"generate_sham_schedule: real_perturbations_per_pass must "
            f"be >= 0; got {real_perturbations_per_pass}."
        )
    if shams_per_pass < 0:
        raise ValueError(
            f"generate_sham_schedule: shams_per_pass must be >= 0; got "
            f"{shams_per_pass}."
        )
    if telemetry_length < 3:
        raise ValueError(
            f"generate_sham_schedule: telemetry_length must be >= 3 to "
            f"admit any sham placement (the placement window excludes "
            f"step 0 and step telemetry_length-1); got "
            f"{telemetry_length}."
        )
    placement_window = list(range(1, telemetry_length - 1))
    if shams_per_pass > len(placement_window):
        raise ValueError(
            f"generate_sham_schedule: shams_per_pass={shams_per_pass} "
            f"exceeds the placement window size "
            f"{len(placement_window)} (telemetry_length="
            f"{telemetry_length} minus the two boundary steps)."
        )
    # Seed a fresh RNG for determinism; do not touch the global RNG.
    rng = random.Random(seed)
    entries: list[ShamScheduleEntry] = []
    for ckpt_id in checkpoint_ids:
        for pass_index in range(passes_per_checkpoint):
            # Drawn without replacement within this pass; collisions
            # across passes / checkpoints are fine and expected.
            sham_ts = rng.sample(placement_window, shams_per_pass)
            sham_ts.sort()
            for sham_t in sham_ts:
                entries.append(
                    ShamScheduleEntry(
                        checkpoint_id=ckpt_id,
                        pass_index=pass_index,
                        sham_t=sham_t,
                        sham_payload={
                            "is_sham": True,
                            "source": "phase_12_calibration",
                            "seed": seed,
                        },
                    )
                )
    return ShamSchedule(
        entries=tuple(entries),
        real_perturbations_per_pass=real_perturbations_per_pass,
        shams_per_pass=shams_per_pass,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Injection.
# ---------------------------------------------------------------------------


def inject_sham_events(
    perturbation_timeline: PerturbationTimeline,
    sham_entries: tuple[ShamScheduleEntry, ...],
    *,
    agent_step_wallclock_lookup: dict[int, int],
) -> PerturbationTimeline:
    """Produce a new :class:`PerturbationTimeline` with the sham events
    added at the scheduled timestamps.

    The original timeline is not mutated (both are Pydantic frozen).
    Each sham entry becomes a :class:`PerturbationEvent` with
    ``t=entry.sham_t``,
    ``wallclock_ms=agent_step_wallclock_lookup[entry.sham_t]``,
    ``payload=entry.sham_payload`` (with the
    :class:`ShamScheduleEntry` validator's
    ``is_sham=True`` already in place), and ``is_sham=True``.

    The merged events list is sorted by ``t`` to preserve the
    :class:`PerturbationTimeline`'s sorted-unique invariant. If a sham's
    ``t`` collides with an existing real-perturbation ``t``, the
    timeline's validator raises (the calibration cannot tolerate the
    ambiguity — re-generate the schedule with a different seed or
    widen the placement window).

    Out of scope: handling a sham whose ``t`` is not in the
    ``agent_step_wallclock_lookup`` (the caller must supply a complete
    lookup over the pass's agent-step ``t`` values; missing keys raise
    ``KeyError`` from the dict). This is intentional — a sham scheduled
    at a step that doesn't exist in the telemetry is a programming
    error, not a runtime tolerated case.
    """
    if not sham_entries:
        # No-op: return the original timeline unchanged (it's
        # immutable; safe to return the same object).
        return perturbation_timeline
    sham_events: list[PerturbationEvent] = []
    for entry in sham_entries:
        if entry.sham_t not in agent_step_wallclock_lookup:
            raise ValueError(
                f"inject_sham_events: sham_t={entry.sham_t} has no "
                f"corresponding agent-step in the wallclock lookup. The "
                f"caller must supply a complete lookup over the pass's "
                f"agent-step t values."
            )
        sham_events.append(
            PerturbationEvent(
                t=entry.sham_t,
                wallclock_ms=agent_step_wallclock_lookup[entry.sham_t],
                payload=dict(entry.sham_payload),
                is_sham=True,
            )
        )
    merged = list(perturbation_timeline.events) + sham_events
    merged.sort(key=lambda e: e.t)
    return PerturbationTimeline(
        events=tuple(merged),
        run_id=perturbation_timeline.run_id,
        checkpoint_id=perturbation_timeline.checkpoint_id,
    )
