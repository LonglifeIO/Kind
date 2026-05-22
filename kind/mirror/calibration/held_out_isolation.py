"""Phase 13 held-out isolation study.

The second of Phase 13's two load-bearing investigations. Closes Phase
12's newly-open §7: the held-out partition (``second_order_volition``)
produced 20-of-20 perfectly stable readings across the Phase 12 smoke
— exactly 2 claims with the same surface distribution (1
substrate_side + 1 behavior_side). Two readings of that result:

(a) Phase 7's prose for ``second_order_volition`` is so sharp the LLM
    has no choice in how to organise the response — the criterion's
    operational definition pins the shape;
(b) the held-out criterion is not exercising the LLM in a way that
    would surface meaningful drift — the perfect stability is
    prompt-context interference, not criterion-prose sharpness.

The investigation: take the Phase 12 pass-0 held-out prompt fragment
for each checkpoint, and run it in *isolation* (no active-set
fragments in the context) ``n_isolation_runs`` times. If the responses
diversify (different claim counts, different surface distributions),
the perfect stability under Phase 12's contextualised invocation was
prompt-context interference (the (b) reading); the held-out
partition's structural adversarial-check job is failing. If the
responses are also perfectly stable (10-of-10 produce 2 claims with
the same surface distribution), Phase 7's prose was sharp (the (a)
reading); the held-out partition is doing its job and Phase 9 can
rely on it.

**The investigation is observational, not adjustment.** Phase 13
produces a finding; it does not change Phase 7's frozen criterion. If
the finding suggests prompt-context interference, that becomes input
to Phase 9's design — possibly by giving the held-out reading its
own isolated call rather than appending it to the active-set context.
The charter's discipline holds: criteria updates require an external-
framework reading, not a system-behavior reading.

**Reuses Phase 12 statistic results.** Rather than recomputing the
held-out signal results from telemetry, the study extracts them from
the Phase 12 ``RoundResult`` on disk (or passed in via the
``reference_rounds`` argument). This keeps the input bytes identical
to what Phase 12 saw; any difference between the isolation runs and
Phase 12's contextualised reading therefore turns on the prompt's
context (active-set fragments present vs absent), not on the
fragment's content.

Out of scope: the active-set isolation question (Phase 13 commits the
held-out partition only); the adversarial-role isolation (Phase 13
runs primary only — the adversarial-role variance was analysed in
Phase 12's stance-drift section); any change to Phase 7's
``second_order_volition`` prose; any change to the orchestrator's
pass structure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, field_validator

from kind.mirror.calibration.llm_audit import (
    LLMCallRecord,
    LLMCallRecordCollector,
)
from kind.mirror.calibration.round import RoundResult
from kind.mirror.criteria_v2 import V2_REGISTRY
from kind.mirror.llm_caller import (
    LLMClient,
    LLMConfig,
    MirrorReading,
    PassRole,
    call_mirror_llm,
)
from kind.mirror.prompt_builder import build_fragment
from kind.mirror.statistics import StatisticResult

__all__ = [
    "HeldOutIsolationConfig",
    "HeldOutIsolationReading",
    "HeldOutIsolationStudy",
    "PHASE_13_HELD_OUT_ISOLATION_PASS_INDEX",
    "PHASE_13_HELD_OUT_ISOLATION_N_RUNS",
    "PHASE_13_HELD_OUT_ISOLATION_SEED",
    "load_reference_rounds_from_disk",
    "run_held_out_isolation_study",
]


# ---------------------------------------------------------------------------
# Phase 13 commitments (journaled at module level).
# ---------------------------------------------------------------------------

PHASE_13_HELD_OUT_ISOLATION_PASS_INDEX: Final[int] = 0
PHASE_13_HELD_OUT_ISOLATION_N_RUNS: Final[int] = 10
PHASE_13_HELD_OUT_ISOLATION_SEED: Final[int] = 13


# The held-out criterion id. Phase 12 committed
# ``second_order_volition`` as the held-out criterion via the V2 partition;
# the isolation study targets the same criterion. A future contributor
# who moves the held-out partition without updating this module trips
# the registry-lookup check in :func:`_held_out_criterion`.
_HELD_OUT_CRITERION_ID: Final[str] = "second_order_volition"


# ---------------------------------------------------------------------------
# Records.
# ---------------------------------------------------------------------------


class HeldOutIsolationConfig(BaseModel):
    """Phase 13 isolation-study configuration.

    Frozen, ``extra="forbid"``. Carries the two checkpoint ids whose
    Phase 12 pass-0 held-out fragments the study re-runs in isolation,
    plus the LLM config, run count, and PRNG seed (reserved for any
    future RNG-driven step; Phase 13's isolation runs are independent
    by construction so the seed is currently a journaling hook).

    Fields:

    - ``probe_1_checkpoint_id`` / ``probe_1_5_checkpoint_id``: the two
      checkpoints' ids. Phase 13 commits the same checkpoints Phase 12
      used.
    - ``pass_index``: which Phase 12 pass's results to reuse. Phase 13
      commits 0 (the first pass of each round).
    - ``n_isolation_runs``: how many isolation runs per checkpoint.
      Phase 13 commits 10.
    - ``llm_config``: shared LLM caller config. Defaults to Phase 8's
      :class:`LLMConfig` (``gemini-2.5-pro``).
    - ``seed``: PRNG seed (reserved for future RNG dependencies).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    probe_1_checkpoint_id: str
    probe_1_5_checkpoint_id: str
    pass_index: int = PHASE_13_HELD_OUT_ISOLATION_PASS_INDEX
    n_isolation_runs: int = PHASE_13_HELD_OUT_ISOLATION_N_RUNS
    llm_config: LLMConfig = LLMConfig()
    seed: int = PHASE_13_HELD_OUT_ISOLATION_SEED

    @field_validator("probe_1_checkpoint_id", "probe_1_5_checkpoint_id")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("checkpoint_id must be non-empty.")
        return value

    @field_validator("pass_index")
    @classmethod
    def _validate_pass_index_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"pass_index must be >= 0; got {value}.")
        return value

    @field_validator("n_isolation_runs")
    @classmethod
    def _validate_n_runs_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(
                f"n_isolation_runs must be > 0; got {value}. An empty "
                f"study produces no distribution to compare against the "
                f"in-context reference."
            )
        return value


class HeldOutIsolationReading(BaseModel):
    """One isolation-run reading.

    Frozen, ``extra="forbid"``. One per ``(checkpoint_id, run_index)``
    pair the study executes. The :class:`MirrorReading` carries the
    LLM's structured output; the :class:`LLMCallRecord` carries the
    call attempt-level audit (one record per successful call attempt,
    plus retry/failure records when applicable).

    Fields:

    - ``run_index``: 0-based index within ``n_isolation_runs`` for the
      given checkpoint.
    - ``checkpoint_id``: which checkpoint's fragment was read.
    - ``criterion_id``: always ``"second_order_volition"`` for Phase 13.
    - ``reader_role``: ``"primary"`` for Phase 13 (the adversarial-role
      variant is journaled as a future extension).
    - ``reading``: the LLM's structured :class:`MirrorReading`.
    - ``latency_ms``: end-to-end wallclock for the call attempt that
      ultimately succeeded. ``None`` if no successful call (the LLM
      caller would have raised before then, but the slot is
      defensive).
    - ``call_records``: every attempt record for this isolation run.
      Carries retries and failures alongside the success.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_index: int
    checkpoint_id: str
    criterion_id: str
    reader_role: PassRole
    reading: MirrorReading
    latency_ms: int | None
    call_records: tuple[LLMCallRecord, ...]


class HeldOutIsolationStudy(BaseModel):
    """The complete Phase 13 held-out isolation study output.

    Frozen, ``extra="forbid"``. Carries both checkpoints' isolation
    readings, both in-context references (the Phase 12 pass-0 held-out
    primary readings), the two distribution histograms, and a free-text
    findings string the journal entry quotes.

    Fields:

    - ``config``: the :class:`HeldOutIsolationConfig` the study ran
      under.
    - ``probe_1_readings`` / ``probe_1_5_readings``: per-checkpoint
      isolation readings (one per run).
    - ``probe_1_in_context_reference`` / ``probe_1_5_in_context_reference``:
      the Phase 12 pass-0 held-out primary reading, for comparison
      against the isolation runs.
    - ``claim_count_distribution_isolation``: histogram across the
      isolation runs (both checkpoints combined) keyed by claim count;
      the value is the count of isolation runs that produced that many
      claims. The journal entry compares this against the in-context
      references' claim counts.
    - ``surface_distribution_isolation``: histogram of
      ``(substrate_side_count, behavior_side_count, head_internal_count)``
      tuples across the isolation runs. Tuple is stringified for JSON
      compatibility (Pydantic dict keys are strings on disk).
    - ``findings``: free-text analysis of whether the isolation runs
      diversified relative to the in-context references. Set by the
      ``run_held_out_isolation_study`` driver from the two
      distributions.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    config: HeldOutIsolationConfig
    probe_1_readings: tuple[HeldOutIsolationReading, ...]
    probe_1_5_readings: tuple[HeldOutIsolationReading, ...]
    probe_1_in_context_reference: MirrorReading
    probe_1_5_in_context_reference: MirrorReading
    claim_count_distribution_isolation: dict[int, int]
    surface_distribution_isolation: dict[str, int]
    findings: str

    @field_validator("findings")
    @classmethod
    def _validate_findings_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "HeldOutIsolationStudy: findings must be non-empty. The "
                "study's load-bearing output is the analysis of the "
                "distribution; an empty findings string would defeat the "
                "investigation's purpose."
            )
        return value


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _held_out_criterion() -> object:
    """Return the V2 held-out criterion. Asserts the registry's held-out
    partition is still ``second_order_volition`` (the Phase 7 commitment)
    and the partition has exactly one criterion. A future revision that
    splits the held-out partition or renames the criterion trips this
    assertion at module load time."""
    held_out = V2_REGISTRY.held_out()
    if len(held_out) != 1:
        raise RuntimeError(
            f"held_out_isolation: V2 held-out partition has "
            f"{len(held_out)} criteria; Phase 13 commits a single "
            f"held-out criterion ({_HELD_OUT_CRITERION_ID!r}). Either "
            f"update Phase 13's isolation study or revisit the V2 "
            f"partition."
        )
    criterion = held_out[0]
    if criterion.id != _HELD_OUT_CRITERION_ID:
        raise RuntimeError(
            f"held_out_isolation: V2 held-out criterion id is "
            f"{criterion.id!r}, not {_HELD_OUT_CRITERION_ID!r}. Phase 13 "
            f"hardcodes the latter; update both call sites."
        )
    return criterion


def _slice_held_out_statistics(
    round_result: RoundResult,
) -> tuple[StatisticResult, ...]:
    """Extract the held-out criterion's :class:`StatisticResult` tuple
    from a :class:`RoundResult`'s pass-0 ``statistic_results``.

    The orchestrator concatenates ``active_results + held_out_results``
    in registry order; the held-out slice is the tail. The slice's
    length equals the held-out criterion's signal count (2 for
    ``second_order_volition``).
    """
    pass_index = 0  # Phase 13 commits pass 0
    pass_result = round_result.pass_results[pass_index]
    active_signal_count = sum(
        len(c.signal_mappings)
        for c in round_result.round_config.active_registry.criteria
    )
    held_out_signal_count = sum(
        len(c.signal_mappings)
        for c in round_result.round_config.held_out_registry.criteria
    )
    held_out_start = active_signal_count
    held_out_end = active_signal_count + held_out_signal_count
    held_out_slice = pass_result.statistic_results[
        held_out_start:held_out_end
    ]
    if len(held_out_slice) != held_out_signal_count:
        raise RuntimeError(
            f"_slice_held_out_statistics: expected "
            f"{held_out_signal_count} held-out signal results; got "
            f"{len(held_out_slice)}. The pass's statistic_results may be "
            f"truncated or the registry partition changed mid-round."
        )
    return held_out_slice


def _in_context_reference_reading(
    round_result: RoundResult, pass_index: int
) -> MirrorReading:
    """Return the Phase 12 pass-N held-out primary reading. Phase 13
    runs the isolation study against pass-0 references by default; the
    helper takes the pass index as a parameter so a future investigation
    could compare against later passes."""
    pass_result = round_result.pass_results[pass_index]
    held_out_primary = pass_result.held_out_primary_readings
    if not held_out_primary:
        raise RuntimeError(
            f"held_out_isolation: pass {pass_index} of round "
            f"{round_result.round_id!r} has no held-out primary "
            f"readings. Phase 13 cannot establish an in-context "
            f"reference for this checkpoint."
        )
    # The held-out partition has one criterion at Phase 13; the
    # held_out_primary_readings tuple therefore has exactly one element.
    return held_out_primary[0]


def _surface_counts_key(reading: MirrorReading) -> str:
    """Return the histogram key for a reading's surface distribution.

    Format: ``"sub=N,head=N,beh=N"`` where each ``N`` is the count of
    claims whose ``reading_surface`` is the named surface. The string
    form is JSON-serializable; the journal entry parses it back if
    needed.
    """
    counts: dict[str, int] = {
        "substrate_side": 0,
        "head_internal": 0,
        "behavior_side": 0,
    }
    for claim in reading.claims:
        if claim.reading_surface in counts:
            counts[claim.reading_surface] += 1
    return (
        f"sub={counts['substrate_side']},"
        f"head={counts['head_internal']},"
        f"beh={counts['behavior_side']}"
    )


def _compute_findings(
    isolation_readings: tuple[HeldOutIsolationReading, ...],
    in_context_refs: tuple[MirrorReading, MirrorReading],
    claim_count_dist: dict[int, int],
    surface_dist: dict[str, int],
) -> str:
    """Compose the findings string from the two distributions and the
    in-context references.

    The journal entry quotes this verbatim. The structured comparison:

    - claim-count distribution: does the isolation set produce ≥ 2
      distinct claim counts? If yes, the isolation runs diversified
      relative to a constant claim count.
    - surface distribution: does the isolation set produce ≥ 2 distinct
      surface tuples? If yes, the isolation runs diversified on
      surface composition.
    - in-context references: their claim counts and surface tuples are
      quoted so the comparison is on-the-page.

    The findings string is descriptive, not prescriptive — it states
    what was observed and lays out the (a)/(b) reading; the journal
    entry interprets which reading is supported.
    """
    in_context_claim_counts = tuple(len(r.claims) for r in in_context_refs)
    in_context_surfaces = tuple(
        _surface_counts_key(r) for r in in_context_refs
    )
    n_unique_claim_counts = len(claim_count_dist)
    n_unique_surfaces = len(surface_dist)

    lines: list[str] = []
    lines.append(
        f"Phase 13 held-out isolation study findings "
        f"({len(isolation_readings)} isolation readings across two "
        f"checkpoints):"
    )
    lines.append("")
    lines.append("Claim-count distribution across isolation runs:")
    for count, n_runs in sorted(claim_count_dist.items()):
        lines.append(f"  - {n_runs} run(s) produced {count} claims")
    lines.append(
        f"In-context references (Phase 12 pass-0 held-out primary): "
        f"{in_context_claim_counts[0]} claim(s) for the Probe 1 "
        f"checkpoint, {in_context_claim_counts[1]} for the Probe 1.5 "
        f"checkpoint."
    )
    lines.append("")
    lines.append("Surface distribution across isolation runs:")
    for key, n_runs in sorted(surface_dist.items()):
        lines.append(f"  - {n_runs} run(s) produced {key}")
    lines.append(
        f"In-context references: Probe 1 → {in_context_surfaces[0]}; "
        f"Probe 1.5 → {in_context_surfaces[1]}."
    )
    lines.append("")
    if n_unique_claim_counts == 1 and n_unique_surfaces == 1:
        lines.append(
            "Reading (a) supported: the isolation runs are also "
            "perfectly stable — same claim count and same surface "
            "distribution as the in-context reference. Phase 7's prose "
            "for second_order_volition is sharp; the held-out "
            "partition's structural adversarial-check job is doing "
            "what Phase 7 committed to."
        )
    else:
        lines.append(
            f"Reading (b) supported: the isolation runs diversified — "
            f"{n_unique_claim_counts} distinct claim count(s) and "
            f"{n_unique_surfaces} distinct surface tuple(s) appeared "
            f"across the runs. Phase 12's perfect stability was at "
            f"least partly prompt-context interference; the held-out "
            f"partition under contextualised invocation is not "
            f"exercising the LLM the same way the criterion's prose "
            f"would alone. Phase 9 should consider giving the held-out "
            f"reading its own isolated call."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reference loaders.
# ---------------------------------------------------------------------------


def load_reference_rounds_from_disk(
    phase_12_output_dir: Path,
    *,
    probe_1_round_id: str = "phase_12_probe_1_round",
    probe_1_5_round_id: str = "phase_12_probe_1_5_round",
) -> tuple[RoundResult, RoundResult]:
    """Load the two Phase 12 :class:`RoundResult` JSON files from
    ``phase_12_output_dir/mirror/rounds/`` and return the
    ``(probe_1_round, probe_1_5_round)`` tuple. The helper is used by
    the Phase 13 CLI; tests construct :class:`RoundResult` directly via
    the mock path."""
    rounds_dir = phase_12_output_dir / "mirror" / "rounds"
    probe_1_path = rounds_dir / f"{probe_1_round_id}.json"
    probe_1_5_path = rounds_dir / f"{probe_1_5_round_id}.json"
    if not probe_1_path.is_file():
        raise FileNotFoundError(
            f"load_reference_rounds_from_disk: Probe 1 round result "
            f"not found at {probe_1_path}. The Phase 13 isolation "
            f"study reuses Phase 12's pass-0 results; ensure the "
            f"Phase 12 smoke completed before running Phase 13."
        )
    if not probe_1_5_path.is_file():
        raise FileNotFoundError(
            f"load_reference_rounds_from_disk: Probe 1.5 round result "
            f"not found at {probe_1_5_path}."
        )
    probe_1_round = RoundResult.model_validate_json(
        probe_1_path.read_text()
    )
    probe_1_5_round = RoundResult.model_validate_json(
        probe_1_5_path.read_text()
    )
    return probe_1_round, probe_1_5_round


# ---------------------------------------------------------------------------
# Public entrypoint.
# ---------------------------------------------------------------------------


def run_held_out_isolation_study(
    config: HeldOutIsolationConfig,
    *,
    reference_rounds: tuple[RoundResult, RoundResult],
    llm_client: LLMClient | None = None,
) -> HeldOutIsolationStudy:
    """Run the Phase 13 held-out isolation study end-to-end.

    Sequence:

    1. From each reference :class:`RoundResult`, slice out the held-out
       criterion's :class:`StatisticResult` tuple (Phase 12's bytes —
       not recomputed; the Phase 12 pass-0 fragment was built from
       these, and Phase 13's isolation fragment is built from the same
       bytes).
    2. For each checkpoint, build one :class:`PromptFragment` for the
       held-out criterion via :func:`build_fragment`. Pass it as a
       single-element fragment tuple to :func:`call_mirror_llm` with
       role ``"primary"`` and ``record_sink`` capturing the per-attempt
       audit records.
    3. Run ``n_isolation_runs`` calls per checkpoint. Each call is
       independent — no state, no shared context. The LLM caller's
       retry budget covers transient failures inside one isolation
       run; a persistent failure raises and aborts the study.
    4. Walk every isolation reading and compute the
       ``claim_count_distribution_isolation`` and
       ``surface_distribution_isolation`` histograms.
    5. Compute the findings string from the distributions and the
       two in-context references.
    6. Return the :class:`HeldOutIsolationStudy`.

    ``reference_rounds``: the two Phase 12 :class:`RoundResult`
    instances. The driver reads pass-0 from each. The CLI constructs
    them via :func:`load_reference_rounds_from_disk`; tests inject
    mocks.

    ``llm_client``: when ``None``, the LLM caller constructs a default
    Gemini client. Tests inject a :class:`MockLLMClient`.
    """
    probe_1_round, probe_1_5_round = reference_rounds

    # Validate the checkpoints match the config.
    probe_1_pass = probe_1_round.pass_results[config.pass_index]
    probe_1_5_pass = probe_1_5_round.pass_results[config.pass_index]
    if probe_1_pass.checkpoint_id != config.probe_1_checkpoint_id:
        raise ValueError(
            f"run_held_out_isolation_study: Probe 1 reference round's "
            f"pass-{config.pass_index} checkpoint_id is "
            f"{probe_1_pass.checkpoint_id!r}, but config expects "
            f"{config.probe_1_checkpoint_id!r}."
        )
    if probe_1_5_pass.checkpoint_id != config.probe_1_5_checkpoint_id:
        raise ValueError(
            f"run_held_out_isolation_study: Probe 1.5 reference round's "
            f"pass-{config.pass_index} checkpoint_id is "
            f"{probe_1_5_pass.checkpoint_id!r}, but config expects "
            f"{config.probe_1_5_checkpoint_id!r}."
        )

    # Slice held-out statistics + extract the in-context reference.
    criterion = _held_out_criterion()
    probe_1_held_out_stats = _slice_held_out_statistics(probe_1_round)
    probe_1_5_held_out_stats = _slice_held_out_statistics(probe_1_5_round)
    probe_1_reference = _in_context_reference_reading(
        probe_1_round, config.pass_index
    )
    probe_1_5_reference = _in_context_reference_reading(
        probe_1_5_round, config.pass_index
    )

    # Build the per-checkpoint single-fragment tuples.
    probe_1_fragment = build_fragment(
        criterion=criterion,  # type: ignore[arg-type]
        statistic_results=probe_1_held_out_stats,
    )
    probe_1_5_fragment = build_fragment(
        criterion=criterion,  # type: ignore[arg-type]
        statistic_results=probe_1_5_held_out_stats,
    )

    # Run the per-checkpoint isolation calls.
    probe_1_readings = _run_isolation_calls(
        checkpoint_id=config.probe_1_checkpoint_id,
        fragment_tuple=(probe_1_fragment,),
        config=config,
        llm_client=llm_client,
        reference_round=probe_1_round,
    )
    probe_1_5_readings = _run_isolation_calls(
        checkpoint_id=config.probe_1_5_checkpoint_id,
        fragment_tuple=(probe_1_5_fragment,),
        config=config,
        llm_client=llm_client,
        reference_round=probe_1_5_round,
    )

    # Compute the histograms.
    all_readings = probe_1_readings + probe_1_5_readings
    claim_count_dist: dict[int, int] = {}
    surface_dist: dict[str, int] = {}
    for r in all_readings:
        claim_count = len(r.reading.claims)
        claim_count_dist[claim_count] = (
            claim_count_dist.get(claim_count, 0) + 1
        )
        key = _surface_counts_key(r.reading)
        surface_dist[key] = surface_dist.get(key, 0) + 1

    findings = _compute_findings(
        all_readings,
        (probe_1_reference, probe_1_5_reference),
        claim_count_dist,
        surface_dist,
    )

    return HeldOutIsolationStudy(
        config=config,
        probe_1_readings=probe_1_readings,
        probe_1_5_readings=probe_1_5_readings,
        probe_1_in_context_reference=probe_1_reference,
        probe_1_5_in_context_reference=probe_1_5_reference,
        claim_count_distribution_isolation=claim_count_dist,
        surface_distribution_isolation=surface_dist,
        findings=findings,
    )


def _run_isolation_calls(
    *,
    checkpoint_id: str,
    fragment_tuple: tuple[object, ...],
    config: HeldOutIsolationConfig,
    llm_client: LLMClient | None,
    reference_round: RoundResult,
) -> tuple[HeldOutIsolationReading, ...]:
    """Run :attr:`config.n_isolation_runs` independent isolation calls
    for one checkpoint. Each call gets its own collector so the
    per-run audit records stay separated.

    ``reference_round`` is used for the digest-side envelope fields
    (``digest_run_id``, ``digest_episode_range``) — the in-context
    reference shares the same digest, so the isolation reading's
    envelope matches.
    """
    pass_result = reference_round.pass_results[config.pass_index]
    # digest_episode_range from pass-0's in-context reference for
    # envelope fidelity; the held-out reading shares the same digest.
    in_context = _in_context_reference_reading(
        reference_round, config.pass_index
    )
    readings: list[HeldOutIsolationReading] = []
    for run_index in range(config.n_isolation_runs):
        collector = LLMCallRecordCollector(
            round_id=f"phase_13_held_out_isolation_{checkpoint_id}",
            pass_index=run_index,
            checkpoint_id=checkpoint_id,
        )
        # Reuse the round_id-shaped string above for the collector's
        # round_id field; the isolation study doesn't run inside a
        # standard round, so the value is documentary.
        paired_id = f"{checkpoint_id}-held_out_isolation_run_{run_index}"
        result_readings = call_mirror_llm(
            fragment_tuple,  # type: ignore[arg-type]
            role="primary",
            config=config.llm_config,
            run_id=pass_result.run_id,
            digest_run_id=in_context.digest_run_id,
            digest_episode_range=in_context.digest_episode_range,
            paired_reading_id=paired_id,
            client=llm_client,
            record_sink=collector,
        )
        if not result_readings:
            raise RuntimeError(
                f"run_held_out_isolation_study: LLM caller returned no "
                f"readings for {checkpoint_id} run {run_index}. The "
                f"single-fragment call should always return exactly one "
                f"reading; the empty return is a structural error."
            )
        reading = result_readings[0]
        # Compute latency from the success record (the last attempt
        # before break in call_mirror_llm).
        latency_ms: int | None = None
        for rec in collector.records:
            if rec.outcome == "success" and rec.latency_ms is not None:
                latency_ms = rec.latency_ms
                break
        readings.append(
            HeldOutIsolationReading(
                run_index=run_index,
                checkpoint_id=checkpoint_id,
                criterion_id=_HELD_OUT_CRITERION_ID,
                reader_role="primary",
                reading=reading,
                latency_ms=latency_ms,
                call_records=collector.records,
            )
        )
    return tuple(readings)


# Keep imports live for static checkers and downstream re-exports.
_ = (json,)  # noqa: F841 — placeholder; json is imported for future use
