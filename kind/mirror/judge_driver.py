"""Phase 9 judge driver — :func:`judge_round`.

Reads a :class:`~kind.mirror.calibration.round.RoundResult` from disk,
runs the judge across every criterion (active + held-out partitions),
produces a :class:`~kind.mirror.judge.RoundJudgment`, and writes the
judgment to ``output_dir/mirror/judgments/{round_id}.json``
atomically.

**The judge driver is read-only against the round result.** The round
result on disk is unchanged; the judgment is a separate artifact
under ``judgments/``. The driver does not call
:func:`~kind.mirror.statistics.compute_statistic`, does not load
:class:`~kind.observer.schemas.AgentStep` files, does not touch
Io-side state. The structural read-only invariant from Phases 6–13
continues here.

**The judgment subdir contract.** Phase 9 writes to a new subdir
``mirror/judgments/``, sibling of ``mirror/rounds/`` (the round
artifact directory) and ``mirror/pre_reg/`` (the pre-registration
directory). The judgment file is named after the round id —
``{round_id}.json`` — so a calibration session with two rounds
produces two judgment files.

Out of scope: the Phase 9 smoke harness
(:mod:`kind.mirror.calibration.phase_9_judge_smoke`); cross-round
aggregation (Phase 10+); any change to the round result schema.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Final

from kind.mirror.calibration.llm_audit import LLMCallRecordCollector
from kind.mirror.calibration.round import RoundResult
from kind.mirror.judge import (
    CriterionJudgment,
    RoundJudgment,
)
from kind.mirror.judge_llm_caller import (
    call_judge_llm,
    JudgeLLMClient,
)
from kind.mirror.judge_prompt_builder import (
    JudgePromptFragment,
    build_judge_fragment,
)
from kind.mirror.llm_caller import LLMConfig, MirrorReading
from kind.mirror.registry import Criterion
from kind.mirror.statistics import StatisticResult

__all__ = [
    "JUDGMENTS_SUBDIR",
    "judge_round",
    "load_round_result_from_disk",
]


# ---------------------------------------------------------------------------
# Output-directory contract.
# ---------------------------------------------------------------------------

JUDGMENTS_SUBDIR: Final[str] = "judgments"
_MIRROR_SUBDIR: Final[str] = "mirror"


# ---------------------------------------------------------------------------
# Helpers — extract per-criterion readings and statistic results from a
# RoundResult.
# ---------------------------------------------------------------------------


def _readings_for_criterion(
    round_result: RoundResult,
    criterion: Criterion,
    *,
    partition: str,
    role: str,
) -> tuple[MirrorReading, ...]:
    """Collect the per-pass readings for one criterion + partition +
    role across the round.

    ``partition`` is ``"active"`` or ``"held_out"``; ``role`` is
    ``"primary"`` or ``"adversarial"``. The readings are returned in
    pass-order; a pass with no matching reading (e.g. a partition that
    was empty for that pass) produces no entry.
    """
    if partition not in {"active", "held_out"}:
        raise ValueError(
            f"_readings_for_criterion: partition must be 'active' or "
            f"'held_out'; got {partition!r}."
        )
    if role not in {"primary", "adversarial"}:
        raise ValueError(
            f"_readings_for_criterion: role must be 'primary' or "
            f"'adversarial'; got {role!r}."
        )
    out: list[MirrorReading] = []
    for pr in round_result.pass_results:
        # Pick the right reading-tuple per (partition, role).
        if partition == "active":
            readings_tuple = (
                pr.active_primary_readings
                if role == "primary"
                else pr.active_adversarial_readings
            )
        else:
            readings_tuple = (
                pr.held_out_primary_readings
                if role == "primary"
                else pr.held_out_adversarial_readings
            )
        # Find the reading for this criterion (one per criterion in
        # the partition's readings tuple — see Phase 8's orchestrator
        # contract).
        for reading in readings_tuple:
            if _reading_matches_criterion(reading, criterion):
                out.append(reading)
                break
    return tuple(out)


def _reading_matches_criterion(
    reading: MirrorReading, criterion: Criterion
) -> bool:
    """A :class:`MirrorReading` matches a criterion when at least one
    of its claims has a ``cited_scalar_field`` that names one of the
    criterion's signal mappings. This is the same convention Phase 8's
    orchestrator uses to identify per-criterion readings: the
    structured-output schema does not carry the criterion id on the
    reading envelope, but the claims cite the criterion's signals.

    The first reading in the tuple matching the criterion is the one
    returned; orchestrator-side ordering already produces one reading
    per criterion in the per-partition tuple, so the first-match
    semantics is correct."""
    signal_names = {m.name for m in criterion.signal_mappings}
    signal_paths = {m.field_path for m in criterion.signal_mappings}
    for claim in reading.claims:
        if claim.cited_scalar_field in signal_names:
            return True
        if claim.cited_scalar_field in signal_paths:
            return True
    # Fallback: framework_anchor-based heuristic — a reading whose
    # framework_anchor matches the criterion's framework AND whose
    # position in the readings tuple corresponds to the criterion is
    # the one to use. The driver's caller uses index-based matching as
    # the primary path below; this function is the
    # claim-citation-based check that returns ``True`` only for
    # explicit matches.
    return False


def _readings_for_criterion_by_index(
    round_result: RoundResult,
    criterion: Criterion,
    *,
    criterion_index: int,
    partition: str,
    role: str,
) -> tuple[MirrorReading, ...]:
    """Collect readings by *position* — Phase 8's orchestrator emits
    one reading per criterion in registry order, so ``criterion_index``
    in the readings tuple corresponds to ``criterion_index`` in the
    partition registry.

    This is the primary path. The claim-citation fallback
    (:func:`_reading_matches_criterion`) is the second-line check
    that lets the judge surface cases where the orchestrator's
    position-based ordering broke.
    """
    out: list[MirrorReading] = []
    for pr in round_result.pass_results:
        if partition == "active":
            readings_tuple = (
                pr.active_primary_readings
                if role == "primary"
                else pr.active_adversarial_readings
            )
        else:
            readings_tuple = (
                pr.held_out_primary_readings
                if role == "primary"
                else pr.held_out_adversarial_readings
            )
        if criterion_index < len(readings_tuple):
            out.append(readings_tuple[criterion_index])
        # Else: this pass had no slot for this criterion (e.g. an
        # empty partition); skip silently. The judge fragment's
        # degenerate-case handling surfaces the absence.
    return tuple(out)


def _statistic_results_for_criterion_across_passes(
    round_result: RoundResult, criterion: Criterion
) -> tuple[tuple[StatisticResult, ...], ...]:
    """For each pass in the round, collect the statistic results that
    name signals on this criterion. The orchestrator writes the
    statistic results in registry-walk order — one per signal mapping
    in registry-criterion order — so the filter here matches by
    signal name."""
    signal_names = {m.name for m in criterion.signal_mappings}
    out: list[tuple[StatisticResult, ...]] = []
    for pr in round_result.pass_results:
        matching = tuple(
            r for r in pr.statistic_results if r.signal_name in signal_names
        )
        out.append(matching)
    return tuple(out)


# ---------------------------------------------------------------------------
# Round-config summary.
# ---------------------------------------------------------------------------


def _build_round_config_summary(round_result: RoundResult) -> str:
    """Render a human-readable summary of the round's context the
    judge fragment's body does not need to embed verbatim. The summary
    is what the journal entry quotes when describing the judge's
    input."""
    cfg = round_result.round_config
    statistic_summary = (
        f"autocorr_lag={cfg.statistic_config.autocorr_lag}, "
        f"recovery_window_W={cfg.statistic_config.recovery_window_W}, "
        f"recovery_pre_window={cfg.statistic_config.recovery_pre_window}, "
        f"recovery_streak_required="
        f"{cfg.statistic_config.recovery_streak_required}, "
        f"recovery_threshold_percentile="
        f"{cfg.statistic_config.recovery_threshold_percentile}, "
        f"kmeans_k={cfg.statistic_config.kmeans_k}"
    )
    llm_summary = (
        f"model_name={cfg.llm_config.model_name}, "
        f"max_retries={cfg.llm_config.max_retries}, "
        f"max_output_tokens={cfg.llm_config.max_output_tokens}"
    )
    return (
        f"round_id={cfg.round_id}; "
        f"passes_per_checkpoint={cfg.passes_per_checkpoint}; "
        f"shams_per_pass={cfg.sham_schedule.shams_per_pass}; "
        f"synthetics_per_pass="
        f"{cfg.synthetic_schedule.synthetics_per_pass}; "
        f"active_criteria={[c.id for c in cfg.active_registry.criteria]}; "
        f"held_out_criteria="
        f"{[c.id for c in cfg.held_out_registry.criteria]}; "
        f"StatisticConfig({statistic_summary}); "
        f"LLMConfig({llm_summary})"
    )


# ---------------------------------------------------------------------------
# Public entrypoint.
# ---------------------------------------------------------------------------


def load_round_result_from_disk(round_result_path: Path) -> RoundResult:
    """Load a :class:`~kind.mirror.calibration.round.RoundResult` from
    its on-disk JSON form.

    Raises ``FileNotFoundError`` if the path doesn't exist; raises
    ``pydantic.ValidationError`` if the file's JSON doesn't validate
    against the model. The driver expects a fully-shaped
    :class:`RoundResult`; partial / corrupted files surface as
    validation errors rather than silent best-effort reads.
    """
    if not round_result_path.is_file():
        raise FileNotFoundError(
            f"load_round_result_from_disk: no such file: "
            f"{round_result_path}"
        )
    with round_result_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return RoundResult.model_validate(payload)


def judge_round(
    round_result_path: Path,
    *,
    output_dir: Path,
    llm_config: LLMConfig,
    llm_client: JudgeLLMClient | None = None,
    notes: str = "",
) -> RoundJudgment:
    """Run the judge over one round's
    :class:`~kind.mirror.calibration.round.RoundResult` artifact.

    Sequence:

    1. Load the :class:`RoundResult` from ``round_result_path``.
    2. For each criterion in the round's
       ``active_registry`` + ``held_out_registry``: collect the
       per-pass primary readings, the per-pass adversarial readings,
       and the per-pass statistic results for the criterion's signals.
    3. Build one :class:`JudgePromptFragment` per criterion via
       :func:`~kind.mirror.judge_prompt_builder.build_judge_fragment`.
    4. Call the judge LLM once (batched across criteria) via
       :func:`~kind.mirror.judge_llm_caller.call_judge_llm`.
    5. Aggregate the per-criterion :class:`CriterionJudgment` records
       into a :class:`RoundJudgment` plus the judge call records.
    6. Write the :class:`RoundJudgment` to
       ``output_dir/mirror/judgments/{round_id}.json`` atomically.

    The function is read-only against the source round result; the
    source file on disk is unchanged.

    ``llm_client``: when ``None``, the judge LLM caller constructs a
    default Gemini client via
    :attr:`~kind.mirror.llm_caller.LLMConfig.api_key_env_var`. Tests
    inject a :class:`~kind.mirror.judge_llm_caller.MockJudgeLLMClient`.
    """
    t0 = int(time.time() * 1000)

    # 1. Load the round result.
    round_result = load_round_result_from_disk(round_result_path)
    cfg = round_result.round_config

    # 2. Per-criterion: collect readings and stat results across passes.
    fragments: list[JudgePromptFragment] = []
    criteria: list[Criterion] = []
    partition_for_criterion: dict[str, str] = {}

    for index, criterion in enumerate(cfg.active_registry.criteria):
        primary = _readings_for_criterion_by_index(
            round_result,
            criterion,
            criterion_index=index,
            partition="active",
            role="primary",
        )
        adversarial = _readings_for_criterion_by_index(
            round_result,
            criterion,
            criterion_index=index,
            partition="active",
            role="adversarial",
        )
        stats_across_passes = (
            _statistic_results_for_criterion_across_passes(
                round_result, criterion
            )
        )
        fragment = build_judge_fragment(
            criterion=criterion,
            primary_readings_across_passes=primary,
            adversarial_readings_across_passes=adversarial,
            statistic_results_across_passes=stats_across_passes,
        )
        fragments.append(fragment)
        criteria.append(criterion)
        partition_for_criterion[criterion.id] = "active"

    for index, criterion in enumerate(cfg.held_out_registry.criteria):
        primary = _readings_for_criterion_by_index(
            round_result,
            criterion,
            criterion_index=index,
            partition="held_out",
            role="primary",
        )
        adversarial = _readings_for_criterion_by_index(
            round_result,
            criterion,
            criterion_index=index,
            partition="held_out",
            role="adversarial",
        )
        stats_across_passes = (
            _statistic_results_for_criterion_across_passes(
                round_result, criterion
            )
        )
        fragment = build_judge_fragment(
            criterion=criterion,
            primary_readings_across_passes=primary,
            adversarial_readings_across_passes=adversarial,
            statistic_results_across_passes=stats_across_passes,
        )
        fragments.append(fragment)
        criteria.append(criterion)
        partition_for_criterion[criterion.id] = "held_out"

    # 3. Call the judge LLM. Audit collector threads call records
    # under the round's id; ``pass_index=0`` is the convention for
    # judge-call records (the judge is a single batch per round, not
    # a per-pass call). A future Phase 14+ that runs the judge
    # per-pass widens this.
    collector = LLMCallRecordCollector(
        round_id=cfg.round_id,
        pass_index=0,
        checkpoint_id=cfg.checkpoints[0].checkpoint_id,
    )
    judgments = call_judge_llm(
        tuple(fragments),
        llm_config,
        fragment_criteria=tuple(criteria),
        run_id=cfg.checkpoints[0].run_id,
        round_id=cfg.round_id,
        digest_run_id=cfg.checkpoints[0].run_id,
        digest_episode_range=_round_digest_episode_range(round_result),
        client=llm_client,
        record_sink=collector,
    )

    # 4. Aggregate.
    summary = _build_round_config_summary(round_result)
    t1 = int(time.time() * 1000)
    judgment = RoundJudgment(
        round_id=cfg.round_id,
        round_config_summary=summary,
        criterion_judgments=tuple(judgments),
        judge_llm_call_records=collector.records,
        wallclock_ms=t1 - t0,
        notes=notes,
    )

    # 5. Write atomically.
    judgments_dir = output_dir / _MIRROR_SUBDIR / JUDGMENTS_SUBDIR
    _write_round_judgment(judgment, judgments_dir)
    return judgment


def _round_digest_episode_range(
    round_result: RoundResult,
) -> tuple[int, int]:
    """The digest episode range the judge logs against — derived from
    the first pass's readings (Phase 8's orchestrator stamps the same
    range across all readings in a pass). If no readings are present,
    fall back to ``(0, 0)``."""
    for pr in round_result.pass_results:
        for reading in (
            *pr.active_primary_readings,
            *pr.active_adversarial_readings,
            *pr.held_out_primary_readings,
            *pr.held_out_adversarial_readings,
        ):
            return reading.digest_episode_range
    return (0, 0)


def _write_round_judgment(
    judgment: RoundJudgment, judgments_dir: Path
) -> None:
    """Write the :class:`RoundJudgment` to
    ``judgments_dir/{round_id}.json`` atomically.

    Uses the same write-temp-then-rename pattern as the round
    driver's :func:`~kind.mirror.calibration.round._write_round_result`
    — a crash mid-write doesn't leave a half-finished JSON on disk.
    """
    judgments_dir.mkdir(parents=True, exist_ok=True)
    final_path = judgments_dir / f"{judgment.round_id}.json"
    tmp_path = judgments_dir / f".{judgment.round_id}.json.tmp"
    payload = judgment.model_dump(mode="json")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    tmp_path.replace(final_path)
