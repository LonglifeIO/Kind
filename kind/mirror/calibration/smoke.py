"""Phase 12 smoke harness — :func:`run_phase_12_smoke` and
:class:`Phase12SmokeResult`.

This is the load-bearing engineering exercise the plan calls Phase 12:
the first time the full mirror execution plane runs against the real
Gemini API, against real Probe 1 + Probe 1.5 checkpoint telemetry. The
readings it produces are not load-bearing scientific findings about Io;
they are load-bearing engineering findings about the mirror.

**Phase 12 commitments (journaled at module-level constants).** Two
rounds:

- Round 1: one Probe 1 checkpoint, 5 passes, the Phase 8 default
  :class:`~kind.mirror.statistics.StatisticConfig`, the Phase 8 default
  :class:`~kind.mirror.llm_caller.LLMConfig`
  (``gemini-2.5-pro``, 8192 max output tokens, 3 max retries), 2 real
  perturbations + 1 sham per pass, sham-schedule seed 42.
- Round 2: one Probe 1.5 checkpoint, same configuration except the
  checkpoint id and sham-schedule seed 43.

Both rounds run; :func:`~kind.mirror.calibration.round_diff.compute_round_diff`
is called with no rationales (the configs should be bit-identical on
the diffed dimensions); the diff is the structural verification that
the rounds are directly comparable.

**The mirror does not see sham labels in its prompt.** The shams are
synthesized orchestrator-side per the Phase 12 plan; the prompt cites
the merged :class:`~kind.mirror.perturbation_align.PerturbationTimeline`
for step-range citations only.

**The LLM-call audit is the round-spanning artifact.** Each pass's
:class:`~kind.mirror.calibration.llm_audit.LLMCallRecordCollector`
appends records; the records flow from
:class:`~kind.mirror.calibration.round.RoundResult` (per-round) up to
:class:`Phase12SmokeResult` (cross-round). The audit is built from the
concatenated records via
:meth:`~kind.mirror.calibration.llm_audit.LLMCallAudit.from_records`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Final

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from kind.mirror.calibration.llm_audit import LLMCallAudit
from kind.mirror.calibration.round import (
    CheckpointSpec,
    RoundConfig,
    RoundResult,
    run_round,
)
from kind.mirror.calibration.round_diff import RoundDiff, compute_round_diff
from kind.mirror.calibration.sham_schedule import generate_sham_schedule
from kind.mirror.llm_caller import LLMClient, LLMConfig
from kind.mirror.statistics import StatisticConfig

__all__ = [
    "PHASE_12_PASSES_PER_CHECKPOINT",
    "PHASE_12_REAL_PERTURBATIONS_PER_PASS",
    "PHASE_12_SHAMS_PER_PASS",
    "PHASE_12_PROBE_1_SEED",
    "PHASE_12_PROBE_1_5_SEED",
    "PHASE_12_PROBE_1_ROUND_ID",
    "PHASE_12_PROBE_1_5_ROUND_ID",
    "Phase12SmokeResult",
    "run_phase_12_smoke",
]


# ---------------------------------------------------------------------------
# Phase 12 commitments (journaled at module-level).
# ---------------------------------------------------------------------------

PHASE_12_PASSES_PER_CHECKPOINT: Final[int] = 5
PHASE_12_REAL_PERTURBATIONS_PER_PASS: Final[int] = 2
PHASE_12_SHAMS_PER_PASS: Final[int] = 1
PHASE_12_PROBE_1_SEED: Final[int] = 42
PHASE_12_PROBE_1_5_SEED: Final[int] = 43
PHASE_12_PROBE_1_ROUND_ID: Final[str] = "phase_12_probe_1_round"
PHASE_12_PROBE_1_5_ROUND_ID: Final[str] = "phase_12_probe_1_5_round"

_AGENT_STEP_SUBDIR: Final[str] = "agent_step"


# ---------------------------------------------------------------------------
# Result.
# ---------------------------------------------------------------------------


class Phase12SmokeResult(BaseModel):
    """The full Phase 12 smoke output.

    Frozen, ``extra="forbid"``. Carries both rounds' results, the diff
    record between them, the cross-round LLM-call audit, the wallclock,
    and free-text notes.

    On-disk form: ``output_dir/mirror/phase_12_smoke_result.json``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "0.1.0"
    probe_1_round: RoundResult
    probe_1_5_round: RoundResult
    round_diff: RoundDiff
    llm_call_audit: LLMCallAudit
    wallclock_ms: int
    notes: str


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _count_agent_step_rows(run_dir: Path) -> int:
    """Count the rows across every ``agent_step/shard-*.parquet`` in
    ``run_dir/telemetry/``. The count is the ``telemetry_length`` the
    sham scheduler uses to bound its placement window.

    Returns 0 if the directory is missing or empty; the caller validates
    the count is large enough for the sham placement.
    """
    shard_dir = run_dir / "telemetry" / _AGENT_STEP_SUBDIR
    if not shard_dir.is_dir():
        return 0
    total = 0
    for shard in sorted(shard_dir.glob("shard-*.parquet")):
        table = pq.read_table(str(shard))  # type: ignore[no-untyped-call]
        total += table.num_rows
    return total


def _build_round_config(
    *,
    round_id: str,
    checkpoint: CheckpointSpec,
    seed: int,
    telemetry_length: int,
    llm_config: LLMConfig,
    notes: str,
) -> RoundConfig:
    """Construct a Phase-12-shaped :class:`RoundConfig` for one round.

    The shared commitments (``passes_per_checkpoint``,
    ``real_perturbations_per_pass``, ``shams_per_pass``) come from the
    module-level constants. The seed and checkpoint differ between
    Round 1 and Round 2.
    """
    sham_schedule = generate_sham_schedule(
        checkpoint_ids=(checkpoint.checkpoint_id,),
        passes_per_checkpoint=PHASE_12_PASSES_PER_CHECKPOINT,
        real_perturbations_per_pass=PHASE_12_REAL_PERTURBATIONS_PER_PASS,
        shams_per_pass=PHASE_12_SHAMS_PER_PASS,
        telemetry_length=telemetry_length,
        seed=seed,
    )
    return RoundConfig(
        round_id=round_id,
        checkpoints=(checkpoint,),
        passes_per_checkpoint=PHASE_12_PASSES_PER_CHECKPOINT,
        statistic_config=StatisticConfig(),
        llm_config=llm_config,
        sham_schedule=sham_schedule,
        notes=notes,
    )


def _write_smoke_result(result: Phase12SmokeResult, mirror_dir: Path) -> None:
    """Write the :class:`Phase12SmokeResult` to disk atomically."""
    mirror_dir.mkdir(parents=True, exist_ok=True)
    final_path = mirror_dir / "phase_12_smoke_result.json"
    tmp_path = mirror_dir / ".phase_12_smoke_result.json.tmp"
    payload = result.model_dump(mode="json")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    tmp_path.replace(final_path)


# ---------------------------------------------------------------------------
# Public entrypoint.
# ---------------------------------------------------------------------------


def run_phase_12_smoke(
    probe_1_checkpoint: CheckpointSpec,
    probe_1_5_checkpoint: CheckpointSpec,
    *,
    output_dir: Path,
    llm_api_key_env_var: str = "GEMINI_API_KEY",
    llm_client: LLMClient | None = None,
    notes: str = "",
) -> Phase12SmokeResult:
    """Run the full Phase 12 smoke end-to-end against the real Gemini API.

    The plan's prototype lists ``probe_1_checkpoint_id`` and
    ``probe_1_5_checkpoint_id`` as the primary parameters; this
    implementation takes :class:`CheckpointSpec` for each (carrying
    ``run_id``, ``checkpoint_id``, ``run_dir``) because the round
    driver needs the on-disk run directory to find telemetry. The CLI
    at :file:`scripts/run_phase_12_smoke.py` accepts the corresponding
    flat arguments and constructs the specs here.

    Sequence:

    1. Count telemetry rows per checkpoint (the sham scheduler's
       placement window).
    2. Build Round 1's :class:`RoundConfig` (Probe 1 checkpoint, seed
       42) and Round 2's (Probe 1.5 checkpoint, seed 43).
    3. Run Round 1, then Round 2 — sequentially, not in parallel. The
       sequential order keeps the LLM-call audit's records in a stable
       order and avoids interleaving the two rounds' API rate limits.
    4. Compute the :class:`RoundDiff` (should be empty on the diffed
       dimensions — the configs are bit-identical except for fields
       outside the diff scope, namely ``checkpoints``, ``sham_schedule``,
       and ``round_id``).
    5. Aggregate the cross-round
       :class:`~kind.mirror.calibration.llm_audit.LLMCallAudit` from
       the concatenated per-round records.
    6. Write :class:`Phase12SmokeResult` to disk atomically.

    ``llm_client``: if not provided, the orchestrator's default
    :class:`~kind.mirror.llm_caller._GeminiLLMClient` is constructed
    inside :func:`~kind.mirror.llm_caller.call_mirror_llm` using
    :envvar:`GEMINI_API_KEY` (or ``llm_api_key_env_var``). Tests inject
    a :class:`~kind.mirror.llm_caller.MockLLMClient`.

    The ``llm_api_key_env_var`` argument is honored only when
    ``llm_client`` is ``None`` (the real-API path); the
    :class:`LLMConfig` constructed for both rounds picks it up via the
    ``api_key_env_var`` field.
    """
    t0 = int(time.time() * 1000)

    # 1. Telemetry-length per checkpoint.
    probe_1_length = _count_agent_step_rows(probe_1_checkpoint.run_dir)
    probe_1_5_length = _count_agent_step_rows(probe_1_5_checkpoint.run_dir)
    if probe_1_length < 3:
        raise ValueError(
            f"run_phase_12_smoke: Probe 1 checkpoint at "
            f"{probe_1_checkpoint.run_dir} has telemetry_length="
            f"{probe_1_length}, which is below the sham scheduler's "
            f"minimum (3). Either provide a different checkpoint or "
            f"check that the parquet shards exist."
        )
    if probe_1_5_length < 3:
        raise ValueError(
            f"run_phase_12_smoke: Probe 1.5 checkpoint at "
            f"{probe_1_5_checkpoint.run_dir} has telemetry_length="
            f"{probe_1_5_length}, which is below the sham scheduler's "
            f"minimum (3)."
        )

    # The two rounds share an LLMConfig (and therefore a model_name,
    # max_output_tokens, max_retries). The api_key_env_var honors the
    # smoke harness argument so a non-default env var (e.g. a project-
    # specific key) lands in the config.
    llm_config = LLMConfig(api_key_env_var=llm_api_key_env_var)

    # 2. Build the two round configs.
    probe_1_config = _build_round_config(
        round_id=PHASE_12_PROBE_1_ROUND_ID,
        checkpoint=probe_1_checkpoint,
        seed=PHASE_12_PROBE_1_SEED,
        telemetry_length=probe_1_length,
        llm_config=llm_config,
        notes=(
            "Phase 12 smoke Round 1 — Probe 1 checkpoint; 5 passes; "
            "default StatisticConfig; default LLMConfig "
            "(gemini-2.5-pro); 2 real perturbations + 1 sham per pass; "
            "sham-schedule seed 42."
        ),
    )
    probe_1_5_config = _build_round_config(
        round_id=PHASE_12_PROBE_1_5_ROUND_ID,
        checkpoint=probe_1_5_checkpoint,
        seed=PHASE_12_PROBE_1_5_SEED,
        telemetry_length=probe_1_5_length,
        llm_config=llm_config,
        notes=(
            "Phase 12 smoke Round 2 — Probe 1.5 checkpoint; 5 passes; "
            "default StatisticConfig; default LLMConfig "
            "(gemini-2.5-pro); 2 real perturbations + 1 sham per pass; "
            "sham-schedule seed 43."
        ),
    )

    # 3. Run the two rounds sequentially.
    probe_1_round = run_round(
        probe_1_config, output_dir=output_dir, llm_client=llm_client
    )
    probe_1_5_round = run_round(
        probe_1_5_config, output_dir=output_dir, llm_client=llm_client
    )

    # 4. Cross-round diff — no rationales passed; if the configs diverge
    # on the diffed dimensions the call raises. Phase 12's contract is
    # that they don't.
    round_diff = compute_round_diff(
        probe_1_config,
        probe_1_5_config,
        rationales=None,
        notes=(
            "Phase 12 smoke: Round 1 vs Round 2 structural diff. The "
            "configs share StatisticConfig and LLMConfig by "
            "construction; the diff should have empty "
            "statistic_config_changes and llm_config_changes. "
            "checkpoint_ids and sham_schedule differ between rounds by "
            "design and are not in the diff scope."
        ),
    )

    # 5. Aggregate the cross-round audit.
    all_records = (
        probe_1_round.llm_call_records + probe_1_5_round.llm_call_records
    )
    audit = LLMCallAudit.from_records(all_records)

    t1 = int(time.time() * 1000)
    result = Phase12SmokeResult(
        probe_1_round=probe_1_round,
        probe_1_5_round=probe_1_5_round,
        round_diff=round_diff,
        llm_call_audit=audit,
        wallclock_ms=t1 - t0,
        notes=notes,
    )

    # 6. Write to disk.
    _write_smoke_result(result, output_dir / "mirror")
    return result
