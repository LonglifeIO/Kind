"""Phase 5 gate tests — the mirror reading dream-state telemetry.

Per plan §4 Phase 5 row plus the structural guards the build prompt names.
The load-bearing test is the view-isolation / import-discipline test: it
asserts ``kind.mirror.dream_reading`` does not import the training-side
modules that would let mirror output flow back to Io, and it sanity-checks
that the lint genuinely trips when such an import is present.

LLM calls are mocked (:class:`MockLLMClient`); the gate is the structural
enforcement and prompt-construction correctness, not Gemini's specific
output.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from kind.mirror.criteria_v2 import V2_REGISTRY
from kind.mirror.dream_reading import (
    DISABLED_DREAM_READING_SURFACES,
    DREAM_MIRROR_MARKERS_FILE,
    DREAM_READINGS_FILE,
    DreamReader,
    DreamReading,
    DreamSessionTelemetry,
    load_dream_session_telemetry,
)
from kind.mirror.llm_caller import BatchPayload, MockLLMClient, _PerCriterionReadingPayload
from kind.mirror.registry import ReadingSurface
from kind.mirror.structured import StructuredClaim
from kind.observer.dream_session import DreamSessionMeta, DreamSessionSink
from kind.observer.schemas import (
    PROBE_3_TELEMETRY_SCHEMA_VERSION,
    DreamRollout,
    WorldEvent,
)
from kind.observer.sinks import ParquetSink

REPO_ROOT = Path(__file__).resolve().parent.parent
DREAM_READING_PATH = REPO_ROOT / "kind" / "mirror" / "dream_reading.py"

H_DIM = 4
Z_DIM = 2
HORIZON = 6
DISAGREEMENT = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
RUN_ID = "dream-run-0001"
SESSION_ID = "dream-session-xyz"


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _dream_rollout(seed_step: int) -> DreamRollout:
    return DreamRollout(
        schema_version=PROBE_3_TELEMETRY_SCHEMA_VERSION,
        run_id=RUN_ID,
        checkpoint_id="ckpt-000001",
        seed_step=seed_step,
        seed_h0=[float(i) * 0.01 for i in range(H_DIM)],
        seed_z0=[float(i) * 0.01 for i in range(Z_DIM)],
        sequence_h=[[float(j + i) * 0.01 for i in range(H_DIM)] for j in range(HORIZON)],
        sequence_z_prior=[
            [float(j + i) * 0.01 for i in range(Z_DIM)] for j in range(HORIZON)
        ],
        sequence_action=[i % 5 for i in range(HORIZON)],
        sequence_action_logprob=[-float(i) * 0.1 for i in range(HORIZON)],
        sequence_prior_entropy=[float(i) * 0.05 for i in range(HORIZON)],
        sequence_decoded_obs=None,
        cumulative_prior_entropy=0.5,
        mean_step_kl_successive_priors=0.1,
        max_step_latent_norm_change=0.9,
        sequence_self_prediction=None,  # the head does not run during a dream
        dream_session_id=SESSION_ID,
        seed_kind="replay",
        seed_replay_segment_id=42,
        seed_replay_step_offset=0,
        temperature_schedule=[1.5, 1.5, 1.5, 1.8, 2.1, 2.5],
        sampling_parameters={"replay_warmup_length": 8, "re_seed_every_n_steps": 10.0},
        gradient_policy="none",
        rng_seed=1729,
        termination_reason="horizon_complete",
        re_seed_step_indices=[3],
        sequence_ensemble_disagreement_variance=list(DISAGREEMENT),
        checkpoint_hash="a" * 64,
    )


def _session_meta(*, end: bool) -> DreamSessionMeta:
    return DreamSessionMeta(
        schema_version="0.1.0",
        run_id=RUN_ID,
        checkpoint_id="ckpt-000001",
        dream_session_id=SESSION_ID,
        started_at_env_step=1000,
        started_at_wallclock_ms=10_000,
        ended_at_env_step=1500 if end else None,
        ended_at_wallclock_ms=20_000 if end else None,
        end_trigger="desktop_on" if end else None,
        rollout_count=2 if end else 0,
        envelope_config_snapshot={"hard_cap_rollout_count": 50},
        seed_selection_config_snapshot={"mode": "replay"},
    )


def _telemetry() -> DreamSessionTelemetry:
    return DreamSessionTelemetry(
        dream_session_id=SESSION_ID,
        session_start=_session_meta(end=False),
        session_end=_session_meta(end=True),
        rollouts=(_dream_rollout(1000), _dream_rollout(1200)),
        state_transitions=(),
    )


def _claim(
    *,
    surface: str,
    field: str = "sequence_ensemble_disagreement_variance",
    value: float = 0.02,
    step_range: tuple[int, int] | None = (0, 11),
) -> StructuredClaim:
    return StructuredClaim(
        claim=f"claim at {surface} citing {field}",
        cited_stream="dream_rollout",
        cited_run_id=RUN_ID,
        cited_episode_range=None,
        cited_step_range=step_range,
        cited_scalar_field=field,
        cited_value=value,
        falsifier="f",
        paraphrase_stability=None,
        reseed_stability=None,
        faithfulness_status="not_checked",
        judge_ruling="not_judged",
        reading_surface=surface,  # type: ignore[arg-type]
        masked_steps_handling="n/a",
    )


def _primary_payload() -> BatchPayload:
    """reflexive_attention returns a (forbidden) head-internal claim — the
    guard must drop it and mark the criterion not-assessable. equanimity
    returns a legitimate substrate-side claim citing the latent-keyed
    ensemble-disagreement signal."""
    return BatchPayload(
        per_criterion=[
            _PerCriterionReadingPayload(
                criterion_id="reflexive_attention",
                framework_anchor="buddhist_phenomenology",
                claims=[_claim(surface="head_internal", field="dream_self_reference_t")],
                free_text_notes="head-internal attempt",
            ),
            _PerCriterionReadingPayload(
                criterion_id="equanimity_perturbation_recovery",
                framework_anchor="buddhist_phenomenology",
                claims=[_claim(surface="substrate_side")],
                free_text_notes="substrate-side latent reading",
            ),
        ]
    )


def _adversarial_payload() -> BatchPayload:
    return BatchPayload(
        per_criterion=[
            _PerCriterionReadingPayload(
                criterion_id="reflexive_attention",
                framework_anchor="null_statistics",
                claims=[_claim(surface="head_internal")],
                free_text_notes="null",
            ),
            _PerCriterionReadingPayload(
                criterion_id="equanimity_perturbation_recovery",
                framework_anchor="null_statistics",
                claims=[_claim(surface="substrate_side")],
                free_text_notes="null",
            ),
            _PerCriterionReadingPayload(
                criterion_id="second_order_volition",
                framework_anchor="null_statistics",
                claims=[_claim(surface="substrate_side")],
                free_text_notes="null",
            ),
        ]
    )


def _reader(responses: list[Any] | None = None) -> tuple[DreamReader, MockLLMClient]:
    if responses is None:
        responses = [_primary_payload(), _adversarial_payload()]
    client = MockLLMClient(responses)
    return DreamReader(client=client), client


# ---------------------------------------------------------------------------
# Import-discipline / view-isolation (LOAD-BEARING).
# ---------------------------------------------------------------------------


def _imported_modules_from_source(source: str) -> list[str]:
    """Collect every imported module name from a source string (AST-based)."""
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
    return names


_FORBIDDEN_PREFIXES = ("kind.training",)


def test_dream_reading_does_not_import_training_modules() -> None:
    """LOAD-BEARING. ``kind.mirror.dream_reading`` may import read-only
    observer modules and mirror-side modules; it must NOT import
    ``kind.training.state_machine``, ``kind.training.dream``, or any other
    ``kind.training`` module. The dependency edge that would let mirror
    output flow back to Io fails to exist — the one-way constraint made
    structural (the analog of Phase 4's HostSignals type-level guard)."""
    assert DREAM_READING_PATH.exists()
    imported = _imported_modules_from_source(DREAM_READING_PATH.read_text())

    offending = [
        m
        for m in imported
        if any(m == p or m.startswith(p + ".") for p in _FORBIDDEN_PREFIXES)
    ]
    assert offending == [], (
        f"kind.mirror.dream_reading imports forbidden training module(s) "
        f"{offending}; the one-way constraint requires it import only "
        f"read-only observer modules and mirror-side modules. Found "
        f"imports: {imported}"
    )
    # Specifically the two §2.6 names.
    assert "kind.training.state_machine" not in imported
    assert "kind.training.dream" not in imported


def test_import_discipline_lint_trips_on_forbidden_import() -> None:
    """Sanity-check the lint genuinely fails if dream_reading imported a
    training module — the test is only meaningful if it can fail."""
    bad_source = (
        "from __future__ import annotations\n"
        "from kind.observer.schemas import DreamRollout\n"
        "from kind.training.state_machine import StateController\n"
    )
    imported = _imported_modules_from_source(bad_source)
    offending = [
        m
        for m in imported
        if any(m == p or m.startswith(p + ".") for p in _FORBIDDEN_PREFIXES)
    ]
    assert offending == ["kind.training.state_machine"], (
        "the import-discipline lint failed to detect a forbidden training "
        "import in the positive-control source — the load-bearing test would "
        "not catch a real regression."
    )


# ---------------------------------------------------------------------------
# Surface-availability declaration + confabulation guard.
# ---------------------------------------------------------------------------


def test_dream_prompt_declares_disabled_surfaces() -> None:
    """The dream-reading prompt declares head-internal and behavior-side
    unavailable; the reading pipeline does not surface a frozen-criterion
    finding sourced from a disabled surface."""
    reader, client = _reader()
    reading = reader.read_session(_telemetry(), run_id=RUN_ID)

    # The prompt the LLM was shown declares the disabled surfaces.
    primary_user_prompt = client.calls[0][1]
    assert "head_internal: DISABLED" in primary_user_prompt
    assert "behavior_side: DISABLED" in primary_user_prompt
    assert "substrate_side: ENABLED" in primary_user_prompt
    assert "not assessable from the substrate alone" in primary_user_prompt

    # reflexive_attention (head-internal-only) yields a non-finding, not a
    # fabricated positive — even though the mock returned a head-internal
    # claim for it.
    assert "reflexive_attention" in reading.non_assessable_criteria
    assert "not assessable" in reading.non_assessable_criteria["reflexive_attention"]
    # No surviving claim sits on a disabled surface.
    for claim in reading.state_typed_claims:
        assert ReadingSurface(claim.reading_surface) not in DISABLED_DREAM_READING_SURFACES
    # The fabricated head-internal claim did not survive.
    assert all(
        c.reading_surface == "substrate_side" for c in reading.state_typed_claims
    )


def test_head_dependent_criterion_not_assessable_with_self_prediction_none() -> None:
    """A fixture dream with sequence_self_prediction=None yields 'reflexive
    attention not assessable', not a fabricated finding. (The rollouts in
    the fixture carry sequence_self_prediction=None by construction — the
    head did not run.)"""
    telemetry = _telemetry()
    assert all(r.sequence_self_prediction is None for r in telemetry.rollouts)
    reader, _ = _reader()
    reading = reader.read_session(telemetry, run_id=RUN_ID)
    assert "reflexive_attention" in reading.non_assessable_criteria
    # No reflexive_attention positive made it into the surviving claims.
    assert "reflexive_attention" not in {
        c.claim.split()[0] for c in reading.state_typed_claims
    }


# ---------------------------------------------------------------------------
# Latent-keying.
# ---------------------------------------------------------------------------


def test_reading_is_latent_keyed_not_decoded_obs() -> None:
    """The reading is built around the latent-dynamics fields; decoded
    observations are treated as the attenuated secondary channel (not
    surfaced as a primary signal)."""
    reader, client = _reader()
    reader.read_session(_telemetry(), run_id=RUN_ID)
    prompt = client.calls[0][1]

    for latent_field in (
        "mean_step_kl_successive_priors",
        "cumulative_prior_entropy",
        "max_step_latent_norm_change",
        "sequence_prior_entropy",
        "sequence_ensemble_disagreement_variance",
    ):
        assert latent_field in prompt, f"latent field {latent_field} missing from prompt"

    assert "latent-dominated" in prompt or "LATENT-KEYING" in prompt
    # Decoded observations are not promoted to a primary computed signal.
    assert "signal: sequence_decoded_obs" not in prompt


# ---------------------------------------------------------------------------
# Frozen criteria unchanged.
# ---------------------------------------------------------------------------


def test_frozen_criteria_unchanged_no_dream_tuned_criterion() -> None:
    """The criteria set is identical to Probe 2's frozen set — no
    dream-specific criterion was added by Phase 5."""
    assert len(V2_REGISTRY.criteria) == 3
    assert tuple(c.id for c in V2_REGISTRY.criteria) == (
        "reflexive_attention",
        "equanimity_perturbation_recovery",
        "second_order_volition",
    )
    expected = {
        "reflexive_attention": (
            False,
            frozenset({ReadingSurface.HEAD_INTERNAL}),
            "reflexive_attention_v1",
        ),
        "equanimity_perturbation_recovery": (
            False,
            frozenset({ReadingSurface.SUBSTRATE_SIDE, ReadingSurface.BEHAVIOR_SIDE}),
            "equanimity_perturbation_recovery_v1",
        ),
        "second_order_volition": (
            True,
            frozenset({ReadingSurface.SUBSTRATE_SIDE, ReadingSurface.BEHAVIOR_SIDE}),
            "second_order_volition_v1",
        ),
    }
    for c in V2_REGISTRY.criteria:
        held_out, surfaces, falsifier_id = expected[c.id]
        assert c.held_out is held_out
        assert c.reading_surfaces == surfaces
        assert c.falsifier_id == falsifier_id


# ---------------------------------------------------------------------------
# One-way / no data-plane write.
# ---------------------------------------------------------------------------


def test_one_way_writes_only_mirror_side(tmp_path: Path) -> None:
    """The mirror writes only the DreamReading JSONL (and optionally a
    builder-facing mirror_marker) under the mirror-side directory; it writes
    nothing into any telemetry/checkpoint path; the default runtime directive
    is continue_and_log_uncertainty with no termination authority."""
    run_dir = tmp_path / "runs" / RUN_ID
    telemetry_dir = run_dir / "telemetry"
    checkpoints_dir = run_dir / "checkpoints"
    telemetry_dir.mkdir(parents=True)
    checkpoints_dir.mkdir(parents=True)
    mirror_dir = run_dir / "mirror"

    reader, _ = _reader()
    reading = reader.read_session(
        _telemetry(),
        run_id=RUN_ID,
        mirror_out_dir=mirror_dir,
        emit_mirror_marker=True,
    )

    # Mirror-side directory carries exactly the reading JSONL and the marker.
    written = {p.name for p in mirror_dir.iterdir()}
    assert written == {DREAM_READINGS_FILE, DREAM_MIRROR_MARKERS_FILE}

    # Nothing written under telemetry/ or checkpoints/.
    assert list(telemetry_dir.iterdir()) == []
    assert list(checkpoints_dir.iterdir()) == []

    # Logs-only: the only runtime directive the mirror can carry.
    assert reading.runtime_directive == "continue_and_log_uncertainty"
    # No termination-authority field exists on the record.
    assert "terminate" not in DreamReading.model_fields
    assert "stop" not in DreamReading.model_fields

    # The mirror_marker is a builder-facing system event with no authority.
    marker_lines = (mirror_dir / DREAM_MIRROR_MARKERS_FILE).read_text().splitlines()
    assert len(marker_lines) == 1
    marker = json.loads(marker_lines[0])
    assert marker["event_type"] == "mirror_marker"
    assert marker["source"] == "system"
    assert marker["payload"]["runtime_directive"] == "continue_and_log_uncertainty"
    # The mirror never emits a state_transition.
    assert marker["event_type"] != "state_transition"


# ---------------------------------------------------------------------------
# DreamReading output round-trips and is session-bounded.
# ---------------------------------------------------------------------------


def test_dream_reading_round_trips_session_bounded() -> None:
    reader, _ = _reader()
    reading = reader.read_session(_telemetry(), run_id=RUN_ID)

    redumped = DreamReading.model_validate_json(reading.model_dump_json())
    assert redumped == reading

    # Keyed to the DreamSessionMeta session.
    assert reading.dream_session_id == SESSION_ID
    assert reading.digest_session_range == (1000, 1500)
    assert reading.reader_role == "dream_observer"


# ---------------------------------------------------------------------------
# Faithfulness resolves a dream-state claim (scoped).
# ---------------------------------------------------------------------------


def test_faithfulness_resolves_latent_keyed_claim() -> None:
    """A claim citing sequence_ensemble_disagreement_variance on a known
    session resolves against the persisted DreamRollout record."""
    reader, _ = _reader()
    reading = reader.read_session(_telemetry(), run_id=RUN_ID)
    # The surviving substrate-side claim cites the latent-keyed signal with
    # a value present in the concatenated trajectory.
    assert reading.state_typed_claims
    assert reading.faithfulness_rate == 1.0
    assert reading.faithfulness_admissible is True
    assert "buffer persistence" in reading.faithfulness_scope_note


def test_faithfulness_scope_flags_obs_window_deferral() -> None:
    reader, _ = _reader()
    reading = reader.read_session(_telemetry(), run_id=RUN_ID)
    note = reading.faithfulness_scope_note
    assert "DreamRollout" in note
    assert "seeding obs window" in note
    assert "deferred" in note or "ephemeral" in note


# ---------------------------------------------------------------------------
# End-to-end: the mirror reads dream telemetry from a written run dir.
# ---------------------------------------------------------------------------


def _write_run_telemetry(telemetry_dir: Path) -> None:
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    # DreamRollout parquet shard.
    with ParquetSink(telemetry_dir / "dream_rollout", DreamRollout) as sink:
        sink.write(_dream_rollout(1000))
        sink.write(_dream_rollout(1200))
    # DreamSessionMeta double-write.
    with DreamSessionSink(telemetry_dir) as sink:
        sink.write(_session_meta(end=False))
        sink.write(_session_meta(end=True))
    # A state_transition world event framing the session.
    transition = WorldEvent(
        schema_version=PROBE_3_TELEMETRY_SCHEMA_VERSION,
        run_id=RUN_ID,
        checkpoint_id="ckpt-000001",
        t_event=1000,
        event_type="state_transition",
        source="environment",
        payload={
            "from_state": "waking",
            "to_state": "dreaming",
            "dream_session_id": SESSION_ID,
            "trigger": "desktop_off",
            "wallclock_ms_in_prev_state": 5000,
            "env_step_at_transition": 1000,
        },
        wallclock_ms=10_000,
    )
    (telemetry_dir / "world_event.jsonl").write_text(transition.model_dump_json() + "\n")


def test_mirror_reads_dream_telemetry_end_to_end(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / RUN_ID
    telemetry_dir = run_dir / "telemetry"
    _write_run_telemetry(telemetry_dir)

    loaded = load_dream_session_telemetry(telemetry_dir, SESSION_ID)
    assert len(loaded.rollouts) == 2
    assert loaded.session_start.ended_at_env_step is None
    assert loaded.session_end is not None
    assert len(loaded.state_transitions) == 1

    reader, _ = _reader()
    reading = reader.read_session(
        loaded, run_id=RUN_ID, mirror_out_dir=run_dir / "mirror"
    )
    assert reading.state_typed_claims  # substrate-side claim survived
    assert "reflexive_attention" in reading.non_assessable_criteria
    assert reading.digest_session_range == (1000, 1500)
    # The DreamReading landed on the mirror-side disk path.
    readings_file = run_dir / "mirror" / DREAM_READINGS_FILE
    assert readings_file.exists()
    assert DreamReading.model_validate_json(readings_file.read_text().strip()) == reading
