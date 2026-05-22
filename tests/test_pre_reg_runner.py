"""Probe 2 Phase 5 gate tests — runner glue for the pre-registration sink.

Phase 5 (plan §2.5; synthesis §2.4 element 1; plan §4 row 5) lands the
runner glue around the pre-registration sink that Phase 0 built: the
``RunnerConfig.pre_reg_dir`` field, the ``Runner.emit_pre_registration``
method, and the ``runner.close()`` cascade that closes the
:class:`~kind.observer.pre_reg.PreRegSink` alongside the four telemetry
sinks. The pre-registration template at
``docs/workingjournal/probe2_templates/pre_registration.md`` is the
hand-fill prose form the builder reads later; the structured JSONL the
sink writes is the machine-readable form.

Phase 0's ``tests/test_pre_reg.py`` covers the sink and model in
isolation; this file covers the runner-level integration. Mirrors the
transport-pair fixture pattern from ``tests/test_lesion.py``.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from kind.agents.world_model import WorldModelConfig
from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import GridWorldConfig
from kind.env.transport import EnvTransportClient, EnvTransportServer
from kind.mirror.registry import ReadingSurface
from kind.observer.pre_reg import (
    PRE_REG_FILE,
    PRE_REG_SCHEMA_VERSION,
    PreRegistration,
    PreRegSinkClosedError,
)
from kind.training.runner import Runner, RunnerConfig

# ---- shared fixtures (parallel test_lesion.py) --------------------------


def _quiet_grid_world_config() -> GridWorldConfig:
    return GridWorldConfig(
        episode_length=50,
        initial_regrowth_p=0.0,
        drift_p_min=0.0,
        drift_p_max=0.0,
        drift_magnitude_per_step=0.0,
        n_initial_resources=2,
    )


def _tiny_world_model_config() -> WorldModelConfig:
    return WorldModelConfig(
        obs_channels=1,
        obs_size=32,
        h_dim=32,
        z_dim=4,
        embed_dim=32,
        num_actions=5,
        action_emb_dim=8,
        mlp_hidden=32,
        free_bits_per_dim=1.0,
        self_prediction_hidden=32,
    )


def _make_runner_config(
    *,
    tmp_path: Path,
    run_id: str = "probe2-phase5-test",
    pre_reg_dir: Path | None = None,
) -> RunnerConfig:
    return RunnerConfig(
        world_model_config=_tiny_world_model_config(),
        run_id=run_id,
        telemetry_dir=tmp_path / "telemetry",
        checkpoints_dir=tmp_path / "checkpoints",
        action_dim=5,
        ensemble_k=2,
        imagination_horizon=4,
        replay_capacity=200,
        replay_sequence_length=4,
        replay_batch_size=2,
        train_every_n_env_steps=1,
        warmup_env_steps=5,
        dream_cadence_env_steps=50,
        dream_horizon=4,
        checkpoint_every_n_env_steps=60,
        parquet_rows_per_shard=10,
        device="cpu",
        pre_reg_dir=pre_reg_dir,
    )


@contextmanager
def _transport_pair(
    *,
    seed: int = 42,
    run_id: str = "probe2-phase5-env",
) -> Iterator[
    tuple[EnvTransportClient, EnvTransportServer, EnvServer, threading.Thread]
]:
    config = EnvServerConfig(
        grid_world_config=_quiet_grid_world_config(),
        seed=seed,
        world_event_handler=lambda _r: None,
        run_id=run_id,
    )
    env_server = EnvServer(config)
    transport_server = EnvTransportServer(env_server, host="127.0.0.1", port=0)
    server_thread = threading.Thread(
        target=transport_server.serve_forever,
        name="PreRegRunnerTestEnvTransportServer",
        daemon=True,
    )
    server_thread.start()

    client = EnvTransportClient(
        host="127.0.0.1",
        port=transport_server.actual_port,
        world_event_handler=lambda _r: None,
    )
    try:
        yield client, transport_server, env_server, server_thread
    finally:
        try:
            client.close()
        finally:
            transport_server.shutdown()
            server_thread.join(timeout=5.0)


def _record(**overrides: Any) -> PreRegistration:
    """Build a complete v2 PreRegistration with all required fields
    populated. Mirrors ``tests/test_pre_reg.py::_full_record`` but kept
    local so the two test files stay independent."""
    base: dict[str, Any] = {
        "run_id": "probe2-phase5-test",
        "timestamp_ms": 1_730_000_000_000,
        "criteria_active": [
            "equanimity_perturbation_recovery",
            "head_internal_sp_err_distribution",
        ],
        "criteria_held_out": ["behavior_side_scalar_conditioning"],
        "signal_mappings": {
            "equanimity_perturbation_recovery": [
                "kl_aggregate_t",
                "ensemble_disagreement",
            ],
            "head_internal_sp_err_distribution": [
                "self_prediction_error_t",
            ],
        },
        "falsifiers": {
            "equanimity_perturbation_recovery": (
                "if recovery shape is oscillatory or absent within 20 steps"
            ),
            "head_internal_sp_err_distribution": (
                "if sp_err distribution KS-D < 0.10 vs frozen-target run"
            ),
        },
        "scalar_checks": {
            "equanimity_perturbation_recovery": ["kl_aggregate_t"],
            "head_internal_sp_err_distribution": ["sp_err KS-D"],
        },
        "reading_surfaces_per_criterion": {
            "equanimity_perturbation_recovery": [
                ReadingSurface.SUBSTRATE_SIDE,
                ReadingSurface.HEAD_INTERNAL,
            ],
            "head_internal_sp_err_distribution": [ReadingSurface.HEAD_INTERNAL],
        },
        "asymmetry_of_access": (
            "Io reads scalar self_prediction_error_t on PolicyView; the "
            "mirror reads the full self_prediction_t vector and longitudinal "
            "cross-run analysis."
        ),
        "builder_mode": "skeptic",
        "expected_outcome": (
            "first round expects equanimity at substrate-side and "
            "head-internal to admit at moderate strength."
        ),
        "expected_outcome_per_surface": {
            ReadingSurface.SUBSTRATE_SIDE: "weak admit",
            ReadingSurface.HEAD_INTERNAL: "moderate admit",
            ReadingSurface.BEHAVIOR_SIDE: "no admit (held out)",
        },
        "substrate_decisions_off_table": [
            "RSSM lineage choice",
            "K=5 ensemble cardinality",
        ],
        "column_init": "small_gaussian",
        "new_actor_readable_interfaces_added": [],
    }
    base.update(overrides)
    return PreRegistration(**base)


# ---- (1) sink construction is gated on pre_reg_dir ----------------------


def test_runner_without_pre_reg_dir_has_no_sink(tmp_path: Path) -> None:
    """Probe 1 / Probe 1.5 default: no ``pre_reg_dir``, no sink, no
    artifact created. The runner is non-breaking for callers that don't
    opt in."""
    config = _make_runner_config(tmp_path=tmp_path, pre_reg_dir=None)
    with _transport_pair() as (client, _ts, env_server, _thread):
        runner = Runner(config, client, env_server=env_server)
        try:
            assert runner._pre_reg_sink is None
            # No pre_reg directory should have been created on the
            # filesystem — the runner's construction is purely additive.
            assert not (tmp_path / "pre_reg").exists()
        finally:
            runner.close()


def test_runner_with_pre_reg_dir_constructs_sink(tmp_path: Path) -> None:
    """Probe 2 caller opts in by setting ``pre_reg_dir``. The sink is
    constructed at __init__ time; the directory is created."""
    pre_reg_dir = tmp_path / "pre_reg"
    config = _make_runner_config(tmp_path=tmp_path, pre_reg_dir=pre_reg_dir)
    with _transport_pair() as (client, _ts, env_server, _thread):
        runner = Runner(config, client, env_server=env_server)
        try:
            assert runner._pre_reg_sink is not None
            assert pre_reg_dir.exists()
            assert pre_reg_dir.is_dir()
        finally:
            runner.close()


# ---- (2) emit_pre_registration writes a valid record --------------------


def test_emit_pre_registration_writes_to_pre_reg_jsonl(tmp_path: Path) -> None:
    """A single ``emit_pre_registration`` call writes exactly one
    JSONL line at ``<pre_reg_dir>/pre_reg.jsonl`` carrying a valid
    PreRegistration."""
    pre_reg_dir = tmp_path / "pre_reg"
    config = _make_runner_config(tmp_path=tmp_path, pre_reg_dir=pre_reg_dir)
    record = _record()

    with _transport_pair() as (client, _ts, env_server, _thread):
        runner = Runner(config, client, env_server=env_server)
        try:
            runner.emit_pre_registration(record)
        finally:
            runner.close()

    path = pre_reg_dir / PRE_REG_FILE
    assert path.exists(), f"expected {path} to exist after emit_pre_registration"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = PreRegistration.model_validate_json(lines[0])
    assert parsed == record
    assert parsed.schema_version == PRE_REG_SCHEMA_VERSION


# ---- (3) multiple emits append to the same JSONL ------------------------


def test_emit_pre_registration_appends_multiple_records(tmp_path: Path) -> None:
    """Multiple ``emit_pre_registration`` calls within a single Runner
    instance produce a JSONL (newline-delimited) file with one record
    per line — no overwrite, no merge."""
    pre_reg_dir = tmp_path / "pre_reg"
    config = _make_runner_config(tmp_path=tmp_path, pre_reg_dir=pre_reg_dir)
    r1 = _record(timestamp_ms=1_730_000_000_001)
    r2 = _record(
        timestamp_ms=1_730_000_000_002,
        criteria_active=["self_prediction_quadruplet"],
        criteria_held_out=[],
        signal_mappings={"self_prediction_quadruplet": ["sp_err"]},
        falsifiers={"self_prediction_quadruplet": "if no spike-recovery shape"},
        scalar_checks={"self_prediction_quadruplet": ["sp_err"]},
        reading_surfaces_per_criterion={
            "self_prediction_quadruplet": [
                ReadingSurface.SUBSTRATE_SIDE,
                ReadingSurface.HEAD_INTERNAL,
            ]
        },
    )
    r3 = _record(timestamp_ms=1_730_000_000_003, builder_mode="proponent")

    with _transport_pair() as (client, _ts, env_server, _thread):
        runner = Runner(config, client, env_server=env_server)
        try:
            runner.emit_pre_registration(r1)
            runner.emit_pre_registration(r2)
            runner.emit_pre_registration(r3)
        finally:
            runner.close()

    path = pre_reg_dir / PRE_REG_FILE
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    parsed = [PreRegistration.model_validate_json(line) for line in lines]
    assert parsed[0] == r1
    assert parsed[1] == r2
    assert parsed[2] == r3


# ---- (4) emit_pre_registration raises when no pre_reg_dir ---------------


def test_emit_pre_registration_raises_without_pre_reg_dir(tmp_path: Path) -> None:
    """A runner constructed without ``pre_reg_dir`` raises
    :class:`RuntimeError` on ``emit_pre_registration``. The opt-in
    semantic is enforced at the public API."""
    config = _make_runner_config(tmp_path=tmp_path, pre_reg_dir=None)

    with _transport_pair() as (client, _ts, env_server, _thread):
        runner = Runner(config, client, env_server=env_server)
        try:
            with pytest.raises(RuntimeError, match="pre_reg_dir"):
                runner.emit_pre_registration(_record())
        finally:
            runner.close()


# ---- (5) emit_pre_registration raises after close ------------------------


def test_emit_pre_registration_raises_after_close(tmp_path: Path) -> None:
    """``Runner.close()`` flips ``_closed`` to True; subsequent
    ``emit_pre_registration`` calls raise. Mirrors the ``run()``
    closed-state check at the public API."""
    pre_reg_dir = tmp_path / "pre_reg"
    config = _make_runner_config(tmp_path=tmp_path, pre_reg_dir=pre_reg_dir)

    with _transport_pair() as (client, _ts, env_server, _thread):
        runner = Runner(config, client, env_server=env_server)
        runner.close()
        with pytest.raises(RuntimeError, match="closed"):
            runner.emit_pre_registration(_record())


# ---- (6) close cascade closes the pre_reg sink --------------------------


def test_runner_close_closes_pre_reg_sink(tmp_path: Path) -> None:
    """``runner.close()`` closes the underlying :class:`PreRegSink`
    alongside the four telemetry sinks. Verified by attempting to write
    through the sink directly after close — the sink raises
    :class:`PreRegSinkClosedError`. The sink's ``fsync`` runs as part of
    its ``close()``, so the data already written lands on disk."""
    pre_reg_dir = tmp_path / "pre_reg"
    config = _make_runner_config(tmp_path=tmp_path, pre_reg_dir=pre_reg_dir)
    record = _record()

    with _transport_pair() as (client, _ts, env_server, _thread):
        runner = Runner(config, client, env_server=env_server)
        runner.emit_pre_registration(record)
        sink = runner._pre_reg_sink
        assert sink is not None
        runner.close()
        # Sink is closed: writing directly through the sink raises.
        with pytest.raises(PreRegSinkClosedError):
            sink.write(record)

    # The pre-emitted record landed on disk — fsync ran during close().
    path = pre_reg_dir / PRE_REG_FILE
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert PreRegistration.model_validate_json(lines[0]) == record


def test_runner_close_is_idempotent_with_pre_reg_sink(tmp_path: Path) -> None:
    """Calling ``runner.close()`` twice is a no-op on the second call,
    even with a pre_reg sink attached. Matches the existing
    close-idempotence pattern."""
    pre_reg_dir = tmp_path / "pre_reg"
    config = _make_runner_config(tmp_path=tmp_path, pre_reg_dir=pre_reg_dir)

    with _transport_pair() as (client, _ts, env_server, _thread):
        runner = Runner(config, client, env_server=env_server)
        runner.close()
        runner.close()  # second close is a no-op, not a raise


# ---- (7) pre-registration template file ---------------------------------


_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "workingjournal"
    / "probe2_templates"
    / "pre_registration.md"
)


def test_pre_registration_template_exists() -> None:
    """The hand-fill template lives at the documented path and is
    non-empty. The structured JSONL the sink writes is the
    machine-readable form; this template is the prose form the builder
    fills in once per round."""
    assert _TEMPLATE_PATH.exists(), (
        f"expected pre-registration template at {_TEMPLATE_PATH}"
    )
    text = _TEMPLATE_PATH.read_text(encoding="utf-8")
    assert text.strip(), "pre-registration template is empty"


def test_pre_registration_template_covers_all_v2_model_fields() -> None:
    """The template is a hand-fill scaffold that maps to the v2
    :class:`PreRegistration` model. Every required field on the model
    must appear in the template prose so the builder fills in a
    complete record. Schema-version / timestamp_ms style fields are
    journaled as identification slots; ``schema_version`` defaults to
    ``PRE_REG_SCHEMA_VERSION`` and is not user-edited."""
    text = _TEMPLATE_PATH.read_text(encoding="utf-8")
    required_field_names = [
        "run_id",
        "timestamp_ms",
        "criteria_active",
        "criteria_held_out",
        "signal_mappings",
        "falsifiers",
        "scalar_checks",
        "reading_surfaces_per_criterion",
        "asymmetry_of_access",
        "builder_mode",
        "expected_outcome",
        "expected_outcome_per_surface",
        "substrate_decisions_off_table",
        "column_init",
        "new_actor_readable_interfaces_added",
    ]
    missing = [name for name in required_field_names if name not in text]
    assert not missing, (
        f"pre-registration template is missing prose coverage for "
        f"required model fields: {missing}"
    )
