"""Phase 8 gate test — the *semantic* read-only invariant.

Phase 6's :func:`tests.test_criterion_registry.test_registry_no_method_returns_writer`
and Phase 7's
:func:`tests.test_signal_mapping.test_signal_mapping_no_writer_shape`
were *structural* — they checked field/method names against a forbidden
list. Phase 8 adds the first *semantic* checks: an orchestrator method
named ``compute_reading`` could still write to the wrong path or
construct the wrong object; structural-name lists wouldn't catch it.

Three load-bearing checks live here:

1. ``test_orchestrator_writes_only_to_mirror_side`` — end-to-end pass
   under a tracked temp tree; every filesystem write is under
   ``runs/{run_id}/mirror/``.
2. ``test_orchestrator_does_not_construct_actor_or_world_model`` —
   monkey-patched constructors record their calls; assert zero
   constructions during a pass.
3. ``test_orchestrator_does_not_call_runner_step`` — monkey-patched
   :meth:`Runner.run` records its calls; assert zero invocations.

The Phase 7 newly-open question on whether the
structural-vs-semantic split should land a shared helper is resolved
here: **the structural and semantic shapes are different.** The
structural tests pin field/method names; the semantic tests pin
*behavior under exercise*. A future contributor reviewing the four
tests (Phase 6's name list, Phase 7's signal-mapping name list, plus
the three checks below) sees the lineage, and the structural-name
list is documented as "the recurring shape" rather than promoted to a
shared helper. The decision is journaled in Phase 8's entry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kind.mirror.criteria_v2 import V2_REGISTRY
from kind.mirror.llm_caller import (
    BatchPayload,
    LLMConfig,
    MockLLMClient,
    _PerCriterionReadingPayload,
)
from kind.mirror.orchestrator import (
    PassConfig,
    run_adversarial_pass,
)
from kind.mirror.registry import CriterionRegistry
from kind.mirror.structured import StructuredClaim


# ---------------------------------------------------------------------------
# Shared fixtures (parallel test_orchestrator.py for testbed homogeneity).
# ---------------------------------------------------------------------------


def _build_agent_step_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(120):
        rows.append(
            {
                "schema_version": "0.2.0",
                "run_id": "probe2-inv-test",
                "checkpoint_id": "ckpt-000001",
                "t": i,
                "episode_id": i // 60,
                "step_in_episode": i % 60,
                "wallclock_ms": i * 100,
                "h_t": [0.01 * i + 0.001 * j for j in range(4)],
                "z_t": [0.0] * 4,
                "encoder_embedding_t": [0.0] * 4,
                "policy_entropy_t": 1.0,
                "kl_aggregate_t": 0.5,
                "action_t": i % 5,
                "action_logprob_t": -1.0,
                "obs_hash_t": f"obs_{i % 7}",
                "q_params_t": ([0.0] * 4, [0.0] * 4),
                "p_params_t": ([0.0] * 4, [0.0] * 4),
                "kl_per_dim_t": [0.0] * 4,
                "recon_loss_t": 0.0,
                "intrinsic_signal_t": 0.0,
                "self_prediction_t": [0.0] * 4,
                "self_prediction_error_t": 0.0,
                "self_prediction_error_masked_t": False,
            }
        )
    return rows


def _write_agent_step_shards(
    telemetry_dir: Path, rows: list[dict[str, Any]]
) -> None:
    shard_dir = telemetry_dir / "agent_step"
    shard_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    shard_path = shard_dir / "shard-000000.parquet"
    with shard_path.open("wb") as fh:
        pq.write_table(table, fh)  # type: ignore[no-untyped-call]


def _write_world_event_log(
    telemetry_dir: Path, perturbations: list[dict[str, Any]]
) -> None:
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    path = telemetry_dir / "world_event.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for p in perturbations:
            fh.write(json.dumps(p) + "\n")


def _claim() -> StructuredClaim:
    return StructuredClaim(
        claim="x",
        cited_stream="agent_step",
        cited_run_id="probe2-inv-test",
        cited_episode_range=(0, 1),
        cited_step_range=(0, 50),
        cited_scalar_field="h_t",
        cited_value=0.0,
        falsifier="x",
        paraphrase_stability=None,
        reseed_stability=None,
        faithfulness_status="not_checked",
        judge_ruling="not_judged",
        reading_surface="head_internal",
        masked_steps_handling="n/a",
    )


def _per_criterion(*, criterion_id: str) -> _PerCriterionReadingPayload:
    return _PerCriterionReadingPayload(
        criterion_id=criterion_id,
        framework_anchor="buddhist_phenomenology",
        claims=[_claim()],
        free_text_notes="notes",
    )


def _build_mock_client() -> MockLLMClient:
    active_ids = ["reflexive_attention", "equanimity_perturbation_recovery"]
    held_out_ids = ["second_order_volition"]
    return MockLLMClient(
        [
            BatchPayload(per_criterion=[_per_criterion(criterion_id=cid) for cid in active_ids]),
            BatchPayload(per_criterion=[_per_criterion(criterion_id=cid) for cid in active_ids]),
            BatchPayload(per_criterion=[_per_criterion(criterion_id=cid) for cid in held_out_ids]),
            BatchPayload(per_criterion=[_per_criterion(criterion_id=cid) for cid in held_out_ids]),
        ]
    )


def _set_up_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "probe2-inv-test"
    telemetry_dir = run_dir / "telemetry"
    _write_agent_step_shards(telemetry_dir, _build_agent_step_rows())
    _write_world_event_log(
        telemetry_dir,
        [
            {
                "schema_version": "0.2.0",
                "run_id": "probe2-inv-test",
                "checkpoint_id": "ckpt-000001",
                "t_event": 50,
                "event_type": "builder_perturbation",
                "source": "builder",
                "payload": {"kind": "test"},
                "wallclock_ms": 5000,
            }
        ],
    )
    return run_dir


def _make_config(run_dir: Path) -> PassConfig:
    return PassConfig(
        run_id="probe2-inv-test",
        checkpoint_id="ckpt-000001",
        run_dir=run_dir,
        active_registry=CriterionRegistry(criteria=V2_REGISTRY.active()),
        held_out_registry=CriterionRegistry(criteria=V2_REGISTRY.held_out()),
        llm_config=LLMConfig(),
    )


# ---------------------------------------------------------------------------
# 1. writes_only_to_mirror_side — end-to-end path tracking.
# ---------------------------------------------------------------------------


def test_orchestrator_writes_only_to_mirror_side(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wrap ``pathlib.Path.open`` to record every open-for-write; assert
    every recorded path is under ``runs/{run_id}/mirror/``.

    The check covers both atomic JSON writes (via ``Path.open("w")``) and
    PreRegSink's appended JSONL (``Path.open("a")``). It does NOT cover
    pyarrow's parquet writes since the orchestrator does not write
    parquet shards — those are runner-side. The test still catches a
    future contributor who routes a parquet write through the
    orchestrator's path-construction code, because parquet's pre-flight
    open is also via ``Path.open``.

    The forbidden roots are listed explicitly so a future contributor
    sees what 'agent-readable location' means at the test level.
    """
    write_paths: list[Path] = []
    real_open = Path.open

    def tracking_open(
        self: Path, mode: str = "r", *args: Any, **kwargs: Any
    ) -> Any:
        if any(ch in mode for ch in ("w", "a", "x", "+")):
            write_paths.append(Path(str(self)))
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)

    run_dir = _set_up_run_dir(tmp_path)
    config = _make_config(run_dir)
    # Clear the writes from setup; we only care about what the
    # orchestrator's call produces.
    write_paths.clear()

    run_adversarial_pass(config, llm_client=_build_mock_client())

    assert write_paths, "expected at least one write during a pass"
    mirror_root = run_dir / "mirror"
    forbidden_roots = [
        run_dir / "telemetry",
        run_dir / "checkpoints",
    ]
    for p in write_paths:
        # Resolve relative to absolute to compare path prefixes.
        p_abs = p.resolve() if p.is_absolute() else p
        try:
            p_abs.relative_to(mirror_root.resolve())
        except ValueError:
            pytest.fail(
                f"orchestrator wrote to {p_abs}, which is not under "
                f"{mirror_root}. The one-way invariant requires every "
                f"orchestrator write under runs/{{run_id}}/mirror/."
            )
        for forbidden in forbidden_roots:
            try:
                p_abs.relative_to(forbidden.resolve())
                pytest.fail(
                    f"orchestrator wrote to {p_abs}, which is under the "
                    f"forbidden root {forbidden}. Agent-readable "
                    f"locations are off-limits."
                )
            except ValueError:
                pass  # Good — not under the forbidden root.


# ---------------------------------------------------------------------------
# 2. does_not_construct_actor_or_world_model.
# ---------------------------------------------------------------------------


def test_orchestrator_does_not_construct_actor_or_world_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Monkey-patch the constructors of :class:`kind.agents.actor.Actor`
    and :class:`kind.agents.world_model.WorldModel` to record calls. The
    orchestrator must construct neither during a pass.

    Why this matters: the spec's prose 'the orchestrator constructs the
    runner with pre_reg_dir set' would, if taken literally, build a
    Runner — which builds both an Actor and a WorldModel on device.
    The semantic check pins the structural invariant against the prose:
    the orchestrator honors Phase 5's *on-disk* pre-reg contract
    (a ``pre_reg.jsonl`` in a configured directory) via
    :class:`PreRegSink` directly, NOT by constructing a Runner.
    """
    from kind.agents.actor import Actor
    from kind.agents.world_model import WorldModel

    construction_log: list[str] = []
    real_actor_init = Actor.__init__
    real_world_model_init = WorldModel.__init__

    def tracking_actor_init(
        self: Actor, *args: Any, **kwargs: Any
    ) -> None:
        construction_log.append("Actor")
        real_actor_init(self, *args, **kwargs)

    def tracking_world_model_init(
        self: WorldModel, *args: Any, **kwargs: Any
    ) -> None:
        construction_log.append("WorldModel")
        real_world_model_init(self, *args, **kwargs)

    monkeypatch.setattr(Actor, "__init__", tracking_actor_init)
    monkeypatch.setattr(WorldModel, "__init__", tracking_world_model_init)

    run_dir = _set_up_run_dir(tmp_path)
    config = _make_config(run_dir)
    run_adversarial_pass(config, llm_client=_build_mock_client())

    assert construction_log == [], (
        f"orchestrator constructed {construction_log} during a pass; "
        f"the one-way invariant forbids Actor / WorldModel construction "
        f"on the mirror side"
    )


# ---------------------------------------------------------------------------
# 3. does_not_call_runner_step.
# ---------------------------------------------------------------------------


def test_orchestrator_does_not_call_runner_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Monkey-patch :meth:`kind.training.runner.Runner.run` to record
    calls. The orchestrator must not invoke the runner's training loop."""
    from kind.training.runner import Runner

    call_log: list[str] = []
    real_run = Runner.run

    def tracking_run(self: Runner, *args: Any, **kwargs: Any) -> Any:
        call_log.append("Runner.run")
        return real_run(self, *args, **kwargs)

    monkeypatch.setattr(Runner, "run", tracking_run)

    run_dir = _set_up_run_dir(tmp_path)
    config = _make_config(run_dir)
    run_adversarial_pass(config, llm_client=_build_mock_client())

    assert call_log == [], (
        f"orchestrator called {call_log} during a pass; the one-way "
        f"invariant forbids Runner.run() invocation on the mirror side"
    )
