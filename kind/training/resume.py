"""Biography continuation — the resume entry point (continuation plan C1).

One thin, tested seam between "a paused biography on disk" and "the same
mind continuing": find the newest checkpoint, restore it into a freshly
constructed :class:`~kind.training.runner.Runner`, and stamp the
``world_event`` stream with a **resume marker** so the boundary is a
first-class, analyzable event in the record.

The marker rides the existing sham-perturbation surface
(:meth:`EnvServer.fire_sham_perturbation` — flag-only, no grid mutation,
no RNG advance, observation byte-identical), so no new event type or
schema version is needed; the Phase-3 anchor extraction already excludes
``is_sham`` records by test.

What resume does and does not preserve (plan grounding): the checkpoint
restores the **mind** — weights, optimizers, RNG, runtime state
(``h/z/a_prev``), offline metabolic/controller state. The **world** is a
fresh :class:`GridWorld` process: board and drift-p re-roll, equivalent
to an episode boundary plus a drift-p reset to the config default. The
marker payload records both sides so the analysis can treat the boundary
honestly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq

if TYPE_CHECKING:
    from kind.env.env_server import EnvServer
    from kind.env.transport import EnvTransportClient
    from kind.training.runner import Runner

__all__ = ["continuation_counters", "resume_from_latest_checkpoint"]


def continuation_counters(telemetry_dir: Path) -> tuple[int, int]:
    """The fresh world's counter seeds for a resumed session:
    ``(initial_env_step, initial_episode_id)`` for
    :class:`GridWorldConfig`, read from the run's own ``agent_step``
    record so ``t`` stays monotonic and ``episode_id`` unique across
    the biography.

    The first resumed row stamps its indices from the **checkpointed
    pending state** (measured 2026-07-08: it is the step taken *from*
    the state the mind paused in — ``t = last_recorded + 1``, still in
    the paused episode); env-derived rows follow from the seeds. So
    ``initial_env_step = last_t + 1`` makes the biography's ``t``
    strictly monotonic across the boundary, and
    ``initial_episode_id = last_episode + 1`` keeps episode ids unique
    (the re-rolled world is a new episode). ``(0, 0)`` when no
    telemetry exists.

    Caveat, documented not solved: after a *crash* (post-checkpoint
    rows orphaned in telemetry, then state replayed from the earlier
    checkpoint), the replayed steps' stamps can overlap the orphans —
    the known bounded-loss window. Clean session boundaries (the
    biography's normal case: run-to-target, final checkpoint at the
    target) have no orphans.
    """
    last_t = -1
    last_episode = -1
    for shard in sorted((telemetry_dir / "agent_step").glob("*.parquet")):
        data = pq.read_table(  # type: ignore[no-untyped-call]
            shard, columns=["t", "episode_id"]
        ).to_pydict()
        if data["t"]:
            last_t = max(last_t, max(int(t) for t in data["t"]))
            last_episode = max(
                last_episode, max(int(e) for e in data["episode_id"])
            )
    if last_t < 0:
        return (0, 0)
    return (last_t + 1, last_episode + 1)


def resume_from_latest_checkpoint(
    runner: "Runner",
    client: "EnvTransportClient",
    env_server: "EnvServer",
    *,
    marker_extra: dict[str, Any] | None = None,
) -> str:
    """Load the newest checkpoint into ``runner``, start the fresh
    world, and emit the resume marker. Returns the checkpoint id.
    Raises :class:`FileNotFoundError` when the run has no checkpoint to
    resume from (a fresh run should not call this).

    Ordering is the seam's whole job: ``Runner.run`` deliberately skips
    ``connect()`` when a checkpoint was loaded (it consumes the loaded
    runtime state instead), but ``connect()`` is also what *starts* the
    env server — so on resume the connection is made here, explicitly,
    and the fresh world's initial observation is discarded (the mind
    continues from its checkpointed state; the re-rolled world enters
    through the next step's observation, one more world change of the
    kind episode boundaries already produce).

    Must be called after construction and before :meth:`Runner.run`.
    """
    latest = runner.checkpoint_manager.latest()
    if latest is None:
        raise FileNotFoundError("no checkpoint to resume from for this run")
    runner.load_checkpoint(latest)
    client.connect()  # starts the fresh world; initial EnvStep discarded
    payload: dict[str, Any] = {
        "resumed_from_checkpoint": latest,
        "world_note": "fresh world process: board and drift-p re-rolled",
    }
    if marker_extra:
        payload.update(marker_extra)
    env_server.fire_sham_perturbation("resume_marker", payload)
    return latest
