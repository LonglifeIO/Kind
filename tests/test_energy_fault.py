"""Probe 4.5 Phase 2 — the fallible world (S-ENV + S-TEL) tests.

Authority: the frozen pre-registration §4
(``docs/decisions/probe4_5_preregistration_2026-07-13.md``) and the plan's
Phase 2 test list: determinism (same seeds → identical fault schedules);
statistics within the frozen band (duty cycle, spacing, duration);
default-off byte-identity; **no observation marker** (the render path is
untouched — the sensor honestly reports the fault's *consequences*, never
its *presence*); fault events validator-enforced; export pinned.

The frozen §4 values are pinned as config defaults; behavioral tests use
smaller test-scoped bands where the frozen gaps (first onset ≥ 151 steps)
would outlast a starving stationary agent's energy.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import EnvStep, GridWorld, GridWorldConfig
from kind.observer.fault_intervals import (
    FaultInterval,
    fault_intervals_from_events,
    fault_mask,
)
from kind.observer.schemas import (
    PROBE_4_5_WORLD_EVENT_SCHEMA_VERSION,
    WorldEvent,
    export_json_schema_v0_8_0,
    export_json_schema_v0_9_0,
)

# A test-scoped band small enough that faults land while a stationary
# agent still has energy (the frozen §4 gaps put the first onset past the
# stationary starvation horizon). NOT the frozen values — those are pinned
# in test_frozen_band_is_the_default below.
_FAST_FAULTS = dict(
    energy_fault_enabled=True,
    energy_fault_decay_multiplier=2.5,
    energy_fault_duration_min=5,
    energy_fault_duration_max=10,
    energy_fault_gap_base=10,
    energy_fault_gap_jitter=5,
)


def _fault_trace(
    config: GridWorldConfig, seed: int, actions: list[int]
) -> tuple[list[bool], list[float], list[float], list[bytes]]:
    """(fault_active, true_energy, sensed_energy, observation-bytes) per step."""
    world = GridWorld(config, seed=seed)
    step = world.reset()
    faults: list[bool] = []
    true_e: list[float] = []
    sensed: list[float] = []
    obs: list[bytes] = []
    for action in actions:
        step = world.step(action)
        faults.append(world.fault_active)
        true_e.append(world.state.true_energy)
        sensed.append(step.sensed_energy)
        obs.append(step.observation.tobytes())
    return faults, true_e, sensed, obs


# ---- the frozen band ------------------------------------------------------


def test_frozen_band_is_the_default() -> None:
    """The prereg §4 values are the config defaults (off by default)."""
    c = GridWorldConfig()
    assert c.energy_fault_enabled is False
    assert c.energy_fault_decay_multiplier == 2.5
    assert c.energy_fault_duration_min == 20
    assert c.energy_fault_duration_max == 40
    assert c.energy_fault_gap_base == 150
    assert c.energy_fault_gap_jitter == 300


def test_schedule_statistics_within_frozen_band() -> None:
    """Realized durations in {20..40}, gaps in {150..450}, duty ≈ 9% and
    under the 15% ceiling (prereg §4 — asserted on the generator's realized
    statistics)."""
    config = GridWorldConfig(energy_fault_enabled=True)
    n_steps = 12_000
    world = GridWorld(config, seed=20260713)
    world.reset()
    active: list[bool] = []
    for _ in range(n_steps):
        world.step(4)  # stay: the schedule is action-independent
        active.append(world.fault_active)
    arr = np.asarray(active)

    # Extract complete runs of active / inactive steps.
    change = np.flatnonzero(np.diff(arr.astype(np.int8)))
    bounds = np.concatenate([[0], change + 1, [n_steps]])
    durations = []
    gaps = []
    for lo, hi in zip(bounds[:-2], bounds[1:-1]):  # complete runs only
        run_len = int(hi - lo)
        if arr[lo]:
            durations.append(run_len)
        elif lo > 0:  # the leading gap is truncated by reset, skip it
            gaps.append(run_len)
    assert len(durations) >= 20, "too few intervals for statistics"
    assert min(durations) >= 20 and max(durations) <= 40
    assert min(gaps) >= 150 and max(gaps) <= 450
    duty = float(arr.mean())
    assert duty <= 0.15, f"duty ceiling violated: {duty}"
    assert 0.05 <= duty <= 0.13, f"duty far from the ≈9% design point: {duty}"


# ---- determinism ----------------------------------------------------------


def test_same_seed_same_schedule_and_energy() -> None:
    rng = np.random.default_rng(3)
    actions = [int(a) for a in rng.integers(0, 5, 400)]
    config = GridWorldConfig(**_FAST_FAULTS)  # type: ignore[arg-type]
    t1 = _fault_trace(config, seed=11, actions=actions)
    t2 = _fault_trace(config, seed=11, actions=actions)
    assert t1 == t2


def test_schedule_is_action_independent() -> None:
    """The fault process is a pure function of seed + step count — actions
    cannot influence it (nothing Io does can steer the world's faults)."""
    config = GridWorldConfig(**_FAST_FAULTS)  # type: ignore[arg-type]
    stay = _fault_trace(config, seed=11, actions=[4] * 400)
    rng = np.random.default_rng(5)
    moving = _fault_trace(
        config, seed=11, actions=[int(a) for a in rng.integers(0, 4, 400)]
    )
    assert stay[0] == moving[0]  # identical fault_active traces


# ---- default-off byte-identity + opacity ----------------------------------


def test_fault_touches_nothing_spatial_and_leaves_no_marker() -> None:
    """Same seed, faults on vs off: every observation byte-identical (no
    marker anywhere in the render path; the fifth RNG stream leaves the
    original four untouched), while the energy consequences diverge from
    the first onset — the sensor reports consequences, never presence."""
    rng = np.random.default_rng(7)
    actions = [int(a) for a in rng.integers(0, 5, 120)]
    base = dict(episode_length=50)
    off = _fault_trace(
        GridWorldConfig(**base),  # type: ignore[arg-type]
        seed=23,
        actions=actions,
    )
    on = _fault_trace(
        GridWorldConfig(**base, **_FAST_FAULTS),  # type: ignore[arg-type]
        seed=23,
        actions=actions,
    )
    assert on[3] == off[3], "observations must be byte-identical"
    assert not any(off[0]), "faults off → never active"
    assert any(on[0]), "faults on → intervals occurred"
    first_onset = on[0].index(True)
    assert on[1][:first_onset] == off[1][:first_onset]
    assert on[1][first_onset] < off[1][first_onset], (
        "the onset step must decay faster (consequences are real)"
    )
    # The honest sensor keeps tracking the *faulted* truth: with lag 1 and
    # σ=0.05 the sensed value stays within noise+quantum of the lagged true.
    lagged_true = [off[1][0]] + on[1][:-1]  # lag-1 series under faults...
    quantum = 1.0 / (GridWorldConfig().energy_obs_quantization_levels - 1)
    for t in range(first_onset, len(actions)):
        assert abs(on[2][t] - lagged_true[t]) <= 4 * 0.05 + quantum


def test_no_fault_field_on_the_agent_visible_record() -> None:
    """EnvStep (the only agent-visible surface) carries no fault field."""
    field_names = {f.name for f in dataclasses.fields(EnvStep)}
    assert not any("fault" in name for name in field_names)


def test_energy_floor_still_non_terminal_under_faults() -> None:
    """Faults accelerate depletion; they never terminate. A stationary
    starving agent floors at 0 and the env keeps stepping."""
    config = GridWorldConfig(**_FAST_FAULTS)  # type: ignore[arg-type]
    world = GridWorld(config, seed=1)
    world.reset()
    for _ in range(400):
        step = world.step(4)
    assert world.state.true_energy == 0.0
    assert step.env_step == 400  # still running


# ---- config validation ----------------------------------------------------


def test_fault_config_validation() -> None:
    with pytest.raises(ValueError, match="decay_multiplier"):
        GridWorld(
            GridWorldConfig(
                energy_fault_enabled=True, energy_fault_decay_multiplier=0.5
            ),
            seed=0,
        )
    with pytest.raises(ValueError, match="duration"):
        GridWorld(
            GridWorldConfig(
                energy_fault_enabled=True,
                energy_fault_duration_min=30,
                energy_fault_duration_max=20,
            ),
            seed=0,
        )
    with pytest.raises(ValueError, match="gap_base"):
        GridWorld(
            GridWorldConfig(
                energy_fault_enabled=True, energy_fault_gap_base=0
            ),
            seed=0,
        )
    # Disabled → the fault knobs are inert and unvalidated (house pattern:
    # default-off configs cannot be misconfigurations).
    GridWorld(
        GridWorldConfig(
            energy_fault_enabled=False, energy_fault_decay_multiplier=0.5
        ),
        seed=0,
    )


# ---- S-TEL: validator, emission, join -------------------------------------


def _fault_event(
    t: int, transition: str, *, schema_version: str = "0.5.0"
) -> WorldEvent:
    return WorldEvent(
        schema_version=schema_version,
        run_id="r",
        checkpoint_id=None,
        t_event=t,
        event_type="energy_fault_event",
        source="environment",
        payload={"transition": transition, "decay_multiplier": 2.5},
        wallclock_ms=0,
    )


def test_fault_event_validator_enforces_version_and_payload() -> None:
    _fault_event(10, "onset")  # well-formed constructs
    with pytest.raises(ValueError, match="schema_version"):
        _fault_event(10, "onset", schema_version="0.4.0")
    with pytest.raises(ValueError, match="missing"):
        WorldEvent(
            schema_version=PROBE_4_5_WORLD_EVENT_SCHEMA_VERSION,
            run_id="r",
            checkpoint_id=None,
            t_event=10,
            event_type="energy_fault_event",
            source="environment",
            payload={"transition": "onset"},
            wallclock_ms=0,
        )
    with pytest.raises(ValueError, match="onset.*offset"):
        _fault_event(10, "sideways")


def test_env_server_emits_fault_edges_boundaries_included(
    tmp_path: Path,
) -> None:
    """Stepping through episode boundaries, every fault edge is emitted and
    the observer-side join reconstructs the per-step fault state exactly —
    a dropped boundary edge would invert the mask for a whole interval."""
    events: list[WorldEvent] = []
    config = EnvServerConfig(
        grid_world_config=GridWorldConfig(
            episode_length=7,  # boundaries land inside fault cycles
            **_FAST_FAULTS,  # type: ignore[arg-type]
        ),
        seed=99,
        world_event_handler=events.append,
        run_id="fault-emission-test",
    )
    server = EnvServer(config)
    server.start()
    n_steps = 200
    state_mask: list[bool] = []
    for _ in range(n_steps):
        server.step(4)
        state_mask.append(server.grid_world_state.energy_fault_active)
    server.close()

    fault_events = [
        e.model_dump() for e in events if e.event_type == "energy_fault_event"
    ]
    assert fault_events, "no fault edges emitted"
    assert all(
        e["schema_version"] == PROBE_4_5_WORLD_EVENT_SCHEMA_VERSION
        and e["source"] == "environment"
        and e["payload"]["decay_multiplier"] == 2.5
        for e in fault_events
    )
    intervals = fault_intervals_from_events(fault_events)
    joined = fault_mask(intervals, t_start=1, t_end=n_steps + 1)
    assert joined.tolist() == state_mask


def test_fault_join_semantics_and_malformed_streams() -> None:
    events = [
        {"event_type": "energy_fault_event", "t_event": 5,
         "payload": {"transition": "onset", "decay_multiplier": 2.5}},
        {"event_type": "env_reset", "t_event": 6, "payload": {}},  # ignored
        {"event_type": "energy_fault_event", "t_event": 9,
         "payload": {"transition": "offset", "decay_multiplier": 2.5}},
        {"event_type": "energy_fault_event", "t_event": 15,
         "payload": {"transition": "onset", "decay_multiplier": 2.5}},
    ]
    intervals = fault_intervals_from_events(events)
    assert intervals == (
        FaultInterval(t_onset=5, t_offset=9),
        FaultInterval(t_onset=15, t_offset=None),  # open at stream end
    )
    # [t_onset, t_offset): steps 5–8 faulted, 9 not; open interval extends.
    mask = fault_mask(intervals, t_start=0, t_end=20)
    assert mask.tolist() == [
        t in {5, 6, 7, 8} or t >= 15 for t in range(20)
    ]
    with pytest.raises(ValueError, match="double onset"):
        fault_intervals_from_events(
            [events[0], {**events[0], "t_event": 6}]
        )
    with pytest.raises(ValueError, match="offset without onset"):
        fault_intervals_from_events([events[2]])
    with pytest.raises(ValueError, match="not time-ordered"):
        fault_intervals_from_events([events[3], events[0]])


# ---- export pinned --------------------------------------------------------


def test_v0_9_0_export_byte_stable_matches_disk_and_carries_the_fault() -> None:
    """Deterministic, matches the checked-in ``schemas/v0.9.0.json``, and
    advertises the fault event + record version. If this fails after an
    intentional schema change, regenerate via export_json_schema_v0_9_0 and
    commit the result."""
    first = export_json_schema_v0_9_0()
    assert first == export_json_schema_v0_9_0()
    on_disk = (
        Path(__file__).resolve().parents[1] / "schemas" / "v0.9.0.json"
    ).read_bytes()
    assert first == on_disk, (
        "schemas/v0.9.0.json is out of date — regenerate via "
        "export_json_schema_v0_9_0() and commit the result."
    )
    document = json.loads(first)
    assert document["schema_version"] == "0.9.0"
    assert document["world_event_schema_version"] == (
        PROBE_4_5_WORLD_EVENT_SCHEMA_VERSION
    )
    enum = document["models"]["telemetry"]["WorldEvent"]["properties"][
        "event_type"
    ]["enum"]
    assert "energy_fault_event" in enum


def test_v0_8_0_export_now_frozen_reads_checked_in_bytes() -> None:
    """The Probe 4 export is a frozen historical artifact after the 4.5
    widening (house pattern): it returns the checked-in bytes, which do NOT
    carry the fault event."""
    on_disk = (
        Path(__file__).resolve().parents[1] / "schemas" / "v0.8.0.json"
    ).read_bytes()
    frozen = export_json_schema_v0_8_0()
    assert frozen == on_disk
    enum = json.loads(frozen)["models"]["telemetry"]["WorldEvent"][
        "properties"
    ]["event_type"]["enum"]
    assert "energy_fault_event" not in enum
