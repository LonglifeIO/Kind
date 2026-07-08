"""Probe 4 Phase 2 — perturbation generator unit tests (plan §S-PERT).

The generator is a pure decision engine; these tests pin determinism
(given seed + trajectory), the spacing law, the not-self decoupling in
place (exclusion radius) and time (consumption deferral), and the
adjacent-cluster placement.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from kind.env.grid_world import CellType
from kind.env.perturbation_generator import (
    PerturbationGenerator,
    PerturbationGeneratorConfig,
    PlannedPerturbation,
)


def _empty_grid(size: int = 8) -> NDArray[np.uint8]:
    return np.full((size, size), CellType.EMPTY.value, dtype=np.uint8)


def _poll_run(
    generator: PerturbationGenerator,
    *,
    steps: int,
    agent_pos: tuple[int, int] = (0, 0),
) -> list[tuple[int, PlannedPerturbation]]:
    """Poll the generator over a static empty world; collect firings."""
    fired: list[tuple[int, PlannedPerturbation]] = []
    for env_step in range(1, steps + 1):
        planned = generator.poll(
            env_step=env_step,
            grid=_empty_grid(),
            agent_pos=agent_pos,
            consumed_last_step=False,
        )
        if planned is not None:
            fired.append((env_step, planned))
    return fired


# ---- config validation -----------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_spacing_steps": 0},
        {"spacing_jitter_steps": -1},
        {"cells_per_event": 0},
        {"exclusion_radius": -1},
    ],
)
def test_config_validation(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        PerturbationGeneratorConfig(seed=0, **kwargs)


# ---- determinism ------------------------------------------------------------


def test_deterministic_given_seed_and_trajectory() -> None:
    config = PerturbationGeneratorConfig(
        seed=11, min_spacing_steps=5, spacing_jitter_steps=3
    )
    run_a = _poll_run(PerturbationGenerator(config), steps=100)
    run_b = _poll_run(PerturbationGenerator(config), steps=100)
    assert run_a == run_b
    assert len(run_a) >= 5  # spacing <= 8 over 100 steps


def test_different_seeds_differ() -> None:
    base = {"min_spacing_steps": 5, "spacing_jitter_steps": 5}
    run_a = _poll_run(
        PerturbationGenerator(PerturbationGeneratorConfig(seed=1, **base)),
        steps=200,
    )
    run_b = _poll_run(
        PerturbationGenerator(PerturbationGeneratorConfig(seed=2, **base)),
        steps=200,
    )
    assert run_a != run_b


# ---- spacing law ------------------------------------------------------------


def test_inter_event_spacing_respects_floor_and_jitter() -> None:
    config = PerturbationGeneratorConfig(
        seed=3, min_spacing_steps=6, spacing_jitter_steps=4
    )
    fired = _poll_run(PerturbationGenerator(config), steps=300)
    steps = [env_step for env_step, _ in fired]
    assert len(steps) >= 10
    gaps = [b - a for a, b in zip(steps, steps[1:])]
    # With no deferrals (static empty world, no consumption) each event
    # fires exactly when due, so gaps realize the drawn law.
    assert all(6 <= gap <= 10 for gap in gaps)
    # Jitter is actually exercised (not a constant cadence).
    assert len(set(gaps)) > 1


def test_no_fire_before_first_due_step() -> None:
    config = PerturbationGeneratorConfig(
        seed=4, min_spacing_steps=10, spacing_jitter_steps=0
    )
    generator = PerturbationGenerator(config)
    for env_step in range(1, 10):
        assert (
            generator.poll(
                env_step=env_step,
                grid=_empty_grid(),
                agent_pos=(0, 0),
                consumed_last_step=False,
            )
            is None
        )
    assert (
        generator.poll(
            env_step=10,
            grid=_empty_grid(),
            agent_pos=(0, 0),
            consumed_last_step=False,
        )
        is not None
    )


# ---- not-self: place -------------------------------------------------------


def test_cells_respect_exclusion_radius() -> None:
    config = PerturbationGeneratorConfig(
        seed=5,
        min_spacing_steps=2,
        spacing_jitter_steps=0,
        exclusion_radius=2,
    )
    agent_pos = (4, 4)
    fired = _poll_run(
        PerturbationGenerator(config), steps=200, agent_pos=agent_pos
    )
    assert fired
    for _, planned in fired:
        for r, c in planned.cells:
            chebyshev = max(abs(r - agent_pos[0]), abs(c - agent_pos[1]))
            assert chebyshev > 2, (
                f"cell ({r},{c}) inside exclusion radius of {agent_pos}"
            )


def test_defers_when_no_eligible_cells() -> None:
    """A world with no eligible cells (all resources) defers the event;
    the event fires later when eligibility returns."""
    config = PerturbationGeneratorConfig(
        seed=6, min_spacing_steps=3, spacing_jitter_steps=0
    )
    generator = PerturbationGenerator(config)
    full_grid = np.full((8, 8), CellType.RESOURCE.value, dtype=np.uint8)
    for env_step in range(1, 10):
        assert (
            generator.poll(
                env_step=env_step,
                grid=full_grid,
                agent_pos=(0, 0),
                consumed_last_step=False,
            )
            is None
        )
    # Eligibility returns → the deferred event fires now.
    assert (
        generator.poll(
            env_step=10,
            grid=_empty_grid(),
            agent_pos=(0, 0),
            consumed_last_step=False,
        )
        is not None
    )


# ---- not-self: time ---------------------------------------------------------


def test_defers_on_consumption_co_timing() -> None:
    config = PerturbationGeneratorConfig(
        seed=7, min_spacing_steps=3, spacing_jitter_steps=0
    )
    generator = PerturbationGenerator(config)
    due = generator.next_due_step
    assert (
        generator.poll(
            env_step=due,
            grid=_empty_grid(),
            agent_pos=(0, 0),
            consumed_last_step=True,
        )
        is None
    )
    # Next boundary without a consumption: fires.
    assert (
        generator.poll(
            env_step=due + 1,
            grid=_empty_grid(),
            agent_pos=(0, 0),
            consumed_last_step=False,
        )
        is not None
    )


def test_consumption_deferral_can_be_disabled() -> None:
    config = PerturbationGeneratorConfig(
        seed=7,
        min_spacing_steps=3,
        spacing_jitter_steps=0,
        defer_on_consumption=False,
    )
    generator = PerturbationGenerator(config)
    assert (
        generator.poll(
            env_step=generator.next_due_step,
            grid=_empty_grid(),
            agent_pos=(0, 0),
            consumed_last_step=True,
        )
        is not None
    )


# ---- cluster placement ------------------------------------------------------


def test_cluster_is_adjacent_and_sized() -> None:
    config = PerturbationGeneratorConfig(
        seed=8, min_spacing_steps=2, spacing_jitter_steps=0, cells_per_event=3
    )
    fired = _poll_run(PerturbationGenerator(config), steps=100)
    assert fired
    for _, planned in fired:
        cells = planned.cells
        assert len(cells) == 3
        assert len(set(cells)) == 3
        # Connectivity: every cell after the anchor is 4-adjacent to some
        # earlier cluster member (the growth invariant).
        for index, (r, c) in enumerate(cells[1:], start=1):
            assert any(
                abs(r - pr) + abs(c - pc) == 1 for pr, pc in cells[:index]
            )


def test_single_cell_event() -> None:
    config = PerturbationGeneratorConfig(
        seed=9, min_spacing_steps=2, spacing_jitter_steps=0, cells_per_event=1
    )
    fired = _poll_run(PerturbationGenerator(config), steps=20)
    assert fired
    assert all(len(planned.cells) == 1 for _, planned in fired)
