"""World v2 stage presets (plan §"The stage presets").

Cumulative, named world configurations for the biography launcher's
``--world-stage`` flag. Each stage is a dated, journaled world-change
event that arrives in Io's continuing life via checkpoint-resume; the
exact values are stimulus knobs (DP5), revisable at pauses, never
success criteria.

Stages defined so far:

* ``default`` — today's world, byte-identical (the flag exists so a
  fresh launch and a pre-e0 resume stay expressible).
* ``e0`` — the world stops forgetting (synthesis E0): no board
  resample, plus a small persistent terrain feature — one L-shaped
  corridor of wall cells that does not partition the grid (S3's
  trivial-loop confound is the constraint; connectivity is
  test-enforced).
* ``e1`` — + the somatic trail (synthesis E1): cells Io vacates are
  stamped ``CellType.TRAIL`` and decay back over ``TRAIL_DECAY_STEPS``
  steps — self-caused, spatially extended, frequently observed
  dynamics (S1's verified capacity preference), the cheapest contact
  pilot (C1-cautious).
* ``e2`` — + the hidden clock (synthesis E2): an unobserved phase at
  ``BLOOM_CELL`` blooms its Moore ring in the trail vocabulary every
  ``BLOOM_PERIOD`` steps for ``BLOOM_DURATION`` steps — the cleanest
  "exercise h as a clock" structure; one clock only (the
  temporal-superposition warning).
* ``e3`` — + weather-food (synthesis E3): uniform regrowth replaced by
  one drifting 3×3 patch on a deterministic bounce law — regrowth
  concentrated under it, sparse elsewhere. Food reads as process, not
  confetti; break-even stays possible but not ambient. Crowd-out (C4)
  is watched by the boundary analyzer's occupancy-share diagnostic.
* ``e4`` — + one autonomous mover (synthesis E4): a wandering
  WALL-vocabulary cell, displaced one cell by Io's contact. **A pilot,
  removable at any pause without ceremony (DP3)** — if its
  disagreement never localizes, removal is a capacity finding, not a
  failure.

Requesting an undefined stage raises.
"""

from __future__ import annotations

import dataclasses
from typing import Final

from kind.env.grid_world import GridWorldConfig

# The e0 terrain: one interior L (6 cells) — a corridor shape touching
# no grid edge, so the 8×8 grid cannot be partitioned. Chosen at build
# time as a stimulus knob (DP5); journaled in
# ``docs/workingjournal/worldv2.md``.
E0_WALLS: Final[tuple[tuple[int, int], ...]] = (
    (2, 2),
    (3, 2),
    (4, 2),
    (5, 2),
    (5, 3),
    (5, 4),
)

# The e1 decay horizon (steps a footprint persists): the synthesis's
# ~40–60 band, taken at the plan's midpoint. A stimulus knob (DP5).
TRAIL_DECAY_STEPS: Final[int] = 50

# The e2 clock (stimulus knobs, DP5): the source sits in the open
# quadrant away from the E0 corridor — its 8-cell Moore ring is fully
# in bounds and wall-free. Period ~12 (the plan's value; inside the
# measured h-trace horizon of ~40 and the BPTT window of 32, so the
# phase is carryable); blooms last 2 steps.
BLOOM_CELL: Final[tuple[int, int]] = (6, 6)
BLOOM_PERIOD: Final[int] = 12
BLOOM_DURATION: Final[int] = 2

# The e3 weather (stimulus knobs, DP5; plan preset values). At these
# rates an 8×8 board carries ~9 patch cells at 0.06 and ~49 outside
# cells at 0.001 → ~0.6 regrowths/step near the patch, ~0.05 far from
# it: foraging under the weather is comfortably above break-even
# (~0.18 meals/step), grazing far from it is not — possible, never
# ambient. The patch starts at the grid center heading (1,1); its size
# and law live in GridWorldConfig defaults.
PATCH_STEP_EVERY: Final[int] = 20
PATCH_P_INSIDE: Final[float] = 0.06
PATCH_P_OUTSIDE: Final[float] = 0.001

# The e4 mover (stimulus knobs, DP5): starts in the corner opposite
# the E0 corridor, moves every 2 steps with the plan's turn hazard.
MOVER_START: Final[tuple[int, int]] = (0, 7)
MOVER_STEP_EVERY: Final[int] = 2
MOVER_TURN_HAZARD: Final[float] = 0.02

WORLD_STAGES: Final[tuple[str, ...]] = (
    "default",
    "e0",
    "e1",
    "e2",
    "e3",
    "e4",
)


def apply_world_stage(config: GridWorldConfig, stage: str) -> GridWorldConfig:
    """Return ``config`` with the named stage's world changes applied.

    ``default`` returns ``config`` unchanged (byte-identical world).
    Stages are cumulative by construction — each later stage's function
    of ``config`` includes every earlier stage's changes.
    """
    if stage == "default":
        return config
    if stage == "e0":
        return dataclasses.replace(
            config,
            episode_resample=False,
            walls=E0_WALLS,
        )
    if stage == "e1":
        return dataclasses.replace(
            apply_world_stage(config, "e0"),
            trail_enabled=True,
            trail_decay_steps=TRAIL_DECAY_STEPS,
        )
    # Landing order re-ratified by the builder 2026-07-09 after the
    # session-4 e1 read (the trail's food-shadow starved the forage
    # loop; both dead phases were food-economy failures): weather (e3)
    # lands BEFORE the clock (e2). Stage names keep their synthesis
    # meanings; the cumulative chains encode the landing order
    # e0 → e1 → e3 → e2 → e4.
    if stage == "e3":
        return dataclasses.replace(
            apply_world_stage(config, "e1"),
            regrowth_mode="patch",
            patch_step_every=PATCH_STEP_EVERY,
            patch_p_inside=PATCH_P_INSIDE,
            patch_p_outside=PATCH_P_OUTSIDE,
        )
    if stage == "e2":
        return dataclasses.replace(
            apply_world_stage(config, "e3"),
            bloom_cell=BLOOM_CELL,
            bloom_period=BLOOM_PERIOD,
            bloom_duration=BLOOM_DURATION,
        )
    if stage == "e4":
        return dataclasses.replace(
            apply_world_stage(config, "e2"),
            mover_enabled=True,
            mover_start=MOVER_START,
            mover_step_every=MOVER_STEP_EVERY,
            mover_turn_hazard=MOVER_TURN_HAZARD,
        )
    raise ValueError(
        f"unknown world stage {stage!r}; defined stages: {WORLD_STAGES}"
    )
