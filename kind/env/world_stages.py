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

Later stages (e1 trail, e2 hidden clock, e3 weather-food, e4 mover)
are added by their own phases; requesting an undefined stage raises.
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

WORLD_STAGES: Final[tuple[str, ...]] = ("default", "e0")


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
    raise ValueError(
        f"unknown world stage {stage!r}; defined stages: {WORLD_STAGES}"
    )
