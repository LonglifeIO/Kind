"""The ``ReadingSurface`` enum — a leaf vocabulary module (no heavy deps).

``ReadingSurface`` names the three surfaces the mirror produces readings *at*
(substrate-side, head-internal, behavior-side; Probe 2 v2 synthesis §2.2). It is
shared vocabulary used by both the mirror layer (``kind.mirror.registry``, which
re-exports it) and the observer layer (``kind.observer.pre_reg``, whose
``PreRegistration`` keys two dicts with it).

**Why it lives here (Phase 8a).** Previously the enum was defined in
``kind.mirror.registry`` and imported by ``kind.observer.pre_reg``. That inverted
the layering (observer reaching up into mirror) and created a real import cycle:
``pre_reg`` → ``mirror.registry`` → (``mirror.__init__``'s eager calibration
chain) → ``mirror.orchestrator`` → ``pre_reg`` (partial). It only bit on a *cold*
import of ``pre_reg`` / ``kind.training.runner`` as the very first module in a
process; under the test suite a mirror-importing module loaded first and
pre-resolved it. Moving the enum to this observer-level leaf breaks the cycle at
its root: ``pre_reg`` now depends only on the observer layer, and
``mirror.registry`` re-exports the enum so every existing
``from kind.mirror.registry import ReadingSurface`` keeps working unchanged.

The enum is ``str``-valued, so members serialize identically to their string
values — moving the definition changes no JSON/Parquet serialization.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["ReadingSurface"]


class ReadingSurface(str, Enum):
    """Surfaces the mirror produces readings *at* (Probe 2 v2 synthesis §2.2).

    Values match the strings used by Phase 0's
    :data:`kind.mirror.structured.ReadingSurface` Literal so the str-valued enum
    members serialize identically to those strings.
    """

    SUBSTRATE_SIDE = "substrate_side"
    HEAD_INTERNAL = "head_internal"
    BEHAVIOR_SIDE = "behavior_side"
