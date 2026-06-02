"""Cold-import regression (Phase 8a) — the observer→mirror cycle stays broken.

Before Phase 8a, importing ``kind.training.runner`` (or ``kind.observer.pre_reg``)
as the *very first* module in a fresh process raised ``ImportError`` from a real
cycle: ``pre_reg`` → ``mirror.registry`` → (``mirror.__init__``'s eager
calibration chain) → ``mirror.orchestrator`` → ``pre_reg`` (partial). It was
masked under the suite because a mirror-importing test loaded first. Phase 8a
moved the shared ``ReadingSurface`` enum to the observer-level leaf
``kind.observer.reading_surface`` so the observer layer no longer reaches up into
the mirror layer. These tests import each module *cold* in a subprocess (a clean
interpreter, no other imports first) to prove the cycle stays broken.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "kind.training.runner",
        "kind.observer.pre_reg",
        "kind.mirror.registry",
        "kind.mirror",
    ],
)
def test_module_imports_cold_in_a_fresh_process(module: str) -> None:
    """Each module imports cleanly as the first import in a fresh interpreter."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"cold `import {module}` failed (the observer→mirror cycle is back?):\n"
        f"{result.stderr}"
    )


def test_reading_surface_reexport_identity() -> None:
    """``mirror.registry`` re-exports the *same* enum object the leaf defines, so
    no consumer of ``from kind.mirror.registry import ReadingSurface`` changes."""
    from kind.mirror.registry import ReadingSurface as FromRegistry
    from kind.observer.reading_surface import ReadingSurface as FromLeaf

    assert FromRegistry is FromLeaf
    assert [s.value for s in FromRegistry] == [
        "substrate_side",
        "head_internal",
        "behavior_side",
    ]
