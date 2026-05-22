"""Window — a read-only viewer over the mirror's on-disk records.

Window is practical infrastructure, not a substrate phase: it surfaces
the records Phases 9–13 wrote so the builder can check in on a long
Probe 4 run without hand-inspecting JSON. It is journaled as
"Phase 11.5" to make the deviation from the synthesis-spec'd phase
sequence visible while preserving the synthesis's phase numbering.

Window is consumed as a CLI (``scripts/run_window.py``); the public
library surface is just :func:`~kind.window.server.create_app`.
"""

from kind.window.server import create_app

__all__ = ["create_app"]
