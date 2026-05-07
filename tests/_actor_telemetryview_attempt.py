"""Fixture for ``test_actor_forward_telemetryview_argument_fails_mypy_strict``.

This file is intentionally type-incorrect. The test
``test_actor_forward_telemetryview_argument_fails_mypy_strict`` invokes
mypy ``--strict`` against this file via subprocess and asserts the
type error is reported. The file constructs a call site that passes a
``TelemetryView`` where a ``PolicyView`` is expected, which violates
the synthesis §Q5 / plan §7 Level 2 type-signature opacity boundary.

Do NOT import this file from production code or other tests — it would
fail mypy at import time. The leading underscore on the filename is the
convention that pytest's collection mechanism does not pick up; mypy is
invoked on this file explicitly by name from the test that needs it.

Probe 1.5 v2: this fixture verifies that the type-level opacity boundary
extends to the new ``self_prediction``, ``self_prediction_error_masked``
fields on ``TelemetryView`` and the new ``self_prediction_error`` field
on ``PolicyView`` — a ``TelemetryView`` argument is *not* a valid
``PolicyView`` even though both share ``h``, ``z`` and the scalar reads
the same name on PolicyView. Structural typing has no role here; mypy
``--strict`` catches the nominal mismatch.
"""

from __future__ import annotations

from kind.agents.actor import Actor
from kind.agents.views import TelemetryView


def call_actor_with_telemetry_view(actor: Actor, view: TelemetryView) -> None:
    """Deliberately type-incorrect call. mypy --strict reports:

        error: Argument 1 to "forward" of "Actor" has incompatible type
        "TelemetryView"; expected "PolicyView"

    The line below has NO ``type: ignore`` — the whole point of this
    fixture is for the test to run mypy and assert the error is reported.
    """
    actor.forward(view)
