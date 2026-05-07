"""Phase 8 sanity test: ``scripts/smoke_mps.py`` exists and is importable.

The actual MPS smoke is run manually by the human builder on the
canonical Mac (``python scripts/smoke_mps.py``); pytest only checks
that the script is present, importable, and exposes the expected
``main()`` entry point. Running the script automatically in pytest
would require MPS, which is platform-specific, and would side-track
the gate-time test suite from its CPU-correctness focus (plan §5).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "smoke_mps.py"


def test_smoke_mps_script_exists() -> None:
    """The script file is checked into ``scripts/smoke_mps.py``."""
    assert _SCRIPT_PATH.is_file(), (
        f"plan §5 expects {_SCRIPT_PATH} to exist as a runnable script"
    )


def test_smoke_mps_script_is_importable_and_has_main() -> None:
    """Loading the module does not error and ``main()`` is exposed.

    The module is loaded by spec rather than a regular ``import`` so
    pytest discovery does not pull it onto every test run. The
    contract: ``main()`` exists and is callable; the ``__main__`` block
    routes its return value through ``sys.exit``. The signature is not
    asserted beyond callability — the smoke is meant to be invoked
    without arguments (``python scripts/smoke_mps.py``).
    """
    spec = importlib.util.spec_from_file_location(
        "kind_smoke_mps_module", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main"), (
        "scripts/smoke_mps.py must define main() — the script's "
        "entry-point function the __main__ block routes through sys.exit"
    )
    assert callable(module.main)
