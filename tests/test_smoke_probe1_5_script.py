"""Phase 6 sanity test: ``scripts/smoke_probe1_5.py`` exists and is importable.

The actual MPS smoke is run manually by the human builder on the
canonical Mac (``python scripts/smoke_probe1_5.py``); pytest only checks
that the script is present, importable, and exposes the expected
``main()`` entry point. Running the script automatically in pytest
would require MPS, which is platform-specific, and would side-track
the gate-time test suite from its CPU-correctness focus (plan §5).

Phase 5's journal entry (under "_load_script_module() and the
sys.modules registration") flagged a Python 3.14 trap: scripts that
define ``@dataclass``-decorated classes at module level fail to load
via ``spec_from_file_location`` + ``exec_module`` unless the module is
registered in ``sys.modules`` *before* ``exec_module`` runs. The
Probe 1.5 smoke does not currently define any ``@dataclass`` classes
(it uses a plain class with ``__slots__`` for the per-step result
record, intentionally avoiding the trap), but this loader registers
the module in ``sys.modules`` regardless — defense in depth, and
symmetric with ``tests/test_probe1_5_control_frozen_target.py``'s
loader. The autouse ``_restore_sys_modules`` fixture cleans up after
each test so re-loads are clean.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "smoke_probe1_5.py"
_MODULE_NAME = "kind_smoke_probe1_5_module"


def _load_script_module() -> ModuleType:
    """Load the script as a fresh module under a unique name.

    See the module docstring on the sys.modules registration discipline.
    """
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_smoke_probe1_5_script_exists() -> None:
    """The script file is checked into ``scripts/smoke_probe1_5.py``."""
    assert _SCRIPT_PATH.is_file(), (
        f"plan §5 expects {_SCRIPT_PATH} to exist as a runnable script"
    )


def test_smoke_probe1_5_script_is_importable_and_has_main() -> None:
    """Loading the module does not error and ``main()`` is exposed.

    The contract: ``main()`` exists and is callable; the ``__main__``
    block routes its return value through ``sys.exit``. The signature
    is not asserted beyond callability — the smoke is meant to be
    invoked without arguments (``python scripts/smoke_probe1_5.py``),
    matching ``scripts/smoke_mps.py``'s convention.
    """
    module = _load_script_module()
    assert hasattr(module, "main"), (
        "scripts/smoke_probe1_5.py must define main() — the script's "
        "entry-point function the __main__ block routes through sys.exit"
    )
    assert callable(module.main)


def test_smoke_probe1_5_script_uses_v0_2_0_schema_version() -> None:
    """The smoke writes synthetic AgentStep records at SCHEMA_VERSION
    (``"0.2.0"``) per plan §5.1 step 8 — the round-trip exercise is
    meaningful only against the v2 schema. The DreamRollout, ReplayMeta,
    and WorldEvent records continue to stamp PROBE_1_SCHEMA_VERSION as
    the Phase 0 writer-migration discipline names; only AgentStep and
    DreamRollout writers are migrated to ``"0.2.0"`` at Probe 1.5.
    """
    module = _load_script_module()
    # The constants the smoke's _exercise_sinks references are imported
    # at module load time from kind.observer.schemas; the test verifies
    # the import worked and the v2 constant has the expected value.
    assert module.SCHEMA_VERSION == "0.2.0", (
        f"smoke_probe1_5 must use schema_version 0.2.0; got "
        f"{module.SCHEMA_VERSION!r}"
    )
    assert module.PROBE_1_SCHEMA_VERSION == "0.1.0", (
        f"smoke_probe1_5 must keep PROBE_1_SCHEMA_VERSION as 0.1.0 for "
        f"the unmigrated writers; got {module.PROBE_1_SCHEMA_VERSION!r}"
    )


def test_smoke_probe1_5_script_exposes_thresholds_per_plan_5_3_5_4() -> None:
    """Plan §5.3 and §5.4 name specific thresholds the smoke decides
    pass/fail/warn against. The script exposes them as module-level
    constants so a future builder can audit them against §9.1's
    Probe-1-relative calibration without reading the loop body.

    The values pinned here are the plan §9.1 defaults plus the Phase 6
    calibration of the world-model gradient-norm bar. Plan §9.1: "1000
    is a stance call ... the build phase tunes if smoke surfaces a
    tighter bound." Phase 6's smoke surfaced the inverse — Probe 1
    baseline ~25k, Probe 1.5 baseline ~45k — and the threshold was
    raised to 1e6 to catch genuine pathologies without false-alarming
    on the substrate's normal first-step behavior under random obs.
    The other thresholds are unchanged from §9.1: Probe 1's
    early-mean × 0.7 for the KL floor, late-mean × 1.5 for recon
    climbing, BYOL/SPR convention 100× for EMA divergence.
    """
    module = _load_script_module()
    assert module._WORLD_MODEL_GRAD_NORM_HARD_BAR == 1e6, (
        "plan §5.3 / §9.1 + Phase 6 calibration: world-model "
        "gradient-norm hard bar is 1e6 (raised from the 1000 stance "
        "call after smoke surfaced ~45k baseline at production sizes)"
    )
    assert module._EMA_DIVERGENCE_HARD_BAR == 100.0, (
        "plan §5.3 / §9.1: EMA target divergence hard bar is 100× "
        "online L2"
    )
    assert module._KL_FLOOR_SOFT_BAR == 7.16, (
        "plan §5.4 / §9.1: KL floor soft bar is 0.7 × Probe 1's early "
        "mean of 10.23 = 7.16"
    )
    assert module._RECON_CLIMB_SOFT_BAR == 48.68, (
        "plan §5.4 / §9.1: recon climb soft bar is 1.5 × Probe 1's "
        "late mean of 32.45 = 48.68"
    )
    assert module._RUN_LEN_SOFT_BAR == 20, (
        "plan §5.4: KL/recon soft warnings fire at >20 consecutive "
        "steps"
    )
    assert module._ACTOR_NEW_COL_GRAD_FLOOR == 1e-6, (
        "plan §5.4: actor new-input-column gradient-norm soft warning "
        "floor is 1e-6 (synthesis §1.7(a) failure-mode (a) early signal)"
    )


def test_smoke_probe1_5_script_uses_production_sizes() -> None:
    """Plan §5.1: smoke runs at production sizes (h=200, z=16, K=5,
    head_hidden=200, batch=16, seq=32, ema_decay=0.99). The script
    pins these as module-level constants so the smoke surfaces
    issues at the same scale the env-coupled run will hit.
    """
    module = _load_script_module()
    assert module._H_DIM == 200
    assert module._Z_DIM == 16
    assert module._K_ENSEMBLE == 5
    assert module._HEAD_HIDDEN == 200
    assert module._BATCH_SIZE == 16
    assert module._SEQUENCE_LENGTH == 32
    assert module._EMA_DECAY == 0.99
    assert module._NUM_TRAINING_STEPS == 100
    assert module._TARGET_MODE == "online"
    assert module._LOSS_FORM == "cosine"


@pytest.fixture(autouse=True)
def _restore_sys_modules() -> Any:
    """Each ``_load_script_module()`` call exec's the script under a
    fresh module name. The fixture removes the module from
    ``sys.modules`` after each test so re-loads pick up file changes
    (relevant only if the script is edited mid-test-session) and so
    state doesn't leak between tests."""
    yield
    sys.modules.pop(_MODULE_NAME, None)
