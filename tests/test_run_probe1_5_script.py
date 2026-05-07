"""Phase 7 sanity test: ``scripts/run_probe1_5.py`` exists, exposes
``main()`` and the public-by-name :func:`make_runner_config` helper, and
``--dry-run`` constructs a :class:`RunnerConfig` with
``self_prediction_target_mode='online'`` without standing up the env
server, transport, MPS, or the ``runs/`` directory.

The full env-coupled run is the human-builder-driven step (~16 minutes
on the canonical Mac per Phase 6's wall-time projection of 142.7 ms ×
5000 steps + warmup overhead). pytest's role is the structural-
correctness gate: the script is checked in, importable, exposes the
expected entry points, and the dry-run path is side-effect-free.

Plan §1 Phase 7 / build prompt:

1. Script exists, importable, exposes ``main()``.
2. ``make_runner_config`` is exposed as a public-by-name helper and
   produces a :class:`RunnerConfig` with
   ``self_prediction_target_mode='online'`` on both the runner-level
   field and the nested :class:`WorldModelConfig` field — symmetric
   with Phase 5's frozen-target control helper.
3. ``main(['--dry-run'])`` returns 0, prints a summary that names
   ``self_prediction_target_mode='online'``, does *not* create the
   ``runs/`` directory, and does *not* invoke side-effect-bearing
   modules (``Runner``, ``EnvServer``, ``EnvTransportClient``,
   ``EnvTransportServer``, ``_detect_mps_or_exit``).
4. Plan §1 Phase 7 + §8.4: seed=42, total_env_steps=5000.
5. Schema version constant pinned at ``"0.2.0"`` (the Phase 0 SCHEMA_VERSION
   the script imports from ``kind.observer.schemas``).
"""

from __future__ import annotations

import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "run_probe1_5.py"
_MODULE_NAME = "kind_run_probe1_5_module"


def _load_script_module() -> ModuleType:
    """Load the script as a fresh module under a unique name.

    The module is registered in ``sys.modules`` *before*
    ``exec_module`` runs because the script defines a ``@dataclass``
    class (``_Progress``) at module level. Phase 5's journal entry
    (under "_load_script_module() and the sys.modules registration")
    flagged the Python 3.14 trap: ``@dataclass`` resolves field type
    annotations at class-definition time via
    ``sys.modules.get(cls.__module__).__dict__``; if the module is not
    in ``sys.modules`` at decoration time, the dataclass machinery
    raises ``AttributeError: 'NoneType' object has no attribute
    '__dict__'``. The autouse ``_restore_sys_modules`` fixture below
    cleans up after each test.
    """
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_probe1_5_script_exists() -> None:
    """The script file is checked into ``scripts/run_probe1_5.py``."""
    assert _SCRIPT_PATH.is_file(), (
        f"plan §1 Phase 7 expects {_SCRIPT_PATH} to exist as a runnable "
        f"script"
    )


def test_run_probe1_5_script_is_importable_and_has_main() -> None:
    """Loading the module does not error and ``main()`` is exposed.

    The contract: ``main()`` exists and is callable; the ``__main__``
    block routes its return value through ``sys.exit``. The signature
    accepts ``argv: list[str] | None = None`` so test code (and the
    ``--dry-run`` flag) can pass argv directly.
    """
    module = _load_script_module()
    assert hasattr(module, "main"), (
        "scripts/run_probe1_5.py must define main() — the script's "
        "entry-point function the __main__ block routes through sys.exit"
    )
    assert callable(module.main)


def test_run_probe1_5_exposes_make_runner_config_helper() -> None:
    """The Phase 7 build prompt asks the test to verify ``--dry-run``
    constructs a ``RunnerConfig`` with ``self_prediction_target_mode=
    'online'`` without parsing dry-run stdout. The script exposes
    ``make_runner_config(...)`` (no leading underscore) so the test
    can call it directly — the Phase 5 pattern carried forward.
    """
    module = _load_script_module()
    assert hasattr(module, "make_runner_config"), (
        "scripts/run_probe1_5.py must expose make_runner_config(...) "
        "as a public-by-name helper for the Phase 7 dry-run test"
    )
    assert callable(module.make_runner_config)


def test_make_runner_config_sets_target_mode_to_online(
    tmp_path: Path,
) -> None:
    """``make_runner_config`` produces a ``RunnerConfig`` whose
    ``self_prediction_target_mode`` is ``'online'`` on both the
    runner-level field and the nested ``WorldModelConfig``.

    The runner's ``__init__`` does ``dataclasses.replace`` on the
    world-model config from the runner-level field; setting both to
    ``'online'`` here verifies the helper does not silently drop the
    Probe 1.5 main path on either surface (which would be a bug the
    runner's silent-replace would mask: if only the runner-level field
    is set, the substrate still ends up ``'online'`` *via* the
    runner's replace, but the dry-run summary would surface stale
    values until runner construction).
    """
    module = _load_script_module()
    config = module.make_runner_config(
        run_id="probe1_5-test",
        telemetry_dir=tmp_path / "telemetry",
        checkpoints_dir=tmp_path / "checkpoints",
    )

    assert config.self_prediction_target_mode == "online", (
        "RunnerConfig.self_prediction_target_mode must be 'online' for "
        "the Probe 1.5 main run (synthesis §1.2 / plan §6 row 1)"
    )
    assert (
        config.world_model_config.self_prediction_target_mode == "online"
    ), (
        "WorldModelConfig.self_prediction_target_mode must be 'online' "
        "for the Probe 1.5 main run; the dry-run summary reads this "
        "field directly"
    )


def test_make_runner_config_preserves_probe_1_seed_and_total_steps(
    tmp_path: Path,
) -> None:
    """Plan §1 Phase 7 + §8.4: the Probe 1.5 main run uses the same
    seed and total env steps as Probe 1's run for direct comparability.
    The seed is consumed by the env-server (via
    :func:`_make_env_server`), not the runner config; the test asserts
    the script's module-level constants. ``_TOTAL_ENV_STEPS`` is
    likewise a script-level constant the runner reads at run() time.

    The cadence deltas from plan §6 defaults (warmup, checkpoint,
    dream) match ``run_probe1.py`` so the Probe 1.5 main run's
    trajectories align with Probe 1's reference run on the cadence
    dimensions the comparison reads.
    """
    module = _load_script_module()
    assert getattr(module, "_SEED") == 42, (
        "plan §1 Phase 7: main run must use seed=42 (matches Probe 1)"
    )
    assert getattr(module, "_TOTAL_ENV_STEPS") == 5000, (
        "plan §1 Phase 7: main run must use total_env_steps=5000 "
        "(matches Probe 1)"
    )
    assert getattr(module, "_WARMUP_ENV_STEPS") == 200, (
        "plan §1 Phase 7: warmup_env_steps=200 (matches run_probe1.py)"
    )
    assert getattr(module, "_CHECKPOINT_EVERY_N_ENV_STEPS") == 2500, (
        "plan §1 Phase 7: checkpoint_every_n_env_steps=2500 (matches "
        "run_probe1.py)"
    )
    assert getattr(module, "_DREAM_CADENCE_ENV_STEPS") == 1000, (
        "plan §1 Phase 7: dream_cadence_env_steps=1000 (matches "
        "run_probe1.py and plan §6)"
    )


def test_run_probe1_5_pins_schema_version_at_v0_2_0() -> None:
    """The script imports ``SCHEMA_VERSION`` from ``kind.observer.schemas``
    so the dry-run summary surfaces the version the runner stamps on
    every emitted record. Phase 0 settled ``SCHEMA_VERSION = '0.2.0'``;
    the script must reference that constant rather than a literal so
    a future bump propagates automatically.
    """
    module = _load_script_module()
    assert module.SCHEMA_VERSION == "0.2.0", (
        f"Phase 0 settled SCHEMA_VERSION='0.2.0'; got "
        f"{module.SCHEMA_VERSION!r}"
    )


def test_run_probe1_5_target_mode_constant_is_online() -> None:
    """The script's module-level ``_TARGET_MODE`` constant pins the
    Probe 1.5 main run to ``'online'``. Annotated as the same
    ``Literal["online", "frozen", "environmental"]`` the
    ``RunnerConfig.self_prediction_target_mode`` field declares so
    mypy ``--strict`` sees the value flowing into both constructors as
    the right type (Phase 5 journal flagged this gotcha for
    control-variant scripts).
    """
    module = _load_script_module()
    assert module._TARGET_MODE == "online", (
        "_TARGET_MODE must be 'online' for the Probe 1.5 main run; the "
        "frozen-target and environmental-auxiliary controls are "
        "separate scripts (plan §2.8)"
    )


def test_dry_run_returns_zero_and_prints_target_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calling ``main(['--dry-run'])`` returns 0, prints a summary that
    surfaces ``self_prediction_target_mode='online'``, and does *not*
    create the ``runs/`` directory.

    Run from inside ``tmp_path`` so the dry-run summary's
    ``run_dir.resolve()`` lands under the test's tmp dir; this lets
    the test verify the side-effect-free property by checking that
    no ``runs/`` subdirectory was created.
    """
    module = _load_script_module()
    monkeypatch.chdir(tmp_path)

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = module.main(["--dry-run"])

    assert exit_code == 0, (
        f"main(['--dry-run']) must return 0 on success; got {exit_code}"
    )

    output = buffer.getvalue()
    assert "DRY RUN" in output, (
        "dry-run summary must include the 'DRY RUN' marker so the user "
        "can tell the run did not actually execute"
    )
    assert "self_prediction_target_mode" in output, (
        "dry-run summary must surface the self_prediction_target_mode "
        "field so the test (and the human builder) can verify the "
        "Probe 1.5 main path was wired up before launching the real "
        "run"
    )
    assert "online" in output, (
        "dry-run summary must show the target mode as 'online' for "
        "the Probe 1.5 main run"
    )
    # Schema version surfacing — defense in depth that the dry-run
    # sees the v2 schema and not a stale stamp.
    assert "0.2.0" in output, (
        "dry-run summary must surface schema_version=0.2.0"
    )

    # Side-effect-free check: the dry-run path must not create the
    # runs/ directory or any subdirectories under it.
    runs_dir = tmp_path / "runs"
    if runs_dir.exists():
        children = list(runs_dir.iterdir())
        assert not children, (
            f"--dry-run must not create any run subdirectories; "
            f"found: {children}"
        )


def test_dry_run_does_not_invoke_side_effect_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dry-run path must not call ``torch.backends.mps.is_available``,
    must not stand up an :class:`EnvServer`, must not open transport
    sockets, and must not construct a :class:`Runner`. Hooking each at
    the script's module surface and asserting they were not invoked
    pins the structural property the build prompt names ("constructs
    a RunnerConfig with the correct target_mode without actually
    running"). Symmetric with the Phase 5 frozen-target control
    test's guarded-surfaces check.
    """
    module = _load_script_module()
    monkeypatch.chdir(tmp_path)

    invocations: dict[str, int] = {
        "Runner": 0,
        "EnvServer": 0,
        "EnvTransportClient": 0,
        "EnvTransportServer": 0,
        "_detect_mps_or_exit": 0,
    }

    def _record(name: str) -> Any:
        def _fail(*_args: Any, **_kwargs: Any) -> None:
            invocations[name] += 1
            raise AssertionError(
                f"--dry-run must not invoke {name}; the test guards "
                f"the structural side-effect-free property"
            )

        return _fail

    monkeypatch.setattr(module, "Runner", _record("Runner"))
    monkeypatch.setattr(module, "EnvServer", _record("EnvServer"))
    monkeypatch.setattr(
        module, "EnvTransportClient", _record("EnvTransportClient")
    )
    monkeypatch.setattr(
        module, "EnvTransportServer", _record("EnvTransportServer")
    )
    monkeypatch.setattr(
        module, "_detect_mps_or_exit", _record("_detect_mps_or_exit")
    )

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = module.main(["--dry-run"])

    assert exit_code == 0
    assert all(count == 0 for count in invocations.values()), (
        f"--dry-run invoked guarded surfaces: {invocations}"
    )


def test_dry_run_summary_includes_run_id_with_probe1_5_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run-id format is ``probe1_5-YYYYMMDD-HHMMSS``. The dry-run
    summary surfaces the run id; the test pins the prefix so a future
    rename of ``_RUN_ID_PREFIX`` is caught by both the test and (if
    someone calls ``call_mirror.py`` against the run id) the mirror's
    glob-or-explicit-arg path.

    The underscore between ``probe1`` and ``5`` is intentional — the
    Phase 0 journal's writer-migration discipline names the run id
    convention; ``call_mirror.py``'s default glob is ``probe1-*``
    (literal hyphen), which means a Probe 1.5 run id requires the
    explicit ``run_id`` argument to ``call_mirror.py`` rather than
    relying on the latest-run default. The test's job here is to pin
    the prefix; the mirror-call discipline lives in the journal entry.
    """
    module = _load_script_module()
    monkeypatch.chdir(tmp_path)

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = module.main(["--dry-run"])

    assert exit_code == 0
    output = buffer.getvalue()
    assert "probe1_5-" in output, (
        "dry-run summary must surface a run_id with the 'probe1_5-' "
        "prefix; the underscore distinguishes Probe 1.5 from Probe 1's "
        "run-id namespace"
    )


@pytest.fixture(autouse=True)
def _restore_sys_modules() -> Any:
    """Each ``_load_script_module()`` call exec's the script under a
    fresh module name. The fixture removes the module from
    ``sys.modules`` after each test so re-loads pick up file changes
    (relevant only if the script is edited mid-test-session) and so
    state doesn't leak between tests."""
    yield
    sys.modules.pop(_MODULE_NAME, None)
