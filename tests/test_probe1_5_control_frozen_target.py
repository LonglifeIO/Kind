"""Phase 5 sanity test: ``scripts/probe1_5_control_frozen_target.py``
exists, exposes ``main()``, and ``--dry-run`` constructs a
:class:`RunnerConfig` with ``self_prediction_target_mode='frozen'``.

The full training run is the human-builder-driven step (the script
takes ~13 minutes on the canonical Mac per Probe 1's wall budget); the
lean revision (plan §13) explicitly does not require pytest to run the
full control. What pytest does:

1. Confirms the script file is checked in at the documented path.
2. Confirms ``main()`` is exposed and callable.
3. Confirms ``make_runner_config`` is exposed as a public-by-name
   helper and produces a :class:`RunnerConfig` with
   ``self_prediction_target_mode='frozen'`` on both the runner-level
   field and the nested :class:`WorldModelConfig` field (the runner
   then ``dataclasses.replace``-s the world-model config from the
   runner-level field at construction; both being ``'frozen'`` here is
   the test that the helper sets up the right initial conditions).
4. Confirms ``main(["--dry-run"])`` runs to completion, returns 0,
   prints a summary that includes the lesion's
   ``self_prediction_target_mode='frozen'`` line, and does *not*
   create the run directory under ``runs/`` (the dry-run path is
   side-effect-free).
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
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "probe1_5_control_frozen_target.py"


def _load_script_module() -> ModuleType:
    """Load the script as a fresh module under a unique name.

    The module must be registered in ``sys.modules`` *before*
    ``exec_module`` runs because the script defines ``@dataclass``
    classes (e.g. ``_Progress``) whose decorator resolves field type
    annotations at class-definition time via
    ``sys.modules.get(cls.__module__).__dict__``. If the module is not
    in ``sys.modules`` at decoration time, the dataclass machinery
    raises ``AttributeError: 'NoneType' object has no attribute
    '__dict__'``. The autouse ``_restore_sys_modules`` fixture below
    cleans up after each test."""
    spec = importlib.util.spec_from_file_location(
        "kind_probe1_5_control_frozen_target_module", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_script_file_exists() -> None:
    """The lean revision (plan §13) requires this script as the only
    Phase 5 control variant; the path is the one ``run_probe1.py`` is
    parallel to."""
    assert _SCRIPT_PATH.is_file(), (
        f"plan §2.8 + §13 lean revision expects {_SCRIPT_PATH} to exist "
        f"as a runnable script"
    )


def test_script_exposes_main_callable() -> None:
    """The script's entry point is ``main()`` — the ``__main__`` block
    routes its return value through ``sys.exit`` (same convention as
    ``scripts/run_probe1.py`` and ``scripts/smoke_mps.py``)."""
    module = _load_script_module()
    assert hasattr(module, "main"), (
        "scripts/probe1_5_control_frozen_target.py must define main()"
    )
    assert callable(module.main)


def test_script_exposes_make_runner_config_helper() -> None:
    """The Phase 5 build prompt asks the test to verify ``--dry-run``
    constructs a ``RunnerConfig`` with the correct target_mode without
    actually running. The script exposes ``make_runner_config(...)``
    (no leading underscore) so the test can call it directly without
    standing up the env server."""
    module = _load_script_module()
    assert hasattr(module, "make_runner_config"), (
        "scripts/probe1_5_control_frozen_target.py must expose "
        "make_runner_config(...) as a public-by-name helper for the "
        "Phase 5 dry-run test"
    )
    assert callable(module.make_runner_config)


def test_make_runner_config_sets_target_mode_to_frozen(
    tmp_path: Path,
) -> None:
    """``make_runner_config`` produces a ``RunnerConfig`` whose
    ``self_prediction_target_mode`` is ``'frozen'`` on both the
    runner-level field and the nested ``WorldModelConfig``.

    The runner's ``__init__`` does ``dataclasses.replace`` on the
    world-model config using the runner-level field as the source of
    truth; setting both to ``'frozen'`` here verifies the helper does
    not silently drop the lesion on either surface (which would be a
    bug the runner's silent-fallback would mask: if only the
    runner-level field is set, the substrate still ends up ``'frozen'``
    *via* the runner's replace, but the dry-run summary would then
    not surface the lesion until runner construction)."""
    module = _load_script_module()
    config = module.make_runner_config(
        run_id="probe1_5_control_frozen_target-test",
        telemetry_dir=tmp_path / "telemetry",
        checkpoints_dir=tmp_path / "checkpoints",
    )

    assert config.self_prediction_target_mode == "frozen", (
        "RunnerConfig.self_prediction_target_mode must be 'frozen' for "
        "the frozen-target control run (plan §8.1)"
    )
    assert (
        config.world_model_config.self_prediction_target_mode == "frozen"
    ), (
        "WorldModelConfig.self_prediction_target_mode must be 'frozen' "
        "for the frozen-target control run; the dry-run summary reads "
        "this field directly"
    )


def test_make_runner_config_preserves_probe_1_seed_and_total_steps(
    tmp_path: Path,
) -> None:
    """Plan §8.4 requires the control run to use the same seed and
    total env steps as the Probe 1.5 main run (and Probe 1's run, by
    transitivity). The seed is consumed by the env-server, not the
    runner, so the test asserts the script's module-level constants
    rather than fields on the RunnerConfig."""
    module = _load_script_module()
    assert getattr(module, "_SEED") == 42, (
        "plan §8.4: control run must use seed=42 (matches Probe 1 / "
        "Probe 1.5 main)"
    )
    assert getattr(module, "_TOTAL_ENV_STEPS") == 5000, (
        "plan §8.4: control run must use total_env_steps=5000 (matches "
        "Probe 1 / Probe 1.5 main)"
    )


def test_dry_run_returns_zero_and_prints_target_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calling ``main(['--dry-run'])`` returns 0, prints a summary that
    includes ``self_prediction_target_mode: frozen``, and does *not*
    create the run directory.

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
        "lesion was wired up before launching the real run"
    )
    assert "frozen" in output, (
        "dry-run summary must show the target mode as 'frozen'"
    )
    assert "frozen_target" in output, (
        "dry-run summary must surface the lesion_kind so the operator "
        "sees the mirror_marker payload's name"
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


def test_dry_run_does_not_import_torch_mps_or_open_sockets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dry-run path must not call ``torch.backends.mps.is_available``,
    must not stand up an :class:`EnvServer`, must not open transport
    sockets, and must not construct a :class:`Runner`. Hooking each at
    the script's module surface and asserting they were not invoked
    pins this to the structural property the build prompt names
    ("constructs a RunnerConfig with the correct target_mode without
    actually running")."""
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


def test_dry_run_summary_includes_lesion_rationale_constants() -> None:
    """The script's ``_LESION_KIND`` and ``_LESION_RATIONALE`` constants
    are what the mirror_marker payload carries. The test pins
    ``_LESION_KIND == 'frozen_target'`` (the Phase 5 build prompt
    names this specific value) and verifies the rationale is non-empty
    and references the synthesis section the lesion calibrates against."""
    module = _load_script_module()
    assert module._LESION_KIND == "frozen_target", (
        "Phase 5 build prompt: payload['lesion_kind'] must be "
        "'frozen_target'"
    )
    rationale = module._LESION_RATIONALE
    assert isinstance(rationale, str) and rationale.strip(), (
        "_LESION_RATIONALE must be a non-empty string explaining what "
        "the frozen-target lesion is and what it tests"
    )
    # Lightweight content check — the rationale should reference the
    # failure mode the lesion calibrates against (synthesis §1.7(a)).
    assert "1.7" in rationale or "failure mode (a)" in rationale, (
        "_LESION_RATIONALE should reference synthesis §1.7(a) or "
        "'failure mode (a)' so a future reader sees the lesion's "
        "anchor in the project documents without having to dig"
    )


@pytest.fixture(autouse=True)
def _restore_sys_modules() -> Any:
    """Each ``_load_script_module()`` call exec's the script under a
    fresh module name. The fixture removes the module from
    ``sys.modules`` after each test so re-loads pick up file changes
    (relevant only if the script is edited mid-test-session)."""
    yield
    sys.modules.pop("kind_probe1_5_control_frozen_target_module", None)
