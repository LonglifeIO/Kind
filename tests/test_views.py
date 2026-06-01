"""Phase 3c gate test for ``kind/agents/views.py``.

The ``split`` function projects a ``WorldModelStep`` plus an intrinsic
signal into a ``(PolicyView, TelemetryView)`` tuple. This test verifies
field membership, per-field tensor identity (no unnecessary copies),
the frozen-ness of both views, and the return-order contract. The
dependency-lint tests verify plan §7 Level 1 — that ``kind/agents/actor.py``
(Phase 3b's eventual home, currently a placeholder) imports
``PolicyView`` and never ``TelemetryView``, scoped via AST inspection.

CPU only at this phase. The MPS smoke is Phase 8.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest
import torch

from kind.agents.views import PolicyView, TelemetryView, split
from kind.agents.world_model import WorldModelStep


# ---- shared fixtures ------------------------------------------------------


def make_world_model_step(
    *,
    batch: int = 2,
    h_dim: int = 8,
    z_dim: int = 4,
    embed_dim: int = 16,
    obs_size: int = 32,
) -> WorldModelStep:
    """Construct a ``WorldModelStep`` whose tensors carry recognisable
    shapes and values.

    Each field gets a freshly-allocated tensor so the identity tests can
    distinguish "the view holds the same object as the step" from "the
    view holds a tensor that happens to be equal".
    """
    return WorldModelStep(
        h=torch.randn(batch, h_dim),
        z=torch.randn(batch, z_dim),
        q_params=(torch.randn(batch, z_dim), torch.randn(batch, z_dim)),
        p_params=(torch.randn(batch, z_dim), torch.randn(batch, z_dim)),
        kl_per_dim=torch.randn(batch, z_dim).abs(),
        recon=torch.randn(batch, 1, obs_size, obs_size),
        embed=torch.randn(batch, embed_dim),
        self_prediction=torch.randn(batch, h_dim),
    )


# ---- gate test (plan §4 — Phase 3c, view split half) ----------------------


def _split_with_defaults(
    step: WorldModelStep,
    intrinsic: torch.Tensor,
    *,
    self_prediction_error: torch.Tensor | None = None,
    self_prediction_error_masked: bool = False,
) -> tuple[PolicyView, TelemetryView]:
    """Test helper: call ``split`` with a default zero scalar so existing
    tests that only care about the Probe 1 fields don't have to thread
    the Probe 1.5 v2 args through every call. Tests that exercise the
    new fields pass them explicitly."""
    if self_prediction_error is None:
        self_prediction_error = torch.zeros(())
    return split(
        step,
        intrinsic,
        self_prediction_error=self_prediction_error,
        self_prediction_error_masked=self_prediction_error_masked,
    )


def test_gate_split_produces_correct_views() -> None:
    """``split(step, intrinsic, ...)`` returns the two views with the
    right field memberships and value-equal contents.

    Probe 1.5 v2 extension: PolicyView's field set is exactly
    ``{h, z, self_prediction_error}``; TelemetryView's field set
    extends to include the full ``self_prediction`` vector and the
    ``self_prediction_error_masked`` flag.
    """
    step = make_world_model_step()
    intrinsic = torch.tensor(0.5)
    sp_error = torch.tensor(0.42)

    policy, telemetry = split(
        step,
        intrinsic,
        self_prediction_error=sp_error,
        self_prediction_error_masked=False,
    )

    # PolicyView: exactly three fields — h, z, self_prediction_error —
    # with values from the inputs.
    policy_field_names = {f.name for f in dataclasses.fields(policy)}
    assert policy_field_names == {"h", "z", "self_prediction_error"}
    assert torch.equal(policy.h, step.h)
    assert torch.equal(policy.z, step.z)
    assert torch.equal(policy.self_prediction_error, sp_error)

    # TelemetryView: exactly the ten fields the v2 synthesis names.
    telemetry_field_names = {f.name for f in dataclasses.fields(telemetry)}
    assert telemetry_field_names == {
        "h",
        "z",
        "q_params",
        "p_params",
        "kl_per_dim",
        "recon_loss",
        "embed",
        "intrinsic_signal",
        "self_prediction",
        "self_prediction_error_masked",
    }

    # Values: each TelemetryView field carries the corresponding input value.
    assert torch.equal(telemetry.h, step.h)
    assert torch.equal(telemetry.z, step.z)
    assert torch.equal(telemetry.q_params[0], step.q_params[0])
    assert torch.equal(telemetry.q_params[1], step.q_params[1])
    assert torch.equal(telemetry.p_params[0], step.p_params[0])
    assert torch.equal(telemetry.p_params[1], step.p_params[1])
    assert torch.equal(telemetry.kl_per_dim, step.kl_per_dim)
    # recon_loss is populated from step.recon — see the journal entry for
    # Phase 3c on the naming-dissonance decision.
    assert torch.equal(telemetry.recon_loss, step.recon)
    assert torch.equal(telemetry.embed, step.embed)
    assert torch.equal(telemetry.intrinsic_signal, intrinsic)
    assert torch.equal(telemetry.self_prediction, step.self_prediction)
    assert telemetry.self_prediction_error_masked is False


# ---- return-order contract -----------------------------------------------


def test_split_returns_policy_view_first_telemetry_view_second() -> None:
    """Plan §7 contract: downstream unpacks as ``policy, telemetry``."""
    step = make_world_model_step()
    result = _split_with_defaults(step, torch.tensor(0.0))
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], PolicyView)
    assert isinstance(result[1], TelemetryView)


# ---- frozen-ness ---------------------------------------------------------


def test_policy_view_is_frozen_against_field_reassignment() -> None:
    """Assignment to a frozen dataclass field raises at runtime; the
    setattr-via-builtin path bypasses static type-checking so the test can
    exercise the runtime check without a type-ignore."""
    policy = PolicyView(
        h=torch.zeros(1, 8),
        z=torch.zeros(1, 4),
        self_prediction_error=torch.zeros(()),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(policy, "h", torch.zeros(1, 8))
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(policy, "z", torch.zeros(1, 4))
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(policy, "self_prediction_error", torch.tensor(99.0))


def test_telemetry_view_is_frozen_against_field_reassignment() -> None:
    """Same frozen-ness check for ``TelemetryView`` — including the
    intrinsic signal field, which the actor must never be able to
    swap out from under the mirror's reading, and the Probe 1.5 v2
    self-prediction fields, which encode the asymmetry between Io's
    reading (the scalar) and the mirror's reading (the full vector
    plus the masked flag)."""
    step = make_world_model_step()
    _, telemetry = _split_with_defaults(step, torch.tensor(0.0))
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(telemetry, "h", torch.zeros(1, 8))
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(telemetry, "intrinsic_signal", torch.tensor(99.0))
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(telemetry, "self_prediction", torch.zeros(1, 8))
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(telemetry, "self_prediction_error_masked", True)


# ---- field exclusivity (plan §7 Level 3, runtime-shape) ------------------


def test_policy_view_does_not_carry_any_telemetry_only_fields() -> None:
    """The synthesis §Q5 self-opacity boundary in dataclass-field form:
    ``PolicyView`` must not expose any of the substrate's exposed-to-the-
    mirror fields, including the intrinsic signal, the full
    self-prediction vector, or the masked-step flag.

    Probe 1.5 v2: PolicyView does carry ``self_prediction_error`` (the
    single Watts-heuristic exception articulated in synthesis §1.3 (v2)
    / §2(b) (v2)); this test pins the *negative* — none of the
    forbidden mirror-side fields appear on PolicyView. The positive
    field-set assertion lives in
    ``test_policy_view_field_set_is_exactly_h_z_self_prediction_error``
    below.
    """
    field_names = {f.name for f in dataclasses.fields(PolicyView)}
    forbidden = {
        "q_params",
        "p_params",
        "kl_per_dim",
        "kl_aggregate",
        "recon",
        "recon_loss",
        "embed",
        "intrinsic_signal",
        "self_prediction",
        "self_prediction_error_masked",
    }
    leaked = field_names & forbidden
    assert leaked == set(), (
        f"PolicyView leaks telemetry-only fields {leaked} — synthesis §Q5 "
        f"self-opacity boundary, plan §7 Level 3 enforcement"
    )


def test_telemetry_view_carries_intrinsic_signal_on_purpose() -> None:
    """The dual: the intrinsic signal lives on TelemetryView; the mirror
    needs it to track exploration drive against builder-perturbation
    rate (Probe 4 distinguishability)."""
    field_names = {f.name for f in dataclasses.fields(TelemetryView)}
    assert "intrinsic_signal" in field_names


def test_telemetry_view_carries_full_substrate_surface() -> None:
    """All Probe 1 substrate fields plus the intrinsic signal plus the
    Probe 1.5 v2 self-prediction surface (full vector + masked flag) are
    present; nothing the mirror needs is omitted."""
    field_names = {f.name for f in dataclasses.fields(TelemetryView)}
    expected = {
        "h",
        "z",
        "q_params",
        "p_params",
        "kl_per_dim",
        "recon_loss",
        "embed",
        "intrinsic_signal",
        "self_prediction",
        "self_prediction_error_masked",
    }
    assert field_names == expected


# ---- tensor identity (no unnecessary copies) ----------------------------


def test_split_does_not_copy_tensors() -> None:
    """Memory and gradients both benefit from sharing references. The
    views are projections, not duplications."""
    step = make_world_model_step()
    intrinsic = torch.tensor(0.0)
    sp_error = torch.tensor(0.3)

    policy, telemetry = split(
        step,
        intrinsic,
        self_prediction_error=sp_error,
        self_prediction_error_masked=False,
    )

    # PolicyView aliases h, z, and the scalar.
    assert policy.h is step.h
    assert policy.z is step.z
    assert policy.self_prediction_error is sp_error

    # TelemetryView aliases everything from the step plus the input
    # intrinsic and the input self-prediction vector.
    assert telemetry.h is step.h
    assert telemetry.z is step.z
    assert telemetry.q_params is step.q_params
    assert telemetry.p_params is step.p_params
    assert telemetry.kl_per_dim is step.kl_per_dim
    assert telemetry.recon_loss is step.recon
    assert telemetry.embed is step.embed
    assert telemetry.intrinsic_signal is intrinsic
    assert telemetry.self_prediction is step.self_prediction


def test_split_preserves_gradient_chain() -> None:
    """A tensor with ``requires_grad=True`` going into the step retains
    its grad attribute on the way out via the views — gradient flow from
    the world model into the actor (PolicyView) and from the world model
    into the loss (TelemetryView) must not be silently severed by ``split``.
    """
    h = torch.randn(2, 8, requires_grad=True)
    z = torch.randn(2, 4, requires_grad=True)
    sp = torch.randn(2, 8, requires_grad=True)
    step = WorldModelStep(
        h=h,
        z=z,
        q_params=(torch.randn(2, 4), torch.randn(2, 4)),
        p_params=(torch.randn(2, 4), torch.randn(2, 4)),
        kl_per_dim=torch.randn(2, 4).abs(),
        recon=torch.randn(2, 1, 32, 32),
        embed=torch.randn(2, 16),
        self_prediction=sp,
    )
    sp_error = torch.tensor(0.5, requires_grad=True)
    policy, telemetry = split(
        step,
        torch.tensor(0.0),
        self_prediction_error=sp_error,
        self_prediction_error_masked=False,
    )
    assert policy.h.requires_grad
    assert policy.z.requires_grad
    assert policy.self_prediction_error.requires_grad
    assert telemetry.h.requires_grad
    assert telemetry.z.requires_grad
    assert telemetry.self_prediction.requires_grad


# ---- dependency lint (plan §7 Level 1) -----------------------------------


REPO_ROOT = Path(__file__).resolve().parent.parent
ACTOR_PATH = REPO_ROOT / "kind" / "agents" / "actor.py"


def _imported_names_in(path: Path) -> list[str]:
    """Collect the source-side names of every ``Import`` / ``ImportFrom``
    in the file. AST-based, so robust against the name appearing in
    comments, docstrings, or string literals.

    For aliased imports (``from m import X as Y``), the source name ``X``
    is what's collected — the alias is irrelevant; we want to catch
    aliasing as an attempted bypass, not let it through.
    """
    source = path.read_text()
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
    return names


def test_actor_module_does_not_import_telemetry_view() -> None:
    """Plan §7 Level 1 self-opacity boundary, AST form.

    The actor must not import ``TelemetryView`` in any form. Catches
    direct imports, aliased imports, and module-level imports of
    ``kind.agents.views`` whose attribute access could expose it (the
    third case is enforced by extending the assertion to reject the
    full module name as well — see comment below).
    """
    assert ACTOR_PATH.exists(), f"missing actor module at {ACTOR_PATH}"

    imported = _imported_names_in(ACTOR_PATH)
    assert "TelemetryView" not in imported, (
        "kind/agents/actor.py must not import TelemetryView — synthesis "
        "§Q5 self-opacity boundary, plan §7 Level 1 enforcement. "
        f"Found imports: {imported}"
    )

    # Also reject ``import kind.agents.views`` (whole-module import that
    # would let the actor reach TelemetryView via attribute access). The
    # acceptable form is ``from kind.agents.views import PolicyView``.
    assert "kind.agents.views" not in imported, (
        "kind/agents/actor.py must not import the whole views module — "
        "it would expose TelemetryView via attribute access. Use "
        "'from kind.agents.views import PolicyView' instead."
    )


def test_actor_module_imports_policy_view() -> None:
    """The dual to the negative lint: the actor module must import
    ``PolicyView`` so the lint's positive case is grounded in something
    that exists. Phase 3b's actor will then use it in earnest."""
    assert ACTOR_PATH.exists()
    imported = _imported_names_in(ACTOR_PATH)
    assert "PolicyView" in imported, (
        "kind/agents/actor.py must import PolicyView — plan §7 Level 1 "
        f"enforcement requires the symbol to be present. Found: {imported}"
    )


# ---- module-discipline sanity --------------------------------------------


def test_views_module_exposes_only_the_three_documented_symbols() -> None:
    """``__all__`` is the documented surface; nothing else should be
    re-exported by ``import *``. Catches accidental additions to the
    module surface in future edits."""
    import kind.agents.views as views_module

    assert hasattr(views_module, "__all__")
    assert set(views_module.__all__) == {"PolicyView", "TelemetryView", "split"}


def test_actor_module_exposes_actor_and_action_output() -> None:
    """Phase 3b populated the previously-empty ``__all__``. The lint
    tests above (``test_actor_module_does_not_import_telemetry_view``
    and ``test_actor_module_imports_policy_view``) now run against the
    real actor module and continue to pass: PolicyView is imported (used
    in ``Actor.forward``'s signature); TelemetryView is not imported
    anywhere. See ``tests/test_actor.py`` for the actor's behavioural
    tests."""
    import kind.agents.actor as actor_module

    assert hasattr(actor_module, "__all__")
    assert set(actor_module.__all__) == {"Actor", "ActionOutput"}


# ===========================================================================
# Probe 1.5 Gate Test #4 — opacity boundary preserved with the revised
# PolicyView field set (plan §4 / §7.4)
# ===========================================================================


def test_policy_view_field_set_is_exactly_h_z_self_prediction_error() -> None:
    """``dataclasses.fields(PolicyView)`` produces exactly
    ``{"h", "z", "self_prediction_error"}`` — the synthesis §1.3 (v2)
    interface-level opacity commitment in dataclass-field form.

    Stronger than the existing
    ``test_policy_view_does_not_carry_any_telemetry_only_fields`` (which
    lists forbidden fields by name). This test is the structural
    stability check for the v2 PolicyView extension: any future field
    addition to PolicyView fails this test, and any rename of the v2
    extension also fails. The synthesis §2(b) (v2) four-part discipline
    applies to any further extension — i.e. a future probe adding a
    new actor-readable field must update this test along with the
    journal entry that addresses (i) which affordance, (ii) minimum
    form, (iii) alternatives considered, (iv) failure-mode controls.
    """
    field_names = {f.name for f in dataclasses.fields(PolicyView)}
    assert field_names == {"h", "z", "self_prediction_error"}, (
        f"PolicyView field set drifted: {field_names}. The synthesis §1.3 "
        f"(v2) interface-level opacity boundary requires exactly "
        f"{{h, z, self_prediction_error}}; any further extension goes on "
        f"TelemetryView (the affordance-side surface), and any extension "
        f"to PolicyView requires the §2(b) (v2) four-part discipline at "
        f"design time."
    )


def test_actor_forward_rejects_telemetry_view_at_runtime() -> None:
    """``Actor.forward(telemetry_view)`` raises ``AttributeError`` at
    runtime — TelemetryView lacks the ``self_prediction_error`` field
    that PolicyView requires (the synthesis §1.3 (v2) field set),
    so the actor's forward fails cleanly when a TelemetryView is
    smuggled past the type-checker rather than silently falling back
    to a degraded behavior.

    Plan §4 gate test #4 (b): "Actor.forward(telemetry_view) raises a
    runtime error (constructed by passing a TelemetryView into a
    function expecting PolicyView)." The cleanest realisation given
    the v2 design (TelemetryView gains ``self_prediction`` (vector)
    and ``self_prediction_error_masked`` (bool) but NOT the scalar
    ``self_prediction_error``): the runtime rejection is the
    AttributeError the actor's first attribute access produces.

    The asymmetry between Io's reading (the scalar) and the mirror's
    reading (full vector + masked flag + everything else) is
    preserved at the runtime level *via the field set difference*
    rather than via a fallback-to-h-z-only behavior. The actor cannot
    silently consume a TelemetryView even if smuggled past the
    type-checker — the forward attempts to read a field TelemetryView
    does not carry. Type-level rejection
    (``test_actor_forward_telemetryview_argument_fails_mypy_strict``)
    is the primary defense; this test pins the runtime backstop.
    """
    import torch
    from kind.agents.actor import Actor

    step = make_world_model_step()
    intrinsic = torch.tensor(0.0)
    sp_error = torch.tensor(0.5)

    policy, telemetry = split(
        step,
        intrinsic,
        self_prediction_error=sp_error,
        self_prediction_error_masked=False,
    )

    actor = Actor(
        h_dim=step.h.shape[-1],
        z_dim=step.z.shape[-1],
        action_dim=5,
    )

    # PolicyView path works.
    canonical = actor.forward(policy)
    assert canonical.logits.shape[0] == step.h.shape[0]

    # TelemetryView path fails at runtime — TelemetryView does not have
    # ``self_prediction_error``. Smuggling past the type-checker yields
    # an AttributeError when the actor's forward tries to read the
    # field. This IS the runtime rejection plan §4 gate test #4 (b)
    # asks for.
    with pytest.raises(AttributeError, match="self_prediction_error"):
        actor.forward(telemetry)  # type: ignore[arg-type]


def test_actor_forward_telemetryview_argument_fails_mypy_strict() -> None:
    """``mypy --strict`` against the fixture ``_actor_telemetryview_attempt.py``
    reports the expected argument-type error.

    Plan §7 Level 2 enforcement extended to the Probe 1.5 v2 fields:
    the type-level rejection of ``TelemetryView`` as an argument to
    ``Actor.forward`` survives the addition of the new fields on both
    views.

    The fixture file (``tests/_actor_telemetryview_attempt.py``) is
    deliberately type-incorrect; this test invokes mypy on it via
    subprocess and asserts the expected error appears.
    """
    import shutil
    import subprocess

    fixture_path = Path(__file__).parent / "_actor_telemetryview_attempt.py"
    assert fixture_path.exists(), f"missing fixture at {fixture_path}"

    if shutil.which("mypy") is None:
        pytest.skip("mypy not on PATH; skipping type-check enforcement test")

    result = subprocess.run(
        ["mypy", "--strict", "--no-incremental", str(fixture_path)],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
    )

    # mypy returns non-zero on any error.
    assert result.returncode != 0, (
        f"mypy --strict accepted the type-incorrect fixture "
        f"(returncode={result.returncode}); the type-level opacity "
        f"boundary at plan §7 Level 2 has been weakened. "
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    combined = (result.stdout + result.stderr).lower()
    assert "incompatible type" in combined or "telemetryview" in combined, (
        f"mypy --strict failed but the error does not mention the "
        f"argument type mismatch. Expected 'incompatible type' or "
        f"'TelemetryView' in mypy's output. "
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_ast_lint_passes_with_extended_telemetry_view() -> None:
    """The existing ``test_actor_module_does_not_import_telemetry_view``
    AST-lint test (above) continues to pass post-Probe-1.5, even though
    ``TelemetryView`` has gained two new fields and ``PolicyView`` has
    gained one new field. This test explicitly re-asserts that
    invariant from the perspective of "the new fields haven't widened
    the actor module's import surface" — defensive against a future
    refactor that might import TelemetryView for reading the new fields.
    """
    imported = _imported_names_in(ACTOR_PATH)
    assert "TelemetryView" not in imported
    assert "PolicyView" in imported
    # Sanity: the new field name on PolicyView is not imported as a
    # standalone symbol either (we only need the dataclass; the field
    # is accessed via attribute on a PolicyView instance).
    assert "self_prediction_error" not in imported
    assert "self_prediction_error_masked" not in imported
    assert "self_prediction" not in imported


# ===========================================================================
# Probe 1.5 Phase 2 — split with the new kwargs populates both views
# ===========================================================================


def test_split_kwargs_populate_both_views_correctly() -> None:
    """The extended ``split`` signature accepts ``self_prediction_error``
    and ``self_prediction_error_masked`` as keyword-only arguments and
    populates both views from them: the scalar lands on PolicyView's
    ``self_prediction_error``; the masked flag lands on TelemetryView's
    ``self_prediction_error_masked``; the WorldModelStep's
    ``self_prediction`` vector lands on TelemetryView's
    ``self_prediction``."""
    step = make_world_model_step()
    intrinsic = torch.tensor(0.7)
    sp_error = torch.tensor(0.123)

    policy, telemetry = split(
        step,
        intrinsic,
        self_prediction_error=sp_error,
        self_prediction_error_masked=True,
    )

    # PolicyView: the scalar is the one passed in.
    assert torch.equal(policy.self_prediction_error, sp_error)
    # TelemetryView: the vector is from the step; the flag is from kwargs.
    assert torch.equal(telemetry.self_prediction, step.self_prediction)
    assert telemetry.self_prediction_error_masked is True

    # Re-do with masked=False to confirm the flag passes through.
    policy2, telemetry2 = split(
        step,
        intrinsic,
        self_prediction_error=sp_error,
        self_prediction_error_masked=False,
    )
    assert telemetry2.self_prediction_error_masked is False
    # The scalar reaches both views identically (PolicyView is the
    # primary site; the mirror also reads it from the AgentStep record
    # the runner emits — TelemetryView itself doesn't carry the scalar
    # field, just the vector and masked flag).
    assert torch.equal(policy2.self_prediction_error, sp_error)


def test_split_requires_keyword_only_self_prediction_args() -> None:
    """The Probe 1.5 v2 ``split`` signature uses keyword-only marker
    (``*,``) so the new args cannot be passed positionally. This is the
    plan §2.3 verbatim spec; the marker is what catches a future
    refactor that tries to slip a new argument into the positional
    chain (and would silently shift the meaning of existing args)."""
    step = make_world_model_step()
    intrinsic = torch.tensor(0.0)
    sp_error = torch.tensor(0.0)

    # Positional call must fail.
    with pytest.raises(TypeError):
        split(step, intrinsic, sp_error, False)  # type: ignore[misc]

    # Kwarg call works.
    policy, telemetry = split(
        step,
        intrinsic,
        self_prediction_error=sp_error,
        self_prediction_error_masked=False,
    )
    assert isinstance(policy, PolicyView)
    assert isinstance(telemetry, TelemetryView)


# ===========================================================================
# Probe 3 Phase 5 — the mirror's one-way constraint, extended to the
# dream-reading boundary (plan §2.6 / §4 Phase 5 row).
#
# The existing dependency lint above pins the actor↔TelemetryView boundary
# (Io must not read the mirror's surface). The dual Probe 3 boundary is that
# the mirror's dream-reading layer must not import the training-side modules
# that would let its output flow back to Io. ``kind.mirror.dream_reading`` may
# import read-only observer models and mirror-side modules; it must not import
# ``kind.training.state_machine``, ``kind.training.dream``, or any other
# ``kind.training`` module. Making that dependency edge fail to exist is the
# one-way constraint made structural.
# ===========================================================================


DREAM_READING_PATH = REPO_ROOT / "kind" / "mirror" / "dream_reading.py"


def _imported_module_names_in(path: Path) -> list[str]:
    """Collect the *module* names imported in the file (``ImportFrom.module``
    and ``Import`` alias names), so a ``from kind.training.x import Y`` is
    detected by its module path rather than by the symbol ``Y``."""
    tree = ast.parse(path.read_text())
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module is not None:
                modules.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
    return modules


def test_dream_reading_does_not_import_training_modules() -> None:
    """The mirror's dream-reading layer is one-way: no ``kind.training``
    import. Catches ``kind.training.state_machine`` and
    ``kind.training.dream`` by name and any other training submodule by
    prefix."""
    assert DREAM_READING_PATH.exists(), f"missing module at {DREAM_READING_PATH}"
    modules = _imported_module_names_in(DREAM_READING_PATH)
    offending = [
        m for m in modules if m == "kind.training" or m.startswith("kind.training.")
    ]
    assert offending == [], (
        "kind/mirror/dream_reading.py imports training-side module(s) "
        f"{offending} — the mirror's one-way constraint (plan §2.6) forbids "
        f"any kind.training import. Found modules: {modules}"
    )
    assert "kind.training.state_machine" not in modules
    assert "kind.training.dream" not in modules


def test_dream_reading_import_lint_trips_on_forbidden_import() -> None:
    """The lint genuinely fails when a training import is present — the
    structural guard is only meaningful if it can catch a regression. We
    assert against a positive-control source rather than mutating the real
    module on disk."""
    bad_source = (
        "from kind.observer.schemas import DreamRollout\n"
        "from kind.training.dream import emit_dream\n"
    )
    tree = ast.parse(bad_source)
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    offending = [
        m for m in modules if m == "kind.training" or m.startswith("kind.training.")
    ]
    assert offending == ["kind.training.dream"]
