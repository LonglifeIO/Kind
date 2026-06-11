"""Probe 3.5 Phase 2 — structural guards for the pragmatic preference.

Three belts, each with a positive control so the guard is only trusted if it
can fail:

1. **Dream-path unreachability (import lint).** The offline regime —
   ``kind/training/dream.py``, ``kind/training/dream_seed.py``,
   ``kind/training/state_machine.py``, ``kind/observer/dream_session.py`` —
   must not import ``kind.agents.preference`` (the pragmatic term's only
   home) and must never reference ``imagine_and_compute_loss`` (the
   pragmatic term's only call site). Dreams are not for anything (F5); a
   pragmatic term computable from the dream regime would make them *for*
   energy management.

2. **Dream-path unreachability (behavioral backstop).** With
   ``energy_log_preference`` monkeypatched to raise at *both* its definition
   site and the actor's bound name, a full ``emit_dream_rollout`` (both
   action policies, actor present) completes — the dream path provably never
   computes the term. The same patch makes a preference-carrying
   ``imagine_and_compute_loss`` raise, proving the tripwire is live.

3. **Marker belt.** The Phase-8b content-blindness checker, extended with
   ``energy`` / ``pragmatic`` / ``sensed`` markers, holds on
   ``MetabolicState`` (DP2: the budget stays untouched — this file extends
   the belt *test-side* without modifying ``tests/test_metabolic_reentry.py``,
   per the Phase-2 build instruction) and on ``PolicyView`` (DP4: the field
   set stays exactly ``{h, z, self_prediction_error}``; no energy- or
   pragmatic-named field can appear).
"""

from __future__ import annotations

import ast
import typing
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pytest
import torch

import kind.agents.actor as actor_module
import kind.agents.preference as preference_module
from kind.agents.actor import Actor
from kind.agents.ensemble import LatentDisagreementEnsemble
from kind.agents.preference import EnergyPreferenceConfig
from kind.agents.views import PolicyView
from kind.agents.world_model import WorldModel, WorldModelConfig
from kind.training.dream import (
    DreamRolloutConfig,
    compute_checkpoint_hash,
    emit_dream_rollout,
)
from kind.training.dream_seed import SeedSelectionConfig
from kind.training.protection import MetabolicState
from kind.training.replay import SequenceReplayBuffer, Transition
from tests.test_metabolic_reentry import (
    _FORBIDDEN_TYPES,
    _IO_DERIVED_NAME_MARKERS,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

# The offline regime: every module on the dream/offline path. The preference
# must be unreachable from all of them.
_OFFLINE_REGIME_MODULES = (
    _REPO_ROOT / "kind" / "training" / "dream.py",
    _REPO_ROOT / "kind" / "training" / "dream_seed.py",
    _REPO_ROOT / "kind" / "training" / "state_machine.py",
    _REPO_ROOT / "kind" / "observer" / "dream_session.py",
)

_PREFERENCE_MODULE = "kind.agents.preference"

# Phase-2 marker-belt extension (plan §"Do-not-touch" structural guards):
# energy-, pragmatic-, and sensed-named fields become unrepresentable on the
# guarded surfaces. Superset of the Phase-8b belt by construction.
_EXTENDED_MARKERS = _IO_DERIVED_NAME_MARKERS + ("energy", "pragmatic", "sensed")

_CONTENT_BLIND_SCALARS = {int, float, bool, str}


# ---- belt 1: import lint ---------------------------------------------------


def _imported_modules_from_source(source: str) -> list[str]:
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
    return names


def _references_identifier(source: str, identifier: str) -> bool:
    """True if any attribute access or bare name in ``source`` is exactly
    ``identifier`` (AST-based — comments and docstrings don't count)."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == identifier:
            return True
        if isinstance(node, ast.Name) and node.id == identifier:
            return True
    return False


def test_offline_regime_does_not_import_the_preference() -> None:
    """LOAD-BEARING. No offline-regime module imports
    ``kind.agents.preference`` (directly or via ``from kind.agents import
    preference``). The pragmatic term's home is unreachable from the dream
    path at the module level."""
    for path in _OFFLINE_REGIME_MODULES:
        assert path.exists(), path
        imported = _imported_modules_from_source(path.read_text())
        offending = [
            m
            for m in imported
            if m == _PREFERENCE_MODULE or m.startswith(_PREFERENCE_MODULE + ".")
        ]
        assert offending == [], (
            f"{path.name} imports the preference module {offending}; the "
            f"dream/offline regime must have no route to the pragmatic term "
            f"(F5: dreams are not for anything)."
        )
        # ``from kind.agents import preference`` shows up as kind.agents +
        # an alias named "preference" — catch the alias form too.
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "kind.agents":
                names = [alias.name for alias in node.names]
                assert "preference" not in names, (
                    f"{path.name} imports preference via 'from kind.agents "
                    f"import preference'"
                )


def test_offline_regime_never_references_the_pragmatic_call_site() -> None:
    """LOAD-BEARING. ``imagine_and_compute_loss`` is the pragmatic term's
    only call site; no offline-regime module references it. (dream.py
    imports ``Actor`` for the temperature-modified action policy — that is
    ``Actor.forward`` only, and this lint is what keeps the loss entry point
    out of reach.)"""
    for path in _OFFLINE_REGIME_MODULES:
        assert not _references_identifier(
            path.read_text(), "imagine_and_compute_loss"
        ), (
            f"{path.name} references imagine_and_compute_loss — the "
            f"imagination-training objective (and with it the pragmatic "
            f"term) must not be reachable from the offline regime."
        )
        assert not _references_identifier(
            path.read_text(), "energy_log_preference"
        ), f"{path.name} references energy_log_preference"


def test_import_lint_trips_on_injected_preference_import() -> None:
    """Positive control: the lint genuinely fails on each forbidden form."""
    direct = "from kind.agents.preference import energy_log_preference\n"
    imported = _imported_modules_from_source(direct)
    assert any(m == _PREFERENCE_MODULE for m in imported)

    module_form = "import kind.agents.preference\n"
    imported = _imported_modules_from_source(module_form)
    assert any(m == _PREFERENCE_MODULE for m in imported)

    call_site = "loss = actor.imagine_and_compute_loss(wm, ens, h, z)\n"
    assert _references_identifier(call_site, "imagine_and_compute_loss")


# ---- belt 2: behavioral backstop -------------------------------------------


_H_DIM = 32
_Z_DIM = 8
_NUM_ACTIONS = 5


def _tiny_world_model() -> WorldModel:
    torch.manual_seed(0)
    wm = WorldModel(
        WorldModelConfig(
            obs_channels=1,
            obs_size=32,
            h_dim=_H_DIM,
            z_dim=_Z_DIM,
            embed_dim=32,
            num_actions=_NUM_ACTIONS,
            action_emb_dim=8,
            mlp_hidden=32,
        )
    )
    wm.eval()
    return wm


def _tiny_ensemble() -> LatentDisagreementEnsemble:
    torch.manual_seed(1)
    return LatentDisagreementEnsemble(
        h_dim=_H_DIM,
        z_dim=_Z_DIM,
        action_dim=_NUM_ACTIONS,
        K=2,
        action_emb_dim=8,
        mlp_hidden=32,
    )


def _tiny_actor() -> Actor:
    torch.manual_seed(2)
    return Actor(
        h_dim=_H_DIM, z_dim=_Z_DIM, action_dim=_NUM_ACTIONS, mlp_hidden=32
    )


def _full_buffer() -> SequenceReplayBuffer:
    buf = SequenceReplayBuffer(capacity=2000, sequence_length=16)
    gen = torch.Generator()
    gen.manual_seed(42)
    for i in range(1500):
        buf.insert(
            Transition(
                obs=torch.rand((1, 32, 32), generator=gen),
                action=i % _NUM_ACTIONS,
                next_obs=torch.rand((1, 32, 32), generator=gen),
                env_step=i,
                episode_id=i // 100,
                step_in_episode=i % 100,
            )
        )
    return buf


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _raise_if_called(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError(
        "energy_log_preference was computed from the dream/offline regime"
    )


def test_dream_regime_provably_never_computes_the_pragmatic_term(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LOAD-BEARING. With the preference function replaced by a tripwire at
    both its definition site and the actor's bound name, the full dream
    rollout — uniform_random AND temperature_modified_actor action policies,
    actor present — completes without touching it."""
    monkeypatch.setattr(preference_module, "energy_log_preference", _raise_if_called)
    monkeypatch.setattr(actor_module, "energy_log_preference", _raise_if_called)

    wm = _tiny_world_model()
    ensemble = _tiny_ensemble()
    actor = _tiny_actor()
    buf = _full_buffer()
    ckpt_hash = compute_checkpoint_hash(wm, actor, ensemble)

    for policy in ("uniform_random", "temperature_modified_actor"):
        record = emit_dream_rollout(
            world_model=wm,
            actor=actor,
            ensemble=ensemble,
            replay_buffer=buf,
            seed_selection_config=SeedSelectionConfig(mode="replay"),
            config=DreamRolloutConfig(horizon=10, action_policy=policy),  # type: ignore[arg-type]
            dream_session_id="guard-sess",
            env_step_at_emit=5000,
            run_id="guard-run",
            checkpoint_id=None,
            checkpoint_hash=ckpt_hash,
            rng=_gen(7),
            device=torch.device("cpu"),
        )
        assert len(record.sequence_action) == 10


def test_tripwire_is_live_on_the_waking_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control: the same patch DOES trip when the waking objective
    computes the term — the backstop can fail, so its silence on the dream
    path means something."""
    monkeypatch.setattr(actor_module, "energy_log_preference", _raise_if_called)
    wm = _tiny_world_model()
    ensemble = _tiny_ensemble()
    actor = _tiny_actor()
    h_0 = torch.zeros(2, _H_DIM)
    z_0 = torch.zeros(2, _Z_DIM)
    with pytest.raises(AssertionError, match="dream/offline regime"):
        actor.imagine_and_compute_loss(
            wm,
            ensemble,
            h_0,
            z_0,
            horizon=3,
            energy_preference=EnergyPreferenceConfig(precision=1.0),
        )


# ---- belt 3: marker belt ----------------------------------------------------


def _extended_offenders(dc: type) -> list[str]:
    """The Phase-8b content-blindness checker with the Phase-2 extended
    marker set (energy / pragmatic / sensed)."""
    hints = typing.get_type_hints(dc)
    offenders: list[str] = []
    for name, annotation in hints.items():
        if annotation in _FORBIDDEN_TYPES:
            offenders.append(name)
            continue
        if annotation not in _CONTENT_BLIND_SCALARS:
            offenders.append(name)
            continue
        low = name.lower()
        if any(marker in low for marker in _EXTENDED_MARKERS):
            offenders.append(name)
    return offenders


def test_extended_belt_is_a_superset_of_the_phase8b_belt() -> None:
    """The extension adds markers; it removes none — the original guard's
    coverage is preserved verbatim (test_metabolic_reentry.py unmodified)."""
    assert set(_IO_DERIVED_NAME_MARKERS).issubset(set(_EXTENDED_MARKERS))
    for marker in ("energy", "pragmatic", "sensed"):
        assert marker in _EXTENDED_MARKERS
        assert marker not in _IO_DERIVED_NAME_MARKERS


def test_metabolic_state_clean_under_extended_belt() -> None:
    """LOAD-BEARING (DP2). No energy/pragmatic/sensed-named field exists on
    ``MetabolicState`` — an energy quantity gating dream entry is
    unrepresentable. The budget surface itself is untouched this phase."""
    assert _extended_offenders(MetabolicState) == []


def test_policy_view_clean_under_extended_belt_and_frozen_field_set() -> None:
    """LOAD-BEARING (DP4). PolicyView's field set is exactly
    ``{h, z, self_prediction_error}`` and no field name carries an
    energy/pragmatic/sensed/decode marker — the actor reads no energy
    quantity, directly or by another name.
    ``new_actor_readable_interfaces_added`` this phase is the pragmatic
    term in the *objective*, not any new readable field."""
    names = {f.name for f in fields(PolicyView)}
    assert names == {"h", "z", "self_prediction_error"}
    for name in names:
        low = name.lower()
        for marker in ("energy", "pragmatic", "sensed", "decode"):
            assert marker not in low, (name, marker)


def test_extended_belt_trips_on_injected_energy_fields() -> None:
    """Positive control: the extended checker fails on exactly the fields the
    Phase-8b belt could not catch."""

    @dataclass(frozen=True)
    class _BadWithEnergy:
        window_compute_seconds: float
        energy_level: float

    @dataclass(frozen=True)
    class _BadWithPragmatic:
        pragmatic_share: float

    @dataclass(frozen=True)
    class _BadWithSensed:
        sensed_value: float

    assert _extended_offenders(_BadWithEnergy) == ["energy_level"]
    assert _extended_offenders(_BadWithPragmatic) == ["pragmatic_share"]
    assert _extended_offenders(_BadWithSensed) == ["sensed_value"]
