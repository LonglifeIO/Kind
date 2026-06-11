"""Probe 3.5 Phase 3 — the §7 dream passive-decode monitor.

The pre-registered resolved sub-decision #1 (frozen pre-reg §7), built at
Phase 3: dream rollouts record ``decode_energy`` alongside
``sequence_decoded_obs`` — **observer-side telemetry only**. The dream regime
stays preference-free and loss-free with respect to this monitor:

- **Byte-identity**: with the monitor on vs off at the same rng, every field
  of the emitted record except ``sequence_decoded_energy`` / the version stamp
  is identical — the monitor reads the belief and changes nothing (there is no
  dream loss; the rollout is the dream's entire output, so record-identity *is*
  the loss-free proof at the behavioral level).
- **Tripwire** (the house style from ``tests/test_pragmatic_guards.py``): with
  ``energy_log_preference`` replaced by a raiser at both its definition site
  and the actor's bound name, a monitor-ON rollout completes — watching the
  belief never computes the preference. Io's dream gains nothing to optimize;
  the mirror gains a window.
- **No-grad / opt-in / legacy**: emission is under ``no_grad`` (no parameter
  gradient anywhere); the flag defaults off and the off-path record is
  stamped at the Probe-3 version with the field None (legacy byte-identical);
  the version-gated validator requires the field at the Phase-3 dream version.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

import kind.agents.actor as actor_module
import kind.agents.preference as preference_module
from kind.agents.ensemble import LatentDisagreementEnsemble
from kind.agents.world_model import WorldModel, WorldModelConfig
from kind.observer.schemas import (
    PROBE_3_5_PHASE3_DREAM_SCHEMA_VERSION,
    PROBE_3_TELEMETRY_SCHEMA_VERSION,
    DreamRollout,
    export_json_schema_v0_7_0,
)
from kind.training.dream import (
    DreamRolloutConfig,
    compute_checkpoint_hash,
    emit_dream_rollout,
)
from kind.training.dream_seed import SeedSelectionConfig
from kind.training.replay import SequenceReplayBuffer, Transition

_H_DIM = 32
_Z_DIM = 8
_NUM_ACTIONS = 5
_HORIZON = 12


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


def _emit(
    wm: WorldModel,
    ensemble: LatentDisagreementEnsemble,
    buf: SequenceReplayBuffer,
    *,
    record_decoded_energy: bool,
    rng_seed: int = 7,
) -> DreamRollout:
    return emit_dream_rollout(
        world_model=wm,
        actor=None,
        ensemble=ensemble,
        replay_buffer=buf,
        seed_selection_config=SeedSelectionConfig(mode="replay"),
        config=DreamRolloutConfig(
            horizon=_HORIZON, record_decoded_energy=record_decoded_energy
        ),
        dream_session_id="monitor-sess",
        env_step_at_emit=5000,
        run_id="monitor-run",
        checkpoint_id=None,
        checkpoint_hash=compute_checkpoint_hash(wm, None, ensemble),
        rng=_gen(rng_seed),
        device=torch.device("cpu"),
    )


def test_default_off_is_legacy_byte_identical() -> None:
    """Off (the default): Probe-3 version stamp, field None — the emission is
    the pre-monitor record exactly."""
    wm, ens, buf = _tiny_world_model(), _tiny_ensemble(), _full_buffer()
    rec = _emit(wm, ens, buf, record_decoded_energy=False)
    assert rec.schema_version == PROBE_3_TELEMETRY_SCHEMA_VERSION
    assert rec.sequence_decoded_energy is None


def test_monitor_on_changes_nothing_but_the_monitor_field() -> None:
    """LOAD-BEARING (the loss-free proof). Same models, same rng: the ON
    record equals the OFF record on every field except the monitor field and
    the version stamp that announces it — the passive decode consumes no RNG
    and perturbs nothing. The dream has no loss; the rollout record is the
    dream's entire output, so record-identity is the 'dream loss byte-identical
    with the monitor on vs off' guarantee at the only surface that exists."""
    wm, ens, buf = _tiny_world_model(), _tiny_ensemble(), _full_buffer()
    off = _emit(wm, ens, buf, record_decoded_energy=False)
    on = _emit(wm, ens, buf, record_decoded_energy=True)

    off_dict: dict[str, Any] = off.model_dump()
    on_dict: dict[str, Any] = on.model_dump()
    assert off_dict.pop("sequence_decoded_energy") is None
    monitor_seq = on_dict.pop("sequence_decoded_energy")
    assert off_dict.pop("schema_version") == PROBE_3_TELEMETRY_SCHEMA_VERSION
    assert on_dict.pop("schema_version") == PROBE_3_5_PHASE3_DREAM_SCHEMA_VERSION
    assert on_dict == off_dict

    assert isinstance(monitor_seq, list)
    assert len(monitor_seq) == _HORIZON
    assert all(isinstance(v, float) for v in monitor_seq)


def test_monitor_never_computes_the_preference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tripwire (house style): the preference function raises if touched, at
    both its definition site and the actor's bound name; a monitor-ON rollout
    completes — the monitor watches the belief, never the preference."""

    def _raise(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("preference computed from the dream regime")

    monkeypatch.setattr(preference_module, "energy_log_preference", _raise)
    monkeypatch.setattr(actor_module, "energy_log_preference", _raise)
    wm, ens, buf = _tiny_world_model(), _tiny_ensemble(), _full_buffer()
    rec = _emit(wm, ens, buf, record_decoded_energy=True)
    assert rec.sequence_decoded_energy is not None
    assert len(rec.sequence_decoded_energy) == _HORIZON


def test_monitor_emission_leaves_no_gradient_anywhere() -> None:
    """The decode runs under the rollout's no_grad: no world-model parameter
    accumulates a gradient from a monitor-ON emission."""
    wm, ens, buf = _tiny_world_model(), _tiny_ensemble(), _full_buffer()
    _emit(wm, ens, buf, record_decoded_energy=True)
    assert all(p.grad is None for p in wm.parameters())
    assert all(p.grad is None for p in ens.parameters())


def test_phase3_dream_version_requires_the_monitor_field() -> None:
    """Version-gated validator: a Phase-3 dream record without the monitor
    field is rejected; Probe-3 records with the field absent stay valid."""
    wm, ens, buf = _tiny_world_model(), _tiny_ensemble(), _full_buffer()
    on = _emit(wm, ens, buf, record_decoded_energy=True)
    payload = on.model_dump()
    payload["sequence_decoded_energy"] = None
    with pytest.raises(ValueError, match="sequence_decoded_energy"):
        DreamRollout.model_validate(payload)
    payload["schema_version"] = PROBE_3_TELEMETRY_SCHEMA_VERSION
    legacy = DreamRollout.model_validate(payload)
    assert legacy.sequence_decoded_energy is None


def test_v0_7_0_export_byte_stable_matches_disk_and_carries_the_field() -> None:
    """The Phase-3 export is byte-stable, matches ``schemas/v0.7.0.json``,
    and its DreamRollout model carries the monitor field. If a model
    intentionally changes, regenerate via export_json_schema_v0_7_0 and
    commit."""
    import json
    from pathlib import Path

    first = export_json_schema_v0_7_0()
    assert first == export_json_schema_v0_7_0()
    disk = Path(__file__).resolve().parents[1] / "schemas" / "v0.7.0.json"
    assert disk.read_bytes() == first, (
        "schemas/v0.7.0.json no longer matches the live models — regenerate "
        "via export_json_schema_v0_7_0() and commit the result."
    )
    document = json.loads(first)
    dream_rollout = document["models"]["telemetry"]["DreamRollout"]
    assert "sequence_decoded_energy" in dream_rollout["properties"]
