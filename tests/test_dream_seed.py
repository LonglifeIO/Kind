"""Phase 1 gate tests for ``kind/training/dream_seed.py``.

Three tests per amended §4 Phase 1 row: deterministic re-encoding in replay
mode (the Option-1 reproducibility chain), perturbation magnitude in
perturbed_prior mode, and convex combination in hybrid mode. All run on CPU
against small fixed seeds — no MPS / CUDA required.
"""

from __future__ import annotations

import torch

from kind.agents.world_model import WorldModel, WorldModelConfig
from kind.training.dream_seed import (
    DreamSeed,
    SeedSelectionConfig,
    _perturb_anchor,
    _run_warmup,
    select_seed,
)
from kind.training.replay import SequenceReplayBuffer, Transition


def _make_world_model(seed: int = 0) -> WorldModel:
    torch.manual_seed(seed)
    cfg = WorldModelConfig(
        obs_channels=1,
        obs_size=32,
        h_dim=200,
        z_dim=16,
        embed_dim=256,
        num_actions=5,
    )
    wm = WorldModel(cfg)
    wm.eval()
    return wm


def _populate_buffer(
    buffer: SequenceReplayBuffer,
    n_transitions: int = 1500,
    episode_length: int = 200,
    obs_seed: int = 42,
) -> None:
    """Insert deterministic random transitions with regular episode boundaries.

    ``episode_id`` flips every ``episode_length`` transitions; ``env_step``
    is the insertion index. Action cycles through ``num_actions``. The
    obs tensors are drawn from a seeded ``torch.Generator`` so the test is
    bit-stable across runs.
    """
    obs_gen = torch.Generator()
    obs_gen.manual_seed(obs_seed)
    for i in range(n_transitions):
        obs = torch.rand((1, 32, 32), generator=obs_gen)
        next_obs = torch.rand((1, 32, 32), generator=obs_gen)
        buffer.insert(
            Transition(
                obs=obs,
                action=i % 5,
                next_obs=next_obs,
                env_step=i,
                episode_id=i // episode_length,
                step_in_episode=i % episode_length,
            )
        )


def test_replay_mode_deterministic_reencoding() -> None:
    wm = _make_world_model(seed=0)
    buf = SequenceReplayBuffer(capacity=2000, sequence_length=32)
    _populate_buffer(buf, n_transitions=1500, episode_length=200)
    cfg = SeedSelectionConfig(
        mode="replay",
        replay_min_segment_age_steps=1000,
        replay_warmup_length=8,
    )

    # (a) Two calls from the same incoming RNG state produce byte-identical
    #     h_init / z_init / segment_id / rng_seed.
    g1 = torch.Generator()
    g1.manual_seed(12345)
    seed1 = select_seed(buf, wm, cfg, g1)

    g2 = torch.Generator()
    g2.manual_seed(12345)
    seed2 = select_seed(buf, wm, cfg, g2)

    assert isinstance(seed1, DreamSeed)
    assert seed1.mode == "replay"
    assert seed1.replay_step_offset == 0
    assert seed1.replay_segment_id is not None
    assert seed1.perturbation_magnitude is None
    assert seed1.hybrid_mixture_alpha is None
    assert seed1.h_init.shape == (1, wm.config.h_dim)
    assert seed1.z_init.shape == (1, wm.config.z_dim)

    assert seed1.rng_seed == seed2.rng_seed
    assert seed1.replay_segment_id == seed2.replay_segment_id
    assert torch.equal(seed1.h_init, seed2.h_init)
    assert torch.equal(seed1.z_init, seed2.z_init)

    # (b) Independently re-feed the obs/action window identified by
    #     (replay_segment_id, replay_step_offset, replay_warmup_length)
    #     through the same world model from a zero (h, z, a) start under
    #     the same RNG → byte-equal (h_init, z_init).
    obs_seq, action_seq = buf.get_window(
        seed1.replay_segment_id, cfg.replay_warmup_length
    )
    assert obs_seq.shape == (cfg.replay_warmup_length, 1, 32, 32)
    assert action_seq.shape == (cfg.replay_warmup_length,)

    # Mirror select_seed's internal use of seed_gen exactly: seed from
    # rng_seed, advance past the window-pick randint, then run warmup.
    seed_gen = torch.Generator()
    seed_gen.manual_seed(seed1.rng_seed)
    starts = buf.valid_seed_window_starts(
        length=cfg.replay_warmup_length,
        min_age_steps=cfg.replay_min_segment_age_steps,
    )
    pick = int(torch.randint(0, len(starts), (1,), generator=seed_gen).item())
    assert starts[pick] == seed1.replay_segment_id
    h_re, z_re = _run_warmup(wm, obs_seq, action_seq, seed_gen)
    assert torch.equal(h_re, seed1.h_init)
    assert torch.equal(z_re, seed1.z_init)

    # min_segment_age honored: every valid start's env_step is at least
    # replay_min_segment_age_steps older than the buffer's latest insert.
    latest_env_step = 1499
    assert starts, "no valid windows in test fixture — buffer too small?"
    for env_step in starts:
        assert env_step <= latest_env_step - cfg.replay_min_segment_age_steps


def test_perturbed_prior_magnitude() -> None:
    wm = _make_world_model(seed=0)
    # No buffer reads in perturbed_prior mode, but the API still accepts one.
    buf = SequenceReplayBuffer(capacity=100, sequence_length=32)
    cfg = SeedSelectionConfig(mode="perturbed_prior", perturbation_sigma=0.1)
    h_dim = wm.config.h_dim
    z_dim = wm.config.z_dim

    # Known anchor: h-side ones, z-side zeros. The split point is h_dim.
    anchor_h = torch.ones(1, h_dim)
    anchor_z = torch.zeros(1, z_dim)
    anchor = torch.cat([anchor_h, anchor_z], dim=-1)

    g = torch.Generator()
    g.manual_seed(7)
    seed = select_seed(buf, wm, cfg, g, perturbed_prior_anchor=anchor)

    assert seed.mode == "perturbed_prior"
    assert seed.replay_segment_id is None
    assert seed.replay_step_offset is None
    assert seed.hybrid_mixture_alpha is None
    assert seed.perturbation_magnitude is not None
    assert seed.h_init.shape == (1, h_dim)
    assert seed.z_init.shape == (1, z_dim)

    # ||h_init - anchor_h|| ≈ sigma * sqrt(h_dim). For h_dim=200 and
    # sigma=0.1 the expected magnitude is ≈ 1.414; ±20% tolerance covers
    # finite-sample variance.
    realized_h = float(torch.linalg.norm(seed.h_init - anchor_h).item())
    expected_h = 0.1 * (h_dim**0.5)
    assert expected_h > 0
    assert abs(realized_h - expected_h) / expected_h < 0.2, (
        f"realized ||h_init - anchor_h||={realized_h:.4f} not within ±20% of "
        f"expected {expected_h:.4f}"
    )

    # perturbation_magnitude is the L2 of the realized (concatenated) noise.
    # By construction the noise on z_init = z_init - anchor_z = z_init.
    realized_combined = float(
        torch.linalg.norm(
            torch.cat(
                [(seed.h_init - anchor_h).flatten(), (seed.z_init - anchor_z).flatten()]
            )
        ).item()
    )
    assert abs(seed.perturbation_magnitude - realized_combined) < 1e-5


def test_hybrid_convex_combination() -> None:
    wm = _make_world_model(seed=0)
    buf = SequenceReplayBuffer(capacity=2000, sequence_length=32)
    _populate_buffer(buf, n_transitions=1500, episode_length=200)
    cfg = SeedSelectionConfig(
        mode="hybrid",
        perturbation_sigma=0.1,
        hybrid_alpha_distribution="fixed_0_5",
        replay_min_segment_age_steps=1000,
        replay_warmup_length=8,
    )

    g = torch.Generator()
    g.manual_seed(99)
    seed = select_seed(buf, wm, cfg, g)

    # All four provenance fields populated for hybrid.
    assert seed.mode == "hybrid"
    assert seed.replay_segment_id is not None
    assert seed.replay_step_offset == 0
    assert seed.perturbation_magnitude is not None
    assert seed.hybrid_mixture_alpha == 0.5

    # Reproduce the internal computation by replaying the same operations
    # on a freshly-seeded generator. The order in select_seed's hybrid
    # branch is: (1) window pick, (2) warmup (consumes L * z_dim normals),
    # (3) perturbation noise on h then z, (4) alpha (no rng draws for
    # "fixed_0_5").
    seed_gen = torch.Generator()
    seed_gen.manual_seed(seed.rng_seed)
    starts = buf.valid_seed_window_starts(
        length=cfg.replay_warmup_length,
        min_age_steps=cfg.replay_min_segment_age_steps,
    )
    pick = int(torch.randint(0, len(starts), (1,), generator=seed_gen).item())
    assert starts[pick] == seed.replay_segment_id

    obs_seq, action_seq = buf.get_window(starts[pick], cfg.replay_warmup_length)
    h_replay, z_replay = _run_warmup(wm, obs_seq, action_seq, seed_gen)
    h_perturbed, z_perturbed, magnitude = _perturb_anchor(
        h_replay, z_replay, cfg.perturbation_sigma, seed_gen
    )

    expected_h = 0.5 * h_replay + 0.5 * h_perturbed
    expected_z = 0.5 * z_replay + 0.5 * z_perturbed
    assert torch.allclose(seed.h_init, expected_h, atol=1e-6)
    assert torch.allclose(seed.z_init, expected_z, atol=1e-6)
    assert abs(seed.perturbation_magnitude - magnitude) < 1e-5
