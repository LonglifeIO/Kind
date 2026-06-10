"""Phase 1 gate tests for the Probe 1.5 self-prediction head and EMA target.

Plan §4 names the first three gates this file owns:

* **#1 self-prediction forward shape** — ``WorldModel.step`` returns a
  ``WorldModelStep`` with a ``self_prediction`` tensor of shape
  ``(B, h_dim)``; ``WorldModel.compute_self_prediction_target`` returns a
  shape-``(B, h_dim)`` tensor for any of the three target modes; the
  configured loss form (cosine or MSE) applied to a known fixed-pair input
  produces the expected scalar arithmetically.
* **#2 EMA target update mechanics** — after a single in-place perturbation
  of the online encoder + GRU, ``_update_ema_target`` shifts the EMA
  parameters toward the online parameters by exactly ``(1 - ema_decay)``;
  with the online held fixed, the per-step distance to the target decays
  monotonically by the configured ``ema_decay`` factor (the BYOL/SPR
  convention).
* **#3 self-prediction loss decreases** — on a fixed synthetic sequence,
  100 training steps of the world model with the auxiliary loss reduce
  ``self_prediction_loss`` (cosine form) measurably without the loss
  diverging or going NaN; recon and KL also remain finite.

Plan §4 gates #4 (opacity boundary) and #5 (integration smoke) are Phases
2 and 6 respectively and live in their own files. The gate-summary
meta-test (``test_gate_summary.py``) gains entries for #1–#3 in Phase 6
when all five Probe 1.5 gates exist; Phase 1 does not extend it.

Plus three additional unit tests for the constructor logic, the
``_frozen_projection`` allocation discipline, and the
``compute_self_prediction_target`` per-mode contract — each Phase 5 will
exercise via end-to-end runs, but Phase 1 needs the mode-by-mode forward
sanity already in place.

CPU only. Tiny sizes for speed; the 100-step gate test is the largest at
``h=8``, ``z=4``, ``embed=16``, batch=2 — well under one second on the
canonical machine.
"""

from __future__ import annotations

import pytest
import torch

from kind.agents.world_model import (
    SelfPredictionHead,
    WorldModel,
    WorldModelConfig,
    WorldModelStep,
)


# ---- shared fixtures -------------------------------------------------------


def _small_config(
    *,
    target_mode: str = "online",
    loss_form: str = "cosine",
    h_dim: int = 8,
    z_dim: int = 4,
    embed_dim: int = 16,
) -> WorldModelConfig:
    """Tiny but real config for fast CPU tests.

    Sizes are scaled down from Probe 1 defaults; ratios are kept intact so
    the forward exercises the same conduits at the same proportions.
    Different ``target_mode`` lets one fixture serve all three test paths.
    """
    return WorldModelConfig(
        obs_channels=1,
        obs_size=32,
        h_dim=h_dim,
        z_dim=z_dim,
        embed_dim=embed_dim,
        num_actions=5,
        action_emb_dim=8,
        mlp_hidden=16,
        free_bits_per_dim=1.0,
        self_prediction_hidden=16,
        ema_decay=0.99,
        self_prediction_target_mode=target_mode,  # type: ignore[arg-type]
        self_prediction_loss_form=loss_form,  # type: ignore[arg-type]
    )


def _fresh_inputs(
    config: WorldModelConfig, batch: int = 2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = torch.randn(batch, config.obs_channels, config.obs_size, config.obs_size)
    h = torch.zeros(batch, config.h_dim)
    z = torch.zeros(batch, config.z_dim)
    a = torch.zeros(batch, dtype=torch.long)
    return obs, h, z, a


# ---- gate test #1 (self-prediction forward shape) --------------------------


def test_gate_self_prediction_forward_shape() -> None:
    """Plan §4 gate #1 — Phase 1 portion (head's output and the per-mode
    targets; the view-side and AgentStep round-trip portions land in
    Phases 2 and 0 respectively).

    What this checks:

    1. ``WorldModel.step`` populates ``WorldModelStep.self_prediction`` with
       a tensor of shape ``(B, h_dim)``.
    2. ``WorldModel.compute_self_prediction_target`` returns shape
       ``(B, h_dim)`` for each of the three target modes
       (``"online"``, ``"frozen"``, ``"environmental"``).
    3. ``WorldModel.loss`` with the configured loss form applied to a
       known fixed-pair input produces the expected scalar arithmetically:

       * ``cosine``: ``1 - cos_sim(predicted, target)`` matches the
         hand-computed cosine distance on a deterministic input pair.
       * ``mse``: ``F.mse_loss(predicted, target)`` matches the
         hand-computed MSE on the same pair.
    """
    config = _small_config()
    wm = WorldModel(config)
    batch = 3
    obs, h, z, a = _fresh_inputs(config, batch=batch)

    # 1. step's self_prediction shape.
    step = wm.step(obs, h, z, a)
    assert isinstance(step, WorldModelStep)
    assert step.self_prediction.shape == (batch, config.h_dim)
    assert torch.isfinite(step.self_prediction).all()

    # 2a. compute_self_prediction_target — online mode.
    online_target = wm.compute_self_prediction_target(obs, h, z, a)
    assert online_target.shape == (batch, config.h_dim)
    assert torch.isfinite(online_target).all()
    # The online target must be detached — gradient must not flow back
    # into the EMA target's parameters (they have ``requires_grad=False``)
    # nor into the action embedding (which is shared with the online net).
    assert not online_target.requires_grad

    # 2b. compute_self_prediction_target — frozen mode (fresh WorldModel
    # configured with target_mode="frozen" so _frozen_projection is
    # allocated).
    frozen_config = _small_config(target_mode="frozen")
    frozen_wm = WorldModel(frozen_config)
    frozen_target = frozen_wm.compute_self_prediction_target(obs, h, z, a)
    assert frozen_target.shape == (batch, frozen_config.h_dim)
    assert torch.isfinite(frozen_target).all()
    assert not frozen_target.requires_grad

    # 2c. compute_self_prediction_target — environmental mode (requires
    # next_obs; the test verifies both that next_obs is required and
    # that the projected output has the right shape).
    env_config = _small_config(target_mode="environmental")
    env_wm = WorldModel(env_config)
    next_obs = torch.randn(
        batch, env_config.obs_channels, env_config.obs_size, env_config.obs_size
    )
    env_target = env_wm.compute_self_prediction_target(
        obs, h, z, a, next_obs=next_obs
    )
    assert env_target.shape == (batch, env_config.h_dim)
    assert torch.isfinite(env_target).all()
    assert not env_target.requires_grad
    # next_obs is required when target_mode="environmental".
    with pytest.raises(ValueError, match="next_obs"):
        env_wm.compute_self_prediction_target(obs, h, z, a, next_obs=None)

    # 3a. Loss form — cosine — on a known fixed-pair input. The synthesis
    # commits cosine as the default (BYOL convention; Tian/Chen/Ganguli
    # 2021's collapse-prevention analysis); the arithmetic must match
    # ``1 - cos_sim(predicted, target.detach()).mean()``.
    cosine_wm = WorldModel(_small_config(loss_form="cosine"))
    fake_step = WorldModelStep(
        h=torch.zeros(2, cosine_wm.config.h_dim),
        z=torch.zeros(2, cosine_wm.config.z_dim),
        q_params=(
            torch.zeros(2, cosine_wm.config.z_dim),
            torch.zeros(2, cosine_wm.config.z_dim),
        ),
        p_params=(
            torch.zeros(2, cosine_wm.config.z_dim),
            torch.zeros(2, cosine_wm.config.z_dim),
        ),
        kl_per_dim=torch.zeros(2, cosine_wm.config.z_dim),
        recon=torch.zeros(2, 1, 32, 32),
        embed=torch.zeros(2, cosine_wm.config.embed_dim),
        # Two known unit-orthogonal vectors → cos_sim = 0 → loss = 1.
        self_prediction=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 2
        ),
        energy_pred=torch.zeros(2, 1),
    )
    target_orth = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 2)
    loss_dict = cosine_wm.loss(
        fake_step, torch.zeros(2, 1, 32, 32), target_h_next=target_orth
    )
    assert "self_prediction_loss" in loss_dict
    assert loss_dict["self_prediction_loss"].item() == pytest.approx(1.0, abs=1e-6)

    # Two known parallel vectors → cos_sim = 1 → loss = 0.
    target_parallel = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 2
    )
    loss_parallel = cosine_wm.loss(
        fake_step, torch.zeros(2, 1, 32, 32), target_h_next=target_parallel
    )
    assert loss_parallel["self_prediction_loss"].item() == pytest.approx(
        0.0, abs=1e-6
    )

    # 3b. Loss form — mse — on the same fixed pair.
    mse_wm = WorldModel(_small_config(loss_form="mse"))
    pred = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 2)
    target = torch.tensor([[2.0, 4.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 2)
    fake_step_mse = WorldModelStep(
        h=torch.zeros(2, mse_wm.config.h_dim),
        z=torch.zeros(2, mse_wm.config.z_dim),
        q_params=(
            torch.zeros(2, mse_wm.config.z_dim),
            torch.zeros(2, mse_wm.config.z_dim),
        ),
        p_params=(
            torch.zeros(2, mse_wm.config.z_dim),
            torch.zeros(2, mse_wm.config.z_dim),
        ),
        kl_per_dim=torch.zeros(2, mse_wm.config.z_dim),
        recon=torch.zeros(2, 1, 32, 32),
        embed=torch.zeros(2, mse_wm.config.embed_dim),
        self_prediction=pred,
        energy_pred=torch.zeros(2, 1),
    )
    loss_mse = mse_wm.loss(
        fake_step_mse, torch.zeros(2, 1, 32, 32), target_h_next=target
    )
    # MSE = mean of (target - pred)^2 across all elements.
    # diff per non-zero elem: 1, 2, 3 → squared 1, 4, 9 → sum 14;
    # spread across 8 elements per row × 2 rows = 16 elements; mean = 14/8 = 1.75.
    assert loss_mse["self_prediction_loss"].item() == pytest.approx(1.75, abs=1e-6)


# ---- gate test #2 (EMA target update mechanics) ----------------------------


def test_gate_ema_target_update_mechanics() -> None:
    """Plan §4 gate #2 — the EMA target tracks the online parameters at the
    rate the configured ``ema_decay`` specifies.

    Three sub-checks. First: at construction, the EMA target is
    parameter-equal to the online network (``_initialize_ema_target_from_online``
    runs in ``__init__``). Second: after a single perturbation of the
    online parameters and a single ``_update_ema_target`` call, every EMA
    target parameter equals exactly
    ``ema_decay * target_before + (1 - ema_decay) * online_now``. Third:
    with the online held fixed (no further perturbations), 10
    consecutive ``_update_ema_target`` calls reduce the per-step
    parameter distance to the online target by exactly the
    ``ema_decay`` factor each step — i.e., the configured BYOL/SPR
    convergence rate.
    """
    config = _small_config()
    wm = WorldModel(config)
    ema_decay = config.ema_decay

    # 1. At construction, target == online for both encoder and gru_cell.
    for (n_t, p_t), (n_o, p_o) in zip(
        wm.target_encoder.named_parameters(),
        wm.encoder.named_parameters(),
        strict=True,
    ):
        assert torch.equal(p_t.data, p_o.data), (
            f"encoder param {n_t!r} differs from {n_o!r} at init: "
            f"_initialize_ema_target_from_online should make them equal"
        )
    for (n_t, p_t), (n_o, p_o) in zip(
        wm.target_gru_cell.named_parameters(),
        wm.gru_cell.named_parameters(),
        strict=True,
    ):
        assert torch.equal(p_t.data, p_o.data), (
            f"gru_cell param {n_t!r} differs from {n_o!r} at init"
        )

    # 2. After one in-place perturbation of online + one EMA update, the
    # EMA target's params equal exactly the convex combination the rule
    # specifies. The perturbation is large (unit-Gaussian per element)
    # so the equality is meaningful — a no-op EMA update would still
    # pass the equality if the perturbation were tiny.
    target_before_encoder = {
        n: p.data.clone() for n, p in wm.target_encoder.named_parameters()
    }
    target_before_gru = {
        n: p.data.clone() for n, p in wm.target_gru_cell.named_parameters()
    }
    with torch.no_grad():
        for p in wm.encoder.parameters():
            p.add_(torch.randn_like(p))
        for p in wm.gru_cell.parameters():
            p.add_(torch.randn_like(p))
    online_after_encoder = {
        n: p.data.clone() for n, p in wm.encoder.named_parameters()
    }
    online_after_gru = {n: p.data.clone() for n, p in wm.gru_cell.named_parameters()}

    wm._update_ema_target()

    for n, p in wm.target_encoder.named_parameters():
        expected = (
            ema_decay * target_before_encoder[n]
            + (1.0 - ema_decay) * online_after_encoder[n]
        )
        assert torch.allclose(p.data, expected, atol=1e-6), (
            f"encoder EMA update violated convex-combination rule on {n!r}"
        )
    for n, p in wm.target_gru_cell.named_parameters():
        expected = (
            ema_decay * target_before_gru[n]
            + (1.0 - ema_decay) * online_after_gru[n]
        )
        assert torch.allclose(p.data, expected, atol=1e-6), (
            f"gru_cell EMA update violated convex-combination rule on {n!r}"
        )

    # The action embedding is *not* EMA-tracked (plan §2.2 names exactly
    # encoder + gru_cell). Confirm there's no shadow ``target_action_embedding``
    # attribute and that the online action_embedding's gradient path is
    # untouched by the EMA mechanics.
    assert not hasattr(wm, "target_action_embedding"), (
        "EMA target's scope drifted: only encoder + gru_cell are EMA-tracked"
    )

    # 3. Hold online fixed (no further perturbations); run 10 consecutive
    # EMA updates and verify that the per-step distance to the online
    # decays monotonically by exactly ``ema_decay`` (= 0.99) per step —
    # the closed-form BYOL/SPR convergence rate when online is held
    # constant: ``|target_k - online| = ema_decay^k * |target_0 - online|``.
    distances: list[float] = []
    for _ in range(10):
        wm._update_ema_target()
        d_sq = 0.0
        for online_p, target_p in zip(
            wm.encoder.parameters(),
            wm.target_encoder.parameters(),
            strict=True,
        ):
            d_sq += float((target_p.data - online_p.data).pow(2).sum().item())
        for online_p, target_p in zip(
            wm.gru_cell.parameters(),
            wm.target_gru_cell.parameters(),
            strict=True,
        ):
            d_sq += float((target_p.data - online_p.data).pow(2).sum().item())
        distances.append(d_sq**0.5)

    # Monotonicity: each distance is strictly less than the previous.
    for i in range(1, len(distances)):
        assert distances[i] < distances[i - 1], (
            f"EMA convergence not monotonic: distances={distances}; "
            f"step {i} did not decrease from step {i - 1}"
        )
    # Convergence rate: ratio between consecutive distances should equal
    # ``ema_decay`` to within a tight tolerance (the only source of
    # deviation is float32 round-off at this scale).
    for i in range(1, len(distances)):
        ratio = distances[i] / distances[i - 1]
        assert ratio == pytest.approx(ema_decay, abs=1e-4), (
            f"convergence rate inconsistent with ema_decay={ema_decay!r}: "
            f"step {i} ratio={ratio!r}; distances={distances}"
        )


# ---- gate test #3 (self-prediction loss decreases) -------------------------


def test_gate_self_prediction_loss_decreases() -> None:
    """Plan §4 gate #3 — on a fixed synthetic sequence, 100 training steps
    of the world model with the auxiliary self-prediction loss reduce the
    cosine self-prediction loss measurably without any of recon, KL, or
    self-prediction loss going non-finite.

    Setup is deterministic (``torch.manual_seed`` at the top, no
    randomness in inputs across steps): the same observation tensor and
    the same prior ``(h, z, a)`` are reused for every training step. This
    is a contrived setup — the world model overfits to the fixed input —
    but that's the point: the head should learn to predict the EMA
    target's value of ``bar{h}_t`` in the limit, and the cosine loss
    should fall well below its initial value within 100 steps. The test
    is sized to be deterministic *and* to exercise the full (forward,
    target compute, loss compute, backward, optimizer step, EMA update)
    cycle the runner will execute in Phase 3.
    """
    torch.manual_seed(7)

    config = _small_config()
    wm = WorldModel(config)
    optimizer = torch.optim.Adam(
        [p for p in wm.parameters() if p.requires_grad], lr=1e-3
    )
    obs, h, z, a = _fresh_inputs(config, batch=2)
    lambda_self = 1.0  # Phase 1 test exercises the head's gradient
    # cleanly; runner uses 0.1 in Phase 3 (plan §6 row 1).

    losses: list[float] = []
    for _ in range(100):
        optimizer.zero_grad()
        step = wm.step(obs, h, z, a)
        target_h = wm.compute_self_prediction_target(obs, h, z, a)
        loss_dict = wm.loss(step, obs, target_h_next=target_h)
        # All three named instability indicators (synthesis §3 / Nilaksh
        # et al. via plan §5.3) must remain finite at every step.
        assert torch.isfinite(loss_dict["recon"]), (
            f"recon went non-finite at step {len(losses)}: {loss_dict['recon']}"
        )
        assert torch.isfinite(loss_dict["kl"]), (
            f"kl went non-finite at step {len(losses)}: {loss_dict['kl']}"
        )
        assert torch.isfinite(loss_dict["self_prediction_loss"]), (
            f"self_prediction_loss went non-finite at step {len(losses)}: "
            f"{loss_dict['self_prediction_loss']}"
        )
        losses.append(float(loss_dict["self_prediction_loss"].item()))
        combined = loss_dict["total"] + lambda_self * loss_dict["self_prediction_loss"]
        combined.backward()
        optimizer.step()
        wm._update_ema_target()

    # Initial vs final self-prediction loss. The cosine loss starts near 1
    # (the head's output is roughly random vs the EMA target's
    # roughly-random GRU output) and falls as the head learns to track the
    # EMA. The threshold is forgiving (initial mean / 2) — the gate's
    # purpose is to detect divergence or no-learning, not to pin a
    # specific numeric target.
    initial_mean = sum(losses[:10]) / 10.0
    final_mean = sum(losses[-10:]) / 10.0
    assert final_mean < initial_mean / 2.0, (
        f"self_prediction_loss did not decrease enough: initial_mean={initial_mean}, "
        f"final_mean={final_mean}; expected final < initial/2"
    )


# ---- regression: existing Probe 1 path holds at default --------------------


def test_probe_1_path_still_passes_at_target_mode_online() -> None:
    """Plan §2.2 ``Tests`` — regression test that the existing Probe 1
    forward / backward / loss path remains correct when
    ``target_mode="online"`` (the new default).

    This is a smaller, sharper check than the Probe 1 gate in
    ``test_agent_forward.py`` — that gate is itself updated to consume the
    Probe 1.5 path; this regression test verifies the *narrow* claim that
    callers can still run forward + backward without using the head at
    all (the runner will use the head, but other callers — e.g. the dream
    rollout machinery in Phase 3, the smoke script's tiny-cycle-without-aux
    path — must remain free to ignore the auxiliary loss).
    """
    config = _small_config()
    wm = WorldModel(config)
    obs, h, z, a = _fresh_inputs(config)
    step = wm.step(obs, h, z, a)
    # No target_h_next → self_prediction_loss is zero, total excludes it.
    loss = wm.loss(step, obs)
    assert loss["self_prediction_loss"].item() == pytest.approx(0.0)
    loss["total"].backward()
    # Every trainable parameter except the head's MLP gets a gradient
    # (the head's forward output flows nowhere when self_prediction_loss
    # is zero — backward through ``total`` does not visit the head).
    for name, param in wm.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("self_prediction_head."):
            assert param.grad is None, (
                f"unexpected gradient on {name}: when self_prediction_loss "
                f"is zero, the head's MLP should not be visited by backward "
                f"through total"
            )
            continue
        # Probe 3.5: the energy branch is analogous to the head — with no
        # ``sensed_energy_target`` the energy reconstruction term is excluded
        # from ``total``, so the energy decoder is not visited by backward, and
        # with no ``sensed_energy`` fused the energy encoder is not in the graph
        # at all. Both are correctly grad-free on this Probe-1-shaped path.
        if name.startswith("energy_encoder.") or name.startswith(
            "energy_decoder."
        ):
            assert param.grad is None, (
                f"unexpected gradient on {name}: with no sensed_energy / "
                f"sensed_energy_target, the energy branch should not be visited "
                f"by backward through total"
            )
            continue
        assert param.grad is not None, f"no gradient on {name}"
        assert torch.isfinite(param.grad).all()


# ---- unit tests for SelfPredictionHead alone -------------------------------


def test_self_prediction_head_forward_shape_and_finite() -> None:
    """``SelfPredictionHead`` is a small MLP ``h_dim → h_dim``; its forward
    must preserve the leading batch dim and emit a finite tensor.
    """
    head = SelfPredictionHead(h_dim=8, hidden_dim=16)
    h = torch.randn(4, 8)
    pred = head(h)
    assert pred.shape == (4, 8)
    assert torch.isfinite(pred).all()


def test_self_prediction_head_has_two_hidden_layers_at_default() -> None:
    """Plan §6 row 5 default — 2 hidden layers, hidden=200 (here scaled to
    16 for the small config). The head is structurally three Linear
    layers: input projection, hidden, output. The test counts named
    Linears so a future refactor that drops a layer fails this loudly.
    """
    head = SelfPredictionHead(h_dim=8, hidden_dim=16)
    linears = [
        m for m in head.modules() if isinstance(m, torch.nn.Linear)
    ]
    assert len(linears) == 3, (
        f"plan §6 row 5 default — head should have 2 hidden layers + 1 "
        f"output Linear; got {len(linears)} Linears"
    )


# ---- unit test for constructor + frozen projection -------------------------


def test_frozen_projection_allocated_only_when_target_mode_frozen() -> None:
    """Plan §2.2 — ``_frozen_projection`` is allocated at construction
    only when ``target_mode == "frozen"``; otherwise it is ``None``.
    Storing the projection in a parameter with ``requires_grad=False``
    keeps it serialisable through ``state_dict`` (Phase 3 checkpoint
    extension) without entering the optimizer's parameter set.
    """
    online_wm = WorldModel(_small_config(target_mode="online"))
    assert online_wm._frozen_projection is None

    env_wm = WorldModel(_small_config(target_mode="environmental"))
    assert env_wm._frozen_projection is None

    frozen_wm = WorldModel(_small_config(target_mode="frozen"))
    assert frozen_wm._frozen_projection is not None
    assert frozen_wm._frozen_projection.shape == (
        frozen_wm.config.h_dim,
        frozen_wm.config.h_dim,
    )
    # Random-orthogonal: the projection's columns (or rows) should be
    # close to orthonormal, so ``proj @ proj.T ≈ I``.
    proj = frozen_wm._frozen_projection.data
    identity_check = proj @ proj.T
    eye = torch.eye(frozen_wm.config.h_dim)
    assert torch.allclose(identity_check, eye, atol=1e-4), (
        "frozen projection is not random-orthogonal (orthogonality "
        "broken at construction); plan §2.2 / §8.1 require "
        "torch.nn.init.orthogonal_"
    )
    # The projection participates in state_dict but not in
    # named_parameters with requires_grad=True.
    assert not frozen_wm._frozen_projection.requires_grad
    state_dict = frozen_wm.state_dict()
    assert "_frozen_projection" in state_dict


def test_environmental_projection_only_when_needed() -> None:
    """Plan §2.2 — the environmental-mode projection is allocated only
    when ``target_mode == "environmental"`` *and* ``embed_dim != h_dim``
    (otherwise the EMA encoder's output already has the right dim and no
    projection is needed).
    """
    # embed_dim != h_dim → projection allocated, frozen.
    cfg_with_proj = _small_config(
        target_mode="environmental", h_dim=8, embed_dim=16
    )
    wm_with_proj = WorldModel(cfg_with_proj)
    assert wm_with_proj._environmental_projection is not None
    for p in wm_with_proj._environmental_projection.parameters():
        assert not p.requires_grad

    # embed_dim == h_dim → no projection needed.
    cfg_no_proj = _small_config(
        target_mode="environmental", h_dim=16, embed_dim=16
    )
    wm_no_proj = WorldModel(cfg_no_proj)
    assert wm_no_proj._environmental_projection is None

    # Non-environmental mode → never allocated.
    cfg_online = _small_config(target_mode="online")
    wm_online = WorldModel(cfg_online)
    assert wm_online._environmental_projection is None


def test_ema_target_parameters_are_requires_grad_false() -> None:
    """Plan §2.2 — both ``target_encoder`` and ``target_gru_cell`` carry
    ``requires_grad=False`` on every parameter; backward must never
    populate ``.grad`` on them (the EMA update is the only path that
    changes their values).
    """
    wm = WorldModel(_small_config())
    for p in wm.target_encoder.parameters():
        assert not p.requires_grad
    for p in wm.target_gru_cell.parameters():
        assert not p.requires_grad

    # Drive a backward and verify no .grad is populated on the EMA target
    # parameters.
    obs, h, z, a = _fresh_inputs(wm.config)
    step = wm.step(obs, h, z, a)
    target_h = wm.compute_self_prediction_target(obs, h, z, a)
    loss = wm.loss(step, obs, target_h_next=target_h)
    (loss["total"] + loss["self_prediction_loss"]).backward()

    for p in wm.target_encoder.parameters():
        assert p.grad is None
    for p in wm.target_gru_cell.parameters():
        assert p.grad is None


def test_initialize_ema_target_from_online_overwrites_target() -> None:
    """Plan §2.4 / §2.2 — ``_initialize_ema_target_from_online`` is the
    Phase 3 checkpoint-load helper that copies the online encoder + GRU
    state into the EMA target (the path used when loading a Probe 1
    checkpoint that has no EMA target weights on disk). The helper must
    work even when the EMA target's current state is divergent from the
    online — i.e., the helper *overwrites* the EMA target's state, it
    does not blend.
    """
    wm = WorldModel(_small_config())
    # Drive the EMA target away from the online by perturbing the online
    # and running the EMA update once — a regular EMA-tracked state, not
    # an init-time identity.
    with torch.no_grad():
        for p in wm.encoder.parameters():
            p.add_(torch.randn_like(p))
    wm._update_ema_target()
    # Confirm the EMA target now differs from the online (a normal
    # mid-training state).
    diffs = [
        (p_t.data - p_o.data).abs().sum().item()
        for p_t, p_o in zip(
            wm.target_encoder.parameters(), wm.encoder.parameters(), strict=True
        )
    ]
    assert any(d > 1e-6 for d in diffs)
    # Re-initialise; target must now equal online again.
    wm._initialize_ema_target_from_online()
    for p_t, p_o in zip(
        wm.target_encoder.parameters(), wm.encoder.parameters(), strict=True
    ):
        assert torch.equal(p_t.data, p_o.data)


# ---- compute_self_prediction_target — per-mode contract ---------------------


def test_compute_self_prediction_target_online_matches_target_gru_application() -> None:
    """Online-mode target equals the EMA GRU's recurrence on
    ``(h_prev, z_prev, a_prev)`` exactly. The arithmetic must agree with a
    hand-computed reference using the same EMA-tracked GRU.
    """
    wm = WorldModel(_small_config(target_mode="online"))
    obs, h, z, a = _fresh_inputs(wm.config)
    target = wm.compute_self_prediction_target(obs, h, z, a)
    # Reference: same operation under the EMA GRU + the online action
    # embedding (action_embedding is shared, not EMA-tracked).
    with torch.no_grad():
        a_emb = wm.action_embedding(a)
        gru_input = torch.cat([z, a_emb], dim=-1)
        reference = wm.target_gru_cell(gru_input, h)
    assert torch.equal(target, reference)


def test_compute_self_prediction_target_frozen_uses_projection_on_online_h_t() -> None:
    """Frozen-mode target equals the random-orthogonal projection of the
    *online* GRU's recurrence output (plan §2.2: "random-orthogonal
    projection of the *current* online ``h_t``").
    """
    wm = WorldModel(_small_config(target_mode="frozen"))
    obs, h, z, a = _fresh_inputs(wm.config)
    target = wm.compute_self_prediction_target(obs, h, z, a)
    with torch.no_grad():
        a_emb = wm.action_embedding(a)
        gru_input = torch.cat([z, a_emb], dim=-1)
        h_t_online = wm.gru_cell(gru_input, h)
        assert wm._frozen_projection is not None
        reference = torch.nn.functional.linear(h_t_online, wm._frozen_projection)
    assert torch.equal(target, reference)
