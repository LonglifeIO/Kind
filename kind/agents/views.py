"""PolicyView / TelemetryView opacity boundary.

Two frozen dataclasses and one ``split`` function. The actor reads
``PolicyView`` only — the deterministic recurrent state ``h``, the
sampled posterior latent ``z``, and (Probe 1.5 v2) the scalar
``self_prediction_error`` Io reads as the minimum self-pointing quantity
required for the second success criterion's "capacity to take its own
processing as an object of attention" affordance. The mirror caller and
telemetry writers read ``TelemetryView`` — everything in ``PolicyView``
plus the substrate's posterior/prior parameters, per-dimension KL, the
reconstruction tensor, the encoder embedding, the intrinsic signal
Phase 3b's ensemble produces, and (Probe 1.5) the full self-prediction
vector ``ĥ_{t+1}`` plus the masked-step flag. The ``split`` function is
the only place where a ``WorldModelStep`` is projected into the two
views.

**Probe 2 v2 ``zero_or_randomize_scalar`` lesion (plan §2.5; synthesis
§2.4 element 4).** ``split`` accepts an optional
``lesion_zero_or_randomize`` kwarg ``"zero" | "randomize" | None``. When
non-None, ``split`` overrides the ``self_prediction_error`` field on the
returned ``PolicyView`` with either ``0.0`` (zero variant) or
``Uniform(empirical_min, empirical_max)`` (randomize variant); the
TelemetryView's ``self_prediction_error_masked`` field is set to True
on the lesioned step (the scalar's value is sentinel, not empirical).
The TelemetryView's ``self_prediction`` vector is *not* overridden —
the substrate-side reading sees the head's output unchanged; only the
actor's behavior-side input is lesioned. The runner threads the kwarg
from ``RunnerConfig.lesion_kind`` plus
``RunnerConfig.lesion_zero_or_randomize_variant``; the empirical bounds
come from ``RunnerConfig.lesion_zero_or_randomize_empirical_min`` /
``..._max`` (defaults to ``[0.0, 1.0]`` covering the cosine loss
form's range; a real Probe 2 lesion run journals the bounds it pulls
from the source run's empirical distribution).

The opacity boundary is enforced at three levels per implementation plan
§7, in order of strictness from "checked at test time" to "checked at
type-check time" to "checked at runtime":

1. **Module boundaries.** ``kind/agents/views.py`` exports both classes;
   ``kind/agents/actor.py`` imports only ``PolicyView``. The dependency
   lint in ``tests/test_views.py`` walks the actor module's AST and
   rejects any import of ``TelemetryView`` (including aliased forms).
   The Probe 1.5 v2 additions to TelemetryView (``self_prediction``,
   ``self_prediction_error_masked``) do not change this — the actor
   still cannot reach TelemetryView at all.

2. **Type signatures.** ``Actor.forward(self, view: PolicyView)`` is the
   only entry point for the actor's view of the world model. mypy
   ``--strict`` catches passing a ``TelemetryView`` where a
   ``PolicyView`` is expected. The new field on PolicyView
   (``self_prediction_error``) is structurally part of PolicyView's
   type and reaches the actor through the existing entry point — no
   new entry point, no new type signature widening.

3. **Frozen dataclasses.** Both views are ``@dataclass(frozen=True)``.
   Attempting to assign to a field raises ``FrozenInstanceError``.
   Tensors referenced by the views are still mutable in place via
   PyTorch — the boundary is structural-by-default, not adversarial
   (synthesis §Q5). The Probe 1.5 scalar on PolicyView is frozen against
   reassignment exactly the same way ``h`` and ``z`` are.

The Probe 1.5 v2 single-scalar exception. Synthesis §1.3 (v2) lands on
**interface-level opacity preserved at the level the project documents
describe — Io does not read quantities about its own processing where
the read-access would install behavior or capability the project
documents do not describe — with one explicit and bounded
Watts-heuristic exception: the scalar ``self_prediction_error`` on
PolicyView, justified by the second success criterion's "capacity to
take its own processing as an object of attention" affordance, which is
the minimum form of read-access that delivers the affordance**. The
field set on PolicyView is exactly ``{h, z, self_prediction_error}``
and nothing else; future probes adding new actor-readable fields must
address the §2(b) (v2) four-part discipline at design time.

The split is a pure data-projection. The view is the data; the actor is
responsible for the ``concat([view.h, view.z, view.self_prediction_error
.unsqueeze(-1)], dim=-1)`` operation as the first step of its forward
pass (synthesis §Q5: no learned ``f(h, z)`` projection inside the view,
no preprocessing smuggled in).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from kind.agents.world_model import WorldModelStep


@dataclass(frozen=True)
class PolicyView:
    """The actor's view onto the world model. Exactly ``{h, z,
    self_prediction_error}`` and nothing else.

    All three fields are unavoidable inputs for action selection under
    partial observability plus the synthesis §1.3 (v2) self-attention
    affordance; restricting further would cripple the actor without
    buying meaningful additional opacity. Restricting more *broadly* —
    e.g. exposing posterior parameters, KL, the intrinsic signal, the
    full self-prediction vector, or any other quantity the mirror
    reads — is what the synthesis explicitly rules out: routing those
    quantities into the actor's input is a structural form of
    self-readout that crosses the synthesis §2(b) (v2) Watts-heuristic
    discipline.

    The ``self_prediction_error`` field is the v2 revision (synthesis
    §1.3): a scalar (shape ``()`` for the env-step path, shape ``(B,)``
    for batched paths) Io reads as the minimum self-pointing quantity
    required for the second success criterion's affordance. This is
    the single explicit Watts-heuristic exception the synthesis §2(b)
    (v2) authorizes at Probe 1.5 with the four-part discipline (which
    affordance, minimum form, alternatives considered, failure-mode
    controls) — all four journaled in `docs/decisions/Kind_probe1_5_synthesis.md`
    §1.3 / §2(b) and the Phase 2 entry of `docs/workingjournal/probe1_5.md`.

    The view is a pure data container. No ``concat`` field, no learned
    ``f(h, z, scalar)`` projection, no method that derives anything
    from the fields. The actor is responsible for the concat as the
    first operation of its forward pass; smuggling preprocessing into
    the view would be the kind of "implicit self-summary" the synthesis
    §Q5 specifically warns against.
    """

    h: Tensor
    z: Tensor
    self_prediction_error: Tensor


@dataclass(frozen=True)
class TelemetryView:
    """The mirror's view onto the world model. Everything in ``PolicyView``
    plus the substrate's exposed surface, the intrinsic signal, the full
    self-prediction vector, and the masked-step flag.

    The mirror caller reads this; the telemetry sink consumes it; the
    runner converts it into ``AgentStep`` records for serialisation.
    The intrinsic signal lives only here, never on ``PolicyView``: the
    actor's training loop receives the intrinsic value as a scalar
    argument from outside the view, not as an introspectable attribute.
    The Probe 1.5 ``self_prediction`` vector (``ĥ_{t+1}``, length
    ``h_dim``) and ``self_prediction_error_masked`` flag (true on the
    first step of each episode, when no empirical reading exists yet)
    likewise live only here — Io reads only the scalar derivative of
    the head's loss via PolicyView, not the full vector.

    ``recon_loss`` carries the reconstruction tensor from
    ``WorldModelStep.recon`` (shape ``(B, obs_channels, obs_size,
    obs_size)``). The field name follows the synthesis §Q3 telemetry
    schema's ``recon_loss_t`` (which IS a scalar in serialised form);
    the runner is responsible for converting the in-memory tensor to
    the serialised scalar at telemetry-write time. See the journal
    entry for Phase 3c for the naming-dissonance discussion.
    """

    h: Tensor
    z: Tensor
    q_params: tuple[Tensor, Tensor]
    p_params: tuple[Tensor, Tensor]
    kl_per_dim: Tensor
    recon_loss: Tensor
    embed: Tensor
    intrinsic_signal: Tensor
    self_prediction: Tensor
    self_prediction_error_masked: bool
    # Probe 3.5: the world model's decoded energy prediction (``(B, 1)``),
    # mirror-side only. Energy never reaches PolicyView (DP4) — it enters the
    # actor only implicitly through ``h, z``. The sensed/true energy scalars
    # are env-side (assembled into ``AgentStep`` by the runner), not
    # WorldModelStep fields, so they are not carried here.
    energy_pred: Tensor


def split(
    step: WorldModelStep,
    intrinsic: Tensor,
    *,
    self_prediction_error: Tensor,
    self_prediction_error_masked: bool,
    lesion_zero_or_randomize: Literal["zero", "randomize"] | None = None,
    lesion_empirical_min: float = 0.0,
    lesion_empirical_max: float = 1.0,
    lesion_rng: torch.Generator | None = None,
) -> tuple[PolicyView, TelemetryView]:
    """Project a ``WorldModelStep`` plus per-step signals into
    ``(PolicyView, TelemetryView)``.

    The intrinsic signal is the ensemble disagreement variance; the
    self-prediction error is the per-step scalar (computed by the runner
    via ``WorldModel.compute_self_prediction_target`` and the configured
    loss form before being passed in); the masked flag is True on the
    first step of each episode (when no empirical reading exists). The
    split function does not validate any of these — the views are pure
    data containers, and the runner is what knows how to compute them.

    Tensors are not copied: the views share references with the input
    step, so memory is not duplicated and gradients flow cleanly. The
    frozen dataclass prevents accidental field reassignment after
    construction; deeper mutation discipline (no in-place tensor edits)
    is a runtime convention enforced by the actor's signature
    constraints and by the rule that the actor never sees the
    ``TelemetryView`` reference at all.

    The full self-prediction vector ``step.self_prediction`` lands on
    ``TelemetryView.self_prediction`` only; the scalar derivative the
    runner computes lands on ``PolicyView.self_prediction_error`` and
    on ``TelemetryView`` (implicitly via the runner's separate
    population of ``AgentStep.self_prediction_error_t``). The asymmetry
    — Io reads the scalar; the mirror reads the vector plus the scalar
    plus the masked flag — is the synthesis §1.4 (v2) reading-asymmetry
    in code.

    The return order — ``(PolicyView, TelemetryView)`` — is part of
    the contract. Downstream code unpacks by position::

        policy, telemetry = split(
            step, intrinsic,
            self_prediction_error=scalar,
            self_prediction_error_masked=is_first_step,
        )

    **Probe 2 v2 ``zero_or_randomize_scalar`` lesion (plan §2.5).** When
    ``lesion_zero_or_randomize`` is ``"zero"``, the returned
    ``PolicyView.self_prediction_error`` is forced to ``0.0`` and the
    ``TelemetryView.self_prediction_error_masked`` flag is set to
    ``True`` (the scalar's value is sentinel, not empirical). When
    ``"randomize"``, the scalar is drawn from
    ``Uniform(lesion_empirical_min, lesion_empirical_max)``; the masked
    flag is also set ``True``. Either variant lesions only the actor's
    behavior-side input — the head and the EMA target continue to train
    normally and the substrate-side reads exactly like the un-lesioned
    run. The mirror writes ``self_prediction_error_t`` from the
    lesioned (sentinel) value via the runner's existing telemetry
    plumbing; the masked flag is what distinguishes a lesion-overridden
    record from a genuine first-step-of-episode mask. ``lesion_rng`` is
    the ``torch.Generator`` the runner threads through for
    determinism; if ``None``, the device's default generator is used.
    """
    if lesion_zero_or_randomize == "zero":
        scalar_for_view = torch.zeros(
            (), device=step.h.device, dtype=step.h.dtype
        )
        masked_for_view = True
    elif lesion_zero_or_randomize == "randomize":
        # Uniform(min, max) on the same device/dtype as ``step.h`` so the
        # actor's concat in forward picks up no device-mismatch error.
        # ``torch.rand`` accepts an optional ``generator`` kwarg; the
        # runner threads its sample-RNG through for determinism on CPU
        # tests and falls back to the default generator on accelerators
        # whose generator API is partial.
        unit = torch.rand(
            (), generator=lesion_rng, device=step.h.device, dtype=step.h.dtype
        )
        scalar_for_view = lesion_empirical_min + unit * (
            lesion_empirical_max - lesion_empirical_min
        )
        masked_for_view = True
    else:
        scalar_for_view = self_prediction_error
        masked_for_view = self_prediction_error_masked

    policy = PolicyView(
        h=step.h,
        z=step.z,
        self_prediction_error=scalar_for_view,
    )
    telemetry = TelemetryView(
        h=step.h,
        z=step.z,
        q_params=step.q_params,
        p_params=step.p_params,
        kl_per_dim=step.kl_per_dim,
        recon_loss=step.recon,
        embed=step.embed,
        intrinsic_signal=intrinsic,
        self_prediction=step.self_prediction,
        self_prediction_error_masked=masked_for_view,
        energy_pred=step.energy_pred,
    )
    return policy, telemetry


__all__ = ["PolicyView", "TelemetryView", "split"]
