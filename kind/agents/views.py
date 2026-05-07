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


def split(
    step: WorldModelStep,
    intrinsic: Tensor,
    *,
    self_prediction_error: Tensor,
    self_prediction_error_masked: bool,
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
    """
    policy = PolicyView(
        h=step.h,
        z=step.z,
        self_prediction_error=self_prediction_error,
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
        self_prediction_error_masked=self_prediction_error_masked,
    )
    return policy, telemetry


__all__ = ["PolicyView", "TelemetryView", "split"]
