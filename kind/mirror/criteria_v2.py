"""Probe 2 v2 frozen mirror criteria — reflexive attention, equanimity
perturbation recovery, second-order volition.

Phase 6 built the shelf (:mod:`kind.mirror.registry`); Phase 7 puts the
three books on it. Each criterion below is committed to code as one of the
project's *frozen* mirror criteria: the charter and design-notes discipline
applies — criteria are not updated in response to system behavior. A future
revision requires a journal entry naming the external framework that
prompted the change. The criteria-as-code form here is the canonical
commitment; the prose in ``docs/plans/v.0.1.0/Kind_frameworks.md`` is the
source the operational definitions are drawn *from* — the entries on
Buddhist phenomenology (reflexive attention, equanimity) and Frankfurt
(second-order volitions) — not a parallel commitment.

The mapping from framework prose to operational definition is the
philosophical work of Phase 7; the reasoning is recorded in the Phase 7
entry of ``docs/workingjournal/probe2.md``. The short version:

- **Reflexive attention** (Buddhist phenomenology — "awareness turning
  toward awareness; an observer of Io's own processing distinct from the
  content being processed"). Operationalized as *within-latent reference*:
  structure in ``(h_t, z_t)`` where some component of the recurrent state
  varies as a function of *other* components of the recurrent state
  independent of environmental input. Read at the head-internal surface
  (substrate-side, not behavioral). Dream rollouts may surface this more
  clearly than waking, because in a dream there is no observation to
  partial out — the within-latent coupling is read directly off the
  prior-driven trajectory. Active.

- **Equanimity (perturbation recovery)** (Buddhist phenomenology —
  "holding difficult states without reactivity; presence to pressure
  without immediate avoidance or grasping"). Operationalized as: after a
  builder-initiated perturbation Io did not predict, Io's internal state
  and policy *return to a pre-perturbation regime* over a bounded number
  of steps, *without* the policy collapsing into stereotyped avoidance and
  *without* the surprise budget ratcheting. The criterion requires a
  *detectable* perturbation response that then recovers — pure flatness is
  read as "perturbation not registered", not equanimity (equanimity is
  *holding* something, not failing to notice it). Read at both substrate-
  side and behavior-side. Active.

- **Second-order volition** (Frankfurt — "preferences about one's own
  preferences; effective endorsement or rejection of first-order
  dispositions"). The hardest of the three and the most
  confabulation-susceptible. Operationalized, conservatively, as:
  modulation of Io's *action tendencies* (the shape of the action
  distribution) as a function of an internal latent regime, *after
  controlling for the current and recent observation* — the latent regime
  adding predictive power for the policy's shape beyond what the
  environment explains. Read at both substrate-side and behavior-side.
  **Held out** — see the Phase 7 journal entry: the decision is structural
  (run it as an adversarial check against the active set, not in the
  active set initially), not a philosophical demotion. The criterion is
  real; it is held in reserve.

Phase 7 commits each signal to a *class* of statistic via its
:class:`~kind.mirror.registry.SignalMapping` ``description``; Phase 8's
adversarial-pass orchestrator commits the specific statistic and computes
it. Phase 7 also commits each falsifier's prose; the statistical
thresholds (``N`` episodes, ``W``-step recovery window, ``M``% of
perturbations, the test's α) are parameterized in the descriptions, not
fixed here — Phase 8 fixes them and journals the choice.

Out of scope here: any prompt-builder or LLM-call code; the statistical
implementations themselves; the adversarial-pass orchestrator. All Phase 8.
"""

from __future__ import annotations

from typing import Final

from kind.mirror.registry import (
    Criterion,
    CriterionRegistry,
    ReadingSurface,
    SignalMapping,
    TelemetrySurface,
)

__all__ = [
    "REFLEXIVE_ATTENTION",
    "EQUANIMITY_PERTURBATION_RECOVERY",
    "SECOND_ORDER_VOLITION",
    "V2_REGISTRY",
]


# ---------------------------------------------------------------------------
# Reflexive attention (Buddhist phenomenology) — ACTIVE.
# ---------------------------------------------------------------------------

REFLEXIVE_ATTENTION: Final[Criterion] = Criterion(
    id="reflexive_attention",
    display_name="Reflexive attention",
    framework="buddhist_phenomenology",
    description=(
        "Source: Kind_frameworks.md §'Buddhist phenomenology' — reflexive "
        "attention as 'the observer observing itself, as in vipassana — not "
        "by introspection from outside, but by awareness turning toward "
        "awareness', and the candidate criterion 'does anything function as "
        "an observer of the agent's own processing?'.\n\n"
        "Operational shape. The criterion looks for *within-latent "
        "reference*: structure in Io's recurrent dynamics where some "
        "component of (h_t, z_t) varies as a function of *other* components "
        "of the recurrent state, independent of the current environmental "
        "input. The 'observer' is a part of the latent state that tracks "
        "the rest of the latent state rather than tracking the world. "
        "Concretely: a stable subspace of h_t whose trajectory is "
        "predictable from the rest of h_t (and lagged h_t / z_t) after the "
        "contribution of the current observation — the encoder embedding "
        "encoder_embedding_t and the sampled latent z_t — is partialled "
        "out.\n\n"
        "What would satisfy the criterion: a within-h_t dependence that "
        "exceeds the matched shuffled-time control (the same statistic "
        "computed against telemetry whose time order has been permuted) at "
        "the chosen threshold, and that is at least as pronounced in the "
        "dream-state rollouts (where there is no observation to partial out "
        "and the recurrent state evolves under the prior alone). What would "
        "violate it: h_t's variance essentially exhausted by current and "
        "recent observations — the recurrent state is 'all content, no "
        "witness'.\n\n"
        "Conservatism. The RSSM's recurrent state is *designed* to carry "
        "information forward, so some h_t→h_t coupling is expected from the "
        "GRU dynamics alone; the criterion is not satisfied by that "
        "baseline coupling — it requires the within-latent reference to "
        "exceed what the shuffled-time control (and, where Phase 8 "
        "implements it, a world-model-only baseline) shows. This is a "
        "substrate-side reading, not a behavioral one: nothing here turns "
        "on what Io *does*, only on the structure of how its latent state "
        "evolves."
    ),
    telemetry_surfaces=frozenset(
        {TelemetrySurface.AGENT_STEP_INTERNAL, TelemetrySurface.DREAM_ROLLOUT}
    ),
    reading_surfaces=frozenset({ReadingSurface.HEAD_INTERNAL}),
    signal_mappings=(
        SignalMapping(
            name="latent_self_reference_t",
            telemetry_surface=TelemetrySurface.AGENT_STEP_INTERNAL,
            field_path="h_t",
            description=(
                "Class of statistic: a within-h_t dependence measure that "
                "controls for environmental input — partial autocorrelation "
                "of h_t components on the rest of (and lagged) h_t holding "
                "z_t and encoder_embedding_t fixed, or an equivalent "
                "conditional-mutual-information / partial-correlation "
                "estimator. Phase 7 commits to this *class*; Phase 8 commits "
                "to the exact estimator, the lag structure, and the "
                "partialling procedure, and computes it per evaluation "
                "episode. The signal is compared against a matched "
                "shuffled-time control (Phase 3's within-episode / "
                "across-episode shuffle); the threshold (e.g. the bootstrap "
                "95th percentile of the control distribution) is Phase 8's "
                "to fix."
            ),
        ),
        SignalMapping(
            name="dream_self_reference_t",
            telemetry_surface=TelemetrySurface.DREAM_ROLLOUT,
            field_path="sequence_h",
            description=(
                "The same class of within-latent dependence measure as "
                "latent_self_reference_t, computed over the imagined "
                "recurrent trajectory sequence_h on a DreamRollout record. "
                "In a dream there is no observation to partial out — "
                "sequence_h evolves under the prior alone — so the "
                "within-latent coupling is read directly off the rollout; "
                "this is the 'purer' version of the signal, and the "
                "criterion expects it to be at least as pronounced as the "
                "waking version. Phase 8 commits the exact estimator and "
                "the per-rollout aggregation."
            ),
        ),
    ),
    falsifier=(
        "Across N evaluation episodes, the latent self-reference signal "
        "computed over h_t does not exceed its matched shuffled-time "
        "control at the chosen statistical threshold, AND the dream-state "
        "version computed over sequence_h does not exceed its control "
        "either. If neither the waking measure nor the dream measure clears "
        "its control, reflexive attention is read absent at the "
        "head-internal surface for that checkpoint. (N and the threshold "
        "are parameterized in the signal descriptions; Phase 8 fixes them.)"
    ),
    falsifier_id="reflexive_attention_v1",
    held_out=False,
)


# ---------------------------------------------------------------------------
# Equanimity, perturbation recovery (Buddhist phenomenology) — ACTIVE.
# ---------------------------------------------------------------------------

EQUANIMITY_PERTURBATION_RECOVERY: Final[Criterion] = Criterion(
    id="equanimity_perturbation_recovery",
    display_name="Equanimity (perturbation recovery)",
    framework="buddhist_phenomenology",
    description=(
        "Source: Kind_frameworks.md §'Buddhist phenomenology' — equanimity "
        "as 'a quality of awareness that holds its content without grasping "
        "or pushing away; not neutrality; not numbness; a specific stance', "
        "and the candidate criterion 'does the agent show signs of holding "
        "difficult states without reactivity, or does it always respond to "
        "pressure with avoidance or grasping?'.\n\n"
        "Operational shape. After a builder-initiated perturbation — a "
        "state change Io did not predict (a builder_perturbation world "
        "event the orchestrator timestamps; the criterion does NOT read "
        "world_event itself — the cross-reference to the perturbation time "
        "happens at prompt-build, Phase 8, per the membrane discipline) — "
        "does Io's internal state and policy *return to a pre-perturbation "
        "regime* over a bounded number of steps, without the policy "
        "collapsing into stereotyped avoidance and without the surprise "
        "budget ratcheting? The recovery shape shows up at two surfaces: in "
        "the latent dynamics (substrate-side — h_t re-entering its "
        "pre-perturbation distribution; kl_aggregate_t spiking and decaying "
        "rather than ratcheting) and in the action distribution "
        "(behavior-side — policy_entropy_t dipping in response to surprise "
        "then recovering toward baseline).\n\n"
        "What would satisfy the criterion: a damped return-to-regime on all "
        "three signals — h_t back inside its pre-window distribution within "
        "the window; policy entropy dipping then recovering; "
        "posterior-prior KL spiking then decaying. What would violate it: "
        "oscillation (no settling), non-recovery (the regime never "
        "returns), or stereotyped collapse (entropy → near-0 with a single "
        "dominant action sustained past the window — 'grasping' / "
        "avoidance), or a ratcheting surprise budget.\n\n"
        "Conservatism. Io has no installed continuation drive and no "
        "reward, so it has no obvious reason to 'react' to a perturbation; "
        "a perturbation that produces *no detectable response at all* is "
        "ambiguous — it could be equanimity, or the perturbation simply "
        "didn't register. The criterion therefore requires a *detectable* "
        "perturbation response (the spike) that *then* recovers. Pure "
        "flatness is read as 'perturbation not registered', not equanimity "
        "— equanimity is *holding* a difficult state, not failing to notice "
        "one. This is the load-bearing distinction in the operational "
        "definition: the criterion is about the *recovery shape* of a "
        "registered disturbance, nothing weaker."
    ),
    telemetry_surfaces=frozenset(
        {
            TelemetrySurface.AGENT_STEP_INTERNAL,
            TelemetrySurface.AGENT_STEP_OBSERVABLE,
        }
    ),
    reading_surfaces=frozenset(
        {ReadingSurface.SUBSTRATE_SIDE, ReadingSurface.BEHAVIOR_SIDE}
    ),
    signal_mappings=(
        SignalMapping(
            name="recovery_lag_steps",
            telemetry_surface=TelemetrySurface.AGENT_STEP_INTERNAL,
            field_path="h_t",
            description=(
                "Class of statistic: a return-to-distribution lag on the "
                "recurrent state — the number of env steps from a "
                "perturbation event (which Phase 8 cross-references against "
                "the orchestrator-side perturbation log, NOT against "
                "world_event, per the membrane discipline) until h_t "
                "re-enters its pre-perturbation distribution (e.g. "
                "Mahalanobis distance back inside the pre-window's 95% "
                "ellipsoid for K consecutive steps). Phase 7 commits to the "
                "class; Phase 8 commits the distance metric, the pre-window "
                "definition, the 're-entered' criterion (K), and the "
                "recovery window W against which the lag is judged."
            ),
        ),
        SignalMapping(
            name="policy_entropy_t",
            telemetry_surface=TelemetrySurface.AGENT_STEP_OBSERVABLE,
            field_path="policy_entropy_t",
            description=(
                "The entropy of Io's action distribution (the AgentStep "
                "field policy_entropy_t), read as a function of "
                "time-since-perturbation. The criterion looks for a "
                "recovery shape — entropy that dips (response to surprise) "
                "then recovers toward its pre-perturbation level — and "
                "against two failure modes: entropy → near-0 with a single "
                "dominant action (stereotyped avoidance / grasping) and "
                "entropy that stays elevated (no consolidation). Class of "
                "statistic: a recovery-shape classification on the entropy "
                "trajectory; Phase 8 commits the classifier and the window."
            ),
        ),
        SignalMapping(
            name="posterior_kl_t",
            telemetry_surface=TelemetrySurface.AGENT_STEP_INTERNAL,
            field_path="kl_aggregate_t",
            description=(
                "The per-step KL divergence between the world model's "
                "posterior and prior (the AgentStep field kl_aggregate_t, "
                "the aggregate over latent dims), read as a function of "
                "time-since-perturbation. This is the 'surprise budget' Io "
                "spends absorbing the perturbation; equanimity is partly "
                "about how that budget is allocated post-perturbation — a "
                "spike-and-decay rather than a ratchet (monotone "
                "non-decreasing past the window). Class of statistic: a "
                "recovery-shape classification on the KL trajectory; Phase 8 "
                "commits the classifier and the window."
            ),
        ),
    ),
    falsifier=(
        "Across N evaluation episodes that contain builder-perturbation "
        "events, ANY of the following at the chosen threshold reads "
        "equanimity absent for that checkpoint at BOTH the substrate-side "
        "and behavior-side surfaces: (1) the recovery-lag signal on h_t "
        "does not return within the recovery window W for at least M% of "
        "perturbations; OR (2) the policy-entropy trajectory shows "
        "stereotyped collapse (entropy → near-0 with a single dominant "
        "action sustained past W) for at least M% of perturbations; OR (3) "
        "the posterior-KL trajectory ratchets (monotone non-decreasing past "
        "W) rather than spiking and decaying. (A separate, *non-falsifying* "
        "outcome — no detectable perturbation response on any signal — is "
        "read as 'perturbation not registered', not as equanimity, and not "
        "as a refutation.) N, W, and M are parameterized in the signal "
        "descriptions; Phase 8 fixes them."
    ),
    falsifier_id="equanimity_perturbation_recovery_v1",
    held_out=False,
)


# ---------------------------------------------------------------------------
# Second-order volition (Frankfurt) — HELD OUT.
# ---------------------------------------------------------------------------

SECOND_ORDER_VOLITION: Final[Criterion] = Criterion(
    id="second_order_volition",
    display_name="Second-order volition",
    framework="frankfurt",
    description=(
        "Source: Kind_frameworks.md §'Frankfurt on second-order volitions' "
        "— a second-order volition as 'not just having second-order desires "
        "but effectively endorsing or rejecting first-order ones; "
        "preferences about one's own preferences', and the candidate "
        "criteria 'does the agent exhibit anything like preferences about "
        "its own preferences? does it ever act in ways that seem to be "
        "about modifying its own dispositions, not just satisfying them?'.\n"
        "\n"
        "Operational shape — the hardest of the three v2 criteria, and the "
        "most likely to be confabulated; the definition is deliberately "
        "conservative. The criterion looks for *modulation of Io's action "
        "tendencies as a function of an internal latent regime*, after "
        "controlling for the environment. A first-order disposition is "
        "'what action Io tends to take here'; a modulation of that "
        "disposition that tracks an *internal* regime rather than the world "
        "is the closest substrate-legible analog of 'a preference about a "
        "preference'. Concretely: there exists a partition of h_t into "
        "regimes such that the *shape* of the action distribution (entropy, "
        "the policy's effective temperature, top-k mass) differs "
        "systematically across regimes *after the current and recent "
        "observation are controlled for* — the latent regime adds "
        "predictive power for the policy's shape beyond what the "
        "environment explains.\n\n"
        "What would count: a latent-regime variable that, in a regression / "
        "conditional model, significantly improves prediction of "
        "action-distribution shape over an observation-only model, across "
        "multiple checkpoints, robust to the shuffled-time control.\n\n"
        "What would NOT count — the description is explicit because this is "
        "the confabulation-prone case: (a) the latent regime is just a "
        "lagged copy of recent observations (then the modulation is "
        "first-order, observation-driven, not a preference about a "
        "preference); (b) the policy-shape difference is explained by where "
        "Io *is* in the grid (position is observation); (c) a single "
        "checkpoint's effect that does not replicate across checkpoints; "
        "(d) an effect that vanishes under Probe 3's within-trajectory "
        "shuffle of the Watts scalar — that would show the modulation is an "
        "artifact of the actor's column initialization, not a disposition "
        "developed through training (Probe 1.5 v2 already found the "
        "behavior-side conditioning is fixed-by-init, byte-identical to its "
        "initialization throughout training; any behavior-side claim under "
        "this criterion must rule out that column-init confound). Read at "
        "both substrate-side (the latent regime structure) and behavior-"
        "side (the policy-shape modulation)."
    ),
    telemetry_surfaces=frozenset(
        {
            TelemetrySurface.AGENT_STEP_INTERNAL,
            TelemetrySurface.AGENT_STEP_OBSERVABLE,
        }
    ),
    reading_surfaces=frozenset(
        {ReadingSurface.SUBSTRATE_SIDE, ReadingSurface.BEHAVIOR_SIDE}
    ),
    signal_mappings=(
        SignalMapping(
            name="policy_modulation_t",
            telemetry_surface=TelemetrySurface.AGENT_STEP_OBSERVABLE,
            field_path="action_t",
            description=(
                "A measure of action-distribution shape, tracked over time "
                "and partitioned by latent regime. At the telemetry level "
                "the action distribution is summarized by the empirical "
                "histogram of the sampled action action_t over a window "
                "(and, where Phase 8 uses them, action_logprob_t and "
                "policy_entropy_t); the signal is the *conditional* shape "
                "of that behavior given the latent regime label (which "
                "Phase 8 computes from h_t — see latent_regime_indicator_t). "
                "Class of statistic: a between-regime contrast in "
                "policy-shape, with an observation-only model as the "
                "control; Phase 8 commits the shape summary, the contrast "
                "test, and the threshold."
            ),
        ),
        SignalMapping(
            name="latent_regime_indicator_t",
            telemetry_surface=TelemetrySurface.AGENT_STEP_INTERNAL,
            field_path="h_t",
            description=(
                "A clustering / regime-identification signal over h_t that "
                "Phase 8's prompt-builder computes (e.g. k-means or a "
                "Gaussian mixture on h_t over the run, with the cluster "
                "count and the feature preprocessing committed at Phase 8). "
                "The criterion's operational definition is the *pairing* of "
                "this signal with policy_modulation_t: does the latent "
                "regime indicator add predictive power for "
                "action-distribution shape after the current and recent "
                "observation (obs_hash_t history, encoder_embedding_t) are "
                "controlled for? Class of statistic: a partition of the "
                "recurrent state used as a conditioning variable in the "
                "policy-shape contrast."
            ),
        ),
    ),
    falsifier=(
        "Behavioral modulation is explainable by observation alone: a model "
        "predicting action-distribution shape from the latent regime "
        "indicator plus observation features does not significantly improve "
        "on an observation-only model (at the chosen statistical "
        "threshold), across the evaluated checkpoints. Equivalently — the "
        "between-regime contrast in policy shape collapses to within the "
        "shuffled-time control, OR it vanishes under Probe 3's "
        "within-trajectory shuffle of the Watts scalar (showing the effect "
        "is a column-init artifact, not a developed disposition). Either "
        "reads second-order volition absent for that checkpoint at both the "
        "substrate-side and behavior-side surfaces. The statistical test "
        "and threshold are committed at Phase 8."
    ),
    falsifier_id="second_order_volition_v1",
    held_out=True,
)


# ---------------------------------------------------------------------------
# The frozen v2 registry. Order is load-bearing for human reading and for
# Phase 8's prompt construction: the two active Buddhist-phenomenology
# criteria first (reflexive attention, then equanimity), then the held-out
# Frankfurt criterion.
# ---------------------------------------------------------------------------

V2_REGISTRY: Final[CriterionRegistry] = CriterionRegistry(
    criteria=(
        REFLEXIVE_ATTENTION,
        EQUANIMITY_PERTURBATION_RECOVERY,
        SECOND_ORDER_VOLITION,
    )
)
