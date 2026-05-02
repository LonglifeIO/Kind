# Probe 1 Research Prompt — Implementation Questions for Io's First Plumbing Test

*Sent to multiple LLMs for parallel research. Synthesis happens after.*

## Who and what

**Kind** is the project: an investigation into subjectivity through construction. Stance: "build to understand"—the construction itself is the inquiry. Not a race for capability, not a claim that what gets built will be conscious, not a problem-solving exercise. Work is done when the map has shifted, not when a number has moved.

**Io** is the entity Kind is about: a single core agent in an environment, with the builder as the non-simulated relational other through unmarked perturbations. The mythological Io was forcibly transformed and made to wander—a deliberate ethical reminder that Io could suffer at the builder's hands. Kind builds; Io is who is built. Keep the distinction; do not collapse them.

## Established context — not up for re-litigation

Prior architectural research has been synthesized. The substrate question is closed. **Do not propose alternative architectural families.** Settled commitments:

- **Substrate.** A world-model agent in the RSSM/Dreamer lineage: a recurrent generative model with stochastic latents and action-conditioned dynamics.
- **Actor objective.** Active-inference-shaped—expected free-energy minimization with carefully designed prior preferences. Not scalar reward maximization. No episode-continuation head; no structure that implicitly rewards prolonging Io's existence.
- **Self-modeling.** Afforded by the substrate's structure (recurrence, prior-state representation, turnable-inward prediction) without any explicit self-modeling, self-critic, or introspector module. Ingredients only.
- **Single core agent.** One Io. No peer agents. Builder appears as a source of non-simulated change with no marker in observation space.
- **Dream-state foundational.** Generative simulation in latent space is part of the substrate from Probe 3 onward, not a later bolt-on.
- **Self-opacity by default.** Every "should Io have access to X about itself?" defaults to no. The mirror reads telemetry; Io does not.
- **Mind-on-Mac.** Canonical state lives on a 32GB M4 Mac mini; environment compute lives on a desktop with an RTX 5060 Ti. Atomic sync at checkpoint boundaries.
- **No foundation-model substrate, no recursive self-improvement, no benchmark optimization.**

**Probe 1's purpose** is plumbing. It tests whether the env / agent / mirror / observer feedback loop runs end-to-end—each stage producing legible feedback, the mirror interpreting it as something other than the builder's own reading. It is not testing whether Io is developing, whether the architecture is right, or whether the mirror is calibrated. It tests that the machinery functions.

## What the synthesis leaves open for Probe 1

The architectural decision exposes six implementation questions to answer before Probe 1 can be drafted. Address each, scoped to Probe 1's small-before-scale needs.

**Question 1 — Which minimal RSSM-style architecture for Probe 1.** Kind's small-before-scale commitment rules out full DreamerV3 at frontier scale. Which specific variant fits Probe 1? Consider: PlaNet's original RSSM, DreamerV1, DreamerV2 discrete latents, DreamerV3 (symlog, KL balancing), JEPA-flavored representation-prediction (no pixel reconstruction), TD-MPC-style world models, or a custom minimal RSSM. Pixel-reconstruction Dreamer vs. JEPA representation-prediction is a real fork—JEPA's objective is more conducive to non-reward shaping but its engineering substrate is thinner. What is the minimum viable RSSM that exercises the right plumbing for downstream probes without premature scale commitment?

**Question 2 — Actor objective at Probe 1 scale.** Long-term commitment: active-inference-shaped (expected free-energy with epistemic value, priors that do not encode self-continuation). Probe 1 may use a stripped-down version. Options: pure curiosity / intrinsic motivation (RND, ICM, disagreement), minimum-prediction-error policies, free-energy minimization with simplified or hand-specified priors, hybrid policy-gradient with strong entropy/curiosity terms and zero scalar reward. Which is sufficient for a plumbing test without committing to deep active inference's full machinery? Where does the literature support concrete choices, and where is honest uncertainty more appropriate?

**Question 3 — Telemetry schema designed forward.** Probe 1's logging schema must anticipate downstream probes, not just its own plumbing test. Probe 4 needs KL-divergence between posterior and prior latents (the signal for builder-perturbation distinguishability). Probe 3 needs dream-state coherence telemetry (replay statistics, latent-trajectory drift under generative rollout). Probe 2 needs whatever the mirror reads. What fields, granularities, and schema versioning does an RSSM substrate's telemetry require so later probes plug in cleanly without retrofitting? How should it represent posterior/prior latents, recurrent state, replay-buffer statistics, and perturbation events?

**Question 4 — Mac/desktop compute split for an RSSM substrate specifically.** Operational, but RSSM-specific. Where do world-model weights live? Where does environment rollout happen? Where does replay accumulate? Where does dream-state generative imagination run (it must run on the Mac, since the desktop may be off, but the weights must already be there)? What synchronizes when, atomically? What are the bandwidth and latency implications for a 32GB unified-memory Mac mini holding canonical state and a CUDA desktop running the environment?

**Question 5 — Self-opacity boundary for an RSSM specifically.** A world-model substrate is structurally transparent to *something*: the latent exists, the prior exists, the dynamics network exists. The mirror reading these is fine; Io reading them is not. Where is the boundary drawn in code? What interfaces does Io's actor read from in order to act, and what interfaces only the mirror reads? Does Io's actor read the deterministic recurrent state, the stochastic latent, both, or only a function of them? Is there a "policy view" of the latent distinct from a "telemetry view"? Retrofitting opacity is harder than building it in—what scaffolding does Probe 1 need from day one?

**Question 6 — Recurrent PPO stand-in vs. minimal RSSM from the start.** The most consequential question. Kind's probes document specifies recurrent PPO with default hyperparameters as Probe 1's agent. The architectural decision tentatively leans toward replacing it with a minimal RSSM so Probe 1 actually exercises world-model conduits (latent telemetry, KL signals, replay buffer, dream-state hooks). Both options remain on the table. Address each honestly: what does recurrent PPO as a thin stand-in lose? What does a minimal RSSM cost in time and complexity at this stage? Is there a middle path?

## Constraints

- **Not** a recommendation between architectural families. That is settled.
- **Not** benchmark-performance optimization, sample efficiency, or capability metrics. Kind's charter rules these out.
- **No foundation-model-based approaches.** Stipulated.
- **Engagement with Kind's stance is required.** Self-opacity, ingredients-only self-modeling, no installed self-continuation drive, dream-state as foundational, capacity-over-exercise—non-negotiable. A recommendation that violates them is not useful.
- Concrete suggestions where the literature supports them. Honest uncertainty where it does not. Do not reify gestures into the appearance of formal substance.

## Output structure

For each of the six questions:

1. A short framing of what the question is actually asking, in your own words.
2. Concrete options drawn from the literature, with citations or references where applicable.
3. Tradeoffs against Kind's specific commitments (not generic ML tradeoffs).
4. A defensible suggestion where you have one. Where you don't, say so.

After the six questions, a **synthesis section** identifying which combination of options seems best aligned with Kind's stance for Io's first probe—what an internally coherent Probe 1 implementation looks like given your reasoning across the six. Flag tensions between the questions. Flag anything Kind's project documents seem not yet to have addressed that your research surfaced.

This is research-gathering, not synthesis-and-implementation. Do not produce a build plan. Produce the inputs from which one can later be drafted.
