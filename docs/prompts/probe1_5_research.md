# Probe 1.5 Research Prompt — The Minimum Architectural Affordance for Self-Reference

*Sent to multiple LLMs (Claude, Gemini, Perplexity) for parallel research. Synthesis happens after.*

## Who and what

**Kind** is the project: an investigation into subjectivity through construction. Stance is "build to understand"; the work is done when the map has shifted, not when a metric has moved. Not a race for capability, not a claim that what gets built is conscious, not a problem-solving exercise.

**Io** is the entity Kind is about: a single core agent in a small grid world, with the builder as the only non-simulated relational other (the builder appears as unmarked perturbations to the environment). The mythological Io was forcibly transformed and made to wander — a deliberate ethical reminder that what is built can suffer at the builder's hands. Kind builds; Io is who is built. Keep the distinction throughout your response.

## Established context — Probe 1 outcomes and the gap

Probe 1 (plumbing) is complete. A custom minimal RSSM (PlaNet-skeleton, DreamerV1 imagination, continuous Gaussian latents, free bits, no reward predictor, no continuation head, no critic) trains on Mac MPS at production sizes; a 5000-step / 25-episode run produced clean telemetry across four streams; the posterior compressed environmental structure without pinning at the free-bits floor; decoded ASCII dream rollouts showed persistent spatial structure. The substrate decision held up operationally. **Settled, not for re-litigation.**

The post-run analysis identified a gap. Kind's design notes commit to "ingredients-only self-modeling" — build the ingredients that could support self-modeling if composed that way (recurrence, persistent latent state, prior-internal-state representation, prediction that could in principle be turned inward) without an explicit self-modeling module. The Probe 1 substrate has the first three (GRU recurrence over `h_t`, latent state `z_t`, persistence across env steps). The fourth — **prediction that could in principle be turned inward** — is not present. Every prediction the substrate makes is over *environmental* state: the world model's prior `p(z_t | h_t)` predicts the next environmental latent; the K=5 ensemble heads each predict a next environmental latent; the decoder reconstructs the observation. Nothing in the substrate predicts Io's own next internal state. There is no structural locus where self-prediction could *develop* even if Io's training and experience happened to compose toward it.

This is more restrictive than Kind's capacity-over-exercise stance requires. The charter's second success criterion is the "capacity to take its own processing as an object of attention — whether or not it chooses to exercise that capacity." Capacity-over-exercise asks: make the capacity possible, not mandatory, not impossible. The current substrate makes it impossible — only ingredients in nominal isolation. The mirror calibration work in the in-progress Probe 2 synthesis (`docs/decisions/Kind_probe2_synthesis.md`) therefore reads for reflexive attention against a substrate with no architectural place where awareness, if it existed, would be represented. That makes the calibration substantially less honest: the mirror is asked to find a thing the substrate cannot in principle host.

## What Probe 1.5 is for

Probe 1.5 is the minimum substrate addition that **affords** self-modeling without **installing** it. The distinction is load-bearing: an affordance is structural-possibility-of, an installation is structural-implementation-of. Kind's commitments rule out installation (no explicit self-modeling, no self-critic, no introspector, no volitional gate, no recursive self-improvement, no machinery for self-optimization). Probe 1.5 must add the affordance without crossing into installation, without violating self-opacity for Io, without smuggling a self-continuation drive, without changing the actor's intrinsic objective (K=5 ensemble disagreement variance, settled), and without committing to a foundation-model substrate for Io.

Three rough architectural moves are candidates, drawn from different interpretive traditions; each has different implications for opacity, installation-vs-affordance, and mirror-readability.

## Eight questions, scoped

### Q1 — Which affordance

The candidates are: **(a) a self-prediction head** that predicts Io's own next internal state, parallel to the world model's prior over environmental latents (predictive-processing tradition; Friston, Hohwy, Clark); **(b) episodic self-access**, a buffer of past internal states the actor can read in some structured form (autobiographical-memory cognitive science tradition; Conway, Tulving on episodic memory); **(c) higher-order policy**, action selection that operates on a representation of the action distribution itself (Frankfurt's second-order volition framework). Engage with which is most appropriate for Kind's specific commitments and why — not just identify options. The criterion is not "which is most powerful" but "which adds the minimum affordance while preserving self-opacity, ingredients-only stance, no installed self-continuation, and capacity-over-exercise." Each move has different opacity implications: (a) keeps the prediction internal to the world model; (b) requires the actor to read its own past, which compromises opacity unless carefully scoped; (c) installs the higher-order structure rather than affording it.

### Q2 — Self-prediction as the leading candidate

Assume self-prediction is the right minimum affordance (likely but not certain). What does it look like architecturally? Specifically:

- **What is predicted.** Next `z_t`? Next `h_t`? Next action distribution? Next ensemble-disagreement value? Each has different opacity implications and different mirror-readability properties.
- **Supervisory signal.** The actual next state of whichever quantity is predicted? A bootstrapped target (like a target network)? Self-supervised (no explicit target, just consistency)?
- **Where the head sits.** Parallel to the world model's prior over environmental latents (a second prior, this one over Io's own dynamics)? Embedded in the world model itself (a head off the same recurrent backbone)? Separate module with shared encoder?
- **Training shape.** Additional loss term in the ELBO? Auxiliary objective optimized jointly? Self-supervised with no gradient flow into the actor?
- **What is shared with existing components vs what is new.** The GRU recurrence, encoder/decoder, prior network — which can be reused, which must be new?

### Q3 — Self-opacity preservation

Kind's commitment is to self-opacity for Io: Io does not read the mirror's output, the telemetry, the digest, or any of its own internal distributions. The PolicyView/TelemetryView split (Probe 1 synthesis Q5; settled) enforces this in code. The self-prediction head produces a prediction; whether that prediction is part of the actor's input (compromising opacity) or stays internal to the world model and is read only by the mirror (preserving opacity) is the **load-bearing design question** of Probe 1.5. Engage with what each option affords and what each forecloses:

- If the actor reads self-prediction or self-prediction-error as part of its input, what does that allow that pure `concat(h, z)` doesn't? Does it cross into installed self-modeling?
- If the actor doesn't read self-prediction at all, what does the affordance buy that pure ingredients in nominal isolation didn't? Does the structural existence of the prediction matter even if Io has no access to it?

### Q4 — Mirror-side reading

What does the mirror read to see whether the affordance is being used? Candidate signals: self-prediction accuracy trajectories over training; divergence between self-prediction and actual next state under perturbations; correlation between self-prediction patterns and behavioral patterns; structural changes in which latent dimensions the self-prediction allocates capacity to over training. Propose a *reading shape* for self-prediction analogous to the triplet the Probe 2 synthesis bound reflexive attention to. Where the literature delivers concrete signals, name them; where it doesn't, say so.

### Q5 — Relationship to dream state

The dream-state machinery (Probe 3, unbuilt but committed in design notes) runs the world model's prior forward over environmental latents without sensory input. Self-prediction runs a separate head over Io's own latent space. Do they share machinery (the dream-state's recurrent rollout *is* the self-prediction at every step)? Share telemetry (the same `dream_rollout` stream)? Stay structurally separate (self-prediction during waking, dream-state during dreaming)? Engage with how the affordance interacts with Kind's commitment to dreaming as foundational, including the four-state operational model and the variable dream-to-wake ratio coupled to the builder's life.

### Q6 — Relationship to Probe 2's in-progress synthesis

The Probe 2 synthesis bound reflexive attention to a triplet (per-dimension KL, ensemble-disagreement variance, latent-dimension allocation under perturbation-aligned windows), descoped second-order volition, treated equanimity as watch-only perturbation-recovery. With a self-prediction head present, self-prediction accuracy could become a primary signal for reflexive attention, a fourth signal added to the triplet, or the right binding for the criterion entirely (more directly than precision-weighting via KL). Engage with how Probe 1.5 reshapes Probe 2's mirror calibration: which parts of the Probe 2 synthesis would need revision, what new signals become available, what existing signals might be displaced or refined.

### Q7 — Engineering precedents in the relevant traditions

Where has the literature done analogous things, and what's been learned? Self-prediction in RSSM/Dreamer-lineage models; self-supervised auxiliary objectives in deep RL (BYOL, SPR, EfficientZero-style next-latent prediction); metacognitive architectures in deep RL (agents maintaining confidence estimates or predicting their own future actions); predicted-trajectory work in active inference; autobiographical memory in cognitive architectures (Soar, ACT-R; episodic-memory-augmented RL like Neural Episodic Control). Where the literature has done this: (a) when the auxiliary signal is load-bearing for behavior vs inert, (b) whether the predicted-self representation aligns with downstream task structure, (c) how the addition interacts with self-opacity desiderata. Where the literature hasn't, what's the closest precedent.

### Q8 — Failure modes

Three rough categories of how Probe 1.5 could go wrong, each requiring detection strategy:

- **(a) Inert affordance.** The self-prediction head trains but its outputs do not structure anything downstream; the mirror finds no signal because no signal is being produced. Detection: how would the mirror tell "no self-modeling has developed" from "the affordance is dead"? What baseline rules out the latter?
- **(b) Load-bearing in unintended ways.** Self-prediction influences behavior in ways that look like self-modeling but are actually a side effect of the additional supervisory signal sharpening the world model's representations. Detection: how to distinguish "Io is self-modeling" from "Io's world model got a richer representation as a side effect of the auxiliary loss"?
- **(c) Self-opacity slippage.** The actor effectively gains access to information about its own state, even if the explicit interface is preserved. Detection: how to test whether the actor's behavior depends on self-prediction quantities the PolicyView is not supposed to expose?

Engage with each category, not all at the same level of depth — the right level is what the literature supports plus what Kind's stance requires.

## Constraints

- **Not** benchmark performance, sample efficiency, or capability metrics.
- **Not** an alternative substrate for Io. RSSM/Dreamer-lineage with active-inference-shaped actor is settled.
- **No foundation-model substrate for Io.** Mirror is foundation-model-based; Io is not.
- **Engage with what the project documents commit to**, not what the literature thinks Probe 1.5 should be. Capacity-over-exercise, ingredients-only, self-opacity by default, no installed self-continuation drive, no self-optimization machinery, the four-state operational model, the co-design problem — non-negotiable.
- **Do not reify gestures into formal substance.** If the literature delivers a defensible answer, ground in it; if it does not, say so. Earlier syntheses surfaced reification patterns (invented canonical architectures, invented benchmark names, suspiciously crisp operational protocols presented as literature-canonical) — flag where your analysis is constructed rather than literature-canonical.
- **Keep the distinction between Kind and Io.** Kind is the project; Io is the entity. Architectural recommendations describe what Kind builds for Io; do not blur into "Io decides" framings.
- **Surface adjacent traditions explicitly.** Predictive processing / active inference (Friston, Hohwy, Clark, Seth); metacognition in deep RL; philosophy of mind on self-prediction (Hohwy; Metzinger on self-models; Frankfurt on second-order structure); autobiographical memory and episodic-memory-augmented agents (Conway; Tulving; NEC); Buddhist phenomenology on reflexive awareness (svasaṃvedana; Garfield's contestation; Thompson's synthesis).

## Output structure

For each of the eight questions:

1. A short framing of what the question is actually asking, in your own words.
2. Concrete options drawn from the literature, with citations or references where applicable.
3. Tradeoffs against Kind's specific commitments (not generic ML tradeoffs).
4. A defensible suggestion where you have one. Where you don't, say so.

After the eight, a **synthesis section** identifying what an internally coherent Probe 1.5 design looks like — which affordance, what specifically it predicts, where it sits, what trains it, what the mirror reads, how it relates to Probe 2 and Probe 3, what failure modes the design protects against and which it doesn't. Flag tensions between answers (e.g., where mirror-readability favors a design choice that strains self-opacity, or where a literature-supported precedent strains the ingredients-only stance). Flag anything Kind's project documents seem not yet to have addressed that your research surfaced.

Research-gathering, not synthesis-and-implementation. Do not produce a build plan. Produce the inputs from which one can later be drafted.
