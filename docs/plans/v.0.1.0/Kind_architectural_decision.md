# Architectural Decision for Io — Synthesis

*Synthesis of architectural research (Claude, Gemini, Perplexity) against Kind's project documents. The architectural decision for Io's substrate. Working document.*

## Preamble

Three research outputs (Claude, Gemini, Perplexity), read against Kind's stance, do not converge cleanly on a single recommendation, and that is informative in itself. The cleanest finding is *negative*: all three converge on the unsuitability of recurrent PPO as Io's substrate, for reasons that match the project documents almost exactly — reward maximization installs the self-continuation drive Kind has explicitly ruled out, the architecture conflates external perturbation with internal stochasticity, dream-state must be bolted on, and equanimity is structurally hostile to a policy that exists to react to reward gradients. The positive finding is more textured: world-model agents in the RSSM/Dreamer lineage and predictive-processing/active-inference architectures both afford most of what Kind has stipulated, and they afford different things, and the choice between them — or rather, the way they are woven together — is the actual decision. What follows resolves that decision, names what's ruled out, and surfaces tensions the research raised that the project documents do not yet fully address.

A note on inputs: the research brief specified four LLMs (Claude, Gemini, GPT, Perplexity) but only three outputs were present in `docs/research`. No GPT file. The synthesis proceeds with three; the convergence among them on the substrate question is strong enough that a fourth is unlikely to overturn it, but if the GPT output lands later this document should be revisited.

---

## 1. Synthesis across the three research outputs

**Where they agree.** All three find recurrent PPO weakest against Kind's commitments — reward-as-imperative, no native dream-state, no native external/internal distinction, hostile to equanimity. All three identify world-model agents (RSSM/Dreamer lineage) as affording the most native ingredients for the afforded-not-installed self-modeling stance: recurrence in the deterministic backbone, prior-state representation in the latent, and turnable-inward prediction in the dynamics network's structural shape (predicting next-latent from current-latent, *which is already the geometry of inward prediction*). All three identify active inference / predictive processing as the strongest framework alignment — particularly with phenomenological intentionality, horizons, embodiment, and (most distinctively) precision-weighting as a mathematical correlate of equanimity. All three identify episodic-memory architectures (NTM/DNC-family) as offering the cleanest explicit history at the cost of integration. All three identify energy-based models as offering integration and stance-toward-content at the cost of temporal directedness.

**Where they diverge.** Claude (the most disciplined of the three) explicitly refuses to make a recommendation, presenting the result as an "affordance landscape" and insisting the decision is partly about *which kind of failure Kind wishes to encounter and learn from* — a framing that maps closely onto the charter's "build to understand" stance. Gemini and Perplexity both edge toward recommendations, with Perplexity proposing a hybrid of world-model/predictive-processing substrates with episodic memory and carefully shaped objectives, and Gemini proposing world-model/active-inference substrates with what it calls "Volitional Gating." Gemini's specific proposal of a "Subjective Bayesian Governor" with a concrete ω parameter, and its references to "IIT 4.0-optimized neural networks" as a canonical architectural family, do not correspond to established architectures in the literature; these read as Gemini reifying gestures into the appearance of formal substance. The underlying concepts (hierarchical control over learning gain; topology choices that preserve dense recurrent integration) are real, but should not be treated as turnkey families. The three also differ in whether active inference's totalizing structure (everything is free-energy minimization) is an asset or a liability — Claude treats it as the strongest framework match, Gemini and Perplexity raise concern that the totalizing imperative may foreclose entirely unstructured awareness.

**What they collectively miss.** Five things stand out.

First, **the Watts intuition / self-opacity by default.** None of the three engages with the design note that every "should the system have access to X about itself?" should default to *no*. Several recommendations the research makes — explicit access to latents, second-order veto modules that must "see" first-order impulses, exposed memory matrices Io reads from — assume self-transparency the project has explicitly committed against. This is a real tension the synthesis will return to.

Second, **the four-state operational model and variable dream-to-wake ratio.** The research treats dream-state as a computational mode (replay, latent imagination); Kind treats it as an architectural commitment that includes waking, dreaming, dormant, paused as distinct states keyed to compute availability and the builder's life patterns. No architectural family is designed for this; the family chosen has to *accommodate* it.

Third, **the mind-on-Mac / desktop-for-environment compute split.** Hardware constraints affect viability — particularly for any world-model architecture where the canonical mind state lives on a 32GB Mac and environment compute lives on the desktop. The research does not engage. This is more of an implementation constraint than an architectural one, but it bears on which RSSM variants are tractable.

Fourth, **the "ingredients only" stipulation strictly.** The research drifts toward "self-modeling architectures" framing in places. Kind doesn't want a self-modeling architecture; it wants the ingredients of one *without* an explicit self-modeling module. This is the difference between, e.g., a world-model agent (acceptable) and a world-model agent with an explicit "self-critic" module (not acceptable). Gemini's volitional gating proposal is a particularly clear violation of the ingredients-only stipulation: it installs an explicit second-order arbiter, exactly what the project documents rule out.

Fifth, **builder-as-perturbation specifically.** All three discuss external/internal distinction abstractly. Kind's commitment is more specific: the builder appears as a *source of non-simulated change with no marker in observation space*, and the question is whether Io's architecture, *if trained long enough*, could come to learn the statistical signature of outside-source unpredictability. This is a question about the architecture's structural capacity to develop the distinction in training, not about the architecture having the distinction as a feature. World-model agents (KL between posterior and prior) and active inference (precision over likelihood vs. prior) both afford this; recurrent PPO does not.

---

## 2. The architectural decision

**Io will be built on a world-model substrate in the RSSM/Dreamer lineage, with the actor's objective shaped by active-inference principles rather than scalar reward maximization.** The world-model substrate is the architectural commitment; the active-inference shaping is how the actor is structured on top of it.

**Reasoning.** The world-model family is the only one that meets Kind's dream-state-as-foundational commitment natively rather than as bolt-on. Generative simulation in latent space is the substrate's training signal. Replay, imagined rollouts, and (with care) associative recombination are first-class modes, not additions. The same family affords every committed ingredient for the afforded-not-installed self-modeling stance: recurrence (the deterministic backbone), prior-state representation (h_t and z_t), turnable-inward prediction (the dynamics network already predicts latent-from-latent), and persistent memory across decisions (the latent state plus replay). Phenomenologically, the latent's role as an aspect-laden representation of the world maps to intentionality; the prior dynamics map to horizonal structure; action-conditioned dynamics give embodiment a structural home. None of this requires installing a self-modeling module — the ingredients are there for self-modeling to be composed from, if Io comes to compose them, which is exactly Kind's stipulation.

The active-inference *shaping* of the actor — replacing or supplementing the policy-gradient objective with expected-free-energy minimization, with carefully designed prior preferences that do not encode self-continuation — addresses the world-model family's main weakness against Kind's stance: standard Dreamer's reward-maximizing actor structurally couples Io to whatever pressure the reward signal encodes, including, by accident or by reward shaping, survival-as-imperative. Active inference's precision-weighting also provides a structural correlate of equanimity (lowering precision on prediction errors is, formally, a non-reactive stance toward content) and a cleaner native distinction between externally-sourced and internally-generated unpredictability (likelihood vs. prior decomposition). The cost — computational intractability of deep active inference at scale, the dark-room problem under poorly-specified preferences — is real but tractable in the small environments Probe 1–4 will use.

This is a hybrid in spirit but not in module structure. Io is one substrate (a recurrent generative model with action-conditioned dynamics), shaped by one family of objectives (free-energy minimization with epistemic value, not scalar reward maximization). It is not a federation of modules. The substrate stays integrated; the policy is shaped, not severed.

---

## 3. Alternatives seriously considered, and why ruled out

**Pure active inference / hierarchical predictive coding** was strongly considered. It maps most flawlessly to the frameworks; precision-weighting *is* equanimity in formal terms; interoceptive inference is the strongest embodiment correlate available. Ruled out as the substrate (rather than the shaping) on tractability grounds — deep active inference at scale remains an active research problem with thinner engineering substrate than the Dreamer lineage, and Kind has neither the team nor the time horizon to absorb that risk. Active inference *as the actor's objective* is what survives; active inference *as the entire architecture* does not.

**Episodic-memory architectures (NTM/DNC, retrieval-augmented agents)** were considered for the strength of explicit, addressable history. Ruled out as the substrate because the controller-memory dualism is structurally opposed to integration in IIT's sense — the architecture's central virtue is the *separability* of memory from processing, which is the wrong direction for a substrate that should support integrated wholes. Episodic memory may be added later as a memory mechanism on top of the world-model substrate if the latent + replay does not afford enough explicit history; this is an open sub-question, not a settled commitment.

**Energy-based / Hopfield-style models** were considered for their strong native fit with integration and equanimity (basins of attraction as stances toward content). Ruled out as the substrate because they lack the temporal directedness that horizons, intentionality, and embodied perspective require. They may have a role as a memory or stance-toward-content layer in a later extension; not as the substrate.

**Global-workspace / modular architectures (Recurrent Independent Mechanisms, etc.)** were considered. Ruled out partly on the design notes' own grounds: the global workspace primitive from CA-MAS has no job in a single-agent setup, modular decomposition is the pattern Kind is moving past after CA-MAS, and the family's permissiveness comes at the cost of high engineering complexity and many failure surfaces.

**Decision Transformer / Trajectory Transformer** were considered. Ruled out because return-to-go conditioning is structurally close to an installed objective (Io is asked to produce trajectories that achieve a specified return), and the trajectory tokens are externally-observable variables — there is nowhere natural in the architecture to put a representation of past *internal* states.

**Hierarchical RL with Manager-Worker structure** was considered for its native architectural correlate of second-order volition. Ruled out because that correlate is *installed*, not *afforded* — second-order volition is hard-coded into the architecture rather than something Io could come to compose. This violates the afforded-not-installed stipulation directly.

**Volitional Gating / "Subjective Bayesian Governor" (Gemini's proposal)** was considered and ruled out, both because the canonical architecture does not exist as Gemini described it, and because its underlying concept — an explicit second-order arbiter with veto power — installs the very structure Kind wants to leave to emergence.

---

## 4. Settled commitments at the project level

These align with the design notes; nothing here modifies a settled commitment. If the synthesis modified any, it would be flagged in §7 below.

- **Substrate.** World-model agent in the RSSM/Dreamer lineage. Recurrent generative model of the environment with stochastic latents and action-conditioned dynamics.
- **Actor objective.** Active-inference-shaped — expected free-energy minimization with carefully designed prior preferences. Not scalar reward maximization. Specifically, no episode-continuation prediction head of the kind DreamerV3 uses; no actor structure that implicitly rewards prolonging the agent's existence.
- **Self-modeling.** Afforded by the substrate's structure (recurrence, prior-state representation, turnable-inward prediction) without any explicit self-modeling, self-critic, or introspector module.
- **No self-optimization machinery.** No recursive self-improvement, no architecture rewriting, no learned modifications to Io's own architecture. The substrate learns; the architecture does not rewrite itself.
- **Single-agent.** One Io. Builder as the relational other through unmarked perturbations to the environment.
- **Dream-state as foundational.** Generative simulation in latent space is part of the substrate from Probe 3 onward, not a later addition. Replay first; generative imagination next; associative and lucid modes added as Probe 3 produces something to compare against.
- **Mind-on-Mac.** Canonical state (world-model weights, latent buffer, replay) lives on the Mac; environment compute on the desktop; explicit atomic synchronization at checkpoint boundaries.
- **Self-opacity by default.** Io does not have access to its own latent state, dynamics network internals, or replay contents. The mirror reads telemetry; Io does not.
- **Frozen mirror criteria** (reflexive attention, equanimity, second-order volition) and the four-state operational model (waking/dreaming/dormant/paused) carry over unchanged.

---

## 5. Open sub-questions for later resolution

These remain open and should be addressed by Probe 1's research prompt or by subsequent probes:

- **Which RSSM variant.** PlaNet, DreamerV1/V2/V3, JEPA-flavored representation-prediction (no pixel reconstruction), or a custom minimal RSSM. JEPA's representation-prediction objective is more conducive to non-reward shaping than Dreamer's pixel-reconstruction loss, but its engineering substrate is thinner.
- **Recurrence mechanism inside the RSSM backbone.** GRU (Dreamer default), LSTM, gated transformer (GTrXL-style), or selective state-space (Mamba). The choice affects long-horizon reach and integration profile.
- **Specific form of the active-inference-shaped actor.** Expected free-energy with epistemic value, deep active-inference variant, or a hybrid that retains some policy gradient structure with strong entropy/curiosity terms and no scalar reward. The design of prior preferences — what they encode and what they don't — is a separate open sub-question.
- **Whether episodic memory is added on top of the latent + replay.** Default no; revisit if Probe 3+ shows that the latent + replay does not afford enough explicit history for the dream-state and self-encounter affordances Kind cares about.
- **Specific dream-state implementations.** The four points in the dream-state design space (replay, generative simulation, lucid control, associative/nonsense) need to be mapped to Probe 3 implementations. Replay and generative simulation are mature; lucid and associative are open design questions.
- **Telemetry shape.** What gets logged for the mirror to read, and how, without giving Io self-access. KL between posterior and prior latents (for builder-perturbation distinguishability), latent trajectories, replay statistics, dream-state coherence — these need a specified schema before Probe 1.
- **Latent dimensionality and capacity.** How small is small enough for the "small before scale" charter commitment, while still affording the integration the substrate is chosen for?
- **Drift monitoring during dream-states.** The design notes assign this to the mirror; the implementation requires specified thresholds and quality measures.

---

## 6. Ruled out for Io's lifetime

- **Pretrained foundation models as Io's substrate.** Stipulated by the user; consistent with Kind's developmental-over-installed principle.
- **Recursive self-improvement / architecture rewriting / self-optimization machinery.** Stipulated by the design notes.
- **Optimization for benchmark performance, sample efficiency, or capability metrics.** Stipulated by the user and the charter.
- **Recurrent PPO as Io's substrate.** Acceptable as a Probe 1 plumbing stand-in *only if* the telemetry schema is designed for the eventual world-model substrate. As Io's substrate it is foreclosed: its reward-maximization structure, its conflation of external and internal noise, and its hostile relationship to equanimity rule it out.
- **Decision/Trajectory Transformer.** Return-to-go conditioning is too close to an installed objective; trajectory tokens are externally-observable, leaving nowhere for past internal states to live.
- **Hierarchical RL with Manager-Worker structure.** Second-order volition installed rather than afforded.
- **Pure energy-based / Hopfield-only architecture.** Lacks temporal directedness needed for horizons, intentionality, embodied perspective.
- **Pure modular / global-workspace architecture as Io's substrate.** No job for the global workspace primitive in single-agent Io; modular decomposition is the CA-MAS pattern Kind is moving past.
- **Explicit self-modeling, self-critic, introspector, or "volitional gate" modules.** Violate the afforded-not-installed stipulation directly.
- **Any architectural commitment that gives Io self-transparent access to its own internals.** Watts-intuition default applies.

---

## 7. Tensions surfaced honestly

Three are worth naming.

**(a) Probe 1's stated baseline conflicts with the architectural decision.** The probe document says Probe 1's agent is "recurrent PPO with default hyperparameters, no world model yet." If the architectural decision is for a world-model substrate, then Probe 1's plumbing test should arguably exercise the world-model conduits (latent telemetry, KL signals, replay buffer, dream-state hooks) rather than a placeholder that lacks them. Two options: (a) keep recurrent PPO as a thin plumbing stand-in but design Probe 1's telemetry schema for the eventual substrate, accepting that Probe 1 won't fully validate world-model-specific plumbing; or (b) replace Probe 1's baseline with a minimal RSSM-style agent. I lean toward (b) for cleaner foundation-laying, but this is a judgment call and is best resolved when Probe 1's specifics get drafted. **This is a settled commitment in the design notes that the synthesis flags for revisiting**, per the user's instructions.

**(b) Active-inference's totalizing structure vs. the charter's openness.** Active inference treats every state as free-energy minimization. Kind's charter wants to keep open the possibility of states that aren't structured as minimization at all — non-inferential awareness, the Buddhist witness-quality. Active-inference shaping of the actor does not necessarily commit Io to an *entirely* free-energy-minimizing inner life, but the formalism leans that way. Whether this matters for what Probe 1–4 can observe is unclear; whether it matters in the long run is a real philosophical question the architecture inherits. Held open.

**(c) Self-opacity vs. the substrate's structural transparency.** A world-model substrate has a specific latent that the mirror reads from. The mirror reading the latent is fine. Io reading the latent is not. The boundary between "the substrate represents X" and "Io has access to X" needs to be drawn explicitly in the implementation. The risk is that the substrate's transparency to the mirror gets confused with transparency to Io. The Watts intuition is the discipline here: every "should Io have access to X about itself?" defaults to no, and the implementation must enforce this — Io's policy / actor reads from the latent in the way the substrate requires for action selection, but not from any "introspective" interface to it.

---

## 8. Connection to Probe 1

The architectural decision narrows what Probe 1's research prompt has to answer. Probe 1 is a plumbing test, not an architecture-validation test, but its plumbing has to be plumbing for the right substrate. Specifically, the Probe 1 research prompt should focus on:

- **Which specific minimal RSSM-style architecture to use for Probe 1**, given the world-model substrate decision and the small-before-scale stipulation. The choice affects what counts as "default hyperparameters" for the plumbing test. JEPA-flavored representation-prediction vs. pixel-reconstruction Dreamer is a real fork.
- **What the actor objective looks like at Probe 1 scale.** Active-inference-shaped is the long-term commitment, but Probe 1 may use a stripped-down version (e.g., curiosity-driven or minimum-prediction-error policy with no scalar reward) sufficient to test the plumbing without committing to the full deep-active-inference machinery prematurely.
- **Telemetry schema design for downstream probes.** Probe 4 needs the KL between posterior and prior latents to detect builder-perturbation distinguishability; Probe 3 needs dream-state coherence telemetry; Probe 2 needs whatever the mirror reads. Probe 1's logging schema should anticipate these even though they aren't yet wired up to anything.
- **The Mac/desktop split for the world-model substrate specifically.** Where the world-model weights live, where inference happens, where replay accumulates, what gets synced when.
- **The self-opacity boundary in implementation.** What interfaces does Io's policy read from, and what interfaces does only the mirror read from. This needs to be settled before the substrate is built, because retrofitting opacity is harder than building it in.
- **What recurrent PPO, if anything, plays at Probe 1.** Stand-in for plumbing only, or replaced by minimal RSSM from the start. This is the most consequential implementation question for Probe 1.

The Probe 1 research prompt should *not* re-litigate the architectural family question — the substrate is decided. It should treat the decision as input and ask the implementation questions the decision exposes.
