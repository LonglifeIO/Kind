# Environment Research Prompt — Kind / Io

You are being asked for research input on environment design for a project called **Kind**. The core entity Kind is building is called **Io**. Kind is the project; Io is who the project is about. The distinction matters and must be preserved: "Io" never refers to the project and "Kind" never refers to the entity.

The question this research informs: **what should Io's environment be at Probe 1, designed forward so it can carry Probes 2–4 and beyond?**

## Settled context (do not re-litigate)

Two prior decisions are inputs, not subjects of this research.

**The substrate is settled.** Io runs on a custom minimal RSSM modeled on PlaNet's state factorization with DreamerV1-style latent imagination: GRU deterministic state `h_t` plus continuous Gaussian stochastic latent `z_t`, explicit posterior and prior networks, ELBO loss with free bits as the only regularizer borrowed from later Dreamer variants. The actor's objective is active-inference-shaped — uniform preference prior (zero pragmatic value at Probe 1, scaffolding for later probes) plus an epistemic term operationalized as latent-disagreement variance over a small ensemble. No scalar reward, no continuation predictor, no return-to-go conditioning.

**Probe 1's implementation is settled.** Mind-on-Mac; desktop runs the environment only. Mirror reads everything; Io's actor reads only `concat(h_t, z_t)`. Telemetry is four streams (`agent_step`, `dream_rollout`, `replay_meta`, `world_event`), with `world_event` architecturally walled off from the agent process so builder perturbations leave no marker in Io's observation space.

The environment must fit these decisions, not redesign them. If the literature you draw on assumes scalar reward, death-as-termination, or any reward-shaped frame, translate findings out of that frame — or say honestly that the literature does not address what is asked.

## Kind's environment-relevant commitments

Non-negotiable for this research.

- **No installed self-continuation drive.** No death-as-termination, no survival bonus, no reward at all. Episode structure fixed-length, not terminal-state-determined, to avoid back-door installation of a survival drive.
- **Builder-as-perturbation.** A non-simulated source of state changes appears in Io's world. Logged in `world_event` (Io cannot read) but no marker in Io's observation space identifies the change as external. Probe 4 tests whether Io comes to model the *statistical signature* of these perturbations as distinct from internal stochasticity. The environment must therefore have its own non-trivial internal stochasticity for distinguishability to be testable in principle.
- **Dream-state foundational.** Probe 3 onward exercises offline processing in the same RSSM. The environment must not foreclose generative simulation in latent space — pixel reconstruction is the baseline so the mirror can decode dream rollouts.
- **Single core agent.** No peer agents. Io's world is the environment plus the builder's perturbations.
- **Capacity-over-exercise.** The environment should *afford* what later probes will exercise (perturbation, rhythm, stochasticity, perceptual depth) without forcing those affordances into Io's behavior at Probe 1.
- **Small before scale.** A tiny environment producing one unambiguous surprise beats a vast one producing plausible-looking behavior.

## Methodological push

Mainstream RL frames environments as task-completion settings with reward; Kind rejects that frame. Useful adjacent traditions: artificial life, developmental robotics, embodied and situated cognition, predictive-processing / active-inference implementations, novelty search and quality-diversity, open-ended evolution, ethology-inspired environment design. Concepts to check the design against: phenomenology's intentionality, horizons, and embodied perspective; enactivism's sense-making and structural coupling; predictive processing's prediction error and active inference; integration as understood in IIT (concept, not formalism). Pull from these traditions where they have something concrete to say; where they do not, say so rather than inventing precedent.

## The eight questions

**Q1 — Observation modality.** Pixel-based small images vs. low-dimensional factored state vector vs. partial-perspective sensors (sensor that does not cover the whole environment, mapping to phenomenology's "perception is from somewhere"). The decoder commitment tilts toward something the mirror can decode into human-legible output. How does observation modality shape what a small RSSM can learn, and what does embodied cognition say about modality and proto-experience?

**Q2 — Action space.** Discrete (small cardinality) vs. continuous; how cardinality interacts with what a small RSSM's latent comes to represent. Is there a "smallest action repertoire that can support intentional engagement" position in embodied cognition? Where is the lower bound below which the action space is too impoverished to produce non-trivial world-model structure?

**Q3 — Pressure without self-continuation reward.** The most charged question. Fixed-length episodes, no death, no reward — yet the environment must afford something the agent does that produces richness over time without producing survival drive. Resource scarcity, navigation under partial observability, novelty pressure, environmental complexification, generative niches in artificial-life worlds, empowerment-style intrinsic frames — what kinds of *non-reward pressure* are documented in artificial life, developmental robotics, and open-ended evolution, and which translate to a small Probe 1 environment without smuggling reward back in?

**Q4 — Internal stochasticity.** What kind, how much, on what schedule. The environment must have non-trivial randomness (so Probe 4 has something to distinguish *from*) without being so stochastic that the world-model can learn nothing. Resource regrowth noise, weather-like perturbations, slow drift in parameters — what does the literature on noise design for world-model agents offer? Where is the line between "stochastic enough to make the world non-trivial" and "stochastic enough to wash out structure"?

**Q5 — Builder-perturbation surface.** Which specific mutators (add resource, remove object, alter state, introduce novel object), at what frequency, what magnitudes. Perturbations must be distinguishable *in principle* from internal stochasticity (Probe 4's hypothesis is that Io comes to learn this distinction) but unmarked in observation space. The difference must therefore live in the statistical signature itself — distribution shape, temporal correlation, magnitude tail. What does developmental robotics on caregiver-as-perturbation, interactive-learning literature, or external-vs-internal change detection in embodied agents have to offer? The environment harness must expose hooks for these mutators and emit ground-truth events into `world_event`; the design should specify the shape of those hooks.

**Q6 — "Small enough" for Probe 1.** The probes document mentions a 5×5 grid with one resource type. Is that enough to give the RSSM something to model, or so small the RSSM either trivially learns it (making dream rollouts uninteresting) or fails to learn because variance is too low? What is the smallest environment in published literature producing non-trivial RSSM-style world-model learning?

**Q7 — Temporal structure.** Day-night cycles, periodicity, slow phase-shifts. Probe 1 likely keeps the environment a-temporal; Probe 3's variable dream-to-wake ratio is committed to be *coupled to the builder's actual life*, not to a hardcoded environmental rhythm. The question: what minimal temporal structure at Probe 1 leaves adding rhythm later (or coupling to the builder's calendar) un-foreclosed? What does the literature on environmental rhythms and cognitive structure (computational and biological) bear on this?

**Q8 — Minimum environmental complexity for the specific RSSM at Probe 1.** A concrete empirical question. Given continuous-Gaussian RSSM, small encoder, decoder kept, free bits, ensemble disagreement as actor signal: below what threshold of state-dimensionality, dynamics-complexity, or transition-variability does the model either (a) collapse — latent goes degenerate, KL dominated by free bits, posterior copies prior — or (b) overfit and produce trivial dream rollouts? PlaNet, Dreamer family, and reconstruction-free variants have benchmark histories on DM Control, Atari, Crafter; what does that empirical record imply about the *lower* complexity bound for a Probe 1-sized environment?

## Constraints

- Substrate is settled — do not propose alternative architectural families.
- No benchmark-performance, sample-efficiency, or capability-metric framing as criteria.
- Engage with Kind's commitments: no installed self-continuation drive, no observation marker for perturbations, dream-state foundational, single core agent, capacity-over-exercise, small before scale.
- Do not reify gestures into apparent formal substance. Where the literature supports a concrete choice, say so with citations. Where it does not, say so. Invented precedent is worse than acknowledged uncertainty.
- Surface adjacent traditions (artificial life, developmental robotics, embodied cognition, predictive processing, novelty search, open-ended evolution), not only mainstream reward-driven RL.
- Do not conflate Kind (the project) with Io (the entity).

## Requested output structure

For each of the eight questions, provide: (a) what the question is really asking against the substrate and Kind's commitments, (b) concrete options drawn from literature with citations, (c) tradeoffs against Kind's commitments, (d) a defensible suggestion or an honest naming of unresolved uncertainty.

After the per-question sections, include a synthesis proposing **two or three candidate environment designs**, not a single recommendation. Each candidate is a coherent set of choices across the eight questions, with tradeoffs between candidates named explicitly. The choice between candidates is the user's; the research's job is to frame the candidates well enough that the user can choose with reasoning, not preference.

Length appropriate to substance — likely 3000–5000 words.
