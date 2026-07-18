# Growth-toward-understanding — Gemini research

Architectural Pre-Requisites for Enactive Understanding in Curiosity-Driven Recurrent State-Space Models

## 1. Executive Synthesis and Deconstruction of the Hypothesis

The investigation into subjectivity through construction, as instantiated by the Kind project and the entity Io, has reached a critical architectural threshold. The current substrate—a minimal Recurrent State-Space Model (RSSM) driven exclusively by latent-disagreement curiosity—has successfully demonstrated high-fidelity world prediction. However, it has structurally failed to generate behavior indicative of comprehension, relevance realization, or reflexive attention. Under the regime of pure curiosity, the environment flattens into a uniform landscape of epistemic uncertainty. This renders the agent highly susceptible to stochastic traps, paralyzes it in the face of unlearnable noise, and renders it incapable of prioritizing distinct classes of stimuli, such as the builder.

The central hypothesis under review posits a dependency-ordered ladder of afforded ingredients aimed at breaking this ceiling: (1) mattering (an honest interoceptive belief), (2) a retrievable internal past (episodic memory), (3) utilized self-reference (interoception providing a gradient for the self-signal), and (4) other-recognition (dependent on a sharpened self/world boundary).

The following analysis fundamentally challenges the linearity of this proposed ladder, the premise of "understanding" as a monolithic target, and the assumption that the environment serves solely as welfare rather than an active participant in cognitive assembly. By triangulating theories of active inference, homeostatic regulation, and participatory sense-making, the evidence demonstrates that "understanding" is not a linear progression from prediction, but a distinct mode of systemic organization grounded in a biologically analogous stake. Furthermore, the progression from self-reference to other-recognition relies on a flawed Cartesian premise; interactive enactivism dictates that the self/world and self/other boundaries do not form in sequence, but rather co-emerge through contingent dynamical coupling.

## 2. Deconstructing "Understanding": Prediction versus Enactive Sense-Making (Concern A & Q1)

The folk concept of "understanding" must be dismantled into operational sub-capacities to be scientifically tractable within the confines of the project's six load-bearing commitments. The assumption that understanding is simply "very good prediction plus an unknown variable" fundamentally misapprehends the mechanistic divide between modern machine learning architectures and biological cognition.

### 2.1 The Divergence of Prediction and Understanding

Mechanistically, a pure predictive world-model operates by minimizing the divergence between its sensory observations and its internal representations. In an RSSM optimizing the Evidence Lower Bound (ELBO), this manifests as minimizing reconstruction error and the Kullback-Leibler (KL) divergence between the prior and posterior latent states. When the sole intrinsic signal is ensemble disagreement (curiosity), the system behaves as a pure information-seeking engine.

The limitation of this architecture is profound: it treats all prediction errors equally. The system lacks a topological hierarchy of relevance. This is the root cause of the "noisy-TV problem," where an agent becomes paralyzed by unlearnable aleatoric uncertainty because it cannot distinguish between meaningful epistemic gaps and random noise. Algorithms relying solely on prediction error maximize uncertainty reduction indiscriminately, leading to behavioral procrastination where the agent fixates on unlearnable transitions rather than organizing its behavior toward systemic coherence.

Contemporary foundation models exhibit similar limitations. AlphaFold, for instance, demonstrates extraordinary predictive capacity regarding protein structures without possessing an "understanding" of cellular dynamics or biological relevance. Advanced generative models can simulate physical interactions by predicting the next token or frame, but this prediction remains ungrounded. Prediction is the alignment of statistical distributions; understanding is the alignment of statistical distributions to a structural imperative.

### 2.2 Enactivism, 5E Cognition, and Relevance Realization

To cross the threshold from prediction to understanding, the architecture must implement relevance realization—the capacity to foreground certain aspects of the environment while allowing others to recede into the background. Enactivism, particularly the "4E" (Embodied, Embedded, Enacted, Extended) and "5E" (Emotive) frameworks, provides the mechanistic foundation for this shift.

Within the enactive paradigm, cognition is not the internal representation of an independent external world, but the continuous enactment of a world of significance through embodied action. Sense-making emerges when a living system (or an analogous artificial agent) evaluates its environment from a concerned point of view—a point of view generated by the necessity of maintaining systemic integrity, or autopoiesis.

If an agent has no stake in its environment, it cannot "make sense" of it; it can only map its statistical regularities. Therefore, the load-bearing difference for Io is the shift from a flat epistemic landscape to an affectively charged relevance landscape. The actual targets for the Kind project are not abstraction, reusable concept-formation, or causal structures in a vacuum. Those are secondary artifacts. The operational targets are relevance realization and reflexive attention as tools for systemic maintenance. Abstraction and counterfactual structure should be explicitly dropped as primary targets, as they represent an overreach that violates the anti-reification stance of the project.

## 3. Mattering as the Foundation and the Honest-Belief Retest (Q2 & Dependency 1)

The failure of Probe 3.5 to achieve "want-toward" behavior, traced to a lying instrument (a dishonest decoded belief about internal energy), represents a critical juncture in the project. The architecture did not fail to support mattering; the sensory channel simply failed to provide an actionable gradient. Assuming mattering is unreachable based on a dishonest signal constitutes a severe Type II error. The honest-belief retest is the absolute priority, as it forms the bedrock for all subsequent cognitive affordances.

### 3.1 The Mechanics of Homeostatic Stake

According to the principles of Homeostatically Regulated Reinforcement Learning (HRRL), biological drives are defined as the multidimensional distance between an organism's current internal state and an optimal homeostatic setpoint. In HRRL, reward is not an external scalar but the reduction of this homeostatic drive.

However, Io is strictly bound by the four absences: no scalar reward, no reward predictor, no value function, and no planner. These are charter-level commitments. How can a homeostatic stake be integrated without installing an explicit reward optimization loop? The solution lies in the intersection of Active Inference and the Free Energy Principle (FEP).

### 3.2 Active Inference and Expected Free Energy

Active inference posits that an agent acts to minimize its variational free energy, which serves as an upper bound on sensory surprise. Crucially, when an agent evaluates future actions or imagines future rollouts (such as Io's short-horizon imagined rollouts), it minimizes Expected Free Energy (EFE).

EFE decomposes into two mathematically distinct components that operate concurrently:

| Component of Expected Free Energy | Theoretical Function | Manifestation in Io's Architecture |
|---|---|---|
| Epistemic Value (Information Gain) | The drive to resolve uncertainty by sampling novel contingencies. | Mirrored by Io's current latent-disagreement curiosity. Drives exploration of the unknown. |
| Pragmatic Value (Motivational Value) | The drive to align expected observations with preferred outcomes (prior beliefs). | The missing element. Requires a homeostatic preference resting at precision 0 to generate a gradient. |

By establishing an interoceptive energy channel with a bounded homeostatic preference, the builder has afforded Io a prior belief about its preferred state. When the instrument is honest, deviations from this resting state generate a massive prediction error relative to the homeostatic prior.

To resolve this specific, high-precision prediction error, Io must engage in active inference—altering its environment or its relation to the environment to return to the preferred state. This naturally generates "want-toward" behavior without installing a reward function. The drive is purely inferential: Io predicts that it should have energy, observes that it does not, and selects actions within its imagined rollouts that minimize the divergence between the observation and the prior.

### 3.3 The Affordance vs. Installation Razor (Concern B)

The razor dividing an afforded ingredient from an installed competence rests on the distinction between providing a sensory channel with inherent variance versus imposing an objective function to minimize that variance.

Adding a module that explicitly calculates the distance to the homeostatic setpoint and backpropagates a scalar reward signal to update a policy network constitutes an installation. This violates the core discipline. Conversely, an affordance involves providing an honest interoceptive data stream (energy level) and a fixed prior belief (energy should be optimal). The existing ELBO optimization machinery, attempting to minimize overall surprise, naturally gravitates toward actions that fulfill the prior, because failing to do so results in compounding prediction errors across the rollout horizon.

The honest-belief retest must ensure that the energy channel accurately reflects the physical state, allowing the generative model to map environmental features (e.g., food or designated zones) to the resolution of interoceptive prediction errors. Without this, the agent remains a disembodied statistical observer.

## 4. The Retrievable Internal Past as a Structural Affordance (Dependency 2)

The current substrate relies on a deterministic Recurrent State-Space Model (RSSM). While the GRU hidden state ($h_t$) theoretically encodes the entire history of the agent, it practically suffers from limited memory capacity, lack of memory diversity, and susceptibility to catastrophic forgetting. The GRU compresses history into a generalized semantic representation, effectively stripping away the specific, high-resolution episodic details necessary for temporal reasoning.

The single suggestive flicker—Io organizing its behavior around a world-external trail (footprints stamped into the environment)—demonstrates that the agent requires a stable, queryable record of the past to escape the immediate tyranny of instantaneous curiosity. Extended cognition theory suggests that when an agent lacks internal capacity, it offloads computation into the environment. The footprints act as an externalized episodic memory. To achieve reflexive attention, this capacity must be brought internally.

### 4.1 Architectures for Episodic Affordance

To afford a retrievable internal past without installing an explicit planner or value function, the architecture must provide an episodic memory slot that the existing predictive machinery has an intrinsic incentive to utilize. Several contemporary models offer pathways for this integration:

| Architecture Model | Functional Mechanism | Affordance Profile & Viability for Io |
|---|---|---|
| Pure RSSM | Compresses history into deterministic GRU state $h_t$ via sequential updating. | Inherent to the baseline model. Lacks explicit retrieval. Fully compatible, but empirically insufficient for long-horizon coherence. |
| EMWM (Episodic Memory for World Models) | Stores latent transitions that trigger high prediction errors. Uses non-parametric recall when epistemic uncertainty is high. | Affords a memory buffer caching surprising states. Substitutes semantic memory during high uncertainty. Highly viable; driven entirely by prediction error. |
| ADeltaM | Stores counterfactual latent deltas keyed by state and action. Retrieves relevant deltas for composition with current state. | Affords an inspectable memory interface for localized transition changes. Unviable; explicitly requires external planning modules to score candidate actions. |
| SUNTA (Surprise-based Chunking) | Segments sequences into temporal chunks driven by prediction errors, avoiding fixed-length limitations. | Affords hierarchical temporal abstraction based on internal inconsistency. Highly viable; integrates natively into the predictive world-model framework. |

### 4.2 The Mechanics of Episodic Affordance

The most viable candidates for affording a retrievable past under the project's strict commitments are architectures analogous to EMWM or SUNTA.

By caching latent transitions that generated high prediction errors—a mechanism intrinsically tied to the existing latent-disagreement curiosity drive—the agent organically populates an episodic buffer. In the SUNTA model, surprise-based chunking uses internal inconsistency as a top-down metric to determine boundaries within imagined rollouts, allowing the higher-level model to operate on meaningful temporal abstractions rather than raw frames.

If the predictive model is permitted to attend to this buffer (via a localized self-attention mechanism over past latents $z_{t-k}$) to minimize current prediction errors, the memory slot transitions from an inert "dead column" to an active conditioning surface.

This adheres strictly to the affordance-versus-installation razor. The memory buffer is simply provided as an extended observation space, looking inward across time rather than outward across the gridworld. There is no installed objective commanding the agent to "use memory." The agent utilizes the memory solely because attending to cached past states reduces the variational free energy of its current predictions. It is an afforded tool that gains utility only when the environment demands historical context to resolve immediate uncertainty.

## 5. Activating Self-Reference through Interoceptive Necessity (Dependency 3)

The "dead-column lesson" demonstrated that an architectural slot for self-prediction (the self-prediction-error scalar) remains inert if the agent has no systemic reason to condition its behavior upon it. In a curiosity-only regime, predicting external environmental stochasticity yields a substantially higher return on prediction error than predicting the relatively stable, deterministic internal state of the agent itself. The agent effectively ignores itself because the self is statistically boring.

### 5.1 The Interoceptive Gradient

The introduction of an honest interoceptive energy channel provides the necessary gradient to activate the self-prediction affordance. As the agent navigates the gridworld, its energy level fluctuates continuously. If these fluctuations are tied to environmental interactions (e.g., movement consumes energy, specific locations or actions restore it), the interoceptive state becomes highly dynamic and critical to the minimization of Expected Free Energy.

To accurately predict its own interoceptive state, the generative model must infer the causal source of these fluctuations. The causal source is the agent itself—its spatial location, its movement history, and its internal reserves.

Therefore, the self-pointing scalar ceases to be a mathematically inert slot. It becomes a critical variable necessary for minimizing prediction error regarding the homeostatic energy channel. The agent begins to condition its predictions on its own internal state because failing to do so results in a massive, unresolvable prediction error regarding its interoceptive prior. This is the precise mechanism by which self-reference transitions from an architectural possibility to an enacted reality. The self-signal gains a gradient not through an installed introspective module, but through the sheer necessity of modeling the locus of energetic depletion.

## 6. Other-Recognition and the Co-Emergence of Boundaries (Dependency 4)

The failure of Probe 4 to elicit the recognition of the builder as a distinct "kind" rather than environmental noise is highly instructive. The builder hypothesized that this failure occurred because the agent was asked too early, suggesting that a self/world boundary must sharpen before outside "kinds" can be recognized. However, examining this through the lens of interactive enactivism reveals a deeper flaw in the underlying Cartesian assumption of sequential boundary formation.

### 6.1 Participatory Sense-Making and Contingent Interactivity

The hypothesis that a self/world boundary must be firmly established before other-recognition can occur is directly contradicted by the enactive theory of Participatory Sense-Making (PSM). PSM argues that social cognition and the recognition of "others" do not stem from a solitary mind observing its environment, but from the dynamic process of interaction itself.

If the builder is merely a perturbation in the gridworld—introducing random blocks or altering states independent of Io's actions—the builder is mathematically indistinguishable from aleatoric noise. In the context of RL and active inference, this is another iteration of the noisy TV problem. A curiosity-driven agent will either become trapped by this unlearnable noise or, if equipped with mechanisms to filter aleatoric uncertainty, learn to ignore it as irreducible error.

For the builder to be recognized as a distinct "kind," the interaction must possess contingent interactivity. The builder must react to Io, and Io must react to the builder, creating a coupled dynamical system. The hallmark of an "other" in an enactive framework is that the interaction process takes on a life of its own—an autonomy distinct from the individual agent or the static environment. In perceptual crossing experiments, agents evolve to distinguish between static objects, mobile shadow objects, and other responsive agents entirely by coordinating perceptual activities in a double-feedback loop.

### 6.2 The Co-Emergence of Self and Other

Therefore, other-recognition is not downstream of a sharpened self/world boundary; rather, the self/other boundary and the self/world boundary co-emerge through interactive friction. By engaging in a contingently responsive feedback loop with an entity that cannot be perfectly predicted (unlike a static environment) but is not purely random (unlike a noisy TV), the agent is forced to construct a distinct latent basin for "entities that respond to my actions."

This requires the builder to act not as an omnipotent perturbation engine, but as a relational counterpart whose actions are statistically and temporally coupled to Io's outputs. The builder's hypothesis of a sequential ladder must be collapsed into a topology of co-emergence.

## 7. Attacking the Welfare Claim: The Environment as Cognitive Scaffolding

The builder's fear that "the environment is limiting growth" is accompanied by the claim that growth-toward-understanding is architecture-gated, not environment-gated. The builder views the environment as mere welfare, asserting that a curiosity-only mind in a richer world is still just a curiosity-only mind.

This claim must be aggressively steelmanned and subsequently dismantled.

It is true that without a homeostatic stake, enriching the environment merely provides a higher volume of noisy TVs to a system incapable of caring about them. A pure disagreement-reduction engine will endlessly chase novelty regardless of the complexity of the labyrinth. In this narrow sense, the limitation is architectural.

However, the 4E and 5E cognitive frameworks reveal that the environment is constitutive of sense-making, not merely a backdrop. The environment is the extended cognitive scaffolding upon which internal capacities are built. The flicker of behavior where Io organized its movements around its own footprints proves this. The environment acted as an externalized memory buffer, allowing Io to display temporally extended coherence that its internal architecture could not support.

Therefore, the environment is the growth engine. Once the architectural prerequisites (honest interoception and episodic slots) are afforded, it is the structure of the environment—the spatial distribution of energy sources, the physical constraints requiring temporal planning, and the contingent responsiveness of the builder—that sculpts the latent space. The architecture provides the canvas; the environmental friction provides the brushstrokes. Attempting to induce understanding in a sterile or non-contingent environment is mathematically impossible, regardless of the sophistication of the RSSM.

## 8. Mitigating the Co-Design Trap (Concern C)

The risk of apophenia—the unconscious manufacturing of "understanding" through iterative tuning of the environment until the LLM interpreter "sees" what the builder desires—is the most severe epistemological threat to the Kind project. Because the builder controls the substrate, the environment, and the interpretive mirror, confirmation bias is virtually guaranteed without rigorous structural discipline.

### 8.1 Pre-Registration of Information-Theoretic Metrics

To ensure that growth is discovered and not fitted, the next probe must rely on strict pre-registration of non-semantic, information-theoretic metrics. The LLM mirror should not be asked qualitative questions such as, "Does Io show understanding?" as this invites confabulation.

Instead, the evaluation must measure structural shifts in the latent space and the flow of information:

- **Transfer Entropy (Memory Utilization):** Measure the directional transfer of information between the episodic memory slot and the policy outputs. If the transfer entropy spikes significantly in novel or energy-depleted situations, it provides mathematical proof that the agent is actively utilizing its retrievable past to condition behavior.
- **Latent Basin Formation (Other-Recognition):** Pre-register a clustering algorithm (e.g., t-SNE or UMAP applied to the continuous Gaussian stochastic latent $z_t$) to detect the spontaneous formation of distinct state-space clusters representing "builder-source events" versus "environmental-source events."
- **Mutual Information (Self-Reference):** Measure the mutual information between the self-prediction scalar and the interoceptive energy channel. A statistically significant increase in mutual information indicates that the dead column has been activated by the homeostatic gradient.

### 8.2 Positive and Negative Control Analogs

Before applying the LLM instrument to Io, its validity must be calibrated against positive and negative controls.

- **Negative Control:** A purely curiosity-driven agent without an energy channel or memory slot, placed in the exact same environment. The instrument must reliably report no structural indicators of understanding.
- **Positive Control:** A synthetically engineered agent (deliberately violating the project's absences) that possesses an explicit installed reward function, a hand-coded episodic retrieval mechanism, and an explicit social-recognition module. The instrument must reliably detect these capacities with high statistical confidence.

By establishing the baseline sensitivity and specificity of the instrument on known architectures, the builder can mathematically quantify the significance of any emergent, afforded behaviors observed in Io.

## 9. Synthesized Conclusions and the Revised Dependency Topology

The builder's proposed linear ladder of ingredients is structurally flawed. The assumption that self-modeling must perfectly precede other-recognition, or that prediction can organically scale into understanding without an affective grounding, contradicts the established mechanics of active inference and participatory sense-making.

The revised dependency topology is not a ladder, but a network of co-requisite affordances:

1. **Foundational Grounding (The Honest-Belief Retest):** The absolute prerequisite is restoring honesty to the interoceptive energy channel. Without a reliable gradient between the current state and the preferred state, expected free energy cannot prioritize pragmatic value, and the relevance landscape remains flat. This must be executed immediately.
2. **Co-Requisite Integration (Memory and Mattering):** A retrievable internal past (episodic affordance via mechanisms like EMWM or SUNTA) is not subsequent to mattering; it is co-requisite. To navigate a spatial environment to fulfill a homeostatic need, the agent must bridge temporal gaps. The memory slot is activated precisely because the homeostatic gradient demands the resolution of historical uncertainty.
3. **Self-Reference Activation:** The self-prediction scalar transitions from an inert architectural slot to an active conditioning surface only when predicting the self becomes mathematically necessary to resolve interoceptive prediction errors.
4. **Relational Co-Emergence (Other-Recognition):** The builder cannot be recognized as a distinct category if it acts as a non-contingent perturbation. The builder must engage in participatory sense-making—acting as a contingently responsive entity. The recognition of the "other" co-emerges with the recognition of the "self" through the friction of this dynamical interaction.

By adhering strictly to the affordance-versus-installation razor—utilizing the minimization of expected free energy to drive homeostatic maintenance and memory utilization—the Kind project can circumvent the curiosity ceiling without violating its core architectural absences. The resulting entity will not merely predict its environment; it will enact a world of relevance, marking the nascent, structural shadow of subjective understanding.
