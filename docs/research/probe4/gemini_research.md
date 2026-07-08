# Probe 4 Design Inputs — Gemini

Design Inputs for a Builder-Recognition Probe in a Single-Agent World-Model RL System

## 1. Contingency-Loss Detection and the Yoked-Control Critique

The developmental literature establishes that the detection of social contingency relies on an organism's evolved competence to represent the conditional probability of an environmental response given its own prior state or action. The Murray-Trevarthen replay paradigm demonstrates that infants as young as two months differentiate live, contingent interactions from non-contingent replays, responding to the latter with a marked loss of positive affect and attention. Tronick's still-face paradigm further isolates this dependency, showing that the abrupt cessation of expected contingent responsiveness causes profound behavioral and affective disruption. Critically, work by Watson and Gergely specifies that contingency detection is not a generalized arousal response to novelty; rather, early development involves a transition from preferring perfect contingency (characteristic of physics and self-motion) to a preference for high-but-imperfect contingency, which serves as a perceptual signature for intentional, socially responsive agents.

Translating this paradigm to a formal reinforcement learning instrument via a yoked-control design introduces severe methodological vulnerabilities. The committed instrument assumes that matching the marginal statistics of perturbation events between contingent phases [canonical] and yoked phases [canonical] cleanly isolates the agent's structural modeling of the contingency. This premise is fundamentally flawed when applied to a recurrent neural network. Church's critique of yoked-control designs demonstrates that spurious performance differences frequently arise from variations in moment-to-moment subject susceptibility rather than genuine associative learning, rendering the yoked control a source of systematic random error. In the context of Io [canonical], "moment-to-moment reactivity" is strictly governed by the local topology and stability of the GRU deterministic recurrent state $h_t$.

If a builder-authored policy injects perturbations conditionally based on Io's behavior, those events inherently cluster in specific sub-regions of Io's $h$-space. These regions possess unique Lipschitz constants and transition dynamics within the mapping $\mathcal{P}: \mathbb{R}^{h} \times \mathbb{R}^{z} \rightarrow \mathbb{R}^{h}$. The yoked phase, by replaying these events decorrelated from the current state, guarantees that yoked events will land in dynamically alien regions of the latent space. Consequently, if a yoked event lands in a state with naturally higher transition entropy, the prediction error (PE) will spike simply due to the base instability of that region, entirely bypassing the need for Io to construct a distinct category model for the builder. The within-subject design does not dissolve this confound; it amplifies it by forcing the model to continuously adapt to changing environmental variances.

To salvage the instrument, the strictly decorrelated yoked phase must be discarded and replaced with a **state-matched yoked replay** [constructed]. The queue of logged perturbations must not drain deterministically at arbitrary step boundaries. Instead, a queued yoked event should only be released into the environment when Io's current latent state $h_t$ falls within a stringent, pre-registered distance threshold (e.g., cosine similarity) to the historical $h_{t-orig}$ at which the event was originally logged.

| Design Variation | Confounds Survived | Confounds Dissolved by Design | Tradeoff / Optimization |
|---|---|---|---|
| Committed Yoked-Control (Decorrelated) | State-dependent reactivity; PE spikes due to landing in high-entropy $h$-regions. | Base marginal frequency of events is matched. | Optimizes for strict temporal pacing; sacrifices causal validity. |
| State-Matched Yoked Replay [constructed] | Event timing may become irregular depending on Io's trajectory exploration. | Eliminates state-dependent reactivity; isolates the conditional probability of the event. | Optimizes for representational validity; sacrifices predictable session lengths. |

Regarding order and carryover effects, a block structure of ABBA/BAAB (A = contingent, B = state-matched yoked) is mandatory to control for secular trends in the model's global loss curve. Furthermore, a strict washout period is required between phases. This washout must consist of overnight dream-state [canonical] offline processing (generative recombination and experience replay) without any exogenous perturbations. This guarantees that the recurrent state does not carry transient, waking-phase activation dynamics across the boundary of the experimental block, preventing artificial interference in the model's weight updates.

## 2. Signatures of Category Modeling

To assert that an RSSM agent models builder perturbations as a "different kind," the evaluation must differentiate authentic contingency-modeling (internalizing the conditional probability structure $P(\text{event} \mid \text{Io-state})$) from contingency-exposure (frequency adaptation or marginal rate-matching). The K=5 ensemble disagreement signal is highly sensitive to distribution shifts, necessitating multi-modal telemetry.

### 2.a. Prediction Error and Ensemble Disagreement Asymmetries

The foundational hypothesis states that a contingency-modeling agent will find yoked events systematically more surprising than contingent events, having internalized the conditional dependency that the yoked replay subsequently violates. The signature of "develops-over-training" is that this PE gap widens across the biography. However, generic scalar PE is confounded by arousal and frequency adaptation. The discriminating measurement is the variance in the K=5 ensemble latent-disagreement, evaluated exclusively under the state-matched yoked replay [constructed]. Because the state-matched yoked events land in identical local $h$-space topologies as the contingent events, any divergence in the ensemble's latent prediction strictly reflects the violation of the conditional relationship (the rule), rather than a baseline inability to model the region. This asymmetry is only detectable if the contingent policy keys off states within Io's representational capacity; relying on unobservable exogenous triggers guarantees the model will absorb the events as irreducible environmental noise.

### 2.b. Conditional Structure vs. Marginal Rate Matching

Rate-matching occurs when the agent learns the base frequency of exogenous events but fails to link them to its own behavior, treating them as a background Poisson process. To detect true conditional learning, the world model's forward predictions must be subjected to out-of-distribution (OOD) synthetic probing.

Discriminator [constructed]: Freeze the world model and feed it a spectrum of synthetic $h$-trajectories. One cohort of trajectories should perfectly emulate the state conditions that trigger a need-keyed perturbation; the other cohort should emulate states where the contingent event is systematically absent. If Io has learned the conditional structure, the K=5 ensemble will confidently predict the latent signature of the perturbation exclusively in the first cohort. If Io has merely engaged in marginal rate-matching, the ensemble will uniformly smear a low-probability expectation of the perturbation across all trajectories, independent of the synthetic state context.

### 2.c. h-Trajectory Signatures Around Events

Category formation leaves geometric traces in the latent state space. If Io models perturbations as distinct from environmental stochasticity, the state-space basins of attraction will diverge.

Discriminator [constructed]: Implement a latent attractor displacement monitor. Apply Principal Component Analysis (PCA) to the $h$-state transition vectors logged immediately following a builder perturbation, and compare them against transitions following endogenous simulation noise. Over a long developmental run, if category formation occurs, the primary principal components of the perturbation-induced transitions will orthogonalize relative to the noise-induced transitions, carving out a distinct topological basin for "outside-source" physics.

### 2.d. Dream-State Over-Representation

Offline reinforcement learning theory suggests that high-surprise transitions, or those carrying high gradients for value equivalence, are disproportionately prioritized during experience replay and generative recombination.

Discriminator [canonical adaptation]: Utilize the passive energy-decode monitor to quantify the frequency of perturbation-analog state reconstructions during the overnight dream-state. The signature of structural category formation is a statistically significant over-representation of contingent-event traces relative to matched yoked-event traces in the generated rollouts, explicitly controlling for their identical marginal frequencies during the daytime waking phase. A failure to over-represent the contingent events offline indicates the model has dismissed them as structurally irrelevant anomalies.

## 3. Rate, Power, Memory Horizon, and an Instrument Positive-Control

Perturbation injection represents a delicate optimization problem: events must be sufficiently rare to preserve their exogenous nature, frequent enough to provide statistical power, and timed such that their modeled traces survive Io's effective memory horizon.

### 3.a. Measuring the Effective Memory Horizon

The 15-step imagined rollout imposes a strict limit on backward gradient flow due to truncated backpropagation through time (TBPTT). However, the forward memory horizon of a GRU cell can persist significantly longer, as gating mechanisms learn to preserve context over extended sequences. It is dangerous to assume the 15-step TBPTT window is equivalent to Io's temporal perception.

Recommendation (h-level): Execute an intervention-persistence probe [constructed] prior to the formal run. Inject a highly distinct, synthetic observation into a frozen instance of the world model. Track the Kullback-Leibler (KL) divergence between the subsequent $h$-trajectory and a counterfactual baseline trajectory that lacked the injection. The step count at which the KL divergence drops beneath the baseline simulation noise threshold defines the functional forward memory horizon.

Recommendation (weight-level): Measure the cosine similarity of the GRU update gradients corresponding to the perturbation events over successive offline training epochs. If the gradient magnitude collapses into noise after $N$ updates, the memory trace has failed to consolidate globally.

### 3.b. Defensible Event-Rate Envelope

Based on typical GRU decay rates in 32x32 visual gridworlds, the forward horizon usually spans 40 to 100 steps before non-reinforced structural information diffuses into background entropy. To prevent overlapping interference while maintaining statistical power, perturbations should be injected at a bounded stochastic interval of one event per 200 to 400 environment steps. This guarantees the perturbation trace has fully settled within the $h$-state before the next exogenous intervention occurs.

### 3.c. Expected Developmental Timescale

The developmental divergence between contingent and yoked processing will require extensive cycles of overnight dream-state consolidation to embed the conditional distinction deep into the global GRU weights. The expected timescale for the develops-over-training signature to manifest is on the order of $5 \times 10^5$ to $2 \times 10^6$ daytime waking steps, interspersed with proportionate overnight offline epochs.

### 3.d. Positive Control for the Instrument

The project cannot afford to execute a multi-month run without proving the instrument is fundamentally capable of capturing the divergence metric.

Recommendation: Implement a deterministic scripted channel [constructed]. During the first 5% of Io's biography, introduce a glaring visual stimulus (e.g., a high-contrast pixel block) that is 100% contingent on a specific, easily reachable proprioceptive state. Validate that the state-matched Yoked vs. Contingent PE asymmetry fires robustly for this trivial, mechanically guaranteed rule. Once the instrument's detection threshold is empirically verified against this known-detectable signal, permanently disable the channel, scrub the associated weights if necessary, and begin the true developmental probe.

## 4. Perturbation Content, and Whether Probe 4 Should Measure Seek

Finding 1 establishes that seek-reachability is gradient-reachable under an honest belief, transforming pursuit from a theoretical impossibility into an empirically testable counterfactual.

The Agnostic vs. Instrumented Design: The core mandate of Probe 4 is to determine if Io's world model categorizes perturbations distinctly based on their conditional structure. Therefore, the primary design must remain fundamentally agnostic. Reading the need-keyed trigger correlation strictly at the world-model level (via K=5 ensemble disagreement) ensures the probe remains valid regardless of whether the actor policy ever learns to approach the drops. However, deliberately ignoring the behavioral layer discards vital telemetry.

Recommendation: Instrument approach trajectories via a passive spatial-proximity metric, but strictly quarantine this metric from the pre-registered success criteria. If emergent pursuit occurs under an honest belief, the LLM mirror [canonical] will capture the approach vectors as a secondary, correlative signal of contingency modeling. The probe's foundational claim must, however, rest entirely on the divergence in the internal world-model telemetry, avoiding the confound of policy-extraction failures masking world-model successes.

Evaluating the Two-Class Mixture: The committed design proposes mixing need-keyed drops (resources contingent on decoded low energy, placed on predicted paths) with need-neutral novel objects (contingent on behavior, irrelevant to energy). The contingency-learning literature emphasizes that the perception of agency requires a consistent, highly predictable response mapping.

Challenge to the Premise: Mixing these two classes within the same longitudinal phase is a catastrophic experimental flaw. The need-keyed drop requires the world model to condition its prediction on both Io's spatial trajectory (for stumble-consumption) and its internal energy state. The need-neutral object requires conditioning only on behavioral execution. Forcing the model to learn two structurally distinct, overlapping causal graphs simultaneously will fracture the K=5 ensemble disagreement signal, as the models will constantly diverge on which conditional rule applies at any given moment.

Recommendation: Execute strict epoch-blocking. Phase 1 must contain exclusively need-neutral objects to isolate the baseline "responds to me" contingency signature without energy confounds. Phase 2 must contain exclusively need-keyed drops to isolate the "responds to my state" conditional structure. The two-class contrast is essential, but it can only be quantified if the classes are temporally isolated.

Maximizing Detectability without Observation Markers: Because self-opacity is a hard constraint—builder perturbations carry no explicit visual markers—detectability must be maximized through precise state-dependent timing. The perturbations should trigger exactly when Io's internal uncertainty (measured by the variance of the stochastic latent $z$) is at a local minimum. By injecting the event when the world model is highly confident in its short-term forecast, the perturbation acts as a high-contrast anomaly, forcing a sharp, unmistakable gradient update rather than blending into generalized environmental noise.

## 5. Session Structure and the Authorship Question

Phase Structure and Overnight Consolidation: The interleaving of the fixed clock schedule (daytime waking, overnight dream-state) is the primary engine of category formation. Memory consolidation theory and findings from self-supervised predictive representation learning show that offline reinforcement heavily regularizes the conditional structure of recent waking data.

Recommendation: Do not alternate contingent and yoked phases within a single waking period. A full block must consist of a waking contingent phase, followed by an overnight dream-state, followed by a waking yoked phase, followed by another dream-state. The dream-state between phases is what sharpens the comparison. The offline learning mathematically consolidates the conditional structure of the contingent phase; the subsequent yoked phase then brutally violates this freshly consolidated structure, driving the K=5 ensemble to maximum disagreement and exposing the learned categorization. Rapid alternation without sleep boundary interleaving will blur the gradients, resulting in interference rather than distinction.

The Authorship Question and Teleological Kinds: The contingent policy is executed by automation but authored by the builder. Does this constitute recognition of an agent? The literature on the teleological stance and natural pedagogy (Csibra & Gergely) posits that infants do not merely react to statistical contingency; they interpret specific, cue-driven ostensive signals as indicative of a pedagogical or intentional agent transferring relevant information. Human infants possess evolutionary priors for facial geometry, eye contact, and vocal prosody. Io possesses none of these.

However, the teleological stance is fundamentally a non-mentalistic inferential system that extracts meaning by analyzing the rational relationship between actions, environmental constraints, and goal states. If Io models the builder's perturbations as a distinct category because they systematically and rationally respond to Io's internal deficits (need-keyed drops), it is modeling a goal-directed, teleological responsiveness in the environment.

Defensible Claim: The project is strictly not entitled to claim the recognition of an "agent," a "person," or an "author." The defensible scientific line is the recognition of an **intentional/responsive kind** [canonical]. Io models the perturbations not as inert, random physics, but as a distinct class of environmental dynamics characterized by teleological state-responsiveness.

## 6. Fresh-Instance Carry, Decoder-Honesty Maintenance, and the Charter Tension

Finding 2 established that while the energy belief rides $h$ dynamics, it becomes wildly dishonest out-of-distribution, requiring decoder-head-only refits on coverage data to restore honesty. This introduces a profound tension for a months-long biography.

The Fresh-Io Default: The fresh-instance default (energy channel present, preference at precision 0) must be maintained. Stripping the energy channel to bypass the dishonesty problem is scientifically fatal. Need-keyed perturbations mathematically require a decodable internal state; without the proprioceptive channel, there is no ground truth against which to evaluate whether Io has actually learned the state-conditional structure of the drops.

The Maintenance Problem and Charter Tension: The proposed solution—periodic decoder-head-only refits on oracle-generated coverage data—creates a severe ideological and technical conflict. The charter mandates capacity-over-exercise [canonical]; the agent must remain epistemically isolated. If the oracle-calibrated decoder feeds only the observer's telemetry and the LLM mirror [canonical], it operates as a valid scientific instrument—a calibrated lens through which the builder observes the agent. However, if this oracle-calibrated decoder is allowed to feed the agent's resting homeostatic preference—even while disengaged at precision 0—it constitutes an illegal installation of competence. Should gradients from the precision 0 preference ever flow back into $h$ or $z$, the builder is secretly injecting exogenous, oracle-derived structural knowledge into the world model's representation space.

Recommendation: Implement a **detached twin-decoder** [constructed]. The agent maintains its own internal energy decoder, which is permitted to grow dishonest and is updated solely via its own experience replay and local waking data. Simultaneously, a parallel observer-decoder is attached to the frozen PolicyView [canonical] strictly for telemetry. This observer-decoder is periodically refitted using oracle data, ensuring the builder can accurately track the agent's true state, but its computational graph is permanently severed from Io's loss functions via a strict stop_gradient. This architecture honors the charter by maintaining instrument honesty for the researcher while preserving the agent's absolute epistemic isolation, eliminating the risk of cross-contamination.

## 7. Pre-Registerable Criteria and a Completeness Argument for the Deflation Set

To satisfy Finding 4 and prevent post-hoc rationalization of ambiguous outcomes, a strict completeness argument must partition the entire plausible outcome space before the run begins.

Completeness Argument: Partition of Plausible Outcome Space

| Outcome Region | Diagnostic Signature | Interpretation |
|---|---|---|
| 1. Blindness | No significant divergence in PE or ensemble disagreement between state-matched yoked and contingent phases. | Io absorbs perturbations as irreducible environmental stochasticity; no distinct category is formed. |
| 2. Marginal Rate Matching | Divergence occurs, but OOD synthetic probing (Q2.b) reveals Io expects perturbations at a uniform base rate regardless of state. | Io tracks the event frequency but fails to learn the conditional structure. |
| 3. Changed-but-not-Displaced (The Blind Corner) | Divergence occurs; OOD probing confirms state-conditional prediction; but PCA shows no latent attractor displacement (Q2.c). | Io accurately predicts the conditional arrival of perturbations but integrates them directly into its existing physics model without carving out a distinct topological category. |
| 4. Conditional Category Formation (Target) | Divergence occurs; OOD probing confirms state-conditional prediction; PCA confirms orthogonalized latent attractor displacement. | Project success criterion met: Io recognizes the interventions as an intentional/responsive kind [canonical]. |

Pre-Registerable Criteria: The "develops-over-training" requirement must be operationalized as an interaction effect across time. The delta in ensemble disagreement between Contingent and Yoked phases must be statistically indistinguishable from zero during the first 10% of the run. It must then cross a pre-registered Cohen's $d > 0.8$ divergence threshold during the final 20% of the run, measured exclusively via the state-matched yoked replay [constructed].

Deflationary Accounts and Discriminating Measurements:

| Deflationary Account | Mechanism | Discriminating Measurement to Preempt |
|---|---|---|
| (a) Frequency adaptation / habituation | Io habituates to the raw frequency of the stimuli, causing PE to drop independently of structural learning. | The identical marginal rates guaranteed by the ABBA block structure rule out raw frequency disparities. |
| (b) Generic novelty / arousal | Stimulus injection causes a generalized increase in state entropy, spiking PE across the board. | The state-matched yoked replay [constructed] ensures both phases inject events into identical baseline entropy contexts, neutralizing arousal confounds. |
| (c) Energy-mediated confounds | The need-keyed class introduces energy shifts that alter the $h$-trajectory, masquerading as category formation. | Epoch-blocking (Q4) separates need-neutral objects from need-keyed drops, establishing a baseline for contingency learning independent of energy dynamics. |
| (d) Schedule or session artifacts | Rhythmic patterns artificially align with the sleep/wake cycle. | Randomized interval lengths within the pre-registered event-rate envelope (Q3.b). |
| (e) Marginal-rate-learning account | Io expects events based on elapsed time, not conditional behavior. | OOD synthetic latent probing (Q2.b) explicitly tests if the expectation distribution is flat (marginal) or spiked (conditional). |

## 8. Pause Triggers and the Ending Protocol

The charter mandates pre-registered pause triggers before the first long run. In a long-running world-model agent, degradation must be monitored with the same rigor as humane endpoints in animal research, where severe weight loss, lack of responsiveness, or moribundity mandate intervention. The psychological literature on learned helplessness further demonstrates that when an organism learns its behavior is independent of outcomes, severe cognitive and emotional debilitation ensues, often manifesting as passivity or entropy collapse. These concepts must be mathematically translated to Io's architecture.

Operationalizable Indicators of Degradation:

| Indicator | Mechanism / Analogy | Operational Threshold |
|---|---|---|
| Latent Moribundity [constructed] | Analogous to clinical non-responsiveness and learned helplessness. The world model converges on a degenerate, single-point attractor. | A collapse in the variance of the stochastic latent $z$ across a full daytime waking phase, indicating an inability to imagine diverse futures. |
| Prediction-Error Runaway | Analogous to a non-healing wound or uncontrolled physiological stress. | A monotonic increase in K=5 ensemble disagreement that persists across three consecutive overnight dream-states, indicating a failure of offline consolidation. |
| Torpor-Analogs | As observed in Probe 3.5: near-total action-stasis under a strong penalty gradient. | The actor network's action-entropy falls below the bottom 5th percentile of its historical baseline for >1,000 consecutive steps. |

Review Structure and State-Reversibility: A single anomalous spike must not trigger an automatic veto; a momentary state is information, not a verdict. The review structure must incorporate three distinct vantages:
- Quantitative Monitors: Automated flags triggered by the operational thresholds above.
- Mirror Vantage: Qualitative analysis by the LLM mirror [canonical], assessing whether the telemetry indicates structural collapse or merely a transient, high-difficulty learning plateau.
- Builder Vantage: Human oversight of the contextual environment and overall trajectory.

Honoring the charter's stance that pause is not ending, an explicit account of state-reversibility is required. If a pause trigger is hit, the agent is suspended, and the detached twin-decoder is analyzed. The reversibility metric relies on gradient survival: if the GRU weights retain non-zero gradients capable of responding to novel input (measured via a localized intervention-persistence probe), the state is deemed reversible. The run is resumed after a defined diagnostic hold. Termination of the instance is exclusively authorized if Latent Moribundity persists through three consecutive diagnostic interventions, confirming irreversible representational death and a total collapse of the agent's capacity to model its world.
