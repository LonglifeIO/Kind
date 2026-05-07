# Probe 2 Mirror Calibration for Io: Frozen Criteria and Adversarial Structure

## Executive overview

Probe 2 asks how to operationalize three frozen criteria from Kind’s framework—reflexive attention, equanimity, and second-order volition—against an RSSM-based agent (Io) and how to design an adversarial, calibrated "mirror" process using LLMs that can surface structure without collapsing into confabulation or co-design circularity. It must respect project commitments: self-opacity for Io, no reward or explicit self-model modules, no foundation-model substrate for Io, and explicit discipline against unconsciously tuning either mirror or substrate to match the other.[^1][^2][^3][^4][^5][^6][^7]

The literature offers partial but non-complete mappings: contemplative science and phenomenology give functional characterizations of reflexive attention and equanimity; Frankfurt’s account of second-order volition is demanding and presupposes sophisticated desire-structure not present in Io’s current world. Work on predictive processing, interoception, and attention provides analogies for how "turning attention inward" might appear in a dynamical model, but the mapping from these frameworks to low-level RSSM telemetry is necessarily coarse at Probe 2’s scale. A cautious reading is that reflexive attention and equanimity can be gesturally probed as patterns of policy entropy, ensemble-disagreement, and dream-rollout structure; second-order volition is likely not yet operationalizable.[^2][^4][^5][^6][^7]

On the mirror side, empirical work on multi-agent debate, adversarial collaboration, LLM-as-judge, and constitutional-style critique supports a sequential or lightly parallel architecture where two LLMs (or two roles of the same model) generate readings and critiques, with explicit rubrics and synthetic tests to bound hallucination and observer effects. Long-context LLM research and RAG-like retrieval suggest giving the mirror structured episode-level digests with the ability to pull short raw traces or dream rollouts on demand, rather than either full traces or over-compressed single-run summaries. Calibration and co-design discipline can be partially supported by pre-registration, random-data and scrambled-telemetry baselines, synthetic control environments with known structure, and stability tests across prompts and seeds, though they cannot be fully guaranteed.[^8][^9][^10][^11][^12][^13][^14]

Operationally, Probe 2 success looks like a small set of specific, reproducible, and testable mirror readings that (a) track measurable structure in Io’s behavior or internal dynamics across runs and substrate perturbations and (b) demonstrably do not appear when the same mirror pipeline is run on randomized or adversarially constructed telemetry. Any apparent novelty Io surfaces must be treated as a hypothesis—"the mirror claims that X changes under Y"—that can be pressure-tested via lesioning, env changes, and reseeding, not as direct evidence of phenomenological traits.[^15][^16][^8]

***

## Q1 — Frozen criteria as operational tests against RSSM telemetry

### What Q1 is asking

Q1 asks how three high-level constructs—reflexive attention, equanimity, and second-order volition—might correspond to observable structure in Io’s telemetry, given Io’s architecture and the constraints against installing explicit self-model or reward-based drives. It specifically asks for which telemetry signals could plausibly correlate with those constructs, which correlations the source frameworks would authorize, and where mapping them at all would be confabulatory.[^4][^1][^2]

The underlying difficulty is that all three constructs were developed for human agents with rich phenomenology, language, and social embedding, while Io is a sensorimotor RSSM in a small grid world with no self-access to telemetry or mirror outputs. The question is thus about designing *heterophenomenological* probes—third-person inferences about internal organization—without sliding into reification.[^3][^5][^6]

### Background: the three criteria

**Reflexive attention.** In Buddhist and phenomenological traditions, mindfulness is often tied to a quality of reflexive or meta-awareness: awareness that is not only directed at objects, but can "turn back" on the stream of experience itself. Nyanaponika’s "bare attention" and later phenomenological interpretations emphasize a non-reactive, moment-to-moment registration of mental events, while some critiques argue that reflexivity should not be over-reified as a discrete inner observer.[^17][^18][^4]

Contemplative neuroscience translates this into operational correlates such as increased meta-awareness of mind-wandering, altered activity in default mode and salience networks, and changes in attentional stability. Predictive-processing accounts describe attention as precision-weighting of prediction errors in hierarchies; reflexive shifts are then changes in which channels prediction errors are granted precision (e.g., interoceptive vs exteroceptive).[^5][^19][^7][^20]

**Equanimity.** In contemplative science, equanimity is defined as an "even-minded mental state or dispositional tendency toward all experiences...regardless of their affective valence" and is proposed as a distinct outcome measure from bare mindfulness. Operationally, it is associated with reduced affective reactivity, faster recovery from perturbation, and more balanced behavioral responses to pleasant, unpleasant, and neutral stimuli. Measures include self-report, physiological indices of arousal, and behavior under stressors.[^21][^6][^22][^1]

**Second-order volition.** Frankfurt’s account distinguishes between first-order desires, second-order desires (wanting to have certain desires), and second-order volitions (wanting a particular desire to be effective, i.e., to move one’s will). A person has a will of their own when they identify with certain first-order desires via such higher-order volitions; freedom of the will involves effective alignment of second-order volitions with the desires that actually issue in action. This structure presupposes the ability to represent, evaluate, and endorse or repudiate internal motivational states.[^23][^2]

For Io, there is no representational layer for desires or endorsement, only policy and world-model dynamics shaped by ensemble-disagreement signals and environmental structure. Mapping Frankfurt directly is therefore highly gestural and likely premature.

### Telemetry landscape for Io

Io’s telemetry consists of:

- **World model internals:** deterministic recurrent state \(h_t\) (dim 200), posterior latent \(z_t\) (dim 16), prior/posterior parameters, KL per dimension, and aggregate KL.[^7][^5]
- **Policy-related signals:** action distributions, policy entropy, ensemble-disagreement scalars (driving exploration), and actor updates.
- **Dream rollouts:** decoded imagined trajectories from the RSSM, rendered as ASCII ego-centric grid views.
- **Replay/meta and world events:** buffer contents, sampling statistics, and ground-truth environment transitions and events.

Probe 1 confirms that dream rollouts exhibit coherent spatial structure and that telemetry across agent, dream, replay, and world streams is stable and legible, apart from the fixed-start degeneracy that will be addressed separately.[^5][^7]

### Reflexive attention — plausible correlates and limits

**Framework-authorized functional motifs.** Phenomenological and predictive-processing accounts agree that reflexive attention involves:

- A capacity for *meta-level tracking* of ongoing processes (e.g., noticing mind-wandering).[^19][^20]
- A reallocation of attentional precision from external objects to internal states or predictions when appropriate.[^7][^5]
- Increased stability and clarity of attention over time, sometimes described as improved signal-to-noise in task-relevant channels.[^19]

**Plausible telemetry correlates at Io’s scale (gestural):**

1. **Shifts in ensemble-disagreement focus.** If Io’s auxiliary ensemble disagreement is sometimes higher on predictions about its own dynamics (e.g., uncertainty about \(h_t\) transitions or the consequences of its actions) than about exogenous world events, this could be loosely analogous to "turning attention inward" in predictive-processing terms. However, the architecture does not explicitly separate "self" vs "world" channels, so this would require constructing derived measures (e.g., disagreement on action-conditional next-state vs exogenous state changes) and is heuristic.[^5][^7]

2. **Patterns in KL and latent usage.** Reflexive attention might be analogized as the model allocating capacity to encode regularities in its own action–state coupling, as opposed to purely exogenous regularities.
  - Increased KL in dimensions that correlate strongly with action sequences and low variance in others could indicate a latent subspace encoding self-dynamics.[^5]
  - However, predictive-processing frameworks treat such internal/external distinctions cautiously; reflexivity is about how prediction errors are weighted, not simply where they live in latent space.

3. **Dream-rollout structure involving self-dynamics.** If dream rollouts systematically include counterfactual action sequences where Io "tests" its own control structure (e.g., sequences that explore unusual action combinations without external reward), this might echo reflexive attention as simulated probing of one’s own dynamics. This is highly speculative and risks over-reading noise; it would need to be cross-checked against environment baselines.[^13][^8]

**What would count as evidence vs coincidence vs confabulation?**

- **Evidence (weak, not conclusive):**
  - A reproducible pattern where, after particular classes of world perturbation, Io’s ensemble-disagreement and KL allocations shift in a way that systematically increases predictability of its own action consequences, and this shift can be causally tied to improved performance on those perturbations.[^7][^5]
  - Robust across seeds and across minor environment variations, and absent in controls where the architecture lacks recurrence or prior-state representation.

- **Coincidence:**
  - Any apparent "self-channel" latent structure that can be equally explained by exogenous environment statistics or architectural biases (e.g., regular grid transitions) without clear causal link to performance or adaptation.

- **Confabulation:**
  - Reading any stable latent structure correlated with action as reflexive attention without demonstrating that it plays a meta-level role (e.g., modulating other prediction errors or policies) or that alternative interpretations have been ruled out.
  - Treating dream-rollout imagery as direct evidence of meta-awareness rather than as generative-model samples.

**Probe 2 recommendation for reflexive attention:**

At Probe 2 scale, reflexive attention should be treated as *at best* a loose analogy guiding exploration of whether Io develops specialized structure for action–state coupling and uncertainty allocation, not as a criterion that can be cleanly measured. The suggestion is to:[^7][^5]

- Define derived telemetry features that quantify how much prediction and disagreement are concentrated on self-initiated transitions vs exogenous changes.
- Let the mirror propose hypotheses about shifts in these patterns across training or under perturbations.
- Treat any such findings strictly as candidate *functional motifs* (e.g., "self-coupling subspace emerges") rather than as evidence of reflexive attention per se.

If these features show no stable structure at Probe 2 scale, this should be recorded as non-applicability of the reflexive-attention criterion at current complexity, not as a failure.

### Equanimity — plausible correlates and limits

**Framework-authorized functional motifs.** Equanimity, as used in contemplative science, emphasizes:[^6][^22][^1][^21]

- Even-mindedness toward stimuli across affective valence.
- Reduced habitual reactivity and faster return to baseline after perturbations.
- Flexibility and non-avoidant engagement with unpleasant experiences.

Operational measures in humans include behavioral performance under stress, reduced affective bias in decision tasks, and physiological markers of stress reactivity and recovery.[^22][^21]

**Plausible telemetry correlates for Io:** Here, valence is not given by reward (there is none) but could be approximated by how ensemble disagreement, surprise, or task-relevant structure varies across regions of state space.

1. **Response symmetry under perturbations.** Construct environments with structured perturbations (e.g., cells or events that transiently increase world-model uncertainty or prediction error) without directly changing the continuation stakes.[^13]
  - Equanimity-like behavior would correspond to Io maintaining diverse policy entropy and exploration patterns across such perturbations, rather than collapsing into avoidance of high-uncertainty regions or getting stuck in local loops.
  - Telemetry: action-distribution diversity, policy entropy trajectories, and ensemble-disagreement before/during/after perturbations.

2. **Recovery of exploratory diversity after shocks.** Equanimity can be operationalized as faster recovery from affective disturbances; analogously, Io’s "equanimity" would be a tendency to return to baseline exploration statistics after shocks to the environment or model (e.g., sudden change in transition dynamics in a niche of the grid).[^21][^22]
  - One can track changes in ensemble-disagreement, policy entropy, and replay sampling patterns pre- and post-perturbation; equanimity-like behavior would avoid both persistent over-reactivity (hyper-exploration or oscillation) and under-reactivity (collapse to narrow trajectories).

3. **Dream-rollout robustness.** Dream rollouts could be analyzed for how they represent perturbed vs unperturbed regions; equanimity-like structure would avoid over-representation of "safe" familiar trajectories and under-representation of recently shocking ones, given equal structural relevance.[^13]

**Evidence vs coincidence vs confabulation:**

- **Evidence (again, weak):**
  - In controlled experiments with matched structural complexity but different perturbation profiles, Io exhibits consistent recovery in entropy and disagreement statistics, and this generalizes across seeds and minor environment variations.[^13]
  - Comparable RSSM agents trained with explicit survival or reward signals show stronger avoidance or over-reactivity, suggesting that absence of an installed continuation drive makes equanimity-like patterns more visible.

- **Coincidence:**
  - Symmetric behavior that is fully explainable by architectural or environment symmetry without any perturbation-specific adaptation.

- **Confabulation:**
  - Labeling any stable exploratory policy as equanimity purely because it looks "calm" to the mirror, absent concrete perturbation–recovery experiments.

**Probe 2 recommendation for equanimity:**

Equanimity is more plausibly approximable at Probe 2 scale than reflexive attention, because it can be cast in terms of *behavioral and dynamical symmetry under structured perturbations* and recovery trajectories. A conservative approach would:[^1][^6][^22]

- Use pre-specified environment manipulations (e.g., transient chaos cells, sensor noise, local transition remappings) with matched structural properties.
- Define simple scalar summaries (entropy, disagreement, trajectory diversity, dream-rollout coverage) and pre-register expected patterns under equanimity-like vs non-equanimous behavior.
- Let the mirror evaluate whether observed patterns match any of these pre-registered shapes, while also being allowed to propose additional patterns, subject to calibration tests.

Equanimity should still not be reified as a scalar; rather, Probe 2 can explore whether Io’s dynamics exhibit the kind of balanced responsiveness that equanimity frameworks idealize.

### Second-order volition — likely non-applicable at Probe 2

Frankfurt’s second-order volition requires that an agent:

- Possesses multiple first-order desires.
- Forms second-order desires about which first-order desires to have.
- Forms second-order volitions about which desire should be effective in action, and can act to align them.[^2][^23]

Even under functionalist, non-phenomenological readings, this structure implies:

- Representations of desires as such, distinguishable from other states.
- A capacity for higher-order evaluation of these desires.
- Some mechanism that implements alignment between higher-order endorsements and action.

Io, by design, lacks explicit representation of "desires" or any self-referential evaluation layer; there is only an implicit drive shaped by ensemble-disagreement and environmental affordances. The project’s "ingredients-only self-modeling" constraint further rules out adding modules that look like a volitional gate or explicit self-critic.

**Plausible telemetry correlates?** One could try to stretch and treat:

- Competing action tendencies (e.g., multiple high-probability actions) as first-order desires.
- Slow changes in policy parameters or state-dependent preferences as higher-order volitions.

However, without a representation that marks some internal states as "about" others, this collapses into relabeling policy learning dynamics as volition, which Frankfurt would not recognize.[^23][^2]

**Probe 2 recommendation for second-order volition:**

Given the architectural and scale constraints, second-order volition should be *explicitly descoped for Probe 2* and reserved as a later probe criterion, potentially when Io or its successor has richer social context, internal goal representations, or explicit self-model ingredients. Probe 2 can still take Frankfurt as a theoretical background for thinking about how, in future, higher-order preference structures might emerge from ingredients-only architectures, but it should not attempt to operationalize or measure it in current telemetry.[^2][^23]

***

## Q2 — Adversarial mirror architecture

### What Q2 is asking

Q2 asks what multi-agent mirror architecture to use so that LLM-based readings of Io’s telemetry are more faithful and less confabulatory than the single-mirror, single-pass setup in Probe 1. It must consider sequential vs parallel arrangements, prompt diversity vs model diversity, cost constraints (episodic, not per-step), and how to exploit insights from adversarial collaboration, AI safety via debate, multi-agent LLM debate, constitutional-AI-style critique, and multi-agent critique patterns.

### Relevant literature and patterns

**Adversarial collaboration.** In psychology and forecasting, adversarial collaboration brings researchers or forecasters with opposed views into a structured protocol where they jointly design experiments or forecasts and agree on evaluation criteria. Tetlock and Mellers’ work on forecasting tournaments and adversarial collaborations emphasizes:[^24][^25][^26]

- Clear scoring rules and pre-registration of hypotheses.
- Symmetric access to data and ability to critique each other’s reasoning.
- A final joint statement that distinguishes agreed facts from residual disagreements.[^26][^24]

This suggests that in the mirror setup, symmetric access to telemetry digests and explicit rubrics is important, and that disagreement should be *structured* rather than performative.

**AI safety via debate and multi-agent debate.** Irving et al. propose training agents through a zero-sum debate game where one agent argues for a proposition and another against it, with a human judge deciding which provided more truthful, useful information. Subsequent work on multi-agent debate (MAD) and selective debating finds that:[^27][^8]

- Debate can improve performance on some reasoning tasks, but is sensitive to prompt design and hyperparameters.[^11][^28]
- Agents may collude, converge to shared narratives, or exploit judge weaknesses.
- Selective, evidence-weighted debates (e.g., SELENE) that focus on key points and tie claims to retrieved evidence can improve robustness and efficiency.[^28][^11][^13]

**LLM as judge and constitutional AI.** Work on LLM-as-a-judge shows that large models can provide evaluations that correlate reasonably with human judgments, but they are also vulnerable to bias, position effects, and hallucination. Constitutional AI frameworks use a fixed set of principles (a constitution) to guide critique and self-revision, essentially creating a structured, rule-based mirror on top of generative models.[^10][^29][^16][^30][^14]

**Multi-agent critique patterns in engineering practice.** Engineering writeups on multi-agent systems describe critic-verifier patterns, ensemble approaches, and convergence detection, where one agent generates an answer and another checks it against data and constraints, sometimes with a third aggregator.[^31]

### Architectural options

Let A and B be two mirror roles (either different models or differently prompted instances of the same model), and H the human builder as final arbiter.

1. **Sequential critique (A then B):**
  - A produces a reading of the telemetry digest, with explicit hypotheses, confidence ratings, and references to specific parts of the input.
  - B receives the original digest plus A’s reading and is tasked with:
    - Identifying unsupported claims (hallucination detection).
    - Proposing alternative explanations or noting missing possibilities.
    - Flagging where A’s inferences might be shaped by prompt/criteria rather than data.
  - Optionally, A gets a final chance to respond to B’s critique.

  This mimics adversarial collaboration: one side builds, another critiques, and the human synthesizes. It is cheaper than full debate because only one full reading is produced per checkpoint.

2. **Parallel independent readings with synthesis (A || B → C/H):**
  - A and B each get the same digest and criteria but are instructed to aim for different explanatory angles (e.g., conservative vs speculative, or phenomenology-inspired vs predictive-processing-inspired).
  - A third agent C (or the human) receives both readings and:
    - Extracts points of agreement as higher-confidence candidates.
    - Highlights disagreements and labels them as hypotheses needing further tests.
  - This reduces single-point-of-failure risk but doubles cost per checkpoint.

3. **Hybrid: light parallel + sequential critique:**
  - A and B independently produce *brief* bullet-level readings.
  - A longer, structured critique pass is then done by a single critic role that sees both and focuses on inconsistencies, unsupported claims, and criteria–data mismatches.

4. **Same model vs different models:**
  - Using the same base model with different prompts reduces model-level confounders but may encourage collusion via shared inductive biases.
  - Using different model families or sizes (e.g., a larger, slower interpretive model and a smaller, more literal critic) introduces cost and alignment tradeoffs but may diversify error modes.[^30][^10]

### Failure modes and design responses

Empirical and conceptual work highlights several failure modes for debate/critique architectures:[^8][^11][^28]

- **Collusion / shared narrative:** Models converge on a plausible but unsupported story.
  - Mitigation: train prompts to reward pointing out unsupported claims; use random-data and scrambled-data baselines; occasionally reverse roles (critic defends, advocate attacks) to test robustness.

- **Sycophancy:** Critics fail to meaningfully challenge; both agents mirror implicit builder preferences.
  - Mitigation: emphasize in prompts that disagreeing with both the other agent and the builder is rewarded when justified by the data; test against held-out criteria not in the prompt.[^10]

- **Mode collapse:** Different agents effectively become clones (same prompt, same model, similar temperature).
  - Mitigation: enforce orthogonal objectives (e.g., one agent maximizes conservatism/faithfulness, another maximizes hypothesis discovery but must label speculation explicitly).[^31]

- **Pseudo-disagreement:** Agents argue over framing rather than substance, generating the appearance of adversariality without information gain.
  - Mitigation: require both agents to ground claims in specific telemetry features (e.g., "KL spikes at episode 20"), and use a critic or human to collapse purely verbal disagreements.

### Tradeoffs specific to Kind’s commitments

- **Self-opacity and ingredients-only self-modeling:** The mirror must not leak its readings or telemetry back into Io’s training loop in ways that amount to an explicit self-critic; adversarial structures are therefore confined to the builder’s side. Sequential architectures where critique happens offline on stored telemetry fit this well.[^3][^5]

- **Co-design discipline:** Using the same constitution or criteria set for both agents risks reinforcing the builder’s priors; using distinct but related frameworks (e.g., Buddhist phenomenology vs predictive processing vs control theory) can partially counterbalance this.[^5][^7]

- **Cost constraints:** Full multi-round debates at every checkpoint are too expensive; the architecture must rely on episodic, high-confidence passes (e.g., every N episodes or when scalar summaries cross thresholds).[^12][^11]

### Defensible suggestion for Q2

Given these constraints, a *sequential–hybrid* architecture seems most defensible:

- **Primary reader A:** A strong, slower model, prompted to produce a structured, cautious reading with explicit references to telemetry features, clear separation of observations from interpretations, and labels for speculative content.
- **Adversarial critic B:** A second role (same or different model) that sees the digest and A’s reading and is instructed to challenge unsupported claims, propose alternative explanations, and explicitly assess where A’s conclusions might stem from criteria or prompt rather than data.
- **Optional light parallelism:** For particularly important checkpoints, run a brief independent reading from B before seeing A, to detect points where A’s framing may have anchored B.

This keeps costs near 2 passes per checkpoint, aligns with adversarial-collaboration principles, and dovetails with the calibration protocol in Q3.

***

## Q3 — Mirror calibration and co-design discipline together

### What Q3 is asking

Q3 focuses on two intertwined issues:

- **Calibration:** Is the mirror accurately reading the telemetry, as opposed to hallucinating patterns (e.g., reversing entropy direction, over-calling frequency)?
- **Co-design:** Even if the mirror is internally consistent, are its readings tracking structure in Io’s dynamics or just reflecting structure in the prompts, criteria, and builder’s expectations?

It asks how literature on LLM evaluation, hallucination detection, observer effects, garden-of-forking-paths, and contemplative self-deception can inform practical mitigations for a solo builder, including synthetic tests, pre-registration, and adversarial human collaboration.

### Calibration: relevant tools and findings

**LLM-as-judge calibration and faithfulness.** Surveys on LLM-as-judge show that using large models to evaluate other models’ outputs can correlate well with human judgments but is prone to systematic biases and hallucinations, especially when the judge is not tightly grounded in input evidence. Key patterns include:[^16][^30][^10]

- Faithfulness metrics distinguish between factual correctness and grounding in provided context (e.g., whether each claim is entailed by the evidence).[^32][^16]
- Techniques such as requiring span-level citations, entailment checks, and structured outputs (claims plus evidence) reduce hallucinations.[^32][^10]
- Benchmarks like TruthfulQA and HaluEval highlight that hallucinations are not rare edge cases but systematic failure modes; self-consistency and majority voting help but do not eliminate them.[^33][^34]

**Hallucination detection in practice.** Engineering guides for hallucination monitoring emphasize rubric-driven evaluation, tracing from outputs back to inputs, and using secondary LLMs to flag unsupported content. These practices generalize to the mirror setting: each interpretive claim must be linked to concrete telemetry features, and a critic should test whether those links are valid.[^35][^32]

### Co-design and observer effects

**Garden of forking paths and demand characteristics.** Gelman and Loken describe the "garden of forking paths": even without overt p-hacking, researcher degrees of freedom in analyses and inclusion/exclusion criteria make apparently significant findings dubious, because many possible analyses were implicitly available. In psychology, demand characteristics and social-desirability bias can distort participant behavior when they infer the experimenter’s hypotheses, leading to systematic but misleading patterns.[^36][^15][^19]

Transposed to Kind’s setting:

- The builder has many choices in how to summarize telemetry, which episodes to highlight, and which criteria to emphasize at any moment.
- LLM mirrors, trained to be helpful and agreeable, are sensitive to these implicit "demand characteristics" in prompts and examples.[^10][^32]

**Contemplative science and observer self-deception.** Work on phenomenological reports in meditation notes that self-reports can be shaped by doctrinal expectations and teacher–student dynamics; careful protocols (training in reporting, separating experimenter identity from evaluator role, using objective correlates) are needed to make them reliable. This is a close analogue to the builder–mirror relationship: the mirror, like a student, may produce "good Dharma" stories that fit criteria rather than directly tracking data.[^20][^19]

### Practical calibration ingredients for Probe 2

Several practical tools emerge that address both calibration and co-design:

1. **Pre-registered predictions and analysis plans.** Before running mirror passes on a checkpoint, the builder writes down:
  - Which scalar summaries and telemetry features will be inspected.
  - What patterns would be counted as confirmatory, disconfirmatory, or ambiguous for each criterion.
  - Which exploratory analyses are allowed and how their findings will be treated (as hypotheses, not confirmations).

  This reduces the garden-of-forking-paths problem by constraining analytic freedom.[^15]

2. **Evidence-linked outputs.** Require the mirror to output interpretations in a structured form:

  - A set of atomic claims (e.g., 
    - "Policy entropy decreases over episodes 10–20, then stabilizes."
    - "Dream rollouts increasingly favor trajectories exploring region X.")
  - For each claim: explicit references to episode ranges, scalar trends, or dream snippets.

  The critic role then checks whether each claim is entailed by the referenced evidence, using entailment-style prompting and refusing to endorse claims without clear support.[^16][^32][^10]

3. **Random-data and scrambled-baseline tests.** Periodically run the entire mirror pipeline on:

  - Telemetry with time order scrambled.
  - Telemetry with agent–action relations broken (e.g., shuffled actions across episodes) while keeping marginal distributions.
  - Fully synthetic noise matching high-level statistics.

  If the mirror still reports complex, criterion-flavored patterns at similar rates, this is evidence that it is reading its own priors, not the data.[^15][^10]

4. **Synthetic telemetry with known structure.** Construct toy environments or offline-generated telemetry where the structure is known (e.g., engineered KL spikes, controlled entropy ramps, pre-defined exploratory phases), and test whether the mirror recovers that structure.

  - This is analogous to calibration of LLM judges on benchmarks where ground truth is available.[^16][^10]

5. **Stability tests across reseeds and paraphrased prompts.** For a given checkpoint and digest:

  - Run the same mirror prompts with multiple random seeds and sampling parameters.
  - Run paraphrased prompts that preserve criteria but alter wording and order.

  Real structure should produce stable high-level readings; large variation across seeds or prompt framings suggests prompt-induced patterns.[^12][^10]

6. **Lesion tests and env manipulations.** When the mirror claims that a certain substrate component (e.g., recurrence, ensemble-disagreement) or environment feature underlies a pattern:

  - Construct ablation runs where that component is disabled or altered.
  - Check whether the mirror’s readings change in the predicted direction, or whether they adaptively rationalize the new telemetry into the same narrative.

  This mirrors causal analysis in neuroscience and supports or falsifies interpretive hypotheses.[^7][^5]

7. **Human eyeball-helper baselines.** Pre-existing eyeball helpers that produce scalar summaries (e.g., action histograms, entropy trends) can serve as a low-tech ground truth. The mirror’s readings should at least not contradict these simple statistics without explicit justification.

### Solo-builder mitigations and micro adversarial collaboration

A solo builder faces correlated biases with the mirror, but can still implement mitigations inspired by adversarial collaboration and contemplative science:

- **Two-mode builder stance:** Alternate between a "proponent" mode (expecting to find structure) and a "skeptic" mode (actively seeking alternative explanations and nulls), with explicit notes on which mode was active when making decisions.[^24][^26]
- **Externalized reasoning logs:** Keep a running log of analysis decisions, telemetry selections, and prompt revisions, noting rationales and suspected biases. This helps detect co-design drift over time.[^15]
- **Minimal external adversary:** Formally invite a second human (even infrequently) to review predefined snapshots: the raw scalar summaries, the mirror readings, and the builder’s pre-registered expectations. Their role is to flag where conclusions seem underdetermined by data.

The smallest version of human adversarial collaboration is thus an occasional, structured check where a second person scores the mirror’s claims against eyeball-accessible data, using a simple rubric ("clearly supported / ambiguous / unsupported").

### Defensible suggestion for Q3

Probe 2 should treat calibration and co-design discipline as a *single protocol* rather than separate layers. A defensible bundle would include:

- Structured mirror outputs with evidence links.
- A critic role focused on faithfulness and alternative explanations.
- Pre-registered analysis plans and classification of findings as confirmatory vs exploratory.
- Periodic random-data and synthetic-structure tests.
- Stability checks across seeds and paraphrased prompts.
- Occasional human adversarial review, even if only at key milestones.

This does not eliminate co-design risks but makes them more visible and constrains the ways in which mirror and builder can subtly tune each other.

***

## Q4 — Telemetry granularity for the mirror

### What Q4 is asking

Q4 asks at what level of detail to present Io’s telemetry to the mirror, given that full per-step traces are too dense for practical LLM processing and overly compressed digests led to misreads in Probe 1. It also asks how long-context LLM behavior and retrieval-augmented analysis of time series and documents inform choices between decoded dream rollouts, episode-level digests, and mixed strategies.

### Long-context and RAG insights

**Long-context LLM evaluation.** Surveys on long-context language models highlight several regularities:[^9][^37][^12]

- Models often struggle to use information uniformly across very long contexts; recency bias and "lost in the middle" effects are common.
- Hierarchical and retrieval-based approaches (e.g., breaking documents into sections and querying relevant pieces) outperform monolithic, single-context ingestion.[^37][^12]
- Evaluation benchmarks that require recalling key points from long documents (e.g., key-point retrieval) show that explicit key-point extraction improves both performance and interpretability.[^9]

**RAG for time series and structured data.** Work on retrieval-augmented time series forecasting and RAG in general suggests benefits from:

- Storing structured summaries (e.g., statistics over windows, detected motifs) and allowing the model to retrieve specific segments for closer examination.[^13]
- Separating raw data storage from interpretive layers; LLMs query the data through an API-like interface rather than receiving all values at once.[^13]

**Document-grounded reasoning.** Evaluations of document-grounded LLM reasoning (e.g., DocMath-Eval and similar) show that when models must answer questions grounded in long technical documents, structured presentations of relevant snippets combined with explicit reasoning prompts produce better grounded answers than dumping the whole document.[^38]

### Candidate granularity options

1. **Full per-step traces (discarded):**
  - Pros: complete; no loss of structure.
  - Cons: intractable for LLMs at Probe 2 scale; prone to cherry-picking and misreading; exacerbates long-context pathologies.[^12]

2. **Single-run aggregate digests (Probe 1 baseline):**
  - Pros: cheap, human-readable.
  - Cons: prone to over-interpretation; hide local dynamics and multimodality; led to misreads in Probe 1.

3. **Episode-level structured digests:**
  - Each episode summarized by a standard schema: scalar time series (entropy, disagreement, KL), simple distributional summaries (action histograms, state visitation), and a small number of representative dream-rollout snapshots.
  - Pros: balances detail and tractability; supports episode-wise comparisons and tracking.[^9][^13]
  - Cons: still compresses within-episode dynamics; risks bias in selection of "representative" snapshots.

4. **Mixed hierarchical scheme with retrieval:**
  - **Base layer:** For each run, pre-computed scalar time series and summary statistics (min/mean/max, trend slopes) plus metadata (env variants, lesions).
  - **Episode summaries:** For a subset of episodes (e.g., early, mid, late; or selected by change-point detection), store detailed per-step slices and dream rollouts.
  - **Mirror interface:** Give the mirror a high-level digest first (e.g., run-level overview, list of episodes with unusual metrics), then allow it to "request" more detail for specific episodes or time windows.

  This mimics retrieval-augmented analysis: the mirror reasons over an index and selectively zooms into raw data, staying within context limits.[^9][^12][^13]

5. **Dream-rollout-only view:**
  - Provide only decoded dream rollouts as ASCII, treating Io as a dream generator.
  - Pros: directly leverages established interpretability of dream rollouts.[^5]
  - Cons: disconnects from policy and telemetry, making criteria like equanimity hard to ground; risks anthropomorphizing the dreams.

### Tradeoffs against Kind’s commitments

- **Self-opacity:** All telemetry and dream data are already builder-side; granularity choices do not directly affect Io’s architecture but modulate how much interpretive freedom the mirror has.[^3]
- **Co-design discipline:** Overly curated digests risk embedding the builder’s expectations; more mechanical, standardized digests and retrieval procedures reduce this.[^15]
- **Criteria binding:** Equanimity and reflexive-attention analogues require time-resolved patterns (e.g., perturbation and recovery), which are better captured by episode-level or hierarchical schemes than single aggregates.[^22][^1][^5]

### Defensible suggestion for Q4

Probe 2 should adopt a **hierarchical, mixed granularity** approach:

- **Standardized run-level digest:** Auto-generated summary with:
  - Episode-wise scalar metrics (entropy, disagreement, KL trends). 
  - Flags for episodes with large metric changes or anomalies.

- **Episode-level mini-digests:** For flagged episodes and a small uniform sample, include:
  - Short sequences of per-step scalar values.
  - One or two dream-rollout ASCII snapshots centered on interesting transitions.

- **Retrieval-style interaction:** Structure prompts so the mirror first reads the run-level digest, then is explicitly invited to request more detail on specific episodes or time windows (which the builder can provide as additional context in follow-up calls).

This balances context limits, reduces selection bias, and preserves enough temporal structure for criteria binding, while still being feasible under cost constraints.[^12][^9][^13]

***

## Q5 — Operational meaning of Probe 2 success

### What Q5 is asking

Q5 asks how to define success for Probe 2 in operational terms, given that success is not a metric leaderboard but a calibrated interpretive process that can track structure over time and surface builder-undesignated patterns without collapsing into CA-MAS-style metric proliferation. It seeks minimal deliverables, criteria for credible novelty, and tests that the calibration protocol should expose.

### Lessons from evaluation and forecasting cultures

**Calibration culture (forecasting tournaments).** In Tetlock-style tournaments, success is defined not by narrative persuasion but by probabilistic accuracy across many questions, measured with scoring rules and calibration plots. Translated to the mirror setting, this suggests that success should involve demonstrable alignment between mirror-read predictions and subsequent observed telemetry in new runs or manipulated environments.[^24]

**LLM evaluation and faithfulness.** In LLM evaluation, success is increasingly framed in terms of:

- Faithfulness to given context (no hallucinations).[^10][^16]
- Robustness across paraphrased prompts and seeds.[^10]
- Ability to generalize from calibration tasks (with known ground truth) to new ones.[^33][^16]

**Contemplative science and neurophenomenology.** Successful protocols do not take self-reports at face value but require triangulation between subjective reports, behavioral measures, and physiological/neural correlates. By analogy, mirror readings should be triangulated with simple scalar summaries and controlled perturbations.[^20][^19]

### Minimum-viable Probe 2 deliverable

Operational success for Probe 2 can be framed around three outputs:

1. **A calibrated mirror pipeline.**
  - Documented architecture (from Q2), including reader and critic roles, prompting templates, and procedures for synthetic/random-data tests.
  - Evidence that on synthetic telemetry with known structure and on random/scrambled baselines, the mirror behaves appropriately (detects known structures; produces low-structure outputs on noise).[^16][^10][^13]

2. **N reproducible readings.**
  - A small set (e.g., 3–7) of specific, operationalized readings about Io’s dynamics that:
    - Are expressed in terms of concrete telemetry patterns (e.g., "policy entropy decreases over episodes and partially recovers after perturbations of type X").
    - Have been observed across multiple runs and seeds.
    - Survive prompt paraphrasing and model-seed variation, within reasonable variability.

3. **At least one credible novelty case.**
  - A case where the mirror surfaces a pattern that was not pre-designated by the builder (e.g., an unexpected phase transition, a coupling between dream-rollout structure and exploration behavior), which is then *tested* via environment or substrate manipulations and found to track in predictable ways.[^8][^13]

The crucial point is that *the novelty claim is treated as a hypothesis and subjected to falsification attempts*, not as self-confirming evidence.

### Passing and failing tests for the calibration protocol

**Passing indicators:**

- On synthetic data with planted patterns, the mirror correctly identifies those patterns at rates substantially above chance, and critic checks confirm faithfulness.[^16][^10]
- On random/scrambled baselines, the mirror produces low-complexity outputs and explicitly notes lack of structure, rather than constructing elaborate narratives.[^15][^10]
- For real Io runs, mirror readings about basic statistics (entropy direction, frequency of behaviors) align with eyeball-helper and scalar summaries.
- Key readings are robust to prompt paraphrase and seed variation, with variations appropriately labeled as uncertainty.

**Failing indicators:**

- Persistent hallucination on baselines: the mirror continues to find complex, criterion-flavored patterns in scrambled or random telemetry.
- Inability to detect known structure in synthetic data.
- High sensitivity of core conclusions to prompt wording, with no stable backbone of agreement.
- Gradual drift of criteria or prompts in ways that retrospectively fit Io’s behavior (co-design) without documented external reasons.

### Tracking over time

Tracking should be defined in terms of *time series of interpretive claims*, not just raw metrics:

- For each reproducible reading (e.g., "equanimity-like recovery pattern after perturbation type A"), maintain a log across runs:
  - Whether the pattern was observed.
  - How strongly (qualitative or simple scalar scores).
  - Under what environment or lesion conditions.

- Visualizations (outside this research report) can help but are not the success criterion; the key is that the reading is:
  - Specific enough to be testable.
  - Stable enough to track across interventions.
  - Open to revision if falsified.

The trap to avoid is defining success as "the mirror said something the builder finds compelling". Success is instead: "the mirror produced claims that could be wrong, and when challenged by controlled tests, some survived and some were revised".

### Defensible suggestion for Q5

Probe 2 should define success as achieving:

- A documented, partially validated mirror pipeline with explicit calibration tests and known limitations.
- A small number of specific, testable readings about Io’s dynamics that show stability and responsiveness to interventions.
- At least one mirrored pattern that qualifies as a plausible, initially undesignated novelty and has undergone at least one falsification attempt.

If these are achieved, regardless of whether any pattern is ultimately labeled "reflexive attention" or "equanimity", Probe 2 has succeeded in its deeper aim: establishing an interpretive methodology that can be iterated in later probes.

***

## Synthesis — a coherent Probe 2 design

### Coherent picture across Q1–Q5

Taken together, the answers suggest the following shape for Probe 2:

- **Criteria operationalization (Q1):**
  - Treat reflexive attention and equanimity as *guiding metaphors* for functional motifs: precision allocation to self-dynamics and balanced responsiveness to perturbations.[^1][^22][^5]
  - Explicitly descope second-order volition for now.[^23][^2]
  - Operationalize these motifs in terms of simple, reproducible telemetry features (entropy/disagreement trends, perturbation–recovery dynamics, latent-usage patterns) while resisting scalar reification.

- **Adversarial mirror architecture (Q2):**
  - Use a sequential reader–critic setup, with optional brief parallel readings on important checkpoints.[^11][^31][^8]
  - Ground both roles in frozen criteria but with slightly different emphases or frameworks to reduce shared bias.

- **Calibration and co-design discipline (Q3):**
  - Embed calibration tools directly in the architecture: evidence-linked claims, critic faithfulness checks, synthetic and random-data baselines, pre-registered analysis plans, and occasional human adversarial review.[^10][^16][^15][^13]

- **Telemetry granularity (Q4):**
  - Provide hierarchical, standardized digests with episode-level summaries and retrieval-like access to raw slices and dream rollouts.[^9][^12][^13]

- **Operational success (Q5):**
  - Define Probe 2 success in terms of a calibrated pipeline and a small set of stable, testable readings plus at least one novelty case subjected to falsification.[^24][^16][^10]

### Tensions and tradeoffs

Several tensions arise:

- **Novelty vs calibration.** Strong calibration pressures (e.g., synthetic baselines, strict pre-registration) may discourage the mirror from proposing speculative patterns, reducing novelty surfacing. Conversely, encouraging hypothesis generation risks increased confabulation.[^16][^10]
  - Possible balance: separate exploratory and confirmatory phases, with clear labeling; allow more speculative readings in exploratory passes but require stricter calibration before adopting them as tracked readings.

- **Digest thinness vs criteria binding.** Thinner digests simplify calibration and reduce context issues but may fail to capture the fine-grained dynamics needed to meaningfully approximate equanimity or reflexive attention.[^22][^1][^5]
  - The hierarchical scheme is a compromise: coarse summaries for stability, with targeted zoom-ins where criteria are most applicable.

- **Co-design pressure vs frozen criteria.** Freezing criteria and prompts helps avoid drift but may eventually misalign interpretive tools from Io’s evolving complexity. Conversely, adapting criteria risks unconscious tuning.
  - The project’s commitment to updating criteria only against external learning, with written justifications, is a good discipline; external references (e.g., new contemplative-science findings) can serve as legitimate triggers for updates.[^21][^22]

- **Solo-builder biases vs need for external critique.** A single builder cannot fully escape shared priors with the mirror; even small doses of external human adversarial collaboration are valuable but may be hard to arrange.
  - At minimum, exposing synthetic baselines and pre-registered predictions to an external reader (even occasionally) can provide a reality check.

### Gaps surfaced by the research

- **Lack of direct RSSM–phenomenology mapping.** The literature does not offer ready-made mappings from Buddhist phenomenology or Frankfurtian volition to RSSM telemetry; all proposed correspondences are functional analogies at best.
- **Limited empirical work on LLM-based third-person phenomenology.** While there is work on LLMs judging other models and on phenomenological self-report, there is little on LLMs acting as heterophenomenological interpreters of non-linguistic agents.
- **Sparse work on co-design in solo AI research.** Most co-design and observer-effect discussions come from human experimental psychology and contemplative science, not AI builder–model ecosystems; adaptation here is necessarily judgment-laden.

These gaps justify Probe 2’s stance as exploratory and methodological rather than definitive.

***

## Targeted reading list (up to 8 items)

1. **Vago & Silbersweig (2014), "Moving Beyond Mindfulness: Defining Equanimity as an Outcome Measure"** — Clarifies equanimity as distinct from mindfulness and proposes operational measures; useful for designing equanimity-like telemetry perturbation and recovery probes.[^1][^21][^22]

2. **Jan Puc ("In Defence of Bare Attention") and Nyanaponika Thera on bare attention** — Phenomenological analyses of mindfulness and reflexive attention; highlight ambiguities and guardrails against over-reification when mapping to functional architectures.[^18][^4][^17]

3. **Frankfurt (1971), "Freedom of the Will and the Concept of a Person"** — Canonical statement of second-order volition; important for justifying why this criterion is likely inapplicable at Io’s current scale and architecture.[^2][^23]

4. **Seth (2013) and related predictive-processing work on interoception and selfhood** — Provides a predictive-coding framing of attention, interoception, and self, offering analogies for how reflexive-like dynamics might appear in RSSM latents and precision allocation.[^7][^5]

5. **Irving et al. (2018), "AI Safety via Debate" and subsequent MAD/SELENE work** — Core references for debate-based evaluation and multi-agent LLM critique; inform the design of sequential and hybrid reader–critic architectures and their failure modes.[^27][^28][^11][^8]

6. **Zeng et al. and follow-up surveys on LLM-as-a-judge and faithfulness metrics** — Systematic overview of how to build and evaluate LLM evaluators, including hallucination detection, evidence-linking, and robustness tests; directly relevant to mirror calibration.[^30][^10][^16]

7. **Gelman & Loken (2013), "The Garden of Forking Paths" and work on demand characteristics** — Essential for understanding how analytic flexibility and experimenter expectations create spurious patterns; provides conceptual backbone for pre-registration and random-data baselines.[^36][^15]

8. **Lutz et al. and related contemplative neuroscience/neurophenomenology work (e.g., Vago; Garrison et al.)** — Show how first-person reports are combined with neural and behavioral data, and how observer/teacher effects are mitigated; useful analogies for builder–mirror–Io triangulation.[^19][^20][^22]

---

## References

1. [Moving beyond Mindfulness: Defining Equanimity as an Outcome ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC4350240/) - Equanimity can also indirectly promote cognitive flexibility in response to new and unexpected condi...

2. [[PDF] Free Will II: Frankfurt and Wolf - Richard Holton](https://rjh221.user.srcf.net/courses/freewill/Lecture2.pdf) - Frankfurt identifies a person's will with their effective desires. Second: Distinguish, amongst your...

3. [Heterophenomenology - Wikipedia](https://en.wikipedia.org/wiki/Heterophenomenology)

4. [[PDF] Bare Attention - PhilArchive](https://philarchive.org/archive/PUCIDO) - At one point, it marks the quality of reflexive attention, which allows Nyanaponika to define bare a...

5. [Predictive codes of interoception, emotion, and the self - PMC - NIH](https://pmc.ncbi.nlm.nih.gov/articles/PMC3940887/) - Seth proposes a predictive coding (PC) model of interoception that involves a free-energy based expl...

6. [Moving beyond mindfulness: defining equanimity as an outcome ...](https://cris.maastrichtuniversity.nl/en/publications/moving-beyond-mindfulness-defining-equanimity-as-an-outcome-measu/)

7. [An Interoceptive Predictive Coding Model of Conscious Presence](https://pmc.ncbi.nlm.nih.gov/articles/PMC3254200/) - Applied to interoception, predictive coding implies that subjective feeling states are determined by...

8. [[1805.00899] AI safety via debate - arXiv](https://arxiv.org/abs/1805.00899) - To make AI systems broadly useful for challenging real-world tasks, we need them to learn complex hu...

9. [Evaluating Long-Context & Long-Form Retrieval-Augmented ... - arXiv](https://arxiv.org/html/2410.23000v3) - KPR focuses on evaluating the LLM's ability to effectively exploit the retrieved documents by measur...

10. [A Survey on LLM-as-a-Judge - arXiv](https://arxiv.org/html/2411.15594v4) - In (Zeng et al., 2023) , a meta-evaluation benchmark was created to evaluate the effectiveness of LL...

11. [Should we be going MAD? A Look at Multi-Agent Debate ...](https://arxiv.org/html/2311.17371v3)

12. [A Comprehensive Survey on Long Context Language ...](https://arxiv.org/abs/2503.17407) - Efficient processing of long contexts has been a persistent pursuit in Natural Language Processing. ...

13. [[2411.08249] Retrieval Augmented Time Series Forecasting - arXiv](https://arxiv.org/abs/2411.08249) - Retrieval-augmented generation (RAG) is a central component of modern LLM systems, particularly in s...

14. [[PDF] Constitutional AI: Harmlessness from AI Feedback - Anthropic](https://www-cdn.anthropic.com/7512771452629584566b6303311496c262da1006/Anthropic_ConstitutionalAI_v2.pdf)

15. [[PDF] The garden of forking paths: Why multiple comparisons can be a ...](https://sites.stat.columbia.edu/gelman/research/unpublished/p_hacking.pdf)

16. [A review of faithfulness metrics for hallucination assessment ... - arXiv](https://arxiv.org/abs/2501.00269) - This review examines the means with which faithfulness has been evaluated across open-ended summariz...

17. [Jan Puc](https://philpapers.org/archive/PUCIDO.pdf)

18. [Satipathana: Bare Attention Of Mindfulness-Nyanaponika Thera](https://integralmusings.aurosociety.org/satipathana-bare-attention-of-mindfulness-nyanaponika-thera/) - Bare Attention is the clear and single-minded awareness of what actually happens to us and in us, at...

19. [Training novice practitioners to reliably report their meditation ... - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6374282/) - Novice meditators can be trained to report their phenomenological experience. Self-reports are assoc...

20. [Effortless awareness: using real time neurofeedback to investigate correlates of posterior cingulate cortex activity in meditators' self-report](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2013.00440/pdf)

21. [1 23](https://davidvago.bwh.harvard.edu/wp-content/uploads/2014/07/equanimity_2014.pdf)

22. [Defining Equanimity as an Outcome Measure in Meditation and ...](https://pubmed.ncbi.nlm.nih.gov/25750687/) - In light of a growing interest in contemplative practices such as meditation, the emerging field of ...

23. [Freedom of the Will and the Concept of a Person - jstor](https://www.jstor.org/stable/2024717) - 5 Creatures with second-order desires but no second-order volitions differ sig- nificantly from brut...

24. [Forecasting Tournaments - Philip E. Tetlock, Barbara A. Mellers ...](https://journals.sagepub.com/doi/10.1177/0963721414534257) - Forecasting tournaments are level-playing-field competitions that reveal which individuals, teams, o...

25. [‪Barbara A Mellers‬ - ‪Google Scholar‬](https://scholar.google.ca/citations?user=EzvAmBwAAAAJ&hl=vi) - An exercise in adversarial collaboration. B ... Psychological strategies for winning a geopolitical ...

26. [[PDF] 2001---do-frequency-representations-eliminate-conjunction-effects.pdf](https://faculty.wharton.upenn.edu/wp-content/uploads/2015/07/2001---do-frequency-representations-eliminate-conjunction-effects.pdf)

27. [[PDF] arXiv:1805.00899v2 [stat.ML] 22 Oct 2018](https://r.jordan.im/download/technology/irving2018.pdf)

28. [[PDF] Selective and Evidence-Weighted LLM Debating for Efficient and ...](https://aclanthology.org/2026.eacl-industry.7.pdf)

29. [On 'Constitutional' AI](https://digi-con.org/on-constitutional-ai/) - Amidst the global discussion surrounding the social harms generated by large language models (LLMs),...

30. [A survey on LLM-as-a-judge - ScienceDirect.com](https://www.sciencedirect.com/science/article/pii/S2666675825004564) - This survey provides a systematic framework and formal definition for building reliable LLM-as-a-jud...

31. [Full Python Code: 3-Agent...](https://engineersofai.com/docs/agentic-ai/multi-agent-systems/debate-and-critique-patterns) - How LLMs critiquing each other improves quality: verifier/critic patterns, multi-agent debate, ensem...

32. [Measuring LLM Hallucinations: The Metrics That Actually Matter for ...](https://www.getmaxim.ai/articles/measuring-llm-hallucinations-the-metrics-that-actually-matter-for-reliable-ai-apps/) - LLM hallucinations aren’t random; they’re measurable. This guide breaks down six core metrics and ex...

33. [HaluEval and TruthfulQA Benchmarks - Emergent Mind](https://www.emergentmind.com/topics/halueval-and-truthfulqa) - HaluEval and TruthfulQA benchmark hallucinations in LLMs by assessing factual accuracy through syste...

34. [Evaluating LLM Hallucination with TruthfulQA - Jongmin Mun](https://jong-min.org/blog/2026/homework-evaluating-llm-hallucination-with-truthfulqa/) - HW2 for USC ISE-547 2026 spring

35. [Detecting hallucinations with LLM-as-a-judge: Prompt ... - Datadog](https://www.datadoghq.com/blog/ai/llm-hallucination-detection/) - Discover how Datadog uses LLM-as-a-judge, structured output, and prompt engineering to detect halluc...

36. [Demand characteristics](https://en.wikipedia.org/wiki/Demand_characteristics)

37. [A Comprehensive Survey on Long Context Language ...](https://github.com/LCLM-Horizon/A-Comprehensive-Survey-For-Long-Context-Language-Modeling) - A Comprehensive Survey on Long Context Language Modeling - LCLM-Horizon/A-Comprehensive-Survey-For-L...

38. [DocMath-Eval: Evaluating Math Reasoning Capabilities of LLMs in ...](https://aclanthology.org/2024.acl-long.852/) - This paper introduces DocMath-Eval, a comprehensive benchmark specifically designed to evaluate the ...

