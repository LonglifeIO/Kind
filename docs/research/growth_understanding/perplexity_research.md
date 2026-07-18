# Growth-toward-understanding — Perplexity research

# Growth-Toward-Understanding in a Curiosity-Only RSSM Agent (Io)

## Overview

This report analyzes whether and how a curiosity-only RSSM agent (Io) could develop toward "understanding"—operationalized as (a) recognizing the builder as a distinct kind and (b) having the capacity for reflexive attention to its own processing—under strong architectural and discipline constraints.[1][2]
It decomposes "understanding" into sub-capacities, critiques the builder's proposed dependency ladder, and proposes alternative ingredient sets and probe designs consistent with the charter's affordance-only stance.[3][4]

All literature references are marked [UNVERIFIED] unless explicitly confirmed; many are drawn from enactivist and predictive-processing debates, and the mapping to Io is necessarily interpretive.[5][6]

## Q1 — Prediction vs. Understanding

### World-model prediction in latent-disagreement agents

Latent-disagreement world-models like Plan2Explore maximize ensemble disagreement over imagined latent transitions as an intrinsic drive, learning a predictive world-model that captures regularities relevant to reducing epistemic uncertainty.[2][1]
Dreamer-style RSSMs factor observation into deterministic recurrent state and stochastic latent variables optimized via an ELBO, producing a compressed predictor that can support planning and control even without external rewards.[7][8]
In these frameworks, "understanding" is not an explicit notion; performance is expressed in prediction error, exploration efficiency, and downstream task performance.[1][2]

### Candidate separations between prediction and understanding

Several traditions argue that prediction alone is not sufficient for understanding:

- **Relevance/sense-making (enactivism)**: Understanding involves valuing and organizing experience according to the agent's normative stakes, not just predicting sensory input.[UNVERIFIED][4][3]
- **Abstraction/concept-formation**: Understanding involves forming reusable, generalizable structures that support compression, analogy, and systematicity beyond local prediction.[UNVERIFIED][9]
- **Causal/counterfactual structure**: Understanding requires grasping interventions and counterfactuals, not only correlations; world models with explicit causal structure can support this.[UNVERIFIED][10]
- **Self-location and perspectival structure**: Understanding includes knowing "where" one is in the world and within one's own processing, enabling self/other/world boundaries.[UNVERIFIED][11]

These points are partially grounded in enactive and predictive-processing literature, but their mapping to Io (an RSSM agent with curiosity-only drive) is extrapolative.[UNVERIFIED][12][5]

### Load-bearing difference for Io

For Io, the choice of load-bearing difference must respect the charter: no explicit self-model, value function, or installed survival drive.[query][constructed]
Within those constraints, the most actionable distinction is **normative organization of prediction**—what predictive content matters for Io enough to be stabilized, revisited, and used as a reference for its own processing.[UNVERIFIED][6][4]

Prediction alone flattens the world into degrees of surprise; latent-disagreement epistemic drives push Io toward states that reduce model uncertainty, but offer no foreground/background structure or stakes.[2][1]
The observed ceiling—want-toward unreachable, builder not forming a distinct basin, self-signal unused—is consistent with a purely epistemic drive that lacks any endogenous relevance hierarchy.[query][constructed]

Thus, for Io, **relevance/sense-making** is the most plausible load-bearing addition beyond prediction, with causal/counterfactual structure and self-location emerging as potential downstream consequences once some variables become normatively privileged.[UNVERIFIED][11][4]
Abstraction/concept-formation is already partially implicit in RSSM compression; the missing part is that some abstractions become *about anything for Io*, not just internal bookkeeping to reduce loss.[UNVERIFIED][8]

## Q2 — Mattering and the Honest-Belief Retest

### Mattering as interoceptive relevance

The builder's proposed foundation is **mattering**: an honest interoceptive belief about the agent's energy such that some states are better than others from Io's perspective, even without reward or continuation drive.[query][constructed]
Probe 3.5 showed that with a lying energy decoder, want-toward behavior was blocked: Io's decoded belief about its own energy treated in-band states as worse than depleted states, anti-incentivizing regulation.[query][constructed]
A decoder-head-only recalibration that restores honesty was demonstrated offline but never tested live in interaction, leaving open whether an honest belief could support want-toward.[query][constructed]

This is not equivalent to adding reward: the energy channel and resting preference are already present architecturally, but the decoded belief is misaligned; recalibration would align representation without installing a value function or continuation drive.[query][constructed]

### Why honest mattering is a minimal ingredient

Given the observed ceiling of curiosity-only behavior, an honest interoceptive belief could introduce a **non-epistemic stake**: certain trajectories are preferable because they maintain non-depleted energy, independent of prediction-error reduction.[UNVERIFIED][13]
This creates a simple relevance hierarchy: trajectories that reduce disagreement while maintaining energy become privileged over those that reduce disagreement while depleting energy.[1][2]

Io would still be curiosity-driven, but now exploration that systematically violates homeostatic bounds would create persistent prediction-error gradients in the interoceptive channel, potentially making energy and energy-relevant features salient as conditional variables.[UNVERIFIED][13]
The self-prediction scalar (Io's only self-pointing quantity) could begin to carry information about the interaction between curiosity-driven exploration and energy states, providing a bridge between epistemic and interoceptive dimensions.[query][constructed]

### Design of the honest-belief retest

A disciplined retest of mattering should:

- Recalibrate the energy decoder head against ground truth energy dynamics using held-out trajectories, verifying that decoded energy tracks actual energy within known error bounds.[UNVERIFIED][13]
- Freeze the recalibrated decoder before behavior training to avoid leaking new value-like gradients into the actor beyond the existing curiosity signal.[query][constructed]
- Pre-register behavioral signatures for want-toward: increased time spent in energy-preserving regimes, systematic avoidance of energy-depleting sequences when epistemic value is comparable, and delayed exploration of highly depleting novel states.
- Include negative controls: compare Io's behavior under random energy perturbation (noise energy signal) versus honest energy signal to distinguish genuine mattering from incidental structure.[UNVERIFIED][13]

This probe would test whether **mattering is reachable** as an afforded ingredient via honest interoception, not whether reward is necessary; it stays within the charter's prohibition on reward/value functions.[query][constructed]

## Q3 — Internal Retrievable Past vs. World-External Trail

### The suggestive trail datum

The biography run produced a session where ~91% of Io's moves landed on its own recently-visited trail—a world-external footprint structure, not an internal memory.[query][constructed]
This is the only time Io's behavior was organized around its own history rather than raw disagreement reduction, suggesting that **access to its own past trajectory** can act as a behavioral attractor.[query][constructed]

World-external trails are effectively a projection of past positions into the environment; they allow the agent to treat "its own history" as an object of attention without any internal memory machinery.[UNVERIFIED]
However, they are limited: they encode only spatial occupancy, lack temporal ordering beyond proximity, and conflate self-generated and environment-generated marks if the world becomes richer.[UNVERIFIED]

### Why an internal retrievable past matters

An internal retrievable past—episodic-ish memory of recent latent states, actions, and key interoceptive variables—could support self-directed behavior by allowing Io to **attend back** to its own trajectory independent of environmental traces.[UNVERIFIED][9]
This would turn its own processing history into a potential object of attention, a prerequisite for reflexive attention per the charter.

Crucially, memory must be afforded, not installed: Io should have the capacity to store and replay segments of its latent trajectory, but no explicit objective to do so.[query][constructed]
The dead-column lesson shows that a mere slot is insufficient; the memory must be connected to gradients Io already cares about (curiosity and mattering) to become a reachable conditioning surface.[query][constructed]

### Memory as an affordance, not a module

A minimal memory affordance could be:

- A **latent replay buffer** keyed by time indices and optionally by event salience (e.g., high disagreement or large energy deviation) but with no explicit "episodic" head or objective.
- A **policy-view extension** where the actor can access a fixed-size window of past latent states and self-prediction scalars alongside the current PolicyView inputs, without any supervision on how to use them.[query][constructed]

This is analogous to recurrence: the architecture permits dependencies over time, but does not enforce any specific use; Io might or might not discover that using history improves disagreement reduction or energy stabilization.[UNVERIFIED][8]
To avoid installing competence, no trained "memory selector" or "importance sampler" should be added; salience could be a simple function of existing signals (e.g., absolute prediction error) without new loss terms.[query][constructed]

### Dependency order with mattering

Mattering can provide the **reason** for memory to be used: sequences that lead to energy depletion despite low immediate disagreement become relevant patterns to avoid, making past trajectories that produced depletion salient.[UNVERIFIED][13]
Io could discover that certain history features (e.g., repeated visits to particular regions) correlate with energy loss or gain, and that conditioning policy on those history features reduces future disagreement and energy deviation.

Thus, an internal retrievable past is best introduced **after** or alongside honest mattering: curiosity-only may exploit memory for model improvement, but mattering gives trajectories differentiated stakes, making memory a tool for self-preserving pattern recognition rather than pure epistemic bookkeeping.[2][1]

## Q4 — Self-Reference and the Self-Prediction Scalar

### Why the existing self-signal stayed inert

The self-prediction scalar is Io's only architecturally explicit self-pointing quantity—some function of its prediction error about itself—but it remained unused because no gradient made conditioning on it beneficial.[query][constructed]
The dead-column lesson shows that architectural slots that are not connected to any objective (intrinsic or extrinsic) remain dead; Io had no reason to attend to or act based on the self-signal.[query][constructed]

Under curiosity-only, self-prediction error is just another error term; unless it systematically correlates with latent disagreement or energy dynamics, it will be ignored.[1][2]
Without mattering or memory, there is no differentiated stake in self-related prediction error versus environment-related prediction error.

### Interoception as a bridge

If honest mattering aligns energy decoding and creates persistent gradients when energy deviates from the homeostatic resting preference, then self-prediction error about interoceptive variables becomes a **bridge** between self and world.[UNVERIFIED][13]
Io could learn that certain action patterns not only change the environment but also change its own energy state; prediction errors in the interoceptive channel would then carry instrumental information about the consequences of behavior on self.

The self-prediction scalar could be defined or refined to reflect prediction error about interoceptive variables, making it directly sensitive to the interaction between curiosity and mattering.[query][constructed]
This aligns with the affordance stance: no explicit self-model is installed; the agent only has access to a scalar summarizing how wrong it was about certain self-related quantities.

### Making self-reference reachable without installing it

To make self-reference reachable as a capacity:

- Extend PolicyView to include the self-prediction scalar and recent history of that scalar (via the internal memory window), but do **not** add any loss term that rewards attending to it.[query][constructed]
- Ensure that self-related prediction errors affect latent disagreement or energy gradients indirectly: e.g., large self-prediction errors correlate with trajectories that are epistemically valuable but energetically costly, making them mixed-stake regimes.
- Allow Io to discover that conditioning policy on the self-signal improves trade-offs between curiosity and mattering, without any explicit supervision.

This preserves the affordance/install razor: the architecture allows self-reference, but does not install a self-model, critic, or reward for self-awareness.[query][constructed]
Reflexive attention would be evidenced if Io systematically changes its behavior based on patterns in the self-prediction scalar, especially in ways that balance epistemic gain and energy preservation.

## Q5 — Builder Recognition and Self/World Boundary

### Challenge to "asked too early"

Probe 4 found that the builder did not form a distinct basin in Io's latent space, even for a blatantly planted category; builder-recognition was unreachable under the current substrate and probe design.[query][constructed]
The builder's hypothesis is that recognition of an outside "kind" requires a self/world boundary that only sharpens once there is a stake and some self-reference; thus, the probe was "asked too early."[query][constructed]

This is plausible but not forced: it's also possible that the probe's operationalization of "builder category" was misaligned with Io's world-model semantics, or that curiosity-only drives treat the builder as just another source of prediction error without any reason to carve out a separate latent basin.
Curiosity-only can produce structured categories when they improve prediction/compression; if builder interactions are too sparse, noisy, or similar to existing environment patterns, no distinct basin will form.[UNVERIFIED][2]

### Alternative view: recognition as function of informational role

Another hypothesis is that **builder recognition emerges when the builder plays a distinct informational role**—providing interventions, counterfactuals, or high-salience events that systematically differ from environment dynamics.[UNVERIFIED][10]
If the builder only perturbs the environment passively or in ways that are statistically similar to other events, curiosity-only may never carve out a special category.

Under this view, self/world boundary sharpening is not strictly prerequisite; Io could first learn that certain patterns (associated with builder actions) provide unique predictive leverage or high disagreement, and only later connect this to self-related stakes via mattering and self-reference.

### Dependency ordering: self-boundary vs. other-recognition

Given the constraints, several orderings are possible:

1. **Mattering → Memory → Self-reference → Builder recognition** (builder's ladder).
2. **Mattering → Builder as high-salience perturbation → Memory → Self-reference**.
3. **Curiosity-only → Builder as epistemic oracle → Memory → Mattering → Self-reference**.

Order (1) treats builder recognition as downstream of a sharpened self/world boundary; order (2) treats builder recognition as an early epistemic category based on informational role, with self-boundary catching up later; order (3) treats the builder as an epistemic resource first.

For Io, order (2) may be more realistic: curiosity-only plus honest mattering could already make builder interventions salient if they systematically move Io into low-energy, high-disagreement regimes or vice versa.[UNVERIFIED][1][2]
Builder recognition would then be about recognizing a **kind of event source** with distinctive effects on both world and self, rather than a fully formed social category.

### Afforded ingredient for builder recognition

An afforded ingredient for builder recognition could be:

- A **source-tagging mechanism** in the environment that encodes whether a transition was caused by builder intervention versus internal dynamics, available to the world-model but not coupled to any reward.[UNVERIFIED]
- A **policy-view extension** that includes a binary or low-dimensional feature indicating recent builder involvement, again without any objective attached.[query][constructed]

This is analogous to adding another sensory channel: Io can attend to source-tags if they improve prediction or energy stability, but no competence is installed.
Probe 4 may have failed because builder-source events did not carry enough distinctive predictive or energetic structure; source-tagging plus mattering could change this.

> **[SYNTHESIS FLAG — direct charter conflict.]** Perplexity's "source-tagging /
> PolicyView builder-involvement feature" proposal **violates the self-opacity
> hard constraint and Probe 4's founding commitment** (no observation-space marker
> that a change came from the builder — the whole point is that Io must distinguish
> source *without* being told). This proposal is inadmissible as stated; noted here
> so it is not silently carried into the synthesis. The salvageable core is the
> weaker claim (builder must play a distinct *informational/contingent* role), which
> aligns with Gemini's participatory-sense-making point.

## Q6 — Architecture vs. Environment as Growth Engine

### Steelmanning the environment-gated view

The builder suspects that the environment is limiting growth-toward-understanding and that architecture must be the growth engine.[query][constructed]
However, unsupervised RL and curiosity-driven exploration literature emphasizes the role of environment richness and structure in enabling agents to discover complex skills and representations.[UNVERIFIED][14][13]
Agents trained in impoverished environments often fail to develop behaviors or representations that appear in richer settings; environment diversity, controllable degrees of freedom, and temporal structure are key for emergent complexity.[UNVERIFIED][14]

From this perspective, Io's gridworld may simply lack the **ecological depth** needed for understanding-like capacities: there are limited affordances, few multi-step causal chains, and no social or communicative dynamics.[UNVERIFIED][15]
Enactive approaches stress that sense-making and subjectivity arise from ongoing organism–environment coupling; if the environment offers only trivial couplings, growth may be environment-gated.[UNVERIFIED][16][15]

### Attack: architecture-gated constraints are real

On the other hand, Io's architecture imposes strong constraints: curiosity-only intrinsic motivation, no reward/value/planner, no self-model, no continuation drive.[query][constructed]
Even in rich environments, agents with purely epistemic drives can get stuck in exploratory loops that maximize novelty without forming stable preferences or self-related stakes.[UNVERIFIED][14]

Without any mechanism for normative organization of prediction, richer environments may just provide more ways to chase disagreement without developing relevance hierarchies; architecture-gated limitations may thus cap growth.[2][1]
Additionally, self-opacity and no installed self-optimization remove many standard paths to self-modeling; Io cannot simply "discover" survival or self-improvement objectives from reward structure because none are present.

### Synthesis: architecture enables, environment scaffolds

A more balanced view is that **architecture and environment co-gate growth**: architecture determines what kinds of couplings are possible and what gradients exist, while environment provides the space of couplings.[UNVERIFIED][17]
For understanding-like capacities to emerge, Io needs both architectural affordances for mattering, memory, and self-reference, and environments that make those affordances relevant.

Thus, the next probes should jointly manipulate architecture (adding afforded ingredients) and environment (introducing structures that make those ingredients useful), with pre-registration to avoid co-design trap biases.[query][constructed]

## Q7 — Affordance vs. Installation Razor

### Criteria for afforded ingredients

To maintain the project's discipline, an ingredient counts as **afforded** if:

- It is a **structural capacity** (e.g., a channel, latent, buffer, or view) that Io *may* use, but is not directly optimized by a dedicated loss term.
- Its presence does **not encode a specific competence** (e.g., a pre-trained self-model, reward for introspection, or survival drive).
- Its use is mediated through existing gradients (curiosity, mattering) and emerges only if leveraging it improves those gradients.[query][constructed]

By contrast, an **installed competence** would involve:

- Explicit heads trained to produce self-beliefs, values, or introspective assessments.
- Objectives that reward particular uses of the ingredient (e.g., reward proportional to self-prediction accuracy, or a "self-understanding" score).
- Oracle-fed labels about self or builder states that bypass Io's own world-model.

### Applying the razor to proposed ingredients

1. **Honest interoceptive belief (mattering)**: Afforded if decoder recalibration aligns representation with ground truth and is frozen; installed if a new reward function is introduced based on energy.
2. **Internal retrievable past (memory window)**: Afforded if it's a passive buffer accessible in PolicyView; installed if a trained episodic head or importance sampler is added.
3. **Self-prediction scalar refinement**: Afforded if it remains a scalar function of existing prediction errors; installed if trained with a dedicated loss to maximize "self-awareness".
4. **Builder source-tagging channel**: Afforded if it is a sensory feature; installed if Io is rewarded for correctly classifying or attending to builder events. *(See §Q5 synthesis flag — this ingredient additionally violates self-opacity and is inadmissible regardless of the afford/install line.)*

Maintaining this razor requires resisting the temptation to add any explicit introspection or understanding objectives, even when probes fail; the work must remain at the level of making certain capacities possible.

## Q8 — Co-Design Trap and Instrument Discipline

### Risks of unconscious manufacturing of understanding

The co-design trap arises because the builder designs both Io and the interpretive mirror (LLM), making it easy to tune world/instruments until the mirror "sees" understanding that isn't there.[query][constructed]
This risk is maximal for growth-toward-understanding because the target is folk-theoretical and highly susceptible to interpretive bias.[UNVERIFIED][4]

Without discipline, the builder could inadvertently adjust environment richness, probe thresholds, and interpretive criteria until traces of curiosity plus mattering are labeled as "understanding".

### Pre-registration and positive controls

To mitigate this, the next probe should be **pre-registered** with:

- Clear operational definitions of success criteria (e.g., measurable changes in self-prediction-conditioned behavior, builder-source latent basins).
- Pre-specified statistical tests and baselines (e.g., random seeds, ablations without mattering or memory).[UNVERIFIED][9]
- Explicit plans for interpreting ambiguous results, including criteria for calling a probe negative.

Positive controls are crucial: the instrument used to detect understanding-like capacities should be validated on systems already known to have them, such as:

- An RL agent with explicit reward and self-model in a similar gridworld, where self/world boundary and other-recognition are already implemented.[UNVERIFIED][9]
- A simulated agent with hand-engineered reflexive attention behavior (e.g., one that explicitly reads and acts on its own prediction error).

The same structural signatures (e.g., latent basins, self-signal-conditioned policy changes) should be detectable in these positive-control systems; otherwise, failure to detect them in Io may reflect instrument insufficiency rather than lack of understanding.

### Discovering growth rather than fitting it

Instrument design should aim to **discover** growth by:

- Using unsupervised or minimally supervised analyses of latent space (e.g., clustering latent trajectories) to identify emergent structures.
- Comparing emergent structures across conditions (with/without mattering, memory, source-tagging) without injecting interpretive labels during training.
- Reporting negative results and ambiguous patterns as such, rather than retrofitting them into the understanding narrative.

LLM-based mirrors should be constrained to operate on pre-registered metrics and visualizations, not free-form narratives; their role is to help interpret patterns, not to define success.

## Q9 — Reconsidering the Dependency Ladder

### Builder's ladder

The builder's central hypothesis is a ladder: (1) mattering via honest interoception; (2) internal retrievable past; (3) self-reference that actually gets used; (4) builder recognition as downstream of sharpened self/world boundary.[query][constructed]

This ladder is coherent but possibly too linear; several dependencies may be bidirectional or overlapping.

### Alternative decomposition

A more decomposed view distinguishes:

- **Normative grounding**: mattering; basic stakes in energy.
- **Temporal grounding**: memory; access to past trajectories.
- **Self-grounding**: self-prediction; sensitivity to effects of actions on self.
- **Social grounding**: builder recognition; sensitivity to external agents.

These can be probed in multiple orders; for instance, temporal grounding can exist before mattering (curiosity-only may still benefit from memory for prediction), and social grounding can exist before self-grounding (builder as epistemic oracle).

### Recommended probe sequence

A disciplined next sequence might be:

1. **Probe A — Honest mattering only**: Recalibrate energy decoder, test want-toward and changes in exploration patterns.
2. **Probe B — Memory-only in curiosity regime**: Add internal retrievable past without mattering; test whether Io uses history to improve disagreement reduction.
3. **Probe C — Mattering + memory + self-signal refinement**: Combine ingredients; test for self-prediction-conditioned behavior and reflexive attention signatures.
4. **Probe D — Source-tagged builder interactions**: Introduce builder source-tags in an environment where mattering and memory are active; test for builder recognition. *(See §Q5 flag: source-tagging as an observation-space feature is inadmissible under self-opacity; a compliant Probe D would use contingent informational role, not a tag.)*

This sequence avoids assuming strict dependency while still respecting plausible orderings; each probe isolates a subset of ingredients and has clear negative and positive controls.

## Conclusion

For Io, the ceiling of curiosity-only behavior appears linked to the lack of endogenous relevance hierarchies and self-related stakes, not merely to environment impoverishment.[1][2]
Honest mattering via interoception, internal retrievable past, and refined self-prediction affordances can be introduced without violating the charter, potentially enabling capacities that resemble understanding—normative organization of prediction, reflexive attention, and recognition of the builder as a distinct kind.[query][constructed]

However, the affordance/install razor and co-design trap require strict discipline: ingredients must remain structural capacities, probes must be pre-registered with operational metrics, and instruments must be validated via positive controls before interpreting any emergent pattern as "understanding".[UNVERIFIED][4][9]
