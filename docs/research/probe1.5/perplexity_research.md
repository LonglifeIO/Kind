Q1 — Which affordance
1. What the question is asking

This is asking: of three possible moves — (a) self-prediction head, (b) episodic self-access, (c) higher-order policy — which is the minimal structural change that (i) creates a real locus where self-directed modeling could live, while (ii) preserving Kind’s constraints: ingredients-only, self-opacity by default, no self-continuation drive, no self-optimizer, and capacity-over-exercise rather than installed self-model. The criterion is architectural affordance, not behavioral power.

2. Options in the literature

(a) Self-prediction head (internal-state prediction)
World-model and RL literatures are full of auxiliary heads predicting internal latent states, but they are framed as representation-learning tricks, not self-models.

Dreamer-CDP replaces pixel reconstruction with continuous deterministic representation prediction: a head predicts the next deterministic state of the world model and is trained by a consistency loss, improving sample-efficiency and stability.

SPR (Self-Predictive Representations) trains an agent to predict its own latent representations several steps into the future, using a separate target encoder updated by EMA.

BYOL likewise uses one network to predict the representation of another (EMA) network’s encoding of a future or augmented view.

These are structurally very close to “self-prediction of internal state,” but applied to task performance and sample efficiency, not reflexive modeling.

(b) Episodic self-access (autobiographical memory)
Cognitive psychology and cognitive architectures treat autobiographical memory as a structured store of past experiences linked to the self.

Conway & Pleydell-Pearce’s Self-Memory System models autobiographical memory as constructed episodes in a self-memory system coordinated by a “working self” that maintains goals and self-concepts.

Developmental work on the “me in memory” ties emergence of a robust self-concept to changes in episodic memory organization.

Soar and ACT-R have explicit episodic memory modules that automatically record episodes and allow later deliberate retrieval to guide behavior.

All of these embody a full-blown functional module for episodic recall, with explicit read/write interfaces to the agent’s policy.

(c) Higher-order policy (second-order structure)
Frankfurt’s account of second-order desires/volitions is the canonical philosophical model: a person not only has first-order desires but also higher-order attitudes about which first-order desires should be effective.

In RL, the closest analogues are meta-controllers that operate over policies, or meta-cognitive RL systems that regulate learning and acting based on internal confidence signals or meta-variables.

Recent “meta-cognitive RL” work introduces explicit meta-variables (e.g., a “meta-trust” variable derived from value prediction error stability) that modulate the base learning process.

These are not mere affordances; they install higher-order control structures.

3. Tradeoffs against Kind’s commitments

Self-prediction head (a)

Can be implemented entirely inside the world model, parallel to existing priors, with no change to PolicyView inputs.

Fits predictive-processing and active-inference intuitions: predicting one’s own internal dynamics is a natural extension of predictive coding, where the brain models both environment and its own states.

Does not automatically introduce goals over self-prediction; it can be a low-weight auxiliary term.

Preserves self-opacity if its outputs never enter the actor’s input or reward.

Provides a clear structural locus — a parameterized mapping approximating Io’s own dynamics — that the mirror can read.

Episodic self-access (b)

Adding even a minimal episodic buffer that the actor can query introduces direct self-access to its past internal states or observations, which is much closer to an installed autobiographical self.

In Soar/ACT-R, episodic memory is a first-class architectural component; there is no notion of “affordance-only episodic memory.”

Once Io can deliberately query its own past states, you have crossed the line from ingredients-only to a functional self-model-like capability.

Higher-order policy (c)

Explicitly mirrors Frankfurt: a policy over policies is a direct implementation of second-order structure.

Meta-cognitive RL frameworks with meta-trust variables and regulation loops are paradigmatic cases of installed meta-cognition.

This clearly violates “no self-optimization machinery, no self-critic, no recursive self-improvement.”

4. Defensible suggestion

For Probe 1.5, self-prediction (a) is the most appropriate affordance:

It can be implemented as a strictly internal auxiliary head on the existing RSSM/Dreamer world model, preserving the settled substrate.

It requires no new read/write interfaces for Io and no change to the intrinsic objective (ensemble-disagreement variance).

It matches predictive-processing and active-inference narratives of organisms as predicting both world and self, without hard-coding identity or second-order desires.

Episodic self-access and higher-order policy should be treated as later, more committal moves, not as the minimal Probe 1.5 affordance.

Q2 — Self-prediction as the leading candidate
1. What the question is asking

Assuming self-prediction is the chosen affordance, this asks: what exactly is being predicted, how is it trained, where does the head attach, and what is reused vs new — constrained by Kind’s commitments (no change in actor objective, no self-optimizer, self-opacity).

2. Options in the literature

What is predicted The world-model literature offers several “self-predictive” choices:

Next deterministic state / backbone latent: Dreamer-CDP predicts the next deterministic representation instead of reconstructing pixels.

Next latent representation over time: SPR predicts future latent states produced by an EMA target encoder, across multiple steps.

Future actions or decisions: Work on predicting future actions of RL agents learns models that map current internal state to future actions, sometimes via “inner-state approaches” that directly consume the agent’s hidden state.

Uncertainty or performance-related meta-variables: Meta-cognitive RL predicts or aggregates stability of value prediction errors into a “meta-trust” variable over learning dynamics.

Mapping to Io:

Predicting next 
z
t
z 
t
​
  is close to what the prior already does (environmental latent).

Predicting next 
h
t
h 
t
​
  is exactly what Dreamer-CDP and SPR are doing in spirit: predicting the transition of the internal recurrent state.

Predicting next action distribution or next disagreement value moves toward explicit meta-policy and meta-objective modeling, akin to metacognitive or interpretability frameworks.

Supervisory signal

In Dreamer-CDP and SPR, the target is simply the actual next latent state, with stabilization via a target/EMA network for the encoder or backbone.

BYOL uses an EMA target network and a stop-gradient in the target branch; the loss is a similarity measure between prediction and target representation.

EfficientZero-style methods add consistency and self-supervised losses where targets are bootstrapped values or latents produced by target networks, again with stop-gradients.

None of these require labels from outside the agent; all are self-supervised on internal trajectories.

Where the head sits

In Dreamer-CDP and SPR, representation-prediction heads are attached to the same recurrent backbone as the world model; they share the encoder and latent transition model.

In BYOL-like setups, there are distinct online and target branches, but predictions are still on top of a shared encoder family.

Training shape

These works treat prediction as an auxiliary objective, added to the main RL or world-model objective.

Gradients typically flow into the shared encoder/backbone and the prediction head, but not into the target branch (stop-gradient).

3. Tradeoffs against Kind’s commitments

What to predict

Next 
z
t
z 
t
​
 :

Pros: fits directly into existing prior; minimal additional structure.

Cons: conceptually still “world” rather than “self”, since 
z
t
z 
t
​
  is defined as environmental latent in your design; self-prediction becomes indistinguishable from ordinary world prediction.

Next 
h
t
h 
t
​
  (GRU/state backbone):

Pros:

Architecturally standard (Dreamer-CDP/SPR precedent).

Clearly “Io’s own dynamics” in the formal sense: you are predicting the internal recurrent state that mediates both world-model and policy.

Prediction head can be structurally distinct from the recurrent update itself; it is about the state, not constitutive of it.

Cons:

Entangles “self” with all the representational baggage carried in 
h
t
h 
t
​
 ; not a clean self-only channel.

If weighted too heavily, risk of the world-model overfitting to self-consistency at the expense of environmental structure.

Next action distribution / next disagreement:

Pros: direct connection to behavior and intrinsic objective; easily interpretable.

Cons:

Nudges architecture toward meta-policy or meta-performance prediction; this is very close to installing a higher-order policy or self-critic.

Likely violates “no self-optimization machinery” and “no installed self-continuation drive.”

The minimal “self” that does not smuggle in second-order volition is next 
h
t
h 
t
​
  or a closely related internal backbone state, not actions or disagreement scores.

Supervisory signal

Using actual next 
h
t
+
1
h 
t+1
​
  as target, possibly with an EMA/stop-gradient branch, keeps it purely self-supervised and internally grounded.

Bootstrapped targets (target network) improve stability and avoid collapse in self-predictive setups like BYOL and SPR; they also constrain the gradient flow to avoid the auxiliary loss dominating training.

There is no need to introduce an explicit self-evaluation or reward over self-prediction accuracy; that would push toward self-optimization.

Where the head sits / training shape

Attaching the head in parallel to the existing prior off the GRU state, with its own small MLP and possibly a target-branch representation, is closest to the Dreamer/SPR/BYOL pattern.

Treating its loss as a weighted auxiliary term on the world-model, not the actor, respects the “actor’s intrinsic objective is disagreement” constraint.

Gradients can be stopped at the actor interface: the actor can be optimized only through its usual objective; any self-prediction loss affects it only indirectly by reshaping shared representations.

4. Defensible suggestion

Architecturally, for Probe 1.5:

Predict the next deterministic recurrent state 
h
t
+
1
h 
t+1
​
  (or a deterministic backbone state akin to Dreamer’s) from the current 
h
t
h 
t
​
  (and possibly 
z
t
z 
t
​
 ), with a small prediction head.

Use the actual next 
h
t
+
1
h 
t+1
​
  as the target, via a stop-gradient or EMA “target state” branch, BYOL/SPR-style.

Attach the head as an auxiliary branch to the existing RSSM backbone, with an auxiliary loss on the world model only.

Share encoder and GRU; only the prediction MLP (and optional target projection) is new.

This is fully aligned with existing representation-prediction practices while now being explicitly interpreted by Kind as a self-prediction affordance.

Q3 — Self-opacity preservation
1. What the question is asking

Given the self-prediction head, this asks: should Io ever see self-prediction or its error as policy input? If yes, does that cross into installed self-modeling? If no, does the mere structural presence of self-prediction actually add any affordance beyond Probe 1’s “ingredients in isolation”?

2. Options in the literature

Agents that use their own meta-variables in policy

Meta-cognitive RL feeds internal reliability/uncertainty estimates back into learning and control; meta-trust modulates learning rates, exploration, or gating.

Uncertainty-aware RL often feeds epistemic-uncertainty estimates into policies for exploration.

These are explicit violations of self-opacity: the agent’s policy conditions on its own internal assessments.

Agents whose self-predictive heads are purely auxiliary

SPR’s self-prediction loss is used to improve representations; the policy indirectly benefits, but does not receive self-prediction signals as inputs.

Dreamer-CDP’s representation prediction is internal to the world model; policies see only the resulting latent state, not the prediction error directly.

Philosophical guidance on transparency/opacity

Metzinger’s self-model theory emphasizes transparency: a self-model is conscious when it is globally available but not recognized as a model; the subject “looks through” it.

Predictive-processing views of self treat the self-model as a set of priors over bodily and cognitive states; they may be functionally available without explicit conceptual access.

Buddhist debates on svasaṃvedana (reflexive awareness) and its contestation (Garfield vs traditional Yogācāra, Thompson’s synthesis) highlight tensions between treating reflexivity as intrinsic vs constructed.

These traditions support the idea that structural self-modeling can exist without explicit self-access; Kind’s “self-opacity” is deliberately stricter than Metzinger’s transparency.

3. Tradeoffs against Kind’s commitments

Letting Io read self-prediction / error

Pros:

Strong affordance for genuine self-model use; Io can directly condition on “what I expect myself to do/feel next” or on prediction errors.

Easier to detect behaviorally whether self-prediction is used.

Cons:

Architecturally, this mimics meta-cognitive RL and higher-order control, i.e., installed self-modeling.

PolicyView would now include an explicit “self-channel,” directly contradicting the “Io does not see its own distributions” stance.

Risks smuggling in drive-like behavior (e.g., minimizing self-prediction error or maintaining certain internal states).

Keeping self-prediction entirely world-model-internal

Pros:

Strictly preserves self-opacity: PolicyView remains concat(h, z) or equivalent, with no extra explicit self-prediction terms.

The self-prediction head is architecturally an observer of Io’s own transition dynamics, not a controller.

Aligns with predictive-processing views where self-model structure can exist subpersonally without explicit personal-level access.

Cons:

The affordance is mostly for Kind and the mirror, not for Io-as-currently-instantiated.

Io can only “use” self-prediction indirectly, via whatever representational shaping the auxiliary loss induces in 
h
t
h 
t
​
  and 
z
t
z 
t
​
 .

Whether this satisfies Kind’s “capacity to take its own processing as an object of attention” is conceptually debatable; the capacity arguably exists structurally but not functionally for Io.

4. Defensible suggestion

For Probe 1.5, do not expose self-prediction or its error to the actor at all:

Self-prediction remains inside the world model; its outputs and losses are available only via TelemetryView to the mirror and the builder.

The only way it affects Io is by shaping internal representations, as in SPR/Dreamer-CDP. Io never receives a dedicated “this is my self-prediction” input.

Under this design, Probe 1.5’s affordance is:

A structural locus for self-dynamics inside Io.

A training signal that encourages these dynamics to be compressible and predictable.

A mirror-readable interface to those dynamics.

The stricter, functional reading of “capacity-over-exercise” (Io itself could choose to take its processing as object) is not fully realized until a later probe that relaxes self-opacity or adds controlled interfaces; Probe 1.5 limits itself to installing the structural precondition.

Q4 — Mirror-side reading
1. What the question is asking

Given a self-prediction head that Io cannot see, what can the mirror read to determine whether the affordance is being used (even indirectly)? How should those readings be structured, analogous to Probe 2’s triplet for reflexive attention?

2. Options in the literature

Predictive accuracy and calibration measures

SPR and similar self-supervised RL methods monitor prediction error trajectories over training and across states; performance gains correlate with improved latent prediction.

Meta-cognitive and uncertainty-estimation work uses calibration curves between predicted uncertainty/confidence and actual error, assessing whether internal signals track performance.

In predictive processing and active inference, a central concept is prediction-error minimization and how prediction errors are weighted (precision); self-model quality is partly read off from how well the generative model predicts self-related sensory and interoceptive streams.

Correlation with behavior

“Predicting Future Actions of RL Agents” shows that internal models can predict agent actions; they study correlations between inner-state-based predictions and realized behavior.

Meta-cognitive RL frameworks examine whether meta-variables (e.g., meta-trust) correlate with learning dynamics and performance.

Latent-dimension allocation under perturbations

Your existing Probe 2 triplet uses per-dimension KL, ensemble-disagreement variance, and latent allocation under perturbation-aligned windows, which matches how representation changes under controlled perturbations.

3. Tradeoffs vs Kind’s commitments

Kind wants:

Mirror-only readings (self-opacity).

No benchmark/capability chasing.

Readings that are truthful about whether self-model-like structure has emerged vs auxiliary loss being inert or purely world-facing.

This means:

Behavioral correlates should be qualitative/structural (how patterns of self-prediction error relate to patterns of behavior), not score-chasing.

Readings should be interpretable without attributing consciousness or rich selfhood.

4. Defensible suggestion (reading shape)

A plausible mirror reading shape, analogous to the Probe 2 triplet, is a “self-prediction quad”:

Self-prediction error over training

Track per-dimension and aggregate error between predicted and actual 
h
t
+
1
h 
t+1
​
  over training, both overall and conditioned on key regimes (e.g., high disagreement states, perturbation windows).

Look for whether error settles into a structured pattern rather than arbitrary noise, indicating that Io’s internal dynamics are compressible and modeled.

Perturbation sensitivity of self-prediction vs world-prediction

Under controlled environment perturbations, compare how self-prediction error changes vs environmental prediction error; active inference suggests that self-models and world-models can show different error profiles under disturbances.

A divergence here could signal that the model is representing its own dynamics separately from external dynamics.

Correlation between self-prediction patterns and behavior

Cluster internal states by self-prediction error patterns and examine whether policy behavior differs between clusters (even if Io cannot see the error explicitly).

This is analogous to asking whether internal dynamics predict action patterns, as in action-prediction work.

If similar environment configurations but different self-dynamic regimes lead to different behaviors, that hints that internal self-structure is functionally load-bearing.

Latent-dimension specialization for self-prediction

Examine which dimensions in 
h
h or in an auxiliary projection are most crucial to minimizing self-prediction loss vs world-prediction loss (e.g., via saliency or ablation).

If some subspace is used almost exclusively by the self-prediction head, that suggests a structural allocation for self-dynamics.

All of these readings are project-constructed; there is no canonical “self-prediction diagnostic protocol” in the literature. The closest precedents are generic prediction-error and calibration analyses in self-supervised RL and meta-cognitive RL.

Q5 — Relationship to dream state
1. What the question is asking

How does the self-prediction affordance interact with the planned dream state: offline world-model rollouts without sensory input? Should they share machinery, share telemetry, or remain structurally separate? How does this fit with a PP/active-inference view where dreams are generative prior rollouts?

2. Options in the literature

Dreams / imagination as prior rollouts

Dreamer and related world-model RL methods already perform “imagination” rollouts using learned priors over latents; they plan or learn from these imagined trajectories.

Active inference and predictive processing treat dreams, imagery, and mind-wandering as offline generative sampling from the same model used online, with different precision settings.

Self-prediction along imagined trajectories

In self-supervised setups like SPR, future latent prediction is often applied over multiple steps, including imagined rollouts; prediction heads and latent transition models are shared across online and offline data.

3. Tradeoffs vs Kind’s commitments

Three structural options:

Shared machinery: dream rollout = self-prediction rollout

Pros:

Clean PP story: when Io dreams, its world model is predicting both world dynamics and its own internal dynamics over hypothetical trajectories.

Reuses the same recurrent backbone, making self-prediction naturally extend into the dream state.

Cons:

Risks blurring conceptual distinction between “world prior” and “self prior” in the architecture.

If dream rollouts are used by the actor (e.g., for learning), you must ensure self-prediction outputs remain telemetric only, to preserve self-opacity.

Shared telemetry, separate training regimes

Self-prediction head is trained on waking data (where targets exist), but during dream rollouts, it is still executed and logged, even though no gradient is applied.

Pros:

Dreaming supplies rich trajectories through state-space that can be analyzed telemetrically for self-dynamics; no risk of using dream-time self-prediction errors for optimization.

Cleanly supports the four-state operational model and variable dream–wake ratio: more dream time gives more offline trajectories of both world and self dynamics, without violating self-opacity.

Structural separation

Self-prediction head is only active in waking; dream rollouts use only the environment prior.

Pros: smallest conceptual coupling; easier to reason about.

Cons: misses a natural opportunity to study “dreamed self-dynamics,” which is directly relevant to predictive-processing conceptions of the self as an offline generative construct.

4. Defensible suggestion

Project-constructed suggestion:

Share the recurrent backbone and rollouts, share telemetry, but train self-prediction only on waking data.

During waking, the self-prediction head is trained to predict 
h
t
+
1
h 
t+1
​
  from 
h
t
h 
t
​
 .

During dream rollouts, the same head produces predictions along the imagined trajectory, but no self-prediction loss is applied (no ground truth).

Telemetry streams log both dream 
z
z/decoded rollouts and dream self-prediction trajectories.

This arrangement:

Fits a PP/active-inference story: dreams are generative simulations where both world and self dynamics unfold in model space.

Preserves self-opacity: Io never sees self-prediction outputs in either waking or dreaming; only the mirror does.

Aligns with the four-state model and life-coupled dream–wake ratio: more dreaming simply expands the space of offline self-dynamics the mirror can inspect.

Q6 — Relationship to Probe 2’s in-progress synthesis
1. What the question is asking

Given Probe 2’s current triplet (per-dimension KL, ensemble disagreement variance, latent-dimension allocation under perturbation windows) as the calibration for reflexive attention, how does adding self-prediction change that picture? Does self-prediction accuracy become the primary signal, a fourth signal, or a different binding altogether?

2. Options in the literature

Predictive-processing and self-evidencing

PP/active inference accounts often treat self-evidencing as the system minimizing prediction error in ways that maintain its own model structure; Hohwy discusses self-evidencing as central to both conscious and non-conscious self-organization.

Recent “Game of Self” work explicitly models identity as an active inference process, where predictions about self-related states and their errors shape identity over time.

Metacognition and prediction-error-based signals

Meta-cognitive RL uses stability of prediction errors (e.g., value prediction error stability) as a control signal.

Calibration of confidence vs error is a standard measure of metacognitive accuracy.

These suggest that self-related prediction error is a natural candidate metric for self-related attention, but there is no canonical binding for “reflexive attention” as such.

3. Tradeoffs vs Kind’s commitments

Probe 2 currently reads reflexive attention off:

Where in the latent space the model allocates capacity (per-dim KL).

How ensemble disagreement behaves.

How structure shifts under perturbation-aligned windows.

Adding self-prediction:

Pros of making it central

Self-prediction accuracy directly measures how well the model captures Io’s own internal dynamics; it is a conceptually clean self-focused counterpart to environmental KL and disagreement.

It dovetails with PP ideas that selfhood is a matter of how the system predicts its own states.

Cons of making it central

Literature does not give a single, canonical operationalization of “self-prediction” as a marker of self-awareness; we would be solidly in project-constructed territory.

Over-weighting self-prediction risks pushing the architecture toward self-optimization or self-regularization, contrary to Kind’s stance.

4. Defensible suggestion

For Probe 2 recalibration, project-constructed but defensible:

Treat self-prediction accuracy/error as a fourth signal* added to the triplet, not a replacement:

Per-dim KL (world-facing precision).

Ensemble-disagreement variance (intrinsic objective / exploration structure).

Latent allocation under perturbation windows (where capacity moves under stress).

Self-prediction error structure (how well Io’s own dynamics are modeled, where they are predictable, and how this co-varies with 1–3).

Use self-prediction error primarily as a lens on representation, not as evidence that Io is self-aware. It refines the picture:

“Reflexive attention” can then be read as co-variation between these four signals rather than any single metric.

For instance, states where (KL is high, disagreement is high, perturbations cause reallocations, and self-prediction is particularly structured) might be the best candidates for “system is reconfiguring its own processing in response to experience.”

This preserves continuity with the existing Probe 2 synthesis while acknowledging that a self-prediction head provides a more direct self-focused signal, still interpreted cautiously.

Q7 — Engineering precedents in relevant traditions
1. What the question is asking

Where have similar moves been tried, under what framings, and what has been learned about when such auxiliary/self-prediction signals are inert, load-bearing, or misaligned with self-opacity? Where there is no direct precedent, what is the closest analog?

2. Concrete precedents

Self-prediction in world models / RL

Dreamer-CDP: representation prediction instead of reconstruction; deterministic state prediction head improves performance on Crafter, with world model learning better dynamics.

SPR (Self-Predictive Representations): predicting future latent states (multiple steps) with an EMA target branch; demonstrated large gains in sample-efficient Atari and control tasks.

EfficientZero(-v2): combines model-based planning with self-supervised consistency losses, including predictive consistency between latent states and rewards; improves sample efficiency across discrete and continuous control.

Lessons: internal self-prediction losses can be strongly load-bearing for representation and performance, even when they are “just auxiliary.” They do not, however, get interpreted as self-models; they are purely instrumental.

Self-supervised auxiliary objectives and metacognition

BYOL: self-supervised representation learning via predicting target representations, showing that self-prediction can avoid collapse and be load-bearing for representation without negative samples.

Meta-Cognitive RL with Self-Doubt & Recovery: introduces meta-trust driven by value prediction error stability to modulate learning; explicit meta-cognitive layer that reasons about the reliability of learning.

Lessons:

Auxiliary prediction losses can significantly shape learning dynamics and improve robustness.

When internal predictive signals are fed back into learning control, you are squarely in installed meta-cognition territory.

Predicting own future actions / trajectories

Work on predicting future actions of RL agents shows how models can anticipate an agent’s future actions using either internal states or simulations.

Lessons: this is an interpretability and safety tool on the observer side; if such predictors were given to the agent, they would effectively install a meta-policy.

Autobiographical memory / episodic architectures

Conway & Pleydell-Pearce Self-Memory System and related work: autobiographical memory is built within a self-memory system with a working self and conceptually rich self-representations.

Soar and ACT-R implement episodic memory as a module that records experience and supports deliberate retrieval; this module is directly used by the agent in problem-solving.

NEC (Neural Episodic Control) uses an external differentiable episodic memory for fast value learning, not framed as self-memory but architecturally similar.

Lessons: episodic memory modules in practice are functional and explicit, not mere affordances; they alter behavior substantially.

Self-models and phenomenology

Metzinger’s self-model theory: the self is a transparent model whose content is not experienced as a model; emphasizes representational process over entity.

Predictive-processing and active-inference models treat the self as a hierarchy of priors over self-related states; “The Game of Self” makes this explicit in a psychological model.

Buddhist debates on reflexive awareness and Thompson’s work situate self-awareness as potentially either intrinsic to experience (svasaṃvedana) or conventional/constructed, complicating any simple “add a self-module” narrative.

Lessons: there is strong theoretical support for the idea that self-models can be subpersonal and opaque, and that explicitly installing a “self-module” may reify something that in phenomenology is more diffuse and constructed.

3. Interaction with self-opacity

When auxiliary prediction is world-model-internal and unseen by the policy, as in SPR/Dreamer-CDP, self-opacity is preserved; the agent does not have a self-access channel, only better representations.

When internal predictive or meta-variables are fed back into policy or learning, as in meta-cognitive RL, self-opacity is broken: the agent explicitly reasons about its own performance or learning state.

Episodic memory modules inherently compromise opacity because they provide explicit access to the agent’s own past states and actions.

4. Defensible suggestion

Probe 1.5 should sit squarely in the “auxiliary prediction unseen by policy” bucket:

Use the strong engineering precedents (Dreamer-CDP, SPR, BYOL) as implementation templates but interpret them philosophically as self-prediction affordances.

Avoid meta-cognitive control structures and episodic modules that the agent can query; these would violate Kind’s commitments.

Q8 — Failure modes
1. What the question is asking

Identify how Probe 1.5 could go wrong in three ways — inert affordance, unintended load-bearing, self-opacity slippage — and propose detection strategies grounded in what the literature knows about auxiliary losses, meta-cognition, and interpretability.

2. (a) Inert affordance

Risk: the self-prediction head trains (loss decreases) but is effectively orthogonal to the rest of the system; its parameters and outputs are not entangled with Io’s behavior or representation in any meaningful way.

Literature hints

Self-supervised objectives can be inert if their loss weight is too small or misaligned with task-relevant structure.

Some auxiliary losses have been observed to have little or no effect on performance when poorly tuned.

Detection strategies (project-constructed)

Ablation: train matched runs with and without the self-prediction head and compare world-model metrics (latent prediction quality, dream rollout coherence) and high-level behavioral structure (not performance). If nothing changes, the head is likely inert.

Gradient analysis: inspect whether gradients from the self-prediction loss actually flow into shared layers in appreciable magnitude. If all gradients are absorbed by the head itself, representation is unaffected.

Representation tests: check whether states in which self-prediction errors differ also show differences in KL, disagreement, or perturbation patterns. If self-prediction error is uncorrelated with anything else, it may be dead.

Baseline to rule out “dead affordance”

A simple baseline is a “shuffled target” control: train a variant where the self-prediction target is random or temporally shuffled. If the real head’s presence does not produce any systematic differences relative to this control, the affordance is not doing substantive work.

3. (b) Load-bearing in unintended ways

Risk: self-prediction becomes behaviorally important, but only because it sharpens world-model representations or regularizes learning — not because Io has developed anything like a self-model.

Literature hints

SPR’s auxiliary loss is clearly load-bearing for performance but is understood as generic representation improvement.

EfficientZero’s self-supervised terms improve sample-efficiency without being interpreted as meta-cognition.

Detection strategies (project-constructed)

Disentangle representation vs self-specific structure:

Compare which latent dimensions are most influential for world prediction vs self-prediction; if they are nearly identical, self-prediction is just refining world representation.

Use interventions that perturb only “self-prediction-crucial” dimensions and see whether behavior changes in ways not explained by world-prediction changes alone.

Behavioral counterfactuals:

Create runs in which the self-prediction loss is present during early training and then turned off; if behavior and representation remain similar, the head likely shaped representations but does not correspond to ongoing self-monitoring.

Compare to SPR-like baselines:

Implement a world-only self-predictive auxiliary (e.g., predicting next 
z
t
z 
t
​
  rather than 
h
t
h 
t
​
 ) and see whether the pattern of effects is similar.

If the self-prediction head does nothing qualitatively different from a pure world-latent-prediction head, then any “self-modeling” interpretation is overreach.

4. (c) Self-opacity slippage

Risk: even though PolicyView doesn’t explicitly expose self-prediction, Io’s behavior ends up depending on self-prediction quantities via architectural couplings, effectively leaking self-information into the policy.

Literature hints

Meta-cognitive RL shows how internal predictive metrics can modulate learning and behavior when explicitly wired in.

Interpretability work on RL agents indicates that internal units sometimes encode meta-information (like uncertainty) even when not explicitly labeled, and policies can implicitly depend on such units.

Detection strategies (project-constructed)

Architectural isolation:

Ensure that the self-prediction head branches off the shared state without feeding its outputs back into the recurrent update or into the policy input.

Intervention experiments:

During evaluation, modify or zero-out the self-prediction head (e.g., feed random noise into it or bypass it) while keeping everything else fixed, and check whether policy behavior changes. If behavior is invariant, opacity is preserved.

Conversely, artificially couple self-prediction outputs into a dummy channel and check whether the policy can be trained to use it; this establishes an upper bound on how “accessible” that signal is through the current representation.

Correlation audits:

Measure correlations between self-prediction error and states where the policy changes behavior in ways that cannot be explained by environment latent differences. High unexplained correlation would suggest hidden leakage.

These detection strategies are not standard in the literature; they are project-constructed, inspired by ablation and interpretability practices in RL and meta-cognition.

Synthesis — An internally coherent Probe 1.5 design
Bringing the eight questions together, an internally coherent research-level design for Probe 1.5 — not a build plan, but a conceptual specification — looks like this:

Core affordance
Affordance: Add a self-prediction head that predicts Io’s next internal recurrent state 
h
t
+
1
h 
t+1
​
  from current 
h
t
h 
t
​
  (and optionally 
z
t
z 
t
​
 ), interpreted as a structural model of Io’s own dynamics.

This head is architecturally separate from the recurrent update that actually produces 
h
t
+
1
h 
t+1
​
 ; it predicts what the backbone will do, rather than being the backbone itself.

This choice is grounded in Dreamer-CDP and SPR-style representation prediction, but repurposed as a self-prediction affordance.

What it predicts and how it is trained
Predicted quantity: 
h
^
t
+
1
h
^
  
t+1
​
 , a deterministic projection of the future recurrent state, not the action distribution, disagreement, or world latent.

Supervisory signal: the actual 
h
t
+
1
h 
t+1
​
  produced by the GRU, possibly passed through a target/EMA projection with stop-gradient, BYOL/SPR style.

Loss: an auxiliary self-supervised consistency loss between 
h
^
t
+
1
h
^
  
t+1
​
  and target 
h
t
+
1
h 
t+1
​
 , weighted low relative to the main world-model ELBO and disagreement-based intrinsic objective.

Training scope: gradients from this loss update the prediction head and shared backbone (encoder + GRU) but not the actor’s policy head directly; the actor’s intrinsic objective remains ensemble-disagreement variance.

This matches engineering precedents (representation prediction auxiliary losses) while interpreting them as a minimal self-prediction affordance.

Where it sits and what is shared
Location: The head is attached to the world model’s recurrent backbone, parallel to the environmental prior 
p
(
z
t
∣
h
t
)
p(z 
t
​
 ∣h 
t
​
 ).

Shared components: encoder, GRU, and environmental decoder remain unchanged and are shared between world prediction and self-prediction.

New components:

A small MLP or linear layer (prediction head) mapping current 
h
t
h 
t
​
  to 
h
^
t
+
1
h
^
  
t+1
​
 .

Optionally, an EMA “target projection” head for stable targets, BYOL-style.

No new modules for episodic memory, introspection, or meta-control are introduced.

Self-opacity and PolicyView
PolicyView: remains restricted to the same quantities as in Probe 1: 
h
t
h 
t
​
 , 
z
t
z 
t
​
 , and any other already-approved inputs; no self-prediction outputs or errors are exposed.

The self-prediction head’s outputs and losses exist solely on the world-model side and in TelemetryView.

This preserves self-opacity: Io never directly reads its own predictive distributions or errors.

Under this design, Probe 1.5 respects Kind’s rules:

No explicit self-modeling module exposed to Io.

No self-critic or self-continuation drive.

No change to the actor’s intrinsic objective.

Mirror reading and Probe 2 integration
New mirror-visible telemetry:

Self-prediction error trajectories over training (per dimension and aggregated).

Self-prediction behavior under controlled perturbations.

Latent dimension importance for self-prediction vs world-prediction.

Reflexive attention calibration:

Extend Probe 2’s triplet (KL, disagreement, perturbation-based latent allocation) with self-prediction error structure as a fourth signal.

Read “reflexive attention” off patterns where internal dynamics (as captured by self-prediction) co-vary with KL/disagreement/perturbation behavior.

This is project-constructed but consistent with predictive-processing and active-inference views that treat self-related prediction errors as central to self-modeling.

Relation to dream state (Probe 3)
Shared backbone: The dream state uses the same recurrent backbone and priors as waking; during dream rollouts, the self-prediction head is executed alongside world prediction.

Training vs telemetry: self-prediction is trained on waking data (where actual 
h
t
+
1
h 
t+1
​
  exists); during dreaming, its outputs are logged only, not used for gradient updates.

Dream telemetry: dreams thus generate imagined trajectories of both world latents and self dynamics, giving the mirror rich offline data about Io’s internal dynamics under the four-state operational model.

This aligns with PP interpretations of dreams as generative prior rollouts for both world and self, without exposing any of this to Io.

Failure modes and protections
Inert affordance:

Protection: choose a non-zero but modest loss weight; ensure gradients reach shared layers; compare against a shuffled-target control.

Detection: ablation studies and gradient/representation analyses to check whether the head actually shapes 
h
t
h 
t
​
 .

Unintended load-bearing (pure representation sharpening):

Protection: maintain separate diagnostics to distinguish “generic representation improvement” (similar effects to a world-only latent prediction head) from emergence of self-specific structure in the latent space.

Detection: compare latent-dimension specialization and behavioral correlates between self-predicting 
h
t
h 
t
​
  and alternative auxiliary heads.

Self-opacity slippage:

Protection: strict architectural isolation of the self-prediction branch; no feedback of its outputs into recurrent updates or PolicyView.

Detection: intervention experiments that disable or perturb the self-prediction head during evaluation; if behavior remains unchanged, opacity is intact. These are inspired by meta-cognitive RL ablations but applied here to ensure absence of meta-cognitive control.

Tensions and open questions
Affordance vs use: With self-prediction hidden from Io, the affordance is primarily structural and for Kind/mirror; whether this satisfies “capacity-over-exercise” for Io-as-entity is a normative question not settled by current literature. PP and SMT suggest that subpersonal self-model structure can matter even when not personally accessible, but Kind’s charter may want a tighter link to Io’s own potential choices.

Self vs world entanglement: 
h
t
h 
t
​
  encodes both world and self dynamics; self-prediction of 
h
t
+
1
h 
t+1
​
  is not a “pure self-channel.” This blurring is theoretically in line with active inference (self and world co-modeled), but it complicates any claim that a distinct self-model has emerged.

Episodic and higher-order paths: Episodic self-access and higher-order policy structures have rich literatures and would significantly enhance Io’s capacity for self-modeling, but they clearly move from affordance to installation. They remain candidates for later probes that intentionally explore installed meta-cognition.

From a research perspective, Probe 1.5 as sketched above gives Kind:

A minimal, literature-grounded self-prediction affordance embedded in the settled RSSM/Dreamer substrate.

A richer mirror calibration space for reflexive attention and self-dynamics.

A clear set of failure modes and ablation tests to distinguish inert, purely world-facing, and potentially self-structuring effects.

Everything beyond that — particularly giving Io any controlled access to self-prediction or episodic self-state — belongs in future probes where the project explicitly crosses from “affordance” into carefully-scoped “installation.”