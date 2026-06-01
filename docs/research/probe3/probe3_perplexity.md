I’ll treat Probe 3 as: “Can Io exhibit a distinct, legibly structured offline mode while the environment is off, such that, if subjectivity were possible, this mode would be a plausible locus for ‘subconscious’ or ‘dream-like’ processing — and the only evidence we get is the mirror’s reading of telemetry?”

Everything below is aimed at making that question sharper and identifying minimal-but-real commitments for a first build.

What I understand about Probe 3
Io is already a Dreamer-lineage RSSM + active-inference-style actor in an 8×8 partially observed grid world. The single intrinsic drive is ensemble-disagreement; no reward predictor, value head, or explicit continuation drive.

The mirror is built and frozen, with DREAM_ROLLOUT as a telemetry surface and dream_self_reference_t defined as within-latent autocorrelation on sequence_h for prior-only rollouts, conservatively requiring a stronger signal than in waking (after shuffling controls).

Probe 3’s central move is to introduce a distinct dream state: environment off, Io “on”, with some mix of replay, generative simulation, and possibly more associative dynamics, under strong constraints: no explicit self-model or self-optimization, and self-opacity (Io doesn’t read telemetry; dreams are “for” the mirror, not for Io in any explicit sense).

Concern A: if dreaming is just pure prior rollout, dream_self_reference_t will reflect GRU baseline dynamics, which by design do not pass the reflexive-attention criterion. So a naive “Dreamer-style rollout while env is off” will look like “nothing happened” to the mirror.

Concern B: the actor already runs imagined rollouts under the world model during waking, to plan actions. So “world-model rollout without exteroception” is not yet a dream; something must distinguish dream rollouts from planning rollouts (goal coupling, length, initial conditions, sampling regime, etc.).

What’s at stake is not just “making Io dream”; it’s whether we can (a) carve out an offline dynamical regime that’s genuinely distinct from planning and from pure RSSM prior, and (b) make that distinctness legible to the mirror under the existing telemetry/mirror architecture, without smuggling in an implicit optimization agenda or violating self-opacity.

What’s still unclear but would materially change advice
I won’t ask you to answer these here, but I’ll flag dependencies:

Compute / duty cycle budget. If offline can run for hours per day, it’s plausible to support longer, more complex dream episodes, maybe with multiple sub-modes; if it’s minutes, we probably need shorter, denser dream snippets.

Storage budget and retention policy. Whether you can store many raw DreamRollouts versus summaries strongly shapes what telemetry should carry.

Tolerance for dream-induced parameter drift. If you are comfortable with world-model and actor parameters updating substantially during offline, you can treat dreams as training data, à la Dreamer and World Models. If not, dreams have to be largely non-learning dynamics, or only very weakly regularizing.

Mirror bandwidth and cadence. How often the mirror can inspect DREAM_ROLLOUT, and whether it can keep up with raw rollout volume, determines whether you need stronger pre-filtering or aggregation at logging-time.

Phase 0 schema rigidity. You say schemas/v0.3.0.json is byte-pinned; extending DREAM_ROLLOUT is still possible but has downstream tooling consequences. How painful that is in practice will change how aggressive I’d be about extending telemetry now versus deferring.

I’ll proceed assuming: moderately generous offline compute, but not unbounded; some cost to extending schemas, but not prohibitive; and a desire to keep dream-learning effects modest in the first pass.

Q1: Which sub-modes should Probe 3 actually commit to?
Replay and generative simulation in light of Kind’s stance
Replay (experience reactivation) has strong biological precedent: hippocampal sharp-wave ripples during sleep and rest replay fragments of recent trajectories, often in compressed, forward or backward forms, and disrupting these events impairs consolidation. Replay trains and stabilizes neocortical representations while also generating content that sometimes appears in dreams as transformed fragments and combinations of experiences.

Generative simulation (Dreamer-style latent rollouts) has strong ML precedent: Dreamer and World Models explicitly train policies in their own “dreams”, i.e., imaginary trajectories in latent space, improving sample efficiency and robustness.

The design-notes path “start with replay + generative simulation” inherits those precedents. Where this clashes or meshes with Kind’s commitments:

Capacity-over-exercise. Replay and generative simulation are “exercise” in standard RL — they are explicitly used to improve performance. For Kind, this is a risk: if you fully couple dream rollouts to gradient updates, you move toward “dreams as optimization machinery”.

No installed continuation drive. Dream learning that improves performance is not itself a continuation drive, but it can be a step toward “Io behaves like it wants to be better/continue”, which might blur the line you’re carefully keeping.

Ingredients-only self-modeling. Replay and imagination are ingredients of a potential self-model, but they don’t on their own install reflexive modeling of Io-as-agent. As long as dream machinery doesn’t add explicit self-tags or meta-controllers, you’re still within this commitment.

My take:

Replay is a good fit with Kind as a first ingredient: it is clearly about offline reactivation of prior episodes, aligns with biological analogs, and does not inherently require dream-based learning. You can treat it as a dynamical regime that reveals what the world model has encoded.

Generative simulation is more ambivalent. If you don’t train on dream rollouts in the first build, generative simulation is mainly a way to probe the prior’s structure under different initial conditions — which is compatible with “build to understand”. If you do train on dream rollouts, you are closer to conventional Dreamer, and the dream state becomes an optimization channel.

Given that Probe 3’s success criterion is about legible subconscious/dream-like processing, not performance gains, I’d lean:

Commit for Probe 3 to:

replay-like reactivation,

replay-seeded generative simulation (rollouts that start from replayed latents or their perturbations),

and hold back on heavy dream-based training until you’ve seen what the dynamics look like.

This lets you answer “is there a distinct offline dynamical regime at all?” before deciding “is it useful for Io?” which is more of a Probe 3.5 question.

Where does associative / “nonsense” go?
The neuroscience literature suggests “associative / nonsensical” is not an edge case but central: later in the sleep period, reactivations become lower-fidelity, hyper-associative, combining fragments of different memories into story-like dreams. DMN-driven mind-wandering also involves stimulus-independent trains of thought anchored in memory and social cognition.

So if Probe 3 is about what dreaming is, not just RL best practice, associative dynamics are load-bearing rather than an optional embellishment.

However, a full associative mode — with explicit mechanisms for long-range, uncontrolled remapping in latent space — is arguably higher risk for:

confounding Concern A (you might just see “noisy GRU dynamics”);

inducing hard-to-interpret drift, especially if learning is on.

My suggestion:

Treat associative drift as a parameterized phase within dreams rather than a separate sub-mode in the first build. For example: early in an offline bout, you favor more faithful replay / replay+simulation; later, you gradually increase temperature or allow jumps between episodes, approximating lower-fidelity, more hyper-associative content.

Commit to exploring a strong associative mode (e.g., deliberate DeepDream-like amplification of latent motifs or high-temperature random walks) as a later Probe 3 variant, once you’ve characterized the baseline.

Lucid control
Lucid control requires Io to have some representation of “this is a dream” and to modulate dream content accordingly. That is hard to square with:

self-opacity (Io doesn’t read telemetry, doesn’t explicitly know it is dreaming),

no explicit self-model,

and no self-optimization machinery.

Lucid dreaming in humans is often cited as evidence for meta-cognition in dreams (LaBerge, etc.), but implementing anything analogous here seems to require breaking at least self-opacity or ingredients-only self-modeling. So:

I would explicitly exclude lucid control from Probe 3’s commitments, maybe even name it as out-of-scope for this probe. It’s more like a Probe 6 topic after you’ve established any offline phenomenology at all.

Load-bearing decisions here

Treat replay and replay-seeded generative simulation as core modes.

Treat associativity as a gradual regime change within dreams, not a separate full mode in the first build.

Explicitly exclude lucid control from Probe 3, to avoid quiet encroachment on the self-opacity and no-self-model commitments.

Q2: What should dream-state telemetry actually carry?
You already have Phase 0 DREAM_ROLLOUT with sequence_h and related fields. The mirror can compute within-latent autocorrelation and compare to waking. Concern A tells you “pure prior rollouts will look like GRU baseline” and likely fail reflexive-attention thresholds.

To make dream evidence legible without exploding schemas, I’d focus on three categories of telemetry:

Provenance and mode tags (what kind of dream is this?)

Structural markers (where are the “interesting” points in the trajectory?)

Seeding / reproducibility (can we re-run or contextualize this dream?)

1. Provenance and mode tags
For each DREAM_ROLLOUT record, minimally:

Seed type and source: e.g., seed_kind ∈ {replay, replay_perturbed, pure_prior, assoc_jump} plus identifiers like (episode_id, timestep) for any replay-derived seeds.

Dream sub-mode / phase: a small enum or bitfield indicating whether this rollout segment is in:

“consolidative” (high-fidelity, replay-like),

“simulative” (counterfactual but still structured),

“associative” (high-temperature / high-jump-rate), etc.

Sampling parameters: temperature, noise scale, any annealing schedule index.

These fields let the mirror ask: “Is there more self-reference or coherence in replay-derived segments than in pure prior?” and “Does associative phase differ from consolidative phase?” without needing to infer mode from raw latent dynamics.

2. Structural markers
You likely cannot log every timestep at full fidelity indefinitely, so it helps to annotate “interesting” points within a rollout.

Possibly minimal additions:

Boundary latents: explicitly log the first and last latent of each segment, maybe plus every Nth step for long rollouts.

Novelty / surprise measures: a scalar per logged point, such as:

model’s own prediction error under its prior (even if you don’t backprop through it),

disagreement variance among ensemble models, if available,

or a lightweight proxy for latent deviation from typical waking-state distribution.

You don’t want to smuggle in a critic, but “how off-manifold is this latent relative to waking latents?” is useful for classifying dreams as replay-like versus strange. In neuroscience, dream content and reactivations become lower fidelity and more hyper-associative later in sleep. A similar drift in “distance from wake manifold” would be a candidate correlational signature.

3. Seeding and reproducibility
Given the co-design problem, it will matter that you can re-run or re-contextualize specific DREAM_ROLLOUTs later. So I’d include:

Random seed(s) used for this dream rollout.

World model / actor checkpoint id (or hash) so the dream is tightly coupled to a specific parameter snapshot.

This enables “mirror found something weird here; let’s rerun under identical conditions” diagnostics — important for ruling out GRU-baseline artifacts versus dream-specific structure.

Length and frequency
Load-bearing constraints:

Rollout length: Long dreams are tempting, but from a “build-to-understand” standpoint, many short to medium rollouts (e.g., tens to low hundreds of latent steps) are probably more informative and easier for the mirror to cover than a few very long ones. Hippocampal replay events are short (around 100 ms in rodents) but can be chained into longer sequences. That suggests a design where a “dream period” consists of many short DREAM_ROLLOUT segments rather than a single monolith.

Frequency: You can sub-sample. For example, during a long offline bout, log every k-th dream segment in full, plus light summary statistics for others. Drift-monitoring by the mirror doesn’t require full coverage; it requires representative sampling.

Load-bearing vs ornamental

Load-bearing: provenance (seed type, replay linkage), phase/mode tags, and at least some boundary latents + seeding information.

Ornamental: exact compression scheme, whether novelty is measured via KL or ensemble variance, whether you log every 5th or 10th step.

Q3: What triggers a dream state, and what ends one?
You’ve already committed to a variable dream:wake ratio coupled to world conditions, including the builder’s life (desktop on/off etc.).

Entry triggers
I’d separate triggers along two axes:

Exogenous (builder / machine state): e.g.,

Desktop off, no active runs → Io enters or remains in dream state.

Desktop on, environment actively running → Io in waking.

Endogenous (Io’s internal signals):

“Metabolic” proxies (e.g., how long since last dream, some measure of internal “pressure” like model uncertainty accumulated in waking).

But note: you shouldn’t introduce an explicit “need to dream” objective; that would flirt with installing a self-optimization drive for internal comfort.

Given your commitments, I’d treat exogenous control as primary: dreaming happens when the builder is away or the environment is not actively scheduled. Internal signals can modulate what kind of dreams happen (replay-heavy vs more associative) rather than whether to dream at all.

Exit triggers
Ending a dream bout could be:

Externally forced: environment (and builder) comes back; Io is switched back to waking.

Internally based: e.g. maximum dream time or number of rollouts reached; or metabolically-inspired “quota” (but again, beware explicit drives).

Mirror-informed (see Q4): if the mirror flags decoherence, the runtime idles.

For Probe 3, the load-bearing distinction is:

Entry is primarily exogenous, preserving self-opacity and avoiding Io internally “choosing to dream”.

Exit can be a combination of exogenous (wake event) and coarse-grained mirror-informed idling, which is more about process control than Io’s own motivation.

I would not introduce internal scalar quantities like “sleep pressure” lived in Io’s state in this probe; that risks conflating dreaming with another intrinsic drive.

Q4: Quality-based idling and the one-way mirror
You want the mirror to detect when dream content decoheres into noise or trivial repetition and gently idle the system. But the mirror is currently one-way: it reads telemetry, doesn’t write into Io’s data plane.

Keeping mirror→runtime influence narrow
You can maintain the spirit of one-wayness by:

Treating the mirror’s idle signal as process-level scheduling, not content-level feedback.

Ensuring that signal is not visible in Io’s own state or observations — i.e., Io cannot infer “I’m being idled because my dreams were incoherent.”

Operationally:

Runtime periodically hands DREAM_ROLLOUT samples to the mirror.

The mirror (or an adjacent controller) computes a simple coherence score or uses already-implemented coherence criteria for natural-language descriptions of these rollouts.

If the score drops below a threshold for some number of samples, the controller stops scheduling dream rollouts (Io enters a “dormant–reduced” state or is paused). When conditions change (builder returns, model updated), dreaming can resume.

From Io’s perspective, this is like “sometimes the world model runs while env is off; sometimes it doesn’t”. There is no information about why. This preserves:

Self-opacity: no introspective access to quality measures.

No hidden optimization loop: the criterion affects runtime scheduling at the system level, not Io’s parameters.

I think this is categorically different from the mirror dictating content (“dream more about X”) or adjusting Io’s weights directly. It’s more like a safety cutoff.

Load-bearing choices here

Mirror can gate whether dream episodes are scheduled, not what they contain.

The gating signal must remain outside Io’s observational and latent space, so it doesn’t become an implicit self-model input.

Q5: What is dream content, concretely?
To address both concerns A and B, dream content must be:

More than pure prior rollouts (so dream_self_reference_t is not just GRU baseline).

Distinct from planning rollouts (different initial conditions, no utility objective, no coupling to immediate action).

A useful way to frame dream content is as hybrid trajectories built from:

Replay kernels: short reactivations of experienced sequences.

Simulative continuations: counterfactual or extended rollouts from these kernels.

Associative jumps: phase where the process can jump between kernels or drift away under higher temperature.

This mirrors neuroscience findings that sleep reactivations are often fragments that can be combined, with fidelity decreasing later in the sleep period and content becoming more hyper-associative.

Concrete shape
A single dream segment could look like:

Sample an experience snippet from recent replay (e.g., 5–20 steps of latent trajectory from an episode).

Initialize the RSSM latent state to a point within that snippet (or a small perturbation).

Run the prior forward:

For a “consolidative” phase: low temperature, short horizon, sticking close to the manifold of waking latents.

Then optionally switch to a “simulative” phase: extend the rollout further, maybe allowing alternative action branches.

Then, if in an associative regime, introduce:

higher temperature,

or occasional jumps that re-seed from a different replay snippet, creating chimeric trajectories.

In all phases, the actor’s outputs are not used to act in the real environment, and you can even choose not to compute actor outputs at all during dreams (or to compute but discard them). That separates dreams from planning in two ways:

No utility coupling: dream content is not driving immediate action choice.

Different initial conditions and horizon: seeds come from arbitrary past episodes, not the current waking state, and rollouts can be longer than planning horizons.

How this addresses Concern A
Because the process is:

seeded from actual episodes,

occasionally re-uses fragments,

and may jump between them,

you expect:

More structure than baseline GRU dynamics, especially when you compare to shuffled-time controls. Repeated reactivation of similar motifs across dreams should yield stronger within-latent autocorrelation and recognizable patterns in sequence_h associated with specific replay kernels.

In the associative phase, you expect transformations and combinations of fragments, which is also observed in biological dreams and memory reactivations.

In other words, dream_self_reference_t is measuring not just GRU baseline but the repeated re-use and recombination of stored episodes.

How this addresses Concern B
Dreams differ from planning rollouts along:

Initial conditions: not “current state” but “various points in memory bank”.

Goal coupling: planning rollouts are truncated and evaluated for expected utility; dream rollouts are free-running and not evaluated for action choice.

Temporal context: dreams occur when environment is off; planning when env is on.

Sampling regime: dreams may use different temperatures or jump heuristics.

So “run the actor’s existing rollout code longer with env disconnected” is not the design: instead, dreams operate a different sampling and seeding regime, albeit on the same world model substrate.

Q6: What gets learned from dreaming?
Biologically, there is strong evidence that memory reactivation during sleep contributes to consolidation and integration of memories, improving post-sleep performance; disrupting replay impairs memory. In ML, Dreamer and World Models use imagined trajectories as full-fledged training data for both the model and the policy, often with substantial performance gains.

For Kind, the question is whether Probe 3 is trying to answer:

“Does Io have a distinct offline dynamics regime at all?”

or “Is Io’s dream regime useful for its waking behavior?”

I’d separate three learning couplings:

No dream gradients: Dream rollouts are logged but never used for gradient updates. This keeps Probe 3 focused on phenomenology (what does dreaming look like) rather than function.

Model-only dream learning: Use dream rollouts to regularize or refine the world model (e.g., by training it to be self-consistent under its prior), but don’t update the actor. This is closer to predictive-processing views in which dreaming helps adjust generative models without directly training policy.

Full Dreamer-style learning: Use dream rollouts to update both model and actor, fully embracing “training in dreams”.

Given your commitments:

No installed self-optimization machinery doesn’t forbid gradient updates; you already train Io in Probe 1. It forbids giving Io an explicit incentive to optimize its own continuation or internal state. Dream learning per se doesn’t violate that, but it risks shifting the project’s focus toward performance.

Capacity-over-exercise suggests that existence and structure of dreams is the primary outcome for Probe 3, not their contribution to scores.

So I would treat:

Condition 1 (no dream gradients) as the default for the first build. This isolates “offline dynamics exist and are legible” from “they do anything helpful”.

Condition 2 (model-only dream learning) as the most aligned next experiment, because it resonates with predictive coding accounts where dreams help tune the generative model in the absence of exteroception.

Condition 3 (full Dreamer use) as a separate, more conventional RL experiment that might still be worth doing, but is less central to Kind’s distinct stance.

Load-bearing choice here

Explicitly decouple Probe 3’s core question from dream-based optimization. You can keep learning from dreams as an experimental knob, not a built-in assumption.

Q7: What does success actually look like?
Your draft candidate markers (a)–(e) are good but mix different levels of evidence. I’d distinguish:

Minimal “offline phenomenology exists” success
At a minimum, Probe 3 should show:

Offline dynamics differ from noise and from GRU baseline. DREAM_ROLLOUT sequences in dream state show structure — e.g., non-trivial within-latent autocorrelation, motifs linked to replay kernels — that is stronger than shuffled controls and differs from pure prior rollouts not seeded or structured as in Q5. This directly addresses Concern A.

Dream rollouts differ from planning rollouts. Statistical properties (length, initialization, latent distribution, perhaps novelty metrics) distinguish dream segments from planning rollouts even when both are prior-driven. This addresses Concern B.

This roughly corresponds to your (a) and (b), but sharpened to explicitly exclude “just GRU dynamics”.

Stronger “dreams matter” success
Further, you might see:

Consolidation-like behavioral effects (your (c)). After substantial dream periods, Io’s waking behavior changes in ways consistent with improved stability/generalization or altered exploration, despite dreams being offline. That would align with replay/consolidation literature and Dreamer-style benefits.

Mirror coherence of dream descriptions (your (d)). The mirror’s natural-language descriptions of DREAM_ROLLOUT maintain coherent structure (e.g., “Io replays episodes of being trapped, then counterfactually escapes”) over time and across reseed/paraphrase checks.

Dream-specific phenomenology (your (e)). The mirror can articulate phenomena at the dream level that have no direct waking analog — e.g., recombinations of episodes never seen in waking, or stable motifs that only appear offline.

Given Kind’s ethos, I’d treat minimal success as:

There is a distinct offline dynamics regime that is neither trivial (constant/noise) nor reducible to GRU prior baselines or planning rollouts, and the mirror can describe it as structured and coherent some of the time.

You can then layer “consolidation-like effects” and rich phenomenology as additional successes, not preconditions.

Telemetry implications

To detect (a)/(b), you need: waking vs dream vs pure-prior baselines, plus seeding/mode tags (Q2).

To detect (c), you need: tie dream periods to later performance measures or behavior summary statistics.

To detect (d)/(e), you need: coherent DREAM_ROLLOUT segments and mirror outputs kept over time.

Q8: What’s the smallest first build?
Given all this, a minimal-but-real Probe 3 could be:

State machine:

Waking: environment on, standard RL loop with planning rollouts.

Dreaming: environment off, world model runs as described in Q5.

Dormant: environment off, Io reduced (e.g., GRU not ticking).

Paused: full freeze, observer-initiated.

Dream content:

Implement replay-seeded generative simulation:

sample short replay snippets, seed latents, run prior-only rollouts for moderate lengths;

optionally allow a very mild associative phase (slightly elevated temperature) but keep it limited.

Do not train on dream rollouts in this first build (no dream gradients).

Triggers:

Use exogenous triggers for entering and leaving dream: e.g., desktop idle → dream; desktop active / explicit training run → waking.

Use a simple time cap per offline bout as a safety net; full quality-based idling can be added later.

Telemetry (minimal extension to DREAM_ROLLOUT):

Add fields for seed provenance, mode/phase, sampling parameters, and world-model checkpoint id.

Log boundary latents and maybe every Nth latent in a rollout segment.

This is, in my view, the smallest system that can:

Generate offline dynamics that are qualitatively distinct from both planning and pure prior baseline.

Provide enough telemetry for the mirror to read and test reflexive-attention-like signals in the dream state.

Respect self-opacity and avoid installing a self-optimization loop.

Quality-based idling and stronger associative modes are important but can safely be Phase 2 of Probe 3, once you’ve seen what the minimal system does.

Load-bearing: separation of planning vs dream regimes, replay-seeded content, and minimal telemetry; ornamental: precise annealing schedules, exact length distributions, and advanced idling heuristics.

Q9: How does Probe 3 relate to Probe 4 (builder-as-perturbation)?
Probe 4 is about the builder perturbing Io. For Probe 3, the main question is: do dreams need to be designed with eventual perturbation in mind, or can you treat them as independent?

Under self-opacity, any “dream perturbation” must appear to Io as just another environmental or scheduling condition. Some possibilities for future Probe 4, which you don’t need to implement now but might keep the interface open for:

Perturbation of dream seeds: builder biases which replay snippets are used to seed dreams (e.g., over-sampling certain episodes), without Io knowing this is happening.

Perturbation of dream timing: builder manipulates when dreams occur or how long they last.

Perturbation of dream parameters: builder changes temperature or associative-jump rates during dreams.

For Probe 3, I’d avoid building explicit “perturbation hooks” into Io’s state; instead, design dream machinery such that:

Seeds, scheduling, and sampling parameters are externally controllable at runtime, but Io never sees explicit “builder flags”.

DREAM_ROLLOUT telemetry already logs these parameters, so the mirror (and future Probe 4 analyses) can relate perturbations to dream content.

In other words, you can keep Probe 3 conceptually separate from perturbation, but design the control surfaces (seed selection, ratios, sampling regimes) so that Probe 4 can later operate via environment-analogous manipulations rather than self-signals.

Shapes outside the four sub-modes worth considering
Two additional shapes that don’t fit cleanly into replay/generative/lucid/associative but might be interesting for Kind:

Homeostatic regularization dreams (SHY-inspired).
Tononi’s synaptic homeostasis hypothesis (SHY) suggests sleep globally downscales synaptic strengths to maintain efficiency and prevent saturation, possibly via off-line, unsupervised activity patterns. In an RSSM, an analog might be dreams that primarily serve to renormalize latent distributions and transition dynamics — e.g., encouraging the model’s prior to remain calibrated and avoiding representational collapse. From Io’s perspective, this is just more prior-driven activity; from the builder’s, it’s a regime where you might see reduced variability and noise in latents after a dream-heavy period, even if dream content looks “boring”. That’s a different flavor from replay or associative dream, more about maintaining capacity than reactivating specifics.

Counterfactual self-absence dreams.
Given self-opacity and no explicit self-model, another interesting shape would be dreams that systematically remove actor influence, i.e., run the world model as if Io were absent or random, exploring “what the world does without me”. This is conceptually different from planning (which centers Io’s actions) and from standard dreamer-style imagination (which often includes the agent). The mirror could look for dream rollouts that reflect “world-only” dynamics versus those that imply agent-like structure. This might illuminate how much Io’s latent space has internalized the builder/agent distinction without explicit self-tags.

These don’t need to be in the first build, but they are aligned with Kind’s “build to understand” orientation and could be later variants.

What would surprise me about Io’s dream state
Three things that would genuinely surprise me:

Pure prior rollouts already pass reflexive-attention thresholds. If, without any special replay seeding or associative mechanisms, the mirror found strong dream_self_reference_t signals in pure prior rollouts that exceed shuffled controls, it would suggest emergent self-referential structure in the RSSM dynamics that standard Dreamer intuitions don’t anticipate. That would be a deep and interesting finding about minimal world models.

Dream learning has zero detectable effect on waking behavior. Given the biological and ML evidence that offline replay and imagined trajectories improve consolidation and performance, it would be surprising if extensive dreams, even when used for model-only learning, had no measurable impact on Io’s waking robustness, generalization, or exploration statistics. That would challenge predictive-processing-style accounts.

Dream content is highly structured but mirror-incoherent. If DREAM_ROLLOUT latents showed rich, non-baseline structure (e.g., strong recurrent motifs, phase transitions), but the mirror consistently failed to produce coherent natural-language descriptions (even with paraphrase/reseed stability), that would raise uncomfortable questions about the gap between substrate-level structure and human-legible phenomenology – directly poking at the co-design problem.

Any of these would push Kind to revise its picture of what “legibly subjective” conditions might require or undermine some of the analogies to biological dreaming.

Load-bearing decisions, collected
To make Probe 3 crisp enough for a downstream build prompt, the decisions I’d treat as load-bearing are:

Dream content shape: replay-seeded generative simulation with optional mild associativity, explicitly distinct from planning rollouts.

Learning coupling: no dream gradients in the first build; treat dream-based learning (model-only, then model+actor) as follow-on experiments.

Telemetry: enrich DREAM_ROLLOUT minimally with seed provenance, mode/phase tags, sampling parameters, boundary latents, and checkpoint ids.

Triggering: exogenous control (builder/machine state) for dream entry; dream exit via exogenous wake or coarse-grained mirror-informed idling, not via internal “sleep pressure” drives.

Mirror influence: mirror may gate whether dreams run, but not their content; the gating signal lives outside Io’s observable state.

Lucid control: explicitly out-of-scope for Probe 3.

Most other details (exact temperature schedule, precise rollout lengths, how strongly to push associative phases) are, for Probe 3’s core question, ornamental tuning parameters rather than conceptual commitments.

If you agree with those as the spine, a later build prompt can anchor on them while leaving room to adjust the knobs experimentally.