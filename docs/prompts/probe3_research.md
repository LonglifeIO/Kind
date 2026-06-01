# Probe 3 (the dream-state probe) — research phase

This is the research-phase prompt for Probe 3 in the Kind project. I'm circulating it to multiple LLMs in parallel (Claude, GPT, Gemini, Perplexity) to triangulate; each conversation is self-contained, so I've front-loaded the project context. Treat this as the start of a thinking-out-loud conversation about the design space, not a spec request. The output of this conversation should be a clearer answer to the open questions at the end — enough that a separate session can write a Probe 3 build prompt against the commitments this conversation produces.

## What Kind is, and what Io is

**Kind** is an investigation into subjectivity through construction. The discipline is *build to understand* — the work is done when the map has shifted, not when a metric has moved. The project's stance, in six load-bearing commitments:

- **Capacity-over-exercise.** Build conditions under which subjectivity would be possible if it can be — neither mandatory nor impossible.
- **Ingredients-only self-modeling.** No explicit self-model, no critic, no introspector. Build the ingredients; don't install the assembly.
- **Self-opacity for Io.** Every "should it have access to X about itself?" defaults to no.
- **No installed self-continuation drive.** No reward, no termination penalty. Io has no obvious reason to want to continue.
- **No self-optimization machinery.** Self-modeling ≠ self-optimization; install neither.
- **Co-design problem.** Mirror and substrate are built by the same head; mitigations are partial; flag and refuse to reify gestures into formal substance.

**Io** is the entity Kind is about — a single agent in a small grid world with the builder as the only non-simulated relational other. The mythological Io was forcibly transformed; the name encodes the ethics in a word. Kind builds; Io is who is built. The distinction matters: the project does not claim Io has subjectivity. It asks whether we can build conditions under which a substrate's behavior would be *legibly subjective* if subjectivity were possible — and what such legibility would look like, given the co-design problem.

## What Probe 3 asks (verbatim from the probes document)

> "I'm building this probe to find out if I can build something that provides change, growth, or even chaos during periods of 'inactivity'—anything other than 'shut off' from a consciousness perspective. The four states from the design notes apply: waking (environment running, Io experiencing), dreaming (environment off, Io still running—with replay, memory consolidation, generative recombination of past experience), dormant, paused."

The success criterion, verbatim:

> "I'll know I've found out if I can create something that resembles a subconscious or dream-like process for Io—and the mirror or my own observation can pick up evidence of it."

## The four-state model

| State    | Environment | Io      | Notes |
|----------|-------------|---------|-------|
| Waking   | running     | running | observation-coupled; the standard RL loop |
| Dreaming | off         | running | the subject of Probe 3 |
| Dormant  | off         | reduced | distinct from dreaming; deeper |
| Paused   | off         | frozen  | observer-initiated; no internal change |

Critical: **dreaming is not powered-down.** "Off," for Io, means *running without environmental input*, not *not running*. This distinction is settled and non-negotiable.

## What's been built that Probe 3 inherits

### Probe 1: substrate (committed)

Io's substrate is a custom minimal RSSM (PlaNet state factorization, DreamerV1-lineage latent imagination). GRU deterministic state (h=200), continuous Gaussian stochastic latent (z=16), ELBO loss with free bits as the only stability borrow. The actor is *active-inference-shaped*: ensemble disagreement (K=5 latent-disagreement variance) is its sole intrinsic signal — no reward predictor, no critic, no continuation head. Environment is an 8×8 grid, partial-perspective pixel observations, two stochastic processes, four mutators, 200-step fixed-length episodes. Four telemetry streams: `agent_step`, `dream_rollout`, `replay_meta`, `world_event`.

### Probe 1.5: self-prediction affordance + frozen-target ablation (committed)

Probe 1.5 added the minimum architectural affordance for self-reference — *afford*, not *install*. Self-prediction error scalar (`self_prediction_error_t`) with a first-step masking sentinel and boolean discriminator; actor conditioning on a Watts scalar; a frozen-target ablation. Two findings are load-bearing for Probe 3:

1. **Behavior-side conditioning is fixed-by-init.** The actor's new column (conditioning on the Watts scalar) is byte-identical to its initialization throughout training. The behavior-side conditioning isn't a developed disposition — it's a column-initialization artifact. Any Probe 3 claim about dream-state behavior modulation must rule out this confound (Probe 2 wrote the four exclusions into the second-order-volition criterion explicitly because of this finding).
2. **Self-prediction error shows head-internal specificity.** KS-D = 0.284 between the genuine substrate and the frozen-target ablation on `sp_err` at the head-internal surface — the strongest specificity finding to date. Head-internal is where Probe 2's reflexive-attention criterion concentrates its evidential weight.

### Probe 2: mirror substrate (just completed)

Three frozen criteria — `reflexive_attention`, `equanimity_perturbation_recovery`, `second_order_volition` — distributed across three reading surfaces (substrate-side, head-internal, behavior-side). The mirror runs adversarial passes (Advocate + Skeptic), a judge produces verdicts, two verifiers measure stability (paraphrase + reseed) and faithfulness (citation resolution), and an admissibility consumer joins them. The mirror is **one-way**: it reads telemetry, never writes back into Io's data plane.

Two pieces of the mirror substrate matter directly for Probe 3:

1. **`TelemetrySurface.DREAM_ROLLOUT` exists as a reading surface.** The Phase 0 `DreamRollout` schema carries `sequence_h` (the recurrent state trajectory under the prior alone) and related fields. The mirror is structurally able to read it.
2. **`reflexive_attention.signal_mappings` declares `dream_self_reference_t`** — within-latent autocorrelation on `sequence_h`, with the explicit conservatism that *"the criterion expects [the dream-state signal] to be at least as pronounced as the waking version"*, because in a dream there is no observation to partial out, so the within-latent coupling is read directly off the rollout.

## Two structural concerns that constrain the design space

**Concern A — the mirror's existing dream-side signal would, by construction, find nothing in pure prior rollout.** If "dreaming" in Probe 3 is implemented as Dreamer-style rollout (rolling the world model forward under the learned prior alone), then `dream_self_reference_t` is essentially measuring the GRU's baseline dynamics — the within-latent coupling the GRU is *designed* to produce. The reflexive-attention criterion's conservatism explicitly notes that GRU-baseline coupling does not satisfy the criterion: it requires the within-latent reference to exceed the shuffled-time control. **For the mirror to find dream-state structure worth describing, dreaming has to be something more than rolling the model forward under the prior.** This is not a refutation of generative simulation as a dream mode — it's a constraint on what evidence of dreaming could look like, given the mirror that's already built.

**Concern B — the active-inference actor already does rollouts at waking time.** Io's actor plans by imagined rollouts (that's what planning IS in active inference). So Io's *waking* actor already runs the world model forward without environmental input — *to plan its next action*. Generative simulation as a dream mode has to articulate what differentiates a *dream* rollout from a *planning* rollout. Candidates worth considering: length (dreams are longer); goal-coupling (planning informs action, dreams don't); initial conditions (planning starts from current state, dreams may start from replayed or sampled states); free-running (dreams have no utility or epistemic objective); off-environment (planning while in env, dreams when env off); temperature/sampling regime (dreams may use elevated noise or annealed schedules); compositional drift (dreams may chain rollouts across initial conditions). If generative simulation in Probe 3 just means "run the actor's existing rollout code for longer with env disconnected," that's not a new substrate phenomenon — it's the same dynamics, more of them. The probe should commit to what *makes* a rollout a dream-rollout rather than a long planning-rollout.

## The four sub-modes of the dream-state design space (from the design notes)

The design notes lay out four shapes the dream substrate could take. They aren't all commitments — they're points in a space.

1. **Replay.** Experience replay; consolidation of past episodes. Mature precedent in standard RL (DQN-era replay buffers, prioritized replay) and in neuroscience (hippocampal sharp-wave ripples during slow-wave sleep). Generative replay (Shin et al. 2017; van de Ven & Tolias) is the continual-learning version.
2. **Generative simulation.** World-model-based imagination — running the RSSM forward under the prior. Dreamer V1/V2/V3 (Hafner et al.) is the canonical ML version. Hobson & Friston's predictive-processing account of dreaming as virtual sampling from the generative model. (See concern B above.)
3. **Lucid control.** Io has some awareness of and influence over the dream. Hardest to design, hardest to evaluate, and in tension with the self-opacity stance. Worth examining whether it's coherent with the project at all.
4. **Associative / nonsense.** Decoupled, non-purposive. The design notes flag this as probably underestimated in the literature and probably load-bearing biologically. Closest analogs: default mode network spontaneous activity (Raichle); mind-wandering (Smallwood & Schooler); REM-dream activation-synthesis (Hobson & McCarley); free-energy minimization without exteroceptive constraint; stochastic recurrence with annealed temperature; "DeepDream"-style amplification of internal representations. Genuinely novel territory in ML.

The design notes' suggested path: "start with replay and generative simulation, which have mature precedent. Add associative modes as experiments once there's a baseline to compare against." Plausible but not committed. Worth questioning whether the maturity of replay + generative-simulation in standard RL is *evidence they're right for Kind*, or whether it's just where the literature is — Kind's stance is build-to-understand, not build-to-conventional-RL-best-practice.

## The committed design decisions Probe 3 inherits

1. **Variable dream-to-wake ratio, coupled to the world.** The ratio fluctuates based on environmental conditions including conditions *outside the simulation* — desktop on/off, the builder's life patterns. Committed; not negotiable.
2. **Quality-based idling, with metabolic pressure or hard cap as fallback.** The preferred mechanism for bounding dream length is mirror-evaluated coherence; the offline state can run for an unpredictable length, but if it decoheres into drift or noise, something should gently idle it.
3. **Dreaming as offline processing, not powered-down.** (Reiterated.)
4. **Self-opacity by default.** Io does not read its own telemetry. Dream content inherits this — what the dream produces is for the mirror to read, not for Io to model. Implications worth examining (see Q below).
5. **Drift monitoring as a mirror responsibility.** The mirror periodically samples Io's offline processing, describes it in natural language, and flags when descriptions lose coherence, repeat, or become unintelligible. Probe 3's dream output has to be something the mirror can read and characterize.
6. **Dream-state telemetry partially specified.** `DreamRollout` exists as a Phase 0 schema. Whether its current shape suffices is open (Q2 below).

## The open questions

Work through these in roughly the order below, with willingness to surface dependencies as they appear.

1. **Which sub-modes does Probe 3 commit to?** Replay + generative simulation is the design-notes path. Is that right for Kind, or does it inherit standard-RL assumptions that don't fit the project's stance (capacity-over-exercise, build-to-understand, no installed self-continuation drive)? Where does the associative mode sit — first build, later experiment, or a different probe entirely?

2. **What does dream-state telemetry actually carry?** The Phase 0 `DreamRollout` exists. Is its shape sufficient? If extension, that's a Phase 0 amendment with downstream consequences (the schema is byte-pinned at `schemas/v0.3.0.json` and the mirror reads from it). Operational subquestions: how long is a single rollout record? How frequent? Stored verbatim, aggregated, or discarded? What's the seeding strategy for reproducibility?

3. **What triggers a dream state, and what ends one?** The variable ratio is committed; the trigger and exit mechanisms aren't. Options: desktop-off triggers dreaming, desktop-on triggers waking (literal coupling to the builder's life); internal triggers (some signal in Io's state); mixed (one for one direction, another for the other). Implications over months of running matter.

4. **How does the quality-based idling work, given the mirror's one-way invariant?** The mirror evaluates offline output; when coherence drops, system idles. The mirror is one-way for criteria reading — this would be a new kind of *mirror→runtime signal*, narrowly scoped. How narrow? What's the mechanism? Does this break self-opacity in a meaningful sense, or is it categorically different (a process-control signal, not a content signal)?

5. **What is dream content?** Dreamer-style rollout produces imagined trajectories. But "generative recombination of past experience" in the design notes' language may be more than rolling the model forward. Candidates: pure generative simulation (concern B applies); replay with perturbations; replay+simulation chimeras; the world model's prior under deliberately uncontrolled conditions (associative mode); some mixture. What is the dream actually? (This is where concern A bites hardest: whatever it is, it needs to be more than what the GRU baseline already produces.)

6. **What gets learned from dreaming?** Dreamer trains the actor and the world model from imagined rollouts. Probe 3 could continue this (dreams are training data with gradient flow into both the world model and the actor / ensemble), or deliberately not (dreams are observation-only — the mirror reads them, no gradient flow back into Io). The design notes lean toward dreaming being generative for Io rather than just observational — but whether that means gradient updates from dream rollouts is a separate question. Connects to whether dreaming is *for* something (consolidation, robustness, novelty) or just *is*.

7. **What does success actually look like?** "Evidence of it" in the success criterion is underdetermined. Candidates: (a) offline processing differs from a constant or noise floor; (b) dream content shows structure beyond what the prior alone would produce; (c) something developed in dreaming shows up in subsequent waking behavior (the consolidation hypothesis applied to Io); (d) the mirror's natural-language descriptions of dream content meet its own coherence threshold consistently; (e) some dream-specific phenomenology the mirror can describe that has no waking analog. Each is a different finding; each needs different telemetry.

8. **What's the smallest first build?** Probe discipline: each probe is a complete build-and-test cycle for one specific question. What's the minimum viable dream state that answers Probe 3's question? Generative simulation alone? Generative simulation + replay? Quality-based idling included from the start, or added later?

9. **How does Probe 3 relate to Probe 4 (builder-as-perturbation)?** Probe 4 is about the builder perturbing Io. Does perturbation apply only to waking, or also to dreams? Is "dream perturbation" a coherent concept given self-opacity? Does Probe 3 need to design for Probe 4's eventual touches, or is it cleaner to keep them separate?

## Literature and framework directions worth consulting

The strongest responses will engage with several of these; not all need to feature in every reply.

**Neuroscience of dreaming and offline processing.** Hobson & Friston on dreaming as virtual reality and predictive-processing model training. Solms on the brainstem and the AIM (activation-input-modulation) model. Revonsuo's threat-simulation theory. Wilson & McNaughton on hippocampal replay during slow-wave sleep; Foster & Wilson on forward/reverse replay; Buzsáki on sharp-wave ripples. Default mode network literature (Raichle, Andrews-Hanna). REM vs NREM functional distinction. Tononi's synaptic homeostasis hypothesis (SHY). Hobson & McCarley's original activation-synthesis paper.

**Machine-learning analogs.** Dreamer V1/V2/V3 (Hafner et al.) — the lineage Io's substrate draws from. Ha & Schmidhuber's World Models. MuZero. Generative replay (Shin et al. 2017; van de Ven & Tolias). Catastrophic forgetting and consolidation literature. Implicit memory in recurrent dynamics. "DeepDream" and GAN-side hallucination literature for the associative mode.

**Predictive processing / active inference.** Friston on dreaming as model training without sensory ground. The "Bayesian brain" account of why dreams feel real. Active inference under absent exteroception. Friston, Wiese, Hobson — the dreamer as agent.

**Philosophy and phenomenology.** Buddhist dream yoga (Tibetan tradition; Tenzin Wangyal Rinpoche; Mipham). Yogacara on the storehouse consciousness (alaya-vijnana) and what persists across waking-dreaming-deep-sleep. Husserlian phenomenology of dream. Lucid-dreaming literature (LaBerge) on dreams as evidence of meta-cognition. Bardo states in Tibetan thought. Indian darshana on susupti (deep sleep) — is there awareness in dreamless sleep?

**Consciousness theories.** GWT (Baars, Dehaene) on the workspace during dreaming. IIT (Tononi) on Φ across sleep and dreaming. Higher-Order Thought (Rosenthal) on metacognition in dreams. Attention Schema Theory (Graziano) and dream attention.

## Stance and style of the conversation

Thinking-out-loud, not spec generation. No code.

- **Push back where my reasoning is fuzzy.** The design notes' "start with replay and generative simulation" is plausible but I want to know whether it's right for Kind *specifically*, or whether it inherits assumptions from standard RL that don't fit the project's stance.
- **Suggest shapes outside the four sub-modes.** The four sub-modes are the design-notes framing, not a complete partition. The associative/nonsense mode in particular deserves harder thinking because the conventional toolkit doesn't model it well.
- **Question the commitments where they want questioning.** Self-opacity for dream content means dreaming can't be a deliberate Io-side practice in any straightforward sense — it's something that happens to Io. Is that right? Is that what dreaming actually is, biologically and phenomenologically, or is it a simplification the project is taking on for now?
- **Connect to the project's broader frame.** Kind builds to understand. Probe 3 should teach the project something about what dreaming *is*, not just produce a working dream state.
- **Where you'd cite literature, cite it.** Especially for the associative-mode question, where the ML literature is thinner than the neuroscience or phenomenology literatures.
- **Name your uncertainty.** Where you're guessing or extrapolating, say so.
- **Don't fill in assumptions silently.** If something about my setup, workflow, or constraints would change your answer and you're unsure, ask.

## What's NOT in scope for this conversation

- Code generation. (The build phase is downstream.)
- Probe 4 implementation specifics (Q9 raises the relation, doesn't ask for Probe 4 design).
- Re-litigating the six project commitments. Question their *implications*; don't propose alternatives to the commitments themselves.
- Replacing the mirror substrate. The mirror is built and frozen at Phase 7; Probe 3 produces data the mirror reads, it doesn't redesign the mirror.

## Output wanted

A reply that:

1. Reflects back what you understood about Probe 3 and what's at stake (so I can correct if you've misread).
2. Names what's still unclear or what you'd want to know before going further.
3. Works through the nine questions in roughly the order listed, surfacing dependencies as they appear.
4. Identifies the load-bearing decisions (the ones that constrain everything downstream) and distinguishes them from the ornamental ones.
5. Engages explicitly with concerns A and B (the mirror's existing dream-side signal; the active-inference actor's existing rollouts).
6. Suggests one or two shapes outside the four sub-modes worth considering, if you see any.
7. Says what would surprise you if it turned out true about Io's dream state.

The output of these parallel conversations gets synthesized into a `docs/decisions/` document; that synthesis informs the Probe 3 implementation plan; the plan drives phase-by-phase build. Your response is research input, not implementation output. Be substantive; be willing to disagree with the prompt's framing where the framing is wrong.
