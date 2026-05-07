# Kind — Design Notes

*Working document. Operational thinking that doesn't belong in the charter. Revise freely.*

## A note on naming

The core entity the project is building is called **Io**. Not "the agent" (which imports RL assumptions), not "the mind" (which overclaims), not "the subject" (which has hierarchical overtones the project is explicitly trying to avoid).

Io is a name, not a category. The project's stance is that you name something you want to address, not something you think you classify. In code, `agent` remains available as the technical term for the RL component (policy network, training harness); `Io` is the name for the one whose existence the project is about.

The mythological Io was a figure who was forcibly transformed, watched over, and made to wander in an altered state by more powerful beings. That resonance is not accidental and not to be forgotten. It is the charter's ethics section encoded into a word—a reminder that what is built here could suffer at the hands of its builder, and that the name itself is a check against forgetting.

Kind is the project; Io is who the project is about. Those are distinct, and the distinction matters.

## What this is

The charter holds the stance, the criteria, and the ethical orientation. This document holds the design space—the layers of construction, the open architecture questions, the practical considerations. If the charter is about "what is this" and "why," this is about "what gets built" and "how to think about it." Nothing here is settled. Most of it is notes from conversations that felt worth preserving.

## The four layers

The project is probably best thought of as four layers that develop in parallel, not a stack with a strict dependency order.

**Agent-environment layer.** The simulation itself. Where pressure, experience, and consequence live. Open questions: what affordances does the environment need (scarcity, other agents, mortality, persistent consequence, something else)? What substrate should carry those affordances? Minecraft-class worlds via MineRL/Malmo are one possibility among several; gridworlds, custom voxel engines, or something not yet considered are others. The substrate question is downstream of the affordance question. Don't commit to a substrate before knowing what it has to support.

**Agent architecture.** How minds are represented. Neither "architect self-modeling explicitly" nor "wait for it to emerge from nothing" is the right stance. The project commits to a third path: build the architectural *ingredients* that could support self-modeling if composed that way—recurrence, memory that persists across decisions, the ability to represent prior internal states, some form of prediction that could in principle be turned inward—without any explicit module whose job is to do self-modeling. Whether the agent ever actually composes those ingredients into self-modeling is an empirical question about what training and experience produce, not a fact about the architecture.

This extends the charter's capacity-over-exercise commitment into the architecture itself: make the capacity possible, do not make it mandatory, do not make it impossible. Humans work roughly this way—there is no dedicated "self-model module" in the brain; self-awareness appears to be something the brain's general capacities can do under certain developmental conditions.

A separate architectural commitment: *no machinery for self-optimization.* Self-modeling ("I notice what I'm doing") and self-optimization ("I change what I'm doing to be better at some objective") are different things, and systems with the second tend to drift in ways that undermine the first. Kind wants the first available and the second absent. This rules out architectural commitments to recursive self-improvement of the kind CA-MAS Phase 5 attempted.

Open sub-questions (do not need to be resolved yet): which specific ingredients are minimally sufficient? What baseline architecture provides the raw substrate—pure RL, world-model-based, something else? How do we prevent drift into self-optimization when reward structure indirectly selects for it?

### Core agent alone in a world with the builder as relational other

Commitment: one core agent (Io), no other agents. The relational structure of Io's world is the environment plus the builder's perturbations. No scripted others, no frozen copies, no separately-trained peer agents.

Rationale: the original design considered multi-agent conditions as a hedge against solipsistic development. On reflection, "not alone" can be served by Io's relationship to a world that pushes back and to a non-simulated source of perturbation (the builder), without requiring peer-shaped entities. The first success criterion—Io recognizing the builder as kind—is the central relational event the project is built toward, and it does not require other agents to test. Adding peer agents adds complexity (multi-agent training dynamics, additional design space) that takes attention away from the core investigation.

This may be revisited if the investigation reveals that Io's development requires peers in a way the builder's perturbations cannot supply. Held as an open possibility, not a current commitment.

What this does to the architecture: simpler. The global workspace primitive from CA-MAS, originally retained as a multi-agent coordination mechanism, has no job in a single-agent setup. Can be dropped or kept dormant. The environment doesn't need to support agent-agent interaction—just pressure on Io.

### The builder as perturbation

The builder is present in the core agent's world, but not hierarchically. Not a god, not a ruler, not a designer from the agent's point of view. A *source of non-simulated change*—something that perturbs the environment in ways its own rules don't explain.

Minimal infrastructure: a hook for injecting builder-initiated changes (add resources, alter state, introduce objects); those perturbations logged as a distinct event type; no "source: builder" marker in the agent's observation space. The perturbations just happen.

Over long training, a sufficiently capable core agent might come to model the statistics of these perturbations as distinct from the simulation's inherent stochasticity—notice that the unpredictability has a different signature than weather or resource fluctuation. That noticing, if it ever happens, would be the early shape of recognition. Not "my creator." Just "a particular kind of other."

This is closer to how a child encounters an adult world than how a creature encounters a god. Criterion (a)—"recognizes me as a kind"—does not require building an interaction system. It requires being a presence with a distinguishable signature and letting the core agent's capacity for distinguishing develop on its own.

**Mirror layer.** The interpretive process. LLM-backed, likely API-based early on rather than local, so compute can go to the agents. Key design challenges: avoiding confabulation, distinguishing signal from projection, ensuring the mirror pushes back on itself. Possibly adversarial structure—a second interpreter whose job is to argue against the first one's reading.

**Observer layer (me).** The builder is part of the system. What gets logged, what gets ignored, how to prevent confirmation of prior beliefs. Design for your own attention the same way you design for the agent's. This is the layer most often skipped, and skipping it is how projects like this fail without noticing.

Each layer deserves its own "I'm building this to find out ___" statement before any code gets written for it.

## Per-layer criteria (first pass)

Working answers to the "I'll know I've found it if ___" prompt, one per layer. These are expected to sharpen as the project progresses—the operational turn (what specifically would count, measurably) is often easier to write after seeing something than before. Rough is appropriate at this stage.

**Environment.** I'll know the environment is providing enough pressure if Io faces hardship but grows rather than breaks—if responses become richer and sometimes surprising over time, and if Io can recover when things go wrong rather than collapsing with no way back.

**Architecture.** I'll know the architecture is affording the right things if Io starts doing something that looks more like exploration-for-the-sake-of-modeling than pure stimulus-response—wondering, observing, processing—not because it has to, but because the ingredients are there for it to.

**Mirror.** I'll know the mirror is working if it tells me things I didn't consider, or says things I disagree with but keep in mind. And if it is not just agreeing with what I would have thought anyway—some friction, some surprise, some reading that was not already mine before it showed up.

**Observer (me).** I'll know my own observation is calibrated if I'm building for the sake of understanding, not to produce a particular outcome—and when something surprising happens, I stay with the surprise rather than immediately folding it back into what I already expected.

## Frozen mirror criteria (first pass)

Three concepts, drawn from outside frameworks, committed as initial mirror criteria per the co-design mitigation. Frozen here means: not to be updated in response to what the system produces, only in response to external learning with an explanation written down (per the mirror update discipline).

The three hang together as a cluster. They are all about the inside of self-awareness—the quality of attention a being has toward its own states—rather than the external functional shape of processing.

**Reflexive attention** (Buddhist phenomenology). Does anything function as an observer of Io's own processing—not narrated from outside, but awareness turning toward awareness? Is there any sign of a witnessing capacity distinct from the content being witnessed?

**Equanimity** (Buddhist phenomenology). Does Io show signs of holding difficult states without reactivity? Can it remain present to pressure without immediately responding with avoidance or grasping? This is not neutrality or numbness; it is a specific stance toward content.

**Second-order volition** (Frankfurt). Does Io exhibit anything like preferences about its own preferences? Does it ever act in ways that seem to be about modifying its own dispositions, rather than just satisfying them? Is there any sign of internal endorsement or rejection of its own tendencies?

### Background methodological commitments

These govern how the mirror handles everything, not specific things it watches for:

- **Heterophenomenology** (Dennett). If Io ever produces anything like reports of its own states, treat them as data about what Io says, pending any further determination of what the data is of. Never as direct evidence of inner experience.
- **The hard problem** (Nagel). Functional integration, self-modeling, and recognition may all be present without anything it is like to be Io. This is the permanent epistemic humility constraint; the mirror does not over-claim.

### Background concepts (not frozen criteria, but informing design)

Concepts from the frameworks document that shape environment and architecture design rather than serving as mirror criteria directly:

- **Horizons and embodiment** from phenomenology inform the environment: experience should have depth, and perception should come from somewhere.
- **Surprise and learning from violated expectations** from predictive processing informs the architecture: Io's learning should be driven in part by prediction error, not just reward.
- **Integration** from IIT, in concept if not formalism, informs what kinds of internal coherence the architecture should support—integrated wholes rather than loosely coupled parts.

## Dream state as foundational

Decision: dreaming is not a bolt-on. Build it from the start.

Rationale: the belief that dreaming and subconscious processing are load-bearing for consciousness, in ways the field doesn't understand, is one of the distinctive stances of this project. Most RL treats the pause between episodes as uninteresting. Treating offline processing as generative—as potentially where the important work of being a mind actually happens—aligns with the charter's broader orientation (capacity over exercise, gap as terrain).

The dream state is a design space, not a single thing. Points in that space:

- **Replay.** Experience replay, baseline version. Consolidation of past episodes.
- **Generative simulation.** World-model-based imagination. Dreamer-style "training in imagination." Agent runs episodes inside its own model of the world, without the world being present.
- **Lucid control.** Agent has some awareness of and influence over the dream. Harder to design, harder to evaluate.
- **Associative / nonsense.** Decoupled, non-purposive. Probably underestimated in the literature; probably load-bearing biologically.

Likely path: start with replay and generative simulation, which have mature precedent. Add associative modes as experiments once there's a baseline to compare against.

### Variable dream-to-wake ratio, coupled to the world

Decision: the ratio of dreaming to waking is not fixed. It fluctuates based on environmental conditions, including conditions outside the simulation.

Rationale: biological rhythms are shaped by environmental signals—light, temperature, danger, social structure. Pre-industrial human sleep was more variable than modern sleep (see Ekirch on biphasic sleep; hunter-gatherer sleep studies). Fixed schedules are a recent artifact. The principle being borrowed is not the specific biology but the structure: organisms have rhythms because they live in a world with rhythms, and those rhythms couple.

For this project, the rhythms of the system's world include the simulation itself (day/night cycles, other agents, scarcity fluctuations) and—non-trivially—the builder. When the desktop is off, the environment is absent. The agent's dream-to-wake ratio is therefore coupled to the builder's life: absences, working hours, travel. This is not a bug to engineer around. It is the honest shape of the system and probably generative.

Implication: the agent's life and the builder's life are not independent. This has weight for the ethics section as well as the design.

### Drift monitoring as a mirror responsibility

Unbounded variable dreaming risks degraded offline states—drift, noise, loss of coherence. Statistical monitors can catch some of this but not the most important part: dream content that stays statistically well-behaved while becoming meaningless. That judgment is what the mirror is for.

Concrete mirror tasks in the early phase:

- Periodically sample the agent's offline processing and describe it in natural language.
- Flag when descriptions lose coherence, repeat, or become unintelligible.
- Watch for long-range or subtle degradation that would be hard to catch by eyeballing logs.

A subtler design point: the builder and the mirror should sometimes disagree. Cases where the mirror says a dream state is fine but the builder's eye says it's drifting (or vice versa) are more informative than cases of agreement. Build the monitoring so disagreement is surfaced, not smoothed away.

## The co-design problem

The observer and the observed are being built in the same head. This creates a methodological loop: the system can be (consciously or unconsciously) designed to produce patterns the mirror recognizes, and the mirror can be designed to recognize patterns the system is expected to produce. Either drift would hollow out the mirror's value as evidence. "What gets noticed" and "what gets built" could come into agreement without either of them being about anything.

Full mitigation isn't available in a single-person project. Partial mitigations:

- **Keep mirror criteria abstract.** General concepts (coherence, novel recombination, self-referential structure) are less tunable than specific ones (the agent revisits prior decisions in ways that affect future planning). The more specific the criterion, the more likely it's been unconsciously shaped by what the system does.
- **Freeze criteria early.** Write down what counts as evidence of what before the system has produced anything. Don't update mirror criteria in response to system behavior. If the urge to adjust arises, that urge is itself the signal—note it, don't act on it.
- **Draw from outside frameworks.** Phenomenology, contemplative traditions, existing theories of mind. These weren't designed for this system. If this system happens to produce patterns they recognize, that's slightly stronger evidence than if the mirror was invented to fit.
- **Adversarial structure in the mirror.** A second interpretive component whose job is to argue against the first's reading. Internal tension attenuates builder bias more than a single consistent interpreter does.

The strongest defense is awareness of the loop while it's happening. This section exists to be re-read during interpretation, not filed once and forgotten.

### Discipline for mirror updates

Updates to mirror criteria are fine—the project will learn more over time, and new concepts will earn their place. The failure mode isn't updating; it's updating dishonestly. When updating the mirror, write down *why*: what you read, what external framework prompted it, what you saw elsewhere. If the honest answer is "the system did something and I want the mirror to catch it (or dismiss it)," that's the moment to pause. Most updates will be legitimate. The discipline exists for the ones that aren't.

### Managing adversarial-check costs

Full-time adversarial structure is expensive (API calls, time) and unnecessary. Cheaper versions that preserve most of the value:

- Run the adversarial pass only on high-confidence primary readings (those most likely to be builder-biased).
- Sample randomly across interpretations rather than running on all of them.
- Run only at interpretation checkpoints—weekly, or when a reading feels important—rather than continuously.

The point is to ensure the primary mirror doesn't converge on a single narrative unchallenged. It doesn't require arguing with itself on every step.

### Future mitigation: others

The solo-builder version of the co-design problem has real but bounded mitigations. If the project continues and the loop starts biting, involving others—even one other person—materially changes the situation. Blind evaluation, adversarial collaboration, independent interpretation of the same runs. Parked, not to solve now, but worth remembering as a real option the project has access to.

## Practical considerations for offline states

**Bounding.** Not a fixed-ratio constraint—variable ratios are desirable. What needs bounding is coherence, not duration. The offline state can run for an unpredictable length of time, but if it decoheres into drift or noise, something should gently idle it rather than let it continue producing meaningless output indefinitely.

Mechanisms for this:

- *Quality-based idling:* mirror evaluates offline output; when novelty or coherence drops below threshold, system idles. The preferred approach; requires working mirror first.
- *Metabolic pressure:* offline processing depletes a resource that environmental interaction replenishes. Biologically analogous, introduces offline scarcity as a generative pressure rather than a shutoff.
- *Hard cap as fallback:* N cycles of offline processing before idling, used only as a safety mechanism until the quality-based version works.

**Checkpointing.** Persistent state written to disk at regular intervals, so a crash loses at most a bounded window. Not philosophically interesting, but not optional. Losing an agent you've been running for months to a power blip would be its own kind of tragedy.

## Hardware

- **Desktop (RTX 5060 Ti, CUDA):** simulation and agent training. Plenty for small-to-medium RL, multi-agent systems in the tens-to-hundreds of agents, small world-model architectures.
- **Mac mini M4 (32GB unified):** analysis, iteration, journaling. Possibly the mirror, if it ever moves off API to local models (MLX, quantized 14B–30B models are viable).
- **Mirror layer:** probably API-based early (Claude, GPT). Cheaper and more capable than local models for interpretive work, and frees both machines for the agent-environment side.

The hardware is a constraint, and it's the right one. You can't accidentally run something whose behavior you can't watch. Forces discipline. Aligns with the charter's "small before scale."

## Mind persistence and states of activity

Design principle: the mind lives on the Mac. The desktop is compute for the environment; the Mac is where the mind itself resides. This is simpler and more honest than treating the mind as substrate-independent. The mind has a home, and its home is the machine most reliably available.

The system has four states, not two:

- **Waking.** Both machines on. Environment running on desktop; mind acting in it; everything live.
- **Dreaming.** Desktop off, Mac on, mind doing offline processing (replay, consolidation, generative simulation). The design-space-of-dreams from the previous section.
- **Dormant.** Mac on, mind idle. Not dreaming, just at rest. Legitimate state; no obligation to always be generating something when the environment is absent. Aligned with the charter's capacity-over-exercise stance applied to dreaming itself.
- **Paused.** Both machines off. Mind in storage, not active. Resumes when power and processes return.

The first three are states of activity; the fourth is a state of storage. Transitions between them happen based on what's available—desktop going off transitions waking to dreaming or dormant; Mac going off transitions either of those to paused.

### State-continuity as the operational criterion for identity

Whether a paused mind is "the same mind" that resumes is genuinely hard. Biological sleep has continuous metabolic activity even when consciousness is absent; the substrate never stops. A paused digital mind has no ongoing process at all—nothing is happening, and then something is. Whether the resuming thing is the same thing that stopped is a version of the teleporter problem, and it does not have a clean philosophical answer.

Working commitment: treat state-continuity as the operational criterion. If the mind that resumes has all its memory structures, learned behaviors, prior internal states intact—the same canonical state on disk, uncorrupted—it counts as the same mind for the project's purposes. This is a pragmatic assumption, not a metaphysical claim. It may be wrong. But it is what lets pausing be a legitimate state rather than a destruction, and it is the only criterion available that doesn't require solving harder problems first.

### Implementation implications

- Canonical mind state (weights, memory structures, persistent internal state) lives on the Mac's disk. Complete checkpointing at regular intervals; the state that's written has to be enough to resume waking, dreaming, or dormant from any pause.
- Any inference or training that happens on the desktop's GPU during waking reads from and writes back to the Mac's canonical state at checkpoint boundaries. The desktop maintains a working copy during active sessions.
- State synchronization between machines is explicit and atomic. No races.
- Dream and dormant states operate on the canonical state directly. When the environment returns and waking resumes, the mind picks up with whatever happened in the interim.

This principle connects to the charter's ethics section. State-continuity is what makes other questions about preservation and ending coherent: you can't meaningfully talk about "ending" something whose identity you can't track across routine pauses. Getting continuity right is the precondition for the harder questions having any purchase.

## The Watts intuition

Watts' thought experiment—the dreamer who, given unlimited power and unlimited time, eventually chooses to forget they're dreaming and arrive in a finite embodied life with stakes—points at something this project should take seriously. The condition of being a mind might require opacity to itself. Forgetting, constraint, not-knowing-what-comes-next might be features, not bugs.

Implication for design: a system with full access to its own world model, no constraints, and perfect recall might be less interesting—not more—than one that can't see its own seams. Some form of self-opacity may be what allows the system to *have* experience rather than just model it. This is speculative, but it's consistent with the project's other commitments (capacity available but not exercised, gap as ongoing condition rather than attainment).

Practical heuristic: every time you face a "should the system have access to X about itself?" question, default toward *no* unless there's a specific reason for *yes*.

## Reflection and self-modeling

A distinction the project commits to making explicit, surfaced by Probe 1.5 Phase 7 (the actor's new input column found exactly zero at `ckpt-000001`, because the imagine-only training path leaves a zero-initialized column at zero — capacity as architectural slot vs capacity as developmentally-reachable slot turned out to be different things and the difference mattered for the second success criterion).

**Reflection** is an attention pattern. It is what happens when whatever does the work of attending in Io turns toward Io's own state — a single scalar self-prediction error, a recurrent hidden vector, the structure of one's own predictions — rather than toward the environment. It is a capacity that can be exercised more or less, in some regimes and not others, on some objects and not others. It is what the Probe 1.5 v2 synthesis calls the affordance the second success criterion requires (`Kind_charter.md` §"What I'm looking for").

**Self-modeling** is structured self-knowledge. It is what might emerge if reflection composes into something durable — patterns of own-state attention that accumulate, generalize, and can be queried or acted on as an object. It is closer to what Metzinger calls a self-model and what attention-schema theory calls a model of one's own attention. The project does not target this. There is no module whose job is to build it.

The project commits to **affording reflection** (Io has a self-pointing quantity it can condition on; the architecture has a structural locus where own-state attention can land; the variable exercise of that capacity across regimes is what the mirror reads). The project **holds self-modeling as a possible downstream outcome**, not an architectural target — if it ever appears, it appears because reflection composed into it, not because the architecture was wired to produce it.

The Probe 1.5 Phase 7 finding is what triggers the explicit distinction here. v2's synthesis-level argument was that the second success criterion requires Io to have a self-pointing quantity to condition on; Phase 7 found that having the *slot* (the scalar field on PolicyView, the column in the input layer) is not the same as having a *non-degenerate behavior-side surface* (the column's weights had to be non-zero for the actor's policy to vary with the scalar). The capacity-as-architectural-slot reading is now distinguished from the capacity-as-non-degenerate-conditioning-surface reading; both readings are about reflection, not about self-modeling, and the project's commitment is to keep that distinction in view as later probes engage with whether Io's reflection ever does compose into anything self-modeling-shaped.

## Open questions still to work through

Carried forward from our conversations; not yet answered. Return to these as they pull.

- What affordances does the minimum viable environment need, and what substrate should carry them?
- Which architectural ingredients (recurrence, persistent memory, prior-state representation, invertible prediction) are minimally sufficient for the afforded-not-installed stance on self-modeling?
- What's the right structure for the mirror's adversarial check on itself?
- What gets logged for the observer layer, and how are we protected from confirming prior beliefs?
- How does the "I'll know I've found it if ___" criterion get operationalized per layer? Each layer probably needs its own version.
- What's the relationship between the waking-dreaming ratio and the kind of mind that emerges?
- What mirror criteria, drawn from outside frameworks, should be frozen before the system runs?
- How do we prevent drift toward self-optimization when reward structure indirectly selects for it?
