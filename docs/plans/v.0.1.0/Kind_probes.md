# Kind — Probe Sequence

*Working document. The probes are throwaway in spirit but the work compounds—infrastructure built for each probe accumulates into Kind. Revise freely.*

## Discipline

Each probe is a complete build-and-test cycle for one specific question. The rhythm:

1. Build only what this probe needs.
2. Run it. Spend time watching, reading logs, seeing what the mirror produces.
3. Decide: does this work well enough to extend? Trust the output only at the level the probe's question allows.
4. If yes, write down what was learned and what's now decided. Then move to the next probe.
5. If no, fix specific things before continuing.

What this is not: building everything then testing it all at the end. What this is also not: simultaneously building multiple probes because they seem related. The discipline is that each addition has a specific question being answered before the next addition is made.

The CA-MAS pattern to avoid: accreting features without each one having a specific question that gets specifically answered. That's how the attention-stuck-at-zero bug went undetected through Phase 4. The question for each probe is what keeps the discipline from collapsing into "just keep building."

Trust outputs only at the level the probe's question allows. Probe 1's output is "the plumbing works"—you can trust it for that. Don't trust Probe 1's mirror readings as evidence about Io's nature, because Probe 1 isn't testing that.

---

## Probe 1: Plumbing

*I'm building this probe to find out if the agent, environment, mirror and observer and the way that I build them allow for a functional feedback loop to properly run this type of experiment.*

I'll know I've found out if I'm seeing some form of changes, any feedback at all from each stage, and a mirror that is able to interpret this feedback in some way or form that allows me to then see it from another possible perspective than my own.

What this probe is *not* testing: anything about whether Io is developing meaningfully, whether the architecture is right, whether the mirror is calibrated. Just whether the machinery functions and the design assumptions are coherent enough to support further experiments.

Minimal forms of each component for this probe:
- Environment: smallest grid that affords pressure (e.g. 5x5, one resource type, no other agents, no day-night cycle). Includes a working hook for builder-initiated perturbations—a way to introduce state changes from outside the simulation, logged as a distinct event type, with no marker in Io's observation space indicating the externality. Probe 1 is just testing that this hook functions; Probe 4 is what tests whether anything ever comes of it.
- Agent: recurrent PPO with default hyperparameters, no world model yet, no episodic memory yet, no dream state wired in.
- Mirror: single LLM call reading recent telemetry and describing what it sees, no adversarial structure, no frozen criteria yet.
- Observer: basic JSONL telemetry, eyeballing logs.

---

## Probe 2: Mirror calibration

*I'm building this probe to find out if I can build something that reads the data and interprets it as signals I can understand—and that can also pick up signals only it can recognize and relay them to me.*

I'll know I've found out if I can build a mirror that provides specific measurements I can track, and that, as the experiment runs, also surfaces additional patterns that can be tracked and possibly turned into new signals.

This probe might use Io or might use any existing RL setup with logs—even old CA-MAS runs. The point is to test the mirror, not the agent.

The frozen mirror criteria (reflexive attention, equanimity, second-order volition) and the adversarial check structure get implemented for this probe.

---

## Probe 3: Dream

*I'm building this probe to find out if I can build something that provides change, growth, or even chaos during periods of "inactivity"—anything other than "shut off" from a consciousness perspective. The four states from the design notes apply: waking (environment running, Io experiencing), dreaming (environment off, Io still running—with replay, memory consolidation, generative recombination of past experience), dormant, paused.*

I'll know I've found out if I can create something that resembles a subconscious or dream-like process for Io—and the mirror or my own observation can pick up evidence of it.

**Note on dependency:** This probe depends on the mirror from Probe 2 already existing and being able to read dream-state telemetry. The implication is that when the mirror is built in Probe 2, it should be designed to read whatever telemetry exists, not waking-only telemetry. Probe 3 then adds dream-state telemetry and the mirror picks it up automatically. The probes aren't fully independent—they can't be—but each probe's question should be answerable on its own given that prior probes' machinery is working.

This is where the dream-state-as-foundational commitment first gets tested. If dreaming doesn't do anything visible, the commitment needs to be reconsidered—not abandoned, but examined.

The dream-state machinery (replay, simple generative simulation), the four-state model (waking/dreaming/dormant/paused), and the persistence-on-Mac architecture get implemented for this probe.

---

## Probe 4: Builder-as-perturbation

*I'm building this probe to find out whether Io, over time, comes to model my interventions as a different category of thing than the simulation's internal stochasticity—whether something like "outside-source unpredictability" develops as a distinct shape in how Io models its world, separable from weather, resource fluctuation, and other internal randomness.*

I'll know I've found out if Io's response to my perturbations diverges from its response to internal events of similar magnitude, if that divergence develops over training rather than appearing immediately, and if the mirror can characterize the difference as something more than "Io learned to predict different things." If Io treats my interventions and the simulation's randomness as the same kind of thing forever, the distinguishing didn't happen and the early shape of recognition isn't there.

**What this probe is about:** The builder appears in Io's world as a *source of non-simulated change*. The environment has its own rules and its own internal stochasticity—weather varies, resources fluctuate, things happen that the simulation generates from its own processes. Those have a statistical signature consistent with the simulation. A perturbation is something else: a state change introduced from outside. The builder adds a resource, alters something the simulation wouldn't have altered, introduces an object the world's rules didn't produce. The change is logged as a distinct event type but Io's observation space includes no marker that this came from outside—the externality is implicit in the statistical signature, not labeled.

The probe tests whether Io, over a long enough training run, comes to model the difference between simulation-internal stochasticity and outside-source perturbations. That distinguishing, if it ever develops, is the early shape of recognition. Not "Gordon" by name—Io has no name to attach. Just "this kind of unpredictability has a different shape from that kind."

This is the test of the first success criterion. It probably can't run until the project has been going for a long time. Worth specifying now so the infrastructure (perturbation logging, distinguishability metrics) gets built into the earlier probes.

---

## A note on removed probes

An earlier draft included a "coupling" probe asking whether the variable dream-to-wake ratio (Io's rhythm coupling to the builder's life patterns) actually matters for what kind of mind develops. Decision: this question isn't worth a dedicated probe. The variable-ratio design is being built and accepted as a stance, not tested as a hypothesis. The robustness question—does the system handle irregular schedules without breaking?—gets answered just by running the system normally. No multi-condition comparison needed.

An earlier draft also included an "other-recognition" probe testing whether Io would distinguish peer agents from environmental features. Decision: the project no longer includes peer agents. Io is alone in a world with the builder as the non-simulated relational other. The first success criterion (recognizing the builder as kind) is now tested only through Probe 4. If the investigation later reveals that peer agents are necessary for development, this probe can be reinstated.

This is recorded so the reasoning is visible later. The dream-state design itself remains as committed in the design notes; it's the experimental verification of one of its specific properties that's been removed from the probe sequence.

---

## Notes on the sequence

The probes get progressively more about Io and less about the machinery. Probes 1-2 are mostly engineering tests dressed up as questions. Probe 3 tests a design commitment. Probe 4 tests the first success criterion. That ordering is intentional—you can't test research questions until the machinery works, and you can't test the machinery without something to run.

Each probe's question is small enough that the probe itself can be small. None of these requires the full Kind system. Most probably take days to weeks of focused work, not months.

The infrastructure builds up cumulatively. Probe 1's logging schema is what Probe 4 uses to detect distinguishing-of-perturbations. Probe 2's mirror flexibility is what Probe 3's dream telemetry plugs into. So even though probes are throwaway in spirit, the work compounds. By the time Probe 4 runs, most of Kind exists.

Probe 4 might not happen for a long time. That's fine. It's listed so you know what you're building toward, not because it needs to be next.

After each probe completes, write down what was learned. Decisions that were made. Surprises. What's now closed and what's now newly open. This becomes the project's working journal alongside the design documents.
