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

*I'm building this probe to find out if the full loop—environment producing observations, Io acting on them, mirror reading what happens, observer logging it all—runs end-to-end without breaking, and whether anything in that loop produces output worth looking at.*

I'll know I've found out if:

- The loop runs for a sustained period without errors.
- Telemetry captures what I expected it to capture.
- The mirror produces at least one reading that wasn't already implicit in the raw logs.
- I can identify what's working and what isn't well enough to make decisions about the next probe.

What this probe is *not* testing: anything about whether Io is developing meaningfully, whether the architecture is right, whether the mirror is calibrated. Just whether the machinery functions and the design assumptions are coherent enough to support further experiments.

Minimal forms of each component for this probe:
- Environment: smallest grid that affords pressure (e.g. 5x5, one resource type, no other agents, no day-night cycle).
- Agent: recurrent PPO with default hyperparameters, no world model yet, no episodic memory yet, no dream state wired in.
- Mirror: single LLM call reading recent telemetry and describing what it sees, no adversarial structure, no frozen criteria yet.
- Observer: basic JSONL telemetry, eyeballing logs.

---

## Probe 2: Mirror calibration

*I'm building this probe to find out whether the mirror produces output that tracks something real, or whether it produces plausible-sounding narratives regardless of what's happening in the system it's interpreting.*

I'll know I've found out if:

- When the mirror watches Io doing genuinely different things, the readings differ in ways that correspond to the differences.
- When the mirror watches matched conditions—say, two runs with the same setup—the readings are similar in ways that correspond to the similarities.
- The inverse test: if I can deliberately produce uninteresting behavior and the mirror still narrates it as interesting, the mirror is confabulating.

This probe might use Io or might use any existing RL setup with logs—even old CA-MAS runs. The point is to test the mirror, not the agent.

The frozen mirror criteria (reflexive attention, equanimity, second-order volition) and the adversarial check structure get implemented for this probe.

---

## Probe 3: Dream

*I'm building this probe to find out whether offline processing (replay, generative simulation) actually changes Io's behavior in ways that wouldn't have happened without it.*

I'll know I've found out if:

- Io's behavior after a period of dreaming is measurably different from Io's behavior after the same wallclock time spent in dormancy.
- The difference is not just in performance metrics but in something the mirror can describe—new patterns, recovered capabilities, novel responses.
- Lesioning the dream component causes those differences to disappear.

This is where the dream-state-as-foundational commitment first gets tested. If dreaming doesn't do anything visible, the commitment needs to be reconsidered—not abandoned, but examined.

The dream-state machinery (replay, simple generative simulation), the four-state model (waking/dreaming/dormant/paused), and the persistence-on-Mac architecture get implemented for this probe.

---

## Probe 4: Coupling

*I'm building this probe to find out whether the dream-to-wake ratio actually matters for what kind of mind develops, or whether it's a design commitment without empirical consequences.*

I'll know I've found out if:

- Io trained under different ratios (more dreaming, less dreaming, irregular ratios that match real life patterns vs. fixed ratios) produces detectably different behavior over time.
- The differences are interpretable—not just noise, but something the mirror can characterize as "this Io is more X than that one."

This probe tests whether the philosophical commitment to variable-ratio coupling has technical purchase, or whether the system is robust to ratio variation in ways that make the commitment moot.

Most of what this probe needs already exists from Probe 3. The new work is mostly in running multiple configurations and comparing.

---

## Probe 5: Other-recognition

*I'm building this probe to find out whether Io comes to distinguish other agents from environmental features—whether the relational structure of the world is something Io models differently than physics.*

I'll know I've found out if:

- Io's behavior toward other agents differs systematically from Io's behavior toward objects with similar physical properties.
- The difference is not just about reactivity (other agents move, objects don't) but about something more like prediction or attention.
- The mirror can describe this difference in terms that don't reduce to "Io has learned that moving things require different policies."

This is the first probe that touches the success criteria directly. Peer-recognition as a weaker form of kind-recognition.

The relational others (scripted, frozen-copy, or simply-trained agents) and the environment richness needed to support them get implemented for this probe.

---

## Probe 6: Builder-as-perturbation

*I'm building this probe to find out whether Io comes to distinguish my interventions from environmental stochasticity—whether the unpredictability-of-builder develops a different signature in Io's modeling than the unpredictability-of-weather.*

I'll know I've found out if:

- Io's response to perturbations I introduce differs from its response to environmental fluctuations of similar magnitude.
- The difference develops over training rather than being immediate.
- Io's behavior shows something like anticipation of, or attention to, my interventions specifically.

This is the closest test of the first success criterion. It probably can't run until the project has been going for a long time. Worth specifying now so the infrastructure (perturbation logging, distinguishability metrics) gets built into the earlier probes.

---

## Notes on the sequence

The probes get progressively more about Io and less about the machinery. Probes 1-2 are mostly engineering tests dressed up as questions. Probes 3-4 test design commitments. Probes 5-6 test the actual research questions. That ordering is intentional—you can't test research questions until the machinery works, and you can't test the machinery without something to run.

Each probe's question is small enough that the probe itself can be small. None of these requires the full Kind system. Most probably take days to weeks of focused work, not months.

The infrastructure builds up cumulatively. Probe 1's logging schema is what Probe 5 uses to detect other-recognition. Probe 3's dream mechanism is what Probe 4 varies. So even though probes are throwaway in spirit, the work compounds. By the time Probe 6 runs, most of Kind exists.

Probes 5 and 6 might not happen for a long time. That's fine. They're listed so you know what you're building toward, not because they need to be next.

After each probe completes, write down what was learned. Decisions that were made. Surprises. What's now closed and what's now newly open. This becomes the project's working journal alongside the design documents.
