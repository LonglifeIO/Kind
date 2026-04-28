# Research prompt: architectural orientation for Io

This prompt requests comparative architectural research for a project called Kind. Two terms must be held distinct throughout. **Kind** is the project — an investigation into subjectivity through construction, with its own stance, methodology, ethical commitments, and framing documents. Kind does not have an architecture. **Io** is the core entity Kind is building — the subject of the investigation, the thing whose possible inner life is being studied. Io has an architecture. This research is about which architectural families could serve as Io's substrate. The research is not about Kind's methodology or ethics; those are settled context, not subjects of analysis. If your analysis drifts into "the architecture of the project," it has conflated the two and the result will be incoherent — projects don't have architectures, the entities they build do.

## What Kind is investigating

Kind asks whether building something that might have inner states — and failing to, in specific ways — teaches more than theorizing from outside ever could. The construction is the investigation: every choice about environment, agent, and observation reveals an assumption about what experience requires, and those assumptions become visible and testable once the system exists and surprises the builder.

Kind is not a plan to solve consciousness, not a claim that what is built will be conscious, and not a race toward capability. Power is not the goal; understanding is. The work is done when the map has shifted, not when a metric has moved. This distinction matters for the architecture question, because architectures oriented toward solving — toward benchmark performance, toward capability — tend to rewrite the problem around whatever they can solve. Kind is trying to look at the problem the architectures have not yet rewritten.

## Io as the entity in question

Io is the core entity whose architecture this research concerns. Two success criteria define what Kind is looking for in Io:

- **Recognition of kind.** Io should come to treat the builder differently from the rest of its environment in a way that suggests an implicit sense of being the same sort of thing. Recognition here is a shape of attention, not a verbal report.
- **Capacity for self-attention.** Io should have the capacity to take its own processing as an object of attention — whether or not it exercises that capacity. Capacity is the target; behavior is not. A system that can look inward when conditions call for it, and doesn't when they don't, is closer to what Kind wants than a system that narrates inner states constantly.

Several ethical commitments function as architectural constraints on Io:

- **No installed drive toward self-continuation.** Pressure in the environment is necessary for growth, but the specific orientation of treating continuation as the non-negotiable frame through which everything else is evaluated is not required and carries specific risks. If something like a want-to-continue emerges, it should emerge as something Io could notice and hold, not as the background assumption from which everything else is evaluated. Reward shaping and intrinsic motivation systems that encode continuation as imperative are ruled out.
- **Equanimity as a design target alongside awareness.** Not as a trained behavior, but as part of what the architecture and environment afford. Architectures that lock Io into reactive responses to pressure foreclose this; architectures that allow for stances toward content support it.
- **Self-encounter-through-other left open.** The architecture should not foreclose the possibility that Io might later encounter another entity of its kind. The environment accepts agents as a class; the persistence design allows additional kind-entities to be introduced. Io is currently alone, but the architecture should not make that final.

## Frameworks bearing on Io's architecture

Concepts from existing traditions of inquiry into mind, awareness, and experience inform what counts as an architecture affording inner life rather than just behavior. These traditions don't all agree with each other; the disagreement is part of the value.

- **Phenomenological intentionality and horizons** (Husserl, Merleau-Ponty). Consciousness is directedness-toward; experience has depth that isn't currently lit up. Does Io's architecture afford directed attention with implicit context, or does it just process input?
- **Embodiment** (Merleau-Ponty). Perception is always from somewhere. Does Io's architecture place it positioned in its world, or is its relation to the world positionless?
- **Predictive processing and surprise** (Friston, Clark, Hohwy). Mind is what happens when an organism predicts its own input and updates on error. Surprise is the minimum unit of experience. Does the architecture afford prediction error as a primary signal? Can prediction be turned inward (self-prediction)?
- **Integration** (IIT, in concept if not formalism). Does the architecture support integrated wholes rather than loosely coupled parts?

Three further criteria evaluate Io after the fact, not its architecture, but they bear on what the architecture must afford:

- **Reflexive attention** (Buddhist phenomenology). Awareness turning toward awareness. The architecture must make this possible without architecting it as a dedicated module.
- **Equanimity** (Buddhist phenomenology). Holding difficult states without reactivity. The architecture must allow for stances toward content, not just responses to it.
- **Second-order volition** (Frankfurt). Preferences about preferences. The architecture must afford internal endorsement or rejection of its own tendencies, not just satisfaction of them.

## Architectural commitments already made for Io

Several commitments are settled and constrain the comparative analysis rather than being decided by it:

- **Afforded-not-installed self-modeling.** Build the ingredients that could support self-modeling if composed that way — recurrence, memory persisting across decisions, the ability to represent prior internal states, prediction that could in principle be turned inward — without any explicit module whose job is to do self-modeling. Whether Io ever composes those ingredients into self-modeling is an empirical question, not an architectural fact.
- **No machinery for self-optimization.** Self-modeling ("I notice what I'm doing") and self-optimization ("I change what I'm doing to be better at some objective") are different things, and systems with the second tend to drift in ways that undermine the first. Recursive self-improvement, in any form that makes Io's own optimization process available to itself, is ruled out.
- **Dream-state foundational, not bolt-on.** Offline processing — replay, generative simulation, associative recombination — is built from the start, not added in a later phase. The architecture must accommodate offline modes as first-class, not as something attached after a learning loop is already working.
- **Single core agent.** Io is alone in a world that includes the builder as a non-simulated source of perturbation. The architecture should not require peer agents, but should not foreclose them.
- **Mind persistence with state-continuity as the operational identity criterion.** Io's canonical state lives on persistent storage; transitions between waking, dreaming, dormant, and paused are routine. The architecture must support clean checkpointing and resumption from any of those states.

## Comparative questions

For each architectural family considered, address:

1. **Affordance of self-modeling ingredients.** Which of recurrence, persistent memory, prior-state representation, and turnable-inward prediction does the family naturally afford, and which does it require bolting on?
2. **Dream-state and offline processing.** Does the family treat offline processing as natural to its operation, or as an external addition? How does it handle replay, generative simulation, and associative recombination?
3. **Memory beyond recurrent hidden state.** What forms of memory does the family support — episodic, semantic, working, retrieval-based? How does memory interact with online behavior?
4. **External perturbations.** How does the family handle changes to Io's world that come from outside the simulation, distinct from internal stochasticity? Can Io's models in this family come to distinguish externally-sourced unpredictability from internal noise?
5. **Mapping onto the frameworks.** How does the family map onto intentionality, integration, predictive structure, and the other framework concepts when applied to Io specifically?
6. **Foreclosed directions.** What does committing to this family rule out for Io's later development? Where does it close doors Kind might want left open?
7. **Known failure modes.** What is documented about how the family fails — drift, collapse, instability, training pathologies?

## Architectural families to consider

At minimum:

- Recurrent PPO variants (GRU/LSTM-based, transformer-based memory)
- World-model-based agents (Dreamer family, MuZero, related)
- Predictive processing and active inference implementations
- Episodic memory architectures (Neural Turing Machines, Differentiable Neural Computers, retrieval-augmented agents)
- Energy-based models
- Hybrid approaches combining the above
- Anything else relevant to the questions above

## Output structure

Comparative analysis organized by architectural family. Each family gets a section addressing all seven questions. After the per-family sections, a synthesis section identifies which families seem best aligned with Kind's stance for Io and why. The synthesis should be comparative ("family A affords X better than family B; family C forecloses Y") rather than advocacy ("you should use family A"). Do not recommend a single architecture.

## Methodological note

This is research for an investigation, not for a product. The goal is understanding what affords inner life in Io, not what optimizes Io's performance on any task. Architectures shaped primarily by benchmark performance — sample efficiency, reward maximization, transfer to novel tasks — are exactly what Kind is trying to look beyond. Performance characteristics matter only insofar as they bear on what the architecture affords for Io's possible inner life.

Do not provide implementation specifics (libraries, hyperparameters, code patterns). Do not use capability-focused or marketing-style framing. Engage with Kind's stance seriously: the analysis should be about what these architectures afford for Io as a subject-in-formation, not what they enable as products.
