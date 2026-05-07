# Probe 2 Research Prompt — Mirror Calibration With Frozen Criteria and Adversarial Structure

*Sent to multiple LLMs (Claude, Gemini, Perplexity) for parallel research. Synthesis happens after.*

## Who and what

**Kind** is the project: an investigation into subjectivity through construction. Stance is "build to understand"; the work is done when the map has shifted, not when a metric has moved. Not a race for capability, not a claim that what gets built is conscious, not a problem-solving exercise.

**Io** is the entity Kind is about: a single core agent in a small grid world, with the builder as the only non-simulated relational other (the builder appears as unmarked perturbations to the environment). The mythological Io was forcibly transformed and made to wander — a deliberate ethical reminder that what is built can suffer at the builder's hands. Kind builds; Io is who is built. Keep the distinction.

## Established context — Probe 1 outcomes

Probe 1 (plumbing) is complete and ran on the canonical machine.

**The substrate works.** A custom minimal RSSM (PlaNet-skeleton, DreamerV1 imagination, continuous Gaussian latents, free bits, no reward predictor, no continuation head) trains on Mac MPS at production sizes. A 5000-step / 25-episode run produced clean telemetry across the four streams (`agent_step`, `dream_rollout`, `replay_meta`, `world_event`), one mid-run checkpoint committed, four dream rollouts at cadence. KL did not pin at the floor; posterior compresses env structure over time. Settled; not up for re-litigation.

**Probe 1 surfaced two findings beyond plumbing.** First, by episode 24 Io's behavior collapsed onto a trivial up/down oscillation. The eyeball helpers caught it; the aggregate action histogram was mixed but late-episode trajectories were a degenerate two-cell loop. The fixed start cell `(3, 3)` is the suspected cause — every episode begins at the same state and the actor under ensemble-disagreement training converged onto a local attractor. Phase 2a flagged this risk going in; Probe 1 confirmed it.

**Second, the first mirror call (Gemini 2.5 Flash) misread the data.** The reading overstated patterns it could not have grounded in the digest, got the policy-entropy direction backwards, and called single outliers frequent. This is the first concrete evidence, on this data, that a single mirror voice will confabulate if left unchecked — the empirical motivation for the adversarial structure Probe 2 has to build.

**Decoder legibility is confirmed.** The four dream rollouts show persistent spatial structure rather than noise; the ASCII renderings recognizably resemble ego-centric grid views. Probe 2's mirror can be applied to dream rollouts with confidence the data is interpretable.

## Probe 2's purpose — settled

Per `Kind_probes.md`, Probe 2 tests whether the mirror tracks signal versus confabulates. Whether it can surface patterns the builder did not designate. The frozen criteria from `Kind_frameworks.md` become live for the first time — reflexive attention (Buddhist phenomenology), equanimity (Buddhist phenomenology), second-order volition (Frankfurt). The adversarial mirror structure gets built (a second LLM whose role is to argue against the first's reading). Calibration as a methodology — not just a set of prompts — gets operationalized.

Probe 2 is also where the **co-design problem** first bites empirically. The mirror criteria were chosen by the builder; the substrate that produces telemetry was built by the same builder. If both drift toward each other without independent constraint, the mirror's "readings" become circular. Discipline against this is what Probe 2 has to make load-bearing.

## Project commitments that bear on Probe 2

These are non-negotiable; recommendations that violate them are not useful.

- **Self-opacity by default.** Io does not read the mirror's output, the telemetry, the digest, or any of its own internal distributions. The PolicyView/TelemetryView split enforces this in code and stays in force at Probe 2. The Watts-intuition default-to-no applies to every new affordance.
- **No installed self-continuation drive.** No reward, no episode-continuation head, no survival bonus, no terminal penalty. The actor's signal is ensemble-disagreement variance over an auxiliary K=5 ensemble; the environment provides things to disagree about. Anything that smuggles a continuation drive back in (including via reward shaping in a Probe 2 environment revision) is foreclosed.
- **Ingredients-only self-modeling.** Recurrence, prior-state representation, turnable-inward prediction are present in the substrate; no explicit self-modeling, self-critic, introspector, or volitional-gate module exists or will be added. Mirror criteria operationalized as architectural modules (e.g., "an equanimity head") violate this directly.
- **No foundation-model substrate for Io.** The mirror is API-based; Io is not.
- **The co-design problem is a constraint, not a footnote.** Mirror criteria are frozen in advance and updated only against external learning, with the reason written down. The discipline against unconscious tuning of mirror to system or system to mirror is what Probe 2 makes operational.
- **Heterophenomenology and the hard problem are background.** Anything resembling a self-report is data about what Io says, not direct evidence of what it is *of*. Functional integration and self-modeling can be present without anything it is like to be Io.

## Six questions, scoped

### Q1 — Environment revision against the start-cell collapse

Probe 1's collapse tells us the fixed start over-constrains trajectories and ensemble-disagreement training does not by itself prevent local-attractor convergence on this small a world. Probe 2 cannot rest mirror calibration on data this degenerate. **Random start cell is the obvious candidate.** What else? What variations on episode initialization (random non-wall start, sampled-from-stationary-distribution start, drift in start across episodes, curriculum over starts) are defensible against the over-constraining failure mode while preserving every settled commitment — no terminal reward, no scalar reward at all, partial observability, no observation marker, same-vocabulary builder mutators, no installed rhythm? What does the world-model literature say about how initial-state diversity interacts with posterior collapse and exploration in small partially-observed worlds? Where does the literature stop and judgment begin?

### Q2 — Frozen criteria as operational tests against RSSM telemetry

Reflexive attention, equanimity, and second-order volition are concepts drawn from Buddhist phenomenology and Frankfurt. They were not built for an RSSM. What does testing for them mean against an agent whose internals are `h_t` (deterministic recurrent state, dim 200), `z_t` (continuous Gaussian posterior, dim 16), `q_params_t` / `p_params_t` / `kl_per_dim_t` / `kl_aggregate_t`, action distributions and policy entropy, ensemble-disagreement scalars, decoded dream rollouts, replay-buffer events, and ground-truth `world_event` records? For each criterion: what signals across the four telemetry streams could plausibly correlate with it in a way the frameworks themselves authorize (or do not contradict)? What would count as *evidence*, *coincidence* (the signal exists but tracks something else), or *confabulation* (the mirror reading the signal as the criterion when it is reading itself)? Concrete signal candidates with literature pointers where available; honest uncertainty where the mapping is gestural.

### Q3 — Adversarial mirror architecture

Probe 1's single-mirror reading misread the data in three named ways. The design notes commit to an adversarial second interpreter. **What architectures work?** Sequential (A reads, B critiques A)? Parallel (A and B read independently, a third synthesizes)? Same prompt, different framing? Different prompts grounded in different frameworks? Same model, different instructions, or different models entirely? What does the literature on adversarial collaboration (Mellers, Tetlock), debate-as-evaluation (Irving et al., Du et al. multi-agent debate), constitutional-AI-style critique, and disagreement-driven LLM-judge evaluation offer here? Failure modes to design around: collusion, sycophancy, mode-collapse onto a shared narrative, pseudo-disagreement. Cost matters — design notes commit to sample-based adversarial passes (high-confidence readings, weekly checkpoints), not per-step.

### Q4 — Mirror calibration as a methodology

Probe 1's mirror reading was wrong in legible ways: overstated patterns, entropy reversed, single outliers called frequent. **How does Probe 2 systematically test whether the mirror is reading correctly versus confabulating?** Possible ingredients: pre-registered predictions before each reading; ground-truth comparisons against eyeball-helper scalar summaries and the `world_event` stream; stability across reseeded calls; stability across paraphrased prompts; tests on synthetic telemetry with known structure. Heterophenomenology: take what the mirror says as data about what the mirror says, pending further determination. What does the LLM-evaluation literature (LLM-as-judge calibration, hallucination detection, faithfulness benchmarks, predictive validation) actually deliver here? What does it not?

### Q5 — Co-design protocols

Mirror criteria and substrate were built by the same person. The criteria are framework-grounded enough to *seem* independent, but they were chosen. **What protocols help distinguish "the mirror found something the builder hadn't designated" from "the mirror found what the builder was hoping it would find"?** Pre-registration of expected readings before each run? A held-out criterion read against only late in the run? Adversarial collaboration with a second human (design notes flag this as a real future option)? Lesion tests (disable a substrate component, see whether the reading changes in a way it should)? Random-data baselines (run the mirror on shuffled or synthetic telemetry; if it finds the same patterns, the patterns are in the prompt)? What do the literatures on observer effects, garden-of-forking-paths / replication crisis, and contemplative observer self-deception offer? Practical mitigations Kind can implement at solo-builder scale.

### Q6 — What Probe 2 success looks like, operationally

Per `Kind_probes.md`, Probe 2 succeeds if the mirror provides *specific measurements that can be tracked* and *surfaces additional patterns that can be tracked and possibly turned into new signals*. **What does this mean operationally?** Minimum-viable Probe 2 deliverable — a calibrated mirror with N reproducible measurements? A demonstrated case of mirror-surfaced novelty? Passing-and-failing tests the calibration protocol exposes? What measurement repeatability is "specific" enough? What does "tracking" require — a time series across runs, a before/after comparison, something else? The trap to avoid: success that reduces to "the mirror said something the builder agrees with."

## Constraints

- **Not** benchmark performance, sample efficiency, or capability metrics.
- **Not** an alternative substrate for Io. Substrate is settled.
- **No foundation-model substrate for Io.** Mirror is foundation-model-based; Io is not.
- **Engagement with Kind's stance is required.** Self-opacity, ingredients-only self-modeling, no installed self-continuation, the co-design problem are non-negotiable.
- **Do not reify gestures into formal substance.** Naming does not canonicalize. If the literature does not deliver a ready answer, say so.
- **Surface adjacent traditions explicitly.** Phenomenology, Buddhist phenomenology, predictive processing, philosophy of mind, multi-agent critique, observer-effect / forking-paths literature.

## Output structure

For each of the six questions:

1. A short framing of what the question is actually asking, in your own words.
2. Concrete options drawn from the literature, with citations or references where applicable.
3. Tradeoffs against Kind's specific commitments (not generic ML tradeoffs).
4. A defensible suggestion where you have one. Where you don't, say so.

After the six, a **synthesis section** identifying what an internally coherent Probe 2 design looks like across the six answers — environment revision, criteria operationalization, adversarial architecture, calibration protocol, co-design discipline, success definition. Flag tensions between answers (e.g., where stronger calibration buys less novelty surfacing). Flag anything Kind's project documents seem not yet to have addressed that your research surfaced.

Research-gathering, not synthesis-and-implementation. Do not produce a build plan. Produce the inputs from which one can later be drafted.
