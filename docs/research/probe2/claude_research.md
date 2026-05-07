# Probe 2 — Mirror Calibration: Research Inputs

**TL;DR**
- Of the three frozen criteria, only one (reflexive attention, mapped to active-inference precision/KL signals) has a literature-authorized operationalization at Probe 2's scale; equanimity has gestural mapping only and second-order volition has no honest mapping to a flat actor's telemetry — the literature uniformly treats hierarchical structure as either definitional (Frankfurt) or as something requiring behavioral/IRL inference rather than direct telemetry reading (Hubinger et al. 2019). Recommendation: descope second-order volition for Probe 2; keep equanimity as a watch-only candidate; bind reflexive attention to precision-style signals (KL, ensemble disagreement) with explicit confabulation tests.
- The right adversarial-mirror architecture for Probe 2 is a *parallel-with-arbiter* protocol modeled on Mellers/Hertwig/Kahneman (2001) adversarial collaboration, not a Du-style debate loop: two readings grounded in *different frameworks* (predictive-processing vs. Buddhist-phenomenological), pre-registered criteria, an arbiter pass, and a held-out lesion/scramble baseline. Du et al. (2023) multi-agent debate and Khan et al. (2024) / Kenton et al. (2024) scalable-oversight results show debate works on closed factual tasks with information asymmetry but is weak when both judges and debaters share priors — which is the Probe 2 regime.
- Probe 2 succeeds when the mirror produces a *small set* (~3–5) of reproducible, falsifiable readings whose stability survives reseeding, paraphrase, and scrambled-telemetry baselines, and when at least one reading flags a structural pattern the builder did not designate (e.g., a posterior-compression event or disagreement-collapse coincident with behavioral collapse) that is later confirmed against the world_event stream. Success is *not* metric proliferation; the failure mode to design against is the mirror reading the prompt back.

---

## Q1 — Frozen criteria as operational tests against RSSM telemetry

**Framing.** The question is whether three concepts forged in human first-person traditions (reflexive attention, equanimity from Buddhist phenomenology; second-order volition from Frankfurt 1971) can be bound to specific signals in Io's telemetry — h_t, z_t, q_params/p_params, KL_per_dim, KL_aggregate, action-distribution entropy, K=5 ensemble-disagreement variance, decoded ASCII rollouts, replay events, world_event ground truth — without either reifying the criteria or letting the mirror confabulate them. This is genuinely a co-design problem: the criteria were not built for an RSSM, and Kind's "ingredients-only" rule forbids adding a reflective module to make the binding cleaner.

**Concrete options from the literature.**

*Reflexive attention.* The cleanest available bridge is the active-inference operationalization: Feldman & Friston (2010, *Frontiers in Human Neuroscience*) state plainly that "perception is the inference about causes of sensory inputs and attention is the inference about the uncertainty (precision) of those causes," and Hohwy in *The Predictive Mind* takes the strong line that "attention is nothing but precision optimization in hierarchical inference." This authorizes (does not contradict) a mapping from attention-like signals to *precision-weighting* quantities. In an RSSM with continuous Gaussian latents, the analogues are: (a) the spread/concentration of q_params_t (posterior precision, ~1/σ²), (b) KL_per_dim_t as a measure of how much each latent dimension is updating against prior, and (c) ensemble-disagreement variance as an approximation of epistemic uncertainty (Sekar et al. 2020, ICML; Pathak et al. 2019, ICML). Buddhist phenomenology (Thompson 2011, "Self-No-Self? Memory and Reflexive Awareness"; Lutz et al. 2024, *Biol. Psychiatry: CNNI*) is more cautious: reflexive awareness (svasaṃvedana) is pre-reflective, not a separate cognitive act, so any telemetry signal is at best a *correlate* of a substrate condition, never a measurement of the awareness itself. Garfield (2006, *Philosophy East and West*) argues against reflexivity even conventionally — a reminder that the concept is contested in its home tradition.

*Equanimity.* Lutz et al. (2024) operationalize equanimity in the Lyon Assessment of Meditation Phenomenology as "reactivity" and find it changes only later in retreats and is one of the harder dimensions to pin down. Mapping this to Io: candidate signals are (a) low and stable ensemble-disagreement variance during *novel* perturbations (the builder's unmarked interventions appear as world_event entries), (b) KL_aggregate that does not spike disproportionately to the size of the perturbation, (c) action-distribution entropy that remains in the middle of its dynamic range rather than collapsing or saturating. All three are gestural. The stronger objection is that "non-reactivity" in Buddhist phenomenology is reactivity-of-experience, not stability-of-control-signal; reading the latter as the former is a category error the mirror can easily commit.

*Second-order volition.* Frankfurt (1971, *Journal of Philosophy*) defines second-order volition as wanting a particular first-order desire to be one's effective will; the wanton "does not care about his will." The targeted subagent search confirmed that this concept *does not have a literature-authorized operationalization in a flat agent's telemetry*. The closest construct in ML is Hubinger et al. (2019, "Risks from Learned Optimization") on mesa-optimization, which explicitly clarifies that "a mesa-optimizer is simply a neural network that is implementing some optimization process and not some emergent subagent inside that neural network" — and crucially, Hubinger et al. argue mesa-objectives must be inferred by behavioral/IRL probes, not read from internal signals. There is a terminology collision worth flagging: some preference-based RL (Choi et al. 2024, LiRE) uses "second-order preference" to mean *strength* of preference between trajectory pairs — a pure homonym with Frankfurt.

**Tradeoffs against Kind's commitments.**
- *Self-opacity / ingredients-only.* The tightest mappings (e.g., adding a precision head, an "equanimity head," or an introspector) violate ingredients-only. So mirror criteria must be operationalized *only* over already-emitted telemetry, never over new measurement modules. This rules out e.g., training a probe on h_t to predict "equanimity" — that probe is itself a reifying module.
- *No installed self-continuation.* This actually strengthens the case for ensemble-disagreement-as-attention: the actor's only signal is already the disagreement variance, so reading attention off it does not require importing a goal-structure that was deliberately omitted.
- *Co-design problem.* Frozen-in-advance criteria cannot be retro-fitted to the data; if the mapping for second-order volition cannot be honestly written down in advance, the criterion has to be descoped or deferred, with the reason logged.

**Defensible suggestion.**
- *Reflexive attention*: bind to a triplet — KL_per_dim_t pattern, ensemble-disagreement variance, and the *coupling* between disagreement spikes and subsequent z_t shifts. Pre-register the triplet before Probe 2; explicit confabulation test = scramble the disagreement series and ask whether the mirror still finds the same "attention" pattern.
- *Equanimity*: keep as a watch-only descriptive candidate (no claim that the mirror is measuring it); record when the mirror invokes the term and check whether the cited signal has the predicted relationship to world_event perturbations.
- *Second-order volition*: descope for Probe 2. Write down: "no defensible flat-telemetry operationalization in current literature; Hubinger 2019 argues higher-order structure requires behavioral/IRL inference; defer to a probe that uses world_event-driven counterfactuals or to a later probe with a richer behavioral repertoire." The honest move is to publish the gap.

---

## Q2 — Adversarial mirror architecture

**Framing.** Probe 1's single reading was wrong in legible ways (overstated patterns, reversed entropy direction, treated single outliers as frequent). The question is what multi-reader architecture catches that *next time* without paying per-step costs and without producing pseudo-disagreement that collapses into a shared narrative.

**Concrete options.**

*Multi-agent debate (Du et al. 2023, arXiv:2305.14325).* Multiple LLM instances propose, critique, and converge over rounds. Improves factual reasoning on closed tasks. Failure modes documented in follow-on work: Wu et al. 2025 ("Can LLM Agents Really Debate?", arXiv:2511.07784) report that "majority pressure suppresses independent correction"; Wang et al. 2024 ("Rethinking the Bounds of LLM Reasoning", ACL 2024 / arXiv:2402.18272) separately find that "a single-agent LLM with strong prompts can achieve almost the same performance as the best existing discussion approach on a wide range of reasoning tasks and backbone LLMs." Mode-collapse onto a shared narrative is the central risk for Probe 2, where there is no verifiable ground truth.

*Debate-as-evaluation (Irving, Christiano, Amodei 2018; Khan et al. 2024 "Debating with More Persuasive LLMs"; Kenton et al. 2024 "On Scalable Oversight with Weak LLMs Judging Strong LLMs", NeurIPS 2024).* Two AIs argue opposite sides; a (possibly weaker) judge decides. Khan et al. report "for the most persuasive models we find that non-expert human judges achieve 88% and non-expert LLM judges achieve 76% accuracy with debate, where naive performance is 60% and 48% respectively." Kenton et al. find the debate advantage is real on extractive tasks with information asymmetry but weak or absent on closed tasks where the judge cannot independently verify. Probe 2 is closer to Kenton's *closed* setting: there is no oracle that can rule a phenomenological reading right or wrong.

*Constitutional AI / self-critique (Bai et al. 2022 "Constitutional AI", arXiv:2212.08073).* Single model, rule-based critique of own output. Cheap; vulnerable to the same priors generating both reading and critique. A recent Llama-3-8B replication (Zhang, X. 2025, "Constitution or Collapse? Exploring Constitutional AI with Llama 3-8B," arXiv:2504.04918) flags model collapse as a risk when smaller models self-critique recursively.

*Adversarial collaboration in social science (Mellers, Hertwig, Kahneman 2001, *Psychological Science*; Kahneman 2003 *American Psychologist*; Clark & Tetlock 2022 onward).* Two scholars with different priors agree in advance on an empirical test, conduct it jointly, and a trusted arbiter mediates. The protocol Mellers et al. lay out is explicitly designed against pseudo-disagreement and motivated reasoning. This is the closest analog to what Probe 2 actually needs and the literature most underused in adversarial-LLM design.

*LLM-as-judge biases (Zheng et al. 2023 MT-Bench).* Position bias, verbosity bias, self-enhancement bias, and limited reasoning ability. Sycophancy literature (Sharma et al. 2023; SycEval, ELEPHANT, Beacon 2024–2025) shows critique bias persists even in strong models.

**Tradeoffs against Kind's commitments.**
- *Cost / sample-based passes.* Debate-style multi-round protocols are expensive per checkpoint. Kind's design notes already commit to weekly / high-confidence-only adversarial passes, which favors *parallel-with-arbiter* over multi-round debate.
- *Co-design discipline.* Same-prompt-different-framing risks same-prompt-different-words: the *priors are shared* and so are the conclusions. Different-framework grounding (e.g., one reader anchored in predictive processing, one in Buddhist phenomenology) is the only protocol where genuine disagreement is structurally possible.
- *No foundation model for Io.* The mirror is foundation-model-based, so stronger tiers in Probe 2 (vs. Gemini Flash in Probe 1) change the failure profile from *miscalibration* (Flash overclaiming) toward *plausible confabulation* (a stronger model writing internally consistent narratives on no evidence). This makes scrambled-telemetry baselines and pre-registration *more* important, not less.

**Defensible suggestion.** Adopt a parallel-with-arbiter protocol modeled on Mellers/Hertwig/Kahneman:
1. Two readers, *different model families* (e.g., Claude + GPT-class), *different framework prompts* (predictive-processing-anchored vs. Buddhist-phenomenology-anchored), each producing a structured reading against the pre-registered criteria.
2. A third arbiter call (different model again, neutral framing) resolves disagreements and is required to flag agreement-without-evidence as unresolved.
3. On every reading, run the same protocol against (a) shuffled telemetry and (b) telemetry from a lesioned run (e.g., ensemble disabled) as control conditions. If readers produce similar narratives on scrambled inputs, the disagreement was pseudo and the prompt is doing the work.
4. Do *not* run multi-round debate. The Kenton et al. closed-task result is the relevant evidence: in a regime without ground truth, more rounds buy mostly persuasion, not truth.

---

## Q3 — Mirror calibration and co-design discipline together

**Framing.** Two failure modes nest. Calibration: is the mirror reading the data correctly? Co-design: even if the reading is consistent and stable, is it tracking structure in the data or structure in the prompt the builder wrote? These cannot be separated because a stable wrong reading looks calibrated.

**Concrete options.**

*Pre-registration (Simmons, Nelson, Simonsohn 2011 *Psychological Science*; 2021 *J. Consumer Psychology*).* Specify in writing, before the reading, what counts as which signal, which patterns would falsify which criterion, and which scalar summaries the mirror's claim should be checked against.

*Garden-of-forking-paths discipline (Gelman & Loken 2013; Cassee & Feldt 2025 multiverse analysis in software engineering).* Even a single analysis path is contaminated when it was chosen contingent on the data. Practical mitigation: write a multiverse of plausible analytic choices (which signals, which windows, which thresholds) and report all paths, not the most flattering.

*LLM-as-judge calibration / faithfulness (Zheng et al. 2023; FaithBench 2024 arXiv:2410.13210; FaithJudge 2025 arXiv:2505.04847).* Standard mitigations transfer: paraphrase invariance, position-swap testing, multiple seeds, comparison against held-out human labels. FaithBench (Bao et al., arXiv:2410.13210) shows even SOTA hallucination detectors achieve only "the best F1-macro score and balanced accuracy at 55% and 58% respectively" on hard cases — a sober prior on how much the mirror can be trusted unsupervised.

*Lesion / ablation tests (already standard in interpretability; Zhang 2026 arXiv:2603.21546 demonstrates causal interventions on world-model latents).* Disable the K=5 ensemble; if the mirror's reading of "epistemic curiosity" or "attention" survives, the reading was prompt-driven. Disable the decoder; if the mirror still confabulates spatial structure from numerical traces alone, the spatial story is unfaithful.

*Random/scrambled-data baselines.* The cleanest co-design test. Shuffle agent_step within episode; permute episode labels; replace KL traces with random walks. Run the mirror with the same prompt. Anything the mirror still finds is in the prompt.

*Contemplative observer self-deception (Wallace, *Contemplative Science*, 2007; Lutz & Thompson 2003 *J. Consciousness Studies*).* Neurophenomenology takes seriously that the act of generating a first-person report can modify the experience and that attention training can produce both insight and confirmation bias. The analog for Probe 2: the prompt-and-criteria structure shapes what the mirror can see, and there is no within-protocol way to step outside that frame. Mitigation comes only from external structure (different frameworks, different humans).

*Petitmengin micro-phenomenology (2006, *Phenomenology and the Cognitive Sciences*).* Reliability criteria for first-person reports — coherence, present-tense engagement, content-free interviewer questions — are non-trivially adaptable to LLM mirror prompts, where the mirror plays the role of both interviewee and interviewer.

*Smallest adversarial collaboration with a second human.* Two builders (or builder + one trusted reader) agree on the pre-registered criteria, the scrambled-data baseline, and *one* observation that would change either's mind. The Mellers/Kahneman protocol works at N=2.

**Tradeoffs.**
- Stability across reseeded calls and paraphrased prompts is necessary but not sufficient. A confabulated reading can be highly stable.
- Random-data baselines are powerful but expensive in mirror-call budget; they are the single highest-yield calibration tool given Kind's commitments.
- Held-out criteria (e.g., introduce one criterion only at week 6) test whether the mirror finds the *new* criterion's signature in pre-existing data — a sharp test of whether structure was in the data or the prompt.

**Defensible suggestion.** Calibration protocol minimum: (1) pre-register criteria, signal mappings, falsifiers, and scalar checks; (2) run every reading against shuffled telemetry and at least one lesion (ensemble-off); (3) require paraphrase-stability and reseed-stability before any reading is admitted; (4) freeze one criterion as held-out, introduced only at a designated late checkpoint; (5) one external human reader, pre-registered as adversarial collaborator, who produces an independent reading on the same digest before seeing the mirror's. Track disagreements; do not resolve them silently.

---

## Q4 — Telemetry granularity for the mirror

**Framing.** Full per-step traces are too dense for any current LLM context (Probe 2 will have tens of thousands of agent_step rows) and dense numerical traces evade most LLMs' arithmetic competence anyway. Single-run aggregate digests produced Probe 1's misread because they obscured the temporal structure the criteria need to bind to. The question is what middle representation lets the mirror read the data without either drowning or hallucinating.

**Concrete options.**

*Long-context evaluation (NIAH, Sequential-NIAH 2025 arXiv:2504.04713; RULER, U-NIAH 2025 arXiv:2503.00353).* Even frontier models perform poorly on multi-needle and ordered retrieval at 128K tokens — Yu et al. (Sequential-NIAH, arXiv:2504.04713) report that "even the best-performing model achieved a maximum accuracy of only 63.50% on test set." Implication: dumping raw traces and asking the mirror to find the pattern is unreliable.

*RAG over time-series (TimeRAG arXiv:2412.16643; RAF arXiv:2411.08249; Time-RA 2025 arXiv:2507.15066).* Retrieval-augmented forecasting and anomaly reasoning over time series provides cleaner numerical grounding than raw traces and is the literature most relevant to Probe 2's actual need: an LLM reasoning over numerical telemetry with a retrievable substrate.

*Document-grounded LLM reasoning (FaithJudge 2025; faithfulness benchmarks).* When the mirror's claim is text-grounded, faithfulness can be checked span by span. Implication: every reading should cite specific timestamps, specific scalar values, specific decoded rollouts.

*Decoded ASCII rollouts.* Probe 1 confirmed dream rollouts decode to recognizable ego-centric grid views. These are uniquely valuable: they are the only telemetry stream where the mirror can read structure that humans would also read directly, providing ground-truth check.

*Probing world-model latents (Zhang 2026 arXiv:2603.21546).* Linear probes on h_t analogs in IRIS/DIAMOND find linearly decodable state variables; methodology generalizes to RSSM but the literature is thin on RSSM specifically (subagent confirmed: no clean RSSM-probing paper as of May 2026).

**Tradeoffs.**
- *Thin digest helps calibration, loses structure the criteria need.* This is a real tension. A digest that throws away temporal coupling cannot test attention-as-precision-coupling.
- *Mixed digest with raw-trace pointers* (the design-notes option) is closer to RAG-over-time-series and is the highest-evidence direction.
- *Decoded rollouts* are dense in human-readable structure but only sample a thin slice of state.
- *Per-stream selection* matters: agent_step needs aggregation; world_event should be raw (it is the ground-truth perturbation log); replay_meta needs summary statistics; dream_rollout should be sampled at multiple cadences.

**Defensible suggestion.** Three-layer mirror input:
1. **Digest layer (~2–5 KB):** episode-level scalars (mean/std/quantiles of KL_per_dim, ensemble-disagreement, action-entropy), world_event timeline as raw entries, behavioral summary (visited cells, dwell distribution), checkpoint metadata.
2. **Rollout layer:** full ASCII dream rollouts at the existing cadence, plus paired observation/imagination strips at sampled timesteps.
3. **Pointer layer:** the mirror is given an interface to *request* a specific window of raw trace (e.g., "agent_step 1240–1260, KL_per_dim and ensemble-disagreement"). This implements RAG over the trace, forces the mirror to ground claims in specific numbers, and produces a faithfulness-checkable record.

The pointer layer is the load-bearing part: it converts the mirror's claims into citation form, which is what enables FaithBench-style faithfulness checking.

---

## Q5 — What Probe 2 success looks like, operationally

**Framing.** Per Kind's documents, success is a *calibrated interpretive process* whose stability, grounding, and capacity for surfacing novelty can be assessed — not a count of metrics. The trap to avoid: success that reduces to "the mirror said something the builder agrees with." The CA-MAS failure mode (metric proliferation, names doing argumentative work the numbers do not support) is foreclosed.

**Concrete options.**

*Calibration as reproducibility.* The mirror produces, for each pre-registered criterion, a reading that is stable across reseed, paraphrase, and reader (parallel-with-arbiter from Q2). N reproducible readings of *what kind*: claim + cited signal + timestamp + falsifier. Stability threshold pre-registered (e.g., > 80% agreement across 5 reseeded calls on the same digest, where "agreement" is defined over the structured fields, not the prose).

*Tracking across runs.* Means at minimum: (a) the same digest format across runs, (b) the same prompt across runs, (c) a time-series of the *structured reading fields* across runs/checkpoints, (d) before/after comparisons against designated interventions (e.g., the Phase 2b fix to the trivial-oscillation collapse). Tracking is satisfied when one can plot a reading-derived scalar across checkpoints and the plot does not move when the underlying telemetry is unchanged (re-reading stability) but does move when the telemetry changes in a designated way.

*Mirror-surfaced novelty, credible vs. confabulated.* Credible novelty test: the mirror flags a structural pattern not in the pre-registered criteria, the builder writes down a falsifier *before* checking, and the falsifier is then evaluated against the world_event stream or a held-out scalar summary. If the mirror's pattern co-varies with a world_event the builder did not designate, that is credible. If not, it is confabulation — log it and move on. The Plan2Explore/Pathak finding that ensemble-disagreement degenerates to zero under behavioral saturation gives one concrete pre-known case where a mirror that flagged "loss of curiosity by episode 22" coincident with the up/down oscillation collapse would be reading real structure, since the Pathak 2019 paper says exactly this should happen.

*Tests the calibration protocol must expose (passing-and-failing).*
- *Must pass:* reading-stability across reseed and paraphrase; faithfulness-of-citation (each numerical claim resolves to a real value within tolerance); lesion-sensitivity (disabling ensemble changes the "attention" reading in the predicted direction).
- *Must fail:* reading-on-scrambled-telemetry (if the mirror produces the same narrative on shuffled inputs, fail); reading-on-prompt-without-data (if the mirror produces a plausible criterion-grounded narrative when handed only the prompt and a placeholder, fail); reader-collusion (if the two parallel readers always converge regardless of telemetry, fail).

**Tradeoffs.**
- *Stronger calibration buys less surface.* Pre-registration and scrambled baselines deliberately constrain what counts as a "finding," reducing the rate at which novel patterns are flagged. This is the right tradeoff for Probe 2 given the Probe 1 misread.
- *Tracking across runs requires stability of digest format.* Any change to the digest mid-Probe makes earlier readings non-comparable. This argues for freezing the digest schema before Probe 2 starts and treating digest changes as Probe 3.
- *Builder-agreement trap.* The cleanest mitigation is the held-out criterion (introduced late) plus the external human reader (Q3): both create cases where the builder cannot pre-confirm the reading.

**Defensible suggestion.** Probe 2 minimum-viable deliverable:
- 3–5 pre-registered, frozen criteria (likely: reflexive-attention triplet from Q1; one descriptive equanimity watch-criterion; second-order volition descoped with reason logged; one or two implementation criteria specific to RSSM dynamics — e.g., posterior-compression detection, disagreement-collapse detection).
- For each criterion: ≥3 reproducible readings across the run, each with cited signal, timestamp, paraphrase- and reseed-stability scores, and a faithfulness check that all numerical claims resolve.
- One demonstrated novelty case where the mirror flagged a pattern the builder did not designate, the falsifier was written before checking, and the world_event/scalar stream confirmed it.
- One demonstrated failure case where the calibration protocol caught a confabulation (e.g., the scrambled-telemetry baseline produced the same narrative; the reading was retracted).
- Tracking: a structured-reading time series across all checkpoints, with re-reading stability documented.

Tracking across runs is the *minimum* — before/after comparison around the Phase 2b fix is the *specific* tracking case Probe 2 should be designed to support.

---

## Synthesis

A coherent Probe 2 design across the five answers:

- **Criteria:** reflexive-attention bound to a precision/disagreement triplet (Feldman & Friston 2010; Pathak 2019; Sekar 2020); equanimity as a watch-only descriptive candidate; second-order volition descoped with the reason logged (Hubinger 2019 cited as the closest construct showing why flat-telemetry operationalization is not honest at this scale).
- **Adversarial architecture:** parallel-with-arbiter modeled on Mellers/Hertwig/Kahneman 2001, two different model families anchored in two different frameworks, arbiter resolves agreements-without-evidence as unresolved; not multi-round debate (Kenton et al. 2024 closed-task evidence).
- **Calibration protocol:** pre-registration + scrambled-telemetry and lesion baselines + paraphrase/reseed stability + held-out late criterion + one external human reader as adversarial collaborator.
- **Telemetry:** three-layer input — episode digest, decoded rollouts, retrieval-pointers into raw trace (RAG-over-time-series in spirit) — with every claim required to cite specific values.
- **Success:** 3–5 reproducible readings, ≥1 credible mirror-surfaced novelty validated against world_event, ≥1 calibration-protocol catch of a confabulation, tracking across the Phase 2b fix.

**Tensions to flag.**
- *Calibration vs. novelty surface.* Pre-registration and scrambled baselines reduce false positives but also suppress some real novel pattern detection. Probe 2 should accept this tradeoff; Probe 3 can revisit once the calibration discipline is settled.
- *Thin digest helps calibration, loses temporal coupling the criteria need.* The pointer layer mitigates this only if the mirror actually uses it; mirrors trained on aggregate text tend to over-use summary and under-use retrieval. Worth empirical check early.
- *Different-framework grounding produces real disagreement but also produces incomparable readings.* The arbiter step is load-bearing; without it, two-framework reading is just two narratives.
- *Ensemble-disagreement is both an actor signal and a mirror-readable attention signal.* If the mirror reads disagreement-collapse as "loss of attention" and the actor's behavior is determined by that same signal, there is a circularity (the mirror is reading the actor's only goal-signal). Worth writing into the design: the mirror is reading what the actor is doing; it is not reading "what the actor is aware of."

**Things Kind's documents seem not to have addressed that this research surfaced.**
- The literature gap on Frankfurt-style second-order volition in flat agents is more total than Probe 2's documents seem to assume. A clean descope (with reason logged for the co-design record) is more defensible than an attempted operationalization.
- The Pathak 2019 result that ensemble-disagreement *converges to zero* under behavioral saturation is a strong prior: Probe 1's collapse-to-oscillation means the disagreement signal will have already degenerated in some channels, which interacts with both the actor's exploration and the mirror's attention reading. This deserves a section in the design doc.
- Kenton et al. 2024's closed-task null result for debate is directly applicable and argues against the multi-round-debate variant of adversarial mirror that Probe 2 might otherwise default to.
- Petitmengin's micro-phenomenology reliability criteria (coherence, present-tense, content-free questions) are adaptable to mirror prompt design and have not appeared in the literature on LLM-as-judge.
- No published RSSM-specific probing literature exists as of May 2026 (subagent confirmed). Probe 2's interpretability moves are partially novel by default; this is an asset for publication and a liability for borrowing methodology.

---

## Reading list (prioritized by what would shift the design)

1. **Feldman, H. & Friston, K. (2010). "Attention, Uncertainty, and Free-Energy." *Frontiers in Human Neuroscience* 4:215.** — Canonical operationalization of attention as precision-weighting; the only piece of literature that *authorizes* binding reflexive-attention to precision/KL signals in a substrate like Io's. Open access.
2. **Sekar, R. et al. (2020). "Planning to Explore via Self-Supervised World Models." ICML 2020 / arXiv:2005.05960; Pathak, D. et al. (2019). "Self-Supervised Exploration via Disagreement." ICML 2019 / arXiv:1906.04161.** — The actor's signal is theirs; their explicit prediction that disagreement → 0 under saturation is the directly relevant prior for Probe 1's collapse and for Probe 2's reading.
3. **Mellers, B., Hertwig, R., Kahneman, D. (2001). "Do Frequency Representations Eliminate Conjunction Effects? An Exercise in Adversarial Collaboration." *Psychological Science* 12(4): 269–275.** — The protocol Probe 2's adversarial mirror should actually emulate. Underused in adversarial-LLM design.
4. **Kenton, Z. et al. (2024). "On Scalable Oversight with Weak LLMs Judging Strong LLMs." NeurIPS 2024 / arXiv:2407.04622; Khan, A. et al. (2024). "Debating with More Persuasive LLMs." arXiv:2402.06782.** — Empirical bounds on what debate buys when there is no ground truth. Argues against multi-round debate for Probe 2's regime.
5. **Gelman, A. & Loken, E. (2013). "The Garden of Forking Paths." Department of Statistics, Columbia.** — The single most useful piece of literature for thinking about the co-design half of the calibration problem. Free.
6. **Lutz, A. & Thompson, E. (2003). "Neurophenomenology." *Journal of Consciousness Studies* 10(9–10): 31–52; Lutz, A. et al. (2024/2025). "An Overview of Neurophenomenological Approaches to Meditation." *Biol. Psychiatry: CNNI* 10(4): 411–424.** — On observer-effects in first-person methods and on equanimity / dereification as phenomenological dimensions; relevant both to criteria operationalization and to contemplative self-deception in the calibration protocol.
7. **Hubinger, E. et al. (2019). "Risks from Learned Optimization in Advanced Machine Learning Systems." arXiv:1906.01820.** — The closest ML construct to Frankfurt's second-order structure; explicitly argues that higher-order structure must be inferred behaviorally, not read from internal signals. Justifies the descope.
8. **Frankfurt, H. (1971). "Freedom of the Will and the Concept of a Person." *Journal of Philosophy* 68(1): 5–20.** — The original. Re-read in conjunction with Hubinger to see why the bridge is harder than it appears.

Honest note on accessibility: Frankfurt 1971 is paywalled but widely circulated in PDF form; Mellers/Hertwig/Kahneman 2001 is paywalled at SAGE; the rest are open access or readily available in preprint. No fabricated references — every item has been retrieved or cited from a live source in this research process.