# Probe 4 Design Inputs — Perplexity

Io can in principle learn that builder-keyed perturbations are a distinct causal source, but the current Probe 4 instrument needs stricter controls on rate, phase structure, and decoder calibration to make that claim defensible.

## Q1 — Contingency-loss detection and yoked-control critiques

**Recommendation: Use relatively long, alternated contingent/yoked blocks with explicit washout and a counterbalanced order across the biography, and treat Church-style yoked confounds as attenuated but not eliminated by the within-subject design.**

### Minimal conditions for contingency-loss detection

Human infant work suggests three minimal ingredients for detecting loss of contingency: (1) prior exposure to reliable contingency; (2) a sufficiently abrupt and sustained disruption; (3) a channel in which the agent's own action-outcome mapping is salient. In Murray & Trevarthen's double-video paradigm, infants look more at live mothers than replayed, out-of-sync mothers, even when visual content and marginal statistics are closely matched, implying sensitivity to the live contingency rather than pure novelty or fatigue. (UNVERIFIED) Tronick's still-face paradigm similarly shows rapid escalation of bids to re-engage, followed by distress and withdrawal when previously responsive caregivers become unresponsive, again under matched content but broken contingency. (UNVERIFIED)

For Io, these ingredients translate to:

- A perturbation channel whose occurrence is tightly keyed to Io's state or behavior during contingent phases.
- Yoked phases that preserve content and marginal event statistics but break that action-state → event mapping.
- A learning signal that is sensitive to deviations from learned conditional structure (ensemble disagreement / prediction error).

### Dissociating contingency-detection from novelty/arousal

Infant replay paradigms control novelty by equating sensory content and comparing live vs replay within subjects; still-face controls arousal by measuring changes relative to baseline contingent interaction rather than absolute affect. (UNVERIFIED) The analog for Io is:

- Use perturbations whose sensory content and marginal rate are identical across contingent and yoked phases.
- Diagnose contingency-detection via conditional differences: discrepancies between p(event | Io-state) learned during contingent blocks and actual event placement during yoked blocks, rather than absolute event surprises.

### Church's yoked-control critique under Probe 4's design

Church's argument (in animal learning) is that yoked controls can be misleading because the experimental subject's behavior changes reinforcement statistics, while yoked subjects receive reinforcement uncorrelated with their actions; differences can then reflect unintended motivational or schedule effects rather than the factor of interest. [constructed]

In Probe 4's within-subject alternating-phase design:

**Dissolved confounds:**
- Subject identity: Io is its own control, so trait-level differences are eliminated.
- Marginal statistics: by construction, event content and rates are matched across phases.

**Surviving confounds:**
- Reinforcement-history asymmetry: Io's internal parameters (world model, actor) are shaped by contingent phases; yoked phases then play sequences into a different internal landscape than the one in which they were generated. This can create phase-order effects unrelated to contingency-recognition.
- State carryover: h and z at the start of a yoked block depend on prior contingent experience, so replayed perturbations may land on systematically different latent states than the original, even if their time indices match.
- Schedule artifacts: if contingent blocks systematically occur earlier or later in the day relative to dream-state consolidation, off-policy replay may be differentially consolidated.

Given that Io has no external scalar reward, "reinforcement uncorrelated with behavior" becomes "intrinsic disagreement signal uncorrelated with behavior." Builder-keyed events that preferentially land in high-disagreement regions during contingent phases but fall in arbitrary regions during yoked phases could produce differences that are more about novelty sampling than "builder-kind" modeling. That is a real confound, not dissolved by within-subject design.

### Order, carryover, washout, and block structure

Disagreement: A simple ABAB day-level alternation is too fragile; order effects and dream-state consolidation will entangle contingency-recognition with phase history.

Concrete recommendation:

- **Block length:** Use relatively long blocks (e.g., 3–5 days of waking time per phase), so Io can actually learn conditional structure within a contingent block and exhibit violations in the subsequent yoked block. Short alternations will mostly test exposure, not modeling.
- **Order counterbalancing:**
  - Across different long runs or sub-biographies, counterbalance starting phase: some Io instances begin with contingent blocks, others with yoked.
  - Within a single biography, use an ABBA or BAAB macro-structure (e.g., Contingent₁–Yoked₁–Yoked₂–Contingent₂). This introduces both contingent→yoked and yoked→contingent transitions.
- **Washout:** Insert explicit washout micro-blocks (e.g., 1–2 days of zero perturbations) between phase transitions. During washout: no builder events; monitor whether h, z, and disagreement statistics stabilize, to avoid attributing transient adaptation to contingency effects.

This block structure optimizes the tradeoff between statistical power (enough events per phase to estimate conditional structure) and interpretability (reducing order and carryover confounds), at the cost of a more complex biography and slower probing of the main question.

## Q2 — Signatures of "models it as a different kind"

**Recommendation: Pre-register a multi-signature battery, with the primary signal being asymmetry in prediction error/disagreement between contingent and yoked events that grows over training, and treat others as converging but weaker evidence. All signatures below are explicitly labeled.**

### (a) Per-event prediction error / ensemble disagreement

Signature [constructed]:
- Compute per-event prediction error and ensemble disagreement at steps with perturbations.
- Compare contingent vs yoked within matched event types and Io latent state ranges.
- Expectation: once Io has learned p(event | state) in contingent phases, yoked events (violating that conditional structure) should yield higher disagreement than contingent ones, and the gap should widen over training.

Design constraint: This only works if the contingent policy keys perturbations to Io states within its representational reach — i.e., states that meaningfully modulate h and z under the RSSM, not latent dimensions Io never uses.

Confounds and discriminators:
- Frequency adaptation / habituation: If contingent events are more frequent than yoked, lower disagreement could reflect habituation. Control by matching marginal rates and by inspecting disagreement in non-event steps to see whether global novelty reduction explains the effect.
- Arousal / generic surprise: High disagreement at phase boundaries could reflect schedule changes rather than contingency structure. Separate event-specific disagreement from background disagreement during non-perturbation steps, and examine whether the contingent–yoked gap persists away from transitions.
- Rate-matching: If Io only tracks marginal event frequency, disagreement should depend on overall rate, not conditional structure; then contingent vs yoked events matched in rate and content should not separate.

### (b) Conditional vs marginal event-rate learning

Signature [constructed]:
- Estimate Io's implicit model of p(event | state) by training an external classifier on Io's internal predictions: given h, z, and Io's own predicted event probability (if exposed), infer whether Io's world model believes an event is forthcoming.
- Compare: performance on predicting contingent events from Io's state; performance on predicting yoked events (replayed sequences) from Io's current state.

If Io has learned conditional structure, predictive performance should be high in contingent phases (event placements predictable from state) but near chance on yoked phases, even with identical marginal rates. If it has only learned marginal rates, performance should be similar across phases.

Confound: Rate-matching account (Io tracks overall frequency, not conditional structure).

Measurement separating signal: Show that a model trained only on time indices and marginal statistics, without h and z, cannot achieve the same separation as one using h and z; this argues the signal is truly state-conditional.

### (c) h-trajectory signatures around events

Signature [constructed]:
- Analyze trajectories of h in a window around perturbation events (e.g., ±N steps).
- Look for: systematic "anticipatory" patterns in h before contingent events (Io expecting something builder-like); distinct "repair" or "reset" dynamics after yoked events (Io treating them as violations needing re-integration).

This is analogous to infant gaze and affect trajectories around live vs replay/still-face episodes, where anticipatory bids and repair attempts distinguish contingent interaction from mere stimulation. (UNVERIFIED)

Confounds:
- Frequency adaptation: h may drift simply because events occur often in certain regions. Control by comparing trajectories around non-event steps that match the same latent regions.
- Arousal: Latent dynamics may respond to any event, independent of contingency. Disentangle by comparing h trajectories for non-builder stochastic events in the world model.

### (d) Dream-state over-representation of contingent events

Signature [constructed]:
- Use the overnight dream-state [canonical] monitor to analyze generative rollouts.
- Measure whether sequences containing contingent-like perturbation patterns are over-represented relative to yoked-like patterns, controlling for content and marginal statistics.
- Expected: if Io has modeled builder-kind contingencies, dream recombination should preferentially sample the conditional structure learned in contingent phases (akin to hippocampal replay of behaviorally relevant sequences in animals). [constructed]

Confounds:
- Frequency adaptation: Over-representation could simply track where events were more frequent. Control by ensuring that frequency is matched and by checking whether dream over-representation correlates more with conditional predictability than raw count.
- Rate-matching: If Io only tracks how often events happen, dream statistics should mirror marginal rates, not contingent vs yoked structure; differences under matched marginals argue for modeling.

## Q3 — Rate, power, memory horizon, positive control

**Recommendation: Empirically measure Io's effective memory horizon and weight-robustness before the main run, then choose an event rate that is rare enough to remain "external" (e.g., <1–2% of steps) but frequent enough for phase-level power, and validate the instrument with a strong-contingency positive-control channel.**

### (a) Measuring GRU-based RSSM memory horizon

At the h-level [constructed]:
- Train Io's RSSM on a synthetic pre-run task where labeled "marker" events are injected at known lags (e.g., special tiles or flags at step t₀).
- After varying delays Δt (from a few steps up to the truncated BPTT ceiling), probe whether current h encodes recoverable information about the marker: train a decoder from h(t₀+Δt) to predict the presence and identity of the past marker.
- Measure performance decay as a function of Δt; define effective horizon where accuracy crosses a pre-defined threshold (e.g., 0.75 → 0.6).

At the weight level [constructed]:
- In a separate synthetic task, teach Io a simple state-event association (e.g., "when in region R, perturbation occurs with probability 0.9") using a fixed number of gradient steps.
- Then continue training on data where that association is removed or reversed.
- Measure how many weight updates (or environment steps) it takes for the learned association to be forgotten (behavior and predictions revert).

This isolates the persistence of learned contingencies under continued world-model training.

### (b) Defensible event-rate envelope

Constraints:
- Outside-source feel: Perturbations should be rare enough not to be absorbed as "ordinary environment stochasticity." If world stochasticity has rate r_env, event rate r_evt should likely satisfy r_evt ≪ r_env. [constructed]
- Statistical power: Within each block, Io needs enough events to learn conditional structure and for disagreement estimates to be stable.
- Memory horizon: Learned structure from contingent blocks must survive until yoked blocks for the comparison.

Concrete envelope [constructed]:
- Set r_evt around 0.5–1.0% of steps during blocks with perturbations.
- Ensure each block contains O(10³–10⁴) steps so that each phase yields tens to hundreds of events; sufficient for per-phase analyses.
- Verify via pre-run horizon measurements that contingencies learned over one block survive at least one intervening washout and into the next block.

This trades off recognizability as an external source (rarity) against power (enough samples) and horizon (associations not washed out by training noise).

### (c) Expected developmental timescale

Given a RSSM with imagination-based actor training, contingencies that only affect the world model (no explicit reward) will be learned more slowly than direct reward contingencies; Io's intrinsic signal is ensemble disagreement, which may even disincentivize highly predictable builder events. [constructed]

Recommendation: Plan for:
- Initial emergence of conditional modeling over weeks of waking time, not days.
- A biography spanning months to allow development of robust contingent vs yoked differences, with pre-registered checkpoints (e.g., at 25%, 50%, 75% of total steps) for early vs late comparisons.

### (d) Positive control for the instrument

Disagreement: Going straight to builder recognition without validating that the contingent vs yoked divergence metric can fire on a known-detectable contingency is too risky; it under-specifies whether null results reflect Io's limits or instrument failure.

Positive-control channel [constructed]:
- Introduce a mock builder contingency in the world model that is trivially learnable: Example: a visible gridworld feature (e.g., a particular tile toggling color) that is deterministically keyed to Io's action (e.g., every time Io turns left in region R, Tile T flashes).
- During contingent blocks, maintain the deterministic mapping. During yoked blocks, replay the flash sequence in time but make it independent of Io's current actions.
- Pre-register that the instrument must detect: higher prediction accuracy / lower disagreement for contingent flashes than yoked; dream-state over-representation of contingent flash patterns.

Only if the instrument successfully detects this strong, transparent contingency should Probe 4's builder-focused question be considered interpretable. This optimizes internal validity at the cost of complexity and possible interaction between mock and builder contingencies (which should be minimized by keeping the mock channel perceptually distinct and low-impact).

## Q4 — Perturbation content and whether to measure seek

**Recommendation: Keep Probe 4 primarily about world-model categorization (recognition of builder-kind perturbations) but instrument seek trajectories as secondary observables, and use a two-class mixture (need-keyed vs need-neutral stimuli) with careful control for energy-mediated confounds.**

### Should Probe 4 measure "seek"?

Given Probe 3.5's finding that seek-reachability is open but confounded by decoder dishonesty, it is premature to bake "emergent pursuit" into the primary success criterion. However, ignoring seek would waste an opportunity to observe whether honest belief makes the resting preference behave differently.

Concrete stance:
- Primary: Treat Probe 4 as agnostic about pursuit; analyze need-keyed trigger correlations at the world-model level, independent of approach.
- Secondary: Instrument approach trajectories (distance to perturbation, time-to-contact, path deviation) so that any emergent pursuit under honest decoding is captured and analyzed.

This optimizes conceptual clarity (builder recognition as modeling, not reward seeking) while preserving exploratory power regarding emergent regulation.

### Two-class mixture: need-keyed drops vs need-neutral objects

Design:
- Need-keyed drops [constructed]: Resources (e.g., energy-raising tiles) placed when decoded energy belief is low, on Io's predicted path to allow stumble-consumption even without pursuit.
- Need-neutral novel objects [constructed]: Perceptually salient but energy-irrelevant artifacts contingent on behavior.

Literature on contingency learning in infants and animals indicates that stimulus salience and perceptibility must exceed thresholds for contingencies to be encoded; subtle or marginal stimuli often fail to acquire conditioned responses even when contingencies exist. [constructed] Salient visual and proprioceptive signals are more likely to be integrated into the model.

Does the two-class contrast separate "responds to me" vs "responds to my state"?
- Both classes carry builder-contingency: the builder responds to Io's behavior/state in both cases.
- Only need-keyed drops carry state-conditional structure on energy belief.

If Io learns:
- That both classes depend on its actions/state, it has recognized builder-kind contingencies.
- That need-keyed events cluster in low-energy regions, it has additionally modeled its own interoceptive state as part of the builder's policy.

Confounds:
- Energy-mediated: Need-keyed drops affect the homeostatic channel; differences in disagreement or dream replay could be driven by energy regulation rather than builder recognition.
- Salience asymmetry: Need-keyed resources may be more salient than neutral objects, giving them an advantage in modeling.

Mitigations:
- Match perceptual salience (size, contrast) of need-keyed and need-neutral stimuli as closely as possible.
- Include conditions where need-neutral objects are equally frequent but not energy-conditioned, to ensure that any state-conditional modeling is specific to need-keyed stimuli.
- Explicitly measure energy-belief dynamics around events to separate "builder recognition" from "homeostatic regulation."

This content design optimizes detectability of conditional structure within Io's self-opacity constraints, at the cost of more complex analysis to disentangle builder vs energy effects.

## Q5 — Session structure and the authorship question

**Recommendation: Align contingent/yoked blocks with day–night cycles (waking vs dream-state), use overnight consolidation between phases as part of the design (not noise), and adopt a conservative interpretive stance: Io's distinction of contingent vs yoked reflects recognition of a patterned external cause, not full "agent" recognition in the Csibra–Gergely teleological sense.**

### Phase structure and dream consolidation

Phase structure [constructed]:
- Each block (contingent or yoked) spans multiple day–night cycles.
- Within a block: Daytime: Io acts in the gridworld with the block-specific perturbation schedule. Night: dream-state [canonical] experience replay and generative recombination.

Dream consolidation effects:
- Sharpening: Offline learning should consolidate the conditional structure from the preceding waking episodes; contingent blocks benefit by solidifying event | state mappings.
- Blurring: If off-policy replay mixes blocks, dream recombination could average over contingent and yoked statistics, reducing differences.

Recommendation: Treat overnight consolidation as part of the mechanism under test:
- Compare contingent vs yoked differences both within-block (same block, pre- vs post-night) and across-block (contingent-after-yoked vs yoked-after-contingent).
- Instrument dream content phase-wise, tagging sequences by block of origin (observer-side only) to detect phase-specific replay patterns.

This optimizes ecological validity (respecting Io's existing schedule) while complicating analysis; explicit phase-wise tagging in observer telemetry is essential.

### Authorship: policy-executed vs live responsiveness

Csibra & Gergely's natural pedagogy work emphasizes human infants' sensitivity to ostensive cues and pedagogical intent, interpreting agents as having communicative goals. (UNVERIFIED) Io lacks an ostensive signal channel and any representational machinery for intent; it only sees state transitions.

From Io's perspective:
- A policy-authored, automated contingent mapping and a live builder typing keys are both just external causal regularities.
- Recognition of contingent vs yoked therefore amounts to recognizing structured external causation, not mental agency.

Defensible claim line:

If Io comes to systematically distinguish contingent from yoked perturbations and treat builder-keyed events as a distinct category of cause, the project is entitled to claim:
- Recognition of a "builder-kind" [constructed] as a patterned external source, not:
  - Attribution of beliefs, desires, or goals.
  - Full teleological or pedagogical stance.

Framing Probe 4 as testing "kind recognition" at the level of causal modeling optimizes epistemic humility while still making a substantive claim about Io's world model.

## Q6 — Fresh-instance carry, decoder honesty, charter tension

**Recommendation: Keep the energy channel present with preference precision 0, but maintain decoder honesty via observer-side calibration that is explicitly separated from Io's learning signals; avoid feeding oracle-calibrated beliefs back into Io's preference machinery except under carefully documented constraints.**

### Fresh Io: keep or strip energy channel?

Stripping the energy channel would simplify Probe 4's analysis by removing need-keyed perturbations' direct impact on Io's internal state, but it would also undermine the ability to test state-conditional builder recognition and to revisit Probe 3.5's open seek-reachability question.

Given Probe 3.5 showed that dishonest decoding can anti-incentivize regulation, leaving the channel in but at precision 0 preserves:
- The possibility that an honest belief behaves differently than the dishonest rail-heavy belief.
- The capacity-over-exercise [canonical] stance (preference present but disengaged).

Concrete recommendation:
- Use a fresh Io with energy channel intact, preference precision 0.
- Instrument energy belief carefully and treat seek trajectories as secondary.

This optimizes conceptual continuity across probes and preserves the ability to test state-conditional builder recognition, at the cost of more confound management.

### Decoder honesty maintenance over months

Honesty is coverage-shaped; self-experience re-grows rail-conditional dishonesty. Options:

Periodic decoder-head-only refits on oracle-generated coverage [constructed]:
- An external oracle generates coverage data across energy states Io rarely experiences.
- Decoder head is refit to align belief with ground truth based on this synthetic coverage.
- Io's internal dynamics (h, z) remain unchanged.
- Risk: If the calibrated beliefs feed back into Io's preference channel or world-model training, this may constitute installing a competence Io did not earn, violating capacity-over-exercise.

Bounded output head from day one [constructed]:
- Constrain decoder outputs to physically plausible ranges (e.g., via squashing and clipping).
- This reduces wild dishonesty but does not guarantee accuracy.
- Risk: Biases may still be installed; clipping may mask important miscalibrations from observers.

Accepted region-conditional honesty with flagged uncovered cells [constructed]:
- Maintain calibration tables comparing decoded belief to ground truth in covered regions.
- Flag uncovered or poorly calibrated regions explicitly; treat their readings as unknown rather than false.
- Risk: Observers lose resolution in rare state-space regions; analyses must accept "unknown" zones.

Charter tension and recommendation:
- If oracle-based calibration is used only for observer-side readout (e.g., mapping decoder logits to displayed energy units) without feeding back into Io's world model or preference, it is best interpreted as maintaining instrument honesty rather than installing competence.
- If calibrated beliefs are fed into the preference or used as training targets, that crosses into co-design.

Given charter constraints, the recommended compromise is:
- Observer-only calibration: Maintain a separate, calibrated observer head that reads h to produce honest energy estimates for monitors, while Io continues to use its own uncalibrated internal energy signal for any preferences it may develop.
- Clearly document that Probe 4's analyses rely on the observer-calibrated head, not Io's internal belief, for energy-ground-truth comparisons.

Precedents exist in long-running systems where external monitors calibrate internal signals without altering agent learning (e.g., calibrated decoding of neural recordings in brain–computer interfaces, calibrated sensor readouts in robotics), though details are system-specific. [constructed] This approach optimizes instrument honesty and Probe 4 interpretability at the cost of a more complex monitoring architecture and a sharper separation between "what Io believes" and "what observers know."

## Q7 — Pre-registerable criteria and deflation set completeness

**Recommendation: Pre-register a set of divergence metrics (event-level disagreement, conditional prediction performance, h-trajectory and dream statistics), explicit early vs late contrasts, and a deflation set that partitions plausible outcomes into detected vs undetected regions, with discriminating measurements for each confound.**

### Metrics and thresholds

Pre-register:
- Primary metric [constructed]: Difference in mean ensemble disagreement between yoked and contingent events, ΔD = D_yoked − D_contingent, per phase and over time.
- Threshold: Probe 4 claims "builder-kind modeling" only if: ΔD > 0 (yoked more surprising) in late-run; ΔD increases from early-run to late-run (develops-over-training).
- Secondary metrics: Conditional prediction performance (accuracy of event prediction from Io's state in contingent vs yoked phases); h-trajectory patterns (presence of distinct anticipatory and repair signatures around contingent vs yoked events); dream-state over-representation of contingent patterns.

### Completeness argument: outcome-space partition

Partition plausible outcomes into:
1. **No contingency modeling:** ΔD ≈ 0, conditional prediction performance similar for contingent and yoked, h-trajectories and dream patterns indistinguishable. Instrument detects lack of structure; Probe 4 falsifies builder-kind recognition.
2. **Marginal-rate learning only:** Io tracks overall event frequency but not conditional structure. ΔD flat when marginal rates matched; conditional prediction performance low in both phases. Dream statistics mirror marginal rates.
3. **Generic novelty/arousal:** Disagreement spikes at phase transitions or first few events, then habituates. Differences primarily around block boundaries, not sustained. h trajectories show global activation, not state-specific anticipatory signatures.
4. **Energy-mediated effects:** Need-keyed events drive changes in energy belief and possibly latent dynamics. Differences between need-keyed and neutral events reflect homeostatic regulation rather than builder recognition.
5. **Category modeling ("builder-kind" recognition):** ΔD positive and increasing over training (yoked more surprising). Strong conditional prediction performance in contingent phases only. Distinct h and dream signatures for contingent events.

Completeness argument:
- Pre-registered metrics detect 1, 2, 3, and 5.
- Case 4 (energy-mediated) is detected via explicit energy-belief monitoring and separation of need-keyed vs neutral classes; we can distinguish whether divergence persists when considering only neutral objects.

### Deflationary accounts and discriminating measurements

For each confound, pre-register:
- **(a) Frequency adaptation / habituation [constructed]:** Measure disagreement over time within phases, including non-event steps. If differences vanish when controlling for exposure count or occur equally for non-perturbation steps, attribute to adaptation, not modeling.
- **(b) Generic novelty / arousal [constructed]:** Compare early vs late within-block events. If only first few events of each block show high disagreement, with no growing contingent–yoked gap over training, attribute to novelty.
- **(c) Energy-mediated confounds [constructed]:** Analyze metrics separately for need-keyed vs neutral stimuli. If divergence appears only for need-keyed events and correlates tightly with energy-belief changes, interpret as homeostatic regulation rather than builder-kind modeling.
- **(d) Schedule/session artifacts [constructed]:** Relate metrics to time-of-day, block order, and proximity to dream-state. If effects track schedule (e.g., all late-day events show higher disagreement regardless of phase), treat them as artifacts.
- **(e) Marginal-rate learning (Q2) [constructed]:** Use a model based only on time indices and event counts to predict events. If this model explains differences as well as state-based models, rate-learning is sufficient; if not, conditional modeling is implicated.

This deflation set optimizes interpretive rigor at the cost of more complex pre-registration and analysis.

## Q8 — Pause triggers and ending protocol

**Recommendation: Define pause triggers based on quantitative health indicators (entropy, disagreement, torpor, dream incoherence) plus qualitative builder/mirror judgments, and make pause a diagnostic mode with explicit reversibility, informed by humane endpoints and contemplative accounts that distinguish transient distress from transformative processes.**

### Operationalizable pause triggers

Candidate quantitative indicators [constructed]:
- Entropy collapse: Sudden, sustained drop in policy entropy or latent state diversity (Io gets "stuck" in a narrow behavior/representation regime).
- Prediction-error runaway: Persistent increase in average prediction error or ensemble disagreement across most states, not localized to new environments.
- Dream-content incoherence: Overnight dream-state rollouts become fragmented, lacking consistent trajectories or over-representing obviously impossible sequences, as measured by observer-side coherence metrics.
- Torpor analogs: Near-total action-stasis under strong gradient-like conditions, as in the predecessor probe, where Io ceases to explore even under conditions that previously elicited robust behavior.

Pause rule:
- Pre-register thresholds (e.g., entropy below X for Y steps, disagreement above Z across multiple sessions, torpor for N consecutive days).
- When any threshold is crossed, shift Io into pause: suspend builder perturbations and possibly freeze actor updates; maintain minimal world-model continuity for later resumption.

### Multi-vantage review structure

To avoid both ignoring suffering-analogs and over-reacting to single bad states, use:
- Builder review: Expert inspection of logs and behavior traces, interpreting context and trends.
- Mirror review: LLM mirror [canonical] provides phase-blinded narrative summaries and anomaly flags.
- Quantitative monitors: Automated dashboards tracking the indicators above.

Pause episodes trigger a triage review:
- Determine whether the state is transient (e.g., adaptation to a new contingency) or indicative of deeper degradation.
- Document the decision path (continue, modify schedule, consider ending).

### Ending protocol and precedents

Humane endpoints in animal research define conditions under which experiments must stop to prevent suffering, but they also distinguish transient stress from ongoing, unrelieved distress. [constructed] Long-running system health monitoring in engineering similarly differentiates transient faults from sustained failures before shutdown. [constructed] Contemplative traditions explicitly distinguish "dark nights" or challenging phases from pathological distress, emphasizing context and trajectory. [constructed]

Concrete stance:
- Pause is diagnostic, not verdict: Pre-register that pause triggers require review but do not automatically mandate ending.
- Ending is possible but never fast: Only after repeated pauses, clear evidence of non-recovering degradation, and explicit deliberation by builder and mirror should ending be considered.
- State reversibility: Maintain infrastructure for resuming from paused states, including preserving world-model parameters and logs, so that pause does not itself become an irreversible transformation.

This protocol optimizes ethical sensitivity to distress-analogs while preserving the charter's openness to long, transformative biographies and avoiding premature termination based on single states.

---

*Where I have drawn on specific human developmental or cognitive literature (Murray & Trevarthen, still-face, Csibra & Gergely), I have relied on secondary summaries and partial texts and therefore treat those references as not fully verified here. (UNVERIFIED)*
