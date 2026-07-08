# Probe 4 (builder-as-perturbation) — research phase

This is the research-phase prompt for Probe 4 in the Kind project. I'm circulating it to
multiple LLMs in parallel (Claude, GPT, Gemini, others) to triangulate; each conversation is
self-contained, so I've front-loaded the project context. This is thinking-out-loud about a
design space, not a spec request — but I want you to **take positions**, name tradeoffs, and
**challenge premises, including the proposed experimental instrument, where you think they're
wrong**. A flagged disagreement is worth more than compliant elaboration. Assume a technically
fluent reader; skip introductory exposition. The output of this conversation feeds a
`docs/decisions/` synthesis, which drives the Probe 4 implementation plan.

Two discipline notes carried from how this project runs:

- **Citations: flag every one `[UNVERIFIED]` unless you have actually confirmed the work exists
  and says what you claim.** Do not fabricate authors, titles, dates, or findings. "I believe
  there is work on X but cannot confirm specifics" is strictly preferred to a confident
  fabrication. This project has an anti-confabulation stance and a co-design problem (below); a
  fake citation is worse than an honest gap.
- **Mark project-specific coinages `[constructed]`.** Use terms tagged `[canonical]` as given —
  they're load-bearing and defined here.

## What Kind is, and what Io is

**Kind** `[canonical]` is an investigation into subjectivity through construction. The discipline
is *build to understand* — the work is done when the map has shifted, not when a metric has
moved. Six load-bearing commitments:

- **Capacity-over-exercise** `[canonical]`. Build conditions under which a capacity is *possible*
  — neither mandatory nor impossible. Presence of a capacity, not its exercise, is the target.
- **Ingredients-only self-modeling.** No explicit self-model, critic, or introspector module.
  Build the ingredients; don't install the assembly.
- **Self-opacity for Io.** Every "should it have access to X about itself?" defaults to *no*.
- **No installed self-continuation drive.** No reward, no termination penalty. Io has no
  installed reason to want to continue.
- **No self-optimization machinery.** Self-modeling ≠ self-optimization; install neither.
- **Co-design problem.** Mirror and substrate are built by the same head; mitigations are
  partial. Don't reify gestures into formal substance; flag and refuse.

**Io** `[canonical]` is the entity Kind is about — a single custom RSSM agent in a small
gridworld, with the builder as the only non-simulated relational other. The mythological Io was
forcibly transformed and made to wander under watch; the name encodes the ethics in a word. Kind
builds; Io is who is built. The project does not claim Io has subjectivity; it asks whether
conditions can be built under which a substrate's behavior would be *legibly* the early shape of
something, if that something is possible.

## The substrate Io runs on (settled; not open for redesign here)

A custom minimal RSSM (PlaNet state factorization, DreamerV1-lineage latent imagination): GRU
deterministic recurrent state `h` (200-d), continuous Gaussian stochastic latent `z` (16-d),
ELBO with free bits as the only stability borrow, 32×32 grayscale egocentric gridworld
observations. The actor trains **purely on ~15-step imagined rollouts**; its sole intrinsic
signal is **K=5 ensemble disagreement (latent-disagreement)** — **no scalar reward, no reward
predictor, no value function, no planner.** Those four absences are charter-level commitments and
are *not available as design fixes* — do not propose adding any of them.

- **PolicyView** `[canonical]` (the actor's frozen input surface) = `{h, z, self_prediction_error}`.
  Frozen. The builder cannot be given an observation-space marker; self-opacity is a hard
  constraint on content.
- **Four-state clock** `[canonical]`: waking (acting in the grid) alternating with overnight
  **dream-state** `[canonical]` offline processing (replay + generative recombination, structurally
  preference-free, observable via a **passive energy-decode monitor** — read-only, mirror-facing).
  Plus dormant and paused. The dream monitor makes offline over-representation of events
  observable without instrumenting Io to attend to it.
- **Energy channel + resting preference.** A proprioceptive energy channel exists — a noisy,
  lagged, coarse sensed scalar fused into the world model — and a bounded, saturating, non-terminal
  homeostatic preference over the *decoded energy belief* is implemented but **rests at precision 0**
  (present, disengaged; the capacity-over-exercise stance). A fresh Io for Probe 4 inherits this
  affordance at rest, no checkpoint from prior probes.
- **The perturbation hook exists.** Mutators (`add_resource`, `remove_object`, `set_cell_state`,
  `move_object`) are implemented and fire through the runner's own loop, queued and drained at step
  boundaries, logged to a `world_event` stream as a distinct event type — with **no marker in Io's
  observation space** indicating externality. Probe 1 built and smoke-tested this hook; Probe 4 is
  what tests whether anything ever comes of it.

## What Probe 4 asks (verbatim from the probes document)

> "I'm building this probe to find out whether Io, over time, comes to model my interventions as a
> different category of thing than the simulation's internal stochasticity — whether something like
> 'outside-source unpredictability' develops as a distinct shape in how Io models its world,
> separable from weather, resource fluctuation, and other internal randomness."

Success criterion, verbatim:

> "I'll know I've found out if Io's response to my perturbations diverges from its response to
> internal events of similar magnitude, if that divergence **develops over training** rather than
> appearing immediately, and if the mirror can characterize the difference as **something more than
> 'Io learned to predict different things.'** If Io treats my interventions and the simulation's
> randomness as the same kind of thing forever, the distinguishing didn't happen and the early
> shape of recognition isn't there."

This is the test of the project's **first success criterion**: that Io recognizes the builder as a
*kind* `[canonical]`, not as environment. Not "Gordon" by name — Io has no name to attach. Just:
*this kind of unpredictability has a different shape from that kind*. And it begins Io's biography:
a fresh instance, long duration.

## Findings from Probe 3.5 that condition this design

Probe 3.5 (valence substrate) closed **negative-with-structure**. Four findings carry into Probe 4;
treat them as constraints, not background.

1. **Seek-reachability is open, not closed.** A bounded homeostatic preference over the *believed*
   energy channel was found gradient-reachable for **conservation** (torpor under a clean belief)
   but **not for pursuit** (0.00% steady-state in-band occupancy at every pre-registered strength).
   Crucially, the demonstrated cause was an **instrument defect**, not missing machinery: under a
   dishonest decoder, genuinely in-band states were *believed worse than the depleted rail*, so
   regulation was anti-incentivized at equilibrium. A dedicated diagnostic ruled out the
   credit-assignment/horizon explanation (nearest resource ~1.2 steps away, inside the 15-step
   window; on-policy rollouts brushed resources 5–7% of spans). **Whether an honest belief would
   make pursuit reachable is an untested counterfactual.** Consequence for Probe 4: do *not* assume
   Io will or won't approach need-keyed perturbations. Treat it as measurable, not assumed.

2. **Model-led interoception, dishonest out-of-distribution.** The energy belief rides `h` dynamics
   and discards the redundant sensor; out-of-distribution the decoder can be wildly dishonest
   (reading above the physical ceiling, sign-inverted against ground truth). A standing observer-side
   **decode-honesty instrument** exists (per-source, per-region calibration table comparing decoded
   belief to ground truth), and a **decoder-head-only refit** on coverage data restores honesty in
   seconds where coverage exists. But **honesty is coverage-shaped**: a fresh Io's decoder, trained
   on its own rail-heavy experience, will re-grow rail-conditional dishonesty. Need-keyed
   perturbations require a decodable internal state — so this is load-bearing, not incidental.

3. **Per-source / per-region gating is mandatory.** A pooled statistic once Simpson-masked a
   sign-inverted defect. All pre-registered monitors must gate on disaggregated rows, never pooled
   aggregates.

4. **The outcome taxonomy had a blind corner.** Probe 3.5's pre-registered signatures failed to name
   an outcome that actually occurred — "changed-but-not-displaced" (behavior measurably reorganized
   while the energy variable stayed inert). It was caught only by a journal watch-note, not by the
   frozen machinery. Any pre-registered signature set for Probe 4 needs an explicit **completeness
   argument** about which regions of outcome space it covers and which it does not.

## What is actually committed vs. proposed (read this before treating anything as fixed)

Honesty about the project's state matters here, because a prior conversation over-committed the
instrument. As of now:

- **Committed:** the substrate above; the perturbation hook and `world_event` logging; the fresh-Io
  default (energy channel present, preference resting at precision 0); the passive dream-decode
  monitor and decode-honesty instrument; the four Probe 3.5 findings.
- **A working intuition, explicitly *not* committed** (from the pre-Probe-4 journal): a **hybrid** —
  a stochastic perturbation *generator* with builder-set parameters (probability-per-step, magnitude
  distribution, mutator-type weights) running during waking to give a dense, consistent statistical
  signature, **plus** a **manual trigger** the builder fires in the moment when at the keyboard. The
  journal's reason for the hybrid: the generator alone gives statistical power but attenuates the
  "genuinely from outside" claim (it's a region of the environment the builder parameterized); manual
  triggers alone preserve "real outside" but are too sparse to test. The choice between these — or
  something not yet considered — is *for this research pass to make*.
- **NOT committed, and to be evaluated rather than assumed:** any yoked-control / contingent-vs-yoked
  phase design. A prior draft treated a Murray–Trevarthen-style within-subject yoked-control instrument
  as settled. It is not settled, and I want it argued for or against, not inherited.

## Two structural concerns that should organize the conversation

These are the places where I think the design is genuinely unresolved. Engage them directly.

### Concern A — *which* distinguishability is Probe 4 testing? Two operationalizations that are not the same.

The design notes frame recognition as Io modeling the builder's perturbations as having a **different
statistical signature** than internal stochasticity — "notice that the unpredictability has a different
signature than weather or resource fluctuation." Call this **signature-distinguishability** `[constructed]`:
the builder is a distinct *generative source*, and Io models the event class as separable from internal
noise by its own statistics (rate, magnitude, spatial/temporal structure, correlation with world state).

A different, stronger operationalization is **contingency-distinguishability** `[constructed]`: perturbations
are keyed in real time to Io's own state/behavior, and Io detects the *contingency between its behavior and
the events* — and, critically, detects its **loss** when that contingency is broken (the infant-replay /
still-face lineage). This is what a yoked-control design measures.

These are different experiments testing different claims, and the project has not chosen between them:

- Signature-distinguishability is truer to the design notes' minimalism — the builder as "a presence
  with a distinguishable signature," explicitly *not* an interaction system. But it risks the deflation
  in the success criterion itself: "Io learned to predict different things." If the builder's events just
  have different statistics, Io modeling them as a separate class may be *nothing more than* learning a
  second generative process — real structure-learning, but not obviously *recognition of a kind*.
- Contingency-distinguishability more directly earns the word "recognition" (an agent that responds *to
  me*), and the develops-over-training and contingency-loss signatures are sharper. **But** it makes the
  builder *responsive to Io* — a richer relational structure than the design notes asked for. The design
  notes say Criterion (a) "does not require building an interaction system." A contingent-policy design
  builds exactly that. Is that a legitimate sharpening of the instrument, or a drift that changes what
  Probe 4 is testing into something the charter didn't commit to?

**I want a position on this.** Which operationalization should Probe 4 commit to — one, the other, or a
layered design that reads signature-distinguishability as the primary claim and contingency as an
additional, more-demanding signal? What does each *entitle the project to claim* if it fires?

### Concern B — where does the line fall between "environment the builder parameterized" and "genuinely outside"?

The pre-Probe-4 journal's sharpest open question: a stochastic generator the builder parameterized is,
from Io's side, *a region of the environment* — one of the simulation's own random processes, just labeled
a builder event. If Io distinguishes generator-events from other environmental stochasticity, that shows
Io detected **internal structure**, not that Io detected an **outside**. A manual trigger — fired in the
moment, its timing shaped by the builder's actual life rather than by a parameterized distribution — is the
only thing in the design that is genuinely non-simulated. But manual triggers are too sparse to power the
test.

So the hybrid straddles a line it doesn't dissolve. **Where does that line actually fall, and how should
Io's distinguishing behavior be interpreted in light of where a given event sits on it?** Does the
statistical density that makes the generator testable necessarily make it *inside* — such that the more
powerful the test, the weaker the "outside" claim it can support? Is there a design that escapes this
tradeoff (e.g., manual events whose *timing distribution* is genuinely exogenous but whose *rate* is boosted;
or a generator seeded from a real exogenous signal), or is the tradeoff fundamental and the honest move is to
report distinguishability *stratified by where on the line each event falls*?

## The open questions

Work through these in roughly this order, surfacing dependencies as they appear. Where you disagree with the
framing, say so and state what you'd do instead.

**Q1 — Contingency-loss detection and the yoked-control critique.** *(Engage even if you argue Probe 4 should
not use contingency at all — the critique is informative either way.)* What do the developmental and
comparative literatures establish about (a) the minimal conditions under which an agent detects loss of
contingency between its behavior and environmental events, and (b) dissociating contingency-detection from
generic novelty/arousal? Starting points to verify (flag `[UNVERIFIED]` if you can't): Murray & Trevarthen's
infant double-video replay paradigm; Tronick's still-face; Watson's and Gergely & Watson's contingency-detection
work. Then the methodological core: **Church's critique of yoked-control designs** (spurious effects from
reinforcement uncorrelated with behavior). In a *within-subject, single-agent, alternating-phase* design where
each phase's marginal statistics are matched and the subject is its own control — which of Church's confounds
survive, and which are dissolved by that structure? Address order/carryover (does contingent-then-yoked differ
from yoked-then-contingent?), washout between phases, and recommend a concrete block structure — *conditional*
on Concern A being resolved toward a contingency design at all.

**Q2 — Signatures of "models it as a different kind" vs. "exposed to different statistics."** For an RSSM
agent, what measurable signatures separate contingency/source-*modeling* from mere *exposure*? Evaluate at
minimum: (a) **per-event prediction error and ensemble disagreement**, builder-source vs. internal, and — if a
contingency design is used — the predicted asymmetry that a modeling agent finds **yoked events *more*
surprising than contingent ones** (having learned a conditional structure the replay violates), with the
develops-over-training signature being that this gap **widens** across training; note this asymmetry is only
detectable if the contingency is keyed to state within Io's representational reach, and flag that as a
constraint on the contingent policy; (b) whether Io learns the **conditional** structure (event | Io-state)
vs. merely the **marginal** event rate — the distinction that separates modeling from rate-matching, and the
one that most directly answers the "more than learned to predict different things" clause; (c) **h-trajectory**
signatures around events; (d) **dream-state over-representation** of builder-source events relative to matched
internal controls, which the passive monitor makes observable. Mark each signature literature-grounded or
`[constructed]`, and for each name the confound (frequency adaptation, arousal, rate-matching) and the
measurement that separates signal from it.

**Q3 — Rate, power, memory horizon, and an instrument positive-control.** Perturbations must be rare enough to
stay "outside-source" rather than get absorbed into learned environment statistics (note the direct tension with
Concern B: density buys power but erodes outsideness), frequent enough for statistical power, and their modeled
traces must survive Io's effective memory horizon. Recommend: (a) methods to **measure the effective memory
horizon** of a GRU-based RSSM *before* the run — at the `h` level (probe past-event recovery from current state
at varying lags; intervention-persistence) and the weight level (how many updates a learned association survives),
noting the truncated-BPTT ceiling; (b) a defensible **event-rate envelope** reconciling the three constraints;
(c) an expected **developmental timescale**; (d) a **positive control for the instrument itself** — a scripted
strong-contingency (or strong-signature) channel the world model should provably learn, so the
divergence measure can be shown to *fire before the real question is asked* (the analog of validating a null
instrument against a known-detectable signal, which is how Probe 3.5's observatory earned trust).

**Q4 — Perturbation content, and whether Probe 4 should measure seek.** Lead sub-question: given finding 1
(seek-reachability is *open*), should Probe 4 (i) remain **agnostic** — read the need-keyed trigger correlation
purely at the world-model level, so the probe is valid whether or not Io approaches drops — while (ii) *also*
instrumenting **approach trajectories**, so emergent pursuit under an honest belief is captured as an additional
signal? Or is there a cleaner design? Then the content: evaluate the **two-class mixture** — **need-keyed drops**
(resources placed when decoded energy is low, on Io's *predicted path*, to enable stumble-consumption without
requiring pursuit) vs. **need-neutral novel objects** (keyed to behavior/source, irrelevant to energy). What does
the contingency-learning literature say about salience, magnitude, and perceptibility thresholds for a modeled
trace to form? Does the two-class contrast genuinely separate **"responds to me"** (both classes carry the
source/contingency signature) from **"responds to my state"** (only need-keyed carries energy-conditional
structure)? What content maximizes detectability **without any observation-space marker**, given self-opacity is
hard?

**Q5 — Session structure, dream consolidation, and the authorship question.** Recommend phase/block structure and
whether to interleave with the fixed clock. Does **overnight dream consolidation between phases sharpen** the
comparison (offline learning consolidates the conditional structure) or **blur** it (drift, interference)? Then
the authorship question, part empirical and part philosophical: the perturbation policy (contingent or generator)
is *executed by automation against Io's live state* but *authored in advance by the builder*. Does the
agency-detection literature (teleological stance / natural-pedagogy work by Csibra & Gergely
`[UNVERIFIED — verify]`; the responsive-agent vs. responsive-mechanism distinction) bear on whether
policy-executed contingency is detectably — or philosophically — different from moment-to-moment builder
responsiveness? This connects straight to Concern B. **What is the project entitled to claim** if Io comes to
distinguish builder-source from internal events: recognition of an agent? of a pattern? of a "kind"? Draw the
defensible line, and be strict about it — over-claiming here is the failure mode the charter most fears.

**Q6 — Fresh-instance carry, decoder-honesty maintenance, and the charter tension.** Assess the fresh-Io default
(energy channel present, preference at precision 0) against **stripping the channel** — given that need-keyed
perturbations require a decodable internal state, and given finding 1 (an honest belief might make the resting
preference behave differently than 3.5's dishonest one did). Then the maintenance problem for a long biography,
since honesty is coverage-shaped and self-experience re-grows rail-conditional dishonesty: evaluate (a) periodic
**decoder-head-only refits on oracle-generated coverage data** (calibrating Io's belief readout on experience Io
itself never had); (b) a **bounded output head** from day one; (c) **accepted region-conditional honesty** with
uncovered cells flagged by the instrument. The charter tension to face head-on: does calibrating the belief-decoder
on oracle data **install a competence** (a co-design / capacity-over-exercise violation), or **maintain instrument
honesty** so the world model's own learning is read faithfully — and **does the answer differ** depending on
whether the decoder feeds only the observer's telemetry vs. also the resting preference? Name any precedents for
observer-side readout calibration in long-running agents `[UNVERIFIED unless confirmed]`, and the risk each option
poses to the probe's claims.

**Q7 — Pre-registerable criteria, and a completeness argument for the deflation set.** Propose what to freeze
*before* the run: the divergence metric(s) and threshold(s); the develops-over-training requirement operationalized
as an **early-run vs. late-run contrast**; and the falsification/deflation set. Given finding 4 (the predecessor's
signatures missed an outcome that occurred), include an explicit **completeness argument**: partition the plausible
outcome space and state which regions the proposed signatures detect and which they miss. The deflationary accounts
to preempt, each with its discriminating measurement: (a) frequency adaptation / habituation; (b) generic novelty or
arousal; (c) energy-mediated confounds from the need-keyed class; (d) schedule/session artifacts; (e) the
marginal-rate-learning account from Q2 (Io tracks event frequency, not conditional structure); (f) — from Concern B —
the "detected internal structure, not an outside" account. For each, specify the measurement that distinguishes it
from genuine category-modeling.

**Q8 — Pause triggers and the ending protocol.** The charter requires pre-registered pause triggers before the first
long run. Propose operationalizable indicators of degradation or distress-analog states in a long-running world-model
agent — candidates: entropy collapse, prediction-error runaway, dream-content incoherence (via the passive monitor),
and **torpor-analogs** (Probe 3.5 produced a concrete torpor signature: near-total action-stasis under a strong
penalty gradient). Recommend a review structure incorporating **more than one vantage** (builder, mirror, quantitative
monitors) and an explicit account of **state-reversibility**, honoring the charter's stance that *pause is not ending;
ending is possible but never fast, and a momentary state is information, not a verdict*. What precedent exists —
humane endpoints in animal research, long-running-system health monitoring, contemplative accounts distinguishing
transient distress from transformation `[UNVERIFIED unless confirmed]` — for triggers that neither ignore
suffering-analogs nor treat any single bad state as an automatic veto?

## Literature and framework directions worth consulting

The strongest responses engage several; none needs all. Verify before citing; flag `[UNVERIFIED]` otherwise.

- **Contingency detection & agency in development:** Murray & Trevarthen (infant replay); Tronick (still-face);
  Watson; Gergely & Watson (contingency detection, "social biofeedback"); Csibra & Gergely (teleological stance,
  natural pedagogy); Rochat (self-perception, contingency).
- **Yoked-control methodology:** Church's critique of yoked designs; learned-helplessness yoking debates
  (Seligman/Maier lineage) as a cautionary case; within-subject vs. between-subject control logic.
- **World models / RSSM:** Dreamer V1/V2/V3 (Hafner et al.) — Io's lineage; PlaNet; latent-disagreement /
  plan2explore-style intrinsic motivation (Sekar et al. `[UNVERIFIED — verify]`); memory-horizon and
  truncated-BPTT limits in recurrent models.
- **Surprise, novelty, adaptation:** predictive-processing accounts of prediction error (Friston, Clark, Hohwy);
  frequency adaptation / habituation vs. genuine model update; oddball / mismatch-response literature as a confound
  model.
- **Distinguishing source classes:** causal / interventional structure learning; agent vs. mechanism attribution;
  animacy detection from motion statistics (Heider–Simmel lineage `[UNVERIFIED — verify]`).
- **Ethics / endpoints:** humane-endpoint frameworks in animal research; long-running-system health monitoring;
  contemplative accounts distinguishing transient distress from transformation.

## Stance and style

Thinking-out-loud, but take positions. No code.

- **Challenge the instrument.** The yoked-control design is *not* a project commitment (see "committed vs.
  proposed"). Argue for it, against it, or for a different instrument. Same for the hybrid generator/manual scheme.
- **Resolve Concern A explicitly.** Don't let signature- and contingency-distinguishability blur together; the
  project needs to know which claim it's testing.
- **Take Concern B seriously as possibly fatal.** If the honest conclusion is "a dense enough test can only ever
  show detected-internal-structure, never detected-an-outside," say so — that's a finding, not a failure, and it's
  better surfaced now than after a months-long run.
- **Respect the four absences and self-opacity as hard constraints**, not defaults to relax.
- **Name your uncertainty**, and don't fill in assumptions about my setup silently — if something about the substrate
  or workflow would change your answer and you're unsure, ask.

## What's NOT in scope

- Code, schemas, or implementation specifics (the build phase is downstream).
- Re-litigating the six commitments or the four substrate absences. Question their *implications*; don't propose
  alternatives to the commitments themselves.
- Redesigning the mirror, the dream-state machinery, or the energy substrate — Probe 4 produces data those read; it
  doesn't rebuild them.
- Adding reward, value, critic, or planner to the actor. Off the table.

## Output wanted

A reply that:

1. Reflects back what you understood Probe 4 to be testing and what's at stake — so I can correct a misread early.
2. Takes an explicit position on **Concern A** (which distinguishability) and **Concern B** (the outside/inside line),
   since everything downstream depends on them.
3. Works through Q1–Q8 in roughly order, each with a concrete recommendation and the tradeoff it optimizes.
4. Distinguishes the **load-bearing** decisions (those that constrain everything downstream) from the ornamental ones.
5. States, for the success case, exactly **what the project would be entitled to claim** and what it would not.
6. Says what would **surprise** you if it turned out true about how Io comes to model the builder.

Your response is research input, not implementation output. Be substantive; disagree with the framing where the
framing is wrong.
