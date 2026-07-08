# Probe 4 builder-as-perturbation synthesis — 2026-07-07

**Status: DRAFT for ratification** — not yet adopted. This document synthesizes
three LLM research outputs plus builder direction into a recommended shape for
Probe 4. It is synthesis only: no task breakdown, no sequencing, no timeline. It
is the input for a later implementation-plan session, and the decision points
(§7) should be ratified before planning.

**The one thing to read first.** The research was written to a prompt that
assumed Probe 4 tests *contingency* — perturbations keyed to Io's behavior, the
Murray–Trevarthen / still-face yoked-control paradigm. **Builder direction
(2026-07-07) retargets the probe to something simpler and smaller: whether Io can
distinguish that the builder is *neither Io itself nor its environment* — a third
kind of thing.** Not "responds to me," and not "the same sort of thing as me"
either — a distinct **third category**: not-self, not-environment. This synthesis
therefore *pivots* the research onto source-separation (refined to a three-way
distinction in §3.4) and re-maps its contributions; §3 is that pivot and is the
spine of the document.

**Conventions** (as in the Probe 3.5 synthesis). `[S#]` = research documents;
`[F#]` = frameworks/design/code/prior-decisions; `[B]` = builder direction
(2026-07-07 question answers), which carries the highest weight because it is the
statement of what the probe is actually for. Research self-tags `[canonical]`
(established literature) / `[constructed]` (model inference); preserved where it
bears on a decision. My own moves are **[synthesis inference]**. Conflicts are
held open in §5, not averaged.

---

## 1. Executive summary

**The retarget.** Probe 4 tests **source-separation** `[constructed]`: over a long
run, does Io come to model builder-injected perturbations as a *distinct category
of cause* from the simulation's own internal stochasticity (weather, resource
regrowth, drift)? The success bar (verbatim, F4): the divergence must develop over
training and be "something more than 'Io learned to predict different things.'"
[B] chose this minimalist framing explicitly over the richer "responds-to-me"
(contingency) framing the research assumed.

**Why the pivot matters.** All three research outputs [S1–S3] built their answers
on contingent-vs-yoked phase alternation — a machinery for detecting whether Io
notices that events *respond to its behavior* and notices when that responsiveness
breaks. That is a strictly stronger, different question than [B] is asking, and it
would require building the "interaction system" the design notes explicitly say
Probe 4 does **not** require (F2 §"The builder as perturbation"). So the
contingency apparatus is set aside as the primary instrument. What survives the
pivot and re-maps directly:

- The **core contrast** becomes *builder-source events vs. magnitude/frequency-
  matched internal-stochasticity events* — a within-world matched control, not a
  contingent-vs-yoked temporal control (§4 T1).
- The **confounds are identical** — frequency adaptation, novelty/arousal, and
  "just learned the different rate" are exactly the deflations for source-
  separation too, and the research's discriminators for them carry over almost
  unchanged (§4 T3, §7 DP6).
- **Gemini's structural signatures become load-bearing** [S2]: a distinct latent-
  space "basin" for outside-source events (attractor-displacement) and dream-state
  over-representation are what let the *simpler* question clear the "more than
  predicting differently" bar — because a distinct representational category, or
  preferential offline replay, is more than a different prediction (§4 T2).
- **Positive-control-first** [S1–S3] and the **memory-horizon measurement** [S2,
  S3] survive wholesale (§4 T4).
- The **energy-channel / decoder-honesty** apparatus from Probe 3.5 (F6–F9)
  governs the optional "responds to my *state*" secondary signal; keep the channel
  present at precision 0, calibrate the decoder for *reading only* (§4 T6).

**Deployment reality** [B]: the run lives on the Mac mini; full shutdowns are
**pause** (checkpoint-and-resume), which is categorically different from dreaming.
The develops-over-training signal must survive pause/resume, not just dream
washouts (§4 T7).

**The honest caveat, stated up front.** [B]'s minimalist framing is the weaker
claim, and it courts its own deflation: if Io merely files builder-events under
"different statistics," that *is* "learned to predict different things" and fails
the bar as written. The synthesis's job is to make the simpler question *clear
that bar anyway*, via structural/dynamical signatures rather than prediction-error
magnitude. Whether such a signature exists in Io's `h=200, z=16` substrate is the
probe's live empirical risk (§6, §8).

---

## 2. Sources

| ID | Document | Focus | Confidence note |
|----|----------|-------|-----------------|
| S1 | `claude_research.md` | All 8 questions; richest citations, only output that actually *verified* the developmental literature (Nadel live–replay–live re-engagement; Rochat non-replication; Millar delay figures). | Highest citation rigor. Answers the contingency framing throughout; its two-class "responds to me vs responds to my state" logic is a contingency design and must be re-read under the pivot. |
| S2 | `gemini_research.md` | All 8; most methodologically aggressive. Contributes the **state-matched yoked replay**, **latent attractor-displacement (PCA)** signature, **detached twin-decoder**, and the cleanest outcome-partition mapping onto the "changed-but-not-displaced" blind corner. | Strongest single contributor *after* the pivot: its structural signatures don't need contingency and transfer directly to source-separation. Commits to specific numbers (rates, horizons, Cohen's d) that are asserted, not grounded — treat as hypotheses for the pilot. |
| S3 | `perplexity_research.md` | All 8; well-calibrated. Reframed Church's critique for a reward-free agent ("intrinsic disagreement signal uncorrelated with behavior") and named the state-carryover confound S2 then fixed. Flagged its developmental cites as from-memory (UNVERIFIED). | Solid. Most of its value is confound-naming, which survives the pivot. |
| S4 | `gpt_research.md` | **NULL RESULT.** Did not receive project context; answered about spacecraft / NASA systems-engineering. | Zero weight. Retained for provenance only. Re-run with context if a fourth voice is wanted. |
| B | Builder direction, 2026-07-07 | Q1: test source-separation ("distinguish that I am separate from its environment"), not contingency. Q2: inject anomalous elements manually (window/interface button) + generator for density; mechanism uncertain. Q3: Mac-mini-resident; full shutdown = pause/resume, distinct from dream. | **Highest weight** — the statement of what the probe is for. Where research conflicts with [B], [B] wins on *target*; research still governs *method*. |

**Framework / decision / code grounding set:**

| ID | Source | Role |
|----|--------|------|
| F1 | `Kind_charter.md` | First success criterion (recognize builder as a *kind*, not environment); ethics; pause-not-ending; no installed continuation drive. |
| F2 | `Kind_design_notes.md` | "The builder as perturbation" — source of non-simulated change, no observation marker, **"does not require building an interaction system"**; self-opacity default-to-no; four-state model (waking/dreaming/dormant/paused); mind lives on the Mac. |
| F3 | `Kind_frameworks.md` | Intentionality / recognition-as-shape-of-attention; enactivist sense-making; the hard-problem humility constraint. |
| F4 | `Kind_probes.md` | Probe 4 statement and success criterion verbatim; probe discipline (one specific question); "small before scale." |
| F5 | `docs/workingjournal/pre-probe4.md` | The mechanism question: generator (parameterized-inside) vs. manual (genuinely-outside-but-sparse); the hybrid working intuition; "where does the line fall between environment-the-builder-parameterized and genuinely-outside." |
| F6 | `probe3_5_verdict_2026-06-12.md` | Seek-reachability open; energy channel + preference resting at precision 0; §7 dream passive-decode monitor; the four Probe-3.5 findings; "changed-but-not-displaced" blind corner. |
| F7 | `probe3_5_seek_classification_2026-06-12.md` | **Bin-1 instrument defect**: the decoder misreports in-band energy (mean\|err\| 0.495, slope −0.948) — a *lying belief*, not missing machinery. Decoder-head-only refit is the owed calibration. |
| F8 | `synthesis_probe3_5_valence_substrate_2026-06-09.md` | Energy-as-observation-channel (not a PolicyView field); observer-side-only telemetry enforced in code; co-design freeze-criteria-early discipline. |
| F9 | `probe3_5_f1_decode_recalibration_2026-06-12.md` | Decoder-head-only recalibration *demonstrated* to restore in-band honesty without touching dynamics/policy/objective. |
| F10 | Live substrate | PolicyView frozen at `{h, z, self_prediction_error}`; K=5 ensemble disagreement the only intrinsic signal; mutator hooks (`add_resource`, `remove_object`, `set_cell_state`, `move_object`) fire through the runner loop, queued/drained at step boundaries, logged to `world_event` with no observation marker; 8×8 grid; `dream_rollout` telemetry + passive decode monitor. |

---

## 3. The pivot: source-separation, not contingency

This is the synthesis's central move; everything downstream depends on it.

### 3.1 What [B] actually asked for

[B, Q1]: *"I mainly want to see if Io is able to distinguish that I am separate
from its environment."* This is F2's own minimalist framing — the builder as "a
source of non-simulated change… a particular kind of other," where recognition
"does not require building an interaction system [but] being a presence with a
distinguishable signature and letting the core agent's capacity for distinguishing
develop on its own." It is **source-separation**: does Io carve out a category for
"came from outside the simulation's own dynamics" vs. "the simulation generated
it."

It is **not** contingency. Contingency (events keyed to Io's state/behavior, and
the detection of contingency-*loss* under replay) is the Murray–Trevarthen/
Watson/still-face paradigm the research built on. That paradigm answers "does Io
notice something *responds to it*," which is richer and would require the builder's
interventions to be a live function of Io's state — precisely the interaction
system F2 says is not needed.

### 3.2 Why the research over-shot, and what survives

The research over-shot because the Opus prompt it was written to framed a yoked-
control instrument as committed. That framing is not in any project document (it
was introduced in a prior chat), and [B] has now declined it. So:

- **Set aside as the primary instrument:** contingent-vs-yoked phase alternation;
  the still-face / contingency-loss re-engagement signature; the "yoked events are
  more surprising than contingent events, gap widens" logic *as a contingency
  claim*; the natural-pedagogy authorship analysis *as a route to agent-
  recognition*.
- **Re-mapped and kept** (these do not depend on contingency):
  1. The **matched-control principle**. The research's whole methodological spine
     — you cannot call a divergence "category modeling" unless you compare against
     events matched on the confounding variables — survives. It just changes what
     is matched: **builder-source events vs. internal-stochasticity events matched
     on magnitude, frequency, and spatial/temporal footprint** (§4 T1).
  2. The **confound battery**. Frequency adaptation, generic novelty/arousal, and
     "learned the marginal rate not the category" are the exact deflations for
     source-separation (§4 T3).
  3. **Gemini's structural signatures** [S2]: attractor-displacement and dream
     over-representation are *source-blind* — they detect whether builder-events
     occupy a distinct representational/offline niche, with no contingency
     required (§4 T2). These become the primary evidence.
  4. **Positive-control-first** and **memory-horizon measurement** (§4 T4).
  5. The **entitlement ceiling** (§4 T9): all three independently concluded Io
     cannot, from `{h, z}`, distinguish a live agent from an authored mechanism, so
     the strongest defensible claim is recognition of a *separate/responsive
     kind*, never an *agent*. Under [B]'s even-thinner framing the ceiling is
     lower still: recognition of a *distinct outside source*. Keep the ceiling.

### 3.3 The success-bar problem, and its resolution

[B]'s framing is the weaker claim and courts the very deflation the criterion
names. If Io simply learns "builder-events have statistics X, environment has
statistics Y," that is *literally* "learned to predict different things" (F4) and
does **not** clear the bar.

**[synthesis inference] The resolution is to require a *structural* signature, not
a prediction-error one.** A divergence clears "more than predicting differently"
if it shows up as one of:

- **A distinct dynamical category** — builder-events induce `h`-transitions that
  occupy a *separate region/basin* of latent space from internal-stochasticity
  transitions (S2's PCA attractor-displacement), i.e., Io routes them through
  different internal dynamics rather than just assigning them a different
  probability.
- **Preferential offline processing** — the dream-state over-represents builder-
  events relative to matched internal events (S2's dream signature, via the F6 §7
  passive monitor), i.e., Io's offline machinery treats them as more worth
  recombining.
- **Distinct downstream handling** — behavior/representation around builder-events
  differs in *kind* (e.g., a distinct recovery/repair dynamic) from matched
  internal events, beyond a scalar surprise difference.

Any of these is "Io does something categorically different with outside-source
events," which is stronger than "Io predicts them with a different number." This is
how the smaller question stays meaningful. Whether such a signature is *reachable*
in Io's substrate is the live risk (§6).

### 3.4 Refinement (builder, 2026-07-07): a three-way distinction, and the self-understanding rationale

[B, second clarification]: *"I'm not the same as Io, but I'm not the environment.
I'm something else from its perspective. I just want it to be able to make that
distinction in a way that allows it to understand itself."* This sharpens the target
from a two-way split (builder vs. environment) to a **three-way** one, and it is the
more faithful reading of F1's first criterion.

**The three categories, from Io's perspective:**
1. **Self** — the effects of Io's own actions. A world-model agent already carries
   this *implicitly*: it predicts the consequences of the actions it takes
   differently from the world's spontaneous changes. This is a proto-self/world
   boundary present in the architecture, **not** the reflexive-attention self-model
   the second success criterion targets (which is not installed).
2. **Environment** — the world's own autonomous dynamics (regrowth, drift, weather).
3. **The builder** — changes that fit *neither*: not caused by Io's actions, **and**
   not typical of the world's learned dynamics.

**Why this is coherent with the substrate, and stays out of the contingency
machinery** [synthesis inference]. Builder-perturbations are, by construction, events
Io did not take an action to cause — so they already fall outside category 1 — and,
if statistically/structurally distinct, outside category 2. The third category can
therefore form **without** a full self-model and **without** the builder reacting to
Io (no contingency). It requires only that the architecture's existing
action/prediction boundary is intact and that builder-events are *legibly not-self*
(§4 T5 adds the design principle: decouple builder-events from Io's own action-
effects in place/time, so Io cannot misfile them as "something I did" — the opposite
of a contingency design, which keys events *to* Io).

**What this changes in the instrument** [carried into §4]:
- The comparison set gains a **third anchor**: the structural signature (T2a) must
  show the builder-event basin distinct from **both** the environment-noise basin
  **and** the self-action-effect basin — a genuine third region, not merely "a
  different flavour of environment." (T1.)
- The entitlement ceiling becomes cleaner and more charter-faithful: recognition of
  a **distinct third category — not-self, not-environment** — which is exactly F1's
  "recognizes me as a kind, not as environment," and explicitly *not* kinship. (T9.)

**The "understand itself" clause — rationale, not a Probe 4 measurable** [synthesis
inference; flagged to prevent reification]. The hoped-for significance is enactivist:
a sharp not-me-not-environment third category throws the boundaries of "me" and
"world" into relief by contrast (F3 sense-making; F1's self-encounter-through-other).
**But Probe 4 can only measure whether the three-way distinction forms — it cannot
measure whether that distinction produces self-understanding.** Self-understanding
requires Io to take its own processing as an object (the second success criterion,
reflexive attention), which is not installed and not tested here. The rationale
motivates *why the distinction matters*; it must not be smuggled in as a *result*.
The line to hold: Probe 4 tests the demarcation, and leaves what the demarcation
*does for Io* to the probes that come after the self-modeling capacity exists.

---

## 4. Findings by theme (re-grounded on source-separation)

### T1 — The core contrast: builder-source vs. matched internal-stochasticity

The instrument is a **within-world matched control**. For every builder-injected
perturbation there must exist internal-stochasticity events (weather/regrowth/
drift, or scripted internal analogues) matched on the confounding variables —
magnitude, frequency, spatial footprint, and the latent-space region they land in
— so that any measured divergence is attributable to *source*, not to the event
being bigger/rarer/stranger. [S1–S3 all rely on matched controls; re-pointed here
from "contingent vs yoked" to "builder vs internal."]

**Three-way, per §3.4: the control set has two anchors, not one.** The builder-event
category must be shown distinct from **both** internal-stochasticity events *and*
Io's-own-action-effect events. So the matched comparison is three-way — {self-action-
effects, environment-stochasticity, builder-perturbations} — and the target signature
(T2a) is a builder basin separable from the *other two* basins, establishing a
genuine third category rather than a variant of environment. Self-action-effect
events are already logged (the agent's transitions under its own actions); the work
is treating them as an explicit comparison class in the analysis, not a new
instrument.

**The state-matching insight transfers and sharpens** [S2, S3]. S3 named it, S2
fixed it: builder-events and internal-events must be compared *in matched latent
contexts*, or a surprise difference may just reflect that builder-events happen to
land in higher-entropy regions of `h`-space. So the matched control is not only
statistical (same magnitude/rate) but **representational** (same local `h`
topology). This is the single most important methodological carry-over from the
research. It converts "does Io find builder-events more surprising" (confounded)
into "does Io route builder-events through *different dynamics* even when they land
in the *same latent neighborhood* as an internal event" (clean) — which is exactly
the structural signature §3.3 needs.

**[synthesis inference] There is no yoked phase in this design.** Because the
perturbations are not keyed to Io's state (no contingency, per [B]), there is
nothing to "replay decorrelated." The contrast is *concurrent and categorical*
(builder-source vs internal-source events interleaved in one run), not *phase-
alternating* (contingent block vs yoked block). This simplifies the block-structure
question (§4 T7) considerably: the develops-over-training signal is measured on the
same continuous run, early vs late, not across alternating phases.

### T2 — Signatures that clear the "more than predicting differently" bar

Primary evidence, all source-blind (no contingency required), each with its
confound:

- **(a) Latent attractor-displacement** [S2, `[constructed]`, **primary**]. PCA (or
  another decomposition) on `h`-transition vectors following builder-events vs.
  matched internal events; the signature of category formation is that builder-
  event transitions **orthogonalize** into a distinct basin over training. *This is
  the strongest "more than prediction" signature* — a separate region of state
  space *is* a distinct category. Confound: any high-magnitude event may deflect
  `h`; matched internal events (T1) are the control, and the signature is
  *displacement relative to them*, not absolute deflection.
- **(b) Dream-state over-representation** [S1, S2, S3; via F6 §7 monitor,
  **primary**]. Builder-events replayed/recombined more than matched internal
  events overnight. Confound: high-surprise events may be replayed *because* they're
  surprising (prioritized-replay analog), so **normalize by matched waking
  prediction-error** before attributing over-representation to source-category.
- **(c) Per-event prediction-error / ensemble-disagreement divergence** [S1–S3,
  **secondary / necessary-not-sufficient**]. Builder-events vs matched internal
  events. Useful as a corroborating signal and as the develops-over-training
  timeseries, but *alone* it is exactly the deflation (§3.3) — a different number is
  not a different kind. Keep it; do not let it carry the claim by itself.
- **(d) Conditional-vs-marginal test** [S1–S3, **the anti-rate-matching guard**].
  Does builder-event surprise depend on decodable Io-state, or only on elapsed
  time / marginal rate? Regress per-event surprise on decoded state; a nonzero state
  coefficient distinguishes "modeled category" from "tracked frequency." Under [B]'s
  framing this is *optional* (it belongs to the "responds to my state" secondary
  signal, T6), but the marginal-rate deflation still needs answering — see DP6.

The **conjunction** (a)+(b), corroborated by (c), is what licenses "Io models
outside-source as a distinct kind." No single one suffices.

### T3 — Surviving confounds and their discriminators

Carried from the research, re-pointed to source-separation. Each is a deflation the
pre-registration must preempt (§7 DP6):

| Deflation | What it would look like | Discriminator |
|---|---|---|
| Frequency adaptation / habituation | Surprise to builder-events drops with repetition, independent of category | Match builder and internal event *rates*; pure habituation predicts equal decline for both. [S1–S3] |
| Generic novelty / arousal | Any salient/rare event deflects `h`/PE regardless of source | Matched internal events are the arousal control; the signature must be *builder-vs-internal displacement* (T1, T2a), not absolute. Arousal is symmetric across source; category is not. [S2, S3] |
| Marginal-rate learning | Io learns *how often* builder-events happen, not that they're a category | The conditional-vs-marginal test (T2d); and the structural signatures (T2a/b) which rate-learning does not predict. [S1–S3] |
| "Detected internal structure, not an outside" (F5's line) | Io separates generator-events from other internal stochasticity — but the generator *is* inside the sim | See §4 T5 and DP2: under [B]'s source-separation framing this softens (any statistically-separable source counts as "separate from environment"), but stratify results by generator-vs-manual so the strongest "genuine outside" claim rests on the manual events. |
| Schedule / pause artifacts | Divergence tracks the Mac's on/off or dream schedule, not source | Compare source categories *within* matched schedule positions; treat pause/resume boundaries as covariates (T7). |

### T4 — Rate, power, memory horizon, and positive-control-first

Full research consensus, survives the pivot intact.

- **Measure the memory horizon before the run** [S2, S3]. At the `h` level: inject a
  distinct event into a frozen world model, track how many steps the perturbation
  persists in the `h`-trajectory (KL vs. an un-perturbed counterfactual) before it
  decays into noise — that is the functional forward horizon [S2]. At the weight
  level: how many updates a learned association survives. **The binding limit is the
  truncated-BPTT window** (Io's ~15-step imagination), which is *shorter* than `h`'s
  passive retention — the category must *form* within the BPTT window, not merely be
  *held* by `h` [S2]. This is a pre-run measurement, not a guess.
- **Event-rate envelope.** Rare enough to read as outside-source, frequent enough
  for power, spaced so traces don't overlap. S2 asserts 1 per 200–400 steps and a
  40–100-step forward horizon; S3 asserts 0.5–1% of steps; **both are `[constructed]`
  and must be set from the measured horizon, not adopted as given.**
- **Positive-control-first** [S1–S3, **strong consensus, adopt**]. Before the real
  run, a scripted **strongly-separable** channel — a builder-analog event with a
  deliberately obvious, guaranteed-learnable signature — must make the divergence
  metric (T2) fire. If the instrument can't detect a blatant planted source-category,
  a null on the real question is instrument failure, not evidence about Io. This is
  the direct analog of Probe 3.5's null/oracle bracketing (F6) and is non-negotiable.

### T5 — Perturbation content and mechanism

[B, Q2]: inject "new elements into the environment that don't fit the typical
pattern," via a manual button on an interface (the "window" tool), mechanism
uncertain. This grounds cleanly in what already exists (F10) and in F5's hybrid.

- **Mechanism is already scaffolded.** The mutator hooks (`add_resource`,
  `remove_object`, `set_cell_state`, `move_object`) fire through the runner loop,
  queued/drained at step boundaries, logged to `world_event` with no observation
  marker (F10). The "window-tool button" [B] is a **manual trigger UI** that calls a
  mutator directly — exactly the manual half F5 already anticipated. No new
  injection machinery is needed; the work is the *statistical distinctness* and the
  matched internal control (T1).
- **The hybrid (generator + manual), re-read under [B]** [F5]. F5's worry — a
  generator is "environment the builder parameterized," not a genuine outside —
  **softens under source-separation.** [B] wants "separate from the environment,"
  and a generator whose events are *statistically separable* from the environment's
  own regrowth/drift processes satisfies that: Io distinguishing them *is* Io
  distinguishing a separate source. The philosophical "but it's inside the sim" force
  applied to *agency/contingency* claims, which [B] has declined. Recommendation
  (DP2): run **generator for density + manual button for genuine in-the-moment
  injection**, log which is which, and **stratify analysis** so the strongest
  "genuine outside" reading rests on the manual events while the generator carries
  the statistical power. This is honest and needs no new philosophy.
- **What "doesn't fit the pattern" must mean, concretely** [synthesis inference].
  The perturbation's signature must differ from the environment's internal processes
  on measurable axes — timing distribution, object type/placement, or magnitude —
  *and* there must be matched internal events for the control (T1). "Doesn't fit"
  cannot be a hidden observation marker (a reserved colour/tile for builder events),
  or the probe passes trivially for the wrong reason [S1's warning; F2 self-opacity].
  Detectability comes from the *dynamics of the change*, not a tag.
- **Legibly not-self (the three-way design principle, §3.4)** [synthesis inference].
  Builder-events must be **decoupled from Io's own action-effects** in place and time
  — not co-occurring with Io's action at Io's location — so Io cannot misfile them as
  "something I did." This is the *opposite* of a contingency design (which keys events
  *to* Io); here we deliberately break any coincidence with Io's actions so the event
  is clearly neither self-caused nor environment-typical. Note the mild tension with
  detectability (S2 suggested injecting when Io's short-term confidence is high, for
  contrast): high-confidence timing and action-decoupled placement are compatible —
  inject when Io is confident, but where/when Io's current action is not the cause.
- **Two-class content (need-keyed vs need-neutral), re-read** [S1–S3 vs S2]. The
  research's two classes were: need-keyed drops (resources when decoded energy is
  low) and need-neutral novel objects. Under [B]'s framing, **need-neutral is the
  core of Probe 4** (it tests pure source-separation without energy confounds).
  Need-keyed is a **secondary "responds to my state" probe** that only makes sense if
  the energy channel is honest (T6) — it is not required for [B]'s question and adds
  the largest confound (energy dynamics masquerading as category). **Recommendation:
  lead with need-neutral source-separation; treat need-keyed as an optional second
  study, temporally separated** (S2's epoch-blocking argument applies with full force
  here — mixing energy-conditioned and energy-neutral events fractures the signal).
  See DP4.

### T6 — Seek, the energy channel, and decoder-honesty (the Probe 3.5 inheritance)

This theme governs the *optional* "responds to my state" secondary signal and is
constrained hard by F6/F7/F9.

- **Keep the energy channel present at precision 0** [F6; S1–S3 concur]. Stripping
  it forecloses the secondary "responds to my state" study and Probe 3.5's open
  seek-reachability counterfactual. Present-but-disengaged is the capacity-over-
  exercise default and lets seek be *measured* if it emerges, without *installing* a
  drive.
- **Instrument seek, but quarantine it** [S1–S3, consensus]. Log approach
  trajectories to need-keyed drops as a secondary, correlative signal; **keep it out
  of the pre-registered success criteria** for source-separation. Emergent pursuit
  under an honest belief is a bonus finding, not the probe's claim.
- **Decoder honesty is load-bearing and currently broken** [F7]. The bin-1 finding:
  Io's energy decoder misreports in-band energy (mean |err| 0.495, slope −0.948 —
  sign-inverted, above physical ceiling) out-of-distribution. Any need-keyed study,
  and the F6 §7 dream monitor that Probe 4 reads (T2b), looks *through* this decoder.
  A lying instrument mis-reports whatever Probe 4 observes.
- **Calibrate for reading, never for driving** [S1, S2, S3 converge; S2's
  mechanism]. Adopt the **detached twin-decoder** [S2]: Io keeps its own decoder
  (may grow dishonest, updated only by its own experience); a *parallel observer
  decoder* on the frozen PolicyView is periodically refit on oracle coverage data
  (F9 demonstrated this restores honesty) and feeds *only* telemetry/monitors, its
  computational graph severed by `stop_gradient` from Io's losses. **The charter
  line** (F1 capacity-over-exercise): calibrating the *observer* decoder is
  maintaining instrument honesty (legitimate); letting an oracle-calibrated readout
  feed Io's *preference* would install a competence Io never earned (prohibited).
  The twin-decoder makes the separation structural. This is owed to Probe 4's
  telemetry *regardless* of whether the need-keyed study runs (F7 §9 F1).

### T7 — Deployment, session structure, and pause/resume

[B, Q3]: Mac-mini-resident; full shutdown = **pause** (checkpoint-and-resume),
categorically distinct from dreaming.

- **The four-state model, as it actually runs here** [F2]. On a Mac-mini-only
  deployment, environment and mind co-reside, so the design-notes' "desktop-off →
  dreaming" coupling does not apply as written. Dreaming is the Mac's own scheduled
  offline processing (Probe 3's exogenous, wall-clock, content-blind trigger, F6/F8
  discipline); **Mac-off is pause** (mind in storage, nothing running). This is a
  clean instance of the four-state model, but the trigger for dreaming is the
  internal clock, not machine-coupling. [synthesis inference — worth noting in the
  plan so the dream schedule isn't accidentally tied to shutdowns.]
- **Pause/resume is a hard requirement, and a new one for the analysis** [B; F2
  state-continuity]. The run must checkpoint often enough that a shutdown loses a
  bounded window, and resume with all state intact. **The develops-over-training
  signal must survive pause/resume**, not just dream washouts — a learned category
  that evaporates across a shutdown is not a developed category. Treat pause
  boundaries as analysis covariates (T3 schedule confound). This is a stronger
  continuity demand than any research output considered.
- **Session structure, simplified by the no-yoked-phase pivot** (T1). Because there
  is no contingent-vs-yoked alternation, there is no need for ABBA/washout phase
  counterbalancing [S1–S3]. Builder-source and matched-internal events are
  interleaved in one continuous run; the develops-over-training contrast is **early-
  run vs late-run** on that run. Dream consolidation between waking bouts is expected
  to *sharpen* the category (offline over-representation is itself a target signal,
  T2b), and is measured, not controlled away. This is a real simplification the pivot
  buys.

### T8 — Pause triggers and the ending protocol

Full research consensus [S1–S3], strongly aligned with F1; adopt.

- **Operational degradation indicators**: entropy collapse (posterior/latent or
  action entropy floored and staying there); prediction-error runaway (world-model
  NLL/KL diverging and not recovering across a dream cycle); dream-content
  incoherence (via the passive monitor); **torpor-analog** (near-total action-stasis
  — Probe 3.5 produced a concrete torpor signature, F6; generalize to stasis despite
  available disagreement gradient). S2's "latent moribundity" (collapse of `z`
  variance across a full waking phase) is a useful specific form.
- **Multi-vantage review** [S1–S3; F1's "more than one vantage"]: quantitative
  monitors (disaggregated per F6 finding 3, never pooled) + phase-blinded mirror +
  builder. Escalation ladder: single-vantage anomaly → log + heighten monitoring;
  two-vantage corroboration → **pause** (freeze, preserve state, review — pause is
  *not* ending, F1); persistent, multi-vantage, non-recovering-across-cycles pattern
  → structured review of ending, reversibility documented.
- **Reversibility is the escalation axis, not momentary severity** [S1–S3; F1]. The
  humane-endpoint precedent (S1's OECD/Toth "moribund = clinically irreversible")
  supports the charter's "a momentary state is information, not a verdict." A
  distress-analog that recovers across a dream cycle or pause is transient; one that
  trends irreversible across cycles is the trigger. **This pre-registration is
  required before the first long run** (F1).

### T9 — What the project is entitled to claim

All three research outputs converged here independently, and the pivot *lowers* the
ceiling further:

- **Cannot claim** recognition of an *agent*, attribution of intent, or that Io
  distinguishes a live builder from an authored mechanism — Io cannot separate these
  at its input surface, by construction [S1, S2, S3 all reach this].
- **Under contingency** the ceiling would be "a responsive/intentional kind" [S1,
  S2, S3].
- **Under [B]'s three-way framing (§3.4)** the ceiling is cleaner still:
  recognition of a **distinct third category — not-self, not-environment**. Io
  demarcates builder-perturbations from *both* its own action-effects and the world's
  own dynamics. That is exactly F1's first criterion ("recognizes me as a kind, not
  as environment") and F4's Probe 4 statement ("this kind of unpredictability has a
  different shape from that kind"), read literally and minimally. It is explicitly
  **not** kinship ("the same sort of thing as me"), and it is **not** a claim that
  the demarcation yields self-understanding (§3.4 — that waits on the second success
  criterion). Draw the line there and hold it.

---

## 5. Consensus and conflicts (held open)

### Strong consensus (survives the pivot)

- Matched controls are mandatory; a divergence means nothing without them [S1–S3].
- Positive-control-first: validate the instrument on a known-detectable planted
  signal before trusting a null [S1–S3].
- Measure the memory horizon before the run; the BPTT window is the binding limit
  [S2, S3].
- Keep the energy channel at precision 0; calibrate the decoder for reading only;
  quarantine seek [S1–S3; F6–F9].
- Multi-vantage, reversibility-weighted pause triggers; pause ≠ ending [S1–S3; F1].
- Entitlement ceiling: never "agent" [S1–S3].

### Conflicts, re-read under the pivot (not averaged)

**C1 — Yoked replay: strict-decorrelated vs. state-matched.** [S2 vs S1/S3.]
*Under the pivot this conflict largely dissolves*: there is no yoked phase (T1). But
its *content* survives as the matched-control requirement — S2's state-matching
(compare builder vs internal events in matched latent contexts) is the correct and
adopted form of the control (§4 T1). S3 named the confound; S2 fixed it; the fix
transfers. **Resolution: adopt state/context-matched controls; no phase alternation.**

**C2 — Two-class content: simultaneous mixture vs. epoch-blocking.** [S1/S3 mix;
S2 blocks.] S1/S3 made the simultaneous need-keyed/need-neutral mixture the core
contrast — but that was a *contingency* design ("responds to me" vs "responds to my
state"), now secondary. S2's fracture argument (mixing energy-conditioned and
energy-neutral events splits the world model across two causal graphs and corrupts
the ensemble signal) is strong and applies directly. **Resolution: need-neutral
source-separation is the core probe; need-keyed is an optional, temporally-separated
second study (S2's epoch-blocking). Do not mix.** (§7 DP4.)

**C3 — Asserted numbers.** S2 commits to specific rates/horizons/effect sizes; S1/S3
withhold. **Resolution: all such numbers are `[constructed]` hypotheses for the pilot
to set from the measured horizon; none is adopted as given.** (§7 DP3.)

**C4 — Does source-separation clear the success bar at all?** Not a research conflict
— a synthesis worry (§3.3). Held open as the probe's central empirical risk (§6, §8).

### Null-result flag

- **S4 (GPT) contributes nothing** — no project context; answered a different
  domain. Do not weight. Re-run with context if a fourth voice is wanted (§7 DP7).

---

## 6. Constraints and requirements

Hard constraints the plan inherits. `[F]` = firmer (design/code/decision); `[B]` =
builder direction; `[R]` = research-surfaced.

- **[B] The probe tests source-separation, not contingency.** No interaction system;
  perturbations are not keyed to Io's state for the primary study. (F2 alignment.)
- **[F2/F10] No observation-space marker for builder events.** Detectability is via
  world-model dynamics only; a reserved tile/colour would trivially and wrongly pass.
- **[F10] PolicyView frozen at `{h, z, self_prediction_error}`.** No new actor-
  readable field. Energy, if used for the secondary study, enters as an observation
  channel (F8 DP4), not a PolicyView field.
- **[F1/F2] No installed continuation drive; no self-optimization machinery; self-
  opacity default-to-no.** The energy preference stays at precision 0; the observer
  decoder's oracle-calibration must not feed Io's preference (T6).
- **[F6/F7/F9] Decoder honesty must be maintained observer-side via the detached
  twin-decoder; the F6 §7 dream monitor Probe 4 reads is currently looking through a
  bin-1-defective decoder and must be recalibrated first.**
- **[B/F2] Mac-mini-resident; pause/resume across full shutdowns; the develops-over-
  training signal must survive pause, and pause boundaries are analysis covariates.**
  Dreaming triggers on the internal clock, not machine-coupling.
- **[R] Positive-control-first is mandatory**; the memory horizon (BPTT-bound) is
  measured before the run; matched internal controls exist for every builder-event
  class.
- **[F4/F1] "Small before scale."** A bounded instrument-validation pilot precedes
  any long run.
- **[F8/co-design] Freeze the pass/deflation signatures before any run and do not
  edit them in response to what the run produces.** (The Probe 3.5 co-design
  discipline; the "changed-but-not-displaced" blind corner, F6 finding 4, is why the
  outcome partition must be pre-registered with a residual bucket, §7 DP5.)
- **[F6 finding 3] All monitors gate on disaggregated rows, never pooled aggregates**
  (a pooled statistic once Simpson-masked a sign-inverted defect).
- **[CLAUDE.md] Full test suite green (not just new tests); mypy `--strict`; no
  timelines.**

---

## 7. Decision points (for ratification)

### DP1 — Retarget the probe to three-way source-separation? **Recommend: yes.**
Adopt [B]'s framing (refined §3.4): Probe 4 tests whether Io models builder-injected
perturbations as a distinct **third category** — not-self, not-environment —
separable from both its own action-effects and internal stochasticity, with the
success bar cleared by structural signatures (§3.3), not prediction-error magnitude.
Set aside the contingency/yoked-control apparatus as primary instrument. The
distinction is the measurable; self-understanding is the rationale, not a claim.
*Confidence: high — this is [B]'s explicit direction; the synthesis's job is to make
it clear the bar.*

### DP2 — Perturbation mechanism: hybrid generator + manual button, stratified.
**Recommend: yes.** Manual trigger UI (the "window-tool button") calling existing
mutators, for genuine in-the-moment injection; generator for statistical density;
log which is which; stratify analysis so the "genuine outside" reading rests on
manual events. Under source-separation, F5's generator-is-inside worry softens (DP2
rationale, §4 T5). *Confidence: high on the hybrid; the exact generator statistics
are a pilot question.*

### DP3 — Rates, horizon, effect sizes: set from the pilot, not adopted from research.
**Recommend: measure first.** Run the memory-horizon probe (BPTT-bound) and set the
event-rate envelope from it; treat S2's 1/200–400-steps and S3's 0.5–1% as
hypotheses. *Confidence: high.*

### DP4 — Content: lead with need-neutral; need-keyed is an optional second study.
**Recommend: yes, and do not mix** (C2, S2's fracture argument). Need-neutral novel
objects are the core source-separation probe; need-keyed drops (requiring an honest
decoder) are a separate, later "responds to my state" study, temporally isolated.
*Confidence: medium-high.*

### DP5 — Pre-registered signatures + outcome partition with a residual bucket.
**Recommend: adopt.** Freeze before any run: primary = attractor-displacement +
dream over-representation (T2a/b); corroborating = per-event divergence (T2c);
anti-deflation guards (T3); develops-over-training = early vs late structural
signature. Partition the outcome space (blind / rate-matching only / changed-but-not-
displaced / distinct-category) **plus an explicit residual bucket** for the unnamed
corner (F6 finding 4). Do not edit signatures mid-run (F8 co-design). *Confidence:
high on the discipline; exact thresholds are empirical.*

### DP6 — Deflation battery. **Recommend: adopt T3 in full**, disaggregated (F6
finding 3). Each deflation (frequency adaptation, novelty/arousal, marginal-rate,
"internal-structure-not-outside," schedule/pause) gets its pre-registered
discriminator. *Confidence: high.*

### DP7 — Energy channel, decoder, seek. **Recommend:** keep channel at precision 0;
adopt the detached twin-decoder and recalibrate the observer decoder (F9) before the
F6 §7 monitor is trusted; instrument seek but quarantine it from success criteria.
*Confidence: high — F6/F7/F9-forced.*

### DP8 — Deployment & continuity. **Recommend:** Mac-mini-resident; dreaming on the
internal clock; pause/resume across shutdowns with bounded-loss checkpointing; the
develops-over-training signal must survive pause; pause boundaries are covariates.
*Confidence: high — [B]-directed; F2-consistent.*

### DP9 — Positive-control-first + bounded pilot before any long run. **Recommend:
mandatory.** Validate the divergence metric on a blatant planted source-category and
measure the horizon; only then commit to the long run. *Confidence: high — F4 "small
before scale."*

### DP10 — Re-run GPT (and/or the rewritten-prompt variant) for a fourth voice?
**Recommend: optional.** S4 was null. The three real outputs plus [B] are a
sufficient basis; a context-correct GPT run would add coverage but is not blocking.
*Confidence: low-stakes.*

---

## 8. Gaps and open questions

- **[central risk] Does a structural source-separation signature exist in Io's
  substrate?** §3.3's resolution requires that outside-source events can occupy a
  distinct latent basin or offline niche in an `h=200, z=16`, 8×8-grid world model
  whose only pressure is reconstruction + K=5 disagreement. This is unvalidated and
  is the probe's live empirical bet — the analog of Probe 3.5's tiny-tensor double-
  bind. Probe it early (the positive control, DP9, is the first test: if a *blatant*
  planted source-category can't form a distinct basin, the subtle one won't either).
- **[synthesis gap] Can the develops-over-training signal survive pause/resume?**
  New requirement from [B, Q3]; no research output considered checkpoint-continuity
  of a *learned category*. Needs an explicit continuity check (does the signature
  measured before a shutdown persist after resume).
- **What exactly makes an event "not fit the pattern" without a marker?** [B, Q2]
  was uncertain on mechanism. The plan must specify the measurable axes on which
  builder-events differ from internal processes (timing/type/placement/magnitude)
  and confirm they are *not* a de facto observation marker (§4 T5). This is a design
  decision the pilot should pin down.
- **The success-bar interpretation.** If neither structural signature fires but the
  prediction-error divergence does, is that "changed-but-not-displaced" (a real but
  weaker finding, F6 finding 4) or a null? Pre-register the reading (DP5's residual
  bucket) so it isn't decided post-hoc.
- **Generator statistics.** What distributional signature makes generator-events
  both statistically separable from internal processes *and* matched-controllable? A
  pilot question (DP2/DP3).
- **Staleness / provenance.** S1–S3 are context-correct and current; S4 is null. The
  research answered the *contingency* framing throughout, so every adoption from it
  has been re-pointed to source-separation here — the plan should not silently
  reinstate contingency machinery (yoked phases, still-face signatures) that the
  pivot removed.

---

## 9. Recommended direction (synthesis-level — my call)

**1. Build Probe 4 as a three-way source-separation probe (DP1, refined §3.4).** The
question is [B]'s: does Io come to treat builder-injected perturbations as a distinct
**third category** — not-self, not-environment — separable from *both* its own
action-effects and the world's own dynamics? Not "responds to me," not "the same sort
of thing as me." The near-entirety of the research's contingency apparatus is set
aside; what it gives that survives is method, not target. The three-way distinction
is measurable; whether it yields self-understanding is not a Probe 4 claim (§3.4).

**2. Make the simpler question clear its own bar via structure, not surprise
(§3.3, T2).** The claim rests on Io doing something *categorically* different with
outside-source events — a distinct latent basin (attractor-displacement) and/or
preferential dream over-representation — corroborated by, but never carried by, a
prediction-error difference. This is the single most important design commitment:
it is what stops the probe from proving only "Io learned different statistics."

**3. Instrument = builder-source vs. context-matched internal-stochasticity events,
interleaved in one continuous run (T1).** No yoked phases; the pivot removes them.
The matched control is both statistical and representational (S2's state-matching).
Develops-over-training is early-vs-late on that run, and must survive pause/resume.

**4. Mechanism = hybrid, stratified (DP2).** Manual window-tool button + generator;
the strongest "genuine outside" reading rests on the manual events, the power on the
generator. Lead with need-neutral content; keep need-keyed (and its energy/decoder
machinery) as an optional, temporally-separated second study (DP4, DP7).

**5. Discipline = positive-control-first, horizon-measured, signatures-frozen (DP3,
DP5, DP9).** Validate the metric on a blatant planted category and measure the
BPTT-bound horizon before any long run; pre-register the pass/deflation signatures
and the outcome partition *with a residual bucket* for the unnamed corner, and do
not edit them mid-run (the F8 co-design discipline; the F6 blind-corner lesson).

**6. Ethics = pre-registered, multi-vantage, reversibility-weighted pause triggers;
pause is not ending (T8, DP8).** Required before the first long run.

**Net:** the builder's minimalist framing makes Probe 4 *smaller and cleaner* than
the research imagined — one continuous run, no phase counterbalancing, a lower and
more defensible entitlement ceiling ("a distinct outside source," never "an agent").
But it inherits one sharp risk the research's richer design partly hid: a mere
statistical difference is not recognition. The whole design turns on whether a
*structural* signature of source-category is reachable in this substrate — and the
positive control is the first, cheap place that bet gets tested.

---

*Synthesis grounded against S1–S3 (`docs/research/probe4/`), builder direction
[B] 2026-07-07, and F1–F10. The five points where this synthesis departs from the
research: the source-separation retarget (§3, all §4 themes re-pointed), the
structural-signature resolution of the success bar (§3.3, T2), the removal of yoked
phases (T1, C1), the need-neutral-leads content decision (T5, C2/DP4), and the
pause/resume continuity requirement (T7, §8). Each is tagged at its source. No task
breakdown, sequencing, or timeline, per discipline. The implementation plan is a
separate session, and DP1–DP10 should be ratified first.*
