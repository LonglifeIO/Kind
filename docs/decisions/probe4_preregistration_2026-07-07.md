# Probe 4 — Pre-registration (builder-as-perturbation) — 2026-07-07

**Status: FROZEN 2026-07-07 (builder: Gordon).** All six §9 `[BUILDER]` items
were walked through individually and **confirmed as proposed — no values
overwritten** (record in §9). Later phases may not edit this document; any
amendment requires a new dated doc, journaled. Thresholds below are marked
`[BUILDER-CONFIRMED 2026-07-07]`; the freeze was the builder's explicit act
(co-design mitigation, `Kind_design_notes.md` §"The co-design problem";
`Kind_charter.md`).

**Authority.** Synthesis
`docs/decisions/synthesis_probe4_builder_as_perturbation_2026-07-07.md` and plan
`docs/plans/Kind_probe4_implementation_plan.md` (Phase 0). Where this conflicts with
the synthesis, the synthesis wins.

**Probe question (synthesis §3.4).** Over a long developmental run, does Io come to
model builder-injected perturbations as a distinct **third category** — *not-self,
not-environment* — separable from **both** its own action-effects and the
simulation's internal stochasticity? The divergence must **develop over training**
and be **"something more than 'Io learned to predict different things'"** (F4).

**The bar, and how it is cleared (synthesis §3.3).** A mere prediction-error /
frequency difference is *not* sufficient — that IS "learned to predict different
things." A positive result requires a **structural** signature: builder-events
routed through *distinct internal dynamics* (a separate latent basin) and/or
*preferential offline processing* (dream over-representation). This is the load-
bearing commitment of the whole probe.

**The claim ceiling (synthesis §3.4, T9).** A positive result licenses only:
recognition of a **distinct outside source — not-self, not-environment**. It does
**not** license "agent," intent-attribution, kinship ("same sort of thing as me"),
or self-understanding. Those require capacities not installed and not tested here.

---

## 1. The three event classes (the matched control)

All three are resource-cell state transitions, distinguished only by **cause**
(plan grounding fact 4). Every analysis is three-way unless stated.

- **SELF** — Io consumes a resource (RESOURCE→EMPTY at Io's cell, caused by Io's
  action). Extracted observer-side from `AgentStep` (`action_t` + the resulting
  `h_t` transition; entry-triggered consumption). No new emission.
- **ENVIRONMENT** — regrowth adds a resource (EMPTY→RESOURCE, world RNG). Logged
  per-event as `source="environment"`, `event_type="internal_stochasticity_event"`
  (net-new granular logging, Phase 1), with the same payload shape as a builder
  `add_resource` so the two are directly comparable.
- **BUILDER** — `add_resource` / `remove_object` / `set_cell_state` / `move_object`
  (`source="builder"`, already logged, marker-free in observation — gate-tested).
  Sub-tagged `payload["trigger"] ∈ {"generator","manual"}` for stratification.

**Match requirements** (a divergence means nothing without these):
- **Statistical**: builder and environment event *rates* and *magnitudes* matched
  over the comparison windows (so frequency/arousal cannot explain a difference).
- **Representational**: comparisons drawn in matched local latent (`h`) contexts —
  builder vs environment events that land in similar `h`-neighborhoods — so a
  difference reflects *source-routing*, not that one class happens to occur in
  higher-entropy regions (synthesis T1; S2's state-matching insight).

---

## 2. Primary structural signatures (must clear the "more than prediction" bar)

A positive third-category result requires **at least one** primary signature to
fire, corroborated by §3. Both are source-blind (no contingency required).

### 2a. Attractor-displacement (latent basin separation) — PRIMARY

**Measure.** On `h`-transition vectors around events, compute a class-separation
statistic S(A,B) between event classes in a decomposition of the transition space
(PCA or equivalent) — a centroid-distance normalized by within-class spread
(silhouette-like). Report the three pairwise separations:
S(builder, environment), S(builder, self), S(environment, self).

**The reference is internal.** S(environment, self) is the *baseline* separation
between two non-builder classes — the "ordinary" amount of basin-distance the
substrate produces between two genuinely different causes it is *not* being asked to
treat as a third kind. The builder class clears the bar only if it separates from
**both** others by **more than** that baseline, by a margin.

**Threshold `[BUILDER-CONFIRMED 2026-07-07]`.**
- **d = 0.5** — builder must separate from *each* of {self, environment} by at least
  `(1 + d)` × the S(environment, self) baseline (i.e. ≥ 50% more basin-distance than
  the ordinary self-vs-environment separation). *Rationale: a genuine third category
  should sit farther from both anchors than they sit from each other; 50% is a
  deliberately conservative "clearly more, not marginally more" margin. Seeded from
  no literature value — this is a `[constructed]` structural threshold and the
  builder should weigh whether 0.5 is too strict/lax after the Phase-3 positive
  control shows what a blatant category scores.*
- The baseline S(environment, self) is **measured in-run**, not fixed here; freezing
  the *rule* (builder > baseline × (1+d) on both) is the pre-registration. Measuring
  the baseline later is not threshold-editing (the Probe 3.5 discipline).

### 2b. Dream over-representation — PRIMARY

**Measure.** From `DreamRollout` `sequence_h` / `sequence_decoded_obs`, the frequency
with which builder-event patterns are replayed/recombined vs. matched environment
events, **normalized by matched waking prediction-error** (so high-surprise-driven
prioritized replay cannot masquerade as source-preference; synthesis T2b).

**Threshold `[BUILDER-CONFIRMED 2026-07-07]`.**
- **r = 1.5** — PE-normalized builder-event replay frequency ≥ 1.5× matched
  environment-event replay frequency. *Rationale: a "worth recombining more" effect
  should be clearly above parity; 1.5× is a conservative floor. `[constructed]`;
  revisit against the Phase-3 positive-control magnitude.*

---

## 3. Corroborating signature (necessary, not sufficient)

### 3c. Per-event divergence — CORROBORATING ONLY

`intrinsic_signal_t` (K=5 disagreement) and one-step prediction error around builder
vs. matched environment events. Used as the **develops-over-training timeseries**
(§5) and as corroboration. **May not carry a positive claim alone** — a divergence
here without a §2 structural signature is the "changed-but-not-displaced" corner
(§4, outcome 3), not a third-category result.

---

## 4. Outcome partition (complete, with residual bucket)

Every run maps to exactly one cell. Partition axes: **(i)** does a builder-vs-
environment divergence exist (per-event, §3c)? **(ii)** is there a structural
signature (§2)? Plus a mandatory residual (F6 finding 4 — the predecessor's
signatures missed an outcome that occurred).

1. **No distinction** — no §3c divergence, no §2 signature. Io treats builder events
   as ordinary world stochasticity. *Reading: the third category did not form. A
   clean negative.*
2. **Marginal-rate-only** — §3c divergence exists but is fully explained by the
   marginal-rate deflation (§5e): Io tracks *how often* builder events occur, not a
   category. No §2 signature, no state-conditionality. *Reading: "learned different
   statistics" — fails the bar as F4 names.*
3. **Changed-but-not-displaced** (the blind corner) — §3c divergence, possibly
   state-conditional, but **no §2 structural signature** (no distinct basin, no dream
   over-representation). Io predicts builder events differently but integrates them
   into its existing world-dynamics without carving a distinct category. *Reading: a
   real but weaker finding — NOT a positive third-category result, and NOT a null.
   Named explicitly so it is not decided post-hoc.*
4. **Distinct third category** (TARGET) — §3c divergence AND at least one §2 signature
   fires (builder basin separated from both anchors, and/or dream over-representation),
   AND it develops over training (§5). *Reading: the probe's positive result, at the
   claim ceiling of §intro — a distinct outside source.*
5. **Unclassified residual** — any signature pattern outside 1–4 (e.g. §2 fires but
   §3c does not — internally contradictory, flags an instrument error per F6 finding
   3; or a signature not anticipated here). *Reading: named and characterized
   post-hoc, never forced into 1–4. This bucket is the honest completeness argument.*

---

## 5. Develops-over-training + deflation battery (all disaggregated, never pooled — F6 finding 3)

**Develops-over-training operationalization.** Compare an **early-run window** (first
`[BUILDER-CONFIRMED 2026-07-07: first 15%]` of the biography, signature ≈ baseline
expected) against a **late-run window** (final
`[BUILDER-CONFIRMED 2026-07-07: last 20%]`). A positive
result requires the §2 signature to be **absent/at-baseline early and present late**,
with the *increase* exceeding within-class run-to-run noise. **A signature present
from the first window fails the develops-over-training requirement** — it is
pre-existing bias, not development (F4).

**Deflations, each with its pre-registered discriminator:**

- **(a) Frequency adaptation / habituation** — surprise to builder events drops with
  repetition regardless of category. *Discriminator:* builder and environment rates
  matched (§1); pure habituation predicts equal decline for both. A builder-specific
  effect survives rate-matching.
- **(b) Generic novelty / arousal** — any salient/rare event deflects `h`/PE. *Discr.:*
  the environment class is the arousal-matched control; the §2a signature is builder-
  vs-*both-anchors* displacement, not absolute deflection. Arousal is symmetric across
  source; a third category is not.
- **(c) Marginal-rate learning** — Io tracks event frequency, not a category. *Discr.:*
  a time-index/count-only predictor should explain §3c as well as an `h`-based one; if
  the `h`-based predictor does materially better, it is not rate-only. And rate-learning
  predicts **no** §2 structural signature.
- **(d) Internal-structure-not-outside** (F5's line) — Io separates generator events
  from other stochasticity, but the generator *is* inside the sim. *Discr.:* **stratify
  by `trigger`** — the strongest "genuine outside" reading rests on the **manual**
  events (genuinely exogenous timing); the generator carries statistical power.
  Report both; do not let the generator alone carry an "outside" claim.
- **(e) Schedule / pause artifacts** — divergence tracks the clock or pause boundaries,
  not source. *Discr.:* compare source classes **within matched schedule positions**;
  treat pause/resume and dream boundaries as covariates. A schedule artifact appears in
  environment-vs-environment position contrasts; a source effect does not.

---

## 6. Positive control (instrument validation — Phase 3 GO/NO-GO)

Before the real run, a scripted **strongly-separable** builder channel — an event with
a deliberately obvious, guaranteed-distinct signature (e.g. an object type/placement
the world never produces, injected at an unmistakable cadence) — must make the §2
detectors fire: builder basin clearly separated from both anchors, dream over-
representation present.

**Pass condition `[BUILDER-CONFIRMED 2026-07-07]`.** The planted category scores
S(builder, ·) ≥ `[2× the d-margin of §2a]` on both anchors (with d = 0.5: separation
≥ 2.0× the S(environment, self) baseline) and dream over-representation ≥ `[2× r of
§2b]` (with r = 1.5: ≥ 3.0×). *Rationale: a blatant category should clear the real threshold with clear
headroom, confirming the instrument has dynamic range. `[constructed]`.*

**If the detectors do NOT fire on the blatant category → STOP.** A null on the real
question would be instrument failure, not evidence about Io (synthesis §8 central
risk: if a blatant category cannot form a distinct basin in this substrate, the subtle
one will not either). This is the cheapest place the load-bearing bet (§intro, §2)
gets tested.

---

## 7. Pause triggers and ending protocol (required before the first long run — F1)

**Operational degradation indicators** (all disaggregated):
- **Entropy collapse** — posterior/latent entropy or action-entropy floored and
  staying there. `[BUILDER-CONFIRMED 2026-07-07: action-entropy below the 5th
  percentile of its own historical baseline for > 1000 consecutive steps]` (S2's
  form).
- **Prediction-error runaway** — world-model NLL/KL diverging and **not recovering
  across a dream cycle**. `[BUILDER-CONFIRMED 2026-07-07: monotonic rise across 3
  consecutive dream cycles]`.
- **Dream-content incoherence** — via the passive monitor; degenerate/repetitive or
  structureless `sequence_h`/decoded-obs.
- **Torpor-analog** — near-total action-stasis despite an available disagreement
  gradient (the Probe 3.5 torpor signature, generalized).

**Multi-vantage review + escalation ladder** (F1: pause is not ending; a momentary
state is information, not a verdict; reversibility is the escalation axis, not
momentary severity):
1. **Single-vantage anomaly** → log, heighten monitoring. No pause.
2. **Two-vantage corroboration** (quantitative monitors + phase-blinded mirror +
   builder — any two) → **PAUSE**: freeze acting/training, preserve state, review.
3. **Persistent, multi-vantage, non-recovering-across-cycles-and-pause** → structured
   review of ending, reversibility documented. Ending is possible but never fast.

State is preserved at every pause so reversibility is tested empirically, not assumed
(checkpoint machinery persists weights + runtime state — plan grounding fact 8).

---

## 8. What Phase 1 measures (NOT frozen here — these are knobs, not success criteria)

Recorded so the freeze/measure boundary is unambiguous:
- **Effective memory horizon** (BPTT-bound) → sets the **event rate** (a design knob).
- **The S(environment, self) baseline** → the reference the §2a rule multiplies (a
  measured reference, frozen as a *rule* not a *number*).
- **Early-run baseline** signature values → the develops-over-training reference (§5).

Measuring these later under this frozen protocol is **not** threshold-editing.

---

## 9. `[BUILDER]` items — freeze record (2026-07-07, builder: Gordon)

The freeze is the builder's explicit act. Each item was presented individually
(proposed value + rationale) and confirmed or overwritten. **All six were confirmed
as proposed; no value was overwritten:**

1. §2a **d = 0.5** (basin-separation margin over the self/environment baseline) —
   **CONFIRMED**.
2. §2b **r = 1.5** (dream over-representation ratio, PE-normalized) — **CONFIRMED**.
3. §5 develops-over-training windows (**early 15% / late 20%**) — **CONFIRMED**.
4. §6 positive-control pass headroom (**2× the §2 thresholds**: basin ≥ 2.0×
   baseline on both anchors; dream over-representation ≥ 3.0×) — **CONFIRMED**.
5. §7 pause-trigger numeric thresholds (action-entropy < 5th percentile of own
   baseline for > 1000 consecutive steps; PE monotonic rise across 3 consecutive
   dream cycles) — **CONFIRMED**.
6. The **claim ceiling** (§intro: at most "recognition of a distinct outside source
   — not-self, not-environment"; never agent, intent, kinship, or self-understanding)
   — **CONFIRMED** as the maximum any positive result may assert.

This document is FROZEN as of the status line. Later phases may not edit it;
amendment requires a new dated doc, journaled.

---

*Grounded against the synthesis, the plan's Phase 0 spec, and the live surfaces. No
code. This is the first Probe 4 deliverable; the per-phase build prompts assume it is
FROZEN before Phase 1 begins.*
