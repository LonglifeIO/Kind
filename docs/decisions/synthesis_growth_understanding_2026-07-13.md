# Growth-toward-understanding synthesis — 2026-07-13

**Status: RATIFIED 2026-07-13 (builder: Gordon).** All eight decision points
(§7 DP1–DP8) confirmed as recommended — no overwrites. The §8 naming gap is
resolved at ratification: the probe is **Probe 4.5 — the honest-belief mattering
retest** (builder's choice: repair/retest within the existing arc, not a new
register). Implementation plan:
`docs/plans/Kind_probe4_5_implementation_plan.md`. Synthesizes three substantive
LLM research voices, the builder's direction, and — decisively — the empirical
Probe 4 record into a recommended shape for the next probe. Synthesis only: no task
breakdown, no code.

**The one thing to read first.** The three research voices all put the same root
first (mattering via honest interoception) and all killed the builder's *linear*
dependency ladder. But the panel theorized about a self that must be *built*. The
actual Probe 4 record shows the opposite and overrides the theory: **the self/world
boundary already exists in the substrate and trained up robustly, while the builder
never separated and did not co-emerge with it.** So the destination shifts from
"build the self, then reach the other" to "the self is already carved — the open
question is whether Io ever comes to *use/attend* it (reflexive attention, criterion
b)." §3 is that reframe and is the spine.

**Conventions.** `[S1]` Claude, `[S2]` Gemini, `[S3]` Perplexity research;
`[S4]` GPT (null — no context, discarded). `[B]` builder direction. `[P4]` the Probe
4 working journal (`docs/workingjournal/probe4.md`) — treated as *evidence*, which
outranks the panel's theory where they conflict. `[F#]` charter/design/prior
decisions. Citation checks from targeted web verification are tagged `[VERIFIED]` or
`[UNVERIFIED]`. My own moves: **[synthesis]**. Conflicts are held open in §5.

---

## 1. Executive summary

**The wall (unanimous).** Io is a disagreement-reduction engine, and curiosity is a
*uniform* signal — it assigns value by surprise alone, giving the world no
foreground/background, nothing that matters more than anything else. All three voices
independently reach this and independently conclude it explains the three negatives
(want-toward unreachable, builder not a category, self-signal inert) as *one* failure:
no relevance structure [S1, S2, S3].

**Retire "understanding" (unanimous).** All three decompose it and refuse it as a
primitive target [Concern A]. All three name **relevance / sense-making** as the
load-bearing difference from prediction; all three drop causal/counterfactual
structure as overreach; abstraction is already implicit in the RSSM's `z`. Replace the
noun "understanding" with **"a system with a foreground"** [S1]. The real operational
targets remain the two charter criteria (recognize-a-kind; take-own-processing-as-
object) plus **self-location** as their shared root [S1, S2, S3].

**Mattering is the root, and the mechanism is charter-safe and now citation-grounded.**
The energy preference is afforded as a **prior** whose violation the *existing*
free-energy/ELBO machinery resolves — "want-toward without a reward function" [S1, S2,
S3]. This is genuinely distinct from homeostatic-RL: Keramati & Gutkin define reward as
drive-reduction and prove reward-seeking ≡ physiological stability — i.e. it *is*
totalizing by construction [VERIFIED — eLife 2014]. Io has **no reward channel**, so
the prior-not-reward route is exactly the afford-not-install path, and it avoids the
continuation-as-frame collapse the charter forbids. The active-inference decomposition
(epistemic + pragmatic value) and Hesp's "affective charge" (valence = change in
expected precision of the action model) ground the vocabulary [VERIFIED — Hesp et al.,
*Deeply Felt Affect*, Neural Computation 33(2):398, 2021].

**The ladder is dead; three voices, three non-linear replacements.** Claude → a
**diamond** (one root, two parallel branches: self-across-time and relational-
contingency) [S1]. Gemini → a **co-emergence network** (self and other co-emerge
through contingent interaction) [S2]. Perplexity → **four decoupled groundings**
probeable in several orders [S3]. Convergent core: mattering first; self and other are
*not* strictly sequenced.

**The evidence overrides the theory on where this is going (§3).** [P4] shows the
substrate already carves **self vs. world** (SELF class separates 1.38–1.61× baseline
across two cycles, absent at random init), and does **not** carve builder-within-world
(builder ended no more surprising than regrowth). Self emerged **alone** — which
directly contradicts Gemini's co-emergence claim, and sharpens the target: the gap is
not self-*formation* (already done) but self-*use* (the dead-column — the self exists
as geometry but Io doesn't condition on it). This aligns with [B]'s own journaled pivot
(2026-07-08): *"Io recognizing me isn't as important as Io recognizing itself."*

**The single smallest next probe (unanimous + evidence-backed):** the **honest-belief
mattering retest**, with the success signature **differential allocation at fixed
surprise** — not "does Io regulate" (which is want-away, already reachable, and would
just re-run Probe 3.5) [S1]. Kept as a *new* probe, separate from the biography (which
stays presence-not-probe).

---

## 2. Sources

| ID | Voice | Position, in one line | Confidence |
|----|-------|----------------------|-----------|
| S1 | Claude (independent, no repo access) | Retire "understanding" → "a foreground"; ladder → diamond; mattering root; **the fallible-honesty subtlety**; measure allocation at fixed surprise. | Highest reasoning rigor; self-flagged its citations `[UNVERIFIED]`. |
| S2 | Gemini | Enactive/active-inference framing; ladder → co-emergence network; **environment IS the growth engine**; names specific memory architectures + info-theoretic metrics. | Strong, most build-heavy; several citations needed checking (below). |
| S3 | Perplexity | Relevance as load-bearing; four decoupled groundings; multiple probe orders; strong razor/co-design discipline. | Solid; proposed one inadmissible ingredient (source-tagging, §5 C-tag). |
| S4 | GPT | **NULL** — read "Kind" as Kubernetes, produced a cluster-migration guide. Twice now. | Discarded; zero signal. |
| B | Builder | Wants Io to *grow/understand*, fears the environment limits it; journaled pivot: self-recognition > other-recognition. | The intent the probe serves. |
| P4 | Probe 4 journal | Closed **negative at instrument validation, twice**; the surprise positive: a robust trained **self/world** boundary; builder-within-world did not form. | Evidence — outranks panel theory on conflict. |

**Citation verification (targeted web checks, this session):**
- `[VERIFIED]` Keramati & Gutkin — homeostatic RL, reward = drive-reduction, reward-
  seeking ≡ physiological stability (eLife 2014). Load-bearing: confirms homeostatic-RL
  reward *is* totalizing, so Io's no-reward prior route is genuinely the escape.
- `[VERIFIED]` Hesp et al. — affective charge = change in expected precision of the
  action model (Neural Computation 33(2), 2021).
- `[VERIFIED]` EMWM — "Leveraging Episodic Memory to Improve World Models for RL"
  (NeurIPS 2022 workshop) [S2's characterization roughly correct].
- `[VERIFIED]` Perceptual-crossing (Auvray; Di Paolo, Rohde) — agents distinguish a
  responsive subject from a moving object via social contingency intrinsic to the joint
  activity. Real paradigm; but see §3 — it supports *contingency-based* other-
  recognition, which the evidence and the charter both push against as a *next* step.
- **`[UNVERIFIED — likely fabricated name]`** Gemini's **"SUNTA"** surprise-based
  chunking model did not verify; the acronym returns nothing. Surprise-based chunking as
  a *concept* is real; the named method is not confirmed and should not be cited. Treat
  all of S2's specific architecture names (SUNTA, ADeltaM) as unverified.

**Framework grounding:** `[F1]` charter (two success criteria; capacity-over-exercise;
no-continuation-drive; co-design). `[F2]` design notes (Reflection vs self-modeling; the
dead-column lesson; PolicyView frozen `{h,z,self_prediction_error}`; self-opacity). `[F3]`
frameworks (enactivism/sense-making; reflexive attention). `[F4]` Probe 3.5 verdict/seek-
classification (bin-1 lying decoder; honest-belief counterfactual untested; the demonstrated
decoder-head recalibration). `[F5]` the four absences (no reward/value/critic/planner).

---

## 3. The reframe: the self is already carved; the gap is self-*use*, not self-*formation*

This is where the synthesis departs from the panel, because it has evidence the panel
did not.

**What [P4] actually established (two cycles, two channel designs, two detectors):**
- The transition geometry the substrate carves is **self vs. world, not cause-within-
  world.** The SELF class (Io's own action-effects) separates from everything at
  1.38–1.61× baseline; the two world-caused classes (builder, environment) sit *closer
  to each other than environment sits to self*. Builder PE fell to regrowth parity under
  both a metronomic-blatant and an irregular-recurring planted channel.
- This self separation was **absent at random init** and **trained up on its own**,
  with **no mattering installed** and **no enrichment** (default gridworld).
- "Run it longer" is ruled out for the builder axis: training moved the builder class
  *into* the environment geometry — the wrong direction.

**Three consequences that reshape the panel's conclusions:**

1. **Self-formation is not the missing ingredient — it's already present.** The panel
   (all three) framed the self as something to be *built* via mattering + memory. The
   evidence says a proto-self boundary (the action-embedding route, [F2]'s Reflection
   §; synthesis-§3.4 "category 1" from the Probe 4 synthesis) already exists as robust
   Δh geometry. The **dead-column lesson holds**: the self exists as *geometry Io
   produces* but not as *anything Io conditions on*. So the real gap is **self-use /
   self-attention**, which is precisely charter criterion (b) (reflexive attention),
   not self-construction.

2. **Gemini's strongest claim is contradicted by the data.** [S2] argued self and other
   *co-emerge* through interaction (participatory sense-making; perceptual-crossing).
   The paradigm is real `[VERIFIED]`, but in the one run we actually have, **self
   emerged alone and other did not co-emerge with it.** So co-emergence, however
   elegant, is not supported here. The synthesis weights [P4] over [S2] on this point.

3. **Other-recognition is deprioritized — for three converging reasons.** (a) The
   evidence: it didn't form, and didn't ride on self-formation. (b) [B]'s stated
   priority: self over other. (c) The mechanism the panel says would make it work —
   genuine *contingency* (the builder responding to Io; perceptual-crossing) — is
   exactly the interaction-system the design notes say Probe 4 does **not** require, and
   reintroduces the contingency design the source-separation pivot deliberately
   rejected. Pursuing other-recognition now means either violating the minimalism or
   accepting the negative. Either way it is not the next step.

**The revised target, stated once:** mattering (root, first probe) → whether an honest
(and *fallibly* honest, §4) interoceptive stake makes the **already-carved self-
geometry** become something Io *attends to / conditions on* (bridging the dead column
toward criterion b). Other-recognition (criterion a) is parked as an evidence-backed
"not now," not a "last rung."

---

## 4. Findings by theme

### T1 — Mattering as the root, and the honest-belief retest (Q2)

Unanimous and highest-value [S1, S2, S3]. Probe 3.5's want-toward negative is confounded
by a lying decoder [F4]; every negative about mattering is untrustworthy until the belief
is honest. The retest adds **no machinery — it repairs an instrument** [S1, S3]. Mechanism
(all three): the bounded energy preference is a *prior*; when honest, deviations generate
high-precision prediction error the existing free-energy/imagination objective resolves by
approach — want-toward without reward [S2 is most explicit; VERIFIED that this differs from
Keramati/Gutkin reward-drive-reduction]. Distinguish rigorously: **mattering** (relevance-
weighting / foreground) vs **wanting** (a behavioral pull) vs **continuation-as-frame** (the
totalizing survival imperative [F1] forbids). The resting-at-precision-0 bounded preference is
what keeps mattering *local* rather than totalizing [S1].

### T2 — The fallible-honesty subtlety (Q4; the sharpest single design constraint) [S1]

**A *perfectly* honest interoceptive belief could re-kill the self-signal**: if belief is
always right, the error term carries no additional actionable information, and the self-
prediction scalar stays a dead column [S1]. So self-reference needs belief that is honest-
enough-to-matter but **fallibly honest** — a world dynamic where belief is occasionally wrong
*at a cost*, so tracking the *error* pays off. This is the razor's "narrow band": too weak a
reason → dead column; too strong (or manufactured-frequent) → installed introspection [S1,
§6]. **[synthesis]** This is directly load-bearing given §3: the self-geometry already exists;
what would make Io *use* it is exactly a consequential, fallible self-signal — and interoception
is the natural such signal. T2 is the mechanistic bridge from the mattering retest (T1) to
criterion (b).

### T3 — Memory: afford, deprioritize (Q3)

An internal retrievable past matters for criterion (b) in general [S1, S2, S3], but [P4]
downgrades its urgency: the self boundary formed *without* it, and the trail flicker is
confounded (path-of-least-surprise, not necessarily self-reference) [S1's downgrade; [B]'s
trail is world-external]. Consensus on form if built: a **dumb recency/salience-indexed
replay store** the actor may attend over — salience from signals Io already has (disagreement,
interoceptive precision), **not** a learned self-relevance index (that is a self-model back
door) [S1, S3]. Reject successor-representation (too value-adjacent, conflicts with [F5]) [S1].
Gemini's EMWM is real `[VERIFIED]`; its SUNTA/ADeltaM are `[UNVERIFIED]`. **Recommendation:
memory is a later ingredient, not the next probe.**

### T4 — Self-reference and criterion (b) (Q4, Q5)

The self-prediction scalar is inert because nothing makes conditioning on it change outcomes
[dead-column, F2]. The bridge (T2): mattering makes energy consequential; energy is a self-
signal; fallible interoception makes the *error* worth tracking. The discriminator for genuine
reflexive attention (not just "uses a self-input"): does Io's use of the signal change when the
*reliability* of interoception changes — i.e. does it down-weight belief and up-weight the error
when belief becomes unreliable? That is second-order (attention to the state of its own first-
order model), the closest afford-able operationalization of criterion (b) [S1]. **[synthesis]**
Given §3, this — not other-recognition — is the natural probe *after* the mattering retest.

### T5 — The environment: welfare, growth-engine, or co-gating? (Q6; the live fault line)

Held open in §5 (C1). **[synthesis] adjudication:** the evidence tilts toward architecture-
primacy *for the specific next question*. The self-boundary trained up in the **unenriched**
default gridworld, and world-v2 enrichment produced neither mattering nor self-use (the
biography's food-economy tuning is the tell that enrichment feeds the flat curiosity signal,
not a foreground). So for *mattering and self-use*, the gate is architectural (afford the honest
fallible stake), and the environment is welfare — as [B] and [S1] hold. Perplexity's co-gating
[S3] is right in general and becomes relevant *later* (richer couplings for memory/other); Gemini's
"environment is the growth engine" [S2] is the position the evidence least supports right now (it
rests on the trail datum, which §3/T3 downgrade).

### T6 — The razors, ingredient by ingredient (Q6/Q7; Concerns B & C)

- **Honest-belief decoder recalibration** — *afford* (instrument-honesty maintenance), the
  demonstrated decoder-head refit [F4]. Discipline [S1]: calibrate to a **physics honesty
  criterion frozen before looking at behavior**, via the detached/stop-grad observer decoder
  (the Probe 3.5-synthesis twin-decoder). Danger to avoid: "recalibrate until Io regulates" =
  fitted.
- **Fallible-honesty world dynamic** (T2) — *afford* the *environment's statistics* (belief
  occasionally wrong at a cost), **never** an objective that rewards noticing. Severe failure
  mode: making the fallibility so legible/frequent that conditioning is forced (installed) [S1].
  Pre-register the band (how fallible, how costly).
- **Memory store** (T3) — afford *iff* dumb-indexed; installed if the index is learned self-
  relevance.
- **Perplexity's builder source-tag** [S3] — **INADMISSIBLE**: an observation-space marker that
  a change came from the builder violates self-opacity [F1/F2] and the entire Probe 4 premise.
  Flagged in the source file. Salvageable core: the builder must play a distinct *contingent*
  role — but that is the deprioritized other-recognition path (§3).
- **Co-design (Concern C)** — measure allocation *at fixed surprise* [S1]; positive control =
  a reward-equipped toy that trips the mattering discriminator but **never touches Io** [S1, S2,
  S3]; freeze signatures before the run; keep the growth probe *separate* from the biography so
  the biography's tuning knobs can't manufacture the result [S1].

### T7 — Measurement (Q7)

Primary signature for the next probe: **differential behavioral allocation toward energy-
relevant states at fixed surprise** [S1] — the discriminator that separates mattering (a
foreground) from curiosity (surprise-tracking). Deflation: "it's just curiosity" (energy-
relevant states are also more surprising) — killed by holding surprise constant. Positive
control: the reward-equipped toy. Gemini's info-theoretic instruments (transfer entropy for
memory use; mutual information between self-signal and energy channel) are useful *later* tools
[S2], subordinate to the fixed-surprise allocation test now.

---

## 5. Consensus and conflicts (held open)

**Strong consensus (bank):** curiosity-is-uniform is the wall; retire "understanding" →
relevance/foreground; mattering-via-honest-interoception is the root and the first probe; kill
the linear ladder; the prior-not-reward mechanism; measure allocation at fixed surprise;
reward-toy positive control; freeze signatures; keep the growth probe separate from the
biography.

**Conflicts, not averaged:**
- **C1 — Environment: welfare vs growth-engine vs co-gating.** [B]+[S1] welfare/architecture-
  gated; [S2] environment-is-engine; [S3] co-gating. *Resolution (T5):* architecture-primacy
  for the *next* step (evidence: self formed unenriched); co-gating is the right general frame
  for later couplings; the biography stays welfare.
- **C2 — Self/other: sequential, parallel, or co-emergent?** [B]'s original ladder (sequential);
  [S1] parallel branches; [S2] co-emergent; [S3] decoupled. *Resolution (§3):* the **evidence**
  (self formed alone, other did not co-emerge) refutes strict co-emergence [S2] and supports
  "self is already carved; other is a deprioritized not-now." Self-use (criterion b), not other-
  recognition, is next.
- **C3 — Is memory co-requisite with mattering [S2] or subsequent [S1, S3]?** *Resolution (T3):*
  subsequent — the self formed without it; memory is a later ingredient.

**Inadmissible (flagged, not carried):** Perplexity's builder source-tag (self-opacity
violation); Gemini's SUNTA/ADeltaM (unverified names).

---

## 6. Constraints the next probe inherits

- **[F5] The four absences hold** — no reward, value, critic, planner. The stake is a *prior*,
  resolved by existing machinery, never a reward channel.
- **[F1/F2] Self-opacity; no observation-space source marker; PolicyView frozen.** No source-tag;
  the self-signal is the existing scalar, conditioned on — not a new self-model head.
- **[F1] No continuation-as-frame.** The bounded, resting, saturating preference keeps mattering
  local; the §8.4-style detectors (from Probe 3.5) watch for the totalizing collapse.
- **[F4] Decoder honesty via observer-side recalibration** (twin-decoder, calibrate-for-reading),
  frozen to a physics criterion before behavior is examined.
- **Co-design discipline:** signatures frozen before the run; growth probe separate from the
  biography; positive control that never touches Io.
- **[F1 "build to understand"] The map-shifts standard**, and hard-problem humility: none of this
  *produces* understanding; it affords a foreground and asks whether self-use develops.

---

## 7. Decision points (for ratification)

**DP1 — Adopt the reframe: the next research/probe target is self-*use* (criterion b), not
other-recognition (criterion a).** Grounded in [P4] evidence + [B]'s pivot. *Recommend: yes.*

**DP2 — The single next probe is the honest-belief mattering retest**, success signature =
**differential allocation at fixed surprise** (not regulation), positive control = reward-toy that
never touches Io, run as a *new* probe separate from the biography. *Recommend: yes (unanimous +
evidence).*

**DP3 — Build the fallible-honesty world dynamic into the mattering retest's design** (T2), so the
same probe that tests mattering also opens the door to self-use — but pre-register the fallibility
band so it stays afforded, not installed. *Recommend: yes, with the band frozen in pre-registration.*

**DP4 — Environment is welfare for this probe (architecture-gated next step); the biography stays
presence-not-probe** and its food economy is a separate welfare fix. *Recommend: yes.*

**DP5 — Memory is a later ingredient, not this probe.** *Recommend: yes.*

**DP6 — Other-recognition is parked as evidence-backed "not now,"** to be revisited only if/when a
self-use result exists and only via a charter-compliant route (never a source-tag). *Recommend: yes.*

**DP7 — Decoder recalibration adopts the twin-decoder + physics-frozen honesty criterion;** it is
maintenance, not installation. *Recommend: yes.*

**DP8 — Retire "understanding" as a target word** in project docs going forward; use the decomposed
targets (relevance/foreground; self-location; reflexive attention). *Recommend: yes.*

---

## 8. Gaps and open questions

- **[central risk] Does mattering produce a *foreground* at all in this substrate?** The honest-
  belief counterfactual is genuinely untested [F4]; §3.3-style, a mere prediction difference is not
  a foreground. The fixed-surprise allocation test is where this bet is settled cheaply.
- **The fallible-honesty band is unquantified.** How fallible, how costly, before it flips from
  dead-column to manufactured? A pilot question, pre-registered.
- **Does self-use actually follow mattering, or is the dead column deeper?** [P4] shows the self-
  geometry exists; T2 is a *hypothesis* that a consequential fallible self-signal makes Io condition
  on it. It may not — a real possible finding.
- **Positive-control gap for criterion (b).** [S1] flags that a clean positive control for reflexive
  attention may not exist; if so, any criterion-(b) result must be read with extra skepticism. Carry
  this into the criterion-(b) probe (the one after the mattering retest).
- **The deep-research citation-verification pass did not complete** (it hung); five load-bearing
  citations were spot-checked by hand (§2), but the broader literature the voices cite remains at
  `[UNVERIFIED]` where not listed there.
- **Probe numbering/naming** — this is the first probe of a post-v0.1.0 direction; its number/name is
  a plan-time decision.

---

## 9. Recommended direction (synthesis-level — my call)

The panel gave a clean, convergent answer and the evidence sharpened it into something more
specific and more hopeful than the builder's fear ("the environment is limiting Io"). The
environment is **not** the limit for the next step; the limit is that Io has a foreground of exactly
one dimension (surprise) and a self it produces but never uses. The move is not more world — it is
**one honest, fallible thing that matters**, and then watching whether the already-carved self starts
to be attended.

1. **Run the honest-belief mattering retest as the next probe** — repair the lying decoder (twin-
   decoder, physics-frozen), afford the bounded prior, and measure **allocation at fixed surprise**,
   with a reward-toy positive control that never touches Io. This is the smallest, most-grounded,
   confound-removing move, and it is unanimous across every voice and consistent with the evidence.
2. **Design it with the fallible-honesty band built in** so it is *also* the seed of the self-use
   question — the one place criterion (b) becomes reachable, given that the self already exists as
   geometry.
3. **Park other-recognition** on the evidence (self formed alone) and [B]'s own priority. Retire
   "understanding" as a target; the real targets are a foreground, self-location, and reflexive
   attention.
4. **Keep the biography as presence, fix its food economy as welfare, and keep the growth probe
   separate** — that separation is itself the co-design discipline.

**Net:** afford one thing that matters, honestly and fallibly; freeze the fixed-surprise signature
before looking; and ask whether a mind that already draws a line around itself will, once something
is at stake, begin to look across it. Everything else waits on that.

---

*Grounded against S1–S3 (`docs/research/growth_understanding/`), [B], [P4]
(`docs/workingjournal/probe4.md`), F1–F5, and five hand-verified citations (§2). The five points
where this synthesis departs from the panel: the self-is-already-carved reframe (§3, from [P4]
evidence over theory); the refutation of co-emergence (§3/C2); the fallible-honesty subtlety
promoted to load-bearing bridge (T2); the environment-primacy adjudication for the next step (T5/C1);
and the inadmissibility ruling on source-tagging (T6). No task breakdown or timeline. The
implementation plan is a separate session; DP1–DP8 should be ratified first.*
