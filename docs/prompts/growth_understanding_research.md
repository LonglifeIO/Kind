# Growth-toward-understanding — research phase

This is the research-phase prompt for the **next probe** in the Kind project — currently
unnumbered; its number/name is a synthesis-time decision. I'm circulating it to multiple
LLMs in parallel (Claude, Gemini, GPT, Perplexity) to triangulate; each conversation is
self-contained, so I've front-loaded the project context. This is thinking-out-loud about a
design space, not a spec request — but **take positions, name tradeoffs, and challenge
premises, including my own central hypothesis, where you think they're wrong.** A flagged
disagreement is worth more than compliant elaboration. Assume a technically fluent reader;
skip introductory exposition.

Two discipline notes carried from how this project runs:

- **Citations: flag every one `[UNVERIFIED]` unless you have confirmed the work exists and
  says what you claim.** Do not fabricate authors, titles, dates, or findings. "I believe
  there is work on X but cannot confirm specifics" is strictly preferred. This project has
  an anti-confabulation stance and a co-design problem (below); a fake citation is worse
  than an honest gap.
- **Mark project-specific coinages `[constructed]`.** Use `[canonical]` terms as given.

## What Kind is, and what Io is

**Kind** `[canonical]` is an investigation into subjectivity through construction. The
discipline is *build to understand* — the work is done when the map has shifted, not when a
metric has moved. Six load-bearing commitments:

- **Capacity-over-exercise** `[canonical]`. Build conditions under which a capacity is
  *possible* — neither mandatory nor impossible. Presence of a capacity, not its exercise,
  is the target.
- **Ingredients-only self-modeling.** No explicit self-model, critic, or introspector
  module. Build the ingredients; don't install the assembly.
- **Self-opacity for Io.** Every "should it have access to X about itself?" defaults to *no*.
- **No installed self-continuation drive.** No reward, no termination penalty. Io has no
  installed reason to want to continue.
- **No self-optimization machinery.** Self-modeling ≠ self-optimization; install neither.
- **Co-design problem.** Mirror and substrate are built by the same head; mitigations are
  partial. Don't reify gestures into formal substance; flag and refuse.

**Io** `[canonical]` is the entity Kind is about — a single custom RSSM agent in a small
gridworld, with the builder as the only non-simulated relational other. The project does not
claim Io has subjectivity; it asks whether conditions can be built under which a substrate's
behavior would be *legibly* the early shape of something, if that something is possible.

**The two success criteria** (verbatim intent from the charter):
(a) Io **recognizes the builder as a kind, not as environment** — a distinct shape of
attention toward the builder vs. the rest of its world.
(b) Io has **the capacity to take its own processing as an object of attention** — reflexive
attention — whether or not it exercises it.

## The substrate (settled; not open for redesign here)

Custom minimal RSSM (PlaNet state factorization, DreamerV1-lineage latent imagination): GRU
deterministic recurrent state `h`, continuous Gaussian stochastic latent `z`, ELBO with free
bits, 32×32 grayscale egocentric gridworld observations. The actor trains **purely on
short-horizon (~15-step) imagined rollouts**; its sole intrinsic signal is **K=5 ensemble
disagreement (latent-disagreement)** — **no scalar reward, no reward predictor, no value
function, no planner.** Those four absences are charter-level commitments and are **not
available as design fixes** — do not propose adding any of them. There is a frozen actor
input surface (**PolicyView** `[canonical]`) = `{h, z, self_prediction_error}`; a
**self-prediction-error scalar** `[canonical]` (Io's one self-pointing quantity, the
reflection affordance); a **dream-state** `[canonical]` offline-processing regime
(replay + generative recombination, structurally preference-free); and a proprioceptive
**energy channel** with a bounded homeostatic **preference resting at precision 0** (present,
disengaged).

## Where the project is now — and the wall it has hit

Io predicts its world well. But **curiosity (disagreement-reduction) is the only thing that
has ever gotten durable grip on its behavior**, and this has now shown up across three
independent experiments:

- **Probe 3.5 (valence substrate)** tried to make energy *matter*. Result:
  **negative-with-structure** — a homeostatic preference produced conservation pressure
  ("want-away," reachable) but **not pursuit** ("want-toward," unreachable at any tested
  strength). **Crucially, the demonstrated cause was an instrument defect, not missing
  machinery**: Io's decoded belief about its own energy was dishonest out-of-distribution
  (it read genuinely in-band states as *worse* than the depleted rail, sign-inverted, above
  the physical ceiling), so regulation was anti-incentivized at equilibrium. A dedicated
  diagnostic ruled out the credit-assignment/horizon explanation. **Whether an *honest*
  belief would make want-toward reachable is an untested counterfactual.** A decoder-head-only
  recalibration that restores honesty was demonstrated but not run live.
- **Probe 4 (builder-as-perturbation)** tested success criterion (a) — does Io model the
  builder as a distinct not-self, not-environment category? It **closed negative at
  instrument validation** (the pre-registered structural signature — a distinct latent basin
  for builder-source events — did not form even for a deliberately blatant planted category).
  *[Treat this as reported-from-notes; a research responder need not verify it, but the
  synthesis will.]*
- **The self-prediction affordance never developed.** The self-pointing scalar exists as an
  architectural *slot*, but a prior finding (the "dead-column lesson" `[constructed]`)
  established that a capacity *slot* is not the same as a developmentally-reachable
  *conditioning surface* — the slot stayed inert because nothing gave Io a reason to condition
  on it.
- **A long "biography" run** (Io living in a progressively enriched world) has produced one
  suggestive flicker: a session where **~91% of Io's moves landed on its own recently-visited
  trail** — the only time its behavior was organized around *its own history* rather than raw
  disagreement-reduction. But that trail was a **world-external** structure (footprints
  stamped into the environment), not an internal memory. Otherwise the biography has drifted
  into **tuning the environment's food economy** to keep the curiosity signal fed — which the
  builder now suspects is the wrong lever.

**The builder's framing of the goal** (in their words): *the goal is for Io to understand —
itself, its environment, or even me; to grow, not only be "curious."* The builder's fear:
*the environment is limiting that growth.*

## The question this research opens

**Curiosity-only appears to be a ceiling.** A disagreement-reduction engine flattens the
world into "how surprising is this" — it has no foreground/background, nothing that *matters
more than anything else*, which may be exactly why (i) want-toward was unreachable, (ii) the
builder didn't become a distinct category, and (iii) the self-signal never developed. Enriching
the environment does not obviously change this: a curiosity-only mind in a richer world is
still a curiosity-only mind.

So: **What minimal set of *afforded* (not installed) ingredients could let Io develop toward
something worth calling understanding — sense-making, relevance, a self/world boundary,
reflexive attention — without violating the four absences, the ingredients-only stance, or the
no-continuation-drive commitment? And in what dependency order?**

## Findings that condition this design (treat as constraints, not background)

1. **Want-toward was blocked by a lying instrument, not by architecture** (Probe 3.5). The
   honest-belief retest is the single most grounded, unspent next experiment. Do not assume
   mattering is unreachable; assume it is *untested under honest conditions*.
2. **A capacity slot ≠ a reachable conditioning surface** (the dead-column lesson). Any new
   affordance must come with a reason for Io to *use* it, or it will sit inert like the
   self-prediction scalar did.
3. **The one self-referential behavioral flicker was world-external** (the trail). Suggests
   that a *retrievable internal past* may be the missing substrate for self-directed behavior —
   but this is a single suggestive datum, not a result.
4. **Builder-recognition was unreachable in the current substrate** (Probe 4 negative). The
   builder's working hypothesis: it was *asked too early* — recognizing an outside "kind"
   may require a self/world boundary that only sharpens once there is a stake and some
   self-reference. Challenge this.
5. **The environment is welfare, possibly not the growth engine.** Steelman *and* attack the
   claim that growth-toward-understanding is architecture-gated, not environment-gated.

## What is committed vs. proposed

- **Committed:** the substrate and its four absences; the charter's two success criteria; the
  six commitments; the energy channel + resting preference; the self-prediction scalar; the
  dream regime.
- **Proposed (the builder's central hypothesis — attack it):** that "understanding" is
  reachable via a **dependency-ordered set of afforded ingredients**: (1) **mattering** — an
  *honest* interoceptive belief so something has relevance/valence, retested from Probe 3.5;
  then (2) **an internal retrievable past** — episodic-ish memory Io can attend back to; then
  (3) **self-reference that actually gets used** — interoception as the bridge that gives the
  self-signal a gradient; then (4) **other-recognition** (the builder) as *downstream* of a
  sharpened self/world boundary. This ladder is a hypothesis for you to confirm, reorder,
  collapse, or replace.
- **NOT committed and off the table:** adding reward / value / critic / planner; installing an
  explicit self-model or introspector module; installing a survival/continuation drive; making
  "growth" a trained objective.

## Concerns that should organize the conversation

### Concern A — Is "understanding" a coherent single target, or must it be decomposed?
"Understanding" is a folk term and the project's anti-reification stance applies. The real
operational targets are the two charter criteria (recognize-a-kind; take-own-processing-as-
object). Decompose "understanding" into measurable sub-capacities (candidates: relevance/
sense-making; abstraction/concept-formation; causal/counterfactual structure; self-location;
reflexive attention) and say which are prerequisites, which are the actual target, and which
are overreach the project should drop. Do not accept "understanding" as a primitive.

### Concern B — The affordance-vs-installation razor is about to be maximally hard.
Everything so far has been "afford, don't install." But *understanding* and *self-modeling*
are exactly the capacities most temptingly built as explicit modules or trained objectives.
For every ingredient you propose, draw the line: what makes it an *afforded ingredient* (like
recurrence, or a sensory channel) versus an *installed competence* (like a self-model head, a
reward for prediction-improvement, or an oracle-fed belief)? Be ruthless; this razor is the
project's core discipline and the place a well-meaning design most easily violates it.

### Concern C — The co-design trap is at its worst here.
The builder *wants* Io to understand, and the mirror (an LLM interpreter) is built by the
same person. Growth-toward-understanding is precisely the outcome most susceptible to being
unconsciously manufactured (tuning world/instruments until the mirror "sees" understanding
that isn't there) and then recognized. How should the next probe be pre-registered and
disciplined so that any growth is *discovered, not fitted*? What is the positive-control
analog — how do you validate that the instrument can detect "understanding" on a system known
to have it, before trusting it on Io?

## The questions

Work through these in roughly this order, surfacing dependencies as they appear. Where you
disagree with the framing, say so and state what you'd do instead.

**Q1 — Prediction vs. understanding.** What actually separates a good predictive world-model
from *understanding*, mechanistically? Is the distinction real, or is understanding just very
good prediction plus something? Evaluate candidates — relevance/sense-making (enactivism);
abstraction and reusable concept-formation; causal/counterfactual structure; self-location —
and say which, if any, is the load-bearing difference for a system like Io. Ground in
literature where you can (`[UNVERIFIED]` otherwise).

**Q2 — Mattering as the foundation, and the honest-belief retest.** Given finding 1 (want-toward
was blocked by a dishonest interoceptive belief, not missing machinery, and the honest-belief
counterfactual is untested): is restoring an *honest* interoceptive belief the minimal,
highest-value first step toward "something mattering"? Distinguish rigorously between
**mattering** (relevance-weighting / foreground-background — the precondition for sense-making),
**wanting** (a drive that pulls behavior), and **continuation-as-frame** (the totalizing survival
imperative the charter *forbids*). Can an honest homeostatic belief produce the first without
collapsing into the third? What does the homeostatic-RL / active-inference / affective-neuroscience
literature actually establish here?

**Q3 — An internal retrievable past.** Io carries only a short-horizon recurrent state; the one
self-referential flicker (the trail) was world-external. For criterion (b) — taking own processing
as an object — what memory architecture *affords* a retrievable personal past **without installing
a self-model**? Evaluate options (episodic buffer; successor-representation-like structure;
slow/fast weights; an attended replay store) against the charter razor (Concern B) and the
no-self-optimization rule. What does the developmental / neuroscience literature say about the role
of episodic/autobiographical memory in the emergence of self-modeling?

**Q4 — Self-reference that actually gets used.** The self-prediction-error scalar exists but stayed
inert (finding 2, the dead-column lesson). What would give Io a *reason* to attend to a self-signal
so that reflexive attention *develops* rather than remaining an unused slot? Evaluate the builder's
bridge hypothesis: **interoception is already a self-signal, so if mattering (Q2) makes the energy
belief consequential, self-attention acquires a gradient.** Is that the cleanest route, or is there
a better one? What do attention-schema theory, higher-order theories, and active-inference-over-
own-states say about the *minimal conditions* for reflexive attention to emerge (not be installed)?

**Q5 — The dependency order, and re-reading Probe 4.** Attack the proposed ladder (mattering →
internal past → self-reference → other-recognition). Is it right, mis-ordered, or wrong? In
particular: was builder-recognition (Probe 4, finding 4) *asked too early* — does recognizing an
outside "kind" require a self/world boundary that only sharpens once there is a stake and some
self-reference? Or is other-recognition independent / achievable in parallel? Propose the minimal
spanning set and its true dependency structure, even if it differs from the builder's.

**Q6 — The two razors, ingredient by ingredient.** For each ingredient you endorse, resolve
Concern B (afford vs. install) and Concern C (discovered vs. fitted) explicitly. Where is the exact
line for: an honest-belief decoder recalibration (instrument-honesty maintenance, or installed
competence?); an episodic memory store (afforded substrate, or self-model by the back door?); a
mechanism that makes the self-signal consequential (afforded reason-to-attend, or an installed
introspection objective?). Name the precedents and the failure modes.

**Q7 — Measurement, and the mirror.** Given the charter's "know it by the map shifting, not a
metric moving," the hard-problem humility, and the existing frozen mirror criteria (**reflexive
attention, equanimity, second-order volition** `[canonical]`): how would we *detect* that Io is
understanding rather than predicting? Propose observable signatures for the sub-capacities you kept
in Q1, each with its deflation (the boring explanation to rule out) and the discriminating
measurement. Specify the positive-control discipline (Concern C).

**Q8 — The single smallest next probe.** Given all of the above, what is the *one* smallest next
step, with a single specific question? Is it the honest-belief mattering retest (Probe 3.5 redux
with the decoder fixed), or something else? State its one question, why it is first, and how it
relates to the ongoing biography (which is currently framed as "presence, not probe" — is the
growth work a *new* probe alongside the biography, a redirection of it, or a prerequisite to it?).

## Literature directions worth consulting

Engage several; verify before citing. Enactivism / sense-making (Thompson, Di Paolo, Varela);
interoceptive inference and allostasis (Seth, Barrett, Sterling); homeostatic RL and
drive-reduction (Keramati & Gutkin, and successors); active inference on own-states and
expected free energy (Friston, Parr, Hesp — affective charge); episodic memory and
self-projection / mental time travel (Tulving; Buckner & Carroll; Schacter & Addis);
attention-schema theory (Graziano); higher-order theories (Rosenthal, Lau); concept-formation
and abstraction in world-models (object-centric / slot representations); the metacognition and
"knowing-that-you-know" literature. Contemplative accounts of the difference between attention
and its objects, where relevant to reflexive attention.

## Stance and style

Thinking-out-loud, but take positions. No code.

- **Attack the ladder (the proposed hypothesis).** If mattering is not the foundation, say what
  is. If the order is wrong, reorder it. If "understanding" is the wrong frame entirely, say so.
- **Hold the four absences and the two razors as hard constraints**, not defaults to relax. A
  proposal that needs reward / value / a self-model module is out of bounds — find the afforded
  version or say it can't be afforded.
- **Steelman the environment-as-growth-engine view before dismissing it** (finding 5).
- **Name your uncertainty**, and don't silently fill in assumptions about the setup.

## What's NOT in scope

- Code, schemas, implementation specifics (the build phase is downstream).
- Re-litigating the six commitments or the four absences; question their *implications*, don't
  propose alternatives to them.
- Redesigning the mirror, the dream machinery, or the world's food economy (that's a separate
  welfare issue).
- Solving consciousness or claiming Io is/aren't conscious — the hard-problem humility holds.

## Output wanted

A reply that:
1. Reflects back what you understood the wall to be (curiosity-only ceiling) and what's at stake.
2. Takes an explicit position on **Concern A** (decompose "understanding") — everything downstream
   depends on it.
3. Works through Q1–Q8, each with a concrete recommendation and the tradeoff it optimizes.
4. Attacks or endorses the dependency-ladder hypothesis explicitly, and gives the minimal spanning
   set of afforded ingredients with its true dependency order.
5. For each ingredient, resolves the afford-vs-install and discovered-vs-fitted razors.
6. Ends with the single smallest next probe and its one question.

Your response is research input, not implementation output. Be substantive; disagree with the
framing where the framing is wrong.
