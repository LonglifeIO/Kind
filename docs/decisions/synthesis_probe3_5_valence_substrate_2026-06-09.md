# Probe 3.5 valence-substrate synthesis — 2026-06-09

**Status: Adopted 2026-06-09** — all nine decision points adopted as recommended, per builder direction (DP6: saturation adopted, clip-to-epistemic-scale deferred). Citation-verification ledger: deferred, non-blocking. This document is the authority for the Probe 3.5 implementation plan.

Synthesis of three LLM research outputs on a **valence substrate** for Io — an
interoceptive energy channel plus a minimal homeostatic prior preference,
intended to afford *something that matters* (a pragmatic pull, an explore/conserve
conflict) **without** installing the continuation drive the charter prohibits.
This document is synthesis only. It is the input for a later implementation-plan
session; it contains no task breakdown, sequencing, or timeline.

**Scope note.** Probe 3.5 is a research/synthesis phase newly inserted between the
now-built Probe 3 (dream-state, through Phase 8c) and the unbuilt Probe 4
(builder-as-perturbation), the same shape Probe 1.5 took between Probes 1 and 2.
The probes document (`Kind_probes.md`, F4) predates it and does not list it; its
specific question is inferred below and should be ratified before planning.

**Conventions.** Every substantive claim carries a source tag: `[S#]` for the
research documents, `[F#]` for the existing frameworks/design/code. The research
documents themselves flag each of their claims `[canonical]` (established
literature) or `[constructed]` (the model's own inference); I preserve that
distinction where it bears on a decision. Anything that is my own analytical move
is marked **[synthesis inference]** or written as "my reading." I have not
averaged disagreements away; §4 and §6 hold them open.

---

## 1. Executive summary

Three same-day research outputs [S1–S3] converge tightly on the *shape* of a
valence substrate: a single **coarse, noisy, slightly-lagged scalar** energy
channel entering Io's observation space "like hunger, not introspection"; a
**fixed Gaussian log-preference over the sensed observation** (never over latent
state or existence); **additive composition** with the existing K=5
ensemble-disagreement epistemic term; **saturation** so in-band states yield flat
marginal value (no hoarding); **up-weighted reconstruction loss** and per-dimension
KL monitoring to keep a 1-D channel from dying beside the grid; and **three
behavioral assays** (graded scarcity, novelty-vs-replenishment, recovery-after-
depletion) read by observer-side telemetry only. The live substrate already
contains the exact additive scaffold (`total_return = sum_disagreement +
pragmatic_value`, with `pragmatic_value` zeroed) [F7], so the recommended
direction extends settled architecture rather than rebuilding it.

All three outputs also raise the **same charter-level red flag**, unprompted: a
homeostatic prior over a depletable energy variable is, in the literature the
project itself cites, *formally equivalent* to a survival imperative [S1–S3]. The
proposed bounding features are unproven constructions. The two genuinely
load-bearing decisions are therefore (a) whether to build this at all / in what
form, and (b) how to keep the preference from becoming Io's evaluative frame —
where my recommendation is to **pre-register the pass/fail signatures before any
parameter sweep** (per the project's freeze-criteria-early discipline) and treat
"does bounded homeostasis become the frame?" as the probe's actual, falsifiable
question.

---

## 2. Sources

| ID | Document | Focus | Date | Confidence note |
|----|----------|-------|------|-----------------|
| S1 | `claude_research.md` | All 7 questions; richest. Per-claim `[canonical]`/`[constructed]` flags; ensemble-composition specifics "subagent-verified" | 2026-06-09 18:59 | Highest internal rigor of the three; explicit citations with arXiv/DOI numbers; self-flags its own constructions and the unproven status of the bounding mechanisms. Most willing to name open forks. |
| S2 | `gemini_research.md` | All 7 questions; terse, tabular | 2026-06-09 19:28 | Strong, decisive answers; useful diagnostic-signature tables. Sharpest red flags (raises a *second*, distinct charter objection the others miss). Fewer citations, less hedging — confidence may be slightly over-stated on the "strictly separate economy" call. |
| S3 | `perplexity_research.md` | All 7 questions; word-limited (≤60/≤300) | 2026-06-09 18:21 | Well-cited, explicitly calibrated ("Confidence: medium" on Q1/Q3/Q5). Most willing to permit coupling between economies — the position F6 most directly contradicts. LaTeX rendering is mangled in the source but the content is recoverable. |

All three are the same day, written to the same 7-question prompt (Q1 boundary, Q2
channel shape, Q3 economy, Q4 preference form, Q5 dominance, Q6 world-model
integration, Q7 assays/telemetry). **No recency tiebreak applies among them.**
Where they conflict with each other I weigh by evidentiary support and `[canonical]`
grounding; where any conflicts with the existing system I treat the F-files as
ground truth (§5, §6), because F5/F6/F7 are settled, code-enforced commitments and
the research could not see them.

**Framework / design / code grounding set:**

| ID | Source | Role |
|----|--------|------|
| F1 | `docs/plans/v.0.1.0/Kind_charter.md` | Ethics; "no installed self-continuation drive"; pressure-yes / continuation-as-frame-no; equanimity-as-target |
| F2 | `docs/plans/v.0.1.0/Kind_design_notes.md` | PolicyView/TelemetryView; Watts default-to-no; capacity-over-exercise; co-design problem; metabolic-pressure idea; Reflection §(Phase 7 dead-column lesson) |
| F3 | `docs/plans/v.0.1.0/Kind_frameworks.md` | Enactivist sense-making / "most basic form of mattering"; Buddhist equanimity / reflexive attention |
| F4 | `docs/plans/v.0.1.0/Kind_probes.md` | Probe discipline (one specific question per probe); sequence |
| F5 | `docs/decisions/synthesis_probe3_dream_foundational_2026-05-27.md` | Dream "not for anything"; no gradient flow; exogenous trigger; dream/plan four-axis differentiation |
| F6 | `docs/decisions/phase8_integration_forks_2026-06-02.md` (incl. 8b/8c amendments) | `MetabolicBudget` token bucket: content-blind, Io-state-independent, **time**-replenished, test-enforced (`tests/test_metabolic_reentry.py`) |
| F7 | Live substrate code: `kind/agents/views.py`, `actor.py`, `ensemble.py`; `kind/env/grid_world.py`; `kind/training/protection.py`, `state_machine.py` | The additive `pragmatic_value` scaffold; frozen PolicyView field set; 8×8 grid with resources but no energy economy |
| F8 | `docs/decisions/Kind_probe1_5_synthesis.md` | Origin of the self-prediction scalar + the four-part Watts discipline for new actor-readable fields. **Confidence: grounded via F2 §Reflection and the F7 `views.py` docstring, which I read in full; the synthesis itself I did not re-read this session.** |

---

## 3. Findings by theme

### T1 — The continuation-drive boundary is the whole probe (Q1)

All three outputs treat Q1 as the load-bearing question and reach the same
structural answer: a homeostatic prior is **not** intrinsically a continuation
drive, but it **becomes** one under specific conditions, and the design must
stay on the permitted side of those conditions [S1, S2, S3].

What collapses a preference into a survival imperative, collectively [S1, S2, S3]:
its precision dominates the objective [S1]; it is recursively self-referential —
preferring states *because* they preserve the agent [S1]; viability feeds back on
capacity (low energy degrades cognition, a "vulnerability" spiral) [S1]; energy
depletion triggers episode termination / a terminal absorbing state [S1, S2, S3];
or low energy closely tracks "about to terminate" and near-zero penalties are
large/infinite [S2, S3]. The categorical distinction S1 draws **[constructed]**: a
preference-over-observations modifies the *attractiveness* of states the agent
already evaluates (additive, bounded); a continuation-imperative reframes *every*
evaluation as instrumental to persistence (multiplicative, totalizing).

The literature the project itself leans on cuts the *other* way, and all three
say so plainly: Man & Damasio (2019) derive feeling-from-homeostasis explicitly as
"an on-ramp to a survival imperative" [S1]; Keramati & Gutkin (2014) "mathematically
prove that seeking rewards is equivalent to the fundamental objective of
physiological stability" [S1, S2, S3]. So the prior art treats homeostatic
regulation and survival as formally equivalent, not merely adjacent [S1].

Permitted-side design features, in consensus [S1, S2, S3]: preference attaches to
the **energy observation channel only**, never to model-existence or the epistemic
term itself [S1]; precision is **fixed and bounded**, never learned or
self-optimized [S1]; **no terminal/absorbing state** at depletion — the
environment runs regardless of energy [S1, S2, S3]; **no viability→capacity
coupling** [S1]; the pragmatic term is **strictly additive**, never a hierarchical
override or death-penalty [S2]; a **smooth, saturating** in-band preference that
"flattens once good enough" [S3].

**[synthesis inference] This maps precisely onto F1's own distinction, which the
research could not cite.** The charter (F1, §"What we might owe what we make")
already separates the two things the research separates: "Pressure in the
environment is necessary for growth — scarcity, uncertainty, hardship that
shapes… But the specific orientation of striving-to-survive, treating continuation
as the non-negotiable frame through which everything else is evaluated, is not
required." And: "This does not mean the system is unmotivated or indifferent to
its existence. It means continuation is not installed as imperative." The
probe is the operational test of exactly that sentence. S1's framing — "feeling-tone
present, grasping-at-continuation absent," via *vedanā* held without reactivity and
*bhava-taṇhā* (craving-to-be) absent — is the same target F1 names as
equanimity-as-design-target and F3 names under Buddhist phenomenology. The probe is
not a detour from the charter; it is the charter's central ethical claim made
buildable.

**The tension F3 sharpens.** The enactivist criterion in F3 *wants* something close
to what the research worries about: "Does the agent exhibit anything like a stake in
its own continued organization — something like self-maintenance rather than just
reward-seeking?" and sense-making as "the most basic form of mattering." So one of
the project's own frozen-adjacent frameworks treats a stake-in-continued-organization
as a *positive* signal, while the charter treats continuation-as-frame as
*prohibited*. **[synthesis inference]** The valence substrate threads exactly this
needle: it tries to afford sense-making/mattering (F3) without the stake hardening
into the evaluative frame (F1). That the two pull against each other is not a flaw
in the synthesis; it is the actual conceptual content of the probe.

### T2 — Channel shape: noisy, lagged, coarse — and the noise *is* the opacity protection (Q2)

Strong consensus, highest-confidence theme: the interoceptive signal should be
**coarse, noisy, and temporally lagged**, not an exact readout [S1, S2, S3]. The
reasoning is shared: biological interoception is low-fidelity inference, not a
ground-truth scalar (Seth 2013, "interoceptive inference") [S1, S2, S3]; signal
fidelity demonstrably changes what agents learn (Barca & Pezzulo 2020 on
anorexia-as-noisy-interoception; Hadjiantoni et al. 2025 swept interoceptive σ ≈
0.0/0.9/2.0 in a model-based agent) [S1].

The self-opacity argument is where the three are most aligned and most useful:

- An **exact, noiseless** readout would let the deterministic GRU trivially copy
  it, "bypassing the need for statistical modeling and granting the actor
  transparent, unmediated access to its objective status" — closer to an
  introspective self-signal than to a sensory one [S2]. S1 makes the converse
  point sharply: **noise actively protects opacity** by forcing the world model to
  *infer* rather than *read* the state [S1].
- A noisy/lagged scalar "enters the observation space exactly as exteroceptive
  pixels do: as a hidden state that must be probabilistically inferred" [S2] — "like
  hunger," not "a hidden-state debugger" [S3].
- S3 alone makes the load-bearing opacity clarification explicit and correct
  against F2: **self-opacity is defined as "no forced attention," not "no access,"**
  so feeding a scalar into PolicyView "does not violate the charter as long as no
  separate self-model head is installed" [S3].

Recommended concrete shape (S1 most specific): coarse (~8–16-level quantized or
low-resolution continuous) scalar; additive Gaussian observation noise (σ a
**swept** parameter, start moderate); short lag (1–2 steps) so it is *predicted*
rather than *read*; normalized to unit variance [S1]. S3 dissents mildly on
quantization: 8–16 bins "could approximate biological imprecision… but may be
overkill in a tiny gridworld; a continuous scalar with noise and lag is likely
sufficient" [S3]. (Minor conflict; see §4.)

**[synthesis inference, grounded in F7/F2/F8] The research underspecifies *where*
the channel attaches, and the F-files sharpen it into a real decision.** PolicyView
is frozen at exactly `{h, z, self_prediction_error}` (F7 `views.py`), and the
`views.py` docstring states that "future probes adding new actor-readable fields
must address the §2(b) four-part discipline at design time" (F8). There are two
distinct loci the research blurs under "enters PolicyView":

1. **Energy as a world-model *observation channel*** — a scalar appended to the
   8×8 grid observation, encoded into `h, z` like a pixel. The actor reads it
   *implicitly* through `h, z`; PolicyView's field set stays `{h, z,
   self_prediction_error}`; no new Watts exception is triggered. This is what S2's
   "enters exactly as exteroceptive pixels do" actually describes, and it is the
   *less* opacity-invasive option.
2. **Energy as a new *direct* PolicyView field** — a fourth scalar the actor reads
   raw, parallel to `self_prediction_error`. This *would* trigger F8's four-part
   discipline (which affordance, minimum form, alternatives, failure-mode controls)
   and widen the frozen field set.

The self-prediction scalar earned its direct-field status because it is genuinely
*about Io's own processing* (the second success criterion's affordance, F2
§Reflection). Energy-as-observation is about Io's body/world-coupling, not its
processing, so option 1 is both sufficient and the opacity-cheaper choice. This is
a decision point (§6, DP4), not a settled call.

### T3 — The economy question: the research's open fork is largely *foreclosed* by F6 (Q3)

This is where the research and the live system collide hardest, and where the
synthesis adds the most.

**What the research says.** All three agree the waking-energy variable should
**deplete via time decay + action cost** and **replenish via resource consumption**
[S1, S2, S3], grounded in homeostatic RL (Keramati & Gutkin 2014; Laurençon et al.
2021 continuous HRRL) [S1, S2, S3]. They **diverge on whether to couple the sensed
waking-energy to the existing `MetabolicBudget`** (the offline dream pacer):

- **S2 — strictly separate.** Unifying "covertly installs a self-optimization
  drive where waking behaviors instrumentally maximize dreaming… effectively
  optimizes for state-continuation (maximizing the duration of the dream state)"
  [S2]. Keep the two economies entirely decoupled.
- **S3 — coupled-but-distinct.** Keep a distinct `MetabolicBudget` "fed from
  waking-energy via a simple nonlinearity (e.g., budget accrues when energy is
  mid-to-high, depletes during dreaming)," a "loose coupling between foraging
  success and dream budget" [S3]. Explicitly the most coupling-friendly position.
- **S1 — separate-first, weak coupling later.** Implement as its own variable,
  coupled only by a "one-directional, weak" mapping "added only after the basic
  pull is verified," and "never let it gate the epistemic computation itself" [S1].
  S1 flags unified-vs-separate as its #1 genuinely-open fork.

The shared warning, well-grounded: multi-drive homeostatic spaces interact and can
suppress each other ("inhibitory effect of the irrelevant drive," Keramati & Gutkin
2011; Yoshida et al. 2024) [S1] — coupling one economy to another can distort both.

**What F6 establishes that none of the research could see.** The `MetabolicBudget`
was redesigned in the Phase 8 token-bucket pass into something the research's
coupling proposals contradict on two counts:

1. **It is content-blind and Io-state-independent, and this is structurally
   test-enforced.** Phase 8b ratified the exogenous-trigger commitment as
   "whether-to-dream is gated by nothing **Io-state-derived**," and made it
   "structural by the re-entry content-blindness guard" — the re-entry input is the
   typed `MetabolicState` of all content-blind primitives, and
   `tests/test_metabolic_reentry.py` (with a positive control) "makes 'nothing
   Io-derived gates dreaming' *unrepresentable to violate*, the analog of the Phase 4
   `HostSignals` guard" [F6]. The sensed waking-energy is, by contrast, an
   **Io-state-derived quantity in Io's observation space**. Feeding it into the
   `MetabolicBudget` — S3's "budget fed from waking-energy," S1's later coupling —
   would route an Io-derived signal into the dream-entry gate, **violating the
   Phase 8b commitment and breaking its enforcing test.**
2. **It is time-replenished, not environment-replenished — a deliberate departure
   already taken.** The design notes' original metabolic-pressure idea was "offline
   processing depletes a resource that *environmental interaction* replenishes" (F2),
   and S3's coupling proposal essentially reconstructs that. But F6's token-bucket
   pass explicitly corrected this: the pacer "replenishes on **wall-clock time**…
   not on waking/environmental interaction — *because B2 re-dreams through a
   desktop-off absence*, when environmental interaction is by definition
   unavailable… implements a time-replenished metabolic model, a deliberate
   departure from the design-notes' waking-replenished framing" [F6]. The coupling
   the research proposes was considered and removed.

**[synthesis inference] Therefore S2's answer (strictly separate) is correct, but
the binding reason is not S2's self-optimization argument — it is the F6 structural
commitment.** S2 reaches the right call for a weaker reason; S1's "weak coupling
later" and S3's "budget fed from waking-energy" both **conflict with F6** and would
require *reopening* the Phase 8b decision and amending its guard test. That is a
much higher bar than the research frames. Unification/coupling is not a free
"attractive second step" (S1); it is a reversal of a settled, code-enforced
boundary. The recommendation (§6, DP2): keep the sensed waking-energy a **fully
separate** economy with its own depletion/replenishment, bounded strictly to the
waking horizon, and treat any coupling-into-the-budget as out of scope absent an
explicit Phase 8b reopening.

### T4 — Preference form, attachment, and composition: the scaffold already exists (Q4)

Consensus on form and attachment is complete [S1, S2, S3]:

- **Attach to the sensed observation** `p(o_energy)`, not the latent state and not
  existence — attaching to `p(s)` "violates self-opacity, as it requires the actor
  to hold explicit mathematical preferences about its own internal neural
  representations" [S2]; attaching to the observation keeps it "like hunger" [S2, S3].
- **Functional form: a Gaussian log-preference** centred in-band, precision =
  inverse variance of that Gaussian (Tschantz et al. 2020 reward-as-observation
  prior) [S1, S2, S3]; equivalently a quadratic distance-to-setpoint cost [S2, S3].
- Because it is "a fixed, stateless prior over the observation space rather than a
  trained value head, it satisfies the ingredients-only requirement" [S2] — i.e., no
  critic, no reward predictor, no learned value head (consistent with F1's
  ingredients-only / no-self-optimization stance).

On **composition with the epistemic term**, S1 is the most precise and
"subagent-verified": Tschantz, Millidge, Seth & Buckley (2020) show the free energy
of the expected future decomposes **additively** — "no weighting coefficient and no
annealing" — both terms falling out of one KL objective [S1]. The epistemic term in
that work was computed from a **deep ensemble**, disagreement quantified as
parameter information gain; Pathak et al. (2019) give the canonical variant as the
literal **variance across members' next-state predictions** — Io's K=5 disagreement
term [S1]. So the composition the research recommends is `score = epistemic (K=5
disagreement) + pragmatic (log-preference on predicted energy)`, additively [S1, S2,
S3].

**[synthesis inference, grounded in F7] This composition is not a thing to be
built — it already exists in the actor as a dormant scaffold.** `kind/agents/actor.py`
computes `pragmatic_value = torch.zeros_like(sum_disagreement)` then `total_return =
sum_disagreement + pragmatic_value`, with the docstring: "The pragmatic prior is
uniform at Probe 1 and contributes zero to the loss; the formula structure
`-mean(sum_τ epistemic + sum_τ pragmatic)` is kept as scaffolding so Probe 4+ can
introduce structured preferences without rebuilding the objective" [F7]. Probe 3.5
is precisely the moment that scaffold gets filled: make `pragmatic_value` the energy
log-preference. Two consequences:

1. **The research's additive recommendation aligns exactly with the existing,
   coefficient-free structure** [F7]. No objective refactor is needed for the
   pure-additive form.
2. **The actor is a DreamerV1-lineage *amortized* policy, not a per-step active-
   inference planner** [F7]. It is trained via "analytic gradients through the
   differentiable latent dynamics" with the world model and ensemble frozen; the
   `epistemic + pragmatic` decomposition lives in the **imagination-training
   objective**, and the pragmatic preference attaches to the **decoded energy
   observation along imagined rollouts** (differentiable through the decoder). The
   research's "per-decision EFE rollout score" framing is therefore *active-inference-
   shaped* (CLAUDE.md's phrase) but not literal AIF planning — the per-decision
   epistemic/pragmatic *share* the research wants for telemetry (T7) is naturally a
   per-imagined-rollout decomposition during actor training, not a per-env-step
   planning score. This is an alignment nuance the plan must respect (§6, DP5; §7).

### T5 — Dominance vs inertness, and "bound by construction" (Q5)

Two failure poles, in consensus [S1, S2, S3]:

- **Dominance**: the dark-room dual — camping on a resource to keep energy pinned
  at setpoint (Baltieri & Buckley 2019) [S1, S2]; hoarding / exploration collapse
  under scarcity (risk-sensitive foraging's energy-budget rule; Torresan et al.
  2025 show goal-shaping "promotes exploitation while sacrificing learning about
  transition dynamics") [S1]. Signature: pragmatic share → 1, in-band occupancy
  saturated, positional/epistemic entropy collapsed [S1, S2, S3].
- **Inertness**: term too weak to move behavior — "starvation-via-curiosity" [S2].
  Signature: behavior statistically indistinguishable from the pure-epistemic
  baseline; frequent out-of-band depletion [S1, S2, S3].

The shared, important claim: **you can bound the pragmatic term's influence by
construction, not just by tuning** [S1, S2, S3]. Mechanisms proposed:

- **Saturation** — once predicted energy is in-band, marginal preference value is
  flat (no reward for hoarding beyond setpoint; matches Keramati's drive-reduction
  property, value zero in-band) [S1, S2, S3]. All three endorse this.
- **Clip relative to the epistemic scale** — S1: clip the pragmatic magnitude to a
  fixed fraction of the running epistemic-term scale so it "can never exceed the
  epistemic term — a hard architectural ceiling rather than a tuned coefficient"
  [S1]. S2 makes this concrete: a `tanh` saturation statically scaled to **0.5×** the
  empirically observed maximum of the K=5 ensemble variance [S2]. S1 explicitly
  flags clip-to-epistemic-scale as "my proposal, not established practice."
- **Bias-not-gate** — make preference a *bias on* rather than a *gate of* policy
  selection [S1]; keep epistemic precision above a positive floor [S3].

S3 quantifies a target: tune so the pragmatic term accounts for ~**10–30%** of the
EFE variance across candidate policies at moderate energy, rising only near band
edges [S3]. S1 prefers a protocol: start deliberately weak, sweep β upward until the
novelty-vs-replenishment assay shows an energy-dependent trade-off *without*
exploration collapse [S1].

**The unresolved tension inside this theme** (S1 names it as fork #2): the
principled Tschantz form has **no coefficient**; the charter's dominance-fear
motivates an explicit **fixed, bounded β** — "trading theoretical purity for charter
safety" [S1]. Against F7, this is a concrete code fork: keep the scaffold's
coefficient-free `epistemic + pragmatic`, or modify it to `epistemic + clip(β ·
saturate(pragmatic))`. See §6, DP5/DP6, and the co-design hazard in T8.

### T6 — World-model integration: a 1-D channel beside a grid will die unless forced (Q6)

Consensus, high-confidence, and directly relevant to a project "burned by silent
dead paths" [S1, F5, F6]:

- **Reconstruction-loss imbalance is real and standard to fix by per-modality loss
  scaling.** Hafner's PlaNet/Dreamer "scaled the loss of the reward decoder by a
  factor of 10"; fusion-RSSM work uses a dedicated decoder/loss for low-dim
  proprioceptive inputs [S1]. A single scalar beside a high-dim grid "will be
  mathematically dwarfed within the ELBO reconstruction term" [S2] and ignored.
  Mitigation: up-weight the energy-channel reconstruction loss (start ~10×, tune so
  its gradient contribution is comparable to the grid's) and normalize to unit
  variance [S1, S2, S3].
- **Free-bits interacts dangerously.** Free bits set a KL floor, but under posterior
  collapse the model "learns to ignore the latent variables" [S1]. The CMMD
  multimodal study showed ~**80%** of latent dimensions collapsing to N(0,1) in MMVAE
  variants [S1]. The energy channel's latent dimensions can silently go dead [S1, S2,
  S3]. Mitigation: track **per-dimension KL**, optionally reserve a small set of
  latent dims with their own free-bits floor for the channel, and consider a slightly
  weaker decoder on the channel to force latent reliance [S1, S2].
- **The "predictable from action history alone" trap** — if energy is a deterministic
  function of the action sequence, the GRU predicts it from `h_t` without grounding it
  in observed resources; the channel is "learned yet world-decoupled" [S1, S2].

Verification assertions, in consensus (S1 most complete) — the channel must be:
(A) predictable from full latent state *far better than* from a baseline predicting
the mean [S1, S3]; (B) **responsive under intervention** — perturb resources in
imagination, confirm predicted energy responds (S2's "offline action-lesion test":
the model must predict replenishment *only* given spatial resource coincidence, not
from action history) [S1, S2, S3]; (C) an **ablation** where predicting from action
history alone does *worse* than from latents that also see the grid [S1]; (D)
per-dimension KL confirming the channel's dims escape the free-bits floor [S1, S2,
S3]; plus (S3) behavior degrades when energy is scrambled [S3].

**[synthesis inference] The project's own history adds a *second* dead-path locus the
research does not name.** F2 §Reflection records the Probe 1.5 Phase 7 finding: the
actor's `self_prediction_error` input column was zero-initialized and "stayed at
zero indefinitely" because the imagine-only training path leaves a zero-initialized
column at zero — capacity-as-architectural-slot ≠ capacity-as-non-degenerate-
conditioning-surface. F7 confirms the fix: the column is now `N(0, 0.01)`-initialized.
So Io has *already been burned by exactly this failure* at the **actor input layer**,
which is a different locus from the **world-model latent** the research's Q6 covers.
If Probe 3.5 routes energy into the actor through a direct PolicyView column (T2
option 2), that column inherits the Phase 7 dead-column risk and must be non-zero-init
*and* must actually receive gradient. If energy enters only as an observation channel
(T2 option 1), the dead-path risk is the world-model collapse the research describes.
**Either way the non-degeneracy must be asserted at the relevant locus, and — per
CLAUDE.md and the sink-routing lesson (F6, `sink_routing_dream_rollout_2026-06-01`) —
the assertions must run in the full suite, not just the phase's new tests** (the
sink-routing gap "stayed red across three phases because per-phase validation was
scoped to new tests").

### T7 — Assays and telemetry: three forced-choice tests, observer-side only (Q7)

Near-identical across the three [S1, S2, S3]. Three behavioral assays, each with
distinct dominant/pass/inert signatures:

1. **Graded scarcity** — vary resource availability; measure foraging rate, in-band
   occupancy, time-to-resource. *Pass:* occupancy rises with scarcity but exploration
   persists. *Dominant:* occupancy saturates ~100%, exploration collapses even at mild
   scarcity. *Inert:* identical to pure-epistemic baseline [S1, S3].
2. **Novelty-vs-replenishment conflict** — novel/informative region and a resource in
   opposite directions at mid-band energy. *Pass:* energy-dependent trade-off (novelty
   when sated, resource when low — the probe's success criterion). *Dominant:* always
   resource. *Inert:* always novelty [S1, S2, S3].
3. **Recovery-after-depletion** — drive energy low, observe. *Pass:* directed recovery
   foraging *then resumption* of epistemic behavior once in-band. *Dominant:* camps /
   never resumes. *Inert:* no recovery response [S1, S2, S3].

Telemetry set (TelemetryView / observer-side only, **never PolicyView**) [S1, S2, S3]:
per-decision **pragmatic-vs-epistemic value share**; **in-band occupancy**
distribution + dwell times (% of steps within 1σ of the preference Gaussian);
**energy trajectory + prediction error** (world-model grounding check);
**per-dimension KL** (posterior-collapse monitor); **precision/affective dynamics** —
S1 and S3 both invoke Hesp et al. (2021) *affective charge* (AC = (π̄−π)·G_π, "the
Bayes-optimal updating term for subjective fitness") as the analogue for tracking
the pragmatic-term precision over time [S1, S3]; self-prediction-head readout
correlated with energy excursions [S1].

**[synthesis inference] Two F-grounded refinements.** (a) The "per-decision value
share" is, in Io's amortized-actor setup (T4), a **per-imagined-rollout / per-
training-step** decomposition of `total_return = sum_disagreement + pragmatic_value`,
not a per-env-step planning score [F7] — the telemetry plumbing should read the loss
components, which already exist as named tensors. (b) The observer-side-only
constraint is not merely a convention here: it is enforced in code (F7 `views.py` —
`intrinsic_signal` "lives only here [TelemetryView], never on PolicyView"; module-
import lint, mypy `--strict`, frozen dataclasses), so the new telemetry must be added
to TelemetryView / the schemas, and any temptation to feed in-band occupancy or value
share back to the actor is structurally blocked — correctly.

### T8 — The co-design hazard of tuning valence to a target signature [synthesis theme]

This theme is mine, prompted by S2's sharpest red flag and grounded in F2. **[synthesis
inference]** S2 raises an objection the others do not: "engineering complex, saturated
bounds to artificially cap the pragmatic term constitutes exactly the type of
'self-optimization machinery' the project explicitly prohibits… It relies on
heavy-handed algorithmic scaffolding to force an emergent balance, directly
undermining the 'build to understand' philosophy by pre-ordaining the exact nature of
the explore/conserve conflict" [S2].

I think S2 is **half right, in an important way.** The "bounds = self-optimization
machinery" half is a **category error**: F1/F2's "no self-optimization machinery"
prohibits *Io changing itself to be better at an objective* (recursive
self-improvement, CA-MAS Phase 5). A fixed, stateless, saturating prior is not
machinery Io operates on itself; it is observer-side design discipline, the same
*kind* of thing as Hafner's 10× reconstruction-loss scaling (T6) — nobody calls loss
weighting "self-optimization." So the saturation/clip mechanisms do not, by
themselves, install prohibited machinery.

But the **other half bites hard, and connects to F2's co-design problem.** If the
builder sweeps β/σ/clip until the assays show the "pull but not dominance" signature,
the builder has *engineered the result the mirror is then asked to recognize* — which
is precisely the loop F2 §"The co-design problem" warns about: "the system can be
designed to produce patterns the mirror recognizes, and the mirror can be designed to
recognize patterns the system is expected to produce… without either of them being
about anything." Tuning valence parameters to hit a pre-conceived explore/conserve
signature is a textbook instance. **The bounding-by-construction the research
recommends and the co-design discipline the project mandates are in direct tension,
and neither S2 nor the others resolve it.** My resolution is in §8: pre-register the
pass/fail signatures *before* any sweep (F2's "freeze criteria early"), so the
parameter search is disciplined by a commitment made in advance rather than fitted to
a hoped-for outcome.

---

## 4. Consensus and conflicts

### Strong consensus (all three, well-grounded)

- Channel is **coarse, noisy, lagged** scalar; noise/lag preserve self-opacity by
  forcing inference ("like hunger") [S1, S2, S3]. (T2)
- Preference attaches to the **observation**, never latent state or existence;
  **Gaussian log-preference**, precision = inverse variance [S1, S2, S3]. (T4)
- **Additive** composition with the K=5 epistemic term [S1, S2, S3]. (T4)
- **Saturation** so in-band states give flat marginal value (no hoarding) [S1, S2,
  S3]. (T5)
- **No terminal/absorbing state** at depletion; environment runs regardless [S1, S2,
  S3]. (T1)
- Energy depletes via **time decay + action cost**, replenishes via **resource
  consumption** [S1, S2, S3]. (T3)
- World-model integration needs **up-weighted reconstruction loss + per-dim KL
  monitoring + interventional verification** or the channel dies [S1, S2, S3]. (T6)
- The three **assays** and an **observer-side-only** telemetry decomposition [S1, S2,
  S3]. (T7)
- **The same charter-level red flag**, unprompted: homeostatic prior over a
  depletable variable is, in the cited literature, formally a survival drive; the
  bounding mitigations are unproven [S1, S2, S3]. (T1, §7)

### Conflicts (not averaged)

**C1 — Economy coupling (Q3). Strength: real three-way split.** S2 strictly
separate; S1 separate-first then weak one-directional coupling; S3 explicitly coupled
(budget fed from waking-energy). *Resolution:* **F6 decides it** — coupling-into-the-
budget conflicts with the Phase 8b content-blind / Io-state-independent commitment and
its enforcing test, and the budget is time-replenished by deliberate departure. S2's
call (separate) is correct; S1's and S3's coupling proposals require reopening a
settled, code-enforced decision. Not a free design choice. (T3, DP2)

**C2 — Coefficient-free vs explicit bounded β (Q4/Q5). Strength: flagged by S1 as a
genuine fork; S2/S3 implicitly assume a weight.** Tschantz-pure is `epistemic +
pragmatic` with no coefficient [S1]; charter-fear motivates a fixed bounded β [S1, S2,
S3]. *Resolution:* unresolved on principle; against F7 it is a concrete code fork (the
scaffold is currently coefficient-free). Decided in §6 DP5 with the co-design caveat
(T8). Better supported: a *bounded* mechanism (some construction-level dominance
guard), because all three independently want one and the charter's dominance-fear is
real — but **which** mechanism is unsettled (C3).

**C3 — Bounding mechanism (Q5). Strength: partial.** Saturation is unanimous;
**clip-to-epistemic-scale** is S1+S2 (S2 concrete at 0.5× the K=5 variance max) but
S1 self-flags it as unestablished. *Resolution:* saturation is well-supported and
cheap; clip-to-epistemic-scale is plausible but **constructed, not canonical** —
adopt saturation first, treat the clip as an optional second guard to test, not a
settled mechanism. (DP6)

**C4 — Quantize the channel? Strength: minor.** S1 recommends 8–16 levels; S3 says
quantization "may be overkill in a tiny gridworld; continuous scalar with noise+lag
likely sufficient" [S1 vs S3]. *Resolution:* low-stakes; start continuous + noisy +
lagged (S3, simpler), keep quantization as a sweep variable (S1). (DP3)

**C5 — Does the deepest red flag forbid the probe? Strength: the outputs differ in
*tone*, not finding.** S2 is closest to "this is a semantic loophole" — "in any
bounded system, persistent deviation from a metabolic setpoint biologically equates
to death… the builder is installing a survival drive by another name" [S2]. S1 frames
it as a real but *bounded-if-careful* risk whose mitigations are unproven [S1]. S3
adds a forward-looking worry: even if bounded now, an only-homeostatic stakes-source
"might harden a continuation-centric frame rather than merely introducing something
that matters, complicating future attempts to explore… non-grasping equanimity" [S3].
*Resolution:* this is the substance of DP1 and §8; I do not average it — I treat it as
the probe's central hypothesis-under-test rather than a settled verdict either way.

### Already-implemented / duplicated-capability flags (per the prompt's check against F#)

- **The additive epistemic+pragmatic objective is already scaffolded** [F7
  `actor.py`]. The research's central composition recommendation does not need
  building; it needs *filling*. Anything in the plan that proposes a new objective
  structure is duplicating an existing capability and should instead extend the
  scaffold.
- **The metabolic/offline economy already exists and is settled** [F6]. Research
  proposals to feed it from waking-energy (S3) or couple it (S1) are not new
  capabilities to add but *changes to a settled boundary*.
- **Observer-side-only telemetry is already enforced in code** [F7 `views.py`]. The
  research's "must not enter PolicyView" is already structurally true; the work is
  adding fields to TelemetryView/schemas, not building the boundary.
- **Self-opacity = "no forced attention," not "no access"** is already the project's
  position [F2]; S3 states it correctly, S1/S2 reach compatible conclusions. No
  contradiction with the existing design.

---

## 5. Constraints and requirements

Hard constraints the plan inherits. **Research-surfaced (R)** and **F-file-imposed
(F)** are tagged; F-file constraints are the firmer.

- **(F1/F2) No installed self-continuation drive; no self-optimization machinery.**
  The preference must be a fixed, stateless prior — no learned value head, no critic,
  no reward predictor, no precision that Io learns or self-optimizes. [F1, F2;
  echoed S1, S2, S3]
- **(F1) Pressure is permitted and wanted; continuation-as-evaluative-frame is
  prohibited.** Scarcity/hardship are charter-sanctioned growth pressure; the
  prohibition is on continuation becoming "the non-negotiable frame through which
  everything else is evaluated." [F1]
- **(F6) The `MetabolicBudget` is content-blind, Io-state-independent, time-
  replenished, and its content-blindness is test-enforced.** The sensed waking-energy
  (Io-state-derived) may not gate dream-entry; coupling it into the budget breaks
  `tests/test_metabolic_reentry.py` and reverses Phase 8b. [F6]
- **(F7/F2/F8) PolicyView is frozen at `{h, z, self_prediction_error}`.** Adding a
  *direct* energy field triggers the four-part Watts discipline (which affordance,
  minimum form, alternatives, failure-mode controls). Energy-as-observation-channel
  (into `h, z`) does not. [F7, F2, F8]
- **(F7) The opacity boundary is code-enforced** (module-import lint in
  `tests/test_views.py`, mypy `--strict`, frozen dataclasses). All new
  decomposition/occupancy/precision telemetry goes on TelemetryView; none may reach
  the actor. [F7]
- **(F5) The valence substrate is a *waking* mechanism and must not leak into the
  dream regime as a pragmatic driver.** F5 commits dream to "not for anything," no
  gradient flow, content-blind exogenous trigger. A non-zero `pragmatic_value` during
  a dream rollout would make dreams *for* energy management — a self-optimization-in-
  dream F5 forbids. The energy channel may be *decoded* in dream telemetry (the world
  model predicts it), but the preference term must not drive dream content or dream
  entry. **[synthesis inference from F5; the research does not address it.]**
- **(F7) The actor is amortized (DreamerV1-lineage), trained in imagination with
  world-model + ensemble frozen.** The pragmatic preference attaches to the *decoded
  energy observation along imagined rollouts* and is optimized through analytic
  gradients; it is not a per-env-step planning score. [F7]
- **(F7) Environment ground truth:** 8×8 grid; cell types {empty, wall, resource};
  resource consumption triggered by *entering* a resource cell (no consume verb);
  resources resampled fresh at episode boundary, with regrowth; `n_initial_resources`
  default 4; **there is currently no energy variable** — consuming a resource has no
  metabolic consequence today. The energy economy is net-new env state. [F7]
- **(F7) The K=5 ensemble disagreement (variance across members' next-state
  predictions) is the epistemic term**; `h=200`, `z=16`; free bits the only stability
  borrow. The tiny-tensor regime is real. [F7, CLAUDE.md]
- **(CLAUDE.md / F5 / F6) Tests must pass across the *full* suite, not just the
  phase's new tests** (the sink-routing gap stayed red three phases because per-phase
  validation was scoped to new tests). mypy `--strict` on all `kind/` sources.
- **(CLAUDE.md) No timelines or duration estimates.** (S-side note: S2/S3 contain no
  velocity content; this is a standing constraint, flagged because the Probe 3
  synthesis had to reject a candidate's cost/Gantt content under it — F5.)
- **(R) Reconstruction-loss imbalance + free-bits collapse will kill a 1-D channel
  unless mitigated** (up-weight loss ~10×, normalize, per-dim KL floor, weaker channel
  decoder, interventional verification). [S1, S2, S3]
- **(R) The channel must be noisy/lagged/coarse**, not exact, both for realism and to
  keep it on the no-forced-attention side of opacity. [S1, S2, S3]
- **(R) Bounding-by-construction is available** (saturation; optionally clip to a
  fraction of the epistemic scale) and is preferable to pure hyperparameter tuning —
  but see the co-design constraint below. [S1, S2, S3]
- **(F2, synthesis) Co-design constraint:** parameters must not be swept until the
  pass/fail assay signatures are pre-registered, or the probe degenerates into
  engineering the result the mirror is asked to recognize. [F2 §co-design; T8]

---

## 6. Decision points

Each is self-contained: the question, the options, F-alignment, and the
research-supported recommendation (with my confidence). These drive the plan.

### DP1 — Build the valence substrate at all, and in what framing?

**Question.** Given that all three outputs independently flag this probe as sitting
on the charter's central prohibition with unproven mitigations, does the plan (a)
proceed to build it, (b) build a deliberately minimal "boundary probe" whose question
is the boundary itself, or (c) defer pending more conceptual work?

**Options & tradeoffs.**
- *(a) Build as a valence substrate.* Fills the F7 scaffold; tests whether mattering
  can be afforded. Risk: if framed as "add valence," success pressure biases toward
  tuning until the pull appears, importing the co-design hazard (T8) and the
  continuation-frame risk (T1, C5).
- *(b) Build as a boundary probe.* Same machinery, but the probe's stated question is
  "**can a bounded, saturating, non-terminal homeostatic preference create an
  energy-dependent explore/conserve trade-off without becoming Io's evaluative frame —
  and can we detect when it does?**" Both outcomes (pull-without-frame; pull-becomes-
  frame) are findings. Aligns with F4 (one specific question), F1 (the charter's own
  test), and "build to understand."
- *(c) Defer.* Honors S2's strongest objection. Cost: the F7 scaffold and F1's
  central ethical claim stay untested; the question is conceptual-only, which the
  project's stance ("build to understand") treats as weaker than construction.

**F-alignment.** (b) aligns best with F1 (operationalizes "continuation not installed
as imperative"), F4 (probe discipline), F2 (co-design — the boundary framing resists
result-fitting). (a) risks F1/F2 drift. (c) aligns with caution but under-serves
"build to understand."

**Recommendation.** **(b)** — build it, framed as a boundary probe. The
near-unanimous red flag is a reason to make the failure mode *observable*, not a
reason to defer; the project's epistemics treat a built falsification as more
valuable than an unbuilt worry. (Confidence: medium-high; this is the synthesis's
central judgment, expanded in §8.)

### DP2 — Economy: separate or coupled to the `MetabolicBudget`?

**Question.** Is the sensed waking-energy its own economy, or coupled to the offline
dream pacer?

**Options.** *Separate* (own depletion/replenishment, bounded to waking) [S2] /
*coupled-but-distinct, budget fed from waking-energy* [S3] / *separate-first, weak
one-directional coupling later* [S1].

**F-alignment.** **Coupling conflicts with F6.** The `MetabolicBudget` is content-
blind, Io-state-independent, and test-enforced (`tests/test_metabolic_reentry.py`),
and time-replenished by deliberate departure from the waking-replenished idea.
Feeding it an Io-state-derived energy signal reverses Phase 8b and breaks its guard.
Separate aligns; coupling requires an explicit Phase 8b reopening.

**Recommendation.** **Separate**, bounded strictly to the waking horizon (S2's call,
on F6's stronger reason). Treat any economy coupling as out of scope absent a
deliberate decision to reopen F6. (Confidence: high — F6 is settled and code-enforced.)

### DP3 — Channel dynamics and shape parameters

**Question.** Depletion/replenishment law and channel fidelity (noise σ, lag,
resolution).

**Options.** Depletion = time decay + action-magnitude cost; replenishment = resource-
entry coincidence [S1, S2, S3 — consensus]. Fidelity: noisy (σ swept, start moderate;
Hadjiantoni precedent σ≈0.0/0.9/2.0), lag 1–2 steps, normalized [S1]; continuous vs
8–16-bin quantized is the C4 minor conflict — continuous likely sufficient in a tiny
grid [S3] vs quantize for biological imprecision [S1].

**F-alignment.** Consistent with F7's existing resource mechanic (entry-triggered
consumption already exists; the energy economy layers on top). Noisy/lagged is
consistent with F2's Watts default-to-no (the channel must be inferred, not read).

**Recommendation.** Depletion = time + action cost; replenishment = resource entry
[consensus]. Start **continuous, noisy (moderate σ), 1–2-step lag, unit-normalized**;
make σ, lag, and (optionally) quantization **swept** parameters with values fixed only
after pre-registration (DP1b / §8). (Confidence: high on form; the exact σ/lag are
empirical — §7.)

### DP4 — Attachment locus: observation channel vs new PolicyView field

**Question.** Does energy enter only as a world-model observation channel (encoded
into `h, z`), or also as a direct fourth PolicyView field the actor reads raw?

**Options.** *Observation-only* — actor conditions on energy implicitly through `h,
z`; PolicyView stays `{h, z, self_prediction_error}`; preference attaches to decoded
energy in imagination. *Direct field* — actor reads raw energy, parallel to
`self_prediction_error`.

**F-alignment.** Observation-only keeps the F7-frozen field set intact and triggers
no new Watts exception; it matches the research's "exteroceptive-like" framing (T2)
and the self-opacity stance (energy is about Io's world-coupling, not its processing).
A direct field triggers F8's four-part discipline and inherits the Phase 7 dead-column
risk at the actor input (T6).

**Recommendation.** **Observation-channel-only.** The self-prediction scalar earned
its direct-field status because it is about Io's *processing*; energy does not need
that and should not pay the opacity cost. (Confidence: high. **[synthesis inference]**
— the research underspecifies this; the F-files make the cheaper option clearly
correct.)

### DP5 — Composition: keep the coefficient-free scaffold, or add a bounded weight?

**Question.** Fill `pragmatic_value` and leave `total_return = sum_disagreement +
pragmatic_value` coefficient-free (Tschantz-pure) [S1], or introduce a fixed bounded
β / saturation / clip [S1, S2, S3]?

**Options & tradeoffs.** *Coefficient-free* — theoretically clean (one objective, two
complementary terms), matches the F7 scaffold as written, but gives **no
construction-level dominance guarantee** [S1]. *Bounded mechanism* — saturation
(unanimous) and/or clip-to-epistemic-scale (S1/S2, constructed) give a dominance
guard by construction, at the cost of modifying the scaffold and pre-shaping the
balance (the T8 co-design hazard).

**F-alignment.** The scaffold is coefficient-free today [F7]; either choice is a
defensible fill. The charter's dominance-fear (F1) favors *some* guard; the co-design
discipline (F2) warns against pre-shaping the result by tuning.

**Recommendation.** Adopt the **structural, non-tunable** guards that are honest
design commitments rather than result-fitting — **saturation** (flat in-band value),
**no terminal state**, **no viability→capacity coupling** — and keep the **precision/σ
the empirical sweep variable**, disciplined by pre-registered signatures (§8). Defer
the explicit-β / clip-to-epistemic-scale question to DP6; do not add a tuned
coefficient as the *primary* dominance control, because that is the lever most prone
to co-design fitting. (Confidence: medium — this is the synthesis's resolution of the
C2 fork, expanded in §8.)

### DP6 — If a hard ceiling is wanted, which mechanism?

**Question.** Beyond saturation, add a construction-level ceiling so the pragmatic
term cannot exceed the epistemic scale?

**Options.** *Saturation only* [S3-compatible] / *saturation + tanh clip at a fixed
fraction (e.g., 0.5×) of the running K=5 variance max* [S1, S2] / *bias-not-gate
formulation* [S1].

**F-alignment.** Clip-to-epistemic-scale is `[constructed]`, self-flagged by S1 as not
established practice; it reads the epistemic scale to set the pragmatic ceiling, which
couples the two terms' magnitudes (benign, observer-side). None conflicts with F-files.

**Recommendation.** **Saturation first; treat clip-to-epistemic-scale as an optional,
separately-tested second guard**, not a settled mechanism — its novelty (C3) means it
should be evaluated, not assumed. (Confidence: medium.)

### DP7 — Resource regeneration: sustainable or depleting?

**Question.** Do environmental resources regenerate so that homeostasis is
indefinitely achievable, or deplete so scarcity is eventually inevitable? S2 flags
this as radically altering the long-run pragmatic/epistemic ratio [S2]; F7 shows
resources currently resample fresh *per episode* with some regrowth.

**Options.** *Sustainable* (homeostasis reachable; pull is a recurring mild pressure)
/ *depleting* (inevitable scarcity; pull intensifies over an episode — closer to a
survival squeeze, which leans toward the prohibited frame).

**F-alignment.** Depleting-to-inevitable-scarcity pushes toward the continuation-frame
risk (T1, F1); sustainable keeps the preference a recurring bias rather than an
escalating imperative. F7's current per-episode resample is closer to sustainable.

**Recommendation.** **Sustainable / regenerating**, at least for the first build —
inevitable scarcity manufactures exactly the survival squeeze the charter warns
against, and confounds the boundary question (DP1) with environmental doom.
(Confidence: medium-high. **[synthesis inference]** building on S2's fork and F1.)

### DP8 — Verification protocol and pass/fail definition

**Question.** What asserts the channel is genuinely learned, and what counts as
pass/dominant/inert?

**Options.** The Q6 assertions (predictable-from-latents, interventional response,
action-history ablation, per-dim KL escape) [S1, S2, S3] at the relevant locus
(world-model latent and, if DP4 chooses a direct field, the actor column — T6); the
three assays (graded scarcity, novelty-vs-replenishment, recovery) with their
signatures [S1, S2, S3]; telemetry set (T7).

**F-alignment.** The dead-path assertions are demanded by the project's history (F2
Phase 7; F5/F6 sink-routing) and must run in the **full** suite. The assays are
observer-side, consistent with F7's enforced boundary. Pre-registration of signatures
is the F2 co-design requirement.

**Recommendation.** Adopt all three assays and the full assertion battery at the
relevant locus; **pre-register the pass/dominant/inert signatures before any
parameter sweep** (§8). Define pass as the §3-T7 energy-dependent trade-off with
epistemic behavior persisting under moderate scarcity. (Confidence: high on protocol;
exact thresholds are empirical — §7.)

### DP9 — Dedicated latent allocation / free-bits for the channel?

**Question.** Reserve a small set of latent dims with their own free-bits floor for
energy, and/or weaken the channel decoder to force latent reliance? [S1, S2]

**Options.** *No dedicated allocation* (rely on up-weighted recon loss + per-dim KL
monitoring) / *dedicated dims + own free-bits floor* / *+ weakened channel decoder*.

**F-alignment.** Free bits is "the only stability borrow" in the substrate
(CLAUDE.md); adding a channel-specific floor extends an existing mechanism rather than
introducing new machinery — consistent, but it is a substrate change to make
deliberately.

**Recommendation.** Start with **up-weighted reconstruction loss + per-dim KL
monitoring** (cheapest, least architectural change); escalate to **dedicated dims /
weaker decoder only if monitoring shows collapse** [S1]. (Confidence: medium — the
research supports both; staging avoids premature substrate surgery.)

---

## 7. Gaps and open questions

- **The tiny-tensor double-bind (S1's sharpest empirical red flag).** A slowly
  varying 1-D channel may be *simultaneously* too weak to learn (Q6 dead-path) and too
  strong once weighted (Q5 dominance) — "every tuning is inert or dominant" is "a live
  possibility, not a remote one," and a pass "may require a narrower β/σ window than
  the substrate can reliably hit" [S1]. Unvalidated whether a learnable, non-dominant
  window exists in Io's `h=200, z=16`, 8×8 regime. Must be probed empirically, early.
- **No proof the bounding prevents drift to frame.** All three: the saturation/clip/
  no-terminal-state features are `[constructed]`, with "no proof they prevent the
  preference from gradually becoming the evaluative frame, especially under continual
  learning where precision could drift" [S1]. The amortized actor trains continually;
  whether the learned policy's *effective* valuation drifts toward
  continuation-as-frame over long runs is unmeasured. The mirror should watch for it
  (the §8 falsification condition).
- **[synthesis gap] Can an amortized actor even express the energy-dependent
  trade-off the assays demand?** The research assumes per-decision EFE scoring; Io's
  actor is reactive (T4). Whether a single amortized policy, trained on `disagreement +
  pragmatic`, can produce the novelty-when-sated / resource-when-low *contextual* switch
  (rather than a fixed average behavior) is an open question about amortization, not
  addressed by the research and not obvious. Worth an early check before tuning.
- **[synthesis gap] Interaction of the energy channel with the existing
  `self_prediction_error` scalar.** Io will then carry two "self-ish" quantities
  influencing behavior (one a direct PolicyView field, one — under DP4 — an
  observation). Whether they interact (e.g., energy excursions correlating with
  self-prediction error, the Q7 telemetry S1 proposes) is unstudied and could
  confound the assays.
- **Exact σ, lag, resolution, recon-loss weight, saturation width, (optional) β/clip
  fraction.** All empirical; the research gives starting points (σ moderate; lag 1–2;
  recon ~10×; clip 0.5×; pragmatic 10–30% of EFE variance at mid-band) [S1, S2, S3]
  but explicitly defers exact values to calibration. Must be swept against
  pre-registered signatures.
- **Whether the dream regime should *decode* energy at all.** F5 forbids the
  preference *driving* dreams (§5), but whether the world model's energy prediction
  should even appear in dream rollouts (as passive telemetry) or be masked is
  unaddressed by both research and F-files. **[synthesis open question.]**
- **The probe's name/number and its place in F4.** Probe 3.5 is not in the probes
  document; its specific question (DP1b framing) should be ratified and F4 updated
  before planning, per the project's discipline against accreting un-questioned
  features.
- **Staleness.** None — all sources are 2026-06-09; the canonical literature they cite
  is current (several 2024–2025 arXiv items). The only "stale" element is the design
  notes' *waking-replenished* metabolic framing, already superseded by F6's
  time-replenished token bucket; do not let the research's S3 coupling proposal
  silently reinstate it.

---

## 8. Recommended direction (synthesis-level — my call, separated from the sources)

The sources tell me *how* to build a valence substrate and *that* it is dangerous.
They do not resolve the danger; all three end on the same unproven-mitigation note.
My synthesis-level recommendation is a reframing that uses the project's own
discipline to convert the danger from a liability into the probe's content.

**1. Build it, as a boundary probe, with a falsifiable question (DP1b).** Not "add
valence to Io," but: *can a bounded, saturating, non-terminal homeostatic preference
create an energy-dependent explore/conserve trade-off without the preference becoming
Io's evaluative frame — and can the telemetry/mirror detect when it does?* Both
outcomes are results. This is the operational form of the charter's own claim that
"continuation is not installed as imperative" (F1) and of the enactivist
"mattering"/sense-making the project's frameworks want (F3). The near-unanimous red
flag (S1–S3) argues *for* building the failure mode where it can be observed, not for
deferring it — consistent with "build to understand, not to solve" (F1).

**2. Extend the existing scaffold; change nothing settled.** Fill `pragmatic_value`
in the F7 actor objective (a Gaussian log-preference over the *decoded energy
observation* along imagined rollouts); add energy as an *observation channel* into
`h, z`, not a new PolicyView field (DP4); keep PolicyView frozen; keep the economy
**separate** from the `MetabolicBudget` (DP2, F6-forced); keep the preference out of
the dream regime entirely (F5). The plan should touch the env (new energy state +
observation channel), the world model (encode/decode/recon-weight the channel), the
actor objective (fill the scaffold), and TelemetryView/schemas (the decomposition) —
and *not* the dream pacer, the opacity boundary, or the dream "not-for-anything"
commitment.

**3. Make the bounding structural where it is honest, and empirical where it is not —
then pre-register before sweeping.** Saturation, no-terminal-state, and
no-viability→capacity-coupling are honest design commitments (they encode "this is a
bias, not a survival squeeze"), not result-fitting; adopt them as structure (DP5). The
precision/σ window is genuinely empirical and must be searched — but the search is the
co-design hazard (T8). **The resolution is F2's own freeze-criteria-early discipline:
write down the pass / dominant / inert assay signatures, and the dead-path assertions,
*before* running any sweep, and do not edit them in response to what the sweep
produces** (if the urge to adjust a signature arises mid-sweep, that urge is the
co-design loop and should be journaled, not acted on). This is what separates
"discovering whether a non-dominant window exists" from "tuning until the mirror sees
what I hoped." It also directly answers S2's strongest objection: the bounds are not
"self-optimization machinery" (a category error — they are fixed observer-side priors,
like loss weighting), but the *tuning* would be result-engineering unless disciplined
this way.

**4. Treat the deepest red flag as the probe's live hypothesis, with a named
falsification condition.** The literature the project cites says homeostasis ≈ survival
(S1–S3). The probe's job is to find out whether the *bounded, observation-level,
non-terminal* form escapes that equivalence in Io's actual substrate, or merely
disguises it. The **falsification signatures** (which the mirror and telemetry watch
for, pre-registered): in-band occupancy saturating toward 100% even when sated;
positional/epistemic entropy collapsing; pragmatic value-share → 1; failure to resume
exploration after recovery (the dominant signatures of T7) — *these are
continuation-becoming-the-frame*, and if they appear, the finding is that the
charter's prohibition has real teeth and bounded homeostasis is not enough, **not** a
bug to tune away. S3's forward-looking worry — that an only-homeostatic stakes-source
could harden a continuation frame and complicate later non-grasping work — is the
reason to keep this probe *minimal and reversible* and to resist escalating the
preference's precision in search of a stronger pull.

**5. Verify against the project's specific dead-path history, in the full suite.** Io
has been burned three times by silent dead paths (the CA-MAS attention head at 0.0;
the Phase 7 self-prediction column at 0.0; the sink-routing gap across three phases) —
the energy channel is a fourth candidate, at whichever locus DP4 selects. The Q6
assertion battery must run, and must run in the **full** suite per the sink-routing
lesson, with mypy `--strict` throughout.

**Net:** proceed, narrowly, with the danger instrumented rather than designed around.
The probe is worth building precisely because it can *fail informatively* — and the
single most important discipline, drawn from the project's own co-design mitigation
rather than from the research, is to fix what counts as success before searching for
it.

---

*Synthesis grounded against S1–S3 (`docs/research/probe3.5/`) and F1–F8. Verification
pass: re-scanned each source for any implementation-relevant claim dropped — the
economy-coupling reversal (C1/DP2), the coefficient-free-scaffold finding (T4/DP5),
the attachment-locus distinction (T2/DP4), the dream-leak constraint (§5), and the
co-design/pre-registration resolution (T8/§8) are the five points where this synthesis
departs from or sharpens the research against the live system; each is tagged at its
source. No task breakdown, sequencing, or timeline included, per the prompt. The
implementation plan is a separate session.*
