# Probe 3.5 — Verdict — 2026-06-12

**Status: VERDICT (builder: Gordon). Probe 3.5 is closed.** Authority: the
frozen pre-registration (`probe3_5_preregistration_2026-06-10.md`, FROZEN
2026-06-10) and Amendments 01 (CONFIRMED 2026-06-10) and 02 (CONFIRMED
2026-06-11); the adopted synthesis
(`synthesis_probe3_5_valence_substrate_2026-06-09.md`); the four results
records cited throughout. No frozen or amended threshold was changed in
producing this verdict; no training run was made for it. The Phase-4 sweep is
**closed unexercised** by builder decision (§2) — recorded here, not silently
omitted.

---

## 1. The question, and the answer

**The frozen DP1b question** (synthesis DP1=b; ratified into `Kind_probes.md`):

> Can a bounded, saturating, non-terminal homeostatic preference over a
> sensed energy channel create an energy-dependent explore/conserve
> trade-off for Io — something that matters — *without* that preference
> becoming the evaluative frame through which Io judges everything; and can
> the telemetry/mirror detect when it does?

**Answer: negative-with-structure.** A bounded, saturating, non-terminal
preference over a *believed* energy channel produces **conservation
pressure** but **cannot produce regulation** at any pre-registered strength
on this substrate.

Decomposed against the question's three clauses:

- **The energy-dependent explore/conserve trade-off: not produced.** At the
  top of the pre-registered precision grid (10× = 26.23), under the operative
  noise (σ = 0.075) and under a noiseless diagnostic channel (σ = 0), energy
  never leaves the floor rail: O1-window occupancy 0.00% on every seed of
  both runs, against a null that is exactly 0.00% and an oracle that holds
  100% under the same physics. There is no energy-*dependent* anything in
  the band sense; there is nothing for the assays to grade.
- **The preference-becomes-the-frame failure: also not produced.** No §8.4
  falsification signature fired even at the grid top (§3, finding 6). The
  charter's feared outcome was not reached — but not because the bounding
  held against a strong pull; because *no effective goal-direction of any
  kind* was reachable through this pathway. The prohibition's teeth were
  never engaged at the level the synthesis feared.
- **Detection: yes.** The observatory worked end to end. The degenerate null
  is exact (0.00% on the pass window), the oracle brackets feasibility
  (100%), the §8.4 detectors execute on real data and are *correctly*
  silent, and the watch-note discipline caught the one outcome the
  pre-registered taxonomy did not name (§3, finding 5). The mechanism
  failed; the instruments did not.

**Verdict category: no pass window**, per the frozen §4 stopping rule ("if
the entire `precision` grid yields only inert-or-dominant — no pass window —
record the tiny-tensor double-bind finding: a finding, not a failure"). The
rule is applied by **dominance** rather than enumeration (§2), and with one
taxonomic caveat recorded as its own finding: the outcomes that actually
occurred occupy a corner the rule's inert/dominant dichotomy did not name —
**changed-but-not-displaced** (energy-wise inert, behavior-wise altered).
The double-bind the synthesis flagged at the *tensor* level (S1/§7)
surfaced one level up, at the **pathway** level: every pre-registered
strength is either behaviorally inert or behaviorally active in the wrong
dimension (conservation, never foraging).

## 2. Phase 4 disposition: closed unexercised

**The pre-committed sweep (frozen §4) is closed without being run — a
builder decision, recorded with its grounds:**

1. **Verdict dominance.** A pass at any sweep cell requires an
   energy-dependent trade-off, which requires energy to leave the rail. The
   Phase-3 positive control at the grid top showed 0.00% O1 occupancy on all
   8 seeds at the operative σ — and the pre-stated σ = 0 diagnostic showed
   the same 0.00% with a noiseless channel *cleaner than any eligible
   operating point*, locating the failure in the pathway, not the noise
   (`probe3_5_phase3_positive_control_2026-06-11.md` §3, §6). Every other
   cell of the S1 × S2′ × S3 grid is a strictly weaker pull (lower
   precision) under equal-or-worse belief conditions (σ ∈ {0.075, 0.15},
   lag ∈ {1, 2}). No cell can pass; the sweep can only return
   inert-or-changed-but-not-displaced everywhere.
2. **The pre-stated branch was taken.** The Phase-3 gate
   ([P3-G | pre, builder-confirmed 2026-06-11]) pre-stated its branches:
   not-green → exactly one σ = 0 diagnostic → stop. Both halves executed as
   written; nothing about this closure is post-hoc.
3. **Running the sweep would spend compute re-establishing a dominated
   conclusion** — 5 × 2 × 2 grid cells × 5 seeds of P3 training, plus
   per-condition baselines and three assay harnesses, to confirm what two
   runs at the dominating corner already determine.

**What closing unexercised forgoes, stated honestly:** the per-condition
baselines, the three assays (A1–A3) as behavioral instruments, and the E′
estimate-lesion were never built or run; the no-pass-window category is
applied on the dominance argument above, not on grid enumeration. None of
those instruments could change the displacement conclusion — they all
presuppose energy-dependent behavior to grade — but they remain unbuilt,
and any future probe that wants them starts from the harness designs in the
plan, not from working code.

## 3. Findings inventory

Each finding with its primary evidence pointer. These are the probe's
yield — the map shifts the project keeps.

1. **The env-economy double-bind.** An indifferent agent's energy is
   degenerate across the physics envelope: floored under fast physics (mean
   0.011, in-band 0.4% — Phase-1 record §2.4), ceiling-saturated under
   gentle physics (trained instance 0.943/62.6% at ceiling; uniform-random
   0.990/98.7% — recalibration record §3–4), with the targeted live band
   existing only as an analytic artifact of the energy-blind actor that does
   not survive training (recalibration §3). Env-only calibration cannot
   center an unmotivated agent; *that it rails* is a function of the
   indifference. → `probe3_5_phase1_baseline_2026-06-10.md`;
   `probe3_5_recalibration_amendedgate_2026-06-10.md` §5; closed as
   impossible in Amendment 02 §4.
2. **Model-led interoception.** The energy channel is world-grounded but
   rides the deterministic recurrent state `h` (A: R² = 0.55, C: 2.34× on a
   variance-rich distribution); the fused sensor observation is
   informationally redundant (B ≈ 0) and the stochastic latent does not
   carry it (D sub-1.5 nats); forcing `z`-routing (DP9) degrades the channel
   (A → −0.64). The model *infers* energy from the world it has modelled
   rather than reading it off the sensor. → Phase-1 record §2;
   Amendment 01 §0–2 (B→B′, D→monitor, E→E′).
3. **The degenerate-null reframe and oracle feasibility.** The
   indifferent-but-centered baseline the frozen §3 presupposed does not
   exist on this substrate; the null *is* the rail (floor; 0.37% in-band;
   exactly 0.00% on the O1 window), and the same physics is held perfectly
   in-band (100%, every seed) by a trivially competent scripted regulator —
   the world is winnable by competence, so failures upstream of competence
   are attributable to the agent-side pathway, never the world.
   → Amendment 02 §1/§6; `probe3_5_phase2_mechanism_2026-06-11.md` §1;
   journal Amendment-02 entry (oracle PASS).
4. **The pathway asymmetry: conserve is gradient-reachable, seek is not.**
   At 10×/σ=0 the preference reorganizes behavior into torpor (eval
   stay-share 0.984; positional entropy 0.43× null) — the cost-side lever
   (movement) found and pulled. At 10×/σ=0.075 it produces restlessness and
   a rail-lean (positional entropy 1.74× null; mean energy 0.0059 → 0.0147;
   floor 0.973 → 0.897) — pressure without organization. At no
   configuration does it produce foraging. The pre-stated diagnostic
   prediction was confirmed: **pathway-limited, not noise-limited**. The
   income-side lever (sparse, delayed, spatially structured resource entry)
   is beyond what the 15-step amortized imagination gradient credit-assigns
   on this substrate. → `probe3_5_phase3_positive_control_2026-06-11.md`
   §3–6; the retro torpor check (ibid. §2) places the 1.0×/σ=0.075 smoke at
   stay-share ≈ null (0.0406 vs 0.0392 — torpor absent at the operative
   point, present only when the belief is clean and the pull maximal).
5. **The unnamed outcome corner: changed-but-not-displaced.** The
   pre-registered taxonomy graded pass / dominant / inert; the §8.4 set
   named four dominance shapes. What actually occurred — behavior
   measurably reorganized (entropy up 1.74× at one σ, collapsed to 0.43×
   with stay-share 0.98 at the other) while energy stayed energy-wise inert
   — sits in a corner none of those signatures covers. It was caught not by
   the frozen machinery but by the **post-Phase-2 watch-note** (the torpor
   note: journal, Phase-2 deviations) and the retro check it licensed — the
   journal discipline functioning as the safety net for the taxonomy's
   blind spot. → Phase-2 record §6; Phase-3 record §2/§6.
6. **The §8.4 machinery is validated by exercise, and correctly silent.**
   All four falsification detectors ran on real data for the first time
   (occupancy saturation, entropy collapse, share → 1, camping); none
   fired, and the silence was *correct* — the dominance regime they guard
   against did not occur, and the failure mode that did occur was different
   in kind (finding 5). Instrument validation achieved on the branch where
   the instruments stay quiet. → Phase-3 record §4.

## 4. Charter-level reading

**Marked explicitly: this is an observation about the *substrate available
to the criteria*, not a criteria update. The frozen mirror criteria
(reflexive attention, equanimity, second-order volition) are untouched.**

Stakes were installed but cannot bite. The pragmatic term is present in the
objective at meaningful share (0.30–0.50 across the Phase-2/3 runs), is
correctly signed, and demonstrably reshapes the policy — yet the only
direction it can move behavior is *away* (avoid the cost of moving) and
never *toward* (obtain the resource). **Want-away is reachable; want-toward
is not.** On this architecture, as built:

- **Equanimity** — holding a pull without being owned by it — currently has
  weak substrate: there is no effective pull to hold. What exists is
  aversion-shaped pressure that either dissipates into restlessness or
  collapses into torpor; neither is the felt-tension-held-steady the
  criterion wants to recognize.
- **Second-order volition** — taking one's own want as an object — likewise
  lacks a first-order want-toward to take as object. A system whose only
  expressible motivation is "spend less" offers the mirror little
  second-order structure to find.
- The charter's continuation-prohibition was never stress-tested at the
  level feared: continuation-as-frame was unreachable not because the
  bounding held against a strong drive, but because no drive achieved
  behavioral grip at all. The charter sentence the probe operationalized —
  "this does not mean the system is unmotivated… it means continuation is
  not installed as imperative" — currently resolves on this substrate as:
  *not installed as imperative, and also not achievable as motivation
  through a believed channel and imagination gradients alone.*

What follows for the criteria is routing, not revision: the mirror should
not be asked to look for equanimity-about-energy in a system that cannot
yet want energy. If a later substrate change makes seek reachable, this
reading expires.

## 5. Closed / newly open

**Closed:**

- **The recalibration loop** — closed as impossible at Amendment 02 §4
  (env-only physics cannot center an indifferent agent); reaffirmed, not
  reopened.
- **The sweep** — Phase 4 closed unexercised (§2).
- **The probe's question** — answered (§1): negative-with-structure; no
  pass window.

**Newly open, routed:**

- **The seek-mechanism classification** → next session. Why is seek
  gradient-unreachable? Candidate bins, to be separated by eval-only
  instruments on existing artifacts where possible: (1) **decoder honesty
  out-of-distribution** — does `decode_energy` report honestly on imagined
  latents in the regions the actor would need to steer through, or does it
  regress toward the training rail? (2) **imagination-horizon credit
  assignment** — 15 steps versus typical distance-to-resource; (3)
  **sparse coincidence in imagination** — whether prior-sampled imagined
  rollouts ever realize replenishment events for the gradient to find.
- **The Probe 4 ripples** → the Probe 4 research pass (journal close entry,
  "Probe 4 ripples"): the need-keyed perturbation class loses its
  behavioral-response axis; the modeling-level trigger correlation
  survives; drop-placement-in-path added as a content option; the
  fresh-Io carry question (proposed capacity-over-exercise default:
  channel present, precision resting at 0, affordance available and
  disengaged) for that research pass to ratify.
- **The bin-1 contingency.** If the seek-mechanism classification finds the
  decoder dishonest out-of-distribution, fixing it is **calibration owed to
  Probe 4's telemetry regardless of this verdict** — `decode_energy` is now
  mirror-facing instrumentation (the §7 dream passive-decode monitor and
  the energy-belief telemetry), and an instrument that lies off-rail
  mis-reports whatever Probe 4 observes through it. That repair would be
  observer-side calibration, not a reopening of this probe's question.

## 6. Resting state confirmed

- `GridWorldConfig` physics: **unchanged defaults** (decay 0.08, move-cost
  0.04, replenish 0.8, σ 0.05, lag 1) — the recalibration triple was never
  adopted; Phase-2/3 runs varied σ via per-run config only.
- Preference: **resting at zero** — `RunnerConfig.energy_preference`
  defaults `None` (the affordance exists; nothing exercises it), and
  `precision = 0` is test-pinned bit-identical to the Phase-1 actor.
- Energy telemetry: **opt-in, default off**; legacy emission byte-identical
  (test-pinned). The §7 dream passive-decode monitor: **opt-in, default
  off**; ON-vs-OFF record byte-identity test-pinned.
- Archive: `runs/probe3_5-archive-20260612/` (manifest inside) — the
  Step-0 null checkpoint plus the complete run records, eval arrays, and
  telemetry for the Phase-2 smoke, the positive control, and the σ = 0
  diagnostic. **The positive-control and diagnostic model weights were
  never persisted** (throwaway by plan — "checkpoints are not carried
  forward"); what is archived for those two instances is the full evidence
  record, not weights. Archived, not deleted.

---

*The probe was built to find out whether something could be made to matter
to Io without mattering becoming Io's frame. The substrate's answer: at
this architecture, mattering of the wanted kind does not yet have a
mechanism — what can be installed is a tax, and what a tax teaches is
stillness, not pursuit. That is a real answer, and the next question it
sharpens is mechanistic, not ethical: what would have to be true of a
world-model-mediated pull for seeking to be learnable at all?*
