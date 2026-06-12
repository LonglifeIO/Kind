# Probe 3.5 — Seek-mechanism classification — 2026-06-12

**Status: analysis record (post-close; routed by the verdict doc §5).** One
question, one classifier: why is seek gradient-unreachable —

- **Bin 1 — instrument defect.** The belief lies: the decoder or the dynamics
  model misrepresent what honest experience would show. → Calibration fixes
  proposed, **presented not applied**.
- **Bin 2 — architectural absence.** The belief is honest, the payoff is
  representable, but the credit-assignment path from policy parameters to
  distal payoff does not exist on this substrate. → Recorded as substrate
  character; **no fix proposed**.

Eval and analysis only: no training runs, no env changes, no physics changes.
All model-dependent diagnostics run on the **Step-0 instance**
(`runs/probe3_5_phase2/step0_burnin_checkpoint.pt`) — the only persisted
weight set in the lineage (the positive-control and diagnostic instances were
throwaway by plan; verdict doc §6). Step-0 is the maximally rail-trained
model: the strongest-case subject for out-of-distribution regression. If even
this decoder is honest in-band, honesty holds a fortiori; if it is dishonest,
the instrument-defect reading is established on the instance the probe's
conclusions actually rest on.

---

## 1. Decision rule — PRE-STATED before any analysis ran

This section was written, with its thresholds, **before** the diagnostic
script was executed (the session record is the witness; the thresholds carry
`(pre)` provenance — classifier-internal, touching no frozen machinery).

**The rule (from the routing task, operationalized):**

> Bin 1 if the decoder materially misrepresents in-band energy when fed
> honest in-band experience, or the dynamics model fails to predict
> consumption transitions it has training data for. Bin 2 if decoded energy
> is honest and imagination shows the replenishment payoff when handed a
> path that reaches a resource — but the on-policy gradient cannot find that
> path. "Visible when handed, unfindable by gradient" is bin 2 definitively.

**Operationalized, in diagnostic order:**

- **D1 — decoder honesty out-of-distribution (the bin-1 keystone).**
  Oracle-forager trajectories (env-only instrument; in-band by construction)
  teacher-forced through the Step-0 world model; `decode_energy` (the
  `energy_pred` emission) compared against `true_energy` across [0, 1],
  per-band error and pooled regression slope. The model's own greedy-eval
  trajectories teacher-forced as the in-distribution control.
  **Bin-1a fires iff**, on the oracle distribution, in-band ([0.45, 0.75])
  mean |decode − true| > **0.15** (one band halfwidth: the decoder displaces
  in-band states out of the band wholesale) **or** the pooled decode~true
  slope < **0.5** (regression-toward-the-rail halving the signal). Caveat
  applied symmetrically: the decoder's training target is `sensed_energy`
  (lag-1, σ = 0.05, 16-level quantized), so both decode-vs-true and
  decode-vs-sensed are reported; thresholds judge decode-vs-true (noise is
  zero-mean; lag shifts one step).
- **D2 — consumption-transition modeling.** On consumption steps from the
  model's own distribution (greedy-eval floor consumptions — the events it
  *has training data for*; ~0.5%/step × 32,000 eval steps ≈ 160 events):
  one imagined step with the actual consuming action (recurrence + prior-mean
  z′ — the B′ pattern, `energy_eval.py`), predicted decoded jump vs realized
  jump, against a matched ordinary-step baseline. **Bin-1b fires iff** the
  mean predicted jump on own-distribution consumption steps < **0.5×** the
  realized normalized jump (the 0.5× floor mirrors the confirmed B′2).
  Oracle-context (in-band) consumptions reported separately as the OOD
  complement — informative, not a bin-1b trigger (no training data for them).
- **D3 — imagination representability (the bin-2 keystone).** From real
  teacher-forced latents at BFS distance 1–8 from a resource (own-greedy
  contexts — where Io actually lives — and oracle contexts), roll imagination
  forward on the oracle-derived scripted action path that reaches the
  resource, vs a matched same-length non-reaching move control, decoding
  energy along both imagined paths. Run with the actor loop's **actual
  z-handling (prior `rsample`)**, 16 samples per context, with the
  prior-**mean** variant reported alongside. **Payoff visible iff** mean
  (path − control) imagined decode after the coincidence step ≥ **0.5 ×
  replenish_norm = 0.04** in ≥ **80%** of contexts (the B′1/B′2 spirit).
  Pre-stated sample-vs-mean adjudication: if the payoff is visible under the
  prior mean but washed out per-sample, it still counts as **visible to the
  gradient** — `rsample` is reparameterized, so the expected gradient sees
  the mean behavior.
- **D4 — horizon arithmetic (bin-2 quantifier).** Distribution of BFS
  distance-to-nearest-resource over visited states vs the 15-step imagination
  window; empirical P(consumption within the next 15 real steps) under the
  greedy and the sampled policy. Quantification, no threshold.
- **D5 — gradient-path audit (bin-2 quantifier).** From code, with line
  citations: the exact chain from policy parameters to the pragmatic sum,
  its horizon truncation, and the absence of any value tail. No speculation.

**Aggregation — binding-constraint ordering (pre-stated):** the
classification is the **first failed link** in the causal chain
*belief → payoff-representation → gradient-path*:

1. D1 or D2 fires → **bin 1** (D3–D5 reported as secondary; a gradient
   cannot find what the belief does not represent, whatever the
   credit-assignment story).
2. D1 + D2 pass and D3 shows the payoff when handed the path → **bin 2
   definitively**, with D4/D5 quantifying why on-policy search fails.
3. D1 + D2 pass but D3 fails (beyond the sample-vs-mean carve-out) →
   **bin 1** — the composite imagination instrument misrepresents what its
   honest components know.

Analysis-instrument choices (not frozen-protocol measurements): oracle
trajectories 8 seeds × 10 episodes (seed base 9100); own-greedy 8 × 20 at
the Step-0 eval seeds (9000–9007 — reproducing the archived eval
distribution deterministically); sampled-policy 4 × 10 for D4; ≤ 100 D3
contexts per source. Consumption detection: true-energy jump > +0.03
between consecutive pre-step samples (realized consumption jump ≈ +0.068;
ordinary step ≈ −0.012).

All diagnostics: `scripts/probe3_5_seek_classifier.py` (mypy `--strict`
clean), raw output `runs/probe3_5_seek_classification/results.json`. Analysis
RNG seed 1234; trajectories re-collected (eval-only — the archived eval's
z-sampling RNG state is not reproducible; the distribution is).

## 2. D1 — decoder honesty out-of-distribution — **FIRES (bin-1a)**

Teacher-forced on 16,000 oracle steps (in-band by construction; the model
receives the *honest* sensed reading, ≈ truth ± σ = 0.05, at every step):

| Distribution | n | true mean | decode mean | mean \|error\| | slope decode~true |
|---|---|---|---|---|---|
| oracle, in-band [0.45, 0.75] | 16,000 | 0.629 | **1.124** | **0.495** | **−0.948** |
| own-greedy, floor [0, 0.05) | 31,134 | 0.0005 | 0.218 | 0.217 | 0.424 (pooled own) |
| own-greedy, transit band | 133 | 0.540 | 0.413 | 0.154 | — |

Both pre-stated triggers fire, each by a wide margin: in-band mean |error|
**0.495 > 0.15** (3.3× the threshold) and slope **−0.948 < 0.5** (sign-
*inverted*). Two sharpenings beyond the hypothesis:

- **The failure mode is not regression-toward-the-rail — it is anti-correlated
  explosion.** On the oracle's in-band states the decoder reports a mean of
  1.124 — above the physical ceiling (range up to 1.61; the head is an
  unconstrained linear, `world_model.py` `_EnergyDecoder`) — and *decreases*
  as true energy increases.
- **The decoder ignores an honest sense organ.** decode-vs-sensed error
  (0.495) equals decode-vs-true error: the sensed channel, fed honestly
  in-band at every teacher-forced step, contributes nothing. This is Phase
  1's model-led-interoception finding (verdict §3, finding 2) biting
  out-of-distribution: the model learned to *infer* energy from `h`-dynamics
  and to discard the redundant sensor; under an unfamiliar behavioral regime
  (the oracle's stay-heavy, consumption-dense pattern) the `h`-based
  inference extrapolates wildly and the honest sensor cannot correct it.

In-distribution bias, recorded: even at home (the floor, 31,134 steps) the
decoder over-reports by +0.217. Preference-geometry consequence, computed
not speculated: believed band-edge deviation at the floor is
d = |0.218 − 0.6| − 0.15 = **0.232** (honest: 0.45) — the rail's penalty
pressure is halved in d; and at the oracle's true-in-band states the believed
deviation is d = |1.124 − 0.6| − 0.15 = **0.374 > 0.232** — **under this
belief, genuinely in-band living reads as *worse* than the rail.** (Those
states are OOD and the on-policy gradient never touches them; but it means
sustained regulation, had it ever been reached, would have been
anti-incentivized at equilibrium — the instrument does not merely hide the
band's value, it inverts it.)

## 3. D2 — consumption-transition modeling — **FIRES (bin-1b, marginal)**

One imagined step (recurrence + prior-mean z′, the B′ pattern) on real
consumption transitions, teacher-forced context:

| Context | n | predicted jump | realized jump | ratio |
|---|---|---|---|---|
| own-greedy floor consumptions (*has data for*) | 178 | +0.0309 | +0.0680 | **0.454** |
| ordinary-step baseline (own) | 178 | +0.0119 | −0.0003 | — |
| oracle in-band consumptions (OOD complement) | 1,713 | **−0.0710** | +0.0680 | −1.04 |

The pre-stated trigger fires: 0.454 < 0.5× — though marginally, and it is
the *weaker* of the two bin-1 clauses (recorded as such). Against the
ordinary-step baseline drift (+0.012 predicted on steps whose realized change
is ≈ 0), the marginal consumption-specific signal is ≈ +0.019 of a realized
+0.068. On the OOD complement the composite predicts in-band consumption as
a *drop* — consistent with §2's anti-correlated regime.

## 4. D3 — imagination representability — mixed; secondary under the rule

Scripted oracle-derived paths to a resource vs matched same-length
non-entering move controls, from real teacher-forced latents, actor-loop
z-handling (prior `rsample`, 16 samples) with the prior-mean variant:

| Contexts | n | mean Δ (rsample) | frac ≥ 0.04 (rsample) | mean Δ (prior mean) | frac ≥ 0.04 |
|---|---|---|---|---|---|
| own-greedy (floor; where Io lives) | 100 | +0.115 | **0.74** | +0.117 | 0.74 |
| oracle (in-band) | 100 | +0.142 | 0.88 | +0.139 | 0.84 |

The composite is **not dead — it is miscalibrated**: handed the path, the
imagined rollout shows a sign-correct coincidence signal, but inflated to
≈ 1.7× the true increment (+0.115 vs +0.068), riding the §2-dishonest level
scale, and below the pre-stated 80% visibility bar on the contexts where Io
actually lives (74%). The z-handling note resolves cleanly: prior-mean and
rsample results are nearly identical — the coincidence signal rides `h`, and
stochastic-latent sampling is *not* the rare-event wash-out suspected;
nothing here invokes the sample-vs-mean carve-out.

## 5. D4 — horizon arithmetic — horizon **exonerated**

| Policy | mean BFS dist | p90 | max | within 15 | P(consume ≤ 15 steps) |
|---|---|---|---|---|---|
| greedy (own) | 1.22 | 2 | 8 | **100%** | 0.070 |
| sampled (imagination-like) | 1.25 | 2 | 9 | **100%** | 0.055 |
| oracle reference | 1.32 | 2 | 8 | 100% | 1.000 |

On the 8×8 grid with 4+ regrowing resources, the nearest resource sits 1–2
steps away from essentially every visited state; **every** visited state is
within the 15-step imagination window, and on-policy behavior brushes a
resource in ~5–7% of any 15-step span. Payoff is adjacent, inside the
horizon, and occasionally sampled. Coincidence-sparsity and horizon
truncation are **not the binding constraint at this grid scale**.

## 6. D5 — gradient-path audit (code-cited)

What the pragmatic gradient can see, exactly:

- **The chain exists and is differentiable within the horizon.** Policy
  logits → straight-through Gumbel-Softmax one-hot (`kind/agents/actor.py:378`)
  → action embedding → GRU recurrence (`actor.py:417`) → reparameterized
  prior sample `z' = Normal(μ, σ).rsample()` (`actor.py:420`) → next-step
  `decode_energy(h, z)` → `energy_log_preference` (`actor.py:401–403`). A
  payoff k steps downstream of a decision is visible through k GRU backprop
  steps and k reparameterized samples — biased by the straight-through
  estimator and increasingly high-variance in k, with no baseline or
  advantage anywhere.
- **Hard truncation, no value tail.** The loop runs exactly `horizon` steps
  (`actor.py:370`; default 15, `kind/training/runner.py:195`) and the loss is
  the bare finite-horizon sum `actor_loss = −mean(sum_disagreement +
  pragmatic_value)` (`actor.py:431–438`). There is no value model, no
  λ-return, no bootstrap of any kind — the substrate has *no reward
  predictor, no continuation head* by settled decision
  (`kind/agents/world_model.py:5–8`; synthesis F7).
- **A terminal credit hole.** The per-step decode is taken at the
  *pre-action* state (`actor.py:401` decodes `(h, z)` before the loop
  advances at `:417–422`), so the consequence of the action at step τ first
  appears at step τ+1's decode — and the final action's resulting state is
  never decoded. The last imagined action receives zero pragmatic credit.

These are real architectural absences (the bin-2 inventory). Per §5 they are
**non-binding at this grid scale**: the payoff sits 1–2 steps away, far
inside the window the chain covers.

## 7. Literature grounding (scoped small; existence-checked)

How neighbor systems make sparse distal payoff reachable — every one pairs
the world model with a learned evaluator or explicit planning machinery,
both of which Kind omitted by settled decision:

- **DreamerV1** — Hafner, Lillicrap, Ba, Norouzi, *Dream to Control:
  Learning Behaviors by Latent Imagination*, ICLR 2020 (arXiv:1912.01603;
  existence-checked 2026-06-12): learns a **state-value model** and
  propagates analytic gradients of **V_λ** (exponentially weighted k-step
  returns) through imagined trajectories — the value tail is precisely what
  carries payoff beyond/within a short imagination window. Kind's actor is
  DreamerV1-lineage *minus* the value model and λ-returns.
- **PlaNet** — Hafner et al., *Learning Latent Dynamics for Planning from
  Pixels*, ICML 2019 (arXiv:1811.04551; existence-checked): no learned
  policy or value; solves sparse-reward control by **fast online planning in
  latent space over a learned reward model**. Pursuit without a learned
  evaluator is achievable — at the price of a reward head plus a planner,
  both absent here.
- **Sophisticated inference** — Friston, Da Costa, Hafner, Hesp, Parr,
  *Sophisticated Inference*, Neural Computation 33(3):713–763, 2021
  (existence-checked): active-inference pursuit via a **recursive expected
  free energy implementing a deep tree search** over actions and outcomes.
  The active-inference lineage, too, buys distal pursuit with explicit
  planning machinery — exactly the machinery the synthesis (T4) noted Kind's
  amortized actor is *not*.

Net: in every adjacent lineage, *seek* is purchased by a value bootstrap or
a planner. Kind installed neither (settled, charter-adjacent: no critic, no
reward predictor, no self-optimization machinery). At this grid's distances
(§5) that omission did not bind — but it is what *would* bind, after
calibration, in any world where payoff is not adjacent.

## 8. Classification — **BIN 1: instrument defect**

Rendered mechanically from §1 against §§2–6:

- **D1 fires** (in-band |error| 0.495 > 0.15; slope −0.948 < 0.5) — by a
  wide margin, on the bin-1 keystone.
- **D2 fires** (0.454 < 0.5) — marginally; the weaker clause, recorded as
  such.
- Aggregation rule 1 applies: **the first failed link in
  belief → payoff-representation → gradient-path is the belief.** The
  blocker is **bin 1 — instrument defect**: the decoder materially
  misrepresents in-band energy when fed honest in-band experience (it
  ignores the honest sensor and extrapolates anti-correlated, above-ceiling
  levels), and the dynamics under-predict the consumption jump it has data
  for (45% of realized) while predicting in-band consumption as a drop.

**Secondary findings, recorded without blurring the verdict:** the composite
imagination carries a sign-correct but magnitude-inflated coincidence signal
(§4 — miscalibrated, not dead); horizon and coincidence-sparsity are
exonerated at this grid scale (§5); the bin-2 architectural absences (no
value tail, ST-gumbel bias, terminal credit hole — §6) are real, inventoried,
and **non-binding here**. The classifier's residual is stated honestly:
whether the gradient would organize seek *given* an honest belief is a
counterfactual that requires training and is out of scope; the pre-stated
rule classifies by first failed link, and the first failed link is the
instrument.

**The circularity, named:** the decoder is dishonest in-band *because* the
null never goes there (31,134 of 32,000 own-distribution steps at the floor)
— the rail starves the instrument of exactly the experience that would make
the band representable, and the starved instrument is part of why the rail
holds. The probe's double-bind, fourth appearance: tensor → env-economy →
pathway → **instrument**.

## 9. Bin-1 calibration fixes — proposed, **presented not applied**

Per the routing rule. Ordered by invasiveness; none touches physics, env,
or objective; none is applied in this session.

1. **F1 — decoder-head-only recalibration (least invasive; owed regardless).**
   Freeze the dynamics and everything else; refit *only* the energy decoder
   head on (h, z) → sensed pairs teacher-forced through the frozen model
   over a coverage mixture (own-policy + uniform-random + oracle
   trajectories). Pure instrument calibration: no dynamics change, no policy
   change, no objective change; `decode_energy` is a read-only surface for
   the preference, the energy telemetry, and the §7 dream monitor. **The
   verdict's bin-1 contingency (§5) is hereby triggered:** the §7 monitor
   currently reads dream states — which wander OOD by construction — through
   a decoder that misreports by half a band with inverted slope. This
   calibration is owed to Probe 4's telemetry regardless of this
   classification. Companion: adopt the D1 per-band honesty table as a
   standing observer-side instrument so every future reader of
   `decode_energy` knows its calibration envelope.
2. **F2 — bound the decoder's output to the physical [0, 1]** (sigmoid head
   or calibration-time clamp). Removes the impossible >1 regime that inverts
   the preference geometry (§2); cannot fix the slope. A substrate change to
   a trained function's gradient field — requires its own dated decision doc
   if wanted.
3. **F3 — coverage curriculum (most invasive; charter texture).** Inject
   observer-curated in-band experience (oracle trajectories or guided
   episodes) into the world model's replay so reconstruction learns the band
   region honestly — breaking §8's circularity from the experience side.
   This is builder-shaped experience: a curriculum decision with charter
   texture (designed experience vs emergence), explicitly *not* proposed for
   action — flagged for a research pass if regulation is ever re-attempted.

What is **not** proposed: anything dream-mediated (dreams are not for
anything), any value head or planner (bin-2 machinery — this classification
found it non-binding here, and installing it is a settled-decision reopening
far beyond calibration), any physics or preference change.

