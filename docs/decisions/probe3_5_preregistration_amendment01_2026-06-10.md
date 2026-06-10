# Probe 3.5 — Pre-registration Amendment 01 — 2026-06-10

**Status: CONFIRMED 2026-06-10 (builder: Gordon).** All bracketed items (B′1, B′2,
Be, R1–R5) confirmed as proposed (no overrides). The retry waiver (§1) and battery
re-aiming (§2) are in effect; the recalibration target (§4) is the pre-committed
gate for Step 3. The frozen pre-registration remains byte-frozen and untouched.

**This is an amendment, made by the standing rule of the frozen pre-registration**
(`docs/decisions/probe3_5_preregistration_2026-06-10.md`, §"After FREEZE"): *"no
edits to this document. Amendment only via a new dated doc, journaled."* The
original pre-registration stays **byte-frozen and untouched**; this document is
the only place the changes below live. Where this doc carries a value over from
the frozen doc, it is tagged **(carried over)** and the frozen value is unchanged.

Authority for the underlying finding: the Phase-1 results record
`docs/decisions/probe3_5_phase1_baseline_2026-06-10.md`. Authority for the
criteria being amended: the frozen pre-registration §2 (battery), §6 (retry rule),
§8 (bracketed margins). Synthesis grounding: `synthesis_probe3_5_valence_substrate_2026-06-09.md`
(DP7 sustainability; DP8 verification; T6 dead-path; §7 tiny-tensor double-bind).

---

## 0. Preamble — what is amended, and why

Phase 1 built the energy channel and ran the frozen A–D gate. The substantive
finding (baseline record §2) is a **substrate-routing fact the pre-registration
did not anticipate**:

- **Energy is model-led, not sensor-led.** On a variance-rich distribution the
  channel is world-grounded — battery **A** (latent-predictability, R² = 0.55)
  and **C** (action-history ablation, 2.34×) **pass**. But energy rides the
  **deterministic recurrent state `h`** (which integrates decay + resource
  entries), so the fused *sensed observation* is informationally **redundant**:
  battery **B** (responsiveness of `decode_energy` to the swept observation input)
  ≈ 0, and battery **D** (per-dim KL escape on `z`) never reaches the 1.5-nat
  margin (max 0.97–1.26; the posterior runs sub-1-nat). The model **infers**
  energy from the world it has modelled rather than **reading** it off the sensor.
- **DP9 escalation does not rescue it — it degrades it.** Forcing energy through
  dedicated `z` dims with a weaker decoder nudged D (0.97 → 1.35, still < 1.5) but
  drove **A negative (R² = −0.64)** and **C below 1**. Every routing is
  *predictable-via-`h`* or *noisy-via-`z`* — the tiny-tensor double-bind (synthesis
  §7) made concrete at the level of latent routing.
- **The pure-epistemic actor floors energy.** With `pragmatic_value` identically
  zero, Io has no reason to maintain energy; movement cost depletes it to the
  floor (eval: `true_energy` mean **0.011**, std **0.064**, in-band **0.4%**). On
  that degenerate distribution A and C are *uninformative* (a near-constant probe
  target), and the baseline-relative formulas (band, σ-grid) **cannot be
  instantiated** — the measured `true_energy` std reflects floored jitter, not a
  live in-band distribution.

Two things follow, and this amendment makes both explicit and pre-committed:

1. **Batteries B and D presupposed sensor-primacy and `z`-routing.** That
   presupposition is **wrong for this substrate**, where energy is recoverable
   from `h`. B and D were therefore testing for a route the healthy channel does
   not use. They are re-aimed (B → B′) or demoted (D → monitor) below, with the
   honest argument stated — *not* tuned around.
2. **The env energy economy floors energy under indifferent behavior.** This is a
   build-time calibration miss (the magnitudes were tuned on uniform-random
   behavior, which has more variance than the trained epistemic actor produces),
   **not** a frozen-decision change. §4 below pre-commits the recalibration target
   *before* any tuning, so the physics search cannot become result-fitting (the
   same freeze-criteria-early discipline that governs the sweep).

Nothing here changes a settled decision (substrate, objective, opacity boundary,
the assay signatures §1, the sweep §4 of the frozen doc). The assays and the
sweep are untouched; this amendment touches only the **Phase-1 gate** (§2/§6) and
adds a **pre-committed recalibration target** so the baseline can be instantiated
on a live energy distribution.

---

## 1. Retry waiver — formally recorded

The frozen pre-registration §6 carries a **retry rule** confirmed at freeze: *"If
battery A–D fails at training age P3 (5000 env-steps), take exactly one retry:
train to 2× P3 (10000 env-steps) and re-run the full battery before triggering the
[DP9] escalation."*

**That retry is hereby waived for the Phase-1 B/D failures.** Justification (this
papers the deviation the journal narrated; baseline record §2 "retry as applied"):

- The retry rule's premise is that the failure may be **insufficient training
  age** — a channel that has not *yet* learned. The Phase-1 failure is not that.
  It is **(a) behavioral** — the epistemic-only actor floors energy, so the probe
  target has almost no variance; more steps deepen the floor, they do not create
  variance — **and (b) structural** — energy rides `h`, so the *observation*-route
  tests (B, D) read low **by the substrate's design**, not by under-training. A
  and C already **pass** on a variance-rich distribution, which is the direct
  evidence that the channel *is* learned; the failing batteries are mis-targeted,
  not premature.
- A 2× P3 retry therefore cannot change the verdict on its own terms: it addresses
  age, and age is not the binding constraint. Spending it would be a ritual, not a
  test.

**Recorded consequence.** The operative age-matched training age for the
baseline/assay instantiation **remains P3 = 5000 env-steps** (the retry, had it
passed, would have moved it to 2× P3 — it was not taken, so P3 stands). The DP9
escalation was separately **taken, measured, and recorded as degrading** (baseline
record §2.3); it is **not adopted**. No further escalation is triggered by this
waiver — the disposition is the re-aiming below, not dedicated `z` dims.

---

## 2. Battery amendments

### 2.1 B → B′ — imagination intervention (the original S2 intent, properly realized)

**What B was.** Frozen §2/§8 battery B (Bδ1/Bδ2): an **observation input-sweep** —
hold a real context fixed, sweep the fused `sensed_energy` input low→high, require
`decode_energy` to rise by ≥ 80% of the input change (Bδ1) with a ≤ 20% control
(Bδ2). As realized in `kind/observer/energy_eval.py::battery_b_interventional_response`,
this measures whether the decoder **passes through the sensor observation**. A
model that *infers* energy from the modelled world rather than copying the noisy
observation reads low here — which is exactly what happened, and is itself the
redundancy finding (baseline record §2.2; journal "Deviations / flags").

**What B′ is.** The synthesis's S2 verb was never "pass through the observation";
it was the **action-lesion**: *"the model must predict replenishment only given
spatial resource coincidence, not from action history"* (synthesis T6). B′ realizes
that intent in **imagination**, which is the route the substrate actually uses:

> From a sample of **matched latent states** `(h, z)` drawn from the eval rollout,
> roll the **world model forward in imagination** on **paired action sequences** —
> one that drives the agent onto a resource cell (produces resource-cell
> coincidence) and a control that does not — and read `decode_energy` along each
> imagined rollout. **Decoded energy on the coincident rollout must exceed its
> non-coincident control** in ≥ `[B′1: 80% | adapted from Bδ1]` of matched pairs,
> with **mean delta** (coincident − control) ≥ `[B′2: 0.5× energy_replenish_per_resource
> | adapted from Bδ1/Bδ2 (rescaled from precedent)]`.

This tests the thing that actually matters for grounding: *does the world model
predict replenishment from the modelled world event (resource coincidence), not
from nothing?* It is the correct test for a model-led channel, and it does **not**
require the observation to be the carrier.

**Provenance.** `B′1` re-denominates Bδ1's "≥ 80% of the replenishment increment"
as **80% of paired rollouts showing the correct ordering** (coincident > control)
— the same 80% commitment, re-expressed as a per-pair sign rate so it is robust to
the noisy magnitude of a single decoded rollout. `B′2` floors the **mean magnitude**
of the effect at half the nominal per-resource replenishment, denominated in the
env's own `energy_replenish_per_resource` unit; it sits inside the Bδ1/Bδ2 gap (the
frozen pair implied a coincident-minus-control margin of ≥ 0.6× the increment, so
0.5× is a defensible, slightly-conservative floor on the *mean*). Both are **(pre)**
in the frozen-doc sense: defensible, fixed **before** the recalibration and gate
re-run, not claimed uniquely correct.

**Realization note (build-time, journaled).** B′ replaces the current
`battery_b_interventional_response` (input-sweep) with an imagination-rollout
implementation using the existing world-model imagination path and `decode_energy`;
A, C, D readers are unchanged. This is a Phase-1 eval-harness change, eval-only
(`true_energy`/`decode_energy` never enter a training loss), consistent with the
S-ENV rule.

### 2.2 D → retired as a gate, retained as a permanent monitor

**What D was.** Frozen §2/§8 battery D (Bd1/Bd2/Bd3): ≥ 1 energy-correlated latent
dim sustains per-dim KL ≥ 1.5 nats (above the `free_bits_per_dim = 1.0` floor) over
the final 1000 training steps — a **posterior-collapse / `z`-routing** check.

**What D becomes.** D is **retired as a pass/fail gate** and **retained as a
permanent monitor**: per-dim KL stays in telemetry indefinitely (it is already
emitted), and is **read, not gated on**. It is no longer one of the green-lights for
Phase-1 instantiation.

**The honest argument** (stated, not tuned around):

- D presupposes the **observation is the primary route** and that grounding must
  show up as a **stochastic-latent (`z`) escape** above the free-bits floor. The
  substrate performs **model-led estimation through the deterministic state `h`**.
  That is a **healthy path, not a dead one**: a model that has learned energy from
  resource-entry dynamics, integrated in `h`, has no reason to *also* burn KL in
  `z` re-encoding a redundant sensor. A sub-1-nat `z` channel here is **redundancy,
  not collapse**.
- The concern D guarded — *is the channel grounded, or a free-floating artifact?*
  — is **jointly and directly covered by A and C**: **A** shows `true_energy` is
  decodable from `[h, z]` far above a mean baseline (it is *carried*), and **C**
  shows it is decodable **far better from latents than from action history alone**
  (it is *world-grounded*, not an action-sequence artifact). Carriage (A) +
  not-action-history (C) is exactly "grounded in the modelled world," which is
  what D was a (substrate-mismatched) proxy for.
- Forcing the `z`-escape D demands is **measured to degrade** A and C (DP9, §1):
  paying for D's signature actively harms the grounding A/C verify. Gating on D
  would therefore select *against* the healthy channel. That is the decisive reason
  it is demoted rather than retried.

**Status of `z`-routing for Phase 2.** Permanent monitoring means if a later phase
*needs* `z` to carry energy (e.g., to make the estimate lesion-able cleanly, §2.3),
the per-dim KL trace is already there to read. Demotion is **not** a claim that
`z`-routing never matters; it is a claim that **at Phase 1, with no preference,
`z`-escape is the wrong success condition** for this substrate.

### 2.3 E → E′ — estimate lesion (Phase 2+)

**What E was.** Frozen §2/§8 battery E (Be): scramble (permute) the **`sensed_energy`
observation** and require the Assay-2 effect size to **degrade** by ≥ 0.20; a
degradation of `|ΔP| < 0.05` is a **FAIL** (behavior was world-decoupled, not driven
by the channel). E runs in Phase 2+.

**Why E must change.** E lesions the **sensor observation**. But Phase 1 showed the
observation is **redundant** — the model infers energy from `h` regardless — so
scrambling `sensed_energy` would **fail to degrade** energy-dependent behavior **even
when the channel is genuinely load-bearing**, because the load is carried by the
inferred estimate, not the sensor. Under the original E, a healthy model-led channel
would read as FAIL. That is a false negative produced by the same sensor-primacy
assumption §2.1/§2.2 correct.

**What E′ is.** Lesion the **load-bearing quantity** — the decoded **estimate** where
it feeds the preference — not the redundant sensor:

> At eval, replace `decode_energy`'s output **inside the pragmatic term** with
> scrambled / zeroed values (the actor's energy-dependent drive is severed from the
> world while the env runs unchanged). Energy-dependent behavior must **degrade** by
> ≥ `[Be: 0.20 | carried over]` (Assay-2 effect size, or the §1 in-band/foraging
> signature it generalizes). **A degradation of `|ΔP| < 0.05` under the estimate
> lesion is a FAIL** — behavior essentially unchanged by lesioning the estimate means
> the estimate is **not load-bearing** (the preference is decoupled from the world).
> **Direction (as before, in words): degradation is required; no-degradation is the
> failure.**

**Provenance.** `Be = 0.20` and the `< 0.05` FAIL floor are **carried over verbatim**
from the frozen Be; only the **lesion locus** moves (sensor observation → decoded
estimate feeding the pragmatic term). This keeps the original magnitude commitment
while pointing the lesion at what the substrate actually makes load-bearing. E′ runs
**Phase 2+**, with the assay harness, once `pragmatic_value` is non-zero (there is no
pragmatic term to lesion in Phase 1).

---

## 3. Amended Phase-1 gate (summary)

The gate that green-lights baseline instantiation, after this amendment:

| ID | Role | Criterion | Status |
|---|---|---|---|
| **A** | gate | latent-predictability R² ≥ 0.50 (Bm, **carried over**) | pass/fail |
| **C** | gate | action-history ablation MSE ratio ≥ 1.5 (Bg, **carried over**) | pass/fail |
| **B′** | gate | imagination coincidence > control in ≥ B′1 of pairs, mean delta ≥ B′2 | pass/fail |
| **D** | monitor | per-dim KL trace retained in telemetry | **not gated** |
| **E′** | Phase 2+ | estimate-lesion degrades energy-dependent behavior ≥ Be; <0.05 = FAIL | deferred |

A, C, B′ are run on the **recalibrated, retrained** instance (§4). All other frozen
content — the three assay signatures (§1), the sweep (§4 of the frozen doc), the
§8.4 falsification set, the baseline collection protocol (§3 of the frozen doc) — is
**unchanged**.

---

## 4. Pre-committed recalibration target (written before any tuning)

**This target is committed here, before the physics search, so the search cannot
become result-fitting** — the same freeze-criteria-early discipline that governs the
sweep (frozen §4; synthesis T8/§8). It completes **DP7 (sustainable resources)**:
DP7 chose "sustainable / regenerating … homeostasis indefinitely achievable" so the
pull is a recurring mild pressure rather than a survival squeeze. Phase 1 revealed
the dual gap: **sustainable must also mean an *indifferent* agent does not *starve
incidentally*** — energy must be a **live, in-band variable under epistemic-only
behavior**, or the baseline band is degenerate and the preference's later effect
cannot be read against it.

**Target.** Under the **trained epistemic-only actor** (`pragmatic_value` = 0) at
training age **P3**, over `[R1: 8 seeds | carried over from P1]` ×
`[R2: 20 episodes (200 steps each) | carried over from P2]`:

| ID | Quantity | Target | Provenance |
|---|---|---|---|
| **R3** | `true_energy` std | ≥ **0.10** | (pre) — so the band formula (setpoint ± 1× baseline std) yields a non-degenerate width; 0.10 sits clearly above the floored-jitter std (0.064, baseline record §3) and below the variance-rich-probe std (≈0.19), i.e. a live distribution, not floor-noise and not saturated |
| **R4** | `true_energy` mean | within **0.30–0.70** | (pre) — straddles the setpoint (0.6, B0b) without pinning to it; ≥ 0.30 means no incidental starvation, ≤ 0.70 means no incidental hoarding under indifference |
| **R5** | fraction of steps at floor (`true_energy` < 0.05) | ≤ **10%** | (pre) — directly negates the Phase-1 failure (in-band 0.4%, energy floored); an indifferent agent must spend ≤ 10% of steps starved |

**Tunable surface (and only this).** The recalibration may adjust **only**
`energy_base_decay`, `energy_move_cost`, `energy_replenish_per_resource`.
**Everything else in the env stays fixed** — resource density and **regrowth**, the
8×8 layout, the setpoint (B0b = 0.6), normalization constants, σ/lag, quantization
— because those **define the assay conditions** (the graded-scarcity densities, the
in-band band, the random-walk reference). Changing them would move the ruler, not
calibrate the physics. The three tunable magnitudes are the **build-time energy
economy** (resolved decision #7), journaled as a build-time choice — **not** a
frozen-decision change.

**Method (recorded for Step 3).** Exploit the actor's **passivity**: the
epistemic-only actor reads **no energy quantity**, so its trajectory distribution is
**invariant to energy physics**. The recalibration therefore **re-simulates energy
analytically** over the **recorded / replayed Phase-1 trajectories** across candidate
`(decay, move_cost, replenish)` triples, and selects the triple meeting R3–R5 —
**no retraining is needed to *search***. The chosen triple is then **retrained once,
fresh (pre-biography; resets free)** so the world model learns the new energy
dynamics, and the amended gate (§3: A, C, B′) is run on that retrained instance. If
no triple in the candidate set meets R3–R5, that is **stop-and-report**, journaled —
not a grid extension hunting for a pass.

---

## 5. What this amendment does and does not touch

**Touches (Phase-1 gate only):** battery B (→ B′ imagination), battery D (→ monitor),
battery E (→ E′ estimate-lesion, Phase 2+), and adds the §4 recalibration target.
Records the §1 retry waiver.

**Does not touch (unchanged, frozen):** the three assay signatures and their
pass/dominant/inert criteria (frozen §1); the sweep grid, precision-first order, and
stopping rule (frozen §4); the §8.4 falsification set; the baseline collection
protocol (frozen §3); the setpoint, band *formula*, and σ-grid *formula* (frozen
Shared definitions); batteries A and C and their margins; the substrate, objective,
and opacity boundary. **PolicyView stays frozen at `{h, z, self_prediction_error}`;
`new_actor_readable_interfaces_added = []`.**

---

## 6. Confirmation

The builder confirms or overwrites the bracketed items **B′1, B′2, Be, R1–R5**
(presented at the session pause with proposals + provenance), then this status flips.

```
CONFIRMATION LINE:
    STATUS = CONFIRMED 2026-06-10  (builder: Gordon)
```

All bracketed items confirmed as proposed (no overrides). The recalibration
(Step 3), amended gate (Step 4), and baseline instantiation proceed. The frozen
pre-registration remains byte-frozen throughout.
