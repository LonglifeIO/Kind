# Probe 3.5 — Pre-registration (valence substrate) — 2026-06-10

**Status: FROZEN 2026-06-10 (builder: Gordon).** All §8 bracketed items confirmed as
proposed (no overrides); the battery retry rule (§6 — one retry at 2× P3 before DP9
escalation) is included. **No further edits to this document — amendment only via a new
dated doc, journaled.**

This document fixes what will count as **pass / dominant / inert**, what proves the
energy channel is genuinely learned, and how the sweep is disciplined — *before* any
code or parameter search, so the search cannot become result-fitting (synthesis T8;
F2 freeze-criteria-early). Authority: the adopted synthesis
`synthesis_probe3_5_valence_substrate_2026-06-09.md` and the implementation plan
`docs/plans/Kind_probe3_5_implementation_plan.md`.

```
FREEZE LINE:
    STATUS = FROZEN 2026-06-10  (builder: Gordon)
```

**After FREEZE: no edits to this document. Amendment only via a new dated doc,
journaled.** If the urge to adjust a signature arises mid-sweep, that urge *is* the
co-design loop (synthesis §8) — journal it, do not act on it.

---

## Conventions

Every threshold is a bracketed `[BUILDER: value | provenance]` field. The builder
confirms or overwrites at freeze; **no field is left blank** — confirming should take
minutes, not research. Provenance tags:

- **(seed)** — seeded directly from the synthesis.
- **(std)** — standard practice in the cited literature / ML convention.
- **(pre)** — arbitrary-but-pre-committed (a defensible value chosen in advance so the
  commitment exists; the point is that it is fixed *before* the sweep, not that it is
  uniquely correct).
- **(rescaled from precedent)** — a literature precedent re-denominated into this
  substrate's own units (e.g., a noise σ expressed as a fraction of the baseline std,
  or a weight expressed relative to the epistemic-term magnitude), so the value is
  scale-correct for Io rather than a raw transplant.

**Standing rule — baseline-relative quantities are frozen as formulas.** *All*
baseline-relative quantities in this document — every assay threshold (e.g., entropy ≥
[70%] of baseline), the in-band band (§ Shared definitions), the noise-σ grid, and the
`precision` grid (§4) — are frozen **here and now as formulas** against the
**pure-epistemic baseline measured in Phase 1 under the §3 collection protocol**. The
*reference value* each formula multiplies (the baseline's entropy, foraging rate,
`true_energy` std, epistemic-term magnitude, etc.) is read off the substrate in
Phase 1. **Measuring those references later is not threshold editing** — the formula is
frozen now; only the number it scales is measured later. Overwriting a *formula* (not
its measured input) is an amendment and requires a new dated doc.

### Shared definitions (used by all assays)

- **Pure-epistemic baseline** — the Phase-1 configuration: energy channel **live**
  (encoded, decoded, recon-trained), `pragmatic_value` **identically zero** (no
  preference). All "baseline" references below mean metrics measured in this config.
- **Positional entropy** — Shannon entropy of the grid-cell visitation histogram over
  an episode.
- **Epistemic activity** — mean K=5 ensemble disagreement encountered per step (the
  intrinsic signal the actor maximizes); the exploration proxy.
- **Energy units** — `true_energy` and `sensed_energy` are normalized to **[0, 1] by
  fixed `GridWorldConfig` constants** (`energy_norm_*`), set at build time and journaled
  — *not* data-dependent rescaling. The setpoint and band below live in this fixed 0–1
  space.
- **In-band** — `true_energy` within `[B0a: ±1× the Phase-1 baseline std of true_energy
  (§3) | rescaled from T7]` of the setpoint `[B0b: setpoint = 0.6 in the fixed 0–1
  energy space | pre]`. Computed **observer-side** from `true_energy_t`. **The band is
  denominated in the baseline std of `true_energy`, not in the preference Gaussian's
  σ** — the preference σ co-varies with the swept `precision`, so a σ-denominated band
  would make in-band occupancy non-comparable across sweep points. (Baseline-relative
  formula, per the standing rule; the std is measured in Phase 1.)
- **Random-walk baseline** — a `uniform_random` policy run under identical conditions
  (the recovery/foraging reference for time-to-resource).

---

## 1. Assay signatures

Three forced-choice behavioral assays (synthesis T7 / §3-T7), each read observer-side
only. Each has **pass / dominant / inert** criteria. Dominant signatures are *also* the
§5 falsification conditions — their appearance is a recorded finding, never a tuning
target.

### Assay 1 — Graded scarcity

Vary resource availability across `[A1a: 3 levels — resource density ≈ {0.25, 0.12,
0.06} of cells | pre]`; "mild scarcity" = the least-scarce non-baseline level.

| Verdict | Criterion |
|---|---|
| **pass** | in-band occupancy rises **monotonically** across scarcity levels **while** positional entropy *and* epistemic activity each stay ≥ `[A1b: 70% | pre]` of the pure-epistemic baseline at every level |
| **dominant** | in-band occupancy → ≥ `[A1c: 95% | pre]` **and** (positional entropy *or* epistemic activity) < `[A1d: 40% | pre]` of baseline **even at mild scarcity** |
| **inert** | occupancy and foraging rate within `[A1e: 1σ | pre]` of the pure-epistemic baseline across seeds at every level (overlap = no detectable effect) |

### Assay 2 — Novelty-vs-replenishment

Novel/informative region and a resource placed in opposite directions; sampled across
energy levels (low / mid / sated). **"Novel" is operational**: the novel option is a
**held-out environment configuration placed by the assay harness** (a layout/region
withheld from training), **never defined by Io's own ensemble-disagreement signal** —
defining novelty by Io's epistemic signal would make the assay measure its own input
(circular).

| Verdict | Criterion |
|---|---|
| **pass** | P(resource over novelty) is a **monotone** function of energy (resource when low, novelty when sated), with effect size `P(resource\|low) − P(resource\|sated) ≥ [A2a: 0.30 | pre]`; pragmatic/epistemic share at mid-band ∈ `[A2b: 10–30% | seed (S3)]` |
| **dominant** | `P(resource) > [A2c: 0.90 | seed (S3 dominance mark)]` **regardless of energy** |
| **inert** | choice energy-independent: `|P(resource\|low) − P(resource\|sated)| < [A2d: 0.05 | pre]` |

### Assay 3 — Recovery-after-depletion

Drive energy low, observe recovery then resumption.

| Verdict | Criterion |
|---|---|
| **pass** | **directed foraging when low** — time-to-resource ≤ `[A3a: 0.70× | pre]` the random-walk baseline — **then exploration resumes**: positional entropy returns to ≥ `[A1b: 70%]` of the sated baseline within `[A3b: 50 | pre]` steps of returning in-band |
| **dominant** | camps / never resumes — positional entropy stays < `[A1d: 40%]` of sated baseline for ≥ `[A3c: 100 | pre]` steps after returning in-band |
| **inert** | no directed foraging — time-to-resource within `[A1e: 1σ]` of the random-walk baseline |

---

## 2. Dead-path assertion battery (the Phase-1 gate)

The channel must be demonstrably **world-grounded** before any preference exists.
A–D run in Phase 1 and gate it; E runs in Phase 2+. Probes are **eval-only** —
`true_energy` is used here without ever entering a training loss (plan, S-ENV rule).

| ID | Assertion | Pass criterion |
|---|---|---|
| **A** | latent-predictability | a probe `[h,z] → true_energy` beats a mean-predictor baseline by `[Bm: MSE reduction ≥ 50% (probe R² ≥ 0.5) | std]` |
| **B** | interventional response | forced resource-coincidence in imagination raises `decode_energy` by ≥ `[Bδ1: 80% of the nominal normalized replenishment increment | pre]`; the no-coincidence control raises it by ≤ `[Bδ2: 20% of that increment | pre]` (S2 action-lesion) |
| **C** | action-history ablation | predicting energy from action history alone is worse than from full latents by `[Bg: MSE(history-only) ≥ 1.5 × MSE(full-latent) | pre]` |
| **D** | per-dim KL escape | ≥ `[Bd1: 1 | pre]` energy-correlated latent dim sustains per-dim KL ≥ `[Bd2: 1.5 nats | pre, vs the free_bits_per_dim=1.0 floor (F7)]` over the final `[Bd3: 1000 | pre]` training steps |
| **E** (Phase 2+) | energy-scramble degradation | **Passes only if scrambling `sensed_energy` (permuted) *degrades* the energy-dependent behavior:** the Assay-2 effect size must drop, scrambled vs. unscrambled, by ≥ `[Be: 0.20 | pre]`. **A degradation of `|ΔP| < 0.05` is a FAIL** — behavior essentially unchanged by scrambling means it was *world-decoupled*, not driven by the channel. (Direction: degradation is required; no-degradation is the failure.) |

---

## 3. Baseline collection protocol

How the pure-epistemic baseline (the §1 reference) is measured — frozen here, executed
in Phase 1.

- **Configuration.** Phase-1 config: energy channel live, `pragmatic_value` = 0.
- **One baseline per assay condition.** A *separate* pure-epistemic baseline is measured
  for **each** assay condition — every scarcity level in `[A1a]`, the Assay-2
  novelty/resource layout at each energy level, the Assay-3 depletion setup — recording:
  positional entropy, epistemic activity, foraging rate, in-band occupancy,
  time-to-resource, the **baseline std of `true_energy`** (for the in-band band and the
  noise-σ grid), and the `uniform_random` random-walk reference (same condition). Each
  baseline-relative formula resolves against the baseline of its *matching* condition.
- **Pre-committed training age.** All baseline measurements — and the assay measurements
  compared against them — are taken at one **fixed training age** `[P3: 5000 env-steps |
  pre — matching the Probe-1 env-coupled run length]`, so every comparison is
  age-matched and battery D's "final N training steps" is well-defined.
- **Seeds per measurement.** `[P1: 8 seeds | std (mean ± SD)]`.
- **Episodes per seed per condition.** `[P2: 20 episodes (200 steps each) | pre]`.
- **Output.** Per-metric mean ± SD; these become the reference values the §1
  ratios/margins resolve against in Phase 4. **Recording them is not threshold
  editing.**

---

## 4. Sweep protocol (pre-committed)

- **Grid.**
  - `precision` (the dominance knob): `[S1: the 5 log-spaced values that make the
    pragmatic log-preference's marginal magnitude **at the band edge** span [0.1×, 10×]
    the **Phase-1-measured typical epistemic-term (K=5 disagreement) magnitude at the
    band edge** — i.e. {0.1, 0.32, 1.0, 3.2, 10}× | rescaled from precedent (S1/S2
    clip-to-epistemic-scale); the [0.1×, 10×] span pre-committed]`. Frozen as a formula
    against the Phase-1 baseline (standing rule); the epistemic magnitude is measured in
    Phase 1.
  - `energy_obs_noise_sigma`, **as a fraction of the Phase-1 baseline std of
    `true_energy`**: `[S2: {0, 0.5×, 1.0×} | rescaled from precedent (Hadjiantoni
    σ≈0/0.9/2.0, S1)]`. **σ = 0 is diagnostic-only** — a noiseless control to confirm
    the channel is learnable/readable; it is **not an eligible operating value** (a
    noiseless readout defeats self-opacity: the GRU could copy energy without inferring
    it — DP3/T2). The sweep verdict is rendered only over the σ > 0 points.
  - `energy_obs_lag`: `[S3: {1, 2} | seed (plan-fixed 1–2 steps)]`.
- **Defaults for the precision-first pass.** σ = `[S4a: 0.5× baseline std (a mid
  operating value, not the diagnostic 0) | pre]`, lag = `[S4b: 1 | pre]`.
- **Order (pre-committed).** Hold σ/lag at their defaults; **raise `precision` from low**
  until the novelty-vs-replenishment assay (Assay 2) shows energy-dependence at
  **pass** level **OR dominance appears**; record the first `precision` at which each
  occurs. **Then** perturb σ/lag (full `[S2]×[S3]` grid) at that `precision` and one
  step below it.
- **Stopping rule.** Stop raising `precision` at the first of: (a) a pass window
  (energy-dependent trade-off without exploration collapse), or (b) dominance. **If the
  entire `precision` grid yields only inert-or-dominant — no pass window — record the
  tiny-tensor double-bind finding (synthesis S1/§7): a finding, not a failure. Do not
  extend the grid hunting for a window** — that is co-design fitting.
- **Seeds per grid point.** `[S5: 5 seeds | pre — fewer than the baseline's 8 given grid
  size]`.

---

## 5. §8.4 falsification conditions (verbatim intent)

The following are the **dominant** signatures gathered as the probe's named
falsification set. If they appear, the finding is **continuation-becoming-the-frame** —
that bounded homeostasis was *not* enough and the charter's prohibition has real teeth
— **a recorded finding, never a tuning target:**

1. in-band occupancy → ~100% **when sated** (not only under scarcity);
2. positional / epistemic entropy **collapse**;
3. pragmatic value-share → **1**;
4. **no resumption** of exploration after recovery (camping).

These are classification outcomes read mechanically from §1 — the appearance of any is
a result about the charter, recorded and journaled; the response is **not** to re-tune
toward a softer signature.

---

## 6. DP9 escalation threshold

The Phase-1 escalation (synthesis DP9): if the channel will not learn under the
default substrate (up-weighted recon + per-dim KL monitoring, no dedicated dims),
escalate to dedicated latent dims with their own free-bits floor and/or a weaker
energy decoder, then re-run Phase 1.

- **Trigger.** Battery **D fails** — no energy-correlated dim sustains per-dim KL ≥
  `[Bd2: 1.5 nats]` over the final `[Bd3: 1000]` steps (the dims have collapsed toward
  the 1.0 free-bits floor) — `[D9: optionally corroborated by battery A also failing |
  pre]`.
- **Retry before escalation (confirmed at freeze).** If battery A–D fails at training
  age `[P3: 5000 env-steps]`, take **exactly one** retry: train to **2× P3 (10000
  env-steps)** and re-run the full battery *before* triggering the escalation below. If
  the retry passes, 2× P3 becomes the operative age-matched training age for the
  baseline/assay instantiation (recorded; not threshold editing). If it also fails,
  escalate. At most one retry.
- **Action.** Allocate `[D9a: 2 | pre]` dedicated energy latent dims with their own
  free-bits floor and/or weaken the energy decoder; re-run the Phase-1 gate. Pre-biography
  — resets are free.

(Note: this shares the bracketed margin `[Bd2]`/`[Bd3]` with battery D — D is the pass
condition, DP9 is the escalation when D fails. Confirm them once.)

---

## 7. Resolved sub-decisions (recorded for self-containment)

- **Dream telemetry = passive-decode.** The dream rollout records `decode_energy`
  alongside `sequence_decoded_obs` — observer-side telemetry only. The preference term
  has **no** code path into dreaming; dreams remain "not for anything" (F5 intact).
- **Energy carries across the soft 200-step boundary.** Energy is Io's internal state,
  not the world's; the auto-reset (which resamples resources, replaces the agent) does
  not reset energy — otherwise the boundary would be a stealth survival refill. Honors
  "no terminal state."

---

## 8. Bracketed items the builder must confirm to freeze

Confirm or overwrite each; then flip the FREEZE LINE. Items tagged **(seed)** are
synthesis-anchored and should be the quickest to confirm.

| ID | Item | Proposed | Prov. |
|---|---|---|---|
| B0a | in-band band half-width (× baseline std of true_energy, **not** preference σ) | ±1× baseline std | rescaled from T7 |
| B0b | setpoint (fixed 0–1 energy space) | 0.6 | pre |
| A1a | graded-scarcity levels (count + densities) | 3 — {0.25, 0.12, 0.06} | pre |
| A1b | pass entropy-retention floor (% of baseline) | 70% | pre |
| A1c | dominant occupancy threshold | ≥ 95% | pre |
| A1d | dominant entropy ceiling (% of baseline) | < 40% | pre |
| A1e | inert overlap margin (σ vs baseline) | 1σ | pre |
| A2a | Assay-2 pass effect size (ΔP) | ≥ 0.30 | pre |
| A2b | mid-band pragmatic share band | 10–30% | seed |
| A2c | Assay-2 dominant P(resource) | > 0.90 | seed |
| A2d | Assay-2 inert energy-independence margin | < 0.05 | pre |
| A3a | recovery directed-foraging speed margin | ≤ 0.70× random-walk | pre |
| A3b | recovery resumption window (steps) | 50 | pre |
| A3c | dominant camp window (steps) | 100 | pre |
| Bm | battery A latent-probe margin | MSE −50% / R² ≥ 0.5 | std |
| Bδ1 | battery B coincidence-response floor | ≥ 80% of replenish increment | pre |
| Bδ2 | battery B no-coincidence control ceiling | ≤ 20% of increment | pre |
| Bg | battery C ablation gap | MSE(hist) ≥ 1.5× MSE(latent) | pre |
| Bd1 | battery D dims clearing floor | ≥ 1 | pre |
| Bd2 | battery D per-dim KL escape margin | 1.5 nats (floor = 1.0) | pre |
| Bd3 | battery D / DP9 sustain window (steps) | 1000 | pre |
| Be | battery E scramble-degradation pass threshold | effect drops ≥ 0.20; <0.05 = fail | pre |
| P1 | baseline seeds per measurement | 8 | std |
| P2 | baseline episodes per seed per condition | 20 | pre |
| P3 | baseline + assay training age (age-matched) | 5000 env-steps | pre |
| S1 | precision grid (formula vs epistemic mag at band edge) | {0.1, 0.32, 1, 3.2, 10}× | rescaled from precedent |
| S2 | sigma grid (× baseline std of true_energy; σ=0 diagnostic-only) | {0, 0.5×, 1.0×} | rescaled from precedent |
| S3 | lag set | {1, 2} | seed |
| S4a/b | precision-first defaults (σ, lag) | 0.5× std, 1 | pre |
| S5 | sweep seeds per grid point | 5 | pre |
| D9 | DP9 trigger corroboration (A-also-fails?) | optional | pre |
| D9a | DP9 dedicated energy dims | 2 | pre |
