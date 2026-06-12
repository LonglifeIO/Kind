# Probe 3.5 — decode-honesty instrument + F1 demonstration — 2026-06-12

**Status: post-close instrument adoption + remedy demonstration (routed by
the seek-mechanism classification §9.1, which the verdict doc §5 bin-1
contingency triggered).** Eval and head-only recalibration on an **archived
copy**: no live substrate changes, no F2 (deferred, gate input reported
here), no fresh-instance work, no Probe 4 design. The archived Step-0
original stays byte-intact as lineage evidence (SHA-256 asserted before and
after); the recalibrated variant is a new artifact alongside it.

---

## 1. The standing instrument

The classification's one-off D1 diagnostic is promoted to a standing
observer-side instrument: `kind/observer/decode_honesty.py`
(`run_decode_honesty`; output schema `DECODE_HONESTY_SCHEMA_VERSION =
"0.1.0"`; tests `tests/test_decode_honesty.py`). Given any checkpoint + env
config, it teacher-forces evaluation trajectories from three policy sources
— the instance's **own greedy policy** (on-distribution), the **oracle
forager** (in-band coverage by construction), **uniform-random** (mid
coverage) — and renders the honesty table per source and pooled:

- rows over the five named energy regions — floor-adjacent / below-band /
  in-band / above-band / ceiling-adjacent — edges from config constants
  (band = setpoint ± halfwidth, the frozen B0b / amended B0a′; rail margin =
  one sensing-noise σ, `GridWorldConfig.energy_obs_noise_sigma`);
- columns N, decode mean/std, true mean, bias, mean |error|; plus the global
  decode~true slope and the **three-way error comparison**
  (decode-vs-true, decode-vs-sensed, sensed-vs-true) — so "ignores the
  honest sensor" (classification §2) is a standing readout, not a one-time
  finding;
- **out-of-range mass** (decode outside the physical [0, 1]) — the F2 gate
  input, standing.

Tests are in the instrument-validates-itself style: a synthetic honest
decoder reads honest (bias ≈ 0, slope ≈ 1, errors ≈ 0); a synthetic decoder
with injected bias of known magnitude is detected at exactly that magnitude;
an injected slope is recovered; determinism under fixed seed is asserted on
a random-init model (no test asserts a trained checkpoint's honesty — that
would fit a test to an empirical outcome).

**Calibration-reference note:** this table is the calibration reference for
§7 dream passive-decode monitor readings — dream states wander
off-distribution by construction, and the verdict §5 bin-1 contingency is
exactly that the monitor currently reads them through a decoder whose
calibration envelope was unknown. The table is expected to become
pre-registered Probe 4 instrumentation.

Eval-only: `true_energy` is read as an eval target (the S-ENV rule binds —
it enters no training loss; the instrument trains nothing). Observer-side:
no Io code path imports it.

## 2. F1 demonstration — protocol

Per classification §9.1, on the archived Step-0 instance
(`runs/probe3_5-archive-20260612/step0_null/step0_burnin_checkpoint.pt`,
SHA-256 `9bddae31c5c8e51c3b470890337dea01d0a76a5622efafaaf7dbfd150d1de68b`):

- **Head-only**: refit `energy_decoder` (the 216→64→1 MLP) alone; every
  other parameter frozen and asserted bit-identical post-training.
- **Coverage mixture**, teacher-forced through the frozen model, equal
  thirds (~16k steps each): Step-0's own greedy rail trajectories at the
  archived eval seed series (9000–9007 — the archived eval *distribution*
  reproduced; the archived run's z-sampling RNG state is not reproducible,
  the distribution is, per the classification's collection note), oracle
  in-band trajectories (9100–9107), uniform-random mid-coverage
  (9300–9307).
- **Target is `sensed_energy`, never `true_energy`** — the S-ENV rule binds
  recalibration as much as original training; true energy remains
  eval-only, used by the honesty table alone.
- Instrument run before and after on held-out env seeds (9700/9800/9900
  series; disjoint from the training mixture and from the classification's
  collection seeds).
- Out-of-range mass reported explicitly as the **F2 gate input** — the F2
  decision (bounding the decoder's output) is **deferred, not taken here**;
  it would be a substrate change to a trained function's gradient field and
  requires its own dated decision doc (classification §9.2).
- The recalibrated variant is written as a new artifact alongside the
  archived original (`step0_f1_recalibrated_checkpoint.pt`, provenance
  fields inside; manifest note appended). The original is read, never
  written.

**Training-data provenance, recorded plainly:** the head saw oracle-policy
states no Io-lineage instance ever visited — that is the point of coverage,
and it is recorded, pre-biography, carrying nothing forward.

Script: `scripts/probe3_5_f1_decoder_recalibration.py` (mypy `--strict`
clean); raw output `runs/probe3_5_f1_recalibration/results.json`.

## 3. Demonstration margins — PRE-STATED before any training ran

Written and builder-confirmed **before** the recalibration executed (the
session record is the witness; `[pre, builder-confirmed 2026-06-12]`
provenance — demonstration-internal, touching no frozen machinery). Judged
on the *after* table at the held-out evaluation seeds. Each margin carries
its pre-stated interpretation, so neither a near-miss nor a near-pass can be
relitigated after the numbers exist:

- **M1 — in-band honesty restored.** Oracle-source in-band |bias|
  (|decode mean − true mean| on [0.45, 0.75]) ≤ **0.05** — one
  sensing-noise σ; the broken decoder's in-band |error| was 0.495 (3.3× the
  classification's 0.15 trigger), so this demands a 10× improvement past
  the trigger, not a marginal pass. **Pre-stated fail reading:** in-band
  |bias| > 0.05 reads as *remedy-class-insufficient-at-head-level* —
  implicating the frozen latents upstream, becoming a gate input for F2/F3
  and a Probe 4 design input. Not a retry trigger.
- **M2 — slope sign and range.** Pooled decode~true slope **positive, in
  [0.5, 1.5]** — sign-correct (broken: −0.948, inverted), above the
  classification's 0.5 regression-toward-the-rail trigger reused
  symmetrically, below 1.5 (no over-amplification). **Pre-stated
  interpretation:** with M1 green, an achieved slope materially below 1 is
  the *expected* signature of an honest head on noisy latents (attenuation;
  Phase 1's R² = 0.55) — recorded as a measurement of how much honesty a
  head can extract from these latents, a Probe 4 design input, not a
  near-failure. The 0.5 floor is what makes the M1+M2 pair non-gameable by
  a near-constant decoder (a flat decoder can zero the in-band bias but
  cannot hold the slope).
- **M3 — rail no-regression.** Own-policy-source floor-adjacent mean
  |decode − true| must not exceed its before-value by more than
  **0.02** — the rail readings were the one thing the broken decoder got
  approximately right (|error| ≈ 0.22 vs 0.495 in-band), and coverage
  training must not buy in-band honesty by breaking the readings on the
  distribution where Io actually lives. **Pre-stated expected direction:**
  the coverage mixture includes abundant rail data, so rail error is
  expected to improve or hold; material non-improvement with M3 still
  passing is noted as a head-capacity-tradeoff signal and feeds the F2/F3
  gate input — noted, not failed.

## 4. Results

Refit: 48,000 pairs (16,000 per source, equal thirds), 50 epochs, batch 512,
Adam 1e-3, torch seed 4321; epoch MSE 0.0338 → 0.0094 (the target carries
σ = 0.05 sensing noise plus 16-level quantization, so the floor is
irreducibly ≈ 0.003). All non-head tensors asserted bit-identical; the
archived original's SHA-256 asserted unchanged before and after.

**The before-table reproduces D1 at held-out seeds** — the standing
instrument validates against the classification on env worlds neither the
classifier nor the refit ever touched: oracle-source in-band decode mean
1.136 / |error| 0.508 / slope −0.865 (D1: 1.124 / 0.495 / −0.948); oracle
out-of-range mass 0.8155 (the above-ceiling explosion).

**Margin verdicts (after; held-out seeds 9700/9800/9900):**

| Margin | pre-stated | achieved | verdict |
|---|---|---|---|
| M1 oracle in-band \|bias\| | ≤ 0.05 | **0.007** (decode 0.622 vs true 0.629) | **PASS**, 70× under the broken 0.507 |
| M2 pooled slope | positive, [0.5, 1.5] | **0.902** | **PASS** |
| M3 own-rail \|error\| | ≤ before + 0.02 | 0.193 → **0.025** | **PASS** — improved 7.6×, the pre-stated expected direction |

Selected after-table rows (full tables in `results.json`):

- **oracle (in-band)**: decode 0.622 ± 0.051 vs true 0.629; |decode−true|
  0.040 vs |sensed−true| 0.046 — the decoder now reads *at the honest
  sensor's own accuracy*; out-of-range mass **0.0000** (was 0.8155).
- **own_policy (the rail)**: floor decode −0.007 ± 0.033 vs true 0.000;
  three-way 0.027 / 0.041 / 0.019.
- **pooled**: slope 0.902; |decode−true| 0.075 (was 0.306).

**Instrument note on M2, recorded so the standing table is read correctly:**
the *pooled* slope was already in-range before recalibration (0.813) —
pooling across sources spans the rail-to-band contrast, and that
cross-distribution variance carries the regression even when every
within-source slope is broken. The per-source slopes are the sharper
readout: oracle −0.865 → +0.356, own-policy 0.465 → 0.824, uniform 0.038 →
0.102. M2 is judged on the pooled slope exactly as pre-stated (and passes);
future readings should lean on the per-source rows.

**Attenuation, per the pre-stated M2 interpretation:** the oracle-source
within-band slope after recalibration is 0.356 — an honest head reading a
band-limited true range (±0.15) through latents built from a σ = 0.05,
quantized, lag-1 sensor. With M1 green this is the expected attenuation
signature, recorded as a *measurement of how much honesty a head can
extract from these latents* — a Probe 4 design input, not a near-failure.

**F2 gate input (out-of-range mass, after):** oracle **0.0000**, uniform
0.0043, own_policy **0.5717**, pooled 0.1920. The own-policy mass is the
opposite failure shape from before: it is sub-floor jitter, not explosion —
the rail's true energy sits exactly at 0, and an honest *unbounded* head
fitting a zero-floor target straddles it (floor decode −0.007 ± 0.033;
roughly half the rail readings land a few hundredths below 0). Mass large,
magnitude tiny; before, the mass was above-ceiling at +1.1. This is exactly
what F2 (sigmoid head or clamp) would remove and what the bare number
overstates — **the F2 decision is deferred, not taken here**, per §2.

**Residual calibration envelope, stated honestly:** the uniform-random
source's rare regions stay mis-decoded after recalibration — floor-adjacent
bias +0.661 (n = 107), below-band +0.467. Coverage repairs what coverage
reaches: the head saw 16k uniform-random steps, but the random walk almost
never reaches the floor, and those latent states remain off every training
distribution. The point of the standing table is that this residual is now
*localized and quantified* rather than discovered post-hoc — the calibration
envelope is: honest on-rail, honest in-band, honest at the sensor's own
accuracy on covered regions; not yet honest on rare states under
distribution-shifted behavior.

**Preference-geometry consequence, computed not speculated** (mirror of
classification §2): under the recalibrated belief the floor's believed
band-edge deviation is d = |0.003 − 0.6| − 0.15 = **0.447** (honest: 0.45;
broken: 0.232) and in-band believed deviation ≈ 0 — the rail reads as
honestly bad and the band as honestly good. The inversion the classification
found ("in-band living reads as worse than the rail") is gone *on this
archived copy*. Whether a gradient would organize seek given this honest
belief remains the out-of-scope counterfactual the classification
pre-stated; nothing here trains a policy.

## 5. Discipline

`new_actor_readable_interfaces_added = []`, with the reasoning explicit:

- the instrument is **observer-side** — `kind/observer/decode_honesty.py`
  reads the world model and the env; no Io code path imports it; PolicyView
  stays frozen at `{h, z, self_prediction_error}`;
- the recalibrated head lives on an **archived copy**
  (`step0_f1_recalibrated_checkpoint.pt`), not on any live or
  carried-forward substrate; the working-copy and archived originals are
  byte-identical to their pre-session state;
- `decode_energy`'s actor-objective exposure — the DP5 ledger entry (the
  pragmatic term, Phase 2) — is **untouched and resting at precision
  `None`/0**; recalibrating what a read-only surface *reports* on an
  archived copy adds no new readable quantity to any actor.

No frozen or amended threshold was changed; no physics, env, objective, or
preference change; nothing dream-mediated; no value head or planner.
