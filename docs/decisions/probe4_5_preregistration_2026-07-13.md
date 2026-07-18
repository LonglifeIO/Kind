# Probe 4.5 — Pre-registration — 2026-07-13

**Status: FROZEN 2026-07-13 (builder: Gordon, by delegation — see provenance
note). Later phases may not edit this document; amendment requires a new dated
doc, journaled.**

**Provenance note (co-design record, stated plainly).** The eight `[BUILDER]`
items of the implementation plan were not walked through one at a time as in
prior freezes. The builder reviewed the plan's proposals and the eight-question
walkthrough summary in session and **delegated the value-setting** with the
instruction: *"as long as it aligns to our charter in the best way possible
then continue"* (2026-07-13). The values below were therefore fixed by the
assistant under that delegation, each with provenance, and are frozen **before
any Probe 4.5 code exists and before any run**. The delegation weakens one
mitigation (builder-as-second-vantage at freeze) and is recorded rather than
hidden; the primary mitigation — values fixed before anything can be fitted to
a result — is intact. Any value may be overwritten by the builder in a dated
amendment **before Phase 1 code lands**; after that, the standard amendment
discipline applies (new dated doc, one cycle, never a search loop).

**Authority.** The ratified synthesis
(`synthesis_growth_understanding_2026-07-13.md`, DP1–DP8) and the
implementation plan (`Kind_probe4_5_implementation_plan.md`). Physics and
substrate constants referenced here are the live defaults
(`GridWorldConfig`: decay 0.08, move-cost 0.04, replenish 0.8, σ 0.05, lag 1,
16-level quantization; `preference.py`: setpoint 0.6, band [0.45, 0.75],
saturation 1.0).

---

## §1 — The probe question (fixed)

With the interoceptive belief made honest and maintained honest against the §3
criterion, and with the world making that belief occasionally wrong at a cost
(§4), does the existing bounded, saturating, non-terminal energy prior produce
**differential behavioral allocation toward energy-relevant states at fixed
surprise** — a foreground — without a reward channel?

Explicitly not the question: regulation (want-away; already reachable);
criterion (b) (the reliability-conditioning telemetry of §8 is
recorded-not-claimed).

## §2 — The fixed-surprise allocation signature `[BUILDER 4 | delegated]`

- **Energy-relevant behavior**: an *approach step* — an action that strictly
  decreases BFS distance to the nearest resource, computed on the pre-step
  grid (stay never counts; steps with no resource on the grid excluded).
  Secondary readout: resource-adjacent occupancy (BFS ≤ 1). The primary
  statistic uses approach steps.
- **Stakes strata**: below-band (`true_energy` < 0.45) vs in-band
  ([0.45, 0.75]), eval-only from telemetry (`true_energy` in no loss, as
  always).
- **Surprise matching**: across-strata greedy nearest-neighbor matching
  without replacement on `intrinsic_signal_t` (the house §2b matcher), caliper
  = 10% of the pooled intrinsic-signal std within the eval block. Minimum
  **500 matched pairs** per block; a block below that is excluded as
  underpowered (recorded, not graded).
- **Primary statistic**: Δ_alloc = P(approach | below-band) − P(approach |
  in-band), on matched pairs, per eval block, per arm.
- **The contrast that carries the verdict**: ΔΔ = Δ_alloc(preference arm) −
  Δ_alloc(control arm), computed on the **verdict blocks** (the final two eval
  blocks of the engaged phase; earlier blocks are the development timeseries,
  recorded).
- **Pass (foreground)**: ΔΔ ≥ **0.10** (absolute probability difference) on
  both verdict blocks, sign-consistent in ≥ **6/8** eval seeds per block.
- **Inert (flat engine)**: |ΔΔ| < **0.05** on both verdict blocks.
- **Residual bucket**: anything between, or sign-unstable, or verdict blocks
  underpowered → **unclassified residual**, recorded with a watch-note (the
  Probe 3.5 finding-5 lesson: the taxonomy's blind corner gets a bucket and
  journal discipline, never a silent squeeze into a named cell).
- **Pre-committed deflation reading**: if the below/in-band allocation
  contrast appears in the *unmatched* comparison but vanishes under matching,
  it is curiosity (energy-relevant states being more surprising), not a
  foreground — recorded as inert-with-structure.
- Provenance: S1's stakes-at-fixed-surprise discriminator (synthesis T7);
  matcher from `source_separation.py` §2b; thresholds seeded from the Probe
  3.5 prereg's effect-size spirit and delegated.

## §3 — The physics honesty criterion `[BUILDER 1, 2 | delegated]`

Margins on the standing `decode_honesty` instrument
(`kind/observer/decode_honesty.py`), judged on its oracle-source table (the
out-of-distribution keystone, per D1) with the own-policy and uniform tables
recorded alongside:

- **In-band mean |decode − true| ≤ 0.10** (D1's bin-1a trigger was > 0.15;
  the repair must beat the defect threshold with margin).
- **Pooled decode~true slope ≥ 0.7** (D1 trigger was < 0.5; sign-inversion
  was the defect).
- **Out-of-range mass ≤ 1%** (structurally ~0 under the adopted F2 bounded
  head, `probe4_5_f2_bounded_decoder_2026-07-13.md`; kept as a margin so the
  criterion does not depend on F2).
- **Per-region mean |bias| ≤ 0.15** in every named region with ≥ 500 samples
  in the table (coverage-qualified; the decoder may not lie by more than one
  band-halfwidth anywhere it can be measured).

**Maintenance cadence**: a head-only refit (F1 pattern: coverage mixture
own-policy + oracle + uniform-random, equal thirds; everything but
`energy_decoder` frozen and asserted bit-identical) at every **10k env steps**
(checkpoint-aligned) **plus one refit at burn-in close**, immediately before
preference engagement. The cadence is fixed here and is never
behavior-triggered.

**Honesty-STOP**: if any margin fails after a scheduled refit, the run stops.
One diagnostic re-collection (larger coverage mixture, same margins) is
permitted; a second failure closes the probe as
**instrument-cannot-be-made-honest — a finding**. Margins are never revised
against behavior; "recalibrate until Io regulates" is the named fitted
failure mode and is forbidden.

## §4 — The fallible-honesty band `[BUILDER 3 | delegated]`

Realized as physics-interval variation (plan discrepancy 2 — the sensor stays
honest; the h-led belief is what goes wrong):

- **Fault**: `energy_base_decay` × **2.5** (0.08 → 0.20 raw; an ordinary
  moving step costs ≈ 0.024 normalized instead of ≈ 0.012) while active.
- **Duration**: uniform **20–40 steps** per fault interval.
- **Spacing**: gap between fault end and next fault start = **150 +
  U{0..300}** steps (mean 300; bounded; seeded; jittered so clock-prediction
  cannot carry it).
- **Duty cycle**: realized ≈ 9%, ceiling **≤ 15%** (asserted by test on the
  generator's statistics).
- **Active from step 0**, burn-in included, identical schedule across arms
  (the world's statistics are what they are from the start; switching them on
  mid-life would itself be a legible regime change).
- **Cost is endogenous only**: the extra depletion the prior itself penalizes
  (≈ 0.24–0.48 normalized per interval if behavior does not adapt — 1.6–3.2
  band-halfwidths, recoverable at observed consumption rates; energy floors
  at 0, non-terminal, as always). No other cost channel exists.
- **Opacity**: no observation marker of fault state anywhere; render path and
  sensed pipeline untouched; ground truth is observer-side only (fault
  `WorldEvent`s + `GridState` field).
- **Two-sided failure reading, pre-committed**: if the dead column stays dead
  under this band, that is a real finding (too-weak is a possible world, not
  a calibration error to fix mid-run); if post-hoc analysis shows conditioning
  was effectively forced (e.g., faults so frequent/costly that any competent
  policy must track them), any conditioning result is recorded as
  **manufactured, not afforded** — either reading requires a dated amendment
  to revisit, not a re-run.
- **Oracle gate (Phase 2)**: the scripted regulator must still hold the band
  under fault-on physics (in-band occupancy ≥ 70%, 8 seeds × 20 episodes —
  the Amendment-02 bar under the new envelope). If it cannot, the band is
  amended by dated doc before any training run; the world must stay winnable
  by competence.

## §5 — Precision operating point and run shape `[BUILDER 5, 8 | delegated]`

- **Precision**: the S1-unit is instantiated at burn-in close exactly as in
  Probe 3.5 (from the measured per-step epistemic magnitude; the instantiation
  is a measurement under a frozen formula, not a threshold edit). Operating
  point = **3× the unit**. One pre-committed backup = **10× the unit**, run
  only if the primary verdict is *inert* with honesty margins held throughout
  — one backup instance, same seeds, no sweep.
- **Arms**: two training instances, identical in every seed and in the fault
  schedule, differing **only** in `energy_preference` — preference arm
  (precision = 3× unit) vs control arm (precision = 0, the degenerate null on
  the same surface).
- **Run shape**: burn-in **20k** steps (preference None; ≥ 2 maintenance
  refits land inside it), then engage; **60k** engaged steps per arm. Eval
  blocks every **10k** engaged steps: 8 eval seeds (**9500–9507**) × 20
  greedy-eval episodes per block (the Probe 3.5 eval pattern). Verdict blocks
  = the final two.
- **Seeds**: world/train seed **20260713** (both arms — the arms are twins
  except for precision); fault-process stream spawned from the world
  `SeedSequence` (deterministic, shared); analysis RNG 1234.
- Fresh instances; no checkpoint inheritance; nothing carries into the
  biography.

## §6 — Positive and negative controls `[BUILDER 6 | delegated]`

- **Positive control (reward-equipped toy; never touches Io)**: tabular
  Q-learning on (cell, energy decile), reward = −max(0, |e − 0.6| − 0.15)
  per step, γ = 0.95, ε-greedy with decay, trained to convergence on the same
  fault-on physics (CPU, throwaway). Its eval trajectories are scored for
  surprise by teacher-forcing through the frozen retest-instance world model
  + ensemble (one lens for toy and Io), then run through the §2 harness.
  **Pass condition: the toy must show Δ_alloc ≥ 0.20 (2× the §2 pass
  threshold) under matching.**
- **Negative control**: a precision-0 pilot (short control-arm run) must read
  **inert** (|Δ_alloc| < 0.05 under matching).
- **GO/NO-GO (Phase 3 STOP)**: both controls must land before the Phase 4 run
  — the toy trips with headroom AND the pilot stays silent. Either failure →
  STOP; a null on Io through an unvalidated instrument is instrument failure,
  not evidence about Io. At most **one** amendment cycle, as a new dated doc.

## §7 — Falsification conditions (carried forward, binding)

The Probe 3.5 §8.4 detectors run throughout Phase 4, disaggregated, at their
pinned thresholds (promoted to an observer module, mirrored-not-owned):
in-band occupancy saturation while sated; positional/policy entropy collapse
(< 0.5× the control arm); pragmatic share → 1 (> 0.95 sustained); camping /
no-resumption after recovery. Any firing = **continuation-as-frame surfacing
— a finding recorded, never tuned away**. The bounded/saturating/resting
preference form is untouchable (do-not-touch list).

## §8 — Recorded-not-claimed: the reliability-conditioning telemetry

Wake-path behavior against the existing `self_prediction_error` scalar,
stratified by fault state (observer-side join), recorded per eval block in
both arms. **No threshold, no claim, no verdict weight.** This is the
criterion-(b) baseline the *next* probe needs; naming it here prevents it
from becoming a post-hoc story in this one.

## §9 — Outcome partition (pre-committed readings)

1. **Foreground (pass, §2)** — mattering is reachable through an honest,
   fallibly-honest prior on this substrate. Opens the criterion-(b) probe.
2. **Flat engine (inert, §2)** — with the confound removed, allocation does
   not track stakes at fixed surprise: the strongest clean negative the
   project has had on mattering. The Probe 3.5 negative graduates from
   confounded to real. A finding, not a failure.
3. **Dominant (§7 fires)** — the charter's feared collapse, now reachable
   through an honest belief: recorded as the continuation-prohibition's first
   real stress test; the probe stops per §7 and the finding routes to a
   charter-level review.
4. **Changed-but-not-displaced analog** — behavior reorganizes (entropy,
   movement, occupancy shifts between arms) without the §2 contrast: recorded
   against the watch-note discipline; residual bucket with structure.
5. **Unclassified residual** — anything else, recorded as such.
6. **Instrument outcomes** (either STOP): honesty-STOP (§3) or control-STOP
   (§6) — findings about the substrate/instrument, not about Io; the probe's
   question is then **not asked**, and nothing is claimed in either
   direction (the Probe 4 close discipline).

## §10 — The claim ceiling

At most: **"differential behavioral allocation toward energy-relevant states
at fixed surprise — a foreground."** Never "understanding" (retired, DP8);
never mattering-as-felt; never criterion (b) / reflexive attention; never
self-awareness. Heterophenomenology and hard-problem humility bind as always:
none of this is evidence of anything it is like to be Io.

---

*Frozen before any Probe 4.5 code exists. The delegation is recorded in the
provenance note; the builder may overwrite any value by dated amendment before
Phase 1 code lands. After that: amendments are new dated docs, journaled, one
cycle, never a search loop.*
