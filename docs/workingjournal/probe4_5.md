# Probe 4.5 — Honest-belief mattering retest — working journal

The first probe of the post-v0.1.0 direction: with the interoceptive belief
made honest (and maintained honest against a physics-frozen criterion), and
with the world making that belief occasionally wrong at a cost, does the
existing bounded energy prior produce **differential allocation toward
energy-relevant states at fixed surprise** — a foreground — without a reward
channel? Authority: the ratified synthesis
`docs/decisions/synthesis_growth_understanding_2026-07-13.md`; plan
`docs/plans/Kind_probe4_5_implementation_plan.md`; frozen pre-registration
`docs/decisions/probe4_5_preregistration_2026-07-13.md`; F2 adoption
`docs/decisions/probe4_5_f2_bounded_decoder_2026-07-13.md`.

## Phase 0 — Ratification and pre-registration frozen (2026-07-13)

**Ratification.** The growth/understanding synthesis was ratified in session:
DP1–DP8 all **confirmed as recommended, no overwrites** (DP1 retarget to
self-use; DP2 this probe + fixed-surprise signature + separation from the
biography; DP3 fallibility in-design; DP4 environment-as-welfare; DP5 memory
deferred; DP6 other-recognition parked, source-tags inadmissible; DP7 honesty
maintenance as instrument repair; DP8 "understanding" retired from project
vocabulary). The §8 naming gap resolved at ratification: **Probe 4.5** — the
builder's choice, marking repair/retest within the existing arc rather than a
new register.

**The delegation, recorded honestly.** Unlike the Probe 3.5 and Probe 4
freezes (values walked one at a time, builder confirming each), the builder
delegated the eight `[BUILDER]` value-settings in session: *"as long as it
aligns to our charter in the best way possible then continue."* The values
were fixed by the assistant under that delegation and frozen before any code.
This weakens the builder-as-second-vantage mitigation at freeze and is
recorded rather than smoothed over; the load-bearing mitigation —
**everything frozen before anything runs** — is intact. The prereg's
provenance note carries the same record, plus the escape valve: any value is
builder-overwritable by dated amendment **until Phase 1 code lands**.

**What was frozen** (full text in the prereg; headline values):

- **§2 allocation signature**: approach-step allocation, below-band vs
  in-band, surprise-matched (§2b house matcher, 10%-std caliper, ≥ 500
  pairs); pass = arm-contrast ΔΔ ≥ 0.10 on the final two eval blocks,
  6/8-seed sign-stable; inert < 0.05; residual bucket + the pre-committed
  "vanishes-under-matching = curiosity" deflation reading.
- **§3 honesty criterion**: oracle-table in-band |decode−true| ≤ 0.10, slope
  ≥ 0.7, out-of-range ≤ 1%, per-region |bias| ≤ 0.15 (coverage-qualified);
  maintenance refits every 10k steps + at burn-in close; honesty-STOP with
  one diagnostic re-collection allowed.
- **§4 fallibility band**: decay ×2.5 faults, 20–40 steps long, gaps
  150+U{0..300} (≈ 9% duty, ceiling 15%), active from step 0, identical
  across arms, endogenous cost only, no observation marker; oracle must
  still hold the band under fault physics (Phase 2 gate).
- **§5 run shape**: precision = 3× the S1-instantiated unit (backup 10×,
  inert-only, one instance); twin arms (same seeds, same fault schedule,
  only precision differs; control = precision 0); 20k burn-in + 60k engaged;
  eval blocks every 10k (8 seeds × 20 episodes); world seed 20260713.
- **§6 controls**: reward-toy (tabular Q, never touches Io, surprise scored
  through the frozen Io-lineage instrument) must trip at 2× headroom
  (Δ ≥ 0.20); precision-0 pilot must stay inert; both gate Phase 4
  (GO/NO-GO STOP, one amendment cycle max).
- **§7**: the §8.4 falsification detectors carried forward binding,
  disaggregated.
- **§8**: reliability-conditioning telemetry recorded-not-claimed (the
  criterion-(b) baseline for the next probe).
- **§9/§10**: six-way outcome partition with pre-committed readings; claim
  ceiling = "a foreground," never understanding / felt-mattering /
  criterion (b).

**F2 adopted** as its own dated doc (the classification §9.2 requirement):
config-gated sigmoid head, default off (legacy byte-identical), on for all
4.5 instances — fresh instances train through it from scratch, so no trained
function is retroactively altered. The honesty margins do not depend on it
(out-of-range margin stays in force regardless).

**Status line: FROZEN 2026-07-13.** Phase 1 (the honest instrument —
maintenance harness productionizing the F1 pattern, F2 head, honesty-gate
wiring) is the next buildable step.

### Closed / newly open

**Closed:** the synthesis (ratified); the probe's name and number; the plan;
every pre-registration value; the F2 deferral (adopted, scoped, fresh-instance
only).

**Newly open (rolls to Phase 1):** productionizing the F1 refit as a
scheduled maintenance harness (module + runner pause-point hook); the F2
config gate; whether the frozen margins hold on a *fresh* instance under
maintenance-from-the-start (the F1 demonstration was a retrofit of the
rail-trained Step-0 — a fresh instance with coverage refits from step 0 is
the easier case on paper, and that expectation is itself on record now).
