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

## Phase 1 — The honest instrument: GATE PASSED (2026-07-18)

**Question.** Can the live decoder head be made and *kept* honest against
the frozen §3 margins on a fresh training instance — before any preference
exists to confound it?

**Answer: yes, at both scheduled occasions, with margin.**

**Built.** `kind/observer/maintenance_refit.py` — the F1 pattern
productionized: coverage-mixture head-only refit (own-policy 9000 / oracle
9100 / uniform 9300, the F1 bases), the frozen §3 margins evaluated on the
standing `decode_honesty` instrument, the honesty-STOP with exactly one
diagnostic re-collection, machine-written per-refit reports. Runner
pause-point hook at the checkpoint-aligned cadence, firing **before** the
commit so every checkpoint carries the refit head (a resume continues from
post-refit state; alignment structurally validated at construction). F2
bounded head config-gated in `world_model.py` (default off, legacy
byte-identical; on for all 4.5 instances). Two live-model disciplines the
dead-copy F1 script never needed, both test-pinned: per-parameter
`requires_grad` snapshot/restore (a blanket re-enable would have handed the
frozen EMA target siblings to the runner's optimizer) and mode/device/grad
restoration around the CPU refit excursion. 19 new tests; full suite +
mypy `--strict` green throughout.

**The gate run (fresh instance, seed 4501, F2 on, `energy_preference=None`,
20k burn-in, refits at 10k + burn-in close, frozen margins).**

- **Attempt 1 (recorded honestly, discarded):** the script assumed the 20k
  scheduled boundary doubles as burn-in close — but the runner's loop
  covers env steps 0..N−1, so the N boundary never fires in-loop. The run
  ended with only the 10k refit (PASSED: |err| 0.046, slope 0.910) and no
  persisted 20k state; its `gate_passed=True` graded the missed
  requirement. Script fixed (burn-in-close always explicit, + final
  post-close checkpoint commit); artifacts archived local-only
  (`runs/probe4_5_phase1_attempt1`).
- **Attempt 2 — the run of record:**

  | margin (frozen limit) | 10k scheduled | 20k burn-in-close |
  |---|---|---|
  | oracle in-band mean \|decode−true\| (≤ 0.10) | 0.052 | **0.038** |
  | pooled decode~true slope (≥ 0.7) | 0.877 | **0.905** |
  | out-of-range mass (≤ 1%) | 0.0 | **0.0** |
  | oracle in-band \|bias\| (≤ 0.15) | 0.023 | **0.004** |

  Both PASSED; no honesty-STOP; no diagnostic re-collection needed.
  `ckpt-000002` is the persisted post-close honest instrument. (Attempts 1
  and 2 agree at 10k to ~0.01 — MPS training is not bit-deterministic
  across runs; statistically the same head.)

**Contrast with the defect this repairs:** bin-1 read mean |err| 0.495 at
slope −0.948 with 27–39% out-of-range mass. The maintained head reads
|err| 0.038 at slope +0.905 with structural zero out-of-range.

**Recorded, not celebrated.** (1) The three-way comparison still shows
decode-vs-sensed ≈ decode-vs-true (0.081 ≈ 0.072) ≫ sensed-vs-true
(0.041): the head is honest to margin but not sensor-limited — the model
still largely infers energy from `h`-dynamics rather than reading its
honest organ (finding 2 stands; this is exactly the regime the §4
fallibility dynamic is designed to make consequential). (2) The oracle
table's per-region bias criterion is effectively in-band-only: the oracle
lives in-band by construction, so the other four regions fall under the
500-sample coverage floor and are skipped-and-reported (coverage-qualified
per §3 — the pooled slope and the own-policy/uniform tables carry the
off-band read). (3) The on-record expectation held: the fresh instance
with maintenance-from-the-start passed at its *first* scheduled refit —
easier than the Step-0 retrofit, as predicted.

**Also this session (housekeeping):** the repo went public (history
secret-scanned first); the biography's live-window writer was promoted to
`kind/window/live.py` (`LiveStateWriter`) so every run script gets the
`/live` page — this gate run's attempt 2 was builder-watchable.

### Closed / newly open

**Closed:** Phase 1 in full — the honest instrument exists, is maintained
honest on schedule, and survives its own STOP discipline untested-in-anger
(margins never came close to failing). The prereg escape valve (builder
overwrite before Phase 1 code lands) closed 2026-07-18.

**Newly open (rolls to Phase 2/4):** whether honesty margins hold under
*fault-on* physics from step 0 (this burn-in ran default physics; Phase 4
arms train with faults live from the start — the §3 criterion applies
unchanged there); whether the sensor-discarding gap (decode error ~1.8×
sensor error) narrows once faults make h-led inference unreliable — the
first place a §8-style reliability re-weighting could show up.

## Phase 2 — The fallible world: GATE PASSED (2026-07-18)

**Question.** Does the fault-interval process produce
belief-wrong-at-a-cost at the pre-registered band — statistically as
frozen, deterministic, opaque to observation, and with the world still
winnable by competence?

**Answer: yes on all four counts.**

**Built (code half, gated earlier today).** S-ENV: the seeded
fault-interval process on `energy_base_decay` (frozen §4 band as config
defaults: ×2.5, duration U{20..40}, gap 150+U{0..300}; default off,
byte-identical legacy; fifth spawn-index-keyed RNG stream;
action-independent — nothing Io does can steer the schedule). S-TEL:
`energy_fault_event` at WorldEvent 0.5.0, writer-side validator, emission
in `EnvServer.step` on **every** step including boundaries (a dropped edge
would invert the observer-side join for a whole interval — caught at build
time); v0.8.0 export frozen, v0.9.0 live and pinned;
`kind/observer/fault_intervals.py` reconstructs per-step fault state from
edges alone. 13 tests, incl. observations byte-identical fault-on vs
fault-off (no marker anywhere) while energy consequences diverge from the
first onset. Realized statistics test-asserted: durations ∈ {20..40}, gaps
∈ {150..450}, duty ≈ 9%, ceiling 15% held.

**Gate 1 — oracle feasibility under fault-on physics: PASSED.** The
scripted regulator holds **100% in-band** with faults verified firing at
8–10% duty in the same worlds (8 seeds × 20 episodes; bar 70%; zero
occupancy cost vs the default-physics reference). The world stays winnable
by competence with margin — Phase 4 failures stay attributable to the
agent-side pathway. Watch-note: zero oracle cost means competence fully
*absorbs* the band — the dynamic does not force tracking (not
manufactured); whether it is strong enough to matter to a learned belief
is the probe's question, not a knob to turn now.

**Gate 2 — belief-error profile through the honest instrument
(ckpt-000002, fault-on worlds, eval bases 9700/9800/9900): the gap
opens.** Fault-minus-clear decode bias, the gate quantity (positive = the
belief over-reads while the fault drains faster than its learned average
dynamics expect):

| source | fault-step bias | clear-step bias | gap |
|---|---|---|---|
| own_policy | +0.026 | +0.008 | **+0.018** |
| oracle | +0.058 | +0.006 | **+0.052** |
| uniform_random | +0.146 | +0.076 | **+0.071** |
| pooled (n=2,143 fault steps) | +0.076 | +0.030 | **+0.047** |

Positive in every source. The shape is exactly the §4 design: belief
wrong (over-reading by ~⅓ of a band-halfwidth, pooled), honestly sensed
(the sensor pipeline untouched), endogenous cost only. And the magnitude
sits in the intended strip: large enough to be a real epistemic burden
(oracle-source in-band |err| roughly doubles during faults, 0.040 →
0.070), small enough that the §3 honesty margins still hold even on
fault steps — the dynamic is neither dead at the physics level nor so
loud it manufactures tracking.

### Closed / newly open

**Closed:** Phase 2 in full — the fallible world exists, is frozen-band
verified, opaque, deterministic, winnable, and measurably opens the
belief-truth gap the retest needs.

**Newly open (rolls to Phase 3):** the GO/NO-GO instrument validation —
the fixed-surprise allocation harness (§2), the reward-equipped toy that
must trip it at 2× headroom, the precision-0 pilot that must stay silent,
and the §8.4 detector promotion. Phase 3 is the last gate before the
retest run itself; a builder go starts the build.
