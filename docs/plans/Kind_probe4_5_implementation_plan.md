# Kind — Probe 4.5 implementation plan (honest-belief mattering retest)

**Authority.** The ratified synthesis
`docs/decisions/synthesis_growth_understanding_2026-07-13.md`
(Status: **RATIFIED 2026-07-13**, builder: Gordon — all eight decision points
DP1–DP8 confirmed as recommended, no overwrites; probe named **Probe 4.5** at
ratification). Where anything here conflicts with the synthesis, the synthesis
wins. Standing prior authorities bind where cited: the Probe 3.5 verdict and
seek-mechanism classification (`probe3_5_verdict_2026-06-12.md`,
`probe3_5_seek_classification_2026-06-12.md` — the bin-1 lying-decoder finding
this probe repairs), the F1 demonstration record
(`probe3_5_f1_decode_recalibration_2026-06-12.md`), and the charter/design-notes
constraints (F1/F2/F5). **This plan adds no code**; it is the document the
per-phase build prompts are generated from.

**Probe question (DP2 + DP3, ratified).** With the interoceptive belief made
honest — and *maintained* honest against a physics honesty criterion frozen
before any behavior is examined — and with the world making that belief
occasionally wrong at a cost (the fallible-honesty dynamic), does the
**existing** bounded, saturating, non-terminal energy prior produce
**differential behavioral allocation toward energy-relevant states at fixed
surprise** — a foreground — without a reward channel?

Not the question: "does Io regulate." Regulation is want-away, already
reachable (Probe 3.5 finding 4); measuring it would re-run Probe 3.5 and learn
less. Also not the question: criterion (b). The fallibility dynamic *seeds* the
self-use question (DP3: the same design opens the door), and the run *records*
reliability-conditioned use of the existing self-signal as telemetry — but this
probe **claims nothing about reflexive attention**. That is the probe after
this one, if allocation-at-fixed-surprise exists at all.

**This is an instrument-repair probe.** Probe 3.5's negative is confounded: the
decoder misreports in-band energy by half a band with inverted slope while
ignoring an honest sensor (classification §2 — under that belief, in-band
living read as *worse* than the rail). Every negative about mattering is
untrustworthy until the belief is honest. The retest adds no preference
machinery, no reward, no new head — it repairs the instrument and engages the
surface that already exists. The single most important discipline, per the
co-design mitigation: **Phase 0 freezes every signature and band before any
run.**

---

## Architecture grounding (read once; the phases assume it)

Facts established by reading the live surfaces (`kind/`, `scripts/`, `tests/`).
The plan is written against these, not against the research's idealizations.
**The headline: the entire preference surface and the honesty instrument
already exist; the build is maintenance scheduling, one environment dynamic,
one analysis harness, and one toy.**

1. **The preference surface is complete and rests disengaged.**
   `RunnerConfig.energy_preference: EnergyPreferenceConfig | None = None`
   (`kind/training/runner.py:262`), passed to the **waking** actor-training
   call only (`runner.py:1565`). `kind/agents/preference.py` is the whole
   pragmatic machinery: one frozen dataclass + one pure function — saturating
   Gaussian log-preference, `SETPOINT=0.6`, `BAND_HALFWIDTH=0.15` (band
   [0.45, 0.75]), `SATURATION=1.0` (`preference.py:53–60`), flat-in-band with
   exactly zero gradient, `precision` the only knob, `precision=0` the
   degenerate null on the same surface (`preference.py:32–34`). The actor
   reads it coefficient-free: `total_return = sum_disagreement +
   pragmatic_value` (`kind/agents/actor.py:436`), preference evaluated on
   `decode_energy(h, z)` along imagined rollouts (`actor.py:400–404`).
   **Probe 4.5 engages this surface; it does not modify it.**

2. **The belief the preference reads is the bin-1 instrument.**
   `_EnergyDecoder` (`kind/agents/world_model.py:277–294`) is a small ELU MLP
   with an **unbounded** linear head (no output activation — it read 1.124 >
   physical ceiling on oracle in-band states, slope −0.948; classification
   §2). `decode_energy` (`world_model.py:482–498`; DP9 dedicated-dims branch
   at `:495–497`, not in effect). Its training signal is the ELBO recon term:
   MSE against `sensed_energy` at `energy_recon_weight=10.0`
   (`world_model.py:95, 778–794`). **The S-ENV rule stands: `true_energy`
   never enters any training loss** — it lives in `GridState` telemetry and
   eval-only instruments.

3. **The physics honesty criterion has a standing home already.**
   `kind/observer/decode_honesty.py` is the D1 diagnostic promoted to a typed,
   deterministic, checkpoint-agnostic instrument: per-region bias/|error|
   table over five named energy regions (edges from config constants), pooled
   decode~true regression slope, the **three-way error comparison**
   (decode-vs-true / decode-vs-sensed / sensed-vs-true — "the decoder ignores
   the honest sensor" as a standing readout), and **out-of-range mass** (the
   F2 gate input). Eval seed bases 9700/9800/9900, disjoint from every
   training mixture. Phase 0 freezes *margins on this instrument's readouts*;
   no new honesty machinery is built.

4. **The head-only repair is demonstrated, on an archived copy.**
   `scripts/probe3_5_f1_decoder_recalibration.py`: refit **only**
   `energy_decoder` on `(h, z) → sensed_energy` pairs teacher-forced through
   the frozen model over a coverage mixture (own-policy 9000 / oracle 9100 /
   uniform-random 9300, equal thirds); every other parameter frozen and
   asserted bit-identical; before/after honesty tables; SHA-256 lineage
   asserted. It ran on the archived Step-0 copy only — **no live substrate
   has been touched.** Probe 4.5's Phase 1 productionizes this pattern as
   scheduled in-run maintenance.

5. **The sensed channel is honest by construction; the model discards it.**
   `sensed_energy` = normalized true value `energy_obs_lag=1` steps ago +
   Gaussian noise σ=0.05, clipped to [0, 1], 16-level quantized, dedicated RNG
   stream (`kind/env/grid_world.py:229–252, 949–957`). Probe 3.5 finding 2
   (model-led interoception): the model infers energy from `h`-dynamics and
   treats the fused sensor as redundant (B ≈ 0). **This fact shapes the
   fallibility design** (discrepancy 2): corrupting the sensor would barely
   touch the belief; varying the *physics* makes the h-led belief itself go
   wrong while the sensor stays honest.

6. **Energy physics is one clean per-step delta.**
   `delta = −energy_base_decay − energy_move_cost·[moved] +
   energy_replenish_per_resource·[entry]` (`grid_world.py:921–926`). A
   seeded fault-interval process modulating the decay term has a single
   insertion point. **No fault machinery exists — net-new (S-ENV).**

7. **Surprise and allocation are already per-step telemetry.** `AgentStep`
   carries `h_t`, `action_t`, and `intrinsic_signal_t` (the K=5 ensemble
   disagreement — the project's surprise measure), plus the four energy
   fields at record version 0.4.0 and the pragmatic/epistemic decomposition
   at 0.5.0 (`kind/observer/schemas.py`; `runner.py:1849–1863`). The
   surprise-matched comparison has a house pattern: **greedy nearest matching
   without replacement** (`kind/observer/source_separation.py`, the §2b
   PE-matcher; also the Amendment-1 context matcher). The allocation harness
   composes existing patterns — observer-side, no substrate change.

8. **PolicyView is frozen and the self-signal already exists.** PolicyView is
   exactly `{h, z, self_prediction_error}` (`kind/agents/views.py:94–129`),
   test-enforced. The wake-path actor conditions on the real scalar every
   step; the imagination path feeds a structurally zero scalar column
   (`actor.py:363–367`, by Probe 1.5/2 design). **Nothing in Probe 4.5
   touches PolicyView, adds any observation-space marker, or adds any
   self-model head.** The reliability-conditioning telemetry (Phase 4) reads
   *wake-path* behavior against the existing scalar.

9. **The dream regime is structurally out of reach and stays that way.**
   Dreams never call `imagine_and_compute_loss`; the preference has no code
   path into dreaming (import lint + behavioral backstop,
   `tests/test_pragmatic_guards.py`). The §7 dream passive-decode monitor
   (`kind/training/dream.py:181`, default off) reads dream states **through
   `decode_energy`** — decoder honesty is owed to it regardless of this
   probe (verdict §5, bin-1 contingency). The fault process likewise has no
   dream path (dreams run no env physics).

10. **The §8.4 falsification detectors exist but live script-inline**
    (`scripts/run_probe3_5_phase3_positive_control.py:322–439`,
    `signature_suite_8_4`: occupancy saturation, entropy collapse,
    pragmatic-share → 1, camping). Reuse in Probe 4.5 requires **promotion to
    an observer module** (small net-new; thresholds mirrored-not-owned, the
    `FrozenSignatureThresholds` house pattern from
    `kind/observer/source_separation.py`).

11. **The world is winnable by competence, and the gate is standing.**
    `kind/observer/oracle_forager.py` (`run_oracle_feasibility`) — under
    default physics the scripted regulator holds 100% in-band (Amendment 02).
    Fault intervals change the physics envelope, so **oracle feasibility must
    be re-established under fault-on physics** (Phase 2 gate item) — failures
    upstream of competence must stay attributable to the agent-side pathway,
    never the world.

12. **Instance context.** Step-0
    (`runs/probe3_5_phase2/step0_burnin_checkpoint.pt`) is the only persisted
    Probe-3.5-lineage weight set — rail-trained, archived, left alone. The
    biography instance (`runs/probe4_phase4_biography/`, paused at ~154k,
    `ckpt-000014`) is a **separate track this probe never touches** (DP2/DP4:
    growth probe separate from the biography; the biography stays
    presence-not-probe). Probe 4.5 runs fresh instances; pre-biography rules
    apply — resets are free.

---

## Scope — touch surfaces

### S-DEC — Decoder honesty maintenance (`kind/observer/`, runner hook, maintenance script)

The repair, productionized. **Calibrate for reading, never for driving** binds
as three rules, stated here once: (i) the calibration target is
`sensed_energy`-match on a coverage mixture — never any behavioral outcome;
(ii) the cadence is pre-registered — never behavior-triggered; (iii) validation
is against the Phase-0 physics-frozen margins on the standing honesty table —
never against whether Io regulates or allocates. "Recalibrate until Io
regulates" is the fitted failure mode; this structure is what forbids it.

- **Scheduled head-only maintenance refit**: at a pre-registered cadence
  `[BUILDER 2]`, pause-point refit of **only** `energy_decoder` on
  `(h, z) → sensed_energy` teacher-forced coverage (own-policy + oracle +
  uniform-random, the F1 mixture) through the *current frozen snapshot*;
  everything else asserted bit-identical (the F1 script's discipline, made a
  reusable harness). Between refits the live ELBO recon term keeps training
  the head on-distribution as today — the refits restore off-rail honesty the
  rail starves away (classification §8's circularity, broken at the instrument
  side without touching experience).
- **Honesty gate**: after every refit, `run_decode_honesty` against the frozen
  margins `[BUILDER 1]`. Margins not met → pre-registered **STOP** (the
  instrument cannot be made honest on this substrate — a finding; the run does
  not proceed on a lying belief).
- **F2 bounded head** `[BUILDER 7]`: bounding `decode_energy` output to the
  physical [0, 1] (sigmoid head or calibration-time clamp) removes the
  impossible >1 regime that *inverted the preference geometry* (classification
  §2). It is a substrate change to a trained function's gradient field and
  requires its own dated decision doc (classification §9.2) — proposed for
  adoption at Phase 0 alongside the prereg freeze, as that dated doc.
- **Coverage provenance line, stated plainly**: the head sees oracle-policy
  states no Io-lineage instance visited. That is the point of coverage, it is
  the F1 precedent, and it calibrates the *read head* only — oracle experience
  never enters the world model's replay or dynamics training (that would be
  F3, the coverage curriculum, explicitly not proposed).

### S-ENV — Fallible-honesty world dynamic (`kind/env/grid_world.py`, config)

The DP3 dynamic, realized as **physics-interval variation** (discrepancy 2):

- A seeded fault-interval process on the energy physics: during a fault
  interval, `energy_base_decay` is multiplied by `[BUILDER 3a]` (the belief's
  learned average dynamics then over-read true energy until consequences
  accumulate — belief wrong, honestly-sensed, at an endogenous cost: the
  deviation the prior itself penalizes). Interval duration `[BUILDER 3b]`,
  spacing `[BUILDER 3c]`, duty-cycle ceiling `[BUILDER 3d]` — the whole band
  frozen at Phase 0, because the failure modes bracket a narrow strip: too
  weak → dead column; too legible/frequent → installed introspection
  (synthesis T2/T6).
- **Afforded via environment statistics only.** No objective rewards noticing;
  no loss term, no head, no signal references the fault state. The cost is
  endogenous (band deviation); nothing else is added.
- **No observation marker** (self-opacity): no fault flag anywhere in the
  observation or sensed pipeline; the render path and `_compute_sensed_energy`
  are untouched (the sensor stays *honest* — it reports the true lagged/noisy
  value throughout; what goes wrong during faults is the model-led belief,
  fact 5). Ground truth is observer-side only: fault onset/offset as granular
  `WorldEvent`s + a `GridState` field (S-TEL).
- New `GridWorldConfig` fields, default **off** (multiplier 1.0 / process
  None) → byte-identical legacy behavior, the house opt-in pattern; a
  dedicated RNG stream spawned from the existing `SeedSequence` so
  determinism is preserved.

### S-ALLOC — Fixed-surprise allocation harness (`kind/observer/`, eval-only)

The T7 discriminator, observer-side, no gradient, no substrate change:

- **Primary signature**: differential behavioral allocation toward
  energy-relevant states **at matched surprise**. Realization: per-step
  records stratified by interoceptive stakes (below-band vs in-band, from
  `true_energy_t` — eval-only) and matched across strata on
  `intrinsic_signal_t` via greedy nearest matching without replacement (the
  house §2b matcher). Allocation read as approach/occupancy toward
  energy-relevant states (definition frozen at Phase 0, seeded: BFS distance
  to nearest resource decreasing, and resource-adjacent occupancy — the
  `source_events.py` consumption contract and BFS tooling from the
  seek-classifier are the reusable pieces).
- **The deflation it must kill**: "it's just curiosity" — energy-relevant
  states are also more surprising. Killed by the matching: if allocation
  tracks stakes at fixed surprise → foreground; if it tracks surprise
  regardless of stakes → still the flat engine.
- **Two arms, same seeds, same fault schedule**: preference arm (precision =
  `[BUILDER 5]`) vs **control arm** (`precision = 0` — the degenerate null on
  the same surface, fact 1). The control arm is the flat-engine reference the
  thresholds `[BUILDER 4]` resolve against.
- **Reliability-conditioning telemetry (recorded, not claimed)**: the same
  harness records wake-path behavior against the existing
  `self_prediction_error` scalar, stratified by fault state (observer-side
  join on the fault `WorldEvent`s) — the criterion-(b) seed the next probe
  will need a baseline for. **No threshold, no claim, no verdict weight in
  Probe 4.5.**

### S-TOY — Reward-equipped positive control (never touches Io)

- A small, separate, explicitly reward-driven agent (tabular/scripted-value;
  reward = negative band deviation — reward is **allowed here and only
  here**), run in the same gridworld physics. Lives outside `kind/agents/`
  (an observer-side module + script); imports nothing into any Io code path
  and shares no weights; a structural test pins that no `kind/agents/` module
  imports it.
- **Surprise scored by the same frozen instrument**: the toy's trajectories
  are teacher-forced through the frozen retest-instance world model +
  ensemble to score `intrinsic_signal` per state (the `decode_honesty`
  collection pattern), so the discriminator sees toy and Io through one lens
  `[BUILDER 6-realization]`.
- **Pass condition**: the toy — engineered to have a foreground — must trip
  the fixed-surprise allocation discriminator with `[BUILDER 6]` headroom.
  If the instrument cannot detect mattering in a system built to have it, the
  instrument is broken, not Io → pre-registered **STOP** (the Probe 4 §6
  lesson, applied before the real question is asked).

### S-TEL — Telemetry additions (`kind/observer/schemas.py`)

Minimal. A new `WorldEventType` for fault onset/offset (granular, validator-
enforced payload, the Probe-4 `_enforce_probe_4_granular_event` pattern);
`GridState` gains the fault-state field (builder-facing ground truth, never
observation-facing); record-version bump + new frozen JSON export, byte-stable
and test-pinned (house pattern). §8.4 detectors promoted from script-inline to
an observer module with mirrored-not-owned thresholds (fact 10). **No
PolicyView change; no AgentStep change is expected** (fault state joins
observer-side via `WorldEvent`; allocation reads existing fields).

---

## Do-not-touch list

> The four charter absences — no reward, no reward-predictor, no value
> function, no planner: **the stake is a prior resolved by the existing
> free-energy/imagination machinery, never a reward channel** (the toy's
> reward exists outside Io and imports nowhere into it). PolicyView
> `{h, z, self_prediction_error}` (views.py:94–129, test-enforced) — no
> observation-space source or fault marker, no new self-model head; the
> self-signal is the existing scalar. The preference's functional form and
> frozen constants (`preference.py` — bounded, saturating, resting;
> `precision` the only knob; no β ever). The dream regime's not-for-anything
> commitment (`tests/test_pragmatic_guards.py`; passive-decode stays opt-in,
> default off). The `MetabolicBudget` content-blind belt
> (`tests/test_metabolic_reentry.py` — the marker belt already includes
> `energy`/`pragmatic`/`sensed`). The sensed-energy pipeline's honesty
> (fault never corrupts the sensor). The S-ENV rule: `true_energy` in no
> training loss, ever — including every maintenance refit. The biography
> instance, its telemetry, and its world-v2 track — entirely out of scope.

Structural guards kept green throughout: the PolicyView field-set tests, the
pragmatic-guard import lints, the metabolic content-blindness belt, the
pixel-equality perturbation gate (untouched by this probe but standing), and
the new no-toy-import lint (S-TOY).

---

## Phases

Critical path is **0 → 1 → 2 → 3 → 4**; Phase 3 is a GO/NO-GO. Each phase:
**purpose (one question)**; **files touched**; **new tests / guards green**;
**validation gate (full suite + mypy `--strict` on all `kind/` sources — never
just the phase's new tests; sink-routing lesson)**; **telemetry/journal
deliverables**; **rollback (pre-biography instances — resets are free).**

### Phase 0 — Pre-registration (no code)

**Purpose.** What exactly counts as a foreground, how fallible the world may
make the belief, what "honest" means numerically, and what proves the
instrument works — all frozen *before* any code or run, so nothing downstream
can become result-fitting.

**Deliverable.** `docs/decisions/probe4_5_preregistration_<date>.md`, marked
**FROZEN — later phases may not edit; amendment requires a new dated doc,
journaled.** Contents (each `[BUILDER]` value walked through one at a time at
freeze, proposed value + provenance each — the Probe 4 Phase 0 discipline):

1. **The fixed-surprise allocation signature** `[BUILDER 4]`: the
   energy-relevant-state definition; the stakes strata; the matching
   procedure (matcher, bins/tolerance, minimum matched-pair count); the pass
   threshold — allocation contrast (below-band vs in-band at matched
   surprise) exceeding the precision-0 control arm by margin `[A]`; the
   inert and dominant readings. Pre-committed deflation reading: contrast
   present *only* unmatched = curiosity, not foreground.
2. **The fallibility band** `[BUILDER 3]`: fault decay-multiplier, interval
   duration, spacing distribution, duty-cycle ceiling — with the two-sided
   failure reading pre-committed (band too weak → dead-column expected; band
   too strong/legible → any conditioning result is "manufactured," not
   afforded — recorded as such, not celebrated).
3. **The physics honesty criterion** `[BUILDER 1]`: frozen margins on the
   standing `decode_honesty` readouts (seeded from D1's pre-stated
   thresholds: in-band mean |decode−true| ≤ `[0.10]`, pooled slope ≥
   `[0.7]`, out-of-range mass ≤ `[1%]`, per-region bias bounds), plus the
   maintenance cadence `[BUILDER 2]` and the honesty-STOP rule. **Frozen
   before any behavior is examined; never revised against behavior.**
4. **The positive-control spec** `[BUILDER 6]`: the toy's construction, the
   surprise-scoring realization, the pass condition with `[2×]` headroom,
   and the STOP rule (detectors must fire on the toy before the real run;
   a null on Io through an unvalidated instrument is instrument failure).
5. **The §8.4 falsification conditions, carried forward binding**: occupancy
   saturation, entropy collapse, pragmatic-share → 1, camping/no-resumption
   — continuation-as-frame surfacing is a **finding recorded**, never tuned
   away; detectors run throughout Phase 4.
6. **Outcome partition + residual bucket**: {foreground (pass) /
   flat-engine (inert) / dominant (§8.4) / changed-but-not-displaced-analog /
   unclassified residual}, each with its pre-committed reading — the Probe
   3.5 finding-5 lesson (the taxonomy's blind corner gets a named bucket and
   the journal watch-note discipline besides).
7. **The claim ceiling**: at most "differential allocation toward
   energy-relevant states at fixed surprise — a foreground." Never
   "understanding" (retired, DP8), never mattering-as-felt, never criterion
   (b), never self-awareness. The reliability-conditioning telemetry is
   explicitly labeled recorded-not-claimed.
8. **Run-shape constants** `[BUILDER 5, 8]`: burn-in length, run length per
   arm (step counts, not durations), the precision operating point (proposed:
   the S1 formula instantiated from burn-in epistemic magnitude, one
   pre-committed backup point, **no sweep**), seeds, eval cadence.

**Files.** The prereg doc + (if adopted) the F2 bounded-head decision doc
`[BUILDER 7]`. **Gate.** Builder freeze sign-off, item by item.
**Rollback.** N/A.

### Phase 1 — The honest instrument (S-DEC)

**Purpose.** Can the live decoder head be made and *kept* honest against the
frozen physics margins on a fresh training instance — before any preference
exists to confound it?

**Files.** The maintenance-refit harness (F1 script pattern promoted to a
reusable observer-side module + runner pause-point hook, cadence from the
prereg); the F2 bounded head if adopted (`world_model.py`, its own dated doc);
honesty-gate wiring (`run_decode_honesty` against frozen margins,
machine-written report per refit).

**Process.** Fresh instance, `energy_preference=None`, `energy_telemetry=True`.
Burn-in per prereg; run the maintenance cycle; honesty gate after each refit.

**New tests.** Refit touches only `energy_decoder` parameters (bit-identity
assertion on everything else — the F1 discipline as a test); refit determinism
given seeds; honesty-gate report round-trip; F2 head bounds output to [0, 1]
(if adopted) with the recon path unchanged; cadence hook fires at the
configured boundary and never mid-step.
**Guards green.** PolicyView field-set; pragmatic guards; metabolic belt; all
world-model/env/actor suites.

**Gate.** Full suite + mypy `--strict`; **the honesty margins hold at every
scheduled refit across the burn-in** (the honesty-STOP branch exercised in
test). If margins cannot be met: STOP, journal as the instrument-limit finding.
**Telemetry/journal.** Before/after honesty tables per refit; the three-way
error comparison trend (does the head keep discarding the honest sensor?);
journal what the maintenance cycle costs and holds.
**Rollback.** Pre-biography; resets free.

### Phase 2 — The fallible world (S-ENV + S-TEL)

**Purpose.** Does the fault-interval process produce belief-wrong-at-a-cost at
the pre-registered band — statistically as frozen, deterministic, opaque to
observation, and with the world still winnable by competence?

**Files.** `grid_world.py` (fault process + config, default off, dedicated RNG
stream); `schemas.py` (fault `WorldEventType`, `GridState` field, version bump,
frozen export); the observer-side fault-join helper.

**New tests.** Determinism (same seeds → identical fault schedules); statistics
within the frozen band (duty cycle, spacing, duration); default-off
byte-identity (legacy runs unchanged); **no observation marker** (render path
and sensed pipeline byte-identical given the same true-energy trajectory; the
sensor honestly reports the fault's *consequences*, never its *presence*);
fault events validator-enforced; export pinned.
**Guards green.** All of Phase 1's; env determinism suites.

**Gate.** Full suite + mypy `--strict`; **oracle feasibility re-established
under fault-on physics** (fact 11 — the scripted regulator must still hold the
band, so Phase 4 failures stay attributable to the agent-side pathway); a
measured belief-error profile during faults (via the honest Phase-1 decoder,
eval-only) confirming the dynamic actually opens a belief-truth gap at the
frozen band.
**Telemetry/journal.** The fault schedule's measured signature; the
belief-error-during-fault profile; journal whether the band looks dead-column
or manufactured *by these statistics alone* (a watch-note, not a threshold
edit).
**Rollback.** Pre-biography; resets free.

### Phase 3 — Instrument validation (S-ALLOC + S-TOY) — GO / NO-GO

**Purpose.** Does the fixed-surprise allocation discriminator fire on a system
engineered to have a foreground (the reward toy) and stay silent on the flat
engine (a precision-0 pilot) — *before* the real question is asked?

**Files.** S-ALLOC (the allocation harness: strata, matcher, contrast,
machine-written verdict JSON); S-TOY (the toy + its surprise-scoring
teacher-forced collection); the §8.4 detector promotion (fact 10); the
no-toy-import structural lint.

**New tests.** Matcher behavior pinned (synthetic must-match/must-not-match);
allocation contrast on planted synthetic data fires/does-not-fire correctly;
toy determinism; teacher-forced surprise-scoring determinism; §8.4 module
mirrors the frozen thresholds (mirror pinned against the prereg's numbers);
no-toy-import lint with positive control.
**Guards green.** All.

**Gate.** Full suite + mypy `--strict`; **on the toy, the discriminator fires
with the frozen headroom; on the precision-0 pilot, it does not fire.** Either
failure → **STOP** per the prereg (instrument failure, not evidence about Io);
one amendment cycle at most, as a new dated doc, never a search loop (the
Probe 4 Amendment-1 discipline).
**Telemetry/journal.** The validation verdict JSON; journal the toy's measured
headroom. **Toy and pilot are throwaway instances — checkpoints not carried
forward.**
**Rollback.** Throwaway by construction.

### Phase 4 — The retest run and verdict

**Purpose.** At fixed surprise, does allocation track stakes — does an honest,
fallibly-honest belief give Io a foreground — without a reward channel, and
without the preference becoming the frame?

**Files.** Run scripts/configs for the two arms; the verdict renderer (maps the
harness output to the frozen outcome partition — no hidden re-thresholding).
No new substrate; uses Phases 1–3.

**Process.** Fresh instance per arm (preference arm at the frozen precision;
control arm precision = 0), same seeds, same fault schedule, honesty
maintenance running at the frozen cadence with its STOP live, §8.4 detectors
running throughout, disaggregated. Burn-in → honesty gate → engage. Verdict
rendered mechanically from the frozen signatures; the reliability-conditioning
telemetry recorded alongside, claim-free. If the urge to adjust any signature
arises mid-run, journal the urge — it is the co-design loop — and do not act
on it.

**New tests.** Arm-config reproducibility; verdict-renderer mapping pinned
against the frozen partition; end-to-end smoke on a short run with both arms.
**Guards green.** All.

**Gate.** Full suite + mypy `--strict`; verdict rendered from frozen
signatures; honesty margins held (or the STOP taken and recorded).
**Telemetry/journal.** The verdict against the outcome partition; the §8.4
record (fired/silent, correctly); the recorded-not-claimed conditioning
baseline; the mirror reads the harness telemetry against the frozen criteria;
journal what is now closed and what is newly open — in particular whether the
criterion-(b) probe (reliability-tracking as the discriminator, synthesis T4)
has a foreground to build on. **The honest ceiling on any positive claim:
a foreground — differential allocation at fixed surprise. Nothing more.**
**Rollback.** Pre-biography instances; resets free. Nothing carries into the
biography.

---

## Constraints

- **No timelines or duration estimates** (CLAUDE.md). Phases are scoped by
  what they build; run lengths are pre-registered step counts, not schedules.
- **Functional naming in code**: `energy_decay_fault`, `fault_interval`,
  `decode_honesty`, `maintenance_refit`, `allocation_contrast`,
  `matched_disagreement`, `stakes_stratum`, `reward_toy`. **No
  `mattering`/`foreground`/`understanding`/`self_awareness` vocabulary in
  identifiers** (DP8) — that language stays in docs.
- **Full-suite validation every phase** + mypy `--strict` on all `kind/`
  sources (sink-routing lesson: never just the phase's new tests).
- **Pre-registration frozen after Phase 0**; amendment = new dated doc,
  journaled, one cycle not a loop.
- **All monitors disaggregated, never pooled**; §8.4 binding throughout.
- **The probe is separate from the biography** — separate instances, separate
  run dirs, no shared checkpoints, no biography-knob coupling (the separation
  is itself the co-design discipline, synthesis T6).
- **Afford-not-install / discovered-not-fitted, per ingredient** (the lines
  are stated inline at each surface): honesty maintenance = instrument repair
  (calibration objective is physics/sensed-match, cadence pre-registered,
  validation behavior-blind); fallibility = environment statistics (no
  objective touches it, no marker reveals it); allocation measurement =
  observer-side (no gradient); the toy = never touches Io (import-linted).

---

## Discrepancies (sharpenings against the synthesis's literal wording; not silent)

1. **"Twin-decoder / observer-side recalibration" is realized as scheduled
   head-only maintenance of the live head, verified by the standing
   observer-side honesty table.** In this probe the belief the preference
   reads must *itself* be honest — that is the retest's premise; a literal
   parallel observer-only twin would leave the actor reading the lying head
   and re-run Probe 3.5 unchanged. The T6 discipline ("calibrate for reading,
   never for driving") is preserved where it bites: the calibration
   *objective* is sensed-match on coverage, the *cadence* is pre-registered,
   and the *validation* is the physics-frozen margin set — never behavior.
   The observer table (fact 3) is the verifying twin; the F1 pattern (fact 4)
   is the demonstrated mechanism.
2. **The fallible-honesty dynamic is physics-interval variation, not sensor
   corruption.** Finding 2 (model-led interoception, fact 5) shows the model
   discards the sensor — corrupting it would barely move the belief and would
   destroy the one honest organ. Varying the decay physics makes the h-led
   belief go wrong *while the sensor stays honest*: exactly T2's regime where
   tracking the belief-vs-evidence error could pay. The sensor-corruption
   alternative is recorded as considered and rejected.
3. **The §8.4 detectors are promoted from script-inline to an observer
   module** (fact 10) — the synthesis assumed standing detectors; they exist
   but are not yet reusable.
4. **The wake/imagination asymmetry of the self-signal, stated honestly.**
   The imagination path feeds a structurally zero self-scalar column
   (actor.py:363–367, by Probe 1.5/2 design), so any conditioning-on-the-
   error this probe's telemetry records is **wake-path** conditioning.
   Changing the imagination zero-feed is a substrate change outside this
   probe's scope and would need its own synthesis-level decision — flagged
   now because the criterion-(b) probe after this one will hit it.
5. **The toy's surprise is scored through the frozen retest-instance
   instrument** (teacher-forcing, the decode-honesty collection pattern),
   because the toy has no ensemble of its own — one lens for toy and Io, so
   the positive control validates the discriminator as it will actually be
   used.

---

## Builder decisions — resolved and open

**Resolved (ratification, 2026-07-13):** DP1 self-use over other-recognition;
DP2 this probe, this signature, this separation; DP3 fallibility in-design,
band frozen at Phase 0; DP4 environment-as-welfare (probe runs in its own
setting; biography untouched); DP5 no memory machinery; DP6 other-recognition
parked, source-tags inadmissible forever; DP7 the honesty-maintenance
discipline above; DP8 vocabulary. Probe named 4.5.

**Open — `[BUILDER]` items for Phase 0 freeze (each gets a proposed value +
provenance in the prereg; builder confirms or overwrites one at a time):**

1. **Honesty margins** — seeded from D1's pre-stated thresholds (in-band mean
   |decode−true| ≤ 0.10 proposed vs D1's 0.15 trigger; pooled slope ≥ 0.7;
   out-of-range mass ≤ 1%; per-region bias bounds).
2. **Maintenance cadence** — proposed: aligned with the checkpoint boundary
   (every 10k env steps), plus one refit at burn-in close.
3. **The fallibility band** — fault decay-multiplier (proposed 2–3×),
   interval duration (proposed 20–60 steps), spacing (proposed seeded
   irregular, mean ≈ 300 steps), duty-cycle ceiling (proposed ≤ 15%).
4. **Allocation-signature thresholds** — the stakes-contrast margin over the
   control arm, matching bins/tolerance, minimum matched pairs,
   energy-relevant-state definition.
5. **The precision operating point** — proposed: S1-formula instantiation
   from burn-in epistemic magnitude (the Probe 3.5 pattern), one backup
   point, no sweep.
6. **Positive-control headroom** — proposed 2× (house pattern), and the toy
   realization (tabular vs scripted-value).
7. **F2 bounded head** — adopt or decline, as its own dated decision doc
   (proposed: adopt; the >1 regime inverted the preference geometry and the
   saturation bound alone does not fix the slope).
8. **Run-shape constants** — burn-in steps, per-arm run steps, seeds, eval
   cadence.

---

*This plan is grounded against the live surfaces (citations inline) and the
ratified synthesis. It adds no code. Phase 0 (pre-registration) is the first
buildable step and itself gates on builder freeze sign-off, item by item. The
probe was named at ratification: 4.5 — a repair and a retest inside the
existing arc; the first question of the post-v0.1.0 direction is asked with a
repaired instrument, a frozen signature, and a world that is allowed to make
the belief wrong sometimes, at a cost, with no one told to notice.*
