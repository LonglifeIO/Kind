# Probe 3.5 — Valence substrate — working journal

Boundary probe (DP1=b): *can a bounded, saturating, non-terminal homeostatic
preference over a sensed energy channel create an energy-dependent explore/conserve
trade-off for Io without becoming the evaluative frame — and can the telemetry/mirror
detect when it does?* Both outcomes are findings. Authority: the adopted synthesis
`docs/decisions/synthesis_probe3_5_valence_substrate_2026-06-09.md`; build plan
`docs/plans/Kind_probe3_5_implementation_plan.md`.

## Phase 0 — Pre-registration (no code)

The probe's load-bearing discipline (drawn from F2's co-design mitigation, not the
research): fix what counts as success *before* searching for it, so the parameter
sweep cannot become result-fitting. This session produced the freeze-ready draft and
a post-validation amendment pass on the plan. No code; Phase 1 begins in a later
session, after the builder freezes the draft.

### What this session did

Two documentation deliverables.

**1. Amended the implementation plan (post-validation 2026-06-10).** Four amendments,
each logged at the bottom of the plan:

- **Recon target is `sensed_energy`, explicitly.** The energy reconstruction loss
  targets `sensed_energy` (the normalized sensed scalar), not an unqualified
  "normalized target." Stated as a rule: **`true_energy` never enters any training
  loss** — it lives in exactly two places, `GridState` telemetry and the observer-side
  eval probes. This closes the one ambiguity that could have let ground truth leak into
  a gradient: dead-path battery item A probes latents *against* `true_energy` without
  training on it, which is correct precisely because it is eval-only.
- **Marker belt extended (test-side only).** Phase 2's "reaffirm energy never reaches
  `MetabolicState`" test is realized by adding `energy`, `pragmatic`, and `sensed` to
  `_IO_DERIVED_NAME_MARKERS` in `tests/test_metabolic_reentry.py` (all three were absent
  from the existing belt). This makes an energy-named scalar field on `MetabolicState`
  unrepresentable-to-add — the same structural guarantee that already forbids
  `intrinsic`/`latent`/etc. The `MetabolicBudget`/`MetabolicState` surface itself stays
  untouched (DP2 / F6 intact).
- **Baseline collection protocol.** Phase 0's deliverable gains the protocol for
  measuring the pure-epistemic baseline (the Phase-1 config: channel live, preference
  zero); Phase 1's deliverables gain "record the baseline metrics under the frozen
  protocol." This is what lets the assay thresholds be baseline-*relative* without being
  unfrozen.
- **Open decisions resolved.** The seven builder-open items collapsed: dream telemetry =
  passive-decode (record `decode_energy`, observer-side only, F5 intact); energy carries
  across the soft 200-step boundary; pragmatic/epistemic share on optional `AgentStep`
  fields; record-level version = a fresh `PROBE_3_5_TELEMETRY_SCHEMA_VERSION`; the DP9
  escalation threshold relocated into the pre-registration doc as a bracketed item; the
  remaining structural values (`energy_recon_weight`, embed dim, decoder width, exact
  saturating form) stay build-time fixed choices, journaled. Only `precision`/`sigma`/`lag`
  are swept.

**2. Wrote the pre-registration draft** `docs/decisions/probe3_5_preregistration_2026-06-10.md`.
DRAFT status with a freeze line the builder flips; after freeze, no edits — amendment
only via a new dated doc. The discipline that makes it freeze-ready: **every threshold
is a bracketed `[BUILDER: value | provenance]` field** carrying a proposed value plus
one-line provenance (synthesis seed / standard practice / arbitrary-but-pre-committed),
so confirming takes minutes, not research, and no field is ever a bare blank.
Baseline-relative thresholds are frozen as ratios/margins (e.g., entropy ≥ 70% of the
pure-epistemic baseline); the reference value they multiply is measured in Phase 1 —
stated explicitly as *not* threshold editing. Contents: the three assay signatures
(graded scarcity, novelty-vs-replenishment, recovery-after-depletion) with pass /
dominant / inert criteria; the dead-path battery A–E; the baseline protocol; the
pre-committed sweep (grid bounds, precision-first order, stopping rule including the
tiny-tensor double-bind finding); the §8.4 falsification set verbatim; the DP9 trigger;
the two resolved sub-decisions; and a closing checklist of exactly the bracketed items
to confirm.

### What is now closed

- **The recon-target ambiguity.** `sensed_energy` trains; `true_energy` never enters a
  loss. The fourth dead-path locus (the energy channel) now has its training/eval
  boundary stated as a rule, not left to build-time interpretation.
- **The seven plan-open builder decisions** — all resolved or relocated into the
  bracketed pre-registration fields. The plan's "open" section is now a "resolved"
  section; the only remaining builder action is confirming the bracketed thresholds.
- **The co-design hazard's structural answer.** The pass/dominant/inert signatures and
  the dead-path margins are written down *before* the sweep exists. The dominant
  signatures are gathered as the §8.4 falsification set: their appearance is
  continuation-becoming-the-frame, a recorded finding, **never a tuning target**. The
  freeze line is the commitment device.

### What is newly open

- **Builder freeze sign-off.** The draft is DRAFT until the builder confirms/overwrites
  the bracketed items (listed at the end of the prereg doc and in the prior session's
  hand-off) and flips the freeze line. Phase 1 does not start until then.
- **The tiny-tensor double-bind (still the live empirical risk).** Whether a learnable,
  non-dominant `precision`/`sigma` window even *exists* in Io's `h=200, z=16`, 8×8
  regime is unvalidated — "every tuning is inert or dominant" remains a live possibility
  (synthesis S1/§7). The sweep's stopping rule pre-commits to recording that as a
  *finding*, not extending the grid to hunt for a window.
- **Can an amortized actor express the contextual switch at all?** The assays demand
  novelty-when-sated / resource-when-low; Io's actor is reactive (amortized,
  DreamerV1-lineage), not a per-decision planner. Worth the early check the synthesis
  flags (§7) before any tuning.
- **Phase 1 deliverables** — the channel-without-preference build, the full dead-path
  battery A–D as the gate, and the baseline measurement under the frozen protocol.

### Deviations / flags

- **The synthesis is unchanged.** These amendments sharpen the *plan*, not the decisions;
  all nine DPs stand as adopted.
- **The pre-registration is a DRAFT, not frozen.** Deliberately — the freeze is the
  builder's, and is the load-bearing co-design commitment. Pasting proposed numbers is
  not the same as committing them.
- **Proposed thresholds carry provenance, not authority.** The `(pre)`-tagged numbers
  are arbitrary-but-pre-committed: defensible, chosen in advance, and explicitly *not*
  claimed to be uniquely correct — the point is the commitment exists before the sweep.

### Watts / new-interface entry

`new_actor_readable_interfaces_added = []`. No code this session. The marker-belt
extension *removes* a representable surface (an energy-named `MetabolicState` field
becomes impossible) rather than adding one; it touches no Io read path. PolicyView stays
frozen at `{h, z, self_prediction_error}`.

### Freeze

**2026-06-10 — pre-registration FROZEN** (builder: Gordon). All §8 bracketed items
confirmed as proposed (no overrides); the battery retry rule added at freeze (one retry
at 2× P3 = 10000 env-steps before DP9 escalation). `probe3_5_preregistration_2026-06-10.md`
is now the immutable Phase-1 gate; further changes require a new dated doc. Phase 1 build
begins.

## Phase 1 — Channel without preference

**Purpose (one question).** Does the world model demonstrably learn a *world-grounded*
energy channel with `pragmatic_value` still zero — so channel-learning and preference
effects are never confounded? **Answer: the channel is learnable *in principle* but not
cleanly learned under the pure-epistemic regime — an informative failure (the tiny-tensor
double-bind, synthesis §7), recorded not tuned away.** Full results + the baseline
instantiation: `docs/decisions/probe3_5_phase1_baseline_2026-06-10.md`.

### What was built

Four touch-surfaces, all behind the opacity boundary (PolicyView frozen at
`{h, z, self_prediction_error}`, unchanged — energy enters only the world model, DP4):

- **S-ENV** (`grid_world.py`, `transport.py`): `true_energy ∈ [0,1]` normalized by fixed
  config constants (setpoint 0.6 at `energy_init=0.6·norm_max`); depletion = base decay +
  movement cost (`stay` cheaper); replenishment on resource *entry* (reuses the
  entry-triggered consumption); floor at 0, **no terminal state**, energy **carries across**
  the soft 200-step boundary (`reset()` is the only re-init). `sensed_energy` = the coarse,
  noisy (σ), lagged (1–2), lightly-quantized scalar, driven by a **third RNG stream**
  spawned from the env `SeedSequence` (the first two children — regrowth, drift — stay
  byte-identical, so pre-3.5 env determinism is preserved; test-pinned). `EnvStep.sensed_energy`,
  `GridState.true_energy`.
- **S-WM** (`world_model.py`): `_EnergyEncoder` (1→embed) fused into the posterior
  (posterior input widened to `h+embed+energy_embed`); `_EnergyDecoder` (`h+z→1`) +
  `decode_energy`; `WorldModelStep.energy_pred`; `loss` gains `energy_recon_loss` — **target
  is `sensed_energy`, never `true_energy`** (the S-ENV rule; true energy enters no training
  loss, eval probes only), up-weighted 10×, exposed separately. The EMA image encoder, GRU,
  and self-prediction head are untouched.
- **S-TEL** (`schemas.py`, `views.py`): fresh `PROBE_3_5_TELEMETRY_SCHEMA_VERSION = "0.4.0"`
  with four optional energy fields + a version-gated validator (older shards stay
  backward-readable); frozen export `schemas/v0.5.0.json` (and the prior v0.4.0 export was
  frozen-to-bytes, now a historical artifact). `TelemetryView.energy_pred` (mirror-side
  only). Energy telemetry is an explicit `RunnerConfig.energy_telemetry` opt-in (default off
  → every existing runner/smoke emits "0.2.0" byte-identically).
- **Runner**: fuses `sensed_energy` into `world_model.step` on every step (live + replay
  training); the replay `Transition`/`Batch` carry per-step `sensed_energy`; **actor
  untouched** — `pragmatic_value` stays zero this phase.
- **Eval harness** (`kind/observer/energy_eval.py`): the dead-path battery A–D at the
  frozen pre-registration margins (`DeadPathMargins`), plus a teacher-forced latent
  collector.

### Battery results (the gate) — and the finding

Run on the real trained model (P3 = 5000) and cross-checked with a variance-rich
controlled probe:

- **A (latent-predictability) and C (action-history ablation) PASS on a variance-rich
  distribution** (R² = 0.55 ≥ 0.5; MSE ratio 2.34 ≥ 1.5): the WM decodes `true_energy`
  from `[h, z]` and far better than from action history — **world-grounded, not an
  action-history artifact.**
- **B and D FAIL** (responsiveness ≈ 0; per-dim KL max 0.97–1.26, never reaching the
  1.5-nat margin — the posterior runs sub-1-nat). Diagnosis: **energy rides the
  deterministic recurrent state `h`** (which integrates decay + resource entries), so the
  fused *observation* is informationally **redundant** and the stochastic latent doesn't
  carry it.
- **DP9 escalation (dedicated z-dims + weaker z-only decoder) degrades the channel**
  (A → R² −0.64, C → 0.64) while only nudging D (0.97 → 1.35, still < 1.5). The
  **tiny-tensor double-bind** made concrete: every routing is *predictable-via-`h`* or
  *noisy-via-`z`*.
- **The pure-epistemic actor floors energy** (eval: `true_energy` mean 0.011, in-band
  0.4%). With no preference there is no reason to maintain energy; it depletes to the
  floor. The battery on that degenerate distribution is uninformative (A/C against a
  near-constant target). The double-bind recurs in the env dynamics: fast-enough-to-vary
  energy floors without foraging, and foraging needs the very preference Phase 1 omits.
  **Energy becomes a live, maintained variable only with a reason to maintain it — Phase 2.**

The pre-registered retry/escalation was applied in spirit: retry at 2×P3 cannot fix a
*behavioral* floor; DP9 was taken, measured, and recorded (degrades), not adopted. **The
env was NOT recalibrated to green the battery** — that is the co-design loop the freeze
exists to prevent (§8.4 / DP1b: a recorded finding, never a tuning target).

### What is now closed

- The substrate is built and integrated: full suite **1235 passed / 6 skipped**, mypy
  `--strict` clean (65 sources). The opacity boundary held (PolicyView frozen; the
  marker-belt forbids energy on `MetabolicState`; energy telemetry is mirror-side only).
- The energy channel is **demonstrably decodable and world-grounded** (A, C) — the T6
  dead-path "channel never learns at all" is *not* what happened.

### What is newly open (Phase 2 is gated on these — builder decisions)

- **Env energy-economy recalibration** so energy is a live in-band variable under
  epistemic-only behavior (or an explicit decision that energy is only "live" under a
  preference). Without it the baseline band/σ-grid (B0a/S2) **cannot be instantiated** —
  the measured references are contaminated by the floor.
- **Do batteries B/D fit a substrate where energy is recoverable from `h`?** The
  observation channel is redundant here; B/D presuppose the observation is the primary
  route. Re-examining them is an *amendment via a new dated doc* (the pre-registration is
  frozen).
- **Force energy through `z` (DP9) despite the A/C cost?** Measured to degrade; recorded.

### Deviations / flags

- **Battery B realized as an energy-observation input-sweep** (responsiveness of
  `decode_energy` to the fused sensed input), the amortized-model reading of "interventional
  response / S2 action-lesion." Flagged: a model that infers energy from the world rather
  than copying the noisy observation will read low on this — which is what happened, and is
  itself the redundancy finding.
- **DP9 escalation implemented** (`WorldModelConfig.energy_dedicated_dims`) as the
  pre-registered contingency; default 0 (plan's `h+z` decoder). Taken and journaled per the
  DP9 instruction; degrades A/C, not adopted.
- **Build-time energy magnitudes** (decay/move/replenish) were tuned on uniform-random
  behavior (good variance) and turned out to floor under the trained actor — the §4
  recalibration item. Journaled as a build-time choice, not a frozen-decision change.

### Watts / new-interface entry

`new_actor_readable_interfaces_added = []`. Energy enters **only** through the world-model
observation pathway (the fused proprioceptive branch into `h, z`); PolicyView is unchanged
and its guard tests stay green. The actor never reads any energy quantity directly; the
mirror-side `TelemetryView.energy_pred` and the AgentStep energy fields are observer-side
only.

## Phase 1 (cont.) — Amendment 01, recalibration, amended gate

**Purpose.** Execute the post-Phase-1 amendment + recalibration sequence: re-aim the
Phase-1 gate to the substrate's actual route (energy is model-led via `h`, not
sensor-led via `z`), recalibrate the env energy economy so an indifferent agent's
energy is non-degenerate, re-run the amended gate, and (if green) instantiate the
baseline. **Answer: the amended gate is not green; the recalibration did not transfer
to a trained instance; the baseline was not instantiated — a recorded finding.** Full
results: `docs/decisions/probe3_5_recalibration_amendedgate_2026-06-10.md`.

### Amendment 01 — confirmed

`probe3_5_preregistration_amendment01_2026-06-10.md`, **CONFIRMED 2026-06-10
(builder: Gordon)**, all bracketed items as proposed (no overrides): B → B′
(imagination intervention; B′1 = 80% of pairs, B′2 = 0.5× normalized replenish);
D → retired as a gate, retained as a permanent monitor; E → E′ (estimate-lesion,
Phase 2+; Be = 0.20 carried over); recalibration target R3–R5 (R1 = 8, R2 = 20,
R3 std ≥ 0.10, R4 mean ∈ [0.30,0.70], R5 floor ≤ 0.10). The frozen pre-registration
stays byte-frozen; the amendment is the only place the changes live. The **retry**
(one at 2× P3) was **waived** for the B/D failures and recorded in the amendment: the
Phase-1 failure was behavioral-floor + substrate-routing, not insufficient training
age, so P3 = 5000 stays the operative age.

### What was built

- **B′ + amended gate** in `kind/observer/energy_eval.py` (the frozen A–D
  `run_dead_path_battery` preserved verbatim for provenance): `battery_b_prime_…`
  rolls matched real latents forward one imagined step on a coincident vs control
  action (prior mean for `z'`), comparing `decode_energy`; `run_amended_gate` gates on
  A ∧ C ∧ B′ with D as a monitor. mypy `--strict` clean; 3 new tests (B′ zero-delta on
  identical actions; metric/determinism; gate excludes D).
- **Recalibration harness** (`scripts/recalibrate_probe3_5_energy.py`): trains one
  epistemic instance, records physics-invariant `(is_move, consumed)` flags, and
  re-simulates energy analytically per candidate triple. The analytic re-simulation
  **reproduces the env exactly** (max abs err 0.0) — validated before any candidate
  was trusted.
- **Gate+baseline harness** (`scripts/run_probe3_5_amended_gate.py`): retrains at the
  chosen physics, runs the amended gate on the epistemic distribution, and (if green)
  measures the baseline over 8 eval seeds × 20 episodes on the retrained instance.

### Recalibration + the transfer failure

- Analytic search: only **2 of 140** triples meet R3–R5, both at the **grid edge**
  (`decay = 0.01`, `move_cost = 0.005`). Chosen: `(0.01, 0.005, 4.0)` — analytically
  std 0.284, mean 0.609. Default physics re-simulated reproduces the Phase-1 floor
  (mean 0.006, floor 0.975). Incidental foraging is sparse (consume-rate 0.5%/step;
  the actor moves every step).
- **The chosen triple did not transfer.** On the instance *trained at the chosen
  physics*, the epistemic actor's energy is **ceiling-saturated** (mean 0.943, std
  0.073, 62.6% at ceiling) — **R3 and R4 fail**. The passivity premise is only
  approximate: energy is fused into `h, z`, which the actor reads, so an actor trained
  under live energy forages differently than the default-trained (floored-energy)
  actor whose trajectories drove the analytic selection.
- **Amended gate (on the saturated target): not green** — A R² ≪ 0, C ratio 0.60
  (history beats latents), B′ mean Δ = −0.017 (no imagination replenishment, no decode
  headroom above the ceiling), D max KL ~1.2 (sub-1.5, as Phase 1). Because the target
  is degenerate (ceiling-pinned), A/C/B′ are **uninformative about grounding** — the
  ceiling-side twin of the Phase-1 degenerate-floor caveat.

### What is now closed

- The **env-economy double-bind is confirmed at three levels**: energy is degenerate
  for an indifferent actor across the physics envelope — floored (fast physics),
  ceiling-saturated (gentle physics, trained instance *and* uniform-random), with the
  targeted live band surviving only as an **analytic artifact** of the energy-blind
  default actor. Env-only recalibration cannot manufacture a world-grounded, in-band
  interoceptive channel for an indifferent agent.
- This independently re-confirms the Phase-1 conclusion from a second direction:
  **energy becomes live/in-band only when regulated — a preference (Phase 2).** DP7
  "sustainable" is necessary but not sufficient; sustainability keeps the agent off the
  floor only by making depletion negligible, which saturates rather than centers.

### What is newly open (Phase 2 / next session — builder decisions)

- **Close the recalibration loop or move the band into Phase 2.** Passive-trajectory
  physics selection does not transfer; options are an iterate-until-the-*trained*-
  instance-meets-R3–R5 loop, or instantiating the band against a *regulated* (Phase 2)
  reference. A build-time choice / new dated doc — **out of scope here** (Step-4 "no
  further tuning in-session").
- **Does a pure-epistemic baseline in a non-degenerate form even exist?** If energy is
  in-band only under regulation, the frozen §3 pure-epistemic baseline may need to
  resolve against a regulated reference — a pre-registration amendment question.

### Deviations / flags

- **Three gate eval runs, as diagnosis not tuning.** (1) uniform-random distribution
  (degenerate — saturates); (2) epistemic distribution (the methodologically-correct
  one post-recalibration; substantively examined); (3) an instance-energy diagnostic
  that found the recalibration didn't transfer. **No threshold or physics was changed
  across them** — each run diagnosed *why* the gate fails (wrong distribution →
  saturation → transfer failure). The final reading attributes the failure to the
  degenerate (saturated) target from the transfer failure, **not** to a demonstrated
  channel-grounding failure (an earlier "action-history clock" reading was retracted as
  unsupported on a degenerate target).
- **Chosen triple not adopted.** It saturates a trained instance; it is recorded in the
  results doc, **not** written into `GridWorldConfig` (the default physics stay).
- **Physics-invariance premise documented as approximate.** Step-3's "the actor reads
  no energy quantity, so its trajectory is invariant to energy physics" holds for the
  *direct* read (PolicyView has no energy field) but not the *indirect* one (energy →
  `h, z` → actor) once trained at live physics.

### Watts / new-interface entry

`new_actor_readable_interfaces_added = []`. PolicyView stays frozen at
`{h, z, self_prediction_error}`; B′, the amended gate, and all recalibration/diagnostic
machinery are **observer-side eval only** (`true_energy` / `decode_energy` never enter a
training loss). The amendment *removed* a gate (D → monitor) and re-aimed another
(B → B′); it added no actor read path.
