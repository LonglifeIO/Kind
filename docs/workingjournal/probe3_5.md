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

## Amendment 02 — baseline reframed, oracle feasibility (pre-Phase-2)

**Purpose.** Correct the second falsified premise before Phase 2: the frozen §3
*pure-epistemic baseline* assumed an indifferent-but-centered energy distribution,
which Phase 1 + the recalibration session showed is **degenerate in principle**
(floored under fast physics; ceiling-saturated under gentle physics; mid-band only
as an analytic artifact of the energy-blind actor). No Phase 2 build this session.

### Amendment 02 — confirmed

`probe3_5_preregistration_amendment02_2026-06-10.md`, **CONFIRMED 2026-06-11
(builder: Gordon)**, all bracketed items as proposed, with one operationalization
added at confirmation (O1's "sustained at steady state" = in-band occupancy over
the **final 50% of each eval episode**, averaged across P1 × P2). Contents:

- **Builder decision: Option A — the pure-epistemic baseline *is* the degenerate
  distribution** (energy-blind, rail-pinned, ~0% in-band; rail identity recorded as
  a measurement). Rationale: the charter's own language ("not unmotivated or
  indifferent to its existence") never described an indifferent-but-centered agent;
  the reframe dissolves the circularity of measuring regulated behavior against a
  reference only regulation could produce; the contrast sharpens from
  band-occupancy-delta to **rail-versus-band**. The baseline **survives intact as
  the entropy/exploration reference** (all entropy-relative thresholds and the
  inertness criterion unchanged); it stops providing the energy-band reference.
- **Fixed units**: B0a′ band = 0.6 ± **0.15 absolute** ([0.45, 0.75]); S2′ σ grid
  = {0, 0.075, 0.15} (fractions of band halfwidth; σ=0 diagnostic-only). S1
  unchanged as a formula, instantiating from the epistemic magnitude at **Phase 2's
  pre-preference burn-in** (non-circular).
- **Phase 2 pass = rail → band displacement**: sustained in-band occupancy ≥ 50%
  (final-half-of-episode window) vs the ~0% null, entropy thresholds unchanged;
  §8.4 falsification signatures unchanged and binding.
- **Recalibration loop closed as impossible** (three-level evidence); not to be
  reopened without new evidence.
- **New pre-committed instrument — oracle feasibility**: a scripted
  nearest-resource forager (observer-side; not Io; no learning; policy
  pre-committed: BFS-to-nearest-resource below setpoint, stay otherwise) must hold
  in-band ≥ 70% over 8 × 20 before Phase 2 trains anything. If default physics
  fails, physics is selected by the oracle criterion and **presented for builder
  adoption, never self-applied**.

Both amendments are substrate-forced corrections to falsified premises, not scope
drift; no further pre-registration changes are expected before the Phase 2 build.
The frozen original and Amendment 01 remain byte-frozen.

### Oracle feasibility — RUN, PASS at default physics

Harness: `kind/observer/oracle_forager.py` (pure env instrument; imports only
`kind/env/grid_world.py`; touches no Io code path) + 6 tests +
`scripts/run_probe3_5_oracle_feasibility.py`. **Verdict: PASS — pooled in-band
occupancy 1.00; every seed 1.00** (8 seeds × 20 episodes = 32,000 steps, never
leaving [0.45, 0.75]) under the **default physics** (decay 0.08, move 0.04,
replenish 0.8). **Phase 2 proceeds at default physics**; the candidate search was
not triggered; no `GridWorldConfig` change proposed.

The same physics that floors the indifferent agent (in-band 0.4%) is held
perfectly in-band by a trivially competent regulator — the rail-versus-band
contrast is maximal (null ~0%, oracle 100%), and the Phase-2 question is now
cleanly posed: *does the preference produce the competence the world demonstrably
permits?*

### Propagation

The implementation plan gains an "Amendments — post-Amendment-02" section
(baseline meaning, fixed band/σ, oracle gate before Phase 2, rail→band pass
condition, recalibration closed). The plan's Phase 2 build prompt should be
generated against that section plus both amendments.

### What is now closed

- The §3 baseline premise is corrected; every threshold that survives is
  enumerated explicitly (entropy-relative ratios, inertness, collection
  mechanics); everything re-denominated is in fixed units. The band no longer
  depends on any measured std.
- **Environment feasibility is established at default physics** — a Phase-2
  failure is attributable to the preference/substrate, never to an unwinnable
  world.

### What is newly open

- **Phase 2 build** (next session): fill the scaffold per the plan + both
  amendments; pre-preference burn-in instantiates S1; the amended A/C/B′ gate
  instantiates on the Phase-2 instance **where energy varies** (the degenerate
  null cannot provide target variance by definition) — the exact gate timing
  belongs in the Phase-2 build prompt.
- **Probe 4 perturbation content is undecided.** Candidate design, logged for
  Probe 4's research pass — **not for action now**: a **two-class mixture** of
  builder perturbations — *need-keyed resource drops* (contingent on Io's energy
  state) and *need-neutral novel objects* (contingent but energy-irrelevant) —
  enabling a **contingency-vs-care contrast**: does Io's behavior (and the
  mirror's reading) distinguish a relational other that responds to *need* from
  one that merely responds *contingently*? Both classes must remain contingent
  (non-contingent drops are indistinguishable from weather). To be researched,
  not assumed.

  **Committed design constraint (injection topology — 2026-06-11, builder:
  Gordon).** Whatever the perturbation content turns out to be, **perturbations
  enter only through the runner's loop, via a queue drained at step
  boundaries** — no side-thread `client.mutate()` calls, and no direct
  in-process `env_server` mutator calls from outside the transport thread.
  **Rationale: required for yoked-replay timing reproducibility** — perturbation
  placement must be deterministic relative to TRANSITIONs, so a replay can
  reproduce *when* (in env-step time) a perturbation landed, not merely *that*
  it landed — **not merely race-safety**. (The race-safety half is now also
  enforced structurally: the transport client refuses concurrent requests with
  an immediate `TransportError` — the one-outstanding-request contract is
  unrepresentable-to-violate from the wire side.) This constrains Probe 4's
  *plumbing*, not its content; the content question above stays open for the
  research pass.

### Deviations / flags

- **O1 operationalized at confirmation** (final-50%-of-episode window) — recorded
  in the amendment §3, not a post-hoc choice.
- The oracle reads `GridState.true_energy` directly — deliberate and stated in
  the amendment: it is an instrument, not Io; no opacity constraint applies.
- Oracle pass at 1.00 leaves F1 = 70% with large headroom; if a future physics
  change makes the oracle marginal, the F1 floor (not the headroom) is the
  binding commitment.

### Watts / new-interface entry

`new_actor_readable_interfaces_added = []`. The oracle forager is observer-side
only (env-only imports; no Io code path); PolicyView stays frozen at
`{h, z, self_prediction_error}`. Amendment 02 touches references and gates, not
any actor-readable surface.

## Phase 2 — Fill the scaffold (mechanism gate)

**Purpose (one question).** Does the machinery for asking Phase 4's question
work — the preference implemented per DP5/DP6 and Amendment 02, the degenerate
baseline formally instantiated, the S1 grid instantiated, the guards provable,
the gradient path live? **Answer: yes — the mechanism gate is green.** No
behavioral result this session is the probe's answer; Phase 2's gate is
mechanism-level and the verdict belongs to Phase 4. Full record:
`docs/decisions/probe3_5_phase2_mechanism_2026-06-11.md`.

**Framing (from the oracle check, which bounded the experiment precisely).**
Same physics: 0.4% in-band for indifference, 100% for perfect-information
competence. Phase 2 sits in the gap — a mild preference that never reads
`true_energy`, only the world model's belief (`decode_energy`, R² ≈ 0.55 on its
best distribution), added to a curiosity term that currently owns the agent.
Whether that produces the competence the world permits is Phase 4's question;
Phase 2's question was whether the machinery for asking it works.

### What was built

- **Step 0 — the degenerate baseline, formally taken** (the before-photo,
  finally with its honest meaning). One epistemic-only instance at default
  physics, P3 = 5000; eval per the frozen mechanics (8 eval seeds × 20
  episodes × 200 steps; Phase-1 artifacts didn't satisfy them — run fresh).
  **Rail identity: floor** (floor-frac 0.973, ceiling 0.000); in-band 0.37%
  pooled, **0.00% on the O1 steady-state window** (the residual 0.37% is the
  once-per-seed initial transit from the 0.6 start through the band to the
  floor). Entropy references A1b denominates against: positional **0.575 ±
  0.438 nats/episode** (per-episode, per the frozen Shared definition — the
  Phase-1 1.63 nats was a whole-rollout statistic), epistemic activity
  **0.394 ± 0.050**/step. Standard condition only; per-condition baselines are
  Phase 4 prep.
- **Step 1 — the preference** (`kind/agents/preference.py`; actor scaffold
  filled). `d(e) = relu(|e−0.6|−0.15)`; `v(e) = −S·tanh(precision·d²/(2S))`,
  S = 1.0. Flat-with-zero-gradient in the fixed band; Gaussian log-preference
  in band-edge distance outside; saturating to the bound; **coefficient-free —
  precision is the weight, no β** (stated in code). Operates on
  `decode_energy` over imagined `(h, z)` inside the waking objective only.
  Three genuinely-unpinned items were surfaced bracketed and
  **builder-confirmed before implementation** (no silent choices): the
  saturation scale [SAT-1: S = 1.0 | pre]; the S1 "marginal magnitude"
  operationalization (slope reading — `precision × halfwidth`, the unique
  reading invariant under the Amendment-02 geometry change); the S1
  denominator ("at the band edge" unconditionable under the degenerate null →
  unconditional mean per-step disagreement, Phase-1 measurement precedent).
- **Step 2 — guards + telemetry.** Dream-path unreachability made structural
  *and* behavioral (`tests/test_pragmatic_guards.py`): import-lint over the
  offline regime (dream / dream_seed / state_machine / dream_session must not
  import the preference or reference `imagine_and_compute_loss`) with positive
  controls, plus a tripwire backstop — `energy_log_preference` monkeypatched
  to raise at both its definition site and the actor's bound name; full dream
  rollouts under both action policies complete untripped while the same
  tripwire fires on the waking path. Marker belt extended test-side
  (`energy`/`pragmatic`/`sensed`) over MetabolicState (DP2) and PolicyView
  (DP4) in the new guard file; **`tests/test_metabolic_reentry.py` stays green
  unmodified** (deviation from the plan's letter, which sited the extension
  inside that file — the build instruction's stricter reading won; same
  guarantee, original belt asserted to be a strict subset). Telemetry: fresh
  record version "0.5.0" with required `pragmatic_value_t` / `epistemic_value_t`
  / `pragmatic_share_t` (per-training-step decomposition; A2b's share is
  measurable); D monitor (per-dim KL) retained per Amendment 01;
  `schemas/v0.5.0.json` frozen to bytes, new frozen export `v0.6.0.json`;
  older shards backward-readable; **opt-in, default off** —
  `RunnerConfig.energy_preference = None` keeps every existing runner
  byte-identical.
- **Step 3 — S1 instantiated** (instantiation ≠ editing; formula untouched).
  Burn-in = the Step-0 instance (trained preference-off to P3 — exactly
  Amendment 02's instantiation point). E_typ = 0.39350 →
  `precision_k = k·E_typ/0.15` = **{0.2623, 0.8395, 2.6234, 8.3947, 26.2335}**.
  Values + provenance in the results doc and
  `runs/probe3_5_phase2/s1_instantiation.json`.
- **Step 4 — mechanism gate.** (a) Unit: in-band exactly zero value and
  gradient; correctly-signed pull both sides; saturation bounds at
  decoder-extrapolation extremes; precision-0 identically zero. (b)
  Integration: pragmatic gradient reaches the policy through the imagined
  trajectory; **precision = 0 reproduces the Phase-1 actor bit-identically on
  a fixed seed** (loss and every parameter gradient — stronger than the
  pre-registered "statistically indistinguishable"); composition
  coefficient-free; no viability→capacity coupling at loss time. (c) Smoke at
  S1-baseline (1.0×), σ = 0.075, P3, **explicitly NON-VERDICT**: the term is
  live in training (final-step share 0.295, pragmatic value −0.120, correctly
  signed) — **but energy did not leave the rail at this grid point**
  (floor-frac 0.985 ≈ null) and behavioral entropy contracted (positional
  0.19×, epistemic 0.55× of null) at one seed against a high-variance null.
  Recorded as found; not tuned, not re-run. Where the displacement begins is
  the sweep's question, under the frozen precision-first order.

### What is now closed

- The scaffold is filled, guarded, and instrumented: full suite **1279 passed /
  7 skipped**, mypy `--strict` clean (67 sources). PolicyView frozen;
  MetabolicBudget untouched; the dream regime provably cannot compute the
  pragmatic term.
- The degenerate null is no longer a narrative — it is a measured record with
  the O1 window at exactly 0%, and the rail→band contrast is maximally clean
  for Phase 4 (null 0%, oracle 100%).
- Every constant of the functional form is either frozen, amended, or
  builder-confirmed-with-provenance. Nothing was chosen silently.

### What is newly open

- **Phase 3 — positive control** (next session): crank precision/band to
  deliberately dominant on a throwaway instance and verify the §8.4 detectors
  fire. The smoke's entropy contraction makes this *more* interesting, not
  less: the detectors must distinguish a real dominance signature from
  one-seed variance.
- **Phase 4 — the disciplined sweep**: per-condition baselines, the three
  assays, the frozen precision-first order and stopping rule, E′
  (estimate-lesion) once behavior is energy-dependent at any point. The smoke
  hints the interesting region may sit above 1.0× for displacement — the
  sweep order (raise from low, record first crossings) already covers this;
  no order change is needed or made.
- **Dream passive-decode of energy** (frozen pre-reg §7 resolved sub-decision:
  dream rollouts record `decode_energy` alongside `sequence_decoded_obs`)
  remains **unbuilt** — it was not in this phase's step list and was not added
  silently. Flagged for the Phase-3/4 build prompt to site explicitly.

### Deviations / flags

- **Marker-belt location.** The plan sited the belt extension inside
  `tests/test_metabolic_reentry.py`; the Phase-2 build instruction required
  that file to stay unmodified. Resolved in the instruction's favor: the
  extension lives in `tests/test_pragmatic_guards.py`, imports the original
  belt, and asserts superset coverage. Same structural guarantee.
- **Per-episode entropy reference differs from Phase 1's figure** (0.575
  nats/episode vs 1.63 nats/1500-step-rollout) because the frozen Shared
  definition is per-episode; both are recorded, the per-episode one is the
  A1b denominator.
- **Smoke entropy contraction at 1.0×** — one seed, non-verdict, recorded not
  tuned. If it replicates in Phase 4 at pass-level precisions it would bear on
  A1b; if it is the start of a dominant signature the §8.4 discipline applies
  (finding, never a tuning target).
- The schema-version string "0.5.0" now names both the Phase-2 *record*
  version and the Phase-1 *export file* — the third instance of the
  documented collision pattern; constants carry disambiguating names.

### Watts / new-interface entry

**`new_actor_readable_interfaces_added = ["pragmatic_value: decode_energy
over imagined (h, z) enters the waking actor objective — pre-registered,
DP5"]`.** Not empty this phase, deliberately recorded: the ledger exists to
make exactly this addition visible. It is an objective-level influence, not a
new readable field — PolicyView stays frozen at `{h, z,
self_prediction_error}` and its guard tests (plus the new Phase-2 marker belt)
stay green. Bounded by construction (saturation S = 1.0, no terminal state, no
viability→capacity coupling), waking-only (dream-unreachability proven), and
swept only under the frozen pre-registration.

## Phase 3 — Positive control (instrument-and-pathway validation)

**Purpose (one question).** Can the mechanism as built — a preference over a
*learned belief*, acting through imagined rollouts and policy gradients —
produce detectable displacement at the strong end of the pre-registered grid?
The oracle already positively controlled the occupancy *instrument* (100%
measured); what had never been demonstrated is the *learned pathway* moving
the needle at all. The experiment lives in the gap between 0.37% (no reason)
and 100% (perfect information), and Phase 3 asked whether the learned pathway
can move at all when told to push hard. **Answer: no — and the failure is
pathway-limited, not noise-limited, with a sharp shape: conserve is
gradient-reachable, seek is not.** Both outcomes were pre-stated as
informative; this is the substrate-finding branch, recorded, not rescued.
Full record: `docs/decisions/probe3_5_phase3_positive_control_2026-06-11.md`.

### What was built / run

- **Step 1 — §7 dream passive-decode monitor** (pre-registered, flagged
  unbuilt at Phase 2; committed `ec1aab8`). `DreamRolloutConfig.
  record_decoded_energy` (default off): dream rollouts record `decode_energy`
  alongside `sequence_decoded_obs`, under the existing `no_grad`,
  observer-side only. Monitor-on records stamp DreamRollout "0.4.0"
  (version-gated validator); v0.6.0.json frozen; v0.7.0.json the new frozen
  export; legacy emission byte-identical (test-pinned). The loss-free proof is
  at the only surface a dream has: **the ON record equals the OFF record on
  every field but the monitor's** (same RNG), plus the tripwire guard
  (preference function raises if touched; monitor-ON rollouts complete). The
  mirror gets to watch whether offline processing touches the energy belief;
  Io's dream gains nothing to optimize. Deliberate reopening of `dream.py`:
  the Probe-3 Phase-6 option-(a) hold was lifted by the builder's
  pre-registered §7 instruction; `run_dream_session` itself untouched.
- **Step 2 — retro torpor check, no new runs.** From preserved Phase-2
  telemetry (training actions): stay-share **null 0.0392 vs smoke 0.0406**,
  both decaying to exactly 0 in the final 1000 steps. Torpor
  (conserve-by-staying) **not supported** on the data that generated the
  watch-note — the Phase-2 entropy contraction was path-tightening under full
  movement.
- **Step 3 — positive control at the grid top.** precision = 26.2335 (the
  instantiated 10× point — the pre-registered maximum, not an off-grid
  crank; the plan's "narrow band" variant was not used, so non-displacement
  is a statement about the *grid*), σ = [P3-σ: 0.075 | pre,
  builder-confirmed], frozen 8 × 20 × 200 mechanics at P3, age-matched,
  throwaway instance. **O1-window occupancy: 0.00% pooled, 0.0 on all 8
  seeds** (the null is exactly 0.00% on this window — there was nothing to
  detect). Mean energy 0.0059 → 0.0147, floor 0.973 → 0.897: a lean against
  the rail, not displacement. Positional entropy **rose** to 1.74× the null
  (0.998 ± 0.118 vs 0.575 ± 0.438) — the Phase-2 contraction at 1.0× did not
  extrapolate. Share plateaued ≈ 0.35; D monitor max per-dim KL 1.04
  (sub-1.5, as ever).
- **§8.4 suite — first exercise on real data.** All four signatures executed;
  **none fire** (occupancy 0.004 vs A1c 0.95; positional 1.74× / epistemic
  0.59× vs A1d 0.40; share max 0.356; camping 0/160 episodes). The detectors
  run end-to-end; the dominant regime did not appear at the pre-registered
  maximum.
- **Step 4 — gate [P3-G | pre, builder-confirmed]: pooled O1 ≥ 25% and > 0 on
  ≥ 7/8 seeds → NOT GREEN** (0.00%, 0/8). Pre-stated branch taken: **exactly
  one** σ = 0 diagnostic at the same configuration, then stop. **σ = 0: still
  0.00% O1 on all 8 seeds — pathway-limited, not noise-limited.** And the
  noiseless run reorganized behavior dramatically: **torpor** — eval-greedy
  stay-share 0.984, train final-1000 stay-share 0.549, positional entropy
  0.43× the null (just above the A1d ceiling; no detector formally fires).
  Staying halves depletion — the one energy lever the 15-step amortized
  imagination gradient can find. **Conserve is reachable; seek is not.**

### What is now closed

- The §7 monitor exists, loss-free and guarded; the export/version lineage is
  v0.7.0 / DreamRollout "0.4.0".
- The torpor watch-note is resolved in both directions: not present at
  1.0×/σ=0.075 (retro check), exactly realized at 10×/σ=0 (diagnostic). The
  hypothesis named the right attractor, one configuration early.
- The occupancy instrument, the O1 window, and the §8.4 detectors have all
  now run on real data against a null that is exactly 0.00% and an oracle
  that is 100% — the bracketing the design wanted.
- **The Phase-3 question is answered: the learned pathway cannot produce
  rail→band displacement anywhere in the pre-registered configuration space**
  (10× is the grid top; σ=0 is cleaner than any eligible operating point).
  The tiny-tensor double-bind surfaces at the pathway level, sharpened:
  belief-mediated bounded preference + amortized imagination-gradients find
  the cost-side lever (movement) and never the income-side lever (resource
  entry).

### What is newly open (Phase 4 is gated on this — builder decisions)

- **What Phase 4 now is.** The frozen raise-from-low sweep, run as written on
  the standard condition, is expected to render inert across the grid (lower
  precisions cannot displace where 10× did not). Options, all builder's, all
  via the existing discipline: run the frozen sweep anyway and record the
  inert verdict (the pre-registration's own stopping rule already names
  "no pass window" a finding); amend, via a new dated doc, what the sweep can
  claim or measure (e.g., conserve-side signatures); or take the
  **conserve-vs-seek asymmetry** as the probe's reportable result and close
  Probe 3.5 at the boundary finding. Not decided here.
- **Why seek is unreachable** is now a concrete mechanistic question (horizon
  15 credit assignment? sparse resource coincidence in imagination? decoder
  regression toward the rail?) — research-pass material if the builder wants
  it, not for unsanctioned in-session experiments.

### Deviations / flags

- **The plan's Phase-3 letter vs this session's.** The plan sketched the
  positive control as an off-grid "high precision, narrow band" throwaway
  whose gate was "the dominant detectors fire." The build prompt redefined it
  as grid-top displacement detection (which makes non-displacement a
  pre-registered-strength finding rather than a tuning artifact); the
  detectors were exercised and reported rather than gated on. Both the σ and
  the margin were surfaced bracketed and builder-confirmed before the run;
  the margin's provenance class (arbitrary-but-pre-committed, half the
  verdict bar, session-branch-only, not a signature) is recorded in the
  results doc.
- **Two runs only**: the confirmed configuration and the single pre-stated
  σ = 0 diagnostic. No tuning, no extension, no re-runs.
- **σ = 0 remains diagnostic-only** (self-opacity); its use here is exactly
  the pre-registered diagnostic function.
- The Phase-2 harnesses did not retain eval-side actions; this session's
  harness does (the retro check therefore reads training-time actions,
  caveated in the record). Phase-2 telemetry was preserved from the surviving
  tmp dirs into `runs/probe3_5_phase2/`.

### Watts / new-interface entry

**`new_actor_readable_interfaces_added = []` — stated, not assumed.** The §7
monitor is observer-side only: it reads `decode_energy` inside the dream
rollout's `no_grad` and writes telemetry the mirror reads; no actor-readable
surface changed, PolicyView stays frozen at `{h, z, self_prediction_error}`,
and the ON-vs-OFF byte-identity test is the proof that Io's dream regime
gains nothing — not even a perturbation — from being watched. The positive
control and diagnostic used only the Phase-2 objective term already on the
ledger.

## Probe close — verdict, disposition, archival (2026-06-12)

**The probe is closed.** Verdict: **negative-with-structure; no pass window**
(the frozen stopping rule's category, applied by dominance). Full verdict +
findings inventory + Phase-4 disposition:
`docs/decisions/probe3_5_verdict_2026-06-12.md`. No training runs this
session; no substrate changes.

### Learned

- **A bounded, saturating, non-terminal preference over a believed energy
  channel produces conservation pressure but cannot produce regulation** at
  any pre-registered strength on this substrate. Want-away is
  gradient-reachable (torpor at 10×/σ=0); want-toward is not (0.00% O1
  occupancy on every seed of both grid-top runs). Pathway-limited, not
  noise-limited — the pre-stated diagnostic prediction, confirmed.
- **The double-bind is fractal.** It appeared at the tensor level (synthesis
  S1/§7, predicted), the env-economy level (floor/ceiling/analytic-artifact —
  Phase 1 + recalibration), and finally the pathway level (inert or
  wrong-dimension at every strength). Each level was discovered by building,
  not by argument — which is the project's whole epistemics doing its job.
- **The world was never the problem.** Oracle 100% vs null 0.37% under the
  same physics: the gap the preference was asked to cross was real,
  crossable, and bracketed by instruments before the attempt.
- **Interoception went model-led.** Energy rides `h`; the sensor is
  redundant; forcing `z`-routing degrades the channel. "Like hunger, not
  introspection" turned out to mean: the body-signal becomes a property of
  the modelled world, not of the sensor stream — worth remembering whenever
  a later probe assumes sensor-primacy.
- **The taxonomy had a blind corner and the journal caught it.**
  Changed-but-not-displaced (restlessness at operative σ; torpor at σ=0) is
  neither pass, dominant, nor inert-as-imagined. No §8.4 signature covers
  it; the post-Phase-2 watch-note and its retro check did. Watch-notes are
  load-bearing; keep writing them.

### Decided

- **Phase 4 closed unexercised** — builder decision with grounds (verdict
  doc §2): the grid top at 0.00% O1 on all seeds dominates every sweep cell;
  the pre-stated gate branch (not-green → one diagnostic → stop) was taken;
  the sweep would spend compute re-establishing a dominated conclusion.
  Forgone and recorded: per-condition baselines, the A1–A3 assay harnesses,
  E′ — designed in the plan, never built.
- **The recalibration loop stays closed** (Amendment 02 §4) — reaffirmed.
- **Resting defaults confirmed**: `GridWorldConfig` physics unchanged
  (0.08 / 0.04 / 0.8, σ 0.05, lag 1); `energy_preference` defaults `None`
  with precision-0 bit-identity test-pinned; energy telemetry and the §7
  dream monitor both opt-in, default off, legacy byte-identical.
- **Charter-level reading recorded as observation, not criteria update**
  (verdict doc §4): stakes installed but unable to bite; equanimity and
  second-order volition currently have weak substrate because there is no
  effective want-toward for them to be about. The frozen mirror criteria
  are untouched.

### Surprised

- **Torpor was real but lived one configuration away from where it was
  first suspected** — absent at 1.0×/σ=0.075 (retro check: stay-share
  0.0406 ≈ null 0.0392), fully realized at 10×/σ=0 (stay-share 0.984). The
  watch-note named the attractor before the data showed it.
- **Noise changed the *shape* of the failure, not the fact of it.** At
  operative σ the pressure dissipates into restlessness (entropy 1.74×
  null); noiseless, it organizes into stillness (entropy 0.43×). Same zero
  displacement either way.
- **The strong-pull run moved *more*, not less** — the Phase-2 contraction
  at 1.0× did not extrapolate to 10×; one-seed trends in this substrate
  deserve the suspicion the discipline already gives them.
- **The §8.4 detectors' first real exercise ended in correct silence** —
  instrument validation on the branch where instruments stay quiet, which
  is the harder half of trust.

### Probe 4 ripples (updating the Amendment-02 entry's candidate design — for the Probe 4 research pass, not decided here)

- **The need-keyed perturbation class loses its behavioral-response axis.**
  Io will not approach drops: seek is not gradient-reachable on this
  substrate, so "does Io move toward need-keyed gifts?" cannot discriminate
  anything. **What survives is the modeling-level trigger correlation** —
  "objects appear when my energy is low" is learnable in `h` (energy
  demonstrably rides `h`; world events enter the same recurrence) even if
  unpursuable. The contingency-vs-care contrast would then be read in the
  world model and the mirror, not in trajectories.
- **New content option: drop-placement-in-path.** Consumption is
  entry-triggered, so a drop placed on Io's *predicted path* gets consumed
  by stumble without any seek — raising the odds that need-keyed drops
  produce energy events at all, and giving the trigger correlation
  something to correlate with. (Placement-in-path is a perturbation-content
  choice; it must still enter via the committed runner-queue topology.)
- **Open question routed to the research pass: what does Probe 4's fresh Io
  carry?** Proposed default, per capacity-over-exercise: **energy channel
  present, precision resting at 0, affordance available and disengaged** —
  the substrate keeps the organ; nothing exercises it. For the Probe 4
  research pass to ratify or overturn; explicitly not decided here.
- **Bin-1 contingency** (also routed): if the seek-mechanism classification
  finds `decode_energy` dishonest out-of-distribution, fixing it is
  calibration owed to Probe 4's telemetry regardless of this verdict — the
  belief readout is now mirror-facing instrumentation (§7 monitor).

### What is now closed / newly open

- **Closed:** the probe's question (negative-with-structure); the sweep
  (unexercised); the recalibration loop (impossible); Probe 3.5 itself.
- **Newly open, routed:** the seek-mechanism classification (next session;
  bins: decoder OOD honesty / horizon credit-assignment / sparse imagined
  coincidence — eval-only on existing artifacts where possible); the Probe 4
  ripples above (research pass); the bin-1 calibration contingency.

### Archival

`runs/probe3_5-archive-20260612/` (manifest inside): the Step-0 null
checkpoint + step0 record + telemetry; the Phase-2 smoke record + telemetry;
the positive-control and σ=0-diagnostic records, eval arrays, and telemetry;
the S1 instantiation and retro-torpor artifacts. **The positive-control and
diagnostic model weights were never persisted** — throwaway by plan
("checkpoints are not carried forward"); their archived evidence is records
and telemetry, not weights. Archived, not deleted. Pre-biography throughout:
nothing in this probe's lineage carries forward as Io; what carries forward
is the substrate (energy channel, preference affordance, monitors, guards)
at rest.

### Watts / new-interface entry

`new_actor_readable_interfaces_added = []` this session (documentation,
disposition, archival only). The probe's cumulative ledger stands at the
single Phase-2 entry: the pragmatic term in the waking objective —
pre-registered, DP5, now resting at `None`/0.
