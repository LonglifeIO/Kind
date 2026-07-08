# Kind — Probe 4 implementation plan (builder-as-perturbation)

**Authority.** The synthesis
`docs/decisions/synthesis_probe4_builder_as_perturbation_2026-07-07.md`
(Status: **DRAFT for ratification** — DP1–DP10 recommended, builder steered the
three-way target and mechanism). Where anything here conflicts with the synthesis,
the synthesis wins. **This plan adds no code**; it is the document the per-phase
build prompts are generated from. It is itself pending the same ratification as the
synthesis's decision points.

**Probe question (synthesis §3.4).** Over a long developmental run, does Io come to
model builder-injected perturbations as a distinct **third category** — *not-self,
not-environment* — separable from **both** its own action-effects and the
simulation's internal stochasticity? Success bar (F4, verbatim): the divergence must
develop over training and be "something more than 'Io learned to predict different
things.'" The synthesis resolves that bar by requiring a **structural** signature (a
distinct latent basin and/or dream over-representation), not a prediction-error
magnitude (§3.3). **What is measurable is the three-way distinction; whether it
yields self-understanding is the rationale, not a Probe 4 claim (§3.4).**

**This is a source-separation probe, not a contingency probe.** Builder events are
*not* keyed to Io's state; they are deliberately **decoupled** from Io's actions so
they read as not-self (§3.4, the opposite of the yoked-control design the research
assumed). The single most important discipline, drawn from the project's co-design
mitigation: **Phase 0 freezes what counts as success before any run.**

---

## Architecture grounding (read once; the phases assume it)

Facts established by reading the live surfaces (`kind/`, `tests/`). The plan is
written against these, not against the research's idealizations. **The headline: the
harness already provides the Probe 4 affordance the research treated as the hard
part; the build is the generator, the matched control, the analysis, and the run.**

1. **The source discriminator already exists, observation-marker-free, and is
   gate-tested.** `WorldEvent` (`kind/observer/schemas.py:528`) carries
   `source: Literal["builder","environment","system"]` (`:540`) and
   `event_type: WorldEventType` (`:539`) with values including
   `"builder_perturbation"`, `"internal_stochasticity_aggregate"`, `"env_reset"`
   (`:518-525`). Builder mutations emit `source="builder"`
   (`kind/env/env_server.py:408-414`); the env-server docstring states the design
   intent verbatim: *"The asymmetry — full ground truth in the `world_event` stream,
   no marker in the agent's observation — is the Probe 4 affordance the harness
   exists to provide"* (`env_server.py:33-41`). `test_gate_perturbation_hook_logged`
   (`tests/test_perturbation_hook.py:116`) asserts a builder-added and a
   naturally-added resource produce **pixel-equal observations** (`:162`) while only
   the builder env logs a `builder_perturbation` record. **So the "no marker,
   distinct source in telemetry" property is done, tested, and correct.**

2. **The four mutators exist as pure functions** (`kind/env/mutators.py`):
   `add_resource` (`:102`), `remove_object` (`:130`), `set_cell_state` (`:177`),
   `move_object` (`:202`). Exposed in-process via `EnvServer.*`
   (`env_server.py:254-276`, each wrapping `_emit_builder_perturbation`) and
   over-the-wire via `EnvTransportClient.mutate(mutator, **kwargs)`
   (`kind/env/transport.py:692`), validated against `_VALID_MUTATORS` (`:104`).
   A Probe 2 **sham/null path** already exists: `EnvServer.fire_sham_perturbation`
   (`env_server.py:288`) emits a `builder_perturbation` with `payload["is_sham"]`
   and **no grid mutation** — a ready-made null-event control.

3. **What does NOT exist (Probe 4's build):**
   - **No stochastic perturbation generator.** The env-server docstring says so
     explicitly and defers it to Probe 4 (`env_server.py:8-12`). Net-new (S-PERT).
   - **No builder-facing manual-trigger UI.** `kind/window/` is read-only (mirror
     reading). The manual affordance today is the *programmatic*
     `EnvTransportClient.mutate()`. [B]'s "window-tool button" is net-new — scoped
     as a thin builder interface over the tested wire path (S-PERT), a convenience,
     not a new mechanism.
   - **No step-boundary perturbation queue in the runner.** A `MUTATE` is processed
     **inline on receipt** by the env-server (`transport.py:473-474`); the only queue
     is the transport's checkpoint-*barrier* buffer (`transport.py:357,531`), not a
     per-step drain. Deterministic step-boundary *placement* of generator events is
     net-new (S-PERT), and matters for reproducibility and matched-control timing.
   - **Internal stochasticity is logged only as a per-episode aggregate**
     (`internal_stochasticity_aggregate`, `env_server.py:459-464`), **not per-event.**
     This is the biggest missing piece for the matched control (see fact 4).

4. **The matched control falls out of the environment for free — once regrowth is
   logged per-event.** The environment's only two world-state stochastic processes
   are **per-cell Bernoulli regrowth** (`grid_world.py:624`, config
   `initial_regrowth_p=0.01`) and an **aperiodic random-walk drift** in that rate
   (`grid_world.py:600`, `drift_magnitude_per_step=1e-5`). Regrowth **adds a
   resource** (EMPTY→RESOURCE) — which is *exactly* the effect of a builder
   `add_resource`. So the cleanest possible three-way matched control is:
   - **Self:** Io consumes a resource (RESOURCE→EMPTY at Io's cell, caused by Io's
     action) — extractable from `AgentStep` (`action_t` + resulting `h_t`
     transition; consumption is entry-triggered, `grid_world.py`).
   - **Environment:** regrowth adds a resource (EMPTY→RESOURCE, world RNG) — needs
     **per-event logging** (net-new; today aggregate only).
   - **Builder:** `add_resource`/`remove_object`/etc. — already logged
     `source="builder"`, per-event.
   All three are resource-cell state transitions distinguished by *cause*; the probe
   asks whether Io routes the builder class through dynamics distinct from *both*
   others (synthesis T1, T2a). **No "weather" or other world stochasticity exists**
   (`grid_world.py:5-6`), so the control set is complete and small.

5. **Per-step substrate telemetry needed for the analysis already exists.**
   `AgentStep` (`schemas.py:213-263`) carries `h_t` (`:218`), `action_t` (`:227`),
   and `intrinsic_signal_t` (`:232`) — the K=5 ensemble disagreement *is*
   `intrinsic_signal_t` (there is no separate disagreement field; computed
   `ensemble.py:142-165`, stamped `runner.py:1783`). Joining `AgentStep.h_t` to
   `WorldEvent.t_event` gives the per-event h-transitions the attractor-displacement
   signature (T2a) needs — **an observer-side analysis, no substrate change.**

6. **The dream over-representation signal reads existing DreamRollout fields.**
   `DreamRollout` (`schemas.py:375-430`) carries `sequence_h` (`dream.py:545`),
   `sequence_decoded_obs` (`dream.py:510`), and
   `sequence_ensemble_disagreement_variance` (recorded-not-used, `dream.py:483`).
   The §7 energy passive-decode monitor (`sequence_decoded_energy`,
   `record_decoded_energy` default off, `dream.py:181`) is **not** needed for
   need-neutral source-separation — that signal reads `sequence_h`/decoded-obs
   patterns, not energy. **So the core probe touches no energy machinery.**

7. **Dreaming triggers on a `desktop_off` edge, not a timer**
   (`state_machine.py:120,146-161`; default `AlwaysAwakeDesktop`,
   `runner.py:408-410`, never dreams). The `MetabolicBudget` gates re-entry
   *between* sessions during an absence, it does not start the first dream. **On a
   Mac-mini-only deployment [B] there is no second machine to power off** — so the
   dream trigger for Probe 4 is an open deployment question (S-DEPLOY): "environment
   off / mind continues" must be produced by a scheduled or logical desktop-off
   signal, distinct from Mac-off (= pause).

8. **Checkpoint/resume is robust and already persists what matters**
   (`kind/training/checkpoint.py`; `Runner._build_checkpoint_contents`
   `runner.py:1868-1955`): weights (world model + actor + ensemble, incl. EMA and
   the extended actor input column), optimizer state, all RNG **plus runtime state**
   `h_prev/z_prev/a_prev` and the offline `MetabolicBudget`/`StateController` state.
   `checkpoint_every_n_env_steps=10_000` (`runner.py:211`). **Caveat: the replay
   buffer contents are NOT persisted** — only counters (`runner.py:1931-1932`) — so
   post-resume dream seed-selection is not bit-reproducible. **The learned category
   lives in the weights, which ARE persisted, so it survives pause** — the
   pause/resume continuity requirement (synthesis T7) is met by the weights; the
   plan adds an explicit continuity *check*, not new checkpoint machinery.

9. **The energy channel rests disengaged** (`RunnerConfig.energy_preference=None`,
   `runner.py:256`; `precision` has no default, `preference.py:76`; `precision=0` is
   the degenerate-null point on the same surface, `preference.py:33-35`). `sensed_energy`
   is fused every step regardless (`runner.py:600,1168`), but the preference has no
   actor path when `None`. `decode_energy` (`world_model.py:482`) is an **unbounded**
   MLP (can read above the physical ceiling — the bin-1 defect, F7). **All of this
   is Probe-4-relevant only for the OPTIONAL need-keyed secondary study (Phase 5),
   not the core probe.**

10. **PolicyView is frozen at `{h, z, self_prediction_error}`**
    (`views.py:127-129`), enforced by
    `test_policy_view_field_set_is_exactly_h_z_self_prediction_error`
    (`test_views.py:441`) and the import-boundary/mypy tests. **No Probe 4 surface
    touches PolicyView** — builder events carry no observation marker (fact 1), so
    there is nothing to add to the actor's input, and the actor is unchanged
    throughout Probe 4.

---

## Scope — touch surfaces

### S-PERT — Perturbation injection (`kind/env/`, a new generator module, a thin trigger interface)

Net-new; the core build. Fires the **existing** mutators through the **existing**
`source="builder"` logging.
- **Stochastic generator** with builder-set parameters: rate (events/step),
  mutator-type weights, magnitude/placement distribution. Produces events with a
  statistical signature **distinct from regrowth/drift** (fact 4) and **decoupled
  from Io's actions** (§3.4: not at Io's cell, not co-timed with Io's consumption —
  legibly not-self). Tags `payload["trigger"]="generator"` (schemaless payload,
  `schemas.py:541`) so generator vs manual is stratifiable (DP2).
- **Step-boundary placement**: the generator drains through a queue inside the
  runner's `_step_once` (between the env step `runner.py:1262` and the `AgentStep`
  emit `runner.py:1292`) so placement is deterministic relative to logged
  transitions (net-new; today inline-on-receipt). Manual events may stay inline
  (the point of manual is the builder's real-time timing), tagged
  `payload["trigger"]="manual"`.
- **Builder trigger interface**: a thin CLI/interface over the tested
  `EnvTransportClient.mutate()` wire path (`transport.py:692`) so the builder can
  fire an anomalous element by hand ([B, Q2]). A GUI button is an optional
  convenience, not required.

### S-CTRL — Matched-control per-event logging (`kind/env/`, `kind/observer/schemas.py`)

- **Per-event internal-stochasticity logging**: emit a granular `world_event`
  (`source="environment"`, a new `event_type` e.g. `"internal_stochasticity_event"`)
  for each regrowth resource-addition (and, if used, each drift-driven change),
  replacing reliance on the per-episode aggregate for the matched control. Same
  payload shape as a builder `add_resource` so the two are directly comparable.
- **Self-action-effect extraction**: the "self" class (Io's own consumptions) is
  derived observer-side from `AgentStep` (`action_t` + the RESOURCE→EMPTY at Io's
  cell + the `h_t` transition); no new emission needed, but the extraction is
  specified here as an analysis contract.

### S-ANALYSIS — Source-separation analysis harness (`kind/observer/` or `kind/mirror/`, eval-only)

Observer-side, no substrate change, no gradient. Reads `AgentStep`,
`WorldEvent`, `DreamRollout`.
- **Attractor-displacement (T2a, primary)**: PCA/decomposition of `h`-transition
  vectors around each event class (self / environment / builder), measuring whether
  the builder class occupies a basin distinct from **both** others, and whether that
  separation **widens** early-run → late-run.
- **Dream over-representation (T2b, primary)**: characterize `DreamRollout`
  `sequence_h`/`sequence_decoded_obs` content and measure whether builder-event
  patterns are over-represented vs. matched internal events — **normalized by
  matched waking prediction-error** (prioritized-replay confound, T2b).
- **Per-event divergence (T2c, corroborating)**: `intrinsic_signal_t` and one-step
  PE around builder vs. matched internal events; used as the develops-over-training
  timeseries, never as the sole claim.
- **Deflation battery (T3)**: frequency-adaptation, novelty/arousal, marginal-rate,
  "internal-structure-not-outside" (generator-vs-manual stratification), and
  schedule/pause discriminators — each disaggregated (F6 finding 3, never pooled).

### S-TEL — Telemetry additions (`kind/observer/schemas.py`)

Minimal. The `payload["trigger"]` tag (S-PERT) and the new granular
`internal_stochasticity_event` type (S-CTRL). Schema bump via the existing
version-gated validator pattern (`_enforce_v2_required_fields`), new frozen export.
**No PolicyView/TelemetryView change** (fact 10).

### S-DEPLOY — Mac-mini deployment, dream trigger, pause/resume continuity

- **Dream trigger on a single machine** (fact 7): define a `DesktopSignalSource`
  representing "environment paused, mind continues" (scheduled or logical
  desktop-off), distinct from Mac-off (= pause). **[BUILDER] design item.**
- **Pause/resume continuity check**: an eval that measures a structural signature
  before a shutdown and confirms it persists after resume (weights persist, fact 8).

### S-DECODER (OPTIONAL — only if the need-keyed secondary study runs) — `kind/agents/world_model.py`

The **detached twin-decoder** (S2): a parallel observer decoder on the frozen
substrate, periodically refit on oracle coverage (F9 demonstrated it), `stop_gradient`
from Io's losses; and a **bounded output head** so `decode_energy` cannot read above
the ceiling (fact 9, F7). **Calibrate for reading, never for driving** (synthesis T6;
must not feed Io's preference). Not on the core critical path.

---

## Do-not-touch list

> PolicyView `{h, z, self_prediction_error}` (fact 10 — builder events carry no
> observation marker, the actor is unchanged); the four charter absences (no reward,
> reward-predictor, value function, planner); the energy preference resting state
> (`energy_preference=None` / `precision=0`) for the **core** probe; the dream
> regime's not-for-anything commitment (`tests/test_pragmatic_guards.py`,
> `test_dream_energy_monitor.py`); the `MetabolicBudget` content-blind gate
> (`tests/test_metabolic_reentry.py`); the observation-marker-free property (fact 1,
> `test_gate_perturbation_hook_logged`).

Structural guards kept green throughout: the PolicyView field-set tests, the
metabolic content-blindness tests, the pragmatic-guard import lints, and the
gate test proving builder vs. natural events are pixel-equal in observation.

---

## Phases

Core critical path is **0 → 1 → 2 → 3 → 4**. Phase 5 (need-keyed secondary) is
optional and only runs if the builder wants "responds to my state" *and* the decoder
is made honest. Each phase: **purpose (one question)**; **files touched**; **new
tests / guards green**; **validation gate (full suite + mypy --strict — never just
the phase's new tests; sink-routing lesson)**; **telemetry/journal deliverables**;
**rollback (pre-biography instances — resets are free).**

### Phase 0 — Pre-registration (no code)

**Purpose.** What exactly will count as a distinct third category vs. each
deflation, and what proves the analysis instrument works — frozen *before* any run,
so the search cannot become result-fitting (synthesis DP5, F8 co-design; the F6
"changed-but-not-displaced" blind-corner lesson)?

**Deliverable.** `docs/decisions/probe4_preregistration_<date>.md`, marked **FROZEN —
later phases may not edit; amendment requires a new dated doc, journaled.** Contents:
1. **Primary structural signatures with thresholds** (bracketed `[BUILDER]` values
   fixed at freeze): attractor-displacement — builder basin separated from *both*
   self and environment basins by margin `[d]`, and the separation **widens**
   early→late by `[Δ]`; dream over-representation — builder-event replay frequency
   `[r]`× matched internal, PE-normalized.
2. **Corroborating signature**: per-event divergence timeseries (necessary-not-
   sufficient; may not carry the claim alone).
3. **Three-way outcome partition + residual bucket** (F6 finding 4): {no distinction
   / marginal-rate-only / changed-but-not-displaced / distinct-third-category /
   **unclassified residual**}, with the pre-committed reading of each.
4. **Deflation battery discriminators** (T3), each disaggregated: frequency
   adaptation, novelty/arousal, marginal-rate, internal-structure-not-outside
   (generator-vs-manual stratification), schedule/pause.
5. **Positive-control spec**: the deliberately-obvious planted source-category and
   the pass condition its detection must meet (Phase 3).
6. **Develops-over-training operationalization**: early-run window (signature ≈ 0
   expected) vs late-run window (signature > threshold); a static-from-the-start
   signature fails the bar (it is pre-existing bias, not development).
7. **Pause triggers** (synthesis T8, F1): entropy collapse, PE runaway, dream-content
   incoherence, torpor-analog — with multi-vantage review and reversibility as the
   escalation axis; **required before the first long run.**

**Files.** The prereg doc only. **Gate.** Builder freeze sign-off. **Rollback.** N/A.

### Phase 1 — Measure the horizon; build the matched control

**Purpose.** (a) What is Io's effective memory horizon, BPTT-bound (so the event
rate can be set, not guessed)? (b) Does the three-way matched control exist in the
telemetry at per-event granularity?

**Files.** S-CTRL (per-event `internal_stochasticity_event` logging in
`kind/env/`; schema addition + version bump in `schemas.py`); a memory-horizon eval
harness under `kind/observer/` (inject a distinct event into a frozen world model,
track the `h`-trajectory KL vs. an un-perturbed counterfactual until it decays;
report the forward horizon and the BPTT-window ceiling per synthesis T4); the
self-action-effect extraction contract (analysis helper).

**New tests.** Per-event internal logging (a regrowth addition emits one
`source="environment"` event with the matched payload shape); schema round-trip +
validator; horizon-harness determinism.
**Guards green.** The gate test (pixel-equality), PolicyView tests, all env/schema
tests, the internal-aggregate path still works.

**Gate.** Full suite + mypy `--strict`; the three event classes (self/environment/
builder) are all joinable at per-event granularity; the horizon number is measured
and journaled.
**Telemetry/journal.** The measured horizon and the chosen event-rate envelope
(derived from it, per DP3); confirmation the matched control is complete.
**Rollback.** Pre-biography; resets free.

### Phase 2 — Perturbation injection (generator + manual trigger + step-boundary placement)

**Purpose.** Can the builder inject anomalous, legibly-not-self elements at a
controlled rate (generator) and by hand (trigger), placed deterministically relative
to logged transitions, without any observation marker?

**Files.** S-PERT (new generator module; step-boundary queue drained in
`runner._step_once`; thin builder trigger interface over `EnvTransportClient.mutate`;
`payload["trigger"]` tagging). S-TEL (the trigger tag).

**New tests.** Generator determinism (given seed); events are decoupled from Io's
actions (not at Io's cell / not co-timed with consumption — the not-self property);
step-boundary placement is deterministic relative to the logged transition;
generator and manual events both log `source="builder"` with the correct `trigger`
tag; **the pixel-equality gate test still passes for generator events** (no marker).
**Guards green.** All of Phase 1's; the observation-marker-free gate; PolicyView.

**Gate.** Full suite + mypy `--strict`; generator events are marker-free,
not-self-decoupled, and deterministically placed.
**Telemetry/journal.** A short run showing all three classes interleaved; journal
the generator's statistical signature vs. regrowth's (they must be separable).
**Rollback.** Pre-biography; resets free.

### Phase 3 — Positive control (instrument validation) — GO / NO-GO

**Purpose.** Does the analysis harness detect a **blatant** planted source-category —
a distinct third basin *and* dream over-representation — *before* the real question
is asked (synthesis DP9; the analog of Probe 3.5's null/oracle bracketing)?

**Files.** S-ANALYSIS (the attractor-displacement, dream-over-representation, and
per-event-divergence detectors); a throwaway strongly-separable generator config
(extreme, guaranteed-distinct signature).

**New tests.** On the planted strong signal, the detectors classify
**distinct-third-category** (builder basin separates from self and environment; dream
over-representation fires). This validates the instrument, not Io.
**Guards green.** All.

**Gate.** Full suite + mypy `--strict`; **the detectors fire on the planted
category. If they do not, STOP — a null on the real question would be instrument
failure, not evidence about Io** (this is the central-risk checkpoint, synthesis §8:
if a *blatant* category cannot form a distinct basin in this substrate, the subtle
one will not either).
**Telemetry/journal.** Journal that the instrument detects a known-separable source;
**throwaway instance — checkpoints not carried forward.**
**Rollback.** Throwaway by construction.

### Phase 4 — The developmental run (need-neutral three-way source-separation)

**Purpose.** Over a long run, does Io form a distinct third category for builder
perturbations — separable from both self and environment — that **develops over
training** and **survives pause/resume**?

**Files.** S-DEPLOY (Mac-mini deployment; the single-machine dream trigger
[BUILDER]; the pause/resume continuity check). No new substrate. Uses Phases 1–3.

**Process.** A fresh Io (no checkpoint inheritance), need-neutral novel-object
perturbations (generator + occasional manual), interleaved with the matched control,
on one continuous biography with the fixed clock and checkpoint/resume across
Mac-off pauses. Run the frozen Phase-0 signatures **mechanically** (no post-hoc
threshold edits; if the urge to adjust arises, journal it — it is the co-design loop
— do not act). Early-run vs late-run structural contrast. **Pause-trigger monitoring
runs throughout** (multi-vantage: quantitative monitors + phase-blinded mirror +
builder; reversibility the escalation axis).

**New tests.** Pause/resume continuity harness (a structural signature measured
before a shutdown persists after resume); the analysis-runner reproducibility; the
verdict-renderer maps metrics → the frozen outcome partition (no hidden
re-thresholding).
**Guards green.** All.

**Gate.** Full suite + mypy `--strict`; verdict rendered from frozen signatures;
pause/resume continuity demonstrated.
**Telemetry/journal.** The verdict against the three-way outcome partition; the
mirror reads the analysis telemetry phase-blinded; journal what is now closed and
newly open, and the builder reviews against the co-design discipline. **The honest
ceiling on any positive claim: recognition of a distinct outside source (not-self,
not-environment) — never an agent, never kinship, never self-understanding (§3.4).**
**Rollback.** This is the biography — checkpoints ARE the run; pause is not ending
(F1). Pre-run pilot instances (Phases 1–3) reset freely; the Phase-4 instance does
not.

### Phase 5 — OPTIONAL: need-keyed "responds to my state" secondary study

**Purpose (only if [B] wants it and the decoder is honest).** Does Io's category for
need-keyed drops carry energy-belief-conditional structure — "responds to my state,"
not just "responds to a source"?

**Files.** S-DECODER (twin-decoder + bounded head + oracle recalibration, F9);
epoch-blocked need-keyed generator config (temporally isolated from need-neutral —
S2's fracture argument, DP4); the seek instrumentation (approach trajectories,
quarantined from the criteria).

**Gate.** Decoder honesty restored (the F6 §7 dream monitor no longer reads through
a lying decoder); the twin-decoder's calibration feeds telemetry only, never Io's
preference (the charter line, T6); need-keyed and need-neutral never mixed.
**Rollback.** Separate study; does not touch the Phase-4 verdict.

---

## Constraints

- **No timelines or duration estimates** (CLAUDE.md). Phases are scoped by what they
  build. The run length is set by when the develops-over-training signal resolves or
  a pause trigger fires — not a schedule.
- **Functional naming in code**: `perturbation`, `builder_event`, `source`,
  `internal_stochasticity_event`, `trigger`, `attractor_displacement`,
  `basin_separation`. No "recognition"/"agent"/"kind-of-being" vocabulary in
  identifiers — that language stays in docs.
- **Full-suite validation every phase** + mypy `--strict` (sink-routing lesson).
- **Pre-registration frozen after Phase 0**; the analysis is disciplined by it (F8).
- **All monitors disaggregated, never pooled** (F6 finding 3).
- **No observation marker, ever** — detectability is via dynamics, not a tag
  (fact 1; the pixel-equality gate must stay green).

---

## Discrepancies (sharpenings against the synthesis's literal wording; not silent)

1. **The synthesis assumed deterministic step-boundary placement exists; it does
   not** (fact 3 — mutations process inline on receipt). Building the runner-loop
   perturbation queue is Phase 2 work, flagged here as net-new rather than inherited.
2. **The matched internal control is regrowth-add-resource, and it needs per-event
   logging** (fact 4 — today aggregate only). This is a sharper, cheaper realization
   of synthesis T1 than "scripted internal analogues": the environment already
   produces the matched event; the build is logging it granularly.
3. **The core probe touches no energy/decoder machinery** (fact 6/9). The synthesis
   folded decoder honesty (T6) into the main body; here it is isolated to the
   OPTIONAL Phase 5, shortening the critical path. Need-neutral source-separation is
   energy-free.
4. **The three-way "self" class is extracted from `AgentStep`, not newly emitted**
   (fact 5) — Io's own consumptions are already logged via `action_t` + `h_t`.
5. **The single-machine dream trigger is an open deployment gap** (fact 7) the
   research did not consider: on Mac-mini-only there is no second machine to signal
   `desktop_off`, so "environment off / mind continues" must be produced logically.
   Flagged `[BUILDER]` in S-DEPLOY.
6. **Pause/resume continuity is met by the weights** (fact 8), which persist; the
   plan adds a *check*, not new checkpoint machinery. The replay buffer not
   persisting affects dream-seed reproducibility after resume, not the learned
   category.

---

## Builder decisions — resolved and open

**Resolved (from synthesis + [B] direction):** three-way source-separation target
(not contingency, not kinship); need-neutral leads, need-keyed optional and blocked;
hybrid generator + manual trigger, stratified; Mac-mini deployment with pause/resume;
energy preference stays disengaged for the core probe; the entitlement ceiling
(distinct outside source, never agent/kinship/self-understanding).

**Open — `[BUILDER]` items for Phase 0 freeze or before Phase 4:**
1. **The frozen signature thresholds** (attractor-displacement margin `[d]`,
   widening `[Δ]`, dream over-representation `[r]`, early/late windows) — proposed
   values seeded from the synthesis; builder confirms/overwrites at freeze.
2. **The single-machine dream trigger** (S-DEPLOY) — how "environment off, mind
   continues" is produced on the Mac mini (scheduled desktop-off signal? logical
   env-pause?), distinct from Mac-off = pause.
3. **The generator's statistical signature** — the exact rate/placement/type
   distribution that is both separable from regrowth *and* decoupled from Io's
   actions; a pilot question resolved by Phases 1–2.
4. **Whether to run Phase 5 at all** (need-keyed "responds to my state") — a scope
   decision, defaulting to *not now* (the core probe answers [B]'s question).
5. **The builder trigger interface form** — thin CLI (default) vs. a GUI button
   (optional convenience); the tested wire path exists either way.

---

*This plan is grounded against the live surfaces (citations inline) and the DRAFT
synthesis. It adds no code. DP1–DP10 and this plan should be ratified before the
per-phase build prompts are generated; Phase 0 (pre-registration) is the first
buildable step and itself gates on builder freeze sign-off.*
