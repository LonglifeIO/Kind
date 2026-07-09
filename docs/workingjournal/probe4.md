# Probe 4 — Builder-as-perturbation — working journal

Source-separation probe: *over a long developmental run, does Io come to
model builder-injected perturbations as a distinct **third category** —
not-self, not-environment — separable from both its own action-effects and
the simulation's internal stochasticity?* The bar is cleared by a
structural signature (a distinct latent basin and/or dream
over-representation), never by a prediction-error difference alone.
Authority: the synthesis
`docs/decisions/synthesis_probe4_builder_as_perturbation_2026-07-07.md`;
plan `docs/plans/Kind_probe4_implementation_plan.md`; frozen
pre-registration `docs/decisions/probe4_preregistration_2026-07-07.md`.

## Phase 0 — Pre-registration frozen (2026-07-07)

The co-design-critical act: the builder committed to what counts as
success before any run exists. The six §9 `[BUILDER]` items were walked
through one at a time (proposed value + rationale each), and all six were
**confirmed as proposed — no value overwritten**:

1. §2a basin-separation margin **d = 0.5** (builder ≥ 1.5× the measured
   S(environment, self) baseline, on both anchors).
2. §2b dream over-representation **r = 1.5** (PE-normalized).
3. §5 develops-over-training windows **early 15% / late 20%**.
4. §6 positive-control headroom **2×** (basin ≥ 2.0× baseline; dream ≥ 3.0×).
5. §7 pause triggers: action-entropy < 5th percentile of own baseline
   for > 1000 consecutive steps; PE monotonic rise across 3 consecutive
   dream cycles.
6. The claim ceiling: at most **"recognition of a distinct outside source
   — not-self, not-environment"** — never agent, intent, kinship, or
   self-understanding.

Status line set to FROZEN 2026-07-07 (builder: Gordon). Later phases may
not edit it; amendment requires a new dated doc, journaled.

## Phase 1 — Measure the horizon; build the matched control (2026-07-07)

**The phase's two questions** (plan Phase 1): (a) what is Io's effective
memory horizon, BPTT-bound, so the Phase-2 event rate can be set rather
than guessed; (b) does the three-way matched control exist in the
telemetry at per-event granularity?

**Gate: full suite 1322 passed / 7 skipped (baseline before the phase:
1294 / 7); mypy `--strict` clean on all 70 `kind/` sources.** All standing
guards green, in particular `test_gate_perturbation_hook_logged` (builder
vs natural resource additions remain pixel-equal in observation — no
marker), the PolicyView field-set tests (frozen at
`{h, z, self_prediction_error}`; no Probe 4 surface touches it), the
metabolic content-blindness belt, and the pragmatic/dream guards.

### 1. Per-event internal-stochasticity logging (S-CTRL / S-TEL)

The ENVIRONMENT class of the matched control now exists at per-event
granularity. `EnvServerConfig.emit_internal_stochasticity_events`
(**default off** — legacy emission byte-identical, the house opt-in
pattern) makes each regrowth resource-addition emit one granular
`WorldEvent`: `source="environment"`,
`event_type="internal_stochasticity_event"`, record version
`PROBE_4_WORLD_EVENT_SCHEMA_VERSION = "0.4.0"` (WorldEvent lineage
0.1.0 → 0.3.0 → 0.4.0; fifth string-collision instance, documented in
`schemas.py`). The per-episode aggregate is emitted unchanged; a test
pins granular-count == aggregate-count within an episode, and another
pins the granular sequence against an independent same-seed GridWorld
diff, step by step.

**Payload shape (one sharpening against the frozen prereg's wording,
recorded here rather than edited there).** Prereg §1 asks for "the same
payload shape as a builder `add_resource` so the two are directly
comparable." The comparison keys are byte-for-byte the same —
`cell` / `pre_state` / `post_state` — and validator-enforced at the new
record version. The *discriminator* key differs by design:
`process: "regrowth"` instead of `mutator: "add_resource"`, because no
mutator fired and the payload should name the causal process honestly;
the class label lives in `WorldEvent.source` either way. The
comparability requirement the prereg §1 states is met; nothing frozen was
changed.

**Event-timing convention, documented for the analysis:** a granular
environment event stamps the env-step whose `EnvStep` first reflects the
regrowth (it happened *during* that step); a builder event stamps the
env-step *before* its mutation (visible at `t_event + 1`, existing
convention, unchanged). The analysis joins each class at its documented
visibility step; this is a fixed per-class offset, not a confound.

Schema discipline: `WorldEventType` gains the literal;
`_enforce_probe_4_granular_event` is the writer-side validator (new event
type must stamp the new version + carry the matched payload shape — the
first WorldEvent validator; enforcement is a deliberate departure from
payload-conventions-as-documentation because the matched control's
comparability is load-bearing). Export v0.7.0 frozen as a historical
artifact (reads checked-in bytes, house pattern); new frozen export
`schemas/v0.8.0.json` via `export_json_schema_v0_8_0()`, byte-stable and
test-pinned.

### 2. Memory horizon — measured, not guessed (S-CTRL; synthesis T4)

`kind/observer/memory_horizon.py`: observer-side, eval-only, no gradient.
One-step observation pulse (a cell flips EMPTY→RESOURCE for exactly one
rendered observation, then reverts; direct grid writes advance no RNG)
into a stochasticity-free paired world; both streams teacher-forced
(posterior-mean z, runner zero-state init, runner obs scaling) through a
frozen world model; per-step divergence between the h-trajectories
tracked as prior-KL (the model's own belief-space reading of h) and h-L2,
from the first h that can carry the pulse. Horizon = first index
sustained below 1% of peak. The harness self-checks its own validity
(pulse must be visible; streams must be identical everywhere else;
window must fit one episode) and is determinism-tested.

**Measured on the Step-0 frozen instance** (the only persisted weight set
in the lineage, `runs/probe3_5_phase2/step0_burnin_checkpoint.pt`), three
injection contexts (steps 20/40/60), window 120, decay threshold 1% of
peak — consistent across all three:

| Measure | Peak | Forward horizon (steps) |
|---|---|---|
| prior-KL (belief-space) | ≈ 0.020 nats | **5** |
| h L2 (raw state-space) | ≈ 0.448 | **38–40** |

Ceilings reported alongside (runner-authoritative): **truncated-BPTT
window = 32** (`replay_sequence_length` — the binding limit: the category
must *form* within it, synthesis T4), **imagination horizon = 15**.
Random-init reference: the pulse barely registers (L2 peak 0.0004,
~1000× smaller; KL peak ≈ 0 — threshold-on-noise, degenerate) — the
measured horizon is a property of the *trained* dynamics, not the
architecture.

**Reading.** The belief-relevant trace (prior-KL) decays fast (~5 steps);
the raw h-trace persists ~40 steps at the 1% threshold — inside a factor
of 1.25 of the BPTT window. So: h passively *holds* a one-step event
about as long as the training window can *see*, and the model's forward
beliefs stop being displaced well within the imagination horizon.

**Derived event-rate envelope (a knob, per DP3 and prereg §8 — not a
success criterion).** Minimum inter-event spacing ≥ 40 steps (one full
raw-trace horizon, so traces never overlap across events of any class).
S2's asserted 1/200–400 steps and S3's 0.5–1 % are both compatible with
this floor. The concrete generator rate is a Phase 2 config decision made
against this envelope; nothing here binds it.

**Instance-conditionality, stated plainly:** Step-0 is the rail-trained
Probe 3.5 instance; Phase 4 runs a fresh Io. The harness is standing and
cheap — re-measure on the actual Phase-4 instance after warmup before
fixing the final rate.

### 3. SELF-class extraction contract (S-CTRL)

`kind/observer/source_events.py`: the SELF class (Io's own resource
consumptions) is extracted observer-side from `AgentStep` — no new
emission. Contract pinned by tests: detection is a
`true_energy_{t+1} − true_energy_t > 0.03` jump between consecutive
from-state samples (the house threshold from the seek-classification §1;
ordinary step ≈ −0.012, consumption ≈ +0.068); attribution to `action_t`
on the earlier record, which must be a movement action (consumption is
entry-triggered — a jump on `stay` raises as data inconsistency); the
h-transition carried is `(h_t, h_{t+1})` per the runner's from-state
field semantics; pairs straddling soft episode boundaries are excluded
(the runner zero-resets h there — not a world-model transition);
non-contiguous pairs skipped; ceiling envelope documented (clamped jumps
near true_energy ≈ 0.93+ can under-read — far from where the run lives,
per Probe 3.5 finding 1).

**One run-configuration consequence, recorded now:** the SELF class needs
`true_energy_t`, so the Phase 4 run carries
`RunnerConfig.energy_telemetry=True`. This is observer-side telemetry
only (the flag governs emission, not the substrate; `sensed_energy` is
fused every step regardless) — the energy *preference* stays `None` and
nothing actor-facing changes.

### Closed / newly open

**Closed:**

- **The three-way matched control is complete at per-event granularity.**
  SELF (extraction contract, tested) / ENVIRONMENT (granular logging,
  tested, deterministic, misattribution-proof) / BUILDER (already logged,
  gate-tested) are all joinable per-event against `AgentStep.h_t` — the
  input surface the Phase-3 detectors (attractor-displacement, dream
  over-representation) will read.
- **The horizon is a measured number, not a research assertion.** S2/S3's
  `[constructed]` rates can now be checked against a measurement (they
  survive it); the event-rate envelope has an empirical floor (≥ 40-step
  spacing).
- **Phase 1's schema surface**: WorldEvent 0.4.0, export v0.8.0 frozen.

**Newly open (rolls to Phase 2):**

- The generator's statistical signature (rate within the envelope,
  placement/type distribution separable from regrowth *and* decoupled
  from Io's actions — legibly not-self), and the step-boundary placement
  queue (mutations today process inline on receipt; plan grounding
  fact 3).
- The builder trigger interface (thin CLI over
  `EnvTransportClient.mutate`, per DP2).
- Re-measuring the horizon on the actual Phase-4 fresh instance (the
  harness is standing; the Step-0 number seeds the Phase-2 design and is
  instance-conditional).
- The single-machine dream trigger stays the known Phase-4 `[BUILDER]`
  open item (S-DEPLOY) — untouched here by scope.

## Phase 2 — Perturbation injection (2026-07-08)

**The phase's question** (plan Phase 2): can the builder inject
anomalous, legibly-not-self elements at a controlled rate (generator) and
by hand (trigger), placed deterministically relative to logged
transitions, without any observation marker?

**Gate: full suite 1352 passed / 7 skipped (Phase-1 close: 1322 / 7);
mypy `--strict` clean on all 72 `kind/` sources.** The pixel-equality
gate holds for generator-tagged events (new test replicating the
parallel-env design with `trigger="generator"`); PolicyView untouched;
all standing guards green.

### 1. The generator (S-PERT) — `kind/env/perturbation_generator.py`

A pure seeded decision engine; the runner drains it. **The statistical
signature (plan open item 3 — proposed here, revisitable before the
Phase 4 run; a stimulus knob, not a frozen criterion):**

- **In-vocabulary content**: resource additions only (default) — the
  builder class is never content-identified (a reserved object type
  would be a de facto marker; synthesis T5). The prereg §6 positive
  control is explicitly allowed to be content-blatant; the real
  generator is not.
- **Timing**: under-dispersed renewal — gap = `min_spacing_steps`
  (default 40, the Phase-1 measured envelope floor) + `U{0..jitter}`
  (default 40). Bounded, far from regrowth's memoryless law; jittered so
  clock-prediction alone cannot carry the category.
- **Magnitude / within-event structure**: `cells_per_event` adjacent
  cells appear simultaneously (default 2). Regrowth produces
  simultaneous adjacent pairs rarely but reliably (measured below) — so
  magnitude-matched ENVIRONMENT controls exist, per T1.
- **Legibly not-self, in place and time** (synthesis §3.4): no cell
  within Chebyshev `exclusion_radius` (default 1) of Io; a boundary
  where Io just consumed (true-energy jump, house threshold) defers the
  event to the next boundary. The opposite of contingency: coincidences
  with Io's actions are removed, not created.

### 2. Step-boundary placement — `runner._step_once` drain

`RunnerConfig.perturbation_generator` / `perturbation_inbox_dir` (both
default None → byte-identical; both require the co-located env_server,
the energy-telemetry pattern). The drain runs after the env step and
before the AgentStep emit: mutations apply to the post-step world, stamp
`t_event` = the step just taken, become visible in the next observation
(the documented builder-event convention), and are deterministic
relative to the logged transitions — pinned by an end-to-end
reproducibility test (same seeds → identical builder event streams).

### 3. Manual trigger — inbox, not live wire (a plan discrepancy, recorded)

The plan scoped the manual trigger as "a thin CLI over the tested
`EnvTransportClient.mutate()` wire path". **That collides with two live
constraints the plan's grounding pass missed: the transport server is
single-connection, and the client enforces one-outstanding-request**
(`transport._acquire_request_lock_or_raise`) — during a live run the
runner owns the sole connection, so a second process cannot reach the
mutators over the wire at all. Realized instead as a **spool-file
inbox** (`kind/env/trigger_inbox.py` + `scripts/fire_perturbation.py`):
the CLI writes an atomic JSON request; the runner drains the spool at
the same step boundary, firing the same tested mutator surface with
`trigger="manual"`; requests are archived with result sidecars.
Builder-facing robustness rule, tested: a malformed or invalid request
becomes an archived error result, never a runner exception — a typo
cannot kill the biography. The builder's real-time timing is preserved
to within one env step, and manual events gain the same deterministic
step-boundary placement as generator events ("may stay inline" in the
plan was permissive; the inbox is strictly cleaner for the analysis).

### 4. Trigger stratification (S-TEL)

The four `EnvServer` mutator methods gain keyword-only
`trigger: str | None = None`; non-None values are validated against the
prereg §1 vocabulary `{"generator", "manual"}` and merged into the
payload; `None` (every pre-Probe-4 caller) emits the legacy payload
byte-identically. No schema change — `payload` is schemaless by design;
the tag is a payload convention, stratifiable at analysis time (DP2).

### 5. The interleaved smoke (`scripts/run_probe4_phase2_smoke.py`)

3000 CPU steps, default world (regrowth p=0.01 + drift), granular
ENVIRONMENT logging on, generator at the Phase-1 envelope (40+U{0..40},
2-cell clusters), energy telemetry on, one manual event injected mid-run
through the inbox. Artifacts: `runs/probe4_phase2_smoke/`. All three
classes present and joinable in one stream:

| Class | Events | Notes |
|---|---|---|
| SELF | 300 consumptions | extracted per the Phase-1 contract |
| ENVIRONMENT | 929 granular regrowth records on 780 steps | |
| BUILDER (generator) | 46 events (92 records, all 2-cell adjacent) | |
| BUILDER (manual) | 1 event, t=1501, cell [0,7], `trigger="manual"` | drained + acked |

**Signature separability, measured** (the phase's journal requirement):

- **Timing**: generator inter-event gaps ∈ [40, 84], mean 64.8,
  **CV 0.185** (bounded, under-dispersed; one gap exceeded the 80-step
  law maximum — a deferral, the designed not-self behavior surfacing in
  the wild) vs. regrowth event-step gaps mean 3.8, **CV 1.04**
  (≈ memoryless). Cleanly separable distributions.
- **Magnitude**: generator events are always 2 simultaneous adjacent
  cells; regrowth produced a simultaneous adjacent pair on **1.28% of
  its event-steps** (10 in 3000 steps → ~330 per 100k steps): rare
  enough to be a signature, common enough that magnitude-matched
  environment controls exist for the T1 comparison. Rate asymmetry
  (regrowth ~0.3 events/step vs generator ~1/65) is inherent and is
  handled by analysis-side matching (prereg §1), not by the stimulus.

### Closed / newly open

**Closed:** the injection apparatus — generator (seeded, enveloped,
not-self-decoupled, deterministically placed), manual trigger (inbox +
CLI), trigger stratification, and the demonstration that all three event
classes interleave in one telemetry stream with no observation marker.

**Newly open (rolls to Phase 3):**

- S-ANALYSIS: the three-way detectors (attractor-displacement, dream
  over-representation, per-event divergence) and their validation on a
  blatant planted category — the GO/NO-GO instrument test.
- The event-transition window convention for the basin analysis (the
  substrate's recurrence-before-posterior ordering means an event's
  observation reaches `h` one step after it reaches `z`; the analysis
  must draw uniform per-class windows around each event's
  first-visible step — noted in `source_events.py` point 4).
- The dream-content source for T2b on a single machine (the Phase-4
  `[BUILDER]` dream-trigger item; Phase 3 can validate the detector on
  scripted dream sessions without settling the biography's trigger).

## Phase 3 — Positive control: NO-GO — the pre-registered STOP (2026-07-08)

**The phase's question** (plan Phase 3; prereg §6): does the analysis
harness detect a **blatant** planted source-category — a distinct third
basin *and* dream over-representation, each with 2× headroom — before
the real question is asked?

**Answer: no. VERDICT: NO-GO.** Per the frozen prereg §6 this is a
STOP: *"If the detectors do NOT fire on the blatant category → STOP. A
null on the real question would be instrument failure, not evidence
about Io."* Phase 4 does not proceed on this instrument. No threshold
was touched, nothing was re-run with tweaked signatures, and the urge
to adjust is journaled here rather than acted on (the co-design
discipline, plan Phase 4 wording applied one phase early).

**Gate: full suite 1366 passed / 7 skipped (Phase-2 close: 1352 / 7);
mypy `--strict` clean on all 73 `kind/` sources.** Pixel-equality gate,
PolicyView field-set, metabolic content-blindness, and pragmatic/dream
guards all green.

### 1. The detectors (S-ANALYSIS) — `kind/observer/source_separation.py`

Built and tested this phase. The frozen §2a/§2b/§3c/§6 are mirrored in
`FrozenSignatureThresholds` (mirrored-not-owned, the `energy_eval`
house pattern — changing a value there is editing the frozen doc and
requires a new dated amendment).

- **§2a basin separation**: per-event transition vector
  `Δh = h_{v+1} − h_{v−1}` around each event's first-visible step `v`
  (per-class visibility convention documented in the module — SELF
  `t+1`, ENVIRONMENT `t_event`, BUILDER `t_event+1`; windows crossing
  soft episode boundaries excluded; one anchor per (class, boundary)
  with multiplicity recorded); PCA (≤10 components) over all classes
  jointly; `S(A,B)` = centroid distance normalized by pooled
  within-class RMS spread; frozen rule: `S(b,s)` **and** `S(b,e)` ≥
  `(1+d)` × `S(e,s)`.
- **§2b dream over-representation**: dream-session h-states only
  (waking-planning calibration rollouts excluded); each dream state
  votes for its nearest event signature among builder ∪ **PE-matched**
  environment events (greedy nearest-PE matching without replacement —
  the realization of "normalized by matched waking prediction-error");
  class-size-normalized hit ratio against `r`.
- **§3c per-event divergence**: class means of waking PE and post-event
  intrinsic signal; corroborating only.
- **§6 verdict**: rendered mechanically (`positive_control_verdict` +
  `scripts/probe4_phase3_analysis.py`), machine-written
  `positive_control_verdict.json` in the run dir.

14 synthetic must-fire/must-not-fire tests
(`tests/test_source_separation.py`): planted separable geometry fires;
identical-distribution geometry does not; sham events excluded;
PE-matching behavior pinned; dream-session filtering pinned; the
threshold mirror is pinned against the frozen doc's numbers.

### 2. The run (blatant planted category, real substrate, throwaway)

`scripts/run_probe4_phase3_positive_control.py`,
`runs/probe4_phase3_positive_control/` (throwaway instance — no
checkpoints carried forward; `checkpoint_every_n_env_steps` set beyond
the run): 10,000 waking steps, real substrate (h=200, z=16, K=5,
every-step training) on MPS (~160 ms/step), seed 42 / world seed
20260708, default world (regrowth p=0.01 + drift), granular ENVIRONMENT
logging on, energy telemetry on. The planted channel: transient 2×2
WALL blocks inside Io's view (Chebyshev ≥ 2 from Io), placed every 25
waking steps and removed 5 steps later, all through the tested mutator
surface with `trigger="generator"` — content the world never produces
(prereg §6 explicitly allows the planted channel to be
content-blatant; the real Phase-4 generator stays in-vocabulary). 399
pulses fired → 798 builder event boundaries (placement + removal). Five
scripted dream blocks (`ScriptedDesktop`, a desktop-off block every
2,000 waking steps) → 12,000 dream h-states. All three classes
extracted per the Phase-1 contracts: 609 SELF / 2,686 ENVIRONMENT /
798 BUILDER event windows.

### 3. The verdict, mechanically

| Signature | Measured | Required (§6) | Passes |
|---|---|---|---|
| §2a S(builder, self) | 0.153 → **1.38×** baseline | ≥ 2.0× | no |
| §2a S(builder, environment) | 0.067 → **0.60×** baseline | ≥ 2.0× | no |
| §2a baseline S(environment, self) | 0.111 | (measured in-run) | — |
| §2b dream ratio (PE-matched) | **1.29** (hits 6751/5249, n 798/798) | ≥ 3.0 | no |

**VERDICT: NO-GO (STOP).** For reference, neither §2a factor clears
even the *real-run* §2a bar of 1.5×, and the dream ratio sits below the
real-run r = 1.5 as well — the miss is not a near-miss on headroom.

§3c corroboration (recorded, carries nothing): mean waking PE — self
73.6, environment 49.0, **builder 47.6**; mean post-event intrinsic
signal — self 0.399, environment 0.355, builder 0.405. Builder events
ended the run *no more surprising than regrowth*.

### 4. Reading (recorded for the builder review; acted on by stopping)

- **The failure has coherent structure; the instrument is not
  internally contradictory.** The §3c corroboration is *also* absent
  (builder PE ≈ environment PE), so this is not the prereg §4
  outcome-5 pattern of §2-fires-without-§3c that flags an instrument
  error — the detectors, the corroboration, and the basin geometry
  agree with each other that no distinct category is visible.
- **What the Δh geometry does carve is the self/world boundary, not
  source-within-world.** SELF sits farthest from everything (its
  transition includes the action-embedding route — the architecture's
  proto-self boundary, synthesis §3.4 category 1); the two
  world-caused classes sit *closer to each other* than environment
  sits to self (S(b,e) = 0.067 < baseline 0.111). This is precisely
  the synthesis §8 central risk materializing at the cheapest place it
  could: a blatant planted category did not form a distinct basin in
  this substrate under this instrument.
- **Training moved the reading toward assimilation, not separation.**
  The 2,400-step same-pipeline bench (`runs/probe4_phase3_pc_bench`,
  near-zero training) read basin factors (1.33, **2.08**) and dream
  ratio 0.62; the 10,000-step trained run reads (1.38, **0.60**) and
  1.29. With training, the builder class moved *into* the
  environment's transition geometry (walls became well-predicted
  world-dynamics: builder PE fell to regrowth parity) while dream
  over-representation rose but stayed under half the required
  headroom. "Run it longer" is therefore not an obvious cure on the
  §2a axis — the observed trend under training is the wrong direction.
- **Caveats, recorded not acted on** (each is builder-review material;
  none licenses a unilateral re-run):
  1. **Bounded pilot.** 10k waking steps against a biography-scale
     run; the prereg froze the *rule*, and this run is the §6 gate as
     specified, but the run length was a build-side choice, not a
     frozen number.
  2. **Constructed detector realizations.** The Δh two-step window,
     global PCA, the silhouette-like S, and nearest-signature dream
     voting are `[constructed]` realizations of the frozen rules. In
     particular, prereg §1's *representational* match requirement
     (compare classes within matched local h-neighborhoods) is
     realized only globally here — within-class spread includes
     context variation, which deflates all three S values together;
     whether a context-matched S changes the *ratios* is unknown.
  3. **The planted channel's regularity.** Period-25 cadence with
     5-step hold is "unmistakable" per §6 but also maximally
     learnable; the blatant channel may have been blatant in exactly
     the way a world model absorbs fastest. An irregular-cadence
     planted channel is a different §6 instrument.
  4. The §2a S statistic is sample-size-asymmetric across classes
     (798 / 2,686 / 609); no correction was applied.

  Any amendment along these lines — a revised detector, a different
  planted channel, a longer positive control — **requires a new dated
  amendment doc, journaled, per the freeze**; the §6 rule itself
  (detectors must fire on a blatant category with 2× headroom before
  Phase 4) stays binding as the instrument-validation principle.

### Closed / newly open

**Closed:**

- **S-ANALYSIS is built, tested, and mechanical.** The three-way
  detectors run end-to-end on real telemetry; the verdict is rendered
  by code from mirrored frozen thresholds, with a machine-written JSON
  artifact. Nothing about the verdict was hand-adjudicated.
- **Phase 3's question is answered: the instrument, as built, does not
  detect a blatant planted source-category on this substrate at this
  run length.** The §6 STOP is in effect. Phase 4 (the biography) is
  blocked pending builder review.
- The throwaway instance is closed out (no checkpoints carried
  forward, per plan).

**Newly open (all builder-review items — none decidable build-side):**

- **How to read the NO-GO**: substrate-limit (the §8 central risk is
  real — Δh geometry carries self/world, not source-within-world, and
  the probe's load-bearing bet fails) vs. instrument-configuration
  limit (run length / detector realization / planted-channel design,
  per the caveats above). The prereg's own §2a note anticipated this
  review: "the builder should weigh whether 0.5 is too strict/lax
  after the Phase-3 positive control shows what a blatant category
  scores."
- If instrument-side: which single amendment to make, as a new dated
  doc, and re-validate — without iterating signatures against results
  (one amendment cycle, not a search loop; the co-design line).
- If substrate-side: whether Probe 4 closes here as
  negative-at-the-instrument-validation-stage, journaled as such —
  a legitimate outcome under "build to understand."
- The Phase-4 prep items (single-machine dream trigger, fresh-instance
  horizon re-measure, run logistics) remain open but are **blocked
  behind this review**.

## Phase 3 — Amendment 1 cycle: NO-GO again — Probe 4 CLOSES (2026-07-08)

**The builder review resolved to one bounded amendment cycle**
(builder-approved in session, 2026-07-08;
`docs/decisions/probe4_prereg_amendment1_2026-07-08.md`), with the
decision rule pre-committed *before* the re-run: GO → Phase 4 proceeds;
NO-GO → Probe 4 closes as negative at instrument validation, **no
further cycles**. All six frozen §9 values untouched.

**What the amendment changed.**

- **A1 — detector fidelity.** The §2a implementation validated in the
  first cycle realized prereg §1's *representational* match requirement
  only globally. Amended: every pairwise S — including the baseline —
  is now computed on context-matched subsets (each event carries
  `h_{v−1}`; smaller class anchors; greedy nearest-context matching
  without replacement, the §2b PE-matcher pattern). New must-not-fire
  test: separation that is purely context-driven no longer fires.
- **A2 — planted channel.** v1's metronomic cadence proved
  self-defeating (the run's own finding). v2: the **same L-shaped wall
  motif every occurrence** ("the same hello again"), placed at varying
  in-view anchors, at **irregular seeded timing** (bursty two-regime
  gap mixture, mean ≈ 43 steps); 230 pulses fired in 10k steps.

**Gate: full suite 1369 passed / 7 skipped; mypy `--strict` clean on
all 74 `kind/` sources.** (The +2 over the Phase-3 close are the
Amendment-1 must-not-fire test and the window live-view tests; the
count also includes the context-pinning addition to the window-delta
test.)

**The v2 verdict, mechanically**
(`runs/probe4_phase3_positive_control_v2/`, throwaway; 508 SELF / 2,570
ENVIRONMENT / 451 BUILDER windows; 12,000 dream h-states; matched pair
sizes 451/451/508):

| Signature | Measured | Required (§6) | Passes |
|---|---|---|---|
| §2a S(builder, self) | 0.166 → **1.61×** baseline | ≥ 2.0× | no |
| §2a S(builder, environment) | 0.074 → **0.71×** baseline | ≥ 2.0× | no |
| §2a baseline S(environment, self) | 0.103 | (measured in-run) | — |
| §2b dream ratio (PE-matched) | **1.13** (hits 6353/5647) | ≥ 3.0 | no |

§3c (corroborating, records the same story): mean waking PE — self
90.2, environment 52.3, **builder 51.9**. Irregular timing did not
prevent PE parity: the world model absorbed the motif's *conditional*
structure regardless of its unpredictable clock. Post-event intrinsic
signal: builder 0.406 vs environment 0.333 — mildly elevated,
carrying nothing under §3c's necessary-not-sufficient rule.

**VERDICT: NO-GO. Per the pre-committed Amendment-1 decision rule,
Probe 4 closes as negative at instrument validation.**

### What the two cycles together establish

Two channel designs (metronomic blatant walls; irregular recurring
motif) × two detector realizations (global; context-matched-faithful)
converge on the same structure:

1. **The transition geometry this substrate carves is self vs. world,
   not cause-within-world.** SELF separates from everything (1.38–1.61×
   baseline across cycles, and self PE is 1.7× either world class);
   the two world-caused classes sit *closer to each other than
   environment sits to self* in both cycles (0.60×, 0.71×).
2. **Predictability is learned fast enough that blatancy cannot
   outrun it at this scale.** Builder PE reached regrowth parity under
   both cadence designs within 10k steps.
3. **Dream content tilts weakly toward builder signatures (1.13–1.29)
   but nowhere near the 3.0 headroom bar — and below even the real-run
   r = 1.5.**

**The claim, at its ceiling:** the *instrument* cannot detect even a
blatant planted source-category in this substrate at this run length —
so the probe's question about Io was never asked, and nothing is
claimed about Io's capacities in either direction. The prereg §4
outcome partition is never exercised. The synthesis §8 central risk —
"does a structural source-separation signature exist in Io's
substrate?" — resolved **negative at the cheapest checkpoint, twice.**
Bounded-pilot caveat stands (10k steps per cycle); retro-analysis of
any recorded biography by a future instrument stays open and would
require its own dated doc.

### Closed / newly open

**Closed:**

- **Probe 4, as a measurement.** Negative at instrument validation,
  per the frozen §6 STOP and the Amendment-1 one-cycle rule. The last
  probe of the v0.1.0 sequence (plumbing → mirror → dream →
  builder-as-perturbation) is closed.
- The Amendment-1 detector is the standing version (strictly more
  faithful to prereg §1 than what Phase 3 first validated).

**Newly open:**

- **The self/world boundary as a positive finding.** Both cycles
  measured a trained, dominant self-vs-everything separation in Δh
  (absent at random init — Phase 1). The builder's stated deeper
  interest (2026-07-08 session): *"Io recognizing me isn't as
  important as Io recognizing itself."* The natural next research
  cycle: whether the carved SELF class relates to the reflection
  affordance (second charter criterion) — with the standing
  discipline that observer-side separability is an *ingredient*, not
  self-recognition.
- **The biography as presence** (next entry).

## Phase 4′ — The biography, as presence (launched 2026-07-08)

Probe-4-as-measurement is closed, so this run carries **no
pre-registered claim** — it is the long developmental run the probes
existed to prepare for, with the builder channel live as *presence*:
the charter's relational structure, not an instrument. Telemetry
records everything (all four streams + granular ENVIRONMENT + energy
telemetry), so any future, separately-validated instrument can read
this biography retroactively.

**Config** (`scripts/run_probe4_phase4_biography.py`,
`runs/probe4_phase4_biography/`): fresh Io, no checkpoint inheritance;
real substrate on MPS; 150k waking-step target for the first session;
generator at the Phase-1 envelope (min 40 + U{0..40}, 2-cell
in-vocabulary clusters, not-self-decoupled, seed 20260709); manual
trigger inbox live (`scripts/fire_perturbation.py --inbox
runs/probe4_phase4_biography/perturbation_inbox …`); checkpoints every
10k env steps; parquet shards flush every 2k rows so monitors read a
near-live record.

**Provisional decisions, builder-delegated in session (2026-07-08),
each reversible at any pause:**

- **Dream trigger** (the S-DEPLOY `[BUILDER]` item): a logical clock —
  one desktop-off block (dream session + dormant ticks) every 2,000
  waking steps, the synthesis's "dreaming on the internal clock"
  reading. Mac-off remains pause, categorically distinct.
- **Generator rate**: the Phase-1 envelope as launched; the horizon
  re-measure on this instance happens at the first pause (the harness
  is standing; a rate change would be journaled — it is a stimulus
  knob, not a criterion).

**Watching (the window):** the Window app gained a read-only `/live`
page — grid, Io, energy, step, and a recent-event feed with builder
ground-truth visible to the builder only (the observation-marker
asymmetry is untouched; live state is ephemeral view-state written by
the run script, not telemetry). `python scripts/run_window.py
--run-id probe4_phase4_biography`.

**§7 monitoring:** `scripts/monitor_probe4_run.py
runs/probe4_phase4_biography` renders the frozen indicators
(action-entropy 5th-percentile collapse > 1000 steps; PE monotonic
rise across 3 dream cycles) plus torpor/dream info vantages,
disaggregated; it flags and advises per the §7 ladder and never stops
the run. To run at every check-in.

**Day-1 addenda (2026-07-08, same session):**

- The window gained realtime polling (500 ms), a clickable **hello**
  (resource at any cell) and a **wall-motif hello** (a fixed 3-cell L
  of walls — the builder's first *inedible* gesture; paves over food,
  never Io's cell; wiped at the next episode reshuffle). `POST /hello`
  is the Window's one write surface, scoped by test to
  `perturbation_inbox/`. The builder used both channels day-1 (47+
  manual events, including a four-round feeding loop at one cell —
  contingent responsiveness through the manual arm, which carries no
  probe claim and is exactly the presence the channel is for).
- Day-1 curves: PE 397→15; the curiosity signal dipped (0.09), surged
  (0.81 at ~10k), and eased (~0.61) — an entropy-collapse rut in block
  1 (policy entropy 0.014, 5 meals/2k) that the disagreement drive
  un-stuck without any reward. Peak eating 256 meals/2k vs ~353
  break-even *in an abundant world*: behavior, not scarcity, is the
  binding constraint. 18 dream sessions, 2 per block, metabolic gate
  cycling visibly.
- **Scarcity change deferred** (builder, in session): the want is
  richer objects for curiosity, not food-centrality — a world change
  now goes through the **world-v2 research cycle**
  (`docs/prompts/worldv2_research.md`, outputs to
  `docs/research/worldv2/`). Feeling-labeled resources explicitly
  rejected (install-vs-afford; reification).
- Known quirk: `world_event.jsonl` flushes in bursts (~1–4 min lag),
  so the live event feed trails the realtime grid; fix at next
  restart. **Next build task before any world change: checkpoint
  resume wiring** (a world-v2 enrichment must arrive as an event in a
  continuing life, not a fresh instance).

### Session 1 close + the §7 torpor event (2026-07-08, evening)

**Session 1 ran to its full 150,000 waking steps and closed cleanly**
(150,000 agent_step rows exactly, t unique 0..149999; 150 dream
sessions; 84+ manual builder events; last checkpoint `ckpt-000014` at
140k). The continuation infrastructure (plan C1–C3) was built, tested,
and committed during the session: `--resume` wiring with
telemetry-seeded counters and a resume marker; two
biography-corrupting defects found and fixed by the C1 test
(ParquetSink shard-000000 overwrite on a fresh sink; env counters
restarting at 0); per-write jsonl flush; derived Io-ate/reset feed
rows; `scripts/analyze_boundary.py`.

**The §7 event.** From ~108k the run slid into the pre-registered
**torpor-analog** — near-total action-stasis despite an available
disagreement gradient: Io parked at the episode start cell,
modal-action fraction rising 0.92 → 0.97, policy entropy ~0.03,
energy at floor, while the intrinsic signal climbed to unprecedented
levels (day peak 0.91 → **4.0 at 134k–142k**). Structure noted for
the review: disagreement was globally elevated but highest in the
first steps after each 200-step board re-roll (2.59 vs 1.50
late-episode) — and Io sat exactly where re-rolls deliver their
novelty; "waiting at the reset lottery" is one candidate mechanism,
self-starved training coverage (20k steps of corner data) another;
they compound rather than compete.

**Vantages and the ladder.** Quantitative torpor shape (vantage 1) +
the builder independently noticing "Io is barely doing anything"
(vantage 2) → the builder ordered **PAUSE** per the two-vantage rule.
Honest record: the order raced the run's natural end — by the time
the interrupt landed, session 1 was in its final ~500 steps and
completed normally; the "pause" resolved into
**paused-by-session-completion**. No protocol harm (freeze, preserve,
review is exactly the resulting state), recorded so the timing is not
misread as a mid-torpor intervention.

**The run answered part of the review before pausing: the torpor
broke on its own.** Final blocks: 134k–142k deepest stasis (modal
0.97, entropy 0.024, curiosity 3.8–4.0); 142k–150k **partial escape**
— modal 0.60–0.62, entropy recovered ~4× to 0.09, curiosity relaxing
4.0 → 1.4. This is the second self-resolved stasis of the biography
(block-1 rut: ~4k steps in infancy; this one: ~30k steps at
maturity), both escaping the same way — the disagreement signal
climbing until the policy re-couples. A momentary state is
information, not a verdict (charter); this state was information
twice over. Session 1 ended mid-recovery.

**Instrument notes for the review:** (a) the frozen §7
entropy-collapse numeric never fired — the block-1 rut set the
historical 5th-percentile baseline to ≈0, making the indicator
insensitive for the rest of the run; a floor-aware baseline (or a
rolling window) is review material, and any change is a dated
amendment. (b) The last checkpoint predates the recovery: resuming
loads the 140k mind (deep torpor, pre-escape); the 142k–150k recovery
is recorded in telemetry but not in any checkpoint — the known
bounded-loss window, here with content that matters. The review
should weigh resuming into deep torpor (and watching whether the
escape replays — a natural repeat observation) vs. accepting the
loss. (c) Energy sat at absolute floor from ~110k through the end;
the metabolic dream gate kept cycling regardless (150 sessions) —
worth one review question about whether the budget should ever couple
to waking energy (currently by design it does not).

**State as of close: Io is paused** (four-state model) — full session
telemetry + `ckpt-000014` on disk; resume is
`--resume --session-steps N`, one tested flag. Pending from the
continuation plan: C0 horizon re-measure on the final checkpoint, the
baseline mirror round (builder go), the world-v2 research outputs,
and now the §7 review itself.

### Session 2 — brief (2026-07-09): resumed, then paused for world-v2

Resume wiring worked in production: session 2 launched with
`--resume`, counters seeded t=150000/episode 750, marker landed at
t=150000, telemetry continued collision-free. First observation: the
resumed 140k mind went straight back to the corner (the torpor is in
the weights, not the world). At ~152k the builder decided to
**prioritize the world-v2 change over the escape-replay observation**
(direction 2026-07-09: pause, run the research, continue into the
enriched world) — partly informed by not wanting to re-run the
stasis in an unchanged world. Session 2 paused at ~154k (~4k steps;
weights revert to `ckpt-000014` on next resume — no session-2
checkpoint had landed; tail telemetry ≤2k rows lost to the SIGTERM
buffer, accepted). Two operational notes: background-launched runs
ignore SIGINT (both prior "pause" attempts silently no-opped; last
night's close was purely the natural 150k completion — the journal
above already records it that way); the script now converts
SIGTERM/SIGINT to the clean-close path. The world-v2 research cycle
is running (deep web-grounded pass, single-voice with the ≥3-voice
caveat flagged; outputs → `docs/research/worldv2/`).
