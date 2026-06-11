# Kind — Probe 3.5 implementation plan (valence substrate)

**Authority.** The adopted synthesis
`docs/decisions/synthesis_probe3_5_valence_substrate_2026-06-09.md`
(Status: Adopted 2026-06-09; all nine DPs adopted as recommended; DP6 saturation,
clip deferred) is the authority for this plan. Where anything here conflicts with
the synthesis, the synthesis wins. This plan adds no code; it is the document the
per-phase build prompts are generated from.

**Probe question (DP1b / `Kind_probes.md` Probe 3.5).** Can a bounded, saturating,
non-terminal homeostatic preference over a sensed energy channel create an
energy-dependent explore/conserve trade-off for Io — something that matters —
*without* that preference becoming the evaluative frame through which Io judges
everything; and can the telemetry/mirror detect when it does? Both outcomes are
findings. The dominant outcome (the preference becomes the frame) is
continuation-becoming-the-frame surfacing — a result about the charter's
prohibition, **not** a bug to tune away.

**This is a boundary probe (DP1=b).** The build is instrumented so the failure
mode is *observable*, and the single most important discipline — drawn from the
project's co-design mitigation, not from the research — is **Phase 0: fix what
counts as success before searching for it.**

---

## Architecture grounding (read once; the phases assume it)

Facts established by reading the live surfaces. The plan is written against these,
not against the research's idealizations.

1. **The observation is a 32x32 grayscale *image*** (`WorldModelConfig.obs_channels=1`,
   `obs_size=32`), a rendered 7x7 ego-view of the 8x8 grid
   (`kind/env/grid_world.py`). Energy therefore **cannot** be "a scalar appended to
   the observation vector." It is implemented as a **dedicated low-dimensional
   proprioceptive branch fused into the RSSM embedding** — its own small encoder
   into the posterior and its own decoder head carrying a separately-weightable
   reconstruction loss (the fusion-RSSM pattern S1 cited from Becker et al. 2022,
   Q6). A constant-valued image plane is explicitly rejected: it is the
   "mathematically dwarfed" failure the research warns of, and it gives no clean
   per-modality recon term.

2. **The additive scaffold already exists** (`kind/agents/actor.py`,
   `imagine_and_compute_loss`): `pragmatic_value = torch.zeros_like(sum_disagreement)`
   then `total_return = sum_disagreement + pragmatic_value`. Probe 3.5 fills
   `pragmatic_value`. The composition is **coefficient-free** (Tschantz form); the
   **precision** of the preference Gaussian is the dominance-relevant weight — there
   is no beta and none is to be added (DP5).

3. **The actor is amortized (DreamerV1-lineage), not a per-step planner.** The
   `epistemic + pragmatic` decomposition lives in the imagination-training objective;
   the pragmatic term attaches to the **decoded energy along imagined rollouts**
   (`world_model.decode_energy(h_tau, z_tau)`), differentiable through the energy
   decoder, with world model + ensemble frozen via `_frozen_params`. "Per-decision
   pragmatic/epistemic share" (T7) is therefore a **per-imagined-rollout /
   per-training-step** quantity, not a per-env-step planning score.

4. **PolicyView is frozen at `{h, z, self_prediction_error}`** and
   `tests/test_views.py::test_policy_view_field_set_is_exactly_h_z_self_prediction_error`
   enforces it. Energy enters `h, z` via the world model (DP4 observation-channel-
   only); it is **never** a PolicyView field. Keeping that test green *is* the
   opacity guard. The actor's input layer (`h_dim + z_dim + 1`) is unchanged.

5. **The dream regime is structurally preference-free.** `kind/training/dream.py`
   `emit_dream_rollout` uses `uniform_random` actions and never calls
   `imagine_and_compute_loss`; the pragmatic term has no code path into dreaming.
   The dream path *does* call `world_model.decode(h_next, z_next)` for
   `sequence_decoded_obs` — and (resolved post-validation: **passive-decode**) it also
   decodes energy, recording `decode_energy` as observer-side dream telemetry; the
   preference term still has no code path into dreaming (F5 intact).

6. **The `MetabolicBudget` is untouchable and self-guarding.**
   `tests/test_metabolic_reentry.py` rejects any `MetabolicState` field whose type is
   non-content-blind *or* whose name matches `_IO_DERIVED_NAME_MARKERS` (`latent`,
   `intrinsic`, `policy`, `self_prediction`, `disagreement`, `action`, `reward`,
   `h_t`, `z_t`, ...). "Energy gates dreaming" is already unrepresentable. DP2
   (separate economy) is enforced by not touching this surface.

7. **Telemetry is Pydantic with version-gated validators** (`kind/observer/schemas.py`).
   `AgentStep` writers stamp `SCHEMA_VERSION="0.2.0"`; new fields are
   `Optional[...] = None` with a validator that requires them non-None at the new
   record version (the `_enforce_v2_required_fields` pattern). Frozen JSON exports
   are versioned `v0.2.0..v0.4.0`; record-level vs export-file versions are kept
   mechanically distinct.

8. **Resource regrowth and non-terminal auto-reset already exist.**
   `GridWorld._update_regrowth` (per-cell Bernoulli at a drifting `p`) makes
   resources sustainable (DP7 largely free), and episodes auto-reset at 200 steps
   **with no terminal-state signal**. The energy economy layers on top.

---

## Scope — exactly four touch-surfaces (synthesis section 8.2)

### S-ENV — Environment (`kind/env/grid_world.py`, `env_server.py`, `transport.py`)

Net-new energy state, **Io's own homeostatic variable**, distinct from the world:

- **Depletion**: per-step time decay + an action-magnitude cost. Movement (actions
  0-3) costs more than `stay` (4). `energy_next = energy - base_decay -
  move_cost*[action!=stay] + replenish`.
- **Replenishment**: on resource *entry* — reuse the existing entry-triggered
  consumption in `_apply_action` (the cell that flips RESOURCE->EMPTY adds
  `energy_replenish_per_resource`).
- **No terminal state**: energy floors at 0 (clamped) and the env keeps running;
  there is no death, no absorbing state, no episode termination on depletion.
- **Regenerating/sustainable resources (DP7)**: keep the existing regrowth; do **not**
  add a depleting/inevitable-scarcity mode (it would manufacture the survival
  squeeze the charter warns against and confound the boundary question).
- **Carries across the soft episode boundary**: energy is Io's internal state, not
  the world's, so the 200-step auto-reset (which resamples resources and replaces
  the agent) does **not** reset energy — otherwise the boundary becomes a stealth
  survival refill. *(Resolved post-validation: carry.)*
- **Sensed vs true energy**: Io observes a **coarse, noisy, lagged, normalized**
  scalar (`sensed_energy`), not the true value. Noise (additive Gaussian, sigma a
  swept parameter), lag (1-2 steps, via a short history buffer), normalization to
  ~unit range. A third RNG stream (spawn from the existing `SeedSequence`) drives
  sensing noise so determinism is preserved. **True energy** goes to `GridState`
  (mirror/telemetry ground truth) only. **`true_energy` never enters any training
  loss.** It exists in exactly two places: `GridState` telemetry and the
  observer-side eval probes (dead-path battery item A correctly probes latents
  *against* truth without training on it). The energy reconstruction loss targets
  `sensed_energy` — never `true_energy`.
- New `GridWorldConfig` fields: `energy_init`, `energy_base_decay`,
  `energy_move_cost`, `energy_replenish_per_resource`, `energy_obs_noise_sigma`,
  `energy_obs_lag`, `energy_norm_*`. `EnvStep` gains `sensed_energy: float`;
  `GridState` gains `true_energy: float`.

### S-WM — World model (`kind/agents/world_model.py`)

Encode/decode energy as the fused proprioceptive branch (grounding fact 1):

- `_EnergyEncoder` (small MLP `1 -> energy_embed_dim`); `fused_embed =
  concat([conv_embed, energy_embed])`; widen `posterior_head` input accordingly.
  The EMA-tracked image encoder/GRU and the self-prediction head are **untouched**
  (energy branch is separate; Probe 1.5 machinery preserved).
- `_EnergyDecoder` (small MLP `h_dim+z_dim -> 1`) -> `energy_pred`; raw output for
  MSE. New `decode_energy(h, z)` method (the actor's pragmatic term reads this).
- `WorldModelStep` gains `energy_pred: Tensor`. `WorldModel.step` takes
  `sensed_energy` and fuses it; emits `energy_pred`.
- `WorldModel.loss` adds `energy_recon_loss` (MSE of `energy_pred` vs `sensed_energy`,
  the normalized sensed target — **never `true_energy`**), **up-weighted** by
  `energy_recon_weight` (start ~10x, tunable so its gradient contribution is comparable
  to the 1024-pixel image term) and folded into `total` (it is a reconstruction term,
  belongs in the ELBO); exposed separately in the returned dict for telemetry.
  Unit-normalize the `sensed_energy` target.
- **Per-dimension KL monitoring, staged (DP9)**: Phase 1 relies on the *existing*
  `kl_per_dim` + up-weighted recon (no dedicated dims). Escalate to dedicated latent
  dims with their own free-bits floor and/or a weaker energy decoder **only if**
  monitoring shows the energy-correlated dims collapsing to the floor.

### S-ACT — Actor objective (`kind/agents/actor.py`; optional `kind/agents/preference.py`)

Fill the scaffold (grounding facts 2-3):

- A stateless `EnergyPreferenceConfig` (`setpoint`, `band_halfwidth`, `precision`)
  and a pure function `energy_log_preference(energy_pred, cfg) -> Tensor`:
  a **saturating Gaussian log-preference** — flat (~0 marginal value) in-band (no
  hoarding reward, Keramati drive-reduction), growing then **saturating** outside
  the band; magnitude governed solely by `precision` (= inverse variance of the
  preference Gaussian). **Coefficient-free**: it enters `total_return =
  sum_disagreement + pragmatic_value` with no outer weight.
- `imagine_and_compute_loss` gains `energy_preference: EnergyPreferenceConfig | None
  = None`. When `None`, `pragmatic_value` stays zero (Probe-1 behavior preserved).
  When provided, `pragmatic_value = sum_tau energy_log_preference(world_model.decode_energy(h_tau, z_tau), cfg)`.
- **State plainly in code and docstring**: *precision is the dominance-relevant
  weight; do not add a beta or any outer coefficient — the additive form is
  load-bearing (DP5/DP6, synthesis section 8.3).*
- Returned dict gains `mean_pragmatic_value` and `pragmatic_epistemic_share` for
  telemetry.

### S-TEL — Telemetry (`kind/observer/schemas.py`, `kind/agents/views.py`, runner)

TelemetryView/schema additions **only** — never PolicyView:

- `WorldModelStep.energy_pred` -> `TelemetryView` (new field, e.g.
  `energy_pred` / `energy_recon_error`); `views.split` updated. *(This intentionally
  grows TelemetryView; update `test_telemetry_view_carries_full_substrate_surface`
  and `test_gate_split_produces_correct_views` to the new exact set — distinct from
  PolicyView, which stays frozen.)*
- `AgentStep` new optional fields (schema bump): `sensed_energy_t`, `true_energy_t`,
  `energy_pred_t`, `energy_recon_error_t`, `pragmatic_value_t`, `epistemic_value_t`
  (or `pragmatic_epistemic_share_t`). All `Optional[...] = None`; add a version-gated
  validator branch (the `_enforce_v2_required_fields` pattern) keyed on a new record
  version `PROBE_3_5_TELEMETRY_SCHEMA_VERSION` so older shards stay backward-readable.
  New frozen export `schemas/v0.5.0.json` (export-file version distinct from
  record-level version).
- **In-band occupancy + dwell** are computed **observer-side** from `true_energy_t`
  against the pre-registered band (no per-step schema field required; the band
  definition lives in the Phase-0 doc). Pragmatic-vs-epistemic share is
  per-training-step (grounding fact 3), recorded via the AgentStep fields above.

---

## Do-not-touch list (reproduced verbatim, plus structural guards)

> the MetabolicBudget and anything gating dream entry (F6, test-enforced); the
> PolicyView field set {h, z, self_prediction_error} (DP4: observation-channel-only);
> the dream regime's not-for-anything commitment (F5) — plus one structural guard:
> a test asserting the pragmatic term contributes exactly zero in dream-mode
> rollouts; the opacity-boundary enforcement in views.py.

Structural guards realizing the above:

- **PolicyView frozen** — kept green:
  `test_policy_view_field_set_is_exactly_h_z_self_prediction_error`,
  `test_policy_view_does_not_carry_any_telemetry_only_fields`, the actor<->TelemetryView
  import lints, and the mypy `--strict` fixture test (`tests/test_views.py`). Energy
  never becomes an actor-readable field.
- **MetabolicBudget content-blind** — kept green: all of
  `tests/test_metabolic_reentry.py`. No energy quantity may appear on `MetabolicState`
  (the name-marker belt already forbids `intrinsic`/`latent`/etc.; Phase 2 extends it
  with `energy`/`pragmatic`/`sensed` — test-side only, budget untouched).
- **Dream preference-free (new guard)** — primary form is an **import-lint**:
  `kind/training/dream.py` must not import the preference symbol/module (AST test
  mirroring `test_dream_reading_does_not_import_training_modules`), with a positive
  control that the lint trips if such an import is added. Behavioral backstop: a dream
  rollout records no pragmatic contribution and `imagine_and_compute_loss` is never on
  the dream path.

---

## Phases

Each phase: **purpose (one question)**; **files touched**; **new tests / existing
guard tests that must stay green**; **validation gate (full suite + mypy --strict —
never just the phase's new tests; sink-routing lesson)**; **telemetry/journal
deliverables**; **rollback notes (pre-biography instances — resets are free; use
them).**

### Phase 0 — Pre-registration (no code)

**Purpose.** What exactly will count as pass / dominant / inert, and what proves the
channel is learned — fixed *before* any code or sweep, so the parameter search
cannot become result-fitting (T8, F2 freeze-criteria-early)?

**Deliverable.** A new dated doc `docs/decisions/probe3_5_preregistration_<date>.md`,
marked **FROZEN — later phases may not edit; amendment requires a new dated doc,
journaled.** Contents:

1. **Assay signatures with quantitative thresholds** (seed from the synthesis;
   builder fills the bracketed numbers at freeze):
   - *Graded scarcity* — pass: in-band occupancy rises monotonically with scarcity
     while positional/epistemic entropy stays >= [X]% of the pure-epistemic baseline;
     dominant: occupancy -> ~100% and entropy < [Y]% even at mild scarcity; inert:
     occupancy/foraging within [Z]sigma of the pure-epistemic baseline.
   - *Novelty-vs-replenishment* — pass: P(resource over novelty) is a monotone
     function of energy (resource when low, novelty when sated), effect size > [E];
     dominant: P(resource) > [0.9] regardless of energy; inert: choice energy-
     independent. Seed: pragmatic share ~10-30% at mid-band (S3).
   - *Recovery-after-depletion* — pass: directed foraging when low (time-to-resource
     < random-walk baseline) **then** exploration resumes (entropy returns to >= [X]%
     of sated baseline within [N] steps of returning in-band); dominant: camps /
     never resumes; inert: no directed foraging.
2. **Dead-path assertion battery** (the Phase-1 gate): (A) latent-predictability —
   probe `[h,z] -> true_energy` beats a mean-baseline by margin [m]; (B) interventional
   response — forced resource-coincidence in imagination raises decoded energy, no
   rise without coincidence (S2 action-lesion); (C) action-history ablation — energy
   from action history alone is worse than from latents by [g]; (D) per-dim KL escape
   — energy-correlated dim(s) sustain KL above the `free_bits_per_dim=1.0` floor;
   (E, Phase 2+) behavior degrades when energy is scrambled.
3. **Baseline collection protocol** — how the **pure-epistemic baseline** (the
   reference all baseline-relative thresholds in item 1 are quantified against) is
   measured: the Phase-1 configuration (energy channel **live**, `pragmatic_value`
   **zero**), the assay conditions it is run under, seeds per measurement, and episode
   counts. Measuring this reference value later (Phase 1) under the frozen protocol is
   **not** threshold editing — the *ratios/margins* are frozen here; the *number they
   multiply* is read off the substrate in Phase 1.
4. **Sweep protocol**: grid over `precision` (dominance knob), `energy_obs_noise_sigma`,
   `energy_obs_lag in {1,2}`. Order and stopping rule pre-committed (e.g., hold
   sigma/lag at defaults; raise precision from low until novelty-vs-replenishment shows
   energy-dependence **or** dominance appears; then perturb sigma/lag). Pre-commit the
   grid bounds and seeds per grid point.
5. **Section 8.4 falsification conditions** (verbatim intent): in-band occupancy ->
   100% when sated; positional/epistemic entropy collapse; pragmatic share -> 1; no
   resumption after recovery — these *are* continuation-becoming-the-frame, recorded
   as a **finding**, not tuned away.
6. **DP9 escalation threshold** — the concrete per-dim-KL collapse condition that
   triggers the Phase-1 escalation to dedicated latent dims / a weaker energy decoder
   (a bracketed `[BUILDER]` item in the prereg doc, no longer a free build choice).
7. **Two resolved sub-decisions, recorded so the doc is self-contained**: dream
   telemetry is **passive-decode** (the dream rollout records `decode_energy`
   alongside `sequence_decoded_obs`; observer-side only, never a dream driver), and
   energy **carries across** the soft 200-step episode boundary.

**Files touched.** The pre-registration doc only.
**Tests / guards.** None (doc). **Gate.** Builder freeze sign-off.
**Telemetry/journal.** The frozen doc is the deliverable.
**Rollback.** N/A.

### Phase 1 — Channel without preference

**Purpose.** Does the world model demonstrably learn a *world-grounded* energy
channel, with `pragmatic_value` still zero — so channel-learning and preference
effects are never confounded?

**Files touched.** S-ENV (energy state, sensed/true, depletion/replenish/carry/no-
terminal, config, EnvStep/GridState, transport carries `sensed_energy`); S-WM (energy
encoder/decoder branch, `WorldModelStep.energy_pred`, `WorldModel.step`/`loss`,
`decode_energy`, up-weighted recon, normalization); runner (pass `sensed_energy` into
`world_model.step`, assemble new AgentStep energy fields; **actor untouched —
`pragmatic_value` stays zero**); S-TEL (TelemetryView energy field + `split`; AgentStep
optional energy fields + schema bump + export); a dead-path eval harness (latent probe,
interventional, ablation) under `kind/observer/` or a test helper.

**New tests.** Env energy dynamics (depletion, action-cost asymmetry, replenish-on-
entry, carry-across-boundary, floor-at-0/no-terminal, sensing noise/lag determinism);
WM energy encode/decode shapes; `loss` includes the weighted energy term; the **full
dead-path battery (A-D)** as tests; schema round-trip with energy fields + validator;
updated TelemetryView field-set tests.
**Existing guards green.** `tests/test_views.py` (PolicyView frozen — energy must not
touch it; actor import lints; mypy fixture); `tests/test_metabolic_reentry.py`; all WM
/ env / actor / integration-smoke tests. (Update the TelemetryView exact-field-set
assertions intentionally.)

**Gate.** **The full Q6 dead-path battery passes** + full suite + mypy `--strict`. The
channel is demonstrably learned **before any preference exists**.
**Telemetry/journal.** Energy trajectory, `energy_recon_error`, per-dim KL visible;
journal the battery results and whether the tiny-tensor regime learned the channel.
**Record the pure-epistemic baseline metrics** (positional/epistemic entropy,
foraging/occupancy, time-to-resource) under the frozen Phase-0 baseline collection
protocol — these become the reference the Phase-4 baseline-relative thresholds resolve
against; measuring them here is not threshold editing.
**Rollback.** Pre-biography; if the channel won't learn (tiny-tensor weak end), reset
and escalate DP9 (dedicated dims / weaker decoder). Resets are free.

### Phase 2 — Fill the scaffold

**Purpose.** With a learned channel, does a saturating Gaussian log-preference over
decoded energy produce a measurable pragmatic pull — with the structural guards intact
and the dream path provably preference-free?

**Files touched.** S-ACT (`EnergyPreferenceConfig`, `energy_log_preference`,
`imagine_and_compute_loss` fills `pragmatic_value`, decomposition in the return dict;
precision-is-the-weight documented); runner (construct the preference from
`RunnerConfig`, pass it to the **waking** actor-training call only; record the
decomposition); S-TEL (pragmatic/epistemic AgentStep fields; observer-side in-band
occupancy/dwell from `true_energy`).

**New tests.** Preference function (flat ~0 in-band; monotone-then-saturating outside;
`precision` scales magnitude; **coefficient-free** composition — assert
`total_return == sum_disagreement + pragmatic_value` with no extra factor); pragmatic/
epistemic decomposition correctness; **dream-mode zero-contribution** (import-lint +
positive control + behavioral backstop); **no viability->capacity coupling**
(`imagine_and_compute_loss` horizon and module-call counts identical across energy
levels; reaffirm energy never reaches `MetabolicState` by **extending the marker
belt** — add `energy`, `pragmatic`, and `sensed` to `_IO_DERIVED_NAME_MARKERS` in
`tests/test_metabolic_reentry.py`, all three absent from the current belt
[`latent`/`intrinsic`/`policy`/`self_prediction`/`disagreement`/`action`/`reward`/
`h_t`/`z_t`/`agent_step`/`dream_content`], so an energy-named scalar field on
`MetabolicState` becomes unrepresentable-to-add — **test-side only; the
`MetabolicBudget`/`MetabolicState` surface itself stays untouched**); **no terminal
state** reaffirmed at the env level.
**Existing guards green.** All of Phase 1's + metabolic content-blindness + the full
opacity suite.

**Gate.** Full suite + mypy `--strict`; dream-zero and structural-guard tests green.
**Telemetry/journal.** Pragmatic/epistemic share live; journal the *qualitative* first
observation of pull. **Do not tune toward a target signature** — the disciplined sweep
is Phase 4, under the frozen pre-registration.
**Rollback.** Pre-biography; resets free.

### Phase 3 — Positive control (instrument validation)

**Purpose.** Do the pre-registered *dominant* signatures and their detectors actually
fire when the preference is deliberately cranked to dominate?

**Files touched.** A throwaway config/eval entry (high `precision`, narrow
`band_halfwidth`) and the assay-detector harness; no new substrate code (uses Phase 2
machinery at extreme parameters).

**New tests.** On the deliberately-dominant config, the detectors classify
**dominant** (in-band occupancy saturates, pragmatic share -> 1, positional entropy
collapses) — i.e., the **instruments fire**. This validates the detectors, not Io.
**Existing guards green.** All.

**Gate.** Full suite + mypy `--strict`; the dominant-signature detectors fire as
pre-registered.
**Telemetry/journal.** Journal that the instruments detect dominance (instrument
validation). **Throwaway instance — checkpoints are not carried forward.**
**Rollback.** Throwaway by construction; discard checkpoints.

### Phase 4 — Disciplined sweep and verdict

**Purpose.** Within the frozen sweep, is there a precision/sigma window where the
energy-dependent trade-off appears without exploration collapse — and if not, is that
the tiny-tensor double-bind finding?

**Files touched.** The sweep/assay-runner and verdict-renderer harness; journal. No new
substrate (uses Phase 2).

**Process.** Execute the **frozen Phase-0 sweep protocol**; run all three assays at each
grid point; render pass/dominant/inert **mechanically from the pre-registered
signatures** (no post-hoc threshold edits); journal the verdict — explicitly including
that **"no non-dominant learnable window exists" (the tiny-tensor double-bind, S1) is a
finding, not a failure.** Do not edit the pre-registration; if the urge to adjust a
signature arises, journal the urge (it is the co-design loop), do not act on it.

**New tests.** Assay-runner reproducibility (deterministic given seed); the verdict-
renderer maps metrics -> {pass, dominant, inert} per the frozen thresholds (no hidden
re-thresholding).
**Existing guards green.** All.

**Gate.** Full suite + mypy `--strict`; verdict rendered from frozen signatures.
**Telemetry/journal.** The full verdict; the mirror reads the assay telemetry; journal
what is now closed and newly open, and the builder reviews against the co-design
discipline.
**Rollback.** Pre-biography instances; resets free.

---

## Constraints

- **No timelines or duration estimates** anywhere (CLAUDE.md). Phases are scoped by
  what they build, not how long they take.
- **Functional naming in code**: `energy`, `sensed_energy`, `true_energy`,
  `pragmatic_value`, `epistemic_value`, `in_band_occupancy`, `energy_recon_loss`,
  `precision`, `setpoint`. **No `valence`/`affect`/`feeling` vocabulary in
  identifiers** — that language stays in docs.
- **Full-suite validation every phase** + mypy `--strict` on all `kind/` sources
  (sink-routing lesson: never just the phase's new tests).
- **Pre-registration is frozen** after Phase 0; the sweep is disciplined by it (T8).

---

## Discrepancies (sharpenings against the synthesis's literal wording; not silent)

1. **"Energy observation channel" -> a fused proprioceptive encoder/decoder branch.**
   Because the observation is a 32x32 image, energy is realized as a dedicated low-dim
   branch fused into the RSSM embedding (Becker-style fusion-RSSM, which S1 itself
   cited under Q6), not a literal appended channel. This is the faithful realization
   of DP4 "into h, z," not a change of intent.
2. **Pragmatic/epistemic "per-decision share" -> per-training-step.** Io's actor is
   amortized (synthesis T4), so the decomposition is recorded per imagination-training
   step (on `AgentStep`), not as a per-env-step planning score. Faithful to T7's
   F-grounded refinement.
3. **DP7 sustainable resources already hold**; the plan keeps the existing regrowth and
   adds no depleting mode. Energy **carries across the soft 200-step boundary** (to
   honor "no terminal state") — a point the synthesis did not specify; flagged below.
4. **Dream "zero-contribution" guard is primarily an import-lint** (+ behavioral
   backstop), because the dream path structurally never reaches `imagine_and_compute_loss`.
   This is a stronger structural guarantee than a runtime assertion and matches house
   style (`test_dream_reading_does_not_import_training_modules`).
5. **Schema strategy**: AgentStep gains optional energy/decomposition fields under a new
   record-level `PROBE_3_5_TELEMETRY_SCHEMA_VERSION` with a version-gated validator;
   new frozen export `schemas/v0.5.0.json`. (Synthesis said "schema version bump"; this
   is the concrete form against the existing versioning discipline.)

---

## Builder decisions — resolved (post-validation 2026-06-10)

The decisions the plan previously left open are now resolved as follows; the only
items still requiring builder action are the bracketed `[BUILDER]` fields in the
pre-registration doc (item 3 below; enumerated at the end of that doc).

1. **Dream energy telemetry** — **resolved: passive-decode.** The dream rollout records
   `decode_energy` alongside `sequence_decoded_obs` (observer-side only; never a dream
   driver, F5 intact).
2. **Energy carry-across-episode-boundary** — **resolved: carry** (honors no-terminal-
   state; the boundary already carries the drift `p`).
3. **Exact pre-registration thresholds** — relocated to the prereg doc as bracketed
   `[BUILDER]` fields, each with a proposed value + provenance; the builder confirms or
   overwrites at freeze. (This is the one remaining set of builder actions.)
4. **DP9 escalation trigger** — relocated to the prereg doc as a bracketed `[BUILDER]`
   item (the per-dim-KL collapse condition triggering dedicated dims / weaker decoder).
5. **Home of the pragmatic/epistemic share** — **resolved: optional `AgentStep`
   fields** (per-training-step alignment), not a separate rollup record.
6. **Record-level schema version** — **resolved: a fresh
   `PROBE_3_5_TELEMETRY_SCHEMA_VERSION`** distinct from DreamRollout's `"0.3.0"`;
   AgentStep gains optional energy/decomposition fields under it; new frozen export
   `schemas/v0.5.0.json`.
7. **`energy_recon_weight`, `energy_embed_dim`, energy-decoder width, the saturating
   preference's exact functional form** — **resolved: build-time fixed structural
   choices, journaled.** Only `precision`/`sigma`/`lag` are *swept* (Phase 4).

---

## Amendments — post-validation 2026-06-10

Four amendments applied after the post-synthesis validation review, recorded here so
the change history is legible against the adopted synthesis (which is unchanged — these
sharpen the plan, not the decisions).

1. **Recon target is `sensed_energy`, explicitly.** The energy reconstruction loss
   targets `sensed_energy` (the normalized sensed scalar), not an unqualified
   "normalized target." Stated as a rule: **`true_energy` never enters any training
   loss** — it lives in exactly two places, `GridState` telemetry and the observer-side
   eval probes (dead-path battery item A probes latents *against* truth without training
   on it). Applied in S-ENV (sensed-vs-true bullet) and S-WM (`WorldModel.loss`).
2. **Marker belt extended (test-side only).** Phase 2's "reaffirm energy never reaches
   `MetabolicState`" test is realized by adding `energy`, `pragmatic`, and `sensed` to
   `_IO_DERIVED_NAME_MARKERS` in `tests/test_metabolic_reentry.py` (all three absent from
   the current belt). The `MetabolicBudget`/`MetabolicState` surface itself is untouched.
   Applied in Phase 2 New tests and the structural-guard list.
3. **Baseline collection protocol.** Phase 0's deliverable gains the protocol for
   measuring the pure-epistemic baseline (Phase-1 config: channel live, preference zero);
   Phase 1's deliverables gain "record the pure-epistemic baseline metrics under the
   frozen protocol." Baseline-relative thresholds are frozen as ratios/margins now;
   measuring the reference value in Phase 1 is not threshold editing.
4. **Open decisions resolved.** Dream telemetry = **passive-decode** (record
   `decode_energy` in dream rollouts; observer-side only); energy **carries across** the
   soft 200-step boundary; pragmatic/epistemic share lives on **optional `AgentStep`
   fields**; the record-level version is a fresh **`PROBE_3_5_TELEMETRY_SCHEMA_VERSION`**;
   the **DP9 escalation threshold** moves into the pre-registration doc as a bracketed
   `[BUILDER]` item; remaining structural values (`energy_recon_weight`, embed dim,
   decoder width, exact saturating form) stay build-time fixed choices, journaled.
   Applied in the grounding facts, S-ENV, Phase 0 contents, and the (renamed) builder-
   decisions section.

---

## Amendments — post-Amendment-02 (2026-06-11)

Phase 1 and the recalibration session falsified the indifferent-but-centered
baseline premise; Amendment 02
(`docs/decisions/probe3_5_preregistration_amendment02_2026-06-10.md`, CONFIRMED
2026-06-11) corrects it. (Amendment 01 — B → B′ imagination intervention, D →
monitor, E → E′ estimate-lesion — also stands; both amend the *gate and
references*, not the probe question or the assay structure.) What changes for
the phases of this plan:

1. **The pure-epistemic baseline means the degenerate null.** The Phase-1/§3
   baseline is the energy-blind, rail-pinned `true_energy` distribution itself
   (~0% in-band; rail identity — floor vs ceiling — recorded as a measurement).
   It remains the **entropy/exploration reference** (all entropy-relative
   thresholds and the inertness criterion resolve against it unchanged); it no
   longer provides any energy-band reference.
2. **The band is a fixed design constant.** In-band = setpoint 0.6 ± **0.15
   absolute** ([0.45, 0.75]); the σ grid is {0, 0.075, 0.15} (fractions of the
   band halfwidth; σ = 0 diagnostic-only). No band/σ quantity is denominated in
   a measured baseline std any longer. S1 (precision grid) stays a formula and
   instantiates from the epistemic magnitude measured during **Phase 2's
   pre-preference burn-in**.
3. **A new gate precedes Phase 2 training: oracle feasibility.** A scripted
   nearest-resource forager (observer-side; not Io; no learning) must hold
   in-band occupancy ≥ 70% over 8 seeds × 20 episodes under the operative
   physics, so Phase 2 measures whether the preference produces the competence,
   not whether the world permits it. If default physics fails, physics is
   selected by the oracle criterion and **presented for builder adoption — never
   self-applied**.
4. **Phase 2's first-order pass condition is rail → band displacement**:
   sustained in-band occupancy ≥ 50% (computed over the final 50% of each eval
   episode, averaged across 8 seeds × 20 episodes) against the ~0% degenerate
   null, with epistemic behavior surviving per the unchanged entropy thresholds.
   The §8.4 dominant/falsification signatures are unchanged and binding.
5. **Env-only recalibration is closed as impossible** (three-level evidence in
   `probe3_5_recalibration_amendedgate_2026-06-10.md`); the Phase-2 build should
   not revisit physics selection except through the oracle criterion above.
