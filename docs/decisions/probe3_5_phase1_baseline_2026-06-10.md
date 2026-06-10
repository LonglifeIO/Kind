# Probe 3.5 — Phase 1 results & baseline instantiation — 2026-06-10

**Status: results record (Phase 1 — channel without preference).** Authority for
the criteria: the frozen pre-registration
`docs/decisions/probe3_5_preregistration_2026-06-10.md` (FROZEN 2026-06-10).
Per the pre-registration standing rule, *instantiating* the baseline-relative
formulas from a measured run is not threshold editing. **No frozen threshold was
changed in producing this record.**

Phase-1 question: *does the world model demonstrably learn a world-grounded energy
channel, with `pragmatic_value` still zero, so channel-learning and preference
effects are never confounded?*

**Short answer: the channel is learnable *in principle* but not cleanly learned
under the pure-epistemic regime — an informative failure, the tiny-tensor
double-bind (synthesis §7) manifesting in the behavior distribution.** The full
A–D battery does **not** pass at the frozen margins. The disciplined response
(boundary-probe framing DP1b; co-design discipline) is to record this and surface
it to the builder — **not** to tune the environment until the battery greens.

---

## 1. What was run

- **Substrate**: the Phase-1 build (energy economy in the env; fused
  proprioceptive branch in the world model; energy telemetry at
  `PROBE_3_5_TELEMETRY_SCHEMA_VERSION`; actor untouched, `pragmatic_value`
  identically zero). Full test suite green (1235 passed), mypy `--strict` clean.
- **Gate run** (`scripts/run_probe3_5_phase1_baseline.py`, default substrate,
  P3 = 5000 env-steps, seed 0): the runner trains world model + ensemble + actor
  (epistemic-only); the dead-path battery A–D is then run on an eval rollout, and
  the pure-epistemic baseline metrics are measured.
- **Controlled learnability probe**: a world-model-only training run on a
  **variance-rich** (uniform-random) behavior distribution (~4000 steps, 1500
  batched updates), to separate *can the WM learn the channel* from *does the
  epistemic actor's distribution let it*.

---

## 2. Battery A–D verdicts (frozen margins)

| Battery | Margin (frozen §8) | Gate run (epistemic actor) | Controlled probe (variance-rich) |
|---|---|---|---|
| **A** latent-predictability | R² ≥ 0.50 | **FAIL** R² = −437 † | **PASS** R² = 0.55 |
| **B** interventional response | rise ≥ 0.80 | **FAIL** 0.006 | FAIL 0.012 |
| **C** action-history ablation | MSE ratio ≥ 1.5 | **FAIL** 0.02 † | **PASS** 2.34 |
| **D** per-dim KL escape | ≥ 1 dim ≥ 1.5 nats | **FAIL** max 1.26 | FAIL max 0.97 |

† **A and C on the gate run are uninformative, not substantively failing**: under
the trained epistemic actor `true_energy` collapses to the floor (mean **0.011**,
std **0.064**, in-band **0.4%**), so the probe *target has almost no variance* —
R² and the MSE ratio are unstable/meaningless against a near-constant target.

**The substantive findings:**

1. **The channel is learnable in principle.** On a variance-rich distribution the
   WM decodes `true_energy` from `[h, z]` (A: R² = 0.55, beats the mean baseline)
   and does so far better than from action history (C: 2.34× ≥ 1.5) — it is
   **world-grounded, not an action-history artifact**.
2. **But it rides the deterministic recurrent state `h`, not the stochastic
   latent `z`.** B (responsiveness to the fused energy *observation*) ≈ 0 and D
   (per-dim KL) never reaches the 1.5-nat margin (max 0.97–1.26, the posterior
   runs in a sub-1-nat regime). Energy is recoverable from the trajectory `h`
   integrates (decay + resource entries), so the fused sensed *observation* is
   informationally **redundant** — the model ignores it and the stochastic
   channel stays near the floor.
3. **DP9 escalation does not rescue it — it degrades the channel.** Taking the
   pre-registered escalation (`energy_dedicated_dims = 2`: weaker z-only decoder
   + raised free-bits floor) pushed D's max KL up (0.97 → 1.35, still < 1.5) but
   drove **A negative (R² = −0.64) and C below 1** — forcing energy through the
   noisy stochastic latent loses the clean deterministic signal. This is the
   **tiny-tensor double-bind (synthesis S1/§7)**: "every tuning is inert or
   dominant" — here, every routing is *predictable-via-`h`* or *noisy-via-`z`*.
4. **The pure-epistemic actor depletes energy to the floor.** With no preference,
   Io has no reason to maintain energy; movement costs deplete it and it floors
   over the run. The double-bind recurs in the *env dynamics*: energy fast enough
   to vary depletes to the floor without foraging, and the only thing that would
   maintain it (foraging) requires the very preference Phase 1 omits. **Energy
   becomes a live, maintained variable only once there is a reason to maintain it
   — which is Phase 2.**

### Pre-registered retry / escalation, as applied

- **Retry at 2× P3** would not help: the failure is *behavioral* (the actor floors
  energy) and *structural* (energy rides `h`), not a training-duration shortfall —
  more steps deepen the floor, they do not create variance.
- **DP9 escalation**: taken and measured (item 3 above); it degrades A/C. Recorded,
  not adopted.

Per the §8.4 discipline and DP1b: this is a **recorded finding, not a tuning
target.** The environment was *not* recalibrated to force a green battery (that
would be the co-design loop the freeze exists to prevent).

---

## 3. Baseline instantiation (against the frozen formulas)

The pre-registration freezes baseline-relative quantities as *formulas*; this
section records the measured references. **Caveat — the references below are
contaminated by the energy-floor finding (§2.4):** measured under the
pure-epistemic actor, `true_energy` is pinned near 0, so the band/σ-grid
instantiations are **not yet usable** and must be re-measured after the env
energy economy is recalibrated so energy is a live in-band variable (a builder
decision — §4). Recording them here is not threshold editing; it documents *why*
the instantiation is blocked.

| Frozen formula | Instantiated reference (gate run, seed 0) | Usable? |
|---|---|---|
| **B0b** setpoint | 0.60 (normalized) | yes (config-fixed) |
| **B0a** in-band band = ±1× baseline std of `true_energy` | std = **0.064** → band 0.60 ± 0.064 | **no** — std reflects floored jitter, not an in-band distribution; in-band occupancy 0.4% |
| **S2** noise σ grid = {0, 0.5×, 1.0×} baseline std of `true_energy` | {0, 0.032, 0.064} | **no** — same contamination |
| **S1** precision grid = {0.1 … 10}× typical epistemic magnitude at band edge | epistemic disagreement mean **0.388**, p90 **0.486** | partial — epistemic magnitude measured, but "at band edge" is undefined while energy never reaches the band |
| positional entropy (pass/dominant entropy-retention reference) | **1.63 nats** (of max 4.16) | provisional |

**Build-time fixed energy economy (journaled, not swept; resolved decision #7):**
`energy_norm=[0,10]`, `init=6.0` (→ setpoint 0.6), `base_decay=0.08`,
`move_cost=0.04`, `replenish_per_resource=0.8`, `obs_noise_sigma=0.05`, `lag=1`,
`quantization_levels=16`. These gave good variance under uniform-random behavior
(std ≈ 0.19, no floor) but **floor under the trained epistemic actor** — the
calibration miss that surfaces finding §2.4.

---

## 4. Builder decisions surfaced (for review before Phase 2)

1. **Env energy-economy recalibration.** The current balance floors energy under
   epistemic-only behavior, so the pure-epistemic baseline is degenerate and the
   battery/band cannot be instantiated. Options: gentler dynamics; a partial
   carry/decay rebalance; or accepting that energy is only "live" under a
   preference (Phase 2) and assessing channel-learnability on a variance-rich
   probe distribution rather than the actor's own. **This is a build-time
   structural choice (journaled), not a frozen-pre-registration edit.**
2. **Do batteries B and D fit this substrate?** Energy is recoverable from the
   deterministic recurrent state, so the *observation* channel is redundant and
   the *stochastic* latent need not carry it. B (observation responsiveness) and D
   (per-dim KL escape) presuppose the observation is the primary route. If that
   presupposition is wrong for this env, B/D may be mis-targeted — a question for
   the builder, **resolved only via a new dated doc** (the pre-registration is
   frozen).
3. **Whether to force energy through `z` (DP9) despite the A/C cost.** Measured to
   degrade the clean channel; recorded for the builder.

---

## 5. Disposition

Phase 1 built the substrate and ran the frozen battery. The gate is **not cleanly
green**; it produced informative findings (the channel is learnable but rides `h`;
the epistemic-only regime floors energy; DP9 degrades it; the band cannot yet be
instantiated). Per DP1b and the co-design discipline these are recorded, not tuned
away. Phase 2 (fill the scaffold) is **gated on the builder's §4 decisions** —
chiefly the env recalibration, without which the preference's effect cannot be
read against a usable baseline.
