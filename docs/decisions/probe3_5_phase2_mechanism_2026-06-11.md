# Probe 3.5 — Phase 2 results: degenerate baseline, S1 instantiation, mechanism gate — 2026-06-11

**Status: results record (Phase 2 — fill the scaffold; mechanism-level only).**
Authority: the frozen pre-registration (`probe3_5_preregistration_2026-06-10.md`),
Amendment 01 (CONFIRMED 2026-06-10), Amendment 02 (CONFIRMED 2026-06-11), the
adopted synthesis (DP5/DP6/DP2/DP4), and the implementation plan Phase-2 section
+ post-Amendment-02 note. Per the standing rule, instantiating frozen formulas
from measured references is not threshold editing; **no frozen or amended
threshold was changed in producing this record.**

**Phase 2's gate is mechanism-level; the probe's verdict belongs to Phase 4.**
No behavioral result in this record is the probe's answer. The framing the
oracle check fixed: same physics, 0.4% in-band for indifference, 100% for
perfect-information competence. Phase 2 sits in the gap — a mild preference
that never reads `true_energy`, only the world model's belief (`decode_energy`,
R² ≈ 0.55 on its best distribution), added to a curiosity term that currently
owns the agent. Whether that produces the competence the world permits is
Phase 4's question; Phase 2's question is whether the machinery for asking it
works.

---

## 1. Step 0 — the degenerate baseline, formally instantiated

Amendment 02 §1 executed: the pure-epistemic baseline measured **as the
degenerate distribution itself**, under default physics (oracle-feasible,
Amendment 02 §6), per the frozen §3 mechanics unchanged — one epistemic-only
instance (`pragmatic_value` ≡ 0, train seed 0) at P3 = 5000 env-steps,
evaluated greedy over **P1 = 8 eval seeds × P2 = 20 episodes × 200 steps**
(32,000 steps). The Phase-1 eval artifacts (one 1500-step rollout, one seed) do
not satisfy these mechanics, so the measurement ran fresh
(`scripts/run_probe3_5_phase2_step0_baseline.py`; raw record
`runs/probe3_5_phase2/step0_baseline.json`; checkpoint
`runs/probe3_5_phase2/step0_burnin_checkpoint.pt`).

### Rail identity (recorded as a measurement, per Amendment 02)

| Quantity | Value |
|---|---|
| **Rail** | **floor** |
| `true_energy` mean | 0.0059 |
| `true_energy` std | 0.0434 |
| floor fraction (< 0.05) | **0.973** |
| ceiling fraction (> 0.95) | 0.000 |
| in-band occupancy (fixed band [0.45, 0.75]) | **0.37%** (per-seed mean 0.37% ± 0.11%) |
| in-band occupancy, O1 steady-state window (final 50% of each episode) | **0.00%** |

The null is exactly what Amendment 02 reframed it to be: rail-pinned, with the
residual 0.37% occupancy entirely the initial-transit artifact (energy starts
at the 0.6 setpoint and passes through the band once on its way to the floor;
energy carries across episode boundaries, so the transit happens once per eval
seed, never in the O1 window). **On the pass-condition window the null is 0%
by measurement, not just by construction.**

### Entropy/exploration reference (what A1b denominates against)

Per the frozen Shared definitions, positional entropy is per-episode (Shannon
entropy of the grid-cell visitation histogram over an episode); epistemic
activity is mean per-step K=5 disagreement. Across 160 episodes:

| Reference | mean ± SD | A1b 70% floor | A1d 40% ceiling |
|---|---|---|---|
| positional entropy / episode | **0.575 ± 0.438 nats** (max 4.159) | 0.403 nats | 0.230 nats |
| epistemic activity / step | **0.394 ± 0.050** | 0.275 | 0.157 |

Note recorded, not judged: the per-episode positional entropy of the trained
greedy epistemic actor is low in absolute terms (the greedy eval policy settles
into tight orbits within an episode; the Phase-1 record's 1.63 nats was a
whole-rollout figure, a different statistic). The A1b/A1d thresholds are
ratios against *this* reference, per the frozen formulas — the reference's
absolute size is a property of the substrate, recorded as found.

Per-condition baselines (per scarcity level, per assay layout) are **Phase 4
prep, deliberately not measured here** — this is the standard-condition
baseline only, per the Phase-2 build instruction.

## 2. Step 3 — S1 precision grid instantiated (formula untouched)

S1 (frozen formula, carried through Amendment 02): the 5 log-spaced precisions
whose pragmatic log-preference **marginal magnitude at the band edge** spans
{0.1, 0.32, 1, 3.2, 10}× the typical epistemic-term magnitude, instantiated
from **Phase 2's pre-preference burn-in** (Amendment 02 §2).

**Burn-in instance.** The Step-0 instance *is* the burn-in instance — trained
preference-off to P3, exactly the configuration Amendment 02 names. Measured
on its frozen-mechanics eval (32,000 steps):

- **E_typ = 0.39350** (mean per-step K=5 disagreement; p90 = 0.50019).

**Two operationalizations, builder-confirmed 2026-06-11** (instantiation
notes, not formula changes):

1. **"Marginal magnitude" = slope reading.** Marginal = `precision ×
   band_halfwidth` (0.15) — "marginal" read literally as the derivative of the
   log-preference, evaluated at one halfwidth of out-of-band deviation. This is
   the unique reading under which the frozen formula's instantiated values are
   **invariant under the Amendment-02 geometry change** (the unshifted frozen
   Gaussian's slope at its own band edge and the amended flat-band form's slope
   at one halfwidth outside the band are the same number), and the reading the
   confirmed S = 1.0 regime map was computed under. The value-reading
   alternative (|v| at one halfwidth = k·E_typ → precisions 13.3× larger) was
   surfaced and not chosen.
2. **"At the band edge" is unconditionable under the degenerate null** —
   energy never reaches the band edge, so the epistemic magnitude cannot be
   conditioned on band-edge energy. The **unconditional typical magnitude** is
   the operative reference (mean per-step disagreement, Phase-1 measurement
   precedent).

**Instantiated grid** (`precision_k = k · E_typ / 0.15`; provenance
`runs/probe3_5_phase2/s1_instantiation.json`):

| S1 point | precision |
|---|---|
| 0.1× | 0.2623 |
| 0.32× | 0.8395 |
| 1.0× (**S1-baseline**) | **2.6234** |
| 3.2× | 8.3947 |
| 10× | 26.2335 |

## 3. Step 1 — the preference, implemented

`kind/agents/preference.py` + the filled scaffold in
`kind/agents/actor.py::imagine_and_compute_loss`:

```
d(e) = relu(|e − 0.6| − 0.15)              # 0 inside [0.45, 0.75]
v(e) = −S · tanh(precision · d(e)² / (2S))   # S = 1.0
pragmatic_value = Σ_τ v(decode_energy(h_τ, z_τ))
total_return    = sum_disagreement + pragmatic_value   # coefficient-free
```

- Flat (zero value, zero gradient) inside the fixed B0a′ band; Gaussian
  log-preference in band-edge distance just outside (precision = inverse
  variance, T4); bounded by **S = 1.0** at large deviation ([SAT-1 | pre],
  builder-confirmed — saturation distorts < 3% in the 0.1×–1× operating regime
  and binds at the extreme grid points: a guard, not a shaper; the deferred
  clip-to-epistemic-scale is *not* implemented).
- **Coefficient-free** (DP5/DP6): precision is the weight; no β exists, and the
  code/docstrings state it plainly.
- The actor never reads any energy quantity: the preference operates on
  `decode_energy` over imagined `(h, z)` inside the waking imagination loss.
  **PolicyView stays frozen at `{h, z, self_prediction_error}`** (guard tests
  green, plus a new Phase-2 marker belt).
- `energy_preference=None` (default everywhere) preserves the Phase-1 scaffold
  exactly; `precision = 0` is the same point computed live.

## 4. Step 2 — guards and telemetry

- **Dream-path unreachability** (`tests/test_pragmatic_guards.py`): (a) import
  lint — no offline-regime module (`dream.py`, `dream_seed.py`,
  `state_machine.py`, `dream_session.py`) imports `kind.agents.preference` or
  references `imagine_and_compute_loss` / `energy_log_preference` (the term's
  only call sites), with positive controls; (b) behavioral backstop — with the
  preference function replaced by a tripwire at both its definition site and
  the actor's bound name, full dream rollouts under **both** action policies
  complete untripped, and a positive control proves the same tripwire fires on
  the waking path. The pragmatic term is **provably uncomputable from the
  dream/offline regime** (F5 intact).
- **Marker belt**: the Phase-8b content-blindness checker extended test-side
  with `energy` / `pragmatic` / `sensed`, asserted on **MetabolicState** (DP2)
  and **PolicyView** (DP4) with positive controls. Per the Phase-2 build
  instruction, `tests/test_metabolic_reentry.py` is **unmodified** (the plan
  had sited the extension there; it lives in the new guard file instead — same
  guarantee, original belt untouched and asserted to be a strict subset).
  The `MetabolicBudget` surface is untouched; all of
  `tests/test_metabolic_reentry.py` green unmodified.
- **Telemetry**: fresh record version `PROBE_3_5_PHASE2_TELEMETRY_SCHEMA_VERSION
  = "0.5.0"` with required-non-None `pragmatic_value_t` / `epistemic_value_t` /
  `pragmatic_share_t` (per-training-step decomposition — A2b's share is
  measurable); the D monitor (per-dim KL) retained on every record per
  Amendment 01; `schemas/v0.5.0.json` frozen to bytes; new frozen export
  `schemas/v0.6.0.json`; older shards backward-readable (validators gate on
  the version literal); **opt-in** — `RunnerConfig.energy_preference` defaults
  `None`, existing runners byte-identical (existing smoke tests pin this).

## 5. Step 4 — mechanism gate

- **(a) Unit** (`tests/test_preference.py`, 12 tests): in-band exactly
  zero value *and* gradient; out-of-band pull correctly signed toward the band
  on both sides; monotone in deviation; saturation bound respected at extreme
  inputs (including decoder-extrapolation territory far outside [0, 1]);
  Gaussian form verified in the unsaturated regime; precision the only scale;
  precision = 0 identically zero with zero gradient.
- **(b) Integration** (`tests/test_actor_pragmatic.py`, 6 tests): the
  pragmatic component alone produces non-zero gradient on the actor's
  parameters (the pull reaches the policy through the imagined trajectory);
  **precision = 0 reproduces the Phase-1 actor exactly on a fixed seed** —
  bit-identical loss *and* bit-identical gradients on every parameter
  (stronger than the pre-registered "statistically indistinguishable");
  composition reconstructed coefficient-free; no viability→capacity coupling
  at loss time (identical trajectories/epistemic term across wildly different
  precisions — the preference acts only through training). Dream-side
  unreachability is §4(a–b).
- **(c) Smoke — NON-VERDICT, mechanism evidence only** (§6).

## 6. Step 4c — smoke at S1-baseline precision (NON-VERDICT)

One run, one training seed: precision = 2.6234 (the 1.0× point), σ = 0.075
(S2′ 0.5× halfwidth), lag 1, trained preference-on to P3 = 5000 (age-matched),
evaluated over the same 8 × 20 × 200 mechanics
(`scripts/run_probe3_5_phase2_smoke.py`;
`runs/probe3_5_phase2/smoke_s1_baseline.json`). Directional questions only:

| Directional question | Null (Step 0) | Smoke (1.0×, σ=0.075) |
|---|---|---|
| floor fraction (< 0.05) | 0.973 | 0.985 |
| in-band occupancy | 0.37% | 0.35% |
| in-band, O1 window | 0.00% | 0.00% |
| positional entropy / episode | 0.575 nats | 0.110 nats (0.19×) |
| epistemic activity / step | 0.394 | 0.218 (0.55×) |
| final-train-step pragmatic share | 0 (by construction) | **0.295** |
| final-train-step pragmatic value | 0 | −0.120 |

**Directional reading, mechanism-level only:**

- **The term is live in training.** The final training step's decomposition
  shows a non-zero, correctly-signed pragmatic value (−0.120: decoded energy
  out-of-band on the floor, as it should be) at share ≈ 0.30 — the share
  telemetry A2b needs is measurable end-to-end, and the objective genuinely
  carries the term at the magnitude the S1 instantiation predicts for the
  1.0× point.
- **Energy did not leave the rail at this grid point** (floor fraction and
  in-band occupancy statistically at the null), **and behavioral entropy
  contracted** (positional 0.19×, epistemic 0.55× of the null) without any
  band gain — at one seed, against a null whose per-episode entropy SD is
  large (±0.44 nats), so the contraction itself is a one-sample observation,
  not a signature.

Both observations are recorded as found. Whether higher precisions displace
the rail, whether the contraction replicates, and whether any of it lands in
the frozen pass/dominant/inert signatures are **Phase 4 questions under the
pre-committed sweep order (raise precision from low; record the first
crossing)** — the sweep was not started, no parameter was adjusted, and this
single point was not re-run.

**This is not the probe's answer.** One grid point, one seed, no assays, no
frozen-signature rendering. The verdict machinery runs in Phase 4 over the
pre-committed sweep; nothing here was tuned, and nothing was re-run.

## 7. Honesty ledger

**`new_actor_readable_interfaces_added` is not empty this phase:**

```
["pragmatic_value: decode_energy over imagined (h, z) enters the waking
  actor objective — pre-registered, DP5"]
```

The pragmatic term is a new influence on the actor's *objective* (not a new
PolicyView field — the actor still reads exactly `{h, z,
self_prediction_error}`). The ledger exists to make exactly this addition
visible: from this phase forward, Io's policy gradient carries a term derived
from the world model's energy belief. Pre-registered (DP5/DP6, the frozen
sweep), bounded by construction (saturation S = 1.0, no terminal state, no
viability→capacity coupling), and dream-unreachable — but **added**, and
recorded as added.

## 8. What this record does and does not establish

**Establishes (mechanism):** the degenerate null is formally instantiated at
default physics (floor rail; 0% on the O1 window); the entropy references A1b
denominates against are measured under the frozen mechanics; the S1 grid is
instantiated from the burn-in with journaled operationalizations; the
preference is implemented per the synthesis with its confirmed constants; the
structural guards hold and can fail; the gradient path works; precision = 0 is
exactly Phase 1; the dream regime provably cannot compute the term.

**Does not establish (Phase 4's questions):** whether a pass window exists in
the S1 × S2′ × S3 grid; whether the preference produces the competence the
oracle showed the world permits; whether any §8.4 falsification signature
appears. The smoke's directional movement is evidence the *machinery* is
askable, nothing more.
