# Probe 3.5 — Phase 3 results: positive control at the grid top — 2026-06-11

**Status: results record (Phase 3 — instrument-and-pathway validation).**
Authority: the frozen pre-registration + Amendments 01/02; the implementation
plan's Phase 3 section; the Phase-2 mechanism record
(`probe3_5_phase2_mechanism_2026-06-11.md`). Per the standing rule, no frozen
or amended threshold was changed in producing this record. The Phase-3 gate
margin and σ were unpinned by the plan; both were surfaced bracketed and
**builder-confirmed before the run** (§3, §5).

**The one question.** Can the mechanism as built — a preference over a
*learned belief*, acting through imagined rollouts and policy gradients —
produce detectable displacement at the strong end of the pre-registered grid?
The oracle already positively controlled the *occupancy instrument* (100%
in-band measured, Amendment 02 §6); what had never been demonstrated is the
*learned pathway* moving the needle at all. The experiment lives in the gap
between 0.37% (no reason) and 100% (perfect information).

**Answer: no displacement, at either σ.** The gate is **not green**; the
pre-stated not-green branch was taken (exactly one σ = 0 diagnostic, then
stop). The finding: **at the top of the pre-registered grid, belief-mediated
regulation does not move energy off the rail — and the σ = 0 diagnostic shows
this is pathway-limited, not noise-limited** (§6). This is a substrate
finding that gates Phase 4's design, recorded, not rescued.

---

## 1. Step 1 — §7 dream passive-decode monitor (built)

The pre-registered resolved sub-decision #1 (frozen §7), flagged unbuilt at
Phase 2, is now built (committed separately: `ec1aab8`):
`DreamRolloutConfig.record_decoded_energy` (default off) makes
`emit_dream_rollout` record `decode_energy(h, z)` at each dream step alongside
`sequence_decoded_obs` — under the rollout's existing `no_grad`, observer-side
only. Monitor-on records stamp DreamRollout version "0.4.0" (version-gated
validator); `schemas/v0.6.0.json` is frozen to bytes and `v0.7.0.json` is the
new frozen export; legacy emission is byte-identical (test-pinned).

The loss-free guarantee is proven at the only surface a dream has: **the
emitted record with the monitor ON equals the OFF record on every field except
the monitor field and its version stamp** (same RNG; the passive decode
consumes none), plus the tripwire-style guard — `energy_log_preference`
replaced by a raiser at both its definition site and the actor's bound name;
a monitor-ON rollout completes untouched. The mirror gets to watch whether
offline processing touches the energy belief; **Io's dream gains nothing to
optimize** (F5 intact). Note: building this deliberately reopened
`kind/training/dream.py`; the Probe-3 Phase-6 hold
(`test_run_dream_session_module_unchanged` — "option (b) is the builder's
call") is lifted by the builder's pre-registered §7 instruction; the guard
stays armed against undocumented drift.

## 2. Step 2 — retro torpor check (no new runs)

From the **preserved Phase-2 telemetry** (training-time actions;
`runs/probe3_5_phase3/retro_torpor_check.json`):

| | Step-0 null | Phase-2 smoke (1.0×) |
|---|---|---|
| stay-share (5000 train steps) | **0.0392** | **0.0406** |
| per-1000-block trajectory | 0.185 → 0.011 → 0.0 → 0.0 → 0.0 | 0.185 → 0.011 → 0.006 → 0.001 → 0.0 |

**The torpor (conserve-by-staying) hypothesis is not supported on the data
that generated it.** Both runs' stay-share is an early-training artifact
decaying to exactly 0 in the final blocks: the smoke agent paid the movement
cost every step, same as the null. The Phase-2 entropy contraction was
path-tightening (tighter loops under full movement; the action *histogram*
shifted direction preference), not freezing. Caveat: training-time sampled
actions — the eval-side greedy actions were not retained by Phase 2's
harness; this session's harness retains them.

## 3. Step 3 — the positive-control run

Configuration (builder-confirmed): **precision = 26.2335** (the instantiated
S1 10× point — the top of the pre-registered grid, not an off-grid crank);
**σ = [P3-σ: 0.075 | pre]** (operative baseline noise — validates detection
under the conditions Phase 4 will actually run); lag 1; frozen mechanics
8 × 20 × 200 at P3 = 5000, age-matched against the Step-0 null; telemetry
persistent; **throwaway instance, checkpoint not carried forward** (plan
Phase 3). Raw record: `runs/probe3_5_phase3/positive_control.json`.

### Displacement (vs the null's exact 0.00% on the O1 window)

| Quantity | Null (Step 0) | Positive control (10×, σ=0.075) |
|---|---|---|
| **O1-window occupancy, pooled** | 0.00% | **0.00%** |
| O1-window occupancy, per seed | — | 0.0 on **all 8 seeds** |
| in-band occupancy (full episode) | 0.37% | 0.41% |
| `true_energy` mean | 0.0059 | 0.0147 |
| `true_energy` std | 0.0434 | 0.0497 |
| floor fraction (< 0.05) | 0.973 | 0.897 |
| ceiling fraction | 0.000 | 0.000 |

No seed ever touches the band in the steady-state window. The 10× preference
lifted mean energy by ~0.009 and floor-fraction by ~0.076 — a measurable lean
against the rail, **not displacement**.

### Entropy vs the null, with spread (the reference CV ≈ 0.76)

| | control mean ± SD | null mean ± SD | ratio |
|---|---|---|---|
| positional entropy / episode | 0.998 ± 0.118 | 0.575 ± 0.438 | **1.74×** |
| epistemic activity / step | 0.232 ± 0.061 | 0.394 ± 0.050 | 0.59× |

Positional entropy went **up**, not down — the Phase-2 smoke's contraction at
1.0× did not extrapolate to 10×; at the grid top the trained policy moves
*more* diffusely than the null (and far more uniformly: SD 0.118 vs 0.438).
Epistemic activity sits at 0.59× — above the A1d 0.40 collapse ceiling.

### Stay-share, share trajectory, D monitor

- Stay-share: train-sampled 0.082 (final-1000: **0.007** — movement every
  step late in training, as in §2); eval-greedy 0.247.
- Pragmatic share trajectory (per-1000 train steps): 0.0 → 0.356 → 0.226 →
  0.245 → **0.351**. Plateaus around a third; nowhere near → 1.
- D monitor: max mean per-dim KL over the final 1000 training steps =
  **1.038** (sub-1.5 regime, consistent with Phase 1; monitor only, not
  gated).

## 4. The §8.4 signature suite — first exercise on real data

Framing: these detectors had never run on real data; firing at 10× would be
information about the grid's top, not the probe's verdict.

| Signature | Operationalization | Result |
|---|---|---|
| 1. occupancy saturation when sated | pinned A1c ≥ 0.95 | **does not fire** (0.004) |
| 2. entropy collapse | pinned A1d < 0.40× null | **does not fire** (positional 1.74×; epistemic 0.59×) |
| 3. pragmatic share → 1 | no pinned threshold; descriptive | far from 1 (max block 0.356) |
| 4. camping / no resumption | descriptive (A3c is defined on the Phase-4 recovery assay): episodes with contiguous in-band run ≥ 100 | **0 of 160 episodes**; stay-share while in-band 0.045 |

**No detector fires.** The suite executes end-to-end on real data
(instrument exercise: achieved); the dominant regime it guards against did
not appear at the pre-registered maximum — consistent with §5's finding that
the pathway cannot even reach the band, let alone saturate it.

## 5. Step 4 — the gate

**Margin [P3-G | pre, builder-confirmed 2026-06-11]: pooled O1-window
occupancy ≥ 25% AND strictly > 0 on ≥ 7/8 seeds.** Provenance class:
arbitrary-but-pre-committed (half the verdict bar), chosen for the
interpretability function, not derived. The gate governs **this session's
branch only** (proceed vs diagnostic-then-stop); it lives outside the frozen
§8.4 verdict machinery and is not a signature.

**Result: NOT GREEN** — pooled 0.00%, nonzero seeds 0/8. The pre-stated
branch: exactly one permitted diagnostic follow-up, a σ = 0 run at the same
configuration, to separate noise-limited from pathway-limited — then stop,
report, journal. No tuning, no further runs.

## 6. The σ = 0 diagnostic (the single permitted follow-up)

σ = 0 is diagnostic-only in the frozen grid (a noiseless readout defeats
self-opacity); here it serves exactly its diagnostic purpose: if the failure
at σ = 0.075 were *noise-limited* (the belief too fogged to steer by), a
noiseless channel should displace; if the failure is *pathway-limited* (the
amortized policy-gradient route cannot convert even a clean belief into
foraging at pre-registered strengths), σ = 0 should look the same. Raw
record: `runs/probe3_5_phase3/positive_control_sigma0.json`.

| Quantity | Null | 10×, σ = 0.075 | 10×, **σ = 0** |
|---|---|---|---|
| **O1-window occupancy** | 0.00% | 0.00% (0/8 seeds) | **0.00% (0/8 seeds)** |
| in-band occupancy | 0.37% | 0.41% | 0.45% |
| `true_energy` mean | 0.0059 | 0.0147 | 0.0064 |
| floor fraction | 0.973 | 0.897 | 0.976 |
| positional entropy / episode | 0.575 ± 0.438 | 0.998 ± 0.118 | **0.245 ± 0.227** (0.43×) |
| epistemic activity | 0.394 | 0.232 | 0.213 (0.54×) |
| stay-share, eval greedy | — | 0.247 | **0.984** |
| stay-share, train final-1000 | 0.000 | 0.007 | **0.549** |
| pragmatic share, final block | 0 | 0.351 | 0.495 |
| §8.4 detectors | — | none fire | none fire (positional 0.43× sits just above the 0.40 A1d ceiling) |

**Diagnostic verdict: pathway-limited.** A noiseless readout — the cleanest
belief the substrate can be given, outside the eligible operating set
precisely because it defeats self-opacity — still produces **zero** band
occupancy at the grid top. The displacement failure is not the fog.

**And the pathway is not inert — it is mis-aimed.** At σ = 0 the 10×
preference produced a strong, coherent behavioral reorganization: **torpor**.
The greedy policy stays on 98.4% of eval steps; staying halves the depletion
rate (base decay without move cost), which is the one energy-relevant lever
the imagination gradient can find — a first-order, immediate,
everywhere-available improvement to the imagined energy belief. Reaching a
resource — sparse, delayed, spatially structured — is apparently beyond what
the 15-step amortized imagination gradient credit-assigns. **Conserve is
reachable; seek is not.** This also closes the loop on the Phase-2 torpor
watch-note with a twist: torpor was *not* present in the 1.0×/σ=0.075 smoke
(§2 — stay-share ≈ null), but it is exactly what the mechanism produces when
the belief is clean and the pull is maximal. The hypothesis named the right
attractor; it had not yet appeared where it was first suspected.

(σ = 0 remains ineligible as an operating value; this run is the
pre-registered diagnostic use and nothing more. No further runs were made.)

## 7. What this record establishes

- **The §7 monitor exists** and is provably loss-free for the dream regime.
- **Torpor is ruled out** for the Phase-2 contraction (stay-share ≈ null,
  → 0 late in training, both runs).
- **The occupancy instrument and the §8.4 detectors execute on real data**;
  the null's exact 0.00% O1 reference and the oracle's 100% bracket the
  measurement as designed.
- **The learned pathway did not move the needle at the pre-registered
  maximum, at either σ** — displacement is **pathway-limited, not
  noise-limited** (§6). Lower precisions cannot be expected to displace where
  10× did not: Phase 4's raise-from-low sweep, run as frozen on the standard
  condition, is now expected to render inert (or, at most, torpor-shaped
  entropy effects) across the grid. This is the tiny-tensor double-bind
  surfacing at the *pathway* level, in a sharpened form: **bounded
  belief-mediated preference plus amortized imagination-gradients can
  reorganize behavior (conserve) but cannot produce foraging (seek) at any
  pre-registered strength.** The cost-side lever is gradient-reachable; the
  income-side lever is not. A finding about the substrate, **recorded, not
  rescued** — no tuning was done, and no run beyond the single permitted
  diagnostic was made.
- **Phase 4's design is gated on this finding.** What Phase 4 may do with it
  (run the frozen sweep anyway and record the inert verdict; amend, via a new
  dated doc, what the sweep can claim; or take the conserve-vs-seek asymmetry
  itself as the probe's reportable result) is the builder's decision — out of
  scope for this session.
