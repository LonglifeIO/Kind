# Probe 3.5 — Recalibration + amended-gate results — 2026-06-10

**Status: results record (Phase 1 — post-Amendment-01).** Authority for the
criteria: the frozen pre-registration `probe3_5_preregistration_2026-06-10.md`
(FROZEN) and the confirmed `probe3_5_preregistration_amendment01_2026-06-10.md`
(CONFIRMED 2026-06-10, builder: Gordon). Per the standing rule, instantiating
baseline-relative formulas from a measured run is not threshold editing; **no
frozen or confirmed threshold was changed in producing this record.**

This records the execution of the post-Phase-1 recalibration sequence: the
analytic energy-physics search (Amendment 01 §4), the retrain at the chosen
physics, and the amended Phase-1 gate (A, C, B′; D monitor). **The amended gate is
not green, and the baseline was not instantiated — a recorded finding, not a
tuning target.** The disposition (Amendment 01 / pre-registration §8.4 / DP1b):
report and journal; no further tuning in-session.

---

## 1. Recalibration search (Amendment 01 §4) — analytic, on replayed trajectories

Method (the passivity exploit, as pre-committed): the epistemic-only actor reads
no energy *quantity* directly, so — to first order — its trajectory distribution
was treated as invariant to energy physics. One epistemic instance was trained at
the **default** physics (P3 = 5000), and its per-step `(is_move, consumed_resource)`
flags were recorded over 8 env-seeds × 20 episodes. Energy was **re-simulated
analytically** for 140 candidate `(decay, move_cost, replenish)` triples.

- **Analytic re-simulation validated exactly.** Re-simulated `true_energy`
  reproduced the env's own series to **max abs err 0.0** at the default triple —
  the re-simulation is the env's energy law, not an approximation.
- **Default physics reproduces the Phase-1 floor.** Re-simulated under the default
  `(0.08, 0.04, 0.8)`: mean **0.006**, std **0.044**, floor-fraction **0.975** —
  confirming the §2.4 floor finding the recalibration was meant to address.
- **Incidental foraging is sparse.** The trained epistemic actor **moves every
  step** (move-rate 1.000) and enters a resource only **0.5%** of steps
  (consume-rate 0.00506) — depletion is relentless, replenishment rare.
- **Only the gentlest physics keep an indifferent agent off the floor.** **2 of
  140** triples met the pre-committed target R3–R5 (std ≥ 0.10, mean ∈ [0.30,0.70],
  floor-fraction ≤ 0.10), both at the **grid edge** (`decay = 0.01`,
  `move_cost = 0.005` — the smallest sampled): replenish 4.0 (std 0.284, mean
  0.609) and replenish 6.0 (std 0.269, mean 0.664).
- **Chosen triple (build-time physics choice, journaled):** `decay = 0.01`,
  `move_cost = 0.005`, `replenish = 4.0` — selected (pre-committed rule) for mean
  closest to the setpoint (0.609 ≈ 0.6). **Analytically it meets R3–R5.**

That only the gentlest-sampled physics clear the target — and only at the grid edge
— is itself the tiny-tensor double-bind (synthesis §7) surfacing in the env
economy: an indifferent forager's energy is live only if depletion is made nearly
negligible.

## 2. Retrain at chosen physics + the amended gate — NOT GREEN

A fresh instance was trained at the chosen physics (P3 = 5000, `pragmatic_value`
= 0). The amended gate (A, C, B′; D monitor — Amendment 01 §3) was run on the
**epistemic actor's** distribution (the non-degenerate one *post*-recalibration;
the Phase-1 precedent is to run decodability where the target varies). B′
intervention contexts were collected from the same rollout (agent one step from a
resource).

| Battery | Margin | Verdict | Value |
|---|---|---|---|
| **A** latent-predictability | R² ≥ 0.50 | **FAIL** | R² = −0.74 (and −27 at 4-seed diag) |
| **C** action-history ablation | ratio ≥ 1.5 | **FAIL** | 0.60 (history *beats* latents) |
| **B′** imagination intervention | ≥80% pairs, mean Δ ≥ 0.20 | **FAIL** | 0.41 of pairs; mean Δ = **−0.017** |
| **D** per-dim KL (monitor) | — (not gated) | monitor | max mean KL 1.13–1.20 (sub-1.5, as Phase 1) |

`gate_passed = false`.

## 3. The decisive diagnostic — the recalibration did **not transfer**

The eval target's variance on the gate rollout was far below the analytic
prediction (eval-half std ≈ 0.04 vs the predicted 0.284), so the gate target was
re-examined directly. **On the instance actually trained at the chosen physics, the
epistemic actor's energy is ceiling-saturated, not the targeted band:**

| Quantity | Analytic search (default-trained actor's trajectories) | **Trained-at-chosen-physics instance** |
|---|---|---|
| mean | 0.609 | **0.943** |
| std | 0.284 | **0.073** |
| floor-fraction (<0.05) | 0.059 | 0.000 |
| ceiling-fraction (>0.95) | ~0.09 | **0.626** |
| R3 std ≥ 0.10 | ✓ | **✗** |
| R4 mean ∈ [0.30,0.70] | ✓ | **✗** |
| R5 floor ≤ 0.10 | ✓ | ✓ |

**The chosen triple, validated analytically, fails R3 and R4 on the instance
trained at that physics.** Cause: the passivity / physics-invariance premise
(Amendment 01 §4 "Method"; the builder's Step-3 framing) is **only approximate**.
Energy is fused into the world model's `h, z` (DP4), and the actor reads `h, z` via
PolicyView — so an actor trained under **live** energy does not behave identically
to the **default-trained** actor (whose floored-energy trajectories drove the
analytic selection). The chosen-trained actor forages enough to **saturate** energy
at the ceiling; the analytic prediction (built on the energy-blind-in-effect
default actor) does not hold. (Training-seed variance may also contribute; the two
were not separated in-session, per "no further tuning".)

**Consequence for the gate.** Because the retrained instance's energy is
ceiling-saturated, the gate ran against a **degenerate (near-constant) target** —
so A (R² ≪ 0 against a pinned target), C (a saturated clock is trivially
history-predictable), and B′ (no decode headroom above the ceiling; mean Δ slightly
negative) are **uninformative about channel grounding**, exactly as the Phase-1
record flagged for its degenerate *floor* (§2 footnote †). The degeneracy moved
from the floor to the ceiling; it did not resolve.

## 4. Auditable distribution diagnostic (gate methodology)

Recorded so the gate's distribution choice is not silent: under the chosen physics
a **uniform-random** policy **saturates** energy (mean 0.990, 98.7% at ceiling) —
the inverse of the Phase-1 floor. Post-recalibration neither a vigorous forager
(uniform-random → ceiling) nor the trained indifferent actor (→ ceiling) yields a
non-degenerate target; the only non-degenerate energy in this whole sequence was
the **analytic** re-simulation over the *default*-trained actor's trajectories,
which did not transfer to a trained instance.

## 5. What this closes and what it leaves open

**The finding (the env-economy double-bind, confirmed at three levels):**
energy under this substrate is degenerate for an indifferent actor across the
physics envelope — *floored* (fast physics, Phase 1), *ceiling-saturated* (gentle
physics, trained instance and uniform-random), with the targeted live band
reachable only as an **analytic artifact** of the energy-blind default actor that
does not survive training at the physics. **Env-economy recalibration alone cannot
manufacture a world-grounded, in-band interoceptive channel for an indifferent
agent.** This sharpens, from a second direction, the Phase-1 conclusion: *energy
becomes a live, maintained, in-band variable only when there is a reason to
maintain it — a preference (Phase 2).* DP7 "sustainable" is necessary but not
sufficient: sustainability keeps the agent off the floor only by making depletion
negligible, which saturates energy rather than centering it.

**Disposition (Amendment 01 / §8.4 / DP1b):** **stop, report, journal — no baseline
instantiated, no further tuning in-session.** The band / σ-grid / S1-precision
formulas remain **un-instantiated** (no non-degenerate baseline exists to resolve
them against — the same blocker as Phase 1, now ceiling-side).

**Builder decisions surfaced for the next session (not acted on here):**
1. **The recalibration method needs the loop closed.** Selecting physics from
   passive (energy-blind) trajectories does not transfer, because the actor is not
   energy-blind once trained. Options: iterate (select → train → re-measure → re-select)
   until the *trained* instance meets R3–R5; or accept that no env-only physics
   makes an indifferent actor's energy in-band and move the band-instantiation into
   Phase 2 (energy live *under* the preference). Either is a build-time choice / a
   new dated doc — **out of scope for this session** (Step-4 "no further tuning").
2. **Whether to instantiate the baseline under the preference (Phase 2) instead.**
   If energy is in-band only when regulated, the pure-epistemic baseline the frozen
   §3 protocol assumes may not exist in a non-degenerate form; the assay
   baseline-relative formulas may need to resolve against a *regulated* reference.
   A pre-registration question (amendment), flagged not decided.

**Unchanged / intact:** PolicyView frozen at `{h, z, self_prediction_error}`;
`new_actor_readable_interfaces_added = []`; the frozen pre-registration byte-frozen;
Amendment 01 confirmed and honored (B′ ran as the imagination intervention; D was a
monitor; the retry was waived as recorded). The energy channel substrate (S-ENV /
S-WM / S-TEL) is unchanged; only the three build-time energy magnitudes were varied,
and they are **not adopted** (the chosen triple saturates a trained instance — it is
recorded, not committed to `GridWorldConfig`).
