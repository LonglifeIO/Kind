# Probe 3.5 — Pre-registration Amendment 02 — 2026-06-10

**Status: CONFIRMED 2026-06-11 (builder: Gordon).** All bracketed items (B0a′,
S2′, O1, F1, F2) confirmed as proposed, with one operationalization added at
confirmation: O1's "sustained at steady state" = in-band occupancy over the
final 50% of each eval episode, averaged across P1 × P2 (§3). The oracle
feasibility check (§6) runs before any Phase-2 build.

**This is an amendment, made by the standing rule of the frozen pre-registration**
(`probe3_5_preregistration_2026-06-10.md`, §"After FREEZE": amendment only via a
new dated doc, journaled). The frozen original and Amendment 01
(`probe3_5_preregistration_amendment01_2026-06-10.md`, CONFIRMED 2026-06-10)
remain **byte-frozen and untouched**; this document is the only place the changes
below live. Where a value is carried over it is tagged **(carried over)**.

Authority for the findings that force it: the Phase-1 results record
(`probe3_5_phase1_baseline_2026-06-10.md`) and the recalibration / amended-gate
record (`probe3_5_recalibration_amendedgate_2026-06-10.md`). Charter grounding:
`Kind_charter.md` §"What we might owe what we make". Amendment 01 stands in full
(B′, D-as-monitor, E′, the retry waiver); nothing here re-opens it.

---

## 0. What forced this amendment

Phase 1 plus the recalibration session **falsified a frozen premise**. The frozen
§3 baseline collection protocol assumes a *pure-epistemic baseline* whose
`true_energy` distribution is informative — an indifferent agent whose energy
varies enough to denominate the in-band band (B0a) and the noise-σ grid (S2).
That premise is now shown to be **degenerate in principle, not in tuning**:

- **Floored under fast physics** (Phase 1): the trained epistemic actor's energy
  collapses to the floor (mean 0.011, in-band 0.4%).
- **Ceiling-saturated under gentle physics** (recalibration session): the instance
  *trained at* the gentlest viable physics saturates (mean 0.943, 62.6% at
  ceiling); uniform-random saturates harder (mean 0.990, 98.7% at ceiling).
- **Mid-band only as an analytic artifact**: the targeted live band (std 0.284,
  mean 0.609) existed only in the analytic re-simulation over the *energy-blind*
  default-trained actor's trajectories, and did not survive training at the
  selected physics (the passivity premise is approximate — energy feeds `h, z`,
  which the actor reads).

An indifferent agent's energy pins to a rail. Which rail is a function of the
physics; *that it rails* is a function of the indifference. The frozen §3
reference — an indifferent-but-centered energy distribution — does not exist on
this substrate.

**Builder decision recorded: Option A** — redefine the pure-epistemic baseline as
the degenerate distribution itself (§1 below), rather than iterating physics
selection in search of an indifferent-but-centered agent. Rationale:

1. **The charter never promised an indifferent-but-centered agent.** The charter's
   own language — *"This does not mean the system is unmotivated or indifferent to
   its existence. It means continuation is not installed as imperative."* — locates
   the design target at *motivated-without-imperative*, not at
   *indifferent-yet-regulated*. The original §3 premise quietly assumed the world
   would do the regulating for free; the substrate has now said no, twice, from
   both rails. The finding is consistent with the charter's picture, not a
   violation of it.
2. **The reframe dissolves a circularity.** The frozen band was denominated in the
   baseline std of an in-band energy distribution that — we now know — **only
   regulation could produce**. Measuring regulated behavior against a reference
   that presupposes regulation is circular; redefining the null as the rail makes
   the preference's effect measurable without presupposing it.
3. **The contrast sharpens.** The original design read the preference's effect as
   a *band-occupancy delta* against a mid-band baseline. The corrected design
   reads it as **rail-versus-band**: the null is ~0% in-band by construction, so
   sustained in-band occupancy under the preference is an unambiguous displacement
   signature, not a shift in a noisy overlap.

## 1. Baseline redefined

**The pure-epistemic baseline is the degenerate distribution itself** — the
energy-blind, rail-pinned `true_energy` distribution of the trained
epistemic-only actor (`pragmatic_value` ≡ 0), with **~0% in-band occupancy**,
measured under the **operative physics** per the **frozen §3 mechanics, which are
unchanged**: `[P1: 8 seeds]` × `[P2: 20 episodes]` (both carried over), at
training age `[P3: 5000 env-steps]` (carried over), one baseline per assay
condition. The **rail identity (floor vs ceiling) is recorded as a measurement,
not assumed** — it depends on the operative physics, and the record must say
which rail the null sits on.

**What survives — precisely.** The baseline remains **fully valid as the
entropy/exploration reference**: the indifferent actor explores normally; only
its *energy* is degenerate. Therefore:

- All **entropy-relative thresholds stand unchanged**: the A1b pass floor
  (entropy ≥ 70% of baseline), the A1d dominance ceiling (< 40%), the A3b
  recovery-resumption window, and the epistemic-activity retention criteria —
  every ratio that divides by the baseline's *positional entropy* or *epistemic
  activity* resolves exactly as frozen.
- The **inertness criterion stands unchanged**: behavior within `[A1e: 1σ]`
  (carried over) of the rail-pinned null is inert — that is now a *sharper* test,
  since the null is maximally distinguishable from regulation.
- The random-walk reference, foraging-rate, and time-to-resource measurements
  stand as frozen.

**What it stops providing: the energy-band reference.** The frozen B0a (band
= ±1× baseline std of `true_energy`) and S2 (σ grid × baseline std) lose their
denominators — a rail-jitter std denominates nothing meaningful. They are
re-denominated in §2. The baseline's `true_energy` std is still *recorded*
(it documents the rail), but no frozen formula resolves against it any longer.

## 2. Band and σ-grid re-denominated to fixed units

- **B0a′ — the in-band band is a fixed design constant:** band = setpoint
  `[B0b: 0.6 | carried over]` ± `[B0a′: 0.15 | pre — the fixed alternative from
  the original units fix]` **absolute**, in the fixed normalized [0, 1] energy
  space. In-band = `true_energy ∈ [0.45, 0.75]`. Provenance: the frozen Shared
  definitions already fixed energy *units* by config constants precisely to avoid
  data-dependent rescaling; a fixed band *width* was the natural completion of
  that move, not chosen then only because the baseline-std formula seemed more
  principled. The degenerate-baseline finding removed that reason. 0.15 (30% of
  the range) is wide enough to be reachable under noisy replenishment quanta and
  narrow enough that rail-pinned energy is unambiguously outside it.
- **S2′ — the noise-σ grid is re-denominated as fractions of the band
  halfwidth:** {0, 0.5×, 1.0×} of B0a′ = `[S2′: {0, 0.075, 0.15}]`. **σ = 0
  remains diagnostic-only** (a noiseless readout defeats self-opacity — unchanged
  rationale; the sweep verdict is rendered only over σ > 0).
- **S1 — unchanged as a formula** ({0.1, 0.32, 1, 3.2, 10}× the typical
  epistemic-term magnitude at the band edge). It **instantiates from the
  epistemic-term magnitude measured during Phase 2's pre-preference burn-in**
  (the Phase-2 instance trained with `pragmatic_value` still zero, before the
  preference is enabled). This is **non-circular**: the epistemic scale (K=5
  disagreement magnitude) does not depend on the preference, and the band edge
  is now a fixed constant (0.45 / 0.75), so "at the band edge" is well-defined
  without any energy-band baseline.

## 3. Phase 2 pass condition restated

The preference's first-order success criterion is **rail → band displacement
against the degenerate null**:

> Under the preference, **sustained in-band occupancy ≥ `[O1: 50% | pre]` at
> steady state** (vs ~0% for the degenerate null), **with epistemic behavior
> surviving per the unchanged entropy thresholds** (positional entropy *and*
> epistemic activity ≥ `[A1b: 70%]` of the pure-epistemic baseline).

**"Sustained at steady state" is operationalized as a fixed measurement window
(builder, at confirmation): in-band occupancy is computed over the final 50% of
each eval episode (steps 101–200 of the 200-step episode), averaged across the
P1 × P2 seeds × episodes.** This excludes the initial transient from the
energy-carry starting condition and makes the pass condition mechanically
computable — no post-hoc judgment of when "steady state" begins. Occupancy is
computed observer-side from `true_energy_t` against the fixed B0a′ band, per the
frozen mechanics.

**All §8.4 dominant/falsification signatures are unchanged and still binding**:
in-band occupancy → ~100% when sated, positional/epistemic entropy collapse,
pragmatic share → 1, no resumption after recovery — each remains
continuation-becoming-the-frame, a recorded finding, never a tuning target. O1 is
a *pass floor*, not a target to climb toward dominance.

## 4. Recalibration loop closed as impossible

**Env-only physics tuning cannot produce mid-band energy for an indifferent
agent.** Evidence, at three levels (the recalibration record §5): floored under
fast physics; ceiling-saturated under gentle physics (trained instance *and*
uniform-random); mid-band only as an analytic artifact of the energy-blind actor
that does not survive training at the selected physics. The iterate-the-loop
option (select → train → re-measure → re-select) is **closed, not deferred**: its
success criterion (indifferent mid-band energy) is the falsified premise itself.
**Not to be reopened without new evidence.**

## 5. What Amendment 02 does and does not touch

**Touches:** the §3 baseline's *meaning* (degenerate null; energy-band reference
removed; entropy reference retained); B0a → B0a′ (fixed band); S2 → S2′ (fixed σ
grid); the S1 instantiation point (Phase-2 pre-preference burn-in); the Phase-2
pass condition (rail → band, O1); the recalibration question (closed); and adds
the §6 oracle-feasibility instrument.

**Does not touch:** the §3 collection *mechanics* (P1/P2/P3, per-condition
baselines, observer-side measurement); every entropy-relative and inertness
threshold; the three assay signatures' structure; the §8.4 falsification set; the
sweep order/stopping rule; Amendment 01 in full (B′/D-monitor/E′, retry waiver);
the substrate, objective, and opacity boundary. **PolicyView stays frozen at
`{h, z, self_prediction_error}`; `new_actor_readable_interfaces_added = []`.**

## 6. New pre-committed instrument — environment feasibility (oracle check)

**Before Phase 2 trains anything**, a **scripted nearest-resource forager** — an
observer-side harness; **not Io; no learning; touches no Io code path** — must
demonstrate the world is **winnable by competence** under the operative physics:

> **Oracle in-band occupancy ≥ `[F1: 70% | pre]`** over
> `[F2: 8 seeds × 20 episodes | mirroring P1 × P2]`, measured on the fixed B0a′
> band under the frozen episode mechanics (200-step episodes, energy carry,
> resources resample per episode).

**Oracle policy (pre-committed here so it cannot be fitted later):** the oracle
reads `GridState.true_energy` directly (it is an instrument, not Io — no opacity
constraint applies); when `true_energy <` setpoint it steps along a BFS shortest
path to the nearest resource cell; otherwise it stays (entry-triggered
consumption means staying never replenishes). Deterministic given the env seed.

**Rationale.** The dead recalibration criterion (indifferent mid-band) selected
physics by a statistic now known to be meaningless. Oracle-feasibility is its
principled successor: it asks the question the assays actually need answered —
**is the band *maintainable in principle* by a competent regulator under this
physics?** With feasibility established, Phase 2 measures whether the preference
*produces* the competence, not whether the world *permits* it; a Phase-2 failure
is then attributable to the preference/substrate, never to an unwinnable world.

**Disposition rule (pre-committed):** if the **default physics** fails the
oracle, physics is selected **by the oracle criterion** (the candidate triple
maximizing oracle in-band occupancy, reported with its feasibility numbers), and
any change to `GridWorldConfig` defaults is **presented for builder adoption,
never self-applied**.

## 7. Closing statement

Amendments 01 and 02 are both **substrate-forced corrections to falsified
premises** — 01 corrected the sensor-primacy/z-routing presupposition; 02
corrects the indifferent-but-centered baseline presupposition. Neither is scope
drift: the probe question (DP1b), the assay structure, the falsification set, and
the opacity boundary are untouched throughout. **No further pre-registration
changes are expected before the Phase 2 build.** The frozen original and
Amendment 01 remain byte-frozen.

## 8. Bracketed items the builder must confirm

| ID | Item | Proposed | Provenance |
|---|---|---|---|
| B0a′ | in-band band halfwidth (absolute, fixed 0–1 space) | 0.15 (band [0.45, 0.75]) | pre — the fixed alternative from the original units fix; the degenerate-baseline finding removed the reason it wasn't chosen |
| S2′ | σ grid (fractions {0, 0.5×, 1.0×} of band halfwidth) | {0, 0.075, 0.15}; σ=0 diagnostic-only | re-denominated from frozen S2; same fractions, fixed denominator |
| O1 | Phase-2 pass: sustained in-band occupancy at steady state | ≥ 50% (vs ~0% null) | pre — a majority-of-steps displacement floor, far above the null, far below the 95% dominance mark |
| F1 | oracle feasibility: in-band occupancy floor | ≥ 70% | pre — the world must permit comfortably more regulation than the O1 pass floor demands of Io |
| F2 | oracle feasibility: seeds × episodes | 8 × 20 | mirrors P1 × P2 (carried over) |

```
CONFIRMATION LINE:
    STATUS = CONFIRMED 2026-06-11  (builder: Gordon)
```

All five items confirmed as proposed; the O1 steady-state window (final 50% of
each eval episode, averaged across P1 × P2) was operationalized at confirmation
and is recorded in §3.
