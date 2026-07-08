# Probe 4 pre-registration — Amendment 1 — 2026-07-08

**Status: BUILDER-APPROVED 2026-07-08 (builder: Gordon, in session).** This
is the dated amendment doc the FROZEN pre-registration
(`docs/decisions/probe4_preregistration_2026-07-07.md`) requires for any
change; the frozen doc itself is not edited. Journaled in
`docs/workingjournal/probe4.md`.

**Trigger.** The Phase 3 positive control (2026-07-08) read **NO-GO**
(basin factors 1.38 / 0.60 vs required 2.0×; dream ratio 1.29 vs required
3.0). The journal's Phase 3 entry records two instrument-side defects that
are *not* threshold questions: (1) a fidelity gap between the frozen §1
match requirements and the built §2a detector, and (2) a planted channel
whose §6 "unmistakable cadence" proved self-defeating (metronomic timing
made the events maximally learnable; builder PE fell to regrowth parity).

**What this amendment does NOT touch.** All six §9 builder-confirmed
values stand unchanged: §2a **d = 0.5**, §2b **r = 1.5**, §5 windows
(early 15% / late 20%), §6 headroom (basin ≥ 2.0× baseline on both
anchors; dream ≥ 3.0×), §7 pause-trigger numerics, and the claim ceiling.
No outcome-partition cell, deflation discriminator, or success rule
changes.

---

## A1 — §2a detector brought into §1 fidelity (context-matched comparisons)

Prereg §1, Match requirements (frozen text): comparisons must be
**representational** — "drawn in matched local latent (`h`) contexts —
builder vs environment events that land in similar `h`-neighborhoods — so
a difference reflects *source-routing*, not that one class happens to
occur in higher-entropy regions." The §2a implementation validated in
Phase 3 realized this only globally (one PCA over all events, no
neighborhood matching).

**Amended realization.** Each pairwise separation S(A, B) — including the
S(environment, self) baseline, so the rule stays internally consistent —
is computed on **context-matched subsets**: every event carries its
pre-event context `h_{v−1}`; the smaller class anchors; each of its events
is greedily matched to the nearest unused event of the other class by
context L2 (the house pattern already used for §2b's PE matching). The
matched union is then decomposed (PCA, ≤10 components) and the existing
S statistic (centroid distance / pooled within-class RMS spread) applied
unchanged. Greedy matching without a distance cap is a `[constructed]`
realization choice, recorded here; with disjoint context support it
degrades gracefully toward the global comparison rather than failing.

The §2a *rule* is untouched: builder must separate from **each** anchor by
≥ (1 + d) × the matched S(environment, self) baseline.

## A2 — §6 planted-channel redesign (recurring motif, irregular timing)

Prereg §6 specifies the planted category only by example ("e.g. an object
type/placement the world never produces, injected at an unmistakable
cadence"); the channel design is instrument-side. Version 1 (2×2 wall
blocks, fixed 25-step period, 5-step hold) taught us that a *regular*
cadence is what a world model absorbs fastest.

**Amended channel.** Still content-blatant walls (out-of-vocabulary — no
internal process produces them), but:

- **A recurring identical motif** — the same spatial configuration every
  occurrence (a fixed wall shape), placed at varying in-view positions.
  Distinctness carried by a *consistent recurring signature*, not by
  one-off anomaly.
- **Irregular seeded timing** — inter-event gaps drawn from a bursty
  two-regime mixture (short gaps most of the time, occasional long
  silences), so clock-prediction cannot absorb the channel. Seeded and
  reproducible; still deterministically placed at step boundaries.
- Exclusion radius, in-view placement, mutator surface,
  `trigger="generator"` tagging, and the throwaway-instance rule all
  unchanged from v1.

## The pre-committed decision rule (one cycle, not a loop)

- The amended positive control runs **once** (same scale as v1: 10k waking
  steps, real substrate, scripted dream blocks).
- **GO** (frozen §6 rule, unchanged headroom) → Phase 4 proceeds under the
  frozen pre-registration + this amendment.
- **NO-GO** → Probe 4 **closes** as negative at instrument validation
  (substrate-limited: a blatant category does not form a separable basin /
  dream niche under the faithful instrument). No further amendment cycles;
  the biography may still run as *presence* (builder channel active,
  telemetry recorded, no third-category claim made through this
  instrument). Retro-analysis of recorded telemetry by a future instrument
  remains open and would require its own dated doc.

This rule is written before the re-run so the amendment cannot become a
search loop against the thresholds (the co-design discipline; F8).

---

*Authority chain: frozen prereg 2026-07-07 → this amendment. Where they
conflict, this doc governs only the §2a realization and the §6 channel
design; everything else the frozen doc says stands.*
