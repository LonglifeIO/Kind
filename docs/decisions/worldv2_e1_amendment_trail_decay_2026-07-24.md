# World v2 — E1 amendment: trail decay 50 → 12 — 2026-07-24

**Status: RATIFIED 2026-07-24 (builder: Gordon — "amend + e3 first";
session 9 runs amended e3 alone before e2 lands, one change at a
time).**

## The finding this answers

Sessions 7–8 (journal `docs/workingjournal/worldv2.md`; fork decision
doc `worldv2_e3_fork_trail_off_diagnostic_2026-07-23.md`): the trail-off
diagnostic decomposed the treadmill. R1: the trail's standing stock
destroyed ~41% of the world's food production (meals 3.6× on removal).
R3: session 7's 10× curiosity rise collapsed without the trail
(5.66 → 0.42) — Io's drive was substantially anchored on its own
footprints. The builder's read (2026-07-24): self-caused structure
should stay *possible* as an attention object, but Io's own footsteps
must not be the **dominant** curiosity source — especially not via a
loop that starves it.

## The mechanism

Io trains on 32-step windows (BPTT horizon). At `trail_decay_steps=50`
a footprint's lifecycle cannot fit inside anything Io can learn: the
ensemble can never master when a footprint fades, so disagreement about
the trail never decays — an **inexhaustible novelty fountain glued to
Io's own feet**, dense (refreshed every step, always adjacent) and
permanently unlearnable. A supernormal stimulus, not an affordance.
The house already encodes the counter-principle: the e2 bloom period
was set to 12 so its pattern is *carryable* — learnable with effort,
inside the horizon.

## The amendment

`TRAIL_DECAY_STEPS` (`kind/env/world_stages.py`): **50 → 12.**

This deliberately exits the e1 synthesis's ~40–60 band. The band
predates the sessions-7–8 evidence; any value above the 32-step horizon
preserves the unlearnable-fountain defect, so staying inside the band
(e.g. 40) would amend nothing. New evidence, dated amendment, builder
ratification — the standard path. Nothing else changes: stamping rule,
refresh-on-revacate, regrowth blocking, and the trail's observation
vocabulary are all as ratified. `GridWorldConfig.trail_decay_steps`'s
default (50) is untouched — the stage preset is the biography's world.

## Expected effects (both directions of the builder's ask)

1. **Dominance breaks by familiarity, not removal.** A 12-step
   lifecycle fits the learnable horizon: the ensemble can finish
   understanding the trail, disagreement decays with mastery, and
   self-attention returns to charter stance — possible, not mandatory,
   not impossible. The capacity (visible self-caused structure) stays.
2. **The starve-loop shrinks ~4×.** Sterilization scales with standing
   trail stock (12/50 ≈ 0.24 → ~180 expected prevented regrowths per
   30k vs session 7's 738), before behavioral feedback.

## Pre-registered readouts for session 9 (amended e3, 30k steps)

- **R1 economy**: meals should land materially above session 7's 416
  (sterilization relief), plausibly below session 8's 1,503 (the trail
  still watches back). Floored/in-band should improve on session 7.
- **R2 attention**: the curiosity trajectory is the signature. S7 was
  monotone rise (fountain), S8 flat-from-start (removal). Amended e3
  predicts a *transition*: engagement that settles as the trail is
  mastered — not a monotone climb. Honest caveat: the policy carries
  ~250k steps of trail-watching habit; one session may show relearning
  dynamics, not equilibrium.
- **R3 space**: the corner pacing loop is expected to persist (R2 of
  the diagnostic found it is the policy's own); any loosening is
  signal, not expectation.
- **Null result** (fountain dynamics return at 12): the dominance is
  not TTL-driven — the next fork is a fresh builder decision, not a
  decay-search loop.

## Sequencing (ratified with this amendment)

Session 9 = amended e3 **alone** (attribution discipline). If it reads
well, session 10 lands e2 (the hidden clock) — the ladder's own
non-self curiosity source — into a world where the trail no longer
drowns the signal.
