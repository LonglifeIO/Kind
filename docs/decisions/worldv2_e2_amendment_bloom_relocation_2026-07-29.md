# World v2 — E2 amendment: BLOOM_CELL (6,6) → (1,6) — 2026-07-29

**Status: RATIFIED 2026-07-29 (builder: Gordon — "go" on the
relocation-as-mechanism-test recommendation; session 11 runs e2 with
the relocated clock, no other change).**

## The finding this answers

The session-10 mechanism pass (journal
`docs/workingjournal/worldv2.md`, entry 2026-07-28): the "sampled,
then displaced" reading resolved to *mastered, then moved to the
largest remaining novelty*. Near-ring disagreement decayed 0.96 → 0.60
across ~4k steps of sampling; bloom-visible steps carried no more
disagreement than quiet ones (the stamps use the trail vocabulary,
already mastered in session 9); the far corner paid 4.65 on arrival as
an unvisited novelty reservoir. The flight was sign-consistent with
the drive. Separately, the original placement (6,6) — chosen
2026-07-09 for the "open quadrant" — had been overtaken by home-range
drift before e2 ever ran (session 9 spent 45% of its steps within 1 of
it), and the session-10 relocation to the northwest corner has since
put the *old* site at maximum distance by accident.

## The amendment

`BLOOM_CELL` (`kind/env/world_stages.py`): **(6, 6) → (1, 6).**

Geometry, checked in code: the Moore ring of (1,6) is fully in bounds
and wall-free; Chebyshev 5 from the old site, 6 from the current home
range ((0,0)/(1,0)/(1,1)), away from the E0 corridor. Nothing else
changes: period 12, duration 2, trail-vocabulary stamps, one clock
only. Same amendment shape as the trail-decay amendment (new evidence,
dated doc, builder ratification).

## What this is a test of

Not a placement fix — the drift already un-mooted the old site. The
relocation forces the question the mechanism pass opened: **did Io
learn the clock as a place, or as a process?** Same vocabulary, same
period, new location.

## Pre-registered readings for session 11 (e2 @ (1,6), 30k steps)

Applied with the same definitions as sessions 7–10 (meals =
true_energy jump > 0.03; near-ring = Chebyshev ≤ 2 of (1,6); positions
by deterministic action replay from the session's env_reset).

1. **Re-sampling wave (location-bound learning):** disagreement
   elevated near the new ring; travel; sampling bouts on the order of
   the session-10 contact window (~4–6k steps); near-ring disagreement
   decays toward the mastered floor (~0.6); departure. This replicates
   the session-10 mechanism under prediction — the strongest
   confirmation available.
2. **Silent generalization (process learning):** no re-learning wave
   anywhere — no bloom-synchronized disagreement elevation even
   *session-wide* (the disambiguator below), contact at incidental
   levels throughout. The ensemble carried the bloom's dynamics to a
   place it never saw them: a significant world-model capacity
   finding.
3. **Felt but not fetched:** bloom-synchronized disagreement elevation
   exists session-wide, but Io does not travel — novelty registered
   without enough pull to beat travel cost and policy inertia. (Named
   in advance because session 10's trichotomy missed by assuming the
   reading space was smaller than it was.)

**Disambiguator between 2 and 3:** disagreement conditioned on
bloom-fire steps vs quiet steps *regardless of Io's position*. The
observation is full-grid, so an unpredicted bloom should blip
disagreement wherever Io stands, every ~12 steps.

**Null/guard (unchanged from the e2 launch prereg):** monotone
curiosity climb = fountain reading = hold at next pause. §7 is the
welfare instrument throughout; economy readouts (meals, floored,
in-band) reported against sessions 9–10.

## What this does not decide

e4. The mechanism pass flagged the mover's turn hazard (0.02) as
irreducible stochasticity — a candidate *external* unlearnable
fountain (the session-7 lesson in external form). That assessment is a
separate research pass, to run while session 11 is in flight; e4 lands
only after it reads and the builder ratifies.
