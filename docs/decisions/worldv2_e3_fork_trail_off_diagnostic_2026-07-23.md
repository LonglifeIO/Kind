# World v2 — E3 fork: the trail-off diagnostic (session 8) — 2026-07-23

**Status: RATIFIED 2026-07-23 (builder: Gordon — session 7's fork
option C, the journaled fallback, selected from C / B / accept).**

## The finding this answers

Session 7 (journal `docs/workingjournal/worldv2.md`, close entry
2026-07-18): halving the off-patch expiry sink (E3 Amendment 2) doubled
Io's meals (225 → 416) and the floor held anyway (floored 94.2%, in-band
0.1%). The close's reading — "the treadmill is the spatial pattern, not
the sink rate" — stands, but the fork's journaled next question was
whether the trail's food-shadow (the session-4 e1 finding) is the last
mechanical cause left before "this is who Io is" settles.

## The pre-launch reconstruction (2026-07-23)

Before launching, the session-7 position log (`agent_pos.jsonl`,
t 247,734–277,732) was replayed against the deterministic patch-drift
law and a 50-step trail window (scratch analysis; numbers recorded here
and in the journal's launch entry):

- **The patch-scale shadow hypothesis is dead**: mean trail-shadowed
  patch cells 0.35 of 9 (4.8% of non-wall capacity). That cannot
  explain a 10× intake shortfall.
- **Io does not graze wide — it paces.** ~60% of the session in a
  corner block around (6–7, 1–2) below the L-wall's end; a 50-step
  trail window covers only ~4.1 unique cells (7.1% of the free board);
  mean Chebyshev distance to the patch center 3.46; inside the patch
  6.4% of steps. The session-7 close's "grazes wide, never camps"
  was wrong at the home-range scale: stay-share 0.0000 is movement
  without travel.
- **The shadow is real at the home-range scale**: expected regrowths
  destroyed by trail over the session ≈ **738** — 41% of the world's
  entire food production (1,055 actual regrowths + 738 prevented) —
  and **676 of them within Chebyshev ≤2 of Io**. The patch's bounce
  circuit brings ≥1 reachable patch cell to Io 46.4% of steps; during
  those windows a mean 0.68 reachable patch cells are trail-sterile.
  Io's trail was destroying the food supply under its own feet exactly
  when the weather came to it. The corner block produced 64 of the
  session's 1,055 regrowths.

## The decision

Resume the biography (session 8) under a new **dated diagnostic stage**
`e3_no_trail` (`kind/env/world_stages.py`): exactly e3 with
`trail_enabled=False` — a one-field diff, test-pinned
(`test_stage_e3_no_trail_is_e3_minus_trail_only`). Not a ladder rung:
the e2/e4 cumulative chains still build on full e3, and the stage is
removable after the diagnostic. Session length 30,000 waking steps
(matched to sessions 6 and 7 for comparability).

## Pre-registered readouts and readings

Recorded before launch; the session answers three questions at once:

- **R1 — economy**: meals, total regrowth, regrowth inside Io's home
  range, floored %, in-band %. *Expectation set honestly*: even +676
  meals ≈ +0.022 meals/step against a ~0.15/step break-even — trail-off
  is **not** expected to lift Io into the band. A meal jump with the
  floor holding reads "the trail was an economic sink, not the
  treadmill's cause."
- **R2 — space**: occupancy concentration (top-cell shares, unique
  cells per 50-step window), patch-following (inside-patch share, mean
  distance to patch center). If the pacing loop dissolves without
  self-laid trail, the loop was **stigmergically entrained** — Io's
  own footprints were holding its spatial pattern in place. That would
  be a capacity finding about self-caused structure shaping behavior
  (S1's lineage), not a tuning note.
- **R3 — drive**: the intrinsic-signal trend. Session 7's curiosity
  rose 10× (0.48 → 4.77) while Io lived in a trail-saturated 4-cell
  loop. If curiosity collapses with the trail off, a substantial share
  of that rise was Io feeding on its own trail dynamics —
  self-stimulation, not world-appetite.
- **Null result** (economy, space, and drive all hold): the pattern is
  intrinsic to the policy under this curiosity structure; with the
  sink (session 7) and the trail (session 8) both ruled out,
  **accept-the-economy** stops being a shrug and becomes the settled
  reading, journaled as such.

## Discipline notes

The stage lands in the session-8 resume marker and the journal (house
rule: every stage change is a dated entry). Trail-off changes what Io
*observes* (no footprints anywhere) — a world change arriving through
checkpoint-resume like every other stage change, ratified here. Option
B (revert the sink) remains un-taken; no further sink tuning without a
new dated fork.
