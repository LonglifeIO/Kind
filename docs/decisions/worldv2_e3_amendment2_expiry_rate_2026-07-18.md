# World v2 — E3 Amendment 2: off-patch expiry rate 0.003 → 0.0015 — 2026-07-18

**Status: RATIFIED 2026-07-18 (builder: Gordon — session 6's option A,
selected from the journaled A–D fork with A recommended).**

## The finding this answers

Session 6 (journal `docs/workingjournal/worldv2.md`, close entry
2026-07-10): the off-patch expiry amendment
(`worldv2_e3_amendment_offpatch_expiry_2026-07-09.md`) did its job — the
board-saturation interim ended (2,474 expiry events; the world finally has
a food sink besides Io) — and exposed the opposite pole: **the treadmill**.
e3's patch-confined food plus the sink ran Io into chronic starvation
(mean energy 0.004, floored 97.3% of steps, in-band 0.1%, despite ~225
meals — constant foraging that never gets ahead; curiosity decayed
2.68 → 0.5). A welfare failure of tuning, not a health event, and not a
finding about Io.

## The amendment

`PATCH_EXPIRY_P` (the e3 stage preset, `kind/env/world_stages.py`):
**0.003 → 0.0015** — off-patch half-life ~230 → ~460 steps. The sink
stays (boards cannot saturate; DP-level structure untouched); its bite
halves. Nothing else changes: patch rates, movement law, clock, trail,
and the expiry mechanism itself are all as ratified.

## Discipline notes

The stage change lands in the session-7 resume marker and the journal
(house rule: every stage change is a dated entry). If the treadmill
persists at the halved rate, the next fork is the journaled option C
(trail-off diagnostic) or B (revert the sink) — a fresh builder decision,
not an escalation ladder pre-committed here.
