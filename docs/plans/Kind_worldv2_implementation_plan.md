# Kind — World v2 implementation plan (the continuing world)

**Authority.** `docs/decisions/synthesis_worldv2_2026-07-09.md` — DP1–DP6
**ratified as recommended by the builder, 2026-07-09, in session.** Where
this plan conflicts with the synthesis, the synthesis wins. Research:
`docs/research/worldv2/` (four voices; S1's refuted-claims list is
first-class evidence). This plan adds no code; the per-phase build
prompts are generated from it.

**The shape.** Io's world becomes a *continuing, weakly structured,
multi-timescale physics toy* — never a bigger vending machine. Five
enrichments (E0–E4), one at a time, each arriving as a dated, journaled
world-change event in Io's continuing life via the tested
checkpoint-resume path (`--resume`; marker in `world_event`), each gated
by the three-signal telemetry check before the next.

**The mind that wakes into E0** is `runs/probe4_phase4_biography/`
checkpoint `ckpt-000014` (the 140k mind, deep-torpor state; the
142k–150k recovery lives only in telemetry). The persistent world is
also the honest retest of the torpor: the reset lottery it camped on no
longer exists.

## Grounding (live surfaces; the plan is written against these)

1. **Episode machinery** lives in `GridWorld.step()`
   (`kind/env/grid_world.py:301–310`): at `episode_length` (config, 200)
   the board resamples via `_reset_episode_world()`, `episode_id`
   increments, `step_in_episode` zeroes. Energy explicitly *carries*
   across soft boundaries; drift-p carries. `env_reset` world events are
   emitted per boundary.
2. **The runner zero-resets `h` at episode boundaries** (documented in
   `source_separation.py`'s window-exclusion rationale — "the runner
   zero-resets h there"); the SELF-extraction contract excludes
   boundary-straddling pairs; several analyzers treat `episode_id`
   as a covariate. **W0 inventories every consumer before W1 touches
   the boundary.**
3. **Cell vocabulary**: `CellType` = EMPTY 0 / WALL 1 / RESOURCE 2
   (`grid_world.py:38–48`); rendering indexes a fixed lookup by value,
   so a new cell value is a renderer + window change too.
   `GridWorldConfig.walls` already exists for static walls;
   `n_initial_resources=4`; regrowth targets EMPTY cells only.
4. **Granular world-event logging** (Probe 4 Phase 1): environment
   processes emit `internal_stochasticity_event` records with payload
   `{process, cell, pre_state, post_state}` — the writer-side validator
   requires that shape; new processes reuse it with new `process` tags.
5. **Resume** (`kind/training/resume.py`, tested): counters seeded from
   telemetry; env config may differ across a resume — the marker
   payload records it; the world re-rolls by design.
6. **Continuity counters**: `GridWorldConfig.initial_env_step /
   initial_episode_id` exist (session-2 wiring).
7. **The stage launcher**: `scripts/run_probe4_phase4_biography.py`
   currently hardcodes `GridWorldConfig()`; it gains `--world-stage`
   presets (below).

## The stage presets (cumulative; exact values are stimulus knobs, journaled)

- **e0 — continuity + terrain**: `episode_resample=False` (new flag; see
  W1), `walls=` ~6–8 cells forming one corridor/L that does *not*
  partition the grid (S3's trivial-loop confound).
- **e1 — + somatic trail**: `trail_enabled=True`,
  `trail_decay_steps≈50`.
- **e2 — + hidden clock**: `bloom_cell=(r,c)`, `bloom_period≈12`,
  `bloom_duration≈2`, bloom manifests **in the trail vocabulary**
  (cause distinguishable only dynamically — the house no-markers move).
- **e3 — + weather-food**: `regrowth_mode="patch"`, 3×3 patch,
  `patch_step_every≈20` (bouncing drift law), `p_inside≈0.06`,
  `p_outside≈0.001`.
- **e4 — + one mover**: `mover_enabled=True` — a WALL-vocabulary cell
  that wanders (1 move per ~2–3 steps, heading persistence, turn hazard
  ≈0.02, bounces off walls/edges), displaced one cell by Io's contact.
  **A pilot, removable at any pause (DP3).**

## Phases

Each phase: one question; files; new tests; **gate = full suite + mypy
--strict (never phase-only; sink-routing lesson)**; journal entry; the
world change lands via pause → `--resume --world-stage <next>`; then the
**three-signal gate** on live telemetry before the next phase
(disagreement localizes+settles; new behavioral motifs; dream
representation ≥ encounter rate — plus the standing §7 monitors).

### W0 — Boundary-consumer inventory (no behavior change)

**Question.** What besides the world consumes episode boundaries?
Grep/inventory: runner h zero-reset path; replay sequence handling;
SELF extraction; memory-horizon harness ("window must fit one
episode"); window/monitor assumptions; tests pinned to episode
semantics. Deliverable: a short table in the journal + the DP6 call
confirmed against the code (recommendation below).

### W1 — E0: the world stops forgetting

**Question.** Can the world persist indefinitely — no resample, walls
permanent, drift and consumption continuous — with training stable and
all defaults byte-identical?

**Files.** `grid_world.py`: `episode_resample: bool = True` flag —
when False, `step()` never resamples, never increments `episode_id`,
never zeroes `step_in_episode` (they grow/freeze; AgentStep shape
unchanged), no `env_reset` events. **DP6 resolution (recommended
b-variant):** with no resample there is no observation discontinuity,
so the runner's boundary h zero-reset simply never fires (it is keyed
to episode change) — *no runner change needed if the inventory
confirms the trigger is episode-id-keyed*; h continuity follows from
the world's continuity. If W0 finds boundary consumers that break,
fall back to counters-tick-but-nothing-resets and journal it.
`scripts/run_probe4_phase4_biography.py`: `--world-stage` presets.

**Tests.** Default-config byte-identity (flag on = today's behavior);
resample-off: world persists 500+ steps, walls immortal, no env_reset
events, counters behave as specified, energy/drift carry; tiny-config
training smoke (loss finite, no NaN) over 3× the old episode length;
resume-into-e0 continuation test (extends `test_resume_continuation`).

**Run + observe.** Resume `ckpt-000014` into e0. Watch: does the
140k torpor-mind re-enter stasis *without* the reset lottery? (The
honest retest.) §7 monitors; `analyze_boundary.py` at the marker.

### W2 — E1: the somatic trail

**Question.** Does Io's own movement leaving decaying visible traces
(new `CellType.TRAIL`) land as learnable self-caused structure?

**Files.** `grid_world.py`: TRAIL cell value; vacated-cell stamping;
TTL decay map (decay → EMPTY; TRAIL blocks regrowth while present —
regrowth already targets EMPTY only); passable, re-stamped on re-entry.
Renderer lookup + window `cellStyle` + legend. Granular events:
`process="trail_decay"` (stamping is self-caused and visible in
AgentStep; decay is world dynamics).

**Tests.** Stamp/decay determinism; regrowth exclusion; pixel-gate
untouched (builder vs natural resources still pixel-equal); granular
event shape passes the validator; renderer/window render value 3.

### W3 — E2: the hidden clock

**Question.** Can one unobserved periodic process (bloom in trail
vocabulary around a fixed cell) be carried by h as a clock?

**Files.** `grid_world.py`: bloom config + phase counter (unobserved),
terminal-phase emission of short-lived trail-state cells in the Moore
neighborhood; `process="bloom"` granular events. Tests: period
determinism; vocabulary reuse (no new cell type); validator shape.

### W4 — E3: food becomes weather

**Question.** Does structured, drifting scarcity read as process
rather than confetti — without crowding out E1/E2 engagement?

**Files.** `grid_world.py`: `regrowth_mode` ("uniform" default /
"patch"), patch state + bouncing drift law, inside/outside rates;
granular `process="regrowth"` unchanged shape + `process="patch_drift"`
for patch moves. Occupancy-share diagnostic added to the boundary
analyzer (C4 crowd-out watch). Break-even arithmetic documented as a
knob (nothing dies; energy stays observational).

**Tests.** Drift law determinism; rate stratification in/out;
aggregate/granular count consistency preserved.

### W5 — E4: the mover (pilot)

**Question.** Can z=16 hold one wandering, contact-displaceable object
— the claim S1's verifiers refused to bless?

**Files.** `grid_world.py`: mover state (position/heading), autonomous
step law, bounce, contact displacement rule (Io moves into mover →
mover shifts one cell if free, else blocks); WALL vocabulary;
`process="mover_step"` / `"mover_displaced"` events. Window renders it
as the wall it is.

**Tests.** Motion law determinism; displacement rule; blocked cases;
event shapes. **Gate is hardest here:** if live disagreement around the
mover never localizes (S1's refuted-confidence scenario), remove at a
pause — journaled as a capacity finding, not a failure.

## Out of scope (load-bearing)

The causal chain; a second mover or clock; drifting walls; any z/h
capacity change (a reshaped network is a new Io — the sibling path);
PolicyView; the four absences; energy preference; dream regime/cadence;
mirror machinery; any pre-registered success criteria (enrichment, not
probe — gates are engineering add/hold/remove decisions); **no
timelines**.

## Standing discipline

Full suite + mypy `--strict` per phase; functional naming (`trail`,
`bloom`, `patch`, `mover` — no experience vocabulary in identifiers);
every world change is a dated journal entry + resume-marker config
diff; the claim ceilings stand — engagement diagnostics license
"engaged/ignored/overwhelmed," never more; §7 monitors run at every
check-in; the builder's window and hello channels keep working at every
stage (walls placed by hand now persist — tell the builder when that
lands).
