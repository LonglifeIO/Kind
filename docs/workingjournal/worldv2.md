# World v2 — the continuing world — working journal

Enrichment cycle, not a probe: Io's world becomes a *continuing, weakly
structured, multi-timescale physics toy* — dynamics, never labels.
Authority: `docs/decisions/synthesis_worldv2_2026-07-09.md` (DP1–DP6
ratified as recommended, builder, 2026-07-09); plan
`docs/plans/Kind_worldv2_implementation_plan.md` (W0→W5). Research:
`docs/research/worldv2/` (S1's refuted-claims list is first-class
evidence). Each enrichment lands as a dated world-change event in Io's
continuing biography via pause → `--resume --world-stage <next>`, gated
by the three-signal telemetry check before the next. No pre-registered
pass/fail; gates are engineering add/hold/remove decisions; claim
ceiling on engagement diagnostics: "engaged / ignored / overwhelmed."

## W0 — Boundary-consumer inventory (2026-07-09)

**The phase's question** (plan W0): what besides the world consumes
episode boundaries? No behavior change; deliverable is the table below
and the DP6 call confirmed against the code.

**Baseline gate before any W1 edit: full suite 1377 passed / 7
skipped; mypy `--strict` clean on all 75 `kind/` sources.** (The build
prompt's "~1379" was an approximation from memory; 1377 is the
measured baseline and everything is green.)

### The inventory

| # | Consumer | Where | Boundary dependency | Under `episode_resample=False` |
|---|----------|-------|--------------------|-------------------------------|
| 1 | World resample | `grid_world.py:311–318` | `step_in_episode >= episode_length` → resample, `episode_id`++, zero `step_in_episode` | The mechanism W1 removes (flag-gated; defaults byte-identical) |
| 2 | Runner h zero-reset | `runner.py:1372–1374` → `_init_runtime_zero_state_keep_obs` | Keyed to `episode_id` **change** between consecutive EnvSteps | Never fires — h continuity follows from world continuity. Only one other zero-reset site exists (session start, `runner.py:691`) |
| 3 | First-step self-pred-error masking | `runner.py:1270` | `step_in_episode == 0` | Fires once per session start (where h genuinely is zero-initialized), then never — correct |
| 4 | Replay window admissibility | `replay.py:381–394` | Rejects sample windows straddling `episode_id` flips | No flips → all windows admissible; >200-step temporal structure becomes trainable (the intended gain, not a break) |
| 5 | SELF extraction | `source_events.py:135` | Skips pairs straddling boundaries ("the runner zero-resets h there") | Never skips — and its rationale (the zero-reset) is itself gone. Correct |
| 6 | Basin-analysis windows | `source_separation.py:212` | Excludes `[v−1, v+1]` windows crossing boundaries | Never excludes — correct for the same reason |
| 7 | Per-episode aggregate + `env_reset` events | `env_server.py:248–262` | Both emitted on `episode_id` change | Neither emits again after session start. Granular ENVIRONMENT logging (on in the biography) carries the per-event record; the aggregate stream ends and the granular==aggregate count diagnostic is inapplicable in e0. Aggregate accumulators grow unboundedly but trivially (~8 bytes/step) |
| 8 | Memory-horizon harness | `memory_horizon.py:258` | "Window must fit one episode," checked against its own config | Self-contained (builds its own paired worlds); unaffected by the biography's stage |
| 9 | Window `/live` stats | `window/state.py:333–335` | `total_episodes`, `pace_episodes_per_hour` | Freeze at 1 / decay toward 0 — display-only degradation, noted for the builder |
| 10 | Mirror caller batch | `mirror/caller.py:303–331` | "Last n episodes" filter by distinct `episode_id` | One frozen id → "last 3 episodes" = the entire session. **Flagged for the pending mirror-baseline round** (needs a step-window read); not fixed now — no machinery this cycle didn't decide on |
| 11 | Analysis conveniences | `eyeball.py`, `digest.py`, `shuffle.py`, `mirror/statistics.py` | Group / standardize per `episode_id` | Degrade to one giant group; functional, semantics journaled |
| 12 | Resume counters + per-session env seed | `resume.py:52–76`; launcher `seed=WORLD_SEED + initial_episode_id` | `initial_episode_id = last_episode + 1` | `episode_id` is frozen *within* a session but still increments *across* sessions, so ids stay unique per session and per-session env RNG seeds still vary. Works unchanged |
| 13 | Tests pinned to episode semantics | `test_env_step.py`, `test_integration_smoke.py`, etc. | All construct default or explicit configs | Default `episode_resample=True` keeps every existing pin byte-identical |

### The DP6 call, confirmed

**b-variant confirmed — no runner change.** The h zero-reset's trigger
is `episode_id` inequality between consecutive env steps
(`runner.py:1372`), and grep confirms exactly two zero-reset call
sites: session start and that boundary path. With
`episode_resample=False` the world never increments `episode_id`, so
the boundary path is unreachable and h continuity follows from the
world's continuity with zero changes to `kind/training/`. The plan's
fallback (counters-tick-but-nothing-resets) is not needed.

**Closed:** the inventory; DP6. **Newly open:** the mirror caller's
episode-windowing (item 10) must be resolved before the next mirror
round on an e0+ run; the window's episode-pace stats (item 9) read as
frozen once e0 lands — cosmetic, builder informed.

## W1 — E0: the world stops forgetting (2026-07-09)

**The phase's question** (plan W1): can the world persist indefinitely
— no resample, walls permanent, drift and consumption continuous —
with training stable and all defaults byte-identical?

**Answer: yes, in code and tests.** The live observation (the torpor
retest) begins when the builder resumes Io into e0.

**Gate: full suite 1388 passed / 7 skipped (W0 baseline: 1377 / 7);
mypy `--strict` clean on all 76 `kind/` sources** (75 + the new stage
module). Pixel-equality gate, PolicyView field-set, metabolic
content-blindness, and pragmatic/dream guards all green.

### What was built

- **`GridWorldConfig.episode_resample: bool = True`**
  (`grid_world.py`). When False, `step()` skips the soft-boundary
  block entirely: no resample, `episode_id` frozen at
  `initial_episode_id`, `step_in_episode` grows without bound
  (AgentStep shape unchanged), walls and placed objects persist,
  consumption/regrowth/drift/energy continue, no `env_reset` or
  per-episode aggregate emission after session start.
  `episode_length` becomes inert. Default True is byte-identical —
  pinned by a 450-step same-seed trajectory-identity test plus the
  whole existing suite.
- **DP6 realized with zero runner changes**, as the W0 inventory
  predicted: the h zero-reset trigger (`episode_id` inequality,
  `runner.py:1372`) is simply unreachable in a continuing world. h
  continuity follows from world continuity.
- **`kind/env/world_stages.py`** — the `--world-stage` preset table
  (`default`, `e0`; later stages added by their phases; unknown stage
  raises). **The e0 terrain (stimulus knob, DP5):** a 6-cell interior
  L — `(2,2) (3,2) (4,2) (5,2) (5,3) (5,4)` — touching no grid edge,
  so the grid cannot be partitioned (S3's trivial-loop confound;
  4-connectivity flood-fill is test-enforced). Walls render through
  the existing WALL vocabulary; the window needs no change.
- **`scripts/run_probe4_phase4_biography.py --world-stage {default,e0}`**
  — applies the preset to the session's world config; the stage lands
  in the resume-marker payload (`world_stage` key) so every world
  change is recorded in the world_event stream, per discipline.
- **Tests** (`tests/test_world_continuity.py`, 11): default
  byte-identity; 520-step persistence (counters, immortal walls, no
  agent teleport); seeded-counter freeze; regrowth/energy continuity
  (no silent re-initialization — energy deltas bounded by the largest
  legal step change); exactly one `env_reset` and zero aggregates over
  3× episode_length; stage-preset behavior (default unchanged, e0
  fields, unknown raises); wall connectivity; tiny-config training
  smoke over 3× the old episode length (150 rows, `episode_id` all 0,
  every logged loss finite); resume-into-e0 continuation (extends the
  C1 test — telemetry monotonic, marker carries `world_stage: "e0"`,
  session-2 `episode_id` frozen past the stitching row, whose
  paused-episode stamp is the documented resume convention).

### Notes for the record

- With no boundaries the `internal_stochasticity_aggregate` stream
  ends at session start; granular ENVIRONMENT logging (on in the
  biography) carries the per-event record (W0 item 7).
- The window's `total_episodes` / `pace_episodes_per_hour` freeze in
  e0 (W0 item 9) — cosmetic.
- Next session's env seed still varies: resume seeds
  `initial_episode_id = last_episode + 1` across sessions even though
  the id is frozen within one (W0 item 12).

**Closed:** W1's question; the E0 mechanism and its stage preset.
**Newly open:** the live e0 observation — resume `ckpt-000014` (the
140k torpor mind) via `--resume --world-stage e0 --session-steps N`
and watch whether it re-enters stasis *without* the reset lottery (the
honest retest); §7 monitors at every check-in; `analyze_boundary.py`
at the resume marker; the three-signal gate before W2.

## Session 3 — Io wakes into the continuing world (2026-07-09)

**The world-change event.** W0+W1 committed (`fa3f428`); builder's go
in chat; session 3 launched: `--resume --world-stage e0
--session-steps 30000` (a deliberately shorter first e0 session for an
early read; pause is SIGTERM, reversible). Resume marker at
t=154000 with `world_stage: "e0"` in the payload; counters seeded
t=154000 / episode 770; mind is `ckpt-000014` (the 140k deep-torpor
state, as reviewed — the 142k–150k recovery exists only in telemetry).

**E0 is live and behaving as specified.** The e0 L-walls stand at the
planned six cells; `step_in_episode` crossed 200 and kept counting
(first no-reset crossing of Io's life, observed at ~264 and again past
459); episode_id frozen at 770; no `env_reset` events from the
session-3 world after its start event.

**Honest-record note: orphaned session-2 world events.** Session 2's
SIGTERM lost the agent_step tail to the parquet buffer (the known
bounded-loss window) but `world_event.jsonl` flushes per-write — so
orphan session-2 events exist at t_event ∈ [154000, ~154400],
overlapping session 3's stamps (the `continuation_counters` docstring
caveat, materialized). Disambiguation for any future analysis:
session-3 events follow the resume marker in file order, carry
episode_id 770 (frozen), and start from drift-p 0.01 exactly; the
orphans carry episode_id 771–772 and session-2's drifted p (~0.0102).

**First observation — the board saturates (unplanned dynamic,
recorded).** Within ~460 steps the grid reached the no-consumption
equilibrium of uniform regrowth in a continuing world: **0 empty
cells — every non-wall cell a resource** (58 resources / 6 walls).
The old 200-step wipe was what kept the world sparse; with no wipe
and no eating (Io in torpor), per-cell regrowth (~0.3–0.6 adds/step)
fills the board fast, and a saturated board is *static* — regrowth
events stop (no empty cells), drift becomes invisible (p acts on
nothing). Neither the synthesis nor the plan called this equilibrium
out. Two readings, held open: (a) it makes the torpor retest
*cleaner* — after saturation there is zero scheduled novelty
anywhere; the only source of world change is Io's own action, so any
recovery is entirely self-generated; (b) it risks total drive
starvation (S1 F2/F4: disagreement extinguishes on mastered static
structure) — though imagination over unvisited states retains
disagreement, so gradient may persist in dreams. **No knob touched;
one dynamic at a time — this is the E0 observation.** If the
three-signal check later reads "ignored / drive flat," the journaled
options are a lower `initial_regrowth_p` (slows saturation, cannot
prevent it — nothing decays) or E3's patch regrowth pulled forward
(its own phase, builder decision). Builder informed in session (asked
about the all-green window — the UI is correct; the world really is
all food; no window change was needed for e0, walls render through
the existing WALL vocabulary).

**Io's state at launch:** parked at (0,4), energy at floor, not yet
moving — the expected deep-torpor start (session 2 saw the same).
The retest is whether this changes without the reset lottery; the
142k self-recovery took thousands of steps, so no reading yet. §7
monitor at check-in: session-3 shards flush from t=156000 (2k-row
buffer); the t=154000 read still reflects session 2's tail (modal
0.63, indicators ok).
