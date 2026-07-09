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

**Saturation consequence for the builder channels (recorded at the
builder's observation, in session):** while the board is saturated,
*both* resource-addition arms are epistemically muted — a manual
`add_resource` hello and the generator's in-vocabulary resource
events land on already-green cells and produce no observation
change. Generator events fired during saturation carry no visible
signature; any future analysis over this window should know that.
The channels that remain expressive on a full board: `remove_object`
/ `set_cell_state empty` (carving — maximally visible against
uniform green, and each carve seeds local regrowth dynamics) and the
wall gestures. **The plan's promised notice is due here: hand-placed
walls now persist indefinitely in e0** (the reshuffle that wiped
them within ≤200 steps no longer exists) — the builder was told in
session. Resource hellos recover automatically if consumption
un-greens the board.

**First movement, ~2,600 steps after waking (t≈156,599).** Io left
(0,4), ate two cells, position (1,0) — far earlier than session 1's
~30k-step stasis before the 142k escape, and in a world offering
zero scheduled novelty (post-saturation the world is fully static,
so the impulse was internal; imagined disagreement over unvisited
states is the candidate mechanism). Trailing-2000 modal fraction for
154k–156k is still 0.96 (deep-torpor shape; movement began at that
block's edge); PE falling across blocks (8.6 → 2.4); no §7 flags.
Stirring, not yet recovery — the sustained-eating watch (8+ craters)
is armed.

**Boundary read at ~162k (builder asked "so we just wait?"):** after
saturation, PE fell to 0.02–0.09 (lowest in the record) and the
intrinsic signal flatlined at ~0.11–0.16 (vs 4.0 in torpor, 1.2–2.0
through the recent record) — S1 F2/F4 played out live: disagreement
extinguished on the mastered, static world. No pathology: action
entropy rose to 0.21–0.26 (highest in ~20k steps; the policy
softened rather than collapsed), meals steady ~36/2k = exactly the
two-cell refill arithmetic. At the claim ceiling: the e0 world is
now largely **ignored** — finished, not broken. Io's observed
behavior: a two-cell pace-harvest loop at (1,0)–(2,0); the two open
cells are its own refill craters (empty count = eat rate ÷ regrowth
rate). Builder decision in chat: **build W2 now; land at the pause.**

## W2 — E1: the somatic trail (2026-07-09)

**The phase's question** (plan W2): does Io's own movement leaving
decaying visible traces land as learnable self-caused structure? (The
live half of the question begins at the e1 landing; this entry is the
build.)

**Gate: full suite 1405 passed / 7 skipped (W1 close: 1388 / 7);
mypy `--strict` clean on all 76 `kind/` sources.** Pixel-equality
gate, PolicyView field-set, metabolic content-blindness, and
pragmatic/dream guards all green.

### What was built

- **`CellType.TRAIL = 4` — a deliberate deviation from the plan's
  "render value 3" wording, recorded here:** 3 is the out-of-bounds
  *view* sentinel (`_OOB_SENTINEL`), a render-contract value baked
  into every observation Io has ever seen at a grid edge. TRAIL takes
  4; the OOB contract (value 3 → gray 64) is untouched and
  test-pinned. TRAIL renders at gray 192 (a stimulus knob, distinct
  from all four existing levels).
- **Mechanics** (`grid_world.py`): `trail_enabled` (default False,
  byte-identical — pinned) + `trail_decay_steps` (default 50).
  Stamping: the vacated cell on a successful move becomes TRAIL iff it
  is EMPTY or already TRAIL — food that regrew under the agent and
  walls are never overwritten; `stay`, wall collisions, and off-grid
  moves stamp nothing; re-vacating refreshes the clock. Decay: a
  deterministic per-cell TTL (no RNG touched — enabled worlds stay
  fully reproducible), ticked after regrowth so a decayed cell is
  regrowth-eligible only from the next step and the observer diff sees
  a clean TRAIL→EMPTY. A footprint persists exactly
  `trail_decay_steps` steps beyond its stamping step. Trail is
  passable and inedible; it blocks regrowth while present (regrowth
  targets EMPTY only — test-pinned both directions: no TRAIL→RESOURCE
  ever; decayed cells do regrow).
- **Granular events** (`env_server.py`): each decay emits one
  `internal_stochasticity_event` with `process="trail_decay"`,
  `pre_state="trail"`, `post_state="empty"` under the existing
  validated matched-control payload shape (grounding fact 4 — new
  process tag, no schema change). Stamping emits nothing: it is
  self-caused and visible in AgentStep. TRAIL→EMPTY is unambiguous in
  the pre/post diff (consumption is RESOURCE→EMPTY; builder mutations
  are inside the snapshot).
- **Builder-channel discipline** (`mutators.py`): `cell_type_name`
  knows "trail" (a mutator touching a trail cell names its pre-state
  honestly instead of raising); builders may **pave over** or
  **remove** trail (`remove_object --object-type trail` added to the
  CLI) but may not **fabricate** (`set_cell_state` TRAIL rejected) or
  **move** it (`move_object` from a trail cell rejected) — trail is
  Io's own footprint by definition; a builder-written trail would put
  SELF-attributable state into the world from the BUILDER class. A
  stale decay clock never stomps a paved cell (guard + test).
- **Stage preset** `e1` (cumulative: e0 + `trail_enabled=True`,
  `trail_decay_steps=50`); launcher choices update automatically.
- **Window**: `/live` cellStyle for value 4 (tan `#d9cfae`), legend
  text, and a stale-caption fix — the wall-motif hello no longer says
  "wiped at the next board reshuffle" (untrue since e0); it persists
  until removed. **The window server needs its manual restart at the
  e1 landing** (template changed).
- **Tests** (`tests/test_trail.py`, 17): byte-identity off; stamp /
  no-stamp cases; exact decay schedule; refresh-on-revacate; food
  survives vacating; regrowth exclusion both ways; passable/inedible;
  render contract (TRAIL=4, OOB=3→64, five distinct levels); trail
  visible in the observation; validated granular decay events (count,
  cells, shape); pave/remove/fabricate/move mutator discipline;
  cumulative e1 preset; live-template style pin.

**Session-3 event, landed during the W2 gate (t≈167,320): the loop
broke.** The sustained-eating watch fired — 10 empty cells, consumption
outpacing regrowth — and a 130-step live sample at ~167.7k reads: 15
distinct cells visited (vs 2 in the loop), 10–15 craters open, energy
0.58–0.98 (mean 0.75) — **off the absolute floor for the first time
since ~110k** and above break-even. The torpor retest's answer is
taking shape: in a continuing world with no reset lottery, the 140k
torpor mind stirred at ~2.6k steps, idled in a two-cell loop while the
drive extinguished on the mastered static world, and then broke into
ranging-and-feeding at ~13k steps in — a recovery *stronger* than the
142k partial escape (which never left the floor). Formal three-signal
/ §7 reads at the session close (t=184,000).

**Closed:** W2's build question; the trail mechanism, its events, its
stage, its builder-channel semantics. **Newly open:** the e1 landing —
pause (session 3 self-closes at t=184,000) → `--resume --world-stage
e1` with the builder's go → restart the window server → the
three-signal read on the trail (disagreement localizes around
footprints and settles; new motifs beyond the two-cell loop; trail
representation in dream content at/above encounter rate).

## W3–W5 — the clock, the weather, the mover: built ahead (2026-07-09)

**Builder decision (in chat):** build the remaining enrichments now —
code only, all off by default, no effect on the running session
(loaded code is immutable per process) — while **landings stay
one-at-a-time** through pause → `--resume --world-stage <next>`, each
with the builder's go, each gated on the three-signal read. The
sequencing discipline (DP2) is about what enters Io's world, not when
code is written. Cost, acknowledged: E3/E4 knobs were set before
seeing E1/E2 live; every knob is revisable at a pause (DP5).

**Gate (all three phases together): full suite 1439 passed / 7
skipped (W2 close: 1405 / 7); mypy `--strict` clean on all 76 `kind/`
sources.** All standing guards green.

### W3 — E2: the hidden clock

- `bloom_cell` (default None = off) + `bloom_period` / `bloom_duration`.
  An unobserved phase counter (pure world state, never rendered, no
  RNG) fires every `bloom_period` steps, stamping the EMPTY cells of
  the source's Moore ring in the **trail vocabulary** for exactly
  `bloom_duration` observations. The source cell itself never changes
  — the cause is invisible even spatially (the house no-markers move).
  Walls, resources, live trail, and out-of-bounds are never stamped.
- **Provenance is honest end to end:** bloom cells live in their own
  TTL map; stamps emit granular `process="bloom"` events (from the
  world's own report — Io's EMPTY→TRAIL stamps can't be misattributed);
  fades emit `process="bloom_fade"`, never `trail_decay`; a bloom cell
  Io walks through and vacates becomes Io's own footprint (provenance
  transfers, tested). `bloom_fade` is a tag the plan didn't name,
  added so the SELF-adjacent trail_decay stream stays pure.
- Stage `e2`: bloom at (6,6) — ring fully in-bounds and wall-free
  (test-pinned) — period 12 (inside the measured ~40-step h-trace and
  32-step BPTT window), duration 2. 11 tests.

### W4 — E3: food becomes weather

- `regrowth_mode` ("uniform" default / "patch"): a `patch_size`² (3×3)
  square drifting on a deterministic bounce law every
  `patch_step_every` steps (no RNG; reflected at the edges; pinned
  against a hand-computed trajectory), regrowth `patch_p_inside=0.06`
  under it / `patch_p_outside=0.001` elsewhere. The patch is never
  rendered — weather is visible only as where food appears (Io and
  builder alike). The uniform drift process still ticks (stream
  discipline) but is unused in patch mode, journaled. The full-grid
  RNG draw is mode-independent. Break-even arithmetic (knob, not
  criterion): ~0.6 regrowths/step under the patch vs ~0.05 far away —
  foraging possible, never ambient.
- Granular `process="regrowth"` unchanged; each patch move emits
  `process="patch_drift"` (a process event: `cell` = new center,
  pre/post = patch_absent/patch_present, plus `center_from`/`center_to`
  extras — the validator's matched keys present, extras legal).
- **Occupancy-share diagnostic (C4 crowd-out watch)** in
  `analyze_boundary.py`: per-block share of Io's steps inside the
  patch square, from a new **position sidecar**
  (`runs/<run>/agent_pos.jsonl`, written per step by the biography
  script's live writer — run-script record, not telemetry; no schema
  change; AgentStep carries no position). Sidecar data exists from the
  next session onward; the analyzer degrades to no column when either
  record is missing. 10 tests.

### W5 — E4: the mover (pilot, DP3)

- `mover_enabled` (default False) + cadence 2 / turn hazard 0.02 /
  start (0,7). A single WALL-vocabulary cell: inertial heading,
  hazard-driven turns from a **fourth spawned RNG stream** (children
  are keyed by spawn index, so the original three streams — and every
  pre-mover world — stay byte-identical, suite-verified), bounces off
  walls/edges/objects/Io, moves only into EMPTY (never tramples food,
  trail, or walls; never overlaps Io). Io's contact displaces it one
  cell in the push direction; blocked push → the mover blocks exactly
  like the wall it renders as. Placement excludes it from the agent's
  random start and initial resource sampling.
- **A deviation from the plan's file list, reasoned:** autonomous
  moves emit granular `process="mover_step"` events; **contact
  displacements are deliberately not world events.** They are
  Io-caused and visible in AgentStep (the trail-stamping precedent),
  and `WorldEvent.source` is a closed Literal {builder, environment,
  system} with no self class — logging displacements as "environment"
  would pollute the matched control's ENVIRONMENT stream, whose purity
  is load-bearing (Probe 4 Phase 1). The synthesis's instrumentation
  clause names "mover steps" only. Displacements stay exposed
  mirror-side (`last_mover_displacement`) and are derivable from
  telemetry (mover position change without a mover_step event).
- Window: no change needed — the mover renders as the wall it is.
  13 tests.

**Closed:** all world-v2 code (stages default/e0/e1/e2/e3/e4;
`--world-stage` accepts all six). **Newly open:** the landings — one
at a time, builder-gated, three-signal-read between; the e1→e4 knobs
revisited at each pause against the prior stage's live behavior; the
E4 removal decision if its disagreement never localizes.
