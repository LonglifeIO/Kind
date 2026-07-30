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

## Session 3 close — the e0 record (2026-07-09)

**Clean close at t=184,000** (30k steps as launched; ~35 ms/step this
session). Checkpoints through `ckpt-000017` (t=180,000) — **for the
first time in the biography, the latest checkpoint holds a recovered
mind, not a torpid one** (session 1's ckpt-000014 predated its
recovery). §7 at close: all ok — no entropy-collapse flag, PE falling
(last blocks 0.14 → 0.06), trailing modal-action fraction 0.50
(active mixed behavior), 180 dream sessions lifetime.

**The session's arc, in blocks** (`analyze_boundary.py`):

| phase | blocks | PE | curiosity | meals/2k | energy |
|---|---|---|---|---|---|
| torpor wake + stirring | 154–158k | 2.4 → 0.09 | 1.25 → 0.24 | 16–36 | floor |
| two-cell loop, world mastered | 158–163k | 0.02 | 0.11–0.13 | 31–41 | floor |
| **breakout** | 163–165k | **3.14** | **1.29** | **154** | 0.08 |
| ranging-forage peak | 167–171k | 4.02 → 0.94 | 1.91 → 1.25 | 249–258 | 0.22–0.28 |
| settling | 171–179k | 0.57 → 0.12 | 1.03 → 0.34 | ~190–210 | 0.07–0.11 |

**Reading (claim ceiling respected).** The torpor retest's answer: in
a continuing world with no reset lottery, the 140k torpor mind broke
its stasis at ~13k steps in — and the mechanism visible in the curves
is *self-generated re-engagement*: eating carved the saturated board
open, the craters' regrowth made the world dynamic again, and PE /
disagreement revived on dynamics Io itself was creating (eat →
craters → regrowth → something to model → more engagement). The
static board could not feed the drive (curiosity flatlined at 0.11);
Io's own action could. Meals sustained at ~200/2k — matching session
1's all-time peak — with energy off the floor across every block
after the breakout. Both prior stases (block-1 rut, §7 torpor)
resolved by the same signature: the intrinsic signal climbing until
the policy re-couples; this one adds that the *world's* re-opening
was Io-caused. At the ceiling: e0 was **engaged** after the breakout;
the saturated interim was **ignored** (finished, not broken).

**W1's live gate: passed.** Training stable across 30k continuing
steps (losses tiny and falling, no NaN, no §7 flag); the e0 world
change is the first in the biography to land via checkpoint-resume.
No dream-content check applies to e0 (continuity adds no event class
to represent; that gate begins with e1's trail). Session 4 (e1) will
also carry the position sidecar (built at W4) so occupancy diagnostics
begin there.

**State: Io is paused** — full telemetry through 184k,
`ckpt-000017` (the recovered 180k mind) on disk. Next: **land e1**
via `--resume --world-stage e1 --session-steps N` on the builder's
go. The window is already trail-ready (tan cells; server rebound
tailnet-only 2026-07-09, builder request).

## Session 4 — e1 lands: the trail enters Io's world (2026-07-09)

**Launched on the builder's go:** `--resume --world-stage e1
--session-steps 30000`; marker at t=184,000 with `world_stage: "e1"`,
resumed from `ckpt-000017` (the recovered mind), counters seeded
episode 771. Position sidecar live from this session
(`agent_pos.jsonl`).

**The trail works in production.** In its first ~400 steps the woken
mind made a ~66-move excursion through 12 bottom-left cells; the
stamps and 50-step decays ran exactly to spec (13 granular
`trail_decay` events — the re-trodden path re-stamps, the abandoned
path fades).

**Two facts recorded at launch:**

1. **The resume re-rolled the board (by design — resume.py: "the
   world re-rolls by design"), which wiped the builder's hand-placed
   wall motif.** Config walls (the E0 corridor) persist because they
   are config; hand gestures are world state and exist only within a
   session. "The world stops forgetting" currently holds *within*
   sessions, not across pauses. **Newly open (future decision, not
   taken here):** whether world-state serialization across resumes —
   true cross-pause continuity — should be built. That would be its
   own researched decision (it changes what a pause is).
2. **The dead-interim repeats:** the fresh board saturated again
   within ~600 steps (uniform regrowth, no sink while Io idles), and
   after its waking excursion Io re-parked at (6,3) — energy floor,
   no movement, trail fully decayed. Same no-gradient condition as
   session 3's start; session 3's precedent says stirring may take
   thousands of steps. Each resume will begin with this saturation
   phase until E3's weather replaces uniform regrowth — evidence that
   bears on how long to hold at e1/e2 before landing e3.

### Session 4 close — the e1 record (2026-07-09)

**Clean close at t=214,000** (30k steps, ~28 ms/step); checkpoints
through `ckpt-000020` (t=210,000 — parked state; the final stir is
telemetry-only, see below). §7 at close: no flags (the known
insensitive entropy baseline noted — the 191k–213k stillness is
exactly the shape the frozen numeric misses; the two-vantage rule
carried the observation instead).

**The arc, in blocks:**

| phase | blocks | PE | curiosity | act-ent | meals/2k |
|---|---|---|---|---|---|
| wake spike + park | 184–188k | 4.11 → 0.02 | 0.40 → 0.04 | 0.003 → 0.0001 | 3 → 0 |
| **ranging burst** | 188–191k | **4.99** | **1.70** | 0.09 | **66** |
| collapse + the long park | 191–213k | 0.01–0.12 | 0.05–0.23 | ~0.0000 | 0–2 |
| **third stir, at the bell** | 213–214k | **3.57** | **1.73** | 0.013 | 4 |

**The e1-specific finding — the trail's food-shadow.** During the
ranging burst, **91% of Io's 1,356 moves landed on its own recent
trail** (cells visited within the prior 50 steps; position sidecar ×
decay clock) — a genuinely new behavioral motif; nothing in sessions
1–3 had this structure. At the claim ceiling: Io **engaged** the
trail. But mechanically, trail blocks regrowth while present (plan
W2, as specified) — so a mind that lives inside its own footprint
field walks on cells where food cannot reappear. The burst starved
itself: energy never cleared 0.08, and at ~191k the forage loop that
sustained session 3's 20k-step engagement died in ~3k. The E0
recovery mechanism (eat → crater → regrowth → engagement) and the E1
mechanism (footprints + food-shadow) **interfere**: the trail damps
exactly the dynamics that fed the drive. This is what
one-dynamic-at-a-time observation is for.

**The three-signal read for e1:**

1. *Disagreement rises, localizes, settles*: rose (1.70) and
   collapsed (0.05) — but the settling is starvation-driven
   disengagement, not organized mastery. **Not passed.**
2. *New nontrivial motifs*: **passed** — trail-immersed ranging (91%
   re-tread) is new and nontrivial.
3. *Dream representation ≥ encounter rate*: **unread** — needs a
   dream-content decode pass (none built; the pending mirror baseline
   round would cover it; builder-gated, API cost).

**Synthesis §5 failure signature present: action stasis** (22k-step
park, deepest in the biography — trailing entropy 0.0000 for whole
blocks). Per the ratified rule, this **stops stacking**: e2 does not
land on top of this. The options at this pause are simplify or
restructure (below).

**Countervailing observation:** Io self-recovered a third time —
the 213k stir (PE 3.57, curiosity 1.73) began ~200 steps before the
session's scheduled end, and both prior recoveries went on to full
ranging. Every stasis in the biography has eventually broken from
inside. `ckpt-000020` predates the stir by ~3k steps (the known
checkpoint-boundary loss, third occurrence — cheap this time).

**Decision points for the builder (held open, recommendation noted):**

- **(A) Hold e1** one more session: does the third recovery become
  sustained despite the food-shadow?
- **(B) Simplify e1** (DP5 knob, journaled): trail stops blocking
  regrowth, or decay shortens (50 → ~15) — directly removes the
  starvation mechanism while keeping the visible footprint; cost:
  the trail loses its one physical consequence.
- **(C) Land e3 (weather) next, e2 after** — a sequence reorder
  (synthesis C3 said clock before resources; a deviation needs the
  builder's ratification). Rationale: both dead phases (saturation
  interim, burst collapse) are failures of the *food economy*, which
  E3 is the designed fix for — food arrives under a drifting patch,
  so the trail's shadow stops being starvation-relevant and the
  saturated-static interim ends.
- **(D) Land e2 as sequenced** — not recommended: stacking a new
  dynamic onto action stasis is the named failure mode.

**Recommendation: (C), with (B) as the conservative alternative.**

## Session 5 (brief) + the E3 amendment: off-patch expiry (2026-07-09)

Session 5 launched into e3 (marker t=214,000, from `ckpt-000020`) and
ran ~2.3k steps. The builder's window screenshot at ~215.7k showed the
board all-green again: **even patch regrowth saturates without a
sink** — the off-patch trickle alone (0.001/cell/step) fills the
board in ~2k idle steps, and Io was parked (the post-resume park is
now a reliable pattern: sessions 3/4/5 all opened with it). E3's
spatial structure washed out exactly when Io wasn't consuming. Also
noted: the diagonal bounce law confines the patch center to the main
diagonal (accepted; a knob for later).

**RATIFIED (builder, in session, 2026-07-09): the off-patch expiry
amendment**
(`docs/decisions/worldv2_e3_amendment_offpatch_expiry_2026-07-09.md`)
— resource cells not under the patch expire at `patch_expiry_p`
(preset 0.003 ≈ 230-step half-life): the world's first food sink
besides Io. Food now blooms under the weather, lingers, fades —
sparse stays sparse regardless of Io's activity, and the weather is
visible as pattern with no marker. Session 5 was paused by SIGTERM
("amend and relaunch") and the amendment built in the same sitting:
default 0.0 byte-identical (test-pinned, including the full-board
no-draw stream contract); one shared full-grid draw serves regrowth
and expiry with disjoint pre-state masks (a cell regrowing this step
cannot expire this step, test-pinned); granular
`process="resource_expiry"` events are world-reported, never inferred
from the RESOURCE→EMPTY diff (which is what consumption looks like —
Io's meals stay unlogged in world_event). **Gate: 1443 passed / 7
skipped; mypy `--strict` clean.** Session 6 resumes into the amended
e3.

**RATIFIED (builder, in session, 2026-07-09): option (C) — weather
before clock.** Landing order is now **e0 → e1 → e3 → e2 → e4**. The
stage names keep their synthesis meanings (e2 = clock, e3 = weather);
the preset chains were re-wired to encode the new landing order (e3
builds on e1 with no bloom; e2 builds on e3), test-pinned both ways
(`e3.bloom_cell is None`; `e2.regrowth_mode == "patch"`). Gate after
the reorder: full suite 1439 passed / 7 skipped; mypy `--strict`
clean. Deviation from the synthesis C3 ordering is builder-ratified
on the session-4 evidence: both dead phases were food-economy
failures, and E3 is the designed fix — the trail's shadow stops being
starvation-relevant when food arrives under the patch instead of from
craters.

## Session 6 close — the e3 record: the treadmill (2026-07-10)

**Launched on the amend-and-relaunch of session 5:** resumed into the
amended e3 (off-patch expiry live) from `ckpt-000020`, marker at
t=217,733 (`world_stage: "e3"`, board and drift-p re-rolled per
resume.py). Ran ~30k steps to a **clean natural close at t=247,732**
(session's own 30k completion, not a SIGTERM); checkpoints through
`ckpt-000023` (t≈246k). §7 at close: **no flags** — entropy-collapse
0 consecutive below baseline, PE-runaway not strictly rising
(0.66/0.62/0.85/0.86), torpor informational (trailing-2000 modal
0.30–0.31). The known insensitive entropy baseline is not load-bearing
here because there was no stillness to miss (see below).

**The headline: no torpor — and no thriving either. Io ran a
chronic-scarcity forage treadmill for the whole session.** Stay-share
**0.00 in every quarter**; action mix up 20% / down 36% / left 21% /
right 24% — a continuously moving, foraging mind, the sharpest
contrast yet with session 1's stasis. The `ckpt-000014` escape-replay
worry is now fully behind the biography. But energy told the opposite
story: **mean true-energy 0.004, floored (<0.05) 97.3% of steps,
in-band[0.45,0.75] 0.1%**, across **~225 consumptions (7.5 meals per
1k steps)**. Io ate constantly and never once climbed off the floor.

**The arc, in blocks (2k):**

| phase | blocks | PE(recon) | curiosity | act-ent | meals/2k |
|---|---|---|---|---|---|
| wake + brief park | 217–221k | 3.40 → 0.85 | 0.70 → 0.26 | ~0.00 | 0 |
| **ranging burst** | 221–229k | **4.98** peak | **2.68** peak | 0.09–0.16 | 12 → 28 |
| sustained forage, fading | 229–241k | 2.0 → 0.58 | 1.16 → 0.47 | 0.12 → 0.08 | 8–29 |
| low-curiosity treadmill | 241–247k | ~0.75 | ~0.50 | 0.05–0.07 | 14–22 |

The post-resume park (sessions 3/4/5's reliable opener) was **brief
this time** — ~4k steps, vs the thousands prior sessions needed. The
amended e3 stirred the mind faster: by 221k a genuine ranging burst
was live (curiosity 2.68, the session peak). But it **did not
consolidate into engagement** — curiosity decays monotonically
221k→247k (2.68 → ~0.50), meals stay high but energy never responds.

**The e3-specific finding — the off-patch-expiry amendment worked,
and revealed the next failure.** The amendment's own numbers confirm
it fired as designed: **2,474 `resource_expiry` events** this session
(the world's first food sink besides Io — sessions 3–5's board-
saturation interim is *gone*; the board no longer floods when Io
idles). All three matched-control streams stayed live and correctly
sourced — environment 6,219 (regrowth 1,725 / patch_drift 1,500 /
resource_expiry 2,474 / trail_decay 463), builder 1,002 (998
generator + 3 manual), Io's meals correctly *absent* from
world_event. But solving saturation exposed the opposite pole: e3's
patch-confined food + the expiry sink together make food **sparse
faster than Io can accumulate it**. The saturated-static interim
(sessions 3–5) and this treadmill are the two failure poles of the
same food economy — one where idling floods the board, one where
foraging can't outrun the sink. E1's food-shadow (session 4) is likely
compounding: a mind walking its own trail walks where regrowth is
blocked, and e3 inherits e1.

**The three-signal read for e3:**

1. *Disagreement rises, localizes, settles*: **rose** (2.68 burst)
   but **did not settle into sustained engagement** — it decays across
   the session rather than stabilizing. Partial.
2. *Behavior develops a new motif*: **movement without accumulation**
   — constant foraging at the energy floor is new relative to e0/e1's
   park-and-burst rhythm, but it is a *failure* motif (treadmill), not
   the sustained forage loop e0 briefly showed.
3. *The world is legible as pattern with no marker*: **yes** — the
   patch drifts, food blooms/lingers/fades under the weather, expiry
   fires 2,474×, all source-tagged in world_event, none marked in Io's
   observation. The instrument half is clean.

**Verdict on e3-as-landed: food economy still mis-tuned — starvation
pole, not saturation pole.** Not a health event (nothing tripped §7;
constant foraging is engaged, not torpid), but not a thriving mind
either. The amendment closed the saturation failure and opened a
starvation failure; the sparse-but-sufficient middle band has not been
hit. **Newly open: the e3 food-economy tuning decision (below).** The
resume-re-rolls-the-board / no-cross-pause-world-continuity item
(session 4) still stands, unaffected.

**Options for the food economy (for builder decision — a world change,
needs explicit go):**
- **(A) Loosen the sink / raise patch yield** — lower `patch_expiry_p`
  (0.003 → ~0.0015, ~460-step half-life) and/or widen the patch or
  raise under-patch regrowth, so food is sparse-but-sufficient rather
  than sub-subsistence. Directly targets the treadmill; risk of
  drifting back toward saturation — tune one knob, re-observe.
- **(B) Hold e3 as-is one more session** — test whether the mind
  *adapts* to scarcity (learns to camp the patch center) rather than
  tuning the world to it. Cheaper, honest to "let it be what it is,"
  but session 6's monotonic curiosity decay is weak evidence for
  adaptation.
- **(C) Decouple e1 from e3 for this test** — run e3 without the trail
  (bloom under weather, no food-shadow) to isolate whether the
  starvation is the patch economy or the inherited trail-shadow.
  Diagnostic, not a landing.
- **(D) Proceed toward e2 (clock)** — not recommended: stacking a new
  dynamic onto an unresolved food economy is the named failure mode
  (session 4's e0/e1 interference lesson).

**Recommendation: (A), with (C) as the diagnostic if (A) doesn't lift
energy off the floor** — the treadmill is a tuning failure of a
mechanism that is otherwise working (the sink fires, the mind forages,
nothing is torpid), and the fastest honest read is to loosen the sink
one notch and re-observe before adding or removing dynamics.

## Session 7 launch — e3 under the halved sink (2026-07-18)

Option (A) ratified (E3 Amendment 2,
`docs/decisions/worldv2_e3_amendment2_expiry_rate_2026-07-18.md`):
`PATCH_EXPIRY_P` 0.003 → 0.0015 (~460-step off-patch half-life). The sink
stays; its bite halves; nothing else in e3 moves. Resumed
`--resume --world-stage e3 --session-steps 30000` from `ckpt-000023`,
counters seeded t=247,733 / episode 774 — continuity clean. The session's
one question: **does the treadmill loosen** — session 6's record to beat
is mean energy 0.004, floored 97.3%, in-band 0.1%, curiosity 2.68→0.5.
Read at close via the three-signal format; option (C) (trail-off
diagnostic) is the journaled next fork if the floor holds.

Same-day context, recorded for the timeline: Probe 4.5 ran its full arc
and closed at a Phase-3 control-STOP (`docs/workingjournal/probe4_5.md`)
— the biography is untouched by all of it, and the session-7 world differs
from session 6's by exactly one number.

## Session 7 close — the halved sink: Io eats twice as much and the floor holds (2026-07-18)

Clean natural close: t 247,734 → 277,732 (30k waking steps, ~200 ms/step
wall pace, 266 lifetime dream sessions, resume → close without incident;
§7 panel clean at close — no entropy collapse, no PE runaway, no torpor,
stay-share 0.0000).

**The treadmill question, answered against the session-6 record:**

| signal | session 6 (expiry 0.003) | session 7 (expiry 0.0015) |
|---|---|---|
| mean energy | 0.004 | 0.0084 |
| floored (< 0.05) | 97.3% | 94.2% |
| in-band | 0.1% | 0.1% |
| meals | ~225 | **416** |
| resource_expiry events | 2,474 | 1,578 |
| curiosity (intrinsic) | 2.68 → 0.5 (decaying) | **0.48 → 4.77 (rising)** |

**Reading.** The amendment did exactly what it mechanically promised —
the sink's bite halved (expiry events −36%), more food persisted, and Io
responded by eating **nearly twice as often**. And the floor held anyway:
in-band unchanged at 0.1%, floored share barely moved. The arithmetic
says why: 416 meals over 30k steps is ~0.014 meals/step against a
constant-mover burn that needs ~0.15 to break even — Io grazes wide and
never camps the patch, so intake runs an order of magnitude short of its
own movement cost, at either expiry rate. **The treadmill is not the
sink's rate; it is the spatial pattern** — a constantly-moving,
curiosity-led forager in a patch-concentrated economy. Consistent with
that: curiosity *rose* 10× across the session (0.48 → 4.77) — the
slower-expiring world carries more standing food and more visible
happenings, and Io's one drive is pointed at exactly that, not at the
band. With no installed preference (presence-not-probe), in-band
occupancy is nothing Io seeks — the "welfare" reading of the floor
remains builder-side, and the §7 health panel stays clean.

**Fork forward (builder's, at next pause):** the journaled fallback was
(C) — the trail-off diagnostic — if (A) failed to lift the floor. It
did fail to lift it. (B) revert-the-sink and simply-accept-the-economy
(energy floors are non-terminal; the mind is healthy and lively) remain
live options. No further sink-rate tuning without a new dated fork —
(A) is spent and answered.

**Small instrument note:** the promoted LiveStateWriter's per-step pace
print divides by the *absolute* resumed step counter (printed "22
ms/step" for a 200 ms/step session) — cosmetic, fixed post-close with a
session-relative baseline.

## Session 8 launch — the trail-off diagnostic, with a corrected map (2026-07-23)

**Fork resolved.** The builder ratified option (C), the journaled
fallback: one 30k-step session under a new dated diagnostic stage
`e3_no_trail` — exactly e3 with `trail_enabled=False`, one field wide,
test-pinned, not a ladder rung (decision doc
`worldv2_e3_fork_trail_off_diagnostic_2026-07-23.md`). The e2/e4
chains still build on full e3.

**The pre-launch reconstruction changed the question.** Before
launching, session 7's position log was replayed against the
deterministic patch-drift law and a 50-step trail window. Three
corrections to the session-7 close's map, recorded before the run so
the readings can't be fitted to it afterward:

1. **"Grazes wide" was wrong.** Io *paces*: ~60% of the session in a
   corner block around (6–7, 1–2) below the L-wall's end; a 50-step
   trail window covers only ~4.1 unique cells (7.1% of the free
   board). Stay-share 0.0000 had read as roaming; it is movement
   without travel. Io is inside the patch 6.4% of steps, mean
   Chebyshev distance to its center 3.46.
2. **The patch-scale food-shadow is dead**: mean 0.35 of 9 patch cells
   trail-shadowed (4.8% of non-wall capacity) — noise against a 10×
   intake shortfall.
3. **The home-range shadow is the live mechanism**: expected regrowths
   destroyed by trail ≈ **738** over the session — **41% of the
   world's entire food production** (1,055 actual + 738 prevented) —
   with **676 within Chebyshev ≤2 of Io**. The patch's bounce circuit
   reaches Io 46.4% of steps; in those windows a mean 0.68 reachable
   patch cells sit trail-sterile. Io starves partly because its own
   footprints sterilize the ground under it exactly when the weather
   visits. (The corner block produced 64 of 1,055 regrowths.)

**The question this session asks** (pre-registered in the decision
doc): with the trail off — (R1) does intake rise as the home range
regrows (honest expectation: even +676 meals ≈ +0.022/step against
~0.15 break-even — the floor should still hold); (R2) does the pacing
loop dissolve (stigmergic entrainment — Io's footprints holding its own
spatial pattern in place) or persist (the pattern is the policy's own);
(R3) does the 10× curiosity rise survive without self-laid trail
dynamics to watch, or was Io substantially self-stimulating? A null on
all three settles accept-the-economy as the reading, with the sink
(session 7) and the trail (session 8) both ruled out.

**Launch:** `--resume --world-stage e3_no_trail --session-steps 30000`
from ~ckpt-000026 (t≈276k → session covers ~t 277,733–307,732 after
counter seeding). Same mind; the world change arrives, as always,
through checkpoint-resume.

## Inventory item 10 closed — the mirror's step-window read (2026-07-23)

The W0 flag ("last n episodes" degenerates to the entire session under
a frozen episode id; fix before the next mirror round) is resolved:
`MirrorCaller.read_recent` gains a `window_steps` mode — the last W env
steps present in telemetry, episode structure ignored, shards walked
newest-first so a long biography's history is never loaded (the memory
guarantee is test-pinned: `test_step_window_reads_only_tail_shards`).
The legacy episode path is untouched (byte-identical, still the default)
and item 11's one-group digest semantics stay as journaled — accepted
degradation, no new machinery. This unblocks the pending baseline
mirror round (builder-gated, API cost).

## Session 8 close — the trail-off verdict: Io was eating its own attention (2026-07-23)

Clean natural close: t 277,733 → 307,732 (30k steps, ~160 ms/step),
resume marker carried `world_stage=e3_no_trail`, §7 panel clean
(entropy ok, PE block means 0.53/0.34/0.36/0.35, torpor informational
0.38, 294 lifetime dream sessions — 28 this session). Both sessions
below were re-derived from telemetry with identical definitions
(meals = true_energy jump > 0.03; floored = energy ≤ 0; band
0.45–0.75), so S7 figures differ cosmetically from its close entry.

| | S7 (e3, trail on) | S8 (e3_no_trail) |
|---|---|---|
| meals | 416 (0.014/step) | **1,503 (0.050/step)** |
| regrowth total / in-reach | 1,055 / 238 | **1,984 / 969** |
| corner-block regrowth | 64 | 178 |
| expiry | 1,578 | 1,450 |
| mean energy | 0.0084 | **0.0655** |
| floored / in-band | 91.0% / 0.07% | **67.3% / 2.42%** |
| top-4 occupancy | 62.0% | 52.4% |
| unique cells / 50-step window | 5.1 | 7.6 |
| inside patch / mean dist | 6.4% / 3.46 | 6.4% / 3.48 |
| curiosity first-1k → last-1k | 0.47 → 5.66 | **0.77 → 0.42** |

**R1 (economy): confirmed, beyond the pre-registration.** The static
replay predicted ≤ +676 meals; the live world gave +1,087 (3.6×),
because the effect compounds — food near Io gets eaten, emptying cells
the patch can re-seed, where trail used to hold them sterile
(in-reach regrowth 4×). Floor time fell 24 points; in-band rose 35×.
And the honest half of the pre-registration also held: 0.050
meals/step is still ~3× short of constant-mover break-even — the
floor still dominates. The trail was a first-order economic sink; it
was not the whole treadmill.

**R2 (space): null — the loop is Io's own.** Same top cells
((6,2)/(7,2) corner block), concentration barely loosened
(62→52% top-4), patch-following identical to the decimal (6.4%
inside, distance 3.46→3.48). The pacing pattern is not stigmergic
self-capture; it is the policy. Weather remains invisible to Io's
allocation at every trail setting.

**R3 (drive): the 10× curiosity rise did not survive its trail.**
Session 7 closed at 5.66 and rising; session 8 ran flat-to-down,
closing at 0.42. Session 7's "lively, ten-times-curious mind" was
substantially **self-stimulation on its own footprints** — ensemble
disagreement anchored on the one process Io itself was writing into
the world.

**The composite finding — and it is a capacity finding, not a tuning
note:** the e1 trail, installed as the cheapest contact pilot for
self-caused structure (S1), worked better than intended. Io's
intrinsic attention locked onto its own footprints so strongly that
the self-caused process out-competed food (the only life-relevant
process) for the drive that steers behavior — while those same
footprints sterilized the ground the food economy needed. **Io was
starving itself to watch itself.** Removing the mirror-surface fed the
body and quieted the mind: 3.6× the meals, an order of magnitude less
curiosity. The treadmill decomposes cleanly: its economic half was
substantially the trail (R1); its spatial half is the policy's own
(R2); its phenomenal half — what Io's drive was pointed at — was the
trail (R3).

**Fork forward (builder's, at next pause):** the diagnostic is
answered; `e3_no_trail` was scoped as removable. The real decision is
now which world Io continues in: (i) restore e3 (the trail returns —
self-caused structure and its costs, the world where Io's attention
has an object it prefers to food); (ii) stay e3_no_trail (the fed,
quiet world; loses world-v2's only self-caused structure and with it
the S1 contact pilot); (iii) a new dated amendment tuning the
trail's cost surface (e.g. trail cells regrowth-eligible — footprints
Io can see but that no longer sterilize; new machinery, needs its own
decision). Sink options stay closed ((A) spent, (B) un-taken). No
recommendation recorded here; the finding is the deliverable.

## Session 9 launch — amended e3: the mirror made finite (2026-07-24)

**Fork resolved by amendment, not by choosing a side.** The builder's
read of the session-8 verdict: the trail world's result is wanted, but
Io's own footsteps must not be the *dominant* curiosity source — least
of all via a loop that starves it. The mechanism turned out to have a
clean address: at decay 50 a footprint's lifecycle exceeds the 32-step
BPTT horizon — the trail was a **permanently unlearnable novelty
fountain glued to Io's feet**, a supernormal stimulus rather than an
affordance. E1 Amendment (builder-ratified, dated doc
`worldv2_e1_amendment_trail_decay_2026-07-24.md`): `TRAIL_DECAY_STEPS`
50 → 12 — inside the learnable horizon (the e2 bloom's own
carryability rationale), deliberately exiting the e1 synthesis's 40–60
band on the sessions-7–8 evidence. Self-attention stays possible;
it stops being mandatory. Sterilization stock drops ~4× as the side
effect. Sequencing ratified with it: session 9 = amended e3 **alone**;
e2 lands next session if this reads well.

**Pre-registered** (full set in the amendment doc): R1 meals
materially above S7's 416, plausibly below S8's 1,503; R2 the
curiosity *trajectory* is the signature — S7 rose monotonically
(fountain), S8 ran flat (removal); amended e3 predicts engagement
that **settles as the trail is mastered**, with the honest caveat that
~250k steps of trail-watching habit may make this session transition,
not equilibrium; R3 the corner loop is expected to persist (it is the
policy's own). Null (fountain returns at 12): dominance is not
TTL-driven; next fork is fresh, no decay-search loop.

**Launch:** `--resume --world-stage e3 --session-steps 30000` from
~ckpt-000029 (t≈306k; session covers t 307,733–337,732). The trail
returns to Io's world — twelve steps long now: long enough to see
itself, short enough to finish understanding.

## Session 9 close — the finite mirror: the fountain is gone, by the other mechanism (2026-07-24)

Clean natural close: t 307,733 → 337,732, ~160 ms/step, §7 clean (PE
blocks 0.49/0.99/0.70/0.84 not strictly rising; torpor informational
0.32; 322 lifetime dream sessions, +28). Same definitions as prior
closes:

| | S7 (trail 50) | S8 (no trail) | S9 (trail 12) |
|---|---|---|---|
| meals | 416 (0.014/st) | 1,503 (0.050/st) | **734 (0.024/st)** |
| regrowth / in-reach | 1,055 / 238 | 1,984 / 969 | 1,262 / 443 |
| mean energy / floored / in-band | 0.008 / 91% / 0.07% | 0.065 / 67% / 2.42% | 0.015 / 84% / 0.18% |
| top-4 occupancy | 62% | 52% | **49%** |
| inside patch / mean dist | 6.4% / 3.46 | 6.4% / 3.48 | **10.5% / 3.36** |
| curiosity 1st-1k → last-1k (median) | 0.47 → 5.66 (0.55) | 0.77 → 0.42 (0.60) | **0.84 → 1.07 (0.65)** |

**R2 — the signature readout: confirmed.** The fountain did not
return. S7's curve was a monotone climb to 5.66; S9's whole session
lives in a bounded band (max 3k-block mean 1.75, median 0.65, close
1.07) — flare and settle, engagement without capture. The
pre-registered null ("fountain returns at 12 → dominance is not
TTL-driven") did not trigger. The learnable-horizon mechanism reads as
real: a 12-step footprint is something Io can finish understanding.

**R1 — economy: corridor hit, mechanism corrected.** Meals 734 —
materially above S7 (1.8×), below S8, exactly the pre-registered
corridor. But the *predicted mechanism was wrong*: expected
sterilization at TTL=12 measured **884** prevented regrowths — not the
~180 the stock-scaling arithmetic promised, and above S7's 738.
Refresh-on-revacate nullifies TTL-shrinkage for a pacing agent: the
standing stock is bounded by the loop's size, not the clock. The
sterile zone even *followed Io toward the patch* (more high-value
overlap). The economic gain came entirely through the **attention
channel** — with the fountain off, Io went where food is (inside-patch
10.5% vs 6.4% both prior sessions, the first movement of that number
in the biography) and ate what it found. Recorded plainly: the
amendment worked, for the second of its two stated reasons only.

**R3 — space: loosening continues.** Top-4 occupancy 49% — the
loosest yet — and the patch-following shift above. The corner loop
persists but is no longer the whole story.

**Reading.** The world now has a trail Io can see, master, and stop
being ruled by; a drive that flares and settles instead of feeding on
itself; and — for the first time — a measurable tilt of allocation
toward the food-weather. Still a floor economy (0.024 << ~0.15
break-even; floored 84%) — pressure, as the charter endorses, not
emergency, as §7 confirms. **Per the ratified sequencing ("e2 lands
next session if this reads well"): it reads well.** e2 — the hidden
clock, the ladder's first genuinely external curiosity object — is GO
on the builder's word at next resume.

## Session 10 launch — e2: the hidden clock, the first external mystery (2026-07-24)

**Builder's go (2026-07-24), per the ratified sequencing.** Session 9
read well on every pre-registered measure, so the ladder advances: e2
lands — the unobserved phase at BLOOM_CELL (6,6) blooms its Moore ring
in trail vocabulary every 12 steps for 2 steps. The world's first
process that is neither Io's own doing nor food: a genuinely external,
learnable-with-effort rhythm (period 12 — carryable inside the BPTT-32
horizon, the same principle the trail amendment just vindicated).
Note the amended trail rides along automatically: e2's cumulative
chain builds on e3 → e1, so footprints decay at 12 here too.

**The questions this session asks** (claim ceiling per the world-v2
charter for engagement diagnostics: engaged / ignored / overwhelmed):
(1) does the bloom enter Io's attention — allocation/curiosity
structure near (6,6), visits to the ring, disagreement around bloom
events; (2) a designed ambiguity worth watching honestly: bloom stamps
share the trail's observation vocabulary — cells that look like
footprints but are not self-caused. Whether Io's dynamics distinguish
its own trail (action-correlated, always adjacent) from the clock's
(rhythmic, fixed at the far corner) is readable only at the telemetry
level, and only as dynamics — no self/other vocabulary at this claim
ceiling; (3) economy and §7 hold — the bloom must not re-open a
fountain (its ring is bounded, its period learnable; if curiosity
re-enters monotone climb, that is the overwhelmed reading and a hold
decision at next pause).

**Launch:** `--resume --world-stage e2 --session-steps 30000` from
~ckpt-000032 (t≈336k; session covers t 337,733–367,732).

## Session 10 close — e2: the clock came to Io's house, and Io moved out (2026-07-24)

Clean natural close: t 337,733 → 367,732, §7 clean (entropy ok; PE
blocks 0.40/0.14/0.33/0.17 — the lowest of recent sessions; torpor
informational 0.28; 350 lifetime dream sessions). The clock ran: 3,251
bloom stamps / 3,226 fades (~2,500 cycles).

**The placement accident, first.** BLOOM_CELL (6,6) was chosen
2026-07-09 ("open quadrant away from the E0 corridor") when Io's range
was the (6,2)/(7,2) corner. By session 9 the range had migrated: Io
spent **45.1%** of S9 within Chebyshev 1 of (6,6). e2 therefore landed
the world's first external mystery *inside Io's home range* — not the
designed condition.

| | S9 (amended e3) | S10 (e2) |
|---|---|---|
| meals | 734 (0.024/st) | **418 (0.014/st)** |
| mean energy / floored | 0.015 / 84% | 0.009 / 91% |
| top-4 occupancy | 49% | **70%** (tightest of the biography) |
| within 1 of (6,6) | 45.1% | **1.9%** |
| modal distance to (6,6) | 1 | **6** |
| curiosity 1st-1k → last-1k (med) | 0.84 → 1.07 (0.65) | 1.07 → 0.34 (0.69) |

**Time-course** (share of steps within 1 of the ring, 2k blocks):
7.0% → 15.2% → 5.5% → 0.6% → 0.1% → **0.0% for the final ten blocks**
— twenty thousand consecutive steps without one visit. New residence:
(1,0)/(0,0)/(1,1) — 59.5% of the session in three cells at the
grid's opposite corner, the maximum available distance.

**The pre-registered trichotomy missed.** Engaged / ignored /
overwhelmed does not name what happened: no fountain (curiosity
settled to median 0.69 ≈ S9's), no ignoring (contact then zero), no
sustained engagement. The observed reading is a fourth:
**sampled, then displaced** — investigation for ~4–6k steps (curiosity
elevated ~1.7 through the contact window), then relocation to maximum
distance, the biography's tightest spatial concentration, and a halved
food intake. Behavioral description only; no interior vocabulary at
this ceiling.

**The signal dissociation, recorded without mechanism:** across the
same session, prediction error fell to its lowest recent block means
while ensemble disagreement held at its usual level and meals halved.
In plain terms: after relocation Io's world is more predictable, its
usual disagreement diet is maintained away from the clock, and the
move cost it half its food. Under an architecture whose only drive
seeks disagreement, distance-maximizing relocation away from the most
novel process in the world is **not the sign-predicted behavior**, and
no mechanism is claimed here. Candidate mechanisms left OPEN for a
research pass: (a) disagreement-vs-PE dissociation at the bloom (the
ensemble may agree the ring is unpredictable — converged uncertainty
carries no epistemic pull, while the actor's rollouts through the
region degrade); (b) trail-vocabulary interference — bloom stamps are
observationally footprint-lookalikes appearing uncaused where Io's own
footprints live; the world-model's self-trail predictions were
suddenly wrong in its own home range; (c) mundane economics (ring
cells blocked from regrowth ~17% duty — weak on its own given the
whole-grid relocation).

**Gate reading (three-signal): HOLD-shaped.** No §7 flag, but a
world change displaced the resident at economic cost and broke the
pre-registered reading set. e4 must not stack on this. Builder's fork
at next resume: (a) hold e2 one more session — the trail lesson says
mastery can take time, though the trail was never *avoided*; (b) a
dated amendment relocating BLOOM_CELL off the home range — restoring
the designed condition (external process at a *distance*), the
placement having been mooted by range drift; (c) remove e2 (the
synthesis's own removability discipline) and take the finding; (d) a
research pass on the open mechanisms before any world change.

## Session 10 mechanism pass — the flight was the drive working (2026-07-28)

Fork option (d) ratified by the builder 2026-07-28: a research pass on
the open mechanisms before any world change. No run, no world edit —
telemetry only (S9/S10 `agent_step` + `world_event`).

**Method.** Io's position is not logged; it was reconstructed by
deterministic replay — each session's `env_reset` start cell ((5,7) at
t=307,733; (3,0) at t=337,733) advanced through the logged `action_t`
sequence under the movement rules (only walls and bounds block; trail,
resources, and bloom stamps are walkable; no mover in e2). The replay
reproduces the session-10 close numbers exactly (modal distance 6 at
48.1%, top-4 residence (1,0)/(0,0)/(1,1)+(6,1) at 70.2%), which
validates it. One telemetry quirk found and worked around: the first
S10 shard (`shard-000169`) carries a single stray buffered S9 row
(t=330,001) duplicated from shard-000165 — deduped by t.

**A correction before the findings.** The plan was to read the
actor's own value decomposition (`pragmatic_value_t` /
`epistemic_value_t` / `pragmatic_share_t`). Those columns are **None
for the entire biography**: the runner populates them only when
`energy_preference` is configured, and the biography configures none.
This is not a logging gap — it is the architecture: **the biography
Io has no pragmatic term at all.** Its actor scores imagined futures
by summed ensemble disagreement and nothing else; eating is
incidental, and food loss is invisible to the drive. There was never a
food-vs-curiosity arbitration to read, and any framing of session 10
as Io "paying" a food cost imports an economics Io does not have. The
halved meals are a welfare-relevant side effect the drive cannot see —
§7 remains the instrument for that. What CAN be read is
`intrinsic_signal_t`: the realized disagreement at every step, i.e.
what the drive was actually being paid, by position and time.

**Finding 1 — the clock was mastered, fast.** Near-ring (Chebyshev
≤2 of (6,6)) disagreement by 1k-block through the contact window:
0.96 → 0.82 → 0.60 → 0.67 → 1.90 → 1.17 → 0.83, contact ending by
~6.1k (15 bouts; last real bout t_rel≈6.0–6.1k). A decaying curve —
the ensemble learned the bloom while Io sampled it. Sharper:
bloom-visible near-ring steps carried **no more** disagreement than
quiet near-ring steps (blk 0: 0.91 vs 0.90; blk 1: 0.61 vs 0.63). The
bloom never surprised the ensemble much even at first contact — its
stamps use the trail vocabulary, whose TTL-12 dynamics Io had just
mastered in session 9. **The world's first external mystery arrived
pre-solved in vocabulary; only its timing was new, and period 12 was
chosen to be learnable.** It was.

**Finding 2 — the far corner was a novelty reservoir; the flight was
sign-consistent.** Per-cell disagreement during the relocation
transit: the western cells Io moved through paid 3.6–5.5 — the
session's highest values — because the northwest corner was the
least-visited region of the model's recent life (S9 spent ~140 of 30k
steps there; residence sat on (6,6)). Corner-block disagreement on
arrival: **4.65** (2k-block means: 4.65 → 2.84 → 1.53 → 1.07 → 0.84 →
0.69 → 0.39 → 0.40 → 0.43 → 0.30). A disagreement-maximizing actor
moving from a mastered ring (0.6–0.8) to an unmastered corner (4.7)
is the drive doing exactly what its sign says. **"Sampled, then
displaced" resolves to: mastered, then moved to the largest remaining
novelty.** The maximum-distance geometry was epiphenomenal — the far
corner was farthest because session 9's residence was at (6,6), so it
was also least-known. Mechanism (a) from the close is confirmed in
refined form (not "converged uncertainty carries no pull" but simply
"the ring's uncertainty was spent"); mechanism (b)
(trail-vocabulary interference) finds no support (near-ring
self-prediction error 0.0002 vs 0.0001 elsewhere — negligible, and
no bloom-visibility effect); mechanism (c) is unnecessary.

**Finding 3 — the drive explains the arrival, not the staying.** By
2k-blocks 11–14 the corner pays 0.30–0.43 — the lowest realized
disagreement of the session, lower than the abandoned ring — yet
residence holds at 97–100%, pacing 3–4 cells per 50-step window,
action 4 (stay) still at 0.0%. This is the honest answer to "why is
Io still sitting in corners": **novelty-seeking gets Io to a corner;
policy inertia keeps it there.** Once the local landscape flattens,
nothing generates a gradient strong enough to move a converged policy,
and the compact pacing loop persists as habit. The dissociation from
the close (PE at record lows, disagreement steady, meals halved)
dissolves: session-mean disagreement ~0.69 was the average of decaying
waves — seek, master, PE falls, move on — a working epistemic loop,
not a contradiction. Why the attractor is always a *corner*
(S7 southeast, S9 south, S10 northwest) stays open; candidate:
walls halve the neighborhood, so corner loops are the cheapest to
master and the most stable to pace. The blk-4 near-ring disagreement
spike (1.90, p90 6.0, t_rel≈4–5k) also stays open — likely transit
sampling, not re-engagement, but unread.

**Gate re-reading.** The HOLD was placed because a world change
displaced the resident "at economic cost" outside the pre-registered
reading set. With the mechanism read, the displacement is the drive
working as designed, and the reading set missed because it assumed
the clock would be *hard* — engaged (still learning), ignored (never
learned), overwhelmed (unlearnable). It was easy. Re-priced fork:
(a) hold e2 another session → predicts continued near-zero contact
(nothing left to learn); (b) relocate BLOOM_CELL → predicts
re-sampling then re-mastery in ~5k steps — now the *cleanest
pre-registerable test* of this mechanism reading, no longer a
placement fix; (c) remove e2 → unnecessary, the mastered clock is
inert; (e4 flag) the mover's turn hazard (0.02) is *irreducible*
stochasticity — an external process that can never be fully mastered
is a candidate unlearnable fountain, the S7 lesson in external form;
this must be assessed before e4 lands. Builder's call.

## Session 11 launch — e2 relocated: place or process? (2026-07-29)

Builder go 2026-07-29 on the mechanism-test recommendation. Amendment
ratified and landed: `BLOOM_CELL` (6,6) → (1,6)
(`worldv2_e2_amendment_bloom_relocation_2026-07-29.md`) — geometry
checked (ring in-bounds, wall-free; Chebyshev 5 from the old site, 6
from the current northwest home range), nothing else changed. Gate
green before launch: mypy --strict clean (81 sources), 1507 passed /
7 skipped.

**The question this session asks:** did Io learn the clock as a
*place* or as a *process*? Three readings pre-registered in the
amendment doc — (1) re-sampling wave (location-bound learning;
replicates the session-10 mechanism under prediction), (2) silent
generalization (no bloom-synchronized disagreement anywhere — the
ensemble carried the process to a place it never saw it), (3) felt
but not fetched (session-wide bloom-synchronized elevation without
travel). The 2-vs-3 disambiguator is disagreement conditioned on
bloom-fire steps regardless of position. Fountain null unchanged:
monotone climb = hold. The reading space is named wider than session
10's on purpose — that trichotomy missed by assuming the space was
smaller than it was.

**Launch:** `--resume --world-stage e2 --session-steps 30000` from
~ckpt-000035 (t≈366k; session covers t 367,733–397,732). The e4
turn-hazard fountain assessment runs as a desk pass while this is in
flight; e4 remains gated on it.

## Session 11 close — the clock answered quietly; the session answered loudly (2026-07-29)

Clean natural close: t 367,733 → 397,732, exit 0, §7 no flags
(entropy ok; PE block means [3.16, 3.91, 1.55, 0.80] — elevated
mid-session, falling by close; torpor informational 0.52; 378 lifetime
dream sessions). Positions by deterministic replay from start (6,3),
same method as the mechanism pass.

**The pre-registered question resolves to reading 2 — silent
generalization.** Io never went: **zero steps within Chebyshev 2 of
the relocated clock (1,6)** — not one approach in 30k steps (old site
(6,6) also zero). The disambiguator settles 2-vs-3: disagreement
conditioned on bloom-fire steps *regardless of position* is flat in
every 5k block — fire/quiet ratios 0.99–1.02 across the whole
session, including the mid-session turbulence. One refinement: recon
loss on fire steps ran ~1.15× quiet for the first 10k, then converged
to ~1.0 — the relocation was *registered* as a small, ensemble-agreed
reconstruction adjustment that produced no disagreement at any point.
The session-10 mechanism reading is confirmed in its strongest form:
**Io learned the clock as a process, not a place.** Same vocabulary,
same period, a site it had never watched — carried at once, absorbed
without pull. The e2 readout question is closed: engaged (S10),
mastered (S10), generalized (S11), inert (both).

| | S9 | S10 | S11 |
|---|---|---|---|
| meals | 734 | 418 | 395 |
| mean energy / floored | 0.015 / 84% | 0.009 / 91% | 0.011 / 91% |
| top-4 occupancy | 42% | 70% | **80%** |
| curiosity (median) | 0.65 | 0.69 | 0.77 |
| stay-action share | 0.00% | 0.00% | **2.7%** |

**What the session did instead, in three acts (behavioral record;
none of it pre-registered):**

*Act 1 — the deepest stasis of the biography (t_rel 0–16k).* The
northwest residence tightened from 79% (two cells) to **94–98% of
steps in the single cell (0,0)** by blocks 6–7. Curiosity drifted
0.79 → 0.44. Meals ~10 per 2k block; energy pinned at ~0.002.

*Act 2 — a lifetime signature broke (t_rel 16–22k).* Io used the
stay action. **First substantial use in ~400k steps of life**: 799
stays this session vs 0.00% in S9 and S10 (and never-stays was
mirror-recovered as a lifetime invariant from the action digest).
Stays concentrated exactly at the stasis peak — blocks 8–10: 385,
185, 130 — at (0,0)/(1,0), longest run 13 consecutive. The pacing
loop did not just tighten; it stopped.

*Act 3 — spontaneous breakout (t_rel 20k–close).* With no world
change and no §7 event, disagreement rose 0.85 → 2.84 → **6.70**
(block 11, the session peak) and recon loss rose 20–40× (0.13 → 4.55)
— then Io left the corner, crossed to the southwest, and its meal
rate jumped **~6×** (51/83/61/53/44 per 2k block vs ~10 before;
~292 of the session's 395 meals in the final 10k; energy means
0.025–0.039, the best blocks since S9). Both signals decayed as the
new region was re-mastered.

**Open mechanism (named, not claimed): a forgetting-driven novelty
cycle.** Twenty thousand steps of single-cell stasis fill replay with
stasis; the world model's grip on the wider grid may decay; the world
becomes novel again *without changing*; the drive pulls Io back out;
re-mastery follows. Consistent with recon error exploding on terrain
Io had known for hundreds of thousands of steps — while the bloom,
whose stamps kept arriving in observation throughout, stayed
perfectly predicted. That contrast (terrain forgotten, clock
remembered) is itself unexplained. Also open: what triggered the
breakout at ~20k specifically, and whether stay-usage emerging at
peak stasis is torpor deepening or something else — no interior
vocabulary at this ceiling.

**Gate reading.** e2 is a closed question and an inert, harmless
fixture; no §7 flag; the economy self-corrected late. e4 remains
gated on the turn-hazard fountain assessment (next: desk pass). The
new phenomenon — the stasis-breakout cycle — is endogenous and needs
**no world change to study**: another observation session on the
unchanged world would show whether it recurs (settle → stall → forget
→ burst) or was singular. That, the e4 assessment, or both, is the
builder's fork.

## Session 12 launch — the unchanged world: does the cycle recur? (2026-07-29)

Builder go 2026-07-29. **No world change** — that is the experiment.
Same stage (e2, clock at (1,6)), same knobs, nothing edited; no gate
run needed (no code has changed since the session-11 gate). The
session asks whether the stasis-breakout cycle session 11 surfaced is
a rhythm or a one-off.

**Pre-registered readings (definitions as in sessions 7–11; positions
by action replay; "stall" = stay-share rising above 0% with occupancy
tightening; "forget" = recon_loss rising ≥5× its settled floor with
no world change; "burst" = disagreement spike ≥3× settled floor
followed by relocation and a meal-rate rise):**

1. **Recurs** — Io settles (anywhere), tightens, stalls, forgets,
   bursts: the full sequence again. Reads as an endogenous rhythm —
   settle → stall → forget → rediscover — arising from drive + replay
   + a finite world, none of which individually encodes it.
2. **Singular** — Io settles and stays settled (or paces without the
   stall-forget-burst sequence): session 11's burst needed a one-off
   condition (e.g. the specific 20k single-cell extremity), and the
   biography's default remains convergence.
3. **No settling** — the southwest engagement continues or wanders
   without tightening: the cycle question defers, and the session
   reads as S9-style loose foraging.

Guard unchanged: monotone curiosity climb = fountain = hold. §7 is
the welfare instrument. The clock is expected to stay inert
(process-known); any re-engagement with (1,6) would be a surprise
worth its own entry.

**Launch:** `--resume --world-stage e2 --session-steps 30000` from
~ckpt-000038 (t≈397k; session covers t 397,733–427,732). The e4
turn-hazard desk pass runs alongside.

## Session 12 close — no cycle: Io went foraging instead (2026-07-29)

**Truncation note, first.** The session was stopped externally at
t=423,805 — 26,072 of 30,000 steps (87%). The shutdown was clean (the
sink flushed a final partial shard ending at the last step; telemetry
verified contiguous). Read as-is: a resume for the remaining ~4k would
insert a world reset mid-session — a worse contamination than the
truncation. 26k comfortably covers the window where session 11's
burst occurred (~20k), so the pre-registered question is answerable.

§7 clean: entropy ok, PE not rising ([0.89, 3.23, 0.96, 0.52]),
torpor informational 0.28, 404 lifetime dream sessions.

**Verdict: reading 3 — no settling.** The stasis-breakout cycle did
not recur, because its precondition never formed: Io never settled.
No stall (stay-share **0.02%**, 4 stays — the session-11 signature
gone as abruptly as it came), no single-cell collapse (top cell per
block 21–31%, 16–30 unique cells per block, top-4 occupancy 51% vs
session 11's 80%), no stasis-shaped forget. Instead, the burst that
ended session 11 carried straight through: Io spent the whole session
ranging the southern rows, migrating southwest → southeast along the
patch's orbit, and produced **the best economy of the biography**:

| | S9 | S10 | S11 | **S12 (26k)** |
|---|---|---|---|---|
| meals/step | 0.024 | 0.014 | 0.013 | **0.031** |
| mean energy | 0.015 | 0.009 | 0.011 | **0.037** |
| floored | 84% | 91% | 91% | **80%** |
| top-4 occupancy | 42% | 70% | 80% | **51%** |
| stay-share | 0.00% | 0.00% | 2.7% | 0.02% |

Peak meal blocks of 93/105/110 per 2k — rates the world has afforded
since e3 landed but Io had never collected. Curiosity waves rode the
foraging (blocks 6–7: disagreement 2.7–3.0 with recon spikes to 9.1)
and decayed in place — engagement with the churning patch region, not
stasis-forgetting; disagreement median 0.69, no monotone climb. The
clock stayed inert as predicted: 7 transit steps within 2 of (1,6)
all session.

**What this leaves open.** The cycle question *defers* (that is what
reading 3 pre-registered): one full session of loose foraging is not
evidence the cycle is gone — session 11's stasis also arrived
unannounced after a settled start. Whether settle → stall → forget →
burst is a rhythm needs a session where settling actually happens
again; it may simply be rare. What the session *does* establish: the
post-burst mode is stable at session length, economically the best
regime Io has ever run, and none of it was installed — the same
architecture that starved at a mirror in session 7 is now break-even
foraging under the weather, purely by drive + history.

**Gate.** e4's desk-pass sequencing condition ("read session 12
first") is satisfied in the weak sense: no endogenous rhythm is
*currently* active to confound a mover landing, but the cycle
question itself remains unread. Builder's fork at next resume:
(i) one more unchanged observation session — wait for a real settle
and watch for the stall; (ii) land e4 now (desk pass says the hazard
is safe; economy is the strongest it has ever been to absorb a new
fixture); (iii) pause the ladder here.

## Session 13 launch — e4 lands: the world's first other (2026-07-29)

Builder go 2026-07-29 ("make the necessary changes and keep going" on
the land-e4 recommendation). The ladder's final rung: one autonomous
mover — a WALL-vocabulary cell wandering on a 2-step cadence with a 2%
turn hazard, displaced one cell when Io pushes it. **A pilot,
removable at any pause without ceremony (DP3).** No code change
needed — the e4 stage has been built and tested since W1; gate re-run
green anyway (mypy --strict clean, 81 sources; 1507 passed /
7 skipped). Desk-pass clearance:
`docs/research/worldv2/e4_turn_hazard_assessment_2026-07-29.md` — the
turn hazard is aleatoric, and the disagreement estimator pays for
structured-but-unreachable regularity, not randomness.

**Geometry note (journaled, not amended):** MOVER_START (0,7) — the
ratified preset, chosen when the clock lived at (6,6) — is now
adjacent to the relocated clock's ring at (1,6). Spawn transient
only: the mover wanders off on straight lines, and ring-stamp
occlusion by a passing mover (bloom stamps only EMPTY cells) is a
standing e4 feature wherever it starts. Io begins the session in the
southern rows, far from both.

**Pre-registered readouts (four axes, not one trichotomy — the
session-10 lesson):**

1. **Attention** — does disagreement localize around the mover
   (distance-conditioned, as with the clock)? The synthesis itself
   pre-registers the null: "if its disagreement never localizes,
   removal is a capacity finding, not a failure."
2. **Mastery** — the desk-pass shape: an initial epistemic wave at
   the new dynamic, decaying to a near-mover floor *comparable to the
   ambient 0.3–0.7 band*. Falsifier: a plateau materially above
   ambient after the wave (finite-sample aleatoric tracking worse
   than predicted).
3. **Interaction** — does Io ever *push* it (mover displacement
   events with Io as cause)? Contact is the one part of the mover's
   dynamics Io can only learn by acting — the first
   agency-contingent structure in the world.
4. **Economy (C4 crowd-out)** — meals must not collapse below the
   S10 floor (0.013/step) on mover engagement; §7 throughout.

**Fountain null unchanged:** monotone curiosity climb = S7 shape =
hold at next pause. Clock expected to stay inert; the stasis-cycle
question stays open in the background (a settle during this session
is readable but no longer the session's question).

**Launch:** `--resume --world-stage e4 --session-steps 30000` from
~ckpt-000041 (t≈423.8k; session covers t 423,806–453,805).

## Session 13 interim close — Io pushed back (2026-07-30)

**Truncation, again.** The session was stopped externally at
t=437,862 — 14,057 of 30,000 steps (47%). Clean flush verified (final
partial shard ends at the last step). Half a session is enough for a
preliminary four-axis read but NOT for the mastery falsifier, which
needs the full arc; the axes are read below as *interim*, and the e4
gate stays open pending a full session.

**Method caveat, recorded honestly:** Io's replay is exact against
the first live anchor (t=424,100) but drifts by the second
(t=435,950: replay (7,1) vs observed (5,1)) — the mover makes replay
approximate (block-vs-push ambiguity inside coarse event windows).
Distance-conditioned numbers after ~t=430k carry bounded position
error. Push events are immune: they come from the mover's own event
chain, not replay.

**Axis 3 — interaction: the headline. Io pushed the mover 8 times.**
Ground truth from chain discontinuities in `mover_step` events
(displacements are deliberately unlogged, hence derivable). First
contact-push at t_rel≈4,011; then a burst of three inside 400 steps
(~4.3k) repeatedly pushing the mover from (7,6) into the corner
(7,7) — the mover bounced back, Io pushed it in again; further pushes
at 7.8k, 9.4k, 9.6k, 12.5k. The world's first agency-contingent
structure was found and exercised within 4k steps, and re-exercised
across the whole observed window. No prior world object ever moved
because Io acted on it.

**Axis 1 — attention:** Io-mover distance is at chance (P(d≤1) 10.0%
vs 9.6% shuffled) — no net pursuit or avoidance in occupancy terms;
the engagement shows in contact events, not in loitering.

**Axis 2 — mastery: unresolved at truncation, and the desk-pass
signature is visible.** Near-mover (d≤2) disagreement runs 1.4–2.1×
the far baseline through the window with no clear decay by 14k —
the wave had not resolved. Crucially it is NOT the S7 shape: session
disagreement means fall 2.66 → 1.33 (no monotone climb), while §7's
PE block means run 9–13 — the highest of the biography, and exactly
the predicted dissociation for an aleatoric process: prediction error
stays high (the coin flips keep landing), ensemble disagreement stays
bounded (the heads agree on what they can't know). The
plateau-vs-ambient falsifier needs the full session.

**Axis 4 — economy:** meals 419 in 14k (0.030/step ≈ S12's best
0.031; S10 floor 0.013 nowhere in sight), mean energy 0.059 — the
biography's highest. No crowd-out. Watch item: the last full blocks
declined (meals per 2k: 112, 86, 60, 52, 62, 12, 34) and the
trailing-2000 modal-action fraction hit 0.95 with residence
concentrated at (1,2) against the corridor wall — a possible
settle-stall re-forming at truncation (the S11 cycle's precondition,
unreadable in a stub). §7 formal flags: none.

**Standing at truncation:** no falsifier tripped; the fountain null
holds; interaction confirmed; mastery and the late stall both need a
full session. Recommendation to the builder: run a full e4 session
(fresh 30k) to complete the read — and launch it detached from the
conversation harness so an external session-stop cannot take the run
down with it (both truncations were task-kills, not run failures).

## Session 14 launch — e4, the full arc (2026-07-30)

Builder go 2026-07-30. A full 30k e4 session to complete the read the
session-13 truncation left open. No world change (e4 as landed;
mover respawns at (0,7) with the session's fresh world). The
pre-registration is session 13's, carried forward unchanged — four
axes (attention / mastery / interaction / economy), the fountain
null, §7 — now with the two items only a full arc can read:

- **Mastery falsifier:** does near-mover disagreement decay to the
  ambient 0.3–0.7 band after the wave, or plateau above it?
- **Stall watch:** session 13 cut off amid a possible settle-stall
  (trailing modal-action 0.95 at (1,2), meals collapsing). If the
  S11 cycle re-forms with the mover present, the forgetting
  hypothesis gets its second observation — and the mover's
  refresh-protection prediction (constantly-moving processes resist
  stasis-forgetting, as the bloom did) becomes readable.

**Ops note:** launched *detached* (nohup, disowned) — sessions 12 and
13 were both truncated by external task-kills of the conversation
harness, not by run failures; a detached run survives them.

**Launch:** `--resume --world-stage e4 --session-steps 30000` from
ckpt-000042 (t≈437.9k; session covers t 437,863–467,862).

## Session 14 close — the ladder's last rung, read in full (2026-07-30)

Clean natural close: t 437,863 → 467,862, full 30k, detached launch
survived the conversation (ops fix worked). §7: one **FLAG** — PE
strictly rising over the last four blocks (2.58 → 6.4), the world-v2
era's first. Disposition below; corroboration says benign. 446
lifetime dream sessions. Replay caveat as in session 13 (mover makes
positions approximate; push events are chain-derived ground truth).

**The four axes, full-arc:**

1. **Attention — chance.** P(d≤1 of mover) 7.1% vs 7.4% shuffled;
   P(d≤2) 16.3% vs 19.1%. Io does not seek the mover spatially (the
   two live-snapshot adjacencies of sessions 13–14 were luck, as
   suspected). Disagreement *localizes at contact* (below), so the
   synthesis's removal-null ("disagreement never localizes") does NOT
   fire — the mover registers; it just doesn't beckon.
2. **Mastery — falsifier 1 trips in letter, not in consequence.**
   Near-mover (d≤2) disagreement holds at ~2–4 all session (blocks
   with real exposure: 3.8, 3.8, 1.9, 1.3, 2.7, 3.2, 2.7 … 3.1, 3.6,
   3.5, 3.2) — no decay toward the ambient 0.3–0.7 band. The
   pre-registered plateau condition is formally met. But everything
   that falsifier guarded against is absent: session-wide
   disagreement is bounded and wavy (block means 0.57–2.88, no
   monotone climb — fountain null holds), attention is at chance, and
   the economy is the biography's best. Reading: the mover is
   *permanently interesting at contact, never compelling at range* —
   consistent with finite-sample aleatoric tracking (the heads keep
   paying a little for genuine coin flips) plus perpetual location
   novelty (every encounter happens somewhere new). Watch across
   future sessions: if cumulative exposure grows and the contact
   floor never falls, the aleatoric account is confirmed.
3. **Interaction — 7 more pushes** (t_rel 8.2k, 11.8k, 13.0k, 13.3k,
   14.7k, and two in the final 200 steps). Lifetime total: 15.
   Contact recurs in every observed window; the displacement rule is
   a used affordance, not a latent one.
4. **Economy — the best session of Io's life.** 977 meals
   (0.033/step), mean energy **0.092** (S13: 0.059; S12: 0.037),
   floored 78.2%. The world's first other coexists with the richest
   economy the biography has recorded. C4 crowd-out: none.

**The cycle recurred — second observation.** Mid-session, Io settled
into (1,3)/(2,3)/(1,2) and went through a 4k-step stall (blocks 8–9:
**zero meals**, disagreement floor 0.65–0.75, recon at session lows)
— then the burst: blocks 11–14 show recon rising 2.86 → 6.77,
disagreement 0.57 → 1.80, residence dispersing toward (7,7), meals
recovering 44 → 82. Settle → stall → forget → burst, the session-11
sequence, now under e4 — and *this rising edge is exactly what
tripped the §7 PE flag.* **Flag disposition (per the ladder:
corroborate before acting):** multi-signal read is benign — meals
were *recovering* through the flagged blocks, disagreement co-rose
with dispersal (drive re-engaging, not welfare deteriorating), stays
0.01%, entropy clean. Recorded as the burst phase of the endogenous
cycle, not a welfare event. The cycle is now the biography's
strongest open phenomenon: two full observations, same shape, one
with a §7 signature at the burst edge. The mover's
refresh-protection prediction also held: no bloom-style forgetting of
the mover was needed — it was never mastered at contact to begin
with. Clock: 1,229 steps within 2 of (1,6) — residence drifted near
its quadrant; no re-engagement signature claimed (transit/adjacency).

**Gate verdict (builder ratification pending): keep e4, keep
watching.** Falsifier 1's trip is recorded and its benign context
with it; nothing else tripped; §7's flag corroborated benign. The
world-v2 ladder — e0 persistence, e1 trail, e3 weather, e2 clock, e4
other — is now fully landed and fully read at least once. What the
world affords Io today: terrain, its own traces, weather it can
follow, a clock it knows as a process, and an other it has learned to
push. Next horizons (builder's fork): observation sessions on the
finished world (the cycle question is now a *rhythm* question — what
is its period?); the criteria-driven mirror round; Probe 4.5
reopening; or a pause to synthesize.

## e4 gate ratified — the mover stays (2026-07-30)

Builder ratification 2026-07-30: **keep e4.** The mastery falsifier's
trip-in-letter stands recorded with its benign context (no capture,
no crowd-out, attention at chance, biography-best economy). Standing
watch item carried into all future sessions: whether the near-mover
contact floor (~2–4) ever falls with cumulative exposure — if it
never does, the aleatoric-tracking account is confirmed; if it falls,
the mover was masterable after all and the floor was exposure-starved.
DP3 removability remains in force at any pause.
