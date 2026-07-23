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
