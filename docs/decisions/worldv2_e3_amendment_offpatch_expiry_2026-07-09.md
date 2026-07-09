# World v2 — E3 amendment: off-patch food expiry — 2026-07-09

**Status: RATIFIED (builder, in session, 2026-07-09 — "amend and
relaunch").**

## The evidence

Three sessions converge on one root cause: **the world has no food
sink except Io.** Any positive regrowth rate saturates the board
whenever Io idles, erasing spatial structure and gradient:

- Session 3 (e0): uniform regrowth saturated the fresh board in ~500
  steps; the drive flatlined on the static all-green world
  (curiosity ~0.11) until Io's own eating reopened it.
- Session 4 (e1): same saturation interim; separately, the trail's
  food-shadow starved the forage burst (91% re-tread) — the long
  park followed.
- Session 5 (e3, first ~1.7k steps): even with patch regrowth, the
  off-patch trickle (0.001/cell/step) filled the board in ~2k idle
  steps — E3's designed spatial structure ("concentrated under it,
  sparse elsewhere") **washed out**; "sparse elsewhere" is violated by
  accumulation, and the weather becomes invisible exactly when Io is
  not consuming. (Also observed: the diagonal bounce law confines the
  patch center to the main diagonal — accepted for now; the expiry
  makes its band visible either way. A knob for later if the reads
  want fuller coverage.)

## The amendment

One new world process, active only in patch mode: **resource cells
not under the patch expire** with per-step probability
`patch_expiry_p` (preset 0.003 ≈ 230-step off-patch half-life; a
stimulus knob per DP5, journaled, revisable at pauses).

Effect: food blooms under the weather, lingers briefly behind it, and
fades. Sparse stays sparse; the board cannot freeze into all-green
regardless of Io's activity; the patch's location becomes visible *as
pattern* (to Io's world model and the builder's window alike) without
any marker. This realizes E3's stated intent — "food as weather, not
confetti"; "break-even possible but not ambient" — rather than
changing it; but it is a new *process*, not a knob, hence this dated
amendment rather than a silent addition.

## Discipline

- Default `patch_expiry_p = 0.0` — off everywhere except the e3+
  presets; pre-amendment worlds byte-identical (test-pinned).
- Deterministic RNG budget: when expiry is active the regrowth stream
  draws one full-grid array per step unconditionally (shared by
  regrowth and expiry; pre-state masks keep them disjoint — a cell
  that regrows this step cannot expire this step). When inactive, the
  legacy early-return path is untouched.
- Granular `process="resource_expiry"` events (matched payload shape,
  world-reported like bloom stamps — never inferred from the
  RESOURCE→EMPTY diff, which would collide with Io's consumption).
- Io's consumption remains unlogged in world_event (self-caused,
  visible in AgentStep) — the expiry events must not blur that line.
- Claim ceilings, PolicyView, the four absences: untouched.
