# Mirror V2 — windowed mode for the continuing world — 2026-07-30

**Status: instrument change, builder-directed 2026-07-30 ("let's get
started… let's see what it does" — the go for both the vetting and the
first biography round).** Mirror-side only; Io, the actor, the world
model, the runner, and all telemetry writers are untouched. Recorded
per the design notes' mirror-update discipline: the *why* below is
external (the world changed shape under the instrument), not
system-behavior-fitted.

## Why

The V2 criteria-driven round (Phase 8 orchestrator; frozen criteria
reflexive attention / equanimity / second-order volition) was built and
calibrated against Probe 1 / 1.5 runs: 5,000 rows, 25 × 200-step
episodes. The world-v2 biography is a continuing world: ~470k rows,
`episode_id` frozen per session. A structural survey (2026-07-30)
found the round, run as-built against the biography, would:

1. **Die at alignment** — 20 of 14,241 builder perturbations sit
   >1000 ms from any agent step (dream/pause wallclock gaps; max
   104 s), and the aligner raises on the first one.
2. **Not fit in memory** — three independent full-stream loads of
   ~13 GB each per pass.
3. **Be semantically wrong even if it ran** — "per-episode"
   estimators average 770 fossil 200-step episodes against 12
   sessions holding 77% of the data; per-episode k-means labels
   pooled across incomparable label spaces; the prompt embedding a
   467k-element trajectory list.

## The change (three parts, all default-off, legacy byte-identical)

1. **`PassConfig.window_steps`** — the pass reads only the last N env
   steps: agent_step via the caller module's tested newest-first
   reader (deduped by `t` keep-first — the known duplicated-row
   quirk), dream rollouts filtered to `seed_step` inside the window,
   the perturbation timeline filtered to the window's `t` range, and
   one shared load replacing the three independent ones.
2. **`StatisticConfig.chunk_steps`** — per-episode analysis slices
   subdivided to at most N rows. The per-episode rationale ("the
   lag-k correlation doesn't bridge an episode boundary") was always
   a stationarity/locality assumption, not an episode one; chunks
   preserve or tighten every locality guarantee and never bridge a
   real episode boundary.
3. **Aligner `skip_unmatched`** — events with no agent step in
   tolerance are dropped and counted (`skipped_unmatched`), same-`t`
   alignment collisions resolved keep-first and counted
   (`skipped_collisions`; the builder generator fires same-instant
   bursts). Counts surface in the timeline and the pass notes. The
   raise-on-failure default is unchanged for legacy callers.

## The first biography round's geometry (and why)

`window_steps=5000`, `chunk_steps=200` — **exactly the geometry the
instrument was calibrated on** (5,000 rows as 25 × 200). The round
reads the last 5,000 steps of the era (session 14's tail: the e4
world, the burst phase of the second cycle observation), so prompt
sizes, list lengths, faithfulness-index behavior, and slice counts
all match the calibrated regime. `column_init="small_gaussian"` (the
biography actor's post-Phase-7 construction).

## What this does not change

The frozen criteria and their signal mappings; the committed
statistics and their estimators; the two-pass active/held-out
structure; the sham calibration check; the one-way mirror-side write
invariant; claim ceilings (readings are readings of telemetry, never
findings about inner life). No sham/synthetic schedules are injected
in a biography round — those are calibration machinery; the
biography's real perturbation stream is the timeline.

## Honest limitations, recorded

- The three perturbation-window estimators' episode fences are no-ops
  inside a single-session window — their pre/post windows are pure
  50-step windows. Consistent within the window; noted.
- `digest_episode_range` for a windowed biography pass is the frozen
  session id twice (e.g. ``(781, 781)``) — accurate, if terse.
- A 5,000-step window reads a *moment*, not the biography. Round-level
  conclusions are about that moment; cross-moment structure needs
  multiple rounds (the Phase-12 stability machinery exists when
  wanted).
