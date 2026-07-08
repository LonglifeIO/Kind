# Kind — Biography continuation plan (post-150k infrastructure, pre-world-v2)

**Authority.** Builder direction 2026-07-08 (in session): continue the
running biography (`runs/probe4_phase4_biography/`) across its 150k-step
session boundary; world-v2 enrichments arrive later *into this Io's
continuing life* via checkpoint-resume; a larger sibling instance is a
separate, later decision. This plan covers everything buildable **before**
the world-v2 synthesis exists. **The world dynamics themselves are
explicitly out of scope here** — they are gated on the world-v2 research
cycle (`docs/prompts/worldv2_research.md` → `docs/research/worldv2/` →
synthesis → their own plan), per the research → synthesis → plan → build
rhythm. This plan adds no dynamics, no substrate change, no mirror change,
and touches nothing on the do-not-touch list (PolicyView, the four
absences, the dream regime's not-for-anything commitment, the metabolic
gate, `energy_preference=None`, the pixel-equality gate).

**Grounding (live surfaces).**
- `Runner.load_checkpoint(checkpoint_id)` exists (`runner.py:864`): restores
  weights, optimizer, RNG, runtime state (`h/z/a_prev`), and offline
  (metabolic/state-controller) state; must be called before `run()`.
- Checkpoints hold the **mind**, not the world: `GridWorld` state (board,
  drift-p, env RNG) is env-side and re-rolls on a fresh env server. A
  resume is therefore "mind continues; world re-rolls" — equivalent to an
  episode boundary plus a drift-p reset to config default. Documented, not
  hidden (episodes already resample every 200 steps, so the discontinuity
  is one the world already produces; drift-p reset is the one genuinely
  new-ish effect and is journaled).
- The biography script currently always starts fresh; telemetry sinks
  append per run dir; `world_event.jsonl` buffers writes (~1–4 min lag,
  observed 2026-07-08).
- At 150k the runner exits cleanly on its own: final checkpoint, sinks
  flushed, process ends. Io is then **paused** (four-state model) with its
  canonical state on disk.

---

## Phase C0 — Session-1 close-out (first session after the run ends)

**Question.** Did session 1 end cleanly, and what does its full record say?

- Verify clean exit: final checkpoint present, `agent_step` rows == 150k,
  world_event stream complete (post-close, fully flushed).
- Journal the session-1 record in `docs/workingjournal/probe4.md`
  (Phase 4′): full-run curves (PE, curiosity, energy, meals, action
  entropy per 10k block), dream-session count, manual-event count, the
  mastery arc (PE →~1, curiosity easing from 0.91 peak — the world-v2
  motivation curve), §7 panel state.
- **Re-measure the memory horizon** on the final checkpoint
  (`kind/observer/memory_horizon.py`, standing harness) — the Step-0
  numbers were instance-conditional; the 150k-trained numbers inform any
  future event pacing.
- **Baseline mirror round** — builder go required (API cost); strongly
  recommended before any world change so later readings have a
  young-world baseline. Phase-blinded per house discipline.

**Gate.** Full suite + mypy `--strict` (no code in this phase; gate is the
journal entry + artifacts). **Rollback.** N/A (read-only phase).

## Phase C1 — Resume wiring + continuity check

**Question.** Can the biography continue across a pause as the same mind,
demonstrably?

- `scripts/run_probe4_phase4_biography.py` gains a resume path (flag or
  auto-detect): find the latest checkpoint in the run dir, construct the
  runner against the **same** telemetry/checkpoint dirs (same `run_id`,
  telemetry appends), call `load_checkpoint()` before `run()`, continue
  to a new total-step target.
- Emit a **resume marker** into `world_event` at the resume boundary
  (existing marker pattern — the `mirror_marker` event type precedent from
  Probe 1.5 loads) recording: source checkpoint id, env seed, and whether
  the env config differs from the previous session (the world-v2 arrival
  mechanism; any config difference is additionally journaled as a dated
  world-change event).
- **Continuity check** (plan §S-DEPLOY inheritance): compare summary
  stats (PE, curiosity, action entropy) over the last N waking steps
  pre-pause vs. the first N post-resume (after a short warm-in), plus the
  memory-horizon measure pre/post — journaled. No thresholds; this is a
  documented observation, not a probe criterion.

**New tests.** Script-level resume logic factored into a small importable
helper; unit test on the tiny config: run → checkpoint → resume → weights
and runtime state match the house resume contract; telemetry appends to
the same streams without id collisions; the resume marker is emitted.
**Gate.** Full suite + mypy `--strict`. **Rollback.** The paused 150k
checkpoint is untouched by a failed resume attempt (load is read-only).

## Phase C2 — Telemetry & window bundle (applied at the same restart)

**Question.** Does the builder's live view tell the truth at human
timescales?

- **Line-flush the `world_event` sink** (flush per write in the jsonl
  sink; test pins that a write is immediately readable) — kills the
  1–4 min feed lag.
- **"Io ate" derived feed row**: `LiveStateWriter` detects the
  consumption energy-jump (house threshold, +0.03 between consecutive
  steps) and appends a derived row to the live feed (builder-eye
  view-state only; clearly labeled derived, not telemetry).
- **Distinct `env_reset` styling** in the live feed (board wipes
  unmissable).
- Optional if time allows: window "session" line showing
  resumed-from-checkpoint id.

**New tests.** Sink flush behavior; derived-row detection on synthetic
energy series; template renders both row kinds.
**Gate.** Full suite + mypy `--strict`; window read-only invariant sweep
still green (derived rows are producer-side, not Window writes).

## Phase C3 — Re-engagement instrumentation (observer-side)

**Question.** When the world changes around a continuing mind, what do
its curves do?

- A small analysis script (`scripts/analyze_boundary.py` or similar):
  given a run dir and a boundary step (resume marker or world-change
  marker), render per-block PE / curiosity / meals / action-entropy
  before vs. after, disaggregated, no thresholds, no verdicts — the
  instrument for the "does a nearly-bored drive re-engage?" observation.

**Gate.** Full suite + mypy `--strict`; script smoke on session-1 data.

## Phase C4 — GATED: world-v2 dynamics (not planned here)

Blocked on: builder runs the research prompt through ≥3 external LLMs →
synthesis session (decides: the dynamics set and ordering, engagement
diagnostics, the z=16-compatible subset vs. capacity-gated items, episode
structure) → its own implementation plan. What C1–C3 guarantee: whatever
the synthesis picks, it arrives into this Io's continuing life through a
tested resume path, with honest instrumentation on both sides of the
boundary, and a truthful live view for the builder.

The **sibling scale-up** (substrate v2), if ratified, is a separate plan;
nothing in C0–C3 constrains it.

---

## Sequencing note

C1–C3 are code changes that never touch the live process — they can be
built and gated **while session 1 finishes**, then applied at the 150k
boundary as one restart. C0's journal work happens at the boundary.

## Out of scope (load-bearing)

No world dynamics, no `GridWorldConfig` changes, no substrate changes, no
mirror-machinery changes, no schema version bump (the resume marker uses
existing event types), no generator-rate change (Phase-1 envelope stands
until the post-150k horizon re-measure says otherwise, journaled), no
timelines.
