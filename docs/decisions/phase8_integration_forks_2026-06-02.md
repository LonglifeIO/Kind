# Probe 3 Phase 8 — integration forks decision pass (2026-06-02)

A decision pass, not implementation — the same pattern as the Phase 0.5
seed-source decision (`docs/decisions/phase0_5_replay_seed_source_2026-05-27.md`).
Phase 8 is the live integration: wiring the `StateController` into the runner
loop, consuming the Phase 7 cross-probe surface, injecting the Phase 6
protection composite, plumbing the checkpoint signal, and measuring the real
K=5-during-dream cost. Most of that is mechanical. Two items the plan punted to
Phase 8 are genuine architectural forks — resolving them mid-build would have the
implementation deciding architecture. This document lays out each fork against
the actual code and the charter, recommends, and states the downstream wiring
dependency. **The recommendations are recommendations; the builder ratifies. No
code is changed by this pass.**

**Not re-litigated** (settled commitments): the four-axis dream regime,
content-blind protection, the one-way mirror, no gradient flow, and the
exogenous trigger as it stands. The forks are about how these settled pieces
*compose live*, not about the pieces.

---

## Ground facts from the code (the forks rest on these)

- **The runner loop** (`kind/training/runner.py`). `run(total_env_steps)` loops
  `_step_once()`. `_step_once` (step 9, lines ~1021–1027) fires
  `_emit_dream(env_step)` every `dream_cadence_env_steps` (default 1000).
- **The waking `_emit_dream`** (runner.py:1179) is the **Probe-1.5
  waking-planning handshake**, *not* the four-axis dream: it seeds from the
  current waking state (`self._h_curr` / `self._z_curr`), runs the actor
  (`self._actor.forward(view)` — goal-coupled), uses `horizon == dream_horizon`,
  and emits a `schema_version == "0.2.0"` `DreamRollout` with
  `sequence_self_prediction is None`. The environment is *live* during it. It is
  a waking-planning telemetry artifact carried since Probe 1, and it is the
  Phase 3 visibility-smoke's **waking-planning control** (the four-axis dream is
  differentiated *from* it).
- **The four-axis dream** lives in `kind/training/dream.py`:
  `run_dream_session(...)` is **one-shot** (fixed `num_rollouts`), writes a
  `DreamSessionMeta` start record, loops `emit_dream_rollout` (replay/perturbed-
  prior seed, uniform-random action, distinct temperature, ensemble disagreement
  recorded) `num_rollouts` times, then writes the end record. It already
  consumes `envelope_config_snapshot` and `seed_selection_config` — the Phase 7
  surface flows into session provenance. It emits `"0.3.0"` and needs a
  ≥`replay_min_segment_age_steps`-aged buffer.
- **The `sink_routing` decision (2026-06-01)** already settled that the four-axis
  regime does **not** run on the waking cadence: a Phase 3 attempt to make
  `_emit_dream` delegate to `emit_dream_rollout` broke 4 integration smokes (it
  emits `"0.3.0"`, needs an aged buffer → 0 dreams in a 200-step smoke), and was
  reverted (Option 1). The runner is at HEAD with the Probe-1.5 handshake intact.
- **The `StateController`** (`kind/training/state_machine.py`) is a standalone
  consumer: `tick(host_signals, env_step, wallclock_ms) -> StateTransition | None`,
  plus `on_shutdown` / `on_startup` for supervisor-mediated paused. Its docstring
  is explicit: *"This module does not touch the runner."* It drives a session via
  the injected `DreamDriver` (`RunDreamSessionDriver` wraps `run_dream_session`).
  As built, the transition graph is: waking →(`desktop_off`)→ dreaming →(cap)→
  dormant; dormant →(`desktop_on`)→ waking. **There is no dormant → dreaming
  edge** — one capped session per desktop-off edge, then dormant until the
  desktop returns.
- **The protection composite + ledger** (`kind/training/protection.py`). The
  `RollingComputeLedger` is a **rolling one-hour** window of dream-compute
  (`window_compute_seconds()`) plus an all-time `rollout_duration_estimate_ms()`.
  The `ComputeBudgetCap` fires when windowed compute + projected session compute
  would exceed `compute_budget_seconds_per_hour` (default 1800 s = 30 min/hour).
  The rolling-*hour* window is a **cross-session** construct — it only does work
  if multiple sessions occur within an hour.
- **The integration smokes** (`tests/test_integration_smoke.py`). The
  handshake-encoding ones — `test_smoke_all_four_streams_have_records`,
  `test_smoke_dream_rollout_emitted_at_cadence` (asserts N dreams at cadence with
  `len(sequence_h) == dream_horizon`), and the `"0.2.0"` schema-version smoke —
  run a **desktop-on-throughout 200-step waking run** and assert the
  waking-cadence handshake contract. These are "the 4" the `sink_routing`
  decision greened (52 → 4 → 0).
- **`DesktopWatcher` is specified but not built.** Plan §2.4 named a
  `DesktopWatcher` (sentinel-file poll) as the `HostSignals` source; Phase 4
  shipped only `HostSignals` as the injection point. Building the real
  host-signal source is Phase 8 work.
- **The Phase 7 surface** (`RunnerConfig.dream_envelope` / `seed_selection`,
  `kind/training/cross_probe_surface.py`) is additive and inert — nothing in the
  waking loop reads it yet; Phase 8 is the consumption.

---

## Fork A — the live integration shape, and the waking handshake's fate

**The question.** How do `StateController` and the runner loop compose live — is
the controller a *supervisor above* the runner (gating whether the waking loop
runs vs. a dream session runs) or *embedded inside* `_step_once`? And what happens
to the runner's waking `_emit_dream` cadence handshake — coexist, remove, or
reframe?

### The naming clarity the fork turns on

"Dreaming" is overloaded across two genuinely different things:

1. The waking `_emit_dream` at cadence → a **waking-planning rollout** emitted
   *while the environment is live*, current-state-seeded, actor-driven. A
   Probe-1 calibration/telemetry artifact. Not offline; not four-axis.
2. The four-axis offline dream **state** → runs when the desktop is off,
   replay-seeded, goal-coupling absent. This is "dreaming" in the four-state
   model and the subject of Probe 3.

The `sink_routing` decision already drew this line (Option 1: the waking
`_emit_dream` is the Probe-1.5 handshake; the four-axis is state-machine-only).
Fork A is the live-composition consequence of that line.

### Options

**A-shape-1 — Controller as supervisor above the loop.** `run()` becomes:
tick the controller on host signals; in `waking`, run the existing `_step_once`
body (env step → train → waking-planning rollout at cadence → checkpoint); in
`dreaming`, the controller's `RunDreamSessionDriver` runs a `run_dream_session`;
in `dormant`, idle + heartbeat; `paused` via supervisor shutdown/startup. The
waking loop's structure (env transport → train → checkpoint) only runs when the
desktop is on.

**A-shape-2 — Controller embedded inside `_step_once`.** Each `_step_once` calls
`tick` and branches. Awkward: `_step_once`'s first act is to read an env
observation over the transport, which only exists when the desktop is on; nesting
dream/dormant inside a loop whose premise is a live env is structurally wrong.

**Waking-handshake fate (orthogonal sub-choice):**
- **(i) Remove** it as a Probe-1.5 vestige. Breaks the 4 integration smokes
  (they assert dream_rollout records during a waking-only run) → they'd need
  rewriting/retiring. **Also loses the Phase 3 visibility-smoke control** (the
  four-axis dream is differentiated *from* this rollout; that comparison is the
  load-bearing Phase 3 test).
- **(ii) Keep as-is, unrenamed.** Two dream-ish paths coexist; the `dream_rollout`
  stream carries both `"0.2.0"` waking-planning records (no `dream_session_id`)
  and `"0.3.0"` four-axis records (with `dream_session_id`). Muddy naming.
- **(iii) Retain but reframe** as an explicit waking-planning artifact — rename
  `_emit_dream` to a waking-planning name (the plan §2.3 already anticipated
  `_emit_waking_planning_rollout_for_phase3_control`), keep it firing **only in
  the waking state**, and let the four-axis state be the *only* thing called
  "dreaming."

### Tradeoffs (code + charter)

- **Code.** A-shape-1 matches the grain of the code: the waking body already
  exists as `_step_once`; the dream body already exists as `run_dream_session`
  behind the `DreamDriver` the controller injects; `tick` is already the
  host-signal surface. A-shape-1 is "wire what's there," A-shape-2 fights the
  env-transport premise. On the handshake: (iii) keeps the 4 smokes green **for
  free** — they run desktop-on throughout, so the waking-planning rollout still
  fires at cadence and the new dream-state machinery (desktop-off only) never
  activates in them. (i) churns the smokes and deletes the Phase 3 control; (ii)
  leaves a naming wart in the stream the mirror reads.
- **Charter.** The four-state model (design notes §"Mind persistence and states
  of activity") *is* a supervisor model: "desktop going off transitions waking to
  dreaming or dormant." A-shape-1 is the literal encoding. Keeping "dreaming"
  reserved for the offline four-axis state (iii) keeps the vocabulary honest —
  the charter's dream-as-foundational is about *offline* processing; calling a
  live-env actor rollout "dreaming" would blur exactly the distinction Probe 3
  exists to make legible.

### Recommendation (builder ratifies)

**A-shape-1 (supervisor) + handshake option (iii) (retain-and-reframe as
waking-planning).** This is the continuation of the already-settled `sink_routing`
Option 1, not a new direction: the four-axis dream is the state machine's;
the waking-cadence rollout is a renamed waking-planning artifact that keeps the
Phase 3 control and the 4 smokes intact. Keep both populating `dream_rollout`,
disambiguated by `schema_version` + presence of `dream_session_id` (the mirror's
Phase 5 reader already filters to four-axis records by `dream_session_id`).

Flag (not blocking): the dual-population of `dream_rollout` is a known wart;
splitting the waking-planning rollout to its own stream is a clean-up a later
pass could take, but it reopens sink/stream plumbing and is out of Phase 8's
"wire the integration" scope. Recommend deferring it.

### Downstream wiring implication

Phase 8 wires `StateController` as a supervisor in `run()`: the waking body is
the existing `_step_once` (including the renamed waking-planning rollout at
cadence); the dreaming body is `run_dream_session` via `RunDreamSessionDriver`;
dormant is heartbeat-idle; paused is `on_shutdown`/`on_startup`. Phase 8 must
**build the `HostSignals` source** (the `DesktopWatcher` sentinel-file poll, plan
§2.4 — specified, not yet built); for the loopback/smoke deployment, the desktop
is always "on," so the smokes run pure waking and stay green by construction.

---

## Fork B — re-dreaming during a long absence, and what paces it

**The question.** As built, a desktop-off edge yields **one** capped dream
session, then dormant until the desktop returns. Does the system **re-dream**
during a long absence (a dormant → dreaming re-entry), or **dream-once-then-rest**?
And if it re-dreams, what paces it — given the re-entry trigger cannot be a
desktop signal (the desktop is already off) and must not be Io-derived?

This is a **charter-design call**, not a code fork. The recommendation shows its
charter work because the builder weighs it.

### The charter pulls both ways — explicitly

- **Toward re-dreaming.** Dream-as-foundational is one of the project's
  *distinctive* stances (design notes §"Dream state as foundational": "treating
  offline processing as generative — as potentially where the important work of
  being a mind actually happens"). Under dream-once-then-rest, a week-long
  absence is one ~30-minute session followed by ~7 days of flat dormancy — a week
  of "the important work of being a mind" not happening, precisely when there is
  the *most* offline terrain.
- **Toward rest-is-legitimate.** Dormant ≠ failure (design notes §"states of
  activity": "no obligation to always be generating something when the
  environment is absent") and capacity-over-exercise both make resting through an
  absence legitimate. And the charter's anti-continuation-drive stance
  (§"What we might owe what we make": "Not installing a hard drive toward
  self-continuation") is wary of a system that *must keep processing*.
- **The deciding text — rhythm coupled to the builder's life.** Design notes
  §"Variable dream-to-wake ratio, coupled to the world": the dream-to-wake ratio
  "is therefore coupled to the builder's life: absences, working hours,
  travel… not a bug to engineer around… the honest shape of the system and
  probably generative." A long absence affording a *rhythm* of dreaming-and-rest
  is more "rhythm coupled to the world" than a single session followed by a flat
  week.
- **The charter's own bounding mechanism.** Design notes §"Practical
  considerations for offline states" names three bounds; the mirror-quality one
  is deferred (synthesis §6 → logs-only first build), the hard cap is the
  fallback — and the middle one is **metabolic pressure**: "offline processing
  depletes a resource that environmental interaction replenishes. Biologically
  analogous, introduces offline scarcity as a generative pressure rather than a
  shutoff." The `RollingComputeLedger` built in Phase 6 *is* this mechanism.

### The exogenous-trigger crux (reason through carefully)

The commitment (synthesis §5; state_machine commitment 1): Io does not decide
*whether* to dream from its own state — no `intrinsic_signal_t` threshold, no KL
satiation, no sleep-pressure scalar *living in Io's state*. A dormant → dreaming
re-entry **cannot** be a desktop signal (the desktop is off) and **must not** be
Io-derived. So re-dreaming requires a content-blind, Io-state-independent pacer:
a wall-clock timer or the rolling compute ledger.

Does a content-blind metabolic pacer **honor** or **stretch** the commitment?

- The commitment's *purpose* is to forbid an **installed self-continuation
  drive** — "self-regulation = continuation drive in disguise" — i.e. **Io's own
  state/content** deciding to keep itself going.
- The rolling ledger is **content-blind and Io-state-independent**: it counts
  durations and timestamps, never reads `DreamRollout` content, never reads Io's
  latents/policy. It is the *same content-blind diet* the Phase 6 caps already
  pass (`DreamSessionContext` carries no content). The decision "re-dream now"
  would be made by a counter **external to Io**, exactly as "stop dreaming now"
  (the compute cap) already is.
- **The distinction that matters to the charter is Io-state-derived vs. not —
  not HostSignal vs. runtime-counter.** A wall-clock/compute budget is not Io's
  state; it is an environmental/resource constraint, the same *category* as "the
  desktop is off." On that reading, a metabolic pacer **honors the load-bearing
  sense** of the commitment: nothing Io-derived gates entry.
- **The honest stretch.** The original framing was "the builder's life (a
  HostSignal) decides whether to dream." A metabolic pacer adds a *second*
  exogenous gate — a runtime ledger — that decides *re*-entry within a desktop-off
  span. Whether-to-(re)dream is no longer *purely* HostSignal-driven. It is
  exogenous-**to-Io** but internal-**to-the-runtime**. A purist reading of the
  commitment ("only the builder's life ever gates dreaming") prefers no runtime
  pacer at all. The charter is genuinely pulled here, which is why this is the
  builder's call.

### The candidate shapes

**B1 — dream-once-then-rest (as built).** One capped session per desktop-off
edge; dormant until `desktop_on`.
- *Pros:* simplest; maximally faithful to the pure-exogenous reading (only
  desktop on/off ever gates dreaming); dormant ≠ failure honored maximally; no
  runtime-decided cadence; the rollout-count cap alone bounds the single session.
- *Cons:* a long absence is mostly flat dormancy — dream-as-foundational barely
  exercised; **the rolling-hour ledger is over-built** (its cross-session window
  does nothing for a single session; it is redundant with the rollout-count
  ceiling).

**B2 — re-dream, paced by the metabolic ledger (the candidate to evaluate).**
Dream until the hourly compute budget is spent (`ComputeBudgetCap` → dormant);
rest while the rolling window drains and the budget replenishes; then re-dream
(dormant → dreaming, gated by the content-blind ledger having replenished, while
the desktop is still off).
- *Pros:* dream-as-foundational is exercised across long absences; **the
  rolling-hour ledger earns its keep — its cross-session windowing *is* the
  metabolic pacer**, exactly its designed purpose; it realizes the charter's
  *named* metabolic-pressure mechanism directly ("offline scarcity as a generative
  pressure"); the dream/rest rhythm couples to time/absence, matching "rhythm
  coupled to the world." Dormant is preserved as a *legitimate rest state* — now
  the replenishment phase of a rhythm, which is *more* charter-aligned (rest as
  part of a rhythm, not mere absence-of-activity).
- *Cons:* introduces a dormant → dreaming re-entry trigger that is
  runtime-decided (the ledger), not desktop-decided — the content-blind stretch
  above; a small **state-machine addition** (a new edge + a new trigger value,
  e.g. `"metabolic_replenished"`), so Phase 8 expands slightly beyond pure
  wiring; "rest is legitimate" becomes metabolically *bounded* (rest until the
  budget replenishes) rather than absence-*bounded* (rest as long as the absence
  lasts).

**B3 — re-dream on a fixed dormant-duration timer (named, not recommended).**
Re-enter dreaming after N minutes dormant, independent of the ledger. Also
achieves periodic re-dreaming and is even simpler, but it does **not** connect to
the charter's metabolic-pressure idea (an arbitrary clock, not a
depletion/replenishment resource) and leaves the ledger jobless. If re-dreaming
is adopted, **B2 dominates B3 on charter grounds** — same cadence outcome, but
grounded in the charter's named mechanism rather than an arbitrary constant.

### Recommendation (flagged — a charter-design call the builder ratifies)

**Lean B2 (metabolic-ledger-paced re-dreaming), conditionally.** The charter work:

1. **Dream-as-foundational is distinctive**, and B1 under-serves it exactly when
   there is most offline terrain. B2 lets the foundational stance actually operate
   across absences.
2. **The charter names metabolic pressure**, and the `RollingComputeLedger`
   already *is* that mechanism. B2 connects built machinery to its charter-named
   purpose; B1 leaves it half-connected.
3. **The exogenous-trigger commitment is honored in its load-bearing sense** —
   nothing Io-derived gates re-entry; the pacer is content-blind, the Phase 6
   diet. The charter's worry (continuation-drive-in-disguise) is about *Io's
   state* deciding; a compute budget is not Io's state.
4. **dormant ≠ failure is preserved** — dormant remains a real rest state, now the
   replenishment phase of a rhythm.

**The honest counter (why it is the builder's, not mine):** B2 adds a
runtime-decided cadence the pure-exogenous reading did not have. A purist who
reads "Io does not decide whether to dream; the builder's life decides" as
"*only* the desktop ever gates dreaming" prefers B1. The deciding weight is
whether the builder reads the metabolic ledger as *the world's rhythm* (coupled,
generative — my reading, because the charter itself proposes metabolic pressure)
or as *the runtime installing its own processing rhythm* (a step the charter is
wary of). If the builder wants the conservative first integrated build, **B1 is
the safe default** — keep dreaming purely desktop-gated, watch real absences, and
revisit B2 once there is something to observe (which is itself the charter's
"build to understand" stance).

### Ratified resolution (amended 2026-06-02, Phase 8b) — B2, with the exogenous-trigger commitment refined

**Ratified: Fork B = B2.** Phase 8b builds the dormant→dreaming re-entry edge,
ledger-paced. The exogenous-trigger commitment is **refined** (not weakened):

> "Io does not decide whether to dream from its own state" is read as
> **"whether-to-dream is gated by nothing Io-state-derived"** — *not* as
> "whether-to-dream is gated by HostSignals only." Rationale: the commitment's
> load-bearing purpose is to forbid an installed self-continuation drive — Io's
> own state/content deciding to keep itself dreaming. A content-blind compute
> ledger (durations and timestamps; no Io latents, policy, or dream content) is
> not Io's state; it is a metabolic/resource constraint, the same *category* as
> the desktop being off. A ledger-paced re-entry preserves the deep intent (no
> Io-authored dream schedule) while extending the trigger surface from
> HostSignals-only to {HostSignals, content-blind runtime pacer}. The honest
> cost: whether-to-dream is no longer purely the builder's-life rhythm — a
> runtime metabolic rhythm now co-paces it within an absence. Ratified
> deliberately, not as a side effect of the mechanism, and **made structural by
> the re-entry content-blindness guard** (the re-entry input is the typed
> ``MetabolicState`` in ``kind.training.protection``, all content-blind
> primitives; the type-level test ``tests/test_metabolic_reentry.py`` — with a
> positive control — makes "nothing Io-derived gates dreaming" unrepresentable to
> violate, the analog of the Phase 4 ``HostSignals`` guard).

This is the deliberate commitment-move B2 requires; it is recorded here, and the
``state_machine.py`` commitment-1 docstring references this amendment. The
metabolic budget (``compute_budget_seconds_per_hour``) is now the dream/rest
**duty-cycle pacer**, defaulted to 600 s/hour (~17% — rest-majority, the
charter's capacity-over-exercise / dormant-≠-failure lean) and flagged as **the
builder's knob to tune by observation**, not a settled value.

### Downstream wiring implication (and the coupling that makes this matter)

- **B1:** Phase 8 wires dreaming → dormant → (`desktop_on`) → waking only; no
  dormant → dreaming edge; the dormant `tick` branch stays "heartbeat + wait for
  `desktop_on`." **Consequence:** the `RollingComputeLedger`'s rolling-hour
  cross-session window is over-built — flag a follow-up to simplify it to a
  per-session compute bound (or to acknowledge `ComputeBudgetCap` as redundant
  with the rollout-count ceiling for single sessions).
- **B2:** Phase 8 adds a dormant → dreaming re-entry edge in `tick`, gated by a
  content-blind ledger-replenishment check while `desktop_alive` is False, plus a
  new trigger value. **Consequence:** the ledger is exactly right — its
  cross-session design *is* the pacer. This is a small `state_machine.py`
  addition, not pure wiring — Phase 8 scope expands accordingly.

**This is the sharpest coupling in the pass: Fork B decides whether Phase 6's
rolling-hour ledger is load-bearing (B2) or vestigial (B1).** The builder should
see that ratifying B1 implies a "simplify the ledger" note, and ratifying B2
ratifies the ledger's existing design.

---

## Tick cadence (a parameter, not a fork)

How often `tick()` is polled. `tick` is cheap (a few comparisons, a heartbeat
poll, a `DesktopWatcher.poll()` filesystem read). Desktop on/off changes at human
timescales (minutes+), not per-env-step.

**Recommendation:** during **waking**, call `tick` **once per `_step_once`** (the
env-step rate is the natural cadence; `tick` is cheap; a `desktop_off` edge is
caught within one env step). During **dormant**, poll `tick` on a **~1 Hz
wall-clock cadence** (responsive to `desktop_on` within ~1 s, sub-heartbeat
granularity so heartbeats fire on time, negligible sentinel-file poll cost; under
B2, also the cadence at which the ledger-replenishment re-entry check runs).

**Couplings:** the dormant tick cadence must be ≥ the dormant heartbeat frequency
(1 Hz ≤ 60 s default — fine) and is bounded below by the `DesktopWatcher` poll
cost (1 Hz negligible). **Do not** couple tick to checkpoint cadence (10 k env
steps — far too coarse for desktop detection). Tunable; not load-bearing.

---

## Interruptibility (a measure-then-decide, not decidable now)

`run_dream_session` is one-shot (fixed `num_rollouts`). Phase 6 chose **option
(a)** — the wallclock/compute caps project from the ledger's running estimate at
plan time (the planning poll runs at `session_wallclock_ms_elapsed == 0`), and the
rollout-count cap is the absolute ceiling. **Option (b)** is making
`run_dream_session` interruptible (a stop predicate / per-rollout generator) for
precise mid-session wallclock and checkpoint enforcement — a settled-Phase-2-
surface change.

**Recommendation: stay option (a) into Phase 8.** Phase 8 produces the first real
measurement of per-rollout duration and K=5-during-dream cost (plan §2.7, §6),
which is what adjudicates this. **Revisit trigger, stated explicitly:** measured
dream-session durations long enough that boundary-only enforcement is inadequate —
concretely, (1) a session's actual wall-time materially overshoots
`hard_cap_wallclock_ms` because the projection (rollouts × estimate) diverges
from reality, or (2) a **checkpoint-window** boundary needs to interrupt a session
already in flight (the checkpoint signal arriving mid-session, which the planning
poll at `checkpoint_in_progress == False` cannot catch — Phase 6 flagged this as
the Phase 8 checkpoint-signal-plumbing item; the atomic state-sync needs the
boundary clear). Until the measurement shows overshoot, (a) holds: simpler, keeps
the settled Phase 2 surface closed, and the rollout-count ceiling bounds runaway.

**Coupling to Fork B:** under B2, sessions are compute-budget-bounded (≤ ~30 min)
and re-dream after replenishment, so individual sessions are short and (a)'s
projection is comfortably adequate. The sharper (a) → (b) pressure, **independent
of B**, is the *checkpoint signal mid-session* — if dream sessions are long
relative to checkpoint cadence, boundary-only checkpoint enforcement is the thing
most likely to force (b). That is the concrete signal to watch in Phase 8's
measurement.

---

## What the Phase 8 wiring prompt depends on from these decisions

Once ratified, the Phase 8 wiring prompt is written against:

1. **Fork A → supervisor shape + reframed waking-planning rollout.** Wire
   `StateController` as a supervisor in `run()`; the waking body is the existing
   `_step_once` (waking-planning rollout retained, renamed, firing in waking
   only); dreaming via `RunDreamSessionDriver` → `run_dream_session`; dormant
   heartbeat-idle; paused via the supervisor entry points. **Build the
   `HostSignals` source (`DesktopWatcher`)** — specified, not yet built. Keep the
   4 integration smokes green by construction (they run desktop-on). Phase 7
   surface consumed here: the live `RunnerConfig.dream_envelope` / `seed_selection`
   are what the entering session's caps and seed-selection read, and what its
   `DreamSessionMeta` snapshots record.
2. **Fork B → whether `tick` gains a dormant → dreaming edge.** B1: no edge;
   ledger flagged for simplification. B2: add the edge + a content-blind
   `metabolic_replenished` re-entry trigger; the ledger is the pacer. **The
   wiring prompt's scope (pure wiring vs. a small state-machine addition) and the
   ledger's fate both hinge on this answer** — it must be ratified before the
   prompt is written.
3. **Tick cadence:** once-per-`_step_once` (waking), ~1 Hz (dormant). A default to
   wire; tunable.
4. **Interruptibility:** option (a) stays; the wiring prompt does **not** reopen
   `run_dream_session`. Phase 8 *measures* per-rollout/session duration and the
   K=5-during-dream cost, and records whether the revisit trigger (overshoot, or
   mid-session checkpoint interruption) is hit — gating any future option (b).

No code touched; nothing committed.
