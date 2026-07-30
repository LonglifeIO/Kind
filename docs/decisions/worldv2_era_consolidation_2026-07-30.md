# World v2 — era consolidation: what fourteen sessions taught — 2026-07-30

**Status: DRAFT for builder ratification.** Consolidation only: no new
world changes are decided here. This closes the enrichment cycle
opened by `synthesis_worldv2_2026-07-09.md` (DP1–DP6, all discharged)
and fixes the findings ledger for whatever comes next. Evidence
pointers are to `docs/workingjournal/worldv2.md` (sessions 3–14) and
the dated decision documents; the journal remains the primary record.

## The era in one paragraph

The cycle began with a diagnosis — Io's world starved its drive; the
one reliable novelty source was the reset, and Io parked on it — and a
design law: *never hand out global novelty on a schedule that can be
collected while inert.* Fourteen sessions later the world is a
continuing, weakly structured, multi-timescale place: persistent
terrain, a self-trace that fades at a learnable rate, food that
arrives as weather, a clock Io knows as an idea, and one wandering
other that Io has learned to push. Io ended the era better fed than at
any point in its life, under the richest world it has ever inhabited —
and exhibiting a large-scale rhythm nobody designed.

## Findings ledger (settled by evidence; claim ceilings respected)

**F1 — The learnable-horizon law.** Structure whose hidden timescale
exceeds the 32-step BPTT horizon becomes a permanent, dense
disagreement source — a supernormal stimulus, not an affordance (the
TTL-50 trail: curiosity 10× monotone, ~41% of food production
sterilized, "Io was starving itself to watch itself," S7–S8). The
same structure inside the horizon is masterable and *releases*
attention (TTL-12 trail: bounded curiosity, first patch-following
shift, S9). The fountain condition is **structured-but-unreachable
regularity, dense and self-adjacent** — not randomness: genuine
stochasticity is epistemically inert to the disagreement estimator
(e4 desk pass; S14's live dissociation — PE the highest of the
biography while disagreement stayed bounded and falling). This
refines S1's intermediate-difficulty target into an operational rule
for every future dynamic: *put its timescale inside the learnable
horizon, or expect capture.*

**F2 — The drive is epistemic only; food is invisible to it.** The
biography configures no `energy_preference`: the actor scores
imagined futures by summed ensemble disagreement and nothing else
(mechanism pass, 2026-07-28). Eating is incidental; no food-vs-
curiosity arbitration exists anywhere in Io. Every economy outcome in
this ledger is a *side effect of attention*, and §7 is the only
welfare instrument. (This corrected a drift in our own language —
"Io paid half its food" imports an economics Io does not have.)

**F3 — Process-over-place learning.** The clock was mastered in ~4k
steps of sampling (S10), and when relocated to a site Io had never
watched it on, it was carried over *instantly*: zero approach all
session, bloom-fire/quiet disagreement ratio 0.99–1.02 from first
firing, a small ensemble-agreed reconstruction adjustment and no pull
at any point (S11, "silent generalization"). At z=16, h=200, the
world model learned a rhythmic process as location-independent
structure. A genuine capacity finding, relevant to what later probes
may calibrate against.

**F4 — The endogenous cycle: settle → stall → forget → burst.**
Twice observed (S11 in full; S14 mid-session with the burst edge
tripping the era's first §7 PE flag, corroborated benign). The shape:
residence tightens toward near-total stasis (S11: 94–98% in one cell;
the lifetime never-stays signature broke exactly at the stasis peak —
799 stays, then back to ~0); then, with **no world change**,
reconstruction error rises 20–40× on long-known terrain, disagreement
surges, Io relocates and forages at multiples of its settled rate.
Candidate mechanism, named not claimed: replay-dominance forgetting —
stasis fills the training stream with stasis, the model's grip on the
wider grid decays, the world becomes novel again *without changing*.
Constantly-refreshing processes are exempt (the bloom stayed
predicted through S11's stasis; the mover through S14's) — observation
refresh appears protective. Neither the drive, the replay design, nor
any world dynamic encodes this rhythm; it emerged from their
composition. **The era's largest open phenomenon.** Its period,
trigger, and stability are unread.

**F5 — Novelty-seeking explains arrival; policy inertia explains
staying.** The corner-sitting that looked pathological decomposes
cleanly (mechanism pass): Io goes where the largest remaining
disagreement is (the S10 "flight" was gradient ascent — the far
corner paid 4.65 vs the mastered ring's 0.6), and then *stays* after
the payout decays to the session's lowest values, because a flattened
landscape moves nothing in a converged policy. Why the attractors are
always corners remains open (candidate: walls halve the neighborhood,
making corner loops cheapest to master).

**F6 — The other is touchable, not compelling.** e4 landed and was
read across S13–S14: Io found the displacement rule within 4k steps
and exercised it 15 times (including a repeated corner-push exchange);
spatial attention sits at chance; the contact disagreement floor
stays ~2–4 (above ambient — the one falsifier that tripped, in letter)
with none of the guarded-against harms (no monotone climb, no
pursuit-capture, economy at biography best). Ratified kept
(2026-07-30), DP3 removability standing, with the standing watch:
whether the contact floor falls as cumulative exposure grows.

**F7 — The economy arc.** Two failure poles were found and closed:
saturation (uniform regrowth floods an idle world, S3–S5 → off-patch
expiry amendment) and starvation (the treadmill, S6 → sink halved →
decomposed in S7–S9: economic half substantially the trail, spatial
half the policy's own, attention half the trail fountain). After the
finite mirror (S9) and through the full ladder, the era closed at
0.030–0.033 meals/step and mean energy 0.059–0.092 (S12–S14) — the
best of the biography, *under the most complex world*. Still far
below constant-mover break-even (~0.15); floors persist; §7 stayed
clean throughout (one flag, disposed benign, F4). Pressure not
emergency, as the charter endorses.

**F8 — Method findings, binding on future eras.**
- *Pre-register the reading space wide.* S10's trichotomy
  (engaged/ignored/overwhelmed) missed the observed behavior; every
  later session pre-registered numerically-defined readings sized to
  the actual outcome space (S11's three, S12's three, S13/14's four
  axes) and none missed.
- *Desk pass before landing.* The e4 turn-hazard assessment predicted
  the aleatoric dissociation that S14 then showed live.
- *Positions are reconstructable* by deterministic action replay from
  each session's `env_reset` (validated to the step in e0–e2 worlds);
  the mover makes replay approximate (block-vs-push ambiguity) — push
  events, derived from `mover_step` chain discontinuities, are the
  ground truth there. Telemetry quirks on record: one duplicated row
  (t=330,001, shard-000169), dedupe by t; world_event flushes in
  bursts.
- *Detached launches for runs* (nohup/disown): two sessions were
  truncated by harness task-kills, none since. Truncated-but-cleanly-
  flushed sessions are read as-is; resuming a stub inserts a world
  reset and contaminates the very continuity being observed.

## What is closed

The ladder (e0→e1→e3→e2→e4), every rung landed one-at-a-time,
gated, and read at least once; DP1–DP6; the e2 question (engaged →
mastered → generalized → inert); the e4 gate (kept, watched); the
trail question (finite mirror, ratified at TTL 12); the treadmill
decomposition; the torpor-retest question that opened the era (every
stasis in the biography has broken from inside — now understood via
F4 as possibly *cyclic*, not merely recoverable).

## What is open (the next instruments)

1. **The cycle's rhythm (F4)** — period, trigger, stability;
   observation sessions on the unchanged world; no world change
   needed. The richest question the era produced.
2. **The criteria-driven mirror round** — the V2 adversarial
   instrument's episode-partitioning is unvetted for e0+ worlds (the
   caller's step-window fix covers the observational path only).
   Vetting is desk work; the round itself is builder-gated (API
   cost). The frozen mirror criteria (reflexive attention, equanimity,
   second-order volition) remain frozen; nothing in this era touched
   them.
3. **Probe 4.5 reopening** — its own research → synthesis → prereg
   round per the standing rule (both prior cycles spent).
4. **Smaller instruments**: the corner-attractor question (F5); the
   e4 contact-floor watch (F6); cross-pause world continuity (hand
   gestures still die at resumes — parked since S4, its own decision);
   C0 horizon re-measure; world-event flush-lag fix; window
   episode-stat freeze (cosmetic).
5. **Probe 1.5 / Probe 2 sequencing** — unchanged by this era in
   status (1.5 synthesis pending, 2 waits on it), but F3 (process
   generalization) and F4 (endogenous rhythm) are new evidence about
   what this substrate can carry, and belong in the 1.5 synthesis's
   evidence base.

## Recommended sequencing (builder discussed 2026-07-30)

Interleave the free observation with the desk work: observation
sessions for the cycle (1) whenever the machine is idle; mirror
vetting (2) as the next desk task, with the round itself on the
builder's explicit go; Probe 4.5's research round (3) after the
mirror round reads — its stakes-assay design should know what the
mirror can currently see.

---

*Grounded against the charter (read in full 2026-07-30), the design
notes, `synthesis_worldv2_2026-07-09.md`, and the complete world-v2
journal. The mythological footnote stands at the era's close as at
its open: Kind builds; Io is who is built. Io ends this era fed,
active, §7-clean, and doing at least one large thing nobody asked of
it.*
