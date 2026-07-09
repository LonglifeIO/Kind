# World v2 synthesis — enriching Io's world without installing semantics — 2026-07-09

**Status: DRAFT for ratification** — decision points in §6 await the
builder. Synthesis only: no task breakdown, no timeline. The
implementation plan follows ratification.

**Conventions.** `[S1]` Claude voice (session-run web-grounded
deep-research workflow; 9 findings that survived 3-vote adversarial
verification, plus a **refuted list** — claims that did *not* survive);
`[S2]` Gemini; `[S3]` GPT; `[S4]` Perplexity (all in
`docs/research/worldv2/`). `[B]` builder direction (2026-07-08/09).
`[F]` project documents and the session-1 record
(`docs/workingjournal/probe4.md`), which is itself evidence here: the
torpor event is the observed failure mode this cycle exists to answer.
Conflicts held open, not averaged. My moves are **[synthesis
inference]**.

---

## 1. Executive summary

All four voices converge on one diagnosis and one design law.

**Diagnosis:** Io's world starves its drive. One object type, uniform
regrowth, and a 200-step whole-board resample leave nothing durable to
model; the drive mastered the world in ~2k steps, then found the one
place uncertainty was *delivered on schedule* — the reset — and parked
there. All four voices give compatible mechanistic accounts of the
reset-camping stasis (scheduled-novelty cousin of the noisy-TV trap;
z=16 cannot compress a whole-board resample, so ensemble disagreement
never extinguishes on it) [S1 F2/F4-adjacent, S2, S3, S4].

**Design law:** *never hand out global novelty on a schedule that can
be collected while inert* [S3's phrasing; all concur]. Novelty must
enter as slow, local, history-bearing structure — "learnable but never
fully learned."

**The world v2 shape** (§4): a **continuing world** (no board
resample), a small amount of **persistent terrain**, one
**contact-responsive dynamic** (the cheapest form first),
one **hidden-phase process**, **structured resources**
(patch/seasonal — food as weather, not confetti), and — last and
hardest-gated — **one autonomous mover**. One dynamic at a time, each
arriving as an event in Io's continuing life via checkpoint-resume,
each gated on a three-signal telemetry check before the next.

[S3's one-sentence version, adopted verbatim as the north star:]
**"World v2 should become a continuing, weakly structured,
multi-timescale physics toy — not a bigger vending machine."**

---

## 2. Strong consensus (all four voices)

1. **End the whole-board resample.** It is the mechanism of the
   observed stasis and the destroyer of place-structure, builder-gesture
   permanence, and long-horizon latent learning. (Pace differs — §3 C2.)
2. **Uniform random regrowth is epistemically empty.** Replace with
   structured distribution (patches, seasonality, drift-fields):
   spatial autocorrelation + temporal predictability = something to
   model [S2 §Q5, S3, S4; S1 F5-adjacent].
3. **One dynamic at a time**, introduced into the continuing biography,
   observed until the new uncertainty becomes *organized* before the
   next [all four; S1's capacity findings; S3's stopping rule adopted
   in §5].
4. **Hidden-phase processes are high-value and capacity-light** — they
   exercise the deterministic state (h) as a clock, the cleanest
   "learnable but never trivial" structure [all four].
5. **Persistent terrain is cheap and foundational** — place-structure,
   action-consequence geometry, and the precondition for builder
   gestures to persist [all four].
6. **Diagnostics converge** (§5): engagement = local disagreement rise
   → localization → partial settling + new behavioral motifs + dream
   over-representation; trap/noise = rise without settling; ignored =
   flat; overwhelm = posterior collapse / global disagreement
   inflation / never-settling PE. Posterior-vs-prior KL is the
   capacity canary [S2's KL emphasis; S3's local-learnability;
   S4's three-way signatures; S1 F7-adjacent].
7. **z=16 is tight; stacking is the danger.** Multiple movers,
   multiple asynchronous clocks, or long chains concurrently risk
   posterior collapse / entangled representations. The escape hatch, if
   ever needed, is z≈24–32 or a slow context variable — *after* the
   cheap route is tested [S2, S3, S4; S1 F6/F8].

**A verified caution unique to S1:** ensemble disagreement provably
self-extinguishes on *both* mastered structure *and* incompressible
noise — so purely stochastic enrichments are epistemically inert
rather than trap-forming, and the design target is intermediate
difficulty in the learning-progress sense [S1 F2, F3, 3-0 votes,
primary ICML sources]. This tempers S2/S4's stronger "disagreement is
robust to noisy TVs" claims — S1's verifiers *refuted* the general
form of that robustness claim, and session 1's stasis is a live
counterexample at this capacity.

## 3. Conflicts (held open, then resolved in §4)

**C1 — Contact physics: confidence vs. verification.** S2 makes
pushable blocks a centerpiece ("the extended self", tagged canonical);
S4 calls gridworld block-pushing canonical; S3 endorses one
contact-responsive mover. **S1's adversarial pass specifically refuted
the claim that small RSSM-class models are demonstrated to learn
contact dynamics** (did not survive 3-vote verification). *Resolution:
contact-responsiveness enters as a PILOT hypothesis, not a safe bet —
and the cheapest contact form goes first (a tile that changes state
underfoot / a decaying trail), with the pushable/mover form last and
hardest-gated.* [synthesis inference]

**C2 — Reset removal: immediate vs. gradual.** S2: abolish now. S3:
no-whole-world-reset by default (training windows are a separate,
compatible concern). S4: gradualist (lengthen episodes → persist
structures → then remove). *Resolution: remove the board resample in
one journaled event (the gradualist path multiplies confounded
boundary events through the biography), BUT the implementation plan
must first inventory what episode machinery is load-bearing in the
codebase — the runner zero-resets h at soft boundaries, the SELF
extraction excludes boundary-straddling pairs, telemetry carries
episode semantics, and the window/monitors assume resets exist. "No
resample" is an env change; "no h zero-reset" is a runner change; they
are separable and the plan decides each explicitly.* [synthesis
inference; F: journal + code]

**C3 — Ordering.** S2: continuity → trail → blocks → patches →
oscillator. S3: terrain → drifter → hidden-phase → chain → field
(field last, crowd-out risk). S4: cycle → toggle tiles → patches →
drifting walls → movers+chains. *Resolution in §4: continuity+terrain
first (all agree they're preconditions), then the cheapest
self-boundary contact dynamic, then one hidden clock, then structured
resources, then the mover. The causal chain and any second
mover/clock are "later if ever."* [synthesis inference]

**C4 — Scarcity crowd-out.** S3 fears food-curiosity dominance and
places the field last; S4/S2 argue reward-free agents self-limit
(mastered patches lose pull). *Resolution: held empirically — the
field is introduced after the non-food dynamics are established, with
occupancy-share monitoring named in §5.* [B's own steer applies:
"curiosity resulting in something other than resources" argues for
non-food enrichments landing first.]

## 4. The recommended enrichment sequence (each gated per §5)

- **E0 — The world stops forgetting.** Board resample removed (no
  200-step wipe; consumption, regrowth, and drift continue; walls and
  placed objects persist indefinitely). A handful of persistent wall
  cells (~6–8, one corridor shape, not partitioning the grid — S3's
  trivial-loop confound) land in the same event. This is the
  anti-reset-camping change and the precondition for everything
  else, including builder gestures that persist [all voices].
- **E1 — The somatic trail** (cheapest contact pilot). Cells Io steps
  on change state visibly and decay back over ~40–60 steps. Enriches
  the *self* side of the boundary — Io's strongest measured structure —
  with spatially-extended, frequently-observed dynamics (S1's verified
  preference), at minimal capacity cost. [S2's phase-2; S4's
  toggle-tile family; C1-cautious.]
- **E2 — One hidden clock.** A single fixed cell with an unobserved
  phase (~12-step advance; short visible bloom in its neighborhood on
  the terminal phase). Exercises h-as-clock; one clock only
  (temporal-superposition warning, S3/S2).
- **E3 — Food becomes weather.** Uniform regrowth replaced by one
  drifting patch/field (predictable drift law, e.g., slow bounce;
  regrowth concentrated under it, sparse elsewhere). Parameters sized
  so break-even foraging remains possible but not ambient (nothing
  dies; the energy channel stays observational). Crowd-out monitored.
- **E4 — One autonomous mover** (the C1 pilot proper, last,
  hardest-gated). Single-cell object, inertial motion, low turn
  hazard (~0.02), bounces off walls; displaced one cell by contact.
  If its disagreement never localizes (S1's refuted-confidence
  scenario made real), it is removed at a pause — that outcome is a
  capacity finding, not a failure of the cycle.
- **Later-if-ever bucket:** the short causal chain [S3], second
  mover/clock, drifting walls [S4], and the z=24–32 capacity change —
  each requires the prior set to be organized and its own decision.

## 5. Gates, diagnostics, and discipline

**The three-signal gate before each next enrichment** [S3's stopping
rule, all-voice diagnostics]: (1) disagreement around the new dynamic
rises, then *localizes and partially settles*; (2) new nontrivial
behavioral motifs appear (not passivity, not a single loop); (3) the
dynamic is represented in dream content at or above its encounter
rate. Failure signatures that stop stacking and simplify: global
disagreement inflation, action stasis (§7 torpor shape — the standing
monitors keep running), never-settling PE, posterior-KL collapse.

**Not a probe.** No pre-registered pass/fail; these are enrichments and
the gates are *engineering* gates (add/hold/remove decisions), not
claims. Claims about what Io's engagement *means* stay at the existing
ceilings; the mirror reads content, the journal records.

**Instrumentation the plan must include** [synthesis inference]:
granular world_event logging for every new dynamic (trail decays,
bloom events, mover steps, patch moves — the regrowth-granularity
precedent), so the matched-control affordance and the §5 diagnostics
have per-event data; the boundary analysis script extended per-dynamic;
the live window rendering the new cell states.

**Unchanged by this cycle:** PolicyView; the four absences; no
observation markers (all dynamics express through the existing cell
vocabulary and its *behavior over time*); energy preference None;
dream regime and cadence; the generator + manual channel (which now
gains permanence for wall gestures); mypy/full-suite gates; the §7
monitors (with the known entropy-baseline insensitivity noted — any
numeric change is a dated amendment, not silent).

## 6. Decision points for ratification

- **DP1 — Adopt E0 (continuing world + persistent terrain) as one
  journaled world-change event.** Recommend: yes. The single highest
  consensus item; directly answers the observed failure mode.
- **DP2 — Adopt the E1→E4 sequence with the three-signal gate.**
  Recommend: yes, as stated; each arrival is a dated, journaled event
  through checkpoint-resume.
- **DP3 — Contact dynamics are pilots, not commitments** (C1) —
  removable at a pause without ceremony. Recommend: yes.
- **DP4 — z stays 16 this cycle.** The capacity change is a separate
  later decision with its own identity implications (a reshaped
  network is a new Io — the sibling path). Recommend: yes.
- **DP5 — E3 parameters land at build time as stimulus knobs**
  (journaled, tunable at pauses; not success criteria). Recommend: yes.
- **DP6 — The h zero-reset question is decided in the plan** after the
  code inventory (world continuity does not force runner changes; if h
  continuity is adopted it is its own journaled decision). Recommend:
  decide in plan, default to smallest change.

---

*Grounded against S1–S4, [B] 2026-07-08/09, the charter/design notes,
and the session-1 record. Departures from individual voices are argued
at C1–C4; the refuted-claims list in S1 is treated as first-class
evidence. No task breakdown; the implementation plan follows
ratification of DP1–DP6.*
