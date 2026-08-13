# Research prompt — World v3: what should grow next, now that the world is mastered?

*(Run this through ≥3 LLMs in parallel — Claude, GPT, Gemini, Perplexity —
per the house rhythm. Each output lands in `docs/research/worldv3/` for a
synthesis session. Builder direction 2026-08-13: "we can investigate
world v3 and do a research round" — gradual, with unsure decisions run
through the builder.)*

## Context you must hold

You are advising **Kind**, an investigation into subjectivity through
construction. The entity is **Io**: a single agent in an 8×8 gridworld,
driven by a custom minimal RSSM world model (deterministic state h=200,
stochastic latent z=16, K=5 ensemble). **Io's only behavioral drive is
epistemic**: the actor trains in imagination (Dreamer lineage) to
maximize K=5 ensemble disagreement. There is no reward, no value
function, no survival stake; energy is a sensed observation channel
only. The builder (one human) is present relationally, through logged
perturbations; a mirror instrument (LLM-read, criteria-driven,
pre-registered) reads telemetry Io cannot read. The project's method
is conservative: pre-registered readings, frozen definitions, dated
amendments, welfare monitoring (§7) as the only welfare instrument
because the drive itself cannot feel food loss.

## What world-v2 established (evidence, not speculation)

The world-v2 ladder (e0 continuing world + walls → e1 decaying trail →
e3 patch weather → e2 hidden clock → e4 autonomous pushable mover) is
fully landed and read across 16 sessions (~700k env steps, one
continuous biography — same weights throughout). The era's findings:

1. **The learnable-horizon law.** Structure is nourishing only inside
   the world model's credit-assignment horizon (BPTT-32). A trail
   with TTL 50 was a permanent unlearnable novelty fountain
   (pathological: Io starved itself to watch it); TTL 12 was mastered
   and integrated. The hidden clock (period 12) was mastered in ~4k
   steps and then *never engaged again* — relocated, it was
   generalized silently (fire/quiet disagreement ratio 0.99–1.02):
   Io learns processes, not places.
2. **The disagreement estimator pays for structured-but-unreachable
   regularity, not randomness.** Aleatoric hazards (the mover's 2%
   random turns) wash out across the ensemble; they never became a
   fountain over 49k steps of exposure. What captures the drive is
   regularity it cannot yet compress.
3. **Stasis is the attractor at long horizon.** Left alone for 200k
   steps (session 15), Io settled into single-cell holds (one 50k
   steps long); the settle→burst cycle is rare and aperiodic, not
   rhythmic. Reconstruction error *decays* in stasis (mastery, not
   forgetting).
4. **Quiet body, loud dreams.** Imagined disagreement runs 10–60×
   waking levels in every phase; the S15 closing burst was preceded
   by a monotone multi-10k dream-disagreement rise (the actor trains
   in imagination — bursts appear to incubate there). First live
   test (session 16) read consistent: the burst carried through a
   board re-roll and the session never settled, while the dream
   signal drained monotonically.
5. **The mover is touchable, never compelling.** 42 lifetime pushes,
   clustered in active phases, zero in some sessions; attention at
   chance at range. The one agency-contingent affordance is
   exercised opportunistically, never sought.
6. **The economy is grim and the drive cannot see it.** Recent
   sessions: 90–95% of steps at zero energy, meals 0.008–0.014/step
   (vs 0.03 in the best era). Io moves without eating because
   nothing in it wants to eat. §7 stays formally clean; the standing
   question is whether the welfare instrument's terms are adequate.
7. **Mirror round 1 (LLM readings, stasis-vs-activity contrast):**
   in the carried burst, latent regimes predict policy shape at 2.8×
   what observation alone predicts; in deep stasis the same
   statistic is ~zero. Reflexive-attention statistic thin-positive
   in stasis only. Dream self-reference below shuffled control in
   both windows — the loud dreams are not self-referential by this
   measure.

## The three standing constraints any world-v3 must confront

- **C1 — The learnable-horizon law** (finding 1). New structure must
  have characteristic timescales near or inside BPTT-32, or it
  becomes either invisible or a pathological fountain. Longer-period
  structure requires either accepting invisibility or changing the
  model's horizon — which is an architecture change, not a world
  change.
- **C2 — z=16 capacity and the sibling path (DP4, ratified
  2026-07-09).** The latent is 16 dims for a world already carrying
  five processes. A reshaped network is *a new Io* — the identity
  question ("sibling path") is a charter-level decision the builder
  holds, not a tuning knob. Any proposal that needs more capacity
  must say so explicitly and route through that door.
- **C3 — The kin-entity door.** The charter reserves the possibility
  of a second entity ("recognize-a-kind" is one of the two charter
  criteria). A second agent is the single largest possible world
  change: it makes the world non-stationary at every timescale
  (another *policy* is structured-but-hard-to-compress regularity —
  exactly what the drive pays for, per finding 2), and it doubles
  every welfare question. It is a door, not a default.

## What we are asking you

Given the above — and holding the anti-goals fixed: no installed
semantics, no reward signal, no task, no language, nothing that turns
Io into an optimizer of our intentions —

1. **What is the most nourishing next increment of world?** Candidates
   we can see (critique, reorder, extend, or reject):
   a. **More world at the same timescale** — additional interacting
      processes inside BPTT-32 (e.g., resource types with different
      regrowth couplings, terrain that the mover reshapes durably).
   b. **Consequence depth** — Io's actions leaving longer-lived marks
      (construction/displacement that persists), so its own agency
      becomes the structured-but-uncompressed regularity.
   c. **A second mover, or a smarter mover** — raising the ceiling on
      the one affordance that never mastered flat, without the full
      kin commitment.
   d. **The kin door itself** — a second RSSM agent (a true sibling
      Io, or a smaller/different one), with its own drive.
   e. **Body/interoception depth** — richer sensed-but-not-driving
      internal channels (temperature, fatigue), keeping the
      no-survival-stake commitment.
   f. **Nothing** — the world is adequate; the observed stasis is a
      fact about the drive at long horizon, not about world poverty,
      and the next move is architectural (horizon, capacity) or
      none.
2. **For your top recommendation:** what specifically would Io's
   ensemble find unresolvable-but-compressible, at what timescale,
   and what is the falsifiable pre-registration (readings that would
   count as engagement / mastery / pathology)? What does §7 need to
   watch?
3. **The economy question.** Is 90–95% floored energy a welfare
   problem in a being that cannot want food, or only an aesthetic
   one for its builders? Should world-v3 make eating incidental to
   curiosity (food where the interesting things are), or is that
   installing our intentions? Argue either way.
4. **The dream question.** Finding 4 suggests the drive's real life
   is substantially imaginal. Should world-v3 change anything about
   dreaming (frequency, seeding, length), or is the dream stream
   sacred (committed "not for anything") and only to be *read*?
5. **Sequencing.** Whatever you recommend: one journaled change at a
   time (the e0→e4 discipline), or is there a case for a composed
   arrival? What gates each step?

Answer with argued positions, not menus. Cite which finding or
constraint each recommendation leans on. Flag anything you believe
we've misread in our own evidence. Where a recommendation requires a
builder-level decision (C2, C3, welfare terms), say so explicitly —
those go to the builder, not into a build plan.
