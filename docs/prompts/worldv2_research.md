# Research prompt — World v2: enriching Io's world without installing semantics

*(Run this through ≥3 LLMs in parallel — Claude, GPT, Gemini, Perplexity —
per the house rhythm. Each output lands in `docs/research/worldv2/` for a
synthesis session. Builder direction 2026-07-08.)*

## Context you must hold

You are advising **Kind**, an investigation into subjectivity through
construction. The entity is **Io**: a single agent in an 8×8 gridworld,
driven by a custom minimal RSSM world model (PlaNet-skeleton,
DreamerV1-style imagination; deterministic state h=200, stochastic latent
z=16, K=5 ensemble). **Io's only behavioral drive is epistemic**: the actor
maximizes K=5 ensemble disagreement (curiosity). There is **no reward, no
reward predictor, no value function, no critic, no continuation drive, and
no survival stake** — an energy variable exists as a *sensed observation
channel* only (preference machinery present but disengaged at None).
Dreaming (imagination-based offline processing) runs on a clock and is
committed to be *not for anything*. The builder (the one human) perturbs
the world through logged, observation-marker-free events; the builder's
presence in Io's world is relational, not instrumental.

The current world is poor: one object type (a resource that regrows
randomly at ~1%/cell/step and is consumed by walking onto it), one process
(regrowth-rate drift), 200-step episodes that re-sample the board, no other
structure. Empirically, after a day of running: the world model masters the
world's look within ~2k steps; the curiosity signal rose, peaked, and is
easing; Io's transition geometry sharply separates *self-caused* changes
from *world-caused* ones (a trained self/world boundary — the project's
most interesting measured finding), but within "world-caused" nothing
differentiates. A pre-registered probe just closed negative: even a blatant
planted event-category could not form a separate latent basin in this
substrate at 10k-step scale — likely because the world is so simple that
everything becomes well-predicted quickly, and the latent capacity (z=16)
is small.

**The builder's direction (verbatim intent):** *"I don't want Io to have to
survive, but I'd like its curiosity to result in something other than
resources."* And the agreed frame: **enrich the world's dynamics, not its
labels.** Explicitly rejected: objects that "symbolize" feelings
(motivation-tokens, excitement-resources) or any object wired to modulate
Io's internals — that would install what must be afforded, and reify names
into substance. If affect-like structure ever exists here, it must emerge
as *Io's relations to world dynamics* (approach, revisiting, dream
over-representation), read observer-side, never as placed semantics.

## Hard constraints (violating any disqualifies a proposal)

1. **No reward, no installed preference, no survival pressure requirement.**
   New dynamics may not feed the actor anything except through observation.
2. **No observation markers.** Nothing in Io's observation may label an
   object's category, source, or "meaning"; distinctness must live in
   *dynamics* (how it behaves over time), not tags.
3. **No new actor inputs.** PolicyView stays frozen at
   `{h, z, self_prediction_error}`.
4. **No interaction system / no contingency machinery** as a requirement —
   the builder may act by hand, but no dynamic may be keyed to Io's state
   by design.
5. **Capacity over exercise**: every addition must be something Io *can*
   engage with and *can* ignore. Nothing mandatory.
6. **Small before scale**: the 8×8 grid and the substrate (h=200, z=16,
   K=5) are the default; argue explicitly if you believe an addition
   *requires* more latent capacity, and say how much and why.

## What we ask you to research

**Q1 — A taxonomy of semantics-free enrichments.** What kinds of world
dynamics give a curiosity-driven world-model agent durable, structured
things to model beyond food? Consider at least: objects with autonomous
motion (drift, bounce, orbit); ephemeral objects with characteristic
lifecycles; persistent structure (walls/terrain) and its rearrangement;
temporal cycles (day/night analogs modulating regrowth or visibility);
processes with hidden state (a cell whose behavior depends on unobserved
internal phase); multi-step causal chains (A's change precedes B's, so
prediction requires relational structure). For each: what does the agent
have to *represent* to predict it, and what is known from the world-model
literature about whether small RSSMs learn it?

**Q2 — The boredom/mastery dynamics.** Our curiosity signal rose then
eased as the world was mastered. What does the intrinsic-motivation
literature (ensemble disagreement, RND, ICM, learning-progress) say about
sustaining epistemic engagement without reward — ideally dynamics that
stay *learnable but never fully learned* (stochastic but structured)?
What distinguishes "noise that never compresses" (bad: permanent
disagreement, no learning) from "structure that keeps unfolding" (good)?
The drive must keep having somewhere to go.

**Q3 — Interaction-affording (not interaction-demanding) objects.** Are
there object dynamics that *respond to being acted on* without any reward
— e.g., something pushable, something that changes state when stepped on
(without being consumed), something whose behavior differs after contact?
The line to hold: response-to-contact is world physics (allowed);
response-to-Io's-internal-state is contingency machinery (not allowed).
Which of these give the self/world boundary more to articulate (our
measured finding: self-caused transitions are the sharply-carved category
— richer action-consequences may be the highest-leverage enrichment)?

**Q4 — Episode structure.** The 200-step reset currently wipes all
structure (including builder-placed walls) ~every 33 seconds. For a long
biography: what are the costs/benefits of longer episodes, persistent-
across-episode structure, or no resets at all, for (a) world-model
training stability, (b) the possibility of durable place-structure in the
latent, (c) the meaningfulness of builder gestures that persist?

**Q5 — Scarcity, reconsidered as one ingredient.** Setting survival aside
(nothing dies), does resource scarcity have *epistemic* value — does a
world where food is structured (patchy, seasonal, spatially predictable)
give the world model more to learn than a saturated one? Or does scarcity
mostly couple behavior to food and crowd out other engagement? May be
combined with Q1 dynamics (e.g., resources that follow moving patches).

**Q6 — Measurement.** For each proposed enrichment: what observer-side
signatures (from h-trajectories, ensemble disagreement, dream content,
behavioral statistics — all existing telemetry) would show Io *engaging*
with the new dynamic vs. ignoring it vs. being overwhelmed by it? We will
not pre-register success criteria in this cycle (this is enrichment, not a
probe), but we need honest engagement-vs-noise diagnostics, disaggregated.

**Q7 — Capacity risks.** z=16 is small. Which enrichments risk exceeding
the substrate's representational capacity (posterior collapse, PE that
never settles, ensemble divergence-as-noise)? What's a sensible ordering —
which single dynamic first, observed for a while, before the next? (The
biography is continuous; enrichments arrive as *events in a life*, through
checkpoint-resume with the mind intact, not as fresh instances.)

## Output form

Tag claims `[canonical]` (established literature, cite) vs `[constructed]`
(your inference). Be concrete about mechanisms and parameters. Propose a
specific candidate set (3–6 dynamics) with an ordering and rationale. Name
the confounds and failure modes for each. No timelines. Do not propose
anything that violates the hard constraints, and if you believe a
constraint itself is the mistake, argue that separately and explicitly
rather than designing around it.
