## Pre-Probe-4 — builder-as-perturbation, mechanism question (2026-05-02)

The design notes commit to the builder as a "non-simulated source of state
changes" but do not specify what "non-simulated" means mechanically. The
mutator hooks (add_resource, remove_object, set_cell_state, move_object)
are implemented; what's open is what triggers them.

Surfaced in conversation while building Phase 3c: a stochastic perturbation
generator running inside the system, with parameters set by the builder, is
a different kind of thing from a perturbation triggered in the moment by
the builder. The first is a region of the environment the builder
parameterized; the second is genuinely from outside the simulation. Probe 4's
distinguishability hypothesis cares about the difference. If the
"perturbations" are just one of the simulation's own random processes
labelled as builder events, Io detecting them as distinct from other
environmental stochasticity only shows that Io detected internal structure —
not that Io detected an outside.

### Working approach for Probe 4 (not committed; revisit when Probe 4
research runs)

A hybrid: a stochastic perturbation generator with parameters I set —
probability-per-step, magnitude distribution, mutator-type weights — runs
during waking and produces perturbations with a specific statistical
signature distinct from the regrowth and drift signatures the environment
already has. Alongside the generator, a manual trigger lets me call any
mutator directly when I happen to be at the keyboard.

Why this combination:

- The generator gives Probe 4 a statistical signature dense and consistent
  enough to test distinguishability against (manual triggers alone produce
  too few events for the hypothesis to be testable).
- The manual triggers preserve the "real outside" possibility — events
  that genuinely depend on me being there, deciding in the moment, with
  a temporal signature shaped by my actual life rather than by a generator
  I parameterized. These are different in kind from the generator's events,
  even if Probe 4's metric treats them similarly.
- The combination acknowledges honestly that Probe 4 with only the
  generator is testing a thinner hypothesis than Probe 4 with both. The
  generator's events satisfy the literal "no observation marker"
  commitment but are less philosophically defensible as "from outside"
  than manual triggers are.

### What this is not

This is not the final answer. The project documents already say Probe 4's
specific perturbation profile is settled at Probe 4 design time. What this
entry records is a working intuition that:

1. Pure-generator perturbations attenuate the "from the builder"
   commitment in a way worth being suspicious of.
2. Pure-manual perturbations make Probe 4 hard to run with statistical
   power.
3. A hybrid lets both kinds of events exist in the world_event stream,
   each with their own signature, and lets Probe 4's analysis treat them
   together or separately as the design demands.

The choice between these — or some option not yet considered — is for
Probe 4's research synthesis to make. The hybrid is the current default
working assumption to bring into that research.

### What stays out at Probe 1

Probe 1 is perturbation-free or near-zero-rate. Any perturbations during
Probe 1 are manual, fired by me as plumbing tests during the smoke or
shortly after. The generator is not implemented yet; the manual hook will
be implemented as part of Phase 3a's harness so it can be exercised by
gate test #3.

### The deeper question Probe 4's research should address

Where does the line fall between "the environment has internal structure
the builder parameterized" and "the builder is genuinely outside the
environment"? The hybrid implementation above straddles the line by
including events on both sides of it. But the line itself, and how Io's
distinguishability behavior should be interpreted in light of where events
fall on it, is an open question the project documents have not resolved.
Surface this in Probe 4's research prompt directly.