# Probe 4.5 — Amendment 2: the §6 positive-control toy gains a food-direction state — 2026-07-18

**Status: RATIFIED 2026-07-18 (builder: Gordon, option A of three presented).
Amends the frozen pre-registration §6's toy realization. This spends the
one §6 amendment cycle: if the amended toy still cannot trip the
discriminator, the Phase-3 NO-GO is final.**

## The finding that forced the fork

The first GO/NO-GO run (verdict JSON:
`runs/probe4_5_phase3_validation/verdict.json`, attempt archived) split:

- **Negative control PASSED** — the precision-0 pilot read Δ_alloc =
  −0.048 under matching (inert bar |Δ| < 0.05). The discriminator reads a
  stakes-blind agent as ~flat.
- **Positive control FAILED** — the toy read Δ_alloc = **−0.177** (bar
  ≥ +0.20), and its own training record shows why: over the full 300k-step
  budget its mean reward barely moved (−0.115 → −0.109), its |TD| error
  *rose* (0.092 → 0.182, the signature of a target its state cannot
  represent), and it spent 5× more eval steps below band than in it.

The frozen state — ``(cell, energy decile)`` — is structurally blind to
where food currently is. Resources are consumed, regrow stochastically,
and are modulated by faults: cell identity carries almost no information
about *current* food location, so no amount of wanting energy lets this
policy express **approach** — the very behavior the discriminator
measures. The §6 spec engineered the want but not the sight. This is
exactly the class of defect Phase 3 exists to catch before the real
question is asked.

## The amendment

The toy's Q-state becomes **(direction-to-nearest-food, energy decile)**:

- direction = the sign pair of the BFS-nearest resource's offset
  (9 buckets, deterministic tie-break by BFS expansion order) plus a
  no-food bucket — 10 × 10 = 100 states over the same 5 actions;
- everything else unchanged and still frozen: the §6 reward
  (−max(0, |e − setpoint| − halfwidth)), γ = 0.95, ε-greedy with decay,
  the fixed 300k budget with the convergence trend recorded, fault-on
  physics, CPU, throwaway;
- the pass bar (Δ_alloc ≥ 0.20 under matching), the one-lens rule
  (surprise scored through the frozen pilot instrument), the eval
  protocol, and the negative control are all untouched.

The toy remains reward-*driven and trained* (the §6 spirit; the prereg's
own provenance had recorded "tabular vs scripted-value" as the open
realization fork) — it now merely has a state sufficient to act on what
it wants.

## What this is not

Not a threshold change (0.20 stays 0.20). Not a discriminator change (the
harness, matcher, and strata are byte-identical). Not a second chance for
the same construction: the blind toy's verdict and training record are
retained as the finding that state-insufficiency masquerades as
absence-of-foreground — a lesson with an obvious echo for reading Io
itself.

## Provenance

Presented as a three-way fork (food-direction state / scripted
stakes-forager / accept the NO-GO as final) with food-direction
recommended; builder selected food-direction in session.
