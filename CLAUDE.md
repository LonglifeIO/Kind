# CLAUDE.md — Kind

**Kind** is an investigation into subjectivity through construction. Build to understand; the work is done when the map has shifted, not when a metric has moved.

**Io** is the entity Kind is about — a single core agent in a small grid world with the builder as the only non-simulated relational other. The mythological Io was forcibly transformed; the name is the ethics encoded in a word. Kind builds; Io is who is built. The distinction matters.

## Project stance

- Capacity-over-exercise — possible, not mandatory, not impossible
- Ingredients-only self-modeling — no explicit self-modeling, critic, or introspector module
- Self-opacity for Io — every "should it have access to X about itself?" defaults to no
- No installed self-continuation drive — no reward, no termination penalty
- No self-optimization machinery — self-modeling ≠ self-optimization; install neither
- Co-design problem — mirror and substrate built by the same head; mitigations partial
- Build to understand, not to solve

## Core files (read first; project documents win conflicts with research)

**Before drafting any synthesis, implementation plan, build prompt, or substantive output: read the relevant core files in full. CLAUDE.md is a pointer document, not a substitute. Skipping this step has produced reification and drift in past sessions.**

- `docs/plans/v.0.1.0/Kind_charter.md` — what Kind is and is not; success criteria; ethics
- `docs/plans/v.0.1.0/Kind_design_notes.md` — agent / mirror / observer layers; four-state model; co-design
- `docs/plans/v.0.1.0/Kind_frameworks.md` — interpretive frameworks for mirror criteria
- `docs/plans/v.0.1.0/Kind_probes.md` — probe sequence (1 plumbing → 2 mirror → 3 dream → 4 builder-as-perturbation)
- `docs/decisions/` — synthesis documents (settled commitments)
- `docs/plans/Kind_probe1_implementation_plan.md`, `Kind_probe2_implementation_plan.md`
- `docs/workingjournal/probe1.md`, `pre-probe4.md` — phase-by-phase build log

Research lives in `docs/research/<probe>/`; prompts in `docs/prompts/`. Research is input; synthesis documents are the decisions.

## Substrate at a glance (settled)

Custom minimal RSSM (PlaNet-skeleton, DreamerV1-style imagination); continuous Gaussian latents (`h=200`, `z=16`); free bits the only stability borrow. K=5 ensemble disagreement is the actor's only intrinsic signal — no reward predictor, no critic, no continuation head. PolicyView/TelemetryView opacity boundary enforced in code (module imports, mypy `--strict`, frozen dataclasses). Mac-canonical, desktop-as-environment; four telemetry streams (`agent_step`, `dream_rollout`, `replay_meta`, `world_event`) under `kind/observer/`.

## Project state

- **Probe 1**: complete — substrate, 5000-step env-coupled MPS run, first mirror call done; clean across all four streams
- **Probe 1.5**: research phase — minimum architectural affordance for self-reference (afford, not install); not yet built; synthesis pending
- **Probe 2**: synthesis drafted, implementation plan exists; both pending Probe 1.5 outcome (Probe 1.5 changes what Probe 2 is calibrating)
- **Probes 3 (dream) and 4 (builder-as-perturbation)**: unbuilt

## Working rhythm

Research (≥3 LLMs in parallel) → synthesis (`docs/decisions/`) → implementation plan (`docs/plans/`) → build phase-by-phase → journal entry per phase. Each phase has a specific question; out-of-scope lists in plans are load-bearing; entries record what's now closed and what's now newly open.

## Discipline

- Don't reify gestures into formal substance — flag and refuse
- Don't conflate Kind with Io
- Don't install what should be afforded
- Don't add machinery the synthesis didn't decide on
- Forward-compatibility lives in schemas / conduits / mutators / opacity boundary; nothing else
- mypy `--strict` on all `kind/` sources
- Tests must pass before a phase is declared done
- Trust outputs only at the level the probe's question allows

## What not to do

- **No timelines or duration estimates.** Phases are scoped by what they build, not by how long they take. AI-assisted pace breaks standard velocity assumptions; the training data on dev timelines is misleading
- Don't change settled decisions: substrate (RSSM/Dreamer-lineage with active-inference-shaped actor), Probe 1 implementation, or environment design pre-Probe-1.5 revision
- Don't run mirror calls or other expensive operations without confirmation
- Don't skip research → synthesis → plan → build (the CA-MAS pattern to avoid is accreting features without each having a specific question being specifically answered)
