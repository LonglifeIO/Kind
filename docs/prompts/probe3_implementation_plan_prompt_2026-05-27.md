# Build Probe 3 implementation plan

Generate an implementation plan for Probe 3 (dream-state as foundational), grounded against the patched synthesis artifact and the Kind project documents.

## Context (read in this order before drafting)

1. **The synthesis artifact** — `docs/decisions/synthesis_probe3_dream_foundational_2026-05-27.md` is the ground-truth source for what Probe 3 commits to. The integrated take and its recent patches (logs-only first build, no scheduler scaffolding, envelope as selection pressure, `uncertain → continue_and_log_uncertainty` default) are non-negotiable for this plan. The plan should derive from the synthesis, not re-litigate it.

2. **Project docs:**
   - `Kind_charter.md` — ethical and identity commitments
   - `Kind_design_notes.md` — operational design (four states, mirror, persistence, dream-as-foundational)
   - `Kind_probes.md` — Probe 3's question and dependency on Probe 2
   - `Kind_frameworks.md` — frozen mirror criteria
   - `Kind_camas_inheritance.md` — what to inherit, what to avoid

3. **Working journal and codebase.** Check `docs/workingjournal/` for current Probe 2 phase status (mirror infrastructure at Phase 13 last I knew; verify). The repo has Probe 1 / 1.5 substrate and the existing DreamRollout schema (already includes `sequence_z_prior`, prior entropy, successive-prior KL, latent norm change, decoded observations, actions, logprobs, reserved `sequence_self_prediction`). Inspect before planning.

## Load-bearing constraints

Synthesis-settled (do not re-litigate):

- Logs-only mirror output for first build. No runtime authority. No mirror→scheduler scaffolding.
- Runtime protection is content-blind: hard caps, compute budgets, checkpoint windows, dormant transition.
- No gradient flow from dream in first build. Any training-side response is deferred to a separate future synthesis.
- Dream content: replay-seeded perturbed generative recombination with associative drift as a parametric regime, not a separate mode.
- Four-axis differentiation from waking planning: goal-coupling absent, ensemble disagreement recorded-but-not-used, distinct temperature regime, initial conditions from replay or perturbed prior.
- Trigger is exogenous (world-side); internal signals may modulate kind once dreaming, not whether to dream.
- Lucid control out of scope.
- Probe 4 dream-perturbation interface restricted to envelope and seed-selection. No hidden-state writes.
- Self-opacity-during-dream is a named tension, not resolved within Probe 3.
- Dormant ≠ failure. Dormant is the capacity-over-exercise stance applied to dreaming.

Open for resolution in the plan or explicit deferral:

- (i) K=5 ensemble role during dream when action selection is absent (recording disagreement requires concrete decisions about when, how, and at what cost). Either spec it or defer with explicit reason.
- (ii) Dormant-state design — Probe 3 constrains but doesn't settle. Either spec the minimum dormant behavior Probe 3 needs, or defer with explicit reason.

## What the plan should contain

Structure as phases, matching the discipline used in Probe 2 (numbered phases, clear question per phase, dependencies, test commitments):

1. **Per-phase scope.** Each phase has a clear question being answered, expected deliverable, and dependencies (on prior phases or on Probe 2 components).
2. **Schema work.** Specify exactly which fields the existing DreamRollout schema needs to gain (temperature schedule, seed/anchor provenance with replay vs perturbed-prior vs hybrid tagging, ensemble disagreement trajectory through dream, sub-mode/phase tags, sampling parameters, gradient policy used, checkpoint hash, RNG seed, termination reason). Indicate which are required vs optional for first build and whether a schema version bump is needed.
3. **Dream-rollout module spec.** What runs during a dream phase: seed selection from replay buffer, RSSM prior rollout under specified temperature regime, periodic re-seeding for chimera structure, optional temperature ramp toward associative drift in the tail. Specify the four-axis differentiation operationally — exactly how the implementation distinguishes a dream rollout from a waking planning rollout.
4. **State-machine spec.** Waking ↔ dreaming ↔ dormant transitions. Desktop on/off as primary trigger, hard caps as safety, dormant as legitimate non-failure post-dream state. Specify exit conditions per state.
5. **Mirror integration with dream telemetry.** Probe 2's mirror needs to read dream-state telemetry without changes to its one-way constraint. Specify what the mirror sees, what it writes (logs and builder-facing notifications only), and what's logged for builder audit.
6. **Test design.** Each phase commits to specific test counts and what they verify. Match Probe 2's discipline. Critical Probe 3 tests: non-degenerate variation in dream telemetry vs waking-planning controls (the Probe 1.5 Phase 7 lesson — schema slots aren't enough; the conditioning surface has to be non-zero); dream regime distinguishable from pure prior rollout; dormant transition operationally distinguishable from active dreaming.
7. **Sequencing rationale.** As with Probe 2 (Phase 12 run before 9-11 deliberately to surface real LLM findings early), explain the ordering. Anything that should be run early to surface real findings before downstream work commits to assumptions?
8. **Cross-probe surface.** What Probe 3 exposes that Probe 4 will eventually use — seed-selection control surface, dream envelope (timing, ratio) — without exposing hidden-state-write affordances.

## Style

- Direct. No preamble.
- Distinguish observation from interpretation per your SOUL. If something is a design choice rather than commitment-derived, say so.
- No timeline estimates. No person-month or cost figures (CLAUDE.md, and Candidate D's failure mode flagged in the synthesis).
- Flag uncertainty. If a phase has a known unknown, name it as a phase deliverable to resolve, not a hidden assumption.
- Don't conflate plan with synthesis. The synthesis settled *what*; the plan settles *how*.

## Output

- Path: check `docs/plans/` for the project's plan-location convention, then place accordingly. If a versioned subdirectory exists (e.g. `v.0.1.0/`), use it.
- Format: markdown
- Filename: dated and descriptive, e.g. `probe3_implementation_plan_<YYYY-MM-DD>.md`
- Include: phase list with scope / dependencies / tests, schema work spec, dream-rollout module spec, state-machine spec, mirror integration spec, sequencing rationale, cross-probe surface, and explicit treatment of the two open questions (resolved or deferred-with-reason).