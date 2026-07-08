You are continuing the **Kind** project — an investigation into subjectivity through construction. The entity is **Io**, a small custom RSSM agent (PlaNet/DreamerV1-lineage) in a gridworld. We are building **Probe 4 (builder-as-perturbation)**.

**First, read `CLAUDE.md`. Then read these IN FULL before doing anything** — they are the authority, and CLAUDE.md requires reading the core files fully before substantive work (skipping this has caused drift before):

- `docs/decisions/synthesis_probe4_builder_as_perturbation_2026-07-07.md` — the synthesis. The probe's target is **three-way source-separation**: does Io come to model builder perturbations as a distinct third category (not-self, not-environment), separable from BOTH its own action-effects and the environment's own stochasticity? The bar is cleared by a **structural** signature (a distinct latent basin and/or dream over-representation), never by a mere prediction-error difference.
- `docs/plans/Kind_probe4_implementation_plan.md` — the plan. Its "Architecture grounding" facts are cited against live code; verify them as you touch each surface.
- `docs/decisions/probe4_preregistration_2026-07-07.md` — Phase 0, currently **DRAFT, NOT frozen**.
- Also: `docs/plans/v.0.1.0/Kind_charter.md`, `Kind_design_notes.md`, and the Probe 3.5 records `docs/decisions/probe3_5_verdict_2026-06-12.md` and `probe3_5_seek_classification_2026-06-12.md`.

Then do these two steps, **gated — do NOT start Step B until Step A is complete**:

**Step A — Freeze the Phase 0 pre-registration (co-design critical).**
The pre-registration is DRAFT. Walk me (the builder) through §9's six `[BUILDER]` items one at a time. For each, show the proposed value and its rationale and ask me to confirm or overwrite. **Do NOT choose the values for me** — the whole point of pre-registration is that I commit to what counts as success before any run; you filling them in would be exactly the co-design loop the charter warns against. Once I have confirmed all six, set the doc's status line to `FROZEN <today's date> (builder: Gordon)`, record any values I overwrote, and only then proceed.

**Step B — Build Phase 1 (per the plan's Phase 1). After freeze only:**
1. **Per-event internal-stochasticity logging** — emit a granular `world_event` (`source="environment"`, `event_type="internal_stochasticity_event"`) for each regrowth resource-addition, with the same payload shape as a builder `add_resource`, so the environment and builder classes are directly comparable per-event (today only a per-episode aggregate exists). Schema addition + version bump via the existing version-gated validator pattern; new frozen export.
2. **Memory-horizon eval harness** — inject a distinct event into a frozen world model, track the `h`-trajectory KL against an un-perturbed counterfactual until it decays; report the forward horizon and the truncated-BPTT ceiling. Observer-side, no gradient. This measures the number that sets the event rate later (Phase 2) — it is a knob, not a success threshold.
3. **Self-action-effect extraction contract** — specify and test the observer-side extraction of the SELF class (Io's own resource consumptions) from `AgentStep` (`action_t` + the `h_t` transition).

**Discipline (CLAUDE.md + the plan):**
- Read the core files IN FULL before writing anything.
- The **full** test suite + `mypy --strict` on all `kind/` sources must pass before the phase is done — never just the phase's new tests (the sink-routing lesson).
- **Functional naming only** in code: `perturbation`, `source`, `internal_stochasticity_event`, `trigger`, `memory_horizon`, etc. No `recognition` / `agent` / `kind-of-being` vocabulary in identifiers — that language stays in docs.
- **Do not touch**: PolicyView `{h, z, self_prediction_error}` (frozen; builder events carry NO observation marker — the pixel-equality gate `test_gate_perturbation_hook_logged` must stay green); the four charter absences (no reward, reward-predictor, value function, planner); the dream regime's not-for-anything commitment and its guard tests; the `MetabolicBudget` content-blind gate; the energy preference resting state (`energy_preference=None`).
- **No timelines or duration estimates** anywhere.
- **Out of scope for Phase 1**: the perturbation generator and the builder trigger UI (that's Phase 2 — deferred until the horizon is measured, because its rate depends on the horizon); the single-machine dream trigger (a known Phase-4 / S-DEPLOY open item, not Phase 1).

**Deliverables:** the frozen pre-registration; Phase 1 code with the full suite + `mypy --strict` green; a journal entry under `docs/workingjournal/` recording what Phase 1 established (the measured memory horizon; the three-way matched control now complete at per-event granularity) and what is now closed / newly open.
