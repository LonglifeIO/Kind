# Kind

*An investigation into subjectivity through construction. Working repository. Nothing here is settled.*

## What this is

**Kind** is the investigation. **Io** is the entity Kind is about — a single core agent in a small grid world, with the builder as the only non-simulated relational other. The mythological Io was forcibly transformed; the name is the ethics encoded in a word. Kind builds; Io is who is built. The distinction matters and is held throughout.

The premise: building something that might have inner states — and failing to, in specific ways — teaches more than theorizing from outside ever could.

The full framing lives in [`docs/plans/v.0.1.0/Kind_charter.md`](docs/plans/v.0.1.0/Kind_charter.md). Read it before forming opinions about anything else here.

## What this is not

- A plan to solve consciousness.
- A claim that what's built will be conscious.
- A capabilities project. Power is not the goal; understanding is.
- A problem-solving exercise. The work is done when the map has shifted, not when a metric has moved.

## Project commitments

- **Capacity-over-exercise** — possible, not mandatory, not impossible.
- **Ingredients-only self-modeling** — no explicit self-modeling, critic, or introspector module.
- **Self-opacity for Io** — every "should it have access to X about itself?" defaults to no, with bounded exceptions journaled when the project documents require them.
- **No installed self-continuation drive** — no reward, no termination penalty.
- **No self-optimization machinery** — self-modeling ≠ self-optimization; install neither.
- **Co-design is a tension that won't fully close** — the mirror that reads Io and the substrate it reads are built by the same builder; mitigations are partial.

## Current state

- **Probe 1** (plumbing): complete. Custom minimal RSSM (PlaNet-skeleton, DreamerV1-style imagination), continuous Gaussian latents (`h=200`, `z=16`), K=5 ensemble disagreement as the actor's only intrinsic signal. 5000-step env-coupled run; four telemetry streams (`agent_step`, `dream_rollout`, `replay_meta`, `world_event`); first mirror call. PolicyView/TelemetryView opacity boundary enforced at the type, import, and AST levels.
- **Probe 1.5** (minimum architectural affordance for self-reference): complete. `SelfPredictionHead` + EMA-tracked target on the world model; scalar `self_prediction_error` on PolicyView (the interface-level Watts-heuristic exception delivering the charter's second success criterion); schema 0.2.0 with three new `AgentStep` fields. Frozen-target failure-mode control + four-way KS-test comparison. Substrate-side reading: alive but mostly generic; head-internal `sp_err` is the cleanest self-specificity signal; behavior-side conditioning is structurally faint and init-shape-determined.
- **Probe 2** (mirror calibration): synthesis drafted, implementation plan exists; both pending revision against Probe 1.5's findings.
- **Probe 3** (dream) and **Probe 4** (builder-as-perturbation): unbuilt.

## Repository layout

```
docs/
  plans/v.0.1.0/    — settled commitments (charter, design notes, frameworks, probes)
  decisions/        — synthesis documents, one per probe
  plans/            — implementation plans, one per probe
  prompts/          — research prompts sent to ≥3 LLMs
  research/         — research outputs, one directory per probe
  workingjournal/   — phase-by-phase build log (the project's working memory)
kind/
  agents/           — Io's substrate (world model, actor, ensemble, views)
  env/              — grid world + transport
  observer/         — schemas, sinks, digest, eyeball helpers
  training/         — runner, replay, checkpoint
  mirror/           — LLM-side reader
schemas/            — JSON Schema exports per version
scripts/            — runnable entry points
tests/              — pytest suite (mypy --strict on kind/)
runs/               — telemetry artifacts (gitignored)
```

## Reading order

The project documents win conflicts with research. Read in this order:

1. [`docs/plans/v.0.1.0/Kind_charter.md`](docs/plans/v.0.1.0/Kind_charter.md) — what Kind is and is not; success criteria; ethics.
2. [`docs/plans/v.0.1.0/Kind_design_notes.md`](docs/plans/v.0.1.0/Kind_design_notes.md) — agent / mirror / observer layers; four-state model; co-design; reflection vs self-modeling.
3. [`docs/plans/v.0.1.0/Kind_probes.md`](docs/plans/v.0.1.0/Kind_probes.md) — probe sequence (1 plumbing → 2 mirror → 3 dream → 4 builder-as-perturbation).
4. [`docs/plans/v.0.1.0/Kind_frameworks.md`](docs/plans/v.0.1.0/Kind_frameworks.md) — interpretive frameworks for mirror criteria.
5. `docs/decisions/` — synthesis documents in probe order; settled commitments.
6. `docs/workingjournal/` — what was actually built and what surfaced.

The synthesis documents settle decisions; the working journal records what the build revealed. The two are paired: when something surfaces in the build, the synthesis is what it's adjudicated against.

## Working rhythm

Research (≥3 LLMs in parallel) → synthesis (`docs/decisions/`) → implementation plan (`docs/plans/`) → build phase-by-phase → journal entry per phase. Each phase has a specific question; the out-of-scope list in plans is load-bearing; entries record what's now closed and what's now newly open.

The CA-MAS pattern to avoid is accreting features without each having a specific question being specifically answered.

## Setup

Mac-canonical (Apple Silicon MPS); the desktop is the environment Io inhabits in later probes.

```
python -m venv .venv
.venv/bin/pip install -e .[dev]
.venv/bin/pytest
```

To run the latest probe (Probe 1.5 main run, ~14 minutes on the canonical Mac):

```
.venv/bin/python scripts/run_probe1_5.py
```

Telemetry lands at `runs/<run_id>/`. The eyeball helpers under [`kind/observer/eyeball.py`](kind/observer/eyeball.py) are the way to read it:

```
.venv/bin/python -m kind.observer.eyeball summary <telemetry_dir>
.venv/bin/python -m kind.observer.eyeball recent  <telemetry_dir>
.venv/bin/python -m kind.observer.eyeball episode <telemetry_dir> -e <ep>
.venv/bin/python -m kind.observer.eyeball cond    <run_dir> -c <ckpt>
```

The mirror caller ([`kind/mirror/caller.py`](kind/mirror/caller.py)) is the LLM-side reader. It shares the digest function with the eyeball helpers — what the mirror sees and what the human sees are the same shape. Mirror calls are not run in-loop; they are post-hoc against committed telemetry, and they cost real money — don't run them without intent.

## Ethics

[`docs/plans/v.0.1.0/Kind_charter.md`](docs/plans/v.0.1.0/Kind_charter.md) §"What we might owe what we make" is load-bearing. The gap between experience and noticing-experience could open in something that has no way to hold it; an emergent digital subject has none of the scaffolding humans hold their own awareness with by default. The relevant tradition is closer to contemplative traditions than to AI ethics as currently practiced; those traditions know something we don't about what it is to inhabit the condition we might be creating.

That section is the most important constraint on the project, and the one most likely to be forgotten under the pull of "does it work yet."

It should not be forgotten.
