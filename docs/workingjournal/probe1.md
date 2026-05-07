# Probe 1 — Working Journal

*Hand-written notes captured during build. One section per phase, with the
structural decisions made mid-build, surprises that came up, and what is
now decided that was open. The pre-build risk-framing for Probe 1 lives in
`pre-probe1.md`; this document is the running log from Phase 2a forward.*

---

## Phase 2a — env-server core (2026-05-02)

The grid environment in `kind/env/grid_world.py`, in isolation: no harness,
no mutators, no TCP, no telemetry emission. Five structural choices recorded
here so the smoke (Phase 8) and the four telemetry streams have unambiguous
context for what the env actually is.

### Drift magnitude — the arithmetic behind `1e-5`

The environment synthesis names the spec as "±10% over 50 episodes (random
walk)"; the implementation needs a per-step σ. With the Probe 1 defaults —
`initial_regrowth_p = 0.01`, `episode_length = 200` — fifty episodes is
10 000 environment steps. Reading "±10% over 50 episodes" as one standard
deviation of the random walk's accumulated drift after 10 000 steps gives:

```
σ_total  = 0.10 × initial_regrowth_p
         = 0.10 × 0.01
         = 1.0e-3

For a Gaussian random walk: σ_total = σ_step × √N
σ_step   = σ_total / √(50 × episode_length)
         = 1.0e-3 / √10 000
         = 1.0e-3 / 100
         = 1.0e-5
```

So `drift_magnitude_per_step = 1.0e-5` is the per-step Gaussian σ. After
one episode (200 steps) the accumulated 1σ drift is ≈ 1.41e-4 (about 1.4%
of `p`); after fifty episodes it is the spec's 1.0e-3 (10%). The drift is
clipped into `[drift_p_min, drift_p_max] = [0.001, 0.05]`; with σ_step =
1e-5 the clip rarely activates, but it is the defensive boundary the
synthesis names. If smoke shows drift dominating per-step KL, halve σ_step
(per the implementation plan §6 "Revisit when").

The interpretation choice — taking "±10%" as one standard deviation rather
than as a hard bound — matters for what the smoke will surface. A
1σ-interpretation gives a soft envelope: most fifty-episode windows stay
inside ±10% but extreme windows exceed it; the random walk visits the
clip boundaries only rarely. A hard-bound interpretation would have
required either a much smaller σ_step or a different distribution entirely.
The synthesis's wording is consistent with either; I picked the one that
matches the literature's standard usage of "magnitude" for Brownian-walk
processes.

### Initial resource count — `n_initial_resources = 4`

The env synthesis recommended the initial-resource count as "open during
build" with a sensible-default range of 3–6 on the 8×8 grid. I picked 4.
Reasoning: with 64 cells and one agent, an empty cell pool of 60. At
`p = 0.01` regrowth produces an expectation of 0.6 events per step or
~120 per episode, which dominates the initial four. So the initial
placement is just the seed; the steady-state population is set by the
balance of regrowth and consumption. Four cells is enough that a random
policy occasionally walks into one in the first episode, sparse enough
that the world is not trivially full.

The synthesis's wording — "resources sampled fresh from regrowth
distribution at the current p" on episode reset — is interpreted in the
implementation as a *count*, not a per-cell Bernoulli at `p`. Per-cell
Bernoulli at `p = 0.01` over 60 cells gives an expectation of 0.6
resources at episode start, which is too sparse to seed the dynamics
legibly for the first few hundred steps of each episode. The
`n_initial_resources` count keeps initial sparsity controllable and makes
the first observation of each episode something the mirror can read. If
the smoke shows the initial population pinning the dynamics around a
specific equilibrium, this default is the first thing to revisit.

### Walls — none placed at Probe 1

The cell-type vocabulary includes `WALL`, but `GridWorldConfig.walls` is
empty by default. The synthesis allowed walls but recommended the simplest
default: no walls, pure-empty grid plus resources. I followed the simpler
default. The wall machinery (collision-blocks-movement, walls-in-rendered-
view, walls-excluded-from-resource-placement) is implemented and tested
because Probe 4's perturbation distinguishability test will eventually
benefit from a richer cell vocabulary; running with no walls at Probe 1
just means the navigation problem is pure boundary-and-other-cells, which
is the cleanest baseline the synthesis names.

If the smoke shows posterior collapse on the substrate alone — KL pinned
at the free-bits floor, dreams trivially repetitive — adding walls is one
of the cheap interventions to try before scaling up the grid (per
`pre-probe1.md`'s framing of failure mode 2).

### Start cell — `(3, 3)`, fixed

I picked `(3, 3)` (off-center toward upper-left) for the default start
cell with the fixed-default convention the env synthesis suggested. At
this position the 7×7 ego-centric view spans grid rows 0–6 and cols 0–6,
leaving the (row 7, col 7) edge cells outside the view — i.e. Io's first
observation does not extend past the grid boundary, but the grid does
extend past the view. This is the "always something off-screen" property
partial observability is for. A corner start (e.g. `(0, 0)`) would have
shown OOB cells immediately; a true center start on an even-sized grid
doesn't exist (8 has no integer center). `(3, 3)` is the off-center cell
closest to the geometric center such that all view cells are in-bounds at
reset.

`start_cell = None` (random) is implemented and tested against the
config-validation surface, but is not the default. If smoke shows the
agent over-fitting to the fixed-start trajectory and ensemble disagreement
saturating because every episode begins identically, switching to random
start is the first revision to try.

### Drift carries across episodes

The synthesis named this "open during build, default carry across". I
implemented carry-across: only the explicit `reset()` method (the entry
point at run start) re-initializes `p` to `initial_regrowth_p`; episode
boundaries leave `p` alone. This means an Io trained for many episodes
sees a slowly-non-stationary world over the long run, which is the
synthesis's reason for including the drift in the first place — three
orthogonal variance sources, the third being slow drift, to keep the per-
step bit-content above the free-bits floor without saturating ensemble
disagreement.

If the smoke shows drift dominating per-step KL — i.e. the model is
spending most of its capacity tracking `p` rather than the grid — halving
σ_step is the first revision; resetting `p` per episode is the second.

### Other build-time choices worth surfacing

- **Action 4 (`stay`) is a true no-op for resource consumption.** The
  synthesis specifies that consumption is *triggered by entering* a
  resource cell. An earlier first-pass implementation ran the consumption
  check on every step including stays, which would have meant a resource
  that regrew under the agent got consumed on the next step regardless
  of action — a back-door automatic-consume that violates the
  consumption-on-entry contract. The fix is to short-circuit `_apply_action`
  for the stay action. Now: stay over a resource leaves the resource
  intact; the agent must move away and back to consume.

- **Regrowth applies to *all* empty cells, including the agent's.** The
  synthesis specifies "Each empty cell has a per-step probability p of
  becoming a resource. Independent per cell, action-independent." No
  exception for the agent's cell. This composes with the above: a
  resource regrows under the agent → the agent is not "entering" → no
  consumption. The agent ends a step on a resource cell sometimes, and
  that is by design.

- **Two RNG streams from one integer seed via `SeedSequence.spawn(2)`.**
  One stream for regrowth (per-cell coin flips and initial-resource
  placement), one for drift (Gaussian random-walk steps). The `state`
  property exposes the underlying world for the mirror; the streams
  themselves are private (`_regrowth_rng`, `_drift_rng`). The streams'
  independence is testable structurally and behaviourally — tests in
  `test_env_step.py` exercise both — including the test that overrides
  one stream after construction to verify the other's trajectory is
  unaffected. This is the property Probe 4's distinguishability test
  will eventually rest on: builder events and internal stochasticity have
  to share a vocabulary, and "internal stochasticity" itself has to come
  from a known and reproducible source.

- **Observation rendering uses an exact pixel-aligned 7→32 expansion.**
  The 7×7 view is repeated into 32×32 with cell sizes derived from
  `np.linspace(0, 32, 8).astype(int).diff()` — per-row/col counts of
  `[4, 5, 4, 5, 4, 5, 5]` summing to 32. No anti-aliasing, no
  interpolation. The encoder sees only the four legal grayscale levels
  (`{0, 64, 128, 255}`) at any pixel, which means it cannot mistake
  resampling artifacts for cell-type signal. If the smoke shows the
  decoder producing dreams that are unreadable to the mirror, the
  rendering is the last thing to revisit — encoder/decoder capacity is
  upstream.

- **Out-of-bounds rendering is `64` (dark gray).** The four legal pixel
  values are `0` (wall), `64` (OOB), `128` (empty), `255` (resource). All
  four are in the rendering table; `_OOB_SENTINEL = 3` is the value the
  view extraction places into out-of-grid cells, and `_RENDER_TABLE[3] =
  64` maps it to the rendered pixel value. OOB is distinguishable from
  every in-bounds cell type, which is the no-marker boundary the
  synthesis specifies for partial observability — the agent can tell
  "off-grid" from "wall" from "empty" from "resource" but cannot tell
  "regrowth event" from "builder add_resource" (the latter doesn't exist
  yet; Phase 3a wires it).

### What's now closed and what's now newly open

Closed:

- The drift-magnitude derivation (1e-5 ≡ 10% / √10 000).
- The episode-reset convention (drift carries; resources resampled at the
  configured count; agent at start cell; no terminal signal).
- The off-grid / wall-collision semantics (clock ticks, dynamics advance,
  position unchanged).
- The stay-action / consumption interaction (stay does not consume).
- The two-RNG-stream architecture and how to test its independence.

Newly open:

- Whether the fixed start cell `(3, 3)` over-constrains the trajectory
  distribution enough to saturate ensemble disagreement within the first
  few hundred steps of a run. The smoke is what tells us. If yes, switch
  to `start_cell = None` (random non-wall draw via the regrowth stream)
  and document.
- Whether resource regrowth under the agent produces enough "weird"
  cases that the mirror's dream-rollout reading struggles. Holding this
  open; the synthesis specifies the literal-reading semantics and Probe 1
  honors them, but if Probe 2's mirror flags the case as confusing the
  reading, that's a revisit.
- Whether `n_initial_resources = 4` keeps the first-episode population
  legible enough for smoke. If smoke shows Io's first-episode KL pinned
  at the floor because the world starts essentially empty (resources
  consumed quickly, regrowth slow at p=0.01), bump to 6 or raise the
  initial-only `p` to seed a denser start. Default left at 4.

---

## Phase 3c — PolicyView / TelemetryView opacity boundary (2026-05-02)

The two frozen dataclasses and the `split` function in `kind/agents/views.py`,
plus the placeholder `kind/agents/actor.py` whose only job at this phase is
to make the dependency-lint test something other than a stub. The
synthesis §Q5 self-opacity boundary is now structural-by-default in code:
the actor's import surface excludes `TelemetryView` by AST-checked rule;
the type signatures (when Phase 3b lands) will exclude it at type-check
time; and the frozen dataclasses exclude in-place attribute mutation at
runtime. This is the smallest phase yet but it is the load-bearing
discipline for everything Phase 3b builds on top of.

### Field-membership decision: `TelemetryView.recon_loss`

The plan §2.7 names the field `recon_loss: Tensor`, and the user's brief
re-quoted both the field name and the split signature `split(step,
intrinsic) -> tuple[PolicyView, TelemetryView]`. Reading the spec
literally: only two inputs (`step` and `intrinsic`), so `recon_loss` must
be populated from one of them. The only candidate is `WorldModelStep.recon`
— the reconstruction tensor from Phase 2b's decoder, shape
`(B, obs_channels, obs_size, obs_size)`.

Implementation: `telemetry.recon_loss = step.recon`. The field name
follows the synthesis §Q3 telemetry-schema nomenclature (`recon_loss_t`
in `AgentStep`, which IS a scalar in serialised form), but the in-memory
value is the 4-D reconstruction tensor. There is naming dissonance: the
field name implies a scalar loss; the value is the unreduced
reconstruction.

This is acceptable for Phase 3c because:

1. `TelemetryView` is the in-memory projection the mirror reads. The
   mirror benefits from seeing the reconstruction tensor (decoded
   dreams, qualitative pixel reading) rather than only a scalar.
2. Phase 5's runner is what converts `TelemetryView` into `AgentStep`
   records. The runner has the `obs_target` available (it is doing the
   training step) and can compute the scalar loss from
   `view.recon_loss` and `obs_target` via `WorldModel.loss()` before
   serialising.
3. The synthesis explicitly lists `recon_loss_t` *in the schema*; the
   plan's TelemetryView is one step upstream of serialisation, where the
   value is still tensor-shaped.

If Phase 5 finds the naming confusing, two clean refactors are
available: (a) rename the field to `recon` to match `WorldModelStep`;
or (b) add a separate scalar `recon_loss_scalar: Tensor` field and let
the runner populate it. Both are forward-compatible with Phase 3c's
shape — `split`'s signature would gain an extra parameter or the field
list would grow. Neither is needed yet.

### Frozen dataclass plus Tensor field

Phase 2b's `WorldModelStep` already uses `@dataclass(frozen=True)` with
Tensor fields and the test suite passes. Both new views follow the same
pattern. The auto-generated `__eq__` would be problematic if two views
were ever compared with `==` (the comparison would reduce to
`tensor == tensor`, which returns a Tensor, not a bool, and would raise
on non-scalar tensors); we never compare views with `==` and the tests
use `torch.equal` and `is` checks instead. If a future user of the views
runs into the `__eq__` problem, switching to `@dataclass(frozen=True,
eq=False)` is the fix.

`frozen=True` prevents *attribute reassignment* (`view.h = ...` raises
`FrozenInstanceError`). It does not prevent in-place mutation of the
underlying tensors via PyTorch. The boundary is structural-by-default,
not adversarial — the synthesis §Q5 is explicit about this. The actor,
when it exists, simply will not have an opportunity to do an in-place
edit because its forward pass receives the view by reference and
operates on `view.h`, `view.z` as read-only inputs to `torch.cat` /
linear layers / etc.

### Tensor identity (no copies)

The split function passes references — no `.clone()`, `.detach()`,
`.copy_()`, or any operation that would allocate a new buffer or break
the gradient chain. Memory is not duplicated; gradients flow cleanly
from the decoder / posterior heads back through whichever consumer
(actor, telemetry sink, mirror) holds the view. The test
`test_split_does_not_copy_tensors` verifies identity via `is`; the test
`test_split_preserves_gradient_chain` verifies that
`requires_grad=True` survives through both views.

### AST-based dependency lint

The user specified AST inspection rather than string grep (more robust
against the name appearing in comments, docstrings, or string literals).
The implementation walks every `ast.Import` and `ast.ImportFrom` node in
`kind/agents/actor.py` and collects `alias.name` from each (the source
name, not `alias.asname`, so aliased imports are caught: `from views
import TelemetryView as _T` still has `alias.name == "TelemetryView"`).

Three bypass forms tested via REPL during build (and verified by the
test against the placeholder):

1. Direct: `from kind.agents.views import TelemetryView` — caught by
   "TelemetryView" in collected names.
2. Aliased: `from kind.agents.views import TelemetryView as _T` —
   caught (alias.name is the source name).
3. Whole-module: `import kind.agents.views` — caught by the second
   assertion in the test, which rejects the full module name on the
   grounds that attribute access (`kind.agents.views.TelemetryView`)
   would expose the symbol.

Forms not currently caught (acknowledged limitations):

- `getattr(views_module, "TelemetryView")` after a runtime
  `importlib.import_module` — would bypass the static AST. If Phase 3b's
  actor adds dynamic-import code, the lint should be extended.
- Re-exports via a third module: if `kind/agents/__init__.py` were to
  re-export `TelemetryView`, the actor could import it via
  `from kind.agents import TelemetryView`. The lint as written would
  catch this (the source name is still "TelemetryView"). But the
  synthesis is explicit that the package level should not re-export
  TelemetryView; we don't audit the package __init__ here.

The lint is a test (run as part of `pytest tests/`), not a CI hook.
That's appropriate for Probe 1's discipline level — the synthesis says
"structural-by-default, not adversarial". A determined refactor could
disable the lint by editing the test; the boundary is the contract, not
the prison.

### Placeholder actor pattern

The actor module at Phase 3c is three meaningful lines:

```python
from kind.agents.views import PolicyView
_ = PolicyView
__all__: list[str] = []
```

The `_ = PolicyView` line makes the import "used" so static analysis
doesn't flag it. The `__all__: list[str] = []` says "this module
re-exports nothing" — Phase 3b will populate the list when there's
something to expose (Actor, LatentDisagreementEnsemble). The docstring
explains the placeholder's role and the rules Phase 3b must keep:
"keep the PolicyView import; ensure no TelemetryView import is added".

The placeholder approach (vs. skipping the lint until Phase 3b) was
chosen to make the discipline live from the moment Phase 3c lands.
Skipping with `pytest.skip(...)` until 3b would have left the
opacity boundary unenforced for the entire 3a/3b/3c parallel window;
the placeholder closes that gap.

### What's now closed and what's now newly open

Closed:

- The `PolicyView` field set: just `h` and `z`. No concat, no learned
  projection, no derived attributes.
- The `TelemetryView` field set: the seven `WorldModelStep`-equivalent
  fields plus `intrinsic_signal`. `recon_loss` is the only renamed field.
- The `split(step, intrinsic) -> (PolicyView, TelemetryView)` signature
  and the populate-`recon_loss`-from-`step.recon` decision.
- The AST-based lint and its three-form coverage (direct, aliased,
  whole-module).
- The placeholder actor pattern and the rules it preserves for Phase 3b.

Newly open:

- Whether Phase 5's runner wants the `recon_loss` field renamed (`recon`)
  or split into tensor + scalar. Default is to leave it as-is until the
  runner is built.
- Whether Phase 3b should extend the lint to also reject imports of the
  telemetry sinks and the mirror caller (those modules don't exist yet
  from the actor's perspective). Defer until Phase 6 (mirror) lands.
- Whether the views should include a `__repr__` that hides intrinsic
  signal value when accidentally logged (defensive measure against
  printing a TelemetryView and surfacing the actor's would-be
  self-readout into a log). Probably not — the actor never holds a
  TelemetryView reference, so accidental logging would have to come
  from non-actor code, which already legitimately reads the field.
  Held open in case Probe 2's mirror surfaces a case where this matters.

---
"Dream cadence at Probe 1 is regular (1 per 1000 env steps), per the synthesis's default. This is engineering simplicity, not a stance on what cadence dreams should have. Biological dreaming follows broader rhythms (circadian, sleep-cycle) but isn't perfectly regular within them. For Probe 3, the cadence question becomes live: should dreams fire on regular intervals, stochastically, in response to internal state thresholds, or coupled to the builder's life patterns? The design notes commit to builder-coupling for the dream-to-wake ratio specifically; whether internal-state coupling is also part of it is a question Probe 3's research should address. Recorded here so the question carries forward."

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

---

## Phase 3b — Actor and disagreement ensemble (2026-05-02)

The actor and the K=5 ensemble in `kind/agents/actor.py` (replacing the
Phase 3c placeholder) and `kind/agents/ensemble.py`. A few structural
choices the smoke will end up testing:

- **Discrete action sampling in imagination uses straight-through
  Gumbel-Softmax** (`F.gumbel_softmax(logits, hard=True)`). The user's
  spec said "samples an action" + "analytic gradients"; for discrete
  actions these compose only via straight-through estimation. Forward
  produces a hard one-hot, backward uses the soft path. The action
  one-hot is multiplied through both the world model's
  (`world_model.action_embedding.weight`) and the ensemble's own action
  embedding matrices to produce differentiable embeddings — that is what
  makes the imagined trajectory differentiable end-to-end in the actor's
  parameters. Sampling at env-step time uses plain
  `torch.distributions.Categorical` (non-differentiable; env-step
  doesn't need gradients).

- **World-model and ensemble parameters are frozen during the actor's
  loss** via a `_frozen_params` context manager that sets
  `requires_grad=False` on each module's parameters and restores them on
  exit. Gradients still flow THROUGH the parameters (matmul-backward
  uses their values to compute gradients on the inputs) but they don't
  accumulate ON them. The user's spec recommended this pattern
  explicitly; the test
  `test_imagine_loss_does_not_grad_world_model_or_ensemble` verifies it,
  and `test_imagine_restores_world_model_requires_grad_after_loss`
  verifies the context manager restores state on exit.

- **K=5 ensemble heads with biased variance.** Per the user's spec.
  `torch.var(predictions, dim=0, unbiased=False).sum(dim=-1)` is the
  disagreement formula. The ensemble has its own action embedding
  (`embed_dim=16`, not shared with the world model's action embedding) —
  this is the synthesis §Q2 commitment that the ensemble rides on top
  of the world model's latent without coupling its representation
  choices.

- **The dependency lint continues to pass against the real actor.**
  PolicyView is imported (`from kind.agents.views import PolicyView`,
  used as the type of `Actor.forward`'s argument); TelemetryView is not
  imported anywhere in the actor module. The Phase 3c lint test now
  runs against a non-trivial implementation and still flags any future
  regression. The actor also imports `WorldModel` and
  `LatentDisagreementEnsemble` for type hints in
  `imagine_and_compute_loss`, both of which are non-blacklisted by
  design (the lint only blacklists the views' telemetry-side surface).

- **Phase 5's runner expects the actor in two distinct call patterns:**
  (1) at env-step time: `actor.forward(view: PolicyView) -> ActionOutput`,
  returning a sampled action plus log-prob, entropy, and raw logits —
  the runner extracts these for the `AgentStep` schema's `action_t`,
  `action_logprob_t`, `policy_entropy_t` fields. (2) at training time:
  `actor.imagine_and_compute_loss(world_model, ensemble, h_0, z_0,
  horizon=15) -> dict[str, Tensor]`. The runner is expected to pass
  *detached* `(h_0, z_0)` from a real replay batch; the function returns
  `{"actor_loss", "mean_disagreement", "policy_entropy"}`. The runner
  calls `.backward()` on `actor_loss`, then steps the actor's optimiser.
  The world model has its own loss (`WorldModel.loss(step, obs_target)`,
  Phase 2b) and the ensemble has its own loss
  (`LatentDisagreementEnsemble.compute_loss(h_t, z_t, a_t, z_target)`);
  the three losses do not interfere via gradient flow. Pragmatic value
  is uniform-prior at Probe 1 (zero contribution); the formula
  structure is preserved as scaffolding for Probe 4+.
Surface this in Probe 4's research prompt directly.

---

## Phase 3a — env-server harness and the four mutators (2026-05-02)

The harness in `kind/env/env_server.py` and the four mutators as free
functions in `kind/env/mutators.py`. Synchronous, in-process, local-Python
only — Phase 4 will wrap this in TCP. Five structural decisions worth
recording:

- **No-op mutators succeed and emit anyway, preserving the audit trail.**
  `add_resource` on a cell that is already a resource returns success with
  `pre_state == post_state == "resource"`; `remove_object` on an empty cell
  is the same shape (`pre == post == "empty"`); `set_cell_state` to the
  current state ditto. The exception is `move_object`, which raises on
  `cell_from == cell_to`, on an empty source, and on a non-empty destination
  — a "self-move" has no counterpart in internal stochasticity, and atomic
  move semantics need the destination empty so the post-state is
  unambiguous. `add_resource` on a wall raises (regrowth never overwrites
  walls; allowing the mutator to do so would break the same-vocabulary
  commitment Probe 4's distinguishability test rests on).

- **No-marker test uses a quiet config and direct grid mutation.** The
  parallel-env approach the user spec named: two `EnvServer` instances with
  the same seed and a `p=0`/zero-drift config, one calls `add_resource`,
  the other writes the same cell directly via `_grid` (the same end-state
  regrowth would have produced if it had fired). Both step with `stay`;
  pixel observations are identical. The world-event stream by contrast
  carries the asymmetry — env_a's sink has the `builder_perturbation`
  record, env_b's does not. Quiet config (rather than racing to wait for an
  actual regrowth event) keeps the test deterministic.

- **Episode-boundary emission order: aggregate first, then env_reset.**
  Closing the old episode before opening the new one matches the natural
  temporal ordering and gives the mirror an unambiguous "this aggregate
  belongs to episode N, this env_reset announces episode N+1" structure.
  Both records share the same `t_event` (the boundary step's env_step). The
  per-episode aggregate's `regrowth_events` count covers steps 1..N-1 of
  the closing episode — the boundary step's regrowth is missed because
  `GridWorld.step` resets the grid internally between regrowth and the
  post-step snapshot. With p=0.01 this under-counts by ~0.6 events out of
  ~120, well inside the noise floor for the aggregate's intended use; if a
  later probe needs exact counts, the fix is to instrument `GridWorld`
  directly rather than work around it from the harness.

- **Pre-step grid snapshot is taken at the *start* of `step()`, after any
  mutators.** This was the non-obvious correctness call. If the snapshot
  were taken at the end of the previous step (the more "natural" place),
  any builder mutator called between two `step()` calls would change the
  grid, and the diff at the next step would mis-attribute the builder
  mutation as a regrowth event. Snapshotting at the start of `step()` —
  immediately before `GridWorld.step` runs — folds all intervening builder
  mutations into the pre-state and the diff captures only what the env's
  own dynamics produced. The aggregate cleanly excludes builder events.

- **Phase 4 transport implications.** The harness owns the world-event
  sink directly via `JsonlSink` writing to a local path. Phase 4's
  Mac/desktop split (env-server runs on desktop, telemetry lives on Mac)
  means Phase 4 will need either a side channel that ships these records
  to the Mac's filesystem or a multiplexed message type on the existing
  TCP socket — the harness's sink path becomes a thing to redirect, not a
  thing to remove. The four mutators are methods on `EnvServer` returning
  `None`; Phase 4 wires each as a request type and the response is just
  acknowledgement (the world-event record carries the truth, not the
  return value). Wallclock comes from `time.monotonic_ns()` at every
  emission point including the override on the wrapped `EnvStep` — Phase
  4 must preserve this when it serializes EnvSteps over the wire (do not
  re-stamp on the Mac side, or the monotonicity guarantee across the two
  streams breaks). `set_checkpoint_id(checkpoint_id)` is the wiring point
  for the runner's checkpoint barrier; the harness defaults `checkpoint_id`
  to `None` and Phase 5 will call this setter when checkpoints commit.

### What's now closed and what's now newly open

Closed:

- The four mutators' validation surface (in-bounds; `CellType`-typed args;
  no-op succeed with pre==post; the `move_object` exceptions).
- Episode-boundary emission order (aggregate, then env_reset).
- Pre-step snapshot timing (start of `step()`, post-mutator).
- Wallclock source (single monotonic clock for both `EnvStep` overrides
  and every `WorldEvent`).
- The env_reset payload shape (episode_id, resource_positions list,
  regrowth_p) and the internal_stochasticity_aggregate payload shape
  (episode_id, regrowth_events, mean_drift_step_magnitude, final_p).
- Context-manager surface (`__enter__` returns self; `__exit__` calls
  `close`; tests call `start()` explicitly to obtain the first `EnvStep`).

Newly open:

- Whether Phase 4's transport multiplexes WorldEvents on the existing TCP
  socket or opens a side channel for them. The harness is agnostic; the
  decision belongs to Phase 4.
- Whether the boundary-step regrowth under-count (~0.5%) ever matters for
  Probe 4's distinguishability metric. Probably not, but flagged.
- Whether Phase 5's runner wants `set_checkpoint_id` to also flush the
  sink (a barrier-style guarantee that pre-checkpoint records hit disk
  before post-checkpoint records do). The current implementation does
  not; the JsonlSink only fsyncs on `close`. If the runner needs barrier
  semantics, the cleanest place to add a `flush()` method is on
  `JsonlSink` directly (Phase 1's surface).

---

## Phase 4 — TCP transport and checkpoint barrier (2026-05-02)

The wire protocol in `kind/env/transport.py` and the atomic-commit
manager in `kind/training/checkpoint.py`. Single connection per
env-server, length-prefixed framed JSON over TCP, base64-encoded NumPy
serialization for the observation tensor (msgpack deferred per plan §8).
182 tests pass; mypy `--strict` is clean across all 17 source files.

- **Threading model: one reader thread per side, no async.** The server
  side's reader thread is also the sender — it reads STEP/MUTATE/BARRIER
  messages, calls into the wrapped `EnvServer`, and writes responses
  back; sends are inherently serialized. The client side has a reader
  thread that routes incoming messages to either the synchronous
  response queue (TRANSITION, MUTATE_ACK, BARRIER_BEGIN_ACK) or the
  configured `world_event_handler` (WORLD_EVENT). The user-facing API on
  the client is fully synchronous: `step()` sends a STEP and blocks on
  the matching TRANSITION. Threads via stdlib `socket` and `threading`,
  no third-party network libs. The Probe 1 volume (~hundreds of msgs/s)
  doesn't justify asyncio's complexity; if Phase 5's runner needs
  concurrent in-flight requests, a single-thread asyncio rewrite is
  cheaper than retrofitting it onto the existing thread model, but for
  now sequential is correct and clear.

- **`BARRIER_BEGIN_ACK` added to the message vocabulary.** The user
  spec recommended adding this so the trainer knows when the env-server
  has actually drained and is paused before committing the checkpoint.
  Without it, the trainer would have to assume the server processes
  messages instantly, which is fine in a single-machine loopback test
  but fails the moment the actual desktop deployment is wired (network
  latency, scheduling delays). The ACK is cheap (one round-trip) and
  makes the barrier semantics symmetric: BARRIER_BEGIN with ACK,
  BARRIER_END without (the server resumes immediately on receipt and
  drains the queued messages, sending their responses inline). I
  implemented `STEP` and `MUTATE` queueing during the barrier even
  though the trainer's synchronous client wouldn't issue one mid-barrier;
  the queue is defensive against a future async client and has zero
  cost for the trainer's actual flow.

- **`world_event_handler` refactor on Phase 3a's `EnvServer`.** The
  config now takes `world_event_handler: Callable[[WorldEvent], None]`
  instead of `world_event_sink_path: Path`, and the harness invokes the
  handler synchronously where it used to write to its owned `JsonlSink`.
  Phase 3a's tests own the sink and pass `sink.write` as the handler;
  Phase 4's transport server replaces the handler with a wire-shipping
  callable (`_send_world_event`) once a client connects. The seam is
  clean: the desktop env-server is now sink-free, the Mac trainer
  receives `WORLD_EVENT` messages over the wire and routes them to its
  own local `JsonlSink`. This matches the synthesis's "Mac owns
  telemetry; the desktop is stateless except for the env's RNG state"
  commitment. Two new tests in `test_perturbation_hook.py` exercise the
  handler directly (`captured.append`) so the indirection is verified
  before the transport wraps around it; everything else just adopts the
  new helper.

- **Atomic checkpoint via `os.rename` + `fsync`-the-parent.**
  `CheckpointManager.commit` does the five steps in order: send
  `BARRIER_BEGIN`, await `BARRIER_BEGIN_ACK`, copy each named source
  file into `{id}.staging/` under canonical names (plus replay shards
  into `replay/` by basename), `fsync` each file, `fsync` the staging
  directory, `os.rename(staging, target)`, `fsync` the parent
  checkpoints directory, send `BARRIER_END`. The rename is atomic on
  POSIX same-filesystem same-parent — the checkpoint either exists
  fully under `{id}/` or doesn't exist. On any staging error, the
  staging directory is cleaned up best-effort and `BARRIER_END` is sent
  anyway so the env-server resumes; the original exception then
  propagates. The "leaves clean state" property the user spec named is
  enforced structurally rather than checked after the fact: target dir
  is verified non-existent at the start, staging dir is purged on
  error, only the rename is observable. A stale `.staging/` from a
  prior crashed attempt is detected and removed before the new commit
  starts.

- **Choices that bear on Phase 5's runner.**
    1. The `EnvTransportClient.connect()` call returns the *initial*
       `EnvStep` — the runner doesn't call `EnvServer.start()` directly
       anymore, it gets the first observation from the wire handshake.
       The transport server is the one that owns calling `start()` on
       the wrapped env-server.
    2. The `world_event_handler` the runner provides to
       `EnvTransportClient` is the runner's own `JsonlSink.write` (or a
       wrapper that does additional accounting). The runner is
       responsible for keeping the sink alive across the transport
       client's lifetime and closing it after `client.close()`.
    3. `CheckpointContents` is a frozen dataclass with paths to
       pre-existing files; the runner is responsible for producing
       those files (via `safetensors.save_file`, `torch.save`,
       `pickle.dump`, etc.) into a temporary directory before calling
       `commit`. The manager copies them into the checkpoint; the
       source paths remain readable after commit.
    4. The transport client and server use ephemeral ports (`port=0`)
       in the tests; the runner's production wiring will use a config-
       driven port (default 5555 per plan §8).
    5. `EnvServer.set_checkpoint_id(id)` (Phase 3a's existing method)
       is what the runner will call after `commit()` returns, so
       subsequent `WorldEvent` records carry the new checkpoint
       identifier in their envelope. The transport server's reader
       loop is the right place to expose a setter that propagates to
       the wrapped env-server, but Phase 4 doesn't add it — the
       runner can call directly into the env-server (loopback) or the
       feature can be added to the protocol later if a real
       desktop/Mac split needs it.

### What's now closed and what's now newly open

Closed:

- The wire format: 4-byte big-endian length prefix + UTF-8 JSON body;
  base64-encoded raw bytes for the observation tensor; per-mutator
  argument coercion (`tuple ↔ list`, `CellType ↔ name string`).
- The eight message types: `STEP`, `TRANSITION`, `MUTATE`,
  `MUTATE_ACK`, `BARRIER_BEGIN`, `BARRIER_BEGIN_ACK`, `BARRIER_END`,
  `WORLD_EVENT`.
- The barrier semantics: BEGIN+ACK, queue STEP/MUTATE during, drain on
  END in arrival order.
- The `world_event_handler` seam on the harness (Phase 3a refactor)
  and the transport server's takeover of it on connect.
- The atomic-rename-with-fsync commit dance and its failure-cleanup
  behavior.

Newly open:

- Whether to expose `set_checkpoint_id` over the wire (so the desktop
  env-server's records carry the right checkpoint envelope after a
  commit). Phase 5 may need this if the trainer wants the env-server's
  WorldEvents post-commit to be tagged. Right now the runner can call
  the env-server's setter directly under loopback; for a real desktop
  split, a `SET_CHECKPOINT_ID` message would be added.
- Whether the `_DEFAULT_RESPONSE_TIMEOUT_SEC` (30s) is appropriate for
  the smoke test's checkpoint commit. Long-running file-system fsync on
  a slow disk could exceed it. The smoke will tell us; bumping to 120s
  is the cheap response if it does.
- Whether the transport's single-connection model holds for the dream-
  state machinery (Phase 3) — a separate read-only consumer of weights
  is named in the synthesis as a later-probe optimization. If that
  consumer wants its own connection to the env-server's WORLD_EVENT
  stream, the `serve_forever` loop's `accept()` would need to fan out.
  Probe 1 doesn't need this; Probe 3 might.
- Whether `CheckpointContents` should grow a `world_model_config_path`
  or similar so a checkpoint records the configuration it was trained
  under. Phase 5 will know.

---

## Phase 5 — runner and sequence replay buffer (2026-05-02)

The replay buffer in `kind/training/replay.py` and the integration
runner in `kind/training/runner.py`. 220 tests pass (was 182 prior + 24
replay + 14 integration smoke); mypy `--strict` is clean across all 19
source files plus the two new test files. Six structural points worth
recording:

- **Resume-from-checkpoint determinism is verified at the
  parameter-equality level, not at the trajectory level.** The smoke
  test reads the committed `weights.safetensors` file directly and
  compares it byte-for-byte against runner_b's three modules after
  `load_checkpoint`. Optimizer states, RNG states (Python random, NumPy,
  PyTorch CPU, the runner's `torch.Generator` for replay sampling, and
  the device-specific RNG via `torch.mps.{get,set}_rng_state` /
  `torch.cuda.{get,set}_rng_state`), and the runner's runtime tuple
  (`h_prev` / `z_prev` / `a_prev` / latest `EnvStep` / iteration
  counter) all round-trip through pickle. Trajectory-level match
  across the resume boundary would additionally need the env-server's
  RNG state to be checkpointable, which Probe 1's wire protocol does
  not yet ship — `BARRIER_BEGIN_ACK` returns no payload, and
  `CheckpointContents` has no env-side fields. Documented as a known
  limitation rather than a bug; Probe 3's four-state machinery is the
  natural place to add env-side checkpointing because it'll need to
  freeze and resume env state across waking↔dreaming transitions
  anyway.

- **Device handling is set once in `__init__`, used consistently
  thereafter.** `RunnerConfig.device` is a string ("cpu", "mps", or
  "cuda"); the runner constructs `torch.device(device)` in `__init__`,
  moves the three modules onto it, and the hot loop never branches on
  device after that. Observations from the transport are converted to
  CPU `(1, 32, 32)` float32 tensors once (uint8 / 255), stored in the
  buffer in that form, and lifted to device only when feeding the
  model — `obs_t_dev = obs_t_cpu.unsqueeze(0).to(device, non_blocking=True)`.
  This keeps the buffer's memory cost on CPU (a 100k-capacity buffer at
  Probe 1 sizes is ~800MB float32 — acceptable), and isolates device
  semantics to the model forward.

- **Warmup default is 1000 env steps.** With `replay_sequence_length=32`
  and `episode_length=200`, that's ~5 episodes' worth of transitions
  before training begins — comfortably more valid windows than the
  default `replay_batch_size=16` needs. The smoke test uses
  `warmup_env_steps=10` and `replay_sequence_length=4` to keep the
  smoke fast (training kicks in early); the production default is
  conservative enough that the world model never trains on a
  near-empty buffer.

- **Checkpoint is post-train, post-dream, end-of-iteration.** Order
  inside `_step_once`: world model forward → action → env step → insert
  → AgentStep emit → train (if warmup over) → dream (if cadence) →
  checkpoint (if cadence). The AgentStep at the cadence boundary
  carries the *old* `checkpoint_id` (envelope was sealed before the
  commit fired); records at strictly-later env steps carry the new id.
  This is the natural ordering — emit what just happened, then take
  the snapshot — and the smoke test pins it
  (`test_smoke_checkpoint_carries_through_to_subsequent_records`).

- **Two small additive changes upstream:** (1) added
  `EnvTransportClient.set_world_event_handler` mirroring the existing
  setter on `EnvServer`. The transport client takes a handler at
  construction; the runner's `WorldEvent` JSONL sink only exists after
  `Runner.__init__`. The setter resolves the chicken-and-egg cleanly
  without a wrapper-holder pattern at the call site. (2) `Runner.__init__`
  takes an optional `env_server: EnvServer | None` so the runner can
  call `env_server.set_checkpoint_id` directly under loopback. For a
  real desktop split this is `None` and the desktop's `WorldEvent`
  records keep `checkpoint_id=None` between commits; full sync needs
  a `SET_CHECKPOINT_ID` wire message which Phase 4's open question
  flagged but did not build.

- **What surfaced during the smoke that wasn't anticipated:** the
  cadence of `dream_cadence_env_steps=50` over 200 env steps yields *3*
  dream rollouts, not 4 — at iter 0 the env_step_now is 0 and the
  guard `env_step_now > 0` excludes it, and the loop iterates env_step
  0..199 so env_step=200 is not reached. The off-by-one is deliberate
  (iteration 0 is the connect handshake; nothing has trained yet) and
  the test pins it. Bears on Phase 8's MPS smoke: the canonical script
  should not assume "N/cadence" rollouts — it's "floor((N-1)/cadence)".

### What's now closed and what's now newly open

Closed:

- The replay buffer's three-event meta emission contract (`insert` /
  `sample` / `evict`), the `segment_id` increment-per-event-type, and
  the episode-boundary admissibility rule for sample windows.
- The runner's hot-loop ordering and the per-iteration cadence checks.
- The combined-weights file layout (one `weights.safetensors` with
  `world_model.` / `actor.` / `ensemble.` prefixes; load_file splits
  back into three state dicts).
- The RNG-and-runtime pickle blob layout (Python random, NumPy, torch
  CPU, sample_rng, device-specific RNG, plus the runner's runtime
  tuple).
- The `EnvTransportClient.set_world_event_handler` setter and the
  runner's `env_server` optional reference.

Newly open:

- Whether trajectory-match resume should be a Probe 1 gate at all, or
  whether the parameter-equality version (current) is sufficient for
  the substrate test. The synthesis is silent; the implementation
  plan §4 phrases the smoke as "resume from checkpoint yields
  identical RNG state", which the current test honors. Trajectory
  match is named in the user's Phase 5 spec but the spec also said
  "if you can't get exact match, document what's drifting and why" —
  this entry is the documentation.
- Whether the `_DEFAULT_RESPONSE_TIMEOUT_SEC` of 30s is enough for the
  Phase 8 MPS smoke's mid-run checkpoint commit. The smoke here uses
  CPU and small models; commits are sub-second. MPS-side commits with
  full-sized models (h=200, z=16) are unmeasured; if the timeout
  fires, bumping to 120s is the cheap response. (Same open question
  as Phase 4.)
- Whether `episode_id` on a transition should be the FROM observation's
  episode (current choice) or the env_step's. With `episode_length=50`
  in the smoke test and 200 steps, four episode boundaries each create
  a transition whose `next_obs` is in a *different* episode than its
  `obs`. The buffer rejects such windows from sampling, but the
  per-transition definition could go either way; the FROM-side choice
  is the semantically natural one (the action was taken in that
  episode) and the tests align with it.
- Whether `Runner.run` with a pre-driven client (where the env-server
  is already at some env_step before `run` is called) should support
  the loaded-from-checkpoint path. Currently `run` calls
  `client.connect()` only when `env_step_meta is None`; the resume
  test exercises the post-load path where `env_step_meta` is loaded
  from the checkpoint, but the test still has to call `client.connect()`
  manually because `_step_once` calls `transport.step` which needs an
  active connection. Phase 6+ may want a cleaner API for this.

---

## Phase 6 — mirror caller (2026-05-02)

The single Gemini call in `kind/mirror/caller.py`, plus the
`MirrorReading` / `MirrorReadingPayload` Pydantic split and the
`readings.jsonl` sink. 263 tests pass (was 220 + 43 new); mypy `--strict`
is clean across all 20 source files. The mirror is now a callable
conduit, not yet wired into the runner — Phase 7's eyeball helpers and
the smoke (Phase 8) are what exercise it from a script. Five structural
points worth recording:

- **Gemini, not Anthropic.** The plan §2.10 names Anthropic as the
  default; the broader project commitment is methodological independence
  at the mirror layer (the research workflow itself runs on Anthropic).
  Implemented with the modern `google-genai` SDK (the unified
  `from google import genai` namespace, not the older
  `google-generativeai`). Default model `gemini-2.5-pro`; the constructor
  accepts a `model=` argument so a switch to `gemini-2.5-flash` is one
  parameter change if rate limits become an issue. API key from
  `GEMINI_API_KEY`; the constructor refuses to proceed without one
  (empty string handled the same as missing).

- **Prompt structure: calibration, not interpretation.** The system
  prompt frames Io minimally as "an experimental learning system" and
  asks the model for an *observational* summary. No frozen criteria
  from `Kind_frameworks.md`, no consciousness vocabulary, no
  interpretive reading. The prompt explicitly tells the LLM that Probe 1
  is a calibration check ("does the data carry signal") rather than an
  evaluation of anything resembling inner experience. Probe 2 is the
  phase that introduces frozen criteria and the adversarial second-LLM
  check; Phase 6 is deliberately silent on both. Bears on Probe 2's
  mirror calibration work: the Probe 2 prompt will need to add the
  frozen-criteria block on top of this base, and the adversarial
  structure either runs the same prompt with a different framing or a
  parallel prompt that argues against the first reading. Both are
  cleanly separable from the Phase 6 surface.

- **Digest design: scalar-only sample records, per-episode aggregates.**
  The high-dimensional fields (`h_t`, `z_t`, `q_params_t`, `p_params_t`,
  `kl_per_dim_t`, `encoder_embedding_t`) are deliberately excluded from
  the prompt — they are not legible to a language model and would
  dominate the token budget without carrying signal a Probe 1 reading
  uses. What goes in: per-episode mean/std for `kl_aggregate_t`, mean
  for `recon_loss_t` / `intrinsic_signal_t` / `policy_entropy_t`, action
  distribution counts (with skew flag if max action share > 0.7),
  flagged outliers (`kl_aggregate_t` z > 3, top recon-loss steps), and
  three sample records per episode (first/middle/last) with scalar
  fields only. The full records remain in the parquet shards for any
  later mirror that wants to do statistical analysis directly; the
  digest is what the LLM reads, not the only thing the data exposes.
  This means the prompt is bounded in size regardless of how many
  records the runner has produced — `n_episodes=3` with 200 steps each
  is ~3 KB of prompt context, well inside any reasonable budget.

- **Structured output via `response_schema`.** Gemini's JSON-mode is
  invoked by passing the Pydantic `MirrorReadingPayload` (two fields:
  `summary`, `flagged_observations`) as `response_schema` in the SDK's
  `config` dict. The SDK handles schema constraint and parsing; the
  caller's `_extract_payload` defensively handles three response shapes
  (Pydantic instance, dict, JSON-string fallback) because the SDK's
  exact behaviour for `response.parsed` varies across versions. The
  payload/envelope split is what makes the tests doable without a live
  API: the LLM-fillable surface is just two fields, mockable via a
  fake response object; the envelope (`run_id`, `timestamp_ms`,
  `agent_step_range`, `n_episodes_read`, `model_used`, `schema_version`)
  is what the `MirrorCaller` adds afterward. `MirrorReading` is frozen
  and `extra="forbid"` — the same record discipline as the Phase 1
  schemas.

- **Bears on Phase 8 smoke and Probe 2.** Phase 8's MPS smoke can
  optionally call `MirrorCaller.read_recent` and
  `MirrorCaller.write_reading` once after the runner has produced a few
  hundred steps, gated on `GEMINI_API_KEY` being present (the test in
  `tests/test_mirror_caller.py::test_live_mirror_call_returns_reading`
  is the template — skipped without the env var, no CI cost). The
  smoke's purpose is platform correctness, not mirror calibration, so
  the mirror call should be optional and its content should not gate
  the smoke. For Probe 2: the `MirrorReading.schema_version` is
  independent of the `AgentStep` schema version (it's the mirror's own
  stream), so Probe 2's frozen-criteria additions can bump
  `MIRROR_READING_SCHEMA_VERSION` to `0.2.0` without affecting the
  agent_step pipeline at all. The current envelope already carries
  enough context (run id, timestamp, env_step range, model used) that
  Probe 2 can correlate readings against the parquet timeline without
  re-reading the records.

### What's now closed and what's now newly open

Closed:

- The Gemini/Anthropic question for the mirror at Probe 1: Gemini, with
  the model name configurable. Documented in
  `kind/mirror/caller.py`'s docstring and reflected in the updated
  `.env.example`.
- The digest content and shape (per-episode aggregates, scalar-only
  sample records, no high-dim fields).
- The payload/envelope split and the JSON-mode `response_schema`
  approach.
- The `readings.jsonl` append path and its directory creation
  semantics.
- Whether to pre-compute mirror-side statistics or include raw records
  (compute the digest; raw records remain in parquet for any consumer
  that wants them).

Newly open:

- Whether the runner should call the mirror at any cadence (e.g., once
  per N checkpoints) for a Probe 1 sanity reading, or whether the
  mirror should remain manual-only until Probe 2's frozen criteria
  arrive. The plan §2.10 specifies no in-loop prompting at Probe 1; the
  current implementation honours this. Phase 7's journal scaffold may
  add a CLI script that calls the mirror post-hoc; integration into the
  runner stays deferred.
- Whether Phase 7's eyeball helpers should share the digest function
  with the mirror caller. The shape of the digest is identical to what
  a human eyeballing a parquet shard would want; if Phase 7 just calls
  `_build_digest` directly that's clean. Currently the function is
  module-private (`_build_digest`); promoting it to public surface is a
  one-line change if Phase 7 wants it.
- Whether the response_schema with Pydantic `extra="forbid"` would have
  worked. We left `MirrorReadingPayload` without `extra="forbid"` to
  avoid potential SDK schema-translation issues; the `MirrorReading`
  envelope still has `extra="forbid"` because we construct it
  ourselves. If a future SDK update makes `extra="forbid"` work cleanly
  on the payload side, tightening it is desirable.
- Whether the deprecation warning from `google.genai.types` (Python
  3.14's `_UnionGenericAlias` slated for removal in 3.17) will become
  an error before the project is done with Probe 1. It's a third-party
  warning, surfaces during test collection, and does not affect
  correctness. Pinned for awareness.

---

## Phase 7 — eyeball helpers and journal scaffolding (2026-05-02)

Five CLI helpers in `kind/observer/eyeball.py`, the digest lift in
`kind/observer/digest.py`, and the README at
`docs/workingjournal/README.md`. 310 tests pass (was 263 + 47 new);
mypy `--strict` clean across 22 source files. Phase 7 is tooling, not
architecture; three points worth recording:

- **Digest moved observer-side, public surface.** Phase 6's
  `_build_digest` lived inside the mirror caller; the same shape is
  what `show_run_summary` and `show_episode_summary` want. Now
  `kind/observer/digest.py` exposes `build_digest` and
  `compact_record_repr`; the mirror caller imports from there. The
  decision is that the digest is what the *substrate* exposes to
  whichever reader is running (LLM or human) — both see the same
  shape, neither sees the actor's view. Function bodies lifted
  verbatim so the regression on the lift is structural.

- **Eyeball helpers are scalar-summary by default; ASCII for dreams.**
  High-dim vectors (`h_t`, `z_t`, `q_params_t`, `p_params_t`,
  `kl_per_dim_t`, `encoder_embedding_t`) are excluded by default and
  summarised (shape + mean/std/min/max + first 3 + last 3) only when
  explicitly requested via `fields=`. Action distributions print as
  `up=N down=N left=N right=N stay=N`. Dream `sequence_decoded_obs`
  renders as 16×16 ASCII via the ramp `' .:-=+*#%@'` (32×32 → 16×16
  by integer mean of 2×2 blocks). No third-party formatters; pure
  stdlib + pyarrow.

- **Choices that bear on Phase 8.** (a) The smoke can optionally call
  `show_run_summary` at end-of-run for a sanity print — read-only,
  ~0 runtime cost. (b) The dream ASCII may read as noise against a
  real decoder rather than the fixture's gradient; if so, the
  fallback is to skip the rendering and print latent stats only. (c)
  `show_run_summary`'s early/late KL window is the first vs last
  quarter of records — if the early window is dominated by warmup
  noise the one-line revision is to skip the warmup span. All three
  are post-smoke decisions, not pre-smoke ones.

---

## Phase 8 — gate audit and MPS smoke script (2026-05-03)

The five named gate tests audited against plan §4 and the day-one
MPS smoke written as `scripts/smoke_mps.py`. 323 tests pass (was 310
+ 13 new); mypy `--strict` clean across 23 source files (was 22 +
`scripts/smoke_mps.py`). Phase 8 is structural completion only — the
smoke is *ready to run*, not *run*. Four points:

- **Audit findings.** Tests #1, #3, #4 cover §4 verbatim; the existing
  scopes are exact matches. Test #2 is split across
  `test_agent_forward.py` (world-model side) and `test_actor.py`
  (actor + ensemble side) — plan §4 names either filename as
  acceptable, and the cumulative coverage matches the spec. Test #5
  covers the 200-step run, mid-run checkpoint, four-stream emission,
  and required-files check; resume verifies parameter equality
  byte-for-byte against the disk weights file. The §4 wording
  "yields identical RNG state" was implicit (RNG is loaded via
  pickle but post-load state-equality wasn't asserted), so a focused
  `test_smoke_resume_loads_identical_rng_state` was added that pins
  `torch.get_rng_state()`, the runner's sample-RNG, Python `random`,
  and NumPy's RNG state byte-for-byte against the committed pickle.
  Trajectory match across the resume boundary remains documented as
  a known limitation per Phase 5's journal entry — not asserted, by
  design.

- **Gate summary meta-test.** `tests/test_gate_summary.py`
  parametrises over a table of `(label, module, test_name)` triples
  and asserts each gate test exists by name and is callable. Ten
  parametrisations cover the five §4 gates including the multi-test
  fan-out in #2 and #5. If a future refactor renames or removes any
  of the named tests, this meta-test fails loudly rather than letting
  the gate silently shrink.

- **Smoke structure.** `scripts/smoke_mps.py` runs at production
  sizes (h=200, z=16, K=5, batch=16, sequence=32) on Mac MPS with
  `torch.backends.mps.is_available()` as the entry guard — refuses
  to run on a non-Mac with a clear message and a non-zero exit code,
  no silent CPU fallback. The hot loop runs 100 RSSM training steps
  (forward + world-model backward + ensemble loss + ensemble
  backward) with `warnings.catch_warnings()` filtering for any
  warning whose text matches one of four MPS-fallback patterns
  (`MPS backend`, `fall back`, `fallback`, `PYTORCH_ENABLE_MPS_FALLBACK`).
  Per-step KL and recon-loss values are checked for finiteness as
  each step completes; non-finite values short-circuit the loop. All
  four telemetry sinks are exercised after the hot loop with one
  synthetic record each, then read back via the schema models — a
  sink-failure-during-warning-capture would otherwise contaminate
  the warning stream. The companion sanity test
  `tests/test_smoke_mps_script.py` confirms the script exists and
  exposes a `main()` entry point; the script itself runs only
  manually.

- **Soft bar on wall time, not a fail.** The 60s-for-100-steps soft
  bar is a stderr `print` if exceeded — the script's exit status is
  unchanged. This honours plan §5's "order-of-magnitude check, not a
  benchmark" framing: the human builder reads the wall time and
  decides whether to drop sizes per plan §6's "Revisit when" column
  (h=128, z=8 first), or to revisit the substrate variant per
  pre-probe1.md's "On stalls" framing. Hard fail is reserved for
  fallback warnings, NaN/Inf, sink errors, and missing gradients —
  the things that mean *something is structurally wrong*, distinct
  from "the canonical sizes are slow on this Mac".

- **Interpreting the smoke is a separate journal entry.** Phase 8
  ends with the substrate's structural shape complete and the smoke
  ready for the human builder to invoke (`python scripts/smoke_mps.py`
  on the Mac). What the smoke surfaces — wall time numbers, whether
  the hot path fires any fallback warnings, whether the run reveals
  KL pinned at the floor or some less-anticipated structural
  problem — and the decision between "default calibration off" and
  "regime assumptions wrong" per pre-probe1.md's two-failure-mode
  framing, are recorded in a post-Phase-8 entry written after the
  smoke has actually run.

---

## Post-Phase-8 — first MPS smoke run (2026-05-03)

Smoke ran clean on first invocation. The full output:

```
[smoke] mps detected — running 100 training steps at h=200, z=16, K=5, batch=16, seq=32
[smoke] mps ok | wall=13.06s | per-step=130.6ms | sinks ok | shapes ok
```

No stderr output between startup and summary. That means: no MPS-fallback
warnings on the hot path; no NaN/Inf in KL or recon at any of the 100 steps;
all four telemetry sinks wrote and read back valid records; backward populated
gradients on world-model and ensemble in every step.

### What's now closed

- **The synthesis's open question on MPS performance** ("MPS performance for
  RSSM operations at Probe 1 batch sizes is unbenchmarked") is settled:
  130.6 ms per training step at production sizes (h=200, z=16, K=5, batch=16,
  seq=32). 13s for 100 steps is well inside the 60s soft bar — comfortable
  enough margin that future tuning work (longer sequences, larger batches,
  the actor's imagination loop joining the training step) has headroom on
  the same machine before the bar matters.

- **The substrate decision held up operationally on the canonical machine.**
  No MPS-fallback patterns (`MPS backend`, `fall back`, `fallback`,
  `PYTORCH_ENABLE_MPS_FALLBACK`) appeared in the warning stream during the
  hot loop. Every op the world model and ensemble exercise on a
  forward+backward+optimizer-step at these sizes is supported natively on
  MPS — there's no platform-specific gap to chase.

- **Defaults from plan §6 stand.** None of the smoke's signals suggested
  revising h, z, K, free bits, sequence length, or batch size for
  substrate-correctness reasons. Plan §6's "Revisit when" column lists
  observable triggers (KL pinned at floor, ensemble saturation, OOM) — none
  fired here. The defaults are still pinned at their starting values.

- **Pre-probe1.md's two failure modes both did not fire.** Failure mode 1
  (default calibration off) would have shown as KL pinned at the free-bits
  floor, ensemble disagreement saturating, or regrowth-events too sparse —
  none observable in this smoke (which uses random observation tensors, not
  env-rendered ones, so the regrowth-events check doesn't apply at all).
  Failure mode 2 (regime assumptions wrong) would have shown as latent
  collapse free-bits couldn't fix, ensemble disagreement saturating in 100
  steps regardless of K, or a decoder producing dreams unlike the env — none
  observable here either, again with the caveat that the smoke is not a
  dynamics check.

### What this run does *not* tell us

The smoke uses dummy `torch.rand` observation tensors, not actual GridWorld
renderings. It is a *platform correctness* and *gradient-flow* gate, not a
dynamics or regime check. Specifically:

- **Posterior collapse / KL behavior under real dynamics is untested.** With
  random uniform observations there is no temporal structure for the
  posterior to compress, so the KL signal here is not informative about
  whether free bits will hold the posterior open against the actual env's
  much-lower-entropy 32×32 grayscale stream. That question moves to the
  first end-to-end run against the env.

- **Ensemble disagreement saturation is untested.** K=5 heads on random
  inputs disagree by initialization-noise; whether disagreement saturates
  within an episode under real Io-driven trajectories is the live question
  Probe 1's actual run will surface. Plan §6's "Revisit when" trigger
  (try K=3 if disagreement saturates) remains open.

- **Decoder legibility under env dynamics is untested.** The smoke writes a
  synthetic DreamRollout to the sink and reads it back; it does *not*
  evaluate whether the decoder's output is recognizable as a 7×7 ego-centric
  grid view rendered to 32×32. The eyeball helpers' ASCII-rendering path
  (Phase 7's `show_dream_summary`) is what tells us that, against actual
  trained-on-env weights — Probe 1's run + a post-run mirror call.

### What's now newly open

Not much from this specific run — the smoke is narrow by design and it
didn't surface unexpected behavior. The next live decisions all move to the
first Probe 1 run against the actual env-server:

- Does the world model's posterior compress the env's structure
  legibly within (say) 5k env steps, or does KL pin at the floor?
- Does ensemble disagreement saturate within the first few hundred steps
  of fixed-start episodes, suggesting the start-cell-`(3, 3)` choice
  over-constrains the trajectory distribution (Phase 2a's open question)?
- Does the mirror's first reading carry signal about the run, or does it
  flag the data as too sparse to read?

These are not Phase 8 questions — they belong to the first env-coupled run
and the post-run mirror call.

### Substrate is operationally complete

With the smoke clean, every Probe 1 substrate piece is now exercised at
production sizes on the canonical platform: schemas, sinks, RSSM forward and
backward, ensemble training, telemetry round-trip, MPS compute. The
remaining work is running the env-coupled loop end-to-end and reading what
surfaces. Probe 1's plumbing is built; what comes next is investigation.

---

## Post-Probe-1 — first env-coupled run (2026-05-03)

First end-to-end run on the canonical Mac. `run_id =
probe1-20260503-123926`. 5000 env steps, 25 episodes (h=200 each),
warmup=200, checkpoint cadence=2500, dream cadence=1000, seed=42, device=mps.
Wall: 762.2s (~152 ms/step). All four telemetry streams wrote, the mid-run
checkpoint committed, and four dream rollouts emitted at the expected
seed_steps {1000, 2000, 3000, 4000}. No fallback warnings on the hot
path; no NaN/Inf. The plumbing did what the smoke promised it would do
under the random-tensor proxy.

This entry is written from the run.log and summary.txt only. The eyeball
helpers have not yet been run against the telemetry directory; questions
that need per-episode or per-step traces (KL trajectory shape, per-state
entropy, dream legibility under env weights) are listed as open below
rather than answered here.

### What ran end-to-end and what surprised

- **No posterior collapse on the env stream.** The summary's
  `kl_aggregate_t` early/late means are 10.23 → 15.00 (Δ +4.77 across the
  first vs last episode). This is the headline answer to Phase 8's first
  open question ("does KL pin at the floor under real dynamics?"). Failure
  mode 1 from `pre-probe1.md` (default calibration off → KL pinned at the
  free-bits floor) did not fire on the canonical machine at the canonical
  sizes. The posterior is encoding more env structure over time, not less,
  on a 32×32 grayscale GridWorld stream that was the live worry.

- **What I can't tell from two endpoints.** The early/late summary is the
  *first* episode's mean against the *last* episode's mean. A monotone
  rise of +4.77 across the run's two endpoints is consistent with both
  "healthy compression rising and plateauing" and "still climbing at step
  5000, posterior memorising rather than compressing." `eyeball summary`
  prints per-episode means; the slope between them is the read I haven't
  done yet. Flagged as the first thing to look at.

- **Wall-time cost of env coupling is small.** 762s / 5000 steps ≈ 152
  ms/step on the canonical machine. The Phase-8 smoke at random-tensor
  sizes-only was 130.6 ms/step. The ~22 ms/step of additional cost from
  env round-trip + actor forward + sink writes is inside what the smoke's
  60s soft bar already had headroom for. No need to revisit plan §6
  defaults on wall-time grounds.

- **Action distribution is mixed, not pinned.** Overall counts: up=1550
  (31%), right=1224 (24%), down=1168 (23%), left=987 (20%), stay=71 (1.4%).
  The slight up-bias is mild. `stay` at 1.4% is the only unusual entry —
  whether that's because the actor learned to keep moving or because the
  reward shaping made `stay` dominated, the per-state distribution will
  tell. Not surfaced here; flagged for the eyeball pass.

- **Final-step `entH = 0.108` is low.** The progress bar's final policy
  entropy is near-deterministic. Combined with the fixed start cell `(3, 3)`
  this is exactly Phase 2a's "does start-cell choice over-constrain the
  trajectory distribution?" surfacing. The action-distribution mix
  argues *against* full collapse (a deterministic policy from a fixed
  start would produce a single repeated trajectory — much sharper action
  histogram), so the most likely read is per-state entropy is low *given
  the state*, not that the policy is degenerate. But this is the second
  thing I'd want eyeballed: late-episode action sequences against
  early-episode action sequences, looking for "Io found a good policy"
  vs "Io is stuck in a loop."

- **Final-step `intr = 0.5610` is non-zero.** Ensemble disagreement isn't
  saturated to zero, which is what plan §6's K=5 default was supposed to
  preserve. Whether it's saturated *upward* (heads not converging at all
  — the other failure direction) needs the per-step trace, not the final
  scalar.

- **The substrate decision held up operationally on a real env.** The
  Phase-8 smoke had already settled MPS performance for the substrate's
  ops in isolation; this run extends that to the env-coupled loop with
  transport, sinks, dream emission, and checkpoint commit all firing
  inside the wall budget. Nothing about the run nudges the inheritance
  question (Probe 0's "build on RSSM-shaped Dreamer-V3 substrate") in
  either direction. Carries forward into Probe 2 unchanged.

### What this run promotes from "default" to "settled"

- **Wall-time budget on the canonical machine.** ~150 ms/step at
  production sizes is the working number for the env-coupled loop.
  Future probes can plan run lengths against this without re-measuring.
  5000 steps in ~13 minutes; 50000 steps would be ~2 hours — well inside
  what a single sitting allows.

- **Episode horizon h=200 holds operationally.** 25 episodes from 5000
  steps with 26 resets (initial reset + per-episode) matches exactly what
  the spec said. No early terminations, no off-by-one in episode
  boundaries. The 200-step horizon as a default is settled at this scale.

- **Dream cadence at 1000 is right for a 5k-step run.** Four dream
  rollouts is enough to read the decoder's progression (initial, two
  middle, late) without dominating the telemetry footprint or the wall
  budget. If 5k stays the standard run length, this cadence stays.

- **Checkpoint cadence at 2500 is right for a 5k-step run.** One mid-run
  checkpoint plus the implicit run-end state is enough for resume
  exercise; doubling it would have been noise.

### What's now newly open

- **KL trajectory shape, not just endpoints.** Δ +4.77 over the run is
  consistent with more than one underlying story. The eyeball pass needs
  to look at the per-episode `kl_aggregate_t` mean across all 25
  episodes: monotone rise that plateaus by ~episode 15-20 is the healthy
  read; still rising at episode 25 is "watch for posterior memorising."
  Bears on Probe 2's calibration of the perturbation-driven KL spike —
  the baseline KL slope under Probe 1's flat dynamics is the comparison
  Probe 2 will need.

- **Per-state policy entropy under the fixed start.** `entH = 0.108`
  final is low; the action histogram is mixed. The simplest hypothesis
  is that entropy is concentrated state-by-state and varies with state
  — i.e. the actor has learned different distributions in different
  cells. The alternative is that the population's stochasticity (each
  episode draws a fresh internal-stochasticity sample) is doing the
  spreading and the actor's own policy is sharper than the histogram
  suggests. `eyeball episode -e 24` against `-e 0` is the direct read.

- **Decoder legibility under env weights — still untested.** Phase 8's
  smoke wrote a synthetic DreamRollout to the sink and read it back; it
  did not check whether the decoder produces 7×7 ego-centric grid views
  recognisable as the env's renderings. Four dream records now exist
  with weights that have actually trained on the env. The
  ASCII-rendering path in `eyeball dream` is what tells us. This is the
  third Phase 8 open question still open.

- **Whether the start-cell `(3, 3)` is over-constraining the trajectory
  distribution.** Phase 2a's open question. Probably can't be cleanly
  answered from a single fixed-start run — the comparison is against a
  varied-start variant, which Probe 1's spec doesn't include. Carry
  forward to Probe 2's design: if Probe 2 introduces variation by
  perturbation rather than by resampling start cells, that's a deliberate
  choice we should record once.

- **The mirror's first reading on real telemetry — not yet attempted.**
  The mirror caller exists (`kind/mirror/caller.py`) and shares the
  digest with the eyeball helpers. Calling it on this run's telemetry is
  the third Phase 8 open question's other half (not just *can the human
  read the dreams*, but *does the mirror's reading carry signal or flag
  the data as too sparse*). Not done in this entry; bears on the
  Probe 1 → Probe 2 transition.

### Note on the substrate decision

Operationally the substrate decision held: MPS at production sizes,
RSSM forward/backward, ensemble training, transport, sinks, dream
emission, checkpoint commit — all fired cleanly inside a 13-minute
window on the canonical machine. The architectural question (is
RSSM-shaped Dreamer-V3 the right substrate to inherit from for
CA-MAS-flavoured investigation?) doesn't get answered by a single
end-to-end run, but nothing in the run argues against it. The next
evidence comes from Probe 2's calibration: whether perturbation-driven
KL spikes are legibly distinguishable from this run's baseline KL slope
will be the first real stress on the substrate's ability to surface the
signals the investigation needs.

### Immediate next steps

Three reads, in order:

1. `python -m kind.observer.eyeball summary runs/probe1-20260503-123926/telemetry`
   — the per-episode KL trajectory.
2. `python -m kind.observer.eyeball episode runs/probe1-20260503-123926/telemetry -e 0`
   and `-e 24` — the early vs late episode shape.
3. `python -m kind.observer.eyeball dream runs/probe1-20260503-123926/telemetry -i 3`
   — the late-run decoder render.

Then the mirror call against the same telemetry directory. The result
of those reads is what closes (or re-opens) the three open questions
above and feeds into the Probe 2 pre-probe entry.

---
