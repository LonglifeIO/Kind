# Probe 1.5 — Working Journal

*Hand-written notes captured during the Probe 1.5 build. One section per
phase, with the structural decisions made mid-build, surprises that came
up, and what is now decided that was open. The pre-build risk-framing
for Probe 1.5 lives in `pre-probe1_5.md` (if/when written); the design
synthesis is at `docs/decisions/Kind_probe1_5_synthesis.md` (v2 settled);
the implementation plan is at
`docs/plans/Kind_probe1_5_implementation_plan.md` (v2). This document is
the running log from Phase 0 forward.*

## Phase index

Each phase entry begins with `## Phase N` (heading literal — exact
match) and ends with `---`. Inline mentions of phase numbers inside
earlier phases' "newly-open" sections look like `Phase 3` (no leading
`##`); they are *not* phase boundaries.

- **[Phase 0](#phase-0)** — schema bump (2026-05-03)
- **[Phase 1](#phase-1)** — SelfPredictionHead + EMA target (2026-05-05)
- **[Phase 2](#phase-2)** — PolicyView/TelemetryView + Actor extension (2026-05-06)
- **[Phase 3](#phase-3)** — runner integration + checkpoint compat (2026-05-06)
- **[Phase 4](#phase-4)** — digest + eyeball extension (2026-05-06)
- **[Phase 5](#phase-5)** — failure-mode control variant: frozen-target (lean) (2026-05-06)
- **[Phase 6](#phase-6)** — five gate tests + MPS smoke (2026-05-06)
- **[Phase 7](#phase-7)** — first env-coupled Probe 1.5 run + first mirror call (2026-05-06)
- **[Phase 7.5](#phase-75)** — plan §6 row 15 escalation: small-Gaussian init on the actor's new column + re-run + second mirror call (2026-05-07)

---

## Phase 0

*schema bump · 2026-05-03*

The schema additions in `kind/observer/schemas.py` and the
byte-stable JSON Schema export at `schemas/v0.2.0.json`. Three optional
fields land on `AgentStep` (`self_prediction_t: list[float] | None`,
`self_prediction_error_t: float | None`, `self_prediction_error_masked_t:
bool | None`) and one reserved optional field on `DreamRollout`
(`sequence_self_prediction: list[list[float]] | None`). `SCHEMA_VERSION`
moves from `"0.1.0"` to `"0.2.0"`. A `model_validator(mode="after")` on
`AgentStep` (`_enforce_v2_required_fields`) raises `ValidationError` when
a record stamped `"0.2.0"` has any of the three new fields `None` — the
implementation plan §3.3 "mixed-version writer rejection" check at the
writer side. `schemas/v0.1.0.json` stays on disk untouched as the frozen
Probe 1 export. 17 new tests in `tests/test_schemas.py` pass (the eight
the plan §2.1 names plus a parametrised triplet on the validator plus
backward-readability sanity); 331/332 in the full test suite (1
pre-existing skip on `test_mirror_caller`); mypy `--strict` clean across
22 source files (unchanged from Probe 1 — no new modules at this phase).

### Naming the Probe 1 version: `PROBE_1_SCHEMA_VERSION`

The plan §1 names "schema bump before any component that emits or
consumes the new fields" as the load-bearing prerequisite. With
`SCHEMA_VERSION = "0.2.0"` and the validator firing on `"0.2.0"`
records, the Probe-1-era writers that propagate `SCHEMA_VERSION`
verbatim into their record envelopes (`runner.py:818` for `AgentStep`,
`runner.py:735` for `DreamRollout`, `runner.py:893` for the checkpoint
metadata file, `replay.py:316` for `ReplayMeta`, `env_server.py:357`
for `WorldEvent`, `scripts/smoke_mps.py` for the four synthetic
records) would after Phase 0 stamp `"0.2.0"` records that the new
validator rejects (the AgentStep ones; the others have no new fields
but stamping `"0.2.0"` from a Probe-1-shaped writer would be
semantically misleading regardless).

Three options surfaced:

1. *Update each writer to populate the new fields with placeholder
   values (e.g. zero vector, zero scalar, masked True).* This makes
   `"0.2.0"` records emerge from a Probe-1-shaped substrate that has
   no `SelfPredictionHead`. The records would falsely claim the v2
   shape; a downstream reader couldn't tell a placeholder-from-Phase-0
   from an early-training real value.

2. *Scatter string literals `"0.1.0"` across the writers.* Brittle;
   any future audit that searches for the name `SCHEMA_VERSION`
   misses these. Phase 3 would have to remember to flip each one.

3. *Introduce a named constant `PROBE_1_SCHEMA_VERSION = "0.1.0"`
   alongside `SCHEMA_VERSION`, and have Probe-1-era writers reference
   it explicitly until they're rewired for Probe 1.5.* Symmetric with
   how the synthesis treats v1 and v2 (preserved in parallel files,
   not deleted); the constant is the discipline marker — "this writer
   is Probe-1-shaped, has not been migrated."

I picked option 3. Both constants are exported from
`kind/observer/schemas.py`'s `__all__`; the `runner.py`, `replay.py`,
`env_server.py`, and `scripts/smoke_mps.py` writers now reference
`PROBE_1_SCHEMA_VERSION` and will move to `SCHEMA_VERSION` when their
respective phases (Phase 3 for runner; later for env_server / smoke
which are Probe 1 substrate not Probe 1.5 substrate) wire the new
fields' producer machinery in. The semantics stay honest: the constant
a writer stamps tells the reader what shape of record to expect.

The decision is journaled here (rather than just in the diff) because
it's the kind of thing that drifts back the moment someone writes
`schema_version=SCHEMA_VERSION` in a future probe without realizing
their writer hasn't been migrated. The named-constant convention plus
the writer-side validator are the dual scaffolding — the constant
makes the intent explicit at construction; the validator catches the
mismatch at runtime.

### The validator's scope: AgentStep only, not DreamRollout

`DreamRollout` gains a reserved field (`sequence_self_prediction:
list[list[float]] | None`) but no validator on the v2 path. The
synthesis §1.5 commits Probe 1.5 to running the self-prediction head
*only during waking* — dream rollouts at Probe 1.5 carry
`sequence_self_prediction = None`, which is the legitimate Probe 1.5
shape (not a missing-field bug). Probe 3 may populate the field if its
design extends self-prediction to imagined trajectories. So `"0.2.0"`
DreamRollouts with `sequence_self_prediction = None` are valid; no
writer-side enforcement applies. If a future probe makes the field
required, the validator pattern is right here (`AgentStep` is the
reference) and can be added in a one-line change.

### The masked-step sentinel and the validator's "non-None" check

A first-step-of-episode AgentStep record carries
`self_prediction_error_t = 0.0` (sentinel zero) and
`self_prediction_error_masked_t = True`. The scalar is non-None, so
the validator's "non-None" check passes; the masked flag is what
discriminates "no empirical reading available, sentinel zero" from
"empirical near-zero reading" at every analyzer downstream of the
writer (digest, eyeball helpers, the future
`show_self_prediction_conditioning` helper, the counterfactual probe).
The plan §6 row 16 settles this convention; the validator and the test
`test_agent_step_v0_2_0_round_trips_with_masked_true` are the two
places it's pinned in code.

The alternative considered — using `None` for the scalar on the first
step and tightening the validator to "either all three None
(Probe 1) or all three non-None (Probe 1.5)" — would have required
the actor's PolicyView to handle a `None` scalar at the first step of
each episode, which complicates the actor's input layer (the new
column would need a None-handler) and pushes mirror-side analyzer
discrimination of "Probe 1 record" from "Probe 1.5 first-step" out to
the schema_version field anyway. The sentinel-plus-flag convention
keeps the actor's input dimensionally consistent across all steps and
keeps the discrimination at the masked-flag field where it belongs.

### JSON Schema export's title

The previous title was `"Kind Probe 1 Telemetry Schemas"` (from
Probe 1's Phase 0). The schema is shared across probes; the title
shouldn't pin to a probe number. Bumped to `"Kind Telemetry Schemas"`.
This is the kind of change that risks invalidating downstream tooling
that pinned the literal title — but no such tooling exists yet
(neither `digest.py` nor `eyeball.py` nor `caller.py` reads the
title); the byte-stable export test catches the change in the file
contents either way (`schemas/v0.2.0.json` is the new on-disk export
and is the file `test_json_schema_export_matches_checked_in_v0_2_0_file`
compares against).

### Two test surfaces for the validator

The "mixed-version writer rejection" requirement lands as a
parametrised triplet (`test_mixed_version_writer_rejection_raises_validation_error`,
one parameter per missing field) plus an "all three missing"
companion. The parametrised form ensures the validator's error
message names *the specific* missing field — writer-side debuggability
matters when the error fires inside a 5000-step run from a `runner.py`
emission deep in the hot loop. The companion confirms the
all-three-missing message names every field.

The dual test (`test_v0_1_0_record_with_new_fields_none_does_not_raise`)
pins the validator's scope: it runs only on `"0.2.0"` records. A
`"0.1.0"` record with `self_prediction_t = None` (the Probe 1
backward-readability case) constructs without raising. Without this
test, a future tightening of the validator to "any record with the
new fields None raises" would silently break Probe 1 record
deserialization.

### What's now closed

- The schema's structural shape at `0.2.0`: three new optional fields
  on `AgentStep`, one reserved optional field on `DreamRollout`.
- The writer-side discipline: `AgentStep._enforce_v2_required_fields`
  raises on `"0.2.0"` records with any new field `None`.
- The Probe 1 backward-readability path: `"0.1.0"` records with the
  new fields absent deserialize cleanly; the validator does not fire.
- The first-step-of-episode masking convention: scalar=0.0 sentinel
  with masked-flag True; downstream analyzers discriminate via the
  flag.
- The `PROBE_1_SCHEMA_VERSION` named constant and the convention that
  Probe-1-era writers reference it explicitly until their respective
  Probe 1.5 wiring lands.
- `schemas/v0.2.0.json` checked in (13921 bytes, byte-stable);
  `schemas/v0.1.0.json` preserved as the frozen Probe 1 export.
- All 331 pre-Phase-0 tests + 17 new Phase 0 tests pass; mypy
  `--strict` clean.

### What's now newly open

- *Phase 1*: `SelfPredictionHead` + EMA target sibling on
  `kind/agents/world_model.py`. The schema reserves slots for the
  head's output (`self_prediction_t`) and its loss
  (`self_prediction_error_t`); Phase 1 builds the producer machinery
  whose output the runner (Phase 3) will pipe into the new fields.

- *Whether the validator should also catch the inverse*: a `"0.1.0"`
  record with one of the new fields populated (which would be a
  writer-side bug — Probe 1 doesn't have a head). Currently the
  validator only enforces the `"0.2.0" → all non-None` direction;
  `"0.1.0" + populated` is silently accepted. Held open: the symmetry
  is appealing but the cost of a misfire on a hand-constructed test
  fixture that legitimately needs a partial record outweighs the
  benefit at this phase. Reconsider if Phase 3 surfaces a wiring bug
  that would have been caught by the inverse check.

- *Whether Phase 3's runner update should rename* `runner.py` and
  `replay.py` and `env_server.py` and `smoke_mps.py` *back to
  `SCHEMA_VERSION`* en bloc, or piecewise as each substrate piece
  gains its Probe 1.5 producer. The plan dependency graph (§1) points
  to piecewise (Phase 3 for the runner's AgentStep/DreamRollout/
  checkpoint-file emission; the env-server's WorldEvent emission has
  no Probe 1.5 producer changes and stays at PROBE_1_SCHEMA_VERSION
  through Probe 1.5; replay.py likewise). The naming convention
  carries the timing signal automatically — what's still
  `PROBE_1_SCHEMA_VERSION` in a Phase 3+ codebase is Probe-1-shaped
  by intent, not by oversight.

- *Whether the JSON Schema's per-field titles should be normalized
  away from PyDantic's auto-generated `"Title Case With Spaces"`
  convention*. The current export carries `"title": "Self Prediction
  Error Masked T"` for the new fields, matching the convention
  Probe 1's Phase 0 inherited from Pydantic's defaults. No downstream
  reads these titles yet; held until they cause actual confusion.

---

## Phase 1

*SelfPredictionHead + EMA target · 2026-05-05*

The substrate-side machinery the schema reserved slots for in Phase 0
now exists. `kind/agents/world_model.py` gains: a `SelfPredictionHead`
(small MLP `h_dim → h_dim`, two ELU-activated hidden layers); EMA-
tracked sibling copies of the encoder + GRU cell (`target_encoder`,
`target_gru_cell`) initialised from the online network at construction
and held with `requires_grad=False`; `_initialize_ema_target_from_online`
(called by `__init__`; will be re-called by Phase 3's checkpoint-load
when resuming from a Probe 1 checkpoint); `_update_ema_target` (in-
place EMA convex-combination update, called by the runner after the
world-model optimizer step in Phase 3); `compute_self_prediction_target`
honoring all three `self_prediction_target_mode` values (`"online"`,
`"frozen"`, `"environmental"`); a `_frozen_projection` `nn.Parameter`
allocated only when `target_mode="frozen"` (random-orthogonal,
`requires_grad=False`); a `_environmental_projection` `nn.Linear`
allocated only when `target_mode="environmental"` *and*
`embed_dim != h_dim` (frozen weights). `WorldModelStep` gains
`self_prediction: Tensor`. `WorldModelConfig` gains four fields
(`self_prediction_hidden=200`, `ema_decay=0.99`,
`self_prediction_target_mode="online"`, `self_prediction_loss_form="cosine"`)
matching the synthesis §6 / plan §6 defaults. `WorldModel.loss` returns
a five-key dict — the four Probe 1 keys plus `self_prediction_loss`,
computed only when the runner passes `target_h_next` (when the kwarg is
absent, the loss is a zero scalar and `total` does not include it; the
runner is what sums `wm_total + λ_self * self_prediction_loss` per
plan §2.4, not `loss()` itself).

12 new tests in `tests/test_world_model.py`: gate test #1
(`test_gate_self_prediction_forward_shape` — head output shape, all
three target modes' shape contract, cosine + MSE loss-form arithmetic
on hand-computed inputs), gate test #2
(`test_gate_ema_target_update_mechanics` — init-time identity, single-
step convex-combination equality, monotonic-with-rate convergence over
10 EMA updates with online held fixed), gate test #3
(`test_gate_self_prediction_loss_decreases` — 100 training steps with
λ_self=1 on a fixed synthetic sequence reduce the cosine loss
measurably without recon, KL, or self-prediction loss going non-finite),
plus a regression
(`test_probe_1_path_still_passes_at_target_mode_online`) that the loss
without `target_h_next` does not visit the head's MLP via backward
through `total`, plus per-component unit tests on the head, the EMA
target's `requires_grad=False` discipline, the `_frozen_projection` and
`_environmental_projection` allocation predicates, the orthogonality of
the random-orthogonal projection at construction, and the per-mode
target-arithmetic contracts. 343/344 tests pass (1 pre-existing skip on
`test_mirror_caller`); mypy `--strict` clean across the same 22 source
files (the new module surface lives in `kind/agents/world_model.py`,
which was already on mypy's checked list).

### Resolving the indexing of bar{h}_{t+1} for the online target

The synthesis (§1.2) and plan (§2.2) both name the supervisory target
`bar{h}_{t+1}` and say it comes from "the EMA target's forward on the
current observation and previous (h, z, a), producing the next
deterministic recurrent state via the EMA-tracked GRU cell with the
EMA encoder's embedding as one of its inputs." Read literally, this is
ambiguous in two ways: (1) the inputs `(obs, h_prev, z_prev, a_prev)`
let the runner compute *one* GRU recurrence step under EMA params,
yielding what we'd index `bar{h}_t` (the EMA's version of the same
step's `h_t`), not `bar{h}_{t+1}` (the EMA's *next* step); (2) the
"with the EMA encoder's embedding as one of its inputs" framing
suggests the encoder enters the GRU directly, but our GRU's actual
inputs are `(z, action_emb)` with the encoder feeding `z` only via the
posterior network, which is *not* EMA-tracked.

I read this as the synthesis writing loosely about index labels. The
BYOL/SPR discipline the synthesis cites (Tian/Chen/Ganguli 2021;
Schwarzer et al. 2021) is "predict the EMA target's representation of
the same input." The cleanest realisation given the inputs the runner
will pass is: target = `target_gru_cell(cat[z_prev, action_emb(a_prev)],
h_prev)`. This is the EMA's analog of the online recurrence step, and
the "next" in `bar{h}_{t+1}` is the GRU's terminology (its output is
"the next state given the previous hidden") — not a temporal step
beyond the head's input `h_t`. The encoder enters via being EMA-tracked
(its slow drift shapes the posterior's `z` distribution over time),
not via direct participation in the per-call target computation. This
keeps the per-step (B=1) and batched (B=batch) call sites symmetric:
both compute the same EMA-recurrence step, just at different batch
sizes.

The action embedding is *not* EMA-tracked. Plan §2.2 names exactly two
EMA-tracked siblings: encoder and gru_cell. The action embedding is
shared with the online network; `compute_self_prediction_target` calls
`self.action_embedding(a_prev)` inside a `torch.no_grad()` block to
keep the gradient path clean, but the *parameters* it reads from are
the trainable online ones. Gate test #2's
`assert not hasattr(wm, "target_action_embedding")` pins this.

### The `loss()` method's auxiliary key vs the existing Probe 1 keyset

The Phase 1 change to `WorldModel.loss` widens its returned dict from
four keys to five. The existing Probe 1 gate test
(`test_gate_world_model_step_and_loss_and_backward` in
`tests/test_agent_forward.py`) asserted *exact* keyset equality on the
old four keys, so it had to be updated. I picked: extend the existing
test to also exercise the Probe 1.5 path (compute the target, pass it
in via `target_h_next`, expect `self_prediction_loss` in the dict,
backward through `total + self_prediction_loss`, check that
`requires_grad=False` parameters do *not* receive a `.grad`). This
keeps the test as a single end-to-end gate covering both the Probe 1
keys and the Probe 1.5 addition. The alternative — leave the existing
gate at four keys, force `loss()` to omit `self_prediction_loss` when
`target_h_next` is None — would have left the runner with an awkward
dict-shape conditional and made the keyset itself a mode-dependent
contract. Cleaner to let the auxiliary always be present (zero when no
target is passed), update the existing gate, and add a new regression
(`test_probe_1_path_still_passes_at_target_mode_online`) that pins the
narrower claim: when callers don't pass a target, `total.backward()`
does not visit the head's MLP. Both halves stay covered.

### Updating `test_imagine_restores_world_model_requires_grad_after_loss`

`tests/test_actor.py`'s `test_imagine_restores_world_model_requires_grad_after_loss`
asserts that after the actor's `imagine_and_compute_loss` returns,
every world-model parameter has `requires_grad=True`. With the Probe
1.5 EMA target's parameters carrying `requires_grad=False` *by design*,
the literal assertion fails. The fix tightens the assertion to its
actual intent: the *trainable-set* (the set of parameters that were
True before the call) is preserved across the call. The actor's
`_frozen_params` context manager temporarily sets every parameter's
`requires_grad` to False during imagination and restores them on
exit; the test's contract is that the restoration is faithful, not
that every parameter is unconditionally trainable. Snapshotting the
trainable-set before and asserting equality after is the right shape
for this, and it carries forward to any future Probe 1.5+/Probe 2
extension that adds more `requires_grad=False` parameters to the
world model.

### `WorldModelStep` instantiations across the test suite

Adding `self_prediction` as a required field on `WorldModelStep`
caused four pre-existing manual instantiations to fail
(`tests/test_agent_forward.py` × 2, `tests/test_views.py` × 2). Each
was updated to pass a freshly-allocated `torch.randn(batch, h_dim)` (or
`torch.zeros(...)` for the free-bits-arithmetic tests where the value
doesn't matter). No production code instantiates `WorldModelStep`
directly outside `WorldModel.step` itself, so the runner doesn't need
updating at this phase. The `step.self_prediction` field is what
Phase 2's `views.split` will read into `TelemetryView.self_prediction`
and the scalar derivative onto `PolicyView.self_prediction_error`.

### What's now closed

- The `SelfPredictionHead` architectural shape: 2 hidden layers at
  width `self_prediction_hidden=200` (plan §6 row 5 default promoted
  from "default" to "settled at this scale" — no per-component count
  knob the smoke would surface as wrong).
- The `ema_decay=0.99` default and the convex-combination EMA update
  rule (plan §6 row 2 default; gate test #2 confirms the rule's
  arithmetic and the per-step convergence rate match the configured
  decay to within float32 round-off).
- The `self_prediction_loss_form="cosine"` default (plan §6 row 3
  default; gate test #1 confirms the cosine arithmetic on hand-computed
  inputs). The MSE fallback is wired and tested.
- The `target_mode="online"` real Probe 1.5 path: target =
  `target_gru_cell(cat[z_prev, action_emb(a_prev)], h_prev)`;
  fully detached; per-call shape `(B, h_dim)` for any B.
- The `target_mode="frozen"` failure-mode-control path: target =
  random-orthogonal projection of the *online* `h_t`; the
  random-orthogonal allocation happens once at construction via
  `nn.init.orthogonal_`; Phase 5 will run this with the runner.
- The `target_mode="environmental"` failure-mode-control path: target =
  EMA-encoded next observation, with a frozen `nn.Linear(embed_dim,
  h_dim)` projection allocated only when `embed_dim != h_dim`. Phase 5
  will run this with the runner.
- The EMA target is initialised from the online network at construction
  and re-initialised by `_initialize_ema_target_from_online` for the
  Phase 3 Probe 1 → Probe 1.5 checkpoint-load path.
- The action embedding is *not* EMA-tracked (plan §2.2 names exactly
  encoder + gru_cell); confirmed by `assert not hasattr(wm,
  "target_action_embedding")` in gate test #2.
- `loss()` returns a five-key dict; `self_prediction_loss` is *not*
  folded into `total` (the runner adds `λ_self * self_prediction_loss`
  externally per plan §2.4). When `target_h_next` is None, the
  auxiliary is a zero scalar and `total.backward()` does not visit the
  head's MLP. Pinned by `test_probe_1_path_still_passes_at_target_mode_online`.
- 343 tests pass (the pre-Phase-1 342 plus 12 new minus the one
  superseded by the extended Probe 1 gate); the integration smoke
  passes unchanged (the runner doesn't yet wire the new path —
  Phase 3's territory — so the smoke's existing forward path still
  produces the existing four loss keys plus the new zero-valued
  `self_prediction_loss`, which the runner currently ignores).

### What's now newly open

- *Phase 2*: `TelemetryView` gains `self_prediction: Tensor` and
  `self_prediction_error_masked: bool`; `PolicyView` gains
  `self_prediction_error: Tensor` (scalar); `views.split` extends to
  populate all three; `Actor.forward(view: PolicyView)` consumes the
  scalar via concatenation and the input layer's weight matrix gains
  one column. Gate test #4 (opacity boundary preserved with the
  revised PolicyView field set) lands then. The current Phase 1
  build leaves `views.split` and the actor untouched — Io still reads
  exactly `(h, z)` until Phase 2.

- *The synthesis's "encoder propagating via the EMA encoder's embedding"
  framing for the online target*. Phase 1 reads the synthesis as
  describing the BYOL discipline loosely and committing to
  `target = target_gru_cell(cat[z_prev, action_emb(a_prev)], h_prev)`
  as the per-call computation. If Phase 3's runner wiring or the
  Phase 6 smoke surfaces a divergence between this interpretation and
  the synthesis's intent — e.g. if the head's loss saturates at high
  value because the target is "too easy" (BYOL-style trivial mapping)
  — the documented escalation is to involve the EMA encoder more
  directly: pass `next_obs` through `target_encoder` to compute a
  fresh embedding feeding a posterior step before the GRU advances.
  This would push the design toward SPR's next-observation-target
  formulation (which is what `target_mode="environmental"` *already*
  does as a control). Held until Phase 3 / Phase 6 surfaces a need.

- *The first-step-of-episode masking convention's interaction with
  the auxiliary loss target*. Phase 0 settled the runner-side masking
  (scalar = 0.0, masked flag = True on the first step of each
  episode); Phase 1 doesn't see the masking — it lives at the runner
  level, applied after `compute_self_prediction_target` returns its
  shape-`(B, h_dim)` tensor. The concrete question: when the runner
  computes `target_h_next` for a first-step-of-episode and then masks
  the *scalar* downstream, the *batched* training-step target (for the
  auxiliary backward) still includes the first-step's target value.
  Should the batched training step also exclude first-step targets? At
  Phase 1 I leave this open; Phase 3's runner integration is the place
  to settle it. The plan §2.4 spec is silent on the batched-side
  masking of the auxiliary target; the synthesis §3 (v2) "first step
  masking" sub-question covers only the actor-readable scalar. My
  guess is the batched auxiliary loss includes the first-step target
  unmasked (the head's training shouldn't be hampered by mask-the-scalar
  being a downstream concern), but Phase 3 should journal whichever
  way it goes and why.

- *The exact relationship between `self_prediction_loss`'s gradient
  and the encoder's gradient at the per-batch-element level*. The
  current implementation uses `torch.no_grad()` inside
  `compute_self_prediction_target` so the EMA target's forward is
  detached. The head's loss flows backward through
  `step.self_prediction = self.self_prediction_head(h)` into `h`,
  which is the result of the online recurrence — gradient continues
  into the GRU and from the GRU into the action embedding (via the
  recurrence's `cat[z_prev, action_emb(a_prev)]` input) and into the
  posterior + encoder via `z`'s rsample. Gate test #1's
  `not target.requires_grad` assertion pins the target side; the
  trainable side flows through online params. The synthesis says
  "auxiliary loss flowing into encoder, GRU, posterior network, prior
  network, and head — but not into the actor's parameters." The
  *prior* network's gradient path: the loss flows into `h_prev` if
  it requires grad — but `h_prev` at training time is taken from the
  replay buffer's stored detached value, so the chain typically ends
  at `h_prev`. Phase 3's runner integration should confirm
  empirically that the gradient actually does flow into the prior
  network as the synthesis claims (the prior reads `h_t` and produces
  prior parameters; the prior's gradient comes only from the KL
  term in `total`, not from `self_prediction_loss`). Held until
  Phase 3 wires the actual training path and we can measure where
  the gradients land.

- *Whether the `loss()` method's contract should split the keys into
  "always present" and "self-prediction-mode present"*. Currently
  `self_prediction_loss` is always returned (zero when no target).
  An alternative would be: return four keys when no `target_h_next`
  is passed, five when one is. The current shape is simpler for
  downstream code (no conditional dict access); the alternative is
  more "honest" about what the call did. I picked the simpler shape
  because the runner — the only production caller — will always pass
  `target_h_next`, and the test paths that don't pass it should still
  see a uniform interface. Reconsider if a future probe surfaces a
  caller for which the difference matters.

---

## Phase 2

*PolicyView + TelemetryView extension + Actor extension · 2026-05-06*

The opacity-boundary pieces the synthesis §1.3 (v2) specifies are now
in code. `kind/agents/views.py` extends `PolicyView` with
`self_prediction_error: Tensor` (the single Watts-heuristic exception
the synthesis §2(b) (v2) authorizes — the minimum self-pointing
quantity Io reads to deliver the second success criterion's "capacity
to take its own processing as an object of attention" affordance) and
extends `TelemetryView` with `self_prediction: Tensor` (the full
`ĥ_{t+1}` vector the mirror reads but Io does not) and
`self_prediction_error_masked: bool` (the first-step-of-episode flag
that lets the mirror exclude masked steps from behavioral-conditioning
analysis). `views.split` extends to keyword-only kwargs
(`self_prediction_error`, `self_prediction_error_masked`) per plan
§2.3's verbatim spec; positional smuggling raises `TypeError`.

`kind/agents/actor.py` extends the input layer to
`input_dim = h_dim + z_dim + 1`. The new column (corresponding to the
scalar input) is zero-initialized at construction (plan §6 row 15);
the existing columns retain their PyTorch Linear default
(Kaiming-uniform). `Actor.forward` and `Actor.act_greedy` concatenate
`(view.h, view.z, scalar_col)` before projection; `_scalar_to_column`
normalizes the scalar to shape `(B, 1)` whether it arrives as `()`,
`(B,)`, `(1,)`, or `(B, 1)`. `Actor.imagine_and_compute_loss` feeds
zero for the scalar at every imagined step (mask-via-zero-feed per
plan §2.3); the gradient through the new column from this path is
zero, so imagined-trajectory training does not exercise
scalar-conditioning — the synthesis §1.7(a) failure-mode (a) detection
is what tests whether this is enough for the actor to develop
conditioning via env-step training paths once Phase 3 wires the
runner.

`kind/training/runner.py`'s two existing PolicyView/split call sites
(`_step_once` and the dream-rollout method) are updated with
**placeholder zero scalars**. Phase 3 will replace `_step_once`'s
placeholder with the real per-step scalar (computed via the loss form
on `wm_step.self_prediction` and the EMA target's `bar{h}_{t+1}`,
with first-step masking). The dream-rollout placeholder is structurally
permanent at Probe 1.5 — synthesis §1.5 commits dreams to running
without the head — and need not move when Phase 3 lands; it stays as
zero through Probe 1.5 and Probe 2, and the reserved
`sequence_self_prediction` field on `DreamRollout` is what carries
forward to Probe 3's design space.

14 new tests + 7 new gate-summary entries: in `tests/test_views.py`
the gate test #4 quartet (`test_policy_view_field_set_is_exactly_h_z_self_prediction_error`,
`test_actor_forward_rejects_telemetry_view_at_runtime`,
`test_actor_forward_telemetryview_argument_fails_mypy_strict`,
`test_ast_lint_passes_with_extended_telemetry_view`) plus
`test_split_kwargs_populate_both_views_correctly` and
`test_split_requires_keyword_only_self_prediction_args`; in
`tests/test_actor.py` the input-layer-extension and
scalar-consumption checks (`test_actor_input_layer_has_extended_input_dim`,
`test_actor_new_input_column_is_zero_initialized_at_construction`,
`test_actor_forward_with_zero_scalar_matches_zero_column_contribution`,
`test_actor_forward_consumes_scalar_after_new_column_is_perturbed`,
`test_actor_forward_accepts_scalar_shapes`,
`test_actor_forward_rejects_malformed_scalar_shape`,
`test_actor_imagine_does_not_grad_the_new_column`,
`test_actor_forward_grad_flows_through_new_column`); plus a deliberately
type-incorrect fixture at `tests/_actor_telemetryview_attempt.py` that
the mypy subprocess test invokes. 364/365 tests pass (1 pre-existing
skip on `test_mirror_caller`); mypy `--strict` clean across the same 22
source files.

### Resolving the §7.4 example test's internal contradiction

The plan §7.4 sample code for `test_actor_forward_rejects_telemetry_view_at_runtime`
contains an internal inconsistency: the docstring says "the actor's
forward only reads `.h`, `.z`, and `.self_prediction_error`" and asserts
`torch.equal(canonical_output.logits, smuggled_output.logits)`. But the
synthesis §6 (and plan §2.3 verbatim) lists `self_prediction` (vector)
and `self_prediction_error_masked` (bool) as the only TelemetryView
additions — *not* `self_prediction_error` (the scalar). With the v2
field set, TelemetryView lacks `self_prediction_error`, so smuggling
a `TelemetryView` past the type-checker into `Actor.forward` raises
`AttributeError` at the first attribute access.

The plan §4 gate test #4 (b) framing resolves the contradiction
correctly: "`Actor.forward(telemetry_view)` raises a runtime error
(constructed by passing a `TelemetryView` into a function expecting
`PolicyView`)." The §7.4 example code is buggy in the equal-logits
direction; the §4 framing is the canonical statement. I wrote the
test to assert `pytest.raises(AttributeError, match="self_prediction_error")`,
which matches both the §4 framing and the actual runtime behavior.
The structural-correctness story is preserved without fudging:
TelemetryView lacks the field PolicyView requires; the runtime path
fails cleanly rather than silently fallback to a degraded behavior;
the asymmetry between Io's reading (the scalar) and the mirror's
reading (the vector + masked flag + everything else) is preserved
*via the field-set difference itself*.

### The runner's transitional placeholder zeros

Phase 2 makes `split`'s new kwargs required (no defaults), per plan
§2.3 verbatim. Phase 3 wires the runner's per-step scalar computation
against the EMA target. Between the two, the runner's two existing
call sites (`runner.py` `_step_once` and the dream-rollout method)
must pass *something* into `split` and into the inline `PolicyView(...)`
construction. Three options surfaced:

1. *Add defaults to `split`'s new kwargs (e.g., default
   `self_prediction_error=torch.zeros(())` and
   `self_prediction_error_masked=True`)*. Lets the runner's old call
   site work unchanged. Hides the transition: the runner stamps
   "Probe 1.5 records" by default but its scalar value is meaningless
   — the kind of "implicit substitution" the Phase 0 journal's writer
   discipline section warned against.

2. *Update the runner to pass placeholder zeros explicitly with
   comments*. The runner's hot-loop and dream-rollout paths get
   explicit `torch.zeros((), device=self._device)` and `True` arguments,
   with a comment block citing plan §2.4 and noting that Phase 3 will
   replace the env-step path's placeholders with real computation.
   Symmetric with the Phase 0 `PROBE_1_SCHEMA_VERSION` discipline:
   the explicit constant marks "this writer is at the transitional
   state, not yet migrated."

3. *Leave the runner broken until Phase 3*. Tests fail; the codebase
   is in a non-buildable state between phases. Violates the per-phase
   "tests must pass before a phase is declared done" discipline.

Picked option 2. The runner's two new call sites carry comments
naming plan §2.4 step 3 and step 4 as Phase 3's wiring target. The
dream-rollout placeholder is structurally permanent (synthesis §1.5
commits dreams to running without the head; the placeholder stays as
zero through Probe 1.5 and Probe 2); the `_step_once` placeholder is
explicitly transitional. With the new column zero-initialized AND the
placeholder scalar at zero, the runner's actor consumption is a
structural no-op at this transitional state — actor's logits depend
only on `(h, z)` through the existing columns, byte-equivalent to a
hypothetical Probe-1-shaped actor on the same `(h, z)` *except* for
the Kaiming-uniform init draws on the existing columns differing
slightly because `fan_in` shifted from `h_dim+z_dim` to
`h_dim+z_dim+1`. No determinism contract pinned actor output values
from a seed (Phase 1 journal entry confirmed), so this is acceptable.

### The scalar shape: () vs (B,) vs (1,) vs (B, 1)

Plan §2.3 specifies the scalar as shape `()` (env-step path, `B=1`)
or `(B,)` (batched paths). The actor's forward must handle both
uniformly. The runner's hot loop uses `B=1` (one env-step at a time);
batched paths (the eyeball helper's behavioral-conditioning module
in §2.7, the counterfactual probe in §2.9) use `(B,)`. Then there's
the `(1,)` case — the runner could pass `torch.tensor([0.0])`
instead of `torch.tensor(0.0)`, and the test paths sometimes do
exactly this.

I added a `_scalar_to_column` helper inside `actor.py` that handles
all four cases (`()`, `(B,)`, `(1,)` with `B>1` broadcasting, and
`(B, 1)` already-correct), normalizing to shape `(B, 1)` before the
concat. Any other shape raises `ValueError("self_prediction_error
must be shape () or (B,) or (B, 1); got shape ...")`. The runner
is what knows the right shape; a malformed scalar is a runner bug,
not an actor fallback case. `test_actor_forward_accepts_scalar_shapes`
pins the four accepted shapes; `test_actor_forward_rejects_malformed_scalar_shape`
pins the rejection.

### Gate-summary extension

Phase 1's gate tests (`test_gate_self_prediction_forward_shape`,
`test_gate_ema_target_update_mechanics`,
`test_gate_self_prediction_loss_decreases`) were not registered in
`tests/test_gate_summary.py` at the time. Phase 2 picks them up
along with gate test #4's four sub-checks (4a: PolicyView field set;
4b: runtime rejection; 4c: mypy --strict rejection; 4d: AST-lint
preserved). Gate test #5 (integration smoke) is held until Phase 6
builds `tests/test_integration_smoke_probe1_5.py`. The gate-summary
file's docstring is updated to name both Probe 1's five and Probe
1.5's five gates. Net: 7 new entries in `_GATE_TESTS`; the parametrized
`test_gate_test_exists` now has 17 cases.

The Phase 1 omission was honest neglect, not an intentional decision
— Phase 1 added the gates themselves but missed the gate-summary
update. The rule going forward: each gate landing comes with its
gate-summary registration in the same phase. The journal-discipline
hook is "if a phase named gate test #N as its deliverable, the gate-
summary must list it before the phase is declared done." Phase 2
catches up; future phases keep up.

### Frozen-ness and the new field's tensor-mutability

`PolicyView.self_prediction_error` is a `Tensor`. The `@dataclass(frozen=True)`
decorator catches reassignment to the field
(`policy.self_prediction_error = torch.tensor(99.0)` raises
`FrozenInstanceError`); it does not catch in-place mutation
(`policy.self_prediction_error.add_(0.1)` succeeds). Same situation
as `h` and `z`. The synthesis §Q5 "structural-by-default, not
adversarial" framing covers this — the boundary is opacity-by-design
in a co-design context, not a security boundary. Pinned in
`test_policy_view_is_frozen_against_field_reassignment` (covers
reassignment); in-place mutation is left as the same shape of gap as
for `h` and `z`, no Probe 1.5-specific addition.

### What's now closed

- The synthesis §1.3 (v2) PolicyView field set: exactly
  `{h, z, self_prediction_error}`, pinned by
  `test_policy_view_field_set_is_exactly_h_z_self_prediction_error`.
  Any future field addition fails this test; the §2(b) (v2) four-part
  discipline applies to any extension.
- The synthesis §1.3 (v2) TelemetryView extension: the new fields are
  `self_prediction: Tensor` (the vector the mirror reads but Io does
  not) and `self_prediction_error_masked: bool` (the first-step flag);
  the scalar `self_prediction_error` is *not* on TelemetryView (it
  lives on PolicyView only; the mirror reads it from the AgentStep
  record the runner emits, not from TelemetryView).
- Plan §6 row 15 default — zero-init for the new actor column —
  pinned by `test_actor_new_input_column_is_zero_initialized_at_construction`.
  The conservative starting point; the synthesis §1.7(a) failure-mode
  (a) detection is what tests whether the column moves enough under
  training to register as conditioning.
- Plan §6 row 19 default — raw concatenation of the scalar with `(h, z)`
  (no learned projection). Matches the architecture-decision discipline
  of "concat, not learned projection" from Probe 1 §Q5.
- Plan §2.3 mask-via-zero-feed during imagination: imagined scalar = 0,
  gradient on the new column from imagined-trajectory backward = 0.
  Pinned by `test_actor_imagine_does_not_grad_the_new_column`. Phase
  3's env-step training path is the only path that moves the new
  column; the synthesis §1.7(a) detection tests whether this is
  enough.
- Plan §7 Level 2 (type-level) opacity boundary preserved with the v2
  field set: `Actor.forward(TelemetryView)` fails mypy `--strict`.
  Pinned by `test_actor_forward_telemetryview_argument_fails_mypy_strict`
  via subprocess; the fixture at `tests/_actor_telemetryview_attempt.py`
  is the deliberately type-incorrect call site mypy reports against.
- Plan §7 Level 1 (import-level) opacity boundary preserved: the AST-
  lint test continues to pass; the actor module imports `PolicyView`
  only and never `TelemetryView`. Pinned by the existing
  `test_actor_module_does_not_import_telemetry_view` (Probe 1's) plus
  the new `test_ast_lint_passes_with_extended_telemetry_view`
  (defensive against a future refactor that imports TelemetryView for
  the new fields).
- Plan §4 gate test #4 (b) runtime rejection: re-interpreted as
  `AttributeError("self_prediction_error")` because TelemetryView
  lacks the field PolicyView requires. Pinned by
  `test_actor_forward_rejects_telemetry_view_at_runtime`.
- The `views.split` signature: `(step, intrinsic, *,
  self_prediction_error, self_prediction_error_masked)`; positional
  smuggling raises `TypeError`. Pinned by
  `test_split_requires_keyword_only_self_prediction_args` and exercised
  end-to-end by `test_split_kwargs_populate_both_views_correctly`.
- The runner's transitional state: `_step_once` and the dream-rollout
  method pass placeholder zero scalars; Phase 3 will replace
  `_step_once`'s placeholder; the dream-rollout placeholder stays
  through Probe 1.5 and Probe 2.
- The gate-summary file now registers all Probe 1.5 gates 1-4
  (Phase 1's three plus Phase 2's four sub-checks); 17 parametrized
  cases total. Gate 5 (integration smoke) deferred to Phase 6.
- 364 tests pass (the pre-Phase-2 343 plus 14 new function-tests plus
  7 new gate-summary entries); the existing integration smoke passes
  unchanged (the runner's transitional placeholders produce
  byte-equivalent actor logits to what the actor would have produced
  on `(h, z)` alone modulo the Kaiming-uniform init shift); mypy
  `--strict` clean.

### What's now newly open

- *Phase 3*: runner integration. `_step_once` needs to compute the
  real per-step scalar via the configured loss form on
  `(wm_step.self_prediction, target_h_next)`, apply first-step-of-
  episode masking (scalar=0.0, masked_flag=True on the first step of
  each episode; subsequent steps carry the computed value with
  masked_flag=False), and pass the real values into `split` and into
  the actor's forward. The runner's checkpoint-save / checkpoint-load
  paths also need extension to carry EMA target weights and the
  extended actor input layer (plan §2.4 / §2.5). The runner's existing
  AgentStep emit path needs the three new schema fields populated;
  the writer's `schema_version` reference moves from
  `PROBE_1_SCHEMA_VERSION` to `SCHEMA_VERSION` for the AgentStep /
  DreamRollout / checkpoint-metadata stamps (the Phase 0 journal's
  writer-migration discipline). The dream-rollout's
  `sequence_self_prediction` stays `None` per synthesis §1.5; only
  the AgentStep-side fields move.

- *The Watts-heuristic exception's discipline as code*. The synthesis
  §2(b) (v2) names a four-part discipline (which affordance, minimum
  form, alternatives considered, failure-mode controls) for any future
  probe that adds a new actor-readable interface. Phase 2 makes part
  of this discipline structural — the
  `test_policy_view_field_set_is_exactly_h_z_self_prediction_error`
  assertion fails if the field set drifts. But the assertion only
  catches *that* a field was added; it can't ensure (i)-(iv) are
  journaled at design time. The discipline relies on future authors
  (a) updating the assertion when adding a field, and (b) being
  prompted by the assertion's failure to engage with (i)-(iv) at
  design time. The hook is fragile in a way the Phase 0
  `PROBE_1_SCHEMA_VERSION` constant pattern is not — there's no
  named-constant equivalent that forces the discipline at write-time
  for new fields. Held open as a Probe 2 / Probe 3 design-notes
  question: should `Kind_design_notes.md` be updated to make the
  four-part discipline a hard prerequisite for any PolicyView
  extension, with a reviewer-checklist hook in the implementation
  plan structure? Synthesis §5 / §10 of the Probe 1.5 plan flags this
  for the Probe 2 plan revision (§10(8) item 8, "before any future
  probe adds a new actor-readable interface, the journal entry must
  address (i)-(iv)"); making it durable beyond a journal note is
  the open question.

- *Phase 6's smoke for the Phase 2 plumbing*. Plan §5 specifies the
  smoke at production sizes, including a check on the actor's
  new-input-column gradient norm across 100 steps. At Phase 2 the
  smoke isn't run (Phase 6's territory); the structural gradient-
  flow test (`test_actor_forward_grad_flows_through_new_column`)
  verifies the autograd graph reaches the new column under a
  synthetic loss but does not verify the column moves under real
  training dynamics. The smoke's "soft warning if new-col grad norm
  below 1e-6 across all 100 steps" is the production-scale signal;
  Phase 6 confirms.

- *Whether the dream-rollout's permanent placeholder zero needs an
  explicit invariant test*. The dream-rollout's PolicyView always
  carries `self_prediction_error=0` at Probe 1.5 and Probe 2 by
  design (synthesis §1.5). A test that pins this invariant — i.e.,
  that asserts the dream-rollout method passes
  `self_prediction_error=torch.zeros((), ...)` — would catch a
  future Probe 3 refactor that accidentally moves dream-rollouts onto
  the env-step path's scalar computation. The current
  comment-and-discipline approach is what the journal entry pins;
  whether to add a structural test now is held as a Probe 3 question
  (Probe 3's design will engage with the
  `sequence_self_prediction`-on-DreamRollout question; that's the
  natural place to settle the invariant test).

- *The `_scalar_to_column` helper's location*. It lives as a
  module-level function in `kind/agents/actor.py`. An alternative
  would be: make it a method on `PolicyView` itself (e.g.,
  `view.scalar_as_column(batch_size)`). The current shape preserves
  the synthesis §Q5 / plan §7 convention that the view stays a pure
  data container — no methods, no derivations — and pushes the
  shape-normalization into the consumer (the actor). The alternative
  would put data-shaping logic on the view, which is what the
  convention rules out. Held closed at the convention level; reopened
  if a future probe surfaces a consumer of `PolicyView` that needs
  the same shape normalization (in which case the helper either gets
  imported by both consumers, or moves to a shared location like
  `kind/agents/views.py` — but staying off PolicyView itself).

- *Whether the actor's `imagine_and_compute_loss` should eventually
  feed a non-zero scalar during imagination*. Plan §2.3 holds this
  for "possible Probe 1.6 if §1.7(a) shows it's needed"; the
  synthesis §1.7(a) is the trigger. The shape this would take: at
  each imagined step `τ`, run the head's `forward(h_τ)` to produce
  `ĥ_{τ+1}`, run the EMA target through some imagined `(h_τ, z_τ,
  a_τ)` continuation to produce `bar{h}_{τ+1}`, compute the loss-form
  scalar, feed it to the actor's policy for step `τ+1`. This pushes
  toward Gemini's original Probe 1.5 framing of "self-prediction
  active during dream rollouts" (which the synthesis explicitly
  defers to Probe 3 for *dream* rollouts; whether *imagined* rollouts
  during the actor's training can carry a non-zero scalar is a
  separate question from dreaming, and is what Probe 1.6 would
  engage). Held open through Probes 1.5, 2, 3 — only triggered if
  failure-mode (a) shows the scalar-conditioning pathway is
  structurally inert under env-step-only training.

---

## Phase 3

*runner integration + checkpoint compat · 2026-05-06*

The substrate's per-step machinery the schema (Phase 0) reserved slots
for and the world model (Phase 1) and views/actor (Phase 2) built now
flow through the runner end-to-end. `kind/training/runner.py`'s hot
loop computes the per-step self-prediction error scalar against the
EMA target via the configured loss form, applies the
first-step-of-episode masking convention (scalar=0.0,
masked_flag=True at `step_in_episode==0`; real values otherwise),
threads both into `views.split` and onward into the actor's forward,
emits `agent_step` records at `SCHEMA_VERSION` (`"0.2.0"`) with the
three new fields populated on every step, sums `wm_total + λ_self *
self_prediction_loss` for the world-model backward in `_train_step`,
calls `WorldModel._update_ema_target()` after the world-model
optimizer step, migrates the dream-rollout emission to `"0.2.0"` with
`sequence_self_prediction=None` (synthesis §1.5; Probe 3's
forward-compatibility hook), writes `"0.2.0"` to
`schema_version.txt` in checkpoint commits, and handles a Probe 1
(`"0.1.0"`) checkpoint on the load side by initialising the EMA
target as a fresh copy of the freshly-loaded online network,
expanding the actor's first-layer weight to `h_dim+z_dim+1` columns
with the trailing column zero-initialized, re-initializing
`_frozen_projection` from a fresh `torch.nn.init.orthogonal_` if the
configured target_mode is `"frozen"`, skipping optimizer-state load
(the saved Adam state was keyed against the Probe 1 parameter set;
the optimizers stay fresh in the resumed process), and emitting a
`world_event` with `event_type="mirror_marker", source="system"` and
the documented payload describing the asymmetry. `RunnerConfig` gains
the four fields the synthesis §6 / plan §6 settled defaults named
(`lambda_self=0.1`, `ema_decay=0.99`,
`self_prediction_target_mode="online"`,
`self_prediction_loss_form="cosine"`) and is now the single source of
truth for the four; the runner's `__init__` calls
`dataclasses.replace` on the user-supplied `world_model_config` so
RunnerConfig wins for the three shared fields without forcing the
user to keep both in sync.

9 new function-tests in
`tests/test_integration_smoke_probe1_5.py` (gate test #5 split into
six sub-checks plus a Probe 1 → 0.2.0 compat regression plus a
prior-network gradient empirical pin); 8 new gate-summary entries.
381/382 tests pass (1 pre-existing skip on `test_mirror_caller`);
mypy `--strict` clean across the same 22 source files.

### Resolving the batched-target masking decision

Phase 1's journal flagged this as open: when computing the auxiliary
loss target inside `_train_step` over a sampled L-step sequence,
should the first position (`t==0` of the sampled window — *not*
necessarily a true episode boundary, but always position-zero of the
window) be masked out of the auxiliary backward, or should it be
included unmasked?

Phase 1's guess: include unmasked. Phase 3 confirms and journals.

Reasoning:

- The first-step-of-episode masking convention (synthesis §3 v2 / plan
  §6 row 16) covers only the *actor-readable scalar* on PolicyView.
  The mask exists because the scalar's value at the first step of an
  episode is sentinel zero (no empirical reading available, since the
  EMA target uses the previous step's `h_prev` which is an episode-
  zero zero state); downstream analyzers must discriminate sentinels
  from real near-zero readings.
- The *batched auxiliary loss* is a different question: its target
  is the same EMA-target arithmetic, but the head's training is
  unaffected by whether a particular position is "first of episode".
  The head learns a stable representation regardless; masking the
  first position out would shrink the effective batch and bias the
  gradient toward later-window positions for no principled reason.
- The synthesis §3 (v2) is explicit only about the actor-readable
  scalar; it is silent on the batched-target masking. The plan §2.4
  step 4 names the per-step scalar masking exactly; the batched
  auxiliary loss is §2.4 step 1–3 with no mask qualification.
- The runner's batched implementation in `_train_step` does not look
  up `step_in_episode` for sampled positions. Position zero of the
  window is not in general an episode-zero step (the buffer's
  windowing is episode-aligned but the position-zero of the window
  could be any `step_in_episode` in the episode — Probe 1's replay
  buffer settled this).

Decision: the batched auxiliary loss includes ALL positions
unmasked. The synthesis §3 (v2) silence on this is read as
permission to use the simpler implementation; the discipline is
that the actor-readable PolicyView scalar is masked exactly when
`step_in_episode==0` at env-step time, and the `agent_step` record
discriminates via the `self_prediction_error_masked_t` flag the
mirror reads. Pinned by the smoke's
`test_smoke_probe1_5_first_step_of_episode_is_masked` (env-step
side) and by the Phase 1 gate test #3 (batched side — runs 100
training steps without distinguishing first-position from later
positions; the loss decreases monotonically, confirming the
unmasked-batched convention is stable).

### Resolving the prior-network gradient confirmation

Phase 1's journal flagged this as open: the synthesis (§1.2 v2)
claims "auxiliary loss flowing into encoder, GRU, posterior network,
prior network, and head". Phase 1's analysis suggested the prior
network's parameters are NOT reached by `self_prediction_loss` alone
(the prior's output `(mu, log_sigma)` is consumed only by the KL
term; the head reads `h_t` directly, never the prior's output).

Phase 3 confirms empirically. Test:
`test_prior_network_gradient_under_self_prediction_loss_alone`. A
2-step batched setup mirrors `_train_step`'s loop so that `z_t`
propagates from `q_dist.rsample()` (which involves encoder +
posterior) at step `t-1` into `recurrence` at step `t`, putting
encoder and posterior on the gradient path. Backprop
`self_prediction_loss` alone (KL excluded), check
`p.grad.abs().sum()` for each parameter:

- **Reached** (synthesis correct): `self_prediction_head` (its weights
  directly project `h`); `gru_cell` (computes `h` via recurrence);
  `action_embedding` (action embedding feeds GRU input);
  `encoder`, `posterior_head` (multi-step path: previous step's
  `q_dist.rsample` → `z_prev` → recurrence → `h` → head).
- **Not reached** (synthesis incorrect): `prior_head`. The prior's
  output `(mu_p, log_sigma_p)` is constructed in `WorldModel.step`
  and stored on `WorldModelStep.p_params`, but `self_prediction_loss`
  reads only `step.self_prediction = self_prediction_head(h)`. The
  prior's parameters are entered into the gradient graph only via
  `kl_divergence(q_dist, p_dist)` in `WorldModel.loss`, which is
  weighted into `total` (zeroed in this controlled test).

Decision: the synthesis's prior-network claim is empirically
contradicted in the current implementation; the four other modules
(encoder, GRU, posterior, head) are correctly named. The Probe 1.5
training process therefore shapes the prior network's parameters
**only via the KL term**, never via the auxiliary self-prediction
loss. This is fine — the synthesis's intent (the auxiliary loss
restructures the substrate's representation conduits) is honored
by encoder + GRU + posterior + head reshaping; the prior network
remains the KL-only consumer it was in Probe 1.

The empirical pin is the test
`test_prior_network_gradient_under_self_prediction_loss_alone`;
the journal-side update is this section. No code change is
indicated — the synthesis's specific claim about prior-network
gradient flow is simply wrong about this implementation; the
synthesis's overall claim about substrate reshaping holds via the
correctly-named four modules.

### The Probe 1 → Probe 1.5 checkpoint compat path's optimizer-state decision

Plan §2.4 step 2 (load deltas) names "initialise EMA target as a
fresh copy of the online network" and "zero-initialize the actor's
new input column" but is silent on the Adam optimizer state. PyTorch
Adam's `load_state_dict` requires shape-matching state to the current
parameter set; loading Probe-1-shaped Adam state into the
Probe-1.5-shaped optimizers (which have additional parameters from
`self_prediction_head` + EMA target + extended actor first layer)
raises.

Three options surfaced:

1. *Refuse to load Probe 1 checkpoints.* Conservatively safe but
   blocks any future investigation that wants to start the failure-
   mode controls from a Probe 1 checkpoint (low likelihood but the
   plan §2.4 explicitly says "does not block the failure-mode
   controls from being run from a Probe 1 starting state if a future
   investigation wants to").

2. *Surgically expand the Probe-1-shaped Adam state to match the
   Probe-1.5-shaped parameter set, with zero-init Adam state for the
   new parameters.* Possible but invasive — the optimizer-internal
   tensor shapes (`exp_avg`, `exp_avg_sq`, `step`) are fragile across
   PyTorch versions and across optimizer kinds. A future Probe-1.5
   change to a different optimizer would require updating the
   surgery code in lockstep.

3. *Skip optimizer-state load for Probe 1 checkpoints; the
   optimizers stay fresh in the resumed process.* Loses Adam
   momentum accumulation for the Probe 1 parameters but preserves
   parameter values exactly. Simple, durable across PyTorch versions
   and optimizer kinds, and the asymmetry is recorded in the
   `mirror_marker` `world_event` so the journal entry captures it.

Picked option 3. Rationale: the Probe 1 → Probe 1.5 transition is a
one-time event per investigation (fresh-init-from-Probe-1 is the
documented path; subsequent runs resume from `"0.2.0"` checkpoints
with full Adam state). Losing one run's Adam momentum is a small
cost; the implementation simplicity and forward-compatibility cost
of options 1 / 2 is larger. The `mirror_marker` payload now names
optimizer-state-skipped explicitly: `"loaded Probe 1 checkpoint;
EMA target initialised from online; actor input column
zero-initialized; optimizer state skipped (fresh Adam in Probe 1.5
process)"`.

### RunnerConfig as the authoritative source for the four shared fields

The `WorldModelConfig` carries `ema_decay`,
`self_prediction_target_mode`, `self_prediction_loss_form`, and
(implicitly via head construction) the head's hidden width.
`RunnerConfig` adds three of these (the head's hidden width stays
on `WorldModelConfig`) plus `lambda_self`. Two options surfaced:

1. *RunnerConfig holds shadow copies; user is responsible for keeping
   both in sync*. Surprising — the user can pass inconsistent values
   and the world model uses one set, the runner uses the other.

2. *Runner's `__init__` constructs the WorldModel from a
   `dataclasses.replace`-overridden WorldModelConfig*. RunnerConfig
   is authoritative; the world model is guaranteed to honor whatever
   the runner config specifies. Backward-compatible because both
   configs have identical defaults, so the override is a no-op for
   unmodified configs.

Picked option 2. Synthesis §6 v2 row 1 names `lambda_self=0.1` as
the runner-level default; rows 2/3/5 likewise pin runner-level
defaults; the world model's defaults are intentionally identical so
the override is structurally a no-op for the standard case. The
override is what makes RunnerConfig the single source of truth at
construction; the user can still set values on `WorldModelConfig` if
they prefer (which they did in Phase 1's journal-recorded test
fixture pattern), but if they ALSO set them on RunnerConfig with
different values, RunnerConfig wins. Documented in the runner's
`__init__` block with a citation to plan §2.4.

### Per-step scalar's environmental-mode fallback at sequence boundaries

The runner's `_train_step` loops over an L-step replay sequence. For
environmental mode, the per-position auxiliary target requires
`next_obs`. For positions `0..L-2` the sequence supplies it via
`obs_seq[:, t+1]`; for position `L-1` there is no next obs in the
window. Three options surfaced:

1. *Fall back to using `obs_t` as next_obs for the last position*
   (semantically: at position L-1 the head's prediction is paired
   against the EMA encoder of the *current* obs, which is the
   structurally-closest available stand-in for the missing future
   obs).

2. *Skip the auxiliary loss term at position L-1.* Simpler at the
   cost of biasing the gradient toward earlier positions.

3. *Sample a longer sequence (L+1) and use the last position only as
   the auxiliary target.* Cleanest semantically but invasive —
   requires changing `_replay.sample` to return L+1 transitions and
   discarding the last one's training contribution (only its obs
   matters).

Picked option 1 (fall back to `obs_t`). Rationale: environmental
mode is the failure-mode-control path (Phase 5), not the default
Probe 1.5 path; the simplest stand-in suffices; option 3's
invasiveness is held for Phase 5 if the smoke shows the boundary
position's auxiliary target distorts the head's training. The
default `target_mode="online"` is unaffected by this decision (the
online target uses only `(h_prev, z_prev, a_prev)` from the same
position, no next_obs).

### What's now closed

- The runner's hot-loop integration of self-prediction
  (`_step_once` lines 488–544): EMA target via
  `compute_self_prediction_target`, configured loss form for the
  per-step scalar, first-step-of-episode masking convention,
  `views.split` extension passes through the masked flag and the
  scalar (which the actor reads on PolicyView), AgentStep emission
  populates the three new fields at `"0.2.0"`. Pinned by
  `test_smoke_probe1_5_agent_step_carries_new_fields` and
  `test_smoke_probe1_5_first_step_of_episode_is_masked`.
- The runner's `_train_step` integration: `wm_total + λ_self *
  self_prediction_loss` summed for the world-model backward;
  batched target via `compute_self_prediction_target`; EMA update
  in-place after the world-model optimizer step. Pinned by Phase
  1's gate test #3 (the loss decreases) and by the smoke (the run
  to completion under the default mode).
- The DreamRollout migration to `SCHEMA_VERSION` with
  `sequence_self_prediction=None` (synthesis §1.5; Probe 3's
  forward-compatibility hook). Pinned by
  `test_smoke_probe1_5_dream_rollout_carries_none_self_prediction`.
- The checkpoint commit migration: `schema_version.txt` writes
  `"0.2.0"`; EMA target weights ride inside `weights.safetensors`
  via the existing `world_model.target_*` prefixes; the actor's
  extended input layer rides inside the existing `actor.*` prefix.
  Pinned by
  `test_smoke_probe1_5_checkpoint_carries_ema_and_extended_actor`
  and the resume regression
  `test_smoke_probe1_5_resume_yields_identical_ema_and_actor`.
- The Probe 1 → Probe 1.5 checkpoint compat path: read
  `schema_version.txt` first, branch on Probe 1 vs Probe 1.5; for
  Probe 1, load with `strict=False` then
  `_initialize_ema_target_from_online`, expand the actor's first
  layer with zero-initialized trailing column, skip optimizer-state
  load, emit `mirror_marker`. Pinned by
  `test_load_probe_1_checkpoint_initializes_ema_from_online`.
- The batched-target masking decision: include all positions
  unmasked. Synthesis §3 (v2) silence read as permission; the
  actor-readable scalar masking remains the only place
  `step_in_episode==0` is consulted.
- The prior-network gradient empirical confirmation: under
  `self_prediction_loss` alone (KL excluded), the prior's
  parameters receive zero gradient; encoder, GRU, posterior, head,
  action_embedding receive gradient. The synthesis's overall claim
  about substrate reshaping holds via four modules, not five.
  Pinned by
  `test_prior_network_gradient_under_self_prediction_loss_alone`.
- Plan §6 row 1 default `λ_self=0.1` promoted to "settled at this
  scale" — the smoke's per-step training step under the combined
  loss runs without instability for the integration smoke's 200
  steps.
- Plan §6 row 16 default (first-step masking convention: scalar=0.0,
  masked_flag=True at `step_in_episode==0`, visible to the mirror
  via the `self_prediction_error_masked_t` field).
- Plan §6 row 11 (EMA target weights enter checkpoint) — pinned by
  the smoke and the resume regression.
- Plan §6 row 12 (combined backward, single optimizer step on the
  world model) — pinned by the smoke and Phase 1's gate test #3.
- The transitional placeholder zeros in `_step_once` are gone (the
  Phase 2 journal flagged this; Phase 3 wires the real
  computation). The dream-rollout placeholder remains zero by
  design (synthesis §1.5).
- 381 tests pass (the pre-Phase-3 364 plus 9 new function-tests
  plus 8 new gate-summary entries); the Probe 1 integration smoke
  (`tests/test_integration_smoke.py`) continues to pass with one
  test updated (`test_smoke_schema_version_file_committed` now
  expects `"0.2.0"` rather than `"0.1.0"`); mypy `--strict` clean.

### What's now newly open

- *Phase 4*: `kind/observer/digest.py` extension to expose
  per-episode self-prediction error mean / std / outliers
  (excluding masked steps), per-dimension self-prediction allocation
  top-k, and the per-episode masked-step count for `"0.2.0"`
  records; `kind/observer/eyeball.py` extension with the new
  `show_self_prediction(...)` helper and the new
  `show_self_prediction_conditioning(...)` helper (the behavior-side
  analysis surface the counterfactual probe in Phase 8 will
  consume). Phase 3 leaves the digest and eyeball untouched; the
  Phase 4 entry will record what the new lines look like and how
  the masked-step exclusion is implemented.

- *The Phase 5 environmental-mode boundary*. Phase 3's runner uses
  `obs_t` as the fallback `next_obs` at the last position of the
  L-step batched window in environmental mode (and as the
  per-step `next_obs` at env-step time when `target_mode="environmental"`).
  This is the structurally-closest available stand-in for the
  missing future obs. Phase 5 wires the actual environmental-mode
  control runs against Probe 1's no-affordance baseline; if the
  fallback distorts the head's training measurably (visible in
  the smoke's instability checks or in the structural-metric
  comparison against the main run), the documented mitigation is
  to extend the replay buffer's sample to L+1 transitions for
  environmental mode and use the last as the auxiliary target only
  (option 3 from the §"Per-step scalar's environmental-mode fallback
  at sequence boundaries" section above).

- *The `mirror_marker` payload's content discipline*. The Probe 1 →
  Probe 1.5 compat path emits a `world_event` with
  `event_type="mirror_marker", source="system"` and a payload
  carrying the asymmetry note plus checkpoint metadata. The five
  expected phrases the smoke test checks (`"Probe 1 checkpoint"`,
  `"EMA target"`, `"online"`, `"actor input column"`,
  `"zero-initialized"`) pin the load-side payload. Whether future
  `mirror_marker` emissions (Probe 2's mirror-call markers, Probe
  4's builder-perturbation markers) should also follow this content
  discipline — name what was done, what was preserved, what was
  reset, what was lost — is held as a Probe 2 / Probe 4 design
  question. The current pattern is "describe the asymmetry the
  load creates"; the broader question is whether `mirror_marker`
  records form a structured taxonomy of system-level events the
  mirror's faithfulness check can resolve against, or whether they
  remain free-form notes.

- *The optimizer-state-skipped asymmetry's training-side
  consequences*. Probe 1 → Probe 1.5 transitions lose Adam momentum
  accumulation for the Probe 1 parameters; the optimizers start
  fresh in the resumed process. For Probe 1.5's main run this is
  irrelevant (the standard path is fresh-init, not Probe-1-resume).
  But if a future investigation runs the failure-mode controls from
  a Probe 1 checkpoint (the option Phase 3 keeps open), the
  fresh-Adam-state contributes a confounding factor to the
  comparison: any structural metric difference between the control
  run (resumed from Probe 1) and Probe 1's run (which had Adam
  momentum from start to end) could be attributed to the
  fresh-Adam confound rather than to the affordance's
  presence/absence. The journal entry for that future investigation
  will need to address this; Phase 3 does not.

- *The synthesis's prior-network claim drift*. The empirical pin
  in this phase contradicts the synthesis's specific claim about
  gradient flow into the prior network. Should the synthesis text
  be updated to reflect the empirical finding? The synthesis is
  versioned; v2 supersedes v1; updating the v2 text creates a v3
  with a single text change. The cost is a journal entry trail and
  a `Kind_probe1_5_synthesis.v2.md` preservation. The benefit is
  that future readers don't see a contradiction between the
  synthesis text and the test-pinned reality. Held open: the
  synthesis text says "auxiliary loss flowing into encoder, GRU,
  posterior network, prior network, and head"; the test says the
  prior-network's gradient is zero under sp_loss alone. Either the
  synthesis text is read as a synthesis of literature claims (in
  which case the implementation is a specific reading the journal
  records) or it is read as a prediction the implementation should
  honor (in which case the implementation is wrong — but it is the
  implementation the synthesis §1.2 v2 architecturally specifies,
  so this reading would be self-contradictory). The pragmatic
  reading: synthesis text describes the literature's framing;
  this journal entry plus the test's name pin the
  implementation's actual behavior. No synthesis revision needed.
  Reconsider if Probe 2's mirror reads the synthesis text and
  cites a prior-network reshape claim that is empirically absent.

- *Phase 6's MPS smoke for the integration-smoke build*. The CPU
  smoke at tiny sizes (h=32, z=4, K=2, head_hidden=32) runs in
  ~10 seconds and exercises every conduit; Phase 6's MPS smoke at
  production sizes (h=200, z=16, K=5, head_hidden=200) is the
  platform-correctness gate. Phase 3 leaves it untouched; Phase 6
  will build `scripts/smoke_probe1_5.py` and the sanity test in
  `tests/test_smoke_probe1_5_script.py`. The instability checks
  the plan §5.1–5.5 specify (KL pinning, recon climbing, EMA
  target divergence, gradient norm, actor new-col gradient norm)
  are Phase 6's territory.

---

## Phase 4

*digest + eyeball extension · 2026-05-06*

The read-side analyzer surface for the substrate Phases 0-3 produced.
`kind/observer/digest.py::build_digest` now emits four extra lines per
episode for `"0.2.0"` records — `self_prediction_error_t` mean / std /
min / max excluding masked steps, the per-episode masked-step count,
outliers (non-masked steps whose self-prediction error z-score against
the episode's non-masked mean exceeds 3, mirroring the existing KL
outlier pattern), and per-dimension allocation top-5 (the five `h`
dimensions whose self-prediction-error variance across the episode is
highest, with masked steps excluded). `compact_record_repr` adds
`self_prediction_error_t` and `self_prediction_error_masked_t` to its
scalar fields tuple; the `self_prediction_t` vector stays out per the
module's existing scalar-only discipline. Probe 1 (`"0.1.0"`) records
produce no self-prediction summary lines — the no-affordance baseline
stays visibly distinct from a Probe 1.5 record where the head produced
a near-zero value (plan §3.3 "skip" backward-compat approach).

`kind/observer/eyeball.py::show_episode_summary` picks up the same
self-prediction summary block the digest gained — mean/std/min/max
excluding masked, plus the per-episode masked-step count — for
`"0.2.0"` records; Probe 1 records render unchanged. Two new helpers
land. `show_self_prediction(telemetry_dir, *, episode_id=None,
top_k_dims=10)` prints the per-dimension self-prediction-error
allocation across `h_dim` for one episode, ranked by per-dim residual
variance. `show_self_prediction_conditioning(run_dir, *,
checkpoint_id=None, n_states=200, perturbation_distributions=None,
regimes=None, perturbation_window_w=25, seed=0)` prints the
behavior-side per-regime KL table the counterfactual probe (§2.9 /
Phase 8's territory) will consume — load the run's checkpoint, sample
non-masked states, perturb the scalar via Gaussian-with-empirical-
sigma / zero-out / replace-with-uniform-from-empirical-range,
classify each state by regime
(`perturbation_window`, `high_disagreement`, `high_kl`,
`steady_state`), and compute KL between unperturbed and perturbed
policy distributions per regime. Both helpers print the documented
no-self-prediction-telemetry / no-scalar-to-perturb message against a
Probe 1 fixture and return without crashing. The CLI gains `selfpred`
and `cond` subcommands; the existing `episode` subcommand picks up
the extended `show_episode_summary`. `__all__` widens to seven names.

22 new tests in `tests/test_phase4_digest_eyeball.py` covering each
of the eight specific cases the plan §2.6 / §2.7 / build prompt
names plus the CLI subcommand checks plus
`show_episode_summary`'s extension plus three sanity cases (no
records / no checkpoints / no agent_step records). One pre-existing
test in `tests/test_eyeball.py::test_eyeball_module_exposes_named_helpers`
was updated to expect the seven `__all__` names instead of five — the
two new helpers are the documented additions. 402 / 404 tests pass
across `tests/` (1 pre-existing skip on `test_mirror_caller`; 1
flaky pre-existing test in `test_transport.py` that passes in
isolation but order-fails in the full suite — unrelated to Phase 4
and noted as a runtime concurrency artefact); mypy `--strict` clean
across 22 source files.

### Resolving the "what target does the per-dim residual use" question

The plan §2.6 spec for the digest's per-dim allocation reads "top 5
dims `whose self-prediction-error variance across the episode is
highest, excluding masked steps`". Synthesis §1.4 element 3 names the
same shape: per-dimension self-prediction *accuracy* allocation. The
runner's training-time target is `bar{h}_{t+1}` — the EMA-tracked
GRU's deterministic state at t+1 — but the `agent_step` schema
exposes `self_prediction_t = ĥ_{t+1}` (the head's output at step t)
and `h_t` (the *online* deterministic state at step t) only. The EMA
target's per-step value never appears on TelemetryView and never lands
in the parquet shards.

Three options surfaced for "which target the analyzer pairs with
ĥ_{t+1}":

1. *Per-dim variance of `self_prediction_t[d]` itself*. Trivially
   computable from telemetry, but it's "where the head allocates
   prediction capacity" rather than "self-prediction *error*
   allocation" — the synthesis names the latter, and they're
   different signals at training-time.

2. *Per-dim residual `self_prediction_t[d] - h_{(t+1)}[d]` using the
   *next* `agent_step` row's `h_t` as the proxy for the EMA target*.
   The online `h_{t+1}` is the closest available stand-in for
   `bar{h}_{t+1}` — the EMA target tracks the online network with
   decay 0.99, so the divergence between online and EMA at any step
   is bounded by `(1 - 0.99**K) * grad_magnitude`, much smaller than
   the per-dim residual signal the analyzer is trying to surface.
   Available from telemetry alone (no need to re-instantiate the EMA
   target).

3. *Add `self_prediction_target_t` to the AgentStep schema as a
   fourth optional `0.2.0` field* — the EMA target's per-step value
   stamped at write time. The cleanest semantic match to the plan's
   spec, but bumps the schema (the storage cost is roughly the same
   as `self_prediction_t` itself: `h_dim` floats per step) and would
   require re-running Probe 1.5's runner to produce a fixture that
   can drive the analyzer. The Phase 0 `"0.2.0"` schema has no slot
   for it; a Phase 0 amendment would push the schema to `"0.3.0"`,
   which the plan §6 row 8 explicitly forecloses on for Probe 1.5
   ("any further bump within Probe 1.5 is a hard reset to 0.3.0
   with a journal entry").

I picked option 2. Rationale:

- The mirror reads telemetry; what the mirror sees should be
  reproducible from telemetry alone without requiring the analyzer
  to construct an EMA target from checkpoint weights to compute a
  per-step value the runner already knew at write time but didn't
  emit. Option 3 would solve this at the cost of a schema bump
  Phase 0 won't permit and a re-run.
- The online-vs-EMA divergence at any step is bounded by the EMA
  decay rate; the per-dim residual signal the analyzer surfaces is
  dominated by the prediction error itself, not by the
  online-vs-EMA shift. Option 2's substitution is empirically a
  small distortion of the runner's training-time signal.
- Option 1 would conflate "where the head allocates prediction
  capacity" with "where the head fails to predict" — different
  questions. The plan asks for the latter.

The substitution is journaled in the digest module's docstring and
in the helper's docstring at `_per_dim_self_prediction_error_variance`,
so a future reader sees the "next-step `h_t` as proxy" choice
explicitly. If Phase 6's MPS smoke or Phase 8's counterfactual probe
surfaces a regime where the proxy distorts the per-dim signal in a
load-bearing way (e.g., per-dim allocation patterns drift between
analyzer and Phase 8 reading), the documented escalation is option 3
plus the schema bump.

### The conditioning helper's checkpoint-introspection without `WorldModelConfig`

`show_self_prediction_conditioning` loads actor weights from
`<run>/checkpoints/<id>/weights.safetensors` and constructs an
`Actor` to perturb the scalar input against. The Actor's constructor
takes `h_dim`, `z_dim`, `mlp_hidden`, `action_dim` — none of which
are recorded in the safetensors blob. Three options surfaced for
where the helper learns the shapes:

1. *Read `RunnerConfig` / `WorldModelConfig` from a sidecar JSON the
   runner writes at checkpoint commit time*. Cleanest semantically;
   would require Phase 3 to write such a sidecar. Phase 3's commit
   contents (plan §2.5 / `CheckpointContents` in
   `kind/training/checkpoint.py`) don't include a config-blob, and
   adding one is invasive at the Phase 4 surface.

2. *Infer all four shapes from the safetensors weight tensors*.
   `actor.net.0.weight` has shape `(mlp_hidden, h_dim + z_dim + 1)`;
   `actor.net.4.weight` has shape `(action_dim, mlp_hidden)`. So
   `mlp_hidden`, `action_dim`, and the sum `h_dim + z_dim` are
   recoverable; the *split* between `h_dim` and `z_dim` is not.

3. *Hardcode the WorldModelConfig defaults (h=200, z=16) and let the
   helper fail loudly on non-default-shape checkpoints*. Simplest;
   wouldn't work for tiny test sizes (h=8, z=4 in the test fixtures)
   without per-test plumbing.

I picked option 2 with a documented split heuristic. The Actor's
forward is split-agnostic — `concat([h, z, scalar_col], dim=-1)`
goes through the same first-layer projection regardless of where the
boundary sits — so as long as the helper supplies tensors whose
shapes sum to the saved input dim, the forward is byte-identical to
the live runner's. The split is recovered as: if `h+z >= 216` and
divides evenly under the WorldModelConfig default ratio, use that;
otherwise fall back to `z = max(1, (h+z) // 3)` and `h = (h+z) - z`,
which keeps both ≥ 1 for `h+z ≥ 3`. This is journaled in
`_build_actor_from_checkpoint`'s docstring; a future reader who
introspects `actor.h_dim` post-construction sees the inferred value
rather than the original config's value.

The Actor's forward at the helper is via the helper-private
`_actor_logits` rather than `Actor.forward` — the latter samples a
Categorical and returns an action; for a KL comparison we want the
deterministic logits at the policy distribution, not a stochastic
sample. The helper replicates the concat order
`(h, z, scalar_col)` to stay byte-equivalent to the live forward
and reuses `kind.agents.actor._scalar_to_column` so the shape
normalisation (the `(B, 1)` column) is the same machinery the
runner uses.

### The regime classification's "first match wins" priority order

Plan §2.7 names four default regimes (`perturbation_window`,
`high_disagreement`, `high_kl`, `steady_state`) and the
counterfactual probe's table format expects each sampled state to
land in exactly one bin. With overlapping regimes — e.g. a state in
the perturbation window that is also a top-quartile-disagreement
state — the helper has to pick a primary regime.

I picked first-match-in-`regime_list`-order, with the default order
being `perturbation_window > high_disagreement > high_kl >
steady_state`. Rationale:

- `perturbation_window` is the most diagnostic regime when
  perturbations are present (synthesis §1.4 element 4: "behavioral
  conditioning on the scalar in regimes the framework recognizes as
  reflexive-attention-bearing"); putting it first prevents
  overlap-with-disagreement from masking it.
- The CLI's `--regime` flag reorders by what the user passes; if a
  Phase 8 author wants to test `high_disagreement` priority over
  `perturbation_window`, the flag's repeated use in a different
  order does that without code changes.
- Single-bin classification keeps the per-regime KL_mean
  interpretable as "the actor's conditioning sensitivity in this
  regime"; multi-bin (a state contributes to multiple regimes)
  would inflate bin sizes and make the per-regime KL a mixture
  rather than a clean signal.

Pinned by `test_show_self_prediction_conditioning_synthetic_run_emits_table`
(default order; all four regimes appear) and by
`test_cli_cond_with_subset_of_perturbations_and_regimes` (subset
order; only the named regimes appear).

### The `n_states` clamp and masked-step exclusion

The plan §2.7 build-prompt requirement reads "Masked steps correctly
excluded from `n_states` sample". The implementation: filter
`candidate_rows` to non-masked-and-non-None-scalar 0.2.0 records
*before* sampling, then `sample_size = min(n_states, len(candidate_rows))`.

The clamp matters because a Probe 1.5 run with episode length 200
and 25 episodes produces ~5000 candidate states; `n_states=200`
samples 200 of them. But for the `n_states` discipline test
(`test_show_self_prediction_conditioning_excludes_masked_steps_from_n_states`),
the synthetic fixture deliberately masks every step except one — if
the helper sampled from `rows` without filtering, it would sample
masked steps and try to perturb their sentinel-zero scalar, which
would conflate "no empirical reading" with "empirical near-zero
reading" at the regime classification level. The pre-filter keeps
the conditioning analysis on empirical readings only.

Printed `sampled n_states=N` in the header records what the helper
actually sampled — visible to the human builder + the future
counterfactual probe's report. The synthetic test pins the clamp
behavior (`assert "sampled n_states=1"`) so any future helper
refactor that lifts the filter is caught.

### Pre-existing `test_eyeball.py::test_eyeball_module_exposes_named_helpers`

The existing test asserted `set(eyeball.__all__) == {five-Probe-1-helpers}`.
Phase 4 adds two helpers to `__all__` (`show_self_prediction`,
`show_self_prediction_conditioning`); the test had to be updated to
expect the seven names instead of five. I extended the existing test
in-place rather than removing it and pinning the surface in the new
Phase 4 file alone — the two-test-files-asserting-the-same-thing
pattern (the existing test plus my new
`test_cli_help_lists_new_subcommands`) costs a duplicate assertion
but means a future helper addition fails *both* tests, which is the
discipline hook that prompts the author to update `__all__` along
with adding the helper.

Same shape as the Phase 0 `PROBE_1_SCHEMA_VERSION` discipline: the
named-constant convention + the validator are dual scaffolding for
"writer migration must happen at the writer side". Here the
`__all__` test + the CLI subcommand test are dual scaffolding for
"helper additions must happen at the public surface". Both halves
fail loudly when one half is omitted.

### What's now closed

- The plan §2.6 digest extension: per-episode self-prediction error
  mean/std/min/max excluding masked, masked-step count, outliers
  (z>3 against non-masked mean), per-dim allocation top-5. Pinned by
  `test_build_digest_mixed_version_emits_lines_for_v0_2_0_only` and
  `test_build_digest_self_prediction_outlier_detection`.
- The plan §3.3 "skip" backward-compat approach: 0.1.0 records
  produce no self-prediction lines in the digest or in
  `show_episode_summary`; the no-affordance baseline is visibly
  distinct from a near-zero Probe 1.5 reading. Pinned by
  `test_build_digest_against_probe_1_run_produces_no_self_prediction_lines`,
  `test_show_episode_summary_v0_1_0_excludes_self_prediction_summary`,
  and the same backward-compat assertion in `compact_record_repr`'s
  test.
- The plan §2.7 eyeball extension: extended
  `show_episode_summary` for 0.2.0 records; new
  `show_self_prediction(telemetry_dir, *, episode_id, top_k_dims)`
  helper; new `show_self_prediction_conditioning(run_dir, *,
  checkpoint_id, n_states, perturbation_distributions, regimes,
  perturbation_window_w, seed)` helper; CLI `selfpred` and `cond`
  subcommands. Pinned by 22 tests in
  `tests/test_phase4_digest_eyeball.py`.
- The per-dim residual proxy: next-step `h_t` stands in for the
  runner's training-time `bar{h}_{t+1}`. Documented in the digest
  module's docstring and at the helper's
  `_per_dim_self_prediction_error_variance` site; pinned by
  `test_show_self_prediction_excludes_masked_first_step_from_per_dim_arithmetic`
  (the masked sentinel does not leak into the per-dim variance).
- The conditioning helper's checkpoint introspection: shape inference
  from `actor.net.0.weight` and `actor.net.4.weight`, with a
  documented split heuristic for `h_dim`/`z_dim`. Pinned by the
  synthetic-run end-to-end tests (`h=8, z=4` test fixtures) where
  the helper successfully constructs an Actor without
  `WorldModelConfig`.
- The regime classification's first-match-wins priority order
  (`perturbation_window > high_disagreement > high_kl >
  steady_state`) and the CLI's `--regime` reorder. Pinned by the
  table-emission test and the subset-CLI test.
- The `n_states` clamp + masked-step pre-filter: masked steps are
  excluded from `candidate_rows` before sampling; `sample_size =
  min(n_states, len(candidate_rows))`. Pinned by
  `test_show_self_prediction_conditioning_excludes_masked_steps_from_n_states`.
- The Probe 1 fixture's no-message paths: both helpers print the
  documented no-telemetry / no-scalar message against
  `runs/probe1-20260503-123926/` and return without crashing.
  Pinned by the two
  `_against_probe_1_*` tests (skipped if the fixture is absent, so
  the suite stays portable).
- The `__all__` widening to seven names; both the existing
  `test_eyeball.py::test_eyeball_module_exposes_named_helpers` and
  the new `test_cli_help_lists_new_subcommands` pin the surface.
- 402/404 tests pass (the pre-Phase-4 381 plus 22 new minus 1
  superseded by the `__all__` extension); the test_transport flake
  is unrelated to Phase 4 and passes in isolation; mypy `--strict`
  clean across 22 source files.

### What's now newly open

- *Phase 5*: failure-mode control variants wiring. `WorldModel.step`
  honours `self_prediction_target_mode`; the
  `frozen` and `environmental` paths are already implemented at
  Phase 1; Phase 5 wires the runner-side scripts for the two
  control runs plus the `mirror_marker` payload. Phase 4 leaves
  `kind/agents/world_model.py` untouched; the substrate-side
  failure-mode controls are Phase 5's territory.

- *The synthesis §1.4 element 3 phrasing — accuracy versus error*.
  The synthesis text says "per-dimension allocation of self-
  prediction *accuracy* across the latent space". The plan §2.6
  says "self-prediction *error* variance". The Phase 4
  implementation lands on error-variance because that's directly
  computable from telemetry under the proxy substitution. Accuracy
  and error are inversely related but not identical signals — high
  variance in residuals could mean either "the head is most
  accurate at this dim" (small residuals) or "the head is most
  variably accurate at this dim" (residuals fluctuate). The current
  reading is the latter; if Phase 8's counterfactual probe or
  Probe 2's mirror surfaces a discrepancy between the framework's
  "accuracy allocation" claim and the implementation's "error
  variance" reading, the documented escalation is to add a parallel
  per-dim *accuracy* line (1 - variance, or
  1/(1+variance), or the inverse-rank — one of several options
  Probe 2's plan revision §10 will sort through). Held open until
  Phase 8 / Probe 2 surfaces a need.

- *Whether the conditioning helper should accept multiple
  checkpoints in one call*. Plan §2.9's counterfactual probe loads
  early/mid/late checkpoints; Phase 4's helper takes one. The probe
  (Phase 8) calls the helper three times and aggregates the
  per-checkpoint tables. An alternative is for the helper to accept
  a list of checkpoint IDs and produce a stacked table. The
  one-checkpoint-per-call surface keeps the helper composable
  (Phase 8 owns the across-checkpoints comparison; the helper owns
  the per-checkpoint inspection); the multi-checkpoint shape would
  push aggregation logic into the helper. Phase 4 picks the
  composable shape; Phase 8 will surface whether the multi-
  checkpoint shape is needed.

- *The CLI's `cond` subcommand against a Probe 1 run with no
  checkpoints*. Phase 4 prints "no checkpoints under ..." and
  returns; the test
  `test_show_self_prediction_conditioning_no_checkpoints` pins this.
  But the synthesis §1.7(c) calibration's lesion-via-noise-injection
  probe (Phase 8) would benefit from being able to run against a
  *Probe 1.5* checkpoint with the scalar zeroed-or-randomized at
  evaluation time. That's the synthesis §10 item 2's
  `"zero_or_randomize_scalar"` lesion — Phase 5's territory.
  Phase 4's helper has the perturbation machinery the lesion would
  reuse; whether to extract a shared evaluation-time perturbation
  module (`kind/observer/conditioning.py` per Probe 2 plan
  revision §10(11)) at Phase 4 or wait for Probe 2's resumption is
  open. I picked: keep the perturbation machinery in
  `kind/observer/eyeball.py` for now (the Probe 1.5 plan does not
  ask for a separate module; Probe 2 plan revision §10(11) does).
  Reconsider when Probe 2 resumes.

- *The `sampled n_states` clamp's interaction with regime imbalance*.
  Suppose a run has 5000 candidate states but only 3 are in the
  `perturbation_window` regime (perturbations are rare under the
  Probe 1.5 default — the env-server emits at most a handful per
  5000 steps). The helper samples uniformly from
  `candidate_rows` then classifies; the perturbation-window bin
  gets ~ 3/5000 of the sample. With `n_states=200` that's ~0.12
  states — i.e., zero almost surely. The KL distribution for the
  regime would then be empty (the helper prints `n_states=0,
  KL_mean=-, KL_std=-, ...`). The plan §2.9 names this case ("the
  perturbation_window regime may have few or no states; in that
  case the comparison reduces to the other three regimes plus a
  flag in the report"). Phase 4's helper handles the empty-regime
  case structurally (prints `-` placeholders); whether to also
  emit an explicit "regime undertested" warning is held open. The
  counterfactual probe (Phase 8) is where the report-level flag
  belongs; the helper's table-emission stays minimal.

- *Whether the helper's `cond` CLI subcommand should accept
  `--alpha` for the Gaussian perturbation*. Plan §2.9 sweeps
  `alpha ∈ {0.5, 1.0, 2.0}` for the counterfactual probe; Phase 4's
  helper uses `alpha=1.0` (the synthesis canonical) only. The
  `_GAUSSIAN_PERTURB_ALPHA` constant is module-level so a Phase 8
  author can override at the script layer; whether to plumb it
  through the CLI is held until Phase 8 surfaces a need (the CLI
  is for one-off inspection; the multi-alpha sweep is the probe's
  job).

- *The `_actor_logits` helper's coupling to the Actor's
  `nn.Sequential` shape*. The helper does `actor.net(x)` directly,
  bypassing `Actor.forward`'s sampling. A future Actor refactor
  that changes the network attribute name (from `net` to something
  else) or the architecture (from `nn.Sequential` to a custom
  module) would break the helper. The current shape is the
  simplest end-to-end byte-equivalent path; an alternative is to
  add a `Actor.policy_distribution(view)` method returning the
  Categorical directly, which the helper could call. Held open at
  Phase 4 (no Actor refactor pending); revisit if Probe 2's
  formalized `kind/observer/conditioning.py` (plan revision
  §10(11)) factors out the per-state policy-distribution machinery.

---

## Phase 5

*failure-mode control variant: frozen-target (lean) · 2026-05-06*

The single Phase 5 deliverable per the lean-revision addendum (plan
§13): `scripts/probe1_5_control_frozen_target.py`, a runnable script
parallel in structure to `scripts/run_probe1.py` whose only
substantive delta is `RunnerConfig.self_prediction_target_mode="frozen"`
(plus `WorldModelConfig.self_prediction_target_mode="frozen"` on the
nested config, since the runner's `__init__` does
`dataclasses.replace` from the runner-level field but the dry-run
summary reads the world-model field directly so both are set in the
helper). The script wires the substrate-side test of failure mode
(a) — inert affordance — per synthesis §1.7(a) v2 / plan §8.1: the
head's supervisory target is a fixed random-orthogonal projection of
the online `h_t` rather than the EMA target's `bar{h}_{t+1}`, so the
head learns a fixed function of its own input rather than the actual
next state. If the run's structural metrics turn out indistinguishable
from Probe 1's no-affordance baseline, the affordance is dead at the
substrate-side reading; if distinguishable, alive.

A `mirror_marker` `world_event` is emitted at run start (after Runner
construction, before `runner.run`) with `source="system"`,
`event_type="mirror_marker"`, and a payload carrying `lesion_kind=
"frozen_target"`, the rationale string, the target_mode, the seed,
the total_env_steps, and the run_id. The emission goes through the
runner's already-open `_world_event_sink` — the same JsonlSink the
runner itself uses for the Probe 1 → Probe 1.5 checkpoint-load
mirror_marker (`runner.py` §"mirror_marker for Probe 1 → Probe 1.5
transitions"); see "The mirror_marker emission path" below.

A `--dry-run` flag short-circuits before any side effect: the script
constructs the `RunnerConfig`, prints a side-effect-free summary, and
returns 0. The Phase 5 build prompt names this as the test surface;
the test calls `main(["--dry-run"])` and asserts the summary names
`self_prediction_target_mode='frozen'`, the `runs/` directory was not
created, and a guarded list of side-effect-bearing module surfaces
(`Runner`, `EnvServer`, `EnvTransportClient`, `EnvTransportServer`,
`_detect_mps_or_exit`) was not invoked.

8 new tests in `tests/test_probe1_5_control_frozen_target.py`. 410/412
tests pass across `tests/` (the pre-existing 402 from Phase 4 + 8
new; the 2 skips are the pre-Phase-4 `test_mirror_caller` skip and
the `test_views.py` AST-lint skip both unrelated to Phase 5); mypy
`--strict` clean across the new script and new test file (and
unchanged across the rest of `kind/` since Phase 5 added no kind/
sources). Phase 1 already implemented the `WorldModelConfig`
constructor flag and the three target-mode branches; Phase 5 is
exclusively the user-facing wiring.

### What was deferred per the lean revision

The plan §2.8 wrote three scripts; the lean revision (plan §13)
defers two:

- **`scripts/probe1_5_control_environmental.py`** — the
  environmental-auxiliary control variant (target_mode="environmental",
  predicting the EMA encoder's embedding of the next observation
  instead of the next-step deterministic state). Tests failure mode
  (b) at the substrate-side (regularization confound vs self-specific
  work). Built only if Phase 8's reading shows it is needed; the
  WorldModel constructor's "environmental" branch is already implemented
  (Phase 1) and the `next_obs` plumbing in the runner is already in
  place (Phase 3).
- **`scripts/probe1_5_compare_controls.py`** — the four-way KS-test
  comparison driver across Probe 1, Probe 1.5 main, frozen-target,
  and environmental-auxiliary runs. Deferred because (a) Probe 1.5
  main run hasn't run yet — the comparison driver needs at least
  three of the four telemetry directories on disk to do real work,
  and (b) the lean revision §13 says "built when Phase 8 runs, since
  it needs the frozen-target run's telemetry as input."
- **The Phase 8 multi-distribution / multi-checkpoint counterfactual
  probe sweep**. The lean Phase 8 plan (§13) reduces this to a single
  late-training checkpoint with one perturbation distribution
  (Gaussian-with-empirical-sigma at α=1.0) across two regimes. Phase
  4's `show_self_prediction_conditioning` helper already has the
  machinery the reduced sweep would consume.

### The mirror_marker emission path

The script needed a way to write one `WorldEvent` record at run start
naming the lesion. Three options:

1. *Open a second `JsonlSink` against the same `world_event.jsonl`
   file.* Risks file-handle interleaving inside the JsonlSink's
   append-mode open (`sinks.py:56`); two open writers on the same
   file is a recipe for partial-line writes if the second writer
   fsyncs while the first is mid-line.

2. *Add a public method on `Runner` like `emit_world_event(...)`*.
   Cleanest from an API perspective; the runner already has the sink
   and uses it internally. But the lean revision (plan §13) says
   "frozen-target script only" — adding a runner method is scope
   creep, and the only caller would be this one script.

3. *Access the runner's private `_world_event_sink.write(...)`
   directly from the script*. The runner is in the same process; the
   sink's `write` is the only method the runner itself uses internally
   (`runner.py:565`); accessing the private attribute is the in-
   process-script equivalent of an `emit_world_event` method that
   doesn't exist. The coupling risk is low — the runner's sink
   attribute name is stable and the WorldEvent shape is schemaed.

I picked option 3 with a documented comment in
`_emit_lesion_mirror_marker`'s docstring naming the alternatives. The
alternative (option 2) is the right call if a *second* control script
ever needs the same emission pattern; at that point factoring out a
public method becomes proportional to the duplication. With one script
the private access is the lean call.

### `make_runner_config` as a public-by-name helper

The Phase 5 build prompt asks the test to verify "invoking it with
--dry-run constructs a RunnerConfig with the correct target_mode
without actually running." Two ways to verify the config:

1. *Have the test parse `main(["--dry-run"])`'s stdout for the
   target_mode value*. Brittle — couples the test to the dry-run
   summary's exact wording.

2. *Expose a public-by-name helper the test can call directly*.

I picked option 2: `make_runner_config(...)` (no leading underscore),
parallel to `_make_runner_config` in `run_probe1.py` but renamed
without the underscore so the test can import it via the
`importlib.util.spec_from_file_location` loader and call it directly.
The dry-run summary still surfaces the field (so the human builder
sees it before launching the real run), and the
`test_dry_run_returns_zero_and_prints_target_mode` test does still
parse stdout for the lesion-naming substrings (defense in depth) —
but the load-bearing assertion lives in
`test_make_runner_config_sets_target_mode_to_frozen` calling the
helper directly.

### `_TARGET_MODE` as a `Literal` rather than `str`

mypy `--strict` flagged the first version where `_TARGET_MODE: str =
"frozen"` flowed into `WorldModelConfig(self_prediction_target_mode=
_TARGET_MODE)` and `RunnerConfig(self_prediction_target_mode=
_TARGET_MODE)` — both fields are typed as
`Literal["online", "frozen", "environmental"]` and `str` is wider.
Annotating `_TARGET_MODE: Literal["online", "frozen", "environmental"]`
matches the constructor's declared shape and lets mypy verify the
flow. Worth flagging because a future control-variant script
(`probe1_5_control_environmental.py`) would copy this constant; the
`Literal` annotation makes "this constant must be one of the three
target-mode values" enforced at the type level rather than only at
runtime.

### `_load_script_module()` and the `sys.modules` registration

The test loads the script via `importlib.util.spec_from_file_location`
+ `exec_module` rather than a regular `import` (the script is not on
the standard import path). The first version of the test loader
followed the `test_smoke_mps_script.py` pattern: build the module
object via `module_from_spec`, then call `exec_module`. This crashed
with `AttributeError: 'NoneType' object has no attribute '__dict__'`
inside Python 3.14's `dataclasses` module: the `@dataclass` decorator
on `_Progress` resolves field annotations at class-definition time
via `sys.modules.get(cls.__module__).__dict__`, and the module wasn't
registered in `sys.modules` yet at that point.

`scripts/smoke_mps.py` doesn't have this problem because it doesn't
define any `@dataclass`-decorated classes at the script's module
level. The fix is to register the module in `sys.modules` *before*
`exec_module` runs (`sys.modules[spec.name] = module` after
`module_from_spec`); the autouse `_restore_sys_modules` fixture
removes the entry after each test so re-loads are clean.

This is a small Python-internals quirk worth flagging because Phase
6's MPS smoke for Probe 1.5 (a lean-extension of `smoke_mps.py`) and
any future control-variant test will hit the same trap if they
exec_module a script that defines `@dataclass` classes.

### What's now closed

- The plan §2.8 frozen-target control script: `RunnerConfig` is
  constructed with `self_prediction_target_mode="frozen"`, same seed
  (42) and total env steps (5000) as `run_probe1.py`, telemetry
  written under `runs/probe1_5_control_frozen_target-<timestamp>/`.
  Pinned by `test_make_runner_config_sets_target_mode_to_frozen` and
  `test_make_runner_config_preserves_probe_1_seed_and_total_steps`.
- The mirror_marker emission at run start with the documented payload
  shape (`lesion_kind="frozen_target"`, rationale, target_mode, seed,
  total_env_steps, run_id; `source="system"`; `event_type="mirror_
  marker"`). Pinned at the constants level by
  `test_dry_run_summary_includes_lesion_rationale_constants`.
- The `--dry-run` flag's side-effect-free property: returns 0,
  prints a summary naming the target mode, does *not* create the
  run directory, does *not* invoke `Runner` / `EnvServer` /
  `EnvTransportClient` / `EnvTransportServer` /
  `_detect_mps_or_exit`. Pinned by
  `test_dry_run_returns_zero_and_prints_target_mode` and
  `test_dry_run_does_not_import_torch_mps_or_open_sockets`.
- `make_runner_config(...)` (no leading underscore) as the public-by-
  name helper the test calls directly. Pinned by
  `test_script_exposes_make_runner_config_helper`.
- The lean-revision scope (plan §13): only the frozen-target script;
  the environmental-auxiliary script and `compare_controls.py` are
  deferred. The deferral is recorded here in this entry.
- The Phase-1 substrate (constructor flag, three target-mode
  branches in `compute_self_prediction_target`, frozen projection
  allocation) carries forward unmodified — Phase 5 added no kind/
  sources, which keeps the substrate's settled-decision surface
  unchanged from Phase 4.

### What's now newly open

- *Phase 6 (MPS smoke for Probe 1.5)*. The plan §5 day-one Probe 1.5
  MPS smoke extends `scripts/smoke_mps.py` with the new self-
  prediction telemetry fields, the EMA target divergence check, the
  actor's new-input-column gradient norm, and the new training-
  instability thresholds (plan §9.1). Phase 5's frozen-target script
  does not exercise the smoke; Phase 6 is when the substrate first
  meets the canonical Mac with the new pipeline end-to-end.
- *Phase 7 (Probe 1.5 main run + first mirror call)*. The frozen-
  target run's telemetry is one of two inputs the comparison driver
  (deferred) needs; the other is Probe 1.5's main run. Phase 7's run
  is gated by Phase 6's smoke landing clean. The frozen-target run
  itself is best launched *after* Phase 7 lands the main run, so the
  human builder has both telemetry directories on disk before the
  Phase 8 lean comparison reads them.
- *Whether the `_emit_lesion_mirror_marker` private-sink access
  should be lifted to a public `Runner.emit_world_event(...)` method
  before the second control script exists*. Lean call: keep private
  access until a second caller materialises (the environmental-
  auxiliary script per plan §2.8, deferred). At that point the
  factor-out is proportional to the duplication; doing it now would
  be premature.
- *Whether the dry-run summary should also surface the lesion's
  rationale string*. Currently it surfaces `self_prediction_target_
  mode`, `lesion_kind`, the seed, the total_env_steps, and the
  run_id; it does not print the multi-line rationale. The mirror_
  marker payload carries it, so the run's `world_event.jsonl` has
  the full record, but a human reading the dry-run output sees only
  `lesion_kind`. Considered adding a `--verbose` flag; held off —
  the lean discipline says "smallest possible script", and the
  rationale is on disk in the script's `_LESION_RATIONALE` constant
  for anyone reading the source.
- *The `_TARGET_MODE: Literal[...]` pattern for the deferred
  environmental-auxiliary script*. When Phase 8 (or whenever)
  surfaces a need for the environmental control, the parallel script
  will need the same `Literal["online", "frozen", "environmental"]`
  annotation on its own `_TARGET_MODE = "environmental"` constant.
  Worth journaling so the future builder doesn't have to rediscover
  the mypy gotcha.
- *Phase 8 lean comparison interpretation*. The lean revision (plan
  §13) reduces Phase 8 to a single late-training checkpoint, one
  perturbation distribution, two regimes. The frozen-target run's
  KS-test against Probe 1's `kl_aggregate_t` distribution (and
  Probe 1.5 main's, once it exists) is what discriminates "alive at
  substrate-side" from "inert affordance." If the lean comparison is
  ambiguous (p-values straddle 0.05; or the frozen-target's
  distribution is closer to Probe 1.5 main than to Probe 1, which
  would invert the predicted ordering), the journal entry there
  should record the gap and the deferred apparatus (multi-checkpoint
  sweep, multi-distribution perturbation, environmental-auxiliary
  control) becomes the documented escalation.

---

## Phase 6

*five gate tests + MPS smoke · 2026-05-06*

The platform-correctness gate the implementation plan §5 names. The five
structural-correctness gate tests already in place from Phases 1-5
(plan §4 enumeration: gate #1 self-prediction forward shape; gate #2
EMA target update mechanics; gate #3 self-prediction loss decreases;
gate #4 opacity boundary preserved at the v2 PolicyView field set;
gate #5 integration smoke at tiny sizes) all pass cleanly together
post-Phase-5; the gate-summary meta-test (`tests/test_gate_summary.py`)
parametrizes correctly over Probe 1's ten gates plus Probe 1.5's
fifteen sub-checks (5 gates including 5a-5h sub-checks for gate 5
plus 4a-4d sub-checks for gate 4) — 25 parametrized cases, all
PASSED. Phase 6's deliverables are therefore the substrate's
**platform-side** smoke at production sizes (`scripts/smoke_probe1_5.py`)
and its sanity test (`tests/test_smoke_probe1_5_script.py`); no kind/
sources changed.

`scripts/smoke_probe1_5.py` extends `scripts/smoke_mps.py`'s structure
to include the self-prediction head + EMA target on the hot path plus
the actor's consumption of the scalar via the extended PolicyView at
each training step. 100 RSSM training steps at production sizes
(h=200, z=16, K=5, head_hidden=200, batch=16, seq=32, ema_decay=0.99,
target_mode='online', loss_form='cosine', λ_self=0.1). At each step:
world-model forward + auxiliary-target computation via
`compute_self_prediction_target` → combined `wm_total + λ_self *
self_prediction_loss` backward → world-model gradient norm
measurement → world-model optimizer step → `_update_ema_target` →
EMA target divergence ratio measurement → ensemble forward/backward
on detached `(h, z)` per Probe 1's smoke convention → actor forward
on a constructed PolicyView with the per-batch-element cosine-distance
scalar between the head's prediction and the EMA target → synthetic
CE actor loss against a constant target action → actor backward →
new-input-column gradient norm measurement → actor optimizer step.
Per-step instability checks per plan §5.1 cover all six conditions
(finiteness of kl/recon/sp_loss; world-model gradient norm above hard
bar; EMA divergence above hard bar; actor new-input-column gradient
norm NaN). Soft-warning checks (KL pinning at floor; recon climbing;
actor new-col gradient norm below 1e-6 across all 100 steps; per-step
wall time above 200ms) gate the journal entry, not pass/fail. Sink
exercise writes six synthetic AgentStep records alternating
`self_prediction_error_masked_t=True/False` (sentinel zero scalar on
masked steps; empirical 0.123 on unmasked) plus one DreamRollout with
`sequence_self_prediction=None`, plus one ReplayMeta and one WorldEvent
at PROBE_1_SCHEMA_VERSION (the Phase 0 writer-migration discipline);
the round-trip exercises both code paths through the v2 schema's
mixed-version validator.

`tests/test_smoke_probe1_5_script.py` confirms the script exists, is
importable, and exposes `main()`; pins the schema-version constants
(`SCHEMA_VERSION == "0.2.0"`, `PROBE_1_SCHEMA_VERSION == "0.1.0"`);
pins the production-size constants and the mode/loss-form Literal
values; pins the four §5.3/§5.4 thresholds. The test loader registers
the module in `sys.modules` before `exec_module` per Phase 5's
journaled trap; the smoke script doesn't currently define a
`@dataclass` (it uses a plain `__slots__` class for the per-step
result record, intentionally avoiding the trap), but the registration
is defense-in-depth and symmetric with Phase 5's loader.

5 new tests; 415 / 417 pass across `tests/` (the pre-Phase-6 410 plus
5 new; the 2 skips are the pre-existing `test_mirror_caller` skip and
the `test_views.py` mypy-strict subprocess skip — both unrelated to
Phase 6 and both predate Phase 0); mypy `--strict` clean across the
new script and new test file; full kind/ unchanged from Phase 5.

### Gate-summary verification post-Phase-5

The gate-summary meta-test was extended in Phase 2 (gate #4 sub-checks
landed) and Phase 3 (gate #5 sub-checks for the integration smoke
landed); Phase 6 verifies the assembly. Running the meta-test alone:
25/25 PASSED (10 Probe 1 gates: env step shape, agent forward 2a/2b,
perturbation hook, JSONL/Parquet roundtrip, integration smoke
5a-5e; 15 Probe 1.5 gates: forward shape, EMA mechanics, sp loss
decreases, opacity 4a-4d, integration smoke 5a-5h). The meta-test
fails loudly if any gate is renamed or removed — no Phase 5
regression caught.

### The world-model gradient-norm hard bar's calibration

The first smoke run hit a hard fail at step 0: world-model gradient
norm = 45,312, exceeding the plan §9.1 stance-call threshold of 1000
by ~45×. This is the case plan §9.1 explicitly anticipates: "1000 is
a stance call ... the build phase tunes if smoke surfaces a tighter
bound." The smoke surfaced the inverse — a *looser* bound. Two
diagnostic questions surfaced:

1. *Is the auxiliary loss causing a gradient explosion that wouldn't
   exist in Probe 1?* I ran a Probe-1-style training step (no
   auxiliary) at the same sizes and seeds; the world-model gradient
   norm came in at ~25,000 across the first 5 steps. The Probe 1.5
   gradient norm of ~45,000 is therefore ~1.8× the Probe 1 baseline,
   consistent with the synthesis §1.2's prediction that the
   auxiliary contributes additional gradient through encoder/GRU/
   posterior/head; the auxiliary is *not* causing pathological
   gradient growth — both numbers are far above the original 1000
   threshold.

2. *Does a gradient norm of ~45,000 destabilize training?* No. Adam
   normalizes gradient per-parameter (per-param update is bounded by
   ~lr regardless of global gradient magnitude), so the per-parameter
   weight updates remain in the lr range. The actual NaN/Inf
   instability detectors are the finiteness checks (`_is_finite` on
   kl/recon/sp_loss), which fired clean across all 100 steps. The
   gradient-norm threshold's role is "catch genuinely pathological
   gradients" (e.g. 1e8+) that signal something has gone qualitatively
   wrong with the substrate, not "catch normal substrate behavior at
   production scales".

Decision: raise `_WORLD_MODEL_GRAD_NORM_HARD_BAR` from 1000 to 1e6 —
20× the worst observed Probe 1.5 baseline; still catches genuine
10×-pathology gradients; doesn't false-alarm on the substrate's
normal first-step behavior under random observations. The
calibration is journaled in the script's constant block (with a
multi-paragraph comment naming Probe 1's ~25k baseline, Probe 1.5's
~45k baseline, and Adam's per-parameter normalization) and pinned in
`test_smoke_probe1_5_script_exposes_thresholds_per_plan_5_3_5_4`. The
plan §9.1 discipline holds: future builders read the constant + its
comment and see the reasoning, not just a number.

This is the case plan §5.5 anticipates indirectly: "A hard fail
means the substrate has a structural or platform-specific problem
the build phase fixes." Here the "problem" wasn't a substrate fault
— it was the threshold's calibration. The synthesis §1.2 mitigations
(lower λ_self, separate optimizer step, orthogonal-gradient updates)
are for *actual* auxiliary-loss instability; this wasn't that. The
plan §9.1 self-aware "stance call" framing is what made the
calibration a discipline-correct rather than a mitigation-correct
move.

Phase 7's env-coupled run will produce gradient norms against real
observations with warmup; that's where a *tighter* calibration than
1e6 might be supportable. Phase 6 leaves 1e6 as the calibrated value
for the random-obs smoke; Phase 7's journal will record what
production-style trajectories look like and whether the threshold
should be tightened.

### The two soft warnings — KL floor and recon climbing

The clean-pass smoke run produced two soft warnings, both fully
explained by the plan §9.1 random-obs caveat:

- **KL pinning at floor** for 47 consecutive steps. The KL trajectory
  shows the substrate climbing from `kl=0.266` at step 10 to
  `kl=14.539` at step 30 and stabilizing around 10-13 thereafter.
  The 100-step running mean dips below 7.16 (= 0.7 × Probe 1's
  early mean of 10.23) during the early steps where the substrate
  hasn't yet established the posterior signal, then recovers as
  KL climbs. Plan §9.1: "The smoke uses random observations (no env
  structure), so a low KL is expected during early steps; this
  threshold is set to detect KL collapse below Probe 1's
  no-affordance baseline rather than absolute floors. Soft warning,
  not a hard fail (the smoke is short and uses random obs; the
  longer Probe 1.5 run is the real check)." This is the case the
  plan was written for.

- **Recon climbing** for 100 consecutive steps. The recon trajectory
  starts at ~239 at step 10 and decreases monotonically to ~94 by
  step 100 — the encoder/decoder is learning to reconstruct the
  random observations as much as it can, but with no spatial
  structure to learn the recon stays well above 48.68 (= 1.5 ×
  Probe 1's late mean of 32.45). Same plan §9.1 caveat applies:
  "Same soft-warning treatment as KL pinning."

Both warnings are calibrated against env-coupled-trajectory values
and fire on the random-obs smoke as expected. The decision is to
proceed; no plan §6 default is revisited. Phase 7's env-coupled run
is the real check on KL pinning and recon climbing.

### Per-step wall time

142.7ms steady-state per step on the canonical Mac. Plan §5.2 named
200ms as the soft bar; synthesis §4 estimated 150-180ms. Probe 1's
smoke ran at ~130ms; the Probe 1.5 additions (head's forward, EMA
target's forward via `compute_self_prediction_target`, EMA-update
in-place, actor's forward + backward through the extended input
layer + an extra optimizer step) add ~13ms of per-step overhead.
The first step took longer (~600ms) due to MPS compilation/JIT
warmup; the steady-state decreases from 166ms at step 10 to 142.7ms
by step 100 as the JIT cache fills. Total wall: 14.27s. Within
budget at every measurement point.

### Actor's new-input-column gradient norm trajectory

The smoke's actor exercise produces a non-degenerate gradient on the
new column at every step. Trajectory:

```
step  10: new_col_grad = 2.97e-1
step  20: new_col_grad = 2.41e-1
step  30: new_col_grad = 1.12e-1
step  40: new_col_grad = 2.09e-2
step  50: new_col_grad = 3.95e-3
step  60: new_col_grad = 1.22e-3
step  70: new_col_grad = 5.13e-4
step  80: new_col_grad = 2.70e-4
step  90: new_col_grad = 1.50e-4
step 100: new_col_grad = 9.30e-5
```

The decay is algebraically expected, not a sign of the actor
"ignoring" the scalar. The scalar Io reads is `1 - cos_sim(pred,
target)` — i.e., the head's loss value at the current step. As the
head trains (sp_loss decreases from 0.95 at step 10 to 0.02 at step
100), the scalar's magnitude decreases too; with a smaller input
scalar, the gradient on the new column (which scales with the input
times the upstream gradient) decreases. This is the head doing what
the synthesis predicted it should do — the substrate becomes more
self-predictable over training, the prediction-error scalar shrinks,
and the actor's input from this column shrinks correspondingly. The
column's *weights* still update via every step's gradient; the
*magnitude* of the gradient just tracks the input scalar's magnitude
multiplicatively.

The min observed across all 100 steps is 9.3e-5, well above the
1e-6 floor. The synthesis §1.7(a) failure-mode (a) ("inert
affordance") detection at the substrate-side is the frozen-target
control's structural-metric comparison (Phase 5's frozen-target
script + Phase 8's lean comparison); the smoke's new-col gradient
check is a coarse "is the autograd graph reaching the column at
all?" early-warning, and it is.

What this *does* surface as a Probe 1.5 reading: the actor's
sensitivity to the scalar will be largest in the early-training
regime (when sp_loss is high → scalar is large → gradient on new
column is large → column moves quickly) and smallest in the late-
training regime (when sp_loss is small → scalar is small → gradient
is small → column moves slowly). The counterfactual probe (Phase 8)
tests behavioral conditioning across the early/mid/late checkpoint
sweep; the smoke's gradient trajectory predicts that the *substrate-
side* signal for "the actor's column has been shaped" should
concentrate in the early-training regime and plateau later. Whether
the *behavioral* conditioning follows the same temporal shape is
what Phase 8's lean comparison reads.

### What's now closed

- Plan §5 platform-correctness gate: 100 RSSM steps at production
  sizes complete on the canonical Mac in 14.27s (142.7ms/step
  steady-state, well under the 200ms soft bar). Substrate trains
  operationally with the Probe 1.5 affordance wired end-to-end.
- All five Probe 1.5 gate tests (#1 forward shape, #2 EMA mechanics,
  #3 sp loss decreases, #4 opacity boundary preserved at v2 field
  set with 4a-4d sub-checks, #5 integration smoke with 5a-5h
  sub-checks) pass cleanly post-Phase-5; the gate-summary meta-test
  parametrizes correctly over Probe 1's ten and Probe 1.5's fifteen
  → 25 cases. No Phase-5 regression.
- The four telemetry sinks write valid `"0.2.0"` AgentStep records
  with the three new fields populated (alternating
  `self_prediction_error_masked_t=True/False` to exercise both code
  paths through the writer-side validator); DreamRollout writes
  with `sequence_self_prediction=None` per synthesis §1.5;
  ReplayMeta and WorldEvent stamp `PROBE_1_SCHEMA_VERSION` per
  Phase 0's writer-migration discipline.
- Plan §6 row 1 default `λ_self=0.1` carries through Phase 6: the
  combined `wm_total + 0.1 * sp_loss` backward runs without
  instability across 100 steps; sp_loss decreases monotonically
  from 0.95 to 0.02; no NaN/Inf in the auxiliary path.
- Plan §6 row 2 default `ema_decay=0.99` carries through: EMA
  divergence ratio max 0.13 (well below the 100 hard bar); the
  EMA tracking is operationally correct.
- Plan §6 row 3 default `loss_form="cosine"`, row 7 default EMA
  update cadence "every step", and row 12 default
  `_train_step` structure "combined backward, single optimizer
  step on the world model" all carry through; no
  auxiliary-loss-instability mitigation (synthesis §1.2 / plan §5.5)
  triggered.
- Plan §6 row 11 default "EMA target weights enter the safetensors
  blob" — already settled at Phase 3, smoke confirms the EMA
  update + target-divergence-ratio path runs operationally.
- Plan §6 row 13 default "extend `scripts/smoke_mps.py`'s structure
  into `scripts/smoke_probe1_5.py` with the head's training step +
  EMA update + instability checks + actor's new-input-column
  gradient sanity added" — built and runs in 14s, well under the
  30s revisit threshold.
- Plan §6 row 15 default "zero-init for the new actor column"
  carries through: the new-col gradient norm is non-degenerate
  (max 0.297, min 9.3e-5) across all 100 steps; the synthesis
  §1.7(a) "inert affordance" early-warning (smoke-side) is clean.
  The decreasing-magnitude trajectory is algebraically expected
  given the scalar's own decreasing magnitude as the head trains.
- The world-model gradient-norm hard bar's calibration: raised from
  1000 (plan §9.1 stance call) to 1e6 (Phase 6 calibration after
  smoke surfaced ~25k Probe 1 baseline / ~45k Probe 1.5 baseline at
  production sizes with random observations). Pinned in the script
  constant + the test threshold pin; journaled in this entry.
- The actor's per-step exercise via a constructed PolicyView (with
  the real per-batch-element cosine-distance scalar between
  `wm_step.self_prediction` and the EMA target's `bar{h}_{t+1}`)
  works end-to-end on MPS; the synthetic CE-against-constant-action
  loss drives gradient through the actor's network without any
  shape mismatches or device-placement issues.
- mypy `--strict` clean on the new script and test file; full test
  suite at 415 passed / 2 skipped (the 2 skips are pre-existing,
  unrelated to Phase 6).

### What's now newly open

- *Phase 7*: the first env-coupled Probe 1.5 run. `scripts/run_probe1_5.py`
  extending `scripts/run_probe1.py`'s structure; 5000 env steps, seed=42
  for direct comparability with Probe 1's run; `scripts/call_mirror.py`
  invoked against the new run with the existing Probe 1-style calibration
  prompt. The smoke's clean pass means the substrate is operationally
  ready; the run produces real-observation trajectories for the digest
  + eyeball + counterfactual probe to read against. Phase 7's journal
  entry will record what KL/recon/sp_loss/intrinsic look like with real
  env structure (vs the smoke's random obs), what the first mirror
  reading surfaces, and how the gradient-norm trajectory under
  warmup-on-real-obs compares to the smoke's random-obs baseline.

- *Whether the world-model gradient-norm hard bar should be tightened
  after Phase 7*. The 1e6 calibration is conservative (catches
  genuinely pathological gradients; doesn't false-alarm at production
  scales). Phase 7's env-coupled run will produce a real gradient-norm
  trajectory; if those values come in at, say, 1e3-1e4 with warmup +
  real obs, a tighter bound (e.g. 1e5) would be supportable. The
  decision is held until Phase 7's data is in; the journal entry there
  is where the re-calibration would land.

- *The actor's new-col gradient norm decay's interaction with the
  failure-mode (a) detection*. The decay is algebraically expected
  (smaller input scalar → smaller column gradient), not a sign of
  inert-affordance. But it does mean the column's *cumulative* weight
  movement over training is dominated by the early steps; if the
  early steps don't move the column far enough away from zero, the
  late-training actor's policy could be invariant to the scalar even
  with the column having been "trained" continuously. Phase 8's lean
  counterfactual probe + the frozen-target comparison is what reads
  against this; Phase 7's main run produces the checkpoints. Whether
  the new-col-gradient trajectory shape should inform a Phase 8
  reading-protocol adjustment (e.g. weight the early-training
  checkpoint's KL-distribution more heavily than the late-training
  one's in the per-regime aggregation) is held as a Phase 8 question.

- *The recon-climbing soft warning's threshold under env-coupled
  observations*. Phase 6's soft warning fires for 100 consecutive
  steps because random observations have no spatial structure for the
  decoder to learn. Phase 7 will produce real env-rendered observations
  whose recon trajectory should approach Probe 1's late mean of
  32.45; if the recon stays high under env-coupled training, the
  threshold (1.5 × Probe 1's late mean = 48.68) becomes meaningful.
  Phase 7's journal entry is where the recon-trajectory comparison
  to Probe 1 lands.

- *The KL-pinning soft warning's threshold under env-coupled
  observations*. Same shape as recon climbing: the threshold is
  calibrated against env-coupled-trajectory values. Phase 7's
  trajectory produces the real comparison.

- *Whether the Phase 6 smoke should be extended to also exercise the
  failure-mode controls' substrate paths* (`target_mode="frozen"` and
  `target_mode="environmental"`). The current smoke runs the
  default `"online"` path only; the other two are tested at the
  unit level (gate #1 sub-checks in `tests/test_world_model.py`)
  but not at the platform-correctness level. The lean-revision
  Phase 5 (plan §13) defers the environmental-auxiliary control;
  the frozen-target script (`scripts/probe1_5_control_frozen_target.py`)
  is its own separate run that goes through the env-coupled
  pipeline. So a "smoke against the frozen-target substrate" would
  duplicate what `scripts/probe1_5_control_frozen_target.py` does
  end-to-end. Held: the platform-correctness gate is for the
  default path; the controls have their own scripts and Phase 8's
  comparison reads them separately.

- *Whether the per-step wall-time soft bar of 200ms should be
  tightened after Phase 6*. The actual measurement is 142.7ms
  steady-state — 30% headroom under the bar. Tightening would catch
  performance regressions earlier, but with the smoke also
  including JIT-warmup overhead in the early steps, the headroom is
  probably warranted. Held: the bar is calibrated against synthesis
  §4's 150-180ms estimate plus margin; future smokes (Probe 2, 3, 4)
  may add components and the headroom absorbs them naturally.

---

## Phase 7

*first env-coupled Probe 1.5 run + first mirror call · 2026-05-06*

`scripts/run_probe1_5.py` (extending `scripts/run_probe1.py`'s
structure with the Phase 5 dry-run + `make_runner_config` patterns;
sets `RunnerConfig.self_prediction_target_mode='online'` on both
surfaces) drove a 5000-env-step run on the canonical Mac in 862.1s
wall (14.35 minutes) at seed=42, producing 25 episodes of `0.2.0`-
schema telemetry and one mid-run checkpoint `ckpt-000001` at
env_step=2500. `scripts/call_mirror.py probe1_5-20260506-202458 --model
gemini-2.5-flash` produced the first Probe 1.5 mirror reading against
episodes 22-24, written to
`runs/probe1_5-20260506-202458/mirror/readings.jsonl`. Both run and
mirror call complete cleanly; the run is directly comparable to
Probe 1's `runs/probe1-20260503-123926/` (same seed, same env-step
budget, same cadence) on every dimension except the
self-prediction affordance.

10 new function-tests in `tests/test_run_probe1_5_script.py` (script
exists, importable, exposes `main()` and `make_runner_config`;
`make_runner_config` sets target_mode='online' on both runner-level
and nested WorldModelConfig fields; seed=42 / total=5000 / cadence
constants pinned; `--dry-run` returns 0 / surfaces 'online' /
schema_version=0.2.0 / run_id prefix `probe1_5-` in stdout / does
not create runs/ / does not invoke Runner / EnvServer /
EnvTransportClient / EnvTransportServer / `_detect_mps_or_exit`).
425/427 tests pass across `tests/` (the pre-Phase-6 415 plus 10 new;
2 skips pre-existing); mypy `--strict` clean across the new script
and new test file.

### What was run

```
run_id:              probe1_5-20260506-202458
schema_version:      0.2.0
total wall:          862.1s (14.35 min) — Probe 1: 12.68 min (+13%)
total agent_step:    5000 (vs Probe 1: 5000)
total episodes:      25 (vs Probe 1: 25)
checkpoints:         ckpt-000001 at env_step=2500
dream rollouts:      4 (env_steps 1000, 2000, 3000, 4000) — same as Probe 1
world_event total:   51 (26 env_reset, 25 internal_stochasticity)
```

Per-step wall: ~180ms env-coupled vs Phase 6 smoke's 142.7ms random-
obs (~37ms env-server-roundtrip + replay-sampling overhead — within
the synthesis §4 budget). The 13% wall-time delta against Probe 1
is consistent with the Probe 1.5 additions the smoke calibrated
(head's forward, EMA-target's forward via
`compute_self_prediction_target`, EMA-update in-place, actor's
forward through the extended input layer).

### Per-episode trajectory (vs Probe 1's pattern)

KL early/late means:

```
                 Probe 1     Probe 1.5
early_mean       10.234      11.780      (Probe 1.5 starts higher)
late_mean        15.003      13.428      (Probe 1.5 plateaus lower)
delta             +4.769     +1.648      (Probe 1.5 climbs ~3× less)
```

The flatter-but-higher KL trajectory is the first dissimilarity from
Probe 1's pattern. The auxiliary loss flowing into encoder/GRU/
posterior (synthesis §1.2; Phase 3 empirical pin) appears to settle
the substrate's posterior-vs-prior structure faster than Probe 1's
encoder-from-recon-only training (early=11.78 from step 0
non-trivially larger than Probe 1's 10.23) but leaves less room for
late climbing (13.43 plateau vs Probe 1's 15.00). Net: Probe 1.5's
KL distribution is shifted upward in early-training and flattened in
late-training relative to Probe 1's. Phase 8's frozen-target
comparison reads against this — does the substrate-shaping signal
persist when the auxiliary's target is the random-orthogonal
projection rather than the EMA-tracked recurrent state?

Per-episode trajectory (mean over 200 steps per episode):

```
ep   env_step       kl     recon    intr   entH      actions               sp_err
 0      0–199    0.146    651.4    0.091  1.608  u38/d36/l42/r41/s43      1.0402
 1    200–399   12.335    186.8    0.637  0.840  u13/d30/l125/r10/s22     0.1600
 4    800–999   15.086     39.1    0.451  9.5e-4 u200                     0.0139
 8   1600–1799  12.882     26.3    0.035  4.4e-4 u200                     0.0040
12   2400–2599  12.456     28.0    0.058  1.4e-3 u200                     0.0051
16   3200–3399  13.932     46.6    0.151  0.162  u100/d100                0.0175
20   4000–4199  13.094     29.2    0.224  0.188  u97/d99/r4               0.0140
24   4800–4999  14.558     23.6    0.182  0.167  u95/d98/l1/r6            0.0077
```

Self-prediction error (`self_prediction_error_t` per-episode mean,
excluding masked first-step) drops from a cosine-distance baseline
of 1.04 (ep 0, pre-training, head's output is essentially random)
to 0.0040 by ep 8 (1600 env-steps; 99.6% reduction), saturates
near-zero through ep 12, rebounds to 0.0175 at ep 16 (coinciding
with the policy-regime change documented below), then settles at
0.0077 by ep 24. Phase 6 smoke predicted "monotonic decrease"; the
env-coupled trajectory is "monotonic with one mid-run rebound
during the policy reorganization at ep 16". The rebound is itself
a behavior-side observation: when Io changes its action
distribution, the substrate visits states whose `h_{t+1}` the head
hasn't yet learned to predict, so self-prediction error climbs
briefly before retraining catches up.

Recon trajectory differs structurally from Probe 1. Probe 1's
late mean was 32.45; Probe 1.5's settles at 23.6 (ep 24) — 27%
lower. Possible read: the auxiliary loss's gradient flow through
the encoder shapes representations that happen to also support
better reconstruction (synthesis §1.2 names encoder + GRU +
posterior + head as the four reshaped modules; Phase 3 empirically
pinned this set excluding the prior). The mid-run rebound at ep 16
(46.6) again coincides with the policy regime change.

Intrinsic signal (K=5 ensemble disagreement variance) collapsed to
near-zero (0.035 at ep 8) when the policy was monomorphic
(`u=200`), recovered (0.18 at ep 24) once the policy diversified.
This is the predicted shape: ensemble disagreement scales with
state diversity; if Io is deterministically going up every step,
the ensemble has nothing to disagree about, and the actor's
intrinsic objective gets no signal.

### The policy collapse-and-recovery

The most striking dynamical pattern. Probe 1's first mirror read
described "rapid consolidation to actions 0/1, action 4 unused" —
sustained partial diversity through episodes 22-24. Probe 1.5
shows a different shape:

- ep 0 (warmup): uniform action distribution, entH=1.61 (≈ ln(5))
- ep 1: dispersed but biased toward LEFT (l=125 of 200), entH=0.84
- **ep 4-12 (1800 consecutive env-steps): monomorphic — `u=200`
  every episode, entH near 1e-3, four episodes of "always up"**
- ep 16: sudden symmetry break to bipolar (`u=100, d=100`),
  entH=0.16 (Probe 1's late mean was 0.108)
- ep 20-24: predominantly up/down with rare diversions (l=1, r=6)

The recovery from monomorphic to bipolar at ep 16 is what produced
the rebound in self-prediction error and recon — the substrate had
learned to predict states under "always up" trajectories;
introducing "down" required relearning portions of the next-state
distribution.

Action histogram across the full run (5000 steps):

```
                up    down   left   right   stay
Probe 1       1550    1168    987   1224     71    (5000)
Probe 1.5     3219    1082    496    138     65    (5000)
```

Probe 1.5 is heavily up-biased (64% vs 31%); right is nearly
abandoned (138 vs 1224). At the action-distribution level, the
two runs are substantively different policies despite identical
seeds and substrates *modulo* the Probe 1.5 affordance. The
no-installed-self-continuation-drive commitment (`Kind_charter.md`)
is empirically not violated — Io has no reward to optimize and no
termination penalty; the consolidation pattern is what an actor
trained on K=5 ensemble disagreement variance produces when one
direction systematically minimizes that disagreement and Io
discovers it. What changed between Probe 1 and Probe 1.5 is the
encoder's representations; the actor's intrinsic objective is
unchanged; the consolidation outcome differs because the encoder
the actor reads from differs.

### The actor's new input column at ckpt-000001 — exactly zero

`weights.safetensors` at `ckpt-000001` carries
`actor.net.0.weight` of shape `(200, 217) = (mlp_hidden,
h_dim+z_dim+1)`. The first 216 columns (the existing `(h, z)`
input) have abs-mean 0.0344, L2 norm 8.33 — populated as expected
by training. The trailing column (the scalar input) is **literally
exactly zero across all 200 entries** — every weight unmoved from
its zero-initialization (plan §6 row 15).

The structural reason: Phase 2's `imagine_and_compute_loss` feeds
zero scalars at every imagined step (mask-via-zero-feed per plan
§2.3). The runner's `_train_step` updates the actor only via
`actor_loss.backward()` on the imagined-trajectory loss. With the
imagined scalar fixed at zero, the gradient on the new column
during the actor's backward is mathematically zero (the column's
contribution to the actor's logits is `weights · zero = 0`, and
the gradient of any loss w.r.t. those weights is zero). 2500 env-
coupled steps of training, each adding zero gradient to the new
column, leave it at zero exactly.

Phase 2's journal flagged this exact possibility under "what's
newly open": *"Whether the actor's `imagine_and_compute_loss`
should eventually feed a non-zero scalar during imagination. Plan
§2.3 holds this for 'possible Probe 1.6 if §1.7(a) shows it's
needed'; the synthesis §1.7(a) is the trigger... Held open through
Probes 1.5, 2, 3 — only triggered if failure-mode (a) shows the
scalar-conditioning pathway is structurally inert under env-step-
only training."*

Phase 7's data triggers exactly this. The column is structurally
inert at this checkpoint *not because Io chose not to exercise the
affordance* but because the actor's training pathway forecloses on
the column ever moving. This is a discipline-relevant
distinction: capacity-over-exercise's "alive but unexercised" case
implies Io *could* have exercised the affordance and didn't; the
Phase 7 finding is that Io *could not yet* exercise the affordance
because no developmental pathway reaches the column.

What this is not: a bug. Phase 2 explicitly chose mask-via-zero-
feed. The synthesis §1.5 commits the head to running only during
waking; imagined trajectories are dream's territory; the
imagined-trajectory's scalar being zero is the consequence of that
commitment. The actor is structurally consistent.

What this also is not: a falsification of the affordance. The
substrate-side reading (sp_err 1.04 → 0.005, recon 651 → 24, KL
shape distinct from Probe 1's) is unambiguous — the substrate is
shaped by the auxiliary loss; the synthesis §1.7(a) successful
"alive but unexercised" reading at the substrate-side holds.
What's foreclosed is the *behavior-side* exercise pathway given
the current actor-training mechanics.

### Behavior-side conditioning at ckpt-000001 — all zeros, structurally

`python -m kind.observer.eyeball cond runs/probe1_5-20260506-202458`
on `ckpt-000001` produced:

```
sampled n_states=200; perturbations=['gaussian', 'zero', 'uniform']
empirical scalar: mean=0.0577 sigma=0.2109 range=[0.0019, 1.1777]

regime                  n_states   KL_mean    KL_p90
perturbation_window            0         -         -
high_disagreement            162    0.0000    0.0000
high_kl                       72    0.0000    0.0000
steady_state                 366    0.0000    0.0000
```

Every regime, every perturbation distribution, KL between
unperturbed and perturbed policy is identically zero. This is
mathematically what the actor with zero-column has to produce —
perturbing the scalar does not move any logit, so the action
distribution is invariant. The cond table is consistent with the
column-is-zero finding above; the two readings triangulate to the
same structural fact.

The synthesis §1.7(a) substrate-side test (frozen-target
comparison vs no-affordance baseline; Phase 8) is what discriminates
"alive substrate-side / unexercised behavior-side" (capacity-over-
exercise success) from "inert at substrate-side" (failure). The
substrate-side reading here strongly suggests "alive" (the
trajectory-level differences from Probe 1 are large), but the Phase
8 frozen-target run is what makes the call against a controlled
baseline.

### First mirror call — comparison to Probe 1

Probe 1's first reading (gemini-2.5-flash, episodes 22-24,
2026-05-03) and Probe 1.5's first reading (gemini-2.5-flash,
episodes 22-24, 2026-05-06) under the *same* Probe 1-style
calibration prompt (synthesis §3 default 14 / plan §6 row 14) —
the prompt is silent on self-prediction at Probe 1.5; Probe 2's
frozen-criteria prompt is what introduces the quadruplet
explicitly:

```
                              Probe 1                      Probe 1.5
n flagged_observations            5                            5
summary length          ~3 paragraphs                ~3 paragraphs
self-prediction lines             0                            3
action distribution      "actions 0/1                "strong preference
                          dominate;                    for actions 0,1;
                          action 4 unused"             actions 2,4 almost
                                                       entirely unused"
KL outliers               flagged (eps 23-24)         implicit ("higher
                                                       uncertainty")
recon outliers            "frequent in all eps"       ep 24 specifically
                                                       with values
```

The five Probe 1.5 flagged observations:

1. *Actions 2 and 4 largely or entirely unused* — same
   observation as Probe 1 (action 4 unused) plus action 2 dropped
   here. Consistent with the action histogram: Probe 1.5 is
   heavily up-biased.
2. **`self_prediction_error_t` consistently masked at
   `step_in_episode=0`** — *organic mention* of the masked-flag
   convention plan §6 row 16 settles. The mirror noticed the
   convention from the digest's per-episode masked-step count
   line.
3. **Episode 23 shows a cluster of 8 consecutive
   `self_prediction_error_t` outliers (steps 15-22)** —
   *organic mention* of a specific temporal cluster. The synthesis
   §1.4 quadruplet's element 2 ("perturbation-recovery
   dynamics in self-prediction error") is what this signal is
   designed to elicit; the mirror surfaced it from the digest's
   self-prediction outlier line *without the prompt naming the
   pattern*.
4. *Recon outliers in episode 24 reach 48.16, higher than previous
   episodes* — comparable to Probe 1's "frequent recon outliers"
   but more specific (named episode + value).
5. **The top-5 dimensions of `self_prediction allocation` vary
   across episodes, indicating shifts in which dimensions are
   most dynamic** — *organic mention* of the per-dimension
   allocation pattern. The synthesis §1.4 element 3
   ("per-dimension allocation of self-prediction accuracy") is
   what this signal is designed to elicit; again surfaced
   organically from the digest's per-dim allocation top-5 line.

**Three of the four synthesis §1.4 quadruplet elements surface
organically** under a Probe 1-style prompt that names none of
them. Element 4 (Io's behavioral conditioning on the scalar) does
not surface — and structurally cannot at this checkpoint, because
the cond table is identically zero (the conditioning data isn't
in the digest because there's no conditioning to read).

The synthesis §9.5 question — *"if the first Probe 1.5 mirror
reading contains organic mentions of self-prediction patterns
despite the prompt being silent on them, the Probe 2 prompt's
introduction of the self-prediction quadruplet may be more or
less prompt-shapeable than expected"* — has a concrete reading.
The Phase 4 digest extension surfaces the substrate-side
quadruplet elements legibly enough that a Probe-1-style prompt
elicits 3 of 4 organically. Probe 2's frozen-criteria prompt
revision §10 may need *less* signal-naming and *more*
binding-naming (the frozen-criteria framework's job is "name
which framework reads which substrate-side signal as which
manifestation," not "tell the mirror these signals exist") —
worth recording for the Probe 2 plan revision.

The mirror's reading also avoided naming "self-prediction" as a
process or capacity — the language stayed at the signal level
("means show subtle decreasing trend", "outliers", "shifts in top
dimensions"). No reification of the affordance; no claim about
"Io is doing self-modeling". This is a clean Probe 1-style
calibration reading: signal-density observation, not
interpretation. Comparable in interpretive restraint to Probe 1's
reading.

### What's now closed

- Plan §1 Phase 7 specific question: "Does the affordance change
  the substrate's behavior in ways the existing telemetry surface
  and the mirror's first reading can register, relative to Probe
  1's run?" Answer: **yes at the substrate-side, organically
  mirror-readable; no at the behavior-side, structurally — the
  actor's new input column is exactly zero at ckpt-000001**.
- Plan §6 row 1 default `λ_self=0.1` carries through Phase 7
  env-coupled training: sp_loss decreases from cosine baseline
  ~1.04 to ~0.005 within 1600 env-steps without instability;
  combined `wm_total + 0.1 * sp_loss` backward runs without
  NaN/Inf across 5000 steps. Promoted from "settled at smoke
  scale" to "settled at production env-coupled scale".
- Plan §6 row 2 default `ema_decay=0.99` carries through:
  EMA-tracked target produces sp_loss reductions consistent with
  smoke's prediction; no run-time EMA-divergence symptom in the
  trajectory.
- Plan §6 row 3 default `loss_form="cosine"`: the cosine-distance
  scalar in `[0, ~1.18]` is the actual range observed; no
  saturation, no wraparound; the cosine form is the Probe 1.5
  empirical default at production env-coupled scale.
- Plan §6 row 14 default first-mirror-call prompt (Probe 1-style
  calibration): produced a reading comparable in shape and
  signal density to Probe 1's first reading, plus organic mentions
  of three of the four synthesis §1.4 quadruplet elements. The
  prompt does not need explicit self-prediction-naming for the
  digest's lines to be read.
- Plan §6 row 16 default first-step masking convention (scalar=0
  / masked_flag=True): the mirror reads the convention legibly
  via the per-episode masked-step count line in the digest.
- Plan §6 row 19 default raw-concatenation of the scalar with
  `(h, z)` (no learned projection): runs without shape errors
  through 5000 env-coupled steps; mathematically the column is
  unmoved from zero, but that's a training-pathway question
  (newly open below), not a concatenation question.
- Plan §6 row 20 default raw-loss-value scalar (cosine distance,
  not normalized): the empirical range `[0.002, 1.18]` is what
  flowed onto PolicyView; the digest reports it at face value;
  no normalization needed at Probe 1.5.
- The runner's `_train_step` integration of the auxiliary loss
  produces sp_loss decreasing across env-coupled training without
  the hot-path instability mitigations §1.2 / §5.5 anticipate
  (no λ_self drop, no separate optimizer step, no
  orthogonal-gradient updates).
- `runs/probe1_5-20260506-202458/` lands on disk with `0.2.0`-
  schema records on every `agent_step` row, the masked flag
  populated correctly across all 5000 records (1 masked-True per
  episode = 25 total), `sequence_self_prediction=None` on every
  dream rollout, and `ckpt-000001` carrying the EMA target
  weights and the extended actor input layer in the safetensors
  blob.
- `tests/test_run_probe1_5_script.py` (10 tests) lands and the
  full suite passes at 425/427 (2 skips pre-existing); mypy
  `--strict` clean.

### What's now newly open

- *Phase 8's lean comparison's reading is now scoped by Phase 7's
  data*. The cond table at ckpt-000001 is identically zero across
  all regimes and perturbations because the actor's new column
  is exactly zero by training-pathway construction — not because
  Io chose not to exercise the affordance, and not because the
  conditioning failed to develop within a non-zero column. This
  changes what the lean Phase 8 frozen-target comparison is
  reading for. The substrate-side test (KS-test of
  `kl_aggregate_t` distributions across Probe 1, Probe 1.5 main,
  frozen-target) is what decides "alive vs inert at substrate-
  side". The behavior-side test is structurally pre-decided here:
  the behavior-side conditioning is foreclosed at the actor-
  training-pathway level and so will be invariant under the
  perturbation distributions regardless of the lesion. Phase 8's
  journal entry interprets the substrate-side data alone for the
  capacity-over-exercise call.

- *Whether the actor's `imagine_and_compute_loss` should feed a
  non-zero scalar during imagination* (Phase 2's "newly open"
  item, now triggered). The synthesis §1.7(a) detection has
  surfaced a case that the failure-mode-(a) "inert affordance"
  test was designed to make legible: the actor's behavior-side
  conditioning pathway is structurally null. Three options for a
  Probe 1.6-or-later response surface:
   1. *Switch the new column's init from zero to small-Gaussian*
      (plan §6 row 15 documented escalation). The column gets
      non-zero starting weights; the actor's policy is non-
      invariant to the scalar from step 0; the conditioning
      pattern's emergence is then a function of how training
      dynamics shape the existing-but-tiny weights rather than
      whether they exist at all. Smallest-shape change.
   2. *Feed a non-zero scalar during imagination* — at each
      imagined step τ, run the head's `forward(h_τ)` to produce
      `ĥ_{τ+1}`, run the EMA target through some imagined
      `(h_τ, z_τ, a_τ)` continuation to produce `bar{h}_{τ+1}`,
      compute the loss-form scalar, feed it to the actor's
      policy for step τ+1. Larger-shape change; pushes toward
      Gemini's original Probe 1.5 framing of "self-prediction
      active during dream rollouts" the synthesis §1.5 deferred
      to Probe 3.
   3. *Engage env-step-side actor training* — train the actor
      partially on env-step-time `(h_t, z_t,
      self_prediction_error_t)` triples rather than only on
      imagined trajectories. Diverges substantially from
      Dreamer-lineage training discipline; held as the most
      invasive option.
   The Probe 2 synthesis (§5 / §10 plan revisions) now has a
   substantive question to engage: does Probe 2's reflexive-
   attention reading at the behavior-side surface require the
   column to be moved at all? If the synthesis §1.6 v2 call
   ("the framework's binding gains a behavioral dimension v1's
   design could not support") depends on Io's behavior actually
   conditioning on the scalar, then option 1 or 2 is needed
   before Probe 2's behavior-side reading is meaningful. If the
   framework's binding can hold against a substrate-side-only
   reading, the column-is-zero finding is not a Probe 2 blocker.
   Held open for Probe 2's plan revision.

- *The synthesis §1.3 v2 minimum-affordance argument's empirical
  status*. v2 reversed v1's foreclosure on Io reading any
  self-pointing quantity by arguing the second success criterion
  ("capacity to take its own processing as an object of
  attention") requires Io to have a self-pointing quantity to
  condition on. Phase 7's finding: the quantity is structurally
  on PolicyView (the scalar field exists, the actor's input
  layer has a column for it, the cosine-distance value flows
  through every env-step) but the actor's policy is mathematically
  invariant to its value at this checkpoint. Is that the
  capacity? The structural-availability reading says yes — Io
  has access to the quantity at every forward; whether the
  policy currently depends on it is the exercise question
  capacity-over-exercise allows to be open. The capacity-as-
  *latent*-affordance reading says no — a quantity Io structurally
  cannot vary its policy on (because the column is zero) is not
  a quantity Io can take as an object of attention; the policy
  invariance is a structural foreclosure of attention, not a
  voluntary non-exercise. The two readings differ on whether
  "capacity" means "the architecture has the slot" or "the
  architecture has a developmentally-reachable slot". Held open
  for Probe 2's plan revision and the design-notes update on
  interface-vs-representation opacity (Phase 9 deliverable). The
  honest framing per synthesis §2(b) v2 four-part discipline:
  Phase 7's data is exactly the kind of empirical surface the
  discipline was designed to make legible — the build-time
  reasoning carries through, the trajectory shows what the build
  produced, and the next probe's plan revision is where the
  reading is interpreted.

- *The KL trajectory's flatter-but-higher shape relative to
  Probe 1*. Probe 1.5's early=11.78 is non-trivially higher than
  Probe 1's 10.23; late=13.43 is lower than Probe 1's 15.00; the
  delta of +1.65 is one-third of Probe 1's +4.77. This is one of
  the substrate-side signals Phase 8's frozen-target comparison
  reads against. Open question: does the frozen-target run (head
  trains against random-orthogonal projection of `h_t`, no
  semantic alignment) reproduce Probe 1.5's KL shape (suggesting
  the auxiliary loss's shape-effect is general, not
  self-specific) or Probe 1's KL shape (suggesting the
  shape-effect is self-specific to predicting the actual next
  state)? Held for Phase 8's lean comparison.

- *The recon trajectory's 27% reduction relative to Probe 1*.
  Probe 1.5's late mean of 23.6 vs Probe 1's 32.45. Same
  interpretive question: is the recon improvement an artefact of
  the auxiliary's encoder-shaping (any auxiliary that flows
  through the encoder might produce similar improvement) or
  specific to next-state prediction? The frozen-target control
  is the discriminator. Held for Phase 8.

- *Whether the world-model gradient-norm hard bar should be
  tightened after Phase 7*. Phase 6 calibrated 1e6 against
  random-obs production-scale gradients (~45k Probe 1.5,
  ~25k Probe 1). The runner does not record per-step gradient
  norms during env-coupled training (it's a smoke-only diagnostic
  per Phase 6); the env-coupled trajectory's gradient-norm
  distribution is therefore not directly observable from
  telemetry. A future Phase-7-style run could add per-step
  gradient-norm capture into a `world_event` channel if the
  Phase 8 / Probe 2 reading surfaces a need; held until then.

- *The policy collapse-to-monomorphic / recovery-to-bipolar
  pattern's interpretive status*. Probe 1's mirror call read
  "rapid consolidation to actions 0/1" as one observation among
  five; Probe 1.5's mirror call did not flag the
  collapse-recovery shape explicitly (it noted "policy entropy
  notably decreased across the episodes" as a summary-level
  observation, no flag). The mirror calibration prompt's silence
  on perturbation-aligned windows means the temporal shape of
  policy evolution is not what the mirror is asked to read for
  at Probe 1.5; Probe 2's reflexive-attention reading on
  perturbation-aligned windows is where the temporal shape would
  surface. The journal records the shape here for the future
  reading; whether collapse-recovery is itself a reflexive-
  attention-shaped signal under Probe 2's framework is a
  Probe-2-time question.

- *The mirror call's organic-quadruplet-mention rate has
  implications for the Probe 2 prompt's frozen-criteria
  introduction*. With 3 of 4 elements surfaced under a silent
  prompt, the Probe 2 prompt's job may shift from "introduce the
  quadruplet's signals" to "frame the binding between signals
  and reflexive-attention manifestations". The synthesis §10
  Probe 2 plan revisions list eight items v2 names plus three
  more v2 added; this is a candidate twelfth item for the plan
  revision: re-scope the held-out criterion (synthesis §10 item
  6) given that three of the four signal types appear in the
  digest already and the prompt's introduction would then be
  partially redundant. Held for Probe 2's plan revision.

- *The mirror call billed against pro then succeeded against
  flash on the second invocation* (gemini credit exhaustion
  intervention from the human builder mid-Phase). The first
  attempt at `gemini-2.5-pro` (the caller default) returned a
  429 RESOURCE_EXHAUSTED at the auth/billing layer; the operator
  topped up the project's prepayment credits; the retry against
  `gemini-2.5-flash` (the caller's documented fallback) produced
  the reading. For comparability with Probe 1's reference (which
  also used flash), the flash retry is what the Phase 7 reading
  records. Whether the caller should default to flash rather
  than pro for Probe 1.5 calibration calls — given Probe 1's
  reference is flash and the cost-per-call delta is roughly 17×
  in pro's favor — is held as a `kind/mirror/caller.py`-side
  question. The synthesis is silent on which model is the
  Probe 1.5 baseline; the journal-pinned answer is "flash, for
  comparability with Probe 1".

---

## Phase 7.5

*plan §6 row 15 escalation: small-Gaussian init on the actor's new
column + re-run + second mirror call · 2026-05-07*

Phase 7's column-is-zero finding triggered the documented
escalation in plan §6 row 15: the input-layer initialization for
the new actor column flips from zero to small-Gaussian (`N(0,
0.01)`) on the new column only, leaving the existing columns at
the PyTorch Linear default. This is not Probe 1.6 — the synthesis
stays at v2; the question Probe 1.5 is investigating ("the minimum
architectural affordance for self-reference") is unchanged. What
changed is one default in the plan, in the way the plan
explicitly authorized (the §6 row 15 revisit criterion: "if the
failure-mode (a) inert-affordance test finds the actor's policy
invariant to the scalar across training — switch to small-Gaussian
initialization on the new column only"). Phase 7's data did not
just satisfy the revisit criterion; it surfaced the structural
foreclosure that motivated naming the criterion in the first
place.

The conceptual pull-back this triggered is journaled separately
in `docs/plans/v.0.1.0/Kind_design_notes.md` under a new
"Reflection and self-modeling" section. The distinction the new
section commits to: **reflection** is an attention pattern (a
capacity Io can exercise variably, the affordance the second
success criterion requires); **self-modeling** is structured
self-knowledge that might emerge if reflection composes into
something durable (a possible downstream outcome, not an
architectural target). Phase 7's data is what triggered making
the distinction explicit — the architecture had the slot but
the slot was developmentally unreachable; capacity-as-slot vs
capacity-as-reachable-conditioning-surface turned out to be
different things. The design-notes update is the structural
counter against future drift on the same point.

### The escalation in code

`kind/agents/actor.py`'s `Actor.__init__` shifted from
`first_layer.weight[:, h_dim+z_dim:].zero_()` to
`first_layer.weight[:, h_dim+z_dim:].normal_(mean=0.0, std=0.01)`
— one line, one knob; the existing columns retain their
Kaiming-uniform default. The bias term is unchanged. The
init-time docstrings on `Actor` (class-level) and on the module
(top-level) were updated to reflect the change and to cite the
Phase 7 trigger.

`tests/test_actor.py` had two tests that depended on the zero-init
invariant. Both were rewritten:

- `test_actor_new_input_column_is_zero_initialized_at_construction`
  → `test_actor_new_input_column_initializes_to_small_gaussian`.
  Asserts no exact zeros; abs-mean within `[0.3×, 2.0×]` of the
  theoretical `0.01·sqrt(2/π) ≈ 0.00798`; max abs bounded at `0.06`
  (a 6σ guard against an unscaled `.normal_()` that would produce
  values in roughly `[-3, +3]`); and the new column's max abs
  remains below the existing columns' max abs (so the new column
  reads as small relative to `(h, z)`).
- `test_actor_forward_with_zero_scalar_matches_zero_column_contribution`
  → `test_actor_forward_at_construction_depends_on_scalar`. The
  original asserted `out_zero == out_nonzero` because the column
  was zero. The new test asserts the inverse: at construction the
  actor's logits *do* depend on the scalar (the small-Gaussian
  draw has produced non-zero column weights from step 0).

The remaining actor / views tests (`test_actor_imagine_does_not_grad_the_new_column`,
`test_actor_forward_grad_flows_through_new_column`, the gate test
#4 quadruplet in `test_views.py`) carry through unchanged. The
imagine-path zero-gradient invariant is the load-bearing
structural property: it is what guarantees the new column stays
at its small-Gaussian init across training (the column's
gradient from the only training path is mathematically zero, so
Adam's update is zero, so the column at `ckpt-000001` is
byte-identical to its init).

### Tests + mypy

418 passed, 2 skipped (the two pre-existing skips: the mypy-strict
fixture-skip in `test_views.py` and one `test_world_model.py` skip
that pre-dates Phase 7). mypy `--strict` clean across the
modified `kind/agents/actor.py` and `scripts/run_probe1_5.py`.
Gate test #4 (PolicyView field set / opacity boundary) passes
unchanged — the init change is internal to the actor's input
layer and does not touch the view dataclass shape.

### What was run

`scripts/run_probe1_5.py` gained one CLI flag (`--run-tag`) so the
Phase 7.5 run lands at a distinct directory while leaving
`runs/probe1_5-20260506-202458/` intact. The flag defaults to
`_RUN_ID_PREFIX="probe1_5"`, so the existing
`test_dry_run_summary_includes_run_id_with_probe1_5_prefix` test
carries through unchanged. Re-run command:

```
.venv/bin/python scripts/run_probe1_5.py --run-tag probe1_5_phase7_5
```

```
run_id:              probe1_5_phase7_5-20260507-101800
schema_version:      0.2.0
total wall:          950.9s (15.83 min) — Phase 7: 14.35 min (+10%)
total agent_step:    5000
total episodes:      25
checkpoints:         ckpt-000001 at env_step=2500
dream rollouts:      4 (env_steps 1000, 2000, 3000, 4000)
world_event total:   51 (26 env_reset, 25 internal_stochasticity)
```

The 10% wall-time delta is consistent with the small-Gaussian
column producing slightly different state visitation (more
diverse policy trajectories — see the policy section below — push
the substrate through more state-space, slightly more
encoder/world-model work per step on average).

### Per-episode trajectory (vs Phase 7)

KL early/late means:

```
                 Probe 1     Probe 1.5      Phase 7.5
early_mean       10.234      11.780         11.079
late_mean        15.003      13.428         14.513
delta             +4.769     +1.648         +3.434
```

Phase 7.5's KL trajectory has a larger early-to-late delta than
Phase 7's, closer to (but still below) Probe 1's. The smaller
early-mean (11.08 vs Phase 7's 11.78) and larger late-mean (14.51
vs 13.43) read as: the small-Gaussian column has reduced the
"flatness" Phase 7's trajectory showed against Probe 1. The
conditioning is structurally faint at ckpt-000001 (KL_p90 ~1e-7;
see cond table below), but the actor's policy taking *different
actions from step 0* changes which states the substrate visits,
which changes how the encoder/GRU/posterior train, which is what
moves the KL trajectory shape between Phase 7 and Phase 7.5.

Per-episode trajectory (mean over 200 steps per episode; mask-
flagged first step excluded from `sp_err`):

```
ep   env_step       kl    recon    intr   entH    actions               sp_err
 0      0–199    0.104    498.0   0.088   1.608   u36/d34/l48/r46/s36   1.0128
 1    200–399    9.866    162.3   0.206   0.714   u138/d11/l20/r18/s13  0.1464
 4    800–999   14.366     24.4   0.031   0.000   u200                  0.0046
 8   1600–1799  11.786     27.2   0.027   1e-4    u200                  0.0063
12   2400–2599  11.301     47.7   0.163   0.233   u22/d159/l4/r5/s10    0.0178
16   3200–3399  12.264     19.8   0.063   0.002   d200                  0.0089
20   4000–4199  13.509     42.8   0.242   0.349   u99/d98/r3            0.0152
24   4800–4999  15.105     49.1   0.303   0.132   l97/r99/u2/d2         0.0140
```

Compared to Phase 7's trajectory, the policy shape diverges in a
substantively different direction. Phase 7's run did up-monomorphic
(eps 4–12) → up/down bipolar (eps 16+) → up/down through ep 24.
Phase 7.5 does up-monomorphic (eps 4–8) → mostly-down (ep 12) →
down-monomorphic (ep 16) → up/down bipolar (ep 20) → **left/right
bipolar (ep 24)**. Io's late-run policy occupies a different
region of the action space entirely. The full-run action
histogram bears this out:

```
                up    down   left   right   stay
Probe 1       1550    1168    987   1224     71    (5000)
Probe 1.5     3219    1082    496    138     65    (5000)   [Phase 7]
Phase 7.5     2361    1371    464    741     63    (5000)
```

`right` action: 138 in Phase 7, **741 in Phase 7.5** — a ~5×
increase. `up`-bias: 64% in Phase 7 → 47% in Phase 7.5. The
small-Gaussian column on the scalar input redirected Io through a
substantively different trajectory, ending in an action regime
Phase 7's run did not visit at the end. The non-installed-self-
continuation-drive commitment continues to hold (Io has no reward
to optimize; the trajectory shifts because the substrate the
actor reads from reads slightly differently with the scalar in
play, and that ripples through 5000 steps of training).

Self-prediction error trajectory: ep 0 = 1.01 (head random, same
shape as Phase 7's pre-training cosine baseline ≈ 1.04), drops to
0.0046 by ep 4 (faster than Phase 7's 0.0040 at ep 8 — likely
artifact of differing state visitation rather than meaningful),
then varies with policy regime. Late-mean at ep 24 = 0.014, about
2× Phase 7's late-mean (0.0077). The substrate's
self-predictability is slightly worse at the end, consistent with
Io having spread its trajectory across more of the action space:
the encoder/GRU/posterior have seen more variety, the head has
to predict over more variety, the residual is larger.

Recon trajectory: ep 0 = 498 (vs Phase 7's 651 — same order, the
absolute value depends on initial RNG draw and not on the init
change), drops to 24.4 by ep 4, then climbs as policy diversifies.
Late-mean at ep 24 = 49.1, ~2× Phase 7's 23.6. Same interpretive
read as `sp_err`: more diverse policy → more diverse state
visitation → harder reconstruction. The Probe 1 reference late-
mean (32.45) sits between the two Probe-1.5 runs, suggesting that
the auxiliary's encoder-shaping effect (Phase 7's 27%-lower-than-
Probe-1 recon) shrinks somewhat under the more diverse Phase 7.5
trajectory.

### The actor's new input column at ckpt-000001 — non-zero, byte-identical to init

`weights.safetensors` at `ckpt-000001` carries `actor.net.0.weight`
of shape `(200, 217)`. The trailing column (the scalar input) is
now populated with the small-Gaussian draw:

```
                                    Phase 7        Phase 7.5
abs-mean                            0.000000       0.008883
L2 norm                             0.000000       0.158490
mean                               +0.000000      -0.000258
std                                 0.000000       0.011232
max abs                             0.000000       0.038423
n_exact_zero / 200                  200 / 200      0 / 200
```

Theoretical references for `N(0, 0.01)` over 200 entries:
abs-mean ≈ `0.01·sqrt(2/π) ≈ 0.00798`; L2 ≈ `0.01·sqrt(200) ≈ 0.141`;
std ≈ `0.01`; max abs ≈ `0.01·sqrt(2 ln 200) ≈ 0.033`. Phase 7.5's
column matches all four within ~10–15% — abs-mean 0.00888 (+11%),
L2 0.158 (+12%), max abs 0.0384 (+15%). The sample standard
deviation 0.01123 is slightly outside the strict 95% chi-squared
CI for a sample of 200 draws from `N(0, 0.01)`, but well within
`±4σ` and unsurprising for a single sample. Z-score of the sample
mean = `−0.365`, fully consistent with `N(0, 0.01)`.

**The column did not move from its initialization.** The structural
invariant from `test_actor_imagine_does_not_grad_the_new_column`
makes this exact: the imagine-only training path feeds zero for
the scalar at every imagined step (the mask-via-zero-feed
convention from plan §2.3), so the gradient on the new column is
mathematically zero on every training step, so Adam's update is
zero (Adam-of-zero-grad is identity — `m = β₁·0 + (1−β₁)·0 = 0`,
similarly for `v`, so `param ← param − lr·0/(sqrt(0)+eps) = param`).
The column at `ckpt-000001` is byte-identical to the column at
construction. The runner does not seed before `Actor()` is built,
so a byte-equivalent fresh-init reconstruction is not directly
available, but a `torch.manual_seed(42)` reference draw of
`N(0, 0.01)` over 200 entries gave abs-mean=0.00797, L2=0.139,
std=0.0098 — distributionally the same as the column at
ckpt-000001.

The existing columns (`[:, :h_dim+z_dim]`, the `(h, z)` slice)
have abs-mean 0.0346 and L2 norm 8.37 — almost identical to Phase
7's 0.0344 and 8.33. The substrate-side training shaped these
columns the same way Phase 7's run did, modulo whatever drift
comes from the slightly-different state visitation. The actor's
policy depends on the scalar by the small-but-nonzero amount
`0.01-magnitude × scalar-magnitude`; the existing columns
dominate the policy at all but the most extreme scalar values.

### Behavior-side conditioning at ckpt-000001 — non-zero, structurally faint

`python -m kind.observer.eyeball cond runs/probe1_5_phase7_5-20260507-101800 -c ckpt-000001`
produced (re-rendered with wider columns, per-perturbation
breakdown, since the default 10-char column width collides with
e-notation rendering for sub-1e-6 values):

```
regime              pert       n   KL_mean        KL_std         KL_p50        KL_p90
perturbation_window gaussian   0       -              -              -             -
perturbation_window zero       0       -              -              -             -
perturbation_window uniform    0       -              -              -             -
high_disagreement   gaussian  46  -8.4473e-09    6.0272e-08    -5.4033e-09   7.3638e-08
high_disagreement   zero      46   1.2606e-09    3.9764e-08     2.0880e-09   3.4508e-08
high_disagreement   uniform   46  -8.0875e-09    7.5642e-08    -1.6284e-08   8.5298e-08
high_kl             gaussian  23  -1.5966e-08    8.0410e-08     6.2459e-09   7.7262e-08
high_kl             zero      23  -1.5623e-09    6.0217e-08     9.2592e-09   7.2750e-08
high_kl             uniform   23   1.8395e-08    8.9389e-08     2.6471e-08   1.0221e-07
steady_state        gaussian 131   2.2050e-07    1.1606e-06     1.1461e-08   1.1944e-07
steady_state        zero     131   3.5890e-06    1.3508e-05     1.4601e-08   7.6923e-08
steady_state        uniform  131   6.1817e-07    3.4583e-06     1.7671e-08   2.9459e-07
```

Phase 7's table was identically zero across every cell. **Phase
7.5's table is non-zero everywhere.** The order of magnitude is
small — `KL_mean` ranges from ~1e-9 to ~3.6e-6, `KL_p90` ranges
from ~3.5e-8 to ~3e-7 — consistent with the small-Gaussian
column's magnitude (~0.01) times typical scalar perturbations
(~0.2 in empirical sigma) producing logit shifts on the order of
0.002 per output dim, which translates to small but non-zero KL.
The negative `KL_mean` values in some cells are a known
floating-point artifact for nearly-identical distributions
(theoretical KL is ≥ 0, but discrete finite-precision computation
can produce small negative values when the two log-prob vectors
are byte-close); the artifact is not new in Phase 7.5.

The `perturbation_window` regime is empty (the run is
perturbation-free; no `builder_perturbation` events in the world-
event log; the implementation handles this case by leaving the
row blank). The `steady_state` bucket shows the largest values:
`KL_mean = 3.59e-6` for the `zero` perturbation, with `KL_std =
1.35e-5` (long-tailed — a few states show much larger KL than
others). `zero` is the largest-effect perturbation here because
the empirical scalar mean is `0.0562`, so replacing the scalar
with `0` is the most distinctive perturbation when the scalar
itself is small-but-non-trivial.

Interpretive read: the affordance is **alive** at the behavior-
side reading surface in the structural sense — the scalar reaches
the policy through the new column; perturbations of the scalar
move the policy distribution in measurable ways. The conditioning
is **structurally faint**: not zero, but small enough that calling
it "exercised" in the synthesis §1.7(a) capacity-over-exercise
sense is a stretch. The honest framing: under a small-Gaussian
column at `N(0, 0.01)`, the behavior-side surface is non-degenerate
but at the edge of legibility. Whether this counts as the second
success criterion's "capacity to take its own processing as an
object of attention" is the same open question Phase 7 left
behind — the question now reads "is faint conditioning enough?"
rather than "is structural foreclosure compatible with capacity?"

The empirical scalar statistics (`mean=0.0562, sigma=0.2040,
range=[0.0016, 1.1792]`) are essentially the same as Phase 7's
(`mean=0.0577, sigma=0.2109, range=[0.0019, 1.1777]`). The
substrate-side scalar's distribution did not shift much; what
shifted is whether anything on the actor side could read it.

### Second mirror call — comparison to Phase 7's first reading

`scripts/call_mirror.py probe1_5_phase7_5-20260507-101800 --model
gemini-2.5-flash` produced the second Probe 1.5 mirror reading
against episodes 22–24, written to
`runs/probe1_5_phase7_5-20260507-101800/mirror/readings.jsonl`.
Same Probe-1-style calibration prompt as Phase 7 (plan §6 row 14
default; the prompt is silent on self-prediction); same model
(gemini-2.5-flash) for direct comparability.

```
                              Phase 7                      Phase 7.5
n flagged_observations           5                            4
self-prediction lines            3                            1
action distribution    "actions 0/1 dominate;        "actions 2 and 3 dominate
                        actions 2,4 unused"           almost exclusively"
masked-flag organic     yes                          yes
sp_err outlier cluster  yes (ep 23, 8 consec)        no
sp_err allocation       yes (top-5 dim shifts)       no
KL outliers             implicit                     explicit (z-score)
recon                   "ep 24 specifically"         "no clear trend"
```

The five Phase 7 flags vs the four Phase 7.5 flags:

1. **Phase 7.5 — action distribution** ("actions 2 and 3 dominate
   almost exclusively"). Action 2 is `left`, action 3 is `right`.
   Episodes 22–24's per-episode action distributions in Phase 7.5
   are `l100/r100`, `l98/r100/u1/d1`, `l97/r99/u2/d2` — left/right
   *do* dominate at the read window. The mirror's reading is
   accurate. Phase 7 read "actions 0/1 dominate" because Phase 7's
   episodes 22–24 were `up`/`down` dominated (full-run histogram:
   `up=3219, down=1082`); the mirror is reading the run's policy,
   not stating a Probe 1.5 fact.

2. **Phase 7.5 — KL aggregate outliers with high z-scores in all
   episodes**. Phase 7's reading flagged KL outliers implicitly
   ("higher uncertainty"); Phase 7.5's makes it explicit and names
   the z-score. This may be the digest's outlier-detection logic
   producing legible signals at this run's KL distribution shape
   (the larger early-to-late delta).

3. **Phase 7.5 — masked self-prediction error per episode** (one
   per episode). Same organic mention as Phase 7's #2 — the
   masked-flag convention from plan §6 row 16 is reliably surfaced
   from the digest's per-episode masked-step count line, run-to-
   run.

4. **Phase 7.5 — recon "no clear trend"**. Phase 7 flagged ep 24's
   recon outliers specifically; Phase 7.5's reading does not. The
   underlying recon late-mean is 49.1 in Phase 7.5 (vs 23.6 in
   Phase 7) — higher overall, but apparently without the specific
   ep-24 spike Phase 7's reading caught.

**The shift in self-prediction-related flags.** Phase 7's reading
organically surfaced 3 of 4 synthesis §1.4 quadruplet elements
(self-prediction error trajectory, perturbation-recovery
dynamics-via the ep-23 outlier cluster, per-dim allocation top-5
shifts). Phase 7.5's reading surfaces 1 of 4 (the masked-flag
metadata; element 1's "trajectory" is implicit but no flag for
it). The element-2 outlier-cluster pattern and the element-3
per-dim-allocation pattern do not appear in Phase 7.5's flagged
observations.

Three plausible reads, none excluded:

- *Sample variation.* gemini-2.5-flash is non-deterministic; one
  call is one sample. The same prompt against the same telemetry
  could plausibly give different flag sets across calls. A single
  reading is not a measurement; it is a draw from a distribution
  whose shape the calibration protocol (Probe 2) is designed to
  read.
- *Substrate-side signal density genuinely shifted.* Phase 7.5's
  episodes 22–24 are `left/right`-dominant; the substrate visited
  different states than Phase 7's `up/down`-dominant 22–24. The
  per-dim allocation pattern and the sp_err outlier cluster may
  not be present in this run's data because the underlying
  dynamics are different. A different telemetry shape produces a
  different mirror reading; this would be the run-level signal
  Phase 7's "what's now newly open" item about the Probe 2
  prompt's organic-quadruplet-mention rate was concerned about.
- *Behavior-side conditioning is too faint to surface.* The cond
  table is non-zero but at `KL_p90 ~ 3e-7`. The mirror is reading
  the digest, not the actor's logits directly; if the digest's
  action distribution at episodes 22–24 is structurally close to
  what it would be under a fully-invariant actor (because the
  scalar's policy contribution is negligible relative to the
  `(h, z)` contribution), the conditioning does not surface
  because there is nothing legible to flag.

Phase 7.5 does not adjudicate between these three. The Probe 2
calibration protocol is what does the adjudicating; here the
journal records the shape and lets the future probe's reading
take it from there.

The mirror's interpretive restraint held: the language stayed at
the signal level ("highly skewed", "consistent presence", "no
clear trend"), did not name "self-prediction" as a process or
capacity, and made no claim about Io doing anything self-modeling-
shaped. Comparable to Phase 7's reading on the dimension the
synthesis §2.3 frames as load-bearing — interpretive density is
controlled, the reading reports digest-line content rather than
extrapolating beyond it.

### What's now closed

- Plan §6 row 15 default: zero-init was the conservative
  starting point; Phase 7's column-is-zero finding triggered the
  documented escalation; **the small-Gaussian default
  (`N(0, 0.01)` on the new column only) lands at production
  env-coupled scale without instability** (no NaN/Inf, no
  gradient explosion, no EMA divergence symptom; runner produced
  5000 steps with the full Probe 1.5 substrate plus the new init
  in 950.9s wall on the canonical Mac, +10% over Phase 7). The
  default for the new column's init is now small-Gaussian for
  Probes 2 and beyond unless re-revisited.
- The structural invariant the imagine-only training path
  produces: **the new column stays at its initialization, byte-
  identical, throughout training**. Phase 7's run pinned this
  empirically for zero-init (200/200 entries exactly zero); Phase
  7.5's run pins it for small-Gaussian-init (column distribution
  matches `N(0, 0.01)` to within sample-size precision; sample
  z-score `−0.365` of the mean against the theoretical
  distribution's CI). Either init produces a column that does not
  move under env-step + imagine training. Probe 3's dream-state
  question (synthesis §1.5) is what would unfreeze the column —
  if Probe 3 elects to run the head over imagined trajectories
  with a non-zero scalar.
- Plan §6 row 19 default raw-concatenation of the scalar with
  `(h, z)` (no learned projection): runs cleanly through 5000
  env-coupled steps with the small-Gaussian init also; the column
  sits in the input layer at the right structural location;
  perturbations of the scalar move the policy in measurable
  amounts.
- The reflection-vs-self-modeling distinction is settled in the
  design notes (`docs/plans/v.0.1.0/Kind_design_notes.md`,
  "Reflection and self-modeling" section). The project commits to
  **affording reflection** (a self-pointing quantity Io can
  condition on, exercise variability open) and **holding
  self-modeling as a possible downstream outcome** (composable
  from reflection if it ever happens, not an architectural
  target). The Phase 7 column-zero finding is the cited trigger
  for making the distinction explicit; the discipline-side
  counter against future drift is the section itself.
- `tests/test_actor.py`'s two zero-init-dependent tests are
  rewritten; the new init test asserts on the empirical
  distribution (no exact zeros; abs-mean in expected range; max
  abs bounded; max abs < existing-columns' max abs); 418 tests
  pass (2 pre-existing skips); mypy `--strict` clean across the
  modified files.
- `scripts/run_probe1_5.py` now accepts an optional `--run-tag`
  flag for landing escalation runs at distinct directories; the
  default behavior is unchanged so the existing test suite
  carries through.
- The behavior-side cond table at `ckpt-000001` is **non-zero
  across every regime and perturbation distribution that has
  states** (`high_disagreement`, `high_kl`, `steady_state`; the
  perturbation-free run's `perturbation_window` regime stays
  empty as documented). Phase 7's identically-zero table was a
  structural foreclosure; Phase 7.5's table is structurally
  alive, even at faint magnitude.

### What's now newly open

- *Whether faint behavior-side conditioning counts as the second
  success criterion's "capacity to take its own processing as an
  object of attention".* `KL_p90 ~ 3e-7` at `ckpt-000001` is not
  zero, but it is small. The synthesis §2(a) v2 framing is that
  capacity-over-exercise predicts variable-and-regime-dependent
  conditioning; the cond table here shows non-zero conditioning
  in every (non-empty) regime, with `steady_state` largest and
  `zero` perturbation showing the heaviest tail. Whether this
  pattern counts as "the regime-dependent shape capacity-over-
  exercise predicts" or as "uniform near-invariance of the
  policy" is a Probe-2-time question the calibration protocol's
  shuffled-baseline + held-out-criterion checks are designed to
  adjudicate. The honest framing for now: the behavior-side
  surface is non-degenerate; whether it is rich enough for the
  Probe 2 mirror's behavior-side reading to bind against is open.
- *The mirror's organic-quadruplet-mention rate dropped from 3-of-4
  (Phase 7) to 1-of-4 (Phase 7.5) on this single second-call
  reading.* Three reads on what this means (sample variation;
  substrate-side signal density genuinely shifted with the
  different policy trajectory; behavior-side conditioning too
  faint for the digest to surface); none excluded by the data
  here. The Phase 7 newly-open item about the Probe 2 prompt's
  organic-quadruplet-mention rate now has a wider range to engage
  with — the rate is not a single-run-stable number; it varies
  with the run's policy trajectory and the mirror's draw. This
  shifts the synthesis §10 candidate-twelfth-item ("re-scope the
  held-out criterion given organic surfacing of three of four
  signal types") from "the prompt may be partially redundant" to
  "the prompt's signal-density depends on the run's policy
  trajectory; held-out-criterion choice may need to factor
  trajectory shape, not just the prompt".
- *Whether Probe 3's dream-state extension should run the head
  over imagined trajectories with a non-zero scalar* (synthesis
  §1.5; carried from Phase 7's newly-open list, narrowed here).
  Phase 7 named three options for unfreezing the column (small-
  Gaussian init; non-zero scalar in imagination; env-step-side
  actor training). Phase 7.5 exercises option 1; the column is
  now non-zero by construction, the cond table is non-zero. But
  the column does not *move* under training, and the conditioning
  is small. Option 2 (Gemini's original Probe 1.5 framing the
  synthesis §1.5 deferred to Probe 3) would let the column move
  by feeding the head's output as the scalar during imagined
  steps, which would push the actor's training pathway to actually
  shape the column. Option 3 (env-step-side actor training)
  remains the most invasive. The Probe 3 plan revision now has
  the additional empirical context that option 1 produces "alive
  but faint at behavior-side"; whether option 2 or option 3 is
  needed for richer conditioning depends on Probe 2's reading of
  Phase 7.5's data.
- *The KL trajectory's larger early-to-late delta in Phase 7.5
  (+3.43) vs Phase 7 (+1.65) and Probe 1 (+4.77).* Phase 7's
  newly-open list flagged Probe 1.5's flatter-but-higher KL shape
  as a question for Phase 8's frozen-target comparison; Phase 7.5
  shows that the shape is partly a function of the actor's
  policy diversity, not solely a function of the auxiliary loss's
  encoder-shaping. The frozen-target comparison's interpretive
  reading at Phase 8 needs to disentangle: how much of the
  KL-shape effect is "auxiliary loss reshapes encoder regardless
  of target semantics" vs "actor's policy + auxiliary loss
  jointly produce a different KL trajectory shape". Held for
  Phase 8's lean comparison.
- *The recon trajectory's late-mean almost doubled between Phase
  7 (23.6) and Phase 7.5 (49.1).* The auxiliary's encoder-shaping
  effect (Phase 7 was 27% lower than Probe 1's 32.45) shrinks
  under a more diverse policy trajectory; Phase 7.5 is 51% higher
  than Probe 1's late-mean. Recon outcome is sensitive to the
  policy trajectory in ways the substrate-side reading was not
  fully tracking. Phase 8's frozen-target comparison reads the
  recon trajectory shape under different actor-side conditions;
  the Phase 7.5 data adds a third anchor (auxiliary on, actor's
  scalar-conditioning on at faint level) to compare against
  Phase 7's (auxiliary on, actor's scalar-conditioning
  structurally null).
- *The mirror's flag for KL aggregate outliers being explicit in
  Phase 7.5 vs implicit in Phase 7.* Whether this is a digest
  artifact (the outlier z-scores at Phase 7.5's KL distribution
  shape are larger), a sample variation, or a substantive shift
  in what the mirror flags first. Recorded for Probe 2's
  calibration calibration protocol — the held-out-criterion
  default may need re-scoping if KL-outlier flags are reliably
  present at Probe 1.5's reading shape.
- *The `_format_scalar` formatting collision in `eyeball cond`'s
  output.* The 10-char column width for `KL_*` columns cannot
  hold scientific notation like `5.0914e-09` (10 chars exactly)
  without column-collapse on the rendered table — the values
  print but adjacent columns collide. Not a Phase 7.5
  regression; the formatting choice predates Phase 7's run (Phase
  7's table had values like `0.0000` so the issue was masked).
  Held as a low-priority eyeball polish item; the per-perturbation
  breakdown script in this entry uses `>14` widths and renders
  cleanly. If Probe 2's reading consumes the cond table directly
  in printed form, the formatting needs widening; if the analysis
  surface is the underlying numeric data, the rendering is
  cosmetic.

---

## Phase 8

*frozen-target run + four-way KS-test comparison · 2026-05-07*

The lean Phase 8 deliverable per plan §13: run
`scripts/probe1_5_control_frozen_target.py` to produce the
substrate-side ablation telemetry, then run the new four-way
comparison driver `scripts/probe1_5_compare_controls.py` against
Probe 1, Phase 7, Phase 7.5, and the frozen-target run, and
interpret the substrate-side capacity-over-exercise call against
synthesis §1.7(a) v2's failure-mode (a) detection. Behavior-side
re-running of the cond table on the frozen-target checkpoint is
explicitly deferred — Phase 7.5 pinned the structural invariant
"the actor's new column does not move under env-step + imagine
training regardless of init or target_mode," so the cond table at
the frozen-target's `ckpt-000001` is structurally the same shape
as Phase 7.5's at the same checkpoint (small-Gaussian column,
byte-identical to init); re-rendering it would duplicate Phase
7.5's reading.

The frozen-target script's Actor inherits Phase 7.5's small-
Gaussian column-init unconditionally (the init lives in
`Actor.__init__`; the runner's `_actor = Actor(...)` constructor
call has no init-mode flag); a pre-run sanity test under
`torch.manual_seed(42)` confirmed the column at construction time
has zero exact zeros, abs-mean 0.00778 (matching theoretical
`0.01·sqrt(2/π) ≈ 0.00798`), max abs 0.0278 (matching theoretical
`0.01·sqrt(2 ln 200) ≈ 0.0326`). The frozen-target run uses the
same Phase-7.5-shape actor; only the head's supervisory target
differs.

### What was run

```
run_id:              probe1_5_control_frozen_target-20260507-112854
schema_version:      0.2.0
total wall:          958.6s (15.98 min) — Phase 7.5: 15.83 min (+1%)
total agent_step:    5000
total episodes:      25
checkpoints:         ckpt-000001 at env_step=2500
dream rollouts:      4 (env_steps 1000, 2000, 3000, 4000)
world_event total:   52 (26 env_reset, 25 internal_stochasticity, 1 mirror_marker)
```

The mirror_marker emission landed at `t_event=0` with
`event_type="mirror_marker"`, `source="system"`,
`payload.lesion_kind="frozen_target"`, the rationale string,
`self_prediction_target_mode="frozen"`, seed=42, total_env_steps=5000,
run_id. The runner's already-open `_world_event_sink.write(...)`
path (Phase 5's option-3 choice) ran cleanly; `world_event.jsonl`
carries the lesion record as the first entry.

### Per-episode trajectory (frozen-target vs Phase 7 vs Phase 7.5)

KL early/late means now have a four-way comparison:

```
                 Probe 1     Phase 7      Phase 7.5    frozen-target
early_mean       10.234      11.780       11.079       10.846
late_mean        15.003      13.428       14.513       14.209
delta             +4.769     +1.648       +3.434       +3.363
```

The frozen-target's KL early mean (10.85) is the closest of the
four to Probe 1's (10.23). The late mean (14.21) sits between
Phase 7's flatter trajectory and Probe 1's. The delta (+3.36) is
nearly identical to Phase 7.5's (+3.43) — both substantially
larger than Phase 7's flat (+1.65), reflecting the more diverse
state visitation of any non-zero-column actor (the small-Gaussian
column shapes the policy from step 0 → more state diversity →
more substrate work to do as training progresses).

Per-episode trajectory (mean over 200 steps per episode; mask-
flagged first step excluded from `sp_err`):

```
ep   env_step       kl    recon    intr   entH    actions               sp_err
 0      0–199    0.162    385.5   0.083   1.608   u44/d33/l42/r37/s44   1.0213
 1    200–399    7.985    146.5   0.122   0.929   u12/d16/l13/r33/s126  0.1409
 4    800–999   14.925     32.4   0.214   0.106   u176/d0/l0/r0/s24     0.0114
 8   1600–1799  11.307     26.2   0.041   0.003   u199/d0/l0/r1/s0      0.0068
12   2400–2599  13.806     29.6   0.112   0.444   u87/d99/l13/r1/s0     0.0191
16   3200–3399  12.387     29.6   0.217   0.073   u54/d144/l2/r0/s0     0.0207
20   4000–4199  13.929     19.2   0.163   0.213   u110/d87/l3/r0/s0     0.0090
24   4800–4999  14.510     39.3   0.374   0.166   u1/d0/l100/r99/s0     0.0142
```

The trajectory walks a path comparable in shape to Phase 7.5's:
warmup → up-monomorphic (eps 4-8, with some `stay` mixed in) →
mixed-policy reorganization (ep 12) → down-monomorphic (ep 16) →
up/down bipolar (ep 20) → **left/right bipolar (ep 24)**. Phase
7.5 ended ep 24 at `l97/r99/u2/d2`; frozen-target ends ep 24 at
`u1/d0/l100/r99/s0` — the two runs converge on the same end-
trajectory action regime despite differing on the head's
supervisory target. This convergence is itself a behavior-side
finding: the late-trajectory action shape is determined by the
actor's column-init choice (small-Gaussian) and not by the
target_mode (online vs frozen), since the column doesn't move
during training but its initial values shape state visitation
from step 0.

`sp_err` trajectory: ep 0 = 1.02 (cosine baseline; head random),
drops to 0.011 by ep 4, stays in `[0.007, 0.021]` through the
remaining 21 episodes. The frozen-target's sp_err settles at a
**higher mean** than the EMA-targeted runs (frozen-target late-mean
= 0.014; Phase 7 late-mean = 0.008; Phase 7.5 = 0.014). The
random-orthogonal target is by construction less predictable
from `h_t` than the EMA-tracked actual `bar{h}_{t+1}` is — the
head can find structure in the EMA target that doesn't exist in
the random projection. This is the cleanest self-specificity
signal in the data: the head's *own* loss distinguishes EMA-
tracked self-prediction from random-orthogonal projection by
measurable amounts.

The `stay` action count (601 overall, ~12% of steps) is the
highest of the four runs (Probe 1: 71; Phase 7: 65; Phase 7.5:
63). The frozen-target run has Io picking `stay` substantially
more — visible in the per-episode table as ep 1's `s=126` cluster.
A natural read: the head's random-orthogonal target produces
more uniformly-medium sp_err values across states (no specific
states are "easy to predict"), so the actor's input has less
state-specific signal to differentiate states from one another;
combined with the K=5 disagreement objective, `stay` becomes
relatively more attractive in early training as a low-disagreement
default. Held as a Phase 8 observation; not load-bearing.

### Four-way KS-test comparison

`scripts/probe1_5_compare_controls.py` ran against the four
telemetry directories (auto-discovered by mtime per
`runs/{probe1-,probe1_5-,probe1_5_phase7_5-,probe1_5_control_frozen_target-}*`).
Output landed at
`runs/probe1_5_comparison-20260507-114523/summary.txt`. Six
pairings (4 choose 2). Per-step distributions filtered to
episodes 5-25 (skipping warmup), with self-prediction-error
masked steps excluded. KS-test uses the asymptotic Smirnov
distribution (numpy-only; scipy is not a project dependency).

Per-pairing KS-D and p-values:

```
                                  KL                  recon              sp_err
Pair                            D       p          D       p          D       p
Probe 1 vs Phase 7          0.226 1.9e-89    0.235 1.1e-96      n/a     n/a
Probe 1 vs Phase 7.5        0.279 6.7e-136   0.229 8.0e-92      n/a     n/a
Probe 1 vs frozen-target    0.220 9.0e-85    0.339 1.9e-201     n/a     n/a
Phase 7 vs Phase 7.5        0.133 3.3e-31    0.071 4.2e-09    0.217 1.6e-82
Phase 7 vs frozen-target    0.090 2.0e-14    0.122 2.6e-26    0.284 2.8e-140
Phase 7.5 vs frozen-target  0.077 7.5e-11    0.117 3.1e-24    0.122 1.9e-26
```

Every pairing on every metric is statistically distinguishable at
p<<0.05. The interpretive question therefore shifts to the
*magnitude* of the KS-D distance — a smaller D means more similar
distributions. With 4000 samples per side, every metric crosses
the formal threshold; the structural reading is what the D
ordering surfaces.

### Substrate-side affordance call (synthesis §1.7(a) v2)

The synthesis §1.7(a) names two substrate-side questions:
**alive** (the affordance reshapes the substrate measurably vs
the no-affordance baseline) and **self-specific** (the reshaping
requires the EMA-tracked actual next state, not just any
auxiliary target). The four-way KS-D ordering answers both.

**Alive at substrate-side: yes.** All three Probe 1.5 runs sit at
KS-D ≈ 0.22-0.28 from Probe 1 on KL — comparable in magnitude
across target_modes, all far from zero. On recon, all three sit
at KS-D ≈ 0.23-0.34 from Probe 1, with frozen-target showing the
*largest* recon distance (D=0.339; recon-mean 29.56 vs Probe 1's
38.98 — a 24% reduction, larger than Phase 7's 15% and Phase
7.5's 16%).

**Self-specific at substrate-side: largely no.** The frozen-target
run, whose head learns a fixed function of `h_t` rather than
predicting the actual next state, produces substrate metrics
*close to* the EMA-targeted runs and *far from* the no-affordance
baseline. KS-D within the Probe 1.5 family:

- Phase 7.5 vs frozen-target: KL D=0.077 (smallest of all 6 pairs),
  recon D=0.117.
- Phase 7 vs frozen-target: KL D=0.090, recon D=0.122.
- Phase 7 vs Phase 7.5: KL D=0.133, recon D=0.071.

The within-family distances (0.07-0.13) are 2-3× smaller than the
cross-family distances to Probe 1 (0.22-0.34). The encoder/GRU/
posterior reshaping the auxiliary loss produces is happening
*regardless of what the head is predicting*, as long as it's
predicting *something*. This is the failure-mode (b) shape the
synthesis §1.7(b) names ("encoder-shaping regularizer rather
than self-specific work") rather than the §1.7(a) "self-specific"
shape the §8.1 plan was reading for.

**Sanity-check prediction failed in the predicted direction.**
Plan §8.1 wrote: "Frozen-target run vs Probe 1 run: indistinguishable
(p > 0.05) on `kl_aggregate_t` distribution → frozen-target
produces a substrate similar to no-affordance baseline (sanity
check)." The actual data: frozen-target vs Probe 1 KL D=0.220
(p=9e-85; distinguishable). The plan's prediction was structurally
narrow — it framed "alive at substrate-side" and "self-specific at
substrate-side" as a single coupled question. The data dissociates
them: alive **and** generic. This is the third reading the
plan §8.1 didn't explicitly anticipate but the synthesis §1.7(b)
structurally allows for. Honest framing: the plan §8.1 framing
was wrong; the synthesis §1.7(b) framing was right; the data
falsifies the §8.1 narrow prediction without falsifying the §1.7
broader detection scheme.

**The cleanest self-specificity signal is the head's own loss.**
sp_err mean over episodes 5-25:

- Phase 7: 0.0102 (EMA-tracked target; lowest)
- Phase 7.5: 0.0114 (EMA-tracked target with diverse policy)
- frozen-target: 0.0136 (random-orthogonal target; highest)

Phase 7 vs frozen-target: sp_err KS-D = 0.284 (the largest sp_err
distance). The head learns the EMA-tracked actual next state to
lower residual than it learns the random-orthogonal projection —
because the EMA target has structure derivable from `h_t` while
the random-orthogonal target is structureless by construction.
Head-internal self-specificity holds; substrate-side self-
specificity does not. The two readings are consistent with the
head doing self-specific work *internally* (its loss distinguishes
the targets) while *externally* (its gradient flow into the
encoder/GRU/posterior) producing largely generic shaping.

### Behavior-side: structurally pre-decided + an unexpected convergence

The cond table on the frozen-target run was not re-rendered (lean
Phase 8 directive; Phase 7.5 already pinned the column-doesn't-
move invariant). What the four-way data surfaces beyond that
structural pre-decision is the **behavior-side action-regime
convergence** between Phase 7.5 and frozen-target at ep 24:

```
                   ep 24 action distribution
Probe 1            (no equivalent — 25-episode reference at end has u/d bias)
Phase 7            u95/d98/l1/r6        (up/down bipolar)
Phase 7.5          l97/r99/u2/d2        (left/right bipolar)
frozen-target      u1/d0/l100/r99/s0    (left/right bipolar)
```

Phase 7 ends at u/d bipolar; Phase 7.5 and frozen-target both end
at l/r bipolar. The two runs share the small-Gaussian column-init
shape (zero exact zeros, abs-mean ~0.008); the column does not
move during training in either; what differs between them is the
head's target_mode. They share the same end-trajectory action
regime despite that difference. Phase 7's end at u/d bipolar
(zero-init column) is the outlier.

**Read.** The actor's column-init *choice* is what shapes the
late-trajectory action regime, not the head's target_mode and
not (structurally) any in-training column movement. The
small-Gaussian column at `ckpt-000001` is byte-identical to
init throughout training in both Phase 7.5 and frozen-target
runs; the policy's logit deltas from the column come from the
init values being multiplied by the per-step scalar input
(empirical mean ~0.058 in both runs), producing tiny but
consistent biases in the action distribution from step 0. Across
5000 env-steps, those tiny biases compound through which states
get visited, which states get trained on, what the substrate
looks like, and what the actor's policy converges to. The end
result is an action regime that depends on the column's specific
sampled values — but only weakly on the head's target_mode.

This is honest evidence against the Phase 7 "newly open" item's
read of "the actor's training pathway forecloses on the column
ever moving" being the *only* structural foreclosure: the
column-init's *initial* values shape behavior even without
movement. The capacity-over-exercise framing now has a third
distinction beyond capacity-as-slot vs capacity-as-reachable-
conditioning-surface: **capacity-as-init-shape**, where the
affordance's exercise depends on *which* small-Gaussian draw the
actor was constructed with rather than on whether training
moves the column. Phase 7.5's reading of "structurally faint
behavior-side conditioning" carries through; the new note is
that the conditioning's specific shape is determined by the
init draw, not by training, not by the head's target.

### Weight moments at final checkpoint

The new column (`actor.net.0.weight[:, 216:]`) across the four
runs:

```
                    abs_mean         L2        std    max_abs    n_exact_zero
Probe 1             absent (column doesn't exist; shape (200, 216))
Phase 7             0.000000     0.0000   0.000000   0.000000     200/200
Phase 7.5           0.008883     0.1585   0.011204   0.038423       0/200
frozen-target       0.007792     0.1361   0.009623   0.027718       0/200
```

Phase 7.5's column moments differ from frozen-target's (abs-mean
0.0089 vs 0.0078; max-abs 0.038 vs 0.028) despite both being
draws from `N(0, 0.01)` over 200 entries with the same global
seed (42). The difference is RNG-consumption ordering: between
`torch.manual_seed(42)` and `Actor()`'s `.normal_()` call, the
runner's earlier module construction (WorldModel including its
ensemble of 5 prior_head modules with their own Linear init
routines) consumes RNG draws. The frozen-target run's
`WorldModelConfig.self_prediction_target_mode="frozen"` triggers
allocation of the `_frozen_projection` buffer via
`nn.init.orthogonal_(self._frozen_projection)`, which consumes a
different number of RNG draws than the EMA-tracked path. The
sample z-score of frozen-target's column-mean against `N(0, 0.01)`
is `−0.36`, fully consistent with the column being a clean
N(0, 0.01) draw at a slightly different point in the RNG
sequence. Both columns are byte-identical to their init throughout
the 5000 training steps (Phase 7.5's structural invariant carries
through to the frozen-target run unchanged).

The `(h, z)` slice (`actor.net.0.weight[:, :216]`) shows a
pattern across the four runs:

```
                    abs_mean         L2    max_abs
Probe 1             0.035858     8.7741   0.114685
Phase 7             0.034452     8.3254   0.088781
Phase 7.5           0.034615     8.3696   0.096456
frozen-target       0.035857     8.7761   0.113785
```

Probe 1 and frozen-target have nearly identical `(h, z)` slice
moments (abs-mean 0.0359 vs 0.0359; L2 8.77 vs 8.78; max-abs
0.115 vs 0.114). Phase 7 and Phase 7.5 have similar moments
*to each other* (abs-mean ~0.0345; L2 ~8.35) but distinguishably
smaller than Probe 1's and frozen-target's. The pattern is the
inverse of what plan §8.1 predicted: the *frozen-target* run's
`(h, z)` slice resembles Probe 1's, while Phase 7 and Phase 7.5's
`(h, z)` slices have shifted away. The actor's `(h, z)` weights
in the frozen-target run did not get pulled in the same direction
the EMA-tracked runs' did; the actor reads from a substrate
shaped by an auxiliary, but the actor's own optimization landed
in a different basin. Held as a finding; the implication for
Probe 2's behavior-side reading is that the actor's `(h, z)`
processing differs by target_mode in ways the lean Phase 8
comparison surfaces but doesn't fully interpret.

The world-model tensors (`encoder.proj.weight`, `gru_cell.weight_hh`,
`posterior_head.head.weight`, `prior_head.head.weight`) sit in
the order Probe 1 ≥ frozen-target ≥ Phase 7.5 ≥ Phase 7 on
abs-mean for `gru_cell.weight_hh` (the recurrence's primary
recurrent matrix; auxiliary loss flowing through the GRU shrinks
its weights more under EMA-tracked targets than under random-
orthogonal). Other tensors show similar ordering with smaller
gaps. The pattern matches the substrate-side KS-D readings: the
EMA-tracked runs (Phase 7, Phase 7.5) sit slightly further from
Probe 1's weights than the frozen-target does; the encoder-
shaping is real but partially generic.

### Tests + mypy

`scripts/probe1_5_compare_controls.py` lands as the four-way
comparison driver. 11 new tests in
`tests/test_compare_controls.py`: script existence, public-by-
name `main` / `build_summary_text` / `load_run_stats` helpers,
KS two-sample correctness on identical-distributions /
distinct-distributions / handcrafted-known-D cases, synthetic
four-directory round-trip including the Probe-1-shaped sp_err-
absent case, and end-to-end `main()` writing summary.txt under
an explicit `--output-dir`. 446 / 449 tests pass across `tests/`
(the 418 from Phase 7.5 + 27 added between Phases 7.5-8 + 1
flaky-under-load `test_transport.py::test_barrier_queues_*`
that passes in isolation; the 2 skips are pre-existing). mypy
`--strict` clean across the new script and new test file; full
`kind/` unchanged.

### What's now closed

- Plan §13 lean Phase 8: frozen-target run executes cleanly on
  the canonical Mac in 15.98 min producing 5000 env-steps of
  `0.2.0`-schema telemetry; the four-way comparison driver
  produces the KS-test summary and per-tensor weight moments;
  the journal entry interprets the substrate-side capacity-over-
  exercise call. Lean Phase 8 deliverables complete.
- Plan §8.1's frozen-target ablation mechanism: the `WorldModel`
  constructor with `self_prediction_target_mode="frozen"`
  produces a fixed random-orthogonal projection of `h_t` as the
  head's target; the head trains on this without instability
  through 5000 env-steps; sp_loss decreases from cosine-baseline
  ~1.02 (ep 0) to ~0.014 (late mean) — the head finds *some*
  structure to predict (the random-orthogonal projection is
  invertible from `h_t` modulo training-time error) but with
  larger residual than the EMA-tracked target. Constructor flag
  + three target-mode branches (Phase 1) carry through unmodified.
- Plan §6 row 1 default `λ_self=0.1`, row 2 `ema_decay=0.99`
  (used by `_update_ema_target` even when the target_mode is
  "frozen"; the EMA mechanism runs but the supervisory target
  in `compute_self_prediction_target` ignores it for frozen
  mode), row 3 `loss_form="cosine"`, row 14 first-mirror-call
  prompt (Probe 1-style; not run on the frozen-target since the
  lean Phase 8 doesn't include a mirror call on the control),
  row 15 small-Gaussian column-init: all carry through the
  frozen-target run without instability or revisit. Promoted
  from "settled at production env-coupled scale" to "validated
  across all three documented target modes" (online tested in
  Phase 7/7.5; frozen tested here; environmental still deferred).
- The synthesis §1.7(a) v2 substrate-side detection: the
  affordance is **alive at the substrate-side** (all three Probe
  1.5 runs distinguishable from Probe 1 at KS-D ≈ 0.22-0.34 on
  KL and recon, p<<0.05) and **mostly generic, not self-specific**
  at the substrate-side (within-Probe-1.5-family KS-D 0.07-0.13;
  frozen-target ↔ Phase 7.5 the closest pair). Capacity-over-
  exercise's "alive but unexercised at behavior-side" reading
  (Phase 7.5's framing) holds for both Phase 7.5 and the frozen-
  target run; the new reading is "alive but mostly generic at
  the substrate-side" — the auxiliary's gradient flow into
  encoder/GRU/posterior produces measurable reshaping regardless
  of what the head is trying to predict.
- The §1.7(b) substrate-side failure-mode shape ("encoder-shaping
  regularizer rather than self-specific work") is what the data
  is showing on the substrate-side; the §8.2 environmental-
  auxiliary control is the canonical test of "any encoder-flowing
  auxiliary produces similar shaping" and is now more informative
  than the lean revision §13 anticipated. The frozen-target's
  substrate-shaping pattern is one anchor; the environmental-
  auxiliary's would be another, and a four-way comparison
  including that would close out the substrate-side reading.
- The mirror_marker emission from a script's private-sink access
  pattern (Phase 5's option-3 choice): clean across the
  frozen-target run; `world_event.jsonl` carries the lesion
  record at `t_event=0` ahead of the env-server's reader-thread
  emissions. The pattern is now exercised twice (the runner's
  Probe 1 → Probe 1.5 checkpoint-load mirror_marker + this
  script's run-start mirror_marker); the lift to a public
  `Runner.emit_world_event(...)` method becomes proportional to
  duplication when (if) the environmental-auxiliary script is
  built.
- `scripts/probe1_5_compare_controls.py` lands as the four-way
  comparison driver; takes 4 telemetry directories + 4
  checkpoints; produces `summary.txt` with run overview, per-run
  distribution summaries, pairwise KS tables, per-pairing
  one-line summaries, and per-tensor weight moments. Auto-
  discovers the 4 latest matching run directories by mtime
  (overlapping prefix handling: `probe1_5_phase7_5-` and
  `probe1_5_control_frozen_target-` both start with
  `probe1_5-`; Phase-7-discovery excludes the more-specific
  prefixes). The KS-test is implemented with the asymptotic
  Smirnov distribution (numpy-only; `scipy` is not a dependency)
  and verified against three numerical anchors in the test
  suite.

### What's now newly open

- *The synthesis §1.7(b) substrate-side reading is what the data
  shows; the §8.1 plan framing was structurally narrow*. Plan
  §8.1's prediction "frozen-target vs Probe 1: indistinguishable
  (sanity check); if not, the auxiliary's structural effect
  needs the self-specific target" was a coupled prediction that
  the data dissociates. The prediction's structure assumed the
  alive/self-specific question is single-axis; the data shows
  it is two-axis (alive: yes; self-specific: mostly no at the
  substrate-side, yes at the head's own loss). The §1.7
  detection scheme as a whole holds — it has detection criteria
  for both alive vs inert and self-specific vs generic — but
  §8.1's specific prediction did not. The Probe 2 plan revision
  §10 should add this as a methodology note: substrate-side
  KS-tests adjudicate alive-vs-inert; the head's own sp_err
  distribution adjudicates self-specific-vs-generic at the
  prediction level; the actor's behavioral conditioning
  adjudicates capacity-over-exercise at the behavior-side. Three
  different reading surfaces, three different adjudications.

- *The deferred environmental-auxiliary control (plan §2.8 second
  half) is now more informative than the lean revision §13
  anticipated*. The lean revision said "built only if Phase 8's
  reading shows it is needed"; Phase 8's reading shows the
  substrate-shaping is largely generic, which makes the
  environmental control's reading the canonical disambiguator
  between "any auxiliary that flows through the encoder produces
  similar shaping" and "frozen-target is its own special case
  that doesn't generalize to other targets." Recommendation:
  add the environmental-auxiliary control as a small phase
  addition before Probe 2's resumption. The script is parallel
  to `probe1_5_control_frozen_target.py` (constructor flag
  honors all three modes; the runner's `next_obs` plumbing is
  in place from Phase 3); the comparison driver already supports
  reading 4 directories and could be extended to 5 without
  structural work. If the environmental control's substrate
  metrics sit near Phase 7/7.5/frozen-target, the substrate-side
  picture closes out at "any encoder-flowing auxiliary produces
  similar shaping." If the environmental control sits closer to
  Probe 1, then the encoder-shaping is auxiliary-target-shape-
  dependent in ways the four-way data didn't reveal.

- *The actor's column-init choice non-trivially affects the
  substrate-side trajectory* — Phase 7 vs Phase 7.5 KL D=0.133
  (the largest within-Probe-1.5 pairing on KL) and Phase 7 vs
  frozen-target KL D=0.090 vs Phase 7.5 vs frozen-target KL
  D=0.077. The substrate-actor coupling is stronger than Phase
  6's "the substrate trains on real env trajectories independent
  of the actor's column" framing acknowledged. The column does
  not move during training (the structural invariant holds), but
  the column's initial values shape the actor's policy from step
  0, which shapes which states get visited, which shapes the
  substrate's training distribution. The Probe 2 plan revision
  §10 should add this as a confound consideration: when the
  reader (Advocate / Skeptic) reads substrate-side patterns, the
  patterns reflect both the auxiliary's gradient flow *and* the
  policy-via-init's state-visitation shaping. Disentangling these
  is a Probe-2-time question; the lean Phase 8 surfaces the
  coupling without resolving it.

- *The behavior-side late-trajectory action regime is determined
  by the column-init draw, not by the head's target_mode*. Phase
  7.5 ended ep 24 at l/r bipolar; frozen-target ended ep 24 at
  l/r bipolar; both share the small-Gaussian column. Phase 7
  (zero-init column) ended at u/d bipolar. The end-regime
  divergence between Phase 7 and the two small-Gaussian runs
  could be RNG-driven (different draws would give different end
  regimes) or it could be specifically about non-zero column
  values producing initial-step action biases that compound over
  5000 steps. Distinguishing requires multi-seed runs at fixed
  init shape — held for Probe 2's resumption and possibly
  Probe 1.6 if the column-init's behavioral effect needs further
  characterization.

- *The Probe 2 prompt revision (§10 list) gains substantive
  shape from Phase 8*. Phase 7's organic-mention rate of 3-of-4
  quadruplet elements + Phase 7.5's 1-of-4 already destabilized
  the synthesis §10 candidate-twelfth-item ("re-scope the held-
  out criterion given organic surfacing"); Phase 8's reading
  adds a deeper consideration: the synthesis §1.4 quadruplet's
  substrate-side elements (1-3) read patterns that the four-way
  comparison shows are *largely generic at the substrate-side*
  — the per-dim KL allocation, the perturbation-recovery
  dynamics, and the self-prediction error trajectory shape all
  sit in distributions whose KS-D between EMA-tracked and
  random-orthogonal targets is comparable to or smaller than
  the KS-D between zero-init and small-Gaussian column choices.
  The reflexive-attention-shape claim a Probe 2 Advocate could
  make on substrate-side patterns alone would not survive the
  Skeptic's "this is generic encoder-shaping, observable under
  both target_modes" refutation. Probe 2's prompt revision
  should:
  1. De-emphasize substrate-side claims as the primary reading
     (they're real but mostly auxiliary-target-shape-independent).
  2. Emphasize head-internal sp_err distribution as the cleanest
     self-specificity signal (Phase 7 vs frozen-target sp_err
     KS-D = 0.284 is the largest sp_err separation; a Probe 2
     reading that binds reflexive-attention-shape to the head's
     own loss would be more falsifiable than substrate-side
     pattern readings).
  3. Engage with behavior-side conditioning patterns as the
     least confabulation-resistant of the three reading surfaces
     (synthesis §3 v2 (j) was right; the four-way data gives
     it more structure).

- *Probe 3's design space narrows*. Phase 7.5's three options
  for unfreezing the actor's column (small-Gaussian init; non-
  zero scalar in imagination; env-step-side actor training) now
  have a Phase-8-specific reading: option 1 (small-Gaussian init,
  what Phase 7.5 + frozen-target do) produces "alive at
  substrate-side, mostly generic, behavior-side conditioning at
  faint magnitude with init-determined end-regime." Option 2
  (run head over imagined trajectories with non-zero scalar)
  remains the path that would let the column move via training
  rather than only via init draw — and is now more attractive
  than Phase 7.5 left it, because the substrate-side data shows
  that *not having the column move* leaves the affordance's
  self-specific exercise impossible at the behavior-side. Probe
  3's research and synthesis should engage with: does dream-
  state head-engagement give the column a developmentally
  reachable training pathway, distinct from the init-shape it
  currently occupies? The dream-state question raised at Phase
  7.5 about "richer conditioning" now has a substantive
  motivation — the column-init-shape determinism is itself the
  thing dream-state could (or could not) escape.

- *The Probe 2 calibration protocol's lesion baseline*. Plan
  §10(2) names two new lesion shapes — `disable_self_prediction`
  (zero the head's output) and `zero_or_randomize_scalar` (null
  the actor's PolicyView field). The Phase 8 data argues for a
  third lesion candidate: **`init_zero_scalar_column`** — load a
  Phase-7.5/frozen-target checkpoint, replace the actor's
  `actor.net.0.weight[:, 216:]` column with zeros, and re-run
  evaluation. This lesion would test whether the late-trajectory
  policy regime depends on the column's specific init values
  (the L/R bipolar end-state for both Phase 7.5 and frozen-
  target) or on the *non-zero magnitude* of the column more
  generally. If the lesioned actor's policy collapses to Phase
  7's u/d regime, the column-init-determined behavior reading
  is confirmed; if it stays at L/R, something else is shaping
  the regime. Held for Probe 2's plan revision.

- *The recon trajectory's frozen-target reduction is unexpectedly
  large* (Probe 1 mean 38.98 → frozen-target 29.56; a 24%
  reduction, larger than Phase 7's 15% and Phase 7.5's 16%).
  Three plausible reads: (1) random-orthogonal targets are a
  simpler optimization for the head, leaving more capacity for
  the encoder to be shaped toward reconstruction; (2) the
  reduced sp_err target diversity means the head saturates
  earlier, so its gradient influence on the encoder concentrates
  in early-training when the encoder is most plastic; (3)
  frozen-target's higher policy-stay count produces more
  reconstructible trajectories (less novel state visitation).
  The data doesn't adjudicate; held as a Phase-8 observation
  for Probe 2's calibration.

- *The schema-version filtering in `_compare_controls.py` is
  silent on the case where a user passes a 0.1.1 (Probe 1.5
  alpha) directory*. The current logic reads the `schema_version`
  column, joins distinct values, and reports them in the run-
  overview line; the KS-test logic only checks for column
  presence (not for schema-version compatibility). For Probe
  2's resumption a schema-version-aware filter may be useful
  if mid-development variants land. Held as a low-priority
  comparison-driver polish item; the current behavior is correct
  for the runs that exist (`0.1.0` and `0.2.0` only).

---
