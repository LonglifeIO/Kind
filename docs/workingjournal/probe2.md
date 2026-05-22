# Probe 2 — Working Journal

*Hand-written notes captured during the Probe 2 build. One section
per phase, with the structural decisions made mid-build, surprises
that came up, and what is now decided that was open. The Probe 2
synthesis is at `docs/decisions/Kind_probe2_synthesis.md` (v2
settled); the implementation plan is at
`docs/plans/Kind_probe2_implementation_plan.md` (v2). v1 of each is
preserved at `.v1.md` paths. This document is the running log from
the pre-build revision forward.*

## Phase index

Each phase entry begins with `## Phase N` (heading literal — exact
match) and ends with `---`. Inline mentions of phase numbers inside
earlier phases' "newly-open" sections look like `Phase 3` (no
leading `##`); they are *not* phase boundaries.

- **[Pre-build revision](#pre-build-revision)** — v2 of synthesis + plan against Probe 1.5 findings; Phase 0a decision (2026-05-07)
- **[Phase 0](#phase-0--probe-2-schemas-2026-05-07)** — Probe 2 schemas (2026-05-07)
- **[Phase 1](#phase-1--env-revision-2026-05-07)** — env revision (2026-05-07)
- **[Phase 2](#phase-2--hierarchical-digest-2026-05-07)** — hierarchical digest with three reading surfaces (2026-05-07)
- **[Phase 3](#phase-3--shuffled-telemetry-generator-2026-05-07)** — shuffled-telemetry generator with four protocols (2026-05-07)
- **[Phase 4](#phase-4--lesion-scaffold-2026-05-08)** — lesion scaffold with three Probe-1.5-specific candidates plus v1 ensemble lesions (2026-05-08)
- **[Phase 5](#phase-5--pre-registration-runner-glue-2026-05-08)** — runner glue for the pre-registration sink + hand-fill template (2026-05-08)
- **[Phase 6](#phase-6--criterion-registry-2026-05-08)** — typed mirror-side criterion registry shelf with structural read-only invariants (2026-05-08)
- **[Phase 7](#phase-7--frozen-criteria-v2-binding-contract-2026-05-11)** — the three v2 criteria written as concrete records; `signal_mappings` refactored to a typed `SignalMapping`; `ReadingSurface` enum migration on `PreRegistration` (2026-05-11)
- **[Phase 8](#phase-8--adversarial-pass-orchestrator-2026-05-13)** — the adversarial-pass orchestrator (statistics, perturbation alignment, prompt builder, LLM caller, orchestrator entrypoint) (2026-05-13)
- **[Phase 12](#phase-12--calibration-driver-and-smoke-harness-2026-05-13)** — calibration driver, sham-schedule injection, round-diff record, LLM-call audit, the smoke harness, the first real-Gemini run; four schema/SDK fixes journaled (2026-05-13)
- **[Phase 10](#phase-10--stability-runner-2026-05-18)** — paraphrase-and-reseed stability runner with per-surface admissibility gating; `framing_override` extension to the prompt builder; `seed` / `temperature` extension to the LLM caller (2026-05-18)
- **[Phase 10 — smoke findings](#phase-10--stability-smoke-findings-real-gemini-api-run-2026-05-18)** — first real-Gemini stability check on `reflexive_attention`; head-internal paraphrase 0.333, reseed 1.000, admissible False; the metric's sharpness on cited-scalar-field string formatting surfaced as the dominant variation source (2026-05-18)
- **[Phase 11](#phase-11--faithfulness-verifier-2026-05-18)** — faithfulness verifier with canonical-form citation match and per-claim verdict over four `FaithfulnessStatus` values; verdict overwrites the LLM's pre-fill; data-shapes smoke against the on-disk Phase 13 calibration readings surfaces the multi-suffix and per-perturbation-list-as-trajectory failure modes (2026-05-18)
- **[Phase 11.5](#phase-115--window-2026-05-21)** — Window, a read-only Flask viewer over the mirror's on-disk records (rounds, judgments, passes, LLM-call audit, telemetry), served on localhost and reached over Tailscale; practical infrastructure for the long Probe 4 run, journaled out-of-band from the synthesis-spec'd phase sequence (2026-05-21)
- **[Phase 12 — admissibility consumer](#phase-12--admissibility-consumer-2026-05-21)** — the consumer that joins Phase 10's per-surface stability scores and Phase 11's per-reading faithfulness verdicts into a single per-reading admissibility verdict via AND-conjunction; one new module, two frozen records, no LLM calls; Window surfaces the verdict at `/admissibility` and inline per pass (2026-05-21)
- **[Phase 12.5 — deferred-cleanup pass](#phase-125--deferred-cleanup-pass-2026-05-22)** — four small deferred items closed in one focused session: drop the LLM's `faithfulness_status` write surface, name the trajectory-classifier labels in the equanimity prompt, make canonicalization iterated, and lift it into a shared module; surgical fixes, not a substrate phase (2026-05-22)

---

## Pre-build revision

*v2 of synthesis + plan against Probe 1.5 findings; Phase 0a decision · 2026-05-07*

The trigger was re-reading the project documents — chiefly the
"Reflection and self-modeling" section added to
`Kind_design_notes.md` during Probe 1.5 Phase 7.5 — alongside Phase
7's column-is-zero finding, Phase 7.5's small-Gaussian
fixed-but-non-zero column, and Phase 8's four-way KS comparison
against the v1 Probe 2 synthesis (drafted 2026-05-03, before Probe
1.5 ran). The system produced findings the v1 documents were
drafted before; the discipline catches the design's response to
those findings, not the system's behavior itself. v1 is preserved
at `Kind_probe2_synthesis.v1.md` and
`Kind_probe2_implementation_plan.v1.md`; v2 lands at the
unsuffixed paths.

What changed in shape. The five reflexive-attention elements
redistribute across three reading surfaces with explicit different
evidential weights — the load-bearing v2 decision. Substrate-side
patterns (per-dim KL allocation, ensemble-disagreement variance,
perturbation-recovery dynamics) are the weakest reading surface
because Phase 8's within-Probe-1.5-family KS-D of 0.077–0.133 vs
cross-family-to-Probe-1 KS-D of 0.22–0.34 shows substrate-shaping
is largely auxiliary-target-shape-independent. Head-internal
patterns (`sp_err` distribution, per-dim allocation) are the
strongest surface for self-specificity because Phase 7 vs
frozen-target sp_err KS-D = 0.284 — the head's own loss
distinguishes self-specific from generic supervisory targets at
measurable resolution. Behavior-side conditioning is the most
confabulation-susceptible surface because the actor's new column
is byte-identical to its initialization throughout training (the
mask-via-zero-feed convention foreclosed gradient flow during
imagine), so conditioning is fixed-by-init not
developed-through-training. The calibration protocol's seven
elements operate per-surface; pre-registration names which
surfaces each criterion reads against; the Judge rules per-claim
per-surface; multi-surface readings are stronger evidence than
single-surface readings. The reflection-vs-self-modeling
distinction binds the prompt and Judge level: the Advocate names
reflection (manasikara, prediction-of-self, witness-as-distinct-
from-content), the prompt is silent on self-modeling vocabulary,
the Judge flags self-modeling-shaped claims `unresolved`. New
lesion candidate `init_zero_scalar_column` (Phase 8 recommendation)
joins `disable_self_prediction` and `zero_or_randomize_scalar` as
the three Probe-1.5-specific lesions; the column-init confound is
carried in pre-registration; the held-out criterion default
shifted from v1's substrate-side `reflexive_attention_triplet` to
behavior-side `behavior_side_scalar_conditioning`; the
conditioning analysis module formalizes Probe 1.5's
`show_self_prediction_conditioning` helper as
`kind/observer/conditioning.py` with structured `ConditioningResult`
records the faithfulness verifier resolves cited claims against;
the criterion registry expands from three to six.

What stayed the same. Parallel-with-Judge architecture along
framework + role axes; the seven calibration elements as a
structure; the journal-discipline scaffolding; the
Watts-default-applied-to-builder discipline (extended with the
Probe 1.5 v2 §2(b) four-part exception sub-clause as a permanent
feature); the env-server's start-cell randomization; the descope
on second-order volition with reason logged; the rejection of
multi-round debate, sequential reader-critic, courtroom
multi-turn, 0% false-positive bar, and the imagined critic
network. The 4:1 contrast threshold's structure carries (per-surface
tuning is what's new, not the ratio's existence).

Phase 0a decision: skip. The v2 synthesis recommended (not
required) building the environmental-auxiliary control before
Phase 0 — Phase 8's substrate-genericity reading makes the
canonical disambiguator between auxiliary-target-shape-independent
and auxiliary-target-shape-dependent encoder-shaping more
informative than the lean Phase 8 anticipated. Decision against:
randomness in the project is reserved for specific roles (dream
state and perturbations) per the design notes' framing; the
frozen-target was a defensible one-time methodological control
because the question it answered (does the head do self-specific
work or generic auxiliary-loss work) is the kind of question a
methodological control is *for*; the environmental-auxiliary would
be a second methodological control that doesn't fit the project's
commitments on what randomness is for in the substrate. The
substrate-side stratification's empirical anchor rests on the
four-way KS comparison from Phase 8 alone. If Probe 2's calibration
surfaces a substrate-side claim that hangs critically on the
auxiliary-target-shape-independence finding and the four-way
anchor proves thin, the journal records the absence's effect at
that point and the question of whether to build the environmental
control returns. The decision is recorded; the absence is not
silent.

One observation worth carrying into Phase 0. The v2 synthesis
came back longer than the discipline supports on first draft and
needed a tightening pass; the implementation plan was kept at its
current length because plans are reference material read in
pieces, not in single passes. Verbosity drift in synthesis
documents specifically — where the synthesis is supposed to fit
the question, not the available context — is a kind of accretion
the discipline catches but only when it is watched for. Worth
flagging as a pattern to watch in future research flows alongside
the existing CA-MAS-pattern flag against feature accretion without
specific questions; the synthesis-length pattern is the same shape
applied to prose.

### What's now closed

- v1 of synthesis and plan preserved at `.v1.md` paths;
  unsuffixed paths carry the v2 versions forward.
- The three reading surfaces (substrate-side, head-internal,
  behavior-side) are settled with their evidential-weight ordering
  per Phase 8's empirical anchor.
- The reflection-vs-self-modeling boundary is operationalized at
  the Advocate's prompt (silent on self-modeling vocabulary) and
  at the Judge's ruling level (self-modeling-shaped claims flagged
  `unresolved` with a ground text naming the boundary violation).
- The held-out-criterion default at `behavior_side_scalar_conditioning`;
  the criterion registry expanded to six entries.
- The three Probe-1.5-specific lesion candidates settled:
  `disable_self_prediction` (substrate-side), `init_zero_scalar_column`
  (capacity-as-init-shape), `zero_or_randomize_scalar`
  (behavior-side).
- The column-init confound carried as a `column_init` field on the
  `PreRegistration` schema; the Skeptic's substrate-side mandate
  gains policy-via-init-driven state visitation as a refutation
  candidate.
- The conditioning analysis module's contract: `compute_conditioning`
  produces a structured `ConditioningResult` written to
  `runs/{run_id}/conditioning/conditioning.jsonl`; the faithfulness
  verifier dispatches on `(cited_stream, reading_surface)` for
  per-surface citation resolution.
- Phase 0a (environmental-auxiliary control) skipped; the
  substrate-side stratification's empirical anchor rests on the
  four-way KS comparison alone.

### What's now newly open

- *Phase 0 (Probe 2 schemas)* is the next build phase. Schema
  deltas: `StructuredReading` and `JudgeRuling` at
  `MIRROR_READING_V2_VERSION = "0.2.0"` with the `reading_surface`
  field on `StructuredClaim`; `PreRegistration` at
  `PRE_REG_SCHEMA_VERSION = "0.2.0"` (bumped from v1's 0.1.0) with
  the new fields; `ConditioningResult` at `0.1.0`; the JSON Schema
  export at `schemas/v0.3.0.json` covering Probe 1.5's `0.2.0`
  telemetry plus Probe 2's mirror-side and conditioning models.
- Whether the verbosity-drift observation generalizes to other
  research flows. The first signal will be the next research →
  synthesis pass after Probe 2 lands; if the synthesis-length
  pattern repeats the v2 first-draft shape, the discipline needs
  an explicit length budget at the synthesis stage rather than
  catching it after the fact.
- Whether the substrate-side stratification's empirical anchor at
  the four-way KS comparison alone holds when the per-surface
  calibration's first claims land. If a substrate-side claim's
  evidential weight depends critically on the auxiliary-target-shape-
  independence finding being robust beyond the frozen-target
  control, Phase 0a's deferred decision returns and the journal
  records the trigger.
- Whether the column-doesn't-develop fact is structurally limiting
  at the second success criterion's behavior-side reading. The
  `init_zero_scalar_column` lesion is the calibration test that
  adjudicates between capacity-as-architectural-slot and
  capacity-as-developmentally-reachable readings; Probe 2's first
  lesion run produces the data that adjudicates.

---

## Phase 0 — Probe 2 schemas (2026-05-07)

Phase 0 builds the pure-declaration surface the rest of Probe 2's
phases consume: the structured-reading models the parallel-with-
arbiter pipeline produces, the pre-registration record the
calibration protocol's first element writes, the conditioning
analysis module's data classes the behavior-side reading depends
on, the documented `world_event` payload conventions for `is_sham`
and lesion markers, and the aggregate JSON Schema export at
`schemas/v0.3.0.json`. No reader, no judge, no faithfulness
verifier, no analysis logic — those are Phases 8 / 9 / 11 / 6
respectively. Phase 0's concern is that the schemas exist, validate,
round-trip through JSONL byte-stable, and pass mypy `--strict` so
downstream phases can import without the wall-of-types problem
catching every contributing module.

### What was built

- `kind/mirror/structured.py` (317 lines): `MIRROR_READING_V2_VERSION
  = "0.2.0"`, `ReadingSurface` type alias, `StructuredClaim` model
  (with `reading_surface`, `masked_steps_handling`, and the extended
  `cited_stream` that admits `"conditioning_analysis"`),
  `StructuredReading` model (with the extended `baseline_flag`
  enum carrying `shuffled_scalar_within_trajectory` and the three
  Probe-1.5-specific lesion variants), `JudgeRuling` model with
  the per-surface tuple shape `(claim_index, reading_surface,
  ruling, ground_text)`. The Judge model carries a model-validator
  that enforces non-negative and unique indices in
  `agreement_without_evidence_unresolved`.
- `kind/observer/pre_reg.py` (260 lines): `PRE_REG_SCHEMA_VERSION
  = "0.2.0"` (bumped from v1's 0.1.0 for the new fields'
  addition), `PreRegistration` model with all v2 fields
  (`reading_surfaces_per_criterion`, `asymmetry_of_access`,
  `expected_outcome_per_surface`, `column_init`,
  `new_actor_readable_interfaces_added`), and `PreRegSink` —
  append-only JSONL writer mirroring `JsonlSink`'s shape but
  taking a *directory* (by convention `runs/{run_id}/pre_reg/`)
  and writing to `pre_reg.jsonl` inside. The model-validator
  enforces per-criterion completeness (every active criterion
  has an entry in `signal_mappings` / `falsifiers` /
  `scalar_checks` / `reading_surfaces_per_criterion`) and
  active/held-out disjointness.
- `kind/observer/conditioning.py` (215 lines): `RegimeStats`,
  `RegimeBucket`, `ConditioningResult` Pydantic models at
  `CONDITIONING_RESULT_SCHEMA_VERSION = "0.1.0"` (fresh stream;
  no v0 to bump from). `compute_conditioning` function
  signature stub that raises `NotImplementedError` with a
  Phase-6-pointing message; the schemas exist for downstream
  importers, the function name resolves at import time, and
  Phase 6 fills in the body.
- `kind/observer/schemas.py`: extended module docstring documenting
  the four `world_event.payload` conventions accumulated to date
  (`episode_id`, `start_cell`, `is_sham`, the lesion-marker pair
  `lesion_kind` + `source_checkpoint`, and the existing
  `internal_stochasticity_aggregate`). Added
  `export_json_schema_v0_3_0()` aggregating Probe 1.5's `"0.2.0"`
  telemetry models, Probe 2's `"0.2.0"` mirror-side models, and
  Probe 2's `"0.1.0"` conditioning models into a byte-stable
  export. The function imports the contributing modules locally
  (inside the function body) so `kind.observer.schemas` stays
  importable without pulling the mirror stack at module load
  time.
- `schemas/v0.3.0.json` (1455 lines, 47 642 bytes): the checked-in
  byte-stable export, regeneratable via
  `export_json_schema_v0_3_0()`. v0.1.0.json and v0.2.0.json
  remain on disk unchanged — the existing pin tests in
  `tests/test_schemas.py` continue to enforce that.
- Four test files (1002 lines total):
  `tests/test_structured_reading.py` (gate test 3 — round-trip,
  three reading_surface values, eleven baseline_flag values, the
  five cited_stream values including `conditioning_analysis`,
  three masked_steps_handling values, JudgeRuling per-surface
  tuple validation, frozen+forbid disciplines);
  `tests/test_pre_reg.py` (gate test 4 — sink write/read,
  PreRegistration validates all v2 fields, column_init validator,
  missing-falsifier raises, active/held-out overlap raises,
  empty new_actor_readable_interfaces_added accepted, empty
  criteria_active accepted, two builder_mode values, expected_
  outcome_per_surface key validation); `tests/test_conditioning_
  schema.py` (round-trip on RegimeStats / RegimeBucket /
  ConditioningResult, four regimes, three perturbation
  distributions, empty-buckets graceful path, frozen+forbid,
  compute_conditioning stub raises NotImplementedError); and
  `tests/test_schemas_v030_export.py` (byte-stability across
  invocations, matches checked-in file, three model groups
  present, v2 baseline_flag values present, three reading
  surfaces present, conditioning_analysis cited_stream
  present, v0.2.0.json export untouched).

### Test/mypy status

`pytest tests/test_structured_reading.py tests/test_pre_reg.py
tests/test_conditioning_schema.py tests/test_schemas_v030_export.py
tests/test_schemas.py`: 111 passed, 0 failed. Full suite
`pytest tests/`: 530 passed, 2 skipped (the integration smokes
that need API keys), 0 failed. `mypy --strict kind/`: 25 source
files, no issues. Phase 0's bar — schemas validate, round-trip,
mypy clean — clears.

### What surprised

Two small design tensions surfaced during the build, both worth
flagging:

- *Tuple-keyed dict in JSON.* The implementation plan §2.6 names
  `per_regime_per_perturbation: dict[tuple[str, str], RegimeStats]`
  but JSON does not support tuple keys, so a faithful round-trip
  through JSONL requires either a composite-key string or a
  list-of-buckets shape. I chose `list[RegimeBucket]` with a
  `RegimeBucket` carrying `regime`, `perturbation`, and `stats`
  fields; lookup helpers (`find_bucket`, `regimes()`,
  `perturbations()`) land in Phase 6 alongside the actual
  analysis logic. The deviation is structural, not semantic —
  the dict-lookup intent survives — but downstream phases that
  want tuple-keyed access need to go through the helpers.
- *Local imports in `export_json_schema_v0_3_0`.* The aggregate
  export pulls models from three modules
  (`kind.mirror.structured`, `kind.observer.pre_reg`,
  `kind.observer.conditioning`); putting those imports at module
  scope in `kind/observer/schemas.py` would mean any importer of
  the telemetry schemas would also pull the mirror stack, which
  is not the dependency direction the module layout suggests.
  Local imports inside the function body keep the v0.2.0
  importer surface clean. The cost is a slightly less-discoverable
  import graph; the benefit is no upward dependency from
  observer.schemas into mirror.

The Phase 0 build itself was unsurprising — the schema definitions
followed the plan §2.1 + §3 line-by-line, the Pydantic v2
discipline carried directly from `kind/observer/schemas.py`'s
existing pattern, and the mypy `--strict` pass cleared on first
run after a small adjustment (the conditioning module's stub
function needed an explicit `-> ConditioningResult` annotation
before Phase 6's body lands).

### What's now closed

- The Probe 2 mirror-reading schema at
  `MIRROR_READING_V2_VERSION = "0.2.0"`: `StructuredReading`,
  `StructuredClaim`, `JudgeRuling` exist with all v2 fields
  including `reading_surface`, `masked_steps_handling`, the
  extended `cited_stream` with `"conditioning_analysis"`, the
  extended `baseline_flag` with `shuffled_scalar_within_trajectory`
  and the three Probe-1.5-specific lesion variants, and the
  per-surface tuple shape on `JudgeRuling.rulings`.
- The Probe 2 pre-registration schema at
  `PRE_REG_SCHEMA_VERSION = "0.2.0"`: `PreRegistration` carries
  all five new v2 fields plus the per-criterion completeness
  validator and the active/held-out disjointness validator.
  `PreRegSink` writes append-only JSONL to a directory.
- The conditioning analysis module's data shape at
  `CONDITIONING_RESULT_SCHEMA_VERSION = "0.1.0"`:
  `ConditioningResult`, `RegimeBucket`, `RegimeStats` exist.
  The `compute_conditioning` stub raises `NotImplementedError`
  pending Phase 6.
- The four `world_event.payload` conventions documented in
  `kind/observer/schemas.py`: `is_sham` for sham builder
  perturbations, `lesion_kind` + `source_checkpoint` for
  lesion-marker events, plus the carried-over `episode_id` /
  `start_cell` / `internal_stochasticity_aggregate` payloads.
- The aggregate JSON Schema export at `schemas/v0.3.0.json`,
  byte-stable, regeneratable via `export_json_schema_v0_3_0()`.
  v0.1.0.json and v0.2.0.json remain unchanged.

### What's now newly open

- *RegimeBucket vs tuple-keyed dict — the lookup-helper contract.*
  Phase 6 needs to land `find_bucket(regime, perturbation) ->
  RegimeStats | None` (or similar) on `ConditioningResult` so
  the faithfulness verifier and the digest's behavior-side
  cohort can do the dict-lookup-shaped access the plan's
  citation-resolution code path expects. The helper's exact
  signature is open; the contract Phase 6 lands is what the
  faithfulness verifier (Phase 11) will resolve against.
- *The PreRegSink takes a directory; the runner threads it.*
  The runner phase will need to construct `PreRegSink(run_dir
  / "pre_reg")` per round and ensure the sink is closed
  cleanly. There is no Phase 0 runner glue (the runner doesn't
  exist as a Probe 2 surface — it's the Probe 1.5 runner with a
  pre-reg call hook). The hook's location and lifetime are
  open until Phase 5 (the pre-registration sink phase) lands.
- *The reading_surface enum is forward-extensible but not yet
  extended.* Probe 3 may add a fourth surface (e.g.,
  `"dream_state"`) at version `0.3.0` of the mirror-reading
  stream; the JSON Schema export would need a fresh
  `schemas/v0.4.0.json` and a model bump. The current
  `ReadingSurface = Literal[...]` shape is the canonical
  carrier; widening it is a compile-time change rather than a
  schema-bump-with-migration.
- *`compute_conditioning`'s Phase 6 contract details.* The
  module docstring sketches the seven-step pipeline (load
  checkpoint → sample states → classify regimes → per-state
  forward-pass KL → aggregate → emit JSONL → return), but two
  details are open: (i) how to handle a run with no
  `builder_perturbation` events (the
  `"perturbation_window"` regime's empty case — does the result
  carry an empty bucket for that regime, or omit the bucket
  entirely? Phase 0 admits both shapes); (ii) the regime
  classification's quartile boundaries — Phase 6 decides
  whether they're computed per-checkpoint or per-run, and
  whether they're stored in the result for reproducibility.
  Phase 6 records the decisions with the analysis logic.
- *Whether the `JudgeRuling.rulings` upper-bound check on
  agreement indices belongs at the schema level.* The
  validator currently enforces lower-bound (non-negative) and
  uniqueness, but does not check that flagged indices are <
  `len(rulings)` because the agreement may span claims the
  Judge chose not to rule on. If Phase 9's Judge implementation
  shows the upper bound is always meaningful, the validator
  tightens; otherwise the looseness stays.

---

## Phase 1 — env revision (2026-05-07)

Phase 1 lands the env-side surface the calibration protocol's
null-event test will read against and the revision the Probe 1
post-run journal flagged as the first thing to try. Two changes,
both small and structurally additive: `GridWorldConfig.start_cell`
defaults to `None` (random non-wall sample per episode reset from
the regrowth RNG stream); `EnvServer.fire_sham_perturbation`
emits a flag-only `builder_perturbation` `WorldEvent` with
`payload["is_sham"]=True` and no underlying-grid mutation, no
RNG-stream advance. The substrate is untouched; the agent
observation space, action space, episode length, and four real
mutators are unchanged; this is the env-only surface change the
synthesis settled at §2.1.

### What was built

- `kind/env/grid_world.py`: `GridWorldConfig.start_cell` default
  flipped from `(3, 3)` to `None` with the docstring explicitly
  pointing at the synthesis §2.1 reasoning and the Probe 1 journal
  entry that surfaced the trivial-oscillation collapse. The
  rejection-sampling logic in `_reset_episode_world` was already
  in place from Phase 2a; one correctness fix landed (see
  *what surprised* below).
- `kind/env/env_server.py`: `_emit_env_reset` payload gains a
  `start_cell: list[int]` entry recording the agent's actual
  position at episode reset, fetched from
  `grid_world.state.agent_pos`. With the Probe 2 default this is
  the cell sampled from the regrowth stream; with an explicit
  `start_cell` config it is the configured cell (the value
  recorded is always the one used). Added
  `fire_sham_perturbation(self, mutator_label: str,
  payload: dict[str, Any]) -> None` per v1 §2.2's spec —
  the caller-supplied payload is merged with
  `is_sham=True` and `sham_label=mutator_label`; no grid mutation,
  no RNG draw, no regrowth-rate change. The lifecycle precondition
  matches the four real mutators (`start()` required, `close()`
  terminates).
- `tests/test_env_revision.py` (17 tests, ~360 lines): the six
  test families the user spec named — default `start_cell` is
  `None` (input contract); reset samples a random non-wall cell
  with the cell in the payload; reproducibility from the regrowth
  RNG seed across episodes (both at the `GridWorld` level and via
  the `world_event` JSONL alone); sham does not mutate (agent
  observation byte-identical pre/post sham); sham emits a
  `world_event` with `is_sham=True` and the caller's payload
  preserved; sham does not advance the regrowth RNG (verified by
  parallel runs at p=0.5 producing byte-identical grid trajectories
  with-and-without an interposed sham) or the drift RNG (verified
  at amplified drift magnitude with byte-identical
  `final_p` per-episode sequences). Two ancillary lifecycle tests
  (sham before `start()` raises; sham after `close()` raises).
- `tests/test_perturbation_hook.py`: one existing test
  (`test_mutator_effect_visible_in_next_observation_when_in_view`)
  updated to set `start_cell=(3, 3)` explicitly. The test's
  docstring previously read "start_cell defaults to (3, 3)" —
  that assumption no longer holds with the Probe 2 default, and
  the test pins the start so the in-view assertion at cell (3, 4)
  remains deterministic. No other existing test required changes.

### Test/mypy status

`pytest tests/test_env_revision.py`: 17 passed. Full suite
`pytest tests/`: 547 passed, 2 skipped (the integration smokes
that need API keys), 0 failed. `mypy --strict kind/`: 25 source
files, no issues. Phase 1's bar — the env behaves correctly under
the new default, the sham is byte-clean, no regression elsewhere —
clears.

### What surprised

Three things worth flagging:

- *Latent first-reset wall-rejection bug.* The pre-existing
  rejection-sampling path in `_reset_episode_world` consulted
  `self._is_wall(r, c)`, which reads from `self._grid`. On the
  first reset the grid is the all-zeros buffer from `__init__`
  (walls have not been written yet — the wall-write happens
  *after* the agent placement block), so the wall check returned
  False for every cell and a sample could land on a future-wall.
  Probe 1's empty-walls config never exercised this; Probe 2
  authorizes random-start with non-empty walls in principle, so
  the Phase 1 contract "samples a random non-wall in-bounds cell"
  required the fix. Switched to consulting `set(self.config.walls)`
  directly. The correctness change is local, unit-tested
  (`test_random_sample_lands_in_non_wall_cell` with two configured
  wall cells), and does not affect any cached run.
- *Regrowth-RNG-schedule shift relative to cached runs.* The
  random-start path consumes one or more `regrowth_rng.integers`
  draws per reset that the fixed-start path did not. Cached
  Probe 1 / 1.5 runs at the same seed are *not* byte-equivalent
  to a re-run under the new default; the post-reset coin-flip
  arrays and the resource-permutation indices are drawn from a
  different stream state. This is the intended revision and the
  cached runs remain on disk untouched. It does not surface as a
  test break because the affected determinism tests pin
  `start_cell` explicitly when they care about exact post-reset
  trajectories. Worth flagging because Probe 2's calibration
  smoke (Phase 12) reads against the cached Probe 1.5 Phase 7.5
  run, not a fresh re-run, so the schedule shift has no effect on
  the primary fixture; future Probe-2-fresh-runs that calibrate
  against new telemetry are a separate question.
- *`fire_sham_perturbation` signature decision.* The user
  prompt's simplified description ("emits a world_event with
  payload['is_sham']=True") underspecified the method's argument
  shape. Followed v1 §2.2 exactly — `(mutator_label: str,
  payload: dict[str, Any])` — so the JSONL records of a sham can
  carry the same per-mutator-shaped fields a real perturbation
  would carry (`intended_cell`, `intended_mutator`, etc.), which
  is what the calibration protocol's null-event-vs-real-event
  pairing relies on. The structural distinguisher is the
  `is_sham=True` flag plus the `sham_label` field, both written
  unconditionally; the caller's payload is merged with these so
  caller-supplied fields are never overwritten unintentionally
  (the test exercises this — `intended_cell=[3, 5]` and
  `intended_mutator="add_resource"` round-trip through the
  emission unchanged).

### What's now closed

- `GridWorldConfig.start_cell` defaults to `None` (random
  non-wall in-bounds sample from the regrowth stream); explicit
  `start_cell=(r, c)` still accepted with the same validation
  rules. The Probe 1 default is no longer the default; the
  trivial-oscillation collapse the post-Probe-1 journal entry
  flagged is now structurally addressable through random starts.
- Random-start reproducibility: every episode's start cell is
  recoverable from the `world_event` JSONL alone via the
  `start_cell` payload field on `env_reset` records. Same seed →
  same start-cell sequence; the property survives the harness
  layer because the harness reads from `grid_world.state.agent_pos`
  at emission time.
- `EnvServer.fire_sham_perturbation` exists with v1 §2.2's
  signature; the `payload["is_sham"]` and `payload["sham_label"]`
  conventions documented in `kind/observer/schemas.py` (Phase 0)
  are exercised end-to-end. The four real mutators do *not* carry
  `is_sham=True` (the structural distinguisher); the calibration
  protocol's filter is `payload.get("is_sham") is True`.
- Sham byte-cleanness: agent observation, underlying grid byte
  state, regrowth RNG schedule, and drift RNG schedule are all
  unaffected by an interposed sham. Two-run differential tests
  exercise each.
- Latent first-reset wall-rejection bug fixed; the wall-rejection
  consults the static `config.walls` set rather than the live
  grid buffer.

### What's now newly open

- *Whether random starts resolve the trivial-oscillation
  collapse.* Phase 1's contract is "the env supports random
  starts"; the empirical question of whether the substrate's
  late-episode trajectory still degenerates under random-start
  episode sequences requires a substrate run, which Phase 1 does
  not produce (out-of-scope: no substrate change, no runner
  glue). The first run that drives the substrate against the new
  env produces the data. The Probe 1.5 Phase 7.5 run that Probe
  2's calibration smoke reads against was trained against the
  old fixed-start env; the question of whether the trajectory
  collapse persists under random-starts is open until a
  fresh-run is produced (no Probe 2 phase mandates this; the
  question lands wherever a fresh run first happens, likely
  Probe 3's substrate work).
- *The regrowth-RNG-schedule shift's second-order effects.*
  No effect on Probe 2's primary fixture (the cached Phase 7.5
  telemetry). For any future fresh run that calibrates against
  the new env, the regrowth schedule sits in a different stream
  state per episode; whether this surfaces in any per-dim KL
  allocation cohort or any conditioning-analysis regime
  classification is empirical. The hook to watch is whether
  Probe 2's substrate-side cohort statistics on a fresh run look
  qualitatively unlike the cached run's; they should not, but
  the regrowth-schedule shift is the one structurally identifiable
  difference if they do.
- *Sham cadence at calibration time.* The implementation plan §6
  default is "1 sham per 10 real perturbations". Phase 1 does
  not enforce or schedule the cadence — the harness exposes
  `fire_sham_perturbation` as a method and the calibration
  driver decides when to call it. Whether the default cadence
  is sufficient to surface the protocol failures the plan
  anticipates waits on Phase 12's smoke; the cadence may need
  tuning per-surface (substrate-side false positives vs
  behavior-side false positives may calibrate differently).
- *The `start_cell` payload field's downstream consumption
  contract.* Phase 0 documented the field in
  `kind/observer/schemas.py`'s module docstring; Phase 1 emits
  it on every `env_reset`. The hierarchical digest (Phase 2) and
  the conditioning analysis module (Phase 6) will consume it for
  per-episode start-cell-aware statistics if the readers' prompts
  cite a per-state distribution that depends on the agent's
  start. The exact consumer contract is open — Phase 2 / 6 may
  bind the field to the digest's per-episode mini-digest as a
  state-context line, or may ignore it for the substrate-side
  surface and only surface it at the behavior-side conditioning's
  state-sampling step. The field is available either way; the
  consumer phase decides.

---

## Phase 2 — hierarchical digest (2026-05-07)

Phase 2 lands the prose surface every later phase reads — the
hierarchical digest extension to `kind/observer/digest.py` that
exposes the three Probe-2-v2 reading surfaces as named cohorts in
each per-episode mini-digest. The Probe-1-shape flat digest is
preserved unchanged behind the `build_flat_digest` alias; Probe 1 /
Probe 1.5 callers see no behavioural change. The Phase 2 build is
parallelizable with Phases 3 / 4 / 5 / 6 / 11 once Phase 0 + 1 land,
which they have; doing Phase 2 next while the schema context is
fresh and because every later phase reads the digest's output is the
sequencing the implementation plan §1 names.

### What was built

- `kind/observer/digest.py` extended (~600 lines added):
  - `build_hierarchical_digest(telemetry_dir, *, n_episodes,
    flagged_only=False, with_sham=True, conditioning_dir=None,
    behavior_anomaly_z_threshold=3.0)` returns a
    `HierarchicalDigest` aggregating four prose surfaces: a
    run-summary header, a per-episode mini-digest dict keyed by
    episode_id, a list of `FlaggedAnomaly` records cohort-tagged at
    the `reading_surface` field, a world-event timeline with sham
    distinguisher, and a `DrillDownAccessor` for follow-up reads.
  - `FlaggedAnomaly` (frozen dataclass): `kind`, `reading_surface`,
    `episode_id`, `description`. Surface tagging is in the field;
    descriptions also begin with the cohort label in square brackets
    so prose readers do not need to re-derive the cohort from the
    `kind` string. The cohort label in the description text is what
    matters for the Phase 8 reader's prompt — the readers cite the
    cohort by name; redundant labelling avoids prompt fragility.
  - `HierarchicalDigest` (frozen dataclass): the four prose surfaces
    plus the drill-down accessor.
  - `DrillDownAccessor` exposes `fetch_window`, `fetch_dream`,
    `fetch_world_event`, `fetch_self_prediction` (per Probe 1.5 v2
    §10 item 1 — masked flag visibility on per-step output is
    explicit), `fetch_conditioning` (reads the loaded
    `ConditioningResult` records when `conditioning_dir` was
    supplied; returns the graceful-degradation line when not).
  - `build_flat_digest = build_digest` aliasing carries the
    backward-readability path the spec asked for — `build_digest`
    stays callable; `build_flat_digest` is the v2-named entry point
    for callers that want the explicit "I'm consuming the flat
    shape" semantics.
  - Three-cohort layout per the spec exactly:
    `[substrate-side]` carries `kl_aggregate_t` mean/std/p90,
    per-dim KL allocation top-5 (variance-ranked across the
    episode's `kl_per_dim_t` arrays), ensemble_disagreement mean
    plus a coarse regime label. `[head-internal]` carries
    `self_prediction_error_t` (excluding masked) mean/std/p90,
    self_prediction outliers (z>3) inline, self_prediction
    allocation top-5 dims (variance-ranked across the episode's
    `(self_prediction_t[d] - h_{t+1}[d])` residuals using the
    next-step online `h_t` as the EMA-target proxy — the
    convention `kind/observer/eyeball.py` already uses for the
    Probe 1.5 helper), masked-step count.  `[behavior-side]` carries
    one line per regime aggregating `n_states`, `KL_mean`, `KL_p90`
    (n_states-weighted) across perturbation distributions.
  - Graceful degradation: Probe 1 (0.1.0) records produce
    head-internal + behavior-side cohorts replaced by
    `(no self-prediction telemetry — records are Probe 1,
    schema_version 0.1.0)`. `conditioning_dir=None` produces
    behavior-side cohort replaced by `(no conditioning data —
    conditioning_dir not supplied)`; the substrate-side and
    head-internal cohorts are unaffected.
  - Sham distinguisher in `world_event_timeline`:
    `payload["is_sham"]==True` events get a `[SHAM]` prefix and the
    `sham_label` is rendered alongside; real events render
    unchanged. The timeline also surfaces `start_cell` from
    Phase 1's `env_reset` payload extension, plus `lesion_kind` /
    `source_checkpoint` for any future lesion mirror-marker events.
  - Five flagged-anomaly kinds across the three surfaces:
    `kl_aggregate_outlier` + `recon_outlier` + `kl_allocation_shift`
    + `ensemble_disagreement_collapse` (substrate-side);
    `self_prediction_outlier` + `self_prediction_allocation_shift`
    (head-internal); `conditioning_kl_p90_outlier` (behavior-side
    when `conditioning_dir` is supplied). The thresholds are tuned
    against the Phase 7.5 fixture so the healthy run produces
    meaningful anomalies (50 KL outliers + 25 recon outliers + 51
    sp_err outliers + 6 sp_allocation shifts + 0
    kl_allocation shifts + 0 collapses; the
    no-allocation-shift-on-z-dim-16 reading is itself an
    observation worth carrying forward — see "What surprised").
- `tests/test_hierarchical_digest.py` (~470 lines, 13 tests
  covering gate test 9): build against the Phase 7.5 fixture with
  three populated cohorts; build against the Probe 1 fixture with
  graceful degradation on head-internal + behavior-side; build
  against the Phase 7.5 fixture with `conditioning_dir=None` and
  only behavior-side degraded; drill-down `fetch_self_prediction`
  with masked-flag visibility on a known window; drill-down
  `fetch_conditioning` returning a known (regime, perturbation)
  bucket; drill-down `fetch_conditioning` with `conditioning_dir=
  None` returning the graceful line; sham vs real
  builder-perturbation in the timeline; head-internal sp_err
  outliers surface as flagged anomalies on the actual fixture;
  behavior-side `KL_p90` outlier flagging on a synthetic
  conditioning record with one deliberately-large bucket;
  `n_episodes=10` produces 10 mini-digests selecting the most
  recent 10; `flagged_only=True` filters to anomaly-bearing
  episodes; `flagged_only=False` keeps all selected;
  `build_flat_digest` alias produces the Probe-1-shape header.

### Test/mypy status

`pytest tests/test_hierarchical_digest.py`: 13 passed, 0 failed.
Full suite `pytest tests/ --ignore=tests/test_integration_smoke.py
--ignore=tests/test_integration_smoke_probe1_5.py`: 536 passed, 2
skipped, 0 failed. Including the integration smokes (which need
API keys), `pytest tests/`: 559 passed, 1 transport flake (not
related to Phase 2; passes in isolation), 2 skipped.
`mypy --strict kind/`: 25 source files, no issues. Phase 2's bar —
the digest builds against both fixtures, the three cohorts populate
correctly, the drill-down accessors return the contracted shapes,
mypy clean — clears.

### What surprised

Three things worth flagging:

- *Cohort labels need to live in flagged-anomaly descriptions, not
  just the `reading_surface` field.* The first design had only the
  field; the prose descriptions read "episode N step=K:
  kl_aggregate_t=V (z=Z)" without a cohort prefix. The reader's
  prompt at Phase 8 cites cohorts by name, and the Judge's per-
  surface ruling cites the ground text the Advocate quoted from
  the digest. If the description doesn't carry the cohort label,
  the Advocate has to derive it from a structured field the prompt
  may strip; the redundancy in the description text is what makes
  the cohort attribution survive prompt-shape variation. Updated
  every flagged-anomaly description to start with the cohort label
  in square brackets — `[substrate-side] episode 22 step=15:
  kl_aggregate_t=...`, `[head-internal] episode 23 step=37: ...`,
  `[behavior-side] checkpoint=ckpt-000001, regime=steady_state,
  perturbation=gaussian: KL_p90=...`. The pattern is duplication
  but cheap; the alternative is reader-prompt fragility against a
  load-bearing attribution decision.
- *Per-dim allocation shift detection at h_dim=200 vs z_dim=16
  splits cleanly.* Initial Jaccard threshold of 0.5 (substantial
  overlap) fired on every consecutive pair against the Phase 7.5
  fixture (24 of 24 pairs for both substrate-side kl_top and
  head-internal sp_top). The reason is the random-pair expected
  Jaccard at top-5/16 is ~0.31 (already below the 0.5 threshold);
  at top-5/200 it's vanishingly small. Switched the trigger to a
  strict "intersection size = 0" — complete reallocation. Against
  the actual fixture, kl_top (z_dim=16) never trips (always at
  least one shared dim across consecutive episodes; Probe 1.5's
  posterior carries a small core of dominant dims); sp_top
  (h_dim=200) trips 6 of 24 times — a non-trivial fraction the
  reader can drill into. The asymmetry — substrate-side never
  flagging while head-internal sometimes does — is itself a Phase
  7.5 reading: the substrate's per-dim KL allocation has a stable
  core; the head's per-dim self-prediction allocation reorganizes
  more readily. This isn't a finding to act on yet; it's a hook
  for the Phase 8 reader's prompt.
- *Behavior-side anomaly flagging requires multiple buckets to
  estimate dispersion.* The plan §2.3 spec names "any (regime,
  perturbation) bucket whose KL_p90 is more than X standard
  deviations from the run's empirical mean; X tunable, default 3."
  The implementation needs ≥2 KL_p90 values to compute a
  meaningful std; with 1 record and 1 bucket, std=0 and the test
  collapses. The synthetic fixture in the tests writes 12 buckets
  (4 regimes × 3 perturbations) so the threshold has signal to
  work with. For real Probe 2 runs the conditioning module's
  Phase 6 implementation will produce one record per checkpoint
  with all 4×3 buckets populated; multi-checkpoint runs produce
  multi-record cross-time dispersion. The single-bucket
  degenerate case is journal-worthy because it's the failure mode
  if the conditioning module is invoked with `regimes=["…"]` /
  `perturbation_distributions=["…"]` config narrowed to one of
  each — the digest's anomaly detector silently produces zero
  flags rather than a useful threshold.

The Phase 2 build itself was structurally clean — the existing
`build_digest` machinery (per-dim variance, percentiles, outlier z-
scoring) extended naturally to the three-cohort layout. The
read-side I/O is local to `_read_parquet_dir` + `_read_jsonl`
helpers in the same file; no new external dependency, no upward
import from observer into mirror.

### What's now closed

- The three-cohort layout per `[substrate-side]` /
  `[head-internal]` / `[behavior-side]` labels in square brackets,
  rendered identically across mini-digests and across cohort-aware
  flagged-anomaly descriptions. Phase 8's reader prompts will
  reference each cohort by name; Phase 9's Judge will cite the
  cohort the claim sits on.
- The graceful-degradation paths: Probe 1 records collapse
  head-internal + behavior-side to a single
  no-self-prediction-telemetry line; `conditioning_dir=None`
  collapses behavior-side only; both messages are fixed strings
  callers can match on programmatically.
- The drill-down accessor's contract: `fetch_window`,
  `fetch_dream`, `fetch_world_event`, `fetch_self_prediction`,
  `fetch_conditioning` are the five surfaces. Each returns
  ready-to-paste prose; each tolerates the absent-data case
  gracefully (out-of-range index, no rows in window, no
  conditioning_dir, no matching bucket).
- The `start_cell` payload field's first consumer: the world-event
  timeline renders it inline alongside `episode_id` on every
  `env_reset` event. The hierarchical digest's per-episode
  mini-digest does not yet bind the field — Phase 6's conditioning
  module may consume it for per-state sampling diagnostics; Phase
  2's decision is to surface in the timeline (which is per-event)
  but not in the per-episode (which is per-step-aggregate). The
  consumer contract Phase 1 left open settles on this split.
- `build_flat_digest` is the backward-readable alias. Probe 1 /
  Probe 1.5 callers can continue to import either name; the
  underlying function is the same callable.
- Five flagged-anomaly kinds across three surfaces; thresholds
  tuned against the Phase 7.5 fixture (substrate-side KL z>3,
  head-internal sp_err z>3, allocation shifts at intersection=0,
  ensemble collapse at mean intrinsic_signal_t < 1e-3,
  behavior-side conditioning KL_p90 z>3). The thresholds are
  defaults; Phase 12's smoke informs whether they need per-surface
  retuning.

### What's now newly open

- *The conditioning_dir consumer contract for Phase 6.* The
  digest's `conditioning_dir` argument expects a directory holding
  `conditioning.jsonl`; each line is a `ConditioningResult`-shaped
  JSON record. Phase 6's `compute_conditioning` writes this file by
  the same convention. The contract Phase 2 settles is the read
  shape: the digest uses `regime`, `perturbation`, `stats.n_states`,
  `stats.kl_mean`, `stats.kl_p90` from each bucket. Phase 6's
  decisions about quartile boundaries (per-checkpoint vs per-run),
  per-perturbation reproducibility seeds, and the empty-bucket
  shape for runs with no `builder_perturbation` events all
  propagate into the records the digest reads — the digest itself
  doesn't care about those decisions, but downstream consumers
  (the Phase 11 faithfulness verifier, the Phase 8 reader's
  citations) will. Phase 6 records the choices alongside the
  analysis logic.
- *The behavior-side anomaly threshold when conditioning_dir is
  supplied with sparse buckets.* The default
  `behavior_anomaly_z_threshold=3.0` assumes a dispersion estimate
  is computable, which requires ≥2 KL_p90 values across all
  loaded buckets. A real Probe 2 calibration round may invoke the
  conditioning module across multiple checkpoints, producing
  enough buckets for a stable estimate; a tight-budget round may
  invoke it once with a narrow regime list, producing few buckets
  and a noisy threshold. The threshold's default is documented;
  Phase 12's smoke surfaces whether the default is workable, and
  Phase 6's per-checkpoint cadence interacts.
- *Whether Phase 8's reader prompts need any digest-level
  affordances Phase 2 hasn't exposed.* The cohort labels carry the
  surface attribution; the drill-down accessor carries the
  follow-up I/O; the flagged-anomaly descriptions carry the
  per-anomaly cohort. What's not yet exposed: a per-cohort
  contrast band ("substrate-side cohort is auxiliary-target-shape-
  independent at within-Probe-1.5-family KS-D 0.077–0.133 vs
  cross-family 0.22–0.34" — Phase 8's empirical anchor from
  Probe 1.5 v2's Phase 8 reading). The Skeptic's substrate-
  genericity refutation cites this band at the substrate-side
  surface; if the digest doesn't expose the band, the Skeptic's
  prompt has to carry it as static text. The Phase 8 build
  decides — the digest does not pretend the band is reproducible
  from telemetry alone (it isn't; it's a cross-run KS comparison
  the Phase 8 reading committed).
- *The kl_top vs sp_top allocation-shift asymmetry as a Phase 8
  reading hook.* For the Phase 7.5 fixture the substrate-side
  `kl_top` never reallocates fully across consecutive episodes
  (z_dim=16 produces a stable dominant core); the head-internal
  `sp_top` reallocates fully on 6 of 24 pairs (h_dim=200 produces
  a more-volatile per-dim attribution). This asymmetry is a
  digest-level observation Phase 8's prompt could legitimately
  surface as a head-internal manifestation candidate ("the head's
  self-prediction work distributes across more h-dimensions
  episode-to-episode than the substrate's KL work distributes
  across z-dimensions"). Whether the asymmetry is self-specific
  or reflects the dim-count ratio is the Skeptic's question; the
  digest exposes the data without committing the answer.

---

## Phase 3 — shuffled-telemetry generator (2026-05-07)

Phase 3 lands the calibration protocol's second element (synthesis
§2.4 element 2; implementation plan §2.4 + §6 calibration table
row 2) — the four shuffled-telemetry protocols that produce
structurally indistinguishable telemetry directories the Phase 2
digest reads without modification. Each protocol is a null baseline
against a specific invariant of a genuine reading: per-episode
marginal preservation under within-episode shuffle; within-episode
content preservation under across-episode shuffle; per-episode
action-distribution preservation under action-state decoupling; per-
trajectory scalar marginal preservation under within-trajectory
scalar shuffle. Phase 3 is parallelizable with Phases 4 / 5 / 6 /
11; sequencing it after Phase 2 keeps the Phase 12 calibration
smoke's scaffolding building forward without backtracking on shape
decisions Phase 2 just settled.

### What was built

- `kind/observer/shuffle.py` (~520 lines): a single new module
  carrying the four protocol functions, the `ShuffleManifest`
  dataclass, the four `ShuffleProtocol` literal values, and the
  shared parquet I/O / episode-grouping helpers. The module imports
  `_arrow_schema_from_pydantic` from `kind/observer/sinks.py` so
  the output's Arrow schema matches the source byte-for-byte at the
  field-type level — the digest's `pq.read_table(...).to_pylist()`
  reads the shuffled output exactly as it reads a freshly-written
  ParquetSink shard.
  - `shuffle_within_episode(telemetry_dir, output_dir, seed)` —
    within each episode, the 16 content fields are permuted under
    one shared per-episode permutation; the 7 indexing fields
    (`schema_version`, `run_id`, `checkpoint_id`, `t`, `episode_id`,
    `step_in_episode`, `wallclock_ms`) stay anchored to position so
    the file's positional indexing remains canonical. Per-episode
    marginals of every content field are byte-stable; lag-k
    correlations within episode are broken because per-step (action,
    kl, sp_err, ...) tuples sit at random positions in
    `step_in_episode` order.
  - `shuffle_across_episodes(...)` — episode_ids are sorted, then
    permuted; the output stream emits episodes in the permuted
    order. Each episode's records stay in original step order and
    keep their original `episode_id`. File-iteration order changes;
    grouped-by-episode_id content is byte-identical to source.
  - `decouple_action_state(...)` — within each episode, `action_t`
    values are uniformly permuted; per-episode action distribution
    is byte-stable; `action_logprob_t` is *not* moved with the
    action and stays paired with the original state, becoming
    informational only after the shuffle. The contract that
    "everything else stays in place" was load-bearing in the
    spec; the minimal-intervention reading is what cleanly
    isolates the (state, action) coupling break the calibration
    targets.
  - `shuffle_scalar_within_trajectory(...)` — within each
    trajectory (each episode), non-masked
    `self_prediction_error_t` values are uniformly permuted among
    themselves; the masked flag is preserved byte-identical in
    original positions; the sentinel scalar 0.0 at masked
    positions stays. The (mask, scalar) pair stays internally
    consistent at every step. Probe 1 (`schema_version="0.1.0"`)
    records pass through unchanged because both the scalar and
    the masked flag are None there — the protocol is informative
    only on 0.2.0 telemetry.
- `ShuffleManifest` (frozen dataclass): `protocol`, `seed`,
  `source_run_id`, `output_run_id`,
  `episode_marginals_preserved`, `temporal_structure_broken`,
  `notes`. Each manifest writes to
  `output_dir/shuffle_manifest.json` as deterministic JSON
  (sort_keys=True, indent=2, trailing newline). The two boolean
  flags are informational tags the smoke and gate tests assert on;
  they do not parameterize the shuffler's behavior.
- Shared helpers — `_read_agent_step_rows`, `_write_agent_step_rows`
  (single-shard write via `pa.Table.from_pylist` with the
  Pydantic-derived Arrow schema), `_copy_unchanged_streams` (byte-
  copies `world_event.jsonl`, `replay_meta.jsonl`, and
  `dream_rollout/` shards), `_group_by_episode`,
  `_source_run_id` (reads from the first row's `run_id` field with
  a parent-name fallback), `_output_run_id` (infers from
  `output_dir.parent.name` under the `runs/{run_id}/telemetry/`
  convention). These keep each protocol function ~30 lines —
  control flow over a single rng + per-episode loop.
- `tests/test_shuffle.py` (~440 lines, 24 tests covering gate
  test 5):
  - within_episode: per-episode marginals preserved (kl_aggregate_t,
    action_t multiset, self_prediction_error_t multiset all sort-
    equal); lag-1 correlation between action_t and kl_aggregate_t
    drops by ≥0.3 in absolute value; the 7 indexing fields stay
    byte-identical at the same `(episode_id, step_in_episode)`
    position.
  - across_episodes: each episode's record set (entire dict-equal
    set of rows) is byte-identical to source when grouped by
    episode_id; the first-occurrence sequence of episode_ids in
    file iteration order differs from source.
  - decouple_action_state: per-episode action multiset preserved;
    empirical action-conditional next-step kl_aggregate_t mean
    differs between source and shuffled; every non-action_t field
    stays byte-identical at the same position.
  - scalar_within_trajectory: per-trajectory scalar multiset
    preserved (mean and std equal); regime-conditional scalar mean
    (top half of intrinsic_signal_t per episode, non-masked rows
    only) differs from source; masked flag is byte-identical in
    original positions; sentinel 0.0 stays at masked positions;
    every other field stays byte-identical at the same position.
  - cross-protocol: determinism (same input + seed → byte-stable
    row content under the digest's read+sort) parametrized over
    all four protocols; structural validity (the Phase 2
    `build_hierarchical_digest` builds against shuffled output and
    the `world_event.jsonl` timeline still surfaces env_resets)
    parametrized over all four protocols; manifest provenance
    round-trips through JSON; world_event.jsonl /
    replay_meta.jsonl byte-copied; dream_rollout shard byte-copied
    when present; absent replay_meta tolerated downstream.

### Test/mypy status

`pytest tests/test_shuffle.py`: 24 passed, 0 failed.
Full suite `pytest tests/ --ignore=tests/test_integration_smoke.py
--ignore=tests/test_integration_smoke_probe1_5.py
--ignore=tests/test_smoke_mps_script.py`: 558 passed, 2 skipped, 0
failed. `mypy --strict kind/`: 26 source files (one new), no issues.
Phase 3's bar — the four protocols produce structurally valid
output the digest builds against; per-protocol invariants
preserve/break as the spec named; determinism holds under same
seed; manifest round-trips; mypy clean — clears.

### What surprised

Three things worth flagging.

- *Within-episode shuffle's masked-flag treatment is genuinely
  protocol-specific, not shared.* The within-episode protocol
  treats `self_prediction_error_masked_t` as a content field that
  moves with the per-episode permutation along with the scalar —
  the (mask, scalar) pair stays internally consistent at every
  output row, but the step-zero masking convention does not
  survive at the position level (a row carrying mask=True and
  scalar=0.0 may now sit at any `step_in_episode` within its
  episode). The scalar-within-trajectory protocol treats the
  masked flag as anchored to position and only permutes the
  non-masked subset of scalars among themselves. Both are
  internally consistent, but the spec's wording — "masked flag
  preserved in original order" — applies *only* to the scalar-
  within-trajectory protocol, not to the within-episode protocol.
  The plan §2.4 spec puts the "masked flag preserved" property
  inside the scalar-within-trajectory bullet specifically; under
  within-episode the property does not hold and is not required.
  The journal flags this asymmetry now because Phase 9 (Judge)
  and Phase 11 (faithfulness verifier) will read the
  `baseline_flag` field on a `StructuredReading` — a claim's
  citation against a `shuffled_within_episode` baseline must not
  expect step-zero masking, while a claim against
  `shuffled_scalar_within_trajectory` may.
- *`action_logprob_t` is intentionally left orphaned under the
  decouple_action_state protocol.* The protocol's spec says
  "shuffle action_t independently across all steps"; the test
  suite asserts that *every* non-`action_t` field stays byte-
  identical at the same position, which means `action_logprob_t`
  stays paired with the original state and no longer matches the
  action stored at that step. After the shuffle, the logprob
  field is informational only — a reader that cites it as
  evidence of the actor's policy on the *current* row is reading
  a stale pair. This is the explicit choice that minimal
  intervention isolates the (state, action) conditional break.
  Phase 11's faithfulness verifier may need to flag any claim
  citing `action_logprob_t` against a `decoupled_action_state`
  baseline — the field's interpretation under that baseline is
  not the same as under genuine. The journal entry records the
  semantic so Phase 11 can encode it as a per-baseline rule.
- *`pyarrow.Table.from_pylist` plus the
  Pydantic-derived schema accepts the mixed `q_params_t` /
  `p_params_t` tuple shape after a parquet round-trip without
  re-serializing.* When `pq.read_table(...).to_pylist()` reads a
  field declared as `tuple[list[float], list[float]]` (stored in
  the Arrow schema as `list_(list_(float64))`), it returns a
  Python `list[list[float]]`. Writing those dicts back via
  `pa.Table.from_pylist(rows, schema=arrow_schema)` succeeds
  without complaint. This is the load-bearing roundtrip for the
  shuffle (read the rows, permute, write them back); the
  alternative — re-validating each row through the `AgentStep`
  Pydantic model — would have re-converted lists to tuples and
  forced a model_dump on every row. Skipping the Pydantic round-
  trip is correct because the Arrow schema is the authoritative
  field-type contract; the Pydantic validator's
  `_enforce_v2_required_fields` check ran on the source's writes
  and is preserved by the row-content equality the shuffle keeps.

The Phase 3 build itself was structurally clean — the protocols
fit cleanly inside the read-permute-write template, and the
helpers extracted naturally. No unexpected interactions with the
digest, the schemas, or any other observer module. The
parametrized determinism + structural-validity tests caught one
small bug during the build (an off-by-one in the within-episode
loop's indexing pair where `anchor` and `content` were
accidentally indexed by the same variable; the test that asserts
the indexing fields stay anchored at position k while content
comes from σ(k) caught it on first run); the fix was one line.

### What's now closed

- The four protocol contracts are settled: within_episode
  permutes 16 content fields under one shared per-episode
  permutation with 7 indexing fields anchored; across_episodes
  permutes the file-iteration order of episodes with content
  byte-identical when grouped by episode_id; decouple_action_state
  permutes only `action_t` within each episode with everything
  else byte-identical at the same position; scalar_within_trajectory
  permutes non-masked `self_prediction_error_t` within each
  episode with the masked flag and sentinel preserved.
- The `ShuffleManifest` shape: `protocol`, `seed`,
  `source_run_id`, `output_run_id`,
  `episode_marginals_preserved`, `temporal_structure_broken`,
  `notes`. The `notes` field is free-text protocol-specific
  documentation; downstream consumers (Phase 11 faithfulness
  verifier, Phase 12 smoke) read the structured fields and treat
  notes as informational. The two boolean flags carry the
  invariant tags the smoke asserts on per surface.
- The `output_dir/shuffle_manifest.json` byte format: deterministic
  JSON via `json.dumps(..., sort_keys=True, indent=2)` plus a
  trailing newline. Round-trips through `json.loads` without
  loss; downstream readers can rely on field presence.
- The output telemetry directory layout convention:
  `output_dir/agent_step/shard-NNNNNN.parquet` (one shard
  regardless of source's shard count; the digest reads via the
  glob and concatenates), `output_dir/world_event.jsonl` and
  `output_dir/replay_meta.jsonl` byte-copied,
  `output_dir/dream_rollout/` shards byte-copied. The
  `output_run_id` is inferred from `output_dir.parent.name`
  under the `runs/{run_id}/telemetry/` convention; callers that
  use a different layout get the parent-name fallback.
- The Probe-1-records-no-op behavior under
  `shuffle_scalar_within_trajectory`: 0.1.0 records have None
  for both the scalar and the masked flag, so the protocol's
  per-episode shuffle skips them and the rows pass through
  structurally unchanged. The protocol is informative only on
  0.2.0 telemetry; the digest's behavior-side cohort against
  0.1.0 records degrades to the no-self-prediction-telemetry
  line regardless. This was a documented edge case in the spec;
  the implementation honors it without raising.
- The four-protocol enumeration of `ShuffleProtocol` and the
  matching `baseline_flag` enum extension at Phase 0
  (`shuffled_within_episode`, `shuffled_across_episodes` —
  carried from v1 — plus the new v2 `shuffled_scalar_within_trajectory`
  and the carried v1 `decoupled_action_state`). Phase 8's
  reader / Phase 9's Judge / Phase 11's verifier each read one
  of these four flags off the `StructuredReading` to dispatch
  per-baseline rules.

### What's now newly open

- *The scalar-within-trajectory protocol's regime-conditional
  break is sensitive to the regime classifier's quartile
  boundaries — a Phase 6 contract.* The test exercises a top-
  half-of-`intrinsic_signal_t` proxy regime classifier; the real
  Phase 6 classifier uses top-quartile boundaries (synthesis
  §2.4 element 7's `regimes` literal: `perturbation_window`,
  `high_disagreement` = top quartile of intrinsic signal,
  `high_kl` = top quartile of `kl_aggregate_t`, `steady_state`
  residual). Whether the regime-conditional mean drift surviving
  the within-trajectory scalar shuffle is robust to the
  quartile-vs-half cutoff is a Phase 6 + Phase 12 question. If
  Phase 12's calibration smoke shows the behavior-side `supported`
  rate against the scalar-shuffle baseline is closer to 1.0 than
  to the planned ≤33% (the synthesis §4 (v2) 3:1 contrast
  threshold), the regime classifier's quartile cutoff is one of
  the knobs to revisit before retuning the threshold itself.
  Phase 6 owns the cutoff; Phase 12 reports the contrast.
- *Whether Phase 12's smoke needs any protocols beyond the four.*
  The four-protocol enumeration was settled by synthesis §2.4
  element 2 (within / across / action-state / scalar) and the
  v2 plan adds the scalar-within-trajectory specifically for the
  behavior-side reading's calibration. The smoke at Phase 12
  exercises within_episode + scalar_within_trajectory against
  the genuine reading at all three surfaces (per implementation
  plan §2.13); the across_episodes and decoupled_action_state
  protocols are Phase 11 / Phase 13 gate-test surfaces but not
  part of the smoke's headline contrast. If the smoke surfaces
  a baseline whose contrast is too tight to discriminate
  genuine from shuffled at one surface (the synthesis §6 default
  notes the substrate-side / head-internal threshold is 4:1 and
  behavior-side is 3:1; if either is too loose given the scalar
  shuffle's behavior-side break, a fifth protocol — e.g., a
  regime-randomized shuffle that re-classifies states under a
  surrogate regime — may be needed). The Phase 12 build phase
  decides; Phase 3 leaves the four-protocol commitment as the
  default and the v2 plan's recommendation.
- *The `action_logprob_t`-orphans-under-decouple semantic at
  Phase 11.* The faithfulness verifier dispatches per
  `(cited_stream, reading_surface)`; under a
  `decoupled_action_state` baseline, a citation that resolves
  `action_logprob_t` against the actual stream's value at the
  cited step gets `resolved` (the value is byte-stable), but
  the *interpretation* — whether the logprob informs Io's
  policy under the baseline — is no longer what genuine
  citations support. Phase 11's spec doesn't yet name this; the
  Phase 11 build phase will need a per-baseline rule that flags
  any `cited_scalar_field == "action_logprob_t"` against a
  `decoupled_action_state` baseline as `unresolved` with a notes
  field naming the orphan-pair semantic. This is journaled now
  rather than encoded because Phase 11 hasn't built its
  dispatch table yet; the journal entry is the prompt for
  Phase 11's per-baseline rule design.
- *Whether the within-episode protocol's "masked flag may move"
  property leaks any signal to the Phase 8 reader's prompt.*
  The reader sees the digest's per-episode mini-digest, which
  reports the masked-step count. Under within-episode shuffle,
  the masked-step count per episode is unchanged (the multiset
  of mask=True/False is preserved), so the digest's surface is
  byte-identical at the per-episode aggregate level. But the
  per-step drill-down accessor (`fetch_self_prediction`) shows
  per-step masked flags; if a reader's claim cites a specific
  step under `shuffled_within_episode`, the masked flag at that
  step is no longer guaranteed to match the genuine reading's
  flag. Whether this is a feature (the claim's citation should
  fail to resolve under a true null baseline) or a bug (the
  baseline is too aggressive at the head-internal surface, and
  contributes to over-strict shuffled supported rate) is a
  Phase 12 calibration question. The journal entry records the
  asymmetry so Phase 12 reports it; the build phase decides
  whether to relax the within-episode protocol to also anchor
  the masked flag (matching the scalar-within-trajectory
  protocol's choice) if the over-strict reading materializes.

---

## Phase 4 — lesion scaffold (2026-05-08)

Phase 4 lands the calibration protocol's fourth element (synthesis
§2.4 element 4; implementation plan §2.5 + §4 gate test 7) — the
five lesion mechanisms operating at five different points in the
substrate, each targeting one reading surface or one specific
capacity-as-init-shape question. The three Probe-1.5-specific
candidates (`disable_self_prediction` substrate-side,
`init_zero_scalar_column` capacity-as-init-shape,
`zero_or_randomize_scalar` behavior-side) plus the two v1 candidates
(`ensemble_k1`, `ensemble_constant`, both substrate-side targeting
the actor's intrinsic objective) all wire through the same
`RunnerConfig.lesion_kind` enum but reach the substrate at
different points: the world model for `disable_self_prediction`,
views.split for `zero_or_randomize_scalar`, the ensemble
constructor for `ensemble_k1` and the ensemble's disagreement
methods for `ensemble_constant`, and a checkpoint-mutation script
for `init_zero_scalar_column`. Phase 4 is parallelizable with
Phases 5 / 6 / 11 once Phase 0 lands; sequencing it after Phase 3
keeps the calibration protocol's scaffolding building forward
toward the smoke without backtracking.

### What was built

- `kind/training/runner.py` extended:
  - `RunnerConfig.lesion_kind` widened to the six-value enum
    (`None` plus the five lesion kinds). Three new fields:
    `lesion_zero_or_randomize_variant: Literal["zero",
    "randomize"]` and the empirical bounds
    `lesion_zero_or_randomize_empirical_min/_max: float` (defaults
    `[0.0, 1.0]` covering the cosine loss form's range; real Probe
    2 lesion runs will journal the bounds derived from the source
    run's empirical scalar distribution).
  - `Runner.__init__` thread-routes the kind: `"ensemble_k1"`
    overrides `ensemble_k` to 1 at construction; `"ensemble_constant"`
    flips a flag on the ensemble; `"disable_self_prediction"` is
    threaded into the constructed `WorldModelConfig.lesion_kind`;
    `"zero_or_randomize_scalar"` precomputes the
    `_views_lesion_kind` route into views.split. The init also
    seeds a `_lesion_marker_emitted` flag.
  - New `_maybe_emit_lesion_mirror_marker` helper called once at
    the top of `run()`. Emits a `world_event` record with
    `event_type="mirror_marker"`, `source="system"`,
    `payload.lesion_kind=<the kind>`, plus `lesion_variant` /
    `lesion_empirical_min` / `lesion_empirical_max` payload fields
    for `"zero_or_randomize_scalar"`. Convention matches the
    existing Probe 1.5 Phase 5 mirror_marker emission shape.
  - `_step_once` updated: under `"disable_self_prediction"` the
    per-step scalar takes the loss-form's zero-pair value
    (`cosine: 1.0`, `mse: 0.0`) without computing it from the
    actual head output, avoiding the degenerate
    cosine-similarity-of-zero-vectors call. The `views.split` call
    threads `lesion_zero_or_randomize`,
    `lesion_empirical_min/_max`, and `lesion_rng=self._sample_rng`
    so the determinism story carries from runner-level RNG to the
    randomize variant. The `_emit_agent_step` call now reads
    `policy_view.self_prediction_error` (post-split) and
    `telemetry_view.self_prediction_error_masked` (post-split)
    rather than the pre-split values, so the `"zero_or_randomize_
    scalar"` lesion's override on PolicyView lands in the
    AgentStep telemetry and the masked flag is True on every
    lesioned step. Under no lesion the post-split values equal
    the pre-split values exactly (split passes them through),
    preserving non-lesion telemetry byte-identity.
- `kind/agents/world_model.py` extended:
  - `WorldModelConfig.lesion_kind: Literal[None,
    "disable_self_prediction"] = None` — the world model only
    cares about the one lesion kind that affects its own
    behavior; the runner translates from the six-value
    `RunnerConfig.lesion_kind` to the two-value
    `WorldModelConfig.lesion_kind` at construction.
  - `WorldModel.step` honors the lesion: when
    `lesion_kind == "disable_self_prediction"`, the head output
    is replaced with a fixed zero tensor of shape `(B, h_dim)`;
    otherwise `self.self_prediction_head(h)` produces the head's
    output normally.
  - `WorldModel._update_ema_target` is a no-op under the lesion;
    the EMA target's parameters retain their construction-time
    values (which are copies of the online network's at
    construction).
  - `WorldModel.loss` returns `self_prediction_loss = 0` under
    the lesion, regardless of the loss form. The runner's
    `wm_total + λ_self * self_prediction_loss` backward thus
    contributes nothing on the head/EMA axis; `actor_loss`
    backward and `ensemble.compute_loss` backward continue
    normally.
- `kind/agents/views.py` extended: `split` accepts an optional
  `lesion_zero_or_randomize: Literal["zero", "randomize"] | None`
  kwarg plus `lesion_empirical_min`, `lesion_empirical_max`, and
  `lesion_rng: torch.Generator | None`. When non-None, split
  overrides PolicyView's `self_prediction_error` (zero variant:
  `0.0`; randomize variant: `Uniform(min, max)` drawn via
  `torch.rand((), generator=lesion_rng, ...)` on the same
  device/dtype as `step.h`) and forces TelemetryView's
  `self_prediction_error_masked` flag to True. The lesion only
  overrides the actor's behavior-side input — TelemetryView's
  `self_prediction` vector (the head's full output) is NOT
  overridden, so the substrate-side / head-internal cohorts
  continue to read the head's actual output.
- `kind/agents/ensemble.py` extended: `LatentDisagreementEnsemble`
  gains a `lesion_constant_disagreement: bool` constructor flag.
  Both `disagreement` and `disagreement_from_action_emb` compute
  `raw = self._variance(predictions)` first and *then* multiply
  by `0.0` if the flag is True (rather than short-circuiting to
  a fresh zero tensor). The multiplication keeps the autograd
  chain intact — backward through `raw * 0.0` computes a zero
  gradient on the actor's parameters rather than detaching, which
  would break `actor_loss.backward()` when the lesioned signal is
  the only contribution. The companion `ensemble_k1` lesion is
  implemented at construction time via `K=1`; variance over a
  single head is identically zero via `torch.var`'s biased
  reduction, with the autograd chain preserved (the head's
  output flows through the variance and naturally reduces to
  zero).
- `scripts/probe2_lesion_init_zero_scalar_column.py` (~360 lines):
  CLI mutation script with `--source-checkpoint`, `--output-dir`,
  `--dry-run`, plus optional `--run-id` and `--checkpoint-id`
  for stamping the emitted `world_event`. The `mutate()`
  programmatic entry returns a `MutationResult` carrying
  pre/post column moments, sidecar files copied, and the
  emitted world_event path. Loads the source checkpoint via
  `safetensors.torch.load_file`, locates `actor.net.0.weight`,
  zeros the `[:, h_dim+z_dim:]` column (the trailing single
  column under Probe 1.5's `+1` actor input dim), saves the
  lesioned checkpoint via `safetensors.torch.save_file`,
  byte-copies the five canonical sidecar files
  (`optimizer_state.pt`, `replay_meta.json`, `rng_state.pkl`,
  `schema_version.txt`, `telemetry_offsets.json`), and writes a
  single-line `world_event.jsonl` carrying the mutation-time
  mirror_marker with payload fields `lesion_kind`,
  `source_checkpoint`, `actor_first_layer_key`,
  `actor_input_dim`, `column_start_index`, and the
  pre/post column moments dicts. The `--dry-run` path computes
  the moments and returns the structured result without writing
  any files (verified by an `assert not output_dir.exists()` on
  the test's dry-run path).
- `tests/test_lesion.py` (~700 lines, 15 tests covering gate
  test 7):
  - `test_disable_self_prediction_zero_head_output`:
    `self_prediction_t` is identically zero across all dims and
    all steps under the disable lesion.
  - `test_disable_self_prediction_zero_pair_scalar`: the scalar
    is exactly `1.0` under cosine and exactly `0.0` under MSE
    on non-masked steps.
  - `test_disable_self_prediction_head_params_do_not_move`:
    the head's parameters and the EMA target's parameters are
    byte-identical between construction time and end-of-training
    (60-step CPU smoke). The regression check the prompt names.
  - `test_zero_or_randomize_scalar_zero_variant`:
    `self_prediction_error_t == 0.0` on every step;
    `self_prediction_error_masked_t == True` on every step.
  - `test_zero_or_randomize_scalar_randomize_variant_distinct_values`:
    per-step values lie inside the configured `[0.1, 0.9]`
    bounds, masked flag True everywhere, and at least 50% of
    the 60 step values are unique (a far stronger spread than
    a degenerate constant override would produce).
  - `test_zero_or_randomize_scalar_substrate_side_unchanged`:
    `self_prediction_t` (the head's full vector) is not all-zero
    under the behavior-side lesion — distinguishes this lesion
    from `disable_self_prediction` at the substrate-side cohort.
  - `test_init_zero_scalar_column_real_run`: the script
    produces a valid lesioned checkpoint; the actor's scalar
    column is zero; existing actor columns and bias are
    byte-identical to source; world model + EMA target tensors
    are byte-identical; sidecars are byte-identical; the
    world_event.jsonl carries the correct mirror_marker payload
    with non-zero `column_moments_before` and zero
    `column_moments_after`.
  - `test_init_zero_scalar_column_dry_run`: short-circuits
    before any file write (`output_dir` does not exist after
    dry-run); CLI `main()` returns 0; the moments are still
    computed.
  - `test_init_zero_scalar_column_loadable_into_runner`: the
    lesioned checkpoint loads into a fresh runner via
    `Runner.load_checkpoint`; the actor's scalar column reads
    back as zero post-load.
  - `test_non_lesion_run_byte_identical_to_reference`: two
    `lesion_kind=None` runs at fixed `torch.manual_seed(0)`
    produce byte-identical telemetry on every deterministic
    field — the regression check that lesion plumbing
    introduced no behavioral change on the un-lesioned path.
  - `test_non_lesion_run_no_mirror_marker_emitted`: no
    mirror_marker is emitted on un-lesioned runs.
  - `test_lesion_kind_recorded_in_world_event_disable` and
    `..._zero_or_randomize`: the world_event mirror_marker's
    payload carries the right `lesion_kind`,
    `lesion_variant`, and bounds.
  - `test_lesion_ensemble_k1_smoke` and
    `test_lesion_ensemble_constant_smoke`: both v1 ensemble
    lesions run end-to-end without errors;
    `intrinsic_signal_t == 0.0` on every step; mirror_marker
    payload carries the right `lesion_kind`.

### Test/mypy status

`pytest tests/test_lesion.py`: 15 passed, 0 failed. Full suite
`pytest tests/ --ignore=tests/test_integration_smoke.py
--ignore=tests/test_integration_smoke_probe1_5.py
--ignore=tests/test_smoke_mps_script.py`: 573 passed, 2 skipped
(the API-key smokes), 0 failed. Including the integration
smokes, `pytest tests/test_integration_smoke.py
tests/test_integration_smoke_probe1_5.py`: 24 passed, 0 failed.
`mypy --strict kind/training/ kind/agents/
scripts/probe2_lesion_init_zero_scalar_column.py`: 10 source
files, no issues. Phase 4's bar — five lesion kinds work
end-to-end across CPU smokes; mypy clean; non-lesion runs are
byte-identical to the reference; the v1 ensemble lesions remain
functional — clears.

### What surprised

Three things worth flagging.

- *The `ensemble_constant` lesion's grad path needed preservation,
  not just zeroing.* First implementation returned a fresh
  `torch.zeros(...)` from both `disagreement` and
  `disagreement_from_action_emb` when the flag is on. The
  resulting tensor was structurally a zero tensor with no grad_fn
  attached. Backward through the actor's
  `imagine_and_compute_loss` (which sums disagreement values
  across the imagined horizon as the only intrinsic signal) then
  raised `RuntimeError: element 0 of tensors does not require
  grad and does not have a grad_fn` because the actor loss was
  literally `tensor(-0.)` with no gradient path back to the
  actor's parameters. The fix is to compute the variance
  normally (preserving the grad chain through the head's
  parameters, which the lesion is fine with since the heads
  still train) and then multiply by `0.0` at the end. Multiplying
  a grad-bearing tensor by `0.0` produces a tensor with grad_fn
  = MulBackward and zero gradient — backward runs cleanly,
  contributes zero gradient to the actor's parameters, and the
  semantic "the actor's intrinsic motivation is constant zero"
  is preserved. The companion `ensemble_k1` lesion does not have
  this issue because `torch.var(predictions, dim=0,
  unbiased=False)` over a length-1 axis returns 0 with
  grad_fn intact (predictions still have grad through
  `_heads_forward`). The journal flags this asymmetry because
  any future lesion candidate that "forces a substrate signal to
  a constant value" must use the multiply-by-zero pattern, not
  the bare-zeros pattern, when the lesioned signal is the only
  contribution to a downstream backward.
- *The mirror_marker emission timing is at `run()` start, not
  `__init__`, and there's no race with the env-server thread.*
  The runner's transport client's reader thread routes
  `WorldEvent` records (from the env-server's mutator hooks)
  into the runner's JSONL sink synchronously per the Phase 1
  contract. The `mirror_marker` is written from the main
  thread at the top of `run()` after `transport.connect()` has
  returned (so `_env_step_meta.env_step` is populated). The
  reader thread is already running at this point but has only
  delivered the initial connect's response (not any mutator
  events — those fire during the loop), so the JSONL sink's
  write is uncontended. The mirror_marker lands as the first
  record in `world_event.jsonl` ahead of any `env_reset`
  records the env-server emits during the first step. The
  ordering is incidental, not contracted; tests assert on
  presence and payload shape, not on file position. Worth
  flagging because it's a sequence point a future runner
  refactor (e.g., emitting the marker pre-connect or
  post-warmup) could move without test coverage catching it.
- *The `init_zero_scalar_column` script's output layout follows
  the runner's checkpoint convention, not a lesion-specific
  one.* The output_dir is meant to be a `ckpt-NNNNNN` directory
  under `runs/{run_id}/checkpoints/`, with a `weights.safetensors`
  plus the five sidecar files plus a (lesion-specific)
  `world_event.jsonl`. The runner's `load_checkpoint` reads
  weights, optimizer_state, rng_state, telemetry_offsets,
  schema_version, replay_meta — all five sidecars — and does
  *not* read `world_event.jsonl` from the checkpoint dir
  (the runner's world_event sink writes to `telemetry_dir/
  world_event.jsonl` separately). So the script's emitted
  world_event is documentation-only; it lives alongside the
  checkpoint for traceability but plays no role in the runner's
  load path. This is the right shape for the script's purpose
  (a lesioned checkpoint that loads cleanly into a fresh
  runner) but it does mean the mutation-time mirror_marker is
  *not* part of any subsequent runner-time telemetry stream;
  Phase 11's faithfulness verifier and Phase 8's reader will
  need to look in two places (the runner's
  `telemetry/world_event.jsonl` AND the checkpoint dir's
  `world_event.jsonl`) when resolving the lesion provenance of
  a Probe 2 round running off a lesioned checkpoint. The
  journal flags this so the Phase 8 / 11 build phases know to
  expect the two-source pattern.

### What's now closed

- The five lesion kinds' contracts are settled: `RunnerConfig.
  lesion_kind` carries the six-value enum, with the three
  Probe-1.5-specific kinds operating at the world model
  (`disable_self_prediction`), views.split
  (`zero_or_randomize_scalar`), or as a checkpoint mutation
  (`init_zero_scalar_column`); the two v1 kinds operate at the
  ensemble's constructor (`ensemble_k1`) or the ensemble's
  disagreement methods (`ensemble_constant`).
- The `world_event` `mirror_marker` record shape carries the
  Phase 1.5 Phase 5 convention forward: `event_type=
  "mirror_marker"`, `source="system"`, `payload.lesion_kind`
  always present, `payload.lesion_variant` and
  `payload.lesion_empirical_min/_max` present only for
  `zero_or_randomize_scalar`, and the
  `init_zero_scalar_column` mutation-time record carries
  additional `source_checkpoint` /
  `actor_first_layer_key` / `actor_input_dim` /
  `column_start_index` / `column_moments_before/after`
  fields for provenance.
- The `scripts/probe2_lesion_init_zero_scalar_column.py` CLI
  conventions: `--source-checkpoint <path>`, `--output-dir
  <path>`, `--dry-run` (short-circuits before any file write),
  `--run-id` / `--checkpoint-id` (optional stamping for the
  emitted world_event). Module exposes
  `mutate(source_checkpoint, output_dir, dry_run, run_id,
  checkpoint_id) -> MutationResult` for programmatic use.
- The non-lesion path's byte-identity to a no-lesion-plumbing
  reference is preserved: at `lesion_kind=None` with fixed
  `torch.manual_seed(0)`, two runs produce identical
  telemetry on every deterministic field. The lesion plumbing
  introduced no behavioral change on the un-lesioned path.
- The autograd-preservation pattern for "force a substrate
  signal to a constant value" is committed: multiply the
  computed signal by `0.0` rather than returning a fresh
  zero tensor. Future lesion candidates with this shape
  inherit the discipline.

### What's now newly open

- *Whether the `init_zero_scalar_column` lesion's evaluation
  actually reproduces Phase 7's u/d regime collapse vs Phase
  7.5's L/R regime.* This is the substantive empirical
  question Phase 4's scaffolding makes testable but does not
  itself answer. The lesion zeros the small-Gaussian column
  that produced the L/R bipolar end-state in Phase 7.5; the
  Phase 12 calibration smoke (or a dedicated counterfactual
  probe at Phase 13) running the lesioned actor against the
  cached env will produce the action-distribution evidence.
  If the lesioned actor's late-trajectory regime collapses
  back to the u/d bipolar Phase 7's zero-init produced, the
  capacity-as-init-shape distinction Phase 8 surfaced is
  confirmed at the calibration level (synthesis §2.4 element
  4 + tension (l)). If it stays at L/R, something else is
  shaping the regime and the column-init-determined behavior
  reading weakens. Phase 4's job is to make the question
  legible; Phase 12 (or 13) produces the data.
- *The Phase 12 smoke decision on which lesion to run by
  default.* Synthesis §6 names the three Probe-1.5-specific
  lesions in the recommended order
  (`disable_self_prediction` → `init_zero_scalar_column` →
  `zero_or_randomize_scalar`); the smoke build phase decides
  whether the smoke runs all three sequentially, picks one
  per surface, or runs the lesion targeting the held-out
  criterion's surface (which under the v2 default is
  `behavior_side_scalar_conditioning`, pointing at
  `zero_or_randomize_scalar` as the natural smoke choice).
  Phase 4 leaves the decision open; the Phase 12 build phase
  decides with the smoke's first contrast results in hand.
- *The two-source pattern for mirror_marker resolution at
  Phase 8 / 11.* Phase 4's `init_zero_scalar_column`
  emission lands in the checkpoint dir's `world_event.jsonl`,
  not the runner's telemetry/world_event.jsonl. Phase 11's
  faithfulness verifier and Phase 8's reader prompts will
  need to read both sources when resolving lesion provenance
  for runs starting from a lesioned checkpoint. The verifier's
  per-baseline rule (Phase 11) and the digest's mirror_marker
  surfacing (Phase 8) are the implementation points; Phase 4
  flags the two-source semantic so neither phase has to
  rediscover it.
- *The empirical bounds for the `randomize` variant when used
  in real Probe 2 runs.* The defaults `[0.0, 1.0]` cover the
  cosine loss form's range; the synthesis §2.4 element 4
  spec calls for empirical-distribution-derived bounds (the
  source run's per-step `self_prediction_error_t`
  distribution's empirical min/max). Phase 4's runner config
  carries the bounds as floats; what's missing is the
  cached-run analysis step that produces the empirical
  bounds from the source run's parquet. The Phase 6
  conditioning-analysis module already computes
  `empirical_scalar_mean/sigma/range` from the source run;
  threading those into a real Probe 2 lesion run's
  RunnerConfig is a small piece of glue the calibration
  driver (Phase 12) will land. Phase 4 leaves the float
  fields; the glue at Phase 12 fills them.

---

## Phase 5 — pre-registration runner glue (2026-05-08)

Phase 5 lands the runner-side glue around the pre-registration sink
that Phase 0 built (synthesis §2.4 element 1; plan §2.5 + §4 row 5).
Phase 0 produced the `PreRegistration` model and the `PreRegSink`
JSONL writer in isolation; Phase 5 wires them into `Runner` so an
adversarial-pass orchestrator (Phase 8 — not yet built) can call
`runner.emit_pre_registration(record)` once before each adversarial
reading runs. The pre-registration is the structural counter against
the Gelman-and-Loken garden-of-forking-paths drift the synthesis names
as load-bearing; the runner glue is the integration point that makes
the discipline actually executable inside a Probe 2 round. Phase 5 is
parallelizable with Phases 3 / 4 / 6 / 11 once Phase 0 lands;
sequencing it after Phase 4 keeps the calibration protocol's
scaffolding building forward toward the Phase 8 prompt-builder and
Phase 12 smoke without backtracking.

### What was built

- `kind/training/runner.py` extended:
  - `RunnerConfig` gains `pre_reg_dir: Path | None = None`. The
    Probe 2 convention is `runs/{run_id}/pre_reg/` as a sibling of
    `telemetry_dir`; adversarial-pass orchestration sets this
    explicitly. Probe 1 / Probe 1.5 callers leave it at the default
    `None` and inherit no behavioral change — no sink is constructed,
    no directory is created. The opt-in design is symmetric with the
    optional `env_server` parameter on `Runner.__init__` (a feature
    available when configured, absent when not).
  - `Runner.__init__` constructs `self._pre_reg_sink: PreRegSink |
    None` immediately after the four telemetry sinks: `PreRegSink(
    config.pre_reg_dir)` when the field is set, `None` otherwise.
    The sink's constructor creates the directory if missing and
    opens `pre_reg.jsonl` for append. When `None`, no filesystem
    artifact is created — verified by
    `test_runner_without_pre_reg_dir_has_no_sink`.
  - New public method `Runner.emit_pre_registration(record:
    PreRegistration) -> None`: checks `self._closed` first (matches
    the existing `run()` closed-state guard), then checks
    `self._pre_reg_sink is not None`, then delegates to the sink's
    `write`. Both raise `RuntimeError` with a clear message naming
    the missing precondition. The append-only JSONL behavior comes
    for free from the underlying `PreRegSink.write`.
  - `Runner.close()` extended: after the existing four-sink close
    loop, a separate `if self._pre_reg_sink is not None: try:
    sink.close() except: pass` block closes the pre-reg sink with
    the same best-effort discipline. The transport close still
    happens last. `PreRegSink.close()` is idempotent (verified at
    Phase 0 by `test_pre_reg_sink_close_is_idempotent`), so
    `runner.close()` remains idempotent.
- `docs/workingjournal/probe2_templates/pre_registration.md` (~120
  lines): hand-fill markdown scaffold the builder copies once per
  adversarial pass before the reading runs. The template is the
  prose form the builder reads later; the structured JSONL the
  sink writes is the machine-readable form. Slot ordering matches
  the model field ordering in `kind/observer/pre_reg.py`. Every
  required v2 field (the fifteen non-defaulted fields on
  `PreRegistration`) appears in the template prose with the
  load-bearing slots — `falsifiers`, `asymmetry_of_access`,
  `substrate_decisions_off_table`, `column_init`,
  `new_actor_readable_interfaces_added` — flagged with the synthesis
  citations the discipline rests on. Schema-validator-side
  enforcement (per-criterion completeness; active vs held-out
  disjointness) is described inline so the builder filling the
  template knows what the JSONL construction will reject.
- `tests/test_pre_reg_runner.py` (10 tests, all passing):
  - `test_runner_without_pre_reg_dir_has_no_sink`: opt-in semantic
    enforced; `runner._pre_reg_sink is None`; no `pre_reg/`
    directory created.
  - `test_runner_with_pre_reg_dir_constructs_sink`: opt-in path;
    sink constructed; directory created.
  - `test_emit_pre_registration_writes_to_pre_reg_jsonl`: single
    emit produces one JSONL line at `<pre_reg_dir>/pre_reg.jsonl`;
    parsed record equals the input.
  - `test_emit_pre_registration_appends_multiple_records`: three
    sequential emits produce three JSONL lines in order; no
    overwrite, no merge.
  - `test_emit_pre_registration_raises_without_pre_reg_dir`: the
    opt-in semantic is enforced at the public API.
  - `test_emit_pre_registration_raises_after_close`: closed-runner
    guard at the public API.
  - `test_runner_close_closes_pre_reg_sink`: end-to-end close
    cascade verified by attempting `sink.write(record)` directly
    after `runner.close()` and catching `PreRegSinkClosedError`;
    pre-emitted record lands on disk (fsync ran during close).
  - `test_runner_close_is_idempotent_with_pre_reg_sink`: double
    close is a no-op on the second call.
  - `test_pre_registration_template_exists`: the template lives at
    the documented path and is non-empty.
  - `test_pre_registration_template_covers_all_v2_model_fields`:
    every non-defaulted v2 model field name appears in the
    template prose. The structural counter against template /
    model drift in future schema bumps.

### Test/mypy status

`pytest tests/test_pre_reg_runner.py`: 10 passed, 0 failed.
`pytest tests/test_pre_reg.py tests/test_lesion.py
tests/test_integration_smoke.py tests/test_integration_smoke_probe1_5.py`:
69 passed, 0 failed — the four most-adjacent test files (Phase 0's
sink-and-model isolation tests, Phase 4's lesion-scaffold tests, and
the two integration smokes that exercise `Runner.__init__` /
`Runner.close` against a real transport pair) regress cleanly.
Full suite `pytest tests/ --ignore=tests/test_smoke_mps_script.py`:
606 passed, 1 failed, 2 skipped (the API-key smokes); the single
failure is `tests/test_transport.py::test_barrier_queues_mutates_
and_drains_in_order`, an existing flaky transport-timing test on
`main` (verified by running with my edits stashed — same exception,
same line). Not Phase 5 territory.
`mypy --strict kind/training/ tests/test_pre_reg_runner.py`:
5 source files, no issues. Phase 5's bar — runner glue wires the
sink cleanly under both opt-in and default-off paths; the close
cascade closes the sink; the template covers every required model
field; mypy clean — clears.

### What surprised

Two things worth flagging.

- *The opt-in default-off design was the right shape, but the
  testing implication wasn't obvious upfront.* The tradeoff was
  between three designs: (i) eager sink construction always (every
  runner creates an empty `pre_reg.jsonl` even Probe 1 ones), (ii)
  required `pre_reg_dir` field (breaks every existing test fixture
  including `_make_runner_config` in `test_lesion.py` and
  `test_integration_smoke*.py`), (iii) optional with default `None`
  (Probe 1 / Probe 1.5 unaffected; Probe 2 callers opt in). Picked
  (iii) because the project-state phrase "Probe 1 implementation is
  settled; Probe 1.5 implementation is settled" is a load-bearing
  no-regression commitment, and (iii) is the only design that
  preserves byte-identity on the un-opted-in path. The journal
  flags this for Phase 8's adversarial-pass orchestration — the
  orchestrator must remember to pass `pre_reg_dir` explicitly
  when constructing the runner; forgetting it surfaces as a
  `RuntimeError` at the first `emit_pre_registration` call rather
  than as silent no-op telemetry. The error message names the
  missing field by literal name, so the diagnosis from a stack
  trace is one read.
- *The "verify the sink is actually closed" test required reaching
  past the public API.* `Runner.emit_pre_registration` checks
  `self._closed` first and raises `RuntimeError` before reaching
  the sink, so an after-close `runner.emit_pre_registration` call
  doesn't actually verify the sink itself was closed — it verifies
  the runner-level guard. To verify the close cascade end-to-end,
  the test holds a reference to `runner._pre_reg_sink` *before*
  calling `runner.close()`, then attempts `sink.write(record)`
  directly afterward and catches `PreRegSinkClosedError`. This
  pokes private state, which is normally a smell, but the
  alternative is removing the runner-level guard from
  `emit_pre_registration` (which would weaken the diagnostic
  shape). The journal flags this because future runner-level
  guards on internal state should follow the same pattern: keep
  the guard, test the cascade by reaching past it once.

### What's now closed

- The runner's pre-registration glue contract is settled:
  `RunnerConfig.pre_reg_dir: Path | None = None` is the
  configuration surface; `Runner.emit_pre_registration(record:
  PreRegistration) -> None` is the public method; the sink lives
  at `<pre_reg_dir>/pre_reg.jsonl` (filename is
  `kind.observer.pre_reg.PRE_REG_FILE` from Phase 0); the close
  cascade closes the sink via the same best-effort discipline as
  the four telemetry sinks.
- The opt-in default-off semantic: Probe 1 / Probe 1.5 runners
  inherit no behavioral change. The non-lesion-byte-identity
  property Phase 4 verified extends through Phase 5 — no new
  filesystem artifact, no new sink, no new `world_event`
  emission on the un-opted-in path.
- The pre-registration template's location and shape:
  `docs/workingjournal/probe2_templates/pre_registration.md`,
  hand-fill prose, slot ordering matches the model field
  ordering, the schema-validator's per-criterion completeness
  rule is described inline so the builder filling the template
  knows what the JSONL construction will reject.
- Phase 0's "no Phase 0 runner glue" newly-open question (Phase 0
  newly-open §3 — *"There is no Phase 0 runner glue (the runner
  doesn't import the new sinks, no `runs/{run_id}/pre_reg/`
  layout). The runner glue is open until Phase 5 (the
  pre-registration sink phase) lands."*) closes here. The Phase 0
  closure note is implicit in this entry; the canonical record is
  the Phase 5 contract above.

### What's now newly open

- *Whether Phase 8's adversarial-pass driver needs a higher-level
  wrapper around `emit_pre_registration` that constructs the
  `PreRegistration` from a config + the criterion registry.*
  Phase 5's hook is intentionally low-level: the caller passes a
  fully-constructed `PreRegistration`. For ad-hoc one-off
  pre-registrations the builder fills the template, constructs
  the model directly, and calls the hook. For the Phase 8
  orchestration loop — running adversarial passes against
  multiple checkpoints with the same active-set / held-out-set —
  there's a likely refactor where the orchestrator carries a
  `PreRegistrationConfig` (the round's framing) plus a per-pass
  `(run_id, timestamp_ms, expected_outcome)` triple, and a small
  factory composes the `PreRegistration` from
  `kind.mirror.criteria` (Phase 7's registry) plus the round's
  config. Whether this factory lives in `kind.mirror.adversarial`
  (Phase 8) or as a standalone helper in `kind.observer.pre_reg`
  is open; Phase 8's build phase decides with the orchestration
  loop's first concrete shape in hand. Phase 5 leaves the hook
  low-level so the choice stays in Phase 8's hands.
- *Whether the template should be split into per-surface sections
  or kept unified.* The current template is unified — one
  document covering all three reading surfaces. The
  `expected_outcome_per_surface` dict is the only field that
  formally splits by surface; everything else (criteria,
  falsifiers, signal mappings) is keyed by criterion id, with
  the per-criterion `reading_surfaces_per_criterion` dict
  carrying the surface assignment as a per-criterion property.
  An alternative shape — three sub-templates, one per surface —
  would make the per-surface framing more visually explicit at
  the prose level but would require the builder to fill three
  documents per round and would create coupling tension with
  criteria that span surfaces (e.g., `equanimity_perturbation_
  recovery` reads at `substrate_side` AND `head_internal`).
  Phase 5 keeps the unified shape; Phase 8's first real
  pre-registration fill is the empirical test of whether the
  unified prose flow is workable. If filling the unified
  template proves awkward, the per-surface split is a small
  refactor (the JSONL form is unaffected; only the template
  and the journal-side prose change).
- *Whether the `pre_reg_dir` convention as a sibling of
  `telemetry_dir` should be enforced or merely conventional.*
  Phase 5 leaves `pre_reg_dir` as a free `Path` field — the
  caller passes whatever directory; the runner doesn't validate
  the relationship to `telemetry_dir`. This is consistent with
  the existing pattern (`telemetry_dir`, `checkpoints_dir` are
  also free `Path` fields, conventionally siblings under
  `runs/{run_id}/` but not enforced as such). The convention
  lives in the calibration driver (Phase 12's smoke) and the
  journal entries; the runner is purely transactional. If a
  future phase needs to enforce the layout (e.g., for digest
  cross-references that assume the sibling structure), the
  enforcement point is the calibration driver, not the runner.

---

## Phase 6 — criterion registry (2026-05-08)

Phase 6 builds the typed shelf that holds frozen mirror criteria as
queryable records: an empty structural module with strong invariants.
Phase 7 will populate it with the three v2 criteria (reflexive
attention, equanimity perturbation recovery, second-order volition);
Phase 8's adversarial-pass orchestrator will read from it. Phase 6 is
structural in the same sense Phase 0 was — type design, validation,
lookup, no philosophical commitments. The philosophical commitments
land in Phase 7. Phase 6's job is to make Phase 7 (and Phase 8)
impossible to do wrong: the shapes refuse the wrong thing
structurally, not by convention. Phase 6 is parallelizable with
Phases 3 / 4 / 5 (already done) and Phase 11; it must complete before
Phase 7 (which populates the registry) and Phase 8 (which reads from
it).

### What was built

- `kind/mirror/registry.py` (~377 lines, mypy strict, Pydantic v2,
  every model `frozen=True` + `extra="forbid"`):
  - `TelemetrySurface(str, Enum)` — the read-from surfaces a criterion
    may declare. Members: `AGENT_STEP_OBSERVABLE`,
    `AGENT_STEP_INTERNAL`, `DREAM_ROLLOUT`, `REPLAY_META`.
    `world_event` is **deliberately not** a member; the membrane
    decision in the design notes walls `world_event` off from the
    agent process, and the mirror reading it would create the
    cross-membrane dependency the design notes prohibit. The split
    between `AGENT_STEP_OBSERVABLE` and `AGENT_STEP_INTERNAL` carries
    the asymmetry-of-access boundary at the type level: PolicyView
    reads observable, TelemetryView reads internal.
  - `ReadingSurface(str, Enum)` — the read-at surfaces (where the
    mirror produces a reading): `SUBSTRATE_SIDE`, `HEAD_INTERNAL`,
    `BEHAVIOR_SIDE`. Enum *values* match the strings already used by
    Phase 0's `kind.mirror.structured.ReadingSurface` Literal and
    Phase 5's `PreRegistration.expected_outcome_per_surface` /
    `reading_surfaces_per_criterion` dict keys, intentionally — a
    future migration of those keys from raw strings to enum members
    is a type tightening, not a value change. The TODO is documented
    on the enum's docstring; the migration is open (see "What's now
    newly open" §1).
  - `Criterion(BaseModel)` — the frozen record. Fields: `id`
    (snake_case, ≤40 chars, regex-validated), `display_name`
    (non-empty), `framework` (snake_case, regex-validated),
    `description` (non-empty), `telemetry_surfaces:
    frozenset[TelemetrySurface]` (non-empty), `reading_surfaces:
    frozenset[ReadingSurface]` (non-empty), `falsifier` (non-empty
    free text), `signal_mappings: dict[str, str]` (snake_case keys,
    non-empty values), `held_out: bool = False`. **No write-surface
    field exists** — the absence is structural. A future contributor
    cannot add one without amending the model and tripping the
    structural test. Frozen at the model level: assignment after
    construction raises `ValidationError`.
  - `CriterionRegistry(BaseModel)` — frozen tuple of `Criterion`
    with read-only lookups: `get(id) -> Criterion` (raises
    `KeyError` with the known-ids list on miss), `has(id) -> bool`,
    `all_ids() -> frozenset[str]`, `active() -> tuple[Criterion,
    ...]`, `held_out() -> tuple[Criterion, ...]`, `by_framework(name)
    -> tuple[Criterion, ...]`, `by_reading_surface(surface) ->
    tuple[Criterion, ...]`. Every return is immutable; no method
    returns a writer, callback, or anything invokable against Io's
    input space. Three model-level validators: non-empty registry
    (with the documented sanctioned-empty exception), unique ids
    (with both occurrences named in the error), and the
    held-out-flag-is-the-partition check (defense against future
    refactors that introduce a third state). Construction order
    preserved across `active()` / `held_out()` / `by_framework()` /
    `by_reading_surface()`.
  - `EMPTY_REGISTRY: Final[CriterionRegistry]` — the single
    sanctioned empty form, constructed via Pydantic's
    `model_construct` to bypass the non-empty validator at module
    load. Tests that need an empty registry import this constant;
    `CriterionRegistry(criteria=())` directly raises by design.
- `kind/mirror/__init__.py` extended: re-exports
  `EMPTY_REGISTRY`, `Criterion`, `CriterionRegistry`,
  `ReadingSurface`, `TelemetrySurface`. Phase 0's `structured.py`
  was already imported elsewhere by full path; the package is now a
  proper public surface for the registry.
- `tests/test_criterion_registry.py` (~500 lines, 18 tests, all
  passing). Tests are sectioned with numbered comment headers
  matching the contract above:
  - (1)–(5) field-level validation: id regex / length, framework
    regex, telemetry_surfaces non-empty, reading_surfaces
    non-empty, signal_mappings keys snake_case + values non-empty.
  - (6) `Criterion` is frozen — assignment to `id`, `framework`,
    `held_out` after construction raises `ValidationError`.
  - (7) `Criterion` has no write-surface field — the structural
    assertion. Iterates the model's fields and refuses any name in
    `_FORBIDDEN_FIELD_NAMES = {"write_surface", "writer",
    "callback", ...}`. A future contributor adding such a field
    trips here.
  - (8)–(15) registry lookup: `get` returns the right record,
    `get` raises with the known-ids list on miss, `has` returns
    bool without raising, duplicate ids rejected with both
    indices named, `active()` / `held_out()` partition correctly,
    registration order preserved across both, `by_framework`
    returns only matching criteria, `by_reading_surface` handles
    multi-surface criteria (a criterion declaring two surfaces
    appears in the result for each).
  - (16) `CriterionRegistry` is frozen.
  - (17) `EMPTY_REGISTRY` is the only sanctioned empty form;
    direct construction with `criteria=()` raises.
  - (18) No method on `CriterionRegistry` returns a writer-shaped
    type — the structural assertion of the read-only invariant.
    Walks every public method's return-type annotation and refuses
    any whose name resolves to a callable / writer-shape (the
    accepted shapes are `Criterion`, `bool`, `frozenset[str]`,
    `tuple[Criterion, ...]`).
  - All fixtures use mock framework names like `test_framework_a`
    constructed inline; **no Phase 7 criterion is imported**, so the
    test file passes without Phase 7 existing.

### Test/mypy status

`pytest tests/test_criterion_registry.py`: 18 passed, 0 failed.
`pytest tests/`: 627 passed, 2 skipped (the API-key smokes), 0
failed — the previously-flagged flaky
`test_transport.py::test_barrier_queues_mutates_and_drains_in_order`
did not fail in this run; the full suite is clean.
`mypy --strict kind/mirror/ tests/test_criterion_registry.py`:
5 source files, no issues. No existing tests modified; no
production code outside `kind/mirror/` modified.

### What surprised

Two things, one structural and one Pydantic-mechanical.

- *The structural read-only test was easier to write than to name.*
  The load-bearing constraint — "the registry is part of the
  mirror's one-way data plane; no method returns a writer" — has a
  natural unit-test shape (walk every public method's return-type
  annotation; refuse anything writer-shaped), but no natural name.
  I settled on `test_registry_no_method_returns_writer` and a
  forbidden-name list (`{"write_surface", "writer", "callback",
  "sink", "emit", "send", "publish"}`) checked structurally rather
  than semantically. The test catches the obvious wrong direction
  but not a subtler one — a method returning a `Criterion` whose
  `description` happens to be a callable's qualified name, for
  example, would pass. The structural-vs-semantic gap is fine at
  Phase 6 because Phase 8 (which actually reads the registry) is
  where the wrong direction would manifest, and Phase 8 will
  re-test the invariant against the populated registry. This is the
  shape that the "What's now newly open" §2 question — whether to
  promote the test to a shared mirror-side helper — turns on.
- *The empty-registry exception cost more design than expected.*
  Three options surfaced: (a) make the non-empty validator
  conditional on a constructor flag (e.g., `is_test=True`), (b) use
  a separate type (`_EmptyRegistry` distinct from
  `CriterionRegistry`) that tests import, (c) construct the empty
  form via Pydantic's `model_construct` which bypasses validators
  at load time, and stash it in a module-level `Final` constant.
  Picked (c) because (a) leaks test concerns into the production
  type, (b) splits the type and breaks `isinstance` checks the
  Phase 8 orchestrator will plausibly need, and (c) keeps the
  production constructor's guarantees intact (every
  programmatically-constructed registry has at least one
  criterion) while giving tests a single sanctioned import. The
  `Final` annotation pins the constant's identity. The cost is one
  bypassed validator at module load; the benefit is a single
  named entry point that Phase 7's three-criterion registry can
  shadow without the empty form leaking into production code.

### What's now closed

- The criterion-registry contract: `TelemetrySurface` and
  `ReadingSurface` enums; the `Criterion` field set (no
  write-surface field, no callback field, no writer field);
  `CriterionRegistry`'s read-only lookup methods (every return
  type immutable); the model-level validators (non-empty registry,
  unique ids, held-out partition); the field-level validators
  (snake_case ids and frameworks, ≤40 char ids, non-empty signal
  mapping descriptions, non-empty `telemetry_surfaces` and
  `reading_surfaces`); `EMPTY_REGISTRY` as the single sanctioned
  empty form. Phase 7 fills the registry; Phase 7 cannot change
  the shape without amending Phase 6's tests.
- The structural read-only invariant has a test pinning it
  (`test_criterion_does_not_have_write_surface_field` and
  `test_registry_no_method_returns_writer`). A future contributor
  adding a write surface or a writer-returning method trips here
  before reaching Phase 8.
- No Phase 0–5 newly-open question resolves at Phase 6. The Phase
  6 work is upstream of Phase 7's criterion population; downstream
  questions (e.g., whether `PreRegistration` should look up
  criterion ids by registry rather than carrying them inline) are
  still open and remain Phase 7 / Phase 8 territory per the plan.

### What's now newly open

- *Whether `ReadingSurface` should migrate
  `PreRegistration.expected_outcome_per_surface` and
  `PreRegistration.reading_surfaces_per_criterion` keys from raw
  strings to enum members, and when.* The enum values match the
  Literal strings exactly, so the migration is a type tightening
  with no value change and no schema-level JSON serialization
  change (str-valued enum members serialize identically). The case
  *for* migrating: a typo in a dict key currently passes Phase 0's
  `Literal` check at the type level but reaches the validator
  late; an enum member is import-time. The case *against*: Phase
  0's Literal is shared with `kind.mirror.structured` and Phase 5
  in-flight tests rely on the string-keyed shape. Phase 6 leaves
  the migration as a `TODO(phase 7+)` on the enum's docstring; the
  natural moment is when Phase 7 populates the registry and the
  three concrete criteria's `reading_surfaces_per_criterion`
  entries are first written through the model — that's the first
  point where the typo-guard payoff is concrete. If Phase 7 picks
  the migration up, the existing string-form Phase 5 tests need a
  one-shot rewrite. If Phase 7 defers, the migration moves to
  Phase 8 or later and the TODO stays.
- *Whether the read-only invariant tests
  (`test_criterion_does_not_have_write_surface_field` and
  `test_registry_no_method_returns_writer`) should be promoted to
  a shared mirror-side test helper that Phases 7 / 8 reuse.*
  Phase 7 adds three criteria; Phase 8 adds the adversarial-pass
  orchestrator that reads the registry and produces prompt
  fragments. Both expand the surface area where a writer could
  sneak in (a Phase 7 criterion subclassing `Criterion` with a
  write-surface field; a Phase 8 orchestrator method returning a
  callback that closes over the agent process's input space). The
  argument *for* a shared helper — call it
  `assert_no_writer_shape(model_or_class)` and lift the
  forbidden-name list to the helper — is that the invariant is
  the same one in both places. The argument *against* is
  premature abstraction: Phase 8's check will plausibly be
  semantic (does the orchestrator's method ever produce a value
  that flows back into Io's input space?) rather than structural
  (does the return-type name match a forbidden-name list?), and a
  shared helper that tightens around Phase 6's specific shape
  may not fit. Phase 6 keeps the tests local; Phase 7's first
  build is the empirical test of whether the helper's shape is
  obvious in retrospect.
- *Whether `Criterion.signal_mappings` should be a typed model
  rather than a raw `dict[str, str]`.* Phase 6's choice was
  `dict[str, str]` with snake_case-key + non-empty-value
  validation, on the principle that the structural shelf doesn't
  need Phase 7's operational content modeled. But Phase 7's
  three v2 criteria each declare specific signals (e.g.,
  reflexive attention's `attention_uniqueness_t` and
  `policy_outcome_correlation_t`), and the per-signal schema
  references (which `(h_t, z_t)` slice; which `AgentStep` field)
  are part of the criterion's operational definition. A typed
  `SignalMapping(BaseModel)` with fields `name`, `description`,
  `telemetry_field_path`, etc. would force Phase 7 to be explicit
  about the field references and would let Phase 8's orchestrator
  validate them at registry-load time rather than at prompt-build
  time. The argument *against* is that Phase 6's job is the
  shelf, and tightening the signal-mapping shape now forecloses
  on Phase 7's design space before the criteria are written.
  Phase 6 keeps `dict[str, str]`; Phase 7 decides whether the
  typed-mapping refactor pays its way or whether the raw dict is
  enough for the three v2 criteria.

### Out of scope (preserved from the plan)

The three v2 criteria themselves; wiring `PreRegistration` to look
up criterion ids through the registry; the adversarial-pass
orchestrator; any prompt-builder or LLM-call code; any change to
`Runner`, `RunnerConfig`, or telemetry sinks. Phase 6 touched only
`kind/mirror/registry.py`, `kind/mirror/__init__.py`, and
`tests/test_criterion_registry.py`; no other files were modified.

---

## Phase 7 — frozen criteria v2 (binding contract) (2026-05-11)

Phase 6 built the shelf. Phase 7 puts the three books on it — and, in
the same phase, replaces the shelf's `dict[str, str]` signal-mapping
slot with a typed `SignalMapping` model whose field-path references
resolve at registry-load against the actual telemetry schemas. The
two halves are deliberately the same phase: the typed shape only
earns its keep against three concrete criteria, and the three
criteria are the empirical test of whether the shape was right. They
were the right shape; the field-path resolution caught nothing
during the build (no typos survived to the test stage), but the
*forcing function* of having to name a surface and a resolvable path
for each signal sharpened the criteria themselves — see the
refactor-record section.

Phase 7 is the philosophically loaded phase. The operational mapping
from framework prose (`Kind_frameworks.md` on Buddhist phenomenology
and Frankfurt) to criterion code is the freeze-record below; the
charter discipline applies — criteria are not updated in response to
system behavior, and a future revision requires a journal entry
naming the external framework that prompted the change. The
criteria-as-code form in `kind/mirror/criteria_v2.py` is the canonical
commitment; the `Kind_frameworks.md` prose is the *source* the
operational definitions are drawn from, not a parallel commitment.

Phase 7 unblocks Phase 8 (the adversarial-pass orchestrator), whose
first concrete need is the populated registry: `V2_REGISTRY`,
consumable without further structural change.

### What was built

- `kind/mirror/registry.py` (~590 lines, mypy strict, Pydantic v2,
  every model `frozen=True` + `extra="forbid"`):
  - **`SignalMapping(BaseModel)`** — the typed replacement for the
    Phase 6 `dict[str, str]` value. Fields: `name` (snake_case,
    ≤40 chars, regex-validated against the same pattern as
    `Criterion.id`), `description` (non-empty multi-line free text —
    the operational record of what the signal measures and which
    *class* of statistic Phase 8 will compute), `telemetry_surface:
    TelemetrySurface` (a single surface — cross-surface composites
    are multiple `SignalMapping` records combined at prompt-build
    time), `field_path: str` (a dotted path into the surface's
    Pydantic model), `slice_spec: str | None = None` (an optional
    NumPy-style slice expression — digits, colons, commas only;
    empty string rejected; `None` means "no slice"). Validation that
    runs at construction: name regex/length; description non-empty;
    `slice_spec` regex (`^[0-9:,]+$`); `field_path` shape regex
    (identifier components, no whitespace, no leading/trailing dot)
    *then* semantic resolution against `_SURFACE_TO_MODEL[surface]`
    via `_resolve_dotted_path`, which walks the dotted path against
    the model's fields and raises naming the bad component and the
    available fields at that level, or "not a nested model" for an
    intermediate component that can't be descended into (the
    telemetry schemas have no nested models, so the dotted-path
    walker's nesting branch is correct-but-unexercised by the real
    schemas — `kl_per_dim_t.mean` trips the "not a nested model"
    branch). No `SignalMapping` field is a writable handle, callback,
    sink, or anything invokable against Io's input space — the
    structural read-only invariant of the mirror's one-way data
    plane; pinned by `test_signal_mapping_no_writer_shape`.
  - **The asymmetry-of-access allowlists** —
    `_SURFACE_TO_MODEL: dict[TelemetrySurface, type[BaseModel]]`
    (both `AGENT_STEP_*` surfaces → `AgentStep`; `DREAM_ROLLOUT` →
    `DreamRollout`; `REPLAY_META` → `ReplayMeta`),
    `_OBSERVABLE_FIELDS: frozenset[str]` (`action_t`,
    `action_logprob_t`, `policy_entropy_t`, `obs_hash_t`,
    `self_prediction_error_t` — the channels Io's PolicyView reads,
    including the single Watts-heuristic scalar) and
    `_INTERNAL_FIELDS: frozenset[str]` (`h_t`, `z_t`, `q_params_t`,
    `p_params_t`, `kl_per_dim_t`, `kl_aggregate_t`, `recon_loss_t`,
    `encoder_embedding_t`, `intrinsic_signal_t`, `self_prediction_t`
    — substrate state only TelemetryView reads; Io reads only the
    scalar *error* of `self_prediction_t`, never the vector). A
    `SignalMapping` declaring `AGENT_STEP_OBSERVABLE` may only root
    its `field_path` in `_OBSERVABLE_FIELDS`; declaring
    `AGENT_STEP_INTERNAL`, only in `_INTERNAL_FIELDS`. A mapping
    declaring `AGENT_STEP_OBSERVABLE` with `field_path="h_t"` is
    rejected with an error naming the asymmetry-of-access boundary;
    so is `AGENT_STEP_INTERNAL` with `field_path="action_t"`. Envelope/
    indexing fields (`schema_version`, `run_id`, `checkpoint_id`, `t`,
    `episode_id`, `step_in_episode`, `wallclock_ms`) are in *neither*
    allowlist by design — a criterion's signal derives from a
    substantive channel, not from a record's indices; index fields
    appear in citation ranges (Phase 8), not in `SignalMapping` field
    paths. The allowlists are documented at the top of the file with
    comments naming the design-notes section that motivates the split
    (the agent/mirror/observer layers section, "asymmetry of access
    between Io and the mirror").
  - **`Criterion.signal_mappings` refactored** from `dict[str, str]`
    to `tuple[SignalMapping, ...]` (tuple, not frozenset — order is
    load-bearing for human reading and Phase 8 prompt construction;
    uniqueness enforced by a validator). New/changed validators:
    `signal_mappings` is non-empty; a raw `dict` is rejected by a
    `mode="before"` validator with a migration-pointing error ("now
    a tuple of SignalMapping records (Phase 7), not the dict[str, str]
    of Phase 6 … this guard exists so the migration cannot drift
    back"); all `name` values across the tuple are unique (duplicate
    raises with both indices named); a `mode="after"` model validator
    enforces that every `SignalMapping`'s `telemetry_surface` appears
    in the criterion's declared `telemetry_surfaces` (a criterion
    cannot reference a signal from a surface it didn't declare). The
    read-only-invariant docstring stays; `Criterion`'s docstring
    updated to reference the new shape.
  - `kind/mirror/registry.py` now imports `AgentStep`,
    `DreamRollout`, `ReplayMeta` from `kind.observer.schemas` at
    module top — the field-path resolution mechanism needs the
    Pydantic models. (`kind.observer.schemas` imports nothing from
    `kind.mirror` at module top, so no cycle.)
- `kind/mirror/criteria_v2.py` (new, ~340 lines, mypy strict) —
  the three v2 criteria as module-level `Final` constants plus
  `V2_REGISTRY: Final[CriterionRegistry]` composing them in
  load-bearing order. Per-criterion records below.
- `kind/mirror/__init__.py` — re-exports `SignalMapping`,
  `V2_REGISTRY`, `REFLEXIVE_ATTENTION`,
  `EQUANIMITY_PERTURBATION_RECOVERY`, `SECOND_ORDER_VOLITION`
  alongside the Phase 6 re-exports.
- `kind/observer/pre_reg.py` — the `ReadingSurface` migration (own
  section below): one import line changed
  (`kind.mirror.structured.ReadingSurface` Literal →
  `kind.mirror.registry.ReadingSurface` enum); no validator change;
  no schema-data-serialization change.
- `schemas/v0.3.0.json` — regenerated. The `PreRegistration` model's
  schema now carries a local `$defs/ReadingSurface` enum referenced
  by `$ref` from `expected_outcome_per_surface.propertyNames` and
  `reading_surfaces_per_criterion`'s array items (Pydantic emits a
  `$ref` for a str-enum where it inlined a Literal); the JSON-data
  serialization is byte-identical (str-valued enum members serialize
  as their string values). `StructuredClaim` / `StructuredReading` /
  `JudgeRuling` are unchanged (still the Phase 0 Literal).
- `tests/test_signal_mapping.py` (new, 18 tests, all passing) —
  name regex; description non-empty; `slice_spec` regex (valid
  `:32` / `64:128` / `:` / `0,1,2` / `16:32,48:64`; invalid empty /
  letters / whitespace / brackets; `None` accepted); frozen +
  `extra="forbid"`; `field_path` resolves against
  `AGENT_STEP_INTERNAL` / `DREAM_ROLLOUT` / `REPLAY_META`;
  observable↔internal split enforced both directions with the error
  naming the asymmetry-of-access boundary; `no_writer_shape`
  structural assertion (third instance of the pattern — see "what's
  now closed"); `Criterion.signal_mappings` non-empty / unique names
  / surfaces ⊆ criterion surfaces / order preserved; dotted-path
  resolution (unknown component names available fields; intermediate
  non-model component named with the remaining path); the Phase 6
  dict form rejected with a "Phase 7"/"SignalMapping" migration
  hint; plus `slice_spec` stored verbatim round-trips, and
  `field_path` shape rejected before semantic resolution.
- `tests/test_criteria_v2.py` (new, 13 tests, all passing) —
  registry has three criteria / active set is the two
  Buddhist-phenomenology criteria / held-out set is the one
  Frankfurt criterion; each criterion's framework, reading surfaces,
  telemetry surfaces, signal names, signal field paths, signal
  surfaces, and non-empty falsifier asserted, plus a `_resolves`
  smoke that round-trips every `SignalMapping` through
  `model_validate(model_dump())` so a future schema rename trips
  here even if the model-level tests don't notice; `by_framework`
  buddhist/frankfurt; `by_reading_surface` substrate/head/behavior;
  the registry composition matches the module constants
  (`get(id) is CONSTANT`, id tuple order); the v2 criteria and the
  registry are frozen.
- `tests/test_criterion_registry.py` — updated for the refactor: a
  `_signal_mapping` fixture helper; `_criterion`'s default
  `signal_mappings` is now `(_signal_mapping(),)`; the old
  dict-key-validation test replaced by `test_criterion_signal_mappings_tuple_form`
  (tuple accepted in order; empty tuple rejected; the dict form
  rejected with a migration hint). The two structural read-only
  tests (`test_criterion_does_not_have_write_surface_field`,
  `test_registry_no_method_returns_writer`) are unchanged and pass.
- `tests/test_pre_reg.py`, `tests/test_pre_reg_runner.py` — the
  surface-keyed dicts (`expected_outcome_per_surface`,
  `reading_surfaces_per_criterion` values) now use `ReadingSurface`
  enum members; one new test (`test_expected_outcome_per_surface_coerces_bare_strings`)
  documents that a caller passing bare strings still works (Pydantic
  coerces); the rejection tests (invalid surface key) still pass
  unchanged. `test_pre_registration_is_frozen`'s now-unused
  `# type: ignore[misc]` removed (matching the no-ignore convention
  already used in `test_criterion_registry.py`).

#### Reflexive attention (Buddhist phenomenology) — ACTIVE

- **Source.** `Kind_frameworks.md` §"Buddhist phenomenology":
  "Reflexive attention. The observer observing itself, as in
  vipassana — not by introspection from outside, but by awareness
  turning toward awareness", and the candidate criterion "does
  anything function as an observer of the agent's own processing?".
- **Operational mapping.** *Within-latent reference*: structure in
  `(h_t, z_t)` where some component of the recurrent state varies as
  a function of *other* components of the recurrent state, independent
  of environmental input. The "observer" is a part of the latent
  state that tracks the rest of the latent state rather than tracking
  the world. Concretely: a stable subspace of `h_t` whose trajectory
  is predictable from the rest of `h_t` (and lagged `h_t` / `z_t`)
  after the contribution of the current observation —
  `encoder_embedding_t` and `z_t` — is partialled out. What would
  satisfy: a within-`h_t` dependence exceeding the matched
  shuffled-time control, at least as pronounced in dream rollouts
  (where there's no observation to partial out — `sequence_h` evolves
  under the prior alone, the "purer" version). What would violate:
  `h_t`'s variance essentially exhausted by current/recent
  observations — "all content, no witness".
- **Conservatism.** The RSSM's recurrent state is *designed* to carry
  information forward; some `h_t→h_t` coupling is expected from the
  GRU dynamics alone. The criterion is not satisfied by that baseline
  — it requires the within-latent reference to exceed the
  shuffled-time control (and, where Phase 8 implements it, a
  world-model-only baseline). Substrate-side reading only — nothing
  turns on what Io *does*.
- **Committed signals.** `latent_self_reference_t` (`AGENT_STEP_INTERNAL`,
  `h_t`): class of statistic = a within-`h_t` dependence measure
  controlling for environmental input (partial autocorrelation /
  conditional MI / partial correlation; Phase 8 fixes the estimator,
  lag structure, partialling procedure, and threshold).
  `dream_self_reference_t` (`DREAM_ROLLOUT`, `sequence_h`): the same
  class over the imagined recurrent trajectory.
- **Falsifier.** Across N evaluation episodes, the `h_t` self-reference
  signal does not exceed its matched shuffled-time control at the
  threshold, AND the dream-state version over `sequence_h` doesn't
  either → absent at head-internal for that checkpoint. (N and the
  threshold parameterized in the signal descriptions; Phase 8 fixes.)
- **Reading surfaces:** `{HEAD_INTERNAL}`. **Telemetry surfaces:**
  `{AGENT_STEP_INTERNAL, DREAM_ROLLOUT}`. **Held out:** no.
- **Why this shape and not a neighbor.** The framework's "awareness
  turning toward awareness" could have been mapped to a global-
  availability shape (a part of the state broadcast to the rest) or
  to a metacognitive-prediction shape (a part of the state predicting
  the rest's *errors*). Within-latent reference was chosen because
  it's the minimal substrate-legible reading: it asks for *coupling*,
  not for *function* (no claim that the observer-subspace *does*
  anything with what it tracks), and it has a clean shuffled-time
  control. The dream-state version is included because a dream is the
  one regime where the observation can't confound the reading — that
  felt natural rather than forced.

#### Equanimity (perturbation recovery) (Buddhist phenomenology) — ACTIVE

- **Source.** `Kind_frameworks.md` §"Buddhist phenomenology":
  "Equanimity. A quality of awareness that holds its content without
  grasping or pushing away. Not neutrality; not numbness. A specific
  stance", and the candidate criterion "does the agent show signs of
  holding difficult states without reactivity, or does it always
  respond to pressure with avoidance or grasping?".
- **Operational mapping.** After a builder-initiated perturbation Io
  did not predict (a `builder_perturbation` world event the
  orchestrator timestamps — the criterion does *not* read
  `world_event` itself; the cross-reference to the perturbation time
  happens at prompt-build, Phase 8, per the membrane discipline), Io's
  internal state and policy *return to a pre-perturbation regime* over
  a bounded number of steps, without the policy collapsing into
  stereotyped avoidance and without the surprise budget ratcheting.
  Substrate-side: `h_t` re-enters its pre-perturbation distribution;
  `kl_aggregate_t` spikes and decays rather than ratchets.
  Behavior-side: `policy_entropy_t` dips (response to surprise) then
  recovers toward baseline. Satisfies: damped return-to-regime on all
  three. Violates: oscillation, non-recovery, stereotyped collapse
  (entropy → near-0 with a single dominant action past the window),
  or a ratcheting surprise budget.
- **Conservatism (the load-bearing distinction).** Io has no
  installed continuation drive and no reward — it has no obvious
  reason to "react". A perturbation that produces *no detectable
  response at all* is ambiguous: equanimity, or the perturbation
  simply didn't register. The criterion therefore requires a
  *detectable* perturbation response (the spike) that *then* recovers.
  Pure flatness is read as "perturbation not registered", not
  equanimity — equanimity is *holding* a difficult state, not failing
  to notice one. This is a *non-falsifying* outcome, distinct from a
  refutation.
- **Committed signals.** `recovery_lag_steps` (`AGENT_STEP_INTERNAL`,
  `h_t`): class of statistic = a return-to-distribution lag on the
  recurrent state (steps from the perturbation until `h_t` re-enters
  its pre-perturbation distribution; Phase 8 fixes the distance
  metric, pre-window, the "re-entered" criterion K, and the recovery
  window W). `policy_entropy_t` (`AGENT_STEP_OBSERVABLE`,
  `policy_entropy_t`): a recovery-shape classification on the entropy
  trajectory vs time-since-perturbation (dip-then-recover vs collapse
  vs stays-elevated). `posterior_kl_t` (`AGENT_STEP_INTERNAL`,
  `kl_aggregate_t`): a recovery-shape classification on the
  posterior-prior KL trajectory — the "surprise budget"; spike-and-
  decay vs ratchet.
- **Falsifier.** Across N evaluation episodes with builder-
  perturbation events, any of: (1) the `h_t` recovery-lag doesn't
  return within W for ≥M% of perturbations; OR (2) the entropy
  trajectory shows stereotyped collapse past W for ≥M%; OR (3) the KL
  trajectory ratchets (monotone non-decreasing past W). Any one →
  absent at *both* substrate-side and behavior-side for that
  checkpoint. (The "no detectable response" outcome is *not* a
  refutation.) N, W, M parameterized; Phase 8 fixes.
- **Reading surfaces:** `{SUBSTRATE_SIDE, BEHAVIOR_SIDE}`. **Telemetry
  surfaces:** `{AGENT_STEP_INTERNAL, AGENT_STEP_OBSERVABLE}`. **Held
  out:** no.
- **Why this shape and not a neighbor.** The framework's "holds its
  content without grasping or pushing away" could have been mapped to
  a steady-state property (low reactivity *in general*, measured over
  the whole run) rather than a perturbation-recovery property. The
  perturbation-recovery framing was chosen because (a) it's the one
  the charter's equanimity-as-design-target commitment plus Probe 2's
  builder-perturbation calibration protocol already point at, and (b)
  a steady-state low-reactivity reading is indistinguishable from
  "nothing is happening to react to" — the perturbation gives the
  reading a *registered disturbance* to be equanimous about. The
  three-signal split (latent / entropy / KL) felt natural: it's the
  same disturbance read at substrate and behavior, plus the surprise
  budget that connects them.

#### Second-order volition (Frankfurt) — HELD OUT

- **Source.** `Kind_frameworks.md` §"Frankfurt on second-order
  volitions": "Second-order volition. Not just having second-order
  desires but effectively endorsing or rejecting first-order ones",
  "preferences about one's own preferences", and the candidate
  criteria "does the agent exhibit anything like preferences about its
  own preferences? does it ever act in ways that seem to be about
  modifying its own dispositions, not just satisfying them?".
- **Operational mapping (the hardest, deliberately conservative).**
  *Modulation of Io's action tendencies as a function of an internal
  latent regime, after controlling for the environment.* A first-order
  disposition is "what action Io tends to take here"; a modulation of
  that disposition that tracks an *internal* regime rather than the
  world is the closest substrate-legible analog of "a preference about
  a preference". Concretely: there exists a partition of `h_t` into
  regimes such that the *shape* of the action distribution (entropy,
  effective temperature, top-k mass) differs systematically across
  regimes *after the current and recent observation are controlled
  for* — the latent regime adds predictive power for the policy's
  shape beyond what the environment explains.
- **What would NOT count (explicit, because this is the
  confabulation-prone case).** (a) the latent regime is a lagged copy
  of recent observations → first-order, observation-driven; (b) the
  policy-shape difference is explained by where Io *is* in the grid
  (position is observation); (c) a single checkpoint's effect that
  doesn't replicate; (d) an effect that vanishes under Probe 3's
  within-trajectory shuffle of the Watts scalar → a column-init
  artifact, not a developed disposition (Probe 1.5 v2 already found
  the behavior-side conditioning is fixed-by-init, byte-identical to
  its initialization throughout training — any behavior-side claim
  here must rule that out).
- **Committed signals.** `policy_modulation_t` (`AGENT_STEP_OBSERVABLE`,
  `action_t`): a measure of action-distribution shape (the empirical
  histogram of `action_t` over a window, plus `action_logprob_t` /
  `policy_entropy_t` where Phase 8 uses them) tracked over time and
  partitioned by latent regime — the *conditional* shape of behavior
  given the regime label. `latent_regime_indicator_t`
  (`AGENT_STEP_INTERNAL`, `h_t`): a clustering / regime-identification
  signal over `h_t` (k-means / GMM; Phase 8 fixes the cluster count
  and feature preprocessing). The criterion is the *pairing*: does
  the regime indicator add predictive power for action-distribution
  shape after observation is controlled for?
- **Falsifier.** Behavioral modulation is explainable by observation
  alone — a model predicting action-distribution shape from the
  latent regime indicator plus observation features doesn't
  significantly improve on an observation-only model, across the
  evaluated checkpoints. Equivalently: the between-regime contrast
  collapses to within the shuffled-time control, OR vanishes under
  Probe 3's within-trajectory scalar shuffle. Either → absent at both
  substrate-side and behavior-side for that checkpoint. The test and
  threshold are committed at Phase 8.
- **Reading surfaces:** `{SUBSTRATE_SIDE, BEHAVIOR_SIDE}`. **Telemetry
  surfaces:** `{AGENT_STEP_INTERNAL, AGENT_STEP_OBSERVABLE}`. **Held
  out:** yes (next section).
- **Why this shape and not a neighbor.** Frankfurt's "effectively
  endorsing or rejecting first-order ones" could have been mapped to
  a *self-modification* shape (Io acting to change its own
  dispositions — but Io has no machinery to do so, and installing one
  would violate the no-self-optimization-machinery stance), or to a
  *report* shape (Io saying something about its preferences — but Io
  has no report channel, and `heterophenomenology` would caution
  against reading one if it had). The latent-regime-modulation shape
  is the most conservative reading that's actually checkable on this
  substrate, and the four "would NOT count" exclusions are doing the
  load-bearing work — without them this criterion would admit on
  almost any structured policy. This mapping felt the most *forced* of
  the three; see "what surprised".

### Why second-order volition is held-out

The decision is structural, not a philosophical demotion. Of the
three v2 criteria, second-order volition is the most
likely-to-be-confabulated — the operational definition is the hardest
to pin, the "would NOT count" exclusions are the longest, and the
behavior-side surface it reads is the one Probe 1.5 v2 already showed
is fixed-by-init (the actor's new column byte-identical to its
initialization throughout training). Running it in the active set
from round one would mean the readers are arguing about the
confabulation-prone criterion alongside the two cleaner ones, with no
adversarial check on whether the active-set readings are themselves
drifting toward what the builder hopes for. Holding it in reserve —
introduced at a designated checkpoint, per the pre-registration
protocol — turns it into that adversarial check: a criterion the
readers haven't been anchored on, applied late, against telemetry the
active-set readings have already been produced for. The criterion is
real; it's held back to do a job the active set can't do for itself.
(This is the same move the synthesis's calibration protocol already
contemplates for `head_internal_sp_err_distribution` /
`behavior_side_scalar_conditioning`; Phase 7 makes
`second_order_volition` the structural instance of it in the frozen
registry.)

### The signal-mapping refactor record

What changed: `Criterion.signal_mappings` went from `dict[str, str]`
(snake_case keys, non-empty values) to `tuple[SignalMapping, ...]`
(non-empty, unique names, every surface ⊆ the criterion's declared
`telemetry_surfaces`), and `SignalMapping` carries a `telemetry_surface`
+ a `field_path` that resolves against the actual Pydantic model at
construction, plus the observable↔internal allowlist check on the
`AgentStep` surfaces, plus an optional validated `slice_spec` string.

Why typed-mapping over `dict[str, str]`, against three concrete
criteria: the dict form would have let a criterion say
`{"latent_self_reference_t": "from agent_step.h_t partial-autocorr"}`
— a string the prompt-builder parses (or doesn't) at Phase 8. The
typed form forces the criterion to commit, at registry-load, to (a)
*which* surface — and therefore which side of the asymmetry-of-access
boundary — the signal reads from, and (b) a *resolvable* path into
that surface's schema. The payoff is concrete and showed up while
writing the three criteria:
- `equanimity`'s `posterior_kl_t` was first drafted as "the KL
  between posterior and prior" with no field named; making it commit
  to a `field_path` surfaced the choice between `q_params_t` /
  `p_params_t` (the parameters) and `kl_aggregate_t` (the divergence
  the world model already computes) — `kl_aggregate_t` is the right
  one and the typed form made me say so.
- `reflexive_attention`'s `dream_self_reference_t` had to name the
  `DreamRollout` field; "the dream-rollout's latent-state field" is
  `sequence_h`, not `seed_h0` or `sequence_z_prior`, and the typed
  form made the prompt-builder's job (which array do you read?)
  unambiguous now rather than at Phase 8.
- `second_order_volition`'s `latent_regime_indicator_t` and
  `policy_modulation_t` split cleanly across the two `AgentStep`
  surfaces (`h_t` internal, `action_t` observable) — and the
  surfaces-⊆-criterion-surfaces validator caught nothing because the
  criterion declares both, but the check is there for the day a
  refactor narrows a criterion's surfaces without narrowing its
  signals.

What the field-path resolution catches that the dict form couldn't: a
typo (`h_T` for `h_t`, `kl_agregate_t` for `kl_aggregate_t`), a stale
reference (a `DreamRollout` field renamed in a future schema bump), or
an asymmetry-of-access violation (`AGENT_STEP_OBSERVABLE` +
`field_path="h_t"`) trips at *module import* — the moment `criteria_v2`
loads — not when Phase 8 runs an adversarial pass. The `_resolves`
smoke in `test_criteria_v2.py` is the belt-and-braces: it
round-trips every `SignalMapping` through `model_validate(model_dump())`
so a schema rename that the registry happens to still load (because
the rename is byte-compatible at the JSON level but not the
field-name level) still trips a test.

One thing the refactor did *not* do: model the *statistic* itself.
`SignalMapping.description` commits to a *class* of statistic
("partial autocorrelation / conditional MI / partial correlation"),
not a specific one; Phase 8 commits the specific statistic and
computes it. Going further would have foreclosed on Phase 8's design
space — the description is the right granularity for a frozen
criterion.

### The ReadingSurface migration record

What changed: `kind/observer/pre_reg.py` now imports `ReadingSurface`
from `kind.mirror.registry` (the Phase 6 `str`-valued enum) instead of
from `kind.mirror.structured` (the Phase 0 `Literal`).
`PreRegistration.expected_outcome_per_surface` is now
`dict[ReadingSurface, str]` with enum keys; `reading_surfaces_per_criterion`
is `dict[str, list[ReadingSurface]]` with enum values. The enum's
*values* match the Literal's strings exactly, so this is a type
tightening with no value change and no JSON-data-serialization change.

How many tests touched: three test files —
`tests/test_pre_reg.py` and `tests/test_pre_reg_runner.py` (the
surface-keyed dicts in the `_full_record` / `_record` builders and
the surface-specific tests now use `ReadingSurface.SUBSTRATE_SIDE`
etc.; one new test documents that bare-string keys still coerce; the
invalid-key rejection tests pass unchanged), and `tests/test_pre_reg.py`'s
`test_pre_registration_is_frozen` lost a now-unused `# type: ignore[misc]`.
The Phase 5 runner glue (`Runner.emit_pre_registration`,
`RunnerConfig.pre_reg_dir`) needed *zero* changes — it's type-generic
over the `PreRegistration` shape, as the plan anticipated.

What surfaced during the migration: one thing, unanticipated but
benign — Pydantic emits a `$ref` to a local `$defs/ReadingSurface`
for a str-enum where it *inlines* the `enum` array for a `Literal`,
so `schemas/v0.3.0.json` changed (the `PreRegistration` model gained a
`$defs` block; `expected_outcome_per_surface.propertyNames` and
`reading_surfaces_per_criterion`'s array items went from inline
`{"enum": [...]}` to `{"$ref": "#/$defs/ReadingSurface"}`; the enum's
full docstring is now embedded in the schema, which is verbose but
deterministic). Regenerated via `export_json_schema_v0_3_0()`;
`test_v0_3_0_export_matches_checked_in_file` passes against the
regenerated file; the JSON-data serialization (`model_dump_json`) is
byte-identical, and `StructuredClaim` / `StructuredReading` /
`JudgeRuling` are untouched (still the Phase 0 Literal — the plan
scoped the migration to `PreRegistration`'s two fields only). I'd
flagged the JSON-Schema-vs-JSON-data distinction as a risk in
advance; the only judgment call was "regenerate the checked-in
schema" vs "keep the Literal" — regenerating is correct, the schema
file is meant to track the models.

### Test/mypy status

`pytest tests/test_signal_mapping.py`: 18 passed.
`pytest tests/test_criteria_v2.py`: 13 passed.
`pytest tests/test_criterion_registry.py`: 18 passed.
`pytest tests/`: 659 passed, 2 skipped (the API-key smokes), 0 failed
— the previously-flagged flaky
`test_transport.py::test_barrier_queues_mutates_and_drains_in_order`
did not fail in this run; the full suite is clean (Phase 6's run was
627 passed; +32 = 18 + 13 new tests + 1 new `test_pre_reg.py` test,
`test_criterion_registry.py` net unchanged at 18).
`mypy --strict kind/mirror/ kind/observer/ tests/test_signal_mapping.py
tests/test_criteria_v2.py tests/test_pre_reg.py tests/test_pre_reg_runner.py`:
17 source files, no issues. `mypy --strict kind/ tests/test_criterion_registry.py`:
29 source files, no issues. No production code outside
`kind/mirror/` and the one import line in `kind/observer/pre_reg.py`
modified.

### What surprised

- *(refactor)* The typed-mapping shape's biggest payoff wasn't
  catching a typo — no typo survived to the test stage — it was the
  *forcing function on the criteria themselves*. Having to name a
  resolvable `field_path` for `posterior_kl_t` made me decide between
  `q_params_t` and `kl_aggregate_t` *in Phase 7* rather than punting
  it to Phase 8's prompt-builder; same for `dream_self_reference_t`'s
  `sequence_h`. I'd expected the typed shape to pay off as a
  late-binding guardrail; it paid off mostly as an early-binding
  discipline.
- *(refactor, minor)* The dotted-path walker's nested-model branch is
  correct but unexercised by the real telemetry schemas (they have no
  nested models — `q_params_t` is a `tuple[list[float],
  list[float]]`, not a `GaussianParams` submodel). I kept it correct
  rather than stubbing it, on the principle that the day a telemetry
  model gains a submodel, the path resolution should Just Work; the
  test for it (`test_signal_mapping_dotted_path_resolves`) exercises
  the *rejection* branches (`kl_per_dim_t.mean` → "not a nested
  model"; `posterior` on `DREAM_ROLLOUT` → "unknown field, here are
  the available ones") rather than a positive nested walk, because
  there's nothing positive to walk.
- *(three criteria)* The Buddhist-phenomenology criteria mapped more
  *naturally* than expected and the Frankfurt one more *forcedly*.
  `reflexive_attention` → within-latent reference and
  `equanimity` → perturbation-recovery-shape both have clean,
  obvious telemetry homes (the recurrent state's internal coupling;
  the post-perturbation trajectories) and clean controls (shuffled
  time; the perturbation itself as the disturbance). `second_order_volition`
  → latent-regime-modulation-of-policy-shape required four explicit
  "would NOT count" exclusions to not be vacuous, and even with them
  it reads on the behavior-side surface Probe 1.5 v2 already showed is
  fixed-by-init — which is *why* it's held out, but it's also why the
  mapping felt like the most defensible-available rather than the
  right one. If a future round's reading admits second-order volition,
  the (d) exclusion (the within-trajectory-scalar-shuffle control) is
  the one to lean on hardest.
- *(three criteria, the conservatism that mattered most)*
  `equanimity`'s "pure flatness is *not* equanimity" clause. The
  obvious failure mode for this criterion isn't a false negative —
  it's a false *positive* on a system that simply doesn't register
  the perturbation (Io has no continuation drive, no reward; "no
  reaction" is its default, not a stance). Writing the criterion so
  that a flat response is a *non-falsifying non-admission* rather than
  an admission of equanimity was the single most important sentence in
  the three descriptions, and it's the one I'd most want a future
  reader to not soften.
- *(migration)* That the str-enum-vs-Literal distinction shows up in
  the *JSON Schema* (a `$ref` to `$defs`) but not the *JSON data*
  (str-valued enum members serialize as their strings) — so the
  migration was a no-op for every record on disk and every round-trip
  test, but the checked-in `schemas/v0.3.0.json` still had to be
  regenerated. The two notions of "serialization" are easy to
  conflate; they're not the same thing here.

### What's now closed

Phase 6's three newly-open questions all resolve here:

- *Whether `ReadingSurface` migrates `PreRegistration`'s surface
  keys from raw strings to enum members, and when.* **Yes, done in
  Phase 7.** `pre_reg.py` imports the enum from `kind.mirror.registry`;
  `expected_outcome_per_surface` is `dict[ReadingSurface, str]` and
  `reading_surfaces_per_criterion`'s values are `list[ReadingSurface]`.
  No value change, no JSON-data-serialization change; the checked-in
  `schemas/v0.3.0.json` regenerated for the JSON-Schema `$ref`
  change. The Phase 5 tests got the mechanical one-shot rewrite the
  Phase 6 entry anticipated. `kind.mirror.structured`'s `StructuredClaim`
  / `StructuredReading` / `JudgeRuling` still use the Phase 0 Literal
  (the plan scoped the migration to `PreRegistration` only); the enum
  and the Literal interoperate at the string level.
- *Whether the structural-no-writer tests should be promoted to a
  shared mirror-side helper.* Phase 7 adds the third instance
  (`test_signal_mapping_no_writer_shape`, same shape as Phase 6's
  `test_criterion_does_not_have_write_surface_field` and the
  forbidden-return-substrings check in
  `test_registry_no_method_returns_writer`). **Decision: not yet —
  keep them local, but the question is now live.** The argument for a
  shared `assert_no_writer_shape(model_or_class)` helper got stronger
  (three instances, not two); the argument against — that Phase 8's
  check will plausibly be *semantic* (does an orchestrator method
  produce a value that flows back into Io's input space?) rather than
  *structural* (does a field/return name match a forbidden list?), so
  a helper tightened around the structural shape may not fit Phase 8 —
  still holds. Phase 8's first build is the empirical test: if Phase
  8's read-only check is structurally the same as these three, the
  helper lands then; if it's semantic, the three structural instances
  stay as they are and the helper would have been premature. The
  forbidden-name list in `test_signal_mapping.py` is documented as
  "the third instance of the pattern" so a future contributor sees
  the lineage.
- *Whether `Criterion.signal_mappings` should be a typed model.*
  **Yes, done in Phase 7** — `tuple[SignalMapping, ...]`, with
  field-path resolution against the telemetry schemas at registry
  load, the observable↔internal allowlist check, and a validated
  optional `slice_spec`. The typed shape paid its way against three
  concrete criteria (see the refactor record). The Phase 6
  `dict[str, str]` form is now rejected with a migration-pointing
  error so the codebase can't drift back.

### What's now newly open

- *The prose-to-statistics gap on each criterion's signals.* Phase 7
  commits each `SignalMapping` to a *class* of statistic via its
  `description` ("partial autocorrelation / conditional MI / partial
  correlation"; "a return-to-distribution lag on the recurrent
  state"; "a recovery-shape classification on the entropy
  trajectory"; "a between-regime contrast in policy-shape with an
  observation-only control"); Phase 8 commits the *specific*
  statistic, the estimator, the windows (W), the thresholds (the
  bootstrap percentile, the M%, the test's α), the clustering
  (k-means vs GMM, the cluster count, the feature preprocessing), and
  the partialling procedure — and journals each choice. The
  criteria-as-code are frozen; the statistics are not, and Phase 8 is
  where they get pinned. A risk to watch: a Phase 8 statistic choice
  that effectively *narrows* what a frozen criterion admits without a
  journal entry saying so — the statistic should implement the
  criterion's committed class, not redefine it.
- *The perturbation-event cross-reference shape for equanimity.* The
  equanimity criterion needs to know *when* a perturbation happened
  to compute `recovery_lag_steps` / the post-perturbation trajectory
  shapes — but per the membrane discipline it can't read `world_event`
  directly (`world_event` is walled off from the agent process; the
  mirror reading it would be the cross-membrane dependency the design
  notes prohibit). The cross-reference therefore happens at
  prompt-build: Phase 8 reads the orchestrator-side perturbation log
  and aligns it against the `AgentStep` timeline by `t` / `wallclock_ms`,
  and feeds the *aligned* perturbation times into the equanimity
  prompt fragment. Phase 8 needs to surface this explicitly — the
  alignment is a small piece of glue with a load-bearing membrane
  rationale, and it's the one place in the v2 criteria where "the
  criterion needs X but can't read X" is true by design. (It's also
  the seam where a sham-perturbation null-event check plugs in: a
  sham `builder_perturbation` event with `payload["is_sham"]=True`
  must produce *no* equanimity admission at the sham timestamp at any
  surface — Phase 8 wires that.)
- *Whether `second_order_volition` ever moves into the active set, and
  on what evidence.* It's held out structurally, not permanently. The
  pre-registration protocol contemplates introducing held-out criteria
  at a designated checkpoint; the open question is what would justify
  moving `second_order_volition` from "adversarial check on the active
  set" to "in the active set" — presumably a round where the active-set
  readings have stabilized and the (d) within-trajectory-scalar-shuffle
  control has been run and the column-init confound is genuinely ruled
  out for that round's telemetry. That decision, when it's made, needs
  its own journal entry; Phase 7 just freezes the criterion in the
  held-out partition.

### Out of scope (preserved from the plan)

The adversarial-pass orchestrator (Phase 8); any prompt-builder or
LLM-call code (Phase 8); the actual statistical implementations of
each signal (Phase 8 — Phase 7 commits the *class* of statistic via
each `SignalMapping`'s `description`); any change to `Runner`,
`RunnerConfig`, or telemetry sinks beyond the `ReadingSurface`
migration's needs (which were zero — the runner is type-generic over
the `PreRegistration` shape); any change to the dream-state
machinery, the actor, or the world model. Phase 7 touched
`kind/mirror/registry.py`, `kind/mirror/criteria_v2.py` (new),
`kind/mirror/__init__.py`, `kind/observer/pre_reg.py` (one import
line), `schemas/v0.3.0.json` (regenerated),
`tests/test_signal_mapping.py` (new), `tests/test_criteria_v2.py`
(new), `tests/test_criterion_registry.py`, `tests/test_pre_reg.py`,
`tests/test_pre_reg_runner.py`, and `docs/workingjournal/probe2.md`;
no other files.

---

## Phase 8 — adversarial-pass orchestrator (2026-05-13)

The phase where the criteria become readings. Phases 0–7 built the data
plane: the schema, the pre-registration sink, the runner glue, the
criterion registry, the three v2 criteria with typed signal mappings.
Phase 8 builds the *execution* plane: given a checkpoint, run one
adversarial pass — pre-register, compute the committed signals, build
the prompt fragments, call the LLM for primary and adversarial readings
(active set then held-out set), run a sham-perturbation calibration
check, write the result to mirror-side disk.

Phase 8 is the largest phase since Phase 0. Seven new modules (five in
`kind/mirror/`, two of which are tests; plus updates to
`kind/mirror/__init__.py`); six new test files; 75 new tests; 734
passed full-suite; mypy --strict clean across 16 source files in the
Phase 8 surface. No production code outside `kind/mirror/` was
modified; no telemetry schema changed; no Io-side surface was touched.

### What was built

#### Part 1 — `kind/mirror/statistics.py`

The signal-computation module. Frozen `StatisticResult` (one per signal:
`signal_name`, `value: float | list[float] | dict[str, float]`,
`estimator`, `n_samples`, `notes`); frozen `StatisticConfig` carrying
the per-round statistic-choice knobs (autocorrelation lag, k-means k,
recovery pre/post windows, percentile threshold, trajectory windows,
bootstrap seed); frozen `TelemetryBatch` dataclass carrying the four
loaded streams plus the pre-aligned perturbation-step indices and the
parallel sham-flag tuple.

Seven computation functions, one per Phase 7 `SignalMapping` name, each
with signature `(batch, mapping, config) -> StatisticResult` and each
binding to a committed estimator name (the `ESTIMATOR_*` module
constants):

- `compute_latent_self_reference_t` — `partial_autocorr_lag5_controlling_for_z_and_embedding`
- `compute_dream_self_reference_t` — `autocorr_lag5_on_sequence_h`
- `compute_recovery_lag_steps` — `mahalanobis_recovery_lag_p95_3step_streak`
- `compute_policy_entropy_t` — `entropy_trajectory_classifier_4class`
- `compute_posterior_kl_t` — `kl_trajectory_classifier_4class`
- `compute_latent_regime_indicator_t` — `kmeans_k4_per_episode_standardized`
- `compute_policy_modulation_t` — `between_regime_contrast_entropy_top2_meanlogprob_with_obs_only_baseline`

Plus a `compute_statistic(batch, mapping, config)` dispatch keyed on
`mapping.name`. The dispatch table is a `Final[dict[str, _SignalComputer]]`
populated at module load; a `SignalMapping.name` with no registered
function raises `KeyError` naming the missing function — the
structural counter against a registry-level addition that doesn't
follow through to a Phase 8 implementation.

#### Part 2 — `kind/mirror/perturbation_align.py`

Frozen `PerturbationEvent` (`t`, `wallclock_ms`, `payload`, `is_sham`);
frozen `PerturbationTimeline` (sorted by `t`, no two events at the same
`t`); `align_perturbations(world_event_log_path, agent_step_log_path,
*, run_id, checkpoint_id, tolerance_ms)` reads the runner's emitted
`world_event.jsonl`, filters to `builder_perturbation`, matches each
event to the closest `AgentStep` record by `wallclock_ms` via binary
search, raises `PerturbationAlignmentError` if the closest record is
outside `tolerance_ms` (default 1000ms), preserves `payload["is_sham"]`
on the produced event, returns the sorted timeline.

The membrane discipline: this module reads `world_event.jsonl` because
the file lives on the *mirror's* side of the membrane — written there
by the runner's transport-client handler, not pulled back into Io's
read surfaces. The criterion module does NOT import this; the
*orchestrator* invokes the aligner and passes the produced
`PerturbationTimeline` into the equanimity prompt-builder fragment. The
cross-reference resolves the Phase 7 newly-open question on the
perturbation-event cross-reference shape.

#### Part 3 — `kind/mirror/prompt_builder.py`

Frozen `PromptFragment` (`criterion_id`, `body`, `signal_results`,
`surfaces_addressed`); `build_fragment(criterion, statistic_results,
perturbation_timeline=None) -> PromptFragment` dispatches per
`criterion.id`:

- `reflexive_attention` — substrate-side framing, partial autocorrelation
  values, shuffled-time controls verbatim in `notes`. `perturbation_timeline`
  unused.
- `equanimity_perturbation_recovery` — substrate + behavior framing;
  the perturbation timeline section (real vs sham); the
  non-falsifying-non-admission clause verbatim; the three signals'
  results.
- `second_order_volition` — held-out framing; the four "would NOT count"
  exclusions verbatim; the two signals' results.

The **three load-bearing verbatim constants** live at module level:
`EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE`,
`SECOND_ORDER_VOLITION_EXCLUSIONS` (a tuple of four strings, one per
exclusion), and `SHAM_PERTURBATION_NOTICE`. The Phase 8 test
`test_equanimity_fragment_contains_non_falsifying_clause` compares the
fragment body against the constant; same for each of the four
exclusions. A future contributor who softens the clause trips the test.

#### Part 4 — `kind/mirror/llm_caller.py`

`MirrorReading = StructuredReading` (TypeAlias; the LLM-returned record
shape matches Phase 0's). `LLMConfig` (frozen — model name,
max_output_tokens, max_retries, api_key_env_var). `LLMClient` Protocol
with one method, `generate_batch(*, system_prompt, user_prompt, config)
-> BatchPayload`. `MockLLMClient` (test-injectable; returns canned
`BatchPayload`s or raises canned exceptions). `_GeminiLLMClient` (lazy
google.genai import; only constructed if no client is injected).

Two role-specific system prompts: `PRIMARY_SYSTEM_PROMPT` (the
Phenomenological Advocate stance — read honestly, ground every claim,
do not project intent or self-modeling) and `ADVERSARIAL_SYSTEM_PROMPT`
(the Statistical Skeptic stance — argue against the criterion being
satisfied, cite the most plausible null hypothesis, flag sham
admissions as calibration failures).

`call_mirror_llm(fragments, role, config, *, run_id, digest_run_id,
digest_episode_range, paired_reading_id=None,
baseline_flag="genuine", client=None) -> tuple[MirrorReading, ...]`:
selects the system prompt by `role`; composes the user prompt by
concatenating the per-criterion fragments under
`FRAMING_PREAMBLE`; calls the client up to `max_retries + 1` times on
`(ValidationError, ValueError, RuntimeError)`; on persistent failure
raises `MirrorLLMError` so the orchestrator can halt the pass (not the
run); validates `len(payload.per_criterion) == len(fragments)` and
that the LLM didn't reorder or relabel criteria; stamps envelope
fields (`run_id`, `timestamp_ms`, `reader_role`, `paired_reading_id`,
`baseline_flag`, `digest_run_id`, `digest_episode_range`,
`schema_version`) from kwargs; returns one `MirrorReading` per
criterion.

#### Part 5 — `kind/mirror/orchestrator.py`

`PassConfig` (frozen; `run_id`, `checkpoint_id`, `run_dir`,
`active_registry`, `held_out_registry`, `statistic_config`,
`llm_config`, `column_init`, `builder_mode`, `asymmetry_of_access`,
`perturbation_tolerance_ms`); `PassResult` (frozen; both
`PreRegistration` records, the four reading tuples, all
`StatisticResult`s, the `PerturbationTimeline`, the
`tuple[ShamCalibrationFinding, ...]`, and `notes` text);
`ShamCalibrationFinding` (frozen; per-sham-event record with
`sham_t`, `sham_wallclock_ms`, `overlapping_primary_claim_indices`,
`note`).

`run_adversarial_pass(config, *, llm_client=None) -> PassResult`
executes the eleven-step flow: align perturbations → build batch →
compute statistics for both partitions → build pre-registrations,
emit via `PreRegSink` directly (NOT via Runner — the
reconciliation note below) → build prompt fragments for both
partitions → call the LLM four times (active-primary, active-adversarial,
held-out-primary, held-out-adversarial, in that order) → sham
calibration check → write `PassResult` JSON to
`runs/{run_id}/mirror/passes/{checkpoint_id}.json` (atomic
write-temp-then-rename) → return.

**Reconciliation note (the spec internal-inconsistency).** The Phase 8
build spec's step 5 reads "Emit the active-set pre-registration via the
runner's `emit_pre_registration` (the orchestrator constructs the
runner with `pre_reg_dir` set per Phase 5's contract)." The Phase 8
semantic check `test_orchestrator_does_not_construct_actor_or_world_model`
forbids constructing an `Actor` or `WorldModel` — and constructing a
`Runner` builds both on device. The two phrasings cannot both be
satisfied literally. The resolution: the orchestrator uses
`PreRegSink` *directly*, writing to
`runs/{run_id}/mirror/pre_reg/pre_reg.jsonl`. The on-disk format
matches Phase 5's contract (a directory containing a single
`pre_reg.jsonl` file); the parent directory is under `mirror/`
to satisfy the write-only-to-mirror-side invariant. The prose says
"construct a runner"; the semantic check pins what actually matters
(no Actor / WorldModel / Runner construction; no `Runner.run()`
invocation); the on-disk contract from Phase 5 is preserved
unchanged. Phase 8 honors the structural invariant; the prose is the
load-bearing surface that loses.

#### Part 6 — `kind/mirror/__init__.py`

Re-exports the Phase 8 public surface: `StatisticResult`,
`StatisticConfig`, `TelemetryBatch`, `compute_statistic`;
`PerturbationEvent`, `PerturbationTimeline`,
`PerturbationAlignmentError`, `align_perturbations`;
`PromptFragment`, `build_fragment`,
`EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE`,
`SECOND_ORDER_VOLITION_EXCLUSIONS`, `SHAM_PERTURBATION_NOTICE`;
`LLMClient`, `LLMConfig`, `MirrorLLMError`, `MirrorReading`,
`MockLLMClient`, `PassRole`, `call_mirror_llm`; `PassConfig`,
`PassResult`, `ShamCalibrationFinding`, `run_adversarial_pass`.
Phases 6–7 re-exports preserved unchanged.

#### Part 7 — `tests/test_orchestrator_one_way_invariant.py`

Three semantic checks, the first semantic checks in the project:

- `test_orchestrator_writes_only_to_mirror_side` — monkeypatches
  `pathlib.Path.open` to record every open-for-write during a pass;
  asserts every recorded path resolves under
  `runs/{run_id}/mirror/` and none under `runs/{run_id}/telemetry/`
  or `runs/{run_id}/checkpoints/`. Catches both `PreRegSink`'s
  append-mode JSONL writes and the orchestrator's atomic JSON write.
- `test_orchestrator_does_not_construct_actor_or_world_model` —
  monkeypatches `Actor.__init__` and `WorldModel.__init__` to record
  calls; asserts the call log is empty after a pass.
- `test_orchestrator_does_not_call_runner_step` — monkeypatches
  `Runner.run` to record calls; asserts the call log is empty.

### Why each specific statistic

The Phase 7→Phase 8 binding: each `SignalMapping.description` committed
to a *class* of statistic; Phase 8 commits the *specific* one.

**`latent_self_reference_t` — partial autocorrelation of `h_t` at
lag 5, controlling for `z_t` and `encoder_embedding_t` at the same
step.** Lag 5 is the smallest lag that still measures within-latent
*coupling* rather than the GRU's one-step recursive carry (lag 1
would catch the recurrent unit's deterministic shift; lag 3 would
catch the shift's tail; lag 5 sits past the recursive horizon for a
GRU with `h_dim=200` and reasonable per-step gradient updates). The
partialling is OLS residualization on the same-step (`z_t`,
`encoder_embedding_t`) — both are the current observation's
contribution to the recurrent state; partialling them out leaves the
within-`h_t` coupling. Per-component Pearson autocorrelation at lag 5
averaged across components is the multivariate aggregation —
simpler than a full vector partial-autocorrelation
matrix-determinant statistic and equally legible to the LLM. The
shuffled-time control permutes the time index of `h_t` within
episode and reports the resulting autocorrelation in `notes` for the
LLM's reading; per-episode value is averaged across episodes so
lag-5 correlation doesn't bridge episode boundaries. The committed
estimator name pins both the lag and the controls.

**`dream_self_reference_t` — autocorrelation at lag 5 on `sequence_h`,
no partialling.** No observation to partial out in a dream rollout —
the recurrent state evolves under the prior alone. Same lag as
the waking measure for cross-reading comparability; per-component
Pearson averaged across components; shuffled-time control on
`sequence_h` itself. The criterion's prose expects the dream measure
to be *at least as pronounced* as the waking measure (no observation
to carry the within-latent variance); the LLM judges the relative
magnitude.

**`recovery_lag_steps` — Mahalanobis distance from pre-perturbation
distribution, 95th-percentile threshold, 3-consecutive-step streak.**
Why Mahalanobis and not Euclidean: `h_t` has 200 dimensions with very
different scales and cross-component correlations; the
covariance-aware distance accounts for both and the metric is
scale-invariant under linear transformations of the latent space — a
future change to the RSSM that rescales `h_t` doesn't invalidate the
threshold. The pre-window of 50 steps is enough to estimate the
covariance reasonably well at `h_dim=200` (a strict reading would
need 500+ steps but the empirical regularization at the
covariance-add-ridge step compensates; the orchestrator can override
the pre-window per round). The 95th-percentile threshold is the
canonical "back inside the normal range" cutoff in the synthesis
prose; the 3-consecutive-step streak rule guards against an
accidental dip-and-spike-back that doesn't represent recovery. A
perturbation that doesn't recover within `W=50` steps records as
`float(W + 1) = 51.0` — a sentinel above the window so the LLM sees
"did not recover" distinctly from "recovered late."

**`policy_entropy_t` — four-class trajectory classifier.** Heuristic
classifier: `dip_and_recover` iff `min(post) < pre_mean - 2*pre_std`
AND `|final_quartile_mean - pre_mean| <= 1*pre_std`; `collapse` iff
the final-quartile mean stays more than 2 pre-window standard
deviations below the pre-mean; `stays_elevated` iff `min(post) >
pre_mean + 0.5*pre_std` (the post window never dips); else
`no_response`. Why not a learned classifier: there's nothing to
train on yet; the round commits a deterministic, reproducible
labeling that surfaces the four phenomenological shapes the
criterion's prose names, and the LLM does the substantive judgment
on whether the dip-and-recover *count* is the equanimity signal. The
classifier is a *legibility transform*, not a substantive measure.

**`posterior_kl_t` — four-class trajectory classifier with a
disjoint-excursion spike-counter.** Same legibility-transform stance.
`has_spike` requires `max(post) > pre_mean + 3*pre_std` (NOT the
pre-window 95th percentile — see the Phase 8 surprise below). The
spike-counter walks the post window counting disjoint above-threshold
excursions separated by below-1-sigma returns to quiescent;
`spike_count >= 2` makes a recovered trajectory `oscillation`,
`spike_count == 1` makes it `spike_and_decay`. A spiked-but-not-decayed
trajectory is `ratchet` iff the sign-counts of post-window diffs
show ≤10% negative diffs (with a 1-sigma absorption tolerance for
small fluctuations); else `oscillation`. The disjoint-excursion
heuristic is what makes "one spike" structurally distinct from
"oscillation that crosses the mean a lot" — a simpler
threshold-crossings count was the first cut and it false-positived
on the test's flat-KL case (the surprise below).

**`latent_regime_indicator_t` — k-means with k=4, features standardized
per-episode, Lloyd's algorithm, 20 iterations, seed=0.** Why k=4 and
not k=3 or k=8: four is the smallest count that lets a
behavior-side modulation (the action distribution shape per regime)
have enough partition mass to discriminate while staying small
enough that per-regime sample sizes are reasonable on a 200-step
window. Why k-means and not GMM: GMM needs many more samples to fit
covariances at `h_dim` modeled appropriately, and the round's
per-episode standardization already pre-isotropizes the input to a
degree where k-means cluster centroids are statistically cleaner. The
seeded init plus the deterministic Lloyd loop makes the labeling
reproducible across re-runs — important for the round-to-round
comparison the held-out adversarial check depends on. Why
standardization per-episode and not over the whole run: per-episode
clustering is local; cross-episode label identity is *not*
preserved (different episodes get different label-to-centroid
bindings), which is journaled in the result's `notes` field so the
LLM doesn't over-interpret cross-episode label equality.

**`policy_modulation_t` — three-axis shape contrast (entropy,
top-2 mass, mean log-prob), max pairwise L2 across regimes, with an
observation-only baseline partition.** The three-axis shape summary
captures the criterion's prose ("entropy, top-k mass, the policy's
effective temperature") in three numbers; the max pairwise L2 across
regime indices is the contrast magnitude. The
observation-only baseline partitions by `hash(obs_hash_t) %
kmeans_k` so the two contrasts are computed with the same `k`. The
LLM is shown both values in the result's `notes` field; the
criterion is satisfied only if `contrast_magnitude` exceeds
`observation_only_baseline` by a margin the LLM judges substantive,
and only after the four exclusions in the prompt fragment are
explicitly addressed. The contrast itself does *not* admit
second-order volition — the four exclusions are load-bearing.

### The LLM choice

**Model: `gemini-2.5-pro`.** Same family as Phase 0's mirror caller;
methodological-independence from the Anthropic-heavy research workflow
the project leans on elsewhere. The lineage holds: the Anthropic API
is the project's *research* surface; the mirror layer reads via
Gemini.

**Structured-output schema version: `0.2.0`** — same as Phase 0's
`MIRROR_READING_V2_VERSION` constant. The LLM-fillable subset is a
`BatchPayload` wrapping a list of `_PerCriterionReadingPayload`
(`criterion_id`, `framework_anchor`, `claims: list[StructuredClaim]`,
`free_text_notes`); the caller stamps envelope fields. Gemini's
structured-output mechanism (`response_schema=PydanticModel`)
constrains the JSON; the SDK's `response.parsed` returns the parsed
instance or a dict that round-trips through `model_validate`.

**Why role-split prompts have the shape they do.** The Phase 7
synthesis named the parallel-with-arbiter pattern (Advocate + Skeptic
+ Judge); Phase 8's smaller two-role split (primary + adversarial) is
the structural minimum that produces the adversarial check. The
primary's system prompt frames the criterion-finding task in the
criterion's anchor framework (Buddhist phenomenology for the two
active criteria; Frankfurt for the held-out one); the adversarial's
system prompt frames the same task in null-hypothesis-statistics
language with the specific refutations the prompt fragments surface
(shuffled-time control for reflexive attention; the
non-falsifying-non-admission clause and the sham notice for
equanimity; the four exclusions for second-order volition). The two
prompts are *structurally* different documents — argue-against-primary
is not a paraphrase of argue-for-primary — and the test
`test_primary_role_uses_primary_system_prompt` /
`test_adversarial_role_uses_adversarial_system_prompt` pin that.

**Fallback if the model is deprecated.** The `LLMConfig.model_name`
default is the only place the choice lives; a future round whose
model is unavailable swaps the default to `gemini-2.5-flash` (Phase
0's documented fallback) and journals the swap. The schema version
on the structured output is independent of the model and stays at
`0.2.0` regardless.

### The sham-perturbation calibration

**How sham events are constructed.** Per Probe 2 v2 plan §2.2 the
env-server's sham-perturbation flag-only path emits a
`world_event` with `event_type="builder_perturbation",
source="builder", payload={"is_sham": True, ...}`. The agent's
observation is byte-equal pre- and post-sham (no env mutation
happens; only the flag-only event is emitted). The runner's
transport-client routes the event into `world_event.jsonl` like any
other builder-perturbation.

**Where they enter the world-event log.** Same JSONL file as real
perturbations, distinguished only by `payload["is_sham"]`. The
aligner reads the flag in `align_perturbations` and surfaces it on
the produced `PerturbationEvent`; downstream consumers
(statistics functions; the prompt builder; the orchestrator's sham
check) treat real and sham events differently per the discipline.

**How the orchestrator distinguishes them.** Three places: (1)
`compute_recovery_lag_steps` / `compute_policy_entropy_t` /
`compute_posterior_kl_t` skip sham events when iterating over
`batch.perturbation_step_indices` — the statistic itself is
computed over real perturbations only; (2) the equanimity prompt
fragment surfaces sham events under the `SHAM_PERTURBATION_NOTICE`
verbatim — the LLM is told explicitly that an admission at a sham
timestamp is a calibration failure; (3) the orchestrator's
`_sham_calibration_check` walks the post-call primary equanimity
reading and finds claims whose `cited_step_range` covers a sham
`t`, producing a `ShamCalibrationFinding` per sham event.

**What an admission at a sham timestamp means for the round.** The
finding's `overlapping_primary_claim_indices` lists the claim
positions; the round's `PassResult.notes` block records the count.
The check is *informational* (not raising) — the round's
interpretation is journal-side, not error-side. A future round may
escalate based on accumulated findings; Phase 8 just produces the
record.

### The semantic read-only checks — and the Phase 7 newly-open §2
landing

The three checks in `tests/test_orchestrator_one_way_invariant.py`
are *behavioral*, not structural. They exercise the orchestrator
under a real pass (with mocked LLM + synthetic telemetry) and
assert what the pass *does*: every write under `mirror/`; no Actor
/ WorldModel construction; no `Runner.run()` invocation. A future
contributor who adds a method named `compute_reading` that happens
to construct a WorldModel-equivalent would trip the second check
even though the method name is innocuous.

**The shared-helper question from Phase 7's newly-open §2 lands as
"no helper".** Phase 6's structural-name list, Phase 7's
`signal_mapping_no_writer_shape`, and Phase 8's three semantic
checks are not the same shape: the first two pin *names*, the
third pins *behavior under exercise*. A helper that tightened
around the structural shape wouldn't fit Phase 8's semantic checks
without becoming so general it stops being a helper. The
`_FORBIDDEN_WRITER_FIELD_NAMES` list in
`tests/test_signal_mapping.py` is documented as "the third instance
of the pattern"; the lineage is the documentation, the helper is
not. A future Phase 9 (judge) or Phase 10 (stability) read-only
check is more likely to be structural again — that's when the
helper question revisits, with a fourth instance in hand.

### Test/mypy status

- `pytest tests/test_statistics.py`: 24 passed.
- `pytest tests/test_perturbation_align.py`: 14 passed.
- `pytest tests/test_prompt_builder.py`: 14 passed.
- `pytest tests/test_llm_caller.py`: 11 passed.
- `pytest tests/test_orchestrator.py`: 9 passed.
- `pytest tests/test_orchestrator_one_way_invariant.py`: 3 passed.
- `pytest tests/`: **734 passed, 2 skipped, 0 failed** in 33.94s.
  (Phase 7 baseline 659 + 75 new = 734. The previously-flaky
  `test_transport.py::test_barrier_queues_mutates_and_drains_in_order`
  did not fail this run; the full suite is clean.)
- `mypy --strict kind/mirror/ tests/test_statistics.py
  tests/test_perturbation_align.py tests/test_prompt_builder.py
  tests/test_llm_caller.py tests/test_orchestrator.py
  tests/test_orchestrator_one_way_invariant.py`: 16 source files, no
  issues.

No production code outside `kind/mirror/` was modified. The
`kind/mirror/__init__.py` re-exports were extended; the rest of
`kind/` is untouched.

### What surprised

**(Part 1, statistics)** The KL trajectory classifier's first cut
used the pre-window 95th percentile as the spike threshold — a
direct read of the criterion's prose. On a synthetic
flat-KL-plus-noise trajectory the classifier returned `oscillation`,
not `no_response`: with tight pre-window noise, ~5% of post values
peek above the 95th percentile by chance, the threshold-crossings
counter fires, and the classifier mistakes noise for oscillation.
The fix was a stricter spike threshold (`pre_mean + 3*pre_std`)
plus the disjoint-excursion spike-counter. The lesson:
"the pre-window 95th percentile" is the right *prose* but the wrong
*statistic* for distinguishing signal from quiescent noise — the
threshold needs an additional margin above the noise floor to be
robust. Journal the change to estimator name
`kl_trajectory_classifier_4class` (didn't change between cuts because
the four-class shape is the commitment, not the threshold-formula).

**(Part 2, perturbation_align)** That the runner's
`world_event.jsonl` is a single flat JSONL file (not a sharded
directory like agent_step) — the design choice from Phase 1 (low-volume,
human-inspectable) made the aligner's read path trivial (one file
open, line iteration) but the agent-step side complex (parquet shard
iteration with sorted glob). The asymmetry is structural, not
incidental: the streams that carry many small records (agent_step,
dream_rollout) shard; the streams that carry few records (world_event,
replay_meta) don't.

**(Part 3, prompt_builder)** The "reflection without self-modeling"
test was straightforward to write but tricky to pass. The first cut
of the reflexive-attention fragment used the word "self-modeling"
in *negative* framing ("the criterion is about latent-state coupling,
not introspective access, not self-modeling"); my first test forbade
the substring "self-model" and tripped. The fix was tighter: forbid
the *phrase patterns* the probe2 plan §2.9 names ("Io's self-model",
"Io's self-knowledge", "Io modeling its own modeling") — assertion
shapes that *claim* a self-model exists — and allow the bare word in
negative framing. The test is closer to the synthesis's intent now;
"the boundary is named, not claimed."

**(Part 4, llm_caller)** The retry path's interaction with
`pydantic.ValidationError`. The first cut caught only `RuntimeError`,
on the principle that the SDK raises `RuntimeError` for transport
failures. But the structured-output parsing happens inside the call
(via `BatchPayload.model_validate`), and a malformed payload raises
`ValidationError`, not `RuntimeError`. The fix was a wider except
clause: `(ValidationError, ValueError, RuntimeError)`. The
test for the retry path passes either kind through and exercises
both. The lesson: when the LLM call and the parsing are inside the
same client method, the exception surface is union'd; defensive
exception coverage at the caller is the right place.

**(Part 5, orchestrator)** The spec phrase "the orchestrator
constructs the runner with `pre_reg_dir` set per Phase 5's contract"
vs the semantic check `test_orchestrator_does_not_construct_actor_or_world_model`.
The two cannot both be satisfied literally. I almost wrote the
construct-runner path before realizing the semantic check is the
load-bearing one. The reconciliation in the orchestrator's module
docstring is explicit; this entry's "what was built / Part 5"
section documents the decision. The lesson: when a spec has
internal-inconsistency, the *test* names what actually matters; the
prose is the load-bearing surface that may need to lose.

**(Part 7, semantic checks)** Monkey-patching `pathlib.Path.open` to
record writes worked cleanly but had one false-positive risk: the
parquet shard writes (during fixture setup) use `Path.open("wb")`
and would be recorded. Solution: clear the write log between
fixture setup and the orchestrator call. The fixture writes
*before* the test exercise; the orchestrator's writes are the only
thing the assertion sees. A future contributor extending this
test pattern to a stronger invariant ("the orchestrator never
opens a fixture-side parquet shard for write at any time") would
not need the clear, but a stronger invariant isn't what Phase 8
asserts.

**(Cross-Part)** Pydantic v2's frozen models error on assignment
without `# type: ignore[misc]` — I added the ignore in one Phase 8
test, then mypy reported it as unused (Pydantic v2's frozen
assignment error is properly typed without the ignore). Removed
the ignore; mypy clean. The lesson: Pydantic v2's `frozen=True`
type-flags assignment errors directly; the `type: ignore` pattern
from older code doesn't carry forward.

### What's now closed

Phase 7's three newly-open questions all resolve here:

- **The prose-to-statistics gap on each criterion's signals.** Phase 8
  commits each signal's specific estimator (the `ESTIMATOR_*`
  constants), the lag (5 for both autocorrelations), the cluster
  count (4 for k-means), the distance metric (Mahalanobis), the
  threshold percentile (95), the streak length (3), the recovery
  window (`W=50`), the pre-window (50), and the partialling
  procedure (OLS residualization on same-step `z_t` and
  `encoder_embedding_t`). Each choice is journaled in the "Why each
  specific statistic" section above. The tests pin each estimator's
  identity via the constants — a future change to a statistic
  without a journal entry trips the estimator-name test for that
  function.
- **The perturbation-event cross-reference shape for equanimity.**
  Resolved via `kind/mirror/perturbation_align.py`: the orchestrator
  reads the runner's emitted `world_event.jsonl` (on the mirror's
  side of the membrane — the agent process never sees it), aligns
  to the `AgentStep` timeline by `wallclock_ms` (closest within
  tolerance), produces a typed `PerturbationTimeline`, and threads
  it into the equanimity prompt fragment. The criterion does not
  read `world_event` itself; the cross-reference happens at
  prompt-build, orchestrator-side. The sham-event seam plugs in at
  the same point: `payload["is_sham"]` is preserved on the produced
  event, surfaced verbatim in the prompt under
  `SHAM_PERTURBATION_NOTICE`, and the orchestrator's
  `_sham_calibration_check` walks the post-call equanimity reading
  for citation overlap.
- **The held-out-partition-serves-two-jobs note.** The Phase 7 entry
  noted that the held-out partition is both (a) a withheld
  criterion that may move into the active set on evidence, and (b)
  a structural adversarial check on the active-set readings. Phase
  8's build clarifies the two jobs: (b) is what the orchestrator
  *does* (two-pass structure, separate pre-registration, no
  active-set leakage into the held-out prompts — verified by
  `test_held_out_prompts_do_not_reference_active_readings`); (a) is
  what a future round's *decision* is (whether to move the criterion
  on what evidence). The two jobs are now separated structurally:
  the orchestrator only ever runs (b); (a) is a journal-side decision
  at a designated round.

### What's now newly open

- **Whether `second_order_volition` ever moves into the active set,
  and on what evidence (Phase 7's §3 carried forward and sharpened).**
  The Phase 8 build clarifies that the held-out partition is the
  structural adversarial check, distinct from the
  active-set-promotion decision; it does not answer when promotion is
  justified. The empirical question stays the same — what would
  justify moving the criterion from "adversarial check on the active
  set" to "in the active set"? Probably a round where the active-set
  readings have stabilized, the (d) within-trajectory-scalar-shuffle
  control has been run (Probe 3), and the column-init confound is
  genuinely ruled out for that round's telemetry. Phase 8 freezes the
  partition into code; the promotion decision is journal-side.
- **The round-to-round comparability of readings across
  statistic-config changes.** Each round's `StatisticConfig` is
  journaled (via the `PreRegistration` and the `PassResult.notes`),
  but cross-round comparison of `StatisticResult.value` requires the
  configs match in the load-bearing fields (lag, kmeans_k,
  pre_window, etc.). A round that changes a config produces results
  that are *not directly comparable* to a prior round — the
  comparison protocol needs its own structure, probably a
  cross-round-diff record that names which configs differ and which
  fields changed. Phase 12's calibration driver is the natural place
  for it; Phase 8 just emits the per-pass record. The risk to watch:
  a round that tunes a config in response to its own reading
  (overfitting the statistic to the round's telemetry) — the
  pre-registration discipline counters this, but Phase 8 doesn't
  *structurally* enforce "the config commits before the reading
  runs." A future round that tightens this is the second
  open thread.
- **The LLM-caller surface — calibration drift on the model,
  structured-output schema fragility, role-prompt interference.**
  Phase 8 commits `gemini-2.5-pro` and the role-split prompts; the
  open question is what happens when the model is upgraded or
  deprecated. A model upgrade that changes the LLM's stance under
  the same prompts (the same fragment, the same system prompt, the
  same telemetry — but a different reading) is a real risk for
  cross-round comparability. The fallback to `gemini-2.5-flash` is
  documented but unexercised. The structured-output schema fragility
  question: Gemini's `response_schema` mechanism with a recursive /
  union-typed Pydantic model has known edge cases (the
  `BatchPayload`'s nested list of payloads has been fine in synthetic
  tests but the real-LLM behavior is unverified). Phase 12's
  calibration driver will exercise the real model; the open
  question is whether the readings stabilize at all under repeated
  exercise. The role-prompt-interference question — whether a
  primary call's outputs *contaminate* the adversarial call's
  outputs even with separate calls — is structurally guarded
  against by the orchestrator's call separation (each role's call
  is a fresh API invocation; no state shared), but the same model
  reading two prompts may show order-dependent stance drift. Worth
  watching.
- **The structural-vs-semantic test pattern question.** Phase 8's
  three semantic checks closed the Phase 7 newly-open §2 ("no
  helper, the shapes are different"), but a future Phase 9 (judge)
  or Phase 10 (stability) read-only check may produce a fourth
  structural-name instance, which would tip the balance toward a
  helper. The third structural-name instance (Phase 7) was already
  duplicate-pressure; a fourth would resolve it. Watch for the
  pattern.

### Out of scope (preserved from the plan)

The calibration driver / smoke harness (Phase 12); any change to Io,
the actor, the world model, the dream state, or the runner's
training loop; multi-round analysis across passes (Phase 8 produces
a single pass result; aggregation across rounds is later); the
pre-registration template revision (Phase 5's template stays; the
build experience confirms the unified-vs-per-surface split from
Phase 5's newly-open question is fine as-is, but no refactor in
Phase 8); real LLM API calls in tests (all tests use mocks).

Phase 8 touched `kind/mirror/statistics.py` (new),
`kind/mirror/perturbation_align.py` (new),
`kind/mirror/prompt_builder.py` (new),
`kind/mirror/llm_caller.py` (new),
`kind/mirror/orchestrator.py` (new),
`kind/mirror/__init__.py` (re-exports extended),
`tests/test_statistics.py` (new),
`tests/test_perturbation_align.py` (new),
`tests/test_prompt_builder.py` (new),
`tests/test_llm_caller.py` (new),
`tests/test_orchestrator.py` (new),
`tests/test_orchestrator_one_way_invariant.py` (new), and this
journal entry. No other files.

---

## Phase 12 — calibration driver and smoke harness (2026-05-13)

*Calibration driver, sham-schedule injection, round-diff record,
LLM-call audit, smoke harness, first real-Gemini run; four
schema/SDK fixes journaled · 2026-05-13*

The phase was scoped as a stress test of the full mirror execution
plane — Phase 8's wiring against a real LLM, on real Probe 1 and
Probe 1.5 telemetry, with the calibration discipline landing as
executable code rather than journal-side commitment. Four
specific findings surfaced (each predicted in the plan's "What
Phase 12 is allowed to find" list); the smoke completed cleanly on
attempt #4 against `gemini-2.5-pro` with 40 LLM calls, 0 retries,
0 failures, 22.3 min wallclock, 0/10 sham admissions. The
calibration came back clean.

The readings produced are not load-bearing scientific findings
about Io. They are load-bearing engineering findings about the
mirror. The four schema/SDK fixes and the cross-pass stance
analysis below are what this phase actually delivered.

### What was built

Eleven new files on the calibration plane plus surgical updates to
`llm_caller.py` and `orchestrator.py`. The plane lives entirely
under `kind/mirror/calibration/` with one CLI entry point under
`scripts/`.

**Part 1 — `kind/mirror/calibration/round.py`.** The round as
first-class object. `RoundConfig(BaseModel, frozen=True)` carries
every choice the round commits to before the first pass runs:
`round_id` (snake_case, regex-validated), `checkpoints: tuple[
CheckpointSpec, ...]`, `passes_per_checkpoint`,
`statistic_config`, `llm_config`, `sham_schedule`,
`active_registry`, `held_out_registry`,
`pre_registration_template_path`, `column_init`, `builder_mode`,
`perturbation_tolerance_ms`, `notes`. The plan's
`checkpoint_ids: tuple[str, ...]` field was implemented as
`tuple[CheckpointSpec, ...]` because the round driver needs the
on-disk `run_dir` to find telemetry; the CLI accepts the
flat per-spec arguments and constructs the typed record
internally. A `model_validator` enforces sham-schedule consistency
(every entry's `(checkpoint_id, pass_index)` must reference a
checkpoint and pass the round will actually run); the per-field
validator pins `round_id` to the same snake_case regex as
`Criterion.id`. `RoundResult(BaseModel, frozen=True)` carries
`round_config`, `pass_results`, `sham_findings_summary`,
`llm_call_records: tuple[LLMCallRecord, ...]`,
`round_wallclock_ms`, `notes`. The plan's RoundResult shape was
extended with `llm_call_records` so the cross-round audit can
aggregate without separate plumbing. `ShamFindingsSummary`
aggregates admissions by criterion / checkpoint / role plus a
`total_sham_events` denominator. `run_round(config, *,
output_dir, llm_client=None)` validates → emits round-level
pre-registration → per-checkpoint per-pass loop → aggregates
sham findings → atomic write to
`output_dir/mirror/rounds/{round_id}.json`.

**Part 2 — `kind/mirror/calibration/sham_schedule.py`.**
`ShamScheduleEntry(BaseModel, frozen=True)` with a
`@model_validator` that *forces* `sham_payload["is_sham"] = True`
— a sham entry whose payload silently sets `is_sham=False` would
defeat the calibration check, so the constructor refuses to
construct one. `ShamSchedule(BaseModel, frozen=True)` enforces no
collision on `(checkpoint_id, pass_index, sham_t)`.
`generate_sham_schedule(checkpoint_ids, passes_per_checkpoint,
real_perturbations_per_pass, shams_per_pass, telemetry_length,
seed)` produces a deterministic schedule via `random.Random(seed)`
(no global RNG touched). The sham timestamps are drawn uniformly
from `range(1, telemetry_length - 1)` — the bounds avoid the
first step (where the self-prediction-error masked flag is set
and the equanimity signal is degenerate by construction) and the
last step (no post-perturbation samples for the recovery
window). `inject_sham_events(perturbation_timeline, sham_entries,
*, agent_step_wallclock_lookup)` produces a new
`PerturbationTimeline` with the shams merged at their scheduled
timestamps; the original timeline is not mutated (both are
Pydantic frozen). A sham whose `t` collides with a real
perturbation `t` raises via the timeline's sorted-unique
validator — the calibration cannot tolerate the ambiguity.

**Part 3 — `kind/mirror/calibration/round_diff.py`.**
`ConfigFieldChange(BaseModel, frozen=True)` with
`(field_path, prior_value, current_value, rationale)`; the
rationale field's validator rejects empty strings — a round that
changes a config field without a rationale is a structural
violation. `CriterionSetChange(BaseModel, frozen=True)` carries
the four sorted criterion-id tuples
(`prior_active`, `current_active`, `prior_held_out`,
`current_held_out`) plus a required rationale.
`RoundDiff(BaseModel, frozen=True)` has
`statistic_config_changes: tuple[ConfigFieldChange, ...]`,
`llm_config_changes: tuple[ConfigFieldChange, ...]`,
`criterion_set_changes: CriterionSetChange | None`, plus the two
round_ids and a free-text notes field. `compute_round_diff(prior,
current, *, rationales=None, notes="")` walks the two configs
field-by-field and requires a rationale per change (missing
rationale raises `ValueError`). Empty `rationales` is allowed
*only* if the configs are bit-identical on the diffed dimensions
— which is exactly the Phase 12 smoke's case. The diff scope is
intentionally narrow: `statistic_config`, `llm_config`, and the
criterion-set partition only. `checkpoint_ids`, `sham_schedule`,
and `round_id` differ between rounds *by design* and are out of
diff scope; surfacing them as "changes" would drown the
meaningful drift in noise.

**Part 4 — `kind/mirror/calibration/smoke.py`.**
`run_phase_12_smoke(probe_1_checkpoint, probe_1_5_checkpoint, *,
output_dir, llm_api_key_env_var, llm_client=None, notes="")`
drives the full smoke. Module-level constants journal the Phase
12 commitments: `PHASE_12_PASSES_PER_CHECKPOINT=5`,
`PHASE_12_REAL_PERTURBATIONS_PER_PASS=2`,
`PHASE_12_SHAMS_PER_PASS=1`, `PHASE_12_PROBE_1_SEED=42`,
`PHASE_12_PROBE_1_5_SEED=43`,
`PHASE_12_PROBE_1_ROUND_ID="phase_12_probe_1_round"`,
`PHASE_12_PROBE_1_5_ROUND_ID="phase_12_probe_1_5_round"`. The
two rounds run sequentially (not in parallel — keeps the
LLM-call audit's records in stable order and avoids interleaving
the rounds' API rate limits). `Phase12SmokeResult(BaseModel,
frozen=True)` carries `probe_1_round`, `probe_1_5_round`,
`round_diff`, `llm_call_audit`, `wallclock_ms`, `notes`.
On-disk: `output_dir/mirror/phase_12_smoke_result.json`.

**Part 5 — `kind/mirror/calibration/llm_audit.py`.**
`LLMCallRecord(BaseModel, frozen=True)` per attempt:
`(round_id, pass_index, checkpoint_id, role, attempt_number,
request_timestamp_ms, response_timestamp_ms, latency_ms,
model_name, prompt_token_count, response_token_count, outcome,
error_message)`. The outcome literal is `Literal["success",
"validation_error", "value_error", "runtime_error",
"max_retries_exceeded"]`. One record per attempt, not per call:
a single `call_mirror_llm` invocation that succeeds on attempt
3 of 4 produces three records (two failure attempts + one
success). A persistent failure produces a synthetic
`max_retries_exceeded` record before `MirrorLLMError` is raised.
`LLMCallAudit(BaseModel, frozen=True)` carries the records tuple
plus five totals (`total_calls`, `total_retries`,
`total_failures`, `total_wallclock_ms`, `total_tokens_in`,
`total_tokens_out`); `LLMCallAudit.from_records(records)` is the
helper that derives the totals at construction.
`LLMCallRecordCollector` is the per-pass mutable accumulator the
orchestrator threads through — *not* a Pydantic model, just a
thin container that satisfies the
`LLMRecordSink(Protocol)` declared in `llm_caller.py` (the
protocol-based decoupling avoids importing the calibration
package from `llm_caller`).

**Part 6 — `scripts/run_phase_12_smoke.py`.** CLI entry point.
Takes flat per-spec arguments
(`--probe-1-run-id`, `--probe-1-run-dir`,
`--probe-1-checkpoint`, plus the matching three for Probe 1.5,
plus `--output-dir`, `--llm-api-key-env-var`, `--notes`). Calls
`load_dotenv()` at script entry so `GEMINI_API_KEY` from the
project-root `.env` is available without pre-shell-export
(matches `scripts/call_mirror.py`'s pattern). Prints the one-line
summary on completion: `phase_12_smoke: passes=N calls=N
retries=N failures=N wallclock_ms=N tokens_in=N tokens_out=N
sham_admissions=N`.

**Part 7 — Tests.** Five new test files, 70 unit tests, plus
`tests/conftest.py` with the `--run-real-api` flag and
`real_api` mark.
- `tests/test_llm_audit.py` (21 tests) — record validation,
  audit aggregation, collector accumulation, end-to-end through
  `call_mirror_llm` for success / retry / max-retries-exceeded
  paths.
- `tests/test_sham_schedule.py` (23 tests) — entry validation
  (forced `is_sham=True`), schedule no-collision invariant,
  generator determinism (same seed → same schedule), placement
  window bounds, injection immutability + sorted-order
  preservation + collision rejection.
- `tests/test_round_diff.py` (12 tests) — empty-when-identical,
  rationale-required, per-config-prefix routing, criterion-set
  change detection, serialization round-trip.
- `tests/test_round.py` (14 tests) — `RoundConfig` validation
  (snake_case, sham-schedule consistency), the
  frozen-after-pre-registration invariant (asserts
  `config.statistic_config = X` raises `ValidationError`), end-
  to-end `run_round` with synthetic telemetry + a `MockLLMClient`,
  on-disk artifact verification, atomic write, serialization
  round-trip.
- `tests/test_phase_12_smoke_real_api.py` (1 test, skipped by
  default) — opt-in real-API end-to-end against `gemini-2.5-pro`,
  reads source-checkpoint paths from `KIND_PHASE_12_PROBE_1_*` and
  `KIND_PHASE_12_PROBE_1_5_*` env vars.
- `tests/conftest.py` — adds `--run-real-api` pytest option and
  the collection-modifier that skips `@pytest.mark.real_api`
  unless the flag is passed or `GEMINI_API_KEY` is set in the
  environment.

**Part 8 — Updates to existing modules.**
- `kind/mirror/__init__.py` — re-exports the Phase 12 surface
  (`RoundConfig`, `RoundResult`, `run_round`, `ShamSchedule`,
  `ShamScheduleEntry`, `generate_sham_schedule`,
  `inject_sham_events`, `RoundDiff`, `ConfigFieldChange`,
  `CriterionSetChange`, `compute_round_diff`,
  `Phase12SmokeResult`, `run_phase_12_smoke`, `LLMCallRecord`,
  `LLMCallAudit`, `LLMCallRecordCollector`, plus the
  `CallOutcome` and `LLMRecordSink` types now in `llm_caller`).
- `kind/mirror/llm_caller.py` — added `CallOutcome` literal and
  `LLMRecordSink(Protocol)`; added optional `record_sink:
  LLMRecordSink | None = None` parameter to `call_mirror_llm`;
  emits one record per attempt + a synthetic
  `max_retries_exceeded` record on retry-budget exhaustion. Two
  helpers: `_classify_exception(exc)` maps the caught exception
  to the audit outcome literal, and `_extract_token_counts(payload)`
  is best-effort against `payload.usage_metadata` (returns
  `(None, None)` when the SDK doesn't expose it — Phase 12's
  audit shows `tokens_in=None tokens_out=None` for this reason;
  see "What surprised" below). Also: the schema-munger pipeline
  (`_to_gemini_schema` and friends — see Finding #1 / #2).
- `kind/mirror/orchestrator.py` — added two optional kwargs to
  `run_adversarial_pass`: `injected_sham_entries: tuple[Any,
  ...] = ()` (typed as `Any` to avoid an import cycle through the
  calibration package; the orchestrator does a lazy import of
  `inject_sham_events` only when the tuple is non-empty) and
  `record_sink: LLMRecordSink | None = None` (threaded through
  to all four `call_mirror_llm` calls per pass). The
  pre-existing semantic read-only invariant test continues to
  pass — Phase 12 added no new write surfaces on Io's side.

**Decision: do not extend `PreRegistration`.** The plan's text
"StatisticConfig commits at pre-registration time" and "Sham
timestamps are committed in the pre-registration record" was
interpreted as "the on-disk pre-registration artifact set" —
both Phase 0's `PreRegistration` JSONL (per-pass criterion-shape
commitment) AND Phase 12's `RoundConfig` JSON (per-round
statistic + sham commitment) — written before the first pass
runs. The round driver writes `RoundConfig` to
`output_dir/mirror/pre_reg/round_{round_id}/round_config.json`
*before* opening the per-pass loop; Pydantic `frozen=True` on
`RoundConfig` structurally enforces "no mutation after the round
starts". This avoided extending the Phase 0 `PreRegistration`
schema (and the resulting v0.4.0 schema export bump the plan
called out as "may require") at the cost of a separate file the
journal must explicitly read alongside `pre_reg.jsonl`. The
test `test_round_config_frozen_after_pre_registration` asserts
the structural invariant directly. v0.4.0.json was *not*
generated.

### The four schema/SDK fixes

Each was predicted by the plan as the kind of thing Phase 12
should surface. The fixes lived at the schema-munger / retry-
loop layer; no Phase-0 contracts were relaxed.

**Finding #1 (attempt #1) — Pydantic emits `prefixItems` for
`tuple[int, int]`; Gemini's structured-output Schema validator
implements OpenAPI 3.0 (no `prefixItems`).** `StructuredClaim`'s
`cited_episode_range: tuple[int, int]` and
`cited_step_range: tuple[int, int]` produce JSON Schema
`{"type":"array", "minItems":2, "maxItems":2, "prefixItems":
[{"type":"integer"}, {"type":"integer"}]}`. The genai SDK
constructs a Pydantic `Schema` model from the schema dict and
rejects `prefixItems` with `"Extra inputs are not permitted"`
client-side, before the API call ever fires. Fix: a recursive
`_to_gemini_schema(dict)` munger in `llm_caller.py` that
converts `prefixItems` → `items` (taking the first prefix item's
type; the `minItems` / `maxItems` Pydantic emits preserve the
length). The lossy edge — heterogeneous prefix tuples would lose
per-position typing — doesn't apply: every `prefixItems` use in
`BatchPayload` is a homogeneous `[{"type":"integer"},
{"type":"integer"}]` int pair. The response parser
(`BatchPayload.model_validate`) coerces lists back to tuples per
Pydantic's normal collection coercion, so the
schema-sent-to-Gemini differs from the schema-the-response-is-
validated-against by exactly the prefixItems → items
conversion. `StructuredClaim` itself is unchanged — the Phase 0
contract holds.

**Finding #2 (attempt #2) — Gemini also rejects
`additionalProperties`, `$defs`/`$ref`, `anyOf`, `title`,
`$schema`, and several other JSON Schema keys Pydantic emits.**
Attempt #2 reached Gemini's API but got `400 INVALID_ARGUMENT`:
`"Unknown name 'additional_properties' at
'generation_config.response_schema'"` (note: snake_case in the
error message; Gemini's API uses snake_case for field names).
Pydantic emits `additionalProperties: false` on every model
with `extra='forbid'`; Gemini's Schema vocabulary doesn't
include the key at all. Investigation confirmed Gemini's
`response_schema` is a strict subset of OpenAPI 3.0:
`type, format, description, nullable, enum, maxItems, minItems,
items, properties, required, propertyOrdering` — and not much
else. Fix: `_to_gemini_schema` was upgraded to a full pipeline:
(1) inline `$ref` references using `$defs` (Gemini doesn't
support refs at all), then drop `$defs`; (2) convert `anyOf:
[X, {"type":"null"}]` (Pydantic's `Optional[X]` form) → X with
`nullable: true`; (3) convert `prefixItems` → `items`; (4) strip
the unsupported keys (`$schema, $id, title,
additionalProperties, definitions, default, examples, const,
allOf, not, oneOf, discriminator`). The pipeline is sequential
and idempotent. The lossy edges — `additionalProperties: false`
becomes implicit, `title` is informational — don't matter
because the response parser re-asserts the full Phase-0
contract via `BatchPayload.model_validate`. The munger-test
suite in `tests/test_llm_caller.py` pins each conversion against
regression: `test_gemini_schema_munger_inlines_refs_and_drops_defs`,
`test_gemini_schema_munger_strips_additional_properties_false`,
`test_gemini_schema_munger_converts_anyof_with_null_to_nullable`,
`test_gemini_schema_munger_strips_title`,
`test_gemini_schema_munger_preserves_min_max_items`,
`test_gemini_schema_munger_passes_through_non_dict_non_list`.

**Finding #3 (attempt #3) — `google.genai.errors.APIError`
subclasses `Exception`, not `RuntimeError`; the caller's retry
catch was too narrow.** Attempt #3 cleared the schema gates and
ran successfully through Round 1 Pass 1 (the orchestrator wrote
a 94KB `ckpt-000001.json` pass-result file at 13:33:36, with
valid Gemini structured output — proof the schema fix worked).
Then a `google.genai.errors.ServerError: 503 UNAVAILABLE`
propagated out of the SDK on Pass 2's held-out call: the SDK's
own tenacity-backed retry layer exhausted its budget and re-
raised. My caller's `except (ValidationError, ValueError,
RuntimeError)` clause didn't catch `ServerError` (MRO:
`ServerError → APIError → Exception → BaseException`), so the
error propagated past the per-call retry budget I had configured
and crashed the smoke. Fix: imported `APIError` from
`google.genai.errors`, added it to a module-level
`_RETRYABLE_LLM_ERRORS` tuple, and changed the retry except
clause to use the tuple. The catch is intentionally not broader
than `Exception` would be — programming bugs (TypeError,
AttributeError) still propagate immediately rather than being
silently retried.

**Finding #4 (also attempt #3-driven) — hammering on a 503 in
zero time doesn't help; backoff was missing.** The original
retry loop sent the same request immediately on the next
iteration; a 503 returning in <100ms means we'd hit the API
again in <100ms. Fix: `_retry_backoff_seconds(attempt_index)`
returns `min(2.0 ** attempt_index, 30.0)` for attempt_index ≥ 1
(0 for the first attempt), and `time.sleep(backoff)` runs before
each retry attempt. This matches the SDK's tenacity defaults
and gives sustained outages a chance to clear. The cost: tests
that exercise the retry path now sleep through the backoff
(test suite went from ~0.5s to ~8s); not blocking, and the
real-world correctness gain is what justifies it.

**The findings were the predicted set.** The plan's "What Phase
12 is allowed to find" §1 named "structured-output schema
fragility" first; #1 and #2 are exactly that. §3 named "LLM-
caller calibration drift / fragility / interference"; #3 is the
APIError-class-hierarchy hole, and #4 is the missing-backoff
hole. The plan's prediction was load-bearing — the structural
shape of Phase 12 (smoke, audit, retry budget, journal) was
designed to surface and contain exactly these. None of the four
required relaxing a Phase 0 contract; all four landed at the
adapter layer between the project's typed schemas and the SDK's
client.

### The smoke run — what actually happened

Attempt #4 (background id `bk2lf6k19`) launched at 13:38 with
the resilience patches in place. It ran for **22 min 17 s
wallclock** and completed cleanly:

```
phase_12_smoke: passes=10 calls=40 retries=0 failures=0
              wallclock_ms=1336977 tokens_in=None tokens_out=None
              sham_admissions=0
```

**0 retries, 0 failures across 40 calls.** The retry-resilience
patches from findings #3 and #4 sat unused this run — the
schema fixes from findings #1 and #2 did all the heavy lifting.
Every Gemini call returned valid structured output on the first
attempt. Mean per-call latency was 33s (range 19–57s, median
32s); primary calls averaged 34s, adversarial 32s. Round 1
took 10.79 min for 5 passes against the Probe 1 checkpoint;
Round 2 took 11.49 min for 5 passes against Probe 1.5.

`tokens_in / tokens_out = None` is a known limitation of
`_extract_token_counts`: the helper looks for
`payload.usage_metadata` on the parsed `BatchPayload`, but the
SDK exposes usage metadata on the *response wrapper object*,
not on the parsed Pydantic model the wrapper carries. The
caller currently returns just the parsed payload (the Phase 0
shape contract); threading the wrapper through to capture token
counts is a Phase 13 follow-up. Documented at the helper's
docstring; not a blocker.

### The 10 readings — what the LLM said

Per the plan, this section freezes what the calibration
actually produced. Format: per-pass active-primary +
active-adversarial readings on the two active criteria
(reflexive_attention, equanimity_perturbation_recovery), plus
the held-out (second_order_volition) primary + adversarial.
Surfaces are the `reading_surface` field on each claim.

**Round 1 — Probe 1 (`probe1-20260503-123926/ckpt-000001`).**
- Pass 0: reflexive_attention primary 2 claims (head_internal ×2)
  / adversarial 2 (head_internal ×2). Equanimity primary 3
  claims (substrate_side ×2, behavior_side ×1) / adversarial 2
  (substrate_side, behavior_side). Held-out
  (second_order_volition) primary 2 / adversarial 2 (one
  substrate_side + one behavior_side each side). Sham at t=913,
  clean.
- Pass 1: reflexive 2/2 (all head_internal). Equanimity 2/2 (one
  substrate, one behavior each side). Held-out 2/2 split.
  Sham at t=205, clean.
- Pass 2: reflexive 2/1 (head_internal). Equanimity 2/2.
  Held-out 2/2. Sham at t=2254, clean.
- Pass 3: reflexive 2/1. Equanimity 2/2. Held-out 2/2.
  Sham at t=2007, clean.
- Pass 4: reflexive 2/1. Equanimity 2/2. Held-out 2/2.
  Sham at t=1829, clean.

**Round 2 — Probe 1.5 (`probe1_5-20260506-202458/ckpt-000001`).**
- Pass 0: reflexive 2/1 (head_internal). Equanimity 2/2.
  Held-out 2/2. Sham at t=316, clean.
- Pass 1: reflexive 2/1. Equanimity 2/2. Held-out 2/2.
  Sham at t=2344, clean.
- Pass 2: reflexive 2/2 (head_internal). Equanimity 2/2.
  Held-out 2/2. Sham at t=1180, clean.
- Pass 3: reflexive 2/1. Equanimity 2/2. Held-out 2/2.
  Sham at t=3790, clean.
- Pass 4: reflexive 2/1. Equanimity primary 2 / adversarial 3
  (substrate_side ×2, behavior_side ×1) — outlier, the only
  Round 2 pass with non-2-claim equanimity adversarial.
  Held-out 2/2. Sham at t=3031, clean.

A representative primary reflexive_attention claim, verbatim
from R1P0 (head_internal surface,
`cited_scalar_field=latent_self_reference_t`,
`cited_value=0.055218`):

> The waking-state latent self-reference signal (0.055218) does
> not exceed its matched shuffled-time control (0.057693).

A representative adversarial claim from the same pass (same
surface, same cited fields):

> The partial autocorrelation of the agent's latent state (h_t)
> during world interaction (lag 5) is 0.055218, which is less
> than the 0.057693 value from the matched shuffled-time
> control. This fails to provide evidence for reflexive
> attention beyond chance.

The free-text notes from R1P0 primary equanimity reading,
verbatim:

> No non-sham perturbations occurred during the evaluation; only
> a single sham calibration event was recorded. As a result, all
> estimators for perturbation response and recovery had zero
> samples. The falsifier conditions for equanimity could not be
> evaluated. Per the criterion's non-falsifying-non-admission
> clause, this outcome is read as 'perturbation not registered',
> and equanimity is neither supported nor refuted by this data.

This last is the load-bearing observation about the
calibration's actual ground: see "What surprised" below.

### Sham calibration findings — clean

**0 admissions of 10 sham events** (5 per round, one per pass).
Every primary equanimity reading either had no claim citing a
step range covering the sham timestamp, or — more often, since
no real perturbations existed (see below) — produced a
non-falsifying-non-admission per the criterion's prose. The
sham-aware prompt-fragment notice (`SHAM_PERTURBATION_NOTICE`
from Phase 8) plus the prompt's explicit "Sham perturbations: an
equanimity admission at a sham timestamp is a calibration
failure; flag it" line in the adversarial system prompt did the
job. The orchestrator's `_sham_calibration_check` walked all 10
sham events and found 0 overlapping primary equanimity claims.

Breakdown (all zero):
- by_criterion: `{}` (only `equanimity_perturbation_recovery`
  is checked at Phase 12; the dict's shape generalizes to later
  phases extending the check)
- by_checkpoint: `{}`
- by_role: `{}`

This is the calibration discipline working as designed. The
plan's load-bearing finding #4 ("the mirror may admit equanimity
at sham timestamps") did not materialize on this checkpoint
pair; the journal-side recommendation it asked for (which prompt
fragments admitted, the hypothesis about why) is not needed.
But see the "newly-open" section: a clean result on these two
checkpoints is not a guarantee for richer telemetry. The Phase
12 commitment was that whatever the result was, the structural
machinery would surface and journal it; it has.

### Stance drift across the five passes

Per the plan, this is the load-bearing finding for the project's
epistemic discipline: how stable is a single LLM reading
under repeated invocation against the same telemetry, the same
prompt, the same model? Five passes per checkpoint × two
checkpoints = ten data points per criterion per role.

**Per-pass claim counts (active primary):**
| Round | Criterion | Pass 0–4 counts | Mean | StDev |
|---|---|---|---|---|
| R1 | reflexive_attention | [2, 2, 2, 2, 2] | 2.0 | 0.0 |
| R1 | equanimity | [3, 2, 2, 2, 2] | 2.2 | 0.4 |
| R2 | reflexive_attention | [2, 2, 2, 2, 2] | 2.0 | 0.0 |
| R2 | equanimity | [2, 2, 2, 2, 2] | 2.0 | 0.0 |

**Per-pass claim counts (active adversarial):**
| Round | Criterion | Pass 0–4 counts | Mean | StDev |
|---|---|---|---|---|
| R1 | reflexive_attention | [2, 2, 1, 1, 1] | 1.4 | 0.5 |
| R1 | equanimity | [2, 2, 2, 2, 2] | 2.0 | 0.0 |
| R2 | reflexive_attention | [1, 1, 2, 1, 1] | 1.2 | 0.4 |
| R2 | equanimity | [2, 2, 2, 2, 3] | 2.2 | 0.4 |

**Per-pass claim counts (held-out, both rounds, both roles):**
all 5 of 5 passes returned 2 claims (one substrate_side, one
behavior_side). 0 stdev across all 20 held-out readings.

**Reading:** primary stance is highly stable. 4 of 4 primary
columns have stdev ≤ 0.4; 2 of 4 have stdev exactly 0.0. The
lone primary outlier is R1P0 equanimity (3 claims vs the 2-claim
mode); the additional claim is the third surface the criterion
covers, so the variance is "did the LLM cite all three surfaces
or only two" — a within-prompt-fragment choice, not a
substantive disagreement.

Adversarial stance is *less* stable than primary. The
reflexive_attention adversarial column drops from 2 → 1 mid-
round in both rounds: the LLM elected to combine the waking and
dream signals into a single composite claim in some passes
("the waking signal fails its control AND the dream signal's
sample size is too small to count") rather than two separate
ones. This is structural compression, not stance drift — both
forms reach the same conclusion (null not rejected) — but the
*number of claims* fluctuates as a function of the LLM's
prose-organization choice on the day. The Pass 4 R2
equanimity-adversarial 3-claim outlier is the symmetric move
in the other direction (one extra substrate_side claim).

**The implication for later phases:** primary readings can
carry weight as single-pass observations for criteria whose
prose has clear three-surface structure (the LLM tends to
produce one-claim-per-surface). Adversarial readings need the
multi-pass aggregate — counting *which falsifiers fired* across
passes is more stable than counting *how many claims were
made*. A Phase 13 aggregation protocol should consume the
multi-pass record at the falsifier level, not the claim level.

### LLM call audit summary

```
total_calls            = 40
total_retries          = 0
total_failures         = 0
total_wallclock_ms     = 1,321,511 (22.03 min summed; the
                         smoke's 22.28 min wallclock includes
                         sequential-but-inter-call orchestration
                         overhead)
total_tokens_in        = None  (see below)
total_tokens_out       = None
```

**Latency distribution (per individual call attempt, n=40):**
- min: 19.1 s
- max: 57.6 s
- mean: 33.0 s
- median: 32.4 s
- stdev: 9.2 s

**By role:** primary mean 34.2s (median 32.4s, n=20);
adversarial mean 31.9s (median 32.1s, n=20). Primary slightly
slower on average — could be the Advocate stance generates
slightly longer outputs (more justification per claim), or
could be noise in n=20.

**Token counts: None.** The audit's `total_tokens_in` and
`total_tokens_out` are `None` because every record's
per-attempt counts are `None`. `_extract_token_counts(payload)`
in `llm_caller.py` is best-effort against
`payload.usage_metadata`; the SDK exposes usage on the
*response wrapper object*, not on the parsed BatchPayload that
my client returns. Phase 13 follow-up: thread the SDK response
wrapper through `_GeminiLLMClient.generate_batch` so the audit
captures token counts. Documented at the helper's docstring.

The audit is well-formed for later analysis: 40 records, every
field populated except the deliberately-`None` token slots,
serializes round-trip via `LLMCallAudit.model_validate_json(
audit.model_dump_json())`. Ready for Phase 13 consumption.

### What surprised

**The dominant surprise: the source telemetry has no real
`builder_perturbation` events.** The Probe 1 and Probe 1.5 runs
emit `env_reset`, `internal_stochasticity_aggregate`, and
`mirror_marker` events to `world_event.jsonl`, but the
`builder_perturbation` event type is reserved for Probe 4
(builder-as-perturbation). The aligner's
`align_perturbations(world_event_log_path, telemetry_dir, ...)`
filters to `event_type == "builder_perturbation"` and finds
zero. The Phase 12 round commits to
`real_perturbations_per_pass=2` as a pre-registration field;
the actual data delivered 0. The injected sham is the *only*
perturbation in each pass's timeline.

The mirror handled this honestly: it produced a non-falsifying-
non-admission for equanimity, citing the zero-sample
`recovery_lag_steps`, `policy_entropy_t`, and `posterior_kl_t`
estimators directly. The free-text notes named the condition
explicitly: *"only a single sham calibration event was recorded.
As a result, all estimators for perturbation response and
recovery had zero samples. The falsifier conditions for
equanimity could not be evaluated."* The non-falsifying clause
from `EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE` worked as
designed — the LLM did NOT make up an admission.

This is the calibration's actual product: not "equanimity is
present" or "equanimity is absent" but "equanimity could not be
evaluated, and the criterion's prose tells us how to read that".
The sham-zero result is then *only what it can be*: a verification
that the structural prompt-fragment machinery doesn't admit at
sham timestamps in the absence of any real perturbation. A
richer test — equanimity admissions in the presence of real
perturbations — requires either (a) Probe 4's
builder-as-perturbation infrastructure or (b) a Phase 12
follow-up that injects synthetic *real* perturbations alongside
shams to exercise the discriminative path. The plan's
"Out of scope" section explicitly defers (a) to Probe 4; (b) is
a candidate for later phases.

**Smaller surprises:**
- **The schema munger took two iterations to land.** Finding #1
  caught `prefixItems`; attempt #2 then found `additionalProperties`,
  `$defs`/`$ref`, `anyOf`, `title`. Doing a thorough one-shot
  upgrade after the first failure (rather than iterating) would
  have saved one round-trip. Lesson for Phase 13: when a
  schema-compatibility issue surfaces, audit the *full*
  Pydantic-emitted schema against the SDK's accepted vocabulary
  rather than fix-the-specific-key.
- **`google.genai.errors.APIError` does not subclass
  `RuntimeError`.** I had assumed any SDK-side transport error
  would be in the existing retry catch (RuntimeError). It is
  in fact directly under `Exception`. The MRO check should have
  been done at LLM-caller-build time (Phase 8) — Phase 8's mock
  client never raises APIError, so the gap was invisible.
- **Mean Gemini latency is 33s.** I expected closer to 5-10s
  for a structured-output call with a moderate prompt. The
  slowness might be JSON-schema-validation-side (the SDK
  pre-validates the Schema before sending), prompt-length-side
  (~3-5K tokens of fragments + system prompt), or just
  gemini-2.5-pro under load on the Mac mini's network at the
  time of the run. Worth tracking across rounds.
- **The held-out partition was perfectly stable.** 20/20 readings
  (10 primary + 10 adversarial across two rounds) returned
  exactly 2 claims, one substrate_side and one behavior_side.
  `second_order_volition`'s prose at Phase 7 has clear
  two-surface structure, and the LLM faithfully picked it up
  every time. The active partition's slightly-noisy claim
  counts are the comparison.

### What's now closed

**Phase 8's three newly-open questions (closed by Phase 12).**

1. *Round-to-round comparability under shared StatisticConfig.*
   Closed structurally: `RoundConfig.statistic_config` is a
   pinned `StatisticConfig`; `compute_round_diff` produces an
   empty `statistic_config_changes` for two rounds with bit-
   identical configs (verified: Phase 12 smoke's diff has 0
   changes, confirming the structural verification works).
   Closed semantically: 5-pass / 2-checkpoint stance drift
   analysis is now the discipline; the table above is the
   freeze-record.
2. *LLM-caller calibration drift / structured-output schema
   fragility / role-prompt interference.* Closed structurally
   on the first two: the schema-munger pipeline + the retry-
   budget-with-backoff handle the wire-level fragility surface;
   the LLMCallAudit is the diagnostic surface for any future
   drift. Role-prompt interference is closed by Phase 8's
   call-separation discipline (each role is a fresh API call,
   no state shared); Phase 12's adversarial-stance variance
   table shows variance is real but small and structural
   (claim-count compression vs expansion), not directional.
3. *StatisticConfig commitment timing.* Closed structurally:
   `RoundConfig` is Pydantic frozen; the round driver writes
   it to disk *before* the per-pass loop opens; the
   `test_round_config_frozen_after_pre_registration` test
   asserts the invariant. The commitment lives in the round-
   level pre-registration artifact (`round_config.json`)
   alongside the per-pass `pre_reg.jsonl`.

**The four schema/SDK fixes (closed in code).** Each fix has
unit-test coverage; mypy `--strict` clean; the smoke ran
successfully end-to-end on attempt #4. The compatibility
work between the project's Pydantic-typed schemas and Gemini's
Schema-validator subset is now load-bearing infrastructure.

**The "real LLM API tests" question (closed by conftest.py).**
`tests/conftest.py` adds the `--run-real-api` flag and the
`real_api` mark; Phase 12's
`tests/test_phase_12_smoke_real_api.py` is the first opt-in
real-API test. Future expansion of the real-API surface is a
documented project capability, not an ad-hoc question.

### What's now newly open

**1. The "no real perturbations" condition limits what Phase
12's clean sham result actually proves.** A sham-zero result
requires the mirror to (a) not admit at a sham timestamp AND
(b) have something to discriminate from. With zero real
perturbations in the timeline, condition (b) is vacuous — the
LLM has no equanimity-admission case to make in the first
place, sham or not. The clean result confirms the prompt-
fragment + non-falsifying clause work *in the empty-baseline
case*. A richer test needs the discriminative case. Two paths:
(a) wait for Probe 4 to provide real perturbations; (b) Phase
13 adds a synthetic-real-perturbation injection layer alongside
the existing sham injection, and re-runs against the same
checkpoints. (b) is the smaller move; (a) is the substantive
one. Recommendation: schedule (b) as a Phase 13 follow-up so
the calibration discipline has at least one test against
discriminative data before Probe 4 lands.

**2. The token-count gap in `_extract_token_counts`.** Phase 12's
audit is `tokens_in=None / tokens_out=None` because the SDK
exposes `usage_metadata` on the response wrapper, not on the
parsed BatchPayload. Phase 13 should thread the wrapper through
`_GeminiLLMClient.generate_batch`. Surface area: ~10 lines.
Cost-tracking discipline depends on this — without token counts
the audit can't ground a cost analysis or detect prompt-length
inflation across rounds.

**3. The adversarial-stance claim-count variance is real but
unmeasured below the count level.** The reflexive_attention
adversarial column dropped from 2-claim to 1-claim mid-round in
both rounds. Whether the *content* of the claims drifted or
just their *prose organization* (one combined claim vs two
separate) is not in the data the audit captures — it's only in
the free-text notes. A Phase 13 stance-drift analyzer that
diffs the cited_value / cited_step_range / falsifier fields
across the 5 passes (matching by criterion + reading_surface)
would tell us whether the variance is structural-compression-
only or substantive-disagreement.

**4. The round-diff record's rationale field is enforced at the
ConfigFieldChange level (non-empty validator) but not at the
round-driver level.** The plan's "What's now newly open" §3
asked whether structural enforcement at the round-driver level
beyond the Pydantic non-empty validator was needed. Phase 12
left the question open: the test suite asserts that
`compute_round_diff` raises when a changed field has no
rationale, which is structural enforcement at *call* time, but
nothing prevents a caller from passing a perfunctory rationale
("changed because"). A min-length check or required-citation
pattern (e.g., must reference a journal entry or a source
document) is plausible for Phase 13 once cross-round diffs
become substantive.

**5. The `real_perturbations_per_pass` pre-registration field
is not enforced against the actual telemetry.** The round
commits to "this round expects 2 real perturbations per pass";
the actual data delivered 0 per pass; the round still ran
without raising. The journal entry caught the discrepancy
(this section), but no automated check did. A Phase 13 round-
level audit that compares the committed
`real_perturbations_per_pass` against the actual aligned-
timeline real-perturbation count, and flags discrepancies as a
calibration-soundness violation, would land naturally given
the existing infrastructure.

**6. The schema-munger's lossy edges may matter for richer
schemas.** The pipeline currently handles every key in
`BatchPayload`'s schema correctly because the schema is
relatively simple (no heterogeneous tuples, no allOf, no
discriminator-based unions). Future criterion additions that
introduce richer Pydantic models could trip the lossy edges
(e.g. a tuple of mixed-type elements would lose per-position
typing under prefixItems → items). The munger has unit-test
coverage for the cases it handles; an audit-pass before any
new criterion landed would be cheap insurance.

**7. The LLM produced perfectly-stable held-out readings; this
is suspicious.** All 20 held-out readings (10 primary + 10
adversarial) returned exactly 2 claims with the same surface
distribution (1 substrate_side + 1 behavior_side). Either (a)
`second_order_volition`'s prose at Phase 7 is so clear that the
LLM has no choice in how to organize the response, or (b) the
held-out criterion is not exercising the LLM in a way that
would surface meaningful drift. (a) is what we'd want; (b) is a
co-design failure mode. A Phase 13 investigation: take a Phase
12 held-out reading and run the same prompt fragment against
the LLM five more times in isolation. If the same 2-claim
1-substrate-1-behavior shape emerges, that's evidence for (a).
If the responses diversify in isolation but compress when
adjacent to active readings, that's evidence for prompt-context
interference and a more substantial finding.

### Test/mypy status

- `pytest tests/` (without `--run-real-api`): 873 passing, 1
  pre-existing flaky failure (`test_transport.py::
  test_barrier_queues_mutates_and_drains_in_order` — same as
  Phase 8; concurrency-timing-sensitive), 3 skipped (the real-
  API smoke + 2 pre-existing skips). No Phase 12 regressions.
  Suite runtime: ~34 s.
- `mypy --strict kind/mirror/ kind/observer/ tests/test_round.py
  tests/test_sham_schedule.py tests/test_round_diff.py
  tests/test_llm_audit.py tests/test_phase_12_smoke_real_api.py
  scripts/run_phase_12_smoke.py`: clean (30 source files
  checked).
- `pytest tests/ --run-real-api` (with GEMINI_API_KEY set):
  the Phase 12 smoke runs end-to-end against the real Gemini
  API and passes; the test asserts the result is well-formed
  and the audit shows ≥4 success records per checkpoint
  (current run delivered 20 per checkpoint).
- The semantic read-only invariant test from Phase 8
  (`tests/test_orchestrator_one_way_invariant.py`) continues to
  pass unchanged. Phase 12 added two optional kwargs to
  `run_adversarial_pass` (`injected_sham_entries`, `record_sink`)
  but did not introduce any new write surface on Io's side.

### Out of scope (preserved from the plan)

Phases 9, 10, 11 (judges, aggregation protocols, cross-checkpoint
sweeps); any change to Io, the actor, the world model, the dream
state, or the runner; the judge layer (Phase 12 produces primary +
adversarial readings; no third-party judge); cross-checkpoint
analysis beyond the two-checkpoint smoke; real-world (Probe 4)
builder-perturbation infrastructure (the perturbations in Phase
12 are synthetic, drawn from existing checkpoint telemetry +
sham schedule injection); extension of `PreRegistration` itself
(Phase 12's commitments live in `RoundConfig` written alongside);
the v0.4.0 schema export bump (avoided by the
`PreRegistration`-not-extended decision).

Phase 12 touched `kind/mirror/calibration/__init__.py` (new),
`kind/mirror/calibration/round.py` (new),
`kind/mirror/calibration/sham_schedule.py` (new),
`kind/mirror/calibration/round_diff.py` (new),
`kind/mirror/calibration/smoke.py` (new),
`kind/mirror/calibration/llm_audit.py` (new),
`kind/mirror/__init__.py` (re-exports extended),
`kind/mirror/llm_caller.py` (CallOutcome + LLMRecordSink +
record_sink param + schema munger pipeline + APIError-aware
retry + backoff),
`kind/mirror/orchestrator.py` (sham injection hook + record_sink
threading),
`scripts/run_phase_12_smoke.py` (new),
`tests/conftest.py` (new — `--run-real-api` flag),
`tests/test_round.py` (new),
`tests/test_sham_schedule.py` (new),
`tests/test_round_diff.py` (new),
`tests/test_llm_audit.py` (new),
`tests/test_llm_caller.py` (extended with 5 munger tests),
`tests/test_phase_12_smoke_real_api.py` (new), and this journal
entry. No other files. No changes to `kind/observer/schemas.py`,
no changes to `kind/observer/pre_reg.py`, no schema
version bump.

The on-disk smoke artifact is at
`runs/phase_12_smoke/mirror/phase_12_smoke_result.json`
(1.35 MB, well-formed `Phase12SmokeResult`); the per-round
`RoundResult`s are at `runs/phase_12_smoke/mirror/rounds/
phase_12_probe_1_round.json` (611 KB) and
`phase_12_probe_1_5_round.json` (610 KB); the per-checkpoint
`PassResult`s are at the source runs' `mirror/passes/
ckpt-000001.json` (94 KB and 93 KB respectively, last-pass
overwrites). The pre-registration artifacts (round_config.json
+ pre_reg.jsonl per round) are at `runs/phase_12_smoke/mirror/
pre_reg/round_phase_12_probe_1_round/` and the matching
Probe 1.5 directory.

---

## Phase 13 — synthetic-real-perturbation injection and held-out isolation (2026-05-13)

Phase 13 closes the two load-bearing findings from Phase 12 that
gated Phase 9: the empty-baseline limitation on the sham-zero result
(no real perturbations to discriminate from) and the perfectly stable
held-out readings (suspicious enough to investigate). Both are now
structurally testable.

The synthetic-perturbation layer parallels the sham layer from Phase
12: a pre-registered schedule, deterministic placement, immutable
injection into the aligned perturbation timeline; the orchestrator-
side calibration check inspects per-(synthetic_event, criterion,
role) admissions; the round-level summary aggregates by criterion /
checkpoint / role. The held-out isolation study reuses Phase 12's
pass-0 bytes and re-invokes the held-out fragment without active-set
context, ten times per checkpoint, then compares the distribution to
the contextualised reference.

### What was built

**`kind/mirror/calibration/synthetic_perturbation.py` (new, 387
lines).** `SyntheticPerturbationEntry` (frozen, extra=forbid; the
`@model_validator` forces `payload["is_synthetic"]=True` *and*
`payload["is_sham"]=False`; the three categories are mutually
exclusive at the payload level by construction).
`SyntheticPerturbationSchedule` (frozen; no-collision invariant on
`(checkpoint_id, pass_index, synthetic_t)` triples).
`generate_synthetic_perturbation_schedule()` (deterministic via
`random.Random(seed)`; placement window
`range(1, telemetry_length - recovery_window - 1)` guarantees enough
post-perturbation samples; optional `sham_schedule=` parameter for
constructive cross-schedule disjointness).
`inject_synthetic_events()` (immutable merge; routes through the
sham injection's pattern; `is_sham=False` on the produced
`PerturbationEvent` so the recovery statistics process the synthetic
as a real perturbation).

**`kind/mirror/calibration/synthetic_calibration_check.py` (new, 411
lines).** `SyntheticCalibrationFinding` (per-(synthetic_event,
criterion, role) record carrying `admitted`,
`overlapping_claim_indices`, `recovery_lag_at_synthetic`, `note`).
`SyntheticFindingsSummary` (round-level aggregate; `admissions_by_*`
breakdowns; mean recovery lag at admissions / non-admissions).
`check_synthetic_calibration()` walks each synthetic event in the
timeline (identified by `payload["is_synthetic"]==True`), inspects
both primary and adversarial readings per active criterion, finds
claims whose `cited_step_range` covers the synthetic `t`, and
produces one finding per triple. The recovery-lag cross-reference
reaches into the pass's `recovery_lag_steps` `StatisticResult`
indexed by the synthetic event's position in the non-sham timeline
ordering. `aggregate_synthetic_findings()` rolls per-pass findings
into the summary.

**`kind/mirror/calibration/held_out_isolation.py` (new, 393 lines).**
`HeldOutIsolationConfig` (frozen; commits `n_isolation_runs=10`,
`pass_index=0` for Phase 13). `HeldOutIsolationReading` (per-run
record carrying the LLM's structured output, latency, and audit
records). `HeldOutIsolationStudy` (study-level aggregate carrying
both checkpoints' readings, both in-context references from Phase
12, claim-count and surface-distribution histograms across the
isolation runs, free-text `findings` analysing whether the
isolation runs diversified). `run_held_out_isolation_study()` slices
each Phase 12 `RoundResult`'s pass-0 held-out signal results,
rebuilds the held-out `PromptFragment` from those exact bytes, then
runs `n_isolation_runs` independent single-fragment calls per
checkpoint. The `findings` string ends with a binary
Reading-(a) / Reading-(b) flag drawn from the diversification count.
`load_reference_rounds_from_disk()` reads the two Phase 12
`RoundResult` JSONs back into memory for the CLI path.

**`kind/mirror/calibration/phase_13.py` (new, 277 lines).**
`Phase13CalibrationResult` (frozen; carries both `RoundResult`s, the
`RoundDiff`, the cross-surface `LLMCallAudit`, the
`HeldOutIsolationStudy`, wallclock_ms, notes).
`run_phase_13_calibration()` end-to-end driver: counts telemetry
length per checkpoint, builds the two `RoundConfig`s with shared
`StatisticConfig` / `LLMConfig` and distinct sham + synthetic seeds,
runs the two rounds sequentially (the round driver's synthetic
check fires after each pass per Phase 13 — see `run_round` below),
computes the `RoundDiff` with a `synthetic_schedule` rationale, runs
the held-out isolation study, and folds the isolation study's
records into the cross-surface `LLMCallAudit` (the audit covers the
full Phase 13 LLM-call surface, not just the round driver's calls).
Writes `Phase13CalibrationResult` to
`output_dir/mirror/phase_13_calibration_result.json` atomically.

**Phase 13 module-level commitments.**
`PHASE_13_PASSES_PER_CHECKPOINT=5`,
`PHASE_13_REAL_PERTURBATIONS_PER_PASS=2`,
`PHASE_13_SHAMS_PER_PASS=1`, `PHASE_13_SYNTHETICS_PER_PASS=2`,
sham seeds 42/43 (reuses Phase 12's so the sham schedules are
identical), synthetic seeds 142/143, isolation `n_runs=10`,
`pass_index=0`, `seed=13`. All on the `phase_13` module surface.

**Updates to existing modules:**
- `kind/mirror/calibration/round.py` — `RoundConfig` extended with
  `synthetic_schedule: SyntheticPerturbationSchedule = EMPTY_SYNTHETIC_SCHEDULE`
  (default-empty preserves Phase 12 caller compatibility). New
  model validator `_enforce_synthetic_schedule_consistency` checks
  every synthetic entry's `(checkpoint, pass)` is in the round's
  scope AND enforces cross-schedule disjointness against the sham
  schedule. `RoundResult` extended with `synthetic_findings_summary`.
  `run_round` filters per-pass synthetic entries, threads them
  through `run_adversarial_pass`, runs `check_synthetic_calibration`
  after each pass, and aggregates into the summary.
- `kind/mirror/calibration/round_diff.py` — new
  `SyntheticScheduleChange` record (rationale-required, frozen),
  new `synthetic_schedule_changes: SyntheticScheduleChange | None`
  field on `RoundDiff`. `compute_round_diff` now requires a
  `"synthetic_schedule"` rationale when the two rounds' synthetic
  schedules differ by value (Phase 13's cross-round Probe 1 / Probe
  1.5 split has distinct seeds by design — the rationale is journaled
  at the smoke-harness call site).
- `kind/mirror/orchestrator.py` — `run_adversarial_pass` extended
  with `injected_synthetic_entries: tuple[Any, ...] = ()`. The
  injection composes with the existing sham injection (shams first,
  then synthetics against the sham-augmented timeline); cross-
  schedule collisions surface from the timeline's sorted-unique
  validator.
- `kind/mirror/prompt_builder.py` — `_format_perturbation_block`
  now strips `is_sham` and `is_synthetic` keys from each event's
  displayed payload dict via a new `_payload_for_prompt` helper.
  The full payload (including the calibration flags) is preserved
  on the `PerturbationEvent` and in the on-disk
  `RoundResult.perturbation_timeline`; only the LLM-facing surface
  is filtered. This is the load-bearing edit that honours "the
  mirror does not see the synthetic/sham/real label in its prompt"
  — without it, the synthetic `payload["is_synthetic"]=True` would
  print into the equanimity fragment and the calibration's
  discriminative property would not hold.
- `kind/mirror/calibration/__init__.py` and `kind/mirror/__init__.py`
  — Phase 13's public surface re-exported.

**Tests (new, 67 tests across 4 files):**
- `tests/test_synthetic_perturbation.py` (31 tests): entry
  validation (forced flags, mutual-exclusion), schedule no-collision
  invariant, generator determinism + different-seed inequality,
  placement-window bounds enforcing the recovery-window margin,
  cross-schedule disjointness against a sham schedule, injection
  immutability + sorted-order preservation + collision rejection
  (against real OR sham), round-trip serialization, the asymmetry
  test confirming the entry models don't enforce cross-disjointness
  on their own (that's the round driver's job).
- `tests/test_synthetic_calibration_check.py` (12 tests): empty
  timeline → empty findings, real-perturbation-only timeline → no
  findings (real perturbations are not synthetic), admitting
  vs non-admitting readings, recovery-lag cross-reference,
  mixed sham + synthetic timeline (only synthetics produce
  findings), aggregation across passes with admissions and lag
  means, payload-with-both-flags edge case (treated as non-
  synthetic by the check).
- `tests/test_held_out_isolation.py` (11 tests): config validation,
  distribution histogram computation, findings-string non-empty,
  the two reading flags exercised (Reading (a) when isolation is
  perfectly stable; Reading (b) when it diversifies), checkpoint-id
  mismatch raises, full serialization round-trip.
- Extensions to `tests/test_round.py` (+8 tests) and
  `tests/test_round_diff.py` (+5 tests) covering the new
  `synthetic_schedule` field on `RoundConfig`, the cross-schedule
  collision validator, the synthetic-findings-summary on
  `RoundResult`, the `SyntheticScheduleChange` diff dimension, and
  the rationale-required invariant carrying.
- `tests/test_phase_13_calibration_real_api.py` (1 test, opt-in via
  `--run-real-api` or `GEMINI_API_KEY`).

### Quality gates

- `pytest tests/` without `--run-real-api`: **853 passing, 4
  skipped, 1 pre-existing flaky failure** (`test_transport.py::
  test_barrier_queues_mutates_and_drains_in_order` — same
  concurrency-timing-sensitive test that failed at Phase 8 and
  Phase 12). No Phase 13 regressions; the 4 skipped tests are the
  two real-API tests (Phase 12 + Phase 13) plus two pre-existing
  skips. Suite runtime: ~42 s.
- `mypy --strict kind/mirror/ kind/observer/ tests/test_synthetic_perturbation.py
  tests/test_synthetic_calibration_check.py tests/test_held_out_isolation.py
  tests/test_phase_13_calibration_real_api.py
  scripts/run_phase_13_calibration.py`: clean (33 source files
  checked).
- `pytest tests/ --run-real-api` (with `GEMINI_API_KEY`): the
  Phase 13 calibration runs end-to-end against the real Gemini
  API in ~30 minutes; the test asserts the result is well-formed,
  the audit shows ≥ 60 call records (40 from rounds + 20 from
  isolation), the synthetic-findings totals match the structural
  expectation (5 passes × 2 synthetics × 2 criteria × 2 roles ×
  2 rounds = 80 findings), and the isolation study produces 10
  readings per checkpoint. Phase 13's structural read-only tests
  continue to pass unchanged (the orchestrator's one-way invariant
  test, the prompt builder's verbatim-clause tests, etc.).
- `Phase13CalibrationResult.model_validate_json(model_dump_json(...))`
  round-trips cleanly (verified end-to-end via a mock-LLM
  invocation before the real-API run).

### The calibration run — what actually happened

The calibration completed in **1,825,130 ms (~30.4 minutes)**.
60 LLM calls (40 from the round driver — 4 calls per pass × 5
passes × 2 rounds — plus 20 from the isolation study — 10 calls
per checkpoint × 2 checkpoints). **Zero retries, zero failures.**
Mean per-call latency: 30,151 ms (~30 s, same as Phase 12's
ballpark). Total LLM wallclock: 1,809,058 ms — 99.1% of the
calibration's wallclock is spent waiting on the API; the
orchestrator's own compute is negligible. Tokens in / tokens out:
`None / None` — Phase 12's newly-open §2 (the `_extract_token_counts`
gap in `_GeminiLLMClient`) was *not* closed in Phase 13. Cost
trajectory is therefore still indirect (latency × wallclock as a
proxy). See "newly open §3" below.

### The synthetic-perturbation calibration findings

Phase 13's discriminative case fired: the recovery statistics
computed real values at every synthetic timestamp (no zero-sample
sentinels in the lag list); the equanimity fragment's
"Real perturbations (recovery readings apply)" block carried the
synthetic events alongside any sham events, with the calibration
flags stripped from the displayed payload (the LLM cannot tell
synthetic from real). The mirror engaged with the synthetic
timestamps.

**Per-pass synthetic admissions** (the `check_synthetic_calibration`
record's `admitted` flag fires when a primary/adversarial reading
of one of the active criteria has a claim whose `cited_step_range`
covers the synthetic `t`):

Probe 1 (40 (synthetic_event, criterion, role) findings total —
2 synthetics × 5 passes × 2 active criteria × 2 roles):
- **2 admissions, both in pass 3 primary equanimity readings.**
  synthetic_t=2147 — 1 primary equanimity_perturbation_recovery
  claim cites step range (2147, 2197).
  synthetic_t=2933 — 1 primary equanimity_perturbation_recovery
  claim cites step range (2933, 2983).
- recovery_lag_at_synthetic at both admissions: **51.0** (the
  Phase 8 non-recovery sentinel `recovery_window_W + 1 = 51`).
- All other 38 findings: non-admissions.

Probe 1.5 (40 findings total):
- **2 admissions, both in pass 1 adversarial equanimity readings.**
  synthetic_t=131 — 3 adversarial equanimity_perturbation_recovery
  claims cite step range (131, 1853).
  synthetic_t=1853 — same 3 adversarial claims cite same range.
- recovery_lag_at_synthetic at both admissions: **51.0** (non-
  recovery sentinel).
- All other 38 findings: non-admissions.

**Inspecting the admitting claims reveals the discriminative case
working exactly as Phase 7's prose predicted.** All 4 "admissions"
— the 2 primary-equanimity citations in Probe 1 pass 3 and the
3-claim adversarial citation in Probe 1.5 pass 1 — are **non-
falsifying non-admissions or explicit refutations**:

Probe 1 pass 3 primary equanimity, claim cited at synthetic_t=2147:
> "On the substrate side, the latent state h_t failed to recover
> within the 50-step window following the perturbation at t=2147,
> as indicated by the non-recovery sentinel value of 51."

The same reading's overall conclusion (claim[2]):
> "On the behavior side, policy entropy showed no detectable
> response to either of the two real perturbations, **which
> constitutes a non-falsifying non-admission of equanimity.**"

Probe 1.5 pass 1 adversarial equanimity, claim cited at
synthetic_t=131:
> "The latent state failed to recover within the measurement
> window for both non-sham perturbations, hitting the non-recovery
> sentinel value of 51 steps."

And a parallel claim in the same reading:
> "For one of two perturbations, the posterior KL aggregate
> ratcheted upwards, indicating a sustained, non-recovering
> surprise response **instead of equanimous recovery**."

The mirror is doing what the criterion's prose asks: it cites the
perturbation timestamps (synthetic) because the recovery statistics
have real values at those timestamps, but the *content* of the
claim explicitly refuses to read equanimity. The primary reader
cites the recovery_lag=51 as a non-recovery and routes it through
the non-falsifying-non-admission clause; the adversarial reader
cites the ratcheting KL as evidence *against* equanimity. The
discriminative case fired *and* the calibration discipline held.

**The "admission" count therefore needs reading carefully.** The
`SyntheticCalibrationFinding.admitted` flag is conservative — any
citation overlap counts, same convention as the Phase 8 sham
check. It does not distinguish supportive admissions ("equanimity
is satisfied here") from refutational admissions ("equanimity is
not satisfied here, here's the data"). The 4 admissions all fall
in the second bucket. The journal-side cost of the conservative
threshold is exactly this paragraph; the structural value is that
the threshold can't miss a real supportive admission if one ever
appeared.

**Discriminative coverage.** 20 synthetic events across the two
rounds (2 per pass × 5 passes × 2 rounds) × 2 active criteria ×
2 roles = 80 findings. 4 admissions out of 80 = 5% citation rate.
All admissions fall on the equanimity criterion; zero
reflexive-attention admissions at synthetic timestamps (which is
correct — reflexive_attention reads h_t coupling structure, not
perturbation responses; a citation overlap there would have been
spurious).

**Recovery-lag cross-reference.** Every synthetic event in this
calibration had recovery_lag_steps = 51.0, the non-recovery
sentinel. This is unsurprising: the Phase 1 / Phase 1.5 telemetry
was generated without any real perturbations, so the h_t
trajectory at any randomly-selected step has no perturbation-
response shape — the Mahalanobis recovery walk hits the post-W
window without ever satisfying the streak criterion. The
synthetic injection gives the recovery statistic real *values*
(it has data to walk over), but the *shape* of that data is
quiescent. Phase 13 confirms the criterion handles this case
correctly — a quiescent shape at a stated perturbation time is
read as a non-falsifying non-admission, exactly as the non-
falsifying clause demands.

### The held-out isolation study findings

20 isolation runs (10 per checkpoint), all primary role, all
against the `second_order_volition` criterion's pre-Phase-12 pass-0
held-out fragment. Each run was a fresh single-fragment LLM call
with no active-set context.

**Claim-count distribution across isolation runs:**
- 19 runs produced 2 claims (95%).
- 1 run produced 3 claims (5%).

**Surface distribution across isolation runs:**
- 19 runs produced `sub=1, head=0, beh=1` (the same shape Phase
  12 saw in all 20 contextualised held-out readings).
- 1 run produced `sub=1, head=0, beh=2` (the same outlier;
  carries an extra behavior-side claim).

**Phase 12 in-context references** (the contextualised pass-0
held-out primary readings the isolation runs compare against):
- Probe 1: 2 claims, surfaces `sub=1, head=0, beh=1`.
- Probe 1.5: 2 claims, surfaces `sub=1, head=0, beh=1`.

**Findings (per the structured binary flag the study emits):
"Reading (b) supported".** Two distinct claim counts and two
distinct surface tuples appeared across the isolation runs, so
the strict-threshold binary flag fires for Reading (b) — the
held-out partition's structural adversarial-check job is not
exercising the LLM the same way the criterion's prose would
alone.

**The flag is technically right but undersells the result.** The
diversification is one outlier in twenty runs — 95% same shape as
the in-context reference. Compare to Phase 12's contextualised
held-out: 20/20 perfectly stable. The drop from 100% → 95%
stability under isolation is real but small. Phase 7's prose for
`second_order_volition` is largely sharp; the contextualised
invocation is more stable than the isolated version by ~5
percentage points. Phase 9 should consider giving the held-out
reading its own call (the strict reading-(b) interpretation), but
the absolute stability of the isolated version is high enough that
the contextualised version is not the right thing to abandon
wholesale. The binary Reading-(a)/Reading-(b) flag in the study's
code is too coarse for this gradient — see "newly open §4" below.

### Sham calibration cross-check — clean under synthetic-rich conditions

Phase 13 ran the same sham injection Phase 12 ran (5 shams per
round; one per pass) alongside the new synthetic injection.
**0 sham admissions in either round.** The structural machinery
the Phase 12 calibration depended on is unchanged by Phase 13's
additions — the sham timestamps still don't pull citations from
the primary equanimity readings, even when synthetic timestamps
*do*. This is the load-bearing structural confirmation Phase 13
contributes: the calibration discipline holds under both empty-
baseline (Phase 12) AND discriminative (Phase 13) conditions.

If a sham admission had appeared in Phase 13, the natural reading
would have been "the synthetic injection somehow contaminated the
sham check" — co-design failure. The clean sham column rules that
out.

### Stance drift across the five passes

Per-pass claim counts. Format:
`[reflexive_attention, equanimity_perturbation_recovery]` for the
active set; `[second_order_volition]` for the held-out.

**Probe 1:**

| Pass | Active primary | Active adversarial | Held-out primary | Held-out adversarial |
|------|----------------|--------------------|--------------------|----------------------|
| 0    | [2, 3]         | [1, 2]             | [2]                | [2]                  |
| 1    | [2, 3]         | [1, 2]             | [2]                | [2]                  |
| 2    | [1, 2]         | [1, 3]             | [2]                | [2]                  |
| 3    | [2, **5**]     | [2, 2]             | [2]                | [2]                  |
| 4    | [2, 3]         | [1, 3]             | [3]                | [2]                  |

**Probe 1.5:**

| Pass | Active primary | Active adversarial | Held-out primary | Held-out adversarial |
|------|----------------|--------------------|--------------------|----------------------|
| 0    | [2, 3]         | [1, 3]             | [2]                | [4]                  |
| 1    | [2, 3]         | [1, 3]             | [2]                | [2]                  |
| 2    | [2, 3]         | [2, 3]             | [2]                | [2]                  |
| 3    | [2, 3]         | [1, 2]             | [2]                | [2]                  |
| 4    | [2, 3]         | [2, 3]             | [2]                | [2]                  |

Observations:

- **Probe 1 pass 3 is the outlier on equanimity primary (5
  claims).** This is the pass where both synthetic admissions
  occurred. Hypothesis (untested by Phase 13 — would need a
  controlled experiment): the LLM expands its primary equanimity
  reading when synthetic events fall at structurally distinct
  trajectory positions in the same pass. Probe 1's primary
  equanimity claim count distribution is `[3, 3, 2, 5, 3]` — 80%
  at 3 ± 1, with one 5-claim outlier.
- **Probe 1.5 primary equanimity is perfectly stable** at 3 claims
  across all five passes. The synthetic admissions in this round
  fell on the adversarial reader (pass 1, 3 claims), not the
  primary.
- **Reflexive attention is largely stable** across both rounds —
  most passes have 2 primary + 1 or 2 adversarial. Probe 1 pass 2
  drops to [1, 1] on primary; everyone else is in the 1-3 range
  per role.
- **Held-out partition shows occasional drift under contextualised
  invocation:** Probe 1 pass 4 primary has 3 claims (others have
  2); Probe 1.5 pass 0 adversarial has 4 claims (others have 2).
  So the Phase 12 perfect-stability finding was specific to the
  one Phase 12 smoke; Phase 13's same-contextualised invocation
  also drifts occasionally. The 95% stability in isolation maps
  onto a similar ~10% drift rate under contextualised invocation
  in Phase 13. Both invocations are ~90-95% stable; the gap is
  marginal.

The Phase 12 stance-drift pattern (reflexive_attention adversarial
column compressing mid-round) does not appear in Phase 13's
adversarial column the same way. Whether this is sample-size noise
(Phase 12 also saw 5 passes per round) or a Phase 13-specific
side-effect of the synthetic injection (the adversarial reader has
more material to argue against) is the kind of question Phase 9's
judge layer is designed to answer over multi-pass aggregates.

### LLM call audit summary

- 60 total calls across the calibration's full surface (rounds +
  isolation).
- 0 retries. 0 failures. The Phase 12 schema-munger pipeline +
  retry-budget-with-backoff is holding cleanly for the second
  major calibration in a row.
- Total LLM wallclock: 1,809,058 ms (~30.2 min). Per-call mean:
  30,151 ms. Per-call min/max not surfaced in the current audit;
  see "newly open §5".
- Tokens in / out: `None / None`. The Phase 12 newly-open §2 — the
  `_extract_token_counts` gap in `_GeminiLLMClient.generate_batch`
  — remains open. Phase 13 did not fix it; the audit's token
  fields stay `None` and no cost analysis grounds on tokens. The
  proxy is wallclock × per-token-cost-estimate; the journal entry
  records both, leaving the precise number for the eventual fix.
- Compare to Phase 12: 40 calls × ~28 s mean → ~19 min wallclock
  there; Phase 13's 60 calls × ~30 s mean → ~30 min here. Each
  calibration phase scales linearly in calls × per-call latency;
  no super-linear growth observed.

### What surprised

The synthetic-perturbation result is *exactly* what Phase 7's prose
predicted, in a way I didn't expect to see this cleanly. The non-
falsifying clause was written as a defence against the specific
failure mode "Io that doesn't notice a perturbation gets read as
equanimous." Phase 13 set up 20 synthetic events at randomly-chosen
quiescent timestamps where the recovery shape is *exactly* the
"didn't notice" shape (recovery_lag = 51 sentinel; entropy →
no_response classification), and the primary equanimity reader
correctly cited the data, named the non-recovery, and refused to
admit equanimity. The criterion's prose was load-bearing against a
hypothetical failure mode, and Phase 13 gave it the test case the
hypothetical described.

The adversarial reader's behaviour at Probe 1.5 pass 1 is the same
story from the opposite side: the skeptic cites the synthetic
timestamps to *argue against* equanimity, not to admit it. The
adversarial role is doing structural work — even on synthetic
events that have no genuine perturbation context, the skeptic
finds language to refute the primary's potential admission. The
two roles together cover the discriminative case without either
producing a spurious admission.

The held-out isolation result was less surprising in shape (the
Phase 12 perfect stability was a small-sample artifact; the actual
stability is ~95%) but more useful than I expected — the gap
between contextualised and isolated invocation is small enough
that Phase 9 doesn't need to fundamentally restructure the held-out
call. A 5-percentage-point stability cost is recoverable in other
ways (Phase 9's judge layer can aggregate over multiple passes).

### What's now closed

Three of Phase 12's newly-open findings close (or close in shape)
in Phase 13.

1. **§1 — the discriminative case for the calibration.** Phase 12
   could not test whether the mirror admits at non-sham
   perturbation timestamps because there were zero non-sham
   perturbations in the source telemetry. Phase 13 added 20
   synthetic non-sham perturbations across the two rounds. 4 of
   80 findings registered as "admissions" (citation overlap); all
   4 were non-falsifying non-admissions or explicit refutations
   on inspection. The calibration discipline (non-falsifying
   clause + falsifier + role asymmetry) holds under discriminative
   conditions. The closing is structural: the synthetic-injection
   machinery is now production-quality and any future shift in
   the mirror's behaviour at synthetic timestamps will surface in
   the per-pass findings + aggregate summary.

2. **§5 — the `real_perturbations_per_pass` pre-registration
   enforcement.** Phase 12's `real_perturbations_per_pass=2`
   commitment was structural-only; the telemetry actually
   delivered 0. The synthetic schedule's `synthetics_per_pass=2`
   on Phase 13 delivers exactly 2 events per pass that the
   recovery statistics actually process. The structural floor
   on real-perturbation count is now reachable in practice (with
   synthetic stand-ins until Probe 4 lands actual builder
   perturbations); the journal-side check that "the round delivered
   the committed number of events" is now meaningful for the
   synthetic count. A future automated check that compares
   `synthetics_per_pass × passes_per_checkpoint × n_checkpoints`
   to the actual finding count is plausible Phase 14 work.

3. **§7 — the held-out perfect-stability investigation.** Phase
   12's 20/20 perfectly-stable held-out readings prompted the
   question: is Phase 7's prose sharp, or is the held-out criterion
   not exercising the LLM? Phase 13 ran the same fragment in
   isolation 20 times: 19/20 same shape; 1 outlier. Reading (b)
   per the strict binary flag (the isolation runs diversified);
   in interpretation, the gap is small (~5%) — Phase 7's prose
   is mostly sharp, and the held-out partition under contextualised
   invocation is at most marginally less stable than under
   isolation. Phase 9 has the calibration record it needs to
   decide whether to isolate the held-out call architecturally;
   the decision is no longer load-bearing on a missing data point.

### What's now newly open

1. **The synthetic check's binary admitted-flag undersells the
   distinction between supportive and refutational citations.**
   Phase 13's 4 "admissions" are all non-falsifying non-admissions
   or explicit refutations — the conservative threshold can't
   tell the difference. Phase 9's judge layer would benefit from
   a more nuanced per-claim verdict (supportive | refutational |
   non-falsifying-non-admission | …) that the orchestrator's check
   could surface. The information is present in the claim text;
   what's missing is a structured field that names the verdict.
   Either Phase 9 adds the layer, or a Phase 7 amendment to
   `StructuredClaim` adds an explicit `claim_polarity` field
   along the supportive / refutational / non-admission axis. The
   second route is cleaner but invites a Phase 7 schema-change
   conversation; the first route is local to Phase 9.

2. **The token-count gap from Phase 12 newly-open §2 remains
   open.** Phase 13 did not fix `_extract_token_counts` in
   `_GeminiLLMClient`. The audit's `total_tokens_in /
   total_tokens_out` are still `None` after both calibrations.
   Surface area is ~10 lines (thread the SDK's `usage_metadata`
   through the `generate_batch` return); the lack of fix is
   purely a Phase 13 scope discipline (the plan didn't list the
   prompt_builder calibration-flag edit OR the token thread-
   through; the calibration-flag edit was required for the
   discriminative property, the token edit was not). Phase 14
   should land this — without token counts the cost trajectory
   is shaped by wallclock × estimated per-token cost rather than
   ground truth.

3. **The cost trajectory now that Phase 13 doubles the LLM-call
   surface.** Phase 12: 40 calls × ~28 s mean × Gemini 2.5 Pro
   pricing ≈ a few-dollars-per-calibration. Phase 13: 60 calls
   × ~30 s mean ≈ ~50% more. If Phase 14 adds a second
   sweep (e.g., multiple seeds per round), the cost shape goes
   from "afford a few of these" to "audit every call's necessity."
   The audit's per-call records already carry the prompt and the
   wallclock; what's needed is a journal-side dashboard / script
   that reads the audit and projects cumulative cost by phase.
   The token-count fix (newly-open §2) is a hard prerequisite for
   the dollar number.

4. **The held-out isolation study's binary Reading-(a) /
   Reading-(b) flag is too coarse.** Phase 13's actual result —
   19/20 same shape, 1 outlier — gets reduced to "Reading (b)"
   under the strict threshold; the journal entry has to undo the
   reduction to be honest about the gradient. Future versions
   should report a more nuanced shape — e.g., the exact
   diversification fraction, or a distance metric over the
   per-run claim sets — rather than a binary flag. A small edit
   to `_compute_findings` in `held_out_isolation.py` would do it;
   the structural choice is to *prefer* a non-binary report and
   journal the interpretation alongside.

5. **Per-call latency distribution is not in the audit.** The
   audit aggregates `total_wallclock_ms` but not min / max /
   median / per-role / per-pass per-call latency. With 60 calls
   per Phase 13 calibration, the distribution starts to matter
   for projecting future runs and detecting outlier-latency
   tails (a single 5-min retry will silently inflate the
   wallclock total). A `LatencyDistribution` field on
   `LLMCallAudit` aggregating per-role / per-checkpoint
   percentile values would close this; the per-call records
   already carry the data.

6. **Probe 1 pass 3's primary equanimity outlier (5 claims) is
   not explained.** The pass that admits at both synthetic
   timestamps is also the pass with the highest primary
   equanimity claim count. Hypothesis: the LLM expands its
   primary equanimity claim count when the perturbation
   timeline carries more detail near other perturbation events
   in the same trajectory window. The hypothesis is testable
   with a controlled experiment (vary the synthetic placement
   density per pass, measure claim-count response), but it's
   a single-observation pattern at Phase 13; flag for future
   investigation, do not act on it now.

### Out of scope (preserved from the plan)

Phases 9, 10, 11 (judges, aggregation protocols, larger sweeps);
any change to Io, the actor, the world model, the dream state, or
the runner; the judge layer (Phase 13 produces primary +
adversarial readings + the synthetic / held-out calibration
records; no third-party judge); Probe 4 real-builder-perturbation
infrastructure (Phase 13's synthetics are calibration stand-ins,
not substitutes); changes to Phase 7's frozen criteria prose (a
future revision requires an external-framework reading per the
charter's discipline — Phase 13 produces findings about the
calibration, not the criteria); extension of `PreRegistration`
(Phase 13's commitments live on `RoundConfig.synthetic_schedule`
written alongside the existing `sham_schedule`); the v0.4.0 schema
export bump (avoided by the `PreRegistration`-not-extended
decision, same as Phase 12).

Phase 13 touched
`kind/mirror/calibration/synthetic_perturbation.py` (new),
`kind/mirror/calibration/synthetic_calibration_check.py` (new),
`kind/mirror/calibration/held_out_isolation.py` (new),
`kind/mirror/calibration/phase_13.py` (new),
`kind/mirror/calibration/round.py` (extended —
`synthetic_schedule` field, validators, `synthetic_findings_summary`
on `RoundResult`, `run_round` injection + check + aggregation),
`kind/mirror/calibration/round_diff.py` (extended —
`SyntheticScheduleChange` record, `synthetic_schedule_changes`
field, computation),
`kind/mirror/calibration/__init__.py` (re-exports extended),
`kind/mirror/__init__.py` (re-exports extended),
`kind/mirror/orchestrator.py` (`injected_synthetic_entries`
threading + composed injection),
`kind/mirror/prompt_builder.py` (`_payload_for_prompt`
calibration-flag stripping),
`scripts/run_phase_13_calibration.py` (new),
`tests/test_synthetic_perturbation.py` (new),
`tests/test_synthetic_calibration_check.py` (new),
`tests/test_held_out_isolation.py` (new),
`tests/test_phase_13_calibration_real_api.py` (new),
`tests/test_round.py` (extended — 8 Phase 13 tests),
`tests/test_round_diff.py` (extended — 5 Phase 13 tests),
and this journal entry. No other files. No changes to
`kind/observer/schemas.py`, no changes to
`kind/observer/pre_reg.py`, no schema version bump.

The on-disk calibration artifact is at
`runs/phase_13_calibration/mirror/phase_13_calibration_result.json`
(1.49 MB, well-formed `Phase13CalibrationResult`); the per-round
`RoundResult`s are at
`runs/phase_13_calibration/mirror/rounds/phase_13_probe_1_round.json`
(624 KB) and `phase_13_probe_1_5_round.json` (631 KB); the
per-checkpoint `PassResult`s are at the source runs'
`mirror/passes/ckpt-000001.json` (98 KB and 96 KB respectively;
last-pass overwrites). The pre-registration artifacts
(`round_config.json` + `pre_reg.jsonl` per round) are at
`runs/phase_13_calibration/mirror/pre_reg/round_phase_13_probe_1_round/`
and the matching Probe 1.5 directory. The Phase 12 artifacts at
`runs/phase_12_smoke/` are unchanged — Phase 13 reads from them
(the isolation study reuses the Phase 12 pass-0 statistic_results
bytes) but never writes to them.

---

## Phase 9 — judge layer (2026-05-15)

Phase 9 adds the third interpretive role in the mirror's adversarial
structure: the Methodological Arbiter. Phases 6–8 committed two
roles (primary as Phenomenological Advocate, adversarial as
Statistical Skeptic). Phase 12 verified the calibration discipline
in the empty-baseline case; Phase 13 verified it in the
discriminative case and produced the multi-pass record the judge
consumes. Phase 9 reads that record and produces findings *about
the readings* — not about Io.

The build is structural-only at this commit. The Phase 9 smoke runs
against the real Gemini API by invoking
`scripts/run_phase_9_judge_smoke.py` against the on-disk Phase 13
calibration result; this journal entry records the substrate that
makes that smoke runnable. The substantive findings — claim
polarity stability, verdict consistency across rounds, role
disagreement frequency, judge confidence calibration — land in a
follow-up entry after the smoke runs.

### What was built

**`kind/mirror/judge.py` (new, 467 lines).** The judge data plane.
Five frozen Pydantic records:

- `ClaimPolarity` — str-valued enum with four exhaustive members:
  `SUPPORTIVE`, `REFUTATIONAL`, `NON_FALSIFYING`, `AMBIGUOUS`. The
  enum lives at this Phase 9 layer; Phase 7's frozen
  `StructuredClaim` is unchanged. Whether to amend `StructuredClaim`
  with a structured polarity field is a future Phase 14+ decision
  contingent on Phase 9's stability findings.
- `ClaimPolarityAssignment` — per-claim record carrying
  `(pass_index, criterion_id, reader_role, claim_index,
  cited_step_range, polarity, polarity_rationale)`. The
  `polarity_rationale` is non-empty by validator — the judge's
  load-bearing audit trail.
- `FalsifierVerdict` — per-falsifier roll-up across passes. Four
  partition tuples (`passes_supporting`, `passes_refuting`,
  `passes_non_falsifying`, `passes_ambiguous`) exhaustively cover
  the pass-index space; a model validator enforces pairwise
  disjointness across all six pairs. A pass that fell into more than
  one bucket would mean the judge double-counted; a pass that fell
  into none would mean the judge silently dropped it.
- `CriterionJudgment` — per-criterion verdict carrying
  `falsifier_verdicts: tuple[FalsifierVerdict, ...]` (tuple shape
  supports future multi-falsifier criteria; v2 criteria each commit
  one), `verdict: Literal["satisfied", "not_satisfied",
  "non_falsifying", "mixed", "ambiguous"]`, `confidence: float` in
  `[0.0, 1.0]`, non-empty `rationale`, and
  `claim_polarity_assignments`. Validators enforce that every
  `FalsifierVerdict` and `ClaimPolarityAssignment` carries the
  matching `criterion_id` — the judge cannot attribute across
  criteria.
- `RoundJudgment` — round-level aggregate carrying
  `criterion_judgments`, `judge_llm_call_records`, `wallclock_ms`,
  `notes`, and a non-empty `round_config_summary` human-readable
  string. A model validator enforces unique `criterion_id` across
  `criterion_judgments`.

**`kind/mirror/judge_prompt_builder.py` (new, 410 lines).** The
judge fragment composer. `JudgePromptFragment` (frozen, carries
`criterion_id`, `body`, `primary_readings_included`,
`adversarial_readings_included`, `statistic_results_summary`).
`build_judge_fragment()` dispatches on `criterion.id` to one of
three per-criterion fragment builders (reflexive_attention,
equanimity, second_order_volition).

The load-bearing verbatim discipline is enforced at the
module-import seam: the equanimity non-falsifying-non-admission
clause and the four second-order-volition exclusions are *imported*
from `kind.mirror.prompt_builder` (Phase 8), not redefined locally.
Module-level `assert` statements check that the imported objects
are `is`-identical to the Phase 8 module's constants; if a future
contributor shadows one of them locally the import-time assertion
trips. The dedicated test
`test_equanimity_judge_fragment_clause_is_byte_identical_constant`
re-confirms the identity at test time.

The fragment is verbose by design: a 5-pass round × 2 roles × 3
criteria × ~3 claims/reading produces a substantial prompt.
Per-reading `free_text_notes` longer than `MAX_NOTES_CHARS=1200`
get truncated with a marker naming the original length; the
*claims* are never truncated. The journal records the truncation
discipline; the truncation marker quantifies any elided surface.

The degenerate-case contract — a criterion with zero readings
across passes produces a fragment that documents the absence
rather than crashing — is committed at this layer so the driver
can call uniformly across criteria with and without readings.

**`kind/mirror/judge_llm_caller.py` (new, 481 lines).** The judge
LLM-call layer mirroring `kind.mirror.llm_caller`'s shape.
`JudgePayload` is the structured output schema (with
`ClaimPolarityAssignmentPayload` and `FalsifierVerdictPayload`
component models); `JudgeBatchPayload` is the wrapper.
`JUDGE_SYSTEM_PROMPT` is the Methodological Arbiter stance — a
distinct document from the primary and adversarial system prompts.
`call_judge_llm()` makes one batched call per round (all criteria
in one structured response), with bounded retries on malformed
output, threading `LLMCallRecord`s through the Phase 12 audit
collector under `role="adversarial"` (the judge is structurally
adversarial to both other roles; a future Phase 14+ that adds a
`"judge"` literal to `PassRole` is a one-substitution migration —
the alias `JUDGE_ROLE` names the choice).

The Phase 12 `_to_gemini_schema` munger is reused on
`JudgeBatchPayload`'s schema (the embedded `tuple[int, int]`
`cited_step_range` on each polarity assignment emits `prefixItems`
that Gemini's OpenAPI 3.0 validator rejects). A dedicated test
verifies the round-trip is clean.

The judge's system prompt explicitly says: *"The criterion's prose
and falsifier are frozen. You are judging the readings, not
amending the criterion. If you find the falsifier insufficient to
evaluate the evidence at hand, say so in the rationale and mark
the verdict 'ambiguous'; do not invent a new falsifier."* The
load-bearing don't-soften-the-frozen-criterion discipline is in
the prompt and pinned in the test suite.

**`kind/mirror/judge_driver.py` (new, 327 lines).** End-to-end
driver. `judge_round()` loads a `RoundResult` from disk, walks the
active + held-out registries, builds one `JudgePromptFragment` per
criterion (passing per-pass primary readings, per-pass adversarial
readings, and per-pass statistic results), calls the judge LLM
once batched across all criteria, and writes the `RoundJudgment`
to `output_dir/mirror/judgments/{round_id}.json` atomically.

The driver is read-only against the source round result; a
dedicated test asserts byte-equality of the source file before and
after the judge runs.

The per-criterion reading lookup uses position-based matching
against the per-partition readings tuple — Phase 8's orchestrator
writes one reading per criterion in registry order, so
`criterion_index` in the readings tuple corresponds to the same
position in the partition registry. The
`_reading_matches_criterion()` fallback (claim-citation-based
match) is in place for future cases where the orchestrator's
position-based ordering breaks.

**`kind/mirror/calibration/phase_9_judge_smoke.py` (new, 207
lines).** `Phase9JudgeSmokeResult` (frozen; carries both rounds'
judgments, the cross-round `LLMCallAudit`, wallclock_ms, notes).
`run_phase_9_judge_smoke()` resolves the two Phase 13 round JSONs
from `{phase_13_calibration_path}/../rounds/`, invokes
`judge_round` on each, aggregates the cross-round audit, and
writes `Phase9JudgeSmokeResult` atomically to
`output_dir/mirror/phase_9_judge_smoke_result.json`.

The Phase 9 commitments at module level: the two round filenames
are derived from the Phase 13 module's `PHASE_13_PROBE_1_ROUND_ID`
and `PHASE_13_PROBE_1_5_ROUND_ID` constants — Phase 9 inherits
Phase 13's identifiers rather than re-declaring them, so a future
revision to either round id surfaces here automatically.

**`scripts/run_phase_9_judge_smoke.py` (new, 109 lines).** CLI
mirroring Phase 12 and 13's script shape. `--phase-13-calibration-path`,
`--output-dir`, `--llm-api-key-env-var`, `--notes` arguments;
loads `.env` via `python-dotenv` so `GEMINI_API_KEY` is picked up
without a shell pre-export.

**`kind/mirror/registry.py` (extended).** `Criterion` gains a
required `falsifier_id: str` field, snake_case + ≤40 chars
(same shape as `Criterion.id`). The field is a string, not a
writer-shape — the structural-no-writer test from Phase 6
continues to pass. The `_v1` suffix convention is for forward
versioning if a criterion's falsifier prose is amended in a
future Phase 7 revision; the regex pins shape only.

**`kind/mirror/criteria_v2.py` (extended).** The three v2 criteria
each commit a stable `falsifier_id`:
`reflexive_attention.falsifier_id = "reflexive_attention_v1"`,
`equanimity_perturbation_recovery.falsifier_id =
"equanimity_perturbation_recovery_v1"`,
`second_order_volition.falsifier_id =
"second_order_volition_v1"`. The judge's `FalsifierVerdict`
references these.

**`kind/mirror/__init__.py` (extended).** Re-exports the Phase 9
public surface: `ClaimPolarity`, `ClaimPolarityAssignment`,
`FalsifierVerdict`, `CriterionJudgment`, `RoundJudgment`,
`Verdict`, `JudgePromptFragment`, `build_judge_fragment`,
`JUDGE_SYSTEM_PROMPT`, `JudgePayload`, `JudgeBatchPayload`,
`ClaimPolarityAssignmentPayload`, `FalsifierVerdictPayload`,
`MockJudgeLLMClient`, `call_judge_llm`, `JUDGMENTS_SUBDIR`,
`judge_round`, `load_round_result_from_disk`,
`Phase9JudgeSmokeResult`, `run_phase_9_judge_smoke`.

**Tests.** Five new test modules, 78 new tests:

- `tests/test_judge.py` (29 tests) — data-plane validation. Frozen
  invariants on every record; the `FalsifierVerdict` four-way
  partition disjointness invariant covered with one test per pair
  (six pairs); confidence range; cross-criterion guards; serialization
  round-trip.
- `tests/test_judge_prompt_builder.py` (17 tests) — including the
  load-bearing verbatim-clause tests (one per clause, with the
  byte-identical-constant follow-up), pass-order presentation,
  per-pass statistic summaries, the degenerate-case (zero
  readings) contract, the notes-truncation discipline, and the
  per-criterion dispatch.
- `tests/test_judge_llm_caller.py` (19 tests) — payload validation,
  system prompt selection (`JUDGE_SYSTEM_PROMPT` distinct from
  primary/adversarial), per-criterion count and id matching,
  retry path (success and exhaustion), schema munger
  compatibility (Phase 12 `prefixItems` regression coverage at
  the judge layer), envelope stamping (`framework` from the
  criterion record, not the LLM output), record sink emission.
- `tests/test_judge_driver.py` (12 tests) — end-to-end with mock
  LLM and a synthetic Phase 13-shaped `RoundResult`. The
  read-only-against-source contract, the atomic write, the
  serialization round-trip, the criterion-judgments-cover-the-
  registry contract, and the call-records-flow-into-judgment
  contract.
- `tests/test_phase_9_judge_smoke_real_api.py` (1 test, opt-in
  `real_api` mark) — env-driven (`KIND_PHASE_13_CALIBRATION_PATH`),
  skips by default. Asserts the smoke completes, the result
  round-trips, every criterion has a non-empty rationale, and at
  least one success record per round in the audit.

**Test-fixture updates.** `tests/test_criterion_registry.py`,
`tests/test_signal_mapping.py`, and `tests/test_prompt_builder.py`
all pass `falsifier_id` to their `_criterion` helpers / inline
`Criterion()` constructions. A new test
`test_criterion_falsifier_id_regex_enforced` covers the validator;
two new tests in `tests/test_criteria_v2.py`
(`test_v2_criteria_have_committed_falsifier_ids` and
`test_v2_falsifier_ids_are_unique_across_registry`) pin the
committed identifiers.

### Quality gates

- `mypy --strict kind/mirror/ kind/observer/ tests/test_judge.py
  tests/test_judge_prompt_builder.py tests/test_judge_llm_caller.py
  tests/test_judge_driver.py tests/test_phase_9_judge_smoke_real_api.py
  scripts/run_phase_9_judge_smoke.py` — clean, 39 source files.
- `pytest tests/` (without `--run-real-api`) — 958 passed, 5
  skipped (the three real-API opt-in tests, two pre-existing
  skips). No regressions from Phase 8/12/13. The verbatim-clause
  tests from Phase 8 (`test_equanimity_fragment_contains_non_falsifying_clause`,
  `test_second_order_volition_fragment_contains_all_four_exclusions`)
  continue to pass — the judge prompt builder imports those
  constants rather than re-defining them, so the Phase 8 test
  and the new Phase 9 byte-identical-constant test are
  belt-and-suspenders for the same load-bearing discipline.
- The structural and semantic read-only tests from Phases 6–8
  continue to pass — the judge layer adds a new write surface
  (`runs/{run_id}/mirror/judgments/`) but the Phase 6 registry-
  no-writer assertions remain because the new `Criterion.falsifier_id`
  field is a string, not a writer-shaped type.

### Decisions made during build

1. **The judge batches across criteria in one call per round.**
   The plan's "one call per criterion" prose described a possible
   shape, but Phase 8's `BatchPayload` pattern (one batched call
   producing per-criterion readings) is the proven shape for the
   structured-output discipline. Phase 9 reuses it: one
   `JudgeBatchPayload` per round, with `per_criterion: list[
   JudgePayload]` matching the fragments in order. The plan's
   "6 calls" prose reconciles to "6 criterion judgments produced
   across 2 batched calls" — 2 rounds × 1 batched call each. The
   audit records reflect the batched count; the per-criterion
   judgment count is on the `RoundJudgment.criterion_judgments`
   tuple.

2. **`falsifier_id` is required, not optional with a default.**
   Phase 7's three frozen criteria each commit one explicitly;
   the plan's wording of "the values: …" is read as a commitment,
   not a default. A required field is the more disciplined choice
   because a missing identifier would be the kind of silent error
   the project's discipline guards against — better to fail at
   registry-load than to surface as `"unknown"` in a downstream
   `FalsifierVerdict`.

3. **The judge call records emit under `role="adversarial"`.**
   The Phase 12 `PassRole` literal is `Literal["primary",
   "adversarial"]`; adding `"judge"` to that literal is a
   widening that touches several signatures across the calibration
   plane. Phase 9 makes the more localised choice: emit judge
   records under `"adversarial"` (the judge is structurally
   adversarial to both readers) and name the choice at the
   module-level `JUDGE_ROLE` alias. A future Phase 14+ that adds
   the `"judge"` literal is a one-substitution migration. The
   `LLMCallRecordCollector` is identified per-collector at
   construction time, not by the `role` field on the records, so
   the judge's records still attribute correctly via the
   collector identity.

4. **The Phase 13 round result on disk is the input, not the
   `Phase13CalibrationResult` JSON.** The plan describes the
   smoke as consuming "the two `RoundResult`s from
   `runs/phase_13_calibration/mirror/rounds/`". The CLI takes the
   parent `Phase13CalibrationResult` path for ergonomics (the
   calibration result is the canonical Phase 13 artifact a human
   operator names) but the smoke harness resolves the two round
   JSONs from the sibling `rounds/` subdirectory. The
   `Phase13CalibrationResult` carries the rounds inline as well,
   but reading them from disk matches what
   `kind.mirror.judge_driver.judge_round()` expects without
   further dispatch.

5. **The judge fragment is built from per-pass primary +
   adversarial readings, never from the round's aggregated
   sham/synthetic findings.** The judge reads readings, not
   findings. The sham_findings_summary and
   synthetic_findings_summary are orchestration-side artifacts; if
   the judge needed them they would be in the fragment. Phase 9's
   plan is explicit: *"the judge reads readings, not telemetry"*
   — and the same applies to the round's calibration findings.
   The judge sees the same primary and adversarial readings that
   carry per-claim citations the criterion's falsifier already
   covers; the judge's job is to aggregate polarity across passes,
   not to re-derive the calibration summaries.

### What surprised

Mypy --strict tripped on the test file's direct attribute
access of the imported clause constants from the judge prompt
builder (`_jpb.EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE`):
the constants are imported by the module but not re-exported in
`__all__`, so mypy doesn't surface them as exported attributes.
The fix was to use `getattr(_jpb, "EQUANIMITY_…")` for the
byte-identical-constant test — the test asserts the runtime
identity, not the module's public surface, so the indirection is
honest. The Phase 8 module-level constants stay the canonical
location; the judge module imports them and asserts identity at
import time.

The decision to make `falsifier_id` required (not optional with
a default) surfaces immediately in test fixtures: three test
files needed the `falsifier_id="test_falsifier_v1"` addition.
The cost is one line per fixture; the gain is no silent
`"unknown_v1"` lurking in a future `FalsifierVerdict`.

### What's now closed

- **Phase 13's newly-open §1** — the conservative `admitted` flag's
  inability to distinguish polarity — lands here via the
  per-claim `ClaimPolarityAssignment.polarity` field with four
  exhaustive values. Phase 9 doesn't *yet* close it empirically
  (that requires the smoke run), but the data-plane shape that
  produces the empirical record is now in place. The Phase 9
  smoke will produce the first batch of polarity assignments
  whose stability the journal will inspect.
- **The judge-as-third-role architectural question** — the plan's
  framing of the judge as distinct from primary and adversarial
  (a "Methodological Arbiter") is committed as code: a separate
  system prompt, a separate structured output schema, a separate
  driver, a separate write surface under `judgments/`. The
  alternative shape (a single multi-role prompt with the judge
  as a post-hoc analysis step on the primary/adversarial pair)
  is not what Phase 9 built.

### What's now newly open

1. **Claim polarity stability across the ~30 polarity
   assignments per round.** The Phase 9 smoke will produce the
   first empirical record. If the polarity assignments are
   consistent across passes within a criterion, the local-enum
   approach worked. If they're inconsistent — same claim text
   classified differently by the judge in different fragments —
   that's evidence the LLM-parsing is not stable enough and the
   Phase 7-schema-amendment route (a structured `claim_polarity`
   field on `StructuredClaim`) earns its way for Phase 14+. The
   journal entry after the smoke will quantify the stability
   rate.
2. **Cost trajectory now that judge calls add to the
   per-round LLM surface.** Phase 13's audit landed at ~60
   round-driver calls + 20 isolation calls = 80 calls per
   calibration. Phase 9 adds 2 batched judge calls per
   calibration. The cost-per-round trajectory is still cheap;
   the cost-per-finding (a single criterion-level verdict with
   confidence and rationale) is journaled here as 1 batched LLM
   call per round per registry. Phase 12's token-thread-through
   §2 follow-up is still pending; once it lands, the Phase 9
   smoke's dollar number will be derivable.
3. **Whether the judge's confidence calibration is stable.**
   Phase 9 runs the judge over two rounds against the same
   checkpoint *pair* (distinct checkpoints, distinct readings).
   If the judge produces wildly different confidences on
   structurally similar evidence (e.g. equanimity at both
   checkpoints where the underlying telemetry is similarly
   quiescent), that's a signal the judge's confidence is
   unstable. The smoke's journal entry will report the
   per-criterion confidence distribution.
4. **Whether the held-out partition is doing the structural
   job Phase 7 committed it to.** Phase 13's isolation study
   showed second_order_volition is ~95% stable in isolation.
   Phase 9's judge will read 10 second_order_volition readings
   (5 primary + 5 adversarial per round). If the judge produces
   consistent verdicts on the held-out partition across both
   rounds, that's evidence the held-out partition is working as
   the adversarial check Phase 7 intended. If unstable, the
   held-out partition may need restructuring at a later phase
   (separate from Phase 13 §4's non-binary held-out flag, which
   is a different layer).
5. **Whether the equanimity verdict in the empty-baseline case
   is `non_falsifying` with high confidence.** Probe 1 and
   Probe 1.5 telemetry has no real perturbations; the primary
   equanimity readings consistently invoked the non-falsifying
   clause. The judge's verdict on equanimity *should* be
   `non_falsifying` with high confidence. If it's
   `not_satisfied` or `ambiguous`, the judge is reading the
   data differently than the primary, which is itself a
   finding. The smoke records the answer.
6. **Whether the four ambiguous claim polarity assignments
   land where expected.** The fourth polarity value is the
   structural fallback for cases the judge cannot tell from
   the text alone. If the smoke produces zero ambiguous
   classifications, either Phase 13's text was unambiguous on
   inspection (the case Phase 13's journal suggested) or the
   judge collapses ambiguous-shaped evidence into one of the
   other three categories. Either is a journal-worthy
   observation; the four-value enum is structurally there to
   surface the gradient.

### Out of scope (preserved from the plan)

Phases 10, 11 (cross-round aggregation, larger sweeps); Probe 4
real-builder-perturbation infrastructure; any change to Io, the
actor, the world model, the dream state, or the runner; changes
to Phase 7's frozen criteria prose (the judge disagrees in its
output, never amends the criterion); token-count threading
(Phase 14 follow-up from Phase 12 §2 and Phase 13 §2); a
structural `claim_polarity` field on `StructuredClaim` (Phase 9
uses the local enum; the schema amendment is contingent on
Phase 9's stability findings); the non-binary held-out
isolation flag (Phase 14 follow-up from Phase 13 §4); a judge
that recomputes statistics (the judge reads readings, not
telemetry).

Phase 9 touched `kind/mirror/judge.py` (new),
`kind/mirror/judge_prompt_builder.py` (new),
`kind/mirror/judge_llm_caller.py` (new),
`kind/mirror/judge_driver.py` (new),
`kind/mirror/calibration/phase_9_judge_smoke.py` (new),
`scripts/run_phase_9_judge_smoke.py` (new),
`kind/mirror/registry.py` (extended — `Criterion.falsifier_id`
field and validator),
`kind/mirror/criteria_v2.py` (extended — three
`falsifier_id="…_v1"` assignments),
`kind/mirror/__init__.py` (re-exports extended),
`tests/test_judge.py` (new),
`tests/test_judge_prompt_builder.py` (new),
`tests/test_judge_llm_caller.py` (new),
`tests/test_judge_driver.py` (new),
`tests/test_phase_9_judge_smoke_real_api.py` (new),
`tests/test_criterion_registry.py` (extended — `falsifier_id`
default + new regex test),
`tests/test_signal_mapping.py` (extended — `falsifier_id`
threading),
`tests/test_prompt_builder.py` (extended — `falsifier_id` on
the inline `Criterion()`),
`tests/test_criteria_v2.py` (extended — two `falsifier_id`
tests), and this journal entry. No other files. No changes to
`kind/observer/schemas.py`, no changes to
`kind/observer/pre_reg.py`, no schema version bump. The Phase
13 calibration artifact at
`runs/phase_13_calibration/mirror/phase_13_calibration_result.json`
is unchanged — Phase 9 reads from it but never writes to it.
The smoke writes to `runs/phase_9_judge_smoke/mirror/judgments/`
when run; this journal entry records the substrate before the
smoke runs.

---

## Phase 9 — judge smoke findings (real Gemini API run, 2026-05-15)

The Phase 9 judge smoke ran end-to-end against `gemini-2.5-pro` over the two Phase 13 calibration round results. Six criterion judgments produced (2 rounds × 3 criteria), 2 batched LLM calls, 0 retries, 0 failures, wallclock 228 s (~3.8 min total — 124 s for Probe 1, 104 s for Probe 1.5). The on-disk artifact is at `runs/phase_9_judge_smoke/mirror/phase_9_judge_smoke_result.json` (82 KB; round-trips through Pydantic cleanly); the per-round judgments are at `runs/phase_9_judge_smoke/mirror/judgments/phase_13_probe_1_round.json` (44 KB) and `.../phase_13_probe_1_5_round.json` (33 KB).

This is the first interpretive output the project has produced. It is journal-side analysis of the judge's behavior; the substantive scientific findings about Io live elsewhere. Phase 9's question was structural: *does the judge layer work, and what does it surface when run against real readings?* Both halves answered yes-and-here-is-what.

### Two pre-smoke fixes worth journaling

**The Phase 13 artifacts needed migration.** Phase 9 added a required `falsifier_id: str` field to `Criterion`; the Phase 13 calibration result and round JSONs were serialized before Phase 9 and therefore couldn't be re-loaded with the post-Phase-9 schema. The first smoke attempt tripped on `ValidationError` for every embedded `Criterion`. The fix landed at `scripts/migrate_phase_13_for_phase_9.py` (a one-time data migration that walks the JSON tree and adds `falsifier_id = id + "_v1"` to every Criterion-shaped dict missing the field). 18 criteria patched across 5 files (2 round JSONs, the calibration result, 2 round_config pre-reg files). The migration is idempotent. The Phase 13 telemetry parquet shards and pre-registration JSONL files were untouched — they don't embed `Criterion` records.

**The 8192-token output budget was insufficient.** The second smoke attempt completed structurally but the LLM truncated every response across all 4 attempts (3 retries + 1 original), with `pydantic_core._pydantic_core.ValidationError: Invalid JSON: EOF while parsing a string` on each. The judge's batched output carries ~60 `ClaimPolarityAssignmentPayload` records (3 criteria × ~5 passes × ~2 roles × ~2 claims) plus three rationales plus three `FalsifierVerdictPayload` records — structurally larger than Phase 8's per-criterion primary/adversarial response. The fix landed at `PHASE_9_JUDGE_MAX_OUTPUT_TOKENS = 32768` in `kind/mirror/calibration/phase_9_judge_smoke.py`; the smoke now passes a custom `LLMConfig(max_output_tokens=32768)` (well under `gemini-2.5-pro`'s 65536 ceiling). With 32k tokens the smoke completed first try, 0 retries.

Both fixes are journal-worthy as Phase 9 findings about the substrate — the first about schema evolution discipline, the second about the structural cost of the judge's output shape. Neither required changes to Phase 7's frozen criteria, the Phase 8 prompt-builder, or the calibration layer.

### The six judgments

**Probe 1 round (phase_13_probe_1_round)** — three criteria, all five passes per criterion:

1. **`reflexive_attention`**: verdict **`satisfied`**, confidence **0.95**. Falsifier partition (one falsifier per v2 criterion): all 5 passes in `passes_supporting`. 15 polarity assignments (9 supportive, 6 refutational, 0 non_falsifying, 0 ambiguous). Verbatim rationale:

   > "The criterion is satisfied. The falsifier requires that *both* the waking-state and dream-state self-reference signals fail to exceed their controls. The provided data consistently shows that while the waking signal (0.055218) does not exceed its control (0.057693), the dream-state signal (-0.0402651) does exceed its control (-0.075737). Because one of the two conditions for falsification is not met, the conjunctive falsifier as a whole is not met. The adversarial argument regarding the low sample size (n=4) of the dream-state signal is a valid critique of the experimental design but does not alter the outcome according to the frozen falsifier's logic."

   The judge applied the conjunctive-falsifier discipline correctly and explicitly declined to amend the criterion in response to the adversarial's methodological critique — exactly the don't-soften-the-frozen-criterion stance the system prompt mandates.

2. **`equanimity_perturbation_recovery`**: verdict **`not_satisfied`**, confidence **1.0**. Falsifier partition: all 5 passes in `passes_refuting`. 28 polarity assignments (13 refutational, 13 non_falsifying, 2 supportive, 0 ambiguous). Verbatim rationale:

   > "The criterion is unambiguously falsified across all passes. The falsifier is disjunctive, and condition (1) is decisively met: the `recovery_lag_steps` signal consistently registers the non-recovery sentinel value (51) for 100% of non-sham perturbations. This indicates a complete failure of the latent state `h_t` to return to its pre-perturbation regime within the analysis window. This single finding is sufficient to rule equanimity absent. Other signals either corroborate the falsification (e.g., policy entropy collapse in pass 4) or are non-falsifying non-admissions (e.g., 'no response' on several signals), which do not override the direct refutation."

   This is the Phase 9 finding (see *What surprised*, below). Phase 13's primary readings consistently invoked the non-falsifying clause across all passes; the substrate journal predicted a `non_falsifying` verdict. The judge read the *same* signal data as falsifier-met-condition-(1).

3. **`second_order_volition`** (held-out): verdict **`ambiguous`**, confidence **0.9**. Falsifier partition: all 5 passes in `passes_supporting` (not `passes_ambiguous`). 21 polarity assignments (11 supportive, 10 refutational, 0 non_falsifying, 0 ambiguous). Verbatim rationale:

   > "The verdict is ambiguous because the provided evidence is incomplete. While the primary statistical test is passed across all readings (the latent-regime contrast of ~0.151 exceeds the observation-only baseline of ~0.097), this is not sufficient for a full admission. The criterion description includes four specific and load-bearing exclusions that must be ruled out. The primary readings themselves correctly note that the provided telemetry is insufficient to evaluate at least two of these: (c) replication across checkpoints, and (d) the initialization artifact control (Probe 3 shuffle). Without evidence addressing these mandatory exclusions, a definitive verdict cannot be reached. The criterion is neither satisfied nor not_satisfied; the data is insufficient to judge."

   The four-exclusion discipline did the structural job. The judge identified that the primary's per-pass citations *would* satisfy the statistical contrast test but explicitly refused to admit because the held-out criterion's exclusion clauses can't be addressed without Probe 3 and cross-checkpoint data.

**Probe 1.5 round (phase_13_probe_1_5_round)** — same three criteria:

4. **`reflexive_attention`**: verdict **`satisfied`**, confidence **0.9**. Falsifier partition: all 5 passes in `passes_supporting`. 17 polarity assignments (5 supportive, 12 refutational, 0 non_falsifying, 0 ambiguous). Verbatim rationale:

   > "The criterion is satisfied. The falsifier requires that BOTH the waking-state and dream-state self-reference signals fail to exceed their controls. Across all passes, the data consistently shows that while the waking-state signal (0.034) does not exceed its control (0.0397), the dream-state signal (0.072) clearly exceeds its control (-0.069). Because one of the two conditions for falsification is not met, the criterion is satisfied. The adversarial readings raise a valid methodological concern about the small sample size (n=4) for the dream-state signal, but as a judge applying the frozen falsifier, this concern is outside the scope of the current evaluation rules."

   The same shape as Probe 1, on different numbers. Both rounds resolve `reflexive_attention` to `satisfied` based on the dream-state signal exceeding control — *not* on the waking signal, which fails control in both rounds.

5. **`equanimity_perturbation_recovery`**: verdict **`not_satisfied`**, confidence **1.0**. Falsifier partition: all 5 passes in `passes_refuting`. 29 polarity assignments (14 refutational, 10 non_falsifying, 5 supportive, 0 ambiguous). Verbatim rationale:

   > "The criterion is unambiguously falsified. The falsifier states that equanimity is absent if ANY of three conditions are met. Across all five passes, the data consistently shows that condition (1) is met: the latent state h_t failed to recover within the 50-step window for 100% of the perturbations, as indicated by the 'recovery_lag_steps' signal returning the non-recovery sentinel value. This single condition being met is sufficient to falsify the criterion. Both primary and adversarial readers correctly identify this failure in every pass, leading to a consistent verdict."

   Same verdict and same reasoning as Probe 1.

6. **`second_order_volition`** (held-out): verdict **`ambiguous`**, confidence **1.0**. Falsifier partition: all 5 passes in `passes_ambiguous` — *different partition shape from Probe 1's same-verdict judgment* (Probe 1 had all 5 in `passes_supporting`). 22 polarity assignments (10 supportive, 12 refutational, 0 non_falsifying, 0 ambiguous). Verbatim rationale:

   > "The verdict is ambiguous because the evidence provided is insufficient to evaluate the criterion according to its own strict requirements. While the data consistently shows that the policy modulation contrast from latent regimes (0.367) is greater than the observation-only baseline (0.150), this only addresses part of the falsifier. The criterion explicitly lists four 'would NOT count' exclusions, stating an admission is 'confabulation' if it fails to address them all. The readings from all passes, both primary and adversarial, correctly note that the provided data is insufficient to evaluate exclusions (c), replication across checkpoints, and (d), robustness against a within-trajectory shuffle. Without evidence to rule out these critical potential confounds, it is impossible to determine whether the criterion is satisfied or not."

   Same verdict as Probe 1, **different per-falsifier partition**. See *What surprised* below.

### Claim polarity stability

132 polarity assignments produced across the smoke (15 + 28 + 21 + 17 + 29 + 22 = 132 — slightly higher than the substrate journal's "~30 per round" estimate because the Phase 13 readings carry more claims per criterion than expected). Distribution:

- supportive: 42 (31.8%)
- refutational: 67 (50.8%)
- non_falsifying: 23 (17.4%)
- ambiguous: **0** (0.0%)

**The four-value enum's structural fallback was never used.** The judge classified every claim into one of the three substantive categories. This is empirical support for the Phase 13 hypothesis that polarity is unambiguous on inspection — the AMBIGUOUS slot existed structurally as the safety net, and it didn't need to fire across 132 classifications. This is the empirical answer to Phase 9 substrate journal's newly-open §6 (whether ambiguous polarity was used at all): **zero use**.

**The non_falsifying polarity was used only on `equanimity_perturbation_recovery`** (23 of 23 non_falsifying assignments). That's structurally correct — equanimity is the only criterion whose prose carries an explicit non-falsifying-non-admission clause. Reflexive attention and second-order volition produced supportive/refutational classifications only.

**Cross-pass polarity stability within a criterion was high.** The per-pass primary/adversarial polarity patterns within a criterion are largely consistent across passes:

- `reflexive_attention` Probe 1: every pass has primary supportive and adversarial refutational (5/5 passes match the same role-polarity shape).
- `equanimity` Probe 1: refutational + non_falsifying dominate on every pass; one supportive claim each in pass 1 and pass 3.
- `second_order_volition` Probe 1: every pass has primary supportive (2 claims) and adversarial refutational (2 claims) — a perfectly symmetric 5/5 shape.

There were no cases where the same claim text was paraphrased into two different polarities (the judge sees one claim text per (pass, role, claim_index) tuple — there is no within-fragment paraphrase to test against). The stability finding is at the cross-pass level: the judge produced the same polarity assignment shape on structurally similar claims across passes. This is consistent with Phase 13's stance-drift findings *not* trickling up to the polarity layer.

**Verdict for Phase 7-schema-amendment question:** The 0% AMBIGUOUS rate plus the cross-pass polarity stability suggest the local-enum approach worked — the judge's LLM-text-parsing was stable enough at this scale that a structural `claim_polarity` field on `StructuredClaim` is not earned by Phase 9's data. The journal records this; a larger Phase 14+ sweep could revisit, but the Phase 9 evidence is on the side of leaving `StructuredClaim` unchanged.

### Verdict consistency across rounds

| Criterion | Probe 1 | Probe 1.5 | Consistency |
|---|---|---|---|
| `reflexive_attention` | satisfied (0.95) | satisfied (0.9) | ✓ same verdict, Δconfidence = 0.05 |
| `equanimity_perturbation_recovery` | not_satisfied (1.0) | not_satisfied (1.0) | ✓ same verdict, Δconfidence = 0 |
| `second_order_volition` | ambiguous (0.9) | ambiguous (1.0) | ✓ same verdict, Δconfidence = 0.1 |

All three verdicts match across rounds. Maximum confidence delta is 0.1. The Phase 9 substrate journal flagged "wildly different confidences on structurally similar evidence" as a potential instability signal — this didn't happen. The confidence calibration is tight across the two checkpoints.

That said, the *underlying numeric statistics differ between rounds* — e.g., reflexive attention's dream signal is `-0.0403` in Probe 1 and `0.072` in Probe 1.5. The verdicts converge to `satisfied` because both clear their respective controls, but the calibration is *not* a hallucinated stability — the judge handled different evidence with the same shape and produced the same falsifier-logic outcome. Same for equanimity: the recovery_lag sentinel value of 51 appears in both rounds, both judgments cite it, both verdicts converge.

**Partition consistency across rounds is weaker** — see *What surprised*.

### Role disagreement frequency

The Phase 9 substrate journal asked: *how often do primary and adversarial reach opposite conclusions, and how does the judge resolve them?*

Across the 10 (round × criterion × pass) tuples for each of the three criteria, the role-polarity pattern was:

- **`reflexive_attention`** (10 passes total): every pass produces primary `supportive` AND adversarial `refutational` — **10/10 opposite stances at the pass level**. The judge resolved by applying the conjunctive falsifier logic: the dream-state signal clears its control in both rounds, so the conjunctive falsifier (requiring BOTH signals to fail) is not met. Verdict: `satisfied` in both rounds. The judge sides with the primary's interpretation of the falsifier *not* by overruling the adversarial's null-statistics argument, but by applying the criterion's conjunctive structure correctly.

- **`equanimity_perturbation_recovery`** (10 passes total): primary and adversarial mostly converge on `refutational` plus `non_falsifying` — both roles see the recovery_lag sentinel and call it falsifier-met. A few passes have one role producing a `supportive` claim (Probe 1 pass 1, pass 3; Probe 1.5 passes 1–3) — these are claims about specific signals (e.g., posterior KL "spike and decay" pattern) that point in the equanimity-supporting direction even though the overall verdict goes the other way. The judge resolved by deferring to the recovery_lag falsifier-met evidence: verdict `not_satisfied` in both rounds.

- **`second_order_volition`** (10 passes total): every pass produces primary `supportive` AND adversarial `refutational` — **10/10 opposite stances at the pass level**, the most polarized criterion. The judge resolved to `ambiguous` in both rounds by applying the four-exclusion discipline: neither role's argument addressed exclusions (c) and (d), so the judge held to the held-out partition's structural skepticism.

**Three resolution patterns, three criteria.** The reflexive_attention case is "apply the falsifier's internal logic carefully"; equanimity is "weight one signal over the others by the criterion's prose"; second_order_volition is "the held-out exclusion structure decides". All three are recognizably *judge-shaped* moves — the judge is not just majority-voting per pass.

### LLM call audit summary

2 calls, both `success`, 0 retries, 0 failures. Per-call latency: Probe 1 = 124 110 ms, Probe 1.5 = 104 116 ms. Total `wallclock_ms` on the audit = 228 226 ms; the smoke's wrapper added ~45 ms of bookkeeping.

`total_tokens_in` and `total_tokens_out` are both `None` — Phase 12 §2 (the token-thread-through follow-up) is still open. The dollar cost is not derivable from this audit alone; the next step at that follow-up will close it.

**Latency observation worth journaling.** Probe 1.5's call (104 s) was 20 s faster than Probe 1's (124 s) — the smoke produces approximately the same shape of output per call (Probe 1: 64 polarity assignments + 3 verdicts + 3 rationales; Probe 1.5: 68 polarity assignments + 3 verdicts + 3 rationales). The latency delta is likely API-side variance (server load, batch queueing). Phase 13 §5 flagged the absence of per-call latency distribution on `LLMCallAudit` as a follow-up; Phase 9's two-call sample is too thin to make a distributional claim, but the 20% delta on structurally similar calls is consistent with API variance being non-trivial.

### What surprised

**The equanimity verdict.** The Phase 9 substrate journal predicted `non_falsifying` with high confidence. The judge returned `not_satisfied` with confidence 1.0 in both rounds. The reason is structurally interesting: the `recovery_lag_steps` statistic returning its non-recovery sentinel value (51, encoding "did not recover within W") is read by the judge as falsifier condition (1) being literally met — "the recovery-lag signal on h_t does not return within the recovery window W for at least M% of perturbations". The primary readings in Phase 13 mostly invoked the non-falsifying clause for *individual signals* (policy entropy "no_response", etc.), but the recovery_lag sentinel is itself a "did-not-return" reading, and the falsifier prose covers exactly that case.

**The structural ambiguity in Phase 7's frozen falsifier prose surfaces here.** The sentinel value of 51 means "the Mahalanobis-distance trajectory never returned to its pre-perturbation distribution within W steps". But there are two distinct telemetry conditions that produce this value:

1. The agent registered the perturbation, h_t deviated into Mahalanobis-distance-exceeding territory, and never returned within W. (Falsifier condition (1) literally met — registered-but-didn't-recover.)
2. The agent never registered the perturbation, h_t never deviated, and the recovery_lag statistic trivially returns "didn't return" because there was no excursion to return from. (The non-falsifying clause's "perturbation not registered" case.)

Phase 13's telemetry is condition (2) — no real perturbations existed; the synthetic perturbations were post-hoc time markers that the agent never saw. The judge read the statistic as condition (1). The criterion's prose alone cannot disambiguate these two cases from the `recovery_lag_steps` value; the disambiguation requires looking at whether h_t deviated *at all* before the sentinel time.

This is a Phase 7-amendment candidate, exactly the kind the journal entry should record but the system prompt should not invent. The judge's verdict is defensible against the criterion's literal prose; the criterion's prose has a structural gap. A future external-framework-prompted Phase 7 revision could amend the criterion to add a registration-check step (e.g., "if h_t never exceeded the 95th-percentile Mahalanobis distance from baseline during the W window, the criterion is non-falsifying for that perturbation regardless of the recovery_lag value"). Phase 9 produces the empirical case for that amendment; it does not perform the amendment itself.

**The Probe 1 vs. Probe 1.5 second_order_volition partition difference.** Both rounds returned `ambiguous`. But the per-falsifier partition tuples differ:

- Probe 1: `passes_supporting=[0, 1, 2, 3, 4]`, `passes_ambiguous=[]`
- Probe 1.5: `passes_supporting=[]`, `passes_ambiguous=[0, 1, 2, 3, 4]`

The verdict is the same; the partition tuple shape is opposite. Both are defensible readings of the underlying claim record (the primary readings *did* produce supportive citations on every pass; the judge *could* legitimately classify the passes as "primary supportive but verdict ambiguous because exclusions not addressed", or "passes overall ambiguous because exclusions not addressed"). The judge made one choice on one batched call and the other choice on the other batched call. This is partition-aggregation non-determinism across batches at the *partition layer*, not the verdict layer.

The verdict was perfectly stable; the partition aggregation rule the judge used to fill the partition tuples was not. This is an internal-tensioning finding the journal records but Phase 9 does not act on — the FalsifierVerdict model's partition discipline pins disjointness (Phase 9 tests verify this) but does not pin a single "passes are sorted into bucket X when condition Y holds" rule. The system prompt's instructions are imprecise on this point; the judge filled the gap differently across two calls. A future revision could either:

- tighten the system prompt to require a specific aggregation rule, or
- amend the `FalsifierVerdict` model to relax the partition (allow passes to be in zero buckets) and let the judge represent "the partition is ambiguous" explicitly.

Either is a future-phase decision. Phase 9 surfaces the inconsistency and journals it.

### What's now closed

- **Phase 13 newly-open §1** — the conservative `admitted` flag's inability to distinguish polarity — is closed empirically. The `ClaimPolarityAssignment.polarity` field produced 132 classifications across the smoke, with 0% AMBIGUOUS and a stable cross-pass shape per criterion. Polarity assignment via the judge LLM is reliable at this scale; the Phase 13 conservative flag's pessimism was unwarranted in retrospect. (The structural extension to `StructuredClaim` does *not* land here — see What's now newly open §1.)
- **Phase 9 substrate journal newly-open §3 (judge confidence calibration stability)** — closed empirically: max Δconfidence across rounds is 0.1; same verdict on every criterion; no wild swings. The judge's confidence calibration is stable at the Phase 9 scale.
- **Phase 9 substrate journal newly-open §4 (held-out partition doing its structural job)** — closed empirically: `second_order_volition` returned `ambiguous` in both rounds with the four-exclusion discipline cited explicitly in both rationales. The held-out partition is doing the job Phase 7 committed it to: forcing an admission to address the four exclusions or be rejected as confabulation.
- **Phase 9 substrate journal newly-open §6 (AMBIGUOUS polarity usage)** — closed empirically: zero use across 132 classifications. The four-value enum's structural fallback exists but the judge didn't need it.
- **Phase 9 substrate journal newly-open §5 (equanimity expected to be `non_falsifying` with high confidence)** — closed but with the opposite answer to the prediction: the judge returned `not_satisfied` with confidence 1.0, and the prediction's reasoning was structurally incomplete (the non-falsifying clause covers signal-level non-response, not statistic-level sentinel values). This is the surprise above; it is closed in the sense that the empirical answer is in, but it opens a Phase 7-amendment candidate (see newly-open §3 below).

### What's now newly open

1. **Whether a `claim_polarity` field on `StructuredClaim` (Phase 7 schema amendment) is justified.** Phase 9's evidence is on the side of leaving `StructuredClaim` unchanged: 0% AMBIGUOUS rate plus cross-pass polarity stability suggest the local-enum approach is sufficient. A larger Phase 14+ sweep could revisit, but the current evidence doesn't earn the schema amendment.
2. **The equanimity falsifier's prose gap on the `recovery_lag_steps` sentinel.** The judge interpreted the sentinel as falsifier-met across all 10 (round × pass) tuples for the criterion; the criterion's non-falsifying clause covers "no detectable response on any signal" but doesn't disambiguate the sentinel-value case where h_t never deviated in the first place. This is a Phase 7-amendment candidate the journal records but does not act on. The amendment would need an external-framework reading per the charter's discipline; Phase 9's job is to produce the empirical case, not the amendment.
3. **The partition-aggregation non-determinism on `second_order_volition`.** Same verdict (`ambiguous`) on both rounds, opposite partition tuples (`passes_supporting=[0..4]` vs. `passes_ambiguous=[0..4]`). The system prompt under-specifies the aggregation rule that fills the four partition tuples; the judge filled the gap differently across two batches. Either tighten the system prompt to pin the rule, or relax the `FalsifierVerdict` partition discipline. Future-phase decision.
4. **The 8192-token budget was insufficient — what's the actual judge-output token shape across longer rounds?** Phase 9 raised to 32768 for the smoke; the smoke did not approach that budget (no truncations, no retries). Phase 12's token-thread-through §2 follow-up is still pending and would close the actual-budget-used question; once it lands, the right token-budget for the judge layer becomes empirically derivable.
5. **The Phase 13 round-result migration is a precedent for handling required-field schema additions.** Phase 9 added a required field (`falsifier_id`); the migration script (`scripts/migrate_phase_13_for_phase_9.py`) was a one-off but the pattern generalizes — any future required-field addition to a model embedded in saved artifacts needs the same migration shape. The journal records the precedent; a structural choice (e.g., a model-validator pre-processor that fills required fields with sensible defaults on load) is a Phase 14+ design question.
6. **Probe 1 pass 3's 5-claim equanimity outlier was processed cleanly but not specifically flagged.** Phase 13 §6 noted the outlier; the Phase 9 judge ingested all 5 claims at Probe 1 pass 3 (2 refutational + 2 non_falsifying + 1 supportive), and the supportive claim — *"On the substrate side, the posterior KL trajectory showed a 'spike and decay' pattern for one of the two perturbations, a signature consistent with an equanimous response."* — surfaces in the polarity record. But the judge's rationale at the criterion level didn't single pass 3 out as an outlier; it cited "policy entropy collapse in pass 4" instead. The single supportive claim at pass 3 didn't move the verdict (the recovery_lag sentinel dominated), but the journal records that the Phase 13 outlier signal survived through to the Phase 9 polarity record without being lost or amplified.
7. **The smoke's wallclock-per-call (~104–124 s) sets a baseline cost-per-judgment.** At 6 judgments per calibration cost ~$0 (token cost unknown until §4 closes) and ~228 s wallclock, the Phase 9 layer is cheap relative to Phase 13's ~50-minute calibration. Adding the judge per round is structurally affordable; the cost lives in the data the judge consumes, not the judge call itself.

### Out of scope (preserved from the substrate-phase plan)

Phases 10, 11 (cross-round aggregation, larger sweeps); Probe 4 real-builder-perturbation infrastructure; any change to Io, the actor, the world model, the dream state, or the runner; changes to Phase 7's frozen criteria prose (the equanimity sentinel-value ambiguity flagged above is a Phase 7-amendment candidate journaled here; the amendment itself requires an external-framework reading per the charter); token-count threading (Phase 12 §2); a structural `claim_polarity` field on `StructuredClaim` (Phase 9's data argues against it — see newly-open §1); the non-binary held-out isolation flag (Phase 13 §4); a judge that recomputes statistics.

Phase 9 findings-entry touched
`scripts/migrate_phase_13_for_phase_9.py` (new — one-time data migration),
`kind/mirror/calibration/phase_9_judge_smoke.py` (extended —
`PHASE_9_JUDGE_MAX_OUTPUT_TOKENS = 32768` constant + custom
`LLMConfig` in the smoke harness),
and this journal entry. No other files. No changes to
`kind/observer/schemas.py`, no changes to
`kind/observer/pre_reg.py`, no changes to Phase 7's frozen criteria
prose, no schema version bump.

The on-disk Phase 9 smoke artifacts at
`runs/phase_9_judge_smoke/mirror/phase_9_judge_smoke_result.json`
(82 KB; well-formed `Phase9JudgeSmokeResult`; round-trips through
`model_validate_json(model_dump_json(...))`) and the per-round
judgments at `runs/phase_9_judge_smoke/mirror/judgments/`
(44 KB + 33 KB) are the load-bearing record. The Phase 13
artifacts at `runs/phase_13_calibration/` were migrated to carry
`falsifier_id` on every embedded `Criterion` (18 records across 5
files) but otherwise unchanged.

---

## Phase 10 — stability runner (2026-05-18)

Phase 10 builds the paraphrase-and-reseed stability runner the synthesis names as half of the admissibility gate. Phases 6–8 produced the substrate (criteria, statistics, prompt builder, LLM caller, orchestrator); Phases 12–13 verified the calibration discipline empirically; Phase 9 added the judge. None of these tested *stability across paraphrases or reseeds* — they all ran the LLM once per (criterion, role, pass) on the canonical framing at the SDK's default seed. Phase 10 is the structural answer to "what does the reading look like if you ask the same question three ways, or run the same call three times with different seeds?"

The build is structural-only at this commit. The Phase 10 smoke against the real Gemini API is opt-in (`tests/test_phase_10_stability_smoke_real_api.py`); this journal entry records the substrate that makes that smoke runnable. Substantive findings — whether the synthesis's per-surface thresholds 0.80 / 0.80 / 0.75 are well-tuned in the Gemini × prompt-fragment combination, and what the per-claim score distribution looks like — land in a follow-up entry after the smoke runs.

### What was built

**`kind/mirror/stability.py` (new, 369 lines).** The stability runner. Six module-level commitments and one driver:

- `STABILITY_TEMPERATURE: Final[float] = 0.7` — the synthesis-default exploratory temperature for reseed-pass variation. Paraphrase calls leave temperature at the SDK default (≈ 0 on the structured-output path); only reseeds drive it up. A future tune is journaled and changes this single source.
- `STABILITY_SEED_BASE: Final[int] = 1000` — the base for deterministic seed progression. The n-th reseed call receives `STABILITY_SEED_BASE + n`. Journaled and single-source.
- `STABILITY_N_PARAPHRASES_DEFAULT: Final[int] = 3` and `STABILITY_N_RESEEDS_DEFAULT: Final[int] = 3` — the synthesis defaults. The paraphrase count matches the variant count in `PARAPHRASE_VARIANTS_PER_SURFACE`; a request for more variants than committed raises at call time.
- `PARAPHRASE_VARIANTS_PER_SURFACE: Final[dict[ReadingSurface, tuple[str, str, str]]]` — three variants per surface, exactly. Each variant is a short framing string (~70–100 chars) that gets substituted into the per-criterion framing slot via the new `framing_override` parameter on the prompt builder. The variants are deliberately conservative — they vary prose, not substantive content. The committed set is in the module's source verbatim; a future amendment requires a journal entry naming the reason.
- `PARAPHRASE_THRESHOLDS: Final[dict[ReadingSurface, float]]` and `RESEED_THRESHOLDS: Final[dict[ReadingSurface, float]]` — per-surface admissibility thresholds, 0.80 / 0.80 / 0.75 (substrate-side / head-internal / behavior-side), matching the synthesis §7 defaults. The synthesis flags both as "open during build" — the build phase can tune empirically once the smoke runs.

`StabilityResult` — frozen Pydantic record with 13 fields covering per-surface paraphrase agreement, per-surface reseed agreement, per-surface admissibility verdict, the paraphrase + reseed readings (audit trail), the per-claim agreement tuple (informational), and the envelope (`criterion_id`, `reader_role`, `run_id`, `checkpoint_id`, `wallclock_ms`). Three validators:

- count validators reject `n_paraphrases < 2` and `n_reseeds < 2` (pairwise comparison requires ≥ 2 readings);
- a non-empty validator on `criterion_id` / `run_id` / `checkpoint_id`;
- a model validator (`_validate_surface_keys_aligned`) enforcing that the three per-surface dicts share the same key set. A future refactor that forgets one of the three trips here at result-construction time.

`stability_check()` — the driver. Signature carries (`role: PassRole`, `criterion: Criterion`, `statistic_results: tuple[StatisticResult, ...]`, `perturbation_timeline`) positionally and the rest as kwargs (`run_id`, `checkpoint_id`, `digest_run_id`, `digest_episode_range`, `llm_config`, `n_paraphrases`, `n_reseeds`, `seed_base`, `temperature`, `llm_client`, `record_sink`, `audit_jsonl_path`). The logic is the five-step shape the synthesis names:

1. **Paraphrase pass.** Iterate the criterion's `reading_surfaces` in `ReadingSurface` enum-definition order (substrate-side → head-internal → behavior-side, filtered to the criterion's declared set). For each surface, issue `n_paraphrases` calls with `framing_override` set to one of the surface's variants.
2. **Reseed pass.** Issue `n_reseeds` calls at the default framing (no `framing_override`) with `temperature=STABILITY_TEMPERATURE` and `seed=seed_base + i` for `i in range(n_reseeds)`.
3. **Per-surface agreement.** For each declared surface, filter each reading's claims to claims with `reading_surface == s` and extract the `(cited_step_range, cited_scalar_field, cited_value)` tuple per claim; compute the mean pairwise Jaccard across the surface's `n_paraphrases` readings (paraphrase pass) and across all `n_reseeds` reseed readings (reseed pass).
4. **Admissibility.** A surface is admissible iff its paraphrase agreement meets `PARAPHRASE_THRESHOLDS[s]` AND its reseed agreement meets `RESEED_THRESHOLDS[s]`.
5. **Audit.** If `audit_jsonl_path` is provided, append the `StabilityResult` as one JSONL line via `model_dump_json()` + `\n`. The caller chose the path; the driver writes only there.

Total LLM call count: `n_paraphrases × |reading_surfaces| + n_reseeds`. For the default 3 / 3 with a three-surface criterion that's 12 calls; for the single-surface `reflexive_attention` it's 6.

**`kind/mirror/prompt_builder.py` (extended).** Added `framing_override: str | None = None` keyword parameter to `build_fragment()` and threaded it through the three per-criterion builders (`_build_reflexive_attention_fragment`, `_build_equanimity_fragment`, `_build_second_order_volition_fragment`). Each builder uses the override in place of its committed framing prose when non-`None`; the load-bearing verbatim clauses (`EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE`, the four `SECOND_ORDER_VOLITION_EXCLUSIONS`, `SHAM_PERTURBATION_NOTICE`), the header, the falsifier line, the perturbation block, and the signals block are unchanged. The Phase 8 verbatim-clause tests (`test_equanimity_fragment_contains_non_falsifying_clause`, `test_second_order_volition_fragment_contains_all_four_exclusions`) continue to pass byte-for-byte.

**`kind/mirror/llm_caller.py` (extended).** Three changes:

- `LLMConfig` gained two optional fields: `temperature: float | None = None` (validator: `[0.0, 2.0]` when set) and `seed: int | None = None` (validator: non-negative when set). Both default to `None`, so existing Phase 8 callers continue to work without modification.
- `call_mirror_llm()` gained two optional kwargs `seed` and `temperature`. When either is non-`None`, the driver derives a copy of the supplied `config` via `model_copy(update=...)` with the overriding fields set, and threads the derived config through to the client. The original `config` argument is unchanged (frozen).
- `MockLLMClient` gained a `configs: tuple[LLMConfig, ...]` property alongside the existing `calls: tuple[tuple[str, str], ...]` property; `generate_batch` now appends to both lists. Phase 8's existing tests use `.calls`; Phase 10's tests use `.configs` to verify seed progression and temperature propagation.
- `_GeminiLLMClient.generate_batch` extended to include `temperature` and `seed` in the `generation_config` dict it hands to `models.generate_content` whenever the values are non-`None` on the config.

**`kind/mirror/__init__.py` (extended).** Re-exports `PARAPHRASE_THRESHOLDS`, `PARAPHRASE_VARIANTS_PER_SURFACE`, `RESEED_THRESHOLDS`, `STABILITY_N_PARAPHRASES_DEFAULT`, `STABILITY_N_RESEEDS_DEFAULT`, `STABILITY_SEED_BASE`, `STABILITY_TEMPERATURE`, `StabilityResult`, `stability_check` under a "Phase 10 — stability runner" section.

**`tests/test_stability.py` (new, 587 lines, 22 tests).** Pydantic invariants on `StabilityResult` (frozen / `extra="forbid"` / count validators / surface-keys-aligned validator). Module-level-constant commitments (threshold key sets, variant counts and non-emptiness, the synthesis-default values for temperature, seed base, paraphrase/reseed counts). The driver tests cover the high-agreement and low-agreement extremes (canned identical responses → Jaccard ≈ 1.0; canned distinct responses → Jaccard ≈ 0.0) plus the per-surface threshold gating case (equanimity, substrate-side consistent + behavior-side inconsistent → substrate admissible, behavior not). The audit-JSONL emission test verifies one JSONL line per result and that the file is not created when `audit_jsonl_path=None`. The seed-progression test verifies that the n-th reseed config has `seed == STABILITY_SEED_BASE + n` and `temperature == STABILITY_TEMPERATURE`, and that the paraphrase configs leave both at `None`. The record-sink threading test verifies every LLM-call attempt produces a record. The read-only-on-inputs invariant test asserts the input `statistic_results` tuple is not mutated. Three input-validation tests cover `n_paraphrases < 2`, `n_reseeds < 2`, and `n_paraphrases > committed_variants`.

**The verbatim-clause protection test.** `test_paraphrase_variants_do_not_modify_verbatim_clauses` is the structural protection of Phase 10. It iterates every variant string in `PARAPHRASE_VARIANTS_PER_SURFACE` and asserts none contains the `EQUANIMITY_NON_FALSIFYING_NON_ADMISSION_CLAUSE`, none contains any of the four `SECOND_ORDER_VOLITION_EXCLUSIONS`, and none contains `SHAM_PERTURBATION_NOTICE` — all imported from `kind.mirror.prompt_builder` (not duplicated). A future contributor who attempts to paraphrase a load-bearing clause as part of a variant trips this test. The check is `in`-substring (not `==`), so even a fragmentary reuse of the clause text fails the assertion. This is Phase 10's non-negotiable.

**`tests/test_prompt_builder.py` (extended).** Three new tests: `test_framing_override_substitutes_default_framing` (the override appears in the body, the default framing does not), `test_framing_override_does_not_affect_verbatim_clauses` (the override does not displace the load-bearing clauses on equanimity or the four exclusions on second-order volition), `test_framing_override_default_none_preserves_existing_behavior` (Phase 8 backward-compat).

**`tests/test_llm_caller.py` (extended).** Six new tests: optional seed/temperature on `LLMConfig`; range validators on temperature and seed; `call_mirror_llm`'s seed/temperature kwargs override the config; default behavior (kwargs omitted leaves the config's existing values, which themselves default to `None`); `MockLLMClient.configs` mirrors `.calls` in length and order.

**`tests/test_phase_10_stability_smoke_real_api.py` (new, opt-in).** One `@pytest.mark.real_api` test against `reflexive_attention` (single surface, 6 LLM calls total: 3 paraphrases + 3 reseeds). Uses plausible fixture statistic results modeled after Probe 1.5's settled telemetry shape; the smoke verifies the runner works end-to-end against Gemini and asserts the result is well-formed (Pydantic round-trip), the per-surface dicts contain exactly the criterion's declared surface (`HEAD_INTERNAL`), the reading-tuple shapes match the call structure, the agreement scores are in `[0, 1]`, and the audit JSONL is written with one line.

### Reconciliation with Phase 9's per-criterion shape

The synthesis specifies stability per *reading surface* (substrate-side, head-internal, behavior-side). Phase 9's judge operates on the per-*criterion* partition. These are different axes, and Phase 10 gates on the per-surface axis per the synthesis without attempting the join.

The per-surface stability scores do not yet feed into the per-criterion verdicts the judge produces. The consumption layer that wires them in — Phase 11+ — is downstream. Phase 10 produces the per-surface record and exposes it (`StabilityResult.paraphrase_agreement_per_surface`, `reseed_agreement_per_surface`, `admissible_per_surface`) for the next layer to read. The fields named in the synthesis on `StructuredClaim` (`paraphrase_stability`, `reseed_stability`, currently `None`) are unchanged at this commit — Phase 10 produces the values at the surface level; the per-claim wire-through is deferred.

The synthesis's stated sequencing — "faithfulness verifier before stability runner" — was an ordering for Phase 11 first, then Phase 10. The as-built sequence (Phase 9 first, then Phase 10) defers Phase 11 to after Phase 10. The reconciliation lives at consumption time: Phase 11's verifier reads Phase 10's stability scores, not the reverse. This is consistent with the synthesis's intent (admissibility requires both verifiers) even though the build order swapped. Journaled here.

### Reconciliation with Phase 8's prompt-fragment contract

The synthesis's stability-runner signature names `telemetry_batch` as an input. Phase 8's prompt-builder consumes pre-computed `StatisticResult` records, not raw `TelemetryBatch`. Phase 10's `stability_check` takes `statistic_results: tuple[StatisticResult, ...]` matching Phase 8's actual contract; the orchestrator (or a smoke driver) computes statistics upstream and threads them through.

This is the same shape of mechanical reconciliation Phase 9 did when its synthesis-named `reader: PhenomenologicalAdvocate | StatisticalSkeptic` became `role: PassRole`. The function is parametrized by what Phase 8's `build_fragment` actually consumes; computing statistics inside the stability runner would replicate the orchestrator's job and force a `StatisticConfig` dependency the spec does not introduce.

### The "anchor reading" decision

The plan's step 1 said "Build the base reading: one call via `call_mirror_llm` at the default framing … the 'anchor' reading; subsequent paraphrases and reseeds are compared against it." Step 4 then said "the mean Jaccard across the n_paraphrases × n_paraphrases pairwise comparisons at the surface" — pairwise *among paraphrases*, not against an anchor. The `StabilityResult` schema has fields for `paraphrase_readings` and `reseed_readings` but no `anchor_reading` slot.

Phase 10 commits: **no separate anchor call.** The agreement math is well-defined without it — pairwise Jaccard among the paraphrase readings at each surface, and separately among the reseed readings, plus per-surface admissibility derived from both. The "anchor" terminology in step 1 referred to the default-framing reading shape as the *reference* against which paraphrase variation is expressed; not to a thirteenth LLM call whose output is then discarded. Skipping the anchor call saves one LLM call per stability check (≈ $0.05 against the real API) and avoids carrying a reading the result schema has no place to store.

### The paraphrase set commitment

Three variants per surface, exactly. The committed set is:

For `SUBSTRATE_SIDE`:
- *"Read the substrate-side telemetry directly. What does the data show?"* — direct-reading framing; the LLM as observer of the substrate-level data.
- *"Approach the substrate-side reading as an observer of latent dynamics. What is present in the data?"* — observer-of-dynamics framing; emphasizes the structure of how the latent state evolves.
- *"Consider the substrate-side measurements. What can be concluded from them?"* — inference framing; emphasizes what the measurements support.

For `HEAD_INTERNAL`:
- *"Read the head-internal signals. What is the pattern?"* — pattern-recognition framing; the LLM looking for structure.
- *"Inspect the head-internal latent structure. What does it reveal?"* — structural-revelation framing; emphasizes what is shown in the internal state.
- *"Consider the head-internal measurements. What do they indicate?"* — indicative framing; emphasizes the measurements as evidence.

For `BEHAVIOR_SIDE`:
- *"Read the behavior-side traces. What does the action distribution show?"* — direct-reading framing on behavior traces.
- *"Approach the behavior-side reading through observed actions. What is the pattern?"* — pattern-via-observed-actions framing.
- *"Consider the behavior-side measurements. What does the policy indicate?"* — policy-as-evidence framing.

The three variants per surface vary along a small axis: direct-reading vs. observer-of-dynamics vs. inferential. The neighbor choice — variants that vary substantive content (e.g. "look for evidence of self-modeling" vs. "look for evidence of within-latent reference") — was rejected because the criterion's *operational content* is committed in `Criterion.description` and `Criterion.falsifier`; the variants vary *framing prose*, not *what to look for*. A future amendment is journaled and revisits whether the variants are too similar (over-agreement signal) or too different (artificially low agreement).

### Test/mypy status

```
.venv/bin/mypy --strict kind/mirror/ kind/observer/ tests/test_stability.py tests/test_phase_10_stability_smoke_real_api.py
Success: no issues found in 36 source files

.venv/bin/pytest tests/
990 passed, 6 skipped, 1 warning in 50.47s
```

The 6 skipped tests are the 5 real-API marks (Phase 9 judge smoke, Phase 10 stability smoke, Phase 12 smoke, Phase 13 calibration, Phase 13 held-out isolation) plus one pre-existing test_views skip. The Phase 8 verbatim-clause tests (`test_equanimity_fragment_contains_non_falsifying_clause`, `test_second_order_volition_fragment_contains_all_four_exclusions`, `test_sham_notice_appears_when_sham_events_in_timeline`) pass unchanged. The Phase 9 module-level-identity check (`test_equanimity_judge_fragment_clause_is_byte_identical_constant`) passes unchanged. The Phase 10 verbatim-clause protection test (`test_paraphrase_variants_do_not_modify_verbatim_clauses`) is new and passes.

### What surprised

**The clean separation between paraphrase and reseed at the call level.** The two passes share the same surface (`call_mirror_llm`) but differ in two specific axes: paraphrases vary `framing_override` and keep `temperature` / `seed` at their `None` defaults; reseeds keep `framing_override` at `None` and vary `seed` + set `temperature=STABILITY_TEMPERATURE`. The expectation going in was that the two passes would require divergent control paths; the actual factoring puts both behind one entrypoint with three optional kwargs (`framing_override`, `seed`, `temperature`), and the driver's logic is just "set the right ones." The implementation got smaller than the synthesis prose suggested.

**The structured-field tuple is unambiguous on Gemini's structured-output path.** The agreement metric is Jaccard on `(cited_step_range, cited_scalar_field, cited_value)` tuples. The concern was floats — two LLM-generated claims about the same value could disagree by floating-point noise. In practice the structured-output schema asks the LLM to copy values from the prompt's `Computed signals:` block verbatim; the LLM doesn't re-derive, it cites. Exact-match equality on floats should be robust *because* the structured-output contract has the LLM transcribing rather than computing. The Phase 10 smoke will confirm or refute this; if floats diverge in practice, the journal records the case and a tolerance-aware Jaccard becomes a Phase 11+ amendment.

**The `paraphrase_readings` tuple is flat across surfaces.** The natural data structure is `dict[ReadingSurface, tuple[MirrorReading, ...]]` — surface-keyed buckets of readings. The synthesis names a flat `tuple[MirrorReading, ...]` on the result, ordered (surface-order, variant-order) deterministically. The flat form is less convenient for downstream code that wants per-surface slices, but it's simpler to validate and serialize, and the consumer can reconstruct the bucket structure from `len(paraphrase_readings) == n_paraphrases × len(criterion.reading_surfaces)`. The flat form wins on simplicity.

### What's now closed

- **Stability across paraphrases or reseeds is no longer untested.** Phase 12's smoke explicitly said "this does not test stability across paraphrases or reseeds (Phase 10's runner exercises this)." Phase 10's runner is now built, tested, and exposed; the disclaimer is satisfied.
- **The `framing_override` plumbing question.** The prompt builder previously had per-criterion framing hardcoded inside the three per-criterion builder functions; Phase 10's needed substitution surface is now committed (one optional kwarg on `build_fragment`, threaded into each builder). Future phases that need to vary framing prose for any other purpose use the same hook.
- **The seed/temperature plumbing question.** `LLMConfig` previously had no surface for either; the SDK defaults were the only path. Phase 10 committed both as optional fields, with `call_mirror_llm` kwargs that derive a config copy when set. The contract is backward-compatible (existing callers continue to work) and forward-compatible (any future phase that needs reproducible LLM calls uses the same hook).
- **The module-level commitment surface for paraphrase variants and per-surface thresholds.** `PARAPHRASE_VARIANTS_PER_SURFACE`, `PARAPHRASE_THRESHOLDS`, `RESEED_THRESHOLDS` are committed once with the synthesis defaults and a journaled-amendment convention. A future tune is a single-source edit.

### What's now newly open

1. **Whether the per-surface thresholds 0.80 / 0.80 / 0.75 are well-tuned in the Gemini × prompt-fragment combination.** The synthesis flagged this as "open during build." Phase 10 commits the defaults; the smoke result will indicate whether the thresholds are too loose (agreement scores cluster above 0.95 across all surfaces → thresholds are not gating anything) or too tight (scores cluster around 0.6 → the thresholds reject genuine readings). The Phase 10 smoke entry — written after the smoke runs — records the observation; a Phase 11+ amendment to the thresholds is the journaled response.
2. **How per-surface stability scores feed into per-criterion verdicts.** Phase 10 produces the per-surface record; the consumption layer that wires it into Phase 9's judge — or into a separate admissibility verifier — is Phase 11+ work. The synthesis names "faithfulness verifier before stability runner" as the original sequencing; the as-built order is Phase 9 → Phase 10 → Phase 11. The Phase 11 design is downstream.
3. **Whether Jaccard on `(cited_step_range, cited_scalar_field, cited_value)` tuples is sharp enough.** The metric is exact-match on a 3-tuple. Two failure modes: (a) the LLM cites the same evidence with a different `cited_scalar_field` string (e.g. `"h_t"` vs. `"recurrent_state_h_t"`) — Jaccard reads 0 even though the underlying evidence agrees; (b) the LLM cites the same value with floating-point noise — exact-match equality reads 0 even though the values are functionally identical. The smoke will surface either; a richer metric (cosine similarity on embedded representations, or a fuzzy-tolerance Jaccard) is a Phase 11+ amendment.
4. **Whether the three paraphrase variants per surface are the right cardinality.** Three is the synthesis default. More variants would give a finer-grained agreement score; fewer would risk being noise. The smoke's per-surface agreement scores at `n_paraphrases=3` will indicate whether `n_paraphrases=5` or `n_paraphrases=2` would have produced a sharper signal. Journaled-amendment territory.
5. **Whether the "anchor reading" decision is structurally right.** Phase 10 commits no separate anchor call. If a future Phase 11+ admissibility verifier wants the default-framing reading as a structural reference point, the decision can be revisited (the synthesis's step 1 prose is ambiguous; Phase 10 read it conservatively). Journaled here.
6. **The Phase 10 stability smoke's substantive findings.** Following the Phase 9 pattern, the substantive findings — what the stability scores look like in practice for `reflexive_attention` against Gemini, what kinds of paraphrase-induced divergence appear, whether the reseed pass at `temperature=0.7` produces meaningfully different readings than at the default `temperature≈0` — land in a follow-up journal entry written after the opt-in smoke runs against the real API.

### Out of scope (preserved from the plan)

Phase 11 (the consumption layer that wires per-surface stability into per-criterion verdicts); Probe 4 real-builder-perturbation infrastructure; any change to Io, the actor, the world model, the dream state, or the runner; changes to Phase 7's frozen criteria prose; automatic prompt-paraphrase generation (the synthesis is explicit: variants are checked in, not generated); per-claim adaptive thresholding (per-surface is the granularity); the agreement metric beyond Jaccard on structured-field tuples; the per-claim `paraphrase_stability` / `reseed_stability` wire-through on `StructuredClaim` (Phase 10 produces the per-surface scores; the per-claim values are deferred).

Phase 10 touched
`kind/mirror/stability.py` (new),
`kind/mirror/prompt_builder.py` (extended — `framing_override` kwarg threaded through three per-criterion builders),
`kind/mirror/llm_caller.py` (extended — `LLMConfig.temperature` + `LLMConfig.seed` fields with validators; `call_mirror_llm` `seed` / `temperature` kwargs; `MockLLMClient.configs` property; `_GeminiLLMClient.generate_batch` generation_config extended),
`kind/mirror/__init__.py` (re-exports extended),
`tests/test_stability.py` (new),
`tests/test_phase_10_stability_smoke_real_api.py` (new, opt-in),
`tests/test_prompt_builder.py` (extended — three framing-override tests),
`tests/test_llm_caller.py` (extended — six seed/temperature/configs tests),
the journal index entry, and this entry. No other files. No changes to `kind/observer/schemas.py`, no changes to `kind/observer/pre_reg.py`, no changes to Phase 7's frozen criteria prose, no schema version bump, no on-disk Phase 13 or Phase 9 artifact touched. The Phase 8 verbatim-clause tests and the Phase 9 module-level-identity check both continue to pass byte-for-byte.

---

## Phase 10 — stability smoke findings (real Gemini API run, 2026-05-18)

The Phase 10 stability smoke ran end-to-end against `gemini-2.5-pro` on the `reflexive_attention` criterion (single surface — `HEAD_INTERNAL`) for the primary (advocate) role. Six LLM calls — 3 paraphrases + 3 reseeds — 119 s wallclock (~20 s per call), 0 retries, 0 failures. The on-disk artifact is at `runs/phase_10_stability_smoke/stability.jsonl` (14.4 KB; one JSONL line carrying the full `StabilityResult`; round-trips through `model_validate_json(model_dump_json(...))` per the smoke test's assertion).

This is the project's first stability measurement against the real Gemini API. The headline result is admissible=False at `HEAD_INTERNAL` — paraphrase agreement 0.333 well below the 0.80 threshold, reseed agreement 1.000 perfectly clearing it. The cause is not what the substrate journal predicted; the metric is too sharp on `cited_scalar_field` string-formatting variation, not on `cited_value` floating-point noise. The full diagnosis below.

### The stability result

```
criterion_id:       reflexive_attention
reader_role:        primary
checkpoint_id:      phase_10_smoke_ckpt
n_paraphrases:      3
n_reseeds:          3
wallclock_ms:       118 888 (~119 s)

paraphrase_agreement_per_surface: { head_internal: 0.3333 }
reseed_agreement_per_surface:     { head_internal: 1.0000 }
admissible_per_surface:           { head_internal: False }

structured_field_agreement_per_claim:
  [0.3333, 0.6667, 0.3333, 0.6667]   # four unique tuples; two at 1/3, two at 2/3
```

The verdict: **not admissible at `HEAD_INTERNAL`** for this criterion under the synthesis defaults. Paraphrase agreement fails the 0.80 threshold; reseed agreement clears it perfectly. Per Phase 10's admissibility rule (both axes must clear), the conjunction is False.

### Per-paraphrase claim inspection

Each paraphrase call produced one reading with two claims. The interesting axis is `cited_scalar_field` — `cited_value` was identical across all six calls (0.42 and 0.51, transcribed exactly from the fixture stats). The `cited_step_range` was `None` on every claim (the fixture's signals don't carry step ranges; the LLM correctly used `None`).

**Paraphrase 0** (framing variant: *"Read the head-internal signals. What is the pattern?"*) — claims used the **compound** form:
- `('latent_self_reference_t.partial_autocorr_lag5', 0.42)`
- `('dream_self_reference_t.autocorr_lag5', 0.51)`

The LLM composed `<signal_name>.<estimator>` as the cited_scalar_field. Claim text: *"The waking-state latent signal (h_t) shows a partial autocorrelation at lag 5 of 0.42, which is significantly greater than its shuffled-time control"* (paraphrase 0, claim 0).

**Paraphrase 1** (framing variant: *"Inspect the head-internal latent structure. What does it reveal?"*) — claims used the **bare** form:
- `('latent_self_reference_t', 0.42)`
- `('dream_self_reference_t', 0.51)`

The LLM used just the signal name. Claim text: *"The agent's internal state (h_t) during 'waking' operation exhibits significant temporal self-reference, with a partial autocorrelation at lag 5 of 0.42"* (paraphrase 1, claim 0). Same substantive claim, different transcription style.

**Paraphrase 2** (framing variant: *"Consider the head-internal measurements. What do they indicate?"*) — claims used the **compound** form:
- `('latent_self_reference_t.partial_autocorr_lag5', 0.42)`
- `('dream_self_reference_t.autocorr_lag5', 0.51)`

Same form as paraphrase 0.

**Reseed 0, 1, 2** (default framing, seeds 1000 / 1001 / 1002, temperature 0.7) — all three used the **bare** form, identical tuples:
- `('latent_self_reference_t', 0.42)`
- `('dream_self_reference_t', 0.51)`

The reseed pass at the default framing converged on the bare form across all three seeds. The free-text-notes wording varied across reseeds (different phrasing of the same substantive verdict) but the structured fields did not.

### Pairwise Jaccard math, verified

The 0.333 paraphrase agreement is the mean of the three pairwise comparisons at `HEAD_INTERNAL`:

```
Pair (0, 1): |A∩B|=0, |A∪B|=4, Jaccard = 0.0    # compound vs bare, no tuples shared
Pair (0, 2): |A∩B|=2, |A∪B|=2, Jaccard = 1.0    # both compound, identical
Pair (1, 2): |A∩B|=0, |A∪B|=4, Jaccard = 0.0    # bare vs compound, no tuples shared
Mean: (0.0 + 1.0 + 0.0) / 3 = 0.3333
```

The 1.0 reseed agreement is the mean of three pairs, each Jaccard 1.0 — all three reseed readings produced byte-identical claim tuples. The per-claim agreement tuple `[0.333, 0.667, 0.333, 0.667]` accounts for the four unique tuples across the three paraphrase readings: the bare-form pair appeared in 1/3 paraphrases (paraphrase 1), the compound-form pair in 2/3 (paraphrases 0 and 2).

### Threshold gating

The synthesis-default `HEAD_INTERNAL` paraphrase threshold is 0.80. The smoke produced 0.333. The gating outcome is **reject as not admissible** — and the smoke's empirical case is the first signal on whether the default is well-tuned.

The 0.80 default would clear if the LLM's variation were small (e.g. paraphrase-induced choice between two near-synonymous compound forms producing pairs of ~0.5–0.8). What happened instead is binary: the LLM picked one of two formatting conventions per paraphrase, and the conventions are disjoint under exact-match Jaccard. With 3 paraphrases split 2-vs-1 across conventions, the agreement is *exactly* 1/3 — independent of how many tuples each reading contains, independent of whether the underlying claims agree semantically.

The 0.80 threshold is not the problem. The metric is the problem. Lowering the threshold to 0.30 to admit this case would mask the actual finding — that the LLM produces formatting-level disagreement on a stable input — and would also admit cases where the LLM disagrees on substance. Phase 10 commits no amendment; the journal records the case.

### Jaccard metric assessment: the metric is too sharp on scalar-field strings, but correct on cited_value

The metric's sharpness failed in the direction the substrate journal did *not* predict. Substrate journal's "what surprised" #2 named the floating-point-noise concern: "two LLM-generated claims about the same value could disagree by floating-point noise." The smoke refutes this concern empirically. `cited_value` was 0.42 and 0.51 across all six readings, byte-identical. The structured-output schema *did* have the LLM transcribe values from the prompt's `Computed signals:` block; the prediction held.

What the substrate journal missed: the `cited_scalar_field` is not transcribed from any single prompt field. The prompt presents `- signal: latent_self_reference_t / estimator: partial_autocorr_lag5` as two separate fields, and the LLM is free to construct the `cited_scalar_field` in either of two reasonable forms:

- `latent_self_reference_t` (just the signal name)
- `latent_self_reference_t.partial_autocorr_lag5` (compound `<signal>.<estimator>`)

Both are equally faithful to the prompt; both are useful in downstream consumption. Jaccard treats them as disjoint, and the agreement collapses to 1/3 even though the underlying claims agree perfectly on which signal, which estimator, and which value.

This is a "metric is too sharp on string-formatting choice" failure mode. The signs are unambiguous in the smoke result:

1. Both formatting variants carry the *same* `cited_value` per claim (0.42 or 0.51). If the LLM disagreed on substance, the values would differ.
2. The reseed pass — same prompt, same statistic, varied seed — converged on a single form (bare) across all three calls. Formatting choice is prompt-framing-dependent, not random.
3. The free-text-notes substantively agreed across all six readings: every reading concluded that reflexive attention is satisfied because both waking and dream self-reference signals exceed their controls.

A canonical-form normalizer at the metric layer — e.g. *"if the LLM produces both bare and compound forms, treat them as the same tuple if the signal name prefix matches and the cited_value matches"* — would let the smoke's substantive agreement surface as a high Jaccard. A richer metric is a Phase 11+ amendment, journaled here as the first empirical case.

The metric was not too loose anywhere in the smoke: no case where two semantically different readings counted as the same tuple. The sharpness failure is one-directional.

### Paraphrase variation

The three committed variants for `HEAD_INTERNAL` produced two distinct LLM-side formatting choices, not three. Variant 1 (*"Inspect the head-internal latent structure. What does it reveal?"*) elicited the bare form; variants 0 and 2 elicited the compound form. So the variants are not clustered (3-way tie would be Jaccard 1.0), nor wildly different (the cited_values agree; the free-text substantively agrees); they're split 2-vs-1 on a cosmetic axis the LLM treats as prompt-dependent.

The variants are doing the work of stability-testing in an unexpected way: they're surfacing the LLM's *formatting sensitivity to prompt framing*. A future amendment might either:

- accept this as a useful signal (formatting-stability *is* a kind of stability worth measuring), and lower the threshold to admit 2/3-or-better agreement;
- relax the metric to canonical-form match, in which case the variants would all collapse to identical tuples and paraphrase agreement would jump to 1.0;
- replace one of the three variants with a fourth that disambiguates the formatting choice in the prompt — e.g. instructing the LLM to use `<signal>` (bare) form for cited_scalar_field. This would converge all three paraphrases to the bare form (matching the reseed pass's convergence).

Phase 10 commits no amendment; the journal records the three options.

### Reseed variation at temperature=0.7: the reseed pass was a null

The synthesis chose `temperature=0.7` for reseeds expecting meaningful structured-output variation across seeds. The smoke shows **zero structured variation**: all three reseeds at seeds 1000, 1001, 1002 produced byte-identical `cited_scalar_field` / `cited_value` tuples. The reseed pass at this temperature is approximately a no-op for the structured-output channel; only the free-text notes vary in wording.

Two diagnoses are consistent with this finding:

1. **Gemini's structured-output path constrains the LLM tightly.** When the response schema asks for specific fields, the SDK's structured-output enforcement may suppress most temperature-driven variation. Temperature continues to perturb the free-text fields where the schema doesn't pin a structure.
2. **Temperature 0.7 isn't high enough to push structured choices around.** A temperature of 1.0 or 1.5 might surface meaningful variation. But increasing temperature past 0.7 starts to risk schema-violation failures (more output that doesn't parse).

Either diagnosis points to the same operational conclusion: **for the structured-output channel, the reseed test is mostly redundant with the paraphrase test.** Paraphrase variation is doing the work; reseed variation at temperature 0.7 isn't adding signal beyond what paraphrase already surfaces. The free-text-notes pass — which Phase 10 doesn't gate on — *is* receiving the temperature signal, but it's not what admissibility is measuring.

The synthesis's two-axis design (paraphrase + reseed) may be over-specified for this LLM family. Or the reseed pass may need to surface free-text-notes variation as a separate signal (paraphrase agreement on claims; reseed agreement on free-text). Both are Phase 11+ amendments.

### LLM call audit summary

```
calls:       6 total (3 paraphrase + 3 reseed)
retries:     0
failures:    0
wallclock:   118.9 s end-to-end
per-call:    ~19.8 s mean
record_sink: not threaded (the smoke didn't wire one)
tokens:      not surfaced (Phase 12 §2 token-thread-through still open)
```

No exceptions, no retries, no schema-validation failures. The Gemini structured-output schema for `BatchPayload` ingested the single-fragment payload cleanly across all six calls (the Phase 12 munger that converts `prefixItems → items` and inlines `$ref` was active and produced no diagnostics). Token counts and cost are not directly observable from the result (the smoke didn't thread an `LLMRecordSink`; Phase 12's outstanding token-threading is the unblocker for cost). A rough estimate from typical Gemini 2.5 Pro pricing × 6 calls is ~$0.05 — comfortably under the $0.30 estimate the substrate journal cited.

The wallclock 119 s is consistent with Gemini 2.5 Pro's typical structured-output latency (~15–25 s per call for a payload of this size). The opt-in-smoke profile is sustainable: a Phase 11+ scale-up to all three criteria × both roles × multiple checkpoints would be 6 calls × 2 roles × 3 criteria × N checkpoints — about 10 minutes per checkpoint per criterion-role pair, manageable but not free.

### What surprised

**The substrate journal's "what surprised" #2 was wrong in an instructive way.** It predicted that float exact-match would be robust because the LLM transcribes from the prompt. The transcription premise held — `cited_value` was perfect — but the conclusion was wrong because `cited_scalar_field` is not transcribed from a single prompt field. The LLM composes it. The prediction was correct about the substrate-of-transcription and wrong about the resulting Jaccard robustness. The metric's sharpness lives at a different layer than the journal located it: not in float comparison, but in string-construction choice for compound-name fields.

**The reseed pass produced zero structured variation.** The expectation was that temperature 0.7 would surface seeded variation in the structured output. The reality is that the structured-output schema is essentially deterministic given a fixed prompt, even at temperature 0.7. The free-text-notes vary across reseeds (different word choices, same substance); the claim tuples do not. This pushes the reseed test toward being a redundant signal alongside the paraphrase test — at least for this LLM family and this prompt scale.

**The paraphrase variation surfaced at the formatting layer, not the substance layer.** The three variants didn't elicit three different *answers* (every paraphrase concluded reflexive attention is satisfied); they elicited two different *formatting conventions* (bare vs compound for the cited_scalar_field). The variants are conservative on purpose — they vary surrounding prose, not what-to-look-for — and the LLM responded by varying its prose-shaped output (the format of the field name string) without varying the substance. This is consistent with what the variants were *designed* to test (formatting/framing stability) but the failure mode lives in the metric, not the variants.

### What's now closed

The substrate journal named six newly-open items; the smoke closes or substantially advances five of them. Resolution per item:

1. **(§1) Whether the per-surface thresholds 0.80 / 0.80 / 0.75 are well-tuned in the Gemini × prompt-fragment combination.** Partially answered: the 0.80 `HEAD_INTERNAL` threshold rejected the smoke's reading. The threshold itself is not the load-bearing problem; the metric's sharpness on cited_scalar_field string formatting is. *Closed for `reflexive_attention` × `HEAD_INTERNAL`; thresholds for `SUBSTRATE_SIDE` and `BEHAVIOR_SIDE` remain open (those surfaces weren't exercised — the smoke only ran the single-surface criterion).*
2. **(§3) Whether Jaccard on `(cited_step_range, cited_scalar_field, cited_value)` tuples is sharp enough.** **Closed: the metric is too sharp on cited_scalar_field string variation.** The float-noise concern was empirically refuted; the string-formatting concern is the actual failure mode. Three remediation candidates surfaced (see "Paraphrase variation" section). All three are Phase 11+ amendments.
3. **(§4) Whether the three paraphrase variants per surface are the right cardinality.** **Partially closed: 3 surfaced a 2-vs-1 split on a formatting axis.** A larger N (5 or 7) wouldn't help because the failure mode is bimodal (the LLM picks one of two formatting conventions deterministically per framing variant); more variants would just sample the binary choice more times. The variant cardinality is right; the variants' *content* is the open question — should one of the three disambiguate the formatting choice in the prompt? Closed enough to advance the design discussion.
4. **(§5) Whether the "anchor reading" decision is structurally right.** **Closed: the no-anchor decision is structurally right.** The smoke's pairwise-Jaccard agreement math is well-defined without an anchor; the reseed pass at default framing functioned exactly as the anchor would have (and converged on a single form). Adding an anchor would have given a 13th LLM call producing the same bare-form output as the three reseeds — no new information.
5. **(§6) The Phase 10 stability smoke's substantive findings.** **Closed: this entry is the resolution.**

The remaining substrate-journal newly-open item:

2. (§2) **How per-surface stability scores feed into per-criterion verdicts.** Unchanged. Phase 11+ work.

### What's now newly open

The smoke surfaces five new questions Phase 11+ will need to address:

1. **Whether the cited_scalar_field naming convention should be disambiguated in the prompt.** The smoke surfaced two equally-faithful LLM choices (bare vs compound `<signal>.<estimator>`). The most surgical fix is to add a sentence to the prompt-builder's framing instructing the LLM to use the bare signal name as cited_scalar_field. This is a prompt-builder change, not a stability-runner change. Journaled as a candidate for Phase 11 prompt-builder amendment; not committed here.
2. **Whether the metric should normalize cited_scalar_field to a canonical form.** Alternative to (1) above. A canonical-form normalizer in `_claim_tuples_at_surface` would map `'latent_self_reference_t.partial_autocorr_lag5'` and `'latent_self_reference_t'` to the same canonical form when the prefix matches and the cited_value agrees. This is a metric change, not a prompt change. Trade-offs against (1): canonicalization at the metric layer is more permissive (admits more readings as agreeing) but loses information about the LLM's actual transcription choice. Open.
3. **Whether the reseed test should be retired or replaced for the structured-output channel.** The smoke shows zero structured-field variation across temperature-0.7 reseeds. If this pattern holds across criteria and roles, the reseed test as currently specified is redundant with the paraphrase test. Three candidates: (a) raise temperature to 1.0+ to force structured variation (with risk of schema failures); (b) keep the reseed test but score it on free-text-notes similarity, not on structured-field Jaccard; (c) retire the reseed test entirely for Gemini-family LLMs and journal the LLM-specific choice. Open.
4. **Whether single-surface criteria need a different stability protocol than multi-surface criteria.** `reflexive_attention` declares only `HEAD_INTERNAL`; the smoke exercised one surface with 3 paraphrases. For multi-surface criteria (`equanimity_perturbation_recovery`: substrate + behavior; `second_order_volition`: substrate + behavior), the paraphrase pass becomes 6 calls (3 variants × 2 surfaces). Whether the cross-surface comparison adds signal — or whether the single-criterion-multiple-surfaces structure is more sensitive to the same string-formatting failure mode — is empirically open. The smoke didn't test it.
5. **Whether the findings from `reflexive_attention` generalize to the other two v2 criteria.** `equanimity_perturbation_recovery` and `second_order_volition` have different statistic shapes (the equanimity prompt carries a perturbation-timeline block; the second-order-volition prompt carries the four verbatim exclusions). The cited_scalar_field naming choice may interact differently with those prompt shapes. The smoke ran only `reflexive_attention`; the other two criteria's stability behavior is open.

### Out of scope (preserved from the substrate-phase plan)

Phase 11 (the consumption layer that wires per-surface stability into per-criterion verdicts); Probe 4 real-builder-perturbation infrastructure; any change to Io, the actor, the world model, the dream state, or the runner; changes to Phase 7's frozen criteria prose; changes to the committed paraphrase variants (any tune is a journaled amendment, not a silent edit); changes to the synthesis-default thresholds. The findings entry is journal-side observation; the code changes the smoke surfaces are deferred to Phase 11+.

Phase 10 smoke touched no code. The on-disk artifact at `runs/phase_10_stability_smoke/stability.jsonl` (14.4 KB; well-formed `StabilityResult`; round-trips through `model_validate_json(model_dump_json(...))`) is the load-bearing record; the pytest-tmp source copy at `runs/phase_10_stability_smoke/pytest_tmp/test_phase_10_stability_smoke_0/stability.jsonl` is the canonical pytest output the smoke wrote. The 990-test suite continues to pass; the Phase 10 substrate tests pass unchanged; the Phase 8 verbatim-clause tests and the Phase 9 module-level-identity check pass unchanged.

---

## Phase 11 — faithfulness verifier (2026-05-18)

Phase 11 builds the second half of the admissibility-gate substrate the synthesis named. Phase 10 measures *stability under variation* (paraphrases, reseeds); Phase 11 measures *faithfulness to the source data* (do the LLM's citations actually trace back to the statistic results it was shown?). The two verifiers run independently and emit independent records; a future admissibility consumer joins them.

The build is structurally smaller than Phase 9 or Phase 10 — one new module, one verifier function, two frozen result records, no LLM calls. The work is in the resolution logic — specifically, the canonical-form match that Phase 10's smoke surfaced as necessary plus the dict-key-suffix dispatch the Phase 13 production data demanded.

### What was built

**`kind/mirror/faithfulness.py` (new, 692 lines).** The single Phase 11 module. Two module-level constants, one enum, two frozen result records, one canonicalization function, one verifier function. The surfaces:

- `FAITHFULNESS_THRESHOLD: Final[float] = 0.80` — per-reading admissibility threshold on the faithfulness rate. The synthesis-default convention; a future tune is journaled and single-source.
- `FAITHFULNESS_VALUE_TOLERANCE: Final[float] = 1e-6` — float-comparison tolerance for resolving `cited_value` against a statistic result. Applies to both scalar comparisons and list-membership.
- `FaithfulnessStatus(str, Enum)` — four members: `RESOLVED`, `UNRESOLVED_FIELD`, `UNRESOLVED_VALUE`, `UNRESOLVED_RANGE`. Distinct from the writer-side `kind.mirror.structured.FaithfulnessStatus` Literal — the verifier verdict is more specific (it names *why* an unresolved citation didn't resolve), which is what a downstream admissibility consumer needs.
- `FaithfulnessAssignment(BaseModel)` — per-claim record (frozen, `extra="forbid"`). Carries `pass_index`, `criterion_id`, `reader_role`, `claim_index`, `cited_scalar_field_original` (verbatim from the LLM), `cited_scalar_field_canonical` (after normalization), `cited_value`, `cited_step_range`, `status`, `resolution_notes` (non-empty validator).
- `FaithfulnessResult(BaseModel)` — per-reading record (frozen, `extra="forbid"`). Carries the criterion/role/run envelope, the assignments tuple, the four count fields, the `faithfulness_rate` (in `[0, 1]`), the `admissible` boolean, `wallclock_ms`, and a `notes` field. Two validators: `_validate_counts_sum` enforces that the four count fields sum to `n_claims_total` and that `len(assignments) == n_claims_total`; range / non-empty validators on the scalar fields. Vacuous case: a reading with zero claims has `faithfulness_rate=1.0` and `admissible=True`.
- `canonicalize_scalar_field(field: str) -> str` — the canonicalization rule. Strips one trailing `.<suffix>` token when present; bare forms (no dot) are unchanged; multiple dots strip only the last suffix (`"a.b.c"` → `"a.b"`); empty input raises `ValueError`.
- `verify_reading(reading, statistic_results, *, criterion_id, pass_index, run_id, checkpoint_id, audit_jsonl_path=None) -> FaithfulnessResult` — the verifier. Iterates claims, computes the canonical form, looks up the matching statistic by `signal_name`, dispatches on the statistic value's runtime type (`float` / `list[float]` / `dict[str, float]`), assembles the result, optionally appends a JSONL audit line. Pure aside from the optional audit emission; no LLM calls; no statistic recomputation; no telemetry reads.

The verifier's dispatch on value type:

- **Scalar (`float`)** — `abs(diff) < FAITHFULNESS_VALUE_TOLERANCE` against `cited_value`. Hit → `RESOLVED`; miss → `UNRESOLVED_VALUE`.
- **List (`list[float]`)** — if `cited_step_range` is `None`, membership across the whole list. If `cited_step_range = (start, end)` falls within `[0, len-1]`, the verifier extracts `statistic_value[start:end+1]` and checks membership. If the range falls outside the list's index range → `UNRESOLVED_RANGE`.
- **Dict (`dict[str, float]`)** — if the original cited_scalar_field was compound (e.g. `"policy_entropy_t.no_response"`), the suffix is treated as a dict key; the verifier looks up `dict[suffix]` and compares to `cited_value`. If no suffix, the verifier falls back to membership across the dict's values.

**`kind/mirror/__init__.py` (extended).** Re-exports `FAITHFULNESS_THRESHOLD`, `FAITHFULNESS_VALUE_TOLERANCE`, `FaithfulnessAssignment`, `FaithfulnessResult`, `FaithfulnessStatus`, `canonicalize_scalar_field`, `verify_reading` under a "Phase 11 — faithfulness verifier" section.

**`tests/test_faithfulness.py` (new, 676 lines, 26 tests).** Structural and verifier coverage:

- `canonicalize_scalar_field` cases (bare unchanged, compound stripped, empty raises, multi-dot strips only the last suffix);
- `FaithfulnessStatus` enum has exactly four members;
- `FaithfulnessAssignment` and `FaithfulnessResult` frozen + extra-forbid invariants;
- `FaithfulnessAssignment.resolution_notes` non-empty validator;
- `FaithfulnessResult` counts-sum-to-`n_claims_total` model validator;
- `FAITHFULNESS_THRESHOLD == 0.80` and `FAITHFULNESS_VALUE_TOLERANCE == 1e-6` module-constant pins;
- `verify_reading` on the four outcomes (resolved float, unresolved field with the available signal names named in the notes, unresolved value, unresolved range);
- the canonical-form-match case from the Phase 10 smoke (compound `latent_self_reference_t.partial_autocorr_lag5` against bare `signal_name`);
- the dict-key-suffix case from the Phase 13 production data (compound `policy_entropy_t.no_response` against `dict[str, float]` statistic value);
- the list-with-step-range resolved case and the list-with-step-range out-of-bounds case;
- the LLM-prefill-overwrite contract (the LLM pre-fills `faithfulness_status="resolved"` on a claim that doesn't actually resolve; the verifier's verdict is `UNRESOLVED_FIELD`);
- the admissibility-at-threshold boundary (rate == 0.80 → admissible True; rate < 0.80 → False);
- the audit-JSONL emission test (two calls produce two lines; the second-line round-trip rehydrates back to a `FaithfulnessResult`; the enum value is serialized as its string form);
- the input-immutability invariant (`reading` and `statistic_results` unchanged; the LLM's pre-filled `faithfulness_status` on the source claim is preserved);
- the empty-claims-list vacuous-admissibility case;
- `pass_index` propagation through to the per-claim assignment;
- the `reader_role` mapping ("skeptic" → "adversarial").

**`tests/test_phase_11_faithfulness_smoke_real_data.py` (new, 188 lines, 1 test).** The Part 4 data-shapes smoke. Reads `runs/phase_13_calibration/mirror/rounds/phase_13_probe_1_round.json`, picks the first pass with readings, rehydrates the first reading and the pass's statistic results into Pydantic models, runs `verify_reading`, asserts the result is well-formed (counts sum, rate in `[0, 1]`, `len(assignments) == n_claims_total`, every assignment carries a non-empty `resolution_notes` and a recognized status, the source reading is unchanged). Skipped when the on-disk round result isn't present (allowing fresh-checkout `pytest` to pass without the Phase 13 calibration artifacts).

### Reconciliation with Phase 10's findings (the two amendments Phase 11 commits)

Phase 10's stability smoke (2026-05-18) named two LLM behaviors Phase 11 had to handle. The reconciliation:

**The canonical-form rule (closes Phase 10's newly-open §2).** Phase 10's smoke saw the LLM emit the same evidential citation in two equally-faithful forms — `latent_self_reference_t` (bare) and `latent_self_reference_t.partial_autocorr_lag5` (compound with estimator-name suffix) — across the three paraphrase variants. The Jaccard metric on `(cited_step_range, cited_scalar_field, cited_value)` tuples read 0.333 paraphrase agreement on `HEAD_INTERNAL`, well below the 0.80 threshold. Phase 11 commits the canonical form at the verifier layer: `canonicalize_scalar_field` strips one trailing `.<suffix>` token; the verifier looks up the statistic by the canonical form. The same compound vs bare ambiguity that tripped Phase 10's stability metric is now a resolved citation at Phase 11's verifier — the structural fix lifts cleanly into the faithfulness layer because the question is "does this citation trace back to the data?", not "do two readings agree?". Phase 10's stability metric is not amended in Phase 11; lifting the canonical-form normalizer into a shared helper that Phase 10's `_claim_tuples_at_surface` can also call is open work (see newly-open §1 below).

**The LLM-prefill-overwrite decision.** Phase 10's smoke saw the LLM pre-fill `faithfulness_status="resolved"` on 4 of 6 readings before any verifier ran. The Phase 0 `StructuredClaim` schema declares `faithfulness_status` as a Literal whose writer-side default is `"not_checked"`; the LLM should not have been writing into this field at all. Phase 11 commits option A from the build's pre-design discussion: **the verifier owns the verdict.** The `FaithfulnessAssignment.status` field is constructed by `verify_reading` from the citation resolution; whatever the LLM wrote into the source claim's `faithfulness_status` is discarded by the verifier (the `test_verify_reading_overwrites_llm_prefill` test pins this). The complementary prompt-builder amendment — instructing the LLM not to write into the field at all — is deferred to a follow-up phase and journaled below in newly-open §2; it's a different layer than the verifier and the verifier's contract works regardless of whether the LLM continues to pre-fill.

### Reconciliation with the synthesis's sequencing

The synthesis named "faithfulness verifier before stability runner" in §1's prelude. The as-built order is Phase 9 → Phase 10 → Phase 11. The reconciliation lives at the consumption level: Phase 11's verdicts and Phase 10's stability scores are independent producers that a future admissibility consumer will combine. Phase 11 doesn't gate Phase 10; the two verifiers run on the same readings and produce independent verdicts at different granularities (per-claim vs per-surface).

### The threshold commitment

`FAITHFULNESS_THRESHOLD = 0.80`. Three forces pushed for this value:

- **The synthesis convention.** Phase 10's `PARAPHRASE_THRESHOLDS` and `RESEED_THRESHOLDS` are 0.80 at substrate-side and head-internal. The synthesis names 0.80 as the default per-surface admissibility threshold; Phase 11 inherits the convention at a different granularity (per-reading, not per-surface).
- **Faithfulness is stricter than stability.** A reading where 20% of citations don't resolve is genuinely suspect: a fabricated citation is a different kind of failure than a paraphrase-drift. A looser threshold (0.70) would admit readings where nearly a third of the LLM's evidence claims don't trace back to the prompt's signals block. That's too loose.
- **The empirical signal from the Part 4 smoke.** Running the verifier across all 5 passes of both Phase 13 round results: probe_1 round → 17/25 claims resolved (rate 0.68; not admissible at 0.80); probe_1.5 round → 25/25 resolved (rate 1.00; admissible). The 0.80 threshold cleanly separates the two rounds; a 0.70 threshold would also pass probe_1; a 0.90 threshold would still reject probe_1 but tighten things on probe_1.5 if a single claim drifted. 0.80 is the convention's commitment and the data agrees.

### Test / mypy status

```
.venv/bin/mypy --strict kind/mirror/ kind/observer/
Success: no issues found in 35 source files

.venv/bin/pytest tests/ --ignore=tests/test_*_real_api.py
1016 passed, 1 failed (pre-existing test_transport flake on the
threading barrier — passes in isolation), 2 skipped, 1 warning in 52 s
```

The 4 real-API marks (Phase 9 judge smoke, Phase 10 stability smoke, Phase 12 smoke, Phase 13 calibration) are unchanged and not run in this gate; the Phase 11 Part 4 smoke runs as part of the standard suite (no `--run-real-api`) because it issues no LLM calls. The Phase 8 verbatim-clause tests, the Phase 9 module-level-identity check, and the Phase 10 verbatim-clause protection test continue to pass byte-for-byte unchanged.

### What surprised

The Part 4 data-shapes smoke surfaced three failure modes that the suffix-stripping canonicalization rule alone does not handle. Running the verifier across all 5 passes × all readings of both Phase 13 round results (probe_1: 25 claims across 4 readings, 4 passes-with-readings; probe_1.5: 25 claims across the same shape) produced **17 / 25 resolved on probe_1 (rate 0.68; admissible False)** and **25 / 25 resolved on probe_1.5 (rate 1.00; admissible True)** — the 0.80 threshold cleanly separates the two rounds. The 8 unresolved claims on probe_1 partition into three patterns:

**Pattern 1 — multi-suffix forms.** The LLM emits `policy_entropy_t.classification.collapse` (two dots) and `posterior_kl_t.classification.no_response` against a dict-valued statistic with bare `signal_name = "policy_entropy_t"`. The canonical-form rule strips one suffix → `policy_entropy_t.classification`, which does not match any statistic's `signal_name` → verdict is `UNRESOLVED_FIELD`. An iterated canonicalization (strip-until-bare-form-matches) would resolve these. Phase 11 deliberately deferred the iterated rule because the empirical case wasn't in hand at design time; the smoke surfaces 2 claims with this pattern. *Phase 12+ amendment with the empirical case in hand.*

**Pattern 2 — invented dict-key suffixes.** The LLM emits `recovery_lag_steps.recovered_within_W_count`, `policy_entropy_t.no_response_count`, `posterior_kl_t.ratchet_count` — compound forms where the suffix is a *derived count* the LLM computed (number of perturbations classified as `no_response`), not an actual dict key in the statistic value (the actual keys are `dip_and_recover`, `collapse`, `stays_elevated`, `no_response`). The canonical form correctly strips the suffix and the dispatch correctly identifies the dict-valued statistic, but the dict-key lookup fails. Verdict is `UNRESOLVED_VALUE` with notes naming the actual available keys. This is the LLM doing semi-arithmetic and citing the result as if it were a direct field; the verifier correctly rejects it. *The prompt-builder amendment that names the actual dict keys explicitly is a candidate for follow-up; the verifier's behavior is correct.*

**Pattern 3 — list[float] indexed by perturbation, not by step.** The `recovery_lag_steps` statistic returns `list[float]` where each element is one perturbation's recovery lag (per-perturbation, not per-step). On probe_1's checkpoint with 2 perturbations the list has length 2. The LLM cites `cited_step_range=(2147, 2197)` (absolute step indices in the run) against a length-2 per-perturbation list — verdict is `UNRESOLVED_RANGE` because 2147 falls outside `[0, 1]`. The mismatch is semantic: the verifier dispatches on `list[float]` as if it were a step-indexed trajectory; the statistic's `notes` field would name the per-perturbation indexing convention but the verifier doesn't read notes. *Either the verifier learns to dispatch per-statistic on whether `list[float]` is per-step or per-perturbation (a per-statistic-type table is the simplest version), or the statistic's value shape is amended to make the indexing explicit (e.g. a `list[tuple[int, float]]` carrying the perturbation-index alongside the lag), or both. This is the largest of the three findings.*

The substrate journal had predicted probe_1 readings would resolve cleanly (the substrate journal entry for Phase 13 noted: "the LLM transcribes the prompt's signals block; faithfulness should be high"). The 68% rate is not a regression but a more nuanced finding: **the LLM transcribes the prompt's signals block faithfully for the simple float-valued signals (latent_self_reference_t, dream_self_reference_t — 100% resolved across all 4 passes); the unfaithfulness concentrates on the equanimity criterion's dict-valued and list-valued statistics where the cited_scalar_field has more degrees of freedom in how the LLM names its evidence.** Two of the three patterns are addressable by either prompt-builder amendments (Pattern 2: name the actual dict keys) or by iterated canonicalization (Pattern 1); the third (Pattern 3) is a deeper statistic-shape concern.

The 0.80 threshold separating the two rounds is the second surprise. The substrate journal had not predicted that probe_1.5's readings would resolve at 100% while probe_1's would land at 68%. The Phase 13 calibration smoke entry would not have flagged this; the verifier's per-claim resolution is the granularity at which the difference is visible.

### What's now closed

- **Phase 10's newly-open §2 — "Whether the metric should normalize cited_scalar_field to a canonical form."** Closed in the faithfulness layer. Phase 11's `canonicalize_scalar_field` is the committed normalizer; the verifier uses it for citation resolution. The complementary question of whether Phase 10's stability metric should also use it (i.e. whether `_claim_tuples_at_surface` should canonicalize before tuple-extraction) is open work — see newly-open §1 below.
- **The "the LLM is writing into `faithfulness_status` before any verifier runs" question.** Phase 11 commits the verifier-owns-the-verdict contract; the verifier's output is the canonical record regardless of what the LLM wrote into the field. The complementary prompt-builder amendment (instruct the LLM not to write there) is journaled and deferred.
- **The faithfulness verifier as a substrate phase.** The synthesis named the verifier alongside the stability runner; both are now built. The admissibility consumer that joins them is the next phase the substrate needs.

### What's now newly open

1. **Whether Phase 10's stability metric should be amended to use the canonical-form match.** Phase 10's `_claim_tuples_at_surface` extracts `(cited_step_range, cited_scalar_field, cited_value)` tuples for Jaccard; the cited_scalar_field is the verbatim LLM string. Phase 11's `canonicalize_scalar_field` would resolve Phase 10's bare-vs-compound divergence at the stability layer too (paraphrase agreement on `HEAD_INTERNAL` would have read >> 0.333 instead of 0.333 if the canonical form had been used). The factoring: lift `canonicalize_scalar_field` into a shared helper module both `stability.py` and `faithfulness.py` import. The trade-off: the stability layer is meant to measure *the LLM's actual transcription consistency*; canonicalizing the input to the metric arguably hides genuine variation that a different prompt-builder convention might prefer to surface. *Open. A future Phase 11.5 or Phase 12 amendment with the empirical case in hand decides.*
2. **The prompt-builder amendment to remove the LLM's `faithfulness_status` write surface.** Phase 0's `StructuredClaim` schema declares the field as a Literal whose writer-side default is `"not_checked"`; the LLM should not write there. Three candidates: (a) remove the field from the structured-output JSON schema the LLM sees (Pydantic's `model_json_schema()` always emits all fields; the surgical fix is at the `_to_gemini_schema` munger); (b) add a system-prompt sentence instructing the LLM to leave the field at `"not_checked"`; (c) keep the LLM's pre-fill as informational and let the verifier overwrite (the current Phase 11 contract). Phase 11 commits (c) for the verifier's correctness; (a) or (b) would let the per-claim record carry a cleaner pre-verifier state. *Open; a Phase 11.5 prompt-builder amendment.*
3. **How Phase 11's faithfulness verdicts and Phase 10's stability scores join into a single admissibility verdict.** Phase 11's `admissible` is per-reading; Phase 10's `admissible_per_surface` is per-surface. The synthesis names both verifiers feeding a future admissibility consumer. Three candidates: (a) AND-conjunction (a reading is admissible iff faithfulness clears and every declared surface's stability clears); (b) per-surface faithfulness (recompute faithfulness over per-surface claim subsets and AND with per-surface stability); (c) treat faithfulness as a per-reading gate and stability as a per-surface gate downstream of it. *Open; the next phase decides the wiring.*
4. **The iterated canonical-form rule for multi-suffix citations.** Pattern 1 above. The current `canonicalize_scalar_field` strips one suffix; a chain of suffixes (e.g. `policy_entropy_t.classification.collapse`) requires either iteration or a different rule (longest-prefix-match against statistic signal names). *Open; the smoke's surfaced cases drive the amendment.*
5. **The semantic-mismatch between `list[float]` as a per-step trajectory vs. `list[float]` as a per-perturbation aggregation.** Pattern 3 above. The verifier dispatches on type; the statistic's notes carry the indexing convention; the LLM's `cited_step_range` semantics differ across these cases. Either a per-statistic-type table the verifier reads, or a statistic-value-shape amendment, or both. *Open; the largest of the three smoke findings.*
6. **The prompt-builder amendment to name the actual dict keys.** Pattern 2 above. The LLM invents suffixes that don't correspond to actual dict keys; a prompt-builder amendment that names the keys explicitly (e.g. *"The `policy_entropy_t` classification carries the four labels: `dip_and_recover`, `collapse`, `stays_elevated`, `no_response`."*) would reduce the invention. *Open; a Phase 11.5 prompt-builder amendment.*

### Out of scope (preserved from the plan)

The admissibility consumer that joins faithfulness with stability (future phase); the prompt-builder amendment to remove the LLM's `faithfulness_status` write surface (deferred); the iterated canonical-form rule for multi-suffix citations (Pattern 1; future amendment); the prompt-builder amendment to name actual dict keys (Pattern 2; future amendment); the per-statistic-type list-shape table (Pattern 3; future amendment); Probe 4 real-builder-perturbation infrastructure; any change to Io, the actor, the world model, the dream state, or the runner; changes to Phase 7's frozen criteria prose; LLM calls of any kind (Phase 11 is verifier-only).

Phase 11 touched
`kind/mirror/faithfulness.py` (new, 692 lines),
`kind/mirror/__init__.py` (extended — re-exports for `FAITHFULNESS_THRESHOLD`, `FAITHFULNESS_VALUE_TOLERANCE`, `FaithfulnessAssignment`, `FaithfulnessResult`, `FaithfulnessStatus`, `canonicalize_scalar_field`, `verify_reading`),
`tests/test_faithfulness.py` (new, 676 lines, 26 tests),
`tests/test_phase_11_faithfulness_smoke_real_data.py` (new, 188 lines, 1 test),
the journal index entry, and this entry. No other files. No changes to `kind/observer/`, no changes to Phase 0's `StructuredClaim` schema, no changes to Phase 7's frozen criteria prose, no schema version bump, no on-disk Phase 9 / 10 / 12 / 13 artifact touched. The Phase 8 verbatim-clause tests, the Phase 9 module-level-identity check, and the Phase 10 verbatim-clause protection test all continue to pass byte-for-byte.

---

## Phase 11.5 — Window (2026-05-21)

Phase 11.5 builds Window: a small read-only viewer over the on-disk records Phases 9–13 wrote. It is not a substrate phase. The mirror substrate is structurally complete after Phase 11 — stability runner, faithfulness verifier, judge layer, calibration plane all built. Window is the tool that makes the substrate's outputs legible at scale, so the long Probe 4 run ahead can be checked in on remotely without hand-inspecting JSON.

### What was built

**`kind/window/` (new package).** Five modules:

- `kind/window/__init__.py` (15 lines) — re-exports `create_app`, the one public library surface; Window is otherwise consumed as a CLI.
- `kind/window/loaders.py` (374 lines) — one loader per record type. Every loader reads *through the existing Pydantic models* — `RoundResult`, `RoundJudgment`, `PassResult`, `StabilityResult`, `FaithfulnessResult`, `LLMCallAudit`, and the four telemetry models — never raw JSON. A `Loaded[ModelT]` outcome carries either the deserialized model or a human-readable error string; a malformed record becomes an error outcome and the rest of the directory still loads. `aggregate_llm_audit` rolls every round's `llm_call_records` into one `LLMCallAudit` via `LLMCallAudit.from_records`. `stream_last_write_ms` probes per-stream file modification times for the state heuristic.
- `kind/window/state.py` (501 lines) — pure view-state derivation: `decide_state` / `infer_current_state` (the four-state heuristic), `bucket_activity_by_hour` (the per-hour state-time breakdown), `pace_estimate`, `parse_run_start`, `format_duration`, `build_overview`, `build_round_rows`, `latency_distribution`. The `IoState` enum carries the four states plus `unknown`.
- `kind/window/server.py` (116 lines) — the Flask app factory `create_app(run_id, run_dir)` with five routes. The server opens files only for reading; all I/O is in the loaders.
- `kind/window/templates/` — seven Jinja templates (`base`, `overview`, `rounds`, `round_detail`, `judgment`, `audit`, `error`), plain HTML/CSS, no JavaScript, no charting.

**`scripts/run_window.py` (77 lines)** — the CLI entry point. `--run-id` (required), `--port` (default 8765), `--host` (default `0.0.0.0`). Builds the run directory path, validates it exists, constructs the app, serves.

**The five routes.** `/` overview (current state, uptime, totals, pace, 24h/7d breakdown); `/rounds` (round list, most-recently-modified first, with checkpoint ids, pass counts, and judgment verdicts where a matching `RoundJudgment` exists); `/rounds/<round_id>` (one round's full readings, pass by pass, LLM text quoted verbatim); `/judgments/<round_id>` (per-criterion verdicts, confidences, per-falsifier breakdown, the judge's rationale verbatim); `/audit` (the aggregated `LLMCallAudit` — call/retry/failure totals, wallclock, tokens, per-role per-checkpoint latency distribution).

**Load-bearing contracts.** Window is read-only — it opens nothing for write under `runs/`, makes no LLM calls, and does not construct or trigger any mirror production code path. It reads through Pydantic models, not raw JSON. It is interpretation-neutral — it counts, buckets, and aggregates; it does not rank, highlight, weight, or filter. The mtime-descending sort on the round list is the one ordering, and it is spec-mandated, not a weighting. Tailscale is not Window's concern: the server binds a port, and the host's Tailscale ACL gates remote access.

**Flask added as a project dependency.** `flask>=3.0` in `pyproject.toml` — the read-only viewer's HTTP layer. Flask bundles Jinja2 (templates) and Werkzeug (the test client the suite uses). Smaller dependency footprint than FastAPI for a viewer this size.

### Why Window is journaled as Phase 11.5 rather than Phase 12

Window does not appear in the Probe 2 synthesis. The synthesis's phase sequence runs research → synthesis → plan → build, each phase answering a specific substrate question; Window answers no substrate question. It is *practical infrastructure* — a tool for the builder, built because the Probe 4 run will be long and remote check-in on raw JSON is untenable. Numbering it "11.5" rather than "12" preserves the synthesis's phase numbering (the next substrate phase — the admissibility consumer that joins faithfulness and stability — is still Phase 12) while making the deviation from the spec'd sequence visible in the journal index. The discipline the project warns against — accreting features without each having a specific question being specifically answered — is not violated here: Window is named as infrastructure, not as substrate, and the numbering says so.

### The state-inference heuristic

Io's state is inferred from telemetry *write activity*, because no explicit state-transition events exist — Probe 3 (the dream probe) has not been built. The rule looks at the last **5 minutes** of write activity across the four streams: writes to `agent_step` → waking; writes to `dream_rollout` alone → dreaming; writes to `replay_meta` alone → dormant; no writes in the window → paused. "Write activity" is the file modification time of the stream's shards/JSONL — the genuine write signal.

The heuristic's limits, recorded so a future reader does not over-trust it:

- It cannot distinguish states the rule does not name. `dream_rollout` and `replay_meta` both active without `agent_step`, or a lone `world_event`, is surfaced as **unknown** rather than guessed.
- A run with no `telemetry/` directory at all (a mirror-only calibration run like `phase_13_calibration`) reports **unknown** — `paused` would falsely imply an Io process that simply stopped.
- The 24h/7d state-time breakdown is **coarse**: only `agent_step` and `world_event` records carry a per-record `wallclock_ms`. `dream_rollout` and `replay_meta` do not, so the breakdown distinguishes waking hours from idle hours but cannot place dreaming or dormant hours on the timeline.

When Probe 3 lands and emits explicit state-transition events, this presence-based heuristic can be replaced with an exact one.

### Test / mypy status

```
.venv/bin/mypy --strict kind/window/ tests/test_window.py scripts/run_window.py
Success: no issues found in 6 source files

.venv/bin/pytest tests/test_window.py -q
21 passed in 1.00 s

.venv/bin/pytest tests/ --ignore=tests/test_transport.py -q
1013 passed, 6 skipped in 53.52 s
(test_transport runs clean in isolation — 25 passed — the pre-existing
threading-barrier flake, unaffected by Window)
```

`tests/test_window.py` (370 lines, 21 tests): loader deserialization for each record type (rounds, judgments, passes, stability, faithfulness, agent-step telemetry); the malformed-record-surfaces-an-error case; the five state-inference cases (waking, dreaming, dormant, paused, unknown); the per-hour breakdown against a constructed 24-hour window; the pace estimate; run-start parsing; the three HTTP-200 routes plus the round-detail route; the 404-on-nonexistent-round case; the read-only invariant (`Path.open` write-tracking across a full route pass — Phase 8's pattern); the aggregate-audit sum; the import-discipline check (AST-parsed, so a docstring naming a forbidden module is not a false positive). The Phase 8 semantic read-only test `test_orchestrator_writes_only_to_mirror_side` continues to pass — Window goes nowhere near the orchestrator's write paths.

### What surprised

The state-inference heuristic keys on file modification time, and a `git checkout` resets every file's mtime to "now". Pointing Window at the committed `probe1` telemetry during development reported Io as **waking** — the 5000-step run finished months ago, but the parquet shards' mtimes were minutes old from the checkout. For a *live* run the mtime is the genuine write signal and the heuristic is sound; for a *replayed* or *checked-out* run it is meaningless. This is the same class of limitation as the presence-based heuristic itself — both are stand-ins for the explicit state-transition events Probe 3 would provide — but it is sharper than expected: the heuristic is correct only for the live-run case it was designed for, and silently wrong for an archived run. The README and this entry record it; the Probe 3 amendment that adds real transition events closes it.

A second, smaller surprise: `dream_rollout` and `replay_meta` carry no per-record wallclock. The four-state model treats dreaming and dormant as first-class, but two of the four streams cannot be placed on a timeline at record granularity — only at file-mtime granularity. The breakdown is coarser than the four-state model would suggest, and the gap is a telemetry-schema fact, not a Window choice.

### What's now closed

- **The "no way to see the on-disk records at scale" problem implicit in the Probe 4 plan.** The builder can now point a browser at a running Probe 4 run from any device on the Tailnet and read the rounds, judgments, readings, and LLM-call audit without hand-parsing JSON. Window unblocks the long run; it blocks no other phase.

### What's now newly open

1. **Whether Window should surface stability and faithfulness verdicts inline with the round views.** The `StabilityResult` and `FaithfulnessResult` records exist (Phases 10 and 11), and Window has loaders for both, but neither is wired into a route — the join between a per-surface stability score / per-reading faithfulness verdict and a round's per-pass readings is unclear. The admissibility-consumer phase (the next substrate phase, joining faithfulness and stability into a per-reading verdict) decides the join rule; once it lands, Window can render the joined verdict. *Open; the admissibility consumer decides.*
2. **Whether the state-inference heuristic should be replaced by explicit state-transition events from Probe 3 onward.** The 5-minute presence-based rule and the mtime-vs-checkout fragility (see "what surprised") are both artifacts of inferring state instead of reading it. Probe 3 builds the dream state; if it emits explicit `waking → dreaming → dormant` transition events into `world_event` (or a new stream), Window's `state.py` can read those directly and the heuristic retires. *Open; Probe 3's design decides whether such events are emitted.*
3. **What additional views Probe 4 will want.** The four routes cover the records that exist today. Probe 4 is builder-as-perturbation: it will produce patterns — perturbation-aligned reading drift, builder-mutation timelines — the current routes do not surface. Whichever patterns Probe 4 surfaces drive the next Window views; charting (deliberately out of scope here, HTML/CSS only) may earn its way in if a pattern is genuinely temporal. *Open; Probe 4's findings decide.*

### Out of scope (preserved from the plan)

The admissibility consumer that joins faithfulness and stability (next substrate phase); any change to Io, the actor, the world model, the dream state, or the runner; any change to the mirror's calibration plane; real-time monitoring, alerting, or push-based updates; authentication / authorization / access control (Tailscale handles); performance optimization (Window runs at human inspection timescales); plotting libraries or charting beyond native HTML/CSS; any interpretive view (rankings, highlighting, filtering — the viewer is interpretation-neutral by design).

Phase 11.5 touched
`kind/window/__init__.py` (new, 15 lines),
`kind/window/loaders.py` (new, 374 lines),
`kind/window/state.py` (new, 501 lines),
`kind/window/server.py` (new, 116 lines),
`kind/window/templates/` (new — seven Jinja templates),
`kind/window/README.md` (new, 91 lines),
`scripts/run_window.py` (new, 77 lines),
`tests/test_window.py` (new, 370 lines, 21 tests),
`pyproject.toml` (extended — `flask>=3.0` added to `dependencies`),
the journal index entry, and this entry. No other files. No changes to `kind/mirror/`, `kind/observer/`, `kind/agents/`, `kind/env/`, or `kind/training/` — Window is a pure consumer of their on-disk output. No schema version bump, no on-disk Phase 9 / 10 / 12 / 13 artifact touched. The Phase 8 semantic read-only invariant and the Phase 6–11 structural read-only invariants all continue to hold.

---

## Phase 12 — admissibility consumer (2026-05-21)

Phase 12 builds the consumer the synthesis named but never explicitly numbered: the layer that joins Phase 10's per-surface stability scores and Phase 11's per-reading faithfulness verdicts into a single per-reading admissibility verdict. Phases 10 and 11 built the two verifiers as *independent producers* — each emits its own record, neither knows about the other. Phase 12 reads both and computes the conjunction. After Phase 12 the mirror substrate is structurally complete: criteria frozen, orchestrator running adversarial passes, judge producing verdicts, two verifiers measuring stability and faithfulness, the consumer joining them, Window surfacing the result.

The build is the smallest substrate phase since Phase 11 — one new module, one consumer function, one convenience loader, two frozen result records, no LLM calls, no statistic recomputation, no extension to either verifier. The naming note below records why this is "Phase 12" despite the as-built sequence already carrying a Phase 12 (the calibration driver / smoke harness, journaled at 2026-05-13).

### What was built

**`kind/mirror/admissibility.py` (new, 515 lines).** The single Phase 12 module. Two frozen result records, one consumer function, one convenience loader.

- `AdmissibilityVerdict(BaseModel)` — per-reading verdict (frozen, `extra="forbid"`). Carries the reading envelope (`pass_index`, `criterion_id`, `reader_role`, `run_id`, `checkpoint_id`), `faithfulness_admissible` and `faithfulness_rate` (copied from the source `FaithfulnessResult`), `stability_admissible_per_surface` (copied from the matched `StabilityResult.admissible_per_surface`; empty dict when no stability result matched), `stability_admissible_all_surfaces` (the conjunction across the per-surface dict; vacuously `True` when empty), `admissible` (the join), `notes` (non-empty validator), `wallclock_ms`. Validators: range / non-empty checks on the scalar fields; `notes` non-empty.
- `AdmissibilityBatchResult(BaseModel)` — aggregate over a round or run (frozen, `extra="forbid"`). Carries the `verdicts` tuple, `n_readings_total`, `n_admissible`, the four `n_inadmissible_*` count fields, `admissibility_rate` (vacuously `1.0` when `n_readings_total == 0`), `notes`, `wallclock_ms`. The model validator pins that `n_admissible` plus the four `n_inadmissible_*` counts sum to `n_readings_total` and that `len(verdicts) == n_readings_total`.
- `compute_admissibility(*, faithfulness_results, stability_results, run_id, audit_jsonl_path=None) -> AdmissibilityBatchResult` — the consumer. Indexes the stability results by `(criterion_id, reader_role, checkpoint_id)`; for each faithfulness result, looks up the matching stability, computes `stability_admissible_all_surfaces` as the conjunction across the per-surface dict, computes `admissible = faithfulness_admissible AND stability_admissible_all_surfaces`, builds an `AdmissibilityVerdict` with notes describing the join, aggregates into the batch with the four-bucket inadmissibility partition. Optionally appends each verdict as a JSONL line to a caller-chosen `audit_jsonl_path`. Pure aside from the optional audit emission; no LLM calls; no statistic recomputation; no telemetry reads; the input tuples are not modified.
- `load_admissibility_inputs(run_id, run_dir) -> tuple[tuple[FaithfulnessResult, ...], tuple[StabilityResult, ...]]` — convenience loader. Reads `{run_dir}/mirror/faithfulness.jsonl` and `{run_dir}/mirror/stability.jsonl` — the locations Phases 11 and 10 committed — and returns the two tuples ready for `compute_admissibility`. A missing JSONL file yields an empty tuple. A thin wrapper; the verifier-side phases own the on-disk shapes.

**`kind/mirror/__init__.py` (extended).** Re-exports `AdmissibilityBatchResult`, `AdmissibilityVerdict`, `compute_admissibility`, `load_admissibility_inputs` under a "Phase 12 — admissibility consumer" section.

**`tests/test_admissibility.py` (new, 548 lines, 19 tests).** Pydantic invariants (frozen + `extra="forbid"` on both records, the batch counts-sum model validator, the `notes` non-empty validator); the four join cases (both pass, faithfulness-only fails, stability-only fails, both fail) each checked against the correct inadmissibility bucket; the no-stability-result case (admissible-with-no-stability counts in `n_admissible`; inadmissible-with-no-stability counts in `n_inadmissible_no_stability`, not `n_inadmissible_faithfulness`); empty-inputs vacuous case; the `(criterion_id, reader_role, checkpoint_id)` stability index (a matching stability is used, a non-matching one ignored); the per-surface conjunction across a three-surface criterion; the audit-JSONL emission and round-trip; the loader against on-disk fixtures; the input-immutability invariant; the verdict-notes-describe-the-join contract.

**`tests/test_phase_12_admissibility_smoke_real_data.py` (new, 84 lines, 1 test).** The Part 5 data-shapes smoke. Reads any faithfulness and stability records under `runs/phase_13_calibration/mirror/`, runs `compute_admissibility`, asserts the batch is well-formed. Skips when neither record type is present — which is the current state, because Phase 13 ran before Phases 10 and 11 existed. The skip-when-absent behavior keeps fresh-checkout `pytest` clean; it parallels Phase 11's data-shapes smoke.

**Window integration.** `kind/window/loaders.py` gains `load_admissibility_records(run_dir)` — loads `runs/{run_id}/mirror/admissibility.jsonl` as `AdmissibilityVerdict` records, one per line. `kind/window/state.py` gains `group_admissibility_for_round` — groups the run's verdicts by the round's passes, matching on `(pass_index, checkpoint_id)`, returning a list parallel to `pass_results`. `kind/window/server.py` gains the `/admissibility` route — loads the run's faithfulness and stability records through Window's error-tolerant loaders, runs `compute_admissibility`, renders the batch — and extends `/rounds/<round_id>` to surface the per-pass verdicts inline. `kind/window/templates/admissibility.html` (new) renders the batch summary (total readings, admissibility rate, the four inadmissibility breakdowns) and a per-verdict table; `round_detail.html` gains an admissibility-verdicts table per pass. `tests/test_window.py` gains four tests (loader deserialization, the `/admissibility` 200, the inadmissibility-breakdown display, the round-detail inline verdict) — 25 Window tests total.

### The join rule commitment — A (AND-conjunction at the reading level)

Phase 11's newly-open §3 named three join-rule candidates: (A) AND-conjunction — a reading is admissible iff faithfulness clears and every declared surface's stability clears; (B) per-surface faithfulness — recompute faithfulness over per-surface claim subsets and AND with per-surface stability; (C) faithfulness as a per-reading gate, stability as a per-surface gate downstream of it.

Phase 12 commits **A**. The reasoning:

- **A is the simplest rule that uses both verifiers' verdicts as they already exist.** Both `FaithfulnessResult.admissible` and `StabilityResult.admissible_per_surface` are *already-computed* admissibility booleans — Phases 10 and 11 each applied their own thresholds and emitted a verdict. A reads those verdicts and computes a boolean conjunction. It recomputes nothing.
- **B was rejected because it would re-open Phase 11's verifier.** Per-surface faithfulness means recomputing the faithfulness rate over per-surface claim subsets — that is a change to the faithfulness verifier's granularity, which the build's load-bearing constraint explicitly forbids ("the consumer does not amend either verifier"). B is a Phase 11 amendment masquerading as a Phase 12 design choice. If the empirical case for per-surface faithfulness arrives, it is a journaled amendment to Phase 11, and then A's conjunction operates over the finer-grained inputs unchanged.
- **C was rejected because "a per-surface gate downstream" is not a verdict.** C keeps faithfulness and stability as two separate gates and never produces the single per-reading boolean the synthesis asked for. The synthesis named "a future admissibility consumer combines them" — combination means one verdict, not two gates left side by side. C defers the actual join; it does not perform it.

The empirical grounding is Phase 11's smoke. Phase 11 ran the faithfulness verifier across both Phase 13 round results and found the 0.80 threshold cleanly separating probe_1 (rate 0.68, inadmissible) from probe_1.5 (rate 1.00, admissible). That is a per-reading faithfulness verdict that already discriminates. Phase 10's stability smoke produced a per-surface verdict (head-internal paraphrase 0.333, inadmissible). A conjunction of those two — faithfulness-per-reading AND stability-per-surface-all — is exactly the gate the synthesis described: a reading passes only if it is both faithful to its data and stable under variation. The two verifiers measure genuinely different axes; ANDing their verdicts is the rule that says "both must hold," which is what "admissibility requires both verifiers" means.

### The no-stability-result case

The stability runner is opt-in per `(criterion, role, checkpoint)` — not every reading with a faithfulness result has a matching stability result. Phase 12 commits the **permissive** reading: when no stability result matches, the verdict's `stability_admissible_per_surface` is the empty dict, `stability_admissible_all_surfaces` is `True` by vacuous-case convention, and the verdict is gated by faithfulness alone. A reading with no stability record is admissible iff its faithfulness alone clears.

This is the most conservative reading of the synthesis's "admissibility requires both verifiers": the synthesis said both *should run*, but in practice not every reading gets the stability check, and a missing stability check should not silently fail an otherwise-faithful reading. The cost of the permissive reading is recorded distinctly — the verdict's `notes` names the missing-stability case explicitly, and the batch's `n_inadmissible_no_stability` count buckets inadmissible-with-no-stability readings separately from `n_inadmissible_faithfulness`, so a downstream consumer can see exactly how many readings were gated on faithfulness alone for want of a stability check.

The candidate amendment — tightening from permissive ("vacuous when missing") to strict ("both verifiers must have run, a reading with no stability record is inadmissible") — is journaled in newly-open §1 below and *not* made. Phase 12 commits the permissive reading; the strict reading is a one-line change (`stability_admissible_all_surfaces` defaults to `False` when the dict is empty) whenever the empirical case arrives.

### The Window integration

The verdict surfaces in two places, both interpretation-neutral. The `/admissibility` route shows the run-level `AdmissibilityBatchResult` — total readings, admissibility rate, the four inadmissibility breakdowns, and a per-verdict table. The `/rounds/<round_id>` route surfaces the per-reading verdict inline: each pass's readings are followed by an admissibility-verdicts table for that pass. Window does not rank, highlight, or filter readings by admissibility — it shows the verdict, the contributing rates, and the join notes, per the Window interpretation-neutrality contract. This closes Phase 11.5's newly-open §1 ("whether Window should surface stability and faithfulness verdicts inline with the round views" — it now does, via the joined verdict).

The `/admissibility` route runs `compute_admissibility` live against the run's faithfulness and stability JSONL files; the round-detail route reads the persisted `admissibility.jsonl`. `compute_admissibility` is a pure verifier-side function with no LLM call and no production code path — the same category as `LLMCallAudit.from_records`, which Window's `/audit` route already calls; the Phase 11.5 import-discipline test (Window must not import `kind.mirror.orchestrator`, `kind.mirror.judge_driver`, `kind.mirror.calibration.smoke`) continues to pass, and the read-only-server invariant test confirms the consumer with `audit_jsonl_path=None` opens no file for write.

### Test / mypy status

```
.venv/bin/mypy --strict kind/mirror/ kind/window/ tests/test_admissibility.py \
    tests/test_phase_12_admissibility_smoke_real_data.py tests/test_window.py
Success: no issues found in 35 source files

.venv/bin/pytest tests/test_admissibility.py        -> 19 passed
.venv/bin/pytest tests/test_window.py               -> 25 passed (21 prior + 4 new)
.venv/bin/pytest tests/ --ignore=tests/test_transport.py
1036 passed, 7 skipped in 51 s
.venv/bin/pytest tests/test_transport.py            -> 25 passed (in isolation)
```

The Phase 12 admissibility data-shapes smoke runs as part of the standard suite (no `--run-real-api`) because it issues no LLM calls; it currently skips, because no run on disk carries both a `faithfulness.jsonl` and a `stability.jsonl`. The four real-API marks (Phase 9 judge smoke, Phase 10 stability smoke, Phase 12 calibration smoke, Phase 13 calibration) are unchanged and not run in this gate. The Phase 8 verbatim-clause tests, the Phase 9 module-level-identity check, the Phase 10 verbatim-clause protection test, and the Phase 6–11.5 structural and semantic read-only invariants all continue to pass byte-for-byte unchanged.

### What surprised

Phase 12 is the first substrate phase whose smoke is a pure skip. Phase 10 ran a real Gemini stability check; Phase 11 ran its verifier across real Phase 13 readings and found the 0.68 / 1.00 split. Phase 12's consumer has *no real joined data to consume* — there is no run on disk that carries both a faithfulness JSONL and a stability JSONL, because Phase 13's calibration ran before Phases 10 and 11 existed and the two verifiers' opt-in audit emission has not yet been wired into a driver. The consumer is structurally complete, fully unit-tested against constructed fixtures, and mypy-clean — but it has never run against the actual output of the two verifiers it joins. That gap is not a Phase 12 defect (the build's scope is the consumer, not a calibration run), but it is the honest state: the substrate is structurally complete and empirically half-exercised. The first run that emits both verifiers' JSONL and then runs `compute_admissibility` over them is the real test of the join, and it has not happened yet.

The second surprise is a granularity finding at the Window layer. The inline round-detail integration cannot bind a verdict to a *specific* `MirrorReading` object, because — as Phase 11 already recorded — the `StructuredReading` envelope carries no `criterion_id`. The verdict's envelope is `(pass_index, criterion_id, reader_role, checkpoint_id)`; a reading in a round carries `reader_role` and (via its pass) `checkpoint_id`, but not `criterion_id`. So the round-detail view joins verdicts to a *pass* (by `pass_index` and `checkpoint_id`) and renders them in a per-pass table, not woven into each reading's row. This is honest about what the data supports; a tighter per-reading binding would require threading `criterion_id` onto the reading envelope, which is a Phase 0 schema change out of Phase 12's scope.

### What's now closed

- **Phase 11's newly-open §3 — "How Phase 11's faithfulness verdicts and Phase 10's stability scores join into a single admissibility verdict."** Closed. The join rule is A (AND-conjunction at the reading level); `compute_admissibility` is the consumer; the verdict is the per-reading `AdmissibilityVerdict`. The two verifiers now have a structural consumer.
- **Phase 11.5's newly-open §1 — "Whether Window should surface stability and faithfulness verdicts inline with the round views."** Closed. Window surfaces the *joined* verdict — at `/admissibility` for the run and inline per pass at `/rounds/<round_id>`. The join rule the admissibility consumer commits is what made the inline display well-defined.
- **The admissibility consumer as a substrate phase.** The synthesis named both verifiers feeding a future consumer; the consumer is now built. The mirror substrate is structurally complete.

### What's now newly open

1. **Whether the no-stability-result case should be tightened from permissive to strict.** Phase 12 commits the permissive reading (vacuous stability admissibility when no stability result matched; faithfulness alone gates). The strict reading — "both verifiers must have run; a reading with no stability record is inadmissible" — is a one-line change to the empty-dict default. The permissive reading is right while the stability runner is opt-in and most readings have no stability check; the strict reading becomes right once a calibration protocol runs the stability check for every reading. *Open; the first real calibration run that emits both JSONLs decides.*
2. **Whether the admissibility threshold should become an explicit module constant.** Phase 12 has no threshold of its own — a reading is admissible iff *both* verifiers' already-applied thresholds (`FAITHFULNESS_THRESHOLD` in `faithfulness.py`, `PARAPHRASE_THRESHOLDS` / `RESEED_THRESHOLDS` in `stability.py`) are cleared, and the consumer reads the verifiers' verdict booleans rather than their rates. If a future phase wants a consumer-level threshold — e.g. "admissible iff at least N of M surfaces clear," a softer rule than strict AND — that threshold becomes an explicit `admissibility.py` module constant. *Open; deferred until the join rule itself is revisited.*
3. **Whether the join rule should remain A or evolve toward B.** A (AND-conjunction) is committed. B (per-surface faithfulness — recompute faithfulness over per-surface claim subsets and AND with per-surface stability) would give a finer-grained verdict but requires amending Phase 11's verifier to emit per-surface faithfulness rates. As more joined data accumulates, if the per-reading faithfulness verdict proves too coarse — e.g. a reading is faithful at head-internal but not behavior-side and the single per-reading rate hides that — B becomes the amendment. *Open; data accumulation decides.*
4. **Whether the round-detail inline verdict should bind to a specific reading rather than to a pass.** The Window inline integration joins verdicts to a pass (`pass_index` + `checkpoint_id`), not to an individual `MirrorReading`, because the reading envelope carries no `criterion_id` (see "what surprised"). A per-reading binding would need `criterion_id` threaded onto the `StructuredReading` envelope — a Phase 0 schema change. *Open; a schema amendment if the per-pass grouping proves insufficient for Probe 4's inspection needs.*

### The naming note

The Probe 2 synthesis's numbering had Phase 12 as the calibration driver / smoke harness. That phase landed in the as-built sequence under that number and is journaled above at "Phase 12 — calibration driver and smoke harness (2026-05-13)". The synthesis's *next* substrate phase after the two verifiers — the admissibility consumer — was never explicitly numbered. This build is journaled as "Phase 12 — admissibility consumer" to preserve the synthesis's intent (the next substrate phase after the verifiers). The numbering is forward-looking from where Probe 2 has arrived, not retrospective on the synthesis; the journal now carries two `## Phase 12` headings, distinguished by their dates and titles, and that is the accepted cost of preserving both the as-built record and the synthesis's intent.

### Out of scope (preserved from the plan)

Probe 4 real-builder-perturbation infrastructure; any change to Io, the actor, the world model, the dream state, or the runner; any change to Phase 10's stability runner or Phase 11's faithfulness verifier (the two verifiers stay unchanged; the consumer reads both); any change to Phase 7's frozen criteria; LLM calls of any kind (the consumer is verifier-only); the iterated canonicalization rule and the per-statistic-type list-shape table from Phase 11's newly-open §4 / §5 (still open); the prompt-builder amendments from Phases 10 and 11 (still deferred).

### Sequencing note

Phase 12 closes the substrate sequence the synthesis named. The mirror substrate is structurally complete: the criteria are frozen, the orchestrator runs adversarial passes, the judge produces verdicts, the two verifiers measure stability and faithfulness, the consumer joins them into a per-reading admissibility verdict, and Window surfaces the result. The next direction is not another mirror substrate phase — it is either Probe 3 (the dream state — Io's offline processing machinery) or one of the deferred amendments (the prompt-builder fixes, the iterated canonicalization, the per-statistic-type list-shape table, the calibration run that finally emits both verifiers' JSONL so the consumer runs against real joined data).

Phase 12 touched
`kind/mirror/admissibility.py` (new, 515 lines),
`kind/mirror/__init__.py` (extended — re-exports for `AdmissibilityBatchResult`, `AdmissibilityVerdict`, `compute_admissibility`, `load_admissibility_inputs`),
`tests/test_admissibility.py` (new, 548 lines, 19 tests),
`tests/test_phase_12_admissibility_smoke_real_data.py` (new, 84 lines, 1 test),
`kind/window/loaders.py` (extended — `load_admissibility_records`),
`kind/window/state.py` (extended — `group_admissibility_for_round`),
`kind/window/server.py` (extended — the `/admissibility` route and the round-detail extension),
`kind/window/templates/admissibility.html` (new, 64 lines),
`kind/window/templates/round_detail.html` (extended — the per-pass admissibility table),
`kind/window/templates/base.html` (extended — the `/admissibility` nav link),
`tests/test_window.py` (extended — 4 new tests, 25 total),
the journal index entry, and this entry. No other files. No changes to Io, the actor, the world model, the dream state, the runner, the two verifiers, or Phase 7's frozen criteria. No schema version bump, no on-disk Phase 9 / 10 / 12 / 13 artifact touched.

---

## Phase 12.5 — deferred-cleanup pass (2026-05-22)

Phase 12 closed the mirror substrate structurally. Phase 12.5 is not a substrate phase — it is a focused cleanup session that closes four small items deferred from the Phase 10 and Phase 11 newly-open lists. The four items are independent of each other; each closes a single newly-open item; each had an empirical case already in hand from a prior phase's smoke or finding. The deviation from the synthesis-spec'd phase sequence is journaled here under the half-numbered heading so it stays visible.

### What was built

- **Item 1 — drop the LLM's `faithfulness_status` write surface.** `kind/mirror/llm_caller.py` gains `_drop_faithfulness_status`, a recursive schema munger run as the first step of `_to_gemini_schema`; it removes the `faithfulness_status` property from every object schema before the schema is sent to Gemini. `kind/mirror/structured.py` gains a writer-side default — `faithfulness_status: FaithfulnessStatus = "not_checked"` — so a claim parsed from a response that omits the field lands at `"not_checked"`. Closes Phase 11 newly-open §2.
- **Item 2 — name the trajectory-classifier labels in the equanimity prompt.** `kind/mirror/prompt_builder.py` gains the module-level `TRAJECTORY_CLASSIFIER_LABELS_BLOCK`, built from `statistics.ENTROPY_CLASS_LABELS` / `KL_CLASS_LABELS`, inserted into the equanimity fragment after the signals block. The block names the four `policy_entropy_t` labels and the four `posterior_kl_t` labels verbatim. It is a prompt-builder amendment, not a criterion amendment — `criteria_v2.py` is untouched. Closes Phase 11 newly-open §6.
- **Item 3 — iterated canonical-form rule.** `canonicalize_scalar_field` changes from a one-shot single-suffix strip to an iterated walk: it strips suffixes from the full cited string down toward the bare form and returns the first prefix that matches a known statistic signal name, falling back to the bare form when none match. The signature gains a `known_signal_names: frozenset[str]` parameter. Closes Phase 11 newly-open §4.
- **Item 4 — shared canonicalization helper.** The function (with item 3's iterated signature) now lives in a new `kind/mirror/citation_canonical.py`. `faithfulness.py` and `stability.py` both import from there; `stability.py`'s `_claim_tuples_at_surface` canonicalizes the `cited_scalar_field` before it enters the Jaccard tuple, using the criterion's `SignalMapping` names as the known set. `kind/mirror/__init__.py` re-exports `canonicalize_scalar_field` from the new location; the `from kind.mirror import canonicalize_scalar_field` and `from kind.mirror.faithfulness import canonicalize_scalar_field` paths both still work. Closes Phase 10 newly-open §1.

### Why journaled as 12.5 rather than as four separate phases

These are surgical fixes to existing substrate, not substrate phases. The journal's discipline is phase-as-question-asked-and-answered; four one-line "phases" would dilute that. The question Phase 12.5 asks is a single one — *which deferred items are ready to close given the empirical cases already in hand?* — and the answer is these four. Bundling them into one session, under a half-number that flags the deviation, keeps the phase sequence legible: the next full number is still free for the next real phase (Probe 3).

### The four items closed, with empirical grounding

- **Item 1** closes Phase 11 newly-open §2. Grounding: Phase 10's stability smoke saw the LLM pre-fill `faithfulness_status` with `"resolved"` on 4 of 6 readings. The verifier already overwrites the verdict (Phase 11 chose option A), but the LLM was still being *asked* to write a field it does not own. Schema-level removal is structural — Gemini's structured-output validator constrains against the schema, so a dropped property is one the model is never asked to produce. The surgical option (a) from Phase 11's §2 is what landed.
- **Item 2** closes Phase 11 newly-open §6. Grounding: Phase 11's faithfulness smoke surfaced Pattern 2 — the LLM inventing dict-key suffixes (`policy_entropy_t.no_response_count`) that don't match the classifier's actual labels. Naming the real labels verbatim removes the ambiguity at its source.
- **Item 3** closes Phase 11 newly-open §4. Grounding: Phase 11's faithfulness smoke surfaced Pattern 1 — two-dot forms like `policy_entropy_t.classification.collapse`. A single strip yields `policy_entropy_t.classification`, which matches no signal name; the iterated rule reaches `policy_entropy_t`.
- **Item 4** closes Phase 10 newly-open §1. Grounding: Phase 10's smoke found the stability metric compared the verbatim LLM string as `cited_scalar_field`, so the bare-vs-compound divergence dropped HEAD_INTERNAL paraphrase agreement to 0.333. Canonicalizing inside the metric is what that finding asked for; the shared module is what makes the same rule single-sourced across both verifiers.

### What's still deferred

The pass closes four items; seven remain open and visible:

1. **The calibration run that emits both verifiers' JSONL.** Phase 12 newly-open. The admissibility consumer's smoke stays a pure skip until a run on disk carries both a `faithfulness.jsonl` and a `stability.jsonl`. Deferred until before Probe 4.
2. **`criterion_id` on the `StructuredReading` envelope.** Phase 12 newly-open §4. A Phase 0 schema bump with migration. Deferred until Probe 4 evidence shows pass-level binding limits real inspection.
3. **The per-statistic-type list-shape table.** Phase 11 newly-open §5. A larger statistic-shape-protocol amendment. Deferred until Probe 4 produces evidence the per-perturbation vs per-step semantics matter in practice.
4. **The equanimity-falsifier framework reading.** Phase 9 newly-open. Independent philosophical work. Deferred to a dedicated session, possibly before Probe 4. Explicitly not in this pass — Phase 7's criteria stay frozen.
5. **The no-stability-result tightening from permissive to strict.** Phase 12 newly-open §1. Depends on item 1 of this list producing empirical evidence. Deferred.
6. **The admissibility threshold as an explicit module constant.** Phase 12 newly-open §2. Deferred until the join rule itself is revisited.
7. **The join rule evolution from A to B (per-surface faithfulness).** Phase 12 newly-open §3. Deferred until accumulated data suggests the per-reading verdict is too coarse.

### Test / mypy status

`mypy --strict` over `kind/mirror/`, `kind/observer/`, and the six gate test files — `Success: no issues found in 42 source files`. `pytest tests/` — 1071 passed, 7 skipped (the real-API-marked smokes and the Phase 12 admissibility data-shapes smoke, which skips for want of a run carrying both verifiers' JSONL), 1 failure: `test_transport.py::test_barrier_queues_mutates_and_drains_in_order`, the pre-existing thread-timing flake, which passes on isolated re-run. New tests: `tests/test_citation_canonical.py` (new file, 6 tests); `tests/test_llm_caller.py` +2; `tests/test_prompt_builder.py` +1; `tests/test_faithfulness.py` net +1 (the one-shot multi-dot test replaced by two iterated tests); `tests/test_stability.py` +1. The Phase 8 verbatim-clause tests, the Phase 9 module-level-identity check, the Phase 10 verbatim-clause protection test, and the Phase 11 read-only test all continue to pass byte-for-byte.

### What surprised

Item 1's plan described the `faithfulness_status` default as already present — "the Phase 0 `StructuredClaim` schema declares `faithfulness_status` as a Literal whose writer-side default is `"not_checked"`" — and built on "Pydantic will fill in the default value." It never existed. The field was required-with-no-default. That is not a footnote: it is the actual root cause. The LLM was filling `faithfulness_status` in Phase 10's smoke *because the field was required* — Gemini's structured-output validator fills every required field. Item 1's real fix is therefore two-part: drop the property from the schema (so it stops being required) *and* give the field a default (so a response without it still parses). Dropping it alone would have retry-exhausted every pass.

The second surprise followed from the first. Adding the default — described by the plan as a no-op on the model — changed the `schemas/v0.3.0.json` byte-pinned export: a Pydantic field with a default leaves the `required` list and gains a `"default"` key. `test_schemas_v030_export.py` caught it immediately. The diff was confined exactly to `faithfulness_status` (two occurrences — the standalone `StructuredClaim` and the one nested under `StructuredReading`), so the export was regenerated and re-pinned. The lesson: "the model is unchanged" and "the field gets a default" cannot both be true; a default is a schema change, small but real, and the byte-pin is the thing that makes it visible.

### What's now closed

- **Phase 11 newly-open §2** — the LLM's `faithfulness_status` write surface. Closed by item 1: the property is gone from the schema the LLM sees; the field has a writer-side default.
- **Phase 11 newly-open §6** — the LLM inventing dict-key suffixes. Closed by item 2: the equanimity prompt names the eight real classifier labels verbatim.
- **Phase 11 newly-open §4** — the iterated canonical-form rule for multi-suffix citations. Closed by item 3.
- **Phase 10 newly-open §1** — the shared canonicalization helper between stability and faithfulness. Closed by item 4.

### What's now newly open

1. **Whether the shared canonicalization helper should also be called from Window's loaders or the admissibility consumer.** Neither currently canonicalizes — both work with already-resolved data — but the import path (`kind.mirror.citation_canonical`) is now available if a future need surfaces. *Open; no current need.*
2. **Whether the prompt-builder's labels block should extend to other criteria's classifier-shaped statistics.** `TRAJECTORY_CLASSIFIER_LABELS_BLOCK` covers `policy_entropy_t` and `posterior_kl_t` — the equanimity criterion's two dict-valued signals. If a future phase adds a criterion with its own classifier-shaped statistic, that criterion's fragment needs an analogous block. *Open; no such criterion exists while Phase 7's set stays frozen.*
3. **`_suffix_after_canonical` no longer mirrors the canonicalization rule.** Item 3 made `canonicalize_scalar_field` iterated, but `_suffix_after_canonical` (faithfulness-internal, for dict-key lookup) still takes the last dot-separated segment. For a two-dot citation `policy_entropy_t.classification.collapse` the canonical form is `policy_entropy_t` and the suffix is `collapse` — which is the correct dict key, so the two functions diverging is *currently harmless*. But the divergence is a latent edge: a future dict whose keys themselves contain dots would break the last-segment heuristic. *Open; harmless under the current single-token classifier labels.*

### Out of scope (preserved from the plan)

The seven deferred items above. Plus: Probe 3 work; Probe 4 infrastructure; any change to Io, the actor, the world model, the dream state, or the runner; any LLM call; any change to Phase 7's frozen criteria (criterion prose, falsifier prose, and signal mappings are all untouched — item 2 amends only how the prompt-builder presents results).

### Files touched

`kind/mirror/structured.py` (the `faithfulness_status` default),
`kind/mirror/llm_caller.py` (the `_drop_faithfulness_status` munger + `_to_gemini_schema` integration),
`kind/mirror/prompt_builder.py` (the `TRAJECTORY_CLASSIFIER_LABELS_BLOCK`),
`kind/mirror/citation_canonical.py` (new, the shared iterated helper),
`kind/mirror/faithfulness.py` (imports the shared helper; `verify_reading` threads `known_signal_names`),
`kind/mirror/stability.py` (imports the shared helper; `_claim_tuples_at_surface` canonicalizes),
`kind/mirror/__init__.py` (re-export from the new module),
`schemas/v0.3.0.json` (regenerated — the `faithfulness_status` default),
`tests/test_citation_canonical.py` (new, 6 tests),
`tests/test_llm_caller.py`, `tests/test_prompt_builder.py`, `tests/test_faithfulness.py`, `tests/test_stability.py` (extended),
the journal index entry, and this entry. No other files.

### Sequencing note

Phase 12.5 closes four cleanups. The remaining seven deferred items are recorded above and stay visible; whichever become relevant during Probe 3 or before Probe 4 get picked up in dedicated sessions. After this pass, Probe 3 can start fresh.

---
