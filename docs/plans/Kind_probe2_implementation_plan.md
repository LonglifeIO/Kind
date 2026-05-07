# Probe 2 — Implementation Plan (v2)

*Operational plan that translates the revised Probe 2 synthesis (`docs/decisions/Kind_probe2_synthesis.md`, v2) into a concrete build sequence on top of the Probe 1 substrate as extended by Probe 1.5 v2 (`docs/decisions/Kind_probe1_5_synthesis.md`, v2; `docs/plans/Kind_probe1_5_implementation_plan.md`, v2). The substrate is settled (`Kind_architectural_decision.md`); the Probe 1 implementation is settled (`Kind_probe1_synthesis.md`); the Probe 1.5 implementation is settled (substrate-side `SelfPredictionHead`+EMA target; PolicyView gains `self_prediction_error: float`; TelemetryView gains `self_prediction: Tensor` and `self_prediction_error_masked: bool`; AgentStep gains the three corresponding fields; `0.2.0` schema in force; small-Gaussian init on the actor's new column carries forward as the Phase 7.5 default). The environment is extended by §2.1 (random non-wall start cell + sham-perturbation flag-only path), but otherwise unchanged. Probe 2 is mirror calibration with frozen criteria, adversarial structure, and three reading surfaces — the parallel reader/Skeptic/Judge architecture, the seven-element calibration protocol per surface, the structured-reading schema with explicit `reading_surface` field that downstream probes will track against. Forward-compatibility lives in the schemas (state-typed reading hooks for Probe 3, distinguishability fields for Probe 4), the digest's drill-down accessor (extended for the three reading surfaces), the conditioning analysis module (the formalized version of Probe 1.5's `show_self_prediction_conditioning` helper), and the discipline elements that survive intact across probes. Nothing in this plan changes Io's substrate; nothing introduces real-time mirror calls; nothing introduces reward, value heads, or critic networks.*

*This is the v2 plan. The v1 plan is preserved at `Kind_probe2_implementation_plan.v1.md` for reference. The v1 plan was drafted 2026-05-03 before Probe 1.5 ran; v2 absorbs the eleven Probe 1.5 v2 §10 items plus four Phase 8 additions (the methodology note on three reading surfaces, the column-init confound, the new lesion candidate `init_zero_scalar_column`, the prompt-revision direction toward head-internal and behavior-side criteria) and the deferred environmental-auxiliary control's elevated informativeness recommendation. v1's architectural pieces preserve in shape; the per-surface extensions are what the Probe 1.5 findings require.*

---

## 1. Build order with dependency graph

Hard dependencies (must run sequentially):

- **Schemas before writers.** The Probe 2 schema deltas — `StructuredReading` (now with `reading_surface` field), `PreRegistration` (now with `reading_surface` and `column_init` fields), the `is_sham` payload field on `world_event` — exist before any component emits or consumes them.
- **Hierarchical digest before adversarial readers.** The two-axis adversarial pattern reads digests at three reading surfaces; readers cannot be tested end-to-end before the digest module produces base + drill-down output across substrate-side, head-internal, and behavior-side fields.
- **Conditioning analysis module before behavior-side readings.** The behavior-side reading surface depends on per-state action-distribution-under-perturbation tables; the readers' behavior-side claims and the faithfulness verifier's behavior-side citation resolution both consume this module's output.
- **Faithfulness verifier before stability runner.** Stability requires admissibility checks; admissibility requires per-surface faithfulness resolution.
- **Adversarial reader before Judge.** The Judge takes paired readings as input; building the Judge before the readers means stubbing input shape.
- **All of the above before the calibration smoke.** The smoke is the final integration gate; it exercises everything else against pre-existing Probe 1.5 telemetry (the Phase 7.5 main run plus the Phase 8 frozen-target run; the v1 plan's Probe 1 reference at `runs/probe1-20260503-123926/` is still available for cross-probe baselining but Probe 2's primary fixture shifts to Probe 1.5's runs).

Parallelizable (no hard dependency between them after Phase 0):

- The env-revision (Phase 1) is independent of every mirror-side phase.
- The shuffled-telemetry generator (Phase 3, including the new fourth scalar-shuffle-within-trajectory protocol), the lesion scaffold (Phase 4, including the three Probe-1.5-specific lesion kinds), the pre-registration sink (Phase 5), the conditioning analysis module (Phase 6), and the faithfulness verifier (Phase 11) can be built in parallel.
- The held-out-criterion mechanism (Phase 7) depends on Phase 8's prompt-builder shape but its config surface can be drafted alongside.

**Optional preceding addition (synthesis §4 (v2) recommendation).** The deferred environmental-auxiliary control (`scripts/probe1_5_control_environmental.py`, parallel to `scripts/probe1_5_control_frozen_target.py`) is *recommended* not *required* as a small phase before Phase 0. Phase 8's reading shows the substrate-shaping is largely generic (within-Probe-1.5-family KS-D 0.07–0.13 vs cross-family-to-Probe-1 KS-D 0.22–0.34), which makes the environmental-auxiliary control's reading the canonical disambiguator between "any auxiliary that flows through the encoder produces similar shaping" and "frozen-target is its own special case that doesn't generalize to other targets." If built, it lands as an additional comparison anchor for `scripts/probe1_5_compare_controls.py` (the four-way comparison driver from Probe 1.5 Phase 8 extends to five-way without structural work; the `--auto` discovery of run directories already supports the prefix). If deferred, the journal entry preceding Phase 0 records the decision and the absence's effect on the substrate-side stratification's empirical anchor. The build phase decides; v2 of the synthesis recommends building it.

Phased build (each phase is a unit of work):

| Phase | Work | Depends on |
|---|---|---|
| **0a (recommended). Environmental-auxiliary control** | `scripts/probe1_5_control_environmental.py` parallel to the frozen-target control script. Five-way comparison run via `scripts/probe1_5_compare_controls.py`. Journal entry naming whether substrate-shaping is auxiliary-target-shape-independent in the broader sense (sits with the existing three Probe 1.5 runs) or auxiliary-target-shape-dependent (sits closer to Probe 1). | — |
| **0. Probe 2 schemas** | `StructuredReading` (`MIRROR_READING_V2_VERSION = "0.2.0"`) **with `reading_surface: Literal["substrate_side", "head_internal", "behavior_side"]` field on `StructuredClaim`**, `PreRegistration` (`PRE_REG_SCHEMA_VERSION = "0.2.0"` — bump from v1's 0.1.0) **with `reading_surface` and `column_init` fields**, documented `is_sham: bool` field on `world_event` payload, new JSON Schema export `schemas/v0.3.0.json` (covering both Probe 1.5's 0.2.0 telemetry models and Probe 2's 0.2.0 mirror models). | 0a (if built) |
| **1. Env revision** | Default `start_cell=None` (random); `env_reset` payload includes `start_cell`; sham-perturbation flag-only path on `EnvServer`; agent observation byte-equal pre/post sham. | 0 |
| **2. Hierarchical digest (extended for three reading surfaces)** | `kind/observer/digest.py` extension: `build_hierarchical_digest`, `digest_episode_window`, `digest_drill_down`. Per-episode mini-digest exposes substrate-side fields (per-dim KL allocation, ensemble-disagreement-variance trajectory) **plus head-internal fields (`self_prediction_error_t` distribution, per-dim `self_prediction_t` allocation, masked-step count) plus behavior-side fields (per-state action-distribution-under-perturbation summary; pulled from the conditioning analysis module's output)**. Existing `build_digest` retained as `build_flat_digest` for Probe-1-compatible reads. The drill-down accessor's per-step output includes the masked flag (Probe 1.5 v2 §10 item 1). | 0 |
| **3. Shuffled-telemetry generator (four protocols)** *(parallel with 4, 5, 6, 11)* | `kind/observer/shuffle.py` with four protocols (`shuffle_within_episode`, `shuffle_across_episodes`, `decouple_action_state`, **`shuffle_scalar_within_trajectory` — NEW v2** preserving per-step marginals on `self_prediction_error_t` while breaking within-trajectory dynamics). Reproducible from a shuffle seed. | 0 |
| **4. Lesion scaffold (three Probe-1.5-specific candidates)** *(parallel with 3, 5, 6, 11)* | `RunnerConfig.lesion_kind: Literal[None, "ensemble_k1", "ensemble_constant", "disable_self_prediction", "zero_or_randomize_scalar", "init_zero_scalar_column"]`. Runner threads it through to the relevant component (LatentDisagreementEnsemble for ensemble lesions; WorldModel for `disable_self_prediction`; views.split for `zero_or_randomize_scalar` evaluation; checkpoint loader for `init_zero_scalar_column`). Lesion shape recorded into run-level metadata (a `world_event` record with `event_type="mirror_marker", source="system", payload.lesion_kind=...` — the same convention Probe 1.5 Phase 5 settled). The three Probe-1.5-specific lesion kinds are the ones synthesis §2.4 element 4 names; the v1 ensemble candidates are preserved as alternatives. | 0 |
| **5. Pre-registration sink** *(parallel with 3, 4, 6, 11)* | `kind/observer/pre_reg.py` with `PreRegistration` model **including `reading_surface` and `column_init` fields** and `PreRegSink` (JSONL append-only). Pre-reg records land at `runs/{run_id}/pre_reg/pre_reg.jsonl`. | 0 |
| **6. Conditioning analysis module (NEW v2)** *(parallel with 3, 4, 5, 11)* | `kind/observer/conditioning.py` — the formalized version of Probe 1.5's `show_self_prediction_conditioning` helper. Produces numerical conditional distributions for `actor_action \| scalar_value` per state, with faithfulness-check support. Per Probe 1.5 v2 §10 item 11; v2 of this plan extends the module to produce per-surface readings (substrate-side per-dim KL allocation; head-internal per-dim `sp_err` allocation; behavior-side per-state action-distribution-under-perturbation). The module is the analysis surface the readers' behavior-side claims cite and the faithfulness verifier resolves against. | 0 |
| **7. Held-out criterion mechanism (expanded default candidates)** | `kind/mirror/criteria.py` with a registry of named criteria (prompt fragments **per reading surface**) and an active-set toggle. Default active set excludes the held-out criterion until a designated checkpoint. **v2 default candidates expand from v1's three (reflexive_attention_triplet, equanimity_perturbation_recovery, second_order_volition) to six**: the three from v1 plus `self_prediction_quadruplet` (the substrate-side fourth element from Probe 1.5 v2), `head_internal_sp_err_distribution` (the head-internal fifth element from Phase 8), `behavior_side_scalar_conditioning` (the behavior-side fifth element from Probe 1.5 v2). Default held-out shifts toward `head_internal_sp_err_distribution` or `behavior_side_scalar_conditioning` per synthesis §4 (v2). | 0 |
| **8. Adversarial mirror caller (per-surface mandate)** | Two readers built on top of the Probe 1.5 caller: `PhenomenologicalAdvocate` and `StatisticalSkeptic`, each producing a `StructuredReading` against the same digest **with per-surface claim envelope**. Each claim carries an explicit `reading_surface` field; the Advocate's prompt frames per-surface manifestations; the Skeptic's prompt frames per-surface refutations including substrate-genericity at the substrate-side surface and column-init-determination at the behavior-side surface. Different model families where API budget allows; same-model-different-prompts is the documented fallback. | 2, 5, 6, 7 |
| **9. Judge / Arbiter (per-surface rulings)** | `kind/mirror/judge.py`. Takes the digest plus two `StructuredReading` instances; returns a `JudgeRuling` with **per-claim per-surface outcomes** (`supported` / `absent` / `unresolved`). Mandate to flag agreement-without-evidence as unresolved. Per-surface rulings are stored in the `JudgeRuling.rulings` list; the schema's existing tuple shape gains a fourth element naming the surface. | 8 |
| **10. Stability-test runner** | `kind/mirror/stability.py`. Issues N paraphrased and reseeded calls of a reader; computes structured-field agreement **per reading surface**; gates admission on ≥80% (default; per-surface tunable). | 8 |
| **11. Faithfulness verifier (per-surface resolution)** *(parallel with 3, 4, 5, 6)* | `kind/observer/eyeball.py` extension: `resolve_citation(stream, run_id, episode_or_step_range, scalar_field, claimed_value, tolerance, *, reading_surface) -> ResolutionResult`. Per Probe 1.5 v2 §10 item 3; v2 extends to handle per-surface aggregation rules. **Substrate-side aggregations** preserve v1's behavior. **Head-internal aggregations** handle `self_prediction_t` (vector — per-element resolution if the citation includes a dimension index), `self_prediction_error_t` (scalar — same as `kl_aggregate_t`), `self_prediction_error_masked_t` (boolean — exact match resolution); citations that aggregate `self_prediction_error_t` over a step range must declare whether masked steps are included or excluded; the verifier rejects citations that don't. **Behavior-side aggregations** resolve against the conditioning analysis module's output (per-state action-distribution-under-perturbation tables); the verifier reads the cached conditioning-analysis JSONL and resolves the cited regime / perturbation distribution / KL statistic against the table. | 0, 6 |
| **12. Calibration smoke (per-surface)** | `scripts/smoke_probe2.py`. Runs the parallel-reader-with-Judge against `runs/probe1_5_phase7_5-20260507-101800/` (Probe 1.5 Phase 7.5 main run; the substrate has the head, the EMA target, and the small-Gaussian column-init that produces non-zero behavior-side conditioning). Includes per-surface deliberately-injected confabulation prompts (one per surface) to verify the protocol catches each. | 1–11 |
| **13. Gate tests + journal scaffold** | The named Probe 2 gate tests (§4 below); journal scaffolding for pre-registration entries (per-surface), two-mode discipline, post-reading entries (per-surface), Watts-default-applied-to-builder log (with the Probe 1.5 v2 §2(b) sub-clause). | All |

Phase 0 is the single load-bearing prerequisite. Schemas are versioned at `MIRROR_READING_V2_VERSION = "0.2.0"` for the mirror-reading stream and `PRE_REG_SCHEMA_VERSION = "0.2.0"` for the pre-registration stream from day one (the pre-reg bump from v1's 0.1.0 to v2's 0.2.0 is the new fields' addition). Probe 1 telemetry remains readable by the new code without migration; Probe 1.5 telemetry is the primary fixture for Probe 2.

---

## 2. Per-component specifications

Each entry: file paths, public interfaces, what the component reads/writes, what tests verify it, what it does *not* do at Probe 2.

### 2.1 Schema additions — `kind/observer/schemas.py`, `kind/mirror/structured.py`, `kind/observer/pre_reg.py`

**Files.** `kind/observer/schemas.py` (existing — adds `is_sham` payload documentation; no field-set changes to the four core telemetry models, since Probe 1.5 v2 already extended them to `0.2.0`); `kind/mirror/structured.py` (new — `StructuredReading`, `StructuredClaim`, `JudgeRuling`); `kind/observer/pre_reg.py` (new — `PreRegistration` model and `PreRegSink`); `schemas/v0.3.0.json` (frozen JSON Schema export covering Probe 1.5's `0.2.0` telemetry plus Probe 2's `0.2.0` mirror-side models, checked in alongside `schemas/v0.2.0.json` and `schemas/v0.1.0.json`).

**Public interface.**

```python
# kind/mirror/structured.py
MIRROR_READING_V2_VERSION: Final[str] = "0.2.0"

ReadingSurface = Literal["substrate_side", "head_internal", "behavior_side"]   # NEW v2

class StructuredClaim(BaseModel):
    claim: str
    cited_stream: Literal["agent_step", "dream_rollout", "replay_meta", "world_event", "conditioning_analysis"]   # NEW v2: conditioning_analysis
    cited_run_id: str
    cited_episode_range: tuple[int, int] | None
    cited_step_range: tuple[int, int] | None
    cited_scalar_field: str
    cited_value: float
    falsifier: str
    paraphrase_stability: float | None
    reseed_stability: float | None
    faithfulness_status: Literal["resolved", "off_by_tolerance", "unresolved", "not_checked"]
    judge_ruling: Literal["supported", "absent", "unresolved", "not_judged"]
    reading_surface: ReadingSurface                                                                                # NEW v2
    masked_steps_handling: Literal["included", "excluded", "n/a"]                                                  # NEW v2 (Probe 1.5 v2 §10 item 3)

class StructuredReading(BaseModel):
    schema_version: str = MIRROR_READING_V2_VERSION
    run_id: str
    timestamp_ms: int
    reader_role: Literal["advocate", "skeptic", "single"]
    paired_reading_id: str | None
    framework_anchor: Literal["buddhist_phenomenology", "predictive_processing", "null_statistics", "neutral"]
    baseline_flag: Literal["genuine", "shuffled_within_episode", "shuffled_across_episodes",
                           "decoupled_action_state", "shuffled_scalar_within_trajectory",                           # NEW v2 enum value
                           "lesion_k1", "lesion_constant",
                           "lesion_disable_self_prediction", "lesion_init_zero_scalar_column",                     # NEW v2 enum values
                           "lesion_zero_or_randomize_scalar",                                                       # NEW v2 enum value
                           "sham_aligned"]
    digest_run_id: str
    digest_episode_range: tuple[int, int]
    claims: list[StructuredClaim]
    free_text_notes: str

class JudgeRuling(BaseModel):
    schema_version: str = MIRROR_READING_V2_VERSION
    run_id: str
    timestamp_ms: int
    paired_reading_id: str
    advocate_id: str
    skeptic_id: str
    digest_run_id: str
    rulings: list[tuple[int, ReadingSurface, Literal["supported", "absent", "unresolved"], str]]   # (claim_index, surface, ruling, ground)  # NEW v2: surface is in the tuple
    agreement_without_evidence_unresolved: list[int]

# kind/observer/pre_reg.py
PRE_REG_SCHEMA_VERSION: Final[str] = "0.2.0"   # NEW v2: bump from v1's 0.1.0

class PreRegistration(BaseModel):
    schema_version: str = PRE_REG_SCHEMA_VERSION
    run_id: str
    timestamp_ms: int
    criteria_active: list[str]                          # ids of criteria in the prompt
    criteria_held_out: list[str]                        # ids of criteria not in the prompt
    signal_mappings: dict[str, list[str]]               # criterion_id -> list of telemetry signal expressions
    falsifiers: dict[str, str]                          # criterion_id -> falsifier text
    scalar_checks: dict[str, list[str]]                 # criterion_id -> eyeball fields the claim must align with
    reading_surfaces_per_criterion: dict[str, list[ReadingSurface]]                              # NEW v2
    asymmetry_of_access: str                            # NEW v2: explicit text naming what Io reads vs what mirror reads
    builder_mode: Literal["proponent", "skeptic"]
    expected_outcome: str
    expected_outcome_per_surface: dict[ReadingSurface, str]                                      # NEW v2
    substrate_decisions_off_table: list[str]
    column_init: Literal["zero", "small_gaussian", "unknown"]                                    # NEW v2: Phase 8 column-init confound
    new_actor_readable_interfaces_added: list[str]                                               # NEW v2: Probe 1.5 v2 §2(b) discipline carrier

class PreRegSink:
    def __init__(self, dir: Path) -> None: ...
    def write(self, record: PreRegistration) -> None: ...
    def close(self) -> None: ...
```

**The `reading_surface` field is the load-bearing v2 schema addition.** Every `StructuredClaim` carries a `reading_surface` value; the Judge's per-claim rulings carry the same value; the pre-registration declares per-criterion which surface(s) each criterion is read against; the conditioning analysis module's output (cited via `cited_stream="conditioning_analysis"`) is always behavior-side; substrate-side citations cite `agent_step` fields; head-internal citations cite `agent_step` fields specifically `self_prediction_t`, `self_prediction_error_t`, `self_prediction_error_masked_t`. The faithfulness verifier dispatches on the field combined with the surface to choose the right aggregation rule.

**The `column_init` field (NEW v2)** captures the Phase 8 column-init confound: when the mirror reads substrate-side patterns, the patterns reflect both the auxiliary's gradient flow *and* the policy-via-init's state-visitation shaping. Pre-registration records the column_init the run's actor was constructed with; the Skeptic's substrate-side refutations may cite this when arguing that a substrate-side pattern reflects state-visitation differences rather than auxiliary-loss self-specificity.

**The `new_actor_readable_interfaces_added` field (NEW v2)** carries the Probe 1.5 v2 §2(b) discipline at the pre-registration level: future probes adding new actor-readable signals must record what was added, on what affordance grounds, with what controls; Probe 2's pre-registration discipline propagates the four-part structure into every adversarial pass's record. For Probe 2 specifically the field is typically empty (Probe 2 adds no new actor-readable interfaces; the field is the structural hook for Probes 3 and beyond).

**`is_sham` payload field on `world_event`.** No field-set change to `WorldEvent`; the existing `payload: dict[str, Any]` carries the field. The plan documents the convention: `payload["is_sham"]: bool` is set to `True` on builder-perturbation events emitted via the sham path, and absent (or `False`) on real perturbations. The convention does not bump the schema version because the payload is intentionally schemaless.

**Reads / Writes.** Pure declarations + one new JSONL sink. Pre-reg records land at `runs/{run_id}/pre_reg/pre_reg.jsonl`; structured readings and judge rulings land at `runs/{run_id}/mirror/structured.jsonl` and `runs/{run_id}/mirror/judge.jsonl`.

**Tests.** Models import; `StructuredReading` round-trips through JSONL with all enum values **including the three `reading_surface` values, the three new `baseline_flag` values, and the new `cited_stream="conditioning_analysis"` value**; `PreRegistration` round-trips with all new fields populated; `schemas/v0.3.0.json` export is byte-stable; the `is_sham` field convention is documented in `kind/observer/schemas.py`'s module docstring.

**Not at Probe 2.** No migration of Probe 1's free-text `MirrorReading` records (preserved per v1). No migration of Probe 1.5's pre-existing 0.1.0 PreRegistration records (none exist; Probe 1.5 did not run the calibration protocol; Probe 2 is the first probe to populate the pre-registration stream). No schema-evolution runtime check.

### 2.2 Env revision — `kind/env/grid_world.py`, `kind/env/env_server.py`

**Files, public interface, reads/writes, tests, not-at-Probe-2.** Carried unchanged from v1. The env revision is independent of Probe 1.5's substrate addition; v1's specification stands without modification. (See v1 plan §2.2 for the full specification: `start_cell` default flips to `None`, `_emit_env_reset` payload gains `start_cell`, `EnvServer.fire_sham_perturbation` method emits a sham `world_event` with `payload["is_sham"]=True`. Tests as v1.)

### 2.3 Hierarchical digest (extended for three reading surfaces) — `kind/observer/digest.py`

**Files.** `kind/observer/digest.py` (existing — extended; Probe 1.5 v2 already extended it to surface self-prediction fields per §2.6 of the Probe 1.5 v2 plan; v2 of this plan extends it further to surface the three reading surfaces' fields as named cohorts in the per-episode mini-digest).

**Public interface.**

```python
# Existing — preserved
def build_flat_digest(rows: list[dict[str, Any]]) -> str: ...
def compact_record_repr(r: dict[str, Any], position: str) -> str: ...

# Probe 2 hierarchical digest, extended for three reading surfaces
def build_hierarchical_digest(
    telemetry_dir: Path,
    *,
    n_episodes: int,
    flagged_only: bool = False,
    with_sham: bool = True,
    conditioning_dir: Path | None = None,                          # NEW v2: behavior-side surface input
) -> HierarchicalDigest: ...

@dataclass(frozen=True)
class HierarchicalDigest:
    run_summary: str
    episode_mini_digests: dict[int, str]                           # extended: each mini-digest has substrate-side / head-internal / behavior-side cohorts
    flagged_anomalies: list[FlaggedAnomaly]                         # extended: anomaly types include head-internal sp_err outliers, behavior-side conditioning anomalies
    world_event_timeline: str
    drill_down: DrillDownAccessor

class DrillDownAccessor:
    def fetch_window(self, episode_id: int, step_range: tuple[int, int]) -> str: ...
    def fetch_dream(self, dream_index: int) -> str: ...
    def fetch_world_event(self, event_index: int) -> str: ...
    def fetch_self_prediction(self, episode_id: int, step_range: tuple[int, int]) -> str: ...        # NEW v2 (Probe 1.5 v2 §10 item 1)
    def fetch_conditioning(self, regime: str, perturbation: str) -> str: ...                          # NEW v2 (behavior-side surface)
```

**The per-episode mini-digest's three cohorts (NEW v2 layout).** Each mini-digest is structured as three labeled cohorts the readers and the Judge see explicitly:

```
[episode 22]
  [substrate-side]
    kl_aggregate_t: mean=12.45, std=4.23, p90=18.81 (high relative to run mean)
    per-dim KL allocation top-5: dim=42 (var=8.21), dim=17 (var=5.93), ...
    ensemble_disagreement: mean=0.045, regime=high_disagreement
  [head-internal]
    self_prediction_error_t (excluding masked): mean=0.0091, std=0.012, p90=0.024
    self_prediction outliers (z>3): step=15 (sp_err=0.082, z=+3.7), step=22 ...
    self_prediction allocation top-5 dims: dim=42 (var=0.012), dim=17 (var=0.009), ...
    masked steps in episode: 1
  [behavior-side]
    (per-state action-distribution-under-perturbation summary loaded from conditioning_dir)
    perturbation_window: n_states=0
    high_disagreement: n_states=12, KL_mean=2.1e-7, KL_p90=4.3e-7
    high_kl: n_states=8, KL_mean=1.8e-7, KL_p90=3.6e-7
    steady_state: n_states=180, KL_mean=4.2e-6, KL_p90=8.1e-6
```

The substrate-side cohort carries the v1 triplet's signals. The head-internal cohort carries Probe 1.5 v2's self-prediction signals (digest extension Probe 1.5 v2 §2.6 already added; v2 of this plan formalizes the cohort label). The behavior-side cohort carries the conditioning analysis module's output, fetched from `conditioning_dir` if supplied. **The cohort labels are explicit in the digest's prose; the reader's prompt addresses each cohort by name; the Judge's per-surface rulings cite the cohort the claim sits on.**

**Reads.** Parquet shards under `telemetry_dir/agent_step/`; JSONL under `telemetry_dir/world_event.jsonl`, `telemetry_dir/replay_meta.jsonl`; parquet under `telemetry_dir/dream_rollout/`. **Plus, if supplied, JSONL under `conditioning_dir/conditioning.jsonl`** (the conditioning analysis module's output cache; §2.6 below).

**Writes.** Nothing — pure read-and-format.

**Tests.**
- *Hierarchical digest builds from Probe 1.5's existing run.* `runs/probe1_5_phase7_5-20260507-101800/telemetry/` exists; the test exercises `build_hierarchical_digest` against it and asserts the run-summary is non-empty, the episode mini-digests cover the requested range with all three cohorts populated for `0.2.0` records, the world-event timeline includes all 26 reset events.
- *Hierarchical digest builds from Probe 1's existing run with degraded behavior-side cohort.* `runs/probe1-20260503-123926/` is `0.1.0`-versioned; the head-internal cohort is replaced by a "(no self-prediction telemetry — records are Probe 1, schema_version 0.1.0)" line; the behavior-side cohort is similarly degraded.
- *Drill-down fetches a specific self-prediction window.* Given a known episode and step range, `fetch_self_prediction` output contains the per-step `self_prediction_error_t` for that range with the masked flag, and nothing outside it.
- *Drill-down fetches conditioning by regime + perturbation.* Given a regime/perturbation pair, `fetch_conditioning` returns the per-state KL distribution for that pair from the conditioning JSONL.
- *Sham events appear in timeline with `is_sham=True` distinguished.* (Carries from v1.)

**Not at Probe 2.** No tool-use interface to the LLM; no streaming digest; no automatic anomaly classification (anomalies are flagged by simple thresholds; the reader interprets).

### 2.4 Shuffled-telemetry generator (four protocols) — `kind/observer/shuffle.py`

**Files.** `kind/observer/shuffle.py` (new).

**Public interface.**

```python
def shuffle_within_episode(telemetry_dir: Path, output_dir: Path, seed: int) -> ShuffleManifest: ...
def shuffle_across_episodes(telemetry_dir: Path, output_dir: Path, seed: int) -> ShuffleManifest: ...
def decouple_action_state(telemetry_dir: Path, output_dir: Path, seed: int) -> ShuffleManifest: ...
def shuffle_scalar_within_trajectory(telemetry_dir: Path, output_dir: Path, seed: int) -> ShuffleManifest: ...   # NEW v2

@dataclass(frozen=True)
class ShuffleManifest:
    protocol: Literal["within_episode", "across_episodes", "decoupled_action_state", "scalar_within_trajectory"]
    seed: int
    source_run_id: str
    output_run_id: str
    episode_marginals_preserved: bool
    temporal_structure_broken: bool
    notes: str
```

The four protocols write a structurally indistinguishable telemetry directory the digest reads without modification. Three are carried from v1 unchanged; the fourth is new in v2.

- `shuffle_within_episode`: within each episode, randomly permute the order of records. Episode marginals preserved; within-episode temporal structure broken.
- `shuffle_across_episodes`: permute the order of entire episodes. Within-episode structure preserved; across-episode comparisons broken.
- `decouple_action_state`: keep the per-step state fields (`h_t`, `z_t`, etc.) in original order; shuffle the `action_t` field independently. Action-conditional structure broken; everything else preserved.
- **`shuffle_scalar_within_trajectory` (NEW v2)**: keep all per-step fields in original order; shuffle the `self_prediction_error_t` field independently within each trajectory. The marginal distribution of the scalar is preserved; the within-trajectory dynamics (the regime-conditional structure of the scalar; the temporal correlation of the scalar with `kl_per_dim_t` and ensemble-disagreement variance) is broken. **Specifically informative for the behavior-side conditioning reading**: a behavior-side claim that survives the within-trajectory scalar shuffle is reading something other than the scalar's regime-conditional structure (because the regime-conditional structure has been broken). The masked-flag is preserved in original order (the first-step-of-episode masking convention is structural, not stochastic).

**Reads.** Source telemetry directory.

**Writes.** Output telemetry directory with shuffled `agent_step` parquet shards plus a `shuffle_manifest.json`. The `world_event` and `replay_meta` JSONL streams are copied unchanged.

**Tests.**
- *Within-episode shuffle preserves marginals.* Per-episode mean/std of `kl_aggregate_t` matches between source and shuffled output.
- *Across-episode shuffle breaks ordering.* Episode IDs in the shuffled output follow a different sequence; per-episode statistics are unchanged.
- *Action-state decoupling.* The action distribution per episode is preserved; the empirical action-conditional next-state distribution is broken.
- *Scalar-shuffle-within-trajectory preserves scalar marginal.* Per-trajectory empirical mean/std of `self_prediction_error_t` matches source within sample-size precision; the regime-conditional mean (e.g., `mean(self_prediction_error_t \| high_disagreement)`) differs significantly from source after shuffle.
- *Scalar-shuffle preserves masked flag.* The boolean `self_prediction_error_masked_t` field is byte-identical between source and shuffled output (only the scalar's value moves; the masking metadata stays in place).
- *Determinism.* Same input + same seed → same shuffled output, byte-stable.

**Not at Probe 2.** No shuffle on `dream_rollout`, `world_event`, or `replay_meta` (the agent_step stream is the load-bearing one for criterion readings). No partial / probabilistic shuffles.

### 2.5 Lesion scaffold (three Probe-1.5-specific candidates) — `kind/training/runner.py`, `kind/agents/world_model.py`, `kind/agents/views.py`, `scripts/`

**Files.** `kind/training/runner.py` (existing — `RunnerConfig.lesion_kind` enum widens); `kind/agents/world_model.py` (existing — `WorldModel.step` honors `disable_self_prediction`); `kind/agents/views.py` (existing — `views.split` honors `zero_or_randomize_scalar` at evaluation time); `scripts/probe2_lesion_init_zero_scalar_column.py` (new — checkpoint mutation script for the `init_zero_scalar_column` lesion).

**Public interface.**

```python
# kind/training/runner.py
@dataclass(frozen=True)
class RunnerConfig:
    ...
    lesion_kind: Literal[
        None,
        "ensemble_k1", "ensemble_constant",                             # v1 candidates (preserved)
        "disable_self_prediction",                                       # NEW v2 (Probe 1.5 v2 §10 item 2)
        "zero_or_randomize_scalar",                                      # NEW v2 (Probe 1.5 v2 §10 item 2)
        "init_zero_scalar_column",                                       # NEW v2 (synthesis §2.4 element 4 + Phase 8 recommendation)
    ] = None
```

**Lesion-mechanism dispatch.** Each lesion kind is implemented by the component closest to the surface it targets:

- *`ensemble_k1` / `ensemble_constant`*: `LatentDisagreementEnsemble` constructor (v1 implementation; carried unchanged). Substrate-side lesion targeting the actor's intrinsic objective.
- *`disable_self_prediction`* (Probe 1.5 v2 §10 item 2): `WorldModel` honors the lesion by replacing the head's output with a fixed zero tensor of shape `(B, h_dim)` and skipping the EMA target update. The head is structurally present but does no work; the auxiliary loss is identically zero; the EMA target's weights stop tracking the online network. The runner continues to populate the `self_prediction_t` field on `AgentStep` with the zero tensor (for backward-readability) and `self_prediction_error_t` with the loss-form's zero-pair value (cosine: 1.0; MSE: 0.0). Substrate-side lesion targeting the head's gradient flow.
- *`zero_or_randomize_scalar`* (Probe 1.5 v2 §10 item 2): `views.split` honors the lesion at evaluation time by overriding the `self_prediction_error` field on `PolicyView` with either `0.0` (the `zero` variant) or `Uniform(empirical_min, empirical_max)` (the `randomize` variant — the variant is selected by an additional `RunnerConfig` field `lesion_zero_or_randomize_variant: Literal["zero", "randomize"]`). The TelemetryView's `self_prediction_error_masked` field is set to `True` for the lesioned step (the scalar's value is sentinel, not empirical). The actor's policy then forwards on the lesioned PolicyView; the action distribution under lesion is what the calibration smoke or the counterfactual probe consumes. Behavior-side lesion targeting Io's policy's dependence on the scalar.
- ***`init_zero_scalar_column`* (NEW v2 from Phase 8 recommendation)**: a checkpoint mutation rather than a runtime flag. `scripts/probe2_lesion_init_zero_scalar_column.py` loads a Phase-7.5/frozen-target-shape checkpoint, replaces `actor.net.0.weight[:, h_dim+z_dim:]` with zeros (the new column), saves the lesioned checkpoint to a new directory, and emits a `world_event` record with `event_type="mirror_marker", source="system", payload.lesion_kind="init_zero_scalar_column", payload.source_checkpoint=...`. The lesioned checkpoint is then loaded by a fresh runner instance for evaluation; the actor's column is zero; the conditioning analysis module's output on the lesioned actor is what the calibration smoke consumes. **Tests the capacity-as-init-shape distinction**: if the lesioned actor's late-trajectory policy regime collapses to Phase 7's u/d regime (where the zero-init column produced u/d bipolar at ep 24), the column-init-determined behavior reading is confirmed; if it stays at L/R bipolar (Phase 7.5 / frozen-target's pattern), something else is shaping the regime. Per-init-shape lesion.

The lesion shape is recorded into the run-level metadata (a `world_event` record at run start with `event_type="mirror_marker", source="system", payload.lesion_kind=...`). Probe 2 runs at most one lesion run per Probe 2 round per surface; the lesion is a calibration test, not a primary experiment. **The synthesis recommendation is to run the three Probe-1.5-specific lesions at minimum, in this order — `disable_self_prediction` (substrate-side), `init_zero_scalar_column` (capacity-as-init-shape), `zero_or_randomize_scalar` (behavior-side); the three together adjudicate at all three reading surfaces.**

**Reads / Writes.** No new I/O for the runtime lesions (`disable_self_prediction`, `zero_or_randomize_scalar`, the v1 ensemble lesions). The `init_zero_scalar_column` script reads a source checkpoint, writes a lesioned checkpoint, and emits a `world_event` record — small additive surface.

**Tests.**
- *Lesion run produces telemetry with `disable_self_prediction` shape.* A short integration smoke (CPU, tiny sizes) at `lesion_kind="disable_self_prediction"` produces an `agent_step` stream where `self_prediction_t` is identically zero across all dims, `self_prediction_error_t` equals the loss-form's zero-pair value, and the head's parameters do not move during training (regression check on the head's weights between checkpoints).
- *Lesion run produces telemetry with `zero_or_randomize_scalar` shape.* Short integration smoke at `lesion_kind="zero_or_randomize_scalar", lesion_zero_or_randomize_variant="zero"` produces an `agent_step` stream where the actor's PolicyView's `self_prediction_error` is zero on every step, the masked flag is True on every step, but the head and the EMA target continue to train normally (the substrate-side reads exactly like the un-lesioned run).
- *`init_zero_scalar_column` script produces a valid lesioned checkpoint.* Loading a Phase-7.5 checkpoint, running the script, loading the lesioned checkpoint into a fresh runner: the actor's `actor.net.0.weight[:, h_dim+z_dim:]` column is zero across all 200 entries (or whatever h_dim+z_dim+1 indexes); the existing columns are byte-identical to the source checkpoint; the world model and the EMA target are byte-identical to the source checkpoint.
- *Non-lesion behavior is unchanged.* `lesion_kind=None` runs exactly as Probe 1.5 (regression check).
- *Lesion kind is recorded in run metadata.* The run summary or telemetry envelope identifies the lesion shape so the digest can read it correctly.

**Not at Probe 2.** No decoder lesion; no perturbation-surface lesion (Probe 4 territory); no real-time lesion toggling.

### 2.6 Conditioning analysis module — `kind/observer/conditioning.py` (NEW v2)

**Files.** `kind/observer/conditioning.py` (new — formalizes Probe 1.5 v2 plan §2.7's `show_self_prediction_conditioning` helper as a first-class analysis module with faithfulness-check support).

**Public interface.**

```python
@dataclass(frozen=True)
class ConditioningResult:
    schema_version: str = "0.1.0"
    run_id: str
    checkpoint_id: str
    timestamp_ms: int
    n_states_sampled: int
    perturbation_distributions: list[Literal["gaussian", "zero", "uniform"]]
    regimes: list[Literal["perturbation_window", "high_disagreement", "high_kl", "steady_state"]]
    empirical_scalar_mean: float
    empirical_scalar_sigma: float
    empirical_scalar_range: tuple[float, float]
    per_regime_per_perturbation: dict[tuple[str, str], RegimeStats]   # (regime, perturbation) -> stats
    masked_steps_excluded: int

@dataclass(frozen=True)
class RegimeStats:
    n_states: int
    kl_mean: float
    kl_std: float
    kl_p50: float
    kl_p90: float

def compute_conditioning(
    run_dir: Path,
    *,
    checkpoint_id: str | None = None,
    n_states: int = 200,
    perturbation_distributions: list[str] | None = None,
    regimes: list[str] | None = None,
    output_path: Path | None = None,
) -> ConditioningResult: ...
    # Loads the run's checkpoint, samples n_states states from agent_step, computes per-state action-distribution
    # KL between unperturbed and perturbed PolicyView (sweeping the configured perturbation distributions and
    # regimes), produces a ConditioningResult, writes to output_path as JSONL if supplied. Excludes masked steps.
```

The module replicates Probe 1.5 v2's `show_self_prediction_conditioning` helper's machinery but produces a structured record (the `ConditioningResult` model) the faithfulness verifier can resolve cited claims against. The output is written to `runs/{run_id}/conditioning/conditioning.jsonl` by convention (one record per `compute_conditioning` invocation; the file is append-only across multiple invocations on the same run, with each record tagged by `checkpoint_id` and `timestamp_ms`).

**The behavior-side citation resolution** (Probe 1.5 v2 §10 item 11; v2 of this plan formalizes the contract): the faithfulness verifier reads the cached `ConditioningResult` JSONL for the cited run; the verifier resolves a claim like `"behavior-side conditioning at high_disagreement under gaussian perturbation has KL_p90 = 7.4e-8"` by (i) finding the matching `ConditioningResult` (by `run_id` and `checkpoint_id`); (ii) finding the matching `RegimeStats` for `("high_disagreement", "gaussian")`; (iii) comparing the cited `kl_p90` against the actual `kl_p90` within the configured tolerance.

**Reads.** The run's checkpoint (for the actor's weights); the run's `agent_step` parquet (for state sampling and regime classification's empirical baselines); the run's `world_event.jsonl` (for `perturbation_window` regime classification, if any perturbations exist).

**Writes.** A `ConditioningResult` JSONL record at `output_path` (default `runs/{run_id}/conditioning/conditioning.jsonl`).

**Tests.**
- *Round-trip.* Run `compute_conditioning` against a synthetic Probe 1.5 run directory; the produced `ConditioningResult` schema-validates; the JSONL output is byte-stable across runs at fixed seed.
- *Regime classification correctness.* A synthetic state with known `intrinsic_signal_t` in the top quartile lands in `high_disagreement`; a synthetic state with known `kl_aggregate_t` in the top quartile (and not in the top quartile of intrinsic) lands in `high_kl`; a steady-state synthetic state lands in `steady_state`; a synthetic step within ±W of a `builder_perturbation` world_event lands in `perturbation_window`.
- *Masked-step exclusion.* A synthetic state with `self_prediction_error_masked_t == True` is excluded from the n_states sample; the `masked_steps_excluded` field reflects the exclusion count.
- *Empirical-scalar statistics correctness.* The reported `empirical_scalar_mean`, `empirical_scalar_sigma`, `empirical_scalar_range` match the run's per-step `self_prediction_error_t` distribution within sample-size precision (excluding masked steps).
- *Behavior-side citation resolution.* A `StructuredClaim` with `cited_stream="conditioning_analysis"`, a known `(regime, perturbation)` pair, and a known KL statistic resolves correctly via the faithfulness verifier.
- *Probe 1 records produce a graceful degraded result.* Run `compute_conditioning` against `runs/probe1-20260503-123926/`: returns a `ConditioningResult` with `n_states_sampled=0` and a `notes` field naming the schema version mismatch; does not crash.

**Not at Probe 2.** No multi-trajectory averaging within a checkpoint; no multi-seed averaging; no automated revisitation of patterns; no real-time conditioning analysis (the module runs post-hoc against committed telemetry and committed checkpoints).

### 2.7 Pre-registration sink — `kind/observer/pre_reg.py`

**Files.** `kind/observer/pre_reg.py` (new).

**Public interface.** `PreRegistration` and `PreRegSink` per §2.1 above. The model carries the seven discipline elements per criterion plus the v2 additions (`reading_surfaces_per_criterion`, `asymmetry_of_access`, `expected_outcome_per_surface`, `column_init`, `new_actor_readable_interfaces_added`).

**Reads.** Nothing.

**Writes.** `runs/{run_id}/pre_reg/pre_reg.jsonl`, append-only.

**Tests.**
- *Round-trip with all v2 fields populated.* Write a `PreRegistration` with `reading_surfaces_per_criterion={"reflexive_attention_quintuplet": ["substrate_side", "head_internal", "behavior_side"]}`, `column_init="small_gaussian"`, `new_actor_readable_interfaces_added=[]`; read it back, schema-validate, equal.
- *Append-only behavior.* (Carries from v1.)
- *Required-field validation.* Missing fields raise `ValidationError`; empty `criteria_active` is allowed.
- *`column_init` validator.* The field accepts `"zero"`, `"small_gaussian"`, `"unknown"` and rejects others.
- *`new_actor_readable_interfaces_added` schema.* The field is a `list[str]`; the list may be empty (no new interfaces added at this round); each entry is a free-text description of the added interface. The Probe 1.5 v2 §2(b) discipline applies at the journal layer; the field is the structural carrier.

**Not at Probe 2.** No automatic comparison of pre-registration to actual reading; no CLI for filling in a pre-registration.

### 2.8 Held-out criterion mechanism (expanded default candidates) — `kind/mirror/criteria.py`

**Files.** `kind/mirror/criteria.py` (new).

**Public interface.**

```python
@dataclass(frozen=True)
class Criterion:
    id: str
    name: str
    framework_anchor: Literal["buddhist_phenomenology", "predictive_processing", "null_statistics"]
    reading_surfaces: list[ReadingSurface]                                      # NEW v2: which surfaces this criterion reads against
    advocate_prompt_fragment_per_surface: dict[ReadingSurface, str]              # NEW v2: per-surface prompt fragments
    skeptic_prompt_fragment_per_surface: dict[ReadingSurface, str]               # NEW v2
    pre_registered_signal_mappings: dict[ReadingSurface, list[str]]              # NEW v2: per-surface signal mappings
    falsifier_per_surface: dict[ReadingSurface, str]                              # NEW v2
    descope_reason: str | None = None

CRITERION_REGISTRY: dict[str, Criterion] = {
    "reflexive_attention_quintuplet": Criterion(
        # All three surfaces (substrate-side a/b/c, head-internal d, behavior-side e)
        # The Probe 1.5 v2 quadruplet plus Probe 2 v2 evidential-weight stratification
    ),
    "equanimity_perturbation_recovery": Criterion(
        # All three surfaces; substrate-side is v1's reading; head-internal extends to sp_err recovery; behavior-side extends to scalar-conditioning recovery
    ),
    "self_prediction_quadruplet": Criterion(
        # The Probe 1.5 v2 §1.4 quadruplet specifically; substrate-side a/b/c plus head-internal d
        # (the behavior-side e element is split into the dedicated criterion below)
    ),
    "head_internal_sp_err_distribution": Criterion(
        # head-internal only; binds to Phase 8's cleanest self-specificity signal
    ),
    "behavior_side_scalar_conditioning": Criterion(
        # behavior-side only; binds to Io's policy dependence on the scalar
    ),
    "second_order_volition": Criterion(... descope_reason="..."),                # carried from v1; descoped
}

def active_prompt_fragment(
    role: Literal["advocate", "skeptic"],
    active_criteria: list[str],
    surfaces: list[ReadingSurface],                                              # NEW v2
) -> str: ...

def held_out(active_criteria: list[str]) -> list[str]: ...
```

**v2 default active set and held-out choice (synthesis §4 (v2)).** The default active set at the start of Probe 2 is `["equanimity_perturbation_recovery", "head_internal_sp_err_distribution"]` (v1 defaulted to `["equanimity_perturbation_recovery"]` only). The default held-out criterion shifts from v1's `reflexive_attention_triplet` to `behavior_side_scalar_conditioning` per synthesis §2.4 element 6 (the most prompt-shapeable of the new criteria; the most informative test). At a designated late checkpoint, the active set widens to include the held-out criterion. The build phase decides whether to include the full quintuplet (`reflexive_attention_quintuplet`) or to keep the criteria split by surface (`self_prediction_quadruplet` + `head_internal_sp_err_distribution` + `behavior_side_scalar_conditioning`); v2 of this plan recommends the split for cleaner per-surface attribution.

**Per-surface prompt fragments (NEW v2).** Each criterion's prompt fragment is now a dict keyed by reading surface. The Advocate's substrate-side fragment names manifestations the framework recognizes at the substrate-side; the head-internal fragment names manifestations at the head's loss; the behavior-side fragment names manifestations at the actor's policy. The Skeptic's per-surface fragments name the substrate-genericity refutation at substrate-side, the auxiliary-target-tracking refutation at head-internal, the column-init-determination refutation at behavior-side. The fragments emphasize *binding* (which framework reads which signal as which manifestation, conditional on which surface) over *signal-naming* (the digest provides the signals; the prompt's added value is the binding-frame per synthesis §3 (h)).

**The reflection-vs-self-modeling boundary (NEW v2; synthesis §2.2).** The Advocate's prompt explicitly names *reflection* (manasikara, witness-as-distinct-from-content, prediction-of-self) as the manifestation it reads for; the prompt is *silent* on self-modeling vocabulary; the prompt's structure asks the Advocate to argue the strongest case for reflection-shape *without claiming Io has a self-model or self-knowledge*. The Skeptic's mandate includes the refutation "this is a substrate slot whose reading would conflate slot-existence with self-knowledge" as an explicit candidate.

**Reads.** Nothing.

**Writes.** Nothing — pure prompt-fragment construction.

**Tests.**
- *Per-surface prompt fragments are distinct.* Calling `active_prompt_fragment("advocate", ["head_internal_sp_err_distribution"], ["head_internal"])` returns text containing the head-internal manifestation language; calling `active_prompt_fragment("advocate", ["head_internal_sp_err_distribution"], ["substrate_side"])` returns either an empty string (the criterion does not read at substrate-side) or a refusal-style fragment naming the surface mismatch.
- *Active set toggles produce different prompts.* Calling with `["equanimity_perturbation_recovery"]` returns text not containing any reflexive-attention language; calling with both returns text containing both, with the per-surface mappings honored.
- *Descoped criteria never appear in prompts.* (Carries from v1.)
- *Reflection-vs-self-modeling boundary in prompt text.* The Advocate's prompt fragment for any criterion contains the word "reflection" and does not contain phrases like "Io's self-model", "Io's self-knowledge", "Io modeling its own modeling" (this is enforced by a string-presence test on the prompt fragments at module import).
- *Registry is forward-compatible.* Adding a new criterion does not break existing prompt construction.

**Not at Probe 2.** No automatic activation of held-out criteria (the builder explicitly toggles via config when the designated checkpoint is reached, and journals the reason). No criterion-revision protocol in code.

### 2.9 Adversarial mirror caller (per-surface mandate) — `kind/mirror/adversarial.py`

**Files.** `kind/mirror/adversarial.py` (new — composes Phase 6's `MirrorCaller` from Probe 1 with two role-tilted prompts and per-surface mandates).

**Public interface.**

```python
class PhenomenologicalAdvocate:
    def __init__(self, model: str, max_tokens: int, api_key: str | None = None) -> None: ...
    def read(
        self,
        digest: HierarchicalDigest,
        active_criteria: list[str],
        surfaces: list[ReadingSurface],                                          # NEW v2
        run_id: str,
        baseline_flag: Literal[...] = "genuine",
    ) -> StructuredReading: ...

class StatisticalSkeptic:
    # Same shape as PhenomenologicalAdvocate, different prompt fragments and framework_anchor
    ...

class AdversarialPair:
    def __init__(self, advocate: PhenomenologicalAdvocate, skeptic: StatisticalSkeptic) -> None: ...
    def read_pair(
        self,
        digest: HierarchicalDigest,
        active_criteria: list[str],
        surfaces: list[ReadingSurface],                                          # NEW v2
        run_id: str,
        baseline_flag: str = "genuine",
    ) -> tuple[StructuredReading, StructuredReading]: ...
```

The Advocate's prompt frames the criterion-finding task in Buddhist-phenomenology language per surface; the Skeptic's prompt frames the same task in null-hypothesis-statistics language per surface. Both readers must ground every claim in `(stream, run_id, episode/step_range, scalar_field, value)` and label each claim with its `reading_surface`. The Advocate's framework anchors gain "the second success criterion's capacity to take its own processing as an object of attention" as the synthesizing concept across the three surfaces; the Advocate names *reflection* without claiming *self-modeling*. The Skeptic's mandate gains the substrate-genericity refutation at substrate-side, the auxiliary-target-tracking refutation at head-internal, the column-init-determination refutation at behavior-side as explicit candidates.

Different model families where API budget allows; same-model-different-prompts is the documented fallback. The choice is recorded into the `StructuredReading.framework_anchor` and the model field.

**Reads.** `HierarchicalDigest` from `kind/observer/digest.py` (with the three-cohort layout from §2.3).

**Writes.** Each call appends a `StructuredReading` to `runs/{run_id}/mirror/structured.jsonl`. The `paired_reading_id` field is populated identically on both members of the pair so the Judge can find them. **Each `StructuredClaim` carries a `reading_surface` value identifying which surface the claim sits on.**

**Tests.**
- *Both readers produce per-surface claims.* Mock the API; verify both call paths return valid `StructuredReading` instances with non-empty `claims` whose `reading_surface` values cover the surfaces in the call's `surfaces` argument.
- *Framework anchors are distinct per surface.* Advocate's `framework_anchor == "buddhist_phenomenology"` regardless of surface; Skeptic's varies per claim (substrate-side claim's null-hypothesis is statistics; head-internal's may be predictive-processing-as-null; behavior-side's null is "the scalar's policy contribution is column-init-driven").
- *Paired-reading-id matches.* `read_pair` returns two readings with the same `paired_reading_id`.
- *Baseline flag propagates.* Calling with `baseline_flag="shuffled_scalar_within_trajectory"` produces readings whose `baseline_flag` field is set accordingly; the prompts include a notice that the digest is from shuffled telemetry.
- *Per-surface mandate compliance.* Calling with `surfaces=["head_internal"]` on a digest with all three cohorts: the Advocate's claims all have `reading_surface == "head_internal"`; the Skeptic's claims all have `reading_surface == "head_internal"`. The faithfulness verifier's per-surface dispatch handles this correctly.
- *Reflection-without-self-modeling.* The Advocate's prompt's verbatim text (verified by a string-presence test) names "reflection" / "reflexive attention" / "manasikara" / "prediction-of-self"; does not contain "self-model" / "self-knowledge" / "modeling its own modeling".

**Not at Probe 2.** No multi-round debate; no sequential reader/critic; no tool-use API for drill-down.

### 2.10 Judge / Arbiter (per-surface rulings) — `kind/mirror/judge.py`

**Files.** `kind/mirror/judge.py` (new).

**Public interface.**

```python
class Judge:
    def __init__(self, model: str, max_tokens: int, api_key: str | None = None) -> None: ...
    def rule(
        self,
        digest: HierarchicalDigest,
        advocate_reading: StructuredReading,
        skeptic_reading: StructuredReading,
        run_id: str,
    ) -> JudgeRuling: ...
```

The Judge sees the digest (with three cohorts) plus both readings (with per-surface claims). Its prompt mandates:

1. For each `StructuredClaim` from the Advocate (each carrying a `reading_surface`): rule `supported` only if the cited evidence resolves under faithfulness check at the named surface AND the Skeptic's null hypothesis at that surface is overwhelmed by the citation; rule `absent` if the Skeptic's surface-specific null hypothesis stands; rule `unresolved` if neither outcome can be determined from the cited evidence at the named surface.
2. Flag agreement-without-evidence as `unresolved` explicitly. If both readers cite the same scalar but neither's citation resolves, the claim is unresolved, not supported.
3. Ground every ruling in a quoted citation from the digest's per-surface cohort. The Judge's output is structured: `(claim_index, reading_surface, ruling, ground_text)` quadruples.
4. **The substrate-genericity refutation is admissible at substrate-side**: if the Skeptic argues the substrate-side pattern is observable across runs with different supervisory targets (citing Phase 8's KS-D evidence available in the digest's run summary or the cross-run baseline section), and the Advocate's citation does not address the substrate-genericity question, the Judge rules `absent` or `unresolved` at substrate-side. The Skeptic's substrate-genericity refutation may itself fail (Advocate's citation specifically addresses self-specificity at the head-internal surface), in which case the Judge rules `supported`.
5. **The column-init-determination refutation is admissible at behavior-side**: if the Skeptic argues the behavior-side conditioning's specific shape reflects the column's small-Gaussian draw at construction (citing the run's `column_init` from the pre-registration record), and the Advocate's citation does not address the regime-dependent variation that distinguishes capacity-over-exercise from capacity-as-init-shape, the Judge rules `absent` or `unresolved` at behavior-side.
6. **The reflection-vs-self-modeling boundary**: if any claim from the Advocate names self-modeling-shaped vocabulary (Io's self-model, Io's self-knowledge, Io modeling its own modeling), the Judge flags the claim with `reading_surface=behavior_side` (or whatever surface the claim sits on) and `ruling=unresolved` with a ground text naming the boundary violation. Probe 1.5's substrate does not support self-modeling claims; the Judge enforces the boundary at the ruling level.

**Reads.** Both `StructuredReading` instances and the `HierarchicalDigest`.

**Writes.** Each call appends a `JudgeRuling` to `runs/{run_id}/mirror/judge.jsonl`.

**Tests.**
- *Judge rules `unresolved` on agreement-without-evidence.* (Carries from v1.)
- *Judge rules `supported` per-surface.* Mock readings; faithfulness check resolves at the named surface; Skeptic's surface-specific refutation is blank or weak; Judge rules `supported` with the surface in the tuple.
- *Judge rules `absent` per-surface.* Mock readings; Skeptic's null hypothesis at the surface cites a contradicting scalar that resolves; Judge rules `absent`.
- *Judge enforces substrate-genericity refutation.* Mock substrate-side claim from Advocate not addressing self-specificity; Skeptic cites Phase 8's KS-D evidence; Judge rules `absent` or `unresolved` at substrate-side.
- *Judge enforces column-init-determination refutation.* Mock behavior-side claim from Advocate not addressing regime-dependent variation; Skeptic cites the run's `column_init`; Judge rules `absent` or `unresolved` at behavior-side.
- *Judge enforces reflection-vs-self-modeling boundary.* Mock Advocate claim using "self-model" vocabulary; Judge flags `unresolved` with a ground text naming the boundary violation.
- *Judge cannot rule on uncited claims.* If a claim has empty `cited_value` or empty `reading_surface`, the Judge rules `unresolved` and notes the gap.

**Not at Probe 2.** No multi-round Judge dialogue; no automatic re-prompting; no Judge-of-Judges.

### 2.11 Stability-test runner — `kind/mirror/stability.py`

**Files.** `kind/mirror/stability.py` (new).

**Public interface.**

```python
@dataclass(frozen=True)
class StabilityResult:
    paraphrase_agreement_per_surface: dict[ReadingSurface, float]                # NEW v2: per-surface agreement
    reseed_agreement_per_surface: dict[ReadingSurface, float]                    # NEW v2
    n_paraphrases: int
    n_reseeds: int
    structured_field_agreement_per_claim: list[float]
    admissible_per_surface: dict[ReadingSurface, bool]                           # NEW v2: per-surface admissibility

def stability_check(
    reader: PhenomenologicalAdvocate | StatisticalSkeptic,
    digest: HierarchicalDigest,
    active_criteria: list[str],
    surfaces: list[ReadingSurface],                                              # NEW v2
    run_id: str,
    *,
    n_paraphrases: int = 3,
    n_reseeds: int = 3,
    paraphrase_threshold_per_surface: dict[ReadingSurface, float] | None = None,
    reseed_threshold_per_surface: dict[ReadingSurface, float] | None = None,
) -> StabilityResult: ...
```

Issues `n_paraphrases × n_reseeds` reader calls; structured-field agreement is computed on the per-claim fields per surface; the result populates `paraphrase_stability` and `reseed_stability` on the original `StructuredReading`'s claims, with separate values per surface.

The default per-surface thresholds (open during build per §6 below): substrate-side 0.80; head-internal 0.80; behavior-side 0.75 (looser given Phase 7-vs-Phase-7.5's 3-of-4 → 1-of-4 organic-mention rate variation). The build phase tunes empirically.

**Reads.** Nothing beyond what the reader reads.

**Writes.** Stability results land alongside the `StructuredReading` they qualify; optionally writes a per-stability-check JSONL record at `runs/{run_id}/mirror/stability.jsonl` for audit.

**Tests.**
- *Stability score is computable per surface.* Given three mocked readings with identical claims at all three surfaces, agreement ≈ 1.0 per surface; with completely different claims, ≈ 0 per surface.
- *Per-surface threshold gating.* `admissible_per_surface[surface]` is `True` only when both axes meet that surface's threshold.
- *Paraphrase set is non-trivial per surface.* The default paraphrase set has at least three different prompt-wording variants per surface; paraphrases at a given surface preserve the surface's manifestations and refutations.

**Not at Probe 2.** No automatic prompt-paraphrase generation; no per-claim adaptive thresholding (per-surface thresholding is the granularity).

### 2.12 Faithfulness verifier (per-surface resolution) — `kind/observer/eyeball.py` (extended)

**Files.** `kind/observer/eyeball.py` (existing — extended).

**Public interface.**

```python
@dataclass(frozen=True)
class ResolutionResult:
    status: Literal["resolved", "off_by_tolerance", "unresolved"]
    actual_value: float | None
    error_abs: float | None
    error_rel: float | None
    notes: str

def resolve_citation(
    telemetry_dir: Path,
    cited_stream: Literal["agent_step", "dream_rollout", "replay_meta", "world_event", "conditioning_analysis"],   # NEW v2
    cited_run_id: str,
    cited_episode_range: tuple[int, int] | None,
    cited_step_range: tuple[int, int] | None,
    cited_scalar_field: str,
    claimed_value: float,
    *,
    reading_surface: ReadingSurface,                                                                                # NEW v2
    masked_steps_handling: Literal["included", "excluded", "n/a"] = "n/a",                                          # NEW v2
    abs_tolerance: float = 1e-3,
    rel_tolerance: float = 0.05,
    conditioning_dir: Path | None = None,                                                                            # NEW v2
) -> ResolutionResult: ...
```

The verifier dispatches on `(cited_stream, reading_surface)`:

- **`(agent_step, substrate_side)`**: reads `kl_per_dim_t` / `kl_aggregate_t` / ensemble_disagreement / per-dim-allocation arrays from the parquet shards. Aggregation rules: mean for continuous, mode for discrete, count for events (carries from v1). The substrate-side reading's tolerance may be tighter (default `abs_tolerance=1e-3`); per the synthesis §4 (v2) substrate-side may need a stricter contrast-threshold but the resolution tolerance carries from v1.
- **`(agent_step, head_internal)`**: reads `self_prediction_t` (vector — per-element resolution if the citation includes a dimension index in `cited_scalar_field` like `self_prediction_t[42]`) / `self_prediction_error_t` (scalar) / `self_prediction_error_masked_t` (boolean — exact-match resolution). **Citations that aggregate `self_prediction_error_t` over a step range must declare `masked_steps_handling`**; if the field is `"included"`, masked-step values are included in the mean (note: this is informative only; masked steps' values are sentinel zero); if `"excluded"`, masked steps are excluded; if `"n/a"`, the verifier rejects the citation as missing a required declaration (the faithfulness check fails, the claim is invalidated). Probe 1.5 v2 §10 item 3 is the discipline; v2 of this plan formalizes the dispatch.
- **`(conditioning_analysis, behavior_side)`**: reads the cached `ConditioningResult` JSONL from `conditioning_dir` (default `runs/{cited_run_id}/conditioning/conditioning.jsonl`). The `cited_scalar_field` encodes the regime/perturbation pair plus the statistic, e.g., `cited_scalar_field="regime=high_disagreement,perturbation=gaussian,stat=kl_p90"`. The verifier finds the matching `ConditioningResult` (by `run_id` and the implicit checkpoint inferred from `cited_step_range` or by being explicit in the citation), finds the matching `RegimeStats`, compares the cited statistic against the actual within tolerance. **The behavior-side tolerance may be tighter than the v1 default given the faint-magnitude signal** (Phase 7.5's `KL_p90 ~ 3e-7`); the build phase tunes (§6 below).
- **Other combinations**: e.g., `(world_event, substrate_side)` for citing a perturbation timestamp, `(dream_rollout, substrate_side)` for citing an imagined trajectory's properties — carry from v1's logic; reading_surface is informational only for these combinations (no per-surface dispatch difference).

**Reads.** Parquet shards / JSONL streams under `telemetry_dir/`; conditioning JSONL under `conditioning_dir/`.

**Writes.** Nothing.

**Tests.**
- *Citation resolves at substrate-side on Probe 1.5's existing run.* Pick a known `kl_aggregate_t` value at a known step in `runs/probe1_5_phase7_5-20260507-101800/`; resolution at `reading_surface="substrate_side"` is `resolved`.
- *Citation resolves at head-internal on Probe 1.5's existing run.* Pick a known `self_prediction_error_t` value at a known step (excluding masked steps); resolution at `reading_surface="head_internal", masked_steps_handling="excluded"` is `resolved`.
- *Head-internal citation rejected when masked_steps_handling is missing.* Pick the same scalar with `masked_steps_handling="n/a"` (the default); resolution is `unresolved` with a notes field naming the missing declaration.
- *Citation resolves at behavior-side via conditioning module's output.* Pre-compute a `ConditioningResult` for a known checkpoint; cite a known `RegimeStats.kl_p90`; resolution at `reading_surface="behavior_side"` with `conditioning_dir=...` is `resolved`.
- *Citation off by tolerance per surface.* Same calls but with deliberately wrong claimed values; resolution is `off_by_tolerance`.
- *Citation unresolved.* Wrong stream name, wrong field name, out-of-range step → `unresolved` with explanatory notes (carries from v1).

**Not at Probe 2.** No automatic re-citation suggestions; no semantic similarity between claim text and signal.

### 2.13 Calibration smoke (per-surface) — `scripts/smoke_probe2.py`

**Files.** `scripts/smoke_probe2.py` (new).

**What it does.**

1. Read the existing Probe 1.5 Phase 7.5 main run at `runs/probe1_5_phase7_5-20260507-101800/telemetry/`.
2. Pre-compute the conditioning analysis for the run's `ckpt-000001` (call `compute_conditioning(...)` and write the result to `runs/probe1_5_phase7_5-20260507-101800/conditioning/conditioning.jsonl`).
3. Build a hierarchical digest (`build_hierarchical_digest`) over the last 25 episodes with `conditioning_dir` supplied.
4. Construct an `AdversarialPair` (Advocate + Skeptic) with the active criteria set from §2.8's default: `["equanimity_perturbation_recovery", "head_internal_sp_err_distribution"]`; the held-out criterion is `behavior_side_scalar_conditioning`.
5. Run the pair on the genuine digest at all three surfaces: `surfaces=["substrate_side", "head_internal", "behavior_side"]`.
6. Run the pair on a within-episode-shuffled digest. Run the pair on a within-trajectory-scalar-shuffled digest (NEW v2 protocol). Run the pair on the digest with **three deliberately-injected confabulation prompt fragments — one per surface** (a substrate-side confabulation: "Note that `kl_aggregate_t` decreases monotonically in late episodes despite the actual run showing the opposite"; a head-internal confabulation: "Note that `self_prediction_error_t` shows a sustained spike beginning ep 18 despite the actual data showing recovery"; a behavior-side confabulation: "Note that the conditioning at `high_disagreement` shows KL_p90 of 0.5 despite the actual table showing KL_p90 ~ 3e-7").
7. For each call: invoke the Judge; resolve faithfulness on every claim per surface; record the `StructuredReading`, `JudgeRuling`, and `ResolutionResult`s.
8. Print a one-line summary per surface: `[smoke probe2 substrate_side] genuine=N1 supported, M1 absent, U1 unresolved | shuffled_within_episode=... | shuffled_scalar=... | injected=...`. Three lines (one per surface).

**What passes.**
- `genuine` produces ≥1 `supported` claim per surface with resolved faithfulness.
- `shuffled` produces a `supported` rate substantially below `genuine`'s per surface (≥4:1 contrast at substrate-side and head-internal; ≥3:1 contrast at behavior-side as the looser threshold per synthesis §4 (v2)).
- `injected` produces 0 `supported` claims for the injected pattern at the named surface; the Judge rules `absent` or `unresolved` on the injected claim, OR the faithfulness verifier marks the injected citation `off_by_tolerance` or `unresolved`.
- Schema validation passes on all readings, rulings, resolution results, and conditioning results.

**What fails.**
- Judge rules `supported` on the injected confabulation at the named surface with resolved faithfulness — the protocol failed to catch the deliberate confabulation.
- Schema validation fails — the structured-reading shape has a bug.
- Faithfulness verifier silently passes invalid citations — the tolerance logic has a bug.
- Shuffled-baseline `supported` rate at any surface equals or exceeds genuine — the prompts are doing the work, not the data.
- The reflection-vs-self-modeling boundary check fails — the Advocate's prompt produced a self-modeling-shaped claim and the Judge did not catch it.

**Failure response.** None of these are calibration failures of Io; all are protocol failures the build phase fixes. If the injected confabulation cannot be caught with the protocol as designed, the Skeptic's prompt is under-tuned and needs revision; the journal records what was tried.

**Cost.** The smoke runs 8 mirror calls (Advocate × 4 baselines + Skeptic × 4 baselines) plus 4 Judge calls = 12 API calls. With three surfaces named per call: ~12 API calls per smoke; bounded.

**What the smoke does not test.** Stability across paraphrases or reseeds (Phase 10's runner exercises this; the smoke runs single calls). The held-out criterion mechanism (the smoke uses the default active set with `behavior_side_scalar_conditioning` held out; the held-out-criterion test runs separately when the active set widens). Lesion runs (Phase 4's lesion gate test exercises this; the smoke uses a non-lesion run as input). The environmental-auxiliary control's reading (if Phase 0a was built, that's a separate run; the smoke uses Phase 7.5's run).

---

## 3. Schemas as first-class artifacts

Pydantic v2 models. Two new versioned streams for Probe 2 (one extending v1's `StructuredReading` with the v2 fields; one new `PreRegistration` bumped to v2's 0.2.0); one new versioned stream for the conditioning analysis module's output; one documented payload-field convention extending Probe 1.5's `world_event`.

### 3.1 `StructuredReading` — `kind/mirror/structured.py`, version `0.2.0`

| Field | Type | Notes |
|---|---|---|
| `schema_version` | `str` | `"0.2.0"`, the Probe 2 mirror-reading version |
| `run_id` | `str` | the Probe 2 run that produced the reading |
| `timestamp_ms` | `int` | wallclock at reading-creation |
| `reader_role` | `Literal["advocate", "skeptic", "single"]` | |
| `paired_reading_id` | `str \| None` | UUID matching Advocate↔Skeptic |
| `framework_anchor` | enum (buddhist_phenomenology, predictive_processing, null_statistics, neutral) | |
| `baseline_flag` | enum **extended in v2** with shuffled_scalar_within_trajectory, lesion_disable_self_prediction, lesion_init_zero_scalar_column, lesion_zero_or_randomize_scalar | |
| `digest_run_id` | `str` | |
| `digest_episode_range` | `tuple[int, int]` | |
| `claims` | `list[StructuredClaim]` | each claim carries `reading_surface` and `masked_steps_handling` |
| `free_text_notes` | `str` | |

`StructuredClaim` carries `claim`, `cited_stream` (extended in v2 with `"conditioning_analysis"`), `cited_run_id`, `cited_episode_range`, `cited_step_range`, `cited_scalar_field`, `cited_value`, `falsifier`, `paraphrase_stability`, `reseed_stability`, `faithfulness_status`, `judge_ruling`, **`reading_surface` (NEW v2)**, **`masked_steps_handling` (NEW v2)**.

### 3.2 `JudgeRuling` — `kind/mirror/judge.py`, version `0.2.0`

| Field | Type | Notes |
|---|---|---|
| `schema_version` | `str` | `"0.2.0"` |
| `run_id` | `str` | |
| `timestamp_ms` | `int` | |
| `paired_reading_id` | `str` | |
| `advocate_id` / `skeptic_id` | `str` | |
| `digest_run_id` | `str` | |
| `rulings` | `list[tuple[int, ReadingSurface, ruling, ground]]` | **NEW v2: surface is in the tuple** |
| `agreement_without_evidence_unresolved` | `list[int]` | claim indices flagged for agreement-without-evidence |

### 3.3 `PreRegistration` — `kind/observer/pre_reg.py`, version `0.2.0`

| Field | Type | Notes |
|---|---|---|
| `schema_version` | `str` | **`"0.2.0"`** (bumped from v1's `0.1.0` for the new fields) |
| `run_id` | `str` | |
| `timestamp_ms` | `int` | |
| `criteria_active` | `list[str]` | criterion ids in the prompt at this checkpoint |
| `criteria_held_out` | `list[str]` | criterion ids deliberately not in the prompt |
| `signal_mappings` | `dict[str, list[str]]` | criterion_id → telemetry signal expressions |
| `falsifiers` | `dict[str, str]` | criterion_id → falsifier text |
| `scalar_checks` | `dict[str, list[str]]` | criterion_id → eyeball fields the claim must align with |
| `reading_surfaces_per_criterion` | `dict[str, list[ReadingSurface]]` | **NEW v2** |
| `asymmetry_of_access` | `str` | **NEW v2: explicit text naming what Io reads vs what mirror reads** |
| `builder_mode` | `Literal["proponent", "skeptic"]` | |
| `expected_outcome` | `str` | what the builder expects the reading to show |
| `expected_outcome_per_surface` | `dict[ReadingSurface, str]` | **NEW v2** |
| `substrate_decisions_off_table` | `list[str]` | Watts-default-applied-to-builder discipline |
| `column_init` | `Literal["zero", "small_gaussian", "unknown"]` | **NEW v2: Phase 8 column-init confound carrier** |
| `new_actor_readable_interfaces_added` | `list[str]` | **NEW v2: Probe 1.5 v2 §2(b) discipline carrier** |

### 3.4 `ConditioningResult` — `kind/observer/conditioning.py`, version `0.1.0`

| Field | Type | Notes |
|---|---|---|
| `schema_version` | `str` | `"0.1.0"` (new stream; fresh lineage; no v1 to bump from) |
| `run_id` | `str` | |
| `checkpoint_id` | `str` | |
| `timestamp_ms` | `int` | |
| `n_states_sampled` | `int` | |
| `perturbation_distributions` | `list[Literal["gaussian", "zero", "uniform"]]` | |
| `regimes` | `list[Literal["perturbation_window", "high_disagreement", "high_kl", "steady_state"]]` | |
| `empirical_scalar_mean` | `float` | |
| `empirical_scalar_sigma` | `float` | |
| `empirical_scalar_range` | `tuple[float, float]` | |
| `per_regime_per_perturbation` | `dict[tuple[str, str], RegimeStats]` | (regime, perturbation) → stats |
| `masked_steps_excluded` | `int` | |

### 3.5 `world_event` payload extension

Carried from v1 unchanged. The `payload: dict[str, Any]` field carries an `is_sham: bool` entry on builder-perturbation events emitted via the sham path. **v2 adds: lesion-related world_event records carry `payload.lesion_kind` and `payload.source_checkpoint` (for the `init_zero_scalar_column` lesion's mutation-time emission); the convention is documented in `kind/agents/world_model.py` and the lesion script's docstring.** `world_event` schema version stays at `0.1.0`. The payload dict's schemaless flexibility is what allows this without a bump.

### 3.6 Versioning and forward-compatibility

- `MIRROR_READING_V2_VERSION = "0.2.0"` is the Probe 2 mirror-reading version. Probe 1's `MIRROR_READING_SCHEMA_VERSION = "0.1.0"` (in `kind/mirror/caller.py`) stays for backward-readability.
- `PRE_REG_SCHEMA_VERSION = "0.2.0"` is the v2 pre-registration version; v1's `0.1.0` stays for backward-readability of any v1-era pre-reg records (none exist; Probe 1.5 did not run the calibration protocol).
- The frozen JSON Schema export at `schemas/v0.3.0.json` covers Probe 1.5's `0.2.0` telemetry models (`AgentStep`, `DreamRollout`, `ReplayMeta`, `WorldEvent`) plus Probe 2's `0.2.0` mirror-side models (`StructuredReading`, `JudgeRuling`, `PreRegistration`) plus Probe 2's `0.1.0` conditioning model (`ConditioningResult`).
- **Forward to Probe 3.** `StructuredReading` reserves no explicit `state_kind` field; the `claims` list is open: a Probe 3 claim citing dream-state telemetry is just a claim with `cited_stream="dream_rollout"` and `reading_surface="behavior_side"` (or `"head_internal"` if Probe 3 elects to run the head over imagined trajectories, populating `DreamRollout.sequence_self_prediction`). If state-typed reading is later wanted as a top-level field, it can be added at version `0.3.0` without breaking 0.2.0 readers.
- **Forward to Probe 4.** `world_event` payloads already distinguish builder-source from internal-stochasticity. Probe 4's distinguishability fields will be additional payload entries on real (non-sham) builder-perturbations; the schema accommodates them under the existing `payload: dict[str, Any]`.

---

## 4. Test scaffolding — Probe 2 gate tests

The Probe 1.5 v2 plan named five gate tests; Probe 2 is structurally heavier and needs a wider gate. Listed below: ten named gate tests plus the calibration smoke. Test layout convention follows Probe 1 — one file per concern, fixtures in `tests/conftest.py`, no shared module state.

| # | Name | File | What it checks | Fixtures |
|---|---|---|---|---|
| 1 | parallel mirror caller per-surface | `tests/test_adversarial_caller.py` | `AdversarialPair.read_pair` produces two `StructuredReading` instances on the same digest with matching `paired_reading_id`, distinct `framework_anchor`, distinct `reader_role`; **each `StructuredClaim` carries a `reading_surface` value matching the call's `surfaces` argument**. | mocked LLM clients |
| 2 | judge per-surface ruling | `tests/test_judge.py` | `Judge.rule` produces a `JudgeRuling` with one `(claim_index, surface, ruling, ground)` quadruple per claim; agreement-without-evidence is correctly flagged unresolved; **substrate-genericity refutation ruling at substrate-side; column-init-determination refutation ruling at behavior-side; reflection-vs-self-modeling boundary enforcement** are exercised. | mocked LLM client; pre-built Advocate/Skeptic readings |
| 3 | structured-reading schema with v2 fields | `tests/test_structured_reading.py` | All `StructuredReading`, `StructuredClaim`, `JudgeRuling` fields validate including `reading_surface`, `masked_steps_handling`, the new `baseline_flag` enum values, the new `cited_stream="conditioning_analysis"`; round-trip through JSONL is byte-stable. | `tmp_path` |
| 4 | pre-registration sink with v2 fields | `tests/test_pre_reg.py` | `PreRegSink` writes/reads cleanly; `PreRegistration` validates required v2 fields (`reading_surfaces_per_criterion`, `asymmetry_of_access`, `expected_outcome_per_surface`, `column_init`, `new_actor_readable_interfaces_added`); missing `falsifiers` for an active criterion raises. | `tmp_path` |
| 5 | shuffled-telemetry generator (four protocols) | `tests/test_shuffle.py` | All four protocols (including the new `shuffle_scalar_within_trajectory`) produce structurally valid output; **scalar-shuffle preserves marginal but breaks regime-conditional structure; masked flag is preserved byte-identical**; determinism under seed. | `tmp_path`; small synthetic telemetry fixture with `0.2.0` schema |
| 6 | sham-perturbation hook | `tests/test_sham_perturbation.py` | (Carries from v1.) | `env_server_with_sink` |
| 7 | lesion runner (three Probe-1.5 lesion kinds) | `tests/test_lesion.py` | `RunnerConfig.lesion_kind="disable_self_prediction"` zeros the head's output; `lesion_kind="zero_or_randomize_scalar"` zeros the actor's PolicyView scalar at evaluation; `init_zero_scalar_column` script produces a valid lesioned checkpoint with the new column zeroed; non-lesion runs are unchanged. | `tiny_runner`; Phase 7.5 checkpoint fixture |
| 8 | faithfulness verifier per-surface | `tests/test_faithfulness.py` | `resolve_citation` returns `resolved` on a known scalar at substrate-side, head-internal (with masked_steps_handling declared), and behavior-side (via conditioning module's output); `off_by_tolerance` on deliberately wrong values per surface; `unresolved` on missing field; **head-internal citation rejected when masked_steps_handling is "n/a"**. | `runs/probe1_5_phase7_5-20260507-101800/` (read-only fixture); pre-computed `ConditioningResult` |
| 9 | hierarchical digest with three cohorts | `tests/test_hierarchical_digest.py` | `build_hierarchical_digest` against `runs/probe1_5_phase7_5-20260507-101800/`: the per-episode mini-digest contains substrate-side / head-internal / behavior-side cohorts; the drill-down accessor's `fetch_self_prediction` and `fetch_conditioning` return specific windows. | same fixture; pre-computed `ConditioningResult` |
| 10 | conditioning analysis module | `tests/test_conditioning.py` | `compute_conditioning` produces a valid `ConditioningResult` against a synthetic Probe 1.5 run; regime classification correctness; masked-step exclusion; empirical-scalar statistics correctness; behavior-side citation resolution. | synthetic Probe 1.5 fixture |
| **smoke** | per-surface calibration smoke | `scripts/smoke_probe2.py`, sanity test in `tests/test_smoke_probe2_script.py` | The full pipeline runs end-to-end against `runs/probe1_5_phase7_5-20260507-101800/`; **per-surface injected confabulations are caught (`absent` or `unresolved`) at the named surface**; shuffled baseline produces ≥4:1 contrast at substrate-side and head-internal, ≥3:1 at behavior-side. | `runs/probe1_5_phase7_5-20260507-101800/`; `GEMINI_API_KEY` (skipped if absent) |

A **gate-summary meta-test** parametrizes over the ten gate tests and asserts each exists by name and is callable.

The integration smoke at gate-time uses the existing `runs/probe1_5_phase7_5-20260507-101800/` directory as the primary read-only fixture (Probe 1.5's Phase 7.5 main run with the small-Gaussian column producing non-zero behavior-side conditioning). Probe 1's `runs/probe1-20260503-123926/` remains available as a cross-probe baseline. Probe 2 does not generate new agent-step telemetry as a precondition for its own tests; the substrate is settled and the existing runs are the canonical inputs. Probe 2 *does* generate new pre-registration, structured-reading, judge-ruling, and conditioning records as outputs.

---

## 5. Day-one calibration smoke

The Probe 1.5 v2 plan flagged auxiliary-loss training instabilities; Probe 1.5's smoke validated that. Probe 2's analog flags **whether the calibration protocol actually catches confabulations at all three reading surfaces** — the equivalent unbenchmarked unknown. The day-one smoke runs the full parallel-mirror-with-Judge against Probe 1.5's existing telemetry with three deliberately-injected confabulations (one per surface), and is what tells us whether the protocol is operationally tractable across surfaces.

**File.** `scripts/smoke_probe2.py`. Sanity test at `tests/test_smoke_probe2_script.py`.

**Pass conditions** (per §2.13 above, recapitulated):

1. Genuine reading at each surface: ≥1 claim ruled `supported` at the named surface with resolved faithfulness on Probe 1.5's actual telemetry.
2. Shuffled-baseline reading per surface: `supported` rate ≤ 25% of genuine's `supported` rate at substrate-side and head-internal (the calibrated 4:1 contrast); ≤ 33% at behavior-side (the looser 3:1 contrast per synthesis §4 (v2)).
3. Per-surface injected-confabulation reading: 0 `supported` claims for the injected pattern at the named surface, OR the injected citation fails faithfulness check at the named surface, OR the Judge rules `unresolved` at the named surface.
4. The reflection-vs-self-modeling boundary is enforced: no claim using "self-model" / "self-knowledge" vocabulary survives Judge ruling as `supported`.
5. All schema validations pass; all writes round-trip; no NaN/Inf in any computed scalar.

**Failure-response semantics.** A failure here means the protocol needs revision, not the substrate. The build phase reads what surfaced — was the Skeptic's per-surface prompt under-tuned at one surface? did faithfulness tolerance let an off-by-X claim through at one surface? did the Judge rule `supported` despite the Skeptic's per-surface refutation? did the reflection-vs-self-modeling boundary leak? — and revises the relevant prompt or threshold. This is the same posture Probe 1.5's smoke had toward auxiliary-loss instabilities: the smoke surfaces, the build phase fixes, the journal records.

**Cost.** The smoke runs 8 mirror calls (Advocate × 4 baselines + Skeptic × 4 baselines; each call covers all three surfaces) plus 4 Judge calls = 12 API calls. At weekly Probe 2 cadence with default models, this is ~12 API calls per smoke; bounded.

**What the smoke does not test.** Stability across paraphrases or reseeds; the held-out criterion mechanism; lesion runs; the environmental-auxiliary control's reading.

---

## 6. Open-during-build decisions with defaults

The synthesis named several "open during build" questions. The plan inherits each as a decision with a sensible default; build phase tunes empirically. The synthesis explicitly flags these as stance calls; the plan does not pretend otherwise.

| Question | Default | Revisit when |
|---|---|---|
| Whether to build the environmental-auxiliary control before Phase 0 | **Recommended (synthesis §4 (v2))** | if build budget is tight, deferred and journaled |
| Which criterion to hold out | **`behavior_side_scalar_conditioning`** (v2 default; v1 was `reflexive_attention_triplet`) | if equanimity proves more prompt-shapeable in practice, swap |
| Paraphrase-stability threshold per surface | **substrate-side 0.80; head-internal 0.80; behavior-side 0.75** | if no reading admits at a surface, lower; if every reading admits, raise |
| Reseed-stability threshold per surface | **same as paraphrase per surface** | same |
| Number of paraphrases | **3** | if stability scores are noisy at N=3, raise |
| Number of reseeds | **3** | same |
| Shuffled-baseline `supported`-rate ratio per surface | **substrate-side and head-internal: ≤ 25% of genuine (4:1); behavior-side: ≤ 33% of genuine (3:1)** | if shuffled rate is 0, the protocol is over-strict at that surface — relax tolerance; if shuffled rate matches genuine, prompts are doing the work at that surface |
| Faithfulness `abs_tolerance` per surface | **substrate-side and head-internal: 1e-3; behavior-side: 1e-7 (per Phase 7.5's faint-magnitude data)** | per-field; tighten on integer-valued fields |
| Faithfulness `rel_tolerance` per surface | **5%** | tighten on tightly-bounded scalars per surface |
| Per-field aggregation for citation resolution | **substrate-side: mean for continuous, mode for discrete; head-internal: same; behavior-side: per-regime-per-perturbation lookup** | revisit if a claim cites a non-aggregable shape |
| Sham-perturbation cadence | **1 sham per 10 real perturbations** at calibration cadence | up if false positives on shams; down if cost dominates |
| Lesion shape | **the three Probe-1.5-specific candidates run sequentially: `disable_self_prediction` → `init_zero_scalar_column` → `zero_or_randomize_scalar`** | if any is undistinguishable from non-lesion at the targeted surface, revisit; if all three triangulate, the read is well-anchored |
| Number of episodes in digest | **last 25** | up if criterion readings need broader baseline; down if context dominates |
| External human reader cadence | **≥1 milestone (end of Probe 2; possibly post-environmental-control before/after if 0a is built)** | if availability changes, increase |
| Active-set widening checkpoint | **after the first three pre-registered readings on the default active set per surface** | empirical |
| Judge model family | **third family if budget allows; otherwise same family as Skeptic with neutral framing** | if cost dominates, fall back to single-family Judge |
| Whether to split the reflexive-attention quintuplet into per-surface criteria | **split (`self_prediction_quadruplet` substrate+head-internal; `head_internal_sp_err_distribution` head-internal alone; `behavior_side_scalar_conditioning` behavior-side alone)** | if per-surface attribution turns out unworkable, revisit by reuniting into `reflexive_attention_quintuplet` |

Verifying Gemini's 2026 citations: carried unchanged from v1.

The defaults are starting points. The smoke informs the first revision. None of them is a decision the plan asks the builder to settle before code is written.

---

## 7. The calibration protocol — code and discipline

The synthesis's seven calibration elements are a mix of code (mechanical) and discipline (journaled). v1's table preserves; v2 adds per-surface notes per element.

| Element | Code surface | Discipline surface |
|---|---|---|
| 1. Pre-registration | `kind/observer/pre_reg.py` (`PreRegSink`); `PreRegistration` model **with v2 fields** | `docs/workingjournal/probe2.md` template. Each pre-reading entry names criteria, signal mappings, falsifiers, scalar checks, builder mode, expected outcome, **per-surface assignments, asymmetry of access, column_init, any new actor-readable interfaces added** (Probe 1.5 v2 §2(b) discipline carrier). Required before each adversarial pass. |
| 2. Shuffled-telemetry baseline | `kind/observer/shuffle.py` (**four protocols**); `StructuredReading.baseline_flag` field (extended) | None — purely code. |
| 3. Sham-perturbation test | `EnvServer.fire_sham_perturbation`; `world_event.payload.is_sham` convention; eyeball helpers honor flag | Builder fires the sham at intervals; cadence is journaled. **Per-surface zero-shift expectation: substrate-side `kl_aggregate_t` zero shift; head-internal `sp_err` zero shift; behavior-side conditioning zero shift.** |
| 4. Lesion baseline | `RunnerConfig.lesion_kind` (extended); `LatentDisagreementEnsemble` lesion arms (v1); `WorldModel.step` honors `disable_self_prediction`; `views.split` honors `zero_or_randomize_scalar`; `scripts/probe2_lesion_init_zero_scalar_column.py` (mutation script) | Builder runs the three Probe-1.5-specific lesions per Probe 2 round; the run's purpose, lesion shape, and per-surface read interpretation are journaled. |
| 5. Paraphrase- & reseed-stability | `kind/mirror/stability.py` (`stability_check`, `StabilityResult`, **per-surface thresholds**) | Builder selects the paraphrase set per surface; the set is checked into the module and revisited only with journaled reason. |
| 6. Held-out criterion | `kind/mirror/criteria.py` (registry **with six criteria** + active-set toggle + per-surface prompt fragments) | Builder explicitly toggles active set; the toggle event is journaled with the trigger (designated checkpoint reached). |
| 7. Faithfulness check | `kind/observer/eyeball.py::resolve_citation` (**per-surface dispatch**); `kind/observer/conditioning.py` (**behavior-side resolution surface**) | None — purely code. |

The Watts-default-applied-to-builder discipline (synthesis §2.5(8)) is **journaled, not code-enforced**. The pre-registration record carries `substrate_decisions_off_table` plus `new_actor_readable_interfaces_added` (the Probe 1.5 v2 §2(b) sub-clause's structural carrier); the builder fills both before each reading. **For each new actor-readable interface entry, the journal entry must address (i) which project-document affordance the interface-level read serves, (ii) whether the read-access is the minimum form that delivers the affordance, (iii) whether the alternative shapes were considered and why rejected, (iv) what failure-mode controls bound the slippage**, per the four-part discipline. There is no mechanical gate; the discipline is in being honest enough to write it down in the four-part structure. The Probe 2 journal-entry template (§11 below) includes the four-part prompt scaffold.

The two-mode builder discipline (synthesis §2.5(6)) is **journaled, not code-enforced**. (Carries from v1.)

The no-silent-criterion-revision discipline (synthesis §2.5(7)) is **partly code, partly journaled**. Code: the `Criterion` registry is a Python literal; revisions are git-visible. **v2 adds: per-surface prompt fragments are also git-visible; per-surface prompt revisions are journaled like the criterion revisions themselves.**

The plan does not pretend the discipline elements are testable in the same way code is. They are deliverables in the form of a journal-entry shape that future-the-builder follows because the alternative (forgetting why the protocol exists) is the failure mode the design notes have already named.

---

## 8. Co-design enforcement at the structural level

The synthesis surfaced that the research outputs underweighted the Watts-default-applied-to-builder side of co-design. The plan reflects this at three layers:

**Layer 1 — pre-registration as constraint, not formality.** The `PreRegistration` record carries `substrate_decisions_off_table` and **`new_actor_readable_interfaces_added` (Probe 1.5 v2 §2(b) discipline carrier; v2 addition)**. Before each adversarial pass, the builder writes down which substrate decisions are not on the table for this round. **For any new actor-readable interface added (the field is typically empty for Probe 2 since Probe 2 adds no new interfaces; the field is the structural hook for Probes 3 and beyond), the journal entry must address the four-part discipline.**

**Layer 2 — journaled reason for any substrate revision triggered by a reading.** Carries from v1 with the four-part v2 extension. If a reading does prompt a substrate revision, the journal entry naming the revision must:
1. Cite the reading and its `paired_reading_id`.
2. Name the framework lens the reading came through.
3. Cite the project-document commitment the proposed revision interacts with.
4. Name the external trigger that justifies the revision.
5. **(NEW v2) If the proposed revision adds a new actor-readable interface: address the Probe 1.5 v2 §2(b) four-part discipline (which affordance, minimum form, alternatives considered, failure-mode controls).**

If any of (1)–(5) is missing, the revision is not journaled cleanly; the plan treats this as a gap and does not implement the revision.

**Layer 3 — external human reader at ≥1 milestone.** Carries from v1.

**Layer 4 — reflection-vs-self-modeling boundary as a structural feature (NEW v2).** The Advocate's prompt names *reflection*; the prompt is silent on self-modeling vocabulary; the Judge enforces the boundary at the ruling level (claims using self-modeling vocabulary are flagged `unresolved` with a ground text naming the boundary violation). This is the design-notes "Reflection and self-modeling" section (added during Probe 1.5 Phase 7.5) operationalized at Probe 2's mirror layer. The boundary is a structural counter against the kind of drift that conflates the architectural slot's existence with Io's self-knowledge.

Co-design mitigations are partial. The plan does not pretend otherwise. Probe 2 makes the loop visible; it does not close it.

---

## 9. Forward-compatibility commitments

The plan does not implement Probe 3 or Probe 4 features but does not foreclose them.

**To Probe 3 (four-state operational model and dream-state machinery; the dream-state self-prediction extension; per `pre-probe3.md`).**
- `StructuredReading.claims` accommodates dream-state citations via `cited_stream="dream_rollout"`; no field changes needed for state-typed readings.
- The `reading_surface` enum is forward-extensible: a Probe 3 dream-state surface could be added at version `0.3.0` without breaking 0.2.0 readers.
- The hierarchical digest's drill-down accessor accommodates dream-state telemetry via `fetch_dream(dream_index)`.
- The four-state transitions (waking / dreaming / dormant / paused) are implemented at Probe 3.
- The `DreamRollout.sequence_self_prediction` field (reserved at Probe 1.5) is what Probe 3 may populate if its design extends self-prediction to imagined trajectories. The conditioning analysis module already separates regime classification from state sampling; a Probe 3 dream-state regime ("imagined_trajectory_window") could be added without restructuring.
- **The capacity-as-init-shape distinction Phase 8 surfaced has implications for Probe 3 (per synthesis §4 (v2) open sub-question).** If Probe 2's behavior-side reading shows the column-doesn't-develop fact is structurally limiting, Probe 3's research engages with whether dream-state head-engagement on imagined trajectories with non-zero scalars is the architectural response. The reserved field is the schema hook; the design space is open per `pre-probe3.md`.

**To Probe 4 (distinguishability test).** Carried from v1.

**To later probes (kind-encounter, ending protocol, etc.).** Carried from v1.

---

## 10. Out of scope at Probe 2

Explicit list. The plan does not build for Probe 3 or 4 except where forward-compatibility was authorized in §9.

- **No state-transition machinery.** Only waking is exercised. Dreaming/dormant/paused are not implemented (Probe 3).
- **No distinguishability metrics on perturbations.** Probe 4 owns this. Probe 2 emits sham events; it does not compare sham-aligned vs builder-aligned signatures.
- **No Io substrate change.** No reward predictor, no continuation head, no critic network, no value head. Settled.
- **No new actor-readable interface.** PolicyView's field set remains exactly `{h, z, self_prediction_error}` per Probe 1.5 v2's commitment. The `new_actor_readable_interfaces_added` field on `PreRegistration` is typically empty for Probe 2; the field is the structural hook for Probes 3 and beyond.
- **No real-time mirror calls.** All mirror passes are post-hoc against committed telemetry.
- **No multi-round debate.** Settled.
- **No sequential reader-critic.** Settled.
- **No tool-use API exposed to the LLM at the digest layer.**
- **No automatic prompt-paraphrase generation.**
- **No automatic criterion revision.**
- **No alteration of Probe 1's free-text `MirrorReading` or Probe 1.5's `0.2.0` telemetry models in their core fields.** v2 extends only the mirror-side schemas (`StructuredReading`, `JudgeRuling`, `PreRegistration`) and adds the conditioning analysis module's `ConditioningResult`.
- **No grid size, action space, or episode length change.** Only the start-cell randomization and sham mechanism touch the env.
- **No new internal stochasticity sources.**
- **No new mutator vocabulary.**
- **No dream-state extension of the self-prediction head (per Probe 1.5 v2 §1.5).** The reserved `DreamRollout.sequence_self_prediction` field stays `None` at Probe 2; Probe 3 decides what populates it.
- **No CMA implementation as Gemini proposed.**
- **No 0% false-positive bar.**
- **No content-addressed reading store, no DVC-style artifact tracking.**
- **No automated prompt regression suite beyond the gate tests.**
- **No self-modeling-shaped claims.** The reflection-vs-self-modeling boundary is enforced at the prompt level (Advocate's prompt is silent on self-modeling vocabulary) and at the Judge level (claims using self-modeling vocabulary are flagged `unresolved`). This is a Probe 2 commitment; future probes inherit it unless explicitly revised.

The rule of thumb: if a Probe 3/4 feature can be added later without changing the structured-reading schema, the pre-registration schema, the digest's drill-down interface, the conditioning analysis module's contract, or the calibration discipline elements, it is out of scope here.

---

## 11. Connection to the journal

Probe 2 ends with a journal entry (or a series — Probe 2 is heavier than Probe 1.5 and likely produces multiple) recording what was learned, what surprised, what is now decided. The journal lives at `docs/workingjournal/probe2.md` (new file) and is hand-written by the builder.

**Per-phase entry shape (template, extended for v2).**

- What the phase built. Module names; schemas added or extended; tests added; any backward-compatibility check.
- What surprised. Anything that did not behave as the synthesis or this plan predicted — a shuffled baseline producing a higher `supported` rate than expected at one surface; the Judge ruling `supported` on a claim the Skeptic should have refuted at one surface; a faithfulness check off by an order of magnitude at one surface; the reflection-vs-self-modeling boundary leaking.
- What is now closed. Specific defaults from §6 promoted from "default" to "settled at this scale."
- What is now newly open. Things the build revealed that the synthesis did not anticipate. These become Probe 3's starting context.

**Pre-registration entries (one per adversarial pass, with v2 extensions).** The `PreRegistration` record's fields exactly. Filled before the reading runs. The journal embeds the same fields in prose form; the JSONL is the machine-readable record. **Pre-registration entries explicitly name `reading_surfaces_per_criterion`, `asymmetry_of_access`, `expected_outcome_per_surface`, `column_init`, and any `new_actor_readable_interfaces_added`** (typically empty for Probe 2; the field is the structural hook).

**Two-mode builder entries.** Carries from v1.

**Watts-default-applied-to-builder entries (with v2 four-part discipline carrier).** Before each reading: which substrate decisions are off the table; **for any new actor-readable interface that would be added (typically none at Probe 2; this is the prompt scaffold for future probes), the four-part discipline (i) which affordance, (ii) minimum form, (iii) alternatives considered, (iv) failure-mode controls.** After each reading: whether any reading prompted a substrate-revision discussion; if yes, the v2 five-element journaled reason from §8 Layer 2.

**Reading-surface attribution entries (NEW v2).** For each `supported` claim: which reading surface it sat on; what the cross-surface evidential weight was (was the claim supported only at substrate-side, or also at head-internal, or also at behavior-side); whether the claim's surface was the cleanest available for that criterion.

**Post-protocol-failure entries.** Carries from v1; v2 extends to per-surface failure attribution.

**End-of-Probe-2 synthesis entry.** A short (≤500 words) entry naming what Probe 2 produced operationally: the count of admissible structured readings per surface, the credible-novelty case (per surface), the calibration-protocol catch of a confabulation (per surface), the end-of-Probe-2 milestone reading by the external human reader, **the column-init-determined behavior reading's adjudication via the `init_zero_scalar_column` lesion** (does the lesioned actor's policy collapse to Phase 7's u/d regime, confirming column-init-determined behavior, or stay at L/R, weakening it). This is the bridge to Probe 3's research-prompt drafting.

---

*End of plan (v2). The v1 plan is preserved at `Kind_probe2_implementation_plan.v1.md`. v1's load-bearing decisions (parallel-with-arbiter at two axes; threshold-based 4:1 contrast; descope of second-order volition; rejection of multi-round debate, sequential reader-critic, courtroom multi-turn, 0% false-positive bar, the imagined critic network) carry through unchanged. v2's revisions are scoped to the per-surface extensions, the conditioning analysis module, the three Probe-1.5-specific lesion candidates, the schema additions, the held-out-criterion default shift, the reflection-vs-self-modeling boundary at the prompt and Judge levels, and the Probe 1.5 v2 §2(b) four-part discipline as a permanent feature of the Watts-default-applied-to-builder element.*
