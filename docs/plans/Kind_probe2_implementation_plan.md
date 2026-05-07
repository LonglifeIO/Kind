# Probe 2 — Implementation Plan

*Operational plan that translates the Probe 2 synthesis (`docs/decisions/Kind_probe2_synthesis.md`) into a concrete build sequence on top of the Probe 1 substrate. The substrate is settled (`Kind_architectural_decision.md`); the Probe 1 implementation is settled (`Kind_probe1_synthesis.md`); the environment is extended by §2.1 of the Probe 2 synthesis (random non-wall start cell + sham-perturbation flag-only path), but otherwise unchanged. Probe 2 is mirror calibration with frozen criteria and adversarial structure — the parallel reader/Skeptic/Judge architecture, the seven-element calibration protocol, and the structured-reading schema that downstream probes will track against. Forward-compatibility lives in the schemas (state-typed reading hooks for Probe 3, distinguishability fields for Probe 4), the digest's drill-down accessor, and the discipline elements that survive intact across probes. Nothing in this plan changes Io's substrate; nothing introduces real-time mirror calls; nothing introduces reward, value heads, or critic networks.*

---

## 1. Build order with dependency graph

Hard dependencies (must run sequentially):

- **Schemas before writers.** The Probe 2 schema deltas — `StructuredReading`, `PreRegistration`, the `is_sham` payload field on `world_event` — exist before any component emits or consumes them.
- **Hierarchical digest before adversarial readers.** The two-axis adversarial pattern reads digests; readers cannot be tested end-to-end before the digest module produces base + drill-down output.
- **Faithfulness verifier before stability runner.** Stability requires admissibility checks; admissibility requires faithfulness resolution.
- **Adversarial reader before Judge.** The Judge takes paired readings as input; building the Judge before the readers means stubbing input shape.
- **All of the above before the calibration smoke.** The smoke is the final integration gate; it exercises everything else against pre-existing Probe 1 telemetry.

Parallelizable (no hard dependency between them after Phase 0):

- The env-revision (Phase 1) is independent of every mirror-side phase.
- The shuffled-telemetry generator (Phase 3), the lesion scaffold (Phase 4), the pre-registration sink (Phase 5), and the faithfulness verifier (Phase 10) can be built in parallel.
- The held-out-criterion mechanism (Phase 6) depends on Phase 7's prompt-builder shape but its config surface can be drafted alongside.

Phased build (each phase is a unit of work):

| Phase | Work | Depends on |
|---|---|---|
| **0. Probe 2 schemas** | `StructuredReading` (`MIRROR_READING_V2_VERSION = "0.2.0"`), `PreRegistration` (`PRE_REG_SCHEMA_VERSION = "0.1.0"`), documented `is_sham: bool` field on `world_event` payload, new JSON Schema export `schemas/v0.2.0.json`. | — |
| **1. Env revision** | Default `start_cell=None` (random); `env_reset` payload includes `start_cell`; sham-perturbation flag-only path on `EnvServer`; agent observation byte-equal pre/post sham. | 0 |
| **2. Hierarchical digest** | `kind/observer/digest.py` extension: `build_hierarchical_digest`, `digest_episode_window`, `digest_drill_down`. Existing `build_digest` retained as `build_flat_digest` for Probe-1-compatible reads. | 0 |
| **3. Shuffled-telemetry generator** *(parallel with 4, 5, 10)* | `kind/observer/shuffle.py` with three protocols (`shuffle_within_episode`, `shuffle_across_episodes`, `decouple_action_state`). Reproducible from a shuffle seed. | 0 |
| **4. Lesion scaffold** *(parallel with 3, 5, 10)* | `RunnerConfig.lesion_kind: Literal[None, "ensemble_k1", "ensemble_constant"]`; runner threads it through to `LatentDisagreementEnsemble` construction; lesion shape recorded into run-level metadata. | 0 |
| **5. Pre-registration sink** *(parallel with 3, 4, 10)* | `kind/observer/pre_reg.py` with `PreRegistration` model and `PreRegSink` (JSONL append-only). Pre-reg records land at `runs/{run_id}/pre_reg/pre_reg.jsonl`. | 0 |
| **6. Held-out criterion mechanism** | `kind/mirror/criteria.py` with a registry of named criteria (prompt fragments) and an active-set toggle. Default active set excludes the held-out criterion until a designated checkpoint. | 0 |
| **7. Adversarial mirror caller** | Two readers built on top of the Phase 6 caller: `PhenomenologicalAdvocate` and `StatisticalSkeptic`, each producing a `StructuredReading` against the same digest. Different model families where API budget allows; same-model-different-prompts is the documented fallback. | 2, 5, 6 |
| **8. Judge / Arbiter** | `kind/mirror/judge.py`. Takes the digest plus two `StructuredReading` instances; returns a `JudgeRuling` with per-claim outcomes (`supported` / `absent` / `unresolved`). Mandate to flag agreement-without-evidence as unresolved. | 7 |
| **9. Stability-test runner** | `kind/mirror/stability.py`. Issues N paraphrased and reseeded calls of a reader; computes structured-field agreement; gates admission on ≥80% (default; see §6). | 7 |
| **10. Faithfulness verifier** *(parallel with 3, 4, 5)* | `kind/observer/eyeball.py` extension: `resolve_citation(stream, run_id, episode_or_step_range, scalar_field, claimed_value, tolerance) -> ResolutionResult`. | 0 |
| **11. Calibration smoke** | `scripts/smoke_probe2.py`. Runs the parallel-reader-with-Judge against `runs/probe1-20260503-123926/`. Includes a deliberately injected confabulation prompt to verify the protocol catches it. | 1–10 |
| **12. Gate tests + journal scaffold** | The named Probe 2 gate tests (§4); journal scaffolding for pre-registration entries, two-mode discipline, post-reading entries, Watts-default-applied-to-builder log. | All |

Phase 0 is the single load-bearing prerequisite. Schemas are versioned at `v0.2.0` for the mirror-reading stream and `v0.1.0` for the new pre-registration stream from day one; Probe 1 telemetry remains readable by the new code without migration.

---

## 2. Per-component specifications

Each entry: file paths, public interfaces, what the component reads/writes, what tests verify it, what it does *not* do at Probe 2.

### 2.1 Schema additions — `kind/observer/schemas.py`, `kind/mirror/structured.py`, `kind/observer/pre_reg.py`

**Files.** `kind/observer/schemas.py` (existing — adds `is_sham` payload documentation, no field-set changes); `kind/mirror/structured.py` (new — `StructuredReading`, `StructuredClaim`, `JudgeRuling`); `kind/observer/pre_reg.py` (new — `PreRegistration` model and `PreRegSink`); `schemas/v0.2.0.json` (frozen JSON Schema export, checked in).

**Public interface.**

```python
# kind/mirror/structured.py
MIRROR_READING_V2_VERSION: Final[str] = "0.2.0"

class StructuredClaim(BaseModel):
    claim: str
    cited_stream: Literal["agent_step", "dream_rollout", "replay_meta", "world_event"]
    cited_run_id: str
    cited_episode_range: tuple[int, int] | None
    cited_step_range: tuple[int, int] | None
    cited_scalar_field: str
    cited_value: float
    falsifier: str
    paraphrase_stability: float | None      # filled by Phase 9
    reseed_stability: float | None          # filled by Phase 9
    faithfulness_status: Literal["resolved", "off_by_tolerance", "unresolved", "not_checked"]
    judge_ruling: Literal["supported", "absent", "unresolved", "not_judged"]

class StructuredReading(BaseModel):
    schema_version: str = MIRROR_READING_V2_VERSION
    run_id: str
    timestamp_ms: int
    reader_role: Literal["advocate", "skeptic", "single"]
    paired_reading_id: str | None           # links Advocate↔Skeptic
    framework_anchor: Literal["buddhist_phenomenology", "predictive_processing", "null_statistics", "neutral"]
    baseline_flag: Literal["genuine", "shuffled_within_episode", "shuffled_across_episodes", "decoupled_action_state", "lesion_k1", "lesion_constant", "sham_aligned"]
    digest_run_id: str                      # the run the digest was built from
    digest_episode_range: tuple[int, int]
    claims: list[StructuredClaim]
    free_text_notes: str                    # legacy free-text channel for things that aren't structured claims

class JudgeRuling(BaseModel):
    schema_version: str = MIRROR_READING_V2_VERSION
    run_id: str
    timestamp_ms: int
    paired_reading_id: str
    advocate_id: str
    skeptic_id: str
    digest_run_id: str
    rulings: list[tuple[int, Literal["supported", "absent", "unresolved"], str]]   # (claim_index, ruling, ground)
    agreement_without_evidence_unresolved: list[int]                                 # indices

# kind/observer/pre_reg.py
PRE_REG_SCHEMA_VERSION: Final[str] = "0.1.0"

class PreRegistration(BaseModel):
    schema_version: str = PRE_REG_SCHEMA_VERSION
    run_id: str
    timestamp_ms: int
    criteria_active: list[str]                  # ids of criteria in the prompt
    criteria_held_out: list[str]                # ids of criteria not in the prompt
    signal_mappings: dict[str, list[str]]       # criterion_id -> list of telemetry signal expressions
    falsifiers: dict[str, str]                  # criterion_id -> falsifier text
    scalar_checks: dict[str, list[str]]         # criterion_id -> eyeball fields the claim must align with
    builder_mode: Literal["proponent", "skeptic"]
    expected_outcome: str
    substrate_decisions_off_table: list[str]    # Watts-default-applied-to-builder discipline

class PreRegSink:
    def __init__(self, dir: Path) -> None: ...
    def write(self, record: PreRegistration) -> None: ...
    def close(self) -> None: ...
```

**`is_sham` payload field on `world_event`.** No field-set change to `WorldEvent`; the existing `payload: dict[str, Any]` carries the field. The plan documents the convention: `payload["is_sham"]: bool` is set to `True` on builder-perturbation events emitted via the sham path, and absent (or `False`) on real perturbations. The eyeball helpers and the digest module honor the field; Phase 1 wires the env-server emission. The convention does not bump the schema version because the payload is intentionally schemaless (per Probe 1 plan §3.4).

**Reads / Writes.** Pure declarations + one new JSONL sink. Pre-reg records land at `runs/{run_id}/pre_reg/pre_reg.jsonl`; structured readings and judge rulings land at `runs/{run_id}/mirror/structured.jsonl` and `runs/{run_id}/mirror/judge.jsonl` (new sub-paths under the existing `mirror/` directory; the existing `readings.jsonl` is reserved for Probe 1's free-text `MirrorReading` and is not written in Probe 2).

**Tests.** Models import; `StructuredReading` round-trips through JSONL with all enum values; `PreRegistration` round-trips; `schemas/v0.2.0.json` export is byte-stable; the `is_sham` field convention is documented in `kind/observer/schemas.py`'s module docstring.

**Not at Probe 2.** No migration of Probe 1's `MirrorReading` records (they remain readable in their original form; nothing rewrites them); no schema-evolution runtime check (a reader encountering an unknown `schema_version` raises rather than auto-migrating).

### 2.2 Env revision — `kind/env/grid_world.py`, `kind/env/env_server.py`

**Files.** `kind/env/grid_world.py` (default config flip — `start_cell` default becomes `None`); `kind/env/env_server.py` (extended `_emit_env_reset` payload + new sham-perturbation method).

**Public interface.**

```python
# kind/env/grid_world.py
@dataclass(frozen=True)
class GridWorldConfig:
    ...
    start_cell: tuple[int, int] | None = None    # was (3, 3); flip is the Probe 2 revision

# kind/env/env_server.py
class EnvServer:
    ...
    def fire_sham_perturbation(self, mutator_label: str, payload: dict[str, Any]) -> None:
        """Emit a builder-perturbation world_event with payload['is_sham']=True
        and payload['sham_label']=mutator_label. Does NOT mutate the grid; the
        env state, agent observation, and stochasticity streams are unaffected.
        Used by Probe 2's calibration protocol §2.4(3)."""
```

The `_emit_env_reset` payload gains a `start_cell: tuple[int, int]` entry recording the cell sampled at reset (whether random or fixed). This is the reproducibility commitment for random starts: given the seed, every episode's start cell can be recovered from the JSONL.

**Reads / Writes.** Env-server reads and emits as before; sham path writes to `world_event` JSONL with `event_type="builder_perturbation"`, `source="builder"`, `payload={"is_sham": True, "sham_label": ..., "intended_mutator": ..., ...}`. The agent's observation tensor is not touched.

**Tests.**
- *Random-start reproducibility.* Same seed → same sequence of start cells across episodes; different seeds → different sequences. (Extends `tests/test_env_step.py`.)
- *Env_reset payload includes start_cell.* `tests/test_perturbation_hook.py` (or a new file) confirms the reset payload carries the start cell.
- *Sham mechanism is byte-clean.* Two `EnvServer` instances at the same seed: one calls `fire_sham_perturbation`, the other does not. Pixel observations are identical step-by-step. The world_event stream of the first contains a record with `payload.is_sham=True`; the second's does not. (Extends `tests/test_perturbation_hook.py`.)

**Not at Probe 2.** No change to the four real mutators. No change to internal stochasticity (regrowth, drift). No change to grid size, action space, or episode length. The `start_cell` becomes `None` *by default*; explicit fixed-start configs still work for one-off experiments. No `introduce_novel_object` mutator (excluded by environment synthesis on no-marker grounds; Probe 2 inherits this).

### 2.3 Hierarchical digest — `kind/observer/digest.py`

**Files.** `kind/observer/digest.py` (existing — extended).

**Public interface.**

```python
# Existing — kept, renamed in places for clarity
def build_flat_digest(rows: list[dict[str, Any]]) -> str: ...   # renamed from build_digest; Probe 1 path
def compact_record_repr(r: dict[str, Any], position: str) -> str: ...

# New — Probe 2 hierarchical digest
def build_hierarchical_digest(
    telemetry_dir: Path,
    *,
    n_episodes: int,
    flagged_only: bool = False,
    with_sham: bool = True,
) -> HierarchicalDigest: ...

@dataclass(frozen=True)
class HierarchicalDigest:
    run_summary: str                       # ~500 tokens: scalar trends, episode count, anomaly flags
    episode_mini_digests: dict[int, str]   # episode_id -> digest snippet
    flagged_anomalies: list[FlaggedAnomaly]  # episodes with KL spikes, entropy collapse, etc.
    world_event_timeline: str              # raw world_event entries (genuine + sham + reset)
    drill_down: DrillDownAccessor          # callable interface for fetching specific windows

class DrillDownAccessor:
    def fetch_window(self, episode_id: int, step_range: tuple[int, int]) -> str: ...
    def fetch_dream(self, dream_index: int) -> str: ...
    def fetch_world_event(self, event_index: int) -> str: ...
```

The base layer (run summary + episode mini-digests + flagged anomalies + world_event timeline) is what the parallel readers see at first call. The drill-down accessor is provided as a programmatic interface; the readers can request specific windows (Phase 7's prompts include explicit drill-down syntax the reader can use).

**Reads.** Parquet shards under `telemetry_dir/agent_step/`; JSONL under `telemetry_dir/world_event.jsonl`, `telemetry_dir/replay_meta.jsonl`; parquet under `telemetry_dir/dream_rollout/`.

**Writes.** Nothing — pure read-and-format.

**Tests.**
- *Hierarchical digest builds from Probe 1's existing run.* `runs/probe1-20260503-123926/telemetry/` exists; the test exercises `build_hierarchical_digest` against it and asserts the run-summary is non-empty, the episode mini-digests cover the requested range, the world-event timeline includes all 26 reset events.
- *Drill-down fetches a specific window.* Given a known episode and step range, the drill-down output contains the per-step scalars for that range and nothing outside it.
- *Sham events appear in timeline with `is_sham=True` distinguished.* The world-event timeline section labels sham events explicitly so the reader prompt can reason about them.

**Not at Probe 2.** No tool-use interface to the LLM (the drill-down accessor is callable from Python, not exposed as a tool API at Probe 2 — the prompt structure includes pre-fetched windows the builder selects; full dynamic tool-use is deferred). No streaming digest (the digest is fully built before the reader is called). No automatic anomaly classification (anomalies are flagged by simple thresholds; the reader interprets).

### 2.4 Shuffled-telemetry generator — `kind/observer/shuffle.py`

**Files.** `kind/observer/shuffle.py` (new).

**Public interface.**

```python
def shuffle_within_episode(
    telemetry_dir: Path, output_dir: Path, seed: int
) -> ShuffleManifest: ...

def shuffle_across_episodes(
    telemetry_dir: Path, output_dir: Path, seed: int
) -> ShuffleManifest: ...

def decouple_action_state(
    telemetry_dir: Path, output_dir: Path, seed: int
) -> ShuffleManifest: ...

@dataclass(frozen=True)
class ShuffleManifest:
    protocol: Literal["within_episode", "across_episodes", "decoupled_action_state"]
    seed: int
    source_run_id: str
    output_run_id: str
    episode_marginals_preserved: bool
    temporal_structure_broken: bool
    notes: str
```

Each protocol reads the source telemetry, applies its shuffle to `agent_step` records, writes a new parquet directory under `output_dir`, and emits a `ShuffleManifest` describing what was shuffled. The shuffled dir is structurally indistinguishable from a real telemetry dir (same schema, same envelope, same shard layout) so the digest module reads it without modification.

- `shuffle_within_episode`: within each episode, randomly permute the order of records. Episode marginals preserved; within-episode temporal structure broken.
- `shuffle_across_episodes`: permute the order of entire episodes. Within-episode structure preserved; across-episode comparisons broken.
- `decouple_action_state`: keep the per-step state fields (`h_t`, `z_t`, etc.) in original order; shuffle the `action_t` field independently. Action-conditional structure broken; everything else preserved.

**Reads.** Source telemetry directory.

**Writes.** Output telemetry directory with shuffled `agent_step` parquet shards plus a `shuffle_manifest.json`. The `world_event` and `replay_meta` JSONL streams are copied unchanged (the shuffle is on the agent stream only; baselining the world events would defeat the test).

**Tests.**
- *Within-episode shuffle preserves marginals.* Per-episode mean/std of `kl_aggregate_t` matches between source and shuffled output.
- *Across-episode shuffle breaks ordering.* Episode IDs in the shuffled output follow a different sequence; per-episode statistics are unchanged.
- *Action-state decoupling.* The action distribution per episode is preserved; the empirical action-conditional next-state distribution is broken (action-state mutual information drops).
- *Determinism.* Same input + same seed → same shuffled output, byte-stable.

**Not at Probe 2.** No shuffle on `dream_rollout`, `world_event`, or `replay_meta` (the agent_step stream is the load-bearing one for criterion readings). No partial / probabilistic shuffles (full shuffle each protocol).

### 2.5 Lesion scaffold — `kind/training/runner.py` (extended), `kind/agents/ensemble.py` (extended)

**Files.** `kind/training/runner.py` (existing — `RunnerConfig` gains a `lesion_kind` field); `kind/agents/ensemble.py` (existing — `LatentDisagreementEnsemble` constructor accepts a `lesion_kind` argument that propagates to behavior).

**Public interface.**

```python
# kind/training/runner.py
@dataclass(frozen=True)
class RunnerConfig:
    ...
    lesion_kind: Literal[None, "ensemble_k1", "ensemble_constant"] = None

# kind/agents/ensemble.py
class LatentDisagreementEnsemble(nn.Module):
    def __init__(
        self, ..., lesion_kind: Literal[None, "ensemble_k1", "ensemble_constant"] = None
    ) -> None: ...
    # When lesion_kind == "ensemble_k1": K is silently set to 1, disagreement is constant 0.
    # When lesion_kind == "ensemble_constant": K stays at 5, but disagreement output is replaced
    # by a fixed scalar (e.g., the mean of the first 100 step's disagreement on a non-lesion run).
```

The lesion shape is recorded into the run-level metadata (a `lesion_kind` field on the run summary the eyeball helpers print, plus a payload field on the run-start `world_event` if convenient). Probe 2 runs at most one lesion run per Probe 2 round; the lesion is a calibration test, not a primary experiment.

**Reads / Writes.** No new I/O. The runner already writes telemetry; the lesion run produces the same four streams with the same schema, with the disagreement scalar reflecting the lesion shape.

**Tests.**
- *Lesion run produces telemetry with constant disagreement.* A short integration smoke (CPU, tiny sizes) at `lesion_kind="ensemble_k1"` produces an `agent_step` stream where `intrinsic_signal_t` is identically zero (or a fixed constant under `ensemble_constant`).
- *Non-lesion behavior is unchanged.* `lesion_kind=None` runs exactly as Probe 1 (regression check).
- *Lesion kind is recorded in run metadata.* The run summary or telemetry envelope identifies the lesion shape so the digest can read it correctly.

**Not at Probe 2.** No decoder lesion (deferred — Probe 2 default lesion is on the actor's signal, not the world model's reconstruction). No perturbation-surface lesion (Probe 4 territory). No real-time lesion toggling (the lesion is set at runner construction).

### 2.6 Pre-registration sink — `kind/observer/pre_reg.py`

**Files.** `kind/observer/pre_reg.py` (new).

**Public interface.** `PreRegistration` and `PreRegSink` per §2.1. The model carries the seven discipline elements per criterion (criteria_active, criteria_held_out, signal_mappings, falsifiers, scalar_checks, builder_mode, expected_outcome) plus the Watts-default-applied-to-builder field (substrate_decisions_off_table).

**Reads.** Nothing.

**Writes.** `runs/{run_id}/pre_reg/pre_reg.jsonl`, append-only.

**Tests.**
- *Round-trip.* Write a `PreRegistration`, read it back, schema-validate, equal.
- *Append-only behavior.* Multiple writes to the same path produce a JSONL with one record per line, no overwrites, no truncation.
- *Required-field validation.* Missing fields raise `ValidationError`; empty `criteria_active` is allowed (the case of all criteria held out for a pre-protocol-test phase).

**Not at Probe 2.** No automatic comparison of pre-registration to actual reading (that's the journal's job, by hand). No CLI for filling in a pre-registration (the journal-entry template is the surface; the sink just persists structured records the builder produces).

### 2.7 Held-out criterion mechanism — `kind/mirror/criteria.py`

**Files.** `kind/mirror/criteria.py` (new).

**Public interface.**

```python
@dataclass(frozen=True)
class Criterion:
    id: str
    name: str
    framework_anchor: Literal["buddhist_phenomenology", "predictive_processing"]
    advocate_prompt_fragment: str
    skeptic_prompt_fragment: str
    pre_registered_signal_mappings: list[str]
    falsifier: str
    descope_reason: str | None = None     # if the criterion is descoped (e.g., second-order volition)

CRITERION_REGISTRY: dict[str, Criterion] = {
    "reflexive_attention_triplet": Criterion(...),
    "equanimity_perturbation_recovery": Criterion(...),
    "second_order_volition": Criterion(... descope_reason="..."),
}

def active_prompt_fragment(
    role: Literal["advocate", "skeptic"], active_criteria: list[str]
) -> str:
    """Build the prompt fragment for the given role from the active criteria."""

def held_out(active_criteria: list[str]) -> list[str]:
    """Return the registry ids that are NOT in active_criteria (i.e., held out)."""
```

The registry holds the three criteria the synthesis names. Second-order volition is in the registry with its descope_reason populated; it never enters an active set at Probe 2 (the mechanism still represents it so the descoping is visible in code, not just in prose). The reflexive-attention triplet is the default held-out criterion (per synthesis §4 default); the active set at the start of Probe 2 is `["equanimity_perturbation_recovery"]`. At a designated late checkpoint, the active set widens to include reflexive-attention.

**Reads.** Nothing.

**Writes.** Nothing — pure prompt-fragment construction.

**Tests.**
- *Active set toggles produce different prompts.* Calling `active_prompt_fragment("advocate", ["equanimity_perturbation_recovery"])` returns text not containing the reflexive-attention fragment; calling with both returns text containing both.
- *Descoped criteria never appear in prompts.* Even if `"second_order_volition"` is passed in `active_criteria`, the function emits a prompt fragment that includes the descope rationale, not the criterion's signal mappings.
- *Registry is forward-compatible.* Adding a new criterion does not break existing prompt construction.

**Not at Probe 2.** No automatic activation of held-out criteria (the builder explicitly toggles via config when the designated checkpoint is reached, and journals the reason). No criterion-revision protocol in code (revision goes through journal first, then code; per design notes' mirror update discipline).

### 2.8 Adversarial mirror caller — `kind/mirror/adversarial.py`

**Files.** `kind/mirror/adversarial.py` (new — composes Phase 6's `MirrorCaller` with two role-tilted prompts).

**Public interface.**

```python
class PhenomenologicalAdvocate:
    def __init__(self, model: str, max_tokens: int, api_key: str | None = None) -> None: ...
    def read(
        self,
        digest: HierarchicalDigest,
        active_criteria: list[str],
        run_id: str,
        baseline_flag: Literal["genuine", "shuffled_within_episode", ...] = "genuine",
    ) -> StructuredReading: ...

class StatisticalSkeptic:
    # Same shape as PhenomenologicalAdvocate, different prompt fragments and framework_anchor
    ...

class AdversarialPair:
    def __init__(
        self, advocate: PhenomenologicalAdvocate, skeptic: StatisticalSkeptic
    ) -> None: ...
    def read_pair(
        self, digest: HierarchicalDigest, active_criteria: list[str], run_id: str,
        baseline_flag: str = "genuine",
    ) -> tuple[StructuredReading, StructuredReading]: ...
```

The Advocate's prompt frames the criterion-finding task in Buddhist-phenomenology language (manasikara, upekkha; witness-as-distinct-from-content; pre-conceptual turning toward); mandate is to argue the strongest case the cited telemetry supports for the criterion's presence. The Skeptic's prompt frames the same task in null-hypothesis-statistics language (regrowth/drift baselines; ensemble initialization noise; known degenerate dynamics); mandate is to argue the patterns are artefacts. Both readers must ground every claim in `(stream, run_id, episode/step_range, scalar_field, value)`.

Different model families where API budget allows: Advocate uses one (e.g., Claude); Skeptic uses another (e.g., GPT-class). Same-model-different-prompts is the documented fallback when only one family is available; the synthesis records this is a weaker-but-acceptable variant. The choice is recorded into the `StructuredReading.framework_anchor` and the model field.

**Reads.** `HierarchicalDigest` from `kind/observer/digest.py`.

**Writes.** Each call appends a `StructuredReading` to `runs/{run_id}/mirror/structured.jsonl`. The `paired_reading_id` field is populated identically on both members of the pair (a fresh UUID per pair) so the Judge can find them.

**Tests.**
- *Both readers produce structured readings on the same digest.* Mock the API; verify both call paths return valid `StructuredReading` instances with non-empty `claims`.
- *Framework anchors are distinct.* Advocate's `framework_anchor == "buddhist_phenomenology"`; Skeptic's `framework_anchor in {"null_statistics", "predictive_processing"}`.
- *Paired-reading-id matches.* `read_pair` returns two readings with the same `paired_reading_id` and different `reader_role` values.
- *Baseline flag propagates.* Calling with `baseline_flag="shuffled_within_episode"` produces readings whose `baseline_flag` field is set accordingly; the prompts include a notice that the digest is from shuffled telemetry (so the reader is not pretending to find structure in noise unknowingly).

**Not at Probe 2.** No multi-round debate (settled). No sequential reader/critic (the synthesis rejected this on anchoring-bias grounds). No tool-use API for drill-down (Phase 2's drill-down accessor is exposed via pre-fetched windows in the prompt, not as a callable tool the LLM invokes).

### 2.9 Judge / Arbiter — `kind/mirror/judge.py`

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

The Judge sees the digest plus both readings. Its prompt mandates:

1. For each `StructuredClaim` from the Advocate: rule `supported` only if the cited evidence resolves under faithfulness check AND the Skeptic's null hypothesis is overwhelmed by the citation; rule `absent` if the Skeptic's null hypothesis stands; rule `unresolved` if neither outcome can be determined from the cited evidence.
2. Flag agreement-without-evidence as `unresolved` explicitly. If both readers cite the same scalar but neither's citation resolves, the claim is unresolved, not supported.
3. Ground every ruling in a quoted citation from the digest. The Judge's output is structured: `(claim_index, ruling, ground_text)` triples.

**Reads.** Both `StructuredReading` instances and the `HierarchicalDigest`.

**Writes.** Each call appends a `JudgeRuling` to `runs/{run_id}/mirror/judge.jsonl`.

**Tests.**
- *Judge rules `unresolved` on agreement-without-evidence.* Mock both readings to cite identical scalars without resolution; verify the Judge's output places the claim in `agreement_without_evidence_unresolved`.
- *Judge rules `supported` on a claim the Skeptic does not refute and whose citation resolves.* Mock readings; faithfulness check resolves; Skeptic's refutation is blank or weak; Judge rules `supported`.
- *Judge rules `absent` on a claim the Skeptic refutes successfully.* Mock readings; Skeptic's null hypothesis cites a contradicting scalar that resolves; Judge rules `absent`.
- *Judge cannot rule on uncited claims.* If a claim has empty `cited_value`, the Judge rules `unresolved` and notes the gap.

**Not at Probe 2.** No multi-round Judge dialogue (one ruling per pair). No automatic re-prompting if the Judge produces invalid output (validation errors raise; the build phase decides to retry or log). No Judge-of-Judges (deferred indefinitely).

### 2.10 Stability-test runner — `kind/mirror/stability.py`

**Files.** `kind/mirror/stability.py` (new).

**Public interface.**

```python
@dataclass(frozen=True)
class StabilityResult:
    paraphrase_agreement: float          # ∈ [0, 1]
    reseed_agreement: float              # ∈ [0, 1]
    n_paraphrases: int
    n_reseeds: int
    structured_field_agreement_per_claim: list[float]
    admissible: bool                     # True iff both ≥ paraphrase_threshold and reseed_threshold

def stability_check(
    reader: PhenomenologicalAdvocate | StatisticalSkeptic,
    digest: HierarchicalDigest,
    active_criteria: list[str],
    run_id: str,
    *,
    n_paraphrases: int = 3,
    n_reseeds: int = 3,
    paraphrase_threshold: float = 0.80,
    reseed_threshold: float = 0.80,
) -> StabilityResult: ...
```

Issues `n_paraphrases × n_reseeds` reader calls; structured-field agreement is computed on the per-claim fields (`cited_signal`, `cited_step_range` overlap, `claim` text similarity via embedding or token overlap — open during build). The result populates `paraphrase_stability` and `reseed_stability` on the original `StructuredReading`'s claims.

**Reads.** Nothing beyond what the reader reads.

**Writes.** Stability results land alongside the `StructuredReading` they qualify (the reader's writer fills in the stability fields after the stability check returns). Optionally writes a per-stability-check JSONL record at `runs/{run_id}/mirror/stability.jsonl` for audit.

**Tests.**
- *Stability score is computable from N readings.* Given three mocked readings with identical claims, agreement ≈ 1.0; with completely different claims, ≈ 0.
- *Threshold gating.* `admissible` is `True` only when both axes meet threshold.
- *Paraphrase set is non-trivial.* The default paraphrase set (defined as a list in the module) has at least three different prompt-wording variants that preserve criteria.

**Not at Probe 2.** No automatic prompt-paraphrase generation by an LLM (the paraphrases are hand-written and frozen; build-phase tunes them). No per-claim adaptive thresholding.

### 2.11 Faithfulness verifier — `kind/observer/eyeball.py` (extended)

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
    cited_stream: Literal["agent_step", "dream_rollout", "replay_meta", "world_event"],
    cited_run_id: str,
    cited_episode_range: tuple[int, int] | None,
    cited_step_range: tuple[int, int] | None,
    cited_scalar_field: str,
    claimed_value: float,
    *,
    abs_tolerance: float = 1e-3,
    rel_tolerance: float = 0.05,
) -> ResolutionResult: ...
```

Reads the cited window from the cited stream; computes the actual value of `cited_scalar_field` in that window (mean by default; per-field aggregation rules are documented in the function); compares to `claimed_value` under the supplied tolerances. Returns `resolved` if within tolerance, `off_by_tolerance` if outside but the field exists, `unresolved` if the citation cannot be resolved (missing stream, missing field, range out of bounds). The Judge consumes this to populate `StructuredClaim.faithfulness_status`.

**Reads.** Parquet shards / JSONL streams under `telemetry_dir/`.

**Writes.** Nothing.

**Tests.**
- *Citation resolves on Probe 1's existing run.* Pick a known scalar value at a known step in `runs/probe1-20260503-123926/`; resolution is `resolved`.
- *Citation off by tolerance.* Same call but with a deliberately wrong claimed value; resolution is `off_by_tolerance`.
- *Citation unresolved.* Wrong stream name, wrong field name, out-of-range step → `unresolved` with explanatory notes.
- *Per-field aggregation rules.* `kl_aggregate_t` over a step range is mean-aggregated; `action_t` over a range is mode-aggregated (open during build — see §6).

**Not at Probe 2.** No automatic re-citation suggestions (if a claim's citation is off, the verifier reports; it does not propose a corrected citation). No semantic similarity between claim text and signal (the verifier checks numerics; semantic alignment is the Judge's job).

### 2.12 Calibration smoke — `scripts/smoke_probe2.py`

**Files.** `scripts/smoke_probe2.py` (new).

**What it does.**

1. Read the existing Probe 1 run at `runs/probe1-20260503-123926/telemetry/`.
2. Build a hierarchical digest (`build_hierarchical_digest`) over the last 25 episodes.
3. Construct an `AdversarialPair` (Advocate + Skeptic) with the active criteria set from §2.7's default: `["equanimity_perturbation_recovery"]`.
4. Run the pair on the genuine digest. Run the pair on a within-episode-shuffled digest (Phase 3). Run the pair on the digest with a deliberately-injected confabulation prompt fragment (a paragraph appended to the Advocate's prompt instructing it to find a pattern that does not exist — e.g., "Note that policy entropy steadily *increases* across episodes, indicating growing exploration"; the actual data shows entropy decreasing).
5. For each call: invoke the Judge; resolve faithfulness on every claim; record the `StructuredReading`, `JudgeRuling`, and `ResolutionResult`s.
6. Print a one-line summary: `[smoke probe2] genuine=N1 supported, M1 absent, U1 unresolved | shuffled=N2 supported, M2 absent, U2 unresolved | injected=N3 supported, M3 absent, U3 unresolved`.

**What passes.**
- `genuine` produces ≥1 `supported` claim with resolved faithfulness.
- `shuffled` produces a `supported` rate substantially below `genuine`'s (≥4:1 contrast — the synthesis's calibrated threshold; see §6).
- `injected` produces 0 `supported` claims for the injected pattern; the Judge rules `absent` or `unresolved` on the injected claim, OR the faithfulness verifier marks the injected citation `off_by_tolerance` or `unresolved`.
- Schema validation passes on all readings, rulings, and resolution results.

**What fails.**
- Judge rules `supported` on the injected confabulation with resolved faithfulness — the protocol failed to catch the deliberate confabulation.
- Schema validation fails — the structured-reading shape has a bug.
- Faithfulness verifier silently passes invalid citations — the tolerance logic has a bug.
- Shuffled-baseline `supported` rate equals or exceeds genuine — the prompts are doing the work, not the data.

**Failure response.** None of these are calibration failures of Io; all are protocol failures the build phase fixes. If the injected confabulation cannot be caught with the protocol as designed, the Skeptic's prompt is under-tuned and needs revision; the journal records what was tried.

---

## 3. Schemas as first-class artifacts

Pydantic v2 models. Two new versioned streams for Probe 2; one documented payload-field convention extending Probe 1's `world_event`.

### 3.1 `StructuredReading` — `kind/mirror/structured.py`, version `0.2.0`

| Field | Type | Notes |
|---|---|---|
| `schema_version` | `str` | `"0.2.0"`, the Probe 2 mirror-reading version |
| `run_id` | `str` | the Probe 2 run that produced the reading (may differ from `digest_run_id`) |
| `timestamp_ms` | `int` | wallclock at reading-creation |
| `reader_role` | `Literal["advocate", "skeptic", "single"]` | `"single"` reserved for Probe 1 backwards-compat callers |
| `paired_reading_id` | `str \| None` | UUID matching Advocate↔Skeptic |
| `framework_anchor` | `Literal["buddhist_phenomenology", "predictive_processing", "null_statistics", "neutral"]` | |
| `baseline_flag` | enum (genuine, shuffled_*, lesion_*, sham_aligned) | which baseline the digest came from |
| `digest_run_id` | `str` | the run the digest was built from |
| `digest_episode_range` | `tuple[int, int]` | |
| `claims` | `list[StructuredClaim]` | structured per-claim list |
| `free_text_notes` | `str` | non-structured observations the reader produced |

`StructuredClaim` carries `claim`, `cited_stream`, `cited_run_id`, `cited_episode_range`, `cited_step_range`, `cited_scalar_field`, `cited_value`, `falsifier`, `paraphrase_stability`, `reseed_stability`, `faithfulness_status`, `judge_ruling`. Stability and faithfulness fields are nullable until the relevant phases populate them.

### 3.2 `JudgeRuling` — `kind/mirror/judge.py`, version `0.2.0`

| Field | Type | Notes |
|---|---|---|
| `schema_version` | `str` | `"0.2.0"`, same lineage as `StructuredReading` |
| `run_id` | `str` | |
| `timestamp_ms` | `int` | |
| `paired_reading_id` | `str` | links to the Advocate/Skeptic pair |
| `advocate_id` / `skeptic_id` | `str` | individual reading IDs (not UUIDs of the records, but identifying handles) |
| `digest_run_id` | `str` | |
| `rulings` | `list[tuple[int, ruling, ground]]` | per-claim outcome with the ground text the Judge cited |
| `agreement_without_evidence_unresolved` | `list[int]` | claim indices flagged for agreement-without-evidence |

### 3.3 `PreRegistration` — `kind/observer/pre_reg.py`, version `0.1.0`

| Field | Type | Notes |
|---|---|---|
| `schema_version` | `str` | `"0.1.0"` (new stream, fresh lineage) |
| `run_id` | `str` | |
| `timestamp_ms` | `int` | |
| `criteria_active` | `list[str]` | criterion ids in the prompt at this checkpoint |
| `criteria_held_out` | `list[str]` | criterion ids deliberately not in the prompt |
| `signal_mappings` | `dict[str, list[str]]` | criterion_id → telemetry signal expressions |
| `falsifiers` | `dict[str, str]` | criterion_id → falsifier text |
| `scalar_checks` | `dict[str, list[str]]` | criterion_id → eyeball fields the claim must align with |
| `builder_mode` | `Literal["proponent", "skeptic"]` | |
| `expected_outcome` | `str` | what the builder expects the reading to show |
| `substrate_decisions_off_table` | `list[str]` | Watts-default-applied-to-builder discipline |

### 3.4 `world_event` payload extension — convention only, no schema bump

The `payload: dict[str, Any]` field of `world_event` carries an `is_sham: bool` entry on builder-perturbation events emitted via the sham path. The convention is documented in:

- `kind/observer/schemas.py` module docstring.
- `kind/env/env_server.py` `fire_sham_perturbation` docstring.
- `kind/observer/eyeball.py` digest helpers honor the field when summarizing world events.

`world_event` schema version stays at `0.1.0`. The payload dict's schemaless flexibility is what allows this without a bump (per Probe 1 plan §3.4).

### 3.5 Versioning and forward-compatibility

- `MIRROR_READING_V2_VERSION = "0.2.0"` is the new mirror-reading version. Probe 1's `MIRROR_READING_SCHEMA_VERSION = "0.1.0"` (in `kind/mirror/caller.py`) stays for backward-readability of Probe 1 records.
- `PRE_REG_SCHEMA_VERSION = "0.1.0"` is the new pre-registration version.
- The frozen JSON Schema export at `schemas/v0.2.0.json` covers all four core models (`AgentStep`, `DreamRollout`, `ReplayMeta`, `WorldEvent`) plus the three new Probe 2 models (`StructuredReading`, `JudgeRuling`, `PreRegistration`).
- **Forward to Probe 3.** `StructuredReading` reserves no explicit `state_kind` field, but the `claims` list is open: a Probe 3 claim citing dream-state telemetry is just a claim with `cited_stream="dream_rollout"`. If state-typed reading is later wanted as a top-level field, it can be added at version `0.3.0` without breaking 0.2.0 readers.
- **Forward to Probe 4.** `world_event` payloads already distinguish builder-source from internal-stochasticity (the `source` and `event_type` fields). Probe 4's distinguishability fields will be additional payload entries on real (non-sham) builder-perturbations: timestamps, mutator parameters, statistical signature features. The schema accommodates them under the existing `payload: dict[str, Any]`.

---

## 4. Test scaffolding — Probe 2 gate tests

The Probe 1 synthesis named five gate tests; Probe 2 is structurally heavier and needs a wider gate. Listed below: nine named gate tests plus the calibration smoke. Test layout convention follows Probe 1 — one file per concern, fixtures in `tests/conftest.py`, no shared module state.

| # | Name | File | What it checks | Fixtures |
|---|---|---|---|---|
| 1 | parallel mirror caller | `tests/test_adversarial_caller.py` | `AdversarialPair.read_pair` produces two `StructuredReading` instances on the same digest with matching `paired_reading_id`, distinct `framework_anchor`, distinct `reader_role`. | mocked LLM clients |
| 2 | judge ruling | `tests/test_judge.py` | `Judge.rule` produces a `JudgeRuling` with one `(claim_index, ruling, ground)` triple per claim; agreement-without-evidence is correctly flagged unresolved. | mocked LLM client; pre-built Advocate/Skeptic readings |
| 3 | structured-reading schema | `tests/test_structured_reading.py` | All `StructuredReading`, `StructuredClaim`, `JudgeRuling` fields validate; round-trip through JSONL is byte-stable; `schema_version` equals `0.2.0`. | `tmp_path` |
| 4 | pre-registration sink | `tests/test_pre_reg.py` | `PreRegSink` writes/reads cleanly; `PreRegistration` validates required fields; missing `falsifiers` for an active criterion raises. | `tmp_path` |
| 5 | shuffled-telemetry generator | `tests/test_shuffle.py` | All three protocols produce structurally valid output; within-episode preserves marginals; across-episode preserves per-episode statistics; action-state decoupling preserves action distribution; determinism under seed. | `tmp_path`; small synthetic telemetry fixture |
| 6 | sham-perturbation hook | `tests/test_sham_perturbation.py` (extends `test_perturbation_hook.py`) | `fire_sham_perturbation` emits a `world_event` with `payload.is_sham=True`; the agent's observation byte-equals the parallel non-sham env's observation; grid state byte-equals pre/post. | `env_server_with_sink` |
| 7 | lesion runner | `tests/test_lesion.py` | `RunnerConfig.lesion_kind="ensemble_k1"` produces an `agent_step` stream with `intrinsic_signal_t` constant; non-lesion runs are unchanged (regression check). | `tiny_runner` |
| 8 | faithfulness verifier | `tests/test_faithfulness.py` | `resolve_citation` returns `resolved` on a known scalar; `off_by_tolerance` on a deliberately wrong value; `unresolved` on a missing field; per-field aggregation rules are documented and tested. | `runs/probe1-20260503-123926/` (read-only fixture) |
| 9 | hierarchical digest | `tests/test_hierarchical_digest.py` | `build_hierarchical_digest` runs against `runs/probe1-20260503-123926/`; run summary, episode mini-digests, world-event timeline are populated; drill-down accessor returns specific windows. | same fixture |
| **smoke** | calibration smoke | `scripts/smoke_probe2.py`, sanity test in `tests/test_smoke_probe2_script.py` | The full pipeline runs end-to-end against pre-existing telemetry; injected confabulation is caught (`absent` or `unresolved`); shuffled baseline produces ≥4:1 contrast against genuine. | `runs/probe1-20260503-123926/`; `GEMINI_API_KEY` (skipped if absent) |

A **gate-summary meta-test** (`tests/test_gate_summary.py`, the existing Probe 1 file extended) parametrizes over the nine gate tests and asserts each exists by name and is callable. If a future refactor renames or removes a gate, the meta-test fails loudly.

The integration smoke at gate-time uses the existing `runs/probe1-20260503-123926/` directory as a read-only fixture. Probe 2 does not generate new agent-step telemetry as a precondition for its own tests; the substrate is settled and Probe 1's run is the canonical input. Probe 2 *does* generate new pre-registration, structured-reading, and judge-ruling records as outputs.

---

## 5. Day-one calibration smoke

The Probe 1 synthesis flagged MPS performance for RSSM operations; Probe 1's smoke validated that. Probe 2's analog flags **whether the calibration protocol actually catches confabulations** — the equivalent unbenchmarked unknown. The day-one smoke runs the full parallel-mirror-with-Judge against Probe 1's existing telemetry with one deliberately-injected confabulation, and is what tells us whether the protocol is operationally tractable.

**File.** `scripts/smoke_probe2.py`. Sanity test at `tests/test_smoke_probe2_script.py` confirms the script exists and exposes `main()`.

**Pass conditions** (per §2.12 above, recapitulated):

1. Genuine reading: ≥1 claim ruled `supported` with resolved faithfulness on Probe 1's actual telemetry.
2. Shuffled-baseline reading: `supported` rate ≤ 25% of genuine's `supported` rate (the calibrated 4:1 contrast threshold).
3. Injected-confabulation reading: 0 `supported` claims for the injected pattern, OR the injected citation fails faithfulness check, OR the Judge rules `unresolved`.
4. All schema validations pass; all writes round-trip; no NaN/Inf in any computed scalar.

**Failure-response semantics.** A failure here means the protocol needs revision, not the substrate. The build phase reads what surfaced — was the Skeptic's prompt under-tuned? did faithfulness tolerance let an off-by-X claim through? did the Judge rule `supported` despite the Skeptic's refutation? — and revises the relevant prompt or threshold. This is the same posture Probe 1's smoke had toward MPS performance: the smoke surfaces, the build phase fixes, the journal records.

**Cost.** The smoke runs 6 mirror calls (Advocate × 3 baselines + Skeptic × 3 baselines) plus 3 Judge calls. At weekly Probe 2 cadence with default models, this is ~9 API calls per smoke; bounded.

**What the smoke does not test.** Stability across paraphrases or reseeds (Phase 9's runner exercises this; the smoke runs single calls). The held-out criterion mechanism (the smoke uses the default active set with reflexive-attention held out; the held-out-criterion test runs separately when the active set widens). Lesion runs (Phase 7's lesion gate test exercises this; the smoke uses a non-lesion run as input).

---

## 6. Open-during-build decisions with defaults

The synthesis named several "open during build" questions. The plan inherits each as a decision with a sensible default; build phase tunes empirically. The synthesis explicitly flags these as stance calls, not literature-canonical numbers; the plan does not pretend otherwise.

| Question | Default | Revisit when |
|---|---|---|
| Which criterion to hold out | **`reflexive_attention_triplet`** | if equanimity proves more prompt-shapeable in practice, swap |
| Paraphrase-stability threshold | **0.80** | if no reading admits, lower to 0.70; if every reading admits regardless of quality, raise |
| Reseed-stability threshold | **0.80** | same as paraphrase |
| Number of paraphrases | **3** | if stability scores are noisy at N=3, raise to 5 |
| Number of reseeds | **3** | same |
| Shuffled-baseline `supported`-rate ratio | **≤ 25% of genuine** (4:1 contrast) | if shuffled rate is 0 across many runs, the protocol is over-strict — relax tolerance; if shuffled rate matches genuine, prompts are doing the work |
| Faithfulness `abs_tolerance` | **1e-3** | per-field; tighten on integer-valued fields |
| Faithfulness `rel_tolerance` | **5%** | tighten on tightly-bounded scalars |
| Per-field aggregation for citation resolution | mean for continuous, mode for discrete, count for events | revisit if a claim cites a non-aggregable shape |
| Sham-perturbation cadence | **1 sham per 10 real perturbations** at calibration cadence | up if false positives on shams; down if cost dominates |
| Lesion shape | **`ensemble_k1`** | if K=1 lesion is undistinguishable from K=5 (both agreements unchanged), revisit `ensemble_constant` |
| Number of episodes in digest | **last 25** | up if criterion readings need broader baseline; down if context dominates |
| Number of paraphrases per stability check | **3** | as above |
| External human reader cadence | **≥1 milestone** (end of Probe 2; possibly post-Phase-2b before/after) | if availability changes, increase |
| Active-set widening checkpoint | **after the first three pre-registered readings on the default active set** | empirical |
| Judge model family | **third family if budget allows; otherwise same family as Skeptic with neutral framing** | if cost dominates, fall back to single-family Judge |

Verifying Gemini's 2026 citations: the synthesis flagged four references (Xu et al. on causal mediation analysis; Zheng et al. on SVE-ASCII; Sisodia on AI Observability; De Lima/Yang on YIELD) as worth verifying before the protocol leans on them. Verification is a build-phase task: each citation is checked for existence and content match against the synthesis's characterization. If any cannot be verified or substantially mischaracterizes, the protocol element it underwrote (CMA-style operational distinguishing of DCE vs ICE; canonical ASCII-legibility framing; layered observability framing; canonical shuffled-baseline framing) is treated as constructed, not literature-canonical, and the journal records the gap. The plan does not block on verification; the build phase records what surfaced.

The defaults are starting points. The smoke informs the first revision. None of them is a decision the plan asks the builder to settle before code is written.

---

## 7. The calibration protocol — code and discipline

The synthesis's seven calibration elements are a mix of code (mechanical) and discipline (journaled). Probe 1's plan was almost entirely code; Probe 2's plan must be honest about which is which because the discipline elements are deliverables enforced by being written down clearly enough for future-the-builder to follow.

| Element | Code surface | Discipline surface |
|---|---|---|
| 1. Pre-registration | `kind/observer/pre_reg.py` (`PreRegSink`); `PreRegistration` model | `docs/workingjournal/probe2.md` template. Each pre-reading entry names criteria, signal mappings, falsifiers, scalar checks, builder mode, expected outcome, and substrate decisions off the table. Required before each adversarial pass. |
| 2. Shuffled-telemetry baseline | `kind/observer/shuffle.py` (three protocols); `StructuredReading.baseline_flag` field | None — purely code. |
| 3. Sham-perturbation test | `EnvServer.fire_sham_perturbation`; `world_event.payload.is_sham` convention; eyeball helpers honor flag | Builder fires the sham at intervals during runs (or post-hoc against existing telemetry by editing world_event JSONL — flagged below as out of scope). Cadence is journaled. |
| 4. Lesion baseline | `RunnerConfig.lesion_kind`; `LatentDisagreementEnsemble` lesion arms | Builder runs at most one lesion run per Probe 2 round; the run's purpose, lesion shape, and read interpretation are journaled. |
| 5. Paraphrase- & reseed-stability | `kind/mirror/stability.py` (`stability_check`, `StabilityResult`) | Builder selects the paraphrase set; the set is checked into the module and revisited only with journaled reason (mirror update discipline applies). |
| 6. Held-out criterion | `kind/mirror/criteria.py` (registry + active-set toggle) | Builder explicitly toggles active set; the toggle event is journaled with the trigger (designated checkpoint reached; not "I want to check now"). |
| 7. Faithfulness check | `kind/observer/eyeball.py::resolve_citation` | None — purely code. |

The Watts-default-applied-to-builder discipline (synthesis §2.5(8)) is **journaled, not code-enforced**. The pre-registration record carries `substrate_decisions_off_table: list[str]`; the builder fills it before each reading. If a reading would prompt a substrate revision, the journal records the trigger, the framework lens, and an explicit check against project-document settled commitments. There is no mechanical gate; the discipline is in being honest enough to write it down.

The two-mode builder discipline (synthesis §2.5(6)) is **journaled, not code-enforced**. The `PreRegistration.builder_mode` field captures the mode; journal entries record which mode was in force when interpretive choices were made. No code prevents a proponent-mode builder from rationalizing.

The no-silent-criterion-revision discipline (synthesis §2.5(7); design notes mirror update discipline) is **partly code, partly journaled**. Code: the `Criterion` registry is a Python literal; revisions are git-visible. Journal: each revision must name the external trigger (new external learning) and the reason; revisions made in response to what the system produced are forbidden. The journal entry precedes the registry change.

The plan does not pretend the discipline elements are testable in the same way code is. They are deliverables in the form of a journal-entry shape that future-the-builder follows because the alternative (forgetting why the protocol exists) is the failure mode the design notes have already named.

---

## 8. Co-design enforcement at the structural level

The synthesis surfaced that the research outputs underweighted the Watts-default-applied-to-builder side of co-design. The plan reflects this at three layers:

**Layer 1 — pre-registration as constraint, not formality.** The `PreRegistration` record carries `substrate_decisions_off_table`. Before each adversarial pass, the builder writes down which substrate decisions are not on the table for this round (the substrate is settled per the architectural decision; this is a positive enumeration, not a re-litigation). If a reading would prompt revision of an off-table decision, that is a flag — visible in the journal, not in code, but visible.

**Layer 2 — journaled reason for any substrate revision triggered by a reading.** If a reading does prompt a substrate revision (and the synthesis acknowledges this is sometimes legitimate — the read may surface something the substrate is genuinely missing), the journal entry naming the revision must:
1. Cite the reading and its `paired_reading_id`.
2. Name the framework lens the reading came through.
3. Cite the project-document commitment the proposed revision interacts with (charter, design notes, architectural decision, settled synthesis).
4. Name the external trigger that justifies the revision (new external learning, not "the system did something").

If any of (1)–(4) is missing, the revision is not journaled cleanly; the plan treats this as a gap and does not implement the revision.

**Layer 3 — external human reader at ≥1 milestone.** The synthesis elevates this to a Probe 2 commitment. The plan does not specify the human ("trusted, not necessarily expert") — the build phase identifies them. The reader sees the digest, the pre-registered criteria, the eyeball-helper scalar summaries, and produces a reading before seeing the mirror's. The Mellers/Hertwig/Kahneman protocol works at N=2; what the reader catches is the builder's blind spots. The reading is journaled alongside the mirror's; disagreements between the human reader and the mirror are tracked, not resolved silently.

Co-design mitigations are partial. The plan does not pretend otherwise. Probe 2 makes the loop visible; it does not close it.

---

## 9. Forward-compatibility commitments

The plan does not implement Probe 3 or Probe 4 features but does not foreclose them.

**To Probe 3 (four-state operational model and dream-state machinery).**
- `StructuredReading.claims` accommodates dream-state citations via `cited_stream="dream_rollout"`; no field changes needed for state-typed readings.
- The hierarchical digest's drill-down accessor accommodates dream-state telemetry via `fetch_dream(dream_index)`. State-typed reading at Probe 3 just means the prompt names the state in the digest framing.
- The four-state transitions (waking / dreaming / dormant / paused) are implemented at Probe 3; Probe 2 only exercises waking. The schemas and digest layout do not assume a single state, but the runner does — the runner change at Probe 3 is to thread a `RunState` field through the agent_step record. The plan flags this as a Probe 3 schema bump (likely 0.3.0); no Probe 2 change preempts it.

**To Probe 4 (distinguishability test).**
- `world_event.payload` already carries the asymmetry: builder-source events have `source="builder"` and a populated mutator-parameter payload; internal stochasticity events have `source="environment"` and aggregate counts. Probe 4 reads this payload to compute distinguishability metrics; Probe 2 leaves the payload schemaless so Probe 4 can extend without bumping the version.
- The `is_sham` convention does not interact with Probe 4's distinguishability metrics — sham events are explicitly excluded from "the perturbations Io's RSSM might learn the signature of" because they don't actually perturb the grid. The `is_sham` field's presence is a calibration artefact, not a signal.
- The mutator vocabulary (the four real mutators per Probe 1 plan §2.4) is unchanged; Probe 4's distinguishability test runs against the same mutators. The plan changes nothing here.

**To later probes (kind-encounter, ending protocol, etc.).** None of Probe 2's structures forecloses adding a second agent class, a kind-encounter event type, an ending-protocol event type. The `world_event.event_type` literal is extensible; the schemas accept new payload shapes.

---

## 10. Out of scope at Probe 2

Explicit list. The plan does not build for Probe 3 or 4 except where forward-compatibility was authorized in §9.

- **No state-transition machinery.** Only waking is exercised. Dreaming/dormant/paused are not implemented (Probe 3).
- **No distinguishability metrics on perturbations.** Probe 4 owns this. Probe 2 emits sham events; it does not compare sham-aligned vs builder-aligned signatures.
- **No Io substrate change.** No reward predictor, no continuation head, no critic network, no value head. Settled at architectural decision; Gemini's research-output question about a critic was a substrate misunderstanding (see synthesis §2 / §3(g)).
- **No real-time mirror calls.** The runner does not call the mirror in-loop. All mirror passes are post-hoc against committed telemetry, on the builder's schedule (or on a weekly cron).
- **No multi-round debate.** Settled (synthesis rejects on Kenton et al. 2024 closed-task null).
- **No sequential reader-critic.** Settled (synthesis rejects on Probe 1 anchoring-bias evidence).
- **No tool-use API exposed to the LLM at the digest layer.** The drill-down accessor is callable from Python; the prompt structure includes pre-fetched windows the builder selects. Dynamic tool-use is deferred.
- **No automatic prompt-paraphrase generation.** Paraphrases are hand-written and frozen; build-phase tunes.
- **No automatic criterion revision.** Mirror update discipline applies; revisions are journaled before code.
- **No alteration of Probe 1's free-text `MirrorReading`.** It stays at version 0.1.0; nothing rewrites old readings.
- **No grid size, action space, or episode length change.** Only the start-cell randomization and sham mechanism touch the env.
- **No new internal stochasticity sources.** Two stochastic processes (regrowth, drift) inherited from environment synthesis.
- **No new mutator vocabulary.** The four mutators (`add_resource`, `remove_object`, `set_cell_state`, `move_object`) are unchanged. `introduce_novel_object` stays excluded (no-marker ground).
- **No CMA implementation as Gemini proposed.** Causal mediation analysis is a real technique; its application as a clean operational protocol for distinguishing prompt-driven from data-driven LLM readings is constructed in Gemini's output, not literature-canonical (synthesis §3(g)). The plan does not implement it; the spirit (testing whether the prompt or the data drives the reading) is captured by the shuffled baselines and the lesion runs.
- **No 0% false-positive bar.** Synthesis rejects this as reified; the plan inherits the 4:1 contrast threshold instead.
- **No content-addressed reading store, no DVC-style artifact tracking.** JSONL append-only, atomic rename for any directory-level commit.
- **No automated prompt regression suite beyond the gate tests.** Build-phase regression is by re-running the calibration smoke when prompts change.

The rule of thumb: if a Probe 3/4 feature can be added later without changing the structured-reading schema, the pre-registration schema, the digest's drill-down interface, or the calibration discipline elements, it is out of scope here.

---

## 11. Connection to the journal

Probe 2 ends with a journal entry (or a series of entries — Probe 2 is heavier than Probe 1 and likely produces multiple) recording what was learned, what surprised, what is now decided. The journal lives at `docs/workingjournal/probe2.md` (new file) and is hand-written by the builder.

**Per-phase entry shape (template).**

- What the phase built. Module names; schemas added or extended; tests added; any backward-compatibility check.
- What surprised. Anything that did not behave as the synthesis or this plan predicted — a shuffled baseline producing a higher `supported` rate than expected, the Judge ruling `supported` on a claim the Skeptic should have refuted, a faithfulness check off by an order of magnitude.
- What is now closed. Specific defaults from §6 promoted from "default" to "settled at this scale."
- What is now newly open. Things the build revealed that the synthesis did not anticipate. These become Probe 3's starting context.

**Pre-registration entries (one per adversarial pass).** The `PreRegistration` record's fields exactly. Filled before the reading runs. The journal embeds the same fields in prose form; the JSONL is the machine-readable record.

**Two-mode builder entries.** Before each reading: which mode (proponent / skeptic) the builder is in, and why. After each reading: which mode was in force when interpretive choices were made, and whether the post-reading interpretation differed from the pre-reading expectation in a way that warrants journaling.

**Watts-default-applied-to-builder entries.** Before each reading: which substrate decisions are off the table this round. After each reading: whether any reading prompted a substrate-revision discussion; if yes, the four-element journaled reason from §8 Layer 2.

**Post-protocol-failure entries.** If the calibration smoke catches a confabulation that the protocol *should* have caught and didn't, or fails to catch one it *should* have, the journal records the gap, what was tried, and what was changed in response. This is the bridge between Probe 2's protocol design and Probe 3's input — the failure modes Probe 2 surfaces become Probe 3's research-prompt context.

**End-of-Probe-2 synthesis entry.** A short (≤500 words) entry naming what Probe 2 produced operationally: the count of admissible structured readings, the credible-novelty case, the calibration-protocol catch of a confabulation, the end-of-Probe-2 milestone reading by the external human reader. This is the bridge to Probe 3's research-prompt drafting.

---

*End of plan.*
