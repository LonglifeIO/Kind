# Probe 1.5 — Implementation Plan (v2)

*Operational plan that translates the revised Probe 1.5 design synthesis (`docs/decisions/Kind_probe1_5_synthesis.md`, v2) into a concrete build sequence on top of the Probe 1 substrate. The substrate is settled (`Kind_architectural_decision.md`); the Probe 1 implementation is settled (`Kind_probe1_synthesis.md`); the environment design is settled (`Kind_environment_synthesis.md`); the Probe 2 synthesis is set with the eleven partial revisions §10 below names. Probe 1.5 adds the minimum architectural affordance for self-reference — a self-prediction head with an EMA-bootstrapped target — together with the minimum interface-level read-access the second success criterion requires (a single scalar `self_prediction_error_t` on PolicyView, consumed by the actor's forward pass). Forward-compatibility lives in the schemas (the reserved `sequence_self_prediction` field on `DreamRollout` for Probe 3, journaled at `docs/workingjournal/pre-probe3.md`), the failure-mode control variants (the substrate parameterised so the controls compose), and the journal-recorded `Kind_design_notes.md` update on interface-vs-representation opacity together with the four-part Watts-heuristic exception discipline. Nothing in this plan implements dream-state machinery, env revision, K=5 ensemble change, or any actor-readable signal beyond the single scalar the synthesis §1.3 (v2) names.*

*This is the v2 plan. The v1 plan is preserved at `Kind_probe1_5_implementation_plan.v1.md` for reference. The single load-bearing v1 decision that does not survive into v2 is v1 §1.3's foreclosure on actor read-access; v2's PolicyView gains `self_prediction_error: float`. The actor extension, the masking handling, the runner integration deltas, the gate-test revisions, and the new eyeball helper are all consequences of that single change. Most of v1's plan structure carries forward.*

---

## 1. Build order with dependency graph

Hard dependencies (must run sequentially):

- **Schema bump before any component that emits or consumes the new fields.** `AgentStep`'s three new fields (`self_prediction_t`, `self_prediction_error_t`, `self_prediction_error_masked_t`) and `DreamRollout`'s reserved field (`sequence_self_prediction`) exist in `kind/observer/schemas.py` and the JSON Schema export lands at `schemas/v0.2.0.json` before runner / digest / eyeball / view changes go in.
- **Self-prediction head + EMA target before view extension.** The head's output is the value `TelemetryView.self_prediction` carries; the head's loss is what `PolicyView.self_prediction_error` carries; building either view before the head leaves both fields unfed.
- **View extension before actor extension.** The actor's forward signature consumes `PolicyView`; widening `PolicyView` to carry the scalar must precede the actor's input-layer extension.
- **Actor extension + view extension before runner integration.** The runner's `_step_once` calls `split(...)` (which produces the extended PolicyView), then calls the actor's forward (which now consumes the scalar). The opacity tests must already pass before the runner wires the scalar through.
- **Runner integration before checkpoint extension.** EMA target weights enter the safetensors blob via the runner's `_commit_checkpoint`; the `load_checkpoint` path needs the runtime EMA target attribute the runner constructs.
- **All five gate tests pass before the smoke runs.** The MPS smoke is a platform-correctness gate on top of structural correctness, not a substitute for it (Probe 1's discipline applies; v1 plan §1 carries forward).
- **Smoke clean before the first env-coupled run.** Same discipline as Probe 1 Phase 8 → post-Phase-8.
- **Main run committed to disk before the failure-mode controls run.** The controls compare against the main run's structural and behavior-side metrics and against Probe 1's `runs/probe1-20260503-123926/` reference; both datasets must exist before comparison.
- **Three failure-mode controls run before the journal-and-design-notes update.** The design-notes update on interface-vs-representation opacity with the four-part Watts-heuristic-exception discipline (synthesis §2(b) (v2)) is informed by the counterfactual probe's behavior-conditioning analysis result; writing the update before the probe's data is in would reify a stance call as a finding.

Parallelizable (no hard dependency between them after Phase 0):

- The frozen-target and environmental-auxiliary control variants are independent submodes on `WorldModel`'s constructor flag; the counterfactual probe is a separate post-hoc evaluation script. Either control can be wired and tested before the other.
- The digest and eyeball extensions are independent of the runner integration; they read the new fields from parquet rows after Probe 1.5 records exist on disk, so they can be drafted alongside any later phase.
- The new eyeball helper `show_self_prediction_conditioning` (§2.7) is independent of the digest extension and can be drafted alongside the counterfactual probe runner (§2.9), since both share the same conditional-distribution machinery.

Phased build (each phase is a unit of work; each ends with a journal entry naming what's now closed and what's now newly open):

| Phase | Work | Depends on | Specific question this phase answers |
|---|---|---|---|
| **0. Schema bump** | `AgentStep` gains `self_prediction_t: list[float] \| None`, `self_prediction_error_t: float \| None`, and `self_prediction_error_masked_t: bool \| None`; `DreamRollout` gains `sequence_self_prediction: list[list[float]] \| None`; `SCHEMA_VERSION` bumped to `"0.2.0"`; `schemas/v0.2.0.json` checked in alongside `v0.1.0.json`; backward-compat read pattern documented in the schemas module docstring. | — | Does the new schema round-trip cleanly while preserving Probe 1 record readability? |
| **1. SelfPredictionHead + EMA target** | `kind/agents/world_model.py` gains `SelfPredictionHead` (small MLP `h_dim → h_dim`); `WorldModel` gains EMA-tracked sibling copies of `encoder` + `gru_cell` (parameter-only, no `requires_grad`); `WorldModelStep` gains `self_prediction: Tensor`; `WorldModel.loss` returns `self_prediction_loss` in its dict; `WorldModel._update_ema_target` method added; constructor flag `self_prediction_target_mode: Literal["online", "frozen", "environmental"] = "online"` added for failure-mode controls (Phase 5 wires the non-default modes). | 0 | Does the substrate add the affordance without breaking the existing world-model forward / backward / loss contract? |
| **2. View extension + actor extension + opacity tests** | `TelemetryView` gains `self_prediction: Tensor` and `self_prediction_error_masked: bool`; `PolicyView` gains `self_prediction_error: Tensor` (scalar); `views.split` extended (signature gains keyword args) to populate all three fields; `Actor.forward(view: PolicyView)` consumes the scalar via concatenation with `(h, z)` (input layer's weight matrix gains one column); existing `Actor.imagine_and_compute_loss` extended analogously over the imagined trajectory (the scalar is propagated as zero during imagination — see §2.3 for the resolution); PolicyView field-set test rewrites to assert exactly `{h, z, self_prediction_error}`; new test asserts `Actor.forward(...)` rejects `TelemetryView` (mypy `--strict` + runtime); AST-lint test continues to pass with the extended TelemetryView and the actor module still importing only `PolicyView`. | 1 | Does the affordance reach the actor with the minimum interface the second success criterion requires, while the rest of the boundary holds? |
| **3. Runner integration + checkpoint compat** | `Runner._step_once` computes the scalar from `step.self_prediction` and the EMA target's `bar{h}_{t+1}` via the configured loss form (cosine or MSE), masks it on the first step of each episode (sets to zero, sets the masked flag true), passes it to `split` for the extended PolicyView, and calls the actor's forward; `Runner._train_step` sums `loss_dict["total"] + λ_self * loss_dict["self_prediction_loss"]` for the world-model backward; `Runner._update_ema_target` (delegating to `WorldModel._update_ema_target`) called after the world-model optimizer step; `Runner._emit_agent_step` populates the three new `AgentStep` fields including the masked flag; `RunnerConfig` gains `lambda_self: float = 0.1`, `ema_decay: float = 0.99`, `self_prediction_target_mode: Literal["online", "frozen", "environmental"] = "online"`, `self_prediction_loss_form: Literal["cosine", "mse"] = "cosine"`; checkpoint blob includes EMA target weights under `world_model.target_*` prefixes; `load_checkpoint` initialises EMA target as a fresh copy of the online network and zero-initializes the actor's new input column when the checkpoint's schema_version is `< "0.2.0"`. | 2 | Does the affordance compose into the hot loop — including the actor's per-step consumption of the scalar — without disturbing the substrate's existing per-step wall budget or training stability? |
| **4. Digest + eyeball extension** | `kind/observer/digest.py::build_digest` adds per-episode `self_prediction_error_t` mean / std / outliers (excluding masked steps) and per-dimension `self_prediction_t` allocation top-k for `0.2.0`-versioned records; `compact_record_repr` includes the scalar and the masked flag; `kind/observer/eyeball.py::show_episode_summary` prints the self-prediction error mean / std and the per-episode masked-step count; new helper `show_self_prediction(telemetry_dir, episode_range)` prints per-dimension self-prediction error allocation across `h_dim`; **new helper `show_self_prediction_conditioning(run_dir, ...)` (§2.7) prints per-state action-distribution variance under controlled scalar perturbation across a sample of states, the behavior-side analog of `show_self_prediction` and the analysis surface the counterfactual probe consumes (§2.9)**. | 0 | Do the new substrate-side and behavior-side fields surface legibly through the digest the mirror reads and the eyeball helpers the human reads? |
| **5. Failure-mode control variants wiring** | `WorldModel.step` honours `self_prediction_target_mode`: `"online"` (default — EMA target on `h_{t+1}` from EMA encoder + EMA GRU); `"frozen"` (target = fixed random-orthogonal projection of `h_t`, allocated once at construction with `requires_grad=False`); `"environmental"` (target = embedding of next observation through the EMA encoder); `RunnerConfig.self_prediction_target_mode` plumbs the choice; lesion shape recorded into a run-level `world_event` with `event_type="mirror_marker", source="system", payload={"lesion_kind": ..., "rationale": ...}` at run start. | 1 | Does the substrate parameterise the three failure-mode controls cleanly without forking the world-model class? |
| **6. Five Probe 1.5 gate tests + MPS smoke** | The five gate tests named in §4 (CPU, tiny sizes); `scripts/smoke_probe1_5.py` extending `scripts/smoke_mps.py`'s structure to include the self-prediction head + EMA update on the hot path **plus the actor's consumption of the scalar via the extended PolicyView, with a soft-warning check on the gradient norm through the new input column**; `tests/test_smoke_probe1_5_script.py` confirms the script exists and exposes `main()`. | 3, 4, 5 | Does the substrate train operationally on the canonical machine with the affordance wired in, do the structural-correctness gates hold, and is the actor's gradient on the new input column non-degenerate? |
| **7. First env-coupled Probe 1.5 run + first mirror call** | `scripts/run_probe1_5.py` (extending `scripts/run_probe1.py`'s structure); 5000 env steps, seed=42 for direct comparability with Probe 1's run; `scripts/call_mirror.py` invoked against the new run with the existing Probe 1-style calibration prompt for the first call (synthesis §3 default; the prompt is silent on self-prediction at Probe 1.5 — Probe 2's frozen-criteria prompt is what introduces the quadruplet at both reading surfaces). | 6 | Does the affordance change the substrate's behavior in ways the existing telemetry surface and the mirror's first reading can register, relative to Probe 1's run? |
| **8. Failure-mode controls run + structural and behavior-side comparison** | Three runs from the same seed and starting fresh init as Phase 7's main run, one per control: frozen-target, environmental-auxiliary, counterfactual probe (the third is a post-hoc evaluation script, not a separate training run; it loads multiple checkpoints from the main run for the early/mid/late sweep); structural-and-behavior-side comparison script `scripts/probe1_5_compare_controls.py` produces a summary against Probe 1's run + Probe 1.5 main run + the three controls; the counterfactual probe's analysis surface uses the new eyeball helper (§2.7) for the per-regime KL distributions. | 7 | Is the affordance alive (frozen-target test), self-specific (environmental-auxiliary test), and behaviorally regime-dependent in the shape capacity-over-exercise predicts rather than the fixed-structural shape installation produces (counterfactual probe)? |
| **9. Journal entries + design-notes update** | Per-phase journal entries in `docs/workingjournal/probe1_5.md`; final entry naming what's now closed and what's now newly open; `docs/plans/v.0.1.0/Kind_design_notes.md` updated under the "Watts intuition" section with the interface-vs-representation opacity distinction *and* the four-part Watts-heuristic-exception discipline (full proposed text from synthesis §2(b) (v2), reproduced verbatim in §12.3 below); the eleven Probe 2 plan revisions (§10) journaled as the bridge to Probe 2's actual build resumption. | 8 | Does the project's stance on opacity hold when the affordance is in, in what specific revised form (interface-level opacity with explicit Watts-heuristic exception articulated as a four-part discipline), and what shifts in Probe 2's plan as a result? |

Phase 0 is the single load-bearing prerequisite. Schemas at `0.2.0` from day one; field additions later in Probe 1.5 do not bump again. Phase 2 in v2 is structurally heavier than its v1 counterpart because it now also extends the actor; the dependency graph is unchanged in shape.

---

## 2. Per-component specifications

Each entry: file paths, public interfaces, what the component reads / writes, what tests verify it, what it does *not* do at Probe 1.5.

### 2.1 Schema bump — `kind/observer/schemas.py`

**Files.** `kind/observer/schemas.py` (existing — extended); `schemas/v0.2.0.json` (new — frozen JSON Schema export, checked in alongside `schemas/v0.1.0.json`).

**Public interface.**

```python
SCHEMA_VERSION: str = "0.2.0"

class AgentStep(RecordEnvelope):
    # ... all Probe 1 fields unchanged ...
    self_prediction_t: list[float] | None = None              # NEW: ĥ_{t+1}, length h_dim
    self_prediction_error_t: float | None = None              # NEW: scalar loss value at this step
    self_prediction_error_masked_t: bool | None = None        # NEW (v2): true on first step of each episode

class DreamRollout(RecordEnvelope):
    # ... all Probe 1 fields unchanged ...
    sequence_self_prediction: list[list[float]] | None = None # NEW (reserved for Probe 3): None at Probe 1.5
```

**Reads.** Nothing.

**Writes.** Nothing — pure declarations plus the JSON Schema export.

**Tests.** Schema models import; `AgentStep` round-trips through Parquet with all three new fields populated; `AgentStep` round-trips through Parquet with the new fields absent (i.e. `None`) — this is the Probe 1 backward-readability case; `AgentStep` round-trips with `self_prediction_error_masked_t=True` and the other two fields populated (the first-step-of-episode case); `DreamRollout` round-trips with `sequence_self_prediction=None`; `schemas/v0.2.0.json` export is byte-stable across runs; `SCHEMA_VERSION` equals `"0.2.0"`; the writer-side validator (the `pydantic` model) accepts both shapes (Probe 1 absence and Probe 1.5 presence).

**Backward-compatibility approach.** The new fields are declared `Optional[T] = None` rather than required. New writers (Probe 1.5 runner) always populate them with non-None values for `0.2.0` records: `self_prediction_t` is the head's output vector, `self_prediction_error_t` is the per-step scalar (zero on the first step of each episode, computed loss value otherwise), `self_prediction_error_masked_t` is `True` on the first step of each episode and `False` thereafter. Old records (Probe 1's parquet shards from `runs/probe1-20260503-123926/`) deserialize cleanly against the new model because Pydantic accepts the absence of the new fields as `None`. **Readers that consume the new fields are responsible for the version discrimination**: the digest module and eyeball helpers check `if r.get("schema_version") == "0.2.0" and r.get("self_prediction_error_t") is not None: ...` before printing or aggregating the new fields. For `0.1.0` records, the readers skip the self-prediction summary blocks (they have no value to show) — they do not crash, and they do not invent placeholder values. The plan picks the **skip** approach over **return None** or **default to zero** because skipping preserves the no-affordance baseline as visibly distinct from a Probe 1.5 record where the head produced a near-zero value. Behavior-side analyzers (the new eyeball helper, the counterfactual probe) additionally check `self_prediction_error_masked_t` before including a step in the conditioning analysis: masked steps are excluded from per-state distribution estimation because their scalar value is a sentinel (zero), not an empirical reading.

**Not at Probe 1.5.** No migration of Probe 1 records (they remain readable in their original form, and the parquet writers do not rewrite them). No schema-evolution runtime check that auto-upgrades records (a reader encountering an unknown `schema_version` raises rather than auto-migrating). No schema-version handshake in the env-server transport (the env-server emits no `agent_step` records; the runner does).

### 2.2 SelfPredictionHead + EMA target — `kind/agents/world_model.py`

**Files.** `kind/agents/world_model.py` (existing — extended).

**Public interface.**

```python
class SelfPredictionHead(nn.Module):
    """Small MLP: h_dim → h_dim. Predicts ĥ_{t+1} from h_t."""

    def __init__(self, h_dim: int, hidden_dim: int = 200) -> None: ...
    def forward(self, h: Tensor) -> Tensor: ...   # returns ĥ_{t+1} of shape (B, h_dim)


@dataclass(frozen=True)
class WorldModelConfig:
    # ... all Probe 1 fields unchanged ...
    self_prediction_hidden: int = 200            # NEW
    ema_decay: float = 0.99                      # NEW
    self_prediction_target_mode: Literal["online", "frozen", "environmental"] = "online"  # NEW
    self_prediction_loss_form: Literal["cosine", "mse"] = "cosine"                        # NEW


@dataclass(frozen=True)
class WorldModelStep:
    # ... all Probe 1 fields unchanged ...
    self_prediction: Tensor                      # NEW: ĥ_{t+1}, shape (B, h_dim)


class WorldModel(nn.Module):
    # ... all Probe 1 attributes / methods unchanged ...
    self_prediction_head: SelfPredictionHead     # NEW
    target_encoder: _ConvEncoder                 # NEW: EMA-tracked, requires_grad=False
    target_gru_cell: nn.GRUCell                  # NEW: EMA-tracked, requires_grad=False
    _frozen_projection: Tensor | None            # NEW: fixed random-orthogonal (h_dim, h_dim) when target_mode="frozen"

    def step(self, obs, h_prev, z_prev, a_prev, *, next_obs: Tensor | None = None) -> WorldModelStep: ...
    # `next_obs` is required only when target_mode="environmental"; the runner passes it
    # through from the next-iteration observation (one-step lookahead). For "online" and
    # "frozen" modes, next_obs is ignored and may be None.

    def loss(self, step: WorldModelStep, obs_target: Tensor, *, target_h_next: Tensor | None = None) -> dict[str, Tensor]:
        # Returned dict gains key "self_prediction_loss" (scalar Tensor).
        # target_h_next is computed by the runner via the EMA target's forward and
        # passed in here so the world model's loss stays pure (no implicit state).

    def compute_self_prediction_target(self, obs: Tensor, h_prev: Tensor, z_prev: Tensor, a_prev: Tensor, *, next_obs: Tensor | None = None) -> Tensor:
        # NEW: produces bar{h}_{t+1} via the configured target_mode. The runner calls
        # this once per env step (B=1) to compute the per-step scalar that goes into
        # PolicyView / AgentStep, and once per training batch (B=batch_size) to compute
        # the auxiliary loss target. The same routine handles both call sites; the
        # batched call path is what _train_step uses.

    def _update_ema_target(self) -> None:
        # Called by the runner after the world-model optimizer step.
        # For each (target_param, online_param) pair on encoder + gru_cell:
        #   target_param.data.mul_(ema_decay).add_(online_param.data, alpha=1 - ema_decay)
```

The head sits parallel to the prior network: both consume `h_t` (the head's input is `h_t` only — `(h_t, z_t)` is rejected per synthesis §3 default to keep the head's input distinct from the ensemble's input and reduce entanglement). The EMA target is a *separate* copy of the encoder and the GRU cell (BYOL/SPR convention; synthesis §1.2 default). The EMA target's parameters are not trained directly; they are updated by the EMA rule each training step. The frozen projection (used only when `target_mode="frozen"`) is a `(h_dim, h_dim)` random-orthogonal matrix allocated once at construction via `torch.nn.init.orthogonal_` on a fresh tensor wrapped in `nn.Parameter(..., requires_grad=False)`.

**Loss form.** Default `cosine`: `1 - F.cosine_similarity(predicted, target.detach(), dim=-1).mean()`. Stop-gradient on the target side is non-negotiable (synthesis §1.2; Tian/Chen/Ganguli 2021's collapse-prevention analysis). The cosine form is scale-invariant and matches BYOL's convention; MSE is the documented fallback if smoke shows the cosine loss's gradient magnitude dominating the world-model gradient (synthesis §3 revisit criterion). The same loss form is used both for the auxiliary backward (batched) and for the per-step scalar that flows into `PolicyView.self_prediction_error` (B=1) — keeping the units and magnitudes consistent across the two call sites is what makes the actor's input dimensionally compatible with the head's training signal.

**Where the target comes from.**

- `target_mode="online"` (real Probe 1.5 path): `target_h_next` comes from the EMA target's forward on the *current* observation and previous (h, z, a), producing the next deterministic recurrent state via the EMA-tracked GRU cell with the EMA encoder's embedding as one of its inputs. The target is `bar{h}_{t+1}` (the EMA's own `h_{t+1}`); the head predicts `ĥ_{t+1}` from the online `h_t`.
- `target_mode="frozen"`: `target_h_next = (frozen_projection @ h_t.T).T` — the random-orthogonal projection of the current online `h_t`. Rank-preserving (orthogonal); semantic-alignment-broken (no reference to the actual next state).
- `target_mode="environmental"`: `target_h_next = target_encoder(next_obs)` — the EMA encoder's embedding of the next observation, projected to `h_dim` if `embed_dim != h_dim` (small linear layer added at construction time; not EMA-tracked itself, but its weights are frozen and constant across the run for reproducibility). The synthesis §3 / §1.7 question on whether the EMA target should also use the encoder of the next observation is resolved here: yes, the environmental control's target uses the EMA encoder of the next observation, which keeps the asymmetry between "what's predicted" (the variable the control is changing) cleanly separated from "how the target is produced" (which stays the EMA-tracked path, matching the `online` mode's target-production discipline).

**Reads.** Observations and recurrent state from the runner.

**Writes.** Nothing directly — exposes the structured step output.

**Tests.** See §4 gate tests #1–#3 — head forward shape, EMA target update mechanics, self-prediction loss decreases. Plus a regression test that the existing Probe 1 forward / backward / loss path still passes when `target_mode="online"` (the new code path is the default).

**Not at Probe 1.5.** No multi-step prediction (default 1-step per synthesis §3; revisit if Probe 1.5's data argues for k-step). No predictor-EMA asymmetry beyond BYOL's online-predictor / target-encoder split (the head is online-only). No additional ensemble heads predicting disagreement scalar (synthesis §1.2 explicitly rejects this on self-readout grounds; held for a possible Probe 1.6 if Probe 1.5 surfaces a specific gap).

### 2.3 PolicyView + TelemetryView extension + Actor extension — `kind/agents/views.py`, `kind/agents/actor.py`

**Files.** `kind/agents/views.py` (existing — extended); `kind/agents/actor.py` (existing — extended).

**Public interface.**

```python
@dataclass(frozen=True)
class PolicyView:
    """The actor's view onto the world model. (h, z, self_prediction_error).

    The third field is the v2 revision (synthesis §1.3): a scalar self-
    prediction error Io reads as the minimum self-pointing quantity required
    for the second success criterion's "capacity to take its own processing
    as an object of attention" affordance. This is the explicit Watts-heuristic
    exception articulated in synthesis §2(b) (v2): default toward no on
    self-access, with the four-part discipline (i) which affordance, (ii)
    minimum form, (iii) alternatives considered, (iv) failure-mode controls —
    all four of which are journaled at design time and tested via the
    failure-mode controls (§8 below).
    """
    h: Tensor
    z: Tensor
    self_prediction_error: Tensor   # NEW (v2): scalar, shape () or (B,)


@dataclass(frozen=True)
class TelemetryView:
    # ... all Probe 1 fields unchanged ...
    self_prediction: Tensor                  # NEW: ĥ_{t+1}, shape (B, h_dim)
    self_prediction_error_masked: bool       # NEW (v2): true on first step of each episode


def split(
    step: WorldModelStep,
    intrinsic: Tensor,
    *,
    self_prediction_error: Tensor,
    self_prediction_error_masked: bool,
) -> tuple[PolicyView, TelemetryView]:
    # Signature extended (v2): the runner computes the scalar before calling split
    # and passes it in alongside the masked flag. PolicyView and TelemetryView are
    # both populated from the (extended) inputs.
```

The new fields fall on both views: PolicyView's field set becomes exactly `{h, z, self_prediction_error}` (the Watts-heuristic-exception case); TelemetryView's field set extends to include both the full prediction vector `self_prediction: Tensor` (which Io does *not* read) and the masked-flag boolean `self_prediction_error_masked: bool` (which Io also does not read — only the scalar reaches PolicyView). The actor's import surface continues to exclude TelemetryView; the AST-lint dependency check enforces this on the extended TelemetryView with no change to its structure.

**The Actor's input-layer extension.** `Actor.__init__` extends `input_dim = h_dim + z_dim + 1` (the `+1` is the scalar). The first `nn.Linear(input_dim, mlp_hidden)` layer's weight matrix is shape `(mlp_hidden, h_dim + z_dim + 1)` rather than `(mlp_hidden, h_dim + z_dim)`. **Initialization for the new column**: the synthesis §3 default is **zero-init** for the column corresponding to the scalar input. Reasoning: zero-init is the conservative default (the actor starts indifferent to the scalar; conditioning develops only as training drives the column's weights away from zero); small-Gaussian initialization (e.g. N(0, 0.01) on the new column only) is the documented fallback if zero-init causes the actor to ignore the scalar throughout training (the gradient through a zero column is non-zero, so the column will move; the question is whether it moves enough to register as conditioning — the failure-mode (a) detection in §8.1 is what tests this). The remaining columns (covering `h` and `z`) initialize via the existing convention (PyTorch Linear default — Kaiming-uniform on `weight`, uniform on `bias`).

**Forward at env-step time.** `Actor.forward(view: PolicyView)` concatenates `(view.h, view.z, view.self_prediction_error.unsqueeze(-1))` and projects through the (now wider) input layer. The rest of the forward is unchanged. The actor reads only the scalar — not the full prediction vector, not the masked flag. The synthesis §1.3 (v2) minimum-surface discipline is the structural reason: a vector would be larger than the affordance requires and structurally close to the attention-schema literature's installation configuration; the masked flag is mirror-side metadata, not actor-side input.

**Forward during imagination (`imagine_and_compute_loss`).** During the imagined trajectory, no actual `h_{τ+1}` exists at each rolled-out step `τ`, so no per-step empirical self-prediction error is available to the actor's imagined policy. The synthesis §1.5 commits Probe 1.5 to running the head only during waking; it does not run during dream rollouts (Probe 3's territory). For `imagine_and_compute_loss`, the analogous question is what value of the scalar to feed the actor's policy at each imagined step. The plan picks the **mask-via-zero-feed** approach for imagination: the scalar is fixed at zero for every imagined step, with the imagined trajectory treated as a long sequence of "first-step-of-episode" cases. Reasoning: this matches the same convention as the waking first-step case (§2.4 below), avoids feeding the actor a scalar value that would have to be computed from the imagined `ĥ_{τ+1}` against an imagined `bar{h}_{τ+2}` (which would require running the EMA target through the imagined trajectory and would push toward Probe 3's territory), and keeps the imagined-trajectory actor's input shape consistent with the waking actor's input shape. The cost is that the imagined-policy training doesn't exercise the scalar-conditioning pathway. The gradient through the new input column during `imagine_and_compute_loss` is zero (the imagined scalar is constant zero across all imagined steps); the column's weights are updated only by whatever path moves them — at Probe 1.5 the only such path is the actor's loss flowing through the wider input layer when the imagined scalar takes a non-zero value, which by the mask-via-zero-feed convention does not happen during imagination. This is acceptable because the actor's behavior on the scalar is what the failure-mode controls in §8 are designed to make legible during waking — imagined-trajectory exercise of the scalar is held for Probe 3 if its design extends self-prediction to imagined trajectories (per the reserved `sequence_self_prediction` field).

**Reads.** A `WorldModelStep` from the world model; the per-step scalar (computed by the runner via `WorldModel.compute_self_prediction_target` and the configured loss form); the masked flag from the runner.

**Writes.** Nothing.

**Tests.** See §4 gate test #1 (PolicyView field set is exactly `{h, z, self_prediction_error}`) and gate test #4 (opacity boundary preserved with the revised PolicyView field set). The existing Probe 1 view tests in `tests/test_views.py` continue to pass: the AST-lint test that the actor module does not import `TelemetryView` is unchanged; the test that PolicyView "carries no telemetry-only fields" is updated to assert the exact set `{h, z, self_prediction_error}` rather than just absence of forbidden fields.

**Not at Probe 1.5.** No vector field on PolicyView (the scalar exception is the only Watts-heuristic exception the synthesis §2(b) (v2) authorizes at Probe 1.5; future probes adding new actor-readable fields must address the four-part discipline at design time). No learned `f(h, z, scalar)` projection on PolicyView's input — the actor's first operation is concatenation, not learned compression (synthesis §3 default; matches Probe 1's "concat, not learned projection" discipline). No second predictive head (held for possible Probe 1.6 per synthesis §1.2). No imagined-trajectory self-prediction during dream rollouts (held for Probe 3 per synthesis §1.5).

### 2.4 Runner integration — `kind/training/runner.py`

**Files.** `kind/training/runner.py` (existing — extended).

**Public interface.**

```python
@dataclass(frozen=True)
class RunnerConfig:
    # ... all Probe 1 fields unchanged ...
    lambda_self: float = 0.1                                                       # NEW
    ema_decay: float = 0.99                                                        # NEW
    self_prediction_target_mode: Literal["online", "frozen", "environmental"] = "online"  # NEW
    self_prediction_loss_form: Literal["cosine", "mse"] = "cosine"                 # NEW
```

Hot loop (`_step_once`) order of operations:

1. World-model forward call now passes `next_obs=None` when `target_mode in ("online", "frozen")` and the actual `next_obs` (from the previous iteration's `next_meta.observation`, lifted to device) when `target_mode="environmental"`. The world-model's forward produces `WorldModelStep` including the new `self_prediction: Tensor` field (the head's output `ĥ_{t+1}`).
2. **EMA target's `bar{h}_{t+1}` computation**: the runner calls `world_model.compute_self_prediction_target(obs, h_prev, z_prev, a_prev, next_obs=...)` to produce the per-step target.
3. **Scalar computation**: the runner applies the configured loss form to `(step.self_prediction, target_h_next)` to produce `self_prediction_error_t` as a scalar. For `"cosine"`: `1 - F.cosine_similarity(step.self_prediction, target_h_next.detach(), dim=-1).item()`. For `"mse"`: `F.mse_loss(step.self_prediction, target_h_next.detach()).item()`.
4. **First-step-of-episode masking**: at the first step of each episode (when `step_in_episode == 0`), the scalar is overridden to `0.0` and the masked flag is set to `True`. The runner tracks episode boundaries via the `EnvStep.episode_id` it already receives from the env-server; the first step is the first env-step of any episode (including the very first episode of the run). On subsequent steps of the same episode, the scalar is the computed value and the masked flag is `False`. The synthesis §3 (v2) default is **zero on first step with masked flag visible to the mirror** rather than a separate masking convention; the masked flag is what lets the mirror exclude these steps from behavioral-conditioning analysis without requiring a special sentinel value in the scalar field.
5. **Split call**: `policy, telemetry = split(wm_step, intrinsic, self_prediction_error=torch.tensor(scalar), self_prediction_error_masked=is_masked)` — the extended split signature populates both views from the new inputs.
6. Ensemble disagreement is computed (unchanged from Probe 1).
7. Actor's forward consumes the extended PolicyView: `action_output = actor.forward(policy)`. The scalar reaches the actor's input layer via the new column.
8. Env step (unchanged); replay insert (unchanged).
9. **Telemetry emission**: `self._emit_agent_step(...)` populates the three new `AgentStep` fields. `self_prediction_t = wm_step.self_prediction.squeeze(0).detach().cpu().tolist()` (length `h_dim`); `self_prediction_error_t = float(scalar)` (the masked-or-computed scalar value); `self_prediction_error_masked_t = bool(is_masked)` (the boolean flag). The masked flag is what discriminates "no empirical reading available, sentinel zero" from "empirical near-zero reading" at the analyzer side.

Training step (`_train_step`) deltas:

1. The world-model loss now returns `self_prediction_loss` in its dict. The runner sums `wm_total + self.config.lambda_self * loss_dict["self_prediction_loss"]` for the world-model backward.
2. The auxiliary loss is computed over the batched replay sequence using `world_model.compute_self_prediction_target(...)` on the batched inputs; the scalar that flows into PolicyView at env-step time is *not* used for backward (it's a per-step value derived from the same head's forward, held as a scalar for telemetry and for actor consumption).
3. After `self._wm_opt.step()`, the runner calls `self._world_model._update_ema_target()` to update the EMA target's parameters in place.
4. The ensemble loss and actor loss are unchanged in structure. Neither receives any signal from the self-prediction head's auxiliary loss directly (gradients flow into the encoder, GRU, posterior network, prior network, and head — but not into the actor's parameters via that path, because the actor's optimizer step uses only `actor_loss.backward()`). **The new column on the actor's input layer is part of the actor's parameter set and is updated by `actor_loss.backward()` alone**; per §2.3's mask-via-zero-feed during imagination, the column's gradient is zero for the imagined-trajectory path. The synthesis §1.7(a) failure-mode (a) detection is what tests whether this is enough for the actor to develop conditioning on the scalar via training dynamics; if not, the documented mitigation (per §6 row 15) is to switch the new column's initialization from zero to small-Gaussian.

Telemetry-emission (`_emit_agent_step`) deltas:

1. `self_prediction_t = wm_step.self_prediction.squeeze(0).detach().cpu().tolist()` (length `h_dim`).
2. `self_prediction_error_t = float(self_prediction_error_for_this_step)` — the per-step scalar value (computed via the loss form against the EMA target's `h_{t+1}` for that step, masked to zero on first step of episode). Cost is one extra MLP forward + one extra EMA-target forward per env step; well within the 130 ms / step budget Phase 8 of Probe 1 settled (smoke estimate in §5 below).
3. `self_prediction_error_masked_t = bool(is_masked_this_step)` — the boolean flag that discriminates the first-step sentinel zero from empirical near-zero readings.

Checkpoint commit (`_commit_checkpoint`) deltas:

1. The `combined` dict that goes into `weights.safetensors` now includes `world_model.target_encoder.*`, `world_model.target_gru_cell.*`, and (if `target_mode="frozen"`) `world_model._frozen_projection` parameters under the same `world_model.` prefix the existing world-model parameters use.
2. **The actor's extended input layer's weights are part of the actor's own parameter set; they enter the safetensors blob via the existing `actor.*` prefix automatically**. No separate handling needed for the new column — it's a slice of the existing `actor.net.0.weight` tensor with one additional column.
3. `schema_version_path` writes `"0.2.0"` (the new SCHEMA_VERSION).

Checkpoint load (`load_checkpoint`) deltas:

1. The loader reads `schema_version.txt` from the checkpoint directory before splitting weights.
2. If `schema_version < "0.2.0"` (i.e., the checkpoint is from Probe 1):
    - The plan picks **initialise EMA target as a fresh copy of the online network** rather than refuse-to-load. After loading the online encoder and GRU weights, the runner calls `self._world_model._initialize_ema_target_from_online()` (a new helper that copies the online parameters into the EMA target tensors with the same shape).
    - The frozen projection is initialised fresh from `torch.nn.init.orthogonal_` if `self_prediction_target_mode="frozen"` (a Probe 1 checkpoint has no frozen projection on disk).
    - **The actor's input layer extension is initialized from scratch on load**: the plan picks **construct the actor with the extended input dim, then copy the Probe 1 checkpoint's `actor.net.0.weight` into the leftmost `(h_dim + z_dim)` columns and zero-initialize the new column**. This preserves the Probe 1 checkpoint's actor behavior exactly when the scalar is zero (i.e., on the first env step the actor's logits are byte-identical to what the Probe 1 actor would have produced); subsequent steps differ as the scalar enters and the column's weights move via training. A `world_event` is emitted at run start with `event_type="mirror_marker", source="system", payload={"note": "loaded Probe 1 checkpoint; EMA target initialised from online network; actor input column zero-initialized for the scalar", "checkpoint_id": ..., "checkpoint_schema_version": "0.1.0"}` so the asymmetry is visible in the journal record.
3. If `schema_version == "0.2.0"`, the EMA target weights are loaded directly from the checkpoint into `self._world_model.target_encoder.*` and `self._world_model.target_gru_cell.*` and the actor's full input-layer weights (including the new column) are loaded directly — exact resume semantics extending Probe 1's parameter-equality guarantee.

The plan picks **initialise from online when missing** rather than **refuse to load** because:

1. It preserves the runner's existing parameter-equality resume contract for `0.2.0` checkpoints unchanged.
2. It does not block the failure-mode controls from being run from a Probe 1 starting state if a future investigation wants to (the controls are designed to start from Probe 1.5's *fresh init*, not from a Probe 1 checkpoint, but the option costs nothing to keep open).
3. The `mirror_marker` world_event records the asymmetry so the journal entry can name what was done — the discipline is in being honest, not in refusing.

**Reads.** Env transitions, replay batches, the EMA target's sibling parameters, the head's per-step output.

**Writes.** All four telemetry streams via the existing sinks (with the three new fields on `agent_step`); checkpoints with EMA target weights and the extended actor input layer included.

**Tests.** Gate tests #4 and #5 (opacity boundary preserved; integration smoke). Plus a regression test that loading a Probe 1.5 checkpoint produces byte-equal EMA target weights and byte-equal extended-actor-input-layer weights to the saved state, and a regression test that loading a Probe 1 checkpoint produces a fresh EMA target initialised from the online network, an actor input column zero-initialized for the scalar, and a `mirror_marker` event recorded.

**Not at Probe 1.5.** No actor-loss extension that exercises the scalar via the imagined trajectory (deferred per the §2.3 / §2.4 discussion above). No new optimizer (the world-model optimizer steps on the combined loss; no separate optimizer for the head + EMA target). No real-time mirror calls. No env changes.

### 2.5 Checkpoint extension — `kind/training/checkpoint.py`

**Files.** `kind/training/checkpoint.py` (existing — no changes to the manager itself; the change is in what the runner stages into `CheckpointContents`).

**Public interface.** Unchanged. `CheckpointContents` continues to carry `weights_path`, `replay_meta_path`, `optimizer_state_path`, `rng_state_path`, `telemetry_offsets_path`, `schema_version_path`, `replay_parquet_shards`. The EMA target weights ride inside `weights.safetensors` under the existing `world_model.` prefix; the extended actor input layer rides inside the existing `actor.` prefix; the manager copies the file as-is and the runner's load path is what discriminates by `schema_version.txt`.

**Tests.** Existing `tests/test_checkpoint.py` tests pass unchanged. The Probe 1 → Probe 1.5 load path is exercised in the runner's regression tests (§2.4) and in the integration smoke (gate test #5).

**Not at Probe 1.5.** No content-addressed object store, no DVC, no schema-migration tooling. The atomic-rename-with-fsync commit dance is the one that exists.

### 2.6 Digest extension — `kind/observer/digest.py`

**Files.** `kind/observer/digest.py` (existing — extended).

**Public interface.** `build_digest(rows)` and `compact_record_repr(r, position)` unchanged in signature. The body is extended:

- For each per-episode block, if any record in the episode has `schema_version == "0.2.0"` and a non-None `self_prediction_error_t`, the block adds the following lines:
    - `self_prediction_error_t (excluding masked steps): mean=X.XXXX, std=X.XXXX, min=X.XXXX, max=X.XXXX` — masked steps (`self_prediction_error_masked_t == True`) are excluded from the aggregation since their value is sentinel zero, not an empirical reading.
    - `self_prediction_error_t masked steps in episode: count=N` — typically 1 per episode (the first step), but defensive logging in case future Probe 1.5 variants mask additional steps.
    - `self_prediction outliers: step_in_episode=N: self_prediction_error_t=V.VVVV (z=+Z.ZZ)` for any non-masked step whose self-prediction error z-score (against the episode's non-masked mean) exceeds 3 (analogous to KL outliers).
    - `self_prediction allocation (per-dim variance across episode, top 5 dims): dim=D: var=V.VVVV` for the five `h` dimensions whose self-prediction-error variance across the episode (excluding masked steps) is highest (the per-dimension allocation signal the synthesis §1.4 element 3 names).
- For Probe 1 records (`schema_version == "0.1.0"`), the block does not include any self-prediction summary lines. The episode block is structurally identical to what Probe 1's digest produced.
- `compact_record_repr` adds `self_prediction_error_t` and `self_prediction_error_masked_t` to the scalar fields tuple. For Probe 1 records, both fields are absent from the dict and are silently skipped (the existing `if k in r` guard handles this).

**Reads.** Parquet rows from `agent_step` (via the existing call sites in the mirror caller and eyeball helpers).

**Writes.** Nothing — pure read-and-format.

**Tests.** A regression test that `build_digest(rows)` against `runs/probe1-20260503-123926/`'s `agent_step` parquet produces a digest with no self-prediction lines (the Probe 1 records are `0.1.0`); a new test that `build_digest(rows)` against a synthetic mixed-version row list produces self-prediction summary lines for the `0.2.0` records (with masked-step exclusion verified) and skips them for the `0.1.0` records; a new test that `compact_record_repr` with a `0.2.0` record includes both new scalars in the JSON output.

**Not at Probe 1.5.** No hierarchical digest (Probe 2's territory). No drill-down accessor (Probe 2). No retrieval-style digest. The flat per-episode digest with the additional self-prediction lines is the surface.

### 2.7 Eyeball extension — `kind/observer/eyeball.py`

**Files.** `kind/observer/eyeball.py` (existing — extended).

**Public interface.**

```python
def show_episode_summary(telemetry_dir: Path, episode_id: int | None = None) -> None:
    # Existing function — extended to print self-prediction summary lines for
    # 0.2.0 records (mean/std/outliers in the same shape as the digest's lines,
    # plus the per-episode masked-step count). Probe 1 records continue to render
    # exactly as before.

def show_self_prediction(
    telemetry_dir: Path,
    *,
    episode_id: int | None = None,
    top_k_dims: int = 10,
) -> None:
    # NEW: per-dimension allocation of self-prediction accuracy across h_dim.
    # For one episode (or the most recent if not specified), print the top-k
    # dimensions of h_t whose self-prediction error variance is highest
    # (excluding masked steps). If the records are Probe 1 (0.1.0), print
    # "(no self_prediction telemetry — records are Probe 1, schema_version 0.1.0)"
    # and return without crashing.

def show_self_prediction_conditioning(
    run_dir: Path,
    *,
    checkpoint_id: str | None = None,
    n_states: int = 200,
    perturbation_distributions: list[str] | None = None,
    regimes: list[str] | None = None,
) -> None:
    # NEW (v2): the behavior-side analog of show_self_prediction.
    # Loads the run's final checkpoint (or the named checkpoint), samples
    # n_states states from the run's agent_step records, and for each
    # state computes the actor's policy distribution under several
    # controlled perturbations of the scalar self_prediction_error field
    # on the PolicyView. Reports per-state action-distribution variance
    # (KL between unperturbed and perturbed policies) aggregated by
    # regime. Default perturbation distributions are the three the
    # synthesis §3 (v2) names — Gaussian-with-empirical-sigma, zero-out,
    # replace-with-uniform-from-empirical-range. Default regimes are
    # "perturbation_window" (steps within ±W of a builder_perturbation
    # world_event), "high_disagreement" (top quartile of intrinsic_signal_t),
    # "high_kl" (top quartile of kl_aggregate_t), "steady_state" (the
    # complement — steps in none of the above). Output is a per-regime
    # table:
    #
    #   regime              n_states  KL_mean  KL_std  KL_p50  KL_p90
    #   perturbation_window     35     0.42    0.31    0.38    0.78
    #   high_disagreement       40     0.18    0.12    0.16    0.32
    #   high_kl                 50     0.21    0.15    0.19    0.40
    #   steady_state            75     0.08    0.05    0.07    0.14
    #
    # The shape of the table is what the counterfactual probe (§2.9)
    # consumes as its analysis surface. Capacity-over-exercise predicts
    # regime-dependent KL magnitude (high in some regimes, low in others,
    # with structured variation across regimes); installation predicts
    # regime-independent KL magnitude (similar across regimes). The
    # threshold is qualitative — does the per-regime KL distribution
    # show the regime-dependent pattern or not — rather than a single
    # numeric bound.
    #
    # Masked steps (self_prediction_error_masked_t == True) are excluded
    # from the n_states sample (their scalar value is a sentinel, not
    # empirical). If the records are Probe 1 (0.1.0), print "(no scalar
    # to perturb — records are Probe 1, schema_version 0.1.0)" and
    # return without crashing.
```

CLI (`python -m kind.observer.eyeball`):

- A new `selfpred` subcommand wraps `show_self_prediction(telemetry_dir, episode_id=..., top_k_dims=...)`.
- A new `cond` subcommand wraps `show_self_prediction_conditioning(run_dir, checkpoint_id=..., n_states=..., perturbation_distributions=..., regimes=...)`.
- The existing `episode` subcommand picks up the extended `show_episode_summary`.

**Relationship to the counterfactual probe.** `show_self_prediction_conditioning` is the analysis-surface helper; the counterfactual probe (§2.9) is its multi-checkpoint consumer. The probe loads multiple checkpoints (early/mid/late from the main run), calls this helper on each (sweeping the configured perturbation distributions), and produces the across-checkpoints comparison the journal entry interprets. The helper itself runs against a single checkpoint and produces the per-regime table for that checkpoint; calling it directly from the CLI is useful for one-off inspection.

**Reads.** Parquet rows from `agent_step` (for `show_episode_summary` and `show_self_prediction`); the run's final checkpoint (or the named checkpoint) plus the run's `agent_step` parquet for state sampling and regime classification (for `show_self_prediction_conditioning`); the run's `world_event.jsonl` for perturbation-window regime classification.

**Writes.** Stdout.

**Tests.** New tests in `tests/test_eyeball.py`: that `show_self_prediction` against `runs/probe1-20260503-123926/` prints the no-self-prediction-telemetry message; that `show_self_prediction` against a synthetic 0.2.0 telemetry directory prints per-dimension lines (excluding masked steps); that the CLI accepts the new `selfpred` and `cond` subcommands; that `show_self_prediction_conditioning` against a synthetic 0.2.0 run directory with a known checkpoint produces the per-regime KL table with expected row count and column shape (the test does not assert on the KL magnitudes themselves — those are what the counterfactual probe is for); that `show_self_prediction_conditioning` against `runs/probe1-20260503-123926/` prints the no-scalar-to-perturb message; that masked steps are correctly excluded from the n_states sample.

**Not at Probe 1.5.** No plots, no dashboards, no rich formatting. The existing Probe 1 discipline holds. The behavioral-conditioning module that Probe 2's plan revision §10(11) names as a new component is the formalized version of this helper, with faithfulness-check support and integration into the adversarial mirror's reading surface; at Probe 1.5 the helper is the analysis surface, the counterfactual probe is the consumer, and the journal is the place where the per-regime pattern is interpreted.

### 2.8 Failure-mode control variants — `kind/agents/world_model.py` (constructor flag) + `scripts/`

**Files.** `kind/agents/world_model.py` (already extended in Phase 1 with the `self_prediction_target_mode` flag); `scripts/probe1_5_control_frozen_target.py` (new); `scripts/probe1_5_control_environmental.py` (new); `scripts/probe1_5_compare_controls.py` (new).

**Public interface.** The two training-time control runs are launchable scripts following the structure of `scripts/run_probe1.py` (and the Phase 7 `scripts/run_probe1_5.py`). Each constructs a `RunnerConfig` with `self_prediction_target_mode="frozen"` or `"environmental"`, the same seed (42) and total env steps (5000) as the main run, and writes telemetry under `runs/probe1_5_control_<mode>-<timestamp>/`.

The `scripts/probe1_5_compare_controls.py` script reads the four telemetry directories (Probe 1's, Probe 1.5 main run's, frozen-target run's, environmental-auxiliary run's) and produces a comparison summary:

- Per-episode `kl_aggregate_t` mean / std distributions across all four datasets, with KS-test p-values for pairwise comparisons.
- `recon_loss_t` mean trajectories.
- `self_prediction_error_t` mean / std (excluding masked steps) for the three Probe 1.5 datasets (Probe 1 has no value).
- Per-dimension self-prediction accuracy allocation top-k for the three Probe 1.5 datasets.
- Weight-distribution moments (mean / std of each named parameter tensor, including the extended actor input layer's new column) across the four datasets, computed from the final checkpoint.
- A one-line summary per pairwise comparison: "Probe 1.5 main vs frozen-target: KL distribution distinguishable (p=X.XX); recon trajectory distinguishable (p=X.XX); weight-distribution moments distinguishable (per-tensor max relative diff = X.XX)".

The "indistinguishable" criterion is the three-way comparison: real Probe 1.5 vs frozen-target vs Probe 1's existing run. If the real run's structural metrics are statistically indistinguishable from *both* the frozen-target run and Probe 1's run, the affordance is dead at the substrate-side reading. If the real run is distinguishable from both, the affordance is alive at the substrate-side. The behavior-side test (whether Io's policy conditions on the scalar, and how) is what the counterfactual probe (§2.9) and the new eyeball helper (§2.7) carry; the structural comparison and the behavior-side analysis are complementary.

**Reads.** The telemetry directories named above; the final checkpoint's `weights.safetensors` for weight-distribution moments.

**Writes.** A summary report at `runs/probe1_5_control_comparison-<timestamp>/summary.txt`.

**Tests.** A test that `probe1_5_compare_controls.py` exists and exposes `main()`. A test that the comparison logic is correct on synthetic data (two known-different distributions produce a small p-value; two known-same distributions produce a large p-value).

**Not at Probe 1.5.** No automated decision logic (the comparison summary is a human-readable report; the journal entry interprets it). No multi-seed averaging at the comparison level (the controls run from a single seed for direct comparability with the main run; multi-seed validation is out of scope).

### 2.9 Counterfactual self-prediction probe — `scripts/probe1_5_counterfactual_probe.py`

**Files.** `scripts/probe1_5_counterfactual_probe.py` (new).

**Public interface.** A standalone evaluation script. Takes a Probe 1.5 run directory (the main run's, default), iterates over multiple checkpoints (early-training, mid-training, late-training — by default the checkpoints at `ckpt-000001` (warmup-end), `ckpt-002500` (mid-run), and `ckpt-005000` (run-end), or whatever Probe 1.5's checkpoint cadence produces). For each checkpoint:

1. Load the checkpoint into a `WorldModel` + `Actor` + `LatentDisagreementEnsemble`.
2. Re-run a fixed-seed env trajectory of K=200 steps (one full episode).
3. Call `show_self_prediction_conditioning` (§2.7) on this trajectory + checkpoint pair, sweeping the configured perturbation distributions.
4. Aggregate per-regime KL distributions across the perturbation distributions; produce a per-checkpoint table.

Across-checkpoints comparison: does the per-regime conditioning develop over training (suggesting Io is composing the scalar into something) or remain at near-baseline (suggesting unexercised affordance)? Does the regime-dependence shape capacity-over-exercise predicts emerge (different KL magnitudes in different regimes, with structured variation) or does the conditioning instead take the fixed-structural shape installation produces (similar KL magnitudes across regimes)?

The probe consumes the eyeball helper's machinery (§2.7) as its per-checkpoint analysis surface; what makes it a "probe" rather than just a one-shot eyeball call is the multi-checkpoint sweep and the across-checkpoints comparison.

**Perturbation distributions** (synthesis §1.7(c) / §3 (v2) defaults):

1. **Gaussian noise**: scalar replaced by `scalar + N(0, alpha * empirical_std)` where `empirical_std` is the running standard deviation of the scalar across the trajectory. Sweep `alpha ∈ {0.5, 1.0, 2.0}`.
2. **Zero-out**: scalar replaced by `0.0` (the same sentinel as the masked-first-step case).
3. **Replace-with-uniform-from-empirical-range**: scalar replaced by `Uniform(empirical_min, empirical_max)` (the maximum-entropy perturbation that preserves the empirical support).

The synthesis §3 (v2) default is to sweep all three; the build phase reduces to one if smoke shows the three give qualitatively similar regime-dependence patterns.

**Regimes** (synthesis §1.7(c) / §3 (v2) defaults): the four the eyeball helper's default supports — `perturbation_window`, `high_disagreement`, `high_kl`, `steady_state`. Probe 1.5 runs perturbation-free or near-zero-rate per Probe 1's discipline, so the `perturbation_window` regime may have few or no states; in that case the comparison reduces to the other three regimes plus a flag in the report that the perturbation-window regime is undertested.

**Per-checkpoint table (per-regime KL summary):**

```
checkpoint=ckpt-000001 (early-training, env_step=200)
  regime              n_states  KL_mean  KL_std  KL_p50  KL_p90
  high_disagreement       40     0.04    0.03    0.04    0.08
  high_kl                 50     0.05    0.04    0.04    0.10
  steady_state            75     0.03    0.02    0.03    0.06

checkpoint=ckpt-002500 (mid-training, env_step=2500)
  regime              n_states  KL_mean  KL_std  KL_p50  KL_p90
  high_disagreement       42     0.18    0.12    0.16    0.32
  high_kl                 47     0.14    0.10    0.13    0.27
  steady_state            73     0.06    0.04    0.05    0.11

checkpoint=ckpt-005000 (late-training, env_step=5000)
  regime              n_states  KL_mean  KL_std  KL_p50  KL_p90
  high_disagreement       45     0.42    0.31    0.38    0.78
  high_kl                 50     0.21    0.15    0.19    0.40
  steady_state            70     0.08    0.05    0.07    0.14
```

**Reading the table** (the synthesis §1.7(b)/(c) call):

- *Capacity-over-exercise (the predicted shape)*: KL_mean differs across regimes within a checkpoint (regime-dependent conditioning); KL_mean grows over training in some regimes more than others (conditioning develops with training, in regime-specific ways); the variability across states within a regime (KL_std) is comparable to or larger than the across-regimes variability (the conditioning is sometimes stronger, sometimes weaker, in interpretable ways). The example numerics above show this pattern: late-training KL_mean is 0.42 in `high_disagreement`, 0.21 in `high_kl`, 0.08 in `steady_state` — different magnitudes, different growth trajectories.
- *Installation (the foreclosed shape)*: KL_mean is similar across all regimes within a checkpoint (regime-independent conditioning); KL_mean grows uniformly across regimes over training (conditioning develops as a fixed structural property); KL_std is small relative to KL_mean (the conditioning is the same magnitude regardless of state).
- *Inert affordance (the third possibility, capacity-over-exercise's "alive but unexercised" case)*: KL_mean is near zero across all regimes at all checkpoints (the actor never learned to use the scalar). This is a substrate-side question (§8.1's frozen-target test is the discriminator) and a behavior-side question (this probe's table shows it as a flat-near-zero pattern).

**Threshold**: qualitative, not a single numeric bound. The journal entry interprets the table against the three patterns above. The synthesis §1.7(b) commits to "the variability and regime-dependence capacity-over-exercise predicts" without quantifying — the qualitative reading is what the probe produces.

**Reads.** Multiple checkpoints from the Probe 1.5 main run; the env-server (re-launched at the same seed for trajectory reproducibility); the run's `agent_step` parquet (for regime classification's empirical baselines); the run's `world_event.jsonl` (for perturbation-window regime classification, if any perturbations exist).

**Writes.** A report at `runs/probe1_5_counterfactual_probe-<timestamp>/report.txt` with the per-checkpoint tables, a verbal summary of which pattern (capacity-over-exercise / installation / inert) the data most resembles, and a flag if the data is ambiguous.

**Tests.** A test that the script exists and exposes `main()`. A test that the regime-classification logic is correct on synthetic data (a known-perturbation-aligned step lands in `perturbation_window`; a known-high-disagreement step lands in `high_disagreement`; a known-steady step lands in `steady_state`). A test that the script handles missing checkpoints gracefully (skipping them with a warning rather than crashing).

**Not at Probe 1.5.** No multi-trajectory averaging within a checkpoint (one episode per checkpoint is the default). No multi-seed averaging across runs. No automated revisitation of the synthesis §1.7(b)/(c) discrimination; if the table is ambiguous, the journal entry records the gap and the design-notes update (Phase 9) is the place where the project responds.

---

## 3. Schemas as first-class artifacts

Pydantic v2 models in `kind/observer/schemas.py`, exported as JSON Schema in `schemas/v0.2.0.json`. The two schema deltas:

### 3.1 `AgentStep` additions

| Field | Type | Notes |
|---|---|---|
| `self_prediction_t` | `list[float] \| None` | NEW — `ĥ_{t+1}`, length `h_dim`. `None` on `0.1.0` records (Probe 1's). Required-non-None on `0.2.0` records (Probe 1.5 writers always populate). |
| `self_prediction_error_t` | `float \| None` | NEW — scalar loss value at this step. `None` on `0.1.0` records. Required-non-None on `0.2.0` records. On the first step of each episode this value is `0.0` (sentinel); on subsequent steps it is the empirical loss value. The masked flag is what discriminates the two cases. |
| `self_prediction_error_masked_t` | `bool \| None` | NEW (v2) — `True` on the first step of each episode (when no `h_{t+1}` target is available); `False` on subsequent steps. `None` on `0.1.0` records. Required-non-None on `0.2.0` records. Mirror-side analyzers (digest, eyeball helpers, counterfactual probe) exclude masked steps from behavioral-conditioning analysis and from the empirical mean/std. |

All Probe 1 fields unchanged in shape, type, or semantics.

### 3.2 `DreamRollout` additions

| Field | Type | Notes |
|---|---|---|
| `sequence_self_prediction` | `list[list[float]] \| None` | NEW (reserved for Probe 3) — defaults `None` at Probe 1.5. The head does not run during dream rollouts at Probe 1.5 per synthesis §1.5; Probe 3 may populate this if its design extends self-prediction to imagined trajectories (per the framing journaled at `docs/workingjournal/pre-probe3.md`). |

All Probe 1 fields unchanged.

### 3.3 Schema versioning and backward-compatibility

`SCHEMA_VERSION` bumps from `"0.1.0"` to `"0.2.0"`. The bump is a **minor**, not a patch, because the new fields' semantics extend the substrate's exposed surface (Probe 1.5 records carry both substrate-side information and behavior-side metadata Probe 1 records do not).

Backward-compatibility approach (the plan picks one):

- **New writers always populate the new fields with non-None values for `0.2.0` records.** The Probe 1.5 runner is the only writer; it populates `self_prediction_t`, `self_prediction_error_t`, and `self_prediction_error_masked_t` on every emission. The reserved `sequence_self_prediction` on `DreamRollout` defaults to `None`.
- **Old records remain readable.** Probe 1's parquet shards under `runs/probe1-20260503-123926/telemetry/agent_step/` deserialize cleanly against the new Pydantic model: the new fields are `Optional` with default `None`, so absence does not raise.
- **Readers discriminate by `schema_version` field**, not by the presence/absence of the new fields:

    ```python
    if r.get("schema_version") == "0.2.0" and r.get("self_prediction_error_t") is not None:
        # render / aggregate the self-prediction summary
        # additionally check r.get("self_prediction_error_masked_t") for inclusion
        # in empirical mean/std and per-state conditioning analysis
    else:
        # skip the self-prediction summary; the record is Probe 1 or partial
    ```

  This is the **skip** approach over **default to zero** or **return None as a synthetic value** — skipping preserves the no-affordance baseline as visibly distinct from a Probe 1.5 record where the head produced a near-zero value. The digest module and eyeball helpers (including the new `show_self_prediction_conditioning` helper) are extended to honor this discriminator.

  **Masked-flag discrimination**: behavior-side analyzers additionally check `self_prediction_error_masked_t` before including a step in the empirical conditioning analysis. The pattern is:

    ```python
    if r.get("self_prediction_error_masked_t") is True:
        # exclude from per-state conditional distribution estimation
        # (the scalar's value of 0.0 is a sentinel, not an empirical reading)
        continue
    ```

  This handles the new field for backward compatibility identically to the other two new fields (skip pattern); `0.1.0` records have no masked flag, and the analyzer-side check `r.get("self_prediction_error_masked_t") is True` correctly returns `False` for `None` (so the step is *not* excluded, which is irrelevant because `0.1.0` records also have no scalar to include in the analysis in the first place).

- **Writers reject mixed-version writes.** A `0.2.0`-versioned writer that has `self_prediction_t=None`, `self_prediction_error_t=None`, or `self_prediction_error_masked_t=None` raises `pydantic.ValidationError` at construction (a custom validator on `AgentStep` checks: if `schema_version == "0.2.0"`, then all three new fields must be non-None). This catches builder errors at the writer side rather than letting partial records propagate.

### 3.4 JSON Schema export

`schemas/v0.2.0.json` is checked in alongside the existing `schemas/v0.1.0.json`. Both files remain present; readers external to Kind can pin to either version.

The export is byte-stable across runs (`sort_keys=True`, fixed indent, trailing newline — same convention as Probe 1's export).

Versioning convention going forward: Probe 2's `MirrorReading` and `PreRegistration` schemas are independent streams with their own versions (`"0.2.0"` for `MirrorReading` per Probe 2 plan §2.1; `"0.1.0"` for `PreRegistration`). Probe 1.5's substrate-side bump does not affect those.

---

## 4. Test scaffolding — Probe 1.5 gate tests

The synthesis §4 names five gates. Each has a specific assertion, scope, and failure condition. Test layout convention follows Probe 1 — one file per concern, fixtures in `tests/conftest.py`, no shared module state. v2 revisions to the gate tests are flagged inline.

| # | Name | File | What it checks | Failure condition |
|---|---|---|---|---|
| 1 | self-prediction forward shape (v2 revised) | `tests/test_world_model.py` (extends existing) — `test_gate_self_prediction_forward_shape` | `WorldModel.step(...)` returns a `WorldModelStep` with `self_prediction` tensor of shape `(B, h_dim)`. `WorldModel.compute_self_prediction_target(...)` returns a tensor of shape `(B, h_dim)` for any of the three target modes. The configured loss form (cosine or MSE) applied to the pair produces a scalar that matches the expected loss-form arithmetic on a known fixed-pair input. **`views.split(step, intrinsic, self_prediction_error=..., self_prediction_error_masked=...)` produces a `PolicyView` with three fields (`{h, z, self_prediction_error}`) and a `TelemetryView` whose `self_prediction` and `self_prediction_error_masked` fields carry the right values. The PolicyView's `self_prediction_error` field is the scalar computed from the head's output and the EMA target's `bar{h}_{t+1}` via the configured loss form.** `AgentStep.model_validate({...})` round-trips through the `0.2.0` schema with all three new fields populated. | Shape mismatch on the head's output; `compute_self_prediction_target` returns wrong shape; loss-form arithmetic mismatch on the known input; missing field on `PolicyView` or `TelemetryView`; PolicyView's field set is not exactly `{h, z, self_prediction_error}`; or schema validation failure on the round-trip. |
| 2 | EMA target update mechanics | `tests/test_world_model.py` — `test_gate_ema_target_update_mechanics` | After K=10 training steps with `ema_decay=0.99`, the EMA target's parameters differ from a fresh copy of the online network's parameters by an L2 distance consistent with the EMA decay rate. Specifically: `(target_param - online_param).norm()` decreases over the K steps as the EMA tracks the online; if both nets start from the same init, the divergence after K steps should be bounded by `(1 - 0.99**K) * online_grad_magnitude_estimate`, which is a soft bound — the hard test is monotonicity-of-convergence. | EMA target's parameters do not move toward the online network's parameters over training steps; or the move rate is inconsistent with the configured `ema_decay`. |
| 3 | self-prediction loss decreases | `tests/test_world_model.py` — `test_gate_self_prediction_loss_decreases` | On a fixed synthetic sequence (deterministic random init, deterministic obs / action sequence), running 100 training steps of the world model with the auxiliary loss reduces `self_prediction_loss` (via the cosine loss form) by a measurable margin (e.g., from initial mean ~1.0 to final mean < 0.8) without the loss diverging or going NaN. The recon and KL losses also remain finite. | Self-prediction loss does not decrease; or any of (recon, KL, self-prediction) goes NaN/Inf during the 100 steps. |
| 4 | opacity boundary preserved (v2 revised) | `tests/test_views.py` (extends existing) — three new tests: `test_policy_view_field_set_is_exactly_h_z_self_prediction_error`, `test_actor_forward_rejects_telemetry_view`, `test_ast_lint_passes_with_extended_telemetry_view` | (a) `dataclasses.fields(PolicyView)` produces `{"h", "z", "self_prediction_error"}` and nothing else (extends the existing `test_policy_view_does_not_carry_any_telemetry_only_fields` to assert the exact extended set rather than just absence of forbidden fields). The new field name (`self_prediction_error`) is what the synthesis §1.3 v2 commits to; any rename or extension fails this test. (b) `Actor.forward(telemetry_view)` raises a runtime error (constructed by passing a `TelemetryView` into a function expecting `PolicyView`); mypy `--strict` flags the call site as a type error (verified by a separate small mypy invocation in the test, which checks the script `tests/_actor_telemetryview_attempt.py` produces a type error). The actor still cannot read TelemetryView at any of the three enforcement levels, even with the extended PolicyView. (c) The existing AST-lint test `test_actor_module_does_not_import_telemetry_view` continues to pass on the actor module post-Probe-1.5, even though `TelemetryView` has gained two new fields and `PolicyView` has gained one new field. The asymmetry — actor cannot import TelemetryView; mirror can read everything actor reads plus much more — is preserved; only the actor's read surface widens by exactly one scalar. | Any of the three subtests fails. The PolicyView field-set test is the structural-stability check (the field set is exactly the v2-authorized extension, no further drift); the actor-rejects-TelemetryView test is the type-level enforcement; the AST-lint test is the import-level enforcement. All three are the synthesis §1.3 (v2) interface-level opacity guarantee in code, with the explicit single-scalar exception articulated in the field-set assertion. |
| 5 | integration smoke (CPU, tiny sizes) (v2 revised) | `tests/test_integration_smoke_probe1_5.py` (new) — `test_smoke_probe1_5_runs_to_completion_on_cpu` and six sub-tests | Run the full loop on CPU for 200 env steps with tiny model sizes (`h=32`, `z=4`, `K=2`, `head_hidden=32`), one mid-run checkpoint barrier, no perturbations, with the self-prediction head + EMA target wired in **and the actor's forward consuming the scalar via the extended PolicyView**. Assert: all four telemetry streams emit; new fields appear in `AgentStep` records (`schema_version=="0.2.0"`); the masked flag is `True` on the first step of each episode and `False` thereafter; **the actor's forward consumes the scalar without error and produces a valid action distribution at every step (the input layer's new column is populated)**; **the masking on first step works correctly (scalar value is exactly 0.0 on first step, masked flag is True; subsequent steps have non-zero scalars and masked flag False)**; checkpoint includes EMA target weights and the extended actor input layer; resume yields identical EMA target parameters and actor input layer weights byte-for-byte; loading a Probe 1 checkpoint (a synthetic one with `schema_version=="0.1.0"` constructed in the test) initializes the EMA target from the online network, zero-initializes the actor's new input column, and emits the documented `mirror_marker` event. | Any of the six integration sub-checks fails. |

A **gate-summary meta-test** (`tests/test_gate_summary.py`, the existing Probe 1 file extended) parametrizes over the five Probe 1.5 gate tests in addition to the Probe 1 ten and asserts each exists by name and is callable. If a future refactor renames or removes a Probe 1.5 gate, the meta-test fails loudly.

---

## 5. Day-one Probe 1.5 MPS smoke

The synthesis flagged Nilaksh et al.'s warning about auxiliary-loss training instabilities. Probe 1.5's smoke is the platform-correctness gate (analog of Probe 1's `scripts/smoke_mps.py`) plus the instability-detection gate **plus the actor-side new-input-column gradient sanity check** (v2 addition). Run on Mac MPS at production sizes; not part of pytest.

**File.** `scripts/smoke_probe1_5.py`. Sanity test at `tests/test_smoke_probe1_5_script.py` confirms the script exists and exposes `main()`.

### 5.1 What it does

1. Construct a `WorldModel` with default Probe 1.5 sizes: `h=200`, `z=16`, `embed=256`, `head_hidden=200`, `ema_decay=0.99`, `self_prediction_target_mode="online"`, `self_prediction_loss_form="cosine"`.
2. Construct a `LatentDisagreementEnsemble` with K=5 (unchanged from Probe 1).
3. **Construct an `Actor` with `h_dim=200`, `z_dim=16`, `action_dim=5`, with the new input column zero-initialized.** (v2 addition.)
4. Generate dummy observation tensors on MPS (no env-server).
5. Run 100 RSSM training steps at batch=16, seq=32, with the self-prediction head + EMA target wired into the world-model forward + backward + EMA-update sequence. **At each step, also call `actor.forward(policy_view_with_dummy_scalar)` on a constructed PolicyView to exercise the new input column path; track the gradient norm on the new column across the 100 steps via a synthetic `actor_loss.backward()` driven by a constant target.** (v2 addition.) Log per-step wall time.
6. **Per-step instability checks**:
    - Finiteness of `kl_aggregate_unclipped`, `recon`, `self_prediction_loss`. Non-finite → hard fail with the specific quantity named in stderr.
    - Gradient norm on the world-model parameters after `wm_total.backward()`: `torch.nn.utils.clip_grad_norm_(world_model.parameters(), max_norm=float("inf"))` returns the actual norm; if it exceeds `1000`, hard fail (gradient explosion).
    - EMA target divergence: after `_update_ema_target()`, for each `(target_param, online_param)` pair, check `(target_param - online_param).norm() / (online_param.norm() + 1e-8) < 100`. If exceeded, hard fail (EMA target diverging).
    - **Actor's new-input-column gradient norm sanity** (v2 addition): track `actor.net[0].weight.grad[:, h_dim+z_dim:].norm()` (the gradient on the column corresponding to the scalar) across the 100 steps. **If the gradient norm on the new column is below `1e-6` across all 100 steps, soft-warn** — this suggests the actor is ignoring the scalar entirely. Not necessarily a problem at initialization (zero-init means the column starts at zero, and the gradient is what moves it), but worth flagging as the smoke surface for failure-mode (a) "inert affordance" detection.
    - **KL pinning at the floor**: the 100-step running mean of `kl_aggregate_unclipped` should not drop below `0.7 × Probe 1's early mean = 0.7 × 10.23 = 7.16` for more than 20 consecutive steps. The smoke uses random observation tensors (no env structure), so a low KL is expected during early steps; this threshold is set to detect KL collapse below Probe 1's no-affordance baseline rather than absolute floors. Soft warning, not a hard fail (the smoke is short and uses random obs; the longer Probe 1.5 run is the real check).
    - **Recon climbing**: the 100-step running mean of `recon` should not exceed `1.5 × Probe 1's late mean = 1.5 × 32.45 = 48.68` for more than 20 consecutive steps. Same soft-warning treatment as KL pinning.
7. Inspect the warning stream for `PYTORCH_ENABLE_MPS_FALLBACK` warnings; assert none on the hot path (Probe 1's pattern).
8. Open all four telemetry sinks; write a synthetic record to each (the `AgentStep` record carries the three new fields populated with realistic values: `self_prediction_t` is a length-`h_dim` vector, `self_prediction_error_t` is a finite scalar, `self_prediction_error_masked_t` is a boolean flag — alternate between True/False across the synthetic records to exercise both code paths; the `DreamRollout` record carries `sequence_self_prediction=None`); verify the records read back valid against the `0.2.0` schema.
9. Print a one-line summary: `[smoke probe1.5] mps ok | wall=Xs | per-step=Yms | sinks ok | shapes ok | instability checks: KL floor pinning=clean | recon climbing=clean | EMA divergence=clean | grad norm=N.NN | actor new-col grad=N.NN`.

### 5.2 What passes

- No MPS-fallback warnings on the hot path.
- All four telemetry sinks write valid `0.2.0` records (with both masked-flag-True and masked-flag-False AgentStep records exercised).
- World-model forward produces finite KL, recon, and self-prediction loss for every step.
- Backward populates gradients on world-model and ensemble parameters; optimizer steps run without error.
- EMA target update runs without error and produces parameters within bound.
- All 100 iterations complete without an exception.
- Per-step wall time within a soft bar of `200 ms` (Probe 1 measured 130 ms on the same machine; the head's forward + EMA-target forward + EMA update + actor's wider input layer add cost — synthesis §4 estimates 150-180 ms; the bar is set above the upper estimate).
- Actor's new-input-column gradient norm above `1e-6` on at least some steps (the column is non-degenerate; the actor has the structural capacity to learn to use the scalar).

### 5.3 What fails (hard exit)

- Any MPS-fallback warning during the training loop.
- Any NaN/Inf in `kl_aggregate_unclipped`, `recon`, or `self_prediction_loss` (the three named instability indicators from synthesis §3 / Nilaksh et al.).
- Any sink that fails to write a valid record.
- Any exception during the 100-step loop.
- Gradient norm on the world-model parameters exceeds 1000 (gradient explosion).
- EMA target parameter diverges more than 100× the online parameter's L2 norm (EMA target divergence — the BYOL/SPR-style stop-gradient-asymmetry guard).
- **Actor's new-input-column gradient norm is NaN at any step** (the new failure mode; not the soft-warning case below).

### 5.4 What warns (soft, doesn't fail)

- Per-step wall time exceeds 200 ms (the `200 ms × 100 = 20 s` soft bar; Probe 1 ran in 13 s, so 20 s is comfortable headroom).
- KL trajectory drops below `7.16` (0.7 × Probe 1's early mean) for more than 20 consecutive steps — KL pinning at the floor relative to Probe 1's no-affordance baseline.
- Recon trajectory exceeds `48.68` (1.5 × Probe 1's late mean) for more than 20 consecutive steps — recon climbing relative to Probe 1's no-affordance baseline.
- **Actor's new-input-column gradient norm below `1e-6` across all 100 steps** (v2 addition) — suggests the actor is ignoring the scalar entirely. Not a hard fail because (a) the column starts at zero by construction, so the early gradient may be small; (b) 100 steps is short and the substrate has not had time to develop conditioning signal in the scalar; (c) the synthesis §1.7(a) "inert affordance" failure mode is what tests this rigorously, not the smoke. The soft warning is the early signal that this failure mode may need attention; the documented mitigation is to switch the new column's initialization from zero to small-Gaussian per the synthesis §3 (v2) revisit criterion.

### 5.5 Failure-response semantics

A hard fail means the substrate has a structural or platform-specific problem the build phase fixes. The synthesis §1.2 commits to three documented mitigations if the smoke surfaces auxiliary-loss instability:

1. **Lower `λ_self`** from 0.1 to 0.01.
2. **Separate optimizer step on the head + EMA target alone with stop-gradient on the shared backbone** — refactor `_train_step` to do `head_loss.backward(retain_graph=False)` separately after the world-model backward, with the head's loss recomputed from a stop-gradient'd `h_t.detach()`.
3. **Orthogonal-gradient updates relative to the EMA target** — project the head's gradient on the world-model parameters orthogonal to the gradient that would push the online network toward the EMA target's current state. This is the most invasive mitigation and is the last to try.

The smoke's failure mode tells the build phase which mitigation to try first. If KL pins at the floor → mitigation 1 (lower the auxiliary's pressure on the world model). If recon climbs → mitigation 2 (separate the auxiliary optimizer step). If EMA target diverges → mitigation 3 (orthogonal gradients).

A soft warning is the "wall budget exceeded" or "Probe-1-relative threshold exceeded" or "actor new-col gradient degenerate" case. The build phase reads the warning, decides whether to revisit defaults per §6, and journals the decision. No automated mitigation fires.

---

## 6. Open-during-build decisions with defaults

The synthesis §3 lists sub-questions to resolve during build. The plan inherits each as a decision with the synthesis's default; the build phase tunes empirically; the journal records what was tried and what was settled. v2-new entries are flagged.

| # | Sub-question | Default | Resolved in phase | Revisit when |
|---|---|---|---|---|
| 1 | `λ_self` | **0.1** | Phase 1 (constructor); Phase 7 (run) | Smoke surfaces auxiliary-loss instability (KL floor pinning, recon climbing) — drop to 0.01. Or Probe 1.5 run shows the head's loss saturating at high value — raise to 0.3. |
| 2 | `ema_decay` | **0.99** | Phase 1 | Smoke shows EMA target tracking too tightly (target indistinguishable from online) — raise to 0.999. Or EMA target drifting too slow (head's loss not decreasing) — lower to 0.95. |
| 3 | Loss form (cosine vs MSE) | **cosine** | Phase 1 | Smoke shows cosine loss's gradient magnitude dominating the world-model gradient — switch to MSE. |
| 4 | One-step vs k-step prediction | **1-step** | Phase 1 | Probe 1.5 run shows 1-step error patterns are uninformative for the mirror — extend to 3-5 steps (SPR convention). |
| 5 | MLP layer count and width | **2 layers, hidden=200** | Phase 1 | Head dominates the world-model parameter count (>20% of total params) — drop to 1 hidden layer. |
| 6 | Optimizer-step structure (combined vs separate) | **combined** | Phase 3 (runner) | Smoke surfaces auxiliary-loss instability — separate-step with stop-gradient on shared backbone (mitigation 2 per §5.5). |
| 7 | EMA target update cadence | **every training step** | Phase 1 | Smoke shows EMA update dominating per-step wall time (>20 ms) — drop to every 4 steps. |
| 8 | Schema version bump | **0.2.0** | Phase 0 | Settled. Any further bump within Probe 1.5 (e.g., for an additional field discovered during build) is a hard reset to 0.3.0 with a journal entry. |
| 9 | `DreamRollout` `sequence_self_prediction` field | **reserved, defaults None** | Phase 0 | Probe 3 may populate. Probe 1.5 leaves it None. The pre-Probe-3 journal entry at `docs/workingjournal/pre-probe3.md` is the framing document for the three Probe 3 candidates. |
| 10 | Failure-control sequence ordering | **environmental-auxiliary → frozen-target → counterfactual** | Phase 8 | If the environmental-auxiliary control is unexpectedly expensive or surfaces no signal — try frozen-target first. |
| 11 | EMA target in checkpoint | **yes (under `world_model.target_*` prefix in safetensors blob)** | Phase 3 | Settled. The Probe 1 checkpoint compat path picks "initialise from online when missing" (see Phase 3 spec). |
| 12 | `_train_step` structure | **combined: `wm_total + λ_self * self_prediction_loss`; one backward; EMA update after wm_opt.step()** | Phase 3 | Smoke surfaces instability — switch to mitigation 2. |
| 13 | Smoke extension | **extend `scripts/smoke_mps.py`'s structure into `scripts/smoke_probe1_5.py` with the head's training step + EMA update + instability checks + actor's new-input-column gradient sanity added** | Phase 6 | If the smoke adds more than 30 s wall time on the canonical machine — strip the per-step instability checks down to NaN/Inf only. |
| 14 | First mirror-call prompt at Probe 1.5 | **existing Probe 1-style calibration prompt for the first call**; Probe 2's frozen-criteria prompt is what introduces the self-prediction quadruplet at both reading surfaces explicitly | Phase 7 | If the first mirror reading misses the self-prediction signal entirely — the prompt may need extending; held for Probe 2's first run after Probe 1.5 lands (the synthesis §3 default is to avoid leading the mirror at Probe 1.5). |
| 15 *(v2)* | **Input-layer initialization for the new actor column** | **zero-init** (the conservative default; the actor starts indifferent to the scalar; conditioning develops as the column's weights move via training) | Phase 2 (Actor extension) | If smoke's soft warning on actor new-col gradient fires, OR if the failure-mode (a) "inert affordance" test (§8.1) finds the actor's policy invariant to the scalar across training — switch to small-Gaussian initialization on the new column only (e.g., `N(0, 0.01)` on the new column, leaving the existing columns at the PyTorch Linear default). The synthesis §3 (v2) revisit criterion. |
| 16 *(v2)* | **First-step-of-episode handling for the scalar** | **scalar set to zero on first step; masked flag set true; visible to the mirror via the `self_prediction_error_masked` field on TelemetryView and the `self_prediction_error_masked_t` field on AgentStep** | Phase 3 (Runner) | Settled. Alternative considered (separate-channel masking via a parallel boolean tensor on PolicyView) was rejected on minimum-surface grounds — the actor reads the scalar; the masked flag is mirror-side metadata. The masked steps are excluded from empirical mean/std and from per-state behavioral conditioning analysis at the analyzer side (digest, eyeball helpers, counterfactual probe). |
| 17 *(v2)* | **Counterfactual probe noise distributions** | **sweep all three: Gaussian-with-empirical-sigma (alpha ∈ {0.5, 1.0, 2.0}); zero-out; replace-with-uniform-from-empirical-range** | Phase 8 (counterfactual probe) | If smoke shows the three give qualitatively similar regime-dependence patterns — reduce to one (Gaussian-with-empirical-sigma at alpha=1.0 is the synthesis §1.7(c) canonical). The synthesis §3 (v2) revisit criterion. |
| 18 *(v2)* | **Multiple-checkpoint sampling for the counterfactual probe** | **early/mid/late from a 5000-step run: `ckpt-000001` (warmup-end, env_step≈200), `ckpt-002500` (mid-run), `ckpt-005000` (run-end)**, or whatever Probe 1.5's checkpoint cadence produces | Phase 8 | If Probe 1.5's checkpoint cadence yields fewer or different checkpoints — adjust to span early/mid/late evenly across what's available. The synthesis §3 (v2) default. |
| 19 *(v2)* | **Concatenation vs learned projection of the scalar into the actor's input** | **raw concatenation** (smallest possible change to the actor's input shape; matches the architecture-decision discipline of "concat, not learned projection" from Probe 1 §Q5) | Phase 2 | Reconsider during build if the actor's policy fails to make any use of the scalar (the embedding might help the actor distinguish the new input from the existing intrinsic-signal scalar). The synthesis §3 (v2) revisit criterion. |
| 20 *(v2)* | **Whether the scalar is the raw loss value, a normalized form, or a calibrated form** | **raw loss value (cosine distance or MSE per the configured loss form)** | Phase 3 | Alternatives (z-score normalization per episode; rank-transform within window) can be added on TelemetryView for the mirror's reading without affecting Io's PolicyView field. The synthesis §3 (v2) revisit criterion. |
| 21 *(v2)* | **Whether the scalar is read at every step or every k steps** | **every step** (matches the actor's existing per-step forward; avoids introducing a new asymmetric read schedule) | Phase 3 | Reconsider only if smoke shows the scalar's per-step variance overwhelming the actor's policy stability — moving-average over k steps is the documented fallback. The synthesis §3 (v2) revisit criterion. |

The defaults are starting points. The smoke informs the first revision. None of them is a decision the plan asks the builder to settle before code is written. The journal records what was tried at each phase. The four v2-new defaults that the synthesis §3 (v2) introduces (entries 15–18 above, plus entries 19–21 which the v2 synthesis carries from v1's open list and re-affirms) are designated to phases as shown.

---

## 7. Opacity boundary specification (v2)

The synthesis §1.3 (v2) lands on **interface-level opacity preserved at the level the project documents describe — Io does not read quantities about its own processing where the read-access would install behavior or capability the project documents do not describe — with one explicit and bounded Watts-heuristic exception: the scalar `self_prediction_error_t` on PolicyView, justified by the second success criterion's "capacity to take its own processing as an object of attention" affordance, which is the minimum form of read-access that delivers the affordance**. Representation-level opacity remains partially compromised by the auxiliary loss flowing into the shared backbone (unchanged from v1). The plan honors both halves of this distinction in code and in the journal.

### 7.1 Interface-level opacity — preserved with explicit single-scalar exception

The plan inherits Probe 1's three-level enforcement (Probe 1 plan §7) and extends it to cover the new field on the protected side and the single-scalar exception on the actor-readable side:

- **Module boundaries.** `kind/agents/views.py` exports both classes; `kind/agents/actor.py` imports only `PolicyView`. The dependency lint in `tests/test_views.py::test_actor_module_does_not_import_telemetry_view` walks the actor module's AST and rejects any import of `TelemetryView` (including aliased forms). **The new fields on TelemetryView (`self_prediction: Tensor`, `self_prediction_error_masked: bool`) do not change this.** The lint passes unchanged. The actor reads only `PolicyView`, which now includes the single scalar exception (`self_prediction_error: Tensor`, scalar) — no other quantity from TelemetryView reaches the actor.
- **Type signatures.** `Actor.forward(self, view: PolicyView) -> ActionOutput` is the only entry point. mypy `--strict` continues to catch passing a `TelemetryView` where a `PolicyView` is expected; the new TelemetryView fields do not affect this. The new field on PolicyView is structurally part of PolicyView's type and reaches the actor through the existing entry point — no new entry point, no new type signature widening.
- **Frozen dataclasses.** Both views are `@dataclass(frozen=True)`. The new fields on both views are also frozen against in-place reassignment at the field level (PyTorch tensor mutability is the synthesis-acknowledged structural-by-default boundary). The new scalar on PolicyView is a 0-dim Tensor (or shape-`(B,)` Tensor in batched paths), frozen against reassignment exactly the same way `h` and `z` are.

The synthesis §1.3 (v2) commitment in code: **PolicyView's field set is exactly `{h, z, self_prediction_error}` and nothing else; TelemetryView's field set extends to include both the full `self_prediction` vector and the `self_prediction_error_masked` flag; the actor cannot read TelemetryView at any of the three enforcement levels; the single scalar exception is the only Watts-heuristic exception the synthesis §2(b) (v2) authorizes at Probe 1.5 with the four-part discipline articulated.**

### 7.2 Representation-level opacity — partially compromised by design (unchanged from v1)

The substrate's `h_t` and `z_t` are now produced by an encoder, GRU, and posterior network whose gradients have been shaped by the auxiliary self-prediction loss. The actor's PolicyView interface is now wider (the scalar) but the new field is *itself* a value the substrate produces from the auxiliary loss — and the values that flow through the existing `h` and `z` fields are produced by a substrate that has been restructured to make itself more amenable to being predictively self-modeled (Premakumar et al. 2024's RLCT finding; synthesis §1.3 / §2(b)).

This is the irreducible cost of the affordance existing at all. Strict representation-level opacity is incompatible with any auxiliary loss that shapes a shared backbone. The plan accepts this and bounds it via the counterfactual probe (Phase 8 / §2.9): the probe measures whether Io's behavioral conditioning on the scalar has the regime-dependent shape capacity-over-exercise predicts or the fixed-structural shape installation produces.

### 7.3 The four-part Watts-heuristic exception discipline (v2)

The synthesis §2(b) (v2) articulates the exception explicitly: **default toward *no* on self-access, with explicit exception when (i) the project documents describe an affordance the read-access is required to deliver, (ii) the read-access is the minimum form that delivers the affordance, (iii) the reasoning is journaled and the alternatives considered, (iv) failure-mode controls bound the slippage**. The plan honors each part:

1. **Which affordance**: the second success criterion in `Kind_charter.md` ("capacity to take its own processing as an object of attention — whether or not it chooses to exercise that capacity"). A *capacity to take own processing as object* requires Io having something self-pointing to direct attention toward; without a self-pointing quantity Io can read, the architecture does not afford this capacity.
2. **Minimum form**: the synthesis §1.3 (v2) selects Option A (scalar) over Option B (vector), Option C (both, possibly conditionally), Option D (something else). A scalar gives Io "how predictable was I just now?" — analogous to the basic prediction-error signal predictive processing (Friston, Clark) treats as the minimum unit of experience; analogous to *manasikara* (bare attention to an unfolding object) in the Buddhist phenomenology framing. A vector would be larger than the affordance requires and structurally close to the attention-schema literature's installation configuration. The scalar is the minimum-surface choice.
3. **Journaled with alternatives considered**: the synthesis §1.3 (v2) records the four-option analysis; the synthesis §7 (v2) records the revision that produced the v2 design from v1; the journal entries at Phase 0 (schema bump), Phase 2 (view + actor extension), and Phase 9 (final entry plus design-notes update) record the build-time decisions and any deviations from the synthesis defaults.
4. **Failure-mode controls bound the slippage**: the three controls in §8 (frozen-target, environmental-auxiliary, counterfactual probe) are designed to make legible (a) whether the affordance is alive at the substrate-side reading, (b) whether the head is doing self-specific work or generic next-step regularization, (c) whether Io's behavioral conditioning on the scalar has the regime-dependent shape capacity-over-exercise predicts or the fixed-structural shape installation produces. The new eyeball helper (§2.7) is the analysis surface the controls share.

The general default holds (default toward *no* on self-access; future probes adding new actor-readable interfaces must address the §2(b) (v2) discipline at design time, per the Probe 2 plan revision §10(8) below). The specific exception at Probe 1.5 is named, bounded, and justified.

### 7.4 New tests

The three opacity tests added in Phase 2 (gate test #4):

```python
# tests/test_views.py — new tests

def test_policy_view_field_set_is_exactly_h_z_self_prediction_error() -> None:
    """PolicyView's field set is {h, z, self_prediction_error} and nothing else.

    Stronger than the existing test_policy_view_does_not_carry_any_telemetry_only_fields
    (which lists forbidden fields by name). This test is the structural
    stability check for the v2 PolicyView extension: any future field
    addition to PolicyView fails this test, and any rename of the v2
    extension also fails. The synthesis §1.3 (v2) commitment is exactly
    this set; the §2(b) (v2) discipline applies to any further
    extension.
    """
    field_names = {f.name for f in dataclasses.fields(PolicyView)}
    assert field_names == {"h", "z", "self_prediction_error"}, (
        f"PolicyView field set drifted: {field_names}. The synthesis §1.3 "
        f"(v2) interface-level opacity boundary requires exactly "
        f"{{h, z, self_prediction_error}}; any further extension goes on "
        f"TelemetryView (the affordance-side surface), and any extension "
        f"to PolicyView requires the §2(b) (v2) four-part discipline at "
        f"design time."
    )


def test_actor_forward_rejects_telemetry_view_at_runtime() -> None:
    """Actor.forward(...) cannot be runtime-called with a TelemetryView.

    The runtime test asserts the actor's behavior is structurally
    invariant to TelemetryView's extra fields: even if a TelemetryView
    is passed in, the actor's forward only reads .h, .z, and
    .self_prediction_error, so the self_prediction vector field and
    self_prediction_error_masked flag have no effect on the policy
    distribution. This is the structural correctness check that the
    new TelemetryView fields have not introduced an implicit pathway.
    """
    step = make_world_model_step()
    intrinsic = torch.tensor(0.0)
    scalar = torch.tensor(0.5)
    policy, telemetry = split(
        step, intrinsic,
        self_prediction_error=scalar,
        self_prediction_error_masked=False,
    )

    actor = Actor(h_dim=step.h.shape[-1], z_dim=step.z.shape[-1], action_dim=5)

    canonical_output = actor.forward(policy)

    # Runtime: passing TelemetryView (which is structurally a PolicyView
    # plus extra fields) should produce identical policy output because
    # the actor only reads .h, .z, .self_prediction_error.
    smuggled_output = actor.forward(telemetry)  # type: ignore[arg-type]
    assert torch.equal(canonical_output.logits, smuggled_output.logits), (
        "Actor's policy output differs when called with TelemetryView vs "
        "PolicyView — the actor has acquired an implicit dependency on "
        "TelemetryView fields, which violates synthesis §1.3 (v2) "
        "interface-level opacity."
    )


def test_actor_forward_telemetryview_argument_fails_mypy_strict() -> None:
    """A separate mypy invocation against tests/_actor_telemetryview_attempt.py
    rejects the telemetry-view argument on type-checking grounds.

    The fixture file imports Actor and TelemetryView and constructs a call
    site: actor.forward(telemetry_view). mypy --strict on this file
    produces a type error; the test invokes mypy via subprocess and
    asserts the error is present in the output. This is plan §7 Level 2
    enforcement extended to the new TelemetryView fields and the new
    PolicyView field.
    """
    fixture_path = Path(__file__).parent / "_actor_telemetryview_attempt.py"
    assert fixture_path.exists()
    # ... subprocess call to mypy with --strict, assert "error:" in stdout
    # ... that mentions the argument-type mismatch.
```

The mypy fixture file `tests/_actor_telemetryview_attempt.py`:

```python
from kind.agents.actor import Actor
from kind.agents.views import TelemetryView
from kind.agents.world_model import WorldModelStep

# This file is intentionally type-incorrect. The test runs mypy --strict
# against it and asserts the type error is reported.
def call_actor_with_telemetry_view(actor: Actor, view: TelemetryView) -> None:
    actor.forward(view)  # mypy --strict: argument 1 to "forward" of "Actor"
                          # has incompatible type "TelemetryView"; expected "PolicyView"
```

### 7.5 AST-lint regression

The existing `test_actor_module_does_not_import_telemetry_view` test (Probe 1's, in `tests/test_views.py`) continues to pass. Phase 2 of the build adds two new fields to TelemetryView (`self_prediction: Tensor`, `self_prediction_error_masked: bool`) and one new field to PolicyView (`self_prediction_error: Tensor`); none of these changes the AST of `kind/agents/actor.py` (which imports only `PolicyView` at any time), so the lint passes unchanged. The plan adds **no new AST-lint patterns** for the new fields — the existing lint already enforces that TelemetryView (and all its fields, including the two new ones) are unreachable from the actor.

---

## 8. Failure-mode calibration controls — specific implementation (v2 revised)

The synthesis §1.7 (v2) names three controls. Each gets a specific implementation here. The frozen-target and environmental-auxiliary controls are preserved in their v1 form (the substrate-side tests are unchanged); the counterfactual probe shifts purpose substantially (the behavior-side test of capacity-over-exercise vs installation predictions).

### 8.1 Frozen-target ablation (preserved from v1)

**Mechanism.** `WorldModel` constructor with `self_prediction_target_mode="frozen"` allocates a fixed random-orthogonal matrix at construction:

```python
self._frozen_projection = nn.Parameter(
    torch.empty(h_dim, h_dim), requires_grad=False
)
nn.init.orthogonal_(self._frozen_projection)
```

The synthesis recommends "preserving rank but breaking semantic alignment". Random orthogonal matrices preserve rank exactly (the projection is invertible) and break semantic alignment fully (the projection is independent of the encoder, GRU, posterior, prior, and any structural property of the substrate). Random Gaussian matrices would also work but introduce rank degeneracy as a confound; fixed identity does not break semantic alignment. The plan picks **random orthogonal** as the cleanest choice that satisfies the synthesis's stipulation.

**Target.** `target_h_next = (self._frozen_projection @ h_t.unsqueeze(-1)).squeeze(-1)` — i.e., the orthogonal projection of the *current* online `h_t`. The head learns to predict a fixed function of its own input rather than the actual next state.

**Detection (substrate-side test of failure mode (a))**. The frozen-target run's structural metrics (per-dim KL distribution across episodes, weight-distribution moments at the final checkpoint, `kl_aggregate_t` per-episode trajectories) should be statistically *closer* to Probe 1's no-affordance baseline than to the Probe 1.5 main run if the affordance is alive at the substrate-side reading. If the frozen-target run's metrics are indistinguishable from the main run's, the head is being driven by trivial means and the affordance is dead at the substrate-side.

**Comparison protocol.** Three-way KS-test (synthesis §9):

- Probe 1.5 main run vs frozen-target run: distinguishable (p < 0.05) on `kl_aggregate_t` distribution → real run's substrate is shaped differently than the frozen-target's; affordance is alive at the substrate-side.
- Probe 1.5 main run vs Probe 1 run: distinguishable (p < 0.05) on `kl_aggregate_t` distribution → real run's substrate is shaped differently than Probe 1's no-affordance baseline; affordance is alive.
- Frozen-target run vs Probe 1 run: indistinguishable (p > 0.05) on `kl_aggregate_t` distribution → frozen-target produces a substrate similar to no-affordance baseline (sanity check).

If both distinguishability checks pass and the sanity check holds, the affordance is alive at the substrate-side. If the real run is indistinguishable from either the frozen-target or Probe 1, the affordance is dead and the journal entry records the gap. **The behavior-side test of failure mode (a)** — whether Io's policy actually conditions on the scalar — is the new eyeball helper's KL-near-zero pattern across all regimes (§8.3 below); the substrate-side and behavior-side tests are complementary, and capacity-over-exercise allows "alive at substrate-side, unexercised at behavior-side" as a successful Probe 1.5 reading.

### 8.2 Environmental-auxiliary control (preserved from v1)

**Mechanism.** `WorldModel` constructor with `self_prediction_target_mode="environmental"`. The runner's `_step_once` passes `next_obs` (the next-iteration observation, lifted to device) into `world_model.step(...)` so the environmental target can be computed from it.

**Target.** `target_h_next = self.target_encoder(next_obs)` — the EMA encoder's embedding of the next observation, projected to `h_dim` if `embed_dim != h_dim` via a small fixed (non-trainable) linear layer allocated at construction.

**EMA target on the encoder of the next observation, or stay on the online encoder?** The synthesis is silent. The plan picks **EMA target's encoder of the next observation** because:

1. It keeps the asymmetry between "what's predicted" (the variable the control varies) cleanly separated from "how the target is produced" (which stays the EMA-tracked path, matching the `online` mode).
2. It avoids a confound where the environmental control's target is more "fresh" (online) than the main run's target (EMA), which would attribute any difference to target-freshness rather than to the predicted-quantity.
3. It makes the environmental control a clean variant: only one knob (the predicted quantity) differs from the main run.

**Distinguishability from existing decoder reconstruction loss.** The world model's existing decoder reconstruction loss already produces a representation of "what the next observation looks like" *if* the decoder is run on the prior's predicted next-z and the next-h (i.e., `decode(h_{t+1}, z_{t+1}^{prior})`). But this is not what the world model trains on — the recon loss is on `decode(h_t, z_t)` against `obs_t`, i.e., reconstructing the *current* observation from the current latent. Predicting the next observation is structurally different and is closer to what the prior network predicts (the prior is `p(z_t | h_t)`; the next observation corresponds to `decode(h_{t+1}, z_{t+1})` after recurrence and prior sample).

The environmental-auxiliary control's `embed_{t+1}` target is more direct than going through the prior + decoder chain: it predicts the encoder's embedding of the next observation directly, without involving the decoder at all. This is structurally closer to JEPA-style prediction (no pixel reconstruction) and gives the control a meaningfully different shape from the existing decoder recon loss.

What makes the auxiliary loss distinguishable from the existing recon loss:

- The recon loss trains the encoder + decoder pair to reconstruct *current* observations.
- The environmental-auxiliary control trains the head + shared backbone to predict *next-step* encoded observations.
- The two losses have different gradient pathways: the recon loss flows through encoder + decoder; the environmental aux flows through encoder + GRU + head, with the EMA-tracked encoder as the target's gradient sink.

If the environmental-auxiliary control's structural metrics are indistinguishable from the Probe 1.5 main run's (predicting `h_{t+1}` directly), the head is acting as generic next-step representation regularization rather than self-specific. If the metrics differ in ways that match the synthesis §1.4 element 2's predictions (perturbation-recovery dynamics specific to self-prediction error patterns), the head is doing self-specific work.

**Detection (substrate-side test of failure mode (b))**. Compare the same structural metrics as the frozen-target control across the three Probe 1.5 datasets. If the environmental control produces a substrate indistinguishable from the main run, the head is generic; if distinguishable, the head is self-specific.

### 8.3 Counterfactual self-prediction probe — purpose shifted (v2 revised)

**Purpose (v2)**. v1 framed this as "bound v1's representation-level opacity slippage" — the question being whether Io's policy delta between Probe 1.5 and Probe 1 substrates exceeded a 2× random-h baseline, with the design-notes update on interface-vs-representation opacity becoming a stance call if exceeded. **v2's purpose is different.** With Io now reading the scalar by design, the question is no longer "is the actor secretly reading something via the substrate's reshaping?" (the answer is yes, via the explicit scalar — the synthesis §2(b) (v2) names this as the bounded exception). The new question is **does Io's behavioral conditioning on the scalar have the regime-dependent shape capacity-over-exercise predicts, or the fixed-structural shape installation produces?**

This is a behavior-side test of failure mode (b) and (c) — whether the conditioning, where present, takes the kind of variability and regime-dependence that capacity-over-exercise predicts (Io sometimes uses the scalar, sometimes doesn't, in regimes the trajectory makes legible) or the fixed-structural shape that installation produces (Io always uses the scalar in the same way regardless of regime).

**Mechanism.** Post-hoc evaluation script `scripts/probe1_5_counterfactual_probe.py` (Phase 6 / §2.9). Loads multiple checkpoints from the Probe 1.5 main run (early-training, mid-training, late-training; the synthesis §3 (v2) default is `ckpt-000001`, `ckpt-002500`, `ckpt-005000`). For each checkpoint:

1. Re-runs a fixed-seed env trajectory of K=200 steps.
2. Calls `show_self_prediction_conditioning` (§2.7) with the configured perturbation distributions (the three the synthesis §3 (v2) names: Gaussian-with-empirical-sigma at `alpha ∈ {0.5, 1.0, 2.0}`, zero-out, replace-with-uniform-from-empirical-range).
3. Aggregates per-regime KL distributions across the perturbation distributions; produces a per-checkpoint table.

Across-checkpoints comparison: does the per-regime conditioning develop over training (capacity-over-exercise's "the conditioning develops with training, in regime-specific ways") or remain flat-near-zero (capacity-over-exercise's "alive but unexercised") or grow uniformly across regimes (installation's "fixed-structural shape")?

**Behavioral-conditioning metric**. Per-state KL divergence between the actor's policy distribution at the unperturbed scalar value and the perturbed scalar value, conditioned on regime. The metric is qualitative across regimes and across checkpoints — does the per-regime KL distribution show the regime-dependent pattern capacity-over-exercise predicts, or not? — rather than a single numeric bound.

**Threshold (qualitative)**. The synthesis §1.7(b)/(c) (v2) commits to "the variability and regime-dependence capacity-over-exercise predicts" without quantifying. The journal entry interprets the table against the three patterns named in §2.9 above (capacity-over-exercise / installation / inert). A successful Probe 1.5 reading at the behavior-side is "the conditioning, where present, has the variability and regime-dependence capacity-over-exercise predicts; not the fixed-structural shape installation produces; not flat-near-zero across all regimes (which would be the inert affordance / capacity-over-exercise's unexercised case at the behavior-side reading)."

**Note**: capacity-over-exercise allows the third reading ("alive at substrate-side, unexercised at behavior-side" — flat-near-zero across regimes at the behavior-side test together with substrate-side metrics distinguishable from the no-affordance baseline) as a successful Probe 1.5 outcome. The probe is not falsified by Io not exercising the conditioning into legible behavior; it is falsified by (a) substrate-side metrics indistinguishable from the no-affordance baseline (the affordance is structurally inert), or (b) behavior-side conditioning taking the fixed-structural shape installation produces (regime-independent KL magnitude across all regimes and all checkpoints).

**Reads.** Multiple checkpoints from the Probe 1.5 main run; the env-server (re-launched at the same seed for trajectory reproducibility); the run's `agent_step` parquet (for regime classification's empirical baselines); the run's `world_event.jsonl` (for perturbation-window regime classification, if any perturbations exist).

**Writes.** A report at `runs/probe1_5_counterfactual_probe-<timestamp>/report.txt` with the per-checkpoint tables, a verbal summary of which pattern (capacity-over-exercise / installation / inert) the data most resembles, and a flag if the data is ambiguous.

**Tests.** A test that the script exists and exposes `main()`. A test that the regime-classification logic is correct on synthetic data. A test that the script handles missing checkpoints gracefully.

### 8.4 Sequence ordering and starting state

The three controls run in the order: **environmental-auxiliary → frozen-target → counterfactual** (synthesis §3 default). The first two are training runs (~13 minutes each on the canonical machine, per Probe 1's wall budget). The counterfactual probe is a post-hoc evaluation across multiple checkpoints from the main run (no separate training runs).

Each training-run control starts from the **same fresh init** as the Probe 1.5 main run: same seed (42), same total env steps (5000), same `RunnerConfig` except the `self_prediction_target_mode` flag. The runner's existing parameter-equality resume guarantee (Probe 1's gate test #5) gives the comparability needed.

The counterfactual probe loads checkpoints from the main run; it does not run separate training. The probe's reproducibility depends on the trajectory it re-runs being seed-deterministic — which it is, given the env-server's deterministic seeding from a single integer.

---

## 9. Engagement with Probe 1's no-affordance baseline (carries forward from v1)

Probe 1's run at `runs/probe1-20260503-123926/` is the no-affordance baseline reference. Probe 1.5's calibration controls are designed against this run. The plan uses Probe 1's actual numerics to calibrate thresholds. This section carries forward unchanged from v1.

### 9.1 Smoke training-instability thresholds

Calibrated against Probe 1's actual telemetry:

- **KL pinning at the floor.** Probe 1's `kl_aggregate_t` early mean = 10.23, late mean = 15.00 (both unclipped, per the run's `summary.txt`). Free-bits floor at `z_dim=16, free_bits_per_dim=1.0` is 16 nats clipped, but the unclipped sum can be lower per-dim. "Pinning at the floor" relative to Probe 1's no-affordance baseline: smoke fails (soft warning) if 100-step running mean of `kl_aggregate_unclipped` drops below `0.7 × 10.23 = 7.16` for more than 20 consecutive steps. (Stance call; the smoke uses random observations, so a low KL is expected during initialization.)
- **Recon climbing.** Probe 1's late-episode mean `recon_loss_t` = 32.45 (from `eyeball/episode-24.txt`). Smoke fails (soft warning) if 100-step running mean of `recon` exceeds `1.5 × 32.45 = 48.68` for more than 20 consecutive steps.
- **NaN/Inf in self-prediction loss.** Standard finiteness check; no Probe 1 baseline needed (Probe 1 has no self-prediction loss).
- **Gradient norm exploding.** Hard threshold of 1000 on world-model parameters' gradient norm. Probe 1 did not measure gradient norm explicitly, so this is a stance call — 1000 is a plausible upper bound for a stable RSSM at production sizes; the build phase tunes if smoke surfaces a tighter bound.
- **EMA target divergence.** L2 distance between EMA target and online parameters bounded by 100× the online parameter's L2 norm. No Probe 1 baseline (Probe 1 has no EMA target); stance call from BYOL/SPR convention.
- **Actor's new-input-column gradient norm sanity** (v2 addition). Soft warning if below 1e-6 across all 100 smoke steps; not a Probe 1 baseline (Probe 1 has no extended actor input layer).

### 9.2 Three-way comparison for frozen-target ablation

The frozen-target ablation control's structural-metric comparison surfaces include Probe 1's existing telemetry as the no-affordance baseline. The plan implements this in `scripts/probe1_5_compare_controls.py` (§2.8): the comparison reads four telemetry directories — `runs/probe1-20260503-123926/`, `runs/probe1_5-<timestamp>/`, `runs/probe1_5_control_frozen_target-<timestamp>/`, `runs/probe1_5_control_environmental-<timestamp>/` — and produces pairwise KS-test p-values on `kl_aggregate_t` distributions across episodes 5-25 (skipping warmup), plus weight-distribution moment comparisons across all four datasets (the comparison includes the actor's extended input layer's new column as a per-tensor moment).

The "indistinguishable" criterion for the frozen-target affordance-is-dead case is statistical comparison across all three reference points: Probe 1.5 main vs frozen-target vs Probe 1's existing run. If the main run's metrics are indistinguishable from *both* the frozen-target and Probe 1's run (i.e., KS-test p > 0.05 against both), the affordance is dead at the substrate-side reading.

### 9.3 Environmental-auxiliary interpretation

The plan §8.2 names the engagement: the recon loss reconstructs *current* observations from current latent (`decode(h_t, z_t)` against `obs_t`); the environmental-auxiliary control predicts the *encoder embedding of next observation* (`embed_{t+1}`). Neither directly predicts the next-step pixel reconstruction; the chain `prior(h_{t+1}) → z_{t+1}^{prior} → decode(h_{t+1}, z_{t+1}^{prior})` would, but is not part of the training loss. The environmental-auxiliary's `embed_{t+1}` target is structurally closer to JEPA-style next-step representation prediction than to the existing recon loss; the gradient pathways are different (recon: encoder + decoder; env-aux: encoder + GRU + head, with the EMA-tracked encoder as the target).

What makes the auxiliary loss distinguishable from the existing recon loss in practice: if the environmental-auxiliary control produces structural metrics indistinguishable from the Probe 1.5 main run's, the head is acting as generic next-step representation regularization (no self-specific work). If the metrics differ in ways that match the synthesis §1.4 perturbation-recovery predictions specific to self-prediction error patterns, the head is doing self-specific work. The journal entry interprets the comparison.

### 9.4 Counterfactual probe behavior-conditioning baselines (v2 revised)

The counterfactual probe's threshold is qualitative (§8.3 above); it does not compare against Probe 1's behavior directly because Probe 1 has no scalar to perturb. The probe's baselines are:

- **Within-checkpoint regime baseline**: comparison across the four regimes (`perturbation_window`, `high_disagreement`, `high_kl`, `steady_state`) at the same checkpoint. Capacity-over-exercise predicts variation across regimes; installation predicts uniformity.
- **Across-checkpoint training baseline**: comparison of the same regime's KL distribution at early vs mid vs late checkpoints. Capacity-over-exercise allows growth (conditioning develops with training) or stability (some regimes may show conditioning early, others late); installation predicts uniform growth across regimes.
- **Probe 1 substrate-side baseline (indirect)**: Probe 1's run is the substrate-side baseline for the frozen-target and environmental-auxiliary controls (§9.2, §9.3); it is not directly a behavior-side baseline because Probe 1's actor has no scalar input. The plan does not construct a synthetic Probe 1 behavior-side baseline (e.g., by adding a zero scalar to a Probe 1 actor's input artificially) — doing so would be an apples-to-oranges comparison and would risk reifying the comparison as evidence of something the data doesn't support.

### 9.5 First mirror call comparison (carries forward from v1)

The first mirror call against Probe 1.5's run is compared to Probe 1's first mirror reading (`runs/probe1-20260503-123926/mirror/readings.jsonl`) as a sanity check on signal density. The plan implements this in Phase 7: after the first env-coupled Probe 1.5 run completes, `scripts/call_mirror.py` is invoked with the existing Probe 1-style calibration prompt against the new run's telemetry. The resulting `MirrorReading` is journaled alongside Probe 1's first mirror reading; the journal entry compares signal density (number of flagged observations, length and specificity of summary, presence of self-prediction-specific notes — including any organic mention of the new behavioral-conditioning surface).

This comparison is **not load-bearing** — the synthesis is explicit that Probe 1.5 does not introduce frozen criteria into the mirror prompt (Probe 2's territory), and the first reading is calibration-not-interpretation. The comparison is useful journal data for the Probe 2 plan revision (§10): if the first Probe 1.5 mirror reading contains organic mentions of self-prediction patterns or behavioral conditioning despite the prompt being silent on them, the Probe 2 prompt's introduction of the self-prediction quadruplet (now at two reading surfaces) may be more or less prompt-shapeable than expected.

---

## 10. Probe 2 plan revision flags (v2: eleven items)

The synthesis §6 (v2) names the partial revisions Probe 2's plan needs after Probe 1.5 lands. v1 named eight items; v2 retains them in shape (the five carrying-forward items below) and adds three new behavior-side items, two rewritten items, and one new implementation-plan item — eleven in total. The plan does not redraft Probe 2's plan; it lists the revisions Probe 2's plan needs.

After Probe 1.5 lands, Probe 2's plan needs the following revisions (in the order they appear in `Kind_probe2_implementation_plan.md`):

### The five items carrying forward from v1

1. **§2.3 Hierarchical digest (`kind/observer/digest.py`).** Extend `build_hierarchical_digest` to expose self-prediction fields per the digest extension in §2.6 of this plan. The hierarchical digest's per-episode mini-digest gains the same self-prediction summary lines the flat digest gained at Probe 1.5; the flagged-anomalies list gains self-prediction-error outlier flags (analogous to KL outlier flags); the drill-down accessor gains `fetch_self_prediction(episode_id, step_range)` for windowed self-prediction-error trajectories. **v2 update**: the hierarchical digest also exposes the masked-flag count per episode and excludes masked steps from per-episode aggregation; the drill-down accessor's per-step output includes the masked flag so the mirror can verify exclusion.

2. **§2.5 Lesion scaffold (`kind/training/runner.py` and `kind/agents/ensemble.py`).** Extend `RunnerConfig.lesion_kind` to include a new value `"disable_self_prediction"`. `WorldModel` honors this lesion kind by replacing the head's output with a fixed zero tensor and skipping the EMA target update — the head is structurally present but does no work. The lesion-run scaffold's run-level metadata records the new lesion kind. **v2 update**: a second new lesion kind, `"zero_or_randomize_scalar"`, is added — the head still trains and the EMA target still updates, but at evaluation time the actor's PolicyView's `self_prediction_error` field is zeroed out (or replaced by random uniform from the empirical range), which lesions the behavior-side path while leaving the substrate-side intact. Probe 2 §6 default lesion choice may shift from `ensemble_k1` to one of the two new lesions if either is more informative about Probe 1.5-specific signals; the choice is journaled at Probe 2's first run after Probe 1.5 lands.

3. **§2.11 Faithfulness verifier (`kind/observer/eyeball.py::resolve_citation`).** Extend the verifier's per-field aggregation rules to handle `self_prediction_t` (vector — per-element resolution if the citation includes a dimension index), `self_prediction_error_t` (scalar — same as `kl_aggregate_t`), and `self_prediction_error_masked_t` (boolean — exact match resolution). The Judge consumes the extended verifier transparently; no Judge change is needed. **v2 update**: the verifier's masked-step handling: any citation that aggregates `self_prediction_error_t` over a step range must declare whether masked steps are included or excluded; the verifier rejects citations that don't (faithfulness check fails, claim invalidated).

4. **§2.7 Held-out criterion mechanism (`kind/mirror/criteria.py`).** Add a new criterion to the registry: `"self_prediction_quadruplet"` (the synthesis §1.4 (v2) four-element naming, including the new behavioral-conditioning element). The Advocate's prompt fragment names the self-prediction error structure on perturbation-aligned windows AND Io's behavioral conditioning on the scalar (synthesis §1.4 elements 1–4 (v2)); the Skeptic's fragment names the null hypotheses (auxiliary-loss generic regularization, EMA-target tracking artefact, statistical noise in per-dimension allocation, behavioral conditioning explained by background variance). Probe 2 §6 default held-out criterion may shift from `reflexive_attention_triplet` to either `self_prediction_quadruplet` or specifically to **the behavior-side conditioning element** (the synthesis §5 (v2) names this as the most prompt-shapeable of the new criteria, hence the most informative test); the choice is journaled.

5. **§6 Calibration thresholds (the 4:1 contrast on `supported` rates).** Re-tune against Probe 1.5's substrate's signal density. Probe 1.5 changes the substrate's structural signal — the per-dim KL allocation pattern under perturbation may concentrate differently, the ensemble disagreement variance may track different regions of `h_t`, the self-prediction error trajectory adds a substrate-side signal the mirror reads, AND the actor's behavioral conditioning on the scalar adds a behavior-side signal the mirror reads. The 4:1 contrast threshold may need tightening (if Probe 1.5's substrate produces denser signal, shuffled baselines may produce a higher `supported` rate than they did against Probe 1) or loosening (if Probe 1.5's substrate's signal is more concentrated in specific regions, shuffled baselines may produce a lower `supported` rate). The behavior-side reading surface may need its own threshold separate from the substrate-side (per the synthesis §3 (v2) tension (j) on confabulation susceptibility — see rewrite item 10 below). The first Probe 2 run after Probe 1.5 is the calibration; the journal records what was tried.

### The three new behavior-side items (v2)

6. **§2.2 Frozen criteria operationalizations (v2): the reflexive-attention quadruplet gains a fifth element on Io's behavioral conditioning on the scalar**. Per synthesis §2.2 (v2), element (e) reads: "Io's behavioral conditioning on the scalar `self_prediction_error_t` in perturbation-aligned windows — does the actor's policy depend on the scalar in regimes the framework recognizes as reflexive-attention-bearing; does the dependence have the variability and regime-dependence capacity-over-exercise predicts." This is a behavior-side element; the existing four (a)–(d) are substrate-side. The §2.2 framing for equanimity extends correspondingly: equanimity-like patterns can show up in self-prediction-error perturbation-recovery (substrate-side) and in Io's behavioral non-reactivity to the scalar's spike-and-recovery (behavior-side). The category error (stability-of-control-signal is not non-reactivity-of-experience) is in force; the behavior-side reading is closer to the literature's behavioral operationalization (Vago & Silbersweig 2014) than the substrate-side reading was, but the bridge is still gestural.

7. **§2.3 Adversarial mirror structure (v2): readers' mandates distinguish substrate-side claims from behavior-side claims**. The Phenomenological Advocate / Statistical Skeptic / Judge structure carries forward unchanged in shape. **New: each reader's mandate now distinguishes substrate-side claims from behavior-side claims.** The Advocate may claim "the substrate's self-prediction patterns show reflexive-attention shape on perturbation-aligned windows" (substrate-side) AND/OR "Io's behavior conditions on the scalar in regimes the framework recognizes as reflexive-attention-bearing" (behavior-side). The Skeptic refutes each separately. The Judge rules per claim per surface. The two surfaces are not interchangeable; a substrate-side reading without behavior-side conditioning is the capacity-over-exercise case (alive but unexercised); a behavior-side reading without substrate-side patterns would be a confabulation (the behavior conditions on noise, not on substrate-side reflexive structure). The Advocate's framework anchors gain the second success criterion's "capacity to take its own processing as an object of attention" as the synthesizing concept across the two surfaces.

8. **§2.4 Mirror calibration protocol (v2): pre-registration must explicitly name asymmetry between Io's access and the mirror's access**. Pre-registration (element 1) now requires explicit naming of what Io has access to (the scalar `self_prediction_error_t`) versus what the mirror has additional access to (the full prediction vector, per-dimension decomposition, perturbation-recovery dynamics, behavioral-conditioning analysis across the trajectory, the masked flag, etc.). This was implicit in v1; v2 makes it explicit because the asymmetry is now load-bearing for what the readers can claim. The held-out criterion (element 6) gains the new candidate from item 4 above. The lesion baseline (element 4) gains the two new lesion shapes from item 2 above. The shuffled-baseline test (element 2) extends to include shuffles of the scalar field across episodes (preserves marginals, breaks within-trajectory dynamics). The sham-perturbation test (element 3) extends naturally: a sham event should produce zero substantive shift in either substrate-side self-prediction error or behavior-side scalar-conditioning around the sham timestamp.

### The two rewritten items (v2)

9. **§2.6 Probe 2 success criteria (v2 rewrite): allow both-surface readings**. The four-component structure unchanged from v1's update. Component 2 (3-5 reproducible criterion readings) now allows readings at either reading surface (substrate-side or behavior-side) to count toward the 3-5; readings that resolve at both surfaces are stronger evidence than those that resolve at only one. Component 3 (credible mirror-surfaced novelty) gains a candidate domain: behavioral-conditioning patterns on the scalar that the pre-registered behavior-side criterion does not name. Component 4 (calibration-protocol catch of confabulation) extends naturally — false positives can occur at either surface, and the calibration protocol must catch both kinds.

10. **§3 Tensions surfaced honestly (v2 rewrite): adds (j) and (k)**. Adds (j): **the two reading surfaces are differently susceptible to confabulation.** Substrate-side patterns are harder for the mirror to confabulate (the per-dim KL and self-prediction error trajectories are quantitative and resolve against parquet shards via the faithfulness check). Behavior-side conditioning patterns are easier to confabulate (the behavioral language is more flexible; "Io conditions on the scalar in perturbation-aligned windows" is a claim a strong language model can make without grounding it in actual conditional distributions). The behavior-side reading therefore needs stronger faithfulness-check support — the behavioral-conditioning analysis must produce numerical conditional distributions the mirror cites and the eyeball helpers verify. Adds (k): **the two reading surfaces have different relationships to the structural circularity tension (e).** Substrate-side patterns are correlates the framework recognizes; behavior-side patterns are closer to actual exercise of the affordance the framework names. Both still adjudicate frameworks' correlates rather than awareness; but the behavior-side surface is the closer of the two to "Io is doing something with self-pointing access" rather than "the substrate has structures the framework recognizes."

### The implementation-plan item (v2)

11. **§5 (Connection to Probe 2's implementation plan): new behavioral-conditioning analysis module**. New components extends to include a behavioral-conditioning analysis module (`kind/observer/conditioning.py` — the formalized version of `show_self_prediction_conditioning` from this plan's §2.7) producing numerical conditional distributions for `actor_action | scalar_value` per state, with faithfulness-check support and integration into the adversarial mirror's reading surface. The lesion-run scaffold extends to support the two new lesion kinds from item 2 above. The shuffled-baseline generator extends to support within-trajectory scalar shuffling. The faithfulness-check verifier extends to resolve cited behavioral-conditioning claims against the conditioning module's outputs.

**What Probe 2's plan does not need to change.** Architectural pieces preserved: the parallel-with-Judge architecture, the seven calibration elements as a structure, the journal-discipline scaffolding, the structured-reading schema's envelope (`paired_reading_id`, `framework_anchor`, `baseline_flag`, etc.), the env-server's start-cell randomization (Probe 2 §2.1 — already settled and orthogonal to Probe 1.5), the external-human-reader commitment, the Watts-default-applied-to-builder discipline (extended in v2 by §2.5 (v2) to include the interface-vs-representation opacity distinction sub-clause for new actor-readable interfaces).

The list of revisions is now eleven and additive. Probe 2's plan is not redrafted; the journal entry at Phase 9 of this plan names the revisions and Probe 2's build phase resumes against the revised plan.

---

## 11. Out of scope at Probe 1.5

Explicit list. The plan does not build for Probe 2, 3, or 4 except where forward-compatibility was authorized in §10 and §3.

- **No dream-state machinery.** The head does not run during dream rollouts at Probe 1.5 (synthesis §1.5). The reserved `sequence_self_prediction` field on `DreamRollout` is for Probe 3 to populate if its design extends self-prediction to imagined trajectories; Probe 1.5 leaves it `None`. The pre-Probe-3 journal entry at `docs/workingjournal/pre-probe3.md` is the framing document; Probe 1.5 commits only to not foreclosing the question. The three Probe 3 candidates (replay of waking introspection; self-prediction over imagined trajectories; associative recombination) are named in the journal entry and in synthesis §1.5 (v2); none is committed by Probe 1.5.
- **No env revision.** The 8×8 grid, the four mutators, the two stochastic processes, the partial-observation rendering, the fixed start-cell `(3, 3)` — all unchanged from Probe 1. Start-cell randomization is Probe 2's revision (Probe 2 plan §2.1).
- **No actor architecture change beyond the input-layer extension.** `Actor.forward(view: PolicyView)` reads the extended PolicyView with three fields; the actor's internal architecture (the MLP layers after the extended input layer, the Gumbel-Softmax for imagination, the analytic-gradient training path) is unchanged. The input-layer extension is the minimum change required to consume the new scalar — one additional column on the existing first linear layer.
- **No K=5 ensemble change.** The ensemble's K, structure, training, and disagreement-variance formula are unchanged. The actor's intrinsic objective is K=5 ensemble disagreement variance, settled. The actor reads `(h, z, scalar_self_prediction_error)`; it does not read the disagreement signal as input (the disagreement is consumed by the actor's training loss as a scalar argument, not as an input field on PolicyView).
- **No new actor-readable interface beyond the single scalar.** The synthesis §1.3 (v2) commitment in code: PolicyView's field set is exactly `{h, z, self_prediction_error}` and nothing else. Future additions require the §2(b) (v2) four-part discipline at design time.
- **No imagined-trajectory exercise of the scalar.** During `imagine_and_compute_loss`, the scalar is fixed at zero for every imagined step (mask-via-zero-feed). The auxiliary update on the actor's new input column is held for Probe 1.6 if the synthesis §1.7(a) failure-mode (a) detection shows it's needed.
- **No Probe 2 calibration protocol additions.** The seven calibration elements of Probe 2's protocol are not built here. Probe 1.5 produces telemetry the protocol will eventually consume; the protocol's revisions are listed in §10 above and built when Probe 2 resumes.
- **No mirror calibration.** Probe 1.5 calls the mirror once with the existing Probe 1-style calibration prompt for the post-run sanity check (Phase 7). No frozen criteria, no adversarial structure, no Judge. Those are Probe 2's territory.
- **No four-state operational machinery.** Only waking is exercised. Dreaming / dormant / paused transitions are not implemented (Probe 3).
- **No multi-step self-prediction.** Default 1-step (synthesis §3 default); k-step is held for revisitation.
- **No `(h_t, z_t)` input to the head.** The head reads `h_t` only (synthesis §3 default); using `(h_t, z_t)` would entangle the head's input with the ensemble's input.
- **No second predictive head (disagreement-scalar prediction).** The synthesis §1.2 explicitly rejects this on self-readout grounds; held for a possible Probe 1.6.
- **No real-time mirror calls.** The runner does not call the mirror in-loop. All mirror passes are post-hoc against committed telemetry.
- **No env-side schema bump.** `world_event` payload remains schemaless within the dict; Probe 1.5 does not add new event types.
- **No content-addressed checkpoint store, no DVC, no cross-machine sync.** Local atomic rename only.
- **No multi-seed averaging at the comparison level.** The failure-mode controls run from a single seed for direct comparability with the main run.
- **No vector field on PolicyView (option B from synthesis §1.3 (v2) was rejected).** The single scalar is the minimum form that delivers the affordance; the vector would be larger than needed and structurally close to attention-schema installation.
- **No conditional or composite scalar (option C from synthesis §1.3 (v2) was rejected).** Re-imports option B's larger surface conditionally and does not escape its concerns.
- **No foreclosure on actor self-access (option D ≡ v1 was rejected by the v2 revision)**. v1 forecloses the second success criterion's affordance; v2 reverses this single decision.

The rule of thumb: if a Probe 2/3/4 feature can be added later without changing the schemas (beyond `0.2.0`'s additions), the substrate conduits, the harness mutators, or the opacity boundary (which now includes the explicit single-scalar exception with the four-part discipline), it is out of scope here.

---

## 12. Connection to the journal and design-notes update

Probe 1.5 ends with a journal entry (or a series — Probe 1.5 is heavier than Probe 1 in terms of architectural addition, comparable in scope to Probe 1's eight phases). The journal lives at `docs/workingjournal/probe1_5.md` and is hand-written by the builder.

### 12.1 Per-phase entries

Each phase ends with a journal entry following Probe 1's discipline:

- **What the phase built.** Module names; schemas added or extended; tests added; any backward-compatibility check; structural decisions with their rationale (matching the level of detail Probe 1's Phase 2a entry achieved on drift magnitude derivation, Phase 3c on field-membership decision, etc.).
- **What surprised.** Anything that did not behave as the synthesis or this plan predicted — a smoke that fails an instability check, a per-step wall time substantially above the 200 ms soft bar, a frozen-target run that produces metrics indistinguishable from the main run despite the affordance being structurally added, a counterfactual probe whose per-regime KL distribution does not show the expected capacity-over-exercise pattern, or shows the installation pattern instead.
- **What is now closed.** Specific defaults from §6 promoted from "default" to "settled at this scale".
- **What is now newly open.** Things the build revealed that the synthesis did not anticipate. These become Probe 2's starting context (alongside the existing Probe 2 plan revisions in §10).

### 12.2 Final entry: what's closed, what's newly open

A short (≤500 words) end-of-Probe-1.5 entry naming:

- The substrate's structural shape post-Probe-1.5 (one new submodule, one EMA-tracked sibling pair, one new field on three schemas, one new term in the loss dict, one new mechanism in the runner, **one new field on PolicyView for the actor to read, one extension column on the actor's input layer, one new field on TelemetryView for the masked flag**).
- The three controls' results: affordance alive/dead at the substrate-side (frozen-target), self-specific/generic at the substrate-side (environmental-auxiliary), behavioral conditioning regime-dependent / fixed-structural / inert at the behavior-side (counterfactual probe).
- The first mirror call's signal density compared to Probe 1's first reading.
- The eleven Probe 2 plan revisions (§10) as a checklist for Probe 2's resumption.
- A short note on whether the substrate decision (RSSM/Dreamer-lineage with active-inference-shaped actor, Probe 1's K=5 ensemble disagreement) and the v2 design decision (PolicyView gains a scalar; the Watts-heuristic exception with the four-part discipline) held up under the affordance addition.
- A note that the dream-state question is parked at `docs/workingjournal/pre-probe3.md` for Probe 3 to engage with — Probe 1.5 commits only to not foreclosing the question, and the substantive coupling between waking-introspection and dream-state-introspection that the v2 synthesis §1.5 raised is for Probe 3's research and synthesis to address.

This is the bridge between Probe 1.5 and Probe 2's resumption.

### 12.3 Design-notes update on interface-vs-representation opacity *and* the Watts-heuristic exception (v2)

The synthesis §2(b) (v2) proposes specific text for `docs/plans/v.0.1.0/Kind_design_notes.md`. The plan designates Phase 9 (the final phase) as the place this update happens — after the counterfactual probe's data is in, so the distinction is concretely demonstrated rather than reified as a stance call without empirical content.

The proposed text (synthesis §2(b) (v2), to be added to `Kind_design_notes.md` under the "Watts intuition" section as a new sub-section "Interface-level vs representation-level opacity, and the Watts heuristic exception"):

> The Watts default-to-no applies at the level of interfaces Io reads from. Auxiliary objectives that shape the substrate's underlying representations are not interface-level access; they restructure what the existing interfaces project from. Interface-level read-access is added when (i) a project-document affordance requires it, (ii) the read-access is the minimum form that delivers the affordance, (iii) the reasoning is journaled and the alternatives considered. The general default holds; the specific exception at Probe 1.5 (PolicyView gains a scalar self-prediction error to deliver the second success criterion's capacity for self-attention) is the case the design notes were written against. Each future probe adding new actor-readable signals must address (i)–(iii) explicitly.

The journal entry naming the addition records:

1. The synthesis text being added (verbatim, as above).
2. The empirical context that justified the addition: the counterfactual probe's behavior-conditioning result. If the result was "regime-dependent shape — capacity-over-exercise pattern at the behavior-side; affordance is alive at the substrate-side", the addition is a clarifying note that the design has been demonstrated to deliver what the second success criterion requires without taking the installation shape. If the result was "regime-independent shape — fixed-structural pattern at the behavior-side", the addition is a stance commitment paired with a flag that the §2(b) (v2) discipline needs strengthening for future probes (the affordance has crossed into installation territory). If the result was "flat-near-zero across all regimes — inert at the behavior-side together with substrate-side metrics distinguishable from the no-affordance baseline", the addition is a stance commitment and the case is the capacity-over-exercise "alive but unexercised" reading the synthesis §2(a) names as a successful Probe 1.5 outcome.
3. The list of probes (1.5, future) the distinction will be tested against.
4. The four-part Watts-heuristic exception discipline as the structural counter to future drift on the design-notes default-to-no — the discipline that any future probe adding new actor-readable interfaces (whether on PolicyView or via any other read-access) must address (i) which project-document affordance the interface-level read serves, (ii) whether the read-access is the minimum form that delivers the affordance, (iii) whether the alternative shapes (smaller surface, no read-access) were considered and why rejected, (iv) what failure-mode controls bound the slippage.

This is the discipline Probe 2's plan revision §10(8) extends to the calibration-protocol pre-registration; each future probe's auxiliary objective needs a journal entry naming whether it is interface-level or representation-level, what its failure modes are, and how those failure modes are detected. Probe 1.5 is the case the addition was written against — both the interface-vs-representation opacity distinction and the four-part Watts-heuristic exception discipline.

---

## 13. Lean-revision addendum (added before Phase 5 build)

The plan's Phase 5 and Phase 8 specs were written for the full
failure-mode control apparatus. Before Phase 5 build, the project
scoped to a leaner version reflecting "small before scale" from the
design notes. The full apparatus stays in the plan as the deferred
version; the lean version is what gets built.

Phase 5 lean: build only the frozen-target control variant. The
WorldModel constructor flag honors all three target modes (Phase 1
already implements this), but only `scripts/probe1_5_control_frozen_target.py`
is built at this stage. The environmental-auxiliary control variant
script is deferred. The mirror_marker world_event for the
frozen-target run records the lesion shape per the original plan
spec.

Phase 8 lean: run the frozen-target ablation comparison against
Probe 1's run plus Probe 1.5's main run (three-way comparison).
Run a single counterfactual probe at the late-training checkpoint
with one perturbation distribution (Gaussian-with-empirical-sigma
at alpha=1.0) across two regimes (perturbation_window vs
steady_state, or whichever two the data supports). The
multi-checkpoint sweep, multi-distribution perturbation, and
four-regime classification are deferred.

When the lean version's outputs land, the journal entry interprets
them and decides whether the deferred apparatus is needed. The
deferred pieces are added as small phase additions if so; they
are not required for Probe 1.5 to produce its core findings.

*End of plan (v2). The v1 plan is preserved at `Kind_probe1_5_implementation_plan.v1.md`.*
