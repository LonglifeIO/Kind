# Probe 1 — Implementation Plan

*Operational plan that translates the Probe 1 implementation synthesis (`docs/decisions/Kind_probe1_synthesis.md`) and the environment synthesis (`docs/decisions/Kind_environment_synthesis.md`) into a concrete build sequence. The substrate is settled (RSSM/Dreamer-lineage with active-inference-shaped actor, per `Kind_architectural_decision.md`); the Probe 1 implementation is settled (custom minimal RSSM, ensemble-disagreement actor, four telemetry streams, PolicyView/TelemetryView split); the environment is settled (8×8 grid, partial-perspective pixels, two stochastic processes, four mutators, 200-step episodes). This plan does not relitigate those decisions. It specifies the build order, the components, the schemas, the interfaces, the tests, and the day-one smoke test that confirms the loop runs end-to-end. Probe 1 is plumbing — the env / agent / mirror / observer feedback loop running on the real substrate, with telemetry that downstream probes can read. Forward-compatibility lives in the schemas, the substrate conduits, the harness mutators, and the opacity boundary; nothing else is built for Probes 2, 3, or 4.*

---

## 1. Build order with dependency graph

Hard dependencies (must run sequentially):

- **Schemas before writers.** Nothing in the codebase emits a record before its Pydantic model exists in `kind/observer/schemas.py` and is checked in.
- **Telemetry sinks before producers.** Sink modules (JSONL, Parquet) exist before any component that writes telemetry.
- **RSSM world model before actor.** The actor reads `h_t` and `z_t`. Building actor scaffolding without the world model produces a stub that has to be rewritten.
- **Env-server before integration.** The TCP env protocol must round-trip a single step before the runner is wired.
- **Smoke test as final gate.** All other components must exist and pass their own tests before the integration smoke runs end-to-end on MPS.

Parallelizable (no hard dependency between them):

- The four schema files can be drafted in parallel (one per stream).
- Env-server core (grid dynamics, observation rendering) and RSSM core (encoder, dynamics, decoder, ELBO) are independent until the runner ties them together.
- Mirror caller scaffolding is independent of the agent loop.
- Test fixtures (dummy env, dummy world-model state) can be built alongside the components they exercise.

Phased build (each phase is a unit of work):

| Phase | Work | Depends on |
|---|---|---|
| **0. Schemas** | Pydantic models for `agent_step`, `dream_rollout`, `replay_meta`, `world_event`; common envelope; `schemas/v0.1.0.json` checked in. | — |
| **1. Telemetry sinks** | JSONL writer (append-only, fsync on close); Parquet writer (Arrow-backed, sharded); record envelope wrapper; `run_id` / `checkpoint_id` injection. | 0 |
| **2a. Env-server core** *(parallel with 2b)* | 8×8 grid, three cell types, two stochastic processes, episode reset, deterministic seeding. No network yet. | 0 |
| **2b. RSSM core** *(parallel with 2a)* | Encoder, GRU recurrence, prior network, posterior network, decoder, ELBO with free bits. Pure forward/backward, no actor, no replay. | 0 |
| **3a. Env-server harness** *(parallel with 3b, 3c)* | Mutator hooks (`add_resource`, `remove_object`, `set_cell_state`, `move_object`); `world_event` emission; episode reset events. | 1, 2a |
| **3b. Actor & ensemble** *(parallel with 3a, 3c)* | K-head ensemble of one-step latent predictors; disagreement variance; analytic-gradient actor head; uniform preference prior placeholder. | 2b |
| **3c. View split** *(parallel with 3a, 3b)* | `PolicyView` and `TelemetryView` modules; world model's forward pass returns both; type signatures enforce who reads what. | 2b |
| **4. Transport & barrier** | TCP socket protocol (action → transition); checkpoint barrier (env-pause-and-drain). | 2a, 3a |
| **5. Runner** | Replay buffer (FIFO sequence segments); training loop; world-model update + actor update; checkpoint manager. | 1, 3b, 3c, 4 |
| **6. Mirror caller** | Single LLM call reading recent `agent_step` records; structured Pydantic output. | 1 |
| **7. Observer / journal** | Journal scaffold (`docs/workingjournal/probe1.md`); telemetry-eyeballing helpers (no viz). | 1 |
| **8. Five tests + smoke** | The five named tests plus the MPS smoke test. | All |

The plan treats Phase 0 (schemas) as the single load-bearing prerequisite. Schemas are versioned at `v0.1.0` from day one; field additions in later probes will not break older readers.

---

## 2. Per-component specifications

Each entry: file paths, public interfaces, what the component reads/writes, what tests verify it, what it does *not* do at Probe 1.

### 2.1 Schemas — `kind/observer/schemas.py`

**Files.** `kind/observer/schemas.py` (Pydantic models); `schemas/v0.1.0.json` (frozen JSON Schema export, checked in).

**Public interface.** Four Pydantic models — `AgentStep`, `DreamRollout`, `ReplayMeta`, `WorldEvent` — each subclassing a shared `RecordEnvelope` that carries `schema_version: str`, `run_id: str`, `checkpoint_id: str | None`. A `SCHEMA_VERSION = "0.1.0"` module constant.

**Reads.** Nothing.

**Writes.** Nothing. Schemas are pure declarations.

**Tests.** Schema models import; all four can be instantiated with valid dummy data; JSON Schema export is byte-stable across runs.

**Not at Probe 1.** No migration logic (no prior versions exist); no Avro/Protobuf parallel definitions.

### 2.2 Telemetry sinks — `kind/observer/sinks.py`

**Files.** `kind/observer/sinks.py`.

**Public interface.**

```python
class JsonlSink:
    def __init__(self, path: Path, schema: type[BaseModel]) -> None: ...
    def write(self, record: BaseModel) -> None: ...
    def close(self) -> None: ...

class ParquetSink:
    def __init__(self, dir: Path, schema: type[BaseModel], rows_per_shard: int = 10_000) -> None: ...
    def write(self, record: BaseModel) -> None: ...
    def close(self) -> None: ...
```

`agent_step` and `dream_rollout` go to `ParquetSink` (columnar, downstream-friendly). `replay_meta` and `world_event` go to `JsonlSink` (low-volume, human-inspectable).

**Reads.** Schema models from `schemas.py`.

**Writes.** Files under `runs/{run_id}/telemetry/`.

**Tests.** JSONL roundtrip (write → read → schema-validate → equal); Parquet column dtypes match schema; sinks fsync on close.

**Not at Probe 1.** No streaming readers; no compaction; no compression beyond Parquet defaults.

### 2.3 Env-server core — `kind/env/grid_world.py`

**Files.** `kind/env/grid_world.py`.

**Public interface.**

```python
class GridWorld:
    def __init__(self, config: GridWorldConfig, seed: int) -> None: ...
    def reset(self) -> Observation: ...
    def step(self, action: int) -> tuple[Observation, EnvStep]: ...
    @property
    def state(self) -> GridState: ...   # full underlying grid; mirror-side, not agent-side
```

8×8 underlying grid; three cell types (empty, wall, resource); 7×7 ego-centric partial view rendered to 32×32 grayscale (default; RGB switch deferred to smoke-test outcome). Five discrete actions (`up`, `down`, `left`, `right`, `stay`). Resource consumption is a state change triggered by entering a resource cell, not a verb. Two stochastic processes: per-cell Poisson regrowth and aperiodic random-walk drift in regrowth rate. Two separable RNG streams, both reproducible from the seed.

**Reads.** Config (regrowth rate, drift magnitude, episode length, resolution).

**Writes.** Nothing directly — emits `EnvStep` objects the harness wraps into `agent_step` records.

**Tests.** Step shape (observation tensor dims, action range); deterministic replay from seed; episode resets at 200 steps cleanly with no terminal-state signal in observation.

**Not at Probe 1.** No day-night cycle; no toroidal topology; no `introduce_novel_object` mutator (excluded by environment synthesis on no-marker grounds); no movable NPCs.

### 2.4 Env-server harness & mutators — `kind/env/env_server.py`, `kind/env/mutators.py`

**Files.** `kind/env/env_server.py` (TCP listener, run loop, telemetry emission); `kind/env/mutators.py` (the four named mutators).

**Public interface.** The four mutator methods on the env-server — `add_resource(cell)`, `remove_object(cell, type)`, `set_cell_state(cell, state)`, `move_object(cell_from, cell_to)`. Each invokes the underlying `GridWorld.state` mutation and emits a `WorldEvent` record into the `world_event` stream with `source="builder"`. Episode resets emit `WorldEvent(event_type="env_reset", source="environment")`. Internal stochasticity events are emitted as aggregate per-episode counts at Probe 1 (per-event logging deferred to Probe 4).

**Reads.** TCP socket (actions from Mac trainer); `GridWorld.state`.

**Writes.** TCP socket (transitions to Mac trainer); `world_event` JSONL stream (Probe 1 emits these directly to the Mac's telemetry sink, separate from the agent process).

**Tests.** Perturbation hook test — invoke `add_resource`, confirm a `world_event` record appears with the right fields and that the agent's observation contains no marker.

**Not at Probe 1.** Mutators are not invoked by any scheduler; the harness exposes them but Probe 1 runs perturbation-free or near-zero (one-or-zero invocations as a plumbing test).

### 2.5 RSSM world model — `kind/agents/world_model.py`

**Files.** `kind/agents/world_model.py`.

**Public interface.**

```python
class WorldModel(nn.Module):
    def encode(self, obs: Tensor) -> Tensor: ...                                  # encoder embedding
    def prior(self, h: Tensor) -> tuple[Tensor, Tensor]: ...                       # μ, log-σ for p(z|h)
    def posterior(self, h: Tensor, embed: Tensor) -> tuple[Tensor, Tensor]: ...    # μ, log-σ for q(z|h,o)
    def recurrence(self, h: Tensor, z: Tensor, a: Tensor) -> Tensor: ...           # GRU step
    def decode(self, h: Tensor, z: Tensor) -> Tensor: ...                          # recon
    def step(self, obs, h_prev, z_prev, a_prev) -> WorldModelStep: ...             # full forward
```

`WorldModelStep` is a frozen dataclass containing `h, z, q_params, p_params, kl_per_dim, recon, embed`. The world model's forward returns this object; consumers project to `PolicyView` or `TelemetryView` (see §7).

Sizes default to `h_dim=200`, `z_dim=16` (continuous Gaussian latent). Free bits applied per-dimension: `kl_loss = max(free_bits, kl_per_dim).sum()`. Loss = recon + KL + free-bits-clipped KL aggregate. No reward predictor, no continuation predictor, no return-to-go conditioning.

**Reads.** Observations from env-server.

**Writes.** Nothing directly — exposes the structured step output.

**Tests.** Agent forward — given dummy obs and prior `(h, z)`, returns a valid `WorldModelStep` with shapes matching the config.

**Not at Probe 1.** No JEPA-style reconstruction-free predictor; no categorical latents; no robustness machinery (symlog, twohot, percentile normalization, KL balancing, unimix); no DreamerV3 stability tricks beyond free bits.

### 2.6 Actor & ensemble — `kind/agents/actor.py`, `kind/agents/ensemble.py`

**Files.** `kind/agents/actor.py` (policy head, analytic-gradient training); `kind/agents/ensemble.py` (K-head one-step latent predictors and disagreement variance).

**Public interface.**

```python
class LatentDisagreementEnsemble(nn.Module):
    def predict_next_latent(self, h, z, a) -> Tensor: ...         # shape: K × z_dim
    def disagreement(self, h, z, a) -> Tensor: ...                # scalar variance across K heads

class Actor(nn.Module):
    def __init__(self, view: PolicyView, action_dim: int) -> None: ...
    def forward(self, view: PolicyView) -> ActionOutput: ...      # action, logprob, entropy
```

Actor reads only `PolicyView` (concat of `h_t`, `z_t`). K=5 ensemble heads riding the shared RSSM core. Intrinsic signal = ensemble disagreement variance, clipped at a floor. Pragmatic prior is uniform at Probe 1 — contributes zero pragmatic value but is kept in the loss as scaffolding (`pragmatic_value(uniform_prior) + epistemic_value`) so Probe 4+ can introduce structure without refactoring.

Optimizer: DreamerV1-style analytic gradients through the differentiable latent dynamics. PPO-on-RSSM is documented as fallback in code comments only — not implemented at Probe 1 unless the analytic-gradient actor proves unstable.

**Reads.** `PolicyView` only.

**Writes.** Action and `intrinsic_signal_t` (for telemetry only, not for self-readout).

**Tests.** Agent forward — actor produces a valid action for a `PolicyView` constructed from world-model output; ensemble produces K predictions with finite disagreement.

**Not at Probe 1.** No PPO fallback wired up; no posterior-prior KL routed into reward (settled in synthesis); no learned `f(h,z)` projection (concat only); no introspective head whose target is `h_t`.

### 2.7 PolicyView / TelemetryView — `kind/agents/views.py`

**Files.** `kind/agents/views.py`.

**Public interface.**

```python
@dataclass(frozen=True)
class PolicyView:
    h: Tensor
    z: Tensor

@dataclass(frozen=True)
class TelemetryView:
    h: Tensor
    z: Tensor
    q_params: tuple[Tensor, Tensor]
    p_params: tuple[Tensor, Tensor]
    kl_per_dim: Tensor
    recon_loss: Tensor
    embed: Tensor
    intrinsic_signal: Tensor

def split(step: WorldModelStep, intrinsic: Tensor) -> tuple[PolicyView, TelemetryView]: ...
```

Frozen dataclasses prevent in-place mutation; type hints make `Actor.forward(view: PolicyView)` the only way the actor sees state. The mirror caller and telemetry writers import `TelemetryView`; the actor module imports only `PolicyView`. This is the in-process realization of the synthesis's two-interface boundary.

**Tests.** Unit test that `Actor` cannot be constructed with a `TelemetryView` (mypy-checked) and that `PolicyView` carries no posterior/prior parameters at runtime.

**Not at Probe 1.** No subprocess isolation (deferred — interfaces designed to allow it later); no opaque handle / capability tokens.

### 2.8 Transport & checkpoint barrier — `kind/env/transport.py`, `kind/training/checkpoint.py`

**Files.** `kind/env/transport.py` (TCP socket protocol, length-prefixed framed messages); `kind/training/checkpoint.py` (atomic checkpoint, barrier).

**Public interface.** Mac sends `{"action": int, "request_id": int}`; desktop returns `{"observation": [...], "env_step": int, "episode_id": int, "step_in_episode": int, "wallclock_ms": int}`. Encoding: compact JSON with base64-encoded NumPy serialization for the observation tensor. msgpack with NumPy support is a documented optimization to consider only if the JSON path becomes a measured bottleneck (which is not expected at Probe 1 sizes — 32×32 grayscale tensors are small).

Checkpoint barrier (Probe 1's protocol):

1. Mac trainer sends `BARRIER_BEGIN` to env-server.
2. Env-server stops accepting new actions; drains in-flight transitions to the trainer.
3. Trainer commits replay buffer, weights, optimizer state, RNG state, telemetry offsets to a staging directory.
4. Atomic rename of staging directory into place.
5. Trainer sends `BARRIER_END`; env-server resumes.

Atomic checkpoint contents under a directory that is `os.rename`-d into place after fsync: `weights.safetensors`, `replay_meta.json`, replay parquet shards as of the checkpoint, `optimizer_state.pt`, `rng_state.pkl`, `telemetry_offsets.json`, `schema_version.txt`.

**Reads.** Network frames; replay buffer state.

**Writes.** Network frames; checkpoint directory.

**Tests.** Integration smoke covers the round-trip; barrier protocol is exercised by a mid-run checkpoint in the smoke test.

**Not at Probe 1.** No content-addressed object store; no DVC; no rsync between machines (Probe 1 runs both ends locally on Mac for the smoke; the Mac/desktop split is wired but not stress-tested).

### 2.9 Runner & replay — `kind/training/runner.py`, `kind/training/replay.py`

**Files.** `kind/training/runner.py` (training loop); `kind/training/replay.py` (FIFO sequence buffer).

**Public interface.**

```python
class SequenceReplayBuffer:
    def __init__(self, capacity: int, sequence_length: int = 32) -> None: ...
    def insert(self, transition: Transition) -> ReplayMeta: ...
    def sample(self, batch_size: int) -> tuple[Batch, ReplayMeta]: ...

class Runner:
    def run(self, total_steps: int) -> None: ...
```

Runner ties together: env-server client, world model, actor, replay, telemetry sinks, checkpoint manager. Hot loop: receive transition → encode → step world model → split views → sample action → step env → append to replay → emit `agent_step`. Training loop: sample replay batch → world-model update + actor update → emit nothing more than what the training step naturally produces (training-step records are *not* a fifth stream at Probe 1; they live in the existing Parquet shards as fields on `agent_step`). Periodic dream rollout at default cadence (~1 per 1k env steps), horizon H=15.

**Reads.** Env-server transitions; replay batches.

**Writes.** All four telemetry streams (via sinks); checkpoints (via barrier).

**Tests.** Integration smoke test exercises a short end-to-end run.

**Not at Probe 1.** No prioritized replay; no curious replay; no multi-environment training; no curriculum.

### 2.10 Mirror caller — `kind/mirror/caller.py`

**Files.** `kind/mirror/caller.py`.

**Public interface.**

```python
class MirrorCaller:
    def __init__(self, model: str, max_tokens: int) -> None: ...
    def read_recent(self, telemetry_dir: Path, n_episodes: int) -> MirrorReading: ...
```

Single Anthropic API call (model from config, default `claude-sonnet-4-6`). Reads the most recent N episodes' `agent_step` records via `TelemetryView` (i.e., it has access to posteriors/priors/KL/recon, not to anything the actor doesn't see — the mirror sees *more*, not less). Returns a structured Pydantic `MirrorReading` (a free-text summary plus a few flagged fields).

**Reads.** Parquet shards under `runs/{run_id}/telemetry/agent_step/`.

**Writes.** A JSONL log of mirror readings under `runs/{run_id}/mirror/`.

**Tests.** Not in the five gating tests. Smoke verifies the mirror is callable but does not assert on its content.

**Not at Probe 1.** No adversarial structure (Probe 2); no frozen criteria (Probe 2); no in-loop prompting; no tool use; no `world_event` emission with `event_type=mirror_marker` (the mirror does not write to the world-event stream at Probe 1; the schema reserves the slot for Probe 2).

### 2.11 Observer / journal — `docs/workingjournal/probe1.md`

**Files.** `docs/workingjournal/probe1.md` (Markdown, hand-written); `kind/observer/eyeball.py` (small helpers — pretty-print a Parquet shard, count `world_event` types, dump the most recent N `agent_step` rows).

**Public interface.** No code surface to speak of. The journal is a Markdown document; the eyeball helpers are CLI scripts.

**Reads.** Telemetry sinks.

**Writes.** Stdout (eyeballing); the journal is hand-written.

**Tests.** None — the journal is not under test.

**Not at Probe 1.** No visualization beyond eyeballing JSONL/Parquet; no dashboards; no plots.

---

## 3. Schemas as first-class artifacts

Pydantic v2 models in `kind/observer/schemas.py`, exported as JSON Schema in `schemas/v0.1.0.json`. The four schemas, with field lists drawn directly from the Probe 1 synthesis §Q3:

### 3.1 `AgentStep`

| Field | Type | Notes |
|---|---|---|
| `schema_version` | `str` | semver, `"0.1.0"` at Probe 1 |
| `run_id` | `str` | UUID per run |
| `checkpoint_id` | `str \| None` | last checkpoint at write time |
| `t` | `int` | global env step |
| `episode_id` | `int` | |
| `step_in_episode` | `int` | |
| `wallclock_ms` | `int` | per record, for Probe 3 builder-coupling |
| `h_t` | `list[float]` | deterministic recurrent state |
| `q_params_t` | `tuple[list[float], list[float]]` | posterior μ, log-σ |
| `p_params_t` | `tuple[list[float], list[float]]` | prior μ, log-σ |
| `z_t` | `list[float]` | sampled posterior latent |
| `kl_per_dim_t` | `list[float]` | per-dim KL (Probe 4 distinguishability often concentrates) |
| `kl_aggregate_t` | `float` | sum |
| `recon_loss_t` | `float` | aggregate; per-modality fields reserved |
| `action_t` | `int` | discrete action index |
| `action_logprob_t` | `float` | |
| `policy_entropy_t` | `float` | |
| `obs_hash_t` | `str` | content hash, not raw obs |
| `intrinsic_signal_t` | `float` | ensemble disagreement variance |
| `encoder_embedding_t` | `list[float]` | |

### 3.2 `DreamRollout`

| Field | Type | Notes |
|---|---|---|
| envelope (3 fields) | | as above |
| `seed_step` | `int` | |
| `seed_h0` | `list[float]` | |
| `seed_z0` | `list[float]` | |
| `sequence_h` | `list[list[float]]` | length H |
| `sequence_z_prior` | `list[list[float]]` | length H |
| `sequence_action` | `list[int]` | |
| `sequence_action_logprob` | `list[float]` | |
| `sequence_prior_entropy` | `list[float]` | |
| `sequence_decoded_obs` | `list[bytes] \| None` | decoded observation tensors at fixed cadence |
| `cumulative_prior_entropy` | `float` | |
| `mean_step_kl_successive_priors` | `float` | |
| `max_step_latent_norm_change` | `float` | |

### 3.3 `ReplayMeta`

| Field | Type | Notes |
|---|---|---|
| envelope (3 fields) | | |
| `event_type` | `Literal["insert", "sample", "evict"]` | |
| `t_event` | `int` | |
| `segment_id` | `int` | |
| `segment_start` | `int` | |
| `segment_end` | `int` | |
| `priority` | `float \| None` | nullable; reserved for Probe 3+ |
| `buffer_size` | `int` | |
| `total_segments` | `int` | |

### 3.4 `WorldEvent`

| Field | Type | Notes |
|---|---|---|
| envelope (3 fields) | | |
| `t_event` | `int` | env step at event |
| `event_type` | `Literal["builder_perturbation", "env_reset", "internal_stochasticity_aggregate", "mirror_marker"]` | |
| `source` | `Literal["builder", "environment", "system"]` | |
| `payload` | `dict[str, Any]` | mutator type, cell coords, pre/post local state, episode aggregates, etc. |
| `wallclock_ms` | `int` | |

**Source / event-type pairs.** Each event type has a single canonical source:

| `event_type` | `source` | Emitted by | Probe 1 status |
|---|---|---|---|
| `builder_perturbation` | `builder` | env-server, on mutator invocation | wired; perturbation-free or near-zero-rate at Probe 1 |
| `env_reset` | `environment` | env-server, on episode boundary | wired and emitted every 200 steps |
| `internal_stochasticity_aggregate` | `environment` | env-server, per-episode rollup | wired; per-event logging deferred to Probe 4 |
| `mirror_marker` | `system` | mirror process | **reserved for Probe 2 — no component emits these at Probe 1** |

`mirror_marker` is in the schema so Probe 2's mirror can mark its own readings against the telemetry timeline without a schema bump. Its absence at Probe 1 is intentional, not an oversight.

**Payload flexibility.** `payload: dict[str, Any]` is intentionally schemaless within itself. The same `WorldEvent` model carries both per-event records (Probe 4 will use this for individual builder mutators) and per-episode aggregates (Probe 1 uses this for `internal_stochasticity_aggregate` rollups). Adding new payload shapes for future probes does not bump the schema version; the envelope and field set stay stable while the inner payload structure evolves. This makes the forward-compatibility intentional rather than incidental.

**Versioning.** `SCHEMA_VERSION = "0.1.0"` is checked into `kind/observer/schemas.py`. Field additions in later probes bump the patch or minor and remain backward-readable; deprecations are marked, never deleted. The frozen JSON Schema export (`schemas/v0.1.0.json`) is committed alongside the Python models so external readers (Probe 2's mirror, future analysis scripts) have a versioned source of truth that does not require importing Kind.

---

## 4. Test scaffolding — the five tests

The Probe 1 synthesis named exactly five tests as the gate. No more at the gate; additional tests are deferred to later probes.

| # | Name | File | What it checks | Fixtures |
|---|---|---|---|---|
| 1 | env step shape | `tests/test_env_step.py` | `GridWorld.reset()` returns observation of shape `(32, 32)`; `GridWorld.step(a)` returns observation + `EnvStep` with valid fields; episode resets cleanly at 200 steps; deterministic from seed. | `grid_world_factory` (fresh env per test) |
| 2 | agent forward | `tests/test_agent_forward.py` | `WorldModel.step(obs, h_prev, z_prev, a_prev)` returns a `WorldModelStep` with correct dtypes and shapes; `Actor` consumes the resulting `PolicyView` and produces a valid action with finite logprob; ensemble disagreement is finite. | `dummy_obs`, `dummy_prior_state` |
| 3 | perturbation hook logged | `tests/test_perturbation_hook.py` | Calling `env_server.add_resource((3, 4))` causes a `WorldEvent` record to land in the `world_event` JSONL with `source="builder"`, the correct cell coords, and pre/post local state; the agent's observation contains no marker that distinguishes the cell from a regrowth event. | `env_server_with_sink` |
| 4 | JSONL roundtrip | `tests/test_jsonl_roundtrip.py` | Writing 100 `AgentStep` records via the Parquet sink and 100 `WorldEvent` records via the JSONL sink; reading them back via Pyarrow and Pydantic; equality after schema validation; `schema_version` round-trips. | `tmp_path` |
| 5 | integration smoke | `tests/test_integration_smoke.py` | Run the full loop on CPU for 200 env steps with a tiny world model (h=32, z=4, K=2), one mid-run checkpoint barrier, no perturbations. Asserts: all four streams have records; checkpoint directory has all required files; resume from checkpoint yields identical RNG state. | `tiny_runner` |

The integration smoke at gate-time runs on **CPU** for speed and determinism; the **MPS smoke test** (§5) runs separately on the target platform with the actual Probe 1 sizes and is the platform-correctness gate.

Test layout convention: one file per test, no shared module state, fixtures in `tests/conftest.py`. Pytest discovery covers everything in `tests/`.

---

## 5. Day-one smoke test

The synthesis flagged that **MPS performance for RSSM operations at Probe 1 batch sizes is unbenchmarked**. The day-one smoke test runs on the target platform (Mac, MPS), with the actual Probe 1 sizes, and is what tells us whether the substrate decision is operationally tractable on the canonical machine.

**File.** `scripts/smoke_mps.py`.

**What it does.**

1. Construct a `WorldModel` with default Probe 1 sizes (h=200, z=16, encoder/decoder for 32×32 grayscale).
2. Construct a `LatentDisagreementEnsemble` with K=5.
3. Generate dummy observation tensors on MPS (no env-server; this isolates compute from I/O).
4. Run 100 RSSM training steps (forward + backward + optimizer step) at batch size 16 with sequence length 32. Log wall time per step and total wall time.
5. Inspect the warning stream for `PYTORCH_ENABLE_MPS_FALLBACK` warnings; assert none on the hot path. (CPU fallbacks during init are tolerated; fallbacks during training-step ops are not.)
6. Open all four telemetry sinks; write a synthetic record to each; verify the records read back valid.
7. Print a one-line summary: `[smoke] mps ok | wall=Xs | per-step=Yms | sinks ok | shapes ok`.

**What passes.** No MPS-fallback warnings on the hot path; total wall time finite and below a soft bar (default: 60s for 100 steps — order-of-magnitude check, not a benchmark); all four telemetry streams write valid records; world-model forward produces finite KL and finite recon loss.

**What fails.** Any MPS-fallback warning during the training loop; any NaN/Inf in KL or recon; any sink that fails to write a valid record. Failure means: the substrate decision is fine but the implementation has a platform-specific gap. Fix it before going further; do not work around with broad `PYTORCH_ENABLE_MPS_FALLBACK=1`.

**Fallback semantics if MPS is too slow.** The synthesis specifies the documented Probe-2-or-later fallback: train on desktop, sync weights to Mac at checkpoint boundaries with explicit sync semantics. **Probe 1 does not attempt this.** Probe 1 is testing canonical-on-Mac. If MPS is too slow at Probe 1 sizes, the implementer drops to smaller sizes (h=128, z=8) and re-runs the smoke; if that still fails, Probe 1 stalls and the synthesis's open question on MPS becomes the next decision to make.

---

## 6. Open-during-build decisions with defaults

The two synthesis documents named several "open during build" questions. The plan acknowledges these as decisions to be made during the build with sensible defaults, not as decisions that block the build. Defaults the implementer should start with:

| Question | Default | Revisit when |
|---|---|---|
| Latent dim `z_t` | **16** | smoke shows posterior collapses (KL pinned at free-bits floor) → try 32, or switch to categorical |
| Recurrent dim `h_t` | **200** | OOM on MPS → 128 |
| Free bits per dim | **1.0 nat** | KL pinned at floor through 5k+ steps → lower to 0.5 |
| Ensemble size K | **5** | disagreement saturates within an episode → try 3 (cheaper) |
| Replay sequence length | **32** | dream rollouts feel disconnected from observed trajectories → 16 or 64 |
| Dream rollout horizon H | **15** | imagined latents diverge by step 5 → shorten |
| Dream rollout cadence | **1 per 1000 env steps** | mirror complains there are no dreams to read → more frequent |
| Pixel resolution | **32×32 grayscale** | decoded dreams unreadable → 32×32 RGB |
| Resource regrowth `p` | **0.01 per empty cell per step** | regrowth events <1 per episode → 0.02 |
| Drift magnitude | **±10% over 50 episodes (random walk)** | drift dominates per-step KL → halve |
| Episode length | **200 steps** (settled) | — |
| Grid size | **8×8** (settled) | smoke shows posterior collapses on the substrate alone → 9×9 with movable object |

These defaults are starting points. The smoke test informs the first revision. None of them is a decision the plan asks the user to make before code is written.

**On stalls.** If smoke-test stability does not converge after several rounds of revisiting these defaults, the synthesis's open path is to revisit the specific minimal-RSSM variant (categorical latents, JEPA-style reconstruction-free predictor) — *not* the substrate decision, which is settled. The arbiter is convergence, not calendar. The journal records what surfaced across the rounds, and that record is what informs the next decision.

---

## 7. PolicyView / TelemetryView code-level enforcement

The synthesis settled that Io's actor reads `concat(h_t, z_t)` and nothing else, while the mirror reads everything. This is enforced at three levels in code:

**Level 1 — module boundaries.** `kind/agents/views.py` exports both classes. `kind/agents/actor.py` imports only `PolicyView`. `kind/observer/sinks.py` and `kind/mirror/caller.py` import only `TelemetryView`. A dependency lint (a small script in `tests/test_view_isolation.py`) fails CI if `kind/agents/actor.py` imports `TelemetryView`.

**Level 2 — type signatures.** `Actor.forward(self, view: PolicyView) -> ActionOutput` is the only entry point. mypy is run in `--strict` mode on `kind/agents/`; passing a `TelemetryView` where a `PolicyView` is expected fails type-checking.

**Level 3 — frozen dataclasses.** Both views are `@dataclass(frozen=True)`. The actor cannot mutate state in place. The intrinsic signal arrives in the actor's training loop as a scalar argument from outside the actor module — it is not a field on `PolicyView` and not introspectable from inside the actor.

**The world-model module's forward pass.** `WorldModel.step(...)` returns `WorldModelStep`, an internal container with everything. The `views.split(step, intrinsic)` function is the *only* place where the split happens. Two consumers downstream:

- The actor, which receives the `PolicyView`.
- The telemetry/mirror path, which receives the `TelemetryView`.

The function signature of `split` returns the tuple in a specific order; downstream code unpacks by position and the static analyzer enforces that the actor only ever holds a `PolicyView` reference.

**What this prevents.**

- The actor cannot read `q_params_t` or `p_params_t` (no posterior-prior KL routed into reward by accident).
- The actor cannot read `recon_loss_t`, `kl_per_dim_t`, or `intrinsic_signal_t` as fields (the intrinsic signal *value* enters the actor's training loss as a scalar argument, not as an introspectable attribute).
- Future refactors that add fields to `TelemetryView` do not silently widen what the actor sees.

**What this does not pretend to prevent.** A determined refactor can always break opacity. The boundary is structural-by-default, not adversarial. Probe 1 builds the discipline; Probe 2+ may move the actor into a separate process if subprocess isolation becomes warranted (the interfaces are designed to allow it).

---

## 8. Mac/desktop deployment plan

**Canonical state lives on the Mac.** Weights, replay buffer, optimizer state, RNG state, telemetry shards. The Mac is the trainer.

**Desktop runs the env-server only.** A single Python process exposes the `GridWorld` over TCP. No model compute runs on the desktop at Probe 1.

**Network protocol.** Length-prefixed framed messages over TCP socket. Default port 5555 (configurable). Encoding: compact JSON with base64-encoded NumPy serialization for the observation tensor. msgpack with NumPy support is a documented optimization to consider only if the JSON path becomes a measured bottleneck — not expected at Probe 1 sizes. Three message types:

| Type | Direction | Payload |
|---|---|---|
| `STEP` | Mac → desktop | `{action: int, request_id: int}` |
| `TRANSITION` | desktop → Mac | `{observation: ndarray, env_step: int, episode_id: int, step_in_episode: int, wallclock_ms: int, request_id: int}` |
| `BARRIER_BEGIN` / `BARRIER_END` | Mac → desktop | `{checkpoint_id: str}` |

**Checkpoint barrier protocol.** Already detailed in §2.8. Five steps; atomic rename on commit; resume from checkpoint is exact.

**Telemetry path.** `agent_step` and `dream_rollout` are written by the Mac trainer (they are produced there). `world_event` is written by the env-server *but to the Mac's filesystem* — the env-server's connection to the Mac includes a side channel (a second TCP connection or a multiplexed message type) for telemetry. The synthesis specifies the Mac owns telemetry; the desktop is stateless except for the env's RNG state, which is part of the canonical checkpoint.

**Probe 1 runs both ends locally on the Mac for the smoke.** The TCP protocol is wired and tested locally; the Mac/desktop physical split is a deployment detail Probe 1 validates by running over loopback. Stress-testing the actual two-machine setup with desktop power events is deferred to Probe 3 (which is when the four-state model's transitions actually need to work end-to-end).

**What runs where (Probe 1 deployment).**

| Process | Machine | Notes |
|---|---|---|
| Trainer (world model + actor + replay + checkpoint manager) | Mac | MPS hot path |
| Env-server | desktop *or* Mac (loopback) at Probe 1 | Probe 1 default: loopback for the smoke |
| Telemetry sinks | Mac | Owned by the Mac filesystem |
| Mirror caller | Mac | Anthropic API calls |
| Journal | Mac | Hand-edited |

---

## 9. Out of scope at Probe 1

Explicit list. The plan does not build for Probes 2, 3, or 4 except where forward-compatibility was authorized in the synthesis decisions (telemetry schema, substrate conduits, harness mutators, opacity boundary).

- **No dream rollouts driving behavior.** Dreams emit telemetry; nothing trains on them.
- **No mirror calibration logic.** Single LLM call, no frozen criteria, no adversarial check (Probe 2).
- **No perturbation rate tuning, distinguishability metrics, or BOCPD/PELT changepoint detection.** Probe 4 owns this.
- **No four-state operational machinery.** Only waking is exercised. Dreaming/dormant/paused transitions are not implemented (Probe 3).
- **No multi-environment training.** One env-server, one trainer, one Io.
- **No hyperparameter sweeps.** Defaults from §6; smoke informs first revision; no grid search.
- **No visualization beyond eyeballing JSONL.** No dashboards, no tensorboard, no plots. The journal is hand-written.
- **No PPO fallback wired up in code.** Documented in comments as the documented fallback if the analytic-gradient actor proves unstable; not built.
- **No JEPA-style reconstruction-free predictor.** Decoder kept (mirror legibility).
- **No categorical latents.** Continuous Gaussian only.
- **No DreamerV3 robustness machinery.** Symlog, twohot, percentile normalization, KL balancing, unimix — none.
- **No subprocess isolation between actor and world model.** In-process; interfaces designed to allow it later.
- **No introspective head whose target is `h_t`.** The architectural decision rules this out as installed self-modeling.
- **No `introduce_novel_object` mutator.** Excluded by environment synthesis on no-marker grounds.
- **No day-night cycle, no installed periodicity.** A-temporal in periodicity.
- **No content-addressed checkpoint store, no DVC, no rsync between machines** for the smoke. Local atomic rename only.
- **No fifth telemetry stream for training-step records.** Training-step diagnostics live as fields on `agent_step`.
- **No prioritized or curious replay.** FIFO sequence segments.

The rule of thumb: if a Probe 2/3/4 feature can be added later without changing the schemas, the substrate conduits, the harness mutators, or the opacity boundary, it is out of scope here.

---

## 10. Connection to the journal

Probe 1 ends with a journal entry recording what was learned, what surprised, what is now decided. The journal lives at `docs/workingjournal/probe1.md` and is hand-written by the builder, not by the system.

**First entry covers (template — actual content is the builder's):**

- What ran end-to-end. The smoke test result; the wall time per training step on MPS; the four streams' first records pasted in.
- What surprised. Anything that did not behave as the synthesis predicted — KL collapsing despite free bits, ensemble disagreement saturating earlier than expected, decoder producing illegible dreams, a mutator hook misfiring.
- What is now decided that was open. Specific defaults from §6 that the smoke promoted from "default" to "settled at this scale."
- What is now newly open. Things the build revealed that the synthesis did not anticipate. These become Probe 2's starting context.
- A short note on whether the substrate decision held up operationally on the canonical machine.

The journal is the bridge between Probe 1 and Probe 2's research-prompt drafting. Probe 2 begins by reading this entry.

---

*End of plan.*
