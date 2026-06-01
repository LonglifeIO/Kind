# Probe 3 — Implementation Plan

*Operational plan that translates the Probe 3 synthesis (`docs/decisions/synthesis_probe3_dream_foundational_2026-05-27.md`) into a concrete build sequence on top of the Probe 1 / 1.5 substrate as extended by Probe 2's mirror infrastructure (Phase 12.5 settled, 2026-05-22). The synthesis settled the integrated take and the load-bearing constraints — logs-only mirror output for first build; no gradient flow from dream; replay-seeded perturbed generative recombination with associative drift as a parametric regime; four-axis differentiation from waking planning; exogenous trigger; envelope-and-seed-selection-only Probe 4 interface; self-opacity-during-dream held open; dormant ≠ off held open. This plan does not relitigate those decisions. It specifies the build order, the components, the schemas, the interfaces, the tests, and the early visibility smoke that gates whether the dream regime drives non-degenerate variation before the downstream phases commit to assumptions. Forward-compatibility lives in the dream-rollout module's seed-selection and envelope control surfaces (the Probe 4 perturbation hooks) and in the DreamRollout schema's provenance/control fields; nothing else is built for Probe 4.*

*The synthesis settled what; this plan settles how. Where the plan chooses among options the synthesis left open at the implementation level (the exact temperature schedule shape, the dream-actor convention, whether to bump telemetry to `0.3.0`, the precise mirror-side dream-reading prompt shape), the choice is named as a plan-level design choice with a default that the smoke informs.*

---

## 1. Build order with dependency graph

Hard dependencies (must run sequentially):

- **Schemas before writers.** The Probe 3 DreamRollout extensions and the new `state_transition` / `dormant_heartbeat` `world_event` payload conventions exist before the dream-rollout module or the state-machine module emit records.
- **Replay seed selection before dream-rollout module.** The dream rollout consumes a seed (replay snippet, perturbed prior, or hybrid); seed selection must produce typed seed objects before the rollout module can be tested end-to-end.
- **Dream-rollout module before visibility smoke.** Phase 3 (the load-bearing early empirical test, modelled on Probe 2's Phase 12 ordering) gates the downstream phases. The smoke runs the new dream regime + a waking-planning control + a pure-prior control against an existing Probe 1.5 checkpoint and asserts non-degenerate variation in DreamRollout telemetry across the three. If the smoke fails (the dream regime collapses to either control), Phases 4–8 are not built until the dream regime is revised.
- **Visibility smoke before state-machine.** The state machine consumes the dream-rollout module; if the dream regime is degenerate, building the state machine commits scaffolding to a non-load-bearing rollout.
- **State machine before mirror integration.** The mirror reads state-typed telemetry; the state must be observable in the telemetry timeline before the mirror's dream-reading prompt is wired.
- **Hard caps / runtime protection before any end-to-end Probe 3 run.** Content-blind safety (caps, checkpoint windows, compute budgets, dormant transition) must be in place before any run that could produce decoherence is started.

Parallelizable (no hard dependency between them after Phase 0):

- Schema additions (Phase 0), the cross-probe-surface preparation (Phase 7), and the gate-test scaffold (Phase 8) can be drafted in parallel against the schema deltas.
- Within Phase 2, the seed-selection module, the temperature-schedule generator, and the ensemble-disagreement-recording extension are independent.

Phased build (each phase is a unit of work with a single question):

| Phase | Question being answered | Deliverable | Depends on |
|---|---|---|---|
| **0. Probe 3 schemas** | What fields does the dream regime need on `DreamRollout`, on `WorldEvent` payloads, and on the new `DreamSessionMeta` JSONL stream for the four-axis differentiation, provenance, runtime-protection, and state-machine telemetry to be visible? | `DreamRollout` extension (fourteen optional fields, listed §3.1; nine of them unconditionally required when `schema_version == "0.3.0"`); `WorldEvent` payload conventions for the two new event types `state_transition` and `dormant_heartbeat` (Literal extended); existing `builder_perturbation` remains the builder-side perturbation event type and is not newly added by Probe 3; new `DreamSessionMeta` model + JSONL sink at `kind/observer/dream_session.py`; schema version bump to `"0.3.0"` for telemetry; `schemas/v0.4.0.json` checked in. | — |
| **1. Replay seed selection** | How does a dream rollout get its initial conditions in a way that satisfies the four-axis differentiation's axis 4 (initial conditions from replay or perturbed prior, not current state), produces provenance the faithfulness verifier can resolve, and exposes a Probe-4-perturbable control surface? | `kind/training/dream_seed.py` with `select_seed(buffer, mode, rng, ...) -> DreamSeed`; the three seed modes (replay, perturbed_prior, hybrid) operationalized; provenance fields populated (episode + step refs for replay; perturbation magnitude for perturbed_prior; mixture composition for hybrid). | 0 |
| **2. Dream-rollout module (four-axis-differentiated)** | What runs during a dream phase such that the rollout is *operationally* distinct from a waking planning rollout on all four axes (goal-coupling absent; ensemble disagreement recorded-but-not-used; distinct temperature regime; replay/perturbed-prior seed)? | `kind/training/dream.py` exposing `DreamRollout` emission with the four-axis differentiation specified §2.3; replaces the Probe 1.5 `Runner._emit_dream` method (the old method is preserved as the waking-planning control for Phase 3); periodic re-seeding for chimera structure; tail temperature ramp for associative drift; gradient policy hard-pinned to `"none"`; ensemble disagreement computed and recorded but never read for action selection. | 0, 1 |
| **3. Visibility smoke (load-bearing early phase)** | Does the new dream regime drive non-degenerate variation in `DreamRollout` telemetry relative to (a) a waking-planning-with-env-off control (the preserved Probe 1.5 `_emit_dream` method) and (b) a pure-prior-rollout control (no actor, no seed perturbation, no temperature regime)? This is the Probe 1.5 Phase 7 lesson re-applied: schema slots are not enough; the conditioning surface must be non-zero. | `scripts/smoke_probe3_visibility.py` runs three rollout regimes against an existing Probe 1.5 checkpoint (`runs/probe1_5_phase7_5-20260507-101800/`, or the latest available) and reports per-field KS-D between the three. Pass criterion: dream regime's distributional distance to both controls exceeds a configured threshold (default KS-D ≥ 0.15 on at least three of {`mean_step_kl_successive_priors`, `cumulative_prior_entropy`, `max_step_latent_norm_change`, `sequence_ensemble_disagreement_variance` aggregate, decoded-obs entropy aggregate}). Fail mode: dream collapses to one or both controls — revise the dream regime before Phase 4. | 2 |
| **4. State machine (waking / dreaming / dormant / paused)** | What state-machine shape lets the four states be operationally distinguishable in telemetry, with exogenous (desktop on/off) primary trigger, hard-cap *interfaces* as content-blind exit conditions, and dormant ≠ off as a named but not yet fully designed state? | `kind/training/state_machine.py` with `StateController` exposing `current_state`, `tick(host_signals) -> StateTransition`, and the state/transition table listed §2.4; `DesktopWatcher` (or interface) supplying host-signal events; state transitions emit `world_event` records via the existing sink; dormant minimum behavior (operationally distinguishable per §2.5; deeper design deferred). Phase 4 defines the protection hooks and transition semantics; Phase 6 implements and verifies the protection policies against those hooks. | 3 |
| **5. Mirror integration with dream telemetry** | How does Probe 2's mirror read dream-state telemetry without violating the one-way constraint (no runtime authority, no scheduler control, no Io-readable channel), while producing readings the builder can audit? | `kind/mirror/dream_reading.py` — a new state-typed reading pass for `state == "dreaming"` digests; logs-only output (no `world_event` writes besides `mirror_marker` of the existing `system` kind); per-surface prompt extensions to `criteria_v2.py` (dream-state evidential weight: substrate-side reads the dream regime's distributional properties; head-internal disabled for first build since the head does not run during dream per Probe 1.5 v2 §1.5; behavior-side disabled for first build since the actor's policy is not committed during dream per axis 1). Mirror reads `DreamRollout` records + `state_transition` events + the new `DreamSessionMeta` JSONL; mirror writes its reading to `runs/{run_id}/mirror/dream_readings.jsonl` only. | 0, 2, 4 |
| **6. Content-blind runtime protection** | What protective machinery does the runtime own (vs the mirror), and how does it enforce the synthesis's "incoherent dreams remain observable, mirror does not silently shape the dream-length distribution" commitment? | Implementation + verification pass over the Phase 4 protection hooks in `state_machine.py`: hard cap on dream-session wall-clock duration; hard cap on within-session dream-rollout count; checkpoint-window-based forced transition to dormant; rolling one-hour compute-budget cap. All four are content-blind: they consume `wallclock_ms`, env-step counts, rollout counts, compute ledger entries, and checkpoint cadence — never `DreamRollout` content, never mirror readings. Mirror's `uncertain` rulings default to `continue_and_log_uncertainty` (the runtime does *not* read mirror verdicts to schedule). | 4 |
| **7. Cross-probe surface (Probe 4 envelope and seed-selection hooks)** | What control surfaces does Probe 3 expose for Probe 4 to perturb, without exposing hidden-state-write affordances? | A typed `DreamEnvelopeConfig` and `SeedSelectionConfig` on `RunnerConfig`, plus a minimal builder-perturbation entry point (`scripts/perturb_dream_envelope.py`, using the already-existing `builder_perturbation` `WorldEventType`) that can modify these configs at checkpoint boundaries for Phase 7's end-to-end surface test. No hidden-state-write affordance is exposed; no `world_event` payload field carries a state-write request. The configs are content-blind from Io's side (the actor and world model do not read them). The script is deliberately thin and may be replaced when Probe 4 designs its actual perturbation workflow. | 4, 6 |
| **8. Gate tests + integration smoke + journal scaffold** | Does the assembled Probe 3 substrate pass the four named gate tests plus the integration smoke without regressing Probe 1.5 / Probe 2 invariants? | The eight named tests §4; the integration smoke covering one full waking→dreaming→dormant→waking cycle on the canonical machine; journal scaffolding at `docs/workingjournal/probe3.md` mirroring Probe 2's structure (phase headings, "what's closed / what's newly open" rhythm). | All |

Phase 0 is the single load-bearing prerequisite. Phase 3 is the early-empirical-finding phase whose result gates Phases 4–8 (modelled on Probe 2's Phase 12-before-9-11 decision). The plan does not pre-commit to building Phases 4–8 if Phase 3 finds the dream regime degenerate; the journal records the find and the revision before downstream work continues.

---

## 2. Per-component specifications

Each entry: file paths, public interfaces, what the component reads/writes, what tests verify it, what it does *not* do at Probe 3.

### 2.1 Schema additions — `kind/observer/schemas.py`, `kind/observer/dream_session.py`

**Files.** `kind/observer/schemas.py` (existing — extended; SCHEMA_VERSION bumps to `"0.3.0"`); `kind/observer/dream_session.py` (new — `DreamSessionMeta` model + JSONL sink); `schemas/v0.4.0.json` (new frozen JSON Schema export covering Probe 1.5's `0.2.0` telemetry plus Probe 3's `0.3.0` `DreamRollout` plus the new `DreamSessionMeta`).

**`DreamRollout` extensions (fourteen optional fields, full table in §3.1).** All fourteen are declared `Optional[T] | None = None` so that Probe 1 (`"0.1.0"`) and Probe 1.5 (`"0.2.0"`) `DreamRollout` records remain readable against the new model — the absence of the new fields becomes `None`. A `model_validator` on `DreamRollout` enforces that any record stamped `schema_version == "0.3.0"` populates the nine *unconditionally required* fields with non-None values; the remaining five fields are not unconditional — three are *conditional* (the seed-provenance fields, required only when `seed_kind` triggers them) and two are *genuinely optional* (`sub_mode_tags`, and `re_seed_step_indices` which is None if no re-seeding occurred this dream).

**`WorldEvent` literal extensions.** `WorldEventType` gains two new values: `"state_transition"` and `"dormant_heartbeat"`. `source` literal gains no new values (`"environment"` covers both — state transitions are environment-driven per the synthesis's exogenous-trigger commitment). Payload conventions:

- `state_transition` payload: `{from_state, to_state, dream_session_id, trigger, wallclock_ms_in_prev_state, env_step_at_transition}` where `from_state` and `to_state` are `Literal["waking", "dreaming", "dormant", "paused"]` and `trigger` is `Literal["desktop_off", "desktop_on", "mac_off", "mac_on", "hard_cap_wallclock", "hard_cap_rollout_count", "checkpoint_window", "compute_budget"]`.
- `dormant_heartbeat` payload: `{dormant_started_at_ms, dormant_wallclock_ms_elapsed, mac_alive: true}` — a periodic ping (default every 60s of dormant wallclock) so the absence-of-dreaming is observable as a positive signal in the telemetry timeline rather than as silent gaps.

**`DreamSessionMeta` (new model, new JSONL stream).** One record per dream *session* (from `waking → dreaming` transition to next transition out of dreaming). Distinct grain from `DreamRollout` (which is per imagination horizon, ~1 per 1k env steps or as configured; many rollouts can fire within a session).

```python
class DreamSessionMeta(RecordEnvelope):
    schema_version: str
    run_id: str
    checkpoint_id: str | None

    dream_session_id: str                      # UUID
    started_at_env_step: int
    started_at_wallclock_ms: int
    ended_at_env_step: int | None              # None while session in flight
    ended_at_wallclock_ms: int | None
    end_trigger: Literal["desktop_on", "hard_cap_wallclock", "hard_cap_rollout_count",
                         "checkpoint_window", "compute_budget", "mac_off", None]
    rollout_count: int                         # populated at session end
    envelope_config_snapshot: dict[str, Any]   # the DreamEnvelopeConfig at session start
    seed_selection_config_snapshot: dict[str, Any]   # the SeedSelectionConfig at session start
```

The session record is written twice — once at session start with `ended_*` fields `None`, and again at session end with all fields populated — into an append-only JSONL stream at `runs/{run_id}/telemetry/dream_session.jsonl`. The double-write is the canonical pattern: the start record is the marker the mirror can find ASAP for in-flight dream observation; the end record is the closure the faithfulness verifier resolves against.

**Reads / Writes.** Pure declarations + one new JSONL sink. Telemetry sink wiring is identical to existing `replay_meta` / `world_event` JSONL patterns; reuse `JsonlSink`.

**Tests (6).**
- `DreamRollout` round-trip with all fourteen new fields populated; `schema_version == "0.3.0"`.
- `DreamRollout` from Probe 1.5 (`"0.2.0"`) reads cleanly with all fourteen new fields = `None`.
- The `_enforce_v3_required_fields` validator rejects a `"0.3.0"` record with `seed_kind == None`.
- `WorldEvent` with `event_type == "state_transition"` and the payload convention round-trips.
- `DreamSessionMeta` round-trip; the start-then-end double-write pattern (two records, same `dream_session_id`) reads back correctly.
- `schemas/v0.4.0.json` export is byte-stable.

**Not at Probe 3.** No migration of Probe 1/1.5 `DreamRollout` records (they remain `"0.1.0"` / `"0.2.0"`-versioned in their original runs). No payload changes to `mirror_marker`. No new `source` literal values.

### 2.2 Replay seed selection — `kind/training/dream_seed.py`

*Amended 2026-05-27 per `docs/decisions/phase0_5_replay_seed_source_2026-05-27.md`. Replay-mode `(h_init, z_init)` are produced by re-encoding a short obs/action window read from the existing `SequenceReplayBuffer`; the buffer's storage shape is unchanged. The pre-amendment spec assumed historical `(h, z)` were stored in the buffer alongside obs — they are not (`Transition` in `kind/training/replay.py:65–79` carries only `obs`, `action`, `next_obs`, `env_step`, `episode_id`, `step_in_episode`). The amendment changes `select_seed`'s replay-mode semantics, adds one `SeedSelectionConfig` field (`replay_warmup_length`), reinterprets two existing DreamRollout provenance fields without bumping the schema, and rewords Phase 1 test 1 as a deterministic re-encoding test. Synthesis §3 axis 4 ("initial conditions from replay or perturbed prior, not current state") is preserved; the broader plan is untouched.*

**Files.** `kind/training/dream_seed.py` (new).

**Public interface.**

```python
@dataclass(frozen=True)
class DreamSeed:
    mode: Literal["replay", "perturbed_prior", "hybrid"]
    h_init: Tensor                                # shape (1, h_dim)
    z_init: Tensor                                # shape (1, z_dim)

    # Provenance — populated per mode.
    # For replay/hybrid, under the amended Option-1 semantics:
    #   replay_segment_id = env_step of the start of the obs/action warmup window
    #   replay_step_offset = warmup-window-local offset of the conditioning start
    #     (first build always 0; the seed reads the entire window from its start)
    # The warmup length itself is recorded in DreamRollout.sampling_parameters
    # under the key "replay_warmup_length".
    replay_segment_id: int | None
    replay_step_offset: int | None
    perturbation_magnitude: float | None          # L2 norm of the perturbation applied to the prior
    hybrid_mixture_alpha: float | None            # 0..1; alpha=1.0 is pure replay; alpha=0.0 is pure perturbed_prior

    rng_seed: int                                  # per-seed RNG state for reproducibility

@dataclass(frozen=True)
class SeedSelectionConfig:
    mode: Literal["replay", "perturbed_prior", "hybrid"] = "replay"
    perturbation_sigma: float = 0.1                # for perturbed_prior and hybrid
    hybrid_alpha_distribution: Literal["uniform_0_1", "fixed_0_5", "beta_2_2"] = "uniform_0_1"
    replay_min_segment_age_steps: int = 1000       # exclude very recent segments (associative-drift hygiene)
    replay_warmup_length: int = 8                  # # of obs/action steps re-encoded from a zero (h, z, a)
                                                   # start to develop the seed; default per the Phase 0.5
                                                   # decision; Phase 3 smoke informs revision

def select_seed(
    replay_buffer: SequenceReplayBuffer,
    world_model: WorldModel,                       # required: produces (h, z) from the warmup window
    config: SeedSelectionConfig,
    rng: torch.Generator,
    *,
    perturbed_prior_anchor: Tensor | None = None,  # used when mode == "perturbed_prior" with no replay step;
                                                   # for hybrid, the replay re-encoded (h, z) is the anchor
                                                   # and this argument is ignored
) -> DreamSeed: ...
```

**Replay-mode operational semantics.** `select_seed(mode="replay")`:

1. Sample a valid contiguous window of length `replay_warmup_length` from `replay_buffer` (uniform over valid window starts, respecting `replay_min_segment_age_steps` and episode boundaries — reuse the buffer's existing window-validity logic). The window is identified by its start transition's `env_step`.
2. Read the obs and action sequences from that window. Phase 1 chooses the buffer-access shape: either extend `SequenceReplayBuffer` with a `get_window(start_env_step, length) -> tuple[Tensor, Tensor]` helper, or have `select_seed` consume an already-sampled `Batch`. Either is consistent with this spec.
3. Run the world model forward from `(h=0, z=0, a=0)` over the `replay_warmup_length` steps, using the posterior `q(z | h, obs)` and the recurrence `h_{t+1} = GRU(h_t, z_t, a_t)`. `rng` drives the posterior sampling. The final `(h_final, z_final)` after the warmup is the seed.
4. Populate `DreamSeed.replay_segment_id = start_env_step`, `DreamSeed.replay_step_offset = 0`. `rng_seed` records the RNG state used so the seed is reproducible against a fixed checkpoint and a fixed buffer state.

**Hybrid-mode operational semantics.** `select_seed(mode="hybrid")` first runs the replay-mode procedure above to obtain `(h_replay, z_replay)`, then perturbs that anchor by a `perturbation_sigma`-scaled Gaussian to obtain `(h_perturbed, z_perturbed)`, then returns the convex combination under `hybrid_mixture_alpha` (drawn from `hybrid_alpha_distribution`). The `perturbed_prior_anchor` argument is ignored in hybrid mode — the anchor is the re-encoded replay endpoint.

**Reads.** Replay buffer; world model (forward, no backward); RNG; optional anchor for pure perturbed_prior mode.

**Writes.** Nothing — produces a `DreamSeed` object the dream-rollout module consumes. `replay_warmup_length` is propagated into the emitted `DreamRollout.sampling_parameters` dict at the writer (§2.3 / Phase 2) under the key `"replay_warmup_length"`.

**Tests (3).**
- *Replay-mode determinism and re-encoding faithfulness.* Given a fixed `WorldModel` (fixed weights), a populated `SequenceReplayBuffer`, a fixed `torch.Generator` state, and a `SeedSelectionConfig` with `mode="replay"` and a configured `replay_warmup_length`, two calls to `select_seed` from the same RNG state produce byte-identical `DreamSeed.h_init` / `DreamSeed.z_init`. Additionally, re-feeding the obs/action window identified by `(replay_segment_id, replay_step_offset, replay_warmup_length)` through the same world model from a zero `(h, z, a)` start under the same RNG reproduces the same `(h_init, z_init)` byte-for-byte. `replay_min_segment_age_steps` is honored (no segment whose start `env_step` is within that many steps of the buffer's latest insert is sampled).
- Perturbed-prior mode: with `perturbation_sigma=0.1` and a known anchor, `||DreamSeed.h_init - anchor_h||` is within ±20% of `0.1 * sqrt(h_dim)` (the expected L2 of a sigma=0.1 Gaussian perturbation on `h_dim` dimensions).
- Hybrid mode: with `hybrid_alpha_distribution="fixed_0_5"`, the returned seed equals `0.5 * (h_replay, z_replay) + 0.5 * (h_perturbed, z_perturbed)`, where `(h_replay, z_replay)` is the replay-mode re-encoding result and `(h_perturbed, z_perturbed)` is the same anchor with the perturbation applied.

**Not at Probe 3.** No prioritized replay sampling (FIFO segment uniform sampling is the Probe 3 default); no learned seed selection; no mirror-driven seed selection (that would couple mirror to dream content, which the synthesis disallows for first build); no storage of historical `(h, z)` in the replay buffer (Option 2, rejected by `docs/decisions/phase0_5_replay_seed_source_2026-05-27.md`).

### 2.3 Dream-rollout module (four-axis-differentiated) — `kind/training/dream.py`

**Files.** `kind/training/dream.py` (new); `kind/training/runner.py` (existing — `Runner._emit_dream` is renamed to `_emit_waking_planning_rollout_for_phase3_control` and preserved unchanged as the Phase 3 visibility-smoke control; a new `Runner._emit_dream` delegates to `kind/training/dream.py`).

**Public interface.**

```python
@dataclass(frozen=True)
class DreamRolloutConfig:
    horizon: int = 30                                       # dream rollouts are longer than waking planning (default 15) by design
    temperature_mode: Literal["scheduled", "identity"] = "scheduled"
                                                            # off-switch for Phase 3 controls: "identity" forces multiplier=1.0 every step
    prior_temperature_schedule: TempSchedule = TempSchedule(
        head_value=1.5, tail_value=2.5, ramp_start_fraction=0.6
    )                                                       # T=1.5 → 2.5 in the tail 40% of the rollout; ignored if temperature_mode == "identity"
    re_seed_every_n_steps: int = 10                        # periodic re-seed for chimera structure; 0 disables
    re_seed_mode: Literal["resample_seed", "perturb_in_place"] = "resample_seed"
    action_policy: Literal["uniform_random", "temperature_modified_actor"] = "uniform_random"
    actor_action_temperature: float = 1.5                   # used only when action_policy == "temperature_modified_actor";
                                                            # *independent* of prior_temperature_schedule (see flag in §2.3 axis-1/axis-3 note)
    record_ensemble_disagreement: bool = True
    seed_strategy_for_control: Literal["normal", "prior_only"] = "normal"
                                                            # off-switch for Phase 3 pure-prior control: "prior_only" ignores the supplied
                                                            # DreamSeed and samples (h_init, z_init) from the world model's prior directly
    gradient_policy: Literal["none"] = "none"               # frozen at first build; widening requires a separate synthesis

@dataclass(frozen=True)
class TempSchedule:
    head_value: float
    tail_value: float
    ramp_start_fraction: float                              # 0..1; the fraction of horizon at which the ramp starts

def emit_dream(
    *,
    world_model: WorldModel,
    actor: Actor | None,                                    # None when action_policy == "uniform_random"; required for temperature_modified_actor
    ensemble: LatentDisagreementEnsemble,
    seed: DreamSeed,
    config: DreamRolloutConfig,
    dream_session_id: str,
    env_step_at_emit: int,
    checkpoint_id: str | None,
    rng: torch.Generator,
    device: torch.device,
) -> DreamRollout: ...
```

**The four-axis differentiation operationally specified.** The dream rollout differs from the preserved waking-planning rollout on all four axes; the differences are not vague guarantees but concrete code paths.

| Axis | Waking-planning rollout (preserved Probe 1.5 `_emit_dream`) | Dream rollout (new `emit_dream`) |
|---|---|---|
| **1. Goal-coupling** | Calls `actor.forward(view)` at every step. The action driving the next h is sampled from the actor's policy, which is shaped by the ensemble-disagreement intrinsic signal during training. Goal-coupling is the active-inference actor's intrinsic drive. | Default `action_policy="uniform_random"`: actions are drawn from `Uniform(0, action_dim)`, never from the actor. The actor is not loaded for the rollout in the default config. If `action_policy="temperature_modified_actor"` is chosen by configuration, the actor's softmax is divided by the independent `actor_action_temperature` scalar before action sampling — the goal-coupling is *softened* but not absent. **Default is `uniform_random`**, the cleanest goal-coupling-absent reading per synthesis §3 axis 1. |
| **2. Ensemble disagreement** | The ensemble is not stepped during the rollout. The actor consumes the disagreement signal during *training* (analytic gradients through imagine) but the dream-rollout-as-telemetry path does not compute it. | The ensemble *is* stepped at every dream step: `ensemble.disagreement(h, z, a_indices)` produces the per-step variance, which is written into `sequence_ensemble_disagreement_variance: list[float]` on `DreamRollout`. The disagreement is **never read** by any code path — not by the rollout's action policy (which is `uniform_random` by default), not by an internal trigger, not by the mirror's runtime control (the mirror is logs-only). The disagreement is recorded as substrate-side telemetry only. |
| **3. Temperature / sampling regime** | The prior is sampled at the world model's training-time temperature (effectively T=1.0, the prior's intrinsic sigma). | The prior's sigma is multiplied by `prior_temperature_t` at every step, where `prior_temperature_t` follows the configured `TempSchedule`. Default schedule: T=1.5 for the head 60% of the horizon, ramping linearly to T=2.5 over the tail 40%. The tail ramp is the *associative drift parametric regime* the synthesis §2 names — not a separate mode, just a parameter trajectory within the dream rollout. |
| **4. Initial conditions** | The seed is `self._h_curr`, `self._z_curr` — the runner's latest waking state. The rollout extends the current waking trajectory. | The seed is a `DreamSeed` produced by §2.2's `select_seed`, parameterized by `SeedSelectionConfig`. Default mode is `"replay"`: the seed is drawn from a replay segment at least `replay_min_segment_age_steps` steps old. The rollout is anchored to past experience, not current state. Periodic re-seeding (`re_seed_every_n_steps`) produces the chimera structure: the rollout's effective trajectory is a concatenation of K=horizon/re_seed_every_n_steps sub-trajectories, each starting from a fresh seed, producing the "imagined latents jump" signature distinct from a single continuous prior rollout. |

**The synthesis §3 commitment that "a dream rollout does all four jointly" is realized as the conjunction of: default config `action_policy="uniform_random"` + `record_ensemble_disagreement=True` + non-default `prior_temperature_schedule` + `select_seed(config)` with `mode != "current_waking"`.** All four are simultaneous, not sequential. Toggling any one off recovers a different regime (a waking-planning control, a pure-prior rollout, etc.); the visibility smoke (Phase 3) confirms the conjunction matters.

**The dream actor convention (plan-level design choice).** The default `action_policy="uniform_random"` is the plan's choice; the synthesis §3 axis 1 says "no action commitment during dream" without specifying the action distribution. `"temperature_modified_actor"` is the only non-default alternative the smoke can compare against in first build. A `"no_action"` mode (a zero-embedding action passed to the recurrence) is *not* included in the first-build Literal: the world model's `GRU(h_t, z_t, a)` recurrence already takes an action embedding at every step, and a "null action" path would require changing the world-model API. That change is out of scope for Probe 3; if a future probe wants the no-action mode, it adds the API path through its own synthesis. The plan recommends `"uniform_random"` for first build because it provides a concrete non-degenerate action distribution distinct from waking, which the four-axis test can detect.

**Flag — axis-1 / axis-3 coupling under `temperature_modified_actor`.** `actor_action_temperature` is held *separate* from `prior_temperature_schedule` rather than reusing the per-step prior multiplier. The two are conceptually distinct surfaces: prior temperature is axis 3 (sampling regime on the latent prior); actor action temperature is axis 1 (action policy commitment). Sharing one schedule across both would couple the axes and make the visibility smoke (Phase 3) unable to attribute distributional shifts to either axis cleanly. The split is the plan's structural enforcement of the four-axis independence; the cost is one extra config knob.

**Flag — associative-drift rollout-vs-session interpretive call.** The default tail-ramp schedule is *per rollout* (the temperature climbs across steps inside a single dream rollout). An alternative reading of "associative drift" is *per session* (later rollouts in a dream session run at higher temperature than earlier rollouts in the same session). Probe 3 implements the rollout-level reading as the first-build default because it is the more directly testable signature (KS-D on per-step latent distance within a rollout). The session-level reading is not built; if Phase 3 finds the rollout-level signature insufficient, the session-level alternative is the first knob to try.

**Gradient policy.** `gradient_policy: Literal["none"] = "none"` is the only valid value in first build. The Literal is *closed* at first build — not Literal["none", "world_model_only", "full"] — to make a future widening explicit. Widening to `"world_model_only"` or `"full"` requires its own synthesis (per the synthesis §1 commitment: "any training-side response is deferred to a separate future synthesis"). The closed Literal is the structural enforcement of that commitment.

**Reads.** World model (forward, no backward); ensemble (disagreement, no backward); replay buffer (via `select_seed`); RNG.

**Writes.** One `DreamRollout` record per call (via the runner's existing `_dream_sink`).

**Tests (5).**
- *Four-axis simultaneous differentiation.* Given a fixed seed, running `emit_dream` with default config and the waking-planning control on the same checkpoint produces `DreamRollout` records whose `seed_kind`, `temperature_schedule`, `sequence_ensemble_disagreement_variance` (None for control), and `sequence_action` distribution differ on all four axes.
- *Temperature schedule ramp-up.* With default `TempSchedule`, the effective sigma multiplier at step 0 is 1.5; at step `int(0.6 * horizon)` it begins to climb; at step `horizon - 1` it is approximately 2.5 within numerical tolerance.
- *Periodic re-seeding produces chimera.* With a configured `re_seed_every_n_steps=N`, the `re_seed_step_indices: list[int]` field on the resulting `DreamRollout` is `[N, 2N, ...]`; `max_step_latent_norm_change` is large at the re-seed indices (the latent jumps). The test verifies the mechanism under the configured value, not the default cadence as an immutable contract.
- *Ensemble disagreement is recorded and bounded.* `sequence_ensemble_disagreement_variance` has length `horizon`, all finite, all non-negative.
- *Gradient policy is enforced.* Running `emit_dream` with `torch.set_grad_enabled(True)` does not produce gradients on world model or actor parameters (assert with `world_model.parameters()` gradient norms zero after the call).

**Not at Probe 3.** No lucid control (out of scope per synthesis §7). No mirror-driven action policy (the mirror is logs-only). No introspective self-prediction during dream (the head-runs-only-during-waking commitment resolves to `docs/decisions/Kind_probe1_5_synthesis.md` §1.5 and is reiterated in `kind/observer/schemas.py` on `DreamRollout`; `sequence_self_prediction` stays `None` — the reserved slot is for a future probe). No gradient flow (closed Literal enforces it).

### 2.4 State machine — `kind/training/state_machine.py`

**Files.** `kind/training/state_machine.py` (new).

**Public interface.**

```python
@dataclass(frozen=True)
class StateTransition:
    from_state: Literal["waking", "dreaming", "dormant", "paused"]
    to_state: Literal["waking", "dreaming", "dormant", "paused"]
    dream_session_id: str | None
    trigger: Literal["desktop_off", "desktop_on", "mac_off", "mac_on",
                     "hard_cap_wallclock", "hard_cap_rollout_count",
                     "checkpoint_window", "compute_budget"]
    env_step_at_transition: int
    wallclock_ms_at_transition: int

@dataclass(frozen=True)
class DreamEnvelopeConfig:
    hard_cap_wallclock_ms: int = 30 * 60 * 1000           # 30 minutes per dream session
    hard_cap_rollout_count: int = 50                       # max DreamRollouts per session
    checkpoint_window_force_dormant: bool = True           # at checkpoint boundary, dreaming → dormant
    dormant_heartbeat_interval_ms: int = 60_000            # one heartbeat per minute of dormant
    compute_budget_seconds_per_hour: float = 1800.0        # 30 min/hour cap on dream compute

class StateController:
    def __init__(self, envelope: DreamEnvelopeConfig, seed_selection: SeedSelectionConfig) -> None: ...
    @property
    def current_state(self) -> Literal["waking", "dreaming", "dormant", "paused"]: ...
    def tick(self, host_signals: HostSignals, env_step: int, wallclock_ms: int) -> StateTransition | None: ...
    # ... emits a DreamSessionMeta start record on waking → dreaming, an end record on dreaming → *

@dataclass(frozen=True)
class HostSignals:
    desktop_alive: bool
    checkpoint_in_progress: bool

class DesktopWatcher:
    """Reads desktop on/off state. Probe 3 default: a file watcher on a sentinel file
    the desktop writes/deletes; Probe 4 may replace with a network heartbeat."""
    def poll(self) -> bool: ...                              # returns current desktop_alive
```

**The four states, their exit conditions, and what runs in each.**

| State | What runs | Primary exit trigger | Hard-cap exit triggers (content-blind) |
|---|---|---|---|
| **Waking** | Env-server live; agent acts; world model trains; agent_step telemetry; periodic waking-planning rollouts (the preserved control, off the visibility smoke onward) | `desktop_off` → `dreaming` | process-supervisor stop → `paused` outside `tick` |
| **Dreaming** | No env-server (or env-server idle); world model frozen for gradient (per `gradient_policy="none"`); dream rollouts at configured cadence within the session; ensemble disagreement recorded each rollout; `DreamSessionMeta` start record at entry | `desktop_on` → `waking`; otherwise after exit triggers | `hard_cap_wallclock` / `hard_cap_rollout_count` / `checkpoint_window` / `compute_budget` → `dormant`; process-supervisor stop → `paused` outside `tick` |
| **Dormant** | World model and actor weights resident in memory; no GRU ticking; no dream rollouts; periodic `dormant_heartbeat` `world_event` emitted; canonical state remains on Mac disk per Probe 1 / 1.5 persistence | `desktop_on` → `waking` | process-supervisor stop → `paused` outside `tick` |
| **Paused** | Nothing — process not running. Resumes when both machines on. | `mac_on` AND `desktop_on` → `waking`; `mac_on` AND `desktop_off` → `dormant` (or `dreaming` if envelope permits) | — |

**The state machine's four load-bearing commitments.**

1. *Exogenous primary trigger.* `desktop_off` / `desktop_on` are the only "Io-side has nothing to do with this" signals that gate state entry. No `intrinsic_signal_t` threshold, no KL satiation, no sleep pressure scalar living in Io's state — the synthesis §5 rejection of internal triggers is enforced by the *absence* of any field on `HostSignals` that depends on Io's state.
2. *Hard caps are content-blind.* All four hard-cap triggers consume only `wallclock_ms`, env-step counts, rollout-count integers, and the boolean `checkpoint_in_progress`. The compute budget is enforced by a rolling one-hour ledger of dream-compute seconds maintained by the `StateController`; when the ledger exceeds `compute_budget_seconds_per_hour`, the next `tick` emits `trigger="compute_budget"` and transitions `dreaming → dormant`. None consumes `DreamRollout` content; none consumes mirror readings. The mirror does not get a runtime channel here.
3. *Paused is supervised, not tick-detected.* `paused` is the state recorded on process shutdown / startup by the runner supervisor, not a transition inferred from `HostSignals.mac_alive` while the Mac is already off. `tick()` only runs while the process is alive; on restart, supervisor startup emits `mac_on` and resumes to `waking` or `dormant` based on `DesktopWatcher`.
4. *Dormant is operationally distinguishable from dreaming.* The two states have different `world_event` signatures (dormant emits `dormant_heartbeat` periodically; dreaming emits `DreamRollout` records and `state_transition` events but no `dormant_heartbeat`). The mirror reading whether Io is dreaming or dormant is unambiguous from the telemetry alone — the synthesis §10's constraint that dream and dormant be operationally distinguishable in signature is enforced at the telemetry-emission level, not deferred to interpretation.

**Reads.** `HostSignals` (from `DesktopWatcher`); env-step counter (from runner); wallclock; checkpoint barrier state; rolling dream-compute ledger. Process-supervisor startup/shutdown handles `paused` transitions outside `tick`.

**Writes.** `world_event` records (`state_transition`, `dormant_heartbeat`); `DreamSessionMeta` records to the new JSONL stream.

**Tests (4).**
- *Transition table coverage.* For each permitted runtime transition (waking↔dreaming, dreaming↔dormant, dormant↔waking) plus supervisor-mediated paused transitions, a unit test asserts the right trigger produces the right transition and the right `world_event` record is emitted.
- *Hard cap fires on wallclock.* With `hard_cap_wallclock_ms=1000`, after 1.5s of `dreaming` state, the next `tick` produces a `dreaming → dormant` transition with `trigger="hard_cap_wallclock"`.
- *Dormant heartbeat cadence.* With `dormant_heartbeat_interval_ms=100`, 1 second of dormant state produces ~10 heartbeats (within ±2 for timer jitter).
- *No Io-state-derived trigger exists in the public interface.* A presence test on the `HostSignals` and `StateController.tick` signatures asserts no field accepts a scalar derived from agent_step or any Io-internal state.

**Not at Probe 3.** No clear-light substrate continuation, ālaya-as-stream, or other dormant-content-side design (per synthesis §10 — Probe 3 *constrains* the dormant design without resolving it; the deeper question is held open). No lucid control over transitions. No mirror→scheduler channel (per synthesis §6 — held as a future permitted option, not first-build scaffolding).

### 2.5 Dormant minimum behavior (resolved to Probe 3's required level)

**Files.** Folded into `kind/training/state_machine.py` (no separate module).

**Scope of the resolution.** Probe 3 requires that dormant be (a) operationally distinguishable from dreaming and from waking in the telemetry timeline, and (b) a legitimate non-failure post-dream state. Both are achieved by:

- Dormant emits `dormant_heartbeat` `world_event` records at configured cadence. The heartbeat carries `mac_alive: true` plus elapsed dormant wallclock. Dormant gaps (no records of any kind) would be indistinguishable from `paused`; heartbeats positively assert "not paused, just resting."
- Dormant does not emit `DreamRollout` records. Dreaming does. The two telemetry signatures are disjoint by construction.
- Dormant does not step the world model GRU; the canonical state on disk does not change during dormant beyond the heartbeat record. This matches the synthesis §10 framing of dormant as the four-state model's *non-dreaming offline* state, with the deeper substrate-continuation design (clear-light, ālaya-as-stream, susupti) explicitly deferred.

**What is deferred.** The synthesis §10 names but does not resolve whether dormant should have *its own* generative signature (e.g., a slow background process, an EMA-style storehouse, a substrate-continuation regime). Probe 3's choice is the minimum: no generative process runs during dormant beyond the heartbeat emission. If a future probe finds this minimum too thin, the dormant-side design becomes that probe's question, and the heartbeat-and-no-GRU baseline is the comparison anchor.

**Open question (ii) resolution.** This section resolves the synthesis's open question (ii) at the level Probe 3 needs (operationally distinguishable, non-failure, minimum behavior specified). The deeper dormant-state design is *explicitly deferred*. The journal records both the resolution and the deferral so a future probe's research can begin from the recorded constraint rather than re-deriving it.

### 2.6 Mirror integration with dream telemetry — `kind/mirror/dream_reading.py`

**Files.** `kind/mirror/dream_reading.py` (new); `kind/mirror/criteria_v2.py` (existing — extended with a `dream_state` reading-context tag per criterion); `kind/mirror/prompt_builder.py` (existing — extended to emit a dream-state framing block when the digest's session contains dreaming-state telemetry).

**Public interface.**

```python
@dataclass(frozen=True)
class DreamReading:
    schema_version: str
    run_id: str
    timestamp_ms: int
    reader_role: Literal["dream_observer"]
    dream_session_id: str
    digest_run_id: str
    digest_session_range: tuple[int, int]                     # (start env_step, end env_step) of session
    state_typed_claims: list[StructuredClaim]                 # claims tagged with state="dreaming"
    free_text_notes: str

class DreamReader:
    def __init__(self, model: str, max_tokens: int, api_key: str | None = None) -> None: ...
    def read_session(
        self,
        digest: HierarchicalDigest,
        dream_session_id: str,
        active_criteria: list[str],
    ) -> DreamReading: ...
```

**What the mirror sees during dream sessions.** The dream-state reading consumes the same `HierarchicalDigest` Probe 2 produces, but the digest is asked to load a session-bounded range (the env-step range from `DreamSessionMeta` start to end). The reader receives the three-cohort layout from Probe 2's digest:

- **Substrate-side cohort.** `DreamRollout` per-step fields: `sequence_prior_entropy`, `sequence_z_prior` allocation, `sequence_ensemble_disagreement_variance`, `mean_step_kl_successive_priors`, `max_step_latent_norm_change`. The dream session's distributional properties of these are the *legible* substrate-side patterns. The mirror reads them against the active criteria's substrate-side prompts.
- **Head-internal cohort.** *Disabled for first build.* The Probe 1.5 self-prediction head does not run during dream per `docs/decisions/Kind_probe1_5_synthesis.md` §1.5 ("Relationship to dream state") and the `DreamRollout` docstring in `kind/observer/schemas.py`; `DreamRollout.sequence_self_prediction` stays `None`. The reader's prompt explicitly notes the head-internal surface is not available during dream and the mirror should not claim head-internal evidence during dream sessions.
- **Behavior-side cohort.** *Disabled for first build.* The actor's policy is not committed during dream (`action_policy="uniform_random"` by default; even with `"temperature_modified_actor"` the actor's choices do not couple back to the env). Behavior-side conditioning analysis (which depends on per-state action-distribution-under-perturbation) has nothing to read during dream. The reader's prompt notes this and the mirror should not claim behavior-side evidence.

**What the mirror writes during dream sessions.** A `DreamReading` JSONL record to `runs/{run_id}/mirror/dream_readings.jsonl`. Optionally, a `mirror_marker` `world_event` of `source="system"` carrying a builder-facing notification *if* the reading flags an anomaly the builder should know about (the synthesis §6 "builder-facing notifications" path). The `mirror_marker` carries no runtime authority — its emission does not change the state machine, does not interrupt dreaming, does not change the seed selection.

**What the mirror does *not* do during dream sessions.** No scheduler control. No `state_transition` emission (only the state machine emits those, from `source="environment"`). No mirror-driven dream termination (`uncertain` rulings default to `continue_and_log_uncertainty` per synthesis §6; the runtime ignores them anyway because the hard caps are content-blind). No reading or writing of any data-plane field Io can read (the `dream_readings.jsonl` is not on Io's read path).

**The one-way constraint is preserved by construction**: the mirror imports `kind.observer.schemas` and `kind.observer.dream_session` as read-only observer models, plus `kind.mirror.structured` for its own writes; it does not import `kind.training.state_machine` or `kind.training.dream`. The dependency lint from Probe 1's view-isolation test pattern (`tests/test_view_isolation.py`) is extended to cover this boundary.

**Reads.** `DreamRollout` Parquet shards; `DreamSessionMeta` JSONL; `WorldEvent` JSONL (for state-transition context); the existing hierarchical digest infrastructure.

**Writes.** `runs/{run_id}/mirror/dream_readings.jsonl`; optionally a `mirror_marker` `world_event` with `source="system"`, payload carrying a builder-facing notification.

**Tests (3).**
- *Mirror reads dream telemetry end-to-end against a Phase-3-smoke-output run.* The reader produces a `DreamReading` with state-typed claims; the head-internal and behavior-side prompt fragments emit the "not available during dream" framing per the configuration.
- *One-way invariant is preserved at module import boundary.* The view-isolation lint asserts `kind.mirror.dream_reading` may import read-only observer modules (`kind.observer.schemas`, `kind.observer.dream_session`) but does not import `kind.training.state_machine` or `kind.training.dream`.
- *Faithfulness verifier resolves dream-state claims.* A claim citing `sequence_ensemble_disagreement_variance` on a known `dream_session_id` resolves correctly against the cached telemetry.

**Not at Probe 3.** No real-time mirror calls (the reader runs post-hoc against committed telemetry, like Probe 2's caller). No mirror-driven seed selection. No mirror-driven envelope control. No head-internal or behavior-side dream-state criteria (deferred until a future probe wires self-prediction or action-coupling into dream).

### 2.7 K=5 ensemble role during dream (resolved open question i)

**Files.** Folded into §2.3 (`kind/training/dream.py`). No separate module.

**Resolution.** The ensemble disagreement is **computed at every dream step** and **recorded into `sequence_ensemble_disagreement_variance: list[float]` on `DreamRollout`**. It is **never read** by any code path during the rollout — not by the action policy, not by the rollout's exit condition, not by any internal trigger. The synthesis §3 axis 2's framing of "a quantity that constitutively does not exist in waking" (the K=5 disagreement is *used* in waking to drive the actor's intrinsic signal; in dream it is *recorded but not used*) is realized by computing it through the dream's prior-rollout trajectory and writing it as substrate-side telemetry only.

**Cost.** The K=5 ensemble's per-step forward is ~5 small MLP evaluations, comparable to the recurrence + prior step itself. For dream horizon H=30, the per-rollout ensemble cost is ~150 K=5 head evaluations. At the default cadence of ~1 dream rollout per minute and a 30-minute hard cap, a max-length session is ~4,500 head evaluations. That is still expected to be cheap relative to waking world-model training, but Phase 8 records the measured cost rather than relying on this estimate.

**Why not "skip it during dream":** the synthesis §3 axis 2 makes "ensemble disagreement recorded-but-not-used" a *load-bearing* difference between dream and waking. The cost question is real but cheap; recording is the implementation that satisfies the synthesis. The visibility smoke (Phase 3) is what confirms recording-not-using actually produces a distributional signature distinct from waking-planning rollouts (where the ensemble is not stepped at all during the telemetry path).

**Open question (i) resolution.** The synthesis's open question (i) is resolved at "compute every step, record every step, never read at runtime, cost is negligible." The journal records both the resolution and the cost measurement from the integration smoke.

---

## 3. Schemas as first-class artifacts

### 3.1 `DreamRollout` extensions (fourteen fields)

| Field | Type | Required at `"0.3.0"`? | Notes |
|---|---|---|---|
| `dream_session_id` | `str \| None` | yes | UUID linking to a `DreamSessionMeta` record |
| `seed_kind` | `Literal["replay", "perturbed_prior", "hybrid"] \| None` | yes | which seed mode produced the rollout's initial conditions |
| `seed_replay_segment_id` | `int \| None` | conditional (yes if `seed_kind` is `"replay"` or `"hybrid"`) | provenance for replay; under the Phase 0.5 amendment (Option 1) this is the `env_step` of the start of the obs/action warmup window |
| `seed_replay_step_offset` | `int \| None` | conditional (yes if `seed_kind` is `"replay"` or `"hybrid"`) | provenance for replay; under the Phase 0.5 amendment (Option 1) this is the warmup-window-local offset of the conditioning start (first build always 0; the warmup length itself lives in `sampling_parameters["replay_warmup_length"]`) |
| `seed_perturbation_magnitude` | `float \| None` | conditional (yes if `seed_kind` is `"perturbed_prior"` or `"hybrid"`) | L2 norm of the perturbation |
| `temperature_schedule` | `list[float] \| None` | yes | per-step temperature multiplier on the prior's sigma; length = horizon |
| `sub_mode_tags` | `list[str] \| None` | optional | e.g. `["chimera"]`, `["associative_drift_tail"]`; for mirror legibility |
| `sampling_parameters` | `dict[str, float] \| None` | yes | e.g. `{"action_temperature": 1.0, "re_seed_every_n_steps": 10.0}` |
| `gradient_policy` | `Literal["none"] \| None` | yes | closed Literal at first build; widening requires a separate synthesis |
| `rng_seed` | `int \| None` | yes | per-rollout RNG state for byte-stable reproducibility |
| `termination_reason` | `Literal["horizon_complete", "early_terminate_safety", None] \| None` | yes | `early_terminate_safety` reserved for a future safety-cap; default `"horizon_complete"` |
| `re_seed_step_indices` | `list[int] \| None` | optional | the step indices at which periodic re-seeding fired; `None` if none |
| `sequence_ensemble_disagreement_variance` | `list[float] \| None` | yes | length = horizon; the K=5 disagreement trajectory through the dream (resolves open question (i)) |
| `checkpoint_hash` | `str \| None` | yes | the source checkpoint's hash (the `actor + world_model + ensemble` weights' SHA256), so faithfulness can verify the run's identity |

**Provenance-field semantic note (Phase 0.5 amendment).** `seed_replay_segment_id` and `seed_replay_step_offset` were named expecting the rejected Option-2 buffer-stored-latents path. Under the adopted Option 1, the same fields carry Option-1 semantics: `seed_replay_segment_id = env_step` of the warmup window's start (a stable identifier within a run); `seed_replay_step_offset = 0` (the seed always reads the full window from its start in first build). The warmup window length is recorded in `sampling_parameters` under the key `"replay_warmup_length"`. No schema bump — the v0.3.0 fields keep their declared types; only the conventions for populating them change. `docs/decisions/phase0_5_replay_seed_source_2026-05-27.md` records why.

**The Probe 1.5 `sequence_self_prediction` reserved field stays `None`** in Probe 3 first-build dream rollouts — the head runs only during waking. The slot is still the forward-compatibility hook the synthesis §1 commits to defer; if a future probe extends self-prediction to imagined trajectories (per `docs/workingjournal/pre-probe3.md`'s framing 2 — "self-prediction over imagined trajectories"), the slot is already there.

**A `model_validator` on `DreamRollout`** enforces, for `schema_version == "0.3.0"`, that the nine *unconditionally required* fields above are non-None, and that the conditional fields are non-None when their `seed_kind` triggers them. Probe 1 / 1.5 records (`"0.1.0"` / `"0.2.0"`) bypass the validator's required-field check by virtue of their schema_version literal.

### 3.2 `WorldEvent` payload conventions (additive — no field-set change)

`WorldEventType` Literal gains `"state_transition"` and `"dormant_heartbeat"`; it already contains `"builder_perturbation"` from Probe 1/2, so Phase 7 reuses that existing event type rather than adding a third Probe 3 event. Payload conventions are documented in the `kind/observer/schemas.py` module docstring (the same pattern Probe 2 used for `is_sham`, `lesion_kind`, etc. — payload is schemaless within itself, so adding payload shapes does not bump `WorldEvent`'s field-set version, only the closed Literal `WorldEventType`).

### 3.3 Schema version bump rationale

The plan recommends bumping the telemetry schema version from `"0.2.0"` to `"0.3.0"`. The reason: Probe 3 adds fourteen new fields to `DreamRollout`, two new `WorldEventType` Literal values, and a new `DreamSessionMeta` record type. While each individual change could be additive (Optional fields, additive Literal values), the cohort is substantial enough that giving Probe 3 records a distinct version tag is the cleaner discipline. Probe 1.5's pattern (writer-side validator enforces the new fields for the new version; old-version records remain readable with the new fields as `None`) is followed.

`schemas/v0.4.0.json` is the new frozen JSON Schema export covering Probe 1.5's `0.2.0` telemetry (unchanged), Probe 3's `0.3.0` `DreamRollout` (extended), Probe 3's `0.3.0` `WorldEvent` (extended with only `state_transition` and `dormant_heartbeat`; `builder_perturbation` already exists in current `WorldEventType`), and Probe 3's `0.1.0` `DreamSessionMeta`. The Probe 2 mirror-side and conditioning models are unchanged from `schemas/v0.3.0.json`.

**Version-name collision note.** The repo already has `PROBE_2_EXPORT_VERSION = "0.3.0"` for the Probe 2 multi-family JSON Schema export. Probe 3's telemetry record `schema_version == "0.3.0"` is a record-level version, not the export-file version. The implementation should name constants to avoid ambiguity (for example, `TELEMETRY_SCHEMA_VERSION = "0.3.0"` and `JSON_SCHEMA_EXPORT_VERSION = "0.4.0"`) and document that `schemas/v0.4.0.json` contains records stamped `"0.3.0"`.

---

## 4. Test scaffolding — per-phase gate tests

Each phase commits to a specific test count; the totals appear in the table below. The tests are written following Probe 1 / Probe 2's discipline: one file per test file, fixtures in `tests/conftest.py`, mypy `--strict` on all `kind/` sources.

| Phase | Test file(s) | Count | What is verified |
|---|---|---|---|
| 0 | `tests/test_dream_schemas_v0_3_0.py` | 6 | DreamRollout round-trip with new fields; v3 validator behavior; backward read of v0.2.0 records; DreamSessionMeta start/end double-write; WorldEvent state_transition payload; v0.4.0.json export byte-stable |
| 1 | `tests/test_dream_seed.py` | 3 | replay mode deterministic re-encoding (RNG-byte-equal across calls + re-encoded from the recorded obs window byte-for-byte; per Phase 0.5 amendment); perturbed_prior magnitude; hybrid convex combination |
| 2 | `tests/test_dream_rollout_module.py` | 5 | four-axis simultaneous differentiation; temperature schedule ramp; periodic re-seeding chimera signature; ensemble disagreement recording; gradient policy enforcement |
| 3 | `tests/test_dream_visibility_smoke.py` | 2 | the smoke runs end-to-end and produces the comparison report; the report's KS-D thresholds are configurable |
| 4 | `tests/test_state_machine.py` | 4 | transition table coverage; wallclock hard cap; dormant heartbeat cadence; no Io-state-derived trigger in interface |
| 5 | `tests/test_dream_reading.py` | 3 | mirror reads dream telemetry end-to-end; one-way invariant at module-import boundary; faithfulness resolves a dream-state claim |
| 6 | `tests/test_runtime_protection.py` | 3 | wallclock cap fires content-blind; rollout count cap fires content-blind; mirror `uncertain` never short-circuits the runtime |
| 7 | `tests/test_cross_probe_surface.py` | 2 | DreamEnvelopeConfig and SeedSelectionConfig are perturbable from outside; no hidden-state-write affordance is reachable |
| 8 | `tests/test_probe3_integration_smoke.py` + journal scaffold | 2 | end-to-end waking→dreaming→dormant→waking cycle on small sizes; the integration smoke produces all expected telemetry streams |
| **total** | | **30** | |

**Three of the tests are load-bearing** (in the sense that a failure means the synthesis's premise about Probe 3 is not met by the implementation):

1. **`test_dream_rollout_module.py::test_four_axis_simultaneous_differentiation`** — the Probe 1.5 Phase 7 lesson at the module level: the four-axis differentiation is not just a schema slot.
2. **`test_dream_visibility_smoke.py`** — the same lesson at the empirical level: the dream regime drives non-degenerate variation. This is what gates Phases 4–8.
3. **`test_state_machine.py::test_no_io_state_derived_trigger_in_interface`** — the synthesis §5 exogenous-trigger commitment, structurally enforced at the type signature level.

---

## 5. Sequencing rationale

**Why Phase 3 (visibility smoke) runs before Phases 4–8 (state machine, mirror integration, runtime protection, cross-probe surface, integration smoke).** Probe 2 took the same shape — Phase 12 (calibration smoke and first real Gemini run) ran before Phases 9–11 (judge layer, stability runner, faithfulness verifier) deliberately to surface real LLM findings early. The Probe 1.5 Phase 7 lesson is the load-bearing precedent: capacity-as-architectural-slot is not the same as capacity-as-non-degenerate-conditioning-surface. The new actor input column was non-degenerate only after the small-Gaussian init; the schema slot existed at `ckpt-000001` but the behavioral surface was identically zero.

For Probe 3 the analogous risk is: the fourteen new `DreamRollout` fields exist (Phase 0); the dream-rollout module populates them (Phase 2); but the dream regime's distributional properties are indistinguishable from the waking-planning rollout's (because the temperature schedule isn't strong enough, or the re-seeding cadence is wrong, or `uniform_random` is too close to the actor's actual policy on a small action set). If this is true, Phases 4–8 commit machinery to a non-load-bearing rollout — the mirror reads the dream session, finds nothing, the state machine emits transitions, and the result is the Concern A failure mode the synthesis names directly.

Phase 3 is the empirical test that adjudicates between "the dream regime works" and "the dream regime is a relabelling." It runs on an existing Probe 1.5 checkpoint (no new training required), executes the three rollout regimes (dream, waking-planning control, pure-prior control), computes per-field distributional distances, and either passes (Phases 4–8 are worth building) or fails (the dream-rollout module is revised; Phase 3 reruns; downstream stays gated).

**What Phase 3's pass criterion looks like operationally.** KS-D ≥ 0.15 on at least three of {`mean_step_kl_successive_priors`, `cumulative_prior_entropy`, `max_step_latent_norm_change`, `sequence_ensemble_disagreement_variance` aggregate, decoded-obs entropy aggregate}, measured between dream and each of the two controls. The threshold (0.15) is a plan-level design choice with a default — it is calibrated by comparison to Probe 1.5 Phase 8's four-way KS comparison (within-family 0.077–0.133, cross-family 0.22–0.34; the synthesis verification notes name these). 0.15 is just below the cross-family lower bound, biased toward "dream should be at least as different from waking-planning as the Probe 1.5 frozen-target control was from Probe 1." The threshold can be revised after Phase 3 produces its first numbers; the plan does not pre-commit.

**What is not in the early-empirical-finding pattern.** Phase 0 (schemas), Phase 1 (replay seed selection), and Phase 2 (the dream-rollout module) must precede Phase 3 because Phase 3 *consumes* them. Phase 3 cannot run earlier than this. The plan's commitment is to not skip Phase 3 — not to run Phase 3 before Phases 0–2.

---

## 6. Open-during-build decisions with defaults

The plan acknowledges several "open during build" knobs with sensible defaults. None blocks the build.

| Knob | Default | Revisit when |
|---|---|---|
| Dream horizon | **30** | Phase 3 finds dream and waking-planning distributions overlap at the small-horizon end; try 50 |
| Prior temperature head value | **1.5** | Phase 3 KS-D below threshold; try 2.0 |
| Prior temperature tail value | **2.5** | dream rollouts produce numerical instability (NaN in prior log-σ); lower to 2.0 |
| Tail ramp start fraction | **0.6** (last 40% of horizon) | associative drift is undetectable in telemetry; move to 0.4 (earlier ramp) |
| Re-seed every N steps | **10** | chimera signature absent from `max_step_latent_norm_change`; lower to 5 |
| Action policy | **`uniform_random`** | Phase 3 finds dream collapses to pure-prior; try `temperature_modified_actor` |
| Default seed mode | **`replay`** | Phase 3 finds dream collapses to waking-planning (the replay seed is too close to current state); try `perturbed_prior` |
| Replay min segment age | **1000 env steps** | replay seeds correlate with current state in `h_init` cosine similarity; raise to 2000 |
| Hard cap wallclock (dream session) | **30 min** | smoke runs show typical dream sessions exit on this rather than `desktop_on`; raise to 60 min |
| Hard cap rollout count (dream session) | **50** | typical session does not approach this; tighten to 30 |
| Dormant heartbeat interval | **60 s** | mirror complains the dormant timeline is too sparse; shorten to 30 s |
| Compute budget per hour | **30 min** | typical run does not approach this; tighten to 20 min |
| Dream cadence within session | **~1 rollout per minute of dream session wallclock** | dream session is too short to fit many rollouts; tighten cadence to ~1 per 30s |

These defaults are starting points. The visibility smoke (Phase 3) informs the first revision of the temperature, re-seed cadence, and seed mode. The integration smoke (Phase 8) informs the first revision of the envelope caps.

---

## 7. Cross-probe surface

**What Probe 3 exposes that Probe 4 will eventually use.** The synthesis §8 commits Probe 4 to envelope-and-seed-selection perturbation only — no hidden-state writes. Probe 3 realizes the exposed surface as two typed configs on `RunnerConfig`:

- **`DreamEnvelopeConfig`** (§2.4): hard caps, heartbeat interval, compute budget. These are the *when* and *how-long* of dreaming.
- **`SeedSelectionConfig`** (§2.2): seed mode, perturbation sigma, hybrid alpha distribution, replay min segment age. These are the *from-what* of dreaming.

**Probe 4 perturbation mechanism, deliberately minimal.** A new script `scripts/perturb_dream_envelope.py` accepts JSON config deltas and, at the next checkpoint boundary, applies them to the runner's `DreamEnvelopeConfig` and `SeedSelectionConfig`. The perturbation fires a `world_event` using the already-existing `event_type="builder_perturbation"`, `source="builder"`, payload carrying the config delta and a `target="dream_envelope"` or `target="dream_seed_selection"` tag. Like Probe 1's existing mutator hooks, no marker enters Io's observation space — the perturbation is visible in `world_event` (which Io does not read) but not in `agent_step` (which Io reads via `PolicyView`). This is the minimum script needed to test the authorized control surface end-to-end in Phase 7, not Probe 4 scheduler scaffolding; Probe 4 may replace it after its own synthesis.

**What is *not* exposed.** No script, no `RunnerConfig` field, no `world_event` payload shape, and no module API permits writing into Io's hidden state `h_t` or stochastic state `z_t` directly. The synthesis §8 explicitly rejects Candidate B's proposed `h_t` overwrite during dream as a "category violation"; the plan enforces this by *not building the affordance*. A future probe wanting that capability must add it deliberately and through its own synthesis — the absence in Probe 3 is the structural guarantee.

**Why this matters for Probe 4.** Probe 4's research will inherit from this surface: the builder perturbs *what conditions Io's substrate enters during dreams* (the seed selection and the envelope) and *when dreams happen* (via desktop on/off, which is the existing primary trigger). Probe 4 does not perturb the substrate's manifestation under those conditions. The plan's contribution to Probe 4 is the typed surface; Probe 4's research decides what to do with it.

---

## 8. Two open questions: resolution

The synthesis names two implementation-level open questions in its implementation-phase open-questions paragraph ("Two open questions for the implementation phase: (i) the exact shape of the K=5 ensemble's role during dream when action selection is absent; (ii) the dormant-state design, which Probe 3 constrains but does not settle"). The plan resolves both at the level Probe 3 needs:

### 8.1 Open question (i): K=5 ensemble role during dream — **resolved**

Resolution per §2.7: compute disagreement at every dream step; record into `sequence_ensemble_disagreement_variance: list[float]` on `DreamRollout`; never read at runtime. Cost is expected to be low but is measured explicitly (~150 K=5 head evaluations per rollout; ~4,500 at the default max-length session). The resolution is what makes the synthesis §3 axis 2 "recorded-but-not-used" a structural property of the implementation rather than a guarantee in prose.

### 8.2 Open question (ii): dormant-state design — **resolved to Probe 3's required level + deferred for deeper design**

Resolution per §2.5: dormant emits `dormant_heartbeat` `world_event` records periodically; does not emit `DreamRollout` records; does not step the world model GRU. Operationally distinguishable from dreaming and from paused.

**Deferral:** the deeper question — whether dormant should have a substrate-continuation regime (clear-light, ālaya-as-stream, slow EMA storehouse, susupti-analog) — is not resolved here. The synthesis §10 names this as A's framing; Probe 3's commitment is to design *around* it without making the choice. The minimum-dormant baseline (heartbeat, no GRU) is what a future probe revising dormant compares against.

Both resolutions are written into the journal at Phase 0 (open-question track) so they are visible from the start of the build.

---

## 9. Out of scope at Probe 3

Explicit list. The plan does not build for Probe 4 except where forward-compatibility was authorized in the synthesis (envelope and seed-selection control surfaces in §7).

- **No gradient flow from dream.** `gradient_policy: Literal["none"]` is closed; widening requires a separate synthesis.
- **No lucid control.** Out of scope per synthesis §7.
- **No mirror→scheduler channel.** Logs-only mirror per synthesis §6. The future permitted path is named but not built.
- **No internal trigger for dream entry.** No KL satiation, no sleep pressure, no surprise quota. Exogenous trigger only per synthesis §5.
- **No head-internal dream-state criteria.** Self-prediction head does not run during dream per `docs/decisions/Kind_probe1_5_synthesis.md` §1.5 and the `DreamRollout` schema docstring.
- **No behavior-side dream-state criteria.** Actor policy is not committed during dream.
- **No hidden-state-write affordance for Probe 4.** Per synthesis §8.
- **No substrate-continuation design for dormant.** Per §8.2 above.
- **No `sequence_self_prediction` population during dream.** The slot is reserved; populating it is a future probe's choice.
- **No real-time mirror reading during dream.** Post-hoc reads against committed telemetry only.
- **No counterfactual self-absence dreams.** Candidate C's alternative shape (running the world model "as if Io were absent") is interesting but not built; the synthesis did not adopt it.
- **No SHY-style synaptic pruning during dormant.** Candidate B's alternative shape is not built.
- **No revision of Probe 1, Probe 1.5, or Probe 2 substrate.** All three are settled; Probe 3 builds on top.
- **No multi-Io dreaming.** Single core agent per project commitment.
- **No new actor-readable interface.** The Probe 1.5 v2 §2(b) discipline applies as carried forward by `docs/decisions/Kind_probe2_synthesis.md` §2.5(8) and `docs/workingjournal/probe2_templates/pre_registration.md`: `new_actor_readable_interfaces_added = []` at all Probe 3 pre-registration records. Io does not gain a new self-readable quantity during dream.

The rule of thumb: if a Probe 4 (or later) feature can be added without changing the dream-rollout module's four-axis differentiation, the envelope-and-seed-selection control surface, or the closed `gradient_policy` Literal, it is out of scope here.

---

## 10. Connection to the journal

Probe 3 ends with a journal entry at `docs/workingjournal/probe3.md` recording what was learned, what surprised, what is now decided, what is now newly open. The journal follows Probe 2's discipline: phase headings, "what's now closed / what's now newly open" rhythm, an explicit Watts-default-applied-to-builder log entry per phase (the Probe 1.5 v2 §2(b) sub-clause as carried by Probe 2 §2.5(8) applies; Probe 3 phases that touch the design notes' "should the system have access to X about itself?" question record the default-no decision with a one-line reason).

**Pre-registration discipline.** Probe 2's pre-registration sink (`kind/observer/pre_reg.py`) is used during Phase 5 mirror integration. The `PreRegistration` record for each dream-state mirror reading declares the active criteria (substrate-side only for first build), the surfaces per criterion (substrate_side only — head_internal and behavior_side disabled per §2.6), the asymmetry-of-access (Io does not read `dream_readings.jsonl`; the mirror reads `DreamRollout`), the column_init (carries from the Probe 1.5 v2 checkpoint identity), and `new_actor_readable_interfaces_added=[]` (Probe 3 adds none).

**First entry covers (template — actual content is the builder's):**

- What the visibility smoke (Phase 3) found. The dream-vs-waking-planning and dream-vs-pure-prior KS-D values per field; whether the threshold passed; what was revised before passing if not.
- What surprised. The dream regime producing decoded observations the builder finds eye-readable or eye-illegible; the chimera structure being more or less coherent than the synthesis anticipated; the ensemble disagreement trajectory through dream looking like nothing the waking distribution prepared the builder for.
- What is now closed. The two open questions (i) and (ii) per §8.
- What is now newly open. Anything Phase 3's empirical reading surfaced that the synthesis did not anticipate — including whether the substrate-continuation deferral on dormant is starting to bite, and whether the first build's logs-only mirror is informative enough or whether the future-permitted mirror→scheduler channel is the next decision to make.
- A short note on whether the synthesis's load-bearing constraints (logs-only, no gradient flow, envelope-only Probe 4) held up operationally — and which would have been hard to enforce if the discipline had not been written down.

The journal is the bridge between Probe 3 and Probe 4's research-prompt drafting. Probe 4 begins by reading this entry.

---

*End of plan.*
