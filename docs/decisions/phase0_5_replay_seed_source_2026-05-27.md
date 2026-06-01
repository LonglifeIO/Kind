# Phase 0.5 — Replay seed source decision — 2026-05-27

## Recommendation

**Option 1: re-encode from stored obs.** Replay-mode `DreamSeed` is produced by running the world model forward over a short obs/action window read from the existing `SequenceReplayBuffer`, starting from a zero `(h, z)` state. The buffer is unchanged; no `(h, z)` is stored alongside transitions. Provenance is `(start_env_step, warmup_length)` plus the source `checkpoint_hash` (already on the v0.3.0 `DreamRollout` schema).

Recommended `warmup_length` default: **8 conditioning steps**. The visibility smoke (Phase 3) can revise.

Phase 1 test 1 ("byte-match the buffer's segment") is reworded as a *deterministic re-encoding* test, implementable as written under Option 1. Plan §2.2 is amended; §3.1 gets a one-paragraph note reinterpreting two provenance fields without changing the schema.

## The mismatch (observation)

Plan §2.2 specifies `DreamSeed(h_init, z_init, replay_segment_id, replay_step_offset, ...)` with Phase 1 test 1:

> *Replay-mode seed selection: returns a `DreamSeed` whose `(h_init, z_init)` byte-matches the buffer's segment at `(replay_segment_id, replay_step_offset)`*

That test assumes `(h, z)` are stored in the buffer at the recorded `(segment_id, step_offset)`. They are not. Verified directly:

- `kind/training/replay.py:65–79` — `Transition` has six fields: `obs`, `action`, `next_obs`, `env_step`, `episode_id`, `step_in_episode`. No `h`, no `z`.
- `kind/training/replay.py:175–267` — `insert()` and `sample()` move only those six fields; nowhere does the buffer touch `(h, z)`.
- `kind/training/runner.py:946–954` — the runner's `_step_once` constructs `Transition` from exactly those fields before `self._replay.insert(transition)`.

For completeness, the current dream-seeding behavior (which the new replay seed will replace):

- `kind/training/runner.py:988–992` — after every world-model forward, the runner stamps `self._h_curr = wm_step.h.detach()` and `self._z_curr = wm_step.z.detach()` from the latest waking step.
- `kind/training/runner.py:1161–1170` — `_emit_dream` seeds the rollout from `self._h_curr` and `self._z_curr`. The current dream is anchored to live waking state — exactly what synthesis §3 axis 4 forbids for Probe 3.

The mismatch is in §2.2 (and §4 Phase 1 test 1), not in the synthesis. Synthesis §3 axis 4's commitment ("initial conditions from replay buffer or perturbed prior, not current state") is silent on *how* the replay-mode `(h, z)` is derived from the buffer.

## Options considered

### Option 1 — re-encode from stored obs

Replay seed produced at seed-selection time by running the world model forward over a short obs/action window from the buffer, starting from zero `(h, z)`. The output `(h_final, z_final)` is the seed.

- Buffer storage unchanged.
- Provenance is the obs window's `(start_env_step, length)` plus the source checkpoint.
- Determinism conditional on `(buffer state, checkpoint, RNG)` — reproducible.
- Cold-start ok: a Probe 1.5 checkpoint resumed into Probe 3 has obs in the buffer immediately.

### Option 2 — extend the replay buffer to store latents

Modify `SequenceReplayBuffer.Transition` to carry `(h, z)` alongside obs, write at insert time. Seed selection retrieves historical `(h, z)` directly.

- Buffer storage shape changes; `Transition` gains two tensor fields.
- Provenance is byte-equal historical latents; faithfulness verifier just looks up the buffer address.
- Storage cost: `(h_dim + z_dim) * 4 bytes = (200 + 16) * 4 = 864 bytes` per transition; at `replay_capacity=100_000` that's ~86 MB additional RAM.
- Cold-start problem: resuming from a Probe 1.5 checkpoint, the buffer's pre-Probe-3 transitions have no latents — they cannot be Option-2-seeded until the buffer fills with new Probe-3-instrumented transitions (≥ `replay_min_segment_age_steps=1000` of new waking, plus warmup).

### Option 3 — hybrid (considered, not adopted)

Store latents at coarse granularity (every Kth step), re-encode finer granularity from obs. Compresses Option 2's storage cost but inherits its cold-start problem and adds machinery. No clear advantage over Option 1.

## Criteria weighed

| # | Criterion | Option 1 | Option 2 |
|---|---|---|---|
| 1 | Fidelity to synthesis §3 axis 4 | "Past conditioning, current substrate" — the obs anchor is past; the current world model manifests its present-day generative model of those past observations. | "Past everything" — the latents are frozen artifacts of the substrate at recording time. The current world model reads in latents produced by an older version of itself. |
| 2 | Impact on Probe 1/2 substrate | None — `SequenceReplayBuffer` unchanged. | `Transition` dataclass changes; buffer's insert path changes (runner must supply `h`, `z` at insert); checkpoint resume path unaffected (buffer is in-memory only at Probe 1). Bounded but non-zero. |
| 3 | Storage cost | Zero. | ~86 MB additional RAM at `replay_capacity=100_000`. |
| 4 | Seed-time cost | One world-model forward per warmup step (default 8). Negligible against the per-rollout cost (horizon-30 prior rollout + 30 ensemble disagreements). | Memory read. Faster, but the seed-time cost is not the bottleneck on a dream session. |
| 5 | Determinism / provenance legibility | `(h, z) = f(obs_window, checkpoint_weights)`. Reproducibility = `(buffer obs, checkpoint hash, RNG)` — every piece is already a Probe 3 artifact (the buffer is in-memory but obs are persisted in the agent_step / world_event streams under Probe 1's existing telemetry, and `checkpoint_hash` is on v0.3.0 `DreamRollout`). | `(h, z)` is byte-fixed at recording time. Reproducibility = `(segment_id, step_offset)`. Simpler check; but requires either persisting the buffer or accepting that the verifier can only resolve dreams whose buffer windows are still resident. |
| 6 | Phase 1 byte-match test | Test 1 rewording required: from "byte-matches the buffer's segment" to "byte-equal re-encoding from `(start_env_step, warmup_length)` under the recorded checkpoint." Implementable. | Test 1 implementable as-written *once the buffer is extended*. |
| 7 | Capacity-over-exercise | Consistent. Nothing about Option 1 installs a drive or signals optimization. | Consistent. Storing latents is a memory shape, not a behavior. |

### Cold-start interpretation (interpretation, not observation)

Both options would meet axis 4 for a from-scratch Probe 3 run. The asymmetry surfaces on Probe-1.5-checkpoint-resumed runs (the realistic Probe 3 starting state): Option 1 can immediately use any transition in the buffer; Option 2 must wait for `replay_min_segment_age_steps` of *Probe-3-instrumented* waking before any dream session can begin. This pushes the visibility smoke (Phase 3) downstream of an additional ~1000 env steps of fresh data accumulation. Not a hard blocker, but a concrete asymmetry the criteria above don't capture cleanly.

### Synthesis-level interpretive question

Synthesis §3 axis 4's "initial conditions from replay buffer or perturbed prior, not current state" is satisfied by both options. The deeper interpretive question — *what does "from past" mean when the substrate is current?* — is open to two readings:

- **"Past anchor through current substrate"** (Option 1): the substrate is what's running; asking it to "dream" means letting it generate from a non-current anchor. Storing historical latents would create a dual provenance (current substrate consuming an older substrate's outputs), which is a subtle category confusion. Yogācāra-flavored: past obs are bīja-like; the current substrate is what perfumes them into current latents.
- **"Past everything"** (Option 2): the dream is re-entry of frozen historical state. Hippocampal-replay-flavored: the recorded patterns are exactly what they were.

The synthesis is silent on this. The integrated take §1 framing — "dreaming is afforded; it is not exercised in the service of waking competence" — and §2's "manifestation under withdrawn conditioning" lean (interpretively) toward Option 1: the substrate manifests, given an altered conditioning input; the substrate doing the manifesting is the substrate that exists right now.

This interpretation is mine. A reader could weigh the synthesis differently and prefer Option 2 on the "past everything" reading. The recommendation does not depend on this interpretive call alone — Option 1 also wins on impact (criterion 2), storage cost (3), and the cold-start asymmetry above. The interpretive question is named so the journal's resolution is reviewable.

## Reasoning for the recommendation

Option 1 wins on five of the seven criteria (1, 2, 3, 5, with 6 a small rewording cost; 7 a tie) and on the cold-start asymmetry. Criterion 4 (seed-time cost) favors Option 2 only marginally and the cost is negligible against the rollout. The interpretive call on criterion 1 is the only place where a different reader might prefer Option 2, and the other criteria are sufficient to recommend Option 1 even under the contrary reading.

Concretely, Option 1 means:

- `SeedSelectionConfig` gains one field: `replay_warmup_length: int = 8` — the number of obs/action conditioning steps used to develop `(h, z)` from a zero start.
- `select_seed` for `mode="replay"`:
  1. Sample a valid window of length `replay_warmup_length` from the buffer (uniform over valid window starts, respecting `replay_min_segment_age_steps`).
  2. Read the obs and action sequences from that window.
  3. Run the world model forward from `(h=0, z=0, a=0)` over those `warmup_length` steps using the posterior `q(z | h, obs)` and the recurrence `h_{t+1} = GRU(h_t, z_t, a_t)`. RNG drives the posterior sampling.
  4. Return `DreamSeed(h_init=h_final, z_init=z_final, ...)` plus provenance.
- Provenance fields on `DreamSeed` are populated as: `replay_segment_id = start_env_step` (stable identifier of the window within a run), `replay_step_offset = 0` (the window starts at the segment start; the seed reads the entire window).
- `replay_warmup_length` is recorded in the `DreamRollout.sampling_parameters` dict at emission time so a verifier has the warmup-window length without needing to consult `SeedSelectionConfig` separately.

The `replay_warmup_length=8` default is a starting point. Eight steps from a zero start gives the GRU time to integrate enough obs context that `h_final` is not effectively zero, but is short enough that the seed-time cost stays negligible (8 world-model forwards × ~ms each). Phase 3's smoke runs against this default; if dream rollouts collapse toward pure-prior behavior because `h_final` carries too little obs structure, `warmup_length` is the first knob to raise (to 16, then 32).

A simpler "single-step encode" alternative (run the world model once from zero with a single obs, no warmup) was considered. It is rejected because the resulting `h` is one GRU step from zero, dominated by the GRU bias terms rather than any obs structure — the dream's "initial conditions from replay" would be operationally indistinguishable from "initial conditions from a zero state." This would make the visibility-smoke gating (Phase 3) noisier than necessary on a parameter Probe 3 actually controls.

## Synthesis-level commitments preserved

- **§3 axis 4** ("initial conditions from replay buffer or perturbed prior, not current state") — preserved. Option 1 uses the obs anchor from the buffer; the seed is *not* derived from `self._h_curr` / `self._z_curr`.
- **§1 (capacity over exercise)** — preserved. The seed mechanism affords a non-current-state dream entry; it does not install a drive.
- **§1.5 (head runs only during waking)** — preserved. The warmup-window world-model forward consumes obs and produces `(h, z)`; the self-prediction head is not invoked during warmup or during the dream rollout.
- **§8 (Probe 4 envelope-and-seed-selection only)** — preserved and slightly strengthened. `replay_warmup_length` joins `SeedSelectionConfig` as another envelope-side knob Probe 4 can perturb; the perturbation surface is wider, but it is still *content-blind from Io's side* (the actor and the world model do not read `SeedSelectionConfig`).
- **§5 (exogenous trigger)** — preserved. Seed selection happens inside a dream session that was already entered via the state machine's exogenous trigger.

## Synthesis-level commitments re-interpreted

None — but two readings are made explicit so the journal records them:

- **"From past" as "past obs anchor, current substrate."** Option 1 commits to this reading. The alternative reading ("byte-equal historical latents") is consistent with §3 axis 4 but is not adopted in first build for the reasons above.
- **DreamRollout provenance field names.** The v0.3.0 schema fields `seed_replay_segment_id` and `seed_replay_step_offset` were named expecting Option-2-like semantics ("which historical segment, what offset within it"). Under Option 1 the names are slightly misleading: `seed_replay_segment_id` is best read as "stable identifier of the obs window (the start env_step within a run)" and `seed_replay_step_offset` as "where the warmup begins within the window (first build always 0)." `replay_warmup_length` is carried in `sampling_parameters`. No schema bump; the v0.3.0 fields stay as-is; the journal records the semantic reinterpretation.

## Open follow-ups for the chosen path

1. **`replay_warmup_length` default.** 8 is plausible but unconfirmed; Phase 3's smoke informs the first revision.
2. **`sampling_parameters` key vocabulary.** `replay_warmup_length` joins the dict; Phase 0 journal noted the vocabulary is established at Phase 2 emission. Phase 2's writer needs to commit to the key name (recommended: literal `"replay_warmup_length"`, integer-valued).
3. **Buffer access API for windows.** Phase 1 will need a way to read an obs/action window by start position. Two options at implementation: extend `SequenceReplayBuffer` with `get_window(start_env_step, length) -> tuple[Tensor, Tensor]`, or have `select_seed` accept a sampled `Batch` and operate on it directly. The plan amendment does not pre-commit; Phase 1's choice is a code-organization decision.
4. **Hybrid mode under Option 1.** `select_seed(mode="hybrid")` first computes the replay-mode `(h_replay, z_replay)` via re-encoding, then uses it as the perturbed-prior anchor, then convex-combines under `hybrid_mixture_alpha`. This is what the plan §2.2 description already implies; the re-encoding step is now the explicit first step.
5. **Cold-start with Probe 1.5 checkpoints.** Once Phase 1 is implemented, a small sanity check on a resumed Probe-1.5 buffer (the buffer has obs from before Probe 3 instrumentation) confirms `select_seed(mode="replay")` produces a usable seed without instrumentation gaps. If it does not, the cold-start asymmetry needs revisiting before Phase 3.
6. **What the faithfulness verifier checks.** Option 1's reproducibility chain is `(obs window from buffer, checkpoint hash, RNG seed) → (h_init, z_init)`. The verifier will need access to the obs window. Whether to persist replay-buffer obs to disk (so verifiers can resolve dreams against retired checkpoints) is a deferred question; Probe 3 does not commit to it.
