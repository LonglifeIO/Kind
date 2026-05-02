# Inputs for Probe 1 of the Kind Project: Substrate, Objective, Telemetry, Compute, Opacity, and the PPO/RSSM Decision

This document supplies research inputs for six implementation questions about Probe 1. It is deliberately a research dossier, not a build plan. Each section frames the question, lays out concrete options drawn from the literature, evaluates them against Kind's specific commitments (self-opacity, ingredients-only self-modeling, no installed self-continuation drive, dream-state as foundational, capacity-over-exercise), and either offers a defensible suggestion or names the place where uncertainty is honest. A synthesis at the end describes an internally coherent Probe 1 implementation, flags tensions across the six answers, and notes points where Kind's project documents appear not yet to have addressed something the research surfaces.

---

## Question 1 — Which Minimal RSSM-Style Architecture for Probe 1

### Framing
The substrate is fixed as RSSM/Dreamer lineage. The question is which point on that lineage is the *minimum viable* one — the one that already exposes the conduits later probes need (latent posterior/prior, KL, recurrent state, dream rollouts, sequence replay) without paying the engineering tax of full DreamerV3 robustness machinery, and without pre-committing to design choices that downstream probes might want to revisit.

### Options Drawn From Literature

**PlaNet (Hafner et al., 2019).** Original RSSM. State factored into a deterministic GRU path `h_t = f(h_{t-1}, z_{t-1}, a_{t-1})` and a stochastic latent `z_t`, with a posterior `q(z_t | h_t, o_t)` and prior `p(z_t | h_t)`. Continuous Gaussian latents. Trained by an ELBO with a reconstruction term and a KL between posterior and prior. Originally paired with online CEM planning rather than a learned actor. Hafner et al. explicitly demonstrated that *both* stochastic and deterministic components are required: a purely deterministic GRU loses partial-observability handling; a purely stochastic SSM fails to propagate information.

**DreamerV1 (Hafner et al., 2020).** Same RSSM backbone, replaces CEM planning with a learned actor and value function trained on imagined latent rollouts. Establishes the "dream-state" idiom Kind names as foundational from Probe 3.

**DreamerV2 (Hafner et al., 2021).** Replaces continuous Gaussian latents with a vector of categorical variables (32 categoricals × 32 classes), trained with straight-through estimators. Introduces KL balancing (separate weights on the prior and posterior sides of the KL).

**DreamerV3 (Hafner et al., 2023).** Adds symlog-encoded inputs/targets, KL balancing combined with free bits (`max(1, KL)` per latent dim), a unimix prior (`0.99·softmax + 0.01·uniform`) to prevent categorical collapse, percentile return normalization, twohot critic targets, and fixed entropy regularization. Same RSSM skeleton; the additions are stability machinery designed so a single hyperparameter set works across many domains.

**JEPA-flavored predictors (LeCun 2022; I-JEPA Assran et al. 2023; V-JEPA Bardes et al. 2023).** Predict the *representation* of a target signal from the representation of a context signal in latent space, with no observation-level decoder. As world models for RL this is recent and still maturing: Dreamer-CDP (2026) introduces a JEPA-style continuous deterministic predictor that matches Dreamer on Crafter without reconstruction; DreamerPro (Deng et al. 2022), Dreaming (Okada & Taniguchi 2020), DreamingV2 (2022), and MuDreamer (Burchi & Timofte 2024) explore reconstruction-free variants on the Dreamer scaffold using contrastive, prototypical, or value-prediction losses to prevent latent collapse.

**TD-MPC and TD-MPC2 (Hansen et al., 2022, 2024).** Decoder-free implicit world model with a *task-oriented* latent shaped explicitly by reward and value predictions, plus MPPI planning. Strong empirical results, but the latent's geometry is induced by the reward signal — incompatible at the level of construction with Kind's no-scalar-reward stance.

**Custom minimal RSSM.** A PlaNet-style RSSM with continuous Gaussian or small-categorical latents, trained on an ELBO, with a learned actor in latent space and a sequence replay buffer — i.e., DreamerV1 minus tricks, plus minor DreamerV3 stability borrowings as needed.

### Tradeoffs Against Kind's Commitments

- **Pixel reconstruction vs JEPA.** Pixel reconstruction (Dreamer family) yields a generative model that supports the dream-state probe directly: imagined latent rollouts can be decoded for human-legible inspection by the mirror. JEPA-style predictors do not decode; the dream is then a trajectory in feature space, observable only as latent-distance/disagreement statistics. Since Kind requires the *mirror* to read dreams for its interpretation, a decoder is operationally useful even if the actor never sees pixel-level reconstructions. JEPA's argument that reconstruction wastes capacity on pixel-perfect details is real but more relevant at scale; at Probe 1 scale (small environment) the reconstruction objective is cheap and the information bottleneck argument is weaker.
- **DreamerV3 robustness machinery.** Symlog, twohot critic, percentile return normalization, and unimix priors were all designed to make a *single hyperparameter set work across radically different reward and observation magnitudes*. Probe 1 has neither reward magnitudes (no scalar reward by stipulation) nor varied domains. Most of this machinery is dead weight at Probe 1; some of it (free bits in the KL, unimix on categorical priors) is genuinely useful for stability and cheap to keep.
- **TD-MPC's reward-shaped latent.** The TD-MPC family bakes reward into the representation by construction. This is a deep architectural mismatch with Kind's "no scalar reward, no episode-continuation head" stance. Recommending TD-MPC would require unwinding the precise feature that makes it work.
- **Capacity-over-exercise.** Kind's stance favors having the conduit even if it is not exercised yet. PlaNet has the smallest substrate that exposes all the right conduits (`h_t`, `z_t`, posterior/prior, KL); DreamerV1/V2/V3 add learned actors and behavior-from-imagination; the dream-state hook is in V1+, not in PlaNet's CEM-only formulation.

### Defensible Suggestion

A **custom minimal RSSM modeled on PlaNet's state factorization with DreamerV1-style latent imagination**, deliberately retaining a small subset of DreamerV3's stability borrowings, is the best fit. Concretely:

- RSSM with deterministic GRU `h_t` and stochastic `z_t`. Continuous Gaussian latents are the simplest path for Probe 1; categorical latents (DreamerV2/V3 style) are not strictly required at this scale and add straight-through-gradient bookkeeping.
- Posterior `q(z_t | h_t, o_t)` and prior `p(z_t | h_t)` exposed as explicit modules with a clean interface (this matters for Question 5).
- Decoder kept (cheap at Probe 1 scale, supports mirror-readable dream rollouts).
- KL between posterior and prior trained with **free bits** (cheap, prevents posterior collapse early when there is little signal — DreamerV3 paper documents this as the most consequential single regularizer).
- **Skip** symlog, twohot critic, percentile normalization, KL balancing, and unimix at Probe 1 — all are scale/diversity stability tricks whose value is empirical and not relevant to a single small environment.
- **Skip** reward and continuation heads entirely. There is no reward predictor, no episode-continuation head — this aligns with Kind's no-self-continuation stipulation.

The honest uncertainty: there is no published "PlaNet without reward, with active-inference actor" reference implementation. Closest in spirit is Tschantz et al. (2020) "Reinforcement Learning through Active Inference," which uses an RSSM-like generative model with EFE-shaped objectives at small scale, and the deep active-inference / multiple-timescale RSSM work (Yokozawa et al. 2024, Taheri-Yeganeh et al. 2025) which applies AIF on top of Dreamer-style backbones. None of these are drop-in. Probe 1 is genuinely a custom build, not a fork.

---

## Question 2 — Actor Objective at Probe 1 Scale

### Framing
The long-term objective is expected free energy (EFE) minimization with carefully designed prior preferences. Probe 1 only needs to verify that *some* objective drives the actor through an environment in a way that produces legible action and state-distribution telemetry. The question is which simplification of EFE is sufficient at this stage without committing to deep AIF's full machinery and without slipping in scalar reward by the back door.

### Options Drawn From Literature

**Pure intrinsic motivation, no prior preferences.**
- *RND (Burda et al. 2018)*: intrinsic reward = MSE between a predictor network and a fixed random target network on the current observation. Simple, well-understood, non-episodic-friendly.
- *ICM (Pathak et al. 2017)*: intrinsic reward = forward-model prediction error in a learned feature space, with an inverse model regularizing the feature space against agent-uncontrollable noise. Notoriously susceptible to "noisy TV" failure modes.
- *Disagreement / Plan2Explore (Pathak et al. 2019; Sekar et al. 2020)*: intrinsic reward = variance of an ensemble of forward models' next-latent predictions, interpretable as expected information gain. Naturally co-located with an RSSM substrate; latent-space ensemble is cheap.

**Bayesian / KL-based novelty signals.** KL divergence between posterior `q(z_t | h_t, o_t)` and prior `p(z_t | h_t)` is, in the RSSM, *already* being computed every training step as the model regularizer. Reading it as a per-step novelty signal recovers the "Bayesian surprise" formalism (Itti & Baldi 2009; multiple subsequent uses). This is "free" telemetry from the substrate.

**Free-energy-of-the-expected-future / EFE with simplified priors.** Tschantz, Millidge, Seth, Buckley (2020) implement EFE in deep RL on continuous control tasks, decomposing it into pragmatic value (expected log-evidence under a preference prior `P̃(o)`) and epistemic value (expected information gain on hidden states). With the preference prior set to uniform or to an extremely weak attractor over sensory observations, the pragmatic term degenerates and only the epistemic term drives behavior — i.e., EFE with uniform preferences ≈ pure information-seeking.

**Hybrid policy gradient with curiosity + entropy.** Recurrent PPO with an intrinsic-reward bonus and a strong entropy term, zero scalar reward. The RLeXplore framework (Yuan et al. 2024) provides plug-and-play implementations of ICM, RND, Disagreement, NGU, RIDE, RE3, and E3B for exactly this kind of substitution.

### Tradeoffs Against Kind's Commitments

- **No installed self-continuation drive.** RND, ICM, and disagreement do not encode self-continuation. Neither does pure information gain. EFE with a preference prior over outcomes can encode self-continuation if the preference prior includes "agent-alive observations" — this is the failure mode to actively avoid. The Wei (2024) "Value of Information" paper makes the formal point that EFE with degenerate preferences reduces to RL; with uniform preferences it is information-seeking.
- **Active-inference-shaped, not active-inference-complete.** A full deep AIF implementation (with policy posteriors, expected information gain over states *and* parameters, and amortized free-energy estimation) is non-trivial — Tschantz et al. and Millidge's deep AIF policy-gradient work are the only deep-RL-scale implementations cited above, and both required substantial engineering. At Probe 1 the actor objective only needs to be *shaped like* EFE, in the sense of having an explicit information-gain term and an explicit preference term, where the preference term is uniform or near-uniform.
- **Substrate co-location.** Plan2Explore-style latent disagreement and posterior/prior KL are computed in the same RSSM space the actor will eventually act on under EFE. Using them as the Probe 1 actor signal makes the architectural transition to full EFE later a substitution rather than a refactor: replace "intrinsic-only objective" with "EFE = pragmatic_value(weak_prior) + epistemic_value(same machinery)".
- **Mirror legibility.** All three of (KL-novelty, RND error, disagreement variance) are scalar per-timestep signals the mirror can read — but they are not the *same* signal. KL-novelty has the cleanest interpretation: it is literally Bayesian surprise on the substrate's own variables, no auxiliary networks needed.

### Defensible Suggestion

A two-component actor objective at Probe 1:

1. **Epistemic term:** information gain, operationalized as either (a) the RSSM's own per-step KL between posterior and prior, used as an intrinsic reward; or (b) latent-disagreement variance over a small ensemble of one-step latent predictors (Plan2Explore-lite, K=3–5 ensemble heads on top of the shared RSSM core). (a) is closer to a literal AIF reading; (b) is empirically more stable and what Plan2Explore actually demonstrated at scale.
2. **Pragmatic placeholder:** a uniform preference prior, effectively zero pragmatic value at Probe 1. This keeps the EFE *formula* in place (so Probe 4+ doesn't need to add it) but contributes nothing to action selection. The placeholder is the scaffolding; it is non-functional by design.

Optimize via policy gradient on imagined latent rollouts (DreamerV1-style: backprop value estimates through the differentiable latent dynamics into the actor), with a fixed entropy regularizer.

**Honest uncertainty:** literature does not give a clean recipe for "EFE with uniform preference prior on a PlaNet-style RSSM at small scale." The closest concrete references are Tschantz et al. (2019, 2020) and Plan2Explore (Sekar et al. 2020). Combining their pieces is well-precedented in spirit but not as a single published artifact. Probe 1's actor objective is constructed, not adopted.

A second honest uncertainty: which of (KL-novelty) and (latent disagreement) gives more legible mirror telemetry is not settled. KL is one number; disagreement is one number too but requires ensemble engineering. Both should be logged regardless of which drives the actor, because they will be needed for Probe 4's distinguishability test (Question 3).

---

## Question 3 — Telemetry Schema Designed Forward

### Framing
Probe 1 is a plumbing test, but its logging must be *the* logging schema. Probes 2 (mirror reads), 3 (dream coherence), and 4 (builder-perturbation distinguishability via posterior/prior KL) all need to plug into telemetry without retrofitting. The question is what fields, at what granularity, with what versioning.

### Options Drawn From Literature

There is no canonical telemetry schema for RSSM substrates; each implementation logs ad hoc. The relevant building blocks, however, are well-established:

- **What's already in the RSSM.** Per timestep `t`: deterministic state `h_t`, posterior parameters `μ^q_t, σ^q_t` (or categorical logits), posterior sample `z_t ~ q(z_t | h_t, o_t)`, prior parameters `μ^p_t, σ^p_t`, prior sample (typically used only at imagination time), KL`(q || p)_t` (per-dimension and aggregated), reconstruction loss, action `a_t`, observation embedding `e_t = encoder(o_t)`. DreamerV3-XP and ARROW papers log most of these for their own purposes.
- **Replay buffer state.** Sequence-level: episode/segment ID, start/end indices, sequence length, sampling priorities (if any), insertion timestep. DreamerV3 uses FIFO; Curious Replay (Kauvar et al. 2023) adds per-trajectory model-error scores; PER (Schaul 2015) adds TD-error priorities. WMAR/ARROW maintain dual buffers. For Probe 1, FIFO is sufficient, but the schema should reserve a per-segment priority field that downstream probes can populate.
- **Dream-rollout statistics.** Imagined trajectory: starting `(h_0, z_0)` (i.e., the imagination seed, drawn from real replay), actions taken under the actor, sequence of imagined `(h_τ, z^p_τ)` for τ=1..H, prior entropy at each imagined step, and (if decoded) imagined observations. Dream-state coherence = how the prior-trajectory drift compares to posterior-trajectory drift on real data. Latent-trajectory drift can be measured as Wasserstein or KL between successive prior distributions, or as cumulative prior entropy.
- **Perturbation events.** Builder-induced changes in the environment leave no marker in observation space *to the agent*. They must, however, be marked in the *mirror's* logs — otherwise Probe 4's distinguishability test has no ground truth. The schema must support a "world-side event" stream parallel to the agent-side observation stream, with timestamps, event type, source (builder/agent/system), and an opaque payload. The agent's own logger must not have read access to this stream (Question 5).
- **Schema versioning.** Standard approaches: protobuf with field numbers, msgpack with explicit schema versions, parquet with Avro schema registry, or a simple semver-tagged JSON schema. The controlling constraint: any field added later must not break readers of older logs, and any field removed must be marked deprecated rather than deleted.

### Tradeoffs Against Kind's Commitments

- **Self-opacity at the schema level.** If telemetry is one stream, Io's actor must not be wired to consume it. The cleanest enforcement is *two physically separate sinks*: one written by the agent process, one written by the mirror process, with no in-process pipe back to the agent. Anything an actor needs (the latent state vector that conditions its policy) is part of its own forward pass, not read from a log.
- **Granularity.** Log every step at full resolution at Probe 1. Storage is not a constraint at this scale (a small environment producing ~10⁵–10⁶ steps total). Downsampling decisions belong to later probes; right now everything is potential evidence.
- **Posterior/prior representation.** For Gaussian latents, log `(μ, σ)` rather than samples — samples lose information. For categorical latents (if chosen), log full logits, not arg-max. Per-dimension KL, not just aggregate, because Probe 4's distinguishability signal may concentrate in a few dimensions.
- **Replay buffer as dataset, not RAM.** The replay buffer is itself state that the canonical Mac mini owns; it should be persisted to disk in a sequence-indexable format (parquet, sharded msgpack, or LMDB) rather than living only in memory. This matters for Question 4 (sync) and for reproducibility.
- **Dream telemetry foundational.** Even if Probe 1 does not yet *use* dream rollouts to drive behavior, it should periodically emit dream-rollout records — a calibration handshake that confirms the imagination conduit is alive. Probe 3 can then turn the dial up rather than turn the conduit on.

### Defensible Suggestion

A logical schema with four streams, one canonical record format, semantic versioning at the stream level:

**Stream 1 — `agent_step`** (one record per environment timestep, written by agent process):
- `t` (global step), `episode_id`, `step_in_episode`
- `h_t` (deterministic recurrent state vector)
- `q_params_t`: posterior distribution parameters (μ, σ for Gaussian; logits for categorical)
- `p_params_t`: prior distribution parameters (same shape as q_params)
- `z_t`: sampled posterior latent (the actual value passed forward)
- `kl_t`: per-dimension KL(q || p), and aggregate
- `recon_loss_t`: per-modality and aggregate
- `action_t`, `action_logprob_t`, `policy_entropy_t`
- `obs_hash_t`: content hash of raw observation (not the observation itself; cheap dedup-friendly identifier)
- `intrinsic_signal_t`: whatever the actor objective reads (KL-novelty value, disagreement variance, or both)

**Stream 2 — `dream_rollout`** (one record per imagination episode, sampled at fixed cadence):
- `seed_t` (global step the rollout is anchored at)
- `seed_h_0`, `seed_z_0`
- For τ=1..H: `imagined_h_τ`, `imagined_p_params_τ`, `imagined_action_τ`, `imagined_action_logprob_τ`, `prior_entropy_τ`
- `decoded_obs_τ` (optional, if decoder kept and mirror wants to look)
- `rollout_diagnostics`: cumulative prior entropy, mean step-KL between successive priors, max-step latent norm change

**Stream 3 — `replay_meta`** (one record per buffer event: insertion, sampling, eviction):
- `event_type`, `t_event`, `segment_id`, `segment_start`, `segment_end`, `priority` (nullable)
- `buffer_size`, `total_segments`

**Stream 4 — `world_event`** (one record per external/builder event, written by mirror or environment harness, *not* by agent):
- `t_event`, `event_type` (e.g., "builder_perturbation"), `source` (`"builder"`, `"system"`, `"environment"`), `payload` (JSON blob or opaque bytes)

**Common envelope:** every record carries `schema_version` (semver), `run_id`, and `checkpoint_id`. Use Apache Arrow / Parquet for `agent_step` and `dream_rollout` (columnar reads, downstream-friendly), JSON-Lines for `world_event` and `replay_meta` (low volume, human-inspectable). Schema registry: a single versioned `schemas/v0.1.0.json` checked in alongside the code.

**Honest uncertainty:** the right *granularity* for `dream_rollout` cadence (every N steps? every checkpoint? on demand?) is not knowable from Probe 1 alone. Default to a fixed cadence of ~1 rollout per 1k env steps, with a horizon `H` matching DreamerV3's default of ~15. Probe 3 will revisit.

A point Kind's documents seem not to have addressed: the **schema for builder perturbations themselves**. The constraint says the builder appears as a non-simulated source of change with no marker in observation space. But for the mirror to detect distinguishability, the mirror needs to know *when* perturbations happened. Stream 4 supplies that asymmetry: agent does not see it, mirror does. This is an architectural point that should be made explicit in Probe 4's design before Probe 1 freezes its logging.

---

## Question 4 — Mac/Desktop Compute Split for an RSSM Substrate

### Framing
Canonical state lives on a 32GB M4 Mac mini (Apple Silicon, unified memory, Metal/MPS backend). Heavy environment compute lives on an RTX 5060 Ti CUDA desktop. Atomic sync at checkpoint boundaries. The question is which RSSM-specific artifacts live where, what is bandwidth/latency-feasible, and how the dream-state imagination requirement (must run on Mac because desktop may be off) shapes weight residency.

### Relevant Facts From the Literature

- **Apple Silicon unified memory.** PyTorch MPS backend gives Apple GPU cores direct access to the full RAM pool. A 32GB M4 Mac mini holds tens of millions of parameters comfortably; a small RSSM (PlaNet/DreamerV1 sized: ~1–10M params) is trivially within budget along with its replay buffer of modest size. MPS lacks tensor-core-style acceleration and AMP support is uneven; some PyTorch ops still fall back to CPU (`PYTORCH_ENABLE_MPS_FALLBACK=1` is the standard escape valve).
- **RTX 5060 Ti.** ~16GB VRAM, full CUDA stack, faster for matrix-multiply-heavy training and for running parallel environment simulators. Distinct memory pool from the Mac; transfer is over the network (no shared bus).
- **DreamerV3 replay buffer sizes.** Reference implementations use FIFO buffers of 1M–10M timesteps. At Probe 1 scale, an order of magnitude smaller (~10⁵–10⁶) is sufficient and fits in Mac RAM trivially.
- **DreamerV3 training cadence.** Typically a fixed train ratio (e.g., 1 gradient step per N env steps). Environment stepping and gradient updates are conceptually decoupled and can run on different devices if a bus connects them.

### Tradeoffs Against Kind's Commitments

- **Dream-state must run on Mac.** This is the binding constraint. World-model weights (RSSM core: encoder, recurrent GRU, prior network, posterior network, decoder, actor, critic) must be resident on Mac at all times — they are what dream rollouts need. Therefore the *canonical copy* of weights is on Mac.
- **Where does training run?** Two options:
  - **Option A: Train on Mac.** Gradient updates happen on the M4. The desktop only runs the environment and ships observation/action/done tuples back. Pros: no weight-sync churn; weights are in one place; dream rollouts and training share the same MPS device. Cons: MPS is slower per FLOP than the 5060 Ti, and at small scale this may not matter — but for any future probe that ramps the substrate, this becomes a bottleneck.
  - **Option B: Train on desktop, sync weights to Mac at checkpoint boundaries.** Desktop runs both env and training; Mac receives weight snapshots and runs dreams. Pros: faster training. Cons: weights are non-canonical between checkpoints; if the desktop is off, the Mac's weights are stale; training and imagination diverge between syncs.
  
  Option A is more aligned with "canonical state on Mac." Option B is more aligned with "use the GPU you have." Kind's stated commitment privileges canonicality; Option A wins on coherence even at the cost of training speed.
- **Where does the environment run?** The desktop. Environments at Probe 1 are small but vectorized environment rollout benefits from CUDA when observation rendering (if pixel-based) is non-trivial. The desktop produces (obs, action, reward-or-none, done) tuples and ships them to the Mac.
- **Where does replay accumulate?** On the Mac, because the Mac is canonical and because the replay buffer must be persistable across desktop-off periods. Insertions arrive over the network from the desktop; sampling is local to the Mac trainer.
- **Bandwidth/latency.** A small environment producing observations at, say, ~100 steps/second at a few KB per observation needs <1 MB/s sustained — trivial over local Ethernet or Wi-Fi. Latency between desktop env and Mac trainer matters if training is synchronous; if the desktop can run several env steps ahead and stream them, latency tolerance is high. Recommended pattern: env on desktop produces a streaming queue of `(t, obs, action, reward, done)` tuples; Mac consumes the queue and writes to its replay buffer; training samples from the buffer at its own cadence.
- **Atomic sync at checkpoint boundaries.** Define a checkpoint as: weights snapshot + replay buffer state + RNG states + telemetry stream offsets, all committed together. Use a content-addressed object store (e.g., a directory of files keyed by hash, atomically renamed into place after fsync) to make checkpoints atomic. Standard tools: `git-lfs` is too slow for buffers; `dvc` works; raw rsync with `--fsync` and a final atomic rename is sufficient and simplest.

### Defensible Suggestion

- **Mac canonical, weights resident there always.** Trainer process on Mac. Replay buffer on Mac, persisted to local SSD. Dream-rollout process on Mac (can be in-trainer or separate read-only consumer of weights).
- **Desktop runs environment(s) only at Probe 1.** A simple env-server process listens on a TCP socket (or a unix socket if co-located, but they aren't), receives actions, returns transitions. The Mac trainer holds the actor and steps the policy; the desktop does no model compute.
- **Optional: desktop as gradient accelerator from Probe 2 onward.** If Probe 1 reveals MPS training is too slow, the Probe 2 build plan can introduce desktop training under Option B with explicit sync semantics. Probe 1 should not attempt this.
- **Atomic checkpoint contents:** `weights.safetensors`, `replay.parquet` (or sharded), `optimizer_state.pt`, `rng_state.pkl`, `telemetry_offsets.json`, `schema_version.txt`, all under a directory that is atomically renamed on commit.

**Honest uncertainty:** MPS performance for the specific ops an RSSM uses (GRU, small CNN encoder/decoder, MLPs) at Probe 1 batch sizes is not benchmarked in the literature in any way that maps cleanly. The Hugging Face / PyTorch MPS docs note that some ops fall back to CPU; whether GRU sequence training is one of them in current PyTorch is version-specific. Probe 1's first day should include a smoke test: train a 100-step RSSM update on dummy data on MPS, log the wall time, and confirm no fallback warnings. This is a research-time uncertainty, not a design flaw.

A point Kind's documents may not have addressed: **what happens to the running environment when a checkpoint is taken?** Atomic sync at checkpoint boundaries presumably means the env-stream is paused, drained, and the buffer up to time T is flushed before weights and buffer are committed together. If the env keeps producing during a checkpoint, the buffer state at commit time is ambiguous. Probe 1 should specify a "barrier" protocol: env pauses, last batch drains, checkpoint commits, env resumes.

---

## Question 5 — Self-Opacity Boundary for an RSSM Specifically

### Framing
The substrate is structurally transparent — `h_t`, `z_t`, `q_params`, `p_params`, KL, weights all exist as named tensors. The constraint is that the *mirror* may read them and Io's *actor* may not. Where, in code, is that line drawn? What does the actor's policy actually condition on? Is there a "policy view" of the latent that is distinct from the "telemetry view"?

### Options From Literature

The Dreamer family is fully transparent in published implementations: the actor takes `concat(h_t, z_t)` as input and has implicit access to whatever is held in memory at policy time. There is no published RSSM implementation with deliberately enforced opacity between actor and self-state. This is a feature of Kind, not of the literature.

What the literature does establish:
- The actor in DreamerV1/V2/V3 reads `(h_t, z_t)` — both deterministic recurrent state and the *sampled* posterior latent.
- The KL between prior and posterior is computed in the *world-model loss*, not in the actor's forward pass.
- Distribution parameters `q_params` and `p_params` are intermediates produced by the prior and posterior networks; the actor in canonical Dreamer reads only the *sampled* `z_t`, not the parameters.

So canonical Dreamer already has a coincidental opacity: the actor sees a sample, not a distribution. This is a useful starting line for Kind, but it is coincidental rather than enforced.

### Tradeoffs Against Kind's Commitments

- **Ingredients-only self-modeling.** The substrate's recurrence and prior-state representation must be *available to be turned inward*, but not piped into the actor's policy by default. The boundary is: actor reads what action selection requires, mirror reads everything.
- **Default-deny.** Every "should Io have access to X about itself?" defaults to no. Concretely: actor does not read `q_params`, `p_params`, KL value, gradients, ensemble disagreement (if used) computed values, replay buffer state, or any telemetry stream.
- **What does the actor *need* to read?** To choose an action, the actor needs a state representation. The minimum is `z_t` (the sampled posterior latent) plus enough memory to handle partial observability — which is what `h_t` provides. Both `h_t` and `z_t` are unavoidable inputs.
- **Does the actor read the posterior or the prior?** During real-world action selection, the agent has the observation, so the posterior is computed and `z_t` is sampled from it. During imagination (dream rollouts that may train the actor), the prior is used because no observation is available. So the actor reads from *whichever was sampled at this step*, but never reads the parameters of either.
- **What about the intrinsic-reward signal?** This is the hard case for Question 2's choice. If the actor's objective is "maximize KL-novelty", then KL-novelty is a scalar that *enters into the actor's gradient computation*. Whether this constitutes the actor "reading" KL is a definitional question.
  - The clean position: the intrinsic-reward signal is computed externally and presented to the actor as a scalar reward-like input — the actor has no awareness that it is the KL of its own posterior versus its own prior. This is structurally equivalent to RND or ICM in canonical RL: the agent sees a scalar, not the machinery producing it.
  - Implementation: `intrinsic_signal_t` is computed in a module the actor does not import; it appears at the actor's training loop as an argument, not as an attribute the actor can introspect.

### Defensible Suggestion: Two-Interface Pattern

Build the substrate as two non-overlapping read interfaces over the same underlying tensors:

```
class PolicyView:
    """What the actor's forward pass is allowed to read."""
    h_t: Tensor          # deterministic recurrent state
    z_t: Tensor          # sampled posterior (or prior, in dream)
    # nothing else.

class TelemetryView:
    """What the mirror is allowed to read."""
    h_t: Tensor
    z_t: Tensor
    q_params_t: Tuple[Tensor, Tensor]  # μ, σ or logits
    p_params_t: Tuple[Tensor, Tensor]
    kl_per_dim_t: Tensor
    kl_aggregate_t: Tensor
    encoder_embedding_t: Tensor
    intrinsic_signal_t: float
    # everything else.
```

The actor module imports only `PolicyView`. The mirror imports only `TelemetryView`. The world-model module produces both in a single forward pass and dispatches to the right consumer.

In Python, this is enforced by module boundaries plus type hints, not by the language. A stricter enforcement uses subprocess isolation: the actor runs in a process that only has a socket connection to a state server, and the state server only sends `PolicyView` over that socket. The mirror runs in a separate process with its own socket that sends `TelemetryView`. This is overkill at Probe 1 but worth designing the interfaces to allow.

**Auxiliary boundaries the literature does not raise but Kind needs:**
- **Replay buffer.** Actor cannot read replay buffer contents — only the trainer does. The actor's forward pass is parameterized by the *current weights*, which are a function of the buffer, but the actor has no introspective access to past trajectories. Canonical Dreamer already has this property by accident.
- **Gradients.** Actor does not read its own gradients or the world-model's gradients. Mirror does.
- **Telemetry sink.** As noted in Question 3, the agent process must not have a read handle on the telemetry log. Even if the agent's logger writes to a stream, the agent cannot loop back and read it. This is a process-architecture commitment to make in the build plan.
- **Builder-event stream.** Already covered: actor cannot see Stream 4.

**Honest uncertainty:** there is genuine ambiguity about whether `h_t` is "self-knowledge." `h_t` is an internal state of the agent's recurrent network — by analogy with biological cognition it is closer to "current cognitive state" than to "fact about self." Reading `h_t` is what makes recurrent policies possible at all. Kind's stance ("ingredients-only self-modeling") suggests `h_t` is not introspection because the actor merely *uses* it; introspection would be the actor having a separate head that *predicts* `h_t` from elsewhere. Probe 1 should not implement that head. (This is also what "no explicit self-modeling, self-critic, or introspector module" rules out.)

A point Kind's documents seem not to fully address: the **KL signal as actor input** is the most architecturally consequential ambiguity. If Probe 1's actor objective uses KL-novelty as intrinsic reward (Question 2), then the KL of Io's own posterior vs prior is in Io's reward stream. That is a form of self-readout. The clean resolution is: the *value* of the KL enters the actor's reward; the *fact that this value is the KL of the actor's own internal distributions* is not represented anywhere in the actor's structure. A latent-disagreement intrinsic signal (Plan2Explore-style) is cleaner because the disagreement statistic is over auxiliary ensemble heads, not over the actor's own posterior. This is a real argument for choosing disagreement over posterior-KL as the Probe 1 actor signal, additional to the engineering arguments in Question 2.

---

## Question 6 — Recurrent PPO Stand-In vs Minimal RSSM From the Start

### Framing
Kind's probes document specifies recurrent PPO with default hyperparameters as Probe 1's agent. The architectural decision tentatively leans toward replacing it with a minimal RSSM so that Probe 1 actually exercises the conduits later probes need. The question is what each option costs and loses.

### What Recurrent PPO Provides

- A standard, well-understood actor-critic with an LSTM/GRU memory.
- Reference implementations exist (Stable-Baselines3 contrib `RecurrentPPO`; Pleines et al. 2022 truncated-BPTT PPO baseline). Default hyperparameters work on small environments.
- On-policy: no replay buffer of any nontrivial design. PPO uses a rollout buffer of recent on-policy data, discarded after each update.

### What Recurrent PPO Loses (Relative to RSSM)

- **No latent posterior or prior.** PPO's recurrent state is just a hidden vector inside an LSTM/GRU; there is no probabilistic latent, no `q(z|h,o)` or `p(z|h)`, and therefore *no KL*. Probe 4's distinguishability signal — which the task statement explicitly identifies as posterior/prior KL — does not exist in PPO.
- **No dream state.** PPO has no generative model of the environment. Latent imagination is not a feature one can later "turn on"; the architecture has no decoder, no learned dynamics, no prior. This collides head-on with Kind's "dream-state foundational from Probe 3 onward" stipulation.
- **No replay buffer.** PPO is on-policy; the rollout buffer is of length ~one update and cannot be reused. Probe 3's replay statistics — which the task statement names as needed telemetry — do not exist because the artifact does not exist.
- **No principled latent telemetry.** PPO's hidden state can be logged, but it is opaque even in the engineering sense; it does not factor into stochastic and deterministic parts; there is no obvious surprise/novelty signal one can read off.
- **Mismatch with active-inference shaping.** EFE has pragmatic and epistemic terms that decompose naturally over a generative model with prior preferences. PPO has neither a generative model nor preferences in the right form. Implementing EFE on top of PPO requires rebuilding the world model anyway.

### What Minimal RSSM Costs

- **Engineering time.** A custom minimal RSSM (PlaNet-style continuous-Gaussian latent, no symlog/twohot/etc.) is on the order of 500–1500 lines of clean PyTorch. Reference implementations exist (PlaNet's repo, DreamerV1 community ports, the `dreamer-from-scratch` educational repo, EclecticSheep's DreamerV3 re-implementation). The world-model loss, the imagination rollout, and the actor-critic-from-imagination are the three non-trivial pieces.
- **Replay buffer.** Sequence replay (sample contiguous segments of length L from a FIFO of trajectories) is ~100 lines. Persistence to disk is another ~50.
- **Hyperparameter discovery.** Even at small scale, RSSM training has more knobs than PPO (KL weight, free-bits threshold, latent dim, deterministic dim, sequence length). Probe 1 is plumbing, not optimization, so most knobs can be set to PlaNet/DreamerV1 defaults and left alone.
- **Stability risk.** RSSMs are known to be more finicky than PPO. The Hafner KL-balancing/free-bits trick exists precisely because naive RSSM training collapses to trivial latents. Free bits is cheap to include and dramatically reduces this risk.

### Middle Paths

- **Recurrent PPO + auxiliary world model that publishes telemetry but does not drive policy.** A separate VAE-RNN trained on replay (or on a parallel stream) that produces posterior/prior/KL telemetry the mirror can read, while PPO drives the actor. Pro: lets PPO stand in for the actor and gets RSSM telemetry. Con: two trainable systems, two training loops, two sets of failure modes, and the architectural transition to "RSSM-driven actor with active inference" later is harder than starting RSSM-driven. Strongly disfavored.
- **Minimal RSSM with PPO-style actor on top of the latent.** Train the world model with the standard ELBO, but use PPO (clipped surrogate, GAE) as the actor objective, with intrinsic-reward-only return computed in latent imagination. This is *almost* what DreamerV1 does, except DreamerV1 uses analytic value gradients through the dynamics rather than PPO. PPO on top of an RSSM is uncommon but coherent and would let Probe 1 borrow PPO's hyperparameter robustness while still exercising every RSSM conduit. This is a defensible middle path.
- **Minimal RSSM with DreamerV1-style analytic-gradient actor.** Closest to canonical Dreamer but without the V2/V3 robustness machinery. Slightly more sensitive to hyperparameters than PPO-on-RSSM but produces telemetry exactly aligned with later probes.

### Tradeoffs Against Kind's Commitments

- **Capacity-over-exercise.** Recurrent PPO does not have the *capacity* to support Probes 3 and 4 even latently. Choosing it now means Probe 2 or Probe 3 will require substrate replacement, not extension. Kind's stance on capacity says: build the conduit even if not exercised. PPO does not have the conduit.
- **Plumbing test of what?** The whole point of Probe 1 is to verify the env→agent→mirror→observer loop. If the agent is PPO, the loop carries action-observation-reward-style signals — but the most distinctive *substrate-side* signals (posterior, prior, KL, dream rollouts) are absent. Probe 1 with PPO tests the loop minus the parts that matter most to later probes. That is a weak plumbing test for this project specifically.
- **Self-opacity.** Both PPO and RSSM admit Question 5's two-view design, but the boundary in PPO is much weaker (there is less internal structure to be opaque about). The opacity argument for RSSM is that opacity-by-design is meaningful when there is something substantive to be opaque about.

### Defensible Suggestion

**Replace recurrent PPO with a minimal RSSM at Probe 1.** The custom-minimal-RSSM-with-PlaNet-skeleton-and-DreamerV1-imagination-actor described in Question 1, with the actor objective described in Question 2, is the right substrate. The engineering cost is real but bounded (~1–2 weeks of careful work for a single experienced implementer) and pays off in not having to rebuild the substrate at Probe 2 or 3. PPO as Probe 1 baseline is a placeholder that misnames the question — it tests "can we get any agent through any environment with any logger" rather than "can we get the substrate's signature signals through the loop end-to-end."

If risk tolerance is low and a fallback is wanted, the **PPO-actor-on-RSSM-substrate** middle path keeps the world-model conduits while borrowing PPO's hyperparameter robustness for the actor. The world model trains the same way; only the actor's loss changes from "analytic-gradient through dynamics" to "PPO surrogate on intrinsic-reward returns from imagination". This loses some sample efficiency but gives up nothing structural.

**Honest uncertainty:** the literature does not directly compare "RSSM with PPO actor" to "RSSM with Dreamer analytic actor" with no extrinsic reward. Both are coherent. Probe 1 should pick one and commit; the choice does not foreclose later probes.

---

## Synthesis: A Coherent Probe 1

### The Combination

A defensible, internally coherent Probe 1 implementation that respects every one of Kind's commitments:

1. **Substrate:** custom minimal RSSM, PlaNet-skeleton (deterministic GRU `h_t` + continuous Gaussian stochastic `z_t`, posterior `q(z|h,o)`, prior `p(z|h)`), with a small CNN/MLP encoder, MLP decoder, free-bits KL regularizer borrowed from DreamerV3, no other DreamerV3 robustness machinery. No reward predictor. No continuation/discount predictor.
2. **Actor objective:** intrinsic-only, with a structural placeholder for EFE's pragmatic term. Latent-disagreement (Plan2Explore-lite, K=3–5 ensemble of one-step latent predictors) as the intrinsic signal, optimized via DreamerV1-style analytic gradients through latent imagination *or* a PPO surrogate on imagined returns. Uniform preference prior, contributing zero pragmatic value, kept in the loss as scaffolding for Probe 4+.
3. **Telemetry:** four streams — `agent_step`, `dream_rollout`, `replay_meta`, `world_event` — versioned with semver, columnar storage for the high-rate streams, full posterior and prior parameters logged per step, dream rollouts emitted at fixed cadence even though Probe 1 doesn't yet test them.
4. **Compute split:** Mac canonical, weights resident there always, replay buffer on Mac, trainer on Mac, dream-rollout process on Mac. Desktop runs the environment(s) only and streams transitions over a socket. Atomic checkpoints commit weights + replay + RNG + telemetry offsets together, with an environment-pause barrier protocol.
5. **Self-opacity:** `PolicyView` / `TelemetryView` interface separation enforced at module level. Actor reads only `(h_t, z_t)`. Mirror reads everything. Intrinsic signal computed in a module the actor does not import. Builder-event stream is separate and the agent process has no read handle on it.
6. **Replace recurrent PPO with the minimal RSSM at Probe 1.** Use a PPO actor on the RSSM substrate as a fallback only if Dreamer-style analytic-gradient actor training proves unstable on the chosen environment.

### Tensions Across Questions

- **Q2 vs Q5: the KL-as-reward problem.** If the actor's intrinsic signal is the RSSM's own posterior/prior KL (the cleanest reading of EFE's epistemic value at Probe 1 scale), then KL of Io's own internals is in Io's reward stream — a structural form of self-readout, even if the actor cannot introspect that this is what it is. Latent-disagreement (over auxiliary ensemble heads) sidesteps this because the disagreement is over heads that are not Io's policy state. The synthesis above resolves this by recommending disagreement, not posterior-KL, as the actor signal — but posterior-KL is still logged as telemetry that the *mirror* can read.
- **Q1 vs Q6: minimum-viable vs robustness machinery.** The Probe 1 RSSM should be minimal (no symlog, twohot, etc.) for clarity and to avoid premature commitment. But free bits is a single-line fix for a known failure mode; including it is principled. The line between "robustness machinery to skip" and "essential stability" is not crisp; the synthesis includes only free bits and skips everything else, but a Probe 1 that hits training instability may have to revisit.
- **Q1 vs Q2: reconstruction as decoder for mirror.** Pixel reconstruction (Dreamer family) is conducive to mirror legibility because dreams can be decoded into observations the mirror can read. JEPA-style predictors are more conducive to non-reward shaping and arguably purer. The synthesis chose reconstruction, not because of the shaping argument but because of the mirror argument — at Probe 1 scale, a decoder is cheap and the mirror needs something to look at.
- **Q4 vs Q6: training speed vs canonicality.** Mac-only training is canonical but slower than desktop CUDA training. RSSM training is more compute-intensive than PPO. If RSSM-on-Mac is too slow, Probe 1 risks failing as a plumbing test for unrelated reasons. The synthesis bets MPS performance is adequate at this scale, but a smoke test on day one is non-negotiable.

### Points Kind's Project Documents Seem Not Yet to Have Addressed

- **Builder-event ground truth schema.** Probe 4 needs to know *when* the builder perturbed; if Stream 4 is not specified now, Probe 4 will need to retrofit. The synthesis above puts it in Probe 1's telemetry schema explicitly, but Kind's documents (per the task description) do not seem to have specified this asymmetric logging arrangement.
- **Dream-rollout cadence and decoding policy at Probe 1.** Even though dreams are foundational from Probe 3, Probe 1 must emit them periodically to verify the conduit. The cadence and whether to decode imagined latents back into observations need to be spelled out before the schema freezes.
- **Environment-pause-and-drain barrier protocol** at checkpoint boundaries. "Atomic sync at checkpoint boundaries" implies but does not specify a barrier; Probe 1 should specify one.
- **The KL-as-self-readout question (Q5)** is a genuine architectural decision point that the project's stance on self-opacity does not fully resolve. The synthesis offers a defensible position (use disagreement, not posterior-KL, as the actor signal), but Kind's commitments could be read either way and the team should make this choice consciously.
- **What "minimal" means for the world-model reconstruction target.** Decoder kept (synthesis's choice) commits Probe 1 to pixel-or-feature reconstruction loss. Alternatives — predicting actions taken (DreamerPro), predicting value (MuDreamer), or pure JEPA-style feature prediction (Dreamer-CDP) — are reconstruction-free but require collapse-prevention machinery (batch norm, prototypes, stop-gradient placement) that the literature explicitly warns about. The synthesis chose reconstruction for simplicity and mirror-legibility; a future probe may want to revisit, and Kind's documents do not seem to have foreclosed the question.
- **What the mirror's reading actually looks like.** Probe 2's mirror is named in the constraints but its interface is not specified at this level. Probe 1's telemetry schema is designed to be mirror-ready in the sense that all the candidate fields are logged, but the mirror's actual computation — what it interprets as "something other than the builder's own reading" — is not yet pinned down. This is appropriately Probe 2's concern, but Probe 1's schema should not hardcode assumptions about what the mirror will need to compute beyond logging the obvious latent-distribution and dream-rollout fields.

### What This Document Is Not

Not a build plan. Not a recommendation between architectural families (the family is fixed). Not a benchmark-performance discussion. The synthesis above identifies a coherent set of choices that respect Kind's stance and exposes the conduits later probes need; the next step is to convert these inputs into a sequenced implementation plan with milestones, smoke tests, and acceptance criteria — that conversion is a separate piece of work.