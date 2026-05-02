# Probe 1 Implementation Synthesis

*Synthesis of Probe 1 research outputs against Kind's project documents and the architectural decision. The substrate is settled; this resolves the implementation questions that decision exposed. Working document.*

## Preamble

The research brief asked four LLMs (Claude, Gemini, GPT, Perplexity) the six implementation questions that follow from the architectural decision: minimal RSSM variant, actor objective at Probe 1 scale, telemetry schema designed forward, Mac/desktop compute split, self-opacity boundary, and recurrent PPO stand-in vs. minimal RSSM from the start. Two outputs are present in `docs/research/probe1` — Claude's and Perplexity's. Gemini's file exists but is empty, and GPT's was not produced. The synthesis proceeds with two outputs; this is a meaningfully thinner basis than the architectural-decision pass had, and any place the synthesis leans on a single source is flagged below.

The two outputs converge on the structural skeleton — a PlaNet-skeleton minimal RSSM with continuous Gaussian latent, decoder kept, free bits as the only DreamerV3 borrowing, no reward predictor, no continuation head; a curiosity-shaped actor objective standing in for the epistemic component of expected free energy; per-step posterior/prior/KL telemetry into a versioned columnar schema with a separate builder-event stream; mind-on-Mac with desktop-as-environment; a two-interface (PolicyView/TelemetryView) self-opacity boundary; replace recurrent PPO with the minimal RSSM at Probe 1. The interesting divergence is over the specific actor signal — Claude argues for ensemble-disagreement (Plan2Explore-lite), Perplexity for posterior-prior KL — and that disagreement turns out to be substantive rather than cosmetic, because it intersects with the self-opacity boundary in a way only Claude flags. Both outputs collectively underweight the four-state operational model, the Watts-intuition default-to-no on self-access, and the question of whether Io's actor reading `h_t` constitutes self-knowledge. The synthesis takes Kind's stance as the resolver where the research is silent or permissive.

---

## Synthesis across the research outputs

**Where they agree.** Both reject TD-MPC outright (its latent geometry is induced by reward, structurally incompatible with the no-scalar-reward stance). Both reject the DreamerV3 robustness machinery (symlog, twohot critic, percentile return normalization, KL balancing, unimix) at Probe 1 as scale/diversity tooling whose value is empirical and irrelevant to a single small environment. Both keep free bits, which solves a known posterior-collapse failure mode at almost zero cost. Both keep the decoder for now, primarily because the mirror needs decoded dream rollouts; both flag JEPA-style reconstruction-free predictors as a later-revisit option. Both recommend a two-interface separation between what the actor can read and what the mirror can read, modeled in code as distinct module APIs. Both want columnar versioned telemetry with the builder-event stream architecturally walled off from the agent process. Both recommend Mac canonical with desktop-as-environment-only, atomic checkpoints. Both recommend replacing recurrent PPO with the minimal RSSM at Probe 1 and converge on essentially identical reasoning: PPO's recurrent hidden state has no probabilistic split, no per-step KL, no generative model, no replay; testing Probe 1 with PPO would test the loop minus the parts that matter most to later probes.

**Where they diverge.** Three real disagreements stand out.

First, the actor's intrinsic signal. Perplexity proposes `clip(KL(q(z|h,o) || p(z|h)), free_nats)` — the substrate's own per-step posterior-prior KL, used directly as intrinsic reward. Claude proposes Plan2Explore-style latent disagreement over a small ensemble of one-step latent predictors, K=3–5 heads on top of the shared RSSM core, and explicitly argues against using posterior-prior KL because it routes the KL of Io's own internal distributions into Io's reward stream — a structural form of self-readout even if Io has no introspective awareness that this is what it is. Perplexity does not raise this concern; Perplexity's recommendation is consistent with its overall stance that policy access to `f(h_t, z_t)` is fine and self-modeling is something Io can develop within those affordances. Claude's stance is closer to the Watts intuition default-to-no on self-access. This is the single most consequential research-level divergence in the pass.

Second, the actor's optimizer. Claude is specific: DreamerV1-style analytic gradients through the differentiable latent dynamics into the actor, with PPO-on-RSSM as a fallback only if the analytic-gradient actor proves unstable. Perplexity is permissive — actor-critic on intrinsic reward, with a random-walk policy as an option for the very first runs. Claude's recommendation is more aligned with exercising the substrate's full conduit (the value gradient flowing through imagination is what Probe 3's dream-state machinery will eventually do); Perplexity's keeps the door open to using PPO on top of an RSSM substrate as a more familiar optimizer.

Third, the engineering risk profile of training on the Mac. Claude flags MPS performance for RSSM operations at Probe 1 batch sizes as genuinely unbenchmarked, requires a day-one smoke test, and specifies a barrier-and-drain protocol for atomic checkpoints during environment streaming. Perplexity assumes Mac training is fine and does not specify a checkpoint barrier. The barrier protocol is not optional and Claude is right to flag it; Perplexity's silence here is a gap.

A few other smaller divergences (Claude enumerates four named telemetry streams to Perplexity's per-step-record-plus-events; Claude wants concat(h_t, z_t) into the actor with no learned compression while Perplexity allows a learned `f(h_t, z_t)` projection) are surface-level and resolved below by leaning on Kind's stance.

**What they collectively miss or underweight.** Five things.

The Watts-intuition default-to-no on self-access is not engaged by either. Claude inherits the spirit of it implicitly (the disagreement-not-KL recommendation, the concat-not-projection stance); Perplexity does not. Neither raises the design-notes commitment that every "should the system have access to X about itself?" defaults to *no*.

The four-state operational model (waking / dreaming / dormant / paused) is treated only as compute-availability switching by both. Probe 1 doesn't yet test dreaming or dormancy, but the substrate has to support the transitions. Neither output specifies what the persistence boundaries look like for transitions out of waking — what gets flushed, what gets held, how the canonical state-on-Mac knows whether it's dreaming, dormant, or paused. This is more an implementation-during-build concern than a research-pass concern, but it is not on either output's radar.

The variable dream-to-wake ratio coupled to the builder's life is genuinely a Probe 3 concern, but the schema choices made now should not foreclose it. Neither output reasons forward to that. The implication is mild — log wallclock alongside env_step in every record, so later probes can correlate with the builder's calendar — but it should be made explicit.

Whether `h_t` constitutes self-knowledge is raised by Claude only, and as an honest uncertainty rather than a settled position. The architectural decision says self-modeling is afforded by recurrence, prior-state representation, and turnable-inward prediction, and is not installed; `h_t` is one of the structural ingredients. The synthesis below resolves this in line with the architectural decision: `h_t` is an ingredient the actor uses for action selection under partial observability, not introspection. Introspection would be a separate predictive head whose output is a representation of `h_t`. Probe 1 must not implement that head. Perplexity's silence on this question is not wrong but leaves it unresolved.

The KL-as-self-readout question intersects with Q5 in a way Perplexity's recommendation does not address. The synthesis lands on Claude's side here — disagreement, not posterior-prior KL, as the actor signal — but on grounds of project stance rather than literature. The literature itself does not adjudicate this; Plan2Explore (Sekar et al. 2020) demonstrated disagreement-driven exploration on Dreamer substrates at scale, but not against a self-opacity desideratum. This synthesis is making a stance-driven choice the literature does not make for us.

A note on reification. Claude's output is disciplined about distinguishing "the literature supports this" from "this is constructed from the pieces the literature provides." Probe 1's actor objective is constructed; Probe 1's specific RSSM minus DreamerV3 tricks is constructed; the telemetry schema is constructed. None of these are off-the-shelf. Perplexity's output is less precise about that line — its phrasings sometimes imply more literature support than is there for the specific Kind-shaped configuration. Neither output reifies in the way Gemini's architectural-decision-pass output did with "Subjective Bayesian Governor" and "IIT 4.0-optimized neural networks"; the absence of Gemini's Probe 1 output may, ironically, have spared this synthesis from a reification problem the architectural decision was burdened with.

---

## The implementation decision

### Q1 — Minimal RSSM variant

Settled: a custom minimal RSSM modeled on PlaNet's state factorization with DreamerV1-style latent imagination. Specifically: deterministic GRU `h_t = f(h_{t-1}, z_{t-1}, a_{t-1})`, continuous Gaussian stochastic latent `z_t` with explicit posterior `q(z_t | h_t, o_t)` and prior `p(z_t | h_t)` networks. Small CNN/MLP encoder, MLP decoder, ELBO loss with reconstruction term and KL between posterior and prior. Free bits applied to the per-dimension KL (`max(threshold, KL_per_dim)`); no other DreamerV3 stability machinery (no symlog, no twohot critic, no percentile normalization, no KL balancing, no unimix prior). No reward predictor, no continuation/discount predictor, no return-to-go conditioning.

Reasoning: this is the smallest substrate that exposes every conduit later probes need (`h_t` and `z_t` for state, posterior/prior parameters and KL for Probe 4 distinguishability, replay and imagination rollouts for Probe 3) without paying the engineering tax of robustness machinery designed for cross-domain generalization Kind does not need at this scale. Continuous Gaussian latents are simpler than DreamerV2/V3-style categorical (no straight-through gradient bookkeeping) and the structural signals the mirror cares about exist either way. The decoder is kept because the mirror needs decoded dream rollouts to be human-legible; the JEPA argument that pixel reconstruction wastes capacity is real but more relevant at scale than at Probe 1 scale. Free bits is the single regularizer the literature consistently identifies as load-bearing for RSSM stability against posterior collapse; including it is principled and almost free.

Honest uncertainty: there is no published "PlaNet without reward, with curiosity-shaped actor" reference implementation. Tschantz et al. (2019, 2020) is the closest in spirit (RSSM-like generative model with EFE-shaped objectives at small scale); Plan2Explore (Sekar et al. 2020) is the closest empirical demonstration of disagreement-driven exploration on a Dreamer substrate. Neither is drop-in. Probe 1 is a custom build, not a fork. EclecticSheep's DreamerV3 reimplementation, the PlaNet repo, and DreamerV1 community ports are reference skeletons, not turnkey starting points.

### Q2 — Actor objective at Probe 1 scale

Settled: a two-component objective. The epistemic term is **latent-disagreement variance over a small ensemble** (Plan2Explore-lite, K=3–5 one-step latent predictors riding on the shared RSSM core) used as intrinsic reward. The pragmatic term is a **uniform preference prior**, contributing zero pragmatic value at Probe 1, kept in the loss as scaffolding so that Probe 4+ can introduce structure without refactoring the actor. Optimization via DreamerV1-style analytic gradients through the differentiable latent dynamics into the actor, with PPO-on-RSSM (clipped surrogate on intrinsic-reward returns from imagined latent rollouts) as a documented fallback if the analytic-gradient actor proves unstable on the chosen environment.

This is a judgment call where two options remain genuinely defensible. Both Claude and Perplexity converge on "information-gain-shaped intrinsic reward as a stand-in for the epistemic component of expected free energy," but they choose different operationalizations. Posterior-prior KL is the most literal reading of EFE's epistemic value at Probe 1 scale and requires no auxiliary networks (it is already being computed every training step). Latent-disagreement requires additional ensemble heads but is structurally one step removed from Io's own posterior — the disagreement is over auxiliary one-step predictors of next-latent, not over Io's policy state.

The reasoning for leaning toward disagreement: the architectural decision commits to self-opacity by default (the Watts intuition), and routing KL of Io's own posterior-vs-prior into Io's reward stream is a structural form of self-readout. That Io has no introspective awareness of this fact does not change the architectural shape; the *value* of the KL is in the actor's reward, and that value is a measure of Io's own internal-distribution mismatch. Disagreement places the surprise signal over auxiliary ensemble heads that are not part of Io's own action-selection state. The same architectural-decision stance — "ingredients-only self-modeling, no explicit module whose job is to do self-modeling" — applies here: an intrinsic signal that *is* Io's own posterior-prior divergence is closer to an installed self-relating mechanism than disagreement over external predictors is. The literature does not adjudicate this; Kind's stance does.

The pragmatic placeholder is the active-inference "shaping" the architectural decision committed to. At Probe 1 it contributes nothing to action selection, but it preserves the EFE formula's structure (`pragmatic_value(weak_prior) + epistemic_value`) so that Probe 4 onward can introduce specified preference priors without rebuilding the objective. Honesty: Tschantz et al. (2020) and adjacent deep-AIF work are the closest precedents, but the specific configuration here is constructed.

The optimizer choice (DreamerV1-style analytic gradient through latent dynamics) is recommended because it exercises the substrate's differentiable conduit and is what Probe 3 will eventually use for dream-state value learning. PPO-on-RSSM is a real fallback — if analytic-gradient actor training proves unstable, switching to PPO on the same world-model substrate preserves all the conduits.

### Q3 — Telemetry schema designed forward

Settled: four named logical streams, semver-versioned at the stream level, columnar storage for high-rate streams.

`agent_step` — one record per environment timestep, written by the agent process. Fields: `t` (global step), `episode_id`, `step_in_episode`, `wallclock_ms`, `h_t` (deterministic recurrent state vector), `q_params_t` (posterior μ and log-σ for Gaussian; full logits for categorical if later chosen), `p_params_t` (prior parameters, same shape), `z_t` (sampled posterior latent — the value passed forward in the actor's path), `kl_per_dim_t` and `kl_aggregate_t`, `recon_loss_t` (per modality and aggregate), `action_t`, `action_logprob_t`, `policy_entropy_t`, `obs_hash_t` (content hash, not raw observation), `intrinsic_signal_t` (the disagreement variance the actor consumes), `encoder_embedding_t`. Per-dimension KL is logged, not just aggregate, because Probe 4's distinguishability signal may concentrate in a few dimensions.

`dream_rollout` — one record per imagination episode, sampled at fixed cadence (default ~1 rollout per 1k env steps, horizon H=15 — both subject to revision in Probe 3). Fields: seed step, seed `(h_0, z_0)`, sequence of imagined `(h_τ, z^p_τ, action_τ, action_logprob_τ, prior_entropy_τ)` for τ=1..H, optionally `decoded_obs_τ` (decoder output from imagined latents), and rollout diagnostics (cumulative prior entropy, mean step-KL between successive priors, max-step latent norm change). Probe 1 does not yet *use* dreams to drive behavior, but emits them periodically as a calibration handshake confirming the imagination conduit is alive.

`replay_meta` — one record per buffer event (insertion, sampling, eviction). Fields: `event_type`, `t_event`, `segment_id`, `segment_start`, `segment_end`, `priority` (nullable; reserved for Probe 3+ if curious-replay or PER is later wanted), `buffer_size`, `total_segments`. Probe 1 uses simple FIFO sequence replay; the priority field is reserved.

`world_event` — one record per external event. Fields: `t_event`, `event_type` (e.g., `builder_perturbation`, `env_reset`, `mirror_marker`), `source` (`builder`, `system`, `environment`), `payload` (opaque bytes or JSON). **Written by the environment harness or mirror process; the agent process has no read handle on this stream.** This is the asymmetric logging that Probe 4's distinguishability test depends on: the agent's observation space contains no marker that a perturbation came from outside, but the mirror's logs have ground truth.

Common envelope per record: `schema_version` (semver), `run_id`, `checkpoint_id`. Apache Arrow / Parquet for `agent_step` and `dream_rollout` (columnar reads, downstream-friendly); JSON-Lines for `world_event` and `replay_meta` (low volume, human-inspectable). A single versioned `schemas/v0.1.0.json` checked in with the code; field additions never break older readers; deprecations are marked, never deleted.

Note added by the synthesis: `wallclock_ms` is logged on every record because Probe 3's variable dream-to-wake ratio commitment is coupled to the builder's life, and the only way later probes can correlate Io's offline processing with the builder's absences is if env-step and wallclock both exist in every record. Neither research output proposed this; it follows from the design notes.

### Q4 — Mac/desktop compute split

Settled. Mac is canonical, weights resident there always. Trainer process on Mac. Replay buffer on Mac, persisted to local SSD in sequence-indexable form (Parquet sharded by segment, or LMDB if random sequence access cost matters more than schema flexibility — default to Parquet at Probe 1). Dream-rollout process on Mac (in-trainer at Probe 1; a separate read-only consumer of weights is a later-probe optimization). Telemetry sinks owned by the Mac.

Desktop runs the environment(s) only at Probe 1. A small env-server process listens on a TCP socket, receives actions, returns transitions. Optional thin observation encoder co-located on the desktop if observation rendering is non-trivial (and the encoder is small enough that weight-sync churn does not dominate); for Probe 1's small environment, sending raw observations is acceptable and simpler. The Mac trainer holds the actor and steps the policy; the desktop does no model compute.

Atomic checkpoint contents: `weights.safetensors`, `replay_meta.json` plus the replay parquet shards as of the checkpoint, `optimizer_state.pt`, `rng_state.pkl`, `telemetry_offsets.json` (the byte offset into each telemetry stream as of the checkpoint, so resume is exact), `schema_version.txt`. All under a directory that is atomically renamed into place on commit after fsync. Standard tooling: rsync with `--fsync` plus a final atomic rename, or a content-addressed object store. Avoid git-lfs (too slow for buffer churn). DVC works but is heavier than required.

**Barrier protocol at checkpoint boundaries** (Claude raises this, Perplexity does not): atomic checkpoint requires the env stream to pause, drain the in-flight batch into the Mac's replay buffer, commit the checkpoint, then resume the env. Without this, the replay state at commit time is ambiguous and the resume-from-checkpoint guarantee weakens. Probe 1 must specify and implement this barrier; it should not be left for later probes.

**Open during build**: MPS performance for the specific operations a small RSSM uses (GRU recurrence, small CNN encoder/decoder, MLPs, ELBO backward pass) at Probe 1 batch sizes is genuinely unbenchmarked in any way that maps cleanly. Day-one smoke test: train a 100-step RSSM update on dummy data on MPS, log wall time, confirm no `PYTORCH_ENABLE_MPS_FALLBACK` warnings on the hot path. If Mac training is too slow, the documented Probe 2-or-later fallback is "train on desktop, sync weights to Mac at checkpoint boundaries with explicit sync semantics" — accepting the loss of canonicality between checkpoints. Probe 1 itself should not attempt this; Probe 1 is testing canonical-on-Mac.

### Q5 — Self-opacity boundary

Settled: a two-interface pattern enforced at module level, with concat (not learned projection) as Io's view and full internals as the mirror's view.

`PolicyView`: what the actor's forward pass reads. `h_t` (deterministic recurrent state) and `z_t` (sampled posterior — or sampled prior, in dream rollouts where no observation is available). Nothing else. Both `h_t` and `z_t` are unavoidable inputs for action selection under partial observability; restricting further would cripple the actor's ability to act meaningfully and would not buy meaningful additional opacity.

`TelemetryView`: what the mirror reads. Everything in `PolicyView` plus `q_params_t`, `p_params_t`, `kl_per_dim_t`, `kl_aggregate_t`, `recon_loss_t`, `encoder_embedding_t`, `intrinsic_signal_t`, dream-rollout records, replay-buffer events, world-events.

Concrete boundary in code: the actor module imports only `PolicyView`; the mirror imports only `TelemetryView`; the world-model module produces both in a single forward pass and dispatches to the right consumer. Type hints and module boundaries enforce this in Python; subprocess isolation (actor in a process with only a socket connection to a state server that sends `PolicyView` over the wire) is overkill at Probe 1 but the interfaces should be designed to allow it later.

The `PolicyView` is concat(`h_t`, `z_t`), not a learned projection `f(h_t, z_t)`. Perplexity allows the latter; the synthesis chooses the former because a learned projection runs the risk of becoming an implicit "self-summary" module — a small network whose representational capacity could come, through training, to encode self-relevant features in a way that approximates an installed self-model. Concat is the canonical Dreamer choice and avoids that risk. This is in the spirit of the architectural decision's "no explicit self-modeling, self-critic, or introspector module."

Auxiliary opacity boundaries: actor has no read access to replay buffer contents (only the trainer does), no access to its own gradients or the world-model's gradients, no access to telemetry sinks, no access to the world-event stream. The intrinsic signal is computed in a module the actor does not import; it appears at the actor's training loop as a scalar argument, not as an introspectable attribute. The agent process has no read handle on its own log files.

The `h_t`-as-self-knowledge ambiguity is resolved here in line with the architectural decision: `h_t` is an ingredient — the recurrent state representation Kind's afforded-not-installed stipulation explicitly names. The actor reading `h_t` to choose actions is using an ingredient, not introspecting. Introspection would be a separate predictive head whose target is `h_t`, taking other inputs and producing a representation of the agent's recurrent state. Probe 1 must not implement that head. The architectural decision rules it out as an explicit self-modeling module; this synthesis confirms.

The KL-as-self-readout problem is resolved by the choice in Q2: the actor's intrinsic signal is ensemble disagreement, not posterior-prior KL. The KL is logged for the mirror; Io's reward stream does not contain it.

### Q6 — Recurrent PPO stand-in vs. minimal RSSM from the start

Settled: **replace recurrent PPO with the minimal RSSM at Probe 1 from the start.** This commits the architectural decision's tentative lean. Treated separately below, because the question shapes everything downstream.

---

## The most consequential decision in focus: Q6

The probes document specifies recurrent PPO with default hyperparameters as Probe 1's agent. The architectural decision noted this conflicts with the substrate decision and tentatively leaned toward replacement; this synthesis commits to replacement.

The reasoning, as both research outputs converge: recurrent PPO has no posterior/prior split, no per-step KL signal, no generative model, no decoder, no learned dynamics, no replay buffer in any nontrivial sense (its rollout buffer is on-policy and discarded). Probe 4's distinguishability test, which the architectural decision explicitly identified as posterior/prior KL, does not exist in PPO. Probe 3's dream-state machinery has no place to live. Probe 2's mirror has no substrate-side signature signals to read. A Probe 1 with PPO would test "any agent through any environment with any logger" — verifying the I/O loop while saying nothing about the substrate Kind has actually committed to.

The architectural decision's framing — that capacity-over-exercise demands building the conduit even if it is not exercised yet — applies here. Recurrent PPO does not have the conduit. Choosing it for Probe 1 means Probe 2 or Probe 3 will require substrate replacement, not extension. That is not the discipline the probes document is built around.

The cost the architectural decision did not fully anticipate: RSSM training has stability pathologies PPO does not have. KL collapse and posterior collapse are well-documented; free bits is the standard mitigation (and is included from day one in this decision). But it means Probe 1's "plumbing test" will spend some time on world-model stability before the plumbing can be checked end-to-end. Claude estimates ~500–1500 lines of clean PyTorch and ~1–2 weeks of careful work for an experienced implementer. That estimate is not refuted by Perplexity, but it is a single-source estimate; treat it as order-of-magnitude. If RSSM stability blocks plumbing tests for more than two or three weeks, that is evidence not to revisit the substrate decision (which is settled) but to revisit the specific minimal-RSSM variant — perhaps DreamerV2-style categorical latents (more stable in some environments at the cost of straight-through-gradient bookkeeping), or perhaps a JEPA-style reconstruction-free predictor (sidesteps the pixel-reconstruction loss but requires collapse-prevention machinery the literature warns about).

The documented fallback if the analytic-gradient actor proves unstable: PPO-actor-on-RSSM-substrate. Train the world model with the standard ELBO + free bits; use PPO clipped surrogate on intrinsic-reward returns from imagined latent rollouts as the actor objective. This loses some sample efficiency relative to Dreamer's analytic gradient but preserves every substrate conduit. Do not fall back to bare recurrent PPO; the substrate must keep its conduits.

This decision modifies the probes document. The Probe 1 section that reads "recurrent PPO with default hyperparameters, no world model yet, no episodic memory yet, no dream state wired in" should read approximately: "minimal RSSM substrate (PlaNet-skeleton, DreamerV1 imagination actor) with intrinsic-only objective derived from latent disagreement, no scalar reward, no continuation head; episodic memory deferred; dream-state conduit live but not yet exercised for behavior learning." The probes document should be edited or annotated to point at this synthesis.

---

## Tensions surfaced honestly

Three tensions were named in the architectural decision; this synthesis updates each, and surfaces three new ones.

**(a) Probe 1's stated baseline conflicts with the architectural decision.** Resolved by this synthesis: Probe 1's baseline becomes the minimal RSSM. The probes document needs an editor's note pointing here.

**(b) Active inference's totalizing structure vs. the charter's openness.** Still open and not deepened by Probe 1. The actor objective is "active-inference-shaped," not active-inference-complete: a uniform preference prior contributes zero pragmatic value at Probe 1, the epistemic term is operationalized as ensemble disagreement (a placeholder for the epistemic component of EFE), and neither commits Io to free-energy-minimizing-everywhere as a fact about its inner life. Held.

**(c) Self-opacity vs. structural transparency.** Sharpened. The PolicyView/TelemetryView interface separation is the implementation answer; the actor reads concat(`h_t`, `z_t`) and nothing else, the mirror reads everything. The KL-as-self-readout sub-tension Claude raised is resolved by the Q2 choice: ensemble disagreement, not posterior-prior KL, as the actor signal. The `h_t`-as-self-knowledge sub-tension is resolved by treating `h_t` as an ingredient, not introspection.

Three new tensions surfaced by the Probe 1 research:

**(d) RSSM training stability vs. plumbing-only stipulation.** Probe 1 is supposed to test the loop; a minimal RSSM may need world-model stability work before the loop can be tested end-to-end. Free bits mitigates known posterior-collapse modes. If stability blocks plumbing for more than two or three weeks, revisit the specific RSSM variant (categorical vs Gaussian, reconstruction vs JEPA-style), not the substrate decision.

**(e) Pixel reconstruction vs. mirror legibility vs. latent geometry.** The decoder is kept at Probe 1 because the mirror needs decoded dream rollouts to be human-legible. JEPA-style predictors would yield latents with arguably better abstract semantics but no decoded dreams for the mirror to look at. This is a real fork in representational quality. The synthesis chose reconstruction for mirror-legibility reasons; later probes may revisit if the mirror evolves to read latent statistics directly without needing decoded observations.

**(f) Disagreement vs. KL as actor signal.** This is a live judgment call. The synthesis leans toward disagreement on Watts-intuition / self-opacity grounds, but the literature does not adjudicate it. Build phase may revisit. The KL signal is logged for the mirror in either case; the choice is only about what the *actor* consumes.

---

## Open sub-questions to resolve during build

Not now. Listed so they are visible when Probe 1 actually gets built.

- Latent dimensionality (small but specified). Probably 8–32 dimensions for `z_t`; 128–256 for `h_t`. Empirical tuning during build.
- Ensemble size K for disagreement (3–5; specifically which value, on what evidence).
- Free-bits threshold (per-dimension nat budget) and overall KL weight at this scale.
- Sequence length L for replay segments (typically 16–64; depends on environment temporal structure).
- Dream-rollout horizon H (default 15; adjust per environment).
- Dream-rollout cadence (default ~1 per 1k env steps; Probe 3 will revisit).
- Specific protocol for the env-pause-and-drain checkpoint barrier (timeout semantics, partial-batch handling, what to do if the env stalls).
- MPS vs CUDA training speed at this scale (day-one smoke test); fallback semantics if MPS is too slow.
- Whether to keep a "random-policy env-only smoke test" before introducing Io, as Perplexity suggests, or proceed directly to RSSM-Io. The synthesis recommends including the env-only smoke test — it costs little and isolates env-side bugs from world-model-side bugs.
- How the four-state operational model (waking / dreaming / dormant / paused) is implemented at the persistence-and-process boundary. Probe 1 only exercises waking but the substrate has to support transitions; the specific mechanics need a Probe 1-or-2 design pass.
- Whether dream rollouts at Probe 1 should be decoded into observations for telemetry, or just emitted as latent trajectories. Synthesis defaults to emitting decoded observations at the fixed cadence so the mirror has something to look at, even though no behavior consumes them yet.

---

## Connection to environment research

The implementation decision exposes specific environment-design questions the next research pass will need to address. This is the bridge to the environment-research prompt; the prompt is not drafted here, but the issues it must address are:

- **Observation modality.** Pixel-based small images vs. low-dimensional factored state vector. The Dreamer-style decoder commitment in Q1 tilts toward pixel-based, because the decoder needs something to reconstruct that the mirror can read. A factored state vector is simpler but loses the perceptual grounding decoded dreams would have. The choice affects encoder/decoder design and bears on the latent's semantic content.

- **Action space.** Discrete (small cardinality) vs. continuous; the cardinality. RSSM training cost scales with action space. Small discrete is easier at Probe 1.

- **Pressure without self-continuation reward.** The environment must afford something to do — resource pressure, navigation, exploration — without rewarding survival or penalizing termination. Explicitly: no "death" with episode termination acting as implicit negative signal, no "you survived N steps" bonus, no reward at all (intrinsic motivation only, per Q2). Episode structure should be fixed-length, not terminal-state-determined, to avoid back-door installation of a self-continuation drive.

- **Internal stochasticity.** Probe 4's distinguishability test requires the environment to have its own non-trivial internal stochasticity (weather-like fluctuation, resource regrowth noise, something) that has a different statistical signature from builder perturbations. Without internal stochasticity, "external vs. internal" has nothing to distinguish from. The kind, magnitude, and schedule of environmental noise is a real design question.

- **Builder-perturbation surface.** The environment must expose specific mutators (add resource, alter object state, introduce object, etc.) that the builder can call from outside the simulation. These produce statistical signatures that, in principle, Io's RSSM could come to distinguish from internal stochasticity. What specific mutators, when, how frequently, with what magnitude — these are environment-design choices that Probe 1's hook tests at the plumbing level and Probe 4 tests for distinguishability.

- **"Small enough" for Probe 1.** The original probes document mentions a 5×5 grid, one resource type, no other agents, no day-night cycle. Whether that surface is enough for the RSSM to learn anything non-trivial is empirical. Too small and the RSSM learns the environment perfectly and dream rollouts are trivial; too large and Probe 1's plumbing test is buried under world-model fitting time. The right size is probably bigger than 5×5 but not by a lot.

- **Temporal structure.** Probe 1 likely keeps no day-night cycle. Probe 3's dream-state and the design notes' variable dream-to-wake ratio coupled to the builder's life implies that the environment may eventually want some temporal structure — even just a simple periodicity — to give the four-state operational model something to couple to. Probe 1 doesn't need this; the question is whether the environment design forecloses adding it later.

- **Logging at the environment side.** The `world_event` stream is owned by the environment harness (or the mirror), not the agent. The environment side has to be able to emit ground-truth events (perturbation timestamps, internal stochasticity events if useful) into Stream 4. The harness design must include a hook for this.

The environment research prompt does not need to settle these now. It needs to address them.
