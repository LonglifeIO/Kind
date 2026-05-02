Io’s Probe 1 wants the smallest thing that is actually a world‑model agent: an RSSM whose latents, KLs, and replay/dream hooks are real, but with minimal capacity and stripped‑down objectives so you can debug plumbing before getting entangled with scale or full active inference. Below, each question in turn, then a synthesis of a coherent Probe 1 combo.

Question 1 — Minimal RSSM architecture
1. What this is really asking
You need the simplest recurrent state‑space model that:

Has the same structural ingredients as the later probes (deterministic + stochastic state, prior vs posterior, KL, imagination rollouts).

Is small enough to implement and debug quickly.

Does not lock you into full DreamerV3 scale or a JEPA tooling ecosystem you don’t yet have.

2. Concrete options from the literature
All candidates share the basic RSSM pattern: a deterministic recurrent state 
h
t
h 
t
​
  and a stochastic latent 
z
t
z 
t
​
 , with a transition prior 
p
(
z
t
∣
h
t
,
a
t
−
1
)
p(z 
t
​
 ∣h 
t
​
 ,a 
t−1
​
 ) and an observation‑corrected posterior 
q
(
z
t
∣
h
t
,
x
t
)
q(z 
t
​
 ∣h 
t
​
 ,x 
t
​
 ).

PlaNet‑style RSSM + MPC planning.

Uses an RSSM trained via ELBO with pixel reconstruction; planning is via CEM directly in latent space, no separate actor.

Pros: very small, clear separation of “world model” and “controller”; imagination rollouts are already core.

Cons: online planning adds complexity and compute; not aligned with your long‑term “agent with learned policy” picture.

DreamerV1‑style RSSM, continuous latents.

RSSM like PlaNet, plus decoder for pixel reconstruction and a reward model; actor‑critic learns from imagined trajectories in latent space.

Latents split into deterministic 
h
t
h 
t
​
  and Gaussian stochastic 
z
t
z 
t
​
 ; actor and critic consume 
(
h
t
,
z
t
)
(h 
t
​
 ,z 
t
​
 ).

DreamerV2‑style RSSM, discrete latents.

Same high‑level RSSM, but stochastic state is a vector of categoricals (e.g., 32×32), which improves multi‑modal modeling and stability in Atari‑like regimes.

KL between prior and posterior is central to the model loss; discrete latents tend to give more structured representations at scale.

DreamerV3 “bag of tricks”.

Same basic RSSM as V2 with symlog scaling, KL balancing, and other normalization/stability tricks to make one hyperparameter set work across many domains.

Pros: robustness at scale; cons: overkill for Probe 1, implementation more elaborate.

Decoder‑free / JEPA‑style world models.

I‑JEPA and V‑JEPA train an encoder + predictor to predict masked representation patches, not pixels; this shapes latents semantically and weakens pixel‑level constraints.

Dreamer‑CDP and related work add a JEPA‑style predictor to Dreamer, operating on compact deterministic states to achieve reconstruction‑free world models with Dreamer‑level performance.

TD‑MPC‑style implicit latent dynamics.

TD‑MPC/TD‑MPC2 learn an implicit, decoder‑free task‑oriented latent dynamics model and plan in latent space with a terminal value function.

Excellent for control, but less of a textbook RSSM: the latent’s semantics are tightly coupled to value optimization.

Bare‑bones custom RSSM.

A minimal “Dreamer‑like” RSSM: GRU backbone, small Gaussian latent, simple decoder; no heavy regularization tricks.

You can take a reference implementation (e.g., DreamerV2 codebases) and scale it down aggressively.

3. Tradeoffs against Kind’s commitments
Against your specific stance:

Dreamer‑family vs JEPA‑only.

Dreamer‑style ELBO with pixel reconstruction gives you a clear, verifiable link between latent state and sensory space; mirror can use reconstructions and reconstruction loss as sanity checks.

JEPA‑only models weaken that link: they deliberately avoid pixel reconstruction to encourage abstract semantics. That is great for representation quality but makes “mirror reads telemetry” less grounded early on.

Complexity vs plumbing.

DreamerV3’s symlog, KL balancing, etc. solve stability at scale across diverse domains. You don’t need that bag of tricks to test whether KL is logged correctly and dream rollouts execute.

Task‑centric latents (TD‑MPC) vs task‑agnostic world model.

TD‑MPC latents are explicitly optimized for return and MPC performance. That leans toward “capability‑first” and reward optimization, which runs against your “no benchmark race, no scalar reward” charter.

Discrete vs continuous latents.

Discrete latents (DreamerV2) are a good inductive bias for non‑smooth dynamics and can make KL/predictive structure clearer, but they add implementation complexity (straight‑through gradients, categorical sampling).

For plumbing, Gaussian latents are simpler; the structural signals the mirror cares about (prior/posterior KL, latent drift) exist either way.

4. Defensible suggestion
For Probe 1, a “DreamerV1‑lite” RSSM is a good fit:

Deterministic GRU backbone 
h
t
h 
t
​
  + low‑dim Gaussian latent 
z
t
z 
t
​
 , with explicit prior and posterior networks.

Simple pixel (or low‑dim observation) decoder so ELBO is well‑defined and visually debuggable.

No DreamerV3‑style tricks; just basic KL term with free nats and reconstruction loss as in standard Dreamer implementations.

Small hidden sizes and short rollout horizons tuned for your test env, not for Atari‑scale performance.

This architecture exercises all the plumbing you care about—RSSM latents, prior/posterior KL, replay, and imagination—without locking you into a specific large‑scale variant or a JEPA‑only ecosystem. You can introduce JEPA‑style prediction heads (or move to Dreamer‑CDP) in later probes once the telemetry and mirror path are trusted.

Question 2 — Actor objective at Probe 1 scale
1. What this is really asking
You need a behavior‑generating objective that:

Produces non‑degenerate trajectories so world‑model and telemetry plumbing can be exercised.

Respects “no scalar external reward maximizing, no self‑continuation prior”.

Is a reasonable stepping stone toward expected free‑energy minimization, but doesn’t require full deep active inference machinery yet.

2. Concrete options from the literature
Full active inference / expected free energy (EFE).

Active inference defines behavior as minimizing expected free energy, combining risk (preference violation) and ambiguity (epistemic value) in a single objective.

In continuous, long‑horizon settings this decomposes into variational free energy (perception) plus expected free energy over future trajectories (action). Implementations exist but are still computationally demanding and mostly tested in small or discrete domains.

Intrinsic motivation via prediction error or novelty.

Curiosity methods treat intrinsic reward as prediction error of a dynamics or representation model (ICM, forward model error, world‑model reconstruction loss, etc.).

Latent world‑model based exploration uses errors or uncertainty in the latent space as intrinsic rewards.

Intrinsic disagreement.

Ensemble‑based disagreement (e.g., InDRiVE) uses variance between world‑model ensemble predictions as a purely intrinsic exploration signal within a Dreamer‑like framework.

This yields task‑agnostic latent representations and exploration without external reward, then allows zero/few‑shot downstream task conditioning.

Minimum prediction‑error policies.

Policy tries to minimize reconstruction or dynamics prediction error directly (e.g., “keep the world familiar”). This is close to minimizing variational free energy with very simple priors.

Hybrid policy gradient with strong entropy/curiosity.

Conventional actor‑critic whose reward is a weighted sum of intrinsic terms (prediction error, novelty) and perhaps sparse extrinsic signals, plus strong entropy regularization.

This covers most “RL but without task rewards” settings and is well‑trodden in code and training behavior.

3. Tradeoffs against Kind’s commitments
Full EFE now vs later.

Implementing expected free energy in a Dreamer‑style world model for continuous control is at the edge of current research; most work either uses small discrete models or specialized setups.

For Probe 1, the risk is spending your time debugging approximate EFE gradients instead of the plumbing the probe is meant to test.

Curiosity / prediction error as EFE proxy.

Epistemic terms in EFE boil down to expected information gain or reduction in uncertainty. Intrinsic rewards based on prediction error or latent surprise are a straightforward operationalization of “go where the model is wrong”.

Using RSSM’s own prior/posterior KL as an intrinsic reward aligns especially well: KL already quantifies surprise and is central to the world‑model loss.

No self‑continuation drive.

You can keep episode length fixed and avoid any scalar reward terms that depend on survival, step count, or “not dying”, which avoids giving Io an implicit self‑preservation prior. This is easy with curiosity‑only rewards and fixed horizons.

Transparency for the mirror.

Intrinsic rewards grounded in KL(post||prior) or reconstruction error are easy for the mirror to interpret later (Probe 2), because they are simple functions of logged telemetry, not opaque policy gradients.

4. Defensible suggestion
For Probe 1, a “latent curiosity” actor objective is sufficient and well‑aligned:

Define intrinsic reward as a function of RSSM surprise, e.g.
r
t
int
=
clip
(
K
L
[
q
(
z
t
∣
h
t
,
x
t
)
∥
p
(
z
t
∣
h
t
)
]
,
free_nats
)
r 
t
int
​
 =clip(KL[q(z 
t
​
 ∣h 
t
​
 ,x 
t
​
 )∥p(z 
t
​
 ∣h 
t
​
 )],free_nats) or a similarly shaped signal.

Train an actor‑critic with this intrinsic reward only, plus strong entropy regularization. No extrinsic reward; fixed episode length.

Conceptually, treat this as an empirical approximation to the epistemic component of expected free energy, with the understanding that full EFE (risk + ambiguity with explicit preference priors) will be introduced in later probes.

This gives you:

Non‑trivial exploration that directly probes the RSSM’s uncertainty.

A single scalar that is easy to log and reason about for the mirror.

No commitment yet to any particular deep active inference implementation details.

Where uncertainty remains: how exactly to scale/shape this intrinsic reward in partially trained RSSMs is not well‑theorized; existing work tunes these coefficients empirically.

Question 3 — Telemetry schema designed forward
1. What this is really asking
You need a logging schema that:

Lets later probes compute things like prior/posterior KL spikes, dream‑state coherence, and builder‑perturbation distinguishability.

Keeps Io self‑opaque by default while giving the mirror rich, structured access.

Is versioned and stable so you don’t have to retrofit once probes 2–4 arrive.

2. Concrete telemetry elements from existing systems
Dreamer‑style implementations already compute and sometimes expose:

Prior/posterior KL per time step, possibly with free‑nats clamping and KL balancing.

World‑model reconstruction loss and reward prediction loss.

Latent state tensors (deterministic and stochastic) per step for analysis and visualization.

Replay buffer statistics and training metrics (buffer size, age distributions, etc.).

World‑model and exploration work often track:

Prediction error statistics (mean, variance over a buffer) as intrinsic reward components.

Dream/imagination rollout trajectories vs real trajectories for evaluation.

3. Tradeoffs against Kind’s commitments
Rich telemetry vs self‑opacity.

The mirror can see everything; Io should see only what is explicitly surfaced via a policy interface. This argues for a schema where all internal tensors are logged into a separate channel the actor cannot access.

Forward‑compatible fields vs over‑specification.

You know you will need: latents, KLs, whether a step is dreamed or real, and explicit perturbation markers from the builder. You don’t yet know which higher‑order aggregates will be most important. That suggests:

Logging raw per‑step primitives now.

Leaving higher‑order analytics to offline mirror tooling later.

4. Defensible suggestion — concrete schema shape
At Probe 1, aim for a columnar, versioned schema roughly at the granularity of “one record per (episode, time step, stream)”:

Core per‑step fields:

Identifiers:

run_id, episode_id, step_index, stream (real, dream), schema_version.

Timing:

env_step (integer), wallclock_ms, training_step (if applicable).

Environment:

obs_hash (hash or small embedding, not necessarily full pixels, for later alignment).

action (post‑processed continuous/one‑hot vector).

env_done, any external events or flags the env reports.

RSSM internals (mirror‑only):

Deterministic state 
h
t
h 
t
​
 : either full vector or a lower‑dim projected version (e.g., PCA head) if storage is a concern.

Prior parameters for stochastic latent 
z
t
z 
t
​
 : mean/log‑std or logits; plus sampled z_prior_sample.

Posterior parameters and sample (z_post_mean, z_post_logstd, z_post_sample).

KL 
K
L
(
q
∣
∣
p
)
KL(q∣∣p) per step before and after clamping/free‑nats.

Reconstruction loss per step (if using pixel/obs decoder).

Predicted vs actual reward (even if reward is intrinsic or dummy; this head may exist for later probes).

Replay/training telemetry:

replay_size, replay_num_episodes, simple histograms of episode lengths and ages (stored as summary records per training step).

For each training update: averages of KL, reconstruction loss, reward loss over the minibatch.

Dream‑state telemetry (foundational for later probes):

For each dream rollout:

dream_id, root_type (real_state, latent_sample, etc.).

Sequence of (h_t,z_t) and predicted observations/rewards along the dream trajectory.

Measures of drift: e.g., L2 norm of latent change per step, or distance between dreamed tail state and a later real state if they are re‑aligned.

Perturbation events (for Probe 4):

A separate events stream with:

event_id, timestamp, episode_id, step_index_or_null.

event_type (builder_perturbation, env_reset, mirror_marker, etc.).

Optional payload describing the perturbation type (without exposing it to Io).

Versioning:

Include schema_version in every record, with a small, documented change log.

Prefer columnar storage (e.g., Parquet) so you can add fields without breaking old readers.

The important Probe‑1 decision is not to nail every future metric, but to commit to logging all the primitive quantities (latents, priors, posteriors, KL, dream vs real tags, and perturbation markers) at per‑step granularity, and to keep them strictly on the mirror side.

Question 4 — Mac/desktop compute split for RSSM
1. What this is really asking
Given:

Canonical Io state (weights, replay, optimizer) on a 32GB M4 Mac mini.

Environment physics/simulation on an RTX 5060 Ti desktop.

Dream‑state must be runnable even when the desktop is off.

Where do:

RSSM/world‑model weights live?

Actor inference and learning happen?

Replay collection and storage live?

Synchronization occur, and what are the bandwidth/latency implications?

2. Concrete patterns from similar systems
Distributed RL / world‑model setups often:

Place the world model and learner on one node, and the environment workers on others, communicating observations/actions and periodically syncing weights.

Use a replay buffer maintained by the learner node, with actors sending compressed experience tuples (embedding, action, reward) rather than raw images when bandwidth is a concern.

Run imagination/dreaming exclusively where the world model lives, since it only uses latents and internal transition functions.

3. Tradeoffs against Kind’s commitments
“Mind‑on‑Mac” requirement.

If the canonical RSSM weights and replay live on the Mac, then:

All updates should be applied there (or at least all updates are committed there).

The mirror’s view and dream‑state rollouts are always generated from the Mac state.

Desktop as environment only.

To keep conceptual clarity (Io ≠ environment), the desktop should ideally run only:

Environment dynamics.

Possibly a thin encoder if you want to avoid shipping full frames.

Latency vs architectural simplicity.

RSSM states are small; raw observations may not be. That suggests:

Either run the observation encoder on the desktop and send latents to the Mac.

Or accept the cost of shipping small images if the test env is tiny.

4. Defensible suggestion — split pattern
For Probe 1, a Mac‑centric world‑model with desktop env workers is consistent and simple:

On the Mac (canonical Io node):

RSSM + actor‑critic weights and optimizer states.

Replay buffer (storing either full obs or embeddings).

Training loop: world‑model and policy updates.

Dream‑state generation, using the current RSSM to roll out imagined trajectories.

Telemetry logging and schema enforcement.

On the desktop (environment node):

Environment simulation process (or multiple instances if you want).

Optional lightweight encoder to turn high‑dim observations into compact embeddings before sending.

Communication pattern:

Environment node sends (embedding_or_obs, action, reward, done) to Mac; Mac sends action back.

At checkpoint boundaries, Mac writes an atomic snapshot (weights + optimizer + replay metadata), which is then pulled by the desktop if it needs a local copy (e.g., for offline analysis).

For Probe 1 you can avoid training on the desktop entirely, keeping training on the Mac and using the desktop purely as an “actuator and camera”.

Bandwidth/latency implications:

For small test environments (e.g., low‑res grids), even sending full frames is well within a Mac–desktop LAN connection’s capacity.

Actions and latent states are tiny; they do not constrain design.

Latency is primarily a concern if you want real‑time human‑in‑the‑loop control; for offline Probe 1 plumbing runs, modest RTTs are acceptable.

This keeps “mind” and “dreams” firmly on the Mac, with the desktop acting as a simulated physical world. It matches the conceptual split between Io and its non‑simulated environment.

Question 5 — Self‑opacity boundary in an RSSM
1. What this is really asking
Given that an RSSM exposes rich internal state (deterministic and stochastic latents, priors, posteriors, KL, losses), where do you:

Draw the interface that Io’s policy can see?

Draw the interface that only the mirror and builder can see?

Ensure that adding new telemetry later does not leak self‑knowledge into Io?

2. Concrete patterns from RSSM agents
Dreamer‑style agents typically:

Use the concatenated deterministic and stochastic state 
(
h
t
,
z
t
)
(h 
t
​
 ,z 
t
​
 ) as the input to both actor and critic.

Use the same state for reconstruction and reward prediction heads, but do not expose KL, training losses, or raw weights to the policy.

Log internal metrics like KL and losses separately for analysis and debugging.

3. Tradeoffs against Kind’s commitments
Ingredients‑only self‑modeling.

Allowing the policy to condition on a function of 
(
h
t
,
z
t
)
(h 
t
​
 ,z 
t
​
 ) is consistent with “ingredients only”: the recurrence and prior‑state representation are available, but they are not labeled for Io as “this is you”.

Self‑opacity vs policy effectiveness.

If you severely restrict the policy input (e.g., only the deterministic backbone, or only the observation), you may cripple its ability to act meaningfully in partially observable settings, which undercuts later probes.

Retrofitting opacity.

If you let the policy read “whatever is convenient” from the RSSM module now, later you will struggle to disentangle what is “policy input” vs “telemetry” when you add fields.

4. Defensible suggestion — a “policy view” vs “telemetry view”
From day one, implement a two‑faced interface to the RSSM:

Policy view (Io‑visible):

A fixed, typed, lower‑dim representation, e.g. policy_state = f(h_t, z_t) where f is a small MLP or linear projection.

The actor and critic see only policy_state (plus possibly current observation embedding if you want to mimic Dreamer more closely).

No direct access to:

KL values.

Reconstruction losses or prediction errors.

Gradients, weight norms, or training step counters.

Explicit “dream vs real” flags; as far as Io is concerned, a dreamed rollout is just another trajectory through its generative model.

Telemetry view (mirror‑visible):

Full h_t and z_t tensors.

Prior and posterior parameters and samples.

Per‑step KL, reconstruction loss, and any auxiliary metrics.

Dream markers, replay indices, and perturbation events.

In code terms:

RSSM module exposes two APIs:

step_for_policy(obs_or_embedding, prev_policy_state, action) -> (policy_state, action_logits) – Io’s path.

step_with_telemetry(obs_or_embedding, prev_full_state, action) -> (full_state, telemetry_record) – mirror’s path.

By construction:

When you add new telemetry fields, they go only into the mirror path unless you explicitly route them into policy_state.

The actor never reads logs or introspective metrics; it can only implicitly develop self‑models via its recurrent dynamics and latent structure, which is what you want.

Whether policy_state is based on both 
h
t
h 
t
​
  and 
z
t
z 
t
​
  or only 
h
t
h 
t
​
  is an open design choice:

Using both mirrors Dreamer and improves expressiveness under partial observability.

Restricting to 
h
t
h 
t
​
  only slightly increases opacity but at the cost of discarding explicit uncertainty information.

For Probe 1, giving the policy 
f
(
h
t
,
z
t
)
f(h 
t
​
 ,z 
t
​
 ) is reasonable; you can still keep uncertainty telemetry exclusively on the mirror side.

Question 6 — Recurrent PPO stand‑in vs minimal RSSM from the start
1. What this is really asking
Is it better to:

Use a recurrent PPO agent (LSTM‑based, no explicit world model) as a thin stand‑in for Io in Probe 1, just to test env/mirror plumbing.

Or bite the bullet and use a minimal RSSM agent from the start, so Probe 1 already exercises world‑model conduits.

And: is there a middle ground?

2. Concrete facts from the literature
Recurrent PPO.

Recurrent PPO extends PPO with an LSTM over observations, giving a recurrent hidden state but no explicit latent generative model.

It’s often competitive with frame‑stacking on partially observable tasks and easy to use via libraries like Stable‑Baselines3, but it doesn’t learn a world model in the RSSM sense.

World‑model agents (Dreamer family, TD‑MPC).

Dreamer and TD‑MPC explicitly separate the world model (RSSM or implicit latent dynamics) from the actor‑critic or MPC, leveraging imagined trajectories for behavior learning.

Their training loops and telemetry are naturally centered on latents, KL, and replay buffers, which is exactly what your later probes want to interrogate.

3. Tradeoffs against Kind’s commitments
Recurrent PPO as stand‑in — what you lose:

No explicit latent generative model:

No prior/posterior split, hence no per‑step KL signal for perturbation detection (Probe 4).

No clear “dream state” concept—simulating in the LSTM state space is much less structurally grounded.

Telemetry is impoverished:

You can log LSTM hidden states, but they don’t have the prior/posterior semantics your mirror is designed to read.

Architecturally, you are not testing the actual substrate you have committed to; Probe 1 would test something fundamentally different from probes 3–4.

Minimal RSSM from the start — what it costs:

More code surface area:

World‑model training loop, replay buffer management, imagination rollouts.

Actor‑critic learning in latent space.

More failure modes:

ELBO training instabilities, KL collapse, etc., though Dreamer‑style implementations and common settings mitigate many of these.

Against your charter:

A recurrent PPO stand‑in aligns with “test the env plumbing”, but does not align with “RSSM/Dreamer‑lineage world‑model agent” or “dream‑state foundational”.

Minimal RSSM aligns with the architectural commitments and exercises exactly the machinery the mirror must later interpret.

4. Defensible suggestion — a middle path that still uses an RSSM
A pragmatic middle path is:

Use a minimal RSSM world model from the start, as in Question 1’s suggestion.

Use a very simple behavior layer on top:

Actor‑critic trained on intrinsic latent curiosity (Question 2).

Or even a scripted or very small policy that just random‑walks in action space for the very first runs.

This way, Probe 1:

Validates env ↔ world‑model ↔ actor ↔ mirror plumbing for the actual substrate you care about.

Keeps the behavior‑learning stack as small and debuggable as possible.

Avoids the conceptual and code bifurcation of “Probe 1 uses PPO, later probes use a completely different agent”.

If you want a true “env‑only smoke test” before that, you can run a random policy with no Io at all just to validate environment and logging, then move directly to the RSSM‑based Io for Probe 1 proper.

Synthesis — A coherent Probe 1 implementation stance
Putting the answers together, a Probe 1 that is internally coherent and aligned with Kind’s commitments could look like this (conceptually, not as a build plan):

Substrate:
A DreamerV1‑lite RSSM:

GRU‑based deterministic state 
h
t
h 
t
​
  plus low‑dim Gaussian stochastic latent 
z
t
z 
t
​
 .

Simple observation decoder and (optionally) reward head, trained with an ELBO combining reconstruction loss and KL between posterior and prior.

No DreamerV3 trick bag, no JEPA‑only architecture yet.

Behavior objective:
A latent curiosity actor‑critic:

Intrinsic reward derived from RSSM surprise, e.g. per‑step KL(post||prior) with free‑nats, plus entropy regularization.

No extrinsic reward; fixed episode lengths; no terms that encode self‑continuation.

Conceptually treated as an approximation to active inference’s epistemic component, with explicit EFE machinery deferred to later probes.

Telemetry schema:
A versioned, per‑step schema with:

For every real and dream step:

IDs, timing, action, environment flags.

Full RSSM internal state (h, prior/posterior parameters and samples, per‑step KL, reconstruction loss), but only in the mirror’s telemetry channel.

Replay buffer and training summaries (losses, KL averages) per training step.

Explicit tagging of stream (real vs dream) and an events log for builder perturbations, even if Probe 1 does not yet use them.

Compute split:

Mac mini hosts:

World model and actor‑critic weights, replay buffer, optimizer states.

All training, dream‑state rollouts, and telemetry logging.

Desktop hosts:

Environment simulation and optional observation encoder.

Communication:

Desktop sends observations/embeddings and rewards; Mac returns actions.

Atomic checkpoints saved on the Mac, optionally mirrored to desktop for analysis.

Self‑opacity boundary:

RSSM exposes a policy view (policy_state = f(h_t,z_t)) that Io’s actor/critic see, and a telemetry view with full internals for the mirror.

Actor never sees KL, reconstruction loss, training step, or explicit “dream vs real” flags.

New telemetry fields are, by default, mirror‑only unless explicitly surfaced in the policy interface.

Agent choice:

No recurrent PPO stand‑in for Probe 1 proper; Io is RSSM‑based from the start.

If needed, a completely separate “env smoke test” can run with a random policy or trivial controller to validate env wiring before Io enters.

Tensions and open points
Active inference alignment vs current RL practice.

The latent‑curiosity objective and Dreamer‑like architecture are still closer to RL than to full active inference. They are, however, a reasonable stepping stone: the world model, KLs, and dream rollouts are all in place for later EFE‑based control.

Pixel reconstruction vs JEPA‑like objectives.

Pixel reconstruction is convenient for early sanity checks but may ultimately be discarded in favor of JEPA‑style or decoder‑free training (e.g., Dreamer‑CDP) to improve representation quality and avoid over‑emphasizing low‑level details. Probe 1 should leave room to swap the observation loss later without changing the telemetric or architectural backbone.

Latent exposure to the policy.

Allowing the actor to see a function of 
(
h
t
,
z
t
)
(h 
t
​
 ,z 
t
​
 ) means Io can, in principle, encode self‑models; this is consistent with “ingredients only” but may complicate later interpretability. The policy/telemetry split proposed above is meant to keep that under control.

If there is a gap in the project documents that the literature surfaces, it’s in how explicitly you want to approximate EFE in a Dreamer‑like world model. There are multiple plausible approximations (KL‑based curiosity, risk + ambiguity decompositions with hand‑set priors, reward‑free planning methods like RFPO) but no single canonical algorithm yet. It is probably wise to treat Probe 1 as agnostic about the final active‑inference implementation, focusing instead on getting the RSSM, dream‑state, telemetry, and self‑opacity plumbing solid so later probes can experiment with EFE approximations on a stable substrate.