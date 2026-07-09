# World v2 research — Claude (deep-research workflow, 2026-07-09)

*(Session-run web-grounded pass against `docs/prompts/worldv2_research.md`:
5 search angles → source fetch → per-claim 3-vote adversarial verification →
synthesis. One of the parallel voices for the world-v2 synthesis. Findings
carry verification votes; the *refuted* section is load-bearing — claims
other voices may assert confidently that did not survive adversarial
checking.)*

## Summary

The literature supports enriching Io's gridworld with dynamics that are learnable-but-not-yet-learned rather than merely stochastic: ensemble disagreement (Io's only drive) provably self-extinguishes both on mastered structure and on incompressible noise, so neither saturated abundance nor raw randomness can sustain engagement — only structure that keeps unfolding can (multi-step causal chains, hidden-state processes, autonomously moving objects). Disagreement-based curiosity is comparatively robust to the noisy-TV trap that captures prediction-error methods, so stochastic enrichments are safe-ish but epistemically inert; the design target is intermediate-difficulty dynamics in the learning-progress sense, which also predicts a natural self-organized ordering when dynamics are introduced one at a time. Observer-side, learning-progress (model-improvement) and aleatoric/epistemic decomposition give concrete reward-free diagnostics separating 'engaging new dynamic' from 'stochastic capture' from 'ignoring'. Capacity cautions for the small RSSM are real: Gaussian z=16 is a documented bottleneck class relative to categorical latents, reconstruction loss can fail to encode dynamics carried by tiny objects, KL balancing is the standard defense against posterior collapse, and standard recurrent latents track out-of-view moving objects poorly — favoring one-dynamic-at-a-time introduction, dynamics that are spatially extended or frequently observed, and KL-balancing-style training hygiene. Reward-free epistemic drives have empirically latched onto interaction-affording objects and partial causal chains (Crafter), supporting contact-responsive, chain-structured enrichments — though the evidence that a small RSSM can master contact physics specifically did not survive verification.

## Verified findings

### F1 (high confidence; votes 3-0, 3-0)

Io's drive is well-characterized in the literature: Plan2Explore/Pathak-style intrinsic reward is defined purely as the variance across K one-step ensemble predictors, computable without observing the next state and without any external reward, and suffices for fully self-supervised exploration and skill learning. (Merges claims 0 and 4.)

**Evidence:** Sekar et al. (ICML 2020): 'the variance of the ensemble serves as an estimate of uncertainty'; Pathak et al. (ICML 2019): reward 'does not depend on the next state... purely a mental simulation of the ensemble.' Verifier confirmed Io's implementation (kind/agents/ensemble.py) is a faithful variant: K=5 heads predicting next z (not image embedding), biased variance summed over z dims — mechanism-level match, not exact replication.

- https://arxiv.org/abs/2005.05960
- https://arxiv.org/abs/1906.04161

### F2 (high confidence; votes 3-0, 3-0, 3-0, 2-1)

Ensemble disagreement self-extinguishes on incompressible stochasticity: given enough samples all members converge to predicting the mean of the stochastic transition, so disagreement decays even in stochastic environments; empirically (Noisy MNIST) disagreement converges to valuing deterministic and stochastic states equally while prediction error permanently rewards stochasticity, and disagreement methods explore past stochastic trap states that prediction-error methods revisit. Design implication: purely stochastic enrichments (unpatterned flicker, pure-noise objects) will not durably hold Io's attention — but neither will they permanently trap it. (Merges claims 1, 2, 3, 12.)

**Evidence:** Three peer-reviewed ICML primary sources agree on the mechanism; quotes verified verbatim ('the variance of the outputs in ensemble will drop preventing the agent from getting stuck in stochastic local-minima of exploration'). Key qualifier carried from verifiers: the guarantee is asymptotic ('given enough samples') — finite K=5 ensembles retain elevated disagreement in noisy regions pre-convergence, so escape can be sample-inefficient; breakdown documented only in high-dimensional pixel-noise regimes irrelevant at 8x8 gridworld scale.

- https://arxiv.org/abs/2005.05960
- https://arxiv.org/abs/1906.04161
- https://arxiv.org/abs/1810.12162

### F3 (high confidence; votes 3-0, 3-0)

Prediction-error-based curiosity systematically fails on unlearnable and especially action-dependent noise sources (the noisy-TV trap): naive prediction-error agents get stuck at randomness sources and fail at exploration, and cannot separate aleatoric risk from epistemic uncertainty. This is the canonical trap Io's disagreement drive avoids by construction — but any observer-side metric built on raw prediction error inherits it. (Merges claims 5 and 10; overlaps 12.)

**Evidence:** Mavor-Parker et al. (ICML 2022 spotlight): prediction-error curiosity 'fails when faced with action-dependent noise sources'; action-dependent stochastic traps 'immobilise conventional curiosity driven agents.' Hou et al. 2025 and MAX (ICML 2019) independently corroborate. Correctly scoped to prediction-error methods; disagreement and learning-progress methods are the documented mitigations.

- https://arxiv.org/abs/2102.04399
- https://arxiv.org/abs/2509.25438
- https://arxiv.org/abs/1810.12162

### F4 (medium confidence; votes 2-1)

Ensemble disagreement also collapses once learnable structure is mastered — members agree on training-set transitions and disagree only on unvisited ones — so disagreement vanishes even in stochastic environments once everything learnable is learned. This predicts Io's observed mastery/boredom curve and is the central enrichment constraint: dynamics must keep supplying NEW learnable structure (unfolding hidden state, chained consequences, slow non-stationarity), not just stochasticity, to sustain the drive.

**Evidence:** MAX (ICML 2019) quote verified verbatim; Plan2Explore states the stochastic-environment extension explicitly. Split vote, but verifier rated evidence high and noted Io's record disagreement at the reset cell is consistent: regrowth drift plus board resampling mean the 'everything learnable is learned' limit never holds there. In-the-limit result; continually trained finite ensembles retain residual disagreement.

- https://arxiv.org/abs/1810.12162
- https://arxiv.org/abs/2005.05960

### F5 (high confidence; votes 3-0, 3-0, 3-0)

Learning progress (LP) is the canonical principle for dynamics that stay engaging: rewarding the derivative of prediction error targets activities of intermediate difficulty ('learnable but not yet learned') as an emergent property, loses interest in both too-easy and unpredictable-but-unlearnable activities, and self-organizes a developmental curriculum — the agent focuses on the maximal-LP activity, plateaus, and shifts to the next without external ordering. For Kind this is a design-and-diagnosis lens, not a drive to install: enrichments should sit in the LP sweet spot, and sequential one-at-a-time introduction mimics the curriculum LP would self-select. (Merges claims 7, 8, 9.)

**Evidence:** Oudeyer, Gottlieb & Lopes 2016 (primary, field-defining review), quotes verified from PDF text: the hand-vs-passing-car example; 'an explicit measure of intermediate complexity is not computed... it is an emergent property'; the four-activity curriculum schematic. Corroborated by IMGEP/automatic-curriculum literature and human behavioral evidence (Nat. Comms 2021). Caveat: idealized model — practical LP estimators are noisy and naive per-region LP can be distracted by aleatoric noise (GRIMGEP).

- http://www.pyoudeyer.com/oudeyerGottliebLopesPBR16Preprint.pdf

### F6 (high confidence; votes 3-0, 3-0)

Observer-side diagnostics for 'engages vs ignores vs trapped' exist without touching the agent's drive: (a) tracking model improvement (learning progress) operationally separates learnable transitions from unlearnable ones — LP is near zero on incompressible noise but positive on unfolding structure; (b) separately predicting the mean and variance of future states isolates aleatoric (irreducible) variance from epistemic uncertainty. Combined with disagreement's known signatures (decays on noise, decays on mastery, stays high on unvisited/unfolding structure), these give a three-way decomposition computable from telemetry alone. (Merges claims 6 and 11.)

**Evidence:** AMA (ICML 2022 spotlight): 'generating separate forward predictions for the mean and variance of future states and reducing intrinsic rewards for those transitions with high aleatoric variance'; LPM (2025, resting on the 2007-2012 peer-reviewed LP lineage): 'rewards model improvements... effectively rewards the agent for observing learnable transitions.' Caveats: aleatoric/epistemic decomposition is not surgically clean (early-training conflation, OOD unreliability); LP is a noisy second-order signal harder to estimate than prediction error.

- https://arxiv.org/abs/2102.04399
- https://arxiv.org/abs/2509.25438

### F7 (high confidence; votes 2-1, 3-0, 3-0, 3-0)

The small-RSSM capacity picture: both the deterministic and stochastic paths are load-bearing (agent does not learn without the deterministic path; without stochasticity the model cannot capture multiple futures); Gaussian latents (Io's z=16 style) were outperformed by categorical latents on 42/55 Atari tasks, marking them a plausible bottleneck class for complex/multimodal dynamics; KL balancing (training the prior faster than the posterior, alpha=0.8) is the standard defense against posterior collapse via posterior-entropy inflation; and reconstruction loss can fail to encode dynamics carried by very small objects (DreamerV2's one systematic failure, Video Pinball, attributed to a single-pixel ball). Together: introduce one dynamic at a time, prefer dynamics whose state footprint in the observation is not vanishingly small, and monitor KL/posterior health when adding multimodal dynamics. (Merges claims 14, 15, 16, 17.)

**Evidence:** All quotes verified verbatim against PlaNet (ICML 2019) and DreamerV2 (ICLR 2021). Transfer qualifiers from verifiers: the categorical-vs-Gaussian and KL-balancing ablations are Atari-scale with the mechanism for categorical superiority explicitly unknown to the authors; the single-pixel caution is an author hypothesis and is quantitatively milder at 8x8 (one cell is 1/64 of the observation, not 1/4096); TD-MPC shows deterministic-only latents can work but only in value-grounded settings, not Io's reward-free RSSM class.

- https://planetrl.github.io/
- https://arxiv.org/pdf/2010.02193

### F8 (medium confidence; votes 2-1, 2-1)

Standard recurrent/memory-augmented world-model latents retain almost nothing about autonomously moving objects once out of the observation window: baselines hallucinate new objects and forget old ones (RSSM degrades to a blurry average), and probes on baseline latents decode out-of-view moving content on under 1% of timesteps. Caution for enrichments with autonomous motion under partial observability — though Io's 8x8 grid, if fully observed, largely sidesteps this. (Merges claims 18 and 19.)

**Evidence:** Flow Equivariant World Models (ICML 2026), quotes verified ('the DFoT and SSM models frequently hallucinate new objects and forget old ones'; '<1% of timesteps'). Heavy qualifiers: single method paper by authors of the competing architecture, two toy benchmarks; the <1% probes are nonlinear (2-layer conv + MLP, ~7M params), computed over all timesteps, and were run on diffusion-transformer baselines — the paper's RSSM baseline was not probed, so extension to Io's architecture is inference, not measurement.

- https://arxiv.org/pdf/2601.01075

### F9 (medium confidence; votes 3-0)

Reward-free epistemic drives empirically latch onto interaction-affording objects and multi-step causal chains: in Crafter with no extrinsic reward, Plan2Explore forages and fights creatures, and RND collects stones, occasionally coal, and builds furnaces (a genuine multi-step chain: wood → table → pickaxe → stone → furnace). This is the best available evidence that contact-responsive objects and causal chains can become objects of engagement for Io's drive class — evidence of engagement, not competence.

**Evidence:** Hafner (ICLR 2022), quote verified verbatim in Section 4.2. Single source, and effect sizes are marginal: Plan2Explore 2.1%, RND 2.0% vs random 1.6% (human 50.5%); random already unlocks the six easiest achievements; the paper frames some Plan2Explore behavior instrumentally ('to ensure longer survival'). 'Durable' engagement is a gloss — the paper reports behavior frequencies, not persistence over training.

- https://arxiv.org/abs/2109.06780

## Refuted (did NOT survive verification)

- **Action-dependent stochastic traps immobilize conventional curiosity-driven agents (the agent becomes stationary at the noise source), and aleatoric mapping agents empirically escape such traps — a documented analog of the reset-cell stasis observed in the Kind run, where the agent sat still because resets deliver scheduled novelty.**  
  (vote 1-2; source https://arxiv.org/abs/2102.04399)

- **Uncertainty-estimation-based intrinsic rewards (the family that includes ensemble-disagreement methods like Plan2Explore) do eventually escape noisy-TV traps as training unfolds, but at the cost of poor sample efficiency and high computational cost.**  
  (vote 0-3; source https://arxiv.org/abs/2509.25438)

- **Ensemble-disagreement novelty measures are robust to irreducible environment stochasticity (the noisy-TV/stochastic-trap problem): unlearnable noise makes all ensemble members similarly uncertain rather than making them conflict, so disagreement does not reward incompressible noise. This directly supports using disagreement (as Io does) rather than raw prediction error when adding stochastic dynamics to the gridworld.**  
  (vote 1-2; source https://arxiv.org/abs/1810.12162)

- **A small RSSM trained from 64x64x3 pixels can learn to predict hard physical dynamics, including contact dynamics between an actuated body and an object and hard-to-predict collisions with the ground — evidence that contact/interaction-affording dynamics (e.g., pushable objects, response-to-contact) are within reach of RSSM-class latent world models.**  
  (vote 1-2; source https://planetrl.github.io/)

## Caveats

Vote splits and single sources: claims 13 (mastery-collapse of disagreement), 12 (trap-state revisiting), and both flow-equivariant-world-model claims (18, 19) passed only 2-1; the LP finding rests on one primary review (though by the field's originators, heavily corroborated); the Crafter finding is a single paper with marginal effect sizes. Asymptotic guarantees: every 'disagreement is noise-robust' result is in-the-limit — Io's K=5 ensemble under continual training can retain elevated disagreement at stochastic sites for a long time, which matters for interpreting the reset-cell episodes. Scale transfer: the categorical-latent, KL-balancing, and single-pixel-object results are Atari-scale with pixel reconstruction; transfer to a z=16 Gaussian RSSM on an 8x8 symbolic-ish grid is inference throughout. Four notable refutations shape what is NOT supported: (a) the reset-cell stasis has no verified literature analog — 'reset-camping as documented curiosity reward-hacking' failed verification, so treat it as a novel/undocumented observation; (b) the claim that a small RSSM demonstrably learns contact/pushable-object dynamics failed (1-2), so interaction-affording physics being within z=16 reach is untested, not established; (c) two broad noise-robustness formulations also failed, reinforcing that robustness claims must carry the asymptotic qualifier. Research need 4 (episode structure, long/infinite episodes vs resets) and need 7's concrete ordering received no surviving direct evidence — the ordering recommendation above is constructed from the LP-curriculum and capacity findings, not cited. Time-sensitivity is low (definitional claims about fixed 2019-2021 methods), except the 2025-2026 preprints (LPM, FloWM) which are recent and less settled.

## Open questions

- Is degenerate reset-camping (an agent parking at the episode-reset cell because resets deliver scheduled novelty) documented anywhere as a curiosity analog of reward hacking at episode boundaries? The one candidate claim was refuted; this may be a genuinely novel failure-mode observation worth writing up, and the literature offers no tested remedy (longer/infinite episodes, reset randomization) specific to disagreement-driven agents.
- Can a z=16 continuous-Gaussian RSSM actually learn contact/pushable-object dynamics? The supporting claim failed verification, and the Crafter evidence shows engagement, not mastery — this is an empirical question the probe itself would have to answer, ideally with a pre-registered latent-probe test.
- How should a learning-progress diagnostic be computed observer-side from Io's existing telemetry (per-region model-improvement over the four streams) without installing any new drive or actor input — and how noise-robust is per-cell LP estimation at 8x8 scale given the GRIMGEP-style aleatoric-distraction failure?
- Does the disagreement-collapse-on-mastery result hold quantitatively for continually trained small ensembles under slow non-stationarity (regrowth drift), i.e., what drift rate keeps disagreement in the LP sweet spot rather than either extinguishing or reading as permanent noise?

## Sources consulted

- https://arxiv.org/abs/2005.05960  (quality primary, angle: Intrinsic-motivation theory: learnable-but-never-mastered dynamics, claims 5)
- https://arxiv.org/abs/1906.04161  (quality primary, angle: Intrinsic-motivation theory: learnable-but-never-mastered dynamics, claims 5)
- https://arxiv.org/abs/2102.04399  (quality primary, angle: Intrinsic-motivation theory: learnable-but-never-mastered dynamics, claims 4)
- http://www.pyoudeyer.com/oudeyerGottliebLopesPBR16Preprint.pdf  (quality primary, angle: Intrinsic-motivation theory: learnable-but-never-mastered dynamics, claims 5)
- https://arxiv.org/abs/2509.25438  (quality primary, angle: Intrinsic-motivation theory: learnable-but-never-mastered dynamics, claims 5)
- https://arxiv.org/abs/1810.12162  (quality primary, angle: Intrinsic-motivation theory: learnable-but-never-mastered dynamics, claims 5)
- https://planetrl.github.io/  (quality primary, angle: RSSM capacity and what small latent world models can predict, claims 5)
- https://arxiv.org/pdf/2010.02193  (quality primary, angle: RSSM capacity and what small latent world models can predict, claims 5)
- https://arxiv.org/pdf/2601.01075  (quality primary, angle: RSSM capacity and what small latent world models can predict, claims 5)
- https://arxiv.org/abs/2109.06780  (quality primary, angle: Environment design for reward-free open-ended exploration, claims 5)
- https://arxiv.org/html/2510.19788v1  (quality primary, angle: Environment design for reward-free open-ended exploration, claims 5)
- https://arxiv.org/pdf/2002.12292  (quality primary, angle: Environment design for reward-free open-ended exploration, claims 5)
- https://arxiv.org/html/2408.09807v3  (quality primary, angle: Episode structure failure modes: reset exploitation and reset-free training, claims 5)
- https://arxiv.org/abs/1808.04355  (quality primary, angle: Episode structure failure modes: reset exploitation and reset-free training, claims 5)
- https://arxiv.org/pdf/1901.10995  (quality primary, angle: Episode structure failure modes: reset exploitation and reset-free training, claims 5)
- https://arxiv.org/pdf/2005.05960  (quality primary, angle: Observer-side diagnostics of engagement from latents and behavior, claims 5)
- https://arxiv.org/pdf/2102.04399  (quality primary, angle: Observer-side diagnostics of engagement from latents and behavior, claims 5)
- https://arxiv.org/pdf/2509.25438  (quality primary, angle: Observer-side diagnostics of engagement from latents and behavior, claims 5)
- https://arxiv.org/html/2603.21546v1  (quality primary, angle: Observer-side diagnostics of engagement from latents and behavior, claims 5)
- https://proceedings.mlr.press/v164/seyde22b/seyde22b.pdf  (quality primary, angle: Observer-side diagnostics of engagement from latents and behavior, claims 4)
- https://arxiv.org/pdf/2504.03861  (quality primary, angle: Observer-side diagnostics of engagement from latents and behavior, claims 5)

## Run stats

```json
{
 "angles": 5,
 "sourcesFetched": 21,
 "claimsExtracted": 103,
 "claimsVerified": 25,
 "confirmed": 21,
 "killed": 4,
 "unverified": 0,
 "afterSynthesis": 9,
 "urlDupes": 1,
 "budgetDropped": 8,
 "agentCalls": 103
}
```