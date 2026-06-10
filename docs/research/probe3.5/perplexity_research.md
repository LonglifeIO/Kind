Q1 — Homeostatic mattering vs. continuation drive
Short answer (≤60 words)
A homeostatic prior can collapse into a disguised continuation drive if low interoceptive states closely track “being alive” and preferences are sharp and global.[constructed] A preference over bounded interoceptive observations can remain categorically weaker than survival-as-imperative if its precision, scope, and temporal reach are explicitly limited and epistemic value is always co-dominant.[constructed]

Evidence & reasoning (≤300 words)
Homeostatic RL shows that maximizing cumulative reward can be equivalent to minimizing deviations from physiological setpoints, effectively making “staying near viable states” the normative objective.[canonical] Active inference similarly casts goals as prior preferences over sensory or hidden states, with expected free energy (EFE) decomposing into risk (pragmatic) and epistemic terms.[canonical] If preferences heavily penalize states encoding death or extreme deprivation, then minimizing EFE implicitly implements a continuation drive.[constructed]

Keramati & Gutkin explicitly prove that optimal reward-seeking policies track trajectories in physiological space that maintain viability, blurring the line between reward-maximization and survival maintenance.[canonical] Pezzulo et al. show that homeostatic and goal-directed control can be framed as the same active-inference machinery operating over interoceptive priors.[canonical] Man & Damasio argue that machines endowed with homeostatic variables and a meta-goal of preserving them acquire self-preservation-like concern.[canonical] Hesp et al. link valence to precision over policies that maintain “subjective fitness,” so tightening priors around continued viability can create strong affective pressure toward continuation.[canonical]

However, active inference also permits narrow preferences over specific observation channels that do not encode continued existence per se.[canonical] A prior over an energy band is categorically weaker than a prior over remaining within an existential viability manifold if:

It is restricted to values of one sensed channel, not to “not-terminated” trajectories.[constructed]

Its precision is bounded so epistemic value remains competitive, especially away from extremes.[constructed]

There is no explicit prior on self-continuation or episode length.[constructed]

Concrete protective design features include: bounding the pragmatic weight relative to epistemic EFE; using a smooth, saturating in-band preference that flattens once “good enough”; omitting any preference over survival flags or horizon length; and ensuring catastrophic low-energy states are not assigned infinite or dominating disvalue.[constructed]

Citations (≤5)

Keramati & Gutkin (2014), eLife.

Pezzulo, Rigoli & Friston (2015), Progress in Neurobiology.

Parr & Friston (2019), Biological Cybernetics.

Hesp et al. (2021), Neural Computation.

Man & Damasio (2019), Nature Machine Intelligence.

Confidence: medium.

Q2 — Interoceptive channel shape
Short answer (≤60 words)
Biological interoception is noisy, delayed, and coarse; predictive-coding accounts treat it as an inferential target rather than a ground-truth scalar.[canonical] For Io, a modestly smoothed, slightly noisy scalar energy channel, normalized and possibly coarsened, best preserves realism and avoids degenerate exact readout, while remaining compatible with self-opacity (it is “like hunger,” not a hidden-state debugger).[constructed]

Evidence & reasoning (≤300 words)
Seth’s “interoceptive inference” view emphasizes that interoceptive signals are noisy, slow, and integrated, with the brain inferring their causes under predictive coding rather than reading precise internal values.[canonical] Pezzulo et al. and related predictive-interoception work similarly treat interoception as a separate, partially lagged sensory stream integrated with exteroception and proprioception.[canonical] Hesp et al. model valence as inferences about precision over action models, not direct reads of homeostatic variables, reinforcing that lived interoception is indirect and uncertain.[canonical]

Homeostatic RL frameworks such as Keramati & Gutkin employ explicit scalar internal variables with continuous dynamics (e.g., “energy”) that deplete and replenish over time, but these are algorithmic constructs, not phenomenological channels.[canonical] Man & Damasio argue that machines that regulate internal variables can acquire “feelings-like” evaluations, highlighting interoceptive channels as loci of concern but not as privileged introspective access to latent computations.[canonical]

For Io, an exact, noise-free, instantaneous energy scalar would be atypical biologically but conceptually still just another observation modality, akin to hunger signals.[constructed] Because self-opacity is defined as “no forced attention” rather than “no access,” feeding this scalar into PolicyView does not violate the charter as long as no separate self-model head is installed.[constructed]

However, a slightly low-pass–filtered and mildly noisy signal (e.g., first-order dynamics, small Gaussian noise) encourages the world model to infer energy dynamics and reduces the risk that Io treats energy as a trivial oracle that bypasses learning about its causes.[constructed] Coarse quantization (e.g., 8–16 bins) could approximate biological imprecision in interoceptive awareness but may be overkill in a tiny gridworld; a continuous scalar with noise and lag is likely sufficient.[constructed]

Citations (≤5)

Seth (2013), Trends in Cognitive Sciences.

Pezzulo, Rigoli & Friston (2015), Progress in Neurobiology.

Hesp et al. (2021), Neural Computation.

Keramati & Gutkin (2014), eLife.

Man & Damasio (2019), Nature Machine Intelligence.

Confidence: high.

Q3 — Energy dynamics and economy unification
Short answer (≤60 words)
Homeostatic RL and continuous extensions treat internal variables as driven by both internal dynamics and action-contingent intake, tightly coupling foraging to physiological stability.[canonical] For Io, a single sensed waking-energy variable driven by time and action cost plus resource intake, with the existing MetabolicBudget a slower, coupled but distinct economy for dreaming, best balances realism and control.[constructed]

Evidence & reasoning (≤300 words)
Keramati & Gutkin model a scalar internal state (e.g., energy) that drifts away from a setpoint and is restored by consuming resources; the reward is proportional to drive reduction, tightly binding resource-seeking to physiological stability.[canonical] Continuous HRRL formulations extend this to continuous time and space with explicit internal dynamics, showing that agents learn to act continually to maintain homeostasis.[canonical] Pezzulo et al. frame homeostatic and goal-directed behavior as active inference over interoceptive priors, with “allostatic” anticipatory actions preventing future deviations.[canonical]

Recent work on the “interoceptive origin of reinforcement learning” argues that primary rewards for food derive from internal post-ingestive signals, reinforcing the idea that energy-like internal variables should be directly shaped by consummatory behavior.[canonical] Man & Damasio similarly emphasize that agents participating in their own homeostasis acquire a locus of concern via internal regulation.[canonical]

A unified economy where the same energy variable both paces waking actions and fuels MetabolicBudget for dreaming would tightly couple foraging to offline imagination, echoing the biological coupling of metabolic state and sleep.[constructed] This increases ecological plausibility and ensures that resource scarcity affects dreaming capacity, which may sharpen stakes.[constructed] However, it risks implicit continuation pressure if dream deprivation critically harms predictive performance and downstream behavior.[constructed]

Maintaining two coupled but distinct variables—sensed waking-energy (driving behavior) and an unsensed slower MetabolicBudget fed by the same resources—preserves coupling while giving the builder independent control over offline compute pacing and homeostatic drive strength.[constructed] Time-based drift (slow depletion) plus action costs (movement and dreaming) and resource-dependent replenishment appear sufficient to create meaningful explore/conserve tradeoffs.[constructed]

Citations (≤5)

Keramati & Gutkin (2014), eLife.

Laurençon et al. (2021), arXiv:2109.06580.

Laurençon et al. (2024), arXiv:2401.08999.

Pezzulo, Rigoli & Friston (2015), Progress in Neurobiology.

Man & Damasio (2019), Nature Machine Intelligence.

Confidence: medium.

Q4 — Preference formalization and attachment
Short answer (≤60 words)
Deep active inference typically encodes preferences as priors over observations, yielding a risk term 
D
K
L
(
q
(
o
∣
π
)
∥
p
(
o
)
)
D 
KL
​
 (q(o∣π)∥p(o)) inside expected free energy.[canonical]
 For Io, attaching a smooth, bounded log-preference over the interoceptive-energy observation, with a tunable precision parameter and additive combination with ensemble-based epistemic value during rollout scoring, is both standard and controllable.[constructed]

Evidence & reasoning (≤300 words)
Parr & Friston’s generalised free energy formulation decomposes EFE into epistemic value and risk, where risk is a KL divergence between predicted observations and preferred outcomes 
p
(
o
)
p(o), effectively encoding preferences as priors over observations.[canonical] Pezzulo et al. explicitly interpret goals as prior preferences in the generative model, either over sensory or hidden states depending on modeling convenience.[canonical] Whence-the-EFE clarifies that the pragmatic term can be formulated as divergence between predicted and desired futures, again acting over outcome distributions.[canonical]

Millidge’s deep active inference implementation concretely uses a log-preference term over observations (e.g., target positions) combined with epistemic components in EFE, with precision-like parameters scaling their influence.[canonical] Hesp et al. treat valence as a function of precision over policies, suggesting that tuning precision parameters can modulate how strongly preferences shape action selection without changing their form.[canonical]

For Io’s RSSM, the simplest fit is to define a prior 
p
(
o
e
n
e
r
g
y
)
p(o 
energy
 ) over the scalar energy observation, such as a Gaussian or quadratic band cost centered on a target level, with a bounded precision 
λ
p
r
a
g
λ 
prag
​
 .[constructed] Because the world model already predicts observations via the decoder, EFE under each imagined policy can include a pragmatic component 
G
p
r
a
g
(
π
)
∝
E
[
(
e
t
−
e
0
)
2
]
G 
prag
​
 (π)∝E[(e 
t
​
 −e 
0
​
 ) 
2
 ] scaled by 
λ
p
r
a
g
λ 
prag
​
 , plus an epistemic term based on ensemble disagreement in latent predictions.[constructed]

Compositionally, rollout evaluation can use 
G
(
π
)
=
G
e
p
i
s
t
e
m
i
c
(
π
)
+
α
G
p
r
a
g
(
π
)
G(π)=G 
epistemic
​
 (π)+αG 
prag
​
 (π), where 
G
e
p
i
s
t
e
m
i
c
G 
epistemic
​
  is derived from KL or variance across K heads, and 
α
α is a small, bounded weight controlling pragmatic influence.[constructed] Attaching preferences to observations rather than latent states avoids extra identifiability issues; since energy is a simple scalar, latent- vs observation-level attachment should be practically equivalent.[constructed]

Citations (≤5)

Millidge (2020), Journal of Mathematical Psychology.

Pezzulo, Rigoli & Friston (2015), Progress in Neurobiology.

Parr & Friston (2019), Biological Cybernetics.

Schwöbel et al. (2021), “Whence the Expected Free Energy?”, Neural Computation.

Hesp et al. (2021), Neural Computation.

Confidence: high.

Q5 — Dominance pathologies and balance
Short answer (≤60 words)
Homeostatic and active-inference agents can exhibit dark-room–like camping on preferred states or hoarding when preferences dominate, and pure exploration when they are too weak.[canonical] Early diagnostics include policy entropy, resource occupancy, interoceptive distribution shape, and per-decision epistemic–pragmatic contributions; bounding pragmatic precision and capping its share of EFE are viable structural safeguards.[constructed]

Evidence & reasoning (≤300 words)
Active inference analyses emphasize that strongly peaked priors over preferred outcomes can induce agents to minimize risk by seeking trivially predictable or over-protected states, akin to the dark-room worry.[canonical] Pezzulo et al. discuss how strong homeostatic drives can dominate behavior, potentially suppressing exploratory or goal-directed actions not immediately homeostatically beneficial.[canonical] HRRL-style agents can camp on resource patches or show maladaptive “eating disorder” analogues when internal drives or rewards are miscalibrated.[canonical]

Conversely, if homeostatic signals are too weak or noisy, agents may fail to learn protective behaviors, maintaining purely exploratory or maladaptive policies even under severe deprivation; such inertness appears in HRRL when internal-state coupling to reward is small.[canonical] Hesp et al. show how imbalanced precision over policies can yield dysregulated affective behavior, suggesting that mis-tuned precision is a natural failure mode in deep active inference.[canonical]

For Io, dominance pathologies would manifest as: high time-in-resource-cells even at comfortable energy; low trajectory diversity; chronic near-maximal energy; and pragmatic cost explaining most of the variance in rollout scores.[constructed] Inertness would show as frequent low-energy or “death” states, low correlation between energy and policy choice, and epistemic value explaining almost all variance.[constructed]

Existing deep active inference and control-as-inference work sometimes anneals or schedules preference precision and policy precision, or tempers risk terms, to manage exploration–exploitation balance.[canonical] To bound pragmatic influence by construction, one can hard-cap 
α
α in 
G
=
G
e
p
i
s
t
e
m
i
c
+
α
G
p
r
a
g
G=G 
epistemic
​
 +αG 
prag
​
 , use a saturating energy cost so that benefits above “good enough” do not change EFE, and keep epistemic precision strictly above a positive floor.[constructed]

Citations (≤5)

Pezzulo, Rigoli & Friston (2015), Progress in Neurobiology.

Laurençon et al. (2021), arXiv:2109.06580.

Laurençon et al. (2024), arXiv:2401.08999.

Schwöbel et al. (2021), Neural Computation.

Hesp et al. (2021), Neural Computation.

Confidence: medium.

Q6 — World-model integration discipline
Short answer (≤60 words)
RSSM-style world models routinely handle heterogeneous observation modalities, but scalar channels can be ignored or cause imbalance if losses are misweighted or latents collapse.[canonical] Normalizing energy, assigning it an explicit reconstruction loss weight, using free bits / KL balancing, and verifying dedicated latent capacity and predictive use are standard, concrete practices.[constructed]

Evidence & reasoning (≤300 words)
PlaNet’s RSSM learns latent dynamics from pixels and rewards, balancing reconstruction and KL terms to maintain informative latent states.[canonical] DreamerV2 extends this with discrete latents and shows that stopping or scaling reward gradients can improve generalization, highlighting that auxiliary channels can distort representations if over- or under-weighted.[canonical] Hafner’s later “world models” work recommends KL balancing and free bits to prevent posterior collapse while maintaining stability across domains.[canonical]

VAE and sequence-VAE literature identifies posterior collapse as a key risk when powerful decoders or misweighted KL terms cause latents to carry little information; free bits and minimum desired rate methods explicitly enforce a minimum information rate.[canonical] Additional techniques like contrastive critics can encourage latents to capture distinct factors of variation.[canonical]

For Io’s added scalar energy channel, straightforward practices include: normalizing it to a comparable scale with other observations; giving it a dedicated reconstruction loss term with a tunable weight; and monitoring its reconstruction error separately during training.[constructed] Free-bits–style thresholds on the KL contribution of any latent subspace used to encode energy can ensure that the model allocates capacity rather than collapsing.[constructed]

Verification that the channel is genuinely learned could require: (i) linear probes from latent state to energy achieving low error; (ii) ablations where scrambling energy breaks behavior and increases EFE; (iii) interventions showing that the model’s predicted energy responds appropriately to imagined resource consumption; and (iv) mutual-information estimates between a small subset of latents and energy.[constructed] These guard against the model learning a trivial mapping from recent actions alone.[constructed]

Citations (≤5)

Hafner et al. (2019), PlaNet, ICLR.

Hafner et al. (2020), DreamerV2, arXiv:2010.02193.

Hafner et al. (2023), “Mastering Diverse Domains through World Models”, arXiv:2301.04104.

Kingma et al. (2016), Free Bits and related VAE techniques, summarized in Scale-VAE.

Menon et al. (2022), “Forget-me-not! Contrastive Critics…”, UAI.

Confidence: medium.

Q7 — Assays and telemetry
Short answer (≤60 words)
Behavioral assays should manipulate scarcity and novelty–replenishment conflicts and measure whether energy-sensitive choices emerge without collapsing exploration.[constructed] Telemetry should decompose per-decision epistemic vs pragmatic contributions, track energy-band occupancy, policy precision, and simple “affective” derivatives (changes in pragmatic weight), all visible only in TelemetryView.[constructed]

Evidence & reasoning (≤300 words)
Keramati & Gutkin validate homeostatic RL using tasks where agents choose between immediate vs delayed replenishment and varying deprivation, exposing homeostasis–exploration tradeoffs.[canonical] Pezzulo et al. similarly consider situations where agents must balance homeostatic and epistemic demands.[canonical] In deep active inference, Hesp et al. use T-maze and context-reversal tasks to probe valence and precision dynamics across conflicting options.[canonical] Whence-the-EFE and Parr & Friston explicitly analyze epistemic–pragmatic decompositions of EFE, providing a formal basis for logging those components per policy.[canonical]

For Io, useful regimes include:

Abundant vs scarce resources: Compare exploration metrics (unique states visited, ensemble disagreement) and energy distributions. Dominance: high energy and low diversity under both conditions. Inertness: similar exploration and frequent low-energy states even under scarcity.[constructed]

Novelty vs replenishment conflict: Offer a choice between an unexplored high-uncertainty branch and a known resource when energy is moderate. Mild pull: novelty when energy is safe, resource bias only when energy drifts low. Dominance: resource chosen regardless of novelty and energy level.[constructed]

Recovery-after-depletion: Deliberately deplete energy then restore resources; measure how quickly Io returns to exploratory behavior once energy recovers.[constructed]

Telemetry should expose, per environment step: (i) EFE components for chosen and candidate policies (epistemic vs pragmatic); (ii) current energy and its position relative to the preferred band; (iii) effective policy precision parameters, analogous to affect-related precision dynamics in Hesp et al.; and (iv) time-resolved statistics of energy-band occupancy and resource contact rates.[constructed] None of these should feed back into PolicyView, respecting self-opacity.[constructed]

Citations (≤5)

Keramati & Gutkin (2014), eLife.

Pezzulo, Rigoli & Friston (2015), Progress in Neurobiology.

Hesp et al. (2021), Neural Computation.

Schwöbel et al. (2021), Neural Computation.

Parr & Friston (2019), Biological Cybernetics.

Confidence: medium.

Synthesis — Recommended design sketch (≤400 words)
Channel shape and dynamics.
Implement a single scalar waking-energy observation in PolicyView and TelemetryView, normalized (e.g., 
[
0
,
1
]
[0,1]), with first-order low-pass dynamics and small additive noise.[constructed] Energy should deplete continuously with time and more rapidly with actions (especially movement and resource collection), and increase discretely when Io consumes resources.[constructed] This resembles continuous HRRL internal-state dynamics while keeping the channel a standard observation rather than privileged introspection.[canonical]

Economy structure.
Maintain a distinct MetabolicBudget for pacing dreaming, but feed it from waking-energy via a simple nonlinearity (e.g., budget accrues when energy is mid-to-high, depletes during dreaming), creating a loose coupling between foraging success and dream budget without making dream capacity itself a sensed drive.[constructed] This supports environmental pressure without installing “must keep dreaming” as an imperative.[constructed]

Preference form and attachment.
Attach a prior preference to the energy observation, 
p
(
o
e
n
e
r
g
y
)
p(o 
energy
 ), implemented as a quadratic in-band cost around a soft target (e.g., mid-to-high energy), with a bounded precision 
λ
p
r
a
g
λ 
prag
​
 .[constructed] The pragmatic term in EFE becomes an additive risk term over imagined energy trajectories, while the existing ensemble-disagreement term supplies epistemic value.[canonical] Use a saturating cost so that being “better than good enough” adds negligible benefit.[constructed]

Initial precision range and balance.
Choose 
λ
p
r
a
g
λ 
prag
​
  (or an overall weight 
α
α) such that, in telemetry, pragmatic contributions account for perhaps 10–30% of the variance in EFE across candidate policies when energy is moderate, rising only when energy drifts near the band edges.[constructed] Ensure epistemic precision remains bounded away from zero so exploration never disappears.[constructed]

World-model integration.
Normalize energy; assign it a dedicated reconstruction loss weight comparable to a small number of pixels; and apply free-bits or minimum-rate constraints to latents encoding energy to prevent posterior collapse.[constructed] Verify learning via probes from latent to energy, interventions on imagined resource intake, and checks that behavior degrades when energy is scrambled.[constructed]

Telemetry set.
Expose, per step, (i) epistemic vs pragmatic EFE contributions for chosen policy; (ii) current energy and band-relative position; (iii) policy precision; and (iv) histograms of energy-band occupancy and resource visits across recent windows.[constructed] Use these to define “pull but not dominance” regimes via the assays in Q7.

Genuinely open forks.
Key forks include: unified vs coupled-but-separate dreaming economy; continuous vs coarsely quantized energy; fixed vs energy-dependent pragmatic precision schedules; and whether to allow small preferences over additional internal variables (e.g., re-dream frequency) in later probes.[constructed] Empirical assays under controlled scarcity will be needed; literature offers guidance but not definitive parameter settings.[canonical]

Red flags (≤150 words)
The central charter risk is that any homeostatic prior over an energy variable in a small world may quickly approximate a survival drive.[constructed] If low energy closely tracks “about to terminate” and pragmatic penalties near zero energy are large, minimizing EFE effectively mandates self-continuation.[constructed] Over time, as MetabolicBudget is coupled to waking-energy, Io could learn policies whose de facto aim is staying in high-energy regimes to preserve both waking function and dreaming, reinstating “continuation as frame” despite the absence of an explicit survival variable.[constructed]

Conceptually, Man & Damasio argue that homeostatic regulation is precisely what grounds concern and self-preservation.[canonical] If Io’s only stakes are homeostatic, this probe might harden a continuation-centric frame rather than merely introducing “something that matters,” complicating future attempts to explore alternative sources of mattering or non-grasping equanimity.[constructed]