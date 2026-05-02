Io’s Probe‑1 environment should be a very small, partially observable 2D world rendered as low‑resolution pixels, with a tiny discrete action set, modest but structured stochasticity, and explicit hooks for rare builder perturbations — designed so the RSSM has just enough spatial, temporal, and causal structure to prevent latent collapse while remaining simple enough that dream rollouts are legible and non‑trivial.

Below, each question is treated in turn ((a)–(d)), followed by two concrete minimal candidates and one “richer but still small” candidate environment.

Observation modality (Q1)
(a) What this is really asking

Against the chosen RSSM (PlaNet‑style deterministic GRU 
h
t
h 
t
​
  plus Gaussian 
z
t
z 
t
​
 ) and Dreamer‑style latent imagination from pixels, the observation question is:

What form of sensory stream gives a small RSSM enough structural richness (spatial layout, object identity, occlusion, local vs global information) to learn non‑trivial latent dynamics, without overwhelming it with irrelevant detail?

How does that choice relate to embodied cognition ideas: perception as field‑of‑view from somewhere, horizons, object permanence, and the conditions for anything like proto‑intentional “aboutness” to emerge?

Io’s actor only ever sees 
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
 ), but the encoder–decoder connects those latents to a chosen observation space. So the modality determines what kinds of regularities the RSSM can discover and what kind of “world” dream rollouts can depict.

(b) Concrete options from literature

Full‑frame low‑res pixels (global view)

PlaNet and Dreamer learn world models directly from small image observations (e.g., 
64
×
64
64×64 pixels) and build compact latents that track object types, positions, and their interactions.

Many visual control benchmarks (DMControl, Atari) use full‑frame observations; Dreamer’s RSSM learns latent “concepts” like positions and velocities implicitly from those pixels.

Partial-perspective pixels (ego‑centric or windowed view)

Recent world‑model work explicitly studies partially observed dynamic worlds where the agent only sees a window of a larger map; the model must represent off‑screen dynamics in its latent memory.

In FloWM‑style setups, a fixed “field of view” patch is read from a larger hidden state; the rest of the world evolves outside the current view, forcing the model to maintain a structured representation of unobserved regions.

Low‑dimensional factored state vectors (symbolic grid position, resource counts)

Classic gridworld and simple control tasks often expose factored, low‑dimensional states (e.g., coordinates, presence of walls, resource counts) instead of pixels.

Early world‑model work and some continual‑RL benchmarks (MiniGrid variants, DoorKey) use symbolic states or very simple encodings; Dreamer‑like agents can solve these tasks from such compact states.

Hybrid: pixels + structured features

Hybrid “mixture of world models” approaches combine pixel‑based world models with latent models, where the latent part captures high‑level motion and the pixel model handles fine‑grained details, improving embodied planning performance.

Object‑centric world models (e.g., FIOC‑WM) factor pixels into object latents, then plan in terms of object–object interactions.

In embodied/developmental robotics, perception is typically local, partial, and action‑dependent (e.g., a camera on a moving robot, tactile arrays), and many groups emphasize that partial, situated sensing is important for “sense‑making” and structural coupling with the environment.

(c) Tradeoffs vs Kind’s commitments

Pixel vs factored state

Pixels align with Kind’s decoder/mirror requirement: decoded dream rollouts are human‑legible; this is exactly the path PlaNet/Dreamer take.

Symbolic/factored state makes RSSM training easier and reduces risk of nuisance‑feature leakage, but weakens the link to embodied perception and deprives Io of phenomena like occlusion and local perspective.

Full‑frame vs partial view

Full‑frame small images simplify modeling but make Io’s “perspective” god‑like rather than situated; the agent doesn’t experience horizons or the epistemic value of moving to reveal occluded regions.

Partial view directly instantiates phenomenological ideas of a “world beyond the current horizon,” and is strongly linked to world‑model work on partially observed maps. This also interacts well with Io’s epistemic actor objective: information‑seeking actions have clear meaning (move to see more).

Resolution and channel count

Higher resolution / texture diversity can cause “distraction” in the latent — encoding background details unrelated to meaningful dynamics.

Low‑resolution, low‑color images (e.g., 
16
×
16
16×16–
32
×
32
32×32 grayscale) reduce that distraction, making it more likely that the RSSM encodes controllable structure rather than visual noise.

(d) Suggestion / uncertainty

Suggestion: Use small pixel observations with an ego‑centric, partial field of view over a slightly larger underlying grid. Concretely: an underlying 
5
×
5
5×5–
7
×
7
7×7 world, with Io observing a 
3
×
3
3×3 or 
5
×
5
5×5 window rendered as a 
16
×
16
16×16 or 
32
×
32
32×32 grayscale image.

This preserves human‑decodable dreams, enforces “perception from somewhere,” and forces the RSSM to carry off‑screen structure in 
h
t
,
z
t
h 
t
​
 ,z 
t
​
 .

Uncertainty: There is no direct empirical literature on “minimum pixel complexity for proto‑intentionality”; most world‑model studies target performance on tasks, not phenomenological richness. So the choice is principled but not precedent‑backed in that specific sense.

Action space (Q2)
(a) What this is really asking

Given a small RSSM and a curiosity‑like actor (latent disagreement), what is the minimal action repertoire that still supports:

Non‑trivial sensorimotor contingencies (actions change what is seen),

Emergence of an internal world model with meaningful structure, and

Something like intentional engagement (Io can do enough to have projects, not just random twitching).

The question is explicitly not “what action set maximizes task success,” but “what lower bound on action richness is needed so the environment is more than a passive movie.”

(b) Concrete options from literature

Tiny discrete movement sets (gridworld style)

Canonical gridworlds use 4 or 5 discrete actions: move up/down/left/right, sometimes with a no‑op.

Dreamer‑like agents have been applied to MiniGrid tasks (DoorKey‑9x9, LavaCrossing) with such tiny discrete action sets and still learned useful world models and policies in latent space.

Richer discrete sets (movement + interaction)

Many developmental‑robotics studies give agents a small set of actions that combine movement and object interaction (e.g., reach, grasp, poke), enabling exploration of tool use or vocalizations; intrinsic motivation modulates which actions are used when.

Social‑robotics systems equip robots with discrete “behavior libraries” for interaction (gestures, utterances), but that’s more about social repertoire than physical world modeling.

Low‑dimensional continuous actions

Dreamer and PlaNet handle continuous control well (e.g., torques for DMControl tasks), using continuous action vectors of modest dimensionality (e.g., 2–8 dims) to steer movement.

In continuous arenas, even a few continuous degrees of freedom (e.g., thrust in x/y) can produce rich trajectories, but tuning dynamics and scales becomes more delicate.

Morphological‑development work emphasizes that action space and body morphology jointly shape what can be learned: poorly chosen morphologies or action repertoires can make control harder and inflate the needed control structure.

(c) Tradeoffs vs Kind’s commitments

Discrete vs continuous

Discrete actions simplify the control head and make it easier to reason about how actions map to environment dynamics (important for later probes analyzing Io’s model).

Continuous actions can encode more nuanced control, but greatly expand the policy search space and complicate interpretation; they’re unnecessary for a Probe‑1 tiny world.

Cardinality and intentionality

With only a single action or two symmetric actions (e.g., left/right), the agent’s action space may be too impoverished to support varied epistemic foraging; disagreement‑driven exploration may collapse to trivial oscillations.

A 4‑direction movement plus optional no‑op (5 actions) matches standard gridworlds and MiniGrid and is sufficient to support navigation, exploration, and interaction (e.g., “step onto” a resource).

Interaction actions

Adding an explicit “interact/use” action may be premature at Probe 1 if the environment has only a single resource type with simple pickup mechanics; stepping onto a cell can be sufficient to trigger change, reducing action complexity.

(d) Suggestion / uncertainty

Suggestion: A 5‑action discrete set: {up, down, left, right, stay}, with interactions encoded as state changes when Io enters or stays on certain cells. This matches literature on minimal yet non‑trivial gridworld control while keeping the policy head small.

Uncertainty: Embodied cognition has no crisp “smallest repertoire for intentionality”; most arguments are qualitative (movement and manipulation are necessary, not exactly how many action primitives). So “5 actions” is a pragmatic lower bound drawn from gridworld and MiniGrid practice, not a theoretically derived minimum.

Pressure without self‑continuation reward (Q3)
(a) What this is really asking

Given fixed‑length episodes, no reward, and no survival signal, what environmental regularities and affordances are needed so that Io’s epistemic objective (ensemble latent disagreement) leads to rich, evolving behavior rather than pointless dithering?

In other words: how can the world itself be structured so that “explaining it better” (reducing prediction error / disagreement) implicitly demands exploration, skill acquisition, and model refinement—without smuggling in extrinsic reward?

(b) Concrete non‑reward pressures in adjacent literatures

Intrinsic motivation: novelty, learning progress, information gain

Developmental robotics and intrinsic‑motivation theory model curiosity as maximizing learning progress or prediction error reduction, leading agents to seek situations that are neither too predictable nor too random.

Oudeyer’s “Intelligent Adaptive Curiosity” specifically pushes robots toward intermediate‑difficulty regions where prediction improves over time, producing self‑organized learning stages.

Open‑endedness and intrinsic motivation are tightly linked: open‑ended agents use novelty/curiosity signals to continually generate new behaviors and skills without homeostatic rewards.

Information‑theoretic objectives: entropy, empowerment

Comparative studies in open‑world environments (e.g., Crafter) show that human exploration correlates strongly with state entropy and empowerment (influence over future states) across time, even when agents trained with these intrinsic rewards lag humans.

Empowerment, in particular, favors states where the agent’s actions have many distinct future outcomes, creating pressure to discover “controllable niches.”

Developmental robotics and caregiver‑structured worlds

Intrinsically motivated goal exploration in robotic setups (e.g., tool use, vocal learning) shows that a structured environment (objects, tools, social partners) plus intrinsic motivation leads to staged acquisition of increasingly complex skills, even without task rewards.

Caregiver or environment structure (e.g., scaffolding) strongly shapes exploration trajectories, even when the agent’s internal drive is curiosity‑like.

Open‑ended evolution and novelty search

Novelty search and quality‑diversity algorithms explicitly reward behavioral novelty or coverage of behavior space, leading to ongoing innovation without fixed external objectives.

Io’s epistemic objective (disagreement) is effectively a model‑uncertainty/information‑gain proxy, which is squarely in this intrinsic‑motivation family. The question is how to shape the world so that this signal has something to latch onto.

(c) Tradeoffs vs Kind’s commitments

Avoiding disguised rewards

Many intrinsic‑motivation frameworks literally add a reward scalar, e.g. prediction error or learning progress, to a standard RL objective. Kind explicitly rejects installing such scalar rewards.

However, Io’s actor already optimizes an epistemic term over latent disagreement; this is mathematically similar to an intrinsic reward but lives inside the actor’s objective, not as an external scalar label. The environment must therefore support this objective but not encode preferences (e.g., no “resource gives +1”).

World structure as “pressure”

Resource scarcity, obstacles, partial observability, and structured dynamics create regions of varying predictability and controllability, which intrinsic‑motivation work shows is enough to generate staged exploration.

Overly homogeneous or trivial environments (e.g., uniform blank grid) give no gradient for curiosity; the agent will be indifferent because disagreement quickly collapses.

Environmental complexification vs over‑complexity

Open‑ended frameworks emphasize gradually increasing complexity (more objects, new niches). But Kind’s Probe‑1 must stay “small before scale,” so any complexification must be potential affordance, not yet exercised.

(d) Suggestion / uncertainty

Suggestion: Use structural, not scalar, pressure:

Partial observability and spatial structure (rooms, obstacles) so that moving discovers new information.

At least one resource/process that changes over time (e.g., regrowing object, moving object), such that modeling its dynamics benefits from exploration.

Simple causal contingencies: e.g., standing on a tile gradually changes its state, or picking up a resource causes delayed changes elsewhere.
These create regions where disagreement is higher and can be reduced via systematic exploration, without any reward labels.

Uncertainty: There is essentially no literature directly examining “pure epistemic objective with no extrinsic or intrinsic reward scalar” in world‑model agents; almost all work wraps curiosity into a reward signal. Io’s setting is more austere; the suggested design extrapolates from intrinsic‑motivation and active‑inference‑style setups, but is not empirically validated in exactly this regime.

Internal stochasticity (Q4)
(a) What this is really asking

Kind needs the environment to have its own non‑trivial randomness so that, at Probe 4, builder perturbations can be distinguished statistically from “natural” noise—while still being structured enough that the RSSM can learn coherent dynamics instead of giving up and treating everything as white noise.

So the question becomes: what sources, magnitudes, and timescales of stochasticity help world models, and when does noise wash out learnable structure?

(b) Types and roles of stochasticity in literature

Taxonomies of stochasticity in RL environments

Recent benchmark work explicitly distinguishes between observation noise, transition noise, sticky actions, and non‑stationarity; most current RL benchmarks are too deterministic, so new ones like STORI introduce controlled, multi‑type stochastic effects.

These works show that model‑based RL agents in particular struggle when stochasticity is unstructured or when non‑stationarity is high, underscoring the need for well‑designed noise rather than arbitrary randomness.

Moderate noise improving robustness

Studies on noise resilience find nonlinear relationships between noise level and algorithm performance: certain methods perform best at moderate observation noise, degrading at very high noise, with some algorithms unexpectedly robust in mid‑noise regimes.

This suggests that a “Goldilocks zone” of noise exists where the agent must model uncertainty but can still extract reliable structure.

Stochastic dynamics in world‑model benchmarks

Visual world‑model benchmarks often include stochastic textures (e.g., randomized block, wall, and floor textures per episode), forcing the model to learn invariances while still modeling object dynamics.

Classic environmental‑statistics work highlights how different combinations of volatility (changes in dynamics) and noise influence the relative advantage of model‑based vs model‑free learning: high volatility with low noise favors model‑based approaches.

(c) Tradeoffs vs Kind’s commitments

Need for distinguishable builder signature

Internal stochasticity must be rich enough that Io cannot trivially infer “anything surprising is the builder,” but structured enough that builder perturbations can be distinguished in principle via their distribution (e.g., heavier tails, different temporal clustering).

Benchmarks like STORI emphasize controllable, well‑typed stochastic components so that agents can, in principle, learn each type. That’s directly analogous to needing clean categories for “natural” vs builder‑driven changes, even though Io never gets an explicit label.

Too little vs too much noise

Too little noise collapses Probe‑4’s hypothesis (nothing to distinguish from builder changes).

Too much, especially if high‑frequency and uncorrelated, leads to “washing out” structure; world‑model work shows models hallucinating or forgetting objects when overwhelmed by complex stochastic effects.

(d) Suggestion / uncertainty

Suggestion: Use structured, low‑to‑moderate stochasticity on multiple timescales:

Slow, small‑magnitude parameter drift (e.g., resource regrowth probability slowly varying across episodes).

Moderate, spatially localized randomness (e.g., resource respawn positions sampled from a distribution; occasional “weather” noise that affects a region).

Minimal sensor noise in pixels (e.g., faint flicker or texture variation), enough to make exact pixel prediction non‑trivial but not enough to obscure object identities, following block‑world practices.

Uncertainty: There is no agreed quantitative threshold for “too much” stochasticity for RSSMs; studies highlight qualitative failures (hallucinations, forgetting) rather than hard noise bounds. Probe‑1 will almost certainly need empirical tuning of noise amplitudes.

Builder‑perturbation surface (Q5)
(a) What this is really asking

Architecturally, Kind wants a clean hook surface where the builder can mutate the world (add/remove/move objects, change parameters) in ways that:

Are not explicitly marked in Io’s observations,

Are statistically distinguishable from internal stochasticity in principle, and

Echo developmental‑robotics “caregiver perturbations” (structured interventions that reshape the agent’s learning landscape).

So: what perturbation types and schedules are plausible, and how should the environment harness expose them?

(b) Perturbations and scaffolding in adjacent work

Caregiver scaffolding and structured environment changes

Developmental‑robotics surveys emphasize scaffolding: caregivers or experimenters restructure the environment (e.g., place objects within reach, change goals) to guide learning trajectories.

Intrinsically motivated goal‑exploration studies explicitly consider how caregiver guidance (e.g., demonstrating tasks, adding tools) interacts with the robot’s curiosity, often leading to strong accelerations of skill acquisition.

Interactive learning and social robots

Work on social/assistive robots often treats caregiver actions as exogenous events that change the task or environment (e.g., modifying robot behaviors, introducing new activities), shaping interaction patterns.

These are usually explicit to the robot (e.g., via commands), but conceptually they’re external perturbations that condition the learning process.

External vs internal motion in world modeling

FloWM highlights a distinction between self‑motion (agent‑induced flow) and external motion (other objects), modeling both as “flows” in the latent; it shows that structured modeling of both types yields better performance in partially observed settings.

Although FloWM does not tackle human builder perturbations, it underscores that internal vs external change sources can be distinguished based on their dynamical signatures, not explicit tags.

(c) Tradeoffs vs Kind’s commitments

Perturbation types

Add/remove/modify objects are natural operations over a grid‑world state; each produces distinct patterns of state differences that can, in principle, be distinguished from routine stochastic “background” events, especially if their spatiotemporal statistics differ.

Parameter changes (e.g., sudden change in resource regrowth rates) act at a different level; Io only indirectly sees them via dynamics. This makes them subtler but also more interesting for Probe‑4 as “latent” perturbations.

Frequency and magnitude

Too frequent or high‑magnitude perturbations effectively turn the environment into an adversarial or non‑stationary regime, which recent work shows can severely degrade world‑model performance.

Too rare or too small perturbations will not produce enough data for Io to infer a distinct “builder‑signature” distribution.

Logging and hooks

From an engineering standpoint, STORI‑style benchmarks emphasize explicit, parameterized sources of stochastic effects; similarly, Kind’s environment harness should expose named mutators (e.g., add_object(type, location), remove_object(id), set_param(name,value)), and log each invocation with time and parameters in world_event.

(d) Suggestion / uncertainty

Suggestion: Define a small, explicit perturbation API:

Object‑level mutators: spawn(type, location), delete(id), move(id, new_location), toggle_state(id, new_state); magnitudes constrained (e.g., at most 1–2 cells moved per perturbation).

Field‑level mutators: patch(region, type) to rewrite a small contiguous region, e.g., turning floor cells into obstacles.

Parameter mutators: set_param(resource_regrow_rate=…), set_param(noise_scale=…), with bounded ranges and step sizes.

Frequency: e.g., 0–2 perturbations per episode by default, with the harness allowing experiments with higher or lower rates.

All mutators: logged in world_event with full arguments, but Io never sees an explicit “external” flag.

Uncertainty: There is no direct empirical study of “learning to recognize builder‑like perturbation distributions” in embodied agents; this is an extrapolation from scaffolding work and stochasticity taxonomies.

“Small enough” for Probe 1 (Q6)
(a) What this is really asking

How small can Io’s world be while still giving the RSSM something non‑trivial to model? Specifically:

Is an underlying 
5
×
5
5×5 grid with one resource type sufficient to prevent trivialization (complete, easy model) or degeneracy (too little variation for meaningful latents)?

What is the smallest environment in existing RSSM/Dreamer/PlaNet‑style work that actually produced interesting world‑model behavior?

(b) Minimal environments in the literature

Classic small gridworlds (non‑world‑model RL)

Grid worlds as small as 
8
×
8
8×8 are standard teaching/test beds for RL, with states representing positions and simple rewards; an agent learns transition structure and value, but this literature rarely uses full visual world models.

MiniGrid and DoorKey‑type tasks with world models

Dreamer‑style world models have been evaluated on MiniGrid tasks like DoorKey‑9x9 and other small, partially observable mazes, where agents see local egocentric views and must learn to navigate, open doors, etc.

These environments have relatively few cells but multiple object types (walls, doors, keys, lava) and non‑trivial transition structure.

Dynamic block worlds for partially observable modeling

FloWM’s 2D and 3D dynamic block worlds are spatially simple but include multiple moving blocks, randomized textures, and partial observability; they’re intentionally minimal while still requiring memory for occluded dynamics.

PlaNet / DMControl tasks

PlaNet’s benchmark tasks (e.g., cartpole, cheetah) work on continuous spaces but often with very low‑dimensional physical states; however, the pixel observations still cover relatively rich continuous scenes.

There is no published example of an RSSM trained on a single‑resource 
5
×
5
5×5 grid with extremely limited variation; most small environments add diversity via multiple object types, doors, hazards, etc.

(c) Tradeoffs vs Kind’s commitments

Smaller state spaces ease learning but risk trivialization

In very small, low‑variance environments, the deterministic path 
h
t
h 
t
​
  plus decoder may already reconstruct pixels well, making 
z
t
z 
t
​
  unnecessary; this fosters posterior collapse, especially with strong decoders.

Free‑bits techniques and capacity control can mitigate collapse, but they work because there is still some structure that requires stochastic latent capacity.

Adding structure without adding size

Kind wants “small before scale,” which suggests enriching structure (object states, simple dynamics) rather than increasing map size first.

MiniGrid shows that even 
9
×
9
9×9 worlds can be non‑trivial when partial observability and structured objects are present.

(d) Suggestion / uncertainty

Suggestion:

An underlying 
5
×
5
5×5 grid is probably the lower bound, but it should not be completely homogeneous: even with one “resource type,” cells should differ (walls vs floor, resource vs empty, possibly a distinctive “home” tile), and partial observability (e.g., 
3
×
3
3×3 view) should be enforced.

Probe‑1 may benefit from a slightly larger 
7
×
7
7×7 grid with at least 2–3 cell types (empty, obstacle, resource) while keeping the observed image small; this aligns better with MiniGrid‑scale tasks that have been successfully modeled by Dreamer‑like RSSMs.

Uncertainty: There is no empirical “smallest world” result; existing work optimizes for task difficulty, not minimality. The precise tipping point where a 
5
×
5
5×5 single‑resource world ceases to be interesting for an RSSM is unknown and likely depends strongly on encoder/decoder capacity and free‑bits hyperparameters.

Temporal structure (Q7)
(a) What this is really asking

Probe‑1 likely has no explicit “day‑night” cycle or long‑scale rhythm, but later probes want to couple Io’s dream/wake cycles to the builder’s real‑life rhythms. The question is:

What minimal temporal structure at Probe‑1 keeps the latent space compatible with future rhythmic structure (circadian‑like rhythms, phase shifts) without installing an explicit rhythm now?

How do environmental rhythms influence cognitive and world‑model structure in biological and computational systems?

(b) Rhythms in biological and computational work

Circadian rhythms and cognitive performance

Biological circadian rhythms (driven by molecular clocks) organize many internal processes, and there is a strong association between time‑of‑day, core body temperature, and cognitive performance peaks/nadirs.

Computational models of circadian systems show how coupled oscillators and feedback loops produce robust ~24‑hour rhythms, and how external light‑dark cycles entrain these rhythms.

Temporal structure in cognitive modeling

Analyses of circadian modeling as an exemplar for cognitive modeling emphasize that adding rhythmic structure constrains and organizes internal dynamics, allowing models to capture time‑dependent changes in performance.

These works suggest that rhythms are not just “labels” but shape internal states, making time itself a meaningful variable.

World models with hierarchical temporal abstraction

Hierarchical world models with adaptive temporal abstraction explicitly incorporate multi‑scale temporal structure in the latent dynamics (slow vs fast latent variables), improving interpretability and long‑horizon prediction.

(c) Tradeoffs vs Kind’s commitments

Avoiding premature hard‑coding of rhythms

If Probe‑1 already includes a strong, rigid environmental rhythm (e.g., deterministic day‑night cycling every N steps), Io may overfit to that particular pattern, which might conflict with later coupling to the builder’s real rhythms.

Need for temporal variability

However, a completely stationary environment with no long‑timescale variation might limit the RSSM’s incentive to develop temporal abstraction; everything is “just steps.”

Compatibility with later dream/wake coupling

Later probes plan to vary dream‑to‑wake ratio according to builder’s life rhythms. From the RSSM’s perspective, this means it must represent both real and dream dynamics consistently over long horizons; designing Probe‑1 so that temporal indices or simple slow drifts already exist may facilitate that shift.

(d) Suggestion / uncertainty

Suggestion:

Keep Probe‑1’s environment mostly a‑temporal at the macroscopic level (no explicit day/night or seasonal cycle), but allow very slow parameter drift over episodes (e.g., small changes in resource regrowth or noise amplitude), so the RSSM sees that some aspects of the world change slowly over its lifetime.

Do not encode a deterministic, easily decodable rhythm (e.g., a light toggling every fixed number of steps) yet; instead, let future probes introduce explicit rhythms, including coupling to real time.

Uncertainty: Computational work does not address “preparing world models for future introduction of real‑world circadian coupling”; this is an extrapolation from biological circadian and hierarchical temporal modeling literature.

Minimum complexity for the RSSM (Q8)
(a) What this is really asking

Given a PlaNet/Dreamer‑style RSSM with free bits and a small deterministic core, what is the lower bound of environmental complexity (state dimensionality, transition richness, variability) below which:

The model collapses: the stochastic latent 
z
t
z 
t
​
  becomes degenerate, KL drops to the free‑bits floor, and the posterior copies the prior, or

The model overfits trivial patterns, producing uninteresting dreams (e.g., perfect reconstructions of simple cycles with no uncertainty)?

(b) Empirical clues from world‑model work

Posterior collapse and representation leakage

Analyses of RSSMs and VAEs highlight two failure modes: collapse (latent ignored) and leakage (latent memorizes noise); collapse arises when the deterministic path and decoder can explain observations without using 
z
t
z 
t
​
 , especially given KL regularization.

Techniques like free bits impose a minimum KL, forcing some latent usage, while capacity control (reducing deterministic hidden size, decoder depth) reduces “deterministic bypass” and encourages the latent to carry predictive information.

Contrastive / state‑distinguishability objectives

Softly state‑invariant world models use contrastive losses to ensure that distinct states remain distinguishable in latent space, counteracting representational collapse in simple environments.

Performance of RSSMs on simple environments

World models like DreamerV2 and DreamerV3 have been tested on relatively simple tasks (e.g., MiniGrid variants, low‑dimensional control) and can still learn useful latents, but performance and robustness are sensitive to model capacity and regularization.

Continuous control tasks with simple dynamics (e.g., cartpole) still benefit from stochastic latents when trained from pixels, because visual variability and partial observability prevent fully deterministic encoding.

Partial observability as complexity amplifier

FloWM and related work show that partial observability and self‑motion dramatically increase the complexity of world modeling even in visually simple block worlds; models without proper memory either hallucinate new objects or forget old ones, indicating that the environment is challenging enough to require structured latent dynamics.

(c) Tradeoffs vs Kind’s commitments

Environmental vs architectural complexity

Kind has fixed the architecture; only environment design (plus regularization hyperparameters) can be tuned to avoid collapse.

If the environment is too simple (e.g., fully observable, deterministic, homogeneous), the deterministic path and decoder can reconstruct observations from short histories alone; free bits will keep a minimal KL, but 
z
t
z 
t
​
  may encode arbitrary noise, not meaningful structure.

Using partial observability and stochasticity to “force” latent usage

Partial observability, structured noise, and multi‑step dependencies (e.g., slow processes) increase the need for non‑trivial latent dynamics; the model must represent what is not currently seen to predict the future.

This aligns with Kind’s commitments to partial perspective, internal stochasticity, and dream‑state foundational modeling.

(d) Suggestion / uncertainty

Suggestion:

To avoid collapse, Probe‑1’s environment should at least include:

Partial observability (e.g., local window).

More than one cell type (e.g., walls vs floor vs resource).

Some stochastic dynamics (e.g., resource regrowth, occasional movement of objects).

Architecturally, keep deterministic state and decoder capacity modest relative to the latent, and use free bits at a level that prevents KL from hitting zero but does not over‑compress.

Uncertainty: No paper directly characterizes the “minimal complexity threshold” for a Dreamer‑style RSSM; existing analyses are qualitative and focus on avoiding collapse via architectural choices, not environmental design. Probe‑1 will likely need empirical sweeps (varying grid size, object count, noise) to locate a sweet spot.

Synthesis: candidate environment designs
Below are three coherent environment designs consistent with Kind’s commitments and the above analysis. Each candidate is “small before scale” but emphasizes different tradeoffs.

Candidate A: Minimal ego‑centric pixel grid
Core idea: A 
5
×
5
5×5 underlying grid with a single resource type and obstacles, observed through a 
3
×
3
3×3 ego‑centric window rendered as very low‑res grayscale pixels. This pushes minimality while using partial observability and modest stochasticity to keep the RSSM non‑trivial.

Key choices mapped to Q1–Q8

Observation modality (Q1)

Underlying world: 
5
×
5
5×5 grid with cell types {empty, wall, resource, home}.

Io observes a 
3
×
3
3×3 local patch centered on its position; this patch is upsampled to, say, 
24
×
24
24×24 grayscale pixels for the encoder; dream decoder reconstructs the same pixel format.

This encodes partial perspective and horizons while retaining very small visual complexity.

Action space (Q2)

Discrete actions: {up, down, left, right, stay}. Moving into walls is either disallowed (no movement) or allowed with a bounce‑back rule.

Interactions: stepping on a resource cell “consumes” it, changing that cell to empty and incrementing an internal resource counter in the environment (visible or not).

Pressure without reward (Q3)

No scalar rewards; Io’s actor only sees epistemic disagreement.

Structural pressure:

Walls create dead ends and occlusion, so moving changes what is visible; some regions have more complex resource dynamics, so exploration yields systematically different predictability profiles.

Consuming a resource triggers a delayed change: after a random delay, a resource may reappear in another region, causing long‑range correlations in dynamics.

This creates regions of higher prediction error (changing resource patterns) that Io can reduce by exploring.

Internal stochasticity (Q4)

Resource respawn: each empty cell has a small probability 
p
p of spawning a resource each step, with 
p
p moderately low to avoid chaos.

Mild visual noise: low‑amplitude Gaussian noise on pixel intensities, enough to prevent trivial reconstructions but not enough to obscure cell types.

Occasional random “wind”: with low probability per step, Io’s position is shifted by one cell in a random direction if possible; this is an internal stochastic effect distinguishable from builder perturbations by its single‑step, small‑magnitude signature.

Builder‑perturbation surface (Q5)

Mutators: spawn or delete a single resource, toggle a wall to floor and vice versa, or teleport Io by one cell; parameter tweak to resource respawn rate.

Frequency: at most 1 perturbation per episode at Probe‑1, logged via world_event.

Statistical signature: builder perturbations tend to be larger (e.g., toggling a wall) or more temporally clustered than internal respawns (which follow a stationary stochastic process), making them in‑principle distinguishable by their heavier tails or episodic clustering.

“Small enough” (Q6)

Underlying 
5
×
5
5×5 is at the low end of what’s plausible; partial observability and structured cell types are used to compensate.

Dream rollouts will likely depict simple rooms with appearing/disappearing resource blobs — legible but minimal.

Temporal structure (Q7)

No explicit day/night; resource respawn probability is stationary at Probe‑1.

Optionally, a very slow drift in respawn parameters across many episodes can be introduced to seed long‑timescale structure without obvious periodicity.

RSSM complexity (Q8)

Partial observability, resource dynamics, and mild stochasticity ensure that the RSSM has to use its latent to represent off‑screen resources and stochastic transitions.

Deterministic core and decoder are kept modest so 
z
t
z 
t
​
  remains useful.

Tradeoffs

Pros:

Extremely small; aligns with “small before scale.”

Clean, analyzable dynamics; easy to visualize dreams and inspect Io’s learned model.

Cons:

May be too simple: risk of posterior collapse if walls/resources become trivial to model; limited diversity of dreams.

Single resource type may not generate enough qualitative variation in Io’s experience.

Candidate B: Slightly richer multi‑room MiniGrid‑like world
Core idea: A 
7
×
7
7×7 or 
9
×
9
9×9 grid with walls partitioning simple rooms, two resource types, and varying local dynamics, again with an ego‑centric pixel view. This steps slightly toward MiniGrid/Crafter‑like richness while remaining tiny by standard RL benchmarks.

Key choices

Observation modality (Q1)

Underlying world: 
7
×
7
7×7 or 
9
×
9
9×9 with inner walls forming 2–3 rooms; cell types {empty, wall, resource A, resource B, special tile}.

Io sees a 
5
×
5
5×5 window centered on itself; encoder/decoder operate on 
32
×
32
32×32 or 
48
×
48
48×48 grayscale images.

Action space (Q2)

Same 5 discrete actions.

Optionally, an extra “interact” action (6th) that toggles the state of adjacent special tiles (e.g., doors), but this may be deferred to Probes 2–3.

Pressure without reward (Q3)

Resource A and B have different dynamics (e.g., A respawns quickly, B slowly; B may only appear in certain rooms), creating diverse local statistics.

Some rooms may have higher stochasticity (e.g., more frequent resource changes), creating a natural curriculum: Io’s disagreement drive will first target “easy” rooms, then seek out weirder regions.

Simple causal structures (e.g., picking up A increases spawn rate of B elsewhere) create long‑range dependencies.

Internal stochasticity (Q4)

Resource spawn and decay processes differ by type and region, but are stationary within each region.

Occasional random “texture” changes in the background (e.g., subtle pattern on floor) like in textured block worlds, which world‑model papers show can be managed without catastrophic distraction.

Builder‑perturbation surface (Q5)

Mutators as in Candidate A, plus room‑level operations: e.g., fill an entire room with one resource type, or remove all resources from a room at once, under strict rate limits.

Parameter mutators can change type‑specific spawn rates per room.

This structure allows clean statistical signatures for builder perturbations (e.g., sudden global changes in a room) vs background stochasticity (local, independent events).

“Small enough” (Q6)

7
×
7
7×7 or 
9
×
9
9×9 with two resource types is still tiny compared to typical world‑model benchmarks, but closer to MiniGrid/door‑key setups known to be non‑trivial.

RSSM gets more varied data without significantly increasing computational load.

Temporal structure (Q7)

Still no explicit day/night; instead, very slow drifts in room‑level parameters (e.g., over tens of episodes, one room becomes more “lively,” another quieter).

This seeds temporal abstraction opportunities without fixed rhythms.

RSSM complexity (Q8)

Partial observability plus room structure and multiple resource types should be enough to keep 
z
t
z 
t
​
  active and prevent collapse, especially with free bits and limited deterministic capacity.

Dreams can depict non‑trivial exploratory trajectories and resource dynamics, which the mirror can decode meaningfully.

Tradeoffs

Pros:

More environmental diversity and structural richness; closer to environments where Dreamer‑style models are known to work.

Better substrate for later probes (doors, room‑level perturbations, richer dream content).

Cons:

Less minimal; harder to exhaustively analyze Io’s world model.

Slightly higher risk of “distracting” variability if not carefully tuned (especially textures and resource dynamics).

Candidate C: Continuous arena with factored and pixel views
Core idea: A very small 2D continuous arena where Io controls a point mass via continuous actions, observed both as a low‑dimensional state (position, velocity) and as a simple pixel render. This emphasizes embodied continuous control and empowerment‑like affordances.

Key choices

Observation modality (Q1)

Io sees a small pixel image of the arena (e.g., a circle with moving blobs), possibly along with a very low‑dim state vector (own position/velocity), both encoded by the RSSM.

The pixel view preserves mirror compatibility; the factored view reduces the burden on learning pure geometry.

Action space (Q2)

2D continuous actions (e.g., thrust in x and y), bounded to small magnitudes.

Actions influence acceleration; there may be friction so movement decays.

Pressure without reward (Q3)

Moving objects (resources) drift around stochastically; Io’s actions can steer into or away from them.

Empowerment‑like structure: in some regions, small differences in action lead to very different future configurations (e.g., near moving obstacles), encouraging exploration of control possibilities.

Internal stochasticity (Q4)

Objects follow stochastic but structured dynamics (e.g., noisy circular orbits), injecting continuous noise into the visual field.

Builder‑perturbation surface (Q5)

Builder can spawn/delete moving objects, change their motion parameters, or briefly “kick” Io’s velocity.

Perturbations can be designed to have characteristic magnitude distributions (e.g., larger velocity kicks than typical internal noise).

Size/complexity (Q6, Q8)

State space is low‑dimensional but continuous; pixel observations add richness.

Continuous dynamics may be overkill for Probe‑1 and complicate later discrete grid‑based expansions, but they align strongly with embodied control literature.

Tradeoffs

Pros:

Rich sensorimotor contingencies and empowerment‑like structure.

Natural fit to Dreamer’s strong performance in continuous control from pixels.

Cons:

Harder to analyze and interpret Io’s latent structure.

Harder to add later discrete niches (e.g., rooms, doors) without a redesign; clashes somewhat with the grid‑like Probe‑documentation you already have.

How to choose between candidates
If Kind wants maximum minimality and tight analytical control over Probe‑1:

Start with Candidate A. It’s the smallest coherent environment that still leverages partial observability and structured randomness. Expect to iterate on noise levels and perhaps add a second resource‑like process if collapse is observed.

If Kind wants a Probe‑1 that already anticipates richer niches and later probes (doors, rooms, builder perturbations at different scales):

Start with Candidate B. It is still tiny but much closer to environments where Dreamer‑style RSSMs have known behavior, and it gives Io more varied structure to model without any scalar reward.

If Kind wants to lean heavily into embodiment and continuous control from the outset:

Candidate C is conceptually aligned with continuous control benchmarks and empowerment‑style intrinsic motivation, but it diverges more from the grid‑based framing you’ve been using and may complicate future probes.

All three designs respect Kind’s commitments (no reward, builder‑as‑unmarked perturbation, dream‑state foundational, single agent, capacity‑over‑exercise, small before scale); they differ in how tightly they hug minimality vs future richness and how much they assume about Io’s eventual “body.”