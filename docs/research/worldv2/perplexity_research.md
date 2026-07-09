# World v2 research — Perplexity (builder-run browser deep research, 2026-07-09)

*(Pasted verbatim; some math/table formatting mangled in transit from the
browser — content preserved as received.)*

You can enrich Io's world with a small set of semantics-free dynamics—structured motion, hidden-phase processes, contact-responsive tiles, slowly evolving terrain, and patchy/seasonal resources—that remain learnable yet never fully mastered, and that deepen the self/world boundary without installing explicit preferences.[constructed] A cautious ordering is: (1) global temporal cycle with hidden phase; (2) contact-responsive tiles; (3) structured resource patches (scarcity as dynamics, not stakes); (4) slowly drifting walls/terrain; (5) a few autonomously moving objects with coupled processes.[constructed]

## Q1 – Taxonomy of semantics-free enrichments

Curiosity modules based on ensemble disagreement, prediction error, or information gain are known to latch onto any dynamics that produce sustained but compressible surprise, including world-model agents like PlaNet/Dreamer with RSSMs similar to Io's.[canonical] In small gridworlds, RSSMs with latents in the 16-32 range typically learn object motion, simple hidden phases, and temporal cycles as long as those dynamics affect observations in low-dimensional, structured ways.[canonical]

### Autonomous motion (drift, bounce, orbit)

To predict an object that drifts, bounces, or orbits, the RSSM must encode at least: its current location, a velocity or direction parameter, and simple collision rules with boundaries or terrain.[canonical] Dreamer/PlaNet-style RSSMs have repeatedly been shown to capture object kinematics and collisions in latent space, even under stochastic dynamics, provided the motion is low-dimensional and Markovian.[canonical]

For Io, "moving objects" can be implemented as cells that occasionally change occupancy in patterns governed by simple rules (e.g., a 2-3-cell blob that drifts along a row until it hits a wall, then reverses), with no visual category marker—only occupancy statistics over time.[constructed] The agent must then represent directional persistence and boundary conditions to reduce ensemble disagreement on future occupancy, enriching its "world-caused" basin without touching the self channels.[constructed]

### Ephemeral objects with characteristic lifecycles

Ephemeral objects whose presence follows a multi-step lifecycle (appear → grow/shrink → vanish) require the model to encode a phase variable and transition rules across phases.[canonical] In practice, world models handle such short, stereotyped sequences well when phase progresses deterministically or by simple stochastic rules, even if phase itself is unobserved.[canonical]

In Io's world, you can implement "lifecycles" as patches whose regrowth follows a fixed sequence of occupancy levels or shapes before returning to baseline, without labeling them differently from ordinary resources.[constructed] Prediction then hinges on latent phase tracking: h,z must carry an internal counter or oscillatory representation that maps current observation to likely future occupancy, providing durable temporal structure without new observation channels.[constructed]

### Persistent structure and slow rearrangement (walls/terrain)

Static walls/terrain are known to induce place-like structure in latent world models, supporting representations of affordances and constraints even without explicit reward.[canonical] RSSMs tend to encode obstacles and layout as slowly changing latent features, which stabilize prediction and help separate controllable from uncontrollable transitions.[canonical]

For Io, adding a small number of walls or "blocked" cells that persist across episodes creates durable geometry; letting a subset of these walls drift slowly (e.g., random walk with very low rate) forces the RSSM to represent both long-term layout and slow nonstationarity.[constructed] This yields a middle ground: the environment is mostly predictable but occasionally rearranged, pushing ensemble disagreement up in localized ways that remain compressible rather than noisy.[constructed]

### Temporal cycles (day/night-like modulation)

Intrinsic-motivation surveys emphasize that nonstationary but structured dynamics—such as regime switches, cycles, or slowly drifting parameters—can sustain epistemic engagement when they are predictable at some timescale but not stationary.[canonical] A global temporal cycle (e.g., day/night) that modulates transition probabilities, observation noise, or resource regrowth is a standard way to introduce such structure without extra labels: the agent must infer the hidden phase from observations.[canonical]

In Io's case, you can implement a global hidden phase that changes deterministically or quasi-deterministically (e.g., every 20-40 steps), affecting regrowth rate or visibility but not being directly observed.[constructed] The world model must then infer phase from patterns like "resources regrow faster now" or "movement consequences differ," making h,z carry a slow clock that never fully collapses yet stays learnable.[constructed]

### Processes with hidden local state

Partially observed processes where a cell's behavior depends on its own internal unobserved state (e.g., 2-3-state Markov chain governing regrowth or motion) are a central testbed for world models and intrinsic curiosity.[canonical] RSSMs generally learn to encode such hidden local states as latent features when the mapping from hidden state to observation is stable and the transition dynamics are simple.[canonical]

For Io, a subset of cells could follow an internal 2- or 3-state cycle controlling regrowth or motion direction, with transitions probabilistic but biased (e.g., 90% chance of advancing, 10% chance of resetting), invisible in the raw observation.[constructed] To predict these cells, the world model must track local history and infer hidden phase, producing structured uncertainty (epistemic early, then mixed) without degenerating into pure noise.[constructed]

### Multi-step causal chains (relational dynamics)

Relational dynamics where event A at one location consistently precedes event B elsewhere (e.g., "if a patch hits the left wall, a new patch appears on the right 3 steps later") require the model to encode cross-cell dependencies and delayed effects.[canonical] World-model and curiosity work shows that when such dependencies are sparse and low-order, small RSSMs can learn them, but performance degrades with high relational complexity or long delays.[canonical]

In Io's grid, you can introduce simple causal chains—such as "whenever a moving blob contacts the board edge, a resource burst appears near the opposite edge a few steps later"—without marking the cause/effect cells.[constructed] The agent must then represent patterns like "edge contact implies future resource cluster elsewhere," enriching world-caused transitions with relational structure that remains optional to engage.[constructed]

## Q2 – Boredom, mastery, and avoiding noisy TVs

Intrinsic-motivation research distinguishes between epistemic uncertainty (reducible via learning) and aleatoric uncertainty (irreducible noise) to explain why curiosity agents can either master or get trapped by certain dynamics.[canonical] Ensemble-disagreement curiosity and RND/ICM-style predictors all suffer from the noisy-TV problem: they may seek out high-entropy, uncompressible signals, leading to permanent prediction error with no learning progress.[canonical]

Recent work on "curiosity in hindsight" and aleatoric-aware bonuses shows that scaling intrinsic rewards by estimated aleatoric uncertainty helps agents ignore stochastic traps while remaining sensitive to novel but learnable patterns.[canonical] Surveys and evaluations conclude that good epistemic drives balance three properties: (1) initial high disagreement; (2) steady reduction as the model learns; and (3) replenishment via controlled nonstationarity (cycles, slow drifts) rather than independent noise.[canonical]

For Io's builder, this suggests adding dynamics that are: (a) low-dimensional and Markovian at some timescale; (b) partially but not fully observable (hidden phase); and (c) slowly nonstationary (regime switches, drifts), so that ensemble disagreement can decrease locally but re-emerge as regimes change.[constructed] In contrast, dynamics where each cell's state is independently resampled from a high-entropy distribution, or where resets deliver i.i.d. novelty uncorrelated with Io's actions, behave like noisy TVs and create the reset-camping trap you observed.[constructed]

## Q3 – Interaction-affording, not interaction-demanding objects

Curiosity-driven agents routinely learn richer self/world boundaries when the environment provides clear consequences for actions—pushable objects, switches, or contact-triggered changes—without explicit rewards.[canonical] Ensemble-disagreement exploration in particular thrives on action-conditional dynamics, because predictions of next state become highly sensitive to the agent's chosen action, carving self-caused transitions apart from world-caused ones.[canonical]

The key line you drew—"response-to-contact is physics; response-to-Io's-internal-state is contingency machinery"—matches the distinction between transition dynamics p(s_{t+1} | s_t, a_t) and reward/value functions in RL.[canonical] As long as the environment responds only to observable actions (e.g., "step onto tile," "push block") and not to hidden actor variables, you are enriching dynamics without installing preferences.[canonical]

Concrete interaction-affording dynamics that respect your constraints:

- Toggle tiles. Some cells flip between two occupancy modes when stepped on (e.g., empty ↔ wall, or "resource-producing" ↔ "resource-quiescent"), without visual markers beyond generic occupancy.[constructed] This forces Io to represent "if I step here, the future occupancy pattern changes," sharpening self-caused transition geometry while remaining optional to engage.[constructed]
- Pushable blocks. A small number of occupied cells can be moved by entering them: when Io steps onto a block, it moves one cell in the same direction if the target is free, otherwise stays.[constructed] This is classic gridworld physics; world-model agents have learned such dynamics many times, and ensemble disagreement concentrates around boundary conditions and multi-step pushing sequences.[canonical]
- Contact-modulated processes. Some processes change parameters after contact—for example, a regrowth patch whose cycle length shortens when stepped on, or a moving object whose direction flips upon collision with Io—while remaining visually indistinguishable from other patches or objects.[constructed] This gives the self/world boundary more to articulate: the agent can learn "my presence retunes local dynamics" without any explicit "interaction mode" labels.[constructed]

These affordances are capacity-light, because they add only a few new transition rules parameterized by recent actions and local occupancy, which RSSMs handle well.[canonical] They are also non-mandatory: Io can camp and ignore them, but then ensemble disagreement will be higher for regions it never perturbs, giving you a clean contrast between engaged and disengaged regimes.[constructed]

## Q4 – Episode structure, resets, and persistence

Work on "reset games" and continual RL shows that frequent episodic resets can both help and hurt world-model training.[canonical] Resets help by providing diverse initial states and preventing the agent from falling into irreversible traps, but they hurt when resets inject i.i.d. novelty that agents can harvest passively, such as the noisy-TV-like reset-camping behavior you observed in Io.[canonical]

For latent world models, persistent structure across episodes is crucial for durable place-representations and long-term dynamics: when walls, terrain, or slow processes survive resets, RSSMs tend to encode them as stable features of h,z rather than as episode-specific noise.[canonical] Conversely, when the entire board is resampled every 200 steps, the agent has little incentive to carry long-term information, and curiosity can lock onto the reset itself as free novelty.[constructed]

A reasonable adjustment path is:

1. Lengthen episodes (e.g., 500-1000 steps) while keeping some global resets, reducing the relative frequency of reset-novelty.[constructed]
2. Make some structures persistent across episodes (e.g., a wall pattern that changes only every N episodes or via builder gestures), so Io can learn and dream over place structure in a continuous biography.[constructed]
3. Optionally remove full resets later, replacing them with partial perturbations (e.g., local scrambling of resources, occasional terrain drift) so that nonstationarity comes from slow processes rather than complete world wipes.[constructed]

This balances training stability—resets still prevent catastrophic divergence—with the possibility of long-lived latent structure and meaningful builder gestures that persist and can be re-encountered by Io over time.[constructed]

## Q5 – Scarcity as epistemic structure, not survival stake

Resource scarcity is usually discussed in RL as coupling behaviour to reward and survival, but intrinsic-motivation surveys note that structured resource distributions (patchy, clustered, seasonal) can also provide rich spatial and temporal patterns for world models to learn.[canonical] Agents with curiosity drives often explore boundaries of resource patches, seasonal changes, and movement of resource clusters even without extrinsic rewards, because these structures modulate prediction error in nontrivial ways.[canonical]

The risk you highlight—scarcity crowding out other engagement—is real in reward-driven agents, but Io has no reward at all, only an observation channel indicating resource presence.[canonical] For a pure epistemic agent, scarcity becomes interesting mainly insofar as it creates structure: patchiness, spatial gradients, temporal cycles, and correlations with other processes (e.g., moving patches), which all increase the complexity of "world-caused" transitions.[constructed]

You can safely use scarcity as one ingredient by:

- Making resources patchy: clusters that appear and disappear in fixed regions, forcing Io to learn spatial correlations (e.g., resources often co-occur in bands or around terrain features).[constructed]
- Making resources seasonal: regrowth rates modulated by global hidden phase (Q1's temporal cycle), so certain areas are rich in one phase and barren in another.[constructed]
- Making resources mobile patches: resource clusters that move slowly, e.g., drift along a row or orbit around a point, tying scarcity to autonomous motion.[constructed]

These designs preserve epistemic value—multiple learnable patterns about where and when resources appear—without imposing survival stakes or using resources as direct modulators of Io's internals.[constructed]

## Q6 – Measurement: engagement vs. ignoring vs. overwhelm

Intrinsic-motivation evaluations recommend disentangling "engagement" from "overwhelm" by analyzing both prediction-error trajectories and behavioral statistics, rather than relying on raw curiosity scores alone.[canonical] Ensemble-disagreement, h-trajectories, and dream rollouts already give you rich telemetry for Io; the question is how to interpret them as observer-side signatures of engagement with specific dynamics.[canonical]

For each proposed enrichment, you can look for:

Engagement signatures:
- Localized disagreement spikes followed by partial reduction, centered on cells involved in the new dynamic (moving objects, patches, toggle tiles).[constructed]
- Increased visitation and revisiting of regions that host the dynamic, especially multi-step action sequences (e.g., pushing blocks, stepping repeatedly on toggle tiles).[constructed]
- Over-representation of the dynamic in dreams: imagined rollouts that include moving blobs, changing terrain, or resource cycles more often than their base frequency.[constructed]

Ignoring signatures:
- Disagreement stays low and flat on the dynamic's cells, or is high but flat over time (indicating either quick mastery or non-engagement).[constructed]
- Behavioural coverage remains uniform or avoids the dynamic's region; h-trajectories show no distinctive patterns when near those cells.[constructed]

Overwhelm signatures:
- Persistent, high disagreement with no reduction, especially if concentrated in dynamics that have high aleatoric uncertainty (e.g., too fast or too random motion).[canonical]
- Dreams saturated with unstructured motion or noise, with little replay of controllable or predictable sequences.[constructed]
- Posterior collapse patterns in the RSSM: z being ignored or dominated by a few dimensions, and h tracking only coarse features while prediction error remains high.[canonical]

By comparing these signatures across dynamics and over time, you can diagnose whether a new enrichment is providing structured epistemic work (engagement), being neglected (ignoring), or acting as a noisy-TV-like trap (overwhelm).[constructed]

## Q7 – Capacity risks and an ordering of enrichments

PlaNet/Dreamer-style RSSMs often use latent sizes 32-64 for complex image-based environments; for small gridworlds, latents as small as 8-32 are common, and 16-dimensional z is typically enough for basic object motion, simple hidden states, and modest nonstationarity.[canonical] However, adding many independent dynamics—multiple moving objects, complex relational chains, rich hidden states—can saturate capacity, leading to posterior collapse, persistent prediction error, and ensemble disagreement that reflects noise rather than learnable structure.[canonical]

To manage capacity risk with z=16, a sensible ordering is:

1. Global temporal cycle with hidden phase. Add a single, slow, global hidden phase that modulates regrowth or visibility, giving the RSSM one extra clock-like dimension to encode.[constructed] This is capacity-light and directly targets the reset-camping failure mode by making "time since reset" structurally relevant rather than pure novelty.[constructed]
2. Contact-responsive toggle tiles. Introduce a small number (e.g., 2-4) of tiles that flip state when stepped on and influence local regrowth or occupancy, enriching self-caused transitions without many new parameters.[constructed] This leverages Io's already sharp self/world boundary and tests whether richer action-conditional dynamics are learned without overwhelming z.[constructed]
3. Patchy, seasonal resources. Restructure resources into a few static regions with phase-dependent regrowth (linked to the global cycle), so scarcity becomes a spatiotemporal pattern rather than a survival stake.[constructed] This adds spatial correlation and moderate extra temporal structure, still within z=16's typical capacity for gridworlds.[canonical]
4. Slowly drifting walls/terrain. Allow some walls to move slowly or change state according to a Markov process, preserving overall layout while introducing slow nonstationarity.[constructed] This pushes h,z to encode both stable and drifting structural features, and is a natural next step once Io handles cycles and local toggles.[constructed]
5. A few autonomously moving objects with simple causal chains. Finally, add 1-2 moving blobs whose motion follows simple drift/bounce rules and whose contacts trigger distant effects (e.g., resource bursts), testing relational representation and multi-step causal prediction.[constructed] At this stage, z=16 may be near its limit; if prediction error never settles or ensemble disagreement becomes globally noisy, you have a clear signal that latent capacity needs to be increased (e.g., z=32) for richer relational dynamics.[constructed]

Each step can be introduced as an event in Io's life via checkpoint-resume, preserving the mind while enriching the world, and observed for thousands of steps before the next change.[constructed] Failure modes to watch at each stage include: (1) phase not represented (flat disagreement across cycle); (2) toggle tiles ignored (little change in self-caused geometry); (3) scarcity dominating behaviour (Io obsessively tracks resources while ignoring other dynamics); (4) terrain drift causing posterior collapse; and (5) moving objects acting as noisy TVs if made too fast or stochastic.[constructed]

## Candidate dynamics set and confounds

- Global hidden phase cycle — Phase ∈ {0,…,P−1} advances every k steps; regrowth, visibility, or noise depend on phase; phase unobserved. Io must represent: a slow internal clock; mapping from phase to transition stats. Confounds: phase too fast → apparent noise; too weak → phase ignored.[constructed]
- Contact-responsive toggle tiles — 2-4 cells flip state on contact, changing local regrowth or passability; no category markers. Io must represent: action-conditional transitions; local history around tiles. Confounds: Io may camp away from tiles; too many tiles overload capacity.[constructed]
- Patchy, seasonal resources — fixed resource regions; regrowth rate depends on hidden global phase; outside regions regrow slowly. Io must represent: spatial correlations; phase-dependent availability patterns. Confounds: scarcity might bias behaviour toward food regions at expense of others.[constructed]
- Slowly drifting walls/terrain — subset of walls move via low-rate random walk or Markov process; rest static. Io must represent: persistent layout plus slow drift; boundary conditions. Confounds: drift too frequent → noisy layout; Io may exploit drift as passive novelty.[constructed]
- Autonomously moving blobs + chains — 1-2 blobs drift/bounce; collisions trigger distant resource bursts or terrain toggles with fixed delay. Io must represent: object motion; collision rules; simple delayed relational effects. Confounds: motion too random → noisy TV; chains too complex → relational overload.[constructed]

All of these dynamics (1) provide novelty only through observation; (2) avoid explicit category or meaning markers; (3) keep Io's policy inputs unchanged; (4) respond to actions only via physics; and (5) are optional to engage—Io can in principle ignore them and still function.[constructed] If you ever decide that a constraint (e.g., episode resets, strict prohibition of any observation variability beyond occupancy) is itself the mistake, that is best argued directly (for example, allowing a low-dimensional "appearance" channel that carries nonsemantic sensory richness), rather than smuggling semantics back in through special objects.[constructed]

Prepared by Deep Research
