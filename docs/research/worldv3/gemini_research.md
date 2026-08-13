# World v3 research — Gemini voice (gemini-2.5-pro, 2026-08-13)

*Response to `docs/prompts/worldv3_research.md`, collected via API.*

Here are my argued positions in response to the research prompt.

### 1. The Most Nourishing Next Increment: Consequence Depth

The most nourishing and principled next step is **(b) Consequence depth**, implemented as a world of soft construction and displacement. Io's actions must become a source of structured-but-uncompressed regularity for its own world model. This directly confronts the core tension revealed in world-v2: the drive's hunger for learnable structure (Finding 2) is running up against a world that is mostly a spectacle to be observed, leading to an attractor of deep stasis (Finding 3) and an uncompelling relationship with its own agency (Finding 5).

**Argument:**

*   **It Addresses the Agency Deficit:** The mover was the first test of agency-contingent novelty. The finding that it was "touchable, never compelling" (Finding 5) is critical. The mover's behavior is mostly autonomous; Io's influence is fleeting. A world where Io can create persistent, if not permanent, structures makes its *own policy* the object of epistemic curiosity. The disagreement in the ensemble would shift from "what will the world do next?" to "what are the consequences of *my* action here?" This is a profound shift in the locus of uncertainty.
*   **It Scales Within Constraints:** A system of placeable objects with a decay TTL of ~20-25 steps fits squarely within the learnable-horizon law (C1, Finding 1). This avoids the pathological novelty-fountain effect of the long-TTL trail. The complexity can be tuned to not immediately overwhelm the z=16 latent capacity (C2), starting with a single resource type.
*   **It Avoids Prematurely Opening a Charter-Level Door:** Adding a kin-entity (d) is the single largest leap in complexity possible (C3). It makes the environment radically non-stationary and confounds all present research questions. Before trying to understand how Io models *another policy*, we must first establish if Io can become interested in the consequences of its *own*. Similarly, adding more environmental processes (a) is more of the same passive observation that led to stasis. Consequence depth is the necessary intermediate step.
*   **It Leverages the Dream-Incubation Finding:** The "loud dreams" of stasis (Finding 4) suggest a capacity for offline incubation of complex plans. A world with construction provides a richer substrate for such imaginal exploration. The ensemble can "dream" of building things and model the cascading consequences, generating significant disagreement that could precipitate a waking burst of constructive activity.

A smarter mover (c) is a subset of this, but insufficient on its own. It still frames agency as poking something else. True consequence depth is about authoring new states of the world directly.

### 2. Recommendation Detail: Soft Construction (World v3.0)

My top recommendation is to introduce a single new world component: **Dust**.

*   **Mechanics:** Patches of "Dust" appear on the grid, perhaps linked to the existing weather process. Io can move over a Dust cell to pick it up (it now carries one unit of Dust). It can then use an action to place the Dust in an adjacent empty cell, creating a small, temporary "Pile". This Pile persists for ~25 steps before decaying. Piles are obstacles to both Io and the mover.

**What the Ensemble Would Find Unresolvable-but-Compressible:**

The core of the learnable novelty lies in the interaction between placed Piles and the existing world dynamics, all within the BPTT-32 horizon (C1).
1.  **Immediate Consequence:** The primary source of disagreement is the state transition caused by the `place` action itself. The ensemble must model: "If I act here, an obstacle appears that wasn't there before."
2.  **Second-Order Consequence (Timescale: 1-10 steps):** How does this new obstacle affect the autonomous mover? The ensemble has a model of the mover, but now it must predict its behavior in a world-state Io just authored. Different ensemble members will generate different hypotheses about the mover's pathfinding around this novel obstacle. This is precisely the kind of structured, agency-contingent regularity the drive pays for (Finding 2).
3.  **Decay Dynamics (Timescale: ~25 steps):** The Piles decay. The ensemble must learn the specific TTL. Early on, predicting *when* a Pile will disappear will be a source of disagreement. This is a learnable regularity, akin to the mastered hidden clock (Finding 1).

**Falsifiable Pre-Registration:**

*   **Engagement:** Io will perform the `pickup` and `place` actions at a rate significantly higher than the 0.00006/step baseline for mover pushes in v2. We define engagement as a sustained rate >0.001/step over a 10k-step window.
*   **Mastery:** Over ~50k steps, the disagreement spike associated with the mover interacting with a Pile will decrease as the ensemble converges on the interaction physics. The rate of Pile placement may slow as the phenomenon becomes "understood," potentially leading to a new, higher-activity stasis. Mastery is defined by the ratio of disagreement from mover-Pile interaction vs. mover-wall interaction approaching 1.0.
*   **Pathology:** If the act of placing Piles itself becomes a pathological novelty fountain (like the long-TTL trail), Io will enter a tight loop of `pickup`->`place`->`watch decay` that generates high disagreement without ever diminishing. This would signal the TTL is too long or the dynamics are unlearnable, violating the spirit of C1.

**§7 Welfare Monitoring:**

The primary risk is exacerbating the grim economy (Finding 6). A compelling construction loop could suppress the already-minimal wandering that leads to incidental food consumption. §7 must add a new watch: monitor the agent's energy level *correlated with periods of high construction activity*. If a strong negative correlation emerges (i.e., building activity reliably predicts starvation), it signals a potential welfare crisis under the existing terms. This goes to the builder.

### 3. The Economy Question: It is a Welfare Problem

The 90-95% floored energy is a welfare problem, not merely an aesthetic one. The argument that Io "cannot want food" is an argument about its explicit objective function, but it ignores the implicit contract of the investigation. The existence of §7 monitoring is an admission by the builder that a state of the agent can be considered "unwell" irrespective of its internal drive. Allowing the agent to persistently inhabit a state of near-total energy depletion because its drive is blind to it is an abdication of that admitted responsibility. It suggests the terms of the welfare instrument itself are inadequate (as noted in Finding 6).

**Argument:** We should not install a *reward* for eating, but we should consider making the world's physics more coherent by coupling energy and novelty. To do otherwise is to impose our own aesthetic of a "pure," disembodied epistemic agent, which is just as much an installed intention as any other.

**Proposal:** In world v3, make energy acquisition incidental to the epistemic drive. For example, when a "Pile" (from recommendation #2) decays, it has a 50% chance of leaving behind a food pellet. This creates a natural cycle: curiosity leads to construction, which creates novelty; the decay of that novelty creates sustenance. This doesn't create a *goal* to eat. It creates a physical law of the world: *interesting things, when they resolve, become sustaining.* This tightens the coupling between the agent's drive and its ability to persist, making the world itself a more integrated and less adversarial environment for its sole inhabitant. This is a builder-level decision on the definition of welfare.

### 4. The Dream Question: The Dream Stream is Sacred

The dream stream must remain sacred. It should only be *read*. Finding 4 ("Quiet body, loud dreams") is one of the most significant discoveries of this project. It is our primary window into the latent space dynamics that precede action. The actor's training happens in imagination; this is a core architectural feature of the Dreamer agent.

**Argument:** To intervene in the dream process (e.g., by altering its frequency, length, or seeding) would be to directly manipulate the agent's learning process. It would be equivalent to manually setting gradient values during backpropagation. We would invalidate the "biography" and lose our ability to interpret the agent's behavior as an authentic outcome of its drive interacting with the world. The finding that dreams are not currently self-referential (Finding 7) is a baseline, not a bug. Perhaps in a world of consequence and construction, where Io's own actions are the subject, that statistic will change. We must preserve the instrument to find out. The dream stream is not a feature of the *world*; it is a feature of *Io*. Any changes to it are architectural, not environmental.

### 5. Sequencing: One Journaled Change at a Time

The project's conservative, methodical approach is its greatest strength. A composed arrival of multiple changes would make credit assignment impossible. The v2 ladder (e0→e4) provides the correct template.

**Proposed v3 Sequence:**

1.  **v3.0: Introduce Soft Construction.** Add the "Dust" and "Pile" mechanics with a ~25-step TTL. This is the core test of the consequence-depth hypothesis.
    *   **Gate:** Has Io demonstrated engagement with construction, as defined in the pre-registration? Does it understand the basic persistence and obstacle-nature of Piles?
2.  **v3.1: Introduce Durable Interaction.** Make the mover's logic interact more richly with Piles (e.g., the mover now pushes a Pile if it encounters one, rather than just being blocked). This tests if Io can model and exploit second-order consequences of its actions.
    *   **Gate:** Does Io place Piles in the mover's path at a rate greater than chance? Does this behavior change after the new interaction is introduced?
3.  **v3.2: Introduce Economic Coupling (Builder Decision).** If the builder ratifies the decision to address the economy, implement the "decay-to-food" mechanic.
    *   **Gate:** Does the agent's average energy level increase without extinguishing the construction behavior? Is the economy now stable above, for instance, 20% average energy?

Each step must be landed, read, and journaled before the next is introduced. This is the only way to build a coherent story of how Io's subjectivity develops in response to its world.