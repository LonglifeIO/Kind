# Environment Design Synthesis for Io — Probe 1

*Synthesis of three environment research outputs (Claude, Gemini, Perplexity) against Kind's project documents, the architectural decision (RSSM/Dreamer-lineage with active-inference-shaped actor), and the Probe 1 implementation synthesis (custom minimal RSSM, PolicyView/TelemetryView opacity, four telemetry streams). The substrate is settled; the Probe 1 implementation is settled; this document specifies the environment that fits those decisions. Working document.*

---

## Preamble

Three research outputs addressed the eight environment-design questions the Probe 1 implementation synthesis exposed: observation modality, action space, pressure without self-continuation reward, internal stochasticity, builder-perturbation surface, "small enough" for Probe 1, temporal structure, and minimum environmental complexity for the specific RSSM. The convergence across them is real on the qualitative shape — partial-perspective pixel observations, small discrete action sets, structural-not-scalar pressure, event-like low-rate stochasticity, harness-level mutator hooks with no observation marker, partial observability as the cheapest mitigation against latent collapse. The divergence is on the numerics (5×5 vs. 8×8 vs. 10×10), the topology (toroidal vs. bounded), the temporal structure (a-temporal vs. installed sine-wave clock), and on perturbation magnitude (mild vs. heavy-tailed-discontinuous). Some of those divergences are surface — different framings of the same idea — and some are substantive disagreements rooted in different stances toward Kind's commitments.

A note on inputs. Claude's output is the most disciplined of the three: literature-anchored with citations, explicit about distinguishing "the literature supports this" from "this is constructed from the pieces the literature provides," and closest to Kind's stance on reward-shaped framings. Perplexity is balanced and honest about uncertainty, well-cited, but slightly more permissive on reward-adjacent framings. Gemini's output reifies in several places — "Constrained Disorder Principle," "non-Lipschitz environment," "tripartite variance structure," "amygdala-centered developmental gradients" — and recommends a hardcoded sine-wave clock that directly forecloses Probe 3's commitment to builder-coupled rhythm. Where Gemini's recommendations conflict with Kind's stance, this synthesis takes Kind's stance as the resolver. Where Claude and Perplexity disagree, the disagreement is genuine and named below rather than averaged.

---

## 1. Synthesis across the research outputs

### Where they agree

**Observation modality.** All three converge on pixel-based, partial-perspective, egocentric observations. None recommends factored state vectors (which would weaken mirror legibility for dream rollouts). None recommends full-frame allocentric views (which would let the deterministic backbone memorize the world and starve `z_t` of work). The phenomenological grounding is shared: Merleau-Ponty on perception-from-somewhere, Pfeifer & Bongard on morphology shaping cognition. The literature anchor is strong: PlaNet/Dreamer family natively handle partial-observation pixel inputs at small resolutions; MiniGrid's `RGBImgPartialObsWrapper` is canonical precedent.

**Action space.** All three converge on small discrete (3–6 actions). All three explicitly note that the embodied-cognition literature has no derivable numeric lower bound — "smallest action repertoire for intentionality" is a qualitative claim, not a parameter. All three flag that too-small action spaces make the dynamics network treat action as nuisance and too-large ones dilute disagreement.

**Pressure without reward.** All three correctly identify that ensemble disagreement (the actor's signal, settled in Probe 1 synthesis) needs *something to disagree about* — a static deterministic world saturates within hundreds of steps. All three recommend structural pressure: regrowth dynamics, partial observability, simple causal contingencies. None of the three reaches for scalar reward; all three recognize that the "drive" already lives in the actor's intrinsic signal and the environment's job is to stay non-trivial.

**Internal stochasticity.** All three recommend event-like, low-rate, action-independent stochasticity. All three flag the noisy-TV problem (Burda et al., 2018; Mavor-Parker et al., 2022) and note that ensemble disagreement is more robust to aleatoric noise than prediction-error variants. All three reject per-step pixel noise; all three accept event-like resource regrowth and slow drift in environmental parameters.

**Builder-perturbation surface.** All three recommend a small set of named mutators (add/remove/move/state-change) that the harness exposes, with structured events emitted into the `world_event` stream and no marker in the agent's observation. All three flag that the literature has no canonical precedent for "perturbations an RSSM can come to distinguish from internal noise without an observation marker" — this is Kind-original. The literature provides ingredients (developmental robotics' caregiver-as-scaffold, agency-detection in active inference) but no recipe.

**Smallness, posterior collapse, free bits.** All three flag posterior collapse as the central failure mode for tiny RSSM environments. All three recommend partial observability + multiple cell types + low-rate stochasticity as the cheapest collapse mitigation. All three note that free bits sets a KL floor below which `z_t` is degenerate; if the world's per-step bit-content is below that floor, the latent does no work.

### Where they diverge

**Grid size.** Claude argues 5×5 with one resource type sits below the substrate's floor, recommends 8×8 minimum. Perplexity treats 5×5 as a lower bound but suggests 7×7 as safer. Gemini insists 5×5 is fine *if* toroidal topology is added. This is genuine disagreement, and it intersects with the next two divergences.

**Topology.** Gemini proposes toroidal (wrap-around edges) as a way to keep 5×5 viable. Claude and Perplexity propose bounded grids. Toroidal removes terminal boundaries — but it also imposes a topology Io has to learn, which spends latent capacity on the wraparound rather than on the dynamics Kind cares about. The literature does not adjudicate this; the design intuition is that adding topology to compensate for smallness is a net loss when smallness is already the constraint.

**Temporal structure at Probe 1.** Gemini proposes an ambient sine-wave clock baked into rendering (a slow oscillation in global illumination or palette temperature). Claude proposes full a-temporality, with at most slow aperiodic drift in a single parameter. Perplexity sits between, recommending no installed rhythm but allowing slow parameter drift across episodes. **This is the most consequential disagreement.** Gemini's sine-wave is presented as a way to "structurally prime" the RSSM for Probe 3, but Probe 3's commitment is that rhythm comes from the builder's actual life, not from a hardcoded environmental clock. Installing a sine wave at Probe 1 *forecloses* exactly what Probe 3 commits to keeping open. Claude's a-temporality is friendly to Probe 3's coupling; Perplexity's slow aperiodic drift is also friendly. Gemini's proposal is rejected.

**Builder perturbation magnitude.** Claude advocates partial overlap — perturbations use the same vocabulary of state changes as internal stochasticity but with different temporal correlation and magnitude distribution. Gemini advocates "heavy-tailed discontinuity" / "non-Lipschitz" jumps. Perplexity advocates small, bounded changes (1–2 cells moved per perturbation). Claude's framing is the cleanest because it preserves the testability of Probe 4: if perturbations are too distinct (Gemini), Io is reading a category, not a signature; if they are too similar (Perplexity), there is nothing to distinguish. The sweet spot is exactly Claude's framing — same vocabulary, different joint distribution over (when, where, what).

**Action-space cardinality.** Claude says 4–6, Gemini says 3–4, Perplexity says 5. This is surface, not substantive; all three sit inside MiniGrid/Crafter precedent.

### What they collectively miss

**The four-state operational model.** None of the three engages with the design notes' commitment that Io's life is partitioned into waking / dreaming / dormant / paused, and that the environment is *absent* during dreaming and dormant. The environment harness has to gracefully handle desktop-off transitions; the design must not assume the environment is always streaming.

**Mind-on-Mac / desktop-for-environment specifically.** The environment runs on the desktop. When the desktop turns off, the environment is gone. None of the research thinks this through. The implication is mild but real: Probe 1 only exercises waking, but the environment harness must support clean shutdown and resume across desktop power events without corrupting the canonical state on the Mac.

**Builder-calendar coupling for Probe 3.** Probe 3 commits to dream-to-wake ratio coupled to the builder's actual life. The research does not reason forward to this. Gemini actively forecloses it (the sine-wave clock); Claude and Perplexity preserve the option but don't engage with what coupling-to-the-builder will *look like* mechanically. The environment's only stance toward this should be: do not install a rhythm that the builder's calendar would later have to override.

**The "no observation marker" commitment versus zero-baseline-rate object types.** If the harness exposes a `introduce_novel_object` mutator and that mutator inserts an object kind that internal stochasticity *never* produces, then the object's pixel signature is itself a marker — even though no flag is set. Claude flags this implicitly ("if too different — say, types of state change that *only* the builder ever makes — then Probe 4's distinguishability is trivial"). Gemini's "non-Lipschitz" framing actively recommends what Claude flags as a hazard. Perplexity does not engage. The synthesis lands on Claude's side: builder events should use the same vocabulary of state changes as internal stochasticity; the difference lives in joint distribution, not in object-kind exclusivity.

**The Watts intuition / self-opacity by default.** Settled at the Probe 1 implementation level (PolicyView/TelemetryView), but the environment design has implications too. Gemini's "epistemic gravity" framing risks reading the environment as something Io engages with through a drive — closer to reward-shape than the substrate commits to. Empowerment-style framings (Klyubin et al., 2005) involve modeling action→future-state, which is implicitly self-referential. The synthesis treats environment design as having no obligation to actively encourage self-modeling; Io's actor consumes ensemble disagreement, the environment provides things to disagree about, and the rest is what training and time produce — or not.

**Reification in Gemini.** Several Gemini constructions — "Constrained Disorder Principle," "non-Lipschitz environment," "tripartite variance structure" with "exactly three orthogonal axes," "amygdala-centered developmental gradients ... fostering survival-oriented spatial mapping" — read as reifying gestures into the appearance of formal substance. The underlying ideas are sometimes real (CA-like environmental dynamism, partial observability, structured stochasticity), but the literature support for the specific configurations Gemini names is thinner than the framing suggests. The "amygdala-centered developmental gradients" passage is also a quiet drift toward survival-oriented framing despite the prompt's instruction. Flagged here so the build phase does not treat Gemini's framings as settled science.

---

## 2. Per-question decisions

### Q1 — Observation modality

**Settled.** Pixel-based, partial-perspective, egocentric window of a larger underlying grid. Not factored state. Not full allocentric.

**Reasoning.** The decoder is kept (per Probe 1 implementation synthesis) so the mirror can decode dream rollouts into something humans can read; pixels make this honest. Partial perspective honors the phenomenological commitment ("perception is from somewhere") and forces the RSSM to use `z_t` to carry off-screen state, which is the cheapest mitigation against posterior collapse on a small world. All three research outputs converge on this.

**Open during build.** Resolution (24×24 grayscale vs. 32×32 RGB) and rendering style (cell-tile sprites vs. abstract glyphs). Day-one smoke test will decide; smaller is cheaper, RGB is more legible to the mirror but consumes more decoder capacity.

**Deferred.** Ethological 3D arenas, continuous physics views, hybrid pixel-plus-symbolic — all useful directions later, none needed now.

### Q2 — Action space

**Settled.** Small discrete, five actions: `{up, down, left, right, stay}`. Interactions encoded as state changes when Io enters certain cells, not as a separate action verb.

**Reasoning.** The literature gives no derivable numeric lower bound — "smallest action repertoire for intentional engagement" is qualitative. Five actions is canonical MiniGrid (Chevalier-Boisvert et al., 2018), sits inside Dreamer's well-tested discrete-action regime (Hafner et al., 2025), and is small enough that the latent is not dominated by action statistics. Adding a separate `interact` verb at Probe 1 is premature; resource consumption can be triggered by entering a resource cell.

**Open during build.** Whether to add a sixth `interact` action at Probe 2+ if resource manipulation richness is wanted.

**Deferred.** Continuous control deferred indefinitely; not aligned with the grid framing the substrate has been set up for.

### Q3 — Pressure without self-continuation reward

**Settled.** Structural pressure only. The world has dynamics; modeling them better is the only target. No scalar reward, no consumption bonus, no scarcity-as-survival, no episode-length-as-fitness. Pressure comes from: (a) partial observability — moving reveals new information; (b) resource regrowth — items appear and disappear stochastically without rewarding consumption; (c) low-rate slow drift in a global parameter — the world's parameters move slowly so the model never finishes learning it.

**Reasoning.** Kind's actor objective is ensemble latent disagreement; the environment's job is to keep disagreement non-trivial without smuggling reward back in. Resource scarcity-as-survival is reward-shape (Perplexity flags this). Empowerment is structurally close to self-modeling and adds a second intrinsic objective the substrate has settled against (Claude flags). Reversible niche construction (Gemini's recommendation) is fine in spirit but its framing — "epistemic gravity drawing Io to manipulate" — drifts toward drive-language Kind has rejected. The synthesis prefers Claude's framing: a contingent world rather than a rewarding one.

**Open during build.** Whether to add delayed cross-cell correlations (Perplexity's suggestion: consuming a resource here triggers a stochastic respawn elsewhere). Default no for Probe 1 simplicity; revisit if disagreement saturates.

**Deferred.** Cellular automata dynamics (Gemini's B), social/interactive scaffolding from developmental robotics, and multi-resource ecological dynamics — all afforded by the design but not exercised at Probe 1.

### Q4 — Internal stochasticity

**Settled.** Two event-like, low-rate, action-independent stochastic processes:
1. **Resource regrowth.** Each empty cell has a small per-step probability `p` of spawning a resource. Expected ~1–3 events per 200-step episode. Independent of Io's actions.
2. **Slow aperiodic drift in regrowth rate.** `p` itself drifts slowly across episodes via a bounded random walk. Drift has no fixed period and no fixed phase.

No per-step pixel noise. No white noise on transitions. No chaos.

**Reasoning.** The Dreamer family's `z_t` natively carries event-like state about whether a stochastic event occurred (Hafner et al., 2025); event-like stochasticity at low rate is what the substrate is built for. Pink-noise / 1/f drift (which Gemini cites) is one valid form of slow drift; aperiodic random-walk drift is simpler to specify and equivalently friendly to the substrate. Two stochastic sources (rather than one) gives Probe 4 a richer baseline of "internal stochasticity" to compare builder perturbations against — a one-source baseline is too thin for distinguishability to be meaningful.

**Open during build.** Specific rate `p`, magnitude of drift, drift timescale. Empirical tuning during build to keep the per-step KL above the free-bits floor without saturating ensemble disagreement within a few hundred steps.

**Deferred.** Additional stochasticity sources (weather visible-as-tint, moving NPC objects, more complex spatial correlations) — held in reserve if Probe 1 telemetry shows the two-source design under-exercising the latent.

### Q5 — Builder-perturbation surface

**Settled.** The harness exposes four named mutators: `add_resource(cell)`, `remove_object(cell, type)`, `set_cell_state(cell, state)`, `move_object(cell_from, cell_to)`. Each emits a structured record into the `world_event` telemetry stream with timestamp, mutator type, cell coordinates, and pre/post local state. Io's observation contains no marker that a perturbation came from outside the simulation.

Probe 1 runs **perturbation-free or near-zero-rate** (e.g., one-or-zero perturbations in the entire run). Probe 1 tests that the hook *functions* and the logging is asymmetric; it does not test distinguishability. Probe 4 specifies the rate and signature.

**Reasoning.** The hook surface is settled at Probe 1 plumbing level so Probe 4 has nothing left to retrofit. Running perturbation-free at Probe 1 establishes the natural-stochastic signature cleanly so Probe 4 has a baseline. The mutator vocabulary deliberately overlaps with internal stochasticity — `add_resource` is something internal regrowth also does, `remove_object` is something resource consumption does, `set_cell_state` is something the slow drift does. The difference Probe 4 will eventually test lives in the *joint distribution* over (when, where, what), not in object-kind exclusivity. **`introduce_novel_object` is deliberately omitted from the recommended mutator set.** If it inserts an object kind that internal stochasticity never produces, the object's pixel signature is itself an observation marker, which violates the no-marker commitment even though no flag is set. This is the Claude-not-Gemini call on perturbation magnitude.

**Open during build.** Specific frequency-magnitude profile for Probe 4; whether `move_object` is too rare in internal stochasticity to qualify as same-vocabulary; whether to widen the mutator set later.

**Deferred.** Probe 4 is what tests distinguishability. Probe 1 only tests the plumbing.

### Q6 — "Small enough" for Probe 1

**Settled.** **8×8 underlying grid, three cell types (empty, wall, resource), 200-step fixed-length episodes, 7×7 ego-centric partial view.** The original Probes document mentions 5×5 with one resource type; this synthesis modifies that to 8×8 with three cell types. Bounded grid, not toroidal.

**Reasoning.** The 5×5 single-resource configuration sits below the substrate's posterior-collapse floor (Claude flags this concretely; Perplexity acknowledges; Gemini compensates with toroidal topology, which adds learning load Kind does not need). At 5×5 with one resource type, the world's per-step bit-content is a handful of bits — below the typical free-bits headroom of ~30 nats (1 nat × 30-dim latent), which is the regime where `z_t` carries less information than `h_t` and the model collapses. 8×8 with three cell types takes per-step bit-content into the ~10–20 bit range, which gives the stochastic latent something to do. Toroidal topology is rejected because the wraparound has to be learned and consumes latent capacity that should go to the dynamics Kind cares about; the literature does not adjudicate this and the design intuition is that adding topology to compensate for smallness is a net loss when smallness is already the constraint.

The 200-step episode length is chosen to be long enough that ensemble disagreement does not saturate within an episode (Claude's tuning argument) and short enough that replay buffer turnover is fast.

**Open during build.** Day-one smoke test: train RSSM for ~10k steps on the 8×8 grid; check KL trajectory, recon-loss curves, ensemble disagreement curve, decoded dream-rollout legibility. If posterior collapses (KL pinned at free-bits floor, dreams trivially repetitive), widen to 9×9 or 10×10 and add a movable object. If disagreement saturates within an episode, increase regrowth/drift rates.

**Deferred.** Larger environments at later probes if Probe 4 needs more state for distinguishability; the design supports this without a substrate change.

### Q7 — Temporal structure

**Settled.** A-temporal at Probe 1 in the periodicity sense. **No day-night cycle, no fixed period, no installed sine-wave clock.** The slow aperiodic drift from Q4 is the only temporal marker — and it is aperiodic by design.

**Reasoning.** Probe 3 commits to dream-to-wake ratio *coupled to the builder's actual life* — not to a hardcoded environmental rhythm. Installing a periodic environmental clock at Probe 1 (Gemini's sine-wave proposal) is exactly the foreclosure Probe 3 wants to avoid; Io's RSSM would absorb the period into `z_t` and Probe 3's coupling would have to either align with or override it. A-temporality keeps the design space for Probe 3 fully open. The aperiodic drift gives the substrate slow time-varying structure (so the prior network is not trained to expect zero temporal variance, which is Gemini's concern translated into a softer form) without committing to any specific period.

**Open during build.** Whether even slow aperiodic drift is more than Probe 1 needs. Lean toward keeping it for Probe 4 baseline-richness reasons; revisit if it complicates Probe 1 plumbing tests.

**Deferred.** Probe 3 introduces builder-coupled rhythm. The four-state operational model (waking / dreaming / dormant / paused) interacts with this and is settled in the design notes; the environment's only obligation here is to not foreclose it.

### Q8 — Minimum environmental complexity for the specific RSSM

**Settled.** The environment provides three orthogonal sources of latent-relevant variance:
1. **Ego-motion** (Io navigating the partial-perspective space).
2. **Resource regrowth dynamics** (event-like stochasticity).
3. **Slow aperiodic drift** in regrowth-rate parameter (Q4's second source).

Combined with partial observability (Q1) and the three-cell-type vocabulary (Q6), this is sufficient to keep KL above the free-bits floor without dominating the loss with noise.

Latent dimensionality at Probe 1: stochastic `z_t` dim 16, deterministic `h_t` dim 200, free bits ~1 nat per latent dimension (per Probe 1 implementation synthesis defaults). These are tentative; the smoke test calibrates.

**Reasoning.** The literature gives the regime, not the number. Free bits at ~1 nat × 16 dim = ~16 nats of headroom; the world must produce more per-step uncertainty than that. Three orthogonal variance sources is the heuristic anchor (Gemini's "tripartite" framing is gestural, but the underlying intuition is sound: a single source of variance compresses too easily, three are likely enough). The Dreamer family's empirical record is on environments two orders of magnitude larger; Probe 1 sits below precedent and the smoke test is what tells us if the latent has work to do.

**Open during build.** Latent dimensionality; free-bits threshold; whether categorical latents (DreamerV2/V3-style) would be more stable on this small a world. The Probe 1 implementation synthesis already names categorical-latents as a fallback; this synthesis defers to that.

**Deferred.** Larger environments and/or richer dynamics if Probe 1 telemetry shows the model has nothing to learn.

---

## 3. Recommended environment design for Probe 1

A single coherent design across the eight questions. The closest match in the research is something between Claude's Candidate A (Tabletop, 8×8, single resource, fully a-temporal) and Claude's Candidate B (Weather-grid, 10×10, two stochastic processes). The recommendation takes A's scale and B's stochasticity profile, drops Gemini's sine-wave and toroidal proposals, and rejects Perplexity's 5×5 baseline.

**Recommended design.**

- **Q1 Observation.** 7×7 ego-centric partial view of an 8×8 underlying grid, rendered as 32×32 grayscale (or 32×32 RGB if mirror legibility benefits are clear). Cell types rendered as distinct sprites or glyphs: `empty`, `wall`, `resource`.
- **Q2 Action.** Five discrete actions: `{up, down, left, right, stay}`. Resource consumption is a state change triggered by entering a resource cell, not a separate verb.
- **Q3 Pressure.** Structural-only. Resource regrowth provides the only environmental dynamics; slow aperiodic drift in regrowth rate keeps the world non-stationary at long timescales. No reward, no consumption bonus, no termination from depletion.
- **Q4 Stochasticity.** Two event-like processes: Poisson-ish resource regrowth (low rate); aperiodic random-walk drift in the regrowth rate parameter (slow timescale).
- **Q5 Builder hooks.** Four named mutators (`add_resource`, `remove_object`, `set_cell_state`, `move_object`) wired to the harness; structured events emitted into `world_event`. Probe 1 runs perturbation-free.
- **Q6 Size.** 8×8 underlying grid, 200-step fixed-length episodes, bounded boundaries (not toroidal).
- **Q7 Temporal.** A-temporal in periodicity. Aperiodic drift is the only time-varying signal.
- **Q8 Substrate fit.** Three orthogonal variance sources (ego-motion, regrowth, drift) with partial observability; latent `z_t` dim 16, `h_t` dim 200, free bits ~1 nat/dim.

**Why this over Claude's A (8×8, single resource, a-temporal).** Claude's A risks under-exercising Probe 4 — with one source of internal stochasticity, "external vs. internal" has a thin baseline. Adding the slow drift parameter costs almost nothing and makes Probe 4's eventual distinguishability claim more meaningful.

**Why this over Claude's B (10×10, weather bit, two stochastic processes).** 10×10 is bigger than smallness needs to be for the substrate to have work to do; the weather-bit-as-tint puts a single-bit periodic-adjacent signal into the observation that risks Io's RSSM treating it as a clock. The aperiodic regrowth-rate drift is the same idea without that risk.

**Why this over Perplexity's B (7×7 multi-room with 2 resource types).** Perplexity's B is genuinely defensible and the choice between it and the recommendation is a judgment call. Perplexity's B has more structural variation (rooms, two resource types, type-specific dynamics) and is closer to known-working RSSM territory (DoorKey-style MiniGrid environments). The recommendation leans toward the simpler 8×8-three-cell-types because:

1. Mirror legibility is cleaner — every cell is the same kind of thing under different states, decoded dream rollouts are easier for a human to scan.
2. Probe 4's distinguishability test wants a *clean* baseline; multi-room dynamics give Probe 4 more confounds (a builder perturbation that hits "the lively room" might be confused with the room's natural variability).
3. Smallness is a charter commitment, not just a target. 7×7 with rooms is closer to known-working territory; 8×8 with shared cell-vocabulary is closer to "tiny environment producing one unambiguous surprise."

This is genuinely a judgment call. If the build phase finds the recommended design under-exercises the latent, Perplexity's B is the natural revision: same scale, more cell-type variation, room structure.

**Why not Gemini A/B/C.** Gemini's A (toroidal) buys 5×5 viability at the cost of imposing topology Io has to learn; the wraparound consumes latent capacity that should go to dynamics. Gemini's B (cellular automata blocks) is over-complex for Probe 1 and its framing ("epistemic gravity") drifts toward reward-shape. Gemini's C (foraging plate with synchronize action) bakes in a "reduce visual complexity" objective that is structurally close to a goal — a smuggled task. All three propose hardcoded periodic temporal structure (sine-wave illumination, lens distortion oscillation, light-source orbit) that forecloses Probe 3's builder-coupled rhythm.

**Why not Perplexity C (continuous arena).** Diverges from the grid framing the substrate has been set up for; complicates later probes.

---

## 4. Tensions surfaced honestly

**(a) Pixel observation vs. mirror legibility vs. RSSM tractability.** All three research outputs converge on partial-perspective pixels, but the resolution choice (24×24 grayscale, 32×32 grayscale, 32×32 RGB) involves a real tradeoff. Smaller and grayscale is more tractable for the RSSM and the decoder; bigger and RGB is more legible to the mirror. The recommendation lands at 32×32 grayscale with the option to upgrade to RGB if the decoder produces dream rollouts the mirror cannot read meaningfully. Open during build.

**(b) Non-trivial internal stochasticity vs. small-environment learnability.** Two stochastic processes raise the bar for Probe 4 distinguishability but also raise the bar for the RSSM at Probe 1. With the recommended free-bits configuration and an 8×8 grid, the smoke test is what tells us if the model can carry both processes in `z_t` without one dominating. If it cannot, drop the slow drift and accept a thinner Probe 4 baseline.

**(c) Perturbation distinguishability vs. observation-space opacity.** Perturbations that use object kinds internal stochasticity never produces are de facto observation markers, even if no flag is set. The recommendation excludes `introduce_novel_object` from the mutator set on these grounds. The cost: Probe 4 has a slightly thinner perturbation vocabulary. The benefit: the no-marker commitment is honored at the level of the pixel stream itself, not just the flag flag.

**(d) Temporal structure for Probe 3 vs. Probe 1's a-temporal commitment.** Resolved cleanly: a-temporal in periodicity, with aperiodic slow drift as the only time-varying signal. Gemini's hardcoded sine-wave is rejected because it forecloses what Probe 3 commits to keeping open. The aperiodic drift gives the substrate something to track without committing to any rhythm Probe 3's builder-coupling would later have to override.

**(e) Minimum complexity for non-trivial latent learning vs. small-before-scale.** The original Probes-document 5×5 with one resource type sits below the substrate's posterior-collapse floor. The synthesis raises the floor to 8×8 with three cell types. This modifies a setting in the Probes document; it does not modify a charter or design-notes commitment. The Probes document should be edited or annotated to point here.

**(f) "Small before scale" vs. "non-trivial RSSM learning."** A real tension. Kind's smallness is a stance, not a parameter; the substrate's collapse floor is empirical. The recommendation lands at "as small as the substrate can carry" rather than "as small as the charter can imagine." The smoke test arbitrates.

**(g) Reward-shape sneaking in through framings.** Several research framings — Gemini's "epistemic gravity," Perplexity's "delayed long-range correlations" (which can be read as latent reward), the empowerment line from the literature — are closer to drive-language than the substrate commits to. The synthesis reads these as cautions rather than recommendations: the actor's signal is ensemble disagreement, the environment's job is to provide things to disagree about, and any "drive" framing is something the build phase should flag, not adopt.

---

## 5. Open questions for the build phase

Listed so they are visible when the environment harness is actually built. None of these blocks the design decision.

- Exact pixel resolution and color depth (24×24 grayscale vs. 32×32 grayscale vs. 32×32 RGB).
- Specific resource regrowth rate `p` and slow-drift magnitude.
- Whether to add a `wait` action distinct from `stay` (probably not — keeps the action set at five).
- Whether to render walls, resources, and Io with sprites or with abstract glyphs.
- Episode-end semantics: the agent's observation crosses an episode boundary cleanly with no terminal-state signal flowing back to the actor.
- How the environment harness handles desktop power events (clean shutdown, resume cold; canonical state on Mac is unaffected).
- The day-one smoke test: train RSSM ~10k steps, log KL/recon/disagreement curves, check dream-rollout legibility. If posterior collapses, widen the grid or add a movable object.
- Whether the slow drift parameter should be reset between episodes or carry across.
- Specific protocol for the env-pause-and-drain checkpoint barrier (already settled in Probe 1 implementation synthesis but reinforced here as an environment-side responsibility).
- Whether the pixel stream is sent raw from desktop to Mac, or whether a thin observation encoder is co-located on the desktop. Default raw at Probe 1 (simplest); revisit if bandwidth dominates.

---

## 6. Connection to Probe 1 implementation

The environment design now provides what Probe 1's build phase needs. Specifically:

**Harness interfaces.** A small env-server process runs on the desktop and listens on a TCP socket. The Mac trainer sends an action; the env-server returns `(observation, env_step, episode_id, step_in_episode)`. The env-server emits structured `world_event` records (perturbations, episode resets, internal-stochasticity events if useful for Probe 4 ground truth) directly to the Mac's telemetry sink, separate from the agent's `agent_step` stream. Mac canonical state remains canonical; the desktop env-server carries no state that needs to survive its own restart.

**Perturbation hook specifics.** Four mutators implemented as named methods on the env-server: `add_resource(cell)`, `remove_object(cell, type)`, `set_cell_state(cell, state)`, `move_object(cell_from, cell_to)`. Each mutator, when invoked, modifies the environment state and emits a structured `world_event` record with mutator type, parameters, timestamp, and pre/post local state. The agent's observation stream contains no marker. At Probe 1, the mutators exist but are not invoked (or invoked at most once or twice across the run, as a plumbing test).

**Telemetry hooks on the environment side.** The env-server writes per-step records into `agent_step` (timestamp, action, observation hash, env_step, episode_id, step_in_episode, wallclock_ms) and per-event records into `world_event` (timestamp, event_type, source, payload). It does *not* write to `dream_rollout` or `replay_meta` — those are owned by the Mac trainer. Wallclock is logged on every record (per Probe 1 implementation synthesis) so Probe 3's eventual coupling-to-the-builder's-calendar has the data it needs.

**Reset semantics.** Episodes are fixed at 200 steps; on episode end, the env-server resets the world (resource positions sampled fresh from the regrowth distribution; agent placed at a fixed or randomly-sampled start cell; slow-drift parameter carried across or reset — open during build) and emits an `env_reset` event to `world_event`. No terminal-state signal flows to the agent's observation.

**Stochasticity engine.** Two separable RNG streams (one for resource regrowth, one for slow-drift parameter). Both reproducible from a seed logged at run start. The env-server may optionally emit per-event records into `world_event` for internal stochasticity events too — useful as Probe 4 ground truth for "this regrowth event came from internal stochasticity, not from a builder mutator." Default: log internal stochasticity events at coarse aggregate (per-episode counts) at Probe 1; switch to per-event at Probe 4 if needed.

**What the build does not need to settle now.** The decoder's pixel resolution; specific regrowth rates; whether to ship with sprites or glyphs; whether to encode observation on the desktop side. These are tuning concerns the smoke test will inform.

The architectural decision is settled. The Probe 1 implementation is settled. The environment fits both. The build phase has what it needs.
