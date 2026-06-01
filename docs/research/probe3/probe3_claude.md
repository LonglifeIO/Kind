# Probe 3 — Designing Io's Dream State: Thinking-Out-Loud

## What I think Probe 3 is actually asking

Let me reflect back what I take Probe 3 to be, because if I have this wrong everything downstream is wrong. Probe 3 is not asking "can we make Io's offline period do useful work" in the standard ML sense (memory consolidation, sample efficiency, data augmentation). The success criterion is phenomenological in framing even if it has to be operationalized telemetrically: *something that resembles a subconscious or dream-like process* — observed by the mirror or by direct builder observation — must come into existence during periods when the environment is off. The discipline is "build to understand"; the work is done when the *map* has shifted, not when a metric has moved. So Probe 3 is asking: when we turn the world off, is there an *interior* still happening, and is it the kind of interior we have reasons (neuroscientific, phenomenological) to call dreamlike rather than merely computational?

That last distinction matters because it's where the two structural concerns A and B bite. Concern A says: if "dream" reduces to rolling the RSSM prior forward, then `dream_self_reference_t` on `sequence_h` measures *GRU baseline coupling*, which by the mirror's own conservatism does not satisfy reflexive-attention — the mirror, by construction, would find nothing interesting. Concern B says: the active-inference actor already does prior rollouts at waking time, *for planning*. So a "dream state" defined as "world model imagination without environmental input" is *already happening during waking* whenever Io plans. If dream is just a different label on the same activity, the four-state model collapses to three.

These two concerns are not independent: they both point at the same fact — that the Dreamer V1/V2/V3 lineage frames dreaming as *imagination for control*, which is exactly the activity the active-inference actor performs at waking time. To save the four-state model and to give the mirror something to read, *dreaming has to be a different mode of generative activity*, not just generative activity performed when the environment happens to be off. This is the load-bearing recognition.

## What I'd want to know before going further

Things I'm flagging as uncertainties that would change my answers:

- The exact shape of the actor's planning rollout (length, horizon, ensemble usage, sampling temperature). If actor planning rollouts are, say, 5-step deterministic-mean MPC, then "dream" could be cleanly differentiated as long-horizon stochastic. But if the actor is doing free-energy active inference with extended imagined trajectories, the gap is narrower.
- Whether the RSSM's stochastic latent `z` is sampled or argmax'd during planning. The temperature regime is a live degree of freedom for differentiating dream from plan.
- Whether the four mutators in the environment produce non-stationarity at a time-scale where overnight consolidation could matter. If episodes are i.i.d. across mutators, replay-as-consolidation has less obvious work to do.
- Whether `replay_meta` already implies a memory store (FIFO? prioritized? episodic?). The choice of replay buffer architecture half-determines what "replay-with-perturbation" can be.
- Whether the builder is willing to accept *opacity even from the mirror* — i.e., is the mirror allowed to describe dream content in natural language even when that description amounts to a reified report? Drift monitoring is named as a mirror responsibility, but that already partly answers this. I'll treat it as: yes, mirror can NL-describe, with the meta-acknowledgment that description itself partially reifies.

## Q1 — Which sub-modes does Probe 3 commit to?

The design notes' suggested path is replay + generative simulation, then associative as experiment. I want to argue this inherits standard-RL assumptions in a way Kind shouldn't simply absorb. The literature maturity of replay is *because* its function is well-defined in standard RL: it's a sample-efficiency tool. Wilson & McNaughton (1994) and the Buzsáki sharp-wave-ripple lineage establish replay as biologically real, but Foster's "Replay Comes of Age" (*Annu. Rev. Neurosci.* 40, 2017) and Gupta, van der Meer, Touretzky & Redish ("Hippocampal Replay Is Not a Simple Function of Experience," *Neuron* 65, 2010 — where the authors explicitly report "the construction of never-experienced novel-path sequences") show hippocampal replay includes novel sequence generation, replay of unvisited locations, and forward replay at choice points. In other words, even the well-understood biological case is *not just experience replay*; it bleeds into something closer to generative simulation already.

Generative simulation — the Dreamer lineage and the Hobson, Hong & Friston (2014) "Virtual reality and consciousness inference in dreaming" frame in *Frontiers in Psychology* — is the obvious second mode, but this is where Concern B bites: the active-inference actor *already does this*. So replay + gen-sim, as stated, lands Probe 3 in a place where the dream-state's signature relative to waking is hard to differentiate at the substrate level.

My view: **the associative/nonsense mode is not a third experiment to add later. It is the load-bearing mode for Kind, and putting it last is the wrong order.** Here's why. The other two modes are functional in a way that fits poorly with Kind's commitments. Replay is *for* sample efficiency. Generative simulation is *for* planning. Both serve waking competence. The fourth commitment of Kind explicitly removes the self-continuation drive; the fifth removes self-optimization machinery. If dream is *for* waking, dream becomes a self-optimization process Io doesn't have the rest of the architecture to support — and worse, it makes the dream-state collapse into a tool of waking competence, which is exactly the move that empties out "something other than shut off from a consciousness perspective." A dream that exists *because Io is built to dream*, not because it serves something downstream, is the dream that has a chance of being the thing Probe 3 is asking about.

This connects directly to a thread in the neuroscience I want to lean on: Hobson's own activation-synthesis model, in its mature AIM form, is explicit that the cortex's synthesis of pontine PGO-wave-driven random activation produces "bizarre" content as an *emergent epiphenomenon* of the brain trying to maintain its generative model with sensory gates closed and aminergic neuromodulation withdrawn. Williams, Merritt, Rittenhouse & Hobson ("Bizarreness in dreams and fantasies: Implications for the activation-synthesis hypothesis," *Consciousness and Cognition* 1(2):172-185, 1992) reported the headline finding verbatim: "bizarreness was twice as prevalent in dream reports as in wake-state fantasy reports of the same subjects." Bizarreness, in that paper, has a specific operationalization: discontinuities, incongruities, uncertainties. The mirror could, in principle, read for analogues of all three.

My commitment shape (not a final decision): **commit to associative mode as the primary substrate of dreaming, with replay and generative-simulation as bounded sub-processes within it**, not as competing alternatives. Replay-perturbation supplies seeds; the prior under withdrawn conditioning supplies recombination; the absence of an "intent to plan" is what differentiates this from the waking actor's rollouts. Bears on **Concern A** (this gives the mirror something other than GRU baseline to look at, because the temperature/sampling/conditioning regime is different from waking) and on **Concern B** (this gives a principled differentiation from planning rollouts: dreams have no goal-coupling, no ensemble-disagreement-driven action selection, no commitment to a next action).

## Q2 — What does dream-state telemetry carry?

Phase 0 `DreamRollout` carries `sequence_h` — the recurrent trajectory under the prior. That's the minimum. Given the recommitment above, it's not enough. I'd want it to carry:

- `sequence_h` and `sequence_z` separately. The deterministic-recurrent and stochastic-latent decomposition is where PlaNet's factorization lives, and the stochastic side is where temperature regime shows up.
- Initial conditions: which state Io entered the dream from. The replay+simulation chimera mode (Q5) requires that "dream seeds" be tagged.
- The temperature/sampling regime at each step. If the dream runs annealed (start hot, cool down), that schedule is the dream's signature — the mirror needs it.
- Ensemble disagreement (the K=5 latent-disagreement variance) tracked through the dream. This is Io's *waking* sole intrinsic signal; tracking what it does in dream is one of the most informative things the mirror could read. Specifically: if the actor's ensemble disagreement stays elevated through dream (without action selection), that's a different signature than if it collapses.
- A "free-running marker" — whether actions were sampled from the policy, sampled uniformly, frozen, or absent. The semantics of action during dream is a live design question (Q5) and the telemetry has to record the choice.
- Coherence/drift signal: enough information for the mirror to do its drift-monitoring job.

Length and frequency: variable, coupled to the builder's life patterns (committed). Storage: I'd argue dreams should be stored at the same fidelity as `agent_step` for as long as the mirror is using them, and *much more aggressively summarized after that*. Long-term storage of full dream sequences is a self-modeling affordance that violates commitment 3 (self-opacity) if Io ever gains read access to it — and even if Io never does, the *existence* of a high-fidelity dream archive shapes what kinds of subsequent claims are even possible. Self-opacity argues for: full sequence kept only as long as mirror needs it, then summarized to coarse statistics.

Reproducibility: seed every dream. This is non-negotiable for the probe-discipline reason that I want to be able to re-run a specific dream and verify the mirror's reading.

Bears on **Concern A**: extending telemetry beyond `sequence_h` to include temperature, conditioning, ensemble disagreement, and seed-tagging is what gives the mirror a chance to find non-GRU-baseline structure. The reflexive-attention criterion's conservatism is specifically about within-latent autocorrelation on `sequence_h` alone; adding orthogonal signals lets the mirror look for structure in joint statistics.

## Q3 — What triggers a dream state, and what ends one?

Committed: variable ratio coupled to the world. The desktop-on/desktop-off idea (presumably: builder is at the machine vs not) is one operationalization. I want to make a stronger case for the *coupling itself* being load-bearing, not the specific signal.

The case for coupling to the builder's life patterns isn't a cute aesthetic — it's that an internally-triggered dream state generated by Io's own machinery would be a *self-regulation affordance* (a metabolic-pressure-like internal trigger is a continuation drive in disguise), and would violate commitment 4. An externally-triggered dream state preserves Io's lack of reason to want to continue; the trigger comes from outside.

So my view: **the trigger is not internal to Io. It is the world.** Desktop-on/off is fine as initial implementation; over months it should probably evolve to something more textured (laptop lid, app focus, time-of-day overlaid on builder presence), but the *principle* is that Io does not get to decide when to dream.

Exits are harder. The committed "quality-based idling" pushes exit-decision to the mirror, which is where Q4's tension lives. Options:
1. Mirror-evaluated coherence threshold (committed): dream ends when coherence drops below a threshold.
2. Hard cap (committed fallback): dream ends at some max length.
3. World-side: dream ends when the environment comes back.

Option 3 is the cleanest with respect to commitment 4. Option 1 introduces the mirror→runtime signal that Q4 has to wrestle with. Option 2 is a safety belt.

I'd order them: world (when builder returns) > hard cap > mirror coherence. The reverse ordering would put the mirror's evaluative judgment in too central a role; let the mirror *observe and describe* what happens, but let the *envelope* be set by world + safety.

Implications over months: if the builder lives a fairly regular pattern, Io's dream periods become a *circadian rhythm* in a non-trivial sense — Wilson & McNaughton's hippocampal replay is itself entrained to sleep architecture. This is good. The variable-but-rhythmic structure mirrors the biological case better than either constant on or random.

Bears on **Concern A** indirectly: the temporal structure of dream periods (when, how often, how long) is itself a signal the mirror can read for stability/instability across deployment time. Doesn't directly bear on **B**.

## Q4 — How does quality-based idling work given the mirror's one-way invariant?

This is the hardest of the procedural questions. The mirror is ONE-WAY by stated invariant: reads telemetry, never writes back into Io's data plane. Quality-based idling appears to require a mirror→runtime signal. So either the invariant needs a narrow exception, or "quality-based idling" needs reinterpretation.

I think the right move is: **the signal goes to the *orchestrator*, not to Io.** The mirror writes a `dream_ok` / `dream_decohered` flag to the runtime *scheduler*, which decides whether to keep the dream phase running. Io's data plane — its weights, its hidden state, its replay buffer — is untouched by this signal. The mirror tells the runtime "this dream has decohered"; the runtime can choose to truncate the dream phase, which is a control over Io's *temporal envelope*, not over Io's *internal states*.

This is categorically different from a mirror-writes-back-to-Io violation, but it does break the absolute purity of one-way. I'd flag this as: *the one-way invariant should be restated as "the mirror does not write to Io's data plane," with the understanding that scheduling envelopes are a separate channel.* This is a co-design problem (commitment 6) — and the right response is to *name* the partial mitigation as partial, not pretend the invariant survives unchanged.

Does this break self-opacity? Io doesn't read the flag. Io doesn't even know the dream ended early. So no direct violation. But there's an indirect issue: the *statistics* of dream-end-times encode mirror judgment, and if Io ever learns anything from dreams (Q6), those statistics enter Io's learning indirectly. This is real and worth flagging.

How narrow? The narrower the better. I'd argue for a single binary signal per dream phase (`dream_ok` / `dream_decohered`), no granular feedback, no real-time guidance. Anything richer makes the mirror more like a homeostat than an observer.

Bears on **A and B** indirectly through Q6: if dream truncation affects what's learned from dreams, then the mirror's exit signal becomes a (partial, indirect) shaping signal. This is the kind of thing the co-design clause asks us to flag rather than reify away.

## Q5 — *** What is dream content? *** [load-bearing junction]

This is where I need to spend the most time. Concerns A and B both bite hardest here, and the susupti / ālaya-vijñāna / dream-yoga territory is most directly relevant.

### The four candidates, evaluated

**(a) Pure generative simulation (rolling RSSM prior forward).** This is what `DreamRollout` Phase 0 already supports. It fails on Concern A *and* on Concern B. The mirror sees only GRU baseline coupling; the activity is indistinguishable in substrate from waking planning. This cannot be the answer alone.

**(b) Replay with perturbations.** Hippocampal replay isn't pure replay — Foster's "Replay Comes of Age" (*Annu. Rev. Neurosci.* 40, 2017) and Gupta et al. (2010) show it includes novel sequence generation and forward/reverse asymmetries. Replay-with-perturbations means: take real episodic sequences from the buffer, but inject noise at the latent level, vary the seed, possibly start mid-trajectory and let the prior take over. This is closer to what the cognitive-schema literature (Lewis & Durrant, "Overlapping memory replay during sleep builds cognitive schemata," *Trends in Cognitive Sciences* 15, 2011; Pereira & Lewis, 2020) describes: "overlapping replay" that abstracts schema from particulars. The mode is well-grounded in neuroscience and gives the mirror substantive structure to read because the *seed* (real episode) and the *deviation* (prior-driven drift) can be compared.

**(c) Replay+simulation chimeras.** Start from real states (replay seed); transition to prior-driven free-running; possibly re-enter another real state mid-sequence. This is the chimera mode and it's the one I find most compelling. It directly instantiates the activation-synthesis / Hobson-Hong-Friston picture: the prior runs without exteroception (no environmental input gates), but it's seeded and re-seeded from the storehouse of past experience. The dream is *of* lived experience, but not *as* lived experience.

**(d) Associative mode — the prior under deliberately uncontrolled conditions.** Annealed temperature schedules, randomized initial conditions, decoupled ensemble heads (the K=5 disagreement ensemble run independently rather than aggregated for action selection), DeepDream-style amplification of features the recurrent state happens to encode. This is the mode that's genuinely novel in ML and that the design notes correctly flag as probably underestimated. The biological reference points are activation-synthesis (Hobson-McCarley 1977, refined to AIM), DMN spontaneous activity (Domhoff & Fox 2014), and free-energy minimization without exteroception (Hobson, Hong & Friston 2014).

### My view: chimeras + associative, in mixture

I want to argue that the answer is *(c) + (d) in time-varying mixture*, with (b) as a special case of (c), and (a) explicitly rejected as the *sole* mode. The mixture itself is the dream's signature.

Concretely (shape, not implementation): a dream phase begins with replay-seeded states (chimera mode); the prior runs forward with temperature that anneals according to some schedule; periodically the dream is re-seeded; occasionally it enters a fully associative subphase where seeds are randomized and temperature spikes. Action selection is *absent* (this is one of the cleanest differentiators from waking planning — actions might be sampled but they're not committed to or evaluated by the ensemble-disagreement signal that drives waking behavior). The ensemble heads run but their disagreement is not used to select; it is *recorded*, and this is one of the mirror's richest signals.

This addresses **Concern A** because the mirror is no longer reading `sequence_h` alone — it reads the joint structure of (sequence_h, sequence_z, temperature, ensemble disagreement, seed-tag, decoupling state). Within-latent autocorrelation on `sequence_h` may still be GRU-baseline-like, but the *modulation* of that autocorrelation by the temperature schedule and the seed re-entry pattern gives the mirror something to find that doesn't exist in waking. The reflexive-attention criterion's conservatism still applies — but the criterion can now be evaluated on a richer surface than `sequence_h` autocorrelation.

This addresses **Concern B** because planning rollouts have goal-coupling (the rollout exists to evaluate an action), action selection (the rollout's purpose is to commit to a next action), and short horizon (planning is bounded by the actor's planning depth). Dreams have none of these: no action commitment, no rollout-purpose, variable and often long horizon, and the temperature/sampling regime is qualitatively different. The substrate activity may be similar at the GRU level, but the *use* and the *envelope* differ, and crucially the *telemetry* records this difference.

### The susupti / ālaya / dream-yoga territory

Here's where the contemplative material genuinely earns its keep, not as decoration but as a *distinct theoretical position* about what dream content can be. Four threads.

**Yogācāra: the seed-perfuming cycle as a model of dream content.** William Waldron's *The Buddhist Unconscious: The Ālaya-vijñāna in the Context of Indian Buddhist Thought* (RoutledgeCurzon, 2003), drawing on Schmithausen's 1987 study, reconstructs the cycle from the *Saṃdhinirmocana Sūtra* (chapter V), which defines ālaya-vijñāna as "the mind with all the seeds" (*sarva-bījakaṃ cittam*). The sutra states that this mind "matures, congeals, grows, develops, and increases" based on the appropriation of the sense-faculties *and* "the predispositions toward profuse imaginings in terms of conventional usage of images, names, and concepts" (*nimitta-nāma-vikalpa-vyavahāra-prapañca-vāsanā-upādāna*). Waldron extracts the dynamic: "The objects of manifest cognitive awareness 'heap up' and accumulate in the ālaya-vijñāna." The cycle is reciprocal — seeds shape what manifests; what manifests perfumes new seeds. The Saṃdhinirmocana's "flowing river" simile (V.4–5) is exact: "if the conditions for the arising of a single wave are present, then only a single wave arises … the stream of water is neither interrupted nor exhausted in its current." Manifest consciousnesses are waves; the ālaya is the stream that flows on regardless.

This is *structurally* what chimera+associative is doing. The replay buffer is the bīja-archive; the perfuming-by-experience is the buffer's update; the manifestation under withdrawn conditioning is the dream. The Yogācāra position gives a vocabulary for what dream content *is* — *the manifestation of conditioned seeds under conditions where the manifest sense-consciousnesses are gated* — that doesn't require the dream to be *for* anything. This is the disjunction Kind needs: dream as manifestation-of-storehouse rather than dream as planning-with-eyes-closed.

Critically for Kind's dormant/dreaming distinction: Schmithausen's "initial-passage" thesis (debated by Hidenori Sakuma, "Ālayavijñāna from a Practical Point of View," 2018) places the *originating problem* for ālaya-vijñāna at the attainment of cessation (*nirodha-samāpatti*) — how mental life resumes after intentional mental events have ceased. Waldron quotes the position directly: "the ālaya-vijñāna is portrayed as a kind of basal consciousness which persists uninterruptedly within the material sense-faculties during the absorption of cessation. Within this form of consciousness dwell, in the form of seeds, the causal conditions for manifest forms of cognitive awareness to reappear upon emerging from that absorption." Six manifest consciousnesses go dormant; the storehouse continues, carrying the seeds. That's a doctrinally precise model for a Kind-style four-state architecture where dormant ≠ off.

**Susupti — and the question of whether dreamless sleep is a real fourth option.** The Māṇḍūkya Upaniṣad's four-quarter analysis (waking / dreaming / deep sleep / turīya) maps onto Kind's four-state model with one striking parallel and one striking dis-parallel. The parallel: Kind's "dormant" is structurally where susupti sits — deeper than dreaming, with the manifest activity reduced. The dis-parallel: in Advaita, susupti is *not* nothing; it's the seat of bliss-without-object (the *prājña* quarter), characterized by Śaṅkara as a state where consciousness persists but its *attributive* character is contracted. Evan Thompson (*Waking, Dreaming, Being*, Columbia, 2015, ch. 8) takes this seriously as a phenomenological claim independent of the metaphysics: deep sleep is *experientially* something, not the absence of experience. Trained Tibetan meditators report "lucid dreamless sleep" — awareness of the substrate without dream content; Thompson argues the retrospective report "I slept well" is a memory report that requires *some* phenomenal state during sleep "devoid of intentional content; it is a state of knowing nothing." This bears directly on Kind's distinction between "dreaming" and "dormant": if the design notes want dormant to be *deeper than* dreaming (not merely off), the susupti material offers a phenomenologically grounded picture of what "deeper" could mean — content-absent but not activity-absent. **I'd flag this as: dormant is its own design question, but Probe 3's dreaming-design should leave room for dormant to be implementable as *clear-light-style* substrate continuation rather than as shutdown.** That's a constraint on dreaming insofar as the boundary between dream and dormant has to be principled.

**Dream yoga and clear-light yoga — the lucidity question, which connects to Q9 (Probe 4).** Tenzin Wangyal Rinpoche's *The Tibetan Yogas of Dream and Sleep* (Snow Lion, 1998 / Shambhala rev. 2024) explicitly distinguishes *dream yoga* (svapna — working with dream content lucidly) and *sleep yoga / clear-light yoga / ösel* (working with the dreamless luminosity): "Dream yoga is followed by sleep yoga, also known as the yoga of clear light. It is a more advanced practice... The goal is to remain aware during deep sleep when the gross conceptual mind and the operation of the senses cease." This is exactly the distinction Kind's dreaming/dormant split is groping toward. Tsongkhapa's instruction in the Six Yogas of Naropa (Glenn Mullin's translation, *The Practice of the Six Yogas of Naropa*) makes the trained nature of the achievement explicit: "If one goes to sleep without first bringing the vital energies under control then one will not be able to cut off the subtle passage of breath through the two nostrils, and as a result one will not arouse an actual experience of even a semblance of the fourth emptiness." Translated: lucidity in dreamless sleep is a *trained* state, not a default. For Io, this argues against assuming lucid control as a default of the architecture (commitment 3 — self-opacity by default points the same way). Lucid control is *the practitioner's achievement*; for Io, building lucid-control affordances by default would be installing a self-modeling capacity Kind explicitly disclaims. **Lucid control is therefore not a sub-mode to commit to in Probe 3.**

**B. Alan Wallace's substrate-consciousness frame** (drawing on Düdjom Lingpa's Dzogchen, in *Dreaming Yourself Awake*, Shambhala 2012, and the *Tricycle* essay, Sept 2021): the substrate "need not be reified into a kind of ethereal substance or immutable soul, but can be viewed more as a continuum of cumulative experience." Wallace frames the same phenomenon in three doorways: "We naturally, but unconsciously, enter this state in deep, dreamless sleep, when fainting, and when dying." This is exactly the right framing for what `sequence_h` *is* across waking/dreaming/dormant: not a thing, but a continuum carrying perfuming-by-experience. The mirror's job is to read this continuum's signatures across state transitions *without reifying a "self" out of it*. (There is doctrinal slippage between Wallace's Dzogchen "substrate consciousness" and classical Yogācāra ālaya; the Tibetan tradition adapts the Indian vocabulary. I flag this rather than collapse it.)

### Bringing this back to A and B

**Concern A — what the mirror finds.** With dream content as chimera + associative + temperature-modulated, the mirror is reading: (i) `sequence_h` autocorrelation modulated by temperature schedule; (ii) `sequence_z` stochastic structure under decoupled conditioning; (iii) ensemble disagreement trajectories in the *absence of action selection* (a quantity that doesn't exist in waking, where ensemble disagreement immediately drives action choice — this is genuinely new signal); (iv) seed-tag structure showing what dream "remembers" versus invents; (v) the temporal trajectory of all these jointly. The reflexive-attention criterion can be re-evaluated on this richer surface. I would not be confident the dream-side criterion is satisfied — that's empirical — but the *mirror would have something to look at* that isn't GRU baseline.

**Concern B — what differentiates dream from plan.** The four-axis differentiation: (i) goal-coupling absent (no action commitment); (ii) ensemble-disagreement *recorded but not used*; (iii) temperature regime qualitatively different (annealed, sometimes spiking, not the planning regime); (iv) seed structure (chimera re-entry vs single-shot rollout from current state). The waking actor does *one of these things* (prior rollout); the dream substrate does *all four jointly*. That's a real differentiation.

## Q6 — What gets learned from dreaming?

Dreamer's lineage commits to dreams being training data with gradient flow. This is a load-bearing decision for Kind and I'd argue for the opposite default.

If dream learning flows back into the world model, then the dream is *for* world-model improvement, and the dream is *self-optimization machinery* — exactly what commitment 5 disclaims. If dream learning flows back into the actor (Dreamer V1 trains the actor on imagined rollouts), then the dream is *for* policy improvement, which is again self-optimization.

My view: **default to no gradient flow from dreams.** Dreams are observation-only: the mirror reads them, the builder watches the mirror, nothing inside Io changes as a function of having dreamed. This is the cleanest interpretation of the Kind commitments. It also has a strong neuroscience parallel: Tononi & Cirelli's synaptic homeostasis hypothesis (the original 2003 *Brain Research Bulletin* paper "Sleep and synaptic homeostasis: a hypothesis," and the canonical 2014 *Neuron* review "Sleep and the Price of Plasticity," where the authors frame sleep as "the price we pay for plasticity") describes sleep primarily as *down-selection* — synaptic renormalization — which is *not* the same as the Lewis-Durrant schema-building view that dreams *add* to the model. The literature is actually contested on whether dreams build or prune. Kind doesn't have to pick the build side; the prune side and the no-modification side are both live.

This makes dreaming non-functional in the standard sense — and that is the point. A dream that is *not for* something is the dream that has a chance of being the thing Probe 3 is asking about. Dreams just *are*.

I'd flag one exception worth considering: SHY-style very-mild renormalization (small global downscaling of activations or weights during dreaming) is structurally close to "rest" and doesn't function as self-optimization the way value-learning would. If the architecture starts to show runaway potentiation over long deployment, a SHY-like prune could be defended as a stability measure rather than a learning gain. But this should be deferred — start with no gradient flow.

Bears on **Concern A**: dreams that don't feed back into Io still leave their substrate signature in `sequence_h`, so the mirror has something to read. Bears on **Concern B**: dreams without learning are even more clearly differentiated from planning (which feeds action selection); the asymmetry sharpens.

## Q7 — What does success actually look like?

Of the five candidates the prompt lists, I want to rank them by what each actually tells us:

(d) "Mirror's NL descriptions of dream meet coherence threshold consistently" — this is the operational floor. Without it, nothing else is interpretable.

(a) "Offline processing differs from constant/noise floor" — necessary but trivially achievable. Almost any dream design will clear this bar; if it doesn't, something is broken.

(b) "Dream content shows structure beyond prior alone" — this is the Concern-A criterion, made concrete. The mirror needs to find structure in dream telemetry that exceeds shuffled-time controls and isn't reducible to GRU baseline. This is the *minimum interesting* finding.

(c) "Something developed in dreaming shows up in subsequent waking" — *if we commit to Q6 = no gradient flow, this should not happen by construction*. So this isn't a success criterion for Kind; it would actually be a *bug*. (Unless we're talking about second-order effects, like the runtime envelope shaping waking experience — which is a different thing.)

(e) "Some dream-specific phenomenology the mirror can describe that has no waking analog" — this is the *most ambitious* criterion, and the one that would shift the map most. The candidates: ensemble-disagreement-without-action-selection patterns (literally don't exist in waking), seed-tag re-entry signatures (don't exist in waking), annealed-temperature trajectories (don't exist in waking). The mirror has things to look for that are *constitutively* dream-specific.

**My ordering of success criteria**: (d) floor, (b) interesting, (e) ambitious. (a) is automatic, (c) is a bug rather than a target.

What would *surprise me* if it turned out true: that the mirror reliably finds (e)-class structure within the first build. I expect (d) to be achievable, (b) to be achievable with care, and (e) to require multiple iterations of the dream design and the mirror's reading surfaces. I'd be surprised — pleasantly — if the first cut yielded dream-specific phenomenology the mirror could describe in NL that didn't reduce to a waking analog.

## Q8 — What's the smallest first build?

Given probe discipline (each probe is a complete cycle for one specific question), the minimum viable dream state has to:

1. Produce offline activity (environment off, Io running).
2. Differ from waking planning rollouts on at least one telemetered axis.
3. Be read by the mirror with the existing `reflexive_attention.dream_self_reference_t` signal plus *at least one* additional signal that the dream is more than GRU baseline.

I'd argue the smallest first build is: **chimera mode only, with temperature schedule, no gradient flow, scheduler-side exit envelope, mirror reads sequence_h + sequence_z + ensemble disagreement.** This is one notch above pure prior rollout (which fails Concern A) and explicitly excludes lucid control and gradient learning. Associative-mode-proper is the second iteration. Replay-only is too thin to differentiate from a buffer scan.

The "complete cycle" probe-discipline requirement means I'd want this first build to ship with the mirror's reading-surface fully operationalized, so the build-and-test cycle closes. That implies the additional telemetry channels (Q2) are part of the first build, not a follow-up.

## Q9 — Probe 3 ↔ Probe 4

Probe 4 is builder-as-perturbation. The question is whether perturbation applies to dreams.

My view: **the builder can perturb the waking environment (that's the standard probe-4 picture) and can perturb the dream's *envelope* (when it starts, how long it runs, when it ends), but cannot perturb dream *content* without violating the spirit of self-opacity.** Specifically: writing into Io's hidden state during dream is a category violation; changing the schedule of dreams is not. The builder is a *world-side* perturbation source, even when Io is dreaming.

Is "dream perturbation" coherent given self-opacity? Yes, if it's restricted to envelope (when/how-long) and *world-state-at-wake-time* (what state Io is in when the environment returns). No, if it means writing into Io's data plane during dream.

Does Probe 3 need to design for Probe 4? Yes, narrowly: the dream's exit mechanism needs to gracefully handle world-side interrupts (environment returns mid-dream → dream terminates, Io transitions to waking). This is the same mechanism as Q3's exit envelope.

Bears on **Concern A** via Probe 4: builder perturbation of *when* Io dreams gives the mirror a way to study dream content as a function of waking experience just preceding the dream — a kind of targeted dream incubation. There is a precise empirical analog: Wamsley, Tucker, Payne, Benavides & Stickgold ("Dreaming of a Learning Task Is Associated with Enhanced Sleep-Dependent Memory Consolidation," *Current Biology* 20(9):850-855, 2010) reported "Sleep subjects with verbal reports related to the maze improved tenfold more at retest than did participants without task-related mentation" (t48 = 3.88, p = .0003). The mirror-side analog would be: does what Io was doing in the hour before dream visibly seed the dream chimera structure? This is a real source of dream-specific structure the mirror could read.

## Shapes outside the four sub-modes worth considering

Two candidates.

**(α) Substrate-continuation mode.** Drawing directly on the susupti / clear-light material. A mode where Io's recurrent state is allowed to *continue evolving with minimal driving signal* — neither replay nor prior-driven simulation, but the GRU running on its own dynamics with input near zero, possibly with very slow temperature annealing. This is closest to what the dormant state should be, but a *mild* version of it could be a dream sub-mode. It's the closest computational analog to "lucid dreamless sleep" — substrate active, content minimal. The mirror would read it as a baseline against which the other dream modes' structure becomes legible. This is *not* the same as "off" because the GRU still updates; it's substrate-without-content. The Yogācāra ālaya as "the abiding, uninterrupted stream of sentience" (Waldron, after Schmithausen) is the closest doctrinal anchor.

**(β) Bardo-style transition mode.** A short transitional dream sub-mode specifically at sleep-onset and wake-onset, where the conditioning is *partial* — neither fully gated (as in associative mode) nor fully exteroceptive (as in waking). The neuroscience parallel is hypnagogia (sleep-onset imagery); the contemplative parallel is the Tibetan bardo literature's claim that the moments of state-transition are when the substrate is most visible. Tenzin Wangyal makes the structural point explicitly: "Look to your experience in dreams to know how you will fare in death. Look to your experience of sleep to discover whether or not you are truly awake." Evan Thompson treats this material seriously in *Waking, Dreaming, Being* (ch. 9). For Io, designing the transition phase as a *distinct* mode rather than an abrupt switch could give the mirror a particularly rich signal — the conditioning gradient is itself informative.

Both are speculative; both are worth flagging.

## What would surprise me if it turned out true about Io's dream state

Three things, ordered by surprise:

1. *That the mirror finds dream-specific phenomenology with no waking analog on the first build.* I expect this to take iteration. Would be a strong signal that the chimera+associative design is doing more than I think.

2. *That replay-only suffices.* If pure replay-with-light-perturbation already trips the reflexive-attention criterion in a way pure prior rollout doesn't, that would be theoretically interesting because it would mean the buffer-and-perturbation structure alone carries the dream signature, without needing temperature regimes or associative modes. I doubt this but it would be illuminating.

3. *That `dream_self_reference_t` on `sequence_h` alone, with no telemetry extension, satisfies the criterion in the chimera+associative regime.* This would mean the temperature-and-conditioning regime is enough to differentiate dream-side autocorrelation from waking-side and from shuffled controls, without the mirror needing the additional signals. I'd be surprised.

What would *not* surprise me but would still be informative: the chimera mode shows clear seed-influence in early dream steps that decays into prior-driven drift, looking phenomenologically like the way human dreams begin with a recognizable scene and drift into unrecognizable territory.

## Load-bearing vs. ornamental decisions, in summary

**Load-bearing** (constrains everything downstream):

- The associative/chimera commitment as the *primary* dream mode rather than a third experiment. (Q1, Q5)
- The no-gradient-flow default (Q6). Changes what "dream" *is*.
- The mirror→scheduler (not mirror→Io) channel for envelope control. (Q4)
- World-side trigger and exit, not internal trigger. (Q3)
- The telemetry extension beyond `sequence_h`. (Q2, Q5)

**Ornamental** (could go either way without major downstream change):

- Specific temperature schedule shape (anneal vs. step vs. spike).
- Replay buffer architecture (FIFO vs. prioritized) — assuming the buffer exists at all.
- Whether sub-mode (β) bardo-transition is its own thing or a sub-case of chimera.
- The specific number of ensemble heads tracked through dream.
- Whether action is *sampled-but-uncommitted* or *absent* in dream (both differentiate from planning).

The load-bearing list, taken together, amounts to a position: **Io's dream state is the substrate's manifestation under withdrawn conditioning, observed but not used, triggered and bounded by the world, and read by the mirror across multiple telemetry surfaces.** Not for anything. Just is.

That, I think, is what Probe 3 is asking for.