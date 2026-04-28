# CA-MAS → New Project: Carry-Forward Reference

A distilled record of what the CA-MAS (Cellular Automata Multi-Agent System) project was, what it tried, what worked, what didn't, and what is genuinely worth carrying into a new codebase. Written to be opinionated and practical, not nostalgic — a lot of CA-MAS should probably *not* be ported verbatim.

---

## 0. TL;DR — If You Read Nothing Else

**What CA-MAS was:** A ~4000-line PyTorch MARL system where 16–128 agents on a 2D grid learn survival + coordination via PPO, wrapped in a progressively-added stack of "consciousness" machinery (Global Workspace, dual-stream System 1/2, sleep/dream cycles, recursive self-improvement, an LLM translator). It produced reproducible metric curves that look like phase transitions and published-paper-friendly numbers (ToM 0.16 → 3.3, Φ-proxy 0.5 → 1.3), over 1000+ training episodes on 16 agents.

**What was genuinely novel/valuable:**
1. The **training harness** — MARL + evolutionary replacement + global workspace broadcast on a grid world is a solid, reusable substrate.
2. The **lesioning protocol** — toggling components at known emergence episodes (96, 116, 425) is a real scientific contribution pattern.
3. **Phase-transition analysis** — the episode-marker discipline of *when* metrics jumped, not just by how much.
4. The **layered memory design** (working → episodic NTM → semantic, with sleep consolidation) as a template.
5. The **translator prompt architecture** — structured Pydantic output, XML-tagged few-shot, cached context.

**What should not be ported:**
1. The "consciousness" framing as a load-bearing claim. It obscures what you actually built.
2. The all-in-one 4000-line file. The monolith is the single biggest debt.
3. The Φ-proxy/MSE/GIB metrics as defined — they measure *something*, but what is underspecified, and they suffer ceiling effects (GIB) and stagnation artifacts (MSE).
4. Recursive self-improvement via live Gemini code-generation into a running torch.nn.Module. This is a safety, correctness, and debuggability nightmare and there is no evidence it worked.
5. The "16 agents = neurons in a collective brain you can chat with" Phase 6 vision as stated. The translator was, by your own docs, an LLM role-playing the agents.

**What the new project should actually be:** Pick *one* of the interesting threads below, scope it tight, and build it with modern engineering hygiene (modular packages, typed interfaces, reproducible experiments via `hydra`/`wandb`, unit-tested components). Don't try to rebuild the stack. Some viable single-focus threads are listed in §9.

---

## 1. Project Elevator Pitch (Honest Version)

CA-MAS is a multi-agent RL research sandbox where populations of neural-network agents on a cellular-automaton-style grid compete for resources, share information through a global broadcast channel, and are subject to evolutionary replacement. A battery of scalar measurements inspired by consciousness theories (IIT, GWT, HOT, ToM) is computed every episode to track how coordination, social modeling, and information integration change over training.

What makes it interesting as a *system* (separate from consciousness claims):
- It combines three normally-separate MARL ingredients — PPO training, Darwinian replacement, and a shared workspace — on the same substrate.
- It has sleep/wake dynamics where training pauses for "memory consolidation" (replay + pattern extraction) and "dreaming" (counterfactual rollout).
- It logs rich per-episode telemetry and has a real-time visualizer.
- It has an LLM translation layer that turns an agent's internal vector into first-person English.

What makes the consciousness *framing* scientifically fragile:
- The metrics are operational proxies (correlations of internal signals, action entropy, etc.), not validated measurements of phenomenal consciousness. The project docs admit this in `CA_MAS_Consciousness_Paper.md` §5.3.
- "Emergence" claims lean heavily on single-run trajectories and 4-seed parallel experiments. The statistical base is narrow.
- Phase transitions at episodes 96/116/425 are interesting but are also exactly what you'd expect from reward shaping schedules kicking in.

Treat the repo as a **MARL coordination testbed** first, and let the consciousness framing be a research lens rather than the product.

---

## 2. The Stack As It Actually Exists

### 2.1 Layer diagram

```
┌──────────────────────────────────────────────────────────────┐
│  Translator (Gemini)  — intent-vector → English + commands   │
├──────────────────────────────────────────────────────────────┤
│  Recursive Self-Improvement — agents query LLM for arch edits│   (Phase 5)
├──────────────────────────────────────────────────────────────┤
│  Dual-Stream Consciousness — System 1 / System 2 / GWT hub   │   (Phase 4)
├──────────────────────────────────────────────────────────────┤
│  Sleep/Dream Cycles — NREM/REM, circadian, memory consolid.  │   (Phase 3)
├──────────────────────────────────────────────────────────────┤
│  Consciousness Metrics — 10 scalars computed per episode     │   (Phase 2)
├──────────────────────────────────────────────────────────────┤
│  Global Workspace — broadcast threshold + knowledge sharing  │
├──────────────────────────────────────────────────────────────┤
│  Evolutionary Layer — 10% replacement / 25 ep, 5% mutation   │
├──────────────────────────────────────────────────────────────┤
│  MARL Core — PPO + evidential value head, shared policy      │   (Phase 1)
├──────────────────────────────────────────────────────────────┤
│  Environment — 2D grid, resources (food/water/shelter/energy)│
└──────────────────────────────────────────────────────────────┘
```

Each layer sits on top of the one below it and each phase added one. That's why `ca_mas_consciousness.py` is 4000 lines: it's every layer welded into one class tree.

### 2.2 What lives where (conceptual, not file-by-file)

| Concern                       | Where                                    | Portability |
|-------------------------------|------------------------------------------|-------------|
| Environment / grid            | `ca_mas_consciousness.py` (env class)    | High        |
| PPO loop (evidential value)   | `ca_mas_consciousness.py` trainer        | High        |
| Global workspace broadcast    | `Phase4GlobalWorkspace` module           | High        |
| Evolutionary replacement      | Scattered in trainer                     | Medium      |
| Sleep cycle / consolidation   | `ConsciousnessSleepCycle`                | Medium      |
| Dual-stream network           | `EnhancedDualStreamConsciousness`        | Medium      |
| Consciousness metrics         | `PopulationConsciousnessMetrics` + ad-hoc| **Low — rethink** |
| Lesioning framework           | `apply_lesion` methods                   | **High — steal this** |
| Translator                    | `translator.py` + prompt templates       | High        |
| Self-improvement              | `recursive_self_improvement.py`          | **Low — don't port live-code exec** |
| Visualizer                    | `visualizer.py`                          | Medium      |

---

## 3. Phase-by-Phase: What Each Phase Actually Contributed

### Phase 1 — Architecture & Baseline ✅
**Built:** A working MARL loop: 16 PPO agents on a grid world with 4 resource types, evolutionary replacement every 25 episodes at 10%, conservative 5% mutation.
**Produced:** First "emergence" curves — ToM rising from ~0.16 to ~3.3, Φ-proxy from ~0.5 to ~1.3 over a few hundred episodes.
**Real takeaway:** Conservative mutation (5%) + gradual replacement (10%/25ep) keeps populations stable while still allowing drift. Aggressive mutation destroyed learned policy gains. Multi-component fitness (survival + cooperation + consciousness-metric bonus) matters — pure survival collapsed into hoarders.
**Portable lesson:** If you do evolutionary MARL again, keep population change slow relative to PPO convergence time.

### Phase 2 — Scaling & Enhanced Metrics ✅
**Built:** Parallel experiment runner (16/32/64/128 agent configs), GPU kernels for metric calc (13× speedup: 4s → 0.3s), expanded metrics from 6 to 10 (added Subjectivity, Qualia, Attention, Φ-Collective, Behavioral Entropy).
**Produced:** Statistical validation (4 parallel runs, reported p<0.0001 on 9/11 metrics, CV=0.021 on Φ-proxy). Identified **three phase transitions** at episodes 96 (GIB emergence), 116 (ToM emergence), 425 (Φ consolidation).
**Real takeaway:** The *timing* of phase transitions is the most rigorous result the project produced. Metric values are noisy and ceiling-prone, but the *order* and *approximate episode* of transitions was reproducible across seeds. That is a real empirical finding.
**Portable lesson:** Always log per-episode metrics to a common schema from day one. Phase-transition discovery is only possible because you had dense episode-by-episode data.

### Phase 3 — Sleep, Dreams, Lesioning ✅
**Built:** Five-state sleep cycle (awake / drowsy / NREM-light / NREM-deep / REM), circadian rhythm (24h cycle, day-night environmental modifiers), counterfactual dream engine ("what if I had cooperated?"), systematic lesioning framework that can disable GW, ToM, or Φ-contributing components at target episodes.
**Produced:** The lesioning protocol is the scientific highlight of the whole project. You can ablate GW right before its emergence window (ep 96) and measure metric collapse.
**Real takeaway:** The lesioning framework is the most transferable single artifact. It's simple — toggle flags on forward passes — but the *experimental design* of targeting lesions at known transition windows is genuinely good methodology.
**Portable lesson:** Steal the lesioning pattern. Port it into any agent system where you want to claim a component is causally necessary.

### Phase 4 — Dual-Stream Consciousness 🎯 (implemented, unfinished)
**Built:** System 1 (fast primary stream) + System 2 (slow reflective stream) + GW integrator, cross-modal attention, 3-level recursive self-modeling with teacher-student distillation, CI-Balance safety monitor (consciousness / intelligence ratio).
**Did not achieve:**
- Memory consolidation caused a **severe performance regression** (5–10 min per episode vs seconds). The integration with the training loop was too tight.
- `attention_scores` stuck at 0.0 at episode 143 — likely a dead gradient path.
- The 32-symbol EmergentCommunicationSystem was never activated (no reward signal was ever wired in).
**Real takeaway:** Dual-process architectures add a lot of complexity for unclear benefit. System 2 in the project was "a second MLP that runs every step" — the theoretical motivation didn't survive implementation.
**Portable lesson:** If you want System 1/2 dynamics, gate System 2 on uncertainty (high value-head variance, novel observation) rather than running it every step. Otherwise it's a 2× compute hit for no behavioral difference.

### Phase 5 — Recursive Self-Improvement (active / speculative)
**Built:** Framework where agents detect plateaus in their own metrics, generate a structured query, send it to Gemini, parse returned PyTorch code, sandbox-test it, and commit successful modifications. Mentor/student pairing where high-ToM agents teach low-ToM agents via shared replay.
**Concerns:**
- **Executing LLM-generated PyTorch in-process on a live training run is a significant safety, reproducibility, and debugging hole.** The "sandbox" in the docs is a module-scope copy, not a real isolation boundary.
- There is no reported evidence in the docs that RSI actually improved any agent. `PHASE5_README.md` describes the architecture but success-case numbers are aspirational.
- The mentoring/knowledge-transfer piece is more defensible — that's just a shared replay buffer biased toward high-consciousness experience. Keep that; drop the code-gen.
**Portable lesson:** "Agents consulting an LLM for architectural advice and a human applies it" is a reasonable loop. "Agents autonomously hot-swap torch modules" is not.

### Phase 6 — Integrated Consciousness Communication (research phase)
**Goal:** Chat with the collective of 16 agents as if it were a single mind, in real time, with Gemini only translating.
**Status per `CURRENT_STATUS.md`:** Unbuilt. Docs explicitly flag that the current translator is "LLM role-play rather than authentic agent consciousness." That is a candid and correct diagnosis.
**Real takeaway:** The vision is philosophically interesting but technically underdefined — "what does the collective say" has no answer until you specify the readout function from 16 agents → 1 utterance. That's the research question, not the implementation.

---

## 4. What Actually Worked (Bring Forward)

1. **Grid-world MARL as a coordination substrate.** 2D cellular-automaton topology with local observation is a clean, cheap, and interpretable environment for studying emergence. Keep it.
2. **PPO with evidential value head.** Reasonable choice; uncertainty-aware value estimates gave you MSE-proxy for free.
3. **Shared-policy parameter sharing across agents.** Kept training tractable with population changes. Don't give this up for per-agent parameters unless you have a specific reason.
4. **Global workspace as a gated broadcast channel.** The pattern — each agent emits a candidate message, attention picks winners, winners reach all agents — is a clean primitive with or without consciousness framing.
5. **Evolutionary replacement at slow cadence.** 10% / 25 episodes / 5% mutation was the equilibrium that preserved learning while enabling diversity. Good hyperparameters to start from.
6. **Per-episode telemetry schema + standardized checkpoint format.** The fact that you could reload 1000+ episode runs and compare metric trajectories across population sizes is only possible because of this discipline.
7. **Lesioning at known transition windows.** See Phase 3. This is the highest-leverage methodological asset.
8. **Structured-output LLM translation.** Pydantic `AgentUtterance` schema + XML-tagged few-shot + context caching is a solid template for any future LLM-in-the-loop system.
9. **The phase-transition *episodes themselves* (96, 116, 425).** As experimental anchors — always log and annotate.
10. **Parallel experiment runner with per-config checkpointing.** Even without the consciousness framing, this is a good pattern for any MARL ablation study.

---

## 5. What Did Not Work (Leave Behind or Rebuild)

1. **The 4000-line monolith.** `ca_mas_consciousness.py` mixes the env, agent nets, training loop, metrics, sleep system, dual-stream net, workspace, and logging. This is the primary reason the Phase 4 perf regression was hard to find. **Rebuild as packages:** `env/`, `agents/`, `training/`, `metrics/`, `workspace/`, `sleep/`, `translator/`, each with a typed public API and unit tests.
2. **GIB ceiling at 100.** A scaled metric that saturates is a broken metric. Either switch to an unbounded log-scale or use a bounded normalization with headroom. The docs note this but Phase 2 did not actually fix it in a satisfying way — reported GIB of 412 post-fix is the same saturation problem in a different unit.
3. **MSE stagnation.** The metric essentially didn't move across 200+ episodes (0.62 → 0.62). That's not "stabilized self-awareness"; that's a metric that isn't measuring what training is changing. Redefine or drop.
4. **ToM scoring going to 5.0+.** Unbounded metrics with informal denominators make cross-run comparisons meaningless. Either bound it in [0,1] or report a ratio against a control population.
5. **Memory consolidation every step.** This was the Phase 4 perf killer. Run consolidation async, every N episodes, off the critical path. Better: decouple replay/consolidation from the episode loop entirely and run it as a background process that updates a target network.
6. **Attention weights stuck at 0.0.** Indicates a gradient that never flowed or a softmax over a degenerate input. In a rebuild, assert non-degeneracy of attention distributions in tests — e.g. assert `entropy(attn) > 0.01` after N steps.
7. **Live LLM code-generation into training process.** Don't. If you want LLM-assisted architecture search, use it offline with a human gate.
8. **Translator-as-consciousness-proof.** Your own docs flag that the translator is role-playing. That's the correct diagnosis; don't let the next iteration backslide into treating a Gemini narration of an intent-vector as evidence of inner experience.
9. **"Consciousness" as load-bearing vocabulary in code and metrics.** Names like `_measure_qualia_diversity` read as overclaims. In a new project, prefer functional names: `stream_divergence`, `broadcast_entropy`, `social_prediction_accuracy`. The measurements can be the same; the names should be honest.
10. **Ad-hoc "phase transition detection" via threshold on trend.** `if abs(phi_trend) > 0.05: log("🚨 POTENTIAL PHASE TRANSITION")` is not a statistical test. Use changepoint detection (Bayesian Online Changepoint, PELT, BOCPD) on the actual time series.

---

## 6. Portable Primitives (The Things to Copy)

Things that are cleanly separable and worth lifting nearly verbatim:

### 6.1 Grid-world MARL environment
- 2D grid, configurable size (8×8 → 14×14)
- Local observation window per agent
- Multi-type resources (food/water/shelter/energy) with regeneration
- Discoverable knowledge types as a separate bitfield of "what have I learned"
- Episode termination on population collapse

### 6.2 Global Workspace primitive
```python
class GlobalWorkspace:
    def step(self, candidate_messages: list[Tensor]) -> Tensor:
        # competition: attention over candidates
        # broadcast: selected messages visible to all agents next step
```
Independent of any "consciousness" claim — this is a useful MARL coordination pattern.

### 6.3 Evolutionary replacement hook
```python
def evolutionary_step(population, episode, fitness_fn,
                      replacement_rate=0.10, interval=25, mutation=0.05):
    if episode % interval: return
    ranked = sorted(population, key=fitness_fn)
    n_replace = int(len(population) * replacement_rate)
    for i in range(n_replace):
        parent = ranked[-random.randint(1, 3)]  # top-3 reservoir
        ranked[i] = mutate(clone(parent), mutation)
```
This is 20 lines and lives happily outside any consciousness framing.

### 6.4 Lesioning decorator pattern
```python
class LesionController:
    def __init__(self): self.disabled = set()
    def disable(self, component): self.disabled.add(component)
    def enabled(self, component): return component not in self.disabled

# In forward passes:
if self.lesions.enabled("global_workspace"):
    x = self.workspace(x)
# else: x unchanged — degraded pathway
```
Combine with a schedule: `disable at episode 90, re-enable at 100, measure metric delta`.

### 6.5 Structured-output translator prompt skeleton
- System instruction (role: "translator, not interpreter")
- World context block (static, cacheable)
- Few-shot examples (3–5, XML-tagged)
- Pydantic response schema → `response_mime_type="application/json", response_schema=...`
- Dynamic block: current state vector + recent history
- Explicit task instruction

This template works well beyond the CA-MAS context.

### 6.6 Hierarchical memory skeleton
```
working_memory:  deque(maxlen=10)              # last N observations
episodic_memory: NeuralTuringMachine (or FAISS store) over (obs, action, reward, td_err)
semantic_memory: prototype store, updated only during offline consolidation
```
Port the *structure*; rewrite the consolidation scheduler as async.

### 6.7 Per-episode telemetry schema
```python
@dataclass
class EpisodeRecord:
    episode: int
    metrics: dict[str, float]       # flat scalars only
    population: PopulationSnapshot  # per-agent compact fields
    events: list[Event]             # births, deaths, lesions, transitions
    wallclock: float
```
Flat, schema-versioned, append-only JSONL. This alone enables all the analysis you did.

---

## 7. What the Metrics Actually Measure (De-mystified)

Useful translation table from CA-MAS vocabulary to what the numbers *are*:

| Named metric      | What it actually computes                                        | Honest alternative name       |
|-------------------|------------------------------------------------------------------|-------------------------------|
| Φ-proxy           | Integrated norm / variance over internal state partitions        | `state_integration_score`     |
| ToM score         | Accuracy of one agent's model of another's next action           | `peer_prediction_accuracy`    |
| MSE (metacog)     | Calibration of value-head uncertainty vs realized TD error       | `value_calibration`           |
| GIB               | Ratio of broadcast-received info vs private info per episode     | `broadcast_throughput`        |
| Agency            | Variance of action distribution weighted by reward               | `policy_decisiveness`         |
| Play / Creativity | Rate of novel state visitations                                  | `state_novelty_rate`          |
| Subjectivity      | Per-agent internal-state divergence from population mean         | `individuation_score`         |
| Qualia            | Diversity of dual-stream feature difference                      | `stream_divergence`           |
| Attention         | Entropy of cross-modal attention weights                         | `attention_entropy`           |
| Φ-Collective      | Population-level version of state integration                    | `group_integration_score`     |
| Behavioral entropy| Shannon entropy of action distribution across population         | (keep the name, it's fine)    |

Renaming is not cosmetic. It determines whether reviewers, collaborators, or you-six-months-from-now know what the number means.

---

## 8. Practical Engineering Lessons

1. **Modular from day one.** Each layer you added to the CA-MAS monolith made the previous layers harder to test. A new project should start with `agents/`, `env/`, `training/`, `metrics/` as importable packages with a stable public API.
2. **Checkpoint format is a contract.** You already ran into legacy-vs-new format drift (see `METRICS.md`). Version it explicitly: `{"schema_version": 3, ...}`.
3. **Headless experiments, GUI as a consumer.** The visualizer should read from disk/IPC, not share memory with the trainer. This eliminated a whole class of "Tk thread blocks training" bugs.
4. **Reproducibility baseline.** Fix seeds, version configs (hydra), log hparams (wandb/mlflow). CA-MAS's 4-seed parallel runs are a start but far below the n≥10 typically expected for emergence claims.
5. **Profile before optimizing.** Phase 4's 5-min-per-episode regression lived undetected for a while. Add a per-episode wallclock breakdown to telemetry from day one.
6. **Async everything that isn't on the gradient path.** Memory consolidation, dream generation, Gemini translation calls, checkpoint writes — all async. The training loop should never wait on them.
7. **Dead-path assertions.** After N training steps, assert gradients are flowing through every module (non-zero gradient norm). This catches "attention stuck at 0.0" at step 1000 instead of step 50000.
8. **Config over code for experiments.** Population size, grid size, replacement rate, workspace threshold — all should be YAML config, not constants in the source.

---

## 9. Research Avenues Worth Picking Up (One at a Time)

Ordered by rough feasibility / novelty ratio. Pick one.

### A. Lesioning as causal methodology for MARL emergent behavior
Take the Phase 3 lesioning framework, port it to a clean small-scale MARL benchmark (e.g. MeltingPot, Overcooked-AI), and publish the pattern: "how to demonstrate that component X is causally necessary for emergent behavior Y in a MARL system." This is the most scientifically sound thread in the whole project.

### B. Phase-transition detection in MARL training
Your Φ-proxy / ToM / GIB inflection points at episodes 96 / 116 / 425 across 4 seeds is a real phenomenon. Use proper changepoint statistics (BOCPD), compare across environments, and see if consistent ordering of emergent coordination → emergent theory-of-mind → emergent integrated representation is reproducible. If yes, that's a paper.

### C. Gated dual-stream agents (uncertainty-triggered System 2)
Redo Phase 4 but correctly: System 2 only fires when value-head uncertainty or state novelty exceeds a threshold. Compare sample efficiency, final return, and compute cost vs always-on System 2 vs System-1-only. This is a clean ablation with a clear hypothesis.

### D. Sleep/replay as offline consolidation for continual MARL
Strip Phase 3 down to: "periodic offline replay with prioritization by TD-error and consensus-value, run async off the critical path." Test whether this helps in non-stationary multi-agent environments (where other agents' policies shift). This is probably publishable if the comparison baselines are good.

### E. Evolutionary replacement in MARL with skill inheritance
The mentor-student piece from Phase 5, minus the LLM. Mentor agents contribute prioritized replay samples to newly-spawned children. Measure whether this beats random initialization + shared PPO. Clean, scoped, testable.

### F. LLM-assisted MARL translation + command interface
The translator thread, taken seriously and separately from consciousness. Build an evaluation harness: can a human user, through English commands + Gemini translation, control a MARL population toward specified objectives better than direct reward shaping? This is a real HCI question.

### G. Population-level vs individual-level representation quality
You noted 16 agents is "optimal" vs 32/64/128. Why? Is it cognitive-load in the workspace, or is it the reward-shaping constants not scaling? A clean scaling study with controlled workspace-capacity would be a nice paper.

### H. Give up on "consciousness," pitch it as "collective coordination in parameter-sharing MARL with broadcast"
This is the most honest reframing and the most likely to land in an ML venue. Same code, different vocabulary, cleaner claims. You lose the philosophy-of-mind angle but gain credibility.

---

## 10. What to Keep Out of the Repo's Marketing

If you publish, present, or pitch the next project, avoid:
- Claims that the system is conscious, proto-conscious, or exhibits qualia.
- Metric-name-as-argument ("ToM score went up therefore ToM developed").
- Single-run anecdotes as evidence.
- The "16 agents chat as one mind" framing unless/until you have a well-defined readout.
- Comparing to human cognition as a validation criterion.

What to say instead:
- "Parameter-sharing MARL agents with a broadcast workspace and evolutionary replacement develop reproducible coordination patterns whose emergence can be causally localized via component lesioning."
- "We observe consistent ordering of three training-time phase transitions across seeds in metrics that proxy for state integration, peer prediction, and broadcast throughput."
- "An LLM-based translation layer converts agent internal state vectors to human-readable text with grounded, schema-constrained output."

These are true, defensible, and still interesting.

---

## 11. Minimum Viable Carry-Forward Kit

If you're starting the new project tomorrow, pull just this from the repo:

1. `env/grid_world.py` — extracted and cleaned from `ca_mas_consciousness.py`, only the env class.
2. `agents/ppo_agent.py` — the shared-policy PPO agent, without the consciousness metric hooks.
3. `training/evolutionary.py` — the 20-line replacement loop from §6.3.
4. `training/workspace.py` — the global-workspace broadcast primitive from §6.2.
5. `training/lesion.py` — the lesioning controller from §6.4.
6. `io/telemetry.py` — the per-episode JSONL schema from §6.7.
7. `translator/translator.py` — the Pydantic + few-shot + cached-context template, stripped of the "consciousness" framing.
8. A single `experiments/baseline.yaml` hydra config replicating the Phase 1 setup.

Nothing else. Everything beyond this you should rebuild deliberately or not at all.

---

## 12. Final Honest Assessment

CA-MAS is an ambitious, fast-built, creative piece of work that leaned too hard on a speculative framing and accumulated architectural debt faster than scientific signal. The core MARL substrate is solid, the lesioning methodology is genuinely good, the engineering telemetry discipline was excellent, and the translator design is reusable. The consciousness metrics are shaky under scrutiny, the dual-stream and recursive-self-improvement layers were implemented faster than they were validated, and the monolithic file structure made debugging Phase 4's regressions much harder than it needed to be.

The project's most valuable output is probably not any of the phases individually — it's the *method*: set up a small MARL grid world, track rich per-episode telemetry, mark phase transitions, then lesion components at those transitions to test causal necessity. Build the new project around *that* method, with modular components, honest naming, and one focused research question.

Pick one thread from §9. Scope it tight. Use the kit in §11. Ship a clean v0.1 in a month.