# Arc → Kind extractions

Provenance log. Each entry records a Kind file that drew on Arc, with extraction date, source path(s), and **extent**:

- **verbatim** — copied with renames only; behavior unchanged.
- **structure-consulted** — Kind file written fresh; Arc file informed shape but no code-level reuse.
- **pattern-only** — Kind file written fresh; Arc file's pattern named for orientation, no direct reference.

Arc lives at `/Users/gordon/Documents/Kind/Arc/` (separate git repo, read-only reference; should be moved out of the Kind working tree before further work).

---

## Probe 1 — Plumbing (planning phase, 2026-04-28)

**No verbatim extractions performed.** Two Arc components were considered as verbatim candidates and rejected:

- `Arc/src/ca_mas_consciousness.py:843+` (`NeuralCellularAgent`) — too entangled with `EnhancedDualStreamConsciousness`, `EnhancedRecursiveSelfModel`, and `EmergentCommunicationSystem`. Stripping it down to a clean recurrent PPO would amputate most of the class. Cleaner to write fresh against an external reference.
- `Arc/src/ca_mas_consciousness.py:6376+` (`EvidentialPPOTrainer`) — clean per the survey, but overkill for Probe 1, which uses default-hparam clip-PPO without an evidential value head. Park for Probe 4 if uncertainty-aware values become useful for distinguishability.

Files Kind will fresh-write with structure consulted from Arc (or external references):

| Kind file | Reference | Extent |
| --- | --- | --- |
| `kind/env/grid.py` | `Arc/src/ca_mas_consciousness.py:2998+` (`CA_MAS_Environment`) | structure-consulted |
| `kind/agents/recurrent_ppo.py` | CleanRL `ppo_atari_lstm.py` (external); Arc considered, rejected | external reference; Arc not used |
| `kind/training/ppo_trainer.py` | CleanRL recurrent PPO loop; Schulman 2017 (clip-PPO), Schulman 2016 (GAE); Arc considered, rejected | external references; Arc not used |
| `kind/mirror/caller.py` | `Arc/src/translator.py` (Pydantic schema + XML-tagged few-shot + context caching skeleton) | pattern-only |
| `kind/observer/records.py` | `docs/plans/v.0.1.0/camas_carryover.md` §6.7 (per-episode telemetry schema) | structure-consulted |
| `kind/env/perturbation.py` | none | fresh — no Arc precedent |
| `kind/env/reward.py` | none | fresh — Arc reward was entangled with consciousness metrics |

**Renames** at the function/class level when Arc names appear in Kind code or comments:

| Arc | Kind | Reason |
| --- | --- | --- |
| `CAMASTranslator` | `Mirror` | Role as interpreter, not narrator |
| `AgentUtterance` | `MirrorReading` | Reading observation, not generating speech |
| `translate_agent_state(...)` | `read_telemetry(...)` | Reading what is, not narrating from inside |

---

## Future probes (forward-looking)

Anticipated extractions, to be confirmed at probe time:

| Probe | Kind file | Arc source | Extent |
| --- | --- | --- | --- |
| Probe 2 | `kind/training/ablation.py` | `Arc/src/ca_mas_consciousness.py:1520+` (`ConsciousnessLesioning`) | verbatim with renames + component-registry generalization |
| Probe 3 | `kind/dream/hierarchical_memory.py` | `Arc/src/memory_system/hierarchical_memory.py` | verbatim |
| Probe 3 | `kind/dream/episodic_memory.py` | `Arc/src/memory_system/episodic_memory.py` | verbatim |
| Probe 3 | `kind/dream/consolidation.py` | `Arc/src/memory_system/consolidation.py` | verbatim |
| Probe 3 | `kind/dream/state_machine.py` | `Arc/src/ca_mas_consciousness.py:1215+` (`ConsciousnessSleepCycle`) | pattern-only |
| Probe 4 | `kind/agents/evidential_head.py` | `Arc/src/ca_mas_consciousness.py:843+` (evidential head only, ~30 lines) | verbatim with renames |
| Probe 4 | `kind/training/ppo_trainer_evidential.py` | `Arc/src/ca_mas_consciousness.py:6376+` (`EvidentialPPOTrainer`) | verbatim with renames |
| Probe 4 | `kind/observer/changepoint.py` | none — Arc's ad-hoc threshold not portable; use `ruptures` (BOCPD/PELT) | fresh, external library |

This table is a forecast, not a commitment. Each row gets re-evaluated when the probe begins.
