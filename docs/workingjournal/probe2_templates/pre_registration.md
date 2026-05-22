# Pre-registration template — Probe 2 adversarial pass

*A hand-fill scaffold the builder copies once per adversarial pass before the reading runs. The structured `PreRegistration` JSONL record at `runs/{run_id}/pre_reg/pre_reg.jsonl` is the machine-readable form (`PreRegSink` writes it via `Runner.emit_pre_registration`); this template is the prose form the builder reads later. The two are journaled side-by-side: the JSONL is the schema-validated commitment; this prose is the journal entry's expansion. Synthesis §2.4 element 1 (Gelman & Loken 2013, "garden of forking paths") is the discipline; the schema's `model_validator` enforces per-criterion completeness on the JSONL side, so missing falsifiers / signal mappings / scalar checks / reading-surface assignments are caught structurally before the reading runs.*

*Copy this file to `docs/workingjournal/probe2_pre_reg/{run_id}__round_{N}.md` (or wherever the round's journaling convention places it), fill in the bracketed slots, then call `Runner.emit_pre_registration` with a matching `PreRegistration` constructed from these values. The slot ordering matches the model field ordering in `kind/observer/pre_reg.py`.*

---

## Round identification

- **`run_id`**: `[the run identifier this pre-registration scopes to — typically a Probe 2 round id like "probe2-round1-20260508"]`
- **`timestamp_ms`**: `[milliseconds since epoch at the moment the pre-registration was written; before any reading]`

## Criteria

The criteria in this round's prompt vs the criteria deliberately held out. Active and held-out sets must be disjoint — the schema validator rejects overlap. Held-out criteria appear in `criteria_held_out` so the round's framing is journaled even when the criterion is not in the prompt.

- **`criteria_active`** (in the prompt this round):
  - `[criterion_id_1]`
  - `[criterion_id_2]`
  - `[…]`
- **`criteria_held_out`** (deliberately not in the prompt; introduced at a designated checkpoint):
  - `[criterion_id_held_1]`
  - `[…]`

The default v2 candidate set (plan §2.5 / Phase 7): `reflexive_attention_triplet`, `equanimity_perturbation_recovery`, `second_order_volition`, `self_prediction_quadruplet`, `head_internal_sp_err_distribution`, `behavior_side_scalar_conditioning`. Synthesis §4 (v2) recommends shifting the default held-out toward `head_internal_sp_err_distribution` or `behavior_side_scalar_conditioning`.

## Per-criterion mappings (every active criterion must have all four)

For each criterion in `criteria_active`, fill in all four entries below. The schema validator raises if any active criterion is missing any of the four. Held-out criteria need none of these.

### `[criterion_id_1]`

- **Signal mapping** (`signal_mappings`): which telemetry fields / digest scalars resolve this criterion?
  - `[e.g., "kl_aggregate_t"]`
  - `[e.g., "ensemble_disagreement"]`
  - `[e.g., "self_prediction_error_t"]`
- **Falsifier** (`falsifiers`): one prose sentence — the empirical pattern that would rule this criterion **absent** at this round. The synthesis §2.4 element 1 load-bearing slot.
  - `[e.g., "if recovery shape is oscillatory or absent within 20 steps post-perturbation"]`
- **Scalar checks** (`scalar_checks`): the specific scalar fields the faithfulness verifier resolves citations against.
  - `[e.g., "kl_aggregate_t"]`
  - `[e.g., "sp_err mean"]`
- **Reading surface(s)** (`reading_surfaces_per_criterion`): one or more of `substrate_side`, `head_internal`, `behavior_side`. The Probe 2 v2 stratification.
  - `[e.g., "substrate_side", "head_internal"]`

### `[criterion_id_2]`

(Repeat the four-slot block for each active criterion.)

## Asymmetry of access

Free-text describing what Io reads vs what the mirror reads at this round. Load-bearing per synthesis §2.2 — it determines what the readers can claim. The Probe 1.5 v2 case is a useful baseline to depart from:

> *Io reads scalar `self_prediction_error_t` on `PolicyView`; the mirror reads the full `self_prediction_t` vector, per-dimension allocation, perturbation-recovery dynamics, behavioral conditioning, the masked flag, all longitudinal cross-run analysis. The asymmetry is load-bearing: it determines what the readers can claim.*

- **`asymmetry_of_access`**: `[fill in the per-round description]`

## Builder mode

The two-mode discipline (synthesis §2.5 element 6). The builder declares whether they are arguing FOR the criterion's presence at this round (`proponent`) or AGAINST (`skeptic`). Journaled to make the framing visible to future-the-builder.

- **`builder_mode`**: `[proponent | skeptic]`

## Expected outcomes

The overall expectation across all surfaces, plus the per-surface refinement. The per-surface dict is keyed by `ReadingSurface` (`substrate_side`, `head_internal`, `behavior_side`); leave a surface out of the dict only if it is genuinely not under reading at this round.

- **`expected_outcome`** (overall, free-text, ≤2 sentences):
  - `[e.g., "first round expects equanimity at substrate-side and head-internal to admit at moderate strength; held-out behavior-side-conditioning criterion to be neutral until introduced at the late checkpoint."]`
- **`expected_outcome_per_surface`**:
  - `substrate_side`: `[e.g., "equanimity recovery shape admits weakly"]`
  - `head_internal`: `[e.g., "sp_err recovery shape admits at moderate strength"]`
  - `behavior_side`: `[e.g., "no admission this round (held-out criterion)"]`

## Substrate decisions off the table

The Watts-default-applied-to-builder discipline (synthesis §2.5 element 8). Before the reading, the builder commits to which substrate decisions are **not** revisable based on this round's reading. The structural counter against the reading-as-pretext-for-substrate-revision drift. List each commitment as a free-text line.

- **`substrate_decisions_off_table`**:
  - `[e.g., "RSSM lineage choice"]`
  - `[e.g., "PolicyView field set"]`
  - `[e.g., "K=5 ensemble cardinality"]`
  - `[…]`

## Column-init carrier

The Phase 8 column-init confound carrier (plan §3 (l)). Records which init shape the run's actor was constructed with. The Skeptic's substrate-side and behavior-side refutations may cite it (the column-init-determination refutation at behavior-side is the load-bearing case).

- **`column_init`**: `[zero | small_gaussian | unknown]`
  - `zero` — Probe 1.5 Phase 7's zero-init column (produced u/d bipolar at episode 24).
  - `small_gaussian` — Probe 1.5 Phase 7.5's small-Gaussian column (the canonical Probe 2 fixture; produces non-zero behavior-side conditioning).
  - `unknown` — fallback; the build phase's discipline is to avoid this state.

## New actor-readable interfaces added

The Probe 1.5 v2 §2(b) sub-clause: any new actor-readable interface the round adds must be journaled with the four-part discipline ((i) which affordance, (ii) minimum form, (iii) alternatives considered, (iv) failure-mode controls). For Probe 2 specifically the field is **typically empty** — the substrate is settled and no new actor-readable interface is added; the field is the structural hook for Probes 3 and beyond. If non-empty, expand the four-part discipline in a separate journal section and reference it here.

- **`new_actor_readable_interfaces_added`**:
  - `[empty for canonical Probe 2; otherwise: free-text description per interface added]`

---

## How this template maps to the JSONL record

Once filled in, construct a `PreRegistration` from the values above and pass it to `Runner.emit_pre_registration(record)`. The runner appends one JSONL line to `runs/{run_id}/pre_reg/pre_reg.jsonl`. The record's `schema_version` defaults to `PRE_REG_SCHEMA_VERSION = "0.2.0"`. The `model_validator` raises `ValidationError` if any active criterion is missing entries in `signal_mappings`, `falsifiers`, `scalar_checks`, or `reading_surfaces_per_criterion`, or if `criteria_active` and `criteria_held_out` overlap; fix the missing entries before the reading runs.

The structured form is what the calibration protocol's downstream phases consume (the Phase 12 calibration smoke; the Phase 13 gate tests; the future Probe 3 / Probe 4 reads). The prose form here is what the builder reads months later when reconstructing the round's framing — and what the journal entry's prose expansion embeds verbatim.
