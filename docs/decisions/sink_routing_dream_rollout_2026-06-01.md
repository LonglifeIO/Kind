# Sink routing for dict-typed telemetry fields — 2026-06-01

A scoped diagnose → fix → adopt pass (not a probe phase) resolving the
`ParquetSink`/dict-field gap that Probe 3 Phase 3 surfaced as plan-level
question 1. It also records the resolution of the Phase 3 dream-emission fork
that the fix uncovered, and adopts a full-suite-per-phase validation
convention.

## The gap

Probe 3 Phase 0 added `DreamRollout.sampling_parameters: dict[str, float | int |
str | bool]`. `ParquetSink` derives its Arrow schema from the Pydantic model at
construction and *raised* on any `dict`-origin annotation (`sinks.py`:
"ParquetSink: dict fields are not supported — route the record to JsonlSink
instead"). The runner wires the `dream_rollout` stream to
`ParquetSink(..., DreamRollout, ...)`, so constructing that sink — or any
`Runner`, or any test that builds a `DreamRollout` ParquetSink — raised.

**Diagnosis (observation).** The full `pytest` suite had **52 failures**, and
all 52 traced to the single `sinks.py` dict-rejection at construction time (104
matching traceback lines = 52 location lines + 52 `E ValueError` lines; no
failed test had a non-matching error). They were red since Phase 0 and went
undetected because per-phase validation ran only each phase's *new* tests, not
the full suite. Scope: the only dict-typed field on a Parquet-routed model is
`DreamRollout.sampling_parameters` (`WorldEvent.payload` is dict but routes to
`JsonlSink`; `AgentStep`/`ReplayMeta` have none) — so a *generic* fix covers the
gap and any future dict field without being DreamRollout-specific.

## The fix (chosen: generic JSON-encode in ParquetSink)

`ParquetSink` stores dict-typed fields JSON-encoded in a `pa.string()` column.
The write path `json.dumps`es dict fields (None stays a null cell); a symmetric
read path `json.loads`es them back to dicts before `model_validate`, so
consumers transparently see a dict, not a string.

- `_arrow_type_for(dict)` returns `pa.string()` instead of raising.
- `json_encoded_field_names(model)` is the single source of truth for which
  fields are JSON-encoded (dict-origin after unwrapping `Optional`), shared by
  write (encode) and read (decode).
- New read helpers: `decode_parquet_row(model, row)` (JSON-decodes the model's
  dict fields; no-op for dict-free models), `model_validate_parquet_row(model,
  row)` (decode + validate), `read_parquet_dir(dir, model)` (the canonical
  shard reader). The canonical Window loader (`kind/window/loaders.py`) routes
  through `model_validate_parquet_row`, so digest/mirror consumers inherit the
  decode for free.
- The `DreamRollout` Pydantic model and the frozen `schemas/v0.4.0.json` export
  are **unchanged** — this is a serialization-layer fix, not a schema change.

### Rationale (interpretation)

- **vs. route DreamRollout to JsonlSink (rejected).** Would store the fat
  sequence arrays (`sequence_z_prior`, `sequence_h`, `sequence_decoded_obs`,
  …) as verbose JSON text, losing Parquet's columnar efficiency, and split
  `dream_rollout` off from the other columnar stream.
- **vs. schema change flattening `sampling_parameters` into typed scalars
  (rejected).** Reopens the settled Phase 0 schema, forces a new frozen
  `v0.4.0.json`, and loses the dict's variable-key flexibility across regimes
  (the keys differ between dream / pure-prior / control configs).
- **JSON-string encoding** keeps the columnar streams together, is a no-op for
  dict-free models (so the shared sink stays safe for `agent_step`), greens
  anything that hit the dict wall (generic, not DreamRollout-specific), and
  leaves the logical model and its JSON Schema export untouched. The
  heterogeneous value union (`int | float | str | bool`) round-trips through
  JSON with types intact, where a native Arrow `map`/`struct` could not.

## The Phase 3 dream-emission fork (resolved: Option 1)

Fixing the sink took the suite **52 → 4**. The 4 remaining failures were *not*
the sink: they were the Probe 3 Phase 3 runner change (a new `_emit_dream`
delegating to the four-axis `emit_dream_rollout`) running on the waking
cadence. The integration smokes encode the **Probe 1.5 waking-cadence
calibration-handshake contract** (`schema_version == "0.2.0"`, `len(seq) ==
dream_horizon`, current-state seed, `sequence_self_prediction is None`); the
four-axis regime (`"0.3.0"`, replay-seeded, needs a ≥`replay_min_segment_age_steps`
-aged buffer → 0 dreams in a 200-step smoke) is incompatible on that path.

**Decision (Option 1).** The waking-cadence `_emit_dream` returns to the Probe
1.5 calibration handshake; the four-axis `emit_dream_rollout` / `run_dream_session`
are reserved for Phase 4's state-machine-driven dream sessions (plan §2.3 /
Phase 4: "the state machine consumes the dream-rollout module"). Concretely the
Phase 3 runner rename + delegation was reverted (`kind/training/runner.py` is
back to its pre-Phase-3 state); the four-axis regime lives in
`kind/training/dream.py`, exercised today by the visibility smoke (which calls
`emit_dream_rollout` directly and uses its own standalone handshake replica) and
later by Phase 4. This greened the 4. The alternative (keep the delegation on
the waking cadence, rewrite the smokes to the 0.3.0 contract + add a short-run
seed config) was rejected: it relocates the four-axis regime onto the waking
path and rewrites a contract the test suite encodes.

## Convention adopted: full suite per phase

From Phase 4 onward, per-phase validation runs the **full** `pytest` suite (must
be green), not just the phase's new tests. The 52-failure gap existed for three
phases precisely because validation was scoped to new tests. Recorded in
`CLAUDE.md` (Discipline) and the Phase 3 journal.

## Known unrelated flake (out of scope)

`tests/test_transport.py::test_barrier_queues_mutates_and_drains_in_order` is
pre-existing flaky (≈3/5 fail in isolation; a threading/timing barrier test),
not among the 52, independent of this fix. Flagged, not fixed — fixing it is
out of this pass's scope. It can make a single full-suite run non-deterministic;
re-run in isolation to distinguish it from a real regression.

## What changed

- `kind/observer/sinks.py` — `_arrow_type_for(dict)` → `pa.string()`;
  `json_encoded_field_names`, `decode_parquet_row`, `model_validate_parquet_row`,
  `read_parquet_dir`; `ParquetSink.write` JSON-encodes dict fields.
- `kind/window/loaders.py` — the parquet loader reconstructs rows via
  `model_validate_parquet_row` (decode-aware).
- `kind/training/runner.py` — reverted to pre-Phase-3 (Option 1).
- `scripts/smoke_probe3_visibility.py` — comments updated (the renamed method no
  longer exists; the replica mirrors the Probe 1.5 `_emit_dream` handshake).
- `tests/test_sinks.py` — dict-field round-trip + helper coverage.
- `CLAUDE.md`, `docs/workingjournal/probe3.md` — convention + journal.

## Verification

- Full `pytest` suite green (52 → 0; the integration smokes pass on the restored
  handshake), modulo the known `test_transport` flake.
- `schemas/v0.4.0.json` byte-identical; `DreamRollout` model unchanged.
- A dict-bearing `"0.3.0"` `DreamRollout` (heterogeneous `sampling_parameters`)
  writes and reads back through `ParquetSink` with the dict and value types
  intact (`tests/test_sinks.py`).
- `mypy --strict` clean on touched modules.

## Watts default

Serialization-layer only — no Io-readable surface. The encode/decode is a
storage detail of the telemetry sink, which Io does not read; `PolicyView` is
untouched. The runner revert restores the prior waking-cadence behavior, adding
no new actor-readable interface. `new_actor_readable_interfaces_added = []`
continues to hold.
