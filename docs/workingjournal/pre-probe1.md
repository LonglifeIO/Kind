# Probe 1 — Pre-Build Notes

## Risks named going in

The defaults in §6 (latent dim 16, h=200, free bits 1.0/dim, K=5, etc.) are
calibrated to give the substrate work to do without overwhelming it on the
8×8 grid. Two failure modes to distinguish when reading smoke output:

1. Default calibration off. Examples: KL pinned at free-bits floor for
   a few thousand steps, ensemble disagreement saturating within an episode,
   regrowth events too rare to register. Response: revise defaults per §6's
   "Revisit when" column. Re-run smoke.

2. Small-environment regime assumptions wrong. Examples: latent collapses
   in a way free bits doesn't fix, ensemble disagreement saturating in 100
   steps regardless of K, decoder producing dreams that look nothing like
   the env. Response: this is data for Probe 2's research prompt, not a
   tuning problem. Capture in journal what surfaced; do not retune
   indefinitely.

The plan does not promise the smoke passes on the first try.

## Build notes — decisions worth surfacing

These are the structural calls made mid-build that the smoke will end up
testing. Recorded here so that when the smoke surfaces something, the
distinction between "tuning miss" and "regime-assumption miss" has the
right context.

### Phase 1 — sinks (2026-05-02)

PyArrow column schema is derived from the Pydantic model's field info up
front, not inferred from the first batch. This means an Optional field
stays nullable across all shards even if the early shards happen to have
all-None for it. The cost is a small Pydantic→Arrow type converter; the
benefit is that reads can concatenate across shards without fighting
schema drift. ParquetSink rejects dict-bearing models (WorldEvent.payload)
at __init__, not at write time — the routing mistake (sending a JSONL
stream to Parquet) surfaces at sink construction, not 10k records in.

### Phase 2b — world model (2026-05-02)

Three structural choices that inform what to look at when smoke results
come back:

1. log-σ on the prior and posterior heads is **not bounded**. Free bits is
   the only stability mechanism, per plan §2.5's rule against DreamerV3
   robustness machinery. If the smoke shows σ collapsing toward zero
   (posterior collapse despite free bits) or exploding, that is failure
   mode (2) above — a regime-assumption miss, not a tuning miss. Resist
   adding a clamp without first reading what the assumption actually was.

2. Encoder is 3 conv layers (16/32/64) with a final linear projection to
   `embed_dim=256`. The plan permitted 3 or 4. 3 keeps the parameter count
   modest and gives a 4×4 spatial bottleneck before the projection; if
   reconstructions are blurry in a way the mirror can't read, 4 layers
   (and a 2×2 bottleneck) is the first thing to try.

3. `WorldModel.loss()` returns `kl_aggregate_unclipped` alongside the
   free-bits-clipped `kl`. The unclipped value is *for telemetry only* —
   the mirror needs it to track posterior-prior drift across episodes.
   The actor must not consume it (synthesis §Q5: posterior-prior KL into
   Io's reward stream is a structural form of self-readout). The
   PolicyView/TelemetryView split (Phase 3c) is what enforces this in
   code; the world model just exposes the conduit honestly.

Defaults from §6 (h=200, z=16, free bits 1.0/dim) are pinned in
`WorldModelConfig` as the production sizes; the gate test runs at
h=32/z=4/embed=64 for CPU speed without changing the structural shape.
