# Probe 4.5 — F2: bounded energy-decoder head — ADOPTED 2026-07-13

**Status: ADOPTED (builder: Gordon, by the delegation recorded in
`probe4_5_preregistration_2026-07-13.md`). This is the dated decision doc the
seek-mechanism classification §9.2 required before F2 could be taken.**

## The decision

`_EnergyDecoder` gains a **config-gated sigmoid output head**:
`WorldModelConfig.energy_decoder_bounded: bool = False`. When False (the
default), the head is the existing unbounded linear — every legacy instance,
test, and archived artifact is byte-identical. When True (set for **all Probe
4.5 instances**), the head's output passes through a sigmoid, bounding
`decode_energy` to the physical [0, 1].

## Why now, and why this form

- The unbounded head is what let the bin-1 defect invert the preference
  geometry: on oracle in-band states the decoder read a mean of **1.124 —
  above the physical ceiling** — making genuinely in-band living *read as
  worse than the rail* (classification §2). The saturation bound in
  `preference.py` caps the damage but cannot fix the geometry; removing the
  impossible regime removes the inversion channel.
- The classification deferred F2 because it is "a substrate change to a
  trained function's gradient field" — a retroactive change to the archived
  Step-0 instance. That concern does not apply here: **Probe 4.5 runs fresh
  instances that train through the sigmoid from scratch.** No trained
  function is altered; no archived artifact is touched.
- Sigmoid over calibration-time clamp: a clamp has zero gradient outside the
  bounds (dead for both the recon term and the maintenance refits); the
  sigmoid is smooth everywhere and matches the [0, 1] target space of
  `sensed_energy`.

## What this does not do

- It does not fix the slope or honesty — that is the §3 maintenance cycle's
  job (F1 pattern). F2 removes the impossible regime; the prereg margins
  judge honesty.
- It does not touch the archived Step-0 instance, the biography instance, or
  any default-config path (default False, byte-identity test-pinned).
- It is not a new head, loss, or signal — same parameters plus one fixed
  nonlinearity, gated by config.

## Scope

`kind/agents/world_model.py` (`_EnergyDecoder` + `WorldModelConfig` field),
Probe 4.5 Phase 1. Tests: default-off byte-identity; bounded-on output range;
recon path unchanged in form. The prereg's out-of-range-mass margin (≤ 1%)
stays in force independently of this adoption, so the honesty criterion does
not silently depend on F2.
