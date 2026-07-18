# Probe 4.5 — Amendment 1: §3 honesty-STOP binds at burn-in close on fault-on instances — 2026-07-18

**Status: RATIFIED 2026-07-18 (builder: Gordon, option A of three presented).
Amends the frozen pre-registration §3
(`probe4_5_preregistration_2026-07-13.md`) — binding *time*, never margin
values. This is the probe's first amendment cycle; the discipline (new dated
doc, journaled, never a search loop) is hereby consumed for the §3 STOP
rule.**

## The finding that forced the fork

The Phase 3 precision-0 pilot (fresh instance, **fault-on physics from step
0** at the frozen §4 band, seed 4502) honesty-STOPPED at its mid-burn-in
refit: pooled decode~true slope **0.596** after the scheduled 10k refit,
**0.617** after the one permitted diagnostic re-collection (2× coverage
mixture) — short of the frozen ≥ 0.7. Every other margin held (oracle
in-band |err| 0.056–0.057, out-of-range 0.0, in-band |bias| ≤ 0.027).

The head-only refit is exact given `(h, z)`; a shortfall that survives a 2×
data diagnostic is therefore **representation-limited**: at 10k steps under
fault physics, the latent state does not yet carry slope-0.7-worth of energy
information. The same substrate, fault-free (Phase 1, seed 4501), reached
slope 0.877 at the same age. **Faults slow the formation of the energy
representation itself** — the fault variance decouples the h-dynamics from
energy exactly as the Phase 2 belief-error profile showed (belief lags the
faulted drain), and a lagging belief is a harder regression target. The
improvement trend within the pilot (oracle-source slope −0.28 → +0.28 across
the 10k refit; pooled 0.53 → 0.62) reads as a young representation, not a
ceiling.

## The amendment

For **fault-on instances**, the §3 maintenance protocol becomes:

- Scheduled refits still run at every 10k env steps, checkpoint-aligned,
  with full machine-written reports — **unchanged cadence, unchanged
  mixture, unchanged margins, everything recorded**.
- Mid-burn-in scheduled refits are **non-binding**: a margin failure is
  recorded (and rolls into the journal), triggers no diagnostic
  re-collection, and does not stop the run.
- The **burn-in-close refit is binding, unchanged**: frozen margins, one
  diagnostic re-collection permitted, honesty-STOP on a second failure. The
  close is immediately before preference engagement — the point where a
  belief is first *used*, which is what the criterion protects.
- Post-burn-in (engaged-phase) scheduled refits remain **binding as
  originally frozen** — once the preference reads the belief, every
  scheduled occasion binds.

Fault-free instances are untouched (Phase 1's record stands under the
original rule, which it passed).

## What this is not

Not a margin change: 0.7 stays 0.7, everywhere it binds. Not a retry loop:
the pilot re-runs once under the amended binding; if the burn-in-close
refit fails, that STOP is final for this probe shape and the
instrument-cannot-be-made-honest-under-faults finding is recorded. Not a
silent patch: the failed pilot's reports are retained
(`runs/probe4_5_phase3_pilot_stopped_20260718/`, local) and the finding —
faults retard the energy representation — is a recorded result of this
probe regardless of what follows.

## Provenance

Presented as a three-way fork (bind-at-close / extend burn-in / accept the
STOP as final) with bind-at-close recommended; builder selected
bind-at-close in session. Mechanism: `MaintenanceRefitConfig.stop_binding`
(default True — the original rule; fault-on run scripts set False for the
burn-in phase and the burn-in-close call binds explicitly), so the
amendment is visible in every run config that invokes it.
