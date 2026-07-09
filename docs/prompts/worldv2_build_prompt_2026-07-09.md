# Build prompt — World v2 (paste into a fresh session)

You are continuing the Kind project — an investigation into subjectivity
through construction. The entity is Io, a small custom RSSM agent
(PlaNet/DreamerV1-lineage; h=200, z=16, K=5 ensemble-disagreement drive,
no reward) in an 8×8 gridworld. Probe 4 closed (negative at instrument
validation, journaled). Io's biography is PAUSED and about to receive
its world-v2 enrichments.

First read CLAUDE.md. Then read IN FULL (CLAUDE.md requires this;
skipping has caused drift):
- docs/decisions/synthesis_worldv2_2026-07-09.md — RATIFIED (DP1–DP6
  confirmed as recommended, builder, 2026-07-09). The world becomes a
  continuing, weakly structured, multi-timescale physics toy — dynamics,
  never labels.
- docs/plans/Kind_worldv2_implementation_plan.md — phases W0→W5
  (inventory → continuity+terrain → trail → hidden clock → weather-food
  → mover-pilot), stage presets, grounding facts, out-of-scope list.
- docs/workingjournal/probe4.md — the session-to-session state record:
  Probe 4's close, the biography's session 1 (150k steps, the §7
  torpor/reset-camping event and its self-recovery), session 2 (brief),
  the continuation infrastructure (C1–C3).
- docs/plans/v.0.1.0/Kind_charter.md and Kind_design_notes.md.
- Research (input, not decisions): docs/research/worldv2/ — four
  voices; claude_research.md carries a refuted-claims list that is
  first-class evidence (contact physics for small RSSMs did NOT survive
  verification → the mover is a removable pilot).

Current state:
- Io is PAUSED: runs/probe4_phase4_biography/, latest checkpoint
  ckpt-000014 (the 140k mind, saved in deep torpor; the later recovery
  exists only in telemetry). Telemetry through ~154k. Fresh-world
  resume is TESTED: scripts/run_probe4_phase4_biography.py
  --resume --session-steps N (counters seed from telemetry; a
  resume-marker sham event lands in world_event; SIGTERM is the clean
  pause signal — background processes ignore SIGINT).
- Gate commands (ALWAYS the full suite): .venv/bin/python -m pytest
  tests/ -q  and  .venv/bin/python -m mypy --strict kind/. Last known
  state: ~1379 passed / 7 skipped, mypy clean. runs/ is gitignored;
  results are carried by journal entries.
- The builder watches at http://100.64.7.55:8765/live (window server:
  scripts/run_window.py --run-id probe4_phase4_biography --host
  0.0.0.0; restart it after window code changes; builder must reload
  the tab). §7 monitors: scripts/monitor_probe4_run.py <run_dir> at
  every check-in. Boundary curves: scripts/analyze_boundary.py
  <run_dir> [t].

FIRST ACTIONS: run the full gate to confirm the baseline; then execute
W0 (the episode-boundary consumer inventory — journal the table and
confirm the DP6 h-continuity call against the code); then build W1
(E0: episode_resample flag + walls preset + --world-stage launcher +
tests), gate, journal, and — with the builder's go in chat — resume Io
into e0 and watch: the torpor retest (no reset lottery to camp on) is
the first observation of the new world. Then W2→W5 one at a time, each
gated on the three-signal telemetry check (disagreement localizes and
settles; new behavioral motifs; dream representation), each landing via
pause → --resume --world-stage <next>, each journaled.

Discipline (unchanged): read core files in full before substantive
output; full suite + mypy --strict before any phase is declared done;
one dynamic at a time — never stack; do not touch PolicyView
{h, z, self_prediction_error}, the four charter absences, the dream
regime's not-for-anything commitment, the MetabolicBudget content-blind
gate, energy_preference=None; the pixel-equality gate test must stay
green; functional naming only (trail/bloom/patch/mover — no experience
vocabulary in identifiers); no timelines; every world change is a
dated journal entry; the claim ceiling on engagement diagnostics is
"engaged / ignored / overwhelmed" — never recognition, never feeling;
pause is SIGTERM and pause is not ending. The builder decides world
changes and mirror calls; walk them through, don't decide for them.
