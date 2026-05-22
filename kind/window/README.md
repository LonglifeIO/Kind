# Window

Window is a **read-only viewer** over the on-disk records the mirror
writes — the `RoundResult`, `RoundJudgment`, `PassResult`,
`LLMCallAudit`, and telemetry records produced by Phases 9–13. It
serves a small HTML interface so the builder can check in on a long
Probe 4 run without hand-inspecting JSON.

## What Window is — and is not

- **It is** a read-only viewer. Window opens files for reading and
  writes nowhere under `runs/`. It makes no LLM calls. It does not
  touch Io's process.
- **It is not** a monitoring agent. There is no alerting, no push, no
  real-time update — Window is a tool for human inspection at human
  timescales.
- **It is not** interpretive. Window displays records as they are. It
  does not rank, filter, highlight, or weight what it shows.
  Interpretation stays journal-side.

Window reads through the existing Pydantic models, not raw JSON. If a
future phase changes a model's shape, Window inherits the change. If a
record fails to deserialize, Window surfaces the error in the page
rather than papering over it.

## Running it

```
python scripts/run_window.py --run-id phase_13_calibration
```

Arguments:

- `--run-id` (required) — the run to view; its records live under
  `runs/<run-id>/`.
- `--port` (default `8765`) — the local port to serve on.
- `--host` (default `0.0.0.0`) — the bind address. The default lets
  Tailscale reach the server; `127.0.0.1` restricts it to localhost.

## Reaching it over Tailscale

Window serves on `<host>:<port>`. Remote access is the host's
Tailscale setup — Window has no config for it. With the host on the
Tailnet, open `http://<mac-mini-tailscale-name>:8765/` from any device
on the Tailnet. Window does not authenticate, encrypt, or restrict
access; the Tailscale ACL is what gates per-device access if it ever
matters.

## The routes

- `/` — **overview.** Run id and start time, Io's current state,
  wallclock since start, total steps and episodes, pace
  (episodes/hour), and a coarse state-time breakdown over the last 24
  hours and 7 days.
- `/rounds` — **rounds.** The `RoundResult` records under
  `mirror/rounds/`, most-recently-modified first, with checkpoint ids,
  pass count, and judgment verdicts where a matching `RoundJudgment`
  exists.
- `/rounds/<round_id>` — **round detail.** One round's full readings,
  pass by pass.
- `/judgments/<round_id>` — **judgment detail.** One round's
  per-criterion verdicts, confidences, per-falsifier breakdown, and the
  judge's rationale.
- `/audit` — **LLM-call audit.** The `LLMCallAudit` aggregated across
  every round in the run: call / retry / failure totals, wallclock,
  tokens, and a per-role per-checkpoint latency distribution.

## The state-inference rule, and its limits

Io's current state is inferred from telemetry **write activity** — no
explicit state-transition events exist before Probe 3. The rule looks
at the last **5 minutes** of write activity across the four telemetry
streams:

- writes to `agent_step` → **waking**
- writes to `dream_rollout` *alone* → **dreaming**
- writes to `replay_meta` *alone* → **dormant**
- no writes in the window → **paused**

Anything the rule cannot cleanly resolve — `dream_rollout` and
`replay_meta` both active without `agent_step`, or a lone
`world_event` — is surfaced as **unknown** rather than guessed. A run
with no `telemetry/` directory at all (a mirror-only calibration run)
also reports **unknown**.

The 24h / 7d state-time breakdown is **coarse**: `agent_step` and
`world_event` records carry a per-record wallclock, but `dream_rollout`
and `replay_meta` do not — so a real run's breakdown distinguishes
waking hours from idle hours but cannot place dreaming or dormant hours
on the timeline. When Probe 3 lands and emits explicit state-transition
events, the presence-based heuristic can be replaced with an exact one.
