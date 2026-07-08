"""Probe 4 — §7 pause-trigger monitors (observer-side, read-only).

Renders the FROZEN pre-registration's §7 operational-degradation
indicators against a run's telemetry, disaggregated (F6 finding 3 —
never pooled). Run at every check-in:

    python scripts/monitor_probe4_run.py runs/probe4_phase4_biography

Frozen numeric indicators (prereg §7, BUILDER-CONFIRMED 2026-07-07):

- **Entropy collapse** — action entropy below the 5th percentile of its
  own historical baseline for > 1000 consecutive steps.
- **Prediction-error runaway** — monotonic rise across 3 consecutive
  dream cycles. The biography's dream cadence is one block per 2000
  waking steps, so waking-PE block means between dream blocks realize
  "across dream cycles".

Informational vantages (no frozen numeric; multi-vantage per §7):
torpor-analog hint (trailing action-stasis fraction) and dream-session
counts. This script LOGS AND FLAGS ONLY — the §7 escalation ladder is
human: single-vantage anomaly → heighten monitoring; two-vantage
corroboration → PAUSE and review; nothing here stops a run.

Reads only flushed parquet shards (the biography flushes every 2000
rows), so readings lag the live run by up to one shard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ENTROPY_PERCENTILE = 5.0
ENTROPY_RUN_STEPS = 1_000
PE_BLOCK_STEPS = 2_000  # the biography's dream cadence
PE_RISING_BLOCKS = 3
TORPOR_WINDOW = 2_000


def _load_columns(
    telemetry_dir: Path, columns: list[str]
) -> dict[str, np.ndarray]:
    tables = []
    for shard in sorted((telemetry_dir / "agent_step").glob("*.parquet")):
        tables.append(pq.read_table(shard, columns=["t", *columns]))
    if not tables:
        return {}
    rows: dict[str, list[float]] = {name: [] for name in ["t", *columns]}
    for table in tables:
        data = table.to_pydict()
        for name in rows:
            rows[name].extend(data[name])
    order = np.argsort(np.asarray(rows["t"]))
    return {
        name: np.asarray(values, dtype=np.float64)[order]
        for name, values in rows.items()
    }


def check_entropy_collapse(entropy: np.ndarray) -> tuple[bool, str]:
    if entropy.size < ENTROPY_RUN_STEPS:
        return False, f"insufficient history ({entropy.size} steps)"
    p5 = float(np.percentile(entropy, ENTROPY_PERCENTILE))
    below = entropy < p5
    trailing = 0
    for value in below[::-1]:
        if not value:
            break
        trailing += 1
    flagged = trailing > ENTROPY_RUN_STEPS
    return flagged, (
        f"5th pct baseline={p5:.4f}; trailing consecutive below={trailing} "
        f"(flag > {ENTROPY_RUN_STEPS})"
    )


def check_pe_runaway(recon: np.ndarray) -> tuple[bool, str]:
    n_blocks = recon.size // PE_BLOCK_STEPS
    if n_blocks < PE_RISING_BLOCKS + 1:
        return False, f"insufficient dream cycles ({n_blocks} blocks)"
    means = [
        float(recon[i * PE_BLOCK_STEPS : (i + 1) * PE_BLOCK_STEPS].mean())
        for i in range(n_blocks)
    ]
    tail = means[-(PE_RISING_BLOCKS + 1) :]
    rising = all(b > a for a, b in zip(tail, tail[1:]))
    return rising, (
        f"last {PE_RISING_BLOCKS + 1} block means: "
        f"{[round(m, 2) for m in tail]} (flag if strictly rising)"
    )


def torpor_hint(actions: np.ndarray) -> str:
    window = actions[-TORPOR_WINDOW:]
    if window.size == 0:
        return "no data"
    values, counts = np.unique(window, return_counts=True)
    top = float(counts.max()) / window.size
    return (
        f"trailing-{window.size} modal-action fraction {top:.2f} "
        f"(informational; stasis despite gradient is the §7 torpor shape)"
    )


def dream_sessions(telemetry_dir: Path) -> str:
    path = telemetry_dir / "dream_session.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        return "no dream sessions recorded yet"
    lines = sum(1 for line in path.read_text().splitlines() if line.strip())
    return f"{lines} dream-session records"


def main(run_dir: Path) -> int:
    telemetry = run_dir / "telemetry"
    data = _load_columns(
        telemetry, ["policy_entropy_t", "recon_loss_t", "action_t"]
    )
    print(f"§7 monitors — {run_dir}  (flushed telemetry only)")
    if not data:
        print("no flushed agent_step shards yet — nothing to read")
        return 0
    steps = int(data["t"].size)
    print(f"steps read: {steps}")

    entropy_flag, entropy_note = check_entropy_collapse(
        data["policy_entropy_t"]
    )
    pe_flag, pe_note = check_pe_runaway(data["recon_loss_t"])
    print(f"[{'FLAG' if entropy_flag else 'ok  '}] entropy collapse: {entropy_note}")
    print(f"[{'FLAG' if pe_flag else 'ok  '}] PE runaway: {pe_note}")
    print(f"[info] torpor: {torpor_hint(data['action_t'])}")
    print(f"[info] dreams: {dream_sessions(telemetry)}")
    if entropy_flag or pe_flag:
        print(
            "→ single-vantage anomaly: heighten monitoring; corroborate "
            "via the window/mirror before any pause (§7 ladder)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
