"""Continuation plan C3 — boundary re-engagement curves (observer-side).

Renders per-block means of the biography's core curves (prediction
error, curiosity/intrinsic signal, energy, action entropy, meals)
before vs. after a boundary step — a resume, a world change, or any
step of interest. Disaggregated, no thresholds, no verdicts: this is
the instrument for the "does a nearly-bored drive re-engage when the
world changes?" observation, not a probe.

    python scripts/analyze_boundary.py runs/probe4_phase4_biography [boundary_t]

Without ``boundary_t``, the boundary defaults to the latest resume
marker (``sham_label == "resume_marker"``) in the world_event stream.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

BLOCK = 2_000
CONSUMPTION_JUMP = 0.03  # house threshold

COLUMNS = [
    "t",
    "recon_loss_t",
    "intrinsic_signal_t",
    "true_energy_t",
    "policy_entropy_t",
]


def _load(telemetry_dir: Path) -> dict[str, np.ndarray]:
    rows: dict[str, list[float]] = {name: [] for name in COLUMNS}
    for shard in sorted((telemetry_dir / "agent_step").glob("*.parquet")):
        data = pq.read_table(shard, columns=COLUMNS).to_pydict()
        for name in COLUMNS:
            rows[name].extend(data[name])
    order = np.argsort(np.asarray(rows["t"]))
    return {
        name: np.asarray(values, dtype=np.float64)[order]
        for name, values in rows.items()
    }


def _patch_centers(telemetry_dir: Path) -> list[tuple[int, int, int]]:
    """(t_event, center_r, center_c) per patch move, in stream order.

    Empty when the run has no weather patch (pre-e3). The first entry
    is synthesized from the first move's ``center_from`` so the
    timeline covers the stretch before the first move.
    """
    path = telemetry_dir / "world_event.jsonl"
    if not path.exists():
        return []
    centers: list[tuple[int, int, int]] = []
    for line in path.read_text().splitlines():
        if '"patch_drift"' not in line:
            continue
        record = json.loads(line)
        payload = record.get("payload", {})
        if payload.get("process") != "patch_drift":
            continue
        t_event = int(record["t_event"])
        if not centers:
            from_r, from_c = payload["center_from"]
            centers.append((0, int(from_r), int(from_c)))
        to_r, to_c = payload["center_to"]
        centers.append((t_event, int(to_r), int(to_c)))
    return centers


def _positions(run_dir: Path) -> dict[int, tuple[int, int]]:
    """t → agent position from the run script's sidecar (empty if the
    sidecar predates world v2 W4)."""
    path = run_dir / "agent_pos.jsonl"
    if not path.exists():
        return {}
    out: dict[int, tuple[int, int]] = {}
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
            out[int(record["t"])] = (
                int(record["pos"][0]),
                int(record["pos"][1]),
            )
        except (ValueError, KeyError, IndexError):
            continue  # torn tail line from the live buffer
    return out


def _occupancy_share(
    positions: dict[int, tuple[int, int]],
    centers: list[tuple[int, int, int]],
    lo: int,
    hi: int,
    patch_half: int = 1,
) -> float | None:
    """Share of steps in [lo, hi) spent inside the patch square (C4
    crowd-out watch). None when either record is missing."""
    if not positions or not centers:
        return None
    inside = 0
    counted = 0
    idx = 0
    for t in range(lo, hi):
        pos = positions.get(t)
        if pos is None:
            continue
        while idx + 1 < len(centers) and centers[idx + 1][0] <= t:
            idx += 1
        _, cr, cc = centers[idx]
        counted += 1
        if abs(pos[0] - cr) <= patch_half and abs(pos[1] - cc) <= patch_half:
            inside += 1
    if counted == 0:
        return None
    return inside / counted


def _latest_resume_marker(telemetry_dir: Path) -> int | None:
    path = telemetry_dir / "world_event.jsonl"
    if not path.exists():
        return None
    marker_t: int | None = None
    for line in path.read_text().splitlines():
        if '"resume_marker"' not in line:
            continue
        record = json.loads(line)
        if record.get("payload", {}).get("sham_label") == "resume_marker":
            marker_t = int(record["t_event"])
    return marker_t


def _block_rows(
    arr: dict[str, np.ndarray], lo: int, hi: int
) -> list[tuple[str, float, float, float, float, int]]:
    """(label, pe, intrinsic, energy, entropy, meals) per block in [lo, hi)."""
    t = arr["t"]
    out = []
    start = lo
    while start < hi:
        stop = min(start + BLOCK, hi)
        mask = (t >= start) & (t < stop)
        if mask.sum() == 0:
            start = stop
            continue
        energy = arr["true_energy_t"][mask]
        meals = int((np.diff(energy) > CONSUMPTION_JUMP).sum())
        out.append(
            (
                f"{start}-{stop}",
                float(arr["recon_loss_t"][mask].mean()),
                float(arr["intrinsic_signal_t"][mask].mean()),
                float(energy.mean()),
                float(arr["policy_entropy_t"][mask].mean()),
                meals,
            )
        )
        start = stop
    return out


def main(run_dir: Path, boundary: int | None) -> int:
    telemetry = run_dir / "telemetry"
    if boundary is None:
        boundary = _latest_resume_marker(telemetry)
    if boundary is None:
        print("no boundary given and no resume marker found")
        return 1
    arr = _load(telemetry)
    t = arr["t"]
    t_min, t_max = int(t.min()), int(t.max()) + 1
    window = 5 * BLOCK
    print(
        f"boundary t={boundary}  (record spans {t_min}..{t_max - 1}; "
        f"±{window} steps shown, {BLOCK}-step blocks)"
    )
    positions = _positions(run_dir)
    centers = _patch_centers(telemetry)
    show_occupancy = bool(positions) and bool(centers)
    header = (
        f"{'block':>16} {'pred-err':>9} {'curiosity':>10} {'energy':>7} "
        f"{'act-ent':>8} {'meals':>6}"
    )
    if show_occupancy:
        header += f" {'in-patch':>9}"
    for side, lo, hi in (
        ("BEFORE", max(t_min, boundary - window), boundary),
        ("AFTER", boundary, min(t_max, boundary + window)),
    ):
        print(f"--- {side} ---")
        print(header)
        for label, pe, ig, en, ent, meals in _block_rows(arr, lo, hi):
            row = (
                f"{label:>16} {pe:>9.2f} {ig:>10.4f} {en:>7.3f} "
                f"{ent:>8.4f} {meals:>6}"
            )
            if show_occupancy:
                block_lo, block_hi = (int(s) for s in label.split("-"))
                share = _occupancy_share(
                    positions, centers, block_lo, block_hi
                )
                row += f" {share:>9.2f}" if share is not None else f" {'—':>9}"
            print(row)
    return 0


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    boundary_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
    sys.exit(main(run_dir, boundary_arg))
