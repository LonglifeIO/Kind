"""Probe 3 Phase 7 — minimal builder-perturbation entry point (surface-test only).

**Deliberately minimal.** This script exercises the authorized cross-probe
surface (``kind.training.cross_probe_surface``) end-to-end: it reads a JSON
config delta restricted to dream-envelope + seed-selection fields, *stages* it
(parse + validate — where the restriction trips), and *applies* it at a
checkpoint boundary, producing new frozen configs whose snapshots are what the
next ``DreamSessionMeta`` would record as provenance.

It is **not** Probe 4 scheduler scaffolding. It does not schedule, does not loop,
does not fire a ``builder_perturbation`` ``world_event`` (that type stays
reserved for Probe 4's own perturbation workflow), and — structurally — cannot
write into Io's hidden state (the typed delta in
``kind.training.cross_probe_surface`` has no field that addresses Io's interior,
and forbids extra keys, so a hidden-state delta is rejected at parse time).
Probe 4 may replace this script after its own synthesis.

The live runner-loop consumption of the perturbed configs is Phase 8; here the
delta is applied onto the *default* configs (or a base-config JSON via
``--base``) purely to demonstrate the surface round-trip.

Usage::

    python scripts/perturb_dream_envelope.py --delta delta.json
    python scripts/perturb_dream_envelope.py --delta delta.json --out new_configs.json

where ``delta.json`` is, e.g.::

    {
      "dream_envelope": {"hard_cap_rollout_count": 30},
      "seed_selection": {"mode": "hybrid", "perturbation_sigma": 0.2}
    }

A delta carrying any other key (``h``, ``z``, ``latents``, ``weights``,
``actor``, ...) is rejected with a validation error — by design.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from kind.training.cross_probe_surface import (
    PerturbedConfigs,
    apply_at_checkpoint_boundary,
    stage_perturbation,
)
from kind.training.dream_seed import SeedSelectionConfig
from kind.training.state_machine import DreamEnvelopeConfig


def _snapshots(configs: PerturbedConfigs) -> dict[str, dict[str, object]]:
    """The provenance snapshots the next ``DreamSessionMeta`` would record."""
    return {
        "envelope_config_snapshot": dataclasses.asdict(configs.dream_envelope),
        "seed_selection_config_snapshot": dataclasses.asdict(configs.seed_selection),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a builder dream-envelope / seed-selection perturbation at a "
            "checkpoint boundary (surface-test only; no builder_perturbation "
            "event is fired, no hidden-state write is possible)."
        )
    )
    parser.add_argument(
        "--delta",
        required=True,
        type=Path,
        help=(
            "path to a JSON config delta with optional 'dream_envelope' and "
            "'seed_selection' objects; any other key is rejected."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "optional path to write the resulting config snapshots as JSON "
            "(the next DreamSessionMeta's provenance); prints to stdout if omitted."
        ),
    )
    args = parser.parse_args(argv)

    delta_path: Path = args.delta
    if not delta_path.is_file():
        print(f"perturb_dream_envelope: delta not found: {delta_path}", file=sys.stderr)
        return 1

    # Stage: parse + validate. The restriction trips here — a delta carrying a
    # hidden-state key (or an extra key, or an invalid enum) is rejected.
    try:
        perturbation = stage_perturbation(delta_path.read_bytes())
    except ValidationError as exc:
        print(
            "perturb_dream_envelope: delta rejected — only dream-envelope and "
            "seed-selection fields are addressable (no hidden-state writes):\n"
            f"{exc}",
            file=sys.stderr,
        )
        return 2

    # Apply at a checkpoint boundary onto the current (here: default) configs.
    # Staged, not live: the inputs are not mutated; new frozen configs are
    # produced for the next session to start from.
    current = PerturbedConfigs(
        dream_envelope=DreamEnvelopeConfig(),
        seed_selection=SeedSelectionConfig(),
    )
    result = apply_at_checkpoint_boundary(
        perturbation,
        dream_envelope=current.dream_envelope,
        seed_selection=current.seed_selection,
    )

    snapshots = _snapshots(result)
    payload = json.dumps(snapshots, indent=2, sort_keys=True)
    if args.out is not None:
        out_path: Path = args.out
        out_path.write_text(payload + "\n", encoding="utf-8")
        print(
            f"perturb_dream_envelope: applied; wrote new config snapshots to "
            f"{out_path}",
            file=sys.stderr,
        )
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
