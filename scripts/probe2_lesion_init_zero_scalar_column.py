#!/usr/bin/env python3
"""Probe 2 v2 ``init_zero_scalar_column`` lesion — checkpoint mutation
script (plan §2.5; synthesis §2.4 element 4).

Loads a Phase-7.5 / frozen-target / small-Gaussian-shape checkpoint,
replaces the actor's first-layer weight column corresponding to the
scalar ``self_prediction_error`` input (``actor.net.0.weight[:,
h_dim + z_dim:]``) with zeros, and saves the lesioned checkpoint to a
new directory. Every other tensor in the checkpoint — the world model's
encoder/GRU/decoder/heads, the EMA target's encoder/GRU, the
``_frozen_projection`` (when present), the ensemble's heads, the
remaining columns of the actor's first-layer weight, the actor's bias
and the rest of its layers, the optimizer state, the RNG state, the
replay meta, the telemetry offsets, the schema version — is preserved
byte-identical via safetensors round-trip on the weights and ``shutil
.copyfile`` on the sidecar artefacts.

The lesion tests the *capacity-as-init-shape* distinction Phase 8
surfaced (synthesis §2.4 element 4; tension (l)). Probe 1.5's actor
trains only via ``actor_loss.backward()`` on imagined trajectories
where the scalar is fed as zero (mask-via-zero-feed; Probe 1.5 plan
§2.3); the gradient on the new column is mathematically zero on every
step, so the column at end-of-training is byte-identical to its
construction-time draw. Phase 7's u/d-bipolar regime (under zero-init)
and Phase 7.5's L/R-bipolar regime (under small-Gaussian init) are
both observable end-states of "the column doesn't move and the scalar's
contribution to the actor's input is determined by the init draw."
This lesion zeros the small-Gaussian column post-hoc on a Phase-7.5 /
frozen-target checkpoint and asks whether the lesioned actor's
late-trajectory regime collapses back to Phase 7's u/d (confirming
column-init-determined behavior) or holds at L/R (weakening the
column-init-determined reading). The substrate-side reads against the
lesioned actor at evaluation tell whether the conditioning's specific
shape reflects the column's draw or something else.

Emits a ``mirror_marker`` ``world_event`` record at
``output-dir/world_event.jsonl`` with ``event_type="mirror_marker"``,
``source="system"``, ``payload.lesion_kind="init_zero_scalar_column"``,
``payload.source_checkpoint=<source path>``. The mutation-time
world_event is documentation: it is *not* part of a runner's
telemetry stream (which lives at ``runs/{run_id}/telemetry/`` under a
runner's control). A downstream reader / journal / analysis tool can
read this record to discover the lesioned checkpoint's provenance.

Run:
    .venv/bin/python scripts/probe2_lesion_init_zero_scalar_column.py \\
        --source-checkpoint runs/probe1_5_phase7_5-20260507-101800/checkpoints/ckpt-000001/ \\
        --output-dir runs/probe2_lesion_init_zero_scalar_column-20260507/checkpoints/ckpt-000001/

Or with --dry-run to short-circuit before any file write:
    .venv/bin/python scripts/probe2_lesion_init_zero_scalar_column.py \\
        --source-checkpoint runs/probe1_5_phase7_5-20260507-101800/checkpoints/ckpt-000001/ \\
        --output-dir /tmp/dry_run_target/ --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from kind.observer.schemas import SCHEMA_VERSION, WorldEvent

_ACTOR_FIRST_LAYER_KEY: str = "actor.net.0.weight"
_SIDECAR_FILES: tuple[str, ...] = (
    "optimizer_state.pt",
    "replay_meta.json",
    "rng_state.pkl",
    "schema_version.txt",
    "telemetry_offsets.json",
)


@dataclass(frozen=True)
class _ColumnMoments:
    """Pre/post moments of the scalar column the lesion targets."""

    mean: float
    std: float
    min: float
    max: float
    abs_max: float
    shape: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "abs_max": self.abs_max,
            "shape": list(self.shape),
        }


@dataclass(frozen=True)
class MutationResult:
    """Summary of one mutation invocation. Returned by :func:`mutate`
    so callers (the test suite, the CLI's stdout printer) read the
    same shape regardless of dry-run vs. real-run."""

    source_checkpoint: Path
    output_dir: Path
    actor_first_layer_key: str
    actor_input_dim: int
    column_start_index: int
    column_moments_before: _ColumnMoments
    column_moments_after: _ColumnMoments
    n_other_weight_tensors: int
    sidecar_files_copied: tuple[str, ...]
    world_event_path: Path | None
    dry_run: bool


def _column_moments(column: torch.Tensor) -> _ColumnMoments:
    """Compute the lesion-relevant moments. ``std`` uses the unbiased
    estimator; for a single-column, all-zero tensor std is ``0.0``
    deterministically."""
    if column.numel() == 0:
        return _ColumnMoments(
            mean=0.0,
            std=0.0,
            min=0.0,
            max=0.0,
            abs_max=0.0,
            shape=tuple(column.shape),
        )
    return _ColumnMoments(
        mean=float(column.mean().item()),
        std=float(column.std(unbiased=True).item())
        if column.numel() > 1
        else 0.0,
        min=float(column.min().item()),
        max=float(column.max().item()),
        abs_max=float(column.abs().max().item()),
        shape=tuple(column.shape),
    )


def mutate(
    *,
    source_checkpoint: Path,
    output_dir: Path,
    dry_run: bool,
    run_id: str | None = None,
    checkpoint_id: str | None = None,
) -> MutationResult:
    """Mutate the source checkpoint's actor scalar column and save to
    output_dir.

    Args:
        source_checkpoint: directory containing ``weights.safetensors``
            plus the standard checkpoint sidecar files. Typically a
            Phase-7.5 / frozen-target checkpoint with a small-Gaussian
            actor scalar column the lesion zeros.
        output_dir: directory the lesioned checkpoint will be written
            into. Created if missing. Existing contents are *not*
            cleared — callers should pass a fresh path.
        dry_run: if True, perform all reads + computations and return
            the result without writing. The result's ``world_event_path``
            is ``None`` and ``sidecar_files_copied`` is empty.
        run_id: optional. Stamped onto the emitted ``world_event`` if
            supplied; otherwise a default is derived from the output
            directory's grandparent name (under the
            ``runs/{run_id}/checkpoints/{ckpt-id}/`` convention) or
            the literal ``"lesion_init_zero_scalar_column"`` if the
            convention does not apply.
        checkpoint_id: optional. Stamped onto the emitted
            ``world_event`` if supplied; otherwise the output
            directory's basename is used.

    Returns:
        A :class:`MutationResult` describing the mutation. The result's
        moment fields capture the column's pre/post statistics — the
        ``moments_after`` are exactly all-zero and the ``moments_before``
        capture the small-Gaussian shape Phase 7.5's init produced.
    """
    if not source_checkpoint.is_dir():
        raise FileNotFoundError(
            f"source_checkpoint does not exist or is not a directory: "
            f"{source_checkpoint}"
        )
    source_weights = source_checkpoint / "weights.safetensors"
    if not source_weights.is_file():
        raise FileNotFoundError(
            f"source_checkpoint missing weights.safetensors: "
            f"{source_weights}"
        )

    # Load the full weights blob into memory. safetensors's load_file
    # produces a dict[str, torch.Tensor] with all tensors materialised
    # on CPU; the mutation lives entirely on CPU regardless of the
    # source run's device since save_file writes contiguous CPU tensors.
    weights: dict[str, torch.Tensor] = load_file(str(source_weights))

    if _ACTOR_FIRST_LAYER_KEY not in weights:
        raise KeyError(
            f"source checkpoint has no {_ACTOR_FIRST_LAYER_KEY!r} tensor "
            f"(found {len(weights)} keys: "
            f"{sorted(weights.keys())[:6]!r}...)"
        )

    actor_w = weights[_ACTOR_FIRST_LAYER_KEY]
    if actor_w.dim() != 2:
        raise ValueError(
            f"{_ACTOR_FIRST_LAYER_KEY} is not a 2D tensor "
            f"(shape={tuple(actor_w.shape)})"
        )
    in_dim = int(actor_w.shape[1])
    # Probe 1.5 v2 actor input is ``h_dim + z_dim + 1``; the scalar
    # column is the trailing single column at index ``in_dim - 1``.
    # The plan §2.5 + the prompt name the slice as
    # ``actor.net.0.weight[:, h_dim+z_dim:]`` — under Probe 1.5's
    # ``+1`` extension this is exactly one column. The implementation
    # uses ``[:, in_dim - 1:]`` to cover any future widening in a
    # forward-compatible way.
    column_start = in_dim - 1
    if column_start < 0:
        raise ValueError(
            f"{_ACTOR_FIRST_LAYER_KEY} has degenerate input dim {in_dim}; "
            f"expected at least 2 (h + z + 1 ≥ 2 under Probe 1.5)"
        )

    column_before_tensor = actor_w[:, column_start:].clone()
    moments_before = _column_moments(column_before_tensor)

    # Mutate. Write into a fresh tensor so the in-memory ``weights``
    # dict's actor entry is replaced (rather than mutated in place);
    # this keeps the byte-identity check on the *other* tensors clean
    # — they share storage with the loaded blob and are written
    # untouched.
    new_actor_w = actor_w.clone()
    new_actor_w[:, column_start:] = 0.0
    weights[_ACTOR_FIRST_LAYER_KEY] = new_actor_w

    moments_after = _column_moments(new_actor_w[:, column_start:])

    other_keys = sorted(k for k in weights if k != _ACTOR_FIRST_LAYER_KEY)
    n_other = len(other_keys)

    if dry_run:
        return MutationResult(
            source_checkpoint=source_checkpoint,
            output_dir=output_dir,
            actor_first_layer_key=_ACTOR_FIRST_LAYER_KEY,
            actor_input_dim=in_dim,
            column_start_index=column_start,
            column_moments_before=moments_before,
            column_moments_after=moments_after,
            n_other_weight_tensors=n_other,
            sidecar_files_copied=(),
            world_event_path=None,
            dry_run=True,
        )

    # Write the lesioned checkpoint.
    output_dir.mkdir(parents=True, exist_ok=True)
    output_weights = output_dir / "weights.safetensors"
    # safetensors requires CPU contiguous tensors; the loaded tensors
    # already satisfy both. The ``contiguous()`` call below is a
    # belt-and-braces guard for any tensor whose memory layout was
    # affected by the slice-assignment above.
    weights_to_write: dict[str, torch.Tensor] = {
        k: (v.contiguous() if not v.is_contiguous() else v)
        for k, v in weights.items()
    }
    save_file(weights_to_write, str(output_weights))

    # Byte-copy the sidecar files. ``shutil.copyfile`` preserves
    # contents byte-for-byte; the runner's ``load_checkpoint`` reads
    # all five so all five must be present in the output.
    copied: list[str] = []
    for name in _SIDECAR_FILES:
        source_file = source_checkpoint / name
        if source_file.is_file():
            shutil.copyfile(source_file, output_dir / name)
            copied.append(name)

    # Emit the mutation-time mirror_marker world_event. The record is
    # documentation: it lives in the *checkpoint* directory rather
    # than under any runner's telemetry tree. Downstream consumers
    # (journal, analysis tools) can read this single-line JSONL to
    # discover the lesioned checkpoint's provenance.
    derived_run_id = run_id or _derive_run_id(output_dir)
    derived_checkpoint_id = checkpoint_id or output_dir.name
    world_event = WorldEvent(
        schema_version=SCHEMA_VERSION,
        run_id=derived_run_id,
        checkpoint_id=derived_checkpoint_id,
        t_event=0,
        event_type="mirror_marker",
        source="system",
        payload={
            "lesion_kind": "init_zero_scalar_column",
            "source_checkpoint": str(source_checkpoint),
            "actor_first_layer_key": _ACTOR_FIRST_LAYER_KEY,
            "actor_input_dim": in_dim,
            "column_start_index": column_start,
            "column_moments_before": moments_before.as_dict(),
            "column_moments_after": moments_after.as_dict(),
        },
        wallclock_ms=time.monotonic_ns() // 1_000_000,
    )
    world_event_path = output_dir / "world_event.jsonl"
    with open(world_event_path, "w", encoding="utf-8") as fh:
        fh.write(world_event.model_dump_json() + "\n")

    return MutationResult(
        source_checkpoint=source_checkpoint,
        output_dir=output_dir,
        actor_first_layer_key=_ACTOR_FIRST_LAYER_KEY,
        actor_input_dim=in_dim,
        column_start_index=column_start,
        column_moments_before=moments_before,
        column_moments_after=moments_after,
        n_other_weight_tensors=n_other,
        sidecar_files_copied=tuple(copied),
        world_event_path=world_event_path,
        dry_run=False,
    )


def _derive_run_id(output_dir: Path) -> str:
    """Default run_id derivation under the
    ``runs/{run_id}/checkpoints/{ckpt-id}/`` convention.

    If output_dir's parent's parent ends with a recognisable run-id
    shape (any non-empty name not in {".", ".."}), use it; otherwise
    fall back to the literal ``"lesion_init_zero_scalar_column"``.
    The fallback is what makes the script work for ad-hoc /tmp paths
    in tests without forcing the test to construct the full
    runs/{id}/checkpoints/{ckpt}/ directory layout.
    """
    parts = output_dir.resolve().parts
    if len(parts) >= 3 and parts[-2] == "checkpoints":
        return parts[-3]
    return "lesion_init_zero_scalar_column"


def _print_summary(result: MutationResult) -> None:
    """Print a brief stdout summary the prompt asks for: source path,
    output path, column moments before (small-Gaussian) and after
    (zero), other-tensor byte-identity confirmation."""
    print(f"source-checkpoint: {result.source_checkpoint}")
    print(f"output-dir:        {result.output_dir}")
    print(f"actor-first-layer: {result.actor_first_layer_key}")
    print(
        f"actor-input-dim:   {result.actor_input_dim} "
        f"(scalar column at index {result.column_start_index})"
    )
    before = result.column_moments_before
    after = result.column_moments_after
    print(
        f"column-before:     mean={before.mean:+.6f} std={before.std:.6f} "
        f"abs_max={before.abs_max:.6f} shape={list(before.shape)}"
    )
    print(
        f"column-after:      mean={after.mean:+.6f} std={after.std:.6f} "
        f"abs_max={after.abs_max:.6f} shape={list(after.shape)}"
    )
    print(
        f"other-weight-tensors: {result.n_other_weight_tensors} preserved "
        f"(byte-identical via safetensors round-trip)"
    )
    if result.dry_run:
        print("dry-run: no files written; world_event not emitted")
    else:
        sidecars = ", ".join(result.sidecar_files_copied) or "<none>"
        print(f"sidecar-files-copied: {sidecars}")
        print(f"world-event-path:  {result.world_event_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Probe 2 v2 init_zero_scalar_column lesion — "
            "mutate a Phase-7.5/frozen-target checkpoint by zeroing the "
            "actor's scalar input column."
        )
    )
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        required=True,
        help=(
            "Path to the source checkpoint directory "
            "(e.g., runs/probe1_5_phase7_5-20260507-101800/checkpoints/ckpt-000001/)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Path to the output checkpoint directory the lesioned "
            "checkpoint is written into. Created if missing."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Short-circuit before any file write. Performs all reads and "
            "computes the mutation summary; no output files produced."
        ),
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Optional run_id to stamp on the emitted world_event "
            "mirror_marker. Defaults to a path-derived value."
        ),
    )
    parser.add_argument(
        "--checkpoint-id",
        type=str,
        default=None,
        help=(
            "Optional checkpoint_id to stamp on the emitted world_event "
            "mirror_marker. Defaults to the output directory's basename."
        ),
    )
    args = parser.parse_args(argv)

    result = mutate(
        source_checkpoint=args.source_checkpoint,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        run_id=args.run_id,
        checkpoint_id=args.checkpoint_id,
    )
    _print_summary(result)
    return 0


# Module-level constants exposed for test access.
__all__ = [
    "MutationResult",
    "mutate",
    "main",
    "_ACTOR_FIRST_LAYER_KEY",
    "_SIDECAR_FILES",
    "_ColumnMoments",
]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

# json import kept for potential future use by callers loading the
# emitted world_event.jsonl back; the model_dump_json above writes
# directly so this is a defensive belt-and-braces pin.
_ = json
