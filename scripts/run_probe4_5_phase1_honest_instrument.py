"""Probe 4.5 Phase 1 — the honest instrument: fresh-instance burn-in with the
scheduled maintenance cycle live.

The Phase-1 *gate* run, not a unit test (plan §Phase 1). It trains a fresh
Probe 4.5 instance (F2 bounded head ON per the adopted decision doc; energy
telemetry on; ``energy_preference=None`` — no preference exists yet to
confound the instrument) with the S-DEC maintenance harness engaged at the
frozen prereg §3 cadence: a decoder-head-only refit + honesty gate at every
10k env steps, checkpoint-aligned. The phase question: **can the live decoder
head be made and kept honest against the frozen physics margins on a fresh
training instance?**

Gate: the §3 margins hold at every scheduled refit across the burn-in. A
failure follows the frozen STOP rule (one diagnostic re-collection, then
honesty-STOP) — the script records the finding and exits nonzero; the
margins are never revised against behavior.

When ``--steps`` is a multiple of the refit cadence (the default 20k is),
the final scheduled refit coincides with the prereg's burn-in-close refit
and is recorded as such; otherwise an explicit burn-in-close cycle runs
after the loop.

Usage::

    python scripts/run_probe4_5_phase1_honest_instrument.py \
        [--steps 20000] [--seed 4501] [--device mps] \
        [--run-dir runs/probe4_5_phase1]

Prints a JSON summary; the journal entry is written by hand from it and the
per-refit reports under ``<run-dir>/maintenance/``.
"""

from __future__ import annotations

import argparse
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from kind.agents.world_model import WorldModelConfig
from kind.env.env_server import EnvServer, EnvServerConfig
from kind.env.grid_world import GridWorldConfig
from kind.env.transport import EnvTransportClient, EnvTransportServer
from kind.observer.maintenance_refit import (
    HonestyStopError,
    MaintenanceRefitConfig,
    run_scheduled_maintenance,
)
from kind.training.runner import Runner, RunnerConfig
from kind.window.live import LiveStateWriter


@contextmanager
def _transport_pair(
    grid_cfg: GridWorldConfig, seed: int, run_id: str
) -> Iterator[tuple[EnvTransportClient, EnvServer]]:
    config = EnvServerConfig(
        grid_world_config=grid_cfg,
        seed=seed,
        world_event_handler=lambda _r: None,
        run_id=run_id,
    )
    env_server = EnvServer(config)
    transport_server = EnvTransportServer(env_server, host="127.0.0.1", port=0)
    server_thread = threading.Thread(
        target=transport_server.serve_forever,
        name="Probe45Phase1EnvTransportServer",
        daemon=True,
    )
    server_thread.start()
    client = EnvTransportClient(
        host="127.0.0.1",
        port=transport_server.actual_port,
        world_event_handler=lambda _r: None,
    )
    try:
        yield client, env_server
    finally:
        try:
            client.close()
        finally:
            transport_server.shutdown()
            server_thread.join(timeout=5.0)


def _refit_summaries(maintenance_dir: Path) -> list[dict[str, Any]]:
    """Condense every machine-written refit report to its gate-relevant core."""
    summaries: list[dict[str, Any]] = []
    for path in sorted(maintenance_dir.glob("refit_*.json")):
        report = json.loads(path.read_text())
        after_pooled = report["after"]["pooled"]
        summaries.append(
            {
                "file": path.name,
                "env_step": report["env_step"],
                "occasion": report["occasion"],
                "all_passed": report["verdict"]["all_passed"],
                "checks": {
                    c["name"]: {"value": c["value"], "passed": c["passed"]}
                    for c in report["verdict"]["checks"]
                },
                "coverage_skipped_regions": report["verdict"][
                    "coverage_skipped_regions"
                ],
                # The three-way error comparison trend (does the head keep
                # discarding the honest sensor?) — the journal deliverable.
                "pooled_three_way": {
                    "decode_vs_true": after_pooled[
                        "decode_vs_true_abs_error_mean"
                    ],
                    "decode_vs_sensed": after_pooled[
                        "decode_vs_sensed_abs_error_mean"
                    ],
                    "sensed_vs_true": after_pooled[
                        "sensed_vs_true_abs_error_mean"
                    ],
                },
                "pooled_out_of_range_mass": after_pooled["out_of_range_mass"],
                "refit_mse": {
                    "first_epoch": report["refit"]["first_epoch_mse"],
                    "final_epoch": report["refit"]["final_epoch_mse"],
                },
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20_000)
    # Phase-1 validation-instance seed — a build-time choice, journaled; NOT
    # the prereg §5 world/train seed 20260713, which is reserved for the
    # Phase-4 arms (fresh instances; nothing carries forward from this run).
    parser.add_argument("--seed", type=int, default=4501)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument(
        "--run-dir", type=str, default="runs/probe4_5_phase1"
    )
    parser.add_argument(
        "--refit-every", type=int, default=10_000, help="prereg §3 cadence"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "wiring/device smoke: shrink the maintenance collection protocol "
            "and loosen the margins so the run continues past the refit "
            "(exercising the refit → restore → keep-training path on the "
            "real device). NOT a gate run: gate runs use the frozen §3 "
            "margins; smoke success is the plumbing completing."
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = Path(args.run_dir)
    maintenance_dir = run_dir / "maintenance"
    grid_cfg = GridWorldConfig()  # live default physics (no faults: Phase 2)

    if args.smoke:
        from kind.observer.maintenance_refit import HonestyMargins

        maintenance = MaintenanceRefitConfig(
            grid_world_config=grid_cfg,
            out_dir=maintenance_dir,
            refit_every_n_env_steps=args.refit_every,
            margins=HonestyMargins(
                oracle_in_band_abs_error_max=1e6,
                pooled_slope_min=-1e6,
                pooled_out_of_range_mass_max=1.0,
                oracle_region_abs_bias_max=1e6,
                region_min_n=1,
            ),
            train_seeds_per_source=1,
            train_episodes_per_seed=1,
            diagnostic_seeds_per_source=2,
            refit_epochs=2,
            honesty_seeds_per_source=1,
            honesty_episodes_per_seed=1,
        )
    else:
        maintenance = MaintenanceRefitConfig(
            grid_world_config=grid_cfg,
            out_dir=maintenance_dir,
            refit_every_n_env_steps=args.refit_every,
            # margins / mixture / hyperparameters: the frozen §3 defaults.
        )
    run_cfg = RunnerConfig(
        # F2 bounded head ON — a Probe 4.5 instance, trained through the
        # sigmoid from scratch (decision doc: fresh instances only).
        world_model_config=WorldModelConfig(energy_decoder_bounded=True),
        run_id="probe4_5_phase1",
        telemetry_dir=run_dir / "telemetry",
        checkpoints_dir=run_dir / "checkpoints",
        checkpoint_every_n_env_steps=args.refit_every,
        energy_telemetry=True,
        energy_preference=None,  # no preference exists to confound Phase 1
        maintenance_refit=maintenance,
        device=args.device,
    )

    summary: dict[str, Any] = {
        "LABEL": (
            "Probe 4.5 Phase 1 — honest instrument burn-in. Gate: the frozen "
            "prereg §3 margins hold at every scheduled refit."
        ),
        "steps": args.steps,
        "seed": args.seed,
        "device": args.device,
        "refit_every_n_env_steps": args.refit_every,
        "f2_bounded_head": True,
    }

    stop: HonestyStopError | None = None
    with _transport_pair(grid_cfg, seed=args.seed, run_id="probe4_5_phase1") as (
        client,
        env_server,
    ):
        # Builder's eye: the Window's /live page (run
        # ``scripts/run_window.py --run-id probe4_5_phase1``).
        live_writer = LiveStateWriter(
            env_server, run_dir, run_id=run_cfg.run_id
        )
        runner = Runner(
            run_cfg, client, env_server=env_server, step_callback=live_writer
        )
        try:
            runner.run(total_env_steps=args.steps)
            # The runner's loop covers env steps 0..steps-1, so a boundary
            # at exactly ``steps`` never fires in-loop — the prereg's
            # burn-in-close refit therefore ALWAYS runs explicitly here
            # (attempt-1 lesson, 2026-07-18: the old "coincides with the
            # final scheduled refit" branch was an off-by-one and the close
            # refit never ran). A final checkpoint commits afterward so the
            # honest post-close decoder persists for the Phase-2 profile.
            run_scheduled_maintenance(
                runner._world_model,  # noqa: SLF001 — pause point; loop over
                runner._actor,  # noqa: SLF001
                maintenance,
                env_step=args.steps,
                occasion="burn_in_close",
                run_label=run_cfg.run_id,
            )
            runner._commit_checkpoint(args.steps)  # noqa: SLF001
        except HonestyStopError as e:
            stop = e
        finally:
            runner.close()

    summary["refits"] = _refit_summaries(maintenance_dir)
    summary["honesty_stop"] = None if stop is None else str(stop)
    summary["gate_passed"] = stop is None and all(
        r["all_passed"] for r in summary["refits"]
    )

    out_path = run_dir / "phase1_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\n[out] wrote {out_path}")
    if not summary["gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
