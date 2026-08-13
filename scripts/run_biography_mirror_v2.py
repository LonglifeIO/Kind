"""Run one windowed V2 adversarial mirror pass against the biography.

The first criteria-driven reading of Io's continuing life (decision doc
``docs/decisions/mirror_v2_windowed_mode_2026-07-30.md``): the last
``--window-steps`` env steps (default 5,000 — the instrument's
calibrated geometry, as 200-step analysis chunks), the frozen V2
criteria (active: reflexive attention, equanimity; held-out:
second-order volition), primary + adversarial readings for each
partition. Four LLM calls total (gemini-2.5-pro; ``GEMINI_API_KEY``
from the project ``.env``).

Builder-gated: this script makes real API calls. Outputs land under
``runs/<run>/mirror/`` (pre_reg/, passes/) — the one-way mirror-side
invariant; nothing Io can read is touched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from kind.mirror.criteria_v2 import V2_REGISTRY
from kind.mirror.llm_caller import LLMConfig
from kind.mirror.orchestrator import PassConfig, run_adversarial_pass
from kind.mirror.registry import CriterionRegistry
from kind.mirror.statistics import StatisticConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One windowed V2 adversarial mirror pass."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/probe4_phase4_biography"),
    )
    parser.add_argument("--run-id", default="probe4-phase4-biography")
    parser.add_argument(
        "--checkpoint-id",
        required=True,
        help="The checkpoint this pass evaluates (e.g. ckpt-000045).",
    )
    parser.add_argument("--window-steps", type=int, default=5000)
    parser.add_argument(
        "--window-end-t",
        type=int,
        default=None,
        help=(
            "Anchor the window's end at this env step instead of the "
            "telemetry tail (reads [end - window_steps + 1, end]). Lets "
            "a pass read a historical moment after later sessions have "
            "appended shards."
        ),
    )
    parser.add_argument(
        "--chunk-steps",
        type=int,
        default=200,
        help="Analysis-slice length (default 200 = calibrated geometry).",
    )
    args = parser.parse_args()

    load_dotenv()

    config = PassConfig(
        run_id=args.run_id,
        checkpoint_id=args.checkpoint_id,
        run_dir=args.run_dir,
        active_registry=CriterionRegistry(criteria=V2_REGISTRY.active()),
        held_out_registry=CriterionRegistry(criteria=V2_REGISTRY.held_out()),
        statistic_config=StatisticConfig(chunk_steps=args.chunk_steps),
        llm_config=LLMConfig(),
        column_init="small_gaussian",
        window_steps=args.window_steps,
        window_end_t=args.window_end_t,
    )
    result = run_adversarial_pass(config)

    print(f"pass complete: run={args.run_id} ckpt={args.checkpoint_id}")
    print(f"notes:\n{result.notes}")
    for label, readings in (
        ("active primary", result.active_primary_readings),
        ("active adversarial", result.active_adversarial_readings),
        ("held-out primary", result.held_out_primary_readings),
        ("held-out adversarial", result.held_out_adversarial_readings),
    ):
        n_claims = sum(len(r.claims) for r in readings)
        print(f"  {label}: {len(readings)} reading(s), {n_claims} claim(s)")
    print(f"statistics computed: {len(result.statistic_results)}")
    print(f"written under: {args.run_dir / 'mirror'}")


if __name__ == "__main__":
    main()
