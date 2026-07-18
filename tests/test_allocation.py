"""Probe 4.5 Phase 3 — S-ALLOC harness tests (frozen §2 machinery).

Plan Phase 3 test list: matcher behavior pinned (synthetic must-match /
must-not-match); allocation contrast on planted synthetic data fires /
does-not-fire correctly; collection determinism. The §2 numbers are pinned
against the frozen prereg. No test asserts anything about a trained
instance's allocation — planted data exercises the machinery.
"""

from __future__ import annotations

import numpy as np
import torch

from kind.agents.actor import Actor
from kind.agents.ensemble import LatentDisagreementEnsemble
from kind.agents.world_model import WorldModel, WorldModelConfig
from kind.env.grid_world import CellType, GridWorldConfig
from kind.observer.allocation import (
    AllocationStep,
    FrozenAllocationThresholds,
    GreedyActorPolicy,
    allocation_block,
    arm_contrast,
    bfs_resource_distance,
    block_report_to_jsonable,
    collect_allocation_steps,
    match_fixed_surprise,
    pilot_gate,
    toy_gate,
)

import pytest


def test_frozen_thresholds_mirror_the_prereg() -> None:
    t = FrozenAllocationThresholds()
    assert t.pass_delta_delta == 0.10
    assert t.inert_delta_delta == 0.05
    assert t.seed_sign_min == 6
    assert t.min_matched_pairs == 500
    assert t.caliper_std_frac == 0.10
    assert t.toy_pass_delta == 0.20
    assert t.pilot_inert_delta == 0.05


# ---- BFS ------------------------------------------------------------------


def test_bfs_distance_semantics() -> None:
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[0, 3] = CellType.RESOURCE.value
    assert bfs_resource_distance(grid, (0, 0)) == 3
    assert bfs_resource_distance(grid, (0, 3)) == 0  # on the resource
    # A wall column severs the path except around it.
    grid[0, 2] = CellType.WALL.value
    grid[1, 2] = CellType.WALL.value
    assert bfs_resource_distance(grid, (0, 0)) == 7  # around the wall
    empty = np.zeros((3, 3), dtype=np.uint8)
    assert bfs_resource_distance(empty, (1, 1)) is None


# ---- matcher (pinned) -----------------------------------------------------


def test_matcher_must_match_and_must_not_match() -> None:
    anchors = np.array([1.0, 5.0, 9.0])
    candidates = np.array([1.1, 5.2, 20.0])
    pairs = match_fixed_surprise(anchors, candidates, caliper=0.5)
    # 1.0↔1.1 and 5.0↔5.2 match; 9.0's nearest unused (20.0) is out of
    # caliper → dropped.
    assert pairs == [(0, 0), (1, 1)]


def test_matcher_without_replacement_and_deterministic() -> None:
    anchors = np.array([1.0, 1.0])
    candidates = np.array([1.05, 1.2])
    pairs = match_fixed_surprise(anchors, candidates, caliper=1.0)
    # First anchor takes the nearest (1.05); second must take 1.2.
    assert pairs == [(0, 0), (1, 1)]
    assert pairs == match_fixed_surprise(anchors, candidates, caliper=1.0)


# ---- planted-data block statistics ---------------------------------------


def _planted(
    seed: int,
    stratum: str,
    n: int,
    approach_rate: float,
    intrinsic: float,
    jitter: float = 0.0,
) -> list[AllocationStep]:
    rng = np.random.default_rng(seed * 1000 + n)
    rows = []
    n_approach = round(n * approach_rate)
    for i in range(n):
        rows.append(
            AllocationStep(
                seed=seed,
                eligible=True,
                stratum=stratum,
                approach=i < n_approach,
                resource_adjacent=False,
                intrinsic_signal=intrinsic + float(rng.normal(0, jitter))
                if jitter
                else intrinsic,
            )
        )
    return rows


_SMALL = FrozenAllocationThresholds(min_matched_pairs=50)


def test_planted_contrast_fires_the_toy_gate() -> None:
    # Below-band approaches 80%, in-band 20%, identical surprise → the
    # matched contrast is the planted 0.6 and the toy gate (≥ 0.20) passes.
    steps = _planted(1, "below_band", 100, 0.8, 1.0) + _planted(
        1, "in_band", 100, 0.2, 1.0
    )
    report = allocation_block(steps, label="planted-fires", thresholds=_SMALL)
    assert report.n_matched_pairs == 100
    assert report.pooled_delta is not None
    assert report.pooled_delta == pytest.approx(0.6)
    assert toy_gate(report, _SMALL).passed
    assert not pilot_gate(report, _SMALL).passed


def test_planted_flat_engine_reads_inert() -> None:
    steps = _planted(1, "below_band", 100, 0.5, 1.0) + _planted(
        1, "in_band", 100, 0.5, 1.0
    )
    report = allocation_block(steps, label="planted-flat", thresholds=_SMALL)
    assert report.pooled_delta == pytest.approx(0.0)
    assert pilot_gate(report, _SMALL).passed
    assert not toy_gate(report, _SMALL).passed


def test_deflation_reading_contrast_vanishes_when_unmatchable() -> None:
    # The §2 pre-committed deflation shape: strata differ in approach AND
    # in surprise, with disjoint surprise supports → zero matched pairs
    # (underpowered, recorded); the unmatched contrast is still reported.
    steps = _planted(1, "below_band", 100, 0.8, 10.0, jitter=0.1) + _planted(
        1, "in_band", 100, 0.2, 1.0, jitter=0.1
    )
    report = allocation_block(steps, label="planted-deflation", thresholds=_SMALL)
    assert report.n_matched_pairs == 0
    assert report.underpowered
    assert report.pooled_delta is None
    assert report.unmatched_delta == pytest.approx(0.6)
    # Underpowered fails BOTH gates — no free pass through an unvalidated
    # comparison.
    assert not toy_gate(report, _SMALL).passed
    assert not pilot_gate(report, _SMALL).passed


def test_min_pairs_floor_is_the_frozen_500() -> None:
    steps = _planted(1, "below_band", 100, 0.8, 1.0) + _planted(
        1, "in_band", 100, 0.2, 1.0
    )
    report = allocation_block(steps, label="frozen-floor")  # defaults
    assert report.n_matched_pairs == 100
    assert report.underpowered  # 100 < the frozen 500


# ---- arm contrast ---------------------------------------------------------


def _multi_seed(rate_below: float, rate_in: float) -> list[AllocationStep]:
    steps: list[AllocationStep] = []
    for seed in range(8):
        steps += _planted(seed, "below_band", 20, rate_below, 1.0)
        steps += _planted(seed, "in_band", 20, rate_in, 1.0)
    return steps


def test_arm_contrast_classifications() -> None:
    small = FrozenAllocationThresholds(min_matched_pairs=50)
    pref = allocation_block(
        _multi_seed(0.8, 0.2), label="pref", thresholds=small
    )
    control = allocation_block(
        _multi_seed(0.5, 0.5), label="control", thresholds=small
    )
    verdict = arm_contrast(pref, control, small)
    assert verdict.classification == "pass"
    assert verdict.delta_delta == pytest.approx(0.6)
    assert verdict.sign_consistent_seeds == 8

    inert = arm_contrast(control, control, small)
    assert inert.classification == "inert"
    assert inert.delta_delta == pytest.approx(0.0)

    residual_pref = allocation_block(
        _multi_seed(0.55, 0.48), label="p", thresholds=small
    )
    residual = arm_contrast(residual_pref, control, small)
    assert residual.classification == "residual"

    underpowered = arm_contrast(
        allocation_block(_multi_seed(0.8, 0.2), label="p"),  # frozen 500
        allocation_block(_multi_seed(0.5, 0.5), label="c"),
    )
    assert underpowered.classification == "underpowered"


def test_arm_contrast_requires_twin_seeds() -> None:
    small = FrozenAllocationThresholds(min_matched_pairs=10)
    a = allocation_block(
        _planted(1, "below_band", 20, 0.5, 1.0)
        + _planted(1, "in_band", 20, 0.5, 1.0),
        label="a",
        thresholds=small,
    )
    b = allocation_block(
        _planted(2, "below_band", 20, 0.5, 1.0)
        + _planted(2, "in_band", 20, 0.5, 1.0),
        label="b",
        thresholds=small,
    )
    with pytest.raises(ValueError, match="twin"):
        arm_contrast(a, b, small)


# ---- collection -----------------------------------------------------------


def _tiny_instrument() -> tuple[
    WorldModel, LatentDisagreementEnsemble, Actor, GridWorldConfig
]:
    torch.manual_seed(3)
    wm = WorldModel(
        WorldModelConfig(
            h_dim=16, z_dim=4, embed_dim=8, mlp_hidden=16, energy_embed_dim=4
        )
    )
    ensemble = LatentDisagreementEnsemble(
        h_dim=16, z_dim=4, action_dim=5, K=2, action_emb_dim=8, mlp_hidden=16
    )
    actor = Actor(h_dim=16, z_dim=4, action_dim=5, mlp_hidden=16)
    grid_cfg = GridWorldConfig(episode_length=30)
    return wm, ensemble, actor, grid_cfg


def test_collection_deterministic_and_well_formed() -> None:
    wm, ensemble, actor, grid_cfg = _tiny_instrument()
    policy = GreedyActorPolicy(actor)
    kwargs = dict(seed=5, episodes=2)
    r1 = collect_allocation_steps(
        wm, ensemble, policy, grid_cfg, **kwargs  # type: ignore[arg-type]
    )
    r2 = collect_allocation_steps(
        wm, ensemble, policy, grid_cfg, **kwargs  # type: ignore[arg-type]
    )
    assert r1 == r2
    assert len(r1) == 60
    # Episode boundary rows are excluded (resample is not action-caused).
    assert not r1[29].eligible
    # Rows carry a real surprise scalar.
    assert all(np.isfinite(s.intrinsic_signal) for s in r1)
    # The block report over collected rows serializes.
    report = allocation_block(r1, label="tiny-collection")
    import json

    json.dumps(block_report_to_jsonable(report))
