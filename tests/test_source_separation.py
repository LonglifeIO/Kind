"""Probe 4 Phase 3 — source-separation detector tests (plan §S-ANALYSIS).

Synthetic must-fire / must-not-fire validation of the frozen-prereg
detectors: a planted distinct third category must make §2a and §2b fire;
a builder class statistically indistinguishable from the environment
class must not. Event-anchor timing conventions (the per-class
first-visible step) are pinned against hand-built telemetry.
"""

from __future__ import annotations

import numpy as np
import pytest

from kind.observer.schemas import (
    PROBE_3_5_TELEMETRY_SCHEMA_VERSION,
    PROBE_4_WORLD_EVENT_SCHEMA_VERSION,
    AgentStep,
    WorldEvent,
)
from kind.observer.source_separation import (
    EventAnchor,
    EventWindow,
    FrozenSignatureThresholds,
    basin_separation,
    collect_event_windows,
    dream_over_representation,
    extract_event_anchors,
    per_event_divergence,
    positive_control_verdict,
)

_H_DIM = 8


def _make_step(
    *,
    t: int,
    episode_id: int = 0,
    action: int = 1,
    true_energy: float = 0.5,
    h: list[float] | None = None,
    recon_loss: float = 1.0,
    intrinsic: float = 0.1,
) -> AgentStep:
    return AgentStep(
        schema_version=PROBE_3_5_TELEMETRY_SCHEMA_VERSION,
        run_id="synthetic",
        checkpoint_id=None,
        t=t,
        episode_id=episode_id,
        step_in_episode=t % 200,
        wallclock_ms=t,
        h_t=h if h is not None else [float(t)] * _H_DIM,
        q_params_t=([0.0], [0.0]),
        p_params_t=([0.0], [0.0]),
        z_t=[0.0],
        kl_per_dim_t=[0.0],
        kl_aggregate_t=0.0,
        recon_loss_t=recon_loss,
        action_t=action,
        action_logprob_t=0.0,
        policy_entropy_t=0.0,
        obs_hash_t="hash",
        intrinsic_signal_t=intrinsic,
        encoder_embedding_t=[0.0],
        self_prediction_t=[0.0],
        self_prediction_error_t=0.0,
        self_prediction_error_masked_t=False,
        sensed_energy_t=true_energy,
        true_energy_t=true_energy,
        energy_pred_t=true_energy,
        energy_recon_error_t=0.0,
    )


def _world_event(
    *,
    t_event: int,
    event_type: str,
    payload: dict[str, object],
    schema_version: str = "0.1.0",
) -> WorldEvent:
    return WorldEvent(
        schema_version=schema_version,
        run_id="synthetic",
        checkpoint_id=None,
        t_event=t_event,
        event_type=event_type,  # type: ignore[arg-type]
        source="builder" if event_type == "builder_perturbation" else "environment",
        payload=payload,
        wallclock_ms=t_event,
    )


def _granular_env_event(t_event: int, cell: list[int]) -> WorldEvent:
    return _world_event(
        t_event=t_event,
        event_type="internal_stochasticity_event",
        payload={
            "process": "regrowth",
            "cell": cell,
            "pre_state": "empty",
            "post_state": "resource",
        },
        schema_version=PROBE_4_WORLD_EVENT_SCHEMA_VERSION,
    )


def _builder_event(
    t_event: int, cell: list[int], trigger: str = "generator"
) -> WorldEvent:
    return _world_event(
        t_event=t_event,
        event_type="builder_perturbation",
        payload={
            "mutator": "add_resource",
            "cell": cell,
            "pre_state": "empty",
            "post_state": "resource",
            "trigger": trigger,
        },
    )


# ---- anchor extraction: the first-visible conventions -----------------------


def test_anchor_visible_step_conventions() -> None:
    """SELF v = t+1; ENVIRONMENT v = t_event; BUILDER v = t_event + 1."""
    steps = [
        _make_step(t=t, true_energy=0.5 if t <= 10 else 0.58)
        for t in range(8, 14)
    ]  # consumption jump between t=10 and t=11 → SELF anchor v = 11
    events = [
        _granular_env_event(9, [1, 1]),
        _builder_event(12, [2, 2]),
    ]
    anchors = extract_event_anchors(steps, events)
    by_class = {a.source_class: a for a in anchors}
    assert by_class["self"].visible_step == 11
    assert by_class["environment"].visible_step == 9
    assert by_class["builder"].visible_step == 13
    assert by_class["builder"].trigger == "generator"


def test_anchor_multiplicity_and_dedup() -> None:
    """Multi-cell boundaries collapse to one anchor with multiplicity."""
    steps = [_make_step(t=t) for t in range(5)]
    events = [
        _builder_event(2, [1, 1]),
        _builder_event(2, [1, 2]),
        _granular_env_event(3, [4, 4]),
        _granular_env_event(3, [5, 5]),
        _granular_env_event(3, [6, 6]),
    ]
    anchors = extract_event_anchors(steps, events)
    builder = [a for a in anchors if a.source_class == "builder"]
    environment = [a for a in anchors if a.source_class == "environment"]
    assert len(builder) == 1 and builder[0].multiplicity == 2
    assert len(environment) == 1 and environment[0].multiplicity == 3


def test_sham_perturbations_are_not_anchors() -> None:
    steps = [_make_step(t=t) for t in range(5)]
    sham = _world_event(
        t_event=2,
        event_type="builder_perturbation",
        payload={"is_sham": True, "sham_label": "add_resource"},
    )
    anchors = extract_event_anchors(steps, [sham])
    assert all(a.source_class != "builder" for a in anchors)


# ---- window collection -------------------------------------------------------


def test_window_delta_and_boundary_exclusion() -> None:
    steps = [
        _make_step(t=0, h=[0.0] * _H_DIM),
        _make_step(t=1, h=[1.0] * _H_DIM),
        _make_step(t=2, h=[3.0] * _H_DIM),
        # Episode boundary between t=3 and t=4:
        _make_step(t=3, h=[4.0] * _H_DIM),
        _make_step(t=4, episode_id=1, h=[0.0] * _H_DIM),
        _make_step(t=5, episode_id=1, h=[1.0] * _H_DIM),
    ]
    anchors = [
        EventAnchor("environment", visible_step=1, multiplicity=1, trigger=None),
        EventAnchor("environment", visible_step=4, multiplicity=1, trigger=None),
    ]
    windows = collect_event_windows(steps, anchors)
    # The boundary-crossing window (v=4 needs t=3..5 across episodes) is
    # excluded; the clean one survives with delta = h_2 - h_0.
    assert len(windows) == 1
    assert windows[0].anchor.visible_step == 1
    np.testing.assert_allclose(windows[0].delta_h, np.full(_H_DIM, 3.0))
    np.testing.assert_allclose(windows[0].signature_h, np.full(_H_DIM, 3.0))
    # Amendment 1: the matching context is the pre-event state h_{v-1}.
    np.testing.assert_allclose(windows[0].context_h, np.zeros(_H_DIM))


# ---- §2a basin separation ----------------------------------------------------


def _synthetic_windows(
    *,
    builder_offset: float,
    n_per_class: int = 40,
    seed: int = 0,
    builder_pe: float = 5.0,
    env_pe: float = 5.0,
) -> list[EventWindow]:
    """Gaussian clouds in transition space: self at 0, environment at a
    fixed offset along dim 1, builder at ``builder_offset`` along dim 2.
    ``builder_offset=0`` puts builder *inside* the environment cloud.
    Contexts are drawn from one shared distribution so the Amendment-1
    matching pairs events freely and class geometry alone decides."""
    rng = np.random.default_rng(seed)
    windows: list[EventWindow] = []
    centers = {
        "self": np.zeros(_H_DIM),
        "environment": np.eye(_H_DIM)[1] * 2.0,
        "builder": np.eye(_H_DIM)[1] * 2.0
        + np.eye(_H_DIM)[2] * builder_offset,
    }
    pes = {"self": 5.0, "environment": env_pe, "builder": builder_pe}
    for name, center in centers.items():
        for index in range(n_per_class):
            delta = center + rng.normal(0.0, 0.3, _H_DIM)
            windows.append(
                EventWindow(
                    anchor=EventAnchor(
                        name,  # type: ignore[arg-type]
                        visible_step=index,
                        multiplicity=1,
                        trigger=None,
                    ),
                    delta_h=delta,
                    signature_h=center + rng.normal(0.0, 0.3, _H_DIM),
                    context_h=rng.normal(0.0, 1.0, _H_DIM),
                    waking_pe=pes[name] + float(rng.normal(0.0, 0.5)),
                    intrinsic_after=0.1,
                )
            )
    return windows


def test_basin_fires_on_planted_distinct_category() -> None:
    windows = _synthetic_windows(builder_offset=8.0)
    report = basin_separation(windows)
    assert report.passes
    baseline = report.s_environment_self
    assert report.s_builder_self >= 2.0 * baseline  # headroom for §6 too
    assert report.s_builder_environment >= 2.0 * baseline


def test_basin_silent_when_builder_matches_environment() -> None:
    windows = _synthetic_windows(builder_offset=0.0)
    report = basin_separation(windows)
    assert not report.passes
    # Builder sits inside the environment cloud: near-zero separation.
    assert report.s_builder_environment < report.s_environment_self


def test_basin_silent_when_separation_is_context_driven() -> None:
    """Amendment 1 must-not-fire: builder deltas differ from environment
    deltas only because builder events land in different (but
    overlapping) h-contexts — the same context→delta law governs both.
    Context matching compares like-context events, so no basin fires;
    the v1 global comparison would have read the context shift as
    separation."""
    rng = np.random.default_rng(7)
    windows: list[EventWindow] = []

    def _law(c: float) -> np.ndarray:
        return np.eye(_H_DIM)[3] * c

    specs = {
        "self": ("fixed", 0.0),  # genuinely distinct class, fixed context
        "environment": ("uniform", (0.0, 4.0)),
        "builder": ("uniform", (1.0, 5.0)),  # shifted, overlapping
    }
    for name, (kind, param) in specs.items():
        for index in range(60):
            if kind == "fixed":
                context_scale = float(param)  # type: ignore[arg-type]
                delta = np.eye(_H_DIM)[1] * 2.0 + rng.normal(0.0, 0.3, _H_DIM)
            else:
                low, high = param  # type: ignore[misc]
                context_scale = float(rng.uniform(low, high))
                delta = _law(context_scale) + rng.normal(0.0, 0.3, _H_DIM)
            windows.append(
                EventWindow(
                    anchor=EventAnchor(
                        name,  # type: ignore[arg-type]
                        visible_step=index,
                        multiplicity=1,
                        trigger=None,
                    ),
                    delta_h=delta,
                    signature_h=delta,
                    context_h=np.eye(_H_DIM)[0] * context_scale,
                    waking_pe=5.0,
                    intrinsic_after=0.1,
                )
            )
    report = basin_separation(windows)
    assert not report.passes
    # Like-context builder/environment events transition identically:
    # the matched builder-environment separation sits below the
    # environment-self baseline (a genuinely different class).
    assert report.s_builder_environment < report.s_environment_self
    assert report.matched_pair_sizes["builder_environment"] == 60


def test_basin_requires_all_three_classes() -> None:
    windows = [
        w
        for w in _synthetic_windows(builder_offset=4.0)
        if w.anchor.source_class != "self"
    ]
    with pytest.raises(ValueError, match="self has 0"):
        basin_separation(windows)


# ---- §2b dream over-representation -------------------------------------------


def test_dream_fires_when_dreams_cluster_on_builder_signatures() -> None:
    windows = _synthetic_windows(builder_offset=8.0)
    builder_signatures = np.vstack(
        [w.signature_h for w in windows if w.anchor.source_class == "builder"]
    )
    rng = np.random.default_rng(1)
    dream_states = builder_signatures[
        rng.integers(0, len(builder_signatures), 200)
    ] + rng.normal(0.0, 0.05, (200, _H_DIM))
    report = dream_over_representation(dream_states, windows)
    assert report.passes
    assert report.ratio is not None and report.ratio >= 3.0  # §6 headroom


def test_dream_silent_on_source_indifferent_dreams() -> None:
    """Dreams that sample builder and environment signatures evenly give
    a ratio near parity — below r."""
    windows = _synthetic_windows(builder_offset=8.0)
    signatures = np.vstack(
        [
            w.signature_h
            for w in windows
            if w.anchor.source_class in ("builder", "environment")
        ]
    )
    rng = np.random.default_rng(2)
    dream_states = signatures[
        rng.integers(0, len(signatures), 400)
    ] + rng.normal(0.0, 0.05, (400, _H_DIM))
    report = dream_over_representation(dream_states, windows)
    assert not report.passes
    assert report.ratio is not None and report.ratio < 1.5


def test_dream_pe_matching_uses_equal_class_sizes() -> None:
    windows = _synthetic_windows(builder_offset=8.0)
    report = dream_over_representation(
        np.zeros((10, _H_DIM)), windows
    )
    assert report.n_builder == report.n_environment_matched == 40


def test_dream_no_states_gives_none_ratio() -> None:
    windows = _synthetic_windows(builder_offset=8.0)
    report = dream_over_representation(np.zeros((0, 0)), windows)
    assert report.ratio is None and not report.passes


# ---- §3c + §6 -----------------------------------------------------------------


def test_per_event_divergence_reports_class_means() -> None:
    windows = _synthetic_windows(
        builder_offset=8.0, builder_pe=9.0, env_pe=3.0
    )
    report = per_event_divergence(windows)
    assert report.counts == {"self": 40, "environment": 40, "builder": 40}
    assert report.mean_pe["builder"] > report.mean_pe["environment"]


def test_positive_control_verdict_go_and_stop() -> None:
    thresholds = FrozenSignatureThresholds()
    blatant = _synthetic_windows(builder_offset=10.0)
    basin = basin_separation(blatant, thresholds)
    builder_signatures = np.vstack(
        [w.signature_h for w in blatant if w.anchor.source_class == "builder"]
    )
    rng = np.random.default_rng(3)
    dreams = builder_signatures[
        rng.integers(0, len(builder_signatures), 300)
    ] + rng.normal(0.0, 0.05, (300, _H_DIM))
    dream = dream_over_representation(dreams, blatant, thresholds)
    verdict = positive_control_verdict(basin, dream, thresholds)
    assert verdict.go and verdict.reasons == ()

    subtle = _synthetic_windows(builder_offset=0.0)
    basin_null = basin_separation(subtle, thresholds)
    dream_null = dream_over_representation(
        np.zeros((10, _H_DIM)), subtle, thresholds
    )
    verdict_null = positive_control_verdict(basin_null, dream_null, thresholds)
    assert not verdict_null.go
    assert len(verdict_null.reasons) >= 1


def test_frozen_thresholds_mirror_the_prereg() -> None:
    """The code-side mirror must match the FROZEN doc's confirmed values
    (d=0.5, r=1.5, 2x headroom → 2.0x basin / 3.0x dream)."""
    thresholds = FrozenSignatureThresholds()
    assert thresholds.basin_margin_d == 0.5
    assert thresholds.dream_ratio_r == 1.5
    assert thresholds.positive_control_basin_factor == 2.0
    assert thresholds.positive_control_dream_ratio == 3.0
