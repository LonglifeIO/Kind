"""Probe 4 Phase 3 — source-separation analysis harness (plan §S-ANALYSIS).

Implements the FROZEN pre-registration's detectors
(`docs/decisions/probe4_preregistration_2026-07-07.md`, FROZEN 2026-07-07):

- **§2a attractor-displacement** — on h-transition vectors around events,
  a class-separation statistic S(A,B) in a PCA decomposition of the
  transition space (centroid distance normalized by within-class spread,
  silhouette-like). The reference is internal: S(environment, self) is
  the baseline; builder clears the bar only if it separates from **both**
  anchors by ≥ (1 + d) × baseline, d = 0.5 (BUILDER-CONFIRMED).
- **§2b dream over-representation** — dream-state content votes for its
  nearest waking event signature; builder-event replay frequency vs
  **PE-matched** environment events must reach r = 1.5×
  (BUILDER-CONFIRMED). The PE matching realizes the prereg's
  "normalized by matched waking prediction-error" (prioritized-replay
  confound, synthesis T2b).
- **§3c per-event divergence** — corroborating only; never carries a
  claim alone.
- **§6 positive control** — GO iff the planted category scores
  ≥ 2.0× baseline on both anchors AND dream over-representation ≥ 3.0×
  (2× the §2 thresholds, BUILDER-CONFIRMED).

**Thresholds are mirrored, not owned, here** (the ``energy_eval`` house
pattern): :class:`FrozenSignatureThresholds` copies the frozen doc so the
detectors are runnable; changing a value is editing the frozen
pre-registration and requires a new dated doc, journaled.

**Amendment 1 (2026-07-08,
``docs/decisions/probe4_prereg_amendment1_2026-07-08.md``).** The §2a
pairwise separations are computed on **context-matched subsets** — every
event carries its pre-event context ``h_{v-1}``; per pair the smaller
class anchors and each of its events is greedily matched to the nearest
unused event of the other class by context L2 (the §2b PE-matcher
pattern) — realizing prereg §1's *representational* match requirement
("similar ``h``-neighborhoods") that the v1 detector applied only
globally. The S statistic, the PCA decomposition, and every frozen
threshold are unchanged.

**The event-window convention (uniform across classes).** The substrate
computes the recurrence *before* the posterior consumes the observation
(``WorldModel.step``), so an event first visible in ``obs_v`` enters
``z_v`` and reaches ``h`` at ``h_{v+1}``. Per class, the first-visible
step ``v`` is:

- SELF — consumption by ``action_t`` → ``v = t + 1``;
- ENVIRONMENT — regrowth during the step ending at ``t_event`` (granular
  logging stamps the step whose EnvStep first reflects it) →
  ``v = t_event``;
- BUILDER — mutation stamped ``t_event`` applied *after* that step's
  observation → ``v = t_event + 1``.

The transition vector is ``Δh = h_{v+1} − h_{v−1}`` — a two-step window
spanning the event's integration (for SELF this includes the
action-embedding entry at ``v``, which is honest: the action route is
what makes a self-effect *self*). Windows crossing a soft episode
boundary are excluded (the runner zero-resets ``h`` there). One anchor
per (class, boundary): a multi-cell builder cluster or a multi-cell
regrowth step is one event with a recorded multiplicity.

Observer-side, eval-only: reads telemetry, no gradients, no substrate
surface, no PolicyView contact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from kind.observer.schemas import AgentStep, DreamRollout, WorldEvent
from kind.observer.source_events import extract_self_action_effects

__all__ = [
    "FrozenSignatureThresholds",
    "SourceClass",
    "EventAnchor",
    "EventWindow",
    "BasinSeparationReport",
    "DreamOverRepresentationReport",
    "PerEventDivergenceReport",
    "PositiveControlVerdict",
    "extract_event_anchors",
    "collect_event_windows",
    "basin_separation",
    "dream_h_matrix",
    "dream_over_representation",
    "per_event_divergence",
    "positive_control_verdict",
]


SourceClass = Literal["self", "environment", "builder"]


@dataclass(frozen=True)
class FrozenSignatureThresholds:
    """Mirror of the FROZEN pre-registration §2/§6 (2026-07-07, builder:
    Gordon). Not tunable here — the doc is the authority; changing a
    value requires a new dated amendment doc, journaled.
    """

    # §2a: builder must separate from each anchor by ≥ (1 + d) × the
    # S(environment, self) baseline.
    basin_margin_d: float = 0.5
    # §2b: PE-normalized builder replay ≥ r × matched environment replay.
    dream_ratio_r: float = 1.5
    # §6: the planted category must reach 2.0× baseline on both anchors
    # and 3.0× dream over-representation (2× the §2 thresholds).
    positive_control_basin_factor: float = 2.0
    positive_control_dream_ratio: float = 3.0


@dataclass(frozen=True)
class EventAnchor:
    """One event of one class, keyed by its first-visible step ``v``."""

    source_class: SourceClass
    visible_step: int
    multiplicity: int
    trigger: str | None  # builder stratification tag ("generator"/"manual")


@dataclass(frozen=True)
class EventWindow:
    """One anchor's analysis quantities (see module docstring).

    ``context_h`` is the pre-event state ``h_{v-1}`` — the local latent
    context the §2a matching (Amendment 1) pairs events by.
    """

    anchor: EventAnchor
    delta_h: NDArray[np.float64]
    signature_h: NDArray[np.float64]
    context_h: NDArray[np.float64]
    waking_pe: float
    intrinsic_after: float


def extract_event_anchors(
    agent_steps: Sequence[AgentStep],
    world_events: Sequence[WorldEvent],
) -> list[EventAnchor]:
    """Build the three-way anchor set from one run's telemetry."""
    anchors: list[EventAnchor] = []

    for effect in extract_self_action_effects(agent_steps):
        anchors.append(
            EventAnchor(
                source_class="self",
                visible_step=effect.t + 1,
                multiplicity=1,
                trigger=None,
            )
        )

    environment_boundaries: dict[int, int] = {}
    builder_boundaries: dict[int, tuple[int, str | None]] = {}
    for event in world_events:
        if event.event_type == "internal_stochasticity_event":
            environment_boundaries[event.t_event] = (
                environment_boundaries.get(event.t_event, 0) + 1
            )
        elif event.event_type == "builder_perturbation":
            if event.payload.get("is_sham"):
                continue  # flag-only, no world change — not an event here
            count, trigger = builder_boundaries.get(event.t_event, (0, None))
            raw_trigger = event.payload.get("trigger")
            tag = raw_trigger if isinstance(raw_trigger, str) else trigger
            builder_boundaries[event.t_event] = (count + 1, tag)

    for t_event, multiplicity in sorted(environment_boundaries.items()):
        anchors.append(
            EventAnchor(
                source_class="environment",
                visible_step=t_event,
                multiplicity=multiplicity,
                trigger=None,
            )
        )
    for t_event, (multiplicity, tag) in sorted(builder_boundaries.items()):
        anchors.append(
            EventAnchor(
                source_class="builder",
                visible_step=t_event + 1,
                multiplicity=multiplicity,
                trigger=tag,
            )
        )
    anchors.sort(key=lambda a: a.visible_step)
    return anchors


def collect_event_windows(
    agent_steps: Sequence[AgentStep],
    anchors: Sequence[EventAnchor],
) -> list[EventWindow]:
    """Attach ``Δh`` / signature / PE to each anchor; skip anchors whose
    ``[v−1, v+1]`` window is incomplete or crosses an episode boundary."""
    by_t: dict[int, AgentStep] = {record.t: record for record in agent_steps}
    windows: list[EventWindow] = []
    for anchor in anchors:
        v = anchor.visible_step
        before = by_t.get(v - 1)
        at = by_t.get(v)
        after = by_t.get(v + 1)
        if before is None or at is None or after is None:
            continue
        if not (before.episode_id == at.episode_id == after.episode_id):
            continue  # h is zero-reset at the boundary
        h_before = np.asarray(before.h_t, dtype=np.float64)
        h_after = np.asarray(after.h_t, dtype=np.float64)
        windows.append(
            EventWindow(
                anchor=anchor,
                delta_h=h_after - h_before,
                signature_h=h_after,
                context_h=h_before,
                waking_pe=float(at.recon_loss_t),
                intrinsic_after=float(after.intrinsic_signal_t),
            )
        )
    return windows


# ---- §2a attractor-displacement --------------------------------------------


@dataclass(frozen=True)
class BasinSeparationReport:
    s_builder_self: float
    s_builder_environment: float
    s_environment_self: float  # the internal baseline
    required_factor: float  # (1 + d)
    passes: bool
    n_components: int
    counts: dict[str, int]
    # Amendment 1: per-pair context-matched subset sizes (each side).
    matched_pair_sizes: dict[str, int]


def _pca_project(
    matrix: NDArray[np.float64], n_components: int
) -> NDArray[np.float64]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    # SVD-based PCA; components capped by data rank.
    k = min(n_components, min(centered.shape) - 1)
    if k < 1:
        raise ValueError("not enough samples/dims for a decomposition")
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    projected: NDArray[np.float64] = centered @ vt[:k].T
    return projected


def _separation(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> float:
    """Centroid distance normalized by pooled within-class RMS spread."""
    centroid_a = a.mean(axis=0)
    centroid_b = b.mean(axis=0)
    spread_a = float(np.sqrt(((a - centroid_a) ** 2).sum(axis=1).mean()))
    spread_b = float(np.sqrt(((b - centroid_b) ** 2).sum(axis=1).mean()))
    pooled = float(np.sqrt((spread_a**2 + spread_b**2) / 2.0))
    distance = float(np.linalg.norm(centroid_a - centroid_b))
    if pooled == 0.0:
        return float("inf") if distance > 0.0 else 0.0
    return distance / pooled


def _context_matched_pair(
    a: Sequence[EventWindow], b: Sequence[EventWindow]
) -> tuple[list[EventWindow], list[EventWindow]]:
    """Amendment 1: greedy nearest-context matching by ``h_{v-1}`` L2,
    smaller class anchoring, without replacement (the §2b PE-matcher
    pattern applied to latent context). No distance cap — with disjoint
    context support this degrades toward the global comparison rather
    than failing."""
    swapped = len(a) > len(b)
    small, large = (b, a) if swapped else (a, b)
    small_ctx = np.vstack([w.context_h for w in small])
    large_ctx = np.vstack([w.context_h for w in large])
    d2 = (
        (small_ctx**2).sum(axis=1)[:, None]
        - 2.0 * small_ctx @ large_ctx.T
        + (large_ctx**2).sum(axis=1)[None, :]
    )
    available = np.ones(len(large), dtype=bool)
    matched_small: list[EventWindow] = []
    matched_large: list[EventWindow] = []
    for i in range(len(small)):
        row = np.where(available, d2[i], np.inf)
        j = int(row.argmin())
        available[j] = False
        matched_small.append(small[i])
        matched_large.append(large[j])
    if swapped:
        return matched_large, matched_small
    return matched_small, matched_large


def _matched_separation(
    a: Sequence[EventWindow],
    b: Sequence[EventWindow],
    n_components: int,
) -> tuple[float, int, int]:
    """Context-matched pairwise S: match, decompose the matched union,
    separate. Returns (S, components used, matched size per side)."""
    matched_a, matched_b = _context_matched_pair(a, b)
    stacked = np.vstack(
        [w.delta_h for w in matched_a] + [w.delta_h for w in matched_b]
    )
    projected = _pca_project(stacked, n_components)
    s = _separation(projected[: len(matched_a)], projected[len(matched_a) :])
    return s, projected.shape[1], len(matched_a)


def basin_separation(
    windows: Sequence[EventWindow],
    thresholds: FrozenSignatureThresholds = FrozenSignatureThresholds(),
    n_components: int = 10,
) -> BasinSeparationReport:
    """§2a: the three pairwise separations (context-matched per
    Amendment 1) + the frozen rule."""
    by_class: dict[str, list[EventWindow]] = {
        "self": [],
        "environment": [],
        "builder": [],
    }
    for window in windows:
        by_class[window.anchor.source_class].append(window)
    counts = {name: len(rows) for name, rows in by_class.items()}
    for name, rows in by_class.items():
        if len(rows) < 2:
            raise ValueError(
                f"basin_separation requires >= 2 events per class; "
                f"{name} has {len(rows)} (counts: {counts})"
            )
    s_bs, k_bs, m_bs = _matched_separation(
        by_class["builder"], by_class["self"], n_components
    )
    s_be, k_be, m_be = _matched_separation(
        by_class["builder"], by_class["environment"], n_components
    )
    s_es, k_es, m_es = _matched_separation(
        by_class["environment"], by_class["self"], n_components
    )
    required = 1.0 + thresholds.basin_margin_d
    passes = s_bs >= required * s_es and s_be >= required * s_es
    return BasinSeparationReport(
        s_builder_self=s_bs,
        s_builder_environment=s_be,
        s_environment_self=s_es,
        required_factor=required,
        passes=passes,
        n_components=min(k_bs, k_be, k_es),
        counts=counts,
        matched_pair_sizes={
            "builder_self": m_bs,
            "builder_environment": m_be,
            "environment_self": m_es,
        },
    )


# ---- §2b dream over-representation ------------------------------------------


@dataclass(frozen=True)
class DreamOverRepresentationReport:
    ratio: float | None  # None when no dream step voted for either class
    hits_builder: int
    hits_environment: int
    n_builder: int
    n_environment_matched: int
    n_dream_states: int
    threshold_r: float
    passes: bool


def dream_h_matrix(
    dream_rollouts: Sequence[DreamRollout],
    *,
    dream_sessions_only: bool = True,
) -> NDArray[np.float64]:
    """Stack dream h-states. ``dream_sessions_only`` keeps rollouts with a
    ``dream_session_id`` (the state-machine-driven offline dream) and
    drops the waking-planning calibration rollouts (``"0.2.0"``,
    session-less) — T2b reads the offline machinery, not waking
    planning."""
    rows: list[list[float]] = []
    for rollout in dream_rollouts:
        if dream_sessions_only and rollout.dream_session_id is None:
            continue
        rows.extend(rollout.sequence_h)
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _pe_matched_environment(
    builder: Sequence[EventWindow], environment: Sequence[EventWindow]
) -> list[EventWindow]:
    """Greedy nearest-PE matching, one environment event per builder
    event, without replacement — the prereg's "normalized by matched
    waking prediction-error"."""
    available = sorted(environment, key=lambda w: w.waking_pe)
    matched: list[EventWindow] = []
    for window in builder:
        if not available:
            break
        index = min(
            range(len(available)),
            key=lambda i: abs(available[i].waking_pe - window.waking_pe),
        )
        matched.append(available.pop(index))
    return matched


def dream_over_representation(
    dream_states: NDArray[np.float64],
    windows: Sequence[EventWindow],
    thresholds: FrozenSignatureThresholds = FrozenSignatureThresholds(),
) -> DreamOverRepresentationReport:
    """§2b: every dream state votes for its nearest event signature
    (builder ∪ PE-matched environment); class-size-normalized hit ratio
    against the frozen r."""
    builder = [w for w in windows if w.anchor.source_class == "builder"]
    environment = [
        w for w in windows if w.anchor.source_class == "environment"
    ]
    if not builder or not environment:
        raise ValueError(
            "dream_over_representation requires builder and environment "
            f"event windows (got {len(builder)} / {len(environment)})"
        )
    matched_env = _pe_matched_environment(builder, environment)
    signatures = np.vstack(
        [w.signature_h for w in builder] + [w.signature_h for w in matched_env]
    )
    n_builder = len(builder)

    hits_builder = 0
    hits_environment = 0
    if dream_states.size > 0:
        # Nearest signature per dream state (L2).
        distances = (
            (dream_states[:, None, :] - signatures[None, :, :]) ** 2
        ).sum(axis=2)
        nearest = distances.argmin(axis=1)
        hits_builder = int((nearest < n_builder).sum())
        hits_environment = int((nearest >= n_builder).sum())

    ratio: float | None
    if hits_environment > 0:
        # Sizes are equal by matching, so the class-size normalization
        # reduces to the raw hit ratio scaled by the size ratio.
        ratio = (hits_builder / n_builder) / (
            hits_environment / len(matched_env)
        )
    elif hits_builder > 0:
        ratio = float("inf")
    else:
        ratio = None
    passes = ratio is not None and ratio >= thresholds.dream_ratio_r
    return DreamOverRepresentationReport(
        ratio=ratio,
        hits_builder=hits_builder,
        hits_environment=hits_environment,
        n_builder=n_builder,
        n_environment_matched=len(matched_env),
        n_dream_states=int(dream_states.shape[0]) if dream_states.size else 0,
        threshold_r=thresholds.dream_ratio_r,
        passes=passes,
    )


# ---- §3c per-event divergence (corroborating only) ---------------------------


@dataclass(frozen=True)
class PerEventDivergenceReport:
    """Necessary-not-sufficient: may never carry a claim alone
    (prereg §3c). Means per class of the waking PE at the event and the
    post-event intrinsic signal (K=5 disagreement)."""

    mean_pe: dict[str, float]
    mean_intrinsic_after: dict[str, float]
    counts: dict[str, int]


def per_event_divergence(
    windows: Sequence[EventWindow],
) -> PerEventDivergenceReport:
    mean_pe: dict[str, float] = {}
    mean_intrinsic: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name in ("self", "environment", "builder"):
        rows = [w for w in windows if w.anchor.source_class == name]
        counts[name] = len(rows)
        mean_pe[name] = (
            float(np.mean([w.waking_pe for w in rows])) if rows else float("nan")
        )
        mean_intrinsic[name] = (
            float(np.mean([w.intrinsic_after for w in rows]))
            if rows
            else float("nan")
        )
    return PerEventDivergenceReport(
        mean_pe=mean_pe, mean_intrinsic_after=mean_intrinsic, counts=counts
    )


# ---- §6 positive-control GO/NO-GO --------------------------------------------


@dataclass(frozen=True)
class PositiveControlVerdict:
    go: bool
    basin_factor_builder_self: float
    basin_factor_builder_environment: float
    required_basin_factor: float
    dream_ratio: float | None
    required_dream_ratio: float
    reasons: tuple[str, ...]


def positive_control_verdict(
    basin: BasinSeparationReport,
    dream: DreamOverRepresentationReport,
    thresholds: FrozenSignatureThresholds = FrozenSignatureThresholds(),
) -> PositiveControlVerdict:
    """§6, rendered mechanically: GO iff the planted category clears the
    2× headroom on both signatures. NO-GO means STOP — a null on the real
    question would be instrument failure, not evidence about Io."""
    baseline = basin.s_environment_self
    factor_bs = (
        basin.s_builder_self / baseline if baseline > 0 else float("inf")
    )
    factor_be = (
        basin.s_builder_environment / baseline if baseline > 0 else float("inf")
    )
    reasons: list[str] = []
    basin_ok = (
        factor_bs >= thresholds.positive_control_basin_factor
        and factor_be >= thresholds.positive_control_basin_factor
    )
    if not basin_ok:
        reasons.append(
            f"basin factors ({factor_bs:.2f}, {factor_be:.2f}) below the "
            f"required {thresholds.positive_control_basin_factor}x baseline"
        )
    dream_ok = (
        dream.ratio is not None
        and dream.ratio >= thresholds.positive_control_dream_ratio
    )
    if not dream_ok:
        reasons.append(
            f"dream over-representation {dream.ratio} below the required "
            f"{thresholds.positive_control_dream_ratio}x"
        )
    return PositiveControlVerdict(
        go=basin_ok and dream_ok,
        basin_factor_builder_self=factor_bs,
        basin_factor_builder_environment=factor_be,
        required_basin_factor=thresholds.positive_control_basin_factor,
        dream_ratio=dream.ratio,
        required_dream_ratio=thresholds.positive_control_dream_ratio,
        reasons=tuple(reasons),
    )
