"""Shared citation canonical-form helper.

Phase 12.5 item 4 lifts :func:`canonicalize_scalar_field` out of
:mod:`kind.mirror.faithfulness` into this shared module so both the
faithfulness verifier (:mod:`kind.mirror.faithfulness`) and the Phase 10
stability runner (:mod:`kind.mirror.stability`) normalize a claim's
``cited_scalar_field`` string the same way. Before this, the stability
metric compared the verbatim LLM string, so the same signal cited bare
in one reading and compound in another read as disagreement — the
divergence that tripped Phase 10's HEAD_INTERNAL paraphrase agreement
to 0.333.

**The canonical-form rule (Phase 12.5 item 3 — iterated).** Phase 10's
smoke surfaced the LLM constructing ``cited_scalar_field`` in multiple
equally-faithful forms — ``latent_self_reference_t`` (bare) and
``latent_self_reference_t.partial_autocorr_lag5`` (compound with an
estimator-name suffix). Phase 13's production data added a dict-key
compound form (``policy_entropy_t.no_response``). Phase 11's
faithfulness smoke surfaced two-dot forms
(``policy_entropy_t.classification.collapse``), which a single-suffix
strip leaves at ``policy_entropy_t.classification`` — still not a
statistic signal name.

The rule is therefore iterated: walk the dot-separated suffix chain
from the full original string down toward the bare form; the first
prefix that matches a known statistic signal name is the canonical
form. If no prefix matches any known signal name, fall back to the bare
(no-dot) form. The known signal names come from the caller — the
faithfulness verifier derives them from its
:class:`~kind.mirror.statistics.StatisticResult` records; the stability
runner derives them from the criterion's
:class:`~kind.mirror.registry.SignalMapping` declarations.
"""

from __future__ import annotations

__all__ = ["canonicalize_scalar_field"]


def canonicalize_scalar_field(
    field: str, known_signal_names: frozenset[str]
) -> str:
    """Return the canonical form of a ``cited_scalar_field`` value.

    The rule (iterated): walk the dot-separated prefix chain from the
    full original string down to the bare first segment; return the
    first prefix that is a member of ``known_signal_names``. If no
    prefix matches, return the bare form (first segment, all suffixes
    stripped).

    Examples (``E`` = ``frozenset({"policy_entropy_t"})``):

    - ``canonicalize_scalar_field("policy_entropy_t", E)`` →
      ``"policy_entropy_t"`` (the full string is itself a known name).
    - ``canonicalize_scalar_field("policy_entropy_t.no_response", E)`` →
      ``"policy_entropy_t"`` (one suffix stripped to reach the match).
    - ``canonicalize_scalar_field("policy_entropy_t.classification.collapse", E)``
      → ``"policy_entropy_t"`` (two suffixes stripped — the iterated
      case Phase 11's faithfulness smoke surfaced).
    - ``canonicalize_scalar_field("policy_entropy_t.classification.collapse",
      frozenset())`` → ``"policy_entropy_t"`` (no known names; fallback
      strips every suffix to the bare form).

    Empty input raises :class:`ValueError` — a ``cited_scalar_field``
    of ``""`` is a caller-level fault, not a canonical form.
    """
    if not field:
        raise ValueError(
            "canonicalize_scalar_field: input must be non-empty; got ''."
        )
    parts = field.split(".")
    for n in range(len(parts), 0, -1):
        prefix = ".".join(parts[:n])
        if prefix in known_signal_names:
            return prefix
    return parts[0]
