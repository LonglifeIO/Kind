"""Phase 12.5 gate test — :mod:`kind.mirror.citation_canonical`.

The shared citation canonical-form helper, exercised in isolation. No
LLM calls; the function is pure. The suite pins the iterated rule
(Phase 12.5 item 3): strip suffixes until a prefix matches a known
signal name, falling back to the bare form when none match.
"""

from __future__ import annotations

import pytest

from kind.mirror.citation_canonical import canonicalize_scalar_field


def test_bare_form_unchanged() -> None:
    assert (
        canonicalize_scalar_field("policy_entropy_t", frozenset())
        == "policy_entropy_t"
    )


def test_single_suffix_stripped_via_fallback() -> None:
    # No known signal names — the fallback strips every suffix to the
    # bare first segment.
    assert (
        canonicalize_scalar_field("policy_entropy_t.no_response", frozenset())
        == "policy_entropy_t"
    )


def test_iterated_strips_multi_suffix_to_known_signal() -> None:
    # The Phase 11 faithfulness-smoke case: a two-dot citation. One
    # strip yields `policy_entropy_t.classification`, which matches no
    # signal name; the iterated rule strips again to reach the match.
    assert (
        canonicalize_scalar_field(
            "policy_entropy_t.classification.collapse",
            frozenset({"policy_entropy_t"}),
        )
        == "policy_entropy_t"
    )


def test_iterated_fallback_strips_all_suffixes() -> None:
    # Same two-dot input, no known signal names — fallback to bare.
    assert (
        canonicalize_scalar_field(
            "policy_entropy_t.classification.collapse", frozenset()
        )
        == "policy_entropy_t"
    )


def test_longest_matching_prefix_wins() -> None:
    # When the full cited string is itself a known signal name (even one
    # that contains dots), it is returned without stripping — the walk
    # checks the full string first.
    assert (
        canonicalize_scalar_field("a.b.c", frozenset({"a.b.c"}))
        == "a.b.c"
    )
    # And an intermediate prefix is preferred over the bare form.
    assert (
        canonicalize_scalar_field("a.b.c", frozenset({"a.b"}))
        == "a.b"
    )


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        canonicalize_scalar_field("", frozenset())
