"""Probe 2 Phase 0 — ``schemas/v0.3.0.json`` export byte-stability test.

Per the implementation plan §3.6 and Phase 0 task: the v0.3.0 export
covers Probe 1.5's ``"0.2.0"`` telemetry models plus Probe 2's
``"0.2.0"`` mirror-side models plus Probe 2's ``"0.1.0"`` conditioning
model; the export is byte-stable across invocations and matches the
file checked in at ``schemas/v0.3.0.json``. v0.1.0.json and v0.2.0.json
remain on disk unchanged (the prior pinning tests in
``tests/test_schemas.py`` continue to enforce that).
"""

from __future__ import annotations

import json
from pathlib import Path

from kind.observer.schemas import (
    PROBE_2_EXPORT_VERSION,
    export_json_schema,
    export_json_schema_v0_3_0,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_FILE_V0_3_0 = REPO_ROOT / "schemas" / "v0.3.0.json"
SCHEMA_FILE_V0_2_0 = REPO_ROOT / "schemas" / "v0.2.0.json"


def test_v0_3_0_export_is_byte_stable() -> None:
    """The same Python source produces the same bytes on every
    invocation. Same discipline as :func:`export_json_schema` v0.2.0."""
    a = export_json_schema_v0_3_0()
    b = export_json_schema_v0_3_0()
    assert a == b


def test_v0_3_0_export_matches_checked_in_file() -> None:
    """The bytes on disk at ``schemas/v0.3.0.json`` match the current
    export. Out-of-sync edits to any of the contributing model modules
    fail here — regenerate via :func:`export_json_schema_v0_3_0` and
    commit the result."""
    assert SCHEMA_FILE_V0_3_0.exists(), (
        f"missing schema export at {SCHEMA_FILE_V0_3_0}"
    )
    assert SCHEMA_FILE_V0_3_0.read_bytes() == export_json_schema_v0_3_0(), (
        "schemas/v0.3.0.json is out of sync with the contributing "
        "model modules — regenerate via export_json_schema_v0_3_0() "
        "and commit the result."
    )


def test_v0_3_0_export_covers_telemetry_mirror_and_conditioning() -> None:
    """The v0.3.0 export aggregates models from three families. The
    ``models`` dict has three top-level groups; each group lists the
    expected model names."""
    document = json.loads(export_json_schema_v0_3_0())

    assert document["schema_version"] == PROBE_2_EXPORT_VERSION
    assert document["title"] == "Kind Probe 2 Schemas"

    assert document["telemetry_schema_version"] == "0.2.0"
    assert document["mirror_schema_version"] == "0.2.0"
    assert document["conditioning_schema_version"] == "0.1.0"

    models = document["models"]
    assert set(models["telemetry"].keys()) == {
        "AgentStep",
        "DreamRollout",
        "ReplayMeta",
        "WorldEvent",
    }
    assert set(models["mirror"].keys()) == {
        "StructuredReading",
        "StructuredClaim",
        "JudgeRuling",
        "PreRegistration",
    }
    assert set(models["conditioning"].keys()) == {
        "ConditioningResult",
        "RegimeBucket",
        "RegimeStats",
    }


def test_v0_2_0_export_unchanged_by_v0_3_0_addition() -> None:
    """Adding the v0.3.0 export must not perturb the v0.2.0 export's
    bytes. The Phase-0 update to ``kind/observer/schemas.py`` extends
    the module docstring and adds a new function but does not touch the
    four telemetry models' definitions; ``schemas/v0.2.0.json`` stays
    bit-exact."""
    assert SCHEMA_FILE_V0_2_0.exists()
    assert SCHEMA_FILE_V0_2_0.read_bytes() == export_json_schema()


def test_v0_3_0_export_contains_v2_baseline_flag_values() -> None:
    """A spot-check on the exported StructuredReading schema: the
    ``baseline_flag`` enum carries the four v2-added values
    (shuffled_scalar_within_trajectory, lesion_disable_self_prediction,
    lesion_init_zero_scalar_column, lesion_zero_or_randomize_scalar).
    If a future edit accidentally drops one, the v0.3.0 export's checked
    bytes change and the matches-checked-in test fails — but this test
    fails first with a more legible message."""
    document = json.loads(export_json_schema_v0_3_0())
    sr = document["models"]["mirror"]["StructuredReading"]
    flag_enum = sr["properties"]["baseline_flag"]["enum"]
    for required in (
        "genuine",
        "shuffled_scalar_within_trajectory",
        "lesion_disable_self_prediction",
        "lesion_init_zero_scalar_column",
        "lesion_zero_or_randomize_scalar",
    ):
        assert required in flag_enum, (
            f"baseline_flag enum missing v2 value {required!r}"
        )


def test_v0_3_0_export_contains_three_reading_surfaces() -> None:
    """The StructuredClaim's ``reading_surface`` enum carries the three
    surfaces: substrate_side, head_internal, behavior_side."""
    document = json.loads(export_json_schema_v0_3_0())
    sc = document["models"]["mirror"]["StructuredClaim"]
    surface_property = sc["properties"]["reading_surface"]
    enum = surface_property.get("enum")
    if enum is None:
        # Pydantic may emit a $ref alias; resolve via $defs.
        ref = surface_property["$ref"].split("/")[-1]
        enum = sc["$defs"][ref]["enum"]
    assert set(enum) == {"substrate_side", "head_internal", "behavior_side"}


def test_v0_3_0_export_contains_conditioning_analysis_cited_stream() -> None:
    """``cited_stream`` enum on StructuredClaim carries the new v2 value
    ``conditioning_analysis``."""
    document = json.loads(export_json_schema_v0_3_0())
    sc = document["models"]["mirror"]["StructuredClaim"]
    stream_property = sc["properties"]["cited_stream"]
    enum = stream_property.get("enum")
    if enum is None:
        ref = stream_property["$ref"].split("/")[-1]
        enum = sc["$defs"][ref]["enum"]
    assert "conditioning_analysis" in enum
