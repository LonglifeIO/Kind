"""Pytest configuration — Phase 12 adds the ``--run-real-api`` flag and
the ``real_api`` mark.

Tests marked ``@pytest.mark.real_api`` are skipped by default. They run
only when ``pytest --run-real-api`` is passed or when the
:envvar:`GEMINI_API_KEY` environment variable is set (the latter so CI
that has the secret configured runs the smoke automatically). This is
the project's first opt-in real-API surface; expanding it to other
tests should be a journaled decision.
"""

from __future__ import annotations

import os

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-real-api",
        action="store_true",
        default=False,
        help=(
            "Run tests marked @pytest.mark.real_api. Defaults to off so "
            "the normal pytest invocation does not consume API quota."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "real_api: mark test as requiring a real LLM API call. Skipped "
        "unless --run-real-api is passed or GEMINI_API_KEY is set.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-real-api"):
        return
    if os.environ.get("GEMINI_API_KEY"):
        return
    skip_real_api = pytest.mark.skip(
        reason=(
            "real-API test: pass --run-real-api or set GEMINI_API_KEY "
            "to opt in"
        )
    )
    for item in items:
        if "real_api" in item.keywords:
            item.add_marker(skip_real_api)
