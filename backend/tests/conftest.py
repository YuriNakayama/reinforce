"""Shared pytest configuration.

Assigns scope markers from the top-level test directory so the CI fast slice
(`-m "not slow and not integration and not e2e"`) stays correct without every
test author remembering to mark their file. `unit` intentionally gets no
marker — it is the default and always runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]

# backend/ must be importable so `pipeline.<family>.case<N>` resolves in tests.
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

_SCOPE_MARKERS = ("integration", "e2e")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    tests_root = Path(__file__).parent
    for item in items:
        try:
            relative = Path(str(item.fspath)).relative_to(tests_root)
        except ValueError:
            continue
        if relative.parts and relative.parts[0] in _SCOPE_MARKERS:
            item.add_marker(getattr(pytest.mark, relative.parts[0]))
