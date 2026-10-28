"""Every sample CSV must give exactly the same profile and cleaning log as before the speedup."""

import json

import pytest

from app.core import profiling as P
from tests.conftest import SAMPLE_NAMES
from tests.golden import diff, golden_path, snapshot


def _golden(name: str) -> dict:
    return json.loads(golden_path(name).read_text("utf-8"))


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_matches_golden(name: str) -> None:
    changes = diff(_golden(name), snapshot(name))
    assert not changes, "\n".join(changes[:30])


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_tiny_type_sample_still_matches_golden(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Types come from a sample, but every count still runs on the full data."""
    monkeypatch.setattr(P, "TYPE_SAMPLE_ROWS", 50)
    changes = diff(_golden(name), snapshot(name))
    assert not changes, "\n".join(changes[:30])
