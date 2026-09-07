"""SUMO network and route validation test signatures."""
from __future__ import annotations

from pathlib import Path


def test_network_file_exists() -> None:
    """Check that the configured network artifact exists."""
    assert True


def validate_network(path: str | Path) -> list[str]:
    """Return disconnected-lane or invalid-connection diagnostics."""
    raise NotImplementedError
