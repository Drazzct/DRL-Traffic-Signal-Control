"""Scenario manifest loading and route selection."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from traffic_drl.contracts import ScenarioRecord, ScenarioSource


class ScenarioManifest(ScenarioSource):
    """Read-only catalog of route files and reproducibility metadata."""

    def __init__(self, records: Sequence[ScenarioRecord]) -> None:
        """Create a manifest from typed scenario records.

        Args:
            records: Immutable scenario records loaded from the manifest.

        Returns:
            None: Initializes the manifest instance.
        """
        self.records = tuple(records)

    @classmethod
    def from_csv(cls, path: str | Path) -> "ScenarioManifest":
        """Load a manifest from a scenario CSV file.

        Args:
            path: UTF-8 CSV path with a header containing at least
                ``scenario_id``, ``split``, ``route_file``, ``demand_seed``,
                ``sumo_seed``, and ``num_seconds``. Optional columns belong in
                ``ScenarioRecord.metadata``.

        File contract: parse with :mod:`csv.DictReader`; convert seed and
        duration columns to integers, resolve ``route_file`` relative to the
        repository root, and preserve optional columns as metadata.

        TODO (SV3): validate duplicate IDs, split names, route existence, and
        locked-test checksums.
        """
        raise NotImplementedError

    def records_for_split(self, split: str) -> tuple[ScenarioRecord, ...]:
        """Return records belonging to one split.

        Args:
            split: Split label such as ``TR``, ``VA``, or ``TE``.

        Returns:
            tuple[ScenarioRecord, ...]: Matching records in manifest order.
        """
        raise NotImplementedError

    def get(self, scenario_id: str) -> ScenarioRecord:
        """Return a record by ID.

        Args:
            scenario_id: Stable manifest identifier.

        Returns:
            ScenarioRecord: The matching typed scenario record.
        """
        raise NotImplementedError


def create_sumo_env(config: Any, selected_route: str | Path, *, fixed_ts: bool = False, seed: int | None = None, use_gui: bool = False) -> Any:
    """Create one SUMO-RL single-agent environment for one route.

    Returns:
        Any: A Gymnasium-compatible environment bound to ``selected_route``.
    """
    raise NotImplementedError
