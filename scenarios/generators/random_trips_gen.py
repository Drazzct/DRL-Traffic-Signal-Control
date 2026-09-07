"""Phase 1 random route generation API."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def generate_random_trips(*, network_file: str | Path, output_route_file: str | Path, begin: int, end: int, period: float, seed: int, vehicle_types_file: str | Path | None = None, additional_options: Mapping[str, Any] | None = None) -> Path:
    """Generate one SUMO route XML file with ``randomTrips.py``.

    Args:
        network_file: SUMO ``.net.xml`` containing nodes, edges, and lanes;
            pass it to SUMO's route generator rather than parsing manually.
        output_route_file: Destination ``.rou.xml`` containing ``<route>`` and
            ``<vehicle>`` elements.
        begin: First simulation second at which trips may depart.
        end: Last simulation second covered by generated trips.
        period: Generation interval in seconds; it is not per-lane flow.
        seed: Random seed recorded in scenario metadata.
        vehicle_types_file: Optional SUMO ``.add.xml`` containing ``<vType>`` definitions.
        additional_options: Extra validated command options for ``randomTrips.py``.

    File contract: read the network/additional XML through SUMO tooling and
    validate the generated route XML with an XML parser plus SUMO route checks.

    Returns:
        Path: The created route XML path.
    """
    raise NotImplementedError


def count_realized_flow(route_file: str | Path, *, approach_ids: list[str] | None = None) -> dict[str, int]:
    """Count generated vehicles by approach for manifest QA.

    Args:
        route_file: SUMO ``.rou.xml`` with vehicle route references.
        approach_ids: Optional stable approach IDs used to group route starts.

    File contract: parse XML with :mod:`xml.etree.ElementTree`; count vehicle
    departures by the first public edge in each route.

    Returns:
        dict[str, int]: Vehicle counts keyed by approach ID.
    """
    raise NotImplementedError


def validate_route_file(route_file: str | Path, *, expected_vehicle_types: set[str] | None = None) -> list[str]:
    """Return route-file validation errors without running a policy.

    Args:
        route_file: SUMO ``.rou.xml`` to parse and validate.
        expected_vehicle_types: Optional allowed IDs from a vehicle-types ``.add.xml`` file.

    File contract: parse ``<route>``, ``<vehicle>``, and ``type`` references;
    do not start SUMO or execute an RL policy.

    Returns:
        list[str]: Validation error messages; an empty list means valid.
    """
    raise NotImplementedError
