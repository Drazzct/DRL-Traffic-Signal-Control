"""Custom mixed-traffic scenario generator API."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def generate_custom_flow(*, scenario_id: str, output_route_file: str | Path, demand_by_approach: Mapping[str, float], turning_ratios: Mapping[str, float], vehicle_mix: Mapping[str, float], begin: int, end: int, seed: int, flow_pattern: str = "uniform", behavior_profiles: Mapping[str, str] | None = None) -> Path:
    """Generate a deterministic SUMO ``.rou.xml`` scenario.

    Args:
        scenario_id: Stable manifest ID written into route metadata.
        output_route_file: Destination XML containing routes, vehicles, and types.
        demand_by_approach: Vehicles/hour or equivalent demand keyed by approach ID.
        turning_ratios: OD/movement proportions keyed by valid movement IDs.
        vehicle_mix: Vehicle-class proportions that must sum to one.
        begin: First departure second.
        end: Last departure second.
        seed: Deterministic route-generation seed.
        flow_pattern: Pattern such as ``uniform``, ``surge``, or ``platoon``.
        behavior_profiles: Optional vehicle-class profile names.

    File contract: write UTF-8 SUMO route XML and validate references against
    the network and vehicle-types XML before returning.

    Returns:
        Path: The created route XML path.
    """
    raise NotImplementedError


def generate_scenario_set(manifest_path: str | Path, *, output_root: str | Path, split: str, seed: int) -> list[Path]:
    """Generate all routes in one allowed split from the manifest.

    Args:
        manifest_path: UTF-8 CSV manifest with scenario IDs, split, route path,
            seeds, duration, demand, turning ratios, and vehicle mix columns.
        output_root: Directory where generated ``.rou.xml`` files are written.
        split: Split to generate; TE must remain locked until approval.
        seed: Seed used for generation randomization.

    File contract: parse the manifest with :class:`csv.DictReader`; write one
    SUMO route XML per selected row and preserve the manifest IDs.

    Returns:
        list[Path]: Paths of the generated route XML files.
    """
    raise NotImplementedError


def write_scenario_report(route_file: str | Path, output_path: str | Path) -> None:
    """Write realized vehicle counts and metadata for reviewer checks.

    Args:
        route_file: Generated SUMO ``.rou.xml`` parsed for route/vehicle counts.
        output_path: UTF-8 ``.json`` or ``.csv`` report destination selected by suffix.

    File contract: serialize structured counts, seeds, and scenario ID with
    :mod:`json` or :mod:`csv`, never free-form text.

    Returns:
        None: The report is written to ``output_path``.
    """
    raise NotImplementedError
