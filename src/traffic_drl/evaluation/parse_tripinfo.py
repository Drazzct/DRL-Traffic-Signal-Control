"""SUMO tripinfo and emissions parsing API."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from traffic_drl.contracts import EmissionMetrics, EpisodeMetrics, TripInfoMetrics


def parse_tripinfo(path: str | Path, episode_seconds: float | None = None) -> TripInfoMetrics:
    """Parse SUMO ``tripinfo.xml`` into traffic metrics.

    Args:
        path: UTF-8 XML file containing one ``<tripinfo>`` element per vehicle,
            with SUMO attributes such as ``duration``, ``waitingTime``,
            ``timeLoss``, and ``depart``/``arrival``.
        episode_seconds: Optional duration of episode in simulation seconds. If
            omitted, duration is inferred from the latest vehicle arrival time.

    Returns:
        TripInfoMetrics: Vehicle count and averaged waiting, loss, travel-time,
        and throughput values.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    xml_path = Path(path)
    if not xml_path.exists():
        raise FileNotFoundError(f"Tripinfo file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    tripinfos = list(root.iter("tripinfo"))
    vehicle_count = len(tripinfos)

    if vehicle_count == 0:
        return TripInfoMetrics(
            vehicle_count=0,
            average_waiting_time=0.0,
            average_time_loss=0.0,
            average_travel_time=0.0,
            throughput=0.0,
        )

    total_waiting_time = 0.0
    total_time_loss = 0.0
    total_duration = 0.0
    max_arrival = 0.0

    for elem in tripinfos:
        waiting = float(elem.get("waitingTime", 0.0))
        loss = float(elem.get("timeLoss", 0.0))
        duration = float(elem.get("duration", 0.0))
        arrival = float(elem.get("arrival", 0.0))

        total_waiting_time += waiting
        total_time_loss += loss
        total_duration += duration
        if arrival > max_arrival:
            max_arrival = arrival

    avg_waiting = total_waiting_time / vehicle_count
    avg_loss = total_time_loss / vehicle_count
    avg_travel = total_duration / vehicle_count

    # Derive throughput in vehicles per episode hour
    if episode_seconds is not None and episode_seconds > 0:
        duration_hours = episode_seconds / 3600.0
    elif max_arrival > 0:
        duration_hours = max_arrival / 3600.0
    else:
        duration_hours = 1.0

    throughput = vehicle_count / duration_hours if duration_hours > 0 else float(vehicle_count)

    return TripInfoMetrics(
        vehicle_count=vehicle_count,
        average_waiting_time=avg_waiting,
        average_time_loss=avg_loss,
        average_travel_time=avg_travel,
        throughput=throughput,
    )


def parse_emissions(path: str | Path) -> EmissionMetrics:
    """Parse a SUMO emissions XML file.

    Args:
        path: UTF-8 XML file containing ``<vehicle>`` emission records or
            SUMO emissions-output elements with ``fuel``, ``CO2``/``co2`` and
            ``NOx``/``nox`` attributes.

    Returns:
        EmissionMetrics: Summed fuel, CO2, and NOx values in documented SUMO units.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    xml_path = Path(path)
    if not xml_path.exists():
        raise FileNotFoundError(f"Emissions file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    total_fuel = 0.0
    total_co2 = 0.0
    total_nox = 0.0

    # Search for all vehicle elements in the emissions XML tree
    for elem in root.iter("vehicle"):
        fuel_val = elem.get("fuel")
        if fuel_val is not None:
            total_fuel += float(fuel_val)

        co2_val = elem.get("CO2") or elem.get("co2")
        if co2_val is not None:
            total_co2 += float(co2_val)

        nox_val = elem.get("NOx") or elem.get("nox")
        if nox_val is not None:
            total_nox += float(nox_val)

    return EmissionMetrics(
        fuel=total_fuel,
        co2=total_co2,
        nox=total_nox,
    )


def merge_tripinfo_metrics(
    tripinfo: TripInfoMetrics,
    emissions: EmissionMetrics,
    *,
    controller: str = "unknown",
    scenario_id: str = "unknown",
    seed: int = 0,
) -> EpisodeMetrics:
    """Merge traffic and environmental metrics into the standard schema.

    Args:
        tripinfo: Parsed traffic values for one run.
        emissions: Parsed environmental values for the same run.
        controller: Optional controller name.
        scenario_id: Optional scenario identifier.
        seed: Optional evaluation seed.

    Returns:
        EpisodeMetrics: Combined record ready for controller/scenario metadata
        to be attached and written to evaluation results.
    """
    return EpisodeMetrics(
        controller=controller,
        scenario_id=scenario_id,
        seed=seed,
        average_waiting_time=tripinfo.average_waiting_time,
        average_queue_length=0.0,
        time_loss=tripinfo.average_time_loss,
        throughput=tripinfo.throughput,
        phase_switch_rate=0.0,
        min_green_violations=0,
        travel_time=tripinfo.average_travel_time,
        fuel=emissions.fuel,
        co2=emissions.co2,
        nox=emissions.nox,
    )
