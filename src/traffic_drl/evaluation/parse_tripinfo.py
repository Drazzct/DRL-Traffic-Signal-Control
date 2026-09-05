"""SUMO tripinfo and emissions parsing API."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from traffic_drl.contracts import EmissionMetrics, EpisodeMetrics, TripInfoMetrics


def parse_tripinfo(path: str | Path) -> TripInfoMetrics:
    """Parse SUMO ``tripinfo.xml`` into traffic metrics.

    Args:
        path: UTF-8 XML file containing one ``<tripinfo>`` element per vehicle,
            with SUMO attributes such as ``duration``, ``waitingTime``,
            ``timeLoss``, and ``depart``/``arrival``.

    File contract: parse with :mod:`xml.etree.ElementTree`, aggregate numeric
    attributes across ``tripinfo`` elements, and derive throughput from the
    number of completed vehicles and episode duration.

    TODO (SV3): define missing-attribute handling and units in the metric contract.

    Returns:
        TripInfoMetrics: Vehicle count and averaged waiting, loss, travel-time,
        and throughput values.
    """
    raise NotImplementedError


def parse_emissions(path: str | Path) -> EmissionMetrics:
    """Parse a SUMO emissions XML file.

    Args:
        path: UTF-8 XML file containing ``<vehicle>`` emission records or
            SUMO emissions-output elements with ``fuel``, ``CO2``/``co2`` and
            ``NOx``/``nox`` attributes.

    File contract: parse with :mod:`xml.etree.ElementTree`, sum each numeric
    pollutant field, and preserve SUMO's documented units for reporting.

    TODO (SV1): confirm the exact emissions schema and unit conversion used by SUMO.

    Returns:
        EmissionMetrics: Summed fuel, CO2, and NOx values in documented SUMO units.
    """
    raise NotImplementedError


def merge_tripinfo_metrics(tripinfo: TripInfoMetrics, emissions: EmissionMetrics) -> EpisodeMetrics:
    """Merge traffic and environmental metrics into the standard schema.

    Args:
        tripinfo: Parsed traffic values for one run.
        emissions: Parsed environmental values for the same run.

    Returns:
        EpisodeMetrics: Combined record ready for controller/scenario metadata
        to be attached and written to evaluation results.
    """
    raise NotImplementedError
