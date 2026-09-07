"""Metric schema and aggregation for traffic-controller comparisons."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from traffic_drl.contracts import BenchmarkReport, EpisodeMetrics, MetricSummary

PRIMARY_METRICS = ("average_waiting_time", "average_queue_length", "time_loss")
CONSTRAINT_METRICS = ("throughput", "phase_switch_rate", "min_green_violations")
ADDITIONAL_METRICS = ("travel_time", "spillback", "recovery_time", "fuel", "co2", "nox", "inference_latency")


def standard_metric_names() -> tuple[str, ...]:
    """Return primary, constraint, and supplementary metric names.

    TODO (SV3): keep this schema synchronized with the final evaluation table.

    Returns:
        tuple[str, ...]: Primary, constraint, and supplementary metric field names.
    """
    raise NotImplementedError


def collect_episode_metrics(info: Mapping[str, Any], *, controller: str, scenario_id: str, seed: int) -> EpisodeMetrics:
    """Normalize one episode's info mapping into the benchmark schema.

    Args:
        info: Environment ``info`` values collected at episode end.
        controller: Stable name of the controller being evaluated.
        scenario_id: Manifest ID used for the episode.
        seed: Demand/SUMO seed used for the episode.

    TODO (SV3): reject missing primary metrics instead of silently inventing values.

    Returns:
        EpisodeMetrics: Typed record containing normalized metric values.
    """
    raise NotImplementedError


def aggregate_metrics(records: Iterable[EpisodeMetrics]) -> tuple[MetricSummary, ...]:
    """Aggregate per-episode records by controller and scenario.

    Args:
        records: Typed episode results collected with matched scenario/seed settings.

    TODO (SV3): report mean, standard deviation, and 95% confidence intervals.

    Returns:
        tuple[MetricSummary, ...]: One aggregate per controller/scenario group.
    """
    raise NotImplementedError


def confidence_interval(values: Iterable[float], confidence: float = 0.95) -> tuple[float, float]:
    """Return a confidence interval for one metric across seeds.

    Args:
        values: Numeric observations from independent evaluation seeds.
        confidence: Requested interval level, normally ``0.95``.

    Returns:
        tuple[float, float]: Lower and upper confidence bounds.
    """
    raise NotImplementedError


def interpret_results(records: Iterable[EpisodeMetrics], *, baseline_controller: str = "fixed_time") -> BenchmarkReport:
    """Interpret metrics against a baseline controller.

    Args:
        records: Episode metrics for baseline and learned controllers.
        baseline_controller: Controller name used for percentage comparisons.

    TODO (SV3): flag regressions in throughput, phase stability, and min-green safety.

    Returns:
        BenchmarkReport: Aggregate summaries, improvements, and constraint violations.
    """
    raise NotImplementedError
