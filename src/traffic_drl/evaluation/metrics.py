"""Metric schema, aggregation, and benchmark interpretation.

Metric categories
-----------------
- **Primary** (proposal metrics): ``average_waiting_time``,
  ``average_queue_length``, ``time_loss``.
- **Constraint** (must not regress): ``throughput``, ``phase_switch_rate``,
  ``min_green_violations``.
- **Supplementary** (environmental / operational): ``travel_time``,
  ``spillback``, ``recovery_time``, ``fuel``, ``co2``, ``nox``,
  ``inference_latency``.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from statistics import mean, stdev
from scipy.stats import t as t_dist  # type: ignore[import]
import math

from traffic_drl.contracts import BenchmarkReport, EpisodeMetrics, MetricSummary


PRIMARY_METRICS: tuple[str, ...] = (
    "average_waiting_time",
    "average_queue_length",
    "time_loss",
)
CONSTRAINT_METRICS: tuple[str, ...] = (
    "throughput",
    "phase_switch_rate",
    "min_green_violations",
)
ADDITIONAL_METRICS: tuple[str, ...] = (
    "travel_time",
    "spillback",
    "recovery_time",
    "fuel",
    "co2",
    "nox",
    "inference_latency",
)


def standard_metric_names() -> tuple[str, ...]:
    """Return the ordered tuple of all benchmark metric field names.

    Order: primary, constraint, supplementary — matching the proposal table.

    Returns:
        tuple[str, ...]: All metric field names in canonical order.
    """
    return PRIMARY_METRICS + CONSTRAINT_METRICS + ADDITIONAL_METRICS


def collect_episode_metrics(
    info: Mapping[str, float | int | bool | None],
    *,
    controller: str,
    scenario_id: str,
    seed: int,
) -> EpisodeMetrics:
    """Normalise one episode's ``info`` mapping into the benchmark schema.

    Required keys (must be present in *info*):
    ``average_waiting_time``, ``average_queue_length``, ``time_loss``,
    ``throughput``, ``phase_switch_rate``, ``min_green_violations``.

    Optional keys: all fields in :data:`ADDITIONAL_METRICS`.

    Args:
        info: Mapping from
            :class:`~traffic_drl.environment.wrappers.MetricsInfoWrapper`
            collected at episode end.
        controller: Stable controller label for the record.
        scenario_id: Manifest ID used for the episode.
        seed: Demand/SUMO seed for the episode.

    TODO (SV3): raise ``KeyError`` for missing primary metrics instead of
    silently substituting ``0.0``.

    Returns:
        EpisodeMetrics: Typed record with normalised metric values.
    """
    raise NotImplementedError


def aggregate_metrics(
    records: Iterable[EpisodeMetrics],
) -> tuple[MetricSummary, ...]:
    """Aggregate per-episode records by (controller, scenario_id) group.

    For each group, computes mean, standard deviation, and 95% confidence
    interval for every metric in :func:`standard_metric_names`.

    Args:
        records: Typed episode results collected with matched scenario/seed
            settings across all controllers.

    TODO (SV3): report mean, standard deviation, and 95% CI.

    Returns:
        tuple[MetricSummary, ...]: One aggregate per (controller, scenario_id) group.
    """
    raise NotImplementedError


def confidence_interval(
    values: Iterable[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a Student-t confidence interval for a metric across seeds.

    Uses the Student-t distribution because the number of evaluation seeds
    is typically small (3–5).

    Args:
        values: Numeric observations from independent evaluation seeds.
        confidence: Requested interval level; defaults to ``0.95``.

    Returns:
        tuple[float, float]: ``(lower, upper)`` confidence bounds.

    Raises:
        ValueError: If fewer than two values are provided.
    """


def interpret_results(
    records: Iterable[EpisodeMetrics],
    *,
    baseline_controller: str = "fixed_time",
) -> BenchmarkReport:
    """Interpret metrics against a named baseline controller.

    For each primary metric, computes the percentage improvement of every
    non-baseline controller relative to *baseline_controller*.  Counts
    constraint violations (e.g. ``min_green_violations > 0``).

    Args:
        records: Episode metrics for the baseline and all learned controllers.
        baseline_controller: Controller name used for percentage comparisons.

    TODO (SV3): flag regressions in throughput, phase stability, and min-green
    safety as hard failures rather than just reporting the percentage.

    Returns:
        BenchmarkReport: Aggregate summaries, improvements, and constraint violations.
    """
    raise NotImplementedError
