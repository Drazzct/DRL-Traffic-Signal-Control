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
    missing_primary = [m for m in PRIMARY_METRICS if m not in info]
    if missing_primary:
        raise ValueError(f"Missing required primary metric(s) in info mapping: {missing_primary}")

    # Extract required primary metrics
    average_waiting_time = float(info["average_waiting_time"])
    average_queue_length = float(info["average_queue_length"])
    time_loss = float(info["time_loss"])

    # Extract constraint metrics with safe conversion and standard defaults
    throughput = float(info.get("throughput", 0.0))
    phase_switch_rate = float(info.get("phase_switch_rate", 0.0))
    min_green_violations = int(info.get("min_green_violations", 0))

    # Extract supplementary metrics if present
    def _opt_float(key: str) -> float | None:
        val = info.get(key)
        return float(val) if val is not None else None

    travel_time = _opt_float("travel_time")
    spillback = _opt_float("spillback")
    recovery_time = _opt_float("recovery_time")
    fuel = _opt_float("fuel")
    co2 = _opt_float("co2")
    nox = _opt_float("nox")
    inference_latency = _opt_float("inference_latency")

    return EpisodeMetrics(
        controller=controller,
        scenario_id=scenario_id,
        seed=seed,
        average_waiting_time=average_waiting_time,
        average_queue_length=average_queue_length,
        time_loss=time_loss,
        throughput=throughput,
        phase_switch_rate=phase_switch_rate,
        min_green_violations=min_green_violations,
        travel_time=travel_time,
        spillback=spillback,
        recovery_time=recovery_time,
        fuel=fuel,
        co2=co2,
        nox=nox,
        inference_latency=inference_latency,
    )


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
    """
    data = [float(v) for v in values]
    n = len(data)
    if n == 0:
        return (0.0, 0.0)
    mean_val = sum(data) / n
    if n == 1:
        return (float(mean_val), float(mean_val))

    variance = sum((x - mean_val) ** 2 for x in data) / (n - 1)
    std_err = math.sqrt(variance) / math.sqrt(n)

    try:
        from scipy import stats

        t_crit = float(stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1))
    except Exception:
        # Analytical approximation fallback
        t_crit = 1.96 if abs(confidence - 0.95) < 1e-4 else 2.0

    margin = t_crit * std_err
    return (float(mean_val - margin), float(mean_val + margin))


def aggregate_metrics(
    records: Iterable[EpisodeMetrics],
) -> tuple[MetricSummary, ...]:
    """Aggregate per-episode records by (controller, scenario_id) group.

    For each group, computes mean, standard deviation, and 95% confidence
    interval for every metric in :func:`standard_metric_names`.

    Args:
        records: Typed episode results collected with matched scenario/seed
            settings across all controllers.

    Returns:
        tuple[MetricSummary, ...]: One aggregate per (controller, scenario_id) group.
    """
    groups: dict[tuple[str, str], list[EpisodeMetrics]] = defaultdict(list)
    for r in records:
        groups[(r.controller, r.scenario_id)].append(r)

    all_metric_keys = standard_metric_names()
    summaries: list[MetricSummary] = []

    for (controller, scenario_id), recs in groups.items():
        sample_count = len(recs)
        means: dict[str, float] = {}
        standard_deviations: dict[str, float] = {}
        confidence_intervals: dict[str, tuple[float, float]] = {}

        for key in all_metric_keys:
            vals = [getattr(r, key) for r in recs if getattr(r, key, None) is not None]
            if not vals:
                continue
            float_vals = [float(v) for v in vals]
            m = sum(float_vals) / len(float_vals)
            means[key] = float(m)
            if len(float_vals) > 1:
                var = sum((x - m) ** 2 for x in float_vals) / (len(float_vals) - 1)
                standard_deviations[key] = math.sqrt(var)
            else:
                standard_deviations[key] = 0.0
            confidence_intervals[key] = confidence_interval(float_vals, confidence=0.95)

        summaries.append(
            MetricSummary(
                controller=controller,
                scenario_id=scenario_id,
                sample_count=sample_count,
                means=means,
                standard_deviations=standard_deviations,
                confidence_intervals=confidence_intervals,
            )
        )

    return tuple(summaries)


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
    records_list = list(records)
    summaries = aggregate_metrics(records_list)

    baseline_by_scenario: dict[str, MetricSummary] = {
        s.scenario_id: s for s in summaries if s.controller == baseline_controller
    }

    # Track constraint violations per controller
    constraint_violations: dict[str, int] = defaultdict(int)
    for r in records_list:
        if r.controller not in constraint_violations:
            constraint_violations[r.controller] = 0
        if r.min_green_violations > 0:
            constraint_violations[r.controller] += r.min_green_violations

    improvements: dict[str, dict[str, float]] = {}

    for s in summaries:
        if s.controller == baseline_controller:
            continue
        base_s = baseline_by_scenario.get(s.scenario_id)
        group_key = f"{s.controller}/{s.scenario_id}"
        metric_improvements: dict[str, float] = {}

        if base_s is not None:
            for metric, cur_val in s.means.items():
                base_val = base_s.means.get(metric)
                if base_val is None:
                    continue

                if metric == "throughput":
                    # Higher is better: positive improvement means throughput increased
                    if base_val != 0.0:
                        imp = (cur_val - base_val) / base_val * 100.0
                    else:
                        imp = 0.0
                    # Flag throughput regression as a constraint concern
                    if cur_val < base_val:
                        constraint_violations[s.controller] += 1
                else:
                    # Delay/queue/loss/emissions: lower is better -> positive means reduced delay
                    if base_val != 0.0:
                        imp = (base_val - cur_val) / base_val * 100.0
                    else:
                        imp = 0.0

                metric_improvements[metric] = float(imp)

        improvements[group_key] = metric_improvements

    return BenchmarkReport(
        baseline_controller=baseline_controller,
        summaries=summaries,
        improvements=improvements,
        constraint_violations=dict(constraint_violations),
    )
