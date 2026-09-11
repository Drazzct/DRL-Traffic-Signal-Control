"""Benchmark evaluation and metric interpretation APIs.

Public API
----------
- :func:`~traffic_drl.evaluation.evaluate_benchmark.evaluate_controller` — single-controller episode loop.
- :func:`~traffic_drl.evaluation.evaluate_benchmark.evaluate_manifest` — multi-scenario/seed loop.
- :func:`~traffic_drl.evaluation.evaluate_benchmark.save_evaluation_results` — persist raw metrics.
- :func:`~traffic_drl.evaluation.evaluate_benchmark.run_benchmark` — full three-controller comparison.
- :func:`~traffic_drl.evaluation.metrics.standard_metric_names` — canonical metric field names.
- :func:`~traffic_drl.evaluation.metrics.collect_episode_metrics` — info dict → EpisodeMetrics.
- :func:`~traffic_drl.evaluation.metrics.aggregate_metrics` — per-group statistics.
- :func:`~traffic_drl.evaluation.metrics.confidence_interval` — Student-t CI.
- :func:`~traffic_drl.evaluation.metrics.interpret_results` — benchmark report generation.
- :func:`~traffic_drl.evaluation.parse_tripinfo.parse_tripinfo` — SUMO tripinfo XML parser.
- :func:`~traffic_drl.evaluation.parse_tripinfo.parse_emissions` — SUMO emissions XML parser.
- :func:`~traffic_drl.evaluation.test.check_environment` — SB3 Gymnasium contract checker.
- :func:`~traffic_drl.evaluation.test.run_smoke_test` — quick reset/step/close test.
"""
from traffic_drl.evaluation.evaluate_benchmark import (
    evaluate_controller,
    evaluate_manifest,
    save_evaluation_results,
    run_benchmark,
)
from traffic_drl.evaluation.metrics import (
    standard_metric_names,
    collect_episode_metrics,
    aggregate_metrics,
    confidence_interval,
    interpret_results,
    PRIMARY_METRICS,
    CONSTRAINT_METRICS,
    ADDITIONAL_METRICS,
)
from traffic_drl.evaluation.parse_tripinfo import (
    parse_tripinfo,
    parse_emissions,
    merge_tripinfo_metrics,
)
from traffic_drl.evaluation.test import check_environment, run_smoke_test

__all__ = [
    # evaluate_benchmark
    "evaluate_controller",
    "evaluate_manifest",
    "save_evaluation_results",
    "run_benchmark",
    # metrics
    "standard_metric_names",
    "collect_episode_metrics",
    "aggregate_metrics",
    "confidence_interval",
    "interpret_results",
    "PRIMARY_METRICS",
    "CONSTRAINT_METRICS",
    "ADDITIONAL_METRICS",
    # parse_tripinfo
    "parse_tripinfo",
    "parse_emissions",
    "merge_tripinfo_metrics",
    # test
    "check_environment",
    "run_smoke_test",
]
