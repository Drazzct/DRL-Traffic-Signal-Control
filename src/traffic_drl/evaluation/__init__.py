"""Benchmark evaluation and metric interpretation APIs."""
from traffic_drl.evaluation.evaluate_benchmark import (
    evaluate_controller,
    evaluate_manifest,
    run_benchmark,
    save_evaluation_results,
)
from traffic_drl.evaluation.metrics import (
    ADDITIONAL_METRICS,
    CONSTRAINT_METRICS,
    PRIMARY_METRICS,
    aggregate_metrics,
    collect_episode_metrics,
    confidence_interval,
    interpret_results,
    standard_metric_names,
)
from traffic_drl.evaluation.parse_tripinfo import (
    merge_tripinfo_metrics,
    parse_emissions,
    parse_tripinfo,
)
from traffic_drl.evaluation.plots import (
    plot_learning_curve,
    plot_metric_comparison,
    plot_queue_heatmap,
)
from traffic_drl.evaluation.test import (
    check_environment,
    run_smoke_test,
)

__all__ = [
    "PRIMARY_METRICS",
    "CONSTRAINT_METRICS",
    "ADDITIONAL_METRICS",
    "standard_metric_names",
    "collect_episode_metrics",
    "aggregate_metrics",
    "confidence_interval",
    "interpret_results",
    "parse_tripinfo",
    "parse_emissions",
    "merge_tripinfo_metrics",
    "evaluate_controller",
    "evaluate_manifest",
    "save_evaluation_results",
    "run_benchmark",
    "plot_metric_comparison",
    "plot_learning_curve",
    "plot_queue_heatmap",
    "check_environment",
    "run_smoke_test",
]
