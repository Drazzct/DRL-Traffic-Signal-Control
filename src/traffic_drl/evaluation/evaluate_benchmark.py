"""Deterministic benchmark evaluation on VA or unlocked TE scenarios."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from traffic_drl.contracts import BenchmarkReport, EpisodeMetrics, EnvironmentFactory, ScenarioSource


def evaluate_controller(model: Any, env: Any, *, controller_name: str, scenario_id: str, seed: int, deterministic: bool = True, episodes: int = 1) -> list[EpisodeMetrics]:
    """Roll out one controller and collect standard traffic metrics.

    Args:
        model: SB3 model or controller implementing ``predict``.
        env: Evaluation environment with the same observation normalization.
        controller_name: Label written to every returned metric record.
        scenario_id: Manifest ID for the route under evaluation.
        seed: Evaluation seed.
        deterministic: Disable exploration during benchmark inference.
        episodes: Number of repeated episodes for this scenario/seed.

    Returns:
        list[EpisodeMetrics]: One typed result per completed episode.
    """
    raise NotImplementedError


def evaluate_manifest(model: Any, env_factory: EnvironmentFactory, manifest: ScenarioSource, *, split: str, seeds: Iterable[int], controller_name: str, deterministic: bool = True, episodes_per_scenario: int = 1) -> list[EpisodeMetrics]:
    """Evaluate every selected scenario with matched seeds.

    Args:
        model: Controller being benchmarked.
        env_factory: Creates a fresh environment for each scenario and seed.
        manifest: Source of scenario records.
        split: ``VA`` for selection or unlocked ``TE`` for final evaluation.
        seeds: Seeds shared across all controllers for fair comparison.
        controller_name: Label for the output records.
        deterministic: Whether policy exploration is disabled.
        episodes_per_scenario: Repetitions per scenario and seed.

    TODO (SV3): prevent TE evaluation before the model and manifest are frozen.

    Returns:
        list[EpisodeMetrics]: Results for every selected scenario, seed, and episode.
    """
    raise NotImplementedError


def save_evaluation_results(records: Iterable[EpisodeMetrics], output_path: str | Path) -> None:
    """Persist raw per-episode records for later statistical analysis.

    Args:
        records: Typed metrics to serialize.
        output_path: UTF-8 ``.csv`` or ``.json`` destination. CSV must contain
            one row per ``EpisodeMetrics``; JSON must contain a list of objects
            with the same field names. Select the serializer from the suffix.

    File contract: write structured data with :mod:`csv` or :mod:`json`, never
    a Python ``repr``.

    Returns:
        None: The serialized results are written to ``output_path``.
    """
    raise NotImplementedError


def run_benchmark(config: Mapping[str, Any]) -> BenchmarkReport:
    """Run the configured fixed-time, actuated, and DRL comparison.

    Args:
        config: Benchmark configuration containing manifest, seeds, and artifacts.

    TODO (SV3): use identical routes and seeds for every controller and produce CI.

    Returns:
        BenchmarkReport: Final typed comparison report.
    """
    raise NotImplementedError
