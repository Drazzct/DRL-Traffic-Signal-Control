"""Deterministic benchmark evaluation on VA or unlocked TE scenarios.

All three baselines — fixed-time, actuated, and DRL — are evaluated through
the same :func:`evaluate_controller` loop.  There is no special-case path for
the fixed-time baseline.

Typical usage
-------------
::

    from traffic_drl.evaluation.evaluate_benchmark import (
        evaluate_controller,
        evaluate_manifest,
        run_benchmark,
    )
    from traffic_drl.baselines import FixedTimeController, make_fixed_time_env
    from traffic_drl.config import load_eval_config

    cfg = load_eval_config("configs/evaluation/phase1_validation.yaml")
    manifest = ScenarioManifest.from_csv(cfg.manifest_path)
    controller = FixedTimeController()
    factory = lambda record, seed: make_fixed_time_env(record, env_config)
    results = evaluate_manifest(
        controller, factory, manifest,
        split=cfg.benchmark.split,
        seeds=cfg.evaluation_seeds,
        controller_name="fixed_time",
    )
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import gymnasium as gym

from traffic_drl.contracts import (
    BenchmarkReport,
    Controller,
    EnvironmentFactory,
    EpisodeMetrics,
    ScenarioSource,
)
from traffic_drl.config import EvalConfig, load_eval_config


def evaluate_controller(
    controller: Controller,
    env: gym.Env,
    *,
    controller_name: str,
    scenario_id: str,
    seed: int,
    deterministic: bool = True,
    episodes: int = 1,
) -> list[EpisodeMetrics]:
    """Roll out one controller and collect standard traffic metrics.

    This function is the single evaluation loop used by **all** controllers:
    fixed-time, actuated, max-pressure, and DRL.

    Args:
        controller: Any object satisfying the
            :class:`~traffic_drl.contracts.Controller` protocol.
        env: A fully configured Gymnasium environment for this scenario/seed,
            with the same observation normalisation used during training (for
            DRL) or no normalisation (for heuristic baselines).
        controller_name: Label written to every returned metric record.
        scenario_id: Manifest ID for the route under evaluation.
        seed: Evaluation seed; used to reset the environment reproducibly.
        deterministic: Disable policy exploration during inference.
        episodes: Number of repeated episodes for this scenario/seed.

    TODO (SV3): implement the reset → predict → step loop; read metrics from
    the ``info`` dict populated by
    :class:`~traffic_drl.environment.wrappers.MetricsInfoWrapper`; measure
    inference latency per step.

    Returns:
        list[EpisodeMetrics]: One typed result per completed episode.
    """
    raise NotImplementedError


def evaluate_manifest(
    controller: Controller,
    env_factory: EnvironmentFactory,
    manifest: ScenarioSource,
    *,
    split: str,
    seeds: Iterable[int],
    controller_name: str,
    deterministic: bool = True,
    episodes_per_scenario: int = 1,
) -> list[EpisodeMetrics]:
    """Evaluate every scenario in *split* with each seed in *seeds*.

    For each (scenario, seed) pair the factory creates a fresh environment,
    then :func:`evaluate_controller` runs *episodes_per_scenario* episodes.

    Args:
        controller: Controller satisfying the
            :class:`~traffic_drl.contracts.Controller` protocol.
        env_factory: Factory satisfying the
            :class:`~traffic_drl.contracts.EnvironmentFactory` protocol; creates
            one fresh environment per (scenario, seed) pair.
        manifest: Scenario source satisfying the
            :class:`~traffic_drl.contracts.ScenarioSource` protocol.
        split: ``VA`` for model selection, or unlocked ``TE`` for final
            evaluation.
        seeds: Seeds shared across all controllers for a fair comparison.
        controller_name: Label for the output records.
        deterministic: Whether policy exploration is disabled.
        episodes_per_scenario: Episode repetitions per (scenario, seed) pair.

    TODO (SV3): prevent TE evaluation before the model and manifest are frozen;
    close the environment after each (scenario, seed) pair.

    Returns:
        list[EpisodeMetrics]: All results across every scenario, seed, and episode.
    """
    raise NotImplementedError


def save_evaluation_results(
    records: Iterable[EpisodeMetrics],
    output_path: str | Path,
) -> None:
    """Persist raw per-episode records for later statistical analysis.

    Selects the serialiser from the file suffix:
    - ``.csv``: one row per ``EpisodeMetrics`` with a header.
    - ``.json``: a JSON array of objects with the same field names.

    Args:
        records: Typed episode metrics to serialise.
        output_path: Destination file.  The parent directory is created if it
            does not exist.

    TODO (SV3): decide whether to write CSV or JSON as the canonical output
    format and document it in the evaluation config.

    Returns:
        None.
    """
    raise NotImplementedError


def run_benchmark(
    config: EvalConfig | str | Path = "configs/evaluation/phase1_validation.yaml",
) -> BenchmarkReport:
    """Run the full fixed-time, actuated, and DRL controller comparison.

    Loads the manifest, creates an environment factory for each controller,
    calls :func:`evaluate_manifest` for each, then calls
    :func:`~traffic_drl.evaluation.metrics.interpret_results` to produce the
    final report.

    Args:
        config: Typed evaluation configuration or path to its YAML file.

    TODO (SV3): use identical routes and seeds for every controller; produce
    95% confidence intervals; prevent TE access before the model is frozen.

    Returns:
        BenchmarkReport: Final typed comparison report.
    """
    if isinstance(config, (str, Path)):
        config = load_eval_config(config)

    raise NotImplementedError
