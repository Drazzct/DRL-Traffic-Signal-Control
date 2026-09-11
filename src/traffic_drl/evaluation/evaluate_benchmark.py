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
from dataclasses import asdict, fields
import json
from pathlib import Path
import time
from typing import Any, Iterable

import gymnasium as gym
import numpy as np

from traffic_drl.contracts import (
    BenchmarkReport,
    Controller,
    EnvironmentFactory,
    EpisodeMetrics,
    ScenarioSource,
)
from traffic_drl.config import EvalConfig, load_eval_config
from traffic_drl.evaluation.metrics import collect_episode_metrics, interpret_results
from traffic_drl.evaluation.parse_tripinfo import parse_tripinfo


def evaluate_controller(
    controller: Controller | None = None,
    env: gym.Env | None = None,
    *,
    controller_name: str,
    scenario_id: str,
    seed: int,
    deterministic: bool = True,
    episodes: int = 1,
    max_steps: int | None = 1000,
    tripinfo_path: str | Path | None = None,
    model: Controller | None = None,
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
        max_steps: Maximum step limit safeguard per episode (default: 1000).
        tripinfo_path: Optional path to SUMO tripinfo.xml file.
        model: Backwards-compatibility alias for controller.

    Returns:
        list[EpisodeMetrics]: One typed result per completed episode.
    """
    actual_controller = controller if controller is not None else model
    if actual_controller is None:
        raise ValueError("A controller or model must be provided.")
    if env is None:
        raise ValueError("A Gymnasium environment must be provided.")

    results: list[EpisodeMetrics] = []

    for ep in range(episodes):
        current_seed = seed + ep
        if hasattr(actual_controller, "reset") and callable(actual_controller.reset):
            actual_controller.reset()

        reset_ret = env.reset(seed=current_seed)
        if isinstance(reset_ret, tuple) and len(reset_ret) == 2:
            obs, info = reset_ret
        else:
            obs, info = reset_ret, {}

        last_info: dict[str, Any] = dict(info) if isinstance(info, dict) else {}
        terminated = False
        truncated = False
        step_count = 0
        latencies: list[float] = []

        while not (terminated or truncated):
            t0 = time.perf_counter()
            action = actual_controller.predict(obs, deterministic=deterministic)
            latencies.append(time.perf_counter() - t0)

            # Handle models where predict returns (action, state)
            if isinstance(action, tuple):
                action = action[0]

            step_ret = env.step(action)
            if len(step_ret) == 5:
                obs, reward, terminated, truncated, s_info = step_ret
            else:
                obs, reward, done, s_info = step_ret
                terminated, truncated = bool(done), False

            step_count += 1
            if isinstance(s_info, dict):
                last_info.update(s_info)

            if max_steps is not None and step_count >= max_steps:
                break

        if latencies:
            last_info["inference_latency"] = float(np.mean(latencies))

        # Check tripinfo XML if provided
        if tripinfo_path is not None and Path(tripinfo_path).exists():
            t_m = parse_tripinfo(tripinfo_path)
            last_info.setdefault("average_waiting_time", t_m.average_waiting_time)
            last_info.setdefault("time_loss", t_m.average_time_loss)
            last_info.setdefault("travel_time", t_m.average_travel_time)
            last_info.setdefault("throughput", t_m.throughput)

        # Fallback defaults for missing primary keys so dummy envs don't crash
        last_info.setdefault("average_waiting_time", 0.0)
        last_info.setdefault("average_queue_length", 0.0)
        last_info.setdefault("time_loss", 0.0)
        last_info.setdefault("throughput", 0.0)
        last_info.setdefault("phase_switch_rate", 0.0)
        last_info.setdefault("min_green_violations", 0)

        ep_metric = collect_episode_metrics(
            last_info,
            controller=controller_name,
            scenario_id=scenario_id,
            seed=current_seed,
        )
        results.append(ep_metric)

    return results


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
    allow_te: bool = False,
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
        allow_te: Explicit flag to allow evaluating on the held-out TE split.

    Returns:
        list[EpisodeMetrics]: All results across every scenario, seed, and episode.
    """
    normalized_split = split.strip().upper()
    if normalized_split == "TE" and not allow_te:
        raise PermissionError(
            "Held-out test split (TE) is locked until the model and manifest are formally frozen."
        )

    records = manifest.records_for_split(split)
    if not records and split != normalized_split:
        records = manifest.records_for_split(normalized_split)

    all_metrics: list[EpisodeMetrics] = []
    seeds_list = list(seeds)

    for record in records:
        for seed in seeds_list:
            env = env_factory(record, seed)
            try:
                metrics = evaluate_controller(
                    controller,
                    env,
                    controller_name=controller_name,
                    scenario_id=record.scenario_id,
                    seed=seed,
                    deterministic=deterministic,
                    episodes=episodes_per_scenario,
                )
                all_metrics.extend(metrics)
            finally:
                if hasattr(env, "close") and callable(env.close):
                    env.close()

    return all_metrics


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

    Returns:
        None.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    suffix = dest.suffix.lower()

    records_list = list(records)
    dict_records = [asdict(r) for r in records_list]

    if suffix == ".csv":
        fieldnames = [f.name for f in fields(EpisodeMetrics)]
        with dest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in dict_records:
                writer.writerow(r)
    elif suffix == ".json":
        with dest.open("w", encoding="utf-8") as f:
            json.dump(dict_records, f, indent=2)
    else:
        raise ValueError(
            f"Unsupported output file extension: '{suffix}'. Supported formats are '.csv' and '.json'."
        )


def run_benchmark(
    config: EvalConfig | str | Path = "configs/evaluation/phase1_validation.yaml",
    *,
    allow_te: bool = False,
    scenario_source: ScenarioSource | None = None,
    env_factory: EnvironmentFactory | None = None,
    controllers: dict[str, Controller] | None = None,
) -> BenchmarkReport:
    """Run the full fixed-time, actuated, and DRL controller comparison.

    Loads the manifest, creates an environment factory for each controller,
    calls :func:`evaluate_manifest` for each, then calls
    :func:`~traffic_drl.evaluation.metrics.interpret_results` to produce the
    final report.

    Args:
        config: Typed evaluation configuration or path to its YAML file.
        allow_te: Allow evaluation on held-out test split (TE).
        scenario_source: Optional pre-loaded ScenarioSource (defaults to loading manifest_path).
        env_factory: Optional custom EnvironmentFactory.
        controllers: Optional dict of named controllers to evaluate. Defaults
            to the standard suite: fixed_time, max_pressure, actuated, drl.

    Returns:
        BenchmarkReport: Final typed comparison report.
    """
    if isinstance(config, (str, Path)):
        config = load_eval_config(config)

    split = config.benchmark.split
    if split.strip().upper() == "TE" and not allow_te:
        raise PermissionError(
            "Held-out test split (TE) is locked until the model and manifest are formally frozen."
        )

    # 1. Resolve scenario source
    manifest: ScenarioSource
    if scenario_source is not None:
        manifest = scenario_source
    else:
        from traffic_drl.environment.scenario_factory import ScenarioManifest

        manifest_p = Path(config.manifest_path)
        if not manifest_p.exists():
            raise FileNotFoundError(f"Scenario manifest not found: {config.manifest_path}")
        manifest = ScenarioManifest.from_csv(manifest_p)

    # 2. Resolve environment factory
    factory: EnvironmentFactory
    if env_factory is not None:
        factory = env_factory
    else:
        from traffic_drl.environment.make_env import create_sumo_env
        from traffic_drl.config import load_env_config

        env_cfg = load_env_config(config.env_config_path)
        factory = lambda record, seed: create_sumo_env(env_cfg, record.route_file, seed=seed)

    # 3. Resolve controllers
    ctrl_suite: dict[str, Controller] = {}
    if controllers is not None:
        ctrl_suite = dict(controllers)
    else:
        # Load baseline controllers
        from traffic_drl.baselines import (
            FixedTimeController,
            MaxPressureController,
            ActuatedController,
        )

        try:
            ctrl_suite["fixed_time"] = FixedTimeController()
        except Exception:
            pass

        try:
            ctrl_suite["max_pressure"] = MaxPressureController()
        except Exception:
            pass

        try:
            ctrl_suite["actuated"] = ActuatedController()
        except Exception:
            pass

        # Load trained DRL model if checkpoint is configured and exists
        ckpt = Path(config.artifacts.model_checkpoint)
        if ckpt.exists():
            try:
                from stable_baselines3 import DQN

                ctrl_suite["dqn"] = DQN.load(str(ckpt))
            except Exception:
                pass

    if not ctrl_suite:
        raise ValueError("No controllers available to benchmark.")

    # 4. Run evaluation across all controllers
    all_records: list[EpisodeMetrics] = []
    for c_name, c_inst in ctrl_suite.items():
        recs = evaluate_manifest(
            c_inst,
            factory,
            manifest,
            split=split,
            seeds=config.evaluation_seeds,
            controller_name=c_name,
            deterministic=config.benchmark.deterministic,
            episodes_per_scenario=config.benchmark.episodes_per_scenario,
            allow_te=allow_te,
        )
        all_records.extend(recs)

    # 5. Persist results if reporting output_dir is specified
    if config.reporting.output_dir:
        out_dir = Path(config.reporting.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_evaluation_results(all_records, out_dir / "metrics.csv")
        save_evaluation_results(all_records, out_dir / "metrics.json")

    # 6. Return interpreted benchmark report
    baseline = "fixed_time" if "fixed_time" in ctrl_suite else next(iter(ctrl_suite))
    return interpret_results(all_records, baseline_controller=baseline)
