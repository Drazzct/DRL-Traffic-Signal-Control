"""Deterministic benchmark evaluation on VA or unlocked TE scenarios."""
from __future__ import annotations

import csv
from dataclasses import asdict, fields
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from traffic_drl.contracts import BenchmarkReport, EpisodeMetrics, EnvironmentFactory, ScenarioSource
from traffic_drl.evaluation.metrics import collect_episode_metrics, interpret_results


def evaluate_controller(
    model: Any,
    env: Any,
    *,
    controller_name: str,
    scenario_id: str,
    seed: int,
    deterministic: bool = True,
    episodes: int = 1,
    tripinfo_path: str | Path | None = None,
) -> list[EpisodeMetrics]:
    """Roll out one controller and collect standard traffic metrics.

    Args:
        model: SB3 model or controller implementing ``predict`` (or callable).
        env: Evaluation environment with the same observation normalization.
        controller_name: Label written to every returned metric record.
        scenario_id: Manifest ID for the route under evaluation.
        seed: Evaluation seed.
        deterministic: Disable exploration during benchmark inference.
        episodes: Number of repeated episodes for this scenario/seed.
        tripinfo_path: Optional path to SUMO tripinfo.xml file.

    Returns:
        list[EpisodeMetrics]: One typed result per completed episode.
    """
    results: list[EpisodeMetrics] = []

    for ep in range(episodes):
        if hasattr(model, "reset") and callable(model.reset):
            model.reset()

        reset_ret = env.reset(seed=seed + ep if seed is not None else None)
        if isinstance(reset_ret, tuple) and len(reset_ret) == 2:
            obs, info = reset_ret
        else:
            obs, info = reset_ret, {}

        last_info: dict[str, Any] = dict(info) if isinstance(info, Mapping) else {}
        terminated = False
        truncated = False
        steps = 0
        phase_switches = 0
        prev_action = None

        while not (terminated or truncated):
            if hasattr(model, "predict") and callable(model.predict):
                pred = model.predict(obs, deterministic=deterministic)
                if isinstance(pred, tuple):
                    action = pred[0]
                else:
                    action = pred
            elif callable(model):
                action = model(obs)
            else:
                action = env.action_space.sample()

            step_ret = env.step(action)
            if len(step_ret) == 5:
                obs, reward, terminated, truncated, step_info = step_ret
            else:
                obs, reward, done, step_info = step_ret
                terminated, truncated = bool(done), False

            steps += 1
            if prev_action is not None and action != prev_action:
                phase_switches += 1
            prev_action = action

            if isinstance(step_info, Mapping):
                last_info.update(step_info)

        # Populate calculated control statistics if not already emitted by env
        if "phase_switch_rate" not in last_info:
            last_info["phase_switch_rate"] = phase_switches / max(1, steps)
        if "min_green_violations" not in last_info:
            last_info["min_green_violations"] = 0

        # If tripinfo XML output path is available, try parsing metrics from it
        t_path = tripinfo_path or last_info.get("tripinfo_output") or last_info.get("tripinfo_path")
        if t_path and Path(t_path).exists():
            try:
                from traffic_drl.evaluation.parse_tripinfo import parse_tripinfo

                t_metrics = parse_tripinfo(t_path)
                last_info["average_waiting_time"] = t_metrics.average_waiting_time
                last_info["time_loss"] = t_metrics.average_time_loss
                last_info["throughput"] = t_metrics.throughput
                last_info["travel_time"] = t_metrics.average_travel_time
                last_info.setdefault("average_queue_length", 0.0)
            except Exception:
                pass

        # Handle SUMO-RL environment native metric names if primary metrics not yet set
        if "average_waiting_time" not in last_info:
            if "system_mean_waiting_time" in last_info:
                last_info["average_waiting_time"] = float(last_info["system_mean_waiting_time"])
            elif "system_total_waiting_time" in last_info:
                last_info["average_waiting_time"] = float(last_info["system_total_waiting_time"])
            else:
                last_info["average_waiting_time"] = 0.0

        if "average_queue_length" not in last_info:
            if "system_total_stopped" in last_info:
                last_info["average_queue_length"] = float(last_info["system_total_stopped"])
            else:
                last_info["average_queue_length"] = 0.0

        if "time_loss" not in last_info:
            last_info["time_loss"] = float(last_info.get("average_waiting_time", 0.0))

        record = collect_episode_metrics(
            last_info,
            controller=controller_name,
            scenario_id=scenario_id,
            seed=seed + ep if seed is not None else 0,
        )
        results.append(record)

    return results


def evaluate_manifest(
    model: Any,
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
        allow_te: Explicit permission flag to evaluate held-out TE split.

    Returns:
        list[EpisodeMetrics]: Results for every selected scenario, seed, and episode.

    Raises:
        PermissionError: If attempting to evaluate locked TE split without permission.
    """
    normalized_split = split.strip().upper()
    if normalized_split == "TE" and not allow_te:
        raise PermissionError(
            "Held-out test split (TE) is locked until the model and manifest are formally frozen."
        )

    records = manifest.records_for_split(split)
    all_metrics: list[EpisodeMetrics] = []
    seeds_list = list(seeds)

    for record in records:
        for seed in seeds_list:
            env = env_factory(record, seed)
            try:
                metrics = evaluate_controller(
                    model,
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


def save_evaluation_results(records: Iterable[EpisodeMetrics], output_path: str | Path) -> None:
    """Persist raw per-episode records for later statistical analysis.

    Args:
        records: Typed metrics to serialize.
        output_path: UTF-8 ``.csv`` or ``.json`` destination. CSV must contain
            one row per ``EpisodeMetrics``; JSON must contain a list of objects
            with the same field names. Select the serializer from the suffix.

    Returns:
        None: The serialized results are written to ``output_path``.

    Raises:
        ValueError: If file suffix is not ``.csv`` or ``.json``.
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


def run_benchmark(config: Mapping[str, Any]) -> BenchmarkReport:
    """Run the configured fixed-time, actuated, and DRL comparison.

    Args:
        config: Benchmark configuration containing manifest, seeds, and artifacts.

    Returns:
        BenchmarkReport: Final typed comparison report.
    """
    benchmark_cfg = config.get("benchmark", {})
    split = benchmark_cfg.get("split", "VA")
    deterministic = benchmark_cfg.get("deterministic", True)
    episodes_per_scenario = benchmark_cfg.get("episodes_per_scenario", 1)
    manifest_path = benchmark_cfg.get("manifest_path", "scenarios/scenario_manifest.csv")
    allow_te = benchmark_cfg.get("allow_te", False)

    seeds = config.get("evaluation_seeds", [101, 102, 103])
    scenario_ids = config.get("scenarios")

    # Load scenario manifest
    manifest_obj = config.get("manifest")
    if manifest_obj is None:
        from traffic_drl.environment.scenario_factory import ScenarioManifest

        manifest_obj = ScenarioManifest.from_csv(manifest_path)

    # Environment factory
    env_factory = config.get("env_factory")
    if env_factory is None:
        from traffic_drl.environment.make_env import create_sumo_env

        env_config = config.get("environment_config", {})
        env_factory = lambda record, seed: create_sumo_env(
            env_config, record.route_file, seed=seed
        )

    all_records: list[EpisodeMetrics] = []

    # Controllers to evaluate
    controllers: dict[str, Any] = config.get("controllers", {})
    if not controllers and "model" in config:
        controller_name = config.get("controller_name", "drl_model")
        controllers[controller_name] = config["model"]

    for name, model in controllers.items():
        records = evaluate_manifest(
            model,
            env_factory,
            manifest_obj,
            split=split,
            seeds=seeds,
            controller_name=name,
            deterministic=deterministic,
            episodes_per_scenario=episodes_per_scenario,
            allow_te=allow_te,
        )
        all_records.extend(records)

    # Save output if reporting path is defined
    reporting_cfg = config.get("reporting", {})
    output_dir = reporting_cfg.get("output_dir")
    if output_dir:
        out_path = Path(output_dir) / "metrics.csv"
        save_evaluation_results(all_records, out_path)

    baseline_name = config.get("baseline_controller", "fixed_time")
    return interpret_results(all_records, baseline_controller=baseline_name)


if __name__ == "__main__":
    from scripts.evaluate import main

    main()

