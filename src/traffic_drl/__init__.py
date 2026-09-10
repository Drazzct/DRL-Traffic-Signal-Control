"""traffic_drl — Deep Reinforcement Learning for Traffic Signal Control.

Package structure
-----------------
- :mod:`traffic_drl.contracts` — shared data classes and protocol interfaces.
- :mod:`traffic_drl.config` — typed configuration dataclasses and YAML loaders.
- :mod:`traffic_drl.run_id` — run-ID generation and output-directory helpers.
- :mod:`traffic_drl.environment` — SUMO-RL environment factories and wrappers.
- :mod:`traffic_drl.baselines` — fixed-time, max-pressure, and actuated controllers.
- :mod:`traffic_drl.train` — DQN/PPO training, checkpointing, and sampling.
- :mod:`traffic_drl.evaluation` — benchmark evaluation, metrics, and plots.
"""
# Core contracts — import these to understand the data flowing through the pipeline.
from traffic_drl.contracts import (
    ScenarioRecord,
    EpisodeMetrics,
    MetricSummary,
    BenchmarkReport,
    TripInfoMetrics,
    EmissionMetrics,
    SmokeTestResult,
    ResumeInfo,
    Controller,
    EnvironmentFactory,
    ScenarioSource,
)

# Configuration dataclasses
from traffic_drl.config import (
    EnvConfig,
    TrainConfig,
    EvalConfig,
    load_env_config,
    load_train_config,
    load_eval_config,
)

# Run ID helpers
from traffic_drl.run_id import generate_run_id, ensure_run_directories

__all__ = [
    # contracts
    "ScenarioRecord",
    "EpisodeMetrics",
    "MetricSummary",
    "BenchmarkReport",
    "TripInfoMetrics",
    "EmissionMetrics",
    "SmokeTestResult",
    "ResumeInfo",
    "Controller",
    "EnvironmentFactory",
    "ScenarioSource",
    # config
    "EnvConfig",
    "TrainConfig",
    "EvalConfig",
    "load_env_config",
    "load_train_config",
    "load_eval_config",
    # run_id
    "generate_run_id",
    "ensure_run_directories",
]
