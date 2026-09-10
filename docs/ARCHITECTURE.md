# Codebase Architecture

This document explains the architecture of the traffic DRL repository. The codebase was heavily refactored to remove untyped components, implement a structured evaluation pipeline, and unify the interfaces.

## 1. Module Boundaries

The repository uses a strictly layered architecture to prevent circular dependencies.

- **`traffic_drl.contracts`**: The bottom layer. Contains simple dataclasses (`EpisodeMetrics`, `BenchmarkReport`) and structural protocols (`Controller`, `EnvironmentFactory`, `ScenarioSource`). No other `traffic_drl` module is imported here.
- **`traffic_drl.config`**: Contains typed configuration dataclasses (`EnvConfig`, `TrainConfig`, `EvalConfig`) and the YAML loaders that parse and validate them.
- **`traffic_drl.environment`**: The SUMO-RL adapter layer. It provides `create_sumo_env` (the canonical environment factory), the `ScenarioManifest` CSV parser, custom SB3 Gym wrappers, observation functions, and custom reward functions.
- **`traffic_drl.baselines`**: Contains heuristic, non-learning algorithms (`FixedTimeController`, `MaxPressureController`, `ActuatedController`). They all satisfy the `Controller` protocol.
- **`traffic_drl.train`**: Handles SB3 algorithm orchestration (`train_dqn`, `train_ppo`), reproducible replay-buffer and model checkpointing (`save_checkpoint`, `load_checkpoint`), and data-leakage-safe scenario sampling (`ScenarioSampler`).
- **`traffic_drl.evaluation`**: The top layer. Defines the unified `evaluate_controller` loop that evaluates both learning and heuristic controllers fairly. It also contains metric aggregation, Student-t confidence intervals, and visualization tools.

## 2. Shared Contracts (Protocols)

To ensure that different components can be swapped or evaluated fairly, the system uses `@runtime_checkable` Python `Protocol` interfaces in `contracts.py`.

### The `Controller` Protocol
Every algorithm—whether a deep reinforcement learning policy or a simple heuristic—must satisfy the `Controller` protocol. This means it must implement:
- `predict(observation, deterministic=True) -> action`
- `reset() -> None`

Because `FixedTimeController`, `MaxPressureController`, `ActuatedController`, and SB3's `DQN`/`PPO` all satisfy this protocol, the benchmark evaluation script (`evaluate_controller`) never needs special `if is_fixed_time:` logic. Every controller is evaluated exactly the same way.

## 3. Configuration Management

Configurations are written in YAML and loaded into heavily typed, frozen dataclasses. The configuration is the Single Source of Truth for an experiment.

- **`EnvConfig`**: Controls SUMO network boundaries, traffic light selection, timing rules (e.g. `min_green`), and normalisation bounds. (See `configs/environment/dev_single_intersection.yaml`).
- **`TrainConfig`**: Points to an environment config, and defines the RL reward structure, hyperparameters, and checkpoint frequency.
- **`EvalConfig`**: Points to an environment config and defines which manifest split (`VA`, `TE`) and reproducibility seeds to use when benchmarking multiple models against baselines.

*Data Deduplication:* No timing or configuration settings (e.g., `yellow_time`) are duplicated in the training or evaluation configurations. They always read from the `env_config_path` referenced inside them.

## 4. Scenario Manifest & Leakage Prevention

The `ScenarioManifest` (`environment/scenario_factory.py`) loads an inventory of `*.rou.xml` traffic routes from a CSV.
To prevent RL agents from "memorizing" validation data, the `ScenarioSampler` (`train/scenario_sampler.py`) must be instantiated with a specific `allowed_split` (e.g. `TR` for Training).
- If the `DQN` training loop asks the sampler for a new route, it will randomly draw from the `TR` pool only.
- The evaluation benchmark loop exclusively loops over the `VA` or `TE` splits.

## 5. Development Roles & Stubs

The codebase is scaffolded with `NotImplementedError` stubs that contain clear `TODO` instructions. The tasks are grouped by role:

- **SV1 (Simulation & Domain):** Implement SUMO environment initialisation (`create_sumo_env`), parse SUMO emissions XML files, and define the real-world road networks.
- **SV2 (RL & Environment):** Implement custom queue observations, reward functions, metric wrappers, and wire up the Stable-Baselines3 algorithms (`train_dqn`).
- **SV3 (Evaluation & Baselines):** Implement the heuristic baselines (`MaxPressureController`, `ActuatedController`), complete the `evaluate_controller` rollout loop, and build the metric visualizations.
