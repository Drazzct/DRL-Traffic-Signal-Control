# Implementation Tasks

This document contains the concrete checklist of tasks for each collaborator to complete the traffic DRL pipeline. All of these correspond directly to `NotImplementedError` stubs tagged with `TODO (SVx)` in the Python code.

## SV1 (Simulation & Domain)
**Goal:** Define the physical simulation, road network, emissions, and scenario validity.

- [ ] **SUMO Environment Initialisation:** Map the fields from `EnvConfig` (e.g. `num_seconds`, `delta_time`, `min_green`, `yellow_time`) to the SUMO-RL `SumoEnvironment` constructor in `src/traffic_drl/environment/make_env.py` (`create_sumo_env`).
- [ ] **Vectorisation Checks:** Verify that `DummyVecEnv` and `VecNormalize` correctly propagate the `.close()` signal to the inner SUMO subprocess in `make_env.py` (`close_environment`).
- [ ] **Emissions XML Schema:** Confirm the exact SUMO emissions output schema and units (Fuel, CO2, NOx, PMx). Map this logic inside `src/traffic_drl/evaluation/parse_tripinfo.py`.
- [ ] **Reward Tuning Support:** Confirm the emissions schema for `src/traffic_drl/environment/custom_rewards.py` to allow the RL agent to receive penalty signals for high emissions.

## SV2 (Reinforcement Learning)
**Goal:** Build the Gymnasium wrappers, design observations/rewards, and orchestrate SB3 training.

- [ ] **Route Switching (Multi-Scenario):** Implement the logic to pass the newly sampled route file to the underlying SUMO-RL environment upon `reset()` in `src/traffic_drl/environment/wrappers.py` (`MultiScenarioWrapper`).
- [ ] **Metric Aggregation:** Read per-lane queues and waiting times from the SUMO-RL signal object, accumulate them, and append episode summaries (e.g., `average_waiting_time`) to the `info` dict in `src/traffic_drl/environment/wrappers.py` (`MetricsInfoWrapper`).
- [ ] **State Vectors:** Implement the exact feature ordering (queue counts, density, phase states) in `src/traffic_drl/environment/custom_observations.py`.
- [ ] **Reward Magnitudes:** Validate reward magnitudes in `src/traffic_drl/environment/custom_rewards.py` so the DQN agent learns effectively without exploding gradients.
- [ ] **Training Adapters:** Map `config.model_hyperparameters` to the SB3 constructors, call `model.learn()`, and wire up the checkpointing callbacks in `src/traffic_drl/train/train_dqn.py` and `src/traffic_drl/train/train_ppo.py`.
- [ ] **Evaluation Normalisation:** Freeze `VecNormalize` (`training=False`, `norm_reward=False`) when building environments for validation/testing in `src/traffic_drl/environment/make_env.py`.
- [ ] **Checkpoint Integrity:** Verify that checkpoint `.zip` and `.pkl` bundles reload correctly in `src/traffic_drl/train/checkpointing.py`.

## SV3 (Evaluation & Baselines)
**Goal:** Build heuristic controllers, enforce data splits, and evaluate model performance.

- [ ] **Manifest Safeguards:** Prevent sampling of `TE` (Test) splits before the model and manifest are explicitly frozen, by adding checksum validations in `src/traffic_drl/train/scenario_sampler.py` and `src/traffic_drl/environment/scenario_factory.py`.
- [ ] **Evaluation Loop:** Implement the `reset` -> `predict` -> `step` loop in `src/traffic_drl/evaluation/evaluate_benchmark.py` (`evaluate_controller`). Read the metrics from the `info` dictionary.
- [ ] **Max Pressure Controller:** Implement the lane-level pressure computation logic (extracting queue counts from the observation vector) in `src/traffic_drl/baselines/max_pressure.py`.
- [ ] **Actuated Controller:** Complete the `ActuatedController` logic acting as a semi-dynamic baseline.
- [ ] **Fixed-Time Controller:** Confirm `fixed_ts=True` properly triggers SUMO-RL's internal static phase plans in `src/traffic_drl/baselines/fixed_time.py`.
- [ ] **Metric Statistics:** Compute mean, standard deviation, and 95% Confidence Intervals for benchmark reports, and flag constraint violations (e.g., degraded throughput) in `src/traffic_drl/evaluation/metrics.py`.
- [ ] **Data Visualisation:** Implement bar charts grouped by scenario, coloured by baseline, using `matplotlib` in `src/traffic_drl/evaluation/plots.py`.

## Weekly Work Allocation

The weekly matrix below is the primary work-allocation view. Each student should first read their column for the current week, then use the detailed tables above it for the exact functions, files and acceptance checks.

The dates follow the 13-week plan beginning on 31/08/2026. Every week has one shared objective, one accountable owner for each workstream and a completion gate. A task is complete only when its output exists and the assigned reviewer has checked it.

### Master Weekly Allocation Matrix

| Week                  | Shared objective                               | SV1 - Simulation and data                                             | SV2 - Environment and DRL                                                              | SV3 - Baselines and evaluation                                  | Completion gate                                             |
| --------------------- | ---------------------------------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------- |
| 1                     | Freeze project contracts and define DEV-00     | Inspect network, map approaches, identify TLS and draft vehicle types | Freeze MDP, observation shape, action mapping and fake environment                     | Define manifest, survey form, splits, seeds and metrics         | Network/control scope and MDP contract approved             |
| 2                     | Connect SUMO to Gymnasium and run the pilot    | Implement route generation/validation and create DEV-00, TR-01, TR-02 | Implement environment factory, observations, rewards, wrappers and DQN pilot           | Implement manifest loading and fixed-time DEV-00 baseline       | Reset/step/close, baseline and checkpoint reload work       |
| 3                     | Calibrate and lock scenario splits             | Calibrate traffic/vehicle behavior and generate TR/VA/TE routes       | Implement sampler and package calibrated DQN pilot                                     | Validate manifest, create checksums and lock TE                 | Second member reproduces DEV-00 and baseline; TE is unused  |
| 4                     | Start repeatable TR-only training              | QA route flow, vehicle classes, teleporting and output paths          | Implement vectorization, TR sampler, DQN construction/training and checkpoint callback | Implement smoke test, initial evaluation runner and seed matrix | Training logs scenario ID and all seeds; only TR is sampled |
| 5                     | Cover the training domain and finish baselines | Validate demand, vehicle-mix, behavior profiles and emissions         | Continue DQN training, schedules and traffic callbacks                                 | Complete fixed-time and actuated/max-pressure evaluation        | All controllers emit the same metric schema                 |
| 6                     | Select the first DQN checkpoint using VA       | Confirm reward/metric measurements and simulation sensitivity         | Finalize reward v1, checkpoint loading and frozen VA normalization                     | Implement XML parsers, metric collection, aggregation and CI    | M2 is selected from VA only; TE remains locked              |
| 7                     | Add emissions evidence and optional PPO        | Collect emissions by scenario and vehicle class                       | Implement/train PPO only after DQN is stable                                           | Compare DQN/PPO/baselines on VA and start ablation matrix       | PPO can be dropped without affecting DQN deliverable        |
| 8                     | Tune only on VA and prepare test artifacts     | Syntax-check and checksum TE files without policy execution           | Run declared VA-only tuning and inference checks                                       | Complete VA selection protocol and benchmark configuration      | Selection criteria are written before TE results            |
| 9                     | Freeze the evaluation bundle and unlock TE     | Freeze simulation inputs and compute artifact checksums               | Freeze model, normalizer, config and inference script                                  | Add benchmark guards, sign bundle and unlock TE                 | No model, reward, route or normalizer changes after unlock  |
| 10                    | Run deterministic TE-01 to TE-04 benchmark     | Run matched simulations and preserve raw SUMO outputs                 | Run frozen deterministic inference and latency logging                                 | Execute benchmark runner and completeness checks                | Failed episodes are recorded, not hidden                    |
| 11                    | Run robustness and stress tests                | Run TE-05/TE-06 and repeated TE-03 stress tests                       | Analyze recovery/failures and fix only technical runner issues                         | Parse all outputs and produce the complete exception report     | Reruns use the same frozen bundle and are documented        |
| 12                    | Produce auditable statistics and report draft  | Create network, calibration and queue figures                         | Write MDP, reward, model and reproducibility methods                                   | Compute CI, comparisons, tables, plots and discussion           | Every number traces to scenario, seed and commit            |
| 13                    | Reproduce, document and submit                 | Finalize SUMO/network/data instructions                               | Finalize README, configs, checkpoints and demo                                         | Finalize report, slides, archive and submission checklist       | A second member reproduces the short pipeline               |
| Buffer 1: 01/12-06/12 | Repair missing or invalid existing results     | Rerun failed simulations and missing seeds                            | Repair critical pipeline/checkpoint defects                                            | Check completeness and approve reruns                           | No new algorithm or scope expansion                         |
| Buffer 2: 07/12-13/12 | Archive and finalize deliverables              | Complete technical appendices and backups                             | Complete code/config archive                                                           | Recheck tables, README, slides and closure record               | Final archive is verified and backed up                     |

### Weekly Operating Rules

For every week, use this sequence:

1. **Start of week:** each student confirms the inputs needed from the previous week.
2. **During the week:** each student works only in their owned files and records decisions in `docs/decisions.md`.
3. **Before the weekly meeting:** the owner attaches the output path, command used, seed and commit hash.
4. **Weekly review:** the assigned reviewer reruns the smallest useful check and records pass/fail plus an issue note.
5. **Gate decision:** the group either accepts the week, records a blocker and recovery action, or moves only the non-blocked work forward.
