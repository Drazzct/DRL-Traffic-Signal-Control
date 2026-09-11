# Data, Files, and Configurations

This document serves as the single source of truth for the project's directory layout, YAML configuration schemas, and CSV manifest structures.

## 1. Where Files Belong

Store files at these repository-relative paths. Never hardcode absolute paths to user home directories.

| File Type              | Directory Location                     | Consumer / Parser                                       |
| ---------------------- | -------------------------------------- | ------------------------------------------------------- |
| Environment Configs    | `configs/environment/`                 | `environment/make_env.py` (via `config.py` dataclasses) |
| Training Configs       | `configs/train/`                       | `train/train_dqn.py`, `train_ppo.py`                    |
| Evaluation Configs     | `configs/evaluation/`                  | `evaluation/evaluate_benchmark.py`                      |
| Scenario Manifest      | `scenarios/scenario_manifest.csv`      | `scenario_factory.py`, `scenario_sampler.py`            |
| Train/Valid Routes     | `scenarios/train/`, `validation/`      | SUMO-RL via `ScenarioManifest`                          |
| SUMO Network           | `sumo/net/`                            | SUMO-RL / SUMO CLI                                      |
| SUMO Additions         | `sumo/additional/`                     | SUMO-RL / SUMO CLI                                      |
| Checkpoints            | `outputs/runs/<run_id>/checkpoints/`                 | SB3 loaders (`checkpointing.py`)                        |
| Tripinfo & Emissions   | `outputs/runs/<run_id>/tripinfo/`           | `evaluation/parse_tripinfo.py`                          |
| Results & Metrics      | `outputs/runs/<run_id>/results/`                     | `evaluation/plots.py`                                   |

## 2. YAML Configuration Schemas

Configurations are parsed into frozen dataclasses via `src/traffic_drl/config.py`.

### Environment Configuration (`EnvConfig`)
Defines the physical boundaries and normalisation of the simulation.
```yaml
network:
  net_file: sumo/net/<network>.net.xml
  route_files: []
  additional_files: [sumo/additional/vehicle_types.add.xml]
traffic_light:
  ts_id: junction_id
  single_agent: true
timing:
  num_seconds: 3600
  delta_time: 5
  min_green: 10
  max_green: 60
  yellow_time: 3
sumo_options:
  use_gui: false
  lateral_resolution: 1.2
  additional_sumo_cmd: ""
  save_tripinfo: false
observation_bounds:
  max_queue_per_lane: 50.0
  max_time_in_phase: 120.0
```

### Training Configuration (`TrainConfig`)
Orchestrates RL algorithm hyper-parameters and reward schemes. Note that it does *not* duplicate timing/network settings, but instead references an `EnvConfig`.
```yaml
experiment:
  name: "dqn_phase1"
  seed: 42
  device: "cpu"
  split_allowed: "TR"
environment:
  env_config_path: configs/environment/dev_single_intersection.yaml
  manifest_path: scenarios/scenario_manifest.csv
reward:
  type: queue_loss
  params: {}
  wrappers: {}
normalization:
  norm_obs: true
  norm_reward: true
  clip_obs: 10.0
  clip_reward: 10.0
  gamma: 0.99
model_hyperparameters:
  policy: MlpPolicy
  learning_rate: 0.0001
  batch_size: 32
  # ... other SB3 specific kwargs
training_control:
  total_timesteps: 100000
  save_freq: 5000
```

### Evaluation Configuration (`EvalConfig`)
Defines a benchmark run to compare multiple controllers against the same baseline environments.
```yaml
benchmark:
  name: "phase1_validation"
  split: "VA"
  deterministic: true
  episodes_per_scenario: 3
env_config_path: configs/environment/dev_single_intersection.yaml
manifest_path: scenarios/scenario_manifest.csv
checksum_file: scenarios/checksums.sha256
evaluation_seeds: [101, 102, 103]
scenarios: [VA-01, VA-02, VA-03]
artifacts:
  model_checkpoint: outputs/runs/my_run_id/checkpoints/dqn_model.zip
  vec_normalize_stats: outputs/runs/my_run_id/checkpoints/dqn_norm.pkl
normalization:
  training: false
  norm_reward: false
reporting:
  output_dir: outputs/runs/my_eval_run/results
  save_tripinfo: true
  metrics_to_collect:
    - average_waiting_time
    - average_queue_length
    - throughput
```

## 3. Scenario Manifest (`ScenarioRecord`)

The manifest controls data splitting (`TR`, `VA`, `TE`) to prevent the RL agents from memorizing evaluation routes. It is parsed by `csv.DictReader` in `scenario_factory.py`.

**Schema:**
```csv
scenario_id,split,route_file,demand_seed,sumo_seed,num_seconds
TR-01,TR,scenarios/train/tr_01.rou.xml,101,42,3600
VA-01,VA,scenarios/validation/va_01.rou.xml,201,42,3600
TE-01,TE,scenarios/test/te_01.rou.xml,301,42,3600
```

- **TR (Training):** Used exclusively by the RL policy to collect rollouts.
- **VA (Validation):** Used to tune hyper-parameters and compare baselines. The model must *never* learn from these environments.
- **TE (Test):** Held-out for the final, frozen benchmark. Must not be sampled until the evaluation bundle is locked.
