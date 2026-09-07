# Interfaces and File Formats

This document is the shared contract for the Phase 1 traffic DRL pipeline. It explains what each component receives, what it returns, which files it reads or writes, and who owns the next implementation step.

The current repository intentionally contains API scaffolding. A function may still raise `NotImplementedError`; the schemas and interfaces below are the stable agreement for later implementation.

## 0. Where Files Belong

Create files at these repository-relative paths. Do not put generated artifacts beside Python source files.

| File family            | Create here                                      | Created by                                    | Read/parsed by                                                                           |
| ---------------------- | ------------------------------------------------ | --------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Environment YAML       | `configs/env/`                                   | SV2                                           | `environment/make_env.py` with YAML parser                                               |
| Training YAML          | `configs/train/`                                 | SV2                                           | `train/train_dqn.py` or `train/train_ppo.py`                                             |
| Evaluation YAML        | `configs/evaluation/`                            | SV3                                           | `evaluation/evaluate_benchmark.py`                                                       |
| Scenario manifest      | `scenarios/scenario_manifest.csv`                | SV3                                           | `environment/scenario_factory.py` and `train/scenario_sampler.py` using `csv.DictReader` |
| Route generator inputs | `scenarios/generators/`                          | SV1                                           | generator modules and SUMO tools                                                         |
| Train routes           | `scenarios/train/`                               | SV1                                           | `ScenarioManifest` and SUMO-RL                                                           |
| Validation routes      | `scenarios/validation/`                          | SV1                                           | validation runner                                                                        |
| Held-out routes        | `scenarios/test/`                                | SV1                                           | final benchmark only after unlock                                                        |
| OSM source             | `data/osm_raw/`                                  | SV1                                           | `netconvert`/`netedit`, never modified in place                                          |
| Survey originals       | `data/survey_raw/`                               | SV3                                           | manual review and CSV import                                                             |
| Processed survey data  | `data/processed/`                                | SV1/SV3                                       | calibration and scenario generators                                                      |
| SUMO network           | `sumo/net/`                                      | SV1                                           | SUMO-RL and SUMO CLI                                                                     |
| SUMO additions         | `sumo/additional/`                               | SV1                                           | SUMO XML parser/CLI                                                                      |
| SUMO launch configs    | `sumo/cfg/`                                      | SV1/SV2                                       | SUMO CLI/GUIs                                                                            |
| Checkpoints            | `outputs/checkpoints/` or external Drive storage | SV2                                           | `checkpointing.py` and SB3 loaders                                                       |
| Training logs          | `outputs/logs/`                                  | SB3 callbacks                                 | evaluation/plotting readers                                                              |
| Tripinfo/emissions     | `outputs/tripinfo/<run_id>/`                     | SUMO process launched by `make_env.py`/runner | `evaluation/parse_tripinfo.py`                                                           |
| Evaluation results     | `outputs/results/`                               | SV3                                           | metrics and plotting modules                                                             |
| Team decisions         | `docs/decisions.md`                              | all members                                   | human review                                                                             |

When a directory is empty, keep its `.gitkeep`; when a real file is created, remove only that directory's redundant `.gitkeep`.

### Path rules

1. Store paths in manifests/configs relative to the repository root, using `/` separators.
2. Resolve paths once at the boundary that reads the file; pass `Path` or a typed contract internally.
3. Never use a path to infer a scenario split. Read `split` from the manifest and validate it.
4. Generated output paths must be configurable; functions must not write to a developer's home directory.
5. Raw survey/OSM files are immutable. Write cleaned data to `data/processed/` or cleaned SUMO files to `sumo/`.

## 1. Ownership

| Area                                                     | Primary owner | Required review                                                          |
| -------------------------------------------------------- | ------------- | ------------------------------------------------------------------------ |
| SUMO network, vehicle types, routes, emissions           | SV1           | SV2 validates environment compatibility; SV3 validates scenario metadata |
| Environment, observations, rewards, DQN/PPO, checkpoints | SV2           | SV1 validates SUMO assumptions; SV3 validates leakage and metrics        |
| Manifest, baselines, metrics, benchmark interpretation   | SV3           | SV1 validates traffic meaning; SV2 validates integration                 |

Every change to a network, action space, reward, split, seed policy, or file schema must be recorded in the project decision notes before it is used by another component.

## 2. Package Boundaries

### `traffic_drl.contracts`

This is the shared vocabulary. Components should exchange these objects instead of inventing local dictionaries.

- `ScenarioRecord`: one immutable scenario manifest row.
- `ScenarioSource`: protocol implemented by a manifest/catalog.
- `EpisodeMetrics`: metrics from one controller episode.
- `MetricSummary`: aggregate values for one controller/scenario group.
- `BenchmarkReport`: interpreted comparison against a baseline.
- `TripInfoMetrics`: traffic values parsed from `tripinfo.xml`.
- `EmissionMetrics`: fuel and pollutant values parsed from emissions XML.
- `SmokeTestResult`: result of a short reset/step check.
- `ResumeInfo`: metadata needed to reproduce and resume training.
- `Controller`: common `predict()` and `reset()` interface for SB3 and heuristic controllers.
- `EnvironmentFactory`: callable contract that creates an environment from a scenario and seed.

### `traffic_drl.environment`

- `make_env.py` creates one SUMO-RL single-agent environment for exactly one route file.
- `scenario_factory.py` loads and selects manifest records.
- `custom_observations.py` subclasses SUMO-RL `ObservationFunction`; it must implement `__call__()` and `observation_space()`.
- `custom_rewards.py` exposes reward functions accepting a SUMO-RL `TrafficSignal`.
- `wrappers.py` contains Gymnasium wrappers. `reset()` returns `(observation, info)` and `step()` returns `(observation, reward, terminated, truncated, info)`.

### `traffic_drl.train`

- `scenario_sampler.py` selects one route from an allowed split per episode.
- `train_dqn.py` and `train_ppo.py` are algorithm adapters.
- `checkpointing.py` owns shared model, replay-buffer, VecNormalize, and resume metadata persistence.
- `callbacks.py` provides SB3 `BaseCallback` subclasses.

### `traffic_drl.evaluation` and `traffic_drl.baselines`

Evaluation consumes the same `ScenarioRecord`, route, and seed for every controller. It returns `EpisodeMetrics`, aggregates them into `MetricSummary`, and produces a `BenchmarkReport`.

## 3. Scenario Contract

### `ScenarioRecord`

A record contains:

| Field         | Type    | Meaning                                                                             |
| ------------- | ------- | ----------------------------------------------------------------------------------- |
| `scenario_id` | string  | Stable ID such as `DEV-00`, `TR-01`, `VA-01`, or `TE-01`.                           |
| `split`       | string  | `DEV`, `TR`, `VA`, or `TE`.                                                         |
| `route_file`  | path    | One SUMO `.rou.xml`; never a comma-separated list.                                  |
| `demand_seed` | integer | Seed used by demand/route generation.                                               |
| `sumo_seed`   | integer | Seed used by the SUMO simulation.                                                   |
| `num_seconds` | integer | Episode duration in simulation seconds.                                             |
| `metadata`    | mapping | Demand profile, turning ratios, vehicle mix, behavior profile, checksum, and notes. |

### Scenario manifest CSV

The manifest is UTF-8 CSV with a header. Required columns:

```
scenario_id,split,route_file,demand_seed,sumo_seed,num_seconds
```

Recommended metadata columns:

```
demand_profile,turning_ratio,vehicle_mix,behavior_profile,checksum
```

Parse with `csv.DictReader`. Convert seeds and duration to integers. Resolve `route_file` relative to the repository root. Preserve optional columns as metadata. Validate duplicate IDs, valid split names, route existence, and checksum policy.

Rules:

1. Training/domain randomization may select only `TR`.
2. Validation may select only `VA` and must not update model weights or normalizer statistics.
3. Test scenarios remain locked until the model and evaluation bundle are frozen.
4. All controllers use the same route and matched seeds for a benchmark comparison.

## 4. SUMO File Formats

### Network: `.net.xml`

A SUMO compiled network containing `<location>`, public `<edge>` elements, `<lane>` children, junctions, connections, and traffic-light definitions. Treat it as SUMO-owned XML. Use `netconvert`, `netedit`, `sumolib`, or SUMO validation commands rather than string manipulation.

Minimum structural shape of `sumo/net/<network>.net.xml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<net version="1.0">
  <location netOffset="0.0,0.0" convBoundary="..." origBoundary="..." />
  <edge id="A1_in" from="A1" to="junction_0" priority="1">
    <lane id="A1_in_0" index="0" speed="13.9" length="100.0"
          shape="0.0,0.0 100.0,0.0" />
  </edge>
  <edge id="A1_out" from="junction_0" to="A1_exit" priority="1">
    <lane id="A1_out_0" index="0" speed="13.9" length="100.0"
          shape="100.0,0.0 200.0,0.0" />
  </edge>
  <junction id="junction_0" type="traffic_light" x="100.0" y="0.0"
            incLanes="A1_in_0" intLanes=":junction_0_0" />
  <connection from="A1_in" to="A1_out" fromLane="0" toLane="0"
              via=":junction_0_0" tl="junction_0" linkIndex="0" />
  <tlLogic id="junction_0" type="static" programID="0" offset="0">
    <phase duration="30" state="Gr" />
    <phase duration="3" state="yr" />
  </tlLogic>
</net>
```

Do not hand-edit a compiled network as the normal workflow. The example is a schema guide only; `netconvert`/`netedit` generate the complete internal edges, connections, and signal links. Parse it with `sumolib.net.readNet()` or validate it by starting SUMO.

SV1 owns the network and must retain the original OSM source separately from the cleaned `.net.xml`.

### Routes: `.rou.xml`

A route file contains:

```xml
<routes>
  <vType id="..." vClass="..." />
  <route id="..." edges="edge_a edge_b" />
  <vehicle id="..." type="..." route="..." depart="..." />
</routes>
```

A project may instead place `<vType>` definitions in an additional file, but every route file must reference valid vehicle types and public network edges. Parse with `xml.etree.ElementTree` for structural checks and validate semantics with SUMO route/network tools.

A generator must report realized vehicle counts by approach after generation. `randomTrips.py --period` is a generation interval, not automatically a per-lane traffic flow.

### Vehicle types/additional XML: `.add.xml`

Create `sumo/additional/vehicle_types.add.xml` with this structure:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<additional>
  <vType id="motorcycle_vn_base"
         vClass="motorcycle"
         length="2.0"
         width="0.7"
         accel="2.0"
         decel="4.0"
         minGap="0.8"
         tau="0.8"
         laneChangeModel="SL2015"
         minGapLat="0.25"
         maxSpeedLat="1.0" />
  <vType id="passenger_vn_base"
         vClass="passenger"
         length="4.5"
         width="1.8"
         accel="2.6"
         decel="4.5"
         minGap="2.5"
         tau="1.0"
         laneChangeModel="SL2015" />
</additional>
```

The root must be `<additional>`. Each `<vType>` requires a unique `id`; route files refer to that ID through `vehicle type="..."`. Parse with an XML parser for structure, then validate IDs and parameter values through SUMO. Vehicle types are owned by SV1 and calibrated values must be recorded separately from these initial defaults.

### Detectors/additional XML

Create `sumo/additional/detectors.add.xml` like this:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<additional>
  <inductionLoop id="A1_queue_detector"
                 lane="A1_in_0"
                 pos="90.0"
                 freq="5"
                 file="outputs/detectors/<run_id>.xml" />
</additional>
```

`lane` must match a lane ID in `.net.xml`; `pos` is measured from the lane start; `freq` is the sampling interval in seconds; and `file` is the SUMO output path. Parse the XML with `ElementTree`, then verify the lane with `sumolib` before running SUMO.

### Traffic-light program/additional XML

If the project supplies a separate signal program, create `sumo/additional/tls_programs.add.xml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<additional>
  <tlLogic id="junction_0" type="static" programID="phase1" offset="0">
    <phase duration="30" state="Gr" />
    <phase duration="3" state="yr" />
    <phase duration="30" state="rG" />
    <phase duration="3" state="ry" />
  </tlLogic>
</additional>
```

The `state` length and signal characters must match the links in the network's traffic light. Do not create a program until NETEDIT/survey evidence confirms that the controlled signal exists.

### Additional files: `.add.xml`

Additional files contain SUMO elements such as `<vType>`, `<vTypeDistribution>`, detectors, or `<tlLogic>`. Keep one source of truth for each additional definition. Parse as XML and validate referenced IDs against the network and route files.

### SUMO configuration: `.sumocfg`

A SUMO configuration XML normally contains `<input>` with `net-file`, `route-files`, and `additional-files`, plus `<time>` settings. Parse as XML. Paths inside the configuration are interpreted relative to the configuration file location unless the SUMO command specifies otherwise.

Create `sumo/cfg/<run_id>.sumocfg` like this:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
  <input>
    <net-file value="../net/<network>.net.xml" />
    <route-files value="../../scenarios/train/tr_01.rou.xml" />
    <additional-files value="../additional/vehicle_types.add.xml,../additional/detectors.add.xml" />
  </input>
  <time>
    <begin value="0" />
    <end value="900" />
  </time>
  <output>
    <tripinfo-output value="../../outputs/tripinfo/<run_id>/tripinfo.xml" />
    <emission-output value="../../outputs/tripinfo/<run_id>/emissions.xml" />
  </output>
  <processing>
    <time-to-teleport value="-1" />
  </processing>
</configuration>
```

The runner must create `outputs/tripinfo/<run_id>/` before SUMO starts. Validate the config by starting `sumo -c sumo/cfg/<run_id>.sumocfg` or using the equivalent SUMO version command, and resolve relative paths from the `.sumocfg` location.

### Generated `tripinfo.xml`

SUMO creates this file; the project does not hand-write it. A completed file has one `<tripinfo>` element per vehicle that finished its trip:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<tripinfos>
  <tripinfo id="veh_0001"
            depart="0.00"
            arrival="42.50"
            duration="42.50"
            routeLength="580.00"
            waitingTime="4.00"
            timeLoss="8.20"
            departDelay="0.00"
            vType="passenger_vn_base" />
</tripinfos>
```

`parse_tripinfo.py:parse_tripinfo()` must open `outputs/tripinfo/<run_id>/tripinfo.xml` with `xml.etree.ElementTree`, iterate over `tripinfo` elements, convert the numeric attributes to `float`, count vehicles, and return `TripInfoMetrics`. The required mapping is:

| XML attribute                      | Contract field         |
| ---------------------------------- | ---------------------- |
| `waitingTime`                      | `average_waiting_time` |
| `timeLoss`                         | `average_time_loss`    |
| `duration`                         | `average_travel_time`  |
| number of `<tripinfo>` elements    | `vehicle_count`        |
| completed vehicles / episode hours | `throughput`           |

The parser must document whether incomplete vehicles are absent from the file or represented separately; never silently treat missing vehicles as zero delay.

### Generated `emissions.xml`

SUMO creates this file when `--emission-output` is enabled. A typical file has vehicle/time-step records like:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<meandata>
  <timestep time="5.00">
    <vehicle id="veh_0001"
             fuel="0.12000"
             CO2="12.50000"
             CO="0.02000"
             HC="0.00100"
             NOx="0.00400"
             noise="65.0" />
  </timestep>
</meandata>
```

`parse_tripinfo.py:parse_emissions()` must read `outputs/tripinfo/<run_id>/emissions.xml` with `ElementTree`, iterate over all `vehicle` elements, sum `fuel`, `CO2`, and `NOx`, and return `EmissionMetrics`. The implementation must preserve the SUMO version's units in metadata and must not confuse `CO` with `CO2` or `NOx`.

### Who creates `tripinfo.xml` and emissions XML?

The parser does **not** create these files. The SUMO process creates them when the runner passes output options to SUMO. The implementation belongs in `traffic_drl.environment.make_env.create_sumo_env()` or in the batch/evaluation runner that owns the SUMO command.

For each run, create a dedicated directory:

```
outputs/tripinfo/<run_id>/
├── tripinfo.xml
└── emissions.xml                 # optional, required for environmental metrics
```

The runner must construct equivalent SUMO options:

```
--tripinfo-output outputs/tripinfo/<run_id>/tripinfo.xml
--emission-output outputs/tripinfo/<run_id>/emissions.xml
```

The exact Python integration depends on whether SUMO-RL starts SUMO through TraCI or `libsumo`, but the ownership is the same:

1. `create_sumo_env()` or the evaluation runner creates `outputs/tripinfo/<run_id>/`.
2. It passes the two output paths to SUMO before the simulation starts.
3. SUMO writes `tripinfo.xml` when vehicles complete and `emissions.xml` when emission output is enabled.
4. The episode closes the SUMO process so the XML files are flushed.
5. `evaluation/parse_tripinfo.py` parses the completed XML files and returns `TripInfoMetrics` and `EmissionMetrics`.
6. `merge_tripinfo_metrics()` combines them into `EpisodeMetrics`, which is saved under `outputs/results/<run_id>/metrics.csv`.

Do not create `tripinfo.xml` by hand and do not parse it while SUMO is still running. If the run is terminated early, record that the file contains completed vehicles only and mark incomplete-vehicle handling in the run metadata.

## 5. Environment and MDP Interfaces

### SUMO-RL observation

`MixedTrafficObservation` subclasses SUMO-RL `ObservationFunction`:

```python
observation_function = MixedTrafficObservation(ts, approach_ids, include_elapsed_time=True)
observation = observation_function()
space = observation_function.observation_space()
```

The observation vector must have a fixed order and match its Gymnasium space. The planned contents are phase encoding, minimum-green state, per-lane density/queue, speed or occupancy, and optional elapsed phase time. SV2 must document the exact order before training.

### Gymnasium environment

Every environment must expose `action_space` and `observation_space` and implement:

```python
observation, info = env.reset(seed=seed)
observation, reward, terminated, truncated, info = env.step(action)
env.close()
```

Actions are discrete and must respect minimum green time and yellow clearance. No component may invent a traffic-light phase that is absent from the validated SUMO network.

### Rewards

Reward functions receive a SUMO-RL `TrafficSignal`. They may combine queue loss, pressure, waiting-time difference, emissions, and phase-switch penalty. Reward terms must be normalized before weights are tuned. Reward improvement is not sufficient evidence if waiting time, queue, throughput, or safety constraints regress.

## 6. Training and Checkpoint Contracts

### Model checkpoint: `.zip`

SB3 writes the serialized policy/model, optimizer state, action/observation spaces, and hyperparameters. Do not parse it manually; load it with `DQN.load()` or `PPO.load()`.

### VecNormalize: `.pkl`

This file stores running observation statistics and, when enabled, reward statistics. Restore it before loading the model:

```python
vec_env = make_vectorized_environment(...)
vec_env = VecNormalize.load("checkpoint_vecnormalize.pkl", vec_env)
vec_env.training = True       # False for evaluation
vec_env.norm_reward = False  # required for comparable evaluation rewards
model = DQN.load("checkpoint.zip", env=vec_env)
```

Parse/load it only with `VecNormalize.load()`; it is not a human-readable configuration file.

### Replay buffer: `.pkl`

A DQN replay buffer is SB3-managed binary/pickle state. Save and load it only through `model.save_replay_buffer()` and `model.load_replay_buffer()`.

### Resume metadata: `.json`

UTF-8 JSON representing `ResumeInfo`:

```json
{
  "num_timesteps": 10000,
  "model_class": "DQN",
  "seed": 5,
  "scenario_split": "TR",
  "config_path": "configs/train_config.yaml",
  "git_commit": "<commit>",
  "extra": {}
}
```

Parse with `json.load` and validate that the checkpoint, normalizer, config, code revision, and seed are compatible before resuming.

A complete resume bundle contains:

```
checkpoint.zip
checkpoint_replay_buffer.pkl      # DQN, optional but recommended
checkpoint_vecnormalize.pkl       # required when normalization is used
checkpoint_resume_info.json
training_config.yaml              # exact configuration used for the run
```

## 7. Evaluation and Metrics

### `EpisodeMetrics`

Required primary metrics:

- `average_waiting_time`
- `average_queue_length`
- `time_loss`

Required constraints:

- `throughput`
- `phase_switch_rate`
- `min_green_violations`

Additional metrics may include `travel_time`, `spillback`, `recovery_time`, `fuel`, `co2`, `nox`, and `inference_latency`.

A record also includes `controller`, `scenario_id`, and `seed`. Use the same field names for fixed-time, actuated, DQN, and proposed controllers.

### Tripinfo XML

SUMO `tripinfo.xml` contains one `<tripinfo>` element per completed vehicle. Aggregate `waitingTime`, `timeLoss`, `duration`, and vehicle count with an XML parser. Derive throughput using the defined episode duration and document whether incomplete vehicles are excluded.

### Emissions XML

SUMO emissions output contains vehicle-level emission records. Aggregate fuel, CO2, and NOx using the exact SUMO attribute names and units from the selected SUMO version. SV1 owns unit verification.

### Results CSV/JSON

Evaluation output is either:

- UTF-8 CSV with one row per `EpisodeMetrics` and a header; or
- UTF-8 JSON containing a list of objects with the same field names.

Use `csv` or `json`/pandas to parse it. Never serialize Python `repr` values as the interchange format.

## 8. Plot and Log Files

- Plot outputs are rendered `.png` or `.pdf` images.
- SB3 CSV logs contain tabular training metrics with headers.
- TensorBoard logs are event-file directories and must be read with TensorBoard/SB3 readers.
- Queue heatmap inputs must contain time, approach ID, and queue length.

## 9. Required Workflow

1. SV1 supplies validated network, vehicle types, route XML, and SUMO outputs.
2. SV2 creates one environment for one selected route and applies the typed observation/reward contracts.
3. SV2 trains only on allowed training scenarios and saves the complete checkpoint bundle.
4. SV3 validates manifest split boundaries and evaluates all controllers on matched routes and seeds.
5. SV3 aggregates `EpisodeMetrics` into confidence intervals and interprets results relative to fixed-time.
6. The team records unresolved assumptions, parser choices, units, and schema changes in documentation before the next phase.

## 10. Copyable File Templates

These are the recommended starting formats. Replace bracketed values with real values; do not invent unlabelled measurements.

### 10.1 Environment config

Create `configs/env/dev_single_intersection.yaml`:

```yaml
network:
  net_file: sumo/net/<network>.net.xml
  route_files: []
  additional_files:
    - sumo/additional/<vehicle_types>.add.xml
traffic_light:
  ts_id: <validated_tls_id>
  single_agent: true
timing:
  num_seconds: 900
  delta_time: 5
  min_green: 30
  yellow_time: 3
sumo_options:
  use_gui: false
  lateral_resolution: 0.4
  additional_sumo_cmd: ""
observation_bounds:
  max_queue_per_lane: 100.0
  max_time_in_phase: 900.0
```

Read with a YAML parser in `make_env.py`; validate that the network, additional files, and TLS ID exist before calling `gym.make("sumo-rl-v0", ...)`.

### 10.2 Training config

Create `configs/train/dqn_phase1.yaml`:

```yaml
experiment:
  name: phase1_dqn
  seed: 5
  device: cpu
  split_allowed: TR
environment:
  env_config_path: configs/env/dev_single_intersection.yaml
  manifest_path: scenarios/scenario_manifest.csv
  num_seconds: 900
  delta_time: 5
  min_green: 30
  yellow_time: 3
reward:
  type: queue_loss
  params: {}
normalization:
  norm_obs: true
  norm_reward: false
  clip_obs: 10.0
  clip_reward: 10.0
model_hyperparameters:
  policy: MlpPolicy
  learning_rate: 0.0001
  batch_size: 32
  buffer_size: 10000
  learning_starts: 100
  gamma: 0.99
training_control:
  total_timesteps: 10000
  save_freq: 1000
```

Read with a YAML parser, validate `split_allowed` against the manifest, and pass only the algorithm-specific `model_hyperparameters` to SB3. Do not pass the whole YAML mapping directly to an SB3 constructor.

### 10.3 Evaluation config

Create `configs/evaluation/phase1_validation.yaml`:

```yaml
benchmark:
  name: phase1_validation
  split: VA
  deterministic: true
  episodes_per_scenario: 1
  manifest_path: scenarios/scenario_manifest.csv
  checksum_file: scenarios/checksums.sha256
evaluation_seeds: [101, 102, 103]
scenarios: [VA-01, VA-02]
artifacts:
  model_checkpoint: outputs/checkpoints/dqn_phase1.zip
  vec_normalize_stats: outputs/checkpoints/dqn_phase1_vecnormalize.pkl
normalization:
  training: false
  norm_reward: false
reporting:
  output_dir: outputs/results/phase1_validation
  save_tripinfo: true
  metrics_to_collect:
    - average_waiting_time
    - average_queue_length
    - time_loss
    - throughput
    - phase_switch_rate
    - min_green_violations
```

Read with a YAML parser. `evaluate_benchmark.py` must verify the checkpoint and normalizer exist, set `training: false`, and reject TE before the model is frozen.

### 10.4 Scenario manifest

Create `scenarios/scenario_manifest.csv`:

```csv
scenario_id,split,route_file,demand_seed,sumo_seed,num_seconds,demand_profile,turning_ratio,vehicle_mix,behavior_profile,checksum
DEV-00,DEV,scenarios/dummy/dev_00.rou.xml,5,5,900,low,baseline,mixed,typical,
TR-01,TR,scenarios/train/tr_01.rou.xml,101,101,3600,normal,calibrated,mixed,typical,
VA-01,VA,scenarios/validation/va_01.rou.xml,201,201,3600,surge,held_out,mixed,typical,
TE-01,TE,scenarios/test/te_01.rou.xml,301,301,3600,held_out,held_out,mixed,edge,<sha256>
```

`ScenarioManifest.from_csv()` should parse this with `csv.DictReader`, convert numeric columns, resolve route paths, and construct `ScenarioRecord` objects. `checksum` is SHA-256 over the route file bytes and is required for locked TE rows.

### 10.5 Survey data

Create immutable originals in `data/survey_raw/` and cleaned tables in `data/processed/`.

`data/processed/turning_counts.csv`:

```csv
session_id,interval_start,interval_end,approach_in,approach_out,vehicle_class,count,reliability,notes
S01,07:00,07:15,A1,A3,MC,42,H,
```

`data/processed/phase_timing.csv`:

```csv
session_id,timestamp,signal_id,phase_id,green_seconds,yellow_seconds,all_red_seconds,notes
S01,07:00,t,0,33,2,0,
```

Parse with `csv.DictReader`; keep counts numeric, vehicle classes from `MC/CAR/BUS/TRK/OTH`, and reliability from `H/M/L`. Never overwrite files under `survey_raw/`.

### 10.6 SUMO vehicle types and detectors

Create `sumo/additional/vehicle_types.add.xml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<additional>
  <vType id="motorcycle_vn_base" vClass="motorcycle"
         length="2.0" width="0.7" accel="2.0" decel="4.0"
         minGap="0.8" tau="0.8" laneChangeModel="SL2015" />
</additional>
```

Create `sumo/additional/detectors.add.xml` for detector definitions. Parse both with an XML parser and validate every referenced lane/edge/type against `sumo/net/<network>.net.xml`.

### 10.7 Checksum file

Create `scenarios/checksums.sha256` only after TE is locked:

```
<64 lowercase hexadecimal SHA-256>  scenarios/test/te_01.rou.xml
<64 lowercase hexadecimal SHA-256>  scenarios/test/te_02.rou.xml
```

Parse as whitespace-separated digest and path pairs. Recompute SHA-256 over file bytes before final evaluation; do not use checksums to select training scenarios.

### 10.8 Evaluation output

Create results under `outputs/results/<run_id>/metrics.csv`:

```csv
controller,scenario_id,seed,average_waiting_time,average_queue_length,time_loss,throughput,phase_switch_rate,min_green_violations,travel_time,fuel,co2,nox
fixed_time,VA-01,101,0.0,0.0,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0
```

Write one row per episode using the exact `EpisodeMetrics` field names. Parse with `csv.DictReader` or pandas and convert numeric fields before aggregation.

### 10.9 Decision and assumption notes

Create `docs/decisions.md`:

```markdown
# Decisions

## DEC-001: <short title>
- Date: YYYY-MM-DD
- Owners: SV1/SV2/SV3
- Decision: <what changed>
- Reason: <evidence or constraint>
- Affected files: <repository-relative paths>
- Validation: <command, reviewer, or result>
```

Create `docs/assumptions.md` with the same decision ID style. Every assumed demand, timing, vehicle ratio, or missing survey value must include its source, units, confidence, and planned validation.
