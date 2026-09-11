# Traffic DRL Project

This repository scaffolds the SUMO-RL and Stable-Baselines3 pipeline for mixed-traffic signal control.

## Documentation

Start with the following implementation and architecture guides:

- [Architecture](docs/ARCHITECTURE.md): An overview of the module boundaries, shared contracts (`Controller` protocol), configuration management, and development roles.
- [Implementation Tasks](docs/IMPLEMENTATION_TASKS.md): The concrete, checklist-style assignment of remaining tasks for SV1, SV2, and SV3 to complete the pipeline.
- [Data and Configuration](docs/DATA_AND_CONFIG.md): The single source of truth for the YAML configuration schemas, directory layout, and CSV scenario manifest parsing.
- [MDP Design](docs/MDP_DESIGN.md): The formal definitions of state, action, and reward terms for the reinforcement learning environment.

## Quick Start

The network, route, configuration, and SUMO additional resources are added separately under `sumo/` and `scenarios/`. Python modules in `src/traffic_drl/` currently define the implementation interfaces and may contain deferred `NotImplementedError` bodies. Check the `TODO (SVx)` tags in the codebase to see exactly what remains to be implemented for your specific role.
