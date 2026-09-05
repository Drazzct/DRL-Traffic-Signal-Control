# Traffic DRL Project

This repository scaffolds the SUMO-RL and Stable-Baselines3 pipeline for mixed-traffic signal control.

Start with:

- [Interfaces and file formats](docs/interfaces_and_formats.md): shared contracts, ownership, schemas, parsing rules, and checkpoint lifecycle.
- [Configuration formats](docs/config_formats.md): YAML configuration templates.
- [MDP description](docs/mdp_descriptions.md): state, action, reward, and episode design.

The network, route, configuration, and SUMO additional resources are added separately. Python modules currently define the implementation interfaces and may contain deferred `NotImplementedError` bodies.
