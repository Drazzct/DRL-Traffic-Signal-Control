"""Environment construction and SUMO-RL customisation APIs.

Public API
----------
- :func:`~traffic_drl.environment.make_env.create_sumo_env` — canonical SUMO-RL env factory.
- :func:`~traffic_drl.environment.make_env.make_dev_environment` — DEV-00 smoke-test env.
- :func:`~traffic_drl.environment.make_env.make_vectorized_environment` — SB3-compatible vectorised env.
- :func:`~traffic_drl.environment.make_env.close_environment` — close env and SUMO process.
- :class:`~traffic_drl.environment.scenario_factory.ScenarioManifest` — CSV-backed scenario catalog.
- :class:`~traffic_drl.environment.wrappers.MultiScenarioWrapper` — per-episode route selection.
- :class:`~traffic_drl.environment.wrappers.MetricsInfoWrapper` — metric accumulation in info dict.
- :func:`~traffic_drl.environment.wrappers.wrap_environment` — apply wrappers in the correct order.
"""
from traffic_drl.environment.make_env import (
    create_sumo_env,
    make_dev_environment,
    make_vectorized_environment,
    close_environment,
)
from traffic_drl.environment.scenario_factory import ScenarioManifest
from traffic_drl.environment.wrappers import (
    MultiScenarioWrapper,
    MetricsInfoWrapper,
    wrap_environment,
)

__all__ = [
    "create_sumo_env",
    "make_dev_environment",
    "make_vectorized_environment",
    "close_environment",
    "ScenarioManifest",
    "MultiScenarioWrapper",
    "MetricsInfoWrapper",
    "wrap_environment",
]
