"""Gymnasium wrappers for scenario selection and metric collection."""
from __future__ import annotations

from typing import Any

import gymnasium as gym


class MultiScenarioWrapper(gym.Wrapper):
    """Create one selected route per episode without concatenating route files."""

    def __init__(self, env: gym.Env, scenario_sampler: Any) -> None:
        super().__init__(env)
        self.scenario_sampler = scenario_sampler
        self.active_scenario: Any = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """Select a scenario before delegating to the wrapped environment.

        Returns:
            tuple[Any, dict[str, Any]]: Initial observation and info containing
            the selected scenario metadata.
        """
        raise NotImplementedError

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Delegate one Gymnasium step and attach scenario metadata.

        Returns:
            tuple[Any, float, bool, bool, dict[str, Any]]: Observation, reward,
            termination flags, and diagnostic info.
        """
        raise NotImplementedError


class MetricsInfoWrapper(gym.Wrapper):
    """Expose traffic metrics in the Gymnasium info mapping."""

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """Reset the environment and initialize metric collection.

        Returns:
            tuple[Any, dict[str, Any]]: Initial observation and metric info.
        """
        raise NotImplementedError

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Collect metrics while preserving the Gymnasium step contract.

        Returns:
            tuple[Any, float, bool, bool, dict[str, Any]]: Standard Gymnasium
            transition with collected metrics in ``info``.
        """
        raise NotImplementedError


def wrap_environment(env: gym.Env, scenario_sampler: Any | None = None) -> gym.Env:
    """Apply the project wrappers in their required order.

    Returns:
        gym.Env: Wrapped environment preserving the Gymnasium API.
    """
    raise NotImplementedError
