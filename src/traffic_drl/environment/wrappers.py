"""Gymnasium wrappers for per-episode scenario selection and metric collection.

Wrapper stack (applied in order by :func:`wrap_environment`)
-------------------------------------------------------------
1. :class:`MultiScenarioWrapper` — selects a new route from the manifest before
   each episode reset.
2. :class:`MetricsInfoWrapper` — accumulates traffic metrics into the
   ``info`` dict at each step and at episode end.

Typical usage
-------------
::

    from traffic_drl.environment.wrappers import wrap_environment
    from traffic_drl.train.scenario_sampler import ScenarioSampler

    sampler = ScenarioSampler(manifest, allowed_split="TR", seed=42)
    wrapped = wrap_environment(base_env, scenario_sampler=sampler)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

if TYPE_CHECKING:
    # Avoid a hard circular import at runtime; ScenarioSampler imports
    # ScenarioRecord from contracts, which has no env dependency.
    from traffic_drl.train.scenario_sampler import ScenarioSampler
    from traffic_drl.contracts import ScenarioRecord


class MultiScenarioWrapper(gym.Wrapper):
    """Select exactly one route per episode — never a comma-joined route list.

    On each ``reset`` call the wrapper asks ``scenario_sampler.sample()`` for
    the next :class:`~traffic_drl.contracts.ScenarioRecord` and reconfigures
    the underlying SUMO-RL environment for that route.

    Attributes:
        scenario_sampler: Restricted to one manifest split; provides
            :meth:`~traffic_drl.train.scenario_sampler.ScenarioSampler.sample`.
        active_scenario: The :class:`~traffic_drl.contracts.ScenarioRecord`
            selected for the current episode; ``None`` before the first reset.

    TODO (SV2): implement ``reset`` to reconfigure the inner SUMO-RL env for
    the newly selected route file without restarting the SUMO process if
    SUMO-RL's ``reset`` supports route switching.
    """

    def __init__(self, env: gym.Env, scenario_sampler: "ScenarioSampler") -> None:
        super().__init__(env)
        self.scenario_sampler: "ScenarioSampler" = scenario_sampler
        self.active_scenario: "ScenarioRecord | None" = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Sample a new scenario then delegate to the wrapped environment.

        Returns:
            tuple: Initial observation and info dict augmented with
            ``{"scenario": self.active_scenario}``.

        TODO (SV2): pass the sampled route file to the underlying SUMO-RL
        environment via its reset ``options`` or by reconfiguring the env.
        """
        raise NotImplementedError

    def step(
        self,
        action: int | np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        """Delegate one Gymnasium step and attach active scenario metadata.

        Returns:
            tuple: Observation, reward, terminated, truncated, and info dict
            augmented with the current scenario metadata.

        TODO (SV2): forward the step unchanged; only augment ``info``.
        """
        raise NotImplementedError


class MetricsInfoWrapper(gym.Wrapper):
    """Accumulate traffic metrics and expose them in the Gymnasium ``info`` dict.

    At each step the wrapper reads per-lane queue and waiting-time values from
    the SUMO-RL traffic-signal object and accumulates them.  At episode end it
    computes episode-level aggregates and attaches them to ``info`` so that
    :class:`~traffic_drl.train.callbacks.TrafficMetricsCallback` can log them.

    Expected ``info`` keys at episode end
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - ``average_waiting_time`` (float)
    - ``average_queue_length`` (float)
    - ``time_loss`` (float)
    - ``throughput`` (float)
    - ``phase_switch_rate`` (float)
    - ``min_green_violations`` (int)

    TODO (SV2): map these keys to the actual SUMO-RL ``info`` fields returned
    by ``SumoEnvironment.step``.
    """

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Reset the environment and clear accumulated metric state.

        Returns:
            tuple: Initial observation and initial info dict.

        TODO (SV2): clear internal accumulators and delegate to the wrapped env.
        """
        raise NotImplementedError

    def step(
        self,
        action: int | np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        """Collect metrics while preserving the Gymnasium step contract.

        Returns:
            tuple: Standard Gymnasium transition with traffic metrics added
            to ``info`` at every step and summarised at episode end.

        TODO (SV2): call ``super().step(action)``, read metrics from the SUMO-RL
        traffic-signal object, accumulate them, and on ``terminated or truncated``
        compute episode aggregates.
        """
        raise NotImplementedError


def wrap_environment(
    env: gym.Env,
    scenario_sampler: "ScenarioSampler | None" = None,
) -> gym.Env:
    """Apply the project wrappers in their required order.

    Order matters: :class:`MultiScenarioWrapper` must be the outermost wrapper
    (applied last) so that ``reset`` route-switching happens before
    :class:`MetricsInfoWrapper` initialises its accumulators.

    Args:
        env: Base SUMO-RL Gymnasium environment from :func:`~traffic_drl.environment.make_env.create_sumo_env`.
        scenario_sampler: If provided, wraps with :class:`MultiScenarioWrapper`.
            Omit for single-scenario evaluation.

    TODO (SV2): apply :class:`MetricsInfoWrapper` first, then
    :class:`MultiScenarioWrapper` if a sampler is provided.

    Returns:
        gym.Env: Wrapped environment preserving the Gymnasium API.
    """
    raise NotImplementedError
