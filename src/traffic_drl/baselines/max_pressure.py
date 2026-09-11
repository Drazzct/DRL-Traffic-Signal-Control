"""Max-pressure heuristic controller — satisfies the :class:`~traffic_drl.contracts.Controller` protocol.

Max-pressure selects the legal phase whose incoming minus outgoing vehicle
counts (pressure) is highest.  It is a classic adaptive baseline for traffic
signal control.

Typical usage
-------------
::

    from traffic_drl.baselines.max_pressure import MaxPressureController
    from traffic_drl.evaluation.evaluate_benchmark import evaluate_controller

    controller = MaxPressureController(min_green=5, yellow_time=2)
    metrics = evaluate_controller(controller, env, controller_name="max_pressure",
                                  scenario_id=record.scenario_id, seed=record.sumo_seed)
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

from traffic_drl.contracts import EpisodeMetrics


class MaxPressureController:
    """Select the legal phase with the highest pressure.

    Satisfies the :class:`~traffic_drl.contracts.Controller` protocol.

    Attributes:
        min_green: Minimum steps the current phase must stay green before a
            switch is allowed.
        yellow_time: Clearance steps inserted during a phase transition.

    TODO (SV3): implement the pressure computation from the observation vector
    and verify it respects the same min-green safety constraint as the DRL agent.
    """

    def __init__(self, min_green: int, yellow_time: int) -> None:
        """Configure legal phase-transition timing.

        Args:
            min_green: Minimum green duration (in environment steps) before a
                phase switch is allowed.
            yellow_time: Clearance duration (in steps) inserted during a change.
        """
        self.min_green = min_green
        self.yellow_time = yellow_time
        self._current_phase: int = 0
        self._steps_in_phase: int = 0

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> int:
        """Return the legal discrete phase action with the highest pressure.

        Args:
            observation: Flat numeric observation vector from the SUMO-RL
                environment.  The vector must contain queue/pressure features
                in the order defined by the active
                :class:`~traffic_drl.environment.custom_observations.MixedTrafficObservation`.
            deterministic: Unused; max-pressure is inherently deterministic.

        TODO (SV3): extract lane-level queue counts from *observation* and
        implement the pressure formula: pressure(p) = sum(incoming) - sum(outgoing).

        Returns:
            int: The index of the phase with maximum pressure, respecting the
            min-green constraint.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset internal phase tracking at the start of an episode.

        Returns:
            None.
        """
        self._current_phase = 0
        self._steps_in_phase = 0
