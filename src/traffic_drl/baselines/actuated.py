"""Actuated signal controller — satisfies the :class:`~traffic_drl.contracts.Controller` protocol.

Actuated control extends the fixed-time plan with vehicle-detection logic:
the green phase is extended while detectors report queued vehicles, up to a
maximum green cap.  SUMO supports actuated control natively via
``<tlLogic type="actuated">``.

Typical usage (once implemented)
---------------------------------
::

    from traffic_drl.baselines.actuated import ActuatedController, make_actuated_env
    from traffic_drl.evaluation.evaluate_benchmark import evaluate_controller

    controller = ActuatedController(min_green=5, max_green=60, yellow_time=2)
    env = make_actuated_env(record, config)
    metrics = evaluate_controller(controller, env, controller_name="actuated",
                                  scenario_id=record.scenario_id, seed=record.sumo_seed)
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

from traffic_drl.contracts import ScenarioRecord
from traffic_drl.config import EnvConfig


class ActuatedController:
    """Actuated signal controller that extends green time based on detector input.

    Satisfies the :class:`~traffic_drl.contracts.Controller` protocol.

    Under SUMO's native actuated mode the signal timing is controlled by the
    simulator; the agent's ``predict`` method returns a no-op action so the
    environment records the actuated behaviour without agent interference.

    Attributes:
        min_green: Minimum green duration before the phase may be extended.
        max_green: Maximum green duration; the phase switches when this is reached.
        yellow_time: Clearance duration during a phase change.

    TODO (SV3): decide whether to use SUMO's native ``tlLogic type="actuated"``
    or implement a custom detector-based logic inside ``predict``.  Document the
    chosen approach and the detector IDs that provide the input signal.
    """

    def __init__(self, min_green: int, max_green: int, yellow_time: int) -> None:
        """Configure actuated phase-timing bounds.

        Args:
            min_green: Minimum green duration (in environment steps).
            max_green: Maximum green cap (in steps) before a forced switch.
            yellow_time: Clearance steps between phase changes.
        """
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self._current_phase: int = 0
        self._steps_in_phase: int = 0

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> int:
        """Return the actuated phase action for the current step.

        Args:
            observation: Flat numeric observation vector from the SUMO-RL
                environment, containing detector occupancy / queue features.
            deterministic: Unused for rule-based actuated control.

        TODO (SV3): read detector occupancy from *observation* and implement
        extend-or-switch logic using ``min_green`` / ``max_green`` bounds.

        Returns:
            int: The current or next phase index decided by the actuated logic.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset internal episode state.

        Returns:
            None.
        """
        self._current_phase = 0
        self._steps_in_phase = 0


def make_actuated_env(record: ScenarioRecord, config: EnvConfig) -> gym.Env:
    """Create a SUMO-RL environment configured for actuated control.

    The SUMO network/additional files must include ``<tlLogic type="actuated">``
    definitions and detector placements.

    Args:
        record: Scenario record providing the route file and seeds.
        config: Shared environment configuration.

    TODO (SV3): map ``config`` fields to the SUMO-RL constructor, confirm the
    actuated ``tlLogic`` is present in the additional files, and verify
    that detector IDs are exposed in the observation.

    Returns:
        gym.Env: A Gymnasium-compatible SUMO-RL environment in actuated mode.
    """
    raise NotImplementedError
