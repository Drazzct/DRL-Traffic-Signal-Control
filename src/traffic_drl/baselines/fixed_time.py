"""Fixed-time signal controller — satisfies the :class:`~traffic_drl.contracts.Controller` protocol.

The fixed-time baseline runs SUMO with ``fixed_ts=True`` so the signal keeps a
pre-programmed phase plan and never responds to traffic state.  It is evaluated
through the same :func:`~traffic_drl.evaluation.evaluate_benchmark.evaluate_controller`
path as every other controller.

Typical usage
-------------
::

    from traffic_drl.baselines.fixed_time import FixedTimeController, make_fixed_time_env
    from traffic_drl.evaluation.evaluate_benchmark import evaluate_controller

    controller = FixedTimeController()
    env = make_fixed_time_env(record, config)
    metrics = evaluate_controller(controller, env, controller_name="fixed_time",
                                  scenario_id=record.scenario_id, seed=record.sumo_seed)
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

from traffic_drl.contracts import ScenarioRecord
from traffic_drl.config import EnvConfig


class FixedTimeController:
    """Controller stub for the fixed-time (pre-timed) signal baseline.

    The fixed-time plan is encoded in the SUMO network/additional files and
    executed by SUMO when ``fixed_ts=True`` is passed to the SUMO-RL
    environment.  ``predict`` is called at each control step but must always
    return the **current** phase (no switching) so the environment's
    ``fixed_ts`` logic takes over.

    Satisfies the :class:`~traffic_drl.contracts.Controller` protocol.

    TODO (SV3): confirm ``fixed_ts=True`` passes through SUMO-RL correctly and
    that the returned action is the identity action for the environment.
    """

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> int:
        """Return the identity (no-switch) action for the current phase.

        Under ``fixed_ts=True`` SUMO-RL ignores the agent action and advances
        the pre-programmed signal plan automatically.  This method exists only
        to satisfy the :class:`~traffic_drl.contracts.Controller` interface so
        that the unified evaluation loop can treat fixed-time like any other
        controller.

        Args:
            observation: Current observation vector (ignored).
            deterministic: Unused; kept for interface compatibility.

        TODO (SV3): verify the correct identity action value for the
        single-intersection environment.

        Returns:
            int: The identity action (phase hold); value is environment-specific.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """No internal state to reset for fixed-time control.

        Returns:
            None.
        """
        # Fixed-time has no learned or accumulated state.
        pass


def make_fixed_time_env(record: ScenarioRecord, config: EnvConfig) -> gym.Env:
    """Create a SUMO-RL environment configured for fixed-time control.

    Passes ``fixed_ts=True`` to the SUMO-RL constructor so that SUMO executes
    the pre-programmed signal plan from the network/additional files.  The
    route file and seeds are taken from *record* to ensure the same traffic
    demand is used across all baselines.

    Args:
        record: Scenario record providing the route file and seeds.
        config: Shared environment configuration (network, timing, SUMO options).

    TODO (SV3): map ``config`` fields to the SUMO-RL constructor and confirm
    that ``fixed_ts=True`` disables agent control correctly.

    Returns:
        gym.Env: A Gymnasium-compatible SUMO-RL environment in fixed-time mode.
    """
    raise NotImplementedError
