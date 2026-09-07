"""ObservationFunction implementations for SUMO-RL."""
from __future__ import annotations

from typing import Any, Sequence

from gymnasium import spaces
from sumo_rl.environment.observations import ObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal


class MixedTrafficObservation(ObservationFunction):
    """SUMO-RL observation function for queue, density, speed and phase data."""

    def __init__(self, ts: TrafficSignal, approach_ids: Sequence[str] = (), include_elapsed_time: bool = True) -> None:
        """Configure an observation function for one SUMO-RL signal.

        Args:
            ts: SUMO-RL traffic signal whose state is observed.
            approach_ids: Stable approach names used for feature ordering.
            include_elapsed_time: Include time held in the current phase.

        TODO (SV2): document and freeze the exact feature ordering for the MDP contract.
        """
        super().__init__(ts)
        self.approach_ids = tuple(approach_ids)
        self.include_elapsed_time = include_elapsed_time

    def __call__(self) -> Any:
        """Return the ordered observation vector consumed by an SB3 policy.

        Returns:
            Any: A fixed-shape numeric observation accepted by the declared space.
        """
        raise NotImplementedError

    def observation_space(self) -> spaces.Space[Any]:
        """Return bounds and shape for the ordered observation vector.

        Returns:
            spaces.Space[Any]: Gymnasium space matching ``__call__()`` exactly.
        """
        raise NotImplementedError


class QueueObservation(MixedTrafficObservation):
    """Minimal queue/phase observation used by the DEV-00 pilot."""

    def __call__(self) -> Any:
        """Return the minimal queue and phase feature vector for DEV-00.

        Returns:
            Any: Fixed-shape queue/phase observation for the pilot environment.
        """
        raise NotImplementedError
