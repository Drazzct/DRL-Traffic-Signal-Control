"""ObservationFunction implementations for SUMO-RL.

Each class subclasses SUMO-RL's ``ObservationFunction`` and produces a
fixed-shape ``numpy`` observation vector consumed by the SB3 policy network.

Feature ordering must be frozen before any model training begins; once frozen it
must match the ``observation_space`` exactly.

TODO (SV2): document and freeze the exact feature ordering for the MDP contract
before starting Phase 1 training.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from gymnasium import spaces
from sumo_rl.environment.observations import ObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal


class MixedTrafficObservation(ObservationFunction):
    """SUMO-RL observation function for queue, density, speed, and phase data.

    Produces a flat ``numpy`` vector containing per-approach features in a
    stable, documented order.  The exact ordering is fixed by
    :attr:`approach_ids` and must be synchronised with the active
    :class:`~traffic_drl.baselines.max_pressure.MaxPressureController`
    observation mapping.

    Attributes:
        approach_ids: Stable sequence of approach (lane-group) identifiers used
            for feature ordering.  Must match the SUMO network geometry.
        include_elapsed_time: If ``True``, appends the number of steps the
            current phase has been held as the last feature.

    TODO (SV2): document and freeze the exact feature ordering for the MDP
    contract before Phase 1 training starts.
    """

    def __init__(
        self,
        ts: TrafficSignal,
        approach_ids: Sequence[str] = (),
        include_elapsed_time: bool = True,
    ) -> None:
        """Configure an observation function for one SUMO-RL signal.

        Args:
            ts: SUMO-RL traffic signal whose state is observed.
            approach_ids: Stable approach names used for feature ordering.
            include_elapsed_time: Append elapsed phase time as the final feature.
        """
        super().__init__(ts)
        self.approach_ids: tuple[str, ...] = tuple(approach_ids)
        self.include_elapsed_time: bool = include_elapsed_time

    def __call__(self) -> np.ndarray:
        """Return the ordered observation vector consumed by an SB3 policy.

        Returns:
            np.ndarray: A one-dimensional float32 array with shape
            ``(observation_space.shape[0],)``.

        TODO (SV2): iterate ``approach_ids``, read queue / density / speed per
        approach from ``self.ts``, and append elapsed phase time if
        ``include_elapsed_time`` is True.
        """
        raise NotImplementedError

    def observation_space(self) -> spaces.Box:
        """Return the ``Box`` space matching the shape of ``__call__()`` exactly.

        Returns:
            spaces.Box: Bounded float32 space with the correct feature count.

        TODO (SV2): compute the feature count from ``len(approach_ids)`` and the
        number of features per approach, plus one for elapsed time if enabled.
        """
        raise NotImplementedError


class QueueObservation(MixedTrafficObservation):
    """Minimal queue-and-phase observation for the DEV-00 pilot.

    A simplified subset of :class:`MixedTrafficObservation` that uses only
    per-approach queue lengths and the current phase index.  Used during the
    DEV-00 smoke-test phase before the full feature set is decided.

    TODO (SV2): freeze the feature count and bounds once the DEV-00 network
    geometry is confirmed.
    """

    def __call__(self) -> np.ndarray:
        """Return the minimal queue and phase feature vector for DEV-00.

        Returns:
            np.ndarray: One-dimensional float32 array ``[queue_0, ..., queue_n, phase_index]``.

        TODO (SV2): read per-approach queue counts from ``self.ts.get_lanes_queue``
        and append the current phase index.
        """
        raise NotImplementedError
