"""ObservationFunction implementations for SUMO-RL.

Each class subclasses SUMO-RL's ``ObservationFunction`` and produces a
fixed-shape ``numpy`` observation vector consumed by the SB3 policy network.

Feature ordering must be frozen before any model training begins; once frozen it
must match the ``observation_space`` exactly.

TODO (SV2): document and freeze the exact feature ordering for the MDP contract
before starting Phase 1 training.
"""
from __future__ import annotations

from typing import Sequence, cast

import numpy as np
from gymnasium import spaces
from sumo_rl.environment.observations import ObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal
from traci._trafficlight import TrafficLightDomain
from traci._edge import EdgeDomain
from traci._lane import LaneDomain
from traci._vehicle import VehicleDomain

PCU_MAP = {
    "passenger": 1.0,
    "bus": 2,
    "truck": 2.5,
    "motorcycle": 0.3,
    "emergency": 1.0,
}

Q_MAX_BY_CLASS = {
    "passenger": 15.0,
    "bus": 3.0,
    "truck": 2.0,
    "motorcycle": 75.0,
    "emergency": 2.0,
}
Q_MAX = Q_MAX_BY_CLASS["passenger"]
METERS_PER_PCU = 5


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
        ring_segment_ids: Sequence[str] = (),
        vehicle_classes: Sequence[str] = ("passenger", "bus", "truck"),
    ) -> None:
        """Configure an observation function for one SUMO-RL signal.

        Args:
            ts: SUMO-RL traffic signal whose state is observed.
            approach_ids: Stable approach names used for feature ordering.
            include_elapsed_time: Append elapsed phase time as the final feature.
        """
        super().__init__(ts)
        self.approach_ids = tuple(approach_ids)
        self.ring_segment_ids = tuple(ring_segment_ids)
        self.vehicle_classes = tuple(vehicle_classes)

        # Dimensions N, M, C are fixed at construction time from the provided IDs.
        # K (num_green_phases) and max_green are read lazily via properties because
        # TrafficSignal.__init__ instantiates the observation class BEFORE calling
        # _build_phases(), so num_green_phases is not yet set on the ts object.
        self.N = len(self.approach_ids)
        self.M = len(self.ring_segment_ids)
        self.C = len(self.vehicle_classes)

    @property
    def K(self) -> int:
        """Number of green phases -- read lazily so _build_phases() has time to run."""
        return self.ts.num_green_phases

    @property
    def max_green(self) -> int:
        """Max green time -- read lazily for the same reason as K."""
        return self.ts.max_green
        
    def __call__(self) -> np.ndarray:
        """Return the ordered observation vector consumed by an SB3 policy.

        Returns:
            np.ndarray: A one-dimensional float32 array with shape
            ``(observation_space.shape[0],)``.
        """
        observation = []
        
        # Constants from MDP_DESIGN.md

        # 1. Inbound approach: f_inbound (N * (1 + C))
        for approach_id in self.approach_ids:
            lanes_count = cast(EdgeDomain, self.ts.sumo.edge).getLaneNumber(approach_id)
            lane_ids = [f"{approach_id}_{i}" for i in range(lanes_count)]
            
            total_pcu = 0.0
            total_capacity_pcu = 0.0
            queue_by_class = {c: 0 for c in self.vehicle_classes}
            
            for lane_id in lane_ids:
                lane_length = cast(LaneDomain, self.ts.sumo.lane).getLength(lane_id)
                total_capacity_pcu += lane_length / METERS_PER_PCU
                
                vehicle_ids = cast(LaneDomain, self.ts.sumo.lane).getLastStepVehicleIDs(lane_id)
                for vehicle_id in vehicle_ids:
                    v_class = cast(VehicleDomain, self.ts.sumo.vehicle).getVehicleClass(vehicle_id)
                    speed = cast(VehicleDomain, self.ts.sumo.vehicle).getSpeed(vehicle_id)
                    
                    # PCU aggregation for density
                    pcu_val = PCU_MAP.get(v_class, 1.0)
                    total_pcu += pcu_val
                    
                    # Queue aggregation for halting vehicles (< 0.1 m/s)
                    if speed < 0.1 and v_class in queue_by_class:
                        queue_by_class[v_class] += 1
            
            # Feature 1: PCU Density (rho_i)
            density = min(1.0, total_pcu / total_capacity_pcu) if total_capacity_pcu > 0 else 0.0
            observation.append(density)
            
            # Feature 2: Queue Vector (q_i) across C classes
            for c in self.vehicle_classes:
                q_ratio = min(1.0, queue_by_class[c] / Q_MAX_BY_CLASS.get(c, Q_MAX_BY_CLASS["passenger"]))
                observation.append(q_ratio)

        # 2. Ring segments: f_circulatory (M)
        # Note: We DO NOT aggregate them! The MDP specifies M separate features, one for each segment.
        for ring_id in self.ring_segment_ids:
            # Get current average speed on the edge
            current_speed = cast(EdgeDomain, self.ts.sumo.edge).getLastStepMeanSpeed(ring_id)
            
            # To get max speed, we check the speed limit of its 0-th lane
            lane0 = f"{ring_id}_0"
            max_speed = cast(LaneDomain, self.ts.sumo.lane).getMaxSpeed(lane0)
            
            # TraCI returns -1.0 if the edge is empty (no vehicles).
            # If empty, vehicles would theoretically travel at free-flow max speed.
            if current_speed < 0:
                current_speed = max_speed
                
            norm_speed = min(1.0, current_speed / max_speed) if max_speed > 0 else 0.0
            observation.append(norm_speed)

        # 3. Active Phase One-Hot: p_t (K)
        current_phase = self.ts.green_phase
        for i in range(self.K):
            observation.append(1.0 if i == current_phase else 0.0)

        # 4. Elapsed Green Ratio: tau_green (1)
        tau_green = min(1.0, self.ts.time_since_last_phase_change / self.max_green)
        observation.append(tau_green)

        return np.array(observation, dtype=np.float32)
    def observation_space(self) -> spaces.Box:
        """Return the ``Box`` space matching the shape of ``__call__()`` exactly.

        Returns:
            spaces.Box: Bounded float32 space with the correct feature count.
        """
        observation_size = self.N * (self.C + 1) + self.M + self.K + 1
        return spaces.Box(
            low = np.zeros(observation_size, dtype=np.float32),
            high = np.ones(observation_size, dtype=np.float32)
        )


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
