"""Reward function signatures for SUMO-RL traffic signal control.

Each function accepts a SUMO-RL ``TrafficSignal`` instance and returns a scalar
``float``.  They are passed as the ``reward_fn`` argument to SUMO-RL's
``SumoEnvironment``.

Reward weights and the chosen combination should be recorded in the experiment
config (``reward.type`` and ``reward.params``) so that results are reproducible.
"""
from __future__ import annotations

from typing import Mapping, Callable

from sumo_rl.environment.traffic_signal import TrafficSignal

from traffic_drl.environment.custom_observations import PCU_MAP


def queue_loss_reward(traffic_signal: TrafficSignal) -> float:
    """Return the negative change in cumulative queue loss.

    A positive return means total queue decreased (improvement); negative means
    it increased (deterioration).

    Args:
        traffic_signal: SUMO-RL ``TrafficSignal`` supplying per-lane queues via
            ``get_lanes_queue()``.

    TODO (SV2): validate the reward magnitude against waiting-time metrics to
    ensure reward and evaluation metrics are consistent.

    Returns:
        float: Scalar reward contribution for the current control step.
    """
    raise NotImplementedError


def pressure_reward(traffic_signal: TrafficSignal) -> float:
    """Return a normalised pressure reward for one SUMO-RL signal.

    Pressure is defined as the difference between the number of vehicles on
    incoming and outgoing lanes.  A lower pressure indicates better throughput.

    Args:
        traffic_signal: SUMO-RL signal whose incoming/outgoing pressure is measured.

    Returns:
        float: Normalised pressure reward for the current control step.
    """
    raise NotImplementedError


def wait_difference_reward(traffic_signal: TrafficSignal) -> float:
    """Return the change in cumulative waiting time as a reward signal.

    A positive value means total waiting time decreased compared to the previous
    step (vehicles cleared faster).

    Args:
        traffic_signal: SUMO-RL signal exposing per-vehicle waiting-time data.

    Returns:
        float: Waiting-time difference for the current control step.
    """
    raise NotImplementedError


def emissions_penalty(traffic_signal: TrafficSignal) -> float:
    """Return an emissions-derived penalty (always non-positive).

    Fuel and exhaust emissions are accumulated across all vehicles in the
    controlled area.  The penalty is non-positive so that lower emissions lead
    to a higher (less negative) reward contribution.

    Args:
        traffic_signal: SUMO-RL signal connected to SUMO emissions data.

    TODO (SV1): confirm the exact emissions output schema, attribute names,
    and unit conversion used by SUMO before enabling this term.

    Returns:
        float: Emissions penalty contribution (≤ 0).
    """
    raise NotImplementedError

# Pass the object to the reward_fn field
class CombinedReward:
    """Combine normalised traffic objectives and a phase-switch penalty.

    Weights should be set via ``reward.params`` in the training config and
    validated to ensure the reward surface does not hide traffic regressions
    (e.g. a large emissions weight masking throughput drops).

    Args:
        wait_weight: Weight applied to PCU delay (W_t).
        queue_weight: Weight applied to inbound queue (Q_t).
        switch_penalty: Additional penalty weight applied when the phase changes.
        delta_time: The simulation step size in seconds, used to track stall duration.
        v_stall: Velocity threshold in m/s below which a lane is considered stalled.
        t_stall_limit: Time limit in seconds before asserting deadlock penalty.
        deadlock_penalty: Constant penalty applied when the network is deadlocked.
        lambda_f: Exponential decay factor for flicker penalty.

    TODO (SV2/SV3): record chosen weights in the experiment config; validate
    that reward improvement correlates with improved primary metrics.
    """

    def __init__(
        self,
        wait_weight: float = 1.5,
        queue_weight: float = 2.0,
        switch_penalty: float = 1.0,
        delta_time: float = 5.0,
        v_stall: float = 0.1,
        t_stall_limit: float = 120.0,
        deadlock_penalty: float = 5.0,
        lambda_f: float = 0.9,
    ):
        self.wait_weight = wait_weight
        self.queue_weight = queue_weight
        self.switch_penalty = switch_penalty
        self.delta_time = delta_time
        self.v_stall = v_stall
        self.t_stall_limit = t_stall_limit
        self.deadlock_penalty = deadlock_penalty
        self.lambda_f = lambda_f

        # Persisted state between environment steps
        self.flicker_state = 0.0
        self.last_phase = None
        self.stall_time = 0.0

    def __call__(self, traffic_signal: TrafficSignal) -> float:
        """Calculate and return the scalar reward for the current step.

        Args:
            traffic_signal: SUMO-RL signal providing current traffic state.

        Returns:
            float: Weighted scalar reward passed to SUMO-RL.
        """
        ts = traffic_signal

        # 1. PCU Delay (W_t)
        # Sum of PCU-weighted waiting times across all vehicles in incoming lanes
        total_wait = 0.0
        for lane in ts.lanes:
            for veh in ts.sumo.lane.getLastStepVehicleIDs(lane):
                wait_time = ts.sumo.vehicle.getAccumulatedWaitingTime(veh)
                v_class = ts.sumo.vehicle.getVehicleClass(veh)
                pcu_val = PCU_MAP.get(v_class, 1.0)
                total_wait += pcu_val * wait_time

        # 2. Total Inbound Queue (Q_t)
        # Total halting vehicles across all incoming lanes
        total_queue = sum(ts.sumo.lane.getLastStepHaltingNumber(lane) for lane in ts.lanes)

        # 3. Flicker Penalty (P_flicker)
        current_phase = ts.green_phase
        if self.last_phase is None:
            self.last_phase = current_phase
            
        phase_changed = 1.0 if current_phase != self.last_phase else 0.0
        self.last_phase = current_phase

        self.flicker_state = self.lambda_f * self.flicker_state + phase_changed
        f_norm = 1.0 / (1.0 - self.lambda_f)
        p_flicker = self.switch_penalty * (self.flicker_state / f_norm)

        # 4. Deadlock Penalty (P_deadlock)
        # Calculate average speed on inbound lanes as a proxy for standstill
        speeds = [ts.sumo.lane.getLastStepMeanSpeed(lane) for lane in ts.lanes]
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
        
        if avg_speed < self.v_stall:
            self.stall_time += self.delta_time
        else:
            self.stall_time = 0.0
        
        p_deadlock = self.deadlock_penalty if self.stall_time > self.t_stall_limit else 0.0

        # 5. Final Reward Calculation
        r_t = (-self.wait_weight * total_wait) - (self.queue_weight * total_queue) - p_flicker - p_deadlock
        return r_t