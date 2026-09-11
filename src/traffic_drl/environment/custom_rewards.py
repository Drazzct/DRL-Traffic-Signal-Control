"""Reward function signatures for SUMO-RL traffic signal control.

Each function accepts a SUMO-RL ``TrafficSignal`` instance and returns a scalar
``float``.  They are passed as the ``reward_fn`` argument to SUMO-RL's
``SumoEnvironment``.

Reward weights and the chosen combination should be recorded in the experiment
config (``reward.type`` and ``reward.params``) so that results are reproducible.
"""
from __future__ import annotations

from typing import Mapping

from sumo_rl.environment.traffic_signal import TrafficSignal


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


def combined_reward(
    traffic_signal: TrafficSignal,
    *,
    queue_weight: float = 1.0,
    pressure_weight: float = 1.0,
    wait_weight: float = 1.0,
    emissions_weight: float = 0.0,
    switch_penalty: float = 0.0,
    context: Mapping[str, float | int | bool] | None = None,
) -> float:
    """Combine normalised traffic objectives and a phase-switch penalty.

    Weights should be set via ``reward.params`` in the training config and
    validated to ensure the reward surface does not hide traffic regressions
    (e.g. a large emissions weight masking throughput drops).

    Args:
        traffic_signal: SUMO-RL signal providing current traffic state.
        queue_weight: Weight applied to :func:`queue_loss_reward`.
        pressure_weight: Weight applied to :func:`pressure_reward`.
        wait_weight: Weight applied to :func:`wait_difference_reward`.
        emissions_weight: Weight applied to :func:`emissions_penalty`.
        switch_penalty: Additional penalty applied when the phase changes.
        context: Optional step context, e.g. ``{"previous_action": 2}``.

    TODO (SV2/SV3): record chosen weights in the experiment config; validate
    that reward improvement correlates with improved primary metrics.

    Returns:
        float: Weighted scalar reward passed to SUMO-RL.
    """
    raise NotImplementedError
