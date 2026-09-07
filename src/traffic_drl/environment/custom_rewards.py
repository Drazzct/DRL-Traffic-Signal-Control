"""Reward function signatures for SUMO-RL traffic signal control."""
from __future__ import annotations

from typing import Any, Mapping


def queue_loss_reward(traffic_signal: Any) -> float:
    """Return the negative change in cumulative queue loss.

    Args:
        traffic_signal: SUMO-RL ``TrafficSignal`` supplying lane queues.

    TODO (SV2): validate the reward against queue and waiting-time metrics.

    Returns:
        float: Scalar reward contribution for the current control step.
    """
    raise NotImplementedError


def pressure_reward(traffic_signal: Any) -> float:
    """Return a normalized pressure reward for one SUMO-RL signal.

    Args:
        traffic_signal: SUMO-RL signal whose incoming/outgoing pressure is measured.

    Returns:
        float: Normalized pressure reward for the current control step.
    """
    raise NotImplementedError


def wait_difference_reward(traffic_signal: Any) -> float:
    """Return the change in cumulative waiting time as a reward.

    Args:
        traffic_signal: SUMO-RL signal exposing cumulative delay information.

    Returns:
        float: Waiting-time difference reward for the current control step.
    """
    raise NotImplementedError


def emissions_penalty(traffic_signal: Any) -> float:
    """Return an emissions-derived reward component.

    Args:
        traffic_signal: SUMO-RL signal connected to SUMO emissions data.

    TODO (SV1): confirm emissions output and units before using this term.

    Returns:
        float: Emissions penalty contribution, normally non-positive.
    """
    raise NotImplementedError


def combined_reward(
    traffic_signal: Any,
    *,
    queue_weight: float = 1.0,
    pressure_weight: float = 1.0,
    wait_weight: float = 1.0,
    emissions_weight: float = 0.0,
    switch_penalty: float = 0.0,
    context: Mapping[str, Any] | None = None,
) -> float:
    """Combine normalized traffic objectives and a phase-switch penalty.

    Args:
        traffic_signal: SUMO-RL signal providing current traffic state.
        queue_weight: Weight for queue-loss reduction.
        pressure_weight: Weight for pressure reduction.
        wait_weight: Weight for waiting-time reduction.
        emissions_weight: Weight for fuel/CO2/NOx penalty.
        switch_penalty: Penalty applied to unnecessary phase changes.
        context: Optional previous-action and normalization metadata.

    TODO (SV2/SV3): record chosen weights in the experiment config and validate
    that reward improvement does not hide traffic regressions.

    Returns:
        float: Weighted scalar reward passed to SUMO-RL.
    """
    raise NotImplementedError
