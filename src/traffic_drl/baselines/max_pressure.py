"""Max-pressure heuristic baseline API."""
from __future__ import annotations

from typing import Any

from traffic_drl.contracts import EpisodeMetrics


class MaxPressureController:
    """Select the legal phase with the highest pressure."""

    def __init__(self, min_green: int, yellow_time: int) -> None:
        """Configure legal phase-transition timing.

        Args:
            min_green: Minimum green duration before a switch is allowed.
            yellow_time: Clearance duration inserted during a phase change.

        TODO (SV3): verify this heuristic respects the same safety constraints as DRL.
        """
        self.min_green = min_green
        self.yellow_time = yellow_time

    def predict(self, observation: Any, *, deterministic: bool = True) -> Any:
        """Return a legal discrete phase action.

        Args:
            observation: Current queue/pressure observation.
            deterministic: Kept for controller compatibility; heuristic is normally deterministic.

        Returns:
            Any: A legal discrete phase action accepted by the environment.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset controller state at the beginning of an episode.

        Returns:
            None: Internal phase/pressure state is reset.
        """
        raise NotImplementedError


def run_max_pressure_episode(env: Any, controller: MaxPressureController, *, seed: int | None = None) -> EpisodeMetrics:
    """Run one max-pressure episode and return standard metrics.

    Args:
        env: Environment exposing the Gymnasium reset/step contract.
        controller: Configured max-pressure controller.
        seed: Episode seed shared with benchmark peers.

    Returns:
        EpisodeMetrics: Typed traffic/control metrics for the episode.
    """
    raise NotImplementedError
