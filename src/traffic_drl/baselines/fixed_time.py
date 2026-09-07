"""Fixed-time controller baseline using the same SUMO-RL route and seeds."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from traffic_drl.contracts import EpisodeMetrics


def create_fixed_time_environment(config: Any, route_file: str | Path, *, seed: int | None = None, use_gui: bool = False) -> Any:
    """Create a fixed-time SUMO-RL environment.

    Args:
        config: Network and timing configuration shared with DRL evaluation.
        route_file: SUMO ``.rou.xml`` with ``<route>`` and ``<vehicle>`` elements;
            validate it with SUMO's route parser.
        seed: Same seed used by the competing controller.
        use_gui: Whether to launch SUMO-GUI.

    TODO (SV3): confirm ``fixed_ts=True`` and identical route/seed inputs.

    Returns:
        Any: A SUMO-RL environment configured for fixed-time control.
    """
    raise NotImplementedError


def run_fixed_time_episode(env: Any, *, seed: int | None = None) -> EpisodeMetrics:
    """Run one fixed-time episode and return typed metrics.

    Args:
        env: Fixed-time SUMO-RL environment.
        seed: Seed used when resetting the environment.

    Returns:
        EpisodeMetrics: Typed traffic/control metrics for the episode.
    """
    raise NotImplementedError


def evaluate_fixed_time(config: Any, route_files: Iterable[str | Path], *, seeds: Iterable[int], output_path: str | Path | None = None) -> list[EpisodeMetrics]:
    """Evaluate the fixed-time baseline over routes and matched seeds.

    Args:
        config: Shared environment configuration.
        route_files: SUMO ``.rou.xml`` files containing route/vehicle definitions.
        seeds: Reproducibility seeds shared with other controllers.
        output_path: Optional UTF-8 CSV/JSON destination for ``EpisodeMetrics``.

    Returns:
        list[EpisodeMetrics]: Results for every route and matched seed.
    """
    raise NotImplementedError
