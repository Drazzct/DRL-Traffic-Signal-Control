"""Factories for the SUMO-RL Gymnasium environment."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def create_sumo_env(config: Any, route_file: str | Path, *, fixed_ts: bool = False, seed: int | None = None, use_gui: bool = False) -> Any:
    """Call SUMO-RL's registered ``sumo-rl-v0`` environment for one route.

    Args:
        config: Network, timing, observation, reward, and SUMO options.
        route_file: Exactly one route XML file for this environment instance.
        fixed_ts: Use the fixed-time baseline when true.
        seed: Reproducibility seed for demand/SUMO behavior.
        use_gui: Launch SUMO-GUI instead of headless SUMO.

    Output contract: the runner must create ``outputs/tripinfo/<run_id>/``
    and pass SUMO's ``--tripinfo-output`` path, plus ``--emission-output``
    when environmental metrics are enabled. SUMO creates those XML files;
    ``evaluation/parse_tripinfo.py`` reads them after the process closes.

    TODO (SV1/SV2): map config fields to the SUMO-RL constructor and verify the
    route is never passed as a comma-joined multi-route list. TODO (SV1):
    decide whether the output options are supplied through `additional_sumo_cmd`
    or explicit SUMO-RL constructor arguments.

    Returns:
        Any: A Gymnasium-compatible SUMO-RL environment with action and
        observation spaces.
    """
    raise NotImplementedError


def make_dev_environment(config: Any, *, seed: int | None = None, use_gui: bool = False) -> Any:
    """Create the short DEV-00 smoke-test environment.

    Args:
        config: Environment configuration containing the DEV-00 route.
        seed: Development seed for the pilot.
        use_gui: Whether to show SUMO-GUI during the smoke test.

    TODO (SV2): connect this factory to the DEV-00 manifest record.

    Returns:
        Any: A short-lived Gymnasium environment for the DEV-00 smoke test.
    """
    raise NotImplementedError


def make_vectorized_environment(config: Any, route_file: str | Path, *, normalize_observations: bool = True, normalize_rewards: bool = False, seed: int | None = None) -> Any:
    """Create a DummyVecEnv and optional VecNormalize wrapper for SB3.

    Args:
        config: SUMO-RL environment configuration.
        route_file: One selected route XML file.
        normalize_observations: Fit/load observation normalization statistics.
        normalize_rewards: Whether reward normalization is enabled during training.
        seed: Seed forwarded to the underlying environment.

    TODO (SV2): fit VecNormalize only on TR and freeze it for VA/TE evaluation.

    Returns:
        Any: A ``DummyVecEnv`` optionally wrapped by ``VecNormalize``.
    """
    raise NotImplementedError


def close_environment(env: Any) -> None:
    """Close the environment and its SUMO process.

    Args:
        env: Gymnasium, vectorized, or wrapped SUMO-RL environment to close.

    Returns:
        None: The environment and its external SUMO process are closed.
    """
    raise NotImplementedError
