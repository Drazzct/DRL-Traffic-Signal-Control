"""Factories for the SUMO-RL Gymnasium environment.

This module is the **canonical** owner of ``create_sumo_env``.  All other
modules that need a SUMO-RL environment should call this factory rather than
constructing one directly.

Dependency graph
----------------
::

    EnvConfig ──► create_sumo_env ──► gym.Env
                       │
                       └─► make_dev_environment   (smoke tests / DEV-00)
                       └─► make_vectorized_environment  (SB3 training)

Typical usage
-------------
::

    from traffic_drl.config import load_env_config
    from traffic_drl.environment.make_env import create_sumo_env

    config = load_env_config("configs/environment/dev_single_intersection.yaml")
    env = create_sumo_env(config, route_file="scenarios/TR-01.rou.xml", seed=42)
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from traffic_drl.config import EnvConfig, load_env_config


# Type aliases for SB3 vectorised environments.  We use strings here so the
# module can be imported even when stable-baselines3 is not installed.
# At runtime the real types are resolved inside the functions that use them.
try:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # type: ignore[import]

    VecEnv = DummyVecEnv | VecNormalize
except ModuleNotFoundError:
    VecEnv = None  # type: ignore[assignment,misc]


def create_sumo_env(
    config: EnvConfig | str | Path,
    route_file: str | Path,
    *,
    fixed_ts: bool = False,
    seed: int | None = None,
    use_gui: bool = False,
    reward_fn: str = "pressure",
) -> gym.Env:
    """Create a single-agent SUMO-RL Gymnasium environment for one route.

    This is the canonical environment factory.  It accepts either a fully
    parsed :class:`~traffic_drl.config.EnvConfig` or a path to a YAML file
    (which is loaded automatically via :func:`~traffic_drl.config.load_env_config`).

    Args:
        config: Typed environment configuration or path to its YAML file.
        route_file: Exactly **one** ``.rou.xml`` route file.  Must not be a
            comma-joined multi-route string; SUMO-RL handles multi-route via
            the manifest + wrapper, not via concatenation.
        fixed_ts: Run the pre-timed signal plan instead of agent control.
            Pass ``True`` when creating the fixed-time baseline environment.
        seed: Reproducibility seed for SUMO demand and behaviour.
        use_gui: Launch ``sumo-gui`` instead of headless ``sumo``.
        reward_fn: Reward function name accepted by SUMO-RL (e.g. ``"pressure"``
            or ``"queue"``).  Custom reward functions should be registered with
            SUMO-RL before calling this factory.

    Output contract: the runner must create ``outputs/tripinfo/<run_id>/`` and
    pass SUMO's ``--tripinfo-output`` and ``--emission-output`` paths via
    ``config.sumo_options.additional_sumo_cmd`` or the SUMO-RL constructor
    kwargs before this factory is called.

    TODO (SV1): map every ``EnvConfig`` field to the SUMO-RL ``SumoEnvironment``
    constructor.  Decide whether ``additional_sumo_cmd`` is passed as a string
    or split into a list.  Verify that ``route_file`` is never a comma-joined
    multi-route value.

    Returns:
        gym.Env: A Gymnasium-compatible SUMO-RL single-agent environment.
    """
    if isinstance(config, (str, Path)):
        config = load_env_config(config)

    raise NotImplementedError


def make_dev_environment(
    config: EnvConfig | str | Path = "configs/environment/dev_single_intersection.yaml",
    *,
    seed: int | None = None,
    use_gui: bool = False,
) -> gym.Env:
    """Create the short DEV-00 smoke-test environment.

    Wraps :func:`create_sumo_env` with the DEV-00 route from the manifest.
    Intended for quick import/step checks, not for training or evaluation.

    Args:
        config: Typed environment configuration or path to its YAML file.
        seed: Development seed for the pilot smoke test.
        use_gui: Whether to show SUMO-GUI during the smoke test.

    TODO (SV2): load the DEV-00 manifest record and pass its ``route_file``
    to :func:`create_sumo_env`.

    Returns:
        gym.Env: A short-lived Gymnasium environment for the DEV-00 smoke test.
    """
    if isinstance(config, (str, Path)):
        config = load_env_config(config)

    raise NotImplementedError


def make_vectorized_environment(
    config: EnvConfig | str | Path,
    route_file: str | Path,
    *,
    normalize_observations: bool = True,
    normalize_rewards: bool = False,
    seed: int | None = None,
) -> "DummyVecEnv | VecNormalize":
    """Create a ``DummyVecEnv`` and optional ``VecNormalize`` wrapper for SB3.

    The returned object is the ``env`` argument accepted by
    :func:`~traffic_drl.train.train_dqn.build_dqn_model`.

    Args:
        config: Typed environment configuration or path to its YAML file.
        route_file: One route XML file selected from the training split.
        normalize_observations: Wrap with ``VecNormalize`` and fit running
            statistics on training data only.
        normalize_rewards: Enable reward normalisation during training.
        seed: Seed forwarded to the underlying SUMO-RL environment.

    TODO (SV2): fit ``VecNormalize`` only on ``TR`` episodes and freeze it
    (``training=False``, ``norm_reward=False``) for ``VA``/``TE`` evaluation.

    Returns:
        DummyVecEnv | VecNormalize: A single-environment vectorised wrapper,
        optionally wrapped by ``VecNormalize``.
    """
    if isinstance(config, (str, Path)):
        config = load_env_config(config)

    raise NotImplementedError


def close_environment(env: gym.Env) -> None:
    """Close the environment and terminate its SUMO subprocess.

    Calls ``env.close()`` which SUMO-RL forwards to the running SUMO process.
    Callers must invoke this after every episode loop to prevent orphaned SUMO
    processes.

    Args:
        env: Any Gymnasium, vectorised, or wrapped SUMO-RL environment.

    TODO (SV1): verify that ``VecNormalize`` / ``DummyVecEnv`` correctly
    propagate ``close()`` to the inner SUMO-RL environment.

    Returns:
        None.
    """
    raise NotImplementedError
