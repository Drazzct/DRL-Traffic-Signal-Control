"""Smoke-test entry points for the Phase 1 environment.

Use these functions after building a new environment to confirm that reset,
step, and close work before starting a long training run.

Typical usage
-------------
::

    from traffic_drl.evaluation.test import check_environment, run_smoke_test
    from traffic_drl.environment.make_env import make_dev_environment

    env = make_dev_environment()
    check_environment(env)          # SB3 Gymnasium contract checker
    result = run_smoke_test(env)    # short reset → random steps → close
    print(result)
"""
from __future__ import annotations

import gymnasium as gym

from traffic_drl.contracts import SmokeTestResult


def check_environment(env: gym.Env) -> None:
    """Run Stable-Baselines3's Gymnasium contract checker on an environment.

    Skips gracefully when SB3 is not installed so that test collection remains
    usable before project dependencies are set up.

    Args:
        env: The Gymnasium environment to validate.  Should be the unwrapped
            SUMO-RL environment or the full wrapper stack.

    TODO (SV2): call ``stable_baselines3.common.env_checker.check_env(env)``
    and ensure all SUMO-RL-specific spaces pass the checker.

    Returns:
        None.
    """
    raise NotImplementedError


def run_smoke_test(
    env: gym.Env,
    *,
    steps: int = 10,
    seed: int = 0,
) -> SmokeTestResult:
    """Reset, take random actions, and close the environment.

    Used as a quick sanity check before committing to a long training run.
    All rewards are collected; the last ``info`` dict is retained for
    inspection.

    Args:
        env: The Gymnasium environment under test (wrapped or unwrapped).
        steps: Maximum random-action transitions to execute.
        seed: Seed passed to the initial ``env.reset()``.

    TODO (SV2): call ``env.reset(seed=seed)``, loop ``env.step(env.action_space.sample())``
    up to *steps* times, collect rewards, and return a :class:`~traffic_drl.contracts.SmokeTestResult`.

    Returns:
        SmokeTestResult: Step count, terminal flags, collected rewards, and
        the final ``info`` dict.
    """
    raise NotImplementedError
