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
    try:
        from stable_baselines3.common.env_checker import check_env

        check_env(env)
        return
    except ModuleNotFoundError:
        pass

    # Direct contract checks fallback
    if not hasattr(env, "action_space"):
        raise AttributeError("Environment missing required 'action_space' attribute.")
    if not hasattr(env, "observation_space"):
        raise AttributeError("Environment missing required 'observation_space' attribute.")
    if not hasattr(env, "reset") or not callable(env.reset):
        raise AttributeError("Environment missing callable 'reset' method.")
    if not hasattr(env, "step") or not callable(env.step):
        raise AttributeError("Environment missing callable 'step' method.")

    reset_out = env.reset(seed=0)
    if not (isinstance(reset_out, tuple) and len(reset_out) == 2):
        raise ValueError(
            f"Gymnasium reset() must return (obs, info) tuple; received {type(reset_out)}."
        )

    obs, info = reset_out
    if not isinstance(info, dict):
        raise TypeError(f"reset() info must be a dict; received {type(info)}.")

    action = env.action_space.sample()
    step_out = env.step(action)
    if not (isinstance(step_out, tuple) and len(step_out) == 5):
        raise ValueError(
            f"Gymnasium step() must return 5-tuple (obs, reward, terminated, truncated, info); received length {len(step_out) if isinstance(step_out, tuple) else type(step_out)}."
        )

    next_obs, reward, term, trunc, step_info = step_out
    if not isinstance(term, (bool, int)):
        raise TypeError(f"terminated flag must be boolean; received {type(term)}.")
    if not isinstance(trunc, (bool, int)):
        raise TypeError(f"truncated flag must be boolean; received {type(trunc)}.")
    if not isinstance(step_info, dict):
        raise TypeError(f"step() info must be a dict; received {type(step_info)}.")


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
    reset_ret = env.reset(seed=seed)
    if isinstance(reset_ret, tuple) and len(reset_ret) == 2:
        obs, info = reset_ret
    else:
        obs, info = reset_ret, {}

    last_info: dict[str, Any] = dict(info) if isinstance(info, dict) else {}
    rewards: list[float] = []
    terminated = False
    truncated = False
    step_count = 0

    for _ in range(steps):
        action = env.action_space.sample() if hasattr(env, "action_space") else 0
        step_ret = env.step(action)

        if len(step_ret) == 5:
            obs, reward, terminated, truncated, s_info = step_ret
        else:
            obs, reward, done, s_info = step_ret
            terminated, truncated = bool(done), False

        rewards.append(float(reward))
        step_count += 1
        if isinstance(s_info, dict):
            last_info.update(s_info)

        if terminated or truncated:
            break

    if hasattr(env, "close") and callable(env.close):
        env.close()

    return SmokeTestResult(
        steps=step_count,
        terminated=bool(terminated),
        truncated=bool(truncated),
        rewards=tuple(rewards),
        last_info=last_info,
    )
