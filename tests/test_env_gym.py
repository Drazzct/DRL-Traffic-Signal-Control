"""Gymnasium and Stable-Baselines3 environment contract tests."""
from __future__ import annotations

from typing import Any


def test_environment_contract_placeholder() -> None:
    """Check the SB3 environment contract on DEV-00."""
    env = make_test_environment()
    from stable_baselines3.common.env_checker import check_env
    check_env(env, warn=True)
    env.close()

def make_test_environment() -> Any:
    """Return the lightweight DEV-00 environment used by tests."""
    from traffic_drl.environment.make_env import make_dev_environment
    return make_dev_environment(use_gui=False)
