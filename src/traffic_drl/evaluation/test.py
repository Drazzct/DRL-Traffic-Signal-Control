"""Smoke-test entry points for the Phase 1 environment."""
from __future__ import annotations

from typing import Any

from traffic_drl.contracts import SmokeTestResult


def check_environment(env: Any) -> None:
    """Run Stable-Baselines3's environment checker on an environment.

    Args:
        env: Gymnasium environment to validate.

    Returns:
        None: Raises or skips on contract failure; otherwise the check passes.
    """
    raise NotImplementedError


def run_smoke_test(env: Any, *, steps: int = 10, seed: int = 0) -> SmokeTestResult:
    """Run reset, random actions and close for a short DEV-00 check.

    Args:
        env: Gymnasium environment under test.
        steps: Maximum random-action transitions to execute.
        seed: Seed passed to the initial reset.

    Returns:
        SmokeTestResult: Step count, terminal flags, rewards, and final info.
    """
    raise NotImplementedError
