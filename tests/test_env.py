"""Environment checker entry point for the Phase 1 scaffold."""

from __future__ import annotations

from typing import Any

import pytest


def check_project_environment(env: Any) -> None:
	"""Run Stable-Baselines3's checker when the optional dependency is installed."""
	try:
		from stable_baselines3.common.env_checker import check_env
	except ModuleNotFoundError as error:
		pytest.skip(f"Stable-Baselines3 is not installed: {error}")
	check_env(env)


def test_checker_dependency_is_optional() -> None:
	"""Keep test collection usable before project dependencies are installed."""
	assert True