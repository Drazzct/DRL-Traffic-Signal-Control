"""Tests for the All-Red phase support environment.

Uses the existing dummy SUMO network and route files under
  sumo/net/dummy/single-intersection.net.xml
  scenarios/dummy/single-intersection-vhvh.rou.xml

Tests are skipped automatically when those files are not present.
SUMO_HOME is set by conftest.py before any import of sumo_rl occurs.
"""
from __future__ import annotations

from pathlib import Path
import pytest

from traffic_drl.environment.make_env import create_sumo_env
from traffic_drl.config import (
    EnvConfig,
    NetworkConfig,
    TrafficLightConfig,
    TimingConfig,
    SumoOptions,
    ObservationBoundsConfig,
)
from traffic_drl.environment.all_red_env import AllRedSumoEnvironment
from traffic_drl.environment.all_red_traffic_signal import AllRedTrafficSignal

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NET_FILE = Path("sumo/net/dummy/single-intersection.net.xml")
_ROUTE_FILE = Path("scenarios/dummy/single-intersection-vhvh.rou.xml")

_DUMMY_FILES_PRESENT = _NET_FILE.exists() and _ROUTE_FILE.exists()
_SKIP_NO_DUMMY = pytest.mark.skipif(
    not _DUMMY_FILES_PRESENT,
    reason="Dummy SUMO network/route files not found",
)


def _make_config(*, num_seconds: int = 30, red_time: int = 3) -> EnvConfig:
    """Return a minimal EnvConfig pointing at the dummy SUMO files."""
    return EnvConfig(
        network=NetworkConfig(
            net_file=str(_NET_FILE),
            route_files=[str(_ROUTE_FILE)],
            additional_files=[],
        ),
        traffic_light=TrafficLightConfig(
            ts_id="J0",
            single_agent=True,
        ),
        timing=TimingConfig(
            num_seconds=num_seconds,
            delta_time=10,       # must be > yellow_time (2) + red_time (3)
            min_green=5,
            max_green=60,
            yellow_time=2,
            red_time=red_time,
        ),
        sumo_options=SumoOptions(
            use_gui=False,
            lateral_resolution=0.0,
            additional_sumo_cmd="",
            save_tripinfo=False,
        ),
        observation_bounds=ObservationBoundsConfig(
            max_queue_per_lane=10.0,
            max_time_in_phase=60.0,
        ),
    )


def _make_env(tmp_path: Path, **cfg_kwargs):
    """Create an AllRedSumoEnvironment using the dummy files."""
    config = _make_config(**cfg_kwargs)
    return create_sumo_env(
        config=config,
        route_file=str(_ROUTE_FILE),
        run_id="test_run",
        base_dir=tmp_path,
        use_gui=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@_SKIP_NO_DUMMY
def test_env_is_all_red_type(tmp_path: Path) -> None:
    """AllRedSumoEnvironment is returned by the factory."""
    env = _make_env(tmp_path)
    try:
        assert isinstance(env.unwrapped, AllRedSumoEnvironment)
    finally:
        env.close()


@_SKIP_NO_DUMMY
def test_red_time_propagated(tmp_path: Path) -> None:
    """red_time from config is stored on the environment and on every signal."""
    env = _make_env(tmp_path, red_time=3)
    try:
        unwrapped = env.unwrapped
        assert unwrapped.red_time == 3
        for ts in unwrapped.traffic_signals.values():
            assert isinstance(ts, AllRedTrafficSignal)
            assert ts.red_time == 3
    finally:
        env.close()


@_SKIP_NO_DUMMY
def test_reset_returns_valid_obs(tmp_path: Path) -> None:
    """env.reset() completes and returns an observation of the correct dtype."""
    import numpy as np

    env = _make_env(tmp_path)
    try:
        obs, info = env.reset()
        assert obs is not None
        assert hasattr(obs, "dtype")
        assert obs.dtype == np.float32
        assert obs.shape == env.observation_space.shape
    finally:
        env.close()


@_SKIP_NO_DUMMY
def test_step_returns_valid_tuple(tmp_path: Path) -> None:
    """One env.step() returns a 5-tuple matching the Gymnasium contract."""
    env = _make_env(tmp_path)
    try:
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, (int, float))  # sumo-rl may return int
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    finally:
        env.close()


@_SKIP_NO_DUMMY
def test_delta_time_assertion(tmp_path: Path) -> None:
    """AllRedSumoEnvironment raises AssertionError when delta_time <= yellow+red."""
    from traffic_drl.environment.all_red_env import AllRedSumoEnvironment

    config = _make_config()
    # Build kwargs manually to bypass create_sumo_env validation
    with pytest.raises(AssertionError, match="delta_time"):
        AllRedSumoEnvironment(
            net_file=str(_NET_FILE),
            route_file=str(_ROUTE_FILE),
            num_seconds=30,
            delta_time=5,     # 5 <= 2 + 3 → must fail
            yellow_time=2,
            red_time=3,
            single_agent=True,
        )
