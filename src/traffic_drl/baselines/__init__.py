"""Non-learning controller baselines.

Public API
----------
All three controllers satisfy the
:class:`~traffic_drl.contracts.Controller` protocol and are evaluated through
the same :func:`~traffic_drl.evaluation.evaluate_benchmark.evaluate_controller`
path.

- :class:`~traffic_drl.baselines.fixed_time.FixedTimeController` — pre-timed plan (SUMO ``fixed_ts=True``).
- :class:`~traffic_drl.baselines.max_pressure.MaxPressureController` — highest-pressure phase selection.
- :class:`~traffic_drl.baselines.actuated.ActuatedController` — detector-driven green extension *(stub)*.
"""
from traffic_drl.baselines.fixed_time import FixedTimeController, make_fixed_time_env
from traffic_drl.baselines.max_pressure import MaxPressureController
from traffic_drl.baselines.actuated import ActuatedController, make_actuated_env

__all__ = [
    "FixedTimeController",
    "make_fixed_time_env",
    "MaxPressureController",
    "ActuatedController",
    "make_actuated_env",
]
