"""Learning-rate schedule callables compatible with Stable-Baselines3.

SB3 accepts a ``Callable[[float], float]`` for its ``learning_rate``
parameter.  The float argument is the *remaining-progress fraction* —
``1.0`` at the start of training, ``0.0`` at the end.

Typical usage
-------------
::

    from traffic_drl.train.lr_schedules import linear_schedule

    model = DQN("MlpPolicy", env, learning_rate=linear_schedule(1e-3))
"""
from __future__ import annotations

import math
from collections.abc import Callable


def linear_schedule(
    initial_value: float,
    final_value: float = 0.0,
) -> Callable[[float], float]:
    """Return an SB3-compatible linear learning-rate schedule.

    The returned callable maps SB3's remaining-progress fraction linearly
    from *initial_value* (progress = 1.0) down to *final_value*
    (progress = 0.0).

    Args:
        initial_value: Learning rate at the start of training.
        final_value: Learning rate at the end of training.

    Returns:
        Callable[[float], float]: SB3 schedule function.

    Example::

        sched = linear_schedule(1e-3, 1e-5)
        sched(1.0)  # 0.001  (start)
        sched(0.5)  # 0.0005
        sched(0.0)  # 0.00001 (end)
    """
    def schedule(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)

    return schedule


def cosine_schedule(
    initial_value: float,
    final_value: float = 0.0,
) -> Callable[[float], float]:
    """Return an SB3-compatible cosine-annealing learning-rate schedule.

    Uses a half-cosine curve so the rate decays quickly at first and slowly
    near the end of training.

    Args:
        initial_value: Learning rate at the start of training (progress = 1.0).
        final_value: Learning rate at the end of training (progress = 0.0).

    Returns:
        Callable[[float], float]: SB3 schedule function.

    Example::

        sched = cosine_schedule(1e-3, 1e-5)
        sched(1.0)  # ≈ 0.001  (start)
        sched(0.5)  # midpoint value
        sched(0.0)  # ≈ 0.00001 (end)
    """
    def schedule(progress_remaining: float) -> float:
        # progress_remaining goes from 1 → 0; cosine goes from π → 0.
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress_remaining)))
        return final_value + cosine_factor * (initial_value - final_value)

    return schedule
