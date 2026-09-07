"""Learning-rate schedule signatures compatible with SB3."""
from __future__ import annotations

from collections.abc import Callable


def linear_schedule(initial_value: float, final_value: float = 0.0) -> Callable[[float], float]:
    """Return an SB3 progress-to-learning-rate callable.

    Args:
        initial_value: Learning rate when SB3 reports one hundred percent progress remaining.
        final_value: Learning rate when progress reaches zero.

    Returns:
        Callable[[float], float]: Function accepting SB3's remaining-progress
        fraction and returning the current learning rate.
    """
    raise NotImplementedError


def cosine_schedule(initial_value: float, final_value: float = 0.0) -> Callable[[float], float]:
    """Return an SB3-compatible cosine schedule callable.

    Args:
        initial_value: Starting learning rate.
        final_value: Ending learning rate.

    Returns:
        Callable[[float], float]: SB3 schedule function for a remaining-progress fraction.
    """
    raise NotImplementedError
