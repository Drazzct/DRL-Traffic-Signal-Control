"""Plotting API for benchmark metrics and SUMO trajectories."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


def plot_metric_comparison(records: Any, metric: str, *, output_path: str | Path, controllers: Sequence[str] | None = None) -> None:
    """Plot one metric across controllers and scenarios.

    Args:
        records: Iterable of ``EpisodeMetrics`` or an equivalent tabular source.
        metric: Field name such as ``time_loss`` or ``throughput``.
        output_path: PNG/PDF image path; select the renderer from its suffix.
        controllers: Optional controller labels to include.

    File contract: create a parent directory if needed and write a real
    matplotlib image, not serialized plotting state.

    Returns:
        None: The rendered chart is written to ``output_path``.
    """
    raise NotImplementedError


def plot_learning_curve(log_path: str | Path, *, output_path: str | Path) -> None:
    """Plot episodic return and loss from an SB3 training log.

    Args:
        log_path: CSV or TensorBoard log directory produced by SB3 callbacks.
            CSV requires metric columns; TensorBoard requires event files.
        output_path: PNG/PDF destination for the rendered figure.

    File contract: inspect the suffix/content first, then parse CSV with
    :mod:`csv`/pandas or TensorBoard event files with the SB3/TensorBoard reader.

    Returns:
        None: The rendered learning-curve image is written to ``output_path``.
    """
    raise NotImplementedError


def plot_queue_heatmap(records: Any, *, output_path: str | Path) -> None:
    """Plot queue length by approach and simulation time.

    Args:
        records: Tabular records containing timestamp, approach ID, and queue length.
        output_path: PNG/PDF destination for the heatmap.

    File contract: output is a rendered image; input records must preserve the
    approach/time keys required to form a rectangular heatmap.

    Returns:
        None: The rendered heatmap is written to ``output_path``.
    """
    raise NotImplementedError
