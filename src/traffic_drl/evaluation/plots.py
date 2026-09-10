"""Plotting API for benchmark metrics and SUMO trajectories.

All plotting functions write a rendered image to disk (PNG or PDF).  They do
not display interactively.  Select the output format from the file suffix of
*output_path*.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from traffic_drl.contracts import EpisodeMetrics


def plot_metric_comparison(
    records: Iterable[EpisodeMetrics],
    metric: str,
    *,
    output_path: str | Path,
    controllers: list[str] | None = None,
) -> None:
    """Plot one metric across controllers and scenarios as a grouped bar chart.

    Args:
        records: Episode metrics for all controllers and scenarios.
        metric: Field name from :class:`~traffic_drl.contracts.EpisodeMetrics`,
            e.g. ``"time_loss"`` or ``"throughput"``.
        output_path: PNG or PDF destination.  The parent directory is created
            if it does not exist.
        controllers: Optional allow-list of controller labels to include.
            ``None`` includes all controllers present in *records*.

    File contract: create the parent directory if needed and write a
    matplotlib image — not serialised plotting state.

    TODO (SV3): implement using matplotlib; group bars by scenario, colour by
    controller; add mean ± CI error bars from
    :func:`~traffic_drl.evaluation.metrics.confidence_interval`.

    Returns:
        None.
    """
    raise NotImplementedError


def plot_learning_curve(
    log_path: str | Path,
    *,
    output_path: str | Path,
) -> None:
    """Plot episodic return and loss from an SB3 training log.

    Args:
        log_path: CSV or TensorBoard log directory produced by SB3 callbacks.
            CSV requires a ``timesteps`` column and at least one metric column.
            TensorBoard logs are read with the SB3/TensorBoard event reader.
        output_path: PNG or PDF destination for the rendered figure.

    File contract: inspect the suffix/content first; parse CSV with
    :mod:`csv` or TensorBoard event files with the appropriate reader.

    TODO (SV3): implement and decide whether to support CSV only or also
    TensorBoard event files.

    Returns:
        None.
    """
    raise NotImplementedError


def plot_queue_heatmap(
    records: Iterable[EpisodeMetrics],
    *,
    output_path: str | Path,
) -> None:
    """Plot queue length by approach and simulation time as a heat-map.

    Args:
        records: Episode metrics containing per-approach queue data.  The
            records must include timestep-level approach/queue information;
            episode-level aggregates are insufficient for this plot.
        output_path: PNG or PDF destination for the heat-map.

    File contract: output is a rendered image; input records must preserve the
    approach/time keys required to form a rectangular heat-map grid.

    TODO (SV3): decide the per-step data format and implement using matplotlib
    or seaborn.

    Returns:
        None.
    """
    raise NotImplementedError
