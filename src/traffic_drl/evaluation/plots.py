"""Plotting API for benchmark metrics and SUMO trajectories.

All plotting functions write a rendered image to disk (PNG or PDF).  They do
not display interactively.  Select the output format from the file suffix of
*output_path*.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from traffic_drl.contracts import EpisodeMetrics


def _to_dataframe(records: Iterable[EpisodeMetrics | dict[str, Any]]) -> pd.DataFrame:
    """Convert an iterable of EpisodeMetrics or dicts into a pandas DataFrame."""
    rows: list[dict[str, Any]] = []
    for r in records:
        if is_dataclass(r):
            rows.append(asdict(r))
        elif isinstance(r, dict):
            rows.append(dict(r))
        elif hasattr(r, "__dict__"):
            rows.append(dict(vars(r)))
        else:
            rows.append(r)
    return pd.DataFrame(rows)


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
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = _to_dataframe(records)
    if df.empty or metric not in df.columns:
        # Create an empty informative placeholder plot if no data
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, f"No data available for metric: {metric}", ha="center", va="center")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    if controllers is not None and "controller" in df.columns:
        df = df[df["controller"].isin(controllers)]

    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        import seaborn as sns

        sns.barplot(
            data=df,
            x="scenario_id" if "scenario_id" in df.columns else df.index,
            y=metric,
            hue="controller" if "controller" in df.columns else None,
            ax=ax,
            capsize=0.1,
            errorbar="sd",
            palette="Set2",
        )
    except Exception:
        # Manual matplotlib fallback if seaborn is not available
        grouped = df.groupby(["scenario_id", "controller"])[metric].agg(["mean", "std"]).reset_index()
        scenarios = grouped["scenario_id"].unique()
        ctrls = grouped["controller"].unique()
        import numpy as np

        x = np.arange(len(scenarios))
        width = 0.8 / max(1, len(ctrls))

        for idx, ctrl in enumerate(ctrls):
            sub = grouped[grouped["controller"] == ctrl]
            pos = x + idx * width - (len(ctrls) - 1) * width / 2
            ax.bar(pos, sub["mean"], width=width, yerr=sub["std"].fillna(0), label=ctrl, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)

    ax.set_title(f"Benchmark Comparison: {metric.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Scenario", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    if ax.get_legend():
        ax.legend(title="Controller", frameon=True)

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    log_p = Path(log_path)
    csv_file = None
    if log_p.is_file():
        csv_file = log_p
    elif log_p.is_dir():
        for candidate in ["progress.csv", "monitor.csv", "metrics.csv"]:
            p = log_p / candidate
            if p.exists():
                csv_file = p
                break

    if csv_file is not None and csv_file.exists():
        # Handle possible comment header in monitor.csv
        with csv_file.open("r", encoding="utf-8") as f:
            first_line = f.readline()
        skiprows = 1 if first_line.startswith("#") else 0
        df = pd.read_csv(csv_file, skiprows=skiprows)
    else:
        df = pd.DataFrame()

    fig, ax1 = plt.subplots(figsize=(10, 5))

    if df.empty:
        ax1.text(0.5, 0.5, f"Log file empty or not found: {log_path}", ha="center", va="center")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    # Identify timestep column
    step_col = None
    for candidate in ["time/total_timesteps", "timesteps", "total_timesteps", "step", "t"]:
        if candidate in df.columns:
            step_col = candidate
            break
    x = df[step_col] if step_col is not None else df.index

    # Identify reward / return column
    rew_col = None
    for candidate in ["rollout/ep_rew_mean", "ep_rew_mean", "reward", "r", "mean_reward"]:
        if candidate in df.columns:
            rew_col = candidate
            break

    # Identify loss column
    loss_col = None
    for candidate in ["train/loss", "loss", "train/critic_loss", "critic_loss"]:
        if candidate in df.columns:
            loss_col = candidate
            break

    if rew_col:
        ax1.plot(x, df[rew_col], color="#2b5c8f", label="Episodic Return", linewidth=2)
        ax1.set_xlabel("Timesteps" if step_col else "Steps", fontsize=12)
        ax1.set_ylabel("Mean Episodic Return", color="#2b5c8f", fontsize=12)
        ax1.tick_params(axis="y", labelcolor="#2b5c8f")
        ax1.grid(True, linestyle="--", alpha=0.6)

    if loss_col:
        ax2 = ax1.twinx() if rew_col else ax1
        ax2.plot(x, df[loss_col], color="#c0392b", linestyle="--", label="Training Loss", alpha=0.8)
        ax2.set_ylabel("Loss", color="#c0392b", fontsize=12)
        ax2.tick_params(axis="y", labelcolor="#c0392b")

    plt.title("Training Learning Curves", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = _to_dataframe(records)

    fig, ax = plt.subplots(figsize=(12, 6))

    if df.empty:
        ax.text(0.5, 0.5, "No queue telemetry records provided", ha="center", va="center")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    # Identify relevant columns
    time_col = next((c for c in ["time", "timestamp", "step", "sim_time"] if c in df.columns), None)
    approach_col = next((c for c in ["approach_id", "approach", "lane_id", "lane"] if c in df.columns), None)
    queue_col = next((c for c in ["queue_length", "queue", "queue_len", "vehicles"] if c in df.columns), None)

    if not (time_col and approach_col and queue_col):
        ax.text(0.5, 0.5, "Records missing required keys: timestamp, approach_id, queue_length", ha="center", va="center")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    pivot = df.pivot_table(index=approach_col, columns=time_col, values=queue_col, aggfunc="mean").fillna(0.0)

    try:
        import seaborn as sns

        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", cbar_kws={"label": "Queue Length (vehicles)"})
    except Exception:
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", origin="lower")
        fig.colorbar(im, ax=ax, label="Queue Length (vehicles)")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        step = max(1, len(pivot.columns) // 10)
        ax.set_xticks(range(0, len(pivot.columns), step))
        ax.set_xticklabels(pivot.columns[::step])

    ax.set_title("Traffic Queue Heatmap by Approach Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Simulation Time (s)", fontsize=12)
    ax.set_ylabel("Approach", fontsize=12)

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
