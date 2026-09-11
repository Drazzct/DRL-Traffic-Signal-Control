"""Run ID management utilities for experiment tracking.

A *run ID* is a short, filesystem-safe string that uniquely identifies one
training or evaluation run.  It is used as the leaf directory under
``outputs/tripinfo/`` and ``outputs/results/``.

Typical usage
-------------
::

    from traffic_drl.run_id import generate_run_id, ensure_run_directories

    run_id = generate_run_id(prefix="dqn")
    tripinfo_dir, results_dir = ensure_run_directories(run_id)
"""
from __future__ import annotations
from typing import Literal

import hashlib
import uuid
from datetime import datetime
from pathlib import Path


def generate_run_id(
    prefix: str = "",
    use_timestamp: bool = True,
    use_uuid: bool = False,
    custom_string: str | None = None,
    run_type: Literal['train', 'val', 'test'] = 'train',
) -> str:
    """Generate a unique, filesystem-safe run ID.

    Args:
        prefix: Optional prefix (e.g. ``"dqn_"``).
        use_timestamp: Include a ``YYYYMMDD_HHMMSS`` timestamp component.
        use_uuid: Append a UUID4 for guaranteed global uniqueness.
        custom_string: Optional free-form suffix.
        run_type: The type of run the environment uses (train, val, test)

    Returns:
        str: A run ID safe for use in file and directory names.

    Examples::

        generate_run_id()                     # "20260910_143022"
        generate_run_id(prefix="exp_")        # "exp_20260910_143022"
        generate_run_id(use_uuid=True)        # "20260910_143022_550e8400..."
        generate_run_id(custom_string="test") # "20260910_143022_test"
    """
    components: list[str] = []

    if prefix:
        components.append(prefix)

    if use_timestamp:
        components.append(datetime.now().strftime("%Y%m%d_%H%M%S"))

    if use_uuid:
        components.append(str(uuid.uuid4()))

    if custom_string:
        components.append(custom_string)
    
    components.append(run_type)

    raw = "_".join(components)
    # Keep only alphanumeric, underscore, and hyphen characters.
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in raw)


def get_run_dir(
    run_id: str,
    base_dir: str | Path = "outputs/runs",
) -> Path:
    """Return the base directory for *run_id*.

    Args:
        run_id: The run identifier.
        base_dir: Base directory under which all runs sit.

    Returns:
        Path: ``<base_dir>/<run_id>``
    """
    return Path(base_dir) / run_id


def get_tripinfo_dir(
    run_id: str,
    base_dir: str | Path = "outputs/runs",
) -> Path:
    """Return the tripinfo output directory for *run_id*."""
    return get_run_dir(run_id, base_dir) / "tripinfo"


def get_results_dir(
    run_id: str,
    base_dir: str | Path = "outputs/runs",
) -> Path:
    """Return the results output directory for *run_id*."""
    return get_run_dir(run_id, base_dir) / "results"


def get_checkpoints_dir(
    run_id: str,
    base_dir: str | Path = "outputs/runs",
) -> Path:
    """Return the checkpoints directory for *run_id*."""
    return get_run_dir(run_id, base_dir) / "checkpoints"


def get_logs_dir(
    run_id: str,
    base_dir: str | Path = "outputs/runs",
) -> Path:
    """Return the logs directory for *run_id*."""
    return get_run_dir(run_id, base_dir) / "logs"


def get_tripinfo_path(
    run_id: str,
    base_dir: str | Path = "outputs/runs",
) -> Path:
    """Return the ``tripinfo.xml`` file path for *run_id*."""
    return get_tripinfo_dir(run_id, base_dir) / "tripinfo.xml"


def get_emissions_path(
    run_id: str,
    base_dir: str | Path = "outputs/runs",
) -> Path:
    """Return the ``emissions.xml`` file path for *run_id*."""
    return get_tripinfo_dir(run_id, base_dir) / "emissions.xml"


def get_metrics_path(
    run_id: str,
    base_dir: str | Path = "outputs/runs",
) -> Path:
    """Return the ``metrics.csv`` file path for *run_id*."""
    return get_results_dir(run_id, base_dir) / "metrics.csv"


def ensure_run_directories(
    run_id: str,
    base_dir: str | Path = "outputs/runs",
) -> tuple[Path, Path, Path, Path]:
    """Create all standard output directories for *run_id* if needed.

    Args:
        run_id: The run identifier.
        base_dir: Base directory for all runs.

    Returns:
        tuple[Path, Path, Path, Path]: ``(tripinfo, results, checkpoints, logs)`` directories.
    """
    tripinfo_dir = get_tripinfo_dir(run_id, base_dir)
    results_dir = get_results_dir(run_id, base_dir)
    checkpoints_dir = get_checkpoints_dir(run_id, base_dir)
    logs_dir = get_logs_dir(run_id, base_dir)
    
    tripinfo_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    return tripinfo_dir, results_dir, checkpoints_dir, logs_dir


def create_sumo_output_options(
    run_id: str,
    base_dir: str | Path = "outputs/runs",
) -> dict[str, str]:
    """Build SUMO command-line options for tripinfo and emissions output.

    The returned dict is intended to be merged into ``additional_sumo_cmd``
    or passed directly to the SUMO-RL constructor.

    Args:
        run_id: The run identifier.
        base_dir: Base directory for all runs.

    Returns:
        dict[str, str]: ``{"tripinfo-output": "<path>", "emission-output": "<path>"}``.
    """
    tripinfo_path = get_tripinfo_path(run_id, base_dir)
    emissions_path = get_emissions_path(run_id, base_dir)
    tripinfo_path.parent.mkdir(parents=True, exist_ok=True)
    emissions_path.parent.mkdir(parents=True, exist_ok=True)
    return {
        "tripinfo-output": str(tripinfo_path),
        "emission-output": str(emissions_path),
    }


def generate_config_based_run_id(config: dict, prefix: str = "") -> str:
    """Generate a deterministic run ID from a configuration dict.

    Hashes the sorted item list of *config* so runs with identical configs
    produce the same ID (useful for deduplication and caching).

    Args:
        config: Flat configuration dictionary.
        prefix: Optional prefix for the run ID.

    Returns:
        str: ``<prefix>_<timestamp>_<8-char hash>`` or ``<timestamp>_<hash>``.
    """
    config_str = str(sorted(config.items()))
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}_{config_hash}" if prefix else f"{timestamp}_{config_hash}"


def get_default_run_id(prefix: str = "") -> str:
    """Generate a default run ID combining timestamp and UUID.

    Args:
        prefix: Optional prefix for the run ID.

    Returns:
        str: A unique, filesystem-safe run ID.
    """
    return generate_run_id(prefix=prefix, use_timestamp=True, use_uuid=True)