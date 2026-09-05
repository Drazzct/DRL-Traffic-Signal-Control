"""Manifest-driven scenario selection and route generation facade."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


def load_manifest(path: str | Path) -> Any:
    """Load the single scenario manifest.

    Args:
        path: UTF-8 CSV with ``scenario_id``, ``split``, route path, seeds,
            duration, demand, turning-ratio, vehicle-mix, and checksum columns.

    File contract: parse with :mod:`csv.DictReader` into shared scenario
    contracts; resolve route paths relative to the repository root.

    Returns:
        ScenarioManifest: Typed manifest used by samplers and evaluators.
    """
    raise NotImplementedError


def select_scenario(manifest: Any, *, split: str, scenario_id: str | None = None, seed: int | None = None) -> Any:
    """Select exactly one scenario from an allowed split.

    Args:
        manifest: Typed manifest loaded from the CSV contract.
        split: Allowed split such as ``DEV``, ``TR``, ``VA``, or ``TE``.
        scenario_id: Optional exact ID; otherwise choose using ``seed``.
        seed: Selection seed for reproducible sampling.

    Returns:
        ScenarioRecord: Exactly one selected scenario.
    """
    raise NotImplementedError


def generate_selected_scenario(manifest: Any, *, split: str, output_root: str | Path, scenario_id: str | None = None, seed: int | None = None) -> Path:
    """Generate the selected route and return its path.

    Args:
        manifest: Typed manifest source.
        split: Split eligible for generation.
        output_root: Directory for generated ``.rou.xml`` files.
        scenario_id: Optional exact scenario ID.
        seed: Generation/selection seed recorded in the route metadata.

    Returns:
        Path: The generated route XML path.
    """
    raise NotImplementedError


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the scenario generation shell script.

    Args:
        argv: Optional command-line arguments without the program name.

    Returns:
        int: Process exit code, where zero indicates success.
    """
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
