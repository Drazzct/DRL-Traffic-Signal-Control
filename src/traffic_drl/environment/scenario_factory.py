"""Scenario manifest loading and split-based record access.

The :class:`ScenarioManifest` is the **sole** concrete implementation of the
:class:`~traffic_drl.contracts.ScenarioSource` protocol.  It is consumed by:

- :class:`~traffic_drl.train.scenario_sampler.ScenarioSampler` — picks one
  route per training episode.
- :func:`~traffic_drl.evaluation.evaluate_benchmark.evaluate_manifest` — iterates
  every validation/test scenario.

File format
-----------
A UTF-8 CSV with a header row containing at least:

    scenario_id, split, route_file, demand_seed, sumo_seed, num_seconds

Any extra columns are stored in :attr:`ScenarioRecord.metadata` as strings.
``route_file`` is resolved to an absolute :class:`pathlib.Path` relative to
``repo_root`` (default: current working directory).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from traffic_drl.contracts import ScenarioRecord, ScenarioSource


_REQUIRED_COLUMNS = {"scenario_id", "split", "route_file", "demand_seed", "sumo_seed", "num_seconds"}
_INT_COLUMNS = {"demand_seed", "sumo_seed", "num_seconds"}


class ScenarioManifest:
    """Read-only catalog of route files and reproducibility metadata.

    Satisfies the :class:`~traffic_drl.contracts.ScenarioSource` protocol.

    Create via the class method :meth:`from_csv`; the constructor is public but
    intended for programmatic construction in tests.

    Attributes:
        records: Immutable tuple of all scenario records in manifest order.
    """

    def __init__(self, records: Sequence[ScenarioRecord]) -> None:
        """Create a manifest from a sequence of typed scenario records.

        Args:
            records: Ordered scenario records; duplicated IDs raise
                ``ValueError`` at construction time.
        """
        self.records: tuple[ScenarioRecord, ...] = tuple(records)
        self._by_id: dict[str, ScenarioRecord] = {}
        for rec in self.records:
            if rec.scenario_id in self._by_id:
                raise ValueError(
                    f"Duplicate scenario_id '{rec.scenario_id}' in manifest."
                )
            self._by_id[rec.scenario_id] = rec

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        repo_root: str | Path | None = None,
    ) -> "ScenarioManifest":
        """Load a scenario manifest from a UTF-8 CSV file.

        Args:
            path: Path to the CSV file.
            repo_root: Root directory used to resolve relative ``route_file``
                paths.  Defaults to the current working directory.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If required columns are missing, duplicate IDs are
                found, or a numeric column cannot be parsed.

        TODO (SV3): add checksum validation for locked TE split records.

        Returns:
            ScenarioManifest: Populated manifest in CSV row order.
        """
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Scenario manifest not found: {csv_path}")

        root = Path(repo_root) if repo_root is not None else Path.cwd()
        records: list[ScenarioRecord] = []

        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise ValueError(f"Manifest CSV '{path}' has no header row.")
            missing = _REQUIRED_COLUMNS - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"Manifest CSV '{path}' is missing required columns: {sorted(missing)}"
                )

            for i, row in enumerate(reader, start=2):  # row 1 = header
                try:
                    route = Path(row["route_file"])
                    if not route.is_absolute():
                        route = root / route

                    # Extra columns go into metadata as strings
                    standard_keys = _REQUIRED_COLUMNS
                    metadata: dict[str, str | int | float | bool] = {
                        k: v for k, v in row.items() if k not in standard_keys
                    }

                    records.append(
                        ScenarioRecord(
                            scenario_id=row["scenario_id"].strip(),
                            split=row["split"].strip(),
                            route_file=route,
                            demand_seed=int(row["demand_seed"]),
                            sumo_seed=int(row["sumo_seed"]),
                            num_seconds=int(row["num_seconds"]),
                            metadata=metadata,
                        )
                    )
                except (ValueError, KeyError) as exc:
                    raise ValueError(
                        f"Failed to parse manifest row {i} in '{path}': {exc}"
                    ) from exc

        return cls(records)

    def records_for_split(self, split: str) -> tuple[ScenarioRecord, ...]:
        """Return all records belonging to *split*, in manifest order.

        Args:
            split: Split label such as ``TR``, ``VA``, or ``TE``.

        Returns:
            tuple[ScenarioRecord, ...]: Matching records; empty if none match.
        """
        return tuple(r for r in self.records if r.split == split)

    def get(self, scenario_id: str) -> ScenarioRecord:
        """Return one record by its stable manifest ID.

        Args:
            scenario_id: The ``scenario_id`` column value.

        Raises:
            KeyError: If no record with that ID exists.

        Returns:
            ScenarioRecord: The matching record.
        """
        try:
            return self._by_id[scenario_id]
        except KeyError:
            raise KeyError(
                f"Scenario '{scenario_id}' not found in manifest. "
                f"Available IDs: {sorted(self._by_id)}"
            ) from None


# Runtime assertion: verify ScenarioManifest satisfies the ScenarioSource protocol.
# Runs at import time — raises AssertionError immediately if the class drifts.
from traffic_drl.contracts import ScenarioSource as _ScenarioSource
assert issubclass(ScenarioManifest, _ScenarioSource), (
    "ScenarioManifest does not satisfy the ScenarioSource protocol."
)
del _ScenarioSource  # keep the module namespace clean
