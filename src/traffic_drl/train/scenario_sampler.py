"""Leakage-aware scenario sampling for training and validation episodes.

The :class:`ScenarioSampler` restricts sampling to a single manifest split
(e.g. ``TR``) to prevent data leakage between training, validation, and test
splits.

Typical usage
-------------
::

    from traffic_drl.environment.scenario_factory import ScenarioManifest
    from traffic_drl.train.scenario_sampler import ScenarioSampler

    manifest = ScenarioManifest.from_csv("scenarios/scenario_manifest.csv")
    sampler = ScenarioSampler(manifest, allowed_split="TR", seed=42)
    record = sampler.sample()
    route_path = str(record.route_file)
"""
from __future__ import annotations

import random

from traffic_drl.contracts import ScenarioRecord, ScenarioSource


class ScenarioSampler:
    """Sample exactly one manifest record per episode from an allowed split.

    Maintains a seeded RNG so that sampling is reproducible across runs with
    the same ``seed``.

    Attributes:
        manifest: The scenario source used for split-based record lookup.
        allowed_split: The only split this sampler is permitted to draw from.
        seed: Sampler-level seed for reproducible episode ordering.

    TODO (SV2/SV3): reject TE sampling before the model is frozen.  Add a
    guard: if ``allowed_split == "TE"``, raise ``PermissionError`` unless an
    explicit ``unlock_te=True`` flag is passed.
    """

    def __init__(
        self,
        manifest: ScenarioSource,
        allowed_split: str,
        seed: int | None = None,
    ) -> None:
        """Create a sampler restricted to one manifest split.

        Args:
            manifest: Typed source of scenario records satisfying
                :class:`~traffic_drl.contracts.ScenarioSource`.
            allowed_split: Split permitted for sampling, e.g. ``"TR"`` or
                ``"VA"``.
            seed: Sampler seed for reproducible record selection.

        Raises:
            ValueError: If *allowed_split* yields no records in *manifest*.
        """
        self.manifest: ScenarioSource = manifest
        self.allowed_split: str = allowed_split
        self.seed: int | None = seed
        self._rng: random.Random = random.Random(seed)

        # Validate that the split has at least one record.
        self._pool: tuple[ScenarioRecord, ...] = tuple(
            manifest.records_for_split(allowed_split)
        )
        if not self._pool:
            raise ValueError(
                f"No records found for split '{allowed_split}' in the manifest."
            )

    def sample(self, *, seed: int | None = None) -> ScenarioRecord:
        """Return one scenario record drawn uniformly from the allowed split.

        Args:
            seed: Optional per-sample seed override.  When provided, a
                temporary RNG seeded with this value is used for this draw
                only; the sampler's internal RNG is unaffected.

        Returns:
            ScenarioRecord: One record from :attr:`allowed_split`.
        """
        rng = random.Random(seed) if seed is not None else self._rng
        return rng.choice(self._pool)

    def sample_route(self, *, seed: int | None = None) -> str:
        """Return one route file path from the allowed split.

        The path is the absolute string representation of
        :attr:`ScenarioRecord.route_file`.  It is always a single file —
        never a comma-joined multi-route string.

        Args:
            seed: Optional per-sample seed override.

        Returns:
            str: Absolute path to the selected route XML file.
        """
        return str(self.sample(seed=seed).route_file)

    def set_split(self, split: str) -> None:
        """Change the allowed split and refresh the record pool.

        Args:
            split: New manifest split label, e.g. ``"VA"``.

        Raises:
            ValueError: If the new split has no records in the manifest.

        Returns:
            None.
        """
        new_pool = tuple(self.manifest.records_for_split(split))
        if not new_pool:
            raise ValueError(
                f"No records found for split '{split}' in the manifest."
            )
        self.allowed_split = split
        self._pool = new_pool
