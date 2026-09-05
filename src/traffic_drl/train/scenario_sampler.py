"""Leakage-aware scenario sampling for training and validation."""
from __future__ import annotations

from typing import Any

from traffic_drl.contracts import ScenarioRecord, ScenarioSource


class ScenarioSampler:
    """Sample exactly one manifest record per episode from an allowed split."""

    def __init__(self, manifest: ScenarioSource, allowed_split: str, seed: int | None = None) -> None:
        """Create a sampler restricted to one manifest split.

        Args:
            manifest: Typed source of scenario records.
            allowed_split: Split permitted for sampling, such as ``TR`` or ``VA``.
            seed: Sampler seed used for reproducible selection.

        TODO (SV2/SV3): reject TE sampling before the model is frozen.
        """
        self.manifest = manifest
        self.allowed_split = allowed_split
        self.seed = seed

    def sample(self, *, seed: int | None = None) -> ScenarioRecord:
        """Return one scenario record from the allowed split.

        Args:
            seed: Optional per-sample seed overriding the sampler seed.

        Returns:
            ScenarioRecord: One scenario from ``allowed_split``.
        """
        raise NotImplementedError

    def sample_route(self, *, seed: int | None = None) -> str:
        """Return one route path, never a comma-joined route list.

        Args:
            seed: Optional per-sample selection seed.

        Returns:
            str: Repository-relative route XML path for one selected scenario.
        """
        raise NotImplementedError

    def set_split(self, split: str) -> None:
        """Change the allowed split explicitly.

        Args:
            split: New manifest split; validation must prevent leakage.

        Returns:
            None: Future calls use the new allowed split.
        """
        raise NotImplementedError
