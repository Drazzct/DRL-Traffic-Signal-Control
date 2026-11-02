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
from multiprocessing import Value
from multiprocessing.sharedctypes import Synchronized

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
        difficulty_unlock_timesteps: tuple[int, ...] | list[int] | dict[int, int] | None = None,
        _shared_timestep: Synchronized | None = None,
    ) -> None:
        """Create a sampler restricted to one manifest split.

        Args:
            manifest: Typed source of scenario records satisfying
                :class:`~traffic_drl.contracts.ScenarioSource`.
            allowed_split: Split permitted for sampling, e.g. ``"TR"`` or
                ``"VA"``.
            seed: Sampler seed for reproducible record selection.
            difficulty_unlock_timesteps: Cumulative timestep thresholds for
                difficulty levels 1, 2, and 3. A sequence is interpreted as
                thresholds in increasing order; a mapping may use difficulty
                numbers as keys. Level 1 is always available at timestep 0.

        Raises:
            ValueError: If *allowed_split* yields no records in *manifest*.
        """
        self.manifest: ScenarioSource = manifest
        self.allowed_split: str = allowed_split
        self.seed: int | None = seed
        self._rng: random.Random = random.Random(seed)
        self.difficulty_unlock_timesteps = self._normalize_unlocks(
            difficulty_unlock_timesteps
        )
        self._current_timestep = 0
        if _shared_timestep is None and difficulty_unlock_timesteps is not None:
            _shared_timestep = Value("q", 0)
        self._shared_timestep = _shared_timestep

        # Validate that the split has at least one record.
        self._pool: tuple[ScenarioRecord, ...] = tuple(
            manifest.records_for_split(allowed_split)
        )
        if not self._pool:
            raise ValueError(
                f"No records found for split '{allowed_split}' in the manifest."
            )
        if difficulty_unlock_timesteps is not None:
            self._configure_curriculum()

    @staticmethod
    def _normalize_unlocks(
        unlocks: tuple[int, ...] | list[int] | dict[int, int] | None,
    ) -> dict[int, int] | None:
        if unlocks is None:
            return None
        if isinstance(unlocks, dict):
            normalized = {int(level): int(timestep) for level, timestep in unlocks.items()}
        else:
            normalized = {
                level: timestep
                for level, timestep in enumerate(unlocks, start=1)
            }
        if set(normalized) != {1, 2, 3}:
            raise ValueError("Curriculum unlocks must define difficulties 1, 2, and 3.")
        normalized.setdefault(1, 0)
        if normalized[1] != 0:
            raise ValueError("Difficulty 1 must be unlocked at timestep 0.")
        if normalized[2] < normalized[1] or normalized[3] < normalized[2]:
            raise ValueError("Curriculum unlock timesteps must be non-decreasing.")
        return normalized

    def _configure_curriculum(self) -> None:
        missing = [
            record.scenario_id
            for record in self._pool
            if record.metadata.get("difficulty") in (None, "")
        ]
        if missing:
            raise ValueError(
                "Curriculum sampling requires difficulty metadata: no difficulty "
                f"metadata for records {sorted(missing)} in split '{self.allowed_split}'."
            )

        for record in self._pool:
            raw = record.metadata["difficulty"]
            try:
                difficulty = int(str(raw).strip())
            except (TypeError, ValueError):
                raise ValueError(
                    f"Scenario '{record.scenario_id}' has invalid difficulty "
                    f"{raw!r}; expected 1, 2, or 3."
                )
            if difficulty not in (1, 2, 3):
                raise ValueError(
                    f"Scenario '{record.scenario_id}' has invalid difficulty "
                    f"{difficulty}; expected 1, 2, or 3."
                )
        self._refresh_curriculum_pool()

    def _refresh_curriculum_pool(self) -> None:
        if self.difficulty_unlock_timesteps is not None:
            unlocked = {
                level for level, threshold in self.difficulty_unlock_timesteps.items()
                if threshold <= self._current_timestep
            }
            self._sampling_pool = tuple(
                record for record in self._pool
                if int(str(record.metadata["difficulty"]).strip()) in unlocked
            )
            return

    @property
    def current_difficulty(self) -> str:
        self._sync_shared_timestep()
        if self.difficulty_unlock_timesteps is None:
            return "1"
        unlocked = [
            level for level, threshold in self.difficulty_unlock_timesteps.items()
            if threshold <= self._current_timestep
        ]
        return str(max(unlocked, default=1))

    def set_timestep(self, timestep: int) -> None:
        """Unlock curriculum levels whose timestep threshold has been reached."""
        if timestep < 0:
            raise ValueError("timestep must be non-negative.")
        self._current_timestep = timestep
        if self._shared_timestep is not None:
            with self._shared_timestep.get_lock():
                self._shared_timestep.value = timestep
        if self.difficulty_unlock_timesteps is not None:
            self._refresh_curriculum_pool()

    def _sync_shared_timestep(self) -> None:
        if self._shared_timestep is None:
            return
        timestep = int(self._shared_timestep.value)
        if timestep != self._current_timestep:
            self._current_timestep = timestep
            self._refresh_curriculum_pool()

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
        self._sync_shared_timestep()
        pool = getattr(self, "_sampling_pool", self._pool)
        if not pool:
            raise ValueError(
                f"No scenarios are unlocked in split '{self.allowed_split}' "
                f"at timestep {self._current_timestep}."
            )
        record = rng.choice(pool)
        return record

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
        if self.difficulty_unlock_timesteps is not None:
            self._configure_curriculum()
