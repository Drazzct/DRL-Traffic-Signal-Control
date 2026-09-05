"""Shared contracts exchanged across the traffic DRL pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class ScenarioRecord:
    """One immutable row from the scenario manifest.

    TODO (SV3): verify split ownership, seeds, and checksum metadata before locking TE.
    """

    scenario_id: str
    split: str
    route_file: Path
    demand_seed: int
    sumo_seed: int
    num_seconds: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeMetrics:
    """Metrics collected from one controller episode.

    ``average_waiting_time``, ``average_queue_length``, and ``time_loss`` are
    the primary proposal metrics. ``throughput``, ``phase_switch_rate``, and
    ``min_green_violations`` are control constraints; remaining fields are
    supplementary traffic or environmental metrics.
    """

    controller: str
    scenario_id: str
    seed: int
    average_waiting_time: float
    average_queue_length: float
    time_loss: float
    throughput: float
    phase_switch_rate: float
    min_green_violations: int
    travel_time: float | None = None
    spillback: float | None = None
    recovery_time: float | None = None
    fuel: float | None = None
    co2: float | None = None
    nox: float | None = None
    inference_latency: float | None = None


@dataclass(frozen=True)
class MetricSummary:
    """Aggregated metric values for one controller/scenario group."""

    controller: str
    scenario_id: str
    sample_count: int
    means: dict[str, float]
    standard_deviations: dict[str, float]
    confidence_intervals: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class BenchmarkReport:
    """Interpreted benchmark output relative to a named baseline."""

    baseline_controller: str
    summaries: tuple[MetricSummary, ...]
    improvements: dict[str, dict[str, float]]
    constraint_violations: dict[str, int]


@dataclass(frozen=True)
class TripInfoMetrics:
    """Traffic metrics parsed from one SUMO tripinfo file."""

    vehicle_count: int
    average_waiting_time: float
    average_time_loss: float
    average_travel_time: float
    throughput: float


@dataclass(frozen=True)
class EmissionMetrics:
    """Environmental metrics parsed from one SUMO emissions file."""

    fuel: float
    co2: float
    nox: float


@dataclass(frozen=True)
class SmokeTestResult:
    """Outcome of a short environment reset/step smoke test."""

    steps: int
    terminated: bool
    truncated: bool
    rewards: tuple[float, ...]
    last_info: dict[str, Any]


@dataclass(frozen=True)
class ResumeInfo:
    """Metadata required to resume a training run reproducibly.

    TODO (SV2): record the exact config, commit, seed, split, and timestep at
    every checkpoint.
    """

    num_timesteps: int
    model_class: str
    seed: int | None = None
    scenario_split: str | None = None
    config_path: Path | None = None
    git_commit: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Controller(Protocol):
    """Common interface for learned and heuristic controllers."""

    def predict(self, observation: Any, *, deterministic: bool = True) -> Any:
        """Return one legal action for an observation.

        Args:
            observation: State produced by the Gymnasium/SUMO-RL environment.
            deterministic: Whether evaluation should disable policy exploration.
        """
        ...

    def reset(self) -> None:
        """Reset controller state before an episode.

        Returns:
            None: The controller is ready for a new episode.
        """
        ...


class EnvironmentFactory(Protocol):
    """Factory contract used by benchmark evaluation."""

    def __call__(self, scenario: ScenarioRecord, seed: int) -> Any:
        """Create one environment for one scenario and seed.

        Args:
            scenario: Route and metadata selected from one manifest split.
            seed: SUMO/environment seed used for this rollout.
        """
        ...


class ScenarioSource(Protocol):
    """Manifest contract required by the scenario sampler."""

    def records_for_split(self, split: str) -> Sequence[ScenarioRecord]:
        """Return immutable records for one split.

        Returns:
            Sequence[ScenarioRecord]: Records matching the requested split.
        """
        ...

    def get(self, scenario_id: str) -> ScenarioRecord:
        """Return one record by ID.

        Returns:
            ScenarioRecord: The matching manifest record.
        """
        ...
