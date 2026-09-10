"""Comprehensive tests for the traffic_drl.evaluation module."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pytest

from traffic_drl.contracts import (
    EpisodeMetrics,
    ScenarioRecord,
    ScenarioSource,
    SmokeTestResult,
)
from traffic_drl.evaluation import (
    ADDITIONAL_METRICS,
    CONSTRAINT_METRICS,
    PRIMARY_METRICS,
    aggregate_metrics,
    check_environment,
    collect_episode_metrics,
    confidence_interval,
    evaluate_controller,
    evaluate_manifest,
    interpret_results,
    merge_tripinfo_metrics,
    parse_emissions,
    parse_tripinfo,
    plot_learning_curve,
    plot_metric_comparison,
    plot_queue_heatmap,
    run_smoke_test,
    save_evaluation_results,
    standard_metric_names,
)


# ==========================================
# 1. Metrics & Schema Tests
# ==========================================


def test_standard_metric_names() -> None:
    """Check standard metric names return the expected primary, constraint, and additional tuples."""
    names = standard_metric_names()
    assert names == PRIMARY_METRICS + CONSTRAINT_METRICS + ADDITIONAL_METRICS
    assert "average_waiting_time" in names
    assert "throughput" in names
    assert "fuel" in names


def test_collect_episode_metrics_success() -> None:
    """Collect valid episode metrics from environment info mapping."""
    info = {
        "average_waiting_time": 12.5,
        "average_queue_length": 4.2,
        "time_loss": 25.0,
        "throughput": 150.0,
        "phase_switch_rate": 0.05,
        "min_green_violations": 0,
        "fuel": 1200.5,
        "co2": 350.0,
    }
    rec = collect_episode_metrics(info, controller="dqn", scenario_id="VA-01", seed=42)

    assert isinstance(rec, EpisodeMetrics)
    assert rec.controller == "dqn"
    assert rec.scenario_id == "VA-01"
    assert rec.seed == 42
    assert rec.average_waiting_time == 12.5
    assert rec.average_queue_length == 4.2
    assert rec.time_loss == 25.0
    assert rec.throughput == 150.0
    assert rec.phase_switch_rate == 0.05
    assert rec.min_green_violations == 0
    assert rec.fuel == 1200.5
    assert rec.co2 == 350.0
    assert rec.travel_time is None


def test_collect_episode_metrics_missing_primary_raises() -> None:
    """Missing primary metrics must raise ValueError instead of defaulting silently."""
    info = {
        "average_waiting_time": 10.0,
        # missing average_queue_length and time_loss
        "throughput": 100.0,
    }
    with pytest.raises(ValueError, match="Missing required primary metric"):
        collect_episode_metrics(info, controller="dqn", scenario_id="VA-01", seed=42)


def test_confidence_interval() -> None:
    """Verify confidence interval edge cases and normal computations."""
    # Empty
    assert confidence_interval([]) == (0.0, 0.0)

    # Single item
    assert confidence_interval([5.0]) == (5.0, 5.0)

    # Multi item
    data = [10.0, 12.0, 11.0, 9.0, 13.0]
    low, high = confidence_interval(data, confidence=0.95)
    mean = sum(data) / len(data)
    assert low < mean < high


def test_aggregate_metrics() -> None:
    """Aggregate per-episode records by controller and scenario."""
    records = [
        EpisodeMetrics(
            controller="dqn",
            scenario_id="VA-01",
            seed=1,
            average_waiting_time=10.0,
            average_queue_length=3.0,
            time_loss=20.0,
            throughput=100.0,
            phase_switch_rate=0.02,
            min_green_violations=0,
        ),
        EpisodeMetrics(
            controller="dqn",
            scenario_id="VA-01",
            seed=2,
            average_waiting_time=12.0,
            average_queue_length=4.0,
            time_loss=22.0,
            throughput=110.0,
            phase_switch_rate=0.04,
            min_green_violations=0,
        ),
    ]
    summaries = aggregate_metrics(records)
    assert len(summaries) == 1
    s = summaries[0]
    assert s.controller == "dqn"
    assert s.scenario_id == "VA-01"
    assert s.sample_count == 2
    assert s.means["average_waiting_time"] == 11.0
    assert s.means["average_queue_length"] == 3.5
    assert s.means["throughput"] == 105.0
    assert s.standard_deviations["average_waiting_time"] > 0
    ci = s.confidence_intervals["average_waiting_time"]
    assert ci[0] < 11.0 < ci[1]


def test_interpret_results() -> None:
    """Interpret metrics relative to baseline controller."""
    records = [
        # Baseline (fixed_time)
        EpisodeMetrics(
            controller="fixed_time",
            scenario_id="VA-01",
            seed=1,
            average_waiting_time=20.0,
            average_queue_length=6.0,
            time_loss=40.0,
            throughput=100.0,
            phase_switch_rate=0.01,
            min_green_violations=0,
        ),
        # DRL controller
        EpisodeMetrics(
            controller="dqn",
            scenario_id="VA-01",
            seed=1,
            average_waiting_time=15.0,  # 25% lower waiting time (improvement)
            average_queue_length=4.5,
            time_loss=30.0,
            throughput=110.0,  # 10% higher throughput (improvement)
            phase_switch_rate=0.03,
            min_green_violations=1,  # 1 violation
        ),
    ]

    report = interpret_results(records, baseline_controller="fixed_time")
    assert report.baseline_controller == "fixed_time"
    assert len(report.summaries) == 2

    # Check improvement: waiting time decreased from 20 to 15 -> (20-15)/20 * 100 = 25%
    imp = report.improvements["dqn/VA-01"]
    assert pytest.approx(imp["average_waiting_time"], rel=1e-3) == 25.0
    # Throughput increased from 100 to 110 -> (110-100)/100 * 100 = 10%
    assert pytest.approx(imp["throughput"], rel=1e-3) == 10.0

    # Constraint violations tracked
    assert report.constraint_violations["dqn"] == 1
    assert report.constraint_violations["fixed_time"] == 0


# ==========================================
# 2. SUMO XML Parsing Tests
# ==========================================


def test_parse_tripinfo(tmp_path: Path) -> None:
    """Parse simulated tripinfo.xml and compute averages and throughput."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<tripinfos>
  <tripinfo id="veh_0" depart="0.0" arrival="60.0" duration="60.0" waitingTime="10.0" timeLoss="15.0" />
  <tripinfo id="veh_1" depart="5.0" arrival="65.0" duration="60.0" waitingTime="20.0" timeLoss="25.0" />
</tripinfos>
"""
    p = tmp_path / "tripinfo.xml"
    p.write_text(xml_content, encoding="utf-8")

    metrics = parse_tripinfo(p, episode_seconds=3600)
    assert metrics.vehicle_count == 2
    assert metrics.average_waiting_time == 15.0
    assert metrics.average_time_loss == 20.0
    assert metrics.average_travel_time == 60.0
    # 2 vehicles in 1 hour (3600s) = 2.0 veh/hr
    assert metrics.throughput == 2.0


def test_parse_tripinfo_empty(tmp_path: Path) -> None:
    """Empty tripinfo XML returns zeroed metrics."""
    xml_content = "<tripinfos></tripinfos>"
    p = tmp_path / "tripinfo_empty.xml"
    p.write_text(xml_content, encoding="utf-8")

    metrics = parse_tripinfo(p)
    assert metrics.vehicle_count == 0
    assert metrics.average_waiting_time == 0.0
    assert metrics.throughput == 0.0


def test_parse_emissions(tmp_path: Path) -> None:
    """Parse emissions XML and sum pollutant values."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<meandata>
  <timestep time="1.0">
    <vehicle id="veh_0" fuel="1.5" CO2="20.0" NOx="0.5" />
    <vehicle id="veh_1" fuel="2.5" CO2="30.0" NOx="1.5" />
  </timestep>
</meandata>
"""
    p = tmp_path / "emissions.xml"
    p.write_text(xml_content, encoding="utf-8")

    em = parse_emissions(p)
    assert em.fuel == 4.0
    assert em.co2 == 50.0
    assert em.nox == 2.0


def test_merge_tripinfo_metrics(tmp_path: Path) -> None:
    """Merge traffic and environmental values into EpisodeMetrics."""
    tripinfo_xml = """<tripinfos><tripinfo id="v0" waitingTime="5.0" timeLoss="8.0" duration="40.0" arrival="100.0" /></tripinfos>"""
    emissions_xml = """<meandata><vehicle fuel="10.0" CO2="50.0" NOx="2.0" /></meandata>"""

    t_file = tmp_path / "tripinfo.xml"
    e_file = tmp_path / "emissions.xml"
    t_file.write_text(tripinfo_xml, encoding="utf-8")
    e_file.write_text(emissions_xml, encoding="utf-8")

    t_m = parse_tripinfo(t_file)
    e_m = parse_emissions(e_file)

    ep_m = merge_tripinfo_metrics(t_m, e_m, controller="baseline", scenario_id="DEV-00", seed=5)
    assert ep_m.controller == "baseline"
    assert ep_m.scenario_id == "DEV-00"
    assert ep_m.average_waiting_time == 5.0
    assert ep_m.fuel == 10.0
    assert ep_m.co2 == 50.0
    assert ep_m.nox == 2.0


# ==========================================
# 3. Benchmark Runner & Mock Tests
# ==========================================


class MockSpace:
    def sample(self) -> int:
        return 0


class MockGymEnv:
    def __init__(self, max_steps: int = 5) -> None:
        self.max_steps = max_steps
        self.step_count = 0
        self.action_space = MockSpace()
        self.observation_space = MockSpace()
        self.is_closed = False

    def reset(self, *, seed: int | None = None) -> tuple[int, dict[str, Any]]:
        self.step_count = 0
        return 0, {
            "average_waiting_time": 5.0,
            "average_queue_length": 2.0,
            "time_loss": 10.0,
            "throughput": 80.0,
            "min_green_violations": 0,
        }

    def step(self, action: Any) -> tuple[int, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        done = self.step_count >= self.max_steps
        info = {
            "average_waiting_time": 5.0 + self.step_count,
            "average_queue_length": 2.0,
            "time_loss": 10.0,
            "throughput": 80.0,
            "min_green_violations": 0,
        }
        return 0, 1.0, done, False, info

    def close(self) -> None:
        self.is_closed = True


class MockController:
    def __init__(self) -> None:
        self.reset_called = False

    def reset(self) -> None:
        self.reset_called = True

    def predict(self, obs: Any, *, deterministic: bool = True) -> int:
        return 0


class MockManifest(ScenarioSource):
    def __init__(self, records: list[ScenarioRecord]) -> None:
        self._records = records

    def records_for_split(self, split: str) -> tuple[ScenarioRecord, ...]:
        return tuple(r for r in self._records if r.split == split)

    def get(self, scenario_id: str) -> ScenarioRecord:
        for r in self._records:
            if r.scenario_id == scenario_id:
                return r
        raise KeyError(scenario_id)


def test_evaluate_controller() -> None:
    """Roll out controller on mock environment."""
    env = MockGymEnv(max_steps=3)
    controller = MockController()
    records = evaluate_controller(
        controller,
        env,
        controller_name="test_ctrl",
        scenario_id="DEV-00",
        seed=10,
        episodes=2,
    )

    assert len(records) == 2
    assert controller.reset_called is True
    assert records[0].controller == "test_ctrl"
    assert records[0].scenario_id == "DEV-00"
    assert records[0].seed == 10
    assert records[1].seed == 11


def test_evaluate_manifest_split_rules() -> None:
    """TE split evaluation must be blocked unless explicitly unlocked."""
    records = [
        ScenarioRecord(
            scenario_id="VA-01",
            split="VA",
            route_file=Path("dummy_va.rou.xml"),
            demand_seed=1,
            sumo_seed=1,
            num_seconds=100,
        ),
        ScenarioRecord(
            scenario_id="TE-01",
            split="TE",
            route_file=Path("dummy_te.rou.xml"),
            demand_seed=2,
            sumo_seed=2,
            num_seconds=100,
        ),
    ]
    manifest = MockManifest(records)
    ctrl = MockController()
    env_factory = lambda rec, seed: MockGymEnv()

    # Evaluating VA split should succeed
    va_results = evaluate_manifest(
        ctrl, env_factory, manifest, split="VA", seeds=[101], controller_name="ctrl"
    )
    assert len(va_results) == 1

    # Evaluating locked TE split should raise PermissionError
    with pytest.raises(PermissionError, match="Held-out test split \\(TE\\) is locked"):
        evaluate_manifest(
            ctrl, env_factory, manifest, split="TE", seeds=[101], controller_name="ctrl"
        )

    # Evaluating TE split with explicit allow_te=True should succeed
    te_results = evaluate_manifest(
        ctrl,
        env_factory,
        manifest,
        split="TE",
        seeds=[101],
        controller_name="ctrl",
        allow_te=True,
    )
    assert len(te_results) == 1


def test_save_evaluation_results(tmp_path: Path) -> None:
    """Save records to CSV and JSON formats."""
    records = [
        EpisodeMetrics(
            controller="dqn",
            scenario_id="VA-01",
            seed=101,
            average_waiting_time=12.0,
            average_queue_length=3.5,
            time_loss=25.0,
            throughput=120.0,
            phase_switch_rate=0.05,
            min_green_violations=0,
        )
    ]

    csv_out = tmp_path / "results.csv"
    save_evaluation_results(records, csv_out)
    assert csv_out.exists()
    content = csv_out.read_text(encoding="utf-8")
    assert "average_waiting_time" in content
    assert "12.0" in content

    json_out = tmp_path / "results.json"
    save_evaluation_results(records, json_out)
    assert json_out.exists()
    j_content = json_out.read_text(encoding="utf-8")
    assert '"controller": "dqn"' in j_content

    with pytest.raises(ValueError, match="Unsupported output file extension"):
        save_evaluation_results(records, tmp_path / "results.txt")


# ==========================================
# 4. Plotting Tests
# ==========================================


def test_plot_metric_comparison(tmp_path: Path) -> None:
    """Render metric comparison bar chart to PNG."""
    records = [
        EpisodeMetrics(
            controller="fixed_time",
            scenario_id="VA-01",
            seed=1,
            average_waiting_time=20.0,
            average_queue_length=5.0,
            time_loss=30.0,
            throughput=100.0,
            phase_switch_rate=0.01,
            min_green_violations=0,
        ),
        EpisodeMetrics(
            controller="dqn",
            scenario_id="VA-01",
            seed=1,
            average_waiting_time=15.0,
            average_queue_length=3.0,
            time_loss=22.0,
            throughput=115.0,
            phase_switch_rate=0.04,
            min_green_violations=0,
        ),
    ]
    out_png = tmp_path / "plots" / "comparison.png"
    plot_metric_comparison(records, "average_waiting_time", output_path=out_png)
    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_plot_learning_curve(tmp_path: Path) -> None:
    """Render learning curves from mock SB3 CSV log."""
    csv_content = """time/total_timesteps,rollout/ep_rew_mean,train/loss
1000,-150.0,0.85
2000,-120.0,0.60
3000,-80.0,0.40
4000,-60.0,0.25
"""
    log_file = tmp_path / "progress.csv"
    log_file.write_text(csv_content, encoding="utf-8")

    out_png = tmp_path / "plots" / "learning_curve.png"
    plot_learning_curve(log_file, output_path=out_png)
    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_plot_queue_heatmap(tmp_path: Path) -> None:
    """Render 2D queue heatmap from tabular telemetry."""
    records = [
        {"time": 0, "approach_id": "North", "queue_length": 2},
        {"time": 5, "approach_id": "North", "queue_length": 4},
        {"time": 0, "approach_id": "South", "queue_length": 1},
        {"time": 5, "approach_id": "South", "queue_length": 3},
    ]
    out_png = tmp_path / "plots" / "heatmap.png"
    plot_queue_heatmap(records, output_path=out_png)
    assert out_png.exists()
    assert out_png.stat().st_size > 0


# ==========================================
# 5. Environment Checker & Smoke Test
# ==========================================


def test_check_environment_passes() -> None:
    """Check environment contract passes on compliant mock environment."""
    env = MockGymEnv()
    check_environment(env)


def test_run_smoke_test() -> None:
    """Execute smoke test on mock environment and verify SmokeTestResult."""
    env = MockGymEnv(max_steps=4)
    result = run_smoke_test(env, steps=10, seed=42)

    assert isinstance(result, SmokeTestResult)
    assert result.steps == 4
    assert result.terminated is True
    assert len(result.rewards) == 4
    assert env.is_closed is True
