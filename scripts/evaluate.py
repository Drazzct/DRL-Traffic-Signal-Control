"""CLI entry point to evaluate controllers on SUMO .net and .rou files."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

# Ensure src/ is in sys.path automatically so traffic_drl can always be imported
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_drl.baselines.fixed_time import FixedTimeController
from traffic_drl.contracts import EpisodeMetrics, StepMetrics
from traffic_drl.evaluation import (
    aggregate_metrics,
    evaluate_controller,
    merge_tripinfo_metrics,
    parse_emissions,
    parse_tripinfo,
    plot_metric_comparison,
    plot_step_evaluation_dashboard,
    plot_step_metric_timeseries,
    save_evaluation_results,
    save_step_metrics,
)
from traffic_drl.run_id import (
    ensure_run_directories,
    generate_run_id,
    get_emissions_path,
    get_tripinfo_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate traffic signal controllers on SUMO network (.net.xml) and route (.rou.xml) files."
    )
    # File inputs
    parser.add_argument(
        "--net",
        type=Path,
        help="Path to SUMO network XML file (.net.xml).",
    )
    parser.add_argument(
        "--rou",
        type=Path,
        help="Path to SUMO route XML file (.rou.xml).",
    )
    parser.add_argument(
        "--tripinfo",
        type=Path,
        help="Path to existing SUMO tripinfo.xml file (if evaluating directly from simulation output).",
    )
    parser.add_argument(
        "--emissions",
        type=Path,
        help="Path to existing SUMO emissions.xml file (optional).",
    )

    # Simulation & Controller configuration
    parser.add_argument(
        "--controller",
        type=str,
        default="fixed_time",
        help="Controller type/name: 'fixed_time', 'random', or 'model' (default: fixed_time).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to trained Stable-Baselines3 model checkpoint (.zip).",
    )
    parser.add_argument(
        "--vec-normalize",
        type=Path,
        help="Path to VecNormalize statistics file (.pkl) for normalizing observations.",
    )
    parser.add_argument(
        "--num-seconds",
        type=int,
        default=1000,
        help="Simulation duration in seconds (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for simulation and route behavior (default: 42).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of evaluation episodes to roll out (default: 1).",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch SUMO-GUI visualization window instead of headless SUMO.",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=50,
        help="Step delay in milliseconds for SUMO-GUI (default: 50ms so vehicles move visibly).",
    )

    # Step-level timeline and comparison options
    parser.add_argument(
        "--plot-steps",
        action="store_true",
        default=True,
        help="Generate per-time-step dynamics dashboard and accumulated diagrams (default: True).",
    )
    parser.add_argument(
        "--no-plot-steps",
        dest="plot_steps",
        action="store_false",
        help="Disable step-level metric recording and plotting.",
    )
    parser.add_argument(
        "--compare-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare evaluated controller with Fixed-Time baseline on matching seeds (default: True). Use --no-compare-baseline to disable.",
    )

    # Output paths
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/results"),
        help="Directory where evaluation metrics CSV and plots will be saved (default: outputs/results).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for computing confidence intervals (default: 0.95 for 95%% CI).",
    )

    return parser.parse_args()


def evaluate_from_xml(args: argparse.Namespace) -> None:
    """Parse existing tripinfo.xml and emissions.xml without re-running simulation."""
    if not args.tripinfo.exists():
        print(f"Error: File '{args.tripinfo}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"\n[+] Reading SUMO output: {args.tripinfo}")
    tripinfo_metrics = parse_tripinfo(args.tripinfo, episode_seconds=args.num_seconds)

    print("\n==========================================")
    print("       SUMO EVALUATION RESULTS            ")
    print("==========================================")
    print(f"Completed Vehicles     : {tripinfo_metrics.vehicle_count}")
    print(f"Average Waiting Time   : {tripinfo_metrics.average_waiting_time:.2f} s")
    print(f"Average Time Loss      : {tripinfo_metrics.average_time_loss:.2f} s")
    print(f"Average Travel Time    : {tripinfo_metrics.average_travel_time:.2f} s")
    print(f"Throughput             : {tripinfo_metrics.throughput:.2f} veh/h")

    if args.emissions and args.emissions.exists():
        emission_metrics = parse_emissions(args.emissions)
        print("------------------------------------------")
        print(f"Total Fuel Consumed    : {emission_metrics.fuel:.2f} ml/mg")
        print(f"Total CO2 Emitted      : {emission_metrics.co2:.2f} mg")
        print(f"Total NOx Emitted      : {emission_metrics.nox:.2f} mg")
    else:
        from traffic_drl.contracts import EmissionMetrics

        emission_metrics = EmissionMetrics(fuel=0.0, co2=0.0, nox=0.0)

    scenario_id = args.tripinfo.stem
    episode_metric = merge_tripinfo_metrics(
        tripinfo_metrics,
        emission_metrics,
        controller=args.controller,
        scenario_id=scenario_id,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / "metrics.csv"
    save_evaluation_results([episode_metric], out_csv)
    print("==========================================")
    print(f"[OK] Results successfully saved to: {out_csv}")



def evaluate_simulation(args: argparse.Namespace) -> None:
    """Run full simulation on .net and .rou files using SUMO-RL environment."""
    if not args.net or not args.net.exists():
        print(f"Error: Network file '{args.net}' does not exist.", file=sys.stderr)
        sys.exit(1)
    if not args.rou or not args.rou.exists():
        print(f"Error: Route file '{args.rou}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"\n[+] Network file : {args.net}")
    print(f"[+] Route file   : {args.rou}")

    # Check for gymnasium and sumo_rl
    try:
        import gymnasium as gym
        import sumo_rl
    except ImportError as exc:
        print(
            f"Error: {exc}. Please install 'sumo-rl' and 'gymnasium' to run simulation.",
            file=sys.stderr,
        )
        sys.exit(1)

    def create_env(
        ep_seed: int,
        tripinfo_file: Path,
        emissions_file: Path,
        use_gui: bool = False,
        fixed_ts: bool = False,
    ) -> gym.Env:
        gui_options = f" --delay {args.delay}" if (use_gui and args.delay > 0) else ""
        cmd = (
            f"--tripinfo-output {tripinfo_file.as_posix()} "
            f"--emission-output {emissions_file.as_posix()} "
            f"--no-step-log true --duration-log.disable true --no-warnings true{gui_options}"
        )
        return gym.make(
            "sumo-rl-v0",
            net_file=str(args.net),
            route_file=str(args.rou),
            use_gui=use_gui,
            num_seconds=args.num_seconds,
            sumo_seed=ep_seed,
            fixed_ts=fixed_ts,
            additional_sumo_cmd=cmd,
            single_agent=True,
        )

    # Controller selection and model loading
    sb3_model = None
    default_name = "model_agent"
    if args.model_path and args.model_path.exists():
        print(f"[+] Loading SB3 model from {args.model_path}...")
        try:
            from stable_baselines3 import PPO, DQN

            model_name_lower = args.model_path.name.lower()
            if "ppo" in model_name_lower:
                sb3_model = PPO.load(str(args.model_path))
                default_name = "ppo_agent"
            elif "dqn" in model_name_lower:
                sb3_model = DQN.load(str(args.model_path))
                default_name = "dqn_agent"
            else:
                try:
                    sb3_model = PPO.load(str(args.model_path))
                    default_name = "ppo_agent"
                except Exception:
                    sb3_model = DQN.load(str(args.model_path))
                    default_name = "dqn_agent"

            controller_name = (
                args.controller if args.controller != "fixed_time" else default_name
            )
            print(f"[+] Successfully loaded {type(sb3_model).__name__} model.")
        except Exception as exc:
            print(f"Error loading model: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        controller_name = args.controller

    class DRLControllerWrapper:
        """Adapter wrapping SB3 model with optional VecNormalize for evaluate_controller."""

        def __init__(self, raw_model, normalizer=None):
            self.raw_model = raw_model
            self.normalizer = normalizer

        def predict(self, obs, deterministic: bool = True):
            if self.normalizer is not None:
                obs = self.normalizer.normalize_obs(obs)
            action, _ = self.raw_model.predict(obs, deterministic=deterministic)
            if isinstance(action, np.ndarray):
                if action.ndim == 0:
                    return int(action)
                elif action.size == 1:
                    return int(action.item())
            return action

        def reset(self):
            if hasattr(self.raw_model, "reset"):
                self.raw_model.reset()

    class BaselineController:
        def __init__(self, action_space):
            self.action_space = action_space

        def predict(self, obs, deterministic=True):
            return self.action_space.sample()

    scenario_id = args.rou.stem
    records: list[EpisodeMetrics] = []
    step_records: list[StepMetrics] = []
    vec_norm = None

    print(f"\n[+] Running evaluation for {controller_name} ({args.episodes} episode(s), each in an independent SUMO session)...")

    for ep in range(args.episodes):
        ep_seed = args.seed + ep
        ep_run_id = generate_run_id(prefix=f"eval_{controller_name}_ep{ep}", run_type="test")
        _, _, *_ = ensure_run_directories(ep_run_id)
        ep_tripinfo = get_tripinfo_path(ep_run_id)
        ep_emissions = get_emissions_path(ep_run_id)

        print(f"  -> Episode {ep + 1}/{args.episodes} (Seed: {ep_seed})...")
        is_fixed = (sb3_model is None and controller_name == "fixed_time")
        try:
            env = create_env(ep_seed, ep_tripinfo, ep_emissions, use_gui=args.gui, fixed_ts=is_fixed)
        except Exception as exc:
            print(f"\n[!] Error starting SUMO environment for episode {ep + 1}: {exc}", file=sys.stderr)
            print("Note: Ensure SUMO is installed and SUMO_HOME environment variable is set.", file=sys.stderr)
            sys.exit(1)

        # Optional VecNormalize loaded once on the first environment
        if vec_norm is None and args.vec_normalize:
            if not args.vec_normalize.exists():
                print(f"Error: VecNormalize file '{args.vec_normalize}' not found.", file=sys.stderr)
                sys.exit(1)
            print(f"  [+] Loading VecNormalize statistics from {args.vec_normalize}...")
            try:
                from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
                dummy_venv = DummyVecEnv([lambda: env])
                vec_norm = VecNormalize.load(str(args.vec_normalize), dummy_venv)
                vec_norm.training = False
                vec_norm.norm_reward = False
                print("  [+] VecNormalize statistics loaded and frozen (training=False, norm_reward=False).")
            except Exception as exc:
                print(f"Error loading VecNormalize: {exc}", file=sys.stderr)
                sys.exit(1)

        if sb3_model is not None:
            model = DRLControllerWrapper(sb3_model, normalizer=vec_norm)
        elif controller_name == "fixed_time":
            model = FixedTimeController()
        else:
            model = BaselineController(env.action_space)

        ep_recs = evaluate_controller(
            controller=model,
            env=env,
            controller_name=controller_name,
            scenario_id=scenario_id,
            seed=ep_seed,
            episodes=1,
            step_metrics_collector=step_records if args.plot_steps else None,
            episode_index_offset=ep,
        )
        env.close()

        if ep_tripinfo.exists():
            t_m = parse_tripinfo(ep_tripinfo, episode_seconds=args.num_seconds)
            e_m = (
                parse_emissions(ep_emissions)
                if ep_emissions.exists()
                else parse_emissions.__globals__["EmissionMetrics"](0.0, 0.0, 0.0)
            )
            sim_rec = ep_recs[0] if ep_recs else None
            merged = merge_tripinfo_metrics(
                t_m,
                e_m,
                controller=controller_name,
                scenario_id=scenario_id,
                seed=ep_seed,
                average_queue_length=sim_rec.average_queue_length if sim_rec else 0.0,
                phase_switch_rate=sim_rec.phase_switch_rate if sim_rec else 0.0,
                min_green_violations=sim_rec.min_green_violations if sim_rec else 0,
                inference_latency=sim_rec.inference_latency if sim_rec else None,
            )
            records.append(merged)
        else:
            records.extend(ep_recs)

    # Optional Baseline Comparison for side-by-side benchmarking
    if args.compare_baseline and controller_name != "fixed_time":
        print(f"\n[+] Running Fixed-Time Baseline ({args.episodes} episode(s), matching seeds)...")
        for ep in range(args.episodes):
            base_ep_seed = args.seed + ep
            base_run_id = generate_run_id(prefix=f"eval_baseline_fixed_time_ep{ep}", run_type="test")
            _, _, *_ = ensure_run_directories(base_run_id)
            base_tripinfo = get_tripinfo_path(base_run_id)
            base_emissions = get_emissions_path(base_run_id)

            print(f"  -> Baseline Episode {ep + 1}/{args.episodes} (Seed: {base_ep_seed})...")
            try:
                base_env = create_env(base_ep_seed, base_tripinfo, base_emissions, use_gui=False, fixed_ts=True)
            except Exception as exc:
                print(f"[!] Warning: Could not run baseline simulation: {exc}", file=sys.stderr)
                break

            base_model = FixedTimeController()
            base_ep_recs = evaluate_controller(
                controller=base_model,
                env=base_env,
                controller_name="fixed_time",
                scenario_id=scenario_id,
                seed=base_ep_seed,
                episodes=1,
                step_metrics_collector=step_records if args.plot_steps else None,
                episode_index_offset=ep,
            )
            base_env.close()

            if base_tripinfo.exists():
                t_m = parse_tripinfo(base_tripinfo, episode_seconds=args.num_seconds)
                e_m = (
                    parse_emissions(base_emissions)
                    if base_emissions.exists()
                    else parse_emissions.__globals__["EmissionMetrics"](0.0, 0.0, 0.0)
                )
                b_sim = base_ep_recs[0] if base_ep_recs else None
                merged_base = merge_tripinfo_metrics(
                    t_m,
                    e_m,
                    controller="fixed_time",
                    scenario_id=scenario_id,
                    seed=base_ep_seed,
                    average_queue_length=b_sim.average_queue_length if b_sim else 0.0,
                    phase_switch_rate=b_sim.phase_switch_rate if b_sim else 0.0,
                    min_green_violations=b_sim.min_green_violations if b_sim else 0,
                    inference_latency=b_sim.inference_latency if b_sim else None,
                )
                records.append(merged_base)
            else:
                records.extend(base_ep_recs)
        print("[+] Baseline evaluation completed.")

    # Aggregate & display
    summaries = aggregate_metrics(records, confidence=args.confidence)
    ci_label = f"{int(args.confidence * 100)}% CI"
    for s in summaries:
        print("\n==========================================")
        print(f"Controller : {s.controller}")
        print(f"Scenario   : {s.scenario_id}")
        print(f"Episodes   : {s.sample_count}")
        print("------------------------------------------")
        for metric, mean_val in s.means.items():
            ci = s.confidence_intervals.get(metric, (mean_val, mean_val))
            print(f"  {metric:<24}: {mean_val:>10.2f} ({ci_label}: [{ci[0]:.2f}, {ci[1]:.2f}])")
        print("==========================================")

    # Save CSV & plot
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / "metrics.csv"
    save_evaluation_results(records, out_csv)
    print(f"\n[OK] Results saved to: {out_csv}")

    out_plot = args.output_dir / "waiting_time_comparison.png"
    plot_metric_comparison(records, "average_waiting_time", output_path=out_plot)
    print(f"[OK] Waiting time comparison plot saved to: {out_plot}")

    out_queue_plot = args.output_dir / "queue_comparison.png"
    plot_metric_comparison(records, "average_queue_length", output_path=out_queue_plot)
    print(f"[OK] Queue comparison plot saved to: {out_queue_plot}")

    out_loss_plot = args.output_dir / "time_loss_comparison.png"
    plot_metric_comparison(records, "time_loss", output_path=out_loss_plot)
    print(f"[OK] Time loss comparison plot saved to: {out_loss_plot}")

    # Step-level timeline metrics and diagrams
    if args.plot_steps and step_records:
        step_csv = args.output_dir / "step_metrics.csv"
        save_step_metrics(step_records, step_csv)
        print(f"[OK] Time-step metrics saved to: {step_csv}")

        dashboard_plot = args.output_dir / "step_evaluation_dashboard.png"
        plot_step_evaluation_dashboard(
            step_records,
            output_path=dashboard_plot,
            title=f"Traffic Dynamics & Evaluation Timeline - {scenario_id}",
        )
        print(f"[OK] Step dashboard plot saved to: {dashboard_plot}")

        cum_wait_plot = args.output_dir / "accumulated_waiting_time.png"
        plot_step_metric_timeseries(
            step_records,
            metric="accumulated_waiting_time",
            output_path=cum_wait_plot,
            title=f"Accumulated Waiting Time Progression - {scenario_id}",
        )
        print(f"[OK] Accumulated waiting time plot saved to: {cum_wait_plot}")

        queue_plot = args.output_dir / "queue_length_timeline.png"
        plot_step_metric_timeseries(
            step_records,
            metric="queue_length",
            output_path=queue_plot,
            title=f"Queue Length Over Time - {scenario_id}",
        )
        print(f"[OK] Queue length timeline plot saved to: {queue_plot}")



def main() -> None:
    args = parse_args()
    if args.tripinfo:
        evaluate_from_xml(args)
    elif args.net and args.rou:
        evaluate_simulation(args)
    else:
        print(
            "Usage Error: Provide either (--net and --rou) to run simulation, "
            "or (--tripinfo) to parse existing SUMO results.\n"
            "Run 'python scripts/evaluate.py --help' for options.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
