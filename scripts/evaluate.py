"""CLI entry point to evaluate controllers on SUMO .net and .rou files."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure src/ is in sys.path automatically so traffic_drl can always be imported
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_drl.evaluation import (
    aggregate_metrics,
    evaluate_controller,
    merge_tripinfo_metrics,
    parse_emissions,
    parse_tripinfo,
    plot_metric_comparison,
    save_evaluation_results,
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

    # Output paths
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/results"),
        help="Directory where evaluation metrics CSV and plots will be saved (default: outputs/results).",
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

    # Configure tripinfo and emissions output directory
    run_id = f"eval_{args.controller}_{args.seed}"
    tripinfo_dir = Path("outputs/tripinfo") / run_id
    tripinfo_dir.mkdir(parents=True, exist_ok=True)
    tripinfo_xml = tripinfo_dir / "tripinfo.xml"
    emissions_xml = tripinfo_dir / "emissions.xml"

    additional_sumo_cmd = (
        f"--tripinfo-output {tripinfo_xml.as_posix()} --emission-output {emissions_xml.as_posix()}"
    )

    print(f"[+] Initializing SUMO-RL (duration: {args.num_seconds}s, GUI: {args.gui})...")
    try:
        env = gym.make(
            "sumo-rl-v0",
            net_file=str(args.net),
            route_file=str(args.rou),
            use_gui=args.gui,
            num_seconds=args.num_seconds,
            additional_sumo_cmd=additional_sumo_cmd,
            single_agent=True,
        )
    except Exception as exc:
        print(f"\n[!] Error starting SUMO environment: {exc}", file=sys.stderr)
        print("Note: Ensure SUMO is installed and SUMO_HOME environment variable is set.", file=sys.stderr)
        print("If you already ran SUMO and have tripinfo.xml, use: --tripinfo <path_to_tripinfo.xml>", file=sys.stderr)
        sys.exit(1)

    # Controller selection
    if args.model_path and args.model_path.exists():
        print(f"[+] Loading SB3 model from {args.model_path}...")
        try:
            from stable_baselines3 import DQN

            model = DQN.load(str(args.model_path))
            controller_name = args.controller or "drl_model"
        except Exception as exc:
            print(f"Error loading model: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        controller_name = args.controller

        class BaselineController:
            def __init__(self, action_space):
                self.action_space = action_space

            def predict(self, obs, deterministic=True):
                return self.action_space.sample()

        model = BaselineController(env.action_space)

    scenario_id = args.rou.stem
    print(f"[+] Running evaluation ({args.episodes} episode(s))...")
    records = evaluate_controller(
        model=model,
        env=env,
        controller_name=controller_name,
        scenario_id=scenario_id,
        seed=args.seed,
        episodes=args.episodes,
    )
    env.close()

    # Parse tripinfo & emissions if generated
    if tripinfo_xml.exists():
        t_metrics = parse_tripinfo(tripinfo_xml, episode_seconds=args.num_seconds)
        e_metrics = (
            parse_emissions(emissions_xml)
            if emissions_xml.exists()
            else parse_emissions.__globals__["EmissionMetrics"](0.0, 0.0, 0.0)
        )
        merged = merge_tripinfo_metrics(
            t_metrics,
            e_metrics,
            controller=controller_name,
            scenario_id=scenario_id,
            seed=args.seed,
        )
        records = [merged]

    # Aggregate & display
    summaries = aggregate_metrics(records)
    for s in summaries:
        print("\n==========================================")
        print(f"Controller : {s.controller}")
        print(f"Scenario   : {s.scenario_id}")
        print(f"Episodes   : {s.sample_count}")
        print("------------------------------------------")
        for metric, mean_val in s.means.items():
            ci = s.confidence_intervals.get(metric, (mean_val, mean_val))
            print(f"  {metric:<24}: {mean_val:>10.2f} (95% CI: [{ci[0]:.2f}, {ci[1]:.2f}])")
        print("==========================================")

    # Save CSV & plot
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / "metrics.csv"
    save_evaluation_results(records, out_csv)
    print(f"\n[OK] Results saved to: {out_csv}")

    out_plot = args.output_dir / "waiting_time_comparison.png"
    plot_metric_comparison(records, "average_waiting_time", output_path=out_plot)
    print(f"[OK] Plot saved to: {out_plot}")



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
