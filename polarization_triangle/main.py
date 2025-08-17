"""
Project unified entry point: calls the following three modules (and all aggregated mode):
1) experiments/zealot_morality_analysis.py
2) experiments/zealot_parameter_sweep.py
3) scripts/run_sobol_analysis.py

Usage examples (Windows PowerShell):
  I. Minimal test (Smoke Test, for quick validation)
    1) Zealot & Morality analysis (quick accumulation + plotting)
       python -m polarization_triangle.main morality --mode full --runs 1 --max-zealots 4 --max-morality 4 --processes 1 --error-band-type std --no-smoothing
       # For plotting only with existing data:
       python -m polarization_triangle.main morality --mode plot --error-band-type std --no-smoothing

    2) Parameter sweep (minimized overhead)
       python -m polarization_triangle.main sweep --runs 1 --steps 50 --processes 1 --data-dir results/zealot_parameter_sweep_smoke

    3) Sobol sensitivity analysis (quick configuration)
       python -m polarization_triangle.main sobol --config quick

  II. Normal test (refer to default parameters of each module)
    1) Zealot & Morality analysis (default parameters)
       python -m polarization_triangle.main morality --mode full --runs 5 --max-zealots 50 --max-morality 30 --processes 1 --error-band-type std --smoothing

    2) Parameter sweep (script default parameters)
       python -m polarization_triangle.main sweep --runs 20 --steps 300 --data-dir results/zealot_parameter_sweep

    3) Sobol sensitivity analysis (standard preset)
       python -m polarization_triangle.main sobol --config standard

  III. All mode (run all three types of tests sequentially at once)
    - Minimal test:
      python -m polarization_triangle.main all --profile smoke --processes 1
    - Normal test:
      python -m polarization_triangle.main all --profile normal
"""

import argparse
import sys
import time


# -------- 1) Zealot & Morality Analysis --------
from polarization_triangle.experiments.zealot_morality_analysis import (
    run_zealot_morality_analysis,
    run_and_accumulate_data,
    plot_from_accumulated_data,
    run_no_zealot_morality_data,
)


# ==========================
# Common tools and configuration
# ==========================

def time_exec(label: str, func, *args, **kwargs):
    start_ts = time.time()
    try:
        result = func(*args, **kwargs)
        duration = time.time() - start_ts
        print(f"\n⏱️ {label} command execution time: {duration:.2f}s")
        return True, None, duration, result
    except Exception as e:
        duration = time.time() - start_ts
        print(f"\n⏱️ {label} command execution time: {duration:.2f}s (failed)")
        return False, str(e), duration, None


def build_sobol_custom_config(args):
    # If command line provides override items, construct custom configuration and return; otherwise return None
    if any([getattr(args, k, None) for k in ("output_dir", "n_samples", "n_runs", "n_processes")]):
        from polarization_triangle.scripts.run_sobol_analysis import create_analysis_configs
        from polarization_triangle.analysis.sobol_analysis import SobolConfig
        base = create_analysis_configs()[args.config]
        return SobolConfig(
            n_samples=args.n_samples or base.n_samples,
            n_runs=args.n_runs or base.n_runs,
            n_processes=args.n_processes or base.n_processes,
            output_dir=args.output_dir or base.output_dir,
            num_steps=base.num_steps,
            base_config=base.base_config,
        )
    return None


def print_all_summary(results, total_elapsed: float):
    # results: [(label, success, error, duration)]
    print("\n" + "=" * 60)
    print("ALL mode execution summary")
    print("=" * 60)
    overall = True
    for label, success, error, duration in results:
        print(f"{label}: {'success' if success else 'failed'}" + (f", error: {error}" if not success and error else ""))
        print(f"  Duration: {duration:.2f}s" if isinstance(duration, (int, float)) else "  Duration: N/A")
        overall = overall and success
    print("-" * 60)
    print(f"Overall result: {'all successful' if overall else 'some failed'}")
    print(f"Total duration: {total_elapsed:.2f}s")


def add_morality_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "morality",
        help="Run or plot Zealot & Morality analysis",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "accumulate", "plot", "no-zealot"],
        default="full",
        help="Run mode: full=accumulate data then plot; accumulate=accumulate only; plot=plot only; no-zealot=collect only non-zealot data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/zealot_morality_analysis",
        help="Output directory",
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per parameter point")
    parser.add_argument("--max-zealots", type=int, default=50, help="Maximum zealot number (used only in full/accumulate modes)")
    parser.add_argument("--max-morality", type=int, default=30, help="Maximum morality ratio % (used in full/accumulate/no-zealot)")
    parser.add_argument("--processes", type=int, default=1, help="Number of parallel processes")

    # Plotting/error band and smoothing configuration (used in plot or full plotting phases)
    parser.add_argument(
        "--error-band-type",
        choices=["std", "percentile", "confidence"],
        default="std",
        help="Error bands type for zealot_numbers plots (plot or full plotting phases)",
    )
    parser.add_argument("--smoothing", dest="smoothing", action="store_true", help="Enable smoothing for morality_ratios plotting")
    parser.add_argument("--no-smoothing", dest="smoothing", action="store_false", help="Disable smoothing for morality_ratios plotting")
    parser.set_defaults(smoothing=True)
    parser.add_argument("--step", type=int, default=2, help="Resampling step size (for smoothing)")
    parser.add_argument(
        "--smooth-method",
        choices=["savgol", "moving_avg", "none"],
        default="savgol",
        help="Smoothing method",
    )

    parser.set_defaults(func=dispatch_morality)


def run_morality_command(args: argparse.Namespace) -> None:
    if args.mode == "full":
        run_and_accumulate_data(
            output_dir=args.output_dir,
            num_runs=args.runs,
            max_zealots=args.max_zealots,
            max_morality=args.max_morality,
            num_processes=args.processes,
        )
        plot_from_accumulated_data(
            output_dir=args.output_dir,
            enable_smoothing=args.smoothing,
            target_step=args.step,
            smooth_method=args.smooth_method,
            error_band_type=args.error_band_type,
        )
    elif args.mode == "accumulate":
        run_and_accumulate_data(
            output_dir=args.output_dir,
            num_runs=args.runs,
            max_zealots=args.max_zealots,
            max_morality=args.max_morality,
            num_processes=args.processes,
        )
    elif args.mode == "plot":
        plot_from_accumulated_data(
            output_dir=args.output_dir,
            enable_smoothing=args.smoothing,
            target_step=args.step,
            smooth_method=args.smooth_method,
            error_band_type=args.error_band_type,
        )
    elif args.mode == "no-zealot":
        run_no_zealot_morality_data(
            output_dir=args.output_dir,
            num_runs=args.runs,
            max_morality=args.max_morality,
            num_processes=args.processes,
        )
    else:
        run_zealot_morality_analysis(
            output_dir=args.output_dir,
            num_runs=args.runs,
            max_zealots=args.max_zealots,
            max_morality=args.max_morality,
            num_processes=args.processes,
            error_band_type=args.error_band_type,
        )


def dispatch_morality(args: argparse.Namespace) -> None:
    time_exec("Morality", run_morality_command, args)


# -------- 2) Parameter Sweep --------
from polarization_triangle.experiments.zealot_parameter_sweep import (
    run_parameter_sweep as run_zealot_parameter_sweep,
    run_plot_only_mode as run_sweep_plot_only,
)


def add_sweep_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "sweep",
        help="Run Zealot parameter sweep or plot only from existing data",
    )
    parser.add_argument("--plot-only", action="store_true", help="Plot-only mode: generate plots from existing data")
    parser.add_argument("--data-dir", type=str, default="results/zealot_parameter_sweep", help="Data directory/output directory")
    parser.add_argument("--runs", type=int, default=20, help="Number of runs per configuration")
    parser.add_argument("--steps", type=int, default=300, help="Number of simulation steps per run")
    parser.add_argument("--initial-scale", type=float, default=0.1, help="Initial opinion scaling factor")
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--processes", type=int, default=None, help="Number of parallel processes (default: use all CPU cores)")
    parser.add_argument("--max-size-mb", type=int, default=500, help="Maximum size limit for individual data file (MB)")
    parser.add_argument("--no-optimize", action="store_true", help="Disable data optimization")
    parser.add_argument("--essential-only", action="store_true", help="Keep only essential statistics to save memory")

    parser.set_defaults(func=dispatch_sweep)


def run_sweep_command(args: argparse.Namespace) -> None:
    if args.plot_only:
        run_sweep_plot_only(args.data_dir)
        return
    run_zealot_parameter_sweep(
        runs_per_config=args.runs,
        steps=args.steps,
        initial_scale=args.initial_scale,
        base_seed=args.base_seed,
        output_base_dir=args.data_dir,
        num_processes=args.processes,
        max_size_mb=args.max_size_mb,
        optimize_data=not args.no_optimize,
        preserve_essential_only=args.essential_only,
    )


def dispatch_sweep(args: argparse.Namespace) -> None:
    time_exec("Sweep", run_sweep_command, args)


# -------- 3) Sobol Sensitivity Analysis --------
from polarization_triangle.scripts.run_sobol_analysis import (
    run_sensitivity_analysis,
    generate_reports,
)


def add_sobol_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "sobol",
        help="Run Sobol sensitivity analysis or generate reports from existing results",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="standard",
        choices=["quick", "standard", "high_precision", "full", "test1"],
        help="Preset configuration",
    )
    parser.add_argument("--load", action="store_true", help="Try to load existing results")
    parser.add_argument("--no-plots", action="store_true", help="Do not generate visualization plots")
    parser.add_argument("--output-dir", type=str, help="Custom output directory (override preset)")
    parser.add_argument("--n-samples", type=int, help="Custom sample number (override preset)")
    parser.add_argument("--n-runs", type=int, help="Custom run number (override preset)")
    parser.add_argument("--n-processes", type=int, help="Custom process number (override preset)")

    parser.set_defaults(func=dispatch_sobol)


def run_sobol_command(args: argparse.Namespace) -> None:
    custom = build_sobol_custom_config(args)
    analyzer, sensitivity_indices = run_sensitivity_analysis(
        config_name=args.config,
        custom_config=custom,
        load_existing=args.load,
    )
    generate_reports(
        analyzer,
        sensitivity_indices,
        create_plots=not args.no_plots,
        config_name=args.config,
        start_time=None,
    )


def dispatch_sobol(args: argparse.Namespace) -> None:
    time_exec("Sobol", run_sobol_command, args)


def add_all_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "all",
        help="Run morality, sweep, sobol three types of tests at once",
    )
    parser.add_argument(
        "--profile",
        choices=["smoke", "normal"],
        default="smoke",
        help="Test level: smoke=minimal test; normal=normal test",
    )
    parser.add_argument("--processes", type=int, default=None, help="Number of parallel processes (override default values of subtasks)")
    parser.add_argument("--output-dir-morality", type=str, default="results/zealot_morality_analysis", help="morality output directory")
    parser.add_argument("--data-dir-sweep", type=str, default="results/zealot_parameter_sweep", help="sweep output/data directory")
    parser.add_argument("--sobol-config", type=str, default=None, help="sobol configuration (if not set, choose quick/standard based on profile)")

    parser.set_defaults(func=dispatch_all)


def dispatch_all(args: argparse.Namespace) -> None:
    # Configuration profiles
    profiles = {
        "smoke": {
            "morality": {
                "num_runs": 1, "max_zealots": 4, "max_morality": 4,
                "smoothing": False, "error_band_type": "std", "target_step": 2, "smooth_method": "savgol",
            },
            "sweep": {
                "runs_per_config": 1, "steps": 50, "initial_scale": 0.1, "base_seed": 42,
                "max_size_mb": 200, "optimize_data": True, "preserve_essential_only": False, "suffix": "_smoke",
            },
            "sobol": "quick",
        },
        "normal": {
            "morality": {
                "num_runs": 5, "max_zealots": 50, "max_morality": 30,
                "smoothing": True, "error_band_type": "std", "target_step": 2, "smooth_method": "savgol",
            },
            "sweep": {
                "runs_per_config": 20, "steps": 300, "initial_scale": 0.1, "base_seed": 42,
                "max_size_mb": 500, "optimize_data": True, "preserve_essential_only": False, "suffix": "",
            },
            "sobol": "standard",
        },
    }

    conf = profiles[args.profile]
    results = []
    total_start = time.time()

    # Morality task
    def _morality_task():
        m = conf["morality"]
        # Run and accumulate
        run_and_accumulate_data(
            output_dir=args.output_dir_morality,
            num_runs=m["num_runs"],
            max_zealots=m["max_zealots"],
            max_morality=m["max_morality"],
            num_processes=(args.processes or 1),
        )
        # Plotting
        plot_from_accumulated_data(
            output_dir=args.output_dir_morality,
            enable_smoothing=m["smoothing"],
            target_step=m["target_step"],
            smooth_method=m["smooth_method"],
            error_band_type=m["error_band_type"],
        )

    ok, err, dur, _ = time_exec("Morality", _morality_task)
    results.append(("Morality", ok, err, dur))

    # Sweep task
    def _sweep_task():
        w = conf["sweep"]
        run_zealot_parameter_sweep(
            runs_per_config=w["runs_per_config"],
            steps=w["steps"],
            initial_scale=w["initial_scale"],
            base_seed=w["base_seed"],
            output_base_dir=(args.data_dir_sweep + w["suffix"]),
            num_processes=(args.processes or 1) if args.profile == "smoke" else args.processes,
            max_size_mb=w["max_size_mb"],
            optimize_data=w["optimize_data"],
            preserve_essential_only=w["preserve_essential_only"],
        )

    ok, err, dur, _ = time_exec("Sweep", _sweep_task)
    results.append(("Sweep", ok, err, dur))

    # Sobol task
    def _sobol_task():
        sobol_cfg = args.sobol_config or conf["sobol"]
        analyzer, sensitivity_indices = run_sensitivity_analysis(
            config_name=sobol_cfg,
            custom_config=None,
            load_existing=False,
        )
        generate_reports(
            analyzer,
            sensitivity_indices,
            create_plots=True,
            config_name=sobol_cfg,
            start_time=None,
        )

    ok, err, dur, _ = time_exec("Sobol", _sobol_task)
    results.append(("Sobol", ok, err, dur))

    print_all_summary(results, time.time() - total_start)
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Polarization-Triangle project unified entry point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_morality_subparser(subparsers)
    add_sweep_subparser(subparsers)
    add_sobol_subparser(subparsers)

    # all mode: run three types at once (morality / sweep / sobol)
    add_all_subparser(subparsers)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    # Multiprocessing on Windows needs to be in the main guard
    sys.exit(main())



