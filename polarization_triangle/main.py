"""
项目统一入口：调用以下三个模块（以及 all 汇总模式）：
1) experiments/zealot_morality_analysis.py
2) experiments/zealot_parameter_sweep.py
3) scripts/run_sobol_analysis.py

用法示例（Windows PowerShell）：
  一、极小测试（Smoke Test，用于快速验证能否跑通）
    1) Zealot & Morality 分析（快速积累+绘图）
       python -m polarization_triangle.main morality --mode full --runs 1 --max-zealots 4 --max-morality 4 --processes 1 --error-band-type std --no-smoothing
       # 若已有数据仅绘图：
       python -m polarization_triangle.main morality --mode plot --error-band-type std --no-smoothing

    2) 参数扫描（最小化开销）
       python -m polarization_triangle.main sweep --runs 1 --steps 50 --processes 1 --data-dir results/zealot_parameter_sweep_smoke

    3) Sobol 敏感性分析（快速配置）
       python -m polarization_triangle.main sobol --config quick

  二、正常测试（参考各模块默认参数）
    1) Zealot & Morality 分析（默认参数）
       python -m polarization_triangle.main morality --mode full --runs 5 --max-zealots 50 --max-morality 30 --processes 1 --error-band-type std --smoothing

    2) 参数扫描（脚本默认参数）
       python -m polarization_triangle.main sweep --runs 20 --steps 300 --data-dir results/zealot_parameter_sweep

    3) Sobol 敏感性分析（standard 预设）
       python -m polarization_triangle.main sobol --config standard

  三、All 模式（一次性串行运行三类测试）
    - 极小测试：
      python -m polarization_triangle.main all --profile smoke --processes 1
    - 正常测试：
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
# 通用工具与配置
# ==========================

def time_exec(label: str, func, *args, **kwargs):
    start_ts = time.time()
    try:
        result = func(*args, **kwargs)
        duration = time.time() - start_ts
        print(f"\n⏱️ {label} 命令耗时: {duration:.2f}s")
        return True, None, duration, result
    except Exception as e:
        duration = time.time() - start_ts
        print(f"\n⏱️ {label} 命令耗时: {duration:.2f}s (失败)")
        return False, str(e), duration, None


def build_sobol_custom_config(args):
    # 如命令行提供覆盖项，则构造自定义配置返回；否则返回 None
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
    print("ALL 模式执行结果汇总")
    print("=" * 60)
    overall = True
    for label, success, error, duration in results:
        print(f"{label}: {'成功' if success else '失败'}" + (f"，错误: {error}" if not success and error else ""))
        print(f"  耗时: {duration:.2f}s" if isinstance(duration, (int, float)) else "  耗时: N/A")
        overall = overall and success
    print("-" * 60)
    print(f"总体结果: {'全部成功' if overall else '存在失败'}")
    print(f"总耗时: {total_elapsed:.2f}s")


def add_morality_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "morality",
        help="运行或绘制 Zealot & Morality 分析",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "accumulate", "plot", "no-zealot"],
        default="full",
        help="运行模式：full=先积累数据再绘图；accumulate=只积累；plot=只绘图；no-zealot=仅收集无zealot数据",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/zealot_morality_analysis",
        help="输出目录",
    )
    parser.add_argument("--runs", type=int, default=5, help="每个参数点的运行次数")
    parser.add_argument("--max-zealots", type=int, default=50, help="最大 zealot 数量（仅 full/accumulate 模式使用）")
    parser.add_argument("--max-morality", type=int, default=30, help="最大 morality ratio %（full/accumulate/no-zealot 使用）")
    parser.add_argument("--processes", type=int, default=1, help="并行进程数")

    # 绘图/误差带与平滑配置（plot 或 full 的绘图阶段会用到）
    parser.add_argument(
        "--error-band-type",
        choices=["std", "percentile", "confidence"],
        default="std",
        help="zealot_numbers 图的 error bands 类型（plot 或 full 的绘图阶段）",
    )
    parser.add_argument("--smoothing", dest="smoothing", action="store_true", help="morality_ratios 绘图启用平滑")
    parser.add_argument("--no-smoothing", dest="smoothing", action="store_false", help="morality_ratios 绘图禁用平滑")
    parser.set_defaults(smoothing=True)
    parser.add_argument("--step", type=int, default=2, help="重采样步长（用于平滑）")
    parser.add_argument(
        "--smooth-method",
        choices=["savgol", "moving_avg", "none"],
        default="savgol",
        help="平滑方法",
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
        help="运行 Zealot 参数扫描或仅从已有数据绘图",
    )
    parser.add_argument("--plot-only", action="store_true", help="仅绘图模式：从已有数据生成图表")
    parser.add_argument("--data-dir", type=str, default="results/zealot_parameter_sweep", help="数据目录/输出目录")
    parser.add_argument("--runs", type=int, default=20, help="每种配置运行次数")
    parser.add_argument("--steps", type=int, default=300, help="每次运行的模拟步数")
    parser.add_argument("--initial-scale", type=float, default=0.1, help="初始意见缩放因子")
    parser.add_argument("--base-seed", type=int, default=42, help="基础随机种子")
    parser.add_argument("--processes", type=int, default=None, help="并行进程数（默认使用所有CPU核心）")
    parser.add_argument("--max-size-mb", type=int, default=500, help="单个数据文件最大大小限制(MB)")
    parser.add_argument("--no-optimize", action="store_true", help="禁用数据优化")
    parser.add_argument("--essential-only", action="store_true", help="仅保留核心统计数据以节省内存")

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
        help="运行 Sobol 敏感性分析或从已有结果生成报告",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="standard",
        choices=["quick", "standard", "high_precision", "full", "test1"],
        help="预设配置",
    )
    parser.add_argument("--load", action="store_true", help="尝试加载已有结果")
    parser.add_argument("--no-plots", action="store_true", help="不生成可视化图表")
    parser.add_argument("--output-dir", type=str, help="自定义输出目录（覆盖预设）")
    parser.add_argument("--n-samples", type=int, help="自定义样本数（覆盖预设）")
    parser.add_argument("--n-runs", type=int, help="自定义运行次数（覆盖预设）")
    parser.add_argument("--n-processes", type=int, help="自定义进程数（覆盖预设）")

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
        help="一次性运行 morality、sweep、sobol 三类测试",
    )
    parser.add_argument(
        "--profile",
        choices=["smoke", "normal"],
        default="smoke",
        help="测试档位：smoke=极小测试；normal=正常测试",
    )
    parser.add_argument("--processes", type=int, default=None, help="并行进程数（覆盖各子任务默认值）")
    parser.add_argument("--output-dir-morality", type=str, default="results/zealot_morality_analysis", help="morality 输出目录")
    parser.add_argument("--data-dir-sweep", type=str, default="results/zealot_parameter_sweep", help="sweep 输出/数据目录")
    parser.add_argument("--sobol-config", type=str, default=None, help="sobol 配置（不设置则根据 profile 选择 quick/standard）")

    parser.set_defaults(func=dispatch_all)


def dispatch_all(args: argparse.Namespace) -> None:
    # 配置档位
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

    # Morality 任务
    def _morality_task():
        m = conf["morality"]
        # 运行-积累
        run_and_accumulate_data(
            output_dir=args.output_dir_morality,
            num_runs=m["num_runs"],
            max_zealots=m["max_zealots"],
            max_morality=m["max_morality"],
            num_processes=(args.processes or 1),
        )
        # 绘图
        plot_from_accumulated_data(
            output_dir=args.output_dir_morality,
            enable_smoothing=m["smoothing"],
            target_step=m["target_step"],
            smooth_method=m["smooth_method"],
            error_band_type=m["error_band_type"],
        )

    ok, err, dur, _ = time_exec("Morality", _morality_task)
    results.append(("Morality", ok, err, dur))

    # Sweep 任务
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

    # Sobol 任务
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
        description="Polarization-Triangle 项目统一入口",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_morality_subparser(subparsers)
    add_sweep_subparser(subparsers)
    add_sobol_subparser(subparsers)

    # all 模式：一次性运行三类（morality / sweep / sobol）
    add_all_subparser(subparsers)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    # Windows 下多进程需要放在 main 守卫中
    sys.exit(main())



