#!/usr/bin/env python3
"""
Sobol敏感性分析主执行脚本
对极化三角框架的关键参数进行敏感性分析
"""

import os
import sys
import argparse
import time
from pathlib import Path
import json
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from polarization_triangle.analysis.sobol_analysis import SobolAnalyzer, SobolConfig
from polarization_triangle.analysis.sensitivity_visualizer import SensitivityVisualizer
from polarization_triangle.core.config import SimulationConfig


def create_analysis_configs(structural_alignment='low', morality_ratio=0.0):
    """创建不同的分析配置"""
    configs = {}
    
    # 根据条件参数调整输出目录后缀
    condition_suffix = f"_sa{structural_alignment}_mr{morality_ratio}"
    
    configs['quick'] = SobolConfig(
        n_samples=64,
        n_runs=2,
        n_processes=2,
        num_steps=100,
        structural_alignment=structural_alignment,
        morality_ratio=morality_ratio,
        output_dir=f"results/sobol_results_quick{condition_suffix}"
    )
    
    configs['standard'] = SobolConfig(
        n_samples=512,
        n_runs=3,
        n_processes=4,
        num_steps=200,
        structural_alignment=structural_alignment,
        morality_ratio=morality_ratio,
        output_dir=f"results/sobol_results_standard{condition_suffix}"
    )

    configs['high_precision'] = SobolConfig(
        n_samples=2048,
        n_runs=10,
        n_processes=8,
        num_steps=300,
        structural_alignment=structural_alignment,
        morality_ratio=morality_ratio,
        output_dir=f"results/sobol_results_test1{condition_suffix}"
    )

    configs['full'] = SobolConfig(
        n_samples=4096,
        n_runs=50,
        n_processes=8,
        num_steps=300,
        structural_alignment=structural_alignment,
        morality_ratio=morality_ratio,
        output_dir=f"results/sobol_results_full{condition_suffix}"
    )

    
    return configs


def save_parameter_record(analyzer: SobolAnalyzer, config_name: str, 
                         start_time: float, end_time: float = None):
    """保存参数配置记录文件"""
    
    print("保存参数配置记录...")
    
    # 创建记录文件路径
    record_file_txt = os.path.join(analyzer.config.output_dir, "parameter_record.txt")
    record_file_json = os.path.join(analyzer.config.output_dir, "parameter_record.json")
    
    # 获取当前时间
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 计算总样本数
    total_samples = analyzer.config.n_samples * (2 * len(analyzer.param_names) + 2)
    total_simulations = total_samples * analyzer.config.n_runs
    total_steps = total_simulations * analyzer.config.num_steps
    
    # 准备参数记录数据
    record_data = {
        "analysis_info": {
            "config_name": config_name,
            "analysis_time": current_time,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)) if end_time else "运行中",
            "duration_seconds": end_time - start_time if end_time else "运行中",
            "output_directory": analyzer.config.output_dir
        },
        "sobol_analysis_config": {
            "parameter_bounds": analyzer.config.parameter_bounds,
            "n_samples": analyzer.config.n_samples,
            "n_runs": analyzer.config.n_runs,
            "num_steps": analyzer.config.num_steps,
            "n_processes": analyzer.config.n_processes,
            "confidence_level": analyzer.config.confidence_level,
            "bootstrap_samples": analyzer.config.bootstrap_samples,
            "save_intermediate": analyzer.config.save_intermediate
        },
        "simulation_config": {
            "num_agents": analyzer.config.base_config.num_agents,
            "network_type": analyzer.config.base_config.network_type,
            "network_params": analyzer.config.base_config.network_params,
            "opinion_distribution": analyzer.config.base_config.opinion_distribution,
            "morality_rate": analyzer.config.base_config.morality_rate,
            "cluster_identity": analyzer.config.base_config.cluster_identity,
            "cluster_morality": analyzer.config.base_config.cluster_morality,
            "cluster_opinion": analyzer.config.base_config.cluster_opinion,
            "influence_factor": analyzer.config.base_config.influence_factor,
            "tolerance": analyzer.config.base_config.tolerance,
            "delta": analyzer.config.base_config.delta,
            "u": analyzer.config.base_config.u,
            "alpha": analyzer.config.base_config.alpha,
            "beta": analyzer.config.base_config.beta,
            "gamma": analyzer.config.base_config.gamma
        },
        "zealot_config": {
            "zealot_count": analyzer.config.base_config.zealot_count,
            "enable_zealots": analyzer.config.base_config.enable_zealots,
            "zealot_mode": analyzer.config.base_config.zealot_mode,
            "zealot_opinion": analyzer.config.base_config.zealot_opinion,
            "zealot_morality": analyzer.config.base_config.zealot_morality,
            "zealot_identity_allocation": analyzer.config.base_config.zealot_identity_allocation
        },
        "computation_complexity": {
            "analyzed_parameters": analyzer.param_names,
            "parameter_count": len(analyzer.param_names),
            "base_samples": analyzer.config.n_samples,
            "total_samples": total_samples,
            "runs_per_sample": analyzer.config.n_runs,
            "total_simulations": total_simulations,
            "steps_per_simulation": analyzer.config.num_steps,
            "total_computation_steps": total_steps,
            "parallel_processes": analyzer.config.n_processes
        },
        "output_metrics": {
            "polarization_metrics": [
                "polarization_index",
                "opinion_variance", 
                "extreme_ratio",
                "identity_polarization"
            ],
            "convergence_metrics": [
                "mean_abs_opinion",
                "final_stability"
            ],
            "dynamics_metrics": [
                "trajectory_length",
                "oscillation_frequency",
                "group_divergence"
            ],
            "identity_metrics": [
                "identity_variance_ratio",
                "cross_identity_correlation",
                "variance_per_identity_1",
                "variance_per_identity_neg1",
                "variance_per_identity_mean"
            ]
        }
    }
    
    # 保存JSON格式
    with open(record_file_json, 'w', encoding='utf-8') as f:
        json.dump(record_data, f, ensure_ascii=False, indent=2)
    
    # 保存文本格式（更易读）
    with open(record_file_txt, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Sobol敏感性分析参数配置记录\n")
        f.write("="*80 + "\n\n")
        
        # 分析信息
        f.write("【分析信息】\n")
        f.write(f"配置名称: {config_name}\n")
        f.write(f"分析时间: {current_time}\n")
        f.write(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        if end_time:
            f.write(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
            f.write(f"总耗时: {end_time - start_time:.2f} 秒\n")
        f.write(f"输出目录: {analyzer.config.output_dir}\n\n")
        
        # Sobol分析配置
        f.write("【Sobol敏感性分析配置】\n")
        f.write(f"基础样本数: {analyzer.config.n_samples}\n")
        f.write(f"总样本数: {total_samples} (N × (2D + 2))\n")
        f.write(f"每个样本运行次数: {analyzer.config.n_runs}\n")
        f.write(f"每次模拟步数: {analyzer.config.num_steps}\n")
        f.write(f"并行进程数: {analyzer.config.n_processes}\n")
        f.write(f"置信水平: {analyzer.config.confidence_level}\n")
        f.write(f"Bootstrap样本数: {analyzer.config.bootstrap_samples}\n")
        f.write(f"保存中间结果: {analyzer.config.save_intermediate}\n\n")
        
        # 敏感性分析参数
        f.write("【敏感性分析参数及范围】\n")
        for param, bounds in analyzer.config.parameter_bounds.items():
            f.write(f"{param}: [{bounds[0]}, {bounds[1]}]\n")
        f.write("\n")
        
        # 网络配置
        f.write("【网络配置】\n")
        f.write(f"节点数量: {analyzer.config.base_config.num_agents}\n")
        f.write(f"网络类型: {analyzer.config.base_config.network_type}\n")
        f.write("网络参数:\n")
        for key, value in analyzer.config.base_config.network_params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # 模拟配置
        f.write("【模拟配置】\n")
        f.write(f"意见分布: {analyzer.config.base_config.opinion_distribution}\n")
        f.write(f"道德化率: {analyzer.config.base_config.morality_rate}\n")
        f.write(f"身份聚类: {analyzer.config.base_config.cluster_identity}\n")
        f.write(f"道德聚类: {analyzer.config.base_config.cluster_morality}\n")
        f.write(f"意见聚类: {analyzer.config.base_config.cluster_opinion}\n")
        f.write(f"影响因子: {analyzer.config.base_config.influence_factor}\n")
        f.write(f"容忍度: {analyzer.config.base_config.tolerance}\n")
        f.write(f"意见衰减率(δ): {analyzer.config.base_config.delta}\n")
        f.write(f"意见激活系数(u): {analyzer.config.base_config.u}\n")
        f.write(f"默认自我激活系数(α): {analyzer.config.base_config.alpha}\n")
        f.write(f"默认社会影响系数(β): {analyzer.config.base_config.beta}\n")
        f.write(f"默认道德化影响系数(γ): {analyzer.config.base_config.gamma}\n\n")
        
        # Zealot配置
        f.write("【Zealot配置】\n")
        f.write(f"Zealot数量: {analyzer.config.base_config.zealot_count}\n")
        f.write(f"启用Zealot: {analyzer.config.base_config.enable_zealots}\n")
        f.write(f"Zealot模式: {analyzer.config.base_config.zealot_mode}\n")
        f.write(f"Zealot意见: {analyzer.config.base_config.zealot_opinion}\n")
        f.write(f"Zealot道德化: {analyzer.config.base_config.zealot_morality}\n")
        f.write(f"按身份分配Zealot: {analyzer.config.base_config.zealot_identity_allocation}\n\n")
        
        # 计算复杂度
        f.write("【计算复杂度】\n")
        f.write(f"分析参数数量: {len(analyzer.param_names)} ({', '.join(analyzer.param_names)})\n")
        f.write(f"基础样本数: {analyzer.config.n_samples}\n")
        f.write(f"总样本数: {total_samples}\n")
        f.write(f"每个样本运行次数: {analyzer.config.n_runs}\n")
        f.write(f"总模拟运行次数: {total_simulations:,}\n")
        f.write(f"每次模拟步数: {analyzer.config.num_steps}\n")
        f.write(f"总计算步数: {total_steps:,}\n")
        f.write(f"并行进程数: {analyzer.config.n_processes}\n\n")
        
        # 输出指标
        f.write("【输出指标】\n")
        f.write("极化相关指标:\n")
        for metric in ["polarization_index", "opinion_variance", "extreme_ratio", "identity_polarization"]:
            f.write(f"  - {metric}\n")
        f.write("收敛相关指标:\n")
        for metric in ["mean_abs_opinion", "final_stability"]:
            f.write(f"  - {metric}\n")
        f.write("动态过程指标:\n")
        for metric in ["trajectory_length", "oscillation_frequency", "group_divergence"]:
            f.write(f"  - {metric}\n")
        f.write("身份相关指标:\n")
        for metric in ["identity_variance_ratio", "cross_identity_correlation", 
                      "variance_per_identity_1", "variance_per_identity_neg1", "variance_per_identity_mean"]:
            f.write(f"  - {metric}\n")
        f.write("\n")
        
        # 指标描述
        f.write("【指标描述】\n")
        metric_descriptions = {
            'polarization_index': 'Koudenburg极化指数，衡量系统整体极化程度',
            'opinion_variance': '意见方差，反映观点分散程度',
            'extreme_ratio': '极端观点比例，|opinion| > 0.8的Agent比例',
            'identity_polarization': '身份间极化差异，不同身份群体平均意见的方差',
            'mean_abs_opinion': '平均绝对意见，系统观点强度',
            'final_stability': '最终稳定性，最后阶段的变异系数',
            'trajectory_length': '意见轨迹长度，观点变化的累积距离',
            'oscillation_frequency': '振荡频率，观点方向改变的频次',
            'group_divergence': '群体分化度，不同身份群体间的意见差异',
            'identity_variance_ratio': '身份方差比，组间方差与组内方差的比值',
            'cross_identity_correlation': '跨身份相关性，不同身份群体意见的相关系数',
            'variance_per_identity_1': '身份群体1方差，identity=1群体内部的意见方差',
            'variance_per_identity_neg1': '身份群体-1方差，identity=-1群体内部的意见方差',
            'variance_per_identity_mean': '身份群体平均方差，两个身份群体方差的均值'
        }
        
        for metric, description in metric_descriptions.items():
            f.write(f"{metric}: {description}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("记录生成完成\n")
        f.write("="*80 + "\n")
    
    print(f"参数记录已保存到:")
    print(f"  - {record_file_txt}")
    print(f"  - {record_file_json}")


def run_sensitivity_analysis(config_name: str = 'standard', 
                           custom_config: SobolConfig = None,
                           load_existing: bool = False,
                           structural_alignment: str = 'low',
                           morality_ratio: float = 0.0):
    """运行敏感性分析"""
    
    # 选择配置
    if custom_config:
        config = custom_config
    else:
        configs = create_analysis_configs(structural_alignment, morality_ratio)
        if config_name not in configs:
            raise ValueError(f"配置 '{config_name}' 不存在。可用配置: {list(configs.keys())}")
        config = configs[config_name]
    
    print(f"使用配置: {config_name}")
    print(f"Structural Alignment: {config.structural_alignment} ({'cluster_identity=True' if config.structural_alignment == 'high' else 'cluster_identity=False'})")
    print(f"Morality Ratio: {config.morality_ratio} (morality_rate={config.morality_ratio})")
    print(f"样本数: {config.n_samples}")
    print(f"运行次数: {config.n_runs}")
    print(f"进程数: {config.n_processes}")
    print(f"模拟步数: {config.num_steps}")
    print(f"输出目录: {config.output_dir}")
    
    # 创建分析器
    analyzer = SobolAnalyzer(config)
    
    # 加载已有结果或运行新分析
    if load_existing:
        print("尝试加载已有结果...")
        try:
            sensitivity_indices = analyzer.load_results()
            if sensitivity_indices:
                print("成功加载已有结果")
            else:
                print("未找到已有结果，开始新分析...")
                sensitivity_indices = analyzer.run_complete_analysis()
        except Exception as e:
            print(f"加载失败: {e}")
            print("开始新分析...")
            sensitivity_indices = analyzer.run_complete_analysis()
    else:
        # 运行完整分析
        sensitivity_indices = analyzer.run_complete_analysis()
    
    return analyzer, sensitivity_indices


def generate_reports(analyzer: SobolAnalyzer, 
                    sensitivity_indices: dict,
                    create_plots: bool = True,
                    config_name: str = "unknown",
                    start_time: float = None):
    """生成分析报告"""
    
    print("\n" + "="*60)
    print("生成分析报告")
    print("="*60)
    
    # 保存参数配置记录
    if start_time:
        save_parameter_record(analyzer, config_name, start_time, time.time())
    
    # 生成摘要表
    try:
        summary_df = analyzer.get_summary_table()
        print("\n敏感性分析摘要 (前10行):")
        print(summary_df.head(10).to_string(index=False))
        
        # 导出Excel报告
        analyzer.export_results()
        
    except Exception as e:
        print(f"生成数据报告时出错: {e}")
    
    # 生成可视化报告
    if create_plots:
        try:
            print("\n生成可视化报告...")
            visualizer = SensitivityVisualizer()
            
            # 创建图表输出目录
            plot_dir = os.path.join(analyzer.config.output_dir, "plots")
            plot_files = visualizer.create_comprehensive_report(
                sensitivity_indices,
                analyzer.param_samples,
                analyzer.simulation_results,
                plot_dir
            )
            
            print(f"可视化报告已保存到: {plot_dir}")
            
        except Exception as e:
            print(f"生成可视化报告时出错: {e}")
    
    # 打印关键发现
    print_key_findings(sensitivity_indices)


def print_key_findings(sensitivity_indices: dict):
    """打印关键发现"""
    print("\n" + "="*60)
    print("关键发现")
    print("="*60)
    
    param_names = ['alpha', 'beta', 'gamma', 'cohesion_factor']
    param_labels = ['α (自我激活)', 'β (社会影响)', 'γ (道德化影响)', 'cohesion_factor (凝聚力)']
    
    # 计算平均敏感性
    all_st_values = []
    all_s1_values = []
    
    for output_name, indices in sensitivity_indices.items():
        all_st_values.append(indices['ST'])
        all_s1_values.append(indices['S1'])
    
    if all_st_values:
        import numpy as np
        mean_st = np.mean(all_st_values, axis=0)
        mean_s1 = np.mean(all_s1_values, axis=0)
        mean_interaction = mean_st - mean_s1
        
        # 参数重要性排序
        importance_ranking = sorted(
            zip(param_labels, mean_st), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        print("\n1. 参数重要性排序 (基于平均总敏感性指数):")
        for i, (param, value) in enumerate(importance_ranking, 1):
            print(f"   {i}. {param}: {value:.3f}")
        
        # 交互效应分析
        print("\n2. 平均交互效应强度 (ST - S1):")
        for param, interaction in zip(param_labels, mean_interaction):
            if interaction > 0.1:
                level = "强"
            elif interaction > 0.05:
                level = "中等"
            else:
                level = "弱"
            print(f"   {param}: {interaction:.3f} ({level})")
        
        # 最敏感的输出指标
        print("\n3. 各输出指标的最敏感参数:")
        for output_name, indices in sensitivity_indices.items():
            max_idx = np.argmax(indices['ST'])
            max_param = param_labels[max_idx]
            max_value = indices['ST'][max_idx]
            print(f"   {output_name}: {max_param} ({max_value:.3f})")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行Sobol敏感性分析')
    parser.add_argument('--config', type=str, default='standard', 
                       choices=['quick', 'standard', 'high_precision', 'full', 'test1'],
                       help='分析配置类型')
    parser.add_argument('--load', action='store_true', 
                       help='尝试加载已有结果')
    parser.add_argument('--no-plots', action='store_true',
                       help='不生成可视化图表')
    parser.add_argument('--output-dir', type=str, 
                       help='自定义输出目录')
    parser.add_argument('--n-samples', type=int,
                       help='自定义样本数')
    parser.add_argument('--n-runs', type=int,
                       help='自定义运行次数')
    parser.add_argument('--n-processes', type=int,
                       help='自定义进程数')
    parser.add_argument('--structural-alignment', type=str, default='low',
                       choices=['low', 'high'],
                       help='结构对齐条件: low (cluster_identity=False) 或 high (cluster_identity=True)')
    parser.add_argument('--morality-ratio', type=float, default=0.0,
                       choices=[0.0, 0.3],
                       help='道德化率: 0.0 或 0.3')
    
    args = parser.parse_args()
    
    # 创建自定义配置（如果提供了自定义参数）
    custom_config = None
    if any([args.output_dir, args.n_samples, args.n_runs, args.n_processes]):
        configs = create_analysis_configs(args.structural_alignment, args.morality_ratio)
        base_config = configs[args.config]
        
        custom_config = SobolConfig(
            n_samples=args.n_samples or base_config.n_samples,
            n_runs=args.n_runs or base_config.n_runs,
            n_processes=args.n_processes or base_config.n_processes,
            output_dir=args.output_dir or base_config.output_dir,
            num_steps=base_config.num_steps,
            structural_alignment=args.structural_alignment,
            morality_ratio=args.morality_ratio,
            base_config=base_config.base_config
        )
    
    try:
        start_time = time.time()
        
        # 运行分析
        analyzer, sensitivity_indices = run_sensitivity_analysis(
            config_name=args.config,
            custom_config=custom_config,
            load_existing=args.load,
            structural_alignment=args.structural_alignment,
            morality_ratio=args.morality_ratio
        )
        
        # 生成报告
        generate_reports(
            analyzer, 
            sensitivity_indices, 
            create_plots=not args.no_plots,
            config_name=args.config,
            start_time=start_time
        )
        
        end_time = time.time()
        print(f"\n总耗时: {end_time - start_time:.2f} 秒")
        print(f"结果保存在: {analyzer.config.output_dir}")
        
    except KeyboardInterrupt:
        print("\n分析被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n分析过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 