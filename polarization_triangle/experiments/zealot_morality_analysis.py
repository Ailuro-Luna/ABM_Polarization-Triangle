"""
Zealot and Morality Analysis Experiment

This experiment analyzes the effects of zealot numbers and morality ratios on various system metrics.
It generates two types of plots:
1. X-axis: Number of zealots
2. X-axis: Morality ratio

For each plot type, it generates 7 different Y-axis metrics:
- Mean opinion
- Variance 
- Identity opinion difference (between identity groups)
- Polarization index
- Variance per identity (+1) - variance within identity group +1
- Variance per identity (-1) - variance within identity group -1
- Variance per identity (combined) - both identity groups on same plot

Total: 14 plots (2 types × 7 metrics)

ERROR BANDS CONFIGURATION:
===========================
For zealot_numbers plots, this experiment supports three types of error bands:

1. Standard Deviation Bands:
   - Shows mean ± standard deviation
   - Traditional statistical measure of spread
   - Good for understanding overall variability

2. Percentile Bands:
   - Shows 25th to 75th percentile range (interquartile range)
   - More robust to outliers
   - Better represents the central 50% of data

3. Confidence Interval Bands:
   - Shows 99% confidence interval using t-distribution
   - Statistical inference about the true population mean
   - Accounts for sample size and uncertainty in the estimate

TO SWITCH BETWEEN ERROR BAND TYPES:
===================================
In the main function (if __name__ == "__main__":), find the section:

    # ===== ERROR BANDS Configuration: Switch by commenting/uncommenting =====
    # Method 1: Standard deviation error bands（均值 ± 标准差）
    # error_band_type = 'std'  # Use standard deviation method
    # Method 2: Percentile error bands（25th-75th百分位数）
    # error_band_type = 'percentile'  # Use percentile method
    # Method 3: Confidence interval error bands（99%置信区间）
    error_band_type = 'confidence'  # Use confidence interval method

To switch between types:
1. Comment out all lines except the one you want to use
2. For example, to use standard deviation:
   - Uncomment: error_band_type = 'std'
   - Comment out the other two options

The generated plots will show the appropriate error band type in:
- Plot titles (e.g., "with Std Dev Bands", "with Percentile Bands (25th-75th)", or "with Confidence Interval (99%)")
- File names (e.g., "zealot_numbers_mean_opinion_mean_with_std_bands.png", "_percentile_bands.png", "_confidence_bands.png")
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import copy
import time
import multiprocessing
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
import itertools
from glob import glob
from scipy import stats

from polarization_triangle.core.config import SimulationConfig, high_polarization_config
from polarization_triangle.core.simulation import Simulation
from polarization_triangle.analysis.statistics import (
    calculate_mean_opinion,
    calculate_variance_metrics,
    calculate_identity_statistics,
    get_polarization_index
)
from polarization_triangle.utils.data_manager import ExperimentDataManager


# =====================================
# Data smoothing and resampling functions
# =====================================

def resample_and_smooth_data(x_values, y_values, target_step=2, smooth_window=3):
    """
    Resample and smooth data
    
    Args:
        x_values: 原始x值数组，如[0,1,2,3,4,5,6,7,8,9,10,...]
        y_values: 原始y值数组
        target_step: 目标步长，如2表示从[0,1,2,3,4,5,...]变为[0,2,4,6,8,10,...]
        smooth_window: 平滑窗口大小
    
    Returns:
        new_x_values, new_y_values: 重采样和平滑后的数据
    """
    # 确保输入是numpy数组
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    
    # 移除NaN值
    valid_mask = ~np.isnan(y_values)
    x_clean = x_values[valid_mask]
    y_clean = y_values[valid_mask]
    
    if len(x_clean) < 2:
        return x_values, y_values
    
    # 1. 首先进行局部平滑（减少噪声）
    if smooth_window >= 3 and len(y_clean) >= smooth_window:
        # 使用移动平均进行初步平滑
        kernel = np.ones(smooth_window) / smooth_window
        y_smoothed = np.convolve(y_clean, kernel, mode='same')
        
        # 处理边界效应
        half_window = smooth_window // 2
        y_smoothed[:half_window] = y_clean[:half_window]
        y_smoothed[-half_window:] = y_clean[-half_window:]
    else:
        y_smoothed = y_clean
    
    # 2. 创建目标x值（重采样）
    x_min, x_max = x_clean[0], x_clean[-1]
    new_x_values = np.arange(x_min, x_max + target_step, target_step)
    
    # 3. 对每个新的x值，使用附近的数据点进行加权平均
    new_y_values = []
    
    for new_x in new_x_values:
        # 找到附近的点进行加权平均
        distances = np.abs(x_clean - new_x)
        
        # 使用高斯权重，距离越近权重越大
        weights = np.exp(-distances**2 / (2 * (target_step/2)**2))
        
        # 只考虑距离在target_step范围内的点
        nearby_mask = distances <= target_step
        if np.sum(nearby_mask) > 0:
            nearby_weights = weights[nearby_mask]
            nearby_y = y_smoothed[nearby_mask]
            
            # 加权平均
            weighted_y = np.average(nearby_y, weights=nearby_weights)
            new_y_values.append(weighted_y)
        else:
            # 如果没有附近的点，使用最近的点
            closest_idx = np.argmin(distances)
            new_y_values.append(y_smoothed[closest_idx])
    
    return new_x_values, np.array(new_y_values)


def apply_final_smooth(y_values, method='savgol', window=5):
    """
    对重采样后的数据进行最终平滑
    
    Args:
        y_values: 重采样后的y值
        method: 平滑方法 ('savgol', 'moving_avg', 'none')
        window: 平滑窗口
    
    Returns:
        平滑后的y值
    """
    if len(y_values) < window or method == 'none':
        return y_values
    
    if method == 'moving_avg':
        # 移动平均
        kernel = np.ones(window) / window
        smoothed = np.convolve(y_values, kernel, mode='same')
    elif method == 'savgol':
        # Savitzky-Golay滤波（需要scipy）
        try:
            from scipy.signal import savgol_filter
            # 确保window是奇数且小于数据长度
            actual_window = min(window if window % 2 == 1 else window-1, len(y_values)-1)
            if actual_window >= 3:
                smoothed = savgol_filter(y_values, actual_window, 2)
            else:
                smoothed = y_values
        except ImportError:
            # 如果没有scipy，使用移动平均
            kernel = np.ones(window) / window
            smoothed = np.convolve(y_values, kernel, mode='same')
    else:
        smoothed = y_values
    
    return smoothed


# =====================================
# 工具函数
# =====================================

def format_duration(duration: float) -> str:
    """
    格式化时间显示
    
    Args:
    duration: 持续时间（秒）
    
    Returns:
    str: 格式化的时间字符串
    """
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"


# 注：save_batch_info 函数已被 ExperimentDataManager 的批次元数据功能替代


# =====================================
# 并行计算支持函数
# =====================================

def run_single_simulation_task(task_params):
    """
    单个模拟任务的包装函数，用于多进程并行计算
    
    Args:
        task_params: 包含任务参数的元组
            (plot_type, combination, x_val, run_idx, steps, process_id, batch_seed)
    
    Returns:
        tuple: (x_val, run_idx, results_dict, success, error_msg)
    """
    try:
        plot_type, combination, x_val, run_idx, steps, process_id, batch_seed = task_params
        
        # 设置进程特定的随机种子，加入批次标识确保不同批次产生不同结果
        np.random.seed((int(x_val * 1000) + run_idx + process_id + batch_seed) % (2**32))
        
        # 构建配置
        base_config = copy.deepcopy(high_polarization_config)
        
        # 设置固定参数
        if plot_type == 'zealot_numbers':
            base_config.morality_rate = combination['morality_rate']
            base_config.zealot_identity_allocation = combination['zealot_identity_allocation']
            base_config.cluster_identity = combination['cluster_identity']
            base_config.enable_zealots = True
            base_config.steps = combination['steps']
            # 设置当前x值对应的参数
            base_config.zealot_count = int(x_val)
            base_config.zealot_mode = combination['zealot_mode']
            if x_val == 0:
                base_config.enable_zealots = False
        else:  # morality_ratios
            base_config.zealot_count = combination['zealot_count']
            base_config.zealot_mode = combination['zealot_mode']
            base_config.zealot_identity_allocation = combination['zealot_identity_allocation']
            base_config.cluster_identity = combination['cluster_identity']
            base_config.enable_zealots = combination['zealot_mode'] != 'none'
            base_config.steps = combination['steps']
            # 设置当前x值对应的参数
            base_config.morality_rate = x_val / 100.0  # 转换为0-1范围
        
        # 运行单次模拟
        results = run_single_simulation(base_config, steps)
        
        return (x_val, run_idx, results, True, None)
        
    except Exception as e:
        error_msg = f"Process {process_id}: Simulation failed for x={x_val}, run={run_idx}: {str(e)}"
        return (x_val, run_idx, None, False, error_msg)


# =====================================
# 核心实验逻辑函数
# =====================================

def create_config_combinations():
    """
    创建实验参数组合配置
    
    该函数生成两类实验的所有参数组合：
    
    1. zealot_numbers实验：测试不同zealot数量对系统的影响
       - 变量：zealot数量 (x轴)
       - 固定：zealot身份分配=True, 身份分布=random
       - 比较：zealot分布模式(random/clustered) × morality比例(0.0/0.3) = 4个组合
    
    2. morality_ratios实验：测试不同morality比例对系统的影响
       - 变量：morality比例 (x轴)
       - 固定：zealot数量=20
       - 比较：zealot模式(random/clustered/none) × zealot身份对齐(True/False) × 
               身份分布(random/clustered) = 10个组合
    
    Returns:
        dict: 包含两类实验配置的字典
            - 'zealot_numbers': 4个参数组合，用于zealot数量实验
            - 'morality_ratios': 10个参数组合，用于morality比例实验
    """
    # 基础配置：使用高极化配置作为模板
    base_config = copy.deepcopy(high_polarization_config)
    base_config.steps = 300  # 每次模拟运行300步
    
    # 初始化两类实验的参数组合容器
    combinations = {
        'zealot_numbers': [],   # 实验1：x轴为zealot数量的参数组合
        'morality_ratios': []   # 实验2：x轴为morality比例的参数组合
    }
    
    # ===== 实验1：zealot数量扫描实验 =====
    # 比较zealot分布模式和morality比例对系统的影响
    # 固定参数：zealot身份分配=True, 身份分布=random
    zealot_clustering_options = ['random', 'clustered']  # zealot分布模式：随机分布 vs 聚集分布
    morality_ratios_for_zealot_plot = [0.0, 0.3]  # 两个morality水平：无道德约束 vs 中等道德约束
    
    for clustering in zealot_clustering_options:
        for morality_ratio in morality_ratios_for_zealot_plot:
            combo = {
                'zealot_mode': clustering,                    # zealot分布模式
                'morality_rate': morality_ratio,              # morality约束强度
                'zealot_identity_allocation': True,           # zealot按身份分配（固定）
                'cluster_identity': False,                    # 身份随机分布（固定）
                'label': f'{clustering.capitalize()} Zealots, Morality={morality_ratio}',
                'steps': base_config.steps
            }
            combinations['zealot_numbers'].append(combo)
    
    # ===== 实验2：morality比例扫描实验 =====
    # 比较三个关键因素的交互影响：zealot分布、zealot身份对齐、身份分布
    # 固定参数：zealot数量=20（中等水平）
    zealot_modes = ['random', 'clustered', 'none']     # zealot模式：随机/聚集/无zealot
    zealot_identity_alignments = [True, False]         # zealot是否按身份分配
    identity_distributions = [False, True]             # 身份分布：随机 vs 聚集
    
    # 固定zealot数量为20，这是一个中等水平，既不会过度影响系统，也能观察到效果
    fixed_zealot_count = 20
    
    for zealot_mode in zealot_modes:
        if zealot_mode == 'none':
            # 无zealot情况：只需要区分身份分布方式，zealot相关参数无意义
            for identity_dist in identity_distributions:
                combo = {
                    'zealot_count': 0,                           # 无zealot
                    'zealot_mode': zealot_mode,                  # 标记为'none'
                    'zealot_identity_allocation': True,          # 默认值（不影响结果）
                    'cluster_identity': identity_dist,           # 身份分布方式
                    'label': f'{zealot_mode.capitalize()}, ID-cluster={identity_dist}',
                    'steps': base_config.steps
                }
                combinations['morality_ratios'].append(combo)
        else:
            # 有zealot情况：需要考虑zealot身份对齐方式和身份分布方式的组合效应
            for zealot_identity in zealot_identity_alignments:
                for identity_dist in identity_distributions:
                    combo = {
                        'zealot_count': fixed_zealot_count,          # 固定zealot数量
                        'zealot_mode': zealot_mode,                  # zealot分布模式
                        'zealot_identity_allocation': zealot_identity,  # zealot身份对齐方式
                        'cluster_identity': identity_dist,           # 身份分布方式
                        'label': f'{zealot_mode.capitalize()} Zealots, ID-align={zealot_identity}, ID-cluster={identity_dist}',
                        'steps': base_config.steps
                    }
                    combinations['morality_ratios'].append(combo)
    
    return combinations


def run_single_simulation(config: SimulationConfig, steps: int = 500) -> Dict[str, float]:
    """
    运行单次模拟并获取最终状态的统计指标
    
    该函数创建一个模拟实例，运行指定步数，然后计算六个关键指标：
    - Mean Opinion: 系统中非zealot agent的平均意见值
    - Variance: 意见分布的方差，衡量意见分化程度
    - Identity Opinion Difference: 不同身份群体间的平均意见差异
    - Polarization Index: 极化指数，衡量系统的极化程度
    - Variance per Identity: 每个身份群体内部的意见方差（两个身份群体分别计算）
    
    Args:
        config (SimulationConfig): 模拟配置对象，包含网络、agent、zealot等参数
        steps (int, optional): 模拟运行的步数. Defaults to 500.
    
    Returns:
        Dict[str, Any]: 包含统计指标的字典
            - 'mean_opinion': 平均意见值 (float)
            - 'variance': 意见方差 (float)
            - 'identity_opinion_difference': 身份间意见差异 (float)
            - 'polarization_index': 极化指数 (float)
            - 'variance_per_identity': 每个身份组的方差 (dict)
                - 'identity_1': identity=1组的方差
                - 'identity_-1': identity=-1组的方差
    
    Raises:
        Exception: 当模拟过程中出现错误时抛出异常
    """
    # 创建模拟实例
    sim = Simulation(config)
    
    # 逐步运行模拟至稳定状态
    for _ in range(steps):
        sim.step()
    
    # 从最终状态计算各项统计指标
    mean_stats = calculate_mean_opinion(sim, exclude_zealots=True)
    variance_stats = calculate_variance_metrics(sim, exclude_zealots=True)
    identity_stats = calculate_identity_statistics(sim, exclude_zealots=True)
    polarization = get_polarization_index(sim)
    
    # 计算identity opinion difference (身份间意见差异)
    identity_opinion_difference = 0.0
    if 'identity_difference' in identity_stats:
        identity_opinion_difference = identity_stats['identity_difference']['abs_mean_opinion_difference']
    else:
        # 理论上在正常情况下不应该到达这里（zealot数量足够小时）
        print("Warning: identity_difference not found, this should not happen under normal conditions")
        identity_opinion_difference = 0.0
    
    # 计算 variance per identity (每个身份组内的方差)
    variance_per_identity = {'identity_1': 0.0, 'identity_-1': 0.0}
    
    # 获取非zealot节点的意见和身份
    # 创建 zealot mask：如果一个agent的ID在 zealot_ids 中，则为True
    zealot_mask = np.zeros(sim.num_agents, dtype=bool)
    if sim.enable_zealots and sim.zealot_ids:
        zealot_mask[sim.zealot_ids] = True
    
    non_zealot_mask = ~zealot_mask
    non_zealot_opinions = sim.opinions[non_zealot_mask]
    non_zealot_identities = sim.identities[non_zealot_mask]
    
    # 分别计算每个身份组的方差
    for identity_val in [1, -1]:
        identity_mask = non_zealot_identities == identity_val
        if np.sum(identity_mask) > 1:  # 至少需要2个节点才能计算方差
            identity_opinions = non_zealot_opinions[identity_mask]
            variance_per_identity[f'identity_{identity_val}'] = float(np.var(identity_opinions))
        else:
            variance_per_identity[f'identity_{identity_val}'] = 0.0
    
    return {
        'mean_opinion': mean_stats['mean_opinion'],
        'variance': variance_stats['overall_variance'],
        'identity_opinion_difference': identity_opinion_difference,
        'polarization_index': polarization,
        'variance_per_identity': variance_per_identity
    }


def run_parameter_sweep(plot_type: str, combination: Dict[str, Any], 
                       x_values: List[float], num_runs: int = 5, num_processes: int = 1, 
                       batch_seed: int = 0) -> Dict[str, List[List[float]]]:
    """
    对特定参数组合进行参数扫描实验
    
    该函数针对给定的参数组合，在x轴的每个取值点运行多次模拟，收集统计数据。
    这是实验的核心执行函数，支持两种类型的扫描：
    - zealot_numbers: 固定morality比例，扫描不同的zealot数量
    - morality_ratios: 固定zealot数量，扫描不同的morality比例
    
    Args:
        plot_type (str): 实验类型
            - 'zealot_numbers': x轴为zealot数量的实验
            - 'morality_ratios': x轴为morality比例的实验
        combination (Dict[str, Any]): 参数组合字典，包含：
            - zealot_mode: zealot分布模式 ('random', 'clustered', 'none')
            - morality_rate: morality比例 (0.0-1.0)
            - zealot_identity_allocation: 是否按身份分配zealot
            - cluster_identity: 是否聚类身份分布
            - label: 组合标签
            - steps: 模拟步数
        x_values (List[float]): x轴扫描的取值列表，如 [0, 1, 2, ...]
        num_runs (int, optional): 每个x值点重复运行次数. Defaults to 5.
        num_processes (int, optional): 并行进程数，1表示串行执行. Defaults to 1.
        batch_seed (int, optional): 批次种子，确保不同批次产生不同结果. Defaults to 0.
    
    Returns:
        Dict[str, List[List[float]]]: 嵌套的结果数据结构
            格式: {metric_name: [x1_runs, x2_runs, ...]}
            其中 x1_runs = [run1_value, run2_value, ...]
            
            包含的指标:
            - 'mean_opinion': 平均意见值的多次运行结果
            - 'variance': 意见方差的多次运行结果  
            - 'identity_opinion_difference': 身份间意见差异的多次运行结果
            - 'polarization_index': 极化指数的多次运行结果
            - 'variance_per_identity_1': identity=1组内方差的多次运行结果
            - 'variance_per_identity_-1': identity=-1组内方差的多次运行结果
    """
    # 选择串行或并行执行
    if num_processes == 1:
        return run_parameter_sweep_serial(plot_type, combination, x_values, num_runs, batch_seed)
    else:
        return run_parameter_sweep_parallel(plot_type, combination, x_values, num_runs, num_processes, batch_seed)


def run_parameter_sweep_serial(plot_type: str, combination: Dict[str, Any], 
                              x_values: List[float], num_runs: int = 5, batch_seed: int = 0) -> Dict[str, List[List[float]]]:
    """
    串行版本的参数扫描（原有逻辑）
    """
    results = {
        'mean_opinion': [],
        'variance': [],
        'identity_opinion_difference': [],
        'polarization_index': [],
        'variance_per_identity_1': [],
        'variance_per_identity_-1': []
    }
    
    base_config = copy.deepcopy(high_polarization_config)
    
    # 设置固定参数
    if plot_type == 'zealot_numbers':
        base_config.morality_rate = combination['morality_rate']
        base_config.zealot_identity_allocation = combination['zealot_identity_allocation']
        base_config.cluster_identity = combination['cluster_identity']
        base_config.enable_zealots = True
        base_config.steps = combination['steps']
    else:  # morality_ratios
        base_config.zealot_count = combination['zealot_count']
        base_config.zealot_mode = combination['zealot_mode']
        base_config.zealot_identity_allocation = combination['zealot_identity_allocation']
        base_config.cluster_identity = combination['cluster_identity']
        base_config.enable_zealots = combination['zealot_mode'] != 'none'
        base_config.steps = combination['steps']
    
    # 对每个x值进行多次运行
    for x_val in tqdm(x_values, desc=f"Running {combination['label']}"):
        runs_data = {
            'mean_opinion': [],
            'variance': [],
            'identity_opinion_difference': [],
            'polarization_index': [],
            'variance_per_identity_1': [],
            'variance_per_identity_-1': []
        }
        
        # 设置当前x值对应的参数
        current_config = copy.deepcopy(base_config)
        if plot_type == 'zealot_numbers':
            current_config.zealot_count = int(x_val)
            current_config.zealot_mode = combination['zealot_mode']
            if x_val == 0:
                current_config.enable_zealots = False
        else:  # morality_ratios
            current_config.morality_rate = x_val / 100.0  # 转换为0-1范围
        
        # 运行多次模拟
        for run in range(num_runs):
            try:
                # 设置随机种子，加入批次标识确保不同批次产生不同结果
                np.random.seed((int(x_val * 1000) + run + batch_seed) % (2**32))
                
                stats = run_single_simulation(current_config)
                # 处理基础指标
                for metric in ['mean_opinion', 'variance', 'identity_opinion_difference', 'polarization_index']:
                    runs_data[metric].append(stats[metric])
                # 处理 variance per identity 指标
                variance_per_identity = stats['variance_per_identity']
                runs_data['variance_per_identity_1'].append(variance_per_identity['identity_1'])
                runs_data['variance_per_identity_-1'].append(variance_per_identity['identity_-1'])
            except Exception as e:
                print(f"Warning: Simulation failed for x={x_val}, run={run}: {e}")
                # 使用NaN填充失败的运行
                for metric in runs_data.keys():
                    runs_data[metric].append(np.nan)
        
        # 将当前x值的所有运行结果添加到总结果中
        for metric in results.keys():
            results[metric].append(runs_data[metric])
    
    return results


def run_parameter_sweep_parallel(plot_type: str, combination: Dict[str, Any], 
                                x_values: List[float], num_runs: int = 5, num_processes: int = 4, 
                                batch_seed: int = 0) -> Dict[str, List[List[float]]]:
    """
    并行版本的参数扫描
    """
    print(f"🚀 使用 {num_processes} 个进程进行并行计算...")
    
    # 创建所有任务
    tasks = []
    for x_val in x_values:
        for run_idx in range(num_runs):
            process_id = len(tasks) % num_processes  # 简单的进程ID分配
            task = (plot_type, combination, x_val, run_idx, combination['steps'], process_id, batch_seed)
            tasks.append(task)
    
    print(f"📊 总任务数: {len(tasks)} (x_values: {len(x_values)}, runs_per_x: {num_runs})")
    
    # 执行并行计算
    try:
        with multiprocessing.Pool(num_processes) as pool:
            # 使用 imap 来显示进度
            results_list = []
            with tqdm(total=len(tasks), desc=f"Running {combination['label']} (parallel)") as pbar:
                for result in pool.imap(run_single_simulation_task, tasks):
                    results_list.append(result)
                    pbar.update(1)
    except Exception as e:
        print(f"❌ 并行计算失败，回退到串行模式: {e}")
        return run_parameter_sweep_serial(plot_type, combination, x_values, num_runs, batch_seed)
    
    # 整理结果
    return organize_parallel_results(results_list, x_values, num_runs)


def organize_parallel_results(results_list: List[Tuple], x_values: List[float], num_runs: int) -> Dict[str, List[List[float]]]:
    """
    将并行计算结果重新组织为原有的数据结构
    """
    # 初始化结果结构
    organized_results = {
        'mean_opinion': [],
        'variance': [],
        'identity_opinion_difference': [],
        'polarization_index': [],
        'variance_per_identity_1': [],
        'variance_per_identity_-1': []
    }
    
    # 统计成功和失败的任务
    success_count = 0
    failure_count = 0
    
    # 按 x_value 分组整理结果
    for x_val in x_values:
        runs_data = {
            'mean_opinion': [],
            'variance': [],
            'identity_opinion_difference': [],
            'polarization_index': [],
            'variance_per_identity_1': [],
            'variance_per_identity_-1': []
        }
        
        # 收集当前 x_val 的所有运行结果
        for run_idx in range(num_runs):
            # 在结果列表中查找对应的结果
            found_result = None
            for result in results_list:
                result_x_val, result_run_idx, result_data, success, error_msg = result
                if result_x_val == x_val and result_run_idx == run_idx:
                    found_result = result
                    break
            
            if found_result and found_result[3]:  # success = True
                result_data = found_result[2]
                # 处理基础指标
                for metric in ['mean_opinion', 'variance', 'identity_opinion_difference', 'polarization_index']:
                    runs_data[metric].append(result_data[metric])
                # 处理 variance per identity 指标
                variance_per_identity = result_data['variance_per_identity']
                runs_data['variance_per_identity_1'].append(variance_per_identity['identity_1'])
                runs_data['variance_per_identity_-1'].append(variance_per_identity['identity_-1'])
                success_count += 1
            else:
                # 处理失败的任务
                if found_result:
                    print(f"⚠️  {found_result[4]}")  # 打印错误信息
                else:
                    print(f"⚠️  Missing result for x={x_val}, run={run_idx}")
                
                # 使用NaN填充失败的运行
                for metric in runs_data.keys():
                    runs_data[metric].append(np.nan)
                failure_count += 1
        
        # 将当前x值的所有运行结果添加到总结果中
        for metric in organized_results.keys():
            organized_results[metric].append(runs_data[metric])
    
    print(f"✅ 并行计算完成: {success_count} 成功, {failure_count} 失败")
    
    return organized_results


# =====================================
# 数据管理函数 (已重构为使用 ExperimentDataManager)
# =====================================

def save_data_with_manager(data_manager: ExperimentDataManager, 
                          plot_type: str, 
                          x_values: List[float], 
                          all_results: Dict[str, Dict[str, List[List[float]]]], 
                          batch_metadata: Dict[str, Any]) -> None:
    """
    使用新的数据管理器保存实验数据
    
    Args:
        data_manager: 数据管理器实例
        plot_type: 'zealot_numbers' 或 'morality_ratios'
        x_values: x轴取值
        all_results: 所有组合的结果数据
        batch_metadata: 批次元数据
    """
    # 转换数据格式以适配新的数据管理器
    batch_data = {}
    
    for combination_label, results in all_results.items():
        batch_data[combination_label] = {
            'x_values': x_values,
            'results': results
        }
    
    # 使用数据管理器保存数据
    data_manager.save_batch_results(plot_type, batch_data, batch_metadata)


# =====================================
# 绘图相关函数
# =====================================

def get_enhanced_style_config(combo_labels: List[str], plot_type: str) -> Dict[str, Dict[str, Any]]:
    """
    为组合标签生成增强的样式配置，特别针对morality_ratios的10条线进行优化
    
    Args:
    combo_labels: 组合标签列表
    plot_type: 图表类型 ('zealot_numbers' 或 'morality_ratios')
    
    Returns:
    dict: 样式配置字典
    """
    # 定义扩展的颜色调色板
    colors = [
        '#1f77b4',  # 蓝色
        '#ff7f0e',  # 橙色  
        '#2ca02c',  # 绿色
        '#d62728',  # 红色
        '#9467bd',  # 紫色
        '#8c564b',  # 棕色
        '#e377c2',  # 粉色
        '#7f7f7f',  # 灰色
        '#bcbd22',  # 橄榄色
        '#17becf',  # 青色
        '#aec7e8',  # 浅蓝色
        '#ffbb78'   # 浅橙色
    ]
    
    # 定义多种线型
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (3, 3)), (0, (1, 1))]
    
    # 定义多种标记
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'H', 'X', '+', 'x']
    
    style_config = {}
    
    if plot_type == 'morality_ratios':
        # 定义颜色映射：按zealot模式和ID-align分组
        zealot_mode_colors = {
            'None': {
                'base': '#505050',      # 深灰色 (ID-cluster=True)
                'light': '#c0c0c0'      # 浅灰色 (ID-cluster=False)
            },
            'Random': {
                'base': '#ff4500',      # 深橙红色 (ID-align=True)
                'light': '#ff8080'      # 浅粉红色 (ID-align=False)  
            },
            'Clustered': {
                'base': '#0066cc',      # 深蓝色 (ID-align=True)
                'light': '#00cc66'      # 亮绿色 (ID-align=False)
            }
        }
        
        # 定义标记映射：按ID-cluster分组
        id_cluster_markers = {
            'True': 'o',      # 圆形表示ID-cluster=True
            'False': '^'      # 三角形表示ID-cluster=False
        }
        
        # 定义标记大小映射：按ID-align分组
        id_align_sizes = {
            'True': 10,        # 大标记表示ID-align=True
            'False': 5         # 小标记表示ID-align=False
        }
        
        for label in combo_labels:
            # 解析标签中的配置信息
            if 'None' in label:
                zealot_mode = 'None'
                if 'ID-cluster=True' in label:
                    id_cluster = 'True'
                    color = zealot_mode_colors[zealot_mode]['base']
                    marker = id_cluster_markers[id_cluster]
                    markersize = 8
                else:
                    id_cluster = 'False'
                    color = zealot_mode_colors[zealot_mode]['light']
                    marker = id_cluster_markers[id_cluster]
                    markersize = 8
                
                style_config[label] = {
                    'color': color,
                    'linestyle': '-',
                    'marker': marker,
                    'markersize': markersize,
                    'group': 'None'
                }
                
            elif 'Random' in label:
                zealot_mode = 'Random'
                id_align = 'True' if 'ID-align=True' in label else 'False'
                id_cluster = 'True' if 'ID-cluster=True' in label else 'False'
                
                color = zealot_mode_colors[zealot_mode]['base'] if id_align == 'True' else zealot_mode_colors[zealot_mode]['light']
                marker = id_cluster_markers[id_cluster]
                markersize = id_align_sizes[id_align]
                
                style_config[label] = {
                    'color': color,
                    'linestyle': '-',
                    'marker': marker,
                    'markersize': markersize,
                    'group': 'Random'
                }
                
            elif 'Clustered' in label:
                zealot_mode = 'Clustered'
                id_align = 'True' if 'ID-align=True' in label else 'False'
                id_cluster = 'True' if 'ID-cluster=True' in label else 'False'
                
                color = zealot_mode_colors[zealot_mode]['base'] if id_align == 'True' else zealot_mode_colors[zealot_mode]['light']
                marker = id_cluster_markers[id_cluster]
                markersize = id_align_sizes[id_align]
                
                style_config[label] = {
                    'color': color,
                    'linestyle': '-',
                    'marker': marker,
                    'markersize': markersize,
                    'group': 'Clustered'
                }
    else:
        # 对于zealot_numbers，使用简单配置
        for i, label in enumerate(combo_labels):
            style_config[label] = {
                'color': colors[i % len(colors)],
                'linestyle': linestyles[i % len(linestyles)],
                'marker': markers[i % len(markers)],
                'markersize': 7,
                'group': 'Default'
            }
    
    # 为 variance_per_identity 指标添加带身份后缀的样式条目
    # 复制每个基础样式，添加 (ID=1) 和 (ID=-1) 后缀的变体
    variance_style_additions = {}
    for label, base_style in style_config.items():
        # 为 ID=1 创建样式（实线 + 圆形标记）
        id1_style = base_style.copy()
        id1_style['linestyle'] = '-'
        # id1_style['marker'] = 'o'
        variance_style_additions[f"{label} (ID=1)"] = id1_style
        
        # 为 ID=-1 创建样式（虚线 + 方形标记，稍小的标记）
        id_neg1_style = base_style.copy()
        id_neg1_style['linestyle'] = '--'
        # id_neg1_style['marker'] = 's'
        # id_neg1_style['markersize'] = max(6, base_style.get('markersize', 10) - 2)
        variance_style_additions[f"{label} (ID=-1)"] = id_neg1_style
        
        # 为 variance_per_identity_combined 添加带加号的变体
        # 为 ID=+1 创建样式（实线）
        id_plus1_style = base_style.copy()
        id_plus1_style['linestyle'] = '-'
        variance_style_additions[f"{label} (ID=+1)"] = id_plus1_style
    
    # 将新的样式条目添加到原始 style_config 中
    style_config.update(variance_style_additions)
    
    return style_config


def get_variance_per_identity_style(identity_label: str, plot_type: str) -> Dict[str, Any]:
    """
    为 variance per identity 图表生成特殊的样式配置
    
    Args:
        identity_label: 带身份标识的标签，如 "Random, ID-align=True (ID=1)"
        plot_type: 图表类型
    
    Returns:
        dict: 样式配置
    """
    # 扩展颜色调色板（去重并确保足够的颜色）
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
        '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d',
        '#9edae5', '#ff1744', '#00e676', '#ffea00', '#651fff', '#ff6f00',
        '#00bcd4', '#795548', '#607d8b', '#e91e63', '#4caf50', '#ffc107'
    ]
    
    # 预定义的标签到颜色索引的映射（避免哈希冲突）
    label_color_mapping = {
        # morality_ratios 实验的10个基础标签
        'Random Zealots, ID-align=True, ID-cluster=False': 0,
        'Random Zealots, ID-align=True, ID-cluster=True': 1,
        'Random Zealots, ID-align=False, ID-cluster=False': 2,
        'Random Zealots, ID-align=False, ID-cluster=True': 3,
        'Clustered Zealots, ID-align=True, ID-cluster=False': 4,
        'Clustered Zealots, ID-align=True, ID-cluster=True': 5,
        'Clustered Zealots, ID-align=False, ID-cluster=False': 6,
        'Clustered Zealots, ID-align=False, ID-cluster=True': 7,
        'None, ID-cluster=False': 8,
        'None, ID-cluster=True': 9,
        # zealot_numbers 实验的4个基础标签
        'Random Zealots, Morality=0.0': 10,
        'Random Zealots, Morality=0.3': 11,
        'Clustered Zealots, Morality=0.0': 12,
        'Clustered Zealots, Morality=0.3': 13,
    }
    
    # 线型组合：实线用于 ID=1，虚线用于 ID=-1
    linestyles = {
        '1': '-',      # 实线用于 identity=1
        '-1': '--'     # 虚线用于 identity=-1
    }
    
    # 标记形状：圆形用于 ID=1，方形用于 ID=-1
    markers = {
        '1': 'o',      # 圆形用于 identity=1
        '-1': 's'      # 方形用于 identity=-1
    }
    
    # 提取身份值
    identity_val = identity_label.split('(ID=')[-1].rstrip(')')
    
    # 提取原始组合标签
    base_label = identity_label.split(' (ID=')[0]
    
    # 使用预定义的映射或回退到哈希方法
    if base_label in label_color_mapping:
        base_color_index = label_color_mapping[base_label]
    else:
        # 回退到哈希方法（用于未预定义的标签）
        base_color_index = abs(hash(base_label)) % len(colors)
    
    # 为ID=-1组选择不同的颜色（确保无冲突）
    if identity_val == '-1':
        # 对于ID=-1，使用一个固定的偏移量确保不重复
        color_index = (base_color_index + 15) % len(colors)
    else:
        color_index = base_color_index
    
    return {
        'color': colors[color_index],
        'linestyle': linestyles.get(identity_val, '-'),
        'marker': markers.get(identity_val, 'o'),
        'markersize': 8 if identity_val == '1' else 6,  # ID=1 稍大的标记
        'group': f'identity_{identity_val}'
    }


def simplify_label(combo_label: str) -> str:
    """
    简化组合标签，同时提供向后兼容的标签转换
    
    将旧格式的标签转换为新格式以保持一致性：
    - "Random, ID-align=..." → "Random Zealots, ID-align=..."
    - "Clustered, ID-align=..." → "Clustered Zealots, ID-align=..."
    
    Args:
    combo_label: 原始组合标签
    
    Returns:
    str: 转换后的标签
    """
    # 向后兼容：将旧格式标签转换为新格式
    if combo_label.startswith('Random, ID-align='):
        return combo_label.replace('Random, ID-align=', 'Random Zealots, ID-align=')
    elif combo_label.startswith('Clustered, ID-align='):
        return combo_label.replace('Clustered, ID-align=', 'Clustered Zealots, ID-align=')
    else:
        # 对于新格式标签或其他类型标签，直接返回
        return combo_label


def plot_results_with_manager(data_manager: ExperimentDataManager, 
                            plot_type: str,
                            enable_smoothing: bool = True,
                            target_step: int = 2,
                            smooth_method: str = 'savgol',
                            error_band_type: str = 'std') -> None:
    """
    使用数据管理器绘制实验结果图表
    
    Args:
        data_manager: 数据管理器实例  
        plot_type: 'zealot_numbers' 或 'morality_ratios'
        enable_smoothing: 是否启用平滑和重采样
        target_step: 重采样的目标步长（比如从步长1变为步长2）
        smooth_method: 平滑方法 ('savgol', 'moving_avg', 'none')
        error_band_type: zealot_numbers图表的error band类型 ('std' 或 'percentile')
    """
    # 统一设置较大的最小字体，提升可读性
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 26,
        'axes.labelsize': 22,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 16
    })

    # 从数据管理器获取绘图数据
    all_results, x_values, total_runs_per_combination = data_manager.convert_to_plotting_format(plot_type)
    
    if not all_results:
        print(f"❌ No data found for {plot_type} plotting")
        return
    
    output_dir = str(data_manager.base_dir)
    metrics = ['mean_opinion', 'variance', 'identity_opinion_difference', 'polarization_index', 
               'variance_per_identity_1', 'variance_per_identity_-1', 'variance_per_identity_combined']
    metric_labels = {
        'mean_opinion': 'Mean Opinion',
        'variance': 'Opinion Variance',
        'identity_opinion_difference': 'Identity Opinion Difference',
        'polarization_index': 'Polarization Index',
        'variance_per_identity_1': 'Variance per Identity (+1)',
        'variance_per_identity_-1': 'Variance per Identity (-1)',
        'variance_per_identity_combined': 'Variance per Identity (Both Groups)'
    }
    
    x_label = 'Number of Zealots' if plot_type == 'zealot_numbers' else 'Morality Ratio (%)'
    
    # 计算总运行次数范围（用于文件名）
    min_runs = min(total_runs_per_combination.values()) if total_runs_per_combination else 0
    max_runs = max(total_runs_per_combination.values()) if total_runs_per_combination else 0
    
    if min_runs == max_runs:
        runs_suffix = f"_{min_runs}runs"
    else:
        runs_suffix = f"_{min_runs}-{max_runs}runs"
    
    # 创建 mean_plots 文件夹
    plot_folders = {
        'mean': os.path.join(output_dir, 'mean_plots')
    }
    
    os.makedirs(plot_folders['mean'], exist_ok=True)

    # 获取样式配置
    combo_labels = list(all_results.keys())
    style_config = get_enhanced_style_config(combo_labels, plot_type)
    
    print(f"\n📝 Style Configuration for {plot_type}: {len(combo_labels)} combinations")
    print(f"✅ Style configuration completed successfully")
    
    # 为每个指标生成高质量的 mean plots
    for metric in metrics:
        if enable_smoothing:
            print(f"  Generating smoothed plot for {metric_labels[metric]} (step={target_step}, method={smooth_method})...")
        else:
            print(f"  Generating high-quality mean plot for {metric_labels[metric]}...")
        
        # 预处理数据：计算均值和标准差（为error bands做准备）
        processed_data = {}
        
        if metric == 'variance_per_identity_combined':
            # 对于合并的 variance per identity 图表，为每个组合创建两条线
            for combo_label, results in all_results.items():
                # 处理 identity=1 的数据
                metric_data_1 = results['variance_per_identity_1']
                means_1, stds_1 = [], []
                lower_percentiles_1, upper_percentiles_1 = [], []
                lower_ci_1, upper_ci_1 = [], []
                for i, x_runs in enumerate(metric_data_1):
                    valid_runs = [val for val in x_runs if not np.isnan(val)]
                    if valid_runs:
                        means_1.append(np.mean(valid_runs))
                        stds_1.append(np.std(valid_runs, ddof=1) if len(valid_runs) > 1 else 0.0)
                        
                        # 计算百分位数
                        lower_p, upper_p = calculate_percentile_bands(valid_runs, percentile_range=(25.0, 75.0))
                        lower_percentiles_1.append(lower_p)
                        upper_percentiles_1.append(upper_p)
                        
                        # 计算置信区间
                        lower_c, upper_c = calculate_confidence_interval(valid_runs, confidence_level=0.95)
                        lower_ci_1.append(lower_c)
                        upper_ci_1.append(upper_c)
                    else:
                        means_1.append(np.nan)
                        stds_1.append(np.nan)
                        lower_percentiles_1.append(np.nan)
                        upper_percentiles_1.append(np.nan)
                        lower_ci_1.append(np.nan)
                        upper_ci_1.append(np.nan)
                
                # 处理 identity=-1 的数据
                metric_data_neg1 = results['variance_per_identity_-1']
                means_neg1, stds_neg1 = [], []
                lower_percentiles_neg1, upper_percentiles_neg1 = [], []
                lower_ci_neg1, upper_ci_neg1 = [], []
                for i, x_runs in enumerate(metric_data_neg1):
                    valid_runs = [val for val in x_runs if not np.isnan(val)]
                    if valid_runs:
                        means_neg1.append(np.mean(valid_runs))
                        stds_neg1.append(np.std(valid_runs, ddof=1) if len(valid_runs) > 1 else 0.0)
                        
                        # 计算百分位数
                        lower_p, upper_p = calculate_percentile_bands(valid_runs, percentile_range=(25.0, 75.0))
                        lower_percentiles_neg1.append(lower_p)
                        upper_percentiles_neg1.append(upper_p)
                        
                        # 计算置信区间
                        lower_c, upper_c = calculate_confidence_interval(valid_runs, confidence_level=0.99)
                        lower_ci_neg1.append(lower_c)
                        upper_ci_neg1.append(upper_c)
                    else:
                        means_neg1.append(np.nan)
                        stds_neg1.append(np.nan)
                        lower_percentiles_neg1.append(np.nan)
                        upper_percentiles_neg1.append(np.nan)
                        lower_ci_neg1.append(np.nan)
                        upper_ci_neg1.append(np.nan)
                
                # 创建两条线的数据
                processed_data[f"{combo_label} (ID=+1)"] = {
                    'means': np.array(means_1),
                    'stds': np.array(stds_1),
                    'lower_percentiles': np.array(lower_percentiles_1),
                    'upper_percentiles': np.array(upper_percentiles_1),
                    'lower_ci': np.array(lower_ci_1),
                    'upper_ci': np.array(upper_ci_1),
                    'identity': '+1',
                    'base_combo': combo_label
                }
                processed_data[f"{combo_label} (ID=-1)"] = {
                    'means': np.array(means_neg1),
                    'stds': np.array(stds_neg1),
                    'lower_percentiles': np.array(lower_percentiles_neg1),
                    'upper_percentiles': np.array(upper_percentiles_neg1),
                    'lower_ci': np.array(lower_ci_neg1),
                    'upper_ci': np.array(upper_ci_neg1),
                    'identity': '-1',
                    'base_combo': combo_label
                }
        elif metric.startswith('variance_per_identity') and metric != 'variance_per_identity_combined':
            # 对于单独的 variance per identity 指标，每个组合标签会被拆分为两条线
            identity_suffix = metric.split('_')[-1]  # '1' or '-1'
            
            for combo_label, results in all_results.items():
                metric_data = results[metric]
                means, stds = [], []
                lower_percentiles, upper_percentiles = [], []
                lower_ci, upper_ci = [], []
                
                for i, x_runs in enumerate(metric_data):
                    valid_runs = [val for val in x_runs if not np.isnan(val)]
                    if valid_runs:
                        means.append(np.mean(valid_runs))
                        stds.append(np.std(valid_runs, ddof=1) if len(valid_runs) > 1 else 0.0)
                        
                        # 计算百分位数
                        lower_p, upper_p = calculate_percentile_bands(valid_runs, percentile_range=(25.0, 75.0))
                        lower_percentiles.append(lower_p)
                        upper_percentiles.append(upper_p)
                        
                        # 计算置信区间
                        lower_c, upper_c = calculate_confidence_interval(valid_runs, confidence_level=0.99)
                        lower_ci.append(lower_c)
                        upper_ci.append(upper_c)
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)
                        lower_percentiles.append(np.nan)
                        upper_percentiles.append(np.nan)
                        lower_ci.append(np.nan)
                        upper_ci.append(np.nan)
                
                # 为 variance per identity 创建带身份标识的标签
                identity_label = f"{combo_label} (ID={identity_suffix})"
                processed_data[identity_label] = {
                    'means': np.array(means),
                    'stds': np.array(stds),
                    'lower_percentiles': np.array(lower_percentiles),
                    'upper_percentiles': np.array(upper_percentiles),
                    'lower_ci': np.array(lower_ci),
                    'upper_ci': np.array(upper_ci)
                }
        else:
            # 对于其他指标，计算均值、标准差、百分位数和置信区间
            for combo_label, results in all_results.items():
                metric_data = results[metric]
                means, stds = [], []
                lower_percentiles, upper_percentiles = [], []
                lower_ci, upper_ci = [], []
                
                for i, x_runs in enumerate(metric_data):
                    valid_runs = [val for val in x_runs if not np.isnan(val)]
                    if valid_runs:
                        means.append(np.mean(valid_runs))
                        stds.append(np.std(valid_runs, ddof=1) if len(valid_runs) > 1 else 0.0)
                        
                        # 计算百分位数（默认使用25th-75th百分位数）
                        lower_p, upper_p = calculate_percentile_bands(valid_runs, percentile_range=(25.0, 75.0))
                        lower_percentiles.append(lower_p)
                        upper_percentiles.append(upper_p)
                        
                        # 计算置信区间（默认使用99%置信区间）
                        lower_c, upper_c = calculate_confidence_interval(valid_runs, confidence_level=0.99)
                        lower_ci.append(lower_c)
                        upper_ci.append(upper_c)
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)
                        lower_percentiles.append(np.nan)
                        upper_percentiles.append(np.nan)
                        lower_ci.append(np.nan)
                        upper_ci.append(np.nan)
                
                processed_data[combo_label] = {
                    'means': np.array(means),
                    'stds': np.array(stds),
                    'lower_percentiles': np.array(lower_percentiles),
                    'upper_percentiles': np.array(upper_percentiles),
                    'lower_ci': np.array(lower_ci),
                    'upper_ci': np.array(upper_ci)
                }
        
        # 添加运行次数信息到标题（显示总run数）
        if plot_type == 'zealot_numbers':
            # 根据error band类型确定标题
            if error_band_type == 'std':
                band_type_str = "Std Dev Bands"
            elif error_band_type == 'percentile':
                band_type_str = "Percentile Bands (25th-75th)"
            elif error_band_type == 'confidence':
                band_type_str = "Confidence Interval (99%)"
            else:
                band_type_str = "Error Bands"
            
            title_suffix = f" with {band_type_str} ({min_runs}-{max_runs} total runs)" if min_runs != max_runs else f" with {band_type_str} ({min_runs} total runs)"
        else:
            title_suffix = f" ({min_runs}-{max_runs} total runs)" if min_runs != max_runs else f" ({min_runs} total runs)"
        
        # 高质量均值曲线图
        # 对于 variance per identity，使用更大的图表以容纳更多线条
        if metric.startswith('variance_per_identity'):
            plt.figure(figsize=(24, 14) if plot_type == 'morality_ratios' else (20, 12))
        else:
            plt.figure(figsize=(20, 12) if plot_type == 'morality_ratios' else (18, 10))
            
        for display_label, data in processed_data.items():
            # 对于 variance per identity，需要从显示标签中提取原始组合标签来获取runs信息
            if metric.startswith('variance_per_identity'):
                # 从 "Original Label (ID=1)" 中提取 "Original Label"
                if metric == 'variance_per_identity_combined':
                    # 对于合并图表，使用 base_combo 字段
                    original_combo_label = data.get('base_combo', display_label.split(' (ID=')[0])
                else:
                    original_combo_label = display_label.split(' (ID=')[0]
                runs_info = total_runs_per_combination.get(original_combo_label, 0)
            else:
                original_combo_label = display_label
                runs_info = total_runs_per_combination.get(display_label, 0)
            
            # 应用平滑和重采样
            if enable_smoothing:
                smoothed_x, smoothed_means = resample_and_smooth_data(
                    np.array(x_values), data['means'], 
                    target_step=target_step, 
                    smooth_window=3
                )
                
                # 最终平滑
                final_means = apply_final_smooth(smoothed_means, method=smooth_method, window=5)
                
                # 使用平滑后的数据
                plot_x, plot_y = smoothed_x, final_means
                
                # 同时更新标签显示平滑信息
                short_label = simplify_label(display_label)
                label_with_runs = f"{short_label} (n={runs_info}, smoothed)"
            else:
                plot_x, plot_y = np.array(x_values), data['means']
                short_label = simplify_label(display_label)
                label_with_runs = f"{short_label} (n={runs_info})"
            
            # 所有 metric 统一使用 style_config
            style = style_config.get(display_label, {})
            
            # 获取颜色和样式
            line_color = style.get('color', 'blue')
            
            # 绘制主要的均值曲线
            plt.plot(plot_x, plot_y, label=label_with_runs, 
                    color=line_color,
                    linestyle=style.get('linestyle', '-'),
                    marker=style.get('marker', 'o'), 
                    linewidth=3.5, markersize=style.get('markersize', 10), alpha=0.85,
                    markeredgewidth=2, markeredgecolor='white')
            
            # 为 zealot_numbers 添加 error bands
            if plot_type == 'zealot_numbers' and not enable_smoothing:
                if error_band_type == 'std' and 'stds' in data:
                    # Use standard deviation method（均值 ± 标准差）
                    means = data['means']
                    stds = data['stds']
                    draw_std_error_bands(x_values, means, stds, line_color, alpha=0.2)
                    
                elif error_band_type == 'percentile' and 'lower_percentiles' in data and 'upper_percentiles' in data:
                    # Use percentile method（25th-75th百分位数）
                    lower_percentiles = data['lower_percentiles']
                    upper_percentiles = data['upper_percentiles']
                    draw_percentile_error_bands(x_values, lower_percentiles, upper_percentiles, line_color, alpha=0.2)
                    
                elif error_band_type == 'confidence' and 'lower_ci' in data and 'upper_ci' in data:
                    # Use confidence interval method（99%置信区间）
                    lower_ci = data['lower_ci']
                    upper_ci = data['upper_ci']
                    draw_confidence_interval_error_bands(x_values, lower_ci, upper_ci, line_color, alpha=0.2)
        
        plt.xlabel(x_label, fontsize=22)
        plt.ylabel(metric_labels[metric], fontsize=22)
        plt.title(f'{metric_labels[metric]} vs {x_label}{title_suffix}', fontsize=26, fontweight='bold')
        plt.tick_params(axis='both', labelsize=18)
        
        # 根据指标类型和线条数量调整图例布局
        if metric == 'variance_per_identity_combined':
            # 合并的 variance per identity 图表：每个组合2条线
            if plot_type == 'morality_ratios':
                # 20条线，使用4列
                plt.legend(bbox_to_anchor=(0.5, -0.20), loc='upper center', ncol=4, 
                          fontsize=14, frameon=True, fancybox=True, shadow=True)
            else:
                # 8条线，使用3列
                plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3, fontsize=16)
        # elif metric.startswith('variance_per_identity'):
        #     # 单独的 variance per identity 图表有更多线条，需要更多列和更小字体
        #     if plot_type == 'morality_ratios':
        #         # 20条线，使用4列
        #         plt.legend(bbox_to_anchor=(0.5, -0.20), loc='upper center', ncol=4, 
        #                   fontsize=10, frameon=True, fancybox=True, shadow=True)
        #     else:
        #         # 8条线，使用3列
        #         plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3, fontsize=11)
        else:
            # 其他指标保持原有布局
            if plot_type == 'morality_ratios':
                plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3, 
                          fontsize=16, frameon=True, fancybox=True, shadow=True)
            else:
                plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=16)
        
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # 为文件名添加平滑标识和error band类型
        if enable_smoothing:
            if plot_type == 'zealot_numbers':
                filename = f"{plot_type}_{metric}_smoothed_step{target_step}_{smooth_method}{runs_suffix}.png"
            else:
                filename = f"{plot_type}_{metric}_smoothed_step{target_step}_{smooth_method}{runs_suffix}.png"
        else:
            if plot_type == 'zealot_numbers':
                # 添加error band类型到文件名
                if error_band_type == 'std':
                    band_type_suffix = "_std_bands"
                elif error_band_type == 'percentile':
                    band_type_suffix = "_percentile_bands"
                elif error_band_type == 'confidence':
                    band_type_suffix = "_confidence_bands"
                else:
                    band_type_suffix = "_error_bands"
                filename = f"{plot_type}_{metric}_mean_with{band_type_suffix}{runs_suffix}.png"
            else:
                filename = f"{plot_type}_{metric}_mean{runs_suffix}.png"
        filepath = os.path.join(plot_folders['mean'], filename)
        
        # 高质量PNG保存 (DPI 300)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='white', 
                   format='png', transparent=False, 
                   pad_inches=0.1, metadata={'Creator': 'Zealot Morality Analysis'})
        
        plt.close()
    
    print(f"  ✅ Generated high-quality mean plots for {plot_type}:")
    print(f"     - Mean line plots: {plot_folders['mean']}")


# =====================================
# 高级接口函数
# =====================================

def run_and_accumulate_data(output_dir: str = "results/zealot_morality_analysis", 
                           num_runs: int = 5, max_zealots: int = 50, max_morality: int = 30,
                           batch_name: str = "", num_processes: int = 1):
    """
    运行测试并使用新的数据管理器保存数据（第一部分）
    
    Args:
    output_dir: 输出目录
    num_runs: 本次运行的次数
    max_zealots: 最大zealot数量
    max_morality: 最大morality ratio (%)
    batch_name: 批次名称，用于标识本次运行
    num_processes: 并行进程数，1表示串行执行
    """
    print("🔬 Running Tests and Accumulating Data with New Data Manager")
    print("=" * 70)
    
    start_time = time.time()
    
    # 创建数据管理器
    data_manager = ExperimentDataManager(output_dir)
    
    # 获取参数组合
    combinations = create_config_combinations()
    
    if not batch_name:
        batch_name = time.strftime("%Y%m%d_%H%M%S")
    
    # 生成批次种子，确保不同批次产生不同的随机结果
    batch_seed = int(time.time() * 1000) % (2**31)  # 使用时间戳生成种子
    
    print(f"📊 Batch Configuration:")
    print(f"   Batch name: {batch_name}")
    print(f"   Number of runs this batch: {num_runs}")
    print(f"   Max zealots: {max_zealots}")
    print(f"   Max morality ratio: {max_morality}%")
    print(f"   Parallel processes: {num_processes} ({'Parallel' if num_processes > 1 else 'Serial'})")
    print(f"   Output directory: {output_dir}")
    print(f"   Storage format: Parquet (optimized for space and speed)")
    print()
    
    # # === 处理图1：x轴为zealot numbers ===
    print("📈 Running Test Type 1: Zealot Numbers Analysis")
    print("-" * 50)
    
    plot1_start_time = time.time()
    
    zealot_x_values = list(range(0, max_zealots + 1, 2))  # 0, 1, 2, ..., n
    zealot_results = {}
    
    for combo in combinations['zealot_numbers']:
        print(f"Running combination: {combo['label']}")
        results = run_parameter_sweep('zealot_numbers', combo, zealot_x_values, num_runs, num_processes, batch_seed)
        zealot_results[combo['label']] = results
    
    # 使用新的数据管理器保存zealot numbers的数据
    zealot_batch_metadata = {
        'batch_id': batch_name,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'experiment_type': 'zealot_numbers',
        'num_runs': num_runs,
        'max_zealots': max_zealots,
        'x_range': [0, max_zealots],
        'combinations_count': len(combinations['zealot_numbers'])
    }
    
    save_data_with_manager(data_manager, 'zealot_numbers', zealot_x_values, zealot_results, zealot_batch_metadata)
    
    plot1_end_time = time.time()
    plot1_duration = plot1_end_time - plot1_start_time
    
    print(f"⏱️  Test Type 1 completed in: {format_duration(plot1_duration)}")
    print()
    
    # === 处理图2：x轴为morality ratio ===
    print("📈 Running Test Type 2: Morality Ratio Analysis")
    print("-" * 50)
    
    plot2_start_time = time.time()
    
    morality_x_values = list(range(0, max_morality + 1, 2))  # 0, 1, 2, ..., n
    morality_results = {}
    
    for combo in combinations['morality_ratios']:
        print(f"Running combination: {combo['label']}")
        results = run_parameter_sweep('morality_ratios', combo, morality_x_values, num_runs, num_processes, batch_seed)
        morality_results[combo['label']] = results
    
    # 使用新的数据管理器保存morality ratio的数据
    morality_batch_metadata = {
        'batch_id': batch_name,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'experiment_type': 'morality_ratios', 
        'num_runs': num_runs,
        'max_morality': max_morality,
        'x_range': [0, max_morality],
        'combinations_count': len(combinations['morality_ratios'])
    }
    
    save_data_with_manager(data_manager, 'morality_ratios', morality_x_values, morality_results, morality_batch_metadata)
    
    plot2_end_time = time.time()
    plot2_duration = plot2_end_time - plot2_start_time
    
    print(f"⏱️  Test Type 2 completed in: {format_duration(plot2_duration)}")
    print()
    
    # 计算总耗时
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "=" * 70)
    print("🎉 Data Collection Completed Successfully!")
    print(f"📊 Batch '{batch_name}' with {num_runs} runs per parameter point")
    print()
    print("⏱️  Timing Summary:")
    # print(f"   Test Type 1 (Zealot Numbers): {format_duration(plot1_duration)}")
    print(f"   Test Type 2 (Morality Ratios): {format_duration(plot2_duration)}")
    print(f"   Total execution time: {format_duration(elapsed_time)}")
    print(f"📁 Data saved using Parquet format in: {output_dir}/")
    
    # 保存实验配置到数据管理器
    experiment_config = {
        'batch_name': batch_name,
        'num_runs': num_runs,
        'max_zealots': max_zealots,
        'max_morality': max_morality,
        'elapsed_time': elapsed_time,
        'total_combinations': len(combinations['zealot_numbers']) + len(combinations['morality_ratios'])
    }
    data_manager.save_experiment_config(experiment_config)
    
    # 显示数据管理器摘要
    print("\n" + data_manager.export_summary_report())


def plot_from_accumulated_data(output_dir: str = "results/zealot_morality_analysis",
                             enable_smoothing: bool = True,
                             target_step: int = 2,
                             smooth_method: str = 'savgol',
                             error_band_type: str = 'std'):
    """
    从新的数据管理器中读取数据并生成图表（第二部分）
    
    注意：zealot_numbers图表将强制关闭平滑以显示error bands，
         morality_ratios图表将使用用户指定的平滑设置
    
    Args:
        output_dir: 输出目录
        enable_smoothing: 是否启用平滑处理（仅影响morality_ratios图表）
        target_step: 重采样步长（2表示从101个点变为51个点）
        smooth_method: 平滑方法 ('savgol', 'moving_avg', 'none')
        error_band_type: zealot_numbers图表的error band类型 ('std' 或 'percentile')
    """
    print("📊 Generating Plots from Data Manager")
    if enable_smoothing:
        print(f"🎯 Smoothing enabled: step={target_step}, method={smooth_method}")
    print("=" * 70)
    
    start_time = time.time()
    
    # 创建数据管理器
    data_manager = ExperimentDataManager(output_dir)
    
    # 显示数据摘要
    print("\n" + data_manager.export_summary_report())
    
    # 生成zealot numbers图表（关闭平滑以显示error bands）
    print("\n📈 Generating Zealot Numbers Plots...")
    zealot_summary = data_manager.get_experiment_summary('zealot_numbers')
    if zealot_summary['total_records'] > 0:
        plot_results_with_manager(data_manager, 'zealot_numbers', 
                                False, target_step, smooth_method, error_band_type)  # 强制关闭平滑，使用指定的error band类型
        if error_band_type == 'std':
            band_type_description = "standard deviation"
        elif error_band_type == 'percentile':
            band_type_description = "percentile (25th-75th)"
        elif error_band_type == 'confidence':
            band_type_description = "confidence interval (99%)"
        else:
            band_type_description = "unknown"
        print(f"✅ Generated {len(zealot_summary['combinations'])} zealot numbers plots with {band_type_description} error bands")
    else:
        print("❌ No zealot numbers data found")
    
    # 生成morality ratios图表（保持用户设置的平滑选项）
    print("\n📈 Generating Morality Ratios Plots...")
    morality_summary = data_manager.get_experiment_summary('morality_ratios')
    if morality_summary['total_records'] > 0:
        plot_results_with_manager(data_manager, 'morality_ratios',
                                enable_smoothing, target_step, smooth_method, 'std')  # morality_ratios不使用error bands，传递默认值
        if enable_smoothing:
            print(f"✅ Generated {len(morality_summary['combinations'])} morality ratios plots with smoothing")
        else:
            print(f"✅ Generated {len(morality_summary['combinations'])} morality ratios plots without smoothing")
    else:
        print("❌ No morality ratios data found")
    
    # 计算总耗时
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "=" * 70)
    print("🎉 Plot Generation Completed Successfully!")
    print(f"📊 Generated plots from Parquet data files")
    if error_band_type == 'std':
        band_type_description = "standard deviation"
    elif error_band_type == 'percentile':
        band_type_description = "percentile (25th-75th)"
    elif error_band_type == 'confidence':
        band_type_description = "confidence interval (99%)"
    else:
        band_type_description = "unknown"
    print(f"📈 Zealot Numbers: {band_type_description} error bands enabled (smoothing disabled)")
    if enable_smoothing:
        print(f"📈 Morality Ratios: Smoothing enabled (step {target_step}, {smooth_method})")
    else:
        print(f"📈 Morality Ratios: Smoothing disabled")
    print(f"⏱️  Total plotting time: {format_duration(elapsed_time)}")
    print(f"📁 Plots saved to: {output_dir}/mean_plots/")


def run_zealot_morality_analysis(output_dir: str = "results/zealot_morality_analysis", 
                                num_runs: int = 5, max_zealots: int = 50, max_morality: int = 30, num_processes: int = 1,
                                error_band_type: str = 'std'):
    """
    运行完整的zealot和morality分析实验（保持向后兼容）
    
    Args:
    output_dir: 输出目录
    num_runs: 每个参数点的运行次数
    max_zealots: 最大zealot数量
    max_morality: 最大morality ratio (%)
    num_processes: 并行进程数，1表示串行执行
    error_band_type: zealot_numbers图表的error band类型 ('std' 或 'percentile')
    """
    print("🔬 Starting Complete Zealot and Morality Analysis Experiment")
    print("=" * 70)
    
    # 第一步：运行测试并累积数据
    run_and_accumulate_data(output_dir, num_runs, max_zealots, max_morality, "", num_processes)
    
    # 第二步：从累积数据生成图表
    plot_from_accumulated_data(output_dir, error_band_type=error_band_type)


def run_no_zealot_morality_data(output_dir: str = "results/zealot_morality_analysis", 
                               num_runs: int = 5, max_morality: int = 30,
                               batch_name: str = "", num_processes: int = 1):
    """
    单独运行 no zealot 的 morality ratio 数据收集（使用新数据管理器）
    
    Args:
    output_dir: 输出目录
    num_runs: 每个参数点的运行次数
    max_morality: 最大 morality ratio (%)
    batch_name: 批次名称
    num_processes: 并行进程数，1表示串行执行
    """
    print("🔬 Running No Zealot Morality Ratio Data Collection with New Data Manager")
    print("=" * 70)
    
    start_time = time.time()
    
    # 创建数据管理器
    data_manager = ExperimentDataManager(output_dir)
    
    # 获取所有参数组合
    combinations = create_config_combinations()
    
    # 只选择 zealot_mode 为 'none' 的组合
    no_zealot_combinations = [combo for combo in combinations['morality_ratios'] 
                             if combo['zealot_mode'] == 'none']
    
    if not no_zealot_combinations:
        print("❌ 没有找到 zealot_mode='none' 的组合")
        return
    
    if not batch_name:
        batch_name = f"no_zealot_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # 生成批次种子，确保不同批次产生不同的随机结果
    batch_seed = int(time.time() * 1000) % (2**31)  # 使用时间戳生成种子
    
    print(f"📊 No Zealot Batch Configuration:")
    print(f"   Batch name: {batch_name}")
    print(f"   Number of runs this batch: {num_runs}")
    print(f"   Max morality ratio: {max_morality}%")
    print(f"   Number of no-zealot combinations: {len(no_zealot_combinations)}")
    print(f"   Output directory: {output_dir}")
    print(f"   Storage format: Parquet (optimized for space and speed)")
    print()
    
    # 设置 morality ratio 的 x 轴取值
    morality_x_values = list(range(0, max_morality + 1, 2))  # 0, 2, 4, ..., max_morality
    morality_results = {}
    
    print("📈 Running No Zealot Morality Ratio Analysis")
    print("-" * 50)
    
    for combo in no_zealot_combinations:
        print(f"Running no-zealot combination: {combo['label']}")
        results = run_parameter_sweep('morality_ratios', combo, morality_x_values, num_runs, num_processes, batch_seed)
        morality_results[combo['label']] = results
    
    # 使用新的数据管理器保存 no zealot morality ratio 数据
    no_zealot_batch_metadata = {
        'batch_id': batch_name,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'experiment_type': 'morality_ratios_no_zealot',
        'num_runs': num_runs,
        'max_morality': max_morality,
        'x_range': [0, max_morality],
        'combinations_count': len(no_zealot_combinations),
        'special_conditions': 'no_zealot_only'
    }
    
    save_data_with_manager(data_manager, 'morality_ratios', morality_x_values, morality_results, no_zealot_batch_metadata)
    
    # 计算耗时
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "=" * 70)
    print("🎉 No Zealot Data Collection Completed Successfully!")
    print(f"📊 Batch '{batch_name}' with {num_runs} runs per parameter point")
    print(f"⏱️  Total execution time: {format_duration(elapsed_time)}")
    print(f"📁 Data saved using Parquet format in: {output_dir}/")
    
    # 保存实验配置到数据管理器
    experiment_config = {
        'batch_name': batch_name,
        'num_runs': num_runs,
        'max_morality': max_morality,
        'elapsed_time': elapsed_time,
        'total_combinations': len(no_zealot_combinations),
        'experiment_type': 'no_zealot_only'
    }
    data_manager.save_experiment_config(experiment_config)
    
    # 显示数据管理器摘要
    print("\n" + data_manager.export_summary_report())


def calculate_percentile_bands(valid_runs: List[float], percentile_range: Tuple[float, float] = (25.0, 75.0)) -> Tuple[float, float]:
    """
    计算给定数据的百分位数区间
    
    Args:
        valid_runs: 有效运行数据列表
        percentile_range: 百分位数范围，默认为 (25.0, 75.0) 即25th-75th百分位数
    
    Returns:
        tuple: (下百分位数, 上百分位数)
    """
    if len(valid_runs) < 2:
        return 0.0, 0.0
    
    lower_percentile = np.percentile(valid_runs, percentile_range[0])
    upper_percentile = np.percentile(valid_runs, percentile_range[1])
    
    return lower_percentile, upper_percentile


def calculate_confidence_interval(valid_runs: List[float], confidence_level: float = 0.99) -> Tuple[float, float]:
    """
    计算给定数据的置信区间
    
    Args:
        valid_runs: 有效运行数据列表
        confidence_level: 置信水平，默认为 0.99 (99%置信区间)
    
    Returns:
        tuple: (置信区间下界, 置信区间上界)
    """
    if len(valid_runs) < 2:
        return 0.0, 0.0
    
    # 计算样本均值和标准误差
    sample_mean = np.mean(valid_runs)
    sample_std = np.std(valid_runs, ddof=1)  # 样本标准差
    sample_size = len(valid_runs)
    standard_error = sample_std / np.sqrt(sample_size)
    
    # 计算t值（使用t分布，对小样本更准确）
    alpha = 1 - confidence_level
    degrees_of_freedom = sample_size - 1
    t_value = stats.t.ppf(1 - alpha/2, degrees_of_freedom)
    
    # 计算置信区间
    margin_of_error = t_value * standard_error
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error
    
    return lower_bound, upper_bound


def draw_std_error_bands(x_values, means, stds, line_color, alpha=0.2):
    """
    绘制标准差 error bands（均值 ± 标准差）
    
    Args:
        x_values: x轴数据
        means: 均值数组
        stds: 标准差数组
        line_color: 线条颜色
        alpha: 透明度
    """
    # 计算上下边界
    upper_bound = means + stds
    lower_bound = means - stds
    
    # 绘制 error bands（使用相同颜色但透明度较低）
    plt.fill_between(x_values, lower_bound, upper_bound, 
                   color=line_color, alpha=alpha, 
                   linewidth=0, interpolate=True)


def draw_percentile_error_bands(x_values, lower_percentiles, upper_percentiles, line_color, alpha=0.2):
    """
    绘制百分位数 error bands（下百分位数 - 上百分位数）
    
    Args:
        x_values: x轴数据
        lower_percentiles: 下百分位数数组
        upper_percentiles: 上百分位数数组
        line_color: 线条颜色
        alpha: 透明度
    """
    # 绘制 error bands（使用相同颜色但透明度较低）
    plt.fill_between(x_values, lower_percentiles, upper_percentiles, 
                   color=line_color, alpha=alpha, 
                   linewidth=0, interpolate=True)


def draw_confidence_interval_error_bands(x_values, lower_ci, upper_ci, line_color, alpha=0.2):
    """
    绘制置信区间 error bands（置信区间下界 - 置信区间上界）
    
    Args:
        x_values: x轴数据
        lower_ci: 置信区间下界数组
        upper_ci: 置信区间上界数组
        line_color: 线条颜色
        alpha: 透明度
    """
    # 绘制 error bands（使用相同颜色但透明度较低）
    plt.fill_between(x_values, lower_ci, upper_ci, 
                   color=line_color, alpha=alpha, 
                   linewidth=0, interpolate=True)


if __name__ == "__main__":
    # 新的分离式使用方法：
    
    # 开始计时
    main_start_time = time.time()
    
    # 方法1：分两步运行
    # 第一步：运行测试并积累数据（可以多次运行以积累更多数据）
    print("=" * 50)
    print("🚀 示例：分步骤运行实验")
    print("=" * 50)
    
    # 数据收集阶段
    data_collection_start_time = time.time()
    
    # 可以多次运行以下命令来积累数据：
    run_and_accumulate_data(
        output_dir="results/zealot_morality_analysis",
        num_runs=100,  # 每次运行100轮测试
        max_zealots=100,  
        max_morality=100,
        # batch_name="batch_001"  # 可选：给批次命名
        num_processes=8  # 使用8个进程进行并行计算
    )
    
    data_collection_end_time = time.time()
    data_collection_duration = data_collection_end_time - data_collection_start_time
    

    # 第二步：绘图阶段

    plotting_start_time = time.time()

    # ===== ERROR BANDS Configuration: Switch by commenting/uncommenting =====
    # Method 1: Standard deviation error bands（均值 ± 标准差）
    # error_band_type = 'std'  # Use standard deviation method
    # Method 2: Percentile error bands（25th-75th百分位数）
    # error_band_type = 'percentile'  # Use percentile method
    # Method 3: Confidence interval error bands（99%置信区间）
    error_band_type = 'confidence'  # Use confidence interval method

    plot_from_accumulated_data(
        output_dir="results/zealot_morality_analysis",
        enable_smoothing=False,       # 不启用平滑
        target_step=2,             # 从步长1重采样到步长2（101个点→51个点）
        smooth_method='savgol',     # 使用Savitzky-Golay平滑
        error_band_type=error_band_type  # 使用上面配置的error band类型
    )
    
    plotting_end_time = time.time()
    plotting_duration = plotting_end_time - plotting_start_time
    
    # 计算总耗时
    main_end_time = time.time()
    total_duration = main_end_time - main_start_time
    
    # 显示耗时总结
    print("\n" + "🕒" * 50)
    print("⏱️  完整实验耗时总结")
    print("🕒" * 50)
    print(f"📊 数据收集阶段耗时: {format_duration(data_collection_duration)}")
    print(f"📈 图表生成阶段耗时: {format_duration(plotting_duration)}")
    print(f"🎯 总耗时: {format_duration(total_duration)}")
    print("🕒" * 50) 