#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据管理工具模块
提供数据保存和加载的工具函数
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import time
from pathlib import Path
import pickle
import json
from datetime import datetime


def save_trajectory_to_csv(history: List[np.ndarray], output_path: str) -> str:
    """
    将轨迹数据保存为CSV文件
    
    参数:
    history -- 意见历史数据列表
    output_path -- 输出CSV文件路径
    
    返回:
    保存的文件路径
    """
    # 转换为numpy数组
    history_array = np.array(history)
    steps, num_agents = history_array.shape
    
    # 创建数据框
    data = {
        'step': [],
        'agent_id': [],
        'opinion': []
    }
    
    # 填充数据
    for step in range(steps):
        for agent_id in range(num_agents):
            data['step'].append(step)
            data['agent_id'].append(agent_id)
            data['opinion'].append(history_array[step, agent_id])
    
    # 创建DataFrame并保存
    df = pd.DataFrame(data)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 保存到CSV
    df.to_csv(output_path, index=False)
    
    return output_path


def save_simulation_data(sim: Any, output_dir: str, prefix: str = 'sim_data') -> Dict[str, str]:
    """
    保存模拟数据到文件，便于后续进行统计分析
    
    参数:
    sim -- 模拟对象
    output_dir -- 输出目录路径
    prefix -- 文件名前缀
    
    返回:
    包含所有保存文件路径的字典
    """
    # 创建目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存轨迹数据
    trajectory_data = {
        'step': [],
        'agent_id': [],
        'opinion': [],
        'identity': [],
        'morality': [],
        'self_activation': [],
        'social_influence': []
    }
    
    # 获取完整历史
    activation_history = sim.get_activation_history()
    
    # 如果存在历史数据
    if sim.self_activation_history:
        # 为每一步、每个agent添加数据
        for step in range(len(sim.self_activation_history)):
            for agent_id in range(sim.num_agents):
                trajectory_data['step'].append(step)
                trajectory_data['agent_id'].append(agent_id)
                # 对于opinion需要从trajectory中获取，如果没有则用当前值
                if hasattr(sim, 'opinion_trajectory') and step < len(sim.opinion_trajectory):
                    trajectory_data['opinion'].append(sim.opinion_trajectory[step][agent_id])
                else:
                    trajectory_data['opinion'].append(sim.opinions[agent_id])
                
                trajectory_data['identity'].append(sim.identities[agent_id])
                trajectory_data['morality'].append(sim.morals[agent_id])
                trajectory_data['self_activation'].append(activation_history['self_activation_history'][step][agent_id])
                trajectory_data['social_influence'].append(activation_history['social_influence_history'][step][agent_id])
    
    # 将数据转换为DataFrame并保存为CSV
    df = pd.DataFrame(trajectory_data)
    trajectory_csv_path = os.path.join(output_dir, f"{prefix}_trajectory.csv")
    df.to_csv(trajectory_csv_path, index=False)
    
    # 保存最终状态数据
    final_state = {
        'agent_id': list(range(sim.num_agents)),
        'opinion': sim.opinions.tolist(),
        'identity': sim.identities.tolist(),
        'morality': sim.morals.tolist(),
        'self_activation': sim.self_activation.tolist(),
        'social_influence': sim.social_influence.tolist()
    }
    
    df_final = pd.DataFrame(final_state)
    final_csv_path = os.path.join(output_dir, f"{prefix}_final_state.csv")
    df_final.to_csv(final_csv_path, index=False)
    
    # 保存网络结构
    network_data = []
    for i in range(sim.num_agents):
        for j in range(i+1, sim.num_agents):  # 只保存上三角矩阵避免重复
            if sim.adj_matrix[i, j] > 0:
                network_data.append({
                    'source': i,
                    'target': j,
                    'weight': sim.adj_matrix[i, j]
                })
    
    df_network = pd.DataFrame(network_data)
    network_csv_path = os.path.join(output_dir, f"{prefix}_network.csv")
    df_network.to_csv(network_csv_path, index=False)
    
    # 保存模拟配置
    config_dict = vars(sim.config)
    config_data = []
    for key, value in config_dict.items():
        # 跳过无法序列化的复杂对象
        if isinstance(value, (int, float, str, bool)) or value is None:
            config_data.append({'parameter': key, 'value': value})
    
    df_config = pd.DataFrame(config_data)
    config_csv_path = os.path.join(output_dir, f"{prefix}_config.csv")
    df_config.to_csv(config_csv_path, index=False)
    
    return {
        'trajectory': trajectory_csv_path,
        'final_state': final_csv_path,
        'network': network_csv_path,
        'config': config_csv_path
    }


class ExperimentDataManager:
    """
    实验数据管理器
    
    专门用于zealot_morality_analysis实验的数据存储和读取，
    优化存储空间和加载速度的平衡。
    
    特点：
    - 使用Parquet格式，平衡压缩率和读取速度
    - 支持批次管理和数据累积
    - 为并行计算预留接口
    - 支持future variance per identity计算需求
    """
    
    def __init__(self, base_dir: str = "results/zealot_morality_analysis"):
        """
        初始化数据管理器
        
        Args:
            base_dir: 基础存储目录
        """
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "experiment_data"
        self.metadata_dir = self.base_dir / "metadata"
        
        # 创建必要的目录结构
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据文件路径
        self.zealot_numbers_file = self.data_dir / "zealot_numbers_data.parquet"
        self.morality_ratios_file = self.data_dir / "morality_ratios_data.parquet"
        
        # 元数据文件路径
        self.batch_metadata_file = self.metadata_dir / "batch_metadata.json"
        self.experiment_config_file = self.metadata_dir / "experiment_config.json"
    
    def save_batch_results(self, 
                          plot_type: str,
                          batch_data: Dict[str, Any],
                          batch_metadata: Dict[str, Any]) -> None:
        """
        保存批次实验结果
        
        Args:
            plot_type: 'zealot_numbers' 或 'morality_ratios'
            batch_data: 批次数据 {combination_label: {x_values: [], results: {}}}
            batch_metadata: 批次元数据
        """
        # 将嵌套的结果数据转换为扁平的DataFrame格式
        rows = []
        batch_id = batch_metadata.get('batch_id', f"batch_{int(time.time())}")
        timestamp = batch_metadata.get('timestamp', datetime.now().isoformat())
        
        for combination_label, combo_data in batch_data.items():
            x_values = combo_data['x_values']
            results = combo_data['results']  # {metric: [[run1, run2, ...], [run1, run2, ...], ...]}
            
            for x_idx, x_value in enumerate(x_values):
                for metric_name, metric_results in results.items():
                    if x_idx < len(metric_results):
                        for run_idx, run_value in enumerate(metric_results[x_idx]):
                            rows.append({
                                'batch_id': batch_id,
                                'timestamp': timestamp,
                                'combination': combination_label,
                                'x_value': x_value,
                                'metric': metric_name,
                                'run_index': run_idx,
                                'value': run_value
                            })
        
        # 创建DataFrame
        new_df = pd.DataFrame(rows)
        
        # 确定目标文件
        target_file = self.zealot_numbers_file if plot_type == 'zealot_numbers' else self.morality_ratios_file
        
        # 追加或创建数据文件
        if target_file.exists():
            # 读取现有数据并合并
            existing_df = pd.read_parquet(target_file)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        
        # 保存为Parquet格式（自动压缩）
        combined_df.to_parquet(target_file, compression='snappy', index=False)
        
        # 更新批次元数据
        self._update_batch_metadata(batch_metadata)
        
        print(f"💾 Saved batch data: {len(rows)} records to {target_file.name}")
    
    def load_experiment_data(self, plot_type: str) -> Optional[pd.DataFrame]:
        """
        加载实验数据
        
        Args:
            plot_type: 'zealot_numbers' 或 'morality_ratios'
        
        Returns:
            DataFrame 或 None
        """
        target_file = self.zealot_numbers_file if plot_type == 'zealot_numbers' else self.morality_ratios_file
        
        if not target_file.exists():
            return None
        
        df = pd.read_parquet(target_file)
        print(f"📂 Loaded {len(df)} records from {target_file.name}")
        return df
    
    def get_experiment_summary(self, plot_type: str) -> Dict[str, Any]:
        """
        获取实验数据摘要统计
        
        Args:
            plot_type: 'zealot_numbers' 或 'morality_ratios'
        
        Returns:
            摘要统计字典
        """
        df = self.load_experiment_data(plot_type)
        if df is None or df.empty:
            return {'total_records': 0, 'combinations': [], 'batches': [], 'metrics': []}
        
        summary = {
            'total_records': len(df),
            'combinations': sorted(df['combination'].unique().tolist()),
            'batches': sorted(df['batch_id'].unique().tolist()),
            'metrics': sorted(df['metric'].unique().tolist()),
            'x_value_range': (df['x_value'].min(), df['x_value'].max()),
            'total_runs_per_combination': {}
        }
        
        # 计算每个组合的总运行次数
        for combo in summary['combinations']:
            combo_data = df[df['combination'] == combo]
            if not combo_data.empty:
                # 计算总运行次数 = 总记录数 / (x值数量 * 指标数量)
                unique_x_values = len(combo_data['x_value'].unique())
                unique_metrics = len(combo_data['metric'].unique())
                total_runs = len(combo_data) // (unique_x_values * unique_metrics) if unique_x_values > 0 and unique_metrics > 0 else 0
                summary['total_runs_per_combination'][combo] = total_runs
        
        return summary
    
    def convert_to_plotting_format(self, plot_type: str) -> Tuple[Dict[str, Dict[str, List[List[float]]]], List[float], Dict[str, int]]:
        """
        将存储的数据转换为绘图格式
        
        Args:
            plot_type: 'zealot_numbers' 或 'morality_ratios'
        
        Returns:
            (all_results, x_values, total_runs_per_combination)
        """
        df = self.load_experiment_data(plot_type)
        if df is None or df.empty:
            return {}, [], {}
        
        # 获取所有唯一值
        combinations = sorted(df['combination'].unique())
        x_values = sorted(df['x_value'].unique())
        metrics = sorted(df['metric'].unique())
        
        # 初始化结果结构
        all_results = {}
        total_runs_per_combination = {}
        
        for combination in combinations:
            combo_data = df[df['combination'] == combination]
            
            # 计算总运行次数
            unique_x_values = len(combo_data['x_value'].unique())
            unique_metrics = len(combo_data['metric'].unique())
            total_runs = len(combo_data) // (unique_x_values * unique_metrics) if unique_x_values > 0 and unique_metrics > 0 else 0
            total_runs_per_combination[combination] = total_runs
            
            # 组织数据为绘图格式
            combo_results = {}
            
            for metric in metrics:
                metric_results = []
                metric_data = combo_data[combo_data['metric'] == metric]
                
                for x_val in x_values:
                    x_data = metric_data[metric_data['x_value'] == x_val]
                    run_values = x_data['value'].tolist()
                    metric_results.append(run_values)
                
                combo_results[metric] = metric_results
            
            all_results[combination] = combo_results
        
        return all_results, x_values, total_runs_per_combination
    
    def _update_batch_metadata(self, batch_metadata: Dict[str, Any]) -> None:
        """
        更新批次元数据
        
        Args:
            batch_metadata: 批次元数据
        """
        # 读取现有元数据
        if self.batch_metadata_file.exists():
            with open(self.batch_metadata_file, 'r', encoding='utf-8') as f:
                all_metadata = json.load(f)
        else:
            all_metadata = {'batches': []}
        
        # 添加新批次
        all_metadata['batches'].append(batch_metadata)
        
        # 保存元数据
        with open(self.batch_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    
    def get_batch_metadata(self) -> Dict[str, Any]:
        """
        获取所有批次元数据
        
        Returns:
            批次元数据字典
        """
        if not self.batch_metadata_file.exists():
            return {'batches': []}
        
        with open(self.batch_metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_experiment_config(self, config: Dict[str, Any]) -> None:
        """
        保存实验配置
        
        Args:
            config: 实验配置字典
        """
        config['saved_at'] = datetime.now().isoformat()
        
        with open(self.experiment_config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def load_experiment_config(self) -> Dict[str, Any]:
        """
        加载实验配置
        
        Returns:
            实验配置字典
        """
        if not self.experiment_config_file.exists():
            return {}
        
        with open(self.experiment_config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def clean_old_data(self, keep_batches: int = 10) -> None:
        """
        清理旧数据，保留最近的批次
        
        Args:
            keep_batches: 保留的批次数量
        """
        # TODO: 实现数据清理逻辑
        # 这个功能可以在将来需要时实现
        pass
    
    def export_summary_report(self) -> str:
        """
        导出实验摘要报告
        
        Returns:
            摘要报告字符串
        """
        zealot_summary = self.get_experiment_summary('zealot_numbers')
        morality_summary = self.get_experiment_summary('morality_ratios')
        batch_metadata = self.get_batch_metadata()
        
        report = []
        report.append("=" * 60)
        report.append("实验数据摘要报告")
        report.append("=" * 60)
        
        report.append(f"\n📊 Zealot Numbers 实验:")
        report.append(f"   总记录数: {zealot_summary['total_records']}")
        report.append(f"   参数组合数: {len(zealot_summary['combinations'])}")
        report.append(f"   批次数: {len(zealot_summary['batches'])}")
        
        report.append(f"\n�� Morality Ratios 实验:")
        report.append(f"   总记录数: {morality_summary['total_records']}")
        report.append(f"   参数组合数: {len(morality_summary['combinations'])}")
        report.append(f"   批次数: {len(morality_summary['batches'])}")
        
        report.append(f"\n📅 批次历史: {len(batch_metadata.get('batches', []))} 个批次")
        
        # 存储空间信息
        zealot_size = self.zealot_numbers_file.stat().st_size if self.zealot_numbers_file.exists() else 0
        morality_size = self.morality_ratios_file.stat().st_size if self.morality_ratios_file.exists() else 0
        total_size = zealot_size + morality_size
        
        report.append(f"\n💾 存储空间:")
        report.append(f"   Zealot Numbers: {zealot_size / 1024:.1f} KB")
        report.append(f"   Morality Ratios: {morality_size / 1024:.1f} KB")
        report.append(f"   总计: {total_size / 1024:.1f} KB")
        
        return "\n".join(report)
