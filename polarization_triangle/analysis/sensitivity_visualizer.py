"""
敏感性分析结果可视化模块
提供多种图表类型来展示Sobol敏感性分析结果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os
import warnings

try:
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    warnings.warn("完整的可视化功能需要matplotlib")

# 尝试设置中文字体，如果失败则使用默认设置
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass  # 如果设置失败，使用默认字体


class SensitivityVisualizer:
    """敏感性分析可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = {
            'alpha': '#FF6B6B',      # 红色 - 自我激活
            'beta': '#4ECDC4',       # 青色 - 社会影响  
            'gamma': '#45B7D1',      # 蓝色 - 道德化影响
            'cohesion_factor': '#96CEB4'  # 绿色 - 凝聚力因子
        }
        
        # 设置图表风格
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def plot_sensitivity_comparison(self, sensitivity_indices: Dict[str, Dict], 
                                  output_name: str, save_path: str = None) -> plt.Figure:
        """绘制单个输出指标的敏感性对比图"""
        if output_name not in sensitivity_indices:
            raise ValueError(f"输出指标 {output_name} 不存在")
        
        indices = sensitivity_indices[output_name]
        param_names = ['alpha', 'beta', 'gamma', 'cohesion_factor']
        
        # 准备数据
        s1_values = indices['S1']
        st_values = indices['ST']
        s1_conf = indices['S1_conf']
        st_conf = indices['ST_conf']
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # 一阶敏感性指数
        x_pos = np.arange(len(param_names))
        bars1 = ax1.bar(x_pos, s1_values, yerr=s1_conf, 
                       color=[self.colors[name] for name in param_names],
                       alpha=0.7, capsize=5)
        ax1.set_xlabel('Parameters')
        ax1.set_ylabel('First-order sensitivity (S1)')
        ax1.set_title(f'{output_name} - First-order sensitivity')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(['α', 'β', 'γ', 'cohesion_factor'])
        ax1.grid(axis='y', alpha=0.3)
        
        # 总敏感性指数
        bars2 = ax2.bar(x_pos, st_values, yerr=st_conf,
                       color=[self.colors[name] for name in param_names],
                       alpha=0.7, capsize=5)
        ax2.set_xlabel('Parameters')
        ax2.set_ylabel('Total sensitivity (ST)')
        ax2.set_title(f'{output_name} - Total sensitivity')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(['α', 'β', 'γ', 'cohesion_factor'])
        ax2.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for i, (s1, st) in enumerate(zip(s1_values, st_values)):
            ax1.text(i, s1 + s1_conf[i] + 0.01, f'{s1:.3f}', 
                    ha='center', va='bottom', fontsize=9)
            ax2.text(i, st + st_conf[i] + 0.01, f'{st:.3f}', 
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_sensitivity_heatmap(self, sensitivity_indices: Dict[str, Dict], 
                               metric_type: str = 'ST', save_path: str = None) -> plt.Figure:
        """绘制敏感性热力图"""
        param_names = ['alpha', 'beta', 'gamma', 'cohesion_factor']
        output_names = list(sensitivity_indices.keys())
        
        # 准备数据矩阵
        data_matrix = []
        for output_name in output_names:
            if metric_type in sensitivity_indices[output_name]:
                row = sensitivity_indices[output_name][metric_type]
                data_matrix.append(row)
            else:
                data_matrix.append([0] * len(param_names))
        
        data_matrix = np.array(data_matrix)
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=self.figsize)
        
        im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')
        
        # 设置坐标轴
        ax.set_xticks(np.arange(len(param_names)))
        ax.set_yticks(np.arange(len(output_names)))
        ax.set_xticklabels(['α', 'β', 'γ', 'cohesion_factor'])
        ax.set_yticklabels(output_names)
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 添加数值标签
        for i in range(len(output_names)):
            for j in range(len(param_names)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(f'{metric_type} Sensitivity Index')
        
        ax.set_title(f'{metric_type} Sensitivity Heatmap')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_interaction_effects(self, sensitivity_indices: Dict[str, Dict], 
                               save_path: str = None) -> plt.Figure:
        """绘制交互效应分析图"""
        param_names = ['alpha', 'beta', 'gamma', 'cohesion_factor']
        output_names = list(sensitivity_indices.keys())
        
        # 计算交互效应强度 (ST - S1)
        interaction_data = []
        for output_name in output_names:
            indices = sensitivity_indices[output_name]
            interactions = np.array(indices['ST']) - np.array(indices['S1'])
            interaction_data.append(interactions)
        
        interaction_matrix = np.array(interaction_data)
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 交互效应热力图
        im1 = ax1.imshow(interaction_matrix, cmap='RdBu_r', aspect='auto')
        ax1.set_xticks(np.arange(len(param_names)))
        ax1.set_yticks(np.arange(len(output_names)))
        ax1.set_xticklabels(['α', 'β', 'γ', 'cohesion_factor'])
        ax1.set_yticklabels(output_names)
        ax1.set_title('Interaction Effect Strength (ST - S1)')
        
        # 添加数值标签
        for i in range(len(output_names)):
            for j in range(len(param_names)):
                text = ax1.text(j, i, f'{interaction_matrix[i, j]:.3f}',
                               ha="center", va="center", 
                               color="white" if abs(interaction_matrix[i, j]) > 0.1 else "black",
                               fontsize=8)
        
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # 2. 平均交互效应条形图
        mean_interactions = np.mean(interaction_matrix, axis=0)
        bars = ax2.bar(range(len(param_names)), mean_interactions,
                      color=[self.colors[name] for name in param_names],
                      alpha=0.7)
        ax2.set_xlabel('Parameters')
        ax2.set_ylabel('Average Interaction Effect')
        ax2.set_title('Average Interaction Effects by Parameter')
        ax2.set_xticks(range(len(param_names)))
        ax2.set_xticklabels(['α', 'β', 'γ', 'cohesion_factor'])
        ax2.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for i, val in enumerate(mean_interactions):
            ax2.text(i, val + 0.001, f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_parameter_ranking(self, sensitivity_indices: Dict[str, Dict], 
                             metric_type: str = 'ST', save_path: str = None) -> plt.Figure:
        """绘制参数重要性排序图"""
        param_names = ['alpha', 'beta', 'gamma', 'cohesion_factor']
        output_names = list(sensitivity_indices.keys())
        
        # 计算每个参数在所有输出指标上的平均敏感性
        param_importance = {}
        for i, param in enumerate(param_names):
            importances = []
            for output_name in output_names:
                if metric_type in sensitivity_indices[output_name]:
                    importances.append(sensitivity_indices[output_name][metric_type][i])
            param_importance[param] = np.mean(importances) if importances else 0.0
        
        # 按重要性排序
        sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 6))
        
        params, values = zip(*sorted_params)
        param_labels = ['α' if p=='alpha' else 'β' if p=='beta' 
                       else 'γ' if p=='gamma' else 'cohesion_factor' for p in params]
        
        bars = ax.barh(range(len(params)), values, 
                      color=[self.colors[p] for p in params], alpha=0.7)
        
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels(param_labels)
        ax.set_xlabel(f'Average {metric_type} Sensitivity')
        ax.set_title(f'Parameter Importance Ranking (based on {metric_type})')
        ax.grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, val in enumerate(values):
            ax.text(val + 0.001, i, f'{val:.3f}', va='center', ha='left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_comprehensive_report(self, sensitivity_indices: Dict[str, Dict],
                                  param_samples: np.ndarray = None,
                                  simulation_results: List[Dict[str, float]] = None,
                                  output_dir: str = "sensitivity_plots") -> Dict[str, str]:
        """创建综合分析报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        plot_files = {}
        
        try:
            # 1. 为每个输出指标创建敏感性对比图
            for output_name in sensitivity_indices.keys():
                filename = f"{output_name}_sensitivity.png"
                filepath = os.path.join(output_dir, filename)
                fig = self.plot_sensitivity_comparison(sensitivity_indices, output_name, filepath)
                plot_files[f"{output_name}_comparison"] = filepath
                plt.close(fig)
            
            # 2. 创建热力图
            for metric_type in ['S1', 'ST']:
                filename = f"heatmap_{metric_type}.png"
                filepath = os.path.join(output_dir, filename)
                fig = self.plot_sensitivity_heatmap(sensitivity_indices, metric_type, filepath)
                plot_files[f"heatmap_{metric_type}"] = filepath
                plt.close(fig)
            
            # 3. 创建交互效应图
            filename = "interaction_effects.png"
            filepath = os.path.join(output_dir, filename)
            fig = self.plot_interaction_effects(sensitivity_indices, filepath)
            plot_files["interaction_effects"] = filepath
            plt.close(fig)
            
            # 4. 创建参数排序图
            for metric_type in ['S1', 'ST']:
                filename = f"parameter_ranking_{metric_type}.png"
                filepath = os.path.join(output_dir, filename)
                fig = self.plot_parameter_ranking(sensitivity_indices, metric_type, filepath)
                plot_files[f"ranking_{metric_type}"] = filepath
                plt.close(fig)
            
            print(f"所有图表已保存到: {output_dir}")
            
        except Exception as e:
            warnings.warn(f"创建图表时出错: {e}")
        
        return plot_files


def create_sensitivity_report_example():
    """创建敏感性分析可视化报告的示例"""
    # 模拟敏感性分析结果
    np.random.seed(42)
    param_names = ['alpha', 'beta', 'gamma', 'cohesion_factor']
    output_names = ['polarization_index', 'opinion_variance', 'extreme_ratio']
    
    sensitivity_indices = {}
    for output_name in output_names:
        sensitivity_indices[output_name] = {
            'S1': np.random.random(4) * 0.5,
            'S1_conf': np.random.random(4) * 0.1,
            'ST': np.random.random(4) * 0.8 + 0.2,
            'ST_conf': np.random.random(4) * 0.1,
            'S2': np.random.random((4, 4)) * 0.3,
            'S2_conf': np.random.random((4, 4)) * 0.1
        }
    
    # 创建可视化器
    visualizer = SensitivityVisualizer()
    
    # 创建综合报告
    plot_files = visualizer.create_comprehensive_report(sensitivity_indices)
    
    return visualizer, sensitivity_indices


if __name__ == "__main__":
    # 运行示例
    create_sensitivity_report_example() 