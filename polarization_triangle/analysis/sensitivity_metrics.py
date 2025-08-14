"""
敏感性分析输出指标计算模块
计算用于Sobol敏感性分析的各种输出指标
"""

import numpy as np  
from typing import Dict, Optional
import warnings

from ..core.simulation import Simulation
from .statistics import (
    calculate_mean_opinion, 
    calculate_variance_metrics,
    calculate_identity_statistics
)


class SensitivityMetrics:
    """敏感性分析指标计算器"""
    
    def __init__(self):
        self.metric_names = [
            # 极化相关指标
            'polarization_index',
            'opinion_variance', 
            'extreme_ratio',
            'identity_polarization',
            
            # 收敛相关指标
            'mean_abs_opinion',
            'final_stability',
            # 'convergence_time',  # 需要更复杂的实现
            
            # 动态过程指标
            'trajectory_length',
            'oscillation_frequency',
            'group_divergence',
            
            # 身份相关指标
            'identity_variance_ratio',
            'cross_identity_correlation',
            
            # Variance per identity 指标
            'variance_per_identity_1',      # identity=1群体的方差
            'variance_per_identity_neg1',   # identity=-1群体的方差
            'variance_per_identity_mean'    # 两个群体方差的均值
        ]
    
    def calculate_all_metrics(self, sim: Simulation) -> Dict[str, float]:
        """计算所有敏感性分析指标"""
        metrics = {}
        
        try:
            # 极化相关指标
            metrics.update(self._calculate_polarization_metrics(sim))
            
            # 收敛相关指标  
            metrics.update(self._calculate_convergence_metrics(sim))
            
            # 动态过程指标
            metrics.update(self._calculate_dynamics_metrics(sim))
            
            # 身份相关指标
            metrics.update(self._calculate_identity_metrics(sim))
            
            # Variance per identity 指标
            metrics.update(self._calculate_variance_per_identity_metrics(sim))
            
        except Exception as e:
            warnings.warn(f"计算指标时出错: {e}")
            metrics = self.get_default_metrics()
        
        return metrics
    
    def _calculate_polarization_metrics(self, sim: Simulation) -> Dict[str, float]:
        """计算极化相关指标"""
        metrics = {}
        
        try:
            # Koudenburg极化指数
            metrics['polarization_index'] = self._calculate_koudenburg_polarization(sim.opinions)
            
            # 意见方差
            metrics['opinion_variance'] = np.var(sim.opinions)
            
            # 极端观点比例 (|opinion| > 0.8)
            extreme_mask = np.abs(sim.opinions) > 0.8
            metrics['extreme_ratio'] = np.mean(extreme_mask)
            
            # 身份间极化差异
            metrics['identity_polarization'] = self._calculate_identity_polarization(
                sim.opinions, sim.identities
            )
            
        except Exception as e:
            warnings.warn(f"计算极化指标时出错: {e}")
            metrics = {
                'polarization_index': 0.0,
                'opinion_variance': 0.0,
                'extreme_ratio': 0.0, 
                'identity_polarization': 0.0
            }
        
        return metrics
    
    def _calculate_convergence_metrics(self, sim: Simulation) -> Dict[str, float]:
        """计算收敛相关指标"""
        metrics = {}
        
        try:
            # 平均绝对意见
            metrics['mean_abs_opinion'] = np.mean(np.abs(sim.opinions))
            
            # 最终稳定性 (最后10%步数内的方差系数)
            if hasattr(sim, 'opinion_history') and len(sim.opinion_history) > 10:
                final_portion = int(len(sim.opinion_history) * 0.1)
                final_opinions = sim.opinion_history[-final_portion:]
                if len(final_opinions) > 1:
                    final_mean = np.mean(final_opinions, axis=0)
                    final_std = np.std(final_opinions, axis=0)
                    # 变异系数的平均值
                    cv_values = []
                    for i in range(len(final_mean)):
                        if final_mean[i] != 0:
                            cv_values.append(abs(final_std[i] / final_mean[i]))
                    metrics['final_stability'] = np.mean(cv_values) if cv_values else 0.0
                else:
                    metrics['final_stability'] = 0.0
            else:
                metrics['final_stability'] = 0.0
                
        except Exception as e:
            warnings.warn(f"计算收敛指标时出错: {e}")
            metrics = {
                'mean_abs_opinion': 0.0,
                'final_stability': 0.0
            }
        
        return metrics
    
    def _calculate_dynamics_metrics(self, sim: Simulation) -> Dict[str, float]:
        """计算动态过程指标"""
        metrics = {}
        
        try:
            # 意见轨迹长度
            metrics['trajectory_length'] = self._calculate_trajectory_length(sim)
            
            # 振荡频率
            metrics['oscillation_frequency'] = self._calculate_oscillation_frequency(sim)
            
            # 群体分化度
            metrics['group_divergence'] = self._calculate_group_divergence(sim)
            
        except Exception as e:
            warnings.warn(f"计算动态指标时出错: {e}")
            metrics = {
                'trajectory_length': 0.0,
                'oscillation_frequency': 0.0,
                'group_divergence': 0.0
            }
        
        return metrics
    
    def _calculate_identity_metrics(self, sim: Simulation) -> Dict[str, float]:
        """计算身份相关指标"""
        metrics = {}
        
        try:
            # 身份内外方差比
            metrics['identity_variance_ratio'] = self._calculate_identity_variance_ratio(
                sim.opinions, sim.identities
            )
            
            # 跨身份相关性
            metrics['cross_identity_correlation'] = self._calculate_cross_identity_correlation(
                sim.opinions, sim.identities
            )
            
        except Exception as e:
            warnings.warn(f"计算身份指标时出错: {e}")
            metrics = {
                'identity_variance_ratio': 0.0,
                'cross_identity_correlation': 0.0
            }
        
        return metrics  
    
    def _calculate_variance_per_identity_metrics(self, sim: Simulation) -> Dict[str, float]:
        """计算 variance per identity 指标"""
        metrics = {}
        
        try:
            # 获取非zealot节点的意见和身份（排除zealot的影响）
            # 创建 zealot mask：如果一个agent的ID在 zealot_ids 中，则为True
            zealot_mask = np.zeros(sim.num_agents, dtype=bool)
            if hasattr(sim, 'enable_zealots') and sim.enable_zealots and hasattr(sim, 'zealot_ids') and sim.zealot_ids:
                zealot_mask[sim.zealot_ids] = True
            
            non_zealot_mask = ~zealot_mask
            non_zealot_opinions = sim.opinions[non_zealot_mask]
            non_zealot_identities = sim.identities[non_zealot_mask]
            
            # 分别计算每个身份组的方差
            variance_identity_1 = 0.0
            variance_identity_neg1 = 0.0
            
            # 计算 identity=1 群体的方差
            identity_1_mask = non_zealot_identities == 1
            if np.sum(identity_1_mask) > 1:  # 至少需要2个节点才能计算方差
                identity_1_opinions = non_zealot_opinions[identity_1_mask]
                variance_identity_1 = float(np.var(identity_1_opinions))
            
            # 计算 identity=-1 群体的方差
            identity_neg1_mask = non_zealot_identities == -1
            if np.sum(identity_neg1_mask) > 1:  # 至少需要2个节点才能计算方差
                identity_neg1_opinions = non_zealot_opinions[identity_neg1_mask]
                variance_identity_neg1 = float(np.var(identity_neg1_opinions))
            
            # 计算两个群体方差的均值
            variance_mean = (variance_identity_1 + variance_identity_neg1) / 2.0
            
            metrics['variance_per_identity_1'] = variance_identity_1
            metrics['variance_per_identity_neg1'] = variance_identity_neg1
            metrics['variance_per_identity_mean'] = variance_mean
            
        except Exception as e:
            warnings.warn(f"计算 variance per identity 指标时出错: {e}")
            metrics = {
                'variance_per_identity_1': 0.0,
                'variance_per_identity_neg1': 0.0,
                'variance_per_identity_mean': 0.0
            }
        
        return metrics
    
    def _calculate_koudenburg_polarization(self, opinions: np.ndarray) -> float:
        """计算Koudenburg极化指数"""
        try:
            # 将意见离散化为5个类别
            categories = np.zeros(len(opinions), dtype=int)
            categories[opinions < -0.6] = 0  # 强烈反对
            categories[(-0.6 <= opinions) & (opinions < -0.2)] = 1  # 反对
            categories[(-0.2 <= opinions) & (opinions <= 0.2)] = 2  # 中立
            categories[(0.2 < opinions) & (opinions <= 0.6)] = 3  # 支持
            categories[opinions > 0.6] = 4  # 强烈支持
            
            # 计算各类别的数量
            n = np.bincount(categories, minlength=5)
            N = len(opinions)
            
            if N == 0:
                return 0.0
            
            # 计算极化指数
            numerator = (2.14 * n[1] * n[3] + 
                        2.70 * (n[0] * n[3] + n[1] * n[4]) + 
                        3.96 * n[0] * n[4])
            denominator = 0.0099 * N * N
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
            
        except Exception as e:
            warnings.warn(f"计算Koudenburg极化指数时出错: {e}")
            return 0.0
    
    def _calculate_identity_polarization(self, opinions: np.ndarray, identities: np.ndarray) -> float:
        """计算身份间极化差异"""
        try:
            unique_identities = np.unique(identities)
            if len(unique_identities) < 2:
                return 0.0
            
            # 计算每个身份群体的平均意见
            group_means = []
            for identity in unique_identities:
                mask = identities == identity
                if np.sum(mask) > 0:
                    group_means.append(np.mean(opinions[mask]))
            
            if len(group_means) < 2:
                return 0.0
            
            # 返回不同身份群体间平均意见的方差
            return np.var(group_means)
            
        except Exception as e:
            warnings.warn(f"计算身份极化时出错: {e}")
            return 0.0
    
    def _calculate_trajectory_length(self, sim: Simulation) -> float:
        """计算意见轨迹长度"""
        try:
            if not hasattr(sim, 'opinion_history') or len(sim.opinion_history) < 2:
                return 0.0
            
            total_length = 0.0
            for i in range(1, len(sim.opinion_history)):
                # 计算每步的欧几里得距离
                diff = sim.opinion_history[i] - sim.opinion_history[i-1]
                step_length = np.sqrt(np.sum(diff**2))
                total_length += step_length
            
            # 归一化到Agent数量
            return total_length / len(sim.opinions)
            
        except Exception as e:
            warnings.warn(f"计算轨迹长度时出错: {e}")
            return 0.0
    
    def _calculate_oscillation_frequency(self, sim: Simulation) -> float:
        """计算振荡频率"""
        try:
            if not hasattr(sim, 'opinion_history') or len(sim.opinion_history) < 3:
                return 0.0
            
            oscillations = 0
            for agent_idx in range(len(sim.opinions)):
                # 计算每个Agent的方向变化次数
                agent_history = [step[agent_idx] for step in sim.opinion_history]
                direction_changes = 0
                
                for i in range(2, len(agent_history)):
                    # 检查方向是否改变
                    prev_diff = agent_history[i-1] - agent_history[i-2]
                    curr_diff = agent_history[i] - agent_history[i-1]
                    
                    if prev_diff * curr_diff < 0:  # 符号相反
                        direction_changes += 1
                
                oscillations += direction_changes
            
            # 归一化
            total_steps = len(sim.opinion_history) - 2
            if total_steps > 0:
                return oscillations / (len(sim.opinions) * total_steps)
            else:
                return 0.0
                
        except Exception as e:
            warnings.warn(f"计算振荡频率时出错: {e}")
            return 0.0
    
    def _calculate_group_divergence(self, sim: Simulation) -> float:
        """计算群体分化度"""
        try:
            # 计算不同身份群体间的意见差异
            unique_identities = np.unique(sim.identities)
            if len(unique_identities) < 2:
                return 0.0
            
            group_opinions = {}
            for identity in unique_identities:
                mask = sim.identities == identity
                if np.sum(mask) > 0:
                    group_opinions[identity] = sim.opinions[mask]
            
            if len(group_opinions) < 2:
                return 0.0
            
            # 计算群体间的平均距离
            total_divergence = 0.0
            comparisons = 0
            
            identities = list(group_opinions.keys())
            for i in range(len(identities)):
                for j in range(i+1, len(identities)):
                    opinions_i = group_opinions[identities[i]]
                    opinions_j = group_opinions[identities[j]]
                    
                    # 计算两个群体的平均意见差异
                    mean_i = np.mean(opinions_i)
                    mean_j = np.mean(opinions_j)
                    divergence = abs(mean_i - mean_j)
                    
                    total_divergence += divergence
                    comparisons += 1
            
            return total_divergence / comparisons if comparisons > 0 else 0.0
            
        except Exception as e:
            warnings.warn(f"计算群体分化度时出错: {e}")
            return 0.0
    
    def _calculate_identity_variance_ratio(self, opinions: np.ndarray, identities: np.ndarray) -> float:
        """计算身份内外方差比"""
        try:
            unique_identities = np.unique(identities)
            if len(unique_identities) < 2:
                return 0.0
            
            # 计算组内方差
            within_group_var = 0.0
            total_count = 0
            
            for identity in unique_identities:
                mask = identities == identity
                group_opinions = opinions[mask]
                if len(group_opinions) > 1:
                    within_group_var += np.var(group_opinions) * len(group_opinions)
                    total_count += len(group_opinions)
            
            if total_count > 0:
                within_group_var /= total_count
            
            # 计算组间方差
            group_means = []
            for identity in unique_identities:
                mask = identities == identity
                if np.sum(mask) > 0:
                    group_means.append(np.mean(opinions[mask]))
            
            between_group_var = np.var(group_means) if len(group_means) > 1 else 0.0
            
            # 计算方差比
            if within_group_var > 0:
                return between_group_var / within_group_var
            else:
                return 0.0
                
        except Exception as e:
            warnings.warn(f"计算身份方差比时出错: {e}")
            return 0.0
    
    def _calculate_cross_identity_correlation(self, opinions: np.ndarray, identities: np.ndarray) -> float:
        """计算跨身份相关性"""
        try:
            unique_identities = np.unique(identities)
            if len(unique_identities) != 2:
                return 0.0
            
            # 获取两个身份群体的意见
            group1_mask = identities == unique_identities[0]
            group2_mask = identities == unique_identities[1]
            
            group1_opinions = opinions[group1_mask]
            group2_opinions = opinions[group2_mask]
            
            # 如果群体大小不同，取较小的大小进行比较
            min_size = min(len(group1_opinions), len(group2_opinions))
            if min_size < 2:
                return 0.0
            
            group1_sample = group1_opinions[:min_size]
            group2_sample = group2_opinions[:min_size]
            
            # 计算相关系数
            correlation = np.corrcoef(group1_sample, group2_sample)[0, 1]
            
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            warnings.warn(f"计算跨身份相关性时出错: {e}")
            return 0.0
    
    def get_default_metrics(self) -> Dict[str, float]:
        """返回默认的指标值（用于错误情况）"""
        return {name: 0.0 for name in self.metric_names}
    
    def validate_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """验证和清理指标值"""
        validated = {}
        
        for name in self.metric_names:
            if name in metrics:
                value = metrics[name]
                # 检查是否为有效数值
                if np.isnan(value) or np.isinf(value):
                    validated[name] = 0.0
                else:
                    validated[name] = float(value)
            else:
                validated[name] = 0.0
        
        return validated
    
    def get_metric_descriptions(self) -> Dict[str, str]:
        """返回指标描述"""
        return {
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