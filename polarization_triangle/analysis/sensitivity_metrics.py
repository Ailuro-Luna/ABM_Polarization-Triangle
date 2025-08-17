"""
Sensitivity analysis output metrics calculation module
Calculate various output metrics for Sobol sensitivity analysis
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
    """Sensitivity analysis metrics calculator"""
    
    def __init__(self):
        self.metric_names = [
            # Polarization-related metrics
            'polarization_index',
            'opinion_variance', 
            'extreme_ratio',
            'identity_polarization',
            
            # Convergence-related metrics
            'mean_abs_opinion',
            'final_stability',
            # 'convergence_time',  # Requires more complex implementation
            
            # Dynamic process metrics
            'trajectory_length',
            'oscillation_frequency',
            'group_divergence',
            
            # Identity-related metrics
            'identity_variance_ratio',
            'cross_identity_correlation',
            
            # Variance per identity metrics
            'variance_per_identity_1',      # Variance of identity=1 group
            'variance_per_identity_neg1',   # Variance of identity=-1 group
            'variance_per_identity_mean'    # Mean variance of the two groups
        ]
    
    def calculate_all_metrics(self, sim: Simulation) -> Dict[str, float]:
        """Calculate all sensitivity analysis metrics"""
        metrics = {}
        
        try:
            # Polarization-related metrics
            metrics.update(self._calculate_polarization_metrics(sim))
            
            # Convergence-related metrics  
            metrics.update(self._calculate_convergence_metrics(sim))
            
            # Dynamic process metrics
            metrics.update(self._calculate_dynamics_metrics(sim))
            
            # Identity-related metrics
            metrics.update(self._calculate_identity_metrics(sim))
            
            # Variance per identity metrics
            metrics.update(self._calculate_variance_per_identity_metrics(sim))
            
        except Exception as e:
            warnings.warn(f"Error calculating metrics: {e}")
            metrics = self.get_default_metrics()
        
        return metrics
    
    def _calculate_polarization_metrics(self, sim: Simulation) -> Dict[str, float]:
        """Calculate polarization-related metrics"""
        metrics = {}
        
        try:
            # Koudenburg polarization index
            metrics['polarization_index'] = self._calculate_koudenburg_polarization(sim.opinions)
            
            # Opinion variance
            metrics['opinion_variance'] = np.var(sim.opinions)
            
            # Extreme opinion ratio (|opinion| > 0.8)
            extreme_mask = np.abs(sim.opinions) > 0.8
            metrics['extreme_ratio'] = np.mean(extreme_mask)
            
            # Polarization difference between identities
            metrics['identity_polarization'] = self._calculate_identity_polarization(
                sim.opinions, sim.identities
            )
            
        except Exception as e:
            warnings.warn(f"Error calculating polarization metrics: {e}")
            metrics = {
                'polarization_index': 0.0,
                'opinion_variance': 0.0,
                'extreme_ratio': 0.0, 
                'identity_polarization': 0.0
            }
        
        return metrics
    
    def _calculate_convergence_metrics(self, sim: Simulation) -> Dict[str, float]:
        """Calculate convergence-related metrics"""
        metrics = {}
        
        try:
            # Mean absolute opinion
            metrics['mean_abs_opinion'] = np.mean(np.abs(sim.opinions))
            
            # Final stability (coefficient of variation within last 10% of steps)
            if hasattr(sim, 'opinion_history') and len(sim.opinion_history) > 10:
                final_portion = int(len(sim.opinion_history) * 0.1)
                final_opinions = sim.opinion_history[-final_portion:]
                if len(final_opinions) > 1:
                    final_mean = np.mean(final_opinions, axis=0)
                    final_std = np.std(final_opinions, axis=0)
                    # Mean coefficient of variation
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
            warnings.warn(f"Error calculating convergence metrics: {e}")
            metrics = {
                'mean_abs_opinion': 0.0,
                'final_stability': 0.0
            }
        
        return metrics
    
    def _calculate_dynamics_metrics(self, sim: Simulation) -> Dict[str, float]:
        """Calculate dynamic process metrics"""
        metrics = {}
        
        try:
            # Opinion trajectory length
            metrics['trajectory_length'] = self._calculate_trajectory_length(sim)
            
            # Oscillation frequency
            metrics['oscillation_frequency'] = self._calculate_oscillation_frequency(sim)
            
            # Group divergence
            metrics['group_divergence'] = self._calculate_group_divergence(sim)
            
        except Exception as e:
            warnings.warn(f"Error calculating dynamic metrics: {e}")
            metrics = {
                'trajectory_length': 0.0,
                'oscillation_frequency': 0.0,
                'group_divergence': 0.0
            }
        
        return metrics
    
    def _calculate_identity_metrics(self, sim: Simulation) -> Dict[str, float]:
        """Calculate identity-related metrics"""
        metrics = {}
        
        try:
            # Identity within/between variance ratio
            metrics['identity_variance_ratio'] = self._calculate_identity_variance_ratio(
                sim.opinions, sim.identities
            )
            
            # Cross-identity correlation
            metrics['cross_identity_correlation'] = self._calculate_cross_identity_correlation(
                sim.opinions, sim.identities
            )
            
        except Exception as e:
            warnings.warn(f"Error calculating identity metrics: {e}")
            metrics = {
                'identity_variance_ratio': 0.0,
                'cross_identity_correlation': 0.0
            }
        
        return metrics  
    
    def _calculate_variance_per_identity_metrics(self, sim: Simulation) -> Dict[str, float]:
        """Calculate variance per identity metrics"""
        metrics = {}
        
        try:
            # Get opinions and identities of non-zealot nodes (exclude zealot influence)
            # Create zealot mask: True if an agent's ID is in zealot_ids
            zealot_mask = np.zeros(sim.num_agents, dtype=bool)
            if hasattr(sim, 'enable_zealots') and sim.enable_zealots and hasattr(sim, 'zealot_ids') and sim.zealot_ids:
                zealot_mask[sim.zealot_ids] = True
            
            non_zealot_mask = ~zealot_mask
            non_zealot_opinions = sim.opinions[non_zealot_mask]
            non_zealot_identities = sim.identities[non_zealot_mask]
            
            # Calculate variance for each identity group separately
            variance_identity_1 = 0.0
            variance_identity_neg1 = 0.0
            
            # Calculate variance of identity=1 group
            identity_1_mask = non_zealot_identities == 1
            if np.sum(identity_1_mask) > 1:  # Need at least 2 nodes to calculate variance
                identity_1_opinions = non_zealot_opinions[identity_1_mask]
                variance_identity_1 = float(np.var(identity_1_opinions))
            
            # Calculate variance of identity=-1 group
            identity_neg1_mask = non_zealot_identities == -1
            if np.sum(identity_neg1_mask) > 1:  # Need at least 2 nodes to calculate variance
                identity_neg1_opinions = non_zealot_opinions[identity_neg1_mask]
                variance_identity_neg1 = float(np.var(identity_neg1_opinions))
            
            # Calculate mean variance of the two groups
            variance_mean = (variance_identity_1 + variance_identity_neg1) / 2.0
            
            metrics['variance_per_identity_1'] = variance_identity_1
            metrics['variance_per_identity_neg1'] = variance_identity_neg1
            metrics['variance_per_identity_mean'] = variance_mean
            
        except Exception as e:
            warnings.warn(f"Error calculating variance per identity metrics: {e}")
            metrics = {
                'variance_per_identity_1': 0.0,
                'variance_per_identity_neg1': 0.0,
                'variance_per_identity_mean': 0.0
            }
        
        return metrics
    
    def _calculate_koudenburg_polarization(self, opinions: np.ndarray) -> float:
        """Calculate Koudenburg polarization index"""
        try:
            # Discretize opinions into 5 categories
            categories = np.zeros(len(opinions), dtype=int)
            categories[opinions < -0.6] = 0  # Strongly oppose
            categories[(-0.6 <= opinions) & (opinions < -0.2)] = 1  # Oppose
            categories[(-0.2 <= opinions) & (opinions <= 0.2)] = 2  # Neutral
            categories[(0.2 < opinions) & (opinions <= 0.6)] = 3  # Support
            categories[opinions > 0.6] = 4  # Strongly support
            
            # Calculate count for each category
            n = np.bincount(categories, minlength=5)
            N = len(opinions)
            
            if N == 0:
                return 0.0
            
            # Calculate polarization index
            numerator = (2.14 * n[1] * n[3] + 
                        2.70 * (n[0] * n[3] + n[1] * n[4]) + 
                        3.96 * n[0] * n[4])
            denominator = 0.0099 * N * N
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
            
        except Exception as e:
            warnings.warn(f"Error calculating Koudenburg polarization index: {e}")
            return 0.0
    
    def _calculate_identity_polarization(self, opinions: np.ndarray, identities: np.ndarray) -> float:
        """Calculate polarization difference between identities"""
        try:
            unique_identities = np.unique(identities)
            if len(unique_identities) < 2:
                return 0.0
            
            # Calculate mean opinion for each identity group
            group_means = []
            for identity in unique_identities:
                mask = identities == identity
                if np.sum(mask) > 0:
                    group_means.append(np.mean(opinions[mask]))
            
            if len(group_means) < 2:
                return 0.0
            
            # Return variance of mean opinions between different identity groups
            return np.var(group_means)
            
        except Exception as e:
            warnings.warn(f"Error calculating identity polarization: {e}")
            return 0.0
    
    def _calculate_trajectory_length(self, sim: Simulation) -> float:
        """Calculate opinion trajectory length"""
        try:
            if not hasattr(sim, 'opinion_history') or len(sim.opinion_history) < 2:
                return 0.0
            
            total_length = 0.0
            for i in range(1, len(sim.opinion_history)):
                # Calculate Euclidean distance for each step
                diff = sim.opinion_history[i] - sim.opinion_history[i-1]
                step_length = np.sqrt(np.sum(diff**2))
                total_length += step_length
            
            # Normalize to agent count
            return total_length / len(sim.opinions)
            
        except Exception as e:
            warnings.warn(f"Error calculating trajectory length: {e}")
            return 0.0
    
    def _calculate_oscillation_frequency(self, sim: Simulation) -> float:
        """Calculate oscillation frequency"""
        try:
            if not hasattr(sim, 'opinion_history') or len(sim.opinion_history) < 3:
                return 0.0
            
            oscillations = 0
            for agent_idx in range(len(sim.opinions)):
                # Calculate direction change count for each agent
                agent_history = [step[agent_idx] for step in sim.opinion_history]
                direction_changes = 0
                
                for i in range(2, len(agent_history)):
                    # Check if direction changed
                    prev_diff = agent_history[i-1] - agent_history[i-2]
                    curr_diff = agent_history[i] - agent_history[i-1]
                    
                    if prev_diff * curr_diff < 0:  # Opposite signs
                        direction_changes += 1
                
                oscillations += direction_changes
            
            # Normalize
            total_steps = len(sim.opinion_history) - 2
            if total_steps > 0:
                return oscillations / (len(sim.opinions) * total_steps)
            else:
                return 0.0
                
        except Exception as e:
            warnings.warn(f"Error calculating oscillation frequency: {e}")
            return 0.0
    
    def _calculate_group_divergence(self, sim: Simulation) -> float:
        """Calculate group divergence"""
        try:
            # Calculate opinion differences between different identity groups
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
            
            # Calculate average distance between groups
            total_divergence = 0.0
            comparisons = 0
            
            identities = list(group_opinions.keys())
            for i in range(len(identities)):
                for j in range(i+1, len(identities)):
                    opinions_i = group_opinions[identities[i]]
                    opinions_j = group_opinions[identities[j]]
                    
                    # Calculate average opinion difference between two groups
                    mean_i = np.mean(opinions_i)
                    mean_j = np.mean(opinions_j)
                    divergence = abs(mean_i - mean_j)
                    
                    total_divergence += divergence
                    comparisons += 1
            
            return total_divergence / comparisons if comparisons > 0 else 0.0
            
        except Exception as e:
            warnings.warn(f"Error calculating group divergence: {e}")
            return 0.0
    
    def _calculate_identity_variance_ratio(self, opinions: np.ndarray, identities: np.ndarray) -> float:
        """Calculate identity within/between variance ratio"""
        try:
            unique_identities = np.unique(identities)
            if len(unique_identities) < 2:
                return 0.0
            
            # Calculate within-group variance
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
            
            # Calculate between-group variance
            group_means = []
            for identity in unique_identities:
                mask = identities == identity
                if np.sum(mask) > 0:
                    group_means.append(np.mean(opinions[mask]))
            
            between_group_var = np.var(group_means) if len(group_means) > 1 else 0.0
            
            # Calculate variance ratio
            if within_group_var > 0:
                return between_group_var / within_group_var
            else:
                return 0.0
                
        except Exception as e:
            warnings.warn(f"Error calculating identity variance ratio: {e}")
            return 0.0
    
    def _calculate_cross_identity_correlation(self, opinions: np.ndarray, identities: np.ndarray) -> float:
        """Calculate cross-identity correlation"""
        try:
            unique_identities = np.unique(identities)
            if len(unique_identities) != 2:
                return 0.0
            
            # Get opinions of two identity groups
            group1_mask = identities == unique_identities[0]
            group2_mask = identities == unique_identities[1]
            
            group1_opinions = opinions[group1_mask]
            group2_opinions = opinions[group2_mask]
            
            # If group sizes are different, use the smaller size for comparison
            min_size = min(len(group1_opinions), len(group2_opinions))
            if min_size < 2:
                return 0.0
            
            group1_sample = group1_opinions[:min_size]
            group2_sample = group2_opinions[:min_size]
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(group1_sample, group2_sample)[0, 1]
            
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            warnings.warn(f"Error calculating cross-identity correlation: {e}")
            return 0.0
    
    def get_default_metrics(self) -> Dict[str, float]:
        """Return default metric values (for error cases)"""
        return {name: 0.0 for name in self.metric_names}
    
    def validate_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Validate and clean metric values"""
        validated = {}
        
        for name in self.metric_names:
            if name in metrics:
                value = metrics[name]
                # Check if it's a valid numeric value
                if np.isnan(value) or np.isinf(value):
                    validated[name] = 0.0
                else:
                    validated[name] = float(value)
            else:
                validated[name] = 0.0
        
        return validated
    
    def get_metric_descriptions(self) -> Dict[str, str]:
        """Return metric descriptions"""
        return {
            'polarization_index': 'Koudenburg polarization index, measures overall system polarization level',
            'opinion_variance': 'Opinion variance, reflects opinion dispersion degree',
            'extreme_ratio': 'Extreme opinion ratio, proportion of agents with |opinion| > 0.8',
            'identity_polarization': 'Inter-identity polarization difference, variance of mean opinions across different identity groups',
            'mean_abs_opinion': 'Mean absolute opinion, system opinion intensity',
            'final_stability': 'Final stability, coefficient of variation in final stage',
            'trajectory_length': 'Opinion trajectory length, cumulative distance of opinion changes',
            'oscillation_frequency': 'Oscillation frequency, frequency of opinion direction changes',
            'group_divergence': 'Group divergence, opinion differences between different identity groups',
            'identity_variance_ratio': 'Identity variance ratio, ratio of between-group to within-group variance',
            'cross_identity_correlation': 'Cross-identity correlation, correlation coefficient of opinions between different identity groups',
            'variance_per_identity_1': 'Identity group 1 variance, opinion variance within identity=1 group',
            'variance_per_identity_neg1': 'Identity group -1 variance, opinion variance within identity=-1 group',
            'variance_per_identity_mean': 'Average identity group variance, mean of variances from both identity groups'
        } 