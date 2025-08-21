"""
Sensitivity analysis result visualization module
Provides various chart types to display Sobol sensitivity analysis results
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
    warnings.warn("Complete visualization functionality requires matplotlib")

# Try to set Chinese fonts, use default settings if failed
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # Support Chinese display
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass  # If setting fails, use default fonts


class SensitivityVisualizer:
    """Sensitivity analysis visualizer"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = {
            'alpha': '#FF6B6B',      # Red - self activation
            'beta': '#4ECDC4',       # Cyan - social influence  
            'gamma': '#45B7D1',      # Blue - moralization influence
            'cohesion_factor': '#96CEB4'  # Green - cohesion factor
        }
        
        # Set chart style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def plot_sensitivity_comparison(self, sensitivity_indices: Dict[str, Dict], 
                                  output_name: str, save_path: str = None) -> plt.Figure:
        """Draw sensitivity comparison chart for single output metric"""
        if output_name not in sensitivity_indices:
            raise ValueError(f"Output metric {output_name} does not exist")
        
        indices = sensitivity_indices[output_name]
        param_names = ['alpha', 'beta', 'gamma', 'cohesion_factor']
        
        # Prepare data
        s1_values = indices['S1']
        st_values = indices['ST']
        s1_conf = indices['S1_conf']
        st_conf = indices['ST_conf']
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # First-order sensitivity index
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
        
        # Total sensitivity index
        bars2 = ax2.bar(x_pos, st_values, yerr=st_conf,
                       color=[self.colors[name] for name in param_names],
                       alpha=0.7, capsize=5)
        ax2.set_xlabel('Parameters')
        ax2.set_ylabel('Total sensitivity (ST)')
        ax2.set_title(f'{output_name} - Total sensitivity')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(['α', 'β', 'γ', 'cohesion_factor'])
        ax2.grid(axis='y', alpha=0.3)
        
        # Add numerical labels
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
        """Draw sensitivity heatmap"""
        param_names = ['alpha', 'beta', 'gamma', 'cohesion_factor']
        output_names = list(sensitivity_indices.keys())
        
        # Prepare data matrix
        data_matrix = []
        for output_name in output_names:
            if metric_type in sensitivity_indices[output_name]:
                row = sensitivity_indices[output_name][metric_type]
                data_matrix.append(row)
            else:
                data_matrix.append([0] * len(param_names))
        
        data_matrix = np.array(data_matrix)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.figsize)
        
        im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')
        
        # Set axes
        ax.set_xticks(np.arange(len(param_names)))
        ax.set_yticks(np.arange(len(output_names)))
        ax.set_xticklabels(['α', 'β', 'γ', 'cohesion_factor'])
        ax.set_yticklabels(output_names)
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add numerical labels
        for i in range(len(output_names)):
            for j in range(len(param_names)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(f'{metric_type} Sensitivity Index')
        
        ax.set_title(f'{metric_type} Sensitivity Heatmap')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_interaction_effects(self, sensitivity_indices: Dict[str, Dict], 
                               save_path: str = None) -> plt.Figure:
        """Draw interaction effects analysis chart"""
        param_names = ['alpha', 'beta', 'gamma', 'cohesion_factor']
        output_names = list(sensitivity_indices.keys())
        
        # Calculate interaction effect strength (ST - S1)
        interaction_data = []
        for output_name in output_names:
            indices = sensitivity_indices[output_name]
            interactions = np.array(indices['ST']) - np.array(indices['S1'])
            interaction_data.append(interactions)
        
        interaction_matrix = np.array(interaction_data)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Interaction effect heatmap
        im1 = ax1.imshow(interaction_matrix, cmap='RdBu_r', aspect='auto')
        ax1.set_xticks(np.arange(len(param_names)))
        ax1.set_yticks(np.arange(len(output_names)))
        ax1.set_xticklabels(['α', 'β', 'γ', 'cohesion_factor'])
        ax1.set_yticklabels(output_names)
        ax1.set_title('Interaction Effect Strength (ST - S1)')
        
        # Add numerical labels
        for i in range(len(output_names)):
            for j in range(len(param_names)):
                text = ax1.text(j, i, f'{interaction_matrix[i, j]:.3f}',
                               ha="center", va="center", 
                               color="white" if abs(interaction_matrix[i, j]) > 0.1 else "black",
                               fontsize=8)
        
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # 2. Average interaction effect bar chart
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
        
        # Add numerical labels
        for i, val in enumerate(mean_interactions):
            ax2.text(i, val + 0.001, f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_parameter_ranking(self, sensitivity_indices: Dict[str, Dict], 
                             metric_type: str = 'ST', save_path: str = None) -> plt.Figure:
        """Draw parameter importance ranking chart"""
        param_names = ['alpha', 'beta', 'gamma', 'cohesion_factor']
        output_names = list(sensitivity_indices.keys())
        
        # Calculate average sensitivity of each parameter across all output metrics
        param_importance = {}
        for i, param in enumerate(param_names):
            importances = []
            for output_name in output_names:
                if metric_type in sensitivity_indices[output_name]:
                    importances.append(sensitivity_indices[output_name][metric_type][i])
            param_importance[param] = np.mean(importances) if importances else 0.0
        
        # Sort by importance
        sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Create figure
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
        
        # Add numerical labels
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
        """Create comprehensive analysis report"""
        os.makedirs(output_dir, exist_ok=True)
        
        plot_files = {}
        
        try:
            # 1. Create sensitivity comparison charts for each output metric
            for output_name in sensitivity_indices.keys():
                filename = f"{output_name}_sensitivity.png"
                filepath = os.path.join(output_dir, filename)
                fig = self.plot_sensitivity_comparison(sensitivity_indices, output_name, filepath)
                plot_files[f"{output_name}_comparison"] = filepath
                plt.close(fig)
            
            # 2. Create heatmaps
            for metric_type in ['S1', 'ST']:
                filename = f"heatmap_{metric_type}.png"
                filepath = os.path.join(output_dir, filename)
                fig = self.plot_sensitivity_heatmap(sensitivity_indices, metric_type, filepath)
                plot_files[f"heatmap_{metric_type}"] = filepath
                plt.close(fig)
            
            # 3. Create interaction effects chart
            filename = "interaction_effects.png"
            filepath = os.path.join(output_dir, filename)
            fig = self.plot_interaction_effects(sensitivity_indices, filepath)
            plot_files["interaction_effects"] = filepath
            plt.close(fig)
            
            # 4. Create parameter ranking charts
            for metric_type in ['S1', 'ST']:
                filename = f"parameter_ranking_{metric_type}.png"
                filepath = os.path.join(output_dir, filename)
                fig = self.plot_parameter_ranking(sensitivity_indices, metric_type, filepath)
                plot_files[f"ranking_{metric_type}"] = filepath
                plt.close(fig)
            
            print(f"All charts have been saved to: {output_dir}")
            
        except Exception as e:
            warnings.warn(f"Error creating charts: {e}")
        
        return plot_files


def create_sensitivity_report_example():
    """Create example sensitivity analysis visualization report"""
    # Simulate sensitivity analysis results
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
    
    # Create visualizer
    visualizer = SensitivityVisualizer()
    
    # Create comprehensive report
    plot_files = visualizer.create_comprehensive_report(sensitivity_indices)
    
    return visualizer, sensitivity_indices


if __name__ == "__main__":
    # Run example
    create_sensitivity_report_example() 