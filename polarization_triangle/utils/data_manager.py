#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data management utility module
Provides utility functions for data saving and loading
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


def save_simulation_data(sim: Any, output_dir: str, prefix: str = 'sim_data') -> Dict[str, str]:
    """
    Save simulation data to files for subsequent statistical analysis
    
    Parameters:
    sim -- Simulation object
    output_dir -- Output directory path
    prefix -- Filename prefix
    
    Returns:
    Dictionary containing all saved file paths
    """
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save trajectory data
    trajectory_data = {
        'step': [],
        'agent_id': [],
        'opinion': [],
        'identity': [],
        'morality': [],
        'self_activation': [],
        'social_influence': []
    }
    
    # Get full history
    activation_history = sim.get_activation_history()
    
    # If history data exists
    if sim.self_activation_history:
        # Add data for each step and each agent
        for step in range(len(sim.self_activation_history)):
            for agent_id in range(sim.num_agents):
                trajectory_data['step'].append(step)
                trajectory_data['agent_id'].append(agent_id)
                # For opinion, get from trajectory; if not available, use current value
                if hasattr(sim, 'opinion_trajectory') and step < len(sim.opinion_trajectory):
                    trajectory_data['opinion'].append(sim.opinion_trajectory[step][agent_id])
                else:
                    trajectory_data['opinion'].append(sim.opinions[agent_id])
                
                trajectory_data['identity'].append(sim.identities[agent_id])
                trajectory_data['morality'].append(sim.morals[agent_id])
                trajectory_data['self_activation'].append(activation_history['self_activation_history'][step][agent_id])
                trajectory_data['social_influence'].append(activation_history['social_influence_history'][step][agent_id])
    
    # Convert data to DataFrame and save as CSV
    df = pd.DataFrame(trajectory_data)
    trajectory_csv_path = os.path.join(output_dir, f"{prefix}_trajectory.csv")
    df.to_csv(trajectory_csv_path, index=False)
    
    # Save final state data
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
    
    # Save network structure
    network_data = []
    for i in range(sim.num_agents):
        for j in range(i+1, sim.num_agents):  # Only save the upper triangular matrix to avoid duplication
            if sim.adj_matrix[i, j] > 0:
                network_data.append({
                    'source': i,
                    'target': j,
                    'weight': sim.adj_matrix[i, j]
                })
    
    df_network = pd.DataFrame(network_data)
    network_csv_path = os.path.join(output_dir, f"{prefix}_network.csv")
    df_network.to_csv(network_csv_path, index=False)
    
    # Save simulation configuration
    config_dict = vars(sim.config)
    config_data = []
    for key, value in config_dict.items():
        # Skip complex objects that cannot be serialized
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
    Experiment Data Manager
    
    Specialized for zealot_morality_analysis experiment data storage and retrieval,
    optimizing the balance between storage space and loading speed.
    
    Features:
    - Uses Parquet format, balancing compression ratio and read speed
    - Supports batch management and data accumulation
    - Reserved interface for parallel computing
    - Supports future variance per identity calculation needs
    """
    
    def __init__(self, base_dir: str = "results/zealot_morality_analysis"):
        """
        Initialize data manager
        
        Args:
            base_dir: Base storage directory
        """
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "experiment_data"
        self.metadata_dir = self.base_dir / "metadata"
        
        # Create necessary directory structure
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Data file paths
        self.zealot_numbers_file = self.data_dir / "zealot_numbers_data.parquet"
        self.morality_ratios_file = self.data_dir / "morality_ratios_data.parquet"
        
        # Metadata file paths
        self.batch_metadata_file = self.metadata_dir / "batch_metadata.json"
        self.experiment_config_file = self.metadata_dir / "experiment_config.json"
    
    def save_batch_results(self, 
                          plot_type: str,
                          batch_data: Dict[str, Any],
                          batch_metadata: Dict[str, Any]) -> None:
        """
        Save batch experiment results
        
        Args:
            plot_type: 'zealot_numbers' or 'morality_ratios'
            batch_data: Batch data {combination_label: {x_values: [], results: {}}}
            batch_metadata: Batch metadata
        """
        # Convert nested result data to a flat DataFrame format
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
        
        # Create DataFrame
        new_df = pd.DataFrame(rows)
        
        # Determine the target file
        target_file = self.zealot_numbers_file if plot_type == 'zealot_numbers' else self.morality_ratios_file
        
        # Append to or create the data file
        if target_file.exists():
            # Read existing data and merge
            existing_df = pd.read_parquet(target_file)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        
        # Save as Parquet format (with automatic compression)
        combined_df.to_parquet(target_file, compression='snappy', index=False)
        
        # Update batch metadata
        self._update_batch_metadata(batch_metadata)
        
        print(f"Saved batch data: {len(rows)} records to {target_file.name}")
    
    def load_experiment_data(self, plot_type: str) -> Optional[pd.DataFrame]:
        """
        Load experiment data
        
        Args:
            plot_type: 'zealot_numbers' or 'morality_ratios'
        
        Returns:
            DataFrame or None
        """
        target_file = self.zealot_numbers_file if plot_type == 'zealot_numbers' else self.morality_ratios_file
        
        if not target_file.exists():
            return None
        
        df = pd.read_parquet(target_file)
        print(f"Loaded {len(df)} records from {target_file.name}")
        return df
    
    def get_experiment_summary(self, plot_type: str) -> Dict[str, Any]:
        """
        Get summary statistics for experiment data
        
        Args:
            plot_type: 'zealot_numbers' or 'morality_ratios'
        
        Returns:
            Dictionary of summary statistics
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
        
        # Calculate the total number of runs for each combination
        for combo in summary['combinations']:
            combo_data = df[df['combination'] == combo]
            if not combo_data.empty:
                # Calculate total runs = total records / (number of x_values * number of metrics)
                unique_x_values = len(combo_data['x_value'].unique())
                unique_metrics = len(combo_data['metric'].unique())
                total_runs = len(combo_data) // (unique_x_values * unique_metrics) if unique_x_values > 0 and unique_metrics > 0 else 0
                summary['total_runs_per_combination'][combo] = total_runs
        
        return summary
    
    def convert_to_plotting_format(self, plot_type: str) -> Tuple[Dict[str, Dict[str, List[List[float]]]], List[float], Dict[str, int]]:
        """
        Convert stored data to plotting format
        
        Args:
            plot_type: 'zealot_numbers' or 'morality_ratios'
        
        Returns:
            (all_results, x_values, total_runs_per_combination)
        """
        df = self.load_experiment_data(plot_type)
        if df is None or df.empty:
            return {}, [], {}
        
        # Get all unique values
        combinations = sorted(df['combination'].unique())
        x_values = sorted(df['x_value'].unique())
        metrics = sorted(df['metric'].unique())
        
        # Initialize result structure
        all_results = {}
        total_runs_per_combination = {}
        
        for combination in combinations:
            combo_data = df[df['combination'] == combination]
            
            # Calculate total number of runs
            unique_x_values = len(combo_data['x_value'].unique())
            unique_metrics = len(combo_data['metric'].unique())
            total_runs = len(combo_data) // (unique_x_values * unique_metrics) if unique_x_values > 0 and unique_metrics > 0 else 0
            total_runs_per_combination[combination] = total_runs
            
            # Organize data into plotting format
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
        Update batch metadata
        
        Args:
            batch_metadata: Batch metadata
        """
        # Read existing metadata
        if self.batch_metadata_file.exists():
            with open(self.batch_metadata_file, 'r', encoding='utf-8') as f:
                all_metadata = json.load(f)
        else:
            all_metadata = {'batches': []}
        
        # Add new batch
        all_metadata['batches'].append(batch_metadata)
        
        # Save metadata
        with open(self.batch_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    
    def get_batch_metadata(self) -> Dict[str, Any]:
        """
        Get all batch metadata
        
        Returns:
            Dictionary of batch metadata
        """
        if not self.batch_metadata_file.exists():
            return {'batches': []}
        
        with open(self.batch_metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_experiment_config(self, config: Dict[str, Any]) -> None:
        """
        Save experiment configuration
        
        Args:
            config: Experiment configuration dictionary
        """
        config['saved_at'] = datetime.now().isoformat()
        
        with open(self.experiment_config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def export_summary_report(self) -> str:
        """
        Export experiment summary report
        
        Returns:
            Summary report string
        """
        zealot_summary = self.get_experiment_summary('zealot_numbers')
        morality_summary = self.get_experiment_summary('morality_ratios')
        batch_metadata = self.get_batch_metadata()
        
        report = []
        report.append("=" * 60)
        report.append("Experiment Data Summary Report")
        report.append("=" * 60)
        
        report.append(f"\nZealot Numbers Experiment:")
        report.append(f"   Total Records: {zealot_summary['total_records']}")
        report.append(f"   Number of Combinations: {len(zealot_summary['combinations'])}")
        report.append(f"   Number of Batches: {len(zealot_summary['batches'])}")
        
        report.append(f"\nMorality Ratios Experiment:")
        report.append(f"   Total Records: {morality_summary['total_records']}")
        report.append(f"   Number of Combinations: {len(morality_summary['combinations'])}")
        report.append(f"   Number of Batches: {len(morality_summary['batches'])}")
        
        report.append(f"\nBatch History: {len(batch_metadata.get('batches', []))} batches")
        
        # Storage space information
        zealot_size = self.zealot_numbers_file.stat().st_size if self.zealot_numbers_file.exists() else 0
        morality_size = self.morality_ratios_file.stat().st_size if self.morality_ratios_file.exists() else 0
        total_size = zealot_size + morality_size
        
        report.append(f"\nStorage Space:")
        report.append(f"   Zealot Numbers: {zealot_size / 1024:.1f} KB")
        report.append(f"   Morality Ratios: {morality_size / 1024:.1f} KB")
        report.append(f"   Total: {total_size / 1024:.1f} KB")
        
        return "\n".join(report)
