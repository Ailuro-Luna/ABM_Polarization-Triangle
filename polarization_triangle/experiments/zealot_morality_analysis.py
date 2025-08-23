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
    # Method 1: Standard Deviation Bands: Shows mean ± standard deviation
    # error_band_type = 'std'  # Use standard deviation method
    # Method 2: Percentile Bands: Shows 25th to 75th percentile range
    # error_band_type = 'percentile'  # Use percentile method
    # Method 3: Confidence Interval Bands: Shows 99% confidence interval
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
        x_values: Original x-value array, e.g. [0,1,2,3,4,5,6,7,8,9,10,...]
        y_values: Original y-value array
        target_step: Target step size, e.g. 2 means changing from [0,1,2,3,4,5,...] to [0,2,4,6,8,10,...]
        smooth_window: Smoothing window size
    
    Returns:
        new_x_values, new_y_values: Resampled and smoothed data
    """
    # Ensure inputs are numpy arrays
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    
    # Remove NaN values
    valid_mask = ~np.isnan(y_values)
    x_clean = x_values[valid_mask]
    y_clean = y_values[valid_mask]
    
    if len(x_clean) < 2:
        return x_values, y_values
    
    # 1. First perform local smoothing (reduce noise)
    if smooth_window >= 3 and len(y_clean) >= smooth_window:
        # Use moving average for initial smoothing
        kernel = np.ones(smooth_window) / smooth_window
        y_smoothed = np.convolve(y_clean, kernel, mode='same')
        
        # Handle boundary effects
        half_window = smooth_window // 2
        y_smoothed[:half_window] = y_clean[:half_window]
        y_smoothed[-half_window:] = y_clean[-half_window:]
    else:
        y_smoothed = y_clean
    
    # 2. Create target x values (resampling)
    x_min, x_max = x_clean[0], x_clean[-1]
    new_x_values = np.arange(x_min, x_max + target_step, target_step)
    
    # 3. For each new x value, use nearby data points for weighted average
    new_y_values = []
    
    for new_x in new_x_values:
        # Find nearby points for weighted average
        distances = np.abs(x_clean - new_x)
        
        # Use Gaussian weights, closer distance means higher weight
        weights = np.exp(-distances**2 / (2 * (target_step/2)**2))
        
        # Only consider points within target_step distance
        nearby_mask = distances <= target_step
        if np.sum(nearby_mask) > 0:
            nearby_weights = weights[nearby_mask]
            nearby_y = y_smoothed[nearby_mask]
            
            # Weighted average
            weighted_y = np.average(nearby_y, weights=nearby_weights)
            new_y_values.append(weighted_y)
        else:
            # If no nearby points, use the closest point
            closest_idx = np.argmin(distances)
            new_y_values.append(y_smoothed[closest_idx])
    
    return new_x_values, np.array(new_y_values)


def apply_final_smooth(y_values, method='savgol', window=5):
    """
    Apply final smoothing to resampled data
    
    Args:
        y_values: Y values after resampling
        method: Smoothing method ('savgol', 'moving_avg', 'none')
        window: Smoothing window
    
    Returns:
        Smoothed y values
    """
    if len(y_values) < window or method == 'none':
        return y_values
    
    if method == 'moving_avg':
        # Moving average
        kernel = np.ones(window) / window
        smoothed = np.convolve(y_values, kernel, mode='same')
    elif method == 'savgol':
        # Savitzky-Golay filtering (requires scipy)
        try:
            from scipy.signal import savgol_filter
            # Ensure window is odd and smaller than data length
            actual_window = min(window if window % 2 == 1 else window-1, len(y_values)-1)
            if actual_window >= 3:
                smoothed = savgol_filter(y_values, actual_window, 2)
            else:
                smoothed = y_values
        except ImportError:
            # If scipy is not available, use moving average
            kernel = np.ones(window) / window
            smoothed = np.convolve(y_values, kernel, mode='same')
    else:
        smoothed = y_values
    
    return smoothed


# =====================================
# Utility functions
# =====================================

def format_duration(duration: float) -> str:
    """
    Format time display
    
    Args:
    duration: Duration in seconds
    
    Returns:
    str: Formatted time string
    """
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"


# Note: save_batch_info function has been replaced by ExperimentDataManager's batch metadata functionality


# =====================================
# Parallel computing support functions
# =====================================

def run_single_simulation_task(task_params):
    """
    Wrapper function for single simulation task, used for multi-process parallel computation
    
    Args:
        task_params: Tuple containing task parameters
            (plot_type, combination, x_val, run_idx, steps, process_id, batch_seed)
    
    Returns:
        tuple: (x_val, run_idx, results_dict, success, error_msg)
    """
    try:
        plot_type, combination, x_val, run_idx, steps, process_id, batch_seed = task_params
        
        # Set process-specific random seed, add batch identifier to ensure different batches produce different results
        np.random.seed((int(x_val * 1000) + run_idx + process_id + batch_seed) % (2**32))
        
        # Build configuration
        base_config = copy.deepcopy(high_polarization_config)
        
        # Set fixed parameters
        if plot_type == 'zealot_numbers':
            base_config.morality_rate = combination['morality_rate']
            base_config.zealot_identity_allocation = combination['zealot_identity_allocation']
            base_config.cluster_identity = combination['cluster_identity']
            base_config.enable_zealots = True
            base_config.steps = combination['steps']
            # Set parameters corresponding to current x value
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
            # Set parameters corresponding to current x value
            base_config.morality_rate = x_val / 100.0  # Convert to 0-1 range
        
        # Run single simulation
        results = run_single_simulation(base_config, steps)
        
        return (x_val, run_idx, results, True, None)
        
    except Exception as e:
        error_msg = f"Process {process_id}: Simulation failed for x={x_val}, run={run_idx}: {str(e)}"
        return (x_val, run_idx, None, False, error_msg)


# =====================================
# Core experiment logic functions
# =====================================

def create_config_combinations():
    """
    Create experimental parameter combination configurations
    
    This function generates all parameter combinations for two types of experiments:
    
    1. zealot_numbers experiment: Test the impact of different zealot numbers on the system
       - Variable: zealot count (x-axis)
       - Fixed: zealot identity allocation=True, identity distribution=random
       - Compare: zealot distribution modes (random/clustered) × morality ratios (0.0/0.3) = 4 combinations
    
    2. morality_ratios experiment: Test the impact of different morality ratios on the system
       - Variable: morality ratio (x-axis)
       - Fixed: zealot count=20
       - Compare: zealot modes (random/clustered/none) × zealot identity alignment (True/False) × 
                 identity distribution (random/clustered) = 10 combinations
    
    Returns:
        dict: Dictionary containing configurations for both experiment types
            - 'zealot_numbers': 4 parameter combinations for zealot count experiments
            - 'morality_ratios': 10 parameter combinations for morality ratio experiments
    """
    # Base configuration: use high polarization config as template
    base_config = copy.deepcopy(high_polarization_config)
    base_config.steps = 300  # Each simulation runs 300 steps
    
    # Initialize parameter combination containers for both experiment types
    combinations = {
        'zealot_numbers': [],   # Experiment 1: parameter combinations with zealot count on x-axis
        'morality_ratios': []   # Experiment 2: parameter combinations with morality ratio on x-axis
    }
    
    # ===== Experiment 1: Zealot count sweep experiment =====
    # Compare the impact of zealot distribution modes and morality ratios on the system
    # Fixed parameters: zealot identity allocation=True, identity distribution=random
    zealot_clustering_options = ['random', 'clustered']  # Zealot distribution modes: random distribution vs clustered distribution
    morality_ratios_for_zealot_plot = [0.0, 0.3]  # Two morality levels: no moral constraint vs moderate moral constraint
    
    for clustering in zealot_clustering_options:
        for morality_ratio in morality_ratios_for_zealot_plot:
            combo = {
                'zealot_mode': clustering,                    # Zealot distribution mode
                'morality_rate': morality_ratio,              # Morality constraint strength
                'zealot_identity_allocation': True,           # Zealot allocation by identity (fixed)
                'cluster_identity': False,                    # Random identity distribution (fixed)
                'label': f'{clustering.capitalize()} Zealots, Morality={morality_ratio}',
                'steps': base_config.steps
            }
            combinations['zealot_numbers'].append(combo)
    
    # ===== Experiment 2: Morality ratio sweep experiment =====
    # Compare interactive effects of three key factors: zealot distribution, zealot identity alignment, identity distribution
    # Fixed parameters: zealot count=20 (moderate level)
    zealot_modes = ['random', 'clustered', 'none']     # Zealot modes: random/clustered/no zealots
    zealot_identity_alignments = [True, False]         # Whether zealots are allocated by identity
    identity_distributions = [False, True]             # Identity distribution: random vs clustered
    
    # Fix zealot count to 20, which is a moderate level that won't overly impact the system while still allowing observable effects
    fixed_zealot_count = 20
    
    for zealot_mode in zealot_modes:
        if zealot_mode == 'none':
            # No zealot case: only need to distinguish identity distribution methods, zealot-related parameters are meaningless
            for identity_dist in identity_distributions:
                combo = {
                    'zealot_count': 0,                           # No zealots
                    'zealot_mode': zealot_mode,                  # Marked as 'none'
                    'zealot_identity_allocation': True,          # Default value (doesn't affect results)
                    'cluster_identity': identity_dist,           # Identity distribution method
                    'label': f'{zealot_mode.capitalize()}, ID-cluster={identity_dist}',
                    'steps': base_config.steps
                }
                combinations['morality_ratios'].append(combo)
        else:
            # With zealots case: need to consider the combined effects of zealot identity alignment and identity distribution methods
            for zealot_identity in zealot_identity_alignments:
                for identity_dist in identity_distributions:
                    combo = {
                        'zealot_count': fixed_zealot_count,          # Fixed zealot count
                        'zealot_mode': zealot_mode,                  # Zealot distribution mode
                        'zealot_identity_allocation': zealot_identity,  # Zealot identity alignment method
                        'cluster_identity': identity_dist,           # Identity distribution method
                        'label': f'{zealot_mode.capitalize()} Zealots, ID-align={zealot_identity}, ID-cluster={identity_dist}',
                        'steps': base_config.steps
                    }
                    combinations['morality_ratios'].append(combo)
    
    return combinations


def run_single_simulation(config: SimulationConfig, steps: int = 500) -> Dict[str, float]:
    """
    Run a single simulation and obtain statistical metrics from the final state
    
    This function creates a simulation instance, runs it for specified steps, then calculates six key metrics:
    - Mean Opinion: Average opinion value of non-zealot agents in the system
    - Variance: Variance of opinion distribution, measuring opinion divergence
    - Identity Opinion Difference: Average opinion difference between different identity groups
    - Polarization Index: Polarization index, measuring system polarization level
    - Variance per Identity: Opinion variance within each identity group (calculated separately for both identity groups)
    
    Args:
        config (SimulationConfig): Simulation configuration object containing network, agent, zealot and other parameters
        steps (int, optional): Number of steps to run the simulation. Defaults to 500.
    
    Returns:
        Dict[str, Any]: Dictionary containing statistical metrics
            - 'mean_opinion': Average opinion value (float)
            - 'variance': Opinion variance (float)
            - 'identity_opinion_difference': Identity opinion difference (float)
            - 'polarization_index': Polarization index (float)
            - 'variance_per_identity': Variance for each identity group (dict)
                - 'identity_1': Variance for identity=1 group
                - 'identity_-1': Variance for identity=-1 group
    
    Raises:
        Exception: Raises exception when errors occur during simulation
    """
    # Create simulation instance
    sim = Simulation(config)
    
    # Run simulation step by step to stable state
    for _ in range(steps):
        sim.step()
    
    # Calculate various statistical metrics from final state
    mean_stats = calculate_mean_opinion(sim, exclude_zealots=True)
    variance_stats = calculate_variance_metrics(sim, exclude_zealots=True)
    identity_stats = calculate_identity_statistics(sim, exclude_zealots=True)
    polarization = get_polarization_index(sim)
    
    # Calculate identity opinion difference (opinion difference between identities)
    identity_opinion_difference = 0.0
    if 'identity_difference' in identity_stats:
        identity_opinion_difference = identity_stats['identity_difference']['abs_mean_opinion_difference']
    else:
        # Theoretically should not reach here under normal conditions (when zealot count is small enough)
        print("Warning: identity_difference not found, this should not happen under normal conditions")
        identity_opinion_difference = 0.0
    
    # Calculate variance per identity (variance within each identity group)
    variance_per_identity = {'identity_1': 0.0, 'identity_-1': 0.0}
    
    # Get opinions and identities of non-zealot nodes
    # Create zealot mask: if an agent's ID is in zealot_ids, then True
    zealot_mask = np.zeros(sim.num_agents, dtype=bool)
    if sim.enable_zealots and sim.zealot_ids:
        zealot_mask[sim.zealot_ids] = True
    
    non_zealot_mask = ~zealot_mask
    non_zealot_opinions = sim.opinions[non_zealot_mask]
    non_zealot_identities = sim.identities[non_zealot_mask]
    
    # Calculate variance for each identity group separately
    for identity_val in [1, -1]:
        identity_mask = non_zealot_identities == identity_val
        if np.sum(identity_mask) > 1:  # Need at least 2 nodes to calculate variance
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
    Perform parameter sweep experiment on specific parameter combination
    
    This function runs multiple simulations at each x-axis value point for a given parameter combination to collect statistical data.
    This is the core execution function of the experiment, supporting two types of sweeps:
    - zealot_numbers: Fix morality ratio, sweep different zealot counts
    - morality_ratios: Fix zealot count, sweep different morality ratios
    
    Args:
        plot_type (str): Experiment type
            - 'zealot_numbers': Experiment with zealot count on x-axis
            - 'morality_ratios': Experiment with morality ratio on x-axis
        combination (Dict[str, Any]): Parameter combination dictionary containing:
            - zealot_mode: Zealot distribution mode ('random', 'clustered', 'none')
            - morality_rate: Morality ratio (0.0-1.0)
            - zealot_identity_allocation: Whether to allocate zealots by identity
            - cluster_identity: Whether to cluster identity distribution
            - label: Combination label
            - steps: Number of simulation steps
        x_values (List[float]): List of x-axis sweep values, e.g. [0, 1, 2, ...]
        num_runs (int, optional): Number of repeated runs per x value point. Defaults to 5.
        num_processes (int, optional): Number of parallel processes, 1 means serial execution. Defaults to 1.
        batch_seed (int, optional): Batch seed to ensure different batches produce different results. Defaults to 0.
    
    Returns:
        Dict[str, List[List[float]]]: Nested result data structure
            Format: {metric_name: [x1_runs, x2_runs, ...]}
            where x1_runs = [run1_value, run2_value, ...]
            
            Included metrics:
            - 'mean_opinion': Multiple run results for average opinion values
            - 'variance': Multiple run results for opinion variance
            - 'identity_opinion_difference': Multiple run results for identity opinion differences
            - 'polarization_index': Multiple run results for polarization index
            - 'variance_per_identity_1': Multiple run results for within-group variance of identity=1
            - 'variance_per_identity_-1': Multiple run results for within-group variance of identity=-1
    """
    # Choose serial or parallel execution
    if num_processes == 1:
        return run_parameter_sweep_serial(plot_type, combination, x_values, num_runs, batch_seed)
    else:
        return run_parameter_sweep_parallel(plot_type, combination, x_values, num_runs, num_processes, batch_seed)


def run_parameter_sweep_serial(plot_type: str, combination: Dict[str, Any], 
                              x_values: List[float], num_runs: int = 5, batch_seed: int = 0) -> Dict[str, List[List[float]]]:
    """
    Serial version of parameter sweep (original logic)
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
    
    # Set fixed parameters
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
    
    # Run multiple times for each x value
    for x_val in tqdm(x_values, desc=f"Running {combination['label']}"):
        runs_data = {
            'mean_opinion': [],
            'variance': [],
            'identity_opinion_difference': [],
            'polarization_index': [],
            'variance_per_identity_1': [],
            'variance_per_identity_-1': []
        }
        
        # Set parameters for the current x value
        current_config = copy.deepcopy(base_config)
        if plot_type == 'zealot_numbers':
            current_config.zealot_count = int(x_val)
            current_config.zealot_mode = combination['zealot_mode']
            if x_val == 0:
                current_config.enable_zealots = False
        else:  # morality_ratios
            current_config.morality_rate = x_val / 100.0  # Convert to 0-1 range
        
        # Run multiple simulations
        for run in range(num_runs):
            try:
                # Set random seed, add batch identifier to ensure different batches produce different results
                np.random.seed((int(x_val * 1000) + run + batch_seed) % (2**32))
                
                stats = run_single_simulation(current_config)
                # Process basic metrics
                for metric in ['mean_opinion', 'variance', 'identity_opinion_difference', 'polarization_index']:
                    runs_data[metric].append(stats[metric])
                # Process variance per identity metric
                variance_per_identity = stats['variance_per_identity']
                runs_data['variance_per_identity_1'].append(variance_per_identity['identity_1'])
                runs_data['variance_per_identity_-1'].append(variance_per_identity['identity_-1'])
            except Exception as e:
                print(f"Warning: Simulation failed for x={x_val}, run={run}: {e}")
                # Fill failed runs with NaN
                for metric in runs_data.keys():
                    runs_data[metric].append(np.nan)
        
        # Add all run results for current x value to total results
        for metric in results.keys():
            results[metric].append(runs_data[metric])
    
    return results


def run_parameter_sweep_parallel(plot_type: str, combination: Dict[str, Any], 
                                x_values: List[float], num_runs: int = 5, num_processes: int = 4, 
                                batch_seed: int = 0) -> Dict[str, List[List[float]]]:
    """
    Parallel version of parameter sweep
    """
    print(f"Using {num_processes} processes for parallel computation...")
    
    # Create all tasks
    tasks = []
    for x_val in x_values:
        for run_idx in range(num_runs):
            process_id = len(tasks) % num_processes  # Simple process ID allocation
            task = (plot_type, combination, x_val, run_idx, combination['steps'], process_id, batch_seed)
            tasks.append(task)
    
    print(f"Total tasks: {len(tasks)} (x_values: {len(x_values)}, runs_per_x: {num_runs})")
    
    # Execute parallel computation
    try:
        with multiprocessing.Pool(num_processes) as pool:
            # Use imap to show progress
            results_list = []
            with tqdm(total=len(tasks), desc=f"Running {combination['label']} (parallel)") as pbar:
                for result in pool.imap(run_single_simulation_task, tasks):
                    results_list.append(result)
                    pbar.update(1)
    except Exception as e:
        print(f"Parallel computation failed, falling back to serial mode: {e}")
        return run_parameter_sweep_serial(plot_type, combination, x_values, num_runs, batch_seed)
    
    # Organize results
    return organize_parallel_results(results_list, x_values, num_runs)


def organize_parallel_results(results_list: List[Tuple], x_values: List[float], num_runs: int) -> Dict[str, List[List[float]]]:
    """
    Reorganize parallel computation results into original data structure
    """
    # Initialize result structure
    organized_results = {
        'mean_opinion': [],
        'variance': [],
        'identity_opinion_difference': [],
        'polarization_index': [],
        'variance_per_identity_1': [],
        'variance_per_identity_-1': []
    }
    
    # Count successful and failed tasks
    success_count = 0
    failure_count = 0
    
    # Group and organize results by x_value
    for x_val in x_values:
        runs_data = {
            'mean_opinion': [],
            'variance': [],
            'identity_opinion_difference': [],
            'polarization_index': [],
            'variance_per_identity_1': [],
            'variance_per_identity_-1': []
        }
        
        # Collect all run results for current x_val
        for run_idx in range(num_runs):
            # Find corresponding results in result list
            found_result = None
            for result in results_list:
                result_x_val, result_run_idx, result_data, success, error_msg = result
                if result_x_val == x_val and result_run_idx == run_idx:
                    found_result = result
                    break
            
            if found_result and found_result[3]:  # success = True
                result_data = found_result[2]
                # Process basic metrics
                for metric in ['mean_opinion', 'variance', 'identity_opinion_difference', 'polarization_index']:
                    runs_data[metric].append(result_data[metric])
                # Process variance per identity metric
                variance_per_identity = result_data['variance_per_identity']
                runs_data['variance_per_identity_1'].append(variance_per_identity['identity_1'])
                runs_data['variance_per_identity_-1'].append(variance_per_identity['identity_-1'])
                success_count += 1
            else:
                # Handle failed tasks
                if found_result:
                    print(f"Warning: {found_result[4]}")  # Print error message
                else:
                    print(f"Warning: Missing result for x={x_val}, run={run_idx}")
                
                # Fill failed runs with NaN
                for metric in runs_data.keys():
                    runs_data[metric].append(np.nan)
                failure_count += 1
        
        # Add all run results for current x value to total results
        for metric in organized_results.keys():
            organized_results[metric].append(runs_data[metric])
    
    print(f"Parallel computation completed: {success_count} successful, {failure_count} failed")
    
    return organized_results


# =====================================
# Data management functions (refactored to use ExperimentDataManager)
# =====================================

def save_data_with_manager(data_manager: ExperimentDataManager, 
                          plot_type: str, 
                          x_values: List[float], 
                          all_results: Dict[str, Dict[str, List[List[float]]]], 
                          batch_metadata: Dict[str, Any]) -> None:
    """
    Save experiment data using the new data manager
    
    Args:
        data_manager: Data manager instance
        plot_type: 'zealot_numbers' or 'morality_ratios'
        x_values: x-axis values
        all_results: Result data from all combinations
        batch_metadata: Batch metadata
    """
    # Convert data format to adapt to new data manager
    batch_data = {}
    
    for combination_label, results in all_results.items():
        batch_data[combination_label] = {
            'x_values': x_values,
            'results': results
        }
    
    # Save data using data manager
    data_manager.save_batch_results(plot_type, batch_data, batch_metadata)


# =====================================
# Plotting related functions
# =====================================

def get_enhanced_style_config(combo_labels: List[str], plot_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Generate enhanced style configurations for combo labels, especially optimized for the 10 lines in morality_ratios
    
    Args:
    combo_labels: List of combination labels
    plot_type: Chart type ('zealot_numbers' or 'morality_ratios')
    
    Returns:
    dict: Style configuration dictionary
    """
    # Define an extended color palette
    colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange  
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf',  # cyan
        '#aec7e8',  # light blue
        '#ffbb78'   # light orange
    ]
    
    # Define multiple line styles
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (3, 3)), (0, (1, 1))]
    
    # Define multiple markers
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'H', 'X', '+', 'x']
    
    style_config = {}
    
    if plot_type == 'morality_ratios':
        # Define color mapping: group by zealot mode and ID-align
        zealot_mode_colors = {
            'None': {
                'base': '#505050',      # dark gray (ID-cluster=True)
                'light': '#c0c0c0'      # light gray (ID-cluster=False)
            },
            'Random': {
                'base': '#ff4500',      # deep orange-red (ID-align=True)
                'light': '#ff8080'      # light pink (ID-align=False)  
            },
            'Clustered': {
                'base': '#0066cc',      # dark blue (ID-align=True)
                'light': '#00cc66'      # bright green (ID-align=False)
            }
        }
        
        # Define marker mapping: group by ID-cluster
        id_cluster_markers = {
            'True': 'o',      # circle for ID-cluster=True
            'False': '^'      # triangle for ID-cluster=False
        }
        
        # Define marker size mapping: group by ID-align
        id_align_sizes = {
            'True': 10,        # large marker for ID-align=True
            'False': 5         # small marker for ID-align=False
        }
        
        for label in combo_labels:
            # Parse configuration information from the label
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
        # For zealot_numbers, use a simple configuration
        for i, label in enumerate(combo_labels):
            style_config[label] = {
                'color': colors[i % len(colors)],
                'linestyle': linestyles[i % len(linestyles)],
                'marker': markers[i % len(markers)],
                'markersize': 7,
                'group': 'Default'
            }
    
    # Add style entries with identity suffixes for variance_per_identity metric
    # Copy each base style and add variants with (ID=1) and (ID=-1) suffixes
    variance_style_additions = {}
    for label, base_style in style_config.items():
        # Create style for ID=1 (solid line + circle marker)
        id1_style = base_style.copy()
        id1_style['linestyle'] = '-'
        # id1_style['marker'] = 'o'
        variance_style_additions[f"{label} (ID=1)"] = id1_style
        
        # Create style for ID=-1 (dashed line + square marker, slightly smaller)
        id_neg1_style = base_style.copy()
        id_neg1_style['linestyle'] = '--'
        # id_neg1_style['marker'] = 's'
        # id_neg1_style['markersize'] = max(6, base_style.get('markersize', 10) - 2)
        variance_style_additions[f"{label} (ID=-1)"] = id_neg1_style
        
        # Add variants with plus sign for variance_per_identity_combined
        # Create style for ID=+1 (solid line)
        id_plus1_style = base_style.copy()
        id_plus1_style['linestyle'] = '-'
        variance_style_additions[f"{label} (ID=+1)"] = id_plus1_style
    
    # Add the new style entries to the original style_config
    style_config.update(variance_style_additions)
    
    return style_config


def get_variance_per_identity_style(identity_label: str, plot_type: str) -> Dict[str, Any]:
    """
    Generate special style configuration for variance per identity plots
    
    Args:
        identity_label: Label with identity identifier, e.g., "Random, ID-align=True (ID=1)"
        plot_type: Chart type
    
    Returns:
        dict: Style configuration
    """
    # Extended color palette (deduplicated and ensuring enough colors)
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
        '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d',
        '#9edae5', '#ff1744', '#00e676', '#ffea00', '#651fff', '#ff6f00',
        '#00bcd4', '#795548', '#607d8b', '#e91e63', '#4caf50', '#ffc107'
    ]
    
    # Predefined mapping of labels to color indices (to avoid hash collisions)
    label_color_mapping = {
        # 10 base labels for morality_ratios experiment
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
        # 4 base labels for zealot_numbers experiment
        'Random Zealots, Morality=0.0': 10,
        'Random Zealots, Morality=0.3': 11,
        'Clustered Zealots, Morality=0.0': 12,
        'Clustered Zealots, Morality=0.3': 13,
    }
    
    # Line style combination: solid for ID=1, dashed for ID=-1
    linestyles = {
        '1': '-',      # solid line for identity=1
        '-1': '--'     # dashed line for identity=-1
    }
    
    # Marker shapes: circle for ID=1, square for ID=-1
    markers = {
        '1': 'o',      # circle for identity=1
        '-1': 's'      # square for identity=-1
    }
    
    # Extract identity value
    identity_val = identity_label.split('(ID=')[-1].rstrip(')')
    
    # Extract original combination label
    base_label = identity_label.split(' (ID=')[0]
    
    # Use predefined mapping or fall back to hash method
    if base_label in label_color_mapping:
        base_color_index = label_color_mapping[base_label]
    else:
        # Fall back to hash method (for undefined labels)
        base_color_index = abs(hash(base_label)) % len(colors)
    
    # Select a different color for the ID=-1 group (ensuring no conflicts)
    if identity_val == '-1':
        # For ID=-1, use a fixed offset to ensure no repetition
        color_index = (base_color_index + 15) % len(colors)
    else:
        color_index = base_color_index
    
    return {
        'color': colors[color_index],
        'linestyle': linestyles.get(identity_val, '-'),
        'marker': markers.get(identity_val, 'o'),
        'markersize': 8 if identity_val == '1' else 6,  # Slightly larger marker for ID=1
        'group': f'identity_{identity_val}'
    }


def simplify_label(combo_label: str) -> str:
    """
    Simplify combination labels and provide backward-compatible label conversion
    
    Converts old format labels to new format for consistency:
    - "Random, ID-align=..." → "Random Zealots, ID-align=..."
    - "Clustered, ID-align=..." → "Clustered Zealots, ID-align=..."
    
    Args:
    combo_label: Original combination label
    
    Returns:
    str: Converted label
    """
    # Backward compatibility: convert old format labels to new format
    if combo_label.startswith('Random, ID-align='):
        return combo_label.replace('Random, ID-align=', 'Random Zealots, ID-align=')
    elif combo_label.startswith('Clustered, ID-align='):
        return combo_label.replace('Clustered, ID-align=', 'Clustered Zealots, ID-align=')
    else:
        # For new format labels or other types, return directly
        return combo_label


def plot_results_with_manager(data_manager: ExperimentDataManager, 
                            plot_type: str,
                            enable_smoothing: bool = True,
                            target_step: int = 2,
                            smooth_method: str = 'savgol',
                            error_band_type: str = 'std') -> None:
    """
    Plot experiment results using the data manager
    
    Args:
        data_manager: Data manager instance  
        plot_type: 'zealot_numbers' or 'morality_ratios'
        enable_smoothing: Whether to enable smoothing and resampling
        target_step: Target step for resampling (e.g., from step 1 to step 2)
        smooth_method: Smoothing method ('savgol', 'moving_avg', 'none')
        error_band_type: Error band type for zealot_numbers plots ('std' or 'percentile')
    """
    # Set a larger minimum font size for better readability
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 26,
        'axes.labelsize': 22,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 16
    })

    # Get plotting data from the data manager
    all_results, x_values, total_runs_per_combination = data_manager.convert_to_plotting_format(plot_type)
    
    if not all_results:
        print(f"No data found for {plot_type} plotting")
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
    
    # Calculate total runs range (for filenames)
    min_runs = min(total_runs_per_combination.values()) if total_runs_per_combination else 0
    max_runs = max(total_runs_per_combination.values()) if total_runs_per_combination else 0
    
    if min_runs == max_runs:
        runs_suffix = f"_{min_runs}runs"
    else:
        runs_suffix = f"_{min_runs}-{max_runs}runs"
    
    # Create mean_plots folder
    plot_folders = {
        'mean': os.path.join(output_dir, 'mean_plots')
    }
    
    os.makedirs(plot_folders['mean'], exist_ok=True)

    # Get style configuration
    combo_labels = list(all_results.keys())
    style_config = get_enhanced_style_config(combo_labels, plot_type)
    
    print(f"\nStyle Configuration for {plot_type}: {len(combo_labels)} combinations")
    print(f"Style configuration completed successfully")
    
    # Generate high-quality mean plots for each metric
    for metric in metrics:
        if enable_smoothing:
            print(f"  Generating smoothed plot for {metric_labels[metric]} (step={target_step}, method={smooth_method})...")
        else:
            print(f"  Generating high-quality mean plot for {metric_labels[metric]}...")
        
        # Preprocess data: calculate mean and standard deviation (for error bands)
        processed_data = {}
        
        if metric == 'variance_per_identity_combined':
            # For the combined variance per identity plot, create two lines for each combination
            for combo_label, results in all_results.items():
                # Process data for identity=1
                metric_data_1 = results['variance_per_identity_1']
                means_1, stds_1 = [], []
                lower_percentiles_1, upper_percentiles_1 = [], []
                lower_ci_1, upper_ci_1 = [], []
                for i, x_runs in enumerate(metric_data_1):
                    valid_runs = [val for val in x_runs if not np.isnan(val)]
                    if valid_runs:
                        means_1.append(np.mean(valid_runs))
                        stds_1.append(np.std(valid_runs, ddof=1) if len(valid_runs) > 1 else 0.0)
                        
                        # Calculate percentiles
                        lower_p, upper_p = calculate_percentile_bands(valid_runs, percentile_range=(25.0, 75.0))
                        lower_percentiles_1.append(lower_p)
                        upper_percentiles_1.append(upper_p)
                        
                        # Calculate confidence interval
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
                
                # Process data for identity=-1
                metric_data_neg1 = results['variance_per_identity_-1']
                means_neg1, stds_neg1 = [], []
                lower_percentiles_neg1, upper_percentiles_neg1 = [], []
                lower_ci_neg1, upper_ci_neg1 = [], []
                for i, x_runs in enumerate(metric_data_neg1):
                    valid_runs = [val for val in x_runs if not np.isnan(val)]
                    if valid_runs:
                        means_neg1.append(np.mean(valid_runs))
                        stds_neg1.append(np.std(valid_runs, ddof=1) if len(valid_runs) > 1 else 0.0)
                        
                        # Calculate percentiles
                        lower_p, upper_p = calculate_percentile_bands(valid_runs, percentile_range=(25.0, 75.0))
                        lower_percentiles_neg1.append(lower_p)
                        upper_percentiles_neg1.append(upper_p)
                        
                        # Calculate confidence interval
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
                
                # Create data for two lines
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
            # For individual variance per identity metrics, each combo label is split into two lines
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
                        
                        # Calculate percentiles
                        lower_p, upper_p = calculate_percentile_bands(valid_runs, percentile_range=(25.0, 75.0))
                        lower_percentiles.append(lower_p)
                        upper_percentiles.append(upper_p)
                        
                        # Calculate confidence interval
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
                
                # Create labels with identity identifiers for variance per identity
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
            # For other metrics, calculate mean, std dev, percentiles, and confidence intervals
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
                        
                        # Calculate percentiles (default: 25th-75th)
                        lower_p, upper_p = calculate_percentile_bands(valid_runs, percentile_range=(25.0, 75.0))
                        lower_percentiles.append(lower_p)
                        upper_percentiles.append(upper_p)
                        
                        # Calculate confidence interval (default: 99%)
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
        
        # Add run count information to the title
        if plot_type == 'zealot_numbers':
            # Determine title based on error band type
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
        
        # High-quality mean curve plot
        # For variance per identity, use a larger figure to accommodate more lines
        if metric.startswith('variance_per_identity'):
            plt.figure(figsize=(24, 14) if plot_type == 'morality_ratios' else (20, 12))
        else:
            plt.figure(figsize=(20, 12) if plot_type == 'morality_ratios' else (18, 10))
            
        for display_label, data in processed_data.items():
            # For variance per identity, extract the original combo label to get run info
            if metric.startswith('variance_per_identity'):
                # Extract "Original Label" from "Original Label (ID=1)"
                if metric == 'variance_per_identity_combined':
                    # For combined plot, use the base_combo field
                    original_combo_label = data.get('base_combo', display_label.split(' (ID=')[0])
                else:
                    original_combo_label = display_label.split(' (ID=')[0]
                runs_info = total_runs_per_combination.get(original_combo_label, 0)
            else:
                original_combo_label = display_label
                runs_info = total_runs_per_combination.get(display_label, 0)
            
            # Apply smoothing and resampling
            if enable_smoothing:
                smoothed_x, smoothed_means = resample_and_smooth_data(
                    np.array(x_values), data['means'], 
                    target_step=target_step, 
                    smooth_window=3
                )
                
                # Final smoothing
                final_means = apply_final_smooth(smoothed_means, method=smooth_method, window=5)
                
                # Use smoothed data
                plot_x, plot_y = smoothed_x, final_means
                
                # Also update label to show smoothing info
                short_label = simplify_label(display_label)
                label_with_runs = f"{short_label} (n={runs_info}, smoothed)"
            else:
                plot_x, plot_y = np.array(x_values), data['means']
                short_label = simplify_label(display_label)
                label_with_runs = f"{short_label} (n={runs_info})"
            
            # All metrics use style_config uniformly
            style = style_config.get(display_label, {})
            
            # Get color and style
            line_color = style.get('color', 'blue')
            
            # Plot the main mean curve
            plt.plot(plot_x, plot_y, label=label_with_runs, 
                    color=line_color,
                    linestyle=style.get('linestyle', '-'),
                    marker=style.get('marker', 'o'), 
                    linewidth=3.5, markersize=style.get('markersize', 10), alpha=0.85,
                    markeredgewidth=2, markeredgecolor='white')
            
            # Add error bands for zealot_numbers
            if plot_type == 'zealot_numbers' and not enable_smoothing:
                if error_band_type == 'std' and 'stds' in data:
                    # Use standard deviation method (mean ± std)
                    means = data['means']
                    stds = data['stds']
                    draw_std_error_bands(x_values, means, stds, line_color, alpha=0.2)
                    
                elif error_band_type == 'percentile' and 'lower_percentiles' in data and 'upper_percentiles' in data:
                    # Use percentile method (25th-75th percentile)
                    lower_percentiles = data['lower_percentiles']
                    upper_percentiles = data['upper_percentiles']
                    draw_percentile_error_bands(x_values, lower_percentiles, upper_percentiles, line_color, alpha=0.2)
                    
                elif error_band_type == 'confidence' and 'lower_ci' in data and 'upper_ci' in data:
                    # Use confidence interval method (99% confidence interval)
                    lower_ci = data['lower_ci']
                    upper_ci = data['upper_ci']
                    draw_confidence_interval_error_bands(x_values, lower_ci, upper_ci, line_color, alpha=0.2)
        
        plt.xlabel(x_label, fontsize=22)
        plt.ylabel(metric_labels[metric], fontsize=22)
        plt.title(f'{metric_labels[metric]} vs {x_label}{title_suffix}', fontsize=26, fontweight='bold')
        plt.tick_params(axis='both', labelsize=18)
        
        # Adjust legend layout based on metric type and number of lines
        if metric == 'variance_per_identity_combined':
            # Combined variance per identity plot: 2 lines per combination
            if plot_type == 'morality_ratios':
                # 20 lines, use 4 columns
                plt.legend(bbox_to_anchor=(0.5, -0.20), loc='upper center', ncol=4, 
                          fontsize=14, frameon=True, fancybox=True, shadow=True)
            else:
                # 8 lines, use 3 columns
                plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3, fontsize=16)
        # elif metric.startswith('variance_per_identity'):
        #     # Individual variance per identity plots have more lines, need more columns and smaller font
        #     if plot_type == 'morality_ratios':
        #         # 20 lines, use 4 columns
        #         plt.legend(bbox_to_anchor=(0.5, -0.20), loc='upper center', ncol=4, 
        #                   fontsize=10, frameon=True, fancybox=True, shadow=True)
        #     else:
        #         # 8 lines, use 3 columns
        #         plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3, fontsize=11)
        else:
            # Other metrics keep original layout
            if plot_type == 'morality_ratios':
                plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3, 
                          fontsize=16, frameon=True, fancybox=True, shadow=True)
            else:
                plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=16)
        
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Add smoothing identifier and error band type to filename
        if enable_smoothing:
            if plot_type == 'zealot_numbers':
                filename = f"{plot_type}_{metric}_smoothed_step{target_step}_{smooth_method}{runs_suffix}.png"
            else:
                filename = f"{plot_type}_{metric}_smoothed_step{target_step}_{smooth_method}{runs_suffix}.png"
        else:
            if plot_type == 'zealot_numbers':
                # Add error band type to filename
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
        
        # High-quality PNG save (DPI 300)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='white', 
                   format='png', transparent=False, 
                   pad_inches=0.1, metadata={'Creator': 'Zealot Morality Analysis'})
        
        plt.close()
    
    print(f"  Done: Generated high-quality mean plots for {plot_type}:")
    print(f"     - Mean line plots: {plot_folders['mean']}")


# =====================================
# High-level interface functions
# =====================================

def run_and_accumulate_data(output_dir: str = "results/zealot_morality_analysis", 
                           num_runs: int = 5, max_zealots: int = 50, max_morality: int = 30,
                           batch_name: str = "", num_processes: int = 1):
    """
    Run tests and save data using the new data manager (Part 1)
    
    Args:
    output_dir: Output directory
    num_runs: Number of runs for this execution
    max_zealots: Maximum number of zealots
    max_morality: Maximum morality ratio (%)
    batch_name: Batch name to identify this run
    num_processes: Number of parallel processes, 1 means serial execution
    """
    print("Running Tests and Accumulating Data with New Data Manager")
    print("=" * 70)
    
    start_time = time.time()
    
    # Create data manager
    data_manager = ExperimentDataManager(output_dir)
    
    # Get parameter combinations
    combinations = create_config_combinations()
    
    if not batch_name:
        batch_name = time.strftime("%Y%m%d_%H%M%S")
    
    # Generate batch seed to ensure different batches produce different random results
    batch_seed = int(time.time() * 1000) % (2**31)  # Use timestamp to generate seed
    
    print(f"Batch Configuration:")
    print(f"   Batch name: {batch_name}")
    print(f"   Number of runs this batch: {num_runs}")
    print(f"   Max zealots: {max_zealots}")
    print(f"   Max morality ratio: {max_morality}%")
    print(f"   Parallel processes: {num_processes} ({'Parallel' if num_processes > 1 else 'Serial'})")
    print(f"   Output directory: {output_dir}")
    print(f"   Storage format: Parquet (optimized for space and speed)")
    print()
    
    # # === Process Plot 1: x-axis is zealot numbers ===
    print("Running Test Type 1: Zealot Numbers Analysis")
    print("-" * 50)
    
    plot1_start_time = time.time()
    
    zealot_x_values = list(range(0, max_zealots + 1, 2))  # 0, 1, 2, ..., n
    zealot_results = {}
    
    for combo in combinations['zealot_numbers']:
        print(f"Running combination: {combo['label']}")
        results = run_parameter_sweep('zealot_numbers', combo, zealot_x_values, num_runs, num_processes, batch_seed)
        zealot_results[combo['label']] = results
    
    # Save zealot numbers data using the new data manager
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
    
    print(f"  Test Type 1 completed in: {format_duration(plot1_duration)}")
    print()
    
    # === Process Plot 2: x-axis is morality ratio ===
    print("Running Test Type 2: Morality Ratio Analysis")
    print("-" * 50)
    
    plot2_start_time = time.time()
    
    morality_x_values = list(range(0, max_morality + 1, 2))  # 0, 1, 2, ..., n
    morality_results = {}
    
    for combo in combinations['morality_ratios']:
        print(f"Running combination: {combo['label']}")
        results = run_parameter_sweep('morality_ratios', combo, morality_x_values, num_runs, num_processes, batch_seed)
        morality_results[combo['label']] = results
    
    # Save morality ratio data using the new data manager
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
    
    print(f"  Test Type 2 completed in: {format_duration(plot2_duration)}")
    print()
    
    # Calculate total time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "=" * 70)
    print("Data Collection Completed Successfully!")
    print(f"Batch '{batch_name}' with {num_runs} runs per parameter point")
    print()
    print("  Timing Summary:")
    # print(f"   Test Type 1 (Zealot Numbers): {format_duration(plot1_duration)}")
    print(f"   Test Type 2 (Morality Ratios): {format_duration(plot2_duration)}")
    print(f"   Total execution time: {format_duration(elapsed_time)}")
    print(f"Data saved using Parquet format in: {output_dir}/")
    
    # Save experiment configuration to data manager
    experiment_config = {
        'batch_name': batch_name,
        'num_runs': num_runs,
        'max_zealots': max_zealots,
        'max_morality': max_morality,
        'elapsed_time': elapsed_time,
        'total_combinations': len(combinations['zealot_numbers']) + len(combinations['morality_ratios'])
    }
    data_manager.save_experiment_config(experiment_config)
    
    # Display data manager summary
    print("\n" + data_manager.export_summary_report())


def plot_from_accumulated_data(output_dir: str = "results/zealot_morality_analysis",
                             enable_smoothing: bool = True,
                             target_step: int = 2,
                             smooth_method: str = 'savgol',
                             error_band_type: str = 'std'):
    """
    Read data from the new data manager and generate plots (Part 2)
    
    Note: For zealot_numbers plots, smoothing will be forced off to show error bands.
         For morality_ratios plots, the user-specified smoothing settings will be used.
    
    Args:
        output_dir: Output directory
        enable_smoothing: Whether to enable smoothing (only affects morality_ratios plots)
        target_step: Resampling step (2 means resampling from 101 to 51 points)
        smooth_method: Smoothing method ('savgol', 'moving_avg', 'none')
        error_band_type: Error band type for zealot_numbers plots ('std' or 'percentile')
    """
    print("Generating Plots from Data Manager")
    if enable_smoothing:
        print(f"Smoothing enabled: step={target_step}, method={smooth_method}")
    print("=" * 70)
    
    start_time = time.time()
    
    # Create data manager
    data_manager = ExperimentDataManager(output_dir)
    
    # Display data summary
    print("\n" + data_manager.export_summary_report())
    
    # Generate zealot numbers plots (with smoothing off to show error bands)
    print("\nGenerating Zealot Numbers Plots...")
    zealot_summary = data_manager.get_experiment_summary('zealot_numbers')
    if zealot_summary['total_records'] > 0:
        plot_results_with_manager(data_manager, 'zealot_numbers', 
                                False, target_step, smooth_method, error_band_type)  # Force smoothing off, use specified error band type
        if error_band_type == 'std':
            band_type_description = "standard deviation"
        elif error_band_type == 'percentile':
            band_type_description = "percentile (25th-75th)"
        elif error_band_type == 'confidence':
            band_type_description = "confidence interval (99%)"
        else:
            band_type_description = "unknown"
        print(f"Done: Generated {len(zealot_summary['combinations'])} zealot numbers plots with {band_type_description} error bands")
    else:
        print("Error: No zealot numbers data found")
    
    # Generate morality ratios plots (keeping user-defined smoothing options)
    print("\nGenerating Morality Ratios Plots...")
    morality_summary = data_manager.get_experiment_summary('morality_ratios')
    if morality_summary['total_records'] > 0:
        plot_results_with_manager(data_manager, 'morality_ratios',
                                enable_smoothing, target_step, smooth_method, 'std')  # morality_ratios does not use error bands, pass default
        if enable_smoothing:
            print(f"Done: Generated {len(morality_summary['combinations'])} morality ratios plots with smoothing")
        else:
            print(f"Done: Generated {len(morality_summary['combinations'])} morality ratios plots without smoothing")
    else:
        print("Error: No morality ratios data found")
    
    # Calculate total time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "=" * 70)
    print("Plot Generation Completed Successfully!")
    print(f"Generated plots from Parquet data files")
    if error_band_type == 'std':
        band_type_description = "standard deviation"
    elif error_band_type == 'percentile':
        band_type_description = "percentile (25th-75th)"
    elif error_band_type == 'confidence':
        band_type_description = "confidence interval (99%)"
    else:
        band_type_description = "unknown"
    print(f"Zealot Numbers: {band_type_description} error bands enabled (smoothing disabled)")
    if enable_smoothing:
        print(f"Morality Ratios: Smoothing enabled (step {target_step}, {smooth_method})")
    else:
        print(f"Morality Ratios: Smoothing disabled")
    print(f"  Total plotting time: {format_duration(elapsed_time)}")
    print(f"Plots saved to: {output_dir}/mean_plots/")


def run_zealot_morality_analysis(output_dir: str = "results/zealot_morality_analysis", 
                                num_runs: int = 5, max_zealots: int = 50, max_morality: int = 30, num_processes: int = 1,
                                error_band_type: str = 'std'):
    """
    Run the complete zealot and morality analysis experiment (backward compatible)
    
    Args:
    output_dir: Output directory
    num_runs: Number of runs per parameter point
    max_zealots: Maximum number of zealots
    max_morality: Maximum morality ratio (%)
    num_processes: Number of parallel processes, 1 means serial execution
    error_band_type: Error band type for zealot_numbers plots ('std' or 'percentile')
    """
    print("Starting Complete Zealot and Morality Analysis Experiment")
    print("=" * 70)
    
    # Step 1: Run tests and accumulate data
    run_and_accumulate_data(output_dir, num_runs, max_zealots, max_morality, "", num_processes)
    
    # Step 2: Generate plots from accumulated data
    plot_from_accumulated_data(output_dir, error_band_type=error_band_type)


def run_no_zealot_morality_data(output_dir: str = "results/zealot_morality_analysis", 
                               num_runs: int = 5, max_morality: int = 30,
                               batch_name: str = "", num_processes: int = 1):
    """
    Run no-zealot morality ratio data collection separately (using new data manager)
    
    Args:
    output_dir: Output directory
    num_runs: Number of runs per parameter point
    max_morality: Maximum morality ratio (%)
    batch_name: Batch name
    num_processes: Number of parallel processes, 1 means serial execution
    """
    print("Running No Zealot Morality Ratio Data Collection with New Data Manager")
    print("=" * 70)
    
    start_time = time.time()
    
    # Create data manager
    data_manager = ExperimentDataManager(output_dir)
    
    # Get all parameter combinations
    combinations = create_config_combinations()
    
    # Select only combinations where zealot_mode is 'none'
    no_zealot_combinations = [combo for combo in combinations['morality_ratios'] 
                             if combo['zealot_mode'] == 'none']
    
    if not no_zealot_combinations:
        print("Error: No combinations with zealot_mode='none' found")
        return
    
    if not batch_name:
        batch_name = f"no_zealot_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # Generate batch seed to ensure different batches produce different random results
    batch_seed = int(time.time() * 1000) % (2**31)  # Use timestamp to generate seed
    
    print(f"No Zealot Batch Configuration:")
    print(f"   Batch name: {batch_name}")
    print(f"   Number of runs this batch: {num_runs}")
    print(f"   Max morality ratio: {max_morality}%")
    print(f"   Number of no-zealot combinations: {len(no_zealot_combinations)}")
    print(f"   Output directory: {output_dir}")
    print(f"   Storage format: Parquet (optimized for space and speed)")
    print()
    
    # Set x-axis values for morality ratio
    morality_x_values = list(range(0, max_morality + 1, 2))  # 0, 2, 4, ..., max_morality
    morality_results = {}
    
    print("Running No Zealot Morality Ratio Analysis")
    print("-" * 50)
    
    for combo in no_zealot_combinations:
        print(f"Running no-zealot combination: {combo['label']}")
        results = run_parameter_sweep('morality_ratios', combo, morality_x_values, num_runs, num_processes, batch_seed)
        morality_results[combo['label']] = results
    
    # Save no-zealot morality ratio data using the new data manager
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
    
    # Calculate time taken
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "=" * 70)
    print("No Zealot Data Collection Completed Successfully!")
    print(f"Batch '{batch_name}' with {num_runs} runs per parameter point")
    print(f"  Total execution time: {format_duration(elapsed_time)}")
    print(f"Data saved using Parquet format in: {output_dir}/")
    
    # Save experiment configuration to data manager
    experiment_config = {
        'batch_name': batch_name,
        'num_runs': num_runs,
        'max_morality': max_morality,
        'elapsed_time': elapsed_time,
        'total_combinations': len(no_zealot_combinations),
        'experiment_type': 'no_zealot_only'
    }
    data_manager.save_experiment_config(experiment_config)
    
    # Display data manager summary
    print("\n" + data_manager.export_summary_report())


def calculate_percentile_bands(valid_runs: List[float], percentile_range: Tuple[float, float] = (25.0, 75.0)) -> Tuple[float, float]:
    """
    Calculate the percentile range for the given data
    
    Args:
        valid_runs: List of valid run data
        percentile_range: Percentile range, defaults to (25.0, 75.0) for 25th-75th percentile
    
    Returns:
        tuple: (lower_percentile, upper_percentile)
    """
    if len(valid_runs) < 2:
        return 0.0, 0.0
    
    lower_percentile = np.percentile(valid_runs, percentile_range[0])
    upper_percentile = np.percentile(valid_runs, percentile_range[1])
    
    return lower_percentile, upper_percentile


def calculate_confidence_interval(valid_runs: List[float], confidence_level: float = 0.99) -> Tuple[float, float]:
    """
    Calculate the confidence interval for the given data
    
    Args:
        valid_runs: List of valid run data
        confidence_level: Confidence level, defaults to 0.99 (99% confidence interval)
    
    Returns:
        tuple: (lower_bound_of_confidence_interval, upper_bound_of_confidence_interval)
    """
    if len(valid_runs) < 2:
        return 0.0, 0.0
    
    # Calculate sample mean and standard error
    sample_mean = np.mean(valid_runs)
    sample_std = np.std(valid_runs, ddof=1)  # Sample standard deviation
    sample_size = len(valid_runs)
    standard_error = sample_std / np.sqrt(sample_size)
    
    # Calculate t-value (using t-distribution, more accurate for small samples)
    alpha = 1 - confidence_level
    degrees_of_freedom = sample_size - 1
    t_value = stats.t.ppf(1 - alpha/2, degrees_of_freedom)
    
    # Calculate confidence interval
    margin_of_error = t_value * standard_error
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error
    
    return lower_bound, upper_bound


def draw_std_error_bands(x_values, means, stds, line_color, alpha=0.2):
    """
    Draw standard deviation error bands (mean ± std)
    
    Args:
        x_values: x-axis data
        means: Array of means
        stds: Array of standard deviations
        line_color: Line color
        alpha: Transparency
    """
    # Calculate upper and lower bounds
    upper_bound = means + stds
    lower_bound = means - stds
    
    # Draw error bands (using the same color but lower alpha)
    plt.fill_between(x_values, lower_bound, upper_bound, 
                   color=line_color, alpha=alpha, 
                   linewidth=0, interpolate=True)


def draw_percentile_error_bands(x_values, lower_percentiles, upper_percentiles, line_color, alpha=0.2):
    """
    Draw percentile error bands (lower percentile - upper percentile)
    
    Args:
        x_values: x-axis data
        lower_percentiles: Array of lower percentiles
        upper_percentiles: Array of upper percentiles
        line_color: Line color
        alpha: Transparency
    """
    # Draw error bands (using the same color but lower alpha)
    plt.fill_between(x_values, lower_percentiles, upper_percentiles, 
                   color=line_color, alpha=alpha, 
                   linewidth=0, interpolate=True)


def draw_confidence_interval_error_bands(x_values, lower_ci, upper_ci, line_color, alpha=0.2):
    """
    Draw confidence interval error bands (lower CI bound - upper CI bound)
    
    Args:
        x_values: x-axis data
        lower_ci: Array of lower confidence interval bounds
        upper_ci: Array of upper confidence interval bounds
        line_color: Line color
        alpha: Transparency
    """
    # Draw error bands (using the same color but lower alpha)
    plt.fill_between(x_values, lower_ci, upper_ci, 
                   color=line_color, alpha=alpha, 
                   linewidth=0, interpolate=True)


if __name__ == "__main__":
    # New separate usage method:
    
    # Start timer
    main_start_time = time.time()
    
    # Method 1: Run in two steps
    # Step 1: Run tests and accumulate data (can be run multiple times to accumulate more data)
    print("=" * 50)
    print("Example: Running experiment in steps")
    print("=" * 50)
    
    # Data collection phase
    data_collection_start_time = time.time()
    
    # The following command can be run multiple times to accumulate data:
    run_and_accumulate_data(
        output_dir="results/zealot_morality_analysis",
        num_runs=100,  # Run 100 tests each time
        max_zealots=100,  
        max_morality=100,
        # batch_name="batch_001"  # Optional: name the batch
        num_processes=8  # Use 8 processes for parallel computation
    )
    
    data_collection_end_time = time.time()
    data_collection_duration = data_collection_end_time - data_collection_start_time
    

    # Step 2: Plotting phase

    plotting_start_time = time.time()

    # ===== ERROR BANDS Configuration: Switch by commenting/uncommenting =====
    # Method 1: Standard Deviation Bands: Shows mean ± standard deviation
    # error_band_type = 'std'  # Use standard deviation method
    # Method 2: Percentile Bands: Shows 25th to 75th percentile range
    # error_band_type = 'percentile'  # Use percentile method
    # Method 3: Confidence Interval Bands: Shows 99% confidence interval
    error_band_type = 'confidence'  # Use confidence interval method

    plot_from_accumulated_data(
        output_dir="results/zealot_morality_analysis",
        enable_smoothing=False,       # Disable smoothing
        target_step=2,             # Resample from step 1 to step 2 (101 points -> 51 points)
        smooth_method='savgol',     # Use Savitzky-Golay smoothing
        error_band_type=error_band_type  # Use the error band type configured above
    )
    
    plotting_end_time = time.time()
    plotting_duration = plotting_end_time - plotting_start_time
    
    # Calculate total time
    main_end_time = time.time()
    total_duration = main_end_time - main_start_time
    
    # Display timing summary
    print("\n" + "=" * 50)
    print("Complete Experiment Timing Summary")
    print("=" * 50)
    print(f"Data collection phase took: {format_duration(data_collection_duration)}")
    print(f"Plot generation phase took: {format_duration(plotting_duration)}")
    print(f"Total time: {format_duration(total_duration)}")
    print("=" * 50) 