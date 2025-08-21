# Usage:
# python -m polarization_triangle.experiments.zealot_parameter_sweep --runs 10 --steps 300
# python -m polarization_triangle.experiments.zealot_parameter_sweep --plot-only

import os
import numpy as np
import itertools
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
import json
import pickle
import argparse
import sys
import gc
from polarization_triangle.experiments.zealot_experiment import run_zealot_experiment
from polarization_triangle.experiments.multi_zealot_experiment import run_multi_zealot_experiment, average_stats, plot_average_statistics, generate_average_heatmaps


def get_object_size(obj):
    """
    Get object size in memory (in bytes)
    
    Parameters:
    obj -- Object to measure
    
    Returns:
    int -- Object size (bytes)
    """
    def get_size(obj, seen=None):
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        
        # Mark objects that have been processed
        seen.add(obj_id)
        
        if isinstance(obj, dict):
            size += sum([get_size(v, seen) for v in obj.values()])
            size += sum([get_size(k, seen) for k in obj.keys()])
        elif hasattr(obj, '__dict__'):
            size += get_size(obj.__dict__, seen)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            size += sum([get_size(i, seen) for i in obj])
        
        return size
    
    return get_size(obj)


def format_size(size_bytes):
    """
    Format byte size to human-readable format
    
    Parameters:
    size_bytes -- Number of bytes
    
    Returns:
    str -- Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.2f} TB"


def save_experiment_data(all_configs_stats, config_names, experiment_params, output_dir, max_size_mb=500):
    """
    Save experiment data to file for subsequent plotting
    If data is too large, will automatically split and save
    
    Parameters:
    all_configs_stats -- Statistical data for all configurations
    config_names -- List of configuration names
    experiment_params -- Experiment parameters
    output_dir -- Output directory
    max_size_mb -- Maximum file size limit (MB), default 500MB
    """
    # Check data size
    total_size = get_object_size(all_configs_stats)
    max_size_bytes = max_size_mb * 1024 * 1024
    
    print(f"Total data size: {format_size(total_size)}")
    print(f"Number of configurations: {len(all_configs_stats)}")
    
    metadata_file = os.path.join(output_dir, "experiment_metadata.json")
    
    # Save metadata
    metadata = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'experiment_params': experiment_params,
        'config_names': config_names,
        'num_configurations': len(config_names),
        'total_data_size_bytes': total_size,
        'total_data_size_formatted': format_size(total_size),
        'is_split': False,
        'split_info': None
    }
    
    if total_size <= max_size_bytes:
        # Data size within limit, save directly
        data_file = os.path.join(output_dir, "experiment_data.pkl")
        
        print(f"Data size within limit, saving directly to: {data_file}")
        with open(data_file, 'wb') as f:
            pickle.dump({
                'all_configs_stats': all_configs_stats,
                'config_names': config_names,
                'experiment_params': experiment_params
            }, f)
        
        metadata['data_file'] = data_file
        
    else:
        # Data too large, need to split and save
        print(f"Data size ({format_size(total_size)}) exceeds limit ({max_size_mb}MB), splitting and saving...")
        
        # Calculate size of each configuration
        config_sizes = []
        for config_name in config_names:
            if config_name in all_configs_stats:
                config_size = get_object_size(all_configs_stats[config_name])
                config_sizes.append((config_name, config_size))
        
        # Sort by size for easier grouping
        config_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # Split data
        chunks = []
        current_chunk = []
        current_size = 0
        
        for config_name, config_size in config_sizes:
            # If single configuration exceeds limit, save separately
            if config_size > max_size_bytes:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_size = 0
                chunks.append([(config_name, config_size)])
                print(f"Warning: Configuration '{config_name}' size ({format_size(config_size)}) exceeds single file limit")
            elif current_size + config_size > max_size_bytes:
                # Current chunk is full, start new chunk
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [(config_name, config_size)]
                current_size = config_size
            else:
                # Add to current chunk
                current_chunk.append((config_name, config_size))
                current_size += config_size
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # Save split data
        split_files = []
        for i, chunk in enumerate(chunks):
            chunk_file = os.path.join(output_dir, f"experiment_data_part_{i+1}.pkl")
            chunk_configs = [config_name for config_name, _ in chunk]
            chunk_data = {config_name: all_configs_stats[config_name] for config_name in chunk_configs}
            chunk_size = sum(config_size for _, config_size in chunk)
            
            print(f"Saving chunk {i+1}/{len(chunks)}: {len(chunk_configs)} configurations, size: {format_size(chunk_size)}")
            
            with open(chunk_file, 'wb') as f:
                pickle.dump({
                    'chunk_configs_stats': chunk_data,
                    'chunk_config_names': chunk_configs,
                    'chunk_index': i + 1,
                    'total_chunks': len(chunks)
                }, f)
            
            split_files.append({
                'file': chunk_file,
                'config_names': chunk_configs,
                'config_count': len(chunk_configs),
                'size_bytes': chunk_size,
                'size_formatted': format_size(chunk_size)
            })
        
        # Update metadata
        metadata['is_split'] = True
        metadata['split_info'] = {
            'total_chunks': len(chunks),
            'split_files': split_files,
            'max_size_mb': max_size_mb
        }
        
        # Clean up memory
        del all_configs_stats
        gc.collect()
        
        print(f"Split save complete, total {len(chunks)} files")
    
    # Save metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Experiment metadata saved to: {metadata_file}")


def load_experiment_data(data_dir):
    """
    Load experiment data from file (supports split files)
    
    Parameters:
    data_dir -- Data directory
    
    Returns:
    tuple -- (all_configs_stats, config_names, experiment_params)
    """
    metadata_file = os.path.join(data_dir, "experiment_metadata.json")
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"Loading experiment data, timestamp: {metadata['timestamp']}")
    print(f"Number of configurations: {metadata['num_configurations']}")
    print(f"Data size: {metadata['total_data_size_formatted']}")
    
    config_names = metadata['config_names']
    experiment_params = metadata['experiment_params']
    
    if not metadata['is_split']:
        # Single file mode
        data_file = metadata['data_file']
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        print(f"Loading data from single file: {data_file}")
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        all_configs_stats = data['all_configs_stats']
        
    else:
        # Split file mode
        split_info = metadata['split_info']
        print(f"Loading data from {split_info['total_chunks']} split files...")
        
        all_configs_stats = {}
        
        for i, file_info in enumerate(split_info['split_files']):
            chunk_file = file_info['file']
            if not os.path.exists(chunk_file):
                raise FileNotFoundError(f"Split file not found: {chunk_file}")
            
            print(f"Loading chunk {i+1}/{split_info['total_chunks']}: {file_info['config_count']} configurations")
            
            with open(chunk_file, 'rb') as f:
                chunk_data = pickle.load(f)
            
            # Merge data
            all_configs_stats.update(chunk_data['chunk_configs_stats'])
            
            # Clean up memory
            del chunk_data
            gc.collect()
        
        print(f"Split file loading complete, loaded {len(all_configs_stats)} configurations in total")
    
    return all_configs_stats, config_names, experiment_params


def generate_plots_from_data(all_configs_stats, config_names, experiment_params, output_dir):
    """
    Generate plots from existing data
    
    Parameters:
    all_configs_stats -- Statistical data for all configurations
    config_names -- List of configuration names 
    experiment_params -- Experiment parameters
    output_dir -- Output directory
    """
    # Set a larger global font for readability
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16
    })

    # Create combined results directory
    combined_dir = os.path.join(output_dir, "combined_results")
    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)
    
    # Get steps parameter
    steps = experiment_params.get('steps', 300)
    
    # Plot combined comparison chart for all parameter combinations
    if len(all_configs_stats) > 1:
        print("\nGenerating combined comparative plots from loaded data...")
        plot_combined_statistics(all_configs_stats, config_names, combined_dir, steps)
    else:
        print("Not enough data for combined plots (need at least 2 configurations)")
    
    print(f"Plots saved to {combined_dir}")


def run_plot_only_mode(data_dir):
    """
    Plot-only mode: generate plots from existing data
    
    Parameters:
    data_dir -- Directory containing experiment data
    """
    print("Running in plot-only mode...")
    print(f"Loading data from: {data_dir}")
    
    try:
        # Load experiment data
        all_configs_stats, config_names, experiment_params = load_experiment_data(data_dir)
        
        # Generate plots
        generate_plots_from_data(all_configs_stats, config_names, experiment_params, data_dir)
        
        print("Plot-only mode completed successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have run the experiment first to generate data files.")
    except Exception as e:
        print(f"Error in plot-only mode: {e}")


def process_single_parameter_combination(params_and_config):
    """
    Function to process a single parameter combination, for use in multiprocessing
    
    Parameters:
    params_and_config -- Tuple containing parameter combination and configuration info
    
    Returns:
    dict -- Dictionary containing combination name, stats, execution time, etc.
    """
    params, config = params_and_config
    morality_rate, zealot_morality, id_clustered, zealot_count, zealot_mode = params
    runs_per_config, steps, initial_scale, base_seed, output_base_dir = config
    
    # Skip invalid combination: if zealot_mode is "none" but zealot_count is not 0
    if zealot_mode == "none" and zealot_count != 0:
        zealot_count = 0  # If mode is "none", force zealot count to 0
    
    # Create folder name from parameter combination description
    folder_name = (
        f"mor_{morality_rate:.1f}_"
        f"zm_{'T' if zealot_morality else 'F'}_"
        f"id_{'C' if id_clustered else 'R'}_"
        f"zn_{zealot_count}_"
        f"zm_{zealot_mode}"
    )
    
    # Create a more readable config name for plots
    mode_display = {
        "none": "No Zealots",
        "clustered": "Clustered",
        "random": "Random",
        "high-degree": "High-Degree"
    }
    readable_name = (
        f"Morality Rate:{morality_rate:.1f};"
        # f"Zealot Morality:{'T' if zealot_morality else 'F'};"
        f"Identity:{'Clustered' if id_clustered else 'Random'};"
        # f"Zealot Count:{zealot_count};"
        f"Zealot Mode:{mode_display.get(zealot_mode, zealot_mode)}"
    )
    
    # Output directory
    output_dir = os.path.join(output_base_dir, folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
        # Record start time
    start_time = time.time()
    process_id = os.getpid()
    print(f"Process {process_id}: Running combination {folder_name}")
    
    # Run experiment multiple times and average
    try:
        # Run experiment, use a different seed base for each process
        adjusted_base_seed = base_seed + (process_id % 10000)  # Adjust seed based on process ID
        
        avg_stats = run_zealot_parameter_experiment(
            runs=runs_per_config,
            steps=steps,
            initial_scale=initial_scale,
            morality_rate=morality_rate,
            zealot_morality=zealot_morality,
            identity_clustered=id_clustered,
            zealot_count=zealot_count,
            zealot_mode=zealot_mode,
            base_seed=adjusted_base_seed,
            output_dir=output_dir
        )
        
        # Record end time and duration
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Process {process_id}: Completed {folder_name} in {elapsed:.1f} seconds")
        
        # Log progress to a log file (each process uses a separate log file)
        log_file = os.path.join(output_base_dir, f"sweep_progress_{process_id}.log")
        with open(log_file, "a") as f:
            f.write(f"Completed: {folder_name}, Time: {elapsed:.1f}s, Process: {process_id}\n")
        
        return {
            'success': True,
            'readable_name': readable_name,
            'avg_stats': avg_stats,
            'elapsed': elapsed,
            'folder_name': folder_name
        }
        
    except Exception as e:
        print(f"Process {process_id}: Error running {folder_name}: {str(e)}")
        # Log error to a log file (each process uses a separate log file)
        error_log_file = os.path.join(output_base_dir, f"sweep_errors_{process_id}.log")
        with open(error_log_file, "a") as f:
            f.write(f"Error in {folder_name}: {str(e)}, Process: {process_id}\n")
        
        return {
            'success': False,
            'readable_name': readable_name,
            'error': str(e),
            'folder_name': folder_name
        }


# Generate all possible parameter combinations and run experiments
def run_parameter_sweep(
    runs_per_config=10,
    steps=100,
    initial_scale=0.1,
    base_seed=42,
    output_base_dir="results/zealot_parameter_sweep",
    num_processes=None,
    max_size_mb=500,
    optimize_data=True,
    preserve_essential_only=False
):
    """
    Run parameter sweep experiment, testing different parameter combinations (multiprocess version)
    
    Parameters:
    runs_per_config -- Number of runs for each parameter configuration
    steps -- Number of simulation steps for each run
    initial_scale -- Scaling factor for initial opinions
    base_seed -- Base random seed
    output_base_dir -- Base directory for results output
    num_processes -- Number of processes to use, defaults to None (use number of CPU cores)
    max_size_mb -- Maximum size limit for a single data file in MB
    optimize_data -- Whether to optimize data to reduce memory usage
    preserve_essential_only -- Whether to preserve only essential statistics
    """
    # Record total start time
    total_start_time = time.time()
    
    # Determine number of processes
    if num_processes is None:
        num_processes = cpu_count()
    
    print(f"Using {num_processes} processes for parallel execution")
    
    # Define parameter value ranges
    morality_rates = [0, 0.3]  # Proportion of moralizing non-zealot people
    zealot_moralities = [True]  # Whether all zealots are moralizing
    identity_clustered = [True, False]  # Whether to initialize with identity-based clustering
    zealot_counts = [20]  # Number of zealots
    zealot_modes = ["none", "clustered", "random", "high-degree"]  # Zealot initialization configurations
    # Create all possible parameter combinations
    param_combinations = list(itertools.product(
        morality_rates, 
        zealot_moralities, 
        identity_clustered, 
        zealot_counts, 
        zealot_modes
    ))
    
    print(f"Total parameter combinations: {len(param_combinations)}")
    print(f"Each combination will be run {runs_per_config} times")
    print(f"Total experiment runs: {len(param_combinations) * runs_per_config}")
    
    # Ensure base output directory exists
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    # Create combined results directory
    combined_dir = os.path.join(output_base_dir, "combined_results")
    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)

    # Prepare parameter combinations and config info
    config_tuple = (runs_per_config, steps, initial_scale, base_seed, output_base_dir)
    params_and_configs = [(params, config_tuple) for params in param_combinations]
    
    # Collect average statistics for all parameter combinations
    all_configs_stats = {}
    config_names = []
    
    # Use multiprocessing to handle parameter combinations in parallel
    print("\nStarting parallel processing of parameter combinations...")
    
    if num_processes > 1:
        # Multiprocess version
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(process_single_parameter_combination, params_and_configs),
                total=len(params_and_configs),
                desc="Processing combinations"
            ))
    else:
        # Single process version (for debugging)
        results = []
        for params_and_config in tqdm(params_and_configs, desc="Processing combinations"):
            results.append(process_single_parameter_combination(params_and_config))
    
    # Process results
    for result in results:
        if result['success']:
            all_configs_stats[result['readable_name']] = result['avg_stats']
            config_names.append(result['readable_name'])
        else:
            print(f"Failed to process combination: {result['folder_name']}")
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Save experiment data
    experiment_params = {
        'runs_per_config': runs_per_config,
        'steps': steps,
        'initial_scale': initial_scale,
        'base_seed': base_seed,
        'num_processes': num_processes,
        'param_combinations_total': len(param_combinations),
        'successful_combinations': len(all_configs_stats)
    }
    
    print("\nSaving experiment data...")
    save_experiment_data_with_monitoring(
        all_configs_stats, 
        config_names, 
        experiment_params, 
        output_base_dir,
        max_size_mb=max_size_mb,
        optimize_data=optimize_data,
        preserve_essential_only=preserve_essential_only
    )
    
    # Generate plots from experiment data
    print("\nGenerating plots from experiment data...")
    generate_plots_from_data(all_configs_stats, config_names, experiment_params, output_base_dir)
    
    # Merge log files from all processes
    print("\nMerging log files from all processes...")
    
    # Merge progress logs
    progress_log_file = os.path.join(output_base_dir, "sweep_progress.log")
    with open(progress_log_file, "w") as merged_log:
        for file_name in os.listdir(output_base_dir):
            if file_name.startswith("sweep_progress_") and file_name.endswith(".log"):
                process_log_file = os.path.join(output_base_dir, file_name)
                with open(process_log_file, "r") as f:
                    merged_log.write(f.read())
                # Delete process-specific log file
                os.remove(process_log_file)
    
    # Merge error logs
    error_log_file = os.path.join(output_base_dir, "sweep_errors.log")
    error_entries = []
    for file_name in os.listdir(output_base_dir):
        if file_name.startswith("sweep_errors_") and file_name.endswith(".log"):
            process_error_file = os.path.join(output_base_dir, file_name)
            with open(process_error_file, "r") as f:
                error_entries.append(f.read())
            # Delete process-specific log file
            os.remove(process_error_file)
    
    # Create error log file only if there are errors
    if error_entries and any(entry.strip() for entry in error_entries):
        with open(error_log_file, "w") as merged_error_log:
            for entry in error_entries:
                merged_error_log.write(entry)
    
    # Calculate total time
    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time
    hours, remainder = divmod(total_elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\nParameter sweep completed!")
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Processed {len(all_configs_stats)} successful combinations out of {len(param_combinations)} total combinations")
    
    # Log total time to a log file
    with open(os.path.join(output_base_dir, "sweep_summary.log"), "w") as f:
        f.write(f"Parameter Sweep Summary\n")
        f.write(f"======================\n\n")
        f.write(f"Total parameter combinations: {len(param_combinations)}\n")
        f.write(f"Successful combinations: {len(all_configs_stats)}\n")
        f.write(f"Runs per configuration: {runs_per_config}\n")
        f.write(f"Total experiment runs: {len(param_combinations) * runs_per_config}\n")
        f.write(f"Processes used: {num_processes}\n\n")
        f.write(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n")
        
    return total_elapsed


def plot_combined_statistics(all_configs_stats, config_names, output_dir, steps):
    """
    Plot combined comparison charts for all parameter combinations (using Small Multiples facet design)
    
    Parameters:
    all_configs_stats -- Dictionary containing average statistics for all parameter combinations
    config_names -- List of parameter combination names
    output_dir -- Output directory
    steps -- Number of simulation steps
    """
    # Ensure statistics directory exists
    stats_dir = os.path.join(output_dir, "statistics")
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    
    # List of statistic keys
    stat_keys = [
        ("mean_opinions", "Mean Opinion", "Mean Opinion Value"),
        ("non_zealot_variance", "Non-Zealot Variance", "Opinion Variance (Excluding Zealots)"),
        ("cluster_variance", "Cluster Variance", "Mean Opinion Variance within Clusters"),
        ("negative_counts", "Negative Counts", "Number of Agents with Negative Opinions"),
        ("negative_means", "Negative Means", "Mean Value of Negative Opinions"),
        ("positive_counts", "Positive Counts", "Number of Agents with Positive Opinions"),
        ("positive_means", "Positive Means", "Mean Value of Positive Opinions"),
        ("polarization_index", "Polarization Index", "Polarization Index"),
    ]
    
    # Parse config name, group by Zealot Mode
    def parse_config_name(config_name):
        parts = config_name.split(';')
        morality_rate = float(parts[0].split(':')[1])
        identity_type = parts[1].split(':')[1]
        zealot_mode = parts[2].split(':')[1]
        return morality_rate, identity_type, zealot_mode
    
    # Group data by Zealot Mode
    zealot_modes = ["No Zealots", "Clustered", "Random", "High-Degree"]
    grouped_data = {mode: {} for mode in zealot_modes}
    
    for config_name in config_names:
        if config_name in all_configs_stats:
            try:
                morality_rate, identity_type, zealot_mode = parse_config_name(config_name)
                
                # Create simplified label
                simple_label = f"Morality {morality_rate:.1f}, Identity {identity_type}"
                
                # Get data
                if "without Zealots" in all_configs_stats[config_name]:
                    mode_data = all_configs_stats[config_name]["without Zealots"]
                elif len(all_configs_stats[config_name]) > 0:
                    mode_key = list(all_configs_stats[config_name].keys())[0]
                    mode_data = all_configs_stats[config_name][mode_key]
                else:
                    continue
                
                grouped_data[zealot_mode][simple_label] = mode_data
                
            except (IndexError, ValueError, KeyError) as e:
                print(f"Warning: Could not parse config name '{config_name}': {e}")
                continue
    
    # Define colors and line styles
    color_style_map = {
        "Morality 0.0, Identity Clustered": ('#1f77b4', '-'),    # Blue solid line
        "Morality 0.0, Identity Random": ('#1f77b4', '--'),      # Blue dashed line
        "Morality 0.3, Identity Clustered": ('#ff7f0e', '-'),    # Orange solid line
        "Morality 0.3, Identity Random": ('#ff7f0e', '--'),      # Orange dashed line
    }
    
    # Plot faceted chart for each statistic
    for stat_key, stat_label, stat_title in stat_keys:
        # Check if this statistic data exists
        has_stat_data = any(
            any(stat_key in mode_data for mode_data in group_data.values())
            for group_data in grouped_data.values()
        )
        
        if not has_stat_data:
            continue
        
        # First, collect all data to be plotted to calculate global y-axis range
        all_data_values = []
        for zealot_mode in zealot_modes:
            for config_label, mode_data in grouped_data[zealot_mode].items():
                if stat_key in mode_data:
                    data = mode_data[stat_key]
                    all_data_values.extend(data)
        
        # Calculate global y-axis range, add 5% margin
        if all_data_values:
            global_y_min = min(all_data_values)
            global_y_max = max(all_data_values)
            y_range = global_y_max - global_y_min
            y_margin = max(0.05 * y_range, 0.01 * abs(global_y_max))  # 5% margin, min 1%
            global_y_min -= y_margin
            global_y_max += y_margin
        else:
            global_y_min, global_y_max = 0, 1
        
        # Create 2x2 faceted plot
        fig, axes = plt.subplots(2, 2, figsize=(21, 14.5))
        fig.suptitle(f'Comparison of {stat_title} by Zealot Mode', fontsize=24, y=0.98)
        
        # Flatten axes array for easy indexing
        axes_flat = axes.flatten()
        
        # Plot subplot for each Zealot Mode
        for i, zealot_mode in enumerate(zealot_modes):
            ax = axes_flat[i]
            
            # Plot all configurations for this Zealot Mode
            for config_label, mode_data in grouped_data[zealot_mode].items():
                if stat_key in mode_data:
                    data = mode_data[stat_key]
                    color, linestyle = color_style_map.get(config_label, ('#666666', '-'))
                    
                    ax.plot(
                        range(len(data)), 
                        data, 
                        label=config_label,
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.5
                    )
            
            # Set subplot properties
            ax.set_title(f'{zealot_mode}', fontsize=20, fontweight='bold')
            ax.set_xlabel('Step', fontsize=18)
            ax.set_ylabel(stat_label, fontsize=18)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=16)
            ax.tick_params(axis='both', labelsize=16)
            
            # Set uniform y-axis range
            ax.set_ylim(global_y_min, global_y_max)
        
        plt.tight_layout()
        plt.savefig(os.path.join(stats_dir, f"{stat_key}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Generated faceted plot for {stat_key}")
    
    # Generate identity-related faceted plots
    generate_faceted_identity_plots(grouped_data, stats_dir, color_style_map)
    
    # Also keep original combined comparison plots (optional)
    generate_legacy_combined_plots(all_configs_stats, config_names, stats_dir, steps, stat_keys)


def generate_faceted_identity_plots(grouped_data, stats_dir, color_style_map):
    """
    Generate identity-related faceted plots
    """
    # Check if identity data exists
    has_identity_data = any(
        any("identity_1_mean_opinions" in mode_data for mode_data in group_data.values())
        for group_data in grouped_data.values()
    )
    
    if not has_identity_data:
        return
    
    # Generate identity mean opinion faceted plot
    zealot_modes = ["No Zealots", "Clustered", "Random", "High-Degree"]
    
    # First collect all identity data to calculate global y-axis range
    all_identity_values = []
    for zealot_mode in zealot_modes:
        for config_label, mode_data in grouped_data[zealot_mode].items():
            if "identity_1_mean_opinions" in mode_data:
                all_identity_values.extend(mode_data["identity_1_mean_opinions"])
            if "identity_neg1_mean_opinions" in mode_data:
                all_identity_values.extend(mode_data["identity_neg1_mean_opinions"])
    
    # Calculate global y-axis range
    if all_identity_values:
        global_y_min = min(all_identity_values)
        global_y_max = max(all_identity_values)
        y_range = global_y_max - global_y_min
        y_margin = max(0.05 * y_range, 0.01 * abs(global_y_max))
        global_y_min -= y_margin
        global_y_max += y_margin
    else:
        global_y_min, global_y_max = -1, 1
    
    fig, axes = plt.subplots(2, 2, figsize=(23, 15))
    fig.suptitle('Comparison of Mean Opinions by Identity and Zealot Mode', fontsize=24, y=0.98)
    axes_flat = axes.flatten()
    
    for i, zealot_mode in enumerate(zealot_modes):
        ax = axes_flat[i]
        
        # Plot two lines for each configuration (Identity +1 and Identity -1)
        for config_label, mode_data in grouped_data[zealot_mode].items():
            if "identity_1_mean_opinions" in mode_data and "identity_neg1_mean_opinions" in mode_data:
                color, base_linestyle = color_style_map.get(config_label, ('#666666', '-'))
                
                # Identity +1 (use base_linestyle, no marker)
                data_1 = mode_data["identity_1_mean_opinions"]
                ax.plot(
                    range(len(data_1)), 
                    data_1, 
                    label=f'{config_label} - Identity +1',
                    color=color,
                    linestyle=base_linestyle,
                    linewidth=1.5
                )
                
                # Identity -1 (use base_linestyle, add distinct marker)
                data_neg1 = mode_data["identity_neg1_mean_opinions"]
                ax.plot(
                    range(len(data_neg1)), 
                    data_neg1, 
                    label=f'{config_label} - Identity -1',
                    color=color,
                    linestyle=base_linestyle,
                    marker='o',
                    markersize=3,
                    markevery=10,  # Show a marker every 10 points
                    linewidth=1.5
                )
        
        ax.set_title(f'{zealot_mode}', fontsize=20, fontweight='bold')
        ax.set_xlabel('Step', fontsize=18)
        ax.set_ylabel('Mean Opinion', fontsize=18)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=16)
        ax.tick_params(axis='both', labelsize=16)
        
        # Set uniform y-axis range
        ax.set_ylim(global_y_min, global_y_max)
    
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, "identity_mean_opinions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate absolute value of identity opinion difference faceted plot
    
    # First collect all absolute difference data to calculate global y-axis range
    all_abs_diff_values = []
    for zealot_mode in zealot_modes:
        for config_label, mode_data in grouped_data[zealot_mode].items():
            if "identity_opinion_differences" in mode_data:
                differences = mode_data["identity_opinion_differences"]
                abs_differences = [abs(diff) for diff in differences]
                all_abs_diff_values.extend(abs_differences)
    
    # Calculate global y-axis range
    if all_abs_diff_values:
        global_y_min = min(all_abs_diff_values)
        global_y_max = max(all_abs_diff_values)
        y_range = global_y_max - global_y_min
        y_margin = max(0.05 * y_range, 0.01 * abs(global_y_max))
        global_y_min = max(0, global_y_min - y_margin)  # Absolute value cannot be less than 0
        global_y_max += y_margin
    else:
        global_y_min, global_y_max = 0, 1
    
    fig, axes = plt.subplots(2, 2, figsize=(21, 14.5))
    fig.suptitle('Comparison of Absolute Mean Opinion Differences between Identities', fontsize=24, y=0.98)
    axes_flat = axes.flatten()
    
    for i, zealot_mode in enumerate(zealot_modes):
        ax = axes_flat[i]
        
        for config_label, mode_data in grouped_data[zealot_mode].items():
            if "identity_opinion_differences" in mode_data:
                color, linestyle = color_style_map.get(config_label, ('#666666', '-'))
                
                # Calculate absolute value
                differences = mode_data["identity_opinion_differences"]
                abs_differences = [abs(diff) for diff in differences]
                
                ax.plot(
                    range(len(abs_differences)), 
                    abs_differences, 
                    label=config_label,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.5
                )
        
        ax.set_title(f'{zealot_mode}', fontsize=20, fontweight='bold')
        ax.set_xlabel('Step', fontsize=18)
        ax.set_ylabel('|Mean Opinion Difference|', fontsize=18)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=16)
        ax.tick_params(axis='both', labelsize=16)
        
        # Set uniform y-axis range
        ax.set_ylim(global_y_min, global_y_max)
    
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, "identity_opinion_differences_abs.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated identity plots")


def generate_legacy_identity_plots(all_configs_stats, config_names, legacy_dir, colors, linestyles):
    """
    Generate legacy identity-related plots
    """
    # Check if identity data exists
    has_identity_data = False
    for config_name in config_names:
        if config_name in all_configs_stats:
            for mode_key in all_configs_stats[config_name]:
                if "identity_1_mean_opinions" in all_configs_stats[config_name][mode_key]:
                    has_identity_data = True
                    break
            if has_identity_data:
                break
    
    if not has_identity_data:
        return
    
    # Plot combined comparison of average opinions for both identities
    plt.figure(figsize=(20, 10))
    for i, config_name in enumerate(config_names):
        if config_name in all_configs_stats:
            mode_key = list(all_configs_stats[config_name].keys())[0]
            if "identity_1_mean_opinions" in all_configs_stats[config_name][mode_key]:
                # Average opinion for Identity = 1 (solid line)
                data_1 = all_configs_stats[config_name][mode_key]["identity_1_mean_opinions"]
                plt.plot(
                    range(len(data_1)), 
                    data_1, 
                    label=f'{config_name} - Identity +1',
                    color=colors[i % len(colors)], 
                    linestyle='-'
                )
                # Average opinion for Identity = -1 (dashed line)
                data_neg1 = all_configs_stats[config_name][mode_key]["identity_neg1_mean_opinions"]
                plt.plot(
                    range(len(data_neg1)), 
                    data_neg1, 
                    label=f'{config_name} - Identity -1',
                    color=colors[i % len(colors)], 
                    linestyle='--'
                )
    
    plt.xlabel('Step', fontsize=20)
    plt.ylabel('Mean Opinion', fontsize=20)
    plt.title('Comparison of Mean Opinions by Identity across All Parameter Combinations', fontsize=22)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(legacy_dir, "combined_identity_mean_opinions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot combined comparison of absolute value of identity opinion differences
    plt.figure(figsize=(15, 10))
    for i, config_name in enumerate(config_names):
        if config_name in all_configs_stats:
            mode_key = list(all_configs_stats[config_name].keys())[0]
            if "identity_opinion_differences" in all_configs_stats[config_name][mode_key]:
                # Calculate absolute value
                differences = all_configs_stats[config_name][mode_key]["identity_opinion_differences"]
                abs_differences = [abs(diff) for diff in differences]
                plt.plot(
                    range(len(abs_differences)), 
                    abs_differences, 
                    label=config_name,
                    color=colors[i % len(colors)], 
                    linestyle=linestyles[i % len(linestyles)]
                )
    
    plt.xlabel('Step', fontsize=20)
    plt.ylabel('|Mean Opinion Difference|', fontsize=20)
    plt.title('Comparison of Absolute Mean Opinion Differences between Identities', fontsize=22)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(legacy_dir, "combined_identity_opinion_differences_abs.png"), dpi=300, bbox_inches='tight')
    plt.close()


def generate_legacy_combined_plots(all_configs_stats, config_names, stats_dir, steps, stat_keys):
    """
    Generate legacy combined comparison plots (all lines in one plot)
    """
    # Create subdirectory
    legacy_dir = os.path.join(stats_dir, "legacy_combined")
    if not os.path.exists(legacy_dir):
        os.makedirs(legacy_dir)
    
    # Use different colors and line styles
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(config_names))))
    linestyles = ['-', '--', '-.', ':'] * 5
    
    # Plot combined chart for each statistic
    for stat_key, stat_label, stat_title in stat_keys:
        # Check if this statistic data exists
        has_stat_data = False
        for config_name in config_names:
            if config_name in all_configs_stats:
                if "without Zealots" in all_configs_stats[config_name]:
                    if stat_key in all_configs_stats[config_name]["without Zealots"]:
                        has_stat_data = True
                        break
                elif len(all_configs_stats[config_name]) > 0:
                    mode_name = list(all_configs_stats[config_name].keys())[0]
                    if stat_key in all_configs_stats[config_name][mode_name]:
                        has_stat_data = True
                        break
        
        if not has_stat_data:
            continue
            
        plt.figure(figsize=(15, 10))
        
        # Plot a line for each parameter combination
        for i, config_name in enumerate(config_names):
            if config_name in all_configs_stats:
                if "without Zealots" in all_configs_stats[config_name]:
                    if stat_key in all_configs_stats[config_name]["without Zealots"]:
                        data = all_configs_stats[config_name]["without Zealots"][stat_key]
                        plt.plot(
                            range(len(data)), 
                            data, 
                            label=config_name,
                            color=colors[i % len(colors)], 
                            linestyle=linestyles[i % len(linestyles)]
                        )
                elif len(all_configs_stats[config_name]) > 0:
                    mode_name = list(all_configs_stats[config_name].keys())[0]
                    if stat_key in all_configs_stats[config_name][mode_name]:
                        data = all_configs_stats[config_name][mode_name][stat_key]
                        plt.plot(
                            range(len(data)), 
                            data, 
                            label=config_name,
                            color=colors[i % len(colors)], 
                            linestyle=linestyles[i % len(linestyles)]
                        )
        
        plt.xlabel('Step', fontsize=20)
        plt.ylabel(stat_label, fontsize=20)
        plt.title(f'Comparison of {stat_title} across All Parameter Combinations', fontsize=22)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
        plt.tick_params(axis='both', labelsize=16)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(legacy_dir, f"combined_{stat_key}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Also generate identity-related plots in the legacy directory
    generate_legacy_identity_plots(all_configs_stats, config_names, legacy_dir, colors, linestyles)
    
    # Save combined data to CSV file
    csv_file = os.path.join(stats_dir, "combined_statistics.csv")
    with open(csv_file, "w") as f:
        # Write header row
        f.write("step")
        for config_name in config_names:
            if config_name in all_configs_stats:
                for stat_key, _, _ in stat_keys:
                    # Check if this statistic exists
                    has_stat = False
                    if "without Zealots" in all_configs_stats[config_name]:
                        has_stat = stat_key in all_configs_stats[config_name]["without Zealots"]
                    elif len(all_configs_stats[config_name]) > 0:
                        mode_name = list(all_configs_stats[config_name].keys())[0]
                        has_stat = stat_key in all_configs_stats[config_name][mode_name]
                    
                    if has_stat:
                        f.write(f",{config_name}_{stat_key}")
        f.write("\n")
        
        # Write data
        for step in range(steps):
            f.write(f"{step}")
            
            for config_name in config_names:
                if config_name in all_configs_stats:
                    config_stats = all_configs_stats[config_name]
                    
                    # For the case where zealot_mode is "none"
                    if "without Zealots" in config_stats:
                        mode_stats = config_stats["without Zealots"]
                        for stat_key, _, _ in stat_keys:
                            if stat_key in mode_stats and step < len(mode_stats[stat_key]):
                                f.write(f",{mode_stats[stat_key][step]:.4f}")
                            elif stat_key in mode_stats:  # If the key exists but step is out of range
                                f.write(",0.0000")
                    # For other modes
                    elif len(config_stats) > 0:
                        mode_name = list(config_stats.keys())[0]
                        mode_stats = config_stats[mode_name]
                        for stat_key, _, _ in stat_keys:
                            if stat_key in mode_stats and step < len(mode_stats[stat_key]):
                                f.write(f",{mode_stats[stat_key][step]:.4f}")
                            elif stat_key in mode_stats:  # If the key exists but step is out of range
                                f.write(",0.0000")
            
            f.write("\n")
    
    print(f"Combined statistics plots and data saved to {stats_dir}")


def run_zealot_parameter_experiment(
    runs=10,
    steps=100,
    initial_scale=0.1,
    morality_rate=0.0,
    zealot_morality=False,
    identity_clustered=False,
    zealot_count=10,
    zealot_mode="random",
    base_seed=42,
    output_dir=None,
    zealot_identity_allocation=True
):
    """
    Run multiple zealot experiments with specified parameter configuration
    
    Parameters:
    runs -- Number of runs
    steps -- Number of simulation steps per run
    initial_scale -- Scaling factor for initial opinions
    morality_rate -- Proportion of moralizing non-zealot people
    zealot_morality -- Whether all zealots are moralizing
    identity_clustered -- Whether to initialize with identity-based clustering
    zealot_count -- Number of zealots
    zealot_mode -- Zealot initialization configuration
    base_seed -- Base random seed
    output_dir -- Output directory for results
    zealot_identity_allocation -- Whether to allocate zealots by identity, enabled by default, zealots are only assigned to agents with identity 1
    """
    print(f"Running zealot parameter experiment with parameters:")
    print(f"  - Morality rate: {morality_rate}")
    print(f"  - Zealot morality: {zealot_morality}")
    print(f"  - Identity clustered: {identity_clustered}")
    print(f"  - Zealot count: {zealot_count}")
    print(f"  - Zealot mode: {zealot_mode}")
    print(f"  - Runs: {runs}")
    print(f"  - Steps: {steps}")
    
    # Create results directory
    if output_dir is None:
        output_dir = f"results/zealot_parameter_exp_mor{morality_rate}_zm{zealot_morality}_id{identity_clustered}_zn{zealot_count}_zm{zealot_mode}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create separate subdirectories for each run
    run_dirs = []
    for i in range(runs):
        run_dir = os.path.join(output_dir, f"run_{i+1}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        run_dirs.append(run_dir)
    
    # Create average results directory
    avg_dir = os.path.join(output_dir, "average_results")
    if not os.path.exists(avg_dir):
        os.makedirs(avg_dir)
    
    # Run multiple experiments
    run_results = []
    
    # Mode names
    mode_names = ["without Zealots", "with Clustered Zealots", "with Random Zealots", "with High-Degree Zealots"]
    
    # Select mode to run based on zealot_mode
    if zealot_mode == "none":
        # Only run no-zealot mode
        active_mode = "without Zealots"
    elif zealot_mode == "clustered":
        active_mode = "with Clustered Zealots"
    elif zealot_mode == "random":
        active_mode = "with Random Zealots"
    elif zealot_mode == "high-degree":
        active_mode = "with High-Degree Zealots"
    else:
        raise ValueError(f"Unknown zealot mode: {zealot_mode}")
    
    # Collect opinion history from each run to generate average heatmap
    all_opinion_histories = {}
    
    # Collect statistics from each run
    all_stats = {}
    
    for i in tqdm(range(runs), desc="Running experiments"):
        # Use a different random seed for each run
        current_seed = base_seed + i
        # Use a different seed for the network structure to ensure a different network for each run
        network_seed = base_seed + i * 1000  # Use a larger interval to avoid seed conflicts
        
        # Run experiment in a separate directory, using the new built-in zealot functionality
        print(f"\nRun {i+1}/{runs} with seed {current_seed}, network_seed {network_seed}")
        
        # Add a retry mechanism to prevent failures in LFR network generation
        max_retries = 5
        retry_count = 0
        result = None
        
        while retry_count < max_retries:
            try:
                # Run the specified mode, using the new built-in zealot functionality, and pass the network seed
                result = run_zealot_experiment(
                    steps=steps,
                    initial_scale=initial_scale,
                    morality_rate=morality_rate,
                    zealot_morality=zealot_morality,
                    identity_clustered=identity_clustered,
                    num_zealots=zealot_count,
                    zealot_mode=zealot_mode,
                    seed=current_seed,
                    network_seed=network_seed + retry_count * 100,  # Use a different network seed for each retry
                    output_dir=run_dirs[i],
                    zealot_identity_allocation=zealot_identity_allocation
                )
                break  # If successful, break out of the retry loop
            except Exception as e:
                retry_count += 1
                print(f"Attempt {retry_count} failed with error: {str(e)}")
                if retry_count < max_retries:
                    print(f"Retrying with different network seed...")
                else:
                    print(f"All {max_retries} attempts failed. Skipping this run.")
                    break
        
        if result is None:
            print(f"Skipping run {i+1} due to repeated failures.")
            continue
        
        # Collect results
        run_results.append(result)
        
        # Collect statistics and opinion history
        for mode_key, mode_data in result.items():
            if mode_key not in all_opinion_histories:
                all_opinion_histories[mode_key] = []
                all_stats[mode_key] = []
            
            all_opinion_histories[mode_key].append(mode_data["opinion_history"])
            all_stats[mode_key].append(mode_data["stats"])
    
    # Calculate average statistics
    avg_stats = {}
    for mode_key, stats_list in all_stats.items():
        avg_stats[mode_key] = average_stats(stats_list)
    
    # Plot average statistics charts
    active_mode_names = list(avg_stats.keys())
    plot_average_statistics(avg_stats, active_mode_names, avg_dir, steps)
    
    # Generate average heatmaps
    generate_average_heatmaps(all_opinion_histories, active_mode_names, avg_dir)
    
    print(f"\nParameter experiment completed. Average results saved to {avg_dir}")
    return avg_stats


def create_argument_parser():
    """
    Create command-line argument parser
    """
    parser = argparse.ArgumentParser(
        description="Zealot Parameter Sweep - Run a parameter sweep experiment or generate plots from existing data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # Run the full parameter sweep experiment
  python zealot_parameter_sweep.py
  
  # Run the experiment with custom parameters
  python zealot_parameter_sweep.py --runs 10 --steps 200 --processes 4
  
  # Generate plots from existing data only
  python zealot_parameter_sweep.py --plot-only --data-dir results/zealot_parameter_sweep
  
  # Generate plots from a specific directory
  python zealot_parameter_sweep.py --plot-only --data-dir path/to/your/data
        """
    )
    
    parser.add_argument(
        '--plot-only', 
        action='store_true',
        help='Plot-only mode: generate plots from existing data without running the experiment'
    )
    
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default='results/zealot_parameter_sweep',
        help='Path to the data directory (default: results/zealot_parameter_sweep)'
    )
    
    parser.add_argument(
        '--runs', 
        type=int, 
        default=20,
        help='Number of runs for each configuration (default: 20)'
    )
    
    parser.add_argument(
        '--steps', 
        type=int, 
        default=300,
        help='Number of simulation steps for each run (default: 300)'
    )
    
    parser.add_argument(
        '--initial-scale', 
        type=float, 
        default=0.1,
        help='Scaling factor for initial opinions (default: 0.1)'
    )
    
    parser.add_argument(
        '--base-seed', 
        type=int, 
        default=42,
        help='Base random seed (default: 42)'
    )
    
    parser.add_argument(
        '--processes', 
        type=int, 
        default=None,
        help='Number of processes to use (default: None, use all CPU cores)'
    )
    
    parser.add_argument(
        '--max-size-mb', 
        type=int, 
        default=500,
        help='Maximum size limit for a single data file in MB (default: 500)'
    )
    
    parser.add_argument(
        '--no-optimize', 
        action='store_true',
        help='Disable data optimization feature'
    )
    
    parser.add_argument(
        '--essential-only', 
        action='store_true',
        help='Preserve only essential statistics to maximize memory savings'
    )
    
    return parser


def monitor_memory_usage():
    """
    Monitor memory usage of the current process
    
    Returns:
    dict -- Dictionary containing memory usage information
    """
    import psutil
    import os
    
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),  # Resident memory
            'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual memory
            'percent': process.memory_percent(),         # Memory usage percentage
            'available_mb': psutil.virtual_memory().available / (1024 * 1024)  # Available memory
        }
    except ImportError:
        # If psutil is not available, return basic info
        return {
            'rss_mb': 0,
            'vms_mb': 0,
            'percent': 0,
            'available_mb': 0
        }


def optimize_config_stats(config_stats, preserve_essential_only=False):
    """
    Optimize configuration statistics to reduce memory usage
    
    Parameters:
    config_stats -- Configuration statistics data
    preserve_essential_only -- Whether to preserve only essential statistics
    
    Returns:
    dict -- Optimized statistics data
    """
    if not config_stats:
        return config_stats
    
    # List of essential statistics
    essential_stats = [
        'mean_opinions',
        'non_zealot_variance', 
        'polarization_index',
        'identity_1_mean_opinions',
        'identity_neg1_mean_opinions',
        'identity_opinion_differences'
    ]
    
    # List of optional statistics
    optional_stats = [
        'mean_abs_opinions',
        'cluster_variance',
        'negative_counts',
        'negative_means',
        'positive_counts',
        'positive_means',
        'community_variance_history',
        'communities'
    ]
    
    optimized_stats = {}
    
    for mode_name, mode_data in config_stats.items():
        if preserve_essential_only:
            # Preserve only essential statistics
            optimized_mode_data = {}
            for stat_key in essential_stats:
                if stat_key in mode_data:
                    optimized_mode_data[stat_key] = mode_data[stat_key]
        else:
            # Preserve all data, but compress it
            optimized_mode_data = {}
            for stat_key, stat_data in mode_data.items():
                if isinstance(stat_data, (list, np.ndarray)):
                    # Convert to numpy array to save memory
                    optimized_mode_data[stat_key] = np.array(stat_data, dtype=np.float32)
                else:
                    optimized_mode_data[stat_key] = stat_data
        
        optimized_stats[mode_name] = optimized_mode_data
    
    return optimized_stats


def save_experiment_data_with_monitoring(all_configs_stats, config_names, experiment_params, output_dir, 
                                        max_size_mb=500, optimize_data=True, preserve_essential_only=False):
    """
    Save experiment data with memory monitoring and data optimization
    
    Parameters:
    all_configs_stats -- Statistical data for all configurations
    config_names -- List of configuration names
    experiment_params -- Experiment parameters
    output_dir -- Output directory
    max_size_mb -- Maximum file size limit (MB)
    optimize_data -- Whether to optimize data to reduce memory usage
    preserve_essential_only -- Whether to preserve only essential statistics
    """
    print("\n=== Memory Usage Monitoring ===")
    initial_memory = monitor_memory_usage()
    print(f"Memory usage before saving: {initial_memory['rss_mb']:.2f} MB ({initial_memory['percent']:.2f}%)")
    print(f"Available memory: {initial_memory['available_mb']:.2f} MB")
    
    # Data optimization
    if optimize_data:
        print("\n=== Data Optimization ===")
        print("Optimizing data structures to reduce memory usage...")
        
        optimized_stats = {}
        for config_name in config_names:
            if config_name in all_configs_stats:
                optimized_stats[config_name] = optimize_config_stats(
                    all_configs_stats[config_name], 
                    preserve_essential_only=preserve_essential_only
                )
        
        # Replace original data
        all_configs_stats = optimized_stats
        
        # Force garbage collection
        gc.collect()
        
        after_optimization_memory = monitor_memory_usage()
        print(f"Memory usage after optimization: {after_optimization_memory['rss_mb']:.2f} MB ({after_optimization_memory['percent']:.2f}%)")
        memory_saved = initial_memory['rss_mb'] - after_optimization_memory['rss_mb']
        print(f"Memory saved: {memory_saved:.2f} MB")
    
    # Check if there is enough memory for the save operation
    current_memory = monitor_memory_usage()
    if current_memory['available_mb'] < 1000:  # If available memory is less than 1GB
        print(f"Warning: Low available memory ({current_memory['available_mb']:.2f} MB)")
        print("Consider closing other applications or using a smaller dataset")
    
    # Call the original save function
    save_experiment_data(all_configs_stats, config_names, experiment_params, output_dir, max_size_mb)
    
    # Final memory check
    final_memory = monitor_memory_usage()
    print(f"\n=== Memory Usage After Saving ===")
    print(f"Final memory usage: {final_memory['rss_mb']:.2f} MB ({final_memory['percent']:.2f}%)")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Run different functionality based on the mode
    if args.plot_only:
        # Plot-only mode
        print("=" * 50)
        print("ZEALOT PARAMETER SWEEP - PLOT ONLY MODE")
        print("=" * 50)
        run_plot_only_mode(args.data_dir)
    else:
        # Full experiment mode
        print("=" * 50)
        print("ZEALOT PARAMETER SWEEP - FULL EXPERIMENT MODE")
        print("=" * 50)
        print(f"Configuration:")
        print(f"  - Runs per config: {args.runs}")
        print(f"  - Steps per run: {args.steps}")
        print(f"  - Initial scale: {args.initial_scale}")
        print(f"  - Base seed: {args.base_seed}")
        print(f"  - Processes: {args.processes if args.processes else 'All CPU cores'}")
        print(f"  - Output directory: {args.data_dir}")
        print(f"  - Max file size: {args.max_size_mb} MB")
        print(f"  - Data optimization: {'Disabled' if args.no_optimize else 'Enabled'}")
        print(f"  - Essential only: {'Yes' if args.essential_only else 'No'}")
        print("-" * 50)
        
        # Run the parameter sweep experiment
        # Note: On Windows, this code needs to be run inside the if __name__ == "__main__": block
        run_parameter_sweep(
            runs_per_config=args.runs,
            steps=args.steps,
            initial_scale=args.initial_scale,
            base_seed=args.base_seed,
            output_base_dir=args.data_dir,
            num_processes=args.processes,
            max_size_mb=args.max_size_mb,
            optimize_data=not args.no_optimize,
            preserve_essential_only=args.essential_only
        ) 