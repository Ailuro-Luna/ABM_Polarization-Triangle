import os
import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']  # Or 'Arial Unicode MS', etc.
# plt.rcParams['axes.unicode_minus'] = False  # This is the key setting to solve negative sign display issues
from tqdm import tqdm
import copy
import random
from polarization_triangle.experiments.zealot_experiment import run_zealot_experiment
from polarization_triangle.visualization.opinion_viz import draw_opinion_distribution_heatmap

def run_multi_zealot_experiment(
    runs=10, 
    steps=500, 
    initial_scale=0.1, 
    morality_rate=0.0,
    zealot_morality=False,
    identity_clustered=False,
    zealot_count=50,
    zealot_mode=None,
    base_seed=42,
    output_dir=None,
    zealot_identity_allocation=True
):
    """
    Run multiple zealot experiments and calculate average results
    
    Parameters:
    runs -- Number of runs
    steps -- Number of simulation steps per run
    initial_scale -- Scaling factor for initial opinions
    morality_rate -- Proportion of moralizing non-zealot people
    zealot_morality -- Whether all zealots are moralizing
    identity_clustered -- Whether to use clustered initialization by identity
    zealot_count -- Total number of zealots
    zealot_mode -- Zealot initialization configuration ("none", "clustered", "random", "high-degree"), if None run all modes
    base_seed -- Base random seed, each run will use a different seed
    output_dir -- Result output directory
    zealot_identity_allocation -- Whether to allocate zealots by identity, enabled by default, when enabled zealots are only assigned to agents with identity=1
    """
    print(f"Running multi-zealot experiment with {runs} runs...")
    print(f"Parameters: morality_rate={morality_rate}, zealot_morality={zealot_morality}, identity_clustered={identity_clustered}")
    print(f"zealot_count={zealot_count}, zealot_mode={zealot_mode}")
    
    # Create result directory
    if output_dir is None:
        # Create directory name containing parameter information
        dir_name = f"mor_{morality_rate:.1f}_zm_{'T' if zealot_morality else 'F'}_id_{'C' if identity_clustered else 'R'}"
        if zealot_mode:
            dir_name += f"_zn_{zealot_count}_zm_{zealot_mode}"
        results_dir = f"results/multi_zealot_experiment/{dir_name}"
    else:
        results_dir = output_dir
        
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create separate subdirectories for each run
    run_dirs = []
    for i in range(runs):
        run_dir = os.path.join(results_dir, f"run_{i+1}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        run_dirs.append(run_dir)
    
    # Create average results directory
    avg_dir = os.path.join(results_dir, "average_results")
    if not os.path.exists(avg_dir):
        os.makedirs(avg_dir)
        
    # Create statistics subdirectory
    stats_dir = os.path.join(avg_dir, "statistics")
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    
    # Run multiple experiments
    run_results = []
    
    # Determine modes to run
    if zealot_mode is None:
        # If no mode specified, run all modes
        mode_names = ["without Zealots", "with Clustered Zealots", "with Random Zealots", "with High-Degree Zealots"]
    else:
        # If mode specified, run only that mode
        if zealot_mode == "none":
            mode_names = ["without Zealots"]
        elif zealot_mode == "clustered":
            mode_names = ["with Clustered Zealots"]
        elif zealot_mode == "random":
            mode_names = ["with Random Zealots"]
        elif zealot_mode == "high-degree":
            mode_names = ["with High-Degree Zealots"]
        else:
            raise ValueError(f"Unknown zealot mode: {zealot_mode}")
    
    # Collect statistics from each run
    all_stats = {mode: [] for mode in mode_names}
    
    # Collect opinion history from each run for generating average heatmaps
    all_opinion_histories = {mode: [] for mode in mode_names}
    
    for i in tqdm(range(runs), desc="Running experiments"):
        # Use different random seed for each run
        current_seed = base_seed + i
        
        # Run experiment in separate directory, using new zealot functionality
        print(f"\nRun {i+1}/{runs} with seed {current_seed}")
        result = run_zealot_experiment(
            steps=steps,
            initial_scale=initial_scale,
            num_zealots=zealot_count,
            seed=current_seed,
            output_dir=run_dirs[i],
            morality_rate=morality_rate,
            zealot_morality=zealot_morality,
            identity_clustered=identity_clustered,
            zealot_mode=zealot_mode,
            zealot_identity_allocation=zealot_identity_allocation
        )
        
        # Collect results
        run_results.append(result)
        
        # Collect statistics and opinion history
        for mode in mode_names:
            if mode in result:
                all_stats[mode].append(result[mode]["stats"])
                all_opinion_histories[mode].append(result[mode]["opinion_history"])
    
    # Calculate average statistics
    avg_stats = {}
    for mode in mode_names:
        if all_stats[mode]:  # Ensure there is data
            avg_stats[mode] = average_stats(all_stats[mode])
    
    # Plot average statistics charts
    plot_average_statistics(avg_stats, mode_names, avg_dir, steps)
    
    # Generate average heatmaps
    generate_average_heatmaps(all_opinion_histories, mode_names, avg_dir)
    
    print(f"\nMulti-zealot experiment completed. Average results saved to {avg_dir}")
    return avg_stats


def average_stats(stats_list):
    """
    Calculate average statistics from multiple runs
    
    Parameters:
    stats_list -- List containing statistics from multiple runs
    
    Returns:
    dict -- Average statistics
    """
    # Initialize result dictionary
    avg_stats = {}
    
    # Check if there is data
    if not stats_list:
        return avg_stats
    
    # Get keys from first statistics data to initialize average dictionary
    stat_keys = [
        "mean_opinions", "mean_abs_opinions", "non_zealot_variance", 
        "cluster_variance", "negative_counts", "negative_means", 
        "positive_counts", "positive_means", "polarization_index",
        # Add identity-related statistics
        "identity_1_mean_opinions", "identity_neg1_mean_opinions", "identity_opinion_differences"
    ]
    
    # Initialize arrays for each statistic
    for key in stat_keys:
        if key in stats_list[0]:
            avg_stats[key] = np.zeros_like(stats_list[0][key])
    
    # Calculate sum of all runs
    for stats in stats_list:
        for key in stat_keys:
            if key in stats and key in avg_stats:
                # Ensure array lengths are consistent
                min_length = min(len(stats[key]), len(avg_stats[key]))
                avg_stats[key][:min_length] += np.array(stats[key][:min_length])
    
    # Calculate average values
    n = len(stats_list)
    for key in stat_keys:
        if key in avg_stats:
            avg_stats[key] = avg_stats[key] / n
    
    return avg_stats


def generate_average_heatmaps(all_opinion_histories, mode_names, output_dir, heatmap_config=None):
    """
    Generate average opinion distribution heatmaps
    
    Parameters:
    all_opinion_histories -- Dictionary containing opinion histories from all runs
    mode_names -- List of mode names
    output_dir -- Output directory
    heatmap_config -- Heatmap configuration dictionary containing color mapping, scale and other parameters
    """
    print("Generating average heatmaps...")
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set default heatmap configuration
    default_config = {
        'bins': 300,
        'log_scale': True,
        'cmap': 'viridis',
        'vmin': None,
        'vmax': None,
        'custom_norm': None
    }
    
    # Merge user-provided configuration
    if heatmap_config:
        default_config.update(heatmap_config)
    
    for mode in mode_names:
        # Get all opinion histories for this mode
        mode_histories = all_opinion_histories[mode]
        
        if not mode_histories:
            continue
        
        # Calculate average opinion distribution (not average opinion trajectory)
        avg_distribution = calculate_average_opinion_distribution(mode_histories, bins=default_config['bins'])

        # start_step = 900
        start_step = 0
        avg_distribution = avg_distribution[start_step:]
        
        # Plot average heatmap
        heatmap_file = os.path.join(output_dir, f"avg_{mode.lower().replace(' ', '_')}_heatmap.png")
        draw_opinion_distribution_heatmap_from_distribution(
            avg_distribution,
            f"Average Opinion Distribution Evolution {mode} (Multiple Runs)",
            heatmap_file,
            bins=default_config['bins'],
            log_scale=default_config['log_scale'],
            cmap=default_config['cmap'],
            vmin=default_config['vmin'],
            vmax=default_config['vmax'],
            custom_norm=default_config['custom_norm'],
            start_step=start_step
        )


def calculate_average_opinion_distribution(opinion_histories, bins=40):
    """
    Calculate average opinion distribution histogram from multiple runs
    
    Parameters:
    opinion_histories -- List containing opinion histories from multiple runs
    bins -- Number of bins for opinion values
    
    Returns:
    numpy.ndarray -- Average distribution histogram data, shape (time_steps, bins)
    """
    if not opinion_histories:
        return np.array([])
    
    # Get time steps from first history
    num_steps = len(opinion_histories[0])
    
    # Create opinion bins
    opinion_bins = np.linspace(-1, 1, bins + 1)
    
    # Initialize distribution data storage for all runs
    all_distributions = np.zeros((len(opinion_histories), num_steps, bins))
    
    # Calculate distribution histogram for each run
    for run_idx, history in enumerate(opinion_histories):
        for step in range(min(num_steps, len(history))):
            # Calculate opinion distribution for this time step
            hist, _ = np.histogram(history[step], bins=opinion_bins, range=(-1, 1))
            all_distributions[run_idx, step] = hist
    
    # Calculate average distribution
    avg_distribution = np.mean(all_distributions, axis=0)
    
    return avg_distribution


def draw_opinion_distribution_heatmap_from_distribution(distribution_data, title, filename, bins=40, log_scale=True,
                                                       cmap='viridis', vmin=None, vmax=None, custom_norm=None, start_step=0):
    """
    Draw heatmap from pre-calculated distribution data
    
    Parameters:
    distribution_data -- Distribution data, shape (time_steps, bins)
    title -- Chart title
    filename -- Save filename
    bins -- Number of bins for opinion values
    log_scale -- Whether to use logarithmic scale for color representation
    cmap -- Color mapping scheme ('viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'RdBu', 'hot', 'jet', etc.)
    vmin -- Minimum value for color scale, auto-determined if None
    vmax -- Maximum value for color scale, auto-determined if None
    custom_norm -- Custom color normalization object, overrides log_scale, vmin, vmax if provided
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    
    # Get number of time steps
    time_steps = distribution_data.shape[0]
    
    # Create opinion bins
    opinion_bins = np.linspace(-1, 1, bins + 1)
    
    # Create plot
    # Enlarge default font and disable Unicode minus sign to avoid rendering issues
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 26,
        'axes.labelsize': 24,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.unicode_minus': False
    })
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Create coordinates
    x = opinion_bins[:-1] + np.diff(opinion_bins) / 2  # opinion values (bin centers)
    y = np.arange(time_steps)  # time step indices (for plotting, starting from 0)
    
    # Determine color normalization
    if custom_norm is not None:
        # Use custom normalization
        norm = custom_norm
        plot_data = distribution_data
    elif log_scale:
        # Use logarithmic scale, replace 0 values with minimum non-zero value to avoid log(0) error
        min_nonzero = np.min(distribution_data[distribution_data > 0]) if np.any(distribution_data > 0) else 1
        log_data = np.copy(distribution_data)
        log_data[log_data == 0] = min_nonzero
        
        # Set range for logarithmic normalization
        log_vmin = vmin if vmin is not None else min_nonzero
        log_vmax = vmax if vmax is not None else np.max(log_data)
        norm = LogNorm(vmin=log_vmin, vmax=log_vmax)
        plot_data = log_data
    else:
        # Use linear scale
        linear_vmin = vmin if vmin is not None else np.min(distribution_data)
        linear_vmax = vmax if vmax is not None else np.max(distribution_data)
        norm = plt.Normalize(vmin=linear_vmin, vmax=linear_vmax)
        plot_data = distribution_data
    
    # Draw heatmap
    pcm = ax.pcolormesh(x, y, plot_data, norm=norm, cmap=cmap, shading='auto')
    
    # Add colorbar
    cbar = fig.colorbar(pcm, ax=ax)
    try:
        from matplotlib.ticker import LogFormatter, ScalarFormatter
        if custom_norm is None and log_scale:
            cbar.formatter = LogFormatter(base=10.0, labelOnlyBase=False)
            cbar.update_ticks()
        else:
            sf = ScalarFormatter(useMathText=True)
            sf.set_powerlimits((-2, 3))
            cbar.formatter = sf
            cbar.update_ticks()
    except Exception:
        pass
    cbar.set_label('Average Agent Count', fontsize=22)
    cbar.ax.tick_params(labelsize=20)
    
    # If specific numerical range is set, can customize colorbar ticks
    if vmin is not None and vmax is not None:
        if log_scale and not custom_norm:
            # Logarithmic scale ticks
            ticks = []
            current = vmin
            while current <= vmax:
                ticks.append(current)
                current *= 10
            if ticks[-1] < vmax:
                ticks.append(vmax)
            cbar.set_ticks(ticks)
        else:
            # Linear scale ticks
            step = (vmax - vmin) / 5
            cbar.set_ticks([vmin + i*step for i in range(6)])
    
    # Set labels and title
    ax.set_xlabel('Opinion Value', fontsize=24)
    ax.set_ylabel('Time Step', fontsize=24)
    ax.set_title(title, fontsize=26)
    ax.tick_params(axis='both', labelsize=20)
    
    # Optimize Y-axis ticks to prevent overcrowding while showing real time steps
    max_ticks = 10
    tick_step = max(1, time_steps // max_ticks)
    tick_positions = np.arange(0, time_steps, tick_step)  # positions on the plot (starting from 0)
    tick_labels = [str(start_step + pos) for pos in tick_positions]  # displayed real time steps
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    
    # Save chart
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
    # Additionally create a 3D view
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Select some time steps for display (avoid overcrowding)
    step_interval = max(1, time_steps // 20)
    selected_timesteps = np.arange(0, time_steps, step_interval)
    
    # Prepare data for 3D plot
    X, Y = np.meshgrid(x, selected_timesteps)
    selected_data = plot_data[selected_timesteps]
    
    # Draw 3D surface
    surf = ax.plot_surface(X, Y, selected_data, cmap=cmap, edgecolor='none', alpha=0.8)
    
    # Set labels and title
    ax.set_xlabel('Opinion Value', fontsize=18)
    ax.set_ylabel('Time Step', fontsize=18)
    ax.set_zlabel('Average Agent Count', fontsize=18)
    ax.set_title(f"{title} - 3D View", fontsize=20)
    ax.tick_params(axis='both', labelsize=14)
    
    # Fix 3D plot Y-axis ticks to display real time steps
    y_tick_positions = selected_timesteps[::max(1, len(selected_timesteps)//5)]
    y_tick_labels = [str(start_step + pos) for pos in y_tick_positions]
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels)
    
    # Add colorbar
    cbar3d = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar3d.set_label('Average Agent Count', fontsize=16)
    cbar3d.ax.tick_params(labelsize=12)
    
    # Save 3D plot
    waterfall_filename = filename.replace('.png', '_3d.png')
    plt.tight_layout()
    plt.savefig(waterfall_filename, dpi=300)
    plt.close()


def plot_average_statistics(avg_stats, mode_names, output_dir, steps):
    """
    Plot average statistics charts
    
    Parameters:
    avg_stats -- Average statistics data dictionary
    mode_names -- List of mode names
    output_dir -- Output directory
    steps -- Number of simulation steps
    """
    # Ensure statistics directory exists
    stats_dir = os.path.join(output_dir, "statistics")
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    
    step_values = range(steps)
    
    # Use different colors and line styles
    colors = ['blue', 'red', 'green', 'purple']
    linestyles = ['-', '--', '-.', ':']
    
    # 1. Plot average opinion value comparison chart
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(step_values, avg_stats[mode]["mean_opinions"], 
                label=f'{mode} - Mean Opinion', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Mean Opinion')
    plt.title('Average Mean Opinions across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "avg_mean_opinions.png"), dpi=300)
    plt.close()
    
    # 2. Plot average absolute opinion value comparison chart
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(step_values, avg_stats[mode]["mean_abs_opinions"], 
                label=f'{mode} - Mean |Opinion|', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Mean |Opinion|')
    plt.title('Average Mean Absolute Opinions across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "avg_mean_abs_opinions.png"), dpi=300)
    plt.close()
    
    # 3. Plot non-zealot variance comparison chart
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(step_values, avg_stats[mode]["non_zealot_variance"], 
                label=f'{mode}', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Variance')
    plt.title('Average Opinion Variance (Excluding Zealots) across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "avg_non_zealot_variance.png"), dpi=300)
    plt.close()
    
    # 4. Plot intra-community variance comparison chart
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(step_values, avg_stats[mode]["cluster_variance"], 
                label=f'{mode}', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Mean Intra-Cluster Variance')
    plt.title('Average Mean Opinion Variance(Excluding Zealots) within Clusters across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "avg_cluster_variance.png"), dpi=300)
    plt.close()
    
    # 5. Plot negative opinion count comparison chart
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(step_values, avg_stats[mode]["negative_counts"], 
                label=f'{mode}', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Count')
    plt.title('Average Negative Opinion Counts across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "avg_negative_counts.png"), dpi=300)
    plt.close()
    
    # 6. Plot negative opinion mean value comparison chart
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(step_values, avg_stats[mode]["negative_means"], 
                label=f'{mode}', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Mean Value')
    plt.title('Average Negative Opinion Means across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "avg_negative_means.png"), dpi=300)
    plt.close()
    
    # 7. Plot positive opinion count comparison chart
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(step_values, avg_stats[mode]["positive_counts"], 
                label=f'{mode}', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Count')
    plt.title('Average Positive Opinion Counts across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "avg_positive_counts.png"), dpi=300)
    plt.close()
    
    # 8. Plot positive opinion mean value comparison chart
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(step_values, avg_stats[mode]["positive_means"], 
                label=f'{mode}', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Mean Value')
    plt.title('Average Positive Opinion Means across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "avg_positive_means.png"), dpi=300)
    plt.close()
    
    # 9. Plot polarization index comparison chart (if data available)
    has_polarization_data = all(
        "polarization_index" in avg_stats[mode] and len(avg_stats[mode]["polarization_index"]) > 0 
        for mode in mode_names
    )
    
    if has_polarization_data:
        plt.figure(figsize=(12, 7))
        for i, mode in enumerate(mode_names):
            plt.plot(range(len(avg_stats[mode]["polarization_index"])), 
                    avg_stats[mode]["polarization_index"], 
                    label=f'{mode}', 
                    color=colors[i], linestyle='-')
        plt.xlabel('Step')
        plt.ylabel('Polarization Index')
        plt.title('Average Polarization Index across Different Simulations')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(stats_dir, "avg_polarization_index.png"), dpi=300)
        plt.close()
    
    # 10. New: Plot identity average opinion comparison chart
    has_identity_data = all(
        "identity_1_mean_opinions" in avg_stats[mode] and "identity_neg1_mean_opinions" in avg_stats[mode]
        for mode in mode_names
    )
    
    if has_identity_data:
        # 10a. Comparison chart of average opinions for two identity types
        plt.figure(figsize=(15, 7))
        for i, mode in enumerate(mode_names):
            # Average opinion for Identity = 1 (solid line)
            plt.plot(step_values, avg_stats[mode]["identity_1_mean_opinions"], 
                    label=f'{mode} - Identity +1', 
                    color=colors[i], linestyle='-')
            # Average opinion for Identity = -1 (dashed line)
            plt.plot(step_values, avg_stats[mode]["identity_neg1_mean_opinions"], 
                    label=f'{mode} - Identity -1', 
                    color=colors[i], linestyle='--')
        plt.xlabel('Step')
        plt.ylabel('Mean Opinion')
        plt.title('Average Mean Opinions by Identity across Different Simulations')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(stats_dir, "avg_identity_mean_opinions.png"), dpi=300)
        plt.close()
        
        # 10b. Absolute value comparison chart of identity opinion differences
        plt.figure(figsize=(12, 7))
        for i, mode in enumerate(mode_names):
            # Calculate absolute values
            abs_differences = [abs(diff) for diff in avg_stats[mode]["identity_opinion_differences"]]
            plt.plot(step_values, abs_differences, 
                    label=f'{mode}', 
                    color=colors[i], linestyle='-')
        plt.xlabel('Step')
        plt.ylabel('|Mean Opinion Difference|')
        plt.title('Average Absolute Mean Opinion Differences between Identities')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(stats_dir, "avg_identity_opinion_differences_abs.png"), dpi=300)
        plt.close()
    
    # 11. Save average statistical data to CSV file
    stats_csv = os.path.join(stats_dir, "avg_opinion_stats.csv")
    with open(stats_csv, "w") as f:
        # Write header row
        f.write("step")
        for mode in mode_names:
            f.write(f",{mode}_mean_opinion,{mode}_mean_abs_opinion,{mode}_non_zealot_variance,{mode}_cluster_variance")
            f.write(f",{mode}_negative_count,{mode}_negative_mean,{mode}_positive_count,{mode}_positive_mean")
            if "polarization_index" in avg_stats[mode] and len(avg_stats[mode]["polarization_index"]) > 0:
                f.write(f",{mode}_polarization_index")
            # Add identity-related columns
            if "identity_1_mean_opinions" in avg_stats[mode]:
                f.write(f",{mode}_identity_1_mean_opinion,{mode}_identity_neg1_mean_opinion,{mode}_identity_opinion_difference")
        f.write("\n")
        
        # Write data
        for step in range(steps):
            f.write(f"{step}")
            for mode in mode_names:
                f.write(f",{avg_stats[mode]['mean_opinions'][step]:.4f}")
                f.write(f",{avg_stats[mode]['mean_abs_opinions'][step]:.4f}")
                f.write(f",{avg_stats[mode]['non_zealot_variance'][step]:.4f}")
                f.write(f",{avg_stats[mode]['cluster_variance'][step]:.4f}")
                f.write(f",{avg_stats[mode]['negative_counts'][step]:.1f}")
                f.write(f",{avg_stats[mode]['negative_means'][step]:.4f}")
                f.write(f",{avg_stats[mode]['positive_counts'][step]:.1f}")
                f.write(f",{avg_stats[mode]['positive_means'][step]:.4f}")
                # Add polarization index data to CSV if available
                if "polarization_index" in avg_stats[mode] and step < len(avg_stats[mode]["polarization_index"]):
                    f.write(f",{avg_stats[mode]['polarization_index'][step]:.4f}")
                # Add identity-related data
                if "identity_1_mean_opinions" in avg_stats[mode] and step < len(avg_stats[mode]["identity_1_mean_opinions"]):
                    f.write(f",{avg_stats[mode]['identity_1_mean_opinions'][step]:.4f}")
                    f.write(f",{avg_stats[mode]['identity_neg1_mean_opinions'][step]:.4f}")
                    f.write(f",{avg_stats[mode]['identity_opinion_differences'][step]:.4f}")
            f.write("\n")


if __name__ == "__main__":
    # Run multiple zealot experiments
    run_multi_zealot_experiment(
        runs=10,                  # Run 10 experiments
        steps=100,                # 100 steps per run
        initial_scale=0.1,        # Scale initial opinions to 10%
        morality_rate=0.2,        # Moralizing ratio is 20%
        zealot_morality=True,     # All zealots are moralizing
        identity_clustered=True,  # Clustered initialization by identity
        zealot_count=10,          # 10 zealots
        zealot_mode="clustered",  # Use clustered zealot mode
        base_seed=42              # Base random seed
    ) 