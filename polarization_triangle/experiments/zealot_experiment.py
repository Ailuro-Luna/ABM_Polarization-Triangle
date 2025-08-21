import numpy as np
import os
import copy
import time
import matplotlib.pyplot as plt
import networkx as nx
from polarization_triangle.core.config import SimulationConfig, base_config, high_polarization_config
from polarization_triangle.core.simulation import Simulation
from polarization_triangle.visualization.network_viz import draw_network
from polarization_triangle.visualization.opinion_viz import draw_opinion_distribution_heatmap
from polarization_triangle.visualization.rule_viz import (
    draw_interaction_type_usage, 
    draw_interaction_type_cumulative_usage
)
from polarization_triangle.visualization.activation_viz import (
    draw_activation_components,
    draw_activation_history,
    draw_activation_heatmap,
    draw_activation_trajectory
)


def generate_rule_usage_plots(sim, title_prefix, output_dir):
    """
    Generate rule usage statistics plots
    
    Parameters:
    sim -- simulation instance
    title_prefix -- plot title prefix
    output_dir -- output directory
    """
    # Draw rule usage statistics plot
    rule_usage_path = os.path.join(output_dir, f"{title_prefix}_interaction_types.png")
    draw_interaction_type_usage(
        sim.rule_counts_history,
        f"Interaction Types over Time\n{title_prefix}",
        rule_usage_path
    )
    
    # Draw cumulative rule usage statistics plot
    rule_cumulative_path = os.path.join(output_dir, f"{title_prefix}_interaction_types_cumulative.png")
    draw_interaction_type_cumulative_usage(
        sim.rule_counts_history,
        f"Cumulative Interaction Types\n{title_prefix}",
        rule_cumulative_path
    )
    
    # Output rule usage statistics information
    interaction_names = [
        "Rule 1: Same dir, Same ID, {0,0}, High Convergence",
        "Rule 2: Same dir, Same ID, {0,1}, Medium Pull",
        "Rule 3: Same dir, Same ID, {1,0}, Medium Pull",
        "Rule 4: Same dir, Same ID, {1,1}, High Polarization",
        "Rule 5: Same dir, Diff ID, {0,0}, Medium Convergence",
        "Rule 6: Same dir, Diff ID, {0,1}, Low Pull",
        "Rule 7: Same dir, Diff ID, {1,0}, Low Pull",
        "Rule 8: Same dir, Diff ID, {1,1}, Medium Polarization",
        "Rule 9: Diff dir, Same ID, {0,0}, Very High Convergence",
        "Rule 10: Diff dir, Same ID, {0,1}, Medium Convergence/Pull",
        "Rule 11: Diff dir, Same ID, {1,0}, Low Resistance",
        "Rule 12: Diff dir, Same ID, {1,1}, Low Polarization",
        "Rule 13: Diff dir, Diff ID, {0,0}, Low Convergence",
        "Rule 14: Diff dir, Diff ID, {0,1}, High Pull",
        "Rule 15: Diff dir, Diff ID, {1,0}, High Resistance",
        "Rule 16: Diff dir, Diff ID, {1,1}, Very High Polarization"
    ]
    
    # Get interaction type statistics
    interaction_stats = sim.get_interaction_counts()
    counts = interaction_stats["counts"]
    total_count = interaction_stats["total_interactions"]
    
    # Write interaction type statistics to file
    stats_path = os.path.join(output_dir, f"{title_prefix}_interaction_types_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"Interaction type statistics - {title_prefix}\n")
        f.write("-" * 50 + "\n")
        for i, interaction_name in enumerate(interaction_names):
            count = counts[i]
            percent = (count / total_count) * 100 if total_count > 0 else 0
            f.write(f"{interaction_name}: {count} times ({percent:.1f}%)\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total: {total_count} times\n")


def generate_activation_visualizations(sim, trajectory, title_prefix, output_dir):
    """
    Generate activation component related visualizations
    
    Parameters:
    sim -- simulation instance
    trajectory -- opinion trajectory data
    title_prefix -- plot title prefix
    output_dir -- output directory
    """
    # Create activation components subfolder
    activation_folder = os.path.join(output_dir, "activation_components")
    if not os.path.exists(activation_folder):
        os.makedirs(activation_folder)
    
    # 1. Self-activation and social influence scatter plot
    components_path = os.path.join(activation_folder, f"{title_prefix}_activation_components.png")
    draw_activation_components(
        sim,
        f"Activation Components\n{title_prefix}",
        components_path
    )
    
    # 2. Change of self-activation and social influence over time
    history_path = os.path.join(activation_folder, f"{title_prefix}_activation_history.png")
    draw_activation_history(
        sim,
        f"Activation History\n{title_prefix}",
        history_path
    )
    
    # 3. Heatmap of self-activation and social influence
    heatmap_path = os.path.join(activation_folder, f"{title_prefix}_activation_heatmap.png")
    draw_activation_heatmap(
        sim,
        f"Activation Heatmap\n{title_prefix}",
        heatmap_path
    )
    
    # 4. Activation trajectory of selected agents
    trajectory_path = os.path.join(activation_folder, f"{title_prefix}_activation_trajectory.png")
    draw_activation_trajectory(
        sim,
        trajectory,
        f"Activation Trajectories\n{title_prefix}",
        trajectory_path
    )
    
    # 5. Save activation component data to CSV file
    components = sim.get_activation_components()
    data_path = os.path.join(activation_folder, f"{title_prefix}_activation_data.csv")
    with open(data_path, "w") as f:
        f.write("agent_id,identity,morality,opinion,self_activation,social_influence,total_activation\n")
        for i in range(sim.num_agents):
            f.write(f"{i},{sim.identities[i]},{sim.morals[i]},{sim.opinions[i]:.4f}")
            f.write(f",{components['self_activation'][i]:.4f},{components['social_influence'][i]:.4f}")
            f.write(f",{components['self_activation'][i] + components['social_influence'][i]:.4f}\n")


def generate_opinion_statistics(sim, trajectory, zealot_ids, mode_name, results_dir):
    """
    Calculate various opinion statistics without plotting
    
    Parameters:
    sim -- simulation instance
    trajectory -- opinion trajectory data
    zealot_ids -- zealot ID list
    mode_name -- mode name
    results_dir -- result output directory
    
    Returns:
    dict -- dictionary containing various statistical data
    """
    num_steps = len(trajectory)
    
    # 1. Calculate average opinion and average abs(opinion)
    mean_opinions = []
    mean_abs_opinions = []
    
    for step_opinions in trajectory:
        mean_opinions.append(np.mean(step_opinions))
        mean_abs_opinions.append(np.mean(np.abs(step_opinions)))
    
    # 2. Calculate opinion variance of all agents excluding zealots
    non_zealot_var = []
    
    for step_opinions in trajectory:
        # Create opinion array of all agents excluding zealots
        non_zealot_opinions = np.delete(step_opinions, zealot_ids) if zealot_ids else step_opinions
        non_zealot_var.append(np.var(non_zealot_opinions))
    
    # 3. Calculate opinion variance of agents within all clusters (excluding zealots)
    # Get community information
    communities = {}
    for node in sim.graph.nodes():
        community = sim.graph.nodes[node].get("community")
        if isinstance(community, (set, frozenset)):
            community = min(community)
        if community not in communities:
            communities[community] = []
        communities[community].append(node)
    
    cluster_variances = []
    # New: track variance history for each community
    community_variance_history = {}
    
    for step_opinions in trajectory:
        # Calculate variance within each community, then take average
        step_cluster_vars = []
        
        for community_id, members in communities.items():
            # Filter out zealots
            community_non_zealots = [m for m in members if m not in zealot_ids]
            if community_non_zealots:  # Ensure community has non-zealot members
                community_opinions = step_opinions[community_non_zealots]
                community_var = np.var(community_opinions)
                step_cluster_vars.append(community_var)
                
                # Record this community's variance
                if community_id not in community_variance_history:
                    community_variance_history[community_id] = []
                community_variance_history[community_id].append(community_var)
            else:
                # If community has only zealots, record 0 variance
                if community_id not in community_variance_history:
                    community_variance_history[community_id] = []
                community_variance_history[community_id].append(0)
        
        # If there are valid community variances, calculate average
        if step_cluster_vars:
            cluster_variances.append(np.mean(step_cluster_vars))
        else:
            cluster_variances.append(0)
    
    # 4. Count individuals with negative opinions and calculate mean of negative opinions
    # 5. Count individuals with positive opinions and calculate mean of positive opinions
    negative_counts = []
    negative_means = []
    positive_counts = []
    positive_means = []
    
    for step_opinions in trajectory:
        # Get opinions of non-zealots
        non_zealot_opinions = np.delete(step_opinions, zealot_ids) if zealot_ids else step_opinions
        
        # Count negative opinions
        negative_mask = non_zealot_opinions < 0
        negative_opinions = non_zealot_opinions[negative_mask]
        negative_count = len(negative_opinions)
        negative_counts.append(negative_count)
        negative_means.append(np.mean(negative_opinions) if negative_count > 0 else 0)
        
        # Count positive opinions
        positive_mask = non_zealot_opinions > 0
        positive_opinions = non_zealot_opinions[positive_mask]
        positive_count = len(positive_opinions)
        positive_counts.append(positive_count)
        positive_means.append(np.mean(positive_opinions) if positive_count > 0 else 0)
    
    # 6. New: Calculate average opinion of each identity type at each time step and their difference
    identity_1_mean_opinions = []
    identity_neg1_mean_opinions = []
    identity_opinion_differences = []
    
    # Find agents with identity 1 and -1 (excluding zealots)
    identity_1_agents = []
    identity_neg1_agents = []
    
    for i in range(sim.num_agents):
        if zealot_ids and i in zealot_ids:
            continue  # Skip zealots
        if sim.identities[i] == 1:
            identity_1_agents.append(i)
        elif sim.identities[i] == -1:
            identity_neg1_agents.append(i)
    
    for step_opinions in trajectory:
        # Calculate average opinion of agents with identity=1
        if identity_1_agents:
            identity_1_opinions = step_opinions[identity_1_agents]
            identity_1_mean = np.mean(identity_1_opinions)
        else:
            identity_1_mean = 0.0
        identity_1_mean_opinions.append(identity_1_mean)
        
        # Calculate average opinion of agents with identity=-1
        if identity_neg1_agents:
            identity_neg1_opinions = step_opinions[identity_neg1_agents]
            identity_neg1_mean = np.mean(identity_neg1_opinions)
        else:
            identity_neg1_mean = 0.0
        identity_neg1_mean_opinions.append(identity_neg1_mean)
        
        # Calculate difference between average opinions of the two identities
        difference = identity_1_mean - identity_neg1_mean
        identity_opinion_differences.append(difference)

    # Get polarization index history
    polarization_history = sim.get_polarization_history() if hasattr(sim, 'get_polarization_history') else []
    
    # Integrate all statistical data
    stats = {
        "mean_opinions": mean_opinions,
        "mean_abs_opinions": mean_abs_opinions,
        "non_zealot_variance": non_zealot_var,
        "cluster_variance": cluster_variances,
        "negative_counts": negative_counts,
        "negative_means": negative_means,
        "positive_counts": positive_counts,
        "positive_means": positive_means,
        "community_variance_history": community_variance_history,
        "communities": communities,
        "polarization_index": polarization_history,  # Add polarization index history
        # New identity-related statistics
        "identity_1_mean_opinions": identity_1_mean_opinions,
        "identity_neg1_mean_opinions": identity_neg1_mean_opinions,
        "identity_opinion_differences": identity_opinion_differences
    }
    
    # Save statistics data for each mode separately to CSV file
    stats_dir = os.path.join(results_dir, "statistics")
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
        
    file_prefix = mode_name.lower().replace(' ', '_')
    stats_csv = os.path.join(stats_dir, f"{file_prefix}_opinion_stats.csv")
    with open(stats_csv, "w") as f:
        f.write("step,mean_opinion,mean_abs_opinion,non_zealot_variance,cluster_variance,")
        f.write("negative_count,negative_mean,positive_count,positive_mean")
        if polarization_history:
            f.write(",polarization_index")  # Add polarization index column
        # Add identity-related columns
        f.write(",identity_1_mean_opinion,identity_neg1_mean_opinion,identity_opinion_difference")
        f.write("\n")
        
        for step in range(num_steps):
            f.write(f"{step},{mean_opinions[step]:.4f},{mean_abs_opinions[step]:.4f},")
            f.write(f"{non_zealot_var[step]:.4f},{cluster_variances[step]:.4f},")
            f.write(f"{negative_counts[step]},{negative_means[step]:.4f},")
            f.write(f"{positive_counts[step]},{positive_means[step]:.4f}")
            # Add polarization index data to CSV if available
            if polarization_history and step < len(polarization_history):
                f.write(f",{polarization_history[step]:.4f}")
            # Add identity-related data
            f.write(f",{identity_1_mean_opinions[step]:.4f}")
            f.write(f",{identity_neg1_mean_opinions[step]:.4f}")
            f.write(f",{identity_opinion_differences[step]:.4f}")
            f.write("\n")
    
    # Save variance data for each community to separate CSV file
    community_csv = os.path.join(stats_dir, f"{file_prefix}_community_variances.csv")
    with open(community_csv, "w") as f:
        # Write header row
        f.write("step")
        for community_id in sorted(community_variance_history.keys()):
            f.write(f",community_{community_id}")
        f.write("\n")
        
        # Write data
        for step in range(num_steps):
            f.write(f"{step}")
            for community_id in sorted(community_variance_history.keys()):
                if step < len(community_variance_history[community_id]):
                    f.write(f",{community_variance_history[community_id][step]:.4f}")
                else:
                    f.write(",0.0000")  # Prevent index out of bounds
            f.write("\n")
    
    return stats


def plot_community_variances(stats, mode_name, results_dir):
    """
    Plot opinion variance change for each community
    
    Parameters:
    stats -- Statistics data dictionary
    mode_name -- Mode name
    results_dir -- Results output directory
    """
    # Ensure statistics directory exists
    stats_dir = os.path.join(results_dir, "statistics")
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    
    file_prefix = mode_name.lower().replace(' ', '_')
    community_variance_history = stats["community_variance_history"]
    communities = stats["communities"]
    
    # If there are too many communities, the plot might be messy, so limit to show only large communities
    # Calculate size of each community
    community_sizes = {comm_id: len(members) for comm_id, members in communities.items()}
    
    # Sort communities by size
    sorted_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)
    
    # Select top 10 largest communities (or all, if fewer than 10)
    top_communities = [comm_id for comm_id, size in sorted_communities[:min(10, len(sorted_communities))]]
    
    # Plot community variance chart
    plt.figure(figsize=(12, 8))
    
    # Use different colors and line styles
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_communities)))
    linestyles = ['-', '--', '-.', ':'] * 3  # Repeat several times to ensure sufficient line styles
    
    for i, community_id in enumerate(top_communities):
        variance_history = community_variance_history[community_id]
        community_size = community_sizes[community_id]
        plt.plot(range(len(variance_history)), variance_history, 
                label=f'Community {community_id} (size: {community_size})', 
                color=colors[i], linestyle=linestyles[i % len(linestyles)])
    
    plt.xlabel('Step')
    plt.ylabel('Variance')
    plt.title(f'Opinion Variance within Each Community (Excluding Zealots)\n{mode_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, f"{file_prefix}_community_variances.png"), dpi=300)
    plt.close()


def plot_comparative_statistics(all_stats, mode_names, results_dir):
    """
    Plot comparative statistical charts, displaying statistical data from different modes on the same chart
    
    Parameters:
    all_stats -- Dictionary containing statistical data from different modes
    mode_names -- List of names for different modes
    results_dir -- Results output directory
    """
    # Ensure statistics directory exists
    stats_dir = os.path.join(results_dir, "statistics")
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    
    num_steps = len(all_stats[mode_names[0]]["mean_opinions"])
    steps = range(num_steps)
    
    # Use different colors and line styles
    colors = ['blue', 'red', 'green', 'purple']
    linestyles = ['-', '--', '-.', ':']
    
    # 1. Plot average opinion value comparison chart
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(steps, all_stats[mode]["mean_opinions"], 
                label=f'{mode} - Mean Opinion', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Mean Opinion')
    plt.title('Comparison of Mean Opinions across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "comparison_mean_opinions.png"), dpi=300)
    plt.close()
    
    # 2. Plot average absolute opinion value comparison chart
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(steps, all_stats[mode]["mean_abs_opinions"], 
                label=f'{mode} - Mean |Opinion|', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Mean |Opinion|')
    plt.title('Comparison of Mean Absolute Opinions across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "comparison_mean_abs_opinions.png"), dpi=300)
    plt.close()
    
    # 3. Plot non-zealot variance comparison chart
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(steps, all_stats[mode]["non_zealot_variance"], 
                label=f'{mode}', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Variance')
    plt.title('Comparison of Opinion Variance (Excluding Zealots) across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "comparison_non_zealot_variance.png"), dpi=300)
    plt.close()
    
    # 4. Plot intra-community variance comparison chart
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(steps, all_stats[mode]["cluster_variance"], 
                label=f'{mode}', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Mean Intra-Cluster Variance')
    plt.title('Comparison of Mean Opinion Variance within Clusters across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "comparison_cluster_variance.png"), dpi=300)
    plt.close()
    
    # 5. Plot negative opinion count comparison chart
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(steps, all_stats[mode]["negative_counts"], 
                label=f'{mode}', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Count')
    plt.title('Comparison of Negative Opinion Counts across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "comparison_negative_counts.png"), dpi=300)
    plt.close()
    
    # 6. Plot negative opinion mean value comparison chart
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(steps, all_stats[mode]["negative_means"], 
                label=f'{mode}', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Mean Value')
    plt.title('Comparison of Negative Opinion Means across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "comparison_negative_means.png"), dpi=300)
    plt.close()
    
    # 7. Plot positive opinion count comparison chart
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(steps, all_stats[mode]["positive_counts"], 
                label=f'{mode}', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Count')
    plt.title('Comparison of Positive Opinion Counts across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "comparison_positive_counts.png"), dpi=300)
    plt.close()
    
    # 8. Plot positive opinion mean value comparison chart
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(steps, all_stats[mode]["positive_means"], 
                label=f'{mode}', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Mean Value')
    plt.title('Comparison of Positive Opinion Means across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "comparison_positive_means.png"), dpi=300)
    plt.close()
    
    # 9. Plot polarization index comparison chart (if data available)
    has_polarization_data = all(
        "polarization_index" in all_stats[mode] and len(all_stats[mode]["polarization_index"]) > 0 
        for mode in mode_names
    )
    
    if has_polarization_data:
        plt.figure(figsize=(12, 7))
        for i, mode in enumerate(mode_names):
            plt.plot(steps[:len(all_stats[mode]["polarization_index"])], 
                    all_stats[mode]["polarization_index"], 
                    label=f'{mode}', 
                    color=colors[i], linestyle='-')
        plt.xlabel('Step')
        plt.ylabel('Polarization Index')
        plt.title('Comparison of Polarization Index across Different Simulations')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(stats_dir, "comparison_polarization_index.png"), dpi=300)
        plt.close()
    
    # 10. New: Plot identity average opinion comparison chart
    has_identity_data = all(
        "identity_1_mean_opinions" in all_stats[mode] and "identity_neg1_mean_opinions" in all_stats[mode]
        for mode in mode_names
    )
    
    if has_identity_data:
        # 10a. Comparison chart of average opinions for two identity types
        plt.figure(figsize=(15, 7))
        for i, mode in enumerate(mode_names):
            # Average opinion for Identity = 1 (solid line)
            plt.plot(steps, all_stats[mode]["identity_1_mean_opinions"], 
                    label=f'{mode} - Identity +1', 
                    color=colors[i], linestyle='-')
            # Average opinion for Identity = -1 (dashed line)
            plt.plot(steps, all_stats[mode]["identity_neg1_mean_opinions"], 
                    label=f'{mode} - Identity -1', 
                    color=colors[i], linestyle='--')
        plt.xlabel('Step')
        plt.ylabel('Mean Opinion')
        plt.title('Comparison of Mean Opinions by Identity across Different Simulations')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(stats_dir, "comparison_identity_mean_opinions.png"), dpi=300)
        plt.close()
        
        # 10b. Absolute value comparison chart of identity opinion differences
        plt.figure(figsize=(12, 7))
        for i, mode in enumerate(mode_names):
            # Calculate absolute values
            abs_differences = [abs(diff) for diff in all_stats[mode]["identity_opinion_differences"]]
            plt.plot(steps, abs_differences, 
                    label=f'{mode}', 
                    color=colors[i], linestyle='-')
        plt.xlabel('Step')
        plt.ylabel('|Mean Opinion Difference|')
        plt.title('Comparison of Absolute Mean Opinion Differences between Identities')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(stats_dir, "comparison_identity_opinion_differences_abs.png"), dpi=300)
        plt.close()
    
    # 11. Save combined data to CSV file
    stats_csv = os.path.join(stats_dir, "comparison_opinion_stats.csv")
    with open(stats_csv, "w") as f:
        # Write header row
        f.write("step")
        for mode in mode_names:
            f.write(f",{mode}_mean_opinion,{mode}_mean_abs_opinion,{mode}_non_zealot_variance,{mode}_cluster_variance")
            f.write(f",{mode}_negative_count,{mode}_negative_mean,{mode}_positive_count,{mode}_positive_mean")
            if "polarization_index" in all_stats[mode] and len(all_stats[mode]["polarization_index"]) > 0:
                f.write(f",{mode}_polarization_index")
            # Add identity-related columns
            if "identity_1_mean_opinions" in all_stats[mode]:
                f.write(f",{mode}_identity_1_mean_opinion,{mode}_identity_neg1_mean_opinion,{mode}_identity_opinion_difference")
        f.write("\n")
        
        # Write data
        for step in range(num_steps):
            f.write(f"{step}")
            for mode in mode_names:
                f.write(f",{all_stats[mode]['mean_opinions'][step]:.4f}")
                f.write(f",{all_stats[mode]['mean_abs_opinions'][step]:.4f}")
                f.write(f",{all_stats[mode]['non_zealot_variance'][step]:.4f}")
                f.write(f",{all_stats[mode]['cluster_variance'][step]:.4f}")
                f.write(f",{all_stats[mode]['negative_counts'][step]}")
                f.write(f",{all_stats[mode]['negative_means'][step]:.4f}")
                f.write(f",{all_stats[mode]['positive_counts'][step]}")
                f.write(f",{all_stats[mode]['positive_means'][step]:.4f}")
                # Add polarization index data to CSV if available
                if "polarization_index" in all_stats[mode] and step < len(all_stats[mode]["polarization_index"]):
                    f.write(f",{all_stats[mode]['polarization_index'][step]:.4f}")
                # Add identity-related data
                if "identity_1_mean_opinions" in all_stats[mode] and step < len(all_stats[mode]["identity_1_mean_opinions"]):
                    f.write(f",{all_stats[mode]['identity_1_mean_opinions'][step]:.4f}")
                    f.write(f",{all_stats[mode]['identity_neg1_mean_opinions'][step]:.4f}")
                    f.write(f",{all_stats[mode]['identity_opinion_differences'][step]:.4f}")
            f.write("\n")


def run_simulation_and_generate_results(sim, zealot_ids, mode_name, results_dir, steps):
    """
    Run a single simulation and generate all visualization results
    
    Parameters:
    sim -- simulation instance
    zealot_ids -- list of zealot IDs
    mode_name -- Mode name
    results_dir -- Results output directory
    steps -- number of simulation steps
    
    Returns:
    dict -- dictionary containing opinion history and statistical data
    """
    # Generate initial state network graph
    file_prefix = mode_name.lower().replace(' ', '_')
    
    # Draw initial opinion network graph
    draw_network(
        sim, 
        "opinion", 
        f"Initial Opinion Network {mode_name}", 
        f"{results_dir}/{file_prefix}_initial_opinion_network.png"
    )
    
    # Draw initial morality network graph
    draw_network(
        sim, 
        "morality", 
        f"Initial Morality Network {mode_name}", 
        f"{results_dir}/{file_prefix}_initial_morality_network.png"
    )
    
    # Draw initial identity network graph
    draw_network(
        sim, 
        "identity", 
        f"Initial Identity Network {mode_name}", 
        f"{results_dir}/{file_prefix}_initial_identity_network.png"
    )
    
    # Store opinion history and trajectory
    opinion_history = []
    trajectory = []

    # Run simulation
    for _ in range(steps):
        # Update zealot opinions
        if zealot_ids:
            sim.set_zealot_opinions()

        # Record opinion history and trajectory
        opinion_history.append(sim.opinions.copy())
        trajectory.append(sim.opinions.copy())
        
        # Execute simulation step (zealot opinions will be automatically reset in step)
        sim.step()
        
    
    # Generate heatmap
    draw_opinion_distribution_heatmap(
        opinion_history, 
        f"Opinion Evolution {mode_name}", 
        f"{results_dir}/{file_prefix}_heatmap.png"
    )
    
    # Draw final network graph - opinion distribution
    draw_network(
        sim, 
        "opinion", 
        f"Final Opinion Network {mode_name}", 
        f"{results_dir}/{file_prefix}_final_opinion_network.png"
    )
    
    # Draw zealot network graph
    draw_zealot_network(
        sim, 
        zealot_ids, 
        f"Network {mode_name}", 
        f"{results_dir}/{file_prefix}_network.png"
    )
    
    # Generate rule usage statistics plot
    generate_rule_usage_plots(sim, mode_name, results_dir)
    
    # Generate activation component visualization
    generate_activation_visualizations(sim, trajectory, mode_name, results_dir)
    
    # Calculate opinion statistics
    stats = generate_opinion_statistics(sim, trajectory, zealot_ids, mode_name, results_dir)
    
    # Draw community variance plot
    plot_community_variances(stats, mode_name, results_dir)
    
    return {
        "opinion_history": opinion_history,
        "stats": stats
    }


def run_zealot_experiment(
    steps=500, 
    initial_scale=0.1, 
    num_zealots=50, 
    seed=42, 
    output_dir=None, 
    morality_rate=0.0, 
    zealot_morality=False, 
    identity_clustered=False,
    zealot_mode=None,
    zealot_identity_allocation=True,
    network_seed=None
):
    """
    Run zealot experiment, comparing the effects of no zealots, clustered zealots, and random zealots
    
    Parameters:
    steps -- number of simulation steps
    initial_scale -- scaling factor for initial opinions, simulating neutral attitude towards new issues
    num_zealots -- total number of zealots
    seed -- random seed
    output_dir -- results output directory (if None, use default directory)
    morality_rate -- proportion of moralizing non-zealot people
    zealot_morality -- whether all zealots are moralizing
    identity_clustered -- whether to use clustered initialization by identity
    zealot_mode -- zealot initialization configuration ("none", "clustered", "random", "high-degree"), if None run all modes
    zealot_identity_allocation -- whether to allocate zealots by identity, enabled by default, when enabled zealots are only assigned to agents with identity=1
    network_seed -- random seed for network generation, if None use seed
    
    Returns:
    dict -- dictionary containing results from all modes
    """
    # Record start time
    start_time = time.time()
    
    # Set random seed
    np.random.seed(seed)
    
    # Create base simulation instance
    base_config = copy.deepcopy(high_polarization_config)
    base_config.cluster_identity = identity_clustered
    base_config.cluster_morality = False  # Temporarily do not cluster morality
    base_config.cluster_opinion = False
    base_config.opinion_distribution = "uniform"
    base_config.alpha = 0.4
    base_config.beta = 0.12
    
    # Set network seed, add retry mechanism to prevent LFR network generation failure
    if network_seed is not None:
        base_config.network_params["seed"] = network_seed
    
    # Set moralization rate
    base_config.morality_rate = morality_rate

    print(base_config)
    
    # Add network generation retry mechanism
    max_network_retries = 5
    network_retry_count = 0
    base_sim = None
    
    while network_retry_count < max_network_retries:
        try:
            base_sim = Simulation(base_config)
            break  # Exit loop if successfully created
        except Exception as e:
            network_retry_count += 1
            print(f"Network generation attempt {network_retry_count} failed: {str(e)}")
            if network_retry_count < max_network_retries:
                # Adaptively adjust parameters to improve success rate / accelerate failure
                base_config.network_params["min_community"] = min(
                    base_config.network_params.get("min_community", 10) + 10, base_config.num_agents // 2
                )
                base_config.network_params["mu"] = max(base_config.network_params.get("mu", 0.1), 0.3)
                base_config.network_params["max_iters"] = 200
                if network_seed is not None:
                    base_config.network_params["seed"] = network_seed + network_retry_count * 100
                else:
                    base_config.network_params["seed"] = seed + network_retry_count * 100
                print(f"Retrying with network seed: {base_config.network_params['seed']} "
                    f"(min_community={base_config.network_params['min_community']}, mu={base_config.network_params['mu']})")
            else:
                print(f"All {max_network_retries} network generation attempts failed.")
                raise e
    
    if base_sim is None:
        raise RuntimeError("Failed to create base simulation after multiple attempts")
    
    # Scale initial opinions of all agents
    base_sim.opinions *= initial_scale
    
    # Determine which modes to run based on zealot_mode
    run_all_modes = zealot_mode is None
    modes_to_run = []
    
    if run_all_modes or zealot_mode == "none":
        modes_to_run.append("none")
    if run_all_modes or zealot_mode == "clustered":
        modes_to_run.append("clustered")
    if run_all_modes or zealot_mode == "random":
        modes_to_run.append("random")
    if run_all_modes or zealot_mode == "high-degree":
        modes_to_run.append("high-degree")
    
    # Create copies for different zealot distributions
    sims = {}
    zealots = {}
    
    # For no-zealot mode, use base_sim
    if "none" in modes_to_run:
        sims["none"] = base_sim
        zealots["none"] = []
    
    # For other modes, create copies and set zealot configuration
    if "clustered" in modes_to_run:
        clustered_config = copy.deepcopy(base_config)
        clustered_config.enable_zealots = True
        clustered_config.zealot_count = num_zealots
        clustered_config.zealot_mode = "clustered"
        clustered_config.zealot_morality = zealot_morality
        clustered_config.zealot_identity_allocation = zealot_identity_allocation
        # Ensure network seed consistency
        if network_seed is not None:
            clustered_config.network_params["seed"] = network_seed
        sims["clustered"] = Simulation(clustered_config)
        sims["clustered"].opinions *= initial_scale
        sims["clustered"].set_zealot_opinions()  # Reset zealot opinions to avoid scaling
        zealots["clustered"] = sims["clustered"].get_zealot_ids()
    
    if "random" in modes_to_run:
        random_config = copy.deepcopy(base_config)
        random_config.enable_zealots = True
        random_config.zealot_count = num_zealots
        random_config.zealot_mode = "random"
        random_config.zealot_morality = zealot_morality
        random_config.zealot_identity_allocation = zealot_identity_allocation
        # Ensure network seed consistency
        if network_seed is not None:
            random_config.network_params["seed"] = network_seed
        sims["random"] = Simulation(random_config)
        sims["random"].opinions *= initial_scale
        sims["random"].set_zealot_opinions()  # Reset zealot opinions to avoid scaling
        zealots["random"] = sims["random"].get_zealot_ids()
    
    if "high-degree" in modes_to_run:
        degree_config = copy.deepcopy(base_config)
        degree_config.enable_zealots = True
        degree_config.zealot_count = num_zealots
        degree_config.zealot_mode = "degree"
        degree_config.zealot_morality = zealot_morality
        degree_config.zealot_identity_allocation = zealot_identity_allocation
        # Ensure network seed consistency
        if network_seed is not None:
            degree_config.network_params["seed"] = network_seed
        sims["high-degree"] = Simulation(degree_config)
        sims["high-degree"].opinions *= initial_scale
        sims["high-degree"].set_zealot_opinions()  # Reset zealot opinions to avoid scaling
        zealots["high-degree"] = sims["high-degree"].get_zealot_ids()
    
    # Create results directory
    if output_dir is None:
        results_dir = "results/zealot_experiment"
    else:
        results_dir = output_dir
    os.makedirs(results_dir, exist_ok=True)
    
    # Run simulations of various modes and generate results
    results = {}
    
    if "none" in modes_to_run:
        print("Running simulation without zealots...")
        results["without Zealots"] = run_simulation_and_generate_results(
            sims["none"], [], "without Zealots", results_dir, steps
        )
    
    if "clustered" in modes_to_run:
        print("Running simulation with clustered zealots...")
        results["with Clustered Zealots"] = run_simulation_and_generate_results(
            sims["clustered"], zealots["clustered"], "with Clustered Zealots", results_dir, steps
        )
    
    if "random" in modes_to_run:
        print("Running simulation with random zealots...")
        results["with Random Zealots"] = run_simulation_and_generate_results(
            sims["random"], zealots["random"], "with Random Zealots", results_dir, steps
        )
    
    if "high-degree" in modes_to_run:
        print("Running simulation with high-degree zealots...")
        results["with High-Degree Zealots"] = run_simulation_and_generate_results(
            sims["high-degree"], zealots["high-degree"], "with High-Degree Zealots", results_dir, steps
        )
    
    # Collect statistical data from all modes
    all_stats = {}
    for mode_name, mode_results in results.items():
        all_stats[mode_name] = mode_results["stats"]
    
    # Draw comparative statistical plots
    mode_names = list(results.keys())
    if len(mode_names) > 1:  # Only draw comparison plots when there's more than one mode
        print("Generating comparative statistics plots...")
        plot_comparative_statistics(all_stats, mode_names, results_dir)
    
    # Calculate and print total execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"All simulations completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Write execution time information to results directory
    with open(os.path.join(results_dir, "execution_time.txt"), "w") as f:
        f.write(f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n")
        f.write(f"Configuration: steps={steps}, num_zealots={num_zealots}, morality_rate={morality_rate}, zealot_morality={zealot_morality}\n")
        f.write(f"Modes run: {', '.join(modes_to_run)}")
    
    print("All simulations and visualizations completed.")
    
    # Return all results
    return results


def draw_zealot_network(sim, zealot_ids, title, filename):
    """
    Draw network graph marking zealot nodes
    
    Parameters:
    sim -- simulation instance
    zealot_ids -- list of zealot IDs
    title -- chart title
    filename -- output filename
    """
    plt.figure(figsize=(12, 10))
    plt.title(title, fontsize=16)
    
    node_colors = []
    for i in range(sim.num_agents):
        if i in zealot_ids:
            node_colors.append('red')  # Zealot color is red
        else:
            opinion = sim.opinions[i]
            # Color other nodes based on opinion values
            if opinion > 0.5:
                node_colors.append('blue')
            elif opinion < -0.5:
                node_colors.append('green')
            else:
                node_colors.append('gray')
    
    # Draw network
    nx.draw(
        sim.graph,
        pos=sim.pos,
        node_color=node_colors,
        node_size=50,
        edge_color='lightgray',
        with_labels=False,
        alpha=0.8
    )
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Zealot', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Support (>0.5)', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Oppose (<-0.5)', markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Neutral', markerfacecolor='gray', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


if __name__ == "__main__":
    # Run zealot experiment
    run_zealot_experiment(
        steps=500,            # Run 500 steps
        initial_scale=0.1,     # Scale initial opinions to 10%
        num_zealots=10,        # 10 zealots
        seed=114514,            # Fixed random seed for reproducible results
        morality_rate=0.0,     # Moralization rate
        zealot_morality=False,  # Not all moralizing
        identity_clustered=False, # No clustered initialization by identity
        zealot_mode=None,       # Run all modes
        zealot_identity_allocation=True
    ) 