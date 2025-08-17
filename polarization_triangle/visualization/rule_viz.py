import matplotlib.pyplot as plt
import numpy as np

def draw_interaction_type_usage(rule_counts_history, title, filename, smooth=False, window_size=5):
    """
    Draw interaction type frequency change over time。
    
    参数:
    rule_counts_history -- List containing interaction type counts for each time step, shape(time_steps, 16)
    title -- 图表标题
    filename -- 保存文件名
    smooth -- Whether to smooth curves
    window_size -- Smoothing window size
    """
    # Convert to numpy array
    rule_counts = np.array(rule_counts_history)
    
    # Check if array is empty
    if rule_counts.size == 0:
        print("Warning: Interaction type history is empty")
        return
    
    # Get number of time steps and interaction types
    time_steps = rule_counts.shape[0]
    num_rules = rule_counts.shape[1] if len(rule_counts.shape) > 1 else 0
    
    # 如果交互类型数量不符合预期，打印警告
    if num_rules != 16 and num_rules != 8:
        print(f"Warning: Number of interaction types ({num_rules}) is not the expected 8 or 16")
    
    # Prepare time axis
    time = np.arange(time_steps)
    
    # Interaction type names - 现在是16种交互类型
    rule_names = [
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
    
    # 如果是旧模式（8种交互类型），使用旧的名称
    if num_rules == 8:
        rule_names = [
            "Rule 1: Same dir, Same ID, Non-moral, Converge",
            "Rule 2: Same dir, Diff ID, Non-moral, Converge",
            "Rule 3: Same dir, Same ID, Moral, Polarize",
            "Rule 4: Same dir, Diff ID, Moral, Polarize",
            "Rule 5: Diff dir, Same ID, Non-moral, Converge",
            "Rule 6: Diff dir, Diff ID, Non-moral, Converge",
            "Rule 7: Diff dir, Same ID, Moral, Converge",
            "Rule 8: Diff dir, Diff ID, Moral, Polarize"
        ]
    
    # 设置颜色 - 为16种交互类型扩展颜色列表
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
        '#98df8a', '#ff9896', '#c5b0d5', '#c49c94'
    ]
    
    plt.figure(figsize=(14, 10))
    
    # 平滑数据（如果需要）
    if smooth and time_steps > window_size:
        for i in range(num_rules):
            # Use moving average for smoothing
            smoothed = np.convolve(rule_counts[:, i], 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
            # Adjust time axis to match smoothed data
            smoothed_time = np.arange(len(smoothed)) + window_size // 2
            plt.plot(smoothed_time, smoothed, label=rule_names[i], color=colors[i], linewidth=2)
    else:
        # No smoothing
        for i in range(num_rules):
            plt.plot(time, rule_counts[:, i], label=rule_names[i], color=colors[i], linewidth=2)
    
    plt.xlabel('Time Step')
    plt.ylabel('Interaction Type Count')
    plt.title(title)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    plt.grid(True, alpha=0.3)
    
    # Save chart
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    # Additionally create a stacked area chart，showing interaction type proportions
    plt.figure(figsize=(14, 10))
    
    # Calculate total interaction types for each time step
    total_counts = np.sum(rule_counts, axis=1)
    # Avoid division by zero
    total_counts = np.where(total_counts == 0, 1, total_counts)
    
    # Calculate proportions of interaction types
    proportions = rule_counts / total_counts[:, np.newaxis]
    
    # Create stacked area chart
    plt.stackplot(time, 
                 [proportions[:, i] for i in range(num_rules)],
                 labels=rule_names,
                 colors=colors,
                 alpha=0.7)
    
    plt.xlabel('Time Step')
    plt.ylabel('Interaction Type Proportion')
    plt.title(f"{title} - Proportional Usage")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    plt.grid(True, alpha=0.3)
    
    # Save proportion chart
    proportion_filename = filename.replace('.png', '_proportions.png')
    plt.tight_layout()
    plt.savefig(proportion_filename)
    plt.close()

def draw_interaction_type_cumulative_usage(rule_counts_history, title, filename, smooth=False, window_size=5):
    """
    Draw interaction type cumulative count change over time。
    
    参数:
    rule_counts_history -- List containing interaction type counts for each time step, shape(time_steps, 16)
    title -- 图表标题
    filename -- 保存文件名
    smooth -- Whether to smooth curves
    window_size -- Smoothing window size
    """
    # Convert to numpy array
    rule_counts = np.array(rule_counts_history)
    
    # Check if array is empty
    if rule_counts.size == 0:
        print("Warning: Interaction type history is empty")
        return
    
    # Get number of time steps and interaction types
    time_steps = rule_counts.shape[0]
    num_rules = rule_counts.shape[1] if len(rule_counts.shape) > 1 else 0
    
    # 如果交互类型数量不符合预期，打印警告
    if num_rules != 16 and num_rules != 8:
        print(f"Warning: Number of interaction types ({num_rules}) is not the expected 8 or 16")
    
    # Calculate cumulative counts
    cumulative_counts = np.cumsum(rule_counts, axis=0)
    
    # Prepare time axis
    time = np.arange(time_steps)
    
    # Interaction type names - 现在是16种交互类型
    rule_names = [
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
    
    # 如果是旧模式（8种交互类型），使用旧的名称
    if num_rules == 8:
        rule_names = [
            "Rule 1: Same dir, Same ID, Non-moral, Converge",
            "Rule 2: Same dir, Diff ID, Non-moral, Converge",
            "Rule 3: Same dir, Same ID, Moral, Polarize",
            "Rule 4: Same dir, Diff ID, Moral, Polarize",
            "Rule 5: Diff dir, Same ID, Non-moral, Converge",
            "Rule 6: Diff dir, Diff ID, Non-moral, Converge",
            "Rule 7: Diff dir, Same ID, Moral, Converge",
            "Rule 8: Diff dir, Diff ID, Moral, Polarize"
        ]
    
    # 设置颜色 - 为16种交互类型扩展颜色列表
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
        '#98df8a', '#ff9896', '#c5b0d5', '#c49c94'
    ]
    
    plt.figure(figsize=(14, 10))
    
    # 平滑数据（如果需要）
    if smooth and time_steps > window_size:
        for i in range(num_rules):
            # Use moving average for smoothing
            smoothed = np.convolve(cumulative_counts[:, i], 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
            # Adjust time axis to match smoothed data
            smoothed_time = np.arange(len(smoothed)) + window_size // 2
            plt.plot(smoothed_time, smoothed, label=rule_names[i], color=colors[i], linewidth=2)
    else:
        # No smoothing
        for i in range(num_rules):
            plt.plot(time, cumulative_counts[:, i], label=rule_names[i], color=colors[i], linewidth=2)
    
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Interaction Type Count')
    plt.title(title)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    plt.grid(True, alpha=0.3)
    
    # Save chart
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    # Additionally create a stacked area chart，showing interaction type proportions
    plt.figure(figsize=(14, 10))
    
    # Create stacked area chart
    plt.stackplot(time, 
                 [cumulative_counts[:, i] for i in range(num_rules)],
                 labels=rule_names,
                 colors=colors,
                 alpha=0.7)
    
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Interaction Type Count')
    plt.title(f"{title} - Stacked View")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    plt.grid(True, alpha=0.3)
    
    # Save stacked chart
    stacked_filename = filename.replace('.png', '_stacked.png')
    plt.tight_layout()
    plt.savefig(stacked_filename)
    plt.close()

# Keep original function names for backward compatibility
draw_rule_usage = draw_interaction_type_usage
draw_rule_cumulative_usage = draw_interaction_type_cumulative_usage