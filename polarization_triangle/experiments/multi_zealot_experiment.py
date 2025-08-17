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
    生成平均意见分布热图
    
    参数:
    all_opinion_histories -- 包含所有运行的意见历史的字典
    mode_names -- 模式名称列表
    output_dir -- 输出目录
    heatmap_config -- 热力图配置字典，包含颜色映射、尺度等参数
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
        
        # Calculate average opinion distribution（而不是平均意见轨迹）
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
    计算多次运行的平均意见分布直方图
    
    参数:
    opinion_histories -- 包含多次运行意见历史的列表
    bins -- opinion值的分箱数量
    
    返回:
    numpy.ndarray -- 平均分布直方图数据，形状为(time_steps, bins)
    """
    if not opinion_histories:
        return np.array([])
    
    # Get time steps from first history
    num_steps = len(opinion_histories[0])
    
    # 创建opinion的bins
    opinion_bins = np.linspace(-1, 1, bins + 1)
    
    # 初始化所有运行的分布数据存储
    all_distributions = np.zeros((len(opinion_histories), num_steps, bins))
    
    # 对每次运行计算分布直方图
    for run_idx, history in enumerate(opinion_histories):
        for step in range(min(num_steps, len(history))):
            # 计算该时间步的opinion分布
            hist, _ = np.histogram(history[step], bins=opinion_bins, range=(-1, 1))
            all_distributions[run_idx, step] = hist
    
    # 计算平均分布
    avg_distribution = np.mean(all_distributions, axis=0)
    
    return avg_distribution


def draw_opinion_distribution_heatmap_from_distribution(distribution_data, title, filename, bins=40, log_scale=True,
                                                       cmap='viridis', vmin=None, vmax=None, custom_norm=None, start_step=0):
    """
    从预计算的分布数据绘制热力图
    
    参数:
    distribution_data -- 分布数据，形状为(time_steps, bins)
    title -- 图表标题
    filename -- 保存文件名
    bins -- opinion值的分箱数量
    log_scale -- 是否使用对数比例表示颜色
    cmap -- 颜色映射方案 ('viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'RdBu', 'hot', 'jet', etc.)
    vmin -- 颜色尺度的最小值，如果为None则自动确定
    vmax -- 颜色尺度的最大值，如果为None则自动确定
    custom_norm -- 自定义的颜色标准化对象，如果提供则会覆盖log_scale、vmin、vmax
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    
    # 获取时间步数
    time_steps = distribution_data.shape[0]
    
    # 创建opinion的bins
    opinion_bins = np.linspace(-1, 1, bins + 1)
    
    # Create plot
    # 放大默认字体并禁用 Unicode 负号，避免 10^-1 中负号渲染问题
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 26,
        'axes.labelsize': 24,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.unicode_minus': False
    })
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # 创建坐标
    x = opinion_bins[:-1] + np.diff(opinion_bins) / 2  # opinion值（bin中点）
    y = np.arange(time_steps)  # 时间步骤索引（用于绘图，从0开始）
    
    # 确定颜色标准化
    if custom_norm is not None:
        # 使用自定义标准化
        norm = custom_norm
        plot_data = distribution_data
    elif log_scale:
        # 使用对数比例，先将0值替换为最小非零值以避免log(0)错误
        min_nonzero = np.min(distribution_data[distribution_data > 0]) if np.any(distribution_data > 0) else 1
        log_data = np.copy(distribution_data)
        log_data[log_data == 0] = min_nonzero
        
        # 设置对数标准化的范围
        log_vmin = vmin if vmin is not None else min_nonzero
        log_vmax = vmax if vmax is not None else np.max(log_data)
        norm = LogNorm(vmin=log_vmin, vmax=log_vmax)
        plot_data = log_data
    else:
        # 使用线性比例
        linear_vmin = vmin if vmin is not None else np.min(distribution_data)
        linear_vmax = vmax if vmax is not None else np.max(distribution_data)
        norm = plt.Normalize(vmin=linear_vmin, vmax=linear_vmax)
        plot_data = distribution_data
    
    # 绘制热力图
    pcm = ax.pcolormesh(x, y, plot_data, norm=norm, cmap=cmap, shading='auto')
    
    # 添加颜色条
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
    
    # 如果设置了具体的数值范围，可以自定义颜色条刻度
    if vmin is not None and vmax is not None:
        if log_scale and not custom_norm:
            # 对数尺度的刻度
            ticks = []
            current = vmin
            while current <= vmax:
                ticks.append(current)
                current *= 10
            if ticks[-1] < vmax:
                ticks.append(vmax)
            cbar.set_ticks(ticks)
        else:
            # 线性尺度的刻度
            step = (vmax - vmin) / 5
            cbar.set_ticks([vmin + i*step for i in range(6)])
    
    # Set labels and title
    ax.set_xlabel('Opinion Value', fontsize=24)
    ax.set_ylabel('Time Step', fontsize=24)
    ax.set_title(title, fontsize=26)
    ax.tick_params(axis='both', labelsize=20)
    
    # 优化Y轴刻度，防止过密，但显示真实的时间步骤
    max_ticks = 10
    tick_step = max(1, time_steps // max_ticks)
    tick_positions = np.arange(0, time_steps, tick_step)  # 在图上的位置（从0开始）
    tick_labels = [str(start_step + pos) for pos in tick_positions]  # 显示的真实时间步骤
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    
    # Save chart
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
    # 额外创建一个3D视图
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 选择一些时间步骤来展示（避免过度拥挤）
    step_interval = max(1, time_steps // 20)
    selected_timesteps = np.arange(0, time_steps, step_interval)
    
    # 为3D图准备数据
    X, Y = np.meshgrid(x, selected_timesteps)
    selected_data = plot_data[selected_timesteps]
    
    # 绘制3D表面
    surf = ax.plot_surface(X, Y, selected_data, cmap=cmap, edgecolor='none', alpha=0.8)
    
    # Set labels and title
    ax.set_xlabel('Opinion Value', fontsize=18)
    ax.set_ylabel('Time Step', fontsize=18)
    ax.set_zlabel('Average Agent Count', fontsize=18)
    ax.set_title(f"{title} - 3D View", fontsize=20)
    ax.tick_params(axis='both', labelsize=14)
    
    # 修复3D图的Y轴刻度显示真实时间步骤
    y_tick_positions = selected_timesteps[::max(1, len(selected_timesteps)//5)]
    y_tick_labels = [str(start_step + pos) for pos in y_tick_positions]
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels)
    
    # 添加颜色条
    cbar3d = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar3d.set_label('Average Agent Count', fontsize=16)
    cbar3d.ax.tick_params(labelsize=12)
    
    # 保存3D图
    waterfall_filename = filename.replace('.png', '_3d.png')
    plt.tight_layout()
    plt.savefig(waterfall_filename, dpi=300)
    plt.close()


def plot_average_statistics(avg_stats, mode_names, output_dir, steps):
    """
    Plot average statistics charts
    
    参数:
    avg_stats -- 平均统计数据字典
    mode_names -- 模式名称列表
    output_dir -- 输出目录
    steps -- 模拟步数
    """
    # 确保统计目录存在
    stats_dir = os.path.join(output_dir, "statistics")
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    
    step_values = range(steps)
    
    # 使用不同颜色和线型
    colors = ['blue', 'red', 'green', 'purple']
    linestyles = ['-', '--', '-.', ':']
    
    # 1. 绘制平均意见值对比图
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
    
    # 2. 绘制平均绝对意见值对比图
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
    
    # 3. 绘制非zealot方差对比图
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
    
    # 4. 绘制社区内部方差对比图
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
    
    # 5. 绘制负面意见数量对比图
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
    
    # 6. 绘制负面意见均值对比图
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
    
    # 7. 绘制正面意见数量对比图
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
    
    # 8. 绘制正面意见均值对比图
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
    
    # 9. 绘制极化指数对比图（如果有数据）
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
    
    # 10. 新增：绘制identity平均意见对比图
    has_identity_data = all(
        "identity_1_mean_opinions" in avg_stats[mode] and "identity_neg1_mean_opinions" in avg_stats[mode]
        for mode in mode_names
    )
    
    if has_identity_data:
        # 10a. 两种identity的平均opinion对比图
        plt.figure(figsize=(15, 7))
        for i, mode in enumerate(mode_names):
            # Identity = 1的平均opinion（实线）
            plt.plot(step_values, avg_stats[mode]["identity_1_mean_opinions"], 
                    label=f'{mode} - Identity +1', 
                    color=colors[i], linestyle='-')
            # Identity = -1的平均opinion（虚线）
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
        
        # 10b. identity意见差值绝对值对比图
        plt.figure(figsize=(12, 7))
        for i, mode in enumerate(mode_names):
            # 计算绝对值
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
    
    # 11. 保存平均统计数据到CSV文件
    stats_csv = os.path.join(stats_dir, "avg_opinion_stats.csv")
    with open(stats_csv, "w") as f:
        # 写入标题行
        f.write("step")
        for mode in mode_names:
            f.write(f",{mode}_mean_opinion,{mode}_mean_abs_opinion,{mode}_non_zealot_variance,{mode}_cluster_variance")
            f.write(f",{mode}_negative_count,{mode}_negative_mean,{mode}_positive_count,{mode}_positive_mean")
            if "polarization_index" in avg_stats[mode] and len(avg_stats[mode]["polarization_index"]) > 0:
                f.write(f",{mode}_polarization_index")
            # 添加identity相关的列
            if "identity_1_mean_opinions" in avg_stats[mode]:
                f.write(f",{mode}_identity_1_mean_opinion,{mode}_identity_neg1_mean_opinion,{mode}_identity_opinion_difference")
        f.write("\n")
        
        # 写入数据
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
                # 如果有极化指数数据，添加到CSV
                if "polarization_index" in avg_stats[mode] and step < len(avg_stats[mode]["polarization_index"]):
                    f.write(f",{avg_stats[mode]['polarization_index'][step]:.4f}")
                # 添加identity相关数据
                if "identity_1_mean_opinions" in avg_stats[mode] and step < len(avg_stats[mode]["identity_1_mean_opinions"]):
                    f.write(f",{avg_stats[mode]['identity_1_mean_opinions'][step]:.4f}")
                    f.write(f",{avg_stats[mode]['identity_neg1_mean_opinions'][step]:.4f}")
                    f.write(f",{avg_stats[mode]['identity_opinion_differences'][step]:.4f}")
            f.write("\n")


if __name__ == "__main__":
    # Run multiple zealot experiments
    run_multi_zealot_experiment(
        runs=10,                  # 运行10次实验
        steps=100,                # 每次运行100步
        initial_scale=0.1,        # 初始意见缩放到10%
        morality_rate=0.2,        # moralizing的比例为20%
        zealot_morality=True,     # zealot全部moralizing
        identity_clustered=True,  # 按identity进行clustered的初始化
        zealot_count=10,          # 10个zealot
        zealot_mode="clustered",  # 使用clustered zealot模式
        base_seed=42              # 基础随机种子
    ) 