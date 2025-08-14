import matplotlib.pyplot as plt
import numpy as np

def draw_interaction_type_usage(rule_counts_history, title, filename, smooth=False, window_size=5):
    """
    绘制交互类型频率随时间的变化图。
    
    参数:
    rule_counts_history -- 包含每个时间步骤交互类型次数的列表，形状为(time_steps, 16)
    title -- 图表标题
    filename -- 保存文件名
    smooth -- 是否平滑曲线
    window_size -- 平滑窗口大小
    """
    # 转换为numpy数组
    rule_counts = np.array(rule_counts_history)
    
    # 检查数组是否为空
    if rule_counts.size == 0:
        print("警告: 交互类型历史记录为空")
        return
    
    # 获取时间步数和交互类型数量
    time_steps = rule_counts.shape[0]
    num_rules = rule_counts.shape[1] if len(rule_counts.shape) > 1 else 0
    
    # 如果交互类型数量不符合预期，打印警告
    if num_rules != 16 and num_rules != 8:
        print(f"警告: 交互类型数量({num_rules})不是预期的8或16")
    
    # 准备时间轴
    time = np.arange(time_steps)
    
    # 交互类型名称 - 现在是16种交互类型
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
            # 使用移动平均进行平滑
            smoothed = np.convolve(rule_counts[:, i], 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
            # 调整时间轴以匹配平滑后的数据
            smoothed_time = np.arange(len(smoothed)) + window_size // 2
            plt.plot(smoothed_time, smoothed, label=rule_names[i], color=colors[i], linewidth=2)
    else:
        # 不平滑处理
        for i in range(num_rules):
            plt.plot(time, rule_counts[:, i], label=rule_names[i], color=colors[i], linewidth=2)
    
    plt.xlabel('Time Step')
    plt.ylabel('Interaction Type Count')
    plt.title(title)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    # 额外创建一个堆叠面积图，显示交互类型的比例
    plt.figure(figsize=(14, 10))
    
    # 计算每个时间步的交互类型总数
    total_counts = np.sum(rule_counts, axis=1)
    # 避免除以零
    total_counts = np.where(total_counts == 0, 1, total_counts)
    
    # 计算交互类型的比例
    proportions = rule_counts / total_counts[:, np.newaxis]
    
    # 创建堆叠面积图
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
    
    # 保存比例图
    proportion_filename = filename.replace('.png', '_proportions.png')
    plt.tight_layout()
    plt.savefig(proportion_filename)
    plt.close()

def draw_interaction_type_cumulative_usage(rule_counts_history, title, filename, smooth=False, window_size=5):
    """
    绘制交互类型累积次数随时间的变化图。
    
    参数:
    rule_counts_history -- 包含每个时间步骤交互类型次数的列表，形状为(time_steps, 16)
    title -- 图表标题
    filename -- 保存文件名
    smooth -- 是否平滑曲线
    window_size -- 平滑窗口大小
    """
    # 转换为numpy数组
    rule_counts = np.array(rule_counts_history)
    
    # 检查数组是否为空
    if rule_counts.size == 0:
        print("警告: 交互类型历史记录为空")
        return
    
    # 获取时间步数和交互类型数量
    time_steps = rule_counts.shape[0]
    num_rules = rule_counts.shape[1] if len(rule_counts.shape) > 1 else 0
    
    # 如果交互类型数量不符合预期，打印警告
    if num_rules != 16 and num_rules != 8:
        print(f"警告: 交互类型数量({num_rules})不是预期的8或16")
    
    # 计算累积次数
    cumulative_counts = np.cumsum(rule_counts, axis=0)
    
    # 准备时间轴
    time = np.arange(time_steps)
    
    # 交互类型名称 - 现在是16种交互类型
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
            # 使用移动平均进行平滑
            smoothed = np.convolve(cumulative_counts[:, i], 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
            # 调整时间轴以匹配平滑后的数据
            smoothed_time = np.arange(len(smoothed)) + window_size // 2
            plt.plot(smoothed_time, smoothed, label=rule_names[i], color=colors[i], linewidth=2)
    else:
        # 不平滑处理
        for i in range(num_rules):
            plt.plot(time, cumulative_counts[:, i], label=rule_names[i], color=colors[i], linewidth=2)
    
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Interaction Type Count')
    plt.title(title)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    # 额外创建一个堆叠面积图，显示交互类型的比例
    plt.figure(figsize=(14, 10))
    
    # 创建堆叠面积图
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
    
    # 保存堆叠图
    stacked_filename = filename.replace('.png', '_stacked.png')
    plt.tight_layout()
    plt.savefig(stacked_filename)
    plt.close()

# 为保持向后兼容性，保留原有函数名称
draw_rule_usage = draw_interaction_type_usage
draw_rule_cumulative_usage = draw_interaction_type_cumulative_usage