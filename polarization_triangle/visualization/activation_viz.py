import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

def draw_activation_components(sim, title, filename):
    """
    Draw scatter plots and histograms of self-activation and social influence components
    
    Parameters:
    sim -- Simulation object containing self-activation and social influence data
    title -- Chart title
    filename -- Save filename
    """
    # Get activation components from the most recent step
    components = sim.get_activation_components()
    self_activation = components["self_activation"]
    social_influence = components["social_influence"]
    
    # Set figure size
    plt.figure(figsize=(16, 12))
    
    # 1. Scatter plot：Self activation vs Social influence
    plt.subplot(2, 2, 1)
    plt.scatter(self_activation, social_influence, c=sim.opinions, cmap='coolwarm', alpha=0.7)
    plt.colorbar(label='Opinion')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Self Activation')
    plt.ylabel('Social Influence')
    plt.title('Self Activation vs Social Influence')
    plt.grid(True, alpha=0.3)
    
    # 2. Scatter plot：Self activation vs Social influence（colored by identity）
    plt.subplot(2, 2, 2)
    colors = ['#ff7f00' if iden == 1 else '#4daf4a' for iden in sim.identities]
    plt.scatter(self_activation, social_influence, c=colors, alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Self Activation')
    plt.ylabel('Social Influence')
    plt.title('Self Activation vs Social Influence (by Identity)')
    plt.grid(True, alpha=0.3)
    # 添加图例
    patches = [
        Patch(color='#ff7f00', label='Identity: 1'),
        Patch(color='#4daf4a', label='Identity: -1')
    ]
    plt.legend(handles=patches)
    
    # 3. Histogram：Self activation值分布
    plt.subplot(2, 2, 3)
    plt.hist(self_activation, bins=30, alpha=0.7, color='green')
    plt.xlabel('Self Activation Value')
    plt.ylabel('Count')
    plt.title('Distribution of Self Activation')
    plt.grid(True, alpha=0.3)
    
    # 4. Histogram：Social influence值分布
    plt.subplot(2, 2, 4)
    plt.hist(social_influence, bins=30, alpha=0.7, color='purple')
    plt.xlabel('Social Influence Value')
    plt.ylabel('Count')
    plt.title('Distribution of Social Influence')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle腾出空间
    plt.savefig(filename)
    plt.close()

def draw_activation_history(sim, title, filename):
    """
    绘制Self activation和Social influence随时间的变化
    
    Parameters:
    sim -- 模拟对象，包含Self activation和Social influence的历史数据
    title -- Chart title
    filename -- Save filename
    """
    # 获取activation history数据
    history = sim.get_activation_history()
    self_activation_history = history["self_activation_history"]
    social_influence_history = history["social_influence_history"]
    
    # 如果历史数据为空，则返回
    if not self_activation_history or len(self_activation_history) == 0:
        print("警告: activation history数据为空")
        return
    
    # 将列表转换为NumPy数组以便计算
    self_activation_array = np.array(self_activation_history)
    social_influence_array = np.array(social_influence_history)
    
    # 计算每个时间步骤的平均值
    self_activation_mean = np.mean(self_activation_array, axis=1)
    social_influence_mean = np.mean(social_influence_array, axis=1)
    
    # 计算每个时间步骤的标准差
    self_activation_std = np.std(self_activation_array, axis=1)
    social_influence_std = np.std(social_influence_array, axis=1)
    
    # 时间步骤
    time_steps = np.arange(len(self_activation_history))
    
    # 创建图形
    plt.figure(figsize=(14, 10))
    
    # 绘制Self activation平均值
    plt.subplot(2, 1, 1)
    plt.plot(time_steps, self_activation_mean, 'g-', label='Mean Self Activation')
    # 添加标准差带
    plt.fill_between(time_steps, 
                     self_activation_mean - self_activation_std, 
                     self_activation_mean + self_activation_std, 
                     color='g', alpha=0.2)
    plt.xlabel('Time Step')
    plt.ylabel('Self Activation')
    plt.title('Mean Self Activation Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制Social influence平均值
    plt.subplot(2, 1, 2)
    plt.plot(time_steps, social_influence_mean, 'b-', label='Mean Social Influence')
    # 添加标准差带
    plt.fill_between(time_steps, 
                     social_influence_mean - social_influence_std, 
                     social_influence_mean + social_influence_std, 
                     color='b', alpha=0.2)
    plt.xlabel('Time Step')
    plt.ylabel('Social Influence')
    plt.title('Mean Social Influence Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle腾出空间
    plt.savefig(filename)
    plt.close()
    
    # 额外绘制一个组合图，同时显示Self activation和Social influence的变化
    plt.figure(figsize=(14, 8))
    plt.plot(time_steps, self_activation_mean, 'g-', label='Mean Self Activation')
    plt.plot(time_steps, social_influence_mean, 'b-', label='Mean Social Influence')
    plt.xlabel('Time Step')
    plt.ylabel('Activation Value')
    plt.title(f'{title} - Combined View')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存组合图
    combined_filename = filename.replace('.png', '_combined.png')
    plt.savefig(combined_filename)
    plt.close()

def draw_activation_heatmap(sim, title, filename):
    """
    绘制Self activation和Social influence的heatmap，显示这些值随意见和身份的分布
    
    Parameters:
    sim -- Simulation object containing self-activation and social influence data
    title -- Chart title
    filename -- Save filename
    """
    # Get activation components from the most recent step
    components = sim.get_activation_components()
    self_activation = components["self_activation"]
    social_influence = components["social_influence"]
    
    # 创建图形
    plt.figure(figsize=(16, 12))
    
    # 1. Self activation vs 意见的heatmap
    plt.subplot(2, 2, 1)
    hb = plt.hexbin(sim.opinions, self_activation, gridsize=20, cmap='inferno', mincnt=1)
    plt.colorbar(hb, label='Count')
    plt.xlabel('Opinion')
    plt.ylabel('Self Activation')
    plt.title('Self Activation vs Opinion')
    
    # 2. Social influence vs 意见的heatmap
    plt.subplot(2, 2, 2)
    hb = plt.hexbin(sim.opinions, social_influence, gridsize=20, cmap='inferno', mincnt=1)
    plt.colorbar(hb, label='Count')
    plt.xlabel('Opinion')
    plt.ylabel('Social Influence')
    plt.title('Social Influence vs Opinion')
    
    # 3. Self activation + Social influence vs 意见的heatmap
    plt.subplot(2, 2, 3)
    total_activation = self_activation + social_influence
    hb = plt.hexbin(sim.opinions, total_activation, gridsize=20, cmap='inferno', mincnt=1)
    plt.colorbar(hb, label='Count')
    plt.xlabel('Opinion')
    plt.ylabel('Total Activation')
    plt.title('Total Activation vs Opinion')
    
    # 4. Self activation vs Social influence的heatmap
    plt.subplot(2, 2, 4)
    hb = plt.hexbin(self_activation, social_influence, gridsize=20, cmap='inferno', mincnt=1)
    plt.colorbar(hb, label='Count')
    plt.xlabel('Self Activation')
    plt.ylabel('Social Influence')
    plt.title('Self Activation vs Social Influence')
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle腾出空间
    plt.savefig(filename)
    plt.close()
    
    # 按身份分开绘制heatmap
    plt.figure(figsize=(16, 12))
    
    # 获取不同身份的agent索引
    identity_1_idx = np.where(sim.identities == 1)[0]
    identity_neg1_idx = np.where(sim.identities == -1)[0]
    
    # 1. Self activation vs 意见（身份 = 1）
    plt.subplot(2, 2, 1)
    if len(identity_1_idx) > 0:
        hb = plt.hexbin(sim.opinions[identity_1_idx], self_activation[identity_1_idx], 
                        gridsize=20, cmap='Reds', mincnt=1)
        plt.colorbar(hb, label='Count')
    plt.xlabel('Opinion')
    plt.ylabel('Self Activation')
    plt.title('Self Activation vs Opinion (Identity = 1)')
    
    # 2. Social influence vs 意见（身份 = 1）
    plt.subplot(2, 2, 2)
    if len(identity_1_idx) > 0:
        hb = plt.hexbin(sim.opinions[identity_1_idx], social_influence[identity_1_idx], 
                        gridsize=20, cmap='Reds', mincnt=1)
        plt.colorbar(hb, label='Count')
    plt.xlabel('Opinion')
    plt.ylabel('Social Influence')
    plt.title('Social Influence vs Opinion (Identity = 1)')
    
    # 3. Self activation vs 意见（身份 = -1）
    plt.subplot(2, 2, 3)
    if len(identity_neg1_idx) > 0:
        hb = plt.hexbin(sim.opinions[identity_neg1_idx], self_activation[identity_neg1_idx], 
                        gridsize=20, cmap='Blues', mincnt=1)
        plt.colorbar(hb, label='Count')
    plt.xlabel('Opinion')
    plt.ylabel('Self Activation')
    plt.title('Self Activation vs Opinion (Identity = -1)')
    
    # 4. Social influence vs 意见（身份 = -1）
    plt.subplot(2, 2, 4)
    if len(identity_neg1_idx) > 0:
        hb = plt.hexbin(sim.opinions[identity_neg1_idx], social_influence[identity_neg1_idx], 
                        gridsize=20, cmap='Blues', mincnt=1)
        plt.colorbar(hb, label='Count')
    plt.xlabel('Opinion')
    plt.ylabel('Social Influence')
    plt.title('Social Influence vs Opinion (Identity = -1)')
    
    plt.suptitle(f'{title} - By Identity')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle腾出空间
    
    # 保存按身份分类的heatmap
    identity_filename = filename.replace('.png', '_by_identity.png')
    plt.savefig(identity_filename)
    plt.close()
    
    # 按道德化值分开绘制heatmap
    plt.figure(figsize=(16, 12))
    
    # 获取不同道德化值的agent索引
    moral_1_idx = np.where(sim.morals == 1)[0]
    moral_0_idx = np.where(sim.morals == 0)[0]
    
    # 1. Self activation vs 意见（道德化 = 1）
    plt.subplot(2, 2, 1)
    if len(moral_1_idx) > 0:
        hb = plt.hexbin(sim.opinions[moral_1_idx], self_activation[moral_1_idx], 
                        gridsize=20, cmap='Greens', mincnt=1)
        plt.colorbar(hb, label='Count')
    plt.xlabel('Opinion')
    plt.ylabel('Self Activation')
    plt.title('Self Activation vs Opinion (Morality = 1)')
    
    # 2. Social influence vs 意见（道德化 = 1）
    plt.subplot(2, 2, 2)
    if len(moral_1_idx) > 0:
        hb = plt.hexbin(sim.opinions[moral_1_idx], social_influence[moral_1_idx], 
                        gridsize=20, cmap='Greens', mincnt=1)
        plt.colorbar(hb, label='Count')
    plt.xlabel('Opinion')
    plt.ylabel('Social Influence')
    plt.title('Social Influence vs Opinion (Morality = 1)')
    
    # 3. Self activation vs 意见（道德化 = 0）
    plt.subplot(2, 2, 3)
    if len(moral_0_idx) > 0:
        hb = plt.hexbin(sim.opinions[moral_0_idx], self_activation[moral_0_idx], 
                        gridsize=20, cmap='Oranges', mincnt=1)
        plt.colorbar(hb, label='Count')
    plt.xlabel('Opinion')
    plt.ylabel('Self Activation')
    plt.title('Self Activation vs Opinion (Morality = 0)')
    
    # 4. Social influence vs 意见（道德化 = 0）
    plt.subplot(2, 2, 4)
    if len(moral_0_idx) > 0:
        hb = plt.hexbin(sim.opinions[moral_0_idx], social_influence[moral_0_idx], 
                        gridsize=20, cmap='Oranges', mincnt=1)
        plt.colorbar(hb, label='Count')
    plt.xlabel('Opinion')
    plt.ylabel('Social Influence')
    plt.title('Social Influence vs Opinion (Morality = 0)')
    
    plt.suptitle(f'{title} - By Morality')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle腾出空间
    
    # 保存按道德化值分类的heatmap
    morality_filename = filename.replace('.png', '_by_morality.png')
    plt.savefig(morality_filename)
    plt.close()

def draw_activation_trajectory(sim, history, title, filename):
    """
    绘制Self activation和Social influence的trajectory图，显示个别agent的activation components随时间的变化
    
    Parameters:
    sim -- 模拟对象
    history -- 意见历史数据，用于选择代表性的agent
    title -- Chart title
    filename -- Save filename
    """
    # 获取activation history数据
    activation_history = sim.get_activation_history()
    self_activation_history = activation_history["self_activation_history"]
    social_influence_history = activation_history["social_influence_history"]
    
    # 如果历史数据为空，则返回
    if not self_activation_history or len(self_activation_history) == 0:
        print("警告: activation history数据为空")
        return
    
    # 将list转换为numpy数组
    self_activation_array = np.array(self_activation_history)
    social_influence_array = np.array(social_influence_history)
    
    # 时间步骤
    time_steps = np.arange(len(self_activation_history))
    
    # 选择一些代表性的agent进行可视化
    # 基于最终意见的极值和中值
    final_opinions = np.array(history[-1]) if history and len(history) > 0 else sim.opinions
    
    # 找到意见最极端和最中庸的agent
    most_positive_idx = np.argmax(final_opinions)
    most_negative_idx = np.argmin(final_opinions)
    moderate_idx = np.argmin(np.abs(final_opinions))
    
    # 随机选择一些其他agent
    num_random = 2
    random_indices = np.random.choice(
        [i for i in range(sim.num_agents) if i not in [most_positive_idx, most_negative_idx, moderate_idx]],
        num_random, replace=False)
    
    # 选择的所有agent
    selected_indices = [most_positive_idx, most_negative_idx, moderate_idx] + list(random_indices)
    selected_names = ["Most Positive", "Most Negative", "Most Neutral"] + [f"Random {i+1}" for i in range(num_random)]
    
    # 颜色列表
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    # 创建图形
    plt.figure(figsize=(14, 10))
    
    # 1. Self activationtrajectory
    plt.subplot(2, 1, 1)
    for i, idx in enumerate(selected_indices):
        plt.plot(time_steps, self_activation_array[:, idx], 
                 label=f"{selected_names[i]} (ID={sim.identities[idx]}, M={sim.morals[idx]})",
                 color=colors[i % len(colors)])
    
    plt.xlabel('Time Step')
    plt.ylabel('Self Activation')
    plt.title('Self Activation Trajectories for Selected Agents')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Social influencetrajectory
    plt.subplot(2, 1, 2)
    for i, idx in enumerate(selected_indices):
        plt.plot(time_steps, social_influence_array[:, idx], 
                 label=f"{selected_names[i]} (ID={sim.identities[idx]}, M={sim.morals[idx]})",
                 color=colors[i % len(colors)])
    
    plt.xlabel('Time Step')
    plt.ylabel('Social Influence')
    plt.title('Social Influence Trajectories for Selected Agents')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle腾出空间
    plt.savefig(filename)
    plt.close()
    
    # 额外绘制一个对比图，显示意见、Self activation和Social influence的关系
    plt.figure(figsize=(16, 12))
    
    # 确保history列表和activation history长度一致
    if history and len(history) > 0:
        # 可能需要裁剪历史数据，确保长度匹配
        min_length = min(len(time_steps), len(history))
        history_array = np.array(history[:min_length])
        time_steps_adjusted = np.arange(min_length)
    else:
        history_array = np.zeros((len(time_steps), sim.num_agents))
        time_steps_adjusted = time_steps
    
    for i, idx in enumerate(selected_indices):
        plt.subplot(len(selected_indices), 1, i+1)
        
        if history and len(history) > 0:
            opinions = history_array[:, idx]
        else:
            opinions = np.zeros_like(time_steps_adjusted)
        
        # 裁剪激活数据以匹配长度
        adj_self_activation = self_activation_array[:min_length, idx] if history and len(history) > 0 else self_activation_array[:, idx]
        adj_social_influence = social_influence_array[:min_length, idx] if history and len(history) > 0 else social_influence_array[:, idx]
        
        # 创建三个y轴
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        
        # 偏移第三个y轴
        ax3.spines['right'].set_position(('outward', 60))
        
        # 绘制数据
        line1, = ax1.plot(time_steps_adjusted, opinions, 'k-', label='Opinion')
        line2, = ax2.plot(time_steps_adjusted, adj_self_activation, 'g-', label='Self Activation')
        line3, = ax3.plot(time_steps_adjusted, adj_social_influence, 'b-', label='Social Influence')
        
        # 设置标签
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Opinion', color='k')
        ax2.set_ylabel('Self Activation', color='g')
        ax3.set_ylabel('Social Influence', color='b')
        
        # 设置颜色
        ax1.tick_params(axis='y', labelcolor='k')
        ax2.tick_params(axis='y', labelcolor='g')
        ax3.tick_params(axis='y', labelcolor='b')
        
        # 添加图例
        lines = [line1, line2, line3]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        plt.title(f"{selected_names[i]} (ID={sim.identities[idx]}, M={sim.morals[idx]})")
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'{title} - Combined Trajectories')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle腾出空间
    
    # 保存组合trajectory图
    combined_filename = filename.replace('.png', '_combined.png')
    plt.savefig(combined_filename)
    plt.close()