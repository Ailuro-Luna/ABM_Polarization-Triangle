import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.patches import Patch
import numpy as np


def draw_network(sim, mode, title, filename):
    """
    绘制网络图，使用形状区分zealot，边框区分道德化状态
    
    参数:
    sim -- simulation实例
    mode -- 绘制模式：'opinion', 'identity', 'morality'
    title -- 图表标题
    filename -- 输出文件名
    
    可视化规则:
    - 形状：Zealot=正方形，普通Agent=圆形
    - 边框：道德化=有边框，非道德化=无边框
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 获取zealot信息
    zealot_ids = sim.get_zealot_ids() if hasattr(sim, 'get_zealot_ids') else []
    has_zealots = len(zealot_ids) > 0
    
    # 设置节点大小
    node_size = 60
    
    # 根据mode设置颜色
    if mode == "opinion":
        cmap = cm.coolwarm
        norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        node_colors = [cmap(norm(op)) for op in sim.opinions]
        
        # 添加colorbar
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Opinion", fontsize=12)
        
    elif mode == "identity":
        node_colors = ['#ff7f00' if iden == 1 else '#4daf4a' for iden in sim.identities]
        
    elif mode == "morality":
        node_colors = ['#1a9850' if m == 1 else '#d73027' for m in sim.morals]
        
    elif mode == "identity_morality":
        # 综合显示身份和道德化信息的模式
        # 颜色基于身份，边框基于道德化，形状基于zealot状态
        node_colors = ['#ff7f00' if iden == 1 else '#4daf4a' for iden in sim.identities]
    
    # 分组绘制节点：根据zealot状态和道德化状态分为三组
    # 注意：zealot的金色边框优先级高于道德化的黑色边框
    
    # 1. 普通agent，非道德化 (圆形，无边框)
    normal_non_moral = []
    normal_non_moral_colors = []
    for i in range(sim.num_agents):
        if i not in zealot_ids and sim.morals[i] == 0:
            normal_non_moral.append(i)
            normal_non_moral_colors.append(node_colors[i])
    
    if normal_non_moral:
        nx.draw_networkx_nodes(sim.graph, pos=sim.pos, nodelist=normal_non_moral,
                              node_color=normal_non_moral_colors, node_shape='o',
                              node_size=node_size, edgecolors='none', 
                              linewidths=0, alpha=0.9, ax=ax)
    
    # 2. 普通agent，道德化 (圆形，黑色边框)
    normal_moral = []
    normal_moral_colors = []
    for i in range(sim.num_agents):
        if i not in zealot_ids and sim.morals[i] == 1:
            normal_moral.append(i)
            normal_moral_colors.append(node_colors[i])
    
    if normal_moral:
        nx.draw_networkx_nodes(sim.graph, pos=sim.pos, nodelist=normal_moral,
                              node_color=normal_moral_colors, node_shape='o',
                              node_size=node_size, edgecolors='black', 
                              linewidths=1, alpha=0.9, ax=ax)
    
    # 3. 所有zealot (圆形，金色边框) - 不论是否道德化都用金色边框
    if zealot_ids:
        zealot_colors = [node_colors[i] for i in zealot_ids]
        nx.draw_networkx_nodes(sim.graph, pos=sim.pos, nodelist=zealot_ids,
                              node_color=zealot_colors, node_shape='o',
                              node_size=node_size, edgecolors='gold', 
                              linewidths=2, alpha=0.9, ax=ax)
    
    # 绘制边
    nx.draw_networkx_edges(sim.graph, pos=sim.pos, edge_color="#888888", 
                          alpha=0.7, width=1.2, ax=ax)
    
    # 设置标题和样式
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal', 'box')
    ax.axis('off')
    
    # 创建图例
    legend_patches = []
    
    # 基于mode添加颜色图例
    if mode == "identity":
        legend_patches.extend([
            Patch(color='#ff7f00', label='Identity: 1'),
            Patch(color='#4daf4a', label='Identity: -1')
        ])
    elif mode == "morality":
        legend_patches.extend([
            Patch(color='#1a9850', label='Moralizing'),
            Patch(color='#d73027', label='Non-moralizing')
        ])
    elif mode == "identity_morality":
        legend_patches.extend([
            Patch(color='#ff7f00', label='Identity: 1'),
            Patch(color='#4daf4a', label='Identity: -1')
        ])
    
    # 添加边框说明
    if has_zealots or any(sim.morals == 1):
        # 添加分隔线
        if legend_patches:
            legend_patches.append(Patch(color='white', alpha=0, label=''))  # 空白分隔
        
        # 边框说明
        if any(sim.morals == 1):
            legend_patches.append(
                Patch(facecolor='lightgray', edgecolor='black', linewidth=2, label='Black border: Moralizing')
            )
        
        if has_zealots:
            legend_patches.append(
                Patch(facecolor='lightgray', edgecolor='gold', linewidth=2.5, label='Gold border: Zealot')
            )
    
    # 根据模式调整图例位置，避免与colorbar冲突
    if legend_patches:
        if mode == "opinion":
            # opinion模式下，图例放在左下角，避免与右侧colorbar冲突
            ax.legend(handles=legend_patches, loc='lower left', bbox_to_anchor=(0, 0), 
                     title="Legend", frameon=True, fontsize=9)
        else:
            # 其他模式下，图例放在右侧
            ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.02, 1), 
                     title="Legend", frameon=True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()