import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.patches import Patch
import numpy as np


def draw_network(sim, mode, title, filename):
    """
    Draw network graph, using shapes to distinguish zealots and borders to distinguish moralization states
    
    Parameters:
    sim -- simulation instance
    mode -- Drawing mode: 'opinion', 'identity', 'morality'
    title -- Chart title
    filename -- Output filename
    
    Visualization rules:
    - Shape: Zealot=square, Regular Agent=circle
    - Border: Moralized=with border, Non-moralized=no border
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Get zealot information
    zealot_ids = sim.get_zealot_ids() if hasattr(sim, 'get_zealot_ids') else []
    has_zealots = len(zealot_ids) > 0
    
    # Set node size
    node_size = 60
    
    # Set colors based on mode
    if mode == "opinion":
        cmap = cm.coolwarm
        norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        node_colors = [cmap(norm(op)) for op in sim.opinions]
        
        # Add colorbar
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Opinion", fontsize=12)
        
    elif mode == "identity":
        node_colors = ['#ff7f00' if iden == 1 else '#4daf4a' for iden in sim.identities]
        
    elif mode == "morality":
        node_colors = ['#1a9850' if m == 1 else '#d73027' for m in sim.morals]
        
    elif mode == "identity_morality":
        # Mode for combined display of identity and moralization information
        # Color based on identity, border based on moralization, shape based on zealot state
        node_colors = ['#ff7f00' if iden == 1 else '#4daf4a' for iden in sim.identities]
    
    # Group and draw nodes: divide into three groups based on zealot and moralization status
    # Note: The gold border for zealots has priority over the black border for moralization
    
    # 1. Normal agent, non-moralized (circle, no border)
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
    
    # 2. Normal agent, moralized (circle, black border)
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
    
    # 3. All zealots (circle, gold border) - use gold border regardless of moralization
    if zealot_ids:
        zealot_colors = [node_colors[i] for i in zealot_ids]
        nx.draw_networkx_nodes(sim.graph, pos=sim.pos, nodelist=zealot_ids,
                              node_color=zealot_colors, node_shape='o',
                              node_size=node_size, edgecolors='gold', 
                              linewidths=2, alpha=0.9, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(sim.graph, pos=sim.pos, edge_color="#888888", 
                          alpha=0.7, width=1.2, ax=ax)
    
    # Set title and style
    ax.set_title(title, fontsize=18)
    ax.set_aspect('equal', 'box')
    ax.axis('off')
    
    # Create legend
    legend_patches = []
    
    # Add color legend based on mode
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
    
    # Add border descriptions
    if has_zealots or any(sim.morals == 1):
        # Add separator
        if legend_patches:
            legend_patches.append(Patch(color='white', alpha=0, label=''))  # Blank separator
        
        # Border descriptions
        if any(sim.morals == 1):
            legend_patches.append(
                Patch(facecolor='lightgray', edgecolor='black', linewidth=2, label='Black border: Moralizing')
            )
        
        if has_zealots:
            legend_patches.append(
                Patch(facecolor='lightgray', edgecolor='gold', linewidth=2.5, label='Gold border: Zealot')
            )
    
    # Adjust legend position based on mode to avoid conflict with colorbar
    if legend_patches:
        if mode == "opinion":
            # In opinion mode, place the legend in the lower-left corner to avoid conflict with the colorbar on the right
            ax.legend(handles=legend_patches, loc='lower left', bbox_to_anchor=(0, 0), 
                     title="Legend", frameon=True, fontsize=9)
        else:
            # In other modes, place the legend on the right side
            ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.02, 1), 
                     title="Legend", frameon=True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()