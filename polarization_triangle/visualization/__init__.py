"""
Visualization module for Polarization Triangle Framework
包含各种可视化工具，用于绘制模拟结果
"""

from .network_viz import draw_network
from .opinion_viz import draw_opinion_distribution_heatmap
from .rule_viz import draw_rule_usage, draw_rule_cumulative_usage
from .activation_viz import (
    draw_activation_components,
    draw_activation_history,
    draw_activation_heatmap,
    draw_activation_trajectory,
)

