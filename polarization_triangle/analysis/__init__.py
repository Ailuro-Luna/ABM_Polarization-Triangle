"""
Analysis模块 - 提供各种分析功能
包括统计分析、激活分析、轨迹分析和敏感性分析
"""

# 现有分析模块
from .statistics import *

# 敏感性分析模块
try:
    from .sobol_analysis import SobolAnalyzer, SobolConfig
    from .sensitivity_metrics import SensitivityMetrics
    from .sensitivity_visualizer import SensitivityVisualizer
    
    __all__ = [
        # 统计分析
        'calculate_mean_opinion',
        'calculate_variance_metrics', 
        'calculate_identity_statistics',
        
        # 轨迹分析
        'calculate_trajectory_metrics',
        
        # 敏感性分析
        'SobolAnalyzer',
        'SobolConfig', 
        'SensitivityMetrics',
        'SensitivityVisualizer'
    ]
    
except ImportError as e:
    # 如果敏感性分析依赖项不可用，仅导出其他模块
    import warnings
    warnings.warn(f"敏感性分析模块不可用: {e}. 请安装必要依赖: pip install SALib pandas seaborn")
    
    __all__ = [
        # 统计分析
        'calculate_mean_opinion',
        'calculate_variance_metrics',
        'calculate_identity_statistics', 
        
        # 轨迹分析
        'calculate_trajectory_metrics'
    ]
