"""
Analysis module - provides various analysis functions
Includes statistical analysis, activation analysis, trajectory analysis and sensitivity analysis
"""

# Existing analysis modules
from .statistics import *

# Sensitivity analysis modules
try:
    from .sobol_analysis import SobolAnalyzer, SobolConfig
    from .sensitivity_metrics import SensitivityMetrics
    from .sensitivity_visualizer import SensitivityVisualizer
    
    __all__ = [
        # Statistical analysis
        'calculate_mean_opinion',
        'calculate_variance_metrics', 
        'calculate_identity_statistics',
        
        # Trajectory analysis
        'calculate_trajectory_metrics',
        
        # Sensitivity analysis
        'SobolAnalyzer',
        'SobolConfig', 
        'SensitivityMetrics',
        'SensitivityVisualizer'
    ]
    
except ImportError as e:
    # If sensitivity analysis dependencies are not available, only export other modules
    import warnings
    warnings.warn(f"Sensitivity analysis module unavailable: {e}. Please install required dependencies: pip install SALib pandas seaborn")
    
    __all__ = [
        # Statistical analysis
        'calculate_mean_opinion',
        'calculate_variance_metrics',
        'calculate_identity_statistics', 
        
        # Trajectory analysis
        'calculate_trajectory_metrics'
    ]
