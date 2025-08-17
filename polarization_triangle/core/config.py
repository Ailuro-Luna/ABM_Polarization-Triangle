# config.py
from dataclasses import dataclass, field
from typing import Dict, Any
import copy

@dataclass
class SimulationConfig:
    num_agents: int = 100
    network_type: str = "lfr"  # Options: "random", "community", "lfr", "ws", "ba"
    network_params: Dict[str, Any] = field(default_factory=lambda: {
        "tau1": 3,
        "tau2": 1.5,
        "mu": 0.1,
        "average_degree": 5,
        "min_community": 50,
    })
    opinion_distribution: str = "twin_peak"  # Options: "uniform", "single_peak", "twin_peak", "skewed"
    coupling: str = "partial"  # Options: "none", "partial", "strong"
    extreme_fraction: float = 0.1
    moral_correlation: str = "partial"  # Options: "none", "partial", "strong"
    morality_rate: float = 0.5  # Moralization rate: value between 0.0 and 1.0, representing the proportion of moralization
    # Clustering parameters
    cluster_identity: bool = False
    cluster_morality: bool = False
    cluster_opinion: bool = False
    cluster_identity_prob: float = 1
    cluster_morality_prob: float = 0.8
    cluster_opinion_prob: float = 0.8
    # Simulation update parameters
    influence_factor: float = 0.1
    tolerance: float = 0.6
    # Update probability parameters (hardcoded probabilities used in opinion updates)
    p_radical_high: float = 0.7
    p_radical_low: float = 0.3
    p_conv_high: float = 0.7
    p_conv_low: float = 0.3

    # identity-related parameters
    identity_issue_mapping: Dict[int, float] = field(default_factory=lambda: {1: 0.3, -1: -0.3})
    identity_influence_factor: float = 0.2
    cohesion_factor: float = 0.2

    # Identity norm strength parameters
    identity_antagonism_threshold: float = 0.8  # Constant parameter A less than 1, defining antagonism threshold

    # Zealot-related parameters
    zealot_count: int = 0  # Number of zealots, 0 means not using zealots
    zealot_mode: str = "random"  # Selection mode: random, clustered, degree
    zealot_opinion: float = 1.0  # Fixed opinion value for zealots
    enable_zealots: bool = False  # Whether to enable zealot functionality
    zealot_morality: bool = False  # Whether all zealots are set to moralizing (morality=1)
    zealot_identity_allocation: bool = True  # Whether to allocate zealots by identity, enabled by default, zealots are only allocated to agents with identity=1 when enabled
    
    # Polarization triangle framework model parameters
    delta: float = 1  # Opinion decay rate
    u: float = 1  # Opinion activation coefficient
    alpha: float = 0.4  # Self-activation coefficient
    beta: float = 0.12  # Social influence coefficient
    gamma: float = 1  # Moralization influence coefficient
    
    # Network pool related parameters
    use_network_pool: bool = True  # Whether to use network pool (only valid for LFR networks)
    # network_pool_dir: str = "network_cache/default_pool"  # Network pool directory
    network_pool_dir: str = "network_cache/default_pool"
    network_pool_random_selection: bool = True  # Whether to randomly select networks from network pool
    
    def copy(self):
        """Create a copy of the current configuration"""
        import copy
        return copy.deepcopy(self)

# Preset configurations:

base_config = SimulationConfig(
    num_agents=500,
    network_type="lfr",
    network_params={
        "tau1": 3,
        "tau2": 1.5,
        "mu": 0.1,
        "average_degree": 5,
        "min_community": 30
    },
    opinion_distribution="twin_peak",
    coupling="none",
    extreme_fraction=0.1,
    moral_correlation="none",
    cluster_identity=True,
    cluster_morality=True,
    cluster_opinion=True,
    cluster_opinion_prob=0.8,
    morality_rate=0.5,  # Medium moralization rate
    # Polarization triangle framework model parameters maintain default values
)

high_polarization_config = copy.deepcopy(base_config)
high_polarization_config.alpha = 0.6
# Default configuration to use
config = base_config