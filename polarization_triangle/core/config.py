# config.py
from dataclasses import dataclass, field
from typing import Dict, Any
import copy

@dataclass
class SimulationConfig:
    num_agents: int = 100
    network_type: str = "lfr"  # 可选："random"、"community"、"lfr"、"ws"、"ba"
    network_params: Dict[str, Any] = field(default_factory=lambda: {
        "tau1": 3,
        "tau2": 1.5,
        "mu": 0.1,
        "average_degree": 5,
        "min_community": 50,
    })
    opinion_distribution: str = "twin_peak"  # 可选："uniform"、"single_peak"、"twin_peak"、"skewed"
    coupling: str = "partial"  # 可选："none"、"partial"、"strong"
    extreme_fraction: float = 0.1
    moral_correlation: str = "partial"  # 可选："none"、"partial"、"strong"
    morality_rate: float = 0.5  # 道德化率：0.0 到 1.0 之间的值，表示道德化的比例
    # 聚类参数
    cluster_identity: bool = False
    cluster_morality: bool = False
    cluster_opinion: bool = False
    cluster_identity_prob: float = 1
    cluster_morality_prob: float = 0.8
    cluster_opinion_prob: float = 0.8
    # 仿真更新参数
    influence_factor: float = 0.1
    tolerance: float = 0.6
    # 更新概率参数（意见更新时用到的硬编码概率）
    p_radical_high: float = 0.7
    p_radical_low: float = 0.3
    p_conv_high: float = 0.7
    p_conv_low: float = 0.3

    # identity相关参数
    identity_issue_mapping: Dict[int, float] = field(default_factory=lambda: {1: 0.3, -1: -0.3})
    identity_influence_factor: float = 0.2
    cohesion_factor: float = 0.2

    # 身份规范强度参数
    identity_antagonism_threshold: float = 0.8  # 小于1的常数参数A，定义对抗阈值

    # Zealot相关参数
    zealot_count: int = 0  # zealot的数量，0表示不使用zealot
    zealot_mode: str = "random"  # 选择模式：random, clustered, degree
    zealot_opinion: float = 1.0  # zealot固定的意见值
    enable_zealots: bool = False  # 是否启用zealot功能
    zealot_morality: bool = False  # zealot是否全部设置为moralizing (morality=1)
    zealot_identity_allocation: bool = True  # 是否按identity分配zealot，默认启用，启用时zealot只分配给identity为1的agent
    
    # 极化三角框架模型参数
    delta: float = 1  # 意见衰减率
    u: float = 1  # 意见激活系数
    alpha: float = 0.4  # 自我激活系数
    beta: float = 0.12  # 社会影响系数
    gamma: float = 1  # 道德化影响系数
    
    # 网络池相关参数
    use_network_pool: bool = True  # 是否使用网络池（仅对LFR网络有效）
    # network_pool_dir: str = "network_cache/default_pool"  # 网络池目录
    network_pool_dir: str = "network_cache/default_pool"
    network_pool_random_selection: bool = True  # 是否随机选择网络池中的网络
    
    def copy(self):
        """创建当前配置的副本"""
        import copy
        return copy.deepcopy(self)

# 预设配置：

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
    morality_rate=0.5,  # 中等道德化率
    # 极化三角框架模型参数保持默认值
)

high_polarization_config = copy.deepcopy(base_config)
high_polarization_config.alpha = 0.6
# 默认使用的配置
config = base_config