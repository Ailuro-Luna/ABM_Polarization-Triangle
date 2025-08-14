import numpy as np
from typing import Dict, List, Optional, Tuple
from polarization_triangle.core.simulation import Simulation


def calculate_mean_opinion(sim: Simulation, exclude_zealots: bool = True) -> Dict[str, float]:
    """
    计算系统的平均意见统计
    
    参数:
    sim -- Simulation实例
    exclude_zealots -- 是否排除zealots，默认为True
    
    返回:
    包含平均意见统计的字典
    """
    opinions = sim.opinions.copy()
    
    # 如果需要排除zealots
    if exclude_zealots and hasattr(sim, 'zealot_ids') and sim.zealot_ids:
        non_zealot_opinions = np.delete(opinions, sim.zealot_ids)
    else:
        non_zealot_opinions = opinions
    
    stats = {
        "mean_opinion": float(np.mean(non_zealot_opinions)),
        "mean_abs_opinion": float(np.mean(np.abs(non_zealot_opinions))),
        "total_agents": len(non_zealot_opinions),
        "excluded_zealots": exclude_zealots and hasattr(sim, 'zealot_ids') and len(sim.zealot_ids) > 0
    }
    
    return stats


def calculate_variance_metrics(sim: Simulation, exclude_zealots: bool = True) -> Dict[str, float]:
    """
    计算系统的方差指标
    
    参数:
    sim -- Simulation实例
    exclude_zealots -- 是否排除zealots，默认为True
    
    返回:
    包含方差指标的字典
    """
    opinions = sim.opinions.copy()
    zealot_ids = sim.zealot_ids if hasattr(sim, 'zealot_ids') else []
    
    # 计算整体方差（排除zealots）
    if exclude_zealots and zealot_ids:
        non_zealot_opinions = np.delete(opinions, zealot_ids)
    else:
        non_zealot_opinions = opinions
    
    overall_variance = float(np.var(non_zealot_opinions))
    
    # 计算社区内部方差
    communities = {}
    for node in sim.graph.nodes():
        community = sim.graph.nodes[node].get("community")
        if isinstance(community, (set, frozenset)):
            community = min(community) if community else -1
        elif community is None:
            community = -1
        
        if community not in communities:
            communities[community] = []
        communities[community].append(node)
    
    # 计算每个社区的方差
    community_variances = []
    community_stats = {}
    
    for community_id, members in communities.items():
        # 过滤掉zealots
        if exclude_zealots and zealot_ids:
            community_non_zealots = [m for m in members if m not in zealot_ids]
        else:
            community_non_zealots = members
            
        if len(community_non_zealots) > 1:  # 至少需要2个agent才能计算方差
            community_opinions = opinions[community_non_zealots]
            community_var = float(np.var(community_opinions))
            community_variances.append(community_var)
            community_stats[f"community_{community_id}"] = {
                "variance": community_var,
                "size": len(community_non_zealots),
                "mean_opinion": float(np.mean(community_opinions))
            }
        elif len(community_non_zealots) == 1:
            community_variances.append(0.0)
            community_stats[f"community_{community_id}"] = {
                "variance": 0.0,
                "size": 1,
                "mean_opinion": float(opinions[community_non_zealots[0]])
            }
    
    # 平均社区内部方差
    mean_intra_community_variance = float(np.mean(community_variances)) if community_variances else 0.0
    
    return {
        "overall_variance": overall_variance,
        "mean_intra_community_variance": mean_intra_community_variance,
        "num_communities": len(communities),
        "community_details": community_stats
    }


def calculate_identity_statistics(sim: Simulation, exclude_zealots: bool = True) -> Dict[str, float]:
    """
    计算按身份分组的统计指标
    
    参数:
    sim -- Simulation实例
    exclude_zealots -- 是否排除zealots，默认为True
    
    返回:
    包含身份统计的字典
    """
    opinions = sim.opinions.copy()
    identities = sim.identities.copy()
    zealot_ids = sim.zealot_ids if hasattr(sim, 'zealot_ids') else []
    
    # 找到不同身份的agents（排除zealots）
    unique_identities = np.unique(identities)
    identity_stats = {}
    
    for identity_val in unique_identities:
        # 找到具有该身份的agents
        identity_agents = np.where(identities == identity_val)[0]
        
        # 排除zealots
        if exclude_zealots and zealot_ids:
            identity_agents = [agent for agent in identity_agents if agent not in zealot_ids]
        
        if len(identity_agents) > 0:
            identity_opinions = opinions[identity_agents]
            
            identity_stats[f"identity_{identity_val}"] = {
                "mean_opinion": float(np.mean(identity_opinions)),
                "variance": float(np.var(identity_opinions)) if len(identity_opinions) > 1 else 0.0,
                "std_dev": float(np.std(identity_opinions)) if len(identity_opinions) > 1 else 0.0,
                "count": len(identity_agents),
                "mean_abs_opinion": float(np.mean(np.abs(identity_opinions)))
            }
    
    # 计算身份间的意见差异
    identity_values = [key for key in identity_stats.keys()]
    if len(identity_values) >= 2:
        # 如果有identity +1和-1，计算它们的差异
        if "identity_1" in identity_stats and "identity_-1" in identity_stats:
            mean_diff = identity_stats["identity_1"]["mean_opinion"] - identity_stats["identity_-1"]["mean_opinion"]
            identity_stats["identity_difference"] = {
                "mean_opinion_difference": float(mean_diff),
                "abs_mean_opinion_difference": float(abs(mean_diff))
            }
    
    return identity_stats


def get_polarization_index(sim: Simulation) -> float:
    """
    获取当前系统的极化指数
    
    参数:
    sim -- Simulation实例
    
    返回:
    极化指数值
    """
    if hasattr(sim, 'calculate_polarization_index'):
        return float(sim.calculate_polarization_index())
    else:
        # 如果simulation没有这个方法，我们自己实现
        opinions = sim.opinions
        
        # 将观点离散化为5个类别
        category_counts = np.zeros(5, dtype=np.int32)
        
        for opinion in opinions:
            if opinion < -0.6:
                category_counts[0] += 1  # 类别1: 非常不同意
            elif opinion < -0.2:
                category_counts[1] += 1  # 类别2: 不同意
            elif opinion <= 0.2:
                category_counts[2] += 1  # 类别3: 中立
            elif opinion <= 0.6:
                category_counts[3] += 1  # 类别4: 同意
            else:
                category_counts[4] += 1  # 类别5: 非常同意
        
        # 获取各类别Agent数量
        n1, n2, n3, n4, n5 = category_counts
        N = sim.num_agents
        
        # 应用Koudenburg公式计算极化指数
        numerator = (2.14 * n2 * n4 + 
                    2.70 * (n1 * n4 + n2 * n5) + 
                    3.96 * n1 * n5)
        
        denominator = 0.0099 * (N ** 2)
        
        if denominator > 0:
            return float(numerator / denominator)
        else:
            return 0.0


def get_comprehensive_statistics(sim: Simulation, exclude_zealots: bool = True) -> Dict[str, any]:
    """
    获取系统的综合统计信息
    
    参数:
    sim -- Simulation实例
    exclude_zealots -- 是否排除zealots，默认为True
    
    返回:
    包含所有统计指标的综合字典
    """
    stats = {
        "mean_opinion_stats": calculate_mean_opinion(sim, exclude_zealots),
        "variance_metrics": calculate_variance_metrics(sim, exclude_zealots),
        "identity_statistics": calculate_identity_statistics(sim, exclude_zealots),
        "polarization_index": get_polarization_index(sim),
        "system_info": {
            "num_agents": sim.num_agents,
            "num_zealots": len(sim.zealot_ids) if hasattr(sim, 'zealot_ids') and sim.zealot_ids else 0,
            "exclude_zealots_flag": exclude_zealots
        }
    }
    
    return stats


def print_statistics_summary(sim: Simulation, exclude_zealots: bool = True) -> None:
    """
    打印系统统计信息的简洁摘要
    
    参数:
    sim -- Simulation实例
    exclude_zealots -- 是否排除zealots，默认为True
    """
    stats = get_comprehensive_statistics(sim, exclude_zealots)
    
    print("=" * 60)
    print("系统统计摘要 (System Statistics Summary)")
    print("=" * 60)
    
    # 基本信息
    print(f"总Agent数量: {stats['system_info']['num_agents']}")
    print(f"Zealot数量: {stats['system_info']['num_zealots']}")
    print(f"是否排除Zealots: {'是' if exclude_zealots else '否'}")
    print()
    
    # 平均意见统计
    mean_stats = stats['mean_opinion_stats']
    print("平均意见统计:")
    print(f"  平均意见值: {mean_stats['mean_opinion']:.4f}")
    print(f"  平均绝对意见值: {mean_stats['mean_abs_opinion']:.4f}")
    print(f"  统计的Agent数量: {mean_stats['total_agents']}")
    print()
    
    # 方差指标
    var_stats = stats['variance_metrics']
    print("方差指标:")
    print(f"  整体方差: {var_stats['overall_variance']:.4f}")
    print(f"  平均社区内部方差: {var_stats['mean_intra_community_variance']:.4f}")
    print(f"  社区数量: {var_stats['num_communities']}")
    print()
    
    # 身份统计
    identity_stats = stats['identity_statistics']
    print("身份统计:")
    for key, values in identity_stats.items():
        if key.startswith('identity_') and key != 'identity_difference':
            print(f"  {key}:")
            print(f"    平均意见: {values['mean_opinion']:.4f}")
            print(f"    方差: {values['variance']:.4f}")
            print(f"    数量: {values['count']}")
    
    if 'identity_difference' in identity_stats:
        diff_stats = identity_stats['identity_difference']
        print(f"  身份间意见差异: {diff_stats['mean_opinion_difference']:.4f}")
        print(f"  身份间意见差异(绝对值): {diff_stats['abs_mean_opinion_difference']:.4f}")
    print()
    
    # 极化指数
    print(f"极化指数 (Polarization Index): {stats['polarization_index']:.4f}")
    print("=" * 60)


def export_statistics_to_dict(sim: Simulation, exclude_zealots: bool = True) -> Dict[str, float]:
    """
    将统计结果导出为扁平化的字典，便于保存到CSV或进一步分析
    
    参数:
    sim -- Simulation实例
    exclude_zealots -- 是否排除zealots，默认为True
    
    返回:
    扁平化的统计数据字典
    """
    stats = get_comprehensive_statistics(sim, exclude_zealots)
    
    # 创建扁平化字典
    flat_dict = {}
    
    # 基本信息
    flat_dict['num_agents'] = stats['system_info']['num_agents']
    flat_dict['num_zealots'] = stats['system_info']['num_zealots']
    
    # 平均意见统计
    mean_stats = stats['mean_opinion_stats']
    flat_dict['mean_opinion'] = mean_stats['mean_opinion']
    flat_dict['mean_abs_opinion'] = mean_stats['mean_abs_opinion']
    flat_dict['total_agents_counted'] = mean_stats['total_agents']
    
    # 方差指标
    var_stats = stats['variance_metrics']
    flat_dict['overall_variance'] = var_stats['overall_variance']
    flat_dict['mean_intra_community_variance'] = var_stats['mean_intra_community_variance']
    flat_dict['num_communities'] = var_stats['num_communities']
    
    # 身份统计
    identity_stats = stats['identity_statistics']
    for key, values in identity_stats.items():
        if key.startswith('identity_') and key != 'identity_difference':
            flat_dict[f'{key}_mean_opinion'] = values['mean_opinion']
            flat_dict[f'{key}_variance'] = values['variance']
            flat_dict[f'{key}_count'] = values['count']
            flat_dict[f'{key}_mean_abs_opinion'] = values['mean_abs_opinion']
    
    if 'identity_difference' in identity_stats:
        diff_stats = identity_stats['identity_difference']
        flat_dict['identity_mean_opinion_difference'] = diff_stats['mean_opinion_difference']
        flat_dict['identity_abs_mean_opinion_difference'] = diff_stats['abs_mean_opinion_difference']
    
    # 极化指数
    flat_dict['polarization_index'] = stats['polarization_index']
    
    return flat_dict 