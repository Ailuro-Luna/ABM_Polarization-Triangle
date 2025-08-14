#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
网络工具模块
提供网络创建和处理的工具函数
"""

import networkx as nx
import numpy as np
from typing import Dict, Any


def create_network(num_agents: int, network_type: str, network_params: Dict[str, Any] = None, 
                   use_network_pool: bool = True, network_pool_dir: str = None,
                   network_pool_random_selection: bool = True) -> nx.Graph:
    """
    根据指定的类型和参数创建网络
    
    参数:
    num_agents -- 代理数量
    network_type -- 网络类型 ("random", "lfr", "community", "ws", "ba")
    network_params -- 网络参数字典
    use_network_pool -- 是否使用网络池（仅对LFR网络有效）
    network_pool_dir -- 网络池目录
    network_pool_random_selection -- 是否随机选择网络池中的网络
    
    返回:
    创建的networkx图对象
    """
    if network_params is None:
        network_params = {}
        
    if network_type == 'random':
        p = network_params.get("p", 0.1)
        return nx.erdos_renyi_graph(n=num_agents, p=p)
    elif network_type == 'lfr':
        # 检查是否使用网络池
        if use_network_pool and network_pool_dir:
            print(f"从网络池加载LFR网络: {network_pool_dir}")
            try:
                from .network_pool import NetworkPool
                pool = NetworkPool(network_pool_dir)
                
                if network_pool_random_selection:
                    G = pool.get_random_network()
                else:
                    G = pool.load_network(index=0)  # 加载第一个网络
                
                if G is not None:
                    # 检查节点数量是否匹配
                    if G.number_of_nodes() != num_agents:
                        print(f"警告: 网络池中的网络节点数({G.number_of_nodes()})与配置不符({num_agents})")
                    return G
                else:
                    print("从网络池加载失败，回退到实时生成")
            except Exception as e:
                print(f"网络池加载失败: {e}，回退到实时生成")
        
        # 实时生成LFR网络（原有逻辑）
        tau1 = network_params.get("tau1", 3)
        tau2 = network_params.get("tau2", 1.5)
        mu = network_params.get("mu", 0.1)
        average_degree = network_params.get("average_degree", 5)
        min_community = network_params.get("min_community", 10)
        seed = network_params.get("seed", 42)
        # 可选快速失败与约束
        max_iters = network_params.get("max_iters", 300)
        max_community = network_params.get("max_community")
        min_degree = network_params.get("min_degree")
        max_degree = network_params.get("max_degree")

        return nx.LFR_benchmark_graph(
            n=num_agents,
            tau1=tau1,
            tau2=tau2,
            mu=mu,
            average_degree=average_degree,
            min_community=min_community,
            max_community=max_community,
            min_degree=min_degree,
            max_degree=max_degree,
            max_iters=max_iters,
            seed=seed
        )
    elif network_type == 'community':
        community_sizes = [num_agents // 4] * 4
        intra_p = network_params.get("intra_p", 0.8)
        inter_p = network_params.get("inter_p", 0.1)
        return nx.random_partition_graph(community_sizes, intra_p, inter_p)
    elif network_type == 'ws':
        k = network_params.get("k", 4)
        p = network_params.get("p", 0.1)
        return nx.watts_strogatz_graph(n=num_agents, k=k, p=p)
    elif network_type == 'ba':
        m = network_params.get("m", 2)
        return nx.barabasi_albert_graph(n=num_agents, m=m)
    else:
        return nx.erdos_renyi_graph(n=num_agents, p=0.1)


def handle_isolated_nodes(G: nx.Graph) -> None:
    """
    处理网络中的孤立点
    
    参数:
    G -- 网络图对象
    
    处理方式:
    1. 找出所有孤立点（度为0的节点）
    2. 为每个孤立点随机连接到网络中的其他节点
    """
    isolated_nodes = [node for node, degree in dict(G.degree()).items() if degree == 0]

    if not isolated_nodes:
        return  # 如果没有孤立点，直接返回

    print(f"检测到 {len(isolated_nodes)} 个孤立点，进行处理...")

    # 获取非孤立节点列表
    non_isolated = [node for node in G.nodes() if node not in isolated_nodes]

    if not non_isolated:
        # 如果所有节点都是孤立的（极少情况），创建一个环形连接
        for i in range(len(isolated_nodes)):
            G.add_edge(isolated_nodes[i], isolated_nodes[(i + 1) % len(isolated_nodes)])
        return

    # 为每个孤立点随机连接到1-3个非孤立节点
    for node in isolated_nodes:
        # 随机决定连接数量，最小1个，最大3个或所有非孤立节点数
        num_connections = min(np.random.randint(1, 4), len(non_isolated))
        # 随机选择连接目标
        targets = np.random.choice(non_isolated, num_connections, replace=False)
        # 添加边
        for target in targets:
            G.add_edge(node, target)
