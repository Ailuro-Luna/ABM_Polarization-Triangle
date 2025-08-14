#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LFR网络生成器
用于批量生成和保存LFR网络，避免在仿真过程中重复生成导致卡死
"""

import networkx as nx
import pickle
import os
import time
from typing import Dict, Any, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_lfr_network(
    n: int = 500,
    tau1: float = 3.0,
    tau2: float = 1.5,
    mu: float = 0.1,
    average_degree: int = 5,
    min_community: int = 30,
    max_community: Optional[int] = None,
    seed: Optional[int] = None,
    max_iters: int = 300,
    timeout: int = 60
) -> Optional[nx.Graph]:
    """
    生成LFR基准网络
    
    参数:
        n: 节点数量
        tau1: 度分布的幂律指数
        tau2: 社区大小分布的幂律指数
        mu: 混合参数（节点连接到其他社区的边的比例）
        average_degree: 平均度数
        min_community: 最小社区大小
        max_community: 最大社区大小
        seed: 随机种子
        max_iters: 最大迭代次数
        timeout: 超时时间（秒）
        
    返回:
        成功时返回networkx.Graph对象，失败时返回None
    """
    logger.info(f"开始生成LFR网络: n={n}, mu={mu}, avg_degree={average_degree}")
    
    start_time = time.time()
    
    try:
        # 参数验证
        if max_community is None:
            max_community = n // 2
            
        # 确保参数合理
        if min_community > n // 2:
            min_community = n // 4
            logger.warning(f"最小社区大小太大，调整为 {min_community}")
            
        if max_community > n:
            max_community = n // 2
            logger.warning(f"最大社区大小太大，调整为 {max_community}")
        
        # 生成LFR网络
        G = nx.LFR_benchmark_graph(
            n=n,
            tau1=tau1,
            tau2=tau2,
            mu=mu,
            average_degree=average_degree,
            min_community=min_community,
            max_community=max_community,
            max_iters=max_iters,
            seed=seed
        )
        
        # 检查是否超时
        if time.time() - start_time > timeout:
            logger.error(f"网络生成超时 ({timeout}秒)")
            return None
            
        # 处理网络连通性
        if not nx.is_connected(G):
            logger.info("网络不连通，正在修复连通性...")
            _ensure_connectivity(G)
            
        logger.info(f"网络生成完成: {G.number_of_nodes()}个节点, {G.number_of_edges()}条边, "
                   f"用时{time.time() - start_time:.2f}秒")
        
        return G
        
    except Exception as e:
        logger.error(f"生成LFR网络时发生错误: {e}")
        return None


def save_lfr_network(G: nx.Graph, filepath: str) -> bool:
    """
    保存LFR网络到文件
    
    参数:
        G: networkx图对象
        filepath: 保存路径（.pkl文件）
        
    返回:
        成功返回True，失败返回False
    """
    try:
        # 确保目录存在（如果有父目录）
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        
        # 保存网络
        with open(filepath, 'wb') as f:
            pickle.dump(G, f)
            
        file_size = os.path.getsize(filepath)
        logger.info(f"网络已保存到 {filepath}, 文件大小: {file_size/1024:.1f} KB")
        return True
        
    except Exception as e:
        logger.error(f"保存网络时发生错误: {e}")
        return False


def _ensure_connectivity(G: nx.Graph) -> None:
    """
    确保网络连通性
    将所有连通分量连接成一个连通网络
    """
    import random
    
    # 获取所有连通分量
    components = list(nx.connected_components(G))
    
    if len(components) <= 1:
        return  # 网络已经连通
    
    logger.info(f"发现 {len(components)} 个连通分量，正在连接...")
    
    # 找到最大的连通分量作为主分量
    largest_component = max(components, key=len)
    largest_nodes = list(largest_component)
    
    # 将其他分量连接到主分量
    for component in components:
        if component == largest_component:
            continue
            
        component_nodes = list(component)
        
        # 从当前分量随机选择1-2个节点
        source_nodes = random.sample(component_nodes, min(2, len(component_nodes)))
        
        # 从主分量随机选择对应数量的节点
        target_nodes = random.sample(largest_nodes, len(source_nodes))
        
        # 建立连接
        for source, target in zip(source_nodes, target_nodes):
            G.add_edge(source, target)
            logger.debug(f"连接分量: {source} -> {target}")
    
    # 验证连通性
    if nx.is_connected(G):
        logger.info(f"网络连通性修复完成，最终有 {G.number_of_edges()} 条边")
    else:
        logger.warning("连通性修复可能失败，网络仍然不连通")


if __name__ == "__main__":
    # 简单测试
    print("测试LFR网络生成...")
    
    G = generate_lfr_network(n=100, mu=0.1, seed=42)
    if G:
        print(f"生成成功: {G.number_of_nodes()}个节点, {G.number_of_edges()}条边")
        
        # 测试保存
        test_path = "./test_lfr_network.pkl"
        if save_lfr_network(G, test_path):
            print(f"保存成功: {test_path}")
            # 清理测试文件
            if os.path.exists(test_path):
                os.remove(test_path)
                print("清理测试文件完成")
    else:
        print("生成失败")
