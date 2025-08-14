#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
网络池管理器
用于批量生成、保存和加载LFR网络池，避免在仿真过程中重复生成网络
"""

import os
import json
import pickle
import time
import logging
from typing import Dict, List, Optional, Tuple
import networkx as nx
from pathlib import Path

from .lfr_generator import generate_lfr_network

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkPool:
    """
    网络池管理器
    负责批量生成、保存和加载LFR网络
    """
    
    def __init__(self, pool_dir: str):
        """
        初始化网络池管理器
        
        参数:
            pool_dir: 网络池存储目录
        """
        self.pool_dir = Path(pool_dir)
        self.metadata_file = self.pool_dir / "pool_metadata.json"
        self.networks_dir = self.pool_dir / "networks"
        
        # 确保目录存在
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.networks_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载或初始化元数据
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """加载网络池元数据"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载元数据失败: {e}")
                return self._create_empty_metadata()
        else:
            return self._create_empty_metadata()
    
    def _create_empty_metadata(self) -> Dict:
        """创建空的元数据结构"""
        return {
            "pool_info": {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_networks": 0,
                "lfr_params": {}
            },
            "networks": {}
        }
    
    def _save_metadata(self):
        """保存元数据到文件"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存元数据失败: {e}")
    
    def generate_pool(
        self, 
        pool_size: int,
        lfr_params: Optional[Dict] = None,
        start_seed: int = 42,
        skip_existing: bool = True
    ) -> bool:
        """
        批量生成网络池
        
        参数:
            pool_size: 要生成的网络数量
            lfr_params: LFR参数字典，如果为None则使用默认参数
            start_seed: 起始随机种子
            skip_existing: 是否跳过已存在的网络
            
        返回:
            成功返回True，失败返回False
        """
        # 使用默认LFR参数
        if lfr_params is None:
            lfr_params = {
                "n": 500,
                "tau1": 3.0,
                "tau2": 1.5,
                "mu": 0.1,
                "average_degree": 5,
                "min_community": 30,
                "timeout": 60
            }
        
        logger.info(f"开始生成网络池: {pool_size}个网络")
        logger.info(f"LFR参数: {lfr_params}")
        
        # 更新元数据中的参数信息
        self.metadata["pool_info"]["lfr_params"] = lfr_params
        
        success_count = 0
        total_start_time = time.time()
        
        for i in range(pool_size):
            network_id = f"network_{i:04d}"
            network_file = self.networks_dir / f"{network_id}.pkl"
            
            # 检查是否跳过已存在的网络
            if skip_existing and network_file.exists():
                logger.info(f"跳过已存在的网络: {network_id}")
                success_count += 1
                continue
            
            # 生成网络
            seed = start_seed + i
            logger.info(f"生成第 {i+1}/{pool_size} 个网络 (seed={seed})")
            
            G = generate_lfr_network(seed=seed, **lfr_params)
            
            if G is not None:
                # 保存网络
                try:
                    with open(network_file, 'wb') as f:
                        pickle.dump(G, f)
                    
                    # 更新元数据
                    self.metadata["networks"][network_id] = {
                        "file": f"networks/{network_id}.pkl",
                        "seed": seed,
                        "nodes": G.number_of_nodes(),
                        "edges": G.number_of_edges(),
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "lfr_params": lfr_params.copy()
                    }
                    
                    success_count += 1
                    logger.info(f"网络 {network_id} 生成并保存成功")
                    
                except Exception as e:
                    logger.error(f"保存网络 {network_id} 失败: {e}")
            else:
                logger.error(f"生成网络 {network_id} 失败")
        
        # 更新总体元数据
        self.metadata["pool_info"]["total_networks"] = success_count
        self.metadata["pool_info"]["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 保存元数据
        self._save_metadata()
        
        total_time = time.time() - total_start_time
        logger.info(f"网络池生成完成: {success_count}/{pool_size} 成功, 总用时 {total_time:.1f} 秒")
        
        return success_count > 0
    
    def load_network(self, network_id: Optional[str] = None, index: Optional[int] = None) -> Optional[nx.Graph]:
        """
        从网络池加载网络
        
        参数:
            network_id: 网络ID（如"network_0001"）
            index: 网络索引（0开始）
            
        返回:
            成功返回networkx.Graph对象，失败返回None
        """
        # 根据索引确定网络ID
        if index is not None:
            network_id = f"network_{index:04d}"
        
        if network_id is None:
            logger.error("必须提供network_id或index参数")
            return None
        
        # 检查网络是否存在于元数据中
        if network_id not in self.metadata["networks"]:
            logger.error(f"网络 {network_id} 不存在于池中")
            return None
        
        # 获取网络文件路径
        network_file = self.pool_dir / self.metadata["networks"][network_id]["file"]
        
        if not network_file.exists():
            logger.error(f"网络文件不存在: {network_file}")
            return None
        
        # 加载网络
        try:
            with open(network_file, 'rb') as f:
                G = pickle.load(f)
            logger.debug(f"成功加载网络 {network_id}: {G.number_of_nodes()}个节点")
            return G
        except Exception as e:
            logger.error(f"加载网络 {network_id} 失败: {e}")
            return None
    
    def get_random_network(self) -> Optional[nx.Graph]:
        """
        随机获取一个网络
        
        返回:
            随机的networkx.Graph对象，如果池为空返回None
        """
        if not self.metadata["networks"]:
            logger.error("网络池为空")
            return None
        
        import random
        network_id = random.choice(list(self.metadata["networks"].keys()))
        return self.load_network(network_id)
    
    def get_pool_info(self) -> Dict:
        """
        获取网络池信息
        
        返回:
            包含池信息的字典
        """
        return {
            "pool_directory": str(self.pool_dir),
            "total_networks": len(self.metadata["networks"]),
            "lfr_params": self.metadata["pool_info"].get("lfr_params", {}),
            "created_at": self.metadata["pool_info"].get("created_at", "Unknown"),
            "last_updated": self.metadata["pool_info"].get("last_updated", "Unknown")
        }
    
    def list_networks(self) -> List[Dict]:
        """
        列出池中所有网络的信息
        
        返回:
            网络信息列表
        """
        networks = []
        for network_id, info in self.metadata["networks"].items():
            networks.append({
                "id": network_id,
                "nodes": info["nodes"],
                "edges": info["edges"], 
                "seed": info["seed"],
                "created_at": info["created_at"]
            })
        return sorted(networks, key=lambda x: x["id"])


def create_default_pool(pool_dir: str, pool_size: int = 50) -> NetworkPool:
    """
    创建默认参数的网络池
    
    参数:
        pool_dir: 池存储目录
        pool_size: 池大小
        
    返回:
        NetworkPool对象
    """
    pool = NetworkPool(pool_dir)
    
    # 使用与config.py中一致的默认参数
    default_params = {
        "n": 500,
        "tau1": 3.0,
        "tau2": 1.5,
        "mu": 0.1,
        "average_degree": 5,
        "min_community": 30,
        "timeout": 60
    }
    
    pool.generate_pool(pool_size, default_params)
    return pool


if __name__ == "__main__":
    # 简单测试
    print("测试网络池系统...")
    
    test_pool_dir = "./test_network_pool"
    pool = NetworkPool(test_pool_dir)
    
    # 生成小规模测试池
    success = pool.generate_pool(pool_size=3, lfr_params={"n": 50, "mu": 0.1})
    
    if success:
        # 测试加载
        print("池信息:")
        info = pool.get_pool_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # 测试网络加载
        G = pool.load_network(index=0)
        if G:
            print(f"成功加载网络: {G.number_of_nodes()}个节点, {G.number_of_edges()}条边")
        
        # 清理测试文件
        import shutil
        if os.path.exists(test_pool_dir):
            shutil.rmtree(test_pool_dir)
            print("清理测试文件完成")
    else:
        print("网络池生成失败")
