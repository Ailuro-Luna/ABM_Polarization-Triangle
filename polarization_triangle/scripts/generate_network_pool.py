#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
网络池生成脚本
用于批量预生成LFR网络池，供后续仿真使用
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from polarization_triangle.utils.network_pool import NetworkPool, create_default_pool
from polarization_triangle.core.config import base_config


def main():
    parser = argparse.ArgumentParser(description="生成LFR网络池")
    parser.add_argument("--pool-dir", type=str, default="network_cache/default_pool",
                        help="网络池存储目录（默认: network_cache/default_pool）")
    parser.add_argument("--pool-size", type=int, default=100,
                        help="网络池大小（默认: 100）")
    parser.add_argument("--nodes", type=int, default=500,
                        help="网络节点数量（默认: 500）")
    parser.add_argument("--mu", type=float, default=0.1,
                        help="LFR混合参数mu（默认: 0.1）")
    parser.add_argument("--avg-degree", type=int, default=5,
                        help="平均度数（默认: 5）")
    parser.add_argument("--min-community", type=int, default=30,
                        help="最小社区大小（默认: 30）")
    parser.add_argument("--start-seed", type=int, default=42,
                        help="起始随机种子（默认: 42）")
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过已存在的网络文件")
    parser.add_argument("--info", action="store_true",
                        help="只显示已存在网络池的信息")
    parser.add_argument("--list", action="store_true",
                        help="列出池中所有网络")
    
    args = parser.parse_args()
    
    # 如果只是查看信息
    if args.info or args.list:
        if not os.path.exists(args.pool_dir):
            print(f"网络池目录不存在: {args.pool_dir}")
            return
        
        pool = NetworkPool(args.pool_dir)
        
        if args.info:
            print("=== 网络池信息 ===")
            info = pool.get_pool_info()
            for key, value in info.items():
                print(f"{key}: {value}")
        
        if args.list:
            print("\n=== 网络列表 ===")
            networks = pool.list_networks()
            print(f"{'ID':<12} {'节点数':<8} {'边数':<8} {'种子':<8} {'创建时间'}")
            print("-" * 60)
            for net in networks:
                print(f"{net['id']:<12} {net['nodes']:<8} {net['edges']:<8} {net['seed']:<8} {net['created_at']}")
        
        return
    
    # 生成网络池
    print("=== LFR网络池生成器 ===")
    print(f"存储目录: {args.pool_dir}")
    print(f"池大小: {args.pool_size}")
    print(f"节点数: {args.nodes}")
    print(f"混合参数mu: {args.mu}")
    print(f"平均度数: {args.avg_degree}")
    print(f"最小社区: {args.min_community}")
    print(f"起始种子: {args.start_seed}")
    print()
    
    # 创建网络池
    pool = NetworkPool(args.pool_dir)
    
    # 设置LFR参数
    lfr_params = {
        "n": args.nodes,
        "tau1": 3.0,
        "tau2": 1.5,
        "mu": args.mu,
        "average_degree": args.avg_degree,
        "min_community": args.min_community,
        "timeout": 60
    }
    
    # 生成网络池
    success = pool.generate_pool(
        pool_size=args.pool_size,
        lfr_params=lfr_params,
        start_seed=args.start_seed,
        skip_existing=args.skip_existing
    )
    
    if success:
        print("\n=== 生成完成 ===")
        info = pool.get_pool_info()
        print(f"成功生成 {info['total_networks']} 个网络")
        print(f"存储位置: {info['pool_directory']}")
        
        # 测试随机加载一个网络
        G = pool.get_random_network()
        if G:
            print(f"测试加载: 成功加载网络({G.number_of_nodes()}节点, {G.number_of_edges()}条边)")
    else:
        print("网络池生成失败")
        sys.exit(1)


def generate_default_pools():
    """
    生成几个常用参数的默认网络池
    """
    print("生成默认网络池...")
    
    # 配置不同的参数组合
    pool_configs = [
        {"name": "default", "mu": 0.1, "nodes": 500, "size": 50},
        {"name": "high_mixing", "mu": 0.3, "nodes": 500, "size": 30},
        {"name": "low_mixing", "mu": 0.05, "nodes": 500, "size": 30},
    ]
    
    for config in pool_configs:
        pool_dir = f"network_cache/{config['name']}_pool"
        print(f"生成 {config['name']} 池 (mu={config['mu']})...")
        
        pool = NetworkPool(pool_dir)
        lfr_params = {
            "n": config["nodes"],
            "mu": config["mu"],
            "average_degree": 5,
            "min_community": 10,
            "timeout": 60
        }
        
        pool.generate_pool(config["size"], lfr_params, skip_existing=True)
        print(f"完成: {pool_dir}")
    
    print("所有默认网络池生成完成")


if __name__ == "__main__":
    main()
