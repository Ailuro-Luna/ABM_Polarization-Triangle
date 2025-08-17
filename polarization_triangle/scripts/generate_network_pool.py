#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network pool generation script
Used for batch pre-generation of LFR network pools for subsequent simulations
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from polarization_triangle.utils.network_pool import NetworkPool, create_default_pool
from polarization_triangle.core.config import base_config


def main():
    parser = argparse.ArgumentParser(description="Generate LFR network pool")
    parser.add_argument("--pool-dir", type=str, default="network_cache/default_pool",
                        help="Network pool storage directory (default: network_cache/default_pool)")
    parser.add_argument("--pool-size", type=int, default=100,
                        help="Network pool size (default: 100)")
    parser.add_argument("--nodes", type=int, default=500,
                        help="Number of network nodes (default: 500)")
    parser.add_argument("--mu", type=float, default=0.1,
                        help="LFR mixing parameter mu (default: 0.1)")
    parser.add_argument("--avg-degree", type=int, default=5,
                        help="Average degree (default: 5)")
    parser.add_argument("--min-community", type=int, default=30,
                        help="Minimum community size (default: 30)")
    parser.add_argument("--start-seed", type=int, default=42,
                        help="Starting random seed (default: 42)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip existing network files")
    parser.add_argument("--info", action="store_true",
                        help="Only display information about existing network pool")
    parser.add_argument("--list", action="store_true",
                        help="List all networks in pool")
    
    args = parser.parse_args()
    
    # If only viewing information
    if args.info or args.list:
        if not os.path.exists(args.pool_dir):
            print(f"Network pool directory does not exist: {args.pool_dir}")
            return
        
        pool = NetworkPool(args.pool_dir)
        
        if args.info:
            print("=== Network Pool Information ===")
            info = pool.get_pool_info()
            for key, value in info.items():
                print(f"{key}: {value}")
        
        if args.list:
            print("\n=== Network List ===")
            networks = pool.list_networks()
            print(f"{'ID':<12} {'Nodes':<8} {'Edges':<8} {'Seed':<8} {'Created Time'}")
            print("-" * 60)
            for net in networks:
                print(f"{net['id']:<12} {net['nodes']:<8} {net['edges']:<8} {net['seed']:<8} {net['created_at']}")
        
        return
    
    # Generate network pool
    print("=== LFR Network Pool Generator ===")
    print(f"Storage directory: {args.pool_dir}")
    print(f"Pool size: {args.pool_size}")
    print(f"Nodes: {args.nodes}")
    print(f"Mixing parameter mu: {args.mu}")
    print(f"Average degree: {args.avg_degree}")
    print(f"Minimum community: {args.min_community}")
    print(f"Starting Seed: {args.start_seed}")
    print()
    
    # Create network pool
    pool = NetworkPool(args.pool_dir)
    
    # Set LFR parameters
    lfr_params = {
        "n": args.nodes,
        "tau1": 3.0,
        "tau2": 1.5,
        "mu": args.mu,
        "average_degree": args.avg_degree,
        "min_community": args.min_community,
        "timeout": 60
    }
    
    # Generate network pool
    success = pool.generate_pool(
        pool_size=args.pool_size,
        lfr_params=lfr_params,
        start_seed=args.start_seed,
        skip_existing=args.skip_existing
    )
    
    if success:
        print("\n=== Generation Complete ===")
        info = pool.get_pool_info()
        print(f"Successfully generated {info['total_networks']} networks")
        print(f"Storage location: {info['pool_directory']}")
        
        # Test loading a random network
        G = pool.get_random_network()
        if G:
            print(f"Test loading: Successfully loaded network ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    else:
        print("Network pool generation failed")
        sys.exit(1)


def generate_default_pools():
    """
    Generate default network pools with commonly used parameters
    """
    print("Generating default network pools...")
    
    # Configure different parameter combinations
    pool_configs = [
        {"name": "default", "mu": 0.1, "nodes": 500, "size": 50},
        {"name": "high_mixing", "mu": 0.3, "nodes": 500, "size": 30},
        {"name": "low_mixing", "mu": 0.05, "nodes": 500, "size": 30},
    ]
    
    for config in pool_configs:
        pool_dir = f"network_cache/{config['name']}_pool"
        print(f"Generating {config['name']} pool (mu={config['mu']})...")
        
        pool = NetworkPool(pool_dir)
        lfr_params = {
            "n": config["nodes"],
            "mu": config["mu"],
            "average_degree": 5,
            "min_community": 10,
            "timeout": 60
        }
        
        pool.generate_pool(config["size"], lfr_params, skip_existing=True)
        print(f"Completed: {pool_dir}")
    
    print("All default network pools generation completed")


if __name__ == "__main__":
    main()
