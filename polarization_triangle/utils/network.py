#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network utility module
Provides utility functions for network creation and processing
"""

import networkx as nx
import numpy as np
from typing import Dict, Any


def create_network(num_agents: int, network_type: str, network_params: Dict[str, Any] = None, 
                   use_network_pool: bool = True, network_pool_dir: str = None,
                   network_pool_random_selection: bool = True) -> nx.Graph:
    """
    Create a network based on the specified type and parameters.
    
    Parameters:
    num_agents -- The number of agents.
    network_type -- The type of network ("random", "lfr", "community", "ws", "ba").
    network_params -- A dictionary of network parameters.
    use_network_pool -- Whether to use a network pool (only effective for LFR networks).
    network_pool_dir -- The directory of the network pool.
    network_pool_random_selection -- Whether to randomly select networks from the pool.
    
    Returns:
    The created networkx graph object.
    """
    if network_params is None:
        network_params = {}
        
    # if network_type == 'random':
    #     p = network_params.get("p", 0.1)
    #     return nx.erdos_renyi_graph(n=num_agents, p=p)
    if network_type == 'lfr':
        # Check whether to use network pool
        if use_network_pool and network_pool_dir:
            print(f"Loading LFR network from network pool: {network_pool_dir}")
            try:
                from .network_pool import NetworkPool
                pool = NetworkPool(network_pool_dir)
                
                if network_pool_random_selection:
                    G = pool.get_random_network()
                else:
                    G = pool.load_network(index=0)  # Load first network
                
                if G is not None:
                    # Check if node count matches
                    if G.number_of_nodes() != num_agents:
                        print(f"Warning: Number of nodes in network pool ({G.number_of_nodes()}) does not match configuration ({num_agents})")
                    return G
                else:
                    print("Failed to load from network pool, falling back to real-time generation")
            except Exception as e:
                print(f"Network pool loading failed: {e}, falling back to real-time generation")
        
        # Real-time LFR network generation (original logic)
        tau1 = network_params.get("tau1", 3)
        tau2 = network_params.get("tau2", 1.5)
        mu = network_params.get("mu", 0.1)
        average_degree = network_params.get("average_degree", 5)
        min_community = network_params.get("min_community", 10)
        seed = network_params.get("seed", 42)
        # Optional fast failure with constraints
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
    # elif network_type == 'community':
    #     community_sizes = [num_agents // 4] * 4
    #     intra_p = network_params.get("intra_p", 0.8)
    #     inter_p = network_params.get("inter_p", 0.1)
    #     return nx.random_partition_graph(community_sizes, intra_p, inter_p)
    # elif network_type == 'ws':
    #     k = network_params.get("k", 4)
    #     p = network_params.get("p", 0.1)
    #     return nx.watts_strogatz_graph(n=num_agents, k=k, p=p)
    # elif network_type == 'ba':
    #     m = network_params.get("m", 2)
    #     return nx.barabasi_albert_graph(n=num_agents, m=m)
    # else:
    #     return nx.erdos_renyi_graph(n=num_agents, p=0.1)


def handle_isolated_nodes(G: nx.Graph) -> None:
    """
    Handle isolated nodes in the network.
    
    Parameters:
    G -- The network graph object.
    
    Processing method:
    1. Find all isolated nodes (nodes with degree 0).
    2. Randomly connect each isolated node to other nodes in the network.
    """
    isolated_nodes = [node for node, degree in dict(G.degree()).items() if degree == 0]

    if not isolated_nodes:
        return  # If there are no isolated nodes, return directly.

    print(f"Detected {len(isolated_nodes)} isolated nodes, processing...")

    # Get a list of non-isolated nodes.
    non_isolated = [node for node in G.nodes() if node not in isolated_nodes]

    if not non_isolated:
        # If all nodes are isolated (a rare case), create a ring connection.
        for i in range(len(isolated_nodes)):
            G.add_edge(isolated_nodes[i], isolated_nodes[(i + 1) % len(isolated_nodes)])
        return

    # Randomly connect each isolated node to 1-3 non-isolated nodes.
    for node in isolated_nodes:
        # Randomly decide the number of connections, minimum 1, maximum 3 or all non-isolated nodes.
        num_connections = min(np.random.randint(1, 4), len(non_isolated))
        # Randomly select connection targets.
        targets = np.random.choice(non_isolated, num_connections, replace=False)
        # Add edges.
        for target in targets:
            G.add_edge(node, target)
