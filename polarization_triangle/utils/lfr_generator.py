#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LFR Network Generator
Used for batch generation and saving of LFR networks, avoiding repeated generation during simulations that could cause deadlocks
"""

import networkx as nx
import pickle
import os
import time
from typing import Dict, Any, Optional
import logging

# Set up logging
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
    Generate LFR benchmark network
    
    Parameters:
        n: Number of nodes
        tau1: Power law exponent for degree distribution
        tau2: Power law exponent for community size distribution
        mu: Mixing parameter (proportion of edges connecting nodes to other communities)
        average_degree: Average degree
        min_community: Minimum community size
        max_community: Maximum community size
        seed: Random seed
        max_iters: Maximum number of iterations
        timeout: Timeout duration (seconds)
        
    Returns:
        networkx.Graph object on success, None on failure
    """
    logger.info(f"Starting LFR network generation: n={n}, mu={mu}, avg_degree={average_degree}")
    
    start_time = time.time()
    
    try:
        # Parameter validation
        if max_community is None:
            max_community = n // 2
            
        # Ensure parameters are reasonable
        if min_community > n // 2:
            min_community = n // 4
            logger.warning(f"Minimum community size too large, adjusting to {min_community}")
            
        if max_community > n:
            max_community = n // 2
            logger.warning(f"Maximum community size too large, adjusting to {max_community}")
        
        # Generate LFR network
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
        
        # Check for timeout
        if time.time() - start_time > timeout:
            logger.error(f"Network generation timeout ({timeout} seconds)")
            return None
            
        # Handle network connectivity
        if not nx.is_connected(G):
            logger.info("Network not connected, fixing connectivity...")
            _ensure_connectivity(G)
            
        logger.info(f"网络生成完成: {G.number_of_nodes()} nodes, {G.number_of_edges()}条边, "
                   f"time taken{time.time() - start_time:.2f} seconds")
        
        return G
        
    except Exception as e:
        logger.error(f"Generate LFR network error occurred: {e}")
        return None



def _ensure_connectivity(G: nx.Graph) -> None:
    """
    Ensure network connectivity
    Connect all connected components into one connected network
    """
    import random
    
    # Get all connected components
    components = list(nx.connected_components(G))
    
    if len(components) <= 1:
        return  # Network is already connected
    
    logger.info(f"Found {len(components)}  connected components, connecting...")
    
    # Find the largest connected component as main component
    largest_component = max(components, key=len)
    largest_nodes = list(largest_component)
    
    # Connect other components to main component
    for component in components:
        if component == largest_component:
            continue
            
        component_nodes = list(component)
        
        # Randomly select from current component1-2 nodes
        source_nodes = random.sample(component_nodes, min(2, len(component_nodes)))
        
        # Randomly select corresponding number of nodes from main component
        target_nodes = random.sample(largest_nodes, len(source_nodes))
        
        # Establish connections
        for source, target in zip(source_nodes, target_nodes):
            G.add_edge(source, target)
            logger.debug(f"Connect components: {source} -> {target}")
    
    # Verify connectivity
    if nx.is_connected(G):
        logger.info(f"Network connectivity repair completed, final {G.number_of_edges()} edges")
    else:
        logger.warning("Connectivity repair may have failed, network still not connected")

