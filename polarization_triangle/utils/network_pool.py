#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network pool manager
Used for batch generation, saving and loading of LFR network pools, avoiding repeated network generation during simulations
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkPool:
    """
    Network pool manager
    Responsible for batch generation, saving, and loading of LFR networks.
    """
    
    def __init__(self, pool_dir: str):
        """
        Initialize the Network Pool Manager.
        
        Parameters:
            pool_dir: The directory to store the network pool.
        """
        self.pool_dir = Path(pool_dir)
        self.metadata_file = self.pool_dir / "pool_metadata.json"
        self.networks_dir = self.pool_dir / "networks"
        
        # Ensure directories exist
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.networks_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load network pool metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                return self._create_empty_metadata()
        else:
            return self._create_empty_metadata()
    
    def _create_empty_metadata(self) -> Dict:
        """Create an empty metadata structure."""
        return {
            "pool_info": {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_networks": 0,
                "lfr_params": {}
            },
            "networks": {}
        }
    
    def _save_metadata(self):
        """Save metadata to a file."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def generate_pool(
        self, 
        pool_size: int,
        lfr_params: Optional[Dict] = None,
        start_seed: int = 42,
        skip_existing: bool = True
    ) -> bool:
        """
        Generate a batch of networks for the pool.
        
        Parameters:
            pool_size: The number of networks to generate.
            lfr_params: A dictionary of LFR parameters. If None, default parameters are used.
            start_seed: The starting random seed.
            skip_existing: Whether to skip generating networks that already exist.
            
        Returns:
            True if successful, False otherwise.
        """
        # Use default LFR parameters if none are provided
        if lfr_params is None:
            lfr_params = {
                "n": 500,
                "tau1": 3.0,
                "tau2": 1.5,
                "mu": 0.1,
                "average_degree": 5,
                "min_community": 10,
                "max_community": 50,
                "timeout": 60
            }
        
        logger.info(f"Starting to generate network pool: {pool_size} networks")
        logger.info(f"LFR parameters: {lfr_params}")
        
        # Update parameter info in metadata
        self.metadata["pool_info"]["lfr_params"] = lfr_params
        
        success_count = 0
        total_start_time = time.time()
        
        for i in range(pool_size):
            network_id = f"network_{i:04d}"
            network_file = self.networks_dir / f"{network_id}.pkl"
            
            # Check if existing networks should be skipped
            if skip_existing and network_file.exists():
                logger.info(f"Skipping existing network: {network_id}")
                success_count += 1
                continue
            
            # Generate network
            seed = start_seed + i
            logger.info(f"Generating network {i+1}/{pool_size} (seed={seed})")
            
            G = generate_lfr_network(seed=seed, **lfr_params)
            
            if G is not None:
                # Save the network
                try:
                    with open(network_file, 'wb') as f:
                        pickle.dump(G, f)
                    
                    # Update metadata
                    self.metadata["networks"][network_id] = {
                        "file": f"networks/{network_id}.pkl",
                        "seed": seed,
                        "nodes": G.number_of_nodes(),
                        "edges": G.number_of_edges(),
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "lfr_params": lfr_params.copy()
                    }
                    
                    success_count += 1
                    logger.info(f"Network {network_id} generated and saved successfully")
                    
                except Exception as e:
                    logger.error(f"Failed to save network {network_id}: {e}")
            else:
                logger.error(f"Failed to generate network {network_id}")
        
        # Update overall metadata
        self.metadata["pool_info"]["total_networks"] = success_count
        self.metadata["pool_info"]["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Save metadata
        self._save_metadata()
        
        total_time = time.time() - total_start_time
        logger.info(f"Network pool generation complete: {success_count}/{pool_size} successful, total time {total_time:.1f} seconds")
        
        return success_count > 0
    
    def load_network(self, network_id: Optional[str] = None, index: Optional[int] = None) -> Optional[nx.Graph]:
        """
        Load a network from the pool.
        
        Parameters:
            network_id: The ID of the network (e.g., "network_0001").
            index: The index of the network (starting from 0).
            
        Returns:
            A networkx.Graph object on success, None on failure.
        """
        # Determine network_id from index if provided
        if index is not None:
            network_id = f"network_{index:04d}"
        
        if network_id is None:
            logger.error("Either network_id or index must be provided.")
            return None
        
        # Check if the network exists in metadata
        if network_id not in self.metadata["networks"]:
            logger.error(f"Network {network_id} does not exist in the pool.")
            return None
        
        # Get the network file path
        network_file = self.pool_dir / self.metadata["networks"][network_id]["file"]
        
        if not network_file.exists():
            logger.error(f"Network file does not exist: {network_file}")
            return None
        
        # Load the network
        try:
            with open(network_file, 'rb') as f:
                G = pickle.load(f)
            logger.debug(f"Successfully loaded network {network_id}: {G.number_of_nodes()} nodes")
            return G
        except Exception as e:
            logger.error(f"Failed to load network {network_id}: {e}")
            return None
    
    def get_random_network(self) -> Optional[nx.Graph]:
        """
        Get a random network from the pool.
        
        Returns:
            A random networkx.Graph object, or None if the pool is empty.
        """
        if not self.metadata["networks"]:
            logger.error("Network pool is empty.")
            return None
        
        import random
        network_id = random.choice(list(self.metadata["networks"].keys()))
        return self.load_network(network_id)
    
    def get_pool_info(self) -> Dict:
        """
        Get information about the network pool.
        
        Returns:
            A dictionary containing pool information.
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
        List information for all networks in the pool.
        
        Returns:
            A list of network information dictionaries.
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
    Create a network pool with default parameters.
    
    Parameters:
        pool_dir: The directory to store the pool.
        pool_size: The size of the pool.
        
    Returns:
        A NetworkPool object.
    """
    pool = NetworkPool(pool_dir)
    
    # Use default parameters consistent with config.py
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
    # Simple test
    print("Testing network pool system...")
    
    test_pool_dir = "./test_network_pool"
    pool = NetworkPool(test_pool_dir)
    
    # Generate a small test pool
    success = pool.generate_pool(pool_size=3, lfr_params={"n": 50, "mu": 0.1})
    
    if success:
        # Test loading
        print("Pool info:")
        info = pool.get_pool_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test network loading
        G = pool.load_network(index=0)
        if G:
            print(f"Successfully loaded network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Clean up test files
        import shutil
        if os.path.exists(test_pool_dir):
            shutil.rmtree(test_pool_dir)
            print("Cleaned up test files.")
    else:
        print("Network pool generation failed.")
