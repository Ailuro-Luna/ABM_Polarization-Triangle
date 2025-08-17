import numpy as np
from typing import Dict, List, Optional, Tuple
from polarization_triangle.core.simulation import Simulation


def calculate_mean_opinion(sim: Simulation, exclude_zealots: bool = True) -> Dict[str, float]:
    """
    Calculate mean opinion statistics of the system
    
    Parameters:
    sim -- Simulation instance
    exclude_zealots -- Whether to exclude zealots, default is True
    
    Returns:
    Dictionary containing mean opinion statistics
    """
    opinions = sim.opinions.copy()
    
    # If need to exclude zealots
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
    Calculate variance metrics of the system
    
    Parameters:
    sim -- Simulation instance
    exclude_zealots -- Whether to exclude zealots, default is True
    
    Returns:
    Dictionary containing variance metrics
    """
    opinions = sim.opinions.copy()
    zealot_ids = sim.zealot_ids if hasattr(sim, 'zealot_ids') else []
    
    # Calculate overall variance (excluding zealots)
    if exclude_zealots and zealot_ids:
        non_zealot_opinions = np.delete(opinions, zealot_ids)
    else:
        non_zealot_opinions = opinions
    
    overall_variance = float(np.var(non_zealot_opinions))
    
    # Calculate variance within communities
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
    
    # Calculate variance for each community
    community_variances = []
    community_stats = {}
    
    for community_id, members in communities.items():
        # Filter out zealots
        if exclude_zealots and zealot_ids:
            community_non_zealots = [m for m in members if m not in zealot_ids]
        else:
            community_non_zealots = members
            
        if len(community_non_zealots) > 1:  # Need at least 2 agents to calculate variance
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
    
    # Average intra-community variance
    mean_intra_community_variance = float(np.mean(community_variances)) if community_variances else 0.0
    
    return {
        "overall_variance": overall_variance,
        "mean_intra_community_variance": mean_intra_community_variance,
        "num_communities": len(communities),
        "community_details": community_stats
    }


def calculate_identity_statistics(sim: Simulation, exclude_zealots: bool = True) -> Dict[str, float]:
    """
    Calculate statistical metrics grouped by identity
    
    Parameters:
    sim -- Simulation instance
    exclude_zealots -- Whether to exclude zealots, default is True
    
    Returns:
    Dictionary containing identity statistics
    """
    opinions = sim.opinions.copy()
    identities = sim.identities.copy()
    zealot_ids = sim.zealot_ids if hasattr(sim, 'zealot_ids') else []
    
    # Find agents of different identities (excluding zealots)
    unique_identities = np.unique(identities)
    identity_stats = {}
    
    for identity_val in unique_identities:
        # Find agents with this identity
        identity_agents = np.where(identities == identity_val)[0]
        
        # Exclude zealots
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
    
    # Calculate opinion differences between identities
    identity_values = [key for key in identity_stats.keys()]
    if len(identity_values) >= 2:
        # If there are identity +1 and -1, calculate their difference
        if "identity_1" in identity_stats and "identity_-1" in identity_stats:
            mean_diff = identity_stats["identity_1"]["mean_opinion"] - identity_stats["identity_-1"]["mean_opinion"]
            identity_stats["identity_difference"] = {
                "mean_opinion_difference": float(mean_diff),
                "abs_mean_opinion_difference": float(abs(mean_diff))
            }
    
    return identity_stats


def get_polarization_index(sim: Simulation) -> float:
    """
    Get polarization index of current system
    
    Parameters:
    sim -- Simulation instance
    
    Returns:
    Polarization index value
    """
    if hasattr(sim, 'calculate_polarization_index'):
        return float(sim.calculate_polarization_index())
    else:
        # If simulation doesn't have this method, we implement it ourselves
        opinions = sim.opinions
        
        # Discretize opinions into 5 categories
        category_counts = np.zeros(5, dtype=np.int32)
        
        for opinion in opinions:
            if opinion < -0.6:
                category_counts[0] += 1  # Category 1: Strongly disagree
            elif opinion < -0.2:
                category_counts[1] += 1  # Category 2: Disagree
            elif opinion <= 0.2:
                category_counts[2] += 1  # Category 3: Neutral
            elif opinion <= 0.6:
                category_counts[3] += 1  # Category 4: Agree
            else:
                category_counts[4] += 1  # Category 5: Strongly agree
        
        # Get agent count for each category
        n1, n2, n3, n4, n5 = category_counts
        N = sim.num_agents
        
        # Apply Koudenburg formula to calculate polarization index
        numerator = (2.14 * n2 * n4 + 
                    2.70 * (n1 * n4 + n2 * n5) + 
                    3.96 * n1 * n5)
        
        denominator = 0.0099 * (N ** 2)
        
        if denominator > 0:
            return float(numerator / denominator)
        else:
            return 0.0
