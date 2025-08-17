# final_optimized_simulation.py
import numpy as np
import networkx as nx
from polarization_triangle.core.config import SimulationConfig
from polarization_triangle.core.dynamics import *
from polarization_triangle.utils.network import create_network, handle_isolated_nodes
from typing import Dict, List

class Simulation:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.num_agents = config.num_agents
        self.network_type = config.network_type

        # Create network
        self.graph = create_network(
            num_agents=self.num_agents,
            network_type=config.network_type,
            network_params=config.network_params,
            use_network_pool=config.use_network_pool,
            network_pool_dir=config.network_pool_dir,
            network_pool_random_selection=config.network_pool_random_selection
        )
        # Remove self-loops
        self.graph.remove_edges_from(nx.selfloop_edges(self.graph))
        # Handle isolated nodes
        handle_isolated_nodes(self.graph)
        # Get adjacency matrix
        self.adj_matrix = nx.adjacency_matrix(self.graph).toarray()

        self.pos = nx.spring_layout(self.graph, k=0.1, iterations=50, scale=2.0)

        # Initialize cluster dominant attributes
        self.cluster_identity_majority = {}
        self.cluster_morality_majority = {}
        self.cluster_opinion_majority = {} if config.cluster_opinion else None

        # Initialize identity-issue association mapping
        self.identity_issue_mapping = config.identity_issue_mapping
        self.identity_influence_factor = config.identity_influence_factor

        # Initialize rule counter history
        self.rule_counts_history = []

        # Initialize zealot-related attributes
        self.zealot_ids = []
        self.zealot_opinion = config.zealot_opinion
        self.enable_zealots = config.enable_zealots or config.zealot_count > 0

        if self.network_type in ['community', 'lfr']:
            for node in self.graph.nodes():
                block = None
                if self.network_type == 'community':
                    block = self.graph.nodes[node].get("block")
                elif self.network_type == 'lfr':
                    block = self.graph.nodes[node].get("community")
                    if isinstance(block, (set, frozenset)):
                        block = min(block)
                if config.cluster_identity and block not in self.cluster_identity_majority:
                    self.cluster_identity_majority[block] = 1 if np.random.rand() < 0.5 else -1
                if config.cluster_morality and block not in self.cluster_morality_majority:
                    self.cluster_morality_majority[block] = sample_morality(config.morality_rate)
                if config.cluster_opinion:
                    if block not in self.cluster_opinion_majority:
                        if config.opinion_distribution == "twin_peak":
                            majority_opinion = np.random.choice([-1, 1]) * np.abs(np.random.normal(0.7, 0.2))
                        elif config.opinion_distribution == "uniform":
                            majority_opinion = np.random.uniform(-1, 1)
                        elif config.opinion_distribution == "single_peak":
                            majority_opinion = np.random.normal(0, 0.3)
                        elif config.opinion_distribution == "skewed":
                            majority_opinion = np.random.beta(2, 5) * 2 - 1
                        else:
                            majority_opinion = np.random.uniform(-1, 1)
                        self.cluster_opinion_majority[block] = majority_opinion

        # Initialize agent attributes
        self.opinions = np.empty(self.num_agents, dtype=np.float64)
        self.morals = np.empty(self.num_agents, dtype=np.int32)
        self.identities = np.empty(self.num_agents, dtype=np.int32)

        # Get model parameters from configuration
        self.delta = config.delta  # Opinion decay rate
        self.u = np.ones(self.num_agents) * config.u  # Opinion activation coefficient
        self.alpha = np.ones(self.num_agents) * config.alpha  # Self-activation coefficient
        self.beta = config.beta  # Social influence coefficient
        self.gamma = np.ones(self.num_agents) * config.gamma  # Moralization influence coefficient
        
        self._init_identities()
        self._init_morality()
        self._init_opinions()
        
        # Initialize zealots (must be after other attributes initialization)
        if self.enable_zealots:
            self._init_zealots()
        
        # Store neighbor list for each agent
        self.neighbors_list = [[] for _ in range(self.num_agents)]
        
        # Initialize neighbor lists
        self._init_neighbors_lists()
        
        # Initialize arrays for monitoring self-activation and social influence
        self.self_activation = np.zeros(self.num_agents, dtype=np.float64)
        self.social_influence = np.zeros(self.num_agents, dtype=np.float64)
        
        # Store historical data
        self.self_activation_history = []
        self.social_influence_history = []
        
        # Add trajectory storage
        self.opinion_trajectory = []
        
        # Initialize polarization index history
        self.polarization_history = []
        
        # Optimized data structure (CSR format)
        self._create_csr_neighbors()

    def _init_zealots(self):
        """
        Initialize zealots according to configuration
        """
        zealot_count = self.config.zealot_count
        zealot_mode = self.config.zealot_mode
        
        if zealot_count <= 0:
            return
        
        # Determine candidate agent pool
        if self.config.zealot_identity_allocation:
            # Select only from agents with identity=1
            candidate_agents = [i for i in range(self.num_agents) if self.identities[i] == 1]
            if len(candidate_agents) == 0:
                print("Warning: No agents with identity=1 found for zealot allocation")
                return
            if zealot_count > len(candidate_agents):
                zealot_count = len(candidate_agents)
                print(f"Warning: zealot_count exceeds agents with identity=1, setting to {len(candidate_agents)}")
        else:
            # Select from all agents
            candidate_agents = list(range(self.num_agents))
            if zealot_count > self.num_agents:
                zealot_count = self.num_agents
                print(f"Warning: zealot_count exceeds agent count, setting to {self.num_agents}")
        
        # Select zealots according to mode
        if zealot_mode == "random":
            # Randomly select specified number of agents from candidate pool as zealots
            self.zealot_ids = np.random.choice(candidate_agents, size=zealot_count, replace=False).tolist()
        
        elif zealot_mode == "degree":
            # Select nodes with highest degree from candidate pool as zealots
            node_degrees = [(node, self.graph.degree(node)) for node in candidate_agents]
            sorted_nodes_by_degree = sorted(node_degrees, key=lambda x: x[1], reverse=True)
            self.zealot_ids = [node for node, degree in sorted_nodes_by_degree[:zealot_count]]
        
        elif zealot_mode == "clustered":
            # Get community information for candidate agents
            communities = {}
            for node in candidate_agents:
                community = self.graph.nodes[node].get("community")
                if isinstance(community, (set, frozenset)):
                    community = min(community)
                if community not in communities:
                    communities[community] = []
                communities[community].append(node)
            
            # Sort by community size
            sorted_communities = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
            
            # Try to select zealots within the same community
            zealots_left = zealot_count
            self.zealot_ids = []
            for community_id, members in sorted_communities:
                if zealots_left <= 0:
                    break
                
                # Decide how many zealots to select from current community
                to_select = min(zealots_left, len(members))
                selected = np.random.choice(members, size=to_select, replace=False).tolist()
                self.zealot_ids.extend(selected)
                zealots_left -= to_select
        
        else:
            raise ValueError(f"Unknown zealot mode: {zealot_mode}")
        
        # Set initial opinions for zealots
        for agent_id in self.zealot_ids:
            self.opinions[agent_id] = self.zealot_opinion
        
        # If configuration requires, set zealots to moralizing
        if self.config.zealot_morality:
            for agent_id in self.zealot_ids:
                self.morals[agent_id] = 1

    def set_zealot_opinions(self):
        """
        Reset zealot opinions to fixed values
        """
        if self.enable_zealots and self.zealot_ids:
            for agent_id in self.zealot_ids:
                self.opinions[agent_id] = self.zealot_opinion

    def get_zealot_ids(self) -> List[int]:
        """
        Get list of zealot IDs
        
        Returns:
        List of zealot IDs
        """
        return self.zealot_ids.copy()

    def _create_csr_neighbors(self):
        """
        Create CSR format neighbor representation
        """
        total_edges = sum(len(neighbors) for neighbors in self.neighbors_list)
        self.neighbors_indices = np.zeros(total_edges, dtype=np.int32)
        self.neighbors_indptr = np.zeros(self.num_agents + 1, dtype=np.int32)
        
        idx = 0
        for i, neighbors in enumerate(self.neighbors_list):
            self.neighbors_indptr[i] = idx
            for j in neighbors:
                self.neighbors_indices[idx] = j
                idx += 1
        self.neighbors_indptr[self.num_agents] = idx

    def _init_identities(self):
        for i in range(self.num_agents):
            if self.config.cluster_identity and self.network_type in ['community', 'lfr']:
                block = None
                if self.network_type == 'community':
                    block = self.graph.nodes[i].get("block")
                elif self.network_type == 'lfr':
                    block = self.graph.nodes[i].get("community")
                    if isinstance(block, (set, frozenset)):
                        block = min(block)
                majority = self.cluster_identity_majority.get(block, 1 if np.random.rand() < 0.5 else -1)
                prob = self.config.cluster_identity_prob
                self.identities[i] = majority if np.random.rand() < prob else -majority
            else:
                self.identities[i] = 1 if np.random.rand() < 0.5 else -1

    def _init_morality(self):
        for i in range(self.num_agents):
            if self.config.cluster_morality and self.network_type in ['community', 'lfr']:
                block = None
                if self.network_type == 'community':
                    block = self.graph.nodes[i].get("block")
                elif self.network_type == 'lfr':
                    block = self.graph.nodes[i].get("community")
                    if isinstance(block, (set, frozenset)):
                        block = min(block)
                majority = self.cluster_morality_majority.get(block, sample_morality(self.config.morality_rate))
                prob = self.config.cluster_morality_prob
                self.morals[i] = majority if np.random.rand() < prob else sample_morality(self.config.morality_rate)
            else:
                self.morals[i] = sample_morality(self.config.morality_rate)

    def _init_opinions(self):
        for i in range(self.num_agents):
            if self.config.cluster_opinion and self.network_type in ['community', 'lfr']:
                block = None
                if self.network_type == 'community':
                    block = self.graph.nodes[i].get("block")
                elif self.network_type == 'lfr':
                    block = self.graph.nodes[i].get("community")
                    if isinstance(block, (set, frozenset)):
                        block = min(block)
                majority = self.cluster_opinion_majority.get(block)
                if majority is None:
                    if self.config.opinion_distribution == "twin_peak":
                        majority = np.random.choice([-1, 1]) * np.abs(np.random.normal(0.7, 0.2))
                    elif self.config.opinion_distribution == "uniform":
                        majority = np.random.uniform(-1, 1)
                    elif self.config.opinion_distribution == "single_peak":
                        majority = np.random.normal(0, 0.3)
                    elif self.config.opinion_distribution == "skewed":
                        majority = np.random.beta(2, 5) * 2 - 1
                    else:
                        majority = np.random.uniform(-1, 1)
                    self.cluster_opinion_majority[block] = majority
                prob = self.config.cluster_opinion_prob
                self.opinions[i] = majority if np.random.rand() < prob else self.generate_opinion(self.identities[i])
            else:
                self.opinions[i] = self.generate_opinion(self.identities[i])

        # # Apply identity-topic association offset
        # for i in range(self.num_agents):
        #     identity = self.identities[i]
        #     association = self.identity_issue_mapping.get(identity, 0)
        #     # Add offset based on identity-topic association
        #     random_factor = np.random.uniform(0.5, 1.0)
        #     shift = association * random_factor
        #     self.opinions[i] = np.clip(self.opinions[i] + shift, -1, 1)

    def generate_opinion(self, identity):
        if self.config.extreme_fraction > 0 and np.random.rand() < self.config.extreme_fraction:
            if self.config.coupling != "none":
                return np.random.uniform(-1, -0.5) if identity == 1 else np.random.uniform(0.5, 1)
            else:
                return np.random.choice([-1, 1]) * np.random.uniform(0.5, 1)
        if self.config.opinion_distribution == "uniform":
            return np.random.uniform(-1, 1)
        elif self.config.opinion_distribution == "single_peak":
            return np.random.normal(0, 0.3)
        elif self.config.opinion_distribution == "twin_peak":
            return np.random.choice([-1, 1]) * np.abs(np.random.normal(0.7, 0.2))
        elif self.config.opinion_distribution == "skewed":
            return np.random.beta(2, 5) * 2 - 1
        else:
            return np.random.uniform(-1, 1)

    def _init_neighbors_lists(self):
        """Initialize neighbor list for each agent"""
        for i in range(self.num_agents):
            # Get neighbor list
            for j in range(self.num_agents):
                if i != j and self.adj_matrix[i, j] > 0:
                    self.neighbors_list[i].append(j)

    # Perceived opinion calculation based on polarization triangle framework
    def calculate_perceived_opinion(self, i, j):
        """
        Calculate agent i's perception of agent j's opinion
        
        Parameters:
        i -- Observer agent's index
        j -- Observed agent's index
        
        Returns:
        Perceived opinion value, taking values -1, 0, or 1
        """
        return calculate_perceived_opinion_func(self.opinions, self.morals, i, j)

    # Calculate relationship coefficient between agents
    def calculate_relationship_coefficient(self, i, j):
        """
        Calculate relationship coefficient between agent i and agent j
        
        Parameters:
        i -- Agent i's index
        j -- Agent j's index
        
        Returns:
        Relationship coefficient value
        """
        # Calculate average perceived opinion value of same-identity neighbors
        sigma_same_identity = calculate_same_identity_sigma_func(
            self.opinions, self.morals, self.identities, 
            self.neighbors_indices, self.neighbors_indptr, i)
        
        return calculate_relationship_coefficient_func(
            self.adj_matrix, 
            self.identities, 
            self.morals, 
            self.opinions, 
            i, j, 
            sigma_same_identity,
            self.config.cohesion_factor
        )
    
    def calculate_polarization_index(self):
        """
        Calculate Koudenburg opinion polarization index
        
        Returns:
        Polarization index value (0-100)
        """
        # 1. Discretize opinions into 5 categories
        category_counts = np.zeros(5, dtype=np.int32)
        
        for opinion in self.opinions:
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
        
        # 2. Get agent count for each category
        n1, n2, n3, n4, n5 = category_counts
        N = self.num_agents
        
        # 3. Apply Koudenburg formula to calculate polarization index
        # Numerator: Calculate weighted sum of opinion pairs crossing neutral point
        numerator = (2.14 * n2 * n4 + 
                    2.70 * (n1 * n4 + n2 * n5) + 
                    3.96 * n1 * n5)
        
        # Denominator: Normalization factor
        denominator = 0.0099 * (N ** 2)
        
        # Calculate polarization index
        if denominator > 0:
            polarization_index = numerator / denominator
        else:
            polarization_index = 0.0
            
        return polarization_index

    # Optimized step method based on polarization triangle framework
    def step(self):
        """
        Execute one simulation step, update all agents' opinions
        Implementation based on polarization triangle framework dynamics equations
        Uses numba-accelerated step_calculation function
        """
        # Initialize rule counts (for compatibility with original code, although this method no longer uses rules)
        rule_counts = np.zeros(16, dtype=np.int32)
        
        # Record current opinions to trajectory
        self.opinion_trajectory.append(self.opinions.copy())
        
        # Use numba-accelerated function for main calculation
        new_opinions, new_self_activation, new_social_influence, rule_counts = step_calculation(
            self.opinions,
            self.morals,
            self.identities,
            self.adj_matrix,
            self.neighbors_indices,
            self.neighbors_indptr,
            self.alpha,
            self.beta,
            self.gamma,
            self.delta,
            self.u,
            self.config.influence_factor,
            self.config.cohesion_factor
        )
        
        # Update states
        self.opinions = new_opinions
        self.self_activation = new_self_activation
        self.social_influence = new_social_influence
        
        # Reset zealot opinions (must be after opinion update)
        self.set_zealot_opinions()
        
        # Store rule counts for compatibility with original code
        self.rule_counts_history.append(rule_counts)
        
        # Store historical data of self-activation and social influence
        self.self_activation_history.append(self.self_activation.copy())
        self.social_influence_history.append(self.social_influence.copy())
        
        # Calculate and store polarization index
        polarization = self.calculate_polarization_index()
        self.polarization_history.append(polarization)
    
    def get_activation_components(self):
        """
        Get self-activation and social influence components from the latest step
        
        Returns:
        Dictionary containing arrays of self-activation and social influence
        """
        return {
            "self_activation": self.self_activation,
            "social_influence": self.social_influence
        }
    
    def get_activation_history(self):
        """
        Get self-activation and social influence components for all historical steps
        
        Returns:
        Dictionary containing lists of historical data for self-activation and social influence
        """
        return {
            "self_activation_history": self.self_activation_history,
            "social_influence_history": self.social_influence_history
        }
    
    def get_polarization_history(self):
        """
        Get polarization index for all historical steps
        
        Returns:
        List of polarization index history
        """
        return self.polarization_history

    def save_simulation_data(self, output_dir: str, prefix: str = 'sim_data') -> Dict[str, str]:
        """
        Save simulation data to files for subsequent statistical analysis
        
        Parameters:
        output_dir -- Output directory path
        prefix -- File name prefix
        
        Returns:
        Dictionary containing all saved file paths
        """

        from polarization_triangle.utils.data_manager import save_simulation_data
        return save_simulation_data(self, output_dir, prefix) 
    
    def get_interaction_counts(self):
        """
        Get statistical information on interaction types, including interaction descriptions
        
        Returns:
        Dictionary containing interaction type counts and descriptions
        """
        # Total count array
        total_counts = np.sum(self.rule_counts_history, axis=0) if len(self.rule_counts_history) > 0 else np.zeros(16)
        
        # Interaction type descriptions
        interaction_descriptions = [
            # Same opinion direction, same identity
            "Rule 1: Same dir, Same ID, {0,0}, High Convergence",
            "Rule 2: Same dir, Same ID, {0,1}, Medium Pull",
            "Rule 3: Same dir, Same ID, {1,0}, Medium Pull",
            "Rule 4: Same dir, Same ID, {1,1}, High Polarization",
            # Same opinion direction, different identity
            "Rule 5: Same dir, Diff ID, {0,0}, Medium Convergence",
            "Rule 6: Same dir, Diff ID, {0,1}, Low Pull",
            "Rule 7: Same dir, Diff ID, {1,0}, Low Pull",
            "Rule 8: Same dir, Diff ID, {1,1}, Medium Polarization",
            # Different opinion direction, same identity
            "Rule 9: Diff dir, Same ID, {0,0}, Very High Convergence",
            "Rule 10: Diff dir, Same ID, {0,1}, Medium Convergence/Pull",
            "Rule 11: Diff dir, Same ID, {1,0}, Low Resistance",
            "Rule 12: Diff dir, Same ID, {1,1}, Low Polarization",
            # Different opinion direction, different identity
            "Rule 13: Diff dir, Diff ID, {0,0}, Low Convergence",
            "Rule 14: Diff dir, Diff ID, {0,1}, High Pull",
            "Rule 15: Diff dir, Diff ID, {1,0}, High Resistance",
            "Rule 16: Diff dir, Diff ID, {1,1}, Very High Polarization"
        ]
        
        # Calculate total count
        total_interactions = np.sum(total_counts)
        
        # Build result dictionary
        result = {
            "total_interactions": total_interactions,
            "counts": total_counts,
            "descriptions": interaction_descriptions,
            "percentages": (total_counts / total_interactions * 100) if total_interactions > 0 else np.zeros(16)
        }
        
        return result 