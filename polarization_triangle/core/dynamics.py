from numba import njit, int32, float64, prange, boolean
import numpy as np

def sample_morality(morality_rate):
    """
    Randomly generate a moral value (0 or 1) based on moralization rate

    Parameters:
    morality_rate -- Float between 0 and 1, representing the probability of moralization

    Returns:
    1 (moralized) or 0 (non-moralized)
    """
    return 1 if np.random.rand() < morality_rate else 0


@njit
def calculate_perceived_opinion_func(opinions, morals, i, j):
    """
    Calculate agent i's perception of agent j's opinion
    
    Parameters:
    opinions -- Array of all agents' opinions
    morals -- Array of all agents' moral values
    i -- Observer agent's index
    j -- Observed agent's index
    
    Returns:
    Perceived opinion value
    """
    z_j = opinions[j]
    m_i = morals[i]
    m_j = morals[j]
    
    if z_j == 0:
        return 0
    elif (m_i == 1 or m_j == 1):
        return np.sign(z_j)  # Return sign of z_j (1 or -1)
    else:
        return z_j  # Return actual value


@njit
def calculate_same_identity_sigma_func(opinions, morals, identities, neighbors_indices, neighbors_indptr, i):
    """
    Calculate average perceived opinion value of agent i's same-identity neighbors (numba accelerated version)
    
    Parameters:
    opinions -- Array of all agents' opinions
    morals -- Array of all agents' moral values
    identities -- Array of all agents' identities
    neighbors_indices -- Neighbor indices in CSR format
    neighbors_indptr -- Neighbor pointers in CSR format
    i -- Agent i's index
    
    Returns:
    Average perceived opinion value of same-identity neighbors, returns 0 if no same-identity neighbors
    """
    sigma_sum = 0.0
    count = 0
    l_i = identities[i]
    
    # Iterate through all neighbors of i
    for idx in range(neighbors_indptr[i], neighbors_indptr[i+1]):
        j = neighbors_indices[idx]
        # If same identity
        if identities[j] == l_i:
            sigma_sum += calculate_perceived_opinion_func(opinions, morals, i, j)
            count += 1
    
    # Return average value, return 0 if no same-identity neighbors
    if count > 0:
        return sigma_sum / count
    return 0.0


@njit
def calculate_relationship_coefficient_func(adj_matrix, identities, morals, opinions, i, j, same_identity_sigmas, cohesion_factor):
    """
    Calculate relationship coefficient between agent i and agent j
    
    Parameters:
    adj_matrix -- Adjacency matrix
    identities -- Identity array
    morals -- Moral values array
    opinions -- Opinions array
    i -- Agent i's index
    j -- Agent j's index
    same_identity_sigmas -- Perceived opinion values array or average of agent i's same-identity neighbors
    
    Returns:
    Relationship coefficient value
    """
    a_ij = adj_matrix[i, j]
    if a_ij == 0:  # If not neighbors, relationship coefficient is 0
        return 0
        
    l_i = identities[i]
    l_j = identities[j]
    m_i = morals[i]
    m_j = morals[j]

    if l_i == l_j:
        cohesion_factor = cohesion_factor
    else:
        cohesion_factor = 0
    
    # Calculate perceived opinions
    sigma_ij = calculate_perceived_opinion_func(opinions, morals, i, j)
    sigma_ji = calculate_perceived_opinion_func(opinions, morals, j, i)
    
    # Calculate relationship coefficient based on polarization triangle framework formula
    if l_i != l_j and m_i == 1 and m_j == 1 and (sigma_ij * sigma_ji) < 0:
        return -a_ij+cohesion_factor
    elif l_i == l_j and m_i == 1 and m_j == 1 and (sigma_ij * sigma_ji) < 0:
        # Use passed same-identity average perceived opinion value
        if sigma_ij == 0:  # Avoid division by zero error
            return a_ij+cohesion_factor
        return ((a_ij / sigma_ij) * same_identity_sigmas)
    else:
        return a_ij+cohesion_factor


@njit
def step_calculation(opinions, morals, identities, adj_matrix, 
                    neighbors_indices, neighbors_indptr,  
                    alpha, beta, gamma, delta, u, influence_factor, cohesion_factor):
    """
    Execute one simulation step calculation, using numba acceleration
    
    Parameters:
    opinions -- Agent opinion array
    morals -- Agent moral values array
    identities -- Agent identity array
    adj_matrix -- Adjacency matrix
    neighbors_indices -- Neighbor index array in CSR format
    neighbors_indptr -- Neighbor pointer array in CSR format
    alpha -- Self-activation coefficient
    beta -- Social influence coefficient
    gamma -- Moralization influence coefficient
    delta -- Opinion decay rate
    u -- Opinion activation coefficient
    influence_factor -- Influence factor
    cohesion_factor -- Identity cohesion factor
    
    Returns:
    Updated opinions, self_activation, social_influence, interaction_counts
    interaction_counts -- Counts of 16 interaction types, arranged in rule order
    """
    num_agents = len(opinions)
    opinion_changes = np.zeros(num_agents, dtype=np.float64)
    self_activation = np.zeros(num_agents, dtype=np.float64)
    social_influence = np.zeros(num_agents, dtype=np.float64)
    
    # Create counters for 16 interaction types
    interaction_counts = np.zeros(16, dtype=np.int32)
    
    # Pre-calculate average perceived opinion of same-identity neighbors for all agents
    same_identity_sigmas = np.zeros(num_agents, dtype=np.float64)
    for i in range(num_agents):
        same_identity_sigmas[i] = calculate_same_identity_sigma_func(
            opinions, morals, identities, neighbors_indices, neighbors_indptr, i)
    
    for i in range(num_agents):
        # Calculate self-perception
        sigma_ii = np.sign(opinions[i]) if opinions[i] != 0 else 0
        
        # Calculate total neighbor influence
        neighbor_influence = 0.0
        
        # Iterate through all neighbors of i (using CSR format)
        for idx in range(neighbors_indptr[i], neighbors_indptr[i+1]):
            j = neighbors_indices[idx]
            
            # Count interaction types
            # Determine if opinion directions are the same
            same_opinion_direction = (opinions[i] * opinions[j] >= 0)
            
            # Determine if identities are the same
            same_identity = (identities[i] == identities[j])
            
            # Get moral states
            m_i = morals[i]
            m_j = morals[j]
            
            # Calculate rule index (0-15)
            # Rule index composition:
            # 0-3: Same opinion direction, same identity (00, 01, 10, 11)
            # 4-7: Same opinion direction, different identity (00, 01, 10, 11)
            # 8-11: Different opinion direction, same identity (00, 01, 10, 11)
            # 12-15: Different opinion direction, different identity (00, 01, 10, 11)
            rule_index = 0
            
            # First set opinion direction and identity bits
            if not same_opinion_direction:
                rule_index += 8
            if not same_identity:
                rule_index += 4
                
            # Then set moral state bits
            rule_index += m_i * 2 + m_j
            
            # Increment corresponding rule count
            interaction_counts[rule_index] += 1
            
            # Continue original calculation logic
            A_ij = calculate_relationship_coefficient_func(
                adj_matrix, 
                identities, 
                morals, 
                opinions, 
                i, j, 
                same_identity_sigmas[i],
                cohesion_factor
            )
            sigma_ij = calculate_perceived_opinion_func(opinions, morals, i, j)
            neighbor_influence += A_ij * sigma_ij
        
        # Calculate and store self-activation term
        # self_activation[i] = alpha[i] * sigma_ii
        self_activation[i] = alpha[i] * opinions[i]
        
        # Calculate and store social influence term
        social_influence[i] = (beta / (1 + gamma[i] * morals[i])) * neighbor_influence
        
        # Calculate opinion change rate
        # Return to neutral opinion term
        regression_term = -delta * opinions[i]
        
        # Opinion activation term
        activation_term = u[i] * np.tanh(
            self_activation[i] + social_influence[i]
        )
        
        # Total change
        opinion_changes[i] = regression_term + activation_term
    
    # Apply opinion changes, use small step size to avoid excessive changes
    opinions_new = opinions.copy()
    opinions_new += influence_factor * opinion_changes
    
    # Ensure opinion values are within [-1, 1] range
    opinions_new = np.clip(opinions_new, -1, 1)
    
    return opinions_new, self_activation, social_influence, interaction_counts