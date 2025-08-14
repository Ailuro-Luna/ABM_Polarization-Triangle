from numba import njit, int32, float64, prange, boolean
import numpy as np

def sample_morality(morality_rate):
    """
    根据道德化率随机生成一个道德值（0或1）

    参数:
    morality_rate -- 介于0和1之间的浮点数，表示道德化的概率

    返回:
    1（道德化）或0（非道德化）
    """
    return 1 if np.random.rand() < morality_rate else 0


@njit
def calculate_perceived_opinion_func(opinions, morals, i, j):
    """
    计算agent i对agent j的意见的感知
    
    参数:
    opinions -- 所有agent的意见数组
    morals -- 所有agent的道德值数组
    i -- 观察者agent的索引
    j -- 被观察agent的索引
    
    返回:
    感知意见值
    """
    z_j = opinions[j]
    m_i = morals[i]
    m_j = morals[j]
    
    if z_j == 0:
        return 0
    elif (m_i == 1 or m_j == 1):
        return np.sign(z_j)  # 返回z_j的符号(1或-1)
    else:
        return z_j  # 返回实际值


@njit
def calculate_same_identity_sigma_func(opinions, morals, identities, neighbors_indices, neighbors_indptr, i):
    """
    计算agent i的同身份邻居的平均感知意见值（numba加速版本）
    
    参数:
    opinions -- 所有agent的意见数组
    morals -- 所有agent的道德值数组
    identities -- 所有agent的身份数组
    neighbors_indices -- CSR格式的邻居索引
    neighbors_indptr -- CSR格式的邻居指针
    i -- agent i的索引
    
    返回:
    同身份邻居的平均感知意见值，如果没有同身份邻居则返回0
    """
    sigma_sum = 0.0
    count = 0
    l_i = identities[i]
    
    # 遍历i的所有邻居
    for idx in range(neighbors_indptr[i], neighbors_indptr[i+1]):
        j = neighbors_indices[idx]
        # 如果是同身份的
        if identities[j] == l_i:
            sigma_sum += calculate_perceived_opinion_func(opinions, morals, i, j)
            count += 1
    
    # 返回平均值，如果没有同身份邻居，则返回0
    if count > 0:
        return sigma_sum / count
    return 0.0


@njit
def calculate_relationship_coefficient_func(adj_matrix, identities, morals, opinions, i, j, same_identity_sigmas, cohesion_factor):
    """
    计算agent i与agent j之间的关系系数
    
    参数:
    adj_matrix -- 邻接矩阵
    identities -- 身份数组
    morals -- 道德值数组
    opinions -- 意见数组
    i -- agent i的索引
    j -- agent j的索引
    same_identity_sigmas -- agent i的同身份邻居的感知意见值数组或平均值
    
    返回:
    关系系数值
    """
    a_ij = adj_matrix[i, j]
    if a_ij == 0:  # 如果不是邻居，关系系数为0
        return 0
        
    l_i = identities[i]
    l_j = identities[j]
    m_i = morals[i]
    m_j = morals[j]

    if l_i == l_j:
        cohesion_factor = cohesion_factor
    else:
        cohesion_factor = 0
    
    # 计算感知意见
    sigma_ij = calculate_perceived_opinion_func(opinions, morals, i, j)
    sigma_ji = calculate_perceived_opinion_func(opinions, morals, j, i)
    
    # 根据极化三角框架公式计算关系系数
    if l_i != l_j and m_i == 1 and m_j == 1 and (sigma_ij * sigma_ji) < 0:
        return -a_ij+cohesion_factor
    elif l_i == l_j and m_i == 1 and m_j == 1 and (sigma_ij * sigma_ji) < 0:
        # 使用传入的同身份平均感知意见值
        if sigma_ij == 0:  # 避免除零错误
            return a_ij+cohesion_factor
        return ((a_ij / sigma_ij) * same_identity_sigmas)
    else:
        return a_ij+cohesion_factor


@njit
def step_calculation(opinions, morals, identities, adj_matrix, 
                    neighbors_indices, neighbors_indptr,  
                    alpha, beta, gamma, delta, u, influence_factor, cohesion_factor):
    """
    执行一步模拟计算，使用numba加速
    
    参数:
    opinions -- 代理意见数组
    morals -- 代理道德值数组
    identities -- 代理身份数组
    adj_matrix -- 邻接矩阵
    neighbors_indices -- CSR格式的邻居索引数组
    neighbors_indptr -- CSR格式的邻居指针数组
    alpha -- 自我激活系数
    beta -- 社会影响系数
    gamma -- 道德化影响系数
    delta -- 意见衰减率
    u -- 意见激活系数
    influence_factor -- 影响因子
    cohesion_factor -- 身份凝聚力因子
    
    返回:
    更新后的opinions, self_activation, social_influence, interaction_counts
    interaction_counts -- 16种交互类型的计数，按照规则顺序排列
    """
    num_agents = len(opinions)
    opinion_changes = np.zeros(num_agents, dtype=np.float64)
    self_activation = np.zeros(num_agents, dtype=np.float64)
    social_influence = np.zeros(num_agents, dtype=np.float64)
    
    # 创建16种交互类型的计数器
    interaction_counts = np.zeros(16, dtype=np.int32)
    
    # 预计算所有agent的同身份邻居平均感知意见
    same_identity_sigmas = np.zeros(num_agents, dtype=np.float64)
    for i in range(num_agents):
        same_identity_sigmas[i] = calculate_same_identity_sigma_func(
            opinions, morals, identities, neighbors_indices, neighbors_indptr, i)
    
    for i in range(num_agents):
        # 计算自我感知
        sigma_ii = np.sign(opinions[i]) if opinions[i] != 0 else 0
        
        # 计算邻居影响总和
        neighbor_influence = 0.0
        
        # 遍历i的所有邻居（使用CSR格式）
        for idx in range(neighbors_indptr[i], neighbors_indptr[i+1]):
            j = neighbors_indices[idx]
            
            # 统计交互类型
            # 判断意见方向是否相同
            same_opinion_direction = (opinions[i] * opinions[j] >= 0)
            
            # 判断身份是否相同
            same_identity = (identities[i] == identities[j])
            
            # 获取道德状态
            m_i = morals[i]
            m_j = morals[j]
            
            # 计算规则索引 (0-15)
            # 规则索引构成：
            # 0-3: 相同意见方向，相同身份 (00, 01, 10, 11)
            # 4-7: 相同意见方向，不同身份 (00, 01, 10, 11)
            # 8-11: 不同意见方向，相同身份 (00, 01, 10, 11)
            # 12-15: 不同意见方向，不同身份 (00, 01, 10, 11)
            rule_index = 0
            
            # 先设置意见方向和身份位
            if not same_opinion_direction:
                rule_index += 8
            if not same_identity:
                rule_index += 4
                
            # 然后设置道德状态位
            rule_index += m_i * 2 + m_j
            
            # 增加对应规则计数
            interaction_counts[rule_index] += 1
            
            # 继续原有的计算逻辑
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
        
        # 计算并存储自我激活项
        # self_activation[i] = alpha[i] * sigma_ii
        self_activation[i] = alpha[i] * opinions[i]
        
        # 计算并存储社会影响项
        social_influence[i] = (beta / (1 + gamma[i] * morals[i])) * neighbor_influence
        
        # 计算意见变化率
        # 回归中性意见项
        regression_term = -delta * opinions[i]
        
        # 意见激活项
        activation_term = u[i] * np.tanh(
            self_activation[i] + social_influence[i]
        )
        
        # 总变化
        opinion_changes[i] = regression_term + activation_term
    
    # 应用意见变化，使用小步长避免过大变化
    opinions_new = opinions.copy()
    opinions_new += influence_factor * opinion_changes
    
    # 确保意见值在[-1, 1]范围内
    opinions_new = np.clip(opinions_new, -1, 1)
    
    return opinions_new, self_activation, social_influence, interaction_counts