from polarization_triangle.utils.data_manager import save_trajectory_to_csv
from typing import List
import numpy as np

def save_trajectory(history: List[np.ndarray], output_path: str) -> str:
    """
    将轨迹数据保存为CSV文件
    
    参数:
    history -- 意见历史数据列表
    output_path -- 输出CSV文件路径
    
    返回:
    保存的文件路径
    """
    return save_trajectory_to_csv(history, output_path) 

def run_simulation_with_trajectory(sim, steps=500):
    history = []
    history.append(sim.opinions.copy())
    for _ in range(steps):
        sim.step()
        history.append(sim.opinions.copy())
    return history