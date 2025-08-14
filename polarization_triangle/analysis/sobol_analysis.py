"""
Sobol敏感性分析框架
对极化三角框架中的关键参数(α, β, γ, cohesion_factor)进行敏感性分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import time
import pickle
import os
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings

try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    from SALib.util import read_param_file
except ImportError:
    warnings.warn("SALib not installed. Install with: pip install SALib")
    saltelli = None
    sobol = None

from ..core.config import SimulationConfig
from ..core.simulation import Simulation
from .sensitivity_metrics import SensitivityMetrics


@dataclass
class SobolConfig:
    """Sobol敏感性分析配置"""
    # 参数范围定义
    parameter_bounds: Dict[str, List[float]] = field(default_factory=lambda: {
        'alpha': [0, 1],        # 自我激活系数
        'beta': [0.0, 0.2],        # 社会影响系数  
        'gamma': [0.2, 2.0],        # 道德化影响系数
        'cohesion_factor': [0.0, 0.5]  # 身份凝聚力因子
    })
    
    # 采样参数
    n_samples: int = 1000           # 基础样本数，总样本数为 N * (2D + 2)
    n_runs: int = 10                 # 每个参数组合运行次数
    
    # 模拟参数
    base_config: Optional[SimulationConfig] = None
    num_steps: int = 300            # 模拟步数
    
    # 计算参数
    n_processes: int = 4            # 并行进程数
    save_intermediate: bool = True   # 是否保存中间结果
    output_dir: str = "sobol_results"  # 输出目录
    
    # 分析参数
    confidence_level: float = 0.95   # 置信水平
    bootstrap_samples: int = 1000     # Bootstrap样本数

    def __post_init__(self):
        if self.base_config is None:
            self.base_config = SimulationConfig(
                num_agents=200,
                network_type='lfr',
                network_params={
                    'tau1': 3, 'tau2': 1.5, 'mu': 0.1,
                    'average_degree': 5, 'min_community': 10
                },
                opinion_distribution='uniform',
                morality_rate=0,
                cluster_identity=False,
                cluster_morality=False,
                cluster_opinion=False,
                # Zealot配置
                zealot_count=30,
                enable_zealots=True,
                zealot_mode="random",
                zealot_morality=True,
                zealot_identity_allocation=False
            )


class SobolAnalyzer:
    """Sobol敏感性分析器"""
    
    def __init__(self, config: SobolConfig):
        self.config = config
        self.param_names = list(config.parameter_bounds.keys())
        self.param_bounds = [config.parameter_bounds[name] for name in self.param_names]
        
        # SALib问题定义
        self.problem = {
            'num_vars': len(self.param_names),
            'names': self.param_names,
            'bounds': self.param_bounds
        }
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 结果存储
        self.param_samples = None
        self.simulation_results = None
        self.sensitivity_indices = None
        
        # 初始化指标计算器
        self.metrics_calculator = SensitivityMetrics()
    
    def generate_samples(self) -> np.ndarray:
        """生成Saltelli采样"""
        if saltelli is None:
            raise ImportError("SALib is required for Sobol analysis")
        
        print(f"正在生成Saltelli样本...")
        print(f"参数数量: {len(self.param_names)}")
        print(f"基础样本数: {self.config.n_samples}")
        print(f"总样本数: {self.config.n_samples * (2 * len(self.param_names) + 2)}")
        
        self.param_samples = saltelli.sample(self.problem, self.config.n_samples)
        
        # 保存样本
        if self.config.save_intermediate:
            np.save(os.path.join(self.config.output_dir, 'param_samples.npy'), 
                   self.param_samples)
        
        print(f"生成样本完成，形状: {self.param_samples.shape}")
        return self.param_samples
    
    def run_single_simulation(self, params: Dict[str, float]) -> Dict[str, float]:
        """运行单次模拟"""
        # 创建配置副本
        config = copy.deepcopy(self.config.base_config)
        
        # 设置参数
        config.alpha = params['alpha']
        config.beta = params['beta'] 
        config.gamma = params['gamma']
        
        # 处理cohesion_factor参数
        if hasattr(config, 'network_params') and config.network_params:
            if isinstance(config.network_params, dict):
                config.network_params = config.network_params.copy()
                config.network_params['cohesion_factor'] = params['cohesion_factor']
            else:
                config.network_params = {'cohesion_factor': params['cohesion_factor']}
        else:
            config.network_params = {'cohesion_factor': params['cohesion_factor']}
        
        # 运行多次取平均
        all_metrics = []
        for run in range(self.config.n_runs):
            try:
                # 创建并运行模拟
                sim = Simulation(config)
                # 运行指定步数
                for _ in range(self.config.num_steps):
                    sim.step()
                
                # 计算指标
                metrics = self.metrics_calculator.calculate_all_metrics(sim)
                all_metrics.append(metrics)
                
            except Exception as e:
                print(f"模拟运行失败: {e}")
                continue
        
        if not all_metrics:
            # 如果所有运行都失败，返回默认值
            return self.metrics_calculator.get_default_metrics()
        
        # 计算平均值
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m and not np.isnan(m[key])]
            if values:
                avg_metrics[key] = np.mean(values)
            else:
                avg_metrics[key] = 0.0
        
        return avg_metrics
    
    def run_batch_simulations(self, param_samples: np.ndarray) -> List[Dict[str, float]]:
        """批量运行模拟"""
        print(f"开始运行 {len(param_samples)} 个参数组合的模拟...")
        
        # 准备参数列表
        param_list = []
        for i, sample in enumerate(param_samples):
            params = {name: sample[j] for j, name in enumerate(self.param_names)}
            param_list.append(params)
        
        results = []
        
        if self.config.n_processes > 1:
            # 并行执行
            with ProcessPoolExecutor(max_workers=self.config.n_processes) as executor:
                # 提交任务
                future_to_params = {
                    executor.submit(self.run_single_simulation, params): i 
                    for i, params in enumerate(param_list)
                }
                
                # 收集结果
                with tqdm(total=len(param_list), desc="执行模拟") as pbar:
                    for future in as_completed(future_to_params):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            print(f"任务执行失败: {e}")
                            results.append(self.metrics_calculator.get_default_metrics())
                        pbar.update(1)
        else:
            # 串行执行
            for params in tqdm(param_list, desc="执行模拟"):
                result = self.run_single_simulation(params)
                results.append(result)
        
        # 保存结果
        if self.config.save_intermediate:
            with open(os.path.join(self.config.output_dir, 'simulation_results.pkl'), 'wb') as f:
                pickle.dump(results, f)
        
        self.simulation_results = results
        return results
    
    def calculate_sensitivity_indices(self, results: List[Dict[str, float]]) -> Dict[str, Dict]:
        """计算Sobol敏感性指数"""
        if sobol is None:
            raise ImportError("SALib is required for Sobol analysis")
        
        print("计算Sobol敏感性指数...")
        
        # 获取所有输出指标名称
        output_names = list(results[0].keys())
        sensitivity_indices = {}
        
        for output_name in output_names:
            try:
                # 提取该指标的所有值
                Y = np.array([r[output_name] for r in results])
                
                # 检查是否有有效值
                if np.all(np.isnan(Y)) or np.all(Y == 0):
                    print(f"警告: 指标 {output_name} 所有值都无效，跳过")
                    continue
                
                # 计算Sobol指数
                Si = sobol.analyze(self.problem, Y, print_to_console=False)
                
                # 存储结果
                sensitivity_indices[output_name] = {
                    'S1': Si['S1'],           # 一阶敏感性指数
                    'S1_conf': Si['S1_conf'], # 一阶敏感性置信区间
                    'ST': Si['ST'],           # 总敏感性指数
                    'ST_conf': Si['ST_conf'], # 总敏感性置信区间
                    'S2': Si['S2'],           # 二阶交互效应
                    'S2_conf': Si['S2_conf']  # 二阶交互效应置信区间
                }
                
            except Exception as e:
                print(f"计算指标 {output_name} 的敏感性时出错: {e}")
                continue
        
        # 保存结果
        if self.config.save_intermediate:
            with open(os.path.join(self.config.output_dir, 'sensitivity_indices.pkl'), 'wb') as f:
                pickle.dump(sensitivity_indices, f)
        
        self.sensitivity_indices = sensitivity_indices
        return sensitivity_indices
    
    def run_complete_analysis(self) -> Dict[str, Dict]:
        """运行完整的Sobol敏感性分析"""
        start_time = time.time()
        
        print("=" * 60)
        print("开始Sobol敏感性分析")
        print("=" * 60)
        
        # 1. 生成参数样本
        if self.param_samples is None:
            self.generate_samples()
        
        # 2. 运行模拟
        if self.simulation_results is None:
            self.run_batch_simulations(self.param_samples)
        
        # 3. 计算敏感性指数
        if self.sensitivity_indices is None:
            self.calculate_sensitivity_indices(self.simulation_results)
        
        end_time = time.time()
        print(f"\n分析完成！总耗时: {end_time - start_time:.2f} 秒")
        
        return self.sensitivity_indices
    
    def load_results(self, results_dir: str = None) -> Dict[str, Dict]:
        """加载已保存的分析结果"""
        if results_dir is None:
            results_dir = self.config.output_dir
        
        # 加载参数样本
        param_file = os.path.join(results_dir, 'param_samples.npy')
        if os.path.exists(param_file):
            self.param_samples = np.load(param_file)
        
        # 加载模拟结果
        results_file = os.path.join(results_dir, 'simulation_results.pkl')
        if os.path.exists(results_file):
            with open(results_file, 'rb') as f:
                self.simulation_results = pickle.load(f)
        
        # 加载敏感性指数
        sensitivity_file = os.path.join(results_dir, 'sensitivity_indices.pkl')
        if os.path.exists(sensitivity_file):
            with open(sensitivity_file, 'rb') as f:
                self.sensitivity_indices = pickle.load(f)
        
        return self.sensitivity_indices
    
    def get_summary_table(self) -> pd.DataFrame:
        """生成敏感性分析摘要表"""
        if self.sensitivity_indices is None:
            raise ValueError("需要先运行敏感性分析")
        
        summary_data = []
        
        for output_name, indices in self.sensitivity_indices.items():
            for i, param_name in enumerate(self.param_names):
                summary_data.append({
                    'Output': output_name,
                    'Parameter': param_name,
                    'S1': indices['S1'][i],
                    'S1_conf': indices['S1_conf'][i],
                    'ST': indices['ST'][i], 
                    'ST_conf': indices['ST_conf'][i],
                    'Interaction': indices['ST'][i] - indices['S1'][i]
                })
        
        return pd.DataFrame(summary_data)
    
    def export_results(self, filename: str = None):
        """导出结果为Excel文件"""
        if filename is None:
            filename = os.path.join(self.config.output_dir, 'sobol_results.xlsx')
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 导出摘要表
            summary_df = self.get_summary_table()
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # 导出详细的敏感性指数
            for output_name, indices in self.sensitivity_indices.items():
                df_data = {
                    'Parameter': self.param_names,
                    'S1': indices['S1'],
                    'S1_conf': indices['S1_conf'],
                    'ST': indices['ST'],
                    'ST_conf': indices['ST_conf']
                }
                df = pd.DataFrame(df_data)
                sheet_name = output_name[:31]  # Excel工作表名称限制
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"结果已导出到: {filename}")


def run_sobol_analysis_example():
    """运行Sobol敏感性分析的示例"""
    # 创建配置
    config = SobolConfig(
        n_samples=100,  # 小样本用于测试
        n_runs=3,
        n_processes=2,
        output_dir="example_sobol_results"
    )
    
    # 创建分析器
    analyzer = SobolAnalyzer(config)
    
    # 运行分析
    results = analyzer.run_complete_analysis()
    
    # 生成摘要
    summary = analyzer.get_summary_table()
    print("\n敏感性分析摘要:")
    print(summary.head(10))
    
    # 导出结果
    analyzer.export_results()
    
    return analyzer


if __name__ == "__main__":
    # 运行示例
    analyzer = run_sobol_analysis_example() 