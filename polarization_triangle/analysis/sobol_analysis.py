"""
Sobol sensitivity analysis framework
Perform sensitivity analysis on key parameters (α, β, γ, cohesion_factor) in the polarization triangle framework
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
    """Sobol sensitivity analysis configuration"""
    # Parameter range definition
    parameter_bounds: Dict[str, List[float]] = field(default_factory=lambda: {
        'alpha': [0, 1],        # Self-activation coefficient
        'beta': [0.0, 0.2],        # Social influence coefficient  
        'gamma': [0.2, 2.0],        # Moralization influence coefficient
        'cohesion_factor': [0.0, 0.5]  # Identity cohesion factor
    })
    
    # Sampling parameters
    n_samples: int = 1000           # Base sample number, total samples = N * (2D + 2)
    n_runs: int = 10                 # Number of runs per parameter combination
    
    # Simulation parameters
    base_config: Optional[SimulationConfig] = None
    num_steps: int = 300            # Number of simulation steps
    
    # Condition parameters
    structural_alignment: str = 'low'   # 'high' (cluster_identity=True) or 'low' (cluster_identity=False)
    morality_ratio: float = 0.0         # Moralization rate: 0.0 or 0.3
    
    # Computation parameters
    n_processes: int = 4            # Number of parallel processes
    save_intermediate: bool = True   # Whether to save intermediate results
    output_dir: str = "sobol_results"  # Output directory
    
    # Analysis parameters
    confidence_level: float = 0.95   # Confidence level
    bootstrap_samples: int = 1000     # Bootstrap sample count

    def __post_init__(self):
        if self.base_config is None:
            # Set cluster_identity and morality_rate according to condition parameters
            cluster_identity_value = (self.structural_alignment == 'high')
            
            self.base_config = SimulationConfig(
                num_agents=200,
                network_type='lfr',
                network_params={
                    'tau1': 3, 'tau2': 1.5, 'mu': 0.1,
                    'average_degree': 5, 'min_community': 10
                },
                opinion_distribution='uniform',
                morality_rate=self.morality_ratio,
                cluster_identity=cluster_identity_value,
                cluster_morality=False,
                cluster_opinion=False,
                # Zealot configuration
                zealot_count=30,
                enable_zealots=True,
                zealot_mode="random",
                zealot_morality=True,
                zealot_identity_allocation=False,
                use_network_pool=False
            )
        else:
            # If base_config is provided, update condition parameters
            self.base_config.morality_rate = self.morality_ratio
            self.base_config.cluster_identity = (self.structural_alignment == 'high')


class SobolAnalyzer:
    """Sobol sensitivity analyzer"""
    
    def __init__(self, config: SobolConfig):
        self.config = config
        self.param_names = list(config.parameter_bounds.keys())
        self.param_bounds = [config.parameter_bounds[name] for name in self.param_names]
        
        # SALib problem definition
        self.problem = {
            'num_vars': len(self.param_names),
            'names': self.param_names,
            'bounds': self.param_bounds
        }
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Result storage
        self.param_samples = None
        self.simulation_results = None
        self.sensitivity_indices = None
        
        # Initialize metrics calculator
        self.metrics_calculator = SensitivityMetrics()
    
    def generate_samples(self) -> np.ndarray:
        """Generate Saltelli samples"""
        if saltelli is None:
            raise ImportError("SALib is required for Sobol analysis")
        
        print(f"Generating Saltelli samples...")
        print(f"Parameter count: {len(self.param_names)}")
        print(f"Base sample count: {self.config.n_samples}")
        print(f"Total sample count: {self.config.n_samples * (2 * len(self.param_names) + 2)}")
        
        self.param_samples = saltelli.sample(self.problem, self.config.n_samples)
        
        # Save samples
        if self.config.save_intermediate:
            np.save(os.path.join(self.config.output_dir, 'param_samples.npy'), 
                   self.param_samples)
        
        print(f"Sample generation complete, shape: {self.param_samples.shape}")
        return self.param_samples
    
    def run_single_simulation(self, params: Dict[str, float]) -> Dict[str, float]:
        """Run single simulation"""
        # Create configuration copy
        config = copy.deepcopy(self.config.base_config)
        
        # Set parameters
        config.alpha = params['alpha']
        config.beta = params['beta'] 
        config.gamma = params['gamma']
        
        # Handle cohesion_factor parameter
        if hasattr(config, 'network_params') and config.network_params:
            if isinstance(config.network_params, dict):
                config.network_params = config.network_params.copy()
                config.network_params['cohesion_factor'] = params['cohesion_factor']
            else:
                config.network_params = {'cohesion_factor': params['cohesion_factor']}
        else:
            config.network_params = {'cohesion_factor': params['cohesion_factor']}
        
        # Run multiple times and take average
        all_metrics = []
        for run in range(self.config.n_runs):
            try:
                # Create and run simulation
                sim = Simulation(config)
                # Run for specified steps
                for _ in range(self.config.num_steps):
                    sim.step()
                
                # Calculate metrics
                metrics = self.metrics_calculator.calculate_all_metrics(sim)
                all_metrics.append(metrics)
                
            except Exception as e:
                print(f"Simulation run failed: {e}")
                continue
        
        if not all_metrics:
            # If all runs fail, return default values
            return self.metrics_calculator.get_default_metrics()
        
        # Calculate average values
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m and not np.isnan(m[key])]
            if values:
                avg_metrics[key] = np.mean(values)
            else:
                avg_metrics[key] = 0.0
        
        return avg_metrics
    
    def run_batch_simulations(self, param_samples: np.ndarray) -> List[Dict[str, float]]:
        """Run batch simulations"""
        print(f"Starting to run simulations for {len(param_samples)} parameter combinations...")
        
        # Prepare parameter list
        param_list = []
        for i, sample in enumerate(param_samples):
            params = {name: sample[j] for j, name in enumerate(self.param_names)}
            param_list.append(params)
        
        results = []
        
        if self.config.n_processes > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.config.n_processes) as executor:
                # Submit tasks
                future_to_params = {
                    executor.submit(self.run_single_simulation, params): i 
                    for i, params in enumerate(param_list)
                }
                
                # Collect results
                with tqdm(total=len(param_list), desc="Executing simulations") as pbar:
                    for future in as_completed(future_to_params):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            print(f"Task execution failed: {e}")
                            results.append(self.metrics_calculator.get_default_metrics())
                        pbar.update(1)
        else:
            # Serial execution
            for params in tqdm(param_list, desc="Executing simulations"):
                result = self.run_single_simulation(params)
                results.append(result)
        
        # Save results
        if self.config.save_intermediate:
            with open(os.path.join(self.config.output_dir, 'simulation_results.pkl'), 'wb') as f:
                pickle.dump(results, f)
        
        self.simulation_results = results
        return results
    
    def calculate_sensitivity_indices(self, results: List[Dict[str, float]]) -> Dict[str, Dict]:
        """Calculate Sobol sensitivity indices"""
        if sobol is None:
            raise ImportError("SALib is required for Sobol analysis")
        
        print("Calculating Sobol sensitivity indices...")
        
        # Get all output metric names
        output_names = list(results[0].keys())
        sensitivity_indices = {}
        
        for output_name in output_names:
            try:
                # Extract all values for this metric
                Y = np.array([r[output_name] for r in results])
                
                # Check if there are valid values
                if np.all(np.isnan(Y)) or np.all(Y == 0):
                    print(f"Warning: All values for metric {output_name} are invalid, skipping")
                    continue
                
                # Calculate Sobol indices
                Si = sobol.analyze(self.problem, Y, print_to_console=False)
                
                # Store results
                sensitivity_indices[output_name] = {
                    'S1': Si['S1'],           # First-order sensitivity index
                    'S1_conf': Si['S1_conf'], # First-order sensitivity confidence interval
                    'ST': Si['ST'],           # Total sensitivity index
                    'ST_conf': Si['ST_conf'], # Total sensitivity confidence interval
                    'S2': Si['S2'],           # Second-order interaction effects
                    'S2_conf': Si['S2_conf']  # Second-order interaction effects confidence interval
                }
                
            except Exception as e:
                print(f"Error calculating sensitivity for metric {output_name}: {e}")
                continue
        
        # 保存结果
        if self.config.save_intermediate:
            with open(os.path.join(self.config.output_dir, 'sensitivity_indices.pkl'), 'wb') as f:
                pickle.dump(sensitivity_indices, f)
        
        self.sensitivity_indices = sensitivity_indices
        return sensitivity_indices
    
    def run_complete_analysis(self) -> Dict[str, Dict]:
        """Run complete Sobol sensitivity analysis"""
        start_time = time.time()
        
        print("=" * 60)
        print("Starting Sobol sensitivity analysis")
        print("=" * 60)
        
        # 1. Generate parameter samples
        if self.param_samples is None:
            self.generate_samples()
        
        # 2. Run simulations
        if self.simulation_results is None:
            self.run_batch_simulations(self.param_samples)
        
        # 3. Calculate sensitivity indices
        if self.sensitivity_indices is None:
            self.calculate_sensitivity_indices(self.simulation_results)
        
        end_time = time.time()
        print(f"\nAnalysis complete! Total time: {end_time - start_time:.2f} seconds")
        
        return self.sensitivity_indices
    
    def load_results(self, results_dir: str = None) -> Dict[str, Dict]:
        """Load saved analysis results"""
        if results_dir is None:
            results_dir = self.config.output_dir
        
        # Load parameter samples
        param_file = os.path.join(results_dir, 'param_samples.npy')
        if os.path.exists(param_file):
            self.param_samples = np.load(param_file)
        
        # Load simulation results
        results_file = os.path.join(results_dir, 'simulation_results.pkl')
        if os.path.exists(results_file):
            with open(results_file, 'rb') as f:
                self.simulation_results = pickle.load(f)
        
        # Load sensitivity indices
        sensitivity_file = os.path.join(results_dir, 'sensitivity_indices.pkl')
        if os.path.exists(sensitivity_file):
            with open(sensitivity_file, 'rb') as f:
                self.sensitivity_indices = pickle.load(f)
        
        return self.sensitivity_indices
    
    def get_summary_table(self) -> pd.DataFrame:
        """Generate sensitivity analysis summary table"""
        if self.sensitivity_indices is None:
            raise ValueError("Need to run sensitivity analysis first")
        
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
        """Export results to Excel file"""
        if filename is None:
            filename = os.path.join(self.config.output_dir, 'sobol_results.xlsx')
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Export summary table
            summary_df = self.get_summary_table()
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Export detailed sensitivity indices
            for output_name, indices in self.sensitivity_indices.items():
                df_data = {
                    'Parameter': self.param_names,
                    'S1': indices['S1'],
                    'S1_conf': indices['S1_conf'],
                    'ST': indices['ST'],
                    'ST_conf': indices['ST_conf']
                }
                df = pd.DataFrame(df_data)
                sheet_name = output_name[:31]  # Excel worksheet name limit
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"Results exported to: {filename}")


def run_sobol_analysis_example():
    """Run example of Sobol sensitivity analysis"""
    # Create configuration
    config = SobolConfig(
        n_samples=100,  # Small sample for testing
        n_runs=3,
        n_processes=2,
        output_dir="example_sobol_results"
    )
    
    # Create analyzer
    analyzer = SobolAnalyzer(config)
    
    # Run analysis
    results = analyzer.run_complete_analysis()
    
    # Generate summary
    summary = analyzer.get_summary_table()
    print("\nSensitivity analysis summary:")
    print(summary.head(10))
    
    # Export results
    analyzer.export_results()
    
    return analyzer


if __name__ == "__main__":
    # Run example
    analyzer = run_sobol_analysis_example() 