#!/usr/bin/env python3
"""
Sobol sensitivity analysis main execution script
Performs sensitivity analysis on key parameters of the polarization triangle framework
"""

import os
import sys
import argparse
import time
from pathlib import Path
import json
from datetime import datetime

# Add project root directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from polarization_triangle.analysis.sobol_analysis import SobolAnalyzer, SobolConfig
from polarization_triangle.analysis.sensitivity_visualizer import SensitivityVisualizer
from polarization_triangle.core.config import SimulationConfig


def create_analysis_configs(structural_alignment='low', morality_ratio=0.0):
    """Create different analysis configurations"""
    configs = {}
    
    # Adjust output directory suffix based on condition parameters
    condition_suffix = f"_sa{structural_alignment}_mr{morality_ratio}"
    
    configs['quick'] = SobolConfig(
        n_samples=64,
        n_runs=2,
        n_processes=2,
        num_steps=100,
        structural_alignment=structural_alignment,
        morality_ratio=morality_ratio,
        output_dir=f"results/sobol_results_quick{condition_suffix}"
    )
    
    configs['standard'] = SobolConfig(
        n_samples=512,
        n_runs=3,
        n_processes=4,
        num_steps=200,
        structural_alignment=structural_alignment,
        morality_ratio=morality_ratio,
        output_dir=f"results/sobol_results_standard{condition_suffix}"
    )

    configs['high_precision'] = SobolConfig(
        n_samples=2048,
        n_runs=10,
        n_processes=8,
        num_steps=300,
        structural_alignment=structural_alignment,
        morality_ratio=morality_ratio,
        output_dir=f"results/sobol_results_test1{condition_suffix}"
    )

    configs['full'] = SobolConfig(
        n_samples=4096,
        n_runs=50,
        n_processes=8,
        num_steps=300,
        structural_alignment=structural_alignment,
        morality_ratio=morality_ratio,
        output_dir=f"results/sobol_results_full{condition_suffix}"
    )

    
    return configs


def save_parameter_record(analyzer: SobolAnalyzer, config_name: str, 
                         start_time: float, end_time: float = None):
    """Save parameter configuration record file"""
    
    print("Saving parameter configuration record...")
    
    # Create record file path
    record_file_txt = os.path.join(analyzer.config.output_dir, "parameter_record.txt")
    record_file_json = os.path.join(analyzer.config.output_dir, "parameter_record.json")
    
    # Get current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate total sample count
    total_samples = analyzer.config.n_samples * (2 * len(analyzer.param_names) + 2)
    total_simulations = total_samples * analyzer.config.n_runs
    total_steps = total_simulations * analyzer.config.num_steps
    
    # Prepare parameter record data
    record_data = {
        "analysis_info": {
            "config_name": config_name,
            "analysis_time": current_time,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)) if end_time else "Running",
            "duration_seconds": end_time - start_time if end_time else "Running",
            "output_directory": analyzer.config.output_dir
        },
        "sobol_analysis_config": {
            "parameter_bounds": analyzer.config.parameter_bounds,
            "n_samples": analyzer.config.n_samples,
            "n_runs": analyzer.config.n_runs,
            "num_steps": analyzer.config.num_steps,
            "n_processes": analyzer.config.n_processes,
            "confidence_level": analyzer.config.confidence_level,
            "bootstrap_samples": analyzer.config.bootstrap_samples,
            "save_intermediate": analyzer.config.save_intermediate
        },
        "simulation_config": {
            "num_agents": analyzer.config.base_config.num_agents,
            "network_type": analyzer.config.base_config.network_type,
            "network_params": analyzer.config.base_config.network_params,
            "opinion_distribution": analyzer.config.base_config.opinion_distribution,
            "morality_rate": analyzer.config.base_config.morality_rate,
            "cluster_identity": analyzer.config.base_config.cluster_identity,
            "cluster_morality": analyzer.config.base_config.cluster_morality,
            "cluster_opinion": analyzer.config.base_config.cluster_opinion,
            "influence_factor": analyzer.config.base_config.influence_factor,
            "tolerance": analyzer.config.base_config.tolerance,
            "delta": analyzer.config.base_config.delta,
            "u": analyzer.config.base_config.u,
            "alpha": analyzer.config.base_config.alpha,
            "beta": analyzer.config.base_config.beta,
            "gamma": analyzer.config.base_config.gamma
        },
        "zealot_config": {
            "zealot_count": analyzer.config.base_config.zealot_count,
            "enable_zealots": analyzer.config.base_config.enable_zealots,
            "zealot_mode": analyzer.config.base_config.zealot_mode,
            "zealot_opinion": analyzer.config.base_config.zealot_opinion,
            "zealot_morality": analyzer.config.base_config.zealot_morality,
            "zealot_identity_allocation": analyzer.config.base_config.zealot_identity_allocation
        },
        "computation_complexity": {
            "analyzed_parameters": analyzer.param_names,
            "parameter_count": len(analyzer.param_names),
            "base_samples": analyzer.config.n_samples,
            "total_samples": total_samples,
            "runs_per_sample": analyzer.config.n_runs,
            "total_simulations": total_simulations,
            "steps_per_simulation": analyzer.config.num_steps,
            "total_computation_steps": total_steps,
            "parallel_processes": analyzer.config.n_processes
        },
        "output_metrics": {
            "polarization_metrics": [
                "polarization_index",
                "opinion_variance", 
                "extreme_ratio",
                "identity_polarization"
            ],
            "convergence_metrics": [
                "mean_abs_opinion",
                "final_stability"
            ],
            "dynamics_metrics": [
                "trajectory_length",
                "oscillation_frequency",
                "group_divergence"
            ],
            "identity_metrics": [
                "identity_variance_ratio",
                "cross_identity_correlation",
                "variance_per_identity_1",
                "variance_per_identity_neg1",
                "variance_per_identity_mean"
            ]
        }
    }
    
    # Save JSON format
    with open(record_file_json, 'w', encoding='utf-8') as f:
        json.dump(record_data, f, ensure_ascii=False, indent=2)
    
    # Save text format (more readable)
    with open(record_file_txt, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Sobol Sensitivity Analysis Parameter Configuration Record\n")
        f.write("="*80 + "\n\n")
        
        # Analysis Info
        f.write("[Analysis Info]\n")
        f.write(f"Config Name: {config_name}\n")
        f.write(f"Analysis Time: {current_time}\n")
        f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        if end_time:
            f.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
            f.write(f"Total Duration: {end_time - start_time:.2f} seconds\n")
        f.write(f"Output Directory: {analyzer.config.output_dir}\n\n")
        
        # Sobol Analysis Config
        f.write("[Sobol Sensitivity Analysis Config]\n")
        f.write(f"Base Samples: {analyzer.config.n_samples}\n")
        f.write(f"Total Samples: {total_samples} (N * (2D + 2))\n")
        f.write(f"Runs per Sample: {analyzer.config.n_runs}\n")
        f.write(f"Steps per Simulation: {analyzer.config.num_steps}\n")
        f.write(f"Parallel Processes: {analyzer.config.n_processes}\n")
        f.write(f"Confidence Level: {analyzer.config.confidence_level}\n")
        f.write(f"Bootstrap Samples: {analyzer.config.bootstrap_samples}\n")
        f.write(f"Save Intermediate Results: {analyzer.config.save_intermediate}\n\n")
        
        # Sensitivity Analysis Parameters
        f.write("[Sensitivity Analysis Parameters and Bounds]\n")
        for param, bounds in analyzer.config.parameter_bounds.items():
            f.write(f"{param}: [{bounds[0]}, {bounds[1]}]\n")
        f.write("\n")
        
        # Network Config
        f.write("[Network Config]\n")
        f.write(f"Number of Agents: {analyzer.config.base_config.num_agents}\n")
        f.write(f"Network Type: {analyzer.config.base_config.network_type}\n")
        f.write("Network Parameters:\n")
        for key, value in analyzer.config.base_config.network_params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Simulation Config
        f.write("[Simulation Config]\n")
        f.write(f"Opinion Distribution: {analyzer.config.base_config.opinion_distribution}\n")
        f.write(f"Morality Rate: {analyzer.config.base_config.morality_rate}\n")
        f.write(f"Identity Clustering: {analyzer.config.base_config.cluster_identity}\n")
        f.write(f"Morality Clustering: {analyzer.config.base_config.cluster_morality}\n")
        f.write(f"Opinion Clustering: {analyzer.config.base_config.cluster_opinion}\n")
        f.write(f"Influence Factor: {analyzer.config.base_config.influence_factor}\n")
        f.write(f"Tolerance: {analyzer.config.base_config.tolerance}\n")
        f.write(f"Opinion Decay Rate (δ): {analyzer.config.base_config.delta}\n")
        f.write(f"Opinion Activation Coefficient (u): {analyzer.config.base_config.u}\n")
        f.write(f"Default Self-Activation Coefficient (α): {analyzer.config.base_config.alpha}\n")
        f.write(f"Default Social Influence Coefficient (β): {analyzer.config.base_config.beta}\n")
        f.write(f"Default Moralizing Influence Coefficient (γ): {analyzer.config.base_config.gamma}\n\n")
        
        # Zealot Config
        f.write("[Zealot Config]\n")
        f.write(f"Zealot Count: {analyzer.config.base_config.zealot_count}\n")
        f.write(f"Enable Zealots: {analyzer.config.base_config.enable_zealots}\n")
        f.write(f"Zealot Mode: {analyzer.config.base_config.zealot_mode}\n")
        f.write(f"Zealot Opinion: {analyzer.config.base_config.zealot_opinion}\n")
        f.write(f"Zealot Morality: {analyzer.config.base_config.zealot_morality}\n")
        f.write(f"Allocate Zealots by Identity: {analyzer.config.base_config.zealot_identity_allocation}\n\n")
        
        # Computational Complexity
        f.write("[Computational Complexity]\n")
        f.write(f"Number of Analyzed Parameters: {len(analyzer.param_names)} ({', '.join(analyzer.param_names)})\n")
        f.write(f"Base Samples: {analyzer.config.n_samples}\n")
        f.write(f"Total Samples: {total_samples}\n")
        f.write(f"Runs per Sample: {analyzer.config.n_runs}\n")
        f.write(f"Total Simulation Runs: {total_simulations:,}\n")
        f.write(f"Steps per Simulation: {analyzer.config.num_steps}\n")
        f.write(f"Total Computation Steps: {total_steps:,}\n")
        f.write(f"Parallel Processes: {analyzer.config.n_processes}\n\n")
        
        # Output Metrics
        f.write("[Output Metrics]\n")
        f.write("Polarization-related Metrics:\n")
        for metric in ["polarization_index", "opinion_variance", "extreme_ratio", "identity_polarization"]:
            f.write(f"  - {metric}\n")
        f.write("Convergence-related Metrics:\n")
        for metric in ["mean_abs_opinion", "final_stability"]:
            f.write(f"  - {metric}\n")
        f.write("Dynamic Process Metrics:\n")
        for metric in ["trajectory_length", "oscillation_frequency", "group_divergence"]:
            f.write(f"  - {metric}\n")
        f.write("Identity-related Metrics:\n")
        for metric in ["identity_variance_ratio", "cross_identity_correlation", 
                      "variance_per_identity_1", "variance_per_identity_neg1", "variance_per_identity_mean"]:
            f.write(f"  - {metric}\n")
        f.write("\n")
        
        # Metric Descriptions
        f.write("[Metric Descriptions]\n")
        metric_descriptions = {
            'polarization_index': 'Koudenburg polarization index, measures overall system polarization',
            'opinion_variance': 'Opinion variance, reflects the dispersion of opinions',
            'extreme_ratio': 'Proportion of extreme opinions, ratio of agents with |opinion| > 0.8',
            'identity_polarization': 'Inter-identity polarization difference, variance of mean opinions between different identity groups',
            'mean_abs_opinion': 'Mean absolute opinion, strength of system opinions',
            'final_stability': 'Final stability, coefficient of variation in the final stage',
            'trajectory_length': 'Opinion trajectory length, cumulative distance of opinion changes',
            'oscillation_frequency': 'Oscillation frequency, frequency of opinion direction changes',
            'group_divergence': 'Group divergence, opinion difference between different identity groups',
            'identity_variance_ratio': 'Identity variance ratio, ratio of between-group variance to within-group variance',
            'cross_identity_correlation': 'Cross-identity correlation, correlation coefficient of opinions between different identity groups',
            'variance_per_identity_1': 'Variance of identity group 1, opinion variance within the identity=1 group',
            'variance_per_identity_neg1': 'Variance of identity group -1, opinion variance within the identity=-1 group',
            'variance_per_identity_mean': 'Mean variance of identity groups, average of the variances of the two identity groups'
        }
        
        for metric, description in metric_descriptions.items():
            f.write(f"{metric}: {description}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Record generation complete\n")
        f.write("="*80 + "\n")
    
    print(f"Parameter record saved to:")
    print(f"  - {record_file_txt}")
    print(f"  - {record_file_json}")


def run_sensitivity_analysis(config_name: str = 'standard', 
                           custom_config: SobolConfig = None,
                           load_existing: bool = False,
                           structural_alignment: str = 'low',
                           morality_ratio: float = 0.0):
    """Run sensitivity analysis"""
    
    # Select configuration
    if custom_config:
        config = custom_config
    else:
        configs = create_analysis_configs(structural_alignment, morality_ratio)
        if config_name not in configs:
            raise ValueError(f"Configuration '{config_name}' does not exist. Available configurations: {list(configs.keys())}")
        config = configs[config_name]
    
    print(f"Using configuration: {config_name}")
    print(f"Structural Alignment: {config.structural_alignment} ({'cluster_identity=True' if config.structural_alignment == 'high' else 'cluster_identity=False'})")
    print(f"Morality Ratio: {config.morality_ratio} (morality_rate={config.morality_ratio})")
    print(f"Number of samples: {config.n_samples}")
    print(f"Number of runs: {config.n_runs}")
    print(f"Number of processes: {config.n_processes}")
    print(f"Number of simulation steps: {config.num_steps}")
    print(f"Output directory: {config.output_dir}")
    
    # Create analyzer
    analyzer = SobolAnalyzer(config)
    
    # Load existing results or run a new analysis
    if load_existing:
        print("Attempting to load existing results...")
        try:
            sensitivity_indices = analyzer.load_results()
            if sensitivity_indices:
                print("Successfully loaded existing results")
            else:
                print("No existing results found, starting new analysis...")
                sensitivity_indices = analyzer.run_complete_analysis()
        except Exception as e:
            print(f"Failed to load: {e}")
            print("Starting new analysis...")
            sensitivity_indices = analyzer.run_complete_analysis()
    else:
        # Run complete analysis
        sensitivity_indices = analyzer.run_complete_analysis()
    
    return analyzer, sensitivity_indices


def generate_reports(analyzer: SobolAnalyzer, 
                    sensitivity_indices: dict,
                    create_plots: bool = True,
                    config_name: str = "unknown",
                    start_time: float = None):
    """Generate analysis reports"""
    
    print("\n" + "="*60)
    print("Generating Analysis Reports")
    print("="*60)
    
    # Save parameter configuration record
    if start_time:
        save_parameter_record(analyzer, config_name, start_time, time.time())
    
    # Generate summary table
    try:
        summary_df = analyzer.get_summary_table()
        print("\nSensitivity Analysis Summary (first 10 rows):")
        print(summary_df.head(10).to_string(index=False))
        
        # Export Excel report
        analyzer.export_results()
        
    except Exception as e:
        print(f"Error generating data report: {e}")
    
    # Generate visualization report
    if create_plots:
        try:
            print("\nGenerating visualization report...")
            visualizer = SensitivityVisualizer()
            
            # Create plot output directory
            plot_dir = os.path.join(analyzer.config.output_dir, "plots")
            plot_files = visualizer.create_comprehensive_report(
                sensitivity_indices,
                analyzer.param_samples,
                analyzer.simulation_results,
                plot_dir
            )
            
            print(f"Visualization report saved to: {plot_dir}")
            
        except Exception as e:
            print(f"Error generating visualization report: {e}")
    
    # Print key findings
    print_key_findings(sensitivity_indices)


def print_key_findings(sensitivity_indices: dict):
    """Print key findings"""
    print("\n" + "="*60)
    print("Key Findings")
    print("="*60)
    
    param_names = ['alpha', 'beta', 'gamma', 'cohesion_factor']
    param_labels = ['α (Self-activation)', 'β (Social influence)', 'γ (Moralizing influence)', 'cohesion_factor']
    
    # Calculate average sensitivity
    all_st_values = []
    all_s1_values = []
    
    for output_name, indices in sensitivity_indices.items():
        all_st_values.append(indices['ST'])
        all_s1_values.append(indices['S1'])
    
    if all_st_values:
        import numpy as np
        mean_st = np.mean(all_st_values, axis=0)
        mean_s1 = np.mean(all_s1_values, axis=0)
        mean_interaction = mean_st - mean_s1
        
        # Parameter importance ranking
        importance_ranking = sorted(
            zip(param_labels, mean_st), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        print("\n1. Parameter Importance Ranking (based on average total sensitivity index):")
        for i, (param, value) in enumerate(importance_ranking, 1):
            print(f"   {i}. {param}: {value:.3f}")
        
        # Interaction effect analysis
        print("\n2. Average Interaction Effect Strength (ST - S1):")
        for param, interaction in zip(param_labels, mean_interaction):
            if interaction > 0.1:
                level = "Strong"
            elif interaction > 0.05:
                level = "Medium"
            else:
                level = "Weak"
            print(f"   {param}: {interaction:.3f} ({level})")
        
        # Most sensitive output metrics
        print("\n3. Most Sensitive Parameter for Each Output Metric:")
        for output_name, indices in sensitivity_indices.items():
            max_idx = np.argmax(indices['ST'])
            max_param = param_labels[max_idx]
            max_value = indices['ST'][max_idx]
            print(f"   {output_name}: {max_param} ({max_value:.3f})")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run Sobol sensitivity analysis')
    parser.add_argument('--config', type=str, default='standard', 
                       choices=['quick', 'standard', 'high_precision', 'full', 'test1'],
                       help='Analysis configuration type')
    parser.add_argument('--load', action='store_true', 
                       help='Attempt to load existing results')
    parser.add_argument('--no-plots', action='store_true',
                       help='Do not generate visualization plots')
    parser.add_argument('--output-dir', type=str, 
                       help='Custom output directory')
    parser.add_argument('--n-samples', type=int,
                       help='Custom number of samples')
    parser.add_argument('--n-runs', type=int,
                       help='Custom number of runs')
    parser.add_argument('--n-processes', type=int,
                       help='Custom number of processes')
    parser.add_argument('--structural-alignment', type=str, default='low',
                       choices=['low', 'high'],
                       help='Structural alignment condition: low (cluster_identity=False) or high (cluster_identity=True)')
    parser.add_argument('--morality-ratio', type=float, default=0.0,
                       choices=[0.0, 0.3],
                       help='Morality ratio: 0.0 or 0.3')
    
    args = parser.parse_args()
    
    # Create custom config if custom parameters are provided
    custom_config = None
    if any([args.output_dir, args.n_samples, args.n_runs, args.n_processes]):
        configs = create_analysis_configs(args.structural_alignment, args.morality_ratio)
        base_config = configs[args.config]
        
        custom_config = SobolConfig(
            n_samples=args.n_samples or base_config.n_samples,
            n_runs=args.n_runs or base_config.n_runs,
            n_processes=args.n_processes or base_config.n_processes,
            output_dir=args.output_dir or base_config.output_dir,
            num_steps=base_config.num_steps,
            structural_alignment=args.structural_alignment,
            morality_ratio=args.morality_ratio,
            base_config=base_config.base_config
        )
    
    try:
        start_time = time.time()
        
        # Run analysis
        analyzer, sensitivity_indices = run_sensitivity_analysis(
            config_name=args.config,
            custom_config=custom_config,
            load_existing=args.load,
            structural_alignment=args.structural_alignment,
            morality_ratio=args.morality_ratio
        )
        
        end_time = time.time()
        print(f"\nTotal duration: {end_time - start_time:.2f} seconds")
        print(f"Results saved in: {analyzer.config.output_dir}")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 