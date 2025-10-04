"""
Main script for running manifold optimization comparisons.
"""

import os
import time
import argparse
from typing import Dict, List, Any, Tuple
import numpy as np
from config import Config
from problems import ProblemLoader, define_cost_function, get_problem_description
from algorithms import (
    MPSOOptimizer, MDEOptimizer, MSMANNOptimizer, 
    MDTMAOptimizer, RICAOptimizer, create_pymanopt_optimizer
)
from utils import ResultVisualizer, save_results_to_file


class ManifoldOptimizationComparison:
    """Main class for running optimization comparisons."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.problem_loader = ProblemLoader(self.config.DATA_DIR)
        self.visualizer = ResultVisualizer(self.config.RESULTS_DIR, self.config.PLOT_STYLE)
        
    def run_comparison(self, problem_name: str, problem_type: str, 
                      algorithms: List[str] = None, num_runs: int = None,
                      population_size: int = None, max_iterations: int = None,
                      rank: int = None, d_ambient: int = None,
                      save_results: bool = True, show_plots: bool = True) -> Tuple[Dict, str]:
        """Run comprehensive comparison of optimization algorithms."""
        
        # Set parameters
        num_runs = num_runs or self.config.DEFAULT_NUM_RUNS
        population_size = population_size or self.config.DEFAULT_POPULATION_SIZE
        max_iterations = max_iterations or self.config.DEFAULT_ITERATIONS
        rank = rank or self.config.DEFAULT_RANK
        d_ambient = d_ambient or self.config.THOMSON_AMBIENT_DIM
        
        if algorithms is None:
            algorithms = [
                'mPSO', 'mDE', 'mSMANN', 'mDTMA', 'RICA',
                'steepest_descent', 'conjugate_gradient', 'trust_regions'
            ]
        
        # Generate unique identifier for this experiment
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{problem_type}_{problem_name}_{timestamp}"
        
        print("=" * 70)
        print(f"MANIFOLD OPTIMIZATION COMPARISON")
        print("=" * 70)
        print(f"Problem: {problem_type.upper()} ({problem_name})")
        print(f"Experiment ID: {experiment_id}")
        print(f"Description: {get_problem_description(problem_type)}")
        print(f"Algorithms: {', '.join(algorithms)}")
        print(f"Runs: {num_runs}, Population: {population_size}, Iterations: {max_iterations}")
        print("=" * 70)
        
        # Load problem data
        try:
            manifold, A_matrix, other_data = self.problem_loader.load_problem(
                problem_name, problem_type, rank, d_ambient)
            cost_function = define_cost_function(problem_type, A_matrix, other_data)
            
            print(f"Manifold: {type(manifold).__name__}")
            print(f"Manifold dimension: {manifold.dim if hasattr(manifold, 'dim') else 'N/A'}")
        except Exception as e:
            print(f"Error loading problem: {e}")
            return {}, experiment_id
        
        # Results storage
        results = {alg: {'costs': [], 'convergence': [], 'runtimes': []} for alg in algorithms}
        
        # Run each algorithm
        for alg_name in algorithms:
            print(f"\n{'='*50}")
            print(f"Running {alg_name}")
            print(f"{'='*50}")
            
            for run in range(num_runs):
                print(f"Run {run + 1}/{num_runs}", end="")
                start_time = time.time()
                
                try:
                    if alg_name in ['steepest_descent', 'conjugate_gradient', 'trust_regions']:
                        # Pymanopt gradient-based methods
                        optimizer = create_pymanopt_optimizer(
                            manifold, cost_function, alg_name, max_iterations, verbose=False)
                    else:
                        # Metaheuristic methods
                        optimizer = self._create_metaheuristic_optimizer(
                            alg_name, manifold, cost_function, population_size, max_iterations, run)
                    
                    best_solution, best_cost, convergence = optimizer.optimize()
                    runtime = time.time() - start_time
                    
                    if best_cost is not None and best_cost != float('inf'):
                        results[alg_name]['costs'].append(float(best_cost))
                        results[alg_name]['convergence'].append([float(x) for x in convergence])
                        results[alg_name]['runtimes'].append(float(runtime))
                        print(f" - Cost: {best_cost:.2e}, Time: {runtime:.2f}s")
                    else:
                        print(f" - FAILED")
                        
                except Exception as e:
                    print(f" - ERROR: {e}")
        
        # Generate visualizations
        if show_plots:
            print("\nGenerating visualizations...")
            self.visualizer.plot_convergence_comparison(
                results, problem_type, experiment_id, 
                save=True, show=True, max_iterations=max_iterations)
            
            self.visualizer.plot_individual_convergence(
                results, problem_type, experiment_id, 
                save=True, show=False, max_iterations=max_iterations)
            
            if problem_type == "thomson":
                self.visualizer.plot_thomson_sphere(
                    results, problem_type, experiment_id, save=True, show=True)
        
        # Save results
        if save_results:
            results_file = os.path.join(self.config.RESULTS_DIR, f'{experiment_id}_results.json')
            save_results_to_file(results, problem_type, experiment_id, results_file)
            print(f"✓ Results saved with unique ID: {experiment_id}")
        
        # Print summary
        self._print_summary(results, problem_type, experiment_id)
        
        return results, experiment_id
    
    def _create_metaheuristic_optimizer(self, algorithm: str, manifold, cost_function,
                                      population_size: int, max_iterations: int, seed: int):
        """Create metaheuristic optimizer instance."""
        params = self.config.get_algorithm_params(algorithm)
        params.update({
            'population_size': population_size,
            'max_iterations': max_iterations,
            'seed': seed,
            'verbose': False
        })
        
        if algorithm == 'mPSO':
            return MPSOOptimizer(manifold, cost_function, **params)
        elif algorithm == 'mDE':
            return MDEOptimizer(manifold, cost_function, **params)
        elif algorithm == 'mSMANN':
            return MSMANNOptimizer(manifold, cost_function, **params)
        elif algorithm == 'mDTMA':
            return MDTMAOptimizer(manifold, cost_function, **params)
        elif algorithm == 'RICA':
            return RICAOptimizer(manifold, cost_function, **params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _print_summary(self, results: Dict, problem_type: str, experiment_id: str):
        """Print summary of results."""
        print("\n" + "=" * 70)
        print(f"SUMMARY RESULTS - {problem_type.upper()}")
        print(f"Experiment ID: {experiment_id}")
        print("=" * 70)
        
        metaheuristics = ['mPSO', 'mDE', 'mSMANN', 'mDTMA', 'RICA']
        gradient_based = ['steepest_descent', 'conjugate_gradient', 'trust_regions']
        
        print("\n--- Metaheuristic Algorithms ---")
        for alg_name in metaheuristics:
            if alg_name in results and results[alg_name]['costs']:
                costs = results[alg_name]['costs']
                runtimes = results[alg_name]['runtimes']
                mean_cost = np.mean(costs)
                std_cost = np.std(costs)
                mean_time = np.mean(runtimes)
                print(f"{alg_name:12} | Cost: {mean_cost:8.2e} ± {std_cost:8.2e} | "
                      f"Time: {mean_time:6.2f}s | Runs: {len(costs)}")
        
        print("\n--- Gradient-based Algorithms ---")
        for alg_name in gradient_based:
            if alg_name in results and results[alg_name]['costs']:
                costs = results[alg_name]['costs']
                runtimes = results[alg_name]['runtimes']
                mean_cost = np.mean(costs)
                std_cost = np.std(costs)
                mean_time = np.mean(runtimes)
                print(f"{alg_name:12} | Cost: {mean_cost:8.2e} ± {std_cost:8.2e} | "
                      f"Time: {mean_time:6.2f}s | Runs: {len(costs)}")
        
        print("=" * 70)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Manifold Optimization Comparison')
    parser.add_argument('--problem', type=str, required=True,
                       choices=['dominant', 'sdp', 'svd', 'procrustes', 'thomson', 'stiffness'],
                       help='Type of optimization problem')
    parser.add_argument('--datafile', type=str, required=True,
                       help='Name of .mat data file (without extension)')
    parser.add_argument('--algorithms', type=str, nargs='+',
                       default=['mPSO', 'mDE', 'mSMANN', 'mDTMA', 'RICA'],
                       help='Algorithms to compare')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of runs per algorithm')
    parser.add_argument('--population', type=int, default=50,
                       help='Population size for metaheuristics')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Maximum number of iterations')
    parser.add_argument('--rank', type=int, default=3,
                       help='Rank parameter for relevant problems')
    parser.add_argument('--d_ambient', type=int, default=3,
                       help='Ambient dimension for Thomson problem')
    parser.add_argument('--no_plots', action='store_true',
                       help='Disable plotting')
    
    args = parser.parse_args()
    
    # Create and run comparison
    comparison = ManifoldOptimizationComparison()
    
    results, experiment_id = comparison.run_comparison(
        problem_name=args.datafile,
        problem_type=args.problem,
        algorithms=args.algorithms,
        num_runs=args.runs,
        population_size=args.population,
        max_iterations=args.iterations,
        rank=args.rank,
        d_ambient=args.d_ambient,
        show_plots=not args.no_plots
    )
    
    print(f"\n✓ Experiment completed successfully!")
    print(f"✓ Experiment ID: {experiment_id}")


if __name__ == "__main__":
    main()