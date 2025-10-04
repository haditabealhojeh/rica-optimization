#!/usr/bin/env python3
"""
Run complete manifold optimization comparison.
"""

import os
import argparse

# Add the current directory to Python path to fix import issues
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import ManifoldOptimizationComparison
from config import Config

def run_single_problem(problem_name, problem_type, algorithms=None):
    """Run comparison for a single problem."""
    
    if algorithms is None:
        algorithms = Config.get_metaheuristic_algorithms()
    
    print(f"\n{'='*60}")
    print(f"RUNNING: {problem_type.upper()} - {problem_name}")
    print(f"{'='*60}")
    
    comparison = ManifoldOptimizationComparison()
    
    # This will now return both results and experiment_id, but we can ignore experiment_id
    results, experiment_id = comparison.run_comparison(
        problem_name=problem_name,
        problem_type=problem_type,
        algorithms=algorithms,
        num_runs=Config.DEFAULT_NUM_RUNS,
        population_size=Config.DEFAULT_POPULATION_SIZE,
        max_iterations=Config.DEFAULT_ITERATIONS,
        show_plots=True
    )
    
    print(f"✓ Experiment completed with ID: {experiment_id}")
    return results

def main():
    parser = argparse.ArgumentParser(description='Run manifold optimization comparisons')
    parser.add_argument('--problem', type=str, required=True,
                       choices=['thomson', 'dominant', 'sdp', 'svd', 'procrustes', 'stiffness'],
                       help='Problem type to run')
    parser.add_argument('--datafile', type=str, required=True,
                       help='Name of .mat data file (without extension)')
    parser.add_argument('--algorithms', type=str, nargs='+',
                       help='Specific algorithms to run (default: all)')
    parser.add_argument('--runs', type=int, default=Config.DEFAULT_NUM_RUNS,
                       help='Number of runs per algorithm')
    
    args = parser.parse_args()
    
    # Run the comparison - same as before!
    run_single_problem(args.datafile, args.problem, args.algorithms)

if __name__ == "__main__":
    main()