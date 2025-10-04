#!/usr/bin/env python3
"""
Run all paper experiments automatically.
"""

import os
from main import ManifoldOptimizationComparison
from config import Config

# Paper experiment configurations
PAPER_EXPERIMENTS = [
    # Thomson problems (Table 3)
    {'name': 'thomson-n50-d3-1', 'type': 'thomson', 'description': 'Thomson n=50'},
    {'name': 'thomson-n75-d3-1', 'type': 'thomson', 'description': 'Thomson n=75'},
    {'name': 'thomson-n100-d3-1', 'type': 'thomson', 'description': 'Thomson n=100'},
    {'name': 'thomson-n125-d3-1', 'type': 'thomson', 'description': 'Thomson n=125'},
    {'name': 'thomson-n150-d3-1', 'type': 'thomson', 'description': 'Thomson n=150'},
    
    # Dominant invariant subspace (Table 5)
    {'name': 'dominant-n128-p3-1', 'type': 'dominant', 'description': 'Dominant p=3'},
    
    # SDP problem (Table 4)
    {'name': 'sdp-n100-p?', 'type': 'sdp', 'description': 'SDP n=100'},
    
    # Truncated SVD (Table 6)
    {'name': 'svd-n42-m60-p5-1', 'type': 'svd', 'description': 'SVD p=5'},
    
    # Procrustes problem (Table 7)
    {'name': 'Procrustes-n3m10N50-1', 'type': 'procrustes', 'description': 'Procrustes'},
    
    # Stiffness learning
    {'name': 'eign-120', 'type': 'stiffness', 'description': 'Stiffness Learning'}
]

def run_all_experiments():
    """Run all paper experiments."""
    
    comparison = ManifoldOptimizationComparison()
    
    for exp in PAPER_EXPERIMENTS:
        print(f"\n{'#'*80}")
        print(f"RUNNING: {exp['description']}")
        print(f"Data file: {exp['name']}.mat")
        print(f"{'#'*80}")
        
        try:
            results = comparison.run_comparison(
                problem_name=exp['name'],
                problem_type=exp['type'],
                algorithms=Config.get_paper_algorithms(),
                num_runs=Config.DEFAULT_NUM_RUNS,
                population_size=Config.DEFAULT_POPULATION_SIZE,
                max_iterations=Config.DEFAULT_ITERATIONS,
                show_plots=True
            )
            
            if results:
                print(f"✓ Completed {exp['description']}")
            else:
                print(f"✗ Failed {exp['description']}")
                
        except Exception as e:
            print(f"✗ Error in {exp['description']}: {e}")
            continue

if __name__ == "__main__":
    run_all_experiments()