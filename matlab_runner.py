"""
MATLAB runner for original manifold optimization algorithms.
"""

import os
import subprocess
import tempfile
import numpy as np
import scipy.io
from typing import Dict, List, Any, Tuple
import json


class MATLABRunner:
    """Run original MATLAB manifold optimization algorithms."""
    
    def __init__(self, matlab_path: str = None, manopt_path: str = None):
        self.matlab_path = matlab_path or self._find_matlab()
        self.manopt_path = manopt_path or self._find_manopt()
        self.temp_dir = tempfile.mkdtemp()
        
    def _find_matlab(self):
        """Find MATLAB installation path."""
        # Common MATLAB installation paths
        possible_paths = [
            "/usr/local/MATLAB/R*/bin/matlab",  # Linux
            "/Applications/MATLAB_*.app/bin/matlab",  # macOS
            "C:\\Program Files\\MATLAB\\R*\\bin\\matlab.exe",  # Windows
        ]
        
        for path_pattern in possible_paths:
            import glob
            matches = glob.glob(path_pattern)
            if matches:
                return matches[0]
        
        raise Exception("MATLAB not found. Please specify matlab_path.")
    
    def _find_manopt(self):
        """Find Manopt installation path."""
        # Check common locations
        possible_paths = [
            "./manopt",
            "../manopt", 
            os.path.expanduser("~/manopt"),
            "/usr/local/manopt"
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, 'manopt.m')):
                return path
        
        # Clone from GitHub if not found
        print("Manopt not found. Cloning from GitHub...")
        subprocess.run([
            'git', 'clone', 'https://github.com/NicolasBoumal/manopt.git', 
            os.path.join(self.temp_dir, 'manopt')
        ])
        return os.path.join(self.temp_dir, 'manopt')
    
    def run_mdtma(self, problem_data: Dict, max_iterations: int = 800, 
                 population_size: int = 100, w0: float = 0.1) -> Dict:
        """Run original mDTMA MATLAB implementation."""
        
        # Save problem data to .mat file
        data_file = os.path.join(self.temp_dir, 'problem_data.mat')
        scipy.io.savemat(data_file, problem_data)
        
        # Create MATLAB script
        matlab_script = f"""
        try
            % Add Manopt to path
            addpath('{self.manopt_path}');
            
            % Load problem data
            load('{data_file}');
            
            % Setup problem structure for mDTMA
            problem = struct();
            problem.M = manifold;
            problem.cost = @(x) cost_function(x);
            
            % Run mDTMA with paper parameters
            [xbest, fbest, convergence] = mDTMA(problem, {population_size}, {max_iterations}, {w0});
            
            % Save results
            results = struct();
            results.xbest = xbest;
            results.fbest = fbest;
            results.convergence = convergence;
            
            save('{os.path.join(self.temp_dir, 'mdtma_results.mat')}', 'results');
            exit(0);
        catch e
            fprintf('MATLAB Error: %s\\n', e.message);
            exit(1);
        end
        """
        
        script_file = os.path.join(self.temp_dir, 'run_mdtma.m')
        with open(script_file, 'w') as f:
            f.write(matlab_script)
        
        # Run MATLAB
        result = subprocess.run([
            self.matlab_path,
            '-batch',
            f"run('{script_file}')"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"MATLAB execution failed: {result.stderr}")
        
        # Load results
        results_file = os.path.join(self.temp_dir, 'mdtma_results.mat')
        if os.path.exists(results_file):
            mat_results = scipy.io.loadmat(results_file)
            return {
                'best_solution': mat_results['results']['xbest'][0,0],
                'best_cost': float(mat_results['results']['fbest'][0,0]),
                'convergence': mat_results['results']['convergence'][0].flatten().tolist()
            }
        else:
            raise Exception("MATLAB did not produce results file")
    
    def run_mPSO(self, problem_data: Dict, max_iterations: int = 800,
                population_size: int = 100) -> Dict:
        """Run original mPSO MATLAB implementation."""
        
        data_file = os.path.join(self.temp_dir, 'problem_data.mat')
        scipy.io.savemat(data_file, problem_data)
        
        matlab_script = f"""
        try
            addpath('{self.manopt_path}');
            load('{data_file}');
            
            problem = struct();
            problem.M = manifold;
            problem.cost = @(x) cost_function(x);
            
            [xbest, fbest, convergence] = manopt_solvers_mPSO(problem, {population_size}, {max_iterations});
            
            results = struct();
            results.xbest = xbest;
            results.fbest = fbest;
            results.convergence = convergence;
            
            save('{os.path.join(self.temp_dir, 'mpso_results.mat')}', 'results');
            exit(0);
        catch e
            fprintf('MATLAB Error: %s\\n', e.message);
            exit(1);
        end
        """
        
        script_file = os.path.join(self.temp_dir, 'run_mpso.m')
        with open(script_file, 'w') as f:
            f.write(matlab_script)
        
        result = subprocess.run([
            self.matlab_path,
            '-batch',
            f"run('{script_file}')"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"MATLAB mPSO failed: {result.stderr}")
        
        results_file = os.path.join(self.temp_dir, 'mpso_results.mat')
        if os.path.exists(results_file):
            mat_results = scipy.io.loadmat(results_file)
            return {
                'best_solution': mat_results['results']['xbest'][0,0],
                'best_cost': float(mat_results['results']['fbest'][0,0]),
                'convergence': mat_results['results']['convergence'][0].flatten().tolist()
            }
        else:
            raise Exception("MATLAB mPSO did not produce results file")
    
    def run_comparison(self, problem_name: str, problem_type: str,
                      algorithms: List[str] = None, num_runs: int = 2,
                      max_iterations: int = 800, population_size: int = 100) -> Dict:
        """Run comparison using original MATLAB implementations."""
        
        if algorithms is None:
            algorithms = ['mDTMA', 'mPSO', 'mDE', 'mSMANN']
        
        results = {}
        
        for alg_name in algorithms:
            print(f"\nRunning {alg_name} with MATLAB...")
            alg_results = {'costs': [], 'convergence': [], 'runtimes': []}
            
            for run in range(num_runs):
                print(f"  Run {run + 1}/{num_runs}", end="")
                
                try:
                    # Load problem data (you'll need to adapt this part)
                    problem_data = self._load_problem_data(problem_name, problem_type)
                    
                    start_time = time.time()
                    
                    if alg_name == 'mDTMA':
                        result = self.run_mdtma(problem_data, max_iterations, population_size)
                    elif alg_name == 'mPSO':
                        result = self.run_mPSO(problem_data, max_iterations, population_size)
                    # Add other algorithms...
                    
                    runtime = time.time() - start_time
                    
                    alg_results['costs'].append(result['best_cost'])
                    alg_results['convergence'].append(result['convergence'])
                    alg_results['runtimes'].append(runtime)
                    
                    print(f" - Cost: {result['best_cost']:.2e}, Time: {runtime:.2f}s")
                    
                except Exception as e:
                    print(f" - ERROR: {e}")
            
            results[alg_name] = alg_results
        
        return results
    
    def _load_problem_data(self, problem_name: str, problem_type: str) -> Dict:
        """Load problem data in format suitable for MATLAB."""
        # This needs to be adapted to your specific problem format
        # You'll need to convert your Python manifolds to MATLAB-compatible format
        
        if problem_type == "dominant":
            # Example for dominant invariant subspace
            return {
                'manifold': 'grassmannfactory(128, 3)',
                'cost_function': '@(X) -0.5 * trace(X'' * A * X)'
            }
        # Add other problem types...
        
        raise ValueError(f"Unsupported problem type: {problem_type}")
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)