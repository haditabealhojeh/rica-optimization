"""
Wrapper for Pymanopt's built-in optimization algorithms.
"""

import time
from typing import List, Tuple, Any
from .base_optimizer import BaseOptimizer


class PymanoptWrapper(BaseOptimizer):
    """Simple wrapper for Pymanopt optimization algorithms."""
    
    def __init__(self, manifold, cost_function, algorithm='trust_regions', 
                 max_iterations=100, seed=None, verbose=True):
        
        super().__init__(manifold, cost_function, population_size=1, 
                        max_iterations=max_iterations, seed=seed, verbose=verbose)
        self.algorithm = algorithm
        
    def optimize(self) -> Tuple[Any, float, List[float]]:
        """Run Pymanopt optimization with compatibility for different versions."""
        
        try:
            # Try new pymanopt API first
            from pymanopt import Problem
            from pymanopt.optimizers import SteepestDescent, TrustRegions, ConjugateGradient
            
            # Import backend decorator
            try:
                from pymanopt import autodiff
                backend = autodiff.backends.get_default_backend()
            except ImportError:
                # Fallback for older versions
                backend = None
            
            # Create the cost function with proper backend decoration
            if backend is not None:
                # For newer pymanopt versions, we need to decorate the cost function
                @backend.compute_cost
                def decorated_cost(point):
                    return self.cost_function(point)
                cost_func = decorated_cost
            else:
                # For older versions, use the cost function directly
                cost_func = self.cost_function
            
            # Create problem
            problem = Problem(manifold=self.manifold, cost=cost_func)
            
            # Select optimizer
            if self.algorithm == 'steepest_descent':
                optimizer = SteepestDescent(max_iterations=self.max_iterations, verbosity=1 if self.verbose else 0)
            elif self.algorithm == 'conjugate_gradient':
                optimizer = ConjugateGradient(max_iterations=self.max_iterations, verbosity=1 if self.verbose else 0)
            elif self.algorithm == 'trust_regions':
                optimizer = TrustRegions(max_iterations=self.max_iterations, verbosity=1 if self.verbose else 0)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
                
        except ImportError:
            # Fallback to old pymanopt API
            try:
                from pymanopt import Problem
                from pymanopt.solvers import SteepestDescent, TrustRegions, ConjugateGradient
                
                # Create problem
                problem = Problem(manifold=self.manifold, cost=self.cost_function)
                
                # Select solver
                if self.algorithm == 'steepest_descent':
                    optimizer = SteepestDescent(maxiter=self.max_iterations, logverbosity=1 if self.verbose else 0)
                elif self.algorithm == 'conjugate_gradient':
                    optimizer = ConjugateGradient(maxiter=self.max_iterations, logverbosity=1 if self.verbose else 0)
                elif self.algorithm == 'trust_regions':
                    optimizer = TrustRegions(maxiter=self.max_iterations, logverbosity=1 if self.verbose else 0)
                else:
                    raise ValueError(f"Unknown algorithm: {self.algorithm}")
                    
            except ImportError:
                raise ImportError("Pymanopt is not installed")
        
        # Run optimization
        start_time = time.time()
        try:
            if self.verbose:
                print(f"Running {self.algorithm}...")
            
            # Run optimization (method name varies by version)
            if hasattr(optimizer, 'run'):
                result = optimizer.run(problem)
            else:
                result = optimizer.solve(problem)
                
            runtime = time.time() - start_time
            
            self.best_solution = result
            self.best_cost = self.cost_function(result)
            self.convergence_history = [self.best_cost]  # Simple convergence
            
            if self.verbose:
                print(f"{self.algorithm} completed in {runtime:.2f}s, cost: {self.best_cost:.6e}")
                
        except Exception as e:
            if self.verbose:
                print(f"Pymanopt {self.algorithm} failed: {e}")
            # Fallback to random point
            self.best_solution = self.manifold.random_point()
            self.best_cost = self.cost_function(self.best_solution)
            self.convergence_history = [self.best_cost]
        
        return self.best_solution, self.best_cost, self.convergence_history


def create_pymanopt_optimizer(manifold, cost_function, algorithm, max_iterations=100, verbose=True):
    """Factory function to create Pymanopt optimizer."""
    return PymanoptWrapper(manifold, cost_function, algorithm, max_iterations, verbose=verbose)