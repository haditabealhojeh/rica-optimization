"""
Base class for manifold optimization algorithms.
"""

import numpy as np
from typing import List, Tuple, Any, Optional
import time

class BaseOptimizer:
    """Base class for manifold optimization algorithms."""
    
    def __init__(self, manifold, cost_function, population_size: int = 50, 
                 max_iterations: int = 100, seed: Optional[int] = None, 
                 verbose: bool = True):
        
        self.manifold = manifold
        self.cost_function = cost_function
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        if seed is not None:
            np.random.seed(seed)
        
        # Results tracking
        self.best_solution = None
        self.best_cost = float('inf')
        self.convergence_history = []
    
    def _initialize_population(self):
        """Initialize population on the manifold."""
        population = []
        for _ in range(self.population_size):
            point = self.manifold.random_point()
            population.append(point)
        return population
    
    def _evaluate_population(self, population):
        """Evaluate cost for each individual in population."""
        costs = []
        for individual in population:
            cost = self.cost_function(individual)
            costs.append(cost)
        return costs
    
    def _update_best(self, population, costs):
        """Update best solution found so far."""
        min_cost_idx = np.argmin(costs)
        if costs[min_cost_idx] < self.best_cost:
            self.best_solution = population[min_cost_idx]
            self.best_cost = costs[min_cost_idx]
    
    def _log_convergence(self, iteration: int, cost: float):
        """Log convergence information."""
        if self.verbose and (iteration % 50 == 0 or iteration == self.max_iterations - 1):
            print(f"Iteration {iteration:4d}: Best Cost = {cost:.6e}")
    
    def optimize(self) -> Tuple[Any, float, List[float]]:
        """Run optimization - MUST BE IMPLEMENTED BY SUBCLASSES."""
        raise NotImplementedError("Subclasses must implement optimize() method")
